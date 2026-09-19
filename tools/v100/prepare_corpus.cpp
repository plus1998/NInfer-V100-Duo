// Build a real-text token corpus with the exact tokenizer embedded in the artifact.
// This reads frontend resources only and never allocates GPU memory.
#include "artifact/reader.h"
#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace frontend = ninfer::targets::qwen3_6::frontend_internal;

struct Options {
    std::size_t chat_tokens = 0;
    std::size_t output_tokens = 1024;
    std::vector<std::filesystem::path> files;
};

std::size_t positive_count(std::string_view raw, const char* flag) {
    std::size_t value = 0;
    const auto [end, error] = std::from_chars(raw.data(), raw.data() + raw.size(), value);
    if (error != std::errc{} || end != raw.data() + raw.size() || value == 0) {
        throw std::invalid_argument(std::string(flag) + " requires a positive token count");
    }
    return value;
}

Options parse_options(int argc, char** argv) {
    Options options;
    bool output_override = false;
    for (int i = 3; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--code-chat" || argument == "--output-tokens") {
            if (++i == argc) { throw std::invalid_argument(std::string(argument) + " needs a value"); }
            const auto count = positive_count(argv[i], argument.data());
            if (argument == "--code-chat") {
                options.chat_tokens = count;
            } else {
                options.output_tokens = count;
                output_override = true;
            }
        } else if (argument.starts_with("--")) {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        } else {
            options.files.emplace_back(argv[i]);
        }
    }
    if (options.files.empty()) { throw std::invalid_argument("list at least one source file"); }
    if (output_override && options.chat_tokens == 0) {
        throw std::invalid_argument("--output-tokens requires --code-chat");
    }
    return options;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot read " + path.string()); }
    std::string text((std::istreambuf_iterator<char>(input)), {});
    if (input.bad()) { throw std::runtime_error("read failed: " + path.string()); }
    return text;
}

std::vector<int> plain_corpus(const frontend::Tokenizer& tokenizer, const Options& options) {
    std::vector<int> result;
    for (const auto& path : options.files) {
        const auto ids = tokenizer.encode(read_file(path) + "\n\n", {.parse_added_tokens = false});
        result.insert(result.end(), ids.begin(), ids.end());
    }
    return result;
}

std::vector<int> code_chat(const frontend::Tokenizer& tokenizer,
                           const frontend::CompiledChatTemplate& chat_template,
                           const Options& options) {
    constexpr std::string_view marker = "__NINFER_V100_CODE_CONTEXT_BODY_7B46D921__";
    frontend::ChatMessage system;
    system.role = ninfer::ChatRole::System;
    system.parts.push_back(frontend::ChatPart::text_part(
        "You are a C++ engineer. Treat the supplied repository excerpts as reference code, "
        "then solve the implementation task at the end. Provide code and concise reasoning."));
    frontend::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(frontend::ChatPart::text_part(
        "Repository reference excerpts follow in their listed order. The final excerpt may "
        "stop mid-file. Use them as engineering context; do not reproduce the excerpts.\n\n```cpp\n" +
        std::string(marker) +
        "\n```\n\nImplementation task:\n"
        "Implement a self-contained C++20 bounded blocking queue for move-only tasks, using "
        "std::mutex and std::condition_variable. Provide push, pop and close. A producer blocks "
        "when full and a consumer blocks when empty. close(Drain) rejects further pushes but lets "
        "consumers finish queued work; close(Cancel) rejects pushes and returns pending tasks to "
        "the caller for cancellation. Popped tasks remain owned by their consumers. Closing must "
        "wake all blocked producers and consumers, be idempotent, and prevent any accepted task "
        "from being lost or delivered twice. Specify how a rejected push preserves ownership "
        "and how conflicting close calls are resolved. Reject zero capacity.\n"
        "Show the complete class, a short two-worker usage example, and explain the synchronization "
        "invariants and thread-joining requirement. Do not execute tasks while holding the queue "
        "lock. Include the necessary standard headers. This is a fixed-budget coding response: "
        "use at most " + std::to_string(options.output_tokens) +
        " output tokens, spending most of them on the implementation and example."));
    const auto rendered = chat_template.render({system, user}, {.enable_thinking = false});
    const auto at = rendered.text.find(marker);
    if (at == std::string::npos ||
        rendered.text.find(marker, at + marker.size()) != std::string::npos) {
        throw std::runtime_error("chat template did not preserve the unique body marker");
    }

    // Template boundaries are encoded with their special tokens. Repository text
    // is encoded as ordinary text, so literal token spellings inside source files
    // cannot create chat turns. The three token spans are the canonical prompt;
    // both engines must consume the saved IDs directly, without re-templating or
    // decoding and retokenizing them (which could change BPE boundary merges).
    auto result = tokenizer.encode(std::string_view(rendered.text).substr(0, at));
    const auto suffix = tokenizer.encode(
        std::string_view(rendered.text).substr(at + marker.size()));
    if (result.size() + suffix.size() >= options.chat_tokens) {
        throw std::invalid_argument("--code-chat budget is too small for the full task and template");
    }
    const std::size_t prefix_tokens = result.size();
    const std::size_t body_budget = options.chat_tokens - prefix_tokens - suffix.size();
    std::size_t body_tokens = 0;
    std::set<std::filesystem::path> visited;
    for (const auto& path : options.files) {
        const auto canonical = std::filesystem::canonical(path);
        if (!visited.insert(canonical).second) {
            throw std::invalid_argument("--code-chat source was listed twice: " + path.string());
        }
        const std::string extension = path.extension().string();
        if (extension != ".cpp" && extension != ".cc" && extension != ".cxx" &&
            extension != ".c" && extension != ".h" && extension != ".hpp" &&
            extension != ".cuh" && extension != ".cu") {
            throw std::invalid_argument("--code-chat expects C++/CUDA source: " + path.string());
        }
        if (body_tokens == body_budget) { continue; }
        const std::string text = "\n// Repository file: " + path.generic_string() + "\n" +
                                 read_file(path) + "\n";
        const auto ids = tokenizer.encode(text, {.parse_added_tokens = false});
        const auto used = std::min(ids.size(), body_budget - body_tokens);
        result.insert(result.end(), ids.begin(), ids.begin() + used);
        body_tokens += used;
        std::cout << "source=" << path.generic_string() << " tokens_used=" << used
                  << " tokens_available=" << ids.size() << '\n';
    }
    if (body_tokens != body_budget) {
        throw std::invalid_argument("source files provide only " + std::to_string(body_tokens) +
                                    " code tokens; need " + std::to_string(body_budget) +
                                    ". List more distinct source files; no text is repeated.");
    }
    result.insert(result.end(), suffix.begin(), suffix.end());
    if (result.size() != options.chat_tokens) {
        throw std::logic_error("code-chat prompt length does not match its exact budget");
    }
    std::cout << "mode=code-chat prompt_tokens=" << result.size()
              << " prefix_tokens=" << prefix_tokens << " code_tokens=" << body_tokens
              << " suffix_tokens=" << suffix.size()
              << " requested_output_tokens=" << options.output_tokens
              << " thinking=false task=bounded-blocking-task-queue\n";
    return result;
}

void write_ids(const char* path, std::span<const int> ids) {
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("cannot create corpus"); }
    for (const int token : ids) { output << token << '\n'; }
    output.close();
    if (!output) { throw std::runtime_error("corpus write failed"); }
    std::cout << "wrote " << ids.size() << " tokens to " << path << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: ninfer_v100_corpus ARTIFACT OUTPUT.ids TEXT_FILE...\n"
                     "       ninfer_v100_corpus ARTIFACT OUTPUT.ids --code-chat 85000 "
                     "[--output-tokens 1024] CPP_OR_CUDA_FILE...\n"
                     "Code-chat preserves the complete task and template at an exact prompt length.\n"
                     "--output-tokens states the fixed response budget in the task; configure both\n"
                     "inference engines with that same generation budget and pass these raw IDs.\n";
        return 2;
    }
    try {
        const Options options = parse_options(argc, argv);
        ninfer::artifact::Reader artifact(argv[1]);
        const auto resource = [&](const char* name) {
            const auto bytes = artifact.payload(name);
            return std::string_view(reinterpret_cast<const char*>(bytes.data.data()), bytes.data.size());
        };
        const frontend::Tokenizer tokenizer({
            resource("frontend/tokenizer.json"), resource("frontend/tokenizer_config.json"),
            resource("frontend/generation_config.json")});
        if (options.chat_tokens != 0) {
            const auto chat_template = frontend::CompiledChatTemplate::resolve(
                resource("frontend/chat_template.jinja"));
            write_ids(argv[2], code_chat(tokenizer, chat_template, options));
        } else {
            write_ids(argv[2], plain_corpus(tokenizer, options));
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
