// Opt-in real-weight test for `--rope yarn` past the native 262,144-position window.
//
// With `rope_mode == Yarn` the engine must ADMIT a `max_context` beyond the artifact's registered
// native capacity, prefill a prompt that genuinely crosses position 262,144, and then answer a
// question whose evidence sits past
// that boundary. A needle is used rather than a "did it produce text" check because garbage past
// the native window is exactly the failure this test exists to rule out, and fluent-but-wrong text
// would pass a liveness check: the model has to READ a fact stored at a position the native rope
// cannot address correctly.
//
// Both tensor-parallel widths run. `--rope yarn` is deliberately not coupled to `--tp`, and the
// single-GPU leg is what proves that -- it is the same YaRN table and the same extended positions
// with no collectives involved, so a failure there separates "YaRN is wrong" from "the split is
// wrong". The tp1 leg uses a smaller ceiling because one device holds all four KV heads (about
// 33 KiB/token at INT8) on top of the whole ~20 GiB model.
//
// Skipped (exit 77) unless NINFER_QWEN3_8_27B_WEIGHTS names an artifact; the tp2 leg additionally
// needs two visible CUDA devices and is skipped alone if only one is present.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

// The artifact's registered native position capacity: the boundary the needle has to sit PAST for
// these legs to mean anything, and the ceiling `--rope native` still enforces.
constexpr std::uint32_t kNativeContext = 262144;
constexpr std::uint32_t kOutputTokens = 24;
constexpr const char* kSecret         = "74812";
// Sentences of filler AFTER the secret, so the needle is not simply the last thing in the window.
constexpr std::size_t kTrailingSentences = 256;
// Headroom left below `max_context` for the generated tokens and for the sizing estimate's own
// error (the estimate is a linear fit through two tokenizer measurements, so it is close but not
// exact).
constexpr std::uint32_t kContextMargin = 4096;

// Distinct filler with no numbers in it, so the only five-digit run anywhere in the prompt is the
// secret. Repetition is fine -- what is under test is position arithmetic, not comprehension --
// but the sentences vary so the tokenizer does not collapse onto one repeated id.
const char* const kFillerSentences[] = {
    "The archivist walked the lower stacks, noting which shelves had settled since the survey.\n",
    "Rain moved across the valley in slow bands, and the reading room stayed quiet all morning.\n",
    "A cataloguer described the binding as sound, the endpapers foxed, the spine lightly rubbed.\n",
    "Visitors were asked to leave bags at the desk and to handle the folios with clean hands.\n",
    "The building's older wing kept its own weather, cooler than the corridor that joined them.\n",
};

std::string filler_block(std::size_t sentences, std::size_t seed) {
    std::string text;
    text.reserve(sentences * 96);
    for (std::size_t index = 0; index < sentences; ++index) {
        text += kFillerSentences[(index + seed) % (sizeof(kFillerSentences) / sizeof(char*))];
    }
    return text;
}

ninfer::PromptInput chat(std::string text) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    // Thinking off: the answer is a five-digit lookup, and a reasoning trace on a 270k-token
    // prompt would multiply the decode time for nothing this test checks.
    input.options.enable_thinking = false;
    return input;
}

// Everything up to and including the secret. Its token count IS the needle's position, which the
// leg asserts is past the native window rather than assuming it.
std::string needle_prefix(std::size_t leading_sentences) {
    std::string text = "Read the following archive log, then answer the question at the end.\n\n";
    text += filler_block(leading_sentences, 0);
    text += "\nThe secret access code for the Meridian vault is ";
    text += kSecret;
    text += ".\n";
    return text;
}

// The full prompt: the prefix above, a little more filler so the secret is not the last thing the
// model saw, then the question.
std::string needle_prompt(std::size_t leading_sentences) {
    std::string text = needle_prefix(leading_sentences);
    text += "\n";
    text += filler_block(kTrailingSentences, 3);
    text += "\nQuestion: what is the secret access code for the Meridian vault? Reply with the "
            "digits only.\n";
    return text;
}

// Chooses the leading filler so the WHOLE prompt lands just under `target_tokens`, by measuring
// this exact filler through the real tokenizer rather than assuming a characters-per-token rate.
// Sizing to the ceiling (instead of to the needle position) is what keeps the prompt admissible:
// `max_context` is fixed when the engine is built, and the engine is what owns the tokenizer.
std::size_t size_leading_filler(const ninfer::Engine& engine, std::uint32_t target_tokens) {
    const std::uint32_t small = engine.count_tokens(chat(needle_prompt(64)));
    const std::uint32_t large = engine.count_tokens(chat(needle_prompt(1088)));
    if (large <= small || target_tokens <= small) { return 0; }
    const double per_sentence = static_cast<double>(large - small) / 1024.0;
    const double needed =
        64.0 + (static_cast<double>(target_tokens) - static_cast<double>(small)) / per_sentence;
    return needed <= 64.0 ? 0 : static_cast<std::size_t>(needed);
}

ninfer::EngineOptions yarn_options(const char* artifact, int tp, std::uint32_t max_context,
                                   std::uint32_t mtp_draft_tokens = 0) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.rope_mode     = ninfer::RopeMode::Yarn;
    options.yarn_factor   = 4.0;
    options.yarn_origin   = kNativeContext;
    options.max_context   = max_context;
    options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
    options.prefill_chunk = 1024;
    options.kv_cache      = ninfer::KvCacheStorage::Int8Group64;
    options.tp            = tp;
    if (mtp_draft_tokens != 0) {
        options.speculative.backend     = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = mtp_draft_tokens;
    }
    if (tp == 2) {
        options.devices = {0, 1};
    } else {
        // The single-GPU leg runs on the LAST visible device: on a two-GPU host device 0 usually
        // carries the display, and this leg wants every byte it can get.
        int devices = 1;
        (void)cudaGetDeviceCount(&devices);
        options.device  = devices > 1 ? devices - 1 : 0;
        options.devices = {options.device};
    }
    return options;
}

ninfer::RequestOptions greedy_options() {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = kOutputTokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    return options;
}

// "Not garbage": the reply is non-empty and dominated by ordinary printable text. A rope that
// diverges past the native window typically produces high-entropy byte soup or an immediate stop,
// and both are caught here even when the needle assertion is not reached.
bool looks_like_text(const std::string& reply) {
    if (reply.empty()) { return false; }
    std::size_t printable = 0;
    for (const char character : reply) {
        const auto value = static_cast<unsigned char>(character);
        if (value == '\n' || value == '\t' || (value >= 0x20 && value < 0x7f)) { ++printable; }
    }
    return printable * 10 >= reply.size() * 9;
}

int run_leg(const char* artifact, int tp, std::uint32_t max_context,
            std::uint32_t mtp_draft_tokens = 0) {
    const std::string label = "yarn tp" + std::to_string(tp) +
                              (mtp_draft_tokens != 0
                                   ? " mtp" + std::to_string(mtp_draft_tokens)
                                   : std::string());
    ninfer::Engine engine(yarn_options(artifact, tp, max_context, mtp_draft_tokens));

    const ninfer::LoadSummary load = engine.load_summary();
    int failures                   = 0;
    if (load.rope_mode != ninfer::RopeMode::Yarn) {
        std::cerr << label << ": load summary does not report yarn\n";
        ++failures;
    }
    if (load.effective_max_context != 4U * kNativeContext) {
        std::cerr << label << ": effective max context is " << load.effective_max_context
                  << ", expected " << 4U * kNativeContext << '\n';
        ++failures;
    }
    // yarn_get_mscale(4.0) = 0.1*ln(4)+1. A native (1.0) mscale here would mean the table was
    // built but the multiplier was dropped somewhere between the plan and the descriptor.
    if (load.yarn_mscale < 1.13 || load.yarn_mscale > 1.15) {
        std::cerr << label << ": yarn mscale is " << load.yarn_mscale << ", expected ~1.1386\n";
        ++failures;
    }
    std::cout << label << ": rope=yarn factor=" << load.yarn_factor
              << " origin=" << load.yarn_origin << " effective_max_context="
              << load.effective_max_context << " mscale=" << load.yarn_mscale << std::endl;

    const std::size_t leading = size_leading_filler(engine, max_context - kContextMargin);
    if (leading == 0) {
        std::cerr << label << ": could not size the filler\n";
        return failures + 1;
    }
    // The needle's own position, measured rather than assumed: this is the number that has to be
    // past 262,144 for the leg to mean anything.
    const std::uint32_t needle_position = engine.count_tokens(chat(needle_prefix(leading)));
    ninfer::PreparedPrompt prompt       = engine.prepare(chat(needle_prompt(leading)));
    const std::uint32_t tokens          = prompt.summary().prompt_tokens;
    std::cout << label << ": prompt is " << tokens << " tokens, needle at position "
              << needle_position << std::endl;
    if (needle_position <= kNativeContext) {
        std::cerr << label << ": the needle sits inside the native 262144-position window ("
                  << needle_position << "); this leg would prove nothing\n";
        return failures + 1;
    }
    if (tokens > max_context) {
        std::cerr << label << ": prompt (" << tokens << ") exceeds this leg's max_context ("
                  << max_context << ")\n";
        return failures + 1;
    }

    const ninfer::GenerationResult result = engine.generate(std::move(prompt), greedy_options());
    const std::string reply               = result.content;
    std::cout << label << ": prefill " << result.timings.prefill_seconds << " s, decode "
              << result.timings.decode_seconds << " s, reply \"" << reply << "\""
              << std::endl;

    if (result.prompt.prompt_tokens <= kNativeContext) {
        std::cerr << label << ": generation reported a prompt inside the native window\n";
        ++failures;
    }
    if (result.generated_token_ids.empty()) {
        std::cerr << label << ": no tokens were generated past the native window\n";
        ++failures;
    }
    if (!looks_like_text(reply)) {
        std::cerr << label << ": reply past the native window is not ordinary text\n";
        ++failures;
    }
    if (mtp_draft_tokens != 0 && result.speculative.rounds == 0) {
        std::cerr << label << ": MTP ran no speculative rounds past the native window\n";
        ++failures;
    }
    if (mtp_draft_tokens != 0 && result.speculative.accepted_tokens == 0) {
        std::cerr << label << ": MTP accepted no drafted token past the native window, so its "
                     "rope path was never exercised there\n";
        ++failures;
    }
    if (reply.find(kSecret) == std::string::npos) {
        std::cerr << label << ": the needle stored past position " << kNativeContext
                  << " was not retrieved (expected \"" << kSecret << "\" in the reply)\n";
        ++failures;
    }
    return failures;
}

// Does the YaRN table actually reach the kernels?
//
// This is NOT implied by the long-context legs above, and measuring it was the point of adding it:
// with the per-device descriptors deliberately nulled (native frequencies evaluated at extended
// positions) the tp1 needle leg STILL answered correctly at position 270,876. A ~3.4% overrun of
// the trained window is evidently within what the checkpoint tolerates by extrapolation, so a
// needle at ~271k is evidence that the extended-position PATH works end to end -- positions, page
// tables, KV plan, attention envelope -- and not evidence that YaRN is applied.
//
// So the wiring gets its own, decisive check, at a short context where it is cheap: the same
// prompt through a native engine and a yarn engine must not produce the same logits. The two
// differ only in the rotary frequency table and its mscale, and a factor-1 table reproduces the
// native path bit-for-bit (tests/ops/test_rope_yarn.cpp's first leg bit-compares override-mode
// output against ordinary-mode output over six geometries) -- so identical logits here would mean
// the descriptor never reached the rope launcher.
int check_yarn_reaches_the_kernels(const char* artifact) {
    const auto capture = [&](ninfer::RopeMode mode) {
        ninfer::EngineOptions options = yarn_options(artifact, 1, 4096);
        options.rope_mode             = mode;
        ninfer::Engine engine(std::move(options));
        engine.debug_enable_logit_capture(true);
        ninfer::RequestOptions request              = greedy_options();
        request.execution.requested_output_tokens   = 1;
        (void)engine.generate(engine.prepare(chat("Name three primary colours.")),
                              std::move(request));
        return engine.debug_last_round_logits_bf16();
    };
    const std::vector<std::uint16_t> native = capture(ninfer::RopeMode::Native);
    const std::vector<std::uint16_t> yarn   = capture(ninfer::RopeMode::Yarn);
    if (native.empty() || native.size() != yarn.size()) {
        std::cerr << "logit capture did not produce comparable vectors\n";
        return 1;
    }
    std::size_t differing = 0;
    for (std::size_t index = 0; index < native.size(); ++index) {
        differing += static_cast<std::size_t>(native[index] != yarn[index]);
    }
    std::cout << "yarn wiring: " << differing << " of " << native.size()
              << " prefill logits differ between native and yarn rope" << std::endl;
    if (differing == 0) {
        std::cerr << "yarn rope produced logits identical to native: the per-device frequency "
                     "descriptor is not reaching the rope kernels\n";
        return 1;
    }
    return 0;
}

// The native default must still refuse what it always refused, on the same binary that now accepts
// the yarn form -- otherwise "yarn raises the ceiling" could be "the ceiling is gone".
int check_native_ceiling_still_binds(const char* artifact) {
    ninfer::EngineOptions options = yarn_options(artifact, 1, kNativeContext + 1);
    options.rope_mode             = ninfer::RopeMode::Native;
    try {
        ninfer::Engine engine(std::move(options));
    } catch (const std::invalid_argument&) {
        return 0;
    }
    std::cerr << "native mode accepted a max_context beyond the registered native capacity\n";
    return 1;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_8_27B_WEIGHTS is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
        std::cout << "skip: no CUDA device\n";
        return 77;
    }
    int failures = check_native_ceiling_still_binds(artifact);
    failures += check_yarn_reaches_the_kernels(artifact);
    if (devices >= 2) {
        failures += run_leg(artifact, 2, 300000);
        // MTP under yarn: the three MTP rope call sites read the SAME per-rank descriptor as the
        // main text path, and the draft/verify rounds run at positions past 262,144 -- so this leg
        // is what shows the extended positions survive the speculative round's own rope, its
        // rope-delta bookkeeping and its I32 position tensors.
        failures += run_leg(artifact, 2, 300000, 3);
    } else {
        std::cout << "skip: the tp2 leg needs two CUDA devices\n";
    }
    // One device holds all four KV heads, so the same prompt needs roughly twice the KV pool here
    // (~33 KiB/token at INT8, about 8.8 GiB at this ceiling, on top of the ~19 GiB model). The
    // ceiling is still chosen so the needle lands several thousand tokens PAST 262,144 rather than
    // just over it.
    failures += run_leg(artifact, 1, 280000);

    if (failures != 0) {
        std::cerr << failures << " yarn long-context failure(s)\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
