// Opt-in Qwen3.8 GGUF Q4_K_M integration gate for two 16 GiB V100s. All engines
// use TP2 and are destroyed before the next one is constructed: this artifact
// cannot fit on one card. NINFER_V100_DUO_PROPOSAL_HEAD selects full (default) or
// optimized; both run the same graph/eager and non-speculative target checks.
//
// This is a behavioral/state gate, not an independent mathematical model oracle.
// The GGML_K and attention Op tests supply the independent FP64/FP32 oracles.
// Here exact graph/eager output and acceptance checks protect graph replay and
// cross-device commits. Fresh, non-speculative teacher-forced logits check every
// MTP output position, recording the chosen token's actual logit deficit (not
// merely the reference's top-two gap). No near-tie tolerance silently licenses
// a different greedy choice. Prefill/decode rounding can cause this strict check
// to fail; such a result requires investigation rather than claiming losslessness.
//
// NINFER_V100_DUO_ARTIFACT=/path/to/qwen3_8_27b_q4_k_m.ninfer ctest -R v100-duo_real
// Add NINFER_V100_DUO_PROPOSAL_HEAD=optimized to exercise the shortlist proposal head.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kContext = 4096;
constexpr std::uint32_t kChunk = 256;
constexpr std::uint32_t kOutputs = 32;
// Registered Qwen3.8 tokenizer domain; the output matrix includes padded rows.
constexpr std::size_t kTokenDomain = 248077;
constexpr std::size_t kProbes = 2;
constexpr std::array<std::size_t, 3> kLogitPositions{0, 15, 31};

using Tokens = std::vector<ninfer::TokenId>;
using Logits = std::vector<std::uint16_t>;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::ProposalHead proposal_head() {
    const char* selected = std::getenv("NINFER_V100_DUO_PROPOSAL_HEAD");
    if (selected == nullptr || std::strcmp(selected, "full") == 0) {
        return ninfer::ProposalHead::Full;
    }
    if (std::strcmp(selected, "optimized") == 0) {
        return ninfer::ProposalHead::Optimized;
    }
    throw std::runtime_error("NINFER_V100_DUO_PROPOSAL_HEAD must be full or optimized");
}

ninfer::EngineOptions engine_options(const char* artifact, bool mtp, bool graphs,
                                     ninfer::ProposalHead head) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.tp = 2;
    options.devices = {0, 1};
    options.max_context = kContext;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(kContext);
    options.prefill_chunk = kChunk;
    options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    options.use_cuda_graph = graphs;
    if (mtp) {
        options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
        options.speculative.proposal_head = head;
    }
    return options;
}

ninfer::RequestOptions request_options(std::uint32_t count) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = count;
    options.execution.allow_prefix_reuse = false;
    options.execution.sampling.temperature = 0.0F;
    options.execution.sampling.presence_penalty = 0.0F;
    options.execution.sampling.frequency_penalty = 0.0F;
    options.stop.include_model_defaults = false;
    return options;
}

ninfer::GenerationResult generate(ninfer::Engine& engine, const Tokens& prompt,
                                  std::uint32_t count = kOutputs) {
    auto result = engine.generate(engine.prepare_tokens(prompt, false), request_options(count));
    require(result.generated_token_ids.size() == count, "generation ended before its token budget");
    require(result.reused_prompt_tokens == 0, "the full-reset probe unexpectedly reused a prefix");
    return result;
}

Tokens chat_tokens(ninfer::Engine& engine, const std::string& text) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    ninfer::ChatMessage message;
    message.parts.push_back({ninfer::MessagePartKind::Text, text, {}});
    input.messages.push_back(std::move(message));
    return engine.prepare(std::move(input)).debug_token_ids();
}

std::array<Tokens, kProbes> prompts(ninfer::Engine& engine) {
    std::string long_text = "Read the following observations and summarize their pattern.\n";
    for (int i = 0; i < 32; ++i) {
        long_text += "Observation " + std::to_string(i) +
                     ": the morning temperature rises, the ice melts, and the river flows faster.\n";
    }
    long_text += "Explain the causal relationship in several sentences.";
    std::array<Tokens, kProbes> result{
        chat_tokens(engine, "Write a Python function that returns the first n Fibonacci numbers. "
                            "Then explain its time and space complexity."),
        chat_tokens(engine, long_text)};
    require(result[0].size() < kChunk, "the short probe no longer fits in one prefill chunk");
    require(result[1].size() > 2 * kChunk && result[1].size() + kOutputs < kContext,
            "the long probe must cross multiple prefill chunks and fit its context");
    return result;
}

float value(std::uint16_t bits) {
    const std::uint32_t raw = std::uint32_t(bits) << 16U;
    float result;
    std::memcpy(&result, &raw, sizeof(result));
    return result;
}

ninfer::TokenId argmax(const Logits& logits) {
    require(logits.size() >= kTokenDomain, "captured logits omit valid vocabulary rows");
    float best = -std::numeric_limits<float>::infinity();
    ninfer::TokenId token = -1;
    for (std::size_t i = 0; i < kTokenDomain; ++i) {
        const float x = value(logits[i]);
        require(std::isfinite(x), "captured logits contain a non-finite vocabulary value");
        if (x > best) {
            best = x;
            token = static_cast<ninfer::TokenId>(i);
        }
    }
    return token;
}

Logits probe(ninfer::Engine& engine, const Tokens& context) {
    const auto result = generate(engine, context, 1);
    auto logits = engine.debug_last_round_logits_bf16();
    const auto best = argmax(logits);
    require(result.generated_token_ids.front() == best,
            "the captured logit argmax is not the token sampled by the same request");
    logits.resize(kTokenDomain);
    return logits;
}

void require_mtp(const ninfer::GenerationResult& result) {
    const auto& stats = result.speculative;
    require(stats.enabled && stats.backend == ninfer::SpeculativeBackend::Mtp &&
                stats.draft_window == 3 && stats.rounds > 0 && stats.drafted_tokens > 0,
            "the MTP3 probe did not execute speculative rounds");
    require(stats.accepted_tokens <= stats.drafted_tokens, "invalid accepted draft count");
}

void compare_rounds(const ninfer::GenerationResult& captured,
                    const ninfer::GenerationResult& eager) {
    require(captured.generated_token_ids == eager.generated_token_ids,
            "graph/eager MTP committed token sequences differ");
    const auto& a = captured.speculative;
    const auto& b = eager.speculative;
    require(a.rounds == b.rounds && a.drafted_tokens == b.drafted_tokens &&
                a.accepted_tokens == b.accepted_tokens && a.fallback_steps == b.fallback_steps &&
                a.accepted_per_position == b.accepted_per_position,
            "graph/eager MTP acceptance or fallback patterns differ");
}

void check_identity(ninfer::Engine& engine) {
    const auto summary = engine.load_summary();
    require(summary.tp == 2 && summary.model_id == "qwen3.8-27b" &&
                summary.weights_id == "gguf-q4-k-m",
            "this gate requires the TP2 Qwen3.8-27B GGUF Q4_K_M artifact");
}

int exercise(const char* artifact, ninfer::ProposalHead head) {
    std::cout << "proposal_head="
              << (head == ninfer::ProposalHead::Full ? "full" : "optimized") << std::endl;
    std::array<Tokens, kProbes> inputs;
    std::array<ninfer::GenerationResult, kProbes> captured;
    std::array<std::array<Logits, kLogitPositions.size()>, kProbes> mtp_logits;
    std::uint64_t accepted_total = 0;
    {
        ninfer::Engine engine(engine_options(artifact, true, true, head));
        check_identity(engine);
        inputs = prompts(engine);
        engine.debug_enable_peer_egress_check(true);
        for (std::size_t p = 0; p < kProbes; ++p) {
            captured[p] = generate(engine, inputs[p]);
            require_mtp(captured[p]);
            accepted_total += captured[p].speculative.accepted_tokens;
            std::cout << "captured probe=" << p << " prompt_tokens=" << inputs[p].size()
                      << " output_tokens=" << captured[p].generated_token_ids.size()
                      << " accepted=" << captured[p].speculative.accepted_tokens
                      << "/" << captured[p].speculative.drafted_tokens << std::endl;
        }
        require(accepted_total > 0, "no proposal was accepted; the MTP commit path was not exercised");
        const auto [rounds, mismatches] = engine.debug_peer_egress_check_counts();
        require(rounds > 0 && mismatches == 0, "TP2 ranks disagree on speculative egress");
        require(engine.memory_summary().cuda_graph_node_count > 0,
                "graphs were requested but no decode graph was captured");
        engine.debug_enable_logit_capture(true);
        for (std::size_t p = 0; p < kProbes; ++p) {
            for (std::size_t i = 0; i < kLogitPositions.size(); ++i) {
                Tokens context = inputs[p];
                const auto& emitted = captured[p].generated_token_ids;
                context.insert(context.end(), emitted.begin(), emitted.begin() + kLogitPositions[i]);
                mtp_logits[p][i] = probe(engine, context);
            }
        }
    }
    {
        ninfer::Engine engine(engine_options(artifact, true, false, head));
        engine.debug_enable_peer_egress_check(true);
        for (std::size_t p = 0; p < kProbes; ++p) {
            const auto eager = generate(engine, inputs[p]);
            require_mtp(eager);
            compare_rounds(captured[p], eager);
        }
        const auto [rounds, mismatches] = engine.debug_peer_egress_check_counts();
        require(rounds > 0 && mismatches == 0, "eager TP2 ranks disagree on speculative egress");
        require(engine.memory_summary().cuda_graph_node_count == 0,
                "the eager control unexpectedly captured a graph");
    }

    std::size_t disagreements = 0;
    float worst_deficit = 0.0F;
    {
        ninfer::Engine engine(engine_options(artifact, false, false, head));
        engine.debug_enable_logit_capture(true);
        for (std::size_t p = 0; p < kProbes; ++p) {
            const auto plain = generate(engine, inputs[p]);
            require(!plain.speculative.enabled, "the target control unexpectedly enabled speculation");
            std::cout << "probe=" << p << " MTP_vs_plain_sequence_equal="
                      << (plain.generated_token_ids == captured[p].generated_token_ids) << std::endl;
            Tokens context = inputs[p];
            for (std::size_t i = 0; i < kOutputs; ++i) {
                const auto logits = probe(engine, context);
                const auto best = argmax(logits);
                const auto emitted = captured[p].generated_token_ids[i];
                require(emitted >= 0 && std::size_t(emitted) < kTokenDomain,
                        "MTP emitted a token outside the registered tokenizer domain");
                if (best != emitted) {
                    const float deficit = value(logits[best]) - value(logits[emitted]);
                    worst_deficit = std::max(worst_deficit, deficit);
                    ++disagreements;
                    std::cerr << "teacher_force probe=" << p << " position=" << i
                              << " target=" << best << " MTP=" << emitted
                              << " emitted_logit_deficit=" << deficit << '\n';
                }
                for (std::size_t j = 0; j < kLogitPositions.size(); ++j) {
                    if (i == kLogitPositions[j]) {
                        require(logits == mtp_logits[p][j],
                                "enabling MTP changed teacher-forced target prefill logit bits");
                    }
                }
                context.push_back(emitted);
            }
        }
    }
    std::cout << "teacher_force_positions=" << kProbes * kOutputs
              << " disagreements=" << disagreements << " worst_emitted_logit_deficit="
              << worst_deficit << std::endl;
    require(disagreements == 0, "MTP outputs failed strict non-speculative teacher-forced argmax");
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_V100_DUO_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_V100_DUO_ARTIFACT is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: V100 Duo integration requires two CUDA devices\n";
        return 77;
    }
    try {
        return exercise(artifact, proposal_head());
    } catch (const std::exception& error) {
        std::cerr << "V100 Duo integration: " << error.what() << '\n';
        return 1;
    }
}
