#pragma once

#include "ninfer/types.h"

#include <chrono>
#include <memory>
#include <utility>

namespace ninfer {

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();

    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] const PromptSummary& summary() const noexcept;
    [[nodiscard]] const PromptPreparationStats& preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

    // Debug-only: the raw token ids this prompt was tokenized to, including any chat-template
    // wrapping -- independent of Engine::count_tokens()'s bare count. Added for the greedy
    // tp1-vs-tp2 parity harness (tools/tp2/parity.cpp), which needs a real chat-templated
    // prompt's ids to build extended "prompt + already-agreed continuation" probes via
    // Engine::prepare_tokens(). Not part of the wire-facing API.
    [[nodiscard]] std::vector<TokenId> debug_token_ids() const;

private:
    class Impl;
    explicit PreparedPrompt(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class GenerationHandle {
public:
    GenerationHandle() noexcept;
    ~GenerationHandle();

    GenerationHandle(GenerationHandle&&) noexcept;
    GenerationHandle& operator=(GenerationHandle&&) noexcept;

    GenerationHandle(const GenerationHandle&)            = delete;
    GenerationHandle& operator=(const GenerationHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept;

    GenerationResult wait(OutputSink* sink = nullptr, const CancellationView& cancellation = {});

private:
    class Impl;
    explicit GenerationHandle(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class Engine {
public:
    explicit Engine(EngineOptions options);
    ~Engine();

    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;

    // Raw token input is retained for parity tools and repeatable performance measurement.
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;

    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PromptCapabilities prompt_capabilities() const;
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    // Establishes queue membership synchronously. Destroying an unconsumed handle cancels its
    // request; wait() owns result consumption and may run independently from GPU execution.
    [[nodiscard]] GenerationHandle
    submit(PreparedPrompt prompt, RequestOptions options,
           std::chrono::steady_clock::time_point pending_deadline = {});

    GenerationResult generate(PreparedPrompt prompt, RequestOptions options,
                              OutputSink* sink                     = nullptr,
                              const CancellationView& cancellation = {});

    [[nodiscard]] const EngineOptions& options() const;
    [[nodiscard]] LoadSummary load_summary() const;
    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    void reset_memory_peaks() noexcept;

    // Debug-only, OFF by default: capture of the full-vocabulary logits behind each sampled token.
    // Not part of the wire-facing API. Until debug_enable_logit_capture(true) is called nothing is
    // allocated or copied and execution is unchanged; debug_last_round_logits_bf16() then returns
    // the raw BF16 bits of the most recent round's logits, in token-id order. Valid only
    // immediately after generate()/wait() returns to the calling thread on a single-lane engine
    // (max_concurrency == 1); with more than one lane in flight there is no per-request
    // association and the buffer's contents are unspecified. Capture happens at prefill's
    // finalization, so a request that asks for exactly one output token -- which completes on
    // prefill's own sampled token -- reads the logits for that token. Added for the greedy
    // tp1-vs-tp2 parity harness (tools/tp2/parity.cpp), which needs exact logit values, not just
    // sampled token ids, to measure per-position agreement and logit distance between engines.
    void debug_enable_logit_capture(bool enabled);
    [[nodiscard]] std::vector<std::uint16_t> debug_last_round_logits_bf16() const;

    // Debug-only, OFF by default: at tp == 2 with MTP, compare rank 1's speculative egress record
    // (licensed tokens/counts, accepted drafts, next proposal) with rank 0's after every round.
    // Both ranks run the acceptance Op over bit-identical inputs, which is why rank 1's copy is
    // never read in production; this turns that argument into a measurement. Returns {rounds
    // compared, field mismatches}, cumulative since the engine was created; a healthy tp2 MTP run
    // reports a positive round count and zero mismatches. No-op at tp1 or without MTP. Added to
    // harden the tp2 MTP lockstep induction -- that both ranks derive the same speculative egress
    // independently -- and not wire-facing.
    void debug_enable_peer_egress_check(bool enabled);
    [[nodiscard]] std::pair<std::uint64_t, std::uint64_t> debug_peer_egress_check_counts() const;

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace ninfer
