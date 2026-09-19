// Opt-in dual-device smoke test for the tensor-parallel forward path.
//
// Load the real qwen3.8-27B NVFP4 artifact across two GPUs, prefill a prompt long enough to cross
// several chunk boundaries, generate greedily, and require the tp2 result to agree with the tp1
// result on the same prompt in the same process. Anything that silently halves a layer,
// mismatches a head map, or drops one rank's contribution shows up here as disagreement rather
// than as a plausible-looking but wrong answer.
//
// Two further legs cover what ninfer-serve does BY DEFAULT and the single-request CLI gate never
// touches -- a batched decode round over two concurrent sequences, and prefix reuse with a nonzero
// suffix (`allow_prefix_reuse` defaults to true). The long-context needle evaluation runs through
// serve, so leaving those uncovered would push the first evidence for them into that evaluation.
//
// Skipped (exit 77) unless NINFER_QWEN3_8_27B_WEIGHTS names an artifact AND at least two CUDA
// devices are visible, so it costs nothing in a default `ctest` run.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kMaxContext   = 8192;
constexpr std::uint32_t kPrefillChunk = 1024;
constexpr std::uint32_t kOutputTokens = 32;
// Long enough that prefill spans four chunks and the last one is partial, so chunk-boundary
// bookkeeping (positions, GDN state carry-over, KV append offsets) is exercised on both devices.
constexpr std::size_t kPromptTokens = 3300;
// The concurrency/reuse legs use a smaller context so two lanes fit one KV pool, and shorter
// prompts so the leg costs seconds rather than minutes.
constexpr std::uint32_t kConcurrentContext = 4096;
constexpr std::size_t kShortPromptTokens   = 600;

ninfer::EngineOptions engine_options(const char* artifact, int tp) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context   = kMaxContext;
    options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(kMaxContext);
    options.prefill_chunk = kPrefillChunk;
    options.kv_cache      = ninfer::KvCacheStorage::Int8Group64;
    options.tp            = tp;
    if (tp == 2) {
        options.devices = {0, 1};
    } else {
        options.device = 0;
    }
    return options;
}

// An engine that can hold two sequences at once. The KV pool must be entitled for both lanes, so
// the capacity is 2x the per-sequence context; `max_context` is smaller than the single-lane
// engine's purely to keep the pool the same size.
//
// CUDA graphs are ON at both widths, which is the product default and is available at tp2 as
// well (see tests/targets/qwen3_6_27b/test_graph_tp2.cpp). Before tp2 had a capture path this leg
// forced them off, because a tp1-with-graphs reference and a tp2-eager subject differed for a
// reason unrelated to the split: a captured decode graph takes its GQA execution envelope from
// the graph profile's frontier RANGE while an eager step uses the exact frontier, and the envelope
// selects the attention split policy. With both sides captured that confound is gone, and running
// graphs here is what makes the batched and prefix-reuse legs exercise the CAPTURED batch>1 decode
// program -- per-batch-size profiles, cross-device edges and all.
ninfer::EngineOptions concurrent_engine_options(const char* artifact, int tp) {
    ninfer::EngineOptions options = engine_options(artifact, tp);
    options.max_context           = kConcurrentContext;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(2 * kConcurrentContext);
    options.max_concurrency = 2;
    options.use_cuda_graph  = true;
    return options;
}

// A deterministic, non-degenerate token sequence: a fixed cycle of ordinary ids, so the test does
// not depend on the tokenizer and every chunk carries different content.
std::vector<ninfer::TokenId> synthetic_prompt(std::size_t tokens = kPromptTokens,
                                              std::size_t seed = 0) {
    std::vector<ninfer::TokenId> prompt;
    prompt.reserve(tokens);
    for (std::size_t index = 0; index < tokens; ++index) {
        prompt.push_back(static_cast<ninfer::TokenId>(1000 + ((index + seed) * 37) % 4096));
    }
    return prompt;
}

// The second concurrency lane needs a prompt that is genuinely DIFFERENT, not a rotation of the
// first: `synthetic_prompt`'s arithmetic progression shifted by a seed is the same pattern
// starting elsewhere, and one such shift was measured to put this model on a knife edge (its
// greedy answer changes between batch 1 and batch 2 at tp1, with no tensor parallelism involved).
// A quadratic sequence gives an unrelated token stream; the leg still verifies stability rather
// than trusting it.
std::vector<ninfer::TokenId> quadratic_prompt(std::size_t tokens) {
    std::vector<ninfer::TokenId> prompt;
    prompt.reserve(tokens);
    for (std::size_t index = 0; index < tokens; ++index) {
        prompt.push_back(
            static_cast<ninfer::TokenId>(1000 + ((index * index * 13) + index * 7) % 4096));
    }
    return prompt;
}

ninfer::RequestOptions greedy_options(std::uint32_t tokens, bool allow_prefix_reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = allow_prefix_reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

std::vector<ninfer::TokenId> generate_greedy(ninfer::Engine& engine,
                                             const std::vector<ninfer::TokenId>& prompt,
                                             std::uint32_t tokens          = kOutputTokens,
                                             bool allow_prefix_reuse       = false) {
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(tokens, allow_prefix_reuse));
    return result.generated_token_ids;
}

int verify_memory_table(const ninfer::LoadSummary& load) {
    if (load.tp != 2) {
        std::cerr << "tp2 load summary does not report two ranks\n";
        return 1;
    }
    const ninfer::DeviceMemoryReport& a = load.devices[0];
    const ninfer::DeviceMemoryReport& b = load.devices[1];
    if (a.device == b.device) {
        std::cerr << "tp2 memory table names the same device twice\n";
        return 1;
    }
    if (a.weights_bytes == 0 || b.weights_bytes == 0 || a.kv_pool_bytes == 0 ||
        a.gdn_state_bytes == 0 || a.workspace_bytes == 0 || a.total_bytes == 0) {
        std::cerr << "tp2 memory table has empty rows\n";
        return 1;
    }
    // The split is symmetric by construction: identical shard geometry on both devices.
    if (a.weights_bytes != b.weights_bytes || a.kv_pool_bytes != b.kv_pool_bytes ||
        a.gdn_state_bytes != b.gdn_state_bytes || a.sequence_bytes != b.sequence_bytes ||
        a.workspace_bytes != b.workspace_bytes) {
        std::cerr << "tp2 memory table is not symmetric across devices\n";
        return 1;
    }
    // Weights are the model's ~19.7 GiB halved; anything close to the whole model on one device
    // means the shard map or the per-device model view silently fell back to a full copy.
    constexpr std::uint64_t kGiB = 1024ULL * 1024ULL * 1024ULL;
    if (a.weights_bytes > 13 * kGiB || a.weights_bytes < 8 * kGiB) {
        std::cerr << "tp2 per-device weight bytes (" << a.weights_bytes
                  << ") are outside the halved-model band\n";
        return 1;
    }
    // Per-device KV per token: 2 of 4 KV heads x head_dim 256 x (K and V) x 1 byte of INT8 code,
    // plus one FP16 scale per 64-element group = 16 KiB + 0.5 KiB = 16.5 KiB. This is the constant
    // the 1M budget extrapolates, so pinning it here is what makes that extrapolation checkable.
    const double kib_per_token =
        static_cast<double>(a.kv_pool_bytes) / static_cast<double>(kMaxContext) / 1024.0;
    if (kib_per_token < 16.0 || kib_per_token > 17.0) {
        std::cerr << "tp2 per-device KV pool is " << kib_per_token
                  << " KiB/token, expected ~16.5 (2 of 4 heads, INT8 + FP16 G64 scales)\n";
        return 1;
    }
    return 0;
}

// --- leg: batched decode over two concurrent sequences -------------------------------------------
//
// Everything the single-request gate exercises runs at batch 1, where `active_sequence_batch_` is
// the trivial case. A batch of two takes a different route through the whole tp2 mixer: batched
// GQA views over each device's own KV pages, the fused GDN snapshot projection instead of the
// prefill conv path, per-lane state slots and KV table rows, and a vocabulary gather that loops
// over columns instead of running once. `max_concurrency` is 1 by default in EngineOptions but 8
// in ninfer-serve, so this is a default-configuration path for the product.
//
// WHAT IS COMPARED, AND WHY. The question this leg owns is "does BATCHING break tp2", which is not
// the same question as "does tp2 match tp1" -- the parity leg above owns that one. So each lane is
// compared against ITS OWN engine's single-lane answer, and tp1 supplies the per-prompt control
// for how much batch-invariance is achievable at all.
//
// That control is necessary, not defensive. Measured on this artifact, some prompts sit on a
// greedy knife edge where ANY last-bit perturbation flips the continuation within a few tokens:
// for one such prompt, tp1's own batch-1 and batch-2 answers agree for only 3 of 32 tokens, and
// tp1 and tp2 agree for only 2 even at batch 1. Demanding exact equality on such a prompt would
// fail every correct implementation. The requirement is therefore that tp2's batching be at least
// as faithful as tp1's on the same prompt, plus a hard exact requirement on at least one lane
// whose prompt IS fully batch-invariant on the reference -- that lane is what gives the leg its
// teeth, and a real defect (a wrong KV table row, a shared GDN state slot, a mis-indexed gather
// column, an out-of-bounds peer write) breaks it immediately rather than late.
struct ConcurrentRound {
    std::vector<ninfer::TokenId> first;
    std::vector<ninfer::TokenId> second;
    std::uint64_t rounds = 0;
    std::uint64_t rows   = 0;
};

ConcurrentRound run_concurrent(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& first,
                               const std::vector<ninfer::TokenId>& second) {
    const ninfer::RuntimeStats before = engine.runtime_stats();
    // Both submitted before either is waited on, so the scheduler can put them in one round.
    ninfer::GenerationHandle handle_first =
        engine.submit(engine.prepare_tokens(first), greedy_options(kOutputTokens, false));
    ninfer::GenerationHandle handle_second =
        engine.submit(engine.prepare_tokens(second), greedy_options(kOutputTokens, false));
    ConcurrentRound out;
    out.first                        = handle_first.wait().generated_token_ids;
    out.second                       = handle_second.wait().generated_token_ids;
    const ninfer::RuntimeStats after = engine.runtime_stats();
    out.rounds                       = after.decode_rounds - before.decode_rounds;
    out.rows                         = after.decode_row_rounds - before.decode_row_rounds;
    return out;
}

std::size_t prefix_agreement(const std::vector<ninfer::TokenId>& a,
                             const std::vector<ninfer::TokenId>& b) {
    std::size_t agreed = 0;
    while (agreed < a.size() && agreed < b.size() && a[agreed] == b[agreed]) { ++agreed; }
    return agreed;
}

int require_batched(const ConcurrentRound& round, const char* label) {
    if (round.rounds == 0 || round.rows <= round.rounds) {
        std::cerr << label << ": no batched decode round was executed (" << round.rounds
                  << " rounds, " << round.rows << " rows) -- the concurrency leg proved nothing\n";
        return 1;
    }
    if (round.first.size() != kOutputTokens || round.second.size() != kOutputTokens) {
        std::cerr << label << ": a concurrent lane did not generate the requested tokens\n";
        return 1;
    }
    return 0;
}

int exercise_concurrent_decode(const char* artifact, const std::vector<ninfer::TokenId>& first,
                               const std::vector<ninfer::TokenId>& second) {
    // Reference: how batch-invariant each prompt is on the single-device implementation.
    std::array<std::size_t, 2> reference{};
    ConcurrentRound tp1;
    {
        ninfer::Engine engine(concurrent_engine_options(artifact, 1));
        const std::vector<ninfer::TokenId> lone_first  = generate_greedy(engine, first);
        const std::vector<ninfer::TokenId> lone_second = generate_greedy(engine, second);
        tp1                                            = run_concurrent(engine, first, second);
        reference[0] = prefix_agreement(lone_first, tp1.first);
        reference[1] = prefix_agreement(lone_second, tp1.second);
    }
    if (const int result = require_batched(tp1, "tp1"); result != 0) { return result; }
    if (tp1.first == tp1.second) {
        std::cerr << "the two concurrency prompts produce the same answer, so the leg cannot tell "
                     "the lanes apart\n";
        return 1;
    }
    if (reference[0] != kOutputTokens && reference[1] != kOutputTokens) {
        std::cerr << "neither concurrency probe is batch-invariant at tp1 (" << reference[0]
                  << " and " << reference[1] << " of " << kOutputTokens
                  << ") -- the leg has no full-strength lane, re-choose a prompt\n";
        return 1;
    }

    ConcurrentRound tp2;
    ConcurrentRound tp2_repeat;
    std::array<std::size_t, 2> observed{};
    {
        ninfer::Engine engine(concurrent_engine_options(artifact, 2));
        const std::vector<ninfer::TokenId> lone_first  = generate_greedy(engine, first);
        const std::vector<ninfer::TokenId> lone_second = generate_greedy(engine, second);
        tp2                                            = run_concurrent(engine, first, second);
        tp2_repeat                                     = run_concurrent(engine, first, second);
        observed[0] = prefix_agreement(lone_first, tp2.first);
        observed[1] = prefix_agreement(lone_second, tp2.second);
    }
    if (const int result = require_batched(tp2, "tp2"); result != 0) { return result; }
    // The tp2-specific hazard for batching is a race between the two devices' streams; a stable
    // repeat is what rules it out.
    if (tp2.first != tp2_repeat.first || tp2.second != tp2_repeat.second) {
        std::cerr << "tp2 batched decode is not reproducible run to run -- the two devices' "
                     "streams are racing\n";
        return 1;
    }
    if (tp2.first == tp2.second) {
        std::cerr << "tp2 batched decode produced the same answer on both lanes -- the lanes are "
                     "sharing state\n";
        return 1;
    }
    for (std::size_t lane = 0; lane < 2; ++lane) {
        if (observed[lane] < reference[lane]) {
            std::cerr << "tp2 batching is less faithful than tp1's on lane " << lane << " ("
                      << observed[lane] << " vs " << reference[lane] << " of " << kOutputTokens
                      << " tokens preserved against the same engine's single-lane answer)\n";
            return 1;
        }
    }
    std::cout << "tp2 batched decode: " << tp2.rows << " rows over " << tp2.rounds
              << " rounds, reproducible, lanes distinct, batch-invariance " << observed[0] << "/"
              << observed[1] << " vs tp1's " << reference[0] << "/" << reference[1] << "\n";
    return 0;
}

// --- leg: prefix reuse ---------------------------------------------------------------------------
//
// `allow_prefix_reuse` defaults to TRUE, so a served conversation re-submits a growing prompt and
// prefills only the suffix. At tp2 that means a prefill chunk with `text_kv_base_ > 0`: absolute
// positions, no GDN state reset, and both devices continuing from KV pages they already hold. This
// leg drives both suffix prefill and exact-frontier sampling through the vocabulary-split head.
int exercise_prefix_reuse(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt) {
    constexpr std::uint32_t kBaselineTokens = 6;
    const ninfer::GenerationResult baseline =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(kBaselineTokens, true));
    if (baseline.generated_token_ids.size() != kBaselineTokens) {
        std::cerr << "prefix-reuse baseline did not generate its tokens\n";
        return 1;
    }

    // (a) NONZERO SUFFIX: resume the retained frontier and prefill two more tokens on top of it.
    std::vector<ninfer::TokenId> continuation = prompt;
    continuation.insert(continuation.end(), baseline.generated_token_ids.begin(),
                        baseline.generated_token_ids.end());
    continuation.push_back(prompt.front());

    const ninfer::RuntimeStats before = engine.runtime_stats();
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(4, true));
    const ninfer::RuntimeStats after = engine.runtime_stats();

    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(prompt.size() + baseline.generated_token_ids.size() - 1);
    if (reused.reused_prompt_tokens != expected_reuse) {
        std::cerr << "tp2 append reuse count is " << reused.reused_prompt_tokens << ", expected "
                  << expected_reuse << '\n';
        return 1;
    }
    if (reused.generated_token_ids.size() != 4) {
        std::cerr << "tp2 suffix prefill did not generate its tokens\n";
        return 1;
    }
    // The suffix really was prefilled, and only the suffix: the whole prompt would be ~600 tokens.
    const std::uint64_t computed = after.computed_prefill_tokens - before.computed_prefill_tokens;
    if (computed == 0 || computed > 8) {
        std::cerr << "tp2 suffix prefill computed " << computed
                  << " tokens, expected a small nonzero suffix\n";
        return 1;
    }

    // (b) ZERO SUFFIX: sample directly from the immediately preceding retained hidden.
    // Both rank-local output heads must run and gather before the bonus token is sampled.
    std::vector<ninfer::TokenId> exact_frontier = continuation;
    exact_frontier.insert(exact_frontier.end(), reused.generated_token_ids.begin(),
                          reused.generated_token_ids.end() - 1);
    const auto before_exact = engine.runtime_stats().computed_prefill_tokens;
    const ninfer::GenerationResult zero_suffix =
        engine.generate(engine.prepare_tokens(exact_frontier), greedy_options(2, true));
    if (zero_suffix.generated_token_ids.size() != 2) {
        std::cerr << "tp2 zero-suffix reuse did not generate its tokens\n";
        return 1;
    }
    if (zero_suffix.reused_prompt_tokens != exact_frontier.size() ||
        zero_suffix.prefix_reuse_path != ninfer::PrefixReusePath::AppendAtFrontier ||
        engine.runtime_stats().computed_prefill_tokens != before_exact ||
        zero_suffix.generated_token_ids.front() != reused.generated_token_ids.back()) {
        std::cerr << "tp2 zero-suffix reuse reported " << zero_suffix.reused_prompt_tokens
                  << " reused tokens; expected exact retained-frontier sampling\n";
        return 1;
    }

    // The engine is still alive: a following request must still work.
    const std::vector<ninfer::TokenId> after_exact =
        generate_greedy(engine, quadratic_prompt(kShortPromptTokens), 4, false);
    if (after_exact.size() != 4) {
        std::cerr << "the engine stopped serving after zero-suffix reuse\n";
        return 1;
    }
    std::cout << "tp2 prefix reuse: " << expected_reuse << " tokens reused, " << computed
              << " prefilled; zero-suffix hit sampled the retained hidden without prefill\n";
    return 0;
}

int exercise(const char* artifact) {
    const std::vector<ninfer::TokenId> prompt = synthetic_prompt();

    std::vector<ninfer::TokenId> tp1_tokens;
    {
        ninfer::Engine engine(engine_options(artifact, 1));
        tp1_tokens = generate_greedy(engine, prompt);
    }
    if (tp1_tokens.size() != kOutputTokens) {
        std::cerr << "tp1 baseline did not generate the requested tokens\n";
        return 1;
    }

    std::vector<ninfer::TokenId> tp2_tokens;
    {
        ninfer::Engine engine(engine_options(artifact, 2));
        if (const int result = verify_memory_table(engine.load_summary()); result != 0) {
            return result;
        }
        tp2_tokens = generate_greedy(engine, prompt);
        // A second run in the same process must be bit-identical: the collectives keep no host
        // state between calls, so a difference here would mean a real ordering hazard.
        const std::vector<ninfer::TokenId> repeat = generate_greedy(engine, prompt);
        if (repeat != tp2_tokens) {
            std::cerr << "tp2 generation is not reproducible within one engine\n";
            return 1;
        }
    }
    if (tp2_tokens.size() != kOutputTokens) {
        std::cerr << "tp2 did not generate the requested tokens\n";
        return 1;
    }

    std::size_t agreed = 0;
    for (std::size_t index = 0; index < kOutputTokens; ++index) {
        if (tp1_tokens[index] != tp2_tokens[index]) { break; }
        ++agreed;
    }
    // Greedy decoding is a hard equality test on every logit argmax, so a genuine numerical
    // agreement shows up as a long identical prefix. The formal parity harness
    // (tools/tp2/parity.cpp, ctest `ninfer_qwen3_8_27b_tp2_parity_test`) owns the cosine/agreement
    // thresholds; this gate only has to catch "the split forward is wrong",
    // which loses agreement immediately rather than late.
    if (agreed < kOutputTokens / 2) {
        std::cerr << "tp1 and tp2 greedy output diverge after " << agreed << " of " << kOutputTokens
                  << " tokens\n";
        return 1;
    }
    std::cout << "tp1/tp2 greedy agreement: " << agreed << "/" << kOutputTokens << " tokens\n";

    // The concurrency and reuse legs need a differently-sized engine (two lanes in one KV pool), so
    // they get their own construction rather than distorting the parity leg's configuration.
    const std::vector<ninfer::TokenId> first  = synthetic_prompt(kShortPromptTokens, 0);
    const std::vector<ninfer::TokenId> second = synthetic_prompt(kShortPromptTokens, 991);
    if (const int result = exercise_concurrent_decode(artifact, first, second); result != 0) {
        return result;
    }
    {
        ninfer::Engine engine(concurrent_engine_options(artifact, 2));
        if (const int result = exercise_prefix_reuse(engine, first); result != 0) { return result; }
    }
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_8_27B_WEIGHTS is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: tensor-parallel smoke test needs two CUDA devices\n";
        return 77;
    }
    if (exercise(artifact) != 0) { return 1; }
    std::cout << "ok\n";
    return 0;
}
