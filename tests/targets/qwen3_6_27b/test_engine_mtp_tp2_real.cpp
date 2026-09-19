// Opt-in dual-device test for MTP3 speculative decoding under tensor parallelism.
//
// Read test_engine_tp2_real.cpp first -- this file is its speculative sibling and reuses its
// structure; what follows documents what is different and, above all, WHICH equality is the right
// one to demand.
//
// THE PRIMARY GATE IS A PER-POSITION ORACLE, NOT TOKEN AGREEMENT. Speculative decoding is lossless
// with respect to the target model's own choices: a draft is committed only if the target's argmax
// at that position agrees with it. That is a statement about POSITIONS, and it is checkable
// directly: for each token t_i the MTP run emitted, teacher-force the context `prompt + t_0..t_{i-1}`
// through a NON-SPECULATIVE engine and ask whether the target's argmax is t_i (leg 0). Nothing
// about batch composition, round boundaries or greedy amplification enters that question, so it is
// the one measurement in this file with no confound in it.
//
// THE OBVIOUS WHOLE-SEQUENCE CRITERION HAS TWO CONFOUNDS, AND BOTH ARE MEASURED HERE RATHER THAN
// ASSERTED IN PROSE:
//
//   1. THE COLUMN-COUNT CONFOUND, which is a SINGLE-DEVICE property. The verify round runs the
//      target over K+1 columns at once; the ordinary decode round runs it over one. Different
//      column counts select different GEMM shapes and a different GQA split policy, so a logit's
//      last bit can differ and greedy amplifies it. `exercise_confound_control` asserts this
//      exists at tp1, on a real reasoning prompt over 96 tokens, before any tp2 criterion leans on
//      it -- if that leg ever stops finding a divergence, the reasoning below has to be revisited.
//   2. THE ORDINARY-SPLIT CONFOUND. The tp2 ORDINARY decode path carries its own last-bit drift
//      from tp1's (`ops::linear_row_parallel`'s own contract records that a split reduction
//      cannot be bit-identical to the whole-K one, and a greedy continuation amplifies it). So
//      "tp2 MTP vs tp2 MTP-off" mixes a speculative question with a plain tensor-parallel one,
//      and on probe A it is that drift, not speculation, that accounts for the whole residual.
//
// So the whole-sequence legs are stated against per-prompt controls:
//
//   * LEG 1 -- tp-faithfulness: `agreement(tp2 MTP, tp1 MTP) >= agreement(tp2 off, tp1 off)`, plus
//     at least one probe EXACT. The ordinary split is the control, because it is what
//     test_engine_tp2_real.cpp already gates.
//   * LEG 2 -- losslessness end to end: on every probe, the tp2 MTP3 answer must reproduce one of
//     the two non-speculative greedy answers token for token. Which one it matches is not a
//     speculative-path question (see confound 2); matching neither is a defect.
//   * LEG 3 -- CUDA graphs: at tp2 the MTP round is captured as ONE cross-device graph holding
//     both devices' nodes, so captured and eager must agree token for token AND accept-for-accept.
//   * LEG 4 -- a batched MTP round over two concurrent lanes, gated by the per-lane oracle plus
//     structural checks. Token agreement across engines is unusable here for a measured reason:
//     the two engines do not even run the same number of rounds.
//   * LEG 5 -- penalties. `SamplingConfig::token_counts` is a raw DEVICE pointer, and the MTP
//     round's acceptance runs on both devices, so this is the leg that would fault (or silently
//     double-count) if rank 1 were handed rank 0's counter lane.
//
// Skipped (exit 77) unless NINFER_QWEN3_8_27B_WEIGHTS names an artifact AND at least two CUDA
// devices are visible, so it costs nothing in a default `ctest` run.

#include "ninfer/engine.h"
#include "product/prompt_input/prompt_input.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kMaxContext        = 8192;
constexpr std::uint32_t kPrefillChunk      = 1024;
constexpr std::uint32_t kOutputTokens      = 64;
constexpr std::uint32_t kControlTokens     = 96;
constexpr std::uint32_t kDraftTokens       = 3;
constexpr std::uint32_t kConcurrentContext = 4096;
constexpr std::size_t kShortPromptTokens   = 600;

// A teacher-forced disagreement is admissible only when it is a near-tie. The floor is derived,
// not fitted: logits here run to roughly +/-20, where BF16's 8 mantissa bits give a quantum of
// 2^-8 * 16 = 0.0625, and a single differing rounding in the last GEMM moves a logit by a few of
// those. 0.5 is eight quanta -- loose enough that a rounding-order flip passes, tight enough that
// a structurally wrong logit (wrong by order one, as the negative controls in
// tools/tp2/parity.cpp show) cannot.
//
// It is a FLOOR on a control-based limit, not the limit itself. The single-device MTP run has its
// own teacher-forced disagreements -- the column-count confound is exactly that -- and they are
// not all near-ties: measured on probe A, tp1 disagrees at 2 of 64 positions with a worst gap of
// 0.75. Holding the split to a tighter standard than the implementation it splits would be
// measuring the confound, so the admissible gap is `max(kNearTieGap, tp1's worst on the same
// prompt)`.
constexpr float kNearTieGap = 0.5F;

// Acceptance floor that does not depend on two engines emitting the same text. Chance acceptance
// is ~1/248077 per drafted token (the draft must hit the target's exact argmax over the whole
// vocabulary), so a quarter of drafts landing is five orders of magnitude above chance and far
// below every configuration measured here (0.47-0.83). A draft side broken by the split -- one
// rank's half of the proposal logits dropped, the id map read on the wrong device -- collapses to
// approximately chance, which is what this sees.
constexpr double kMinAcceptance = 0.25;

ninfer::EngineOptions engine_options(const char* artifact, int tp, bool mtp, bool graphs = true) {
    ninfer::EngineOptions options;
    options.artifact_path  = artifact;
    options.max_context    = kMaxContext;
    options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(kMaxContext);
    options.prefill_chunk  = kPrefillChunk;
    options.kv_cache       = ninfer::KvCacheStorage::Int8Group64;
    options.tp             = tp;
    options.use_cuda_graph = graphs;
    if (tp == 2) {
        options.devices = {0, 1};
    } else {
        options.device = 0;
    }
    if (mtp) {
        // The gated configuration: MTP3 with the small draft head (`--lm-head-draft`), which is
        // the vocabulary-split [131072, 5120] Q4 object plus a REPLICATED id map -- the
        // composition the proposal argmax needs, since that argmax is global over all 131072
        // rows (see tests/ops/test_mtp_split.cpp, Leg D).
        options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens  = kDraftTokens;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    }
    return options;
}

ninfer::EngineOptions concurrent_engine_options(const char* artifact, int tp, bool mtp) {
    ninfer::EngineOptions options = engine_options(artifact, tp, mtp);
    options.max_context           = kConcurrentContext;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(2 * kConcurrentContext);
    options.max_concurrency = 2;
    return options;
}

// Deterministic synthetic prompts, as in the sibling suite: the test must not depend on the
// tokenizer, and every probe must carry different content.
std::vector<ninfer::TokenId> synthetic_prompt(std::size_t tokens, std::size_t seed) {
    std::vector<ninfer::TokenId> prompt;
    prompt.reserve(tokens);
    for (std::size_t index = 0; index < tokens; ++index) {
        prompt.push_back(static_cast<ninfer::TokenId>(1000 + ((index + seed) * 37) % 4096));
    }
    return prompt;
}

std::vector<ninfer::TokenId> quadratic_prompt(std::size_t tokens) {
    std::vector<ninfer::TokenId> prompt;
    prompt.reserve(tokens);
    for (std::size_t index = 0; index < tokens; ++index) {
        prompt.push_back(
            static_cast<ninfer::TokenId>(1000 + ((index * index * 13) + index * 7) % 4096));
    }
    return prompt;
}

// The confound control's prompt is real text, chat-templated by the engine's own frontend, because
// the property it has to exhibit -- a target whose K+1-column and single-column evaluations
// disagree within 96 greedy tokens -- was found on real text and is not a property a synthetic
// token cycle reliably has.
const char* control_prompt_text() {
    return "Explain step by step why the sum of the first n odd numbers equals n squared, then "
           "give a worked example for n=7.";
}

ninfer::RequestOptions greedy_options(std::uint32_t tokens) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    // Start every comparison from a full prefill so both widths use the same prompt partition.
    // The separate prefix gates exercise retained-state and checkpoint continuation.
    options.execution.allow_prefix_reuse = false;
    options.stop.include_model_defaults   = false;
    return options;
}

// The penalty leg's request. Penalties are the only configuration that makes
// `SamplingConfig::token_counts` non-null, and a nonzero temperature is what puts the acceptance
// into sampling mode -- greedy acceptance neither reads nor writes the counters.
ninfer::RequestOptions penalized_options(std::uint32_t tokens) {
    ninfer::RequestOptions options            = greedy_options(tokens);
    options.execution.sampling.temperature    = 0.7F;
    options.execution.sampling.top_p          = 0.8F;
    options.execution.sampling.top_k          = 20;
    options.execution.sampling.presence_penalty  = 0.5F;
    options.execution.sampling.frequency_penalty = 0.3F;
    options.execution.sampling.seed           = 20260826ULL;
    return options;
}

struct Run {
    std::vector<ninfer::TokenId> tokens;
    ninfer::SpeculativeStats speculative;
};

Run generate(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt,
             std::uint32_t tokens = kOutputTokens) {
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(tokens));
    return Run{result.generated_token_ids, result.speculative};
}

Run generate_penalized(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt,
                       std::uint32_t tokens = kOutputTokens) {
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(prompt), penalized_options(tokens));
    return Run{result.generated_token_ids, result.speculative};
}

std::size_t prefix_agreement(const std::vector<ninfer::TokenId>& a,
                             const std::vector<ninfer::TokenId>& b) {
    std::size_t agreed = 0;
    while (agreed < a.size() && agreed < b.size() && a[agreed] == b[agreed]) { ++agreed; }
    return agreed;
}

double acceptance_rate(const ninfer::SpeculativeStats& stats) {
    if (stats.drafted_tokens == 0) { return 0.0; }
    return static_cast<double>(stats.accepted_tokens) / static_cast<double>(stats.drafted_tokens);
}

int require_speculative(const ninfer::SpeculativeStats& stats, const char* label) {
    if (!stats.enabled || stats.backend != ninfer::SpeculativeBackend::Mtp ||
        stats.draft_window != kDraftTokens) {
        std::cerr << label << ": the run did not report an MTP" << kDraftTokens
                  << " configuration\n";
        return 1;
    }
    if (stats.rounds == 0 || stats.drafted_tokens == 0) {
        std::cerr << label << ": no speculative round ran, so the leg proved nothing\n";
        return 1;
    }
    // A draft side that proposes but never lands anything would leave every other leg intact --
    // the target still emits correct tokens, one per round -- while destroying the entire point of
    // the feature. Only an acceptance floor sees it.
    if (stats.accepted_tokens == 0) {
        std::cerr << label << ": " << stats.drafted_tokens
                  << " tokens were drafted and none accepted -- the draft side is not working\n";
        return 1;
    }
    if (stats.accepted_tokens > stats.drafted_tokens) {
        std::cerr << label << ": accepted more tokens than were drafted\n";
        return 1;
    }
    return 0;
}

// --- leg 0: the per-position teacher-forced oracle ------------------------------------------------
//
// `oracle` is a NON-SPECULATIVE engine with logit capture on and one lane. For each emitted token
// it re-prefills `prompt + emitted[0..i)` from scratch, generating exactly one token, and asks
// whether that token -- the target model's own greedy choice at that position -- is what the
// speculative run committed. Every probe is independent: no KV is reused, no round boundary
// survives, and the batch composition of the run under test is irrelevant.

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16U;
    float value                 = 0.0F;
    std::memcpy(&value, &widened, sizeof(value));
    return value;
}

struct Top2 {
    std::int64_t best_index   = -1;
    float best_value          = -std::numeric_limits<float>::infinity();
    float second_value        = -std::numeric_limits<float>::infinity();
    [[nodiscard]] float gap() const { return best_value - second_value; }
};

Top2 top2_of(const std::vector<std::uint16_t>& bits) {
    Top2 out;
    for (std::size_t index = 0; index < bits.size(); ++index) {
        const float value = bf16_to_float(bits[index]);
        if (value > out.best_value) {
            out.second_value = out.best_value;
            out.best_value   = value;
            out.best_index   = static_cast<std::int64_t>(index);
        } else if (value > out.second_value) {
            out.second_value = value;
        }
    }
    return out;
}

struct OracleResult {
    std::size_t positions          = 0;
    std::size_t agreements         = 0;
    std::size_t disagreements     = 0;
    float worst_gap                = 0.0F;
    bool capture_sane              = true;
};

OracleResult teacher_force(ninfer::Engine& oracle, const std::vector<ninfer::TokenId>& prompt,
                           const std::vector<ninfer::TokenId>& emitted) {
    OracleResult out;
    std::vector<ninfer::TokenId> context = prompt;
    context.reserve(prompt.size() + emitted.size());
    for (const ninfer::TokenId expected : emitted) {
        const ninfer::GenerationResult result =
            oracle.generate(oracle.prepare_tokens(context, /*allow_prefix_identity=*/false),
                            greedy_options(1));
        if (result.generated_token_ids.size() != 1 || result.reused_prompt_tokens != 0) {
            out.capture_sane = false;
            return out;
        }
        const ninfer::TokenId produced             = result.generated_token_ids.front();
        const std::vector<std::uint16_t> captured  = oracle.debug_last_round_logits_bf16();
        if (captured.empty()) {
            out.capture_sane = false;
            return out;
        }
        const Top2 ranking = top2_of(captured);
        // The captured vector must be the one the token came from. If the engine's own greedy
        // choice is not this vector's argmax, every gap below is measuring the wrong thing --
        // most plausibly because the capture holds a stale round or spans rows outside the
        // sampling domain.
        if (ranking.best_index != static_cast<std::int64_t>(produced)) {
            out.capture_sane = false;
            return out;
        }
        ++out.positions;
        if (produced == expected) {
            ++out.agreements;
        } else {
            ++out.disagreements;
            out.worst_gap = std::max(out.worst_gap, ranking.gap());
        }
        context.push_back(expected);
    }
    return out;
}

int judge_oracle(const OracleResult& subject, const OracleResult& control, const char* label) {
    if (!subject.capture_sane || !control.capture_sane) {
        std::cerr << label << ": the teacher-forced oracle's own capture check failed\n";
        return 1;
    }
    if (subject.positions == 0 || subject.positions != control.positions) {
        std::cerr << label << ": oracle position counts do not match (" << subject.positions
                  << " vs " << control.positions << ")\n";
        return 1;
    }
    // HOW THE COUNTS ARE JUDGED. When the single-device run agrees with its own target at EVERY
    // position, the prompt has no near-tie site in this horizon and the split must be perfect too
    // -- that is the strong case, and probe A supplies it. When tp1 itself disagrees at d
    // positions, the prompt HAS d near-tie sites within the horizon (that is what the column-count
    // confound is), and the split may legitimately flip a comparable number of them; the bound is
    // stated as 2d, which a structural defect -- wrong shard half, dropped collective, wrong head
    // map -- exceeds by an order of magnitude rather than by one. The gap bound below is what
    // carries the real weight in that regime.
    if (control.disagreements == 0) {
        if (subject.disagreements != 0) {
            std::cerr << label << ": tp1 MTP" << kDraftTokens
                      << " matches its target's own argmax at every one of " << control.positions
                      << " positions and tp2 does not (" << subject.agreements << ")\n";
            return 1;
        }
    } else if (subject.disagreements > 2 * control.disagreements) {
        std::cerr << label << ": tp2 MTP" << kDraftTokens << " disagrees with its target's own "
                  << "argmax at " << subject.disagreements << " of " << subject.positions
                  << " positions, more than twice tp1's " << control.disagreements
                  << " on the same prompt\n";
        return 1;
    }
    const float limit = std::max(kNearTieGap, control.worst_gap);
    if (subject.worst_gap > limit) {
        std::cerr << label << ": a teacher-forced disagreement at tp2 is not a near-tie (worst "
                     "top-2 gap "
                  << subject.worst_gap << " > " << limit << ", the larger of the derived near-tie "
                  << "floor " << kNearTieGap << " and tp1's own worst " << control.worst_gap
                  << ")\n";
        return 1;
    }
    std::cout << label << ": oracle agreement " << subject.agreements << "/" << subject.positions
              << " (tp1 " << control.agreements << "/" << control.positions
              << "), worst disagreement gap " << subject.worst_gap << " (tp1 " << control.worst_gap
              << "), limit " << limit << "\n";
    return 0;
}

// --- legs 1 and 2 --------------------------------------------------------------------------------
struct ProbeResult {
    std::uint64_t peer_egress_rounds = 0;
    std::size_t tp1_mtp_vs_off     = 0;
    std::size_t tp2_mtp_vs_off     = 0;
    std::size_t tp2_off_vs_tp1_off = 0;
    std::size_t tp2_mtp_vs_tp1_mtp = 0;
    bool identical_tp1_tp2         = false;
    bool lossless_against_tp1_off  = false;
    bool lossless_against_tp2_off  = false;
    double tp1_acceptance          = 0.0;
    double tp2_acceptance          = 0.0;
};

int exercise_probe(const char* artifact, const std::vector<ninfer::TokenId>& prompt,
                   const char* label, std::uint32_t tokens, bool run_oracle, ProbeResult& out) {
    Run tp1_mtp;
    Run tp1_off;
    OracleResult tp1_oracle;
    {
        ninfer::Engine engine(engine_options(artifact, 1, /*mtp=*/true));
        // Negative control for the cross-rank egress check below: there is no rank 1 here, so the
        // check must stay inert. A non-zero count would mean it is comparing something other than
        // a peer's record.
        engine.debug_enable_peer_egress_check(true);
        tp1_mtp = generate(engine, prompt, tokens);
        if (engine.debug_peer_egress_check_counts().first != 0) {
            std::cerr << label << ": the cross-rank MTP egress check ran at tp1, where there is no "
                                  "peer to compare against\n";
            return 1;
        }
    }
    {
        ninfer::Engine engine(engine_options(artifact, 1, /*mtp=*/false));
        tp1_off = generate(engine, prompt, tokens);
        if (run_oracle) {
            engine.debug_enable_logit_capture(true);
            tp1_oracle = teacher_force(engine, prompt, tp1_mtp.tokens);
        }
    }
    if (const int result = require_speculative(tp1_mtp.speculative, "tp1 MTP"); result != 0) {
        return result;
    }

    Run tp2_mtp;
    Run tp2_off;
    OracleResult tp2_oracle;
    {
        ninfer::Engine engine(engine_options(artifact, 2, /*mtp=*/true));
        // Rank 1's speculative egress agrees with rank 0's by induction over the increment
        // sites -- but rank 1's copy is written and never read, so nothing else measures it.
        // This compares the two records after every round of the run below. Rank 0 licenses at
        // least one token per round, so a rank 1 that never wrote its egress (all zeros) or wrote
        // a different acceptance would be counted here, not argued away.
        engine.debug_enable_peer_egress_check(true);
        tp2_mtp = generate(engine, prompt, tokens);
        // A second run in the same process must be bit-identical: a difference here is a stream
        // ordering hazard between the two devices, not a numerical one.
        const Run repeat = generate(engine, prompt, tokens);
        if (repeat.tokens != tp2_mtp.tokens) {
            std::cerr << label << ": tp2 MTP generation is not reproducible within one engine\n";
            return 1;
        }
        const std::pair<std::uint64_t, std::uint64_t> egress =
            engine.debug_peer_egress_check_counts();
        if (egress.first == 0) {
            std::cerr << label << ": the cross-rank MTP egress check never ran, so the lockstep "
                                  "claim is still unmeasured\n";
            return 1;
        }
        if (egress.second != 0) {
            std::cerr << label << ": rank 1's MTP egress disagrees with rank 0's in "
                      << egress.second << " fields over " << egress.first
                      << " rounds -- the two ranks are not accepting the same tokens\n";
            return 1;
        }
        out.peer_egress_rounds = egress.first;
    }
    {
        ninfer::Engine engine(engine_options(artifact, 2, /*mtp=*/false));
        tp2_off = generate(engine, prompt, tokens);
        if (run_oracle) {
            engine.debug_enable_logit_capture(true);
            tp2_oracle = teacher_force(engine, prompt, tp2_mtp.tokens);
        }
    }
    if (const int result = require_speculative(tp2_mtp.speculative, "tp2 MTP"); result != 0) {
        return result;
    }
    for (const Run* run : {&tp1_mtp, &tp1_off, &tp2_mtp, &tp2_off}) {
        if (run->tokens.size() != tokens) {
            std::cerr << label << ": a run did not generate the requested tokens\n";
            return 1;
        }
    }

    out.tp2_mtp_vs_tp1_mtp = prefix_agreement(tp1_mtp.tokens, tp2_mtp.tokens);
    out.identical_tp1_tp2  = tp1_mtp.tokens == tp2_mtp.tokens;
    out.tp1_mtp_vs_off     = prefix_agreement(tp1_mtp.tokens, tp1_off.tokens);
    out.tp2_mtp_vs_off     = prefix_agreement(tp2_mtp.tokens, tp2_off.tokens);
    out.tp2_off_vs_tp1_off = prefix_agreement(tp2_off.tokens, tp1_off.tokens);
    out.tp1_acceptance     = acceptance_rate(tp1_mtp.speculative);
    out.tp2_acceptance     = acceptance_rate(tp2_mtp.speculative);
    out.lossless_against_tp2_off = tp2_mtp.tokens == tp2_off.tokens;
    out.lossless_against_tp1_off = tp2_mtp.tokens == tp1_off.tokens;

    // LEG 0. The per-position oracle, judged against the tp1 measurement on the same prompt.
    if (run_oracle) {
        if (const int result = judge_oracle(tp2_oracle, tp1_oracle, label); result != 0) {
            return result;
        }
    }

    // LEG 1. The speculative split must be at least as tp-faithful as the ORDINARY split already
    // gated by test_engine_tp2_real.cpp on the same prompt. Exact equality is not available as a
    // blanket requirement and never was: `ops::linear_row_parallel`'s own contract records that a
    // split reduction is not bit-identical to the whole-K one, so a greedy continuation can
    // diverge on ANY tp2 path, speculative or not. The tp2-off-vs-tp1-off column is that
    // per-prompt control.
    if (out.tp2_mtp_vs_tp1_mtp < out.tp2_off_vs_tp1_off) {
        std::cerr << label << ": tp2 MTP" << kDraftTokens << " tracks tp1 MTP" << kDraftTokens
                  << " for " << out.tp2_mtp_vs_tp1_mtp << " of " << tokens
                  << " tokens, worse than the ordinary path's " << out.tp2_off_vs_tp1_off
                  << " on the same prompt\n";
        return 1;
    }
    // ACCEPTANCE. A split that quietly degrades the draft head -- one rank's half of the proposal
    // logits dropped, the replicated id map read on the wrong device -- keeps emitting correct
    // tokens (the target is still right) and simply stops accepting, so legs 0-2 would all pass.
    // Only an acceptance floor sees it.
    //
    // The floor is ABSOLUTE, and the ratio against tp1 is gated only when the two runs emitted the
    // SAME tokens. Acceptance is a property of the text being generated, not of the
    // implementation: once two runs diverge they are measuring different continuations, and on
    // probe B -- where the tp2 ORDINARY path diverges from tp1's at the same position the
    // speculative one does, i.e. for a reason the ordinary split owns -- the two rates are 0.74
    // and 0.47 on texts that stop being the same after token 12. Gating that ratio would be
    // gating the prompt.
    if (out.tp2_acceptance < kMinAcceptance) {
        std::cerr << label << ": tp2 MTP acceptance " << out.tp2_acceptance << " is below the "
                  << "floor " << kMinAcceptance << " -- the draft side is not landing\n";
        return 1;
    }
    if (out.identical_tp1_tp2 && out.tp2_acceptance < 0.8 * out.tp1_acceptance) {
        std::cerr << label << ": tp2 MTP acceptance " << out.tp2_acceptance
                  << " is below 80% of tp1's " << out.tp1_acceptance
                  << " on a prompt where the two runs emitted identical tokens\n";
        return 1;
    }
    std::cout << label << ": tp2-vs-tp1 " << out.tp2_mtp_vs_tp1_mtp << " (MTP) / "
              << out.tp2_off_vs_tp1_off << " (ordinary), MTP-vs-off " << out.tp1_mtp_vs_off
              << " (tp1) / " << out.tp2_mtp_vs_off << " (tp2), acceptance " << out.tp1_acceptance
              << " (tp1) / " << out.tp2_acceptance << " (tp2)"
              << (out.identical_tp1_tp2
                      ? " [same tokens: the acceptance ratio is gated]"
                      : " [tp1 and tp2 diverge here, so the two acceptance rates describe "
                        "different continuations and only the absolute floor is gated]")
              << "\n";
    return 0;
}

// --- the column-count confound, asserted ----------------------------------------------------------
//
// Everything above that declines to demand `tp2 MTP == tp2 MTP-off` rests on the claim that a
// speculative and an ordinary decode of the SAME model on ONE device can legitimately diverge.
// This leg is that claim as a measurement rather than a comment. It also carries the oracle over
// a longer horizon than the 64-token probes.
int exercise_confound_control(const char* artifact, ProbeResult& out) {
    std::vector<ninfer::TokenId> prompt;
    {
        ninfer::Engine engine(engine_options(artifact, 1, /*mtp=*/false));
        ninfer::PromptInput input =
            ninfer::product::prompt_from_text(control_prompt_text(), /*enable_thinking=*/false);
        prompt = engine.prepare(input).debug_token_ids();
    }
    if (prompt.empty()) {
        std::cerr << "control: the chat template produced no tokens\n";
        return 1;
    }
    if (const int result =
            exercise_probe(artifact, prompt, "control", kControlTokens, /*run_oracle=*/true, out);
        result != 0) {
        return result;
    }
    if (out.tp1_mtp_vs_off >= kControlTokens) {
        std::cerr << "control: tp1 MTP and tp1 MTP-off agree over all " << kControlTokens
                  << " tokens, so the column-count confound this suite's criteria are built "
                     "around is not exhibited -- re-choose the control prompt\n";
        return 1;
    }
    std::cout << "control: the column-count confound is real at tp1 -- MTP and MTP-off first "
                 "differ at position "
              << out.tp1_mtp_vs_off << " of " << kControlTokens << "; the tp2 ordinary split's own "
              << "drift starts at " << out.tp2_off_vs_tp1_off << "\n";
    return 0;
}

// --- leg 3: captured cross-device MTP graph vs the eager two-stream path --------------------------
int exercise_graphs_vs_eager(const char* artifact, const std::vector<ninfer::TokenId>& prompt) {
    Run captured;
    Run eager;
    {
        ninfer::Engine engine(engine_options(artifact, 2, /*mtp=*/true, /*graphs=*/true));
        captured = generate(engine, prompt);
    }
    {
        ninfer::Engine engine(engine_options(artifact, 2, /*mtp=*/true, /*graphs=*/false));
        eager = generate(engine, prompt);
    }
    if (captured.tokens != eager.tokens) {
        std::cerr << "tp2 MTP under CUDA graphs diverges from the eager path after "
                  << prefix_agreement(captured.tokens, eager.tokens) << " of " << kOutputTokens
                  << " tokens\n";
        return 1;
    }
    // The acceptance pattern, not just the token stream, must match: the same accepted set is what
    // makes the two paths the same computation rather than two that happen to converge.
    if (captured.speculative.accepted_tokens != eager.speculative.accepted_tokens ||
        captured.speculative.drafted_tokens != eager.speculative.drafted_tokens ||
        captured.speculative.rounds != eager.speculative.rounds) {
        std::cerr << "tp2 MTP graph and eager runs committed the same tokens through a different "
                     "accept pattern\n";
        return 1;
    }
    std::cout << "tp2 MTP graphs vs eager: identical over " << kOutputTokens << " tokens, "
              << captured.speculative.rounds << " rounds, acceptance "
              << acceptance_rate(captured.speculative) << "\n";
    return 0;
}

// --- leg 4: a batched MTP round over two concurrent lanes -----------------------------------------
struct ConcurrentRound {
    std::vector<ninfer::TokenId> first;
    std::vector<ninfer::TokenId> second;
    ninfer::SpeculativeStats first_stats;
    std::uint64_t rounds = 0;
    std::uint64_t rows   = 0;
};

ConcurrentRound run_concurrent(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& first,
                               const std::vector<ninfer::TokenId>& second) {
    const ninfer::RuntimeStats before = engine.runtime_stats();
    ninfer::GenerationHandle handle_first =
        engine.submit(engine.prepare_tokens(first), greedy_options(kOutputTokens));
    ninfer::GenerationHandle handle_second =
        engine.submit(engine.prepare_tokens(second), greedy_options(kOutputTokens));
    ConcurrentRound out;
    const ninfer::GenerationResult result_first  = handle_first.wait();
    const ninfer::GenerationResult result_second = handle_second.wait();
    out.first                                    = result_first.generated_token_ids;
    out.second                                   = result_second.generated_token_ids;
    out.first_stats                              = result_first.speculative;
    const ninfer::RuntimeStats after             = engine.runtime_stats();
    out.rounds                                   = after.decode_rounds - before.decode_rounds;
    out.rows = after.decode_row_rounds - before.decode_row_rounds;
    return out;
}

int require_batched(const ConcurrentRound& round, const char* label) {
    if (round.rounds == 0 || round.rows <= round.rounds) {
        std::cerr << label << ": no batched MTP round was executed (" << round.rounds
                  << " rounds, " << round.rows << " rows) -- the concurrency leg proved nothing\n";
        return 1;
    }
    if (round.first.size() != kOutputTokens || round.second.size() != kOutputTokens) {
        std::cerr << label << ": a concurrent lane did not generate the requested tokens\n";
        return 1;
    }
    return 0;
}

int exercise_concurrent_mtp(const char* artifact, const std::vector<ninfer::TokenId>& first,
                            const std::vector<ninfer::TokenId>& second) {
    ConcurrentRound tp1;
    std::array<OracleResult, 2> tp1_oracle{};
    {
        ninfer::Engine engine(concurrent_engine_options(artifact, 1, /*mtp=*/true));
        tp1 = run_concurrent(engine, first, second);
    }
    if (const int result = require_batched(tp1, "tp1 MTP"); result != 0) { return result; }
    if (tp1.first == tp1.second) {
        std::cerr << "the two concurrency prompts produce the same answer, so the leg cannot tell "
                     "the lanes apart\n";
        return 1;
    }
    {
        ninfer::Engine oracle(engine_options(artifact, 1, /*mtp=*/false));
        oracle.debug_enable_logit_capture(true);
        tp1_oracle[0] = teacher_force(oracle, first, tp1.first);
        tp1_oracle[1] = teacher_force(oracle, second, tp1.second);
    }

    ConcurrentRound tp2;
    ConcurrentRound tp2_repeat;
    std::array<OracleResult, 2> tp2_oracle{};
    {
        ninfer::Engine engine(concurrent_engine_options(artifact, 2, /*mtp=*/true));
        tp2        = run_concurrent(engine, first, second);
        tp2_repeat = run_concurrent(engine, first, second);
    }
    if (const int result = require_batched(tp2, "tp2 MTP"); result != 0) { return result; }
    if (const int result = require_speculative(tp2.first_stats, "tp2 MTP batched"); result != 0) {
        return result;
    }
    if (tp2.first != tp2_repeat.first || tp2.second != tp2_repeat.second) {
        std::cerr << "tp2 batched MTP decode is not reproducible run to run -- the two devices' "
                     "streams are racing\n";
        return 1;
    }
    if (tp2.first == tp2.second) {
        std::cerr << "tp2 batched MTP decode produced the same answer on both lanes -- the lanes "
                     "are sharing state\n";
        return 1;
    }
    {
        ninfer::Engine oracle(engine_options(artifact, 2, /*mtp=*/false));
        oracle.debug_enable_logit_capture(true);
        tp2_oracle[0] = teacher_force(oracle, first, tp2.first);
        tp2_oracle[1] = teacher_force(oracle, second, tp2.second);
    }

    // THE PER-LANE ORACLE IS THIS LEG'S GATE. Comparing token streams across the two engines is
    // not available here, and the reason is measured rather than statistical: the two engines do
    // not run the same rounds. On this fixture tp1 retires the pair in 21 rounds over 38 rows and
    // tp2 in 23 over 40, because a round's batch composition depends on when each lane finishes
    // prefill -- lanes finish prefill at different times, so some steps run at batch 1 and some
    // at batch 2 -- and at batch B the verify round evaluates B*(K+1)
    // columns while a lone lane evaluates K+1. The per-position oracle has no such dependency: it
    // asks, of each lane separately, whether the batched speculative round committed what the
    // target model itself would commit.
    for (std::size_t lane = 0; lane < 2; ++lane) {
        const std::string label = "tp2 batched MTP lane " + std::to_string(lane);
        if (const int result = judge_oracle(tp2_oracle[lane], tp1_oracle[lane], label.c_str());
            result != 0) {
            return result;
        }
    }
    // The two engines batch differently (see above), so their lanes generate different text after
    // the first divergence and only the absolute floor is gateable here; the tp1 rate is reported
    // beside it.
    const double tp1_batched_acceptance = acceptance_rate(tp1.first_stats);
    const double tp2_batched_acceptance = acceptance_rate(tp2.first_stats);
    if (tp2_batched_acceptance < kMinAcceptance) {
        std::cerr << "tp2 batched MTP acceptance " << tp2_batched_acceptance
                  << " is below the floor " << kMinAcceptance << "\n";
        return 1;
    }
    std::cout << "tp2 batched MTP decode: " << tp2.rows << " rows over " << tp2.rounds
              << " rounds (tp1: " << tp1.rows << "/" << tp1.rounds
              << "), reproducible, lanes distinct, acceptance " << tp2_batched_acceptance
              << " vs tp1's " << tp1_batched_acceptance << "\n";
    return 0;
}

// --- leg 5: penalties, i.e. the one configuration with a device pointer in the sampling config ----
//
// `ops::SamplingConfig::token_counts` is a raw DEVICE pointer, non-null only when a presence or
// frequency penalty is set, and `speculative_accept_greedy_drafts` READS it and atomically WRITES
// it -- but only in sampling mode; greedy acceptance never touches it. The MTP round's acceptance
// is replicated on BOTH devices, so this configuration is the only one that can hand rank 1 a
// pointer into rank 0's arena. Without peer mapping that is an illegal access that kills the
// process; with it, a silent double-increment of the shared counters. Every other leg in this file
// runs greedy and would never notice.
//
// The fix gives rank 1 its own counter lane (PeerRuntime::token_counts) and its own pinned ingress
// record carrying that pointer, with rank 0 as the source of truth. What is asserted here:
// the run completes, it is reproducible, its acceptance tracks tp1's, and the split is no less
// faithful than the ordinary split under the identical sampling configuration. Losslessness is NOT
// assertable under sampling: the acceptance path and the ordinary path draw from different RNG
// purposes, so an MTP and a non-MTP run legitimately differ token for token even at tp1.
int exercise_penalties(const char* artifact, const std::vector<ninfer::TokenId>& prompt) {
    Run tp1_mtp;
    Run tp1_off;
    {
        ninfer::Engine engine(engine_options(artifact, 1, /*mtp=*/true));
        tp1_mtp = generate_penalized(engine, prompt);
    }
    {
        ninfer::Engine engine(engine_options(artifact, 1, /*mtp=*/false));
        tp1_off = generate_penalized(engine, prompt);
    }
    Run tp2_mtp;
    Run tp2_off;
    Run tp2_unpenalized;
    {
        ninfer::Engine engine(engine_options(artifact, 2, /*mtp=*/true));
        tp2_mtp          = generate_penalized(engine, prompt);
        const Run repeat = generate_penalized(engine, prompt);
        if (repeat.tokens != tp2_mtp.tokens) {
            std::cerr << "penalties: tp2 MTP with penalties is not reproducible within one "
                         "engine -- the two ranks' counter lanes are not in step\n";
            return 1;
        }
        // Same seed, same temperature, penalties OFF. The counter pointer is null in that
        // configuration, so this run cannot be affected by the counters at all; if it produced
        // the same tokens as the penalized run, the penalty path would be inert and every check
        // below would be measuring nothing. This is the guard against "fixing" the peer pointer
        // by nulling it on BOTH ranks.
        ninfer::RequestOptions unpenalized               = penalized_options(kOutputTokens);
        unpenalized.execution.sampling.presence_penalty  = 0.0F;
        unpenalized.execution.sampling.frequency_penalty = 0.0F;
        const ninfer::GenerationResult plain =
            engine.generate(engine.prepare_tokens(prompt), unpenalized);
        tp2_unpenalized.tokens = plain.generated_token_ids;
    }
    if (tp2_unpenalized.tokens == tp2_mtp.tokens) {
        std::cerr << "penalties: the penalized and unpenalized tp2 MTP runs are identical at the "
                     "same seed -- the penalty counters are not reaching the sampler\n";
        return 1;
    }
    {
        ninfer::Engine engine(engine_options(artifact, 2, /*mtp=*/false));
        tp2_off = generate_penalized(engine, prompt);
    }
    if (const int result = require_speculative(tp2_mtp.speculative, "tp2 MTP penalized");
        result != 0) {
        return result;
    }
    for (const Run* run : {&tp1_mtp, &tp1_off, &tp2_mtp, &tp2_off}) {
        if (run->tokens.size() != kOutputTokens) {
            std::cerr << "penalties: a run did not generate the requested tokens\n";
            return 1;
        }
    }
    const std::size_t mtp_faithful = prefix_agreement(tp1_mtp.tokens, tp2_mtp.tokens);
    const std::size_t off_faithful = prefix_agreement(tp1_off.tokens, tp2_off.tokens);
    if (mtp_faithful < off_faithful) {
        std::cerr << "penalties: tp2 MTP tracks tp1 MTP for " << mtp_faithful << " of "
                  << kOutputTokens << " tokens, worse than the ordinary path's " << off_faithful
                  << " under the identical sampling configuration\n";
        return 1;
    }
    // Under sampling the two engines' texts diverge by construction, so the absolute floor is what
    // is gated. It is still the check that matters here: if rank 1's acceptance saw different
    // penalty counters from rank 0's, the two replicated accepts would license different prefixes,
    // rank 1's alignment round would mask the wrong columns, and acceptance would collapse toward
    // chance within a few rounds.
    const double tp1_acceptance = acceptance_rate(tp1_mtp.speculative);
    const double tp2_acceptance = acceptance_rate(tp2_mtp.speculative);
    if (tp2_acceptance < kMinAcceptance) {
        std::cerr << "penalties: tp2 MTP acceptance " << tp2_acceptance << " is below the floor "
                  << kMinAcceptance
                  << " -- rank 1's acceptance is not seeing the same penalty counters as rank 0's\n";
        return 1;
    }
    std::cout << "tp2 MTP with presence/frequency penalties: ran, reproducible, penalties live, "
              << "tp2-vs-tp1 " << mtp_faithful << " (MTP) / " << off_faithful
              << " (ordinary), acceptance " << tp1_acceptance << " (tp1) / " << tp2_acceptance
              << " (tp2)\n";
    return 0;
}

// --- the DFlash guard must survive ----------------------------------------------------------------
//
// The startup guard was narrowed, not removed: `--tp 2 --spec mtp` is now legal and
// `--tp 2 --spec dflash` still is not. The DFlash weights are sharded by the load plan exactly as
// the MTP ones were, so nothing about loading rejects it -- only the guard does, and a guard with
// no test is a guard that quietly widens.
int exercise_dflash_still_rejected(const char* artifact) {
    ninfer::EngineOptions options    = engine_options(artifact, 2, /*mtp=*/false);
    options.speculative.backend      = ninfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens = 4;
    try {
        ninfer::Engine engine(options);
    } catch (const std::exception& error) {
        const std::string message = error.what();
        if (message.find("DFlash") == std::string::npos) {
            std::cerr << "tp2 + DFlash was rejected, but not for being DFlash: " << message << "\n";
            return 1;
        }
        std::cout << "tp2 + DFlash still rejected: " << message << "\n";
        return 0;
    }
    std::cerr << "tp2 + DFlash was accepted -- the speculative guard was widened too far\n";
    return 1;
}

int exercise(const char* artifact) {
    if (const int result = exercise_dflash_still_rejected(artifact); result != 0) { return result; }

    // Two synthetic probes. The first is the arithmetic-progression prompt the sibling suite uses;
    // the second is an unrelated quadratic stream, so a prompt-specific knife edge cannot hide a
    // defect in both. The oracle runs on probe A and on the confound control; probe B is the cheap
    // second opinion on the whole-sequence criteria.
    const std::vector<ninfer::TokenId> probe_a = synthetic_prompt(kShortPromptTokens, 0);
    const std::vector<ninfer::TokenId> probe_b = quadratic_prompt(kShortPromptTokens);

    ProbeResult a{};
    ProbeResult b{};
    ProbeResult control{};
    if (const int result =
            exercise_probe(artifact, probe_a, "probe A", kOutputTokens, /*run_oracle=*/true, a);
        result != 0) {
        return result;
    }
    if (const int result =
            exercise_probe(artifact, probe_b, "probe B", kOutputTokens, /*run_oracle=*/false, b);
        result != 0) {
        return result;
    }
    if (const int result = exercise_confound_control(artifact, control); result != 0) {
        return result;
    }

    // (1) At least one synthetic probe must reproduce tp1's MTP answer EXACTLY. Without it, leg
    //     1's "no worse than the ordinary split" could be satisfied by two equally-broken sides.
    if (!a.identical_tp1_tp2 && !b.identical_tp1_tp2) {
        std::cerr << "no probe reproduces the tp1 MTP" << kDraftTokens << " answer exactly ("
                  << a.tp2_mtp_vs_tp1_mtp << " and " << b.tp2_mtp_vs_tp1_mtp << " of "
                  << kOutputTokens << ") -- the leg has no full-strength probe\n";
        return 1;
    }
    // (2) Losslessness, required on both synthetic probes: the tp2 MTP3 answer must reproduce one
    //     of the two non-speculative greedy answers exactly. Which one it matches is not this
    //     task's to choose -- tp2's own ordinary path carries the split's rounding and tp1's
    //     carries the single-column verify shape, and a prompt may sit on a knife edge with
    //     respect to either. Matching NEITHER is a defect. (The confound control is deliberately
    //     excluded: it is chosen to be a prompt where the target's own two evaluations disagree.)
    for (const auto& [probe, name] :
         {std::pair<const ProbeResult*, const char*>{&a, "probe A"},
          std::pair<const ProbeResult*, const char*>{&b, "probe B"}}) {
        if (!probe->lossless_against_tp2_off && !probe->lossless_against_tp1_off) {
            std::cerr << name << ": tp2 MTP" << kDraftTokens
                      << " matches neither the tp2 nor the tp1 non-speculative greedy answer ("
                      << probe->tp2_mtp_vs_off << " and " << probe->tp2_mtp_vs_tp1_mtp << " of "
                      << kOutputTokens
                      << ") -- the split speculative round committed tokens the target model "
                         "would not have\n";
            return 1;
        }
        std::cout << name << ": tp2 MTP" << kDraftTokens << " is token-identical to the "
                  << (probe->lossless_against_tp2_off ? "tp2" : "tp1")
                  << " non-speculative greedy answer (" << kOutputTokens << "/" << kOutputTokens
                  << " tokens); rank 1's MTP egress matched rank 0's over "
                  << probe->peer_egress_rounds << " rounds\n";
    }

    if (const int result = exercise_graphs_vs_eager(artifact, probe_a); result != 0) {
        return result;
    }
    if (const int result = exercise_concurrent_mtp(artifact, probe_a, probe_b); result != 0) {
        return result;
    }
    return exercise_penalties(artifact, probe_a);
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
        std::cout << "skip: tensor-parallel MTP test needs two CUDA devices\n";
        return 77;
    }
    if (exercise(artifact) != 0) { return 1; }
    std::cout << "ok\n";
    return 0;
}
