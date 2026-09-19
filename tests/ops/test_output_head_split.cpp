// Two-device parity + composition suite for the replicated embedding and the row-split output
// head with allgather.
//
// AXIS VOCABULARY. The ShardPlan (targets/qwen3_6_27b/impl/load/bindings.cpp,
// tests/targets/test_shard_map.cpp) calls this a "Rows" split, because `text/output_head`'s
// ARTIFACT storage is `[vocab=248320, hidden=5120]` and the vocabulary axis is the artifact's row
// axis (`append_vocab_rows`, `by_rows(...)`). In the OP sense used by `ops::linear` (weight
// `[N,K]`, `N` = output rows, `K` = input/contraction columns) that same axis is `N`: vocabulary is
// the GEMM's OUTPUT dimension. Splitting the output dimension of a GEMM in half and giving each
// device its own half of the outputs, with no cross-device reduction, is exactly what this
// repository's `ops::linear_column_parallel` implements (see include/ninfer/ops/linear.h's own
// "Column-parallel (output-split)" doc). So "vocab row-split" (artifact/ShardPlan vocabulary) and
// "column-parallel" (ops::linear terminology) name the SAME split of the SAME axis; this file uses
// "column-parallel" throughout because that is the Op this suite calls, and calls out the artifact
// naming here once so a reader bouncing between bindings.cpp and this file does not mistake the two
// vocabularies for two different splits.
//
// COMPOSITION (this is how the tp2 forward loop wires the head). After the two ranks compute their
// 124160-row half of the vocabulary logits (`ops::linear_column_parallel`), `ops::allgather_rows`
// reconstructs the full [248320,T] image on BOTH devices -- that is the Op's own contract
// (include/ninfer/ops/allreduce.h), not a choice this suite makes. The MTP `draft_head` and any
// other dual-device consumer can therefore read either copy. For THIS target, only device 0
// samples: the forward loop feeds `ops::argmax` / `ops::sampling` device 0's reconstructed copy,
// never device 1's -- both copies are numerically identical (proved below, exact byte comparison
// across ranks after the gather) so which copy is fed to sampling is a wiring choice, not a
// correctness one, but exactly one of them is the copy the forward loop actually reads.
//
// allgather_rows's contract gathers over `ne[1]` with `ne[0]` the contiguous per-row length
// (include/ninfer/ops/allreduce.h). `ops::linear`'s own output layout is `[N,T]` with N (here the
// vocabulary half) stored fastest, i.e. contiguous per TOKEN, not per vocabulary row -- the
// transpose of what a direct single gather call over the vocabulary axis would need for T>1. This
// suite therefore issues one allgather_rows call PER TOKEN (`row_length=1`, `rows=124160` per
// rank), exactly mirroring the T=1 vocabulary case tests/ops/test_allreduce.cpp already exercises
// (`allgather_rows [1,248320]`) and generalizing it to a token sweep; no Op or kernel changes were
// needed for this, it is purely how the existing contract composes at T>1. The real decode path
// this feeds is single-token (T=1) by design; the sweep here is this suite's own correctness
// margin, not a claim that a wider batch is production-typical for this Op.
//
// FORMATS. `text/token_embedding` and `text/output_head` always bind the same NumericFormat
// (`endpoint_format()` in bindings.cpp), which resolves per weights profile:
//   Qwen38Nvfp4         -> FP8_E4M3FN_ROW_BF16S   (the flagship "qwen3_8" profile)
//   Qwen38GroupwiseInt  -> W8G32_F16S              (the groupwise "qwen3_8" profile)
//   Qwen36GroupwiseInt  -> Q6G64_F16S              (a "qwen3_6" profile; registered identically
//                                                    to the others and parity-tested here too, so
//                                                    no vocabulary-head format is left
//                                                    registry-probed-only)
//   Qwen36Nvfp4         -> W8G32_F16S              (NVFP4 itself is never bound for vocabulary)
// All three are already fully wired at kernel-dispatch level (fp8_dispatch.cpp/fp8_gemv.cu/
// fp8_small_t.cu/fp8_a16_mma.cu; w8_dispatch.cpp; q6_dispatch.cpp) -- what this suite adds is
// their GPU parity coverage. The vocabulary-head shards were registered and host-probed but never
// GPU-parity-tested, because their weights are ~0.6 GB each to generate per run and their parity
// belongs with the output-head + allgather_rows composition this file exercises anyway.
//
// EMBEDDING. `text/token_embedding` is REPLICATED (ShardPlan: empty shard list), so both devices
// hold an independent full copy, and `ops::embedding` (include/ninfer/ops/embedding.h) already
// takes an explicit `Weight` + `cudaStream_t` per call -- no device-0 assumption exists in the
// wrapper (src/ops/wrapper/embedding.cpp): every code path resolves the table and stream the
// caller passed, never a fixed device. This suite's embedding leg proves that empirically: two
// independently uploaded (but logically identical) replicas, gathered on their OWN device's own
// stream, produce byte-identical output, and that output matches an independent host oracle.
//
// SKIP CODE. Every case below requires two CUDA devices in one process; with fewer than two this
// suite reports 77 (this repository's shared skip code), like tests/ops/test_allreduce.cpp and
// tests/ops/test_linear_split.cpp.
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace qw = ninfer::test::quantized_weight;

namespace {

constexpr double kBf16Ulp = 1.0 / 256.0;

constexpr std::int32_t kVocab  = 248320;
constexpr std::int32_t kHidden = 5120;
constexpr std::int32_t kHalf   = kVocab / 2; // 124160/GPU, per the reviewed ShardPlan.

const char* policy_name(ops::LinearPolicy policy) {
    switch (policy) {
    case ops::LinearPolicy::A16Only:
        return "A16Only";
    case ops::LinearPolicy::AllowA8:
        return "AllowA8";
    case ops::LinearPolicy::AllowA4:
        return "AllowA4";
    }
    return "?";
}

void set_device(const ExecutionContext& ec, int rank) {
    cuda_check(cudaSetDevice(ec.dev[rank]->device), "cudaSetDevice");
}

void synchronize_both(const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaStreamSynchronize(ec.dev[rank]->stream), "cudaStreamSynchronize");
    }
}

// See include/ninfer/ops/allreduce.h's "CALLER OBLIGATION" note: plain cudaMemcpy/cudaMemset
// lands on a device's legacy default stream, which DeviceContext::stream (cudaStreamNonBlocking)
// does not implicitly synchronize with.
void retire_staging(const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
}

void* byte_offset(void* base, std::size_t elements, std::size_t element_size) {
    return static_cast<std::uint8_t*>(base) + elements * element_size;
}

// ---------------------------------------------------------------------------------------------
// Weight construction: the coordinate-origin generator the sibling split suites use, cut to what
// this family needs (row-only origin; the vocabulary axis is N, K=5120 stays whole on both ranks).
// ---------------------------------------------------------------------------------------------
qw::PackedWeight make_head_weight(QType qtype, std::int32_t n, std::uint32_t seed,
                                  std::int32_t row_origin) {
    qw::PatternedWeightOptions options;
    options.row_origin              = row_origin;
    options.decorrelate_coordinates = true; // see quantized_weight.h; harmless when not needed.
    if (qtype != QType::FP8_E4M3FN_ROW_BF16S) {
        // Coordinate codes, not the conformance suites' Hashed pattern: hashing keys on the flat
        // group index, which is not translation invariant and so cannot express a shard.
        options.row_split_scale = qw::RowSplitScalePattern::Small;
        options.row_split_codes = qw::RowSplitCodePattern::Coordinate;
    }
    return qw::make_patterned_weight(qtype, n, kHidden, seed, options);
}

std::vector<std::int32_t> seam_samples(std::int32_t extent) {
    std::vector<std::int32_t> probes{0, 1, 31, 32, 33, 63, 64, 127, 128, 129, 255, 256, 511, 512,
                                     extent / 2, extent - 1};
    std::vector<std::int32_t> result;
    for (const std::int32_t probe : probes) {
        if (probe >= 0 && probe < extent &&
            std::find(result.begin(), result.end(), probe) == result.end()) {
            result.push_back(probe);
        }
    }
    return result;
}

// Proves, at the LOGICAL level and before any kernel runs, that `shard` really is `full`'s row
// block [row_origin, row_origin+shard.n). Mirrors test_linear_split.cpp's
// verify_shard_is_parent_block exactly (same fixture, same guarantee).
int verify_shard_is_parent_block(const std::string& label, const qw::PackedWeight& full,
                                 const qw::PackedWeight& shard, std::int32_t row_origin) {
    int failures = 0;
    for (const std::int32_t row : seam_samples(shard.weight.n)) {
        for (const std::int32_t column : seam_samples(kHidden)) {
            const double got      = qw::logical_weight_fp64(shard, row, column);
            const double expected = qw::logical_weight_fp64(full, row + row_origin, column);
            if (got != expected) {
                std::cerr << label << ": shard is not the parent's block at shard (" << row << ','
                          << column << ") -> parent (" << row + row_origin << ',' << column
                          << "): got " << got << " expected " << expected << '\n';
                ++failures;
            }
        }
    }
    return failures;
}

// The two vocabulary halves must be genuinely different byte payloads -- catches the exact
// degenerate-generator failure mode every sibling split suite guards against (an affine
// code/scale pattern that is periodic at the shard stride would make both halves byte-identical,
// so a swapped or duplicated half would pass a naive comparison silently).
int verify_shards_are_distinct(const std::string& label,
                               const std::vector<std::uint8_t>& first,
                               const std::vector<std::uint8_t>& second) {
    if (first == second) {
        std::cerr << label << ": the two vocabulary halves are byte-identical -- shard identity "
                              "is untested\n";
        return 1;
    }
    return 0;
}

struct DeviceWeight {
    DeviceBuffer payload;
    Weight weight{};
};

DeviceWeight upload_weight(const qw::PackedWeight& packed) {
    DeviceWeight out;
    out.payload = to_device(packed.payload);
    out.weight  = packed.device_weight(out.payload.p);
    return out;
}

std::size_t head_workspace_bytes(QType qtype, std::int32_t n, ops::LinearPolicy policy,
                                 std::int32_t tokens) {
    return ops::linear_workspace_capacity_bytes(qtype, n, kHidden, policy, tokens, tokens);
}

int compare(const std::string& label, const std::vector<double>& got,
           const std::vector<double>& expected, const ReductionCriterion& criterion) {
    const ReductionStats stats = compute_reduction_stats(got.data(), expected.data(),
                                                         static_cast<std::int64_t>(got.size()));
    std::cout << "  " << label << ": max_abs=" << stats.maximum_absolute_error
              << " rel_l2=" << stats.relative_l2
              << " gross_limit=" << gross_error_limit(stats, criterion) << '\n';
    return verify_reduction(label, got, expected, criterion);
}

// Every vocabulary-head route retains A16 compute at every T and policy (linear.h's own doc for
// FP8 `[248320,5120]`; W8/Q6 admit no lower-precision route at all outside NVFP4/FP8), and the
// shard geometries either reuse the tp1 kernel template exactly (FP8: bit-exact, like every other
// column-parallel shard that resolves its parent's own instantiation) or fall back to a generic
// SIMT/MMA launcher (W8/Q6: within 1-2 BF16 ulp, because the tp1 small-T table is an exact-geometry
// match that does not cover the halved extent). One blanket 2-ulp bound covers both,
// same as every sibling split suite; the suite prints the observed error so bit-exact cases show
// it plainly.
constexpr ReductionCriterion kHeadCriterion{/*relative_l2*/ 2.0 * kBf16Ulp,
                                            /*gross_absolute*/ 0.0,
                                            /*gross_relative_to_max_reference*/ 2.0 * kBf16Ulp};

struct HeadCase {
    const char* label;
    QType qtype;
    std::uint32_t seed;
    std::vector<std::int32_t> tokens;
    std::vector<ops::LinearPolicy> policies;
};

// ---------------------------------------------------------------------------------------------
// One case: build the real [248320,5120] vocabulary weight and its two 124160-row shards from one
// logical matrix; run (a) tp1 ops::linear on device 0 over the whole weight and (b)
// ops::linear_column_parallel over the two shards; compare; then compose the split halves back
// into full logits with ops::allgather_rows (looped per token, see file header) and assert argmax
// equality against the tp1 reference across the whole token sweep.
// ---------------------------------------------------------------------------------------------
int run_head_case(const HeadCase& test_case, const ExecutionContext& ec,
                  const ops::PeerEvents& events) {
    std::cout << test_case.label << " [" << kVocab << ',' << kHidden << "] -> [" << kHalf << ','
              << kHidden << "] (column-parallel / vocab-row split)\n";
    int failures = 0;

    // --- the real vocabulary weight and its two standalone halves, from one logical matrix ------
    const qw::PackedWeight full   = make_head_weight(test_case.qtype, kVocab, test_case.seed, 0);
    const qw::PackedWeight shard0 = make_head_weight(test_case.qtype, kHalf, test_case.seed, 0);
    const qw::PackedWeight shard1 =
        make_head_weight(test_case.qtype, kHalf, test_case.seed, kHalf);

    failures +=
        verify_shard_is_parent_block(std::string(test_case.label) + " shard 0", full, shard0, 0);
    failures += verify_shard_is_parent_block(std::string(test_case.label) + " shard 1", full,
                                             shard1, kHalf);
    failures += verify_shards_are_distinct(test_case.label, shard0.payload, shard1.payload);
    if (failures != 0) { return failures; }

    set_device(ec, 0);
    DeviceWeight full_w = upload_weight(full);
    std::array<DeviceWeight, 2> shard_w;
    set_device(ec, 0);
    shard_w[0] = upload_weight(shard0);
    set_device(ec, 1);
    shard_w[1] = upload_weight(shard1);

    for (const std::int32_t tokens : test_case.tokens) {
        // --- activation: replicated on both ranks, exactly what a column-parallel op requires ---
        std::vector<float> activation(static_cast<std::size_t>(kHidden) * tokens);
        fill_uniform(activation, test_case.seed * 31u + static_cast<std::uint32_t>(tokens), -1.0F,
                     1.0F);
        round_to_bf16(activation);

        std::array<DeviceBuffer, 2> x_dev;
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            x_dev[static_cast<std::size_t>(rank)] = to_device_bf16(activation);
        }

        for (const ops::LinearPolicy policy : test_case.policies) {
            const std::string label =
                std::string(test_case.label) + " T=" + std::to_string(tokens) + " " +
                policy_name(policy);

            // --- (a) reference: tp1 kernel, whole weight, device 0 -----------------------------
            const std::size_t full_elements = static_cast<std::size_t>(kVocab) * tokens;
            set_device(ec, 0);
            GuardedDeviceBuffer reference(full_elements * sizeof(std::uint16_t));
            reference.fill(0xff);
            DeviceArena reference_arena(std::max<std::size_t>(
                head_workspace_bytes(test_case.qtype, kVocab, policy, tokens), 1));
            Tensor reference_x(x_dev[0].p, DType::BF16, {kHidden, tokens});
            Tensor reference_out(reference.data(), DType::BF16, {kVocab, tokens});

            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            ops::linear(reference_x, full_w.weight, reference_out, policy, reference_arena,
                        ec.dev[0]->stream);
            cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
            failures += reference.verify_guards(label + " reference");
            const std::vector<double> ref_flat =
                from_device_bf16(reference.data(), full_elements);

            // --- (b) the split form: each rank computes its 124160-row half ---------------------
            const std::size_t half_elements = static_cast<std::size_t>(kHalf) * tokens;
            const std::size_t workspace_bytes = std::max<std::size_t>(
                head_workspace_bytes(test_case.qtype, kHalf, policy, tokens), 1);

            std::array<std::optional<GuardedDeviceBuffer>, 2> shard_out;
            std::array<std::optional<DeviceArena>, 2> arena;
            for (int rank = 0; rank < 2; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                set_device(ec, rank);
                shard_out[slot].emplace(half_elements * sizeof(std::uint16_t));
                shard_out[slot]->fill(0xff);
                arena[slot].emplace(workspace_bytes);
            }

            const std::array<Tensor, 2> x{Tensor(x_dev[0].p, DType::BF16, {kHidden, tokens}),
                                          Tensor(x_dev[1].p, DType::BF16, {kHidden, tokens})};
            const std::array<Weight, 2> w{shard_w[0].weight, shard_w[1].weight};
            const std::array<Tensor, 2> out{
                Tensor(shard_out[0]->data(), DType::BF16, {kHalf, tokens}),
                Tensor(shard_out[1]->data(), DType::BF16, {kHalf, tokens})};
            const std::array<WorkspaceArena*, 2> workspace{&*arena[0], &*arena[1]};

            retire_staging(ec);
            ops::linear_column_parallel(x, w, out, policy, workspace, ec);
            synchronize_both(ec);

            // --- per-rank comparison + provenance: rank r must own vocab rows [r*kHalf,(r+1)*kHalf)
            std::array<std::vector<double>, 2> observed;
            for (int rank = 0; rank < 2; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                set_device(ec, rank);
                failures +=
                    shard_out[slot]->verify_guards(label + " rank " + std::to_string(rank));
                observed[slot] = from_device_bf16(shard_out[slot]->data(), half_elements);

                std::vector<double> expected_block(half_elements);
                for (std::int32_t token = 0; token < tokens; ++token) {
                    const std::size_t source =
                        static_cast<std::size_t>(token) * kVocab +
                        static_cast<std::size_t>(rank) * kHalf;
                    std::copy(ref_flat.begin() + static_cast<std::ptrdiff_t>(source),
                             ref_flat.begin() + static_cast<std::ptrdiff_t>(source + kHalf),
                             expected_block.begin() + static_cast<std::ptrdiff_t>(token) * kHalf);
                }
                failures += compare(label + " rank " + std::to_string(rank), observed[slot],
                                    expected_block, kHeadCriterion);
            }
            // Provenance: the two ranks' halves must genuinely differ (a swapped half would
            // scramble every token id downstream of sampling and must be caught here, not later).
            if (observed[0] == observed[1]) {
                std::cerr << label << ": both ranks produced identical logit halves -- the "
                                      "vocabulary split did not actually split\n";
                ++failures;
            }

            // --- (c) composition: allgather_rows reconstructs full [248320,T] on BOTH devices ---
            // Per file header: one call per token (row_length=1, rows=kHalf per rank), since
            // ops::linear's own [N,T] layout stores the vocab axis fastest (per token), not the
            // token axis -- the transpose of what a single multi-token gather call would need.
            std::array<std::optional<GuardedDeviceBuffer>, 2> full_logits;
            for (int rank = 0; rank < 2; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                set_device(ec, rank);
                full_logits[slot].emplace(full_elements * sizeof(std::uint16_t));
                full_logits[slot]->fill(0xcd);
            }
            retire_staging(ec);
            for (std::int32_t token = 0; token < tokens; ++token) {
                const std::array<Tensor, 2> part{
                    Tensor(byte_offset(shard_out[0]->data(),
                                       static_cast<std::size_t>(token) * kHalf, sizeof(std::uint16_t)),
                           DType::BF16, {1, kHalf}),
                    Tensor(byte_offset(shard_out[1]->data(),
                                       static_cast<std::size_t>(token) * kHalf, sizeof(std::uint16_t)),
                           DType::BF16, {1, kHalf})};
                const std::array<Tensor, 2> destination{
                    Tensor(byte_offset(full_logits[0]->data(),
                                       static_cast<std::size_t>(token) * kVocab,
                                       sizeof(std::uint16_t)),
                           DType::BF16, {1, kVocab}),
                    Tensor(byte_offset(full_logits[1]->data(),
                                       static_cast<std::size_t>(token) * kVocab,
                                       sizeof(std::uint16_t)),
                           DType::BF16, {1, kVocab})};
                ops::allgather_rows(destination, part, ec, events);
            }
            synchronize_both(ec);

            std::array<std::vector<double>, 2> reconstructed;
            for (int rank = 0; rank < 2; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                set_device(ec, rank);
                failures += full_logits[slot]->verify_guards(label + " reconstructed rank " +
                                                             std::to_string(rank));
                reconstructed[slot] = from_device_bf16(full_logits[slot]->data(), full_elements);
                failures += compare(label + " reconstructed rank " + std::to_string(rank),
                                    reconstructed[slot], ref_flat, kHeadCriterion);
            }
            // The Op's own contract: both ranks end up with the IDENTICAL full image (exact byte
            // relocation, no arithmetic -- allreduce.h's own doc). The tp2 forward loop reads
            // device 0's copy for sampling (see file header); this proves device 1's copy would be
            // an equally valid source, which is what makes that a wiring choice, not a correctness
            // requirement.
            if (reconstructed[0] != reconstructed[1]) {
                std::cerr << label << ": the two devices' reconstructed logits disagree after "
                                      "allgather_rows\n";
                ++failures;
            }

            // --- (d) argmax equality: logits feed sampling, so the reconstructed image must pick
            // the SAME token as the tp1 reference on every device, at every T in the sweep. -------
            set_device(ec, 0);
            Tensor ref_logits_tensor(reference.data(), DType::BF16, {kVocab, tokens});
            GuardedDeviceBuffer ref_argmax(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
            Tensor ref_argmax_tensor(ref_argmax.data(), DType::I32, {tokens});
            // GuardedDeviceBuffer's construction-time guard fill is a plain cudaMemset on the
            // legacy default stream (arena.cu's DeviceBuffer::fill), which ec.dev[0]->stream
            // (cudaStreamNonBlocking) does not implicitly order against -- the same caller
            // obligation include/ninfer/ops/allreduce.h documents. Retire it before argmax writes
            // to the same allocation on the non-blocking stream.
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            ops::argmax(ref_logits_tensor, ref_argmax_tensor, kVocab, ec.dev[0]->stream);
            cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
            const auto ref_argmax_host =
                from_device<std::int32_t>(ref_argmax.data(), static_cast<std::size_t>(tokens));

            for (int rank = 0; rank < 2; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                set_device(ec, rank);
                Tensor logits_tensor(full_logits[slot]->data(), DType::BF16, {kVocab, tokens});
                GuardedDeviceBuffer argmax_out(static_cast<std::size_t>(tokens) *
                                               sizeof(std::int32_t));
                Tensor argmax_tensor(argmax_out.data(), DType::I32, {tokens});
                cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); // see ref_argmax note
                ops::argmax(logits_tensor, argmax_tensor, kVocab, ec.dev[rank]->stream);
                cuda_check(cudaStreamSynchronize(ec.dev[rank]->stream), "cudaStreamSynchronize");
                failures += argmax_out.verify_guards(label + " argmax rank " +
                                                     std::to_string(rank));
                const auto argmax_host =
                    from_device<std::int32_t>(argmax_out.data(), static_cast<std::size_t>(tokens));
                if (argmax_host != ref_argmax_host) {
                    std::cerr << label << ": argmax on device " << rank
                              << "'s reconstructed logits disagrees with the tp1 reference\n";
                    ++failures;
                }
            }
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------------------------
// Embedding leg: `text/token_embedding` is REPLICATED (ShardPlan: empty shard list; the sharded
// loader materializes a full independent copy on each device), so the gather itself needs no
// split path at all -- this leg PROVES that rather than assuming it. Deliberately small and
// host-comparable (not the real 248320x5120 table): the per-format numerical behaviour of the
// gather kernel is already exhaustively covered by tests/ops/test_embedding.cpp, which this leg
// does not duplicate. What is new here is cross-device consistency of two independent replicas,
// each driven on its OWN device's own stream -- the thing that would break if
// src/ops/wrapper/embedding.cpp secretly assumed device 0.
// ---------------------------------------------------------------------------------------------
int run_embedding_replication_case(const ExecutionContext& ec) {
    constexpr std::int32_t kSmallVocab = 37;
    constexpr std::int32_t kSmallD     = 16;
    const std::vector<int> ids_host{0, 1, 36, 36, 18, 0, 9, 35, 5}; // includes both edge rows.
    const auto tokens = static_cast<std::int32_t>(ids_host.size());

    std::vector<float> table_f(static_cast<std::size_t>(kSmallVocab) * kSmallD);
    fill_uniform(table_f, 909u, -4.0F, 4.0F);
    round_to_bf16(table_f);
    std::vector<std::uint16_t> table_words(table_f.size());
    for (std::size_t i = 0; i < table_f.size(); ++i) { table_words[i] = f32_to_bf16(table_f[i]); }

    // Independent host oracle: the id's row of the table, read directly (no device involved).
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(kSmallD) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::size_t row = static_cast<std::size_t>(ids_host[static_cast<std::size_t>(t)]);
        std::copy(table_words.begin() + static_cast<std::ptrdiff_t>(row * kSmallD),
                 table_words.begin() + static_cast<std::ptrdiff_t>(row * kSmallD + kSmallD),
                 expected.begin() + static_cast<std::ptrdiff_t>(t) * kSmallD);
    }

    int failures = 0;
    std::array<std::vector<std::uint16_t>, 2> out_host;
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        // Independently uploaded (not aliased, not copied peer-to-peer): the same logical table
        // materialized on each device separately, exactly like the loader's replicated path.
        DeviceBuffer table_dev = to_device(table_words);
        DeviceBuffer ids_dev   = to_device_i32(ids_host);
        GuardedDeviceBuffer out_dev(static_cast<std::size_t>(kSmallD) * tokens *
                                    sizeof(std::uint16_t));
        out_dev.fill(0xcd);

        Weight table{};
        table.qtype           = QType::BF16_CTRL;
        table.layout          = QuantLayout::Contiguous;
        table.scale_dtype     = DType::BF16;
        table.payload         = table_dev.p;
        table.payload_bytes   = table_dev.bytes;
        table.qdata           = table_dev.p;
        table.qhigh           = nullptr;
        table.scales          = nullptr;
        table.ndim             = 2;
        table.shape[0]         = kSmallVocab;
        table.shape[1]         = kSmallD;
        table.padded_shape[0]  = kSmallVocab;
        table.padded_shape[1]  = kSmallD;
        table.n                = kSmallVocab;
        table.k                = kSmallD;

        Tensor ids_tensor(ids_dev.p, DType::I32, {tokens});
        Tensor out_tensor(out_dev.data(), DType::BF16, {kSmallD, tokens});

        retire_staging(ec);
        // The device-0 assumption this leg is built to catch would show up here: rank 1's call
        // uses ec.dev[1]->stream and rank 1's own table/out buffers throughout.
        ops::embedding(ids_tensor, table, out_tensor, ec.dev[rank]->stream);
        cuda_check(cudaStreamSynchronize(ec.dev[rank]->stream), "cudaStreamSynchronize");
        failures += out_dev.verify_guards("embedding replication rank " + std::to_string(rank));
        out_host[static_cast<std::size_t>(rank)] =
            from_device<std::uint16_t>(out_dev.data(), static_cast<std::size_t>(kSmallD) * tokens);
    }

    failures += verify_exact("embedding replication: device 0 vs device 1", out_host[0],
                             out_host[1]);
    failures += verify_exact("embedding replication: device 0 vs host oracle", out_host[0],
                             expected);
    failures += verify_exact("embedding replication: device 1 vs host oracle", out_host[1],
                             expected);

    std::cout << (failures ? "FAIL" : "OK") << " embedding replication ("
              << tokens << " ids, both devices)\n";
    return failures;
}

// ---------------------------------------------------------------------------------------------
// Host-only registry probe, mirrors test_linear_split.cpp's verify_registry(): pure host code, so
// it runs even on a one-GPU box and still catches a vocabulary shard extent dropped from a
// dispatch table. It covers all three vocabulary-head formats (FP8/Q6/W8 124160x5120) -- this
// file's GPU cases below additionally parity-test FP8 and W8 (the two "qwen3_8" profile formats)
// and Q6 (the remaining registered format, so no vocabulary-head format is left
// registry-probed-only).
// ---------------------------------------------------------------------------------------------
int verify_registry() {
    struct Entry {
        QType qtype;
        ops::LinearPolicy policy;
    };
    constexpr auto kA16 = ops::LinearPolicy::A16Only;
    constexpr auto kA8  = ops::LinearPolicy::AllowA8;
    constexpr auto kA4  = ops::LinearPolicy::AllowA4;
    const std::vector<Entry> entries{
        {QType::FP8_E4M3FN_ROW_BF16S, kA16},
        {QType::FP8_E4M3FN_ROW_BF16S, kA8},
        {QType::FP8_E4M3FN_ROW_BF16S, kA4},
        {QType::W8G32_F16S, kA16},
        {QType::W8G32_F16S, kA8},
        {QType::Q6G64_F16S, kA16},
        {QType::Q6G64_F16S, kA8},
    };
    int failures = 0;
    for (const Entry& entry : entries) {
        for (const std::int32_t tokens : {1, 2, 48, 1024}) {
            try {
                (void)ops::linear_workspace_capacity_bytes(entry.qtype, kHalf, kHidden,
                                                           entry.policy, tokens, tokens);
            } catch (const std::exception& error) {
                std::cerr << "registry: vocabulary shard [" << kHalf << ',' << kHidden
                          << "] qtype " << static_cast<int>(entry.qtype) << ' '
                          << policy_name(entry.policy) << " T=" << tokens
                          << " rejected: " << error.what() << '\n';
                ++failures;
            }
        }
    }
    // W8 admits no AllowA4 route outside NVFP4/FP8; the vocabulary shard must still reject it.
    try {
        (void)ops::linear_workspace_capacity_bytes(QType::W8G32_F16S, kHalf, kHidden, kA4, 1, 1);
        std::cerr << "registry: W8 vocabulary shard admitted AllowA4 but must not\n";
        ++failures;
    } catch (const std::exception&) {}
    std::cout << (failures ? "FAIL" : "OK") << " registry: " << entries.size()
              << " vocabulary shard entries admitted\n";
    return failures;
}

} // namespace

int main() {
    int failures = verify_registry();
    if (failures != 0) {
        std::cout << "FAIL output head split (registry)\n";
        return 1;
    }

    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: output head split parity requires two CUDA devices, found "
                  << device_count << '\n';
        return 77;
    }

    const ExecutionContext ec({0, 1});
    const bool peer_access = ops::enable_peer_access(ec);
    std::cout << "peer access: "
              << (peer_access ? "enabled (direct P2P)"
                              : "unavailable (CUDA stages the device-to-device copies through "
                                "host memory)")
              << '\n';
    const ops::PeerEvents events(ec);

    failures += run_embedding_replication_case(ec);

    constexpr auto kA16 = ops::LinearPolicy::A16Only;
    constexpr auto kA8  = ops::LinearPolicy::AllowA8;
    constexpr auto kA4  = ops::LinearPolicy::AllowA4;

    const std::vector<HeadCase> cases{
        // FP8: the flagship ("qwen3_8" Qwen38Nvfp4) profile format. T sweep: 1 (decode edge),
        // 8/48 (within the A16-MMA vocabulary window kFp8VocabularyLastA16MmaT=48), 49 (crosses
        // that window, exercising launch_a16's internal chunking loop at both shard AND full
        // shape). Every policy retains A16 compute for vocabulary (linear.h's own doc) but is
        // exercised anyway since all three are legal policy values callers may pass.
        {"fp8 output_head", QType::FP8_E4M3FN_ROW_BF16S, 301u, {1, 8, 48, 49}, {kA16, kA8, kA4}},
        // W8: the groupwise ("qwen3_8" Qwen38GroupwiseInt) profile format. T sweep: 1/4 (shard's
        // own simt_r8_c4), 16/17 (simt_r8_c8 boundary), 33 (tp1's launch_w8_small_t upper edge,
        // where the shard is already on the generic MMA launcher, because tp1's small-T table is
        // an exact-geometry match that does not cover the halved extent), 48 (past tp1's small-T
        // ceiling entirely).
        {"w8 output_head", QType::W8G32_F16S, 302u, {1, 4, 16, 17, 33, 48}, {kA16}},
        // Q6: registered for the Qwen36GroupwiseInt profile; leaving it registry-probed-only
        // would leave one vocabulary-head format without GPU parity coverage. T sweep covers the
        // shard's own MMA column-width steps.
        {"q6 output_head", QType::Q6G64_F16S, 303u, {1, 4, 16, 24, 32, 48}, {kA16}},
    };

    for (const HeadCase& test_case : cases) { failures += run_head_case(test_case, ec, events); }

    std::cout << (failures ? "FAIL" : "OK") << " output head split\n";
    return failures ? 1 : 0;
}
