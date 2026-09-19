// Two-device parity suite for the tensor-parallel split forms of ops::linear
// (linear_column_parallel / linear_row_parallel, include/ninfer/ops/linear.h).
//
// Every case REQUIRES two CUDA devices driven from ONE process; with fewer than two visible
// devices the suite reports the repository's skip code (77) instead of failing, like
// tests/ops/test_allreduce.cpp.
//
// WHAT IS COMPARED. Not an FP64 oracle -- the per-format Linear conformance suites
// (tests/ops/linear/*) already qualify the kernels against one. This suite asks the one question
// they cannot: does splitting a projection across two devices reproduce the single-device result?
// So each case runs
//
//   (a) the tp1 kernel on device 0 over the WHOLE weight, and
//   (b) the split kernels on both devices over the two SHARDS (+ all-reduce for row-parallel),
//
// and compares (b) against (a).
//
// HOW FULL AND SHARDED WEIGHTS COME FROM THE SAME LOGICAL DATA. This is the load-bearing question
// of the whole suite: comparing two kernels over two *different* weights proves nothing. The
// quantized-weight fixture generates every code and every scale as a pure function of the value's
// GLOBAL (row, column) coordinate, so `make_patterned_weight(qtype, N/2, K, seed, {.row_origin =
// N/2})` is a standalone tensor whose logical values ARE the parent's second row block. Nothing
// is copied or sliced between the two: they meet only at the logical level, which is exactly how
// the tp2 loader works (a shard is a re-encoded standalone tensor of the narrowed shape, never a
// pointer offset into the parent payload -- NVFP4's swizzled scale plane and every row-split plane
// are re-based per shard, and validate_nvfp4_weight would reject a pointer view outright).
//
// Because "the generator is translation invariant" is an assumption and not a proof, every case
// first asserts it: verify_shard_is_parent_block() decodes both tensors through the fixture's
// independent logical decoder (logical_weight_fp64) and requires bit-equal FP64 values over a
// coordinate sample that straddles the NVFP4 32-row and 128-row scale-tile seams, the row-split
// group boundaries, and the shard's own first/last row and column. A generator mistake therefore
// fails as a weight mismatch, loudly and separately from any kernel mismatch.
//
// TOLERANCES.
//   column-parallel: each rank evaluates exactly the dot products the tp1 kernel evaluates for the
//     rows it owns, in the same order, so the result is expected to be bit-identical. The stated
//     criterion is still 2 BF16 ulp, because the shard resolves a different template instance and
//     a schedule is free to reassociate; the suite prints the observed error so a silent drift
//     from "exact" to "within tolerance" is visible.
//   row-parallel: NOT bit-identical and cannot be. Each rank rounds its partial to BF16 storage
//     before the collective adds them, so a split evaluation carries two roundings the whole-K
//     accumulation does not. Those roundings are proportional to the PARTIAL magnitudes, not to
//     the summed magnitude, so the criterion is 2 BF16 ulp of the largest output in the tensor
//     rather than 2 ulp of each element -- otherwise an element where the two partials nearly
//     cancel would fail for a completely expected reason.
#include "ninfer/ops/linear.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include "core/device.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace qw = ninfer::test::quantized_weight;

namespace {

// One BF16 ulp: the format carries 8 significand bits, so 2^-8.
constexpr double kBf16Ulp = 1.0 / 256.0;

enum class SplitAxis : std::uint8_t {
    Column, // output rows split; each rank owns N/2 rows, no communication
    Row,    // input columns split; each rank owns K/2 columns, all-reduced
};

struct Case {
    const char* label;
    QType qtype;
    SplitAxis axis;
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
    std::vector<std::int32_t> tokens;
    std::vector<ops::LinearPolicy> policies;
};

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

// Weights, activations and output poison are staged with cudaMemcpy/cudaMemset, which the runtime
// issues on each device's LEGACY DEFAULT stream. DeviceContext::stream is created with
// cudaStreamNonBlocking and does NOT implicitly synchronize with it, so the staging must be
// retired before the split forms read it. This is the caller obligation the Op contract documents.
void retire_staging(const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
}

qw::PackedWeight make_weight(QType qtype, std::int32_t n, std::int32_t k, std::uint32_t seed,
                             std::int32_t row_origin, std::int32_t column_origin) {
    qw::PatternedWeightOptions options;
    options.row_origin    = row_origin;
    options.column_origin = column_origin;
    // Without this the NVFP4 and FP8 code/scale rules are periodic at exactly the tp2 boundaries on
    // BOTH axes -- rows 7168/8192/17408 (13s = 0 mod 16, 5s = 0 mod 8) and columns 3072/8704
    // (7s = 0 mod 16, 3*(s/16) = 0 mod 8) -- so the two shards would carry byte-identical weights
    // and parity would compare a block against a copy of itself while reporting zero error.
    // verify_shards_are_distinct() is what catches a regression here if this is ever dropped.
    options.decorrelate_coordinates = true;
    if (qtype == QType::NVFP4) {
        options.weight_scale_divisor = 0.125F;
        options.input_scale_divisor  = 3.5F;
    } else if (qtype != QType::FP8_E4M3FN_ROW_BF16S) {
        // Coordinate codes, not the conformance suites' Hashed pattern: hashing keys on the flat
        // group index, which is not translation invariant and so cannot express a shard.
        options.row_split_scale = qw::RowSplitScalePattern::Small;
        options.row_split_codes = qw::RowSplitCodePattern::Coordinate;
    }
    return qw::make_patterned_weight(qtype, n, k, seed, options);
}

// A BF16 control weight is a plain dense matrix, so the quantized fixture does not own it. Same
// contract as the fixture's patterned weights: every word is a pure function of the GLOBAL
// (row, column), so a block generated with an origin IS the parent's block. BOTH coordinates are
// hashed for the same reason the fixture's decorrelate_coordinates exists -- an affine rule is
// periodic at exactly the tp2 strides on either axis (here 2654435761*7168 = 0 mod 256 on rows and
// 40503*3072 = 0 mod 256 on columns), which would make the two shards byte-identical and parity
// vacuous.
std::uint16_t bf16_weight_word(std::int64_t row, std::int64_t column, std::uint32_t seed) {
    const auto hashed_row = static_cast<std::uint32_t>(
        qw::detail::mix64(static_cast<std::uint64_t>(row) + 0x9e37U) >> 24);
    const auto hashed_column = static_cast<std::uint32_t>(
        qw::detail::mix64(static_cast<std::uint64_t>(column) + 0x85ebU) >> 24);
    const auto mixed = hashed_row * 2654435761u + hashed_column * 40503u + seed;
    // A small symmetric value grid keeps the FP32 accumulation well inside BF16 range at K = 5120
    // while still exercising sign changes.
    return f32_to_bf16((static_cast<float>(mixed & 0xffu) - 127.5F) / 512.0F);
}

std::vector<std::uint16_t> bf16_weight_block(std::int32_t n, std::int32_t k, std::uint32_t seed,
                                             std::int32_t row_origin, std::int32_t column_origin) {
    std::vector<std::uint16_t> words(static_cast<std::size_t>(n) * k);
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t column = 0; column < k; ++column) {
            words[static_cast<std::size_t>(row) * k + column] =
                bf16_weight_word(row + row_origin, column + column_origin, seed);
        }
    }
    return words;
}

// Coordinates that straddle every structural seam a shard could get wrong: the NVFP4 32-row
// sub-block and 128-row tile interleave, the row-split 64/32-wide quantization groups, and the
// shard's own first and last index on both axes.
std::vector<std::int32_t> seam_samples(std::int32_t extent) {
    std::vector<std::int32_t> probes{0,   1,   31,  32,  33,  63,  64,          127,
                                     128, 129, 255, 256, 511, 512, extent / 2,  extent - 1};
    std::vector<std::int32_t> result;
    for (const std::int32_t probe : probes) {
        if (probe >= 0 && probe < extent &&
            std::find(result.begin(), result.end(), probe) == result.end()) {
            result.push_back(probe);
        }
    }
    return result;
}

// Proves that the shard tensor really is the parent's block, at the LOGICAL level, before any
// kernel runs. Both sides go through the fixture's own independent decoder.
int verify_shard_is_parent_block(const std::string& label, const qw::PackedWeight& parent,
                                 const qw::PackedWeight& shard, std::int32_t row_origin,
                                 std::int32_t column_origin) {
    for (const std::int32_t row : seam_samples(shard.weight.n)) {
        for (const std::int32_t column : seam_samples(shard.weight.k)) {
            const double got = qw::logical_weight_fp64(shard, row, column);
            const double expected =
                qw::logical_weight_fp64(parent, row + row_origin, column + column_origin);
            if (got != expected) {
                std::cerr << label << ": shard is not the parent's block at shard (" << row << ','
                          << column << ") -> parent (" << row + row_origin << ','
                          << column + column_origin << "): got " << got << " expected " << expected
                          << '\n';
                return 1;
            }
        }
    }
    return 0;
}

// The dense twin of verify_shard_is_parent_block: compares the WORDS the shard generator actually
// emitted against the parent's words at the mapped coordinate, so it is a real check on the two
// buffers rather than a restatement of the generating rule.
int verify_bf16_shard_is_parent_block(const std::string& label,
                                      const std::vector<std::uint16_t>& parent, std::int32_t k,
                                      const std::vector<std::uint16_t>& shard,
                                      std::int32_t shard_n, std::int32_t shard_k,
                                      std::int32_t row_origin, std::int32_t column_origin) {
    for (const std::int32_t row : seam_samples(shard_n)) {
        for (const std::int32_t column : seam_samples(shard_k)) {
            const std::uint16_t got = shard[static_cast<std::size_t>(row) * shard_k + column];
            const std::uint16_t expected =
                parent[static_cast<std::size_t>(row + row_origin) * k + (column + column_origin)];
            if (got != expected) {
                std::cerr << label << ": bf16 shard is not the parent's block at (" << row << ','
                          << column << ")\n";
                return 1;
            }
        }
    }
    return 0;
}

// The two shards must not be the same bytes.
//
// This is the mirror of the cross-rank output check in run_case(), and it is the more important of
// the two, because it fails EARLIER and for a reason no kernel can be blamed for. Both the NVFP4
// and the FP8 patterns are affine in the coordinate, and every registered tp2 boundary annihilates
// the relevant term on BOTH axes (rows 7168/8192/17408: 13s = 0 mod 16; columns 3072/8704:
// 7s = 0 mod 16 and 3*(s/16) = 0 mod 8). When that happens the two shards are byte-identical, so
// `verify_shard_is_parent_block` degenerates into a tautology and parity cannot tell a correct
// shard from a duplicate of the other half -- while every reported error stays at zero. Asserting
// distinctness is the only thing that makes the rest of this suite mean anything.
template <typename Bytes>
int verify_shards_are_distinct(const std::string& label, const Bytes& first, const Bytes& second) {
    if (first == second) {
        std::cerr << label
                  << ": the two shard payloads are byte-identical, so shard identity is untested "
                     "on this axis (the generator's pattern is degenerate at this stride)\n";
        return 1;
    }
    return 0;
}

// One rank's device-resident weight: the packed payload plus the Weight view over it.
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

DeviceWeight upload_bf16_weight(const std::vector<std::uint16_t>& words, std::int32_t n,
                                std::int32_t k) {
    DeviceWeight out;
    out.payload                = to_device(words);
    Weight& w                  = out.weight;
    w.payload                  = out.payload.p;
    w.payload_bytes            = out.payload.bytes;
    w.qtype                    = QType::BF16_CTRL;
    w.layout                   = QuantLayout::Contiguous;
    w.scale_dtype              = DType::BF16;
    w.qdata                    = out.payload.p;
    w.qhigh                    = nullptr;
    w.scales                   = nullptr;
    w.group_size               = 0;
    w.group                    = 0;
    w.ndim                     = 2;
    w.shape[0]                 = n;
    w.shape[1]                 = k;
    w.padded_shape[0]          = n;
    w.padded_shape[1]          = k;
    w.n                        = n;
    w.k                        = k;
    return out;
}

// Transient arena for one rank, sized at that rank's own (shard) problem. Sizing at the shard
// rather than the parent is part of the contract: a rank never stages the peer's activation.
std::size_t shard_workspace_bytes(QType qtype, std::int32_t n, std::int32_t k,
                                  ops::LinearPolicy policy, std::int32_t tokens) {
    return ops::linear_workspace_capacity_bytes(qtype, n, k, policy, tokens, tokens);
}

// Parity verdict, two independent legs.
//
// GROSS leg: the absolute difference of any element, against 2 BF16 ulp of the LARGEST output in
// the tensor. Stated against the tensor's scale rather than per element because a GEMM's rounding
// error is proportional to the magnitudes being accumulated, not to the magnitude that survives
// the accumulation: an output where the contributions nearly cancel is small for a reason that
// has nothing to do with how accurately it was computed. This is the same
// `gross_relative_to_max_reference` leg every per-format Linear conformance suite uses
// (tests/ops/linear/linear_test_common.cpp), which is what makes 2 ulp a comparable number here.
//
// L2 leg: the relative L2 of the whole difference, against the same 2 BF16 ulp. The gross leg
// alone cannot see a systematic bias that stays under the per-element bound everywhere; this one
// can. One ulp is deliberately NOT the bound: a split evaluation differs from the whole-K one by
// two extra BF16 storage roundings of quantities each comparable to the result, so a difference of
// about one ulp is the EXPECTED value, not the limit, and stating the limit there would be a
// coin-flip. (Measured: the largest relative L2 over the whole matrix below is 0.0040, i.e. 1.02
// ulp; the bound leaves 2x margin.)
//
// FP8's A8 activation-quantize kernel (fp8_a8.cu's fp8_a8_quantize_kernel)
// computes ONE scale per TOKEN over the entire K row (a global reduction), unlike NVFP4/W4A4's
// per-16-element-value GROUP scale, which is local and therefore identical whether it is computed
// over the whole row or just one shard's half of it. A row-parallel (K) split changes what "whole
// row" means for each rank's own activation quantize step -- rank r's scale is the max over ITS
// half of K, not the parent's max over the full K -- so the split evaluation is not simply "the
// same quantization plus two extra BF16 roundings" the way every other split route in this suite
// is; it is a structurally DIFFERENT quantization of the activation. This is invisible at A16Only
// (never quantizes the activation) and at every COLUMN-parallel case (K is never split, so the
// scale is identical on both sides -- see the bit-exact fp8_gdn_input column results below), but it
// measurably widens the row-parallel + AllowA8 comparison: observed rel_l2 up to 0.0208 (T=22,
// [5120,6144]->[5120,3072] via linear_add's own split test), well past 2 BF16 ulp. The bound below
// reuses this repository's OWN established FP8-A8 tolerance
// (tests/ops/linear_add/test_fp8.cpp's kA8Tolerance, the accepted A8-vs-FP64-oracle bound), which
// is the right comparable number for "how much error this route's own quantization can introduce."
constexpr ReductionCriterion kFp8A8RowSplitCriterion{/*relative_l2*/ 0.04, /*gross_absolute*/ kBf16Ulp,
                                                     /*gross_relative_to_max_reference*/ 0.06};

// The observed numbers are always printed, so a case that is currently bit-exact (max_abs = 0)
// cannot silently degrade into "merely inside tolerance" without that showing up in the log.
int compare(const std::string& label, const std::vector<double>& got,
            const std::vector<double>& expected, const ReductionCriterion& criterion) {
    const ReductionStats stats = compute_reduction_stats(got.data(), expected.data(),
                                                         static_cast<std::int64_t>(got.size()));
    std::cout << "  " << label << ": max_abs=" << stats.maximum_absolute_error
              << " rel_l2=" << stats.relative_l2 << " gross_limit="
              << gross_error_limit(stats, criterion) << '\n';
    return verify_reduction(label, got, expected, criterion);
}

// Per-case criterion selector: every route except FP8's row-parallel AllowA8 uses the blanket
// 2-ulp bound this whole suite is built on; that one route needs the wider, separately-justified
// bound above (see the comment on kFp8A8RowSplitCriterion).
ReductionCriterion criterion_for(const Case& test_case, ops::LinearPolicy policy) {
    constexpr ReductionCriterion kDefault{/*relative_l2*/ 2.0 * kBf16Ulp, /*gross_absolute*/ 0.0,
                                          /*gross_relative_to_max_reference*/ 2.0 * kBf16Ulp};
    if (test_case.qtype == QType::FP8_E4M3FN_ROW_BF16S && test_case.axis == SplitAxis::Row &&
        policy == ops::LinearPolicy::AllowA8) {
        return kFp8A8RowSplitCriterion;
    }
    return kDefault;
}

// ---------------------------------------------------------------------------------------------
// One case: build the whole weight and the two shards, run tp1 on device 0 and the split form on
// both, compare.
// ---------------------------------------------------------------------------------------------
int run_case(const Case& test_case, const ExecutionContext& ec, const ops::PeerEvents& events) {
    const bool column      = test_case.axis == SplitAxis::Column;
    const bool dense       = test_case.qtype == QType::BF16_CTRL;
    const std::int32_t n   = test_case.n;
    const std::int32_t k   = test_case.k;
    const std::int32_t sn  = column ? n / 2 : n; // shard output rows
    const std::int32_t sk  = column ? k : k / 2; // shard input rows
    const std::string head = std::string(test_case.label) + (column ? " column" : " row");

    std::cout << head << " [" << n << ',' << k << "] -> [" << sn << ',' << sk << "]\n";

    int failures = 0;

    // --- weights: the whole tensor and the two standalone shards, from one logical matrix -------
    std::vector<qw::PackedWeight> packed; // empty for the dense control
    std::vector<std::vector<std::uint16_t>> dense_words;
    if (dense) {
        dense_words.push_back(bf16_weight_block(n, k, test_case.seed, 0, 0));
        for (int rank = 0; rank < 2; ++rank) {
            const std::int32_t row_origin    = column ? rank * sn : 0;
            const std::int32_t column_origin = column ? 0 : rank * sk;
            dense_words.push_back(
                bf16_weight_block(sn, sk, test_case.seed, row_origin, column_origin));
            failures += verify_bf16_shard_is_parent_block(
                head + " shard " + std::to_string(rank), dense_words[0], k, dense_words.back(), sn,
                sk, row_origin, column_origin);
        }
        failures += verify_shards_are_distinct(head, dense_words[1], dense_words[2]);
    } else {
        packed.push_back(make_weight(test_case.qtype, n, k, test_case.seed, 0, 0));
        for (int rank = 0; rank < 2; ++rank) {
            const std::int32_t row_origin    = column ? rank * sn : 0;
            const std::int32_t column_origin = column ? 0 : rank * sk;
            packed.push_back(make_weight(test_case.qtype, sn, sk, test_case.seed, row_origin,
                                         column_origin));
            failures += verify_shard_is_parent_block(head + " shard " + std::to_string(rank),
                                                     packed[0], packed.back(), row_origin,
                                                     column_origin);
        }
        failures += verify_shards_are_distinct(head, packed[1].payload, packed[2].payload);
    }
    if (failures != 0) { return failures; }

    set_device(ec, 0);
    DeviceWeight full = dense ? upload_bf16_weight(dense_words[0], n, k) : upload_weight(packed[0]);
    std::array<DeviceWeight, 2> shard;
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        shard[static_cast<std::size_t>(rank)] =
            dense ? upload_bf16_weight(dense_words[static_cast<std::size_t>(rank) + 1], sn, sk)
                  : upload_weight(packed[static_cast<std::size_t>(rank) + 1]);
    }

    for (const std::int32_t tokens : test_case.tokens) {
        // --- activation: one logical [K,T] matrix, replicated (column) or split by rows (row) ---
        std::vector<float> activation(static_cast<std::size_t>(k) * tokens);
        fill_uniform(activation, test_case.seed * 31u + static_cast<std::uint32_t>(tokens), -1.0F,
                     1.0F);
        round_to_bf16(activation);

        set_device(ec, 0);
        DeviceBuffer full_x = to_device_bf16(activation);

        std::array<DeviceBuffer, 2> shard_x;
        for (int rank = 0; rank < 2; ++rank) {
            std::vector<float> block(static_cast<std::size_t>(sk) * tokens);
            for (std::int32_t token = 0; token < tokens; ++token) {
                const std::size_t source =
                    static_cast<std::size_t>(token) * k + (column ? 0 : static_cast<std::size_t>(rank) * sk);
                std::copy(activation.begin() + static_cast<std::ptrdiff_t>(source),
                          activation.begin() + static_cast<std::ptrdiff_t>(source + sk),
                          block.begin() + static_cast<std::ptrdiff_t>(token) * sk);
            }
            set_device(ec, rank);
            shard_x[static_cast<std::size_t>(rank)] = to_device_bf16(block);
        }

        for (const ops::LinearPolicy policy : test_case.policies) {
            const std::string label =
                head + " T=" + std::to_string(tokens) + " " + policy_name(policy);

            // --- (a) reference: the tp1 kernel, whole weight, device 0 ------------------------
            const std::size_t reference_elements = static_cast<std::size_t>(n) * tokens;
            set_device(ec, 0);
            GuardedDeviceBuffer reference(reference_elements * sizeof(std::uint16_t));
            reference.fill(0xff);
            DeviceArena reference_arena(
                std::max<std::size_t>(shard_workspace_bytes(test_case.qtype, n, k, policy, tokens),
                                      1));
            Tensor reference_x(full_x.p, DType::BF16, {k, tokens});
            Tensor reference_out(reference.data(), DType::BF16, {n, tokens});

            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            ops::linear(reference_x, full.weight, reference_out, policy, reference_arena,
                        ec.dev[0]->stream);
            cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
            failures += reference.verify_guards(label + " reference");
            const std::vector<double> expected =
                from_device_bf16(reference.data(), reference_elements);

            // --- (b) the split form -----------------------------------------------------------
            const std::size_t split_elements = static_cast<std::size_t>(sn) * tokens;
            const std::size_t workspace_bytes =
                std::max<std::size_t>(shard_workspace_bytes(test_case.qtype, sn, sk, policy, tokens), 1);

            // Every per-rank allocation is made with that rank's device current, so each buffer is
            // resident where the split form requires it.
            std::array<std::optional<GuardedDeviceBuffer>, 2> split_out;
            std::array<std::optional<DeviceBuffer>, 2> staging;
            std::array<std::optional<DeviceArena>, 2> arena;
            for (int rank = 0; rank < 2; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                set_device(ec, rank);
                split_out[slot].emplace(split_elements * sizeof(std::uint16_t));
                split_out[slot]->fill(0xff);
                arena[slot].emplace(workspace_bytes);
                staging[slot].emplace(split_elements * sizeof(std::uint16_t));
            }

            // For a column split shard_x[r] is the whole replicated activation on device r
            // (sk == k); for a row split it is that rank's own block of the activation rows.
            const std::array<Tensor, 2> x{Tensor(shard_x[0].p, DType::BF16, {sk, tokens}),
                                          Tensor(shard_x[1].p, DType::BF16, {sk, tokens})};
            const std::array<Weight, 2> w{shard[0].weight, shard[1].weight};
            const std::array<Tensor, 2> out{
                Tensor(split_out[0]->data(), DType::BF16, {sn, tokens}),
                Tensor(split_out[1]->data(), DType::BF16, {sn, tokens})};
            const std::array<ninfer::WorkspaceArena*, 2> workspace{&*arena[0], &*arena[1]};

            retire_staging(ec);
            if (column) {
                ops::linear_column_parallel(x, w, out, policy, workspace, ec);
            } else {
                const std::array<Tensor, 2> staging_view{
                    Tensor(staging[0]->p, DType::BF16, {sn, tokens}),
                    Tensor(staging[1]->p, DType::BF16, {sn, tokens})};
                ops::linear_row_parallel(x, w, out, staging_view, policy, workspace, ec, events);
            }
            synchronize_both(ec);

            // --- comparison --------------------------------------------------------------------
            const ReductionCriterion criterion = criterion_for(test_case, policy);
            std::array<std::vector<double>, 2> observed;
            for (int rank = 0; rank < 2; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                set_device(ec, rank);
                failures += split_out[slot]->verify_guards(label + " rank " + std::to_string(rank));
                observed[slot] = from_device_bf16(split_out[slot]->data(), split_elements);

                if (column) {
                    // Rank r owns output rows [r*sn, (r+1)*sn) of every token.
                    std::vector<double> block(split_elements);
                    for (std::int32_t token = 0; token < tokens; ++token) {
                        const std::size_t source = static_cast<std::size_t>(token) * n +
                                                   static_cast<std::size_t>(rank) * sn;
                        std::copy(expected.begin() + static_cast<std::ptrdiff_t>(source),
                                  expected.begin() + static_cast<std::ptrdiff_t>(source + sn),
                                  block.begin() + static_cast<std::ptrdiff_t>(token) * sn);
                    }
                    failures += compare(label + " rank " + std::to_string(rank), observed[slot],
                                        block, criterion);
                } else {
                    // Both ranks hold the identical summed result.
                    failures += compare(label + " rank " + std::to_string(rank), observed[slot],
                                        expected, criterion);
                }
            }

            // Cross-rank structural checks the per-rank comparison cannot make.
            if (column) {
                // The two ranks must own DIFFERENT output rows. Comparing each against its own
                // block would still pass if both ranks had computed rank 0's rows and the two
                // blocks happened to be close; requiring the outputs to differ closes that.
                if (observed[0] == observed[1]) {
                    std::cerr << label << ": both ranks produced identical output blocks, so the "
                                          "column split did not actually split\n";
                    ++failures;
                }
            } else {
                // allreduce_sum leaves the identical summed result on both ranks: exact, byte for
                // byte, because the local combine is the same computation on the same operands.
                if (observed[0] != observed[1]) {
                    std::cerr << label << ": the two ranks disagree after the all-reduce\n";
                    ++failures;
                }
            }
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------------------------
// Host-only registry probe. linear_workspace_capacity_bytes() runs each format's shape resolver
// for the interval's endpoints and throws for an unregistered problem, so it admits or rejects a
// shard extent without touching a device. Every registered shard extent is listed here, including
// the ones no GPU case above exercises (the vocabulary heads, whose parity is covered by
// tests/ops/test_output_head_split.cpp and whose weights are too large to generate per run).
// ---------------------------------------------------------------------------------------------
int verify_registry() {
    struct Entry {
        QType qtype;
        std::int32_t n;
        std::int32_t k;
        ops::LinearPolicy policy;
    };
    constexpr auto kA16 = ops::LinearPolicy::A16Only;
    constexpr auto kA8  = ops::LinearPolicy::AllowA8;
    constexpr auto kA4  = ops::LinearPolicy::AllowA4;

    const std::vector<Entry> entries{
        // NVFP4: column halves then row halves.
        {QType::NVFP4, 7168, 5120, kA16},    {QType::NVFP4, 7168, 5120, kA4},
        {QType::NVFP4, 8192, 5120, kA16},    {QType::NVFP4, 8192, 5120, kA4},
        {QType::NVFP4, 17408, 5120, kA16},   {QType::NVFP4, 17408, 5120, kA4},
        {QType::NVFP4, 5120, 3072, kA16},    {QType::NVFP4, 5120, 3072, kA4},
        {QType::NVFP4, 5120, 8704, kA16},    {QType::NVFP4, 5120, 8704, kA4},
        // FP8 vocabulary head half.
        {QType::FP8_E4M3FN_ROW_BF16S, 124160, 5120, kA16},
        {QType::FP8_E4M3FN_ROW_BF16S, 124160, 5120, kA8},
        {QType::FP8_E4M3FN_ROW_BF16S, 124160, 5120, kA4},
        // linear_add's row shards (o_proj/gdn_output 6144->3072, mlp/down 17408->8704)
        // and gdn_input_proj's column shard, all now wired into ops::linear's own kernel-level
        // dispatch too (fp8_gemv.cu/fp8_small_t.cu/fp8_a8.cu), plus a GPU parity case below.
        {QType::FP8_E4M3FN_ROW_BF16S, 5120, 3072, kA16},
        {QType::FP8_E4M3FN_ROW_BF16S, 5120, 3072, kA8},
        {QType::FP8_E4M3FN_ROW_BF16S, 5120, 8704, kA16},
        {QType::FP8_E4M3FN_ROW_BF16S, 5120, 8704, kA8},
        {QType::FP8_E4M3FN_ROW_BF16S, 8192, 5120, kA16},
        {QType::FP8_E4M3FN_ROW_BF16S, 8192, 5120, kA8},
        // Q4.
        {QType::Q4G64_F16S, 512, 5120, kA16},   {QType::Q4G64_F16S, 2048, 5120, kA16},
        {QType::Q4G64_F16S, 3072, 5120, kA16},  {QType::Q4G64_F16S, 3584, 5120, kA16},
        {QType::Q4G64_F16S, 17408, 5120, kA16}, {QType::Q4G64_F16S, 65536, 5120, kA16},
        {QType::Q4G64_F16S, 65536, 2048, kA16}, {QType::Q4G64_F16S, 17408, 5120, kA8},
        // Q5.
        {QType::Q5G64_F16S, 512, 5120, kA16},  {QType::Q5G64_F16S, 3072, 5120, kA16},
        {QType::Q5G64_F16S, 3584, 5120, kA16}, {QType::Q5G64_F16S, 5120, 3072, kA16},
        {QType::Q5G64_F16S, 5120, 8704, kA16}, {QType::Q5G64_F16S, 5120, 3072, kA8},
        // Q6 vocabulary head half.
        {QType::Q6G64_F16S, 124160, 5120, kA16}, {QType::Q6G64_F16S, 124160, 2048, kA16},
        // W8.
        {QType::W8G32_F16S, 512, 5120, kA16},    {QType::W8G32_F16S, 3072, 5120, kA16},
        {QType::W8G32_F16S, 7168, 5120, kA16},   {QType::W8G32_F16S, 17408, 5120, kA16},
        {QType::W8G32_F16S, 124160, 5120, kA16}, {QType::W8G32_F16S, 5120, 3072, kA16},
        {QType::W8G32_F16S, 5120, 5120, kA16},   {QType::W8G32_F16S, 5120, 8704, kA16},
        {QType::W8G32_F16S, 5120, 8704, kA8},
        // BF16 control.
        {QType::BF16_CTRL, 7168, 5120, kA16},    {QType::BF16_CTRL, 5120, 3072, kA16},
    };

    int failures = 0;
    for (const Entry& entry : entries) {
        for (const std::int32_t tokens : {1, 2, 48, 1024}) {
            try {
                (void)ops::linear_workspace_capacity_bytes(entry.qtype, entry.n, entry.k,
                                                           entry.policy, tokens, tokens);
            } catch (const std::exception& error) {
                std::cerr << "registry: shard [" << entry.n << ',' << entry.k << "] qtype "
                          << static_cast<int>(entry.qtype) << ' ' << policy_name(entry.policy)
                          << " T=" << tokens << " rejected: " << error.what() << '\n';
                ++failures;
            }
        }
    }

    // The tp1 path must be untouched: a shape that was never registered still is not.
    const std::vector<Entry> rejected{
        {QType::NVFP4, 5120, 5120, kA16},
        {QType::NVFP4, 3584, 5120, kA16},
        {QType::Q5G64_F16S, 4096, 5120, kA16},
        {QType::BF16_CTRL, 8192, 5120, kA16},
        // FP8's residual/gdn_input shards are not vocabulary problems, so AllowA4 (a policy FP8
        // never admits outside the vocabulary escape hatch) must still be rejected here.
        {QType::FP8_E4M3FN_ROW_BF16S, 5120, 3072, kA4},
        {QType::FP8_E4M3FN_ROW_BF16S, 8192, 5120, kA4},
    };
    for (const Entry& entry : rejected) {
        bool threw = false;
        try {
            (void)ops::linear_workspace_capacity_bytes(entry.qtype, entry.n, entry.k, entry.policy,
                                                       1, 1);
        } catch (const std::exception&) { threw = true; }
        if (!threw) {
            std::cerr << "registry: [" << entry.n << ',' << entry.k << "] qtype "
                      << static_cast<int>(entry.qtype) << " was admitted but must not be\n";
            ++failures;
        }
    }
    std::cout << (failures ? "FAIL" : "OK") << " registry: " << entries.size()
              << " shard extents admitted, " << rejected.size() << " non-problems still rejected\n";
    return failures;
}

// Rejection cases the split forms own: only the pair can see them.
int verify_split_rejections(const ExecutionContext& ec, const ops::PeerEvents& events) {
    int failures            = 0;
    const auto expect_throw = [&](const char* what, auto&& body) {
        try {
            body();
        } catch (const std::invalid_argument&) {
            return;
        } catch (const std::exception& error) {
            std::cerr << "split rejection " << what << ": wrong exception: " << error.what()
                      << '\n';
            ++failures;
            return;
        }
        std::cerr << "split rejection " << what << ": accepted an invalid pair\n";
        ++failures;
    };

    constexpr std::int32_t kN = 7168;
    constexpr std::int32_t kK = 5120;
    set_device(ec, 0);
    DeviceBuffer x0(static_cast<std::size_t>(kK) * 2 * sizeof(std::uint16_t));
    DeviceBuffer out0(static_cast<std::size_t>(kN) * 2 * sizeof(std::uint16_t));
    DeviceBuffer stage0(static_cast<std::size_t>(kN) * 2 * sizeof(std::uint16_t));
    set_device(ec, 1);
    DeviceBuffer x1(static_cast<std::size_t>(kK) * 2 * sizeof(std::uint16_t));
    DeviceBuffer out1(static_cast<std::size_t>(kN) * 2 * sizeof(std::uint16_t));
    DeviceBuffer stage1(static_cast<std::size_t>(kN) * 2 * sizeof(std::uint16_t));

    Weight fake{};
    fake.qtype = QType::NVFP4;
    fake.n     = kN;
    fake.k     = kK;

    // Mismatched token counts on the two ranks.
    expect_throw("token count", [&] {
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kK, 2}),
                                      Tensor(x1.p, DType::BF16, {kK, 1})};
        const std::array<Tensor, 2> out{Tensor(out0.p, DType::BF16, {kN, 2}),
                                        Tensor(out1.p, DType::BF16, {kN, 1})};
        ops::linear_column_parallel(x, {fake, fake}, out, ec);
    });

    // Column-parallel with disagreeing K.
    expect_throw("column K", [&] {
        Weight other = fake;
        other.k      = kK / 2;
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kK, 1}),
                                      Tensor(x1.p, DType::BF16, {kK / 2, 1})};
        const std::array<Tensor, 2> out{Tensor(out0.p, DType::BF16, {kN, 1}),
                                        Tensor(out1.p, DType::BF16, {kN, 1})};
        ops::linear_column_parallel(x, {fake, other}, out, ec);
    });

    // Row-parallel with disagreeing N.
    expect_throw("row N", [&] {
        Weight other = fake;
        other.n      = kN / 2;
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kK, 1}),
                                      Tensor(x1.p, DType::BF16, {kK, 1})};
        const std::array<Tensor, 2> out{Tensor(out0.p, DType::BF16, {kN, 1}),
                                        Tensor(out1.p, DType::BF16, {kN / 2, 1})};
        // Real, disjoint staging: aliasing it onto `out` would be a second, unrelated contract
        // violation and would leave it ambiguous which one the Op actually rejected.
        const std::array<Tensor, 2> staging{Tensor(stage0.p, DType::BF16, {kN, 1}),
                                            Tensor(stage1.p, DType::BF16, {kN / 2, 1})};
        ops::linear_row_parallel(x, {fake, other}, out, staging, ec, events);
    });

    // A single-device context is not a split context.
    expect_throw("tp1 context", [&] {
        const ExecutionContext single({0});
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kK, 1}),
                                      Tensor(x1.p, DType::BF16, {kK, 1})};
        const std::array<Tensor, 2> out{Tensor(out0.p, DType::BF16, {kN, 1}),
                                        Tensor(out1.p, DType::BF16, {kN, 1})};
        ops::linear_column_parallel(x, {fake, fake}, out, single);
    });

    std::cout << (failures ? "FAIL" : "OK") << " split rejections\n";
    return failures;
}

} // namespace

int main() {
    // The registry probe is pure host code -- linear_workspace_capacity_bytes only runs each
    // format's shape resolver -- so it runs BEFORE the device checks and reports a real failure
    // even on a machine that has to skip the parity cases. A shard extent silently dropped from a
    // dispatch table is exactly the regression a one-GPU CI box should still be able to catch.
    int failures = verify_registry();
    if (failures != 0) {
        std::cout << "FAIL linear split (registry)\n";
        return 1;
    }

    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: split linear parity requires two CUDA devices, found " << device_count
                  << '\n';
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

    failures += verify_split_rejections(ec, events);

    constexpr auto kA16 = ops::LinearPolicy::A16Only;
    constexpr auto kA4  = ops::LinearPolicy::AllowA4;
    constexpr auto kA8  = ops::LinearPolicy::AllowA8;

    const std::vector<Case> cases{
        // --- column-parallel: mlp/gate_up, attention and GDN input projections -----------------
        // Token counts are chosen to reach every branch the split path rewired, not just the
        // convenient ones:
        //   T=1            the decode edge (launch_nvfp4_decode);
        //   T=5/8/32/48    the small-T and MMA frontiers;
        //   T=128          launch_problem's `tokens <= 128` branch, whose kResidualGeometry test
        //                  is one of the two predicates now resolved through the tp1 PARENT
        //                  geometry rather than the shard's own extent;
        //   T=512          launch_problem's `tokens <= 512` branch, whose GdnInput kOutputRows
        //                  test is the other one -- it only ever fires for the GDN geometry, so
        //                  the gdn_input case below is the only one that proves the inheritance;
        //   T=1024         a multiple of 256, the sole route into the NVFP4 W4A4 TMA kernel. It
        //                  is applied to ALL FIVE shard geometries so every shard TMA descriptor
        //                  is actually built and launched, not just gate_up's and mlp_down's.
        {"nvfp4 gate_up", QType::NVFP4, SplitAxis::Column, 34816, 5120, 11u,
         {1, 5, 48, 128, 512, 1024}, {kA16, kA4}},
        {"nvfp4 attn_input", QType::NVFP4, SplitAxis::Column, 14336, 5120, 12u, {1, 32, 128, 1024},
         {kA16, kA4}},
        {"nvfp4 gdn_input", QType::NVFP4, SplitAxis::Column, 16384, 5120, 13u,
         {1, 8, 128, 512, 1024}, {kA16, kA4}},
        {"q4 gate_up", QType::Q4G64_F16S, SplitAxis::Column, 34816, 5120, 14u, {1, 17}, {kA16}},
        {"w8 attn_input", QType::W8G32_F16S, SplitAxis::Column, 14336, 5120, 15u, {1, 48}, {kA16}},
        {"bf16 attn_input", QType::BF16_CTRL, SplitAxis::Column, 14336, 5120, 16u, {1, 8}, {kA16}},
        // FP8's gdn_input column geometry, wired into ops::linear's own kernel-level
        // dispatch (fp8_gemv.cu/fp8_small_t.cu/fp8_a8.cu) so the generic Op can be exercised
        // end to end. T sweep: 1 decode edge; 2..10 small-T (kFp8LinearSmallTMax<GdnInput>=10);
        // 11 the A8 crossover (inherited from the tp1 GdnInput threshold); 32/128/1024 beyond it.
        {"fp8 gdn_input", QType::FP8_E4M3FN_ROW_BF16S, SplitAxis::Column, 16384, 5120, 17u,
         {1, 8, 10, 11, 32, 128, 1024}, {kA16, kA8}},

        // --- row-parallel: o_proj / gdn output (K=6144) and mlp/down (K=17408) -----------------
        // The residual geometries are the ones whose launch_problem branch is selected by the
        // inherited kResidualGeometry predicate, so T=128 and T=512 matter most here.
        {"nvfp4 o_proj", QType::NVFP4, SplitAxis::Row, 5120, 6144, 21u, {1, 8, 48, 128, 512, 1024},
         {kA16, kA4}},
        {"nvfp4 mlp_down", QType::NVFP4, SplitAxis::Row, 5120, 17408, 22u, {1, 48, 128, 512, 1024},
         {kA16, kA4}},
        {"q5 o_proj", QType::Q5G64_F16S, SplitAxis::Row, 5120, 6144, 23u, {1, 48}, {kA16}},
        {"w8 mlp_down", QType::W8G32_F16S, SplitAxis::Row, 5120, 17408, 24u, {1, 48}, {kA16}},
        {"bf16 o_proj", QType::BF16_CTRL, SplitAxis::Row, 5120, 6144, 25u, {1, 8}, {kA16}},
        // FP8's residual row shapes (o_proj/gdn_output), wired into ops::linear's own
        // kernel-level dispatch AND directly load-bearing for linear_add_row_parallel's plain
        // (residual-free) rank, which issues through ops::linear's dispatch_linear at exactly this
        // shard shape. T sweep: 1 decode edge; 8/24 small-T (kFp8LinearSmallTMax<Residual6144>=24);
        // 25 the A8 crossover; 48/128/1024 beyond it.
        {"fp8 o_proj", QType::FP8_E4M3FN_ROW_BF16S, SplitAxis::Row, 5120, 6144, 26u,
         {1, 8, 24, 25, 48, 128, 1024}, {kA16, kA8}},
        {"fp8 mlp_down", QType::FP8_E4M3FN_ROW_BF16S, SplitAxis::Row, 5120, 17408, 27u,
         {1, 8, 24, 25, 48, 128, 1024}, {kA16, kA8}},
    };

    for (const Case& test_case : cases) { failures += run_case(test_case, ec, events); }

    std::cout << (failures ? "FAIL" : "OK") << " linear split\n";
    return failures ? 1 : 0;
}
