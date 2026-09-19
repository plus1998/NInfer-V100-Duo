// Two-device parity suite for ops::linear_add_row_parallel (include/ninfer/ops/linear_add.h),
// the fused-residual sibling of test_linear_split.cpp. Read that file's header comment first;
// this one only documents what differs.
//
// Every case REQUIRES two CUDA devices driven from ONE process; with fewer than two visible
// devices the suite reports the repository's skip code (77) instead of failing.
//
// WHAT IS COMPARED.
//   (a) the tp1 fused kernel on device 0, over the WHOLE weight, updating a residual in place, and
//   (b) linear_add_row_parallel over the two K-shards + all-reduce, updating a REPLICATED residual
//       -- the same initial bytes on both devices, per the Op's own contract -- in place,
// and (b) is compared against (a). Weight shards come from the SAME translation-invariant
// generator test_linear_split.cpp uses (make_weight / bf16_weight_block below), so the two payloads
// meet only at the logical level, exactly as the tp2 loader's shards do.
//
// THE ONE THING THIS SUITE ADDS BEYOND test_linear_split.cpp: proving the residual is added
// EXACTLY ONCE. linear_add's tp1 kernels fuse the residual into the GEMM epilogue, but a
// row-parallel rank only ever holds a partial sum over its own K block, so linear_add_row_parallel
// folds the residual into rank 0's partial only (rank 1 computes the pure, residual-free GEMM half)
// and lets the one allreduce_sum that follows combine them -- see include/ninfer/ops/linear_add.h's
// design note for the full argument. A double-count bug (residual added on both ranks) would show
// up as approximately 2x the residual's own magnitude added into the result; a dropped-residual bug
// (added on neither rank) would show up as the residual's magnitude missing. Both are far outside
// the 2 BF16 ulp parity bound below, so the ordinary parity comparison already catches either -- no
// separate assertion is needed to make this suite mean what it says.
//
// TOLERANCE. 2 BF16 ulp of the largest output in the tensor, same criterion and same rationale as
// test_linear_split.cpp's row-parallel leg (two extra BF16 roundings from the split evaluation) --
// EXCEPT FP8's AllowA8 route, which needs a wider, separately-justified bound; see
// kFp8A8RowSplitCriterion's comment below (FP8's A8 activation scale is a whole-K-row reduction,
// not a local per-group one, so it is not shard-invariant under a K split).
#include "ninfer/ops/linear_add.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstddef>
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

constexpr double kBf16Ulp = 1.0 / 256.0;

struct Case {
    const char* label;
    QType qtype;
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

void retire_staging(const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
}

// Same construction as test_linear_split.cpp's make_weight: coordinate-origin generation with
// decorrelate_coordinates so the two row shards are provably distinct on both axes, not just the
// row axis (see that file's Critical-1 fix note -- the row-parallel shard boundaries here, K=3072
// and 8704, are exactly the ones that bit the row axis).
qw::PackedWeight make_weight(QType qtype, std::int32_t n, std::int32_t k, std::uint32_t seed,
                             std::int32_t column_origin) {
    qw::PatternedWeightOptions options;
    options.column_origin            = column_origin;
    options.decorrelate_coordinates  = true;
    if (qtype == QType::NVFP4) {
        options.weight_scale_divisor = 0.125F;
        options.input_scale_divisor  = 3.5F;
    } else {
        options.row_split_scale = qw::RowSplitScalePattern::Small;
        options.row_split_codes = qw::RowSplitCodePattern::Coordinate;
    }
    return qw::make_patterned_weight(qtype, n, k, seed, options);
}

std::uint16_t bf16_weight_word(std::int64_t row, std::int64_t column, std::uint32_t seed) {
    const auto hashed_row = static_cast<std::uint32_t>(
        qw::detail::mix64(static_cast<std::uint64_t>(row) + 0x9e37U) >> 24);
    const auto hashed_column = static_cast<std::uint32_t>(
        qw::detail::mix64(static_cast<std::uint64_t>(column) + 0x85ebU) >> 24);
    const auto mixed = hashed_row * 2654435761u + hashed_column * 40503u + seed;
    return f32_to_bf16((static_cast<float>(mixed & 0xffu) - 127.5F) / 512.0F);
}

std::vector<std::uint16_t> bf16_weight_block(std::int32_t n, std::int32_t k, std::uint32_t seed,
                                             std::int32_t column_origin) {
    std::vector<std::uint16_t> words(static_cast<std::size_t>(n) * k);
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t column = 0; column < k; ++column) {
            words[static_cast<std::size_t>(row) * k + column] =
                bf16_weight_word(row, column + column_origin, seed);
        }
    }
    return words;
}

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

int verify_shard_is_parent_block(const std::string& label, const qw::PackedWeight& parent,
                                 const qw::PackedWeight& shard, std::int32_t column_origin) {
    for (const std::int32_t row : seam_samples(shard.weight.n)) {
        for (const std::int32_t column : seam_samples(shard.weight.k)) {
            const double got      = qw::logical_weight_fp64(shard, row, column);
            const double expected = qw::logical_weight_fp64(parent, row, column + column_origin);
            if (got != expected) {
                std::cerr << label << ": shard is not the parent's block at shard (" << row << ','
                          << column << ") -> parent (" << row << ',' << column + column_origin
                          << "): got " << got << " expected " << expected << '\n';
                return 1;
            }
        }
    }
    return 0;
}

int verify_bf16_shard_is_parent_block(const std::string& label,
                                      const std::vector<std::uint16_t>& parent, std::int32_t k,
                                      const std::vector<std::uint16_t>& shard, std::int32_t shard_n,
                                      std::int32_t shard_k, std::int32_t column_origin) {
    for (const std::int32_t row : seam_samples(shard_n)) {
        for (const std::int32_t column : seam_samples(shard_k)) {
            const std::uint16_t got = shard[static_cast<std::size_t>(row) * shard_k + column];
            const std::uint16_t expected =
                parent[static_cast<std::size_t>(row) * k + (column + column_origin)];
            if (got != expected) {
                std::cerr << label << ": bf16 shard is not the parent's block at (" << row << ','
                          << column << ")\n";
                return 1;
            }
        }
    }
    return 0;
}

template <typename Bytes>
int verify_shards_are_distinct(const std::string& label, const Bytes& first, const Bytes& second) {
    if (first == second) {
        std::cerr << label
                  << ": the two shard payloads are byte-identical, so shard identity is untested\n";
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

DeviceWeight upload_bf16_weight(const std::vector<std::uint16_t>& words, std::int32_t n,
                                std::int32_t k) {
    DeviceWeight out;
    out.payload                = to_device(words);
    Weight& w                  = out.weight;
    w.payload                  = out.payload.p;
    w.payload_bytes            = out.payload.bytes;
    w.qtype                    = QType::BF16_CTRL;
    w.layout                   = QuantLayout::Contiguous;
    w.scale_dtype               = DType::BF16;
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

// linear_add's own workspace query: BF16's shard half never reaches linear_add's family (it is
// composed from the already tp2-capable plain linear() + residual_add() -- see
// include/ninfer/ops/linear_add.h), so it needs no workspace at either shape and querying
// linear_add_workspace_capacity_bytes at the shard shape for BF16 would reject a shape linear_add's
// OWN registry was never widened for (deliberately -- see that header's design note).
std::size_t add_workspace_bytes(QType qtype, std::int32_t n, std::int32_t k,
                                ops::LinearPolicy policy, std::int32_t tokens) {
    if (qtype == QType::BF16_CTRL) { return 0; }
    return ops::linear_add_workspace_capacity_bytes(qtype, n, k, policy, tokens, tokens);
}

// Host-side BF16 words for a residual, so it can seed a GuardedDeviceBuffer with copy_from_host
// (which requires a HOST source, unlike to_device_bf16's own device-resident result).
std::vector<std::uint16_t> to_bf16_words(const std::vector<float>& h) {
    std::vector<std::uint16_t> words(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) { words[i] = f32_to_bf16(h[i]); }
    return words;
}

// See test_linear_split.cpp's identical comment on kFp8A8RowSplitCriterion for the full
// argument: FP8's A8 activation-quantize step computes ONE scale per token over the
// WHOLE K row, unlike NVFP4/Q5's per-group scale, which is local and hence split-invariant. Every
// linear_add case is row-parallel (K-split) by construction, so FP8's AllowA8 route is not simply
// "the same quantization plus two extra BF16 roundings" the way the rest of this suite is -- each
// rank's own K-half produces a genuinely different activation quantization than the tp1 kernel's
// whole-row one. Observed rel_l2 up to 0.0208 (T=22, [5120,6144]->[5120,3072]), well past 2 BF16
// ulp. The bound below reuses this repository's own established FP8-A8 tolerance
// (tests/ops/linear_add/test_fp8.cpp's kA8Tolerance).
constexpr ReductionCriterion kFp8A8RowSplitCriterion{/*relative_l2*/ 0.04, /*gross_absolute*/ kBf16Ulp,
                                                     /*gross_relative_to_max_reference*/ 0.06};

int compare(const std::string& label, const std::vector<double>& got,
            const std::vector<double>& expected, const ReductionCriterion& criterion) {
    const ReductionStats stats = compute_reduction_stats(got.data(), expected.data(),
                                                         static_cast<std::int64_t>(got.size()));
    std::cout << "  " << label << ": max_abs=" << stats.maximum_absolute_error
              << " rel_l2=" << stats.relative_l2 << " gross_limit="
              << gross_error_limit(stats, criterion) << '\n';
    return verify_reduction(label, got, expected, criterion);
}

// Every route except FP8's AllowA8 (linear_add is always row-parallel) uses the blanket 2-ulp
// bound; see kFp8A8RowSplitCriterion's comment for why that one route needs a wider bound.
ReductionCriterion criterion_for(const Case& test_case, ops::LinearPolicy policy) {
    constexpr ReductionCriterion kDefault{/*relative_l2*/ 2.0 * kBf16Ulp, /*gross_absolute*/ 0.0,
                                          /*gross_relative_to_max_reference*/ 2.0 * kBf16Ulp};
    if (test_case.qtype == QType::FP8_E4M3FN_ROW_BF16S && policy == ops::LinearPolicy::AllowA8) {
        return kFp8A8RowSplitCriterion;
    }
    return kDefault;
}

// ---------------------------------------------------------------------------------------------
int run_case(const Case& test_case, const ExecutionContext& ec, const ops::PeerEvents& events) {
    const bool dense      = test_case.qtype == QType::BF16_CTRL;
    const std::int32_t n  = test_case.n;
    const std::int32_t k  = test_case.k;
    const std::int32_t sk = k / 2;
    const std::string head = test_case.label;

    std::cout << head << " [" << n << ',' << k << "] -> [" << n << ',' << sk << "] x2\n";

    int failures = 0;

    // --- weights: the whole tensor and the two standalone K-shards, from one logical matrix -----
    std::vector<qw::PackedWeight> packed;
    std::vector<std::vector<std::uint16_t>> dense_words;
    if (dense) {
        dense_words.push_back(bf16_weight_block(n, k, test_case.seed, 0));
        for (int rank = 0; rank < 2; ++rank) {
            const std::int32_t column_origin = rank * sk;
            dense_words.push_back(bf16_weight_block(n, sk, test_case.seed, column_origin));
            failures += verify_bf16_shard_is_parent_block(
                head + " shard " + std::to_string(rank), dense_words[0], k, dense_words.back(), n,
                sk, column_origin);
        }
        failures += verify_shards_are_distinct(head, dense_words[1], dense_words[2]);
    } else {
        packed.push_back(make_weight(test_case.qtype, n, k, test_case.seed, 0));
        for (int rank = 0; rank < 2; ++rank) {
            const std::int32_t column_origin = rank * sk;
            packed.push_back(make_weight(test_case.qtype, n, sk, test_case.seed, column_origin));
            failures += verify_shard_is_parent_block(head + " shard " + std::to_string(rank),
                                                     packed[0], packed.back(), column_origin);
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
            dense ? upload_bf16_weight(dense_words[static_cast<std::size_t>(rank) + 1], n, sk)
                  : upload_weight(packed[static_cast<std::size_t>(rank) + 1]);
    }

    for (const std::int32_t tokens : test_case.tokens) {
        // --- activation: one logical [K,T] matrix, split by rows into the two K-shards ----------
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
                    static_cast<std::size_t>(token) * k + static_cast<std::size_t>(rank) * sk;
                std::copy(activation.begin() + static_cast<std::ptrdiff_t>(source),
                          activation.begin() + static_cast<std::ptrdiff_t>(source + sk),
                          block.begin() + static_cast<std::ptrdiff_t>(token) * sk);
            }
            set_device(ec, rank);
            shard_x[static_cast<std::size_t>(rank)] = to_device_bf16(block);
        }

        // --- the INITIAL residual: one logical [N,T] tensor, REPLICATED (same bytes both sides,
        // exactly what linear_add_row_parallel's contract requires the caller to hand it) --------
        std::vector<float> residual0(static_cast<std::size_t>(n) * tokens);
        fill_uniform(residual0, test_case.seed * 97u + static_cast<std::uint32_t>(tokens) + 1u,
                    -2.0F, 2.0F);
        round_to_bf16(residual0);
        const std::vector<std::uint16_t> residual0_words = to_bf16_words(residual0);

        for (const ops::LinearPolicy policy : test_case.policies) {
            const std::string label = head + " T=" + std::to_string(tokens) + " " +
                                      policy_name(policy);

            // --- (a) reference: the tp1 fused kernel, whole weight, device 0 -------------------
            const std::size_t elements = static_cast<std::size_t>(n) * tokens;
            set_device(ec, 0);
            GuardedDeviceBuffer reference(elements * sizeof(std::uint16_t));
            reference.copy_from_host(residual0_words.data(), elements * sizeof(std::uint16_t));
            DeviceArena reference_arena(
                std::max<std::size_t>(add_workspace_bytes(test_case.qtype, n, k, policy, tokens), 1));
            Tensor reference_x(full_x.p, DType::BF16, {k, tokens});
            Tensor reference_residual(reference.data(), DType::BF16, {n, tokens});

            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            ops::linear_add(reference_x, full.weight, reference_residual, policy, reference_arena,
                            ec.dev[0]->stream);
            cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
            failures += reference.verify_guards(label + " reference");
            const std::vector<double> expected = from_device_bf16(reference.data(), elements);

            // --- (b) the split form: both ranks start from the SAME replicated residual --------
            const std::size_t workspace_bytes = std::max<std::size_t>(
                add_workspace_bytes(test_case.qtype, n, sk, policy, tokens), 1);

            std::array<std::optional<GuardedDeviceBuffer>, 2> split_residual;
            std::array<std::optional<DeviceBuffer>, 2> staging;
            std::array<std::optional<DeviceArena>, 2> arena;
            for (int rank = 0; rank < 2; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                set_device(ec, rank);
                split_residual[slot].emplace(elements * sizeof(std::uint16_t));
                split_residual[slot]->copy_from_host(residual0_words.data(),
                                                     elements * sizeof(std::uint16_t));
                arena[slot].emplace(workspace_bytes);
                staging[slot].emplace(elements * sizeof(std::uint16_t));
            }

            const std::array<Tensor, 2> x{Tensor(shard_x[0].p, DType::BF16, {sk, tokens}),
                                          Tensor(shard_x[1].p, DType::BF16, {sk, tokens})};
            const std::array<Weight, 2> w{shard[0].weight, shard[1].weight};
            const std::array<Tensor, 2> residual{
                Tensor(split_residual[0]->data(), DType::BF16, {n, tokens}),
                Tensor(split_residual[1]->data(), DType::BF16, {n, tokens})};
            const std::array<Tensor, 2> staging_view{
                Tensor(staging[0]->p, DType::BF16, {n, tokens}),
                Tensor(staging[1]->p, DType::BF16, {n, tokens})};
            const std::array<ninfer::WorkspaceArena*, 2> workspace{&*arena[0], &*arena[1]};

            retire_staging(ec);
            ops::linear_add_row_parallel(x, w, residual, staging_view, policy, workspace, ec,
                                         events);
            synchronize_both(ec);

            // --- comparison ----------------------------------------------------------------------
            const ReductionCriterion criterion = criterion_for(test_case, policy);
            std::array<std::vector<double>, 2> observed;
            for (int rank = 0; rank < 2; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                set_device(ec, rank);
                failures +=
                    split_residual[slot]->verify_guards(label + " rank " + std::to_string(rank));
                observed[slot] = from_device_bf16(split_residual[slot]->data(), elements);
                failures += compare(label + " rank " + std::to_string(rank), observed[slot],
                                    expected, criterion);
            }

            // allreduce_sum leaves the identical summed result on both ranks: exact, byte for byte.
            if (observed[0] != observed[1]) {
                std::cerr << label << ": the two ranks disagree after the all-reduce\n";
                ++failures;
            }
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------------------------
// Host-only registry probe, mirroring test_linear_split.cpp's verify_registry(): pure host code,
// runs before the device checks, so a shard extent silently dropped from linear_add's dispatch
// widening is caught even on a one-GPU box.
// ---------------------------------------------------------------------------------------------
int verify_registry() {
    struct Entry {
        QType qtype;
        std::int32_t n;
        std::int32_t k;
        ops::LinearPolicy policy;
    };
    constexpr auto kA16 = ops::LinearPolicy::A16Only;
    constexpr auto kA4  = ops::LinearPolicy::AllowA4;
    constexpr auto kA8  = ops::LinearPolicy::AllowA8;

    const std::vector<Entry> entries{
        {QType::NVFP4, 5120, 3072, kA16},   {QType::NVFP4, 5120, 3072, kA4},
        {QType::NVFP4, 5120, 8704, kA16},   {QType::NVFP4, 5120, 8704, kA4},
        {QType::Q5G64_F16S, 5120, 3072, kA16}, {QType::Q5G64_F16S, 5120, 8704, kA16},
        // FP8's own row-parallel shards.
        {QType::FP8_E4M3FN_ROW_BF16S, 5120, 3072, kA16}, {QType::FP8_E4M3FN_ROW_BF16S, 5120, 3072, kA8},
        {QType::FP8_E4M3FN_ROW_BF16S, 5120, 8704, kA16}, {QType::FP8_E4M3FN_ROW_BF16S, 5120, 8704, kA8},
    };

    int failures = 0;
    for (const Entry& entry : entries) {
        for (const std::int32_t tokens : {1, 2, 48, 1024}) {
            try {
                (void)ops::linear_add_workspace_capacity_bytes(entry.qtype, entry.n, entry.k,
                                                               entry.policy, tokens, tokens);
            } catch (const std::exception& error) {
                std::cerr << "registry: shard [" << entry.n << ',' << entry.k << "] qtype "
                          << static_cast<int>(entry.qtype) << ' ' << policy_name(entry.policy)
                          << " T=" << tokens << " rejected: " << error.what() << '\n';
                ++failures;
            }
        }
    }

    // Never registered (out of scope: W8's own residual family is a different, 2048-wide shape not
    // split by the qwen3_6_27b ShardPlan) and must stay rejected.
    const std::vector<Entry> rejected{
        {QType::W8G32_F16S, 2048, 2048, kA16},
        {QType::NVFP4, 5120, 4096, kA16},
        // FP8's residual shapes are not vocabulary problems, so AllowA4 must still be rejected.
        {QType::FP8_E4M3FN_ROW_BF16S, 5120, 3072, kA4},
    };
    for (const Entry& entry : rejected) {
        bool threw = false;
        try {
            (void)ops::linear_add_workspace_capacity_bytes(entry.qtype, entry.n, entry.k,
                                                           entry.policy, 1, 1);
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

// Rejection cases only the pair can see, mirroring test_linear_split.cpp's verify_split_rejections.
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

    constexpr std::int32_t kN = 5120;
    constexpr std::int32_t kK = 3072;
    set_device(ec, 0);
    DeviceBuffer x0(static_cast<std::size_t>(kK) * 2 * sizeof(std::uint16_t));
    DeviceBuffer r0(static_cast<std::size_t>(kN) * 2 * sizeof(std::uint16_t));
    DeviceBuffer stage0(static_cast<std::size_t>(kN) * 2 * sizeof(std::uint16_t));
    set_device(ec, 1);
    DeviceBuffer x1(static_cast<std::size_t>(kK) * 2 * sizeof(std::uint16_t));
    DeviceBuffer r1(static_cast<std::size_t>(kN) * 2 * sizeof(std::uint16_t));
    DeviceBuffer stage1(static_cast<std::size_t>(kN) * 2 * sizeof(std::uint16_t));

    Weight fake{};
    fake.qtype = QType::NVFP4;
    fake.n     = kN;
    fake.k     = kK;

    expect_throw("token count", [&] {
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kK, 2}),
                                      Tensor(x1.p, DType::BF16, {kK, 1})};
        const std::array<Tensor, 2> r{Tensor(r0.p, DType::BF16, {kN, 2}),
                                      Tensor(r1.p, DType::BF16, {kN, 1})};
        const std::array<Tensor, 2> staging{Tensor(stage0.p, DType::BF16, {kN, 2}),
                                            Tensor(stage1.p, DType::BF16, {kN, 1})};
        ops::linear_add_row_parallel(x, {fake, fake}, r, staging, ec, events);
    });

    expect_throw("row N", [&] {
        Weight other = fake;
        other.n      = kN / 2;
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kK, 1}),
                                      Tensor(x1.p, DType::BF16, {kK, 1})};
        const std::array<Tensor, 2> r{Tensor(r0.p, DType::BF16, {kN, 1}),
                                      Tensor(r1.p, DType::BF16, {kN / 2, 1})};
        const std::array<Tensor, 2> staging{Tensor(stage0.p, DType::BF16, {kN, 1}),
                                            Tensor(stage1.p, DType::BF16, {kN / 2, 1})};
        ops::linear_add_row_parallel(x, {fake, other}, r, staging, ec, events);
    });

    expect_throw("tp1 context", [&] {
        const ExecutionContext single({0});
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kK, 1}),
                                      Tensor(x1.p, DType::BF16, {kK, 1})};
        const std::array<Tensor, 2> r{Tensor(r0.p, DType::BF16, {kN, 1}),
                                      Tensor(r1.p, DType::BF16, {kN, 1})};
        const std::array<Tensor, 2> staging{Tensor(stage0.p, DType::BF16, {kN, 1}),
                                            Tensor(stage1.p, DType::BF16, {kN, 1})};
        // require_split_context rejects `single` before events is ever touched, so the outer
        // (valid) events object is fine to reuse here.
        ops::linear_add_row_parallel(x, {fake, fake}, r, staging, single, events);
    });

    expect_throw("unsupported format", [&] {
        Weight w8 = fake;
        w8.qtype  = QType::W8G32_F16S;
        w8.n      = kN;
        w8.k      = kK;
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kK, 1}),
                                      Tensor(x1.p, DType::BF16, {kK, 1})};
        const std::array<Tensor, 2> r{Tensor(r0.p, DType::BF16, {kN, 1}),
                                      Tensor(r1.p, DType::BF16, {kN, 1})};
        const std::array<Tensor, 2> staging{Tensor(stage0.p, DType::BF16, {kN, 1}),
                                            Tensor(stage1.p, DType::BF16, {kN, 1})};
        ops::linear_add_row_parallel(x, {w8, w8}, r, staging, ec, events);
    });

    std::cout << (failures ? "FAIL" : "OK") << " split rejections\n";
    return failures;
}

} // namespace

int main() {
    int failures = verify_registry();
    if (failures != 0) {
        std::cout << "FAIL linear_add split (registry)\n";
        return 1;
    }

    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: split linear_add parity requires two CUDA devices, found "
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

    failures += verify_split_rejections(ec, events);

    constexpr auto kA16 = ops::LinearPolicy::A16Only;
    constexpr auto kA4  = ops::LinearPolicy::AllowA4;
    constexpr auto kA8  = ops::LinearPolicy::AllowA8;

    // Real model shapes: attention/output & gdn/output share [5120,6144]; mlp/down is [5120,17408].
    // T sweep: 1 (decode edge), 5/8/48 (small-T/MMA), 128/384/512 (W4A4 schedule-threshold seams in
    // nvfp4_linear_add_w4a4.cu's launch_problem), 1024 (multiple of 256 -- the sole route into the
    // NVFP4 W4A4 TMA kernel, exercising the new launch_nvfp4_w4a4_tma_linear_add shard cases).
    const std::vector<Case> cases{
        {"nvfp4 o_proj", QType::NVFP4, 5120, 6144, 31u, {1, 5, 8, 48, 128, 384, 512, 1024},
         {kA16, kA4}},
        {"nvfp4 mlp_down", QType::NVFP4, 5120, 17408, 32u, {1, 8, 48, 128, 512, 1024}, {kA16, kA4}},
        {"q5 o_proj", QType::Q5G64_F16S, 5120, 6144, 33u, {1, 5, 8, 48}, {kA16}},
        {"q5 mlp_down", QType::Q5G64_F16S, 5120, 17408, 34u, {1, 8, 48}, {kA16}},
        {"bf16 o_proj", QType::BF16_CTRL, 5120, 6144, 35u, {1, 5, 8, 48}, {kA16}},
        // FP8's own residual shapes, wired as a TRUE split: src/ops/linear_add/fp8/* instantiates
        // the SAME launch<Geometry> templates at the shard's Tp2Row geometries -- no new kernel,
        // same as NVFP4/Q5. T sweep:
        // 1 decode edge; 5/8/21 small-T; 22 the o_proj A8 crossover (linear_add's own, distinct
        // from ops::linear's generic 25); 25 the mlp_down A8 crossover; 48/128/1024 beyond both.
        {"fp8 o_proj", QType::FP8_E4M3FN_ROW_BF16S, 5120, 6144, 36u,
         {1, 5, 8, 21, 22, 48, 128, 1024}, {kA16, kA8}},
        {"fp8 mlp_down", QType::FP8_E4M3FN_ROW_BF16S, 5120, 17408, 37u,
         {1, 8, 24, 25, 48, 128, 1024}, {kA16, kA8}},
    };

    for (const Case& test_case : cases) { failures += run_case(test_case, ec, events); }

    std::cout << (failures ? "FAIL" : "OK") << " linear_add split\n";
    return failures ? 1 : 0;
}
