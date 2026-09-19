// Two-device parity suite for ops::linear_swiglu_column_parallel
// (include/ninfer/ops/linear_swiglu.h) -- the swiglu-family sibling of test_linear_split.cpp
// (column/row) and test_linear_add_split.cpp (row). Read test_linear_split.cpp's header comment
// first; this one documents what differs.
//
// Every case REQUIRES two CUDA devices driven from ONE process; with fewer than two visible
// devices the suite reports the repository's skip code (77) instead of failing.
//
// THE REAL CALL GRAPH (why this suite is column-only). Variant::post_mixer
// (src/targets/qwen3_6_27b/impl/variant.cpp) is
//
//   ops::linear_swiglu(hidden, weights.gate_up, activation, ...);   // gate_up projection + SiLU
//   ops::linear_add(activation, weights.down, residual, ...);       // a SEPARATE op
//
// so linear_swiglu's own contract stops at `activation`; down/residual/all-reduce belong to
// linear_add_row_parallel(), whose mlp/down [5120,17408] row shards are registered and covered by
// test_linear_add_split.cpp.
// This suite's PRIMARY cases therefore test only the column-parallel gate_up+SiLU split. A second
// group of cases (run_pipeline_case) additionally chains the split gate_up output straight into
// linear_add_row_parallel(), to prove end to end that the two Ops compose exactly the way
// Variant::post_mixer calls them -- not a substitute for the primary cases, a demonstration on top.
//
// THE SHARD IS NOT A SINGLE CONTIGUOUS BLOCK. bindings.cpp's ShardPlan for `mlp/gate_up`
// (`ends("mlp/gate_up")`) appends TWO independent column blocks -- gate [0,17408) and up
// [17408,34816) of the parent -- each split by tp on its own, so rank r's shard weight is the
// CONCATENATION of gate rows [r*8704,(r+1)*8704) and up rows [17408+r*8704,17408+(r+1)*8704): a
// standalone [17408,5120] tensor in the SAME "gate rows [0,M) then up rows [M,2M)" layout
// linear_swiglu() itself assumes, at M=8704. A single-origin generator call (what
// test_linear_split.cpp uses for the OPAQUE-to-ops::linear geometry test of this same shape) would
// silently test a DIFFERENT, wrong operation here -- it would not carry the up half at all. See
// concat_row_blocks()/make_gate_up_shard() below for how the two blocks are produced and spliced,
// and verify_gate_up_shard_is_parent_blocks() for how the splice is proven correct against the
// SAME independent decoder test_linear_split.cpp uses, before any kernel runs.
//
// TWO FORMATS, two different mechanisms (see include/ninfer/ops/linear_swiglu.h's design note):
//   NVFP4  -- a TRUE split: the SAME kernel template, instantiated at the shard's halved N.
//   Q4G64_F16S -- COMPOSED from the already tp2-shard-capable ops::linear() + silu_mul(), because
//     Q4's own linear_swiglu kernels are compile-time-exact at the tp1 shape only.
//
// TOLERANCE. 2 BF16 ulp of the largest output in the tensor (gross_relative_to_max_reference) plus
// a matching relative-L2 leg, exactly test_linear_split.cpp's criterion and rationale.
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

// The one registered gate_up profile both formats share: parent [34816,5120], shard [17408,5120],
// M=8704 (see linear_swiglu.h's doc comment: "gate rows [0,17408) precede up rows [17408,34816)").
constexpr std::int32_t kParentGateUpRows = 34816;
constexpr std::int32_t kInputRows        = 5120;
constexpr std::int32_t kParentHalf       = kParentGateUpRows / 2; // 17408: gate/up boundary
constexpr std::int32_t kShardHalf        = kParentHalf / 2;       // 8704: per-rank M

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

// See test_linear_split.cpp's own retire_staging: DeviceContext::stream does not implicitly
// synchronize with the legacy default stream plain cudaMemcpy/cudaMemset land on.
void retire_staging(const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
}

// Single-origin generation for one contiguous row block, same construction as
// test_linear_split.cpp's make_weight -- decorrelate_coordinates on, so translation-invariant
// blocks separated by a registered tp2 stride are provably distinct rather than accidentally
// periodic (the vacuity trap that suite's own review pass found -- see its header comment).
qw::PackedWeight make_block(QType qtype, std::int32_t n, std::int32_t k, std::uint32_t seed,
                            std::int32_t row_origin) {
    qw::PatternedWeightOptions options;
    options.row_origin              = row_origin;
    options.decorrelate_coordinates = true;
    if (qtype == QType::NVFP4) {
        options.weight_scale_divisor = 0.125F;
        options.input_scale_divisor  = 3.5F;
    } else {
        options.row_split_scale = qw::RowSplitScalePattern::Small;
        options.row_split_codes = qw::RowSplitCodePattern::Coordinate;
    }
    return qw::make_patterned_weight(qtype, n, k, seed, options);
}

// Splices two independently generated, same-(qtype,k) row blocks into one standalone tensor whose
// logical rows are [top's rows][bottom's rows], in that order. Valid whenever every plane's
// byte address is affine in the row index with no INTERNAL padding within one block (true of
// every layout this fixture produces: NVFP4's code plane is `row*k/2 + col/2`; its scale plane is
// tile-major in `row/128` with the within-tile terms depending only on `row%128`, so a block whose
// OWN row count is an exact multiple of 128 places its bytes at exactly the offset a genuine
// n=top+bottom generation would use; RowSplit's code/high/scale planes are all `row*bytes_per_row`
// with no tile complication at all). NVFP4 additionally requires `top.weight.n % 128 == 0` so the
// scale-tile argument above applies; the shard halves used here (8704) satisfy it, since
// the SAME 128-row alignment is required for the shard to be a legal ShardPlan boundary in the
// first place (bindings.h's `is_column_parallel_boundary_valid`).
qw::PackedWeight concat_row_blocks(const qw::PackedWeight& top, const qw::PackedWeight& bottom) {
    if (top.weight.qtype != bottom.weight.qtype || top.weight.k != bottom.weight.k ||
        top.weight.layout != bottom.weight.layout) {
        throw std::invalid_argument("concat_row_blocks: mismatched qtype/layout/k");
    }
    if (top.weight.qtype == QType::NVFP4 && (top.weight.n % 128) != 0) {
        throw std::invalid_argument("concat_row_blocks: NVFP4 top block must be a 128-row tile "
                                    "multiple");
    }
    const auto align256 = [](std::uint64_t bytes) -> std::uint64_t {
        return (bytes + 255U) & ~std::uint64_t{255};
    };

    qw::PackedWeight combined;
    combined.weight                 = top.weight;
    combined.weight.n               = top.weight.n + bottom.weight.n;
    combined.weight.shape[0]        = combined.weight.n;
    combined.weight.padded_shape[0] = combined.weight.n;
    // FP8 fixture requirement, shared verbatim with test_attn_input_proj_split.cpp and
    // test_gdn_projections_split.cpp, which carry their own copies of this splice: FP8's
    // row-scale-v1 layout denormalizes `n` into scale_ne[0]/scale_nb[1..3] (one BF16 word per
    // output row). `combined.weight` above is a byte-copy of `top.weight`, sized for top's own
    // (smaller) n, so those four fields must be re-derived from the COMBINED n or the kernel reads
    // every row beyond top's own half at the wrong scale-plane offset.
    if (combined.weight.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        combined.weight.scale_ne[0] = combined.weight.n;
        combined.weight.scale_nb[1] = static_cast<std::int64_t>(combined.weight.n) * 2;
        combined.weight.scale_nb[2] = combined.weight.scale_nb[1];
        combined.weight.scale_nb[3] = combined.weight.scale_nb[1];
    }

    combined.code_plane_bytes  = top.code_plane_bytes + bottom.code_plane_bytes;
    combined.high_plane_offset = align256(combined.code_plane_bytes);
    combined.high_plane_bytes  = top.high_plane_bytes + bottom.high_plane_bytes;
    combined.scale_plane_offset =
        combined.high_plane_offset + align256(combined.high_plane_bytes);
    combined.scale_plane_bytes = top.scale_plane_bytes + bottom.scale_plane_bytes;

    std::size_t total = combined.scale_plane_offset + combined.scale_plane_bytes;
    const bool nvfp4   = top.weight.qtype == QType::NVFP4;
    if (nvfp4) {
        combined.weight_divisor_offset = total;
        total += 4;
    }
    combined.payload.assign(total, 0);
    // top.weight.payload_bytes was sized for top's own (smaller) n; the combined tensor needs its
    // own byte count, or validate_nvfp4_weight's `payload_bytes < required_payload_bytes` check
    // (and the RowSplit formats' own analogous size checks) rejects an otherwise-correct payload.
    combined.weight.payload_bytes = combined.payload.size();

    const auto copy_plane = [&](std::uint64_t dst_offset, const qw::PackedWeight& block,
                                std::uint64_t block_offset, std::uint64_t block_bytes) {
        if (block_bytes == 0) { return; }
        std::memcpy(combined.payload.data() + dst_offset, block.payload.data() + block_offset,
                   block_bytes);
    };
    copy_plane(0, top, 0, top.code_plane_bytes);
    copy_plane(top.code_plane_bytes, bottom, 0, bottom.code_plane_bytes);
    copy_plane(combined.high_plane_offset, top, top.high_plane_offset, top.high_plane_bytes);
    copy_plane(combined.high_plane_offset + top.high_plane_bytes, bottom, bottom.high_plane_offset,
              bottom.high_plane_bytes);
    copy_plane(combined.scale_plane_offset, top, top.scale_plane_offset, top.scale_plane_bytes);
    copy_plane(combined.scale_plane_offset + top.scale_plane_bytes, bottom,
              bottom.scale_plane_offset, bottom.scale_plane_bytes);
    if (nvfp4) {
        if (std::memcmp(top.payload.data() + top.weight_divisor_offset,
                       bottom.payload.data() + bottom.weight_divisor_offset, 4) != 0) {
            throw std::invalid_argument(
                "concat_row_blocks: mismatched weight_scale_divisor bits between blocks");
        }
        copy_plane(combined.weight_divisor_offset, top, top.weight_divisor_offset, 4);
    }
    return combined;
}

// Rank r's gate_up shard: gate rows [r*half,(r+1)*half) concatenated with up rows
// [2*half+r*half, 2*half+(r+1)*half) of the (conceptual) 2*2*half-row parent -- exactly the two
// append_column_block() calls bindings.cpp's `ends("mlp/gate_up")` rule makes.
qw::PackedWeight make_gate_up_shard(QType qtype, std::int32_t half, std::uint32_t seed, int rank) {
    qw::PackedWeight gate = make_block(qtype, half, kInputRows, seed, rank * half);
    qw::PackedWeight up   = make_block(qtype, half, kInputRows, seed, 2 * half + rank * half);
    return concat_row_blocks(gate, up);
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

// Proves, at the logical level and before any kernel runs, that a shard's gate half and up half
// really are the parent's blocks at the coordinates make_gate_up_shard() intends. Both sides
// decode through the fixture's own independent decoder (logical_weight_fp64), exactly
// test_linear_split.cpp's verify_shard_is_parent_block -- just applied twice, once per half,
// since this shard is not a single contiguous parent block.
int verify_gate_up_shard_is_parent_blocks(const std::string& label, const qw::PackedWeight& shard,
                                          const qw::PackedWeight& gate_block,
                                          const qw::PackedWeight& up_block, std::int32_t half,
                                          std::int32_t rank) {
    int failures = 0;
    for (const std::int32_t local_row : seam_samples(half)) {
        for (const std::int32_t column : seam_samples(kInputRows)) {
            const double from_shard = qw::logical_weight_fp64(shard, local_row, column);
            const double from_gate  = qw::logical_weight_fp64(gate_block, local_row, column);
            if (from_shard != from_gate) {
                std::cerr << label << ": gate half mismatch at shard row " << local_row
                          << " col " << column << ": shard=" << from_shard
                          << " gate_block=" << from_gate << '\n';
                ++failures;
            }
        }
    }
    for (const std::int32_t local_row : seam_samples(half)) {
        for (const std::int32_t column : seam_samples(kInputRows)) {
            const double from_shard = qw::logical_weight_fp64(shard, half + local_row, column);
            const double from_up    = qw::logical_weight_fp64(up_block, local_row, column);
            if (from_shard != from_up) {
                std::cerr << label << ": up half mismatch at shard row " << (half + local_row)
                          << " col " << column << ": shard=" << from_shard
                          << " up_block=" << from_up << '\n';
                ++failures;
            }
        }
    }
    (void)rank;
    return failures;
}

// The two shards must not be byte-identical. Per the pattern's own warning (vacuity has bitten
// twice already, both axes of test_linear_split.cpp) -- here the annihilating stride would be
// `half` (8704) on the row axis inside EACH block, which decorrelate_coordinates defeats; this
// assertion is what would catch a regression if that were ever dropped, on this shard's own
// construction rather than by inheriting test_linear_split.cpp's proof.
int verify_shards_are_distinct(const std::string& label,
                               const std::vector<std::uint8_t>& first,
                               const std::vector<std::uint8_t>& second) {
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

int compare(const std::string& label, const std::vector<double>& got,
            const std::vector<double>& expected) {
    constexpr ReductionCriterion criterion{/*relative_l2*/ 2.0 * kBf16Ulp, /*gross_absolute*/ 0.0,
                                           /*gross_relative_to_max_reference*/ 2.0 * kBf16Ulp};
    const ReductionStats stats = compute_reduction_stats(got.data(), expected.data(),
                                                         static_cast<std::int64_t>(got.size()));
    std::cout << "  " << label << ": max_abs=" << stats.maximum_absolute_error
              << " rel_l2=" << stats.relative_l2 << " gross_limit="
              << gross_error_limit(stats, criterion) << '\n';
    return verify_reduction(label, got, expected, criterion);
}

std::size_t swiglu_workspace_bytes(QType qtype, ops::LinearPolicy policy, std::int32_t tokens) {
    return ops::linear_swiglu_column_parallel_workspace_capacity_bytes(qtype, policy, tokens,
                                                                       tokens);
}

// ---------------------------------------------------------------------------------------------
// PRIMARY CASE: tp1 linear_swiglu() on device 0 over the full [34816,5120] weight vs
// linear_swiglu_column_parallel() over the two [17408,5120] gate/up shards.
// ---------------------------------------------------------------------------------------------
struct Case {
    const char* label;
    QType qtype;
    std::uint32_t seed;
    std::vector<std::int32_t> tokens;
    std::vector<ops::LinearPolicy> policies;
};

int run_case(const Case& test_case, const ExecutionContext& ec) {
    const QType qtype      = test_case.qtype;
    const std::string head = test_case.label;
    int failures           = 0;

    std::cout << head << " gate_up [" << kParentGateUpRows << ',' << kInputRows << "] -> shard ["
              << kParentHalf << ',' << kInputRows << "] x2\n";

    // --- weights: gate/up blocks for both ranks, spliced into shards; verified before upload ----
    qw::PackedWeight gate0 = make_block(qtype, kShardHalf, kInputRows, test_case.seed, 0);
    qw::PackedWeight up0 =
        make_block(qtype, kShardHalf, kInputRows, test_case.seed, kParentHalf);
    qw::PackedWeight gate1 =
        make_block(qtype, kShardHalf, kInputRows, test_case.seed, kShardHalf);
    qw::PackedWeight up1 =
        make_block(qtype, kShardHalf, kInputRows, test_case.seed, kParentHalf + kShardHalf);

    // Intra-shard axis: a shard's own gate block and up block must also be distinct payloads,
    // not just rank 0's shard vs rank 1's. A fixture periodic at this shard's own gap (kParentHalf
    // = 17408, twice the 8704 stride decorrelate_coordinates already defeats on the rank axis)
    // would alias gate and up bytes without changing either rank-vs-rank comparison above, and
    // verify_gate_up_shard_is_parent_blocks() alone cannot catch that -- it only proves each half
    // matches the (equally aliased) parent, which would be an equally silent tautology.
    failures += verify_shards_are_distinct(head + " shard 0 gate-vs-up", gate0.payload, up0.payload);
    failures += verify_shards_are_distinct(head + " shard 1 gate-vs-up", gate1.payload, up1.payload);

    std::array<qw::PackedWeight, 2> shard{concat_row_blocks(gate0, up0),
                                          concat_row_blocks(gate1, up1)};
    failures +=
        verify_gate_up_shard_is_parent_blocks(head + " shard 0", shard[0], gate0, up0, kShardHalf, 0);
    failures +=
        verify_gate_up_shard_is_parent_blocks(head + " shard 1", shard[1], gate1, up1, kShardHalf, 1);
    failures += verify_shards_are_distinct(head, shard[0].payload, shard[1].payload);
    if (failures != 0) { return failures; }

    // The tp1 reference weight, from the SAME logical data (row_origin=0, full 34816 rows) --
    // proves nothing new about the shard (that is what the checks above are for); it is simply the
    // whole-weight input the tp1 kernel needs.
    qw::PackedWeight full = make_block(qtype, kParentGateUpRows, kInputRows, test_case.seed, 0);

    set_device(ec, 0);
    DeviceWeight full_device = upload_weight(full);
    std::array<DeviceWeight, 2> shard_device;
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        shard_device[static_cast<std::size_t>(rank)] =
            upload_weight(shard[static_cast<std::size_t>(rank)]);
    }

    for (const std::int32_t tokens : test_case.tokens) {
        std::vector<float> activation(static_cast<std::size_t>(kInputRows) * tokens);
        fill_uniform(activation, test_case.seed * 31u + static_cast<std::uint32_t>(tokens), -1.0F,
                     1.0F);
        round_to_bf16(activation);

        set_device(ec, 0);
        DeviceBuffer full_x = to_device_bf16(activation);
        std::array<DeviceBuffer, 2> shard_x;
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            shard_x[static_cast<std::size_t>(rank)] = to_device_bf16(activation);
        }

        for (const ops::LinearPolicy policy : test_case.policies) {
            const std::string label =
                head + " T=" + std::to_string(tokens) + " " + policy_name(policy);

            // --- (a) reference: tp1 linear_swiglu, whole weight, device 0 ---------------------
            const std::size_t reference_elements = static_cast<std::size_t>(kParentHalf) * tokens;
            set_device(ec, 0);
            GuardedDeviceBuffer reference(reference_elements * sizeof(std::uint16_t));
            reference.fill(0xff);
            DeviceArena reference_arena(std::max<std::size_t>(
                ops::linear_swiglu_workspace_capacity_bytes(qtype, kParentGateUpRows, kInputRows,
                                                           policy, tokens, tokens),
                1));
            Tensor reference_x(full_x.p, DType::BF16, {kInputRows, tokens});
            Tensor reference_out(reference.data(), DType::BF16, {kParentHalf, tokens});

            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            ops::linear_swiglu(reference_x, full_device.weight, reference_out, policy,
                              reference_arena, ec.dev[0]->stream);
            cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
            failures += reference.verify_guards(label + " reference");
            const std::vector<double> expected =
                from_device_bf16(reference.data(), reference_elements);

            // --- (b) the split form -------------------------------------------------------------
            const std::size_t split_elements = static_cast<std::size_t>(kShardHalf) * tokens;
            const std::size_t workspace_bytes =
                std::max<std::size_t>(swiglu_workspace_bytes(qtype, policy, tokens), 1);

            std::array<std::optional<GuardedDeviceBuffer>, 2> split_out;
            std::array<std::optional<DeviceArena>, 2> arena;
            for (int rank = 0; rank < 2; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                set_device(ec, rank);
                split_out[slot].emplace(split_elements * sizeof(std::uint16_t));
                split_out[slot]->fill(0xff);
                arena[slot].emplace(workspace_bytes);
            }

            const std::array<Tensor, 2> x{Tensor(shard_x[0].p, DType::BF16, {kInputRows, tokens}),
                                          Tensor(shard_x[1].p, DType::BF16, {kInputRows, tokens})};
            const std::array<Weight, 2> w{shard_device[0].weight, shard_device[1].weight};
            const std::array<Tensor, 2> out{
                Tensor(split_out[0]->data(), DType::BF16, {kShardHalf, tokens}),
                Tensor(split_out[1]->data(), DType::BF16, {kShardHalf, tokens})};
            const std::array<ninfer::WorkspaceArena*, 2> workspace{&*arena[0], &*arena[1]};

            retire_staging(ec);
            ops::linear_swiglu_column_parallel(x, w, out, policy, workspace, ec);
            synchronize_both(ec);

            std::array<std::vector<double>, 2> observed;
            for (int rank = 0; rank < 2; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                set_device(ec, rank);
                failures += split_out[slot]->verify_guards(label + " rank " + std::to_string(rank));
                observed[slot] = from_device_bf16(split_out[slot]->data(), split_elements);

                // Rank r owns intermediate rows [r*kShardHalf,(r+1)*kShardHalf) of every token.
                std::vector<double> block(split_elements);
                for (std::int32_t token = 0; token < tokens; ++token) {
                    const std::size_t source = static_cast<std::size_t>(token) * kParentHalf +
                                               static_cast<std::size_t>(rank) * kShardHalf;
                    std::copy(expected.begin() + static_cast<std::ptrdiff_t>(source),
                            expected.begin() + static_cast<std::ptrdiff_t>(source + kShardHalf),
                            block.begin() + static_cast<std::ptrdiff_t>(token) * kShardHalf);
                }
                failures += compare(label + " rank " + std::to_string(rank), observed[slot], block);
            }

            // Cross-rank structural check: the two ranks own DIFFERENT intermediate rows, so their
            // outputs must differ (the column-parallel mirror of test_linear_split.cpp's own
            // "both ranks produced identical output blocks" check).
            if (observed[0] == observed[1]) {
                std::cerr << label << ": both ranks produced identical output blocks, so the "
                                      "column split did not actually split\n";
                ++failures;
            }
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------------------------
// PIPELINE CASE: the REAL Variant::post_mixer call graph, end to end --
//   tp1: linear_swiglu(hidden, gate_up_full, activation) + linear_add(activation, down_full, residual)
//   tp2: linear_swiglu_column_parallel(...) -> linear_add_row_parallel(...)
// Rank r's split gate_up output feeds DIRECTLY into linear_add_row_parallel's x[r] -- no reshaping
// -- exactly the composition include/ninfer/ops/linear_swiglu.h's design note describes.
// ---------------------------------------------------------------------------------------------
struct PipelineCase {
    const char* label;
    QType gate_up_qtype;
    QType down_qtype;
    std::uint32_t seed;
    std::int32_t tokens;
    ops::LinearPolicy policy;
};

int run_pipeline_case(const PipelineCase& test_case, const ExecutionContext& ec,
                      const ops::PeerEvents& events) {
    constexpr std::int32_t kDownRows = 5120; // o_proj/gdn/output width; mlp/down [5120,17408]
    const std::int32_t tokens        = test_case.tokens;
    const std::string head           = test_case.label;
    int failures                     = 0;

    std::cout << head << " pipeline T=" << tokens << ' ' << policy_name(test_case.policy) << '\n';

    // --- gate_up: full + two shards (same construction as run_case) ------------------------------
    qw::PackedWeight gate0 =
        make_block(test_case.gate_up_qtype, kShardHalf, kInputRows, test_case.seed, 0);
    qw::PackedWeight up0 = make_block(test_case.gate_up_qtype, kShardHalf, kInputRows,
                                      test_case.seed, kParentHalf);
    qw::PackedWeight gate1 = make_block(test_case.gate_up_qtype, kShardHalf, kInputRows,
                                        test_case.seed, kShardHalf);
    qw::PackedWeight up1 = make_block(test_case.gate_up_qtype, kShardHalf, kInputRows,
                                      test_case.seed, kParentHalf + kShardHalf);
    // Intra-shard axis, same discipline as run_case: each rank's own gate block vs its own up
    // block must be distinct payloads, not just rank 0's shard vs rank 1's.
    failures += verify_shards_are_distinct(head + " gate_up shard 0 gate-vs-up", gate0.payload,
                                           up0.payload);
    failures += verify_shards_are_distinct(head + " gate_up shard 1 gate-vs-up", gate1.payload,
                                           up1.payload);
    std::array<qw::PackedWeight, 2> gate_up_shard{concat_row_blocks(gate0, up0),
                                                  concat_row_blocks(gate1, up1)};
    failures += verify_shards_are_distinct(head + " gate_up", gate_up_shard[0].payload,
                                           gate_up_shard[1].payload);
    qw::PackedWeight gate_up_full =
        make_block(test_case.gate_up_qtype, kParentGateUpRows, kInputRows, test_case.seed, 0);

    // --- down: full [5120,17408] + two K-shards [5120,8704] (single contiguous column origin,
    // exactly test_linear_add_split.cpp's own construction -- mlp/down is a plain row-parallel
    // split, no gate/up subdivision) --------------------------------------------------------------
    qw::PatternedWeightOptions down_options;
    down_options.decorrelate_coordinates = true;
    if (test_case.down_qtype == QType::NVFP4) {
        down_options.weight_scale_divisor = 0.125F;
        down_options.input_scale_divisor  = 3.5F;
    } else {
        down_options.row_split_scale = qw::RowSplitScalePattern::Small;
        down_options.row_split_codes = qw::RowSplitCodePattern::Coordinate;
    }
    qw::PackedWeight down_full =
        qw::make_patterned_weight(test_case.down_qtype, kDownRows, kParentHalf,
                                  test_case.seed + 1, down_options);
    std::array<qw::PackedWeight, 2> down_shard;
    for (int rank = 0; rank < 2; ++rank) {
        qw::PatternedWeightOptions opts = down_options;
        opts.column_origin              = rank * kShardHalf;
        down_shard[static_cast<std::size_t>(rank)] = qw::make_patterned_weight(
            test_case.down_qtype, kDownRows, kShardHalf, test_case.seed + 1, opts);
    }
    failures += verify_shards_are_distinct(head + " down", down_shard[0].payload,
                                           down_shard[1].payload);
    if (failures != 0) { return failures; }

    set_device(ec, 0);
    DeviceWeight gate_up_full_device = upload_weight(gate_up_full);
    DeviceWeight down_full_device    = upload_weight(down_full);
    std::array<DeviceWeight, 2> gate_up_shard_device;
    std::array<DeviceWeight, 2> down_shard_device;
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        const auto slot                      = static_cast<std::size_t>(rank);
        gate_up_shard_device[slot]           = upload_weight(gate_up_shard[slot]);
        down_shard_device[slot]              = upload_weight(down_shard[slot]);
    }

    std::vector<float> activation(static_cast<std::size_t>(kInputRows) * tokens);
    fill_uniform(activation, test_case.seed * 31u + 7u, -1.0F, 1.0F);
    round_to_bf16(activation);
    std::vector<float> residual0(static_cast<std::size_t>(kDownRows) * tokens);
    fill_uniform(residual0, test_case.seed * 97u + 11u, -2.0F, 2.0F);
    round_to_bf16(residual0);
    std::vector<std::uint16_t> residual0_words(residual0.size());
    for (std::size_t i = 0; i < residual0.size(); ++i) { residual0_words[i] = f32_to_bf16(residual0[i]); }

    set_device(ec, 0);
    DeviceBuffer full_x = to_device_bf16(activation);
    std::array<DeviceBuffer, 2> shard_x;
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        shard_x[static_cast<std::size_t>(rank)] = to_device_bf16(activation);
    }

    // --- (a) reference: tp1 linear_swiglu + linear_add, device 0, exactly Variant::post_mixer ----
    set_device(ec, 0);
    const std::size_t activation_elements = static_cast<std::size_t>(kParentHalf) * tokens;
    const std::size_t residual_elements   = static_cast<std::size_t>(kDownRows) * tokens;
    GuardedDeviceBuffer reference_activation(activation_elements * sizeof(std::uint16_t));
    reference_activation.fill(0xff);
    GuardedDeviceBuffer reference_residual(residual_elements * sizeof(std::uint16_t));
    reference_residual.copy_from_host(residual0_words.data(),
                                      residual_elements * sizeof(std::uint16_t));
    DeviceArena reference_swiglu_arena(std::max<std::size_t>(
        ops::linear_swiglu_workspace_capacity_bytes(test_case.gate_up_qtype, kParentGateUpRows,
                                                   kInputRows, test_case.policy, tokens, tokens),
        1));
    DeviceArena reference_add_arena(std::max<std::size_t>(
        ops::linear_add_workspace_capacity_bytes(test_case.down_qtype, kDownRows, kParentHalf,
                                                test_case.policy, tokens, tokens),
        1));
    Tensor reference_x(full_x.p, DType::BF16, {kInputRows, tokens});
    Tensor reference_activation_view(reference_activation.data(), DType::BF16,
                                     {kParentHalf, tokens});
    Tensor reference_residual_view(reference_residual.data(), DType::BF16, {kDownRows, tokens});

    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    ops::linear_swiglu(reference_x, gate_up_full_device.weight, reference_activation_view,
                      test_case.policy, reference_swiglu_arena, ec.dev[0]->stream);
    ops::linear_add(reference_activation_view, down_full_device.weight, reference_residual_view,
                    test_case.policy, reference_add_arena, ec.dev[0]->stream);
    cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
    failures += reference_activation.verify_guards(head + " reference activation");
    failures += reference_residual.verify_guards(head + " reference residual");
    const std::vector<double> expected =
        from_device_bf16(reference_residual.data(), residual_elements);

    // --- (b) the split pipeline -------------------------------------------------------------------
    std::array<std::optional<DeviceBuffer>, 2> split_activation;
    std::array<std::optional<DeviceArena>, 2> swiglu_arena;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        split_activation[slot].emplace(static_cast<std::size_t>(kShardHalf) * tokens *
                                       sizeof(std::uint16_t));
        swiglu_arena[slot].emplace(std::max<std::size_t>(
            swiglu_workspace_bytes(test_case.gate_up_qtype, test_case.policy, tokens), 1));
    }
    {
        const std::array<Tensor, 2> x{Tensor(shard_x[0].p, DType::BF16, {kInputRows, tokens}),
                                      Tensor(shard_x[1].p, DType::BF16, {kInputRows, tokens})};
        const std::array<Weight, 2> w{gate_up_shard_device[0].weight,
                                      gate_up_shard_device[1].weight};
        const std::array<Tensor, 2> out{
            Tensor(split_activation[0]->p, DType::BF16, {kShardHalf, tokens}),
            Tensor(split_activation[1]->p, DType::BF16, {kShardHalf, tokens})};
        const std::array<ninfer::WorkspaceArena*, 2> workspace{&*swiglu_arena[0],
                                                                &*swiglu_arena[1]};
        retire_staging(ec);
        ops::linear_swiglu_column_parallel(x, w, out, test_case.policy, workspace, ec);
        synchronize_both(ec);
    }

    std::array<std::optional<GuardedDeviceBuffer>, 2> split_residual;
    std::array<std::optional<DeviceBuffer>, 2> staging;
    std::array<std::optional<DeviceArena>, 2> add_arena;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        split_residual[slot].emplace(residual_elements * sizeof(std::uint16_t));
        split_residual[slot]->copy_from_host(residual0_words.data(),
                                             residual_elements * sizeof(std::uint16_t));
        staging[slot].emplace(residual_elements * sizeof(std::uint16_t));
        add_arena[slot].emplace(std::max<std::size_t>(
            ops::linear_add_workspace_capacity_bytes(test_case.down_qtype, kDownRows, kShardHalf,
                                                    test_case.policy, tokens, tokens),
            1));
    }
    {
        const std::array<Tensor, 2> x{
            Tensor(split_activation[0]->p, DType::BF16, {kShardHalf, tokens}),
            Tensor(split_activation[1]->p, DType::BF16, {kShardHalf, tokens})};
        const std::array<Weight, 2> w{down_shard_device[0].weight, down_shard_device[1].weight};
        const std::array<Tensor, 2> residual{
            Tensor(split_residual[0]->data(), DType::BF16, {kDownRows, tokens}),
            Tensor(split_residual[1]->data(), DType::BF16, {kDownRows, tokens})};
        const std::array<Tensor, 2> staging_view{
            Tensor(staging[0]->p, DType::BF16, {kDownRows, tokens}),
            Tensor(staging[1]->p, DType::BF16, {kDownRows, tokens})};
        const std::array<ninfer::WorkspaceArena*, 2> workspace{&*add_arena[0], &*add_arena[1]};
        retire_staging(ec);
        ops::linear_add_row_parallel(x, w, residual, staging_view, test_case.policy, workspace, ec,
                                    events);
        synchronize_both(ec);
    }

    std::array<std::vector<double>, 2> observed;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        failures += split_residual[slot]->verify_guards(head + " pipeline rank " +
                                                        std::to_string(rank));
        observed[slot] = from_device_bf16(split_residual[slot]->data(), residual_elements);
        failures += compare(head + " pipeline rank " + std::to_string(rank), observed[slot],
                           expected);
    }
    if (observed[0] != observed[1]) {
        std::cerr << head << ": the two pipeline ranks disagree after the all-reduce\n";
        ++failures;
    }
    return failures;
}

// ---------------------------------------------------------------------------------------------
// Host-only registry probe: linear_swiglu_column_parallel_workspace_capacity_bytes() runs each
// format's shape/policy resolver without touching a device, so it runs before the GPU checks and
// still catches a regression on a one-GPU CI box.
// ---------------------------------------------------------------------------------------------
int verify_registry() {
    int failures = 0;
    constexpr auto kA16 = ops::LinearPolicy::A16Only;
    constexpr auto kA4  = ops::LinearPolicy::AllowA4;
    constexpr auto kA8  = ops::LinearPolicy::AllowA8;

    for (const std::int32_t tokens : {1, 2, 16, 48, 1024}) {
        try {
            (void)ops::linear_swiglu_column_parallel_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16S, kA16, tokens, tokens);
        } catch (const std::exception& error) {
            std::cerr << "registry: fp8 A16Only T=" << tokens << " rejected: " << error.what()
                      << '\n';
            ++failures;
        }
        try {
            (void)ops::linear_swiglu_column_parallel_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16S, kA8, tokens, tokens);
        } catch (const std::exception& error) {
            std::cerr << "registry: fp8 AllowA8 T=" << tokens << " rejected: " << error.what()
                      << '\n';
            ++failures;
        }
        try {
            (void)ops::linear_swiglu_column_parallel_workspace_capacity_bytes(QType::NVFP4, kA16,
                                                                              tokens, tokens);
        } catch (const std::exception& error) {
            if (tokens > 16) { continue; } // NVFP4 A16 registered only through T=16, like tp1
            std::cerr << "registry: nvfp4 A16Only T=" << tokens << " rejected: " << error.what()
                      << '\n';
            ++failures;
        }
        try {
            (void)ops::linear_swiglu_column_parallel_workspace_capacity_bytes(QType::NVFP4, kA4,
                                                                              tokens, tokens);
        } catch (const std::exception& error) {
            std::cerr << "registry: nvfp4 AllowA4 T=" << tokens << " rejected: " << error.what()
                      << '\n';
            ++failures;
        }
        try {
            (void)ops::linear_swiglu_column_parallel_workspace_capacity_bytes(QType::Q4G64_F16S,
                                                                              kA16, tokens, tokens);
        } catch (const std::exception& error) {
            std::cerr << "registry: q4 A16Only T=" << tokens << " rejected: " << error.what()
                      << '\n';
            ++failures;
        }
    }

    // Volta has a precision-qualified QPN split through T=32; the non-Volta small-T profile
    // retains the historical T<=16 registration.
    bool threw = false;
    try {
        (void)ops::linear_swiglu_column_parallel_workspace_capacity_bytes(QType::NVFP4, kA16, 17,
                                                                          17);
    } catch (const std::exception&) { threw = true; }
#ifdef NINFER_VOLTA_BUILD
    if (threw) {
        std::cerr << "registry: Volta NVFP4 A16Only T=17 was rejected\n";
        ++failures;
    }
#else
    if (!threw) {
        std::cerr << "registry: nvfp4 A16Only T=17 was admitted but must not be\n";
        ++failures;
    }
#endif

    // Q4 rejects any policy beyond A16Only.
    threw = false;
    try {
        (void)ops::linear_swiglu_column_parallel_workspace_capacity_bytes(QType::Q4G64_F16S, kA4,
                                                                          1, 1);
    } catch (const std::exception&) { threw = true; }
    if (!threw) {
        std::cerr << "registry: q4 AllowA4 was admitted but must not be\n";
        ++failures;
    }

    // FP8 rejects AllowA4 (not a policy it admits).
    threw = false;
    try {
        (void)ops::linear_swiglu_column_parallel_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, kA4, 1, 1);
    } catch (const std::exception&) { threw = true; }
    if (!threw) {
        std::cerr << "registry: fp8 AllowA4 was admitted but must not be\n";
        ++failures;
    }

    // W8 is deliberately not registered for the split form (see linear_swiglu.h's design note):
    // its own linear_swiglu profile belongs only to the never-sharded MTP head.
    threw = false;
    try {
        (void)ops::linear_swiglu_column_parallel_workspace_capacity_bytes(QType::W8G32_F16S, kA16,
                                                                          1, 1);
    } catch (const std::exception&) { threw = true; }
    if (!threw) {
        std::cerr << "registry: w8 was admitted but must not be\n";
        ++failures;
    }

    std::cout << (failures ? "FAIL" : "OK") << " registry\n";
    return failures;
}

// Rejection cases the split pair owns: only two ranks together can see them.
int verify_split_rejections(const ExecutionContext& ec) {
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

    set_device(ec, 0);
    DeviceBuffer x0(static_cast<std::size_t>(kInputRows) * 2 * sizeof(std::uint16_t));
    DeviceBuffer out0(static_cast<std::size_t>(kShardHalf) * 2 * sizeof(std::uint16_t));
    set_device(ec, 1);
    DeviceBuffer x1(static_cast<std::size_t>(kInputRows) * 2 * sizeof(std::uint16_t));
    DeviceBuffer out1(static_cast<std::size_t>(kShardHalf) * 2 * sizeof(std::uint16_t));

    Weight fake{};
    fake.qtype        = QType::NVFP4;
    fake.n            = kParentHalf;
    fake.k            = kInputRows;
    fake.padded_shape[0] = kParentHalf;
    fake.padded_shape[1] = kInputRows;

    expect_throw("token count", [&] {
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kInputRows, 2}),
                                      Tensor(x1.p, DType::BF16, {kInputRows, 1})};
        const std::array<Tensor, 2> out{Tensor(out0.p, DType::BF16, {kShardHalf, 2}),
                                        Tensor(out1.p, DType::BF16, {kShardHalf, 1})};
        ops::linear_swiglu_column_parallel(x, {fake, fake}, out, ec);
    });

    expect_throw("column K", [&] {
        Weight other = fake;
        other.k      = kInputRows / 2;
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kInputRows, 1}),
                                      Tensor(x1.p, DType::BF16, {kInputRows / 2, 1})};
        const std::array<Tensor, 2> out{Tensor(out0.p, DType::BF16, {kShardHalf, 1}),
                                        Tensor(out1.p, DType::BF16, {kShardHalf, 1})};
        ops::linear_swiglu_column_parallel(x, {fake, other}, out, ec);
    });

    expect_throw("tp1 context", [&] {
        const ExecutionContext single({0});
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kInputRows, 1}),
                                      Tensor(x1.p, DType::BF16, {kInputRows, 1})};
        const std::array<Tensor, 2> out{Tensor(out0.p, DType::BF16, {kShardHalf, 1}),
                                        Tensor(out1.p, DType::BF16, {kShardHalf, 1})};
        ops::linear_swiglu_column_parallel(x, {fake, fake}, out, single);
    });

    expect_throw("unsupported format", [&] {
        Weight w8 = fake;
        w8.qtype  = QType::W8G32_F16S;
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kInputRows, 1}),
                                      Tensor(x1.p, DType::BF16, {kInputRows, 1})};
        const std::array<Tensor, 2> out{Tensor(out0.p, DType::BF16, {kShardHalf, 1}),
                                        Tensor(out1.p, DType::BF16, {kShardHalf, 1})};
        ops::linear_swiglu_column_parallel(x, {w8, w8}, out, ec);
    });

    std::cout << (failures ? "FAIL" : "OK") << " split rejections\n";
    return failures;
}

} // namespace

int main() {
    int failures = verify_registry();
    if (failures != 0) {
        std::cout << "FAIL linear_swiglu split (registry)\n";
        return 1;
    }

    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: split linear_swiglu parity requires two CUDA devices, found "
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

    failures += verify_split_rejections(ec);

    constexpr auto kA16 = ops::LinearPolicy::A16Only;
    constexpr auto kA4  = ops::LinearPolicy::AllowA4;
    constexpr auto kA8  = ops::LinearPolicy::AllowA8;

    // T sweep: 1 (decode edge); 4/5 (A16<->A4 small-T boundary); 16 (A16Only's own registered
    // ceiling); 48 (FusedW4A4's own ceiling); 49/100/400 (LinearW4A4Post -- the
    // linear()+silu_mul() materialized fallback, which is where the split's Geometry-templated
    // baseline workspace sizing actually gets exercised); 1024 (multiple of 256, the sole route
    // into the NVFP4 W4A4 TMA kernel -- exercises the shard TMA descriptors end to end).
    const std::vector<Case> cases{
        // A16Only is registered only through T=16 (linear_swiglu.h's own documented domain), so it
        // gets its own token list; AllowA4 covers the full sweep including the LinearW4A4Post
        // materialized fallback (49/100/400) and the TMA route (1024).
        {"nvfp4 gate_up A16", QType::NVFP4, 31u, {1, 4, 5, 16}, {kA16}},
        {"nvfp4 gate_up A4", QType::NVFP4, 31u, {1, 4, 5, 16, 48, 49, 100, 400, 1024}, {kA4}},
        {"q4 gate_up", QType::Q4G64_F16S, 32u, {1, 17, 128, 1024}, {kA16}},
        // FP8's own tp2 column shard, wired as a TRUE split (the fused kernel family is
        // Geometry-templated, same as NVFP4 -- see linear_swiglu.h's design note). T sweep: 1 the
        // decode edge; 2 the AllowA8 route's own T=2 A16-SIMT exception; 3/4 small-T
        // (kFp8LinearSmallTMax<MlpGateUp>=4); 5 the first A8 T; 48/128/1024 beyond it, including
        // the fully generic (never a TMA/exact-large-T kernel for this format) large-T chunk loop.
        {"fp8 gate_up", QType::FP8_E4M3FN_ROW_BF16S, 38u, {1, 2, 3, 4, 5, 48, 128, 1024},
         {kA16, kA8}},
    };
    for (const Case& test_case : cases) { failures += run_case(test_case, ec); }

    const std::vector<PipelineCase> pipeline_cases{
        {"nvfp4+nvfp4", QType::NVFP4, QType::NVFP4, 41u, 8, kA4},
        {"nvfp4+nvfp4 T=1024", QType::NVFP4, QType::NVFP4, 42u, 1024, kA4},
        {"q4+q5 (groupwise profile)", QType::Q4G64_F16S, QType::Q5G64_F16S, 43u, 8, kA16},
        // The real Text-layer-56-63 profile -- both gate_up and down bound as FP8 (the
        // flagship qwen3_8 profile's own MLP-tail weights). A16Only only: linear_add's row-parallel
        // AllowA8 route needs the wider tolerance test_linear_add_split.cpp documents
        // (kFp8A8RowSplitCriterion) which this file's own compare() does not carry; that route is
        // already covered directly by test_linear_add_split.cpp's own "fp8 mlp_down" case.
        {"fp8+fp8", QType::FP8_E4M3FN_ROW_BF16S, QType::FP8_E4M3FN_ROW_BF16S, 44u, 8, kA16},
        {"fp8+fp8 T=1024", QType::FP8_E4M3FN_ROW_BF16S, QType::FP8_E4M3FN_ROW_BF16S, 45u, 1024,
         kA16},
    };
    for (const PipelineCase& test_case : pipeline_cases) {
        failures += run_pipeline_case(test_case, ec, events);
    }

    std::cout << (failures ? "FAIL" : "OK") << " linear_swiglu split\n";
    return failures ? 1 : 0;
}
