// Two-device parity suite for ops::attn_input_proj_column_parallel
// (include/ninfer/ops/attn_input_proj.h) -- the fused attention input-projection (qkv+gate)
// sibling of test_linear_split.cpp and test_linear_swiglu_split.cpp. Read test_linear_split.cpp's
// header comment first; this one documents what differs.
//
// Every case REQUIRES two CUDA devices driven from ONE process; with fewer than two visible
// devices the suite reports the repository's skip code (77) instead of failing.
//
// THE REAL PHYSICAL ROW ORDER IS Q | K | GATE | V, NOT "q|k|v|gate". Confirmed against
// bindings.cpp's `shard_mapping` for "attention/query_key_gate_value" (four independent
// append_column_block calls, in that order) and against the pre-existing
// Nvfp4AttentionInputSmallTOutput::store epilogue mapping this family already carried before the
// split existed.
// Parent [14336,5120] = Q[0,6144) | K[6144,7168) | Gate[7168,13312) | V[13312,14336).
//
// THE SHARD IS NOT A SINGLE CONTIGUOUS BLOCK. Because each of the four sections is split by tp
// independently, rank r's fused shard weight is the CONCATENATION, in the SAME Q|K|Gate|V order,
// of rank r's own head-local slice of every section: a standalone [7168,5120] tensor with Query at
// shard-local offset 0 (3072 rows), Key at 3072 (512 rows), Gate at 3584 (3072 rows), Value at
// 6656 (512 rows). See concat_row_blocks()/make_fused_shard() below for how the four blocks are
// produced and spliced, and verify_shard_is_parent_block() (copied from test_linear_split.cpp) for
// how each splice is proven correct against the parent's own logical decode, before any kernel
// runs.
//
// THREE FORMS ARE REGISTERED (see include/ninfer/ops/attn_input_proj.h's design note):
//   NVFP4 fused single-parent -- a TRUE split: the SAME kernel templates (decode/small-T/W4A4/TMA),
//     instantiated at the shard's Geometry.
//   FP8_E4M3FN_ROW_BF16S fused single-parent -- a TRUE split, the same shape of change
//     as NVFP4's: the SAME kernel templates (decode/small-T/A8), instantiated at <Geometry, Output>.
//   Q4G64_F16S/Q5G64_F16S split-storage two-weight form -- the grouped-MMA kernel is already
//     row-count-generic, so the shard always routes through it regardless of T.
//
// TOLERANCE. 2 BF16 ulp of the largest output in the tensor (gross_relative_to_max_reference) plus
// a matching relative-L2 leg, exactly test_linear_split.cpp's criterion and rationale -- applied
// PER SECTION (q/gate/k/v separately), because the interleaved section order makes a whole-buffer
// compare error-prone at localizing a seam bug.
#include "ninfer/ops/attn_input_proj.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include "core/device.h"
#include "core/arena.h"

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

// The real qwen3_6_27b geometry (config.query_size=6144, config.kv_size=1024, hidden=5120), both
// registered forms share it: fused parent [14336,5120] and the two split-storage parents
// [7168,5120] each. Section boundaries per bindings.cpp -- see the header comment above.
constexpr std::int32_t kInputRows      = 5120;
constexpr std::int32_t kQRows          = 6144; // tp1 query/gate section rows
constexpr std::int32_t kKvRows         = 1024; // tp1 key/value section rows
constexpr std::int32_t kFusedRows      = kQRows + kKvRows + kQRows + kKvRows; // 14336
constexpr std::int32_t kSplitRows      = kQRows + kKvRows;                   // 7168, per split weight
constexpr std::int32_t kShardQRows     = kQRows / 2;                         // 3072
constexpr std::int32_t kShardKvRows    = kKvRows / 2;                        // 512
constexpr std::int32_t kShardFusedRows = kShardQRows + kShardKvRows + kShardQRows + kShardKvRows; // 7168
constexpr std::int32_t kShardSplitRows = kShardQRows + kShardKvRows;         // 3584

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

// Single-origin generation for one contiguous row block. decorrelate_coordinates=true, same
// rationale as every sibling split suite: the section strides here (3072, 512, 6144, 7168, ...)
// annihilate the fixture's un-decorrelated affine pattern.
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
// logical rows are [top's rows][bottom's rows] -- copied verbatim from
// test_linear_swiglu_split.cpp's concat_row_blocks(), whose doc comment explains why this is valid
// for every layout the fixture produces (NVFP4 additionally requires the top block's own row count
// to be a 128-row tile multiple; every section here -- 3072, 512, 3584, 6656 -- satisfies it).
qw::PackedWeight concat_row_blocks(const qw::PackedWeight& top, const qw::PackedWeight& bottom) {
    if (top.weight.qtype != bottom.weight.qtype || top.weight.k != bottom.weight.k ||
        top.weight.layout != bottom.weight.layout) {
        throw std::invalid_argument("concat_row_blocks: mismatched qtype/layout/k");
    }
    if (top.weight.qtype == QType::NVFP4 && (top.weight.n % 128) != 0) {
        throw std::invalid_argument(
            "concat_row_blocks: NVFP4 top block must be a 128-row tile multiple");
    }
    const auto align256 = [](std::uint64_t bytes) -> std::uint64_t {
        return (bytes + 255U) & ~std::uint64_t{255};
    };

    qw::PackedWeight combined;
    combined.weight                 = top.weight;
    combined.weight.n               = top.weight.n + bottom.weight.n;
    combined.weight.shape[0]        = combined.weight.n;
    combined.weight.padded_shape[0] = combined.weight.n;
    // FP8 fixture requirement, shared verbatim with test_linear_swiglu_split.cpp and
    // test_gdn_projections_split.cpp, which carry their own copies of this splice: FP8's
    // row-scale-v1 layout denormalizes `n` into scale_ne[0]/scale_nb[1..3] (one BF16 word per
    // output row). `combined.weight` above is a byte-copy of `top.weight`, sized for top's own
    // (smaller) n, so those four fields must be re-derived from the COMBINED n or the kernel reads
    // every row beyond top's own block at the wrong scale-plane offset.
    if (combined.weight.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        combined.weight.scale_ne[0] = combined.weight.n;
        combined.weight.scale_nb[1] = static_cast<std::int64_t>(combined.weight.n) * 2;
        combined.weight.scale_nb[2] = combined.weight.scale_nb[1];
        combined.weight.scale_nb[3] = combined.weight.scale_nb[1];
    }

    combined.code_plane_bytes  = top.code_plane_bytes + bottom.code_plane_bytes;
    combined.high_plane_offset = align256(combined.code_plane_bytes);
    combined.high_plane_bytes  = top.high_plane_bytes + bottom.high_plane_bytes;
    combined.scale_plane_offset = combined.high_plane_offset + align256(combined.high_plane_bytes);
    combined.scale_plane_bytes = top.scale_plane_bytes + bottom.scale_plane_bytes;

    std::size_t total = combined.scale_plane_offset + combined.scale_plane_bytes;
    const bool nvfp4  = top.weight.qtype == QType::NVFP4;
    if (nvfp4) {
        combined.weight_divisor_offset = total;
        total += 4;
    }
    combined.payload.assign(total, 0);
    // top.weight.payload_bytes was sized for top's own (smaller) n; the combined tensor needs its
    // own byte count, or validate_nvfp4_weight's / the RowSplit formats' own size checks reject an
    // otherwise-correct payload.
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

// Proves, at the logical level and before any kernel runs, that `shard` really is the parent's
// block at (row_origin, column_origin) -- copied from test_linear_split.cpp's own
// verify_shard_is_parent_block.
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

// The two shards (or two intra-shard blocks) must not be byte-identical. See
// test_linear_split.cpp's own verify_shards_are_distinct for why this is the load-bearing check of
// the whole suite: the fixture's affine patterns are periodic at several of this family's own
// strides (3072, 512, 6144, 7168 all annihilate the un-decorrelated NVFP4 rule), so a vacuous
// comparison would report zero error even for a completely wrong shard.
template <typename Bytes>
int verify_shards_are_distinct(const std::string& label, const Bytes& first, const Bytes& second) {
    if (first == second) {
        std::cerr << label
                  << ": the two payloads are byte-identical, so shard identity is untested on this "
                     "axis (the generator's pattern is degenerate at this stride)\n";
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
              << " rel_l2=" << stats.relative_l2
              << " gross_limit=" << gross_error_limit(stats, criterion) << '\n';
    return verify_reduction(label, got, expected, criterion);
}

// Rank r's block of a reference section, extracted per-token: reference has `parent_rows` rows and
// this rank owns rows [r*shard_rows, (r+1)*shard_rows) of every token.
std::vector<double> extract_rank_block(const std::vector<double>& reference,
                                       std::int32_t parent_rows, std::int32_t shard_rows,
                                       std::int32_t tokens, int rank) {
    std::vector<double> block(static_cast<std::size_t>(shard_rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t source = static_cast<std::size_t>(token) * parent_rows +
                                   static_cast<std::size_t>(rank) * shard_rows;
        std::copy(reference.begin() + static_cast<std::ptrdiff_t>(source),
                 reference.begin() + static_cast<std::ptrdiff_t>(source + shard_rows),
                 block.begin() + static_cast<std::ptrdiff_t>(token) * shard_rows);
    }
    return block;
}

// ================================================================================================
// Fused single-parent form (NVFP4).
// ================================================================================================

struct FusedShard {
    qw::PackedWeight q;
    qw::PackedWeight k;
    qw::PackedWeight gate;
    qw::PackedWeight v;
    qw::PackedWeight shard; // concat(q, k, gate, v), the real ShardPlan section order
};

// Rank r's fused shard: its own head-local slice of every section, in the parent's Q|K|Gate|V
// order -- exactly what shard_mapping's four independent append_column_block() calls produce (see
// src/targets/qwen3_6_27b/impl/load/bindings.cpp, "attention/query_key_gate_value").
FusedShard make_fused_shard(QType qtype, std::uint32_t seed, int rank) {
    FusedShard out;
    out.q    = make_block(qtype, kShardQRows, kInputRows, seed, rank * kShardQRows);
    out.k    = make_block(qtype, kShardKvRows, kInputRows, seed, kQRows + rank * kShardKvRows);
    out.gate = make_block(qtype, kShardQRows, kInputRows, seed, kQRows + kKvRows + rank * kShardQRows);
    out.v    = make_block(qtype, kShardKvRows, kInputRows, seed,
                          kQRows + kKvRows + kQRows + rank * kShardKvRows);
    out.shard = concat_row_blocks(concat_row_blocks(concat_row_blocks(out.q, out.k), out.gate), out.v);
    return out;
}

int run_fused_case(const ExecutionContext& ec, QType qtype, std::uint32_t seed,
                   const std::string& label, const std::vector<std::int32_t>& tokens_sweep,
                   const std::vector<ops::LinearPolicy>& policies) {
    const std::string head = label;
    std::cout << head << " [" << kFusedRows << ',' << kInputRows << "] -> [" << kShardFusedRows
              << ',' << kInputRows << "]\n";
    int failures = 0;

    qw::PatternedWeightOptions full_options;
    full_options.decorrelate_coordinates = true;
    if (qtype == QType::NVFP4) {
        full_options.weight_scale_divisor = 0.125F;
        full_options.input_scale_divisor  = 3.5F;
    }
    const qw::PackedWeight full =
        qw::make_patterned_weight(qtype, kFusedRows, kInputRows, seed, full_options);

    std::array<FusedShard, 2> shard{make_fused_shard(qtype, seed, 0), make_fused_shard(qtype, seed, 1)};

    for (int rank = 0; rank < 2; ++rank) {
        const std::string label = head + " shard " + std::to_string(rank);
        failures += verify_shard_is_parent_block(label + " q", full, shard[static_cast<std::size_t>(rank)].q,
                                                 rank * kShardQRows, 0);
        failures += verify_shard_is_parent_block(label + " k", full, shard[static_cast<std::size_t>(rank)].k,
                                                 kQRows + rank * kShardKvRows, 0);
        failures += verify_shard_is_parent_block(
            label + " gate", full, shard[static_cast<std::size_t>(rank)].gate,
            kQRows + kKvRows + rank * kShardQRows, 0);
        failures += verify_shard_is_parent_block(
            label + " v", full, shard[static_cast<std::size_t>(rank)].v,
            kQRows + kKvRows + kQRows + rank * kShardKvRows, 0);
        // Intra-shard distinctness: the four sections of ONE rank's shard must not collapse into
        // each other. q-vs-gate is the pattern's own explicit minimum; k-vs-v covers the other pair.
        failures += verify_shards_are_distinct(label + " q-vs-gate",
                                               shard[static_cast<std::size_t>(rank)].q.payload,
                                               shard[static_cast<std::size_t>(rank)].gate.payload);
        failures += verify_shards_are_distinct(label + " k-vs-v",
                                               shard[static_cast<std::size_t>(rank)].k.payload,
                                               shard[static_cast<std::size_t>(rank)].v.payload);
    }
    // Cross-rank distinctness: the two ranks' full shard payloads must differ.
    failures += verify_shards_are_distinct(head, shard[0].shard.payload, shard[1].shard.payload);
    if (failures != 0) { return failures; }

    set_device(ec, 0);
    DeviceWeight full_device = upload_weight(full);
    std::array<DeviceWeight, 2> shard_device;
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        shard_device[static_cast<std::size_t>(rank)] =
            upload_weight(shard[static_cast<std::size_t>(rank)].shard);
    }

    for (const std::int32_t tokens : tokens_sweep) {
        std::vector<float> activation(static_cast<std::size_t>(kInputRows) * tokens);
        fill_uniform(activation, seed * 31u + static_cast<std::uint32_t>(tokens), -1.0F, 1.0F);
        round_to_bf16(activation);

        set_device(ec, 0);
        DeviceBuffer full_x = to_device_bf16(activation);
        std::array<DeviceBuffer, 2> shard_x;
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            shard_x[static_cast<std::size_t>(rank)] = to_device_bf16(activation);
        }

        for (const ops::LinearPolicy policy : policies) {
            const std::string label = head + " T=" + std::to_string(tokens) + " " + policy_name(policy);

            // --- (a) reference: tp1 kernel, whole weight, device 0 -----------------------------
            set_device(ec, 0);
            GuardedDeviceBuffer ref_q(static_cast<std::size_t>(kQRows) * tokens * sizeof(std::uint16_t));
            GuardedDeviceBuffer ref_gate(static_cast<std::size_t>(kQRows) * tokens * sizeof(std::uint16_t));
            GuardedDeviceBuffer ref_k(static_cast<std::size_t>(kKvRows) * tokens * sizeof(std::uint16_t));
            GuardedDeviceBuffer ref_v(static_cast<std::size_t>(kKvRows) * tokens * sizeof(std::uint16_t));
            ref_q.fill(0xff);
            ref_gate.fill(0xff);
            ref_k.fill(0xff);
            ref_v.fill(0xff);
            const std::size_t reference_capacity = ops::attn_input_proj_workspace_capacity_bytes(
                qtype, kFusedRows, kInputRows, policy, tokens, tokens);
            DeviceArena reference_arena(std::max<std::size_t>(reference_capacity, 1));
            Tensor reference_x(full_x.p, DType::BF16, {kInputRows, tokens});
            Tensor reference_q(ref_q.data(), DType::BF16, {kQRows, tokens});
            Tensor reference_gate(ref_gate.data(), DType::BF16, {kQRows, tokens});
            Tensor reference_k(ref_k.data(), DType::BF16, {kKvRows, tokens});
            Tensor reference_v(ref_v.data(), DType::BF16, {kKvRows, tokens});

            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            ops::attn_input_proj(reference_x, full_device.weight, reference_q, reference_gate,
                                 reference_k, reference_v, policy, reference_arena, ec.dev[0]->stream);
            cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
            failures += ref_q.verify_guards(label + " reference q");
            failures += ref_gate.verify_guards(label + " reference gate");
            failures += ref_k.verify_guards(label + " reference k");
            failures += ref_v.verify_guards(label + " reference v");
            const std::vector<double> expected_q =
                from_device_bf16(ref_q.data(), static_cast<std::size_t>(kQRows) * tokens);
            const std::vector<double> expected_gate =
                from_device_bf16(ref_gate.data(), static_cast<std::size_t>(kQRows) * tokens);
            const std::vector<double> expected_k =
                from_device_bf16(ref_k.data(), static_cast<std::size_t>(kKvRows) * tokens);
            const std::vector<double> expected_v =
                from_device_bf16(ref_v.data(), static_cast<std::size_t>(kKvRows) * tokens);

            // --- (b) the split form -------------------------------------------------------------
            const std::size_t split_capacity = ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
                qtype, policy, tokens, tokens);
            std::array<std::optional<GuardedDeviceBuffer>, 2> split_q;
            std::array<std::optional<GuardedDeviceBuffer>, 2> split_gate;
            std::array<std::optional<GuardedDeviceBuffer>, 2> split_k;
            std::array<std::optional<GuardedDeviceBuffer>, 2> split_v;
            std::array<std::optional<DeviceArena>, 2> arena;
            for (int rank = 0; rank < 2; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                set_device(ec, rank);
                split_q[slot].emplace(static_cast<std::size_t>(kShardQRows) * tokens * sizeof(std::uint16_t));
                split_gate[slot].emplace(static_cast<std::size_t>(kShardQRows) * tokens *
                                        sizeof(std::uint16_t));
                split_k[slot].emplace(static_cast<std::size_t>(kShardKvRows) * tokens * sizeof(std::uint16_t));
                split_v[slot].emplace(static_cast<std::size_t>(kShardKvRows) * tokens * sizeof(std::uint16_t));
                split_q[slot]->fill(0xff);
                split_gate[slot]->fill(0xff);
                split_k[slot]->fill(0xff);
                split_v[slot]->fill(0xff);
                arena[slot].emplace(std::max<std::size_t>(split_capacity, 1));
            }

            const std::array<Tensor, 2> x{Tensor(shard_x[0].p, DType::BF16, {kInputRows, tokens}),
                                          Tensor(shard_x[1].p, DType::BF16, {kInputRows, tokens})};
            const std::array<Weight, 2> w{shard_device[0].weight, shard_device[1].weight};
            const std::array<Tensor, 2> q_out{
                Tensor(split_q[0]->data(), DType::BF16, {kShardQRows, tokens}),
                Tensor(split_q[1]->data(), DType::BF16, {kShardQRows, tokens})};
            const std::array<Tensor, 2> gate_out{
                Tensor(split_gate[0]->data(), DType::BF16, {kShardQRows, tokens}),
                Tensor(split_gate[1]->data(), DType::BF16, {kShardQRows, tokens})};
            const std::array<Tensor, 2> k_out{
                Tensor(split_k[0]->data(), DType::BF16, {kShardKvRows, tokens}),
                Tensor(split_k[1]->data(), DType::BF16, {kShardKvRows, tokens})};
            const std::array<Tensor, 2> v_out{
                Tensor(split_v[0]->data(), DType::BF16, {kShardKvRows, tokens}),
                Tensor(split_v[1]->data(), DType::BF16, {kShardKvRows, tokens})};
            const std::array<ninfer::WorkspaceArena*, 2> workspace{&*arena[0], &*arena[1]};

            retire_staging(ec);
            ops::attn_input_proj_column_parallel(x, w, q_out, gate_out, k_out, v_out, policy,
                                                 workspace, ec);
            synchronize_both(ec);

            std::array<std::vector<double>, 2> observed_q;
            std::array<std::vector<double>, 2> observed_gate;
            std::array<std::vector<double>, 2> observed_k;
            std::array<std::vector<double>, 2> observed_v;
            for (int rank = 0; rank < 2; ++rank) {
                const auto slot          = static_cast<std::size_t>(rank);
                const std::string prefix = label + " rank " + std::to_string(rank);
                set_device(ec, rank);
                failures += split_q[slot]->verify_guards(prefix + " q");
                failures += split_gate[slot]->verify_guards(prefix + " gate");
                failures += split_k[slot]->verify_guards(prefix + " k");
                failures += split_v[slot]->verify_guards(prefix + " v");
                observed_q[slot]    = from_device_bf16(split_q[slot]->data(), static_cast<std::size_t>(kShardQRows) * tokens);
                observed_gate[slot] = from_device_bf16(split_gate[slot]->data(),
                                                       static_cast<std::size_t>(kShardQRows) * tokens);
                observed_k[slot] = from_device_bf16(split_k[slot]->data(), static_cast<std::size_t>(kShardKvRows) * tokens);
                observed_v[slot] = from_device_bf16(split_v[slot]->data(), static_cast<std::size_t>(kShardKvRows) * tokens);

                failures += compare(prefix + " q", observed_q[slot],
                                    extract_rank_block(expected_q, kQRows, kShardQRows, tokens, rank));
                failures += compare(prefix + " gate", observed_gate[slot],
                                    extract_rank_block(expected_gate, kQRows, kShardQRows, tokens, rank));
                failures += compare(prefix + " k", observed_k[slot],
                                    extract_rank_block(expected_k, kKvRows, kShardKvRows, tokens, rank));
                failures += compare(prefix + " v", observed_v[slot],
                                    extract_rank_block(expected_v, kKvRows, kShardKvRows, tokens, rank));
            }

            // Cross-rank structural check: the two ranks must own DIFFERENT head-local output
            // blocks (the column-parallel mirror of test_linear_split.cpp's own check).
            if (observed_q[0] == observed_q[1] || observed_k[0] == observed_k[1]) {
                std::cerr << label
                          << ": both ranks produced identical output blocks, so the column split "
                             "did not actually split\n";
                ++failures;
            }
        }
    }
    return failures;
}

// ================================================================================================
// Split-storage two-weight form (Q4G64_F16S query_key + Q5G64_F16S gate_value).
// ================================================================================================

struct SplitStorageShard {
    qw::PackedWeight q_or_gate;
    qw::PackedWeight k_or_v;
    qw::PackedWeight shard; // concat(q_or_gate, k_or_v)
};

SplitStorageShard make_split_storage_shard(QType qtype, std::uint32_t seed, int rank) {
    SplitStorageShard out;
    out.q_or_gate = make_block(qtype, kShardQRows, kInputRows, seed, rank * kShardQRows);
    out.k_or_v    = make_block(qtype, kShardKvRows, kInputRows, seed, kQRows + rank * kShardKvRows);
    out.shard     = concat_row_blocks(out.q_or_gate, out.k_or_v);
    return out;
}

int run_split_storage_case(const ExecutionContext& ec, std::uint32_t seed) {
    const std::string head = "q4/q5 attn_input split-storage";
    std::cout << head << " query_key/gate_value [" << kSplitRows << ',' << kInputRows << "] -> ["
              << kShardSplitRows << ',' << kInputRows << "]\n";
    int failures = 0;

    qw::PatternedWeightOptions qk_options;
    qk_options.decorrelate_coordinates = true;
    qk_options.row_split_scale         = qw::RowSplitScalePattern::Small;
    qk_options.row_split_codes         = qw::RowSplitCodePattern::Coordinate;
    const qw::PackedWeight full_query_key =
        qw::make_patterned_weight(QType::Q4G64_F16S, kSplitRows, kInputRows, seed, qk_options);
    const qw::PackedWeight full_gate_value =
        qw::make_patterned_weight(QType::Q5G64_F16S, kSplitRows, kInputRows, seed + 1, qk_options);

    std::array<SplitStorageShard, 2> query_key{
        make_split_storage_shard(QType::Q4G64_F16S, seed, 0),
        make_split_storage_shard(QType::Q4G64_F16S, seed, 1)};
    std::array<SplitStorageShard, 2> gate_value{
        make_split_storage_shard(QType::Q5G64_F16S, seed + 1, 0),
        make_split_storage_shard(QType::Q5G64_F16S, seed + 1, 1)};

    for (int rank = 0; rank < 2; ++rank) {
        const auto slot          = static_cast<std::size_t>(rank);
        const std::string label = head + " shard " + std::to_string(rank);
        failures += verify_shard_is_parent_block(label + " query", full_query_key,
                                                 query_key[slot].q_or_gate, rank * kShardQRows, 0);
        failures += verify_shard_is_parent_block(label + " key", full_query_key, query_key[slot].k_or_v,
                                                 kQRows + rank * kShardKvRows, 0);
        failures += verify_shard_is_parent_block(label + " gate", full_gate_value,
                                                 gate_value[slot].q_or_gate, rank * kShardQRows, 0);
        failures += verify_shard_is_parent_block(label + " value", full_gate_value, gate_value[slot].k_or_v,
                                                 kQRows + rank * kShardKvRows, 0);
        // Intra-shard distinctness within each weight (query-vs-key, gate-vs-value).
        failures += verify_shards_are_distinct(
            label + " query-vs-key", query_key[slot].q_or_gate.payload, query_key[slot].k_or_v.payload);
        failures += verify_shards_are_distinct(
            label + " gate-vs-value", gate_value[slot].q_or_gate.payload, gate_value[slot].k_or_v.payload);
    }
    failures +=
        verify_shards_are_distinct(head + " query_key", query_key[0].shard.payload, query_key[1].shard.payload);
    failures += verify_shards_are_distinct(head + " gate_value", gate_value[0].shard.payload,
                                           gate_value[1].shard.payload);
    if (failures != 0) { return failures; }

    set_device(ec, 0);
    DeviceWeight full_qk_device = upload_weight(full_query_key);
    DeviceWeight full_gv_device = upload_weight(full_gate_value);
    std::array<DeviceWeight, 2> qk_device;
    std::array<DeviceWeight, 2> gv_device;
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        qk_device[static_cast<std::size_t>(rank)] = upload_weight(query_key[static_cast<std::size_t>(rank)].shard);
        gv_device[static_cast<std::size_t>(rank)] = upload_weight(gate_value[static_cast<std::size_t>(rank)].shard);
    }

    // Route sweep: T=1 exercises the gemv edge on the tp1 reference kernel (irrelevant to the
    // shard, which always routes through the row-generic grouped-MMA kernel -- see
    // attn_input_proj.h's design note); T around the BN=64 tile boundary (63/64/65) proves the
    // partial-tile masking on the shard path; T=17/21 cross the tp1 reference's own small-T/MMA
    // route boundaries.
    const std::vector<std::int32_t> tokens_sweep{1, 2, 16, 17, 21, 48, 63, 64, 65, 128};

    for (const std::int32_t tokens : tokens_sweep) {
        const std::string label = head + " T=" + std::to_string(tokens);
        std::vector<float> activation(static_cast<std::size_t>(kInputRows) * tokens);
        fill_uniform(activation, seed * 37u + static_cast<std::uint32_t>(tokens), -1.0F, 1.0F);
        round_to_bf16(activation);

        set_device(ec, 0);
        DeviceBuffer full_x = to_device_bf16(activation);
        std::array<DeviceBuffer, 2> shard_x;
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            shard_x[static_cast<std::size_t>(rank)] = to_device_bf16(activation);
        }

        // --- (a) reference: tp1 two-weight kernel, whole weights, device 0 ---------------------
        set_device(ec, 0);
        GuardedDeviceBuffer ref_q(static_cast<std::size_t>(kQRows) * tokens * sizeof(std::uint16_t));
        GuardedDeviceBuffer ref_gate(static_cast<std::size_t>(kQRows) * tokens * sizeof(std::uint16_t));
        GuardedDeviceBuffer ref_k(static_cast<std::size_t>(kKvRows) * tokens * sizeof(std::uint16_t));
        GuardedDeviceBuffer ref_v(static_cast<std::size_t>(kKvRows) * tokens * sizeof(std::uint16_t));
        ref_q.fill(0xff);
        ref_gate.fill(0xff);
        ref_k.fill(0xff);
        ref_v.fill(0xff);
        Tensor reference_x(full_x.p, DType::BF16, {kInputRows, tokens});
        Tensor reference_q(ref_q.data(), DType::BF16, {kQRows, tokens});
        Tensor reference_gate(ref_gate.data(), DType::BF16, {kQRows, tokens});
        Tensor reference_k(ref_k.data(), DType::BF16, {kKvRows, tokens});
        Tensor reference_v(ref_v.data(), DType::BF16, {kKvRows, tokens});

        DeviceArena reference_workspace(
            ops::q4_q5_attn_input_proj_workspace_capacity_bytes(tokens, tokens));

        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ops::attn_input_proj(reference_x, full_qk_device.weight, full_gv_device.weight, reference_q,
                             reference_gate, reference_k, reference_v, reference_workspace,
                             ec.dev[0]->stream);
        cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
        failures += ref_q.verify_guards(label + " reference q");
        failures += ref_gate.verify_guards(label + " reference gate");
        failures += ref_k.verify_guards(label + " reference k");
        failures += ref_v.verify_guards(label + " reference v");
        const std::vector<double> expected_q = from_device_bf16(ref_q.data(), static_cast<std::size_t>(kQRows) * tokens);
        const std::vector<double> expected_gate =
            from_device_bf16(ref_gate.data(), static_cast<std::size_t>(kQRows) * tokens);
        const std::vector<double> expected_k = from_device_bf16(ref_k.data(), static_cast<std::size_t>(kKvRows) * tokens);
        const std::vector<double> expected_v = from_device_bf16(ref_v.data(), static_cast<std::size_t>(kKvRows) * tokens);

        // --- (b) the split form -----------------------------------------------------------------
        std::array<std::optional<GuardedDeviceBuffer>, 2> split_q;
        std::array<std::optional<GuardedDeviceBuffer>, 2> split_gate;
        std::array<std::optional<GuardedDeviceBuffer>, 2> split_k;
        std::array<std::optional<GuardedDeviceBuffer>, 2> split_v;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            split_q[slot].emplace(static_cast<std::size_t>(kShardQRows) * tokens * sizeof(std::uint16_t));
            split_gate[slot].emplace(static_cast<std::size_t>(kShardQRows) * tokens * sizeof(std::uint16_t));
            split_k[slot].emplace(static_cast<std::size_t>(kShardKvRows) * tokens * sizeof(std::uint16_t));
            split_v[slot].emplace(static_cast<std::size_t>(kShardKvRows) * tokens * sizeof(std::uint16_t));
            split_q[slot]->fill(0xff);
            split_gate[slot]->fill(0xff);
            split_k[slot]->fill(0xff);
            split_v[slot]->fill(0xff);
        }

        const std::array<Tensor, 2> x{Tensor(shard_x[0].p, DType::BF16, {kInputRows, tokens}),
                                      Tensor(shard_x[1].p, DType::BF16, {kInputRows, tokens})};
        const std::array<Weight, 2> qk_weight{qk_device[0].weight, qk_device[1].weight};
        const std::array<Weight, 2> gv_weight{gv_device[0].weight, gv_device[1].weight};
        const std::array<Tensor, 2> q_out{Tensor(split_q[0]->data(), DType::BF16, {kShardQRows, tokens}),
                                          Tensor(split_q[1]->data(), DType::BF16, {kShardQRows, tokens})};
        const std::array<Tensor, 2> gate_out{
            Tensor(split_gate[0]->data(), DType::BF16, {kShardQRows, tokens}),
            Tensor(split_gate[1]->data(), DType::BF16, {kShardQRows, tokens})};
        const std::array<Tensor, 2> k_out{Tensor(split_k[0]->data(), DType::BF16, {kShardKvRows, tokens}),
                                          Tensor(split_k[1]->data(), DType::BF16, {kShardKvRows, tokens})};
        const std::array<Tensor, 2> v_out{Tensor(split_v[0]->data(), DType::BF16, {kShardKvRows, tokens}),
                                          Tensor(split_v[1]->data(), DType::BF16, {kShardKvRows, tokens})};

        std::array<std::optional<DeviceArena>, 2> split_workspace;
        const std::size_t shard_workspace =
            ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
                QType::Q4G64_F16S, ops::LinearPolicy::A16Only, tokens, tokens);
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            split_workspace[static_cast<std::size_t>(rank)].emplace(shard_workspace);
        }
        const std::array<WorkspaceArena*, 2> workspace{
            &*split_workspace[0], &*split_workspace[1]};

        retire_staging(ec);
        ops::attn_input_proj_column_parallel(x, qk_weight, gv_weight, q_out, gate_out, k_out,
                                             v_out, workspace, ec);
        synchronize_both(ec);

        std::array<std::vector<double>, 2> observed_q;
        std::array<std::vector<double>, 2> observed_gate;
        std::array<std::vector<double>, 2> observed_k;
        std::array<std::vector<double>, 2> observed_v;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot          = static_cast<std::size_t>(rank);
            const std::string prefix = label + " rank " + std::to_string(rank);
            set_device(ec, rank);
            failures += split_q[slot]->verify_guards(prefix + " q");
            failures += split_gate[slot]->verify_guards(prefix + " gate");
            failures += split_k[slot]->verify_guards(prefix + " k");
            failures += split_v[slot]->verify_guards(prefix + " v");
            observed_q[slot]    = from_device_bf16(split_q[slot]->data(), static_cast<std::size_t>(kShardQRows) * tokens);
            observed_gate[slot] = from_device_bf16(split_gate[slot]->data(), static_cast<std::size_t>(kShardQRows) * tokens);
            observed_k[slot]    = from_device_bf16(split_k[slot]->data(), static_cast<std::size_t>(kShardKvRows) * tokens);
            observed_v[slot]    = from_device_bf16(split_v[slot]->data(), static_cast<std::size_t>(kShardKvRows) * tokens);

            failures += compare(prefix + " q", observed_q[slot],
                                extract_rank_block(expected_q, kQRows, kShardQRows, tokens, rank));
            failures += compare(prefix + " gate", observed_gate[slot],
                                extract_rank_block(expected_gate, kQRows, kShardQRows, tokens, rank));
            failures += compare(prefix + " k", observed_k[slot],
                                extract_rank_block(expected_k, kKvRows, kShardKvRows, tokens, rank));
            failures += compare(prefix + " v", observed_v[slot],
                                extract_rank_block(expected_v, kKvRows, kShardKvRows, tokens, rank));
        }

        if (observed_q[0] == observed_q[1] || observed_k[0] == observed_k[1]) {
            std::cerr << label
                      << ": both ranks produced identical output blocks, so the column split did "
                         "not actually split\n";
            ++failures;
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------------------------
// Host-only registry probe: attn_input_proj_column_parallel_workspace_capacity_bytes() runs
// NVFP4's shape resolver without touching a device, and must reject every other qtype. Pure host
// code, so it runs BEFORE the device checks like every sibling split suite's own registry probe.
// ---------------------------------------------------------------------------------------------
int verify_registry() {
    int failures = 0;
    for (const ops::LinearPolicy policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA4}) {
        for (const std::int32_t tokens : {1, 2, 48, 1024}) {
            try {
                (void)ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
                    QType::NVFP4, policy, tokens, tokens);
            } catch (const std::exception& error) {
                std::cerr << "registry: NVFP4 column-parallel workspace " << policy_name(policy)
                          << " T=" << tokens << " rejected: " << error.what() << '\n';
                ++failures;
            }
        }
    }
    // FP8's fused column-parallel shard.
    for (const ops::LinearPolicy policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8}) {
        for (const std::int32_t tokens : {1, 2, 48, 1024}) {
            try {
                (void)ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
                    QType::FP8_E4M3FN_ROW_BF16S, policy, tokens, tokens);
            } catch (const std::exception& error) {
                std::cerr << "registry: FP8 column-parallel workspace " << policy_name(policy)
                          << " T=" << tokens << " rejected: " << error.what() << '\n';
                ++failures;
            }
        }
    }
    // FP8 rejects AllowA4 (not a policy it admits).
    {
        bool threw = false;
        try {
            (void)ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16S, ops::LinearPolicy::AllowA4, 1, 1);
        } catch (const std::exception&) { threw = true; }
        if (!threw) {
            std::cerr << "registry: FP8 AllowA4 was admitted but must not be\n";
            ++failures;
        }
    }
    for (const QType qtype : {QType::BF16_CTRL, QType::W8G32_F16S}) {
        bool threw = false;
        try {
            (void)ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
                qtype, ops::LinearPolicy::A16Only, 1, 1);
        } catch (const std::exception&) { threw = true; }
        if (!threw) {
            std::cerr << "registry: qtype " << static_cast<int>(qtype)
                      << " was admitted by the fused column-parallel workspace query but must not be\n";
            ++failures;
        }
    }
    // GGML_K is the native Qwen3.8 Q4_K_M fused parent used by the V100 Duo profile. The
    // Q4/Q5 entries are the groupwise-int split-storage form used by the Qwen3.6 profile.
    for (const QType qtype : {QType::GGML_K, QType::Q4G64_F16S, QType::Q5G64_F16S}) {
        try {
            (void)ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
                qtype, ops::LinearPolicy::A16Only, 1, 1);
        } catch (const std::exception& error) {
            std::cerr << "registry: supported qtype " << static_cast<int>(qtype)
                      << " was rejected: " << error.what() << '\n';
            ++failures;
        }
    }
    std::cout << (failures ? "FAIL" : "OK") << " registry\n";
    return failures;
}

// Rejection cases the split forms own: only the pair can see them.
int verify_split_rejections(const ExecutionContext& ec) {
    int failures            = 0;
    const auto expect_throw = [&](const char* what, auto&& body) {
        try {
            body();
        } catch (const std::invalid_argument&) {
            return;
        } catch (const std::exception& error) {
            std::cerr << "split rejection " << what << ": wrong exception: " << error.what() << '\n';
            ++failures;
            return;
        }
        std::cerr << "split rejection " << what << ": accepted an invalid pair\n";
        ++failures;
    };

    set_device(ec, 0);
    DeviceBuffer x0(static_cast<std::size_t>(kInputRows) * 2 * sizeof(std::uint16_t));
    DeviceBuffer q0(static_cast<std::size_t>(kShardQRows) * 2 * sizeof(std::uint16_t));
    DeviceBuffer gate0(static_cast<std::size_t>(kShardQRows) * 2 * sizeof(std::uint16_t));
    DeviceBuffer k0(static_cast<std::size_t>(kShardKvRows) * 2 * sizeof(std::uint16_t));
    DeviceBuffer v0(static_cast<std::size_t>(kShardKvRows) * 2 * sizeof(std::uint16_t));
    set_device(ec, 1);
    DeviceBuffer x1(static_cast<std::size_t>(kInputRows) * 2 * sizeof(std::uint16_t));
    DeviceBuffer q1(static_cast<std::size_t>(kShardQRows) * 2 * sizeof(std::uint16_t));
    DeviceBuffer gate1(static_cast<std::size_t>(kShardQRows) * 2 * sizeof(std::uint16_t));
    DeviceBuffer k1(static_cast<std::size_t>(kShardKvRows) * 2 * sizeof(std::uint16_t));
    DeviceBuffer v1(static_cast<std::size_t>(kShardKvRows) * 2 * sizeof(std::uint16_t));

    Weight fake{};
    fake.qtype = QType::NVFP4;
    fake.n     = kShardFusedRows;
    fake.k     = kInputRows;

    // Mismatched token counts on the two ranks.
    expect_throw("token count", [&] {
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kInputRows, 2}),
                                      Tensor(x1.p, DType::BF16, {kInputRows, 1})};
        const std::array<Tensor, 2> q{Tensor(q0.p, DType::BF16, {kShardQRows, 2}),
                                      Tensor(q1.p, DType::BF16, {kShardQRows, 1})};
        const std::array<Tensor, 2> gate{Tensor(gate0.p, DType::BF16, {kShardQRows, 2}),
                                         Tensor(gate1.p, DType::BF16, {kShardQRows, 1})};
        const std::array<Tensor, 2> k{Tensor(k0.p, DType::BF16, {kShardKvRows, 2}),
                                      Tensor(k1.p, DType::BF16, {kShardKvRows, 1})};
        const std::array<Tensor, 2> v{Tensor(v0.p, DType::BF16, {kShardKvRows, 2}),
                                      Tensor(v1.p, DType::BF16, {kShardKvRows, 1})};
        ops::attn_input_proj_column_parallel(x, {fake, fake}, q, gate, k, v, ec);
    });

    // Disagreeing K.
    expect_throw("column K", [&] {
        Weight other = fake;
        other.k      = kInputRows / 2;
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kInputRows, 1}),
                                      Tensor(x1.p, DType::BF16, {kInputRows / 2, 1})};
        const std::array<Tensor, 2> q{Tensor(q0.p, DType::BF16, {kShardQRows, 1}),
                                      Tensor(q1.p, DType::BF16, {kShardQRows, 1})};
        const std::array<Tensor, 2> gate{Tensor(gate0.p, DType::BF16, {kShardQRows, 1}),
                                         Tensor(gate1.p, DType::BF16, {kShardQRows, 1})};
        const std::array<Tensor, 2> k{Tensor(k0.p, DType::BF16, {kShardKvRows, 1}),
                                      Tensor(k1.p, DType::BF16, {kShardKvRows, 1})};
        const std::array<Tensor, 2> v{Tensor(v0.p, DType::BF16, {kShardKvRows, 1}),
                                      Tensor(v1.p, DType::BF16, {kShardKvRows, 1})};
        ops::attn_input_proj_column_parallel(x, {fake, other}, q, gate, k, v, ec);
    });

    // A single-device context is not a split context.
    expect_throw("tp1 context", [&] {
        const ExecutionContext single({0});
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kInputRows, 1}),
                                      Tensor(x1.p, DType::BF16, {kInputRows, 1})};
        const std::array<Tensor, 2> q{Tensor(q0.p, DType::BF16, {kShardQRows, 1}),
                                      Tensor(q1.p, DType::BF16, {kShardQRows, 1})};
        const std::array<Tensor, 2> gate{Tensor(gate0.p, DType::BF16, {kShardQRows, 1}),
                                         Tensor(gate1.p, DType::BF16, {kShardQRows, 1})};
        const std::array<Tensor, 2> k{Tensor(k0.p, DType::BF16, {kShardKvRows, 1}),
                                      Tensor(k1.p, DType::BF16, {kShardKvRows, 1})};
        const std::array<Tensor, 2> v{Tensor(v0.p, DType::BF16, {kShardKvRows, 1}),
                                      Tensor(v1.p, DType::BF16, {kShardKvRows, 1})};
        ops::attn_input_proj_column_parallel(x, {fake, fake}, q, gate, k, v, single);
    });

    std::cout << (failures ? "FAIL" : "OK") << " split rejections\n";
    return failures;
}

} // namespace

int main() {
    int failures = verify_registry();
    if (failures != 0) {
        std::cout << "FAIL attn_input_proj split (registry)\n";
        return 1;
    }

    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: split attn_input_proj parity requires two CUDA devices, found "
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

    failures += verify_split_rejections(ec);
    // T sweep: T=1 (decode edge), small-T/MMA frontiers, T=128 (W4A4 MMA under AllowA4), T=1024 (a
    // multiple of 256 -- the sole route into the NVFP4 W4A4 TMA kernel, exercised on the shard
    // TMA descriptor as well as the tp1 one).
    failures += run_fused_case(ec, QType::NVFP4, 41u, "nvfp4 attn_input fused",
                               {1, 2, 5, 8, 17, 32, 48, 128, 1024},
                               {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA4});
    // FP8's own tp2 column shard, wired as a TRUE split (the fused kernel family is
    // Geometry-templated, same as NVFP4 -- see attn_input_proj.h's design note). T sweep: 1 the
    // decode edge; 2/8/10 small-T (kFp8LinearSmallTMax<AttnInput>=11); 11 the AllowA8 route's own
    // A8 crossover; 32/48/128/1024 beyond it.
    failures += run_fused_case(ec, QType::FP8_E4M3FN_ROW_BF16S, 46u, "fp8 attn_input fused",
                               {1, 2, 8, 10, 11, 32, 48, 128, 1024},
                               {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8});
    failures += run_split_storage_case(ec, 43u);

    std::cout << (failures ? "FAIL" : "OK") << " attn_input_proj split\n";
    return failures ? 1 : 0;
}
