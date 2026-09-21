// Two-device parity suite for ops::gdn_input_proj_column_parallel and
// ops::gdn_gating_proj_column_parallel (include/ninfer/ops/{gdn_input_proj,gdn_gating_proj}.h),
// the fourth op-split sibling. Read test_linear_split.cpp's header comment first, then
// test_attn_input_proj_split.cpp's (the nearest sibling: fused multi-section column split with
// per-section parity and shard-offset derivation); this file copies both disciplines and
// documents what differs.
//
// Every case REQUIRES two CUDA devices driven from ONE process; with fewer than two visible
// devices the suite reports the repository's skip code (77) instead of failing.
//
// GDN INPUT_PROJECTION SECTION ORDER IS Q | K | V | Z. (An early design table entry read
// "qkv | z"; that grouping is wrong.) Confirmed against bindings.cpp's `plan_for` for
// "gdn/query_key_value_z" (four independent append_column_block calls, in that order) and
// src/targets/qwen3_6_27b/impl/config.h (key_dim=2048, value_dim=6144). Parent [16384,5120] =
// Q[0,2048) | K[2048,4096) | V[4096,10240) | Z[10240,16384). Rank r's fused shard weight is the
// CONCATENATION, in the SAME Q|K|V|Z order, of rank r's own head-local slice of every section: a
// standalone [8192,5120] tensor with Q at shard-local offset 0 (1024 rows), K at 1024 (1024 rows),
// V at 2048 (3072 rows), Z at 5120 (3072 rows).
//
// OUTPUT CONTRACT (see include/ninfer/ops/gdn_input_proj.h's design note): the Op keeps the tp1
// Op's own 2-tensor packing (qkv + z), Geometry-halved -- NOT four separate q/k/v/z tensors. Each
// rank's qkv[5120,T] holds shard-local Q[0,1024) | K[1024,2048) | V[2048,5120); z[3072,T] holds
// the rank's Z directly.
//
// THREE FORMS ARE REGISTERED: NVFP4 fused single-parent, FP8_E4M3FN_ROW_BF16S fused single-parent
// (FP8 coverage is mandatory here -- it is the Qwen38Nvfp4 profile's flagship binding for this
// object), and Q4G64_F16S/Q5G64_F16S split-storage two-weight form (query_key + value_z).
// gdn_input_proj's own qkv/z row-offset convention (0/2048/4096 in the reference, 0/1024/2048 in
// the shard) is IDENTICAL across all three forms -- the fused and split-storage tp1 Ops already
// pack their qkv/z outputs the same way (see gdn_input_proj's tp1 two-weight overload's own
// kQkRows/kValueRows constants), so one extract_block() helper serves every case.
//
// GDN_GATING_PROJ INTERIOR ORDER: VERIFIED head-major (row = qk_group*3 + component) against the
// shipped GDN core's own `head_map::qk_head(h_v) = h_v / (H_v/H_qk)` in
// src/ops/linear_attention/gated_delta_net/common.cuh (consumed by recurrent.cuh and by
// chunked/{prepare_wy_wu,output}.cuh, which index g/beta with the same flat t*H_v + h_v layout);
// the test-side oracle tests/ops/gdn_ref.h::qk_head mirrors it exactly. See also the design note
// in include/ninfer/ops/gdn_gating_proj.h. [0,24)/[24,48) each hold a clean set of 8 complete qk
// groups. gdn_gating_proj is BF16_CTRL only (never quantized in any qwen3_8_27b weights
// profile).
//
// TOLERANCE. 2 BF16 ulp of the largest output in the tensor (gross_relative_to_max_reference) plus
// a matching relative-L2 leg, exactly test_linear_split.cpp's criterion and rationale -- applied
// PER SECTION (q/k/v/z separately for gdn_input_proj; g/beta for gdn_gating_proj).
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"

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
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace qw = ninfer::test::quantized_weight;

namespace {

constexpr double kBf16Ulp = 1.0 / 256.0;

// gdn_input_proj geometry (config.key_dim=2048, config.value_dim=6144, hidden=5120).
constexpr std::int32_t kInputRows       = 5120;
constexpr std::int32_t kKeyRows         = 2048; // tp1 Q and K section rows (each)
constexpr std::int32_t kValueRows       = 6144; // tp1 V and Z section rows (each)
constexpr std::int32_t kFusedRows       = 2 * kKeyRows + 2 * kValueRows;      // 16384
constexpr std::int32_t kQkvRows         = 2 * kKeyRows + kValueRows;         // 10240 (Q|K|V)
constexpr std::int32_t kQkRows          = 2 * kKeyRows;                      // 4096, Q4 query_key
constexpr std::int32_t kValueZRows      = 2 * kValueRows;                   // 12288, Q5 value_z
constexpr std::int32_t kShardKeyRows    = kKeyRows / 2;                     // 1024
constexpr std::int32_t kShardValueRows  = kValueRows / 2;                   // 3072
constexpr std::int32_t kShardFusedRows  = 2 * kShardKeyRows + 2 * kShardValueRows; // 8192
constexpr std::int32_t kShardQkvRows    = 2 * kShardKeyRows + kShardValueRows;     // 5120
constexpr std::int32_t kShardQkRows     = 2 * kShardKeyRows;                       // 2048
constexpr std::int32_t kShardValueZRows = 2 * kShardValueRows;                     // 6144

// gdn_gating_proj geometry.
constexpr std::int32_t kGatingHidden     = 5120;
constexpr std::int32_t kGatingHeads      = 48;
constexpr std::int32_t kGatingShardHeads = 24;

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

// Single-origin generation for one contiguous row block, decorrelate_coordinates=true -- the
// section strides here (1024, 2048, 3072, 4096, ...) annihilate the fixture's un-decorrelated
// affine pattern, same rationale as every sibling split suite.
qw::PackedWeight make_block(QType qtype, std::int32_t n, std::int32_t k, std::uint32_t seed,
                            std::int32_t row_origin) {
    qw::PatternedWeightOptions options;
    options.row_origin              = row_origin;
    options.decorrelate_coordinates = true;
    if (qtype == QType::NVFP4) {
        options.weight_scale_divisor = 0.125F;
        options.input_scale_divisor  = 3.5F;
    } else if (qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        // No divisors for FP8 -- the fixture rejects nonzero ones for this format.
    } else {
        options.row_split_scale = qw::RowSplitScalePattern::Small;
        options.row_split_codes = qw::RowSplitCodePattern::Coordinate;
    }
    return qw::make_patterned_weight(qtype, n, k, seed, options);
}

// Splices two independently generated, same-(qtype,k) row blocks into one standalone tensor whose
// logical rows are [top's rows][bottom's rows] -- copied verbatim from
// test_attn_input_proj_split.cpp's concat_row_blocks() (itself copied from
// test_linear_swiglu_split.cpp), whose doc comment explains why this is valid for every layout the
// fixture produces.
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
    // FP8's RowScale metadata denormalizes n into scale_ne[0]/scale_nb[1..3] (one BF16 scale per
    // row, scale_ne[0]==n) -- attn_input_proj's own concat_row_blocks originally lacked this
    // fix-up because it had never been used for FP8; this suite was the first to splice FP8
    // blocks, and test_attn_input_proj_split.cpp now carries the same correction.
    if (top.weight.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        combined.weight.scale_ne[0] = combined.weight.n;
        const std::int64_t stride  = static_cast<std::int64_t>(combined.weight.n) * 2;
        combined.weight.scale_nb[1] = stride;
        combined.weight.scale_nb[2] = stride;
        combined.weight.scale_nb[3] = stride;
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

// op_tester.h's from_device_f32 only accepts a DeviceBuffer&; every FP32 output here lives in a
// GuardedDeviceBuffer's raw pointer, so this local overload mirrors from_device_bf16's own
// pointer-taking sibling.
std::vector<double> from_device_f32_ptr(const void* device, std::size_t n) {
    const std::vector<float> values = from_device<float>(device, n);
    return {values.begin(), values.end()};
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

int verify_fp8_sample(const std::string& label, const qw::PackedWeight& weight,
                      const std::vector<float>& activation, const std::vector<double>& output,
                      std::int32_t output_stride, std::int32_t output_offset,
                      std::int32_t weight_offset, std::int32_t section_rows, std::int32_t tokens) {
    std::vector<double> actual, expected;
    for (const std::int32_t token : {0, tokens / 2, tokens - 1}) {
        for (const std::int32_t row : {0, section_rows / 2, section_rows - 1}) {
            actual.push_back(output[static_cast<std::size_t>(token) * output_stride +
                                    output_offset + row]);
            expected.push_back(qw::dot_fp64(weight, weight_offset + row,
                                            activation.data() + static_cast<std::size_t>(token) *
                                                                    kInputRows,
                                            kInputRows));
        }
    }
    return compare(label + " FP64", actual, expected);
}

// Extracts, per token, `block_rows` rows starting at `row_offset` from a `stride_rows`-row packed
// buffer. One helper serves every section of every form (reference and shard alike), since the
// qkv/z row-offset convention (0/2048/4096 tp1, 0/1024/2048 shard) is identical across NVFP4, FP8,
// and Q4/Q5 -- see the file header comment.
std::vector<double> extract_block(const std::vector<double>& buffer, std::int32_t stride_rows,
                                  std::int32_t row_offset, std::int32_t block_rows,
                                  std::int32_t tokens) {
    std::vector<double> block(static_cast<std::size_t>(block_rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t source = static_cast<std::size_t>(token) * stride_rows + row_offset;
        std::copy(buffer.begin() + static_cast<std::ptrdiff_t>(source),
                 buffer.begin() + static_cast<std::ptrdiff_t>(source + block_rows),
                 block.begin() + static_cast<std::ptrdiff_t>(token) * block_rows);
    }
    return block;
}

// ================================================================================================
// gdn_input_proj: fused single-parent form (NVFP4 or FP8).
// ================================================================================================

struct FusedShard {
    qw::PackedWeight q;
    qw::PackedWeight k;
    qw::PackedWeight v;
    qw::PackedWeight z;
    qw::PackedWeight shard; // concat(q, k, v, z), the real ShardPlan section order
};

FusedShard make_fused_shard(QType qtype, std::uint32_t seed, int rank) {
    FusedShard out;
    out.q = make_block(qtype, kShardKeyRows, kInputRows, seed, rank * kShardKeyRows);
    out.k = make_block(qtype, kShardKeyRows, kInputRows, seed, kKeyRows + rank * kShardKeyRows);
    out.v = make_block(qtype, kShardValueRows, kInputRows, seed,
                       2 * kKeyRows + rank * kShardValueRows);
    out.z = make_block(qtype, kShardValueRows, kInputRows, seed,
                       2 * kKeyRows + kValueRows + rank * kShardValueRows);
    out.shard = concat_row_blocks(concat_row_blocks(concat_row_blocks(out.q, out.k), out.v), out.z);
    return out;
}

int run_fused_case(const ExecutionContext& ec, QType qtype,
                   const std::vector<ops::LinearPolicy>& policies, std::uint32_t seed) {
    const std::string head = (qtype == QType::NVFP4 ? std::string("nvfp4") : std::string("fp8")) +
                             " gdn_input fused";
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
        const auto slot          = static_cast<std::size_t>(rank);
        const std::string label = head + " shard " + std::to_string(rank);
        failures += verify_shard_is_parent_block(label + " q", full, shard[slot].q,
                                                 rank * kShardKeyRows, 0);
        failures += verify_shard_is_parent_block(label + " k", full, shard[slot].k,
                                                 kKeyRows + rank * kShardKeyRows, 0);
        failures += verify_shard_is_parent_block(label + " v", full, shard[slot].v,
                                                 2 * kKeyRows + rank * kShardValueRows, 0);
        failures += verify_shard_is_parent_block(
            label + " z", full, shard[slot].z, 2 * kKeyRows + kValueRows + rank * kShardValueRows,
            0);
        // Intra-shard distinctness: equal-size aliasing candidates. q-vs-k (both 1024 rows) and
        // v-vs-z (both 3072 rows) are exactly the pairs a row-mixup could confuse silently.
        failures +=
            verify_shards_are_distinct(label + " q-vs-k", shard[slot].q.payload, shard[slot].k.payload);
        failures +=
            verify_shards_are_distinct(label + " v-vs-z", shard[slot].v.payload, shard[slot].z.payload);
    }
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

    // T sweep: T=1 (decode edge), small-T/MMA frontiers, T=128 (W4A4/A8 route under the permissive
    // policy), T=1024 (a multiple of 256 -- the sole route into the NVFP4 W4A4 TMA kernel, and the
    // shard's own TMA descriptor per the w4a4.cu/w4a4_tma.cu changes).
    const std::vector<std::int32_t> tokens_sweep{1, 2, 4, 5, 8, 17, 32, 48, 128, 1024};

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
            GuardedDeviceBuffer ref_qkv(static_cast<std::size_t>(kQkvRows) * tokens *
                                       sizeof(std::uint16_t));
            GuardedDeviceBuffer ref_z(static_cast<std::size_t>(kValueRows) * tokens *
                                      sizeof(std::uint16_t));
            ref_qkv.fill(0xff);
            ref_z.fill(0xff);
            const std::size_t reference_capacity = ops::gdn_input_proj_workspace_capacity_bytes(
                qtype, kFusedRows, kInputRows, policy, tokens, tokens);
            DeviceArena reference_arena(std::max<std::size_t>(reference_capacity, 1));
            Tensor reference_x(full_x.p, DType::BF16, {kInputRows, tokens});
            Tensor reference_qkv(ref_qkv.data(), DType::BF16, {kQkvRows, tokens});
            Tensor reference_z(ref_z.data(), DType::BF16, {kValueRows, tokens});

            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            ops::gdn_input_proj(reference_x, full_device.weight, reference_qkv, reference_z, policy,
                                reference_arena, ec.dev[0]->stream);
            cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
            failures += ref_qkv.verify_guards(label + " reference qkv");
            failures += ref_z.verify_guards(label + " reference z");
            const std::vector<double> expected_qkv =
                from_device_bf16(ref_qkv.data(), static_cast<std::size_t>(kQkvRows) * tokens);
            const std::vector<double> expected_z =
                from_device_bf16(ref_z.data(), static_cast<std::size_t>(kValueRows) * tokens);

            // --- (b) the split form -------------------------------------------------------------
            const std::size_t split_capacity =
                ops::gdn_input_proj_column_parallel_workspace_capacity_bytes(qtype, policy, tokens,
                                                                             tokens);
            std::array<std::optional<GuardedDeviceBuffer>, 2> split_qkv;
            std::array<std::optional<GuardedDeviceBuffer>, 2> split_z;
            std::array<std::optional<DeviceArena>, 2> arena;
            for (int rank = 0; rank < 2; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                set_device(ec, rank);
                split_qkv[slot].emplace(static_cast<std::size_t>(kShardQkvRows) * tokens *
                                       sizeof(std::uint16_t));
                split_z[slot].emplace(static_cast<std::size_t>(kShardValueRows) * tokens *
                                     sizeof(std::uint16_t));
                split_qkv[slot]->fill(0xff);
                split_z[slot]->fill(0xff);
                arena[slot].emplace(std::max<std::size_t>(split_capacity, 1));
            }

            const std::array<Tensor, 2> x{Tensor(shard_x[0].p, DType::BF16, {kInputRows, tokens}),
                                          Tensor(shard_x[1].p, DType::BF16, {kInputRows, tokens})};
            const std::array<Weight, 2> w{shard_device[0].weight, shard_device[1].weight};
            const std::array<Tensor, 2> qkv_out{
                Tensor(split_qkv[0]->data(), DType::BF16, {kShardQkvRows, tokens}),
                Tensor(split_qkv[1]->data(), DType::BF16, {kShardQkvRows, tokens})};
            const std::array<Tensor, 2> z_out{
                Tensor(split_z[0]->data(), DType::BF16, {kShardValueRows, tokens}),
                Tensor(split_z[1]->data(), DType::BF16, {kShardValueRows, tokens})};
            const std::array<ninfer::WorkspaceArena*, 2> workspace{&*arena[0], &*arena[1]};

            retire_staging(ec);
            ops::gdn_input_proj_column_parallel(x, w, qkv_out, z_out, policy, workspace, ec);
            synchronize_both(ec);

            std::array<std::vector<double>, 2> observed_qkv;
            std::array<std::vector<double>, 2> observed_z;
            for (int rank = 0; rank < 2; ++rank) {
                const auto slot          = static_cast<std::size_t>(rank);
                const std::string prefix = label + " rank " + std::to_string(rank);
                set_device(ec, rank);
                failures += split_qkv[slot]->verify_guards(prefix + " qkv");
                failures += split_z[slot]->verify_guards(prefix + " z");
                observed_qkv[slot] = from_device_bf16(split_qkv[slot]->data(),
                                                      static_cast<std::size_t>(kShardQkvRows) * tokens);
                observed_z[slot] = from_device_bf16(split_z[slot]->data(),
                                                    static_cast<std::size_t>(kShardValueRows) * tokens);

                if (qtype == QType::FP8_E4M3FN_ROW_BF16S && tokens > 1 && tokens <= 32) {
                    const auto& packed = shard[slot].shard;
                    failures += verify_fp8_sample(prefix + " q", packed, activation,
                                                  observed_qkv[slot], kShardQkvRows, 0, 0,
                                                  kShardKeyRows, tokens);
                    failures += verify_fp8_sample(prefix + " k", packed, activation,
                                                  observed_qkv[slot], kShardQkvRows, kShardKeyRows,
                                                  kShardKeyRows, kShardKeyRows, tokens);
                    failures += verify_fp8_sample(prefix + " v", packed, activation,
                                                  observed_qkv[slot], kShardQkvRows,
                                                  2 * kShardKeyRows, 2 * kShardKeyRows,
                                                  kShardValueRows, tokens);
                    failures += verify_fp8_sample(prefix + " z", packed, activation,
                                                  observed_z[slot], kShardValueRows, 0,
                                                  kShardQkvRows, kShardValueRows, tokens);
                }

                failures += compare(
                    prefix + " q",
                    extract_block(observed_qkv[slot], kShardQkvRows, 0, kShardKeyRows, tokens),
                    extract_block(expected_qkv, kQkvRows, rank * kShardKeyRows, kShardKeyRows,
                                 tokens));
                failures += compare(
                    prefix + " k",
                    extract_block(observed_qkv[slot], kShardQkvRows, kShardKeyRows, kShardKeyRows,
                                 tokens),
                    extract_block(expected_qkv, kQkvRows, kKeyRows + rank * kShardKeyRows,
                                 kShardKeyRows, tokens));
                failures += compare(
                    prefix + " v",
                    extract_block(observed_qkv[slot], kShardQkvRows, 2 * kShardKeyRows,
                                 kShardValueRows, tokens),
                    extract_block(expected_qkv, kQkvRows, 2 * kKeyRows + rank * kShardValueRows,
                                 kShardValueRows, tokens));
                failures += compare(prefix + " z", observed_z[slot],
                                    extract_block(expected_z, kValueRows,
                                                 rank * kShardValueRows, kShardValueRows, tokens));
            }

            // Cross-rank structural check: the two ranks must own DIFFERENT head-local output
            // blocks (the column-parallel mirror of test_linear_split.cpp's own check).
            if (observed_qkv[0] == observed_qkv[1]) {
                std::cerr << label
                          << ": both ranks produced identical qkv output, so the column split did "
                             "not actually split\n";
                ++failures;
            }
            // Intra-rank equal-size aliasing: V (within qkv) must not collapse into Z (separate
            // tensor, same row count).
            for (int rank = 0; rank < 2; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                const auto v_block =
                    extract_block(observed_qkv[slot], kShardQkvRows, 2 * kShardKeyRows,
                                 kShardValueRows, tokens);
                if (v_block == observed_z[slot]) {
                    std::cerr << label << " rank " << rank
                              << ": v output collapsed into z output (equal-size aliasing)\n";
                    ++failures;
                }
            }
        }
    }
    return failures;
}

// ================================================================================================
// gdn_input_proj: split-storage two-weight form (Q4G64_F16S query_key + Q5G64_F16S value_z).
// ================================================================================================

struct SplitStorageShard {
    qw::PackedWeight top;    // Q (query_key) or V (value_z)
    qw::PackedWeight bottom; // K (query_key) or Z (value_z)
    qw::PackedWeight shard;  // concat(top, bottom)
};

SplitStorageShard make_split_storage_shard(QType qtype, std::int32_t half_rows, std::uint32_t seed,
                                           int rank) {
    SplitStorageShard out;
    const std::int32_t shard_half = half_rows / 2;
    out.top    = make_block(qtype, shard_half, kInputRows, seed, rank * shard_half);
    out.bottom = make_block(qtype, shard_half, kInputRows, seed, half_rows + rank * shard_half);
    out.shard  = concat_row_blocks(out.top, out.bottom);
    return out;
}

int run_split_storage_case(const ExecutionContext& ec, std::uint32_t seed) {
    const std::string head = "q4/q5 gdn_input split-storage";
    std::cout << head << " query_key [" << kQkRows << ',' << kInputRows << "] -> [" << kShardQkRows
              << ',' << kInputRows << "], value_z [" << kValueZRows << ',' << kInputRows << "] -> ["
              << kShardValueZRows << ',' << kInputRows << "]\n";
    int failures = 0;

    qw::PatternedWeightOptions qk_options;
    qk_options.decorrelate_coordinates = true;
    qk_options.row_split_scale         = qw::RowSplitScalePattern::Small;
    qk_options.row_split_codes         = qw::RowSplitCodePattern::Coordinate;
    const qw::PackedWeight full_query_key =
        qw::make_patterned_weight(QType::Q4G64_F16S, kQkRows, kInputRows, seed, qk_options);
    const qw::PackedWeight full_value_z =
        qw::make_patterned_weight(QType::Q5G64_F16S, kValueZRows, kInputRows, seed + 1, qk_options);

    std::array<SplitStorageShard, 2> query_key{
        make_split_storage_shard(QType::Q4G64_F16S, kKeyRows, seed, 0),
        make_split_storage_shard(QType::Q4G64_F16S, kKeyRows, seed, 1)};
    std::array<SplitStorageShard, 2> value_z{
        make_split_storage_shard(QType::Q5G64_F16S, kValueRows, seed + 1, 0),
        make_split_storage_shard(QType::Q5G64_F16S, kValueRows, seed + 1, 1)};

    for (int rank = 0; rank < 2; ++rank) {
        const auto slot          = static_cast<std::size_t>(rank);
        const std::string label = head + " shard " + std::to_string(rank);
        failures += verify_shard_is_parent_block(label + " q", full_query_key, query_key[slot].top,
                                                 rank * kShardKeyRows, 0);
        failures += verify_shard_is_parent_block(label + " k", full_query_key, query_key[slot].bottom,
                                                 kKeyRows + rank * kShardKeyRows, 0);
        failures += verify_shard_is_parent_block(label + " v", full_value_z, value_z[slot].top,
                                                 rank * kShardValueRows, 0);
        failures += verify_shard_is_parent_block(label + " z", full_value_z, value_z[slot].bottom,
                                                 kValueRows + rank * kShardValueRows, 0);
        failures += verify_shards_are_distinct(label + " q-vs-k", query_key[slot].top.payload,
                                               query_key[slot].bottom.payload);
        failures += verify_shards_are_distinct(label + " v-vs-z", value_z[slot].top.payload,
                                               value_z[slot].bottom.payload);
    }
    failures +=
        verify_shards_are_distinct(head + " query_key", query_key[0].shard.payload, query_key[1].shard.payload);
    failures +=
        verify_shards_are_distinct(head + " value_z", value_z[0].shard.payload, value_z[1].shard.payload);
    if (failures != 0) { return failures; }

    set_device(ec, 0);
    DeviceWeight full_qk_device = upload_weight(full_query_key);
    DeviceWeight full_vz_device = upload_weight(full_value_z);
    std::array<DeviceWeight, 2> qk_device;
    std::array<DeviceWeight, 2> vz_device;
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        qk_device[static_cast<std::size_t>(rank)] = upload_weight(query_key[static_cast<std::size_t>(rank)].shard);
        vz_device[static_cast<std::size_t>(rank)] = upload_weight(value_z[static_cast<std::size_t>(rank)].shard);
    }

    // Route sweep: T=1 exercises the gemv edge on the tp1 reference kernel (irrelevant to the
    // shard, which always routes through the row-generic grouped-MMA kernel -- see
    // gdn_input_proj.h's design note and q4_q5_gdn_input_plan.cpp); T around the BN=64/128 tile
    // boundaries (63/64/65) proves partial-tile masking; T=17/21 cross the tp1 reference's own
    // small-T/MMA route boundary at T=17.
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
        GuardedDeviceBuffer ref_qkv(static_cast<std::size_t>(kQkvRows) * tokens * sizeof(std::uint16_t));
        GuardedDeviceBuffer ref_z(static_cast<std::size_t>(kValueRows) * tokens * sizeof(std::uint16_t));
        ref_qkv.fill(0xff);
        ref_z.fill(0xff);
        Tensor reference_x(full_x.p, DType::BF16, {kInputRows, tokens});
        Tensor reference_qkv(ref_qkv.data(), DType::BF16, {kQkvRows, tokens});
        Tensor reference_z(ref_z.data(), DType::BF16, {kValueRows, tokens});
        const std::size_t reference_capacity =
            ops::q4_q5_gdn_input_proj_workspace_capacity_bytes(tokens, tokens);
        DeviceArena reference_arena(std::max<std::size_t>(reference_capacity, 1));

        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ops::gdn_input_proj(reference_x, full_qk_device.weight, full_vz_device.weight, reference_qkv,
                            reference_z, reference_arena, ec.dev[0]->stream);
        cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
        failures += ref_qkv.verify_guards(label + " reference qkv");
        failures += ref_z.verify_guards(label + " reference z");
        const std::vector<double> expected_qkv =
            from_device_bf16(ref_qkv.data(), static_cast<std::size_t>(kQkvRows) * tokens);
        const std::vector<double> expected_z =
            from_device_bf16(ref_z.data(), static_cast<std::size_t>(kValueRows) * tokens);

        // --- (b) the split form -----------------------------------------------------------------
        std::array<std::optional<GuardedDeviceBuffer>, 2> split_qkv;
        std::array<std::optional<GuardedDeviceBuffer>, 2> split_z;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            split_qkv[slot].emplace(static_cast<std::size_t>(kShardQkvRows) * tokens * sizeof(std::uint16_t));
            split_z[slot].emplace(static_cast<std::size_t>(kShardValueRows) * tokens * sizeof(std::uint16_t));
            split_qkv[slot]->fill(0xff);
            split_z[slot]->fill(0xff);
        }

        const std::array<Tensor, 2> x{Tensor(shard_x[0].p, DType::BF16, {kInputRows, tokens}),
                                      Tensor(shard_x[1].p, DType::BF16, {kInputRows, tokens})};
        const std::array<Weight, 2> qk_weight{qk_device[0].weight, qk_device[1].weight};
        const std::array<Weight, 2> vz_weight{vz_device[0].weight, vz_device[1].weight};
        const std::array<Tensor, 2> qkv_out{
            Tensor(split_qkv[0]->data(), DType::BF16, {kShardQkvRows, tokens}),
            Tensor(split_qkv[1]->data(), DType::BF16, {kShardQkvRows, tokens})};
        const std::array<Tensor, 2> z_out{Tensor(split_z[0]->data(), DType::BF16, {kShardValueRows, tokens}),
                                          Tensor(split_z[1]->data(), DType::BF16, {kShardValueRows, tokens})};

        retire_staging(ec);
        ops::gdn_input_proj_column_parallel(x, qk_weight, vz_weight, qkv_out, z_out, ec);
        synchronize_both(ec);

        std::array<std::vector<double>, 2> observed_qkv;
        std::array<std::vector<double>, 2> observed_z;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot          = static_cast<std::size_t>(rank);
            const std::string prefix = label + " rank " + std::to_string(rank);
            set_device(ec, rank);
            failures += split_qkv[slot]->verify_guards(prefix + " qkv");
            failures += split_z[slot]->verify_guards(prefix + " z");
            observed_qkv[slot] =
                from_device_bf16(split_qkv[slot]->data(), static_cast<std::size_t>(kShardQkvRows) * tokens);
            observed_z[slot] =
                from_device_bf16(split_z[slot]->data(), static_cast<std::size_t>(kShardValueRows) * tokens);

            failures += compare(
                prefix + " q", extract_block(observed_qkv[slot], kShardQkvRows, 0, kShardKeyRows, tokens),
                extract_block(expected_qkv, kQkvRows, rank * kShardKeyRows, kShardKeyRows, tokens));
            failures += compare(
                prefix + " k",
                extract_block(observed_qkv[slot], kShardQkvRows, kShardKeyRows, kShardKeyRows, tokens),
                extract_block(expected_qkv, kQkvRows, kKeyRows + rank * kShardKeyRows, kShardKeyRows,
                             tokens));
            failures += compare(
                prefix + " v",
                extract_block(observed_qkv[slot], kShardQkvRows, 2 * kShardKeyRows, kShardValueRows,
                             tokens),
                extract_block(expected_qkv, kQkvRows, 2 * kKeyRows + rank * kShardValueRows,
                             kShardValueRows, tokens));
            failures += compare(prefix + " z", observed_z[slot],
                                extract_block(expected_z, kValueRows, rank * kShardValueRows,
                                             kShardValueRows, tokens));
        }

        if (observed_qkv[0] == observed_qkv[1]) {
            std::cerr << label
                      << ": both ranks produced identical qkv output, so the column split did "
                         "not actually split\n";
            ++failures;
        }
    }
    return failures;
}

// ================================================================================================
// gdn_gating_proj: BF16_CTRL, two-weight and fused-ab-weight forms.
// ================================================================================================

std::vector<std::uint16_t> make_bf16_rows(std::int32_t rows, std::int32_t k, std::uint32_t seed) {
    std::vector<float> values(static_cast<std::size_t>(rows) * k);
    fill_uniform(values, seed, -1.0F, 1.0F);
    round_to_bf16(values);
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<std::uint16_t> extract_rows_u16(const std::vector<std::uint16_t>& full,
                                            std::int32_t hidden, std::int32_t row_begin,
                                            std::int32_t rows) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(rows) * hidden);
    std::copy(full.begin() + static_cast<std::ptrdiff_t>(row_begin) * hidden,
             full.begin() + static_cast<std::ptrdiff_t>(row_begin + rows) * hidden, out.begin());
    return out;
}

std::vector<float> extract_rows_f32(const std::vector<float>& full, std::int32_t row_begin,
                                    std::int32_t rows) {
    return {full.begin() + row_begin, full.begin() + row_begin + rows};
}

Weight bf16_weight_view(const DeviceBuffer& buffer, std::int32_t rows, std::int32_t hidden) {
    Weight w{};
    w.qtype            = QType::BF16_CTRL;
    w.layout           = QuantLayout::Contiguous;
    w.payload          = buffer.p;
    w.payload_bytes    = static_cast<std::uint64_t>(rows) * hidden * sizeof(std::uint16_t);
    w.qdata            = buffer.p;
    w.ndim             = 2;
    w.n                = rows;
    w.k                = hidden;
    w.shape[0]         = rows;
    w.shape[1]         = hidden;
    w.padded_shape[0]  = rows;
    w.padded_shape[1]  = hidden;
    return w;
}

int run_gating_case(const ExecutionContext& ec, std::uint32_t seed) {
    const std::string head = "gdn_gating_proj";
    std::cout << head << " a/b [" << kGatingHeads << ',' << kGatingHidden << "] -> ["
              << kGatingShardHeads << ',' << kGatingHidden << "]\n";
    int failures = 0;

    const std::vector<std::uint16_t> full_a = make_bf16_rows(kGatingHeads, kGatingHidden, seed);
    const std::vector<std::uint16_t> full_b = make_bf16_rows(kGatingHeads, kGatingHidden, seed + 1);
    std::vector<float> full_a_log(kGatingHeads);
    std::vector<float> full_dt_bias(kGatingHeads);
    fill_uniform(full_a_log, seed + 2, -2.0F, 1.0F);
    fill_uniform(full_dt_bias, seed + 3, -1.0F, 1.0F);

    std::array<std::vector<std::uint16_t>, 2> shard_a;
    std::array<std::vector<std::uint16_t>, 2> shard_b;
    std::array<std::vector<float>, 2> shard_a_log;
    std::array<std::vector<float>, 2> shard_dt_bias;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot     = static_cast<std::size_t>(rank);
        shard_a[slot]       = extract_rows_u16(full_a, kGatingHidden, rank * kGatingShardHeads,
                                              kGatingShardHeads);
        shard_b[slot]       = extract_rows_u16(full_b, kGatingHidden, rank * kGatingShardHeads,
                                              kGatingShardHeads);
        shard_a_log[slot]   = extract_rows_f32(full_a_log, rank * kGatingShardHeads, kGatingShardHeads);
        shard_dt_bias[slot] = extract_rows_f32(full_dt_bias, rank * kGatingShardHeads, kGatingShardHeads);
    }
    // Cross-rank distinctness: independently-random rows are distinct with overwhelming
    // probability; assert it rather than assume it.
    failures += verify_shards_are_distinct(head + " a", shard_a[0], shard_a[1]);
    failures += verify_shards_are_distinct(head + " b", shard_b[0], shard_b[1]);

    set_device(ec, 0);
    DeviceBuffer full_a_device     = to_device(full_a);
    DeviceBuffer full_b_device     = to_device(full_b);
    DeviceBuffer full_a_log_device = to_device_f32(full_a_log);
    DeviceBuffer full_dt_bias_device = to_device_f32(full_dt_bias);

    std::array<DeviceBuffer, 2> shard_a_device;
    std::array<DeviceBuffer, 2> shard_b_device;
    std::array<DeviceBuffer, 2> shard_a_log_device;
    std::array<DeviceBuffer, 2> shard_dt_bias_device;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        shard_a_device[slot]        = to_device(shard_a[slot]);
        shard_b_device[slot]        = to_device(shard_b[slot]);
        shard_a_log_device[slot]    = to_device_f32(shard_a_log[slot]);
        shard_dt_bias_device[slot]  = to_device_f32(shard_dt_bias[slot]);
    }

    // T sweep: T=1 (the shard's gemv route), T=8 (SmallTSplit10 tp1 boundary), T=48/1024 (the tp1
    // reference crosses into its MMA route; the shard has no MMA route -- see
    // bf16_gdn_gating_proj_kernels.cu's kShardN comment -- and always uses small-T-split10,
    // proving correctness at scale on that fallback, not just small T).
    const std::vector<std::int32_t> tokens_sweep{1, 2, 8, 48, 1024};

    for (const std::int32_t tokens : tokens_sweep) {
        const std::string label = head + " T=" + std::to_string(tokens);
        std::vector<float> activation(static_cast<std::size_t>(kGatingHidden) * tokens);
        fill_uniform(activation, seed * 41u + static_cast<std::uint32_t>(tokens), -1.0F, 1.0F);
        round_to_bf16(activation);

        set_device(ec, 0);
        DeviceBuffer full_x = to_device_bf16(activation);
        std::array<DeviceBuffer, 2> shard_x;
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            shard_x[static_cast<std::size_t>(rank)] = to_device_bf16(activation);
        }

        // --- (a) reference: tp1 two-weight kernel, whole weights, device 0 ----------------------
        set_device(ec, 0);
        const std::size_t reference_ws =
            ops::gdn_gating_proj_workspace_capacity_bytes(kGatingHeads, kGatingHidden, tokens, tokens);
        DeviceArena reference_arena(std::max<std::size_t>(reference_ws, 1));
        GuardedDeviceBuffer ref_g(static_cast<std::size_t>(kGatingHeads) * tokens * sizeof(float));
        GuardedDeviceBuffer ref_beta(static_cast<std::size_t>(kGatingHeads) * tokens * sizeof(float));
        ref_g.fill(0xff);
        ref_beta.fill(0xff);
        Tensor reference_x(full_x.p, DType::BF16, {kGatingHidden, tokens});
        const Weight full_a_weight = bf16_weight_view(full_a_device, kGatingHeads, kGatingHidden);
        const Weight full_b_weight = bf16_weight_view(full_b_device, kGatingHeads, kGatingHidden);
        Tensor reference_a_log(full_a_log_device.p, DType::FP32, {kGatingHeads});
        Tensor reference_dt_bias(full_dt_bias_device.p, DType::FP32, {kGatingHeads});
        Tensor reference_g(ref_g.data(), DType::FP32, {kGatingHeads, tokens});
        Tensor reference_beta(ref_beta.data(), DType::FP32, {kGatingHeads, tokens});

        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ops::gdn_gating_proj(reference_x, full_a_weight, full_b_weight, reference_a_log,
                            reference_dt_bias, reference_arena, reference_g, reference_beta,
                            ec.dev[0]->stream);
        cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
        failures += ref_g.verify_guards(label + " reference g");
        failures += ref_beta.verify_guards(label + " reference beta");
        const std::vector<double> expected_g =
            from_device_f32_ptr(ref_g.data(), static_cast<std::size_t>(kGatingHeads) * tokens);
        const std::vector<double> expected_beta =
            from_device_f32_ptr(ref_beta.data(), static_cast<std::size_t>(kGatingHeads) * tokens);

        // --- (b) the split form ------------------------------------------------------------------
        const std::size_t split_ws =
            ops::gdn_gating_proj_column_parallel_workspace_capacity_bytes(tokens, tokens);
        std::array<std::optional<GuardedDeviceBuffer>, 2> split_g;
        std::array<std::optional<GuardedDeviceBuffer>, 2> split_beta;
        std::array<std::optional<DeviceArena>, 2> arena;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            split_g[slot].emplace(static_cast<std::size_t>(kGatingShardHeads) * tokens * sizeof(float));
            split_beta[slot].emplace(static_cast<std::size_t>(kGatingShardHeads) * tokens * sizeof(float));
            split_g[slot]->fill(0xff);
            split_beta[slot]->fill(0xff);
            arena[slot].emplace(std::max<std::size_t>(split_ws, 1));
        }

        const std::array<Tensor, 2> x{Tensor(shard_x[0].p, DType::BF16, {kGatingHidden, tokens}),
                                      Tensor(shard_x[1].p, DType::BF16, {kGatingHidden, tokens})};
        const std::array<Weight, 2> a_weight{
            bf16_weight_view(shard_a_device[0], kGatingShardHeads, kGatingHidden),
            bf16_weight_view(shard_a_device[1], kGatingShardHeads, kGatingHidden)};
        const std::array<Weight, 2> b_weight{
            bf16_weight_view(shard_b_device[0], kGatingShardHeads, kGatingHidden),
            bf16_weight_view(shard_b_device[1], kGatingShardHeads, kGatingHidden)};
        const std::array<Tensor, 2> a_log{
            Tensor(shard_a_log_device[0].p, DType::FP32, {kGatingShardHeads}),
            Tensor(shard_a_log_device[1].p, DType::FP32, {kGatingShardHeads})};
        const std::array<Tensor, 2> dt_bias{
            Tensor(shard_dt_bias_device[0].p, DType::FP32, {kGatingShardHeads}),
            Tensor(shard_dt_bias_device[1].p, DType::FP32, {kGatingShardHeads})};
        const std::array<Tensor, 2> g_out{
            Tensor(split_g[0]->data(), DType::FP32, {kGatingShardHeads, tokens}),
            Tensor(split_g[1]->data(), DType::FP32, {kGatingShardHeads, tokens})};
        const std::array<Tensor, 2> beta_out{
            Tensor(split_beta[0]->data(), DType::FP32, {kGatingShardHeads, tokens}),
            Tensor(split_beta[1]->data(), DType::FP32, {kGatingShardHeads, tokens})};
        const std::array<ninfer::WorkspaceArena*, 2> workspace{&*arena[0], &*arena[1]};

        retire_staging(ec);
        ops::gdn_gating_proj_column_parallel(x, a_weight, b_weight, a_log, dt_bias, workspace, g_out,
                                             beta_out, ec);
        synchronize_both(ec);

        std::array<std::vector<double>, 2> observed_g;
        std::array<std::vector<double>, 2> observed_beta;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot          = static_cast<std::size_t>(rank);
            const std::string prefix = label + " rank " + std::to_string(rank);
            set_device(ec, rank);
            failures += split_g[slot]->verify_guards(prefix + " g");
            failures += split_beta[slot]->verify_guards(prefix + " beta");
            observed_g[slot] =
                from_device_f32_ptr(split_g[slot]->data(), static_cast<std::size_t>(kGatingShardHeads) * tokens);
            observed_beta[slot] = from_device_f32_ptr(split_beta[slot]->data(),
                                                  static_cast<std::size_t>(kGatingShardHeads) * tokens);

            failures += compare(prefix + " g", observed_g[slot],
                                extract_block(expected_g, kGatingHeads, rank * kGatingShardHeads,
                                             kGatingShardHeads, tokens));
            failures += compare(prefix + " beta", observed_beta[slot],
                                extract_block(expected_beta, kGatingHeads, rank * kGatingShardHeads,
                                             kGatingShardHeads, tokens));
        }
        if (observed_g[0] == observed_g[1]) {
            std::cerr << label
                      << ": both ranks produced identical g output, so the column split did not "
                         "actually split\n";
            ++failures;
        }
    }
    return failures;
}

// Fused ab_weight[96,5120] form (A rows [0,48), B rows [48,96)) -- the production binding for
// `gdn/a_b_projection`. One representative T proves the fused overload's own row-view slicing
// (bf16_row_view, applied to a SHARD-sized [48,5120] parent this time) is correct; the two-weight
// form above already covers the T sweep and per-head parity in full.
int run_gating_fused_case(const ExecutionContext& ec, std::uint32_t seed, std::int32_t tokens) {
    const std::string head = "gdn_gating_proj fused ab_weight";
    const std::string label = head + " T=" + std::to_string(tokens);
    std::cout << label << " [" << 2 * kGatingHeads << ',' << kGatingHidden << "] -> ["
              << 2 * kGatingShardHeads << ',' << kGatingHidden << "]\n";
    int failures = 0;

    const std::vector<std::uint16_t> full_a = make_bf16_rows(kGatingHeads, kGatingHidden, seed);
    const std::vector<std::uint16_t> full_b = make_bf16_rows(kGatingHeads, kGatingHidden, seed + 1);
    std::vector<std::uint16_t> full_ab(static_cast<std::size_t>(2 * kGatingHeads) * kGatingHidden);
    std::copy(full_a.begin(), full_a.end(), full_ab.begin());
    std::copy(full_b.begin(), full_b.end(),
             full_ab.begin() + static_cast<std::ptrdiff_t>(full_a.size()));

    std::vector<float> full_a_log(kGatingHeads);
    std::vector<float> full_dt_bias(kGatingHeads);
    fill_uniform(full_a_log, seed + 2, -2.0F, 1.0F);
    fill_uniform(full_dt_bias, seed + 3, -1.0F, 1.0F);

    std::array<std::vector<std::uint16_t>, 2> shard_ab;
    std::array<std::vector<float>, 2> shard_a_log;
    std::array<std::vector<float>, 2> shard_dt_bias;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        std::vector<std::uint16_t> a_rows =
            extract_rows_u16(full_a, kGatingHidden, rank * kGatingShardHeads, kGatingShardHeads);
        std::vector<std::uint16_t> b_rows =
            extract_rows_u16(full_b, kGatingHidden, rank * kGatingShardHeads, kGatingShardHeads);
        // Intra-shard distinctness: a-vs-b (both 24 rows) -- the equal-size aliasing candidate for
        // the fused ab_weight shard, matching this file's discipline for q-vs-k/v-vs-z elsewhere.
        failures += verify_shards_are_distinct(
            head + " rank " + std::to_string(rank) + " a-vs-b", a_rows, b_rows);
        shard_ab[slot] = a_rows;
        shard_ab[slot].insert(shard_ab[slot].end(), b_rows.begin(), b_rows.end());
        shard_a_log[slot]   = extract_rows_f32(full_a_log, rank * kGatingShardHeads, kGatingShardHeads);
        shard_dt_bias[slot] = extract_rows_f32(full_dt_bias, rank * kGatingShardHeads, kGatingShardHeads);
    }
    failures += verify_shards_are_distinct(head + " ab", shard_ab[0], shard_ab[1]);
    if (failures != 0) { return failures; }

    set_device(ec, 0);
    DeviceBuffer full_ab_device      = to_device(full_ab);
    DeviceBuffer full_a_log_device   = to_device_f32(full_a_log);
    DeviceBuffer full_dt_bias_device = to_device_f32(full_dt_bias);
    std::array<DeviceBuffer, 2> shard_ab_device;
    std::array<DeviceBuffer, 2> shard_a_log_device;
    std::array<DeviceBuffer, 2> shard_dt_bias_device;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        shard_ab_device[slot]      = to_device(shard_ab[slot]);
        shard_a_log_device[slot]   = to_device_f32(shard_a_log[slot]);
        shard_dt_bias_device[slot] = to_device_f32(shard_dt_bias[slot]);
    }

    std::vector<float> activation(static_cast<std::size_t>(kGatingHidden) * tokens);
    fill_uniform(activation, seed * 47u + static_cast<std::uint32_t>(tokens), -1.0F, 1.0F);
    round_to_bf16(activation);
    set_device(ec, 0);
    DeviceBuffer full_x = to_device_bf16(activation);
    std::array<DeviceBuffer, 2> shard_x;
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        shard_x[static_cast<std::size_t>(rank)] = to_device_bf16(activation);
    }

    // --- (a) reference: tp1 fused-parent kernel, device 0 -------------------------------------
    set_device(ec, 0);
    const std::size_t reference_ws =
        ops::gdn_gating_proj_workspace_capacity_bytes(kGatingHeads, kGatingHidden, tokens, tokens);
    DeviceArena reference_arena(std::max<std::size_t>(reference_ws, 1));
    GuardedDeviceBuffer ref_g(static_cast<std::size_t>(kGatingHeads) * tokens * sizeof(float));
    GuardedDeviceBuffer ref_beta(static_cast<std::size_t>(kGatingHeads) * tokens * sizeof(float));
    ref_g.fill(0xff);
    ref_beta.fill(0xff);
    Tensor reference_x(full_x.p, DType::BF16, {kGatingHidden, tokens});
    const Weight full_ab_weight = bf16_weight_view(full_ab_device, 2 * kGatingHeads, kGatingHidden);
    Tensor reference_a_log(full_a_log_device.p, DType::FP32, {kGatingHeads});
    Tensor reference_dt_bias(full_dt_bias_device.p, DType::FP32, {kGatingHeads});
    Tensor reference_g(ref_g.data(), DType::FP32, {kGatingHeads, tokens});
    Tensor reference_beta(ref_beta.data(), DType::FP32, {kGatingHeads, tokens});

    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    ops::gdn_gating_proj(reference_x, full_ab_weight, reference_a_log, reference_dt_bias,
                        reference_arena, reference_g, reference_beta, ec.dev[0]->stream);
    cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
    failures += ref_g.verify_guards(label + " reference g");
    failures += ref_beta.verify_guards(label + " reference beta");
    const std::vector<double> expected_g =
        from_device_f32_ptr(ref_g.data(), static_cast<std::size_t>(kGatingHeads) * tokens);
    const std::vector<double> expected_beta =
        from_device_f32_ptr(ref_beta.data(), static_cast<std::size_t>(kGatingHeads) * tokens);

    // --- (b) the split form ---------------------------------------------------------------------
    const std::size_t split_ws =
        ops::gdn_gating_proj_column_parallel_workspace_capacity_bytes(tokens, tokens);
    std::array<std::optional<GuardedDeviceBuffer>, 2> split_g;
    std::array<std::optional<GuardedDeviceBuffer>, 2> split_beta;
    std::array<std::optional<DeviceArena>, 2> arena;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        split_g[slot].emplace(static_cast<std::size_t>(kGatingShardHeads) * tokens * sizeof(float));
        split_beta[slot].emplace(static_cast<std::size_t>(kGatingShardHeads) * tokens * sizeof(float));
        split_g[slot]->fill(0xff);
        split_beta[slot]->fill(0xff);
        arena[slot].emplace(std::max<std::size_t>(split_ws, 1));
    }

    const std::array<Tensor, 2> x{Tensor(shard_x[0].p, DType::BF16, {kGatingHidden, tokens}),
                                  Tensor(shard_x[1].p, DType::BF16, {kGatingHidden, tokens})};
    const std::array<Weight, 2> ab_weight{
        bf16_weight_view(shard_ab_device[0], 2 * kGatingShardHeads, kGatingHidden),
        bf16_weight_view(shard_ab_device[1], 2 * kGatingShardHeads, kGatingHidden)};
    const std::array<Tensor, 2> a_log{Tensor(shard_a_log_device[0].p, DType::FP32, {kGatingShardHeads}),
                                      Tensor(shard_a_log_device[1].p, DType::FP32, {kGatingShardHeads})};
    const std::array<Tensor, 2> dt_bias{
        Tensor(shard_dt_bias_device[0].p, DType::FP32, {kGatingShardHeads}),
        Tensor(shard_dt_bias_device[1].p, DType::FP32, {kGatingShardHeads})};
    const std::array<Tensor, 2> g_out{Tensor(split_g[0]->data(), DType::FP32, {kGatingShardHeads, tokens}),
                                      Tensor(split_g[1]->data(), DType::FP32, {kGatingShardHeads, tokens})};
    const std::array<Tensor, 2> beta_out{
        Tensor(split_beta[0]->data(), DType::FP32, {kGatingShardHeads, tokens}),
        Tensor(split_beta[1]->data(), DType::FP32, {kGatingShardHeads, tokens})};
    const std::array<ninfer::WorkspaceArena*, 2> workspace{&*arena[0], &*arena[1]};

    retire_staging(ec);
    ops::gdn_gating_proj_column_parallel(x, ab_weight, a_log, dt_bias, workspace, g_out, beta_out, ec);
    synchronize_both(ec);

    std::array<std::vector<double>, 2> observed_g;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot          = static_cast<std::size_t>(rank);
        const std::string prefix = label + " rank " + std::to_string(rank);
        set_device(ec, rank);
        failures += split_g[slot]->verify_guards(prefix + " g");
        failures += split_beta[slot]->verify_guards(prefix + " beta");
        observed_g[slot] =
            from_device_f32_ptr(split_g[slot]->data(), static_cast<std::size_t>(kGatingShardHeads) * tokens);
        const std::vector<double> observed_beta = from_device_f32_ptr(
            split_beta[slot]->data(), static_cast<std::size_t>(kGatingShardHeads) * tokens);

        failures += compare(prefix + " g", observed_g[slot],
                            extract_block(expected_g, kGatingHeads, rank * kGatingShardHeads,
                                         kGatingShardHeads, tokens));
        failures += compare(prefix + " beta", observed_beta,
                            extract_block(expected_beta, kGatingHeads, rank * kGatingShardHeads,
                                         kGatingShardHeads, tokens));
    }
    if (observed_g[0] == observed_g[1]) {
        std::cerr << label
                  << ": both ranks produced identical g output, so the column split did not "
                     "actually split\n";
        ++failures;
    }
    return failures;
}

// ---------------------------------------------------------------------------------------------
// Host-only registry probes: pure host code, run BEFORE the device checks like every sibling
// split suite's own registry probe.
// ---------------------------------------------------------------------------------------------
int verify_registry() {
    int failures = 0;
    for (const QType qtype : {QType::NVFP4, QType::FP8_E4M3FN_ROW_BF16S}) {
        const std::vector<ops::LinearPolicy> policies =
            qtype == QType::NVFP4
                ? std::vector<ops::LinearPolicy>{ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA4}
                : std::vector<ops::LinearPolicy>{ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8};
        for (const ops::LinearPolicy policy : policies) {
            for (const std::int32_t tokens : {1, 2, 48, 1024}) {
                try {
                    (void)ops::gdn_input_proj_column_parallel_workspace_capacity_bytes(
                        qtype, policy, tokens, tokens);
                } catch (const std::exception& error) {
                    std::cerr << "registry: gdn_input " << static_cast<int>(qtype) << ' '
                              << policy_name(policy) << " T=" << tokens
                              << " rejected: " << error.what() << '\n';
                    ++failures;
                }
            }
        }
    }
    for (const QType qtype : {QType::BF16_CTRL, QType::W8G32_F16S, QType::Q4G64_F16S,
                              QType::Q5G64_F16S}) {
        bool threw = false;
        try {
            (void)ops::gdn_input_proj_column_parallel_workspace_capacity_bytes(
                qtype, ops::LinearPolicy::A16Only, 1, 1);
        } catch (const std::exception&) { threw = true; }
        if (!threw) {
            std::cerr << "registry: gdn_input qtype " << static_cast<int>(qtype)
                      << " was admitted by the column-parallel workspace query but must not be\n";
            ++failures;
        }
    }
    for (const std::int32_t tokens : {1, 2, 48, 1024}) {
        try {
            (void)ops::gdn_gating_proj_column_parallel_workspace_capacity_bytes(tokens, tokens);
        } catch (const std::exception& error) {
            std::cerr << "registry: gdn_gating T=" << tokens << " rejected: " << error.what()
                      << '\n';
            ++failures;
        }
    }
    bool threw_bad_interval = false;
    try {
        (void)ops::gdn_gating_proj_column_parallel_workspace_capacity_bytes(4, 1);
    } catch (const std::exception&) { threw_bad_interval = true; }
    if (!threw_bad_interval) {
        std::cerr << "registry: gdn_gating accepted an inverted token interval\n";
        ++failures;
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
    DeviceBuffer qkv0(static_cast<std::size_t>(kShardQkvRows) * 2 * sizeof(std::uint16_t));
    DeviceBuffer z0(static_cast<std::size_t>(kShardValueRows) * 2 * sizeof(std::uint16_t));
    set_device(ec, 1);
    DeviceBuffer x1(static_cast<std::size_t>(kInputRows) * 2 * sizeof(std::uint16_t));
    DeviceBuffer qkv1(static_cast<std::size_t>(kShardQkvRows) * 2 * sizeof(std::uint16_t));
    DeviceBuffer z1(static_cast<std::size_t>(kShardValueRows) * 2 * sizeof(std::uint16_t));

    Weight fake{};
    fake.qtype = QType::NVFP4;
    fake.n     = kShardFusedRows;
    fake.k     = kInputRows;

    expect_throw("token count", [&] {
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kInputRows, 2}),
                                      Tensor(x1.p, DType::BF16, {kInputRows, 1})};
        const std::array<Tensor, 2> qkv{Tensor(qkv0.p, DType::BF16, {kShardQkvRows, 2}),
                                        Tensor(qkv1.p, DType::BF16, {kShardQkvRows, 1})};
        const std::array<Tensor, 2> z{Tensor(z0.p, DType::BF16, {kShardValueRows, 2}),
                                      Tensor(z1.p, DType::BF16, {kShardValueRows, 1})};
        ops::gdn_input_proj_column_parallel(x, {fake, fake}, qkv, z, ec);
    });

    expect_throw("column K", [&] {
        Weight other = fake;
        other.k      = kInputRows / 2;
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kInputRows, 1}),
                                      Tensor(x1.p, DType::BF16, {kInputRows / 2, 1})};
        const std::array<Tensor, 2> qkv{Tensor(qkv0.p, DType::BF16, {kShardQkvRows, 1}),
                                        Tensor(qkv1.p, DType::BF16, {kShardQkvRows, 1})};
        const std::array<Tensor, 2> z{Tensor(z0.p, DType::BF16, {kShardValueRows, 1}),
                                      Tensor(z1.p, DType::BF16, {kShardValueRows, 1})};
        ops::gdn_input_proj_column_parallel(x, {fake, other}, qkv, z, ec);
    });

    expect_throw("tp1 context", [&] {
        const ExecutionContext single({0});
        const std::array<Tensor, 2> x{Tensor(x0.p, DType::BF16, {kInputRows, 1}),
                                      Tensor(x1.p, DType::BF16, {kInputRows, 1})};
        const std::array<Tensor, 2> qkv{Tensor(qkv0.p, DType::BF16, {kShardQkvRows, 1}),
                                        Tensor(qkv1.p, DType::BF16, {kShardQkvRows, 1})};
        const std::array<Tensor, 2> z{Tensor(z0.p, DType::BF16, {kShardValueRows, 1}),
                                      Tensor(z1.p, DType::BF16, {kShardValueRows, 1})};
        ops::gdn_input_proj_column_parallel(x, {fake, fake}, qkv, z, single);
    });

    std::cout << (failures ? "FAIL" : "OK") << " split rejections\n";
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    const bool fp8_only = argc == 2 && std::string_view(argv[1]) == "--fp8-only";
    if (argc != 1 && !fp8_only) {
        std::cerr << "usage: ninfer_gdn_projections_split_test [--fp8-only]\n";
        return 2;
    }
    int failures = verify_registry();
    if (failures != 0) {
        std::cout << "FAIL gdn_projections split (registry)\n";
        return 1;
    }

    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: split gdn_projections parity requires two CUDA devices, found "
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
    cudaDeviceProp device_props{};
    cuda_check(cudaGetDeviceProperties(&device_props, 0), "cudaGetDeviceProperties");
    if (!fp8_only) {
        failures += run_fused_case(
            ec, QType::NVFP4,
            device_props.major >= 12
                ? std::vector{ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA4}
                : std::vector{ops::LinearPolicy::A16Only},
            41u);
    }
    failures += run_fused_case(
        ec, QType::FP8_E4M3FN_ROW_BF16S,
        device_props.major >= 10
            ? std::vector{ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8}
            : std::vector{ops::LinearPolicy::A16Only},
        45u);
    if (!fp8_only) {
        failures += run_split_storage_case(ec, 43u);
        failures += run_gating_case(ec, 51u);
        failures += run_gating_fused_case(ec, 53u, 48);
    }

    std::cout << (failures ? "FAIL" : "OK") << " gdn_projections split\n";
    return failures ? 1 : 0;
}
