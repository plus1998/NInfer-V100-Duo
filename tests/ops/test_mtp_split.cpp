// Two-device parity suite for the MTP layer, the GDN verify-phase state Ops the MTP round drives,
// and the draft head.
//
// WHAT THIS SUITE PROVES, AND WHERE THE BOUNDARY IS.
//
// This is an OP-LEVEL suite. It proves that every op the MTP verify round touches is split-capable
// and that those ops compose; it does not exercise the runtime's own tp2 MTP plumbing
// (`TextContext::mtp_forward_stem_tp2` in src/targets/qwen3_6/impl/runtime/text_context_impl.h,
// `Variant::mtp_post_mixer` / `Variant::mtp_attention_projection` in
// src/targets/qwen3_6_27b/impl/variant.cpp), which is covered end to end on real weights by
// tests/targets/qwen3_6_27b/test_engine_mtp_tp2_real.cpp.
//
// Leg A below therefore reproduces the tp1 MTP round the way `mtp_forward_stem` +
// `mtp_forward_tail` compose it, op by op, and runs the same sequence at tp2 with shard-shaped
// weights, comparing at EVERY stage. It is a genuine composition, not a per-op smoke test: each
// stage consumes the previous stage's real split output.
//
// PER-OBJECT FORMAT AND SPLIT MAP (verified by reading bindings.cpp, not assumed). Every MTP weight
// is bound W8G32_F16S in EVERY weights profile (`bind_mtp` hard-codes the format), and every one of
// them reaches the ShardPlan through a SUFFIX match on a text-layer family:
//
//   mtp/input_projection                       [5120,10240] row-parallel  -> [5120, 5120]/device
//   mtp/layer/attention/query_key_gate_value    [14336,5120] column         -> [7168, 5120]/device
//   mtp/layer/attention/output                  [5120, 6144] row-parallel  -> [5120, 3072]/device
//   mtp/layer/mlp/gate_up                       [34816,5120] column         -> [17408,5120]/device
//   mtp/layer/mlp/down                          [5120,17408] row-parallel  -> [5120, 8704]/device
//   text/draft_head             Q4G64_F16S    [131072,5120] column (vocab) -> [65536,5120]/device
//   text/draft_head_token_ids   I32               [131072]  REPLICATED (see Leg D)
//
// Every one of those shard extents is already registered in `ops::linear`'s own W8/Q4 tuning
// tables (they appear in tests/ops/test_linear_split.cpp's host-only registry probe). This suite
// is what turns those registry-probed entries into GPU-parity-tested ones for the MTP surface,
// and it adds the two pieces that were genuinely missing: `ops::mtp_split_attn_in`'s row geometry
// (7168 = Q 3072 | K 512 | Gate 3072 | V 512), and the observation that at tp2 the fc input is
// never packed at all -- `mtp/input_projection` is row-parallel over the packed 10240-wide axis, so
// device 0's shard contracts with the embedding-norm half and device 1's with the hidden-norm half
// (see include/ninfer/ops/mtp_pack.h).
//
// THE TWO CONTRACTS THE GDN SPLIT WORK LEFT TO THE MTP ROUND (legs B and C):
//   B. FoldGeometry<48,8,24,5120> -- `launch_replay_fold` selects a COMPILE-TIME FoldGeometry and
//      throws for an unregistered one, and a tp2 MTP replay needs this geometry: the qk-head
//      parameter halves too, not only the value heads and channels, because the fold kernel
//      resolves its own value-head -> qk-head map from that pair. Leg B registers and exercises
//      it, per head and per channel, against the tp1 48|16|48|10240 geometry.
//   C. gdn_input_proj_conv_snapshot / conv_record -- the fused projection+conv+state forms reached
//      only from Phase::Verify, so their split belongs with the MTP round rather than with the
//      plain GDN projections. Leg C exercises the split forms over the 5120-channel shard,
//      consistent with `gdn/convolution`'s three-block channel split.
//
// SKIP CODE. Every device leg needs two CUDA devices in one process; with fewer this suite reports
// 77, like every sibling tp2 suite.
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/speculative_round.h"

#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "ops/gqa_attention_fixture.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace qw = ninfer::test::quantized_weight;
namespace gqa = ninfer::test::gqa;

namespace {

constexpr double kBf16Ulp = 1.0 / 256.0;

// One blanket criterion for every GEMM-shaped comparison in this suite, the same 2-BF16-ulp bound
// every sibling split suite uses (see test_linear_split.cpp's criterion comment for why a
// per-element relative criterion is the wrong shape for a GEMM output: an element where the
// contributions nearly cancel is small for reasons unrelated to how accurately it was computed).
// Leg A's later stages are CHAINED -- each consumes the previous stage's split output -- so their
// drift is not bounded by one rounding; the per-stage numbers are printed for every stage.
constexpr ReductionCriterion kSplitCriterion{/*relative_l2*/ 2.0 * kBf16Ulp,
                                             /*gross_absolute*/ 0.0,
                                             /*gross_relative_to_max_reference*/ 2.0 * kBf16Ulp};

// Chained stages accumulate one BF16 storage rounding per split stage. Leg A has at most 5 split
// GEMM stages upstream of its last comparison, so 8 ulp is the derived envelope (5 roundings plus
// the reference's own), not a fitted number. Every stage prints its observed error, so a drift
// from "1 ulp" to "merely inside the envelope" is visible rather than hidden.
constexpr ReductionCriterion kChainedCriterion{/*relative_l2*/ 8.0 * kBf16Ulp,
                                               /*gross_absolute*/ 0.0,
                                               /*gross_relative_to_max_reference*/ 8.0 * kBf16Ulp};

constexpr std::int32_t kHidden      = 5120;
constexpr std::int32_t kFcInputRows = 2 * kHidden;      // mtp_input_rows
constexpr std::int32_t kAttnRows    = 14336;            // mtp_attention_input_rows
constexpr std::int32_t kQSize       = 6144;             // 24 query heads x 256
constexpr std::int32_t kKvSize      = 1024;             // 4 key/value heads x 256
constexpr std::int32_t kIntermediate = 17408;
constexpr std::int32_t kGateUpRows  = 2 * kIntermediate;
constexpr std::int32_t kHeadDim     = 256;
constexpr float kRmsEps             = 1e-6F;
constexpr int kRotaryDim            = 128;
constexpr float kRopeTheta          = 1000000.0F;

// Shard extents (tp == 2).
constexpr std::int32_t kFcShardK       = kHidden;        // row-parallel over the 10240 packed rows
constexpr std::int32_t kAttnShardRows  = 7168;
constexpr std::int32_t kShardQSize     = 3072;           // 12 query heads x 256
constexpr std::int32_t kShardKvSize    = 512;            // 2 key/value heads x 256
constexpr std::int32_t kOProjShardK    = kQSize / 2;     // 3072
constexpr std::int32_t kGateUpShardRows = kGateUpRows / 2; // 17408
constexpr std::int32_t kShardIntermediate = kIntermediate / 2; // 8704

constexpr std::int32_t kDraftRows      = 131072;
constexpr std::int32_t kDraftShardRows = kDraftRows / 2;
constexpr std::int32_t kTokenizerVocab = 248077; // validate_draft_ids' own domain bound

constexpr gqa::Geometry kGlobalAttn{"mtp_27b", 24, 4};
constexpr gqa::Geometry kLocalAttn{"mtp_27b_tp2", 12, 2};

// --- device plumbing ---------------------------------------------------------------------------

void set_device(const ExecutionContext& ec, int rank) {
    cuda_check(cudaSetDevice(ec.dev[rank]->device), "cudaSetDevice");
}

void synchronize_both(const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaStreamSynchronize(ec.dev[rank]->stream), "cudaStreamSynchronize");
    }
}

// See include/ninfer/ops/allreduce.h's "CALLER OBLIGATION": plain cudaMemcpy/cudaMemset land on a
// device's LEGACY DEFAULT stream, which DeviceContext::stream (cudaStreamNonBlocking) does not
// implicitly order against. Every host staging burst is retired before a split form reads it.
void retire_staging(const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
}

void* byte_offset(void* base, std::size_t elements, std::size_t element_size) {
    return static_cast<std::uint8_t*>(base) + elements * element_size;
}

// --- weight fixture (the shared coordinate-origin generator) ------------------------------------

// decorrelate_coordinates = true for the reason every sibling split suite sets it: the NVFP4/FP8
// fixture patterns are affine in BOTH coordinates, and several of this family's own shard strides
// annihilate the relevant term (rows 7168/8192/17408, columns 3072/8704), which would make the two
// shards byte-identical and the whole comparison vacuous while every reported error stayed at 0.0.
// `verify_shards_are_distinct` below is the assertion that catches it.
qw::PackedWeight make_block(QType qtype, std::int32_t n, std::int32_t k, std::uint32_t seed,
                            std::int32_t row_origin, std::int32_t column_origin = 0) {
    qw::PatternedWeightOptions options;
    options.row_origin              = row_origin;
    options.column_origin           = column_origin;
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
// tests/ops/test_attn_input_proj_split.cpp's own copy of test_linear_swiglu_split.cpp's
// concat_row_blocks(), whose doc comment explains why this is valid for every layout the fixture
// produces (NVFP4 additionally requires the top block's row count to be a 128-row tile multiple;
// every block spliced here -- 1024, 2048, 3072, 512, 3584, 6656, 8704 -- satisfies it).
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
    if (combined.weight.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        combined.weight.scale_ne[0] = combined.weight.n;
        combined.weight.scale_nb[1] = static_cast<std::int64_t>(combined.weight.n) * 2;
        combined.weight.scale_nb[2] = combined.weight.scale_nb[1];
        combined.weight.scale_nb[3] = combined.weight.scale_nb[1];
    }

    combined.code_plane_bytes   = top.code_plane_bytes + bottom.code_plane_bytes;
    combined.high_plane_offset  = align256(combined.code_plane_bytes);
    combined.high_plane_bytes   = top.high_plane_bytes + bottom.high_plane_bytes;
    combined.scale_plane_offset = combined.high_plane_offset + align256(combined.high_plane_bytes);
    combined.scale_plane_bytes  = top.scale_plane_bytes + bottom.scale_plane_bytes;

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
    const std::vector<std::int32_t> probes{0,   1,   31,  32,  33,  63,         64,
                                           127, 128, 129, 255, 256, 511,        512,
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

// Proves, at the LOGICAL level and before any kernel runs, that `shard` really is the parent's
// block at (row_origin, column_origin) -- copied from test_linear_split.cpp.
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

int compare(const std::string& label, const std::vector<double>& got,
            const std::vector<double>& expected, const ReductionCriterion& criterion) {
    const ReductionStats stats =
        compute_reduction_stats(got.data(), expected.data(), static_cast<std::int64_t>(got.size()));
    std::cout << "  " << label << ": max_abs=" << stats.maximum_absolute_error
              << " rel_l2=" << stats.relative_l2
              << " gross_limit=" << gross_error_limit(stats, criterion) << '\n';
    return verify_reduction(label, got, expected, criterion);
}

// Rank r's block of a reference tensor stored [parent_rows, tokens]: rows
// [r*shard_rows, (r+1)*shard_rows) of every token.
std::vector<double> extract_rank_block(const std::vector<double>& reference,
                                       std::int32_t parent_rows, std::int32_t row_begin,
                                       std::int32_t shard_rows, std::int32_t tokens) {
    std::vector<double> block(static_cast<std::size_t>(shard_rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t source =
            static_cast<std::size_t>(token) * parent_rows + static_cast<std::size_t>(row_begin);
        std::copy(reference.begin() + static_cast<std::ptrdiff_t>(source),
                  reference.begin() + static_cast<std::ptrdiff_t>(source + shard_rows),
                  block.begin() + static_cast<std::ptrdiff_t>(token) * shard_rows);
    }
    return block;
}

int expect_cross_rank_difference(const std::string& label, const std::vector<double>& first,
                                 const std::vector<double>& second) {
    if (first == second) {
        std::cerr << label << ": both ranks produced identical blocks -- this stage did not "
                              "actually split\n";
        return 1;
    }
    return 0;
}

int expect_cross_rank_equality(const std::string& label, const std::vector<double>& first,
                               const std::vector<double>& second) {
    if (first != second) {
        std::cerr << label << ": the two ranks disagree after an all-reduce, which must leave a "
                              "bit-identical result on both\n";
        return 1;
    }
    return 0;
}

std::vector<float> random_bf16(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    round_to_bf16(values);
    return values;
}

} // namespace

// =================================================================================================
// LEG A -- the MTP verify round, composed at the op level exactly as mtp_forward_stem +
// mtp_forward_tail compose it, run at tp1 on device 0 and at tp2 across both devices.
// =================================================================================================
namespace {

// One logical weight plus its two standalone device shards. For a fused parent whose shard is the
// concatenation of several per-section blocks, `sections` keeps those blocks so shard identity can
// be proven SECTION BY SECTION against the parent (a whole-shard origin check is meaningless when
// the shard is not one contiguous parent block).
struct ShardSection {
    const char* name;
    std::int32_t parent_row_origin;
    qw::PackedWeight block;
};

struct SplitWeight {
    qw::PackedWeight parent;
    std::array<qw::PackedWeight, 2> shard;
    std::array<std::vector<ShardSection>, 2> sections;
};

// Column-parallel (output/N split) of a single-section parent: rank r owns rows [r*n/2,(r+1)*n/2).
SplitWeight make_column_split(QType qtype, std::int32_t n, std::int32_t k, std::uint32_t seed) {
    SplitWeight out;
    out.parent   = make_block(qtype, n, k, seed, 0);
    out.shard[0] = make_block(qtype, n / 2, k, seed, 0);
    out.shard[1] = make_block(qtype, n / 2, k, seed, n / 2);
    return out;
}

// Column-parallel of a TWO-section fused parent (mlp/gate_up: Gate | Up, each `section` rows split
// by its own halving). Rank r's shard is its half of Gate followed by its half of Up -- the order
// bindings.cpp's two append_column_block calls push.
SplitWeight make_two_section_column_split(QType qtype, std::int32_t section, std::int32_t k,
                                          std::uint32_t seed) {
    SplitWeight out;
    out.parent = make_block(qtype, 2 * section, k, seed, 0);
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot         = static_cast<std::size_t>(rank);
        const std::int32_t half = section / 2;
        qw::PackedWeight gate   = make_block(qtype, half, k, seed, rank * half);
        qw::PackedWeight up     = make_block(qtype, half, k, seed, section + rank * half);
        out.shard[slot]         = concat_row_blocks(gate, up);
        out.sections[slot].push_back({"gate", rank * half, std::move(gate)});
        out.sections[slot].push_back({"up", section + rank * half, std::move(up)});
    }
    return out;
}

// Column-parallel of the FOUR-section fused attention parent, in the ShardPlan's real
// Q | K | Gate | V order (four independent append_column_block calls in that order; see
// src/targets/qwen3_6_27b/impl/load/bindings.cpp, "attention/query_key_gate_value").
SplitWeight make_attention_split(QType qtype, std::uint32_t seed) {
    SplitWeight out;
    out.parent                       = make_block(qtype, kAttnRows, kHidden, seed, 0);
    constexpr std::int32_t half_q    = kQSize / 2;
    constexpr std::int32_t half_kv   = kKvSize / 2;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot                   = static_cast<std::size_t>(rank);
        const std::int32_t q_origin       = rank * half_q;
        const std::int32_t k_origin       = kQSize + rank * half_kv;
        const std::int32_t gate_origin    = kQSize + kKvSize + rank * half_q;
        const std::int32_t value_origin   = kQSize + kKvSize + kQSize + rank * half_kv;
        qw::PackedWeight q = make_block(qtype, half_q, kHidden, seed, q_origin);
        qw::PackedWeight k = make_block(qtype, half_kv, kHidden, seed, k_origin);
        qw::PackedWeight g = make_block(qtype, half_q, kHidden, seed, gate_origin);
        qw::PackedWeight v = make_block(qtype, half_kv, kHidden, seed, value_origin);
        out.shard[slot] =
            concat_row_blocks(concat_row_blocks(concat_row_blocks(q, k), g), v);
        out.sections[slot].push_back({"q", q_origin, std::move(q)});
        out.sections[slot].push_back({"k", k_origin, std::move(k)});
        out.sections[slot].push_back({"gate", gate_origin, std::move(g)});
        out.sections[slot].push_back({"v", value_origin, std::move(v)});
    }
    return out;
}

// Section-by-section shard identity for a fused column-parallel parent, plus intra-shard and
// cross-rank distinctness (the assertions that make the numbers below mean anything: without them
// an annihilating shard stride leaves both shards byte-identical and parity cannot tell a correct
// shard from a duplicate of the other half).
int verify_sectioned_column_shard(const std::string& label, const SplitWeight& split) {
    int failures = 0;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        const std::string rank_label = label + " shard " + std::to_string(rank);
        for (const ShardSection& section : split.sections[slot]) {
            failures += verify_shard_is_parent_block(rank_label + ' ' + section.name, split.parent,
                                                     section.block, section.parent_row_origin, 0);
        }
        for (std::size_t i = 0; i + 1 < split.sections[slot].size(); ++i) {
            for (std::size_t j = i + 1; j < split.sections[slot].size(); ++j) {
                failures += verify_shards_are_distinct(
                    rank_label + ' ' + split.sections[slot][i].name + "-vs-" +
                        split.sections[slot][j].name,
                    split.sections[slot][i].block.payload, split.sections[slot][j].block.payload);
            }
        }
    }
    failures += verify_shards_are_distinct(label, split.shard[0].payload, split.shard[1].payload);
    return failures;
}

// Row-parallel (input/K split): rank r owns columns [r*k/2,(r+1)*k/2).
SplitWeight make_row_split(QType qtype, std::int32_t n, std::int32_t k, std::uint32_t seed) {
    SplitWeight out;
    out.parent   = make_block(qtype, n, k, seed, 0, 0);
    out.shard[0] = make_block(qtype, n, k / 2, seed, 0, 0);
    out.shard[1] = make_block(qtype, n, k / 2, seed, 0, k / 2);
    return out;
}

struct UploadedSplit {
    DeviceWeight parent;                  // device 0 only
    std::array<DeviceWeight, 2> shard;
};

UploadedSplit upload_split(const ExecutionContext& ec, const SplitWeight& split) {
    UploadedSplit out;
    set_device(ec, 0);
    out.parent = upload_weight(split.parent);
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        out.shard[static_cast<std::size_t>(rank)] =
            upload_weight(split.shard[static_cast<std::size_t>(rank)]);
    }
    return out;
}

int verify_column_shard_identity(const std::string& label, const SplitWeight& split,
                                 std::int32_t half) {
    int failures = 0;
    for (int rank = 0; rank < 2; ++rank) {
        failures += verify_shard_is_parent_block(label + " shard " + std::to_string(rank),
                                                 split.parent,
                                                 split.shard[static_cast<std::size_t>(rank)],
                                                 rank * half, 0);
    }
    failures += verify_shards_are_distinct(label, split.shard[0].payload, split.shard[1].payload);
    return failures;
}

int verify_row_shard_identity(const std::string& label, const SplitWeight& split,
                              std::int32_t half) {
    int failures = 0;
    for (int rank = 0; rank < 2; ++rank) {
        failures += verify_shard_is_parent_block(label + " shard " + std::to_string(rank),
                                                 split.parent,
                                                 split.shard[static_cast<std::size_t>(rank)], 0,
                                                 rank * half);
    }
    failures += verify_shards_are_distinct(label, split.shard[0].payload, split.shard[1].payload);
    return failures;
}

// Head-major slice of one rank's KV heads out of the global cache (identical in shape to
// test_attention_headlocal.cpp's own slice_cache_heads; the cache planes are head-major, so a
// device's pool is one contiguous slab).
template <typename T>
std::vector<T> slice_head_major(const std::vector<T>& source, std::int32_t leading_extent,
                                std::int32_t padded_context, std::int32_t head_begin,
                                std::int32_t head_count) {
    const std::size_t slab =
        static_cast<std::size_t>(leading_extent) * static_cast<std::size_t>(padded_context);
    std::vector<T> result(slab * static_cast<std::size_t>(head_count));
    std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(slab * head_begin), result.size(),
                result.begin());
    return result;
}

gqa::HostCache slice_cache_heads(const gqa::HostCache& source, int rank) {
    const std::int32_t begin = rank * kLocalAttn.kv_heads;
    gqa::HostCache result{kLocalAttn, source.dtype, source.max_context, source.logical_capacity};
    result.k_bf16 = slice_head_major(source.k_bf16, gqa::kHeadDim, source.logical_capacity, begin,
                                     kLocalAttn.kv_heads);
    result.v_bf16 = slice_head_major(source.v_bf16, gqa::kHeadDim, source.logical_capacity, begin,
                                     kLocalAttn.kv_heads);
    return result;
}

std::vector<double> read_bf16(const DeviceBuffer& buffer, std::size_t elements) {
    return from_device_bf16(buffer.p, elements);
}

struct MtpModel {
    SplitWeight fc;       // [5120,10240] row-parallel
    SplitWeight attn;     // [14336,5120] column
    SplitWeight o_proj;   // [5120,6144]  row-parallel
    SplitWeight gate_up;  // [34816,5120] column
    SplitWeight down;     // [5120,17408] row-parallel
};

struct MtpDeviceModel {
    UploadedSplit fc, attn, o_proj, gate_up, down;
};

// Replicated per-device norm vectors.
struct NormWeights {
    std::array<DeviceBuffer, 2> embedding, hidden, input, post_attention, final_norm, q, k;
};

// ops::rmsnorm takes its gain as a contiguous BF16 [D] Tensor, not a quantized Weight.
Tensor bf16_vector(const DeviceBuffer& buffer, std::int32_t rows) {
    return Tensor(buffer.p, DType::BF16, {rows});
}

} // namespace

namespace {

// The seven replicated norm vectors of one MTP layer, materialized independently on each device
// (exactly what the ShardPlan's replicated family produces).
struct NormSet {
    std::array<DeviceBuffer, 2> embedding, hidden, input, post_attention, final_norm, query, key;
};

// The gain each RMSNorm applies is `1 + weight` (unit_offset), and RMSNorm forces its output to
// unit RMS before applying it, so the gain ALONE sets the scale of everything downstream of that
// norm. The fixture's W8 codes/scales give weight entries of RMS ~2.4, so a GEMM over K columns
// multiplies its input's RMS by ~2.4*sqrt(K). The gains below are chosen to land every GEMM output
// at O(1):
//
//   embedding/hidden norm -> fc      (K = 10240): 1/(2.4*sqrt(10240)) ~ 0.004
//   input norm            -> attn    (K =  5120): 1/(2.4*sqrt(5120))  ~ 0.006
//   post-attention norm   -> gate_up (K =  5120): ~0.006
//   query/key norm, final norm: gain ~1 (their consumers are the 256-wide softmax and the
//                                        suite's own output, both already scale-free)
//
// This is not cosmetic. With unit gains the projections land at magnitude ~4000, the attention
// softmax saturates into a hard argmax over the KV cache, and a one-ulp difference between the tp1
// and split projections flips WHICH key wins -- so the composition becomes chaotic and a parity
// comparison past the attention stage measures nothing. Real MTP activations are O(1) for exactly
// the same reason (their trained weights are ~1/sqrt(K)), so scaling the fixture to O(1) makes the
// suite MORE faithful to the path it stands in for, not less.
NormSet make_norms(const ExecutionContext& ec, std::uint32_t seed) {
    NormSet out;
    const auto upload_pair = [&](std::array<DeviceBuffer, 2>& slot, std::int32_t rows,
                                 std::uint32_t local_seed, float gain) {
        // weight = gain - 1 (+/- 30% jitter on the gain), so `1 + weight` is the intended gain.
        std::vector<float> values =
            random_bf16(static_cast<std::size_t>(rows), local_seed, 0.7F * gain, 1.3F * gain);
        for (float& value : values) { value -= 1.0F; }
        round_to_bf16(values);
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            slot[static_cast<std::size_t>(rank)] = to_device_bf16(values);
        }
    };
    upload_pair(out.embedding, kHidden, seed + 1u, 0.004F);
    upload_pair(out.hidden, kHidden, seed + 2u, 0.004F);
    upload_pair(out.input, kHidden, seed + 3u, 0.006F);
    upload_pair(out.post_attention, kHidden, seed + 4u, 0.006F);
    upload_pair(out.final_norm, kHidden, seed + 5u, 1.0F);
    upload_pair(out.query, kHeadDim, seed + 6u, 1.0F);
    upload_pair(out.key, kHeadDim, seed + 7u, 1.0F);
    return out;
}

// One MTP verify round. The tp1 leg runs the same op sequence on device 0 over whole weights; the
// tp2 leg runs it across both devices over shard weights, and every stage is compared.
int run_mtp_round(const ExecutionContext& ec, const ops::PeerEvents& events,
                  const MtpDeviceModel& weights, const NormSet& norms, std::int32_t tokens,
                  std::int32_t base_keys, std::uint32_t seed) {
    const std::string head = "mtp round T=" + std::to_string(tokens) +
                             " base_keys=" + std::to_string(base_keys);
    std::cout << head << '\n';
    int failures = 0;

    const std::size_t hidden_elements = static_cast<std::size_t>(kHidden) * tokens;
    const std::vector<float> emb_host  = random_bf16(hidden_elements, seed + 11u, -1.0F, 1.0F);
    const std::vector<float> prev_host = random_bf16(hidden_elements, seed + 12u, -1.0F, 1.0F);

    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = base_keys + token;
    }
    const auto total_keys = static_cast<std::uint32_t>(base_keys + tokens);
    const ops::GqaExecutionEnvelope envelope{total_keys, total_keys};

    // One logical KV pool; device 0 gets all 4 KV heads for the tp1 leg, rank r gets its own 2.
    const std::int32_t max_context = base_keys + tokens;
    const gqa::HostCache global_cache =
        gqa::make_cache(kGlobalAttn, DType::BF16, max_context, seed + 13u);
    const std::array<gqa::HostCache, 2> local_cache{slice_cache_heads(global_cache, 0),
                                                    slice_cache_heads(global_cache, 1)};

    const std::size_t attn_workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        kGlobalAttn.q_heads, DType::BF16, envelope, 1, tokens, tokens);
    const std::size_t shard_attn_workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        kLocalAttn.q_heads, DType::BF16, envelope, 1, tokens, tokens);

    // ---------------------------------------------------------------------------------------
    // (a) tp1 reference on device 0.
    // ---------------------------------------------------------------------------------------
    std::vector<double> ref_x, ref_ah, ref_packed, ref_attn, ref_residual_after_o, ref_mh,
        ref_gate_up, ref_act, ref_hidden;
    {
        set_device(ec, 0);
        cudaStream_t s = ec.dev[0]->stream;
        gqa::DeviceCache cache(global_cache, gqa::MappingPattern::Identity);

        DeviceBuffer emb   = to_device_bf16(emb_host);
        DeviceBuffer prev  = to_device_bf16(prev_host);
        DeviceBuffer e(hidden_elements * sizeof(std::uint16_t));
        DeviceBuffer h(hidden_elements * sizeof(std::uint16_t));
        DeviceBuffer fc_in(static_cast<std::size_t>(kFcInputRows) * tokens *
                           sizeof(std::uint16_t));
        DeviceBuffer x(hidden_elements * sizeof(std::uint16_t));
        DeviceBuffer ah(hidden_elements * sizeof(std::uint16_t));
        DeviceBuffer packed(static_cast<std::size_t>(kAttnRows) * tokens * sizeof(std::uint16_t));
        DeviceBuffer q(static_cast<std::size_t>(kQSize) * tokens * sizeof(std::uint16_t));
        DeviceBuffer k(static_cast<std::size_t>(kKvSize) * tokens * sizeof(std::uint16_t));
        DeviceBuffer gate(static_cast<std::size_t>(kQSize) * tokens * sizeof(std::uint16_t));
        DeviceBuffer v(static_cast<std::size_t>(kKvSize) * tokens * sizeof(std::uint16_t));
        DeviceBuffer qn(static_cast<std::size_t>(kQSize) * tokens * sizeof(std::uint16_t));
        DeviceBuffer kn(static_cast<std::size_t>(kKvSize) * tokens * sizeof(std::uint16_t));
        DeviceBuffer attn(static_cast<std::size_t>(kQSize) * tokens * sizeof(std::uint16_t));
        DeviceBuffer o(hidden_elements * sizeof(std::uint16_t));
        DeviceBuffer mh(hidden_elements * sizeof(std::uint16_t));
        DeviceBuffer gate_up(static_cast<std::size_t>(kGateUpRows) * tokens *
                             sizeof(std::uint16_t));
        DeviceBuffer act(static_cast<std::size_t>(kIntermediate) * tokens * sizeof(std::uint16_t));
        DeviceBuffer delta(hidden_elements * sizeof(std::uint16_t));
        DeviceBuffer hidden_out(hidden_elements * sizeof(std::uint16_t));
        DeviceBuffer dpos      = to_device_i32(positions);
        const std::vector<std::int32_t> table_row_host{0};
        DeviceBuffer dtable    = to_device_i32(table_row_host);
        DeviceBuffer workspace_storage(std::max<std::size_t>(attn_workspace_bytes, 256));
        WorkspaceArena workspace(DeviceSpan{workspace_storage.p, workspace_storage.bytes});

        const Tensor t_emb(emb.p, DType::BF16, {kHidden, tokens});
        const Tensor t_prev(prev.p, DType::BF16, {kHidden, tokens});
        Tensor t_e(e.p, DType::BF16, {kHidden, tokens});
        Tensor t_h(h.p, DType::BF16, {kHidden, tokens});
        Tensor t_fc_in(fc_in.p, DType::BF16, {kFcInputRows, tokens});
        Tensor t_x(x.p, DType::BF16, {kHidden, tokens});
        Tensor t_ah(ah.p, DType::BF16, {kHidden, tokens});
        Tensor t_packed(packed.p, DType::BF16, {kAttnRows, tokens});
        Tensor t_q(q.p, DType::BF16, {kHeadDim, kGlobalAttn.q_heads, tokens});
        Tensor t_k(k.p, DType::BF16, {kHeadDim, kGlobalAttn.kv_heads, tokens});
        Tensor t_gate(gate.p, DType::BF16, {kHeadDim, kGlobalAttn.q_heads, tokens});
        Tensor t_v(v.p, DType::BF16, {kHeadDim, kGlobalAttn.kv_heads, tokens});
        Tensor t_qn(qn.p, DType::BF16, {kHeadDim, kGlobalAttn.q_heads, tokens});
        Tensor t_kn(kn.p, DType::BF16, {kHeadDim, kGlobalAttn.kv_heads, tokens});
        Tensor t_attn(attn.p, DType::BF16, {kHeadDim, kGlobalAttn.q_heads, tokens});
        Tensor t_o(o.p, DType::BF16, {kHidden, tokens});
        Tensor t_mh(mh.p, DType::BF16, {kHidden, tokens});
        Tensor t_gate_up(gate_up.p, DType::BF16, {kGateUpRows, tokens});
        Tensor t_act(act.p, DType::BF16, {kIntermediate, tokens});
        Tensor t_delta(delta.p, DType::BF16, {kHidden, tokens});
        Tensor t_hidden(hidden_out.p, DType::BF16, {kHidden, tokens});
        const Tensor t_pos(dpos.p, DType::I32, {tokens});
        const Tensor t_table(dtable.p, DType::I32, {1});

        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        // -- stem
        ops::rmsnorm(t_emb, bf16_vector(norms.embedding[0], kHidden), kRmsEps, true, t_e, s);
        ops::rmsnorm(t_prev, bf16_vector(norms.hidden[0], kHidden), kRmsEps, true, t_h, s);
        ops::mtp_pack_fc_input(t_e, t_h, t_fc_in, s);
        ops::linear(t_fc_in, weights.fc.parent.weight, t_x, s);
        // `x` is the MTP layer's residual and is mutated in place by both residual_add calls
        // below, so the fc stage's own output has to be captured here, not at the end.
        cuda_check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        ref_x = read_bf16(x, hidden_elements);
        ops::rmsnorm(t_x, bf16_vector(norms.input[0], kHidden), kRmsEps, true, t_ah, s);
        // -- attention
        ops::linear(t_ah, weights.attn.parent.weight, t_packed, s);
        ops::mtp_split_attn_in(t_packed, t_q, t_k, t_gate, t_v, s);
        ops::rmsnorm(t_q, bf16_vector(norms.query[0], kHeadDim), kRmsEps, true, t_qn, s);
        ops::rmsnorm(t_k, bf16_vector(norms.key[0], kHeadDim), kRmsEps, true, t_kn, s);
        ops::rope(t_pos, kRotaryDim, kRopeTheta, t_qn, t_kn, s);
        ops::gqa_attention(t_qn, t_kn, t_v, t_pos, Tensor{}, t_table, gqa::kAttentionScale,
                           cache.batch_view(), envelope, workspace, t_attn, s);
        ops::sigmoid_mul(t_gate, t_attn, s);
        Tensor attn_flat = t_attn.view({kQSize, tokens});
        ops::linear(attn_flat, weights.o_proj.parent.weight, t_o, s);
        ops::residual_add(t_o, t_x, s);
        // -- post mixer
        ops::rmsnorm(t_x, bf16_vector(norms.post_attention[0], kHidden), kRmsEps, true, t_mh,
                     s);
        ops::linear(t_mh, weights.gate_up.parent.weight, t_gate_up, s);
        ops::silu_mul(t_gate_up.slice(0, 0, kIntermediate),
                      t_gate_up.slice(0, kIntermediate, kIntermediate), t_act, s);
        ops::linear(t_act, weights.down.parent.weight, t_delta, s);
        ops::residual_add(t_delta, t_x, s);
        ops::rmsnorm(t_x, bf16_vector(norms.final_norm[0], kHidden), kRmsEps, true, t_hidden,
                     s);
        cuda_check(cudaStreamSynchronize(s), "cudaStreamSynchronize");

        ref_ah      = read_bf16(ah, hidden_elements);
        ref_packed  = read_bf16(packed, static_cast<std::size_t>(kAttnRows) * tokens);
        ref_attn    = read_bf16(attn, static_cast<std::size_t>(kQSize) * tokens);
        ref_mh      = read_bf16(mh, hidden_elements);
        ref_gate_up = read_bf16(gate_up, static_cast<std::size_t>(kGateUpRows) * tokens);
        ref_act     = read_bf16(act, static_cast<std::size_t>(kIntermediate) * tokens);
        ref_hidden  = read_bf16(hidden_out, hidden_elements);
    }

    // ---------------------------------------------------------------------------------------
    // (b) tp2: the same op sequence, shard-shaped, across both devices.
    // ---------------------------------------------------------------------------------------
    std::array<std::optional<gqa::DeviceCache>, 2> shard_cache;
    std::array<std::optional<DeviceBuffer>, 2> emb_d, prev_d, e_d, h_d, x_d, ah_d, packed_d, q_d,
        k_d, gate_d, v_d, qn_d, kn_d, attn_d, o_d, mh_d, gate_up_d, act_d, delta_d, hidden_d,
        staging_d, pos_d, table_d, ws_d;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        shard_cache[slot].emplace(local_cache[slot], gqa::MappingPattern::Identity);
        emb_d[slot].emplace(to_device_bf16(emb_host));
        prev_d[slot].emplace(to_device_bf16(prev_host));
        e_d[slot].emplace(hidden_elements * sizeof(std::uint16_t));
        h_d[slot].emplace(hidden_elements * sizeof(std::uint16_t));
        x_d[slot].emplace(hidden_elements * sizeof(std::uint16_t));
        ah_d[slot].emplace(hidden_elements * sizeof(std::uint16_t));
        packed_d[slot].emplace(static_cast<std::size_t>(kAttnShardRows) * tokens *
                               sizeof(std::uint16_t));
        q_d[slot].emplace(static_cast<std::size_t>(kShardQSize) * tokens * sizeof(std::uint16_t));
        k_d[slot].emplace(static_cast<std::size_t>(kShardKvSize) * tokens * sizeof(std::uint16_t));
        gate_d[slot].emplace(static_cast<std::size_t>(kShardQSize) * tokens *
                             sizeof(std::uint16_t));
        v_d[slot].emplace(static_cast<std::size_t>(kShardKvSize) * tokens * sizeof(std::uint16_t));
        qn_d[slot].emplace(static_cast<std::size_t>(kShardQSize) * tokens * sizeof(std::uint16_t));
        kn_d[slot].emplace(static_cast<std::size_t>(kShardKvSize) * tokens * sizeof(std::uint16_t));
        attn_d[slot].emplace(static_cast<std::size_t>(kShardQSize) * tokens *
                             sizeof(std::uint16_t));
        o_d[slot].emplace(hidden_elements * sizeof(std::uint16_t));
        mh_d[slot].emplace(hidden_elements * sizeof(std::uint16_t));
        gate_up_d[slot].emplace(static_cast<std::size_t>(kGateUpShardRows) * tokens *
                                sizeof(std::uint16_t));
        act_d[slot].emplace(static_cast<std::size_t>(kShardIntermediate) * tokens *
                            sizeof(std::uint16_t));
        delta_d[slot].emplace(hidden_elements * sizeof(std::uint16_t));
        hidden_d[slot].emplace(hidden_elements * sizeof(std::uint16_t));
        staging_d[slot].emplace(hidden_elements * sizeof(std::uint16_t));
        pos_d[slot].emplace(to_device_i32(positions));
        table_d[slot].emplace(to_device_i32(std::vector<std::int32_t>{0}));
        ws_d[slot].emplace(std::max<std::size_t>(shard_attn_workspace_bytes, 256));
    }

    const auto tensor2 = [](std::array<std::optional<DeviceBuffer>, 2>& buffers, DType dtype,
                            std::initializer_list<std::int32_t> shape) {
        return std::array<Tensor, 2>{Tensor(buffers[0]->p, dtype, shape),
                                     Tensor(buffers[1]->p, dtype, shape)};
    };

    const std::array<Tensor, 2> t_e  = tensor2(e_d, DType::BF16, {kHidden, tokens});
    const std::array<Tensor, 2> t_h  = tensor2(h_d, DType::BF16, {kHidden, tokens});
    const std::array<Tensor, 2> t_x  = tensor2(x_d, DType::BF16, {kHidden, tokens});
    const std::array<Tensor, 2> t_ah = tensor2(ah_d, DType::BF16, {kHidden, tokens});
    const std::array<Tensor, 2> t_packed =
        tensor2(packed_d, DType::BF16, {kAttnShardRows, tokens});
    const std::array<Tensor, 2> t_o       = tensor2(o_d, DType::BF16, {kHidden, tokens});
    const std::array<Tensor, 2> t_mh      = tensor2(mh_d, DType::BF16, {kHidden, tokens});
    const std::array<Tensor, 2> t_gate_up =
        tensor2(gate_up_d, DType::BF16, {kGateUpShardRows, tokens});
    const std::array<Tensor, 2> t_act =
        tensor2(act_d, DType::BF16, {kShardIntermediate, tokens});
    const std::array<Tensor, 2> t_delta   = tensor2(delta_d, DType::BF16, {kHidden, tokens});
    const std::array<Tensor, 2> t_hidden  = tensor2(hidden_d, DType::BF16, {kHidden, tokens});
    const std::array<Tensor, 2> t_staging = tensor2(staging_d, DType::BF16, {kHidden, tokens});

    // -- stage 1: norms (replicated) then the ROW-PARALLEL fc. No pack: rank 0 feeds `e`
    //    (packed rows [0,5120)) and rank 1 feeds `h` (rows [5120,10240)) -- see mtp_pack.h.
    retire_staging(ec);
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        cudaStream_t s = ec.dev[rank]->stream;
        Tensor emb(emb_d[slot]->p, DType::BF16, {kHidden, tokens});
        Tensor prev(prev_d[slot]->p, DType::BF16, {kHidden, tokens});
        Tensor e = t_e[slot];
        Tensor h = t_h[slot];
        ops::rmsnorm(emb, bf16_vector(norms.embedding[slot], kHidden), kRmsEps, true, e, s);
        ops::rmsnorm(prev, bf16_vector(norms.hidden[slot], kHidden), kRmsEps, true, h, s);
    }
    synchronize_both(ec);
    {
        const std::array<Tensor, 2> fc_input{t_e[0], t_h[1]};
        const std::array<Weight, 2> fc_w{weights.fc.shard[0].weight, weights.fc.shard[1].weight};
        ops::linear_row_parallel(fc_input, fc_w, t_x, t_staging, ec, events);
    }
    synchronize_both(ec);
    {
        std::array<std::vector<double>, 2> observed;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            observed[slot] = read_bf16(*x_d[slot], hidden_elements);
            failures += compare(head + " stage1 fc rank " + std::to_string(rank), observed[slot],
                                ref_x, kSplitCriterion);
        }
        failures += expect_cross_rank_equality(head + " stage1 fc", observed[0], observed[1]);
    }

    // -- stage 2: input_norm (replicated, per rank)
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        Tensor x  = t_x[slot];
        Tensor ah = t_ah[slot];
        ops::rmsnorm(x, bf16_vector(norms.input[slot], kHidden), kRmsEps, true, ah,
                     ec.dev[rank]->stream);
    }
    synchronize_both(ec);
    {
        std::array<std::vector<double>, 2> observed;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            observed[slot] = read_bf16(*ah_d[slot], hidden_elements);
            failures += compare(head + " stage2 input_norm rank " + std::to_string(rank),
                                observed[slot], ref_ah, kChainedCriterion);
        }
        failures += expect_cross_rank_equality(head + " stage2 input_norm", observed[0],
                                               observed[1]);
    }

    // -- stage 3: COLUMN-PARALLEL fused attention projection, then the shard split_attn_in.
    {
        const std::array<Weight, 2> w{weights.attn.shard[0].weight, weights.attn.shard[1].weight};
        ops::linear_column_parallel(t_ah, w, t_packed, ec);
    }
    synchronize_both(ec);
    {
        // Per-SECTION provenance: rank r's shard rows must be rank r's own half of every one of the
        // parent's four sections, in Q | K | Gate | V order.
        struct Section {
            const char* name;
            std::int32_t parent_begin;
            std::int32_t parent_rows;
            std::int32_t shard_begin;
            std::int32_t shard_rows;
        };
        const std::array<Section, 4> sections{
            Section{"q", 0, kQSize, 0, kShardQSize},
            Section{"k", kQSize, kKvSize, kShardQSize, kShardKvSize},
            Section{"gate", kQSize + kKvSize, kQSize, kShardQSize + kShardKvSize, kShardQSize},
            Section{"v", kQSize + kKvSize + kQSize, kKvSize,
                    2 * kShardQSize + kShardKvSize, kShardKvSize}};
        std::array<std::vector<double>, 2> observed;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            observed[slot] =
                read_bf16(*packed_d[slot], static_cast<std::size_t>(kAttnShardRows) * tokens);
        }
        for (const Section& section : sections) {
            for (int rank = 0; rank < 2; ++rank) {
                const auto slot = static_cast<std::size_t>(rank);
                const std::vector<double> got = extract_rank_block(
                    observed[slot], kAttnShardRows, section.shard_begin, section.shard_rows, tokens);
                const std::vector<double> expected = extract_rank_block(
                    ref_packed, kAttnRows,
                    section.parent_begin + rank * (section.parent_rows / 2),
                    section.shard_rows, tokens);
                failures += compare(head + " stage3 attn_proj " + section.name + " rank " +
                                        std::to_string(rank),
                                    got, expected, kChainedCriterion);
            }
        }
        failures += expect_cross_rank_difference(head + " stage3 attn_proj", observed[0],
                                                 observed[1]);
    }

    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        cudaStream_t s = ec.dev[rank]->stream;
        Tensor packed  = t_packed[slot];
        Tensor q(q_d[slot]->p, DType::BF16, {kHeadDim, kLocalAttn.q_heads, tokens});
        Tensor k(k_d[slot]->p, DType::BF16, {kHeadDim, kLocalAttn.kv_heads, tokens});
        Tensor gate(gate_d[slot]->p, DType::BF16, {kHeadDim, kLocalAttn.q_heads, tokens});
        Tensor v(v_d[slot]->p, DType::BF16, {kHeadDim, kLocalAttn.kv_heads, tokens});
        Tensor qn(qn_d[slot]->p, DType::BF16, {kHeadDim, kLocalAttn.q_heads, tokens});
        Tensor kn(kn_d[slot]->p, DType::BF16, {kHeadDim, kLocalAttn.kv_heads, tokens});
        Tensor attn(attn_d[slot]->p, DType::BF16, {kHeadDim, kLocalAttn.q_heads, tokens});
        const Tensor pos(pos_d[slot]->p, DType::I32, {tokens});
        const Tensor table(table_d[slot]->p, DType::I32, {1});
        WorkspaceArena workspace(DeviceSpan{ws_d[slot]->p, ws_d[slot]->bytes});

        ops::mtp_split_attn_in(packed, q, k, gate, v, s);
        ops::rmsnorm(q, bf16_vector(norms.query[slot], kHeadDim), kRmsEps, true, qn, s);
        ops::rmsnorm(k, bf16_vector(norms.key[slot], kHeadDim), kRmsEps, true, kn, s);
        ops::rope(pos, kRotaryDim, kRopeTheta, qn, kn, s);
        ops::gqa_attention(qn, kn, v, pos, Tensor{}, table, gqa::kAttentionScale,
                           shard_cache[slot]->batch_view(), envelope, workspace, attn, s);
        ops::sigmoid_mul(gate, attn, s);
    }
    synchronize_both(ec);
    {
        // Per-head provenance: rank r's local query head h is global head 12r + h.
        std::array<std::vector<double>, 2> observed;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            observed[slot] =
                read_bf16(*attn_d[slot], static_cast<std::size_t>(kShardQSize) * tokens);
            const std::vector<double> expected =
                extract_rank_block(ref_attn, kQSize, rank * kShardQSize, kShardQSize, tokens);
            failures += compare(head + " stage4 attention rank " + std::to_string(rank),
                                observed[slot], expected, kChainedCriterion);
        }
        failures += expect_cross_rank_difference(head + " stage4 attention", observed[0],
                                                 observed[1]);
    }

    // -- stage 5: ROW-PARALLEL o_proj, then the replicated residual add.
    {
        const std::array<Tensor, 2> attn_flat{
            Tensor(attn_d[0]->p, DType::BF16, {kOProjShardK, tokens}),
            Tensor(attn_d[1]->p, DType::BF16, {kOProjShardK, tokens})};
        const std::array<Weight, 2> w{weights.o_proj.shard[0].weight,
                                      weights.o_proj.shard[1].weight};
        ops::linear_row_parallel(attn_flat, w, t_o, t_staging, ec, events);
    }
    synchronize_both(ec);
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        Tensor o = t_o[slot];
        Tensor x = t_x[slot];
        ops::residual_add(o, x, ec.dev[rank]->stream);
        Tensor mh = t_mh[slot];
        ops::rmsnorm(x, bf16_vector(norms.post_attention[slot], kHidden), kRmsEps, true, mh,
                     ec.dev[rank]->stream);
    }
    synchronize_both(ec);
    {
        std::array<std::vector<double>, 2> observed;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            observed[slot] = read_bf16(*mh_d[slot], hidden_elements);
            failures += compare(head + " stage5 post_attn_norm rank " + std::to_string(rank),
                                observed[slot], ref_mh, kChainedCriterion);
        }
        failures += expect_cross_rank_equality(head + " stage5 post_attn_norm", observed[0],
                                               observed[1]);
    }

    // -- stage 6: COLUMN-PARALLEL gate_up + local silu_mul over the shard's own halves.
    {
        const std::array<Weight, 2> w{weights.gate_up.shard[0].weight,
                                      weights.gate_up.shard[1].weight};
        ops::linear_column_parallel(t_mh, w, t_gate_up, ec);
    }
    synchronize_both(ec);
    {
        std::array<std::vector<double>, 2> observed;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            observed[slot] =
                read_bf16(*gate_up_d[slot], static_cast<std::size_t>(kGateUpShardRows) * tokens);
            // Per-HALF provenance: the shard is [this rank's half of Gate | this rank's half of Up].
            const std::vector<double> gate_half = extract_rank_block(
                observed[slot], kGateUpShardRows, 0, kShardIntermediate, tokens);
            const std::vector<double> up_half = extract_rank_block(
                observed[slot], kGateUpShardRows, kShardIntermediate, kShardIntermediate, tokens);
            failures += compare(head + " stage6 gate_up gate rank " + std::to_string(rank),
                                gate_half,
                                extract_rank_block(ref_gate_up, kGateUpRows,
                                                   rank * kShardIntermediate, kShardIntermediate,
                                                   tokens),
                                kChainedCriterion);
            failures += compare(head + " stage6 gate_up up rank " + std::to_string(rank), up_half,
                                extract_rank_block(ref_gate_up, kGateUpRows,
                                                   kIntermediate + rank * kShardIntermediate,
                                                   kShardIntermediate, tokens),
                                kChainedCriterion);
            failures += verify_shards_are_distinct(
                head + " stage6 gate_up gate-vs-up rank " + std::to_string(rank), gate_half,
                up_half);
        }
        failures += expect_cross_rank_difference(head + " stage6 gate_up", observed[0],
                                                 observed[1]);
    }
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        Tensor gate_up = t_gate_up[slot];
        Tensor act     = t_act[slot];
        ops::silu_mul(gate_up.slice(0, 0, kShardIntermediate),
                      gate_up.slice(0, kShardIntermediate, kShardIntermediate), act,
                      ec.dev[rank]->stream);
    }
    synchronize_both(ec);
    {
        std::array<std::vector<double>, 2> observed;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            observed[slot] =
                read_bf16(*act_d[slot], static_cast<std::size_t>(kShardIntermediate) * tokens);
            failures += compare(head + " stage7 silu_mul rank " + std::to_string(rank),
                                observed[slot],
                                extract_rank_block(ref_act, kIntermediate,
                                                   rank * kShardIntermediate, kShardIntermediate,
                                                   tokens),
                                kChainedCriterion);
        }
        failures += expect_cross_rank_difference(head + " stage7 silu_mul", observed[0],
                                                 observed[1]);
    }

    // -- stage 8: ROW-PARALLEL down + residual, then the final norm.
    {
        const std::array<Weight, 2> w{weights.down.shard[0].weight, weights.down.shard[1].weight};
        ops::linear_row_parallel(t_act, w, t_delta, t_staging, ec, events);
    }
    synchronize_both(ec);
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        Tensor delta = t_delta[slot];
        Tensor x     = t_x[slot];
        ops::residual_add(delta, x, ec.dev[rank]->stream);
        Tensor hidden = t_hidden[slot];
        ops::rmsnorm(x, bf16_vector(norms.final_norm[slot], kHidden), kRmsEps, true, hidden,
                     ec.dev[rank]->stream);
    }
    synchronize_both(ec);
    {
        std::array<std::vector<double>, 2> observed;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            observed[slot] = read_bf16(*hidden_d[slot], hidden_elements);
            failures += compare(head + " stage8 mtp_hidden rank " + std::to_string(rank),
                                observed[slot], ref_hidden, kChainedCriterion);
        }
        failures += expect_cross_rank_equality(head + " stage8 mtp_hidden", observed[0],
                                               observed[1]);
    }

    return failures;
}

int run_leg_a(const ExecutionContext& ec, const ops::PeerEvents& events) {
    std::cout << "\n--- Leg A: MTP verify round (W8G32, op-level composition) ---\n";
    constexpr QType kW8 = QType::W8G32_F16S;
    int failures        = 0;

    MtpModel model;
    model.fc      = make_row_split(kW8, kHidden, kFcInputRows, 4101u);
    model.attn    = make_attention_split(kW8, 4102u);
    model.o_proj  = make_row_split(kW8, kHidden, kQSize, 4103u);
    model.gate_up = make_two_section_column_split(kW8, kIntermediate, kHidden, 4104u);
    model.down    = make_row_split(kW8, kHidden, kIntermediate, 4105u);

    failures += verify_row_shard_identity("mtp/input_projection", model.fc, kFcShardK);
    failures += verify_row_shard_identity("mtp/layer/attention/output", model.o_proj, kOProjShardK);
    failures += verify_row_shard_identity("mtp/layer/mlp/down", model.down, kShardIntermediate);
    // The two fused shards are NOT single contiguous parent blocks, so they are verified section by
    // section (four sections for the attention parent, two for gate_up).
    failures +=
        verify_sectioned_column_shard("mtp/layer/attention/query_key_gate_value", model.attn);
    failures += verify_sectioned_column_shard("mtp/layer/mlp/gate_up", model.gate_up);
    if (failures != 0) { return failures; }

    MtpDeviceModel device_model;
    device_model.fc      = upload_split(ec, model.fc);
    device_model.attn    = upload_split(ec, model.attn);
    device_model.o_proj  = upload_split(ec, model.o_proj);
    device_model.gate_up = upload_split(ec, model.gate_up);
    device_model.down    = upload_split(ec, model.down);

    const NormSet norms = make_norms(ec, 4200u);

    // T = 1 is the AR proposal step; T = 6 is a K = 5 verify window (K + 1 columns), the widths
    // mtp_forward_batch actually drives.
    failures += run_mtp_round(ec, events, device_model, norms, 1, 64, 4300u);
    failures += run_mtp_round(ec, events, device_model, norms, 6, 129, 4400u);
    return failures;
}

} // namespace

// =================================================================================================
// LEG B -- CONTRACT 1: FoldGeometry<48, 8, 24, 5120>.
//
// This is the one place in the GDN family where a head split is not free:
// `launch_replay_fold` selects a COMPILE-TIME FoldGeometry from
// (layers, qk_heads, value_heads, conv_channels) and throws for anything unregistered
// (recurrent.cu's trailing `throw std::invalid_argument`). The tp2 registration halves the QK-head
// parameter as well as the value heads and the conv channels, because the fold kernel resolves its
// own value-head -> qk-head map from that pair.
//
// The fold is head-parallel (grid.x = Geometry::kValueHeads, recurrent.cu) and the conv history it
// writes is per channel, so the shard is expected to be BITWISE identical to the tp1 geometry's
// per-head/per-channel result: local value head h is global 24r + h, and h/(24/8) = h/3 =
// (24r + h)/3 - 8r, so a group-aligned split preserves the map exactly. This leg asserts exactly
// that, which makes it a falsifiable prediction rather than a tolerance.
// =================================================================================================
namespace {

constexpr std::int32_t kFoldLayers      = 48;
constexpr std::int32_t kStateDim        = 128;
constexpr std::int32_t kFoldQkHeads     = 16;
constexpr std::int32_t kFoldValueHeads  = 48;
constexpr std::int32_t kFoldConvChannels = 10240;
constexpr std::int32_t kShardQkHeads    = 8;
constexpr std::int32_t kShardValueHeads = 24;
constexpr std::int32_t kShardConvChannels = 5120;
constexpr std::int32_t kRecordCapacity  = 2;
constexpr std::int32_t kFoldWidth       = 4;
constexpr std::int32_t kFoldSlots       = 3;

constexpr std::int32_t kGdnChannels = 10240; // GDN convolution_dim (2*key_dim + value_dim)

// The GDN conv channel map, identical to `gdn/convolution`'s three-block ShardPlan (see
// bindings.cpp and tests/targets/test_shard_map.cpp): Q [0,2048) | K [2048,4096) |
// V [4096,10240), each split by its own head count.
std::int32_t global_conv_channel(std::int32_t local, int rank) {
    constexpr std::int32_t kKeyDim = 2048;
    if (local < 1024) { return rank * 1024 + local; }
    if (local < 2048) { return kKeyDim + rank * 1024 + (local - 1024); }
    return 2 * kKeyDim + rank * 3072 + (local - 2048);
}

// Leg B builds its shard records AND checks them through `global_conv_channel`, so a consistently
// wrong map would pass silently -- the exact circularity a negative control exposed in the GDN
// head-split suite's own conv leg (dropping the K section's per-rank offset from the map went
// undetected until the map was pinned independently). This pins it against the ShardPlan's own
// boundaries (Q [0,2048) | K [2048,4096) | V [4096,10240), each split by its head count) and
// asserts the
// two ranks' channel sets are disjoint and together cover [0,10240).
int verify_conv_channel_map() {
    int failures = 0;
    const struct {
        std::int32_t local_begin;
        std::int32_t count;
        std::int32_t parent_begin;
        std::int32_t parent_section_size;
    } blocks[] = {{0, 1024, 0, 2048}, {1024, 1024, 2048, 2048}, {2048, 3072, 4096, 6144}};
    std::vector<int> owner(static_cast<std::size_t>(kGdnChannels), -1);
    for (int rank = 0; rank < 2; ++rank) {
        for (const auto& block : blocks) {
            for (std::int32_t i = 0; i < block.count; ++i) {
                const std::int32_t local    = block.local_begin + i;
                const std::int32_t expected = block.parent_begin +
                                              rank * (block.parent_section_size / 2) + i;
                const std::int32_t got = global_conv_channel(local, rank);
                if (got != expected) {
                    if (failures == 0) {
                        std::cerr << "  gdn conv channel map: rank " << rank << " local " << local
                                  << " -> " << got << ", the ShardPlan says " << expected << '\n';
                    }
                    ++failures;
                    continue;
                }
                if (got < 0 || got >= kGdnChannels) {
                    ++failures;
                    continue;
                }
                if (owner[static_cast<std::size_t>(got)] != -1) {
                    std::cerr << "  gdn conv channel map: global channel " << got
                              << " is claimed by both devices\n";
                    ++failures;
                    continue;
                }
                owner[static_cast<std::size_t>(got)] = rank;
            }
        }
    }
    for (std::int32_t channel = 0; channel < kGdnChannels; ++channel) {
        if (owner[static_cast<std::size_t>(channel)] == -1) {
            std::cerr << "  gdn conv channel map: global channel " << channel
                      << " is owned by no device\n";
            ++failures;
            break;
        }
    }
    std::cout << (failures ? "FAIL" : "OK")
              << " gdn conv channel map: 2 x 5120 disjoint channels covering [0,10240)\n";
    return failures;
}

std::uint32_t mix32(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

float signed_pattern(std::uint32_t key, float magnitude) {
    const auto centered = static_cast<std::int32_t>(mix32(key) % 2001U) - 1000;
    return static_cast<float>(centered) * (magnitude / 1000.0F);
}

std::uint16_t bf16_pattern(std::uint32_t key, float magnitude) {
    return f32_to_bf16(signed_pattern(key, magnitude));
}

// One side of the fold comparison: a record set + state pool at one geometry, folded in place.
struct FoldSide {
    std::optional<DeviceBuffer> record_storage;
    std::optional<DeviceBuffer> state_storage;
    std::optional<GdnReplayRecords> records;
    std::optional<LinearAttentionStatePool> pool;
};

// Builds every record and initial state as a pure function of the GLOBAL (layer, row, token, head,
// dim) coordinate, so a head or channel permutation -- across devices or inside one device's
// window -- changes bytes and cannot pass any comparison.
void fill_fold_side(FoldSide& side, std::int32_t qk_heads, std::int32_t value_heads,
                    std::int32_t conv_channels, int rank, std::uint32_t seed,
                    const std::vector<std::int32_t>& slots,
                    const std::vector<std::int32_t>& commits) {
    const auto rows = static_cast<std::int32_t>(slots.size());
    const GdnReplayRecordSpec spec{
        .layers          = kFoldLayers,
        .record_capacity = kRecordCapacity,
        .width           = kFoldWidth,
        .conv_channels   = conv_channels,
        .qk_heads        = qk_heads,
        .value_heads     = value_heads,
        .key_dim         = kStateDim,
        .value_dim       = kStateDim,
    };
    LayoutBuilder record_builder;
    const GdnReplayRecordLayout record_layout = plan_gdn_replay_records(record_builder, spec);
    const std::size_t record_bytes            = record_builder.finish(256);
    side.record_storage.emplace(record_bytes);
    cuda_check(cudaMemset(side.record_storage->p, 0, record_bytes), "clear replay records");
    side.records.emplace(DeviceSpan{side.record_storage->p, record_bytes}, record_layout);
    const GdnReplayRecords& records = *side.records;

    std::vector<std::uint16_t> conv(records.conv.numel(), 0);
    std::vector<std::uint16_t> key(records.key.numel(), 0);
    std::vector<std::uint16_t> value(records.value.numel(), 0);
    std::vector<std::uint32_t> gate(records.gate.numel(), 0);
    for (std::int32_t layer = 0; layer < kFoldLayers; ++layer) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::int64_t outer = static_cast<std::int64_t>(layer) * kRecordCapacity + row;
            for (std::int32_t token = 0; token < commits[static_cast<std::size_t>(row)]; ++token) {
                const std::int64_t column = outer * kFoldWidth + token;
                for (std::int32_t c = 0; c < conv_channels; ++c) {
                    const std::int32_t global = conv_channels == kFoldConvChannels
                                                    ? c
                                                    : global_conv_channel(c, rank);
                    conv[static_cast<std::size_t>(column) * conv_channels + c] = bf16_pattern(
                        seed + 11U * static_cast<std::uint32_t>(layer) +
                            131U * static_cast<std::uint32_t>(row) +
                            977U * static_cast<std::uint32_t>(token) +
                            7919U * static_cast<std::uint32_t>(global),
                        0.05F);
                }
                for (std::int32_t head = 0; head < qk_heads; ++head) {
                    const std::int32_t global =
                        qk_heads == kFoldQkHeads ? head : rank * kShardQkHeads + head;
                    const auto base =
                        static_cast<std::size_t>((column * qk_heads + head) * kStateDim);
                    for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                        key[base + static_cast<std::size_t>(dim)] = bf16_pattern(
                            seed + 100003U + 197U * static_cast<std::uint32_t>(layer) +
                                23U * static_cast<std::uint32_t>(row) +
                                11U * static_cast<std::uint32_t>(token) +
                                613U * static_cast<std::uint32_t>(global) +
                                static_cast<std::uint32_t>(dim),
                            0.08F);
                    }
                }
                for (std::int32_t head = 0; head < value_heads; ++head) {
                    const std::int32_t global =
                        value_heads == kFoldValueHeads ? head : rank * kShardValueHeads + head;
                    const auto base =
                        static_cast<std::size_t>((column * value_heads + head) * kStateDim);
                    for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                        value[base + static_cast<std::size_t>(dim)] = bf16_pattern(
                            seed + 200003U + 211U * static_cast<std::uint32_t>(layer) +
                                29U * static_cast<std::uint32_t>(row) +
                                13U * static_cast<std::uint32_t>(token) +
                                829U * static_cast<std::uint32_t>(global) +
                                static_cast<std::uint32_t>(dim),
                            0.08F);
                    }
                    const auto gate_base =
                        static_cast<std::size_t>((column * value_heads + head) * 2);
                    const std::uint32_t gate_key =
                        seed + 31U * static_cast<std::uint32_t>(layer) +
                        17U * static_cast<std::uint32_t>(row) +
                        7U * static_cast<std::uint32_t>(token) +
                        409U * static_cast<std::uint32_t>(global);
                    const float g    = -0.03F - static_cast<float>(mix32(gate_key) % 900U) / 1000.0F;
                    const float beta = 0.05F +
                                       static_cast<float>(mix32(gate_key + 300007U) % 900U) / 1000.0F;
                    gate[gate_base]     = std::bit_cast<std::uint32_t>(g);
                    gate[gate_base + 1] = std::bit_cast<std::uint32_t>(beta);
                }
            }
        }
    }
    cuda_check(cudaMemcpy(records.conv.data, conv.data(), records.conv.bytes(),
                          cudaMemcpyHostToDevice),
               "upload conv records");
    cuda_check(cudaMemcpy(records.key.data, key.data(), records.key.bytes(),
                          cudaMemcpyHostToDevice),
               "upload key records");
    cuda_check(cudaMemcpy(records.value.data, value.data(), records.value.bytes(),
                          cudaMemcpyHostToDevice),
               "upload value records");
    cuda_check(cudaMemcpy(records.gate.data, gate.data(), records.gate.bytes(),
                          cudaMemcpyHostToDevice),
               "upload gate records");

    LayoutBuilder state_builder;
    const LinearAttentionStatePoolLayout state_layout = plan_linear_attention_state_pool(
        state_builder, {.layers         = static_cast<std::uint32_t>(kFoldLayers),
                        .conv_channels  = conv_channels,
                        .conv_width     = 3,
                        .value_heads    = value_heads,
                        .value_head_dim = kStateDim,
                        .key_head_dim   = kStateDim,
                        .slot_count     = kFoldSlots,
                        .conv_dtype     = DType::BF16});
    const std::size_t state_bytes = state_builder.finish(256);
    side.state_storage.emplace(state_bytes);
    cuda_check(cudaMemset(side.state_storage->p, 0, state_bytes), "clear state pool");
    side.pool.emplace(DeviceSpan{side.state_storage->p, state_bytes}, state_layout);

    const std::size_t recurrent_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * value_heads;
    const std::size_t conv_elements = static_cast<std::size_t>(conv_channels) * 3;
    std::vector<float> recurrent(recurrent_elements);
    std::vector<std::uint16_t> conv_state(conv_elements);
    for (std::int32_t layer = 0; layer < kFoldLayers; ++layer) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::int32_t slot = slots[static_cast<std::size_t>(row)];
            for (std::int32_t head = 0; head < value_heads; ++head) {
                const std::int32_t global =
                    value_heads == kFoldValueHeads ? head : rank * kShardValueHeads + head;
                for (std::int32_t element = 0; element < kStateDim * kStateDim; ++element) {
                    recurrent[static_cast<std::size_t>(head) * kStateDim * kStateDim +
                              static_cast<std::size_t>(element)] =
                        signed_pattern(seed + 500009U + 227U * static_cast<std::uint32_t>(layer) +
                                           43U * static_cast<std::uint32_t>(row) +
                                           1013U * static_cast<std::uint32_t>(global) +
                                           static_cast<std::uint32_t>(element),
                                       0.01F);
                }
            }
            for (std::int32_t history = 0; history < 3; ++history) {
                for (std::int32_t c = 0; c < conv_channels; ++c) {
                    const std::int32_t global =
                        conv_channels == kFoldConvChannels ? c : global_conv_channel(c, rank);
                    conv_state[static_cast<std::size_t>(history) * conv_channels +
                               static_cast<std::size_t>(c)] =
                        bf16_pattern(seed + 400009U + 223U * static_cast<std::uint32_t>(layer) +
                                         41U * static_cast<std::uint32_t>(row) +
                                         13U * static_cast<std::uint32_t>(history) +
                                         6151U * static_cast<std::uint32_t>(global),
                                     0.05F);
                }
            }
            const Tensor recurrent_slot =
                side.pool->recurrent_slot(static_cast<std::uint32_t>(layer), slot);
            const Tensor conv_slot = side.pool->conv_slot(static_cast<std::uint32_t>(layer), slot);
            cuda_check(cudaMemcpy(recurrent_slot.data, recurrent.data(),
                                  recurrent.size() * sizeof(float), cudaMemcpyHostToDevice),
                       "upload initial recurrent state");
            cuda_check(cudaMemcpy(conv_slot.data, conv_state.data(),
                                  conv_state.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "upload initial conv state");
        }
    }
}

int run_leg_b(const ExecutionContext& ec) {
    std::cout << "\n--- Leg B: gdn_replay_fold FoldGeometry<48,8,24,5120> (contract 1) ---\n";
    int failures = 0;

    const std::vector<std::int32_t> slots{2, 0};
    const std::vector<std::int32_t> commits{4, 3};
    const auto rows = static_cast<std::int32_t>(slots.size());
    std::vector<ops::GdnReplayFoldRow> fold_rows(static_cast<std::size_t>(rows));
    for (std::int32_t row = 0; row < rows; ++row) {
        fold_rows[static_cast<std::size_t>(row)] = {slots[static_cast<std::size_t>(row)],
                                                    commits[static_cast<std::size_t>(row)]};
    }

    // A shard extent is legal only where REGISTERED: an unregistered neighbour of the tp2 geometry
    // must still be rejected, and rejected for that reason rather than for a structural one
    // (4096 % 128 == 0, so it clears FoldGeometry's own static_assert and is refused purely because
    // no arm claims it). `validate_fold_records` checks the geometry before it reads any tensor, so
    // this probe needs correctly-sized allocations, not meaningful contents.
    {
        set_device(ec, 0);
        constexpr std::int32_t kUnregisteredChannels = 4096;
        const GdnReplayRecordSpec spec{
            .layers          = kFoldLayers,
            .record_capacity = 1,
            .width           = 2,
            .conv_channels   = kUnregisteredChannels,
            .qk_heads        = kShardQkHeads,
            .value_heads     = kShardValueHeads,
            .key_dim         = kStateDim,
            .value_dim       = kStateDim,
        };
        LayoutBuilder record_builder;
        const GdnReplayRecordLayout record_layout = plan_gdn_replay_records(record_builder, spec);
        const std::size_t record_bytes            = record_builder.finish(256);
        DeviceBuffer record_storage(record_bytes);
        const GdnReplayRecords probe_records(DeviceSpan{record_storage.p, record_bytes},
                                             record_layout);
        LayoutBuilder state_builder;
        const LinearAttentionStatePoolLayout state_layout = plan_linear_attention_state_pool(
            state_builder, {.layers         = static_cast<std::uint32_t>(kFoldLayers),
                            .conv_channels  = kUnregisteredChannels,
                            .conv_width     = 3,
                            .value_heads    = kShardValueHeads,
                            .value_head_dim = kStateDim,
                            .key_head_dim   = kStateDim,
                            .slot_count     = 1,
                            .conv_dtype     = DType::BF16});
        const std::size_t state_bytes = state_builder.finish(256);
        DeviceBuffer state_storage(state_bytes);
        const LinearAttentionStatePool probe_pool(DeviceSpan{state_storage.p, state_bytes},
                                                  state_layout);
        const std::array<ops::GdnReplayFoldRow, 1> probe_rows{ops::GdnReplayFoldRow{0, 2}};
        bool rejected = false;
        try {
            ops::gdn_replay_fold(probe_records, probe_pool.all_layers_view(), probe_rows,
                                 ec.dev[0]->stream);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) {
            std::cerr << "  fold: the unregistered 48|8|24|4096 geometry was ACCEPTED but must be "
                         "rejected\n";
            ++failures;
        }
        std::cout << (rejected ? "OK" : "FAIL")
                  << " fold: the unregistered 48|8|24|4096 neighbour is still rejected\n";
    }

    FoldSide tp1;
    std::array<FoldSide, 2> shard;
    set_device(ec, 0);
    fill_fold_side(tp1, kFoldQkHeads, kFoldValueHeads, kFoldConvChannels, 0, 7001u, slots, commits);
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        fill_fold_side(shard[static_cast<std::size_t>(rank)], kShardQkHeads, kShardValueHeads,
                       kShardConvChannels, rank, 7001u, slots, commits);
    }

    set_device(ec, 0);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    ops::gdn_replay_fold(*tp1.records, tp1.pool->all_layers_view(), fold_rows,
                         ec.dev[0]->stream);
    cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ops::gdn_replay_fold(*shard[slot].records, shard[slot].pool->all_layers_view(), fold_rows,
                             ec.dev[rank]->stream);
        cuda_check(cudaStreamSynchronize(ec.dev[rank]->stream), "cudaStreamSynchronize");
    }

    // Per-head / per-channel BITWISE comparison, streamed layer by layer so no side needs the whole
    // 453 MiB pool on the host at once.
    const std::size_t parent_head_elements = static_cast<std::size_t>(kStateDim) * kStateDim;
    std::size_t recurrent_mismatches = 0;
    std::size_t conv_mismatches      = 0;
    std::size_t distinct_head_pairs  = 0;
    for (std::int32_t layer = 0; layer < kFoldLayers && recurrent_mismatches == 0 &&
                                 conv_mismatches == 0;
         ++layer) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::int32_t slot = slots[static_cast<std::size_t>(row)];
            set_device(ec, 0);
            const Tensor parent_recurrent =
                tp1.pool->recurrent_slot(static_cast<std::uint32_t>(layer), slot);
            const auto parent_values = from_device<float>(
                parent_recurrent.data, parent_head_elements * kFoldValueHeads);
            const Tensor parent_conv =
                tp1.pool->conv_slot(static_cast<std::uint32_t>(layer), slot);
            const auto parent_conv_values = from_device<std::uint16_t>(
                parent_conv.data, static_cast<std::size_t>(kFoldConvChannels) * 3);
            // The reference's own heads must differ, or a per-head comparison proves nothing.
            if (layer == 0 && row == 0) {
                for (std::int32_t head = 1; head < kFoldValueHeads; ++head) {
                    const bool same = std::equal(
                        parent_values.begin(),
                        parent_values.begin() + static_cast<std::ptrdiff_t>(parent_head_elements),
                        parent_values.begin() +
                            static_cast<std::ptrdiff_t>(parent_head_elements * head));
                    if (!same) { ++distinct_head_pairs; }
                }
            }
            for (int rank = 0; rank < 2; ++rank) {
                const auto rank_slot = static_cast<std::size_t>(rank);
                set_device(ec, rank);
                const Tensor shard_recurrent =
                    shard[rank_slot].pool->recurrent_slot(static_cast<std::uint32_t>(layer), slot);
                const auto shard_values = from_device<float>(
                    shard_recurrent.data, parent_head_elements * kShardValueHeads);
                for (std::int32_t head = 0; head < kShardValueHeads; ++head) {
                    const std::int32_t global = rank * kShardValueHeads + head;
                    for (std::size_t element = 0; element < parent_head_elements; ++element) {
                        const float got =
                            shard_values[static_cast<std::size_t>(head) * parent_head_elements +
                                         element];
                        const float expected =
                            parent_values[static_cast<std::size_t>(global) * parent_head_elements +
                                          element];
                        if (std::bit_cast<std::uint32_t>(got) !=
                            std::bit_cast<std::uint32_t>(expected)) {
                            if (recurrent_mismatches == 0) {
                                std::cerr << "  fold recurrent state: rank " << rank
                                          << " layer " << layer << " row " << row
                                          << " local head " << head << " (global " << global
                                          << ") element " << element << " differs: got " << got
                                          << " expected " << expected << '\n';
                            }
                            ++recurrent_mismatches;
                        }
                    }
                }
                const Tensor shard_conv =
                    shard[rank_slot].pool->conv_slot(static_cast<std::uint32_t>(layer), slot);
                const auto shard_conv_values = from_device<std::uint16_t>(
                    shard_conv.data, static_cast<std::size_t>(kShardConvChannels) * 3);
                for (std::int32_t history = 0; history < 3; ++history) {
                    for (std::int32_t c = 0; c < kShardConvChannels; ++c) {
                        const std::int32_t global = global_conv_channel(c, rank);
                        const std::uint16_t got =
                            shard_conv_values[static_cast<std::size_t>(history) *
                                                  kShardConvChannels +
                                              static_cast<std::size_t>(c)];
                        const std::uint16_t expected =
                            parent_conv_values[static_cast<std::size_t>(history) *
                                                   kFoldConvChannels +
                                               static_cast<std::size_t>(global)];
                        if (got != expected) {
                            if (conv_mismatches == 0) {
                                std::cerr << "  fold conv history: rank " << rank << " layer "
                                          << layer << " row " << row << " history " << history
                                          << " local channel " << c << " (global " << global
                                          << ") differs\n";
                            }
                            ++conv_mismatches;
                        }
                    }
                }
            }
        }
    }

    if (distinct_head_pairs + 1 != static_cast<std::size_t>(kFoldValueHeads)) {
        std::cerr << "  fold: the tp1 reference's own value-head states are not pairwise distinct ("
                  << distinct_head_pairs << " of " << kFoldValueHeads - 1
                  << " differ from head 0), so the per-head comparison would be vacuous\n";
        ++failures;
    }
    if (recurrent_mismatches != 0) {
        std::cerr << "  fold: " << recurrent_mismatches
                  << " recurrent-state words differ from the tp1 geometry\n";
        ++failures;
    }
    if (conv_mismatches != 0) {
        std::cerr << "  fold: " << conv_mismatches
                  << " conv-history words differ from the tp1 geometry\n";
        ++failures;
    }
    std::cout << "  fold FoldGeometry<48,8,24,5120>: " << kFoldLayers << " layers x " << rows
              << " rows, per-head recurrent + per-channel conv history "
              << (failures == 0 ? "BITWISE identical to the tp1 48|16|48|10240 geometry"
                                : "MISMATCHED")
              << '\n';
    return failures;
}

} // namespace

// =================================================================================================
// LEG C -- CONTRACT 2: gdn_input_proj_conv_snapshot / conv_record over the 5120-channel shard.
//
// These are the fused projection+conv1d+SiLU forms reached only from Phase::Verify (the MTP
// path), so their split belongs with the MTP round rather than with the plain GDN projections.
// The composition is mechanically the GDN input projection's [8192,5120] column shard feeding the
// three-block 5120-channel convolution shard.
// =================================================================================================
namespace {

constexpr std::int32_t kGdnParentRows  = 16384;
constexpr std::int32_t kGdnShardRows   = 8192;
constexpr std::int32_t kGdnQueryRows   = 2048;
constexpr std::int32_t kGdnValueRows   = 6144;
constexpr std::int32_t kGdnZRows       = 6144;

// Rank r's [8192,5120] projection shard: its own half of Q | K | V | Z in the parent's order.
SplitWeight make_gdn_input_split(QType qtype, std::uint32_t seed) {
    SplitWeight out;
    out.parent = make_block(qtype, kGdnParentRows, kHidden, seed, 0);
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot                = static_cast<std::size_t>(rank);
        const std::int32_t q_origin    = rank * 1024;
        const std::int32_t k_origin    = kGdnQueryRows + rank * 1024;
        const std::int32_t v_origin    = 2 * kGdnQueryRows + rank * 3072;
        const std::int32_t z_origin    = 2 * kGdnQueryRows + kGdnValueRows + rank * 3072;
        qw::PackedWeight q = make_block(qtype, 1024, kHidden, seed, q_origin);
        qw::PackedWeight k = make_block(qtype, 1024, kHidden, seed, k_origin);
        qw::PackedWeight v = make_block(qtype, 3072, kHidden, seed, v_origin);
        qw::PackedWeight z = make_block(qtype, 3072, kHidden, seed, z_origin);
        out.shard[slot] = concat_row_blocks(concat_row_blocks(concat_row_blocks(q, k), v), z);
        out.sections[slot].push_back({"q", q_origin, std::move(q)});
        out.sections[slot].push_back({"k", k_origin, std::move(k)});
        out.sections[slot].push_back({"v", v_origin, std::move(v)});
        out.sections[slot].push_back({"z", z_origin, std::move(z)});
    }
    return out;
}

// The Q4/Q5 split-storage two-weight form of the same object: `gdn/query_key` Q4G64_F16S
// [4096,5120] = Q | K and `gdn/value_z` Q5G64_F16S [12288,5120] = V | Z, the binding
// `bind_groupwise_text_layers` uses. The ShardPlan splits each of the two parents by its own
// sections' head counts (`ends("gdn/query_key")` / `ends("gdn/value_z")` in bindings.cpp), so rank
// r's shards are [2048,5120] = Q-half | K-half and [6144,5120] = V-half | Z-half.
SplitWeight make_gdn_query_key_split(std::uint32_t seed) {
    SplitWeight out;
    out.parent = make_block(QType::Q4G64_F16S, 2 * kGdnQueryRows, kHidden, seed, 0);
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot             = static_cast<std::size_t>(rank);
        const std::int32_t q_origin = rank * 1024;
        const std::int32_t k_origin = kGdnQueryRows + rank * 1024;
        qw::PackedWeight q = make_block(QType::Q4G64_F16S, 1024, kHidden, seed, q_origin);
        qw::PackedWeight k = make_block(QType::Q4G64_F16S, 1024, kHidden, seed, k_origin);
        out.shard[slot]    = concat_row_blocks(q, k);
        out.sections[slot].push_back({"q", q_origin, std::move(q)});
        out.sections[slot].push_back({"k", k_origin, std::move(k)});
    }
    return out;
}

SplitWeight make_gdn_value_z_split(std::uint32_t seed) {
    SplitWeight out;
    out.parent = make_block(QType::Q5G64_F16S, kGdnValueRows + kGdnZRows, kHidden, seed, 0);
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot             = static_cast<std::size_t>(rank);
        const std::int32_t v_origin = rank * 3072;
        const std::int32_t z_origin = kGdnValueRows + rank * 3072;
        qw::PackedWeight v = make_block(QType::Q5G64_F16S, 3072, kHidden, seed, v_origin);
        qw::PackedWeight z = make_block(QType::Q5G64_F16S, 3072, kHidden, seed, z_origin);
        out.shard[slot]    = concat_row_blocks(v, z);
        out.sections[slot].push_back({"v", v_origin, std::move(v)});
        out.sections[slot].push_back({"z", z_origin, std::move(z)});
    }
    return out;
}

struct ConvCase {
    const char* label;
    // Q4G64_F16S selects the split-storage TWO-WEIGHT form (Q4 query_key + Q5 value_z); every
    // other qtype selects the fused single-parent form.
    QType qtype;
    ops::LinearPolicy policy;
    std::int32_t batch;
    std::int32_t width;
    bool mixed_widths;

    [[nodiscard]] bool split_storage() const { return qtype == QType::Q4G64_F16S; }
};

// Compares one shard output plane against the tp1 plane it is a channel slice of.
int compare_channel_block(const std::string& label, const std::vector<double>& shard_values,
                          std::int32_t shard_rows, const std::vector<double>& parent_values,
                          std::int32_t parent_rows, std::int32_t parent_row_begin,
                          std::int32_t columns) {
    std::vector<double> expected(static_cast<std::size_t>(shard_rows) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t source = static_cast<std::size_t>(column) * parent_rows +
                                   static_cast<std::size_t>(parent_row_begin);
        std::copy(parent_values.begin() + static_cast<std::ptrdiff_t>(source),
                  parent_values.begin() + static_cast<std::ptrdiff_t>(source + shard_rows),
                  expected.begin() + static_cast<std::ptrdiff_t>(column) * shard_rows);
    }
    return compare(label, shard_values, expected, kSplitCriterion);
}

int run_conv_case(const ExecutionContext& ec, const ConvCase& test_case) {
    const std::int32_t width   = test_case.width;
    const std::int32_t batch   = test_case.batch;
    const std::int32_t columns = width * batch;
    const std::string head =
        std::string(test_case.label) + " B=" + std::to_string(batch) + " W=" + std::to_string(width);
    std::cout << head << '\n';
    int failures = 0;

    const bool split_storage = test_case.split_storage();
    // Exactly one of these is used; the unused one is a 0-row placeholder that is never uploaded.
    const SplitWeight projection =
        split_storage ? SplitWeight{} : make_gdn_input_split(test_case.qtype, 8100u);
    const SplitWeight query_key = split_storage ? make_gdn_query_key_split(8110u) : SplitWeight{};
    const SplitWeight value_z   = split_storage ? make_gdn_value_z_split(8120u) : SplitWeight{};
    if (split_storage) {
        failures += verify_sectioned_column_shard(std::string(test_case.label) + " query_key",
                                                  query_key);
        failures += verify_sectioned_column_shard(std::string(test_case.label) + " value_z",
                                                  value_z);
    } else {
        failures += verify_sectioned_column_shard(std::string(test_case.label) + " projection",
                                                  projection);
    }
    if (failures != 0) { return failures; }

    // conv weight [10240,4] and its 5120-channel three-block shard.
    const std::vector<float> conv_weight_host =
        random_bf16(static_cast<std::size_t>(kGdnChannels) * 4, 8200u, -0.6F, 0.6F);
    std::array<std::vector<float>, 2> conv_weight_shard;
    for (int rank = 0; rank < 2; ++rank) {
        std::vector<float>& out = conv_weight_shard[static_cast<std::size_t>(rank)];
        out.resize(static_cast<std::size_t>(kShardConvChannels) * 4);
        for (std::int32_t tap = 0; tap < 4; ++tap) {
            for (std::int32_t c = 0; c < kShardConvChannels; ++c) {
                out[static_cast<std::size_t>(tap) * kShardConvChannels +
                    static_cast<std::size_t>(c)] =
                    conv_weight_host[static_cast<std::size_t>(tap) * kGdnChannels +
                                     static_cast<std::size_t>(global_conv_channel(c, rank))];
            }
        }
    }

    // conv state pool: one [base, base+W) destination interval per row plus one initial slot each.
    const std::int32_t slots = columns + batch;
    const std::vector<float> conv_state_host = random_bf16(
        static_cast<std::size_t>(kGdnChannels) * 3 * slots, 8300u, -0.4F, 0.4F);
    std::array<std::vector<float>, 2> conv_state_shard;
    for (int rank = 0; rank < 2; ++rank) {
        std::vector<float>& out = conv_state_shard[static_cast<std::size_t>(rank)];
        out.resize(static_cast<std::size_t>(kShardConvChannels) * 3 * slots);
        for (std::int32_t slot = 0; slot < slots; ++slot) {
            for (std::int32_t history = 0; history < 3; ++history) {
                for (std::int32_t c = 0; c < kShardConvChannels; ++c) {
                    out[(static_cast<std::size_t>(slot) * 3 + history) * kShardConvChannels +
                        static_cast<std::size_t>(c)] =
                        conv_state_host[(static_cast<std::size_t>(slot) * 3 + history) *
                                            kGdnChannels +
                                        static_cast<std::size_t>(global_conv_channel(c, rank))];
                }
            }
        }
    }

    std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> base_slots(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> valid(static_cast<std::size_t>(batch), width);
    for (std::int32_t row = 0; row < batch; ++row) {
        initial_slots[static_cast<std::size_t>(row)] = columns + row;
        base_slots[static_cast<std::size_t>(row)]    = row * width;
        if (test_case.mixed_widths) {
            valid[static_cast<std::size_t>(row)] = std::max(1, width - row);
        }
    }

    const std::vector<float> x_host =
        random_bf16(static_cast<std::size_t>(kHidden) * columns, 8400u, -1.0F, 1.0F);

    const std::size_t tp1_ws =
        split_storage ? ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            kGdnQueryRows, kGdnQueryRows, kGdnValueRows, batch, width, width)
                      : ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            test_case.qtype, kGdnParentRows, kHidden, test_case.policy, batch,
                            width, width);
    const std::size_t shard_ws =
        ops::gdn_input_proj_conv_snapshot_column_parallel_workspace_capacity_bytes(
            test_case.qtype, test_case.policy, batch, width, width);
    const std::size_t tp1_record_ws =
        split_storage ? ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            kGdnQueryRows, kGdnQueryRows, kGdnValueRows, batch, width, width)
                      : ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            test_case.qtype, kGdnParentRows, kHidden, test_case.policy, batch,
                            width, width);
    const std::size_t shard_record_ws =
        ops::gdn_input_proj_conv_record_column_parallel_workspace_capacity_bytes(
            test_case.qtype, test_case.policy, batch, width, width);

    // ---- (a) tp1 reference on device 0 ------------------------------------------------------
    std::vector<double> ref_query, ref_key, ref_value, ref_z, ref_state, ref_record,
        ref_record_query, ref_record_value;
    {
        set_device(ec, 0);
        cudaStream_t s = ec.dev[0]->stream;
        DeviceWeight w = upload_weight(split_storage ? query_key.parent : projection.parent);
        DeviceWeight vz;
        if (split_storage) { vz = upload_weight(value_z.parent); }
        DeviceBuffer conv_w   = to_device_bf16(conv_weight_host);
        DeviceBuffer states   = to_device_bf16(conv_state_host);
        DeviceBuffer x        = to_device_bf16(x_host);
        DeviceBuffer initial  = to_device_i32(initial_slots);
        DeviceBuffer base     = to_device_i32(base_slots);
        DeviceBuffer valid_d  = to_device_i32(valid);
        DeviceBuffer query(static_cast<std::size_t>(kGdnQueryRows) * columns *
                           sizeof(std::uint16_t));
        DeviceBuffer key(static_cast<std::size_t>(kGdnQueryRows) * columns * sizeof(std::uint16_t));
        DeviceBuffer value(static_cast<std::size_t>(kGdnValueRows) * columns *
                           sizeof(std::uint16_t));
        DeviceBuffer z(static_cast<std::size_t>(kGdnZRows) * columns * sizeof(std::uint16_t));
        DeviceBuffer record(static_cast<std::size_t>(kGdnChannels) * columns *
                            sizeof(std::uint16_t));
        DeviceBuffer ws(std::max<std::size_t>(std::max(tp1_ws, tp1_record_ws), 256));
        WorkspaceArena arena(DeviceSpan{ws.p, ws.bytes});

        const Tensor t_x(x.p, DType::BF16, {kHidden, width, batch});
        const Tensor t_conv_w(conv_w.p, DType::BF16, {kGdnChannels, 4});
        Tensor t_states(states.p, DType::BF16, {kGdnChannels, 3, slots});
        const Tensor t_initial(initial.p, DType::I32, {batch});
        const Tensor t_base(base.p, DType::I32, {batch});
        const Tensor t_valid = test_case.mixed_widths ? Tensor(valid_d.p, DType::I32, {batch})
                                                      : Tensor{};
        Tensor t_query(query.p, DType::BF16, {kGdnQueryRows, width, batch});
        Tensor t_key(key.p, DType::BF16, {kGdnQueryRows, width, batch});
        Tensor t_value(value.p, DType::BF16, {kGdnValueRows, width, batch});
        Tensor t_z(z.p, DType::BF16, {kGdnZRows, width, batch});
        Tensor t_record(record.p, DType::BF16, {kGdnChannels, width, batch});

        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        if (split_storage) {
            ops::gdn_input_proj_conv_snapshot(t_x, w.weight, vz.weight, t_conv_w, t_states, t_valid,
                                              t_initial, t_base, t_query, t_key, t_value, t_z,
                                              arena, s);
        } else {
            ops::gdn_input_proj_conv_snapshot(t_x, w.weight, t_conv_w, t_states, t_valid, t_initial,
                                              t_base, t_query, t_key, t_value, t_z,
                                              test_case.policy, arena, s);
        }
        cuda_check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        ref_query = read_bf16(query, static_cast<std::size_t>(kGdnQueryRows) * columns);
        ref_key   = read_bf16(key, static_cast<std::size_t>(kGdnQueryRows) * columns);
        ref_value = read_bf16(value, static_cast<std::size_t>(kGdnValueRows) * columns);
        ref_z     = read_bf16(z, static_cast<std::size_t>(kGdnZRows) * columns);
        ref_state = read_bf16(states, static_cast<std::size_t>(kGdnChannels) * 3 * slots);

        // conv_record uses the same operands but a read-only state pool, so it is re-run from the
        // pristine state upload.
        states = to_device_bf16(conv_state_host);
        Tensor t_states_ro(states.p, DType::BF16, {kGdnChannels, 3, slots});
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        if (split_storage) {
            ops::gdn_input_proj_conv_record(t_x, w.weight, vz.weight, t_conv_w, t_states_ro,
                                            t_valid, t_initial, t_record, t_query, t_key, t_value,
                                            t_z, arena, s);
        } else {
            ops::gdn_input_proj_conv_record(t_x, w.weight, t_conv_w, t_states_ro, t_valid,
                                            t_initial, t_record, t_query, t_key, t_value, t_z,
                                            test_case.policy, arena, s);
        }
        cuda_check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        ref_record       = read_bf16(record, static_cast<std::size_t>(kGdnChannels) * columns);
        ref_record_query = read_bf16(query, static_cast<std::size_t>(kGdnQueryRows) * columns);
        ref_record_value = read_bf16(value, static_cast<std::size_t>(kGdnValueRows) * columns);
    }

    // ---- (b) the split forms ----------------------------------------------------------------
    std::array<std::optional<DeviceWeight>, 2> shard_w;
    std::array<std::optional<DeviceWeight>, 2> shard_vz;
    std::array<std::optional<DeviceBuffer>, 2> conv_w_d, states_d, x_d, initial_d, base_d, valid_d,
        query_d, key_d, value_d, z_d, record_d, ws_d;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        shard_w[slot].emplace(
            upload_weight(split_storage ? query_key.shard[slot] : projection.shard[slot]));
        if (split_storage) { shard_vz[slot].emplace(upload_weight(value_z.shard[slot])); }
        conv_w_d[slot].emplace(to_device_bf16(conv_weight_shard[slot]));
        states_d[slot].emplace(to_device_bf16(conv_state_shard[slot]));
        x_d[slot].emplace(to_device_bf16(x_host));
        initial_d[slot].emplace(to_device_i32(initial_slots));
        base_d[slot].emplace(to_device_i32(base_slots));
        valid_d[slot].emplace(to_device_i32(valid));
        query_d[slot].emplace(static_cast<std::size_t>(1024) * columns * sizeof(std::uint16_t));
        key_d[slot].emplace(static_cast<std::size_t>(1024) * columns * sizeof(std::uint16_t));
        value_d[slot].emplace(static_cast<std::size_t>(3072) * columns * sizeof(std::uint16_t));
        z_d[slot].emplace(static_cast<std::size_t>(3072) * columns * sizeof(std::uint16_t));
        record_d[slot].emplace(static_cast<std::size_t>(kShardConvChannels) * columns *
                               sizeof(std::uint16_t));
        ws_d[slot].emplace(std::max<std::size_t>(std::max(shard_ws, shard_record_ws), 256));
    }

    std::array<std::optional<WorkspaceArena>, 2> arena;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        arena[slot].emplace(DeviceSpan{ws_d[slot]->p, ws_d[slot]->bytes});
    }

    const auto make_pair = [](std::array<std::optional<DeviceBuffer>, 2>& buffers, DType dtype,
                              std::initializer_list<std::int32_t> shape) {
        return std::array<Tensor, 2>{Tensor(buffers[0]->p, dtype, shape),
                                     Tensor(buffers[1]->p, dtype, shape)};
    };

    const std::array<Weight, 2> weights{shard_w[0]->weight, shard_w[1]->weight};
    const std::array<Weight, 2> value_z_weights{
        split_storage ? shard_vz[0]->weight : Weight{},
        split_storage ? shard_vz[1]->weight : Weight{}};
    const std::array<Tensor, 2> t_x = make_pair(x_d, DType::BF16, {kHidden, width, batch});
    const std::array<Tensor, 2> t_conv_w =
        make_pair(conv_w_d, DType::BF16, {kShardConvChannels, 4});
    const std::array<Tensor, 2> t_states =
        make_pair(states_d, DType::BF16, {kShardConvChannels, 3, slots});
    const std::array<Tensor, 2> t_initial = make_pair(initial_d, DType::I32, {batch});
    const std::array<Tensor, 2> t_base    = make_pair(base_d, DType::I32, {batch});
    const std::array<Tensor, 2> t_valid =
        test_case.mixed_widths ? make_pair(valid_d, DType::I32, {batch})
                               : std::array<Tensor, 2>{Tensor{}, Tensor{}};
    const std::array<Tensor, 2> t_query = make_pair(query_d, DType::BF16, {1024, width, batch});
    const std::array<Tensor, 2> t_key   = make_pair(key_d, DType::BF16, {1024, width, batch});
    const std::array<Tensor, 2> t_value = make_pair(value_d, DType::BF16, {3072, width, batch});
    const std::array<Tensor, 2> t_z     = make_pair(z_d, DType::BF16, {3072, width, batch});
    const std::array<Tensor, 2> t_record =
        make_pair(record_d, DType::BF16, {kShardConvChannels, width, batch});
    const std::array<WorkspaceArena*, 2> workspace{&*arena[0], &*arena[1]};

    retire_staging(ec);
    if (split_storage) {
        ops::gdn_input_proj_conv_snapshot_column_parallel(
            t_x, weights, value_z_weights, t_conv_w, t_states, t_valid, t_initial, t_base, t_query,
            t_key, t_value, t_z, workspace, ec);
    } else {
        ops::gdn_input_proj_conv_snapshot_column_parallel(t_x, weights, t_conv_w, t_states, t_valid,
                                                          t_initial, t_base, t_query, t_key,
                                                          t_value, t_z, test_case.policy, workspace,
                                                          ec);
    }
    synchronize_both(ec);

    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        const std::string label = head + " snapshot rank " + std::to_string(rank);
        failures += compare_channel_block(
            label + " query", read_bf16(*query_d[slot], static_cast<std::size_t>(1024) * columns),
            1024, ref_query, kGdnQueryRows, rank * 1024, columns);
        failures += compare_channel_block(
            label + " key", read_bf16(*key_d[slot], static_cast<std::size_t>(1024) * columns), 1024,
            ref_key, kGdnQueryRows, rank * 1024, columns);
        failures += compare_channel_block(
            label + " value", read_bf16(*value_d[slot], static_cast<std::size_t>(3072) * columns),
            3072, ref_value, kGdnValueRows, rank * 3072, columns);
        failures += compare_channel_block(
            label + " z", read_bf16(*z_d[slot], static_cast<std::size_t>(3072) * columns), 3072,
            ref_z, kGdnZRows, rank * 3072, columns);

        // Published conv history, channel by channel through the three-block map.
        const auto observed_state =
            read_bf16(*states_d[slot], static_cast<std::size_t>(kShardConvChannels) * 3 * slots);
        std::vector<double> expected_state(observed_state.size());
        for (std::int32_t s = 0; s < slots; ++s) {
            for (std::int32_t history = 0; history < 3; ++history) {
                for (std::int32_t c = 0; c < kShardConvChannels; ++c) {
                    expected_state[(static_cast<std::size_t>(s) * 3 + history) *
                                       kShardConvChannels +
                                   static_cast<std::size_t>(c)] =
                        ref_state[(static_cast<std::size_t>(s) * 3 + history) * kGdnChannels +
                                  static_cast<std::size_t>(global_conv_channel(c, rank))];
                }
            }
        }
        failures += compare(label + " conv history", observed_state, expected_state,
                            kSplitCriterion);
    }

    retire_staging(ec);
    // Restore the pristine state pool: conv_record reads state but must not write it.
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        const DeviceBuffer fresh = to_device_bf16(conv_state_shard[slot]);
        cuda_check(cudaMemcpy(states_d[slot]->p, fresh.p, states_d[slot]->bytes,
                              cudaMemcpyDeviceToDevice),
                   "restore conv state");
    }
    retire_staging(ec);
    if (split_storage) {
        ops::gdn_input_proj_conv_record_column_parallel(t_x, weights, value_z_weights, t_conv_w,
                                                        t_states, t_valid, t_initial, t_record,
                                                        t_query, t_key, t_value, t_z, workspace,
                                                        ec);
    } else {
        ops::gdn_input_proj_conv_record_column_parallel(t_x, weights, t_conv_w, t_states, t_valid,
                                                        t_initial, t_record, t_query, t_key,
                                                        t_value, t_z, test_case.policy, workspace,
                                                        ec);
    }
    synchronize_both(ec);
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        const std::string label = head + " record rank " + std::to_string(rank);
        const auto observed =
            read_bf16(*record_d[slot], static_cast<std::size_t>(kShardConvChannels) * columns);
        std::vector<double> expected(observed.size());
        for (std::int32_t column = 0; column < columns; ++column) {
            const std::int32_t row = column / width;
            const std::int32_t token = column % width;
            const bool defined = token < valid[static_cast<std::size_t>(row)];
            for (std::int32_t c = 0; c < kShardConvChannels; ++c) {
                const std::size_t destination =
                    static_cast<std::size_t>(column) * kShardConvChannels +
                    static_cast<std::size_t>(c);
                // Only the valid prefix of conv_record is semantically defined (gdn_input_proj.h),
                // so the invalid tail is excluded from the comparison by copying the observed value.
                expected[destination] =
                    defined ? ref_record[static_cast<std::size_t>(column) * kGdnChannels +
                                         static_cast<std::size_t>(global_conv_channel(c, rank))]
                            : observed[destination];
            }
        }
        failures += compare(label + " conv_record", observed, expected, kSplitCriterion);
        failures += compare_channel_block(
            label + " query", read_bf16(*query_d[slot], static_cast<std::size_t>(1024) * columns),
            1024, ref_record_query, kGdnQueryRows, rank * 1024, columns);
        failures += compare_channel_block(
            label + " value", read_bf16(*value_d[slot], static_cast<std::size_t>(3072) * columns),
            3072, ref_record_value, kGdnValueRows, rank * 3072, columns);
    }

    // Provenance: the two ranks own disjoint channel sets, so their outputs must differ.
    set_device(ec, 0);
    const auto rank0_value = read_bf16(*value_d[0], static_cast<std::size_t>(3072) * columns);
    set_device(ec, 1);
    const auto rank1_value = read_bf16(*value_d[1], static_cast<std::size_t>(3072) * columns);
    failures += expect_cross_rank_difference(head + " conv value", rank0_value, rank1_value);
    return failures;
}

int run_leg_c(const ExecutionContext& ec) {
    std::cout << "\n--- Leg C: gdn_input_proj_conv_snapshot / conv_record channel split "
                 "(contract 2) ---\n";
    int failures = 0;
    const std::vector<ConvCase> cases{
        {"fp8 gdn conv", QType::FP8_E4M3FN_ROW_BF16S, ops::LinearPolicy::A16Only, 1, 4, false},
        {"fp8 gdn conv", QType::FP8_E4M3FN_ROW_BF16S, ops::LinearPolicy::AllowA8, 2, 3, true},
        {"nvfp4 gdn conv", QType::NVFP4, ops::LinearPolicy::A16Only, 1, 4, false},
        {"nvfp4 gdn conv", QType::NVFP4, ops::LinearPolicy::AllowA4, 2, 3, true},
        // Split-storage two-weight form (Q4 query_key + Q5 value_z), the groupwise profile's
        // binding for this object -- a real tp2 + groupwise + MTP route, so it gets the same GPU
        // parity treatment as the fused forms rather than registry coverage alone. A16 only.
        {"q4/q5 gdn conv", QType::Q4G64_F16S, ops::LinearPolicy::A16Only, 1, 4, false},
        {"q4/q5 gdn conv", QType::Q4G64_F16S, ops::LinearPolicy::A16Only, 2, 3, true},
    };
    for (const ConvCase& test_case : cases) { failures += run_conv_case(ec, test_case); }
    return failures;
}

} // namespace

// =================================================================================================
// LEG D -- the draft head: [131072,5120] Q4G64_F16S vocab-row-split + allgather + argmax + the
// replicated token-id remap.
//
// AXIS VOCABULARY, as in tests/ops/test_output_head_split.cpp: the ShardPlan calls this a "Rows"
// split because the vocabulary axis is the ARTIFACT's row axis; in `ops::linear`'s vocabulary that
// same axis is N, the GEMM's OUTPUT dimension, so the Op invoked is `linear_column_parallel`
// followed by `allgather_rows` -- NOT an all-reduce. Same split, two naming conventions.
//
// WHY `draft_head_token_ids` IS REPLICATED IN THE ShardPlan. The runtime consumes the
// map AFTER a global argmax: `TextContext::proposal_argmax` runs `ops::argmax` over the whole
// 131072-wide proposal logit vector and then `ops::proposal_remap_token_ids(tokens, id_map,
// n=131072, ...)`. The winning row index can land in EITHER device's half, so the sampling device
// needs the whole map, not its own 65536-entry slice. This leg proves the requirement is real
// rather than defensive: it asserts that the sweep actually produces at least one winner in the
// upper half, which a device-0-only half-map could not remap.
// =================================================================================================
namespace {

int run_leg_d(const ExecutionContext& ec, const ops::PeerEvents& events) {
    std::cout << "\n--- Leg D: draft head (Q4G64 vocab split + allgather + argmax + remap) ---\n";
    int failures = 0;

    const SplitWeight head = make_column_split(QType::Q4G64_F16S, kDraftRows, kHidden, 9100u);
    failures += verify_column_shard_identity("text/draft_head", head, kDraftShardRows);
    if (failures != 0) { return failures; }

    // The replicated I32 [131072] row-index -> tokenizer-id map. Distinct ids inside the tokenizer
    // domain, exactly what `validate_draft_ids` (bindings.cpp) requires of the real object.
    std::vector<std::int32_t> token_ids(static_cast<std::size_t>(kDraftRows));
    for (std::int32_t row = 0; row < kDraftRows; ++row) {
        token_ids[static_cast<std::size_t>(row)] =
            static_cast<std::int32_t>((static_cast<std::int64_t>(row) * 3 + 7) % kTokenizerVocab);
    }

    set_device(ec, 0);
    const DeviceWeight parent = upload_weight(head.parent);
    std::array<std::optional<DeviceWeight>, 2> shard;
    std::array<std::optional<DeviceBuffer>, 2> ids_d;
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        set_device(ec, rank);
        shard[slot].emplace(upload_weight(head.shard[slot]));
        ids_d[slot].emplace(to_device_i32(token_ids)); // replicated: an independent full copy
    }

    bool saw_upper_half_winner = false;
    for (const std::int32_t tokens : {1, 5, 17}) {
        const std::string label = "draft_head T=" + std::to_string(tokens);
        const std::vector<float> activation =
            random_bf16(static_cast<std::size_t>(kHidden) * tokens, 9200u + tokens, -1.0F, 1.0F);
        const std::size_t full_elements = static_cast<std::size_t>(kDraftRows) * tokens;
        const std::size_t half_elements = static_cast<std::size_t>(kDraftShardRows) * tokens;

        std::array<std::optional<DeviceBuffer>, 2> x_d;
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            x_d[static_cast<std::size_t>(rank)].emplace(to_device_bf16(activation));
        }

        // (a) tp1 reference.
        set_device(ec, 0);
        GuardedDeviceBuffer reference(full_elements * sizeof(std::uint16_t));
        reference.fill(0xff);
        const Tensor ref_x(x_d[0]->p, DType::BF16, {kHidden, tokens});
        Tensor ref_out(reference.data(), DType::BF16, {kDraftRows, tokens});
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ops::linear(ref_x, parent.weight, ref_out, ec.dev[0]->stream);
        cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
        failures += reference.verify_guards(label + " reference");
        const std::vector<double> ref_flat = from_device_bf16(reference.data(), full_elements);

        // (b) column-parallel halves.
        std::array<std::optional<GuardedDeviceBuffer>, 2> half;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            half[slot].emplace(half_elements * sizeof(std::uint16_t));
            half[slot]->fill(0xff);
        }
        const std::array<Tensor, 2> x{Tensor(x_d[0]->p, DType::BF16, {kHidden, tokens}),
                                      Tensor(x_d[1]->p, DType::BF16, {kHidden, tokens})};
        const std::array<Weight, 2> w{shard[0]->weight, shard[1]->weight};
        const std::array<Tensor, 2> out{
            Tensor(half[0]->data(), DType::BF16, {kDraftShardRows, tokens}),
            Tensor(half[1]->data(), DType::BF16, {kDraftShardRows, tokens})};
        retire_staging(ec);
        ops::linear_column_parallel(x, w, out, ec);
        synchronize_both(ec);

        std::array<std::vector<double>, 2> observed;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            failures += half[slot]->verify_guards(label + " rank " + std::to_string(rank));
            observed[slot] = from_device_bf16(half[slot]->data(), half_elements);
            failures += compare(label + " rank " + std::to_string(rank), observed[slot],
                                extract_rank_block(ref_flat, kDraftRows, rank * kDraftShardRows,
                                                   kDraftShardRows, tokens),
                                kSplitCriterion);
        }
        failures += expect_cross_rank_difference(label, observed[0], observed[1]);

        // (c) allgather_rows, one call per token (ops::linear's [N,T] layout stores the vocabulary
        // axis fastest, per token -- see tests/ops/test_output_head_split.cpp's header).
        std::array<std::optional<GuardedDeviceBuffer>, 2> full;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            full[slot].emplace(full_elements * sizeof(std::uint16_t));
            full[slot]->fill(0xcd);
        }
        retire_staging(ec);
        for (std::int32_t token = 0; token < tokens; ++token) {
            const std::array<Tensor, 2> part{
                Tensor(byte_offset(half[0]->data(), static_cast<std::size_t>(token) * kDraftShardRows,
                                   sizeof(std::uint16_t)),
                       DType::BF16, {1, kDraftShardRows}),
                Tensor(byte_offset(half[1]->data(), static_cast<std::size_t>(token) * kDraftShardRows,
                                   sizeof(std::uint16_t)),
                       DType::BF16, {1, kDraftShardRows})};
            const std::array<Tensor, 2> destination{
                Tensor(byte_offset(full[0]->data(), static_cast<std::size_t>(token) * kDraftRows,
                                   sizeof(std::uint16_t)),
                       DType::BF16, {1, kDraftRows}),
                Tensor(byte_offset(full[1]->data(), static_cast<std::size_t>(token) * kDraftRows,
                                   sizeof(std::uint16_t)),
                       DType::BF16, {1, kDraftRows})};
            ops::allgather_rows(destination, part, ec, events);
        }
        synchronize_both(ec);

        std::array<std::vector<double>, 2> reconstructed;
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            failures += full[slot]->verify_guards(label + " reconstructed " + std::to_string(rank));
            reconstructed[slot] = from_device_bf16(full[slot]->data(), full_elements);
            failures += compare(label + " reconstructed rank " + std::to_string(rank),
                                reconstructed[slot], ref_flat, kSplitCriterion);
        }
        failures += expect_cross_rank_equality(label + " reconstructed", reconstructed[0],
                                               reconstructed[1]);

        // (d) argmax + the replicated token-id remap, on the tp1 reference and on both devices.
        set_device(ec, 0);
        Tensor ref_logits(reference.data(), DType::BF16, {kDraftRows, tokens});
        GuardedDeviceBuffer ref_argmax(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
        Tensor ref_argmax_tensor(ref_argmax.data(), DType::I32, {tokens});
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        ops::argmax(ref_logits, ref_argmax_tensor, kDraftRows, ec.dev[0]->stream);
        cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
        const auto ref_rows =
            from_device<std::int32_t>(ref_argmax.data(), static_cast<std::size_t>(tokens));
        std::cout << "  " << label << " argmax rows:";
        for (const std::int32_t row : ref_rows) {
            std::cout << ' ' << row;
            if (row >= kDraftShardRows) { saw_upper_half_winner = true; }
        }
        std::cout << '\n';
        ops::proposal_remap_token_ids(ref_argmax_tensor,
                                      static_cast<const std::int32_t*>(ids_d[0]->p), kDraftRows,
                                      ec.dev[0]->stream);
        cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "cudaStreamSynchronize");
        const auto ref_tokens =
            from_device<std::int32_t>(ref_argmax.data(), static_cast<std::size_t>(tokens));

        std::vector<std::int32_t> expected_tokens(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) {
            expected_tokens[static_cast<std::size_t>(token)] =
                token_ids[static_cast<std::size_t>(ref_rows[static_cast<std::size_t>(token)])];
        }
        failures += verify_exact((label + " tp1 remap vs host oracle").c_str(), ref_tokens,
                                 expected_tokens);

        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            Tensor logits(full[slot]->data(), DType::BF16, {kDraftRows, tokens});
            GuardedDeviceBuffer rows(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
            Tensor rows_tensor(rows.data(), DType::I32, {tokens});
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            ops::argmax(logits, rows_tensor, kDraftRows, ec.dev[rank]->stream);
            ops::proposal_remap_token_ids(rows_tensor,
                                          static_cast<const std::int32_t*>(ids_d[slot]->p),
                                          kDraftRows, ec.dev[rank]->stream);
            cuda_check(cudaStreamSynchronize(ec.dev[rank]->stream), "cudaStreamSynchronize");
            failures += rows.verify_guards(label + " argmax rank " + std::to_string(rank));
            const auto got =
                from_device<std::int32_t>(rows.data(), static_cast<std::size_t>(tokens));
            failures += verify_exact(
                (label + " remapped draft token rank " + std::to_string(rank)).c_str(), got,
                ref_tokens);
        }
    }

    std::cout << "  draft_head: this fixture's natural argmax winners landed "
              << (saw_upper_half_winner ? "in BOTH vocabulary halves"
                                        : "entirely in the LOWER vocabulary half")
              << "; either way the requirement below is asserted directly rather than sampled\n";

    // WHY draft_head_token_ids MUST BE REPLICATED, asserted deterministically rather than left to
    // whichever rows this fixture's argmax happens to pick. `proposal_remap_token_ids` is applied
    // to a GLOBAL row index in [0,131072) produced by an argmax over the ALLGATHERED logits, so the
    // index can name a row in either device's half. This leg feeds it indices that straddle the
    // 65536 shard boundary -- including the first row of device 1's half and the last row of the
    // vocabulary -- on BOTH devices, against a host oracle. A device holding only its own
    // 65536-entry slice could not answer any of the upper-half rows.
    {
        const std::vector<std::int32_t> probes{0, 1, kDraftShardRows - 1, kDraftShardRows,
                                               kDraftShardRows + 1, kDraftRows - 1};
        std::vector<std::int32_t> expected(probes.size());
        for (std::size_t i = 0; i < probes.size(); ++i) {
            expected[i] = token_ids[static_cast<std::size_t>(probes[i])];
        }
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            set_device(ec, rank);
            GuardedDeviceBuffer rows(probes.size() * sizeof(std::int32_t));
            rows.copy_from_host(probes.data(), probes.size() * sizeof(std::int32_t));
            Tensor rows_tensor(rows.data(), DType::I32,
                               {static_cast<std::int32_t>(probes.size())});
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            ops::proposal_remap_token_ids(rows_tensor,
                                          static_cast<const std::int32_t*>(ids_d[slot]->p),
                                          kDraftRows, ec.dev[rank]->stream);
            cuda_check(cudaStreamSynchronize(ec.dev[rank]->stream), "cudaStreamSynchronize");
            failures += rows.verify_guards("draft_head straddling remap rank " +
                                           std::to_string(rank));
            const auto got = from_device<std::int32_t>(rows.data(), probes.size());
            failures += verify_exact(
                ("draft_head straddling remap rank " + std::to_string(rank)).c_str(), got,
                expected);
        }
        std::cout << "  draft_head: the replicated [131072] id map remaps rows on BOTH sides of the "
                     "65536 shard boundary on BOTH devices -- a vocab-row-split id map could not\n";
    }
    return failures;
}

// =================================================================================================
// Host-only registry probe. Runs before any device check so a one-GPU box still catches an MTP or
// draft-head shard extent dropped from a dispatch table.
// =================================================================================================
int verify_registry() {
    struct Entry {
        const char* label;
        QType qtype;
        std::int32_t n;
        std::int32_t k;
    };
    const std::vector<Entry> entries{
        {"mtp/input_projection shard", QType::W8G32_F16S, kHidden, kFcShardK},
        {"mtp attention shard", QType::W8G32_F16S, kAttnShardRows, kHidden},
        {"mtp o_proj shard", QType::W8G32_F16S, kHidden, kOProjShardK},
        {"mtp mlp/gate_up shard", QType::W8G32_F16S, kGateUpShardRows, kHidden},
        {"mtp mlp/down shard", QType::W8G32_F16S, kHidden, kShardIntermediate},
        {"draft_head shard", QType::Q4G64_F16S, kDraftShardRows, kHidden},
    };
    int failures = 0;
    for (const Entry& entry : entries) {
        for (const std::int32_t tokens : {1, 2, 6, 48}) {
            try {
                (void)ops::linear_workspace_capacity_bytes(entry.qtype, entry.n, entry.k,
                                                           ops::LinearPolicy::A16Only, tokens,
                                                           tokens);
            } catch (const std::exception& error) {
                std::cerr << "registry: " << entry.label << " [" << entry.n << ',' << entry.k
                          << "] T=" << tokens << " rejected: " << error.what() << '\n';
                ++failures;
            }
        }
    }

    // The conv snapshot/record split capacity queries must admit every registered shard format and
    // reject the ones that are deliberately not registered (W8 is the 35B-A3B parent).
    for (const QType qtype : {QType::NVFP4, QType::FP8_E4M3FN_ROW_BF16S, QType::Q4G64_F16S}) {
        try {
            (void)ops::gdn_input_proj_conv_snapshot_column_parallel_workspace_capacity_bytes(
                qtype, ops::LinearPolicy::A16Only, 2, 1, 16);
            (void)ops::gdn_input_proj_conv_record_column_parallel_workspace_capacity_bytes(
                qtype, ops::LinearPolicy::A16Only, 2, 2, 16);
        } catch (const std::exception& error) {
            std::cerr << "registry: gdn conv shard capacity rejected qtype "
                      << static_cast<int>(qtype) << ": " << error.what() << '\n';
            ++failures;
        }
    }
    try {
        (void)ops::gdn_input_proj_conv_snapshot_column_parallel_workspace_capacity_bytes(
            QType::W8G32_F16S, ops::LinearPolicy::A16Only, 1, 1, 4);
        std::cerr << "registry: gdn conv shard capacity admitted W8G32_F16S but must not\n";
        ++failures;
    } catch (const std::exception&) {}
    // Domain guards.
    try {
        (void)ops::gdn_input_proj_conv_record_column_parallel_workspace_capacity_bytes(
            QType::NVFP4, ops::LinearPolicy::A16Only, 1, 1, 4);
        std::cerr << "registry: gdn conv_record shard capacity admitted W=1 but its domain starts "
                     "at 2\n";
        ++failures;
    } catch (const std::exception&) {}

    std::cout << (failures ? "FAIL" : "OK") << " registry: " << entries.size()
              << " MTP/draft shard extents + the conv snapshot/record shard capacity queries\n";
    return failures;
}

// Cross-rank rejection cases only a pair can exercise.
int verify_rejections(const ExecutionContext& ec) {
    int failures = 0;
    const auto expect_throw = [&](const char* label, auto&& body) {
        try {
            body();
            std::cerr << "rejection: " << label << " was accepted but must be rejected\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    };

    // mtp_split_attn_in must reject an unregistered row geometry and a head-count disagreement.
    set_device(ec, 0);
    DeviceBuffer packed(static_cast<std::size_t>(kAttnShardRows) * sizeof(std::uint16_t));
    DeviceBuffer q(static_cast<std::size_t>(kShardQSize) * sizeof(std::uint16_t));
    DeviceBuffer k(static_cast<std::size_t>(kShardKvSize) * sizeof(std::uint16_t));
    DeviceBuffer gate(static_cast<std::size_t>(kShardQSize) * sizeof(std::uint16_t));
    DeviceBuffer v(static_cast<std::size_t>(kShardKvSize) * sizeof(std::uint16_t));
    Tensor tq(q.p, DType::BF16, {kHeadDim, kLocalAttn.q_heads, 1});
    Tensor tk(k.p, DType::BF16, {kHeadDim, kLocalAttn.kv_heads, 1});
    Tensor tgate(gate.p, DType::BF16, {kHeadDim, kLocalAttn.q_heads, 1});
    Tensor tv(v.p, DType::BF16, {kHeadDim, kLocalAttn.kv_heads, 1});
    expect_throw("mtp_split_attn_in with an unregistered row count", [&] {
        const Tensor bad(packed.p, DType::BF16, {kAttnShardRows - 256, 1});
        ops::mtp_split_attn_in(bad, tq, tk, tgate, tv, ec.dev[0]->stream);
    });
    expect_throw("mtp_split_attn_in shard geometry with tp1 head counts", [&] {
        const Tensor shard_packed(packed.p, DType::BF16, {kAttnShardRows, 1});
        Tensor wide_q(q.p, DType::BF16, {kHeadDim, kGlobalAttn.q_heads, 1});
        ops::mtp_split_attn_in(shard_packed, wide_q, tk, tgate, tv, ec.dev[0]->stream);
    });
    std::cout << (failures ? "FAIL" : "OK") << " cross-rank / geometry rejections\n";
    return failures;
}

} // namespace

int main() {
    int failures = verify_registry();
    if (failures != 0) {
        std::cout << "FAIL mtp split (registry)\n";
        return 1;
    }

    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: MTP split parity requires two CUDA devices, found " << device_count
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

    failures += verify_rejections(ec);
    failures += verify_conv_channel_map();
    failures += run_leg_a(ec, events);
    failures += run_leg_b(ec);
    failures += run_leg_c(ec);
    failures += run_leg_d(ec, events);

    std::cout << (failures ? "FAIL" : "OK") << " mtp split (" << failures << " failure(s))\n";
    return failures ? 1 : 0;
}
