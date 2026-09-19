// Host-only table test for the TP2 weight-sharding map.
//
// `plan_for(object, tp, config, profile)` is a pure host computation over TextConfig's compile-time
// dimensions -- no artifact, no device, no kernel. Every case below is derived from the REAL
// object shapes/order bound in targets/qwen3_6_27b/impl/load/bindings.cpp (verified by reading
// that file); see bindings.cpp's plan_for() comments for the two places where the bound objects
// contradicted the original design sketch (attn fused row order, GDN out_proj input width).

#include "targets/qwen3_6_27b/impl/config.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using ninfer::targets::qwen3_6_27b::detail::is_column_parallel_boundary_valid;
using ninfer::targets::qwen3_6_27b::detail::is_row_parallel_boundary_valid;
using ninfer::targets::qwen3_6_27b::detail::plan_for;
using ninfer::targets::qwen3_6_27b::detail::Shard;
using ninfer::targets::qwen3_6_27b::detail::ShardPlan;
using ninfer::targets::qwen3_6_27b::detail::TextConfig;
using ninfer::targets::qwen3_6_27b::detail::WeightsProfile;

namespace {

int g_failures = 0;

void fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    ++g_failures;
}

std::string shard_str(const Shard& s) {
    return "{dev " + std::to_string(s.device) + ", [" + std::to_string(s.row_begin) + ", " +
           std::to_string(s.row_begin + s.row_count) + ")}";
}

std::string plan_str(const ShardPlan& plan) {
    std::string out = "[";
    for (const Shard& s : plan) { out += shard_str(s) + " "; }
    out += "]";
    return out;
}

void expect_plan(const ShardPlan& actual, const ShardPlan& expected, const std::string& label) {
    bool same = actual.size() == expected.size();
    if (same) {
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const Shard& a = actual[i];
            const Shard& e = expected[i];
            if (a.device != e.device || a.row_begin != e.row_begin || a.row_count != e.row_count) {
                same = false;
                break;
            }
        }
    }
    if (!same) {
        fail(label + ": expected " + plan_str(expected) + ", got " + plan_str(actual));
    }
}

void expect_empty(const ShardPlan& actual, const std::string& label) {
    if (!actual.empty()) { fail(label + ": expected empty (replicated), got " + plan_str(actual)); }
}

template <typename Fn>
void expect_throws(Fn&& fn, const std::string& label) {
    try {
        fn();
    } catch (const std::exception&) { return; }
    fail(label + ": expected a throw, none occurred");
}

ShardPlan half_block(std::uint64_t offset, std::uint64_t size) {
    const std::uint64_t chunk = size / 2;
    return {Shard{0, offset, chunk}, Shard{1, offset + chunk, chunk}};
}

ShardPlan concat(const ShardPlan& a, const ShardPlan& b) {
    ShardPlan out = a;
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

} // namespace

int main() {
    const TextConfig config{};
    constexpr auto profile = WeightsProfile::Qwen38Nvfp4;

    // --- tp == 1: degenerate to empty (full copy on device 0), for every family, including
    // families that would otherwise throw at tp > 1 for an unrecognized name. ---
    for (std::string_view object :
         {"text/layers/5/attention/query_key_gate_value", "text/layers/5/attention/output",
          "text/layers/3/gdn/query_key_value_z", "text/layers/3/gdn/output",
          "text/layers/3/gdn/a_projection", "text/layers/3/gdn/a_log",
          "text/layers/5/mlp/gate_up", "text/layers/5/mlp/down", "text/token_embedding",
          "text/output_head", "text/draft_head", "text/draft_head_token_ids",
          "text/final_norm", "mtp/input_projection", "not/a/real/object"}) {
        expect_empty(plan_for(object, 1, config, profile), std::string("tp1 ") + std::string(object));
    }

    // --- attn_input_proj: fused query_key_gate_value, 14336 = Q(6144)|K(1024)|Gate(6144)|V(1024).
    // The section order is q|k|gate|v, confirmed by the Split variant boundary (query_key=Q+K=7168,
    // gate_value=Gate+V=7168). An early design table gave "q|k|v|gate"; that order is wrong and
    // this case is what pins the real one. ---
    {
        const ShardPlan expected = concat(
            concat(half_block(0, 6144), half_block(6144, 1024)),
            concat(half_block(7168, 6144), half_block(13312, 1024)));
        expect_plan(plan_for("text/layers/5/attention/query_key_gate_value", 2, config, profile), expected,
                   "attn fused query_key_gate_value");
        // Same rule applies verbatim to the MTP attention block (same fused shape, 14336).
        expect_plan(plan_for("mtp/layer/attention/query_key_gate_value", 2, config, profile), expected,
                   "mtp attn fused query_key_gate_value");
    }
    // attn_input_proj: split-storage variant (groupwise-int / early nvfp4 layers).
    {
        expect_plan(plan_for("text/layers/3/attention/query_key", 2, config, profile),
                   concat(half_block(0, 6144), half_block(6144, 1024)), "attn split query_key");
        expect_plan(plan_for("text/layers/3/attention/gate_value", 2, config, profile),
                   concat(half_block(0, 6144), half_block(6144, 1024)), "attn split gate_value");
    }
    // o_proj / linear_add: row-parallel over the 6144-wide (query_size) attention-context input.
    {
        const ShardPlan expected = {Shard{0, 0, 3072}, Shard{1, 3072, 3072}};
        expect_plan(plan_for("text/layers/5/attention/output", 2, config, profile), expected, "o_proj");
        expect_plan(plan_for("mtp/layer/attention/output", 2, config, profile), expected, "mtp o_proj");
    }

    // --- GDN input_projection: fused query_key_value_z, 16384 = Q(2048)|K(2048)|V(6144)|Z(6144).
    // The decomposition is qk 4096 | vz 12288, confirmed by the Split variant (query_key=Q+K=4096,
    // value_z=V+Z=12288). An early design table gave "qkv 12288 | z 4096"; that split is wrong and
    // this case is what pins the real one. ---
    {
        const ShardPlan expected =
            concat(concat(half_block(0, 2048), half_block(2048, 2048)),
                  concat(half_block(4096, 6144), half_block(10240, 6144)));
        expect_plan(plan_for("text/layers/3/gdn/query_key_value_z", 2, config, profile), expected,
                   "gdn fused query_key_value_z");
    }
    {
        expect_plan(plan_for("text/layers/3/gdn/query_key", 2, config, profile),
                   concat(half_block(0, 2048), half_block(2048, 2048)), "gdn split query_key");
        expect_plan(plan_for("text/layers/3/gdn/value_z", 2, config, profile),
                   concat(half_block(0, 6144), half_block(6144, 6144)), "gdn split value_z");
    }
    // GDN out_proj / linear_add: row-parallel over the value_dim (6144)-wide input. An early
    // design table gave a 4096-wide input; the real bound shape (bindings.cpp gdn.output, all
    // profiles) is {5120, 6144}, and that is what this case asserts.
    {
        const ShardPlan expected = {Shard{0, 0, 3072}, Shard{1, 3072, 3072}};
        expect_plan(plan_for("text/layers/3/gdn/output", 2, config, profile), expected, "gdn out_proj");
        expect_plan(plan_for("text/layers/3/gdn/output", 2, config, WeightsProfile::Qwen38GgmlK),
                    {Shard{0, 0, 1024}, Shard{1, 1024, 1024},
                     Shard{0, 2048, 1024}, Shard{1, 3072, 1024},
                     Shard{0, 4096, 1024}, Shard{1, 5120, 1024}},
                    "GGUF GDN tiled output columns");
        expect_empty(plan_for("text/layers/3/gdn/output", 1, config, WeightsProfile::Qwen38GgmlK),
                     "GGUF GDN tp1 retains all original columns");
    }

    // --- gdn_gating_proj (a/b) + a_log/dt_bias: 48 rows = 3 rows/alignment-group x 16 groups,
    // column split by group (24/GPU).
    //
    // VERIFIED (mirrors the comment in bindings.cpp above plan_for's gdn/a_b_projection
    // branch): the 48 rows ARE laid out HEAD-MAJOR (row = group_idx * 3 + component) -- row h is
    // GDN value head h, and value head h belongs to qk group h / 3, the mapping the REAL GDN core
    // computes (src/ops/linear_attention/gated_delta_net/common.cuh's `head_map::qk_head`,
    // consumed by recurrent.cuh and chunked/{prepare_wy_wu.cuh,output.cuh}'s own flat
    // `t*H_v + h_v` g/beta indexing); tests/ops/gdn_ref.h::qk_head mirrors that production mapping.
    // So [0,24) and [24,48) each hold a clean set of complete qk groups (0..7 vs 8..15); no group's
    // 3 rows is split across devices. ---
    {
        const ShardPlan expected = {Shard{0, 0, 24}, Shard{1, 24, 24}};
        for (std::string_view leaf :
             {"gdn/a_projection", "gdn/b_projection", "gdn/a_log", "gdn/dt_bias"}) {
            expect_plan(plan_for(std::string("text/layers/3/") + std::string(leaf), 2, config, profile),
                       expected, std::string("gdn gating ") + std::string(leaf));
        }
        expect_plan(plan_for("text/layers/3/gdn/a_b_projection", 2, config, profile),
                   concat(half_block(0, 48), half_block(48, 48)), "gdn fused a_b_projection");
    }

    // --- MLP gate_up: fused, 34816 = Gate(17408)|Up(17408), each split by intermediate/tp. ---
    {
        const ShardPlan expected = concat(half_block(0, 17408), half_block(17408, 17408));
        expect_plan(plan_for("text/layers/5/mlp/gate_up", 2, config, profile), expected, "mlp gate_up");
        expect_plan(plan_for("mtp/layer/mlp/gate_up", 2, config, profile), expected, "mtp mlp gate_up");
    }
    // MLP down: row-parallel over the 17408-wide intermediate input.
    {
        const ShardPlan expected = {Shard{0, 0, 8704}, Shard{1, 8704, 8704}};
        expect_plan(plan_for("text/layers/5/mlp/down", 2, config, profile), expected, "mlp down");
        expect_plan(plan_for("mtp/layer/mlp/down", 2, config, profile), expected, "mtp mlp down");
    }

    // --- token_embedding: replicated (NOT row-split, despite sharing output_head's shape). ---
    expect_empty(plan_for("text/token_embedding", 2, config, profile), "token_embedding");

    // --- output_head / lm_head: row-split by vocab, 248320 -> 124160/GPU. ---
    expect_plan(plan_for("text/output_head", 2, config, profile),
               ShardPlan{Shard{0, 0, 124160}, Shard{1, 124160, 124160}}, "output_head");

    // --- draft_head: row-split by vocab, 131072 -> 65536/GPU. Its companion id map is NOT split:
    // `draft_head_token_ids` is consumed after a GLOBAL argmax over the allgathered
    // 131072-wide proposal logits, so the sampling device needs the whole map -- see the replicated
    // block below and bindings.h's shard taxonomy. ---
    {
        const ShardPlan expected = {Shard{0, 0, 65536}, Shard{1, 65536, 65536}};
        expect_plan(plan_for("text/draft_head", 2, config, profile), expected, "draft_head");
        expect_empty(plan_for("text/draft_head_token_ids", 2, config, profile), "draft_head_token_ids");
    }

    // --- replicated norms: final/input/post-attention/qk norms, gdn/norm, and the MTP-only
    // embedding_norm/hidden_norm. `gdn/norm` is genuinely replicated, not a placeholder: its bound
    // shape is {128}, the per-head-DIMENSION gated_rmsnorm gain, which carries no head axis at
    // all. `gdn/convolution` is NOT replicated -- it carries a real channel split, see below. ---
    for (std::string_view object :
         {"text/final_norm", "text/layers/5/input_norm", "text/layers/5/post_attention_norm",
          "text/layers/5/attention/query_norm", "text/layers/5/attention/key_norm",
          "text/layers/3/gdn/norm", "mtp/layer/input_norm", "mtp/layer/post_attention_norm",
          "mtp/final_norm", "mtp/layer/attention/query_norm", "mtp/layer/attention/key_norm",
          "mtp/embedding_norm", "mtp/hidden_norm"}) {
        expect_empty(plan_for(object, 2, config, profile), std::string("replicated ") + std::string(object));
    }

    // --- gdn/convolution: depthwise conv1d weight over the 10240 GDN qkv channels,
    // stored [4 taps, 10240 channels], so the CHANNEL axis is the artifact's column axis. Split
    // per section by that section's head count, in the Q|K|V order the parent stores and the
    // per-device projection shard reproduces: device r owns Q [1024r, +1024), K [2048+1024r,
    // +1024), V [4096+3072r, +3072) = 5120 channels, concatenated in that order. This is the only
    // object in the whole map whose device shard is more than one column range. ---
    {
        const ShardPlan expected =
            concat(concat(half_block(0, 2048), half_block(2048, 2048)), half_block(4096, 6144));
        expect_plan(plan_for("text/layers/3/gdn/convolution", 2, config, profile), expected,
                   "gdn convolution");
        // The three sections' halves must concatenate to exactly the shard-local q|k|v packing
        // gdn_input_proj_column_parallel writes: qkv[5120,T] as Q [0,1024) | K [1024,2048) |
        // V [2048,5120), i.e. 1024 + 1024 + 3072 (see include/ninfer/ops/gdn_input_proj.h).
        for (int device = 0; device < 2; ++device) {
            std::uint64_t channels = 0;
            for (const Shard& s : expected) {
                if (s.device == device) { channels += s.row_count; }
            }
            if (channels != 5120) {
                fail("gdn convolution device " + std::to_string(device) + ": " +
                     std::to_string(channels) + " channels, expected 5120");
            }
        }
    }

    // --- MTP input_projection: no text-family analog; row-parallel over the mtp_input_rows
    // (10240 = 2*hidden)-wide INPUT axis.
    //
    // VERIFIED against the consumer's own math (it was an assumption when the map was first
    // written): `ops::linear`'s contract is weight [N,K] with K the contraction extent,
    // this object is bound N=5120 / K=10240 (`materialized_weight(..., 5120, 10240)` in
    // bindings.cpp's load block), and `TextContext::mtp_forward_stem` feeds that contraction with
    // `ops::mtp_pack_fc_input(e, h, fc_in)` -> fc_in [10240,T] whose rows [0,5120) are
    // `embedding_norm` and [5120,10240) are `hidden_norm` (mtp_pack.h). So 10240 is the contraction
    // dimension, splitting it is row-parallel by definition, and device r's half is exactly one of
    // the two normalized inputs. Full derivation at plan_for's own comment. ---
    expect_plan(plan_for("mtp/input_projection", 2, config, profile),
               ShardPlan{Shard{0, 0, 5120}, Shard{1, 5120, 5120}}, "mtp input_projection");

    // --- k128 row-split-k128-v1 group boundary (row-parallel objects): standalone validator. ---
    if (!is_row_parallel_boundary_valid(0, 128)) { fail("k128 validator: (0,128) should be valid"); }
    if (is_row_parallel_boundary_valid(100, 100)) {
        fail("k128 validator: (100,100) should be rejected (not 128-aligned)");
    }
    if (is_row_parallel_boundary_valid(128, 100)) {
        fail("k128 validator: (128,100) should be rejected (count not 128-aligned)");
    }

    // --- blockscale-k16-m128x4-v1 row-tile boundary (column-parallel objects that can be
    // NVFP4): standalone validator. Same numeric test as the row-parallel one above (both are
    // 128), but exercised as its own symbol since the two guard different layouts/axes for
    // different (documented, partly unverified) reasons -- see bindings.h. ---
    if (!is_column_parallel_boundary_valid(0, 128)) {
        fail("column-parallel k128 validator: (0,128) should be valid");
    }
    if (is_column_parallel_boundary_valid(100, 100)) {
        fail("column-parallel k128 validator: (100,100) should be rejected (not 128-aligned)");
    }
    if (is_column_parallel_boundary_valid(128, 100)) {
        fail("column-parallel k128 validator: (128,100) should be rejected (count not "
            "128-aligned)");
    }

    // --- k128 rejection end-to-end (row-parallel): mlp/down at tp=32 divides evenly
    // (17408/32=544) but 544 is not a multiple of 128, so the row-parallel split must be
    // rejected even though the head/divisor check alone would have passed. ---
    expect_throws([&] { (void)plan_for("text/layers/5/mlp/down", 32, config, profile); },
                 "mlp/down tp=32 (k128 misaligned)");

    // --- k128 rejection end-to-end (column-parallel, NVFP4-tile guard): mlp/gate_up at tp=32
    // divides evenly (17408/32=544 per Gate/Up half) but 544 is not a multiple of 128, so the
    // column-parallel split must be rejected -- exercising check_nvfp4_tile_alignment=true's
    // append_column_block path, distinct from gdn_gating's check_nvfp4_tile_alignment=false path
    // (tested above, tp=2, 24-row chunks that are NOT 128-aligned but must still succeed). ---
    expect_throws([&] { (void)plan_for("text/layers/5/mlp/gate_up", 32, config, profile); },
                 "mlp/gate_up tp=32 (column-parallel k128 misaligned)");

    // --- head-alignment rejection: attn fused at tp=3 -- query_heads(24) is divisible by 3, but
    // kv_heads(4) is not, so the K/V blocks cannot be split without cutting a KV head in half. ---
    expect_throws(
        [&] { (void)plan_for("text/layers/5/attention/query_key_gate_value", 3, config, profile); },
        "attn fused tp=3 (kv_heads not divisible)");

    // --- tp < 1 is rejected outright. ---
    expect_throws([&] { (void)plan_for("text/token_embedding", 0, config, profile); }, "tp=0");

    // --- unrecognized object family is rejected outright (not silently replicated). ---
    expect_throws([&] { (void)plan_for("vision/merger/fc2", 2, config, profile); }, "unrecognized object");

    // --- shard_mapping_for: the axis the loader slices along. plan_for delegates to it, so the
    // two can never disagree about a family's boundaries; this pins the axis half. Rows narrows
    // the stored row dimension (column-parallel ops), Columns narrows the stored column dimension
    // (row-parallel ops), and Replicated must coincide exactly with an empty plan. ---
    {
        using ninfer::artifact::ShardAxis;
        using ninfer::targets::qwen3_6_27b::detail::shard_mapping_for;
        const auto expect_axis = [&](std::string_view object, ShardAxis expected) {
            const auto mapping = shard_mapping_for(object, 2, config, profile);
            if (mapping.axis != expected) {
                fail(std::string("axis ") + std::string(object) + ": got " +
                     std::to_string(static_cast<int>(mapping.axis)) + ", expected " +
                     std::to_string(static_cast<int>(expected)));
            }
            if ((mapping.axis == ShardAxis::Replicated) != mapping.shards.empty()) {
                fail(std::string("axis ") + std::string(object) +
                     ": replicated and empty-plan disagree");
            }
        };
        for (std::string_view object :
             {"text/layers/5/attention/query_key_gate_value", "text/layers/3/attention/query_key",
              "text/layers/3/attention/gate_value", "text/layers/3/gdn/query_key_value_z",
              "text/layers/3/gdn/query_key", "text/layers/3/gdn/value_z",
              "text/layers/3/gdn/a_projection", "text/layers/3/gdn/a_b_projection",
              "text/layers/3/gdn/a_log", "text/layers/3/gdn/dt_bias", "text/layers/5/mlp/gate_up",
              "text/output_head", "text/draft_head"}) {
            expect_axis(object, ShardAxis::Rows);
        }
        for (std::string_view object :
             {"text/layers/5/attention/output", "text/layers/3/gdn/output",
              "text/layers/5/mlp/down", "mtp/input_projection",
              "text/layers/3/gdn/convolution"}) {
            expect_axis(object, ShardAxis::Columns);
        }
        for (std::string_view object :
             {"text/token_embedding", "text/final_norm", "text/layers/5/input_norm",
              "text/layers/3/gdn/norm", "mtp/hidden_norm", "text/draft_head_token_ids"}) {
            expect_axis(object, ShardAxis::Replicated);
        }
        // tp == 1 degenerates the same way plan_for does, before any family check.
        if (shard_mapping_for("not/a/real/object", 1, config, profile).axis != ShardAxis::Replicated) {
            fail("axis tp1: unrecognized object must degenerate to replicated");
        }
    }

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "shard map table test: all checks passed\n";
    return 0;
}
