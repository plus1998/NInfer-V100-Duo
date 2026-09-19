#pragma once

#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/model_view.h>
#include <ninfer/targets/qwen3_6/startup_features.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {

inline constexpr std::size_t kTextLayers          = 64;
inline constexpr std::size_t kFullAttentionLayers = 16;
inline constexpr std::size_t kGdnLayers           = 48;

// --- TP2 host-side sharding map ---
//
// A `Shard` names one contiguous row range of an object's *logical* 2D matrix that lives on
// one device. `row_begin`/`row_count` always index the same axis the object is bound with in
// `bind_weight`/`bind_device_tensor` above (the artifact's stored [rows, columns] shape), with
// one exception documented per family below:
//
//   - column-parallel objects (the output/row dimension is split, e.g. fused QKV-gate
//     projections, GDN input_projection, MLP gate_up, gdn_gating a/b, a_log/dt_bias, vocab
//     row-splits): `row_begin`/`row_count` slice the object's real stored rows directly. The
//     `row-split-k128-v1` storage layout (src/artifact/storage_layouts.cpp) encodes each row
//     independently, so any row boundary is a valid split point there -- the only requirement is
//     *head alignment* (never cut a projection head/group in half). Formats that can be bound in
//     the NVFP4 profile use `blockscale-k16-m128x4-v1` instead, which is NOT row-independent (see
//     `is_column_parallel_boundary_valid` below).
//   - row-parallel objects (the input/column dimension is split, e.g. o_proj, GDN out_proj, MLP
//     down): the object is stored as [out_rows, in_columns], but a Shard's `row_begin`/
//     `row_count` here index the INPUT dimension -- i.e. rows of the object's *transposed* view.
//     `row-split-k128-v1` groups each row's columns into 64- or
//     32-wide quantization groups, so a column split of a `row-split-k128-v1` object must land on
//     a group boundary (see `is_row_parallel_boundary_valid` below for what this does and does
//     not cover).
//   - channel-split objects (`gdn/convolution`, the only member): a depthwise conv1d weight whose
//     ONE meaningful axis is the channel, and whose storage orientation puts that axis in the
//     artifact's COLUMNS -- stored [4 taps, 10240 channels], while the runtime Tensor is the
//     transpose {10240, 4} and the kernels index it `weight[tap * C + c]`. So this family shares
//     the row-parallel families' axis (`ShardAxis::Columns`, `row_begin`/`row_count` index the
//     stored column dimension) but not their meaning: nothing is reduced across devices, and the
//     boundary requirement is head alignment, exactly as for the column-parallel families. It is
//     one of two families whose per-device shard is MORE THAN ONE range on the column axis: the
//     10240 channels are the GDN input projection's Q(2048) | K(2048) | V(6144) block, each
//     section split by its own head count, so device r owns three disjoint channel blocks
//     concatenated in Q|K|V order (see `plan_for`'s `gdn/convolution` branch). `artifact::
//     tensor_column_slice` admits these ranges for `contiguous-le-v1`, which this object
//     always is (BF16). Qwen38GgmlK gdn/output also uses three ranges: its original GGUF
//     columns are [repeat,key,128], so each rank selects its key heads from each repeat
//     section. Every range preserves whole K256 blocks. The projection Op maps grouped
//     activations to the retained tiled column order.
//   - replicated objects (norms, token_embedding, gdn/norm, draft_head_token_ids): `shards` is
//     empty, meaning a full copy lives on every device (see `WeightPlan::shards` below).
//     `gdn/norm` belongs here on its merits, not by default: its bound shape is {128}, the
//     per-head-DIMENSION `gated_rmsnorm` gain applied as `weight[d]` while normalizing over
//     ne[0]=128, so it carries no head axis to split and every device needs all of it.
//     `draft_head_token_ids` (I32 [131072]) is here for a different reason: it maps a draft-head
//     ROW INDEX to a tokenizer id and is consumed AFTER a GLOBAL argmax over the allgathered
//     131072-wide proposal logits, so the winning index can name a row in either device's half
//     and the sampling device needs the whole map. Its companion weight `draft_head` is still
//     vocab-row-split. Note that neither `gdn/convolution` (channel-split, above) nor
//     `draft_head_token_ids` belongs to the vocab row-split family, however plausible that
//     placement looks from the object names alone.
//
// `tp == 1` always degenerates to an empty ShardPlan for every object (full copy on device 0),
// matching `WeightPlan::shards`'s "empty => replicated/full on device 0" convention; this is the
// same convention used for genuinely-replicated objects, so tp1 callers cannot tell sharded and
// replicated objects apart, by design (there is only one device).
struct Shard {
    int device = 0;
    std::uint64_t row_begin = 0;
    std::uint64_t row_count = 0;
};

using ShardPlan = std::vector<Shard>;

// Shard boundaries in the families below are required to be multiples of this value. Both
// validators use it, for reasons that are now derived rather than guessed -- see each one. The
// validators here are a *planning* pre-check made without knowing an object's runtime
// `NumericFormat`; the binding authority is `artifact::tensor_row_slice` /
// `artifact::tensor_column_slice` (src/artifact/storage_layouts.cpp), which knows the real layout
// and rejects an illegal boundary exactly, per layout, when the shard bytes are planned.
inline constexpr std::uint64_t kQuantRowSplitAlignment = 128;

// True iff [begin, begin + count) is a valid row-parallel (input/COLUMN-dimension) split boundary
// for every layout `attention/output`, `gdn/output`, `mlp/down`, and `mtp/input_projection` can
// carry. Column constraints per layout (all three verified against storage-layouts.md and
// tools/artifact/layouts.py, the encoder of record):
//   - `row-split-k128-v1`: columns are packed into quantization groups (64 for Q4/Q5/Q6G64, 32
//     for W8G32), so a column boundary must be a group multiple. 128 is the layout's own
//     K-alignment unit and a multiple of both group sizes; requiring it additionally makes each
//     shard's own K_pad equal to its K, so no shard inherits the parent's trailing padding
//     groups. This is the binding (strictest) constraint of the three.
//   - `blockscale-k16-m128x4-v1`: the scale plane's second axis is the 64-column scale tile, so
//     columns must be a multiple of 64. 128 satisfies it.
//   - `row-scale-v1`: the code plane is row-major bytes and the BF16 multiplier is per row, so
//     any column boundary is legal. 128 satisfies it.
[[nodiscard]] constexpr bool is_row_parallel_boundary_valid(std::uint64_t row_begin,
                                                             std::uint64_t row_count) {
    return row_begin % kQuantRowSplitAlignment == 0 && row_count % kQuantRowSplitAlignment == 0;
}

// True iff [row_begin, row_begin + row_count) is a valid column-parallel (output/ROW-dimension)
// split boundary under NVFP4's `blockscale-k16-m128x4-v1` row tiling. This is now PROVEN, not
// defensive: the swizzled scale plane stores word (n, g) at
// `(floor(n/128) * K_tiles + floor(g/4)) * 512 + (n%32) * 16 + floor((n%128)/32) * 4 + g%4`
// (storage-layouts.md section 4; produced by `swizzle_nvfp4_scales` in tools/artifact/layouts.py
// as `reshape(N/128, 4, 32, K_tiles, 4).permute(0, 3, 2, 1, 4)`). Rows are therefore interleaved
// *within* each 128-row tile, and only a 128-row-aligned range is a whole number of tiles -- so
// only such a range is both extractable and re-encodable as a standalone tensor.
//
// Applied to the families that are bound `bind_nvfp4_weight` in some profile: `attention/
// query_key_gate_value`, `gdn/query_key_value_z`, and `mlp/gate_up`. Deliberately NOT applied to:
//   - the split-storage twins `attention/query_key`, `attention/gate_value`, `gdn/query_key`,
//     `gdn/value_z`. These exist only in `bind_groupwise_text_layers`, always Q4G64/Q5G64, hence
//     always `row-split-k128-v1`, whose rows are independently addressable -- a 128-row
//     requirement is spurious there. (An earlier revision of this comment claimed these were
//     NVFP4-bound in some profile; that was factually wrong. The fused parents are; these are
//     not.)
//   - `gdn_gating` (`gdn/a_projection`, `gdn/b_projection`, `gdn/a_log`, `gdn/dt_bias`,
//     `gdn/a_b_projection`): always BF16/FP32 `contiguous-le-v1`, row-independent, and split into
//     24-row halves that are not 128-aligned by construction.
//   - the vocab row-splits (`output_head`, `draft_head`): always `row-split-k128-v1` or
//     `row-scale-v1`, both row-independent.
// If any of those families ever gains an NVFP4 binding, `tensor_row_slice` rejects a misaligned
// boundary at bind time with the exact layout reason, so this list being conservative cannot let
// a wrong copy through.
[[nodiscard]] constexpr bool is_column_parallel_boundary_valid(std::uint64_t row_begin,
                                                                std::uint64_t row_count) {
    return row_begin % kQuantRowSplitAlignment == 0 && row_count % kQuantRowSplitAlignment == 0;
}

struct TextConfig;

// A family's shard map together with the axis it splits, which is what a byte-level slice needs
// (`artifact::ShardAxis::Rows` narrows the stored row dimension, `Columns` narrows the stored
// column dimension -- the "transposed view" the `Shard` comment above describes for row-parallel
// objects, and also the channel axis of the channel-split `gdn/convolution`). `plan_for` is this
// minus the axis. Note that `Columns` therefore covers two families with different meanings and
// different range counts per device -- see the `Shard` taxonomy above.
struct ShardMapping {
    artifact::ShardAxis axis = artifact::ShardAxis::Replicated;
    ShardPlan shards;
};

// Same contract as `plan_for` below, plus the axis. `tp == 1` returns a replicated mapping with no
// shards before any family check runs, exactly as `plan_for` does.
[[nodiscard]] ShardMapping shard_mapping_for(std::string_view object, int tp,
                                             const TextConfig& config, WeightsProfile profile);

// Computes the TP2 shard map for one artifact weight object. `object` is matched by suffix
// against the local binder names used in bindings.cpp (e.g. "attention/query_key_gate_value",
// "mlp/down", "gdn/a_log", "output_head"), so it accepts either a bare leaf name or a full
// artifact path ("text/layers/5/mlp/down" and "mlp/down" both match the MLP-down rule).
//
// Throws std::invalid_argument for: `tp < 1`; an object name matching no known family; or a
// split that would violate head alignment or (for row-parallel objects) the k128 group
// boundary. `tp == 1` always returns an empty ShardPlan before any of those checks run.
[[nodiscard]] ShardPlan plan_for(std::string_view object, int tp, const TextConfig& config,
                                 WeightsProfile profile);

struct WeightPlan {
    artifact::ObjectHandle object;
    artifact::NumericFormat format          = artifact::NumericFormat::BF16;
    std::uint32_t weight_scale_divisor_bits = 0;
    std::uint32_t input_scale_divisor_bits  = 0;
    std::vector<Shard> shards; // empty => replicated/full on device 0
};

struct MlpPlan {
    WeightPlan gate_up;
    WeightPlan down;
};

struct SplitAttentionProjectionPlan {
    WeightPlan query_key;
    WeightPlan gate_value;
};

struct FusedAttentionProjectionPlan {
    WeightPlan query_key_gate_value;
};

struct FullAttentionPlan {
    std::variant<SplitAttentionProjectionPlan, FusedAttentionProjectionPlan> projection;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan output;
};

struct SplitGdnInputProjectionPlan {
    WeightPlan query_key;
    WeightPlan value_z;
};

struct FusedGdnInputProjectionPlan {
    WeightPlan query_key_value_z;
};

struct SplitGdnControlProjectionPlan {
    WeightPlan a_projection;
    WeightPlan b_projection;
};

struct FusedGdnControlProjectionPlan {
    WeightPlan a_b_projection;
};

using GdnControlProjectionPlan =
    std::variant<SplitGdnControlProjectionPlan, FusedGdnControlProjectionPlan>;

struct GdnPlan {
    artifact::ObjectHandle a_log;
    artifact::ObjectHandle dt_bias;
    artifact::ObjectHandle convolution;
    GdnControlProjectionPlan control_projection;
    std::variant<SplitGdnInputProjectionPlan, FusedGdnInputProjectionPlan> input_projection;
    artifact::ObjectHandle norm;
    WeightPlan output;
};

struct TextLayerPlan {
    artifact::ObjectHandle input_norm;
    FullAttentionPlan attention{};
    GdnPlan gdn{};
    bool is_full_attention = false;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
};

struct MtpPlan {
    artifact::ObjectHandle input_projection;
    artifact::ObjectHandle embedding_norm;
    artifact::ObjectHandle hidden_norm;
    artifact::ObjectHandle input_norm;
    artifact::ObjectHandle query_key_gate_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle output;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
    artifact::ObjectHandle final_norm;
};

struct BindingPlan {
    qwen3_6::FrontendResourcePlan frontend;
    qwen3_6::StartupFeatures features;
    artifact::NumericFormat draft_format = artifact::NumericFormat::Q4G64_F16S;
    artifact::NumericFormat mtp_format = artifact::NumericFormat::W8G32_F16S;

    WeightPlan token_embedding;
    std::array<TextLayerPlan, kTextLayers> text_layers;
    artifact::ObjectHandle final_norm;
    WeightPlan output_head;
    artifact::ObjectHandle draft_head;
    artifact::ObjectHandle draft_head_token_ids;
    MtpPlan mtp;

    qwen3_6::VisionBackbonePlan vision_backbone;
    qwen3_6::VisionMergerInputPlan vision_merger_input;
    artifact::ObjectHandle vision_merger_fc2;
    artifact::ObjectHandle vision_merger_fc2_bias;
    qwen3_6::VisionMergerNormPlan vision_merger_norm;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

// Binds every artifact object this target needs. `tp` is the tensor-parallel width the binder was
// constructed for: at tp == 1 nothing about the placement changes (identical offsets, identical
// arena size), and at tp > 1 a shard resolver derived from `shard_mapping_for` is installed on the
// binder so each device's arena receives only its own rows/columns.
ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features, int tp = 1);

struct DensePostMixerPayload {
    Weight gate_up;
    Weight down;
};

struct SplitAttentionProjectionPayload {
    Weight query_key;
    Weight gate_value;
};

struct FusedAttentionProjectionPayload {
    Weight query_key_gate_value;
};

using FullAttentionProjectionPayload =
    std::variant<SplitAttentionProjectionPayload, FusedAttentionProjectionPayload>;

struct SplitGdnInputProjectionPayload {
    Weight query_key;
    Weight value_z;
};

struct FusedGdnInputProjectionPayload {
    Weight query_key_value_z;
};

using GdnInputProjectionPayload =
    std::variant<SplitGdnInputProjectionPayload, FusedGdnInputProjectionPayload>;

struct SplitGdnControlProjectionPayload {
    Weight a_projection;
    Weight b_projection;
};

struct FusedGdnControlProjectionPayload {
    Weight a_b_projection;
};

using GdnControlProjectionPayload =
    std::variant<SplitGdnControlProjectionPayload, FusedGdnControlProjectionPayload>;

struct GdnProjectionPayload {
    Tensor a_log;
    Tensor dt_bias;
    GdnControlProjectionPayload control_projection;
    GdnInputProjectionPayload input_projection;
};

struct MtpAttentionPayload {
    Weight packed;
    Weight query;
    Weight key;
    Weight output_gate;
    Weight value;
};

using RuntimeModelView =
    qwen3_6::ModelView<FullAttentionProjectionPayload, GdnProjectionPayload, DensePostMixerPayload,
                       MtpAttentionPayload, DensePostMixerPayload, qwen3_6::DFlashWeights<6>,
                       kFullAttentionLayers, kGdnLayers>;
using FullAttentionWeights = RuntimeModelView::FullLayer;
using GdnWeights           = RuntimeModelView::GdnLayer;
using MtpWeights           = RuntimeModelView::MtpLayer;

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized,
                    int tensor_parallel = 1);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    // The model view for one rank. `runtime` is rank 0's (the only one at tp == 1); `runtime_peer`
    // holds rank 1's at tp == 2. Both describe the SHARD that rank's arena holds, not the whole
    // model -- every sharded extent in the view is already divided by tp.
    [[nodiscard]] const RuntimeModelView& view(int rank) const {
        if (rank == 0) { return runtime; }
        if (rank == 1 && runtime_peer.has_value()) { return *runtime_peer; }
        throw std::out_of_range("qwen3_6_27b model view rank is out of range");
    }

    artifact::MaterializedArtifact backing;
    qwen3_6::FrontendResources frontend;
    int tp = 1;
    RuntimeModelView runtime;
    std::optional<RuntimeModelView> runtime_peer;

private:
    void build_device_view(const BindingPlan& plan, int device, RuntimeModelView& runtime);
};

class LoadedModel::Impl {
public:
    Impl(WeightsProfile weights_profile_in, BindingPlan plan,
         artifact::MaterializedArtifact materialized, int tensor_parallel)
        : weights_profile(weights_profile_in),
          data(std::move(plan), std::move(materialized), tensor_parallel) {}

    WeightsProfile weights_profile;
    LoadedModelData data;
};

} // namespace ninfer::targets::qwen3_6_27b::detail
