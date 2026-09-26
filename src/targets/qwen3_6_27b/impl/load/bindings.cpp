#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include "artifact/typed_binding.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/common/split_launch.h"
#include "ops/linear/fp8/fp8_prepack_sm70.h"
#include "ops/linear/nvfp4/nvfp4_prepack_sm70.h"
#endif
#include "targets/qwen3_6_27b/impl/config.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

using artifact::NumericFormat;

bool is_full_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }

bool is_early_attention_input(std::size_t layer) {
    return layer == 3 || layer == 7 || layer == 11 || layer == 15 || layer == 19 || layer == 23;
}

bool is_bf16_attention_output(std::size_t layer) { return layer == 3 || layer == 7; }

bool is_bf16_gdn_output(std::size_t layer) { return layer == 4; }

NumericFormat endpoint_format(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
        return NumericFormat::Q6G64_F16S;
    case WeightsProfile::Qwen38GroupwiseInt:
    case WeightsProfile::Qwen36Nvfp4:
        return NumericFormat::W8G32_F16S;
    case WeightsProfile::Qwen38Nvfp4:
        return NumericFormat::FP8_E4M3FN_ROW_BF16S;
    case WeightsProfile::Qwen38GgmlK:
        return NumericFormat::GGML_K;
    }
    throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
}

std::uint32_t read_u32_le(std::span<const std::byte> bytes, std::uint64_t offset,
                          std::string_view label) {
    if (offset > bytes.size() || bytes.size() - static_cast<std::size_t>(offset) < 4) {
        throw artifact::ArtifactError(std::string(label) + ": FP32 word is outside payload");
    }
    const std::byte* value = bytes.data() + static_cast<std::size_t>(offset);
    return std::to_integer<std::uint32_t>(value[0]) |
           (std::to_integer<std::uint32_t>(value[1]) << 8U) |
           (std::to_integer<std::uint32_t>(value[2]) << 16U) |
           (std::to_integer<std::uint32_t>(value[3]) << 24U);
}

void require_positive_finite(std::uint32_t bits, std::string_view label) {
    const float value = std::bit_cast<float>(bits);
    if (!std::isfinite(value) || value <= 0.0F) {
        throw artifact::ArtifactError(std::string(label) + ": divisor must be finite and positive");
    }
}

WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape) {
    if (format == NumericFormat::NVFP4) {
        throw std::logic_error("NVFP4 weight requires a paired input divisor");
    }
    return WeightPlan{.object = artifact::bind_device_tensor(binder, name, format, shape),
                      .format = format};
}

WeightPlan bind_nvfp4_weight(artifact::Binder& binder, std::string_view name, std::int32_t rows,
                             std::int32_t columns, std::string_view input_divisor_name) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::ObjectHandle parent      = binder.require_tensor(
        name, NumericFormat::NVFP4, artifact::StorageLayout::BlockScaleK16M128x4V1, shape);
    binder.materialize_on_device(parent);

    const artifact::ObjectHandle input_divisor =
        artifact::bind_tensor(binder, input_divisor_name, NumericFormat::FP32, {},
                              artifact::TensorPlacement::ValidateOnly);
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const std::uint32_t weight_bits =
        read_u32_le(binder.payload(parent).data, geometry.weight_divisor_offset, name);
    const std::uint32_t input_bits =
        read_u32_le(binder.payload(input_divisor).data, 0, input_divisor_name);
    require_positive_finite(weight_bits, name);
    require_positive_finite(input_bits, input_divisor_name);
    return WeightPlan{.object                    = parent,
                      .format                    = NumericFormat::NVFP4,
                      .weight_scale_divisor_bits = weight_bits,
                      .input_scale_divisor_bits  = input_bits};
}

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           const WeightPlan& plan, std::int32_t rows, std::int32_t columns,
                           int device = 0) {
    if (plan.format != NumericFormat::NVFP4) {
        return artifact::materialized_weight(materialized, plan.object, plan.format, rows, columns,
                                             device);
    }

    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    artifact::require_placement_bytes(materialized, plan.object, device, geometry.encoded_bytes);
    const auto* bytes =
        static_cast<const std::byte*>(materialized.device_data(plan.object, device));

    Weight out{};
    out.payload              = bytes;
    out.payload_bytes        = geometry.encoded_bytes;
    out.qtype                = QType::NVFP4;
    out.group_size           = 16;
    out.ndim                 = 2;
    out.qdata                = bytes;
    out.scales               = bytes + geometry.scale_plane_offset;
    out.n                    = rows;
    out.k                    = columns;
    out.group                = 16;
    out.layout               = QuantLayout::BlockScaleK16M128x4;
    out.scale_dtype          = DType::FP8_E4M3FN;
    out.shape[0]             = rows;
    out.shape[1]             = columns;
    out.padded_shape[0]      = rows;
    out.padded_shape[1]      = columns;
    out.weight_scale_divisor = std::bit_cast<float>(plan.weight_scale_divisor_bits);
    out.input_scale_divisor  = std::bit_cast<float>(plan.input_scale_divisor_bits);
    return out;
}

Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    if (block.layout == QuantLayout::GgmlK256 && row_begin >= 0 && row_count > 0 &&
        row_begin + row_count <= block.n) {
        Weight out = block;
        out.qhigh = static_cast<const std::byte*>(block.qhigh) + row_begin * sizeof(std::uint64_t);
        out.n = out.shape[0] = out.padded_shape[0] = row_count;
        if (block.ggml_k_first_q6 >= 0 && block.ggml_k_type_change >= 0) {
            const bool after_change = row_begin >= block.ggml_k_type_change &&
                                      block.ggml_k_type_change < block.n;
            out.ggml_k_first_q6 = block.ggml_k_first_q6 ^ static_cast<int>(after_change);
            const int end = row_begin + row_count;
            out.ggml_k_type_change = block.ggml_k_type_change > row_begin &&
                                             block.ggml_k_type_change < end
                                         ? block.ggml_k_type_change - row_begin
                                         : row_count;
        }
        return out;
    }
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n ||
        block.layout != QuantLayout::RowSplit) {
        throw std::logic_error("invalid target row view");
    }
    const std::uint64_t groups    = static_cast<std::uint64_t>(block.padded_shape[1] / block.group);
    const std::uint64_t low_group = 32;
    const std::uint64_t high_group = block.qtype == QType::Q5G64_F16S   ? 8
                                     : block.qtype == QType::Q6G64_F16S ? 16
                                                                        : 0;
    const std::uint64_t low_row    = groups * low_group;
    const std::uint64_t high_row   = groups * high_group;
    const std::uint64_t scale_row  = groups * 2;
    Weight out                     = block;
    out.qdata                      = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * low_row;
    out.qhigh  = high_group == 0 ? nullptr
                                 : static_cast<const std::byte*>(block.qhigh) +
                                      static_cast<std::uint64_t>(row_begin) * high_row;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

DensePostMixerPayload load_mlp(const MlpPlan& plan,
                               const artifact::MaterializedArtifact& materialized, int tp,
                               int device) {
    DensePostMixerPayload out;
    // gate_up is column-parallel (output rows split); down is row-parallel (input columns split).
    out.gate_up = materialized_weight(materialized, plan.gate_up, 34816 / tp, 5120, device);
    out.down    = materialized_weight(materialized, plan.down, 5120, 17408 / tp, device);
#ifdef NINFER_VOLTA_BUILD
    if (out.gate_up.qtype == QType::NVFP4) {
        ::ninfer::ops::detail::nvfp4_prepack_qpn_sm70(out.gate_up);
        ::ninfer::ops::detail::nvfp4_prepack_qpn_sm70(out.down);
    }
#endif
    return out;
}

FullAttentionProjectionPayload
load_attention_projection(const FullAttentionPlan& plan,
                          const artifact::MaterializedArtifact& materialized, int tp, int device) {
    if (const auto* split = std::get_if<SplitAttentionProjectionPlan>(&plan.projection)) {
        return SplitAttentionProjectionPayload{
            .query_key  = materialized_weight(materialized, split->query_key, 7168 / tp, 5120,
                                              device),
            .gate_value = materialized_weight(materialized, split->gate_value, 7168 / tp, 5120,
                                              device),
        };
    }
    const auto& fused = std::get<FusedAttentionProjectionPlan>(plan.projection);
    return FusedAttentionProjectionPayload{
        .query_key_gate_value =
            materialized_weight(materialized, fused.query_key_gate_value, 14336 / tp, 5120, device),
    };
}

GdnInputProjectionPayload
load_gdn_input_projection(const GdnPlan& plan, const artifact::MaterializedArtifact& materialized,
                          int tp, int device) {
    if (const auto* split = std::get_if<SplitGdnInputProjectionPlan>(&plan.input_projection)) {
        return SplitGdnInputProjectionPayload{
            .query_key = materialized_weight(materialized, split->query_key, 4096 / tp, 5120,
                                             device),
            .value_z = materialized_weight(materialized, split->value_z, 12288 / tp, 5120,
                                           device),
        };
    }
    const auto& fused = std::get<FusedGdnInputProjectionPlan>(plan.input_projection);
    return FusedGdnInputProjectionPayload{
        .query_key_value_z =
            materialized_weight(materialized, fused.query_key_value_z, 16384 / tp, 5120, device),
    };
}

GdnControlProjectionPayload
load_gdn_control_projection(const GdnPlan& plan,
                            const artifact::MaterializedArtifact& materialized, int tp,
                            int device) {
    if (const auto* split = std::get_if<SplitGdnControlProjectionPlan>(&plan.control_projection)) {
        return SplitGdnControlProjectionPayload{
            .a_projection = materialized_weight(materialized, split->a_projection, 48 / tp, 5120,
                                                device),
            .b_projection = materialized_weight(materialized, split->b_projection, 48 / tp, 5120,
                                                device),
        };
    }
    const auto& fused = std::get<FusedGdnControlProjectionPlan>(plan.control_projection);
    return FusedGdnControlProjectionPayload{
        .a_b_projection = materialized_weight(materialized, fused.a_b_projection, 96 / tp, 5120,
                                              device),
    };
}

void bind_groupwise_text_layers(artifact::Binder& binder, BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {5120});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.projection = SplitAttentionProjectionPlan{
                .query_key  = bind_weight(binder, prefix + "attention/query_key",
                                          NumericFormat::Q4G64_F16S, {7168, 5120}),
                .gate_value = bind_weight(binder, prefix + "attention/gate_value",
                                          NumericFormat::Q5G64_F16S, {7168, 5120}),
            };
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output = bind_weight(binder, prefix + "attention/output",
                                                  NumericFormat::Q5G64_F16S, {5120, 6144});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {48});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {48});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 10240});
            target.gdn.control_projection = SplitGdnControlProjectionPlan{
                .a_projection = bind_weight(binder, prefix + "gdn/a_projection",
                                            NumericFormat::BF16, {48, 5120}),
                .b_projection = bind_weight(binder, prefix + "gdn/b_projection",
                                            NumericFormat::BF16, {48, 5120}),
            };
            target.gdn.input_projection = SplitGdnInputProjectionPlan{
                .query_key = bind_weight(binder, prefix + "gdn/query_key",
                                         NumericFormat::Q4G64_F16S, {4096, 5120}),
                .value_z   = bind_weight(binder, prefix + "gdn/value_z", NumericFormat::Q5G64_F16S,
                                         {12288, 5120}),
            };
            target.gdn.norm = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                           NumericFormat::BF16, {128});
            target.gdn.output =
                bind_weight(binder, prefix + "gdn/output", NumericFormat::Q5G64_F16S, {5120, 6144});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        target.mlp.gate_up =
            bind_weight(binder, prefix + "mlp/gate_up", NumericFormat::Q4G64_F16S, {34816, 5120});
        target.mlp.down =
            bind_weight(binder, prefix + "mlp/down", NumericFormat::Q5G64_F16S, {5120, 17408});
    }
}

void bind_nvfp4_text_layers(artifact::Binder& binder, BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {5120});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            WeightPlan input;
            if (is_early_attention_input(layer)) {
                input = bind_weight(binder, prefix + "attention/query_key_gate_value",
                                    NumericFormat::BF16, {14336, 5120});
            } else {
                input = bind_nvfp4_weight(
                    binder, prefix + "attention/query_key_gate_value", 14336, 5120,
                    prefix + "attention/input_projection/input_scale_divisor");
            }
            target.attention.projection =
                FusedAttentionProjectionPlan{.query_key_gate_value = input};
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            if (is_bf16_attention_output(layer)) {
                target.attention.output = bind_weight(binder, prefix + "attention/output",
                                                      NumericFormat::BF16, {5120, 6144});
            } else {
                target.attention.output =
                    bind_nvfp4_weight(binder, prefix + "attention/output", 5120, 6144,
                                      prefix + "attention/output_projection/input_scale_divisor");
            }
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {48});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {48});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 10240});
            target.gdn.control_projection = SplitGdnControlProjectionPlan{
                .a_projection = bind_weight(binder, prefix + "gdn/a_projection",
                                            NumericFormat::BF16, {48, 5120}),
                .b_projection = bind_weight(binder, prefix + "gdn/b_projection",
                                            NumericFormat::BF16, {48, 5120}),
            };
            target.gdn.input_projection = FusedGdnInputProjectionPlan{
                .query_key_value_z =
                    bind_nvfp4_weight(binder, prefix + "gdn/query_key_value_z", 16384, 5120,
                                      prefix + "gdn/input_projection/input_scale_divisor"),
            };
            target.gdn.norm = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                           NumericFormat::BF16, {128});
            if (is_bf16_gdn_output(layer)) {
                target.gdn.output =
                    bind_weight(binder, prefix + "gdn/output", NumericFormat::BF16, {5120, 6144});
            } else {
                target.gdn.output =
                    bind_nvfp4_weight(binder, prefix + "gdn/output", 5120, 6144,
                                      prefix + "gdn/output_projection/input_scale_divisor");
            }
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        target.mlp.gate_up =
            bind_nvfp4_weight(binder, prefix + "mlp/gate_up", 34816, 5120,
                              prefix + "mlp/gate_up_projection/input_scale_divisor");
        target.mlp.down = bind_nvfp4_weight(binder, prefix + "mlp/down", 5120, 17408,
                                            prefix + "mlp/down_projection/input_scale_divisor");
    }
}

bool binds_nvfp4(const artifact::Binder& binder, std::string_view name) {
    const auto* object = binder.find(name);
    const auto* tensor =
        object == nullptr ? nullptr : std::get_if<artifact::TensorDescriptor>(object);
    return tensor != nullptr && tensor->format == NumericFormat::NVFP4;
}

void bind_qwen38_fused_text_layers(artifact::Binder& binder, BindingPlan& out, bool ggml_k) {
    const NumericFormat matrix_format = ggml_k ? NumericFormat::GGML_K : NumericFormat::FP8_E4M3FN_ROW_BF16S;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {5120});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.projection = FusedAttentionProjectionPlan{
                .query_key_gate_value = bind_weight(
                    binder, prefix + "attention/query_key_gate_value", matrix_format, {14336, 5120}),
            };
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output =
                bind_weight(binder, prefix + "attention/output", matrix_format, {5120, 6144});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {48});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {48});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 10240});
            target.gdn.control_projection = FusedGdnControlProjectionPlan{
                .a_b_projection = bind_weight(binder, prefix + "gdn/a_b_projection",
                                              ggml_k ? NumericFormat::GGML_K : NumericFormat::BF16, {96, 5120}),
            };
            target.gdn.input_projection = FusedGdnInputProjectionPlan{
                .query_key_value_z =
                    bind_weight(binder, prefix + "gdn/query_key_value_z", matrix_format, {16384, 5120}),
            };
            target.gdn.norm   = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                             NumericFormat::BF16, {128});
            target.gdn.output = bind_weight(binder, prefix + "gdn/output", matrix_format, {5120, 6144});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {5120});
        if (binds_nvfp4(binder, prefix + "mlp/gate_up")) {
            target.mlp.gate_up =
                bind_nvfp4_weight(binder, prefix + "mlp/gate_up", 34816, 5120,
                                  prefix + "mlp/gate_up_projection/input_scale_divisor");
            target.mlp.down = bind_nvfp4_weight(binder, prefix + "mlp/down", 5120, 17408,
                                                prefix + "mlp/down_projection/input_scale_divisor");
        } else {
            target.mlp.gate_up = bind_weight(binder, prefix + "mlp/gate_up", matrix_format, {34816, 5120});
            target.mlp.down    = bind_weight(binder, prefix + "mlp/down", matrix_format, {5120, 17408});
        }
    }
}

void validate_draft_ids(const artifact::Binder& binder, artifact::ObjectHandle handle) {
    constexpr std::size_t kDraftVocab     = 131072;
    constexpr std::size_t kTokenizerVocab = 248077;
    const auto bytes                      = binder.payload(handle).data;
    std::vector<bool> seen(kTokenizerVocab, false);
    for (std::size_t i = 0; i < kDraftVocab; ++i) {
        const std::byte* value = bytes.data() + i * sizeof(std::uint32_t);
        const std::uint32_t id = std::to_integer<std::uint32_t>(value[0]) |
                                 (std::to_integer<std::uint32_t>(value[1]) << 8U) |
                                 (std::to_integer<std::uint32_t>(value[2]) << 16U) |
                                 (std::to_integer<std::uint32_t>(value[3]) << 24U);
        if (id >= kTokenizerVocab) {
            throw artifact::ArtifactError("draft-head token id is outside tokenizer domain");
        }
        if (seen[id]) { throw artifact::ArtifactError("draft-head token ids are not unique"); }
        seen[id] = true;
    }
}

// --- TP2 shard map ---
// The authoritative statement of the split is this file plus
// tests/targets/test_shard_map.cpp, which pins every plan against the real bound shapes.

// Real fused artifact object shapes fixed outside TextConfig (bind_artifact above).
constexpr std::uint64_t kDraftHeadRows = 131072;

bool ends_with(std::string_view object, std::string_view suffix) {
    return object.size() >= suffix.size() &&
           object.compare(object.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void require(bool condition, std::string_view object, std::string_view reason) {
    if (!condition) {
        throw std::invalid_argument("plan_for(" + std::string(object) + "): " + std::string(reason));
    }
}

// [device * (total / tp), total / tp) -- requires total % tp == 0 (alignment-unit divisibility).
std::pair<std::uint64_t, std::uint64_t> even_chunk(std::uint64_t total, int tp, int device) {
    const std::uint64_t count = total / static_cast<std::uint64_t>(tp);
    return {count * static_cast<std::uint64_t>(device), count};
}

// Appends one Shard per device for a column-parallel split of a single contiguous block
// [block_offset, block_offset + block_size) of the object's real stored rows. `divisor` is the
// alignment unit for this block (a head count for attention/GDN projections, an
// alignment-group count for gdn_gating, or just the block size itself for MLP gate_up and the
// vocab row-splits, which have no head concept at all -- the divisibility check there is really
// just "does this size split evenly by tp").
//
// `check_nvfp4_tile_alignment`: row-split-k128-v1's per-row-independent encoding imposes no
// alignment constraint beyond `divisor` -- but the FUSED attention/GDN input projections and MLP
// gate_up are bound as NVFP4 (blockscale-k16-m128x4-v1) in the nvfp4 weights profiles, whose row
// tiling is 128 (see is_column_parallel_boundary_valid in bindings.h). Pass true for exactly
// those three families; false everywhere else -- the split-storage twins (attention/query_key,
// attention/gate_value, gdn/query_key, gdn/value_z) are groupwise-only, gdn_gating is always
// BF16/FP32 contiguous-le-v1, and the vocab row-splits are always row-split-k128-v1 or
// row-scale-v1. This is a planning pre-check only: the exact per-layout boundary rule is enforced
// against the object's real format by artifact::tensor_row_slice when the shard bytes are planned.
void append_column_block(ShardPlan& plan, std::uint64_t block_offset, std::uint64_t block_size,
                         int tp, std::uint64_t divisor, std::string_view object,
                         bool check_nvfp4_tile_alignment) {
    require(divisor % static_cast<std::uint64_t>(tp) == 0, object,
           "block size is not evenly divisible by tp");
    for (int device = 0; device < tp; ++device) {
        const auto [begin, count] = even_chunk(block_size, tp, device);
        if (check_nvfp4_tile_alignment) {
            require(is_column_parallel_boundary_valid(block_offset + begin, count), object,
                   "column-parallel split violates the blockscale-k16-m128x4-v1 row-tile "
                   "boundary (128) -- see is_column_parallel_boundary_valid in bindings.h");
        }
        plan.push_back(Shard{device, block_offset + begin, count});
    }
}

// Appends one Shard per device for a row-parallel split: `total` is the object's real INPUT
// (column) dimension, sharded per the transposed-view convention documented on `Shard` in
// bindings.h. Each shard must additionally satisfy `is_row_parallel_boundary_valid` -- see that
// function's doc in bindings.h for exactly which storage layouts this is (and is not) derived
// from; `attention/output`, `gdn/output`, and `mlp/down` can be row-split-k128-v1, blockscale, or
// row-scale depending on weights profile, and this check does not distinguish between them.
void append_row_parallel(ShardPlan& plan, std::uint64_t total, int tp, std::uint64_t divisor,
                         std::string_view object) {
    require(divisor % static_cast<std::uint64_t>(tp) == 0, object,
           "input width is not evenly divisible by tp");
    for (int device = 0; device < tp; ++device) {
        const auto [begin, count] = even_chunk(total, tp, device);
        require(is_row_parallel_boundary_valid(begin, count), object,
               "row-parallel split violates the row-split-k128-v1 group boundary (128) -- see "
               "is_row_parallel_boundary_valid in bindings.h");
        plan.push_back(Shard{device, begin, count});
    }
}

// Appends one Shard per device for a head-aligned split of one contiguous block
// [block_offset, block_offset + block_size) of the object's real stored COLUMNS. Used only by
// `gdn/convolution`, whose sliced axis is the depthwise channel axis and whose storage orientation
// puts that axis in the columns (artifact shape [4 taps, 10240 channels], see the family branch
// below). No k128 or NVFP4-tile check applies: the object is always BF16 contiguous-le-v1, whose
// columns are individually addressable.
void append_channel_block(ShardPlan& plan, std::uint64_t block_offset, std::uint64_t block_size,
                          int tp, std::uint64_t divisor, std::string_view object) {
    require(divisor % static_cast<std::uint64_t>(tp) == 0, object,
           "channel block is not evenly divisible by tp");
    for (int device = 0; device < tp; ++device) {
        const auto [begin, count] = even_chunk(block_size, tp, device);
        plan.push_back(Shard{device, block_offset + begin, count});
    }
}

// Vocab row-splits (output_head, draft_head): the row dimension IS the vocab, so this is
// mechanically a column-parallel single-block split, just keyed by row count rather than a head
// count. Always row-split-k128-v1 or row-scale-v1 in every profile (see endpoint_format / the
// draft_head Q4G64_F16S binding above) -- never blockscale, so no NVFP4 tile check applies.
//
// `draft_head_token_ids` is deliberately NOT a member of this family: it is an index map consumed
// after a GLOBAL argmax, so it is replicated. See the replicated block in `shard_mapping` below.
void append_vocab_rows(ShardPlan& plan, std::uint64_t rows, int tp, std::string_view object) {
    append_column_block(plan, 0, rows, tp, rows, object, /*check_nvfp4_tile_alignment=*/false);
}

// The family dispatch behind both `plan_for` and `shard_mapping_for`, so the axis and the shard
// boundaries can never disagree about a family. `tp >= 2` here; the tp<1 / tp==1 contract lives in
// the two public entry points.
ShardMapping shard_mapping(std::string_view object, int tp, const TextConfig& config,
                            WeightsProfile profile) {
    ShardPlan plan;
    const auto ends = [&](std::string_view suffix) { return ends_with(object, suffix); };
    const auto by_rows = [](ShardPlan&& shards) {
        return ShardMapping{artifact::ShardAxis::Rows, std::move(shards)};
    };
    const auto by_columns = [](ShardPlan&& shards) {
        return ShardMapping{artifact::ShardAxis::Columns, std::move(shards)};
    };

    // Replicated: full copy on every device (shards stays empty).
    //
    // `gdn/norm` stays here and that is VERIFIED, not inherited: its real bound shape is {128}
    // (bind_device_tensor above), i.e. the per-head-DIMENSION RMSNorm gain that
    // `ops::gated_rmsnorm` applies as `weight[d]` while normalizing over ne[0]=128
    // (include/ninfer/ops/gated_rmsnorm.h). It carries no head axis at all, so a head-split
    // device still needs all 128 of it. Contrast `gdn/convolution` below, which does carry the
    // head axis and therefore does not belong in this list.
    //
    // `draft_head_token_ids` is here and that is VERIFIED, not inherited: it is the I32 [131072]
    // map from a draft-head ROW INDEX to a tokenizer id, and the runtime consumes it AFTER a
    // global argmax.
    // `TextContext::proposal_argmax` (src/targets/qwen3_6/impl/runtime/text_context_impl.h) runs
    // `ops::argmax` over the whole 131072-wide proposal logit vector and then
    // `ops::proposal_remap_token_ids(tokens, id_map, n=131072, ...)`. At tp2 each device computes
    // half those logits and `ops::allgather_rows` reconstructs the full vector, so the winning index
    // is a GLOBAL row in [0,131072) and can land in either half -- the sampling device needs the
    // WHOLE map, not its own 65536-entry slice. 512 KiB per device, against a ~400 MiB Q4 draft
    // head, so replication costs nothing measurable. `draft_head` itself stays vocab-row-split
    // below.
    if (ends("token_embedding") || ends("final_norm") || ends("input_norm") ||
        ends("post_attention_norm") || ends("attention/query_norm") ||
        ends("attention/key_norm") || ends("gdn/norm") || ends("embedding_norm") ||
        ends("hidden_norm") || ends("draft_head_token_ids")) {
        return {};
    }

    // GDN depthwise conv1d weight: channel-split, NOT replicated.
    //
    // Artifact shape is [4, 10240]: 4 taps x convolution_dim channels, i.e. the CHANNEL axis is
    // the artifact's COLUMN axis (the runtime Tensor is the transpose, {10240, 4}, and the kernels
    // index it `weight[tap * C + c]` -- src/ops/kernel/causal_conv1d.cuh). The 10240 channels are
    // the GDN input projection's qkv block in the same Q|K|V order plan_for splits above:
    // Q [0,2048) = 16 key heads x 128, K [2048,4096) = 16 key heads x 128, V [4096,10240) = 48
    // value heads x 128. The conv is depthwise, so channel c never mixes with any other channel
    // and each device needs exactly the 5120 channels its own projection shard produces.
    //
    // Splitting each of the three sections by its own head count therefore hands device r the
    // channel set [1024r, 1024r+1024) u [2048+1024r, ...) u [4096+3072r, ...), concatenated in
    // that order -- which is byte-for-byte the shard-local q|k|v packing that
    // `gdn_input_proj_column_parallel` writes (q [0,1024), k [1024,2048), v [2048,5120); see
    // include/ninfer/ops/gdn_input_proj.h).
    // The two sides agree structurally, not coincidentally: both derive the boundary from the
    // same contiguous per-head-block split of the same three sections.
    //
    // This object needs three column ranges per device. Qwen38GgmlK gdn/output below also
    // needs three ranges, preserving whole quantization blocks in its tiled V-head order.
    if (ends("gdn/convolution")) {
        const std::uint64_t key   = static_cast<std::uint64_t>(config.key_dim);
        const std::uint64_t value = static_cast<std::uint64_t>(config.value_dim);
        append_channel_block(plan, 0, key, tp, static_cast<std::uint64_t>(config.gdn_key_heads),
                             object);
        append_channel_block(plan, key, key, tp, static_cast<std::uint64_t>(config.gdn_key_heads),
                             object);
        append_channel_block(plan, 2 * key, value, tp,
                             static_cast<std::uint64_t>(config.gdn_value_heads), object);
        return by_columns(std::move(plan));
    }

    // attn_input_proj, fused: 14336 = Q(query_size) | K(kv_size) | Gate(query_size) | V(kv_size).
    // NOTE: this is the real bindings.cpp order (confirmed by the SplitAttentionProjectionPlan
    // boundary: "query_key" = Q+K = query_size+kv_size, "gate_value" = Gate+V). The row order is
    // q | k | gate | v, NOT the "q | k | v | gate" the object name suggests.
    // check_nvfp4_tile_alignment=true: bind_nvfp4_text_layers binds THIS fused object via
    // bind_nvfp4_weight for most layers, so it can be blockscale-k16-m128x4-v1 -- see
    // is_column_parallel_boundary_valid. Its split twins below are a different story.
    if (ends("attention/query_key_gate_value")) {
        const std::uint64_t q = static_cast<std::uint64_t>(config.query_size);
        const std::uint64_t k = static_cast<std::uint64_t>(config.kv_size);
        append_column_block(plan, 0, q, tp, static_cast<std::uint64_t>(config.query_heads), object,
                            true);
        append_column_block(plan, q, k, tp, static_cast<std::uint64_t>(config.kv_heads), object,
                            true);
        append_column_block(plan, q + k, q, tp, static_cast<std::uint64_t>(config.query_heads),
                            object, true);
        append_column_block(plan, q + k + q, k, tp, static_cast<std::uint64_t>(config.kv_heads),
                            object, true);
        return by_rows(std::move(plan));
    }
    // attn_input_proj, split-storage variant: query_key = Q | K (7168), gate_value = Gate | V.
    // check_nvfp4_tile_alignment=false: these two objects exist ONLY in
    // bind_groupwise_text_layers, bound Q4G64_F16S and Q5G64_F16S -- i.e. always
    // row-split-k128-v1, whose rows are independently addressable, so no row-tile constraint
    // applies. (They were previously passed `true` with a comment claiming the NVFP4 profiles
    // bind them; that was wrong -- the NVFP4 profiles bind the *fused* parent instead. The 128
    // requirement was harmless at tp=2 but would have rejected legal splits at higher tp.)
    if (ends("attention/query_key")) {
        const std::uint64_t q = static_cast<std::uint64_t>(config.query_size);
        const std::uint64_t k = static_cast<std::uint64_t>(config.kv_size);
        append_column_block(plan, 0, q, tp, static_cast<std::uint64_t>(config.query_heads), object,
                            false);
        append_column_block(plan, q, k, tp, static_cast<std::uint64_t>(config.kv_heads), object,
                            false);
        return by_rows(std::move(plan));
    }
    if (ends("attention/gate_value")) {
        const std::uint64_t gate = static_cast<std::uint64_t>(config.query_size);
        const std::uint64_t v    = static_cast<std::uint64_t>(config.kv_size);
        append_column_block(plan, 0, gate, tp, static_cast<std::uint64_t>(config.query_heads),
                            object, false);
        append_column_block(plan, gate, v, tp, static_cast<std::uint64_t>(config.kv_heads), object,
                            false);
        return by_rows(std::move(plan));
    }
    // o_proj / linear_add: row-parallel over the query_size-wide attention-context input.
    if (ends("attention/output")) {
        append_row_parallel(plan, static_cast<std::uint64_t>(config.query_size), tp,
                            static_cast<std::uint64_t>(config.query_heads), object);
        return by_columns(std::move(plan));
    }

    // GDN input_projection, fused: 16384 = Q(key_dim) | K(key_dim) | V(value_dim) | Z(value_dim)
    // -- real order confirmed by SplitGdnInputProjectionPlan: "query_key" = Q+K = 2*key_dim
    // (4096), "value_z" = V+Z = 2*value_dim (12288). The layout is qk 4096 | vz 12288, NOT
    // "qkv 12288 | z 4096": Q and K are each key_dim = 2048, V and Z are each value_dim = 6144.
    // check_nvfp4_tile_alignment=true: bind_nvfp4_text_layers binds THIS fused object via
    // bind_nvfp4_weight, so it can be blockscale-k16-m128x4-v1 -- see
    // is_column_parallel_boundary_valid.
    if (ends("gdn/query_key_value_z")) {
        const std::uint64_t key   = static_cast<std::uint64_t>(config.key_dim);
        const std::uint64_t value = static_cast<std::uint64_t>(config.value_dim);
        append_column_block(plan, 0, key, tp, static_cast<std::uint64_t>(config.gdn_key_heads),
                            object, true);
        append_column_block(plan, key, key, tp, static_cast<std::uint64_t>(config.gdn_key_heads),
                            object, true);
        append_column_block(plan, 2 * key, value, tp,
                            static_cast<std::uint64_t>(config.gdn_value_heads), object, true);
        append_column_block(plan, 2 * key + value, value, tp,
                            static_cast<std::uint64_t>(config.gdn_value_heads), object, true);
        return by_rows(std::move(plan));
    }
    // check_nvfp4_tile_alignment=false, same reason as attention/query_key + gate_value above:
    // gdn/query_key (Q4G64) and gdn/value_z (Q5G64) exist only in bind_groupwise_text_layers, so
    // they are always row-split-k128-v1 and never blockscale.
    if (ends("gdn/query_key")) {
        const std::uint64_t key = static_cast<std::uint64_t>(config.key_dim);
        append_column_block(plan, 0, key, tp, static_cast<std::uint64_t>(config.gdn_key_heads),
                            object, false);
        append_column_block(plan, key, key, tp, static_cast<std::uint64_t>(config.gdn_key_heads),
                            object, false);
        return by_rows(std::move(plan));
    }
    if (ends("gdn/value_z")) {
        const std::uint64_t value = static_cast<std::uint64_t>(config.value_dim);
        append_column_block(plan, 0, value, tp, static_cast<std::uint64_t>(config.gdn_value_heads),
                            object, false);
        append_column_block(plan, value, value, tp,
                            static_cast<std::uint64_t>(config.gdn_value_heads), object, false);
        return by_rows(std::move(plan));
    }
    // GDN out_proj / linear_add: row-parallel over the value_dim-wide GDN-context input.
    // NOTE: the bound shape (`gdn.output` in the load block below, both profiles) is
    // {5120, 6144} -- 6144 = value_dim, matching TextConfig::value_dim exactly. It is NOT
    // 5120x4096; 4096 does not appear anywhere in this object.
    if (ends("gdn/output")) {
        if (profile == WeightsProfile::Qwen38GgmlK) {
            // GGUF stores V heads as [repeat,key,128], while the recurrent state and
            // activations are [key,repeat,128]. Retain complete K256 weight blocks:
            // each rank takes its key heads from all three repeat sections. The
            // output-projection Op maps grouped activations to this physical order.
            const auto key_heads = static_cast<std::uint64_t>(config.gdn_key_heads);
            const auto value_heads = static_cast<std::uint64_t>(config.gdn_value_heads);
            const auto repeat_width = static_cast<std::uint64_t>(config.value_dim) *
                                      key_heads / value_heads;
            for (std::uint64_t repeat = 0; repeat < value_heads / key_heads; ++repeat) {
                append_channel_block(plan, repeat * repeat_width, repeat_width, tp,
                                     key_heads, object);
            }
            return by_columns(std::move(plan));
        }
        append_row_parallel(plan, static_cast<std::uint64_t>(config.value_dim), tp,
                            static_cast<std::uint64_t>(config.gdn_value_heads), object);
        return by_columns(std::move(plan));
    }

    // gdn_gating_proj (a/b) + a_log/dt_bias: 48 rows = 3 rows/alignment-group * 16 groups, column
    // split by group (24/GPU at tp=2). `config.gdn_key_heads` (16) is reused here purely as the
    // group COUNT, not as a claim that these are attention/GDN "heads" in the compute sense --
    // these are 16 alignment groups of 3 rows, not literal heads.
    //
    // VERIFIED against the production kernels, not assumed: the 48 rows of
    // a_weight/b_weight/a_log/dt_bias ARE laid out HEAD-MAJOR -- row h (h in [0,48)) is GDN value
    // head h, and value head h belongs to qk (key) head group h / 3. This is exactly the mapping
    // the REAL GDN core computes: src/ops/linear_attention/gated_delta_net/common.cuh's
    // `head_map::qk_head(h_v)` (`h_v / group_size()`, group_size = H_v/H_qk = 48/16 = 3), consumed
    // by recurrent.cuh (decode) and chunked/{prepare_wy_wu.cuh,output.cuh} (prefill, which index
    // g/beta with the same flat `t*H_v + h_v` layout this ShardPlan boundary assumes). The
    // test-only reference model tests/ops/gdn_ref.h::qk_head mirrors this production mapping
    // exactly and was the first place this was checked, but the production kernels are the
    // authority. So row = group_idx * 3 + component holds by construction: the contiguous halves
    // [0,24) / [24,48) each hold a clean, complete set of qk groups (0..7 vs 8..15) -- confirmed,
    // not merely assumed, against the kernels named above.
    // check_nvfp4_tile_alignment=false: gdn/a_projection, gdn/b_projection, gdn/a_log,
    // gdn/dt_bias, and gdn/a_b_projection are always bound BF16 or FP32 (contiguous-le-v1) in
    // every weights profile -- never row-split-k128-v1, blockscale, or row-scale -- so no tile
    // constraint of any kind applies to these objects.
    if (ends("gdn/a_b_projection")) {
        append_column_block(plan, 0, 48, tp, static_cast<std::uint64_t>(config.gdn_key_heads),
                            object, false);
        append_column_block(plan, 48, 48, tp, static_cast<std::uint64_t>(config.gdn_key_heads),
                            object, false);
        return by_rows(std::move(plan));
    }
    if (ends("gdn/a_projection") || ends("gdn/b_projection") || ends("gdn/a_log") ||
        ends("gdn/dt_bias")) {
        append_column_block(plan, 0, 48, tp, static_cast<std::uint64_t>(config.gdn_key_heads),
                            object, false);
        return by_rows(std::move(plan));
    }

    // MLP gate_up, fused: 34816 = Gate(intermediate) | Up(intermediate). There is no "head" here
    // at all -- `divisor == half == intermediate`, so the divisibility check is really just "does
    // the intermediate width split evenly by tp" (see append_column_block's generic error
    // message, deliberately reworded away from "head/group" for this reason).
    // check_nvfp4_tile_alignment=true: bind_nvfp4_text_layers/bind_qwen38_nvfp4_text_layers bind
    // this object via bind_nvfp4_weight for most layers -- see is_column_parallel_boundary_valid.
    if (ends("mlp/gate_up")) {
        const std::uint64_t half = static_cast<std::uint64_t>(config.intermediate);
        append_column_block(plan, 0, half, tp, half, object, true);
        append_column_block(plan, half, half, tp, half, object, true);
        return by_rows(std::move(plan));
    }
    // MLP down: row-parallel over the intermediate-wide input.
    if (ends("mlp/down")) {
        const std::uint64_t intermediate = static_cast<std::uint64_t>(config.intermediate);
        append_row_parallel(plan, intermediate, tp, intermediate, object);
        return by_columns(std::move(plan));
    }

    // MTP input_projection (5120x10240): row-parallel over the mtp_input_rows-wide INPUT axis.
    //
    // VERIFIED against the real MTP kernel, not assumed. No text-family analog exists, so the
    // evidence is the consumer's own math, in three steps that leave no other reading:
    //
    //   1. `ops::linear`'s contract (include/ninfer/ops/linear.h) is weight [N,K] with N the
    //      OUTPUT rows and K the contraction extent. `LoadedModelData` binds this object as
    //      `materialized_weight(backing, plan.mtp.input_projection, W8G32_F16S, /*rows=*/5120,
    //      /*columns=*/10240)` (see `bind_mtp` and the load block below), i.e. N=5120, K=10240 --
    //      so 10240 is the CONTRACTION dimension, not an output dimension.
    //   2. `TextContext::mtp_forward_stem` (src/targets/qwen3_6/impl/runtime/text_context_impl.h)
    //      builds that contraction input with `ops::mtp_pack_fc_input(e, h, fc_in)` -> fc_in
    //      [10240,T], then `ops::linear(fc_in, *mtp_.fc, x)` -> x [5120,T].
    //   3. `mtp_pack_fc_input`'s own definition (include/ninfer/ops/mtp_pack.h) is
    //      out[0:D,t] = embedding_norm[:,t], out[D:2D,t] = hidden_norm[:,t] with D = hidden = 5120.
    //
    // Splitting a contraction axis in half is row-parallel by definition: each device evaluates a
    // full-width partial and one all-reduce completes it. It also fixes WHICH half each device
    // consumes -- device r's shard covers packed rows [5120r, 5120r+5120), which by (3) are exactly
    // `embedding_norm` on rank 0 and `hidden_norm` on rank 1. A tp2 caller therefore skips the pack
    // entirely and feeds `ops::linear_row_parallel({e, h}, shard, ...)` the two unpacked halves;
    // see the tensor-parallelism note on `mtp_pack_fc_input` (include/ninfer/ops/mtp_pack.h).
    if (ends("mtp/input_projection")) {
        const std::uint64_t rows = static_cast<std::uint64_t>(config.mtp_input_rows);
        append_row_parallel(plan, rows, tp, rows, object);
        return by_columns(std::move(plan));
    }

    // Vocab row-splits: replicated token_embedding's twin (output_head) and the draft head.
    // NOTE `draft_head_token_ids` is NOT here -- it is replicated, see the replicated block above.
    if (ends("output_head")) {
        append_vocab_rows(plan, static_cast<std::uint64_t>(config.output_rows), tp, object);
        return by_rows(std::move(plan));
    }
    if (ends("draft_head")) {
        append_vocab_rows(plan, kDraftHeadRows, tp, object);
        return by_rows(std::move(plan));
    }

    throw std::invalid_argument("plan_for: unrecognized object family \"" + std::string(object) +
                                "\"");
}

// Turns a family's shard map into the byte-level placement the binder needs. Devices whose shard
// list is empty under a non-replicated axis cannot happen (every append_* helper emits one shard
// per device), but an empty list is defined as "whole object" by ShardPlacement anyway.
artifact::ShardPlacement shard_placement(std::string_view object, int tp,
                                         const TextConfig& config, WeightsProfile profile) {
    const ShardMapping mapping = shard_mapping_for(object, tp, config, profile);
    artifact::ShardPlacement placement;
    placement.axis = mapping.axis;
    for (const Shard& shard : mapping.shards) {
        if (shard.device < 0 || shard.device >= static_cast<int>(artifact::kMaximumDevices)) {
            throw std::invalid_argument("shard map names a device outside this build's limit");
        }
        placement.device_ranges[static_cast<std::size_t>(shard.device)].push_back(
            artifact::SliceRange{shard.row_begin, shard.row_count});
    }
    return placement;
}

} // namespace

ShardMapping shard_mapping_for(std::string_view object, int tp, const TextConfig& config,
                                WeightsProfile profile) {
    if (tp < 1) { throw std::invalid_argument("plan_for: tp must be >= 1"); }
    if (tp == 1) { return {}; } // degenerate: full copy on device 0, same as replicated.
    return shard_mapping(object, tp, config, profile);
}

ShardPlan plan_for(std::string_view object, int tp, const TextConfig& config,
                    WeightsProfile profile) {
    return shard_mapping_for(object, tp, config, profile).shards;
}

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features, int tp) {
    if (tp < 1 || tp > static_cast<int>(artifact::kMaximumDevices)) {
        throw std::invalid_argument("qwen3_6_27b: tp must be 1 or 2");
    }
    // The binder's arena count and the shard map's device count are two halves of one decision.
    // If they disagree, nothing downstream notices: a tp2 map on a one-device binder would place
    // only device 0's half-shard and size the arena for half a model, and the load would "succeed".
    if (binder.device_count() != tp) {
        throw std::invalid_argument("qwen3_6_27b: binder was built for " +
                                    std::to_string(binder.device_count()) +
                                    " device(s) but bind_artifact was asked for tp " +
                                    std::to_string(tp));
    }
    if (tp > 1) {
        if (features.vision) {
            // The vision tower is out of scope for TP2 and has no shard map, so reject here
            // rather than silently replicating a 4.6 GB backbone onto both devices.
            throw std::invalid_argument("qwen3_6_27b: vision is not supported with tp > 1");
        }
        const TextConfig config{};
        binder.set_shard_resolver([config, tp, weights_profile](std::string_view name) {
            return shard_placement(name, tp, config, weights_profile);
        });
    }
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    out.frontend     = qwen3_6::bind_frontend_resources(binder);
    out.features     = features;
    const bool ggml_k = weights_profile == WeightsProfile::Qwen38GgmlK;
    if (ggml_k && features.vision) {
        throw std::invalid_argument("qwen3.8-27b/gguf-q4-k-m supports Text and MTP; its preserved GGUF Vision weights are not execution-qualified");
    }
    if (ggml_k) {
        out.draft_format = NumericFormat::GGML_K;
        out.mtp_format = NumericFormat::GGML_K;
    }

    const NumericFormat vocabulary_format = endpoint_format(weights_profile);
    out.token_embedding =
        bind_weight(binder, "text/token_embedding", vocabulary_format, {248320, 5120});
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        bind_groupwise_text_layers(binder, out);
        break;
    case WeightsProfile::Qwen36Nvfp4:
        bind_nvfp4_text_layers(binder, out);
        break;
    case WeightsProfile::Qwen38Nvfp4:
        bind_qwen38_fused_text_layers(binder, out, false);
        break;
    case WeightsProfile::Qwen38GgmlK:
        bind_qwen38_fused_text_layers(binder, out, true);
        break;
    default:
        throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
    }
    out.final_norm =
        artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16, {5120});
    out.output_head = bind_weight(binder, "text/output_head", vocabulary_format, {248320, 5120});
    const artifact::TensorPlacement proposal_placement =
        features.optimized_proposal() ? artifact::TensorPlacement::Device
                                      : artifact::TensorPlacement::ValidateOnly;
    out.draft_head = artifact::bind_tensor(binder, "text/draft_head", out.draft_format,
                                           {131072, 5120}, proposal_placement);
    out.draft_head_token_ids = artifact::bind_tensor(
        binder, "text/draft_head_token_ids", NumericFormat::I32, {131072}, proposal_placement);
    validate_draft_ids(binder, out.draft_head_token_ids);

    const artifact::TensorPlacement mtp_placement = features.mtp()
                                                        ? artifact::TensorPlacement::Device
                                                        : artifact::TensorPlacement::ValidateOnly;
    const auto bind_mtp                           = [&](std::string_view name, NumericFormat format,
                              std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, mtp_placement);
    };
    out.mtp.input_projection =
        bind_mtp("mtp/input_projection", out.mtp_format, {5120, 10240});
    out.mtp.embedding_norm       = bind_mtp("mtp/embedding_norm", NumericFormat::BF16, {5120});
    out.mtp.hidden_norm          = bind_mtp("mtp/hidden_norm", NumericFormat::BF16, {5120});
    out.mtp.input_norm           = bind_mtp("mtp/layer/input_norm", NumericFormat::BF16, {5120});
    out.mtp.query_key_gate_value = bind_mtp("mtp/layer/attention/query_key_gate_value",
                                            out.mtp_format, {14336, 5120});
    out.mtp.query_norm = bind_mtp("mtp/layer/attention/query_norm", NumericFormat::BF16, {256});
    out.mtp.key_norm   = bind_mtp("mtp/layer/attention/key_norm", NumericFormat::BF16, {256});
    out.mtp.output =
        bind_mtp("mtp/layer/attention/output", out.mtp_format, {5120, 6144});
    out.mtp.post_attention_norm =
        bind_mtp("mtp/layer/post_attention_norm", NumericFormat::BF16, {5120});
    out.mtp.mlp.gate_up = WeightPlan{
        .object = bind_mtp("mtp/layer/mlp/gate_up", out.mtp_format, {34816, 5120}),
        .format = out.mtp_format};
    out.mtp.mlp.down = WeightPlan{
        .object = bind_mtp("mtp/layer/mlp/down", out.mtp_format, {5120, 17408}),
        .format = out.mtp_format};
    out.mtp.final_norm = bind_mtp("mtp/final_norm", NumericFormat::BF16, {5120});

    const artifact::TensorPlacement vision_placement =
        features.vision ? artifact::TensorPlacement::Device
                        : artifact::TensorPlacement::ValidateOnly;
    if (ggml_k) {
        binder.validate_unconsumed_matching("vision/gguf/");
    } else {
    out.vision_backbone     = qwen3_6::bind_vision_backbone(binder, vision_placement);
    out.vision_merger_input = qwen3_6::bind_vision_merger_input(binder, vision_placement);
    out.vision_merger_fc2   = artifact::bind_tensor(
        binder, "vision/merger/fc2", NumericFormat::W8G32_F16S, {5120, 4608}, vision_placement);
    out.vision_merger_fc2_bias = artifact::bind_tensor(
        binder, "vision/merger/fc2_bias", NumericFormat::BF16, {5120}, vision_placement);
    out.vision_merger_norm = qwen3_6::bind_vision_merger_norm(binder, vision_placement);
    }

    // Validate any companion DFlash/DFlash2 tensors present in artifact so they don't consume VRAM
    binder.validate_unconsumed_matching("dflash");

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized,
                                 int tensor_parallel)
    : backing(std::move(materialized)), tp(tensor_parallel) {
    if (tp != 1 && tp != 2) { throw std::invalid_argument("qwen3_6_27b: tp must be 1 or 2"); }
    if (tp > backing.device_count()) {
        throw std::invalid_argument("qwen3_6_27b: tp exceeds the materialized device count");
    }
    frontend = qwen3_6::take_frontend_resources(backing, plan.frontend);

    build_device_view(plan, 0, runtime);
    if (tp == 2) { build_device_view(plan, 1, runtime_peer.emplace()); }
}

// Builds `device`'s own model view. At tp == 1 this is the whole model on device 0; at tp == 2
// every extent divided by `tp` below is the axis the ShardPlan splits, so the view describes
// exactly the shard that device's arena holds. `artifact::materialized_{tensor,weight}` now check
// the claimed shape against the placed bytes, so a missed division throws instead of reading past
// the shard.
void LoadedModelData::build_device_view(const BindingPlan& plan, int device,
                                        RuntimeModelView& runtime) {
#ifdef NINFER_VOLTA_BUILD
    const ::ninfer::ops::detail::CurrentDeviceScope device_scope;
    ::ninfer::ops::detail::CurrentDeviceScope::set(backing.physical_device(device));
#endif
    runtime.weights_arena = &backing.device_arena(device);
    runtime.features      = plan.features;
    auto& token_embedding = runtime.token_embedding;
    auto& full_layers     = runtime.full_layers;
    auto& gdn_layers      = runtime.gdn_layers;
    auto& final_norm      = runtime.final_norm;
    auto& output_head     = runtime.output_head;

    // Replicated: every device holds the whole object (token_embedding, all RMSNorm gains,
    // gdn/norm, draft_head_token_ids). Sharded extents carry an explicit `/ tp`.
    token_embedding        = materialized_weight(backing, plan.token_embedding, 248320, 5120,
                                                 device);
    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm =
                artifact::materialized_tensor(backing, source.input_norm, NumericFormat::BF16,
                                              {5120}, device);
            target.projection = load_attention_projection(source.attention, backing, tp, device);
            target.query_norm = artifact::materialized_tensor(backing, source.attention.query_norm,
                                                              NumericFormat::BF16, {256}, device);
            target.key_norm   = artifact::materialized_tensor(backing, source.attention.key_norm,
                                                              NumericFormat::BF16, {256}, device);
            target.output = materialized_weight(backing, source.attention.output, 5120, 6144 / tp,
                                                device);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {5120}, device);
            target.post_mixer = load_mlp(source.mlp, backing, tp, device);
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm  = artifact::materialized_tensor(backing, source.input_norm,
                                                               NumericFormat::BF16, {5120}, device);
            target.projection.a_log = artifact::materialized_tensor(
                backing, source.gdn.a_log, NumericFormat::FP32, {48 / tp}, device);
            target.projection.dt_bias = artifact::materialized_tensor(
                backing, source.gdn.dt_bias, NumericFormat::FP32, {48 / tp}, device);
            // Channel-split depthwise conv: the runtime Tensor is the transpose of the artifact's
            // [4 taps, C channels] storage, so the SHARDED axis is ne[0].
            target.convolution = artifact::materialized_tensor(
                backing, source.gdn.convolution, NumericFormat::BF16, {10240 / tp, 4}, device);
            target.projection.control_projection =
                load_gdn_control_projection(source.gdn, backing, tp, device);
            target.projection.input_projection =
                load_gdn_input_projection(source.gdn, backing, tp, device);
            target.norm = artifact::materialized_tensor(backing, source.gdn.norm,
                                                        NumericFormat::BF16, {128}, device);
            target.output =
                materialized_weight(backing, source.gdn.output, 5120, 6144 / tp, device);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {5120}, device);
            target.post_mixer = load_mlp(source.mlp, backing, tp, device);
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("text topology binding is incomplete");
    }
    final_norm = artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16,
                                               {5120}, device);
    output_head = materialized_weight(backing, plan.output_head, 248320 / tp, 5120, device);
#ifdef NINFER_VOLTA_BUILD
    if (output_head.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        ::ninfer::ops::detail::fp8_prepack_qpn_sm70(output_head);
    }
#endif
    if (plan.features.optimized_proposal()) {
        auto& proposal = runtime.optimized_proposal.emplace();
        proposal.head =
            artifact::materialized_weight(backing, plan.draft_head, plan.draft_format,
                                          131072 / tp, 5120, device);
        // Replicated: the winning index comes from a GLOBAL argmax over the allgathered proposal
        // logits and can name a row in either half.
        proposal.token_ids = artifact::materialized_tensor(backing, plan.draft_head_token_ids,
                                                           NumericFormat::I32, {131072}, device);
    }

    if (plan.features.mtp()) {
        // MTP shard map. Every extent that the ShardPlan splits carries an explicit `/ tp`
        // here, on the SAME axis the plan splits:
        //   * input_projection [5120, 10240] is ROW-parallel over its 10240-wide contraction
        //     axis, so `columns` is divided. Device r's shard contracts packed rows
        //     [5120r, 5120r+5120), which by mtp_pack_fc_input's definition are the normalized
        //     EMBEDDING on rank 0 and the normalized HIDDEN on rank 1 -- which is why the tp2
        //     forward skips ops::mtp_pack_fc_input entirely.
        //   * query_key_gate_value [14336, 5120] is column-parallel with per-section head
        //     splits, so the shard's row order is q(3072) | k(512) | gate(3072) | v(512) at
        //     tp == 2 -- the same interleaving the text attention parent uses. The four row
        //     views below are therefore expressed in per-shard section widths, not literals.
        //   * attention/output [5120, 6144] and mlp/down are row-parallel (contraction split);
        //     mlp/gate_up is column-parallel. `load_mlp` already encodes both.
        //   * every norm is replicated: full copy per device, only the `device` argument moves.
        auto& mtp            = runtime.mtp.emplace();
        mtp.input_projection = artifact::materialized_weight(
            backing, plan.mtp.input_projection, plan.mtp_format, 5120, 10240 / tp,
            device);
        mtp.embedding_norm   = artifact::materialized_tensor(backing, plan.mtp.embedding_norm,
                                                             NumericFormat::BF16, {5120}, device);
        mtp.hidden_norm      = artifact::materialized_tensor(backing, plan.mtp.hidden_norm,
                                                             NumericFormat::BF16, {5120}, device);
        mtp.input_norm       = artifact::materialized_tensor(backing, plan.mtp.input_norm,
                                                             NumericFormat::BF16, {5120}, device);
        mtp.attention.packed = artifact::materialized_weight(
            backing, plan.mtp.query_key_gate_value, plan.mtp_format, 14336 / tp, 5120,
            device);
        const std::int32_t mtp_query_section = 6144 / tp;
        const std::int32_t mtp_kv_section    = 1024 / tp;
        mtp.attention.query = row_view(mtp.attention.packed, 0, mtp_query_section);
        mtp.attention.key   = row_view(mtp.attention.packed, mtp_query_section, mtp_kv_section);
        mtp.attention.output_gate =
            row_view(mtp.attention.packed, mtp_query_section + mtp_kv_section, mtp_query_section);
        mtp.attention.value = row_view(mtp.attention.packed,
                                       2 * mtp_query_section + mtp_kv_section, mtp_kv_section);
        mtp.query_norm      = artifact::materialized_tensor(backing, plan.mtp.query_norm,
                                                            NumericFormat::BF16, {256}, device);
        mtp.key_norm        = artifact::materialized_tensor(backing, plan.mtp.key_norm,
                                                            NumericFormat::BF16, {256}, device);
        mtp.output = artifact::materialized_weight(backing, plan.mtp.output,
                                                   plan.mtp_format, 5120, 6144 / tp,
                                                   device);
        mtp.post_attention_norm = artifact::materialized_tensor(
            backing, plan.mtp.post_attention_norm, NumericFormat::BF16, {5120}, device);
        mtp.post_mixer = load_mlp(plan.mtp.mlp, backing, tp, device);
        mtp.final_norm = artifact::materialized_tensor(backing, plan.mtp.final_norm,
                                                       NumericFormat::BF16, {5120}, device);
    }

    if (plan.features.vision) {
        if (tp != 1) {
            throw std::invalid_argument(
                "qwen3_6_27b: Vision has no tensor-parallel forward path yet");
        }
        auto& vision  = runtime.vision.emplace();
        vision.common = qwen3_6::materialize_vision_common(
            backing, plan.vision_backbone, plan.vision_merger_input, plan.vision_merger_norm);
        vision.merger_fc2      = artifact::materialized_weight(backing, plan.vision_merger_fc2,
                                                               NumericFormat::W8G32_F16S, 5120, 4608);
        vision.merger_fc2_bias = artifact::materialized_tensor(backing, plan.vision_merger_fc2_bias,
                                                               NumericFormat::BF16, {5120});
    }
}

} // namespace ninfer::targets::qwen3_6_27b::detail
