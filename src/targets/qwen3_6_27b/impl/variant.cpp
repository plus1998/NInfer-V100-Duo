#include "targets/qwen3_6_27b/impl/variant.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
}

#if defined(NINFER_SM8X_COMPAT) || defined(NINFER_VOLTA_BUILD)
constexpr ops::LinearPolicy kNvfp4TextPolicy = ops::LinearPolicy::A16Only;
constexpr ops::LinearPolicy kFp8TextPolicy   = ops::LinearPolicy::A16Only;
#else
constexpr ops::LinearPolicy kNvfp4TextPolicy = ops::LinearPolicy::AllowA4;
constexpr ops::LinearPolicy kFp8TextPolicy   = ops::LinearPolicy::AllowA8;
#endif

ops::LinearPolicy text_policy(const Weight& weight) {
    switch (weight.qtype) {
    case QType::NVFP4:
        return kNvfp4TextPolicy;
    case QType::FP8_E4M3FN_ROW_BF16S:
        return kFp8TextPolicy;
    default:
        return ops::LinearPolicy::A16Only;
    }
}

constexpr std::size_t kMinimumLeafWorkspaceBytes = 1;

std::size_t gdn_snapshot_workspace_bytes(const Tensor& hidden,
                                         const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch,
                            width, width));
    }
    const Weight& parent =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    return std::max(
        kMinimumLeafWorkspaceBytes,
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            parent.qtype, parent.n, parent.k, text_policy(parent), batch, width, width));
}

std::size_t gdn_record_workspace_bytes(const Tensor& hidden,
                                       const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch,
                            width, width));
    }
    const Weight& parent =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    return std::max(
        kMinimumLeafWorkspaceBytes,
        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            parent.qtype, parent.n, parent.k, text_policy(parent), batch, width, width));
}

std::size_t post_mixer_workspace_bytes(QType gate_up_qtype, QType down_qtype,
                                       ops::LinearPolicy policy, std::int32_t first,
                                       std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
            gate_up_qtype, 2 * TextConfig::intermediate, TextConfig::hidden, policy, first, last));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            down_qtype, TextConfig::hidden, TextConfig::intermediate, policy, first, last));
    }
    return layout.peak_bytes(1);
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window. Early ranges limit empty producer CTAs; later ranges
    // follow measured split-policy transitions until the producer grid reaches its fixed cap.
    return graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    // Bound the final AR window E+2K at split-policy transitions until the grid reaches its cap.
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    // Target verify and MTP batch both have T=K+1 and W=E+K+1. Preserve one concrete INT8
    // implementation per range at the T=4/5/6 launch boundaries.
    if (draft_window == 3) {
        add_shifted(1029, draft_window + 1);
    } else if (draft_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(capacity - 1, ends);
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t, std::uint32_t,
                                                                  std::uint32_t) {
    return {};
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* split = std::get_if<SplitAttentionProjectionPayload>(&weights)) {
        ops::attn_input_proj(hidden, split->query_key, split->gate_value, query, gate, key, value,
                             workspace, stream);
        return;
    }
    const Weight& fused = std::get<FusedAttentionProjectionPayload>(weights).query_key_gate_value;
    ops::attn_input_proj(hidden, fused, query, gate, key, value, text_policy(fused), workspace,
                         stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear_add(attention, weight, residual, text_policy(weight), workspace, stream);
}

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor packed  = workspace.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, cols});
    ops::linear(hidden, weights.packed, packed, stream);
    Tensor query_heads = query.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor key_heads   = key.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    Tensor gate_heads  = gate.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor value_heads = value.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    ops::mtp_split_attn_in(packed, query_heads, key_heads, gate_heads, value_heads, stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena&, cudaStream_t stream) {
    ops::linear_pair(hidden, weights.key, weights.value, key, value, stream);
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena&, cudaStream_t stream) {
    ops::linear(hidden, weights.query, query, stream);
    ops::linear(hidden, weights.output_gate, gate, stream);
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_flat =
        output_gate.view({TextConfig::value_dim, static_cast<int>(hidden.ne[1])});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj(hidden, split->query_key, split->value_z, qkv, output_gate_flat,
                            workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj(hidden, fused, qkv, output_gate_flat, text_policy(fused), workspace,
                        stream);
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_snapshot_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_snapshot(hidden, split->query_key, split->value_z, conv_weight,
                                          conv_states, valid_columns, initial_slot,
                                          snapshot_base_slot, query, key, value, output_gate_view,
                                          leaf_workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_snapshot(hidden, fused, conv_weight, conv_states, valid_columns,
                                      initial_slot, snapshot_base_slot, query, key, value,
                                      output_gate_view, text_policy(fused), leaf_workspace, stream);
}

void Variant::gdn_input_projection_record(const Tensor& hidden, const GdnProjectionWeights& weights,
                                          const Tensor& conv_weight, const Tensor& conv_states,
                                          const Tensor& valid_columns, const Tensor& initial_slots,
                                          Tensor& conv_record, Tensor& query, Tensor& key,
                                          Tensor& value, Tensor& output_gate, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_record_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_record(hidden, split->query_key, split->value_z, conv_weight,
                                        conv_states, valid_columns, initial_slots, conv_record,
                                        query, key, value, output_gate_view, leaf_workspace,
                                        stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_record(hidden, fused, conv_weight, conv_states, valid_columns,
                                    initial_slots, conv_record, query, key, value, output_gate_view,
                                    text_policy(fused), leaf_workspace, stream);
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    qwen3_6::TextPhase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    if (weight.qtype == QType::GGML_K) {
        ops::ggml_k_gdn_output(hidden, weight, residual, workspace, stream);
        return;
    }
    ops::linear_add(hidden, weight, residual, text_policy(weight), workspace, stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* split =
            std::get_if<SplitGdnControlProjectionPayload>(&weights.control_projection)) {
        ops::gdn_norm_gating_proj(residual, norm_weight, eps, split->a_projection,
                                  split->b_projection, weights.a_log, weights.dt_bias, workspace,
                                  hidden, g, beta, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnControlProjectionPayload>(weights.control_projection).a_b_projection;
    ops::gdn_norm_gating_proj(residual, norm_weight, eps, fused, weights.a_log, weights.dt_bias,
                              workspace, hidden, g, beta, stream);
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope        = workspace.scope();
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
    ops::linear_swiglu(hidden, weights.gate_up, activation, text_policy(weights.gate_up), workspace,
                       stream);
    ops::linear_add(activation, weights.down, residual, text_policy(weights.down), workspace,
                    stream);
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor gate_up = workspace.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, cols});
    ops::linear(hidden, weights.gate_up, gate_up, stream);
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
    ops::silu_mul(gate_up.slice(0, 0, TextConfig::intermediate),
                  gate_up.slice(0, TextConfig::intermediate, TextConfig::intermediate), activation,
                  stream);
    Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, cols});
    ops::linear(activation, weights.down, delta, stream);
    ops::residual_add(delta, residual, stream);
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t first,
                                                                       std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, last});
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
    case WeightsProfile::Qwen38GgmlK:
        return 0;
    case WeightsProfile::Qwen36Nvfp4:
        return ops::attn_input_proj_workspace_capacity_bytes(
            QType::NVFP4, 14336, TextConfig::hidden, kNvfp4TextPolicy, first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::attn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, 14336, TextConfig::hidden, kFp8TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen38GgmlK:
        return ops::linear_add_workspace_capacity_bytes(
            QType::GGML_K, TextConfig::hidden, TextConfig::value_dim,
            ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                        TextConfig::query_size,
                                                        ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::NVFP4, TextConfig::hidden,
                                                        TextConfig::query_size, kNvfp4TextPolicy,
                                                        first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S,
                                                        TextConfig::hidden, TextConfig::query_size,
                                                        kFp8TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen38GgmlK:
        return 0;
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return 0;
    case WeightsProfile::Qwen36Nvfp4:
        return ops::gdn_input_proj_workspace_capacity_bytes(QType::NVFP4, 16384, TextConfig::hidden,
                                                            kNvfp4TextPolicy, first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::gdn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, 16384, TextConfig::hidden, kFp8TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen38GgmlK:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            QType::GGML_K, 16384, TextConfig::hidden,
                            ops::LinearPolicy::A16Only, batch_size, first, last));
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                            batch_size, first, last));
    case WeightsProfile::Qwen36Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch_size,
                            first, last));
    case WeightsProfile::Qwen38Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            QType::FP8_E4M3FN_ROW_BF16S, 16384, TextConfig::hidden, kFp8TextPolicy,
                            batch_size, first, last));
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen38GgmlK:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            QType::GGML_K, 16384, TextConfig::hidden,
                            ops::LinearPolicy::A16Only, batch_size, first, last));
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                            batch_size, first, last));
    case WeightsProfile::Qwen36Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch_size,
                            first, last));
    case WeightsProfile::Qwen38Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            QType::FP8_E4M3FN_ROW_BF16S, 16384, TextConfig::hidden, kFp8TextPolicy,
                            batch_size, first, last));
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                    qwen3_6::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen38GgmlK:
        return 0;
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                        TextConfig::value_dim,
                                                        ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(
            QType::NVFP4, TextConfig::hidden, TextConfig::value_dim, kNvfp4TextPolicy, first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S,
                                                        TextConfig::hidden, TextConfig::value_dim,
                                                        kFp8TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first,
                                                                          std::int32_t last) {
    return ops::gdn_norm_gating_proj_workspace_capacity_bytes(TextConfig::gdn_value_heads,
                                                              TextConfig::hidden, first, last);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         qwen3_6::TextPhase, std::int32_t first,
                                                         std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen38GgmlK:
        return post_mixer_workspace_bytes(QType::GGML_K, QType::GGML_K,
                                          ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return post_mixer_workspace_bytes(QType::Q4G64_F16S, QType::Q5G64_F16S,
                                          ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36Nvfp4:
        return post_mixer_workspace_bytes(QType::NVFP4, QType::NVFP4, kNvfp4TextPolicy, first,
                                          last);
    case WeightsProfile::Qwen38Nvfp4: {
        const std::size_t nvfp4 =
            post_mixer_workspace_bytes(QType::NVFP4, QType::NVFP4, kNvfp4TextPolicy, first, last);
        const std::size_t fp8 = post_mixer_workspace_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, QType::FP8_E4M3FN_ROW_BF16S, kFp8TextPolicy, first, last);
        return std::max(nvfp4, fp8);
    }
    }
    throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
}

// --- tp == 2 split leaves ----------------------------------------------------------------------
//
// Each of these is the tp1 leaf above with every per-rank argument taken as an array and the
// matching `*_column_parallel` / `*_row_parallel` Op called ONCE for both ranks. The weight
// variant (fused vs split storage) is resolved from rank 0 and required to agree on rank 1: both
// ranks bind the same artifact objects through the same profile, so a disagreement is a loader
// bug, not a supported configuration.

namespace {

template <class Payload, class Weights>
std::array<const Payload*, 2> require_same_alternative(const std::array<const Weights*, 2>& w,
                                                       const char* label) {
    const auto* a = std::get_if<Payload>(w[0]);
    const auto* b = std::get_if<Payload>(w[1]);
    if (a == nullptr || b == nullptr) {
        throw std::logic_error(std::string(label) + ": tp2 ranks disagree on weight storage form");
    }
    return {a, b};
}

std::array<Weight, 2> pair_of(const Weight& a, const Weight& b) { return {a, b}; }

// Current-device save/restore around a per-rank kernel issue. Only ONE tp2 leaf below needs it --
// `mtp_attention_projection`, whose second stage (`ops::mtp_split_attn_in`) is a purely
// elementwise remap with no cross-device form and no reason for one, so it is issued once per
// rank on that rank's own stream. Every other split leaf delegates wholly to an Op that owns its
// own per-rank issue. This mirrors ops::detail::for_each_rank rather than including an Op's
// private header, exactly as the family schedule does (text_context_impl.h).
class CurrentDevice {
public:
    CurrentDevice() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~CurrentDevice() { (void)cudaSetDevice(previous_); }

    CurrentDevice(const CurrentDevice&)            = delete;
    CurrentDevice& operator=(const CurrentDevice&) = delete;

private:
    int previous_ = 0;
};

template <class Body>
void for_each_rank(const ExecutionContext& ec, Body&& body) {
    const CurrentDevice restore;
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));
        body(rank);
    }
}

// Both registered MTP codecs use A16 projections without transient Linear storage.
void require_mtp_shard(const Weight& a, const Weight& b, const char* label) {
    if (a.qtype != b.qtype ||
        (a.qtype != QType::W8G32_F16S && a.qtype != QType::GGML_K)) {
        throw std::logic_error(std::string(label) +
                               ": unsupported or inconsistent tp2 MTP weight format");
    }
    if (a.k != b.k || a.n != b.n) {
        throw std::logic_error(std::string(label) + ": tp2 MTP shards disagree on shape");
    }
}

} // namespace

void Variant::attention_projection(const std::array<Tensor, 2>& hidden,
                                   const std::array<const FullAttentionProjectionWeights*, 2>& w,
                                   const std::array<Tensor, 2>& query,
                                   const std::array<Tensor, 2>& gate,
                                   const std::array<Tensor, 2>& key,
                                   const std::array<Tensor, 2>& value, qwen3_6::TextPhase,
                                   const std::array<WorkspaceArena*, 2>& workspace,
                                   const ExecutionContext& ec) {
    if (std::holds_alternative<SplitAttentionProjectionPayload>(*w[0])) {
        const auto split = require_same_alternative<SplitAttentionProjectionPayload>(
            w, "attention projection");
        ops::attn_input_proj_column_parallel(
            hidden, pair_of(split[0]->query_key, split[1]->query_key),
            pair_of(split[0]->gate_value, split[1]->gate_value), query, gate, key, value,
            workspace, ec);
        return;
    }
    const auto fused =
        require_same_alternative<FusedAttentionProjectionPayload>(w, "attention projection");
    ops::attn_input_proj_column_parallel(
        hidden, pair_of(fused[0]->query_key_gate_value, fused[1]->query_key_gate_value), query,
        gate, key, value, text_policy(fused[0]->query_key_gate_value), workspace, ec);
}

void Variant::attention_output_projection(const std::array<Tensor, 2>& attention,
                                          const std::array<Weight, 2>& weight,
                                          const std::array<Tensor, 2>& residual,
                                          const std::array<Tensor, 2>& staging, qwen3_6::TextPhase,
                                          const std::array<WorkspaceArena*, 2>& workspace,
                                          const ExecutionContext& ec, const ops::PeerEvents& ev) {
    ops::linear_add_row_parallel(attention, weight, residual, staging, text_policy(weight[0]),
                                 workspace, ec, ev);
}

void Variant::gdn_input_projection(const std::array<Tensor, 2>& hidden,
                                   const std::array<const GdnProjectionWeights*, 2>& w,
                                   const std::array<Tensor, 2>& qkv,
                                   const std::array<Tensor, 2>& output_gate, qwen3_6::TextPhase,
                                   const std::array<WorkspaceArena*, 2>& workspace,
                                   const ExecutionContext& ec) {
    // The caller holds `z` as [head_dim, value_heads, T]; the Op wants the flat [value_dim, T],
    // exactly as the tp1 leaf above does. The shard's value_dim is read off the tensor rather than
    // assumed, so this stays correct if the head split ever changes.
    const std::array<Tensor, 2> output_gate_flat = {
        output_gate[0].view({output_gate[0].ne[0] * output_gate[0].ne[1], hidden[0].ne[1]}),
        output_gate[1].view({output_gate[1].ne[0] * output_gate[1].ne[1], hidden[1].ne[1]})};
    const std::array<const GdnInputProjectionPayload*, 2> input = {&w[0]->input_projection,
                                                                   &w[1]->input_projection};
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(*input[0])) {
        const auto split =
            require_same_alternative<SplitGdnInputProjectionPayload>(input, "GDN input projection");
        ops::gdn_input_proj_column_parallel(hidden,
                                            pair_of(split[0]->query_key, split[1]->query_key),
                                            pair_of(split[0]->value_z, split[1]->value_z), qkv,
                                            output_gate_flat, ec);
        return;
    }
    const auto fused =
        require_same_alternative<FusedGdnInputProjectionPayload>(input, "GDN input projection");
    ops::gdn_input_proj_column_parallel(
        hidden, pair_of(fused[0]->query_key_value_z, fused[1]->query_key_value_z), qkv,
        output_gate_flat, text_policy(fused[0]->query_key_value_z), workspace, ec);
}

void Variant::gdn_input_projection_snapshot(
    const std::array<Tensor, 2>& hidden, const std::array<const GdnProjectionWeights*, 2>& w,
    const std::array<Tensor, 2>& conv_weight, const std::array<Tensor, 2>& conv_states,
    const std::array<Tensor, 2>& valid_columns, const std::array<Tensor, 2>& initial_slot,
    const std::array<Tensor, 2>& snapshot_base_slot, const std::array<Tensor, 2>& query,
    const std::array<Tensor, 2>& key, const std::array<Tensor, 2>& value,
    const std::array<Tensor, 2>& output_gate, qwen3_6::TextPhase,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec) {
    const std::array<const GdnInputProjectionPayload*, 2> input = {&w[0]->input_projection,
                                                                   &w[1]->input_projection};
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(*input[0])) {
        const auto split = require_same_alternative<SplitGdnInputProjectionPayload>(
            input, "GDN snapshot projection");
        ops::gdn_input_proj_conv_snapshot_column_parallel(
            hidden, pair_of(split[0]->query_key, split[1]->query_key),
            pair_of(split[0]->value_z, split[1]->value_z), conv_weight, conv_states, valid_columns,
            initial_slot, snapshot_base_slot, query, key, value, output_gate, workspace, ec);
        return;
    }
    const auto fused =
        require_same_alternative<FusedGdnInputProjectionPayload>(input, "GDN snapshot projection");
    ops::gdn_input_proj_conv_snapshot_column_parallel(
        hidden, pair_of(fused[0]->query_key_value_z, fused[1]->query_key_value_z), conv_weight,
        conv_states, valid_columns, initial_slot, snapshot_base_slot, query, key, value,
        output_gate, text_policy(fused[0]->query_key_value_z), workspace, ec);
}

void Variant::gdn_output_projection(const std::array<Tensor, 2>& hidden,
                                    const std::array<Weight, 2>& weight,
                                    const std::array<Tensor, 2>& residual,
                                    const std::array<Tensor, 2>& staging, qwen3_6::TextPhase,
                                    const std::array<WorkspaceArena*, 2>& workspace,
                                    const ExecutionContext& ec, const ops::PeerEvents& ev) {
    if (weight[0].qtype == QType::GGML_K) {
        ops::ggml_k_gdn_output(hidden, weight, residual, staging, workspace, ec, ev);
        return;
    }
    ops::linear_add_row_parallel(hidden, weight, residual, staging, text_policy(weight[0]),
                                 workspace, ec, ev);
}

void Variant::gdn_control_projection(const std::array<Tensor, 2>& hidden,
                                     const std::array<const GdnProjectionWeights*, 2>& w,
                                     const std::array<Tensor, 2>& g,
                                     const std::array<Tensor, 2>& beta,
                                     const std::array<WorkspaceArena*, 2>& workspace,
                                     const ExecutionContext& ec) {
    // The tp1 leaf fuses the input RMSNorm into the gating GEMM. There is no split form of the
    // fused kernel and no reason for one: the norm is replicated elementwise work over the
    // full-width residual, so the caller runs it per rank and this leaf takes the normalized
    // hidden directly.
    const std::array<const GdnControlProjectionPayload*, 2> control = {&w[0]->control_projection,
                                                                       &w[1]->control_projection};
    const std::array<Tensor, 2> a_log    = {w[0]->a_log, w[1]->a_log};
    const std::array<Tensor, 2> dt_bias  = {w[0]->dt_bias, w[1]->dt_bias};
    if (std::holds_alternative<SplitGdnControlProjectionPayload>(*control[0])) {
        const auto split = require_same_alternative<SplitGdnControlProjectionPayload>(
            control, "GDN control projection");
        ops::gdn_gating_proj_column_parallel(
            hidden, pair_of(split[0]->a_projection, split[1]->a_projection),
            pair_of(split[0]->b_projection, split[1]->b_projection), a_log, dt_bias, workspace, g,
            beta, ec);
        return;
    }
    const auto fused = require_same_alternative<FusedGdnControlProjectionPayload>(
        control, "GDN control projection");
    ops::gdn_gating_proj_column_parallel(
        hidden, pair_of(fused[0]->a_b_projection, fused[1]->a_b_projection), a_log, dt_bias,
        workspace, g, beta, ec);
}

void Variant::post_mixer(const std::array<Tensor, 2>& hidden,
                         const std::array<const PostMixerWeights*, 2>& w,
                         const std::array<Tensor, 2>& residual,
                         const std::array<Tensor, 2>& staging, qwen3_6::TextPhase,
                         const std::array<WorkspaceArena*, 2>& workspace,
                         const ExecutionContext& ec, const ops::PeerEvents& ev) {
    // The activation width is this rank's own gate/up shard, read off the weight rather than
    // assumed: `gate_up` is [2 * intermediate_shard, hidden], so half its rows is the shard.
    const std::int32_t shard_intermediate = w[0]->gate_up.n / 2;
    if (w[1]->gate_up.n != w[0]->gate_up.n) {
        throw std::logic_error("post_mixer: tp2 gate/up shards disagree on width");
    }
    std::array<Tensor, 2> activation{};
    std::array<WorkspaceArena::Scope, 2> scopes = {workspace[0]->scope(), workspace[1]->scope()};
    for (std::size_t rank = 0; rank < 2; ++rank) {
        activation[rank] =
            workspace[rank]->alloc(DType::BF16, {shard_intermediate, hidden[0].ne[1]});
    }
    ops::linear_swiglu_column_parallel(hidden, pair_of(w[0]->gate_up, w[1]->gate_up), activation,
                                       text_policy(w[0]->gate_up), workspace, ec);
    ops::linear_add_row_parallel(activation, pair_of(w[0]->down, w[1]->down), residual, staging,
                                 text_policy(w[0]->down), workspace, ec, ev);
}

void Variant::gdn_input_projection_record(
    const std::array<Tensor, 2>& hidden, const std::array<const GdnProjectionWeights*, 2>& w,
    const std::array<Tensor, 2>& conv_weight, const std::array<Tensor, 2>& conv_states,
    const std::array<Tensor, 2>& valid_columns, const std::array<Tensor, 2>& initial_slots,
    const std::array<Tensor, 2>& conv_record, const std::array<Tensor, 2>& query,
    const std::array<Tensor, 2>& key, const std::array<Tensor, 2>& value,
    const std::array<Tensor, 2>& output_gate, qwen3_6::TextPhase,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec) {
    // The record twin of `gdn_input_projection_snapshot` above, reached only from the speculative
    // verify round: instead of snapshotting the post-round conv state it writes a per-column
    // `conv_record` that the peer's `ops::gdn_replay_fold` later folds at
    // FoldGeometry<48, 8, 24, 5120>. Everything else -- the shard extents, the weight storage
    // form, the per-rank issue -- is the snapshot leaf's.
    const std::array<const GdnInputProjectionPayload*, 2> input = {&w[0]->input_projection,
                                                                   &w[1]->input_projection};
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(*input[0])) {
        const auto split = require_same_alternative<SplitGdnInputProjectionPayload>(
            input, "GDN record projection");
        ops::gdn_input_proj_conv_record_column_parallel(
            hidden, pair_of(split[0]->query_key, split[1]->query_key),
            pair_of(split[0]->value_z, split[1]->value_z), conv_weight, conv_states, valid_columns,
            initial_slots, conv_record, query, key, value, output_gate, workspace, ec);
        return;
    }
    const auto fused =
        require_same_alternative<FusedGdnInputProjectionPayload>(input, "GDN record projection");
    ops::gdn_input_proj_conv_record_column_parallel(
        hidden, pair_of(fused[0]->query_key_value_z, fused[1]->query_key_value_z), conv_weight,
        conv_states, valid_columns, initial_slots, conv_record, query, key, value, output_gate,
        text_policy(fused[0]->query_key_value_z), workspace, ec);
}

void Variant::mtp_attention_projection(
    const std::array<Tensor, 2>& hidden,
    const std::array<const MtpAttentionProjectionWeights*, 2>& w,
    const std::array<Tensor, 2>& query, const std::array<Tensor, 2>& gate,
    const std::array<Tensor, 2>& key, const std::array<Tensor, 2>& value,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec) {
    require_mtp_shard(w[0]->packed, w[1]->packed, "MTP attention projection");
    const std::int32_t columns   = hidden[0].ne[1];
    const std::int32_t attn_rows = w[0]->packed.n;
    if (hidden[1].ne[1] != columns) {
        throw std::logic_error("MTP attention projection: tp2 ranks disagree on token count");
    }
    std::array<Tensor, 2> packed{};
    std::array<WorkspaceArena::Scope, 2> scopes = {workspace[0]->scope(), workspace[1]->scope()};
    for (std::size_t rank = 0; rank < 2; ++rank) {
        packed[rank] = workspace[rank]->alloc(DType::BF16, {attn_rows, columns});
    }
    ops::linear_column_parallel(hidden, pair_of(w[0]->packed, w[1]->packed), packed, ec);
    // `mtp_split_attn_in` selects its section boundaries from the packed row count alone, so the
    // shard geometry (rows [0,3072) Q | [3072,3584) K | [3584,6656) Gate | [6656,7168) V) needs
    // no rank argument. The resulting head indices are DEVICE-LOCAL, which is exactly what the
    // head-local 12|2 gqa_attention downstream consumes.
    for_each_rank(ec, [&](int rank) {
        const auto r          = static_cast<std::size_t>(rank);
        const std::int32_t qh = query[r].numel() / (TextConfig::head_dim * columns);
        const std::int32_t kh = key[r].numel() / (TextConfig::head_dim * columns);
        Tensor query_heads    = query[r].view({TextConfig::head_dim, qh, columns});
        Tensor gate_heads     = gate[r].view({TextConfig::head_dim, qh, columns});
        Tensor key_heads      = key[r].view({TextConfig::head_dim, kh, columns});
        Tensor value_heads    = value[r].view({TextConfig::head_dim, kh, columns});
        ops::mtp_split_attn_in(packed[r], query_heads, key_heads, gate_heads, value_heads,
                               ec.dev[rank]->stream);
    });
}

void Variant::mtp_kv_projection(const std::array<Tensor, 2>& hidden,
                                const std::array<const MtpAttentionProjectionWeights*, 2>& w,
                                const std::array<Tensor, 2>& key,
                                const std::array<Tensor, 2>& value,
                                const std::array<WorkspaceArena*, 2>&, const ExecutionContext& ec) {
    // The tp1 leaf fuses these two into one `linear_pair`; the shard's key and value row views
    // are separate blocks of the same packed shard, so at tp2 they are two column-parallel calls.
    require_mtp_shard(w[0]->key, w[1]->key, "MTP key projection");
    require_mtp_shard(w[0]->value, w[1]->value, "MTP value projection");
    ops::linear_column_parallel(hidden, pair_of(w[0]->key, w[1]->key), key, ec);
    ops::linear_column_parallel(hidden, pair_of(w[0]->value, w[1]->value), value, ec);
}

void Variant::mtp_q_gate_projection(const std::array<Tensor, 2>& hidden,
                                    const std::array<const MtpAttentionProjectionWeights*, 2>& w,
                                    const std::array<Tensor, 2>& query,
                                    const std::array<Tensor, 2>& gate,
                                    const std::array<WorkspaceArena*, 2>&,
                                    const ExecutionContext& ec) {
    require_mtp_shard(w[0]->query, w[1]->query, "MTP query projection");
    require_mtp_shard(w[0]->output_gate, w[1]->output_gate, "MTP gate projection");
    ops::linear_column_parallel(hidden, pair_of(w[0]->query, w[1]->query), query, ec);
    ops::linear_column_parallel(hidden, pair_of(w[0]->output_gate, w[1]->output_gate), gate, ec);
}

void Variant::mtp_post_mixer(const std::array<Tensor, 2>& hidden,
                             const std::array<const MtpPostMixerWeights*, 2>& w,
                             const std::array<Tensor, 2>& residual,
                             const std::array<Tensor, 2>& staging,
                             const std::array<WorkspaceArena*, 2>& workspace,
                             const ExecutionContext& ec, const ops::PeerEvents& ev) {
    // The MTP post-mixer is composed exactly the way the tp1 leaf above composes it -- separate
    // `linear` / `silu_mul` / `linear` / `residual_add`, NOT the fused linear_swiglu + linear_add
    // pair the text post-mixer uses. That is not a stylistic choice: neither
    // `linear_swiglu_column_parallel` nor `linear_add_row_parallel` registers W8G32_F16S, which
    // is the format of every MTP object, and the tp1 MTP leaf already avoids both fused Ops for
    // the same reason. `tests/ops/test_mtp_split.cpp`'s Leg A proves this exact composition at
    // tp2 -- column-parallel gate_up, a shard-local silu_mul over the shard's own gate/up halves,
    // then row-parallel down plus the all-reduce.
    require_mtp_shard(w[0]->gate_up, w[1]->gate_up, "MTP post mixer gate/up");
    require_mtp_shard(w[0]->down, w[1]->down, "MTP post mixer down");
    const std::int32_t shard_intermediate = w[0]->gate_up.n / 2;
    const std::int32_t columns            = hidden[0].ne[1];
    std::array<Tensor, 2> gate_up{};
    std::array<Tensor, 2> activation{};
    std::array<Tensor, 2> delta{};
    std::array<WorkspaceArena::Scope, 2> scopes = {workspace[0]->scope(), workspace[1]->scope()};
    for (std::size_t rank = 0; rank < 2; ++rank) {
        gate_up[rank]    = workspace[rank]->alloc(DType::BF16, {w[rank]->gate_up.n, columns});
        activation[rank] = workspace[rank]->alloc(DType::BF16, {shard_intermediate, columns});
        delta[rank]      = workspace[rank]->alloc(DType::BF16, {TextConfig::hidden, columns});
    }
    ops::linear_column_parallel(hidden, pair_of(w[0]->gate_up, w[1]->gate_up), gate_up, ec);
    // Each rank's gate and up halves are its OWN shard's halves -- the ShardPlan splits gate_up
    // as two independent column blocks, so rank r holds gate rows [r*I/2 ...] and up rows in the
    // matching block, and the SiLU pairing is rank-local with nothing to communicate.
    for_each_rank(ec, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::silu_mul(gate_up[r].slice(0, 0, shard_intermediate),
                      gate_up[r].slice(0, shard_intermediate, shard_intermediate), activation[r],
                      ec.dev[rank]->stream);
    });
    ops::linear_row_parallel(activation, pair_of(w[0]->down, w[1]->down), delta, staging, ec, ev);
    // `delta` is identical on both ranks after the collective, so the residual fold is replicated
    // elementwise work and keeps the residual bit-identical across devices.
    for_each_rank(ec, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        ops::residual_add(delta[r], const_cast<Tensor&>(residual[r]), ec.dev[rank]->stream);
    });
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                             std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, last});
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    return layout.peak_bytes(1);
}

} // namespace ninfer::targets::qwen3_6_27b::detail
