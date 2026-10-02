#pragma once

#include "targets/qwen3_6_27b/impl/tp2_dflash_context.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/swa.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::targets::qwen3_6_27b::detail {

inline constexpr std::int32_t kDFlash2MaskToken = 248070;

inline void prepare_tp2_dflash_draft_inputs(
    const std::array<TP2DFlashContextState*, 2>& state,
    const std::array<const RuntimeModelView*, 2>& model,
    const std::array<Tensor, 2>& anchors, const std::array<Tensor, 2>& frontiers,
    const std::array<Tensor, 2>& valid_columns, std::int32_t batch,
    const ExecutionContext& execution) {
    if (execution.tp != 2 || batch < 1 || batch > 8 || !state[0] || !state[1] ||
        !model[0] || !model[1] || state[0]->draft_ids.ne[0] != state[1]->draft_ids.ne[0] ||
        batch > state[0]->draft_ids.ne[1] || batch > state[1]->draft_ids.ne[1]) {
        throw std::invalid_argument("TP2 DFlash2 draft inputs are invalid");
    }
    const int width = state[0]->draft_ids.ne[0];
    int previous_device = 0;
    CUDA_CHECK(cudaGetDevice(&previous_device));
    try {
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
            Tensor ids = state[rank]->draft_ids.slice(1, 0, batch);
            Tensor positions = state[rank]->draft_positions.slice(1, 0, batch);
            Tensor residual = state[rank]->draft_input.slice(2, 0, batch);
            Tensor flat_residual = residual.view({5120, width * batch});
            ops::prepare_masked_block(anchors[rank], frontiers[rank], valid_columns[rank],
                                      kDFlash2MaskToken, ids, positions,
                                      execution.dev[rank]->stream);
            ops::embedding(ids.view({width * batch}), model[rank]->token_embedding,
                           flat_residual, execution.dev[rank]->stream);
        }
    } catch (...) {
        (void)cudaSetDevice(previous_device);
        throw;
    }
    CUDA_CHECK(cudaSetDevice(previous_device));
}

struct TP2DFlashDraftWorkspaceCapacity {
    std::size_t roots;
    std::size_t ops;
};

[[nodiscard]] inline TP2DFlashDraftWorkspaceCapacity tp2_dflash_draft_workspace_capacity(
    int width, int batch, ops::SwaContextExecutionEnvelope envelope) {
    if (width < 2 || width > 16 || batch < 1 || batch > 8) {
        throw std::invalid_argument("TP2 DFlash2 draft workspace geometry is invalid");
    }
    WorkspaceLayoutBuilder layout;
    {
        auto layer = layout.scope();
        (void)layout.alloc(DType::BF16, {5120, width, batch});
        (void)layout.alloc(DType::BF16, {320, 2, width, batch});
        (void)layout.alloc(DType::BF16, {2048, width, batch});
        (void)layout.alloc(DType::BF16, {512, width, batch});
        (void)layout.alloc(DType::BF16, {512, width, batch});
        (void)layout.alloc(DType::BF16, {2048, width, batch});
        (void)layout.alloc(DType::BF16, {5120, width, batch});
        (void)layout.alloc(DType::BF16, {5120, width, batch});
        (void)layout.alloc(DType::BF16, {5120, width, batch});
        (void)layout.alloc(DType::BF16, {320, 2, width, batch});
        (void)layout.alloc(DType::BF16, {8704, width, batch});
        (void)layout.alloc(DType::BF16, {17408, width * batch});
    }
    return {
        .roots = layout.peak_bytes(256),
        .ops = std::max(ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
                            width, width, batch, batch),
                        ops::swa_workspace_capacity_bytes(envelope, width, width, batch)),
    };
}

inline void tp2_dflash_draft_backbone(
    std::array<Tensor, 2>& residual, const std::array<Tensor, 2>& positions,
    const std::array<Tensor, 2>& valid_columns, const std::array<Tensor, 2>& lanes,
    const std::array<const DFlash2Weights*, 2>& weights,
    const std::array<TP2DFlashContextState*, 2>& state,
    ops::SwaContextExecutionEnvelope envelope, std::array<Tensor, 2>& final_hidden,
    const std::array<WorkspaceArena*, 2>& workspace,
    const std::array<WorkspaceArena*, 2>& op_workspace, const ExecutionContext& execution,
    const ops::PeerEvents& events) {
    if (execution.tp != 2 || !weights[0] || !weights[1] || !state[0] || !state[1] ||
        !workspace[0] || !workspace[1] || !op_workspace[0] || !op_workspace[1] ||
        residual[0].dtype != DType::BF16 ||
        residual[1].dtype != DType::BF16 || residual[0].ne[0] != 5120 ||
        residual[1].ne[0] != 5120 || residual[0].ne[1] != residual[1].ne[1] ||
        residual[0].ne[2] != residual[1].ne[2] || residual[0].ne[1] < 2 ||
        residual[0].ne[1] > 16 || residual[0].ne[2] < 1 || residual[0].ne[2] > 8) {
        throw std::invalid_argument("TP2 DFlash2 draft geometry is invalid");
    }
    const int width = residual[0].ne[1];
    const int batch = residual[0].ne[2];
    const int tokens = width * batch;
    int previous_device = 0;
    CUDA_CHECK(cudaGetDevice(&previous_device));
    try {
        for (int layer = 0; layer < 5; ++layer) {
            auto primary_scope = workspace[0]->scope();
            auto peer_scope = workspace[1]->scope();
            std::array<Tensor, 2> attention_prepared, attention_delta, query, key, value;
            std::array<Tensor, 2> attention, projected, staging, mlp_prepared, mlp_delta;
            std::array<Tensor, 2> activated;
            for (int rank = 0; rank < 2; ++rank) {
                CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
                WorkspaceArena& scratch = *workspace[rank];
                const auto& weights_for_layer = weights[rank]->layers[layer];
                const cudaStream_t stream = execution.dev[rank]->stream;
                attention_prepared[rank] = scratch.alloc(DType::BF16, {5120, width, batch});
                attention_delta[rank] = scratch.alloc(DType::BF16, {320, 2, width, batch});
                ops::rmsnorm_dynamic_grouped_conv_prepare(
                    residual[rank], weights_for_layer.input_norm, 1.0e-6F,
                    weights_for_layer.attention_conv.base_kernel,
                    weights_for_layer.attention_conv.kernel_projection,
                    attention_prepared[rank], attention_delta[rank], *op_workspace[rank], stream);
                query[rank] = scratch.alloc(DType::BF16, {2048, width, batch});
                key[rank] = scratch.alloc(DType::BF16, {512, width, batch});
                value[rank] = scratch.alloc(DType::BF16, {512, width, batch});
                Tensor flat_query = query[rank].view({2048, tokens});
                Tensor flat_key = key[rank].view({512, tokens});
                Tensor flat_value = value[rank].view({512, tokens});
                ops::dflash2_tp2_attn_input_proj(
                    attention_prepared[rank].view({5120, tokens}),
                    weights_for_layer.query_key_value, flat_query, flat_key, flat_value, stream);
                Tensor query_heads = query[rank].view({128, 16, width, batch});
                Tensor key_heads = key[rank].view({128, 4, width, batch});
                Tensor value_heads = value[rank].view({128, 4, width, batch});
                ops::rmsnorm_rope(positions[rank], weights_for_layer.query_norm,
                                  weights_for_layer.key_norm, query_heads, key_heads, stream);
                attention[rank] = scratch.alloc(DType::BF16, {2048, width, batch});
                Tensor attention_heads = attention[rank].view({128, 16, width, batch});
                ops::swa(query_heads, key_heads, value_heads, positions[rank],
                         valid_columns[rank], lanes[rank], 0.08838834764831844F,
                         state[rank]->cache.layer_view(layer), envelope, *op_workspace[rank],
                         attention_heads, stream);
                projected[rank] = scratch.alloc(DType::BF16, {5120, width, batch});
                staging[rank] = scratch.alloc(DType::BF16, {5120, width, batch});
            }
            const std::array<Tensor, 2> attention_flat = {
                attention[0].view({2048, tokens}), attention[1].view({2048, tokens})};
            const std::array<Weight, 2> attention_weights = {
                weights[0]->layers[layer].attention_output,
                weights[1]->layers[layer].attention_output};
            const std::array<Tensor, 2> projected_flat = {
                projected[0].view({5120, tokens}), projected[1].view({5120, tokens})};
            const std::array<Tensor, 2> staging_flat = {
                staging[0].view({5120, tokens}), staging[1].view({5120, tokens})};
            ops::linear_row_parallel(attention_flat, attention_weights, projected_flat,
                                     staging_flat, execution, events);
            for (int rank = 0; rank < 2; ++rank) {
                CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
                const auto& weights_for_layer = weights[rank]->layers[layer];
                const cudaStream_t stream = execution.dev[rank]->stream;
                ops::dynamic_grouped_conv_finish_add(
                    projected[rank], weights_for_layer.attention_conv.base_kernel,
                    attention_delta[rank], residual[rank], stream);
                WorkspaceArena& scratch = *workspace[rank];
                mlp_prepared[rank] = scratch.alloc(DType::BF16, {5120, width, batch});
                mlp_delta[rank] = scratch.alloc(DType::BF16, {320, 2, width, batch});
                ops::rmsnorm_dynamic_grouped_conv_prepare(
                    residual[rank], weights_for_layer.post_attention_norm, 1.0e-6F,
                    weights_for_layer.mlp_conv.base_kernel,
                    weights_for_layer.mlp_conv.kernel_projection,
                    mlp_prepared[rank], mlp_delta[rank], *op_workspace[rank], stream);
                activated[rank] = scratch.alloc(DType::BF16, {8704, width, batch});
                Tensor activation_flat = activated[rank].view({8704, tokens});
                Tensor gate_up = scratch.alloc(DType::BF16, {17408, tokens});
                ops::linear(mlp_prepared[rank].view({5120, tokens}),
                            weights_for_layer.gate_up, gate_up, stream);
                ops::silu_mul(gate_up.slice(0, 0, 8704), gate_up.slice(0, 8704, 8704),
                              activation_flat, stream);
            }
            const std::array<Tensor, 2> activation_flat = {
                activated[0].view({8704, tokens}), activated[1].view({8704, tokens})};
            const std::array<Weight, 2> down_weights = {
                weights[0]->layers[layer].down, weights[1]->layers[layer].down};
            ops::linear_row_parallel(activation_flat, down_weights, projected_flat,
                                     staging_flat, execution, events);
            for (int rank = 0; rank < 2; ++rank) {
                CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
                const auto& weights_for_layer = weights[rank]->layers[layer];
                ops::dynamic_grouped_conv_finish_add(
                    projected[rank], weights_for_layer.mlp_conv.base_kernel,
                    mlp_delta[rank], residual[rank],
                    execution.dev[rank]->stream);
            }
        }
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
            ops::rmsnorm(residual[rank], weights[rank]->final_norm, 1.0e-6F, false,
                         final_hidden[rank], execution.dev[rank]->stream);
        }
    } catch (...) {
        (void)cudaSetDevice(previous_device);
        throw;
    }
    CUDA_CHECK(cudaSetDevice(previous_device));
}

} // namespace ninfer::targets::qwen3_6_27b::detail
