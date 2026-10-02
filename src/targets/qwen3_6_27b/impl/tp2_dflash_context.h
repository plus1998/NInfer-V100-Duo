#pragma once

#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6/impl/runtime/tp2_feature_sink.h"

#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/context_kv_materialize.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scalar.h"
#include <ninfer/targets/qwen3_6/round_state.h>

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::targets::qwen3_6_27b::detail {

struct TP2DFlashContextLayout {
    CyclicKVCacheLayout cache;
    CyclicKVCacheLayout rewrite_checkpoint;
    TensorRegion prefill_features;
    TensorRegion prefill_positions;
    TensorRegion prefill_count;
    TensorRegion prefill_slot;
    TensorRegion prefill_projected;
    TensorRegion prefill_staging;
    TensorRegion prefill_normalized;
    LayoutRegion prefill_workspace;
    TensorRegion pending_features;
    TensorRegion pending_positions;
    TensorRegion pending_counts;
    TensorRegion pending_slots;
    TensorRegion compact_features;
    TensorRegion compact_positions;
    TensorRegion compact_projected;
    TensorRegion compact_staging;
    TensorRegion compact_normalized;
    LayoutRegion compact_workspace;
    TensorRegion draft_input;
    TensorRegion draft_ids;
    TensorRegion draft_hidden;
    TensorRegion draft_positions;
    TensorRegion draft_valid_columns;
    TensorRegion draft_lanes;
    TensorRegion draft_tokens;
    TensorRegion proposal_ids;
    TensorRegion proposal_q;

    [[nodiscard]] static TP2DFlashContextLayout plan(LayoutBuilder& builder,
                                                     std::int32_t prefill_chunk,
                                                     std::int32_t max_concurrency,
                                                     std::int32_t verify_width) {
        if (prefill_chunk <= 0 || max_concurrency <= 0 || max_concurrency > 8 ||
            verify_width < 2 || verify_width > 16) {
            throw std::invalid_argument("TP2 DFlash2 context layout geometry is invalid");
        }
        TP2DFlashContextLayout layout;
        layout.cache = plan_cyclic_kv_cache(builder, 5, 2048, 4, 128, max_concurrency,
                                            DType::FP16);
        layout.rewrite_checkpoint = plan_cyclic_kv_cache(builder, 5, 2048, 4, 128,
                                                         max_concurrency, DType::FP16);
        layout.prefill_features = builder.add_tensor(DType::BF16, {12800, prefill_chunk}, 256,
                                                     "DFlash2 TP2 prefill features");
        layout.prefill_positions = builder.add_tensor(DType::I32, {prefill_chunk}, 256,
                                                      "DFlash2 TP2 prefill positions");
        layout.prefill_count = builder.add_tensor(DType::I32, {1}, 256,
                                                  "DFlash2 TP2 prefill count");
        layout.prefill_slot = builder.add_tensor(DType::I32, {1}, 256,
                                                 "DFlash2 TP2 prefill slot");
        layout.prefill_projected = builder.add_tensor(DType::BF16, {5120, prefill_chunk}, 256,
                                                      "DFlash2 TP2 prefill projected");
        layout.prefill_staging = builder.add_tensor(DType::BF16, {5120, prefill_chunk}, 256,
                                                    "DFlash2 TP2 prefill allreduce staging");
        layout.prefill_normalized = builder.add_tensor(DType::BF16, {5120, prefill_chunk}, 256,
                                                       "DFlash2 TP2 prefill normalized");
        layout.prefill_workspace = builder.add(
            ops::context_kv_materialize_workspace_capacity_bytes(1, 1,
                                                                   std::min(prefill_chunk, 2048)),
            256, "DFlash2 TP2 prefill context workspace");
        layout.pending_features = builder.add_tensor(
            DType::BF16, {12800, verify_width, max_concurrency}, 256,
            "DFlash2 TP2 pending target features");
        layout.pending_positions = builder.add_tensor(
            DType::I32, {verify_width, max_concurrency}, 256,
            "DFlash2 TP2 pending target positions");
        layout.pending_counts = builder.add_tensor(DType::I32, {max_concurrency}, 256,
                                                   "DFlash2 TP2 committed counts");
        layout.pending_slots = builder.add_tensor(DType::I32, {max_concurrency}, 256,
                                                  "DFlash2 TP2 committed slots");
        layout.compact_features = builder.add_tensor(
            DType::BF16, {12800, verify_width, max_concurrency}, 256,
            "DFlash2 TP2 compact pending features");
        layout.compact_positions = builder.add_tensor(
            DType::I32, {verify_width, max_concurrency}, 256,
            "DFlash2 TP2 compact pending positions");
        layout.compact_projected = builder.add_tensor(
            DType::BF16, {5120, verify_width, max_concurrency}, 256,
            "DFlash2 TP2 compact projected context");
        layout.compact_staging = builder.add_tensor(
            DType::BF16, {5120, verify_width, max_concurrency}, 256,
            "DFlash2 TP2 compact allreduce staging");
        layout.compact_normalized = builder.add_tensor(
            DType::BF16, {5120, verify_width, max_concurrency}, 256,
            "DFlash2 TP2 compact normalized context");
        layout.compact_workspace = builder.add(
            ops::context_kv_materialize_workspace_capacity_bytes(max_concurrency, 1,
                                                                   verify_width),
            256, "DFlash2 TP2 compact context workspace");
        layout.draft_input = builder.add_tensor(DType::BF16, {5120, verify_width, max_concurrency},
                                                256, "DFlash2 TP2 draft input");
        layout.draft_ids = builder.add_tensor(DType::I32, {verify_width, max_concurrency},
                                              256, "DFlash2 TP2 anchor and mask ids");
        layout.draft_hidden = builder.add_tensor(DType::BF16, {5120, verify_width, max_concurrency},
                                                 256, "DFlash2 TP2 draft output");
        layout.draft_positions = builder.add_tensor(DType::I32, {verify_width, max_concurrency},
                                                    256, "DFlash2 TP2 draft positions");
        layout.draft_valid_columns = builder.add_tensor(DType::I32, {max_concurrency}, 256,
                                                        "DFlash2 TP2 draft valid columns");
        layout.draft_lanes = builder.add_tensor(DType::I32, {max_concurrency}, 256,
                                                "DFlash2 TP2 draft lanes");
        layout.draft_tokens = builder.add_tensor(DType::I32, {verify_width - 1, max_concurrency},
                                                 256, "DFlash2 TP2 proposed tokens");
        layout.proposal_ids = builder.add_tensor(DType::I32,
                                                 {16, verify_width - 1, max_concurrency}, 256,
                                                 "DFlash2 TP2 proposal ids");
        layout.proposal_q = builder.add_tensor(DType::FP32,
                                               {16, verify_width - 1, max_concurrency}, 256,
                                               "DFlash2 TP2 proposal probabilities");
        return layout;
    }
};

struct TP2DFlashContextState {
    CyclicKVCache cache;
    CyclicKVCache rewrite_checkpoint;
    Tensor prefill_features;
    Tensor prefill_positions;
    Tensor prefill_count;
    Tensor prefill_slot;
    Tensor prefill_projected;
    Tensor prefill_staging;
    Tensor prefill_normalized;
    WorkspaceArena prefill_workspace;
    Tensor pending_features;
    Tensor pending_positions;
    Tensor pending_counts;
    Tensor pending_slots;
    Tensor compact_features;
    Tensor compact_positions;
    Tensor compact_projected;
    Tensor compact_staging;
    Tensor compact_normalized;
    WorkspaceArena compact_workspace;
    Tensor draft_input;
    Tensor draft_ids;
    Tensor draft_hidden;
    Tensor draft_positions;
    Tensor draft_valid_columns;
    Tensor draft_lanes;
    Tensor draft_tokens;
    Tensor proposal_ids;
    Tensor proposal_q;

    TP2DFlashContextState(DeviceSpan backing, const TP2DFlashContextLayout& layout)
        : cache(backing, layout.cache),
          rewrite_checkpoint(backing, layout.rewrite_checkpoint),
          prefill_features(layout.prefill_features.bind(backing)),
          prefill_positions(layout.prefill_positions.bind(backing)),
          prefill_count(layout.prefill_count.bind(backing)),
          prefill_slot(layout.prefill_slot.bind(backing)),
          prefill_projected(layout.prefill_projected.bind(backing)),
          prefill_staging(layout.prefill_staging.bind(backing)),
          prefill_normalized(layout.prefill_normalized.bind(backing)),
          prefill_workspace(layout.prefill_workspace.bind(backing)),
          pending_features(layout.pending_features.bind(backing)),
          pending_positions(layout.pending_positions.bind(backing)),
          pending_counts(layout.pending_counts.bind(backing)),
          pending_slots(layout.pending_slots.bind(backing)),
          compact_features(layout.compact_features.bind(backing)),
          compact_positions(layout.compact_positions.bind(backing)),
          compact_projected(layout.compact_projected.bind(backing)),
          compact_staging(layout.compact_staging.bind(backing)),
          compact_normalized(layout.compact_normalized.bind(backing)),
          compact_workspace(layout.compact_workspace.bind(backing)),
          draft_input(layout.draft_input.bind(backing)),
          draft_ids(layout.draft_ids.bind(backing)),
          draft_hidden(layout.draft_hidden.bind(backing)),
          draft_positions(layout.draft_positions.bind(backing)),
          draft_valid_columns(layout.draft_valid_columns.bind(backing)),
          draft_lanes(layout.draft_lanes.bind(backing)),
          draft_tokens(layout.draft_tokens.bind(backing)),
          proposal_ids(layout.proposal_ids.bind(backing)),
          proposal_q(layout.proposal_q.bind(backing)) {
        if (cache.layer_count() != 5 || cache.capacity() != 2048 ||
            cache.num_kv_heads() != 4 || cache.head_dim() != 128 ||
            cache.layer_view(0).v.dtype != DType::FP16 ||
            rewrite_checkpoint.layer_count() != 5 ||
            rewrite_checkpoint.capacity() != 2048 ||
            rewrite_checkpoint.lane_capacity() != cache.lane_capacity() ||
            rewrite_checkpoint.layer_view(0).v.dtype != DType::FP16 ||
            prefill_features.dtype != DType::BF16 || prefill_features.ne[0] != 12800 ||
            prefill_positions.dtype != DType::I32 ||
            prefill_positions.ne[0] != prefill_features.ne[1] ||
            prefill_count.dtype != DType::I32 || prefill_count.ne[0] != 1 ||
            prefill_slot.dtype != DType::I32 || prefill_slot.ne[0] != 1 ||
            prefill_projected.dtype != DType::BF16 || prefill_projected.ne[0] != 5120 ||
            prefill_projected.ne[1] != prefill_features.ne[1] ||
            prefill_staging.dtype != DType::BF16 || prefill_staging.ne[0] != 5120 ||
            prefill_staging.ne[1] != prefill_features.ne[1] ||
            prefill_normalized.dtype != DType::BF16 || prefill_normalized.ne[0] != 5120 ||
            prefill_normalized.ne[1] != prefill_features.ne[1] ||
            pending_features.dtype != DType::BF16 || pending_features.ne[0] != 12800 ||
            pending_features.ne[1] < 2 || pending_features.ne[1] > 16 ||
            pending_features.ne[2] != cache.lane_capacity() ||
            pending_positions.dtype != DType::I32 ||
            pending_positions.ne[0] != pending_features.ne[1] ||
            pending_positions.ne[1] != pending_features.ne[2] ||
            pending_counts.dtype != DType::I32 ||
            pending_counts.ne[0] != pending_features.ne[2] ||
            pending_slots.dtype != DType::I32 ||
            pending_slots.ne[0] != pending_features.ne[2] ||
            compact_features.dtype != DType::BF16 || compact_features.ne[0] != 12800 ||
            compact_features.ne[1] != pending_features.ne[1] ||
            compact_features.ne[2] != pending_features.ne[2] ||
            compact_positions.dtype != DType::I32 ||
            compact_positions.ne[0] != pending_positions.ne[0] ||
            compact_positions.ne[1] != pending_positions.ne[1] ||
            compact_projected.dtype != DType::BF16 || compact_projected.ne[0] != 5120 ||
            compact_projected.ne[1] != pending_features.ne[1] ||
            compact_projected.ne[2] != pending_features.ne[2] ||
            compact_staging.dtype != DType::BF16 || compact_staging.ne[0] != 5120 ||
            compact_staging.ne[1] != pending_features.ne[1] ||
            compact_staging.ne[2] != pending_features.ne[2] ||
            compact_normalized.dtype != DType::BF16 || compact_normalized.ne[0] != 5120 ||
            compact_normalized.ne[1] != pending_features.ne[1] ||
            compact_normalized.ne[2] != pending_features.ne[2] ||
            draft_input.dtype != DType::BF16 || draft_input.ne[0] != 5120 ||
            draft_input.ne[1] != pending_features.ne[1] ||
            draft_input.ne[2] != pending_features.ne[2] ||
            draft_ids.dtype != DType::I32 || draft_ids.ne[0] != draft_input.ne[1] ||
            draft_ids.ne[1] != draft_input.ne[2] ||
            draft_hidden.dtype != DType::BF16 ||
            !std::equal(std::begin(draft_hidden.ne), std::end(draft_hidden.ne),
                        std::begin(draft_input.ne)) ||
            draft_positions.dtype != DType::I32 ||
            draft_positions.ne[0] != draft_input.ne[1] ||
            draft_positions.ne[1] != draft_input.ne[2] ||
            draft_valid_columns.dtype != DType::I32 ||
            draft_valid_columns.ne[0] != draft_input.ne[2] ||
            draft_lanes.dtype != DType::I32 || draft_lanes.ne[0] != draft_input.ne[2] ||
            draft_tokens.dtype != DType::I32 || draft_tokens.ne[0] != draft_input.ne[1] - 1 ||
            draft_tokens.ne[1] != draft_input.ne[2] ||
            proposal_ids.dtype != DType::I32 || proposal_ids.ne[0] != 16 ||
            proposal_ids.ne[1] != draft_tokens.ne[0] ||
            proposal_ids.ne[2] != draft_tokens.ne[1] ||
            proposal_q.dtype != DType::FP32 ||
            !std::equal(std::begin(proposal_q.ne), std::end(proposal_q.ne),
                        std::begin(proposal_ids.ne))) {
            throw std::invalid_argument("TP2 DFlash2 persistent state layout is invalid");
        }
    }

    [[nodiscard]] std::array<CyclicKVCacheLayerView, 5> layers() const {
        return {cache.layer_view(0), cache.layer_view(1), cache.layer_view(2),
                cache.layer_view(3), cache.layer_view(4)};
    }

    void save_checkpoint(std::int32_t lane, cudaStream_t stream) {
        rewrite_checkpoint.copy_lane_from(cache, lane, stream);
    }

    void restore_checkpoint(std::int32_t lane, cudaStream_t stream) {
        cache.copy_lane_from(rewrite_checkpoint, lane, stream);
    }
};

struct TP2DFlashContextBuffers {
    std::array<Tensor, 2> projected;
    std::array<Tensor, 2> staging;
    std::array<Tensor, 2> normalized;
};

inline void materialize_tp2_dflash_context(
    const std::array<Tensor, 2>& features, const std::array<Tensor, 2>& positions,
    const std::array<Tensor, 2>& counts, const std::array<Tensor, 2>& state_slots,
    const std::array<const DFlash2Weights*, 2>& weights,
    const std::array<std::array<CyclicKVCacheLayerView, 5>, 2>& caches,
    TP2DFlashContextBuffers& buffers, ops::ContextKVMaterializeExecutionEnvelope envelope,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& execution,
    const ops::PeerEvents& events) {
    if (execution.tp != 2 || !weights[0] || !weights[1] || !workspace[0] || !workspace[1] ||
        features[0].ne[0] != 12800 || features[1].ne[0] != 12800 ||
        features[0].ne[1] != features[1].ne[1] || features[0].ne[1] <= 0) {
        throw std::invalid_argument("TP2 DFlash2 context input is invalid");
    }

    const std::array<Weight, 2> projection_weights = {weights[0]->feature_projection,
                                                       weights[1]->feature_projection};
    const std::array<Tensor, 2> flat_features = {
        features[0].view({12800, features[0].ne[1] * features[0].ne[2]}),
        features[1].view({12800, features[1].ne[1] * features[1].ne[2]})};
    const std::array<Tensor, 2> flat_projected = {
        buffers.projected[0].view({5120, flat_features[0].ne[1]}),
        buffers.projected[1].view({5120, flat_features[1].ne[1]})};
    const std::array<Tensor, 2> flat_staging = {
        buffers.staging[0].view({5120, flat_features[0].ne[1]}),
        buffers.staging[1].view({5120, flat_features[1].ne[1]})};
    ops::linear_row_parallel(flat_features, projection_weights, flat_projected, flat_staging,
                             execution, events);

    int previous_device = 0;
    CUDA_CHECK(cudaGetDevice(&previous_device));
    try {
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
            const cudaStream_t stream = execution.dev[rank]->stream;
            ops::rmsnorm(buffers.projected[rank], weights[rank]->context_norm, 1.0e-6F, false,
                         buffers.normalized[rank], stream);
            std::array<ops::ContextKVMaterializeLayerView, 5> layers;
            for (std::size_t layer = 0; layer < layers.size(); ++layer) {
                const auto& layer_weights = weights[rank]->layers[layer];
                layers[layer] = {layer_weights.key, layer_weights.value, layer_weights.key_norm,
                                 caches[rank][layer]};
            }
            ops::context_kv_materialize(buffers.normalized[rank], positions[rank], counts[rank],
                                        state_slots[rank], layers, envelope, *workspace[rank],
                                        stream);
        }
    } catch (...) {
        (void)cudaSetDevice(previous_device);
        throw;
    }
    CUDA_CHECK(cudaSetDevice(previous_device));
}

inline void catch_up_tp2_dflash_context(
    const std::array<TP2DFlashContextState*, 2>& state,
    const std::array<const DFlash2Weights*, 2>& weights,
    const std::array<qwen3_6::DFlash2DecodeState*, 2>& frames,
    const ExecutionContext& execution, const ops::PeerEvents& events, int batch) {
    if (execution.tp != 2 || !state[0] || !state[1] || !frames[0] || !frames[1] ||
        !weights[0] || !weights[1] || batch < 1 ||
        batch > state[0]->pending_features.ne[2] ||
        batch > state[1]->pending_features.ne[2]) {
        throw std::invalid_argument("DFlash2 TP2 context catch-up geometry is invalid");
    }
    const int width = state[0]->pending_features.ne[1];
    if (state[1]->pending_features.ne[1] != width) {
        throw std::invalid_argument("DFlash2 TP2 context catch-up widths disagree");
    }
    TP2DFlashContextBuffers buffers;
    std::array<Tensor, 2> features, positions, counts, slots;
    std::array<WorkspaceArena*, 2> workspace;
    std::array<std::array<CyclicKVCacheLayerView, 5>, 2> caches;
    int previous_device = 0;
    CUDA_CHECK(cudaGetDevice(&previous_device));
    try {
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
            auto& local = *state[rank];
            features[rank] = local.compact_features.slice(2, 0, batch);
            positions[rank] = local.compact_positions.slice(1, 0, batch);
            counts[rank] = local.pending_counts.slice(0, 0, batch);
            slots[rank] = frames[rank]->lanes.slice(0, 0, batch);
            ops::prepare_ragged_prefix(
                local.pending_features, slots[rank],
                frames[rank]->context_frontiers.slice(0, 0, batch),
                frames[rank]->execution_frontiers.slice(0, 0, batch),
                features[rank], positions[rank], counts[rank],
                execution.dev[rank]->stream);
            buffers.projected[rank] = local.compact_projected.slice(2, 0, batch);
            buffers.staging[rank] = local.compact_staging.slice(2, 0, batch);
            buffers.normalized[rank] = local.compact_normalized.slice(2, 0, batch);
            workspace[rank] = &local.compact_workspace;
            caches[rank] = local.layers();
        }
        materialize_tp2_dflash_context(features, positions, counts, slots, weights, caches,
                                       buffers, {0, static_cast<std::uint32_t>(width)},
                                       workspace, execution, events);
    } catch (...) {
        (void)cudaSetDevice(previous_device);
        throw;
    }
    CUDA_CHECK(cudaSetDevice(previous_device));
}

inline constexpr std::array<int, 5> kDFlash2TargetFeatureLayers{5, 19, 33, 47, 61};

inline qwen3_6::detail::TP2FeatureSink make_tp2_dflash_verify_sink(
    const std::array<TP2DFlashContextState*, 2>& state, int batch) {
    if (!state[0] || !state[1] || batch < 1 || batch > state[0]->pending_features.ne[2] ||
        batch > state[1]->pending_features.ne[2] ||
        state[0]->pending_features.ne[1] != state[1]->pending_features.ne[1]) {
        throw std::invalid_argument("TP2 DFlash2 verify capture geometry is invalid");
    }
    const int width = state[0]->pending_features.ne[1];
    return {
        .features = {
            Tensor(state[0]->pending_features.data, DType::BF16, {12800, width * batch}),
            Tensor(state[1]->pending_features.data, DType::BF16, {12800, width * batch})},
        .positions = {
            Tensor(state[0]->pending_positions.data, DType::I32, {width * batch}),
            Tensor(state[1]->pending_positions.data, DType::I32, {width * batch})},
        .layers = kDFlash2TargetFeatureLayers,
    };
}

inline void materialize_tp2_dflash_pending(
    const qwen3_6::detail::TP2FeatureSink& capture,
    const std::array<TP2DFlashContextState*, 2>& state,
    const std::array<const DFlash2Weights*, 2>& weights,
    TP2DFlashContextBuffers& buffers, const std::array<WorkspaceArena*, 2>& workspace,
    const ExecutionContext& execution, const ops::PeerEvents& events, int batch) {
    capture.require_complete();
    if (!state[0] || !state[1] || batch < 1 ||
        batch > state[0]->pending_features.ne[2] ||
        batch > state[1]->pending_features.ne[2] ||
        state[0]->pending_features.ne[1] != state[1]->pending_features.ne[1]) {
        throw std::invalid_argument("TP2 DFlash2 pending batch is invalid");
    }
    const int width = state[0]->pending_features.ne[1];
    if (capture.active_tokens != width * batch ||
        capture.features[0].data != state[0]->pending_features.data ||
        capture.features[1].data != state[1]->pending_features.data ||
        capture.positions[0].data != state[0]->pending_positions.data ||
        capture.positions[1].data != state[1]->pending_positions.data) {
        throw std::invalid_argument("TP2 DFlash2 pending capture does not match state");
    }
    std::array<Tensor, 2> features, positions, counts, slots;
    std::array<std::array<CyclicKVCacheLayerView, 5>, 2> caches;
    for (int rank = 0; rank < 2; ++rank) {
        features[rank] = state[rank]->pending_features.slice(2, 0, batch);
        positions[rank] = state[rank]->pending_positions.slice(1, 0, batch);
        counts[rank] = state[rank]->pending_counts.slice(0, 0, batch);
        slots[rank] = state[rank]->pending_slots.slice(0, 0, batch);
        caches[rank] = state[rank]->layers();
    }
    materialize_tp2_dflash_context(features, positions, counts, slots, weights, caches,
                                   buffers, {0, static_cast<std::uint32_t>(width)},
                                   workspace, execution, events);
}

inline qwen3_6::detail::TP2FeatureSink make_tp2_dflash_prefill_sink(
    const std::array<TP2DFlashContextState*, 2>& state,
    const std::array<const DFlash2Weights*, 2>& weights, TP2DFlashContextBuffers& buffers,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& execution,
    const ops::PeerEvents& events, std::int32_t lane) {
    if (!state[0] || !state[1] || lane < 0 ||
        lane >= state[0]->cache.lane_capacity() ||
        lane >= state[1]->cache.lane_capacity()) {
        throw std::invalid_argument("TP2 DFlash2 prefill lane is invalid");
    }
    return {
        .features = {state[0]->prefill_features, state[1]->prefill_features},
        .positions = {state[0]->prefill_positions, state[1]->prefill_positions},
        .layers = kDFlash2TargetFeatureLayers,
        .consume_prefill = [state, weights, buffers, workspace, &execution, &events, lane](
                               const std::array<Tensor, 2>& features,
                               const std::array<Tensor, 2>& positions, bool rewrite_checkpoint) {
            int previous_device = 0;
            CUDA_CHECK(cudaGetDevice(&previous_device));
            try {
                const int tokens = features[0].ne[1];
                if (tokens <= 0 || features[1].ne[1] != tokens ||
                    positions[0].ne[0] != tokens || positions[1].ne[0] != tokens) {
                    throw std::invalid_argument("TP2 DFlash2 prefill feature count is invalid");
                }
                const int width = std::min(tokens, 2048);
                const int offset = tokens - width;
                std::array<Tensor, 2> feature_chunk, position_chunk, count_chunk, slot_chunk;
                TP2DFlashContextBuffers chunk_buffers;
                std::array<std::array<CyclicKVCacheLayerView, 5>, 2> caches;
                for (int rank = 0; rank < 2; ++rank) {
                    CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
                    const cudaStream_t stream = execution.dev[rank]->stream;
                    ops::set_i32_scalar(state[rank]->prefill_count, width, stream);
                    ops::set_i32_scalar(state[rank]->prefill_slot, lane, stream);
                    feature_chunk[rank] = features[rank].slice(1, offset, width)
                                              .view({12800, width, 1});
                    position_chunk[rank] = positions[rank].slice(0, offset, width)
                                               .view({width, 1});
                    count_chunk[rank] = state[rank]->prefill_count;
                    slot_chunk[rank] = state[rank]->prefill_slot;
                    chunk_buffers.projected[rank] =
                        buffers.projected[rank].slice(1, 0, width).view({5120, width, 1});
                    chunk_buffers.staging[rank] =
                        buffers.staging[rank].slice(1, 0, width).view({5120, width, 1});
                    chunk_buffers.normalized[rank] =
                        buffers.normalized[rank].slice(1, 0, width).view({5120, width, 1});
                    caches[rank] = state[rank]->layers();
                }
                materialize_tp2_dflash_context(feature_chunk, position_chunk, count_chunk,
                                               slot_chunk, weights, caches, chunk_buffers,
                                               {static_cast<std::uint32_t>(width),
                                                static_cast<std::uint32_t>(width)},
                                               workspace, execution, events);
                if (rewrite_checkpoint) {
                    for (int rank = 0; rank < 2; ++rank) {
                        CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
                        state[rank]->save_checkpoint(lane, execution.dev[rank]->stream);
                    }
                }
            } catch (...) {
                (void)cudaSetDevice(previous_device);
                throw;
            }
            CUDA_CHECK(cudaSetDevice(previous_device));
        },
    };
}

} // namespace ninfer::targets::qwen3_6_27b::detail
