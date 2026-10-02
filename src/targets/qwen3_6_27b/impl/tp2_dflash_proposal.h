#pragma once

#include "targets/qwen3_6_27b/impl/tp2_dflash_draft.h"

#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/linear_topk.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::targets::qwen3_6_27b::detail {

struct TP2DFlashProposalCapacity {
    std::array<std::size_t, 2> roots;
    std::size_t topk;
};

[[nodiscard]] inline TP2DFlashProposalCapacity tp2_dflash_proposal_capacity(
    QType head_type, int head_rows, int width, int batch) {
    if (width < 2 || width > 16 || batch < 1 || batch > 8) {
        throw std::invalid_argument("TP2 DFlash2 proposal geometry is invalid");
    }
    const int steps = width - 1;
    WorkspaceLayoutBuilder primary, peer;
    (void)primary.alloc(DType::BF16, {5120, steps, batch});
    (void)primary.alloc(DType::BF16, {256, steps, batch});
    (void)primary.alloc(DType::FP32, {16, steps, batch});
    (void)peer.alloc(DType::BF16, {5120, steps, batch});
    return {{primary.peak_bytes(256), peer.peak_bytes(256)},
            ops::linear_topk_tp2_workspace_capacity_bytes(head_type, head_rows, steps * batch)};
}

inline void tp2_dflash_propose(
    const std::array<Tensor, 2>& final_hidden,
    const std::array<const DFlash2Weights*, 2>& weights,
    const std::array<Weight, 2>& head, const std::array<Tensor, 2>* id_map,
    const Tensor& anchors, const Tensor& base_positions,
    const ops::SamplingConfig* configs, const std::array<Tensor, 2>& drafts,
    const std::array<Tensor, 2>& proposal_ids, const std::array<Tensor, 2>& proposal_q,
    const std::array<WorkspaceArena*, 2>& roots,
    const std::array<WorkspaceArena*, 2>& topk_workspace,
    const ExecutionContext& execution, const ops::PeerEvents& events) {
    if (execution.tp != 2 || !events.live() || !weights[0] || !weights[1] || !roots[0] ||
        !roots[1] || !topk_workspace[0] || !topk_workspace[1] ||
        final_hidden[0].dtype != DType::BF16 || final_hidden[1].dtype != DType::BF16 ||
        final_hidden[0].ne[0] != 5120 || final_hidden[1].ne[0] != 5120 ||
        final_hidden[0].ne[1] != final_hidden[1].ne[1] ||
        final_hidden[0].ne[2] != final_hidden[1].ne[2] ||
        !final_hidden[0].is_contiguous() || !final_hidden[1].is_contiguous()) {
        throw std::invalid_argument("TP2 DFlash2 proposal inputs are invalid");
    }
    const int width = final_hidden[0].ne[1];
    const int batch = final_hidden[0].ne[2];
    for (int rank = 0; rank < 2; ++rank) {
        if (drafts[rank].dtype != DType::I32 || drafts[rank].ne[0] != width - 1 ||
            drafts[rank].ne[1] != batch || drafts[rank].ne[2] != 1 ||
            !drafts[rank].is_contiguous() || proposal_ids[rank].dtype != DType::I32 ||
            proposal_ids[rank].ne[0] != 16 ||
            proposal_ids[rank].ne[1] != width - 1 || proposal_ids[rank].ne[2] != batch ||
            !proposal_ids[rank].is_contiguous() || proposal_q[rank].dtype != DType::FP32 ||
            proposal_q[rank].ne[0] != 16 || proposal_q[rank].ne[1] != width - 1 ||
            proposal_q[rank].ne[2] != batch || !proposal_q[rank].is_contiguous()) {
            throw std::invalid_argument("TP2 DFlash2 proposal output geometry is invalid");
        }
    }
    const auto capacity = tp2_dflash_proposal_capacity(head[0].qtype, head[0].n, width, batch);
    if (roots[0]->capacity() - roots[0]->used() < capacity.roots[0] ||
        roots[1]->capacity() - roots[1]->used() < capacity.roots[1]) {
        throw std::invalid_argument("TP2 DFlash2 proposal workspace is too small");
    }
    const int steps = width - 1;
    const int columns = steps * batch;
    auto primary_scope = roots[0]->scope();
    auto peer_scope = roots[1]->scope();
    std::array<Tensor, 2> masks;
    int previous_device = 0;
    CUDA_CHECK(cudaGetDevice(&previous_device));
    try {
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
            masks[rank] = roots[rank]->alloc(DType::BF16, {5120, steps, batch});
            const auto* source = static_cast<const std::byte*>(final_hidden[rank].data) +
                                 5120 * sizeof(std::uint16_t);
            CUDA_CHECK(cudaMemcpy2DAsync(
                masks[rank].data, static_cast<std::size_t>(5120) * steps * 2,
                source, static_cast<std::size_t>(5120) * width * 2,
                static_cast<std::size_t>(5120) * steps * 2, batch,
                cudaMemcpyDeviceToDevice, execution.dev[rank]->stream));
        }
        CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
        Tensor projected = roots[0]->alloc(DType::BF16, {256, steps, batch});
        Tensor projected_flat = projected.view({256, columns});
        ops::linear(masks[0].view({5120, columns}), weights[0]->hidden_projection,
                    projected_flat, execution.dev[0]->stream);
        Tensor scores = roots[0]->alloc(DType::FP32, {16, steps, batch});
        Tensor flat_ids = proposal_ids[0].view({16, columns});
        Tensor flat_scores = scores.view({16, columns});
        ops::linear_topk_tp2({masks[0].view({5120, columns}),
                              masks[1].view({5120, columns})}, head, id_map,
                             flat_ids, flat_scores,
                             topk_workspace, execution, events);
        CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
        Tensor selected = drafts[0];
        Tensor probabilities = proposal_q[0];
        ops::candidate_selector_path(proposal_ids[0], scores, projected, anchors,
                                     weights[0]->predecessor_codebook,
                                     weights[0]->successor_codebook,
                                     base_positions, configs, selected, probabilities,
                                     *roots[0], execution.dev[0]->stream);
        if (batch == 1 && std::getenv("NINFER_DFLASH2_INSPECT_EDGE") != nullptr) {
            cudaStreamCaptureStatus capture_status;
            CUDA_CHECK(cudaStreamIsCapturing(execution.dev[0]->stream, &capture_status));
            if (capture_status == cudaStreamCaptureStatusNone) {
                CUDA_CHECK(cudaStreamSynchronize(execution.dev[0]->stream));
                std::array<std::int32_t, 16> candidate_ids{};
                std::array<float, 16> unary{};
                std::array<std::uint16_t, 256> projected_host{}, predecessor{};
                std::array<std::uint16_t, 256> successor{};
                std::int32_t anchor = 0, position = 0, draft = 0;
                CUDA_CHECK(cudaMemcpy(candidate_ids.data(), proposal_ids[0].data,
                                      sizeof(candidate_ids), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(unary.data(), scores.data,
                                      sizeof(unary), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(projected_host.data(), projected.data,
                                      sizeof(projected_host), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&anchor, anchors.data, sizeof(anchor), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&position, base_positions.data, sizeof(position), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&draft, drafts[0].data, sizeof(draft), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(predecessor.data(),
                                      static_cast<const std::uint16_t*>(weights[0]->predecessor_codebook.data)
                                          + 256LL * anchor,
                                      sizeof(predecessor), cudaMemcpyDeviceToHost));
                const auto bf16 = [](std::uint16_t bits) {
                    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
                };
                std::fprintf(stderr, "dflash2-edge position=%d anchor=%d draft=%d entries=",
                             position, anchor, draft);
                for (int candidate = 0; candidate < 16; ++candidate) {
                    CUDA_CHECK(cudaMemcpy(successor.data(),
                                          static_cast<const std::uint16_t*>(weights[0]->successor_codebook.data)
                                              + 256LL * candidate_ids[candidate],
                                          sizeof(successor), cudaMemcpyDeviceToHost));
                    double interaction = 0.0;
                    for (int rank = 0; rank < 256; ++rank) {
                        interaction += static_cast<double>(bf16(predecessor[rank])) *
                                       bf16(projected_host[rank]) * bf16(successor[rank]);
                    }
                    std::fprintf(stderr, "%s%d:%g:%g", candidate == 0 ? "" : ",",
                                 candidate_ids[candidate], static_cast<double>(unary[candidate]),
                                 static_cast<double>(unary[candidate]) + interaction);
                }
                std::fputc('\n', stderr);
            }
        }
        CUDA_CHECK(cudaEventRecord(events.inputs_ready(0), execution.dev[0]->stream));
        CUDA_CHECK(cudaSetDevice(execution.dev[1]->device));
        CUDA_CHECK(cudaStreamWaitEvent(execution.dev[1]->stream, events.inputs_ready(0), 0));
        CUDA_CHECK(cudaMemcpyAsync(drafts[1].data, drafts[0].data, drafts[0].bytes(),
                                   cudaMemcpyDeviceToDevice, execution.dev[1]->stream));
        CUDA_CHECK(cudaMemcpyAsync(proposal_ids[1].data, proposal_ids[0].data,
                                   proposal_ids[0].bytes(), cudaMemcpyDeviceToDevice,
                                   execution.dev[1]->stream));
        CUDA_CHECK(cudaMemcpyAsync(proposal_q[1].data, proposal_q[0].data, proposal_q[0].bytes(),
                                   cudaMemcpyDeviceToDevice, execution.dev[1]->stream));
        CUDA_CHECK(cudaEventRecord(events.pull_done(1), execution.dev[1]->stream));
        CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
        CUDA_CHECK(cudaStreamWaitEvent(execution.dev[0]->stream, events.pull_done(1), 0));
    } catch (...) {
        (void)cudaSetDevice(previous_device);
        throw;
    }
    CUDA_CHECK(cudaSetDevice(previous_device));
}

inline void tp2_dflash_run_proposal(
    const std::array<TP2DFlashContextState*, 2>& state,
    const std::array<const RuntimeModelView*, 2>& models,
    const std::array<const DFlash2Weights*, 2>& weights,
    const std::array<Tensor, 2>& anchors, const std::array<Tensor, 2>& frontiers,
    const std::array<Tensor, 2>& valid_columns, const std::array<Tensor, 2>& lanes,
    const Tensor& base_positions, const ops::SamplingConfig* configs,
    const std::array<Weight, 2>& head, const std::array<Tensor, 2>* id_map,
    std::int32_t batch, ops::SwaContextExecutionEnvelope envelope,
    const std::array<WorkspaceArena*, 2>& draft_roots,
    const std::array<WorkspaceArena*, 2>& draft_ops,
    const std::array<WorkspaceArena*, 2>& proposal_roots,
    const std::array<WorkspaceArena*, 2>& topk_workspace,
    const ExecutionContext& execution, const ops::PeerEvents& events) {
    prepare_tp2_dflash_draft_inputs(state, models, anchors, frontiers, valid_columns,
                                    batch, execution);
    std::array<Tensor, 2> residual, positions, hidden, drafts, proposal_ids, proposal_q;
    for (int rank = 0; rank < 2; ++rank) {
        residual[rank] = state[rank]->draft_input.slice(2, 0, batch);
        positions[rank] = state[rank]->draft_positions.slice(1, 0, batch);
        hidden[rank] = state[rank]->draft_hidden.slice(2, 0, batch);
        drafts[rank] = state[rank]->draft_tokens.slice(1, 0, batch);
        proposal_ids[rank] = state[rank]->proposal_ids.slice(2, 0, batch);
        proposal_q[rank] = state[rank]->proposal_q.slice(2, 0, batch);
    }
    tp2_dflash_draft_backbone(residual, positions, valid_columns, lanes, weights, state,
                              envelope, hidden, draft_roots, draft_ops, execution, events);
    tp2_dflash_propose(hidden, weights, head, id_map, anchors[0], base_positions, configs,
                       drafts, proposal_ids, proposal_q, proposal_roots, topk_workspace,
                       execution, events);
}

} // namespace ninfer::targets::qwen3_6_27b::detail
