#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/speculative_round.h"

#include <array>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

using DFlash2Extension = qwen3_6::detail::TP2DFlashExtension<Variant>;

TargetVerifyFrameView dflash2_verify_view(qwen3_6::DFlash2DecodeState& frame,
                                          DFlash2Extension::State& state, int batch,
                                          const GdnReplayRecords* records) {
    const TP2DFlashProposalView proposal = DFlash2Extension::proposal_view(state, batch);
    return TargetVerifyFrameView{
        .ids = frame.verify_ids.slice(1, 0, batch),
        .cache_positions = frame.target_positions.slice(1, 0, batch),
        .rope_positions = frame.target_rope_positions.slice(1, 0, batch),
        .valid_columns = frame.target_valid_columns.slice(0, 0, batch),
        .kv_table_rows = frame.text_kv_table_rows.slice(0, 0, batch),
        .lanes = frame.lanes.slice(0, 0, batch),
        .target_hidden = frame.target_hidden.slice(2, 0, batch),
        .target_logits = frame.target_logits.slice(2, 0, batch),
        .target_tokens = frame.target_tokens.slice(1, 0, batch),
        .drafts = proposal.drafts,
        .current_extents = frame.proposal_extents.slice(0, 0, batch),
        .frontiers = frame.execution_frontiers.slice(0, 0, batch),
        .anchors = frame.anchors.slice(0, 0, batch),
        .licensed_tokens = frame.licensed_tokens.slice(1, 0, batch),
        .licensed_counts = frame.licensed_counts.slice(0, 0, batch),
        .accepted_drafts = frame.accepted_drafts.slice(0, 0, batch),
        .selected_hidden = frame.target_continuation_hidden.slice(1, 0, batch),
        .replay_records = records,
        .sampling = frame.sampling,
        .proposal_ids = proposal.ids,
        .proposal_q = proposal.q,
    };
}

auto dflash2_decode_batch_body(DFlash2BatchContext& state, int batch,
                               ops::SwaContextExecutionEnvelope context_envelope,
                               ops::GqaExecutionEnvelope target_envelope) {
    return [&state, batch, context_envelope, target_envelope] {
        if (batch < 1 || batch > static_cast<int>(kMaximumConcurrency) ||
            state.execution.peer == nullptr || !state.states[0] || !state.states[1] ||
            !state.execution.replay_records || !state.execution.peer->replay_records) {
            throw std::logic_error("DFlash2 decode requires two state and replay replicas");
        }
        const auto& peer = *state.execution.peer;
        const ExecutionContext& execution = *peer.execution;
        const std::array<qwen3_6::DFlash2DecodeState*, 2> frames = {
            &state.frame, &*peer.io->dflash2_decode};
        const std::array<DFlash2Extension::State*, 2> states = state.states;
        const std::array<const LoadedModelData*, 2> models = {
            &state.execution.model, peer.model};
        const std::array<WorkspaceArena*, 2> work = {&state.execution.work, peer.work};
        for_each_rank(execution, [&](int rank) {
            const auto& ingress = rank == 0 ? state.host_ingress : state.peer_host_ingress;
            CUDA_CHECK(cudaMemcpyAsync(frames[rank]->ingress.data, &ingress, sizeof(ingress),
                                       cudaMemcpyHostToDevice, execution.dev[rank]->stream));
        });

        DFlash2Extension::catch_up(states, models, frames, execution, *peer.events, batch);
        state.execution.work.reset();
        peer.work->reset();
        DFlash2Extension::run_proposal(states, models, frames, work, batch,
                                        state.execution.proposal_head, context_envelope,
                                        execution, *peer.events);
        state.execution.work.reset();
        peer.work->reset();

        std::optional<TpExecution> tp = tp_execution(state.execution);
        TextContext card(state.execution.device, state.execution.model, state.execution.work,
                         state.execution.rope_frequency, {}, state.execution.linear_attention,
                         state.execution.io, state.execution.prefill_hidden,
                         state.execution.prefill_chunk, 0, {}, &state.text_cache, nullptr,
                         &*tp);
        const auto primary = dflash2_verify_view(*frames[0], *states[0], batch,
                                                 state.execution.replay_records);
        const auto secondary = dflash2_verify_view(*frames[1], *states[1], batch,
                                                   peer.replay_records);
        for_each_rank(execution, [&](int rank) {
            const auto& view = rank == 0 ? primary : secondary;
            Tensor ids = view.ids;
            Tensor positions = view.cache_positions;
            ops::speculative_prepare_verify_inputs(
                view.anchors, view.drafts, view.frontiers, view.current_extents, ids, positions,
                execution.dev[rank]->stream);
        });
        TP2FeatureSink sink = DFlash2Extension::make_verify_sink(states, batch);
        target_verify_accept(state.execution, state.continuation_hidden_store, card, primary,
                             secondary, target_envelope, &sink);
        const CurrentDevice restore;
        CUDA_CHECK(cudaSetDevice(state.execution.device.device));
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, state.frame.egress.data,
                                   sizeof(state.host_egress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

} // namespace

void capture_dflash2_decode_batch(DFlash2BatchContext& state, int batch,
                                   ops::SwaContextExecutionEnvelope context_envelope,
                                   ops::GqaExecutionEnvelope target_envelope,
                                   DecodeGraphDefinition& definition) {
    capture_graph(state, definition,
                  dflash2_decode_batch_body(state, batch, context_envelope, target_envelope));
}

void dflash2_decode_batch(DFlash2BatchContext& state, int batch,
                          ops::SwaContextExecutionEnvelope context_envelope,
                          ops::GqaExecutionEnvelope target_envelope,
                          DecodeGraphExecutable* executable) {
    run_prepared(state, executable,
                 dflash2_decode_batch_body(state, batch, context_envelope, target_envelope));
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
