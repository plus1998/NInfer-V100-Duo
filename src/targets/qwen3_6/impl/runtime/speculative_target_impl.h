#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::GqaExecutionEnvelope envelope) {
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.lanes, envelope,
                                 frame.target_hidden, frame.target_logits, frame.target_tokens,
                                 *frame.feature_sink);
    } else {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.lanes, envelope,
                                 frame.target_hidden, frame.target_logits, frame.target_tokens);
    }
    ops::speculative_accept_greedy_drafts(frame.target_tokens, frame.target_logits, frame.drafts,
                                          frame.current_extents, frame.frontiers, frame.anchors,
                                          frame.licensed_tokens, frame.licensed_counts,
                                          frame.accepted_drafts, TextConfig::token_domain,
                                          frame.sampling, execution.work, execution.device.stream);
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.lanes, continuation_hidden_store,
                 execution.device.stream);
}

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          TargetVerifyFrameView peer, ops::GqaExecutionEnvelope envelope) {
    if (execution.peer == nullptr) {
        throw std::logic_error("tensor-parallel target verify requires a peer");
    }
    if (frame.replay_records == nullptr || peer.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    if (frame.feature_sink != nullptr || peer.feature_sink != nullptr) {
        // DFlash is the only feature-sink user and it is rejected at tp2 before reaching here.
        throw std::logic_error("tensor-parallel target verify has no feature-sink path");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    card.target_verify_batch({frame.ids, peer.ids},
                             {frame.cache_positions, peer.cache_positions},
                             {frame.rope_positions, peer.rope_positions},
                             {frame.valid_columns, peer.valid_columns},
                             {frame.kv_table_rows, peer.kv_table_rows}, {frame.lanes, peer.lanes},
                             envelope, {frame.target_hidden, peer.target_hidden},
                             {frame.target_logits, peer.target_logits},
                             {frame.target_tokens, peer.target_tokens});
    const ExecutionContext& ec      = *execution.peer->execution;
    WorkspaceArena* work[2]         = {&execution.work, execution.peer->work};
    TargetVerifyFrameView* views[2] = {&frame, &peer};
    for_each_rank(ec, [&](int rank) {
        const auto r              = static_cast<std::size_t>(rank);
        TargetVerifyFrameView& v  = *views[r];
        cudaStream_t stream       = ec.dev[rank]->stream;
        ops::speculative_accept_greedy_drafts(v.target_tokens, v.target_logits, v.drafts,
                                              v.current_extents, v.frontiers, v.anchors,
                                              v.licensed_tokens, v.licensed_counts,
                                              v.accepted_drafts, TextConfig::token_domain,
                                              v.sampling, *work[r], stream);
        ops::speculative_select_accepted_hidden(v.target_hidden, v.accepted_drafts,
                                                v.selected_hidden, stream);
    });
    ops::scatter(frame.selected_hidden, frame.lanes, continuation_hidden_store,
                 execution.device.stream);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
