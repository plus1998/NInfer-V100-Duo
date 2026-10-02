#pragma once

#include "targets/qwen3_6_27b/impl/config.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/tp2_dflash_context.h"
#include "targets/qwen3_6_27b/impl/tp2_dflash_proposal.h"
#include "targets/qwen3_6/impl/runtime/tp2_dflash_extension.h"
#include "ninfer/ops/allreduce.h" // ExecutionContext, ops::PeerEvents (tp2 split forms)
#include <ninfer/targets/qwen3_6/runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {

using GraphExecutionProfile = qwen3_6::GraphExecutionProfile;

// Compile-time data and the three closed execution leaves supplied to the Qwen3.6 family runtime.
// It owns no request state, execution phase, graph object, or schedule callback.
struct Variant {
    using WeightsProfile                 = detail::WeightsProfile;
    using TextConfig                     = detail::TextConfig;
    using VisionConfig                   = detail::VisionConfig;
    using DFlashConfig                   = detail::DFlashConfig;
    using ModelView                      = detail::RuntimeModelView;
    using FullAttentionProjectionWeights = detail::FullAttentionProjectionPayload;
    using GdnProjectionWeights           = detail::GdnProjectionPayload;
    using PostMixerWeights               = detail::DensePostMixerPayload;
    using MtpAttentionProjectionWeights  = detail::MtpAttentionPayload;
    using MtpPostMixerWeights            = detail::DensePostMixerPayload;
    using VisionWeights                  = qwen3_6::VisionWeights;
    using GraphExecutionProfile          = detail::GraphExecutionProfile;

    static constexpr float attention_scale                     = kAttentionScale;
    static constexpr float gdn_scale                           = kGdnScale;
    static constexpr std::uint32_t prefill_chunk_alignment     = kPrefillChunkAlignment;
    static constexpr std::uint32_t maximum_mtp_draft_tokens    = kMaximumMtpDraftTokens;
    static constexpr std::uint32_t maximum_dflash_draft_tokens = kMaximumDFlashDraftTokens;
    static constexpr std::uint32_t maximum_context             = kNativeContext;
    // This checkpoint's text attention is head_dim 256 with a 64-dim partial rotary
    // subspace (32 frequency pairs), exactly the geometry `detail::yarn_scale()` computes a
    // corrected inverse-frequency table for, and the published qwen3.8 long-context deployment
    // config carries `rope_type: yarn` for it. `--rope yarn` is therefore admissible here.
    static constexpr bool supports_yarn_rope                   = true;
    static constexpr bool supports_dflash                      = DFlashConfig::supported;
    static constexpr std::int32_t draft_head_rows              = 131072;

    static void attention_projection(const Tensor& hidden,
                                     const FullAttentionProjectionWeights& weights, Tensor& query,
                                     Tensor& gate, Tensor& key, Tensor& value,
                                     qwen3_6::TextPhase phase, WorkspaceArena& workspace,
                                     cudaStream_t stream);
    static void attention_output_projection(const Tensor& attention, const Weight& weight,
                                            Tensor& residual, qwen3_6::TextPhase phase,
                                            WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_attention_projection(const Tensor& hidden,
                                         const MtpAttentionProjectionWeights& weights,
                                         Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                                         WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_kv_projection(const Tensor& hidden,
                                  const MtpAttentionProjectionWeights& weights, Tensor& key,
                                  Tensor& value, WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_q_gate_projection(const Tensor& hidden,
                                      const MtpAttentionProjectionWeights& weights, Tensor& query,
                                      Tensor& gate, WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                     Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase phase,
                                     WorkspaceArena& workspace, cudaStream_t stream);
    static void
    gdn_input_projection_snapshot(const Tensor& hidden, const GdnProjectionWeights& weights,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_slot,
                                  const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& output_gate, qwen3_6::TextPhase phase,
                                  WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_input_projection_record(
        const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
        const Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slots,
        Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& output_gate,
        qwen3_6::TextPhase phase, WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                      qwen3_6::TextPhase phase, WorkspaceArena& workspace,
                                      cudaStream_t stream);
    static void gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                            float eps, const GdnProjectionWeights& weights,
                                            Tensor& hidden, Tensor& g, Tensor& beta,
                                            WorkspaceArena& workspace, cudaStream_t stream);
    static void post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                           qwen3_6::TextPhase phase, WorkspaceArena& workspace,
                           cudaStream_t stream);
    static void mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                               Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream);
    [[nodiscard]] static std::size_t
    mtp_attention_projection_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                                std::int32_t last);
    [[nodiscard]] static std::size_t
    mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                  qwen3_6::TextPhase phase, std::int32_t first,
                                                  std::int32_t last);
    [[nodiscard]] static std::size_t
    attention_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         qwen3_6::TextPhase phase,
                                                         std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_input_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                  qwen3_6::TextPhase phase, std::int32_t first,
                                                  std::int32_t last);
    [[nodiscard]] static std::size_t gdn_input_projection_snapshot_workspace_capacity_bytes(
        WeightsProfile weights_profile, qwen3_6::TextPhase phase, std::int32_t batch_size,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t gdn_input_projection_record_workspace_capacity_bytes(
        WeightsProfile weights_profile, qwen3_6::TextPhase phase, std::int32_t batch_size,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                   qwen3_6::TextPhase phase, std::int32_t first,
                                                   std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile, qwen3_6::TextPhase phase,
                                        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                                             std::int32_t last);

    // --- tp == 2 split forms -------------------------------------------------------------------
    //
    // One call drives BOTH ranks: every argument is an array indexed by rank, holding that rank's
    // own shard-shaped tensor/weight resident on `ec.dev[rank]` and issued on that rank's stream.
    // The three column-parallel leaves need no communication at all; the two row-parallel leaves
    // (`attention_output_projection`, `gdn_output_projection`, and `post_mixer`'s trailing `down`)
    // carry the block's single all-reduce inside `ops::linear_add_row_parallel`, which also folds
    // the residual in exactly once, on rank 0, before the reduce.
    //
    // `staging[r]` is scratch of the residual's shape on `ec.dev[r]`; it receives the peer's
    // partial and its contents afterwards are unspecified.
    static void attention_projection(const std::array<Tensor, 2>& hidden,
                                     const std::array<const FullAttentionProjectionWeights*, 2>& w,
                                     const std::array<Tensor, 2>& query,
                                     const std::array<Tensor, 2>& gate,
                                     const std::array<Tensor, 2>& key,
                                     const std::array<Tensor, 2>& value, qwen3_6::TextPhase phase,
                                     const std::array<WorkspaceArena*, 2>& workspace,
                                     const ExecutionContext& ec);
    static void attention_output_projection(const std::array<Tensor, 2>& attention,
                                            const std::array<Weight, 2>& weight,
                                            const std::array<Tensor, 2>& residual,
                                            const std::array<Tensor, 2>& staging,
                                            qwen3_6::TextPhase phase,
                                            const std::array<WorkspaceArena*, 2>& workspace,
                                            const ExecutionContext& ec, const ops::PeerEvents& ev);
    static void gdn_input_projection(const std::array<Tensor, 2>& hidden,
                                     const std::array<const GdnProjectionWeights*, 2>& w,
                                     const std::array<Tensor, 2>& qkv,
                                     const std::array<Tensor, 2>& output_gate,
                                     qwen3_6::TextPhase phase,
                                     const std::array<WorkspaceArena*, 2>& workspace,
                                     const ExecutionContext& ec);
    static void gdn_input_projection_snapshot(
        const std::array<Tensor, 2>& hidden,
        const std::array<const GdnProjectionWeights*, 2>& w,
        const std::array<Tensor, 2>& conv_weight, const std::array<Tensor, 2>& conv_states,
        const std::array<Tensor, 2>& valid_columns, const std::array<Tensor, 2>& initial_slot,
        const std::array<Tensor, 2>& snapshot_base_slot, const std::array<Tensor, 2>& query,
        const std::array<Tensor, 2>& key, const std::array<Tensor, 2>& value,
        const std::array<Tensor, 2>& output_gate, qwen3_6::TextPhase phase,
        const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec);
    static void gdn_input_projection_record(
        const std::array<Tensor, 2>& hidden, const std::array<const GdnProjectionWeights*, 2>& w,
        const std::array<Tensor, 2>& conv_weight, const std::array<Tensor, 2>& conv_states,
        const std::array<Tensor, 2>& valid_columns, const std::array<Tensor, 2>& initial_slots,
        const std::array<Tensor, 2>& conv_record, const std::array<Tensor, 2>& query,
        const std::array<Tensor, 2>& key, const std::array<Tensor, 2>& value,
        const std::array<Tensor, 2>& output_gate, qwen3_6::TextPhase phase,
        const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec);
    static void gdn_output_projection(const std::array<Tensor, 2>& hidden,
                                      const std::array<Weight, 2>& weight,
                                      const std::array<Tensor, 2>& residual,
                                      const std::array<Tensor, 2>& staging,
                                      qwen3_6::TextPhase phase,
                                      const std::array<WorkspaceArena*, 2>& workspace,
                                      const ExecutionContext& ec, const ops::PeerEvents& ev);
    // MTP split leaves. `mtp_attention_projection` is column-parallel over the packed
    // [14336, 5120] parent (shard [7168, 5120], whose row order is q | k | gate | v at the
    // per-rank section widths) and then splits each rank's own packed block in place;
    // `mtp_post_mixer` is the MTP layer's swiglu pair, identical in shape to `post_mixer`.
    // `mtp_kv_projection` / `mtp_q_gate_projection` are the prefill-only section subsets, which
    // the tp1 leaf fuses with `linear_pair` and which split into independent column-parallel
    // calls because their two outputs are separate row views of the same shard.
    static void mtp_attention_projection(const std::array<Tensor, 2>& hidden,
                                         const std::array<const MtpAttentionProjectionWeights*, 2>& w,
                                         const std::array<Tensor, 2>& query,
                                         const std::array<Tensor, 2>& gate,
                                         const std::array<Tensor, 2>& key,
                                         const std::array<Tensor, 2>& value,
                                         const std::array<WorkspaceArena*, 2>& workspace,
                                         const ExecutionContext& ec);
    static void mtp_kv_projection(const std::array<Tensor, 2>& hidden,
                                  const std::array<const MtpAttentionProjectionWeights*, 2>& w,
                                  const std::array<Tensor, 2>& key,
                                  const std::array<Tensor, 2>& value,
                                  const std::array<WorkspaceArena*, 2>& workspace,
                                  const ExecutionContext& ec);
    static void mtp_q_gate_projection(const std::array<Tensor, 2>& hidden,
                                      const std::array<const MtpAttentionProjectionWeights*, 2>& w,
                                      const std::array<Tensor, 2>& query,
                                      const std::array<Tensor, 2>& gate,
                                      const std::array<WorkspaceArena*, 2>& workspace,
                                      const ExecutionContext& ec);
    static void mtp_post_mixer(const std::array<Tensor, 2>& hidden,
                               const std::array<const MtpPostMixerWeights*, 2>& w,
                               const std::array<Tensor, 2>& residual,
                               const std::array<Tensor, 2>& staging,
                               const std::array<WorkspaceArena*, 2>& workspace,
                               const ExecutionContext& ec, const ops::PeerEvents& ev);
    static void gdn_control_projection(const std::array<Tensor, 2>& hidden,
                                       const std::array<const GdnProjectionWeights*, 2>& w,
                                       const std::array<Tensor, 2>& g,
                                       const std::array<Tensor, 2>& beta,
                                       const std::array<WorkspaceArena*, 2>& workspace,
                                       const ExecutionContext& ec);
    static void post_mixer(const std::array<Tensor, 2>& hidden,
                           const std::array<const PostMixerWeights*, 2>& w,
                           const std::array<Tensor, 2>& residual,
                           const std::array<Tensor, 2>& staging, qwen3_6::TextPhase phase,
                           const std::array<WorkspaceArena*, 2>& workspace,
                           const ExecutionContext& ec, const ops::PeerEvents& ev);

    [[nodiscard]] static std::vector<GraphExecutionProfile>
    ordinary_graph_profiles(std::uint32_t capacity);
    [[nodiscard]] static std::vector<GraphExecutionProfile>
    mtp_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window);
    [[nodiscard]] static std::vector<GraphExecutionProfile>
    dflash_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window,
                          std::uint32_t batch_size);
};

} // namespace ninfer::targets::qwen3_6_27b::detail

namespace ninfer::targets::qwen3_6::detail {

template <>
struct TP2DFlashExtension<qwen3_6_27b::detail::Variant> {
    static constexpr bool supported = true;
    using Layout = qwen3_6_27b::detail::TP2DFlashContextLayout;
    using State = qwen3_6_27b::detail::TP2DFlashContextState;

    [[nodiscard]] static std::optional<Layout> plan(LayoutBuilder& builder,
                                                     std::int32_t prefill_chunk,
                                                     std::int32_t concurrency,
                                                     std::int32_t verify_width, bool enabled) {
        if (!enabled) { return std::nullopt; }
        return Layout::plan(builder, prefill_chunk, concurrency, verify_width);
    }

    static void bind(std::optional<State>& state, DeviceSpan backing,
                     const std::optional<Layout>& layout) {
        if (layout) { state.emplace(backing, *layout); }
    }

    [[nodiscard]] static TP2FeatureSink make_prefill_sink(
        const std::array<State*, 2>& state,
        const std::array<const qwen3_6_27b::detail::RuntimeModelView*, 2>& models,
        const ExecutionContext& execution, const ops::PeerEvents& events, std::int32_t lane) {
        if (!state[0] || !state[1] || !models[0] || !models[1] ||
            !models[0]->dflash2 || !models[1]->dflash2) {
            throw std::logic_error("DFlash2 prefill needs two bound model and state replicas");
        }
        qwen3_6_27b::detail::TP2DFlashContextBuffers buffers{
            .projected = {state[0]->prefill_projected, state[1]->prefill_projected},
            .staging = {state[0]->prefill_staging, state[1]->prefill_staging},
            .normalized = {state[0]->prefill_normalized, state[1]->prefill_normalized}};
        return qwen3_6_27b::detail::make_tp2_dflash_prefill_sink(
            state, {&*models[0]->dflash2, &*models[1]->dflash2}, buffers,
            {&state[0]->prefill_workspace, &state[1]->prefill_workspace}, execution, events,
            lane);
    }

    [[nodiscard]] static std::size_t kv_payload_bytes(
        const std::optional<Layout>& layout) noexcept {
        return layout ? layout->cache.payload_bytes() +
                            layout->rewrite_checkpoint.payload_bytes() : 0;
    }

    [[nodiscard]] static std::size_t proposal_workspace_capacity(
        std::int32_t width, std::int32_t batch, std::uint32_t max_context,
        ProposalHead head, qwen3_6_27b::detail::WeightsProfile profile) {
        QType head_type;
        switch (profile) {
        case qwen3_6_27b::detail::WeightsProfile::Qwen38Gguf:
            head_type = QType::GGUF;
            break;
        case qwen3_6_27b::detail::WeightsProfile::Qwen38Nvfp4:
            if (head != ProposalHead::Optimized) {
                throw std::invalid_argument("DFlash2 NVFP4 requires the optimized proposal head");
            }
            head_type = QType::Q4G64_F16S;
            break;
        default:
            throw std::invalid_argument("DFlash2 TP2 proposal requires GGUF or NVFP4 weights");
        }
        const auto draft = qwen3_6_27b::detail::tp2_dflash_draft_workspace_capacity(
            width, batch, {0, max_context});
        const auto proposal = qwen3_6_27b::detail::tp2_dflash_proposal_capacity(
            head_type, head == ProposalHead::Optimized ? 65536 : 124160, width, batch);
        std::size_t capacity = 0;
        for (int rank = 0; rank < 2; ++rank) {
            const std::size_t roots = std::max(draft.roots, proposal.roots[rank]);
            const std::size_t scratch = std::max(draft.ops, proposal.topk);
            capacity = std::max(capacity, roots + scratch + 512);
        }
        return capacity;
    }

    static void run_proposal(
        const std::array<State*, 2>& state,
        const std::array<const qwen3_6_27b::detail::RuntimeModelView*, 2>& models,
        const std::array<qwen3_6::DFlash2DecodeState*, 2>& frames,
        const std::array<WorkspaceArena*, 2>& work, int batch, ProposalHead head,
        ops::SwaContextExecutionEnvelope envelope, const ExecutionContext& execution,
        const ops::PeerEvents& events) {
        if (!work[0] || !work[1] || !frames[0] || !frames[1] || !models[0] || !models[1] ||
            !models[0]->dflash2 || !models[1]->dflash2 || !state[0] || !state[1]) {
            throw std::logic_error("DFlash2 proposal requires two loaded replicas");
        }
        const int width = state[0]->draft_ids.ne[0];
        const auto draft = qwen3_6_27b::detail::tp2_dflash_draft_workspace_capacity(
            width, batch, envelope);
        const std::array<const qwen3_6_27b::detail::DFlash2Weights*, 2> weights = {
            &*models[0]->dflash2, &*models[1]->dflash2};
        std::array<Weight, 2> heads;
        std::array<Tensor, 2> maps;
        for (int rank = 0; rank < 2; ++rank) {
            if (head == ProposalHead::Optimized) {
                if (!models[rank]->optimized_proposal) {
                    throw std::logic_error("DFlash2 optimized proposal head is unavailable");
                }
                heads[rank] = models[rank]->optimized_proposal->head;
                maps[rank] = models[rank]->optimized_proposal->token_ids;
            } else {
                heads[rank] = models[rank]->output_head;
            }
        }
        const auto proposal = qwen3_6_27b::detail::tp2_dflash_proposal_capacity(
            heads[0].qtype, heads[0].n, width, batch);
        auto primary_scope = work[0]->scope();
        auto peer_scope = work[1]->scope();
        std::array<std::optional<WorkspaceArena>, 2> roots, scratch;
        std::array<WorkspaceArena*, 2> root_views, scratch_views;
        std::array<Tensor, 2> anchors, frontiers, valid, lanes;
        for (int rank = 0; rank < 2; ++rank) {
            roots[rank].emplace(work[rank]->alloc_bytes(
                std::max(draft.roots, proposal.roots[rank])));
            scratch[rank].emplace(work[rank]->alloc_bytes(
                std::max(draft.ops, proposal.topk)));
            root_views[rank] = &*roots[rank];
            scratch_views[rank] = &*scratch[rank];
            anchors[rank] = frames[rank]->anchors.slice(0, 0, batch);
            frontiers[rank] = frames[rank]->execution_frontiers.slice(0, 0, batch);
            valid[rank] = frames[rank]->target_valid_columns.slice(0, 0, batch);
            lanes[rank] = frames[rank]->lanes.slice(0, 0, batch);
        }
        qwen3_6_27b::detail::tp2_dflash_run_proposal(
            state, models, weights, anchors, frontiers, valid, lanes, frontiers[0],
            frames[0]->sampling, heads, head == ProposalHead::Optimized ? &maps : nullptr,
            batch, envelope, root_views, scratch_views, root_views, scratch_views, execution,
            events);
    }

    static void catch_up(
        const std::array<State*, 2>& state,
        const std::array<const qwen3_6_27b::detail::RuntimeModelView*, 2>& models,
        const std::array<qwen3_6::DFlash2DecodeState*, 2>& frames,
        const ExecutionContext& execution, const ops::PeerEvents& events, int batch) {
        if (!models[0] || !models[1] || !models[0]->dflash2 || !models[1]->dflash2) {
            throw std::logic_error("DFlash2 context catch-up requires two model replicas");
        }
        qwen3_6_27b::detail::catch_up_tp2_dflash_context(
            state, {&*models[0]->dflash2, &*models[1]->dflash2}, frames,
            execution, events, batch);
    }

    [[nodiscard]] static TP2FeatureSink make_verify_sink(const std::array<State*, 2>& state,
                                                          int batch) {
        return qwen3_6_27b::detail::make_tp2_dflash_verify_sink(state, batch);
    }

    [[nodiscard]] static TP2DFlashProposalView proposal_view(State& state, int batch) {
        if (batch <= 0 || batch > state.draft_tokens.ne[1]) {
            throw std::invalid_argument("DFlash2 proposal view has invalid batch size");
        }
        return {state.draft_tokens.slice(1, 0, batch),
                state.proposal_ids.slice(2, 0, batch),
                state.proposal_q.slice(2, 0, batch)};
    }
};

} // namespace ninfer::targets::qwen3_6::detail
