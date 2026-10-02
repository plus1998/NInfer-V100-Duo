#pragma once

#include "core/arena.h"
#include "core/layout.h"
#include "ninfer/ops/swa.h"
#include "ninfer/types.h"
#include "targets/qwen3_6/impl/runtime/tp2_feature_sink.h"

#include <ninfer/targets/qwen3_6/round_state.h>

#include <cstdint>
#include <optional>

namespace ninfer::ops { class PeerEvents; }

namespace ninfer::targets::qwen3_6::detail {

struct TP2DFlashProposalView {
    Tensor drafts;
    Tensor ids;
    Tensor q;
};

template <class Variant>
struct TP2DFlashExtension {
    static constexpr bool supported = false;
    struct Layout {};
    struct State {
        void restore_checkpoint(std::int32_t, cudaStream_t) {
            throw std::logic_error("DFlash2 checkpoint is not supported by this target");
        }
    };

    [[nodiscard]] static std::optional<Layout> plan(LayoutBuilder&, std::int32_t,
                                                     std::int32_t, std::int32_t, bool) {
        return std::nullopt;
    }

    static void bind(std::optional<State>&, DeviceSpan, const std::optional<Layout>&) {}

    [[nodiscard]] static TP2FeatureSink make_prefill_sink(
        const std::array<State*, 2>&,
        const std::array<const typename Variant::ModelView*, 2>&,
        const ExecutionContext&, const ops::PeerEvents&, std::int32_t) {
        throw std::logic_error("TP2 DFlash2 prefill is not supported by this target");
    }

    [[nodiscard]] static std::size_t kv_payload_bytes(const std::optional<Layout>&) noexcept {
        return 0;
    }

    [[nodiscard]] static std::size_t proposal_workspace_capacity(
        std::int32_t, std::int32_t, std::uint32_t, ProposalHead,
        typename Variant::WeightsProfile) {
        return 0;
    }

    static void run_proposal(const std::array<State*, 2>&,
                             const std::array<const typename Variant::ModelView*, 2>&,
                             const std::array<qwen3_6::DFlash2DecodeState*, 2>&,
                             const std::array<WorkspaceArena*, 2>&, int,
                             ProposalHead, ops::SwaContextExecutionEnvelope,
                             const ExecutionContext&, const ops::PeerEvents&) {
        throw std::logic_error("TP2 DFlash2 proposal is not supported by this target");
    }

    static void catch_up(const std::array<State*, 2>&,
                         const std::array<const typename Variant::ModelView*, 2>&,
                         const std::array<qwen3_6::DFlash2DecodeState*, 2>&,
                         const ExecutionContext&, const ops::PeerEvents&, int) {
        throw std::logic_error("TP2 DFlash2 context is not supported by this target");
    }

    [[nodiscard]] static TP2FeatureSink make_verify_sink(const std::array<State*, 2>&, int) {
        throw std::logic_error("TP2 DFlash2 verifier is not supported by this target");
    }

    [[nodiscard]] static TP2DFlashProposalView proposal_view(State&, int) {
        throw std::logic_error("TP2 DFlash2 proposals are not supported by this target");
    }
};

} // namespace ninfer::targets::qwen3_6::detail
