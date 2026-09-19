#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include "ninfer/ops/silu_mul.h"
#include "core/layout.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_config.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_cutlass_sm70.h"
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

enum class Fp8LinearSwiGluRoute : std::uint8_t {
    A16,
    A8,
};

Fp8LinearSwiGluRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 linear_swiglu: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Fp8LinearSwiGluRoute::A16; }
    if (policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("fp8 linear_swiglu admits only A16 or A8");
    }
    return tokens == 1 || tokens >= 3 ? Fp8LinearSwiGluRoute::A8 : Fp8LinearSwiGluRoute::A16;
}

#ifdef NINFER_VOLTA_BUILD
struct Fp8QpnSplitWorkspace {
    DeviceSpan gate;
    DeviceSpan up;
};

template <class Geometry, class Allocator>
Fp8QpnSplitWorkspace allocate_qpn_split_workspace(Allocator& allocator, std::int32_t tokens) {
    constexpr std::int32_t kIntermediate = Geometry::kOutputRows / 2;
    const std::size_t bytes = static_cast<std::size_t>(kIntermediate) * tokens * sizeof(float);
    Fp8QpnSplitWorkspace out;
    out.gate = allocator.alloc_bytes(bytes, 256);
    out.up   = allocator.alloc_bytes(bytes, 256);
    return out;
}

template <class Geometry>
std::size_t qpn_split_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_qpn_split_workspace<Geometry>(layout, tokens);
    return layout.peak_bytes(1);
}

// Below this width groupwise's own crossover (the V100 implementation, Q4's CutlassSm70TensorCore
// route) puts the fused-dequant SIMT/QPN path ahead of paying CUTLASS's fixed T-independent
// dequant cost; above it the dequant-once-then-CUTLASS-Sm70-tensor-core route wins by not
// re-decoding the weight per chunk the way the QPN split route (still used below this width)
// does. Same threshold Q4's sibling route uses -- not independently re-measured for FP8 yet.
constexpr std::int32_t kVoltaCutlassMinT = 33;

template <class Allocator>
Tensor allocate_materialized_workspace(Allocator& allocator, std::int32_t rows, std::int32_t cols) {
    return allocator.alloc(DType::BF16, {rows, cols});
}

std::size_t materialized_workspace_bytes(std::int32_t rows, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_materialized_workspace(layout, rows, cols);
    return layout.peak_bytes(1);
}
#endif // NINFER_VOLTA_BUILD

template <class Geometry>
void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
                cudaStream_t stream) {
    constexpr std::int32_t kIntermediate = Geometry::kOutputRows / 2;
    constexpr std::int32_t kChunk        = kFp8LinearSmallTMax<Geometry>;
#ifdef NINFER_VOLTA_BUILD
    if (x.ne[1] >= kVoltaCutlassMinT) {
        auto scratch_scope = workspace.scope();
        Tensor gate_up = allocate_materialized_workspace(workspace, weight.n, x.ne[1]);
        fp8_linear_swiglu_cutlass_sm70_launch(x, weight, gate_up, workspace, stream);
        silu_mul(gate_up.slice(0, 0, kIntermediate),
                 gate_up.slice(0, kIntermediate, kIntermediate), out, stream);
        return;
    }
#endif
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(out.data) +
                       static_cast<std::int64_t>(token_begin) * kIntermediate * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor output_chunk(output, DType::BF16, {kIntermediate, active});
        if (active == 1) {
            if constexpr (std::is_same_v<Geometry, Fp8MlpGateUpGeometry>) {
                fp8_linear_swiglu_decode_launch(input_chunk, weight, output_chunk, stream);
            } else {
                fp8_linear_swiglu_decode_launch_shard(input_chunk, weight, output_chunk, stream);
            }
#ifdef NINFER_VOLTA_BUILD
        } else if (fp8_linear_swiglu_qpn_split_supported(weight.n, weight.k, active)) {
            // Two independent QPN8 launches (one per weight half) into fp32 scratch, then a small
            // combine kernel -- not the fused single-kernel route (fp8_linear_swiglu_volta_qpn,
            // still built and correct, now unused in production) and not the unfused
            // linear()+silu_mul() composition, which fails this op's correctness test the same
            // way the NVFP4 sibling's did. Mirrors nvfp4_linear_swiglu_qpn_split.cuh exactly: the
            // fused kernel's doubled accumulator/decode registers cost more than sharing the
            // activation load saves. See the NVFP4 decode sweep.
            auto scope                 = workspace.scope();
            Fp8QpnSplitWorkspace scratch =
                allocate_qpn_split_workspace<Geometry>(workspace, active);
            fp8_linear_swiglu_qpn_split_launch(input_chunk, weight, output_chunk,
                                               reinterpret_cast<float*>(scratch.gate.data),
                                               reinterpret_cast<float*>(scratch.up.data), stream);
#endif
        } else {
            if constexpr (std::is_same_v<Geometry, Fp8MlpGateUpGeometry>) {
                fp8_linear_swiglu_small_t_launch(input_chunk, weight, output_chunk, stream);
            } else {
                fp8_linear_swiglu_small_t_launch_shard(input_chunk, weight, output_chunk, stream);
            }
        }
    }
}

} // namespace

template <class Geometry>
std::size_t capacity_bytes_impl(LinearPolicy policy, std::int32_t min_tokens,
                                std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 linear_swiglu workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    (void)resolve_route(policy, max_tokens);
    const bool interval_uses_a8 =
        policy == LinearPolicy::AllowA8 && (min_tokens == 1 || max_tokens >= 3);
    if (interval_uses_a8) {
        return fp8_a8_workspace_capacity_bytes(max_tokens, Geometry::kInputRows);
    }
#ifdef NINFER_VOLTA_BUILD
    // The split route is reused per chunk inside launch_a16's loop, so it never needs more than
    // one chunk's worth of scratch regardless of the overall T being dispatched. Above
    // kVoltaCutlassMinT, launch_a16 instead takes the single-shot CUTLASS route sized to the
    // interval's widest T; an interval straddling the threshold can reach either branch depending
    // on the actual call's T, so the capacity is the max of both, not either alone.
    if (policy == LinearPolicy::A16Only && max_tokens >= 2) {
        std::size_t need = qpn_split_workspace_bytes<Geometry>(
            std::min(max_tokens, kFp8LinearSmallTMax<Geometry>));
        if (max_tokens >= kVoltaCutlassMinT) {
            need = std::max(
                need, materialized_workspace_bytes(Geometry::kOutputRows, max_tokens) +
                          fp8_linear_swiglu_cutlass_workspace_bytes(
                              Geometry::kOutputRows, Geometry::kInputRows, max_tokens));
        }
        return need;
    }
#endif
    return 0;
}

std::size_t fp8_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                       std::int32_t min_tokens,
                                                       std::int32_t max_tokens) {
    return capacity_bytes_impl<Fp8MlpGateUpGeometry>(policy, min_tokens, max_tokens);
}

std::size_t fp8_linear_swiglu_shard_workspace_capacity_bytes(LinearPolicy policy,
                                                             std::int32_t min_tokens,
                                                             std::int32_t max_tokens) {
    return capacity_bytes_impl<Fp8MlpGateUpTp2ColumnGeometry>(policy, min_tokens, max_tokens);
}

void fp8_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Fp8LinearSwiGluRoute::A16) {
        launch_a16<Fp8MlpGateUpGeometry>(x, weight, out, workspace, stream);
        return;
    }
    fp8_linear_swiglu_a8_launch(x, weight, out, workspace, stream);
}

void fp8_linear_swiglu_dispatch_shard(const Tensor& x, const Weight& weight, Tensor& out,
                                      LinearPolicy policy, WorkspaceArena* workspace,
                                      cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Fp8LinearSwiGluRoute::A16) {
        if (workspace == nullptr && x.ne[1] == 1) {
            fp8_linear_swiglu_decode_launch_shard(x, weight, out, stream);
            return;
        }
        if (workspace == nullptr) {
            throw std::invalid_argument(
                "fp8 A16 linear_swiglu column-parallel requires caller workspace");
        }
        launch_a16<Fp8MlpGateUpTp2ColumnGeometry>(x, weight, out, *workspace, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument(
            "fp8 A8 linear_swiglu column-parallel requires caller workspace");
    }
    fp8_linear_swiglu_a8_launch_shard(x, weight, out, *workspace, stream);
}

} // namespace ninfer::ops::detail
