#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/layout.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h"
#endif

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4LinearSwiGluRoute {
    DecodeFusedA16,
    SmallTFusedA16,
    FusedW4A4,
    LinearW4A4Post,
    TmaFusedW4A4,
#ifdef NINFER_VOLTA_BUILD
    VoltaQpnFused,
    VoltaQpnSplit,
    VoltaCutlass,
#endif
};

constexpr std::int32_t kPrimaryT = 1024;

// A pure function of (policy, token count), not of N/K, so it is inherited unchanged by the tp2
// shard -- the same "route selection depends only on the family, not the halved extent" rule
// every split family here follows: tuning is inherited from the tp1 parent, never re-measured.
Nvfp4LinearSwiGluRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 linear_swiglu: T must be positive"); }
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 linear_swiglu admits only A16 or A4");
    }
    if (policy == LinearPolicy::A16Only) {
        if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
#ifdef NINFER_VOLTA_BUILD
        // T>=2 on Volta: two independent QPN2 passes (one per weight half). Gate is retained in
        // fp32 scratch and the up pass applies silu(gate)*up before the single BF16 round -- not
        // the fused single-kernel route (VoltaQpnFused, still built and correct, now unused in production)
        // and not the unfused linear()+silu_mul() composition (tried first, failed the
        // correctness test the same way the fused kernel's own precision note explains). Measured
        // against the fused kernel directly at T=4: 371.6 GB/s per QPN2 pass against the fused
        // kernel's 225.4 GB/s across the same total weight -- the fused kernel's doubled
        // accumulator and decode registers cost more than sharing the activation load saves.
        // Splitting also extends the fast, precision-safe route through T=32 (kNvfp4VoltaQpnMaxTokens),
        // where T=17..32 previously fell to the same untested-precision composition this route
        // replaces at T<=16. See the NVFP4 decode sweep.
        if (nvfp4_linear_swiglu_qpn_split_supported(
                Nvfp4MlpGateUpGeometry::kOutputRows,
                Nvfp4MlpGateUpGeometry::kInputRows, tokens)) {
            return Nvfp4LinearSwiGluRoute::VoltaQpnSplit;
        }
        return Nvfp4LinearSwiGluRoute::VoltaCutlass;
#else
        if (tokens <= 16) { return Nvfp4LinearSwiGluRoute::SmallTFusedA16; }
#endif
        // Above the fused A16 registration, take the same baseline the A4 policy uses for wide T:
        // linear() then silu_mul(). Despite the route's name nothing in it is W4A4 -- it inherits
        // whatever policy is passed down, and linear's A16 path chunks at kNvfp4LastSmallT, so it
        // serves any T. Without this an A16-only build has no MLP route at prefill width.
        return Nvfp4LinearSwiGluRoute::LinearW4A4Post;
    }
    if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
    if (tokens <= 4) { return Nvfp4LinearSwiGluRoute::SmallTFusedA16; }
    if (tokens <= 48) { return Nvfp4LinearSwiGluRoute::FusedW4A4; }
    if (tokens == kPrimaryT) { return Nvfp4LinearSwiGluRoute::TmaFusedW4A4; }
    return Nvfp4LinearSwiGluRoute::LinearW4A4Post;
}

struct Nvfp4LinearSwiGluWorkspace {
    Tensor projected;
    DeviceSpan linear;
};

template <class Geometry, class Allocator>
Nvfp4LinearSwiGluWorkspace allocate_baseline_workspace(Allocator& allocator, std::int32_t tokens) {
    Nvfp4LinearSwiGluWorkspace out;
    out.projected =
        allocator.alloc(DType::BF16, {Geometry::kOutputRows, tokens}, 256);
    // A16 linear needs no transient of its own, and an arena allocation must be nonzero, so the
    // sub-arena is floored. Both the capacity query and the dispatch route through here, so the
    // floor cannot desynchronize them.
    const std::size_t linear_bytes = std::max<std::size_t>(
        linear_workspace_capacity_bytes(QType::NVFP4, Geometry::kOutputRows,
                                        Geometry::kInputRows, kNvfp4InternalPolicy,
                                        tokens, tokens),
        256);
    out.linear = allocator.alloc_bytes(linear_bytes, 256);
    return out;
}

template <class Geometry, class Allocator>
Nvfp4W4a4Workspace allocate_fused_workspace(Allocator& allocator, std::int32_t tokens) {
    return allocate_nvfp4_w4a4_workspace(allocator, tokens, Geometry::kInputRows);
}

template <class Geometry>
std::size_t baseline_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_baseline_workspace<Geometry>(layout, tokens);
    return layout.peak_bytes(1);
}

#ifdef NINFER_VOLTA_BUILD
struct Nvfp4QpnSplitWorkspace {
    DeviceSpan gate;
    DeviceSpan activation;
};

template <class Geometry, class Allocator>
Nvfp4QpnSplitWorkspace allocate_qpn_split_workspace(Allocator& allocator, std::int32_t tokens) {
    constexpr std::int32_t kIntermediate = Geometry::kOutputRows / 2;
    const std::size_t bytes = static_cast<std::size_t>(kIntermediate) * tokens * sizeof(float);
    Nvfp4QpnSplitWorkspace out;
    out.gate = allocator.alloc_bytes(bytes, 256);
    out.activation = allocator.alloc_bytes(
        static_cast<std::size_t>(Geometry::kInputRows) * tokens * sizeof(std::uint16_t),
        256);
    return out;
}

template <class Geometry>
std::size_t qpn_split_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_qpn_split_workspace<Geometry>(layout, tokens);
    return layout.peak_bytes(1);
}

template <class Geometry>
std::size_t cutlass_route_workspace_bytes(std::int32_t tokens) {
    const std::size_t projected = static_cast<std::size_t>(
        Geometry::kOutputRows) * tokens * sizeof(std::uint16_t);
    return projected + nvfp4_cutlass_sm70_workspace_bytes(
                           Geometry::kOutputRows, Geometry::kInputRows, tokens);
}
#endif // NINFER_VOLTA_BUILD

template <class Geometry>
std::size_t fused_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_fused_workspace<Geometry>(layout, tokens);
    return layout.peak_bytes(1);
}

template <class Geometry>
std::size_t capacity_bytes_impl(LinearPolicy policy, std::int32_t min_tokens,
                                std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_swiglu workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    (void)resolve_route(policy, max_tokens);
    if (policy == LinearPolicy::A16Only) {
        // The fused A16 kernel (T==1 only) needs no transient; the baseline route needs its
        // projected plane; the Volta split route needs its gate fp32 scratch plane. Checked via
        // resolve_route rather than a hardcoded token threshold since the boundary differs by
        // build.
        const Nvfp4LinearSwiGluRoute route = resolve_route(policy, max_tokens);
        if (route == Nvfp4LinearSwiGluRoute::LinearW4A4Post) {
            return baseline_workspace_bytes<Geometry>(max_tokens);
        }
#ifdef NINFER_VOLTA_BUILD
        if (max_tokens == 1) { return qpn_split_workspace_bytes<Geometry>(1); }
        if (route == Nvfp4LinearSwiGluRoute::VoltaQpnSplit) {
            return qpn_split_workspace_bytes<Geometry>(max_tokens);
        }
        if (route == Nvfp4LinearSwiGluRoute::VoltaCutlass) {
            return cutlass_route_workspace_bytes<Geometry>(max_tokens);
        }
#endif
        return 0;
    }
    if (max_tokens <= 4) { return 0; }

    std::size_t maximum = 0;
    if (min_tokens <= 48 && max_tokens >= 5) {
        maximum = fused_workspace_bytes<Geometry>(std::min(max_tokens, 48));
    }
    if (min_tokens <= kPrimaryT && max_tokens >= kPrimaryT) {
        maximum = fused_workspace_bytes<Geometry>(kPrimaryT);
    }

    std::int32_t last_baseline = max_tokens;
    if (resolve_route(policy, last_baseline) == Nvfp4LinearSwiGluRoute::TmaFusedW4A4) {
        --last_baseline;
    }
    if (last_baseline >= std::max(min_tokens, 49)) {
        maximum = std::max(maximum, baseline_workspace_bytes<Geometry>(last_baseline));
    }
    return maximum;
}

// Bridges the Geometry-templated dispatch below to the two concretely-named launch symbols each
// .cu file exports (tp1 unchanged name, tp2 shard suffixed `_shard`). Selected at compile time, so
// dispatch_impl<Geometry> never pays a runtime branch for something the type already determines.
template <class Geometry>
void launch_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if constexpr (std::is_same_v<Geometry, Nvfp4MlpGateUpGeometry>) {
        nvfp4_linear_swiglu_decode_launch(x, weight, out, stream);
    } else {
        nvfp4_linear_swiglu_decode_launch_shard(x, weight, out, stream);
    }
}

template <class Geometry>
void launch_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if constexpr (std::is_same_v<Geometry, Nvfp4MlpGateUpGeometry>) {
        nvfp4_linear_swiglu_small_t_launch(x, weight, out, stream);
    } else {
        nvfp4_linear_swiglu_small_t_launch_shard(x, weight, out, stream);
    }
}

template <class Geometry>
void launch_w4a4(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
                 cudaStream_t stream) {
    if constexpr (std::is_same_v<Geometry, Nvfp4MlpGateUpGeometry>) {
        nvfp4_linear_swiglu_w4a4_launch(x, weight, out, workspace, stream);
    } else {
        nvfp4_linear_swiglu_w4a4_launch_shard(x, weight, out, workspace, stream);
    }
}

template <class Geometry>
void launch_tma(const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
                const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                __nv_bfloat16* output, std::int32_t tokens, float alpha, cudaStream_t stream) {
    if constexpr (std::is_same_v<Geometry, Nvfp4MlpGateUpGeometry>) {
        launch_nvfp4_linear_swiglu_w4a4_tma(activation_codes, activation_scales, weight_codes,
                                            weight_scales, output, tokens, alpha, stream);
    } else {
        launch_nvfp4_linear_swiglu_w4a4_tma_shard(activation_codes, activation_scales, weight_codes,
                                                  weight_scales, output, tokens, alpha, stream);
    }
}

// The one route body, parameterized on Geometry. `workspace` is nullable (see
// nvfp4_linear_swiglu_plan.h): the A16 routes never touch it; a W4A4-family route with a null
// workspace throws, matching ops::linear's own nvfp4_dispatch guard.
template <class Geometry>
void dispatch_impl(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena* workspace, cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    if (workspace && policy == LinearPolicy::A16Only && x.ne[1] == 1 &&
        nvfp4_linear_swiglu_qpn_split_supported(weight.n, weight.k, x.ne[1])) {
        auto scope                    = workspace->scope();
        Nvfp4QpnSplitWorkspace scratch =
            allocate_qpn_split_workspace<Geometry>(*workspace, x.ne[1]);
        nvfp4_linear_swiglu_qpn_split_launch(x, weight, out,
                                             reinterpret_cast<float*>(scratch.gate.data),
                                             scratch.activation.data, stream);
        return;
    }
#endif
    switch (resolve_route(policy, x.ne[1])) {
    case Nvfp4LinearSwiGluRoute::DecodeFusedA16:
        if (weight.layout == QuantLayout::VoltaQpnPrepacked) {
            throw std::invalid_argument(
                "prepacked nvfp4 linear_swiglu decode requires caller workspace");
        }
        launch_decode<Geometry>(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::SmallTFusedA16:
        launch_small_t<Geometry>(x, weight, out, stream);
        return;
#ifdef NINFER_VOLTA_BUILD
    case Nvfp4LinearSwiGluRoute::VoltaQpnFused:
        nvfp4_linear_swiglu_volta_qpn_launch(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::VoltaQpnSplit: {
        if (workspace == nullptr) {
            throw std::invalid_argument("nvfp4 linear_swiglu QPN requires caller workspace");
        }
        auto scope                       = workspace->scope();
        Nvfp4QpnSplitWorkspace scratch =
            allocate_qpn_split_workspace<Geometry>(*workspace, x.ne[1]);
        nvfp4_linear_swiglu_qpn_split_launch(x, weight, out,
                                             reinterpret_cast<float*>(scratch.gate.data),
                                             scratch.activation.data, stream);
        return;
    }
    case Nvfp4LinearSwiGluRoute::VoltaCutlass: {
        if (workspace == nullptr) {
            throw std::invalid_argument("nvfp4 linear_swiglu CUTLASS requires caller workspace");
        }
        auto scope = workspace->scope();
        Tensor projected = workspace->alloc(
            DType::BF16, {Geometry::kOutputRows, x.ne[1]}, 256);
        nvfp4_cutlass_sm70_launch(x, weight, projected, *workspace, stream);
        constexpr std::int32_t kIntermediate = Geometry::kOutputRows / 2;
        silu_mul(projected.slice(0, 0, kIntermediate),
                 projected.slice(0, kIntermediate, kIntermediate), out, stream);
        return;
    }
#endif
    case Nvfp4LinearSwiGluRoute::FusedW4A4:
        if (workspace == nullptr) {
            throw std::invalid_argument("nvfp4 linear_swiglu W4A4 requires caller workspace");
        }
        launch_w4a4<Geometry>(x, weight, out, *workspace, stream);
        return;
    case Nvfp4LinearSwiGluRoute::TmaFusedW4A4: {
        if (workspace == nullptr) {
            throw std::invalid_argument("nvfp4 linear_swiglu TMA W4A4 requires caller workspace");
        }
        auto scope                       = workspace->scope();
        const Nvfp4W4a4Workspace scratch = allocate_fused_workspace<Geometry>(*workspace, x.ne[1]);
        launch_nvfp4_w4a4_quantize(x, weight, scratch, stream);
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_tma<Geometry>(scratch.codes, scratch.scales,
                             static_cast<const std::uint8_t*>(weight.qdata),
                             static_cast<const std::uint8_t*>(weight.scales),
                             static_cast<__nv_bfloat16*>(out.data), x.ne[1], alpha, stream);
        return;
    }
    case Nvfp4LinearSwiGluRoute::LinearW4A4Post:
        break;
    }

    if (workspace == nullptr) {
        throw std::invalid_argument(
            "nvfp4 linear_swiglu materialized W4A4 requires caller workspace");
    }
    auto scope                         = workspace->scope();
    Nvfp4LinearSwiGluWorkspace scratch = allocate_baseline_workspace<Geometry>(*workspace, x.ne[1]);
    WorkspaceArena linear_workspace(scratch.linear);
    linear(x, weight, scratch.projected, LinearPolicy::AllowA4, linear_workspace, stream);
    constexpr std::int32_t kIntermediate = Geometry::kOutputRows / 2;
    silu_mul(scratch.projected.slice(0, 0, kIntermediate),
             scratch.projected.slice(0, kIntermediate, kIntermediate), out, stream);
}

} // namespace

std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                         std::int32_t min_tokens,
                                                         std::int32_t max_tokens) {
    return capacity_bytes_impl<Nvfp4MlpGateUpGeometry>(policy, min_tokens, max_tokens);
}

std::size_t nvfp4_linear_swiglu_shard_workspace_capacity_bytes(LinearPolicy policy,
                                                                std::int32_t min_tokens,
                                                                std::int32_t max_tokens) {
    return capacity_bytes_impl<Nvfp4MlpGateUpTp2ColumnGeometry>(policy, min_tokens, max_tokens);
}

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena& workspace,
                                  cudaStream_t stream) {
    dispatch_impl<Nvfp4MlpGateUpGeometry>(x, weight, out, policy, &workspace, stream);
}

void nvfp4_linear_swiglu_dispatch_shard(const Tensor& x, const Weight& weight, Tensor& out,
                                        LinearPolicy policy, WorkspaceArena* workspace,
                                        cudaStream_t stream) {
    dispatch_impl<Nvfp4MlpGateUpTp2ColumnGeometry>(x, weight, out, policy, workspace, stream);
}

} // namespace ninfer::ops::detail
