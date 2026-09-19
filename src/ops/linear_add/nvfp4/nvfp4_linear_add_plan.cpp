#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h"
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4LinearAddRoute : std::uint8_t {
    A16,
#ifdef NINFER_VOLTA_BUILD
    LinearThenAdd,
#endif
    W4A4,
};

// The 6144/17408 families each admit their tp1 extent and its tp2 row-parallel half (3072/8704):
// the shard is the same kernel template at a halved K, so it inherits the parent's measured A16/
// W4A4 crossover rather than getting one re-measured for it (see nvfp4_config.h's
// Nvfp4LinearSmallTProductionSchedule specializations for the shard geometries, which do the same
// inheritance at the schedule level).
bool is_6144_family(std::int32_t input_rows) { return input_rows == 6144 || input_rows == 3072; }
bool is_17408_family(std::int32_t input_rows) { return input_rows == 17408 || input_rows == 8704; }

Nvfp4LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                  LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0 || output_rows != 5120 ||
        !(is_6144_family(input_rows) || is_17408_family(input_rows))) {
        throw std::invalid_argument("nvfp4 linear_add: unsupported shape");
    }
    if (policy == LinearPolicy::A16Only) {
#ifdef NINFER_VOLTA_BUILD
        // The load-time-prepacked MLP down projection is consumed by QPN2 at every decode width;
        // the fused row-major kernels cannot read that layout.
        return Nvfp4LinearAddRoute::LinearThenAdd;
#else
        return Nvfp4LinearAddRoute::A16;
#endif
    }
    if (policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 linear_add: unsupported policy");
    }
    const std::int32_t first_w4a4 = is_6144_family(input_rows) ? 7 : 8;
    return tokens >= first_w4a4 ? Nvfp4LinearAddRoute::W4A4 : Nvfp4LinearAddRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    constexpr std::int32_t kChunk = kNvfp4LastSmallT;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(residual.data) +
                       static_cast<std::int64_t>(token_begin) * weight.n * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor residual_chunk(output, DType::BF16, {weight.n, active});
        if (active == 1) {
            nvfp4_linear_add_decode_launch(input_chunk, weight, residual_chunk, stream);
        } else {
            nvfp4_linear_add_small_t_launch(input_chunk, weight, residual_chunk, stream);
        }
    }
}

#ifdef NINFER_VOLTA_BUILD
template <class Allocator>
Tensor allocate_projected(Allocator& allocator, std::int32_t output_rows, std::int32_t tokens) {
    return allocator.alloc(DType::BF16, {output_rows, tokens}, 256);
}

void launch_linear_then_add(const Tensor& x, const Weight& weight, Tensor& residual,
                            WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope         = workspace.scope();
    Tensor projected    = allocate_projected(workspace, weight.n, x.ne[1]);
    if (x.ne[1] >= 33) {
        nvfp4_cutlass_sm70_launch(x, weight, projected, workspace, stream);
    } else {
        const std::size_t linear_bytes = std::max<std::size_t>(
            linear_workspace_capacity_bytes(QType::NVFP4, weight.n, weight.k,
                                            kNvfp4InternalPolicy, x.ne[1], x.ne[1]),
            256);
        WorkspaceArena linear_workspace(workspace.alloc_bytes(linear_bytes, 256));
        linear(x, weight, projected, kNvfp4InternalPolicy, linear_workspace, stream);
    }
    residual_add(projected, residual, stream);
}

std::size_t linear_then_add_workspace_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                            std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected(layout, output_rows, tokens);
    const std::size_t linear_bytes = tokens >= 33
        ? nvfp4_cutlass_sm70_workspace_bytes(output_rows, input_rows, tokens)
        : std::max<std::size_t>(
              linear_workspace_capacity_bytes(QType::NVFP4, output_rows, input_rows,
                                              kNvfp4InternalPolicy, tokens, tokens),
              256);
    (void)layout.alloc_bytes(linear_bytes, 256);
    return layout.peak_bytes(1);
}
#endif // NINFER_VOLTA_BUILD

} // namespace

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                      std::int32_t input_rows, LinearPolicy policy,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_add workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    const Nvfp4LinearAddRoute route = resolve_route(output_rows, input_rows, policy, max_tokens);
    if (route == Nvfp4LinearAddRoute::W4A4) {
        return nvfp4_w4a4_workspace_capacity_bytes(max_tokens, input_rows);
    }
#ifdef NINFER_VOLTA_BUILD
    if (route == Nvfp4LinearAddRoute::LinearThenAdd) {
        return linear_then_add_workspace_bytes(output_rows, input_rows, max_tokens);
    }
#endif
    return 0;
}

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream) {
    const Nvfp4LinearAddRoute route = resolve_route(weight.n, weight.k, policy, x.ne[1]);
    if (route == Nvfp4LinearAddRoute::A16) {
        launch_a16(x, weight, residual, stream);
        return;
    }
#ifdef NINFER_VOLTA_BUILD
    if (route == Nvfp4LinearAddRoute::LinearThenAdd) {
        if (!workspace) throw std::invalid_argument("nvfp4 linear_add requires workspace");
        launch_linear_then_add(x, weight, residual, *workspace, stream);
        return;
    }
#endif
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 linear_add requires caller workspace");
    }
    auto scope                       = workspace->scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_linear_add_w4a4_launch(x, weight, residual, scratch, stream);
}

} // namespace ninfer::ops::detail
