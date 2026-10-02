#include "ops/dynamic_grouped_conv/w8/w8_dynamic_grouped_conv_add_plan.h"

#include "ops/dynamic_grouped_conv/volta/dflash2_dynamic_conv_volta.h"
#include "ops/linear/w8/w8_launch.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden = 5120;

void require_profile(int input_rows, int width, int batch) {
    if (input_rows != 4096 && input_rows != 17408)
        throw std::invalid_argument("linear dynamic grouped conv add: C must be 4096 or 17408");
    if (width < 2 || width > 16 || batch < 1 || batch > 8)
        throw std::invalid_argument("linear dynamic grouped conv add: invalid W/B profile");
}

} // namespace

std::size_t w8_linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
    int input_rows, int min_width, int max_width, int min_batch, int max_batch) {
    if (min_width < 2 || max_width > 16 || min_width > max_width || min_batch < 1 ||
        max_batch > 8 || min_batch > max_batch)
        throw std::invalid_argument(
            "linear dynamic grouped conv add workspace: invalid W/B interval");
    require_profile(input_rows, min_width, min_batch);
    // The projected BF16 [5120, W*B] materialization buffer.
    return static_cast<std::size_t>(kHidden) * max_width * max_batch * sizeof(std::uint16_t);
}

const char* w8_linear_dynamic_grouped_conv_add_route_name(int input_rows, int width, int batch) {
    require_profile(input_rows, width, batch);
    return "dynamic_grouped_conv_add.w8.sm70.materialized_bf16";
}

void w8_linear_dynamic_grouped_conv_add_dispatch(const Tensor& x, const Weight& weight,
                                                 const Tensor& base, const Tensor& delta,
                                                 Tensor& residual, WorkspaceArena& workspace,
                                                 cudaStream_t stream) {
    const int width   = x.ne[1];
    const int batch   = x.ne[2];
    const int columns = width * batch;
    require_profile(x.ne[0], width, batch);

    auto scope       = workspace.scope();
    Tensor projected = workspace.alloc(DType::BF16, {kHidden, columns});
    const Tensor flat = x.view({x.ne[0], columns});
    if (w8_volta_mma_supported(kHidden, x.ne[0], columns)) {
        launch_w8_volta_mma(flat, weight, projected, stream);
    } else {
        throw std::invalid_argument("linear dynamic grouped conv add: unsupported Volta W8 shape");
    }

    dflash2_dynamic_conv_finish_add_launch(static_cast<const __nv_bfloat16*>(projected.data),
                                           static_cast<const __nv_bfloat16*>(delta.data),
                                           static_cast<const __nv_bfloat16*>(base.data), width,
                                           batch, static_cast<__nv_bfloat16*>(residual.data),
                                           stream);
}

} // namespace ninfer::ops::detail
