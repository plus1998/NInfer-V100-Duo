#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
void launch(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    using Schedule        = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse   = 1.0F / weight.weight_scale_divisor;
    auto* output          = static_cast<__nv_bfloat16*>(residual.data);
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse,
        Nvfp4AddResidualEpilogue{output, Geometry::kOutputRows},
        Nvfp4ContiguousOutput{output, Geometry::kOutputRows});
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream) {
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::Residual6144:
        launch<Nvfp4Residual6144Geometry>(x, weight, residual, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launch<Nvfp4Residual17408Geometry>(x, weight, residual, stream);
        return;
    case Nvfp4Problem::Residual6144Tp2Row:
        launch<Nvfp4Residual6144Tp2RowGeometry>(x, weight, residual, stream);
        return;
    case Nvfp4Problem::Residual17408Tp2Row:
        launch<Nvfp4Residual17408Tp2RowGeometry>(x, weight, residual, stream);
        return;
    case Nvfp4Problem::AttnInput:
    case Nvfp4Problem::GdnInput:
    case Nvfp4Problem::MlpGateUp:
    case Nvfp4Problem::AttnInputTp2Column:
    case Nvfp4Problem::GdnInputTp2Column:
    case Nvfp4Problem::MlpGateUpTp2Column:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

} // namespace ninfer::ops::detail
