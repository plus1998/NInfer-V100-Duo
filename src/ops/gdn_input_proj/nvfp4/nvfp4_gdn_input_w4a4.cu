#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include <type_traits>

namespace ninfer::ops::detail {
namespace {

using M32N64                      = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128                     = Nvfp4W4a4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128                     = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined           = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident            = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;
constexpr std::int32_t kTmaBlockM = 256;

template <class Geometry, class Output, class Schedule>
void launch_gemm(const Weight& weight, Tensor& qkv, Tensor& z, Nvfp4W4a4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    nvfp4_w4a4_mma_kernel<Geometry, Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
        Output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)});
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Output>
void launch_w4a4(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                 Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    launch_nvfp4_w4a4_quantize(x, weight, workspace, stream);
    const std::int32_t tokens = x.ne[1];
    if (tokens >= 1024 && (tokens % kTmaBlockM) == 0) {
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        if constexpr (std::is_same_v<Geometry, Nvfp4GdnInputGeometry>) {
            launch_nvfp4_w4a4_tma_gdn(
                workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales),
                static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data), tokens,
                alpha, stream);
        } else {
            launch_nvfp4_w4a4_tma_gdn_shard(
                workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales),
                static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data), tokens,
                alpha, stream);
        }
        return;
    }
    if (tokens <= 64) {
        launch_gemm<Geometry, Output, M32N64>(weight, qkv, z, workspace, tokens, stream);
    } else if (tokens <= 96) {
        launch_gemm<Geometry, Output, M32N128>(weight, qkv, z, workspace, tokens, stream);
    } else if (tokens <= 128) {
        launch_gemm<Geometry, Output, M128N128Pipelined>(weight, qkv, z, workspace, tokens, stream);
    } else if (tokens <= 192) {
        launch_gemm<Geometry, Output, M64N128>(weight, qkv, z, workspace, tokens, stream);
    } else {
        launch_gemm<Geometry, Output, M128N128Resident>(weight, qkv, z, workspace, tokens, stream);
    }
}

} // namespace

void nvfp4_gdn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    launch_w4a4<Nvfp4GdnInputGeometry, Nvfp4GdnInputOutput>(x, weight, qkv, z, workspace, stream);
}

// The tp2 column shard. TMA descriptors require BlockN=128 to divide every section's row
// count (verified: 1024%128==0, 3072%128==0, just as attn_input_proj's own shard satisfies
// 3072%128==0 and 512%128==0) -- the shard is TMA-capable too.
void nvfp4_gdn_input_w4a4_launch_shard(const Tensor& x, const Weight& weight, Tensor& qkv,
                                       Tensor& z, Nvfp4W4a4Workspace workspace,
                                       cudaStream_t stream) {
    launch_w4a4<Nvfp4GdnInputTp2ColumnGeometry,
               Nvfp4GdnInputShardOutput<Nvfp4GdnInputTp2ColumnGeometry>>(x, weight, qkv, z,
                                                                        workspace, stream);
}

} // namespace ninfer::ops::detail
