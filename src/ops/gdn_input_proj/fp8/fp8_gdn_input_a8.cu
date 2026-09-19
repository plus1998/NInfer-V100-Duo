#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/launcher/kernel_attr_once.h"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <class Geometry, class Output, bool FullTokens>
void launch_mma(const Weight& weight, Tensor& qkv, Tensor& z, Fp8A8Workspace workspace,
                std::int32_t tokens, cudaStream_t stream) {
    using Schedule           = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;
    constexpr int kRowTiles  = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles    = (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const int blocks         = kRowTiles * token_tiles;
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};

    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        ensure_func_attr_per_device(
            fp8_mma_kernel<Geometry, Schedule, FullTokens, Fp8IdentityEpilogue, Output>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
    }
    fp8_mma_kernel<Geometry, Schedule, FullTokens>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), tokens, Fp8IdentityEpilogue{},
            output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Output>
void launch_a8(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
              Fp8A8Workspace workspace, cudaStream_t stream) {
    using Schedule = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    if ((x.ne[1] % Schedule::kBlockTokens) == 0) {
        launch_mma<Geometry, Output, true>(weight, qkv, z, workspace, x.ne[1], stream);
    } else {
        launch_mma<Geometry, Output, false>(weight, qkv, z, workspace, x.ne[1], stream);
    }
}

static_assert((Fp8GdnInputOutput::kQkvRows %
              Fp8LinearA8ProductionSchedule<Fp8GdnInputGeometry>::Type::kBlockRows) == 0);
static_assert((Fp8GdnInputOutput::kZRows %
              Fp8LinearA8ProductionSchedule<Fp8GdnInputGeometry>::Type::kBlockRows) == 0);
static_assert(
    (Fp8GdnInputShardOutput<Fp8GdnInputTp2ColumnGeometry>::kQkvRows %
     Fp8LinearA8ProductionSchedule<Fp8GdnInputTp2ColumnGeometry>::Type::kBlockRows) == 0);
static_assert(
    (Fp8GdnInputShardOutput<Fp8GdnInputTp2ColumnGeometry>::kZRows %
     Fp8LinearA8ProductionSchedule<Fp8GdnInputTp2ColumnGeometry>::Type::kBlockRows) == 0);

} // namespace

void fp8_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_a8<Fp8GdnInputGeometry, Fp8GdnInputOutput>(x, weight, qkv, z, workspace, stream);
}

// The tp2 column shard.
void fp8_gdn_input_a8_launch_shard(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_a8<Fp8GdnInputTp2ColumnGeometry, Fp8GdnInputShardOutput<Fp8GdnInputTp2ColumnGeometry>>(
        x, weight, qkv, z, workspace, stream);
}

} // namespace ninfer::ops::detail
