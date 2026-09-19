#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/launcher/kernel_attr_once.h"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

static_assert((kFp8AttnInputQueryRows %
              Fp8LinearA8ProductionSchedule<Fp8AttnInputGeometry>::Type::kBlockRows) == 0);
static_assert((kFp8AttnInputKeyRows %
              Fp8LinearA8ProductionSchedule<Fp8AttnInputGeometry>::Type::kBlockRows) == 0);
static_assert((kFp8AttnInputGateRows %
              Fp8LinearA8ProductionSchedule<Fp8AttnInputGeometry>::Type::kBlockRows) == 0);
static_assert(
    (Fp8AttentionInputShardOutput<Fp8AttnInputTp2ColumnGeometry>::kQueryRows %
     Fp8LinearA8ProductionSchedule<Fp8AttnInputTp2ColumnGeometry>::Type::kBlockRows) == 0);
static_assert(
    (Fp8AttentionInputShardOutput<Fp8AttnInputTp2ColumnGeometry>::kKeyRows %
     Fp8LinearA8ProductionSchedule<Fp8AttnInputTp2ColumnGeometry>::Type::kBlockRows) == 0);

template <class Geometry, class Output, bool FullTokens>
void launch_mma(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                Fp8A8Workspace workspace, std::int32_t tokens, cudaStream_t stream) {
    using Schedule           = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;
    constexpr int kRowTiles  = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles    = (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const int blocks         = kRowTiles * token_tiles;
    const Output output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };

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
void launch_a8(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
              Tensor& v, Fp8A8Workspace workspace, cudaStream_t stream) {
    using Schedule = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    if ((x.ne[1] % Schedule::kBlockTokens) == 0) {
        launch_mma<Geometry, Output, true>(weight, q, gate, k, v, workspace, x.ne[1], stream);
    } else {
        launch_mma<Geometry, Output, false>(weight, q, gate, k, v, workspace, x.ne[1], stream);
    }
}

} // namespace

void fp8_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_a8<Fp8AttnInputGeometry, Fp8AttentionInputOutput>(x, weight, q, gate, k, v, workspace,
                                                             stream);
}

// The tp2 column shard.
void fp8_attn_input_a8_launch_shard(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, Fp8A8Workspace workspace,
                                    cudaStream_t stream) {
    launch_a8<Fp8AttnInputTp2ColumnGeometry, Fp8AttentionInputShardOutput<Fp8AttnInputTp2ColumnGeometry>>(
        x, weight, q, gate, k, v, workspace, stream);
}

} // namespace ninfer::ops::detail
