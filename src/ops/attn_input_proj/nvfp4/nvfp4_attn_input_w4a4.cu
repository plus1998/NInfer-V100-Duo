#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
struct Nvfp4W4a4AttentionOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        using Sections                     = Nvfp4AttnInputSections<Geometry>;
        constexpr std::int32_t kQueryRows  = Sections::kQueryRows;
        constexpr std::int32_t kKeyRows    = Sections::kKeyRows;
        constexpr std::int32_t kGateRows   = Sections::kQueryRows;
        constexpr std::int32_t kKeyBegin   = kQueryRows;
        constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
        constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;
        if (parent_row < kKeyBegin) {
            return query + static_cast<std::int64_t>(token) * kQueryRows + parent_row;
        }
        if (parent_row < kGateBegin) {
            return key + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kKeyBegin;
        }
        if (parent_row < kValueBegin) {
            return gate + static_cast<std::int64_t>(token) * kGateRows + parent_row - kGateBegin;
        }
        return value + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kValueBegin;
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

using M32N64                      = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128                     = Nvfp4W4a4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128                     = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined           = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident            = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;
constexpr std::int32_t kTmaBlockM = 256;

template <class Geometry, class Schedule>
void launch_gemm(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                 Nvfp4W4a4Workspace workspace, std::int32_t tokens, cudaStream_t stream) {
    static_assert((Nvfp4AttnInputSections<Geometry>::kQueryRows % 128) == 0);
    static_assert((Nvfp4AttnInputSections<Geometry>::kKeyRows % 128) == 0);
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const Nvfp4W4a4AttentionOutput<Geometry> output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    nvfp4_w4a4_mma_kernel<Geometry, Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
        output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void w4a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                 Tensor& v, Nvfp4W4a4Workspace workspace, cudaStream_t stream,
                 void (*tma)(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                             const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
                             __nv_bfloat16*, std::int32_t, float, cudaStream_t)) {
    launch_nvfp4_w4a4_quantize(x, weight, workspace, stream);
    const std::int32_t tokens = x.ne[1];
    if (tokens >= 1024 && (tokens % kTmaBlockM) == 0) {
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        tma(workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
           static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(q.data),
           static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(k.data),
           static_cast<__nv_bfloat16*>(v.data), tokens, alpha, stream);
    } else if (tokens <= 64) {
        launch_gemm<Geometry, M32N64>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 96) {
        launch_gemm<Geometry, M32N128>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 128) {
        launch_gemm<Geometry, M128N128Pipelined>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 192) {
        launch_gemm<Geometry, M64N128>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 384) {
        launch_gemm<Geometry, M128N128Resident>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 512) {
        launch_gemm<Geometry, M128N128Pipelined>(weight, q, gate, k, v, workspace, tokens, stream);
    } else {
        launch_gemm<Geometry, M128N128Resident>(weight, q, gate, k, v, workspace, tokens, stream);
    }
}

} // namespace

void nvfp4_attn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, Nvfp4W4a4Workspace workspace,
                                  cudaStream_t stream) {
    w4a4_launch<Nvfp4AttnInputGeometry>(x, weight, q, gate, k, v, workspace, stream,
                                        &launch_nvfp4_w4a4_tma_attention);
}

void nvfp4_attn_input_w4a4_launch_shard(const Tensor& x, const Weight& weight, Tensor& q,
                                        Tensor& gate, Tensor& k, Tensor& v,
                                        Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    w4a4_launch<Nvfp4AttnInputTp2ColumnGeometry>(x, weight, q, gate, k, v, workspace, stream,
                                                 &launch_nvfp4_w4a4_tma_attention_shard);
}

} // namespace ninfer::ops::detail
