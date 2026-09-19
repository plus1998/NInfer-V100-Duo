#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
struct Nvfp4AttentionInputOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t,
                                          float result) const {
        using Sections                     = Nvfp4AttnInputSections<Geometry>;
        constexpr std::int32_t kQueryRows  = Sections::kQueryRows;
        constexpr std::int32_t kKeyRows    = Sections::kKeyRows;
        constexpr std::int32_t kGateRows   = Sections::kQueryRows;
        constexpr std::int32_t kKeyBegin   = kQueryRows;
        constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
        constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;

        const __nv_bfloat16 result_bf16 = __float2bfloat16_rn(result);
        if (parent_row < kKeyBegin) {
            query[parent_row] = result_bf16;
        } else if (parent_row < kGateBegin) {
            key[parent_row - kKeyBegin] = result_bf16;
        } else if (parent_row < kValueBegin) {
            gate[parent_row - kGateBegin] = result_bf16;
        } else {
            value[parent_row - kValueBegin] = result_bf16;
        }
    }
};

template <class Geometry>
void decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                   Tensor& v, cudaStream_t stream) {
    using Schedule = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;
    using Sections = Nvfp4AttnInputSections<Geometry>;
    static_assert((Sections::kQueryRows % 128) == 0);
    static_assert((Sections::kKeyRows % 128) == 0);

    const Nvfp4AttentionInputOutput<Geometry> output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    constexpr int kBlocks              = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse_weight_divisor,
        Nvfp4IdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream) {
    decode_launch<Nvfp4AttnInputGeometry>(x, weight, q, gate, k, v, stream);
}

void nvfp4_attn_input_decode_launch_shard(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    decode_launch<Nvfp4AttnInputTp2ColumnGeometry>(x, weight, q, gate, k, v, stream);
}

} // namespace ninfer::ops::detail
