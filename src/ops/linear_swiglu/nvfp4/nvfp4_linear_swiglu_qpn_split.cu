#include "core/device.h"
#include "core/tensor.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_qpn_split.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ input,
                                    half* __restrict__ output, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { output[i] = __float2half(__bfloat162float(input[i])); }
}
} // namespace

bool nvfp4_linear_swiglu_qpn_split_supported(std::int32_t n, std::int32_t k,
                                              std::int32_t t) noexcept {
    return n > 0 && (n % 256) == 0 && nvfp4_volta_qpn_supported(n / 2, k, t);
}

void nvfp4_linear_swiglu_qpn_split_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          float* gate_scratch, void* activation_scratch,
                                          cudaStream_t stream) {
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    const std::int32_t kIntermediate = weight.n / 2;
    const std::int32_t kMTileOffset  = kIntermediate / 128;
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    auto* x_fp16 = static_cast<half*>(activation_scratch);
    const std::int64_t activation_count = static_cast<std::int64_t>(k) * t;
    bf16_to_fp16_kernel<<<static_cast<int>((activation_count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), x_fp16, activation_count);

    Weight gate_weight = weight;
    gate_weight.n      = kIntermediate;

    Weight up_weight = weight;
    up_weight.n       = kIntermediate;
    up_weight.qdata   = static_cast<const std::uint8_t*>(weight.qdata) +
                       static_cast<std::int64_t>(kIntermediate) * (k / 2);
    up_weight.scales = static_cast<const std::uint8_t*>(weight.scales) +
        (weight.layout == QuantLayout::VoltaQpnPrepacked
             ? static_cast<std::int64_t>(kIntermediate) * (k / 16)
             : static_cast<std::int64_t>(kMTileOffset) * (k / 64) * 512);

    launch_nvfp4_volta_qpn_with_fp16_activation(
        x, gate_weight, x_fp16, Nvfp4Fp32ContiguousOutput{gate_scratch, kIntermediate},
        kIntermediate, inverse_weight_divisor, stream);
    launch_nvfp4_volta_qpn_with_fp16_activation(
        x, up_weight, x_fp16,
        Nvfp4SwiGluFromGateOutput{gate_scratch, static_cast<__nv_bfloat16*>(out.data),
                                  kIntermediate},
        kIntermediate, inverse_weight_divisor, stream);
    CUDA_CHECK(cudaGetLastError());
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
