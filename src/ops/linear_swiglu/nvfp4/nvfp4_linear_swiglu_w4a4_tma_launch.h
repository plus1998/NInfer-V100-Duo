#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t* activation_codes,
                                         const std::uint8_t* activation_scales,
                                         const std::uint8_t* weight_codes,
                                         const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                         std::int32_t tokens, float alpha, cudaStream_t stream);

// tp2 column-shard form (Nvfp4MlpGateUpTp2ColumnGeometry, 17408x5120 -- see
// src/ops/linear/nvfp4/nvfp4_config.h). Same descriptor/kernel mechanics at the halved N.
void launch_nvfp4_linear_swiglu_w4a4_tma_shard(const std::uint8_t* activation_codes,
                                               const std::uint8_t* activation_scales,
                                               const std::uint8_t* weight_codes,
                                               const std::uint8_t* weight_scales,
                                               __nv_bfloat16* output, std::int32_t tokens,
                                               float alpha, cudaStream_t stream);

} // namespace ninfer::ops::detail
