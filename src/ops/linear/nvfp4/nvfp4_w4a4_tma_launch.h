#pragma once

#include "ops/linear/nvfp4/nvfp4_config.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void launch_nvfp4_w4a4_tma_linear(Nvfp4Problem problem, const std::uint8_t* activation_codes,
                                  const std::uint8_t* activation_scales,
                                  const std::uint8_t* weight_codes,
                                  const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                  std::int32_t tokens, float alpha, cudaStream_t stream);

void launch_nvfp4_w4a4_tma_attention(const std::uint8_t* activation_codes,
                                     const std::uint8_t* activation_scales,
                                     const std::uint8_t* weight_codes,
                                     const std::uint8_t* weight_scales, __nv_bfloat16* query,
                                     __nv_bfloat16* gate, __nv_bfloat16* key, __nv_bfloat16* value,
                                     std::int32_t tokens, float alpha, cudaStream_t stream);

// The tp2 column-shard sibling of launch_nvfp4_w4a4_tma_attention above, instantiated at
// Nvfp4AttnInputTp2ColumnGeometry (each device's 7168-row half: 3072 query | 512 key | 3072 gate |
// 512 value).
void launch_nvfp4_w4a4_tma_attention_shard(const std::uint8_t* activation_codes,
                                           const std::uint8_t* activation_scales,
                                           const std::uint8_t* weight_codes,
                                           const std::uint8_t* weight_scales, __nv_bfloat16* query,
                                           __nv_bfloat16* gate, __nv_bfloat16* key,
                                           __nv_bfloat16* value, std::int32_t tokens, float alpha,
                                           cudaStream_t stream);

void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t* activation_codes,
                               const std::uint8_t* activation_scales,
                               const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                               __nv_bfloat16* qkv, __nv_bfloat16* z, std::int32_t tokens,
                               float alpha, cudaStream_t stream);

// The tp2 column-shard sibling of launch_nvfp4_w4a4_tma_gdn above, instantiated at
// Nvfp4GdnInputTp2ColumnGeometry (each device's 8192-row half: qkv[5120,T] = 1024 query | 1024
// key | 3072 value packed, z[3072,T]).
void launch_nvfp4_w4a4_tma_gdn_shard(const std::uint8_t* activation_codes,
                                     const std::uint8_t* activation_scales,
                                     const std::uint8_t* weight_codes,
                                     const std::uint8_t* weight_scales, __nv_bfloat16* qkv,
                                     __nv_bfloat16* z, std::int32_t tokens, float alpha,
                                     cudaStream_t stream);

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4Problem problem, const std::uint8_t* activation_codes,
                                      const std::uint8_t* activation_scales,
                                      const std::uint8_t* weight_codes,
                                      const std::uint8_t* weight_scales, __nv_bfloat16* residual,
                                      std::int32_t tokens, float alpha, cudaStream_t stream);

} // namespace ninfer::ops::detail
