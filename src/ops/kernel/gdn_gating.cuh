#pragma once

// Elementwise GDN gate preparation; the projection Op also uses FP32 input and TP2 heads.
// Transcendentals use fp32 CUDA math functions, not polynomial approximations.

#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops {

template <class Input>
__global__ void gdn_gating_kernel(const Input* a, const Input* b,
                                  const float* A_log, const float* dt_bias, float* g, float* beta,
                                  std::int64_t n, int heads) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const int h    = static_cast<int>(i % heads);
        const float av = static_cast<float>(a[i]);
        const float bv = static_cast<float>(b[i]);
        const float sp = softplus(av + dt_bias[h]);
        g[i]           = -expf(A_log[h]) * sp;
        beta[i]        = sigmoid(bv);
    }
}

} // namespace ninfer::ops
