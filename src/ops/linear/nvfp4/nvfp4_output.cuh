#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Nvfp4IdentityEpilogue {
    __device__ __forceinline__ float apply(std::int32_t, std::int32_t, float value) const {
        return value;
    }
};

struct Nvfp4ContiguousOutput {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        data[static_cast<std::int64_t>(token) * rows + parent_row] = __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        auto* destination = data + static_cast<std::int64_t>(token) * rows + parent_row;
        store_vec(destination, values);
    }
};

// Writes the mma accumulator straight through, no BF16 round. Split-projection SwiGLU uses this
// for its gate pass; its up pass reads that fp32 plane from a fused output policy and performs the
// activation before the only BF16 round. Splitting keeps each launch at QPN2's tuned
// register/occupancy profile instead of the fused kernel's doubled one.
struct Nvfp4Fp32ContiguousOutput {
    float* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        data[static_cast<std::int64_t>(token) * rows + parent_row] = value;
    }
};

} // namespace ninfer::ops::detail
