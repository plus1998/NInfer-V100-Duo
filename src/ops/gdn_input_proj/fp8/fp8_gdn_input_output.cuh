#pragma once

#include "ops/common/memory.cuh"
#include "ops/linear/fp8/fp8_config.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The tp1-named struct is kept exactly as it was -- every existing tp1 call site
// (fp8_gdn_input_{decode,small_t,a8}.cu) continues to build the identical
// Fp8GdnInputOutput{qkv,z} aggregate it always has, unmodified.
struct Fp8GdnInputOutput {
    static constexpr std::int32_t kQueryRows = 2048;
    static constexpr std::int32_t kKeyRows   = 2048;
    static constexpr std::int32_t kValueRows = 6144;
    static constexpr std::int32_t kQkvRows   = kQueryRows + kKeyRows + kValueRows;
    static constexpr std::int32_t kZRows     = 6144;
    static constexpr std::int32_t kRows      = kQkvRows + kZRows;

    __nv_bfloat16* qkv;
    __nv_bfloat16* z;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kQkvRows) {
            return qkv + static_cast<std::int64_t>(token) * kQkvRows + parent_row;
        }
        return z + static_cast<std::int64_t>(token) * kZRows + parent_row - kQkvRows;
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        *destination(parent_row, token) = __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

static_assert(Fp8GdnInputOutput::kRows == 16384);
static_assert((Fp8GdnInputOutput::kQkvRows % 128) == 0);
static_assert((Fp8GdnInputOutput::kZRows % 128) == 0);

// The tp2 column shard's own 2-tensor output (this device's Q|K|V packed into qkv,
// this device's Z into z), Geometry-parameterized -- exactly the same interior boundary rule
// derived for NVFP4 in nvfp4_gdn_input_output.cuh (Q and K are both key_dim rows, V and Z are
// both value_dim rows). Only the shard Geometry is registered here; FP8 is not optional for this
// object -- it is the flagship Qwen38Nvfp4 profile's own binding for `gdn/query_key_value_z`.
template <class Geometry>
struct Fp8GdnInputShardOutput {
    static constexpr std::int32_t kKeyRows   = 1024;
    static constexpr std::int32_t kValueRows = 3072;
    static constexpr std::int32_t kQkvRows   = 2 * kKeyRows + kValueRows;
    static constexpr std::int32_t kZRows     = kValueRows;

    static_assert(Geometry::kOutputRows == kQkvRows + kZRows);

    __nv_bfloat16* qkv;
    __nv_bfloat16* z;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kQkvRows) {
            return qkv + static_cast<std::int64_t>(token) * kQkvRows + parent_row;
        }
        return z + static_cast<std::int64_t>(token) * kZRows + parent_row - kQkvRows;
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        *destination(parent_row, token) = __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

} // namespace ninfer::ops::detail
