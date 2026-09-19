#pragma once

#include "ops/common/memory.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Row layout within a Q|K|V|Z fused GDN input_projection object: kKeyRows describes BOTH Q and K
// (equal-sized, key_dim), kValueRows describes BOTH V and Z (equal-sized, value_dim) -- verified
// from src/targets/qwen3_6_27b/impl/config.h (key_dim = gdn_key_heads*gdn_key_head_dim,
// value_dim = gdn_value_heads*gdn_value_head_dim) and bindings.cpp's plan_for
// ("gdn/query_key_value_z") append_column_block call order. Tp1 uses <2048,6144>; the tp2 column
// shard (each device's own head-local half: 8 of 16 key heads, 24 of 48 value heads) uses
// <1024,3072> -- Nvfp4GdnInputTp2ColumnGeometry (<8192,5120>, registered in
// src/ops/linear/nvfp4/nvfp4_config.h) already proves the combined row count is
// 128/64-tile-aligned.
template <class Geometry>
struct Nvfp4GdnInputSections;

template <>
struct Nvfp4GdnInputSections<Nvfp4GdnInputGeometry> {
    static constexpr std::int32_t kKeyRows   = 2048;
    static constexpr std::int32_t kValueRows = 6144;
};

// The tp2 column shard, registered as Nvfp4GdnInputTp2ColumnGeometry in nvfp4_config.h.
template <>
struct Nvfp4GdnInputSections<Nvfp4GdnInputTp2ColumnGeometry> {
    static constexpr std::int32_t kKeyRows   = 1024;
    static constexpr std::int32_t kValueRows = 3072;
};

// The tp1-named struct (unparameterized) is kept exactly as it was -- every existing tp1 call site
// (nvfp4_gdn_input_{decode,small_t,w4a4}.cu, nvfp4_w4a4_tma.cu) continues to build the identical
// Nvfp4GdnInputOutput{qkv,z} aggregate it always has, unmodified. The shard uses the new templated
// sibling below instead, so tp1 behavior is untouched by construction.
struct Nvfp4GdnInputOutput {
    static constexpr std::int32_t kQkvRows = 10240;
    static constexpr std::int32_t kZRows   = 6144;

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

static_assert((Nvfp4GdnInputOutput::kQkvRows % 128) == 0);
static_assert((Nvfp4GdnInputOutput::kZRows % 128) == 0);

// The shard sibling. Writes qkv[Geometry-derived kQkvRows,T] (this device's Q|K|V,
// shard-local offsets [0,kKeyRows) | [kKeyRows,2*kKeyRows) | [2*kKeyRows,kQkvRows)) and
// z[kValueRows,T] (this device's Z) -- exactly the tp1 struct's own 2-tensor convention,
// Geometry-parameterized. The output contract the GDN core reads is stated in
// include/ninfer/ops/gdn_input_proj.h.
template <class Geometry>
struct Nvfp4GdnInputShardOutput {
    static constexpr std::int32_t kKeyRows   = Nvfp4GdnInputSections<Geometry>::kKeyRows;
    static constexpr std::int32_t kValueRows = Nvfp4GdnInputSections<Geometry>::kValueRows;
    static constexpr std::int32_t kQkvRows   = 2 * kKeyRows + kValueRows;
    static constexpr std::int32_t kZRows     = kValueRows;

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
