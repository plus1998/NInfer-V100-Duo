#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kFp8AttnInputQueryRows = 6144;
inline constexpr std::int32_t kFp8AttnInputKeyRows   = 1024;
inline constexpr std::int32_t kFp8AttnInputGateRows  = 6144;
inline constexpr std::int32_t kFp8AttnInputKeyBegin  = kFp8AttnInputQueryRows;
inline constexpr std::int32_t kFp8AttnInputGateBegin = kFp8AttnInputKeyBegin + kFp8AttnInputKeyRows;
inline constexpr std::int32_t kFp8AttnInputValueBegin =
    kFp8AttnInputGateBegin + kFp8AttnInputGateRows;

static_assert((kFp8AttnInputQueryRows % 8) == 0);
static_assert((kFp8AttnInputKeyRows % 8) == 0);
static_assert((kFp8AttnInputGateRows % 8) == 0);

struct Fp8AttentionInputOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kFp8AttnInputKeyBegin) {
            return query + static_cast<std::int64_t>(token) * kFp8AttnInputQueryRows + parent_row;
        }
        if (parent_row < kFp8AttnInputGateBegin) {
            return key + static_cast<std::int64_t>(token) * kFp8AttnInputKeyRows + parent_row -
                   kFp8AttnInputKeyBegin;
        }
        if (parent_row < kFp8AttnInputValueBegin) {
            return gate + static_cast<std::int64_t>(token) * kFp8AttnInputGateRows + parent_row -
                   kFp8AttnInputGateBegin;
        }
        return value + static_cast<std::int64_t>(token) * kFp8AttnInputKeyRows + parent_row -
               kFp8AttnInputValueBegin;
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float result) const {
        *destination(parent_row, token) = __float2bfloat16_rn(result);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

// attn_input_proj's own tp2 column shard output -- each device's own head-local
// Q|K|Gate|V sections (query/gate rows halved to 3072, key/value rows halved to 512), Geometry-
// parameterized the same way fp8_gdn_input_output.cuh's Fp8GdnInputShardOutput<Geometry> is for
// its sibling family. The tp1-named Fp8AttentionInputOutput struct above is kept exactly as it
// was -- every existing tp1 call site continues to build the identical 4-tensor aggregate.
template <class Geometry>
struct Fp8AttentionInputShardOutput {
    static constexpr std::int32_t kQueryRows  = 3072;
    static constexpr std::int32_t kKeyRows    = 512;
    static constexpr std::int32_t kGateRows   = 3072;
    static constexpr std::int32_t kKeyBegin   = kQueryRows;
    static constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
    static constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;

    static_assert(Geometry::kOutputRows == kValueBegin + kKeyRows);

    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
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

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float result) const {
        *destination(parent_row, token) = __float2bfloat16_rn(result);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

} // namespace ninfer::ops::detail
