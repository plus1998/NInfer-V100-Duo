#pragma once

// ninfer::ops - split-KV GQA small-T attention shared scaffolding. The bf16 and
// int8 partial kernels live in gqa_attention_decode_bf16.cuh and
// gqa_attention_decode_i8.cuh respectively; they are fully separate kernels (no
// shared body) so each KV format can be optimized independently. This header owns
// only what both share: layout constants, device helpers, and the split reducer.

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ninfer/ops/gqa_attention.h"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaHeadDim = 256;

// The SM shared-memory residency budget the split-KV decode kernels are tuned against.
//
// sm_120 offers 100 KiB of shared memory per SM to a CTA carveout, and every decode instantiation
// carries an explicit `MinBlocksPerSm` in its `__launch_bounds__` -- 2 for the long-window profiles
// that matter here. If the sum of one CTA's shared memory times that occupancy target ever exceeds
// this, the hardware silently drops to fewer CTAs per SM instead of failing: a performance cliff
// with no diagnostic. `kGqaSmallTSplitPageIds` grows with `kGqaAttentionMaximumVisibleKeys`, so
// widening the Op's visible-key domain is exactly the edit that could walk into that cliff. Both
// kernels therefore static_assert their own total against this constant.
//
// At the 1,048,576-key domain, by this same conservative accounting (per CTA, static + dynamic,
// times MinBlocksPerSm), the five largest reachable instantiations are:
//   i8   24|4 TokenTile 6, Wc 6,  Bc 32, static arena,  PageIds 194, MinBlocks 2 -> 99,232 B
//   i8   12|2 TokenTile 6, Wc 6,  Bc 32, static arena,  PageIds  98, MinBlocks 2 -> 98,464 B
//   i8   24|4 TokenTile 5, Wc 8,  Bc 32, static arena,  PageIds 194, MinBlocks 2 -> 88,864 B
//   i8   24|4 TokenTile 6, Wc 12, Bc 64, dynamic arena, PageIds 194, MinBlocks 1 -> 85,984 B
//   bf16 24|4 TokenTile 6, Wc 4,  Bc 32,                PageIds 194, MinBlocks 2 -> 75,296 B
// The tightest is 99,232 of 102,400 B: ~3.1 KiB of headroom, all of it consumed by the page-id
// staging, which is 776 B of that CTA. Doubling the domain again to 2,097,152 keys stays inside
// the budget; quadrupling it does not, and fails to compile here (verified) rather than halving
// occupancy in silence.
inline constexpr int kGqaDecodeSharedResidencyBytes = 100 * 1024;

// Shared-memory arrays are laid out with 16-byte alignment; rounding each one up is the
// conservative accounting (it can only over-estimate the total).
// `__host__ __device__` so a kernel body may evaluate it in a constant expression: nvcc refuses to
// call a host-only constexpr function from a `__global__` function even in a static_assert.
[[nodiscard]] __host__ __device__ inline constexpr int gqa_shared_align16(int bytes) {
    return ((bytes + 15) / 16) * 16;
}

// Upper bound on the physical page ids ONE split stages in shared memory, derived from the Op's
// declared visible-key domain instead of hard-coded.
//
// A split covers `ceil(ceil(window / KeyBlock) / active_splits) * KeyBlock` keys (see the
// `units_per_split` computation in both partial kernels). `active_splits` saturates at
// `Geometry::DecodeSplits`, so the span grows linearly with the window past that point and its
// maximum over the whole declared domain is reached at `kGqaAttentionMaximumVisibleKeys`. The
// trailing `+ 1` covers a split whose first key tile starts mid-page (KeyBlock < page size).
//
// At the 1,048,576-key domain this is 194 ids (776 B) for a DecodeSplits == 85 geometry and 98
// (392 B) for the head-local DecodeSplits == 170 one; the pre-YaRN 262,144-key domain needed 50.
// Getting this wrong is a shared-memory overrun, not a wrong answer, which is why it is computed
// from the same constant the wrapper validates envelopes against.
template <typename Geometry, int KeyBlock>
inline constexpr int kGqaSmallTSplitPageIds =
    (((((static_cast<int>(kGqaAttentionMaximumVisibleKeys) + KeyBlock - 1) / KeyBlock) +
       Geometry::DecodeSplits - 1) /
      Geometry::DecodeSplits) *
         KeyBlock +
     static_cast<int>(kPagedKVPageSize) - 1) /
        static_cast<int>(kPagedKVPageSize) +
    1;

struct GqaAppendInput {
    static constexpr bool writes_cache = true;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
};

struct GqaCachedInput {
    static constexpr bool writes_cache = false;
};

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_cache_index(int physical_page, int kv_head, int d,
                                                        int page_offset) {
    return paged_kv_element_offset<kGqaHeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                   page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_q_index(int q_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kGqaHeadDim) *
                                              (static_cast<std::int64_t>(q_head) +
                                               static_cast<std::int64_t>(Geometry::QHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_new_index(int kv_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaHeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_partial_acc_index(int q_head, int d, int token,
                                                              int split, int tokens) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(Geometry::QHeads) *
                    (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split));
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_partial_stat_index(int q_head, int token, int split,
                                                               int tokens) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(Geometry::QHeads) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split);
}

template <typename Geometry>
__device__ __forceinline__ bool gqa_valid_q_head(int kv_head, int q_head) {
    return kv_head >= 0 && kv_head < Geometry::KVHeads && q_head >= kv_head * Geometry::GroupSize &&
           q_head < (kv_head + 1) * Geometry::GroupSize && q_head < Geometry::QHeads;
}

template <typename Geometry>
__device__ __forceinline__ int gqa_small_t_default_splits(int window) {
    int target_keys_per_split = 480 / Geometry::DecodeSplitScale;
    if (window <= 4096) {
        target_keys_per_split = 64 / Geometry::DecodeSplitScale;
    } else if (window <= 8198) {
        target_keys_per_split = 128 / Geometry::DecodeSplitScale;
    } else if (window <= 16390) {
        target_keys_per_split = 256 / Geometry::DecodeSplitScale;
    }
    constexpr int kMinSplits = 4 * Geometry::DecodeSplitScale;
    int splits               = div_up(window, target_keys_per_split);
    splits                   = splits > kMinSplits ? splits : kMinSplits;
    return splits < Geometry::DecodeSplits ? splits : Geometry::DecodeSplits;
}

template <typename Geometry, bool Int8>
__device__ __forceinline__ int gqa_small_t_active_splits(int window, int launch_capacity,
                                                         int tokens) {
    if (window <= 0) { return launch_capacity; }
    int splits = 0;
    if constexpr (Int8) {
        if (tokens == 5 && window > 128 && window <= 512) {
            splits = div_up(window, 32 / Geometry::DecodeSplitScale);
        } else if (tokens == 6 && window > 128 && window <= 160) {
            splits = div_up(window, 24 / Geometry::DecodeSplitScale);
        } else if (tokens == 6 && window > 5000 && window <= 8198) {
            splits             = div_up(window, 192 / Geometry::DecodeSplitScale);
            constexpr int kMin = 4 * Geometry::DecodeSplitScale;
            constexpr int kMax = 42 * Geometry::DecodeSplitScale;
            splits             = splits > kMin ? splits : kMin;
            splits             = splits < kMax ? splits : kMax;
        } else {
            splits = gqa_small_t_default_splits<Geometry>(window);
        }
    } else {
        splits = gqa_small_t_default_splits<Geometry>(window);
    }
    return splits < launch_capacity ? splits : launch_capacity;
}

__device__ __forceinline__ int gqa_small_t_tc_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ int gqa_small_t_tc_swz32(int row, int col) {
    return (((col >> 3) ^ (row & 3)) << 3) | (col & 7);
}

// Signed int8 QK MMA, k=32 contraction. A = 16x32 s8 (4 regs/thread, 4 s8 each),
// B = 8x32 s8 col-major (2 regs/thread), D = 16x8 s32 (4 regs/thread). The A/B
// register byte layout is identical to the m16n8k16 bf16 fragments loaded by
// ldmatrix_x4/x2 over a d-contiguous int8 tile reinterpreted as
// b16 (two packed int8 per 16-bit lane), so the same ldmatrix helpers and XOR
// swizzle feed this MMA. The s32 accumulator layout matches the bf16 f32
// accumulator (c0/c1 -> row groupID, c2/c3 -> row groupID+8), so score
// consumption is unchanged; only per-64-group scale rescale differs.
template <typename Geometry>
__device__ __forceinline__ void gqa_small_t_tc_row_to_qt(int row, int tokens, int kv_head,
                                                         int& q_head, int& token) {
    token             = row / Geometry::GroupSize;
    const int local_q = row - token * Geometry::GroupSize;
    q_head            = kv_head * Geometry::GroupSize + local_q;
}

template <typename Geometry, int DChunk, bool Int8, bool MultiBatch, bool Masked, bool Offset>
__launch_bounds__(256) __global__ void gqa_attention_small_t_reduce_output_kernel(
    const __nv_bfloat16* partial_acc, const float* partial_m, const float* partial_l,
    const std::int32_t* positions, const std::int32_t* valid_columns, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t batch_size,
    std::int32_t split_count, __nv_bfloat16* out) {
    static_assert(DChunk > 0 && DChunk <= kGqaHeadDim);

    const int q_head      = static_cast<int>(blockIdx.x);
    const int d_start     = static_cast<int>(blockIdx.y) * DChunk;
    const int flat_column = static_cast<int>(blockIdx.z);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid = threadIdx.x;
    if (q_head >= Geometry::QHeads || token >= tokens) { return; }
    if constexpr (MultiBatch) {
        if (batch >= batch_size) { return; }
    }

    if constexpr (Offset) { positions += column_begin; }
    if constexpr (MultiBatch) { positions += batch * full_width; }
    const int last_pos = positions[tokens - 1];
    int output_column  = token;
    if constexpr (Offset) { output_column += column_begin; }
    if constexpr (MultiBatch) { output_column += batch * full_width; }

    if constexpr (MultiBatch) {
        const std::int64_t partial_acc_row = static_cast<std::int64_t>(batch) * kGqaHeadDim *
                                             Geometry::QHeads * tokens * split_count;
        const std::int64_t partial_stat_row =
            static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_acc += partial_acc_row;
        partial_m += partial_stat_row;
        partial_l += partial_stat_row;
    }

    const int window = last_pos + 1;
    const int active_split_count =
        gqa_small_t_active_splits<Geometry, Int8>(window, split_count, tokens);

    __shared__ float reduce[256];

    float local_m = -CUDART_INF_F;
    for (int split = tid; split < active_split_count; split += blockDim.x) {
        local_m = fmaxf(local_m,
                        partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)]);
    }
    reduce[tid] = local_m;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]); }
        __syncthreads();
    }
    const float head_m = reduce[0];
    __syncthreads();

    if (head_m == -CUDART_INF_F) {
        const int d = d_start + tid;
        if (tid < DChunk && d < kGqaHeadDim) {
            out[gqa_q_index<Geometry>(q_head, d, output_column)] = __float2bfloat16(0.0f);
        }
        return;
    }

    float local_l = 0.0f;
    for (int split = tid; split < active_split_count; split += blockDim.x) {
        const float tile_l =
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)];
        if (tile_l > 0.0f) {
            local_l +=
                tile_l *
                expf(partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] -
                     head_m);
        }
    }
    reduce[tid] = local_l;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce[tid] += reduce[tid + stride]; }
        __syncthreads();
    }
    const float head_l = reduce[0];

    const int d = d_start + tid;
    if (tid >= DChunk || d >= kGqaHeadDim) { return; }

    float numerator = 0.0f;
    if (head_l > 0.0f) {
        for (int split = 0; split < active_split_count; ++split) {
            const float tile_l =
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)];
            if (tile_l <= 0.0f) { continue; }
            const float weight = expf(
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] - head_m);
            numerator +=
                __bfloat162float(
                    partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)]) *
                weight;
        }
    }
    bool valid = true;
    if constexpr (Masked) {
        int absolute_column = token;
        if constexpr (Offset) { absolute_column += column_begin; }
        valid = absolute_column < valid_columns[batch];
    }
    const float value = (valid && head_l > 0.0f) ? numerator / head_l : 0.0f;
    out[gqa_q_index<Geometry>(q_head, d, output_column)] = __float2bfloat16(value);
}

} // namespace ninfer::ops
