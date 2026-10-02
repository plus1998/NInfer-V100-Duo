// DFlash2 unary candidate top-16 — sm_70.
//
// Per column of the proposal-head logits, return the 16 highest scores and their global token ids,
// ordered by descending score with exact ties broken by lower id. Physical rows >= valid_rows are
// excluded (full head); the optimized head remaps every row through id_map.

#include "ops/linear_topk/dflash2_linear_topk_volta.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cstdint>
#include <cfloat>

namespace ninfer::ops::detail {
namespace {

constexpr int kTopK   = 16;
constexpr int kBlock  = 256;
constexpr int kTileRows = kDflash2TopKTileRows;

// a strictly beats b under (score desc, id asc)
__device__ __forceinline__ bool better(float sa, int ia, float sb, int ib) {
    return sa > sb || (sa == sb && ia < ib);
}

__device__ __forceinline__ void local_insert(float* s, int* d, float score, int id) {
    if (!better(score, id, s[kTopK - 1], d[kTopK - 1])) { return; }
    int p = kTopK - 1;
    while (p > 0 && better(score, id, s[p - 1], d[p - 1])) {
        s[p] = s[p - 1];
        d[p] = d[p - 1];
        --p;
    }
    s[p] = score;
    d[p] = id;
}

__device__ __forceinline__ void block_best(float score, int id,
                                            float* warp_scores, int* warp_ids,
                                            float& best_score, int& best_id) {
    const int lane = threadIdx.x & 31;
    for (int delta = 16; delta > 0; delta >>= 1) {
        const float other_score = __shfl_down_sync(0xffffffff, score, delta);
        const int other_id = __shfl_down_sync(0xffffffff, id, delta);
        if (lane + delta < 32 && better(other_score, other_id, score, id)) {
            score = other_score;
            id = other_id;
        }
    }
    if (lane == 0) {
        warp_scores[threadIdx.x / 32] = score;
        warp_ids[threadIdx.x / 32] = id;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        best_score = warp_scores[0];
        best_id = warp_ids[0];
        for (int warp = 1; warp < kBlock / 32; ++warp) {
            if (better(warp_scores[warp], warp_ids[warp], best_score, best_id)) {
                best_score = warp_scores[warp];
                best_id = warp_ids[warp];
            }
        }
    }
    __syncthreads();
}

__global__ void dflash2_topk16_kernel(const __nv_bfloat16* __restrict__ logits,
                                      const std::int32_t* __restrict__ id_map, int head_rows,
                                      int valid_rows, int columns, int row_offset, int tiles,
                                      std::int32_t* __restrict__ ids,
                                      float* __restrict__ scores) {
    const int col = static_cast<int>(blockIdx.x);
    const int tile = static_cast<int>(blockIdx.y);
    if (col >= columns) { return; }
    const __nv_bfloat16* logit_col = logits + static_cast<std::int64_t>(col) * head_rows;

    float ls[kTopK];
    int   ld[kTopK];
#pragma unroll
    for (int i = 0; i < kTopK; ++i) {
        ls[i] = -FLT_MAX;
        ld[i] = 0x7fffffff;
    }

    for (int r = tile * kTileRows + static_cast<int>(threadIdx.x);
         r < valid_rows && r < (tile + 1) * kTileRows; r += kBlock) {
        const int gid = id_map != nullptr ? id_map[r] : row_offset + r;
        local_insert(ls, ld, __bfloat162float(logit_col[r]), gid);
    }

    __shared__ float warp_scores[kBlock / 32], best_score;
    __shared__ int warp_ids[kBlock / 32], best_id;
    int next = 0;
    for (int round = 0; round < kTopK; ++round) {
        block_best(ls[next], ld[next], warp_scores, warp_ids, best_score, best_id);
        if (threadIdx.x == 0) {
            const auto output = (static_cast<std::int64_t>(col) * tiles + tile) * kTopK + round;
            scores[output] = best_score;
            ids[output] = best_id;
        }
        if (ld[next] == best_id && next + 1 < kTopK) { ++next; }
        __syncthreads();
    }
}

__global__ void dflash2_finish_topk16_kernel(
    const std::int32_t* __restrict__ partial_ids,
    const float* __restrict__ partial_scores, int tiles, int columns,
    std::int32_t* __restrict__ ids, float* __restrict__ scores) {
    const int col = static_cast<int>(blockIdx.x);
    if (col >= columns) { return; }
    const int entries = tiles * kTopK;
    float local_scores[kTopK];
    int local_ids[kTopK];
#pragma unroll
    for (int index = 0; index < kTopK; ++index) {
        local_scores[index] = -FLT_MAX;
        local_ids[index] = 0x7fffffff;
    }
    for (int index = static_cast<int>(threadIdx.x); index < entries; index += kBlock) {
        const int offset = col * entries + index;
        local_insert(local_scores, local_ids, partial_scores[offset], partial_ids[offset]);
    }
    __shared__ float warp_scores[kBlock / 32], best_score;
    __shared__ int warp_ids[kBlock / 32], best_id;
    int next = 0;
    for (int round = 0; round < kTopK; ++round) {
        block_best(local_scores[next], local_ids[next], warp_scores, warp_ids,
                   best_score, best_id);
        if (threadIdx.x == 0) {
            const int output = col * kTopK + round;
            scores[output] = best_score;
            ids[output] = best_id;
        }
        if (local_ids[next] == best_id && next + 1 < kTopK) { ++next; }
        __syncthreads();
    }
}

__global__ void dflash2_merge_tp2_topk16_kernel(
    const std::int32_t* __restrict__ gathered_ids,
    const float* __restrict__ gathered_scores, std::int32_t* __restrict__ ids,
    float* __restrict__ scores, int columns) {
    const int column = static_cast<int>(blockIdx.x);
    const int rank = static_cast<int>(threadIdx.x);
    if (column >= columns || rank >= kTopK) { return; }
    __shared__ float candidate_scores[2 * kTopK];
    __shared__ int candidate_ids[2 * kTopK];
    candidate_scores[rank] = gathered_scores[column * kTopK + rank];
    candidate_scores[rank + kTopK] = gathered_scores[(columns + column) * kTopK + rank];
    candidate_ids[rank] = gathered_ids[column * kTopK + rank];
    candidate_ids[rank + kTopK] = gathered_ids[(columns + column) * kTopK + rank];
    __syncthreads();
    for (int index = 0; index < kTopK; ++index) {
        if (rank == 0) {
            int best = 0;
            for (int entry = 1; entry < 2 * kTopK; ++entry) {
                if (better(candidate_scores[entry], candidate_ids[entry],
                           candidate_scores[best], candidate_ids[best])) {
                    best = entry;
                }
            }
            scores[column * kTopK + index] = candidate_scores[best];
            ids[column * kTopK + index] = candidate_ids[best];
            candidate_scores[best] = -FLT_MAX;
            candidate_ids[best] = 0x7fffffff;
        }
        __syncthreads();
    }
}

} // namespace

void dflash2_linear_topk16_launch(const Tensor& logits, const std::int32_t* id_map,
                                  std::int32_t valid_rows, Tensor& ids, Tensor& scores,
                                  Tensor& partial_ids, Tensor& partial_scores,
                                  cudaStream_t stream, std::int32_t row_offset) {
    const int head_rows = logits.ne[0];
    const int columns   = logits.ne[1];
    const int tiles = (valid_rows + kTileRows - 1) / kTileRows;
    dflash2_topk16_kernel<<<dim3(columns, tiles), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), id_map, head_rows, valid_rows, columns,
        row_offset, tiles, static_cast<std::int32_t*>(partial_ids.data),
        static_cast<float*>(partial_scores.data));
    CUDA_CHECK(cudaGetLastError());
    dflash2_finish_topk16_kernel<<<columns, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(partial_ids.data),
        static_cast<const float*>(partial_scores.data), tiles, columns,
        static_cast<std::int32_t*>(ids.data), static_cast<float*>(scores.data));
    CUDA_CHECK(cudaGetLastError());
}

void dflash2_merge_tp2_topk16_launch(const Tensor& gathered_ids, const Tensor& gathered_scores,
                                     Tensor& ids, Tensor& scores, cudaStream_t stream) {
    dflash2_merge_tp2_topk16_kernel<<<ids.ne[1], kTopK, 0, stream>>>(
        static_cast<const std::int32_t*>(gathered_ids.data),
        static_cast<const float*>(gathered_scores.data),
        static_cast<std::int32_t*>(ids.data), static_cast<float*>(scores.data), ids.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
