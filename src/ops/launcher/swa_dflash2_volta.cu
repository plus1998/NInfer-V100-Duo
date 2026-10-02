#include "ops/launcher/swa.h"

#include "core/device.h"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kDim = 128;
constexpr int kQueryHeads = 16;
constexpr int kValueHeads = 4;
constexpr int kWindow = 2048;
constexpr int kKeyTile = 32;

__device__ __forceinline__ std::int64_t query_index(int head, int dim, int token) {
    return dim + static_cast<std::int64_t>(kDim) * (head + kQueryHeads * token);
}

__device__ __forceinline__ std::int64_t value_index(int head, int dim, int token) {
    return dim + static_cast<std::int64_t>(kDim) * (head + kValueHeads * token);
}

__global__ void dflash2_swa_partial(
    const __nv_bfloat16* __restrict__ query, const __nv_bfloat16* __restrict__ query_key,
    const __nv_bfloat16* __restrict__ query_value, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
    const __nv_bfloat16* __restrict__ context_key, const __half* __restrict__ context_value,
    int tokens, int padded_capacity, int max_context, int splits, float scale,
    __nv_bfloat16* __restrict__ partial, float* __restrict__ maxima,
    float* __restrict__ denominators, __nv_bfloat16* __restrict__ output) {
    const int dim = threadIdx.x;
    const int head = blockIdx.x % kQueryHeads;
    const int token = blockIdx.x / kQueryHeads;
    const int split = blockIdx.y;
    const int batch = blockIdx.z;
    const int valid = valid_columns[batch];
    const int frontier = positions[static_cast<std::int64_t>(batch) * tokens];
    if (valid < 0 || valid > tokens || frontier < 0 || frontier > max_context) return;

    const std::int64_t query_batch = static_cast<std::int64_t>(kDim) * kQueryHeads * tokens * batch;
    const std::int64_t value_batch = static_cast<std::int64_t>(kDim) * kValueHeads * tokens * batch;
    const std::int64_t partial_batch = static_cast<std::int64_t>(kDim) * kQueryHeads * tokens *
                                       splits * batch;
    const std::int64_t stats_batch = static_cast<std::int64_t>(kQueryHeads) * tokens * splits * batch;
    const std::int64_t row = head + kQueryHeads * token;
    query += query_batch;
    query_key += value_batch;
    query_value += value_batch;
    output += query_batch;
    partial += partial_batch;
    maxima += stats_batch;
    denominators += stats_batch;

    const int context_count = min(frontier, kWindow - 1);
    const int tiles = (context_count + kKeyTile - 1) / kKeyTile;
    const int active = tiles > 0 ? min(tiles, splits) : 1;
    const std::int64_t partial_row = dim + static_cast<std::int64_t>(kDim) *
                                              (row + static_cast<std::int64_t>(kQueryHeads) * tokens * split);
    const std::int64_t stats_row = row + static_cast<std::int64_t>(kQueryHeads) * tokens * split;
    if (token >= valid || split >= active) {
        partial[partial_row] = __float2bfloat16_rn(0.0F);
        if (dim == 0) {
            maxima[stats_row] = -CUDART_INF_F;
            denominators[stats_row] = 0.0F;
        }
        if (token >= valid && splits == 1) {
            output[query_index(head, dim, token)] = __float2bfloat16_rn(0.0F);
        }
        return;
    }

    const int key_head = head / 4;
    const std::int64_t cache_base = static_cast<std::int64_t>(kDim) * padded_capacity *
                                    (key_head + kValueHeads * lanes[batch]);
    const int start = frontier - context_count;
    const int tile_first = static_cast<int>(static_cast<std::int64_t>(tiles) * split / active);
    const int tile_last = static_cast<int>(static_cast<std::int64_t>(tiles) * (split + 1) / active);
    const int key_first = start + tile_first * kKeyTile;
    const int key_last = min(frontier, start + tile_last * kKeyTile);
    const float q = __bfloat162float(query[query_index(head, dim, token)]);
    float numerator = 0.0F;
    float maximum = -CUDART_INF_F;
    float denominator = 0.0F;
    __shared__ float sums[4];
    __shared__ float alpha;
    __shared__ float probability;
    __shared__ float final_denominator;

    for (int key = key_first; key < key_last + (split == active - 1 ? valid : 0); ++key) {
        const bool context = key < key_last;
        if (context && key < positions[static_cast<std::int64_t>(batch) * tokens + token] -
                                  (kWindow - 1)) {
            continue;
        }
        const int query_row = key - key_last;
        const std::int64_t cache_row = cache_base + static_cast<std::int64_t>(kDim) * (key & (kWindow - 1));
        const float key_value = context ? __bfloat162float(context_key[cache_row + dim])
                                        : __bfloat162float(query_key[value_index(key_head, dim, query_row)]);
        const float value = context ? __half2float(context_value[cache_row + dim])
                                    : __bfloat162float(query_value[value_index(key_head, dim, query_row)]);
        const float dot = block_reduce_sum<128>(q * key_value, sums);
        if (dim == 0) {
            const float score = dot * scale;
            const float next = fmaxf(maximum, score);
            alpha = maximum == -CUDART_INF_F ? 0.0F : expf(maximum - next);
            probability = expf(score - next);
            denominator = denominator * alpha + probability;
            maximum = next;
        }
        __syncthreads();
        numerator = numerator * alpha + probability * value;
        __syncthreads();
    }
    if (splits == 1) {
        if (dim == 0) final_denominator = denominator;
        __syncthreads();
        output[query_index(head, dim, token)] =
            __float2bfloat16_rn(final_denominator > 0.0F ? numerator / final_denominator : 0.0F);
    } else {
        partial[partial_row] = __float2bfloat16_rn(numerator);
        if (dim == 0) {
            maxima[stats_row] = maximum;
            denominators[stats_row] = denominator;
        }
    }
}

__global__ void dflash2_swa_reduce(const __nv_bfloat16* __restrict__ partial,
                                   const float* __restrict__ maxima,
                                   const float* __restrict__ denominators,
                                   const std::int32_t* __restrict__ valid_columns,
                                   int tokens, int splits, __nv_bfloat16* __restrict__ output) {
    const int dim = threadIdx.x;
    const int head = blockIdx.x % kQueryHeads;
    const int token = blockIdx.x / kQueryHeads;
    const int batch = blockIdx.y;
    const std::int64_t rows = static_cast<std::int64_t>(kQueryHeads) * tokens;
    const std::int64_t row = head + kQueryHeads * token;
    if (token >= valid_columns[batch]) {
        output[static_cast<std::int64_t>(kDim) * rows * batch + query_index(head, dim, token)] =
            __float2bfloat16_rn(0.0F);
        return;
    }
    const std::int64_t stats_base = rows * splits * batch;
    float maximum = -CUDART_INF_F;
    for (int split = 0; split < splits; ++split) {
        maximum = fmaxf(maximum, maxima[stats_base + row + rows * split]);
    }
    float numerator = 0.0F;
    float denominator = 0.0F;
    if (maximum != -CUDART_INF_F) {
        for (int split = 0; split < splits; ++split) {
            const float weight = expf(maxima[stats_base + row + rows * split] - maximum);
            denominator += weight * denominators[stats_base + row + rows * split];
            numerator += weight * __bfloat162float(
                partial[static_cast<std::int64_t>(kDim) * (row + rows * (split + splits * batch)) + dim]);
        }
    }
    output[static_cast<std::int64_t>(kDim) * rows * batch + query_index(head, dim, token)] =
        __float2bfloat16_rn(denominator > 0.0F ? numerator / denominator : 0.0F);
}

} // namespace

void swa_dflash2_launch(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                        const Tensor& positions, const Tensor& valid_columns, const Tensor& lanes,
                        float scale, const CyclicKVCacheLayerView& context, const SwaPlan& plan,
                        Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out,
                        cudaStream_t stream) {
    if (plan.split_capacity < 1 || plan.split_capacity > kSwaMaxCandidateSplit) {
        throw std::invalid_argument("DFlash2 swa: invalid split plan");
    }
    const int splits = plan.route == SwaRoute::Direct ? 1 : plan.split_capacity;
    const dim3 grid(kQueryHeads * q.ne[2], splits, q.ne[3]);
    dflash2_swa_partial<<<grid, kDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(query_k.data),
        static_cast<const __nv_bfloat16*>(query_v.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<const std::int32_t*>(lanes.data),
        static_cast<const __nv_bfloat16*>(context.k.data),
        static_cast<const __half*>(context.v.data), q.ne[2], context.padded_capacity,
        plan.max_context, splits, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
    if (splits > 1) {
        const dim3 reduce_grid(kQueryHeads * q.ne[2], q.ne[3]);
        dflash2_swa_reduce<<<reduce_grid, kDim, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(partial_acc.data),
            static_cast<const float*>(partial_m.data), static_cast<const float*>(partial_l.data),
            static_cast<const std::int32_t*>(valid_columns.data), q.ne[2], splits,
            static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
