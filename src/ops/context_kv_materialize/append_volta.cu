#include "ops/context_kv_materialize/append_volta.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

__global__ void dflash2_context_append_kernel(const __nv_bfloat16* key,
                                               const __nv_bfloat16* value,
                                               const std::int32_t* positions,
                                               const std::int32_t* counts,
                                               const std::int32_t* state_slots,
                                               __nv_bfloat16* cache_key, __half* cache_value,
                                               int width, int padded_capacity, int heads) {
    const int index = blockIdx.x;
    const int row = blockIdx.y;
    if (index >= counts[row]) { return; }
    const int head = blockIdx.z;
    const int lane = state_slots[row];
    const int slot = positions[row * width + index] & 2047;
    const int source = 128 * (head + heads * (index + width * row)) + threadIdx.x;
    const std::int64_t destination =
        128LL * (slot + padded_capacity * (head + heads * lane)) + threadIdx.x;
    cache_key[destination] = key[source];
    cache_value[destination] = __float2half_rn(__bfloat162float(value[source]));
}

} // namespace

void dflash2_context_append_launch(const Tensor& key, const Tensor& value,
                                   const Tensor& positions, const Tensor& counts,
                                   const Tensor& state_slots, CyclicKVCacheLayerView cache,
                                   std::uint32_t max_count, cudaStream_t stream) {
    const int heads = key.ne[1];
    const dim3 grid(max_count, key.ne[3], heads);
    dflash2_context_append_kernel<<<grid, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(key.data),
        static_cast<const __nv_bfloat16*>(value.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(counts.data),
        static_cast<const std::int32_t*>(state_slots.data),
        static_cast<__nv_bfloat16*>(cache.k.data), static_cast<__half*>(cache.v.data),
        key.ne[2], cache.padded_capacity, heads);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
