#pragma once

#include "core/cyclic_kv_cache.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void dflash2_context_append_launch(const Tensor& key, const Tensor& value,
                                   const Tensor& positions, const Tensor& counts,
                                   const Tensor& state_slots, CyclicKVCacheLayerView cache,
                                   std::uint32_t max_count, cudaStream_t stream);

} // namespace ninfer::ops::detail
