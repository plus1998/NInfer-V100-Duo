#pragma once

// ninfer::ops::detail - private launch prototype for rope. Included by the wrapper
// and defined by the CUDA launcher.

#include "core/tensor.h"
#include "ninfer/ops/rope.h" // ninfer::ops::RopeFrequencyOverride

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// `frequency.inv_frequency == nullptr` selects the native theta-derived dispatch, unchanged.
void rope_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                 const RopeFrequencyOverride& frequency, cudaStream_t stream);

void rope_single_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                        const RopeFrequencyOverride& frequency, cudaStream_t stream);

} // namespace ninfer::ops::detail
