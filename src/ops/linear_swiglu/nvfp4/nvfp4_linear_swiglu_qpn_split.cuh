#pragma once

// Split-projection NVFP4 SwiGLU on Volta tensor cores (sm_70 only): two independent QPN2 launches
// -- one per weight half. The first writes gate to fp32 scratch. The second keeps QPN2's tuned
// register/occupancy profile but applies silu(gate) * up in its output policy before the single
// BF16 round, avoiding an up scratch plane and a third combine launch.
//
// Why not the fused kernel (nvfp4_linear_swiglu_volta_qpn.cuh)? Measured against it directly on
// the gate_up shape at T=4: the fused kernel reads 225.4 GB/s across the whole weight; two
// separate QPN2 passes at QPN2's own tuned schedule read 371.6 GB/s each. The fused kernel pays
// for computing both projections inside one CTA -- double the accumulators, double the decode
// registers -- and that costs more than the shared activation load saves. Splitting keeps each
// launch at QPN2's own measured-fastest register/occupancy profile; the only new cost is a
// output policy's gate read and activation, which are output-sized work next to the weight stream.
//
// Why fp32 scratch rather than composing linear() + silu_mul() (which was tried and reverted for
// the fused kernel's problem too): the same reasoning holds regardless of which QPN kernel writes
// the projection. linear()'s A16 output is BF16, and silu(gate) * up compounds two independently
// BF16-rounded operands multiplicatively, failing the correctness test by ~0.8% against a ~0.35%
// tolerance. Fp32 scratch is the fix either way -- it just also happens to fix the speed problem,
// because it lets each projection run through QPN2's own kernel unmodified instead of a doubled
// one.

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/linear/nvfp4/nvfp4_output.cuh"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

struct Nvfp4SwiGluFromGateOutput {
    const float* gate;
    __nv_bfloat16* out;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float up) const {
        const std::int64_t index = static_cast<std::int64_t>(token) * rows + parent_row;
        out[index] = __float2bfloat16_rn(silu(gate[index]) * up);
    }
};

#endif // sm_70

} // namespace ninfer::ops::detail
