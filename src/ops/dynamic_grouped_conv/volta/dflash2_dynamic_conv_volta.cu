// DFlash2 dynamic grouped convolution — sm_70.
//
// A short (2-tap) depthwise filter with per-group dynamic gains, applied inside one request
// block. 320 groups of 16 channels; taps = {current position, previous position}; two sides.
//
//   Conv_s(X)[c,i,b] = (base[c,0,s] + d_s0) * X[c,i,b]
//                    + 1{i>=1} * (base[c,1,s] + d_s1) * X[c,i-1,b]        (g = c/16)
//
// prepare (side 0) consumes the full W_delta projection `delta` [side(2) x tap(2) x group(320),
// col] (col = i + W*b), writes the filtered input to `prepared`, and stashes the side-1 gains as
// `finish_delta` [group(320) x tap(2) x col]. add (side 1) reads that `finish_delta` back.

#include "core/device.h"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {

namespace {

constexpr int kHidden = 5120;
constexpr int kGroup  = 16;
constexpr int kGroups = kHidden / kGroup; // 320

__device__ __forceinline__ float bf(const __nv_bfloat16* p, std::int64_t i) {
    return __bfloat162float(p[i]);
}

__device__ __forceinline__ int delta_row(int side, int tap, int g) {
    return (side * 2 + tap) * kGroups + g;
}

__global__ void dflash2_dynamic_conv_prepare_kernel(const __nv_bfloat16* __restrict__ n,
                                                    const __nv_bfloat16* __restrict__ delta,
                                                    const __nv_bfloat16* __restrict__ base,
                                                    int width, int batch,
                                                    __nv_bfloat16* __restrict__ prepared,
                                                    __nv_bfloat16* __restrict__ finish) {
    const std::int64_t idx   = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = static_cast<std::int64_t>(kHidden) * width * batch;
    if (idx >= total) { return; }

    const int c   = static_cast<int>(idx % kHidden);
    const int rem = static_cast<int>(idx / kHidden);
    const int i   = rem % width;
    const int b   = rem / width;
    const int g   = c / kGroup;
    const int col = i + width * b;

    const float base_t0 = bf(base, c + static_cast<std::int64_t>(kHidden) * 0);         // tap0 side0
    const float base_t1 = bf(base, c + static_cast<std::int64_t>(kHidden) * 1);         // tap1 side0
    const float d_t0    = bf(delta, delta_row(0, 0, g) + 1280LL * col);
    const float d_t1    = bf(delta, delta_row(0, 1, g) + 1280LL * col);

    const std::int64_t out_index = c + static_cast<std::int64_t>(kHidden) * col;
    float acc                    = (base_t0 + d_t0) * bf(n, out_index);
    if (i >= 1) {
        acc += (base_t1 + d_t1) * bf(n, out_index - kHidden);
    }
    prepared[out_index] = __float2bfloat16_rn(acc);

    if (finish != nullptr && (c % kGroup) == 0) {
        finish[g + 320LL * (0 + 2 * col)] = delta[delta_row(1, 0, g) + 1280LL * col];
        finish[g + 320LL * (1 + 2 * col)] = delta[delta_row(1, 1, g) + 1280LL * col];
    }
}

__global__ void dflash2_dynamic_conv_finish_add_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ finish_delta,
    const __nv_bfloat16* __restrict__ base, int width, int batch,
    __nv_bfloat16* __restrict__ residual) {
    const std::int64_t idx   = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = static_cast<std::int64_t>(kHidden) * width * batch;
    if (idx >= total) { return; }

    const int c   = static_cast<int>(idx % kHidden);
    const int rem = static_cast<int>(idx / kHidden);
    const int i   = rem % width;
    const int b   = rem / width;
    const int g   = c / kGroup;
    const int col = i + width * b;

    const float base_t0 = bf(base, c + static_cast<std::int64_t>(kHidden) * (0 + 2)); // tap0 side1
    const float base_t1 = bf(base, c + static_cast<std::int64_t>(kHidden) * (1 + 2)); // tap1 side1
    const float d_t0    = bf(finish_delta, g + 320LL * (0 + 2 * col));
    const float d_t1    = bf(finish_delta, g + 320LL * (1 + 2 * col));

    const std::int64_t out_index = c + static_cast<std::int64_t>(kHidden) * col;
    float acc                    = (base_t0 + d_t0) * bf(projected, out_index);
    if (i >= 1) {
        acc += (base_t1 + d_t1) * bf(projected, out_index - kHidden);
    }
    residual[out_index] = __float2bfloat16_rn(__bfloat162float(residual[out_index]) + acc);
}


// The W_delta control projection is BF16_CTRL [1280, 5120]; the fork general BF16 linear only
// admits a few fixed shapes, so a small dedicated matvec covers it. One block per (row, col).
__global__ void dflash2_bf16_matvec_kernel(const __nv_bfloat16* __restrict__ w,
                                           const __nv_bfloat16* __restrict__ x, int n_rows, int k,
                                           int cols, __nv_bfloat16* __restrict__ out) {
    const int row = static_cast<int>(blockIdx.x);
    const int col = static_cast<int>(blockIdx.y);
    if (row >= n_rows || col >= cols) { return; }
    const __nv_bfloat16* wrow = w + static_cast<std::int64_t>(row) * k;
    const __nv_bfloat16* xcol = x + static_cast<std::int64_t>(col) * k;
    float acc = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < k; i += static_cast<int>(blockDim.x)) {
        acc += __bfloat162float(wrow[i]) * __bfloat162float(xcol[i]);
    }
    __shared__ float sums[4];
    const float total = block_reduce_sum<128>(acc, sums);
    if (threadIdx.x == 0) {
        out[row + static_cast<std::int64_t>(col) * n_rows] = __float2bfloat16_rn(total);
    }
}

// Keep each weight row resident across eight control columns instead of rereading it for each
// position; the last tile may contain fewer than eight live columns.
__global__ __launch_bounds__(128, 4) void dflash2_bf16_control_w8_kernel(
    const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out, int columns) {
    constexpr int kRows = 1280;
    constexpr int k     = 5120;
    constexpr int kCols = 8;
    constexpr int kWarps = 4;
    const int row = static_cast<int>(blockIdx.x);
    const int column_begin = static_cast<int>(blockIdx.y) * kCols;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const __nv_bfloat16* wrow = w + static_cast<std::int64_t>(row) * k;
    float acc[kCols] = {};

    for (int i = static_cast<int>(threadIdx.x); i < k; i += 128) {
        const float weight = __bfloat162float(wrow[i]);
#pragma unroll
        for (int col = 0; col < kCols; ++col) {
            if (column_begin + col < columns) {
                acc[col] = fmaf(weight,
                                __bfloat162float(x[static_cast<std::int64_t>(column_begin + col) *
                                                     k + i]),
                                acc[col]);
            }
        }
    }

    __shared__ float sums[kWarps][kCols];
#pragma unroll
    for (int col = 0; col < kCols; ++col) {
        acc[col] = warp_reduce_sum(acc[col]);
        if (lane == 0) { sums[warp][col] = acc[col]; }
    }
    __syncthreads();
    if (warp == 0 && lane < kCols && column_begin + lane < columns) {
        // Preserve block_reduce_sum<128>'s pairwise order: (warp0 + warp2) + (warp1 + warp3).
        const float total = (sums[0][lane] + sums[2][lane]) + (sums[1][lane] + sums[3][lane]);
        out[row + static_cast<std::int64_t>(column_begin + lane) * kRows] =
            __float2bfloat16_rn(total);
    }
}

} // namespace

void dflash2_dynamic_conv_prepare_launch(const __nv_bfloat16* n, const __nv_bfloat16* delta,
                                         const __nv_bfloat16* base, int width, int batch,
                                         __nv_bfloat16* prepared, __nv_bfloat16* finish_delta,
                                         cudaStream_t stream) {
    const std::int64_t total = static_cast<std::int64_t>(kHidden) * width * batch;
    const int block          = 256;
    const int grid           = static_cast<int>((total + block - 1) / block);
    dflash2_dynamic_conv_prepare_kernel<<<grid, block, 0, stream>>>(n, delta, base, width, batch,
                                                                    prepared, finish_delta);
    CUDA_CHECK(cudaGetLastError());
}

void dflash2_bf16_control_projection_launch(const __nv_bfloat16* w, const __nv_bfloat16* x,
                                           int n_rows, int k, int cols, __nv_bfloat16* out,
                                           cudaStream_t stream) {
    if (n_rows == 1280 && k == 5120 && cols >= 8) {
        dflash2_bf16_control_w8_kernel<<<dim3(n_rows, (cols + 7) / 8), 128, 0, stream>>>(
            w, x, out, cols);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const dim3 grid(static_cast<unsigned>(n_rows), static_cast<unsigned>(cols));
    dflash2_bf16_matvec_kernel<<<grid, 128, 0, stream>>>(w, x, n_rows, k, cols, out);
    CUDA_CHECK(cudaGetLastError());
}

void dflash2_dynamic_conv_finish_add_launch(const __nv_bfloat16* projected,
                                            const __nv_bfloat16* finish_delta,
                                            const __nv_bfloat16* base, int width, int batch,
                                            __nv_bfloat16* residual, cudaStream_t stream) {
    const std::int64_t total = static_cast<std::int64_t>(kHidden) * width * batch;
    const int block          = 256;
    const int grid           = static_cast<int>((total + block - 1) / block);
    dflash2_dynamic_conv_finish_add_kernel<<<grid, block, 0, stream>>>(projected, finish_delta,
                                                                       base, width, batch,
                                                                       residual);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
