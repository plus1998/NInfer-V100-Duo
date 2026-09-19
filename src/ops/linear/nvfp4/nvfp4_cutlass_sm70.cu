#include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// The artifact stores adjacent E2M1 values in each code byte and one E4M3 scale per K16 group.
// Scales use the BlockScaleK16M128x4 swizzle. Materializing once is intentionally a wide-T
// strategy: CUTLASS can reuse the resulting FP16 matrix across every token tile instead of
// decoding the packed weights again for each tile.
__global__ void dequant_nvfp4_row_to_fp16(const std::uint8_t* __restrict__ codes,
                                           const std::uint8_t* __restrict__ scales, int n, int k,
                                           float inverse_weight_divisor,
                                           cutlass::half_t* __restrict__ out) {
    const int row      = static_cast<int>(blockIdx.y);
    const int byte_idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int bytes_per_row = k / 2;
    if (row >= n || byte_idx >= bytes_per_row) { return; }

    const int k0            = byte_idx * 2;
    const int group         = byte_idx / 8;
    const int scale_tile    = group / 4;
    const int scale_lane    = group & 3;
    const int row_inner     = row & 127;
    const int scales_per_m128 = k / 64;
    const std::int64_t scale_offset =
        static_cast<std::int64_t>((row / 128) * scales_per_m128 + scale_tile) * 512 +
        (row_inner & 31) * 16 + (row_inner >> 5) * 4 + scale_lane;

    const float coefficient = decode_nvfp4_e4m3(scales[scale_offset]) * inverse_weight_divisor;
    const float2 value = decode_nvfp4_e2m1x2(
        codes[static_cast<std::int64_t>(row) * bytes_per_row + byte_idx]);
    cutlass::half_t* out_row = out + static_cast<std::int64_t>(row) * k;
    out_row[k0]     = cutlass::half_t(value.x * coefficient);
    out_row[k0 + 1] = cutlass::half_t(value.y * coefficient);
}

__global__ void dequant_nvfp4_qpn_to_fp16(const std::uint8_t* __restrict__ codes,
                                          const std::uint8_t* __restrict__ scales, int n, int k,
                                          float inverse_weight_divisor,
                                          cutlass::half_t* __restrict__ out) {
    const int row      = static_cast<int>(blockIdx.y);
    const int segment  = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int segments_per_row = k / 8;
    if (row >= n || segment >= segments_per_row) { return; }

    const int group      = segment / 2;
    const int group_half = segment & 1;
    const int local_row  = row & 31;
    const int qp         = local_row / 8;
    const int r          = local_row & 7;
    const int lane       = (qp << 2) | (r & 3) | ((r & 4) << 2);
    const int groups     = k / 16;
    const std::int64_t tuple =
        (static_cast<std::int64_t>(row / 32) * groups + group) * 32 + lane;
    constexpr int inverse_order[16] = {0, 4, 1, 5, 2, 6, 3, 7,
                                       8, 12, 9, 13, 10, 14, 11, 15};
    const std::uint8_t* packed = codes + tuple * 8;
    const float coefficient = decode_nvfp4_e4m3(scales[tuple]) * inverse_weight_divisor;
    cutlass::half_t values[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int logical_k = group_half * 8 + j;
        const int position  = inverse_order[logical_k];
        const float2 pair   = decode_nvfp4_e2m1x2(packed[position / 2]);
        values[j] = cutlass::half_t(((position & 1) == 0 ? pair.x : pair.y) * coefficient);
    }
    auto* destination = reinterpret_cast<uint4*>(
        out + static_cast<std::int64_t>(row) * k + segment * 8);
    *destination = *reinterpret_cast<const uint4*>(values);
}

__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ in,
                                    cutlass::half_t* __restrict__ out, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = cutlass::half_t(__bfloat162float(in[i])); }
}

int div_up_i(int a, int b) { return (a + b - 1) / b; }

using ElementAccumulator     = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA          = cutlass::half_t;
using ElementInputB          = cutlass::half_t;
using ElementOutput          = cutlass::bfloat16_t;
using LayoutInputA           = cutlass::layout::RowMajor;
using LayoutInputB           = cutlass::layout::ColumnMajor;
using LayoutOutput           = cutlass::layout::RowMajor;
using MMAOp                  = cutlass::arch::OpClassTensorOp;
using SmArch                 = cutlass::arch::Sm70;
using ShapeMMAThreadBlock    = cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeMMAWarp           = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAOp             = cutlass::gemm::GemmShape<8, 8, 4>;
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator,
    ElementComputeEpilogue>;
constexpr int kNumStages = 2;
using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB,
                                         ElementOutput, LayoutOutput, ElementAccumulator, MMAOp,
                                         SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
                                         EpilogueOp, SwizzleThreadBlock, kNumStages>;

template <class Allocator>
struct CutlassWorkspace {
    Tensor w_fp16;
    Tensor x_fp16;
    DeviceSpan gemm_workspace;
};

template <class Allocator>
CutlassWorkspace<Allocator> allocate_cutlass_workspace(Allocator& allocator, std::int32_t n,
                                                       std::int32_t k, std::int32_t cols,
                                                       std::size_t gemm_workspace_bytes) {
    CutlassWorkspace<Allocator> out;
    out.w_fp16 = allocator.alloc(DType::FP16, {k, n});
    out.x_fp16 = allocator.alloc(DType::FP16, {k, cols});
    if (gemm_workspace_bytes > 0) { out.gemm_workspace = allocator.alloc_bytes(gemm_workspace_bytes); }
    return out;
}

} // namespace

std::size_t nvfp4_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k,
                                               std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    cutlass::gemm::GemmCoord problem_size(cols, n, k);
    typename Gemm::Arguments arguments{
        problem_size, {nullptr, k}, {nullptr, k}, {nullptr, n}, {nullptr, n},
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}, 1};
    const std::size_t gemm_workspace_bytes = Gemm::get_workspace_size(arguments);
    (void)allocate_cutlass_workspace(layout, n, k, cols, gemm_workspace_bytes);
    return layout.peak_bytes(1);
}

void nvfp4_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream) {
    const std::int32_t k    = x.ne[0];
    const std::int32_t cols = x.ne[1];
    const std::int32_t n    = w.n;
    cutlass::gemm::GemmCoord problem_size(cols, n, k);
    typename Gemm::Arguments sizing_arguments{
        problem_size, {nullptr, k}, {nullptr, k}, {nullptr, n}, {nullptr, n},
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}, 1};
    const std::size_t gemm_workspace_bytes = Gemm::get_workspace_size(sizing_arguments);

    auto scratch_scope = ws.scope();
    CutlassWorkspace<WorkspaceArena> scratch =
        allocate_cutlass_workspace(ws, n, k, cols, gemm_workspace_bytes);
    auto* w_fp16 = static_cast<cutlass::half_t*>(scratch.w_fp16.data);
    auto* x_fp16 = static_cast<cutlass::half_t*>(scratch.x_fp16.data);

    const dim3 block(256);
    if (w.layout == QuantLayout::VoltaQpnPrepacked) {
        const dim3 grid(static_cast<unsigned>(div_up_i(k / 8, 256)), static_cast<unsigned>(n), 1u);
        dequant_nvfp4_qpn_to_fp16<<<grid, block, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), n, k, 1.0F / w.weight_scale_divisor,
            w_fp16);
    } else {
        const dim3 grid(static_cast<unsigned>(div_up_i(k / 2, 256)), static_cast<unsigned>(n), 1u);
        dequant_nvfp4_row_to_fp16<<<grid, block, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), n, k, 1.0F / w.weight_scale_divisor,
            w_fp16);
    }
    CUDA_CHECK(cudaGetLastError());

    const std::int64_t x_count = static_cast<std::int64_t>(cols) * k;
    const int x_blocks         = static_cast<int>((x_count + 255) / 256);
    bf16_to_fp16_kernel<<<x_blocks, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                      x_fp16, x_count);
    CUDA_CHECK(cudaGetLastError());

    Gemm gemm_op;
    typename Gemm::Arguments arguments{
        problem_size, {x_fp16, k}, {w_fp16, k}, {static_cast<ElementOutput*>(out.data), n},
        {static_cast<ElementOutput*>(out.data), n},
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}, 1};
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("nvfp4_cutlass_sm70: CUTLASS can_implement failed");
    }
    status = gemm_op.initialize(arguments, scratch.gemm_workspace.data, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("nvfp4_cutlass_sm70: CUTLASS initialize failed");
    }
    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("nvfp4_cutlass_sm70: CUTLASS gemm() failed");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
