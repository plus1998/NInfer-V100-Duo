// DFlash2 draft QKV projection — sm_70.
//
// W8G32 RowSplit [6144, 5120] -> q [4096, T], k [1024, T], v [1024, T], written directly with the
// three-way split output policy. Runs the fork's proven warp-per-row Volta SIMT GEMM; the split
// points 4096 and 5120 are multiples of the 8-row CTA tile.

#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/token_slices.h"
#include "ops/linear/w8/w8_rowsplit_gemm_simt.cuh"
#include "ops/linear/w8/w8_rowsplit_output.cuh"
#include "ops/linear/w8/w8_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kQueryRows = 4096;
constexpr int kKvRows    = 1024;
constexpr int kParentRows = 6144;
constexpr int kColsPerWarp = 8;
constexpr int kRowsPerBlock = 8;
constexpr int kStagesLocal  = 2;
using Split = W8SplitOutput3<kQueryRows, kKvRows, kKvRows>;

void launch_tile(const __nv_bfloat16* xp, const std::uint8_t* codes, const std::uint8_t* scales,
                 Split output, std::int32_t k, std::int32_t t, std::int32_t padded_k,
                 std::int32_t full_slabs, cudaStream_t stream) {
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    constexpr int kRowsPerCta   = kRowsPerBlock; // ColWarpsPerRow == 1
    constexpr int kColsPerCta   = kColsPerWarp;
    const dim3 grid(static_cast<unsigned>((kParentRows + kRowsPerCta - 1) / kRowsPerCta),
                    static_cast<unsigned>((t + kColsPerCta - 1) / kColsPerCta), 1u);
    w8_rowsplit_gemm_simt_kernel<W8RowSplitSimtSchedule, kColsPerWarp, kRowsPerBlock, kStagesLocal,
                                 /*Full=*/false, W8Epilogue::Store, Split, /*ColWarpsPerRow=*/1>
        <<<grid, kBlockThreads, 0, stream>>>(xp, codes, scales, output, kParentRows, k, t, padded_k,
                                             full_slabs);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void w8_dflash2_attn_input_volta_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                        Tensor& v, cudaStream_t stream) {
    const auto* xp = static_cast<const __nv_bfloat16*>(x.data);
    const Split output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                       static_cast<__nv_bfloat16*>(v.data)};
    const auto* codes  = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales);
    const std::int32_t kk       = x.ne[0];
    const std::int32_t padded_k = weight.padded_shape[1];

    for_each_token_slice(x.ne[1], kColsPerWarp, [&](std::int32_t offset, std::int32_t count) {
        const auto* xslice = xp + static_cast<std::int64_t>(offset) * kk;
        Split slice{output.out0 + static_cast<std::int64_t>(offset) * kQueryRows,
                    output.out1 + static_cast<std::int64_t>(offset) * kKvRows,
                    output.out2 + static_cast<std::int64_t>(offset) * kKvRows};
        const bool aligned_x =
            (kk % 8) == 0 && (reinterpret_cast<std::uintptr_t>(xslice) & 0xfu) == 0;
        const std::int32_t full_slabs = aligned_x ? kk / 1024 : 0;
        launch_tile(xslice, codes, scales, slice, kk, count, padded_k, full_slabs, stream);
    });
}

void w8_dflash2_tp2_attn_input_volta_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                             Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] >= 32) {
        const auto section = [&](int offset, int rows) {
            Weight shard = weight;
            shard.qdata = static_cast<const std::uint8_t*>(weight.qdata) +
                          static_cast<std::size_t>(offset) * 5120;
            shard.scales = static_cast<const std::uint8_t*>(weight.scales) +
                           static_cast<std::size_t>(offset) * 5120 / 32 * 2;
            shard.n = rows;
            shard.shape[0] = rows;
            shard.padded_shape[0] = rows;
            return shard;
        };
        launch_w8_volta_mma(x, section(0, 2048), q, stream);
        launch_w8_volta_mma(x, section(2048, 512), k, stream);
        launch_w8_volta_mma(x, section(2560, 512), v, stream);
        return;
    }
    constexpr int kLocalQueryRows = 2048;
    constexpr int kLocalKvRows = 512;
    using LocalSplit = W8SplitOutput3<kLocalQueryRows, kLocalKvRows, kLocalKvRows>;
    const auto* xp = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales);
    auto* qo = static_cast<__nv_bfloat16*>(q.data);
    auto* ko = static_cast<__nv_bfloat16*>(k.data);
    auto* vo = static_cast<__nv_bfloat16*>(v.data);
    for_each_token_slice(x.ne[1], kColsPerWarp, [&](std::int32_t offset, std::int32_t count) {
        const LocalSplit slice{qo + static_cast<std::int64_t>(offset) * kLocalQueryRows,
                               ko + static_cast<std::int64_t>(offset) * kLocalKvRows,
                               vo + static_cast<std::int64_t>(offset) * kLocalKvRows};
        const dim3 grid(3072 / kRowsPerBlock,
                        static_cast<unsigned>((count + kColsPerWarp - 1) / kColsPerWarp), 1u);
        w8_rowsplit_gemm_simt_kernel<W8RowSplitSimtSchedule, kColsPerWarp, kRowsPerBlock,
                                     kStagesLocal, false, W8Epilogue::Store, LocalSplit, 1>
            <<<grid, kRowsPerBlock * 32, 0, stream>>>(
                xp + static_cast<std::int64_t>(offset) * 5120, codes, scales, slice,
                3072, 5120, count, 5120, 5);
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace ninfer::ops::detail
