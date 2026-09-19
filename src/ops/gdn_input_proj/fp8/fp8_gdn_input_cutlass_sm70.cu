#include "ops/gdn_input_proj/fp8/fp8_gdn_input_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_cutlass_sm70.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
struct CutlassSections;

template <>
struct CutlassSections<Fp8GdnInputGeometry> {
    static constexpr std::int32_t kQkvRows = Fp8GdnInputOutput::kQkvRows;
    static constexpr std::int32_t kZRows   = Fp8GdnInputOutput::kZRows;
};

template <>
struct CutlassSections<Fp8GdnInputTp2ColumnGeometry> {
    static constexpr std::int32_t kQkvRows = 5120;
    static constexpr std::int32_t kZRows   = 3072;
};

template <class Geometry, class Allocator>
Tensor allocate_projected(Allocator& allocator, int tokens) {
    return allocator.alloc(DType::BF16, {Geometry::kOutputRows, tokens});
}

template <class Geometry>
__global__ void split_gdn_output(const __nv_bfloat16* __restrict__ projected,
                                 __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ z,
                                 std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    using Output   = CutlassSections<Geometry>;
    const int rows = Geometry::kOutputRows;
    const int row   = static_cast<int>(i % rows);
    const int token = static_cast<int>(i / rows);
    if (row < Output::kQkvRows) {
        qkv[static_cast<std::int64_t>(token) * Output::kQkvRows + row] = projected[i];
    } else {
        z[static_cast<std::int64_t>(token) * Output::kZRows + row - Output::kQkvRows] =
            projected[i];
    }
}

template <class Geometry>
std::size_t workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected<Geometry>(layout, tokens);
    return layout.peak_bytes(1) +
           fp8_cutlass_sm70_workspace_bytes(Geometry::kOutputRows, Geometry::kInputRows, tokens);
}

template <class Geometry>
void launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
            WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope       = workspace.scope();
    Tensor projected = allocate_projected<Geometry>(workspace, x.ne[1]);
    fp8_cutlass_sm70_launch(x, weight, projected, workspace, stream);
    const std::int64_t count = static_cast<std::int64_t>(x.ne[1]) * Geometry::kOutputRows;
    split_gdn_output<Geometry><<<static_cast<int>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data), static_cast<__nv_bfloat16*>(qkv.data),
        static_cast<__nv_bfloat16*>(z.data), count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::size_t fp8_gdn_input_cutlass_workspace_bytes(std::int32_t tokens) {
    return workspace_bytes<Fp8GdnInputGeometry>(tokens);
}

std::size_t fp8_gdn_input_cutlass_shard_workspace_bytes(std::int32_t tokens) {
    return workspace_bytes<Fp8GdnInputTp2ColumnGeometry>(tokens);
}

void fp8_gdn_input_cutlass_sm70_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                       Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    launch<Fp8GdnInputGeometry>(x, weight, qkv, z, workspace, stream);
}

void fp8_gdn_input_cutlass_sm70_launch_shard(const Tensor& x, const Weight& weight, Tensor& qkv,
                                             Tensor& z, WorkspaceArena& workspace,
                                             cudaStream_t stream) {
    launch<Fp8GdnInputTp2ColumnGeometry>(x, weight, qkv, z, workspace, stream);
}

} // namespace ninfer::ops::detail
