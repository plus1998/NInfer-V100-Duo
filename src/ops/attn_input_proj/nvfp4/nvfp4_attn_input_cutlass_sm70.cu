#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

template <class Geometry, class Allocator>
Tensor allocate_projected(Allocator& allocator, std::int32_t tokens) {
    return allocator.alloc(DType::BF16, {Geometry::kOutputRows, tokens});
}

template <class Geometry>
__global__ void split_output(const __nv_bfloat16* __restrict__ projected,
                             __nv_bfloat16* __restrict__ q,
                             __nv_bfloat16* __restrict__ gate,
                             __nv_bfloat16* __restrict__ k,
                             __nv_bfloat16* __restrict__ v, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    using Sections                      = Nvfp4AttnInputSections<Geometry>;
    constexpr std::int32_t kQueryRows  = Sections::kQueryRows;
    constexpr std::int32_t kKeyRows    = Sections::kKeyRows;
    constexpr std::int32_t kKeyBegin   = kQueryRows;
    constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
    constexpr std::int32_t kValueBegin = kGateBegin + kQueryRows;
    const std::int32_t row = static_cast<std::int32_t>(i % Geometry::kOutputRows);
    const std::int32_t token = static_cast<std::int32_t>(i / Geometry::kOutputRows);
    if (row < kKeyBegin) {
        q[static_cast<std::int64_t>(token) * kQueryRows + row] = projected[i];
    } else if (row < kGateBegin) {
        k[static_cast<std::int64_t>(token) * kKeyRows + row - kKeyBegin] = projected[i];
    } else if (row < kValueBegin) {
        gate[static_cast<std::int64_t>(token) * kQueryRows + row - kGateBegin] = projected[i];
    } else {
        v[static_cast<std::int64_t>(token) * kKeyRows + row - kValueBegin] = projected[i];
    }
}

template <class Geometry>
std::size_t workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected<Geometry>(layout, tokens);
    return layout.peak_bytes(1) + nvfp4_cutlass_sm70_workspace_bytes(
                                      Geometry::kOutputRows, Geometry::kInputRows, tokens);
}

template <class Geometry>
void launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
            WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope       = workspace.scope();
    Tensor projected = allocate_projected<Geometry>(workspace, x.ne[1]);
    nvfp4_cutlass_sm70_launch(x, weight, projected, workspace, stream);
    const std::int64_t count = static_cast<std::int64_t>(x.ne[1]) * Geometry::kOutputRows;
    split_output<Geometry><<<static_cast<int>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(v.data), count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::size_t nvfp4_attn_input_cutlass_workspace_bytes(std::int32_t tokens) {
    return workspace_bytes<Nvfp4AttnInputGeometry>(tokens);
}

std::size_t nvfp4_attn_input_cutlass_shard_workspace_bytes(std::int32_t tokens) {
    return workspace_bytes<Nvfp4AttnInputTp2ColumnGeometry>(tokens);
}

void nvfp4_attn_input_cutlass_sm70_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& gate, Tensor& k, Tensor& v,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    launch<Nvfp4AttnInputGeometry>(x, weight, q, gate, k, v, workspace, stream);
}

void nvfp4_attn_input_cutlass_sm70_launch_shard(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& gate, Tensor& k, Tensor& v,
                                                WorkspaceArena& workspace, cudaStream_t stream) {
    launch<Nvfp4AttnInputTp2ColumnGeometry>(x, weight, q, gate, k, v, workspace, stream);
}

} // namespace ninfer::ops::detail
