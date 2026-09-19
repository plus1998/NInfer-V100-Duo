#include "ops/attn_input_proj/fp8/fp8_attn_input_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_cutlass_sm70.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
struct CutlassSections;

template <>
struct CutlassSections<Fp8AttnInputGeometry> {
    static constexpr std::int32_t kQueryRows = kFp8AttnInputQueryRows;
    static constexpr std::int32_t kKeyRows   = kFp8AttnInputKeyRows;
};

template <>
struct CutlassSections<Fp8AttnInputTp2ColumnGeometry> {
    static constexpr std::int32_t kQueryRows = 3072;
    static constexpr std::int32_t kKeyRows   = 512;
};

template <class Geometry, class Allocator>
Tensor allocate_projected(Allocator& allocator, int tokens) {
    return allocator.alloc(DType::BF16, {Geometry::kOutputRows, tokens});
}

template <class Geometry>
__global__ void split_attn_output(const __nv_bfloat16* __restrict__ projected,
                                  __nv_bfloat16* __restrict__ q,
                                  __nv_bfloat16* __restrict__ gate,
                                  __nv_bfloat16* __restrict__ k,
                                  __nv_bfloat16* __restrict__ v, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    using Sections = CutlassSections<Geometry>;
    constexpr std::int32_t kGateRows  = Sections::kQueryRows;
    constexpr std::int32_t kKeyBegin  = Sections::kQueryRows;
    constexpr std::int32_t kGateBegin = kKeyBegin + Sections::kKeyRows;
    constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;
    const int rows = Geometry::kOutputRows;
    const int row   = static_cast<int>(i % rows);
    const int token = static_cast<int>(i / rows);
    if (row < kKeyBegin) {
        q[static_cast<std::int64_t>(token) * Sections::kQueryRows + row] = projected[i];
    } else if (row < kGateBegin) {
        k[static_cast<std::int64_t>(token) * Sections::kKeyRows + row - kKeyBegin] =
            projected[i];
    } else if (row < kValueBegin) {
        gate[static_cast<std::int64_t>(token) * kGateRows + row - kGateBegin] =
            projected[i];
    } else {
        v[static_cast<std::int64_t>(token) * Sections::kKeyRows + row - kValueBegin] =
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
void launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
            WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope       = workspace.scope();
    Tensor projected = allocate_projected<Geometry>(workspace, x.ne[1]);
    fp8_cutlass_sm70_launch(x, weight, projected, workspace, stream);
    const std::int64_t count = static_cast<std::int64_t>(x.ne[1]) * Geometry::kOutputRows;
    split_attn_output<Geometry><<<static_cast<int>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(v.data), count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::size_t fp8_attn_input_cutlass_workspace_bytes(std::int32_t tokens) {
    return workspace_bytes<Fp8AttnInputGeometry>(tokens);
}

std::size_t fp8_attn_input_cutlass_shard_workspace_bytes(std::int32_t tokens) {
    return workspace_bytes<Fp8AttnInputTp2ColumnGeometry>(tokens);
}

void fp8_attn_input_cutlass_sm70_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                        Tensor& gate, Tensor& k, Tensor& v,
                                        WorkspaceArena& workspace, cudaStream_t stream) {
    launch<Fp8AttnInputGeometry>(x, weight, q, gate, k, v, workspace, stream);
}

void fp8_attn_input_cutlass_sm70_launch_shard(const Tensor& x, const Weight& weight, Tensor& q,
                                              Tensor& gate, Tensor& k, Tensor& v,
                                              WorkspaceArena& workspace, cudaStream_t stream) {
    launch<Fp8AttnInputTp2ColumnGeometry>(x, weight, q, gate, k, v, workspace, stream);
}

} // namespace ninfer::ops::detail
