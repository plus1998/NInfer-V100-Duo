#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_attn_input_cutlass_workspace_bytes(std::int32_t tokens);
[[nodiscard]] std::size_t nvfp4_attn_input_cutlass_shard_workspace_bytes(std::int32_t tokens);
void nvfp4_attn_input_cutlass_sm70_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& gate, Tensor& k, Tensor& v,
                                          WorkspaceArena& workspace, cudaStream_t stream);
void nvfp4_attn_input_cutlass_sm70_launch_shard(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& gate, Tensor& k, Tensor& v,
                                                WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
