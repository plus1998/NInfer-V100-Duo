#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                   std::int32_t min_tokens,
                                                                   std::int32_t max_tokens);
[[nodiscard]] std::size_t nvfp4_gdn_input_shard_workspace_capacity_bytes(
    LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens);

void nvfp4_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream);

void nvfp4_gdn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream);

void nvfp4_gdn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 Nvfp4W4a4Workspace workspace, cudaStream_t stream);

void nvfp4_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                              LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream);

// --- tp2 column-shard siblings, instantiated at Nvfp4GdnInputTp2ColumnGeometry (registered in
// src/ops/linear/nvfp4/nvfp4_config.h). Mirrors attn_input_proj's `_shard` naming exactly. ---

void nvfp4_gdn_input_decode_launch_shard(const Tensor& x, const Weight& weight, Tensor& qkv,
                                         Tensor& z, cudaStream_t stream);

void nvfp4_gdn_input_small_t_launch_shard(const Tensor& x, const Weight& weight, Tensor& qkv,
                                          Tensor& z, cudaStream_t stream);

void nvfp4_gdn_input_w4a4_launch_shard(const Tensor& x, const Weight& weight, Tensor& qkv,
                                       Tensor& z, Nvfp4W4a4Workspace workspace,
                                       cudaStream_t stream);

void nvfp4_gdn_input_dispatch_shard(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    LinearPolicy policy, WorkspaceArena* workspace,
                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
