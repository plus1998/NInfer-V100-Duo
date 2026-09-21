#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                       std::int32_t min_tokens,
                                                                       std::int32_t max_tokens);

void nvfp4_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream);
void nvfp4_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void nvfp4_linear_swiglu_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     WorkspaceArena& workspace, cudaStream_t stream);

#ifdef NINFER_VOLTA_BUILD
void nvfp4_linear_swiglu_volta_qpn_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          cudaStream_t stream);
[[nodiscard]] bool nvfp4_linear_swiglu_volta_qpn_supported(std::int32_t k, std::int32_t t) noexcept;

void nvfp4_linear_swiglu_qpn_split_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          float* gate_scratch, void* activation_scratch,
                                          cudaStream_t stream);
[[nodiscard]] bool nvfp4_linear_swiglu_qpn_split_supported(std::int32_t n, std::int32_t k,
                                                            std::int32_t t) noexcept;
#endif

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena& workspace,
                                  cudaStream_t stream);

// --- tp2 column-shard forms (Nvfp4MlpGateUpTp2ColumnGeometry, 17408x5120) ----------------------
//
// Same kernel templates as the tp1 forms above, instantiated at the shard's halved N (see
// src/ops/linear/nvfp4/nvfp4_config.h and each .cu file's Geometry template parameter). Route
// selection (resolve_route) is inherited from the parent unchanged -- it is a pure function of
// (policy, token count), not of N/K -- so the shard admits exactly the tp1 domain at each policy.
// The dispatch form takes a NULLABLE WorkspaceArena*, matching linear.h/linear_add.h's tp2 split
// convention (a rank may legitimately need zero workspace); a route that resolves to a W4A4 path
// with a null workspace throws, the same way ops::linear's own nvfp4_dispatch does.
[[nodiscard]] std::size_t
nvfp4_linear_swiglu_shard_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                   std::int32_t max_tokens);

void nvfp4_linear_swiglu_decode_launch_shard(const Tensor& x, const Weight& weight, Tensor& out,
                                             cudaStream_t stream);
void nvfp4_linear_swiglu_small_t_launch_shard(const Tensor& x, const Weight& weight, Tensor& out,
                                              cudaStream_t stream);
void nvfp4_linear_swiglu_w4a4_launch_shard(const Tensor& x, const Weight& weight, Tensor& out,
                                           WorkspaceArena& workspace, cudaStream_t stream);

void nvfp4_linear_swiglu_dispatch_shard(const Tensor& x, const Weight& weight, Tensor& out,
                                        LinearPolicy policy, WorkspaceArena* workspace,
                                        cudaStream_t stream);

} // namespace ninfer::ops::detail
