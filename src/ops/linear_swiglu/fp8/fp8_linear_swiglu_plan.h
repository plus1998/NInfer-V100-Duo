#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                     std::int32_t min_tokens,
                                                                     std::int32_t max_tokens);
[[nodiscard]] std::size_t fp8_linear_swiglu_shard_workspace_capacity_bytes(
    LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens);

void fp8_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream);
void fp8_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream);
void fp8_linear_swiglu_a8_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                 WorkspaceArena& workspace, cudaStream_t stream);

#ifdef NINFER_VOLTA_BUILD
void fp8_linear_swiglu_volta_qpn_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
[[nodiscard]] bool fp8_linear_swiglu_volta_qpn_supported(std::int32_t k, std::int32_t t) noexcept;

void fp8_linear_swiglu_qpn_split_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        float* gate_scratch, float* up_scratch,
                                        cudaStream_t stream);
[[nodiscard]] bool fp8_linear_swiglu_qpn_split_supported(std::int32_t n, std::int32_t k,
                                                          std::int32_t t) noexcept;
#endif

void fp8_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream);

// --- tp2 column-shard forms (Fp8MlpGateUpTp2ColumnGeometry, 17408x5120) ------------------------
//
// Same kernel templates as the tp1 forms above, instantiated at the shard's halved N (see
// src/ops/linear/fp8/fp8_config.h and each .cu file's Geometry template parameter). Route
// selection (resolve_route) is inherited from the parent unchanged -- a pure function of
// (policy, token count), not of N/K. Workspace is sized separately because the QPN scratch and
// materialized CUTLASS plane both scale with the shard's halved output extent.
void fp8_linear_swiglu_decode_launch_shard(const Tensor& x, const Weight& weight, Tensor& out,
                                           cudaStream_t stream);
void fp8_linear_swiglu_small_t_launch_shard(const Tensor& x, const Weight& weight, Tensor& out,
                                            cudaStream_t stream);
void fp8_linear_swiglu_a8_launch_shard(const Tensor& x, const Weight& weight, Tensor& out,
                                       WorkspaceArena& workspace, cudaStream_t stream);

// Takes a NULLABLE WorkspaceArena*, matching linear.h/linear_add.h/linear_swiglu.h's tp2 split
// convention (a rank may legitimately need zero workspace); a route that resolves to A8 with a
// null workspace throws, the same way ops::linear's own fp8_dispatch does.
void fp8_linear_swiglu_dispatch_shard(const Tensor& x, const Weight& weight, Tensor& out,
                                      LinearPolicy policy, WorkspaceArena* workspace,
                                      cudaStream_t stream);

} // namespace ninfer::ops::detail
