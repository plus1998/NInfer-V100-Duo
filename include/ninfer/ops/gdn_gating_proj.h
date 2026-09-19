#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/allreduce.h" // ExecutionContext (tp2 split form)

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// The geometry is the fixed implementation profile. Each query covers every T in the inclusive
// interval; invalid profiles or intervals throw.
[[nodiscard]] std::size_t gdn_gating_proj_workspace_capacity_bytes(std::int32_t heads,
                                                                   std::int32_t input_rows,
                                                                   std::int32_t min_tokens,
                                                                   std::int32_t max_tokens);

// Transient capacity for the corresponding pre-normalized control Op. The query follows the same
// interval rules and does not make the optimized route part of the semantic API.
[[nodiscard]] std::size_t gdn_norm_gating_proj_workspace_capacity_bytes(std::int32_t heads,
                                                                        std::int32_t input_rows,
                                                                        std::int32_t min_tokens,
                                                                        std::int32_t max_tokens);

/**
 * Fuses two BF16 projections with Gated DeltaNet gate preparation. For each h,t:
 *
 *   a[h,t]    = sum_k a_weight[h,k] * x[k,t]
 *   b[h,t]    = sum_k b_weight[h,k] * x[k,t]
 *   g[h,t]    = -exp(A_log[h]) * softplus(a[h,t] + dt_bias[h])
 *   beta[h,t] = sigmoid(b[h,t]).
 *
 * `x` is contiguous BF16 [5120,T]; both weights are contiguous BF16_CTRL [48,5120]; A_log and
 * dt_bias are contiguous FP32 [48]; g and beta are distinct contiguous FP32 [48,T]. The numerical
 * contract accepts every positive T. The oracle evaluates the logical formula naively in FP64
 * from the represented inputs. Projection staging, accumulator precision, and any private
 * materialization are implementation choices. The FP32 outputs are promoted and compared directly
 * with that oracle under the Op's named criterion. All inputs and outputs are non-overlapping.
 * `ws` provides the transient capacity reported above and is scoped to the call; there is no
 * persistent state side effect.
 * The 27B GGML_K form preserves Q4_K/Q6_K [48,5120] rows and stored scales. It evaluates the
 * projections into FP32 and applies the same gating formula, using no transient workspace.
 */
void gdn_gating_proj(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                     const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g,
                     Tensor& beta, cudaStream_t stream);

/**
 * Registered contiguous-parent storage forms of gdn_gating_proj:
 *
 * - Qwen3.8-27B: BF16_CTRL `ab_weight [96,5120]`, with A in rows [0,48) and B in [48,96);
 * - Qwen3.6-35B-A3B: BF16_CTRL `ab_weight [64,2048]`, with A in rows [0,32) and B in [32,64).
 * - Qwen3.8-27B: GGML_K GgmlK256 `ab_weight [96,5120]`, in the same A/B row order.
 *
 * The complete immutable parent is the public weight. Its halves are consumed as zero-copy views
 * and produce FP32 g/beta `[heads,T]` under the same logical formula and oracle. All other effects
 * and non-overlap requirements match the two-weight form.
 * Its TP2 GGML_K form has 24 heads and [48,5120] parent rows per device, with FP32 projection
 * and output storage and no additional workspace.
 */
void gdn_gating_proj(const Tensor& x, const Weight& ab_weight, const Tensor& A_log,
                     const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g, Tensor& beta,
                     cudaStream_t stream);

/**
 * Applies the Qwen3.6 GDN input RMSNorm and control projection as one semantic Op:
 *
 *   n[k,t]    = x[k,t] * rsqrt(mean_j(x[j,t]^2) + eps) * (1 + norm_weight[k])
 *   h_ideal[k,t] = n[k,t]
 *   a[r,t]    = sum_k a_weight[r,k] * n[k,t]
 *   b[r,t]    = sum_k b_weight[r,k] * n[k,t]
 *   g[r,t]    = -exp(A_log[r]) * softplus(a[r,t] + dt_bias[r])
 *   beta[r,t] = sigmoid(b[r,t]).
 *
 * `h` is the explicit BF16 output consumed by the other GDN projections. Its BF16 values are
 * promoted and compared directly with `h_ideal`; final storage rounding is not reproduced inside
 * the oracle. The control branch is evaluated directly from `n` and is not required to round
 * through h. Private tensor-core operand staging remains an implementation choice. The tensor and
 * weight domains otherwise match the two-weight gdn_gating_proj form. The implementation may fuse
 * or compose its internal kernels for any positive T; that route is not observable at this
 * boundary.
 */
void gdn_norm_gating_proj(const Tensor& x, const Tensor& norm_weight, float eps,
                          const Weight& a_weight, const Weight& b_weight, const Tensor& A_log,
                          const Tensor& dt_bias, WorkspaceArena& ws, Tensor& h, Tensor& g,
                          Tensor& beta, cudaStream_t stream);

/** The Qwen3.8-27B and Qwen3.6-35B-A3B contiguous-parent storage forms described above. */
void gdn_norm_gating_proj(const Tensor& x, const Tensor& norm_weight, float eps,
                          const Weight& ab_weight, const Tensor& A_log, const Tensor& dt_bias,
                          WorkspaceArena& ws, Tensor& h, Tensor& g, Tensor& beta,
                          cudaStream_t stream);

// --- Tensor-parallel split form (tp == 2) -------------------------------------------------------
//
// DESIGN NOTE. The gdn_gating_proj interior row order is VERIFIED, not assumed (the ShardPlan
// boundary in bindings.cpp's plan_for and in tests/targets/test_shard_map.cpp rests on it):
// row h of a_weight/b_weight/A_log/dt_bias
// (h in [0,48)) is GDN value head h, and value head h belongs to qk (key) head group h / 3 -- the
// mapping the REAL GDN core computes: src/ops/linear_attention/gated_delta_net/common.cuh's
// `head_map::qk_head(h_v)` (`h_v / group_size()`, group_size = H_v/H_qk = 48/16 = 3), consumed by
// recurrent.cuh (decode) and chunked/{prepare_wy_wu.cuh,output.cuh} (prefill, which index g/beta
// with the same flat `t*H_v + h_v` layout this ShardPlan boundary assumes). The test-only reference
// model tests/ops/gdn_ref.h::qk_head mirrors this production mapping exactly and is where this was
// first checked, but the production kernels are the authority, and the same natural 0..47 head
// ordering the GDN input_projection's own V/Z sections already use (verified independently in
// gdn_input_proj's own ShardPlan derivation above) is consistent with it. So the 48 rows ARE laid
// out head-major (row = qk_group*3 + component): [0,24) holds qk groups 0..7 complete, [24,48)
// holds qk groups 8..15 complete -- ShardPlan's `append_column_block` boundary at row 24 is
// therefore correct as committed, confirmed rather than assumed.
//
// OUTPUT CONTRACT: each device writes g[24,T]/beta[24,T] -- this device's 24 of the parent's 48
// value heads (device 0: heads [0,24); device 1: heads [24,48)). The GDN core recovers a global
// value head index by adding `device_rank * 24` to the shard-local row.
//
// FORMATS: BF16_CTRL and GGML_K. The Qwen3.8 GGUF profile uses the latter and preserves the
// parent rows on each rank; no allreduce is needed because this is column-parallel only.

[[nodiscard]] std::size_t
gdn_gating_proj_column_parallel_workspace_capacity_bytes(std::int32_t min_tokens,
                                                          std::int32_t max_tokens);

void gdn_gating_proj_column_parallel(const std::array<Tensor, 2>& x,
                                     const std::array<Weight, 2>& a_weight,
                                     const std::array<Weight, 2>& b_weight,
                                     const std::array<Tensor, 2>& A_log,
                                     const std::array<Tensor, 2>& dt_bias,
                                     const std::array<WorkspaceArena*, 2>& ws,
                                     const std::array<Tensor, 2>& g, const std::array<Tensor, 2>& beta,
                                     const ExecutionContext& ec);

/** Fused-parent form: `ab_weight` is BF16_CTRL or GGML_K [96,5120], A in rows [0,48), B in [48,96). */
void gdn_gating_proj_column_parallel(const std::array<Tensor, 2>& x,
                                     const std::array<Weight, 2>& ab_weight,
                                     const std::array<Tensor, 2>& A_log,
                                     const std::array<Tensor, 2>& dt_bias,
                                     const std::array<WorkspaceArena*, 2>& ws,
                                     const std::array<Tensor, 2>& g, const std::array<Tensor, 2>& beta,
                                     const ExecutionContext& ec);

} // namespace ninfer::ops
