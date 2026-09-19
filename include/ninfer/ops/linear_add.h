#pragma once

// ninfer::ops - fused residual += W @ x.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/allreduce.h" // ExecutionContext, PeerEvents (tp2 split form)
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the A16-only transient capacity required by LinearAdd for every T in the inclusive
 * [min_tokens,max_tokens] interval. The QType and dimensions are the fixed implementation profile.
 * Invalid profiles or intervals throw; a legal static-zero route returns zero.
 */
[[nodiscard]] std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

[[nodiscard]] std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/**
 * Op: linear_add
 *
 * Math / indexing:
 *   ideal[:,t] = residual[:,t] + Linear(x,w)[:,t].
 *
 * Logical shapes:
 *   Contiguous BF16 x [K,T] and residual [N,T]. Registered weights are GGML_K `ggml-k256-v1`,
 *   Q5G64_F16S RowSplit
 *   [5120,17408] or [5120,6144], W8G32_F16S RowSplit [2048,4096] or [2048,6144], NVFP4
 *   BlockScaleK16M128x4 [5120,6144] or [5120,17408], row-scaled
 *   FP8_E4M3FN_ROW_BF16S [5120,6144] or [5120,17408], or BF16_CTRL Contiguous [5120,6144]. T may
 *   be any positive value.
 *
 * Numeric:
 *   The oracle reads a registered BF16 weight directly or exact-decodes a registered packed
 *   weight, then evaluates `ideal` naively in FP64 from the represented inputs. The updated BF16
 *   residual is promoted and compared directly with that result; output storage rounding belongs
 *   to LinearAdd's selected A16, A8, or A4 criterion, not the oracle. Production routes may fuse
 *   or materialize the projection and may choose their natural accumulator, activation
 *   quantization, staging, and workspace precision; those private choices are not semantic
 *   rounding boundaries.
 *
 * Compute policy:
 *   GGML_K, Q5, W8, and BF16_CTRL admit only A16Only. NVFP4 admits A16Only and AllowA4. Row-scaled FP8
 *   admits A16Only and AllowA8. Its two semantic registrations own independent production plans:
 *   [5120,6144] resolves T<22 to A16 and T>=22 to A8, while [5120,17408] resolves T<25 to A16 and
 *   T>=25 to A8. A permissive policy allows the private resolver to select either qualified
 *   arithmetic profile; it does not itself prescribe a kernel.
 *
 * Effects:
 *   Updates the full residual tensor in place; x/weight must not alias residual.
 *
 * Workspace:
 *   Caller-owned transient storage reported by linear_add_workspace_capacity_bytes(), scoped to
 *   the call. A16 routes require no storage; quantized-activation routes use the reported capacity.
 *   There is no persistent state side effect.
 */
void linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena& ws,
                cudaStream_t stream);

void linear_add(const Tensor& x, const Weight& w, Tensor& residual, LinearPolicy policy,
                WorkspaceArena& ws, cudaStream_t stream);

// Exact GDN input permutation fused into a GGML_K projection. The represented input
// is BF16 [128,3,H,T] (H=16, or H=8 per TP2 rank); the packed weight columns are
// [128,H,3]. No weight is requantized: ideal is residual + W @ transpose_heads(x).
// FP64 decodes W's original scales/codes and applies this permutation before the dot.
// Wide prefill uses caller workspace for the GGML_K SM70 tensor-core route. Split form adds the
// residual on rank 0 once and all-reduces both partial projections, following
// linear_add_row_parallel.
void ggml_k_gdn_output(const Tensor& x, const Weight& w, Tensor& residual,
                       WorkspaceArena& workspace, cudaStream_t stream);
void ggml_k_gdn_output(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                       const std::array<Tensor, 2>& residual,
                       const std::array<Tensor, 2>& staging,
                       const std::array<WorkspaceArena*, 2>& workspace,
                       const ExecutionContext& ec,
                       const PeerEvents& events);

// --- Tensor-parallel split form (tp == 2) -----------------------------------------------------
//
// linear_add is a ROW-parallel (input-split) op only: its output is the residual stream, which
// must stay the full, identical [N,T] tensor on both devices for every op downstream of it, so
// there is no column-parallel (output-split) form the way include/ninfer/ops/linear.h has one.
//
// Rank r owns a [K_r,T] activation block and the matching [N,K_r] weight-column shard, exactly as
// linear_row_parallel() does. The one thing this Op adds beyond that pattern is where the residual
// add happens: the tp1 kernels above FUSE it into the GEMM epilogue, but a row-parallel rank only
// ever holds a PARTIAL sum over its own K block, so fusing the (fully-formed, replicated) residual
// into every rank's partial would add it once per rank -- i.e. count it (tp==2) times instead of
// once. It must be added exactly once, and the reduction that combines the partials must not see
// two different bases.
//
// This Op resolves that by folding the residual into the reduction itself rather than by adding it
// again afterwards: rank 0 evaluates `residual = residual + partial_0` with the SAME fused kernels
// linear_add() above uses (residual's incoming value is its own per-rank replicated copy, which is
// correct because it enters the sum exactly once, from exactly one rank); rank 1 evaluates the pure
// GEMM partial `residual = partial_1` (linear(), no residual term, overwriting rank 1's copy, whose
// pre-call bytes are not needed again). The one `allreduce_sum(residual, staging, ec, events)` that
// follows then computes `(residual_in + partial_0) + partial_1`, which both ranks are left holding
// -- the residual added exactly once, before the reduce, with no separate post-reduce add and no
// change to allreduce_sum's own contract (it is called exactly as documented: summing two per-rank
// buffers of the collective's own dtype and shape). This is also numerically tighter than adding
// the residual as its own separate post-reduce step: rank 0's fused kernel rounds the GEMM-plus-
// residual sum to BF16 once instead of twice.
//
// Where a format's linear_add kernels are all EXACT-geometry templates with no runtime-dimensioned
// escape hatch for a halved K (BF16_CTRL today), rank 0 instead composes the already tp2-capable
// plain linear() at the shard shape with the standalone residual_add() Op
// (include/ninfer/ops/residual_add.h) -- the same qualified `x += y` computation allreduce_sum's
// own local combine uses -- so the arithmetic is identical either way; only which kernel performs
// the fused rounding differs. See the implementation for exactly which formats take which path.
//
// Every requirement of linear_add() applies per rank, and the caller obligations of
// linear_row_parallel() (per-rank device/stream residency, the legacy-default-stream trap) apply
// unchanged here too.
//
// Registered formats: GGML_K, NVFP4, Q5G64_F16S, and FP8_E4M3FN_ROW_BF16S are all
// TRUE splits -- each has a runtime-K-dimensioned linear_add kernel family, so rank 0 reaches it
// through dispatch_linear_add exactly as linear_add() itself does, at the halved-K shard geometry.
// BF16_CTRL is COMPOSED, not extended (see above -- its family has no runtime-K escape hatch).
// W8G32_F16S is not registered: its own linear_add profile belongs to a different (non-TP2)
// variant that this repository's ShardPlan never shards.
void linear_add_row_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                             const std::array<Tensor, 2>& residual,
                             const std::array<Tensor, 2>& staging, LinearPolicy policy,
                             const std::array<WorkspaceArena*, 2>& workspace,
                             const ExecutionContext& ec, const PeerEvents& events);

/// A16-only row-parallel form; requires no transient workspace.
void linear_add_row_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                             const std::array<Tensor, 2>& residual,
                             const std::array<Tensor, 2>& staging, const ExecutionContext& ec,
                             const PeerEvents& events);

} // namespace ninfer::ops
