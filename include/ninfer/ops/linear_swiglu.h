#pragma once

// ninfer::ops - fused gate/up projection followed by SwiGLU.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/allreduce.h" // ExecutionContext (tp2 split form)
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the transient capacity required by LinearSwiGLU for every T in the inclusive
 * [min_tokens,max_tokens] interval. The QType and dimensions are the fixed implementation profile.
 * Invalid profiles or intervals throw; a legal static-zero route returns zero.
 */
[[nodiscard]] std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype,
                                                                 std::int32_t gate_up_rows,
                                                                 std::int32_t input_rows,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens);

/**
 * Policy-bearing capacity query. Q4/GGML_K/W8 admit A16Only. NVFP4 admits A16Only through its
 * qualified small-T domain (Volta extends this with the QPN split) and AllowA4 for every positive
 * T. Row-scaled FP8 admits A16Only and AllowA8 for every positive T.
 * A permissive policy covers whichever qualified route the private resolver selects across the
 * requested interval.
 */
[[nodiscard]] std::size_t
linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                       std::int32_t input_rows, LinearPolicy policy,
                                       std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Op: linear_swiglu
 *
 * Math / indexing:
 *   gate_up = Linear(x, gate_up_weight); M=gate_up_rows/2;
 *   ideal[i,t] = SiLU(gate_up[i,t]) * gate_up[M+i,t].
 *
 * Logical shapes / supported domain:
 *   T may be any positive value. The registered profiles are:
 *   - Q4G64_F16S weight [34816,5120], x [5120,T], out [17408,T];
 *   - GGML_K weight [34816,5120], x [5120,T], out [17408,T];
 *   - W8G32_F16S weight [12288,2048], x [2048,T], out [6144,T];
 *   - NVFP4 BlockScaleK16M128x4 weight [34816,5120], x [5120,T], out [17408,T];
 *   - FP8_E4M3FN_ROW_BF16S RowScale weight [34816,5120], x [5120,T], out [17408,T].
 *   Inputs and output are contiguous BF16. Q4/W8 scales are FP16, NVFP4 scales are E4M3FN, and
 *   row-scaled FP8 has one BF16 multiplier per gate/up parent row. Gate rows `[0,17408)` precede
 *   their matching up rows `[17408,34816)`.
 *
 * Numeric:
 *   The oracle exact-decodes the registered weight and evaluates `ideal` naively in FP64 from the
 *   represented inputs. The BF16 output is promoted and compared directly with that result; output
 *   storage rounding belongs to LinearSwiGLU's named activation-compute criterion, not the oracle.
 *   Production routes may fuse or materialize gate/up and may choose their natural accumulator,
 *   staging, and workspace precision; those private choices are not semantic rounding boundaries.
 *   Under AllowA8, row-scaled FP8 resolves T=1 and every T>=3 to A8 and T=2 to fused A16 SIMT;
 *   A16Only uses fused A16 kernels for every positive T.
 *
 * Effects:
 *   Writes the full output; x/weight and output must not alias.
 *
 * Workspace:
 *   Caller-owned transient storage reported by linear_swiglu_workspace_capacity_bytes(),
 *   scoped to the call. W8, NVFP4 A16, and row-scaled FP8 A16 require zero bytes; GGML_K and the
 *   Q4 composed split route use a projected BF16 plane, while A4/A8 routes use caller-owned
 *   activation storage and may use private projection storage. There is no persistent
 *   state side effect.
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena& ws, cudaStream_t stream);

/**
 * A16-only convenience form. Q4/GGML_K/W8 and row-scaled FP8 retain their complete positive-T
 * domain. NVFP4's exact A16 domain follows the registered device profile; larger extents require
 * the policy-bearing form when that profile provides a fallback.
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream);

// --- Tensor-parallel split form (tp == 2) -------------------------------------------------------
//
// linear_swiglu is a COLUMN-parallel (output-split) op only: variant.cpp's real MLP call graph
// (Variant::post_mixer, src/targets/qwen3_6_27b/impl/variant.cpp) is
//
//   ops::linear_swiglu(hidden, weights.gate_up, activation, ...);   // THIS op -- gate_up + SiLU
//   ops::linear_add(activation, weights.down, residual, ...);       // a SEPARATE op
//
// so linear_swiglu's own semantic contract stops at `activation`; the down projection, its
// residual, and its all-reduce belong to linear_add_row_parallel() (include/ninfer/ops/
// linear_add.h), which already registers the mlp/down [5120,17408] row-shard problems. This
// split form does not touch down/residual at all -- do not re-implement what a sibling enabled.
//
// The ShardPlan (src/targets/qwen3_6_27b/impl/load/bindings.cpp, `ends("mlp/gate_up")`) splits
// gate_up [34816,5120] as TWO column blocks -- gate [0,17408) and up [17408,34816) -- each split
// independently by tp, so rank r's shard weight is the CONCATENATION of gate rows
// [r*8704,(r+1)*8704) and up rows [17408+r*8704,17408+(r+1)*8704): a standalone [17408,5120]
// tensor in the SAME "gate rows [0,M) then up rows [M,2M)" layout linear_swiglu() itself already
// assumes, at M=8704 instead of M=17408. Rank r therefore computes
//
//   activation[r][i,t] = SiLU(gate_up[r][i,t]) * gate_up[r][M+i,t],   i in [0,8704), M=8704,
//
// entirely from its own shard and its own (replicated) activation block -- no cross-device
// traffic, matching linear_column_parallel()'s own column-parallel contract. Rank r's output IS
// directly the down-projection's rank r input block (linear_add_row_parallel()'s `x[r]`); the two
// Ops compose end to end without any reshaping between them.
//
// The registered split formats are the following, per the real ShardPlan/binding profiles (bindings.cpp
// `bind_nvfp4_text_layers`/`bind_groupwise_text_layers`/`bind_qwen38_nvfp4_text_layers`):
//   - NVFP4: a TRUE split. The fused decode/small-T/W4A4/TMA kernels are templated on Geometry
//     (src/ops/linear_swiglu/nvfp4/*.cu) and instantiated at BOTH Nvfp4MlpGateUpGeometry (tp1) and
//     Nvfp4MlpGateUpTp2ColumnGeometry (the shard, src/ops/linear/nvfp4/nvfp4_config.h) -- the same
//     kernel template halves its own indexing at the shard's M, the same way every other split
//     family here works: a shard is a standalone tensor of the same layout with one axis narrowed,
//     so the existing kernel instantiated at that shape IS the split kernel. Route selection
//     (resolve_route) is a pure function of (policy, token count), inherited unchanged from the
//     parent.
//   - FP8_E4M3FN_ROW_BF16S (the flagship's own MLP-tail format, Text layers 56-63): a
//     TRUE split, exactly the same shape of change as NVFP4's above. The fused decode/small-T/A8
//     kernels are templated on Geometry (src/ops/linear_swiglu/fp8/*.cu) and instantiated at both
//     Fp8MlpGateUpGeometry (tp1) and Fp8MlpGateUpTp2ColumnGeometry (the shard,
//     src/ops/linear/fp8/fp8_config.h). Route selection is inherited unchanged from the tp1
//     resolve_route (a pure function of (policy, token count)).
//   - GGML_K (native Qwen3.8 Q4_K_M): each rank owns a row-sliced gate/up parent and uses the
//     shared direct GGML decoder followed by the standalone SiLU-multiply epilogue.
//   - Q4G64_F16S (groupwise profile): COMPOSED, not extended. Every one of Q4's own linear_swiglu
//     kernels (src/ops/linear_swiglu/q4/*.cu) is a compile-time-exact template hardcoded to
//     [34816,17408,5120] with no runtime-N/K escape hatch, the same shape of limitation BF16's
//     linear_add family has. Rather than instantiate a whole new exact kernel set for one shard
//     shape, the shard composes the ALREADY tp2-shard-capable ops::linear() (whose Q4 registry
//     admits [17408,5120] as a column geometry) with the standalone silu_mul() Op -- the exact
//     fallback Q4's OWN tp1 "Materialized" route (src/ops/linear_swiglu/q4/q4_linear_swiglu_plan.
//     cpp) already takes above T=48, just requiring a workspace at every token count here (the
//     tp1 Materialized route needs none only because ops::linear()'s A16-only Q4 launchers happen
//     to need none at ANY registered N/K, per ops::linear's own Q4 dispatch; Q4's A16-only
//     convenience overload of THIS Op therefore still requires a non-null workspace, unlike
//     NVFP4's).
//
// W8G32_F16S is NOT registered here: its linear_swiglu profile ([12288,2048]) belongs only to the
// MTP head, which stays single-device (never sharded by the qwen3_6_27b ShardPlan).
//
// Every requirement of linear_swiglu() applies per rank (BF16 dtype, contiguity, 16-byte
// alignment, x/out non-aliasing). The caller obligations of src/ops/common/split_launch.h apply
// unchanged: rank r's work is issued on ec.dev[r]->stream with that rank's device current, and
// activations staged with the plain cudaMemcpy/cudaMemset/<<<...>>> forms must be retired (they
// land on the LEGACY DEFAULT stream, which does not implicitly synchronize with
// DeviceContext::stream) before this call reads them.

/**
 * Returns the caller-owned transient capacity required by linear_swiglu_column_parallel() for
 * every T in [min_tokens,max_tokens], evaluated at the SHARD shape (gate_up_rows=17408,
 * input_rows=5120 for the registered formats). NVFP4 admits A16Only and AllowA4; Q4G64_F16S and
 * GGML_K admit only A16Only. Q4/GGML_K split routes use a caller-owned projected plane, so this
 * function reports their nonzero capacity. Invalid formats, policies, or intervals throw.
 */
[[nodiscard]] std::size_t linear_swiglu_column_parallel_workspace_capacity_bytes(
    QType qtype, LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * @brief Column-parallel (output-split) linear_swiglu across two devices.
 *
 * Rank r computes `out[r] = SiLU(gate_up[r]) * up[r]` over its own shard and replicated
 * activation, per the design note above. `w[0].k` must equal `w[1].k` and both ranks must agree
 * on the token count; the per-rank shard row counts `w[r].n` need not be equal.
 *
 * @param[in] x Per-rank replicated BF16 activation `[K,T]`.
 * @param[in] w Per-rank gate_up weight shard `[N_r,K]` (N_r even; rows `[0,N_r/2)` are this rank's
 * gate slice, `[N_r/2,N_r)` its matching up slice).
 * @param[out] out Per-rank BF16 activation block `[N_r/2,T]`.
 * @param[in] policy Permitted private activation-compute profiles, applied identically per rank.
 * @param[in,out] workspace Per-rank caller-owned transient arena. NVFP4 may pass null for routes
 * that need none; Q4G64_F16S and GGML_K require an arena for their projected plane.
 * @param[in] ec Execution context holding exactly two distinct devices.
 */
void linear_swiglu_column_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                                   const std::array<Tensor, 2>& out, LinearPolicy policy,
                                   const std::array<WorkspaceArena*, 2>& workspace,
                                   const ExecutionContext& ec);

/**
 * A16-only column-parallel convenience form. Passes a null workspace per rank, so it admits NVFP4
 * (whose A16 routes need none) but rejects Q4G64_F16S with a clear throw -- use the policy-bearing
 * overload with an allocated arena for Q4.
 */
void linear_swiglu_column_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                                   const std::array<Tensor, 2>& out, const ExecutionContext& ec);

} // namespace ninfer::ops
