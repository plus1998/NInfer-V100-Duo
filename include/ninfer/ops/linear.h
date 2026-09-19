#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/allreduce.h" // ExecutionContext, PeerEvents (tp2 split forms)

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * @brief Permitted private activation-compute profiles for a linear projection.
 *
 * The policy constrains private route selection; it does not select a kernel or prescribe a
 * particular MMA instruction. The public activation and output tensors remain BF16 for every
 * policy.
 */
enum class LinearPolicy : std::uint8_t {
    A16Only, ///< Admit only A16 compute profiles.
    AllowA8, ///< Admit either A16 or A8 compute profiles.
    AllowA4, ///< Admit either A16 or A4 compute profiles.
};

/**
 * Returns the caller-owned transient capacity required by Linear for every T in the inclusive
 * `[min_tokens,max_tokens]` interval. Invalid registered profiles, policies, or intervals throw;
 * a legal route that requires no transient storage returns zero.
 */
[[nodiscard]] std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                          std::int32_t input_rows,
                                                          LinearPolicy policy,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);

/**
 * @brief Applies a bias-free matrix projection independently to every input column.
 *
 * @details The ideal mathematical result is
 *
 * @f[
 *   \mathrm{ideal}_{n,t} =
 *   \sum_{k=0}^{K-1}
 *     \mathrm{FP32Dequant}(w)_{n,k}\,\mathrm{FP32}(x_{k,t}).
 * @f]
 *
 * `out` stores a BF16 approximation of this ideal result under the named numerical criterion for
 * the selected private activation-compute path.
 *
 * @par Logical tensors and layout
 * `x` is contiguous, non-null, 16-byte-aligned BF16 `[K,T]`, `w` has logical shape `[N,K]`, and
 * `out` is contiguous, non-null, 16-byte-aligned BF16 `[N,T]`. Every logical extent is positive;
 * in particular, `T=0` is invalid rather than a no-op. Dimension zero is stored fastest. The Op has
 * no bias, activation, residual addition, or transpose mode.
 *
 * @par Supported execution domain
 * Registered execution uses GGML_K `ggml-k256-v1`, RowSplit Q4G64_F16S, Q5G64_F16S, Q6G64_F16S, or W8G32_F16S weights
 * with FP16 scales, block-scaled NVFP4 weights, row-scaled FP8_E4M3FN_ROW_BF16S weights, plus
 * registered contiguous BF16_CTRL problems. Each format owns a finite registry of exact physical
 * weight problems and selects its kernel internally; a valid encoding and alignment do not imply
 * arbitrary N/K support. FP8 currently registers `[N,K]` in `{[14336,5120], [16384,5120],
 * [34816,5120], [248320,5120], [5120,6144], [5120,17408]}` at every positive T. The current NVFP4
 * problems register the five non-vocabulary FP8 geometries and accept every positive T. Text and
 * MTP packed-weight problems accept every positive column extent T. Registered Vision problems
 * accept raw-patch P in `{4,8,...,131072}` or merged-token V in `[1,32768]`; a matrix column does
 * not inherently represent a text token. FP32_CTRL is unsupported.
 *
 * @par Numerical contract
 * Test fixture code materializes the persistent weight as its logical FP32 dequantized matrix.
 * The one Linear oracle accepts that matrix and the FP32 values represented by the BF16 activation,
 * evaluates every complete dot product with naive FP64 accumulation, and retains the FP64 result.
 * The BF16 output is promoted and compared against that result. Output representation,
 * accumulator precision, activation quantization, staging, reduction order, and kernel schedule
 * are private implementation effects covered by the named tolerance for the selected
 * activation-compute path; none is copied into the oracle. Kernel, schedule, template instance,
 * host launcher, and T region do not create separate criteria inside one path.
 *
 * @par Compute policy
 * `policy` specifies the permitted private activation-compute set. A permission does not require a
 * corresponding low-precision route: the resolved plan may remain A16 when that is the qualified
 * choice. BF16_CTRL and GGML_K admit only LinearPolicy::A16Only. Registered Q4/Q5/Q6/W8 formats admit
 * LinearPolicy::A16Only and LinearPolicy::AllowA8. The five non-vocabulary FP8 problems admit the
 * same two policies at every positive T. AllowA8 resolves `[14336,5120]` to A16 through T=11 and
 * A8 from T=12; `[16384,5120]` to A16 through T=10 and A8 from T=11; `[34816,5120]` to A8 at T=1,
 * A16 at T=2..4, and A8 from T=5; both `[5120,6144]` and `[5120,17408]` resolve T<25 to A16 and
 * T>=25 to A8. FP8 `[248320,5120]` admits A16Only, AllowA8, and AllowA4; every policy retains A16
 * compute at every positive T. NVFP4 admits A16Only and AllowA4; AllowA4 permits the private
 * resolver to select either a qualified A16 route or activation quantization to NVFP4 at every
 * positive T. The selected route depends only on the registered problem and T.
 *
 * @par Workspace
 * `workspace` is caller-owned call-scoped transient storage sized by
 * linear_workspace_capacity_bytes(). It must not overlap x, any weight plane, or out. Linear does
 * not allocate device memory internally.
 *
 * @param[in] x Contiguous, non-null, 16-byte-aligned BF16 input matrix `[K,T]`.
 * @param[in] w Logical weight matrix `[N,K]` in a registered persistent format and layout.
 * @param[out] out Contiguous, non-null, 16-byte-aligned BF16 output matrix `[N,T]`. It must not
 * overlap `x` or any weight plane.
 * @param[in] policy Permitted private activation-compute profiles.
 * @param[in,out] workspace Caller-owned transient arena.
 * @param[in] stream CUDA stream on which execution is enqueued.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream);

/**
 * @brief Applies the A16-only form of the bias-free matrix projection.
 *
 * @details This overload admits only A16 compute and requires no transient workspace. All tensor,
 * weight, aliasing, and execution-domain requirements of the policy-bearing overload apply.
 *
 * @param[in] x Contiguous BF16 input matrix `[K,T]`.
 * @param[in] w Logical weight matrix `[N,K]` in a registered persistent format and layout.
 * @param[out] out Contiguous BF16 output matrix `[N,T]`.
 * @param[in] stream CUDA stream on which execution is enqueued.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

// --- Tensor-parallel split forms (tp == 2) ----------------------------------------------------
//
// Both forms are compositions of the single-device linear() above, not separate kernels: a shard
// is a standalone `[N,K]` tensor of the same registered format with one axis narrowed -- exactly
// what the tp2 loader materializes into each device's arena -- so the existing kernel run against
// a shard weight already halves its grid along the split axis and already reads only that device's
// bytes. The split forms add the per-rank device/stream discipline, the cross-rank shape
// agreement a single device cannot check, and (row-parallel only) the summing collective.
//
// Every requirement of linear() applies per rank. `x[r]`, `w[r]`, `out[r]`, and `workspace[r]`
// must be resident on `ec.dev[r]`, and rank r's work is enqueued on `ec.dev[r]->stream`. The
// CALLER OBLIGATION of include/ninfer/ops/allreduce.h therefore applies here too: inputs staged
// with the plain cudaMemcpy/cudaMemset/`<<<...>>>` forms land on a device's LEGACY DEFAULT stream,
// which does not implicitly synchronize with `DeviceContext::stream`, and must be retired before
// the call. Neither form synchronizes; on return the work is enqueued on both streams. The
// caller's current CUDA device is preserved.
//
// The shard shapes must themselves be registered problems of the weight's format. The tp2 halves
// of every registered non-vocabulary geometry are registered; an unregistered shard shape is
// rejected exactly as an unregistered whole shape is.

/**
 * @brief Column-parallel (output-split) linear across two devices.
 *
 * Rank r computes `out[r] = w[r] * x[r]`, where `w[r]` is rank r's contiguous block of the logical
 * weight's OUTPUT rows and `x[r]` holds the same replicated activation on both ranks. The two
 * output blocks concatenate along `ne[0]` into the single-device result; nothing is communicated,
 * so this form is exactly two independent `linear()` calls plus validation.
 *
 * `w[0].k` must equal `w[1].k` (both ranks consume the whole input), and both ranks must agree on
 * the token count `x[r].ne[1]`. The per-rank output row counts `w[r].n` need not be equal: an
 * uneven split is legal, and this Op never needs to know the logical total.
 *
 * @param[in] x Per-rank replicated BF16 activation `[K,T]`.
 * @param[in] w Per-rank weight-row shard `[N_r,K]`.
 * @param[out] out Per-rank BF16 output block `[N_r,T]`.
 * @param[in] policy Permitted private activation-compute profiles, applied identically per rank.
 * @param[in,out] workspace Per-rank caller-owned transient arena, or null when the resolved route
 * needs none (see linear_workspace_capacity_bytes(), which is evaluated at the SHARD shape).
 * @param[in] ec Execution context holding exactly two distinct devices.
 */
void linear_column_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                            const std::array<Tensor, 2>& out, LinearPolicy policy,
                            const std::array<WorkspaceArena*, 2>& workspace,
                            const ExecutionContext& ec);

/// A16-only column-parallel form; requires no transient workspace.
void linear_column_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                            const std::array<Tensor, 2>& out, const ExecutionContext& ec);

/**
 * @brief Row-parallel (input-split) linear across two devices, all-reduced.
 *
 * Rank r computes the FULL-width partial `out[r] = w[r] * x[r]`, where `w[r]` is rank r's block of
 * the logical weight's INPUT columns and `x[r]` the matching block of the activation's rows. A
 * single `allreduce_sum` over `out` then leaves the complete `[N,T]` result on both ranks:
 *
 * @f[
 *   \mathrm{out}_{n,t} = \sum_{r} \sum_{k \in \mathrm{block}(r)}
 *     \mathrm{FP32Dequant}(w)_{n,k}\,\mathrm{FP32}(x_{k,t}).
 * @f]
 *
 * `w[0].n` must equal `w[1].n` (both ranks produce every output row) and both ranks must agree on
 * the token count. `w[r].k` need not be equal across ranks. `staging[r]` is scratch of the output's
 * dtype and shape resident on `ec.dev[r]`, must not overlap `out[r]`, and its contents after the
 * call are unspecified; `events` must be live. Consecutive calls sharing the same buffers, staging,
 * and events need no host synchronization between them.
 *
 * @par Numerical note
 * The split reduction is NOT bit-identical to the single-device result and cannot be: each rank
 * rounds its partial to BF16 storage before the collective adds them, so a split evaluation
 * carries two extra roundings the whole-K accumulation does not. The observable difference is
 * bounded by the partial magnitudes, not by the summed magnitude, which is why the split-parity
 * criterion is stated against the largest output rather than per element.
 *
 * @param[in] x Per-rank activation block `[K_r,T]`.
 * @param[in] w Per-rank weight-column shard `[N,K_r]`.
 * @param[in,out] out Per-rank BF16 `[N,T]`; holds rank r's partial until the collective completes,
 * then the identical summed result on both ranks.
 * @param[in,out] staging Per-rank scratch matching `out`, disjoint from it.
 * @param[in] policy Permitted private activation-compute profiles, applied identically per rank.
 * @param[in,out] workspace Per-rank caller-owned transient arena, or null when the resolved route
 * needs none. Sized at the SHARD shape.
 * @param[in] ec Execution context holding exactly two distinct devices.
 * @param[in] events Live cross-device ordering events, as for allreduce_sum().
 */
void linear_row_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                         const std::array<Tensor, 2>& out, const std::array<Tensor, 2>& staging,
                         LinearPolicy policy, const std::array<WorkspaceArena*, 2>& workspace,
                         const ExecutionContext& ec, const PeerEvents& events);

/// A16-only row-parallel form; requires no transient workspace.
void linear_row_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                         const std::array<Tensor, 2>& out, const std::array<Tensor, 2>& staging,
                         const ExecutionContext& ec, const PeerEvents& events);

} // namespace ninfer::ops
