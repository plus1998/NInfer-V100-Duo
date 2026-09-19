#pragma once

#include "core/tensor.h"
#include "ninfer/ops/allreduce.h" // ExecutionContext (tp2 split form)
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Computes four independent linear projections for each token:
 *
 *   q[:,t]    = linear(x[:,t], query_key_weight[0:6144,:])
 *   k[:,t]    = linear(x[:,t], query_key_weight[6144:7168,:])
 *   gate[:,t] = linear(x[:,t], gate_value_weight[0:6144,:])
 *   v[:,t]    = linear(x[:,t], gate_value_weight[6144:7168,:]).
 *
 * All tensors are contiguous BF16. Shapes are x [5120,T], q/gate [6144,T], and k/v [1024,T].
 * T may be any positive value.
 * The two parent weights are RowSplit [7168,5120] with FP16 scales and group size 64:
 * query_key is Q4G64_F16S and gate_value is Q5G64_F16S. The oracle exact-decodes each row and
 * evaluates every projection naively in FP64 from the represented inputs. The BF16 outputs are
 * promoted and compared directly with those ideal values; final output storage rounding belongs
 * to AttnInputProj's named A16 criterion, not the oracle. Production routes choose their private
 * accumulator and staging precision. Inputs and the four outputs must be mutually non-overlapping.
 * Caller-owned transient storage is sized by q4_q5_attn_input_proj_workspace_capacity_bytes().
 * The Op has no persistent state side effect.
 */
void attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     WorkspaceArena& workspace, cudaStream_t stream);

[[nodiscard]] std::size_t q4_q5_attn_input_proj_workspace_capacity_bytes(
    std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Computes the single-parent Q/K/output-gate/V projection.
 *
 * The parent stores rows in physical order query, key, output gate, value while the public output
 * argument order is q, gate, k, v. Every route writes the four independently contiguous final
 * allocations directly; no packed parent output is materialized. The NVFP4 A4 and FP8 A8
 * profiles may use caller-owned transient storage for their private quantized activation.
 *
 * Registered parent forms are:
 *
 * - W8G32_F16S RowSplit `[9216,2048]`, with row counts `[4096,512,4096,512]`. `x` is
 *   BF16 `[2048,T]`, q/gate are BF16 `[4096,T]`, and k/v are BF16 `[512,T]`.
 * - BF16_CTRL Contiguous `[14336,5120]`, with row counts `[6144,1024,6144,1024]`. `x` is
 *   BF16 `[5120,T]`, q/gate are BF16 `[6144,T]`, and k/v are BF16 `[1024,T]`.
 * - NVFP4 BlockScaleK16M128x4 `[14336,5120]`, with the same logical row and tensor shapes as
 *   BF16_CTRL.
 * - FP8_E4M3FN_ROW_BF16S RowScale `[14336,5120]`, with the same logical row and tensor shapes as
 *   BF16_CTRL.
 * - GGML_K `ggml-k256-v1` `[14336,5120]`, preserving Q4_K/Q6_K rows and using the direct native
 *   decoder. The logical row and tensor shapes are the same as BF16_CTRL.
 *
 * `T` is the positive token extent of the Op contract. BF16_CTRL and W8G32_F16S admit only
 * LinearPolicy::A16Only. NVFP4 admits A16Only and AllowA4; AllowA4 permits the private resolver to
 * select either a qualified A16 route or activation quantization to NVFP4 at every positive T.
 * FP8 admits A16Only and AllowA8 at every positive T. AllowA8 currently resolves T<=10 to the
 * qualified A16 CUDA-core route and every T>=11 to private activation quantization followed by the
 * A8 Tensor Core route. A16Only uses the A16 route for every positive T.
 *
 * The oracle evaluates every projection independently with naive FP64 accumulation from the
 * logical values represented by the persistent weight and BF16 activation. The final four BF16
 * stores belong to the Op's criterion for the selected activation-compute path.
 *
 * `workspace` is caller-owned call-scoped transient storage sized by
 * attn_input_proj_workspace_capacity_bytes(). It must not overlap the input, parent weight, or any
 * output. The Op does not allocate device memory internally.
 */
[[nodiscard]] std::size_t
attn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                         std::int32_t input_rows, LinearPolicy policy,
                                         std::int32_t min_tokens, std::int32_t max_tokens);

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                     WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Applies the A16-only single-parent Q/K/output-gate/V projection without transient workspace.
 */
void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);

/**
 * Qwen3.6 companion W8 specialization. The W8G32_F16S RowSplit parent has shape [6144,2048]
 * and stored row order [query 4096, key 1024, value 1024]. `x` is contiguous BF16 [2048,T],
 * q is contiguous BF16 [4096,T], and k/v are contiguous BF16 [1024,T]. Every route writes
 * the three independent final allocations directly; no parent output or transient workspace
 * is materialized. T may be any positive value. Q and K remain raw projection outputs: this
 * Op does not normalize or rotate either tensor.
 */
void attn_input_proj(const Tensor& x, const Weight& query_key_value_weight, Tensor& q, Tensor& k,
                     Tensor& v, cudaStream_t stream);

// --- Tensor-parallel split form (tp == 2) -------------------------------------------------------
//
// attn_input_proj is a COLUMN-parallel (output-split), head-aligned op only: it never allreduces.
// `Variant::attention_projection` (src/targets/qwen3_6_27b/impl/variant.cpp) calls this Op with
// EITHER the fused single-parent weight or the two-weight split-storage payload, and both forms are
// registered here.
//
// The ShardPlan (src/targets/qwen3_6_27b/impl/load/bindings.cpp, `shard_mapping` for
// "attention/query_key_gate_value") splits the fused parent [14336,5120] as FOUR independent column
// blocks, each split by tp on its own: Q [0,6144), K [6144,7168), Gate [7168,13312), V
// [13312,14336) -- the real physical row order (Q|K|Gate|V), NOT the "q|k|v|gate" order an earlier
// design draft stated; confirmed against `bindings.cpp`'s own comment there and against the
// pre-existing `Nvfp4AttentionInputSmallTOutput::store` epilogue mapping in
// src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_small_t.cu, which already encoded this order
// since the tp1 kernel was written. Because each block is split independently, rank r's shard
// weight is the CONCATENATION, in the SAME Q|K|Gate|V section order, of rank r's own head-local
// slice of every section: Q[r*3072,(r+1)*3072) | K[6144+r*512,...) | Gate[7168+r*3072,...) |
// V[13312+r*512,...) -- a standalone [7168,5120] tensor whose four sections sit at shard-local
// offsets 0 (query, 3072 rows), 3072 (key, 512 rows), 3584 (gate, 3072 rows), 6656 (value, 512
// rows). Each device's projection outputs are therefore ALREADY head-local (12 of 24 query/gate
// heads, 2 of 4 key/value heads) with no cross-device traffic -- exactly what the per-rank
// head-local attention core (src/ops/kernel/gqa_attention_geometry.cuh) needs.
//
// The split-storage two-weight form's ShardPlan (`attention/query_key`, `attention/gate_value`)
// splits each of its two [7168,5120] parents the same way -- Q|K and Gate|V respectively -- so
// rank r's query_key shard is the concatenation of its own Q and K column blocks: a standalone
// [3584,5120] tensor (query rows [0,3072), key rows [3072,3584)); its gate_value shard is
// [3584,5120] the same way (gate rows [0,3072), value rows [3072,3584)).
//
// The registered fused/split formats are the following, per the real ShardPlan/binding profiles (bindings.cpp
// `bind_nvfp4_text_layers`/`bind_groupwise_text_layers`/`bind_qwen38_nvfp4_text_layers`):
//   - NVFP4 (fused single-parent): a TRUE split. The decode/small-T/W4A4/TMA kernels
//     (src/ops/attn_input_proj/nvfp4/*.cu) are templated on Geometry and instantiated at both
//     Nvfp4AttnInputGeometry (tp1) and Nvfp4AttnInputTp2ColumnGeometry (the shard,
//     src/ops/linear/nvfp4/nvfp4_config.h) -- the same kernel template halves its own row indexing
//     at the shard's N: a shard is a standalone tensor of the same layout with one axis narrowed,
//     so the existing kernel instantiated at that shape IS the split kernel -- no `_split` kernel
//     exists or should be written. Route selection (resolve_route) is a pure function of (policy,
//     token count), inherited unchanged from the parent.
//   - FP8_E4M3FN_ROW_BF16S (fused single-parent -- the flagship's own bound format for
//     every full-attention layer's input projection): a TRUE split, the same shape of change as
//     NVFP4's above. The decode/small-T/A8 kernels (src/ops/attn_input_proj/fp8/*.cu) are templated
//     on <Geometry, Output> and instantiated at both Fp8AttnInputGeometry (tp1,
//     Fp8AttentionInputOutput) and Fp8AttnInputTp2ColumnGeometry (the shard,
//     src/ops/linear/fp8/fp8_config.h, Fp8AttentionInputShardOutput<Geometry> --
//     src/ops/attn_input_proj/fp8/fp8_attn_input_output.cuh). Route selection is inherited
//     unchanged from the tp1 resolve_route.
//   - Q4G64_F16S / Q5G64_F16S (groupwise split-storage, two-weight form): the grouped-MMA kernel
//     (src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_gemm_mma.cu) is already row-count-generic (it
//     reads its row split from the output tensors' own shapes, not a compile-time constant), so the
//     SAME kernel already serves the shard shape at every T; only the family's own small-T EXACT
//     kernels (q4_q5_attn_input_small_t.cu, compile-time-exact to the tp1 [7168,5120] parent) do
//     not, so the shard path always routes through the grouped-MMA kernel regardless of T -- a
//     documented performance-only gap, not a correctness one: correctness holds at every T>=1
//     (the kernel's FullTiles boolean masks the BN=64 token-tile boundary for any T, and the
//     shard's 3072/512 row counts are both multiples of its BM tile sizes), and the split test
//     sweeps T down to 1.
//   - GGML_K: the native Qwen3.8 Q4_K_M fused parent is row-sliced directly. Each rank computes
//     its four head-local sections with the shared GGML decoder and requires no transient arena.
//
// BF16_CTRL fused forms and the W8G32_F16S companion form are NOT registered here: BF16_CTRL is a
// real, dated, explicitly deferred gap (it is bound only by the Qwen36Nvfp4 profile, not by the
// flagship Qwen38Nvfp4 profile whose FP8 fused form is registered above); W8's companion form
// belongs only to the MTP head, which the qwen3_6_27b ShardPlan never shards.
//
// Every requirement of attn_input_proj() applies per rank (BF16 dtype, contiguity, 16-byte
// alignment, x/weight/output non-aliasing). The caller obligations of src/ops/common/split_launch.h
// apply unchanged: rank r's work is issued on ec.dev[r]->stream with that rank's device current.

/**
 * Returns the caller-owned transient capacity required by the fused-parent
 * attn_input_proj_column_parallel() for every T in [min_tokens,max_tokens], at the SHARD shape
 * (NVFP4, FP8, GGML_K, or Q4/Q5 split storage, depending on qtype). Quantized activation
 * workspace is a pure function of tokens and K=5120; GGML_K needs no transient storage, while
 * the groupwise split-storage form uses a caller-owned projected plane.
 */
[[nodiscard]] std::size_t attn_input_proj_column_parallel_workspace_capacity_bytes(
    QType qtype, LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * @brief Column-parallel (head-aligned, output-split) fused-parent attn_input_proj across two
 * devices. NVFP4, FP8, and GGML_K fused parents are supported.
 *
 * Rank r computes its own head-local q/gate/k/v blocks from its own [7168,5120] weight shard and
 * the (replicated) activation -- see the design note above for the shard's section layout. `w[0].k`
 * must equal `w[1].k` and both ranks must agree on the token count; the per-rank shard row counts
 * need not be equal.
 *
 * @param[in] x Per-rank replicated BF16 activation `[5120,T]`.
 * @param[in] w Per-rank fused query_key_gate_value weight shard `[7168,5120]`.
 * @param[out] q,gate Per-rank BF16 `[3072,T]`. @param[out] k,v Per-rank BF16 `[512,T]`.
 * @param[in] policy Permitted private activation-compute profiles, applied identically per rank.
 * @param[in,out] workspace Per-rank caller-owned transient arena; A16Only routes need none.
 * @param[in] ec Execution context holding exactly two distinct devices.
 */
void attn_input_proj_column_parallel(const std::array<Tensor, 2>& x,
                                     const std::array<Weight, 2>& query_key_gate_value_weight,
                                     const std::array<Tensor, 2>& q, const std::array<Tensor, 2>& gate,
                                     const std::array<Tensor, 2>& k, const std::array<Tensor, 2>& v,
                                     LinearPolicy policy,
                                     const std::array<WorkspaceArena*, 2>& workspace,
                                     const ExecutionContext& ec);

/**
 * A16-only column-parallel convenience form. Passes a null workspace per rank for routes that do
 * not need one (including GGML_K and NVFP4 A16).
 */
void attn_input_proj_column_parallel(const std::array<Tensor, 2>& x,
                                     const std::array<Weight, 2>& query_key_gate_value_weight,
                                     const std::array<Tensor, 2>& q, const std::array<Tensor, 2>& gate,
                                     const std::array<Tensor, 2>& k, const std::array<Tensor, 2>& v,
                                     const ExecutionContext& ec);

/**
 * @brief Column-parallel split-storage (two-weight, Q4G64_F16S/Q5G64_F16S) attn_input_proj across
 * two devices.
 *
 * Rank r computes its own head-local q/k blocks from its own `[3584,5120]` query_key shard (query
 * rows `[0,3072)`, key rows `[3072,3584)`) and its own head-local gate/v blocks from its own
 * `[3584,5120]` gate_value shard (gate rows `[0,3072)`, value rows `[3072,3584)`). This A16 form
 * uses Volta MMA with caller-owned transient storage on SM70 and grouped MMA on SM8x. Size each
 * arena with attn_input_proj_column_parallel_workspace_capacity_bytes(Q4G64_F16S, A16Only, ...).
 *
 * @param[in] x Per-rank replicated BF16 activation `[5120,T]`.
 * @param[in] query_key_weight,gate_value_weight Per-rank RowSplit weight shards `[3584,5120]`.
 * @param[out] q,gate Per-rank BF16 `[3072,T]`. @param[out] k,v Per-rank BF16 `[512,T]`.
 * @param[in] ec Execution context holding exactly two distinct devices.
 */
void attn_input_proj_column_parallel(const std::array<Tensor, 2>& x,
                                     const std::array<Weight, 2>& query_key_weight,
                                     const std::array<Weight, 2>& gate_value_weight,
                                     const std::array<Tensor, 2>& q, const std::array<Tensor, 2>& gate,
                                     const std::array<Tensor, 2>& k, const std::array<Tensor, 2>& v,
                                     const std::array<WorkspaceArena*, 2>& workspace,
                                     const ExecutionContext& ec);

} // namespace ninfer::ops
