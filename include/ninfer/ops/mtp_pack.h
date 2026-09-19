#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: mtp_pack_fc_input
 *
 * Math / indexing:
 *   out[0:D, t] = embedding_norm[:, t]
 *   out[D:2D, t] = hidden_norm[:, t]
 *
 * Logical shapes:
 *   BF16 embedding_norm and hidden_norm [D,T], out [2D,T], contiguous. The registered domains are
 *   D=5120 for Qwen3.6-27B and D=2048 for Qwen3.6-35B-A3B.
 *
 * Numeric:
 *   Exact BF16 element copies; no arithmetic or conversion.
 *
 * Effects:
 *   Writes the full output. Inputs and output must not alias.
 *
 * Workspace:
 *   None. The Op has no persistent state side effect.
 *
 * TENSOR PARALLELISM (tp == 2). This Op needs no split form, and the reason is worth stating
 * because it is not "it is replicated". `mtp/input_projection` is bound [5120, 10240] and the
 * ShardPlan splits it ROW-PARALLEL over its 10240-wide input axis
 * (src/targets/qwen3_6_27b/impl/load/bindings.cpp, `append_row_parallel(..., mtp_input_rows, ...)`),
 * so device r's shard [5120,5120] contracts with exactly the packed rows [5120r, 5120r+5120) --
 * which are `embedding_norm` on rank 0 and `hidden_norm` on rank 1, by this Op's own definition
 * above. A tp2 caller therefore SKIPS the pack entirely and hands
 * `ops::linear_row_parallel({e, h}, shard, ...)` the two unpacked halves directly; the one
 * all-reduce inside that Op leaves the complete [5120,T] result on both devices. Packing first
 * would build a [10240,T] buffer of which each rank reads only its own half.
 */
void mtp_pack_fc_input(const Tensor& embedding_norm, const Tensor& hidden_norm, Tensor& out,
                       cudaStream_t stream);

/**
 * Op: mtp_split_attn_in
 *
 * Math / indexing:
 *   For each token, rows [0,6144), [6144,7168), [7168,13312), and [13312,14336) are copied to
 *   flattened Q[6144], K[1024], Gate[6144], and V[1024], respectively.
 *
 * Logical shapes:
 *   attn_in [14336,T]; q/gate [256,24,T]; k/v [256,4,T], all contiguous BF16.
 *
 *   Two row geometries are registered. The tp == 2 shard of the same object is attn_in [7168,T]
 *   with q/gate [256,12,T] and k/v [256,2,T], selected by attn_in's own row count -- no flag and no
 *   rank argument, exactly as every other split form in ops/ selects a shard. Its section
 *   boundaries are the ShardPlan's: rows [0,3072) Q | [3072,3584) K | [3584,6656) Gate |
 *   [6656,7168) V, i.e. rank r's own half of each of the parent's four sections in the parent's own
 *   order -- the identical layout include/ninfer/ops/attn_input_proj.h derives for the text
 *   layers' copy of this object. The resulting head indices are DEVICE-LOCAL: q/gate head h is
 *   global head 12r + h and k/v head h is global head 2r + h. Nothing downstream renumbers them --
 *   ops::gqa_attention's registered 12|2 geometry is head-local by construction
 *   (src/ops/kernel/gqa_attention_geometry.cuh).
 *
 * Numeric:
 *   Exact BF16 element copies with only an index remap.
 *
 * Effects:
 *   Writes every output element. Outputs and input must be pairwise non-aliasing.
 *
 * Workspace:
 *   None. The Op has no persistent state side effect.
 */
void mtp_split_attn_in(const Tensor& attn_in, Tensor& q, Tensor& k, Tensor& gate, Tensor& v,
                       cudaStream_t stream);

} // namespace ninfer::ops
