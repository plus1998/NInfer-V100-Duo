#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Row counts of the Query and Key sections within a fused Q|K|Gate|V parent/shard, keyed by output
// Geometry. The real physical row order is Q|K|Gate|V, NOT the "q|k|v|gate" an earlier design
// draft stated: confirmed against bindings.cpp's `shard_mapping` for
// "attention/query_key_gate_value" and the pre-existing `Nvfp4AttentionInputSmallTOutput::store`
// mapping this file already carried. Gate rows always equal Query rows and Value rows always
// equal Key rows (the fused object stacks one head-count-matched pair per section), so these two
// values are enough to derive all four section offsets:
//   [0, kQueryRows) query | [kQueryRows, +kKeyRows) key |
//   [.., +kQueryRows) gate | [.., +kKeyRows) value.
template <class Geometry>
struct Nvfp4AttnInputSections;

template <>
struct Nvfp4AttnInputSections<Nvfp4AttnInputGeometry> {
    static constexpr std::int32_t kQueryRows = 6144;
    static constexpr std::int32_t kKeyRows   = 1024;
};

// The tp2 column shard: each device owns half the heads of every section (12 query/gate heads, 2
// key/value heads out of 24/4 total), in the SAME section order as the parent -- see
// include/ninfer/ops/attn_input_proj.h for the ShardPlan derivation.
template <>
struct Nvfp4AttnInputSections<Nvfp4AttnInputTp2ColumnGeometry> {
    static constexpr std::int32_t kQueryRows = 3072;
    static constexpr std::int32_t kKeyRows   = 512;
};

[[nodiscard]] std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);
[[nodiscard]] std::size_t nvfp4_attn_input_shard_workspace_capacity_bytes(
    LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens);

void nvfp4_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, Nvfp4W4a4Workspace workspace,
                                  cudaStream_t stream);

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream);

// --- TP2 column-shard siblings (Nvfp4AttnInputTp2ColumnGeometry, [7168,5120]) ------------------
// Same kernel templates as the tp1 functions above, instantiated at the shard Geometry. Route
// selection (resolve_route) is a pure function of (policy, token count) and is inherited unchanged
// from the tp1 parent: tuning is inherited, never re-measured for a shard.

void nvfp4_attn_input_decode_launch_shard(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_small_t_launch_shard(const Tensor& x, const Weight& weight, Tensor& q,
                                           Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_w4a4_launch_shard(const Tensor& x, const Weight& weight, Tensor& q,
                                        Tensor& gate, Tensor& k, Tensor& v,
                                        Nvfp4W4a4Workspace workspace, cudaStream_t stream);

void nvfp4_attn_input_dispatch_shard(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, LinearPolicy policy,
                                     WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
