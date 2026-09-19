#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q4Q5AttnInputScheduleId {
    ParentSplitFixed,
    GroupedHomogeneousPairMmaR16C64S3,
    GroupedHomogeneousPairMmaR32C64S4,
    CutlassSm70TensorCore,
    VoltaMmaFused,
};

struct Q4Q5AttnInputProblem {
    std::int32_t input_rows;
    std::int32_t query_rows;
    std::int32_t kv_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q4Q5AttnInputPlan {
    Q4Q5AttnInputScheduleId schedule;
    std::size_t workspace_bytes;
};

const char* q4_q5_attn_input_schedule_name(Q4Q5AttnInputScheduleId schedule) noexcept;

bool q4_q5_attn_input_admits(const Q4Q5AttnInputProblem& problem) noexcept;
Q4Q5AttnInputPlan q4_q5_attn_input_resolve_plan(const Q4Q5AttnInputProblem& problem);

std::size_t q4_q5_attn_input_capacity_workspace_bytes(std::int32_t min_cols, std::int32_t max_cols);

void q4_q5_attn_input_execute_plan(const Q4Q5AttnInputPlan& plan, const Tensor& x,
                                   const Weight& query_key_weight, const Weight& gate_value_weight,
                                   Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                   WorkspaceArena& workspace, cudaStream_t stream);
void q4_q5_attn_input_dispatch(const Tensor& x, const Weight& query_key_weight,
                               const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                               Tensor& v, WorkspaceArena& workspace, cudaStream_t stream);

// --- TP2 column-shard sibling --------------------------------------------------------------------
// query_key/gate_value shard shape [3584,5120] (query/gate 3072 rows, key/value 512 rows -- half
// the heads of each section, same section order as the parent -- see
// include/ninfer/ops/attn_input_proj.h for the ShardPlan derivation). Unlike
// the tp1 dispatch above, this always routes through the grouped-MMA kernel
// (q4_q5_attn_input_grouped_mma_r32_c64_s4_launch), which is row-count-generic and therefore
// already correct at the shard shape for every T; the exact small-T kernels in
// q4_q5_attn_input_small_t.cu remain compile-time-exact to the tp1 parent shape and are not used by
// the shard (a documented performance-only gap, not a correctness one: the grouped-MMA kernel is
// correct at every T>=1 at the shard's row counts, and ninfer_attn_input_proj_split_test sweeps
// T down to 1).
bool q4_q5_attn_input_admits_shard(const Q4Q5AttnInputProblem& problem) noexcept;
std::size_t q4_q5_attn_input_shard_capacity_workspace_bytes(std::int32_t min_cols,
                                                           std::int32_t max_cols);
void q4_q5_attn_input_dispatch_shard(const Tensor& x, const Weight& query_key_weight,
                                     const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, WorkspaceArena& workspace,
                                     cudaStream_t stream);

} // namespace ninfer::ops::detail
