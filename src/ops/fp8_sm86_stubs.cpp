// sm_86 replacements for the FP8 A8 compute path.
#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void reject_fp8_a8() {
    throw std::runtime_error("FP8 A8 execution requires an sm_100a or sm_120a GPU");
}

} // namespace

// --- Linear / projection A8 routes -------------------------------------------------------------

void launch_fp8_a8_quantize(const Tensor&, const Weight&, Fp8A8Workspace, cudaStream_t) {
    reject_fp8_a8();
}

void launch_fp8_a8(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace, cudaStream_t) {
    reject_fp8_a8();
}

void fp8_attn_input_a8_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                              Fp8A8Workspace, cudaStream_t) {
    reject_fp8_a8();
}

void fp8_attn_input_a8_launch_shard(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                                    Fp8A8Workspace, cudaStream_t) {
    reject_fp8_a8();
}

void fp8_gdn_input_a8_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Fp8A8Workspace,
                             cudaStream_t) {
    reject_fp8_a8();
}

void fp8_gdn_input_a8_launch_shard(const Tensor&, const Weight&, Tensor&, Tensor&, Fp8A8Workspace,
                                   cudaStream_t) {
    reject_fp8_a8();
}

void fp8_linear_add_a8_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                              cudaStream_t) {
    reject_fp8_a8();
}

void fp8_linear_swiglu_a8_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                 cudaStream_t) {
    reject_fp8_a8();
}

void fp8_linear_swiglu_a8_launch_shard(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                       cudaStream_t) {
    reject_fp8_a8();
}

} // namespace ninfer::ops::detail
