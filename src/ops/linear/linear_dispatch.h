#pragma once

// ninfer::ops - internal (non-public) entry points into ops::linear's own per-format
// dispatch, exposed so a SIBLING family's split form (linear_add_row_parallel is the first
// caller) can issue the plain, residual-free GEMM half of a row-parallel rank without going
// back through the public ops::linear() wrapper, which requires a bound `WorkspaceArena&` and
// therefore cannot express "no workspace" the way this pointer-taking pair can (see
// nvfp4_dispatch.h for the same nullable-workspace convention this mirrors).
//
// These are validated exactly as ops::linear() validates them; the public wrapper is a thin
// pass-through over this same pair (src/ops/linear/linear.cpp).

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

void validate_linear_semantics(const Tensor& x, const Weight& w, const Tensor& out,
                               LinearPolicy policy);

void dispatch_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops
