#include "ninfer/ops/linear_add.h"

#include "ninfer/ops/residual_add.h"
#include "ops/common/split_launch.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/linear_dispatch.h" // detail-free validate_linear_semantics / dispatch_linear
#include "ops/linear/ggml_k/ggml_k.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear_add/bf16/bf16_linear_add_plan.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_add/q5/q5_linear_add_plan.h"
#include "ops/linear_add/w8/w8_linear_add_plan.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_tensor(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t columns,
                    const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != columns || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("linear_add: invalid ") + name);
    }
}

void require_q5(const Weight& w) {
    if (w.qtype != QType::Q5G64_F16S || w.layout != QuantLayout::RowSplit ||
        w.scale_dtype != DType::FP16 || w.group_size != 64 || w.group != 64 ||
        w.padded_shape[0] != w.n || w.padded_shape[1] != w.k || w.qdata == nullptr ||
        w.qhigh == nullptr || w.scales == nullptr) {
        throw std::invalid_argument("linear_add: weight must be Q5G64_F16S row-split");
    }
}

void require_w8(const Weight& w) {
    if (w.qtype != QType::W8G32_F16S || w.layout != QuantLayout::RowSplit ||
        w.scale_dtype != DType::FP16 || w.group_size != 32 || w.group != 32 ||
        w.padded_shape[0] != w.n || w.padded_shape[1] != w.k || w.qdata == nullptr ||
        w.qhigh != nullptr || w.scales == nullptr) {
        throw std::invalid_argument("linear_add: weight must be W8G32_F16S row-split");
    }
}

void require_bf16(const Weight& w) {
    if (w.qtype != QType::BF16_CTRL || w.layout != QuantLayout::Contiguous || w.qdata == nullptr) {
        throw std::invalid_argument("linear_add: weight must be contiguous BF16_CTRL");
    }
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear_add: invalid compute policy");
}

// The per-format validate+dispatch body linear_add() uses, factored out so
// ops::linear_add_row_parallel's fused rank (NVFP4, Q5G64_F16S) can reach it directly with a
// nullable `WorkspaceArena*` -- exactly the reason src/ops/linear/linear_dispatch.h exposes
// ops::dispatch_linear the same way for the row-parallel form's residual-free rank. `ws` is
// dereferenced only by the branches that can actually need transient storage (NVFP4 W4A4, FP8 A8);
// every other branch never reads it.
void dispatch_linear_add(const Tensor& x, const Weight& w, Tensor& residual_out,
                         LinearPolicy policy, WorkspaceArena* ws, cudaStream_t stream) {
    const std::int32_t t = x.ne[1];
    if (t <= 0) { throw std::invalid_argument("linear_add: T must be positive"); }
    require_tensor(x, DType::BF16, w.k, t, "x");
    require_tensor(residual_out, DType::BF16, w.n, t, "residual_out");
    if (overlaps(x, residual_out)) {
        throw std::invalid_argument("linear_add: x and residual_out must not overlap");
    }

    if (w.qtype == QType::GGML_K) {
        detail::ggml_k_project_split(x, w, &residual_out, 1, true, stream, false, ws);
        return;
    }
    if (w.qtype == QType::BF16_CTRL) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("BF16 linear_add admits only A16");
        }
        require_bf16(w);
        if (!detail::bf16_linear_add_admits(w.n, w.k, x.ne[1])) {
            throw std::invalid_argument("linear_add: unsupported BF16 shape");
        }
        if (!aligned_to(x.data, 16) || !aligned_to(residual_out.data, 16) ||
            !aligned_to(w.qdata, 16)) {
            throw std::invalid_argument(
                "linear_add: BF16 requires 16-byte x/residual/weight alignment");
        }
        detail::bf16_linear_add_dispatch(x, w, residual_out, stream);
        return;
    }

    if (w.qtype == QType::Q5G64_F16S) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("Q5 linear_add admits only A16");
        }
        require_q5(w);
        // TP2: also admit the row-parallel shard extents (6144 -> 3072, 17408 -> 8704); see the
        // NVFP4 branch below for the shape-gate rationale.
        const bool supported_shape = (w.n == 5120 && w.k == 17408) || (w.n == 5120 && w.k == 6144) ||
                                     (w.n == 5120 && w.k == 3072) || (w.n == 5120 && w.k == 8704);
        if (!supported_shape) { throw std::invalid_argument("linear_add: unsupported Q5 shape"); }
        if (!aligned_to(x.data, 16) || !aligned_to(residual_out.data, 16) ||
            !aligned_to(w.qdata, 16) || !aligned_to(w.qhigh, 16) || !aligned_to(w.scales, 16)) {
            throw std::invalid_argument(
                "linear_add: Q5 requires 16-byte x/residual/code/high/scale alignment");
        }
        detail::q5_linear_add_dispatch(x, w, residual_out, ws, stream);
        return;
    }

    if (w.qtype == QType::W8G32_F16S) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("W8 linear_add admits only A16");
        }
        require_w8(w);
        if (w.n != 2048 || (w.k != 4096 && w.k != 6144)) {
            throw std::invalid_argument("linear_add: unsupported W8 shape");
        }
        if (!aligned_to(x.data, 16) || !aligned_to(residual_out.data, 16) ||
            !aligned_to(w.qdata, 16) || !aligned_to(w.scales, 16)) {
            throw std::invalid_argument(
                "linear_add: W8 requires 16-byte x/residual/code/scale alignment");
        }
        detail::w8_linear_add_dispatch(x, w, residual_out, stream);
        return;
    }

    if (w.qtype == QType::NVFP4) {
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
            throw std::invalid_argument("NVFP4 linear_add admits only A16 or A4");
        }
        detail::validate_nvfp4_weight(w, "nvfp4 linear_add");
        // TP2: a shard is a standalone [5120,K/2] tensor of the same registered format, never a
        // pointer offset into the parent payload (validate_nvfp4_weight would reject that outright,
        // and the tp2 loader materializes a standalone tensor per device), so the existing
        // W4A4/A16 kernel templates instantiated at Nvfp4Residual*Tp2RowGeometry already halve
        // their K loop and read only that device's bytes. Widening this gate by shape alone is what
        // lets ops::linear_add_row_parallel call this exact public function for its fused rank
        // instead of needing its own parallel entry point.
        const bool supported_shape = (w.n == detail::Nvfp4Residual6144Geometry::kOutputRows &&
                                      w.k == detail::Nvfp4Residual6144Geometry::kInputRows) ||
                                     (w.n == detail::Nvfp4Residual17408Geometry::kOutputRows &&
                                      w.k == detail::Nvfp4Residual17408Geometry::kInputRows) ||
                                     (w.n == detail::Nvfp4Residual6144Tp2RowGeometry::kOutputRows &&
                                      w.k == detail::Nvfp4Residual6144Tp2RowGeometry::kInputRows) ||
                                     (w.n == detail::Nvfp4Residual17408Tp2RowGeometry::kOutputRows &&
                                      w.k == detail::Nvfp4Residual17408Tp2RowGeometry::kInputRows);
        if (!supported_shape) {
            throw std::invalid_argument("nvfp4 linear_add: unsupported weight shape");
        }
        if (!aligned_to(x.data, 16) || !aligned_to(residual_out.data, 16)) {
            throw std::invalid_argument("linear_add: NVFP4 requires 16-byte x/residual alignment");
        }
        detail::nvfp4_linear_add_dispatch(x, w, residual_out, policy, ws, stream);
        return;
    }

    if (w.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
            throw std::invalid_argument("FP8 linear_add admits only A16 or A8");
        }
        (void)detail::validate_fp8_weight(w, "fp8 linear_add");
        // TP2: also admit the row-parallel shard extents (6144 -> 3072, 17408 -> 8704); see the
        // NVFP4 branch above for the shape-gate rationale.
        const bool supported_shape = (w.n == detail::Fp8Residual6144Geometry::kOutputRows &&
                                      w.k == detail::Fp8Residual6144Geometry::kInputRows) ||
                                     (w.n == detail::Fp8Residual17408Geometry::kOutputRows &&
                                      w.k == detail::Fp8Residual17408Geometry::kInputRows) ||
                                     (w.n == detail::Fp8Residual6144Tp2RowGeometry::kOutputRows &&
                                      w.k == detail::Fp8Residual6144Tp2RowGeometry::kInputRows) ||
                                     (w.n == detail::Fp8Residual17408Tp2RowGeometry::kOutputRows &&
                                      w.k == detail::Fp8Residual17408Tp2RowGeometry::kInputRows);
        if (!supported_shape) {
            throw std::invalid_argument("fp8 linear_add: unsupported weight shape");
        }
        if (!aligned_to(x.data, 16) || !aligned_to(residual_out.data, 16)) {
            throw std::invalid_argument("linear_add: FP8 requires 16-byte x/residual alignment");
        }
        if (ws == nullptr) {
            throw std::invalid_argument("fp8 linear_add: requires caller workspace");
        }
        detail::fp8_linear_add_dispatch(x, w, residual_out, policy, *ws, stream);
        return;
    }

    throw std::invalid_argument("linear_add: unsupported weight format");
}

} // namespace

std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                std::int32_t input_rows, std::int32_t min_tokens,
                                                std::int32_t max_tokens) {
    return linear_add_workspace_capacity_bytes(qtype, output_rows, input_rows,
                                               LinearPolicy::A16Only, min_tokens, max_tokens);
}

std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                std::int32_t input_rows, LinearPolicy policy,
                                                std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear_add workspace: invalid token interval");
    }
    if (qtype == QType::GGML_K) {
        (void)linear_workspace_capacity_bytes(qtype, output_rows, input_rows, policy,
                                              min_tokens, max_tokens);
        return detail::ggml_k_cutlass_workspace_bytes(output_rows, input_rows, max_tokens);
    }
    if (qtype == QType::BF16_CTRL) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("linear_add workspace: BF16 admits only A16");
        }
        (void)detail::bf16_linear_add_select(output_rows, input_rows, min_tokens);
        (void)detail::bf16_linear_add_select(output_rows, input_rows, max_tokens);
        return 0;
    }
    if (qtype == QType::W8G32_F16S) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("linear_add workspace: W8 admits only A16");
        }
        (void)detail::w8_linear_add_resolve_plan({output_rows, input_rows, input_rows, min_tokens});
        (void)detail::w8_linear_add_resolve_plan({output_rows, input_rows, input_rows, max_tokens});
        return 0;
    }
    if (qtype == QType::Q5G64_F16S) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("linear_add workspace: Q5 admits only A16");
        }
        return detail::q5_linear_add_capacity_workspace_bytes(output_rows, input_rows, input_rows,
                                                              min_tokens, max_tokens);
    }
    if (qtype == QType::NVFP4) {
        // TP2: also admit the row-parallel halves of the two residual geometries (o_proj /
        // gdn/output 6144 -> 3072, mlp/down 17408 -> 8704). Same shape rule as every other split
        // family -- a shard is a standalone tensor of the registered geometry with K halved, so
        // the gate widens by shape alone; ops::linear_add_row_parallel calls this exact public
        // function per rank, it does not have its own parallel entry point.
        const bool supported =
            (output_rows == detail::Nvfp4Residual6144Geometry::kOutputRows &&
             input_rows == detail::Nvfp4Residual6144Geometry::kInputRows) ||
            (output_rows == detail::Nvfp4Residual17408Geometry::kOutputRows &&
             input_rows == detail::Nvfp4Residual17408Geometry::kInputRows) ||
            (output_rows == detail::Nvfp4Residual6144Tp2RowGeometry::kOutputRows &&
             input_rows == detail::Nvfp4Residual6144Tp2RowGeometry::kInputRows) ||
            (output_rows == detail::Nvfp4Residual17408Tp2RowGeometry::kOutputRows &&
             input_rows == detail::Nvfp4Residual17408Tp2RowGeometry::kInputRows);
        if (!supported || (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
            throw std::invalid_argument("linear_add workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_linear_add_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                                 min_tokens, max_tokens);
    }
    if (qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        // TP2: also admit the row-parallel halves of the two residual geometries (o_proj /
        // gdn/output 6144 -> 3072, mlp/down 17408 -> 8704); see the NVFP4 branch above for the
        // shape-gate rationale.
        const bool supported =
            (output_rows == detail::Fp8Residual6144Geometry::kOutputRows &&
             input_rows == detail::Fp8Residual6144Geometry::kInputRows) ||
            (output_rows == detail::Fp8Residual17408Geometry::kOutputRows &&
             input_rows == detail::Fp8Residual17408Geometry::kInputRows) ||
            (output_rows == detail::Fp8Residual6144Tp2RowGeometry::kOutputRows &&
             input_rows == detail::Fp8Residual6144Tp2RowGeometry::kInputRows) ||
            (output_rows == detail::Fp8Residual17408Tp2RowGeometry::kOutputRows &&
             input_rows == detail::Fp8Residual17408Tp2RowGeometry::kInputRows);
        if (!supported || (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8)) {
            throw std::invalid_argument("linear_add workspace: unsupported FP8 profile");
        }
        return detail::fp8_linear_add_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                               min_tokens, max_tokens);
    }
    throw std::invalid_argument("linear_add workspace: unsupported weight format");
}

void linear_add(const Tensor& x, const Weight& w, Tensor& residual_out, WorkspaceArena& ws,
                cudaStream_t stream) {
    linear_add(x, w, residual_out, LinearPolicy::A16Only, ws, stream);
}

void linear_add(const Tensor& x, const Weight& w, Tensor& residual_out, LinearPolicy policy,
                WorkspaceArena& ws, cudaStream_t stream) {
    validate_policy(policy);
    dispatch_linear_add(x, w, residual_out, policy, &ws, stream);
}

// --- Tensor-parallel split form (tp == 2) -------------------------------------------------------
//
// See include/ninfer/ops/linear_add.h for the full design note. In one line: rank 0 evaluates
// `residual = residual + partial_0` with a fused kernel, rank 1 evaluates the pure GEMM partial
// `residual = partial_1`, and the one allreduce_sum that follows adds the residual exactly once.
namespace {

// Cross-rank agreement only a two-rank call can check; everything a single device can check is
// already checked by dispatch_linear_add / validate_linear_semantics per rank. The split axis's
// own per-rank K extents are deliberately NOT required to match, matching linear_row_parallel.
void validate_add_split_pair(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                             const ExecutionContext& ec) {
    detail::require_split_context(
        ec, "linear_add split: requires an ExecutionContext with two distinct devices");
    if (x[0].ne[1] != x[1].ne[1]) {
        throw std::invalid_argument("linear_add split: both ranks must carry the same token count");
    }
    if (w[0].qtype != w[1].qtype || w[0].layout != w[1].layout) {
        throw std::invalid_argument("linear_add split: both ranks must carry the same weight format");
    }
    if (w[0].n != w[1].n) {
        throw std::invalid_argument(
            "linear_add split: both ranks must produce the same output extent N");
    }
}

void validate_add_split_residency(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                                  const std::array<Tensor, 2>& residual,
                                  const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        detail::require_rank_residency(
            ec, rank, x[slot].data, w[slot].payload, residual[slot].data,
            "linear_add split: every per-rank argument must be resident on ec.dev[rank]");
    }
}

// Rank 0: residual += partial, folding the (replicated) residual into the reduction exactly once.
// NVFP4 and Q5G64_F16S reach this through their own fused linear_add kernels at the shard geometry
// (dispatch_linear_add, widened above to admit the tp2 row-parallel halves). BF16_CTRL's linear_add
// family is entirely exact-geometry kernels with no runtime-K escape hatch for a halved extent (see
// bf16_linear_add_plan.cpp), so instantiating a whole new exact geometry just for the shard would
// be real new kernel work, not the registry work every other split form needs -- instead it
// composes the already tp2-capable plain linear() with the standalone residual_add() Op, which is
// the same qualified `x += y` computation allreduce_sum's own local combine already uses, so the
// arithmetic is identical either way. `scratch` is staging[0]: unused by allreduce_sum until after
// this call, so reusing it as the BF16 decomposition's temporary GEMM output needs no extra buffer.
void issue_fused_rank(const Tensor& x, const Weight& w, Tensor& residual, Tensor& scratch,
                      LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    if (w.qtype == QType::BF16_CTRL) {
        validate_linear_semantics(x, w, scratch, policy);
        dispatch_linear(x, w, scratch, policy, workspace, stream);
        residual_add(scratch, residual, stream);
        return;
    }
    if (w.qtype == QType::GGML_K || w.qtype == QType::NVFP4 || w.qtype == QType::Q5G64_F16S ||
        w.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        // FP8 reaches here through the same "shape alone selects the route" widening as NVFP4/Q5:
        // dispatch_linear_add's FP8 branch (above) already admits the tp2 row-shard extents, and
        // its own fp8_linear_add_dispatch resolves the halved-K geometry via resolve_fp8_problem.
        dispatch_linear_add(x, w, residual, policy, workspace, stream);
        return;
    }
    throw std::invalid_argument("linear_add split: unsupported weight format (NVFP4, Q5G64_F16S, "
                                "FP8_E4M3FN_ROW_BF16S, or BF16_CTRL only)");
}

// Rank 1: residual = partial, the pure residual-free GEMM half, overwriting this rank's own copy
// of the (replicated) residual -- its pre-call bytes are not needed again, since rank 0 already
// carries the one copy that enters the sum. Every format issue_fused_rank above accepts is already
// registered in ops::linear's own tp2 registry at this shard shape, so this is the same public
// entry point linear_row_parallel() itself uses.
void issue_plain_rank(const Tensor& x, const Weight& w, Tensor& residual, LinearPolicy policy,
                      WorkspaceArena* workspace, cudaStream_t stream) {
    validate_linear_semantics(x, w, residual, policy);
    dispatch_linear(x, w, residual, policy, workspace, stream);
}

} // namespace

void linear_add_row_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                             const std::array<Tensor, 2>& residual,
                             const std::array<Tensor, 2>& staging, LinearPolicy policy,
                             const std::array<WorkspaceArena*, 2>& workspace,
                             const ExecutionContext& ec, const PeerEvents& events) {
    validate_policy(policy);
    validate_add_split_pair(x, w, ec);
    validate_add_split_residency(x, w, residual, ec);

    std::array<Tensor, 2> target{residual[0], residual[1]};
    std::array<Tensor, 2> scratch{staging[0], staging[1]};
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot        = static_cast<std::size_t>(rank);
        const cudaStream_t s   = ec.dev[slot]->stream;
        if (rank == 0) {
            issue_fused_rank(x[slot], w[slot], target[slot], scratch[slot], policy,
                             workspace[slot], s);
        } else {
            issue_plain_rank(x[slot], w[slot], target[slot], policy, workspace[slot], s);
        }
    });
    // allreduce_sum checks staging's residency, shape and non-overlap against `residual` itself; not
    // restated here. Its local combine is the same `x += y` computation issue_fused_rank's BF16
    // branch already used, so the two extra roundings a split evaluation always carries (linear.h's
    // row-parallel numerical note) are the only source of divergence from the tp1 fused kernel.
    allreduce_sum(target, staging, ec, events);
}

void linear_add_row_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                             const std::array<Tensor, 2>& residual,
                             const std::array<Tensor, 2>& staging, const ExecutionContext& ec,
                             const PeerEvents& events) {
    linear_add_row_parallel(x, w, residual, staging, LinearPolicy::A16Only, {nullptr, nullptr}, ec,
                            events);
}

void ggml_k_gdn_output(const Tensor& x, const Weight& w, Tensor& residual,
                       WorkspaceArena& workspace, cudaStream_t stream) {
    detail::ggml_k_project_split(x, w, &residual, 1, true, stream, true, &workspace);
}

void ggml_k_gdn_output(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                       const std::array<Tensor, 2>& residual,
                       const std::array<Tensor, 2>& staging,
                       const std::array<WorkspaceArena*, 2>& workspace,
                       const ExecutionContext& ec,
                       const PeerEvents& events) {
    validate_add_split_pair(x, w, ec);
    validate_add_split_residency(x, w, residual, ec);
    detail::for_each_rank(ec, [&](int rank) {
        detail::ggml_k_project_split(x[rank], w[rank], &residual[rank], 1, rank == 0,
                                     ec.dev[rank]->stream, true, workspace[rank]);
    });
    allreduce_sum(residual, staging, ec, events);
}

} // namespace ninfer::ops
