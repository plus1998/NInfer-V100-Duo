#include "ninfer/ops/linear.h"

#include "ops/common/split_launch.h"
#include "ops/linear/linear_dispatch.h"
#include "ops/linear/ggml_k/ggml_k.h"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/w8/w8_dispatch.h"

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t checked_numel(const Tensor& tensor, const char* label) {
    std::int64_t total = 1;
    for (const std::int32_t extent : tensor.ne) {
        if (extent <= 0) {
            throw std::invalid_argument(std::string("linear: ") + label +
                                        " dimensions must be positive");
        }
        if (total > std::numeric_limits<std::int64_t>::max() / extent) {
            throw std::overflow_error("linear: tensor size overflows int64");
        }
        total *= extent;
    }
    return total;
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_linear_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear: invalid compute policy");
}

} // namespace

// Not anonymous-namespace-scoped: src/ops/linear/linear_dispatch.h exposes both of these to
// linear_add.cpp's row-parallel split form, which issues the residual-free half of a row-parallel
// rank through dispatch_linear directly (see that header for why the public linear() overloads,
// which require a bound WorkspaceArena&, cannot express a nullable workspace). Every other call
// site in this file is unqualified and unaffected -- unqualified lookup finds a plain namespace
// member exactly as it found an anonymous-namespace one.
void validate_linear_semantics(const Tensor& x, const Weight& w, const Tensor& out,
                               LinearPolicy policy) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear: x/out must be BF16");
    }
    (void)checked_numel(x, "x");
    (void)checked_numel(out, "out");
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("linear: x must have shape [K,T]");
    }
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("linear: out must have shape [N,T]");
    }
    if (w.n <= 0 || w.k <= 0) {
        throw std::invalid_argument("linear: weight n/k must be positive");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("linear: expected [K,T] x [N,K] -> [N,T]");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear: x/out must be non-null and 16-byte aligned");
    }
    validate_linear_policy(policy);
}

void dispatch_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream) {
    switch (w.qtype) {
    case QType::GGML_K:
        detail::ggml_k_linear(x, w, out, workspace, stream);
        return;
    case QType::Q4G64_F16S:
        detail::q4_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::Q5G64_F16S:
        detail::q5_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q6G64_F16S:
        detail::q6_dispatch(x, w, out, policy, stream);
        return;
    case QType::W8G32_F16S:
        detail::w8_dispatch(x, w, out, policy, stream);
        return;
    case QType::BF16_CTRL:
        detail::bf16_dispatch(x, w, out, policy, stream);
        return;
    case QType::NVFP4:
        detail::nvfp4_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP8_E4M3FN_ROW_BF16S:
        detail::fp8_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP32_CTRL:
    case QType::I32_CTRL:
        break;
    }
    throw std::invalid_argument("linear: unsupported weight qtype");
}

std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                            std::int32_t input_rows, LinearPolicy policy,
                                            std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_linear_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear workspace: invalid token interval");
    }

    switch (qtype) {
    case QType::GGML_K:
        if (output_rows <= 0 || input_rows <= 0 || input_rows % 256 != 0 ||
            policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("linear workspace: invalid GGML K profile");
        }
        return detail::ggml_k_cutlass_workspace_bytes(output_rows, input_rows, max_tokens);
    case QType::Q4G64_F16S:
        (void)detail::select_q4_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q4_launch(output_rows, input_rows, max_tokens, policy);
#ifdef NINFER_VOLTA_BUILD
        {
            std::size_t capacity = 0;
            for (int t = std::max(min_tokens, 9); t <= std::min(max_tokens, 64); ++t) {
                if (detail::q4_volta_mma_supported(output_rows, input_rows, t)) {
                    capacity = std::max(capacity,
                        detail::q4_volta_mma_workspace_bytes(output_rows, input_rows, t));
                }
            }
            return capacity;
        }
#endif
        return 0;
    case QType::Q5G64_F16S:
        (void)detail::select_q5_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q5_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q6G64_F16S:
        (void)detail::select_q6_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q6_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::W8G32_F16S:
        (void)detail::select_w8_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_w8_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::BF16_CTRL:
        (void)detail::select_bf16_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_bf16_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::NVFP4:
        if (!detail::is_nvfp4_linear_problem(output_rows, input_rows) ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
            throw std::invalid_argument("linear workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                             min_tokens, max_tokens);
    case QType::FP8_E4M3FN_ROW_BF16S:
        return detail::fp8_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                           min_tokens, max_tokens);
    case QType::FP32_CTRL:
    case QType::I32_CTRL:
        break;
    }
    throw std::invalid_argument("linear workspace: unsupported weight qtype");
}

void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, policy);
    dispatch_linear(x, w, out, policy, &workspace, stream);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, LinearPolicy::A16Only);
    dispatch_linear(x, w, out, LinearPolicy::A16Only, nullptr, stream);
}

namespace {

// Cross-rank agreement. Everything a single device can check is already checked per rank by
// validate_linear_semantics; these are the invariants only the pair makes sense of. The split
// axis's own extents are deliberately NOT required to match: an uneven split is legal and neither
// form ever needs to know the logical total.
void validate_split_pair(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                         const ExecutionContext& ec, bool column_parallel) {
    detail::require_split_context(
        ec, "linear split: requires an ExecutionContext with two distinct devices");
    if (x[0].ne[1] != x[1].ne[1]) {
        throw std::invalid_argument("linear split: both ranks must carry the same token count");
    }
    if (w[0].qtype != w[1].qtype || w[0].layout != w[1].layout) {
        throw std::invalid_argument("linear split: both ranks must carry the same weight format");
    }
    if (column_parallel) {
        if (w[0].k != w[1].k) {
            throw std::invalid_argument(
                "linear column-parallel: both ranks must consume the same input extent K");
        }
        return;
    }
    if (w[0].n != w[1].n) {
        throw std::invalid_argument(
            "linear row-parallel: both ranks must produce the same output extent N");
    }
}

// Everything a rank owns must live on that rank's device. Compiled out of Release; see
// require_rank_residency.
void validate_split_residency(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                              const std::array<Tensor, 2>& out, const ExecutionContext& ec) {
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        detail::require_rank_residency(
            ec, rank, x[slot].data, w[slot].payload, out[slot].data,
            "linear split: every per-rank argument must be resident on ec.dev[rank]");
    }
}

// Validates both ranks and returns the mutable output views the launchers need.
//
// `dispatch_linear` takes `Tensor&`, and the Op's public arguments are `const std::array<Tensor,2>&`
// (per-rank views the caller owns), so exactly one mutable copy per rank is made here and reused
// for both the validation and the launch. Tensor is a small non-owning view; copying it copies no
// device memory.
std::array<Tensor, 2> validated_outputs(const std::array<Tensor, 2>& x,
                                        const std::array<Weight, 2>& w,
                                        const std::array<Tensor, 2>& out, LinearPolicy policy) {
    std::array<Tensor, 2> destination{out[0], out[1]};
    for (std::size_t slot = 0; slot < 2; ++slot) {
        validate_linear_semantics(x[slot], w[slot], destination[slot], policy);
    }
    return destination;
}

// One rank's single-device projection, issued on that rank's own stream. This is the whole of the
// "split kernel": the shard Weight narrows N (column-parallel) or K (row-parallel), so the
// existing launcher resolves the shard geometry and its grid is already halved along that axis.
void issue_rank(int rank, const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                std::array<Tensor, 2>& out, LinearPolicy policy,
                const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec) {
    const auto slot = static_cast<std::size_t>(rank);
    dispatch_linear(x[slot], w[slot], out[slot], policy, workspace[slot], ec.dev[slot]->stream);
}

} // namespace

void linear_column_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                            const std::array<Tensor, 2>& out, LinearPolicy policy,
                            const std::array<WorkspaceArena*, 2>& workspace,
                            const ExecutionContext& ec) {
    validate_split_pair(x, w, ec, /*column_parallel=*/true);
    // Validate both ranks before issuing either, so a rejected pair enqueues nothing.
    std::array<Tensor, 2> destination = validated_outputs(x, w, out, policy);
    validate_split_residency(x, w, out, ec);
    detail::for_each_rank(
        ec, [&](int rank) { issue_rank(rank, x, w, destination, policy, workspace, ec); });
}

void linear_column_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                            const std::array<Tensor, 2>& out, const ExecutionContext& ec) {
    linear_column_parallel(x, w, out, LinearPolicy::A16Only, {nullptr, nullptr}, ec);
}

void linear_row_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                         const std::array<Tensor, 2>& out, const std::array<Tensor, 2>& staging,
                         LinearPolicy policy, const std::array<WorkspaceArena*, 2>& workspace,
                         const ExecutionContext& ec, const PeerEvents& events) {
    validate_split_pair(x, w, ec, /*column_parallel=*/false);
    std::array<Tensor, 2> destination = validated_outputs(x, w, out, policy);
    validate_split_residency(x, w, out, ec);
    if (!events.live()) { throw std::invalid_argument("linear row-parallel: events must be live"); }

    // Each rank's partial lands directly in out[rank]; allreduce_sum combines in place, so no
    // separate accumulation workspace exists to get out of step with the output. The collective
    // records its inputs_ready event on the same stream the projection above was issued on, which
    // is what orders the peer's read after the partial is complete.
    detail::for_each_rank(
        ec, [&](int rank) { issue_rank(rank, x, w, destination, policy, workspace, ec); });
    // staging[r] must be resident on ec.dev[r], match out[r]'s dtype and shape, and not overlap
    // it; allreduce_sum checks all three (residency and overlap in debug builds) rather than this
    // Op restating them.
    allreduce_sum(out, staging, ec, events);
}

void linear_row_parallel(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                         const std::array<Tensor, 2>& out, const std::array<Tensor, 2>& staging,
                         const ExecutionContext& ec, const PeerEvents& events) {
    linear_row_parallel(x, w, out, staging, LinearPolicy::A16Only, {nullptr, nullptr}, ec, events);
}

} // namespace ninfer::ops
