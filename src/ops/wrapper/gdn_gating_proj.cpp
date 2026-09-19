#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/rmsnorm.h"

#include "ops/common/split_launch.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"
#include "ops/launcher/gdn_gating.h"
#include "ops/linear/ggml_k/ggml_k.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_control_weight(const Weight& w, std::int32_t rows, std::int32_t input_rows,
                         const char* name) {
    if (w.qtype == QType::GGML_K) {
        if (w.layout != QuantLayout::GgmlK256 || w.n != rows || w.k != input_rows ||
            w.ndim != 2 || w.qdata == nullptr || !aligned_to(w.qhigh, 8)) {
            throw std::invalid_argument(std::string("gdn_gating_proj: invalid ") + name);
        }
        return;
    }
    const std::uint64_t payload_bytes = static_cast<std::uint64_t>(rows) *
                                        static_cast<std::uint64_t>(input_rows) *
                                        sizeof(std::uint16_t);
    if (w.qtype != QType::BF16_CTRL || w.layout != QuantLayout::Contiguous ||
        w.payload_bytes < payload_bytes || w.ndim != 2 || w.n != rows || w.k != input_rows ||
        w.shape[0] != rows || w.shape[1] != input_rows || w.padded_shape[0] != rows ||
        w.padded_shape[1] != input_rows || w.qhigh != nullptr || w.scales != nullptr ||
        w.high_plane_bytes != 0 || w.group != 0 || w.group_size != 0 || !aligned_to(w.qdata, 16)) {
        throw std::invalid_argument(std::string("gdn_gating_proj: invalid ") + name);
    }
}

Weight control_row_view(const Weight& parent, std::int32_t row_begin, std::int32_t rows) {
    if (parent.qtype == QType::GGML_K) {
        Weight view = parent;
        view.qhigh = static_cast<const std::uint64_t*>(parent.qhigh) + row_begin;
        view.high_plane_bytes = static_cast<std::uint64_t>(rows) * sizeof(std::uint64_t);
        view.shape[0] = rows;
        view.padded_shape[0] = rows;
        view.n = rows;
        return view;
    }
    const std::size_t row_bytes = static_cast<std::size_t>(parent.k) * sizeof(std::uint16_t);
    const auto* data            = static_cast<const std::uint8_t*>(parent.qdata) +
                       static_cast<std::size_t>(row_begin) * row_bytes;
    Weight view          = parent;
    view.payload         = data;
    view.payload_bytes   = static_cast<std::uint64_t>(rows) * row_bytes;
    view.qdata           = data;
    view.shape[0]        = rows;
    view.padded_shape[0] = rows;
    view.n               = rows;
    return view;
}

struct GdnControlParentGeometry {
    std::int32_t input_rows;
    std::int32_t heads;
};

GdnControlParentGeometry require_control_parent(const Weight& parent) {
    if (parent.n == 96 && parent.k == 5120) {
        require_control_weight(parent, 96, 5120, "ab_weight");
        return {.input_rows = 5120, .heads = 48};
    }
    if (parent.n == 64 && parent.k == 2048) {
        require_control_weight(parent, 64, 2048, "ab_weight");
        return {.input_rows = 2048, .heads = 32};
    }
    throw std::invalid_argument("gdn_gating_proj: unsupported ab_weight geometry");
}

void require_vector_tensor(const Tensor& t, DType dtype, std::int32_t n0, const char* op,
                           const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || !aligned_to(t.data, dtype == DType::FP32 ? 4 : 16)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    }
}

void require_sequence_tensor(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t tokens,
                             const char* op, const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != tokens || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || !aligned_to(t.data, dtype == DType::FP32 ? 4 : 16)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name);
    }
}

void project_ggml_k_control(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                            const Tensor& A_log, const Tensor& dt_bias, Tensor& g, Tensor& beta,
                            cudaStream_t stream) {
    // Each epilogue element reads its two projections before replacing them in place.
    detail::ggml_k_project_split(x, a_weight, &g, 1, false, stream);
    detail::ggml_k_project_split(x, b_weight, &beta, 1, false, stream);
    detail::gdn_gating_launch(g, beta, A_log, dt_bias, g, beta, stream);
}

} // namespace

std::size_t gdn_gating_proj_workspace_capacity_bytes(std::int32_t heads, std::int32_t input_rows,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    return detail::bf16_gdn_gating_capacity_workspace_bytes(heads, input_rows, min_tokens,
                                                            max_tokens);
}

std::size_t gdn_norm_gating_proj_workspace_capacity_bytes(std::int32_t heads,
                                                          std::int32_t input_rows,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens) {
    return detail::bf16_gdn_norm_gating_capacity_workspace_bytes(heads, input_rows, min_tokens,
                                                                 max_tokens);
}

void gdn_gating_proj(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                     const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g,
                     Tensor& beta, cudaStream_t stream) {
    constexpr const char* op  = "gdn_gating_proj";
    const std::int32_t tokens = x.ne[1];
    require_sequence_tensor(x, DType::BF16, 5120, tokens, op, "x");
    require_vector_tensor(A_log, DType::FP32, 48, op, "A_log");
    require_vector_tensor(dt_bias, DType::FP32, 48, op, "dt_bias");
    require_sequence_tensor(g, DType::FP32, 48, tokens, op, "g");
    require_sequence_tensor(beta, DType::FP32, 48, tokens, op, "beta");
    require_control_weight(a_weight, 48, 5120, "a_weight");
    require_control_weight(b_weight, 48, 5120, "b_weight");
    if (a_weight.qtype != b_weight.qtype) {
        throw std::invalid_argument("gdn_gating_proj: control formats disagree");
    }

    if (a_weight.qtype == QType::GGML_K && b_weight.qtype == QType::GGML_K) {
        project_ggml_k_control(x, a_weight, b_weight, A_log, dt_bias, g, beta, stream);
        return;
    }

    detail::bf16_gdn_gating_dispatch(x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, stream);
}

void gdn_gating_proj(const Tensor& x, const Weight& ab_weight, const Tensor& A_log,
                     const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g, Tensor& beta,
                     cudaStream_t stream) {
    constexpr const char* op                = "gdn_gating_proj";
    const std::int32_t tokens               = x.ne[1];
    const GdnControlParentGeometry geometry = require_control_parent(ab_weight);
    require_sequence_tensor(x, DType::BF16, geometry.input_rows, tokens, op, "x");
    require_vector_tensor(A_log, DType::FP32, geometry.heads, op, "A_log");
    require_vector_tensor(dt_bias, DType::FP32, geometry.heads, op, "dt_bias");
    require_sequence_tensor(g, DType::FP32, geometry.heads, tokens, op, "g");
    require_sequence_tensor(beta, DType::FP32, geometry.heads, tokens, op, "beta");

    if (ab_weight.qtype == QType::GGML_K) {
        const Tensor outputs[]{g, beta};
        detail::ggml_k_project_split(x, ab_weight, outputs, 2, false, stream);
        detail::gdn_gating_launch(g, beta, A_log, dt_bias, g, beta, stream);
        return;
    }
    const Weight a_weight = control_row_view(ab_weight, 0, geometry.heads);
    const Weight b_weight = control_row_view(ab_weight, geometry.heads, geometry.heads);
    detail::bf16_gdn_gating_dispatch(x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, stream);
}

void gdn_norm_gating_proj(const Tensor& x, const Tensor& norm_weight, float eps,
                          const Weight& a_weight, const Weight& b_weight, const Tensor& A_log,
                          const Tensor& dt_bias, WorkspaceArena& ws, Tensor& h, Tensor& g,
                          Tensor& beta, cudaStream_t stream) {
    constexpr const char* op  = "gdn_norm_gating_proj";
    const std::int32_t tokens = x.ne[1];
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument("gdn_norm_gating_proj: eps must be positive and finite");
    }
    require_sequence_tensor(x, DType::BF16, 5120, tokens, op, "x");
    require_vector_tensor(norm_weight, DType::BF16, 5120, op, "norm_weight");
    require_sequence_tensor(h, DType::BF16, 5120, tokens, op, "h");
    require_vector_tensor(A_log, DType::FP32, 48, op, "A_log");
    require_vector_tensor(dt_bias, DType::FP32, 48, op, "dt_bias");
    require_sequence_tensor(g, DType::FP32, 48, tokens, op, "g");
    require_sequence_tensor(beta, DType::FP32, 48, tokens, op, "beta");
    require_control_weight(a_weight, 48, 5120, "a_weight");
    require_control_weight(b_weight, 48, 5120, "b_weight");
    if (a_weight.qtype != b_weight.qtype) {
        throw std::invalid_argument("gdn_norm_gating_proj: control formats disagree");
    }

    if (a_weight.qtype == QType::GGML_K && b_weight.qtype == QType::GGML_K) {
        rmsnorm(x, norm_weight, eps, true, h, stream);
        project_ggml_k_control(h, a_weight, b_weight, A_log, dt_bias, g, beta, stream);
        return;
    }

    detail::bf16_gdn_norm_gating_dispatch(x, norm_weight, eps, h, a_weight, b_weight, A_log,
                                          dt_bias, ws, g, beta, stream);
}

void gdn_norm_gating_proj(const Tensor& x, const Tensor& norm_weight, float eps,
                          const Weight& ab_weight, const Tensor& A_log, const Tensor& dt_bias,
                          WorkspaceArena& ws, Tensor& h, Tensor& g, Tensor& beta,
                          cudaStream_t stream) {
    constexpr const char* op  = "gdn_norm_gating_proj";
    const std::int32_t tokens = x.ne[1];
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument("gdn_norm_gating_proj: eps must be positive and finite");
    }
    const GdnControlParentGeometry geometry = require_control_parent(ab_weight);
    require_sequence_tensor(x, DType::BF16, geometry.input_rows, tokens, op, "x");
    require_vector_tensor(norm_weight, DType::BF16, geometry.input_rows, op, "norm_weight");
    require_sequence_tensor(h, DType::BF16, geometry.input_rows, tokens, op, "h");
    require_vector_tensor(A_log, DType::FP32, geometry.heads, op, "A_log");
    require_vector_tensor(dt_bias, DType::FP32, geometry.heads, op, "dt_bias");
    require_sequence_tensor(g, DType::FP32, geometry.heads, tokens, op, "g");
    require_sequence_tensor(beta, DType::FP32, geometry.heads, tokens, op, "beta");

    if (ab_weight.qtype == QType::GGML_K) {
        rmsnorm(x, norm_weight, eps, true, h, stream);
        const Tensor outputs[]{g, beta};
        detail::ggml_k_project_split(h, ab_weight, outputs, 2, false, stream);
        detail::gdn_gating_launch(g, beta, A_log, dt_bias, g, beta, stream);
        return;
    }
    const Weight a_weight = control_row_view(ab_weight, 0, geometry.heads);
    const Weight b_weight = control_row_view(ab_weight, geometry.heads, geometry.heads);
    detail::bf16_gdn_norm_gating_dispatch(x, norm_weight, eps, h, a_weight, b_weight, A_log,
                                          dt_bias, ws, g, beta, stream);
}

// --- Tensor-parallel split form (tp == 2) -------------------------------------------------------
// See include/ninfer/ops/gdn_gating_proj.h for the full design note (the verified interior row
// order, the output contract the GDN core reads, and which formats are registered).
namespace {

constexpr std::int32_t kShardHidden = 5120;
constexpr std::int32_t kShardHeads  = 24;

void validate_column_rank_semantics(const Tensor& x, const Weight& a_weight,
                                    const Weight& b_weight, const Tensor& A_log,
                                    const Tensor& dt_bias, const Tensor& g, const Tensor& beta) {
    constexpr const char* op   = "gdn_gating_proj column-parallel";
    const std::int32_t tokens  = x.ne[1];
    if (tokens <= 0) { throw std::invalid_argument(std::string(op) + ": T must be positive"); }
    require_sequence_tensor(x, DType::BF16, kShardHidden, tokens, op, "x");
    require_vector_tensor(A_log, DType::FP32, kShardHeads, op, "A_log");
    require_vector_tensor(dt_bias, DType::FP32, kShardHeads, op, "dt_bias");
    require_sequence_tensor(g, DType::FP32, kShardHeads, tokens, op, "g");
    require_sequence_tensor(beta, DType::FP32, kShardHeads, tokens, op, "beta");
    require_control_weight(a_weight, kShardHeads, kShardHidden, "a_weight shard");
    require_control_weight(b_weight, kShardHeads, kShardHidden, "b_weight shard");
    if (a_weight.qtype != b_weight.qtype) {
        throw std::invalid_argument("gdn_gating_proj column-parallel: control formats disagree");
    }
}

// Cross-rank agreement only a pair can check; every per-rank invariant is validated separately by
// validate_column_rank_semantics. Mirrors attn_input_proj's own validate_fused_split_pair
// (src/ops/wrapper/attn_input_proj.cpp).
void validate_split_pair(const std::array<Tensor, 2>& x, const ExecutionContext& ec) {
    detail::require_split_context(
        ec, "gdn_gating_proj column-parallel: requires an ExecutionContext with two distinct "
            "devices");
    if (x[0].ne[1] != x[1].ne[1]) {
        throw std::invalid_argument(
            "gdn_gating_proj column-parallel: both ranks must carry the same token count");
    }
}

void dispatch_shard_with_workspace(const Tensor& x, const Weight& a_weight,
                                   const Weight& b_weight, const Tensor& A_log,
                                   const Tensor& dt_bias, WorkspaceArena& ws, const Tensor& g,
                                   const Tensor& beta, std::size_t required_bytes,
                                   cudaStream_t stream) {
    auto scope             = ws.scope();
    const DeviceSpan scratch = ws.alloc_bytes(required_bytes);
    Tensor g_mut(g);
    Tensor beta_mut(beta);
    detail::bf16_gdn_gating_dispatch_shard(x, a_weight, b_weight, A_log, dt_bias, scratch.data,
                                           scratch.bytes, g_mut, beta_mut, stream);
}

void dispatch_shard(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                    const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena* ws,
                    const Tensor& g, const Tensor& beta, cudaStream_t stream) {
    if (a_weight.qtype == QType::GGML_K && b_weight.qtype == QType::GGML_K) {
        Tensor g_mut(g);
        Tensor beta_mut(beta);
        project_ggml_k_control(x, a_weight, b_weight, A_log, dt_bias, g_mut, beta_mut, stream);
        return;
    }
    const std::int32_t tokens        = x.ne[1];
    const std::size_t required_bytes = detail::bf16_gdn_gating_shard_workspace_bytes(tokens);
    if (required_bytes == 0) {
        Tensor g_mut(g);
        Tensor beta_mut(beta);
        detail::bf16_gdn_gating_dispatch_shard(x, a_weight, b_weight, A_log, dt_bias, nullptr, 0,
                                               g_mut, beta_mut, stream);
        return;
    }
    if (ws == nullptr) {
        throw std::invalid_argument(
            "gdn_gating_proj column-parallel: requires caller workspace at T>=2");
    }
    dispatch_shard_with_workspace(x, a_weight, b_weight, A_log, dt_bias, *ws, g, beta,
                                  required_bytes, stream);
}

} // namespace

std::size_t gdn_gating_proj_column_parallel_workspace_capacity_bytes(std::int32_t min_tokens,
                                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument(
            "gdn_gating_proj column-parallel workspace: invalid token interval");
    }
    // small-T-split10 has no upper T bound (see kernels.cu's kShardN comment); the required bytes
    // grow monotonically with T, so the maximum over the interval is at max_tokens.
    return detail::bf16_gdn_gating_shard_workspace_bytes(max_tokens);
}

void gdn_gating_proj_column_parallel(const std::array<Tensor, 2>& x,
                                     const std::array<Weight, 2>& a_weight,
                                     const std::array<Weight, 2>& b_weight,
                                     const std::array<Tensor, 2>& A_log,
                                     const std::array<Tensor, 2>& dt_bias,
                                     const std::array<WorkspaceArena*, 2>& ws,
                                     const std::array<Tensor, 2>& g, const std::array<Tensor, 2>& beta,
                                     const ExecutionContext& ec) {
    validate_split_pair(x, ec);
    for (std::size_t slot = 0; slot < 2; ++slot) {
        validate_column_rank_semantics(x[slot], a_weight[slot], b_weight[slot], A_log[slot],
                                       dt_bias[slot], g[slot], beta[slot]);
    }
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        detail::require_rank_residency(
            ec, rank, x[slot].data, a_weight[slot].payload, g[slot].data,
            "gdn_gating_proj column-parallel: every per-rank argument must be resident on "
            "ec.dev[rank]");
    }
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        dispatch_shard(x[slot], a_weight[slot], b_weight[slot], A_log[slot], dt_bias[slot],
                       ws[slot], g[slot], beta[slot], ec.dev[slot]->stream);
    });
}

void gdn_gating_proj_column_parallel(const std::array<Tensor, 2>& x,
                                     const std::array<Weight, 2>& ab_weight,
                                     const std::array<Tensor, 2>& A_log,
                                     const std::array<Tensor, 2>& dt_bias,
                                     const std::array<WorkspaceArena*, 2>& ws,
                                     const std::array<Tensor, 2>& g, const std::array<Tensor, 2>& beta,
                                     const ExecutionContext& ec) {
    validate_split_pair(x, ec);
    std::array<Weight, 2> a_weight{};
    std::array<Weight, 2> b_weight{};
    for (std::size_t slot = 0; slot < 2; ++slot) {
        const Weight& parent = ab_weight[slot];
        if (parent.n != 2 * kShardHeads || parent.k != kShardHidden) {
            throw std::invalid_argument(
                "gdn_gating_proj column-parallel: unsupported ab_weight shard geometry");
        }
        require_control_weight(parent, 2 * kShardHeads, kShardHidden, "ab_weight shard");
        a_weight[slot] = control_row_view(parent, 0, kShardHeads);
        b_weight[slot] = control_row_view(parent, kShardHeads, kShardHeads);
        validate_column_rank_semantics(x[slot], a_weight[slot], b_weight[slot], A_log[slot],
                                       dt_bias[slot], g[slot], beta[slot]);
    }
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        detail::require_rank_residency(
            ec, rank, x[slot].data, ab_weight[slot].payload, g[slot].data,
            "gdn_gating_proj column-parallel: every per-rank argument must be resident on "
            "ec.dev[rank]");
    }
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        if (ab_weight[slot].qtype == QType::GGML_K) {
            Tensor g_mut(g[slot]);
            Tensor beta_mut(beta[slot]);
            const Tensor outputs[]{g_mut, beta_mut};
            detail::ggml_k_project_split(x[slot], ab_weight[slot], outputs, 2, false,
                                        ec.dev[slot]->stream);
            detail::gdn_gating_launch(g_mut, beta_mut, A_log[slot], dt_bias[slot], g_mut, beta_mut,
                                     ec.dev[slot]->stream);
            return;
        }
        dispatch_shard(x[slot], a_weight[slot], b_weight[slot], A_log[slot], dt_bias[slot],
                       ws[slot], g[slot], beta[slot], ec.dev[slot]->stream);
    });
}

} // namespace ninfer::ops
