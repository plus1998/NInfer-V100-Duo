#include "ninfer/ops/gdn_input_proj.h"

#include "core/layout.h"
#include "ops/common/split_launch.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"
#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"
#include "ops/gdn_input_proj/w8/w8_gdn_input_plan.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/ggml_k/ggml_k.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

void require_conv_tensor(const Tensor& tensor, std::int32_t rows, std::int32_t width,
                         std::int32_t batch, const char* op, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != width ||
        tensor.ne[2] != batch || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + label);
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

void require_single_parent_nonoverlap(const Tensor& x, const Tensor& qkv, const Tensor& z) {
    if (overlaps(x, qkv) || overlaps(x, z) || overlaps(qkv, z)) {
        throw std::invalid_argument("gdn_input_proj: x, qkv, and z must not overlap");
    }
}

struct ConvGeometry {
    std::int32_t width;
    std::int32_t batch;
    std::int32_t aggregate_columns;
};

ConvGeometry require_snapshot_input(const Tensor& x, std::int32_t hidden) {
    constexpr std::int32_t kMaximumBatch = 8;
    constexpr std::int32_t kMaximumWidth = 16;
    const std::int32_t width             = x.ne[1];
    const std::int32_t batch             = x.ne[2];
    if (width <= 0 || batch <= 0 || batch > kMaximumBatch || (batch > 1 && width > kMaximumWidth)) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot: unsupported B/W domain");
    }
    require_conv_tensor(x, hidden, width, batch, "gdn_input_proj_conv_snapshot", "x");
    return {width, batch, width * batch};
}

ConvGeometry require_record_input(const Tensor& x, std::int32_t hidden) {
    constexpr std::int32_t kMaximumBatch = 8;
    constexpr std::int32_t kMinimumWidth = 2;
    constexpr std::int32_t kMaximumWidth = 16;
    const std::int32_t width             = x.ne[1];
    const std::int32_t batch             = x.ne[2];
    if (width < kMinimumWidth || width > kMaximumWidth || batch <= 0 || batch > kMaximumBatch) {
        throw std::invalid_argument("gdn_input_proj_conv_record: unsupported B/T domain");
    }
    require_conv_tensor(x, hidden, width, batch, "gdn_input_proj_conv_record", "x");
    return {width, batch, width * batch};
}

void require_snapshot_operands(const Tensor& conv_weight, const Tensor& conv_states,
                               const Tensor& valid_columns, const Tensor& initial_state_slots,
                               const Tensor& snapshot_base_slots, std::int32_t channels,
                               ConvGeometry geometry) {
    require_matrix(conv_weight, channels, 4, "conv weight");
    if (conv_states.dtype != DType::BF16 || conv_states.ne[0] != channels ||
        conv_states.ne[1] != 3 || conv_states.ne[2] < geometry.aggregate_columns ||
        conv_states.ne[3] != 1 || !conv_states.is_contiguous() ||
        !aligned_to(conv_states.data, 16)) {
        throw std::invalid_argument(
            "gdn_input_proj_conv_snapshot: invalid convolution snapshot state");
    }
    const auto valid_selector = [batch = geometry.batch](const Tensor& selector) {
        return selector.dtype == DType::I32 && selector.ne[0] == batch && selector.ne[1] == 1 &&
               selector.ne[2] == 1 && selector.ne[3] == 1 && selector.is_contiguous() &&
               selector.data != nullptr;
    };
    if (!valid_selector(initial_state_slots) || !valid_selector(snapshot_base_slots)) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot: invalid state selector");
    }
    if (valid_columns.data != nullptr) {
        if (!valid_selector(valid_columns)) {
            throw std::invalid_argument("gdn_input_proj_conv_snapshot: invalid valid columns");
        }
    }
}

Tensor flatten_columns(const Tensor& tensor, std::int32_t rows, ConvGeometry geometry) {
    return Tensor(tensor.data, tensor.dtype, {rows, geometry.aggregate_columns});
}

void require_record_operands(const Tensor& conv_weight, const Tensor& conv_states,
                             const Tensor& valid_columns, const Tensor& initial_state_slots,
                             std::int32_t channels, ConvGeometry geometry) {
    require_matrix(conv_weight, channels, 4, "conv weight");
    if (conv_states.dtype != DType::BF16 || conv_states.ne[0] != channels ||
        conv_states.ne[1] != 3 || conv_states.ne[2] <= 0 || conv_states.ne[3] != 1 ||
        !conv_states.is_contiguous() || !aligned_to(conv_states.data, 16)) {
        throw std::invalid_argument("gdn_input_proj_conv_record: invalid convolution state");
    }
    const auto valid_selector = [batch = geometry.batch](const Tensor& selector) {
        return selector.dtype == DType::I32 && selector.ne[0] == batch && selector.ne[1] == 1 &&
               selector.ne[2] == 1 && selector.ne[3] == 1 && selector.is_contiguous() &&
               selector.data != nullptr;
    };
    if (!valid_selector(initial_state_slots)) {
        throw std::invalid_argument("gdn_input_proj_conv_record: invalid initial state selector");
    }
    if (valid_columns.data != nullptr && !valid_selector(valid_columns)) {
        throw std::invalid_argument("gdn_input_proj_conv_record: invalid valid columns");
    }
}

bool overlaps_range(const Tensor& tensor, const void* base, std::size_t bytes) {
    if (tensor.data == nullptr || base == nullptr || bytes == 0) { return false; }
    const auto tensor_begin = reinterpret_cast<std::uintptr_t>(tensor.data);
    const auto range_begin  = reinterpret_cast<std::uintptr_t>(base);
    return tensor_begin < range_begin + bytes && range_begin < tensor_begin + tensor.bytes();
}

void require_record_nonoverlap(const Tensor& x, const Tensor& conv_weight,
                               const Tensor& conv_states, const Tensor& valid_columns,
                               const Tensor& initial_state_slots, const Tensor& conv_record,
                               const Tensor& query, const Tensor& key, const Tensor& value,
                               const Tensor& z, const WorkspaceArena& workspace) {
    const std::array<const Tensor*, 10> tensors{
        &x,           &conv_weight, &conv_states, &valid_columns, &initial_state_slots,
        &conv_record, &query,       &key,         &value,         &z};
    for (std::size_t lhs = 0; lhs < tensors.size(); ++lhs) {
        if (tensors[lhs]->data == nullptr) { continue; }
        for (std::size_t rhs = lhs + 1; rhs < tensors.size(); ++rhs) {
            if (tensors[rhs]->data != nullptr && overlaps(*tensors[lhs], *tensors[rhs])) {
                throw std::invalid_argument(
                    "gdn_input_proj_conv_record: tensor operands must not overlap");
            }
        }
        if (overlaps_range(*tensors[lhs], workspace.base(), workspace.capacity())) {
            throw std::invalid_argument(
                "gdn_input_proj_conv_record: tensor operand overlaps live workspace");
        }
    }
}

void require_snapshot_nonoverlap(const Tensor& x, const Tensor& conv_weight,
                                 const Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_state_slots,
                                 const Tensor& snapshot_base_slots, const Tensor& query,
                                 const Tensor& key, const Tensor& value, const Tensor& z,
                                 const WorkspaceArena& workspace) {
    const std::array<const Tensor*, 10> tensors{&x,
                                                &conv_weight,
                                                &conv_states,
                                                &valid_columns,
                                                &initial_state_slots,
                                                &snapshot_base_slots,
                                                &query,
                                                &key,
                                                &value,
                                                &z};
    for (std::size_t lhs = 0; lhs < tensors.size(); ++lhs) {
        if (tensors[lhs]->data == nullptr) { continue; }
        for (std::size_t rhs = lhs + 1; rhs < tensors.size(); ++rhs) {
            const bool shared_state_selectors =
                tensors[lhs] == &initial_state_slots && tensors[rhs] == &snapshot_base_slots;
            if (!shared_state_selectors && tensors[rhs]->data != nullptr &&
                overlaps(*tensors[lhs], *tensors[rhs])) {
                throw std::invalid_argument(
                    "gdn_input_proj_conv_snapshot: tensor operands must not overlap");
            }
        }
        if (overlaps_range(*tensors[lhs], workspace.base(), workspace.capacity())) {
            throw std::invalid_argument(
                "gdn_input_proj_conv_snapshot: tensor operand overlaps live workspace");
        }
    }
}

template <std::size_t Count>
void require_parent_nonoverlap(const Weight& weight,
                               const std::array<const Tensor*, Count>& tensors,
                               const WorkspaceArena& workspace, const char* operation) {
    for (const Tensor* tensor : tensors) {
        if (tensor->data != nullptr &&
            overlaps_range(*tensor, weight.payload,
                           static_cast<std::size_t>(weight.payload_bytes))) {
            throw std::invalid_argument(std::string(operation) +
                                        ": tensor operand overlaps parent weight");
        }
    }
    if (weight.payload != nullptr && workspace.base() != nullptr && workspace.capacity() != 0) {
        const auto weight_begin    = reinterpret_cast<std::uintptr_t>(weight.payload);
        const auto workspace_begin = reinterpret_cast<std::uintptr_t>(workspace.base());
        if (weight_begin < workspace_begin + workspace.capacity() &&
            workspace_begin < weight_begin + weight.payload_bytes) {
            throw std::invalid_argument(std::string(operation) +
                                        ": parent weight overlaps live workspace");
        }
    }
}

void require_snapshot_capacity_domain(std::int32_t batch_size, std::int32_t min_width,
                                      std::int32_t max_width) {
    constexpr std::int32_t kMaximumBatch = 8;
    constexpr std::int32_t kMaximumWidth = 16;
    if (batch_size <= 0 || batch_size > kMaximumBatch || min_width <= 0 || max_width < min_width ||
        (batch_size > 1 && max_width > kMaximumWidth)) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot workspace: invalid B/W domain");
    }
}

void require_record_capacity_domain(std::int32_t batch_size, std::int32_t min_width,
                                    std::int32_t max_width) {
    constexpr std::int32_t kMaximumBatch = 8;
    constexpr std::int32_t kMinimumWidth = 2;
    constexpr std::int32_t kMaximumWidth = 16;
    if (batch_size <= 0 || batch_size > kMaximumBatch || min_width < kMinimumWidth ||
        max_width < min_width || max_width > kMaximumWidth) {
        throw std::invalid_argument("gdn_input_proj_conv_record workspace: invalid B/T domain");
    }
}

void require_rowsplit(const Weight& weight, QType qtype, std::int32_t rows, const char* label) {
    const bool q4_planes =
        qtype != QType::Q4G64_F16S || (weight.qhigh == nullptr && weight.high_plane_bytes == 0);
    const bool q5_planes =
        qtype != QType::Q5G64_F16S || (weight.qhigh != nullptr && weight.high_plane_bytes != 0);
    if (weight.qtype != qtype || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 64 || weight.group != 64 ||
        weight.ndim != 2 || weight.n != rows || weight.k != 5120 || weight.shape[0] != rows ||
        weight.shape[1] != 5120 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 5120 || !q4_planes || !q5_planes ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 4) ||
        (qtype == QType::Q5G64_F16S && !aligned_to(weight.qhigh, 16))) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

void require_w8_rowsplit(const Weight& weight, std::int32_t rows, const char* label) {
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != rows || weight.k != 2048 || weight.shape[0] != rows ||
        weight.shape[1] != 2048 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 2048 || weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("gdn_input_proj: invalid compute policy");
}

void require_ggml_k_parent(const Weight& weight, std::int32_t rows, LinearPolicy policy) {
    if (policy != LinearPolicy::A16Only || weight.layout != QuantLayout::GgmlK256 ||
        weight.n != rows || weight.k != 5120 || weight.ndim != 2 ||
        weight.qdata == nullptr || !aligned_to(weight.qhigh, 8)) {
        throw std::invalid_argument("gdn_input_proj: invalid GGML_K parent or policy");
    }
}

void project_ggml_k(const Tensor& x, const Weight& weight, const Tensor& qkv,
                    const Tensor& z, cudaStream_t stream, WorkspaceArena* workspace = nullptr) {
    const Tensor outputs[]{qkv, z};
    detail::ggml_k_project_split(x, weight, outputs, 2, false, stream, false, workspace);
}

void dispatch_single_parent(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                            LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    validate_policy(policy);
    const std::int32_t cols = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("gdn_input_proj: T must be positive"); }

    if (weight.qtype == QType::GGML_K) {
        require_ggml_k_parent(weight, 16384, policy);
        require_matrix(x, 5120, cols, "x");
        require_matrix(qkv, 10240, cols, "qkv");
        require_matrix(z, 6144, cols, "z");
        require_single_parent_nonoverlap(x, qkv, z);
        project_ggml_k(x, weight, qkv, z, stream, workspace);
        return;
    }

    if (weight.qtype == QType::NVFP4) {
        constexpr std::int32_t kHidden  = 5120;
        constexpr std::int32_t kQkvRows = 10240;
        constexpr std::int32_t kZRows   = 6144;
        constexpr std::int32_t kRows    = kQkvRows + kZRows;
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
            throw std::invalid_argument("NVFP4 gdn_input_proj admits only A16 or A4");
        }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(qkv, kQkvRows, cols, "qkv");
        require_matrix(z, kZRows, cols, "z");
        require_single_parent_nonoverlap(x, qkv, z);
        detail::validate_nvfp4_weight(weight, "nvfp4 gdn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("nvfp4 gdn_input_proj: unsupported weight shape");
        }
        detail::nvfp4_gdn_input_dispatch(x, weight, qkv, z, policy, workspace, stream);
        return;
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        constexpr std::int32_t kHidden  = 5120;
        constexpr std::int32_t kQkvRows = 10240;
        constexpr std::int32_t kZRows   = 6144;
        constexpr std::int32_t kRows    = kQkvRows + kZRows;
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
            throw std::invalid_argument("FP8 gdn_input_proj admits only A16 or A8");
        }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(qkv, kQkvRows, cols, "qkv");
        require_matrix(z, kZRows, cols, "z");
        require_single_parent_nonoverlap(x, qkv, z);
        detail::validate_fp8_weight(weight, "fp8 gdn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("fp8 gdn_input_proj: unsupported weight shape");
        }
        detail::fp8_gdn_input_dispatch(x, weight, qkv, z, policy, workspace, stream);
        return;
    }

    constexpr std::int32_t kHidden  = 2048;
    constexpr std::int32_t kQkvRows = 8192;
    constexpr std::int32_t kZRows   = 4096;
    constexpr std::int32_t kRows    = kQkvRows + kZRows;
    if (policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("W8 gdn_input_proj admits only A16");
    }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(qkv, kQkvRows, cols, "qkv");
    require_matrix(z, kZRows, cols, "z");
    require_single_parent_nonoverlap(x, qkv, z);
    require_w8_rowsplit(weight, kRows, "query/key/value/z weight");
    detail::w8_gdn_input_dispatch(x, weight, qkv, z, workspace, stream);
}

detail::Q4Q5GdnInputConvPlan resolve_q4_q5_conv_plan(std::int32_t tokens, std::int32_t batch_size) {
    return detail::q4_q5_gdn_input_conv_resolve_plan({5120, 4096, 12288, 10240, 6144, 5120, tokens},
                                                     batch_size);
}

detail::W8GdnInputConvPlan resolve_w8_conv_plan(std::int32_t tokens, std::int32_t batch_size) {
    return detail::w8_gdn_input_conv_resolve_plan({2048, 8192, 4096, 12288, 2048, tokens},
                                                  batch_size);
}

struct ProjectedWorkspace {
    Tensor projected;
};

template <class Allocator>
ProjectedWorkspace allocate_projected_workspace(Allocator& allocator, std::int32_t channels,
                                                std::int32_t tokens) {
    ProjectedWorkspace out;
    out.projected = allocator.alloc(DType::BF16, {channels, tokens});
    return out;
}

std::size_t composed_snapshot_capacity(std::int32_t channels, std::int32_t aggregate_columns,
                                       std::size_t projection_workspace_bytes) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected_workspace(layout, channels, aggregate_columns);
    if (projection_workspace_bytes != 0) { (void)layout.alloc_bytes(projection_workspace_bytes); }
    return layout.peak_bytes(1);
}

template <class Project>
void compose_batched_snapshot(const Tensor& x, const Tensor& conv_weight, Tensor& conv_states,
                              const Tensor& valid_columns, const Tensor& initial_state_slots,
                              const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                              Tensor& value, Tensor& z, std::int32_t query_rows,
                              std::int32_t key_rows, std::int32_t value_rows, ConvGeometry geometry,
                              WorkspaceArena& workspace, cudaStream_t stream, Project&& project) {
    const std::int32_t channels = query_rows + key_rows + value_rows;
    auto scope                  = workspace.scope();
    ProjectedWorkspace scratch =
        allocate_projected_workspace(workspace, channels, geometry.aggregate_columns);

    Tensor x_flat = flatten_columns(x, x.ne[0], geometry);
    Tensor z_flat = flatten_columns(z, z.ne[0], geometry);
    project(x_flat, scratch.projected, z_flat);

    Tensor projected(scratch.projected.data, DType::BF16,
                     {channels, geometry.width, geometry.batch});
    detail::gdn_projected_conv_snapshot_launch(projected, conv_weight, conv_states, valid_columns,
                                               initial_state_slots, snapshot_base_slots, query, key,
                                               value, stream);
}

template <class Project>
void compose_record(const Tensor& x, const Tensor& conv_weight, const Tensor& conv_states,
                    const Tensor& valid_columns, const Tensor& initial_state_slots,
                    Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                    ConvGeometry geometry, WorkspaceArena& workspace, cudaStream_t stream,
                    Project&& project) {
    auto scope         = workspace.scope();
    Tensor x_flat      = flatten_columns(x, x.ne[0], geometry);
    Tensor record_flat = flatten_columns(conv_record, conv_record.ne[0], geometry);
    Tensor z_flat      = flatten_columns(z, z.ne[0], geometry);
    project(x_flat, record_flat, z_flat);
    detail::gdn_projected_conv_record_launch(conv_record, conv_weight, conv_states, valid_columns,
                                             initial_state_slots, query, key, value, stream);
}

void dispatch_single_parent_snapshot(const Tensor& x, const Weight& weight,
                                     const Tensor& conv_weight, Tensor& conv_states,
                                     const Tensor& valid_columns, const Tensor& initial_state_slots,
                                     const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                     Tensor& value, Tensor& z, LinearPolicy policy,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    validate_policy(policy);

    if (weight.qtype == QType::GGML_K) {
        constexpr const char* op = "gdn_input_proj_conv_snapshot";
        require_ggml_k_parent(weight, 16384, policy);
        const ConvGeometry geometry = require_snapshot_input(x, 5120);
        require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                  snapshot_base_slots, 10240, geometry);
        require_conv_tensor(query, 2048, geometry.width, geometry.batch, op, "query");
        require_conv_tensor(key, 2048, geometry.width, geometry.batch, op, "key");
        require_conv_tensor(value, 6144, geometry.width, geometry.batch, op, "value");
        require_conv_tensor(z, 6144, geometry.width, geometry.batch, op, "z");
        require_snapshot_nonoverlap(x, conv_weight, conv_states, valid_columns,
                                   initial_state_slots, snapshot_base_slots, query, key, value, z,
                                   workspace);
        compose_batched_snapshot(
            x, conv_weight, conv_states, valid_columns, initial_state_slots, snapshot_base_slots,
            query, key, value, z, 2048, 2048, 6144, geometry, workspace, stream,
            [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                project_ggml_k(x_flat, weight, projected, z_flat, stream, &workspace);
            });
        return;
    }

    if (weight.qtype == QType::NVFP4) {
        constexpr std::int32_t kHidden     = 5120;
        constexpr std::int32_t kQueryRows  = 2048;
        constexpr std::int32_t kKeyRows    = 2048;
        constexpr std::int32_t kValueRows  = 6144;
        constexpr std::int32_t kZRows      = 6144;
        constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
        constexpr std::int32_t kParentRows = kChannels + kZRows;
        const ConvGeometry geometry        = require_snapshot_input(x, kHidden);
        detail::validate_nvfp4_weight(weight, "nvfp4 gdn_input_proj_conv_snapshot");
        if (weight.n != kParentRows || weight.k != kHidden) {
            throw std::invalid_argument(
                "nvfp4 gdn_input_proj_conv_snapshot: unsupported weight shape");
        }
        require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                  snapshot_base_slots, kChannels, geometry);
        require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "query");
        require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "key");
        require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "value");
        require_conv_tensor(z, kZRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "z");
        const detail::Nvfp4GdnConvPlan plan =
            detail::nvfp4_gdn_conv_resolve_plan(policy, geometry.width, geometry.batch);
        if (geometry.batch > 1) {
            if (plan.schedule != detail::Nvfp4GdnConvScheduleId::Materialized) {
                throw std::logic_error("batched NVFP4 GDN conv selected a fused schedule");
            }
            compose_batched_snapshot(x, conv_weight, conv_states, valid_columns,
                                     initial_state_slots, snapshot_base_slots, query, key, value, z,
                                     kQueryRows, kKeyRows, kValueRows, geometry, workspace, stream,
                                     [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                                         gdn_input_proj(x_flat, weight, projected, z_flat, policy,
                                                        workspace, stream);
                                     });
            return;
        }
        detail::nvfp4_gdn_snapshot_dispatch(x, weight, conv_weight, conv_states, valid_columns,
                                            initial_state_slots, snapshot_base_slots, query, key,
                                            value, z, policy, workspace, stream);
        return;
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        constexpr std::int32_t kHidden     = 5120;
        constexpr std::int32_t kQueryRows  = 2048;
        constexpr std::int32_t kKeyRows    = 2048;
        constexpr std::int32_t kValueRows  = 6144;
        constexpr std::int32_t kZRows      = 6144;
        constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
        constexpr std::int32_t kParentRows = kChannels + kZRows;
        const ConvGeometry geometry        = require_snapshot_input(x, kHidden);
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
            throw std::invalid_argument("FP8 gdn_input_proj_conv_snapshot admits only A16 or A8");
        }
        detail::validate_fp8_weight(weight, "fp8 gdn_input_proj_conv_snapshot");
        if (weight.n != kParentRows || weight.k != kHidden) {
            throw std::invalid_argument(
                "fp8 gdn_input_proj_conv_snapshot: unsupported weight shape");
        }
        require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                  snapshot_base_slots, kChannels, geometry);
        require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "query");
        require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "key");
        require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "value");
        require_conv_tensor(z, kZRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "z");
        require_snapshot_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                    snapshot_base_slots, query, key, value, z, workspace);
        const std::array<const Tensor*, 10> tensors{&x,
                                                    &conv_weight,
                                                    &conv_states,
                                                    &valid_columns,
                                                    &initial_state_slots,
                                                    &snapshot_base_slots,
                                                    &query,
                                                    &key,
                                                    &value,
                                                    &z};
        require_parent_nonoverlap(weight, tensors, workspace, "fp8 gdn_input_proj_conv_snapshot");
        detail::fp8_gdn_snapshot_dispatch(x, weight, conv_weight, conv_states, valid_columns,
                                          initial_state_slots, snapshot_base_slots, query, key,
                                          value, z, policy, workspace, stream);
        return;
    }

    constexpr std::int32_t kHidden    = 2048;
    constexpr std::int32_t kQueryRows = 2048;
    constexpr std::int32_t kKeyRows   = 2048;
    constexpr std::int32_t kValueRows = 4096;
    constexpr std::int32_t kZRows     = 4096;
    constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    const ConvGeometry geometry       = require_snapshot_input(x, kHidden);
    if (policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("W8 gdn_input_proj_conv_snapshot admits only A16");
    }
    require_w8_rowsplit(weight, kChannels + kZRows, "query/key/value/z weight");
    require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                              snapshot_base_slots, kChannels, geometry);
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "value");
    require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_snapshot",
                        "z");
    if (geometry.batch > 1) {
        compose_batched_snapshot(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                 snapshot_base_slots, query, key, value, z, kQueryRows, kKeyRows,
                                 kValueRows, geometry, workspace, stream,
                                 [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                                     gdn_input_proj(x_flat, weight, projected, z_flat, stream);
                                 });
        return;
    }

    const detail::W8GdnInputConvPlan plan = resolve_w8_conv_plan(geometry.width, geometry.batch);
    if (plan.schedule == detail::W8GdnInputConvScheduleId::DecodeFused) {
        detail::w8_gdn_input_decode_conv_snapshot_launch(
            x, weight, conv_weight, conv_states, valid_columns, initial_state_slots,
            snapshot_base_slots, query, key, value, z, stream);
        return;
    }
    if (plan.schedule == detail::W8GdnInputConvScheduleId::SplitKMmaFused) {
        detail::w8_gdn_input_splitk_conv_snapshot_launch(
            x, weight, conv_weight, conv_states, valid_columns, initial_state_slots,
            snapshot_base_slots, query, key, value, z, stream);
        return;
    }

    auto scope                 = workspace.scope();
    ProjectedWorkspace scratch = allocate_projected_workspace(workspace, kChannels, geometry.width);
    gdn_input_proj(x, weight, scratch.projected, z, stream);
    detail::gdn_projected_conv_snapshot_launch(scratch.projected, conv_weight, conv_states,
                                               valid_columns, initial_state_slots,
                                               snapshot_base_slots, query, key, value, stream);
}

void dispatch_single_parent_record(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                   const Tensor& conv_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, Tensor& conv_record,
                                   Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                   LinearPolicy policy, WorkspaceArena& workspace,
                                   cudaStream_t stream) {
    validate_policy(policy);

    if (weight.qtype == QType::GGML_K) {
        constexpr const char* op = "gdn_input_proj_conv_record";
        require_ggml_k_parent(weight, 16384, policy);
        const ConvGeometry geometry = require_record_input(x, 5120);
        require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                10240, geometry);
        require_conv_tensor(conv_record, 10240, geometry.width, geometry.batch, op, "conv record");
        require_conv_tensor(query, 2048, geometry.width, geometry.batch, op, "query");
        require_conv_tensor(key, 2048, geometry.width, geometry.batch, op, "key");
        require_conv_tensor(value, 6144, geometry.width, geometry.batch, op, "value");
        require_conv_tensor(z, 6144, geometry.width, geometry.batch, op, "z");
        require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                 conv_record, query, key, value, z, workspace);
        compose_record(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                       conv_record, query, key, value, z, geometry, workspace, stream,
                       [&](const Tensor& x_flat, Tensor& record_flat, Tensor& z_flat) {
                           project_ggml_k(x_flat, weight, record_flat, z_flat, stream, &workspace);
                       });
        return;
    }

    if (weight.qtype == QType::NVFP4) {
        constexpr std::int32_t kHidden     = 5120;
        constexpr std::int32_t kQueryRows  = 2048;
        constexpr std::int32_t kKeyRows    = 2048;
        constexpr std::int32_t kValueRows  = 6144;
        constexpr std::int32_t kZRows      = 6144;
        constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
        constexpr std::int32_t kParentRows = kChannels + kZRows;
        const ConvGeometry geometry        = require_record_input(x, kHidden);
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
            throw std::invalid_argument("NVFP4 gdn_input_proj_conv_record admits only A16 or A4");
        }
        detail::validate_nvfp4_weight(weight, "nvfp4 gdn_input_proj_conv_record");
        if (weight.n != kParentRows || weight.k != kHidden) {
            throw std::invalid_argument(
                "nvfp4 gdn_input_proj_conv_record: unsupported weight shape");
        }
        require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                kChannels, geometry);
        require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "conv record");
        require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "query");
        require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "key");
        require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "value");
        require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                            "z");
        require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                  conv_record, query, key, value, z, workspace);

        const detail::Nvfp4GdnConvPlan plan =
            detail::nvfp4_gdn_conv_resolve_plan(policy, geometry.width, geometry.batch);
        if (plan.schedule == detail::Nvfp4GdnConvScheduleId::Materialized && geometry.batch > 1) {
            compose_record(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                           conv_record, query, key, value, z, geometry, workspace, stream,
                           [&](const Tensor& x_flat, Tensor& record_flat, Tensor& z_flat) {
                               gdn_input_proj(x_flat, weight, record_flat, z_flat, policy,
                                              workspace, stream);
                           });
            return;
        }
        if (plan.schedule == detail::Nvfp4GdnConvScheduleId::SmallTFusedA16) {
            detail::nvfp4_gdn_record_small_t_launch(x, weight, conv_weight, conv_states,
                                                    valid_columns, initial_state_slots, conv_record,
                                                    query, key, value, z, stream);
            return;
        }

        auto scope = workspace.scope();
        gdn_input_proj(x, weight, conv_record, z, policy, workspace, stream);
        detail::nvfp4_gdn_record_post_launch(conv_record, conv_weight, conv_states, valid_columns,
                                             initial_state_slots, query, key, value, stream);
        return;
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        constexpr std::int32_t kHidden     = 5120;
        constexpr std::int32_t kQueryRows  = 2048;
        constexpr std::int32_t kKeyRows    = 2048;
        constexpr std::int32_t kValueRows  = 6144;
        constexpr std::int32_t kZRows      = 6144;
        constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
        constexpr std::int32_t kParentRows = kChannels + kZRows;
        const ConvGeometry geometry        = require_record_input(x, kHidden);
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
            throw std::invalid_argument("FP8 gdn_input_proj_conv_record admits only A16 or A8");
        }
        detail::validate_fp8_weight(weight, "fp8 gdn_input_proj_conv_record");
        if (weight.n != kParentRows || weight.k != kHidden) {
            throw std::invalid_argument("fp8 gdn_input_proj_conv_record: unsupported weight shape");
        }
        require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                kChannels, geometry);
        require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "conv record");
        require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "query");
        require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "key");
        require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "value");
        require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                            "z");
        require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                  conv_record, query, key, value, z, workspace);
        const std::array<const Tensor*, 10> tensors{
            &x,           &conv_weight, &conv_states, &valid_columns, &initial_state_slots,
            &conv_record, &query,       &key,         &value,         &z};
        require_parent_nonoverlap(weight, tensors, workspace, "fp8 gdn_input_proj_conv_record");
        detail::fp8_gdn_record_dispatch(x, weight, conv_weight, conv_states, valid_columns,
                                        initial_state_slots, conv_record, query, key, value, z,
                                        policy, workspace, stream);
        return;
    }

    constexpr std::int32_t kHidden    = 2048;
    constexpr std::int32_t kQueryRows = 2048;
    constexpr std::int32_t kKeyRows   = 2048;
    constexpr std::int32_t kValueRows = 4096;
    constexpr std::int32_t kZRows     = 4096;
    constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    const ConvGeometry geometry       = require_record_input(x, kHidden);
    if (policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("W8 gdn_input_proj_conv_record admits only A16");
    }
    require_w8_rowsplit(weight, kChannels + kZRows, "query/key/value/z weight");
    require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots, kChannels,
                            geometry);
    require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "conv record");
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                        "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "value");
    require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                        "z");
    require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                              conv_record, query, key, value, z, workspace);

    if (geometry.batch > 1) {
        compose_record(x, conv_weight, conv_states, valid_columns, initial_state_slots, conv_record,
                       query, key, value, z, geometry, workspace, stream,
                       [&](const Tensor& x_flat, Tensor& record_flat, Tensor& z_flat) {
                           gdn_input_proj(x_flat, weight, record_flat, z_flat, stream);
                       });
        return;
    }
    const detail::W8GdnInputConvPlan plan = resolve_w8_conv_plan(geometry.width, geometry.batch);
    if (plan.schedule == detail::W8GdnInputConvScheduleId::Materialized) {
        compose_record(x, conv_weight, conv_states, valid_columns, initial_state_slots, conv_record,
                       query, key, value, z, geometry, workspace, stream,
                       [&](const Tensor& x_flat, Tensor& record_flat, Tensor& z_flat) {
                           gdn_input_proj(x_flat, weight, record_flat, z_flat, stream);
                       });
        return;
    }
    if (plan.schedule != detail::W8GdnInputConvScheduleId::SplitKMmaFused) {
        throw std::logic_error("W8 ReplaySSM record domain selected a non-record schedule");
    }
    detail::w8_gdn_input_splitk_conv_record_launch(x, weight, conv_weight, conv_states,
                                                   valid_columns, initial_state_slots, conv_record,
                                                   query, key, value, z, stream);
}

} // namespace

void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    Tensor& qkv, Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQkRows     = 4096;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kQkvRows    = kQkRows + kValueRows;
    constexpr std::int32_t kParentRows = kValueRows + kZRows;
    const std::int32_t cols            = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("gdn_input_proj: T must be positive"); }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(qkv, kQkvRows, cols, "qkv");
    require_matrix(z, kZRows, cols, "z");
    require_rowsplit(qk_weight, QType::Q4G64_F16S, kQkRows, "qk weight");
    require_rowsplit(value_z_weight, QType::Q5G64_F16S, kParentRows, "value/z weight");

    detail::q4_q5_gdn_input_dispatch(x, qk_weight, value_z_weight, qkv, z, workspace, stream);
}

std::size_t q4_q5_gdn_input_proj_workspace_capacity_bytes(std::int32_t min_tokens,
                                                           std::int32_t max_tokens) {
    return detail::q4_q5_gdn_input_capacity_workspace_bytes(min_tokens, max_tokens);
}

std::size_t gdn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                                    std::int32_t input_rows, LinearPolicy policy,
                                                    std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gdn_input_proj workspace: invalid token interval");
    }
    if (parent_qtype == QType::GGML_K && parent_rows == 16384 && input_rows == 5120 &&
        policy == LinearPolicy::A16Only) {
        return detail::ggml_k_cutlass_workspace_bytes(parent_rows, input_rows, max_tokens);
    }
    if (parent_qtype == QType::NVFP4) {
        if (parent_rows != detail::Nvfp4GdnInputGeometry::kOutputRows ||
            input_rows != detail::Nvfp4GdnInputGeometry::kInputRows ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
            throw std::invalid_argument("gdn_input_proj workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_gdn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    if (parent_qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        if (parent_rows != detail::Fp8GdnInputGeometry::kOutputRows ||
            input_rows != detail::Fp8GdnInputGeometry::kInputRows ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8)) {
            throw std::invalid_argument("gdn_input_proj workspace: unsupported FP8 profile");
        }
        return detail::fp8_gdn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    if (parent_qtype == QType::W8G32_F16S && parent_rows == 12288 && input_rows == 2048 &&
        policy == LinearPolicy::A16Only) {
        (void)detail::w8_gdn_input_resolve_plan(
            {input_rows, 8192, 4096, parent_rows, input_rows, min_tokens});
        (void)detail::w8_gdn_input_resolve_plan(
            {input_rows, 8192, 4096, parent_rows, input_rows, max_tokens});
        // Zero for every schedule that computes in place; the Volta CUTLASS route
        // stages a dequantized [12288,2048] parent plus the activation cast, and
        // that has to be declared here or the arena rejects it at run time.
        return detail::w8_gdn_input_workspace_bytes(min_tokens, max_tokens);
    }
    throw std::invalid_argument("gdn_input_proj workspace: unsupported parent profile");
}

void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_value_z_weight, qkv, z, policy, &workspace, stream);
}

void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    cudaStream_t stream) {
    dispatch_single_parent(x, query_key_value_z_weight, qkv, z, LinearPolicy::A16Only, nullptr,
                           stream);
}

std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    std::int32_t query_rows, std::int32_t key_rows, std::int32_t value_rows,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    const bool q4_q5 = query_rows == 2048 && key_rows == 2048 && value_rows == 6144;
    const bool w8    = query_rows == 2048 && key_rows == 2048 && value_rows == 4096;
    if (!q4_q5 && !w8) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot workspace: unregistered shape");
    }
    require_snapshot_capacity_domain(batch_size, min_width, max_width);
    const std::int32_t channels = query_rows + key_rows + value_rows;
    if (batch_size > 1) {
        if (q4_q5) {
            (void)resolve_q4_q5_conv_plan(min_width, batch_size);
            (void)resolve_q4_q5_conv_plan(max_width, batch_size);
        } else {
            (void)resolve_w8_conv_plan(min_width, batch_size);
            (void)resolve_w8_conv_plan(max_width, batch_size);
        }
        // The composed path's `projected` buffer stays live across the nested
        // gdn_input_proj(..., projected, ...) call inside its lambda (see
        // compose_batched_snapshot), so that call's own transient need (nonzero only on Volta
        // wide-T) must be added on top, not just sized for `projected` alone.
        const std::size_t projection_workspace =
            q4_q5 ? detail::q4_q5_gdn_input_capacity_workspace_bytes(
                        batch_size * min_width, batch_size * max_width)
                  : 0;
        return composed_snapshot_capacity(channels, batch_size * max_width,
                                          projection_workspace);
    }

    std::int32_t largest_materialized_width = 0;
    if (q4_q5) {
        (void)resolve_q4_q5_conv_plan(min_width, 1);
        (void)resolve_q4_q5_conv_plan(max_width, 1);
        if (max_width >= 7) {
            largest_materialized_width = max_width;
        } else if (min_width <= 4 && max_width >= 4) {
            largest_materialized_width = 4;
        }
    } else {
        (void)resolve_w8_conv_plan(min_width, 1);
        (void)resolve_w8_conv_plan(max_width, 1);
#ifdef NINFER_VOLTA_BUILD
        if (max_width >= 2) { largest_materialized_width = max_width; }
#else
        if (max_width >= 17) { largest_materialized_width = max_width; }
#endif // NINFER_VOLTA_BUILD
    }
    if (largest_materialized_width == 0) { return 0; }
    // Same nesting concern as the batch_size>1 branch above: `scratch.projected` stays live
    // across the nested gdn_input_proj(..., scratch.projected, ...) call (see the
    // Materialized branch of gdn_input_proj_conv_snapshot), so add its own transient need.
    const std::size_t projection_workspace =
        q4_q5 ? detail::q4_q5_gdn_input_capacity_workspace_bytes(1, largest_materialized_width)
              : 0;
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected_workspace(layout, channels, largest_materialized_width);
    if (projection_workspace != 0) { (void)layout.alloc_bytes(projection_workspace); }
    return layout.peak_bytes(1);
}

std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    validate_policy(policy);
    require_snapshot_capacity_domain(batch_size, min_width, max_width);
    if (parent_qtype == QType::GGML_K && parent_rows == 16384 && input_rows == 5120 &&
        policy == LinearPolicy::A16Only) {
        return composed_snapshot_capacity(
            10240, batch_size * max_width,
            detail::ggml_k_cutlass_workspace_bytes(parent_rows, input_rows,
                                                   batch_size * max_width));
    }
    if (parent_qtype == QType::FP8_E4M3FN_ROW_BF16S &&
        parent_rows == detail::Fp8GdnInputGeometry::kOutputRows &&
        input_rows == detail::Fp8GdnInputGeometry::kInputRows &&
        (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8)) {
        return detail::fp8_gdn_snapshot_workspace_capacity_bytes(policy, batch_size, min_width,
                                                                 max_width);
    }
    if (parent_qtype != QType::NVFP4 || parent_rows != detail::Nvfp4GdnInputGeometry::kOutputRows ||
        input_rows != detail::Nvfp4GdnInputGeometry::kInputRows) {
        throw std::invalid_argument(
            "gdn_input_proj_conv_snapshot workspace: unsupported single-parent profile");
    }
    if (batch_size == 1) {
        return detail::nvfp4_gdn_snapshot_workspace_capacity_bytes(policy, min_width, max_width);
    }

    (void)detail::nvfp4_gdn_conv_resolve_plan(policy, min_width, batch_size);
    (void)detail::nvfp4_gdn_conv_resolve_plan(policy, max_width, batch_size);

    constexpr std::int32_t kChannels       = 10240;
    const std::int32_t aggregate_columns   = batch_size * max_width;
    const std::size_t projection_workspace = gdn_input_proj_workspace_capacity_bytes(
        parent_qtype, parent_rows, input_rows, policy, batch_size * min_width, aggregate_columns);
    return composed_snapshot_capacity(kChannels, aggregate_columns, projection_workspace);
}

std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(
    std::int32_t query_rows, std::int32_t key_rows, std::int32_t value_rows,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    const bool q4_q5 = query_rows == 2048 && key_rows == 2048 && value_rows == 6144;
    const bool w8    = query_rows == 2048 && key_rows == 2048 && value_rows == 4096;
    if (!q4_q5 && !w8) {
        throw std::invalid_argument("gdn_input_proj_conv_record workspace: unregistered shape");
    }
    require_record_capacity_domain(batch_size, min_width, max_width);
    if (q4_q5) {
        (void)resolve_q4_q5_conv_plan(min_width, batch_size);
        (void)resolve_q4_q5_conv_plan(max_width, batch_size);
        // The ProjectionEpilogueFused branch and W8 need nothing, but the Materialized
        // fallback's nested gdn_input_proj(..., record_flat, z_flat, ...) call (see
        // compose_record) writes directly into caller-owned conv_record -- no outer
        // "projected" buffer here, unlike the snapshot path -- but still has its own
        // transient need on Volta wide-T (the CUTLASS route), which must be reported.
        return detail::q4_q5_gdn_input_capacity_workspace_bytes(batch_size * min_width,
                                                                 batch_size * max_width);
    }
    (void)resolve_w8_conv_plan(min_width, batch_size);
    (void)resolve_w8_conv_plan(max_width, batch_size);
    return 0;
}

std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    validate_policy(policy);
    require_record_capacity_domain(batch_size, min_width, max_width);
    if (parent_qtype == QType::GGML_K && parent_rows == 16384 && input_rows == 5120 &&
        policy == LinearPolicy::A16Only) {
        return detail::ggml_k_cutlass_workspace_bytes(
            parent_rows, input_rows, batch_size * max_width);
    }
    if (parent_qtype == QType::FP8_E4M3FN_ROW_BF16S &&
        parent_rows == detail::Fp8GdnInputGeometry::kOutputRows &&
        input_rows == detail::Fp8GdnInputGeometry::kInputRows &&
        (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8)) {
        return detail::fp8_gdn_record_workspace_capacity_bytes(policy, batch_size, min_width,
                                                               max_width);
    }
    if (parent_qtype != QType::NVFP4 || parent_rows != detail::Nvfp4GdnInputGeometry::kOutputRows ||
        input_rows != detail::Nvfp4GdnInputGeometry::kInputRows ||
        (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
        throw std::invalid_argument(
            "gdn_input_proj_conv_record workspace: unsupported single-parent profile");
    }
    const detail::Nvfp4GdnConvPlan minimum_plan =
        detail::nvfp4_gdn_conv_resolve_plan(policy, min_width, batch_size);
    const detail::Nvfp4GdnConvPlan maximum_plan =
        detail::nvfp4_gdn_conv_resolve_plan(policy, max_width, batch_size);
    if (batch_size == 1) {
        if (minimum_plan.schedule == detail::Nvfp4GdnConvScheduleId::DecodeFusedA16) {
            throw std::logic_error("ReplaySSM record planner admitted NVFP4 decode");
        }
        if (maximum_plan.schedule == detail::Nvfp4GdnConvScheduleId::SmallTFusedA16) { return 0; }
        return detail::nvfp4_gdn_input_workspace_capacity_bytes(detail::kNvfp4InternalPolicy,
                                                                std::max(min_width, 4), max_width);
    }
    return detail::nvfp4_gdn_input_workspace_capacity_bytes(policy, batch_size * min_width,
                                                            batch_size * max_width);
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& qk_weight,
                                  const Weight& value_z_weight, const Tensor& conv_weight,
                                  Tensor& conv_states, const Tensor& valid_columns,
                                  const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQueryRows  = 2048;
    constexpr std::int32_t kKeyRows    = 2048;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kParentRows = kValueRows + kZRows;
    const ConvGeometry geometry        = require_snapshot_input(x, kHidden);
    require_rowsplit(qk_weight, QType::Q4G64_F16S, kQueryRows + kKeyRows, "qk weight");
    require_rowsplit(value_z_weight, QType::Q5G64_F16S, kParentRows, "value/z weight");
    require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                              snapshot_base_slots, kChannels, geometry);
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "value");
    require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_snapshot",
                        "z");

    if (geometry.batch > 1) {
        compose_batched_snapshot(
            x, conv_weight, conv_states, valid_columns, initial_state_slots, snapshot_base_slots,
            query, key, value, z, kQueryRows, kKeyRows, kValueRows, geometry, ws, stream,
            [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                gdn_input_proj(x_flat, qk_weight, value_z_weight, projected, z_flat, ws, stream);
            });
        return;
    }

    const detail::Q4Q5GdnInputConvPlan plan =
        resolve_q4_q5_conv_plan(geometry.width, geometry.batch);
    if (plan.schedule == detail::Q4Q5GdnInputConvScheduleId::ProjectionEpilogueFused) {
        detail::q4_q5_gdn_input_conv_snapshot_launch(
            x, qk_weight, value_z_weight, conv_weight, conv_states, valid_columns,
            initial_state_slots, snapshot_base_slots, query, key, value, z, stream);
        return;
    }

    auto scope                 = ws.scope();
    ProjectedWorkspace scratch = allocate_projected_workspace(ws, kChannels, geometry.width);
    gdn_input_proj(x, qk_weight, value_z_weight, scratch.projected, z, ws, stream);
    detail::gdn_projected_conv_snapshot_launch(scratch.projected, conv_weight, conv_states,
                                               valid_columns, initial_state_slots,
                                               snapshot_base_slots, query, key, value, stream);
}

void gdn_input_proj_conv_record(const Tensor& x, const Weight& qk_weight,
                                const Weight& value_z_weight, const Tensor& conv_weight,
                                const Tensor& conv_states, const Tensor& valid_columns,
                                const Tensor& initial_state_slots, Tensor& conv_record,
                                Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQueryRows  = 2048;
    constexpr std::int32_t kKeyRows    = 2048;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kParentRows = kValueRows + kZRows;
    const ConvGeometry geometry        = require_record_input(x, kHidden);
    require_rowsplit(qk_weight, QType::Q4G64_F16S, kQueryRows + kKeyRows, "qk weight");
    require_rowsplit(value_z_weight, QType::Q5G64_F16S, kParentRows, "value/z weight");
    require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots, kChannels,
                            geometry);
    require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "conv record");
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                        "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "value");
    require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                        "z");
    require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                              conv_record, query, key, value, z, workspace);

    const detail::Q4Q5GdnInputConvPlan plan =
        resolve_q4_q5_conv_plan(geometry.width, geometry.batch);
    if (plan.schedule == detail::Q4Q5GdnInputConvScheduleId::ProjectionEpilogueFused) {
        detail::q4_q5_gdn_input_conv_record_launch(x, qk_weight, value_z_weight, conv_weight,
                                                   conv_states, valid_columns, initial_state_slots,
                                                   conv_record, query, key, value, z, stream);
        return;
    }
    compose_record(x, conv_weight, conv_states, valid_columns, initial_state_slots, conv_record,
                   query, key, value, z, geometry, workspace, stream,
                   [&](const Tensor& x_flat, Tensor& record_flat, Tensor& z_flat) {
                       gdn_input_proj(x_flat, qk_weight, value_z_weight, record_flat, z_flat,
                                      workspace, stream);
                   });
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, LinearPolicy policy, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    dispatch_single_parent_snapshot(x, query_key_value_z_weight, conv_weight, conv_states,
                                    valid_columns, initial_state_slots, snapshot_base_slots, query,
                                    key, value, z, policy, ws, stream);
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    dispatch_single_parent_snapshot(x, query_key_value_z_weight, conv_weight, conv_states,
                                    valid_columns, initial_state_slots, snapshot_base_slots, query,
                                    key, value, z, LinearPolicy::A16Only, ws, stream);
}

void gdn_input_proj_conv_record(const Tensor& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    dispatch_single_parent_record(x, query_key_value_z_weight, conv_weight, conv_states,
                                  valid_columns, initial_state_slots, conv_record, query, key,
                                  value, z, policy, workspace, stream);
}

void gdn_input_proj_conv_record(const Tensor& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    dispatch_single_parent_record(x, query_key_value_z_weight, conv_weight, conv_states,
                                  valid_columns, initial_state_slots, conv_record, query, key,
                                  value, z, LinearPolicy::A16Only, workspace, stream);
}

// --- Tensor-parallel split form (tp == 2) -------------------------------------------------------
// See include/ninfer/ops/gdn_input_proj.h for the full design note (ShardPlan section layout, the
// output contract the GDN core reads, and which formats are registered).
namespace {

constexpr std::int32_t kShardHidden       = 5120;
constexpr std::int32_t kShardQkvRows      = 5120;
constexpr std::int32_t kShardZRows        = 3072;
constexpr std::int32_t kShardFusedRows    = 8192;  // Nvfp4/Fp8GdnInputTp2ColumnGeometry::kOutputRows
constexpr std::int32_t kShardQueryKeyRows = 2048;  // Q4G64_F16S query_key shard
constexpr std::int32_t kShardValueZRows   = 6144;  // Q5G64_F16S value_z shard

void validate_fused_column_rank_semantics(const Tensor& x, const Weight& w, const Tensor& qkv,
                                          const Tensor& z, LinearPolicy policy) {
    validate_policy(policy);
    const std::int32_t cols = x.ne[1];
    if (cols <= 0) {
        throw std::invalid_argument("gdn_input_proj column-parallel: T must be positive");
    }
    require_matrix(x, kShardHidden, cols, "x");
    require_matrix(qkv, kShardQkvRows, cols, "qkv");
    require_matrix(z, kShardZRows, cols, "z");
    require_single_parent_nonoverlap(x, qkv, z);

    if (w.qtype == QType::GGML_K) {
        require_ggml_k_parent(w, kShardFusedRows, policy);
    } else if (w.qtype == QType::NVFP4) {
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
            throw std::invalid_argument(
                "gdn_input_proj column-parallel: NVFP4 admits only A16 or A4");
        }
        detail::validate_nvfp4_weight(w, "nvfp4 gdn_input_proj column-parallel");
    } else if (w.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
            throw std::invalid_argument(
                "gdn_input_proj column-parallel: FP8 admits only A16 or A8");
        }
        detail::validate_fp8_weight(w, "fp8 gdn_input_proj column-parallel");
    } else {
        throw std::invalid_argument(
            "gdn_input_proj column-parallel: unsupported fused weight format");
    }
    if (w.n != kShardFusedRows || w.k != kShardHidden) {
        throw std::invalid_argument(
            "gdn_input_proj column-parallel: unsupported weight shard shape");
    }
}

// Cross-rank agreement only a pair can check; every per-rank invariant is validated separately by
// validate_fused_column_rank_semantics. Mirrors attn_input_proj's own validate_fused_split_pair
// (src/ops/wrapper/attn_input_proj.cpp).
void validate_fused_split_pair(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                               const ExecutionContext& ec) {
    detail::require_split_context(
        ec, "gdn_input_proj column-parallel: requires an ExecutionContext with two distinct "
            "devices");
    if (x[0].ne[1] != x[1].ne[1]) {
        throw std::invalid_argument(
            "gdn_input_proj column-parallel: both ranks must carry the same token count");
    }
    if (w[0].qtype != w[1].qtype || w[0].layout != w[1].layout) {
        throw std::invalid_argument(
            "gdn_input_proj column-parallel: both ranks must carry the same weight format");
    }
    if (w[0].k != w[1].k) {
        throw std::invalid_argument(
            "gdn_input_proj column-parallel: both ranks must consume the same input extent K");
    }
}

void validate_split_storage_column_rank_semantics(const Tensor& x, const Weight& query_key_w,
                                                   const Weight& value_z_w, const Tensor& qkv,
                                                   const Tensor& z) {
    const std::int32_t cols = x.ne[1];
    if (cols <= 0) {
        throw std::invalid_argument("gdn_input_proj column-parallel: T must be positive");
    }
    require_matrix(x, kShardHidden, cols, "x");
    require_matrix(qkv, kShardQkvRows, cols, "qkv");
    require_matrix(z, kShardZRows, cols, "z");
    require_single_parent_nonoverlap(x, qkv, z);
    require_rowsplit(query_key_w, QType::Q4G64_F16S, kShardQueryKeyRows, "query/key weight shard");
    require_rowsplit(value_z_w, QType::Q5G64_F16S, kShardValueZRows, "value/z weight shard");
}

void validate_split_storage_split_pair(const std::array<Tensor, 2>& x,
                                       const std::array<Weight, 2>& query_key_w,
                                       const std::array<Weight, 2>& value_z_w,
                                       const ExecutionContext& ec) {
    detail::require_split_context(
        ec, "gdn_input_proj column-parallel: requires an ExecutionContext with two distinct "
            "devices");
    if (x[0].ne[1] != x[1].ne[1]) {
        throw std::invalid_argument(
            "gdn_input_proj column-parallel: both ranks must carry the same token count");
    }
    if (query_key_w[0].qtype != query_key_w[1].qtype ||
        value_z_w[0].qtype != value_z_w[1].qtype) {
        throw std::invalid_argument(
            "gdn_input_proj column-parallel: both ranks must carry the same weight format");
    }
    if (query_key_w[0].k != query_key_w[1].k || value_z_w[0].k != value_z_w[1].k) {
        throw std::invalid_argument(
            "gdn_input_proj column-parallel: both ranks must consume the same input extent K");
    }
}

} // namespace

std::size_t gdn_input_proj_column_parallel_workspace_capacity_bytes(QType qtype,
                                                                     LinearPolicy policy,
                                                                     std::int32_t min_tokens,
                                                                     std::int32_t max_tokens) {
    if (qtype == QType::GGML_K && policy == LinearPolicy::A16Only && min_tokens > 0 &&
        max_tokens >= min_tokens) {
        return detail::ggml_k_cutlass_workspace_bytes(
            kShardFusedRows, kShardHidden, max_tokens);
    }
    if (qtype == QType::NVFP4) {
        return detail::nvfp4_gdn_input_shard_workspace_capacity_bytes(policy, min_tokens,
                                                                       max_tokens);
    }
    if (qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        return detail::fp8_gdn_input_shard_workspace_capacity_bytes(policy, min_tokens,
                                                                     max_tokens);
    }
    throw std::invalid_argument(
        "gdn_input_proj column-parallel workspace: unsupported weight format");
}

void gdn_input_proj_column_parallel(const std::array<Tensor, 2>& x,
                                    const std::array<Weight, 2>& query_key_value_z_weight,
                                    const std::array<Tensor, 2>& qkv, const std::array<Tensor, 2>& z,
                                    LinearPolicy policy,
                                    const std::array<WorkspaceArena*, 2>& workspace,
                                    const ExecutionContext& ec) {
    validate_fused_split_pair(x, query_key_value_z_weight, ec);
    // Validate both ranks before issuing either, so a rejected pair enqueues nothing.
    for (std::size_t slot = 0; slot < 2; ++slot) {
        validate_fused_column_rank_semantics(x[slot], query_key_value_z_weight[slot], qkv[slot],
                                             z[slot], policy);
    }
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        detail::require_rank_residency(
            ec, rank, x[slot].data, query_key_value_z_weight[slot].payload, qkv[slot].data,
            "gdn_input_proj column-parallel: every per-rank argument must be resident on "
            "ec.dev[rank]");
    }
    std::array<Tensor, 2> qkv_dst{qkv[0], qkv[1]};
    std::array<Tensor, 2> z_dst{z[0], z[1]};
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        const Weight& w = query_key_value_z_weight[slot];
        if (w.qtype == QType::GGML_K) {
            project_ggml_k(x[slot], w, qkv_dst[slot], z_dst[slot], ec.dev[slot]->stream,
                           workspace[slot]);
        } else if (w.qtype == QType::NVFP4) {
            detail::nvfp4_gdn_input_dispatch_shard(x[slot], w, qkv_dst[slot], z_dst[slot], policy,
                                                   workspace[slot], ec.dev[slot]->stream);
        } else {
            detail::fp8_gdn_input_dispatch_shard(x[slot], w, qkv_dst[slot], z_dst[slot], policy,
                                                 workspace[slot], ec.dev[slot]->stream);
        }
    });
}

void gdn_input_proj_column_parallel(const std::array<Tensor, 2>& x,
                                    const std::array<Weight, 2>& query_key_value_z_weight,
                                    const std::array<Tensor, 2>& qkv, const std::array<Tensor, 2>& z,
                                    const ExecutionContext& ec) {
    gdn_input_proj_column_parallel(x, query_key_value_z_weight, qkv, z, LinearPolicy::A16Only,
                                   {nullptr, nullptr}, ec);
}

void gdn_input_proj_column_parallel(const std::array<Tensor, 2>& x,
                                    const std::array<Weight, 2>& query_key_weight,
                                    const std::array<Weight, 2>& value_z_weight,
                                    const std::array<Tensor, 2>& qkv, const std::array<Tensor, 2>& z,
                                    const ExecutionContext& ec) {
    validate_split_storage_split_pair(x, query_key_weight, value_z_weight, ec);
    for (std::size_t slot = 0; slot < 2; ++slot) {
        validate_split_storage_column_rank_semantics(x[slot], query_key_weight[slot],
                                                      value_z_weight[slot], qkv[slot], z[slot]);
    }
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        detail::require_rank_residency(
            ec, rank, x[slot].data, query_key_weight[slot].payload, qkv[slot].data,
            "gdn_input_proj column-parallel: every per-rank argument must be resident on "
            "ec.dev[rank]");
        detail::require_rank_residency(
            ec, rank, x[slot].data, value_z_weight[slot].payload, z[slot].data,
            "gdn_input_proj column-parallel: every per-rank argument must be resident on "
            "ec.dev[rank]");
    }
    std::array<Tensor, 2> qkv_dst{qkv[0], qkv[1]};
    std::array<Tensor, 2> z_dst{z[0], z[1]};
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot = static_cast<std::size_t>(rank);
        WorkspaceArena no_workspace(DeviceSpan{});
        detail::q4_q5_gdn_input_dispatch(x[slot], query_key_weight[slot], value_z_weight[slot],
                                         qkv_dst[slot], z_dst[slot], no_workspace,
                                         ec.dev[slot]->stream);
    });
}

// --- Tensor-parallel conv_snapshot / conv_record (tp == 2) ---------------------------------------
// See include/ninfer/ops/gdn_input_proj.h for the full design note (geometry, why the shard always
// takes the composed route, and which formats are registered).
namespace {

constexpr std::int32_t kShardQueryRows   = 1024; // 8 of 16 key heads x 128
constexpr std::int32_t kShardKeyRows     = 1024;
constexpr std::int32_t kShardValueRows   = 3072; // 24 of 48 value heads x 128
constexpr std::int32_t kShardConvChannels =
    kShardQueryRows + kShardKeyRows + kShardValueRows; // == kShardQkvRows (5120)
static_assert(kShardConvChannels == kShardQkvRows);

void require_conv_split_pair(const std::array<Tensor, 2>& x, const ExecutionContext& ec,
                             const char* op) {
    const std::string message =
        std::string(op) + ": requires an ExecutionContext with two distinct devices";
    detail::require_split_context(ec, message.c_str());
    if (x[0].ne[1] != x[1].ne[1] || x[0].ne[2] != x[1].ne[2]) {
        throw std::invalid_argument(std::string(op) +
                                    ": both ranks must carry the same width and batch");
    }
}

void require_conv_shard_workspace(const std::array<WorkspaceArena*, 2>& workspace, const char* op) {
    if (workspace[0] == nullptr || workspace[1] == nullptr) {
        throw std::invalid_argument(std::string(op) +
                                    ": the composed shard route requires a caller workspace on "
                                    "every rank");
    }
}

// Per-rank shard validation shared by both Ops' fused and split-storage forms. Returns the
// convolution geometry so the caller does not re-derive it.
ConvGeometry validate_snapshot_shard_rank(const Tensor& x, const Tensor& conv_weight,
                                          const Tensor& conv_states, const Tensor& valid_columns,
                                          const Tensor& initial_state_slots,
                                          const Tensor& snapshot_base_slots, const Tensor& query,
                                          const Tensor& key, const Tensor& value, const Tensor& z) {
    constexpr const char* kOp   = "gdn_input_proj_conv_snapshot";
    const ConvGeometry geometry = require_snapshot_input(x, kShardHidden);
    require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                              snapshot_base_slots, kShardConvChannels, geometry);
    require_conv_tensor(query, kShardQueryRows, geometry.width, geometry.batch, kOp, "query");
    require_conv_tensor(key, kShardKeyRows, geometry.width, geometry.batch, kOp, "key");
    require_conv_tensor(value, kShardValueRows, geometry.width, geometry.batch, kOp, "value");
    require_conv_tensor(z, kShardZRows, geometry.width, geometry.batch, kOp, "z");
    return geometry;
}

ConvGeometry validate_record_shard_rank(const Tensor& x, const Tensor& conv_weight,
                                        const Tensor& conv_states, const Tensor& valid_columns,
                                        const Tensor& initial_state_slots,
                                        const Tensor& conv_record, const Tensor& query,
                                        const Tensor& key, const Tensor& value, const Tensor& z) {
    constexpr const char* kOp   = "gdn_input_proj_conv_record";
    const ConvGeometry geometry = require_record_input(x, kShardHidden);
    require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                            kShardConvChannels, geometry);
    require_conv_tensor(conv_record, kShardConvChannels, geometry.width, geometry.batch, kOp,
                        "conv record");
    require_conv_tensor(query, kShardQueryRows, geometry.width, geometry.batch, kOp, "query");
    require_conv_tensor(key, kShardKeyRows, geometry.width, geometry.batch, kOp, "key");
    require_conv_tensor(value, kShardValueRows, geometry.width, geometry.batch, kOp, "value");
    require_conv_tensor(z, kShardZRows, geometry.width, geometry.batch, kOp, "z");
    return geometry;
}

void validate_fused_shard_weight(const Weight& w, LinearPolicy policy, const char* op) {
    validate_policy(policy);
    if (w.qtype == QType::GGML_K) {
        require_ggml_k_parent(w, kShardFusedRows, policy);
    } else if (w.qtype == QType::NVFP4) {
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
            throw std::invalid_argument(std::string(op) + ": NVFP4 admits only A16 or A4");
        }
        detail::validate_nvfp4_weight(w, op);
    } else if (w.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
            throw std::invalid_argument(std::string(op) + ": FP8 admits only A16 or A8");
        }
        detail::validate_fp8_weight(w, op);
    } else {
        throw std::invalid_argument(std::string(op) + ": unsupported fused weight format");
    }
    if (w.n != kShardFusedRows || w.k != kShardHidden) {
        throw std::invalid_argument(std::string(op) + ": unsupported weight shard shape");
    }
}

std::size_t shard_projection_workspace_bytes(QType qtype, LinearPolicy policy,
                                             std::int32_t min_columns, std::int32_t max_columns,
                                             const char* op) {
    if (qtype == QType::GGML_K && policy == LinearPolicy::A16Only && min_columns > 0 &&
        max_columns >= min_columns) {
        return detail::ggml_k_cutlass_workspace_bytes(
            kShardFusedRows, kShardHidden, max_columns);
    }
    if (qtype == QType::NVFP4) {
        return detail::nvfp4_gdn_input_shard_workspace_capacity_bytes(policy, min_columns,
                                                                       max_columns);
    }
    if (qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        return detail::fp8_gdn_input_shard_workspace_capacity_bytes(policy, min_columns,
                                                                     max_columns);
    }
    if (qtype == QType::Q4G64_F16S || qtype == QType::Q5G64_F16S) {
        return 0; // the Q4/Q5 grouped-MMA route allocates no transient storage.
    }
    throw std::invalid_argument(std::string(op) + ": unsupported weight format");
}

void require_conv_shard_domain(std::int32_t batch_size, std::int32_t min_width,
                               std::int32_t max_width, std::int32_t minimum_width, const char* op) {
    constexpr std::int32_t kMaximumBatch = 8;
    constexpr std::int32_t kMaximumWidth = 16;
    if (batch_size <= 0 || batch_size > kMaximumBatch || min_width < minimum_width ||
        min_width > max_width || (batch_size > 1 && max_width > kMaximumWidth) ||
        (minimum_width > 1 && max_width > kMaximumWidth)) {
        throw std::invalid_argument(std::string(op) + " workspace: invalid B/W domain");
    }
}

// Projects one rank's shard into `destination` [5120, B*W] and writes its z half.
template <class Project>
void compose_shard_conv(const Tensor& x, Tensor& destination, Tensor& z, ConvGeometry geometry,
                        Project&& project) {
    Tensor x_flat = flatten_columns(x, kShardHidden, geometry);
    Tensor z_flat = flatten_columns(z, kShardZRows, geometry);
    project(x_flat, destination, z_flat);
}

} // namespace

std::size_t gdn_input_proj_conv_snapshot_column_parallel_workspace_capacity_bytes(
    QType qtype, LinearPolicy policy, std::int32_t batch_size, std::int32_t min_width,
    std::int32_t max_width) {
    constexpr const char* kOp = "gdn_input_proj_conv_snapshot column-parallel";
    require_conv_shard_domain(batch_size, min_width, max_width, 1, kOp);
    const std::int32_t columns = batch_size * max_width;
    return composed_snapshot_capacity(
        kShardConvChannels, columns,
        shard_projection_workspace_bytes(qtype, policy, batch_size * min_width, columns, kOp));
}

std::size_t gdn_input_proj_conv_record_column_parallel_workspace_capacity_bytes(
    QType qtype, LinearPolicy policy, std::int32_t batch_size, std::int32_t min_width,
    std::int32_t max_width) {
    constexpr const char* kOp = "gdn_input_proj_conv_record column-parallel";
    require_conv_shard_domain(batch_size, min_width, max_width, 2, kOp);
    // conv_record is caller-owned, so the composed record route needs no projected plane -- only
    // whatever activation-quantization storage the projection route itself selects.
    return shard_projection_workspace_bytes(qtype, policy, batch_size * min_width,
                                            batch_size * max_width, kOp);
}

void gdn_input_proj_conv_snapshot_column_parallel(
    const std::array<Tensor, 2>& x, const std::array<Weight, 2>& query_key_value_z_weight,
    const std::array<Tensor, 2>& conv_weight, const std::array<Tensor, 2>& conv_states,
    const std::array<Tensor, 2>& valid_columns, const std::array<Tensor, 2>& initial_state_slots,
    const std::array<Tensor, 2>& snapshot_base_slots, const std::array<Tensor, 2>& query,
    const std::array<Tensor, 2>& key, const std::array<Tensor, 2>& value,
    const std::array<Tensor, 2>& z, LinearPolicy policy,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec) {
    constexpr const char* kOp = "gdn_input_proj_conv_snapshot column-parallel";
    require_conv_split_pair(x, ec, kOp);
    require_conv_shard_workspace(workspace, kOp);
    std::array<ConvGeometry, 2> geometry{};
    for (std::size_t slot = 0; slot < 2; ++slot) {
        validate_fused_shard_weight(query_key_value_z_weight[slot], policy, kOp);
        geometry[slot] = validate_snapshot_shard_rank(
            x[slot], conv_weight[slot], conv_states[slot], valid_columns[slot],
            initial_state_slots[slot], snapshot_base_slots[slot], query[slot], key[slot],
            value[slot], z[slot]);
    }
    if (query_key_value_z_weight[0].qtype != query_key_value_z_weight[1].qtype) {
        throw std::invalid_argument(std::string(kOp) +
                                    ": both ranks must carry the same weight format");
    }
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        detail::require_rank_residency(
            ec, rank, x[slot].data, query_key_value_z_weight[slot].payload, conv_states[slot].data,
            "gdn_input_proj_conv_snapshot column-parallel: every per-rank argument must be "
            "resident on ec.dev[rank]");
    }

    std::array<Tensor, 2> states{conv_states[0], conv_states[1]};
    std::array<Tensor, 2> q_dst{query[0], query[1]};
    std::array<Tensor, 2> k_dst{key[0], key[1]};
    std::array<Tensor, 2> v_dst{value[0], value[1]};
    std::array<Tensor, 2> z_dst{z[0], z[1]};
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot        = static_cast<std::size_t>(rank);
        const Weight& w        = query_key_value_z_weight[slot];
        cudaStream_t stream    = ec.dev[slot]->stream;
        WorkspaceArena& arena  = *workspace[slot];
        auto scope             = arena.scope();
        Tensor projected =
            arena.alloc(DType::BF16, {kShardConvChannels, geometry[slot].aggregate_columns});
        compose_shard_conv(x[slot], projected, z_dst[slot], geometry[slot],
                           [&](const Tensor& x_flat, Tensor& out, Tensor& z_flat) {
                               if (w.qtype == QType::GGML_K) {
                                   project_ggml_k(x_flat, w, out, z_flat, stream, &arena);
                               } else if (w.qtype == QType::NVFP4) {
                                   detail::nvfp4_gdn_input_dispatch_shard(x_flat, w, out, z_flat,
                                                                          policy, &arena, stream);
                               } else {
                                   detail::fp8_gdn_input_dispatch_shard(x_flat, w, out, z_flat,
                                                                        policy, &arena, stream);
                               }
                           });
        Tensor projected_3d(projected.data, DType::BF16,
                            {kShardConvChannels, geometry[slot].width, geometry[slot].batch});
        detail::gdn_projected_conv_snapshot_launch(
            projected_3d, conv_weight[slot], states[slot], valid_columns[slot],
            initial_state_slots[slot], snapshot_base_slots[slot], q_dst[slot], k_dst[slot],
            v_dst[slot], stream);
    });
}

void gdn_input_proj_conv_snapshot_column_parallel(
    const std::array<Tensor, 2>& x, const std::array<Weight, 2>& query_key_weight,
    const std::array<Weight, 2>& value_z_weight, const std::array<Tensor, 2>& conv_weight,
    const std::array<Tensor, 2>& conv_states, const std::array<Tensor, 2>& valid_columns,
    const std::array<Tensor, 2>& initial_state_slots,
    const std::array<Tensor, 2>& snapshot_base_slots, const std::array<Tensor, 2>& query,
    const std::array<Tensor, 2>& key, const std::array<Tensor, 2>& value,
    const std::array<Tensor, 2>& z, const std::array<WorkspaceArena*, 2>& workspace,
    const ExecutionContext& ec) {
    constexpr const char* kOp = "gdn_input_proj_conv_snapshot column-parallel";
    require_conv_split_pair(x, ec, kOp);
    require_conv_shard_workspace(workspace, kOp);
    std::array<ConvGeometry, 2> geometry{};
    for (std::size_t slot = 0; slot < 2; ++slot) {
        require_rowsplit(query_key_weight[slot], QType::Q4G64_F16S, kShardQueryKeyRows,
                         "query/key weight shard");
        require_rowsplit(value_z_weight[slot], QType::Q5G64_F16S, kShardValueZRows,
                         "value/z weight shard");
        geometry[slot] = validate_snapshot_shard_rank(
            x[slot], conv_weight[slot], conv_states[slot], valid_columns[slot],
            initial_state_slots[slot], snapshot_base_slots[slot], query[slot], key[slot],
            value[slot], z[slot]);
    }
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        detail::require_rank_residency(
            ec, rank, x[slot].data, query_key_weight[slot].payload, conv_states[slot].data,
            "gdn_input_proj_conv_snapshot column-parallel: every per-rank argument must be "
            "resident on ec.dev[rank]");
    }

    std::array<Tensor, 2> states{conv_states[0], conv_states[1]};
    std::array<Tensor, 2> q_dst{query[0], query[1]};
    std::array<Tensor, 2> k_dst{key[0], key[1]};
    std::array<Tensor, 2> v_dst{value[0], value[1]};
    std::array<Tensor, 2> z_dst{z[0], z[1]};
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot       = static_cast<std::size_t>(rank);
        cudaStream_t stream   = ec.dev[slot]->stream;
        WorkspaceArena& arena = *workspace[slot];
        auto scope            = arena.scope();
        Tensor projected =
            arena.alloc(DType::BF16, {kShardConvChannels, geometry[slot].aggregate_columns});
        compose_shard_conv(x[slot], projected, z_dst[slot], geometry[slot],
                           [&](const Tensor& x_flat, Tensor& out, Tensor& z_flat) {
                               detail::q4_q5_gdn_input_dispatch(x_flat, query_key_weight[slot],
                                                                value_z_weight[slot], out, z_flat,
                                                                arena, stream);
                           });
        Tensor projected_3d(projected.data, DType::BF16,
                            {kShardConvChannels, geometry[slot].width, geometry[slot].batch});
        detail::gdn_projected_conv_snapshot_launch(
            projected_3d, conv_weight[slot], states[slot], valid_columns[slot],
            initial_state_slots[slot], snapshot_base_slots[slot], q_dst[slot], k_dst[slot],
            v_dst[slot], stream);
    });
}

void gdn_input_proj_conv_record_column_parallel(
    const std::array<Tensor, 2>& x, const std::array<Weight, 2>& query_key_value_z_weight,
    const std::array<Tensor, 2>& conv_weight, const std::array<Tensor, 2>& conv_states,
    const std::array<Tensor, 2>& valid_columns, const std::array<Tensor, 2>& initial_state_slots,
    const std::array<Tensor, 2>& conv_record, const std::array<Tensor, 2>& query,
    const std::array<Tensor, 2>& key, const std::array<Tensor, 2>& value,
    const std::array<Tensor, 2>& z, LinearPolicy policy,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec) {
    constexpr const char* kOp = "gdn_input_proj_conv_record column-parallel";
    require_conv_split_pair(x, ec, kOp);
    require_conv_shard_workspace(workspace, kOp);
    std::array<ConvGeometry, 2> geometry{};
    for (std::size_t slot = 0; slot < 2; ++slot) {
        validate_fused_shard_weight(query_key_value_z_weight[slot], policy, kOp);
        geometry[slot] = validate_record_shard_rank(
            x[slot], conv_weight[slot], conv_states[slot], valid_columns[slot],
            initial_state_slots[slot], conv_record[slot], query[slot], key[slot], value[slot],
            z[slot]);
    }
    if (query_key_value_z_weight[0].qtype != query_key_value_z_weight[1].qtype) {
        throw std::invalid_argument(std::string(kOp) +
                                    ": both ranks must carry the same weight format");
    }
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        detail::require_rank_residency(
            ec, rank, x[slot].data, query_key_value_z_weight[slot].payload, conv_record[slot].data,
            "gdn_input_proj_conv_record column-parallel: every per-rank argument must be resident "
            "on ec.dev[rank]");
    }

    std::array<Tensor, 2> record_dst{conv_record[0], conv_record[1]};
    std::array<Tensor, 2> q_dst{query[0], query[1]};
    std::array<Tensor, 2> k_dst{key[0], key[1]};
    std::array<Tensor, 2> v_dst{value[0], value[1]};
    std::array<Tensor, 2> z_dst{z[0], z[1]};
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot       = static_cast<std::size_t>(rank);
        const Weight& w       = query_key_value_z_weight[slot];
        cudaStream_t stream   = ec.dev[slot]->stream;
        WorkspaceArena& arena = *workspace[slot];
        auto scope            = arena.scope();
        Tensor record_flat = flatten_columns(record_dst[slot], kShardConvChannels, geometry[slot]);
        compose_shard_conv(x[slot], record_flat, z_dst[slot], geometry[slot],
                           [&](const Tensor& x_flat, Tensor& out, Tensor& z_flat) {
                               if (w.qtype == QType::GGML_K) {
                                   project_ggml_k(x_flat, w, out, z_flat, stream, &arena);
                               } else if (w.qtype == QType::NVFP4) {
                                   detail::nvfp4_gdn_input_dispatch_shard(x_flat, w, out, z_flat,
                                                                          policy, &arena, stream);
                               } else {
                                   detail::fp8_gdn_input_dispatch_shard(x_flat, w, out, z_flat,
                                                                        policy, &arena, stream);
                               }
                           });
        detail::gdn_projected_conv_record_launch(record_dst[slot], conv_weight[slot],
                                                 conv_states[slot], valid_columns[slot],
                                                 initial_state_slots[slot], q_dst[slot], k_dst[slot],
                                                 v_dst[slot], stream);
    });
}

void gdn_input_proj_conv_record_column_parallel(
    const std::array<Tensor, 2>& x, const std::array<Weight, 2>& query_key_weight,
    const std::array<Weight, 2>& value_z_weight, const std::array<Tensor, 2>& conv_weight,
    const std::array<Tensor, 2>& conv_states, const std::array<Tensor, 2>& valid_columns,
    const std::array<Tensor, 2>& initial_state_slots, const std::array<Tensor, 2>& conv_record,
    const std::array<Tensor, 2>& query, const std::array<Tensor, 2>& key,
    const std::array<Tensor, 2>& value, const std::array<Tensor, 2>& z,
    const std::array<WorkspaceArena*, 2>& workspace, const ExecutionContext& ec) {
    constexpr const char* kOp = "gdn_input_proj_conv_record column-parallel";
    require_conv_split_pair(x, ec, kOp);
    require_conv_shard_workspace(workspace, kOp);
    std::array<ConvGeometry, 2> geometry{};
    for (std::size_t slot = 0; slot < 2; ++slot) {
        require_rowsplit(query_key_weight[slot], QType::Q4G64_F16S, kShardQueryKeyRows,
                         "query/key weight shard");
        require_rowsplit(value_z_weight[slot], QType::Q5G64_F16S, kShardValueZRows,
                         "value/z weight shard");
        geometry[slot] = validate_record_shard_rank(
            x[slot], conv_weight[slot], conv_states[slot], valid_columns[slot],
            initial_state_slots[slot], conv_record[slot], query[slot], key[slot], value[slot],
            z[slot]);
    }
    for (int rank = 0; rank < 2; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        detail::require_rank_residency(
            ec, rank, x[slot].data, query_key_weight[slot].payload, conv_record[slot].data,
            "gdn_input_proj_conv_record column-parallel: every per-rank argument must be resident "
            "on ec.dev[rank]");
    }

    std::array<Tensor, 2> record_dst{conv_record[0], conv_record[1]};
    std::array<Tensor, 2> q_dst{query[0], query[1]};
    std::array<Tensor, 2> k_dst{key[0], key[1]};
    std::array<Tensor, 2> v_dst{value[0], value[1]};
    std::array<Tensor, 2> z_dst{z[0], z[1]};
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot     = static_cast<std::size_t>(rank);
        cudaStream_t stream = ec.dev[slot]->stream;
        Tensor record_flat = flatten_columns(record_dst[slot], kShardConvChannels, geometry[slot]);
        compose_shard_conv(x[slot], record_flat, z_dst[slot], geometry[slot],
                           [&](const Tensor& x_flat, Tensor& out, Tensor& z_flat) {
                               detail::q4_q5_gdn_input_dispatch(x_flat, query_key_weight[slot],
                                                                value_z_weight[slot], out, z_flat,
                                                                *workspace[slot], stream);
                           });
        detail::gdn_projected_conv_record_launch(record_dst[slot], conv_weight[slot],
                                                 conv_states[slot], valid_columns[slot],
                                                 initial_state_slots[slot], q_dst[slot], k_dst[slot],
                                                 v_dst[slot], stream);
    });
}

} // namespace ninfer::ops
