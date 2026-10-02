#include "ninfer/ops/linear_topk.h"

#include "core/layout.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/gguf/gguf.h"
#include "ops/linear_topk/dflash2_linear_topk_volta.h"
#include "ninfer/ops/linear.h"
#include "ops/linear_topk/linear_topk_workspace.h"

#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

enum class HeadProfile : std::uint8_t {
    W8Full,
    Fp8Full,
    Q4Optimized,
};

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool overlaps(const void* lhs, std::size_t lhs_bytes, const void* rhs, std::size_t rhs_bytes) {
    if (lhs == nullptr || rhs == nullptr || lhs_bytes == 0 || rhs_bytes == 0) { return false; }
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs);
    return lhs_begin < rhs_begin + rhs_bytes && rhs_begin < lhs_begin + lhs_bytes;
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    return overlaps(lhs.data, lhs.bytes(), rhs.data, rhs.bytes());
}

HeadProfile resolve_profile(QType qtype, std::int32_t head_rows, std::int32_t input_rows) {
    if (input_rows != detail::kLinearTopKHidden) {
        throw std::invalid_argument("linear_topk: unsupported head profile");
    }
    if (head_rows == detail::kLinearTopKFullRows && qtype == QType::W8G32_F16S) {
        return HeadProfile::W8Full;
    }
    if (head_rows == detail::kLinearTopKFullRows && qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        return HeadProfile::Fp8Full;
    }
    if (head_rows == detail::kLinearTopKOptimizedRows && qtype == QType::Q4G64_F16S) {
        return HeadProfile::Q4Optimized;
    }
    throw std::invalid_argument("linear_topk: unsupported head profile");
}

void require_matrix(const Tensor& tensor, DType dtype, std::int32_t rows, std::int32_t columns,
                    const char* label, std::uintptr_t alignment = 16) {
    if (tensor.dtype != dtype || tensor.ne[0] != rows || tensor.ne[1] != columns ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, alignment)) {
        throw std::invalid_argument(std::string("linear_topk: invalid ") + label);
    }
}

void validate_io(const Tensor& hidden, const Tensor& candidate_ids,
                 const Tensor& candidate_scores) {
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != detail::kLinearTopKHidden ||
        hidden.ne[1] <= 0 || hidden.ne[2] != 1 || hidden.ne[3] != 1 || !hidden.is_contiguous() ||
        !aligned_to(hidden.data, 16)) {
        throw std::invalid_argument("linear_topk: invalid hidden");
    }
    const std::int32_t columns = hidden.ne[1];
    const auto require_output  = [&](const Tensor& tensor, DType dtype, const char* label) {
        if (tensor.dtype != dtype || tensor.ne[0] != detail::kLinearTopK ||
            tensor.ne[1] != columns || tensor.ne[2] != 1 || tensor.ne[3] != 1 ||
            !tensor.is_contiguous() || !aligned_to(tensor.data, 16)) {
            throw std::invalid_argument(std::string("linear_topk: invalid ") + label);
        }
    };
    require_output(candidate_ids, DType::I32, "candidate_ids");
    require_output(candidate_scores, DType::FP32, "candidate_scores");
    if (overlaps(hidden, candidate_ids) || overlaps(hidden, candidate_scores) ||
        overlaps(candidate_ids, candidate_scores)) {
        throw std::invalid_argument("linear_topk: input and outputs must not overlap");
    }
}

void require_w8(const Weight& head) {
    const bool common =
        head.qtype == QType::W8G32_F16S && head.layout == QuantLayout::RowSplit &&
        head.scale_dtype == DType::FP16 && head.group_size == 32 && head.group == 32 &&
        head.ndim == 2 && head.n == detail::kLinearTopKFullRows &&
        head.k == detail::kLinearTopKHidden && head.shape[0] == head.n && head.shape[1] == head.k &&
        head.padded_shape[0] == head.n && head.padded_shape[1] == head.k && head.qhigh == nullptr &&
        head.high_plane_bytes == 0 && aligned_to(head.qdata, 16) && aligned_to(head.scales, 16);
    if (!common) { throw std::invalid_argument("linear_topk: invalid W8 full head"); }
}

void require_q4(const Weight& head) {
    const bool common =
        head.qtype == QType::Q4G64_F16S && head.layout == QuantLayout::RowSplit &&
        head.scale_dtype == DType::FP16 && head.group_size == 64 && head.group == 64 &&
        head.ndim == 2 && head.n == detail::kLinearTopKOptimizedRows &&
        head.k == detail::kLinearTopKHidden && head.shape[0] == head.n && head.shape[1] == head.k &&
        head.padded_shape[0] == head.n && head.padded_shape[1] == head.k && head.qhigh == nullptr &&
        head.high_plane_bytes == 0 && aligned_to(head.qdata, 16) && aligned_to(head.scales, 16);
    if (!common) { throw std::invalid_argument("linear_topk: invalid Q4 optimized head"); }
}

Tensor column_slice(const Tensor& tensor, int first, int columns) {
    return Tensor(static_cast<std::uint8_t*>(tensor.data) +
                      static_cast<std::int64_t>(first) * tensor.nb[1],
                  tensor.dtype, {tensor.ne[0], columns});
}

void execute(const Tensor& hidden, const Weight& head, const Tensor* id_map, Tensor& ids,
             Tensor& scores, WorkspaceArena& workspace, cudaStream_t stream) {
    const auto profile = resolve_profile(head.qtype, head.n, head.k);
    const std::int32_t valid_rows = profile == HeadProfile::Q4Optimized
                                        ? detail::kLinearTopKOptimizedRows
                                        : detail::kLinearTopKFullValidRows;
    // sm_70 port: materialize the head logits with the general linear op, then a single
    // top-16 selection kernel. The BF16 logit scratch is a rounding the fused kernel avoids.
    for (int first = 0; first < hidden.ne[1];) {
        const int columns = std::min(detail::kLinearTopKMaxChunkColumns, hidden.ne[1] - first);
        auto x          = column_slice(hidden, first, columns);
        auto out_ids    = column_slice(ids, first, columns);
        auto out_scores = column_slice(scores, first, columns);
        auto scope      = workspace.scope();
        Tensor logits   = workspace.alloc(DType::BF16, {head.n, columns});
        const int tiles = (valid_rows + detail::kDflash2TopKTileRows - 1) /
                          detail::kDflash2TopKTileRows;
        Tensor partial_ids = workspace.alloc(DType::I32, {16 * tiles, columns});
        Tensor partial_scores = workspace.alloc(DType::FP32, {16 * tiles, columns});
        linear(x, head, logits, stream);
        detail::dflash2_linear_topk16_launch(
            logits, id_map != nullptr ? static_cast<const std::int32_t*>(id_map->data) : nullptr,
            valid_rows, out_ids, out_scores, partial_ids, partial_scores, stream);
        first += columns;
    }
}
} // namespace

std::size_t linear_topk_tp2_workspace_capacity_bytes(QType qtype, std::int32_t local_rows,
                                                     std::int32_t columns) {
    if (columns < 1 || columns > 120 ||
        !((qtype == QType::W8G32_F16S && local_rows == 124160) ||
          (qtype == QType::Q4G64_F16S && local_rows == 65536) ||
          (qtype == QType::GGUF && (local_rows == 124160 || local_rows == 65536)))) {
        throw std::invalid_argument("linear_topk tp2: unsupported head or column count");
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {local_rows, columns});
    (void)layout.alloc(DType::I32, {16, columns});
    (void)layout.alloc(DType::FP32, {16, columns});
    (void)layout.alloc(DType::I32, {16, 2 * columns});
    (void)layout.alloc(DType::FP32, {16, 2 * columns});
    const int tiles = (local_rows + detail::kDflash2TopKTileRows - 1) /
                      detail::kDflash2TopKTileRows;
    (void)layout.alloc(DType::I32, {16 * tiles, columns});
    (void)layout.alloc(DType::FP32, {16 * tiles, columns});
    const auto linear_bytes = linear_workspace_capacity_bytes(
        qtype, local_rows, 5120, LinearPolicy::A16Only, columns, columns);
    if (linear_bytes != 0) { (void)layout.alloc_bytes(linear_bytes); }
    return layout.peak_bytes(256);
}

void linear_topk_tp2(const std::array<Tensor, 2>& hidden, const std::array<Weight, 2>& head,
                     const std::array<Tensor, 2>* id_map, Tensor& candidate_ids,
                     Tensor& candidate_scores, const std::array<WorkspaceArena*, 2>& workspace,
                     const ExecutionContext& execution, const PeerEvents& events) {
    if (execution.tp != 2 || !execution.dev[0] || !execution.dev[1] ||
        execution.dev[0]->device == execution.dev[1]->device ||
        !workspace[0] || !workspace[1] || head[0].qtype != head[1].qtype ||
        hidden[0].ne[1] != hidden[1].ne[1]) {
        throw std::invalid_argument("linear_topk tp2: invalid ranks");
    }
    const bool full = head[0].n == 124160;
    const int local_rows = full ? 124160 : 65536;
    const int columns = hidden[0].ne[1];
    const std::size_t workspace_bytes =
        linear_topk_tp2_workspace_capacity_bytes(head[0].qtype, local_rows, columns);
    require_matrix(candidate_ids, DType::I32, 16, columns, "tp2 ids");
    require_matrix(candidate_scores, DType::FP32, 16, columns, "tp2 scores");
    if (overlaps(candidate_ids, candidate_scores)) {
        throw std::invalid_argument("linear_topk tp2: candidate outputs overlap");
    }
    if (full ? id_map != nullptr : id_map == nullptr) {
        throw std::invalid_argument("linear_topk tp2: id map does not match head");
    }
    for (int rank = 0; rank < 2; ++rank) {
        require_matrix(hidden[rank], DType::BF16, 5120, columns, "tp2 hidden");
        const Weight& shard = head[rank];
        const auto* scratch_begin =
            static_cast<const std::byte*>(workspace[rank]->base()) + workspace[rank]->used();
        if (shard.qtype == QType::GGUF) {
            detail::validate_gguf_weight(shard, "linear_topk tp2 GGUF shard");
        }
        if (shard.n != local_rows || shard.k != 5120 || shard.ndim != 2 ||
            shard.shape[0] != local_rows || shard.shape[1] != 5120 ||
            shard.padded_shape[0] != local_rows || shard.padded_shape[1] != 5120 ||
            (shard.qtype != QType::GGUF &&
             (shard.layout != QuantLayout::RowSplit || shard.scale_dtype != DType::FP16 ||
              shard.group_size != (full ? 32 : 64) || shard.group != (full ? 32 : 64) ||
              shard.qhigh != nullptr || shard.high_plane_bytes != 0 ||
              !aligned_to(shard.qdata, 16) || !aligned_to(shard.scales, 16))) ||
            workspace[rank]->capacity() - workspace[rank]->used() < workspace_bytes ||
            (id_map && (id_map->at(rank).dtype != DType::I32 ||
                        id_map->at(rank).ne[0] != 131072 ||
                        id_map->at(rank).ne[1] != 1 || id_map->at(rank).ne[2] != 1 ||
                        id_map->at(rank).ne[3] != 1 ||
                        !aligned_to(id_map->at(rank).data, 16) ||
                        !id_map->at(rank).is_contiguous())) ||
            overlaps(hidden[rank], candidate_ids) || overlaps(hidden[rank], candidate_scores) ||
            overlaps(scratch_begin, workspace_bytes, hidden[rank].data, hidden[rank].bytes()) ||
            (id_map && overlaps(scratch_begin, workspace_bytes, id_map->at(rank).data,
                                id_map->at(rank).bytes())) ||
            (shard.payload &&
             (overlaps(scratch_begin, workspace_bytes, shard.payload, shard.payload_bytes) ||
              overlaps(shard.payload, shard.payload_bytes, hidden[rank].data,
                       hidden[rank].bytes()) ||
              (rank == 0 &&
               (overlaps(shard.payload, shard.payload_bytes, candidate_ids.data,
                         candidate_ids.bytes()) ||
                overlaps(shard.payload, shard.payload_bytes, candidate_scores.data,
                         candidate_scores.bytes()))))) ||
            (rank == 0 &&
             (overlaps(scratch_begin, workspace_bytes, candidate_ids.data,
                       candidate_ids.bytes()) ||
              overlaps(scratch_begin, workspace_bytes, candidate_scores.data,
                       candidate_scores.bytes())))) {
            throw std::invalid_argument("linear_topk tp2: invalid shard or workspace");
        }
    }

    auto first_scope = workspace[0]->scope();
    auto second_scope = workspace[1]->scope();
    std::array<Tensor, 2> local_ids, local_scores, gathered_ids, gathered_scores;
    int previous_device = 0;
    CUDA_CHECK(cudaGetDevice(&previous_device));
    try {
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
            WorkspaceArena& scratch = *workspace[rank];
            Tensor logits = scratch.alloc(DType::BF16, {local_rows, columns});
            local_ids[rank] = scratch.alloc(DType::I32, {16, columns});
            local_scores[rank] = scratch.alloc(DType::FP32, {16, columns});
            gathered_ids[rank] = scratch.alloc(DType::I32, {16, 2 * columns});
            gathered_scores[rank] = scratch.alloc(DType::FP32, {16, 2 * columns});
            const int count = full ? (rank == 0 ? local_rows : 248077 - local_rows)
                                   : local_rows;
            const int tiles = (count + detail::kDflash2TopKTileRows - 1) /
                              detail::kDflash2TopKTileRows;
            Tensor partial_ids = scratch.alloc(DType::I32, {16 * tiles, columns});
            Tensor partial_scores = scratch.alloc(DType::FP32, {16 * tiles, columns});
            linear(hidden[rank], head[rank], logits, LinearPolicy::A16Only, scratch,
                   execution.dev[rank]->stream);
            const auto* map = full ? nullptr :
                static_cast<const std::int32_t*>(id_map->at(rank).data) + rank * local_rows;
            detail::dflash2_linear_topk16_launch(logits, map, count, local_ids[rank],
                                                 local_scores[rank], partial_ids, partial_scores,
                                                 execution.dev[rank]->stream,
                                                 rank * local_rows);
        }
        allgather_rows(gathered_ids, local_ids, execution, events);
        allgather_rows(gathered_scores, local_scores, execution, events);
        CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
        detail::dflash2_merge_tp2_topk16_launch(gathered_ids[0], gathered_scores[0],
                                                 candidate_ids, candidate_scores,
                                                 execution.dev[0]->stream);
    } catch (...) {
        (void)cudaSetDevice(previous_device);
        throw;
    }
    CUDA_CHECK(cudaSetDevice(previous_device));
}

std::size_t linear_topk_workspace_capacity_bytes(QType qtype, std::int32_t head_rows,
                                                 std::int32_t input_rows, std::int32_t min_columns,
                                                 std::int32_t max_columns) {
    (void)resolve_profile(qtype, head_rows, input_rows);
    if (min_columns < 1 || max_columns < min_columns) {
        throw std::invalid_argument("linear_topk workspace: invalid column interval");
    }
    const std::int32_t chunk = std::min(max_columns, detail::kLinearTopKMaxChunkColumns);
    const int valid_rows = head_rows == detail::kLinearTopKOptimizedRows
                               ? head_rows : detail::kLinearTopKFullValidRows;
    const int tiles = (valid_rows + detail::kDflash2TopKTileRows - 1) /
                      detail::kDflash2TopKTileRows;
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {head_rows, chunk});
    (void)layout.alloc(DType::I32, {16 * tiles, chunk});
    (void)layout.alloc(DType::FP32, {16 * tiles, chunk});
    return layout.peak_bytes(256);
}

void linear_topk(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                 Tensor& candidate_ids, Tensor& candidate_scores, WorkspaceArena& workspace,
                 cudaStream_t stream) {
    validate_io(hidden, candidate_ids, candidate_scores);
    const HeadProfile profile = resolve_profile(head.qtype, head.n, head.k);
    if (profile == HeadProfile::Q4Optimized || valid_rows != detail::kLinearTopKFullValidRows) {
        throw std::invalid_argument("linear_topk: invalid full-head profile or valid_rows");
    }
    if (profile == HeadProfile::W8Full) {
        require_w8(head);
    } else {
        (void)detail::validate_fp8_weight(head, "linear_topk FP8 full head");
    }

    execute(hidden, head, nullptr, candidate_ids, candidate_scores, workspace, stream);
}

void linear_topk(const Tensor& hidden, const Weight& head, const Tensor& row_to_global_ids,
                 Tensor& candidate_ids, Tensor& candidate_scores, WorkspaceArena& workspace,
                 cudaStream_t stream) {
    validate_io(hidden, candidate_ids, candidate_scores);
    if (resolve_profile(head.qtype, head.n, head.k) != HeadProfile::Q4Optimized) {
        throw std::invalid_argument("linear_topk: invalid optimized-head profile");
    }
    require_q4(head);
    require_matrix(row_to_global_ids, DType::I32, detail::kLinearTopKOptimizedRows, 1,
                   "row_to_global_ids", 4);
    if (overlaps(hidden, row_to_global_ids) || overlaps(candidate_ids, row_to_global_ids) ||
        overlaps(candidate_scores, row_to_global_ids)) {
        throw std::invalid_argument("linear_topk: id map overlaps input or output");
    }

    execute(hidden, head, &row_to_global_ids, candidate_ids, candidate_scores, workspace, stream);
}

} // namespace ninfer::ops
