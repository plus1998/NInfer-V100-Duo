#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_replay.h"

#include "ops/linear_attention/gated_delta_net/common.h"
#include "ops/linear_attention/gated_delta_net/launch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kStateDim    = detail::gated_delta_net::kStateDim;
constexpr std::int32_t kMaximumRows = 8;

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_tensor(const Tensor& tensor, DType dtype, std::initializer_list<std::int32_t> shape,
                    std::uintptr_t alignment, const char* op, const char* label) {
    const Tensor expected(nullptr, dtype, shape);
    if (tensor.data == nullptr || tensor.dtype != dtype || !tensor.is_contiguous() ||
        tensor.bytes() != expected.bytes() || !aligned_to(tensor.data, alignment)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + label);
    }
    for (int dim = 0; dim < 4; ++dim) {
        if (tensor.ne[dim] != expected.ne[dim]) {
            throw std::invalid_argument(std::string(op) + ": invalid " + label);
        }
    }
}

struct MemoryRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};

MemoryRange make_range(const void* pointer, std::size_t bytes, const char* label) {
    if (pointer == nullptr || bytes == 0) {
        throw std::invalid_argument(std::string(label) + " has an empty memory range");
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::overflow_error(std::string(label) + " address range overflows");
    }
    return {begin, begin + bytes};
}

MemoryRange tensor_range(const Tensor& tensor, const char* label) {
    return make_range(tensor.data, tensor.bytes(), label);
}

MemoryRange layer_range(const Tensor& layer0, std::int64_t stride_bytes, std::int32_t layer,
                        const char* label) {
    const auto base    = reinterpret_cast<std::uintptr_t>(layer0.data);
    const auto stride  = static_cast<std::uint64_t>(stride_bytes);
    const auto layer_u = static_cast<std::uint64_t>(layer);
    if (layer_u != 0 && stride > std::numeric_limits<std::uintptr_t>::max() / layer_u) {
        throw std::overflow_error(std::string(label) + " layer offset overflows");
    }
    const auto offset = static_cast<std::uintptr_t>(stride * layer_u);
    if (offset > std::numeric_limits<std::uintptr_t>::max() - base) {
        throw std::overflow_error(std::string(label) + " layer address overflows");
    }
    return make_range(reinterpret_cast<const void*>(base + offset), layer0.bytes(), label);
}

bool overlaps(MemoryRange lhs, MemoryRange rhs) {
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

template <class Ranges>
void require_pairwise_disjoint(const Ranges& ranges, const char* message) {
    for (std::size_t lhs = 0; lhs < ranges.size(); ++lhs) {
        for (std::size_t rhs = lhs + 1; rhs < ranges.size(); ++rhs) {
            if (overlaps(ranges[lhs], ranges[rhs])) { throw std::invalid_argument(message); }
        }
    }
}

void require_scale(float scale, const char* op) {
    const float expected = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    if (!std::isfinite(scale) || scale <= 0.0f || std::abs(scale - expected) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(128)");
    }
}

void validate_replay_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, const Tensor& states,
                            const Tensor& valid_columns, const Tensor& initial_slots,
                            const Tensor& key_record, const Tensor& value_record,
                            const Tensor& gate_record, const Tensor& out) {
    constexpr const char* kOp      = "gated_delta_net_replay_record";
    const std::int32_t qk_heads    = q.ne[1];
    const std::int32_t value_heads = v.ne[1];
    const std::int32_t width       = q.ne[2];
    const std::int32_t rows        = q.ne[3];
    // Registered head geometries. The launcher is entirely runtime-dimensioned in the head
    // counts (`launch_recurrent_record_fixed` takes grid.x = v.ne[1] and builds its value->qk map
    // with `head_map::of(q.ne[1], v.ne[1])` at run time), so a geometry is registered by being
    // QUALIFIED, not by adding a kernel. `8|24` is the tp == 2 shard of `16|48`: both the qk-head
    // and the value-head counts halve, the group size 48/16 == 24/8 == 3 is preserved, and the
    // split is group-aligned so no group of three value heads is ever cut. It is the geometry the
    // MTP/speculative verify round records on each device at tp == 2, and it is the exact partner
    // of FoldGeometry48x24Tp2 in `is_registered_fold_geometry` below -- a record written at this
    // geometry is folded at that one, so the two entries must exist together.
    const bool registered_heads = (qk_heads == 16 && (value_heads == 48 || value_heads == 32)) ||
                                  (qk_heads == 8 && value_heads == 24);
    if (!registered_heads || width < 2 || width > 16 || rows <= 0 || rows > kMaximumRows) {
        throw std::invalid_argument(std::string(kOp) + ": unsupported geometry");
    }
    if (states.ne[3] <= 0) {
        throw std::invalid_argument(std::string(kOp) + ": state slot count must be positive");
    }

    require_tensor(q, DType::BF16, {kStateDim, qk_heads, width, rows}, 8, kOp, "q");
    require_tensor(k, DType::BF16, {kStateDim, qk_heads, width, rows}, 8, kOp, "k");
    require_tensor(v, DType::BF16, {kStateDim, value_heads, width, rows}, 2, kOp, "v");
    require_tensor(g, DType::FP32, {value_heads, width, rows}, 4, kOp, "g");
    require_tensor(beta, DType::FP32, {value_heads, width, rows}, 4, kOp, "beta");
    require_tensor(states, DType::FP32, {kStateDim, kStateDim, value_heads, states.ne[3]}, 16, kOp,
                   "states");
    require_tensor(initial_slots, DType::I32, {rows}, 4, kOp, "initial state slots");
    if (valid_columns.data != nullptr) {
        require_tensor(valid_columns, DType::I32, {rows}, 4, kOp, "valid columns");
    }
    require_tensor(key_record, DType::BF16, {kStateDim, qk_heads, width, rows}, 8, kOp,
                   "key record");
    require_tensor(value_record, DType::BF16, {kStateDim, value_heads, width, rows}, 2, kOp,
                   "value record");
    require_tensor(gate_record, DType::FP32, {2, value_heads, width, rows}, 8, kOp, "gate record");
    require_tensor(out, DType::BF16, {kStateDim, value_heads, width, rows}, 2, kOp, "out");
    require_scale(scale, kOp);

    const std::array<const Tensor*, 12> tensors{&q,
                                                &k,
                                                &v,
                                                &g,
                                                &beta,
                                                &states,
                                                &valid_columns,
                                                &initial_slots,
                                                &key_record,
                                                &value_record,
                                                &gate_record,
                                                &out};
    std::vector<MemoryRange> ranges;
    ranges.reserve(tensors.size());
    for (const Tensor* tensor : tensors) {
        if (tensor->data != nullptr) { ranges.push_back(tensor_range(*tensor, kOp)); }
    }
    require_pairwise_disjoint(ranges, "gated_delta_net_replay_record: tensors must not overlap");
}

// Host-side mirror of NINFER_GDN_FOLD_GEOMETRIES in
// src/ops/linear_attention/gated_delta_net/recurrent.cuh. This TU is plain C++ and this repository
// keeps `.cuh` device-side (included only from `.cu`), so the literals are duplicated here the same
// way `kv_heads_for_q_heads` mirrors the GQA registry in src/ops/wrapper/gqa_attention.cpp. The
// failure mode is safe in one direction only: a MISSING entry makes the Op reject a geometry the
// launcher would have run, while an entry here with no launcher arm would reach
// `launch_replay_fold`'s trailing throw. Keep the two lists edited together.
bool is_registered_fold_geometry(const GdnReplayRecordSpec& spec) {
    const bool geometry_48 = spec.layers == 48 && spec.qk_heads == 16 && spec.value_heads == 48 &&
                             spec.conv_channels == 10240;
    const bool geometry_30 = spec.layers == 30 && spec.qk_heads == 16 && spec.value_heads == 32 &&
                             spec.conv_channels == 8192;
    // FoldGeometry48x24Tp2: device r's half of geometry_48 at tp == 2. Both the qk-head and the
    // value-head counts halve, together with the conv channel count.
    const bool geometry_48_tp2 = spec.layers == 48 && spec.qk_heads == 8 &&
                                 spec.value_heads == 24 && spec.conv_channels == 5120;
    return geometry_48 || geometry_30 || geometry_48_tp2;
}

void validate_fold_records(const GdnReplayRecords& records) {
    constexpr const char* kOp       = "gdn_replay_fold";
    const GdnReplayRecordSpec& spec = records.spec;
    if (!is_registered_fold_geometry(spec) || spec.record_capacity <= 0 ||
        spec.record_capacity > kMaximumRows || spec.width < 2 || spec.width > 16 ||
        spec.key_dim != kStateDim || spec.value_dim != kStateDim) {
        throw std::invalid_argument(std::string(kOp) + ": unsupported record geometry");
    }
    if (spec.layers > std::numeric_limits<std::int32_t>::max() / spec.record_capacity) {
        throw std::overflow_error(std::string(kOp) + ": record outer extent overflows");
    }
    const std::int32_t outer = spec.layers * spec.record_capacity;
    require_tensor(records.conv, DType::BF16, {spec.conv_channels, spec.width, outer}, 256, kOp,
                   "conv records");
    require_tensor(records.key, DType::BF16, {kStateDim, spec.qk_heads, spec.width, outer}, 256,
                   kOp, "key records");
    require_tensor(records.value, DType::BF16, {kStateDim, spec.value_heads, spec.width, outer},
                   256, kOp, "value records");
    require_tensor(records.gate, DType::FP32, {2, spec.value_heads, spec.width, outer}, 256, kOp,
                   "gate records");

    const std::array<MemoryRange, 4> record_ranges{
        tensor_range(records.conv, kOp), tensor_range(records.key, kOp),
        tensor_range(records.value, kOp), tensor_range(records.gate, kOp)};
    require_pairwise_disjoint(record_ranges, "gdn_replay_fold: record planes overlap");
}

void validate_fold_states(const GdnReplayRecords& records,
                          LinearAttentionStateAllLayersView states) {
    constexpr const char* kOp = "gdn_replay_fold";
    const auto& record        = records.spec;
    const auto& state         = states.spec;
    if (state.layers != static_cast<std::uint32_t>(record.layers) ||
        state.conv_channels != record.conv_channels || state.conv_width != 3 ||
        state.value_heads != record.value_heads || state.value_head_dim != kStateDim ||
        state.key_head_dim != kStateDim || state.slot_count <= 0 ||
        state.conv_dtype != DType::BF16) {
        throw std::invalid_argument(std::string(kOp) + ": state geometry does not match records");
    }
    require_tensor(states.conv_layer0, DType::BF16, {record.conv_channels, 3, state.slot_count},
                   256, kOp, "conv state layer 0");
    require_tensor(states.recurrent_layer0, DType::FP32,
                   {kStateDim, kStateDim, record.value_heads, state.slot_count}, 256, kOp,
                   "recurrent state layer 0");
    if (states.conv_layer_stride_bytes <= 0 || states.recurrent_layer_stride_bytes <= 0 ||
        states.conv_layer_stride_bytes % static_cast<std::int64_t>(sizeof(std::uint16_t)) != 0 ||
        states.recurrent_layer_stride_bytes % static_cast<std::int64_t>(sizeof(float)) != 0 ||
        static_cast<std::uint64_t>(states.conv_layer_stride_bytes) < states.conv_layer0.bytes() ||
        static_cast<std::uint64_t>(states.recurrent_layer_stride_bytes) <
            states.recurrent_layer0.bytes()) {
        throw std::invalid_argument(std::string(kOp) + ": invalid state layer stride");
    }

    for (std::int32_t layer = 0; layer < record.layers; ++layer) {
        (void)layer_range(states.conv_layer0, states.conv_layer_stride_bytes, layer,
                          "gdn_replay_fold conv state");
        (void)layer_range(states.recurrent_layer0, states.recurrent_layer_stride_bytes, layer,
                          "gdn_replay_fold recurrent state");
    }
    std::int32_t conv_layer      = 0;
    std::int32_t recurrent_layer = 0;
    while (conv_layer < record.layers && recurrent_layer < record.layers) {
        const MemoryRange conv = layer_range(states.conv_layer0, states.conv_layer_stride_bytes,
                                             conv_layer, "gdn_replay_fold conv state");
        const MemoryRange recurrent =
            layer_range(states.recurrent_layer0, states.recurrent_layer_stride_bytes,
                        recurrent_layer, "gdn_replay_fold recurrent state");
        if (overlaps(conv, recurrent)) {
            throw std::invalid_argument("gdn_replay_fold: state layer regions overlap");
        }
        if (conv.end <= recurrent.begin) {
            ++conv_layer;
        } else {
            ++recurrent_layer;
        }
    }
}

void require_records_disjoint_from_states(const GdnReplayRecords& records,
                                          LinearAttentionStateAllLayersView states) {
    const std::array<MemoryRange, 4> record_ranges{
        tensor_range(records.conv, "gdn_replay_fold conv records"),
        tensor_range(records.key, "gdn_replay_fold key records"),
        tensor_range(records.value, "gdn_replay_fold value records"),
        tensor_range(records.gate, "gdn_replay_fold gate records")};
    for (const MemoryRange record : record_ranges) {
        for (std::int32_t layer = 0; layer < records.spec.layers; ++layer) {
            const MemoryRange conv = layer_range(states.conv_layer0, states.conv_layer_stride_bytes,
                                                 layer, "gdn_replay_fold conv state");
            const MemoryRange recurrent =
                layer_range(states.recurrent_layer0, states.recurrent_layer_stride_bytes, layer,
                            "gdn_replay_fold recurrent state");
            if (overlaps(record, conv) || overlaps(record, recurrent)) {
                throw std::invalid_argument("gdn_replay_fold: records overlap state storage");
            }
        }
    }
}

detail::gated_delta_net::GdnReplayFoldKernelRows
validate_fold_rows(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                   std::span<const GdnReplayFoldRow> rows) {
    if (rows.empty() || rows.size() > static_cast<std::size_t>(records.spec.record_capacity)) {
        throw std::invalid_argument("gdn_replay_fold: active row count is out of range");
    }
    detail::gated_delta_net::GdnReplayFoldKernelRows packed{};
    for (std::size_t row = 0; row < rows.size(); ++row) {
        if (rows[row].linear_state_slot < 0 ||
            rows[row].linear_state_slot >= states.spec.slot_count) {
            throw std::invalid_argument("gdn_replay_fold: linear state slot is out of range");
        }
        if (rows[row].commit_columns < 0 || rows[row].commit_columns > records.spec.width) {
            throw std::invalid_argument("gdn_replay_fold: commit extent is out of range");
        }
        for (std::size_t previous = 0; previous < row; ++previous) {
            if (rows[previous].linear_state_slot == rows[row].linear_state_slot) {
                throw std::invalid_argument("gdn_replay_fold: active state slots must be distinct");
            }
        }
        packed.row[row] = {rows[row].linear_state_slot, rows[row].commit_columns};
    }
    return packed;
}

} // namespace

void gated_delta_net_replay_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& ssm_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, Tensor& key_record,
                                   Tensor& value_record, Tensor& gate_record, Tensor& out,
                                   cudaStream_t stream) {
    validate_replay_record(q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots,
                           key_record, value_record, gate_record, out);
    detail::gated_delta_net::launch_recurrent_record(q, k, v, g, beta, scale, ssm_states,
                                                     valid_columns, initial_state_slots, key_record,
                                                     value_record, gate_record, out, stream);
}

void gdn_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                     std::span<const GdnReplayFoldRow> rows, cudaStream_t stream) {
    validate_fold_records(records);
    validate_fold_states(records, states);
    require_records_disjoint_from_states(records, states);
    const detail::gated_delta_net::GdnReplayFoldKernelRows packed =
        validate_fold_rows(records, states, rows);
    detail::gated_delta_net::launch_replay_fold(records, states, packed,
                                                static_cast<std::int32_t>(rows.size()), stream);
}

} // namespace ninfer::ops
