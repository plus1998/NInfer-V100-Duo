#include "ninfer/ops/context_kv_materialize.h"

#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ops/common/dflash2_rmsnorm_rope_volta.h"
#include "ops/context_kv_materialize/append_volta.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHidden     = 5120;
constexpr std::int32_t kKVSize     = 1024;
constexpr std::int32_t kHeadDim    = 128;
constexpr std::int32_t kKVHeads    = 8;
constexpr std::int32_t kCapacity   = 2048;
constexpr std::int32_t kBlockWidth = 16;
constexpr const char* kOp          = "context_kv_materialize";

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_tensor(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1,
                    std::int32_t n2, std::int32_t n3, std::size_t alignment, const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 ||
        tensor.ne[3] != n3 || !tensor.is_contiguous() || !aligned_to(tensor.data, alignment)) {
        throw std::invalid_argument(std::string(kOp) + ": invalid " + name);
    }
}

void require_weight(const Weight& weight, std::int32_t kv_size, const char* name) {
    const std::uint64_t kCodeBytes =
        static_cast<std::uint64_t>(kv_size) * static_cast<std::uint64_t>(kHidden);
    const std::uint64_t kScaleBytes =
        static_cast<std::uint64_t>(kv_size) * static_cast<std::uint64_t>(kHidden / 32) * 2U;
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group != 32 || weight.group_size != 32 ||
        weight.ndim != 2 || weight.n != kv_size || weight.k != kHidden ||
        weight.shape[0] != kv_size || weight.shape[1] != kHidden ||
        weight.padded_shape[0] != kv_size || weight.padded_shape[1] != kHidden ||
        weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        weight.payload_bytes < kCodeBytes + kScaleBytes || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 4)) {
        throw std::invalid_argument(std::string(kOp) + ": invalid " + name);
    }
}

void require_profile(std::int32_t width, std::int32_t batch) {
    if (batch < 1 || batch > 8 || width < 1) {
        throw std::invalid_argument("context_kv_materialize: invalid W/B");
    }
    if (width <= kBlockWidth) return;
    if (batch == 1 && width <= kCapacity) return;
    throw std::invalid_argument("context_kv_materialize: unsupported W/B profile");
}

void require_interval(std::int32_t batch, std::int32_t min_width, std::int32_t max_width) {
    if (batch < 1 || batch > 8 || min_width < 1 || max_width < min_width) {
        throw std::invalid_argument("context_kv_materialize workspace: invalid interval");
    }
    if (batch == 1) {
        if (max_width > kCapacity) {
            throw std::invalid_argument("context_kv_materialize workspace: W exceeds 2048");
        }
        return;
    }
    if (max_width > kBlockWidth) {
        throw std::invalid_argument("context_kv_materialize workspace: batched W exceeds 16");
    }
}

void validate_cache(const CyclicKVCacheLayerView& cache, std::int32_t padded,
                    std::int32_t lane_capacity, std::int32_t heads) {
    if (cache.capacity != kCapacity ||
        cache.padded_capacity != static_cast<std::uint32_t>(padded) ||
        cache.num_kv_heads != heads || cache.head_dim != kHeadDim ||
        cache.lane_capacity != lane_capacity || padded < kCapacity || lane_capacity <= 0) {
        throw std::invalid_argument("context_kv_materialize: invalid cyclic cache geometry");
    }
    require_tensor(cache.k, DType::BF16, kHeadDim, padded, heads, lane_capacity, 16, "cache K");
    require_tensor(cache.v, DType::FP16, kHeadDim, padded, heads, lane_capacity, 16, "cache V");
}

} // namespace

template <class Allocator>
void allocate_layer_scratch(Allocator& allocator, std::int32_t columns) {
    (void)allocator.alloc(DType::BF16, {kKVSize, columns}); // key (norm+rope in place)
    (void)allocator.alloc(DType::BF16, {kKVSize, columns}); // value
}

std::size_t context_kv_materialize_workspace_capacity_bytes(std::int32_t batch_size,
                                                            std::int32_t min_width,
                                                            std::int32_t max_width) {
    require_interval(batch_size, min_width, max_width);
    // sm_70 port: composed from general W8 linear + head rmsnorm + rope + cyclic append, one
    // layer at a time under a scoped sub-arena, so the peak is a single layer's two BF16
    // [1024, max_width*B] scratch tensors.
    WorkspaceLayoutBuilder layout;
    allocate_layer_scratch(layout, max_width * batch_size);
    return layout.peak_bytes(1);
}

void context_kv_materialize(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    ContextKVMaterializeExecutionEnvelope envelope, WorkspaceArena& workspace,
    cudaStream_t stream) {
    const std::int32_t width = context.ne[1];
    const std::int32_t batch = context.ne[2];
    require_profile(width, batch);
    require_tensor(context, DType::BF16, kHidden, width, batch, 1, 16, "context");
    require_tensor(positions, DType::I32, width, batch, 1, 1, alignof(std::int32_t), "positions");
    require_tensor(counts, DType::I32, batch, 1, 1, 1, alignof(std::int32_t), "counts");
    require_tensor(state_slots, DType::I32, batch, 1, 1, 1, alignof(std::int32_t), "state slots");
    if (envelope.min_count > envelope.max_count ||
        envelope.max_count > static_cast<std::uint32_t>(width) ||
        envelope.max_count > static_cast<std::uint32_t>(kCapacity)) {
        throw std::invalid_argument("context_kv_materialize: invalid execution envelope");
    }

    if (layers.front().cache.padded_capacity >
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("context_kv_materialize: padded capacity exceeds int32");
    }
    const std::int32_t padded = static_cast<std::int32_t>(layers.front().cache.padded_capacity);
    const std::int32_t lane_capacity = layers.front().cache.lane_capacity;
    const std::int32_t heads = layers.front().cache.num_kv_heads;
    if (heads != kKVHeads && heads != kKVHeads / 2) {
        throw std::invalid_argument("context_kv_materialize: KV heads must be 4 or 8");
    }
    const std::int32_t kv_size = heads * kHeadDim;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const ContextKVMaterializeLayerView& layer = layers[index];
        require_weight(layer.key_weight, kv_size, "key weight");
        require_weight(layer.value_weight, kv_size, "value weight");
        require_tensor(layer.key_norm_weight, DType::BF16, kHeadDim, 1, 1, 1, 4, "key norm weight");
        validate_cache(layer.cache, padded, lane_capacity, heads);
    }

    if (envelope.max_count == 0) return;

    // sm_70 port: no fused kernel yet. Compose the five layers from the general W8 projection,
    // per-head RMSNorm, 1-D RoPE and the DFlash2 cyclic append -- exactly the unfused reference
    // in the header's formula. One layer at a time keeps the workspace to a single layer.
    const std::int32_t width_columns = width * batch;
    const Tensor context_flat  = context.view({kHidden, width_columns});
    const Tensor positions_flat = positions.view({width_columns});

    for (const ContextKVMaterializeLayerView& layer : layers) {
        auto layer_scope = workspace.scope();
        Tensor key   = workspace.alloc(DType::BF16, {kv_size, width_columns});
        Tensor value = workspace.alloc(DType::BF16, {kv_size, width_columns});

        linear(context_flat, layer.key_weight, key, stream);
        linear(context_flat, layer.value_weight, value, stream);

        Tensor key_heads = key.view({kHeadDim, heads, width_columns});
        detail::dflash2_rmsnorm_rope_launch(key_heads, layer.key_norm_weight, positions_flat,
                                            1.0e-6F, 1.0e7F, stream);

        Tensor key_batch   = key.view({kHeadDim, heads, width, batch});
        Tensor value_batch = value.view({kHeadDim, heads, width, batch});
        detail::dflash2_context_append_launch(key_batch, value_batch, positions, counts,
                                              state_slots, layer.cache, envelope.max_count, stream);
    }
}

} // namespace ninfer::ops
