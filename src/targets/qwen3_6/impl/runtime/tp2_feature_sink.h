#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail {

struct TP2FeatureSink {
    using PrefillConsumer = std::function<void(const std::array<Tensor, 2>&,
                                               const std::array<Tensor, 2>&, bool)>;

    std::array<Tensor, 2> features;
    std::array<Tensor, 2> positions;
    std::span<const int> layers;
    PrefillConsumer consume_prefill;
    std::array<std::uint32_t, 2> captured_mask{};
    std::array<bool, 2> captured_positions{};
    std::int32_t active_tokens = 0;

    void begin(const std::array<Tensor, 2>& residual) {
        if (layers.empty() || layers.size() > 32 || residual[0].ne[0] <= 0 ||
            residual[0].ne[0] % 2 != 0 || residual[0].ne[1] <= 0 ||
            !std::is_sorted(layers.begin(), layers.end()) ||
            std::adjacent_find(layers.begin(), layers.end()) != layers.end()) {
            throw std::invalid_argument("TP2 feature sink configuration is invalid");
        }
        const std::int32_t rows = static_cast<std::int32_t>(layers.size()) * residual[0].ne[0];
        active_tokens = residual[0].ne[1];
        for (int rank = 0; rank < 2; ++rank) {
            const Tensor& source = residual[rank];
            const Tensor& target = features[rank];
            const Tensor& position = positions[rank];
            if (source.dtype != DType::BF16 || source.ne[0] != residual[0].ne[0] ||
                source.ne[1] != active_tokens || source.ne[2] != 1 || source.ne[3] != 1 ||
                !source.is_contiguous() || source.data == nullptr || target.dtype != DType::BF16 ||
                target.ne[0] != rows / 2 || target.ne[1] < active_tokens ||
                target.ne[2] != 1 || target.ne[3] != 1 || !target.is_contiguous() ||
                target.data == nullptr || position.dtype != DType::I32 ||
                position.ne[0] < active_tokens || position.ne[1] != 1 ||
                position.ne[2] != 1 || position.ne[3] != 1 || !position.is_contiguous() ||
                position.data == nullptr) {
                throw std::invalid_argument("TP2 feature sink tensor shape is invalid");
            }
        }
        captured_mask = {};
        captured_positions = {};
    }

    void capture_layer(int rank, int layer, const Tensor& residual, cudaStream_t stream) {
        if (rank < 0 || rank > 1 || active_tokens <= 0 || residual.dtype != DType::BF16 ||
            residual.ne[0] * static_cast<std::int32_t>(layers.size()) != 2 * features[0].ne[0] ||
            residual.ne[1] != active_tokens || residual.ne[2] != 1 || residual.ne[3] != 1 ||
            !residual.is_contiguous() || residual.data == nullptr) {
            throw std::invalid_argument("TP2 feature capture source is invalid");
        }
        const auto found = std::find(layers.begin(), layers.end(), layer);
        if (found == layers.end()) return;
        const int index = static_cast<int>(found - layers.begin());
        const int hidden = residual.ne[0];
        const int half = features[rank].ne[0];
        const int begin = std::max(index * hidden, rank * half);
        const int end = std::min((index + 1) * hidden, (rank + 1) * half);
        if (begin < end) {
            const int source_offset = begin - index * hidden;
            const int destination_offset = begin - rank * half;
            auto* target = static_cast<std::byte*>(features[rank].data) +
                           destination_offset * sizeof(std::uint16_t);
            const auto* source = static_cast<const std::byte*>(residual.data) +
                                 source_offset * sizeof(std::uint16_t);
            CUDA_CHECK(cudaMemcpy2DAsync(target, features[rank].nb[1], source, residual.nb[1],
                                         static_cast<std::size_t>(end - begin) *
                                             sizeof(std::uint16_t),
                                         active_tokens, cudaMemcpyDeviceToDevice, stream));
        }
        captured_mask[rank] |= 1U << index;
    }

    void capture_positions(int rank, const Tensor& source, cudaStream_t stream) {
        if (rank < 0 || rank > 1 || source.dtype != DType::I32 || source.ne[0] != active_tokens ||
            source.ne[1] != 1 || source.ne[2] != 1 || source.ne[3] != 1 ||
            !source.is_contiguous() || source.data == nullptr) {
            throw std::invalid_argument("TP2 feature positions are invalid");
        }
        CUDA_CHECK(cudaMemcpyAsync(positions[rank].data, source.data,
                                   static_cast<std::size_t>(active_tokens) * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToDevice, stream));
        captured_positions[rank] = true;
    }

    void require_complete() const {
        const std::uint32_t complete = layers.size() == 32 ? ~0U : (1U << layers.size()) - 1U;
        if (active_tokens <= 0 || captured_mask[0] != complete || captured_mask[1] != complete ||
            !captured_positions[0] || !captured_positions[1]) {
            throw std::logic_error("TP2 feature capture is incomplete");
        }
    }

    void consume_prefill_chunk(bool rewrite_checkpoint) {
        require_complete();
        if (!consume_prefill) { throw std::logic_error("TP2 prefill consumer is unavailable"); }
        const std::array<Tensor, 2> feature_window = {
            features[0].slice(1, 0, active_tokens), features[1].slice(1, 0, active_tokens)};
        const std::array<Tensor, 2> position_window = {
            positions[0].slice(0, 0, active_tokens), positions[1].slice(0, 0, active_tokens)};
        consume_prefill(feature_window, position_window, rewrite_checkpoint);
    }
};

} // namespace ninfer::targets::qwen3_6::detail
