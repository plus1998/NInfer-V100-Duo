#include "targets/qwen3_6/impl/runtime/tp2_feature_sink.h"

#include "core/decode_graph.h"
#include "ops/op_tester.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using ninfer::targets::qwen3_6::detail::TP2FeatureSink;

namespace {

constexpr int kHidden = 5120;
constexpr int kHalf = 12800;
constexpr int kCapacity = 11;
constexpr std::array<int, 5> kLayers{5, 19, 33, 47, 61};

struct RankBuffers {
    std::array<DeviceBuffer, kLayers.size()> source;
    DeviceBuffer features;
    DeviceBuffer positions;
    DeviceBuffer source_positions;
};

int run_case(int tokens) {
    std::array<RankBuffers, 2> rank_buffers;
    std::array<Tensor, 2> initial_residual;
    std::array<std::array<Tensor, kLayers.size()>, 2> sources;
    std::array<Tensor, 2> features;
    std::array<Tensor, 2> positions;
    std::array<Tensor, 2> source_positions;
    std::array<std::vector<std::uint16_t>, 2> expected;
    std::array<std::vector<std::int32_t>, 2> expected_positions;
    std::array<cudaStream_t, 2> streams{};

    for (int rank = 0; rank < 2; ++rank) {
        cuda_check(cudaSetDevice(rank), "feature test device");
        cuda_check(cudaStreamCreateWithFlags(&streams[rank], cudaStreamNonBlocking),
                   "feature test stream");
        auto& buffers = rank_buffers[rank];
        expected[rank].assign(static_cast<std::size_t>(kHalf) * tokens, 0);
        for (int index = 0; index < static_cast<int>(kLayers.size()); ++index) {
            std::vector<std::uint16_t> host(static_cast<std::size_t>(kHidden) * tokens);
            for (int column = 0; column < tokens; ++column) {
                for (int row = 0; row < kHidden; ++row) {
                    const auto value = static_cast<std::uint16_t>(0x3b80 + rank * 1024 +
                                                                   index * 170 +
                                                                   (row * 7 + column * 13) % 113);
                    host[static_cast<std::size_t>(column) * kHidden + row] = value;
                    const int global_row = index * kHidden + row;
                    if (global_row >= rank * kHalf && global_row < (rank + 1) * kHalf) {
                        expected[rank][static_cast<std::size_t>(column) * kHalf +
                                       global_row - rank * kHalf] = value;
                    }
                }
            }
            buffers.source[index] = to_device(host);
            sources[rank][index] =
                Tensor(buffers.source[index].p, DType::BF16, {kHidden, tokens});
        }
        buffers.features = DeviceBuffer(static_cast<std::size_t>(kHalf) * kCapacity * 2);
        buffers.features.fill(0xa5);
        features[rank] = Tensor(buffers.features.p, DType::BF16, {kHalf, kCapacity});
        expected_positions[rank].resize(tokens);
        for (int column = 0; column < tokens; ++column)
            expected_positions[rank][column] = 65536 + column;
        buffers.source_positions = to_device(expected_positions[rank]);
        buffers.positions = DeviceBuffer(kCapacity * sizeof(std::int32_t));
        buffers.positions.fill(0xa5);
        source_positions[rank] = Tensor(buffers.source_positions.p, DType::I32, {tokens});
        positions[rank] = Tensor(buffers.positions.p, DType::I32, {kCapacity});
        initial_residual[rank] = sources[rank][0];
        cuda_synchronize();
    }

    bool consumed = false;
    TP2FeatureSink sink{
        .features = features,
        .positions = positions,
        .layers = kLayers,
        .consume_prefill = [&](const std::array<Tensor, 2>& feature_views,
                               const std::array<Tensor, 2>& position_views, bool rewrite) {
            consumed = rewrite;
            for (int rank = 0; rank < 2; ++rank) {
                if (feature_views[rank].ne[0] != kHalf || feature_views[rank].ne[1] != tokens ||
                    position_views[rank].ne[0] != tokens) {
                    throw std::runtime_error("TP2 feature consumer received an invalid window");
                }
            }
        },
    };
    TP2FeatureSink invalid = sink;
    invalid.features[1] = Tensor(features[1].data, DType::BF16, {kHalf - 1, kCapacity});
    try {
        invalid.begin(initial_residual);
        throw std::runtime_error("TP2 feature sink accepted a truncated peer shard");
    } catch (const std::invalid_argument&) {
    }
    sink.begin(initial_residual);
    try {
        sink.consume_prefill_chunk(false);
        throw std::runtime_error("TP2 feature sink accepted an incomplete capture");
    } catch (const std::logic_error&) {
    }
    for (int index = 0; index < static_cast<int>(kLayers.size()); ++index) {
        for (int rank = 0; rank < 2; ++rank) {
            cuda_check(cudaSetDevice(rank), "feature capture device");
            sink.capture_layer(rank, kLayers[index], sources[rank][index], streams[rank]);
        }
    }
    try {
        sink.consume_prefill_chunk(false);
        throw std::runtime_error("TP2 feature sink accepted stale positions");
    } catch (const std::logic_error&) {
    }
    cuda_check(cudaSetDevice(0), "feature position device");
    sink.capture_positions(0, source_positions[0], streams[0]);
    try {
        sink.require_complete();
        throw std::runtime_error("TP2 feature sink accepted a missing peer position");
    } catch (const std::logic_error&) {
    }
    for (int rank = 1; rank < 2; ++rank) {
        cuda_check(cudaSetDevice(rank), "feature position device");
        sink.capture_positions(rank, source_positions[rank], streams[rank]);
    }
    sink.consume_prefill_chunk(true);
    int failures = consumed ? 0 : 1;
    for (int rank = 0; rank < 2; ++rank) {
        cuda_check(cudaSetDevice(rank), "feature verify device");
        cuda_synchronize(streams[rank]);
        const auto actual = from_device<std::uint16_t>(rank_buffers[rank].features,
                                                        static_cast<std::size_t>(kHalf) * kCapacity);
        for (int column = 0; column < tokens; ++column) {
            const std::vector<std::uint16_t> slice(actual.begin() +
                                                        static_cast<std::size_t>(column) * kHalf,
                                                    actual.begin() +
                                                        static_cast<std::size_t>(column + 1) * kHalf);
            const std::vector<std::uint16_t> reference(
                expected[rank].begin() + static_cast<std::size_t>(column) * kHalf,
                expected[rank].begin() + static_cast<std::size_t>(column + 1) * kHalf);
            failures += verify_exact("TP2 feature shard", slice, reference);
        }
        for (std::size_t offset = static_cast<std::size_t>(kHalf) * tokens;
             offset < actual.size(); ++offset) {
            if (actual[offset] != 0xa5a5) {
                ++failures;
                break;
            }
        }
        const auto actual_positions =
            from_device<std::int32_t>(rank_buffers[rank].positions, kCapacity);
        for (int column = 0; column < tokens; ++column)
            if (actual_positions[column] != expected_positions[rank][column]) ++failures;
    }

    sink.begin(initial_residual);
    for (int rank = 0; rank < 2; ++rank) {
        cuda_check(cudaSetDevice(rank), "feature graph device");
        rank_buffers[rank].features.fill(0xa5);
        rank_buffers[rank].positions.fill(0xa5);
        cuda_synchronize();
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(streams[rank], [&] {
            for (int index = 0; index < static_cast<int>(kLayers.size()); ++index)
                sink.capture_layer(rank, kLayers[index], sources[rank][index], streams[rank]);
            sink.capture_positions(rank, source_positions[rank], streams[rank]);
        });
        graph.instantiate(definition);
        for (int replay = 0; replay < 2; ++replay) {
            graph.launch(streams[rank]);
            cuda_synchronize(streams[rank]);
        }
        const auto actual = from_device<std::uint16_t>(rank_buffers[rank].features,
                                                        static_cast<std::size_t>(kHalf) * tokens);
        failures += verify_exact("TP2 graph feature shard", actual, expected[rank]);
        const auto actual_positions =
            from_device<std::int32_t>(rank_buffers[rank].positions, tokens);
        failures += verify_exact("TP2 graph positions", actual_positions, expected_positions[rank]);
        cuda_check(cudaStreamDestroy(streams[rank]), "destroy feature test stream");
    }
    return failures;
}

} // namespace

int main() {
    try {
        int devices = 0;
        cuda_check(cudaGetDeviceCount(&devices), "count feature test devices");
        if (devices < 2) {
            std::cout << "SKIP: TP2 feature capture needs two GPUs\n";
            return 77;
        }
        const int failures = run_case(1) + run_case(7);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " TP2 feature capture\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "TP2 feature capture: " << error.what() << '\n';
        return 1;
    }
}
