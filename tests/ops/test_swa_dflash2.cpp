#include "ninfer/ops/swa.h"

#include "core/decode_graph.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kHeadDim = 128;
constexpr int kQueryHeads = 16;
constexpr int kKvHeads = 4;
constexpr int kWindow = 2048;
constexpr int kPadded = 2056;
constexpr int kLanes = 3;
constexpr float kScale = 0.08838834764831844055F;

constexpr ReductionCriterion kCriterion{
    .relative_l2 = 6.0e-3,
    .gross_absolute = 5.0e-4,
    .gross_relative_to_max_reference = 7.0e-3,
};

std::size_t output_index(int dim, int head, int token, int batch, int width) {
    return static_cast<std::size_t>(dim) + static_cast<std::size_t>(kHeadDim) *
               (head + kQueryHeads * (token + width * batch));
}

std::size_t query_kv_index(int dim, int head, int token, int batch, int width) {
    return static_cast<std::size_t>(dim) + static_cast<std::size_t>(kHeadDim) *
               (head + kKvHeads * (token + width * batch));
}

std::size_t context_index(int dim, int head, int slot, int lane) {
    return static_cast<std::size_t>(dim) + static_cast<std::size_t>(kHeadDim) *
               (slot + kPadded * (head + kKvHeads * lane));
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result[index] = f32_to_bf16(values[index]);
    }
    return result;
}

int run_case(int width, const std::vector<int>& frontiers, const std::vector<int>& live,
             bool boundary, int max_context = 300000, bool graph = false) {
    const int batch = static_cast<int>(frontiers.size());
    const auto output_size = static_cast<std::size_t>(kHeadDim) * kQueryHeads * width * batch;
    const auto kv_size = static_cast<std::size_t>(kHeadDim) * kKvHeads * width * batch;
    const auto context_size = static_cast<std::size_t>(kHeadDim) * kPadded * kKvHeads * kLanes;
    std::vector<float> query(output_size), query_key(kv_size), query_value(kv_size);
    std::vector<float> context_key(context_size), context_value(context_size);
    fill_uniform(query, 101, -0.3F, 0.3F);
    fill_uniform(query_key, 102, -0.3F, 0.3F);
    fill_uniform(query_value, 103, -0.65F, 0.65F);
    fill_uniform(context_key, 104, -0.3F, 0.3F);
    fill_uniform(context_value, 105, -0.65F, 0.65F);
    const std::array<int, kLanes> lanes{2, 0, 1};
    if (boundary) {
        std::fill(query.begin(), query.end(), 0.0F);
        std::fill(query_key.begin(), query_key.end(), 0.0F);
        std::fill(query_value.begin(), query_value.end(), 0.0F);
        std::fill(context_key.begin(), context_key.end(), 0.0F);
        std::fill(context_value.begin(), context_value.end(), 0.0F);
        for (int row = 0; row < batch; ++row) {
            for (int head = 0; head < kKvHeads; ++head) {
                for (int dim = 0; dim < kHeadDim; ++dim) {
                    context_value[context_index(dim, head,
                                                (frontiers[row] - 2048) & (kWindow - 1),
                                                lanes[row])] = 512.0F;
                    context_value[context_index(dim, head,
                                                (frontiers[row] - 2047) & (kWindow - 1),
                                                lanes[row])] = 64.0F;
                }
            }
        }
    }
    round_to_bf16(query);
    round_to_bf16(query_key);
    round_to_bf16(query_value);
    round_to_bf16(context_key);
    std::vector<std::uint16_t> context_half(context_size);
    for (std::size_t index = 0; index < context_size; ++index) {
        context_half[index] = quantized_weight::detail::f32_to_f16(context_value[index]);
        context_value[index] = quantized_weight::detail::f16_to_f32(context_half[index]);
    }
    std::vector<int> positions(static_cast<std::size_t>(width) * batch);
    std::vector<int> selected_lanes(batch);
    for (int row = 0; row < batch; ++row) {
        selected_lanes[row] = lanes[row];
        for (int token = 0; token < width; ++token) {
            positions[static_cast<std::size_t>(width) * row + token] = frontiers[row] + token;
        }
    }

    DeviceBuffer q_device = to_device(bf16_bits(query));
    DeviceBuffer qk_device = to_device(bf16_bits(query_key));
    DeviceBuffer qv_device = to_device(bf16_bits(query_value));
    DeviceBuffer ck_device = to_device(bf16_bits(context_key));
    DeviceBuffer cv_device = to_device(context_half);
    DeviceBuffer positions_device = to_device_i32(positions);
    DeviceBuffer valid_device = to_device_i32(live);
    DeviceBuffer lanes_device = to_device_i32(selected_lanes);
    GuardedDeviceBuffer output_device(output_size * sizeof(std::uint16_t));
    output_device.fill(0xa5);

    const Tensor q(q_device.p, DType::BF16, {kHeadDim, kQueryHeads, width, batch});
    const Tensor qk(qk_device.p, DType::BF16, {kHeadDim, kKvHeads, width, batch});
    const Tensor qv(qv_device.p, DType::BF16, {kHeadDim, kKvHeads, width, batch});
    const Tensor pos(positions_device.p, DType::I32, {width, batch});
    const Tensor valid(valid_device.p, DType::I32, {batch});
    const Tensor lane(lanes_device.p, DType::I32, {batch});
    Tensor output(output_device.data(), DType::BF16, {kHeadDim, kQueryHeads, width, batch});
    const CyclicKVCacheLayerView cache{
        .k = Tensor(ck_device.p, DType::BF16, {kHeadDim, kPadded, kKvHeads, kLanes}),
        .v = Tensor(cv_device.p, DType::FP16, {kHeadDim, kPadded, kKvHeads, kLanes}),
        .capacity = kWindow,
        .padded_capacity = kPadded,
        .num_kv_heads = kKvHeads,
        .head_dim = kHeadDim,
        .lane_capacity = kLanes,
    };
    const ops::SwaContextExecutionEnvelope envelope{0, static_cast<std::uint32_t>(max_context)};
    DeviceArena workspace(ops::swa_workspace_capacity_bytes(envelope, width, width, batch));
    const auto launch = [&](cudaStream_t stream) {
        ops::swa(q, qk, qv, pos, valid, lane, kScale, cache, envelope, workspace, output, stream);
    };
    if (graph) {
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "graph stream");
        {
            DecodeGraphDefinition definition;
            DecodeGraphExecutable executable;
            definition.capture(stream, [&] { launch(stream); });
            executable.instantiate(definition);
            executable.launch(stream);
            cuda_synchronize(stream);
            output_device.fill(0xa5);
            executable.launch(stream);
            cuda_synchronize(stream);
        }
        cuda_check(cudaStreamDestroy(stream), "destroy graph stream");
    } else {
        launch(nullptr);
    }
    cuda_synchronize();
    const auto actual = from_device_bf16(output_device.data(), output_size);
    std::vector<double> measured;
    std::vector<double> expected;
    for (int row = 0; row < batch; ++row) {
        for (int token : {0, width - 1}) {
            for (int head : {0, 7, 15}) {
                const int key_head = head / 4;
                if (token >= live[row]) {
                    for (int dim = 0; dim < kHeadDim; ++dim) {
                        measured.push_back(actual[output_index(dim, head, token, row, width)]);
                        expected.push_back(0.0);
                    }
                    continue;
                }
                const int start = std::max(0, frontiers[row] - (kWindow - 1));
                const int minimum = positions[static_cast<std::size_t>(width) * row + token] -
                                    (kWindow - 1);
                std::vector<double> probabilities;
                std::vector<bool> from_context;
                std::vector<int> selected_keys;
                double maximum = -std::numeric_limits<double>::infinity();
                for (int key = start; key < frontiers[row] + live[row]; ++key) {
                    if (key < minimum) continue;
                    const bool context = key < frontiers[row];
                    const int local = key - frontiers[row];
                    double dot = 0.0;
                    for (int dim = 0; dim < kHeadDim; ++dim) {
                        const float key_value = context
                            ? context_key[context_index(dim, key_head, key & (kWindow - 1), lanes[row])]
                            : query_key[query_kv_index(dim, key_head, local, row, width)];
                        dot += static_cast<double>(query[output_index(dim, head, token, row, width)]) *
                               key_value;
                    }
                    const double score = dot * static_cast<double>(kScale);
                    probabilities.push_back(score);
                    from_context.push_back(context);
                    selected_keys.push_back(key);
                    maximum = std::max(maximum, score);
                }
                double denominator = 0.0;
                for (double& score : probabilities) {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                for (int dim = 0; dim < kHeadDim; ++dim) {
                    double numerator = 0.0;
                    for (std::size_t index = 0; index < selected_keys.size(); ++index) {
                        const int key = selected_keys[index];
                        const float value = from_context[index]
                            ? context_value[context_index(dim, key_head, key & (kWindow - 1), lanes[row])]
                            : query_value[query_kv_index(dim, key_head, key - frontiers[row], row,
                                                         width)];
                        numerator += probabilities[index] * value;
                    }
                    measured.push_back(actual[output_index(dim, head, token, row, width)]);
                    expected.push_back(numerator / denominator);
                }
            }
        }
    }
    const std::string label = "dflash2 swa W=" + std::to_string(width) + " B=" +
                              std::to_string(batch) + (boundary ? " boundary" : " random");
    int failures = verify_reduction(label, measured, expected, kCriterion);
    failures += output_device.verify_guards(label);
    failures += verify_exact((label + " q").c_str(), from_device<std::uint16_t>(q_device, output_size),
                             bf16_bits(query));
    failures += verify_exact((label + " context v").c_str(),
                             from_device<std::uint16_t>(cv_device, context_size), context_half);
    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) return 77;
        int failures = 0;
        failures += run_case(2, {0}, {2}, false);
        failures += run_case(8, {96}, {8}, false, 96, true);
        failures += run_case(8, {2048, 2051, 4099}, {8, 5, 0}, false);
        failures += run_case(8, {2048, 2051, 4099}, {8, 5, 0}, true);
        failures += run_case(16, {8191, 4096}, {16, 16}, false);
        std::cout << (failures == 0 ? "dflash2 swa: PASS\n" : "dflash2 swa: FAIL\n");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "dflash2 swa: " << error.what() << '\n';
        return 1;
    }
}
