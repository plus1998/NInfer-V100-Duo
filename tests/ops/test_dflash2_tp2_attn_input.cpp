#include "ninfer/ops/attn_input_proj.h"

#include "core/decode_graph.h"
#include "ops/input_projection_test_common.h"

#include <array>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

constexpr ReductionCriterion kA16Criterion{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

int verify_segment(const std::string& name, const GuardedBf16Tensor& output,
                   const quantized_weight::PackedWeight& weight, int offset, int rows,
                   const std::vector<float>& input, int tokens) {
    int failures = output.verify_guards(name) + output.verify_fully_written(name);
    const auto values = gather_rows(output.values(), rows, 0, rows, tokens, 31);
    const auto oracle = projection_oracle(weight, offset, rows, input, 5120, tokens, 31);
    return failures + compare(name, values, oracle, kA16Criterion);
}

int run_case(DevicePackedWeight& parent, int tokens,
             const quantized_weight::PackedWeight* full = nullptr, int rank = 0) {
    const auto input = make_bf16_activation(5120, tokens, 521U + tokens);
    const auto input_bits = bf16_bits(input);
    DeviceBuffer device_input = to_device(input_bits);
    GuardedBf16Tensor query(2048, tokens);
    GuardedBf16Tensor key(512, tokens);
    GuardedBf16Tensor value(512, tokens);
    Tensor x(device_input.p, DType::BF16, {5120, tokens});
    Tensor q = query.tensor();
    Tensor k = key.tensor();
    Tensor v = value.tensor();
    if (tokens == 1) {
        Weight invalid = parent.view();
        invalid.n = 6144;
        try {
            ops::dflash2_tp2_attn_input_proj(x, invalid, q, k, v, nullptr);
            std::cerr << "DFlash2 TP2 QKV accepted unsplit parent weight\n";
            return 1;
        } catch (const std::invalid_argument&) {
        }
    }
    if (tokens == 48) {
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "QKV graph stream");
        {
            DecodeGraphDefinition definition;
            DecodeGraphExecutable executable;
            definition.capture(stream, [&] {
                ops::dflash2_tp2_attn_input_proj(x, parent.view(), q, k, v, stream);
            });
            executable.instantiate(definition);
            executable.launch(stream);
            cuda_synchronize(stream);
            executable.launch(stream);
            cuda_synchronize(stream);
        }
        cuda_check(cudaStreamDestroy(stream), "destroy QKV graph stream");
    } else {
        ops::dflash2_tp2_attn_input_proj(x, parent.view(), q, k, v, nullptr);
    }
    cuda_synchronize();
    const std::string label = "DFlash2 TP2 QKV rank=" + std::to_string(rank) +
                              " T=" + std::to_string(tokens);
    const auto& oracle_weight = full ? *full : parent.host;
    const int q_offset = full ? rank * 2048 : 0;
    const int k_offset = full ? 4096 + rank * 512 : 2048;
    const int v_offset = full ? 5120 + rank * 512 : 2560;
    int failures = verify_segment(label + " Q", query, oracle_weight, q_offset, 2048,
                                   input, tokens);
    failures += verify_segment(label + " K", key, oracle_weight, k_offset, 512, input, tokens);
    failures += verify_segment(label + " V", value, oracle_weight, v_offset, 512, input, tokens);
    failures += verify_preserved(label + " input", device_input, input_bits);
    failures += parent.verify_preserved(label + " weight");
    return failures;
}

quantized_weight::PackedWeight make_rank_shard(
    const quantized_weight::PackedWeight& full, int rank) {
    auto local = quantized_weight::make_patterned_weight(QType::W8G32_F16S, 3072, 5120, 509U);
    constexpr std::array<int, 3> full_start{0, 4096, 5120};
    constexpr std::array<int, 3> local_start{0, 2048, 2560};
    constexpr std::array<int, 3> section_rows{2048, 512, 512};
    constexpr int code_row_bytes = 5120;
    constexpr int scale_row_bytes = 5120 / 32 * 2;
    for (std::size_t section = 0; section < full_start.size(); ++section) {
        const int source = full_start[section] + rank * section_rows[section];
        const int dest = local_start[section];
        const std::size_t codes = static_cast<std::size_t>(section_rows[section]) * code_row_bytes;
        const std::size_t scales = static_cast<std::size_t>(section_rows[section]) * scale_row_bytes;
        std::copy_n(full.payload.begin() + static_cast<std::size_t>(source) * code_row_bytes,
                    codes, local.payload.begin() + static_cast<std::size_t>(dest) * code_row_bytes);
        std::copy_n(full.payload.begin() + full.scale_plane_offset +
                        static_cast<std::size_t>(source) * scale_row_bytes,
                    scales, local.payload.begin() + local.scale_plane_offset +
                                static_cast<std::size_t>(dest) * scale_row_bytes);
    }
    return local;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    DevicePackedWeight parent(quantized_weight::make_patterned_weight(
        QType::W8G32_F16S, 3072, 5120, 509U,
        {.row_split_scale = quantized_weight::RowSplitScalePattern::Small,
         .row_split_codes = quantized_weight::RowSplitCodePattern::Hashed}));
    int failures = 0;
    for (int tokens : {1, 8, 9, 24, 32, 48}) { failures += run_case(parent, tokens); }
    const auto full = quantized_weight::make_patterned_weight(
        QType::W8G32_F16S, 6144, 5120, 521U,
        {.row_split_scale = quantized_weight::RowSplitScalePattern::Small,
         .row_split_codes = quantized_weight::RowSplitCodePattern::Coordinate,
         .decorrelate_coordinates = true});
    for (int rank = 0; rank < 2; ++rank) {
        DevicePackedWeight shard(make_rank_shard(full, rank));
        for (int tokens : {24, 48}) { failures += run_case(shard, tokens, &full, rank); }
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " DFlash2 TP2 QKV\n";
    return failures == 0 ? 0 : 1;
}
