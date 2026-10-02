#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/linear.h"

#include "core/device.h"
#include "ops/input_projection_test_common.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace qw = ninfer::test::quantized_weight;

namespace {

constexpr int kHidden = 5120;
constexpr std::array<int, 12> kRows{0, 15, 16, 31, 32, 255, 1023, 2047, 2559, 4095, 4096,
                                     5119};
constexpr ReductionCriterion kCriterion{/*relative_l2=*/8.0e-3,
                                        /*gross_absolute=*/2.0e-3,
                                        /*gross_relative=*/5.0e-3};

void set_rank(const ExecutionContext& ec, int rank) {
    cuda_check(cudaSetDevice(ec.dev[rank]->device), "select TP2 rank");
}

qw::PatternedWeightOptions options(int column_origin) {
    qw::PatternedWeightOptions result;
    result.column_origin = column_origin;
    result.decorrelate_coordinates = true;
    result.row_split_scale = qw::RowSplitScalePattern::Small;
    result.row_split_codes = qw::RowSplitCodePattern::Coordinate;
    return result;
}

struct Rank {
    std::optional<input_projection::DevicePackedWeight> weight;
    DeviceBuffer input;
    DeviceBuffer base;
    DeviceBuffer delta;
    DeviceBuffer residual;
    DeviceBuffer projected;
    DeviceBuffer staging;
};

int run_case(const ExecutionContext& ec, const ops::PeerEvents& events, int local_rows,
             int width, int batch) {
    const int columns = width * batch;
    const int full_rows = local_rows * 2;
    const std::size_t output_count = static_cast<std::size_t>(kHidden) * columns;
    const std::size_t output_bytes = output_count * sizeof(std::uint16_t);
    const auto full = qw::make_patterned_weight(QType::W8G32_F16S, kHidden, full_rows, 701U,
                                                 options(0));

    std::vector<float> input(static_cast<std::size_t>(full_rows) * columns);
    fill_uniform(input, 103U + static_cast<std::uint32_t>(columns), -0.006F, 0.006F);
    round_to_bf16(input);
    std::vector<std::uint16_t> base(kHidden * 4);
    std::vector<std::uint16_t> delta(320 * 2 * columns);
    std::vector<std::uint16_t> residual(output_count);
    for (int i = 0; i < kHidden; ++i) {
        base[2 * kHidden + i] = f32_to_bf16(0.375F + (i % 19) * 0.0005F);
        base[3 * kHidden + i] = f32_to_bf16(-0.15625F + (i % 11) * 0.0005F);
    }
    for (std::size_t i = 0; i < delta.size(); ++i)
        delta[i] = f32_to_bf16((i / 320 % 2 == 0 ? 0.046875F : -0.03125F) +
                               (i % 17) * 0.00025F);
    for (std::size_t i = 0; i < residual.size(); ++i)
        residual[i] = f32_to_bf16(static_cast<float>(static_cast<int>(i % 127) - 63) / 512.0F);

    std::array<Rank, 2> ranks;
    std::array<Tensor, 2> projected;
    std::array<Tensor, 2> staging;
    std::array<Tensor, 2> shard_inputs;
    std::array<Weight, 2> shard_weights;
    std::array<std::vector<std::uint16_t>, 2> inputs;
    int failures = 0;
    for (int rank = 0; rank < 2; ++rank) {
        set_rank(ec, rank);
        auto& state = ranks[rank];
        state.weight.emplace(qw::make_patterned_weight(QType::W8G32_F16S, kHidden, local_rows,
                                                       701U, options(rank * local_rows)));
        for (int row : kRows)
            for (int column : {0, 31, 32, local_rows - 1}) {
                if (qw::logical_weight_fp64(state.weight->host, row, column) !=
                    qw::logical_weight_fp64(full, row, column + rank * local_rows)) {
                    std::cerr << "TP2 finish weight shard differs from parent\n";
                    ++failures;
                }
            }
        std::vector<std::uint16_t> local_input(static_cast<std::size_t>(local_rows) * columns);
        for (int col = 0; col < columns; ++col)
            for (int row = 0; row < local_rows; ++row)
                local_input[static_cast<std::size_t>(col) * local_rows + row] = f32_to_bf16(
                    input[static_cast<std::size_t>(col) * full_rows + rank * local_rows + row]);
        inputs[rank] = std::move(local_input);
        state.input = to_device(inputs[rank]);
        state.base = to_device(base);
        state.delta = to_device(delta);
        state.residual = to_device(residual);
        state.projected = DeviceBuffer(output_bytes);
        state.staging = DeviceBuffer(output_bytes);
        projected[rank] = Tensor(state.projected.p, DType::BF16, {kHidden, width, batch});
        staging[rank] = Tensor(state.staging.p, DType::BF16, {kHidden, width, batch});
        shard_inputs[rank] = Tensor(state.input.p, DType::BF16, {local_rows, columns});
        shard_weights[rank] = state.weight->view();
        Tensor flat_projected = projected[rank].view({kHidden, columns});
        ops::linear(shard_inputs[rank], shard_weights[rank], flat_projected,
                    ec.dev[rank]->stream);
    }
    for (int rank = 0; rank < 2; ++rank) {
        set_rank(ec, rank);
        cuda_synchronize(ec.dev[rank]->stream);
        std::vector<double> partial_expected;
        std::vector<double> partial_actual;
        const auto partial = from_device_bf16(ranks[rank].projected, output_count);
        for (int row : kRows)
            for (int col = 0; col < columns; ++col) {
                partial_expected.push_back(qw::dot_fp64(
                    ranks[rank].weight->host, row,
                    input.data() + static_cast<std::size_t>(col) * full_rows + rank * local_rows,
                    local_rows));
                partial_actual.push_back(partial[static_cast<std::size_t>(col) * kHidden + row]);
            }
        failures += verify_reduction("TP2 finish projection rank=" + std::to_string(rank),
                                     partial_actual, partial_expected, kCriterion);
    }

    std::vector<double> expected;
    expected.reserve(kRows.size() * columns);
    for (int row : kRows) {
        for (int col = 0; col < columns; ++col) {
            const double current = qw::dot_fp64(full, row,
                                                 input.data() + static_cast<std::size_t>(col) *
                                                                    full_rows, full_rows);
            const double previous = col % width == 0 ? 0.0 : qw::dot_fp64(
                full, row, input.data() + static_cast<std::size_t>(col - 1) * full_rows,
                full_rows);
            const double gain0 = bf16_to_f32(base[2 * kHidden + row]) +
                                 bf16_to_f32(delta[(static_cast<std::size_t>(col) * 2) * 320 +
                                                     row / 16]);
            const double gain1 = bf16_to_f32(base[3 * kHidden + row]) +
                                 bf16_to_f32(delta[(static_cast<std::size_t>(col) * 2 + 1) * 320 +
                                                     row / 16]);
            expected.push_back(bf16_to_f32(residual[static_cast<std::size_t>(col) * kHidden + row]) +
                               gain0 * current + gain1 * previous);
        }
    }

    const std::array<Tensor, 2> flat_projected = {projected[0].view({kHidden, columns}),
                                                   projected[1].view({kHidden, columns})};
    const std::array<Tensor, 2> flat_staging = {staging[0].view({kHidden, columns}),
                                                staging[1].view({kHidden, columns})};
    ops::linear_row_parallel(shard_inputs, shard_weights, flat_projected, flat_staging, ec,
                             events);
    for (int rank = 0; rank < 2; ++rank) {
        set_rank(ec, rank);
        auto& state = ranks[rank];
        Tensor base_view(state.base.p, DType::BF16, {kHidden, 2, 2});
        Tensor delta_view(state.delta.p, DType::BF16, {320, 2, width, batch});
        Tensor residual_view(state.residual.p, DType::BF16, {kHidden, width, batch});
        ops::dynamic_grouped_conv_finish_add(projected[rank], base_view, delta_view,
                                              residual_view, ec.dev[rank]->stream);
    }
    std::vector<std::uint16_t> first_result;
    for (int rank = 0; rank < 2; ++rank) {
        set_rank(ec, rank);
        cuda_synchronize(ec.dev[rank]->stream);
        const auto result = from_device<std::uint16_t>(ranks[rank].residual, output_count);
        std::vector<double> actual;
        actual.reserve(expected.size());
        for (int row : kRows)
            for (int col = 0; col < columns; ++col)
                actual.push_back(bf16_to_f32(result[static_cast<std::size_t>(col) * kHidden + row]));
        const std::string label = "TP2 DFlash2 finish C=" + std::to_string(local_rows) +
                                  " W=" + std::to_string(width) + " B=" + std::to_string(batch) +
                                  " rank=" + std::to_string(rank);
        failures += verify_reduction(label, actual, expected, kCriterion);
        if (rank == 0) first_result = result;
        else failures += verify_exact("TP2 finish ranks agree", result, first_result);
        failures += ranks[rank].weight->verify_preserved(label + " weight");
        failures += verify_exact((label + " input").c_str(),
                                 from_device<std::uint16_t>(ranks[rank].input,
                                                            inputs[rank].size()), inputs[rank]);
        failures += verify_exact((label + " base").c_str(),
                                 from_device<std::uint16_t>(ranks[rank].base, base.size()), base);
        failures += verify_exact((label + " delta").c_str(),
                                 from_device<std::uint16_t>(ranks[rank].delta, delta.size()),
                                 delta);
    }
    return failures;
}

} // namespace

int main() {
    try {
        int count = 0;
        cuda_check(cudaGetDeviceCount(&count), "count devices");
        if (count < 2) {
            std::cout << "SKIP: TP2 finish needs two GPUs\n";
            return 77;
        }
        const ExecutionContext ec({0, 1});
        ops::enable_peer_access(ec);
        const ops::PeerEvents events(ec);
        int failures = 0;
        failures += run_case(ec, events, 2048, 2, 1);
        failures += run_case(ec, events, 2048, 8, 3);
        failures += run_case(ec, events, 2048, 16, 8);
        failures += run_case(ec, events, 8704, 8, 3);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " DFlash2 TP2 conv finish\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "DFlash2 TP2 conv finish: " << error.what() << '\n';
        return 1;
    }
}
