#include "ninfer/ops/swa.h"

#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr int kDim = 128;
constexpr int kQHeads = 16;
constexpr int kKvHeads = 4;
constexpr int kCapacity = 2048;
constexpr float kScale = 0.08838834764831844055F;

struct Options {
    int width = 8;
    int batch = 3;
    int context = 2048;
    int warmup = 8;
    int repeat = 30;
};

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const auto next = [&](const char* flag) {
            if (index + 1 >= argc) { throw std::invalid_argument(std::string("missing ") + flag); }
            return std::atoi(argv[++index]);
        };
        if (!std::strcmp(argv[index], "--width")) options.width = next("width");
        else if (!std::strcmp(argv[index], "--batch")) options.batch = next("batch");
        else if (!std::strcmp(argv[index], "--context")) options.context = next("context");
        else if (!std::strcmp(argv[index], "--warmup")) options.warmup = next("warmup");
        else if (!std::strcmp(argv[index], "--repeat")) options.repeat = next("repeat");
        else if (!std::strcmp(argv[index], "--help")) {
            std::printf("usage: %s [--width 1..16] [--batch 1..8] [--context 0..262144] "
                        "[--warmup N] [--repeat N]\n", argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown flag: " + std::string(argv[index]));
        }
    }
    if (options.width < 1 || options.width > 16 || options.batch < 1 || options.batch > 8 ||
        options.context < 0 || options.context > 262144 || options.warmup < 0 ||
        options.repeat < 1) {
        throw std::invalid_argument("invalid DFlash2 SWA profile");
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (!devices) return 77;
        const Options options = parse(argc, argv);
        const int columns = options.width * options.batch;
        DeviceBuffer q = make_bf16(static_cast<std::size_t>(kDim) * kQHeads * columns);
        DeviceBuffer query_key = make_bf16(static_cast<std::size_t>(kDim) * kKvHeads * columns);
        DeviceBuffer query_value = make_bf16(static_cast<std::size_t>(kDim) * kKvHeads * columns);
        DeviceBuffer context_key = make_bf16(static_cast<std::size_t>(kDim) * kCapacity * kKvHeads * options.batch);
        DeviceBuffer context_value = make_zeros(context_key.bytes);
        DeviceBuffer out = make_zeros(q.bytes);
        std::vector<std::int32_t> positions(columns);
        std::vector<std::int32_t> valid(options.batch, options.width);
        std::vector<std::int32_t> lanes(options.batch);
        for (int row = 0; row < options.batch; ++row) {
            lanes[row] = row;
            for (int token = 0; token < options.width; ++token) {
                positions[row * options.width + token] = options.context + token;
            }
        }
        DeviceBuffer device_positions(positions.size() * sizeof(std::int32_t));
        DeviceBuffer device_valid(valid.size() * sizeof(std::int32_t));
        DeviceBuffer device_lanes(lanes.size() * sizeof(std::int32_t));
        device_positions.copy_from_host(positions.data(), device_positions.bytes);
        device_valid.copy_from_host(valid.data(), device_valid.bytes);
        device_lanes.copy_from_host(lanes.data(), device_lanes.bytes);

        const Tensor q_tensor(q.p, DType::BF16, {kDim, kQHeads, options.width, options.batch});
        const Tensor key_tensor(query_key.p, DType::BF16, {kDim, kKvHeads, options.width, options.batch});
        const Tensor value_tensor(query_value.p, DType::BF16, {kDim, kKvHeads, options.width, options.batch});
        const Tensor pos_tensor(device_positions.p, DType::I32, {options.width, options.batch});
        const Tensor valid_tensor(device_valid.p, DType::I32, {options.batch});
        const Tensor lane_tensor(device_lanes.p, DType::I32, {options.batch});
        Tensor output_tensor(out.p, DType::BF16, {kDim, kQHeads, options.width, options.batch});
        const CyclicKVCacheLayerView cache{
            .k = Tensor(context_key.p, DType::BF16, {kDim, kCapacity, kKvHeads, options.batch}),
            .v = Tensor(context_value.p, DType::FP16, {kDim, kCapacity, kKvHeads, options.batch}),
            .capacity = kCapacity,
            .padded_capacity = kCapacity,
            .num_kv_heads = kKvHeads,
            .head_dim = kDim,
            .lane_capacity = options.batch,
        };
        const ops::SwaContextExecutionEnvelope envelope{0, static_cast<std::uint32_t>(options.context)};
        WorkspaceArena workspace(ops::swa_workspace_capacity_bytes(
            envelope, options.width, options.width, options.batch));
        DeviceBuffer flush(256ULL << 20);
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        TimedGraph graph;
        graph.capture(stream, [&](cudaStream_t launch_stream) {
            ops::swa(q_tensor, key_tensor, value_tensor, pos_tensor, valid_tensor, lane_tensor,
                     kScale, cache, envelope, workspace, output_tensor, launch_stream);
        });
        const auto timing = measure_cold_launch([&](cudaStream_t launch_stream) {
            graph.launch(launch_stream);
        }, flush, stream, options.warmup, options.repeat);
        CUDA_CHECK(cudaStreamDestroy(stream));
        std::printf("W,B,context,median_us,min_us,p95_us,graph_nodes\n");
        std::printf("%d,%d,%d,%.3f,%.3f,%.3f,%zu\n", options.width, options.batch,
                    options.context, timing.median_us, timing.min_us, timing.p95_us, graph.nodes());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_swa_dflash2_bench: %s\n", error.what());
        return 1;
    }
}
