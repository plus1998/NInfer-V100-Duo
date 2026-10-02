#include "ninfer/ops/attn_input_proj.h"

#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>

using namespace ninfer;
using namespace ninfer::bench;

int main() {
    try {
        int devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (devices == 0) return 77;
        DeviceBuffer codes(static_cast<std::size_t>(3072) * 5120);
        DeviceBuffer scales(static_cast<std::size_t>(3072) * (5120 / 32) * 2);
        codes.fill();
        scales.fill();
        DeviceBuffer flush(256ULL << 20);
        Weight weight{};
        weight.qtype = QType::W8G32_F16S;
        weight.layout = QuantLayout::RowSplit;
        weight.scale_dtype = DType::FP16;
        weight.group_size = 32;
        weight.group = 32;
        weight.ndim = 2;
        weight.n = 3072;
        weight.k = 5120;
        weight.shape[0] = 3072;
        weight.shape[1] = 5120;
        weight.padded_shape[0] = 3072;
        weight.padded_shape[1] = 5120;
        weight.qdata = codes.p;
        weight.scales = scales.p;
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::printf("T,median_us,min_us,p95_us,graph_nodes\n");
        for (int tokens : {1, 8, 24, 32, 48}) {
            DeviceBuffer input = make_bf16(static_cast<std::size_t>(5120) * tokens);
            DeviceBuffer query = make_zeros(static_cast<std::size_t>(2048) * tokens * 2);
            DeviceBuffer key = make_zeros(static_cast<std::size_t>(512) * tokens * 2);
            DeviceBuffer value = make_zeros(static_cast<std::size_t>(512) * tokens * 2);
            const Tensor x(input.p, DType::BF16, {5120, tokens});
            Tensor q(query.p, DType::BF16, {2048, tokens});
            Tensor k(key.p, DType::BF16, {512, tokens});
            Tensor v(value.p, DType::BF16, {512, tokens});
            TimedGraph graph;
            graph.capture(stream, [&](cudaStream_t launch_stream) {
                ops::dflash2_tp2_attn_input_proj(x, weight, q, k, v, launch_stream);
            });
            const auto result = measure_cold_launch(
                [&](cudaStream_t launch_stream) { graph.launch(launch_stream); },
                flush, stream, 8, 30);
            std::printf("%d,%.3f,%.3f,%.3f,%zu\n", tokens, result.median_us,
                        result.min_us, result.p95_us, graph.nodes());
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "DFlash2 TP2 QKV bench: %s\n", error.what());
        return 1;
    }
    return 0;
}
