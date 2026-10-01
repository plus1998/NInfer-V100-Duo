// GGUF-blocks projection timing at the Qwen3.8-27B TP2 shard shapes: one launch of each
// registered block format, T in {1, 4}, reporting effective weight bandwidth.
#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/gguf/gguf.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

int main() {
    using namespace ninfer;
    struct Shape {
        const char* name;
        int n, k;
    };
    const Shape shapes[] = {{"gdn qkvz shard", 8192, 5120},
                            {"mlp gate_up shard", 17408, 5120},
                            {"mlp down shard", 5120, 8704},
                            {"output head shard", 124160, 5120}};
    const GgufType types[] = {GgufType::IQ3_S,   GgufType::IQ3_XXS, GgufType::IQ4_XS,
                              GgufType::IQ2_XS,  GgufType::IQ2_XXS, GgufType::IQ2_S,
                              GgufType::Q2_K,    GgufType::Q4_K,    GgufType::Q6_K};
    const char* names[] = {"IQ4_XS", "IQ3_S", "IQ3_XXS", "IQ2_XS", "IQ2_XXS",
                           "IQ2_S",  "IQ1_M", "Q2_K",    "Q4_K",   "Q6_K"};
    std::mt19937 rng(7);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (const Shape& shape : shapes) {
        for (const GgufType type : types) {
            if (shape.n == 124160 && type != GgufType::Q4_K) { continue; }
            const std::size_t bytes =
                std::size_t(shape.n) * (shape.k / 256) * gguf_block_bytes(type);
            std::vector<unsigned char> host(bytes);
            for (auto& b : host) { b = static_cast<unsigned char>(rng()); }
            // Small positive FP16 at every possible scale offset keeps values finite.
            const __half scale = __float2half_rn(0.0001F);
            for (std::size_t b = 0; b + gguf_block_bytes(type) <= bytes; b += gguf_block_bytes(type)) {
                for (const int offset : {0, 2}) { std::memcpy(host.data() + b + offset, &scale, 2); }
                if (type == GgufType::Q6_K) { std::memcpy(host.data() + b + 208, &scale, 2); }
                if (type == GgufType::Q2_K) {
                    std::memcpy(host.data() + b + 80, &scale, 2);
                    std::memcpy(host.data() + b + 82, &scale, 2);
                }
            }
            DeviceBuffer weights(bytes);
            weights.copy_from_host(host.data(), bytes);
            Weight w;
            w.qtype = QType::GGUF;
            w.layout = QuantLayout::GgufBlocks;
            w.n = w.shape[0] = w.padded_shape[0] = shape.n;
            w.k = w.shape[1] = w.padded_shape[1] = shape.k;
            w.ndim = 2;
            w.gguf_segment_count = 1;
            w.gguf_segments[0] = {weights.p, shape.n, type};
            w.payload = w.qdata = weights.p;
            for (const bool a8 : {false, true}) {
            for (const int tokens : {1, 4}) {
                std::vector<__nv_bfloat16> x(std::size_t(shape.k) * tokens,
                                             __float2bfloat16_rn(0.01F));
                DeviceBuffer xd(x.size() * 2), yd(std::size_t(shape.n) * tokens * 2);
                xd.copy_from_host(x.data(), xd.bytes);
                Tensor xt(xd.p, DType::BF16, {shape.k, tokens});
                Tensor yt(yd.p, DType::BF16, {shape.n, tokens});
                WorkspaceArena workspace(
                    std::max<std::size_t>(256, ops::detail::gguf_workspace_bytes(shape.n, shape.k, tokens)));
                for (int i = 0; i < 3; ++i) {
                    ops::detail::gguf_project(xt, w, &yt, 1, false, false, a8, &workspace, nullptr);
                }
                constexpr int kIterations = 20;
                CUDA_CHECK(cudaEventRecord(start));
                for (int i = 0; i < kIterations; ++i) {
                    ops::detail::gguf_project(xt, w, &yt, 1, false, false, a8, &workspace, nullptr);
                }
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                const double us = 1000.0 * ms / kIterations;
                std::cout << shape.name << " " << names[static_cast<int>(type)] << " T=" << tokens
                          << (a8 ? " A8 " : " A16 ") << us << " us " << bytes / (us * 1e3)
                          << " GB/s\n";
            }
            }
        }
    }
    return 0;
}
