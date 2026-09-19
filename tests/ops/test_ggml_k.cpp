#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/ggml_k/ggml_k.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;

float half_value(const unsigned char* p) {
    __half value;
    std::memcpy(&value, p, sizeof(value));
    return __half2float(value);
}

// Decode the complete block independently, in GGML's logical group order. This
// oracle does not share the device decoder, staging dtype or reduction tree.
std::array<double, 256> oracle_block(const unsigned char* p, bool q6) {
    std::array<double, 256> result{};
    if (!q6) {
        std::array<int, 8> scale{}, minimum{};
        for (int g = 0; g < 4; ++g) {
            scale[g] = p[4 + g] % 64;
            minimum[g] = p[8 + g] % 64;
            scale[g + 4] = (p[12 + g] % 16) + 16 * (p[4 + g] / 64);
            minimum[g + 4] = (p[12 + g] / 16) + 16 * (p[8 + g] / 64);
        }
        const double d = half_value(p), m = half_value(p + 2);
        for (int pair = 0; pair < 4; ++pair) {
            for (int j = 0; j < 32; ++j) {
                const unsigned code = p[16 + pair * 32 + j];
                result[pair * 64 + j] = d * scale[2 * pair] * (code % 16) -
                                         m * minimum[2 * pair];
                result[pair * 64 + 32 + j] = d * scale[2 * pair + 1] * (code / 16) -
                                              m * minimum[2 * pair + 1];
            }
        }
    } else {
        const double d = half_value(p + 208);
        for (int half = 0; half < 2; ++half) {
            for (int j = 0; j < 32; ++j) {
                const unsigned lo0 = p[half * 64 + j], lo1 = p[half * 64 + 32 + j];
                const unsigned hi = p[128 + half * 32 + j];
                const std::array<int, 4> codes{
                    int(lo0 % 16 + 16 * (hi % 4)) - 32,
                    int(lo1 % 16 + 16 * (hi / 4 % 4)) - 32,
                    int(lo0 / 16 + 16 * (hi / 16 % 4)) - 32,
                    int(lo1 / 16 + 16 * (hi / 64)) - 32};
                for (int section = 0; section < 4; ++section) {
                    const int at = half * 128 + section * 32 + j;
                    const int stored_scale = p[192 + at / 16];
                    const int signed_scale = stored_scale < 128 ? stored_scale : stored_scale - 256;
                    result[at] = d * signed_scale * codes[section];
                }
            }
        }
    }
    return result;
}

void run(int n, int k, int tokens, bool tiled_gdn = false, bool add = false,
         int codec_profile = 0) {
    std::mt19937 rng(8371 + n + tokens);
    std::vector<unsigned char> rows;
    std::vector<std::uint64_t> descriptors(n);
    for (int row = 0; row < n; ++row) {
        const bool q6 = codec_profile == 2 ? true
                        : codec_profile == 1 ? false
                        : codec_profile == 3 ? row < n / 2
                                             : (row % 3) == 1;
        descriptors[row] = (std::uint64_t(rows.size()) << 1) | q6;
        for (int block = 0; block < k / 256; ++block) {
            const std::size_t begin = rows.size();
            const int bytes = q6 ? 210 : 144;
            rows.resize(begin + bytes);
            for (int j = 0; j < bytes; ++j) { rows[begin + j] = rng() & 255; }
            const __half d = __float2half_rn(0.00001f * (1 + rng() % 19));
            std::memcpy(rows.data() + begin + (q6 ? 208 : 0), &d, sizeof(d));
            if (!q6) {
                const __half m = __float2half_rn(0.0001f * (1 + rng() % 11));
                std::memcpy(rows.data() + begin + 2, &m, sizeof(m));
            }
        }
    }
    std::vector<__nv_bfloat16> input(std::size_t(k) * tokens), result(std::size_t(n) * tokens);
    for (auto& value : input) {
        value = __float2bfloat16_rn(float(int(rng() % 2001) - 1000) / 1000.0f);
    }
    DeviceBuffer codes(rows.size()), table(descriptors.size() * sizeof(std::uint64_t));
    DeviceBuffer x(input.size() * 2), y(result.size() * 2);
    codes.copy_from_host(rows.data(), rows.size());
    table.copy_from_host(descriptors.data(), table.bytes);
    x.copy_from_host(input.data(), x.bytes);
    std::vector<__nv_bfloat16> residual(result.size());
    if (add) {
        for (auto& value : residual) {
            value = __float2bfloat16_rn(float(int(rng() % 2001) - 1000) / 2000.0f);
        }
        y.copy_from_host(residual.data(), y.bytes);
    }
    Weight weight;
    weight.qtype = QType::GGML_K;
    weight.layout = QuantLayout::GgmlK256;
    weight.n = n;
    weight.k = k;
    weight.qdata = codes.p;
    weight.qhigh = table.p;
    if (codec_profile != 0) {
        weight.ggml_k_first_q6 = codec_profile == 2 || codec_profile == 3;
        weight.ggml_k_type_change = codec_profile == 3 ? n / 2 : n;
    }
    Tensor xt(x.p, DType::BF16, {k, tokens}), yt(y.p, DType::BF16, {n, tokens});
    std::unique_ptr<WorkspaceArena> workspace;
    const std::size_t workspace_bytes =
        ops::detail::ggml_k_cutlass_workspace_bytes(n, k, tokens);
    if (workspace_bytes != 0) { workspace = std::make_unique<WorkspaceArena>(workspace_bytes); }
    ops::detail::ggml_k_project_split(xt, weight, &yt, 1, add, nullptr, tiled_gdn,
                                      workspace.get());
    CUDA_CHECK(cudaDeviceSynchronize());
    y.copy_to_host(result.data(), y.bytes);

    double error2 = 0, reference2 = 0;
    const int samples = std::min(n, 48);
    for (int sample = 0; sample < samples; ++sample) {
        const int row = sample == samples - 1 ? n - 1 : sample * (n / samples);
        const auto descriptor = descriptors[row];
        const bool q6 = descriptor & 1;
        std::vector<double> decoded(k);
        for (int block = 0; block < k / 256; ++block) {
            const auto values = oracle_block(rows.data() + (descriptor >> 1) +
                                             block * (q6 ? 210 : 144), q6);
            std::copy(values.begin(), values.end(), decoded.begin() + block * 256);
        }
        for (int t = 0; t < tokens; ++t) {
            double reference = add ? double(__bfloat162float(residual[t * n + row])) : 0;
            for (int j = 0; j < k; ++j) {
                // Iterate physical weight columns in GGUF's [repeat,key,dim] order;
                // each represented activation comes from [key,repeat,dim].
                const int key_heads = k / 384;
                const int input_index = tiled_gdn
                    ? ((j / 128 % key_heads) * 3 + j / (128 * key_heads)) * 128 + j % 128
                    : j;
                reference += decoded[j] * double(__bfloat162float(input[t * k + input_index]));
            }
            const double actual = __bfloat162float(result[t * n + row]);
            if (!std::isfinite(actual)) { throw std::runtime_error("non-finite GGML K result"); }
            error2 += (actual - reference) * (actual - reference);
            reference2 += reference * reference;
        }
    }
    const double relative = std::sqrt(error2 / reference2);
    std::cout << "GGML_K N=" << n << " K=" << k << " T=" << tokens
              << " tiled_gdn=" << tiled_gdn << " add=" << add
              << " relative_l2=" << relative << '\n';
    if (relative > 0.004) { throw std::runtime_error("GGML K FP64 oracle mismatch"); }

    if (n == 96 && k == 5120 && tokens == 1 && !tiled_gdn) {
        const std::array<std::int32_t, 4> ids{0, 1, 2, n - 1};
        DeviceBuffer id_buffer(sizeof(ids)), embedded(ids.size() * k * 2);
        id_buffer.copy_from_host(ids.data(), sizeof(ids));
        Tensor id_tensor(id_buffer.p, DType::I32, {static_cast<int>(ids.size())});
        Tensor output(embedded.p, DType::BF16, {k, static_cast<int>(ids.size())});
        ops::detail::ggml_k_embedding(id_tensor, weight, output, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<__nv_bfloat16> values(ids.size() * k);
        embedded.copy_to_host(values.data(), embedded.bytes);
        for (std::size_t token = 0; token < ids.size(); ++token) {
            const auto descriptor = descriptors[ids[token]];
            const bool q6 = descriptor & 1;
            for (int block = 0; block < k / 256; ++block) {
                const auto reference = oracle_block(rows.data() + (descriptor >> 1) +
                                                    block * (q6 ? 210 : 144), q6);
                for (int column = 0; column < 256; ++column) {
                    const auto expected = __float2bfloat16_rn(static_cast<float>(reference[column]));
                    const auto actual = values[token * k + block * 256 + column];
                    if (std::memcmp(&expected, &actual, 2) != 0) {
                        throw std::runtime_error("GGML K embedding represented-value mismatch");
                    }
                }
            }
        }
        std::cout << "GGML_K embedding Q4/Q6 exact BF16 decode passed\n";
    }

    // Fused residual addition and mixed FP32/BF16 row destinations use the same projection.
    if (tokens == 4 && !tiled_gdn && !add) {
        DeviceBuffer fp32(std::size_t(n / 2) * tokens * sizeof(float));
        const Tensor sections[]{Tensor(fp32.p, DType::FP32, {n / 2, tokens}),
                                yt.slice(0, n / 2, n - n / 2)};
        ops::detail::ggml_k_project_split(xt, weight, sections, 2, false, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> first(std::size_t(n / 2) * tokens);
        fp32.copy_to_host(first.data(), fp32.bytes);
        std::vector<__nv_bfloat16> split_result(result.size());
        y.copy_to_host(split_result.data(), y.bytes);
        for (int t = 0; t < tokens; ++t) {
            for (int row = 0; row < n; ++row) {
                const auto expected = result[t * n + row];
                const auto got = row < n / 2 ? __float2bfloat16_rn(first[t * (n / 2) + row])
                                            : split_result[t * n + row];
                if (std::memcmp(&expected, &got, 2) != 0) {
                    throw std::runtime_error("GGML K split destination mismatch");
                }
            }
        }
    }
}
} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { return 77; }
    try {
        for (int profile : {1, 2, 3}) {
            for (int t : {1, 4}) { run(96, 5120, t, false, false, profile); }
        }
        for (int t : {1, 2, 3, 4, 5, 17, 64, 65, 95, 96, 97}) { run(96, 5120, t); }
        run(7168, 5120, 4);
        run(5120, 8704, 4);
        run(17408, 5120, 64);
        // TP2 GDN controls have 48 rows: the last WMMA tile is half full.
        run(48, 5120, 17);
        // Real prefill spans many token tiles; check every output against FP64
        // with mixed Q4/Q6 rows, including the partial final row tile.
        for (int t : {512, 1024}) { run(48, 5120, t); }
        for (int k : {3072, 6144}) {
            for (int t : {1, 4, 17, 65}) {
                run(96, k, t, true, false);
                run(96, k, t, true, true);
            }
        }
        run(48, 3072, 512, true, false);
        run(48, 3072, 512, true, true);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
