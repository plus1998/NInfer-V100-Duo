// GGUF-blocks projection and embedding against an independent FP64 oracle.
//
// The oracle decodes every block on the host with the loops of ggml-quants.c's dequantize_row_*
// functions (the format definition), in FP64, from the same codebooks. It shares no code with the
// device decoders except the verbatim codebook tables.

#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/gguf/gguf.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer;

#define NINFER_GGUF_TABLE(type, name, size) const type name[size] = {
#include "ops/linear/gguf/gguf_tables.inc"
#undef NINFER_GGUF_TABLE

constexpr std::uint8_t kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};

double h(const std::uint8_t* p) {
    __half value;
    std::memcpy(&value, p, 2);
    return __half2float(value);
}

std::uint16_t u16(const std::uint8_t* p) { return std::uint16_t(p[0] | (p[1] << 8)); }
std::uint32_t u32(const std::uint8_t* p) { return u16(p) | (std::uint32_t(u16(p + 2)) << 16); }
std::uint8_t grid_byte(std::uint64_t grid, int j) { return std::uint8_t(grid >> (8 * j)); }

void put_half(std::uint8_t* p, float value) {
    const __half half = __float2half_rn(value);
    std::memcpy(p, &half, 2);
}

// ggml-quants.c dequantize_row_* for one block, in FP64.
std::array<double, 256> oracle(GgufType type, const std::uint8_t* b) {
    std::array<double, 256> y;
    double* out = y.data();
    switch (type) {
    case GgufType::IQ4_XS: {
        const double d = h(b);
        const std::uint16_t scales_h = u16(b + 2);
        const std::uint8_t* qs = b + 8;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((b[4 + ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
            const double dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                out[j] = dl * kvalues_iq4nl[qs[j] & 0xf];
                out[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }
            out += 32;
            qs += 16;
        }
    } break;
    case GgufType::IQ3_S: {
        const double d = h(b);
        const std::uint8_t* qs = b + 2;
        const std::uint8_t* qh = b + 66;
        const std::uint8_t* signs = b + 74;
        const std::uint8_t* scales = b + 106;
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const double db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xf));
            const double db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4));
            for (int half = 0; half < 2; ++half) {
                const double db = half == 0 ? db1 : db2;
                for (int l = 0; l < 4; ++l) {
                    const std::uint32_t g1 = iq3s_grid[qs[2 * l] | ((qh[half] << (8 - 2 * l)) & 256)];
                    const std::uint32_t g2 = iq3s_grid[qs[2 * l + 1] | ((qh[half] << (7 - 2 * l)) & 256)];
                    for (int j = 0; j < 4; ++j) {
                        out[j] = db * grid_byte(g1, j) * (signs[l] & kmask[j] ? -1 : 1);
                        out[j + 4] = db * grid_byte(g2, j) * (signs[l] & kmask[j + 4] ? -1 : 1);
                    }
                    out += 8;
                }
                qs += 8;
                signs += 4;
            }
            qh += 2;
        }
    } break;
    case GgufType::IQ3_XXS: {
        const double d = h(b);
        const std::uint8_t* qs = b + 2;
        const std::uint8_t* ss = b + 66;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const std::uint32_t aux = u32(ss + 4 * ib32);
            const double db = d * (0.5 + (aux >> 28)) * 0.5;
            for (int l = 0; l < 4; ++l) {
                const std::uint8_t signs = ksigns_iq2xs[(aux >> 7 * l) & 127];
                const std::uint32_t g1 = iq3xxs_grid[qs[2 * l]], g2 = iq3xxs_grid[qs[2 * l + 1]];
                for (int j = 0; j < 4; ++j) {
                    out[j] = db * grid_byte(g1, j) * (signs & kmask[j] ? -1 : 1);
                    out[j + 4] = db * grid_byte(g2, j) * (signs & kmask[j + 4] ? -1 : 1);
                }
                out += 8;
            }
            qs += 8;
        }
    } break;
    case GgufType::IQ2_XXS: {
        const double d = h(b);
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const std::uint8_t* aux8 = b + 2 + 8 * ib32;
            const std::uint32_t aux1 = u32(aux8 + 4);
            const double db = d * (0.5 + (aux1 >> 28)) * 0.25;
            for (int l = 0; l < 4; ++l) {
                const std::uint64_t grid = iq2xxs_grid[aux8[l]];
                const std::uint8_t signs = ksigns_iq2xs[(aux1 >> 7 * l) & 127];
                for (int j = 0; j < 8; ++j) {
                    out[j] = db * grid_byte(grid, j) * (signs & kmask[j] ? -1 : 1);
                }
                out += 8;
            }
        }
    } break;
    case GgufType::IQ2_XS: {
        const double d = h(b);
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const double db[2] = {d * (0.5 + (b[66 + ib32] & 0xf)) * 0.25,
                                  d * (0.5 + (b[66 + ib32] >> 4)) * 0.25};
            for (int l = 0; l < 4; ++l) {
                const std::uint16_t q = u16(b + 2 + 2 * (4 * ib32 + l));
                const std::uint64_t grid = iq2xs_grid[q & 511];
                const std::uint8_t signs = ksigns_iq2xs[q >> 9];
                for (int j = 0; j < 8; ++j) {
                    out[j] = db[l / 2] * grid_byte(grid, j) * (signs & kmask[j] ? -1 : 1);
                }
                out += 8;
            }
        }
    } break;
    case GgufType::IQ2_S: {
        const double d = h(b);
        const std::uint8_t* qs = b + 2;
        const std::uint8_t* signs = b + 34;
        const std::uint8_t* qh = b + 66;
        const std::uint8_t* scales = b + 74;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const double db[2] = {d * (0.5 + (scales[ib32] & 0xf)) * 0.25,
                                  d * (0.5 + (scales[ib32] >> 4)) * 0.25};
            for (int l = 0; l < 4; ++l) {
                const std::uint64_t grid = iq2s_grid[qs[l] | (qh[ib32] << (8 - 2 * l) & 0x300)];
                for (int j = 0; j < 8; ++j) {
                    out[j] = db[l / 2] * grid_byte(grid, j) * (signs[l] & kmask[j] ? -1 : 1);
                }
                out += 8;
            }
            qs += 4;
            signs += 4;
        }
    } break;
    case GgufType::IQ1_M: {
        const std::uint8_t* qs = b;
        const std::uint8_t* qh = b + 32;
        const std::uint16_t sc[4] = {u16(b + 48), u16(b + 50), u16(b + 52), u16(b + 54)};
        const std::uint16_t bits = std::uint16_t((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                                                 ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
        std::uint8_t d_bytes[2] = {std::uint8_t(bits), std::uint8_t(bits >> 8)};
        const double d = h(d_bytes);
        for (int ib = 0; ib < 8; ++ib) {
            const double dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
            const double dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
            const int idx[4] = {qs[0] | ((qh[0] << 8) & 0x700), qs[1] | ((qh[0] << 4) & 0x700),
                                qs[2] | ((qh[1] << 8) & 0x700), qs[3] | ((qh[1] << 4) & 0x700)};
            const double delta[4] = {qh[0] & 0x08 ? -0.125 : 0.125, qh[0] & 0x80 ? -0.125 : 0.125,
                                     qh[1] & 0x08 ? -0.125 : 0.125, qh[1] & 0x80 ? -0.125 : 0.125};
            for (int l = 0; l < 4; ++l) {
                const std::uint64_t grid = iq1s_grid[idx[l]];
                for (int j = 0; j < 8; ++j) {
                    out[j] = (l < 2 ? dl1 : dl2) * (std::int8_t(grid_byte(grid, j)) + delta[l]);
                }
                out += 8;
            }
            qs += 4;
            qh += 2;
        }
    } break;
    case GgufType::Q2_K: {
        const double d = h(b + 80), min = h(b + 82);
        const std::uint8_t* q = b + 16;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                std::uint8_t sc = b[is++];
                double dl = d * (sc & 0xF), ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) { *out++ = dl * ((q[l] >> shift) & 3) - ml; }
                sc = b[is++];
                dl = d * (sc & 0xF);
                ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) { *out++ = dl * ((q[l + 16] >> shift) & 3) - ml; }
                shift += 2;
            }
            q += 32;
        }
    } break;
    case GgufType::Q4_K: {
        const double d = h(b), min = h(b + 2);
        const std::uint8_t* scales = b + 4;
        const std::uint8_t* q = b + 16;
        const auto scale_min = [&](int j, int& s, int& m) {
            if (j < 4) {
                s = scales[j] & 63;
                m = scales[j + 4] & 63;
            } else {
                s = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
                m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
            }
        };
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            int s, m;
            scale_min(is, s, m);
            const double d1 = d * s, m1 = min * m;
            scale_min(is + 1, s, m);
            const double d2 = d * s, m2 = min * m;
            for (int l = 0; l < 32; ++l) { *out++ = d1 * (q[l] & 0xF) - m1; }
            for (int l = 0; l < 32; ++l) { *out++ = d2 * (q[l] >> 4) - m2; }
            q += 32;
            is += 2;
        }
    } break;
    case GgufType::Q6_K: {
        const double d = h(b + 208);
        const std::uint8_t* ql = b;
        const std::uint8_t* qh = b + 128;
        const std::int8_t* sc = reinterpret_cast<const std::int8_t*>(b + 192);
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                out[l] = d * sc[is] * q1;
                out[l + 32] = d * sc[is + 2] * q2;
                out[l + 64] = d * sc[is + 4] * q3;
                out[l + 96] = d * sc[is + 6] * q4;
            }
            out += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    } break;
    }
    return y;
}

// Random codes with finite, realistic super-scales at each format's scale fields.
void random_block(GgufType type, std::uint8_t* b, std::mt19937& rng) {
    const int bytes = gguf_block_bytes(type);
    for (int i = 0; i < bytes; ++i) { b[i] = std::uint8_t(rng()); }
    const float d = 0.0005F * float(1 + rng() % 17);
    switch (type) {
    case GgufType::IQ1_M: {
        const __half half = __float2half_rn(d * 4);
        std::uint16_t bits;
        std::memcpy(&bits, &half, 2);
        for (int i = 0; i < 4; ++i) {
            std::uint16_t word = u16(b + 48 + 2 * i);
            word = std::uint16_t((word & 0x0fff) | (((bits >> (4 * i)) & 15) << 12));
            b[48 + 2 * i] = std::uint8_t(word);
            b[49 + 2 * i] = std::uint8_t(word >> 8);
        }
    } break;
    case GgufType::Q2_K:
        put_half(b + 80, d);
        put_half(b + 82, d * 0.5F);
        break;
    case GgufType::Q4_K:
        put_half(b, d);
        put_half(b + 2, d * 0.5F);
        break;
    case GgufType::Q6_K:
        put_half(b + 208, d * 0.05F);
        break;
    default:
        put_half(b, d);
        break;
    }
}

struct HostWeight {
    std::vector<GgufType> types;
    std::vector<int> rows;
    std::vector<std::vector<std::uint8_t>> bytes;
    std::vector<std::unique_ptr<DeviceBuffer>> device;
    int n = 0, k = 0;

    Weight weight() const {
        Weight w;
        w.qtype = QType::GGUF;
        w.layout = QuantLayout::GgufBlocks;
        w.n = w.shape[0] = w.padded_shape[0] = n;
        w.k = w.shape[1] = w.padded_shape[1] = k;
        w.ndim = 2;
        w.gguf_segment_count = static_cast<int>(types.size());
        for (std::size_t s = 0; s < types.size(); ++s) {
            w.gguf_segments[s] = {device[s]->p, rows[s], types[s]};
        }
        w.payload = w.qdata = device[0]->p;
        return w;
    }

    std::vector<double> row(int r) const {
        std::size_t s = 0;
        while (r >= rows[s]) { r -= rows[s++]; }
        const int block_bytes = gguf_block_bytes(types[s]);
        std::vector<double> out(k);
        for (int block = 0; block < k / 256; ++block) {
            const auto values = oracle(types[s], bytes[s].data() +
                                                     (std::size_t(r) * (k / 256) + block) * block_bytes);
            std::copy(values.begin(), values.end(), out.begin() + block * 256);
        }
        return out;
    }
};

HostWeight make_weight(const std::vector<std::pair<GgufType, int>>& segments, int k,
                       std::mt19937& rng) {
    HostWeight w;
    w.k = k;
    for (const auto& [type, rows] : segments) {
        w.types.push_back(type);
        w.rows.push_back(rows);
        w.n += rows;
        std::vector<std::uint8_t> bytes(std::size_t(rows) * (k / 256) * gguf_block_bytes(type));
        for (std::size_t offset = 0; offset < bytes.size(); offset += gguf_block_bytes(type)) {
            random_block(type, bytes.data() + offset, rng);
        }
        w.device.push_back(std::make_unique<DeviceBuffer>(bytes.size()));
        w.device.back()->copy_from_host(bytes.data(), bytes.size());
        w.bytes.push_back(std::move(bytes));
    }
    return w;
}

const char* name(GgufType type) {
    static const char* names[] = {"IQ4_XS", "IQ3_S", "IQ3_XXS", "IQ2_XS", "IQ2_XXS",
                                  "IQ2_S",  "IQ1_M", "Q2_K",    "Q4_K",   "Q6_K"};
    return names[static_cast<int>(type)];
}

int tiled_input(int column, int k) {
    const int key_heads = k / 384;
    return ((column / 128 % key_heads) * 3 + column / (128 * key_heads)) * 128 + column % 128;
}

// Projects through `sections` output tensors and checks every sampled row against FP64.
// The A8 vector route's represented activation: each consecutive 32-value group of the BF16 input
// (in stored column order) becomes step * round(v / step) with step = amax / 127, rounded half to
// even as the device's __float2int_rn does. Defined from the Op contract, not the kernel.
std::vector<double> represented_a8(const std::vector<__nv_bfloat16>& input, int k, int tokens,
                                   bool tiled) {
    std::vector<double> out(input.size());
    for (int t = 0; t < tokens; ++t) {
        for (int group = 0; group < k; group += 32) {
            float amax = 0.0F;
            for (int j = group; j < group + 32; ++j) {
                const int index = tiled ? tiled_input(j, k) : j;
                amax = std::max(amax, std::abs(__bfloat162float(input[t * k + index])));
            }
            const float step    = amax / 127.0F;
            const float inverse = amax > 0.0F ? 127.0F / amax : 0.0F;
            for (int j = group; j < group + 32; ++j) {
                const int index = tiled ? tiled_input(j, k) : j;
                const float v   = __bfloat162float(input[t * k + index]);
                out[t * k + j]  = double(step) * std::nearbyint(v * inverse);
            }
        }
    }
    return out;
}

void check_projection(const HostWeight& host, int tokens, int sections, bool add, bool tiled,
                      bool with_workspace, std::mt19937& rng, const std::string& label,
                      bool allow_a8 = false) {
    const int n = host.n, k = host.k;
    std::vector<__nv_bfloat16> input(std::size_t(k) * tokens);
    for (auto& value : input) { value = __float2bfloat16_rn(float(int(rng() % 2001) - 1000) / 500.0F); }
    std::vector<__nv_bfloat16> initial(std::size_t(n) * tokens);
    for (auto& value : initial) { value = __float2bfloat16_rn(float(int(rng() % 2001) - 1000) / 100.0F); }
    DeviceBuffer x(input.size() * 2);
    x.copy_from_host(input.data(), x.bytes);
    // Each section is its own [rows, T] tensor with a padded column stride, as a fused parent's
    // consumers pass them.
    std::vector<int> section_rows(sections, n / sections);
    section_rows.back() += n - (n / sections) * sections;
    std::vector<std::unique_ptr<DeviceBuffer>> buffers;
    std::vector<Tensor> outputs;
    std::vector<int> strides;
    for (int s = 0; s < sections; ++s) {
        const int stride = section_rows[s] + 8;
        strides.push_back(stride);
        buffers.push_back(std::make_unique<DeviceBuffer>(std::size_t(stride) * tokens * 2));
        std::vector<__nv_bfloat16> fill(std::size_t(stride) * tokens);
        int base = 0;
        for (int i = 0; i < s; ++i) { base += section_rows[i]; }
        for (int t = 0; t < tokens; ++t) {
            for (int r = 0; r < section_rows[s]; ++r) { fill[t * stride + r] = initial[t * n + base + r]; }
        }
        buffers.back()->copy_from_host(fill.data(), buffers.back()->bytes);
        Tensor tensor(buffers.back()->p, DType::BF16, {stride, tokens});
        outputs.push_back(tensor.slice(0, 0, section_rows[s]));
    }
    const Weight weight = host.weight();
    std::unique_ptr<WorkspaceArena> workspace;
    if (with_workspace || allow_a8) {
        const std::size_t bytes = ops::detail::gguf_workspace_bytes(n, k, tokens);
        workspace = std::make_unique<WorkspaceArena>(std::max<std::size_t>(bytes, 256));
    }
    Tensor xt(x.p, DType::BF16, {k, tokens});
    ops::detail::gguf_project(xt, weight, outputs.data(), sections, add, tiled, allow_a8,
                              workspace.get(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    const bool gemm_route = tokens > 16 && with_workspace;
    const bool a8_route   = allow_a8 && !gemm_route;
    const std::vector<double> quantized =
        a8_route ? represented_a8(input, k, tokens, tiled) : std::vector<double>{};
    double error2 = 0, reference2 = 0, worst = 0;
    const int samples = std::min(n, 64);
    for (int sample = 0; sample < samples; ++sample) {
        const int row = sample == samples - 1 ? n - 1 : sample * (n / samples);
        int s = 0, local = row;
        while (local >= section_rows[s]) { local -= section_rows[s++]; }
        std::vector<__nv_bfloat16> column(std::size_t(strides[s]) * tokens);
        buffers[s]->copy_to_host(column.data(), buffers[s]->bytes);
        const auto decoded = host.row(row);
        for (int t = 0; t < tokens; ++t) {
            double reference = add ? double(__bfloat162float(initial[t * n + row])) : 0.0;
            double magnitude = 0.0;
            for (int j = 0; j < k; ++j) {
                const int index = tiled ? tiled_input(j, k) : j;
                const double activation = a8_route ? quantized[t * k + j]
                                                   : double(__bfloat162float(input[t * k + index]));
                const double product = decoded[j] * activation;
                reference += product;
                magnitude += std::abs(product);
            }
            const double actual = __bfloat162float(column[t * strides[s] + local]);
            if (!std::isfinite(actual)) { throw std::runtime_error(label + ": non-finite result"); }
            error2 += (actual - reference) * (actual - reference);
            reference2 += reference * reference;
            worst = std::max(worst, std::abs(actual - reference));
            // One BF16 rounding of the result, FP32 accumulation over K products (and, on the A8
            // route, the FP32 product of the two scales per 8-value term), and on the tensor-core
            // route one FP16 rounding of each weight and activation operand.
            const double tolerance = std::ldexp(std::abs(reference), -8) +
                                     (gemm_route ? std::ldexp(magnitude, -10) : 2e-5 * magnitude) +
                                     (add ? std::ldexp(std::abs(reference), -8) : 0.0);
            if (std::abs(actual - reference) > tolerance) {
                throw std::runtime_error(label + ": row " + std::to_string(row) + " column " +
                                         std::to_string(t) + " differs from the FP64 oracle (" +
                                         std::to_string(actual) + " vs " +
                                         std::to_string(reference) + ")");
            }
        }
    }
    const double relative = std::sqrt(error2 / reference2);
    std::cout << label << " N=" << n << " K=" << k << " T=" << tokens << " sections=" << sections
              << " add=" << add << " tiled=" << tiled << " ws=" << with_workspace
              << " a8=" << a8_route
              << " relative_l2=" << relative << '\n';
    if (!(relative < 4e-3)) { throw std::runtime_error(label + ": FP64 oracle mismatch"); }
}

void check_embedding(const HostWeight& host, const std::string& label) {
    const std::vector<std::int32_t> ids{0, host.n - 1, host.n / 2, 1, host.rows[0] - 1};
    DeviceBuffer id_buffer(ids.size() * 4), out(ids.size() * host.k * 2);
    id_buffer.copy_from_host(ids.data(), id_buffer.bytes);
    Tensor id_tensor(id_buffer.p, DType::I32, {static_cast<int>(ids.size())});
    Tensor output(out.p, DType::BF16, {host.k, static_cast<int>(ids.size())});
    ops::detail::gguf_embedding(id_tensor, host.weight(), output, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> values(ids.size() * host.k);
    out.copy_to_host(values.data(), out.bytes);
    for (std::size_t t = 0; t < ids.size(); ++t) {
        const auto decoded = host.row(ids[t]);
        for (int j = 0; j < host.k; ++j) {
            const __nv_bfloat16 expected = __float2bfloat16_rn(static_cast<float>(decoded[j]));
            if (std::memcmp(&expected, &values[t * host.k + j], 2) != 0) {
                throw std::runtime_error(label + ": embedding differs from the exact BF16 value");
            }
        }
    }
    // The FP32 dequantization is the exact value (every represented value is an FP32 product of
    // an FP16 scale, a small integer scale and an int8 code).
    std::vector<float> full(std::size_t(host.n) * host.k);
    DeviceBuffer full_device(full.size() * 4);
    ops::detail::gguf_dequantize_fp32(host.weight(), static_cast<float*>(full_device.p), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    full_device.copy_to_host(full.data(), full_device.bytes);
    for (int row : {0, host.n - 1, host.n / 3}) {
        const auto decoded = host.row(row);
        for (int j = 0; j < host.k; ++j) {
            if (std::abs(double(full[std::size_t(row) * host.k + j]) - decoded[j]) >
                1e-6 * std::abs(decoded[j]) + 1e-12) {
                throw std::runtime_error(label + ": FP32 dequantization differs from ggml");
            }
        }
    }
    std::cout << label << " embedding and FP32 decode exact\n";
}

} // namespace

int main() {
    try {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) { return 77; }
        std::mt19937 rng(20260930);
        const GgufType all[] = {GgufType::IQ4_XS, GgufType::IQ3_S, GgufType::IQ3_XXS,
                                GgufType::IQ2_XS, GgufType::IQ2_XXS, GgufType::IQ2_S,
                                GgufType::IQ1_M,  GgufType::Q2_K,  GgufType::Q4_K,
                                GgufType::Q6_K};
        // Every format: exact decode, vector passes at 1..9 columns (two passes at 9), and the
        // tensor-core route above 16 columns.
        for (const GgufType type : all) {
            const HostWeight host = make_weight({{type, 96}}, 1024, rng);
            check_embedding(host, name(type));
            for (const int tokens : {1, 3, 8, 9}) {
                check_projection(host, tokens, 1, false, false, false, rng, name(type));
                check_projection(host, tokens, 1, false, false, false, rng, name(type), true);
            }
            check_projection(host, 40, 1, false, false, true, rng, name(type));
        }
        // Real mixed parents: a GDN q|k|v (IQ3_S) + z (IQ3_XXS) at K=5120 into qkv/z sections,
        // an attention q|k|gate|v with four types and sections, residual accumulation, and the
        // tiled GDN output permutation at full and TP2 K.
        {
            const HostWeight gdn = make_weight(
                {{GgufType::IQ3_S, 5120}, {GgufType::IQ3_XXS, 3072}}, 5120, rng);
            for (const int tokens : {1, 4, 64}) {
                check_projection(gdn, tokens, 2, false, false, tokens > 16, rng, "gdn qkv|z");
                check_projection(gdn, tokens, 2, false, false, tokens > 16, rng, "gdn qkv|z", true);
            }
            const HostWeight attention = make_weight({{GgufType::Q2_K, 1536},
                                                      {GgufType::IQ4_XS, 256},
                                                      {GgufType::Q4_K, 1536},
                                                      {GgufType::IQ3_S, 256}},
                                                     5120, rng);
            for (const int tokens : {2, 33}) {
                check_projection(attention, tokens, 4, false, false, tokens > 16, rng, "attention");
            }
            // Large segments take the multi-row vector kernels, at every vector width.
            const HostWeight wide = make_weight({{GgufType::IQ3_S, 2304}, {GgufType::Q4_K, 4096}},
                                                5120, rng);
            for (const int tokens : {1, 4, 7}) {
                check_projection(wide, tokens, 2, true, false, false, rng, "wide");
                check_projection(wide, tokens, 2, true, false, false, rng, "wide", true);
            }
            const HostWeight output = make_weight({{GgufType::IQ4_XS, 512}}, 6144, rng);
            const HostWeight shard = make_weight({{GgufType::Q4_K, 512}}, 3072, rng);
            for (const int tokens : {1, 5, 24}) {
                check_projection(output, tokens, 1, true, true, tokens > 16, rng, "gdn output");
                check_projection(shard, tokens, 1, true, true, tokens > 16, rng, "gdn output tp2");
                check_projection(shard, tokens, 1, true, true, tokens > 16, rng, "gdn output tp2",
                                 true);
            }
            // A 17-column request without workspace falls back to vector passes.
            check_projection(output, 17, 1, true, false, false, rng, "no workspace");
            const Weight view = ops::detail::gguf_row_view(attention.weight(), 1500, 400);
            if (view.gguf_segment_count != 3 || view.gguf_segments[0].rows != 36 ||
                view.gguf_segments[1].rows != 256 || view.gguf_segments[2].rows != 108) {
                throw std::runtime_error("gguf row view: wrong segment split");
            }
        }
        std::cout << "ninfer_gguf_test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ninfer_gguf_test failed: " << error.what() << '\n';
        return 1;
    }
}
