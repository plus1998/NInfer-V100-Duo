#pragma once

// Device decoders of the GGUF block formats registered for Qwen3.8-27B GGUF-blocks artifacts.
// Each decoder yields the eight consecutive values [8*j8, 8*j8+8) of one 256-value block exactly
// as ggml-quants.c's dequantize_row_* functions define them (FP16 super-scales widened to FP32,
// integer sub-scales and codebook entries multiplied in FP32). Blocks are only two-byte aligned in
// a row (110, 98, 74, 66 and 82 bytes are not multiples of four), so every multi-byte field is
// assembled from 16-bit loads.

#include "core/tensor.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::gguf {

#define NINFER_GGUF_TABLE(type, name, size) __device__ const type name[size] = {
#include "ops/linear/gguf/gguf_tables.inc"
#undef NINFER_GGUF_TABLE

constexpr float kIq1Delta = 0.125F;

__device__ __forceinline__ unsigned u8(const unsigned char* p, int offset) { return p[offset]; }

__device__ __forceinline__ unsigned u16(const unsigned char* p, int offset) {
    return *reinterpret_cast<const unsigned short*>(p + offset);
}

__device__ __forceinline__ unsigned u32(const unsigned char* p, int offset) {
    return u16(p, offset) | (u16(p, offset + 2) << 16);
}

__device__ __forceinline__ float f16(const unsigned char* p, int offset) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(u16(p, offset))));
}

// Codebooks are read-only for the kernel's lifetime; route lookups through the read-only cache.
__device__ __forceinline__ std::uint32_t table32(const std::uint32_t* table, unsigned index) {
    return __ldg(reinterpret_cast<const unsigned int*>(table) + index);
}

__device__ __forceinline__ std::uint64_t table64(const std::uint64_t* table, unsigned index) {
    return __ldg(reinterpret_cast<const unsigned long long*>(table) + index);
}

__device__ __forceinline__ unsigned signs7(unsigned index) {
    // ksigns_iq2xs[i] is i with bit 7 set to the parity of i's seven bits (verified against the
    // table in tests/ops/test_gguf.cpp through the decoded values).
    return index | ((__popc(index) & 1U) << 7);
}

// Eight unsigned (or two's-complement) grid bytes, applied with a sign mask and one scale.
__device__ __forceinline__ void grid8(std::uint64_t grid, unsigned signs, float scale,
                                      float (&w)[8]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float magnitude = static_cast<float>((grid >> (8 * j)) & 255U);
        w[j] = scale * ((signs >> j) & 1U ? -magnitude : magnitude);
    }
}

__device__ __forceinline__ void grid4x2(std::uint32_t first, std::uint32_t second, unsigned signs,
                                        float scale, float (&w)[8]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float a = static_cast<float>((first >> (8 * j)) & 255U);
        const float b = static_cast<float>((second >> (8 * j)) & 255U);
        w[j]     = scale * ((signs >> j) & 1U ? -a : a);
        w[j + 4] = scale * ((signs >> (j + 4)) & 1U ? -b : b);
    }
}

template <GgufType Type>
__device__ __forceinline__ void decode8(const unsigned char* block, int j8, float (&w)[8]);

// block_iq4_xs: half d, u16 scales_h, u8 scales_l[4], u8 qs[128].
template <>
__device__ __forceinline__ void decode8<GgufType::IQ4_XS>(const unsigned char* b, int j8,
                                                          float (&w)[8]) {
    const int ib         = j8 >> 2;
    const unsigned low   = (u8(b, 4 + ib / 2) >> (4 * (ib & 1))) & 15U;
    const unsigned high  = (u16(b, 2) >> (2 * ib)) & 3U;
    const float scale    = f16(b, 0) * static_cast<float>(static_cast<int>(low | (high << 4)) - 32);
    const int within     = (j8 & 3) * 8; // [0,32) inside the 32-value sub-block
    const int byte_base  = 8 + ib * 16 + (within & 15);
    const int shift      = within < 16 ? 0 : 4;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const unsigned code = (u8(b, byte_base + j) >> shift) & 15U;
        w[j]                = scale * static_cast<float>(
                   static_cast<signed char>(__ldg(reinterpret_cast<const signed char*>(kvalues_iq4nl) + code)));
    }
}

// block_iq3_s: half d, u8 qs[64], u8 qh[8], u8 signs[32], u8 scales[4].
template <>
__device__ __forceinline__ void decode8<GgufType::IQ3_S>(const unsigned char* b, int j8,
                                                         float (&w)[8]) {
    const int ib32 = j8 >> 2;
    const int l    = j8 & 3;
    const float scale =
        f16(b, 0) * static_cast<float>(1 + 2 * ((u8(b, 106 + ib32 / 2) >> (4 * (ib32 & 1))) & 15U));
    const unsigned qh = u8(b, 66 + ib32);
    const unsigned i0 = u8(b, 2 + ib32 * 8 + 2 * l) | ((qh << (8 - 2 * l)) & 256U);
    const unsigned i1 = u8(b, 2 + ib32 * 8 + 2 * l + 1) | ((qh << (7 - 2 * l)) & 256U);
    grid4x2(table32(iq3s_grid, i0), table32(iq3s_grid, i1), u8(b, 74 + ib32 * 4 + l), scale, w);
}

// block_iq3_xxs: half d, u8 qs[64] grid indices, u32 scales_and_signs[8].
template <>
__device__ __forceinline__ void decode8<GgufType::IQ3_XXS>(const unsigned char* b, int j8,
                                                           float (&w)[8]) {
    const int ib32      = j8 >> 2;
    const int l         = j8 & 3;
    const unsigned aux  = u32(b, 66 + 4 * ib32);
    const float scale   = f16(b, 0) * (0.5F + static_cast<float>(aux >> 28)) * 0.5F;
    const unsigned sign = signs7((aux >> (7 * l)) & 127U);
    grid4x2(table32(iq3xxs_grid, u8(b, 2 + ib32 * 8 + 2 * l)),
            table32(iq3xxs_grid, u8(b, 2 + ib32 * 8 + 2 * l + 1)), sign, scale, w);
}

// block_iq2_xxs: half d, u16 qs[32] = per 32 values u8 grid[4] + u32 signs/scale.
template <>
__device__ __forceinline__ void decode8<GgufType::IQ2_XXS>(const unsigned char* b, int j8,
                                                           float (&w)[8]) {
    const int ib32      = j8 >> 2;
    const int l         = j8 & 3;
    const unsigned aux  = u32(b, 2 + 8 * ib32 + 4);
    const float scale   = f16(b, 0) * (0.5F + static_cast<float>(aux >> 28)) * 0.25F;
    const unsigned sign = signs7((aux >> (7 * l)) & 127U);
    grid8(table64(iq2xxs_grid, u8(b, 2 + 8 * ib32 + l)), sign, scale, w);
}

// block_iq2_xs: half d, u16 qs[32] (9-bit grid index | 7-bit sign index), u8 scales[8].
template <>
__device__ __forceinline__ void decode8<GgufType::IQ2_XS>(const unsigned char* b, int j8,
                                                          float (&w)[8]) {
    const int ib32     = j8 >> 2;
    const int l        = j8 & 3;
    const unsigned q   = u16(b, 2 + 2 * j8);
    const unsigned sub = (u8(b, 66 + ib32) >> (4 * (l / 2))) & 15U;
    const float scale  = f16(b, 0) * (0.5F + static_cast<float>(sub)) * 0.25F;
    grid8(table64(iq2xs_grid, q & 511U), signs7(q >> 9), scale, w);
}

// block_iq2_s: half d, u8 qs[32] grid low bits, u8 signs[32], u8 qh[8], u8 scales[8].
template <>
__device__ __forceinline__ void decode8<GgufType::IQ2_S>(const unsigned char* b, int j8,
                                                         float (&w)[8]) {
    const int ib32     = j8 >> 2;
    const int l        = j8 & 3;
    const unsigned sub = (u8(b, 74 + ib32) >> (4 * (l / 2))) & 15U;
    const float scale  = f16(b, 0) * (0.5F + static_cast<float>(sub)) * 0.25F;
    const unsigned idx = u8(b, 2 + j8) | ((u8(b, 66 + ib32) << (8 - 2 * l)) & 0x300U);
    grid8(table64(iq2s_grid, idx), u8(b, 34 + j8), scale, w);
}

// block_iq1_m: u8 qs[32], u8 qh[16], u16 scales[4] (3-bit sub-scales, FP16 super-scale spread
// over the top nibbles).
template <>
__device__ __forceinline__ void decode8<GgufType::IQ1_M>(const unsigned char* b, int j8,
                                                         float (&w)[8]) {
    const int ib = j8 >> 2;
    const int l  = j8 & 3;
    const unsigned s0 = u16(b, 48), s1 = u16(b, 50), s2 = u16(b, 52), s3 = u16(b, 54);
    const unsigned bits =
        (s0 >> 12) | ((s1 >> 8) & 0x00F0U) | ((s2 >> 4) & 0x0F00U) | (s3 & 0xF000U);
    const float d       = __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
    const unsigned sc   = u16(b, 48 + 2 * (ib / 2));
    const int shift     = 6 * (ib & 1) + (l < 2 ? 0 : 3);
    const float scale   = d * static_cast<float>(2 * ((sc >> shift) & 7U) + 1);
    const unsigned qh   = u8(b, 32 + 2 * ib + l / 2);
    const unsigned high = (l & 1) == 0 ? (qh << 8) & 0x700U : (qh << 4) & 0x700U;
    const unsigned idx  = u8(b, 4 * ib + l) | high;
    const float delta   = (qh & ((l & 1) == 0 ? 0x08U : 0x80U)) != 0 ? -kIq1Delta : kIq1Delta;
    const std::uint64_t grid = table64(iq1s_grid, idx);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float g = static_cast<float>(static_cast<signed char>((grid >> (8 * j)) & 255U));
        w[j]          = scale * (g + delta);
    }
}

// block_q2_K: u8 scales[16] (low nibble scale, high nibble min), u8 qs[64], half d, half dmin.
template <>
__device__ __forceinline__ void decode8<GgufType::Q2_K>(const unsigned char* b, int j8,
                                                        float (&w)[8]) {
    const int i      = 8 * j8;
    const int half   = i >> 7;
    const int rem    = i & 127;
    const int j      = rem >> 5;
    const int within = rem & 31;
    const unsigned sc = u8(b, half * 8 + j * 2 + within / 16);
    const float scale = f16(b, 80) * static_cast<float>(sc & 15U);
    const float min   = f16(b, 82) * static_cast<float>(sc >> 4);
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const unsigned q = (u8(b, 16 + half * 32 + within + v) >> (2 * j)) & 3U;
        w[v]             = scale * static_cast<float>(q) - min;
    }
}

// block_q4_K: half d, half dmin, u8 scales[12] (6-bit scale/min pairs), u8 qs[128].
template <>
__device__ __forceinline__ void decode8<GgufType::Q4_K>(const unsigned char* b, int j8,
                                                        float (&w)[8]) {
    const int group = j8 >> 2;
    unsigned scale_code;
    unsigned min_code;
    if (group < 4) {
        scale_code = u8(b, 4 + group) & 63U;
        min_code   = u8(b, 8 + group) & 63U;
    } else {
        scale_code = (u8(b, 4 + group + 4) & 15U) | ((u8(b, 4 + group - 4) >> 6) << 4);
        min_code   = (u8(b, 4 + group + 4) >> 4) | ((u8(b, 4 + group) >> 6) << 4);
    }
    const float scale = f16(b, 0) * static_cast<float>(scale_code);
    const float min   = f16(b, 2) * static_cast<float>(min_code);
    const int base    = 16 + (group >> 1) * 32 + (j8 & 3) * 8;
    const int shift   = 4 * (group & 1);
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        w[v] = scale * static_cast<float>((u8(b, base + v) >> shift) & 15U) - min;
    }
}

// block_q6_K: u8 ql[128], u8 qh[64], i8 scales[16], half d.
template <>
__device__ __forceinline__ void decode8<GgufType::Q6_K>(const unsigned char* b, int j8,
                                                        float (&w)[8]) {
    const int i       = 8 * j8;
    const int half    = i >> 7;
    const int within  = i & 127;
    const int section = within >> 5;
    const int lane    = within & 31;
    const float scale =
        f16(b, 208) * static_cast<float>(static_cast<signed char>(u8(b, 192 + i / 16)));
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const unsigned low =
            (u8(b, half * 64 + (section & 1) * 32 + lane + v) >> ((section >> 1) * 4)) & 15U;
        const unsigned high = (u8(b, 128 + half * 32 + lane + v) >> (section * 2)) & 3U;
        w[v] = scale * static_cast<float>(static_cast<int>(low | (high << 4)) - 32);
    }
}

// --- int8 decoders for the A8 vector route -----------------------------------------------------
// decode_i8 yields the same eight values as decode8 as `scale * code - minimum`, with the eight
// integer codes packed as signed bytes into two words (values 0..3 in w[0], 4..7 in w[1]), so a dot
// with int8 activations is two dp4a. Every code of the registered formats fits int8.

template <GgufType Type>
inline constexpr bool kHasMinimum = Type == GgufType::Q2_K || Type == GgufType::Q4_K ||
                             Type == GgufType::IQ1_M;

// Expands four sign bits into byte masks (0x00 or 0xff per byte) and negates the selected bytes of
// a word of non-negative grid values.
__device__ __forceinline__ int apply_signs4(std::uint32_t grid, unsigned bits) {
    const unsigned mask = (((bits & 15U) * 0x00204081U) & 0x01010101U) * 0xffU;
    return static_cast<int>(__vsub4(grid ^ mask, mask));
}

// kvalues_iq4nl as little-endian signed bytes.
constexpr std::uint32_t pack4(int a, int b, int c, int d) {
    return (static_cast<std::uint32_t>(a & 255)) | (static_cast<std::uint32_t>(b & 255) << 8) |
           (static_cast<std::uint32_t>(c & 255) << 16) | (static_cast<std::uint32_t>(d & 255) << 24);
}
constexpr std::uint32_t kIq4Words[4] = {pack4(-127, -104, -83, -65), pack4(-49, -35, -22, -10),
                                        pack4(1, 13, 25, 38), pack4(53, 69, 89, 113)};

// Looks up four 4-bit codes (one per byte, 0..15) in kvalues_iq4nl.
__device__ __forceinline__ int iq4_lookup4(unsigned codes) {
    const unsigned low3     = codes & 0x07070707U;
    const unsigned packed   = low3 | (low3 >> 4);
    const unsigned selector = (packed & 0xffU) | ((packed >> 8) & 0xff00U);
    const unsigned low      = __byte_perm(kIq4Words[0], kIq4Words[1], selector);
    const unsigned high     = __byte_perm(kIq4Words[2], kIq4Words[3], selector);
    const unsigned pick     = ((codes >> 3) & 0x01010101U) * 0xffU;
    return static_cast<int>((low & ~pick) | (high & pick));
}

template <GgufType Type>
__device__ __forceinline__ void decode_i8(const unsigned char* block, int j8, int (&w)[2],
                                          float& scale, float& minimum);

template <>
__device__ __forceinline__ void decode_i8<GgufType::IQ4_XS>(const unsigned char* b, int j8,
                                                            int (&w)[2], float& scale, float&) {
    const int ib        = j8 >> 2;
    const unsigned low  = (u8(b, 4 + ib / 2) >> (4 * (ib & 1))) & 15U;
    const unsigned high = (u16(b, 2) >> (2 * ib)) & 3U;
    scale               = f16(b, 0) * static_cast<float>(static_cast<int>(low | (high << 4)) - 32);
    const int within    = (j8 & 3) * 8;
    const int byte_base = 8 + ib * 16 + (within & 15);
    const int shift     = within < 16 ? 0 : 4;
    w[0] = iq4_lookup4((u32(b, byte_base) >> shift) & 0x0f0f0f0fU);
    w[1] = iq4_lookup4((u32(b, byte_base + 4) >> shift) & 0x0f0f0f0fU);
}

template <>
__device__ __forceinline__ void decode_i8<GgufType::IQ3_S>(const unsigned char* b, int j8,
                                                           int (&w)[2], float& scale, float&) {
    const int ib32 = j8 >> 2;
    const int l    = j8 & 3;
    scale = f16(b, 0) * static_cast<float>(1 + 2 * ((u8(b, 106 + ib32 / 2) >> (4 * (ib32 & 1))) & 15U));
    const unsigned qh    = u8(b, 66 + ib32);
    const unsigned i0    = u8(b, 2 + ib32 * 8 + 2 * l) | ((qh << (8 - 2 * l)) & 256U);
    const unsigned i1    = u8(b, 2 + ib32 * 8 + 2 * l + 1) | ((qh << (7 - 2 * l)) & 256U);
    const unsigned signs = u8(b, 74 + ib32 * 4 + l);
    w[0] = apply_signs4(table32(iq3s_grid, i0), signs);
    w[1] = apply_signs4(table32(iq3s_grid, i1), signs >> 4);
}

template <>
__device__ __forceinline__ void decode_i8<GgufType::IQ3_XXS>(const unsigned char* b, int j8,
                                                             int (&w)[2], float& scale, float&) {
    const int ib32       = j8 >> 2;
    const int l          = j8 & 3;
    const unsigned aux   = u32(b, 66 + 4 * ib32);
    scale                = f16(b, 0) * (0.5F + static_cast<float>(aux >> 28)) * 0.5F;
    const unsigned signs = signs7((aux >> (7 * l)) & 127U);
    w[0] = apply_signs4(table32(iq3xxs_grid, u8(b, 2 + ib32 * 8 + 2 * l)), signs);
    w[1] = apply_signs4(table32(iq3xxs_grid, u8(b, 2 + ib32 * 8 + 2 * l + 1)), signs >> 4);
}

__device__ __forceinline__ void grid8_i8(std::uint64_t grid, unsigned signs, int (&w)[2]) {
    w[0] = apply_signs4(static_cast<std::uint32_t>(grid), signs);
    w[1] = apply_signs4(static_cast<std::uint32_t>(grid >> 32), signs >> 4);
}

template <>
__device__ __forceinline__ void decode_i8<GgufType::IQ2_XXS>(const unsigned char* b, int j8,
                                                             int (&w)[2], float& scale, float&) {
    const int ib32     = j8 >> 2;
    const int l        = j8 & 3;
    const unsigned aux = u32(b, 2 + 8 * ib32 + 4);
    scale              = f16(b, 0) * (0.5F + static_cast<float>(aux >> 28)) * 0.25F;
    grid8_i8(table64(iq2xxs_grid, u8(b, 2 + 8 * ib32 + l)), signs7((aux >> (7 * l)) & 127U), w);
}

template <>
__device__ __forceinline__ void decode_i8<GgufType::IQ2_XS>(const unsigned char* b, int j8,
                                                            int (&w)[2], float& scale, float&) {
    const int ib32     = j8 >> 2;
    const int l        = j8 & 3;
    const unsigned q   = u16(b, 2 + 2 * j8);
    const unsigned sub = (u8(b, 66 + ib32) >> (4 * (l / 2))) & 15U;
    scale              = f16(b, 0) * (0.5F + static_cast<float>(sub)) * 0.25F;
    grid8_i8(table64(iq2xs_grid, q & 511U), signs7(q >> 9), w);
}

template <>
__device__ __forceinline__ void decode_i8<GgufType::IQ2_S>(const unsigned char* b, int j8,
                                                           int (&w)[2], float& scale, float&) {
    const int ib32     = j8 >> 2;
    const int l        = j8 & 3;
    const unsigned sub = (u8(b, 74 + ib32) >> (4 * (l / 2))) & 15U;
    scale              = f16(b, 0) * (0.5F + static_cast<float>(sub)) * 0.25F;
    const unsigned idx = u8(b, 2 + j8) | ((u8(b, 66 + ib32) << (8 - 2 * l)) & 0x300U);
    grid8_i8(table64(iq2s_grid, idx), u8(b, 34 + j8), w);
}

template <>
__device__ __forceinline__ void decode_i8<GgufType::IQ1_M>(const unsigned char* b, int j8,
                                                           int (&w)[2], float& scale,
                                                           float& minimum) {
    const int ib = j8 >> 2;
    const int l  = j8 & 3;
    const unsigned s0 = u16(b, 48), s1 = u16(b, 50), s2 = u16(b, 52), s3 = u16(b, 54);
    const unsigned bits =
        (s0 >> 12) | ((s1 >> 8) & 0x00F0U) | ((s2 >> 4) & 0x0F00U) | (s3 & 0xF000U);
    const float d       = __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
    const unsigned sc   = u16(b, 48 + 2 * (ib / 2));
    const int shift     = 6 * (ib & 1) + (l < 2 ? 0 : 3);
    scale               = d * static_cast<float>(2 * ((sc >> shift) & 7U) + 1);
    const unsigned qh   = u8(b, 32 + 2 * ib + l / 2);
    const unsigned high = (l & 1) == 0 ? (qh << 8) & 0x700U : (qh << 4) & 0x700U;
    const float delta   = (qh & ((l & 1) == 0 ? 0x08U : 0x80U)) != 0 ? -kIq1Delta : kIq1Delta;
    // value = scale * (grid + delta) = scale * grid - minimum.
    minimum                  = -scale * delta;
    const std::uint64_t grid = table64(iq1s_grid, u8(b, 4 * ib + l) | high);
    w[0] = static_cast<int>(static_cast<std::uint32_t>(grid));
    w[1] = static_cast<int>(static_cast<std::uint32_t>(grid >> 32));
}

template <>
__device__ __forceinline__ void decode_i8<GgufType::Q2_K>(const unsigned char* b, int j8,
                                                          int (&w)[2], float& scale,
                                                          float& minimum) {
    const int i       = 8 * j8;
    const int half    = i >> 7;
    const int rem     = i & 127;
    const int j       = rem >> 5;
    const int within  = rem & 31;
    const unsigned sc = u8(b, half * 8 + j * 2 + within / 16);
    scale             = f16(b, 80) * static_cast<float>(sc & 15U);
    minimum           = f16(b, 82) * static_cast<float>(sc >> 4);
    const int base    = 16 + half * 32 + within;
    w[0] = static_cast<int>((u32(b, base) >> (2 * j)) & 0x03030303U);
    w[1] = static_cast<int>((u32(b, base + 4) >> (2 * j)) & 0x03030303U);
}

template <>
__device__ __forceinline__ void decode_i8<GgufType::Q4_K>(const unsigned char* b, int j8,
                                                          int (&w)[2], float& scale,
                                                          float& minimum) {
    const int group = j8 >> 2;
    unsigned scale_code;
    unsigned min_code;
    if (group < 4) {
        scale_code = u8(b, 4 + group) & 63U;
        min_code   = u8(b, 8 + group) & 63U;
    } else {
        scale_code = (u8(b, 4 + group + 4) & 15U) | ((u8(b, 4 + group - 4) >> 6) << 4);
        min_code   = (u8(b, 4 + group + 4) >> 4) | ((u8(b, 4 + group) >> 6) << 4);
    }
    scale           = f16(b, 0) * static_cast<float>(scale_code);
    minimum         = f16(b, 2) * static_cast<float>(min_code);
    const int base  = 16 + (group >> 1) * 32 + (j8 & 3) * 8;
    const int shift = 4 * (group & 1);
    w[0] = static_cast<int>((u32(b, base) >> shift) & 0x0f0f0f0fU);
    w[1] = static_cast<int>((u32(b, base + 4) >> shift) & 0x0f0f0f0fU);
}

template <>
__device__ __forceinline__ void decode_i8<GgufType::Q6_K>(const unsigned char* b, int j8,
                                                          int (&w)[2], float& scale, float&) {
    const int i       = 8 * j8;
    const int half    = i >> 7;
    const int within  = i & 127;
    const int section = within >> 5;
    const int lane    = within & 31;
    scale = f16(b, 208) * static_cast<float>(static_cast<signed char>(u8(b, 192 + i / 16)));
    const int low_base  = half * 64 + (section & 1) * 32 + lane;
    const int high_base = 128 + half * 32 + lane;
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const unsigned low  = (u32(b, low_base + 4 * part) >> ((section >> 1) * 4)) & 0x0f0f0f0fU;
        const unsigned high = ((u32(b, high_base + 4 * part) >> (section * 2)) & 0x03030303U) << 4;
        w[part] = static_cast<int>(__vsub4(low | high, 0x20202020U));
    }
}

// Runtime-type dispatch for paths where the type is not a template parameter.
__device__ __forceinline__ void decode8(GgufType type, const unsigned char* block, int j8,
                                        float (&w)[8]) {
    switch (type) {
    case GgufType::IQ4_XS: decode8<GgufType::IQ4_XS>(block, j8, w); return;
    case GgufType::IQ3_S: decode8<GgufType::IQ3_S>(block, j8, w); return;
    case GgufType::IQ3_XXS: decode8<GgufType::IQ3_XXS>(block, j8, w); return;
    case GgufType::IQ2_XS: decode8<GgufType::IQ2_XS>(block, j8, w); return;
    case GgufType::IQ2_XXS: decode8<GgufType::IQ2_XXS>(block, j8, w); return;
    case GgufType::IQ2_S: decode8<GgufType::IQ2_S>(block, j8, w); return;
    case GgufType::IQ1_M: decode8<GgufType::IQ1_M>(block, j8, w); return;
    case GgufType::Q2_K: decode8<GgufType::Q2_K>(block, j8, w); return;
    case GgufType::Q4_K: decode8<GgufType::Q4_K>(block, j8, w); return;
    case GgufType::Q6_K: decode8<GgufType::Q6_K>(block, j8, w); return;
    }
}

} // namespace ninfer::ops::detail::gguf
