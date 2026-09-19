#pragma once

// Test-owned packed-weight fixtures for every registered quantized Weight format. Physical
// row-split and block-scale codecs remain format-specific beneath one logical fixture interface.

#include "core/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::test::quantized_weight {
namespace detail {

inline std::int32_t align_up(std::int32_t x, std::int32_t m) { return ((x + m - 1) / m) * m; }

inline std::size_t align_up_size(std::size_t x, std::size_t m) { return ((x + m - 1) / m) * m; }

inline std::uint64_t mix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

inline std::uint32_t float_bits(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

inline float bits_float(std::uint32_t u) {
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline std::uint32_t round_shift_rne(std::uint32_t v, int shift) {
    if (shift <= 0) { return v; }
    const std::uint32_t mask = (1u << shift) - 1u;
    const std::uint32_t half = 1u << (shift - 1);
    const std::uint32_t base = v >> shift;
    const std::uint32_t rem  = v & mask;
    return base + ((rem > half || (rem == half && (base & 1u))) ? 1u : 0u);
}

inline std::uint16_t f32_to_f16(float f) {
    const std::uint32_t x    = float_bits(f);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    const std::uint32_t ax   = x & 0x7fffffffu;

    if (ax >= 0x7f800000u) {
        const std::uint32_t mant = ax & 0x007fffffu;
        return static_cast<std::uint16_t>(sign | 0x7c00u | (mant ? 0x0200u : 0u));
    }

    int exp            = static_cast<int>((ax >> 23) & 0xffu) - 127 + 15;
    std::uint32_t mant = ax & 0x007fffffu;
    if (exp <= 0) {
        if (exp < -10) { return static_cast<std::uint16_t>(sign); }
        mant |= 0x00800000u;
        const std::uint32_t hm = round_shift_rne(mant, 14 - exp);
        return static_cast<std::uint16_t>(sign | hm);
    }
    if (exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }

    std::uint32_t hm = round_shift_rne(mant, 13);
    if (hm == 0x0400u) {
        hm = 0;
        ++exp;
        if (exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | hm);
}

inline float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    std::uint32_t exp        = (static_cast<std::uint32_t>(h) >> 10) & 0x1fu;
    std::uint32_t mant       = static_cast<std::uint32_t>(h) & 0x03ffu;

    if (exp == 0) {
        if (mant == 0) { return bits_float(sign); }
        int e = -14;
        while ((mant & 0x0400u) == 0) {
            mant <<= 1;
            --e;
        }
        mant &= 0x03ffu;
        return bits_float(sign | (static_cast<std::uint32_t>(e + 127) << 23) | (mant << 13));
    }
    if (exp == 31) { return bits_float(sign | 0x7f800000u | (mant << 13)); }
    exp = exp - 15 + 127;
    return bits_float(sign | (exp << 23) | (mant << 13));
}

inline float bf16_to_f32(std::uint16_t h) {
    return bits_float(static_cast<std::uint32_t>(h) << 16);
}

inline void store_u16_le(std::vector<std::uint8_t>& payload, std::size_t off, std::uint16_t v) {
    payload[off]     = static_cast<std::uint8_t>(v & 0xffu);
    payload[off + 1] = static_cast<std::uint8_t>(v >> 8);
}

inline void store_u32_le(std::vector<std::uint8_t>& payload, std::size_t off, std::uint32_t v) {
    payload[off]     = static_cast<std::uint8_t>(v & 0xffu);
    payload[off + 1] = static_cast<std::uint8_t>((v >> 8) & 0xffu);
    payload[off + 2] = static_cast<std::uint8_t>((v >> 16) & 0xffu);
    payload[off + 3] = static_cast<std::uint8_t>(v >> 24);
}

inline std::uint16_t load_u16_le(const std::vector<std::uint8_t>& payload, std::size_t off) {
    return static_cast<std::uint16_t>(payload[off]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(payload[off + 1]) << 8);
}

inline double decode_e2m1(std::uint8_t word) {
    constexpr double magnitudes[]{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double magnitude = magnitudes[word & 0x07U];
    return (word & 0x08U) == 0 ? magnitude : -magnitude;
}

inline double decode_e4m3fn(std::uint8_t word) {
    const bool negative   = (word & 0x80U) != 0;
    const std::uint32_t e = (word >> 3) & 0x0fU;
    const std::uint32_t m = word & 0x07U;
    double magnitude      = 0.0;
    if (e == 0) {
        magnitude = m == 0 ? 0.0 : static_cast<double>(m) * std::ldexp(1.0, -9);
    } else {
        if (e == 0x0fU && m == 0x07U) {
            throw std::invalid_argument("quantized-weight fixture: E4M3FN NaN scale");
        }
        magnitude = (1.0 + static_cast<double>(m) / 8.0) * std::ldexp(1.0, static_cast<int>(e) - 7);
    }
    return negative ? -magnitude : magnitude;
}

struct QuantSpec {
    int bits;
    int group_size;
    int qmax;
    int qmin;
};

inline QuantSpec quant_spec(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
        return {4, 64, 7, -8};
    case QType::Q5G64_F16S:
        return {5, 64, 15, -16};
    case QType::Q6G64_F16S:
        return {6, 64, 31, -32};
    case QType::W8G32_F16S:
        return {8, 32, 127, -127};
    default:
        throw std::invalid_argument("row-split test packer: unsupported qtype");
    }
}

inline int nibble_bytes_per_group(const QuantSpec& spec) {
    return spec.bits == 8 ? spec.group_size : spec.group_size / 2;
}

inline int high_bytes_per_group(const QuantSpec& spec) {
    if (spec.bits == 8) { return 0; }
    return spec.bits <= 4 ? 0 : spec.group_size * (spec.bits - 4) / 8;
}

inline void pack_lowbit_group(const std::int8_t* codes, const QuantSpec& spec,
                              std::uint8_t* nibble_out, std::uint8_t* high_out) {
    const int nib  = nibble_bytes_per_group(spec);
    const int high = high_bytes_per_group(spec);
    std::fill(nibble_out, nibble_out + nib, static_cast<std::uint8_t>(0));
    if (high != 0) { std::fill(high_out, high_out + high, static_cast<std::uint8_t>(0)); }
    if (spec.bits == 8) {
        for (int i = 0; i < spec.group_size; ++i) {
            nibble_out[i] = static_cast<std::uint8_t>(codes[i]);
        }
        return;
    }
    const std::uint32_t mask = (1u << spec.bits) - 1u;
    for (int i = 0; i < spec.group_size; ++i) {
        const std::uint32_t u   = static_cast<std::uint32_t>(codes[i]) & mask;
        const std::uint32_t low = u & 0x0fu;
        if ((i & 1) == 0) {
            nibble_out[i >> 1] |= static_cast<std::uint8_t>(low);
        } else {
            nibble_out[i >> 1] |= static_cast<std::uint8_t>(low << 4);
        }
        if (high != 0) {
            const std::uint32_t hi = u >> 4;
            if (spec.bits == 5) {
                high_out[i >> 3] |= static_cast<std::uint8_t>((hi & 0x01u) << (i & 7));
            } else {
                const int bit_pos = i * 2;
                high_out[bit_pos >> 3] |= static_cast<std::uint8_t>((hi & 0x03u) << (bit_pos & 7));
            }
        }
    }
}

inline int unpack_lowbit_code(const std::uint8_t* nibble, const std::uint8_t* high,
                              const QuantSpec& spec, int index) {
    if (spec.bits == 8) { return static_cast<std::int8_t>(nibble[index]); }
    const std::uint8_t low_byte = nibble[index >> 1];
    const std::uint32_t low     = (index & 1) ? (low_byte >> 4) : (low_byte & 0x0fu);
    std::uint32_t hi            = 0;
    if (spec.bits == 5) {
        hi = (high[index >> 3] >> (index & 7)) & 0x01u;
    } else if (spec.bits == 6) {
        const int bitpos = index * 2;
        hi               = (high[bitpos >> 3] >> (bitpos & 7)) & 0x03u;
    }
    const std::uint32_t u    = low | (hi << 4);
    const std::uint32_t sign = 1u << (spec.bits - 1);
    const std::uint32_t span = 1u << spec.bits;
    return (u & sign) ? static_cast<int>(u) - static_cast<int>(span) : static_cast<int>(u);
}

} // namespace detail

struct PackedWeight {
    std::vector<std::uint8_t> payload;
    std::vector<float> dequant;
    Weight weight{};
    std::uint64_t code_plane_bytes      = 0;
    std::uint64_t high_plane_offset     = 0;
    std::uint64_t high_plane_bytes      = 0;
    std::uint64_t scale_plane_offset    = 0;
    std::uint64_t scale_plane_bytes     = 0;
    std::uint64_t weight_divisor_offset = 0;

    Weight device_weight(void* device_payload) const {
        Weight w           = weight;
        w.payload          = device_payload;
        w.high_plane_bytes = high_plane_bytes;
        if (device_payload != nullptr) {
            w.qdata  = device_payload;
            w.qhigh  = high_plane_bytes == 0
                           ? nullptr
                           : static_cast<std::uint8_t*>(device_payload) + high_plane_offset;
            w.scales = static_cast<std::uint8_t*>(device_payload) + scale_plane_offset;
        } else {
            w.qdata  = nullptr;
            w.qhigh  = nullptr;
            w.scales = nullptr;
        }
        return w;
    }

    Weight device_row_view(void* device_payload, std::int32_t row_begin,
                           std::int32_t row_count) const {
        Weight w = device_weight(device_payload);
        if (row_begin < 0 || row_count <= 0 || row_begin + row_count > w.n) {
            throw std::invalid_argument("quantized-weight fixture: invalid row view");
        }
        if (w.layout == QuantLayout::RowSplit) {
            const detail::QuantSpec spec      = detail::quant_spec(w.qtype);
            const std::int32_t groups_per_row = w.padded_shape[1] / spec.group_size;
            const std::size_t code_row =
                static_cast<std::size_t>(groups_per_row) * detail::nibble_bytes_per_group(spec);
            const std::size_t high_row =
                static_cast<std::size_t>(groups_per_row) * detail::high_bytes_per_group(spec);
            const std::size_t scale_row =
                static_cast<std::size_t>(groups_per_row) * sizeof(std::uint16_t);
            w.qdata = static_cast<const std::uint8_t*>(w.qdata) +
                      static_cast<std::size_t>(row_begin) * code_row;
            w.qhigh  = high_row == 0 ? nullptr
                                     : static_cast<const std::uint8_t*>(w.qhigh) +
                                          static_cast<std::size_t>(row_begin) * high_row;
            w.scales = static_cast<const std::uint8_t*>(w.scales) +
                       static_cast<std::size_t>(row_begin) * scale_row;
        } else if (w.layout == QuantLayout::BlockScaleK16M128x4) {
            if ((row_begin % 128) != 0 || (row_count % 128) != 0) {
                throw std::invalid_argument(
                    "quantized-weight fixture: NVFP4 row view must align to 128 rows");
            }
            w.qdata = static_cast<const std::uint8_t*>(w.qdata) +
                      static_cast<std::size_t>(row_begin) * w.k / 2;
            w.scales = static_cast<const std::uint8_t*>(w.scales) +
                       static_cast<std::size_t>(row_begin / 128) *
                           static_cast<std::size_t>(w.k / 64) * 512;
        } else {
            throw std::invalid_argument("quantized-weight fixture: unsupported row-view layout");
        }
        w.n               = row_count;
        w.shape[0]        = row_count;
        w.padded_shape[0] = row_count;
        return w;
    }
};

enum class RowSplitScalePattern : std::uint8_t {
    Unit,
    Small,
    Tiny,
};

enum class RowSplitCodePattern : std::uint8_t {
    Coordinate,
    Hashed,
};

struct PatternedWeightOptions {
    RowSplitScalePattern row_split_scale = RowSplitScalePattern::Unit;
    RowSplitCodePattern row_split_codes  = RowSplitCodePattern::Coordinate;
    float weight_scale_divisor           = 0.0F;
    float input_scale_divisor            = 0.0F;

    // TP2 shard support. `row_origin`/`column_origin` place this tensor's logical (0,0) at
    // (row_origin, column_origin) of a larger parent matrix. Every code and every scale the
    // generator emits is a pure function of the GLOBAL coordinate, so
    //
    //   make_patterned_weight(qtype, rc, k,  seed, {.row_origin = r0})                 and
    //   make_patterned_weight(qtype, n,  cc, seed, {.column_origin = c0})
    //
    // are standalone tensors whose logical values equal the parent's rows [r0, r0+rc) and columns
    // [c0, c0+cc) respectively -- which is exactly what the TP2 loader materializes into each
    // device's arena (a shard is a tensor of the same layout with one axis narrowed, never a
    // pointer offset into the parent payload; NVFP4's swizzled scale plane and every row-split
    // plane are re-based per shard). `tests/ops/test_linear_split.cpp` asserts that logical
    // equality through logical_weight_fp64() rather than assuming it.
    //
    // Both default to 0, which reproduces every pre-existing payload byte for byte.
    //
    // Constraints mirror the production shard-boundary rules so an origin the loader would reject
    // cannot be fabricated here: NVFP4 needs `row_origin % 128 == 0` (the 128-row scale tile) and
    // `column_origin % 64 == 0` (the 64-column scale tile); row-split needs
    // `column_origin % 128 == 0` (the layout's K-alignment unit, a multiple of both group sizes).
    // RowSplitCodePattern::Hashed keys on the flat group index rather than the logical coordinate
    // and is therefore not translation invariant; it rejects a non-zero origin.
    std::int32_t row_origin    = 0;
    std::int32_t column_origin = 0;

    // Removes the periodicity of the affine code/scale patterns on BOTH axes. The NVFP4 rules are
    // affine in row and column (`code = row*13 + column*7 + seed`, `scale = row*5 + group*3 +
    // seed`) with moduli 16 and 8, and the FP8 code rule is affine with modulus 16, so two blocks
    // separated by a stride s carry IDENTICAL codes whenever the relevant multiple of s vanishes
    // under those moduli. EVERY tp2 shard boundary in this model is such an s, on both axes:
    //
    //   rows    7168, 8192, 17408:  13s = 0 (mod 16)  and  5s = 0 (mod 8)
    //   columns 3072, 8704:          7s = 0 (mod 16)  and  3*(s/16) = 0 (mod 8)
    //
    // A suite that compares one shard against the other -- which is what BOTH split axes do -- was
    // therefore comparing two byte-identical tensors, and would pass even if a rank had computed
    // the wrong half. This is the single most dangerous failure mode of a shard test, because it
    // is invisible: everything reports zero error.
    //
    // With this set both coordinates are hashed before they enter the pattern. The generator stays
    // a pure function of the GLOBAL coordinate, so shard translation invariance is untouched; only
    // the accidental periodicity goes away. The row-split codec's own `row ^ (row>>8) ^ (row>>16)`
    // row mix and its `group*29` column term are already non-degenerate at these strides; the
    // option is applied uniformly anyway so one flag means one thing in every format.
    //
    // `tests/ops/test_linear_split.cpp` does not trust this: it asserts that the two shard
    // PAYLOADS actually differ, on both axes, in every case. That assertion is what caught the
    // defect this option exists to fix -- twice, once per axis.
    //
    // Defaults to false, which reproduces every pre-existing payload byte for byte.
    bool decorrelate_coordinates = false;
};

// Builds a deterministic full-shape payload without allocating a source or dequantized matrix.
// Large registered-shape conformance tests can then verify deterministic FP64 row samples.
inline PackedWeight make_patterned_weight(QType qtype, std::int32_t n, std::int32_t k,
                                          std::uint32_t seed, PatternedWeightOptions options = {}) {
    if (n <= 0 || k <= 0) {
        throw std::invalid_argument("quantized-weight fixture: shape must be positive");
    }
    if (options.row_origin < 0 || options.column_origin < 0) {
        throw std::invalid_argument("quantized-weight fixture: shard origin must not be negative");
    }
    const std::int32_t row_origin    = options.row_origin;
    const std::int32_t column_origin = options.column_origin;
    // The values every pattern keys on: the global coordinate, optionally hashed to remove the
    // affine patterns' periodicity (see PatternedWeightOptions::decorrelate_coordinates). Each
    // axis gets its own salt so that hashing cannot make the row and column terms agree.
    const auto decorrelate = [&](std::uint64_t absolute, std::uint64_t salt) -> std::uint64_t {
        return options.decorrelate_coordinates ? (detail::mix64(absolute + salt) >> 24) : absolute;
    };
    const auto pattern_row = [&](std::int64_t local_row) -> std::uint64_t {
        return decorrelate(static_cast<std::uint64_t>(local_row + row_origin), 0x9e37U);
    };
    // Per-element column index (NVFP4 codes, FP8 codes).
    const auto pattern_column = [&](std::int64_t local_column) -> std::uint64_t {
        return decorrelate(static_cast<std::uint64_t>(local_column + column_origin), 0x85ebU);
    };
    // Per-group column index; `shift` is the shard's own group origin, in that format's groups.
    const auto pattern_group = [&](std::int64_t local_group, std::uint64_t shift) -> std::uint64_t {
        return decorrelate(static_cast<std::uint64_t>(local_group) + shift, 0xc2b2U);
    };
    if (qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        if (options.weight_scale_divisor != 0.0F || options.input_scale_divisor != 0.0F) {
            throw std::invalid_argument(
                "quantized-weight fixture: divisors do not belong to FP8 weights");
        }

        PackedWeight packed;
        packed.code_plane_bytes = static_cast<std::uint64_t>(n) * k;
        packed.scale_plane_offset =
            detail::align_up_size(static_cast<std::size_t>(packed.code_plane_bytes), 256);
        packed.scale_plane_bytes = static_cast<std::uint64_t>(n) * 2;
        packed.payload.assign(
            static_cast<std::size_t>(packed.scale_plane_offset + packed.scale_plane_bytes), 0);

        constexpr std::uint8_t kCodes[]{
            0x00U, 0x80U, 0x01U, 0x81U, 0x07U, 0x87U, 0x08U, 0x88U,
            0x18U, 0x98U, 0x38U, 0xb8U, 0x44U, 0xc4U, 0x7eU, 0xfeU,
        };
        // This XOR-shift is what makes a ROW-origin shard boundary non-periodic
        // at most strides even with decorrelate_coordinates=false (verified against
        // tests/ops/test_linear_swiglu_split.cpp's and test_attn_input_proj_split.cpp's own
        // negative controls, which do NOT fire at their row strides -- 8704/17408 and
        // 3072/512/6144/7168 respectively -- while tests/ops/test_linear_split.cpp's DOES fire at
        // 8192). This is provable, not incidental, so state the exact boundary rather than leave it
        // an empirical accident future maintainers could silently rely on at the wrong stride:
        //
        // `pattern & 15` only ever reads `row_mix mod 16`, i.e. bits[3:0] of `row ^ (row>>7)`,
        // which equals bits[3:0](row) XOR bits[10:7](row) -- a function of bits[10:0](row) alone.
        // Adding a row-origin shift S changes those 11 bits (and therefore the code plane) UNLESS
        // S is an exact multiple of 2048 (2^11), in which case (row+S) mod 2048 == row mod 2048 for
        // EVERY row and the code plane is exactly periodic at stride S -- annihilating without
        // decorrelate_coordinates, the same failure mode NVFP4's purely-affine pattern has at every
        // stride. (The scale plane below is periodic at any stride that is merely a multiple of 4,
        // `kScales[(pattern_row(row)+seed) & 3]` with no XOR mix at all -- but the much larger code
        // plane's mod-2048 condition is what actually gates whole-payload distinctness.)
        //
        // COLUMN-origin shifts have no such protection: `pattern_column` feeds the code pattern
        // through a plain linear `*7` term with no mixing step, so it annihilates at ANY stride that
        // is a multiple of 16 (7 is invertible mod 16, so `7*(col+C) mod 16 == 7*col mod 16` iff
        // `C mod 16 == 0`) -- confirmed by both test_linear_split.cpp's and
        // test_linear_add_split.cpp's own negative controls firing at their 3072/8704 column
        // strides. Do not assume a "large" stride is automatically safe on either axis: check it
        // against the modulus for the axis actually being shifted, or just leave
        // decorrelate_coordinates=true (every split suite in this repository already does).
        for (std::int32_t row = 0; row < n; ++row) {
            const auto global_row       = static_cast<std::uint32_t>(pattern_row(row));
            const std::uint32_t row_mix = global_row ^ (global_row >> 7);
            for (std::int32_t column = 0; column < k; ++column) {
                const std::uint32_t pattern =
                    row_mix * 13U + static_cast<std::uint32_t>(pattern_column(column)) * 7U + seed;
                packed.payload[static_cast<std::size_t>(row) * k + column] = kCodes[pattern & 15U];
            }
        }
        constexpr std::uint16_t kScales[]{0x3b00U, 0x3b40U, 0x3b80U, 0x3bc0U};
        for (std::int32_t row = 0; row < n; ++row) {
            detail::store_u16_le(
                packed.payload, packed.scale_plane_offset + static_cast<std::size_t>(row) * 2,
                kScales[(static_cast<std::uint32_t>(pattern_row(row)) + seed) & 3U]);
        }

        packed.weight.qtype            = QType::FP8_E4M3FN_ROW_BF16S;
        packed.weight.layout           = QuantLayout::RowScale;
        packed.weight.scale_dtype      = DType::BF16;
        packed.weight.payload          = packed.payload.data();
        packed.weight.payload_bytes    = packed.payload.size();
        packed.weight.high_plane_bytes = 0;
        packed.weight.qdata            = packed.payload.data();
        packed.weight.qhigh            = nullptr;
        packed.weight.scales           = packed.payload.data() + packed.scale_plane_offset;
        packed.weight.group_size       = static_cast<std::uint32_t>(k);
        packed.weight.group            = k;
        packed.weight.ndim             = 2;
        packed.weight.shape[0]         = n;
        packed.weight.shape[1]         = k;
        packed.weight.shape[2]         = 1;
        packed.weight.shape[3]         = 1;
        packed.weight.padded_shape[0]  = n;
        packed.weight.padded_shape[1]  = k;
        packed.weight.padded_shape[2]  = 1;
        packed.weight.padded_shape[3]  = 1;
        packed.weight.scale_ne[0]      = n;
        packed.weight.scale_ne[1]      = 1;
        packed.weight.scale_ne[2]      = 1;
        packed.weight.scale_ne[3]      = 1;
        packed.weight.scale_nb[0]      = 2;
        packed.weight.scale_nb[1]      = static_cast<std::int64_t>(n) * 2;
        packed.weight.scale_nb[2]      = packed.weight.scale_nb[1];
        packed.weight.scale_nb[3]      = packed.weight.scale_nb[1];
        packed.weight.n                = n;
        packed.weight.k                = k;
        return packed;
    }
    if (qtype == QType::NVFP4) {
        if ((n % 128) != 0 || (k % 64) != 0) {
            throw std::invalid_argument(
                "quantized-weight fixture: NVFP4 N must be divisible by 128 and K by 64");
        }
        if (options.row_split_scale != RowSplitScalePattern::Unit ||
            options.row_split_codes != RowSplitCodePattern::Coordinate) {
            throw std::invalid_argument(
                "quantized-weight fixture: row-split patterns do not apply to NVFP4");
        }
        if (!std::isfinite(options.weight_scale_divisor) || options.weight_scale_divisor <= 0.0F ||
            !std::isfinite(options.input_scale_divisor) || options.input_scale_divisor <= 0.0F) {
            throw std::invalid_argument(
                "quantized-weight fixture: NVFP4 divisors must be finite and positive");
        }
        if ((row_origin % 128) != 0 || (column_origin % 64) != 0) {
            throw std::invalid_argument("quantized-weight fixture: NVFP4 shard origin must align "
                                        "to 128 rows and 64 columns");
        }

        PackedWeight packed;
        packed.code_plane_bytes = static_cast<std::uint64_t>(n) * k / 2;
        packed.scale_plane_offset =
            detail::align_up_size(static_cast<std::size_t>(packed.code_plane_bytes), 256);
        packed.scale_plane_bytes     = static_cast<std::uint64_t>(n) * k / 16;
        packed.weight_divisor_offset = packed.scale_plane_offset + packed.scale_plane_bytes;
        packed.payload.assign(static_cast<std::size_t>(packed.weight_divisor_offset) + 4U, 0);

        for (std::int32_t row = 0; row < n; ++row) {
            for (std::int32_t column = 0; column < k; column += 2) {
                const auto code = [&](std::int32_t logical_column) {
                    return static_cast<std::uint8_t>(
                        (static_cast<std::uint32_t>(pattern_row(row)) * 13U +
                         static_cast<std::uint32_t>(pattern_column(logical_column)) * 7U + seed) &
                        0x0fU);
                };
                packed.payload[static_cast<std::size_t>(row) * k / 2 +
                               static_cast<std::size_t>(column / 2)] =
                    static_cast<std::uint8_t>(code(column) | (code(column + 1) << 4));
            }
        }

        constexpr std::uint8_t kScaleWords[]{
            0x00U, // +0
            0x04U, // 0.0078125
            0x08U, // 0.015625
            0x0bU, // 0.021484375
            0x10U, // 0.03125
            0x13U, // 0.04296875
            0x18U, // 0.0625
            0x1dU, // 0.1015625
        };
        const std::int32_t groups_per_row = k / 16;
        const std::int32_t k_tiles        = k / 64;
        for (std::int32_t row = 0; row < n; ++row) {
            const std::int32_t row_tile  = row / 128;
            const std::int32_t row_inner = row % 128;
            for (std::int32_t group = 0; group < groups_per_row; ++group) {
                const std::int32_t scale_tile = group / 4;
                const std::int32_t scale_lane = group % 4;
                const std::size_t stored_offset =
                    packed.scale_plane_offset +
                    static_cast<std::size_t>(row_tile * k_tiles + scale_tile) * 512U +
                    static_cast<std::size_t>(row_inner % 32) * 16U +
                    static_cast<std::size_t>(row_inner / 32) * 4U +
                    static_cast<std::size_t>(scale_lane);
                const std::size_t pattern =
                    (static_cast<std::uint32_t>(pattern_row(row)) * 5U +
                     static_cast<std::uint32_t>(
                         pattern_group(group, static_cast<std::uint64_t>(column_origin / 16))) *
                         3U +
                     seed) &
                    7U;
                packed.payload[stored_offset] = kScaleWords[pattern];
            }
        }
        detail::store_u32_le(packed.payload, packed.weight_divisor_offset,
                             detail::float_bits(options.weight_scale_divisor));

        packed.weight.qtype                = QType::NVFP4;
        packed.weight.layout               = QuantLayout::BlockScaleK16M128x4;
        packed.weight.scale_dtype          = DType::FP8_E4M3FN;
        packed.weight.payload              = packed.payload.data();
        packed.weight.payload_bytes        = packed.payload.size();
        packed.weight.high_plane_bytes     = 0;
        packed.weight.qdata                = packed.payload.data();
        packed.weight.qhigh                = nullptr;
        packed.weight.scales               = packed.payload.data() + packed.scale_plane_offset;
        packed.weight.group_size           = 16;
        packed.weight.group                = 16;
        packed.weight.ndim                 = 2;
        packed.weight.shape[0]             = n;
        packed.weight.shape[1]             = k;
        packed.weight.shape[2]             = 1;
        packed.weight.shape[3]             = 1;
        packed.weight.padded_shape[0]      = n;
        packed.weight.padded_shape[1]      = k;
        packed.weight.padded_shape[2]      = 1;
        packed.weight.padded_shape[3]      = 1;
        packed.weight.n                    = n;
        packed.weight.k                    = k;
        packed.weight.weight_scale_divisor = options.weight_scale_divisor;
        packed.weight.input_scale_divisor  = options.input_scale_divisor;
        return packed;
    }
    if (options.weight_scale_divisor != 0.0F || options.input_scale_divisor != 0.0F) {
        throw std::invalid_argument(
            "quantized-weight fixture: divisors are defined only for NVFP4");
    }
    if ((column_origin % 128) != 0) {
        throw std::invalid_argument(
            "quantized-weight fixture: row-split shard column origin must align to 128");
    }
    if (options.row_split_codes == RowSplitCodePattern::Hashed &&
        (row_origin != 0 || column_origin != 0)) {
        throw std::invalid_argument(
            "quantized-weight fixture: hashed row-split codes are not translation invariant");
    }
    const detail::QuantSpec spec      = detail::quant_spec(qtype);
    const std::int32_t padded_k       = detail::align_up(k, 128);
    const std::int32_t kg             = padded_k / spec.group_size;
    const std::int32_t logical_groups = (k + spec.group_size - 1) / spec.group_size;
    const std::uint64_t groups        = static_cast<std::uint64_t>(n) * kg;

    PackedWeight packed;
    packed.code_plane_bytes =
        groups * static_cast<std::uint64_t>(detail::nibble_bytes_per_group(spec));
    packed.high_plane_offset =
        detail::align_up_size(static_cast<std::size_t>(packed.code_plane_bytes), 256);
    packed.high_plane_bytes =
        groups * static_cast<std::uint64_t>(detail::high_bytes_per_group(spec));
    packed.scale_plane_offset =
        packed.high_plane_offset +
        detail::align_up_size(static_cast<std::size_t>(packed.high_plane_bytes), 256);
    packed.scale_plane_bytes = groups * 2;
    packed.payload.assign(
        static_cast<std::size_t>(packed.scale_plane_offset + packed.scale_plane_bytes), 0);

    const std::uint64_t code_bytes_per_group = detail::nibble_bytes_per_group(spec);
    const std::uint64_t high_bytes_per_group = detail::high_bytes_per_group(spec);
    // Global coordinates of this tensor's origin inside its parent, in the units each pattern
    // keys on. A shard therefore emits exactly the parent's codes and scales for its own range.
    const auto group_shift = static_cast<std::uint64_t>(column_origin / spec.group_size);
    if (options.row_split_codes == RowSplitCodePattern::Coordinate) {
        for (std::uint64_t group_index = 0; group_index < groups; ++group_index) {
            const std::uint64_t row =
                pattern_row(static_cast<std::int64_t>(group_index / static_cast<std::uint64_t>(kg)));
            const std::uint64_t group = group_index % static_cast<std::uint64_t>(kg);
            if (group >= static_cast<std::uint64_t>(logical_groups)) { continue; }
            const std::uint64_t global_group =
                pattern_group(static_cast<std::int64_t>(group), group_shift);
            const std::uint32_t row_mix =
                static_cast<std::uint32_t>(row ^ (row >> 8) ^ (row >> 16));
            for (std::uint64_t byte = 0; byte < code_bytes_per_group; ++byte) {
                std::uint8_t code = static_cast<std::uint8_t>(
                    (row_mix * 37u + global_group * 29u + byte * 17u + seed) & 0xffu);
                if (qtype == QType::W8G32_F16S && code == 0x80u) { code = 0x81u; }
                packed
                    .payload[static_cast<std::size_t>(group_index * code_bytes_per_group + byte)] =
                    code;
            }
            for (std::uint64_t byte = 0; byte < high_bytes_per_group; ++byte) {
                packed.payload[static_cast<std::size_t>(
                    packed.high_plane_offset + group_index * high_bytes_per_group + byte)] =
                    static_cast<std::uint8_t>(
                        (row_mix * 43u + global_group * 31u + byte * 13u + seed * 3u) & 0xffu);
            }
        }
    } else {
        std::uint8_t code_patterns[256][32]{};
        std::uint8_t high_patterns[256][16]{};
        for (std::size_t pattern = 0; pattern < 256; ++pattern) {
            std::uint64_t state = detail::mix64(static_cast<std::uint64_t>(seed) << 32 | pattern);
            for (std::size_t byte = 0; byte < code_bytes_per_group; ++byte) {
                state              = detail::mix64(state + byte);
                std::uint8_t value = static_cast<std::uint8_t>(state >> 56);
                if (qtype == QType::W8G32_F16S && value == 0x80U) { value = 0x81U; }
                code_patterns[pattern][byte] = value;
            }
            for (std::size_t byte = 0; byte < high_bytes_per_group; ++byte) {
                state                        = detail::mix64(state + 0x100U + byte);
                high_patterns[pattern][byte] = static_cast<std::uint8_t>(state >> 56);
            }
        }
        for (std::uint64_t group_index = 0; group_index < groups; ++group_index) {
            const std::uint64_t group = group_index % static_cast<std::uint64_t>(kg);
            if (group >= static_cast<std::uint64_t>(logical_groups)) { continue; }
            const std::uint64_t mixed =
                detail::mix64(group_index ^ (static_cast<std::uint64_t>(seed) << 17));
            const std::size_t pattern = static_cast<std::size_t>(mixed & 0xffU);
            std::memcpy(packed.payload.data() + group_index * code_bytes_per_group,
                        code_patterns[pattern], code_bytes_per_group);
            if (high_bytes_per_group != 0) {
                std::memcpy(packed.payload.data() + packed.high_plane_offset +
                                group_index * high_bytes_per_group,
                            high_patterns[pattern], high_bytes_per_group);
            }
        }
    }
    const std::int32_t partial_group_lanes = k % spec.group_size;
    if (partial_group_lanes != 0) {
        std::int8_t codes[64]{};
        const std::int32_t group = logical_groups - 1;
        for (std::int32_t row = 0; row < n; ++row) {
            const std::size_t group_index = static_cast<std::size_t>(row) * kg + group;
            std::uint8_t* nibble = packed.payload.data() + group_index * code_bytes_per_group;
            std::uint8_t* high   = high_bytes_per_group == 0
                                       ? nullptr
                                       : packed.payload.data() + packed.high_plane_offset +
                                           group_index * high_bytes_per_group;
            for (std::int32_t lane = 0; lane < partial_group_lanes; ++lane) {
                codes[lane] =
                    static_cast<std::int8_t>(detail::unpack_lowbit_code(nibble, high, spec, lane));
            }
            std::fill(codes + partial_group_lanes, codes + spec.group_size,
                      static_cast<std::int8_t>(0));
            detail::pack_lowbit_group(codes, spec, nibble, high);
        }
    }
    constexpr std::uint16_t kUnitScales[]  = {0x3800u, 0x3a00u, 0x3c00u, 0x3d00u};
    constexpr std::uint16_t kSmallScales[] = {0x2040u, 0x2440u, 0x2840u, 0x2c40u};
    constexpr std::uint16_t kTinyScales[]  = {0x1840u, 0x1c40u, 0x2040u, 0x2440u};
    const std::uint16_t* scales            = nullptr;
    switch (options.row_split_scale) {
    case RowSplitScalePattern::Unit:
        scales = kUnitScales;
        break;
    case RowSplitScalePattern::Small:
        scales = kSmallScales;
        break;
    case RowSplitScalePattern::Tiny:
        scales = kTinyScales;
        break;
    }
    for (std::uint64_t i = 0; i < groups; ++i) {
        const std::uint64_t row =
            pattern_row(static_cast<std::int64_t>(i / static_cast<std::uint64_t>(kg)));
        const std::uint64_t group = i % static_cast<std::uint64_t>(kg);
        if (group >= static_cast<std::uint64_t>(logical_groups)) { continue; }
        const std::uint64_t scale_index =
            options.row_split_codes == RowSplitCodePattern::Hashed
                ? (detail::mix64(i ^ (static_cast<std::uint64_t>(seed) << 17)) >> 8) & 3U
                : (row ^ (row >> 8) ^
                   pattern_group(static_cast<std::int64_t>(group), group_shift) ^ seed) &
                      3U;
        detail::store_u16_le(packed.payload,
                             static_cast<std::size_t>(packed.scale_plane_offset + i * 2),
                             scales[scale_index]);
    }

    packed.weight.qtype            = qtype;
    packed.weight.layout           = QuantLayout::RowSplit;
    packed.weight.scale_dtype      = DType::FP16;
    packed.weight.payload          = packed.payload.data();
    packed.weight.payload_bytes    = packed.payload.size();
    packed.weight.high_plane_bytes = packed.high_plane_bytes;
    packed.weight.qdata            = packed.payload.data();
    packed.weight.qhigh =
        packed.high_plane_bytes == 0 ? nullptr : packed.payload.data() + packed.high_plane_offset;
    packed.weight.scales          = packed.payload.data() + packed.scale_plane_offset;
    packed.weight.group_size      = static_cast<std::uint32_t>(spec.group_size);
    packed.weight.group           = spec.group_size;
    packed.weight.ndim            = 2;
    packed.weight.shape[0]        = n;
    packed.weight.shape[1]        = k;
    packed.weight.shape[2]        = 1;
    packed.weight.shape[3]        = 1;
    packed.weight.padded_shape[0] = n;
    packed.weight.padded_shape[1] = padded_k;
    packed.weight.padded_shape[2] = 1;
    packed.weight.padded_shape[3] = 1;
    packed.weight.n               = n;
    packed.weight.k               = k;
    return packed;
}

// Independent logical decode for numerical oracles. The packed representation is materialized as
// its specified logical FP32 weight before the complete Op oracle receives it.
inline double logical_weight_fp64(const PackedWeight& packed, std::int32_t row,
                                  std::int32_t column) {
    const Weight& weight = packed.weight;
    if (row < 0 || row >= weight.shape[0] || column < 0 || column >= weight.shape[1]) {
        throw std::out_of_range("quantized-weight fixture: logical index out of range");
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        if (weight.layout != QuantLayout::RowScale || weight.scale_dtype != DType::BF16 ||
            weight.group != weight.k || weight.group_size != static_cast<std::uint32_t>(weight.k)) {
            throw std::invalid_argument("quantized-weight fixture: invalid FP8 metadata");
        }
        const std::uint8_t code = packed.payload[static_cast<std::size_t>(row) * weight.k + column];
        const std::uint16_t scale_bits = detail::load_u16_le(
            packed.payload, packed.scale_plane_offset + static_cast<std::size_t>(row) * 2);
        const float decoded =
            static_cast<float>(detail::decode_e4m3fn(code)) * detail::bf16_to_f32(scale_bits);
        return static_cast<double>(decoded);
    }

    if (weight.qtype == QType::NVFP4) {
        if (weight.layout != QuantLayout::BlockScaleK16M128x4 ||
            weight.scale_dtype != DType::FP8_E4M3FN || weight.group != 16 ||
            !std::isfinite(weight.weight_scale_divisor) || weight.weight_scale_divisor <= 0.0F) {
            throw std::invalid_argument("quantized-weight fixture: invalid NVFP4 metadata");
        }
        const std::size_t code_offset =
            static_cast<std::size_t>(row) * weight.k / 2 + static_cast<std::size_t>(column / 2);
        const std::uint8_t packed_codes = packed.payload[code_offset];
        const std::uint8_t code = (column & 1) == 0 ? (packed_codes & 0x0fU) : (packed_codes >> 4);

        const std::int32_t group      = column / 16;
        const std::int32_t row_tile   = row / 128;
        const std::int32_t row_inner  = row % 128;
        const std::int32_t scale_tile = group / 4;
        const std::int32_t scale_lane = group % 4;
        const std::int32_t k_tiles    = weight.k / 64;
        const std::size_t scale_offset =
            packed.scale_plane_offset +
            static_cast<std::size_t>(row_tile * k_tiles + scale_tile) * 512U +
            static_cast<std::size_t>(row_inner % 32) * 16U +
            static_cast<std::size_t>(row_inner / 32) * 4U + static_cast<std::size_t>(scale_lane);
        const double decoded = detail::decode_e2m1(code) *
                               detail::decode_e4m3fn(packed.payload[scale_offset]) /
                               static_cast<double>(weight.weight_scale_divisor);
        return static_cast<double>(static_cast<float>(decoded));
    }

    const detail::QuantSpec spec = detail::quant_spec(weight.qtype);
    const std::int32_t padded_k  = weight.padded_shape[1];
    if (padded_k < weight.shape[1] || padded_k % spec.group_size != 0) {
        throw std::invalid_argument("quantized-weight fixture: invalid padded K");
    }
    const std::int32_t groups_per_row = padded_k / spec.group_size;
    const std::int32_t group          = column / spec.group_size;
    const std::int32_t lane           = column - group * spec.group_size;
    const std::size_t group_index     = static_cast<std::size_t>(row) * groups_per_row + group;
    const int nibble_bytes            = detail::nibble_bytes_per_group(spec);
    const int high_bytes              = detail::high_bytes_per_group(spec);

    const std::uint8_t* nibble       = packed.payload.data() + group_index * nibble_bytes;
    const std::uint8_t* high         = high_bytes == 0 ? nullptr
                                                       : packed.payload.data() + packed.high_plane_offset +
                                                     group_index * high_bytes;
    const std::uint16_t stored_scale = detail::load_u16_le(
        packed.payload, packed.scale_plane_offset + group_index * sizeof(std::uint16_t));
    const int signed_code                = detail::unpack_lowbit_code(nibble, high, spec, lane);
    const double exact_stored_fp16_scale = static_cast<double>(detail::f16_to_f32(stored_scale));
    return static_cast<double>(signed_code) * exact_stored_fp16_scale;
}

inline double dot_fp64(const PackedWeight& packed, std::int32_t row, const float* logical_input,
                       std::int32_t k) {
    if (logical_input == nullptr) {
        throw std::invalid_argument("quantized-weight fixture: input must be non-null");
    }
    if (k != packed.weight.shape[1]) {
        throw std::invalid_argument("quantized-weight fixture: logical K mismatch");
    }
    double acc = 0.0;
    for (std::int32_t column = 0; column < k; ++column) {
        acc +=
            logical_weight_fp64(packed, row, column) * static_cast<double>(logical_input[column]);
    }
    return acc;
}

inline std::vector<float> materialize_rows_fp32(const PackedWeight& packed,
                                                std::span<const std::int32_t> rows) {
    if (rows.empty()) {
        throw std::invalid_argument("quantized-weight fixture: row selection is empty");
    }
    std::vector<float> result(static_cast<std::size_t>(packed.weight.k) * rows.size());
    for (std::size_t selected = 0; selected < rows.size(); ++selected) {
        const std::int32_t row = rows[selected];
        for (std::int32_t column = 0; column < packed.weight.k; ++column) {
            result[selected * static_cast<std::size_t>(packed.weight.k) +
                   static_cast<std::size_t>(column)] =
                static_cast<float>(logical_weight_fp64(packed, row, column));
        }
    }
    return result;
}

inline std::vector<float> decode_row_split_lowbit(const std::vector<std::uint8_t>& payload,
                                                  std::int32_t n, std::int32_t k,
                                                  std::int32_t padded_k, QType qtype) {
    const detail::QuantSpec spec = detail::quant_spec(qtype);
    const int nib                = detail::nibble_bytes_per_group(spec);
    const int high_bpr           = detail::high_bytes_per_group(spec);
    const std::int32_t kg        = padded_k / spec.group_size;
    const std::size_t nibble_bytes =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(kg) * static_cast<std::size_t>(nib);
    const std::size_t high_bytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(kg) *
                                   static_cast<std::size_t>(high_bpr);
    const std::size_t high_off  = detail::align_up_size(nibble_bytes, 256);
    const std::size_t scale_off = high_off + detail::align_up_size(high_bytes, 256);

    std::vector<float> deq(static_cast<std::size_t>(n) * k);
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t g = 0; g < kg; ++g) {
            const std::size_t group_index = static_cast<std::size_t>(row) * kg + g;
            const std::uint16_t scale_h = detail::load_u16_le(payload, scale_off + group_index * 2);
            const float scale           = detail::f16_to_f32(scale_h);
            const std::uint8_t* nibble  = payload.data() + group_index * nib;
            const std::uint8_t* high =
                high_bpr == 0 ? nullptr : payload.data() + high_off + group_index * high_bpr;
            for (std::int32_t lane = 0; lane < spec.group_size; ++lane) {
                const std::int32_t kk = g * spec.group_size + lane;
                if (kk >= k) { continue; }
                const int code = detail::unpack_lowbit_code(nibble, high, spec, lane);
                deq[static_cast<std::size_t>(row) * k + kk] = static_cast<float>(code) * scale;
            }
        }
    }
    return deq;
}

inline PackedWeight pack_row_split_lowbit(const std::vector<float>& source, std::int32_t n,
                                          std::int32_t k, QType qtype) {
    if (n <= 0 || k <= 0) {
        throw std::invalid_argument("row-split test packer: shape must be positive");
    }
    if (source.size() != static_cast<std::size_t>(n) * k) {
        throw std::invalid_argument("row-split test packer: source size mismatch");
    }

    const detail::QuantSpec spec = detail::quant_spec(qtype);
    const std::int32_t padded_k  = detail::align_up(k, 128);
    const std::int32_t kg        = padded_k / spec.group_size;
    const int nib                = detail::nibble_bytes_per_group(spec);
    const int high_bpr           = detail::high_bytes_per_group(spec);

    PackedWeight out;
    out.code_plane_bytes = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(kg) *
                           static_cast<std::uint64_t>(nib);
    out.high_plane_offset =
        detail::align_up_size(static_cast<std::size_t>(out.code_plane_bytes), 256);
    out.high_plane_bytes = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(kg) *
                           static_cast<std::uint64_t>(high_bpr);
    out.scale_plane_offset =
        out.high_plane_offset +
        detail::align_up_size(static_cast<std::size_t>(out.high_plane_bytes), 256);
    out.scale_plane_bytes = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(kg) * 2ULL;
    out.payload.assign(static_cast<std::size_t>(out.scale_plane_offset + out.scale_plane_bytes), 0);

    std::int8_t codes[64];
    float vals[64];
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t g = 0; g < kg; ++g) {
            float maxabs = 0.0f;
            for (std::int32_t lane = 0; lane < spec.group_size; ++lane) {
                const std::int32_t kk = g * spec.group_size + lane;
                const float v = (kk < k) ? source[static_cast<std::size_t>(row) * k + kk] : 0.0f;
                vals[lane]    = v;
                maxabs        = std::max(maxabs, std::abs(v));
            }

            std::uint16_t scale_h = detail::f32_to_f16(maxabs / static_cast<float>(spec.qmax));
            if (scale_h == 0 && maxabs > 0.0f) { scale_h = 0x0001u; }
            const float scale = detail::f16_to_f32(scale_h);
            const float inv   = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
            for (std::int32_t lane = 0; lane < spec.group_size; ++lane) {
                const float q = std::nearbyint(vals[lane] * inv);
                const int qi  = std::min(spec.qmax, std::max(spec.qmin, static_cast<int>(q)));
                codes[lane]   = static_cast<std::int8_t>(qi);
            }

            const std::size_t group_index = static_cast<std::size_t>(row) * kg + g;
            detail::pack_lowbit_group(codes, spec, out.payload.data() + group_index * nib,
                                      high_bpr == 0 ? nullptr
                                                    : out.payload.data() + out.high_plane_offset +
                                                          group_index * high_bpr);
            detail::store_u16_le(out.payload, out.scale_plane_offset + group_index * 2, scale_h);
        }
    }

    out.dequant = decode_row_split_lowbit(out.payload, n, k, padded_k, qtype);

    out.weight.qtype            = qtype;
    out.weight.layout           = QuantLayout::RowSplit;
    out.weight.scale_dtype      = DType::FP16;
    out.weight.payload          = out.payload.data();
    out.weight.payload_bytes    = out.payload.size();
    out.weight.high_plane_bytes = out.high_plane_bytes;
    out.weight.qdata            = out.payload.data();
    out.weight.qhigh =
        out.high_plane_bytes == 0 ? nullptr : out.payload.data() + out.high_plane_offset;
    out.weight.scales          = out.payload.data() + out.scale_plane_offset;
    out.weight.group_size      = static_cast<std::uint32_t>(spec.group_size);
    out.weight.group           = spec.group_size;
    out.weight.ndim            = 2;
    out.weight.shape[0]        = n;
    out.weight.shape[1]        = k;
    out.weight.shape[2]        = 1;
    out.weight.shape[3]        = 1;
    out.weight.padded_shape[0] = n;
    out.weight.padded_shape[1] = padded_k;
    out.weight.padded_shape[2] = 1;
    out.weight.padded_shape[3] = 1;
    out.weight.n               = n;
    out.weight.k               = k;
    return out;
}

inline PackedWeight pack_q4_row_split(const std::vector<float>& source, std::int32_t n,
                                      std::int32_t k) {
    return pack_row_split_lowbit(source, n, k, QType::Q4G64_F16S);
}

inline PackedWeight pack_q5_row_split(const std::vector<float>& source, std::int32_t n,
                                      std::int32_t k) {
    return pack_row_split_lowbit(source, n, k, QType::Q5G64_F16S);
}

inline PackedWeight pack_q6_row_split(const std::vector<float>& source, std::int32_t n,
                                      std::int32_t k) {
    return pack_row_split_lowbit(source, n, k, QType::Q6G64_F16S);
}

inline PackedWeight pack_w8g32_row_split(const std::vector<float>& source, std::int32_t n,
                                         std::int32_t k) {
    return pack_row_split_lowbit(source, n, k, QType::W8G32_F16S);
}

} // namespace ninfer::test::quantized_weight
