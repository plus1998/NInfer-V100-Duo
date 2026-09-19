// Byte-level table test for artifact tensor slicing, as used by TP2 sharded materialization.
//
// Host-only: no artifact, no device. For every registered storage layout this builds a parent
// payload from LOGICAL coordinates using the formulas in docs/maintainer/storage-layouts.md,
// applies the PlaneCopy recipe that tensor_row_slice/tensor_column_slice produce, and compares the
// result against a payload built independently for the sub-matrix from the same logical formulas.
// The two sides meet only at the logical level, so a geometry mistake in the slice arithmetic
// (treating the swizzled NVFP4 scale plane as row-major, forgetting a plane's realignment, reading
// the wrong group stride) shows up as a byte mismatch rather than cancelling out.

#include "artifact/reader.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::artifact::SliceRange;
using ninfer::artifact::StorageLayout;
using ninfer::artifact::TensorSlice;

int g_failures = 0;

void fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    ++g_failures;
}

using Bytes = std::vector<std::uint8_t>;

// Deterministic, coordinate-dependent filler: any two distinct logical coordinates get distinct
// enough bytes that a misplaced range is visible.
std::uint8_t pattern(std::uint64_t a, std::uint64_t b, std::uint64_t c) {
    return static_cast<std::uint8_t>((a * 131 + b * 17 + c * 7 + 11) & 0xFFU);
}

Bytes apply_slice(const Bytes& parent, const TensorSlice& slice) {
    Bytes out(static_cast<std::size_t>(slice.encoded_bytes), 0);
    for (std::size_t i = 0; i < slice.prefix.size(); ++i) {
        out[i] = std::to_integer<std::uint8_t>(slice.prefix[i]);
    }
    for (const auto& copy : slice.copies) {
        if (copy.source_offset + copy.bytes > parent.size() ||
            copy.dest_offset + copy.bytes > out.size()) {
            fail("slice copy range escapes its buffer");
            return out;
        }
        for (std::uint64_t i = 0; i < copy.bytes; ++i) {
            out[static_cast<std::size_t>(copy.dest_offset + i)] =
                parent[static_cast<std::size_t>(copy.source_offset + i)];
        }
    }
    return out;
}

void expect_equal(const Bytes& actual, const Bytes& expected, const std::string& label) {
    if (actual.size() != expected.size()) {
        fail(label + ": size " + std::to_string(actual.size()) + ", expected " +
             std::to_string(expected.size()));
        return;
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            fail(label + ": byte " + std::to_string(i) + " is " + std::to_string(actual[i]) +
                 ", expected " + std::to_string(expected[i]));
            return;
        }
    }
}

// --- Layout builders, written from the logical coordinate formulas ------------------------------

// contiguous-le-v1, BF16: row-major words. `row_origin`/`column_origin` shift the coordinate the
// pattern is evaluated at, which is how a sub-matrix's expected payload is produced.
Bytes build_contiguous(std::uint64_t rows, std::uint64_t columns, std::uint64_t row_origin,
                       std::uint64_t column_origin) {
    Bytes out(static_cast<std::size_t>(rows * columns * 2));
    for (std::uint64_t n = 0; n < rows; ++n) {
        for (std::uint64_t k = 0; k < columns; ++k) {
            const std::size_t at = static_cast<std::size_t>((n * columns + k) * 2);
            out[at]              = pattern(row_origin + n, column_origin + k, 0);
            out[at + 1]          = pattern(row_origin + n, column_origin + k, 1);
        }
    }
    return out;
}

// row-split-k128-v1 (storage-layouts.md 3.2-3.6): base plane, 256-aligned high plane, 256-aligned
// binary16 scale plane; every plane traverses row-major by group.
Bytes build_row_split(NumericFormat format, std::uint64_t rows, std::uint64_t columns,
                      std::uint64_t row_origin, std::uint64_t group_origin) {
    const std::array<std::uint64_t, 2> shape = {rows, columns};
    const auto geometry = ninfer::artifact::row_split_geometry(format, shape);
    Bytes out(static_cast<std::size_t>(geometry.encoded_bytes), 0);
    for (std::uint64_t n = 0; n < rows; ++n) {
        for (std::uint64_t g = 0; g < geometry.groups_per_row; ++g) {
            const std::uint64_t index = n * geometry.groups_per_row + g;
            for (std::uint64_t i = 0; i < geometry.low_bytes_per_group; ++i) {
                out[static_cast<std::size_t>(index * geometry.low_bytes_per_group + i)] =
                    pattern(row_origin + n, group_origin + g, i);
            }
            for (std::uint64_t i = 0; i < geometry.high_bytes_per_group; ++i) {
                out[static_cast<std::size_t>(geometry.high_plane_offset +
                                             index * geometry.high_bytes_per_group + i)] =
                    pattern(row_origin + n, group_origin + g, 100 + i);
            }
            for (std::uint64_t i = 0; i < 2; ++i) {
                out[static_cast<std::size_t>(geometry.scale_plane_offset + index * 2 + i)] =
                    pattern(row_origin + n, group_origin + g, 200 + i);
            }
        }
    }
    return out;
}

// blockscale-k16-m128x4-v1 (storage-layouts.md 4): row-major E2M1 code plane, then the swizzled
// E4M3FN scale plane, then the FP32 weight divisor. The swizzle below is the doc's formula, which
// is the same map tools/artifact/layouts.py swizzle_nvfp4_scales produces.
Bytes build_block_scale(std::uint64_t rows, std::uint64_t columns, std::uint64_t row_origin,
                        std::uint64_t group_origin) {
    const std::array<std::uint64_t, 2> shape = {rows, columns};
    const auto geometry = ninfer::artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    Bytes out(static_cast<std::size_t>(geometry.encoded_bytes), 0);
    for (std::uint64_t n = 0; n < rows; ++n) {
        for (std::uint64_t j = 0; j < columns / 2; ++j) {
            out[static_cast<std::size_t>(n * (columns / 2) + j)] =
                pattern(row_origin + n, j + group_origin * 8, 0);
        }
        for (std::uint64_t g = 0; g < geometry.groups_per_row; ++g) {
            const std::uint64_t row_tile   = n / 128;
            const std::uint64_t row_inner  = n % 128;
            const std::uint64_t scale_tile = g / 4;
            const std::uint64_t offset     = (row_tile * geometry.k_tiles + scale_tile) * 512 +
                                         (row_inner % 32) * 16 + (row_inner / 32) * 4 + (g % 4);
            out[static_cast<std::size_t>(geometry.scale_plane_offset + offset)] =
                pattern(row_origin + n, group_origin + g, 200);
        }
    }
    for (std::uint64_t i = 0; i < 4; ++i) {
        out[static_cast<std::size_t>(geometry.weight_divisor_offset + i)] =
            static_cast<std::uint8_t>(0x30 + i);
    }
    return out;
}

// row-scale-v1 (storage-layouts.md 5): row-major E4M3FN code plane, 256-aligned BF16 row scales.
Bytes build_row_scale(std::uint64_t rows, std::uint64_t columns, std::uint64_t row_origin,
                      std::uint64_t column_origin) {
    const std::array<std::uint64_t, 2> shape = {rows, columns};
    const auto geometry =
        ninfer::artifact::row_scale_geometry(NumericFormat::FP8_E4M3FN_ROW_BF16S, shape);
    Bytes out(static_cast<std::size_t>(geometry.encoded_bytes), 0);
    for (std::uint64_t n = 0; n < rows; ++n) {
        for (std::uint64_t k = 0; k < columns; ++k) {
            out[static_cast<std::size_t>(n * columns + k)] =
                pattern(row_origin + n, column_origin + k, 0);
        }
        for (std::uint64_t i = 0; i < 2; ++i) {
            out[static_cast<std::size_t>(geometry.scale_plane_offset + n * 2 + i)] =
                pattern(row_origin + n, 0, 200 + i);
        }
    }
    return out;
}

template <typename Fn> void expect_throws(Fn&& fn, const std::string& label) {
    try {
        fn();
    } catch (const std::exception&) { return; }
    fail(label + ": expected a throw, none occurred");
}

// tensor_column_slice takes a range list; most cases below pass exactly one. The returned array
// is a temporary whose lifetime covers the full expression the span is used in.
std::array<SliceRange, 1> one(SliceRange range) { return {range}; }

} // namespace

int main() {
    // --- contiguous-le-v1 -----------------------------------------------------------------------
    {
        constexpr std::uint64_t kRows = 8;
        constexpr std::uint64_t kCols = 6;
        const std::array<std::uint64_t, 2> shape = {kRows, kCols};
        const Bytes parent                       = build_contiguous(kRows, kCols, 0, 0);

        const std::array<SliceRange, 2> rows = {SliceRange{0, 2}, SliceRange{5, 3}};
        const TensorSlice row_slice = ninfer::artifact::tensor_row_slice(
            StorageLayout::ContiguousLeV1, NumericFormat::BF16, shape, rows);
        Bytes expected = build_contiguous(2, kCols, 0, 0);
        const Bytes tail = build_contiguous(3, kCols, 5, 0);
        expected.insert(expected.end(), tail.begin(), tail.end());
        expect_equal(apply_slice(parent, row_slice), expected, "contiguous row slice");

        const TensorSlice column_slice = ninfer::artifact::tensor_column_slice(
            StorageLayout::ContiguousLeV1, NumericFormat::BF16, shape, one(SliceRange{2, 3}));
        expect_equal(apply_slice(parent, column_slice), build_contiguous(kRows, 3, 0, 2),
                     "contiguous column slice");

        // Multi-range column slice (contiguous-le-v1 only): the shard is the ranges concatenated
        // WITHIN every row, in the order given. This is the shape a depthwise channel split needs
        // -- GDN `gdn/convolution` [4 taps, 10240 channels] hands each device one channel block
        // from each of the Q|K|V sections.
        {
            const std::array<SliceRange, 3> columns = {SliceRange{0, 1}, SliceRange{2, 2},
                                                       SliceRange{5, 1}};
            const TensorSlice multi = ninfer::artifact::tensor_column_slice(
                StorageLayout::ContiguousLeV1, NumericFormat::BF16, shape, columns);
            Bytes expected(static_cast<std::size_t>(kRows * 4 * 2));
            const std::array<std::uint64_t, 4> source = {0, 2, 3, 5};
            for (std::uint64_t n = 0; n < kRows; ++n) {
                for (std::uint64_t k = 0; k < 4; ++k) {
                    const auto at = static_cast<std::size_t>((n * 4 + k) * 2);
                    expected[at]     = pattern(n, source[k], 0);
                    expected[at + 1] = pattern(n, source[k], 1);
                }
            }
            expect_equal(apply_slice(parent, multi), expected, "contiguous multi-range column");
            if (multi.encoded_bytes != kRows * 4 * 2) {
                fail("multi-range column encoded bytes: got " +
                     std::to_string(multi.encoded_bytes));
            }
        }
        expect_throws(
            [&] {
                const std::array<SliceRange, 2> columns = {SliceRange{2, 2}, SliceRange{1, 1}};
                (void)ninfer::artifact::tensor_column_slice(StorageLayout::ContiguousLeV1,
                                                            NumericFormat::BF16, shape, columns);
            },
            "descending column ranges");
        expect_throws(
            [&] {
                const std::array<SliceRange, 2> columns = {SliceRange{0, 3}, SliceRange{2, 2}};
                (void)ninfer::artifact::tensor_column_slice(StorageLayout::ContiguousLeV1,
                                                            NumericFormat::BF16, shape, columns);
            },
            "overlapping column ranges");
    }

    // --- row-split-k128-v1: Q5 has all three planes, W8 has no high plane --------------------
    for (const NumericFormat format : {NumericFormat::Q5G64_F16S, NumericFormat::W8G32_F16S}) {
        constexpr std::uint64_t kRows = 6;
        constexpr std::uint64_t kCols = 256;
        const std::array<std::uint64_t, 2> shape = {kRows, kCols};
        const auto geometry = ninfer::artifact::row_split_geometry(format, shape);
        const std::string label(ninfer::artifact::format_name(format));
        const Bytes parent = build_row_split(format, kRows, kCols, 0, 0);

        // Rows are independently addressable: any boundary is legal, including a non-128 one.
        const std::array<SliceRange, 2> rows = {SliceRange{1, 2}, SliceRange{4, 2}};
        const TensorSlice row_slice = ninfer::artifact::tensor_row_slice(
            StorageLayout::RowSplitK128V1, format, shape, rows);
        Bytes expected = build_row_split(format, 4, kCols, 0, 0);
        {
            // The concatenation is [rows 1,2] then [rows 4,5]; rebuild it plane by plane by
            // overwriting the expected buffer's per-row entries with the parent's coordinates.
            const auto shard_shape = std::array<std::uint64_t, 2>{4, kCols};
            const auto shard       = ninfer::artifact::row_split_geometry(format, shard_shape);
            const std::array<std::uint64_t, 4> source_rows = {1, 2, 4, 5};
            for (std::uint64_t n = 0; n < 4; ++n) {
                for (std::uint64_t g = 0; g < shard.groups_per_row; ++g) {
                    const std::uint64_t index = n * shard.groups_per_row + g;
                    for (std::uint64_t i = 0; i < shard.low_bytes_per_group; ++i) {
                        expected[static_cast<std::size_t>(index * shard.low_bytes_per_group + i)] =
                            pattern(source_rows[n], g, i);
                    }
                    for (std::uint64_t i = 0; i < shard.high_bytes_per_group; ++i) {
                        expected[static_cast<std::size_t>(shard.high_plane_offset +
                                                          index * shard.high_bytes_per_group + i)] =
                            pattern(source_rows[n], g, 100 + i);
                    }
                    for (std::uint64_t i = 0; i < 2; ++i) {
                        expected[static_cast<std::size_t>(shard.scale_plane_offset + index * 2 +
                                                          i)] = pattern(source_rows[n], g, 200 + i);
                    }
                }
            }
        }
        expect_equal(apply_slice(parent, row_slice), expected, label + " row-split row slice");

        // Columns must land on the 128-column K-alignment unit; group_origin is the skipped group
        // count, which is where the sub-matrix's pattern starts.
        const TensorSlice column_slice = ninfer::artifact::tensor_column_slice(
            StorageLayout::RowSplitK128V1, format, shape, one(SliceRange{128, 128}));
        expect_equal(apply_slice(parent, column_slice),
                     build_row_split(format, kRows, 128, 0, 128 / geometry.group_size),
                     label + " row-split column slice");

        expect_throws(
            [&] {
                (void)ninfer::artifact::tensor_column_slice(StorageLayout::RowSplitK128V1, format,
                                                            shape, one(SliceRange{64, 64}));
            },
            label + " row-split column slice off the 128 boundary");

        // Grouped/tiled/swizzled column axes cannot express a multi-range shard; it must be
        // rejected, not mis-encoded. (contiguous-le-v1 is the sole exception -- see above.)
        expect_throws(
            [&] {
                const std::array<SliceRange, 2> columns = {SliceRange{0, 128},
                                                           SliceRange{128, 128}};
                (void)ninfer::artifact::tensor_column_slice(StorageLayout::RowSplitK128V1, format,
                                                            shape, columns);
            },
            label + " row-split multi-range column slice");

        // K not a multiple of 128: the parent pads to K_pad and carries trailing padding groups.
        // A leading 128-column slice is still exact -- its own K_pad equals its K, so it needs
        // none of the parent's padding -- while the remainder cannot be expressed as a shard at
        // all, and must be rejected rather than silently dropping or inheriting padding groups.
        {
            constexpr std::uint64_t kRagged                 = 200; // K_pad 256
            const std::array<std::uint64_t, 2> ragged_shape = {kRows, kRagged};
            const auto ragged = ninfer::artifact::row_split_geometry(format, ragged_shape);
            const Bytes ragged_parent = build_row_split(format, kRows, kRagged, 0, 0);
            const TensorSlice head    = ninfer::artifact::tensor_column_slice(
                StorageLayout::RowSplitK128V1, format, ragged_shape, one(SliceRange{0, 128}));
            expect_equal(apply_slice(ragged_parent, head),
                         build_row_split(format, kRows, 128, 0, 0),
                         label + " ragged-K leading column slice");
            if (ragged.padded_columns != 256) {
                fail(label + " ragged-K fixture is not exercising padding groups");
            }
            expect_throws(
                [&] {
                    (void)ninfer::artifact::tensor_column_slice(StorageLayout::RowSplitK128V1,
                                                                format, ragged_shape,
                                                                one(SliceRange{128, 72}));
                },
                label + " ragged-K trailing column slice (remainder is not 128-aligned)");
        }
    }

    // --- blockscale-k16-m128x4-v1 (NVFP4) ---------------------------------------------------
    {
        constexpr std::uint64_t kRows = 384; // three 128-row tiles
        constexpr std::uint64_t kCols = 192; // three 64-column scale tiles
        const std::array<std::uint64_t, 2> shape = {kRows, kCols};
        const Bytes parent                       = build_block_scale(kRows, kCols, 0, 0);

        // Row slice on 128-row tile boundaries, two disjoint ranges (a fused object's two blocks).
        const std::array<SliceRange, 2> rows = {SliceRange{0, 128}, SliceRange{256, 128}};
        const TensorSlice row_slice = ninfer::artifact::tensor_row_slice(
            StorageLayout::BlockScaleK16M128x4V1, NumericFormat::NVFP4, shape, rows);
        Bytes expected = build_block_scale(256, kCols, 0, 0);
        {
            const auto shard_shape = std::array<std::uint64_t, 2>{256, kCols};
            const auto shard =
                ninfer::artifact::block_scale_geometry(NumericFormat::NVFP4, shard_shape);
            for (std::uint64_t n = 0; n < 256; ++n) {
                const std::uint64_t source = n < 128 ? n : 256 + (n - 128);
                for (std::uint64_t j = 0; j < kCols / 2; ++j) {
                    expected[static_cast<std::size_t>(n * (kCols / 2) + j)] = pattern(source, j, 0);
                }
                for (std::uint64_t g = 0; g < shard.groups_per_row; ++g) {
                    const std::uint64_t offset =
                        ((n / 128) * shard.k_tiles + g / 4) * 512 + (n % 32) * 16 +
                        ((n % 128) / 32) * 4 + (g % 4);
                    expected[static_cast<std::size_t>(shard.scale_plane_offset + offset)] =
                        pattern(source, g, 200);
                }
            }
        }
        expect_equal(apply_slice(parent, row_slice), expected, "NVFP4 row slice");

        // Column slice on 64-column scale-tile boundaries.
        const TensorSlice column_slice = ninfer::artifact::tensor_column_slice(
            StorageLayout::BlockScaleK16M128x4V1, NumericFormat::NVFP4, shape,
            one(SliceRange{64, 128}));
        expect_equal(apply_slice(parent, column_slice),
                     build_block_scale(kRows, 128, 0, 64 / 16), "NVFP4 column slice");

        expect_throws(
            [&] {
                const std::array<SliceRange, 1> bad = {SliceRange{64, 128}};
                (void)ninfer::artifact::tensor_row_slice(StorageLayout::BlockScaleK16M128x4V1,
                                                         NumericFormat::NVFP4, shape, bad);
            },
            "NVFP4 row slice off the 128-row tile boundary");
        expect_throws(
            [&] {
                (void)ninfer::artifact::tensor_column_slice(StorageLayout::BlockScaleK16M128x4V1,
                                                            NumericFormat::NVFP4, shape,
                                                            one(SliceRange{32, 64}));
            },
            "NVFP4 column slice off the 64-column tile boundary");
    }

    // --- row-scale-v1 (FP8) -----------------------------------------------------------------
    {
        constexpr std::uint64_t kRows            = 6;
        constexpr std::uint64_t kCols            = 40;
        const std::array<std::uint64_t, 2> shape = {kRows, kCols};
        const Bytes parent                       = build_row_scale(kRows, kCols, 0, 0);

        const std::array<SliceRange, 1> rows = {SliceRange{2, 3}};
        const TensorSlice row_slice          = ninfer::artifact::tensor_row_slice(
            StorageLayout::RowScaleV1, NumericFormat::FP8_E4M3FN_ROW_BF16S, shape, rows);
        expect_equal(apply_slice(parent, row_slice), build_row_scale(3, kCols, 2, 0),
                     "FP8 row slice");

        // A column slice keeps every row, so every row's BF16 multiplier is replicated verbatim.
        const TensorSlice column_slice = ninfer::artifact::tensor_column_slice(
            StorageLayout::RowScaleV1, NumericFormat::FP8_E4M3FN_ROW_BF16S, shape,
            one(SliceRange{8, 16}));
        expect_equal(apply_slice(parent, column_slice), build_row_scale(kRows, 16, 0, 8),
                     "FP8 column slice");
    }

    // GGML_K keeps row-dependent Q4_K/Q6_K block bytes, and rebuilds shard row descriptors.
    {
        const auto build_blocks = [](std::span<const std::uint64_t> selected,
                                      std::span<const std::uint64_t> blocks) {
            const auto prefix = ((selected.size() * 8 + 255) / 256) * 256;
            Bytes out(prefix, 0);
            for (std::size_t row = 0; row < selected.size(); ++row) {
                const auto original = selected[row];
                const auto tag = original % 2;
                const auto descriptor = ((out.size() - prefix) << 1) | tag;
                for (unsigned byte = 0; byte < 8; ++byte) {
                    out[row * 8 + byte] = static_cast<std::uint8_t>(descriptor >> (8 * byte));
                }
                const auto width = tag ? 210 : 144;
                for (const auto block : blocks) {
                    for (std::uint64_t byte = 0; byte < width; ++byte) {
                        out.push_back(pattern(original, block, byte));
                    }
                }
            }
            return out;
        };
        const auto build = [&](std::span<const std::uint64_t> selected, std::uint64_t first_block,
                                std::uint64_t count) {
            std::vector<std::uint64_t> blocks;
            for (auto block = first_block; block < first_block + count; ++block) {
                blocks.push_back(block);
            }
            return build_blocks(selected, blocks);
        };
        const std::array<std::uint64_t, 5> selected = {0, 1, 2, 3, 4};
        const Bytes parent = build(selected, 0, 4);
        const auto payload = std::as_bytes(std::span(parent));
        const std::array<std::uint64_t, 2> shape = {5, 1024};
        const std::array<SliceRange, 2> ranges = {SliceRange{0, 2}, SliceRange{3, 1}};
        const std::array<std::uint64_t, 3> row_indices = {0, 1, 3};
        const auto row_slice = ninfer::artifact::tensor_row_slice(
            StorageLayout::GgmlK256V1, NumericFormat::GGML_K, shape, ranges, payload);
        expect_equal(apply_slice(parent, row_slice), build(row_indices, 0, 4), "GGML_K mixed rows");
        const auto column_slice = ninfer::artifact::tensor_column_slice(
            StorageLayout::GgmlK256V1, NumericFormat::GGML_K, shape,
            one(SliceRange{256, 512}), payload);
        expect_equal(apply_slice(parent, column_slice), build(selected, 1, 2), "GGML_K columns");
        expect_throws([&] {
            (void)ninfer::artifact::tensor_column_slice(StorageLayout::GgmlK256V1,
                NumericFormat::GGML_K, shape, one(SliceRange{128, 512}), payload);
        }, "GGML_K partial block split");

        // The preserved GGUF GDN output has tiled [repeat,key,128] columns.
        // TP2 takes eight key heads from each of three repeat sections, with
        // both Q4_K and Q6_K block payloads copied exactly into each shard.
        const Bytes gdn_parent = build(selected, 0, 24);
        const std::array<std::uint64_t, 2> gdn_shape = {5, 6144};
        for (std::uint64_t rank = 0; rank < 2; ++rank) {
            const std::array<SliceRange, 3> columns = {
                SliceRange{rank * 1024, 1024}, SliceRange{2048 + rank * 1024, 1024},
                SliceRange{4096 + rank * 1024, 1024}};
            const auto slice = ninfer::artifact::tensor_column_slice(
                StorageLayout::GgmlK256V1, NumericFormat::GGML_K, gdn_shape, columns,
                std::as_bytes(std::span(gdn_parent)));
            std::vector<std::uint64_t> blocks;
            for (std::uint64_t repeat = 0; repeat < 3; ++repeat) {
                for (std::uint64_t block = 0; block < 4; ++block) {
                    blocks.push_back(repeat * 8 + rank * 4 + block);
                }
            }
            expect_equal(apply_slice(gdn_parent, slice), build_blocks(selected, blocks),
                         "GGML_K GDN multi-range columns rank " + std::to_string(rank));
        }
        expect_throws([&] {
            const std::array<SliceRange, 2> columns = {SliceRange{0, 256}, SliceRange{384, 256}};
            (void)ninfer::artifact::tensor_column_slice(StorageLayout::GgmlK256V1,
                NumericFormat::GGML_K, shape, columns, payload);
        }, "GGML_K partial block in later column range");
    }

    // --- range validation -------------------------------------------------------------------
    {
        const std::array<std::uint64_t, 2> shape = {8, 8};
        expect_throws(
            [&] {
                const std::array<SliceRange, 2> descending = {SliceRange{4, 4}, SliceRange{0, 2}};
                (void)ninfer::artifact::tensor_row_slice(StorageLayout::ContiguousLeV1,
                                                         NumericFormat::BF16, shape, descending);
            },
            "descending row ranges");
        expect_throws(
            [&] {
                const std::array<SliceRange, 2> overlapping = {SliceRange{0, 5}, SliceRange{4, 2}};
                (void)ninfer::artifact::tensor_row_slice(StorageLayout::ContiguousLeV1,
                                                         NumericFormat::BF16, shape, overlapping);
            },
            "overlapping row ranges");
        expect_throws(
            [&] {
                const std::array<SliceRange, 1> past_end = {SliceRange{6, 4}};
                (void)ninfer::artifact::tensor_row_slice(StorageLayout::ContiguousLeV1,
                                                         NumericFormat::BF16, shape, past_end);
            },
            "row range past the last row");
        expect_throws(
            [&] {
                (void)ninfer::artifact::tensor_column_slice(
                    StorageLayout::ContiguousLeV1, NumericFormat::BF16, shape,
                    one(SliceRange{6, 4}));
            },
            "column range past the last column");
    }

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "tensor slice table test: all checks passed\n";
    return 0;
}
