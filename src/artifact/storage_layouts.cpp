#include "artifact/reader.h"

#include <array>
#include <limits>
#include <string>

namespace ninfer::artifact {
namespace {

constexpr std::uint64_t kTensorAlignment = 256;
constexpr std::uint64_t kKAlignment      = 128;

// blockscale-k16-m128x4-v1 physical tiling constants (storage-layouts.md section 4, and
// tools/artifact/layouts.py swizzle_nvfp4_scales, which is the encoder of record).
constexpr std::uint64_t kNvfp4RowTile      = 128; // rows per scale-plane tile row
constexpr std::uint64_t kNvfp4TileColumns  = 64;  // logical columns per scale tile (4 groups of 16)
constexpr std::uint64_t kNvfp4TileBytes    = 512; // 128 rows x 4 lanes of one scale tile
constexpr std::uint64_t kNvfp4DivisorBytes = 4;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a * b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

struct QuantGeometry {
    std::uint64_t group_size;
    std::uint64_t base_bytes_per_group;
    std::uint64_t high_bytes_per_group;
};

QuantGeometry quant_geometry(NumericFormat format) {
    switch (format) {
    case NumericFormat::Q4G64_F16S:
        return {64, 32, 0};
    case NumericFormat::Q5G64_F16S:
        return {64, 32, 8};
    case NumericFormat::Q6G64_F16S:
        return {64, 32, 16};
    case NumericFormat::W8G32_F16S:
        return {32, 32, 0};
    default:
        throw ArtifactError("row-split-k128-v1 requires a grouped quantized format");
    }
}

std::uint64_t direct_word_bytes(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
        return 2;
    case NumericFormat::FP32:
    case NumericFormat::I32:
        return 4;
    default:
        throw ArtifactError("contiguous-le-v1 requires BF16, FP32, or I32");
    }
}

} // namespace

std::uint64_t ggml_k_code_offset(std::uint64_t rows) {
    return align_up(checked_mul(rows, 8, "GGML row descriptors"), 256, "GGML code offset");
}

namespace {
std::uint64_t ggml_row(std::span<const std::byte> payload, std::uint64_t row) {
    std::uint64_t value = 0;
    for (unsigned byte = 0; byte < 8; ++byte) {
        value |= static_cast<std::uint64_t>(std::to_integer<unsigned>(payload[row * 8 + byte])) << (byte * 8);
    }
    return value;
}

void set_ggml_row(std::vector<std::byte>& prefix, std::uint64_t row, std::uint64_t value) {
    for (unsigned byte = 0; byte < 8; ++byte) {
        prefix[row * 8 + byte] = static_cast<std::byte>((value >> (byte * 8)) & 255);
    }
}
}

void validate_ggml_k_payload(std::span<const std::uint64_t> shape,
                             std::span<const std::byte> payload) {
    tensor_encoded_size(StorageLayout::GgmlK256V1, NumericFormat::GGML_K, shape, payload.size());
    std::uint64_t cursor = 0;
    for (std::uint64_t row = 0; row < shape[0]; ++row) {
        const auto descriptor = ggml_row(payload, row);
        if ((descriptor >> 1) != cursor) {
            throw ArtifactError("GGML_K row offsets are not canonical contiguous rows");
        }
        cursor = checked_add(cursor, (shape[1] / 256) * ((descriptor & 1) ? 210 : 144), "GGML row bytes");
    }
    const auto code_offset = ggml_k_code_offset(shape[0]);
    if (checked_add(code_offset, cursor, "GGML payload bytes") != payload.size()) {
        throw ArtifactError("GGML_K descriptors disagree with payload byte size");
    }
    for (auto offset = shape[0] * 8; offset < code_offset; ++offset) {
        if (payload[offset] != std::byte{0}) { throw ArtifactError("GGML_K descriptor padding is not zero"); }
    }
}

std::string_view format_name(NumericFormat format) noexcept {
    switch (format) {
    case NumericFormat::BF16:
        return "BF16";
    case NumericFormat::FP32:
        return "FP32";
    case NumericFormat::I32:
        return "I32";
    case NumericFormat::Q4G64_F16S:
        return "Q4G64_F16S";
    case NumericFormat::Q5G64_F16S:
        return "Q5G64_F16S";
    case NumericFormat::Q6G64_F16S:
        return "Q6G64_F16S";
    case NumericFormat::W8G32_F16S:
        return "W8G32_F16S";
    case NumericFormat::NVFP4:
        return "NVFP4";
    case NumericFormat::FP8_E4M3FN_ROW_BF16S:
        return "FP8_E4M3FN_ROW_BF16S";
    case NumericFormat::GGML_K:
        return "GGML_K";
    }
    return {};
}

std::string_view layout_name(StorageLayout layout) noexcept {
    switch (layout) {
    case StorageLayout::ContiguousLeV1:
        return "contiguous-le-v1";
    case StorageLayout::RowSplitK128V1:
        return "row-split-k128-v1";
    case StorageLayout::BlockScaleK16M128x4V1:
        return "blockscale-k16-m128x4-v1";
    case StorageLayout::RowScaleV1:
        return "row-scale-v1";
    case StorageLayout::GgmlK256V1:
        return "ggml-k256-v1";
    }
    return {};
}

std::string_view encoding_name(ResourceEncoding encoding) noexcept {
    switch (encoding) {
    case ResourceEncoding::RawBytesV1:
        return "raw-bytes-v1";
    }
    return {};
}

std::uint64_t tensor_alignment(StorageLayout) noexcept { return kTensorAlignment; }

std::uint64_t resource_alignment(ResourceEncoding) noexcept { return 1; }

std::uint64_t tensor_encoded_size(StorageLayout layout, NumericFormat format,
                                  std::span<const std::uint64_t> shape,
                                  std::uint64_t stored_bytes) {
    if (layout == StorageLayout::GgmlK256V1) {
        if (format != NumericFormat::GGML_K || shape.size() != 2 || shape[0] == 0 ||
            shape[1] == 0 || shape[1] % 256 != 0) {
            throw ArtifactError("ggml-k256-v1 requires GGML_K [N,K] with K divisible by 256");
        }
        const auto blocks = checked_mul(shape[0], shape[1] / 256, "GGML K blocks");
        const auto prefix = ggml_k_code_offset(shape[0]);
        const auto minimum = checked_add(prefix, checked_mul(blocks, 144, "GGML Q4 bytes"), "GGML bytes");
        const auto maximum = checked_add(prefix, checked_mul(blocks, 210, "GGML Q6 bytes"), "GGML bytes");
        if (stored_bytes < minimum || stored_bytes > maximum ||
            (stored_bytes - minimum) % ((shape[1] / 256) * 66) != 0) {
            throw ArtifactError("ggml-k256-v1 byte size does not describe whole Q4_K/Q6_K rows");
        }
        return stored_bytes;
    }
    if (layout == StorageLayout::ContiguousLeV1) {
        if (shape.size() > 16) {
            throw ArtifactError("contiguous-le-v1 supports rank 0 through 16");
        }
        std::uint64_t elements = 1;
        for (const auto dim : shape) {
            if (dim == 0) { throw ArtifactError("tensor shape dimensions must be positive"); }
            elements = checked_mul(elements, dim, "tensor element count");
        }
        return checked_mul(elements, direct_word_bytes(format), "tensor encoded size");
    }

    if (layout == StorageLayout::RowSplitK128V1) {
        if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
            throw ArtifactError("row-split-k128-v1 requires a positive rank-two shape");
        }
        return row_split_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::BlockScaleK16M128x4V1) {
        return block_scale_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::RowScaleV1) {
        return row_scale_geometry(format, shape).encoded_bytes;
    }
    throw ArtifactError("unknown tensor layout");
}

RowSplitGeometry row_split_geometry(NumericFormat format, std::span<const std::uint64_t> shape) {
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("row-split-k128-v1 requires a positive rank-two shape");
    }
    const auto format_geometry = quant_geometry(format);
    RowSplitGeometry out;
    out.rows                 = shape[0];
    out.columns              = shape[1];
    out.padded_columns       = align_up(shape[1], kKAlignment, "padded K");
    out.group_size           = format_geometry.group_size;
    out.groups_per_row       = out.padded_columns / out.group_size;
    out.low_bytes_per_group  = format_geometry.base_bytes_per_group;
    out.high_bytes_per_group = format_geometry.high_bytes_per_group;
    const auto groups        = checked_mul(out.rows, out.groups_per_row, "physical group count");
    out.low_plane_bytes      = checked_mul(groups, out.low_bytes_per_group, "base plane bytes");
    out.high_plane_bytes     = checked_mul(groups, out.high_bytes_per_group, "high plane bytes");
    out.scale_plane_bytes    = checked_mul(groups, 2, "scale plane bytes");
    out.high_plane_offset    = align_up(out.low_plane_bytes, kTensorAlignment, "high plane offset");
    const auto aligned_high =
        align_up(out.high_plane_bytes, kTensorAlignment, "scale plane alignment");
    out.scale_plane_offset = checked_add(out.high_plane_offset, aligned_high, "scale plane offset");
    out.encoded_bytes =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "tensor encoded size");
    return out;
}

BlockScaleGeometry block_scale_geometry(NumericFormat format,
                                        std::span<const std::uint64_t> shape) {
    if (format != NumericFormat::NVFP4) {
        throw ArtifactError("blockscale-k16-m128x4-v1 requires NVFP4");
    }
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("blockscale-k16-m128x4-v1 requires a positive rank-two shape");
    }
    if (shape[0] % 128 != 0 || shape[1] % 64 != 0) {
        throw ArtifactError(
            "blockscale-k16-m128x4-v1 requires N divisible by 128 and K divisible by 64");
    }

    BlockScaleGeometry out;
    out.rows             = shape[0];
    out.columns          = shape[1];
    out.groups_per_row   = shape[1] / 16;
    out.k_tiles          = shape[1] / 64;
    const auto elements  = checked_mul(out.rows, out.columns, "NVFP4 element count");
    out.code_plane_bytes = elements / 2;
    out.scale_plane_offset =
        align_up(out.code_plane_bytes, kTensorAlignment, "NVFP4 scale plane offset");
    out.scale_plane_bytes = elements / 16;
    out.weight_divisor_offset =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "NVFP4 weight divisor offset");
    out.encoded_bytes = checked_add(out.weight_divisor_offset, 4, "NVFP4 tensor encoded size");
    return out;
}

namespace {

// One plane of a multi-plane layout, described in the units the slice arithmetic needs.
struct SlicePlane {
    std::uint64_t source_base = 0; // plane offset inside the parent payload
    std::uint64_t dest_base   = 0; // plane offset inside the shard payload
    std::uint64_t unit_bytes  = 0; // bytes per source unit (row, row tile, ...)
};

void require_slice(bool condition, std::string_view reason) {
    if (!condition) { throw ArtifactError("tensor slice: " + std::string(reason)); }
}

std::uint64_t validate_row_ranges(std::span<const SliceRange> rows, std::uint64_t tensor_rows,
                                  std::uint64_t alignment) {
    require_slice(!rows.empty(), "at least one row range is required");
    std::uint64_t total        = 0;
    std::uint64_t previous_end = 0;
    for (const SliceRange& range : rows) {
        require_slice(range.count != 0, "a row range must be nonempty");
        require_slice(range.begin >= previous_end, "row ranges must be ascending and disjoint");
        require_slice(range.begin <= tensor_rows && tensor_rows - range.begin >= range.count,
                      "a row range reaches past the last row");
        require_slice(range.begin % alignment == 0 && range.count % alignment == 0,
                      "a row range does not land on this layout's row-tile boundary");
        previous_end = range.begin + range.count;
        total        = checked_add(total, range.count, "row slice row count");
    }
    return total;
}

std::uint64_t row_element_stride(std::span<const std::uint64_t> shape) {
    std::uint64_t elements = 1;
    for (std::size_t i = 1; i < shape.size(); ++i) {
        require_slice(shape[i] != 0, "tensor shape dimensions must be positive");
        elements = checked_mul(elements, shape[i], "row element count");
    }
    return elements;
}

// Appends, for every selected row range, one contiguous copy per nonempty plane. Row-addressable
// planes only: `unit_bytes` is the plane's bytes-per-row (or bytes-per-row-tile, with `rows`
// pre-divided by the tile height).
void append_row_plane_copies(TensorSlice& out, std::span<const SlicePlane> planes,
                             std::span<const SliceRange> rows, std::uint64_t unit_rows) {
    for (const SlicePlane& plane : planes) {
        if (plane.unit_bytes == 0) { continue; }
        std::uint64_t destination = 0;
        for (const SliceRange& range : rows) {
            const std::uint64_t units = range.count / unit_rows;
            out.copies.push_back(PlaneCopy{
                .source_offset = checked_add(plane.source_base,
                                             checked_mul(range.begin / unit_rows, plane.unit_bytes,
                                                         "row slice source offset"),
                                             "row slice source offset"),
                .dest_offset   = checked_add(plane.dest_base,
                                             checked_mul(destination, plane.unit_bytes,
                                                         "row slice destination offset"),
                                             "row slice destination offset"),
                .bytes         = checked_mul(units, plane.unit_bytes, "row slice copy bytes"),
            });
            destination += units;
        }
    }
}

// Appends one copy per row for every plane whose rows are strided in the parent (a column slice
// keeps every row but narrows it, so no plane of any layout stays contiguous across rows).
//
// `select_units` is how many units of each row this call copies and `dest_skip_units` where they
// land inside the shard's own (possibly wider) row; a single-range slice passes
// `select_units == dest_units_per_row` and `dest_skip_units == 0`, which is what every quantized
// layout does. A multi-range contiguous-le-v1 slice calls this once per range with the same
// `dest_units_per_row` and an advancing `dest_skip_units`, so the ranges concatenate into one
// narrowed row in the order given.
void append_column_plane_copies(TensorSlice& out, std::span<const SlicePlane> planes,
                                std::uint64_t rows, std::uint64_t source_skip_units,
                                std::uint64_t source_units_per_row,
                                std::uint64_t dest_units_per_row, std::uint64_t select_units,
                                std::uint64_t dest_skip_units = 0) {
    for (const SlicePlane& plane : planes) {
        if (plane.unit_bytes == 0) { continue; }
        const std::uint64_t source_stride =
            checked_mul(source_units_per_row, plane.unit_bytes, "column slice source stride");
        const std::uint64_t dest_stride =
            checked_mul(dest_units_per_row, plane.unit_bytes, "column slice destination stride");
        const std::uint64_t skip =
            checked_mul(source_skip_units, plane.unit_bytes, "column slice source skip");
        const std::uint64_t dest_skip =
            checked_mul(dest_skip_units, plane.unit_bytes, "column slice destination skip");
        const std::uint64_t bytes =
            checked_mul(select_units, plane.unit_bytes, "column slice copy bytes");
        for (std::uint64_t row = 0; row < rows; ++row) {
            out.copies.push_back(PlaneCopy{
                .source_offset = checked_add(
                    plane.source_base,
                    checked_add(checked_mul(row, source_stride, "column slice source offset"), skip,
                                "column slice source offset"),
                    "column slice source offset"),
                .dest_offset   = checked_add(
                    plane.dest_base,
                    checked_add(checked_mul(row, dest_stride, "column slice destination offset"),
                                dest_skip, "column slice destination offset"),
                    "column slice destination offset"),
                .bytes         = bytes,
            });
        }
    }
}

// Same contract as validate_row_ranges, on the column axis. Kept separate so the diagnostics name
// the axis the caller actually asked for.
std::uint64_t validate_column_ranges(std::span<const SliceRange> columns,
                                     std::uint64_t tensor_columns) {
    require_slice(!columns.empty(), "at least one column range is required");
    std::uint64_t total        = 0;
    std::uint64_t previous_end = 0;
    for (const SliceRange& range : columns) {
        require_slice(range.count != 0, "a column range must be nonempty");
        require_slice(range.begin >= previous_end, "column ranges must be ascending and disjoint");
        require_slice(range.begin <= tensor_columns &&
                          tensor_columns - range.begin >= range.count,
                      "the column range reaches past the last column");
        previous_end = range.begin + range.count;
        total        = checked_add(total, range.count, "column slice column count");
    }
    return total;
}

} // namespace

TensorSlice tensor_row_slice(StorageLayout layout, NumericFormat format,
                             std::span<const std::uint64_t> shape,
                             std::span<const SliceRange> rows,
                             std::span<const std::byte> payload) {
    require_slice(!shape.empty(), "a row slice needs a tensor of rank one or higher");
    TensorSlice out;

    if (layout == StorageLayout::GgmlK256V1) {
        validate_ggml_k_payload(shape, payload);
        const auto total_rows = validate_row_ranges(rows, shape[0], 1);
        const auto source_base = ggml_k_code_offset(shape[0]);
        const auto dest_base = ggml_k_code_offset(total_rows);
        out.prefix.resize(dest_base);
        std::uint64_t dest_row = 0;
        std::uint64_t cursor = 0;
        for (const auto& range : rows) {
            for (auto row = range.begin; row < range.begin + range.count; ++row) {
                const auto descriptor = ggml_row(payload, row);
                const auto bytes = (shape[1] / 256) * ((descriptor & 1) ? 210 : 144);
                set_ggml_row(out.prefix, dest_row++, (cursor << 1) | (descriptor & 1));
                out.copies.push_back({source_base + (descriptor >> 1), dest_base + cursor, bytes});
                cursor += bytes;
            }
        }
        out.encoded_bytes = dest_base + cursor;
        return out;
    }

    if (layout == StorageLayout::ContiguousLeV1) {
        // Row-major with no internal structure: one contiguous range per row range.
        const std::uint64_t total_rows = validate_row_ranges(rows, shape[0], 1);
        const std::uint64_t row_bytes  = checked_mul(
            row_element_stride(shape), direct_word_bytes(format), "row slice row bytes");
        out.encoded_bytes = checked_mul(total_rows, row_bytes, "row slice encoded size");
        const std::array<SlicePlane, 1> planes = {SlicePlane{0, 0, row_bytes}};
        append_row_plane_copies(out, planes, rows, 1);
        return out;
    }

    if (layout == StorageLayout::RowSplitK128V1) {
        // Three row-addressable planes (storage-layouts.md 3.7). Any row boundary is legal; the
        // shard keeps the parent's K, hence its groups_per_row, so only the plane offsets move.
        require_slice(shape.size() == 2, "row-split-k128-v1 requires a rank-two shape");
        const std::uint64_t total_rows           = validate_row_ranges(rows, shape[0], 1);
        const RowSplitGeometry parent            = row_split_geometry(format, shape);
        const std::array<std::uint64_t, 2> shard_shape = {total_rows, shape[1]};
        const RowSplitGeometry shard             = row_split_geometry(format, shard_shape);
        out.encoded_bytes                        = shard.encoded_bytes;
        const std::array<SlicePlane, 3> planes   = {
            SlicePlane{0, 0, parent.groups_per_row * parent.low_bytes_per_group},
            SlicePlane{parent.high_plane_offset, shard.high_plane_offset,
                         parent.groups_per_row * parent.high_bytes_per_group},
            SlicePlane{parent.scale_plane_offset, shard.scale_plane_offset,
                         parent.groups_per_row * 2},
        };
        append_row_plane_copies(out, planes, rows, 1);
        return out;
    }

    if (layout == StorageLayout::BlockScaleK16M128x4V1) {
        // The code plane is row-major [N, K/2]. The swizzled scale plane's outermost axis is the
        // 128-row tile (offset (row_tile * K_tiles + scale_tile) * 512), so a 128-row-aligned row
        // range is one contiguous scale-plane range too. The FP32 weight divisor is matrix-level
        // and is therefore copied to every shard.
        require_slice(shape.size() == 2, "blockscale-k16-m128x4-v1 requires a rank-two shape");
        const std::uint64_t total_rows = validate_row_ranges(rows, shape[0], kNvfp4RowTile);
        const BlockScaleGeometry parent = block_scale_geometry(format, shape);
        const std::array<std::uint64_t, 2> shard_shape = {total_rows, shape[1]};
        const BlockScaleGeometry shard  = block_scale_geometry(format, shard_shape);
        out.encoded_bytes               = shard.encoded_bytes;
        const std::array<SlicePlane, 1> code = {SlicePlane{0, 0, shape[1] / 2}};
        append_row_plane_copies(out, code, rows, 1);
        const std::array<SlicePlane, 1> scales = {
            SlicePlane{parent.scale_plane_offset, shard.scale_plane_offset,
                       checked_mul(parent.k_tiles, kNvfp4TileBytes, "NVFP4 row tile bytes")}};
        append_row_plane_copies(out, scales, rows, kNvfp4RowTile);
        out.copies.push_back(PlaneCopy{parent.weight_divisor_offset, shard.weight_divisor_offset,
                                       kNvfp4DivisorBytes});
        return out;
    }

    if (layout == StorageLayout::RowScaleV1) {
        // Row-major code plane plus one BF16 scale word per row; both are row-addressable.
        require_slice(shape.size() == 2, "row-scale-v1 requires a rank-two shape");
        const std::uint64_t total_rows           = validate_row_ranges(rows, shape[0], 1);
        const RowScaleGeometry parent            = row_scale_geometry(format, shape);
        const std::array<std::uint64_t, 2> shard_shape = {total_rows, shape[1]};
        const RowScaleGeometry shard             = row_scale_geometry(format, shard_shape);
        out.encoded_bytes                        = shard.encoded_bytes;
        const std::array<SlicePlane, 2> planes   = {
            SlicePlane{0, 0, shape[1]},
            SlicePlane{parent.scale_plane_offset, shard.scale_plane_offset, 2},
        };
        append_row_plane_copies(out, planes, rows, 1);
        return out;
    }
    throw ArtifactError("unknown tensor layout");
}

TensorSlice tensor_column_slice(StorageLayout layout, NumericFormat format,
                                std::span<const std::uint64_t> shape,
                                std::span<const SliceRange> column_ranges,
                                std::span<const std::byte> payload) {
    require_slice(shape.size() == 2, "a column slice requires a rank-two shape");
    const std::uint64_t rows        = shape[0];
    const std::uint64_t total_count = validate_column_ranges(column_ranges, shape[1]);
    TensorSlice out;

    if (layout == StorageLayout::GgmlK256V1) {
        validate_ggml_k_payload(shape, payload);
        for (const auto& columns : column_ranges) {
            require_slice(columns.begin % 256 == 0 && columns.count % 256 == 0,
                          "ggml-k256-v1 column boundaries must be multiples of 256");
        }
        const auto code_base = ggml_k_code_offset(rows);
        out.prefix.resize(code_base);
        std::uint64_t cursor = 0;
        for (std::uint64_t row = 0; row < rows; ++row) {
            const auto descriptor = ggml_row(payload, row);
            const auto block_bytes = (descriptor & 1) ? 210 : 144;
            set_ggml_row(out.prefix, row, (cursor << 1) | (descriptor & 1));
            for (const auto& columns : column_ranges) {
                const auto bytes = (columns.count / 256) * block_bytes;
                out.copies.push_back({code_base + (descriptor >> 1) + (columns.begin / 256) * block_bytes,
                                       code_base + cursor, bytes});
                cursor += bytes;
            }
        }
        out.encoded_bytes = code_base + cursor;
        return out;
    }

    if (layout == StorageLayout::ContiguousLeV1) {
        // Row-major with no internal structure: every range of every row is one contiguous copy,
        // so any number of ranges concatenates cleanly. This is the only layout that admits more
        // than one range; see the guard below.
        const std::uint64_t word                       = direct_word_bytes(format);
        const std::array<std::uint64_t, 2> shard_shape = {rows, total_count};
        out.encoded_bytes = tensor_encoded_size(layout, format, shard_shape);
        const std::array<SlicePlane, 1> planes = {SlicePlane{0, 0, word}};
        std::uint64_t destination              = 0;
        for (const SliceRange& range : column_ranges) {
            append_column_plane_copies(out, planes, rows, range.begin, shape[1], total_count,
                                       range.count, destination);
            destination += range.count;
        }
        return out;
    }

    // Every remaining layout groups, tiles, or swizzles along the column axis, so a shard made of
    // several disjoint column ranges would not reconstruct into that layout's own geometry. Those
    // families are all row-parallel GEMM weights, whose shard genuinely is one contiguous input
    // range, so the restriction costs nothing and is rejected loudly rather than mis-encoded.
    require_slice(column_ranges.size() == 1,
                  "this storage layout requires one contiguous column range");
    const SliceRange columns = column_ranges.front();

    if (layout == StorageLayout::RowSplitK128V1) {
        // Columns are grouped; a column range must therefore start and end on a group boundary.
        // Requiring the 128-column K-alignment unit (a multiple of every registered group size)
        // additionally makes the shard's own K_pad equal to its K, so the shard needs none of the
        // parent's trailing padding groups and its groups_per_row is exactly the selected count.
        const RowSplitGeometry parent = row_split_geometry(format, shape);
        require_slice(columns.begin % kKAlignment == 0 && columns.count % kKAlignment == 0,
                      "row-split-k128-v1 column ranges must be multiples of 128");
        const std::array<std::uint64_t, 2> shard_shape = {rows, columns.count};
        const RowSplitGeometry shard = row_split_geometry(format, shard_shape);
        out.encoded_bytes            = shard.encoded_bytes;
        const std::uint64_t skip     = columns.begin / parent.group_size;
        const std::array<SlicePlane, 3> planes = {
            SlicePlane{0, 0, parent.low_bytes_per_group},
            SlicePlane{parent.high_plane_offset, shard.high_plane_offset,
                       parent.high_bytes_per_group},
            SlicePlane{parent.scale_plane_offset, shard.scale_plane_offset, 2},
        };
        append_column_plane_copies(out, planes, rows, skip, parent.groups_per_row,
                                   shard.groups_per_row, shard.groups_per_row);
        return out;
    }

    if (layout == StorageLayout::BlockScaleK16M128x4V1) {
        // A column range of whole 64-column scale tiles is contiguous *within* one 128-row tile
        // but strided across row tiles, and the code plane is strided per row. Both are copied as
        // regular strides; the matrix-level divisor is replicated.
        const BlockScaleGeometry parent = block_scale_geometry(format, shape);
        require_slice(columns.begin % kNvfp4TileColumns == 0 &&
                          columns.count % kNvfp4TileColumns == 0,
                      "blockscale-k16-m128x4-v1 column ranges must be multiples of 64");
        const std::array<std::uint64_t, 2> shard_shape = {rows, columns.count};
        const BlockScaleGeometry shard = block_scale_geometry(format, shard_shape);
        out.encoded_bytes              = shard.encoded_bytes;
        const std::array<SlicePlane, 1> code = {SlicePlane{0, 0, 1}};
        append_column_plane_copies(out, code, rows, columns.begin / 2, shape[1] / 2,
                                   columns.count / 2, columns.count / 2);
        const std::array<SlicePlane, 1> scales = {SlicePlane{
            parent.scale_plane_offset, shard.scale_plane_offset, kNvfp4TileBytes}};
        append_column_plane_copies(out, scales, rows / kNvfp4RowTile,
                                   columns.begin / kNvfp4TileColumns, parent.k_tiles,
                                   shard.k_tiles, shard.k_tiles);
        out.copies.push_back(PlaneCopy{parent.weight_divisor_offset, shard.weight_divisor_offset,
                                       kNvfp4DivisorBytes});
        return out;
    }

    if (layout == StorageLayout::RowScaleV1) {
        // The BF16 multiplier is per output row, and a column slice keeps every row, so the whole
        // scale plane is replicated: each shard scales its partial product by the same factor,
        // which is exactly what summing the partials across devices requires.
        const RowScaleGeometry parent                  = row_scale_geometry(format, shape);
        const std::array<std::uint64_t, 2> shard_shape = {rows, columns.count};
        const RowScaleGeometry shard = row_scale_geometry(format, shard_shape);
        out.encoded_bytes            = shard.encoded_bytes;
        const std::array<SlicePlane, 1> code = {SlicePlane{0, 0, 1}};
        append_column_plane_copies(out, code, rows, columns.begin, shape[1], columns.count,
                                   columns.count);
        out.copies.push_back(PlaneCopy{parent.scale_plane_offset, shard.scale_plane_offset,
                                       shard.scale_plane_bytes});
        return out;
    }
    throw ArtifactError("unknown tensor layout");
}

RowScaleGeometry row_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape) {
    if (format != NumericFormat::FP8_E4M3FN_ROW_BF16S) {
        throw ArtifactError("row-scale-v1 requires FP8_E4M3FN_ROW_BF16S");
    }
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("row-scale-v1 requires a positive rank-two shape");
    }

    RowScaleGeometry out;
    out.rows             = shape[0];
    out.columns          = shape[1];
    out.code_plane_bytes = checked_mul(out.rows, out.columns, "FP8 element count");
    out.scale_plane_offset =
        align_up(out.code_plane_bytes, kTensorAlignment, "FP8 scale plane offset");
    out.scale_plane_bytes = checked_mul(out.rows, 2, "FP8 scale plane bytes");
    out.encoded_bytes =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "FP8 tensor encoded size");
    return out;
}

} // namespace ninfer::artifact
