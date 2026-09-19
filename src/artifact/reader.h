#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::artifact {

class ArtifactError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class NumericFormat {
    BF16,
    FP32,
    I32,
    Q4G64_F16S,
    Q5G64_F16S,
    Q6G64_F16S,
    W8G32_F16S,
    NVFP4,
    FP8_E4M3FN_ROW_BF16S,
    GGML_K,
};

enum class StorageLayout {
    ContiguousLeV1,
    RowSplitK128V1,
    BlockScaleK16M128x4V1,
    RowScaleV1,
    GgmlK256V1,
};

enum class ResourceEncoding {
    RawBytesV1,
};

std::string_view format_name(NumericFormat format) noexcept;
std::string_view layout_name(StorageLayout layout) noexcept;
std::string_view encoding_name(ResourceEncoding encoding) noexcept;

std::uint64_t tensor_alignment(StorageLayout layout) noexcept;
std::uint64_t resource_alignment(ResourceEncoding encoding) noexcept;
std::uint64_t tensor_encoded_size(StorageLayout layout, NumericFormat format,
                                  std::span<const std::uint64_t> shape,
                                  std::uint64_t stored_bytes = 0);
std::uint64_t ggml_k_code_offset(std::uint64_t rows);
void validate_ggml_k_payload(std::span<const std::uint64_t> shape,
                             std::span<const std::byte> payload);

struct RowSplitGeometry {
    std::uint64_t rows                 = 0;
    std::uint64_t columns              = 0;
    std::uint64_t padded_columns       = 0;
    std::uint64_t group_size           = 0;
    std::uint64_t groups_per_row       = 0;
    std::uint64_t low_bytes_per_group  = 0;
    std::uint64_t high_bytes_per_group = 0;
    std::uint64_t low_plane_bytes      = 0;
    std::uint64_t high_plane_offset    = 0;
    std::uint64_t high_plane_bytes     = 0;
    std::uint64_t scale_plane_offset   = 0;
    std::uint64_t scale_plane_bytes    = 0;
    std::uint64_t encoded_bytes        = 0;
};

RowSplitGeometry row_split_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct BlockScaleGeometry {
    std::uint64_t rows                  = 0;
    std::uint64_t columns               = 0;
    std::uint64_t groups_per_row        = 0;
    std::uint64_t k_tiles               = 0;
    std::uint64_t code_plane_bytes      = 0;
    std::uint64_t scale_plane_offset    = 0;
    std::uint64_t scale_plane_bytes     = 0;
    std::uint64_t weight_divisor_offset = 0;
    std::uint64_t encoded_bytes         = 0;
};

BlockScaleGeometry block_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct RowScaleGeometry {
    std::uint64_t rows               = 0;
    std::uint64_t columns            = 0;
    std::uint64_t code_plane_bytes   = 0;
    std::uint64_t scale_plane_offset = 0;
    std::uint64_t scale_plane_bytes  = 0;
    std::uint64_t encoded_bytes      = 0;
};

RowScaleGeometry row_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

// --- Tensor slicing (TP2 sharded materialization) ---------------------------------------------
//
// A shard of a tensor is a *standalone tensor of the same layout and format* whose logical shape
// is the parent's with one axis narrowed. Producing one is a pure byte operation: no layout of
// ours interleaves logical values with anything that has to be recomputed, so a shard is always
// some list of contiguous parent byte ranges copied into the shard's own plane offsets. Those
// ranges are what `TensorSlice` carries; the geometry that decides them is documented in
// docs/maintainer/storage-layouts.md and re-derived per layout in storage_layouts.cpp.

// One contiguous byte range copied from a parent payload into a shard.
struct PlaneCopy {
    std::uint64_t source_offset = 0; // relative to the parent object's payload begin
    std::uint64_t dest_offset   = 0; // relative to the shard's own payload begin
    std::uint64_t bytes         = 0;
};

// A half-open [begin, begin + count) range along the sliced axis, in logical coordinates.
struct SliceRange {
    std::uint64_t begin = 0;
    std::uint64_t count = 0;
};

struct TensorSlice {
    // The shard's own encoded size, i.e. tensor_encoded_size() of the narrowed logical shape.
    std::uint64_t encoded_bytes = 0;
    std::vector<PlaneCopy> copies;
    std::vector<std::byte> prefix;
};

// Row (axis 0) slice. The shard is the concatenation of `rows` in the order given; the ranges
// must be nonempty, ascending, disjoint, and inside the tensor. Multiple ranges exist because a
// fused object (attention query|key|gate|value, MLP gate|up) contributes one row block per fused
// sub-matrix to each device.
TensorSlice tensor_row_slice(StorageLayout layout, NumericFormat format,
                             std::span<const std::uint64_t> shape,
                             std::span<const SliceRange> rows,
                             std::span<const std::byte> payload = {});

// Column (axis 1) slice, rank two only. The shard is the concatenation of `columns` within every
// row, in the order given; the ranges must be nonempty, ascending, disjoint, and inside the
// tensor. Only contiguous-le-v1 accepts more than one range -- every other layout groups, tiles,
// or swizzles along the column axis, so a multi-range shard would not reconstruct into its own
// geometry, and >1 range is rejected there. Multiple ranges exist because a depthwise object whose
// channel axis is the column axis (GDN `gdn/convolution`, 10240 = Q|K|V) contributes one channel
// block per fused section to each device, exactly as `tensor_row_slice` does for fused GEMMs.
TensorSlice tensor_column_slice(StorageLayout layout, NumericFormat format,
                                std::span<const std::uint64_t> shape,
                                std::span<const SliceRange> columns,
                                std::span<const std::byte> payload = {});

struct TensorDescriptor {
    std::string name;
    std::vector<std::uint64_t> shape;
    NumericFormat format;
    StorageLayout layout;
    std::uint64_t offset;
    std::uint64_t bytes;
};

struct ResourceDescriptor {
    std::string name;
    ResourceEncoding encoding;
    std::uint64_t offset;
    std::uint64_t bytes;
};

using ObjectDescriptor = std::variant<TensorDescriptor, ResourceDescriptor>;

std::string_view object_name(const ObjectDescriptor& object) noexcept;
std::uint64_t object_offset(const ObjectDescriptor& object) noexcept;
std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept;

struct PayloadSpan {
    std::uint64_t absolute_offset;
    std::span<const std::byte> data;
};

struct ArtifactIdentity {
    std::string model_id;
    std::string weights_id;

    bool operator==(const ArtifactIdentity&) const = default;
};

class Reader {
public:
    static constexpr std::size_t direct_io_alignment = 4096;

    explicit Reader(const std::filesystem::path& path);
    ~Reader();

    Reader(Reader&&) noexcept;
    Reader& operator=(Reader&&) noexcept;
    Reader(const Reader&)            = delete;
    Reader& operator=(const Reader&) = delete;

    const ArtifactIdentity& identity() const noexcept;
    const std::vector<ObjectDescriptor>& objects() const noexcept;
    const ObjectDescriptor* find(std::string_view name) const noexcept;

    std::uint64_t file_bytes() const noexcept;
    std::uint64_t payload_offset() const noexcept;
    PayloadSpan payload(const ObjectDescriptor& object) const;
    PayloadSpan payload(std::string_view name) const;
    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::artifact
