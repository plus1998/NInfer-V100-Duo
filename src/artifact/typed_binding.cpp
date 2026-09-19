#include "artifact/typed_binding.h"

#include "artifact/materializer.h"

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

namespace ninfer::artifact {
namespace {

StorageLayout storage_layout_for(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
    case NumericFormat::FP32:
    case NumericFormat::I32:
        return StorageLayout::ContiguousLeV1;
    case NumericFormat::Q4G64_F16S:
    case NumericFormat::Q5G64_F16S:
    case NumericFormat::Q6G64_F16S:
    case NumericFormat::W8G32_F16S:
        return StorageLayout::RowSplitK128V1;
    case NumericFormat::NVFP4:
        return StorageLayout::BlockScaleK16M128x4V1;
    case NumericFormat::FP8_E4M3FN_ROW_BF16S:
        return StorageLayout::RowScaleV1;
    case NumericFormat::GGML_K:
        return StorageLayout::GgmlK256V1;
    }
    throw std::logic_error("unhandled numeric format");
}

QType qtype_for(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
        return QType::BF16_CTRL;
    case NumericFormat::FP32:
        return QType::FP32_CTRL;
    case NumericFormat::I32:
        return QType::I32_CTRL;
    case NumericFormat::Q4G64_F16S:
        return QType::Q4G64_F16S;
    case NumericFormat::Q5G64_F16S:
        return QType::Q5G64_F16S;
    case NumericFormat::Q6G64_F16S:
        return QType::Q6G64_F16S;
    case NumericFormat::W8G32_F16S:
        return QType::W8G32_F16S;
    case NumericFormat::NVFP4:
        return QType::NVFP4;
    case NumericFormat::FP8_E4M3FN_ROW_BF16S:
        return QType::FP8_E4M3FN_ROW_BF16S;
    case NumericFormat::GGML_K:
        return QType::GGML_K;
    }
    throw std::logic_error("unhandled numeric format");
}

DType dtype_for(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
        return DType::BF16;
    case NumericFormat::FP32:
        return DType::FP32;
    case NumericFormat::I32:
        return DType::I32;
    default:
        throw std::logic_error("quantized format has no direct dtype");
    }
}

Weight contiguous_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                         NumericFormat format, std::int32_t rows, std::int32_t columns,
                         int device) {
    Weight out{};
    require_placement_bytes(materialized, handle, device,
                            static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(columns) *
                                dtype_size(dtype_for(format)));
    out.payload       = materialized.device_data(handle, device);
    out.qdata         = out.payload;
    out.payload_bytes = static_cast<std::uint64_t>(rows) * columns * dtype_size(dtype_for(format));
    out.qtype         = qtype_for(format);
    out.layout        = QuantLayout::Contiguous;
    out.n             = rows;
    out.k             = columns;
    out.ndim          = 2;
    out.shape[0]      = rows;
    out.shape[1]      = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    return out;
}

Weight row_split_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                        NumericFormat format, std::int32_t rows, std::int32_t columns,
                        int device) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const RowSplitGeometry geometry          = row_split_geometry(format, shape);
    require_placement_bytes(materialized, handle, device, geometry.encoded_bytes);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(handle, device));

    Weight out{};
    out.payload          = bytes;
    out.payload_bytes    = geometry.encoded_bytes;
    out.high_plane_bytes = geometry.high_plane_bytes;
    out.qtype            = qtype_for(format);
    out.layout           = QuantLayout::RowSplit;
    out.group_size       = static_cast<std::uint32_t>(geometry.group_size);
    out.qdata            = bytes;
    out.qhigh       = geometry.high_plane_bytes == 0 ? nullptr : bytes + geometry.high_plane_offset;
    out.scales      = bytes + geometry.scale_plane_offset;
    out.n           = rows;
    out.k           = columns;
    out.group       = static_cast<std::int32_t>(geometry.group_size);
    out.scale_dtype = DType::FP16;
    out.ndim        = 2;
    out.shape[0]    = rows;
    out.shape[1]    = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = static_cast<std::int32_t>(geometry.padded_columns);
    return out;
}

Weight row_scale_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                        NumericFormat format, std::int32_t rows, std::int32_t columns,
                        int device) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const RowScaleGeometry geometry          = row_scale_geometry(format, shape);
    require_placement_bytes(materialized, handle, device, geometry.encoded_bytes);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(handle, device));

    Weight out{};
    out.payload         = bytes;
    out.payload_bytes   = geometry.encoded_bytes;
    out.qtype           = qtype_for(format);
    out.layout          = QuantLayout::RowScale;
    out.group_size      = static_cast<std::uint32_t>(geometry.columns);
    out.qdata           = bytes;
    out.scales          = bytes + geometry.scale_plane_offset;
    out.n               = rows;
    out.k               = columns;
    out.group           = columns;
    out.scale_dtype     = DType::BF16;
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    out.scale_ne[0]     = rows;
    out.scale_nb[0]     = 2;
    out.scale_nb[1]     = static_cast<std::int64_t>(rows) * 2;
    out.scale_nb[2]     = out.scale_nb[1];
    out.scale_nb[3]     = out.scale_nb[1];
    return out;
}

} // namespace

ObjectHandle bind_tensor(Binder& binder, std::string_view name, NumericFormat format,
                         std::initializer_list<std::uint64_t> shape, TensorPlacement placement) {
    const ObjectHandle handle =
        binder.require_tensor(name, format, storage_layout_for(format),
                              std::span<const std::uint64_t>(shape.begin(), shape.size()));
    if (placement == TensorPlacement::Device) {
        binder.materialize_on_device(handle);
    } else {
        binder.validate_only(handle);
    }
    return handle;
}

ObjectHandle bind_device_tensor(Binder& binder, std::string_view name, NumericFormat format,
                                std::initializer_list<std::uint64_t> shape) {
    return bind_tensor(binder, name, format, shape, TensorPlacement::Device);
}

ObjectHandle bind_raw_resource(Binder& binder, std::string_view name) {
    const ObjectHandle handle = binder.require_resource(name, ResourceEncoding::RawBytesV1);
    binder.retain_on_host(handle);
    return handle;
}

void require_placement_bytes(const MaterializedArtifact& materialized, ObjectHandle handle,
                             int device, std::uint64_t required_bytes) {
    const std::uint64_t placed = materialized.device_bytes(handle, device);
    if (placed < required_bytes) {
        throw ArtifactError("materialized object placement is " + std::to_string(placed) +
                            " bytes on device " + std::to_string(device) +
                            ", but the requested shape needs " + std::to_string(required_bytes) +
                            " bytes");
    }
}

Tensor materialized_tensor(const MaterializedArtifact& materialized, ObjectHandle handle,
                           NumericFormat format, std::initializer_list<std::int32_t> internal_shape,
                           int device) {
    std::uint64_t elements = 1;
    for (const std::int32_t extent : internal_shape) {
        if (extent <= 0) {
            throw std::invalid_argument("materialized_tensor shape extents must be positive");
        }
        elements *= static_cast<std::uint64_t>(extent);
    }
    require_placement_bytes(materialized, handle, device, elements * dtype_size(dtype_for(format)));
    return Tensor(materialized.device_data(handle, device), dtype_for(format), internal_shape);
}

Weight materialized_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                           NumericFormat format, std::int32_t rows, std::int32_t columns,
                           int device) {
    if (format == NumericFormat::GGML_K) {
        Weight out{};
        out.payload = materialized.device_data(handle, device);
        out.payload_bytes = materialized.device_bytes(handle, device);
        const std::array<std::uint64_t, 2> shape = {
            static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(columns)};
        tensor_encoded_size(StorageLayout::GgmlK256V1, format, shape, out.payload_bytes);
        out.qtype = QType::GGML_K;
        out.layout = QuantLayout::GgmlK256;
        out.qhigh = out.payload;
        out.qdata = static_cast<const std::byte*>(out.payload) + ggml_k_code_offset(rows);
        out.group = out.group_size = 256;
        out.n = out.shape[0] = out.padded_shape[0] = rows;
        out.k = out.shape[1] = out.padded_shape[1] = columns;
        out.ndim = 2;
        const auto profile = materialized.ggml_k_row_profile(handle, device);
        out.ggml_k_first_q6 = profile.first_q6;
        out.ggml_k_type_change = profile.type_change;
        return out;
    }
    if (format == NumericFormat::NVFP4) {
        throw std::invalid_argument(
            "materialized_weight: NVFP4 requires target-validated weight and input divisors");
    }
    if (storage_layout_for(format) == StorageLayout::ContiguousLeV1) {
        return contiguous_weight(materialized, handle, format, rows, columns, device);
    }
    if (storage_layout_for(format) == StorageLayout::RowScaleV1) {
        return row_scale_weight(materialized, handle, format, rows, columns, device);
    }
    return row_split_weight(materialized, handle, format, rows, columns, device);
}

} // namespace ninfer::artifact
