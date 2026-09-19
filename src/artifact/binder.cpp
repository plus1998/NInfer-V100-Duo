#include "artifact/binder.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace ninfer::artifact {
namespace {

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    const std::uint64_t mask = alignment - 1;
    if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
        throw ArtifactError("materialization plan size overflows u64");
    }
    return (value + mask) & ~mask;
}

} // namespace

Binder::Binder(const Reader& reader, int device_count)
    : reader_(reader), consumed_(reader.objects().size(), false),
      planned_(reader.objects().size(), false) {
    if (device_count < 1 || device_count > static_cast<int>(kMaximumDevices)) {
        throw ArtifactError("materialization device count must be 1 or 2");
    }
    materialization_.object_count = reader.objects().size();
    materialization_.device_count = device_count;
}

void Binder::set_shard_resolver(ShardResolver resolver) {
    if (!materialization_.device_objects.empty()) {
        throw ArtifactError("the shard resolver must be installed before any tensor is placed");
    }
    shard_resolver_ = std::move(resolver);
}

ObjectHandle Binder::find_unconsumed(std::string_view name) {
    const auto& objects            = reader_.objects();
    const ObjectDescriptor* object = reader_.find(name);
    if (object == nullptr) {
        throw ArtifactError("required artifact object is missing: " + std::string(name));
    }
    const auto index = static_cast<std::size_t>(object - objects.data());
    if (consumed_[index]) {
        throw ArtifactError("artifact object was bound more than once: " + std::string(name));
    }
    consumed_[index] = true;
    return ObjectHandle{index};
}

ObjectHandle Binder::require_tensor(std::string_view name, NumericFormat format,
                                    StorageLayout layout, std::span<const std::uint64_t> shape) {
    const ObjectHandle handle = find_unconsumed(name);
    const auto* tensor        = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr) {
        throw ArtifactError("required tensor is a resource: " + std::string(name));
    }
    if (tensor->format != format || tensor->layout != layout ||
        !std::equal(tensor->shape.begin(), tensor->shape.end(), shape.begin(), shape.end())) {
        throw ArtifactError("tensor descriptor does not match target contract: " +
                            std::string(name));
    }
    return handle;
}

ObjectHandle Binder::require_resource(std::string_view name, ResourceEncoding encoding) {
    const ObjectHandle handle = find_unconsumed(name);
    const auto* resource      = std::get_if<ResourceDescriptor>(&descriptor(handle));
    if (resource == nullptr) {
        throw ArtifactError("required resource is a tensor: " + std::string(name));
    }
    if (resource->encoding != encoding) {
        throw ArtifactError("resource encoding does not match target contract: " +
                            std::string(name));
    }
    return handle;
}

const ObjectDescriptor& Binder::descriptor(ObjectHandle handle) const {
    if (handle.index >= reader_.objects().size()) {
        throw ArtifactError("artifact object handle is out of range");
    }
    return reader_.objects()[handle.index];
}

PayloadSpan Binder::payload(ObjectHandle handle) const {
    return reader_.payload(descriptor(handle));
}

void Binder::place(ObjectHandle handle, int device, std::uint64_t bytes, std::uint64_t alignment,
                   std::uint64_t logical_rows, std::vector<PlaneCopy> copies) {
    std::uint64_t& capacity =
        materialization_.device_capacity_bytes[static_cast<std::size_t>(device)];
    const std::uint64_t offset = align_up(capacity, alignment);
    if (bytes > std::numeric_limits<std::uint64_t>::max() - offset) {
        throw ArtifactError("materialization plan size overflows u64");
    }
    materialization_.device_objects.push_back(
        DeviceMaterialization{handle, device, offset, bytes, alignment, logical_rows,
                              std::move(copies)});
    capacity = offset + bytes;
}

void Binder::materialize_on_device(ObjectHandle handle) {
    const auto* tensor = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr) {
        throw ArtifactError("resource cannot be materialized as a device tensor");
    }
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(tensor->name));
    }
    const std::uint64_t alignment = tensor_alignment(tensor->layout);
    const ShardPlacement placement =
        shard_resolver_ ? shard_resolver_(tensor->name) : ShardPlacement{};

    for (int device = 0; device < materialization_.device_count; ++device) {
        const std::vector<SliceRange>& ranges =
            placement.device_ranges[static_cast<std::size_t>(device)];
        if (placement.axis == ShardAxis::Replicated) {
            if (!ranges.empty()) {
                throw ArtifactError("a replicated placement must carry no shard ranges: " +
                                    std::string(tensor->name));
            }
            // Whole object. `copies` stays empty: the materializer copies the payload verbatim,
            // which is byte-for-byte what the single-device path has always done.
            place(handle, device, tensor->bytes, alignment, tensor->shape.front(), {});
            continue;
        }
        if (ranges.empty()) {
            throw ArtifactError("sharded object names no range for device " +
                                std::to_string(device) + ": " + std::string(tensor->name));
        }
        TensorSlice slice;
        if (placement.axis == ShardAxis::Rows) {
            slice = tensor_row_slice(tensor->layout, tensor->format, tensor->shape, ranges, payload(handle).data);
        } else {
            // Multiple column ranges are legal only where the layout allows them
            // (contiguous-le-v1 and whole-block ggml-k256-v1);
            // tensor_column_slice enforces that per layout rather than this call site guessing.
            slice = tensor_column_slice(tensor->layout, tensor->format, tensor->shape, ranges, payload(handle).data);
        }
        std::uint64_t covered = 0;
        for (const PlaneCopy& copy : slice.copies) {
            if (copy.bytes == 0 || copy.source_offset > tensor->bytes ||
                tensor->bytes - copy.source_offset < copy.bytes ||
                copy.dest_offset > slice.encoded_bytes ||
                slice.encoded_bytes - copy.dest_offset < copy.bytes) {
                throw ArtifactError("shard slice range is outside its tensor: " +
                                    std::string(tensor->name));
            }
            covered += copy.bytes;
        }
        if (covered > slice.encoded_bytes) {
            throw ArtifactError("shard slice copies overlap: " + std::string(tensor->name));
        }
        // Two copies writing the same shard byte would make the result depend on staging-chunk
        // order. The single-device path got this for free (one copy per object); reinstate it
        // explicitly now that a shard is many copies. Ordering by destination is cheap here --
        // every layout emits its copies plane by plane, so this is nearly sorted already.
        {
            std::vector<const PlaneCopy*> ordered;
            ordered.reserve(slice.copies.size());
            for (const PlaneCopy& copy : slice.copies) { ordered.push_back(&copy); }
            std::sort(ordered.begin(), ordered.end(),
                      [](const PlaneCopy* a, const PlaneCopy* b) {
                          return a->dest_offset < b->dest_offset;
                      });
            for (std::size_t i = 1; i < ordered.size(); ++i) {
                if (ordered[i]->dest_offset <
                    ordered[i - 1]->dest_offset + ordered[i - 1]->bytes) {
                    throw ArtifactError("shard slice writes the same byte twice: " +
                                        std::string(tensor->name));
                }
            }
        }
        std::uint64_t logical_rows = tensor->shape.front();
        if (placement.axis == ShardAxis::Rows) {
            logical_rows = 0;
            for (const SliceRange& range : ranges) { logical_rows += range.count; }
        }
        place(handle, device, slice.encoded_bytes, alignment, logical_rows,
              std::move(slice.copies));
        materialization_.device_objects.back().prefix = std::move(slice.prefix);
    }
    planned_[handle.index] = true;
}

void Binder::retain_on_host(ObjectHandle handle) {
    const auto* resource = std::get_if<ResourceDescriptor>(&descriptor(handle));
    if (resource == nullptr) {
        throw ArtifactError("tensor cannot be retained as a host resource");
    }
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(resource->name));
    }
    materialization_.host_objects.push_back(HostMaterialization{handle});
    planned_[handle.index] = true;
}

void Binder::validate_only(ObjectHandle handle) {
    const ObjectDescriptor& object = descriptor(handle);
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(object_name(object)));
    }
    consumed_[handle.index] = true;
    planned_[handle.index] = true;
}

void Binder::validate_unconsumed_matching(std::string_view prefix) {
    for (std::size_t i = 0; i < reader_.objects().size(); ++i) {
        if (!consumed_[i]) {
            std::string_view name = object_name(reader_.objects()[i]);
            if (prefix.empty() || name.starts_with(prefix)) {
                consumed_[i] = true;
                planned_[i]  = true;
            }
        }
    }
}

MaterializationPlan Binder::finish() {
    const auto it = std::find(consumed_.begin(), consumed_.end(), false);
    if (it != consumed_.end()) {
        const auto index = static_cast<std::size_t>(it - consumed_.begin());
        throw ArtifactError("artifact object was not consumed by the selected target: " +
                            std::string(object_name(reader_.objects()[index])));
    }
    const auto unplanned = std::find(planned_.begin(), planned_.end(), false);
    if (unplanned != planned_.end()) {
        const auto index = static_cast<std::size_t>(unplanned - planned_.begin());
        throw ArtifactError("artifact object has no materialization placement: " +
                            std::string(object_name(reader_.objects()[index])));
    }
    return std::move(materialization_);
}

} // namespace ninfer::artifact
