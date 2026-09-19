#include "artifact/materializer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <span>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::artifact {
namespace {

constexpr std::size_t kSlotBytes        = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaximumSlotCount = 4;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) { throw ArtifactError(label); }
    return a + b;
}

std::uint64_t align_down(std::uint64_t value, std::uint64_t alignment) {
    return value / alignment * alignment;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, const char* label) {
    return checked_add(value, alignment - 1, label) / alignment * alignment;
}

// One staging buffer plus a completion event per destination device: a slot may only be refilled
// once every device has finished reading it.
class Slot {
public:
    Slot(std::size_t bytes, std::span<DeviceContext* const> devices) : buffer(bytes) {
        for (std::size_t i = 0; i < devices.size(); ++i) {
            CUDA_CHECK(cudaSetDevice(devices[i]->device));
            CUDA_CHECK(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
        }
    }

    ~Slot() {
        for (std::size_t i = 0; i < events.size(); ++i) {
            if (pending[i]) { (void)cudaEventSynchronize(events[i]); }
            if (events[i] != nullptr) { (void)cudaEventDestroy(events[i]); }
        }
    }

    void wait() {
        for (std::size_t i = 0; i < events.size(); ++i) {
            if (pending[i]) {
                CUDA_CHECK(cudaEventSynchronize(events[i]));
                pending[i] = false;
            }
        }
    }

    PinnedHostBuffer buffer;
    std::array<cudaEvent_t, kMaximumDevices> events{};
    std::array<bool, kMaximumDevices> pending{};
};

struct CopyRange {
    std::uint64_t source_begin = 0;
    std::uint64_t source_end   = 0;
    std::byte* destination     = nullptr;
    int device                 = 0;
};

struct ReadSpan {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

} // namespace

void* MaterializedArtifact::device_data(ObjectHandle handle) const { return device_data(handle, 0); }

void* MaterializedArtifact::device_data(ObjectHandle handle, int device) const {
    if (device < 0 || device >= stats_.device_count) {
        throw ArtifactError("materialized artifact does not cover that device");
    }
    if (handle.index >= objects_.size() ||
        objects_[handle.index].device[static_cast<std::size_t>(device)] == nullptr) {
        throw ArtifactError("object handle does not name a materialized tensor");
    }
    return objects_[handle.index].device[static_cast<std::size_t>(device)];
}

std::uint64_t MaterializedArtifact::device_bytes(ObjectHandle handle, int device) const {
    if (device < 0 || device >= stats_.device_count) {
        throw ArtifactError("materialized artifact does not cover that device");
    }
    if (handle.index >= objects_.size() ||
        objects_[handle.index].device[static_cast<std::size_t>(device)] == nullptr) {
        throw ArtifactError("object handle does not name a materialized tensor");
    }
    return objects_[handle.index].device_bytes[static_cast<std::size_t>(device)];
}

MaterializedArtifact::GgmlKRowProfile
MaterializedArtifact::ggml_k_row_profile(ObjectHandle handle, int device) const {
    if (device < 0 || device >= stats_.device_count || handle.index >= objects_.size()) {
        throw ArtifactError("materialized artifact does not cover that object and device");
    }
    return objects_[handle.index].ggml_k_profiles[static_cast<std::size_t>(device)];
}

std::span<const std::byte> MaterializedArtifact::resource_bytes(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    return objects_[handle.index].resource;
}

std::vector<std::byte> MaterializedArtifact::take_resource_bytes(ObjectHandle handle) {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    auto& resource = objects_[handle.index].resource;
    stats_.retained_resource_bytes -= resource.size();
    return std::move(resource);
}

DeviceArena& MaterializedArtifact::device_arena() { return device_arena(0); }

DeviceArena& MaterializedArtifact::device_arena(int device) {
    if (device < 0 || device >= stats_.device_count ||
        !device_arena_[static_cast<std::size_t>(device)]) {
        throw ArtifactError("artifact has no device tensor backing");
    }
    return *device_arena_[static_cast<std::size_t>(device)];
}

int MaterializedArtifact::physical_device(int device) const {
    if (device < 0 || device >= stats_.device_count) {
        throw ArtifactError("materialized artifact does not cover that device");
    }
    return physical_devices_[static_cast<std::size_t>(device)];
}

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 std::span<DeviceContext* const> devices, LoadProgress* progress) {
    const int device_count = plan.device_count;
    if (device_count < 1 || device_count > static_cast<int>(kMaximumDevices) ||
        devices.size() < static_cast<std::size_t>(device_count)) {
        throw ArtifactError("materialization plan and execution context disagree on device count");
    }
    MaterializedArtifact out;
    out.objects_.resize(plan.object_count);
    out.stats_.device_count = device_count;
    for (int index = 0; index < device_count; ++index) {
        const auto slot              = static_cast<std::size_t>(index);
        out.physical_devices_[slot]  = devices[slot]->device;
        const std::uint64_t capacity = plan.device_capacity_bytes[slot];
        if (capacity == 0 || capacity > static_cast<std::uint64_t>(SIZE_MAX)) {
            throw ArtifactError("artifact tensor backing size is invalid");
        }
        CUDA_CHECK(cudaSetDevice(devices[slot]->device));
        out.device_arena_[slot] = std::make_unique<DeviceArena>(static_cast<std::size_t>(capacity));
        if (device_count > 1) {
            // A shard's plane offsets are recomputed for its own row/column count, so the
            // alignment gaps between its planes are not covered by any copy. The layouts define
            // those gap bytes as zero; make them so rather than leaving allocator residue.
            //
            // This MUST be stream-ordered on the same stream the copies use. load_stream is
            // created with cudaStreamNonBlocking, so it does not synchronize with the legacy
            // default stream that synchronous-API cudaMemset would run on -- a multi-GiB fill
            // could still be in flight when the first weight copies land and would silently zero
            // them. cudaMemsetAsync on load_stream orders the fill strictly before every copy.
            CUDA_CHECK(cudaMemsetAsync(out.device_arena_[slot]->base(), 0,
                                       static_cast<std::size_t>(capacity),
                                       devices[slot]->load_stream));
        }
        out.stats_.per_device_capacity_bytes[slot] = capacity;
        out.stats_.device_capacity_bytes =
            checked_add(out.stats_.device_capacity_bytes, capacity, "device capacity overflows u64");
    }
    // One placement per (object, device); every device tensor has a device-0 placement, so
    // counting those keeps `tensor_count` the number of distinct tensors at any tp.
    out.stats_.tensor_count = static_cast<std::size_t>(
        std::count_if(plan.device_objects.begin(), plan.device_objects.end(),
                      [](const DeviceMaterialization& placement) { return placement.device == 0; }));
    out.stats_.resource_count = plan.host_objects.size();

    for (const HostMaterialization& placement : plan.host_objects) {
        auto& resource            = out.objects_.at(placement.object.index).resource;
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        resource.assign(payload.data.begin(), payload.data.end());
        out.stats_.retained_resource_bytes += resource.size();
        out.stats_.file_bytes =
            checked_add(out.stats_.file_bytes, resource.size(), "artifact read bytes overflow u64");
    }

    std::size_t range_count = 0;
    for (const DeviceMaterialization& placement : plan.device_objects) {
        range_count += placement.copies.empty() ? 1 : placement.copies.size();
    }
    std::vector<CopyRange> ranges;
    ranges.reserve(range_count);
    std::uint64_t copied         = 0;
    std::uint64_t last_published = 0;
    std::uint64_t total          = 0;
    for (const DeviceMaterialization& placement : plan.device_objects) {
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        const auto slot           = static_cast<std::size_t>(placement.device);
        if (placement.device < 0 || placement.device >= device_count) {
            throw ArtifactError("materialization placement names a device outside the plan");
        }
        DeviceArena& arena = *out.device_arena_[slot];
        DeviceSpan storage = arena.alloc_bytes(static_cast<std::size_t>(placement.bytes),
                                               static_cast<std::size_t>(placement.alignment));
        const auto actual_offset = static_cast<std::uint64_t>(
            static_cast<std::byte*>(storage.data) - static_cast<std::byte*>(arena.base()));
        if (actual_offset != placement.offset) {
            throw ArtifactError("materialization plan does not match artifact payload");
        }
        out.objects_.at(placement.object.index).device[slot]       = storage.data;
        out.objects_.at(placement.object.index).device_bytes[slot] = placement.bytes;
        const auto* tensor = std::get_if<TensorDescriptor>(
            &reader.objects().at(placement.object.index));
        if (tensor != nullptr && tensor->layout == StorageLayout::GgmlK256V1) {
            if (placement.logical_rows == 0 ||
                placement.logical_rows > static_cast<std::uint64_t>(INT32_MAX)) {
                throw ArtifactError("GGML_K materialization has invalid logical row count");
            }
            const std::span<const std::byte> descriptors = placement.prefix.empty()
                ? payload.data
                : std::span<const std::byte>(placement.prefix);
            const auto tag = [&](std::uint64_t row) {
                std::uint64_t descriptor = 0;
                for (int byte = 0; byte < 8; ++byte) {
                    descriptor |= static_cast<std::uint64_t>(
                        std::to_integer<unsigned char>(descriptors[row * 8 + byte])) << (byte * 8);
                }
                return static_cast<int>(descriptor & 1U);
            };
            const int first = tag(0);
            std::uint64_t change = placement.logical_rows;
            for (std::uint64_t row = 1; row < placement.logical_rows; ++row) {
                if (tag(row) != first) { change = row; break; }
            }
            bool compact = true;
            for (std::uint64_t row = change; row < placement.logical_rows; ++row) {
                if (tag(row) == first) { compact = false; break; }
            }
            if (compact) {
                auto& profile = out.objects_[placement.object.index].ggml_k_profiles[slot];
                profile.first_q6 = first;
                profile.type_change = static_cast<std::int32_t>(change);
            }
        }
        auto* const base                                     = static_cast<std::byte*>(storage.data);
        if (!placement.prefix.empty()) {
            if (placement.prefix.size() > placement.bytes) {
                throw ArtifactError("materialization prefix exceeds its allocation");
            }
            CUDA_CHECK(cudaSetDevice(devices[slot]->device));
            CUDA_CHECK(cudaMemcpyAsync(base, placement.prefix.data(), placement.prefix.size(),
                                        cudaMemcpyHostToDevice, devices[slot]->load_stream));
        }
        const auto add_range = [&](std::uint64_t source_offset, std::uint64_t dest_offset,
                                   std::uint64_t bytes) {
            if (source_offset > payload.data.size() ||
                payload.data.size() - source_offset < bytes || dest_offset > placement.bytes ||
                placement.bytes - dest_offset < bytes) {
                throw ArtifactError("materialization plan does not match artifact payload");
            }
            ranges.push_back(CopyRange{
                .source_begin = checked_add(payload.absolute_offset, source_offset,
                                            "artifact tensor source range overflows u64"),
                .source_end   = checked_add(payload.absolute_offset + source_offset, bytes,
                                            "artifact tensor source range overflows u64"),
                .destination  = base + dest_offset,
                .device       = placement.device,
            });
            total = checked_add(total, bytes, "artifact tensor byte count overflows u64");
        };
        if (placement.copies.empty()) {
            if (payload.data.size() != placement.bytes) {
                throw ArtifactError("materialization plan does not match artifact payload");
            }
            add_range(0, 0, placement.bytes);
        } else {
            for (const PlaneCopy& copy : placement.copies) {
                add_range(copy.source_offset, copy.dest_offset, copy.bytes);
            }
        }
    }
    if (ranges.empty()) { throw ArtifactError("materialization plan has no device tensors"); }
    // Destination ranges must stay disjoint *within* a device. The single-device path used to get
    // this from the source-range disjointness check below, which sharding had to drop (a
    // replicated object legitimately reads the same source twice). Check the destinations
    // directly instead: two copies landing on the same arena byte would make the result depend on
    // staging-chunk order.
    {
        std::vector<std::pair<const std::byte*, std::uint64_t>> written;
        written.reserve(ranges.size());
        for (int index = 0; index < device_count; ++index) {
            written.clear();
            for (const CopyRange& range : ranges) {
                if (range.device == index) {
                    written.emplace_back(range.destination, range.source_end - range.source_begin);
                }
            }
            std::sort(written.begin(), written.end());
            for (std::size_t i = 1; i < written.size(); ++i) {
                if (written[i].first < written[i - 1].first + written[i - 1].second) {
                    throw ArtifactError("materialization destination ranges overlap on a device");
                }
            }
        }
    }
    // Source ranges may overlap: a replicated object feeds the same source bytes to both devices, and
    // a shard reads interleaved sub-ranges. The copy loop below intersects every still-live range
    // with each staging chunk independently, so overlap (and even containment) is safe.
    std::sort(ranges.begin(), ranges.end(), [](const CopyRange& a, const CopyRange& b) {
        return a.source_begin < b.source_begin;
    });

    constexpr std::uint64_t alignment = Reader::direct_io_alignment;
    std::vector<ReadSpan> read_spans;
    read_spans.reserve(ranges.size());
    std::uint64_t aligned_read_bytes = 0;
    for (const CopyRange& range : ranges) {
        const std::uint64_t begin = align_down(range.source_begin, alignment);
        if (read_spans.empty() || begin > align_up(read_spans.back().end, alignment,
                                                   "artifact direct I/O span overflows u64")) {
            read_spans.push_back(ReadSpan{begin, range.source_end});
        } else {
            read_spans.back().end = std::max(read_spans.back().end, range.source_end);
        }
    }
    for (const ReadSpan& span : read_spans) {
        aligned_read_bytes = checked_add(
            aligned_read_bytes,
            align_up(span.end - span.begin, alignment, "artifact direct I/O span overflows u64"),
            "artifact direct I/O byte count overflows u64");
    }
    const std::size_t slot_bytes =
        static_cast<std::size_t>(std::min<std::uint64_t>(kSlotBytes, aligned_read_bytes));
    const std::size_t slot_count = static_cast<std::size_t>(
        std::min<std::uint64_t>(kMaximumSlotCount, 1 + (aligned_read_bytes - 1) / slot_bytes));
    std::vector<std::unique_ptr<Slot>> slots;
    slots.reserve(slot_count);
    for (std::size_t i = 0; i < slot_count; ++i) {
        slots.push_back(std::make_unique<Slot>(slot_bytes, devices.first(
                                                               static_cast<std::size_t>(device_count))));
    }
    out.stats_.peak_staging_bytes = static_cast<std::uint64_t>(slot_bytes) * slot_count;

    std::size_t next_slot        = 0;
    std::size_t first_unfinished = 0;
    const auto start             = std::chrono::steady_clock::now();
    if (progress != nullptr && progress->callback) { progress->callback("weights", 0, total); }
    for (const ReadSpan& span : read_spans) {
        for (std::uint64_t source = span.begin; source < span.end; source += slot_bytes) {
            Slot& slot = *slots[next_slot++ % slot_count];
            slot.wait();

            const std::uint64_t remaining = span.end - source;
            const std::size_t request     = static_cast<std::size_t>(std::min<std::uint64_t>(
                slot_bytes,
                align_up(remaining, alignment, "artifact direct I/O request overflows u64")));
            auto destination =
                std::span<std::byte>(static_cast<std::byte*>(slot.buffer.data()), request);
            const std::size_t bytes_read = reader.read_direct(source, destination);
            const std::uint64_t required = std::min<std::uint64_t>(request, remaining);
            if (bytes_read < required) {
                throw ArtifactError("direct artifact read ended before the planned tensor range");
            }
            out.stats_.file_bytes =
                checked_add(out.stats_.file_bytes, bytes_read, "artifact read bytes overflow u64");
            const std::uint64_t chunk_end =
                checked_add(source, bytes_read, "artifact direct I/O result overflows u64");

            // Retire the leading run of ranges this chunk has already left behind. Ranges that
            // still straddle `source` stay in the scan window; they are cheap to re-test.
            while (first_unfinished < ranges.size() &&
                   ranges[first_unfinished].source_end <= source) {
                ++first_unfinished;
            }
            // One pass per destination device: a stream belongs to its device, so the copies are
            // grouped rather than interleaved, and only devices this chunk actually fed record an
            // event on it.
            for (int index = 0; index < device_count; ++index) {
                const auto device_slot = static_cast<std::size_t>(index);
                if (device_count > 1) { CUDA_CHECK(cudaSetDevice(devices[device_slot]->device)); }
                bool fed = false;
                for (std::size_t range_index = first_unfinished;
                     range_index < ranges.size() &&
                     ranges[range_index].source_begin < chunk_end;
                     ++range_index) {
                    const CopyRange& range = ranges[range_index];
                    if (range.device != index) { continue; }
                    const std::uint64_t copy_begin = std::max(source, range.source_begin);
                    const std::uint64_t copy_end   = std::min(chunk_end, range.source_end);
                    if (copy_begin >= copy_end) { continue; }
                    const auto amount = static_cast<std::size_t>(copy_end - copy_begin);
                    CUDA_CHECK(cudaMemcpyAsync(
                        range.destination +
                            static_cast<std::size_t>(copy_begin - range.source_begin),
                        static_cast<std::byte*>(slot.buffer.data()) +
                            static_cast<std::size_t>(copy_begin - source),
                        amount, cudaMemcpyHostToDevice, devices[device_slot]->load_stream));
                    copied =
                        checked_add(copied, amount, "artifact copied byte count overflows u64");
                    out.stats_.per_device_h2d_bytes[device_slot] += amount;
                    fed = true;
                }
                if (fed) {
                    CUDA_CHECK(cudaEventRecord(slot.events[device_slot],
                                               devices[device_slot]->load_stream));
                    slot.pending[device_slot] = true;
                }
            }

            if (progress != nullptr && progress->callback && copied != last_published &&
                copied < total) {
                last_published = copied;
                progress->callback("weights", copied, total);
            }
        }
    }
    for (const auto& slot : slots) { slot->wait(); }
    for (int index = 0; index < device_count; ++index) {
        CUDA_CHECK(cudaStreamSynchronize(devices[static_cast<std::size_t>(index)]->load_stream));
    }
    // Leave the caller on the primary device: the copy loop above walked the device list, and the
    // rest of the load path assumes the current device is still the one it set before calling.
    if (device_count > 1) { CUDA_CHECK(cudaSetDevice(devices[0]->device)); }
    if (copied != total) {
        throw ArtifactError("direct materialization did not cover every tensor byte");
    }
    out.stats_.h2d_bytes = copied;
    out.stats_.upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (progress != nullptr && progress->callback) { progress->callback("weights", copied, total); }
    return out;
}

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 ExecutionContext& execution, LoadProgress* progress) {
    std::array<DeviceContext*, kMaximumDevices> devices{};
    std::size_t count = 0;
    for (auto& slot : execution.dev) {
        if (slot.has_value()) { devices[count++] = &slot.value(); }
    }
    return materialize(reader, plan, std::span<DeviceContext* const>(devices.data(), count),
                       progress);
}

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device, LoadProgress* progress) {
    if (plan.device_count != 1) {
        throw ArtifactError("a multi-device materialization plan needs an ExecutionContext");
    }
    DeviceContext* one = &device;
    return materialize(reader, plan, std::span<DeviceContext* const>(&one, 1), progress);
}

} // namespace ninfer::artifact
