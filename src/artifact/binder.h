#pragma once

#include "artifact/reader.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::artifact {

// One process drives at most two devices (core/device.h ExecutionContext).
inline constexpr std::size_t kMaximumDevices = 2;

enum class TensorPlacement : std::uint8_t {
    Device,
    ValidateOnly,
};

// Which logical axis of a tensor a shard map splits. `Rows` narrows axis 0 (the output/row
// dimension: column-parallel ops), `Columns` narrows axis 1 (the input dimension: row-parallel
// ops). `Replicated` means every device holds the whole object.
enum class ShardAxis : std::uint8_t {
    Replicated,
    Rows,
    Columns,
};

struct ObjectHandle {
    std::size_t index = 0;
};

// The per-device shard map for one object. `device_ranges[d]` lists the ranges of `axis` that
// device d owns, in the order they are concatenated into that device's shard. `Replicated` means
// every device holds the complete object and carries no ranges at all; under `Rows` or `Columns`
// every device in the plan must name at least one range -- an empty list there is rejected rather
// than quietly promoted to a full copy, since "this device happens to own nothing" is far more
// likely to be a shard-map bug than an intent.
struct ShardPlacement {
    ShardAxis axis = ShardAxis::Replicated;
    std::array<std::vector<SliceRange>, kMaximumDevices> device_ranges;
};

struct DeviceMaterialization {
    ObjectHandle object;
    int device              = 0;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 0;
    std::uint64_t logical_rows = 0;
    // Contiguous parent-payload ranges that fill this device's copy. Empty means the whole
    // payload lands at `offset` verbatim -- the only case that exists at tp1.
    std::vector<PlaneCopy> copies;
    std::vector<std::byte> prefix;
};

struct HostMaterialization {
    ObjectHandle object;
};

struct MaterializationPlan {
    std::size_t object_count = 0;
    // Devices this plan targets; device_capacity_bytes[d] is valid for d < device_count.
    int device_count = 1;
    std::array<std::uint64_t, kMaximumDevices> device_capacity_bytes{};
    // One entry per (object, device) pair that receives bytes; ascending by device within object.
    std::vector<DeviceMaterialization> device_objects;
    std::vector<HostMaterialization> host_objects;
};

class Binder {
public:
    // Resolves an object's shard map from its artifact name. Installed by a target that binds for
    // tp > 1; when absent (the tp1 path) every device tensor is placed whole on device 0.
    using ShardResolver = std::function<ShardPlacement(std::string_view)>;

    explicit Binder(const Reader& reader, int device_count = 1);

    // How many device arenas this binder plans for. A target that binds a shard map must check
    // this against its own tp: a tp2 shard map fed to a one-device binder would place only device
    // 0's half-shard and size the arena for half a model, with nothing else noticing.
    [[nodiscard]] int device_count() const noexcept { return materialization_.device_count; }

    void set_shard_resolver(ShardResolver resolver);

    ObjectHandle require_tensor(std::string_view name, NumericFormat format, StorageLayout layout,
                                std::span<const std::uint64_t> shape);
    ObjectHandle require_resource(std::string_view name, ResourceEncoding encoding);

    const ObjectDescriptor& descriptor(ObjectHandle handle) const;
    PayloadSpan payload(ObjectHandle handle) const;
    void materialize_on_device(ObjectHandle handle);
    void retain_on_host(ObjectHandle handle);
    void validate_only(ObjectHandle handle);
    void validate_unconsumed_matching(std::string_view prefix = "");
    MaterializationPlan finish();

private:
    ObjectHandle find_unconsumed(std::string_view name);
    void place(ObjectHandle handle, int device, std::uint64_t bytes, std::uint64_t alignment,
               std::uint64_t logical_rows, std::vector<PlaneCopy> copies);

    const Reader& reader_;
    std::vector<bool> consumed_;
    std::vector<bool> planned_;
    ShardResolver shard_resolver_;
    MaterializationPlan materialization_;
};

} // namespace ninfer::artifact
