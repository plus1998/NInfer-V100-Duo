#pragma once

#include "artifact/binder.h"
#include "core/tensor.h"

#include <cstdint>
#include <initializer_list>
#include <string_view>

namespace ninfer::artifact {

class MaterializedArtifact;

[[nodiscard]] ObjectHandle bind_tensor(Binder& binder, std::string_view name, NumericFormat format,
                                       std::initializer_list<std::uint64_t> shape,
                                       TensorPlacement placement);

[[nodiscard]] ObjectHandle bind_device_tensor(Binder& binder, std::string_view name,
                                              NumericFormat format,
                                              std::initializer_list<std::uint64_t> shape);

[[nodiscard]] ObjectHandle bind_raw_resource(Binder& binder, std::string_view name);

// Wraps device `device`'s copy of `handle` in a Tensor with the caller's internal shape.
//
// The shape is CHECKED against the bytes actually placed for that (object, device) pair. Before
// tensor parallelism landed this was a pure reinterpretation with no validation whatsoever, which
// was harmless only because every placement held the whole object; at tp > 1 a placement holds a
// SHARD, so a stale whole-model literal shape (`{10240, 4}` for a 5120-channel conv shard, say)
// would silently read twice the bytes that exist. The check makes that class of mistake throw.
[[nodiscard]] Tensor materialized_tensor(const MaterializedArtifact& materialized,
                                         ObjectHandle handle, NumericFormat format,
                                         std::initializer_list<std::int32_t> internal_shape,
                                         int device = 0);

[[nodiscard]] Weight materialized_weight(const MaterializedArtifact& materialized,
                                         ObjectHandle handle, NumericFormat format,
                                         std::int32_t rows, std::int32_t columns, int device = 0);

// Shared shape-vs-placement guard, exposed so target-owned wrappers (NVFP4 weights, which the
// generic `materialized_weight` refuses) can discharge the same obligation.
void require_placement_bytes(const MaterializedArtifact& materialized, ObjectHandle handle,
                             int device, std::uint64_t required_bytes);

} // namespace ninfer::artifact
