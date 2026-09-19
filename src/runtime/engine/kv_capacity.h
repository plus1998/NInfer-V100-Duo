#pragma once

#include "runtime/contract/types.h"

#include <cstddef>
#include <span>

namespace ninfer::runtime {

[[nodiscard]] KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                                       const SequenceCapacityCurve& curve,
                                                       std::size_t available_runtime_bytes);

// Resolves one KV capacity plan shared by every device in a tensor-parallel group. KV addressing
// requires every device to carry the same number of Main KV pages (attention/KV addressing is
// already head-local; the page count itself is not), so all devices must agree on a single
// `main_page_groups` / `resolved_tokens` outcome rather than each picking its own.
//
// `curve` describes ONE device's own per-page byte cost. At tp2 that curve already reflects the
// device's own 2-of-4 KV heads (half the bytes of the tp1 4-head curve for the same page geometry
// and page count) -- constructing that per-device curve is target/model-layer work, out of scope
// here. Because the split is symmetric, every device shares the same curve, so this function
// takes one curve plus one budget per device: `available_runtime_bytes_per_device[i]` is device
// i's own free-memory budget (e.g. from its own `cudaMemGetInfo`), read independently per device
// in automatic mode. The plan is sized to the tightest ("bottleneck") device so no device is ever
// overcommitted; every device then reserves `runtime_reservation_bytes` from the single returned
// resolution (same byte count on every device, since the curve is symmetric).
//
// A single-element span reduces to `resolve_kv_capacity` exactly (tp1 stays byte-identical; tp1
// call sites are untouched and do not need to route through this function).
[[nodiscard]] KvCapacityResolution
resolve_kv_capacity_symmetric(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve,
                              std::span<const std::size_t> available_runtime_bytes_per_device);

} // namespace ninfer::runtime
