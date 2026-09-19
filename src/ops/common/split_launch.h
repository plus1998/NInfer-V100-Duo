#pragma once

// ninfer::ops::detail - shared mechanics for the tp2 split forms of the Op families.
//
// Every split-capable Op family has two extra entry points beside its single-device one:
//
//   <op>_column_parallel  each rank owns a contiguous block of the OUTPUT rows. Rank r reads the
//                         full (replicated) activation and its own weight-row shard and writes its
//                         own output block. No communication; the two blocks concatenate to the
//                         tp1 result.
//   <op>_row_parallel     each rank owns a contiguous block of the INPUT rows. Rank r reads its
//                         own activation block and its own weight-column shard and writes a
//                         FULL-width PARTIAL result; one allreduce_sum then leaves the complete
//                         result on both ranks.
//
// Neither form is a new kernel. A shard is a standalone tensor of the same layout with one axis
// narrowed (that is exactly what the tp2 loader materializes into each device's arena), so the
// existing single-device kernel run against a shard Weight already halves its grid along the
// split axis and already reads only the shard's bytes. What the split forms add is (a) the
// per-rank device/stream discipline below, (b) the cross-rank shape agreement checks a single
// device cannot make, and (c) for the row-parallel form, the collective. The only kernel-side
// change a family needs is that its shape registry admits the shard geometries -- see each
// family's *_config.h / *_dispatch.cpp.
//
// STREAMS. Rank r's work is issued on ec.dev[r]->stream, the same stream include/ninfer/ops/
// allreduce.h runs its collectives on, because "the stream a device executes on" is a property of
// its DeviceContext. The CALLER OBLIGATION documented there applies unchanged and transitively:
// activations staged with the plain cudaMemcpy/cudaMemset/<<<...>>> forms land on the device's
// LEGACY DEFAULT stream, which does NOT implicitly synchronize with DeviceContext::stream, and
// must be retired before a split form reads them.

#include "core/device.h" // DeviceContext, ExecutionContext, CUDA_CHECK

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>

namespace ninfer::ops::detail {

// Current-device save/restore. A split form issues work for each device in turn and must not
// leave the caller's current device changed.
class CurrentDeviceScope {
public:
    CurrentDeviceScope() { CUDA_CHECK(cudaGetDevice(&previous_)); }

    ~CurrentDeviceScope() {
        const cudaError_t status = cudaSetDevice(previous_);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during cudaSetDevice: %s: %s\n",
                         cudaGetErrorName(status), cudaGetErrorString(status));
        }
    }

    CurrentDeviceScope(const CurrentDeviceScope&)            = delete;
    CurrentDeviceScope& operator=(const CurrentDeviceScope&) = delete;

    static void set(int device) { CUDA_CHECK(cudaSetDevice(device)); }

private:
    int previous_ = 0;
};

inline void require_split_context(const ExecutionContext& ec, const char* message) {
    if (ec.tp != 2 || !ec.dev[0].has_value() || !ec.dev[1].has_value() ||
        ec.dev[0]->device == ec.dev[1]->device) {
        throw std::invalid_argument(message);
    }
}

#ifndef NDEBUG
// Debug-only residency predicate, the same one include/ninfer/ops/allreduce.h applies to its own
// buffers and for the same reason: passing rank 1 a pointer that lives on device 0 is the single
// most likely tp2 caller mistake, and it otherwise surfaces as a silently wrong result or an
// opaque launch failure much later. It costs a driver round trip per pointer, so it is compiled
// out of the Release build the product ships.
inline void require_resident_on(const void* pointer, int device, const char* message) {
    if (pointer == nullptr) { return; }
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device) {
        throw std::invalid_argument(message);
    }
}
#endif

// Checks that one rank's activation, weight planes, and output all live on that rank's device.
// A no-op in Release.
inline void require_rank_residency([[maybe_unused]] const ExecutionContext& ec,
                                   [[maybe_unused]] int rank,
                                   [[maybe_unused]] const void* activation,
                                   [[maybe_unused]] const void* weight_payload,
                                   [[maybe_unused]] const void* output,
                                   [[maybe_unused]] const char* message) {
#ifndef NDEBUG
    const int device = ec.dev[rank]->device;
    require_resident_on(activation, device, message);
    require_resident_on(weight_payload, device, message);
    require_resident_on(output, device, message);
#endif
}

// Issues `body(rank)` for rank 0 then rank 1 with that rank's device current, restoring the
// caller's current device afterwards. Rank order is fixed and the calls are enqueue-only, so two
// ranks' kernels overlap on the device even though the host issues them in sequence.
template <class Body>
void for_each_rank(const ExecutionContext& ec, Body&& body) {
    const CurrentDeviceScope scope;
    for (int rank = 0; rank < 2; ++rank) {
        CurrentDeviceScope::set(ec.dev[rank]->device);
        body(rank);
    }
}

} // namespace ninfer::ops::detail
