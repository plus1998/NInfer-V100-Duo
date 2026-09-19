#pragma once

// ninfer::ops::detail - per-device once-guard for cudaFuncSetAttribute memoization.
//
// cudaFuncAttributeMaxDynamicSharedMemorySize opt-ins are a PER-DEVICE kernel
// property. Launchers used to cache the result of cudaFuncSetAttribute in a bare
// function-local `static`, which applies the attribute only on whichever device is
// current the first time that template instantiation runs: correct for a single
// device, silently wrong for every other device in a process that later launches
// the same kernel there (cudaErrorInvalidValue at launch, since the opt-in never
// happened on that device). ensure_func_attr_per_device re-applies the attribute
// exactly once per (function, device) pair instead of once per function globally.
//
// IMPORTANT: the once-guard must be keyed on the kernel's runtime ADDRESS, not on
// the `Func` template parameter (its C++ function TYPE). Distinct kernel template
// instantiations frequently share an identical call signature -- e.g. two
// launchers with different Schedule/Geometry template args but the same
// parameter list -- which collapses them onto the same `Func` type. Keying the
// once-guard on `Func` (an earlier version of this helper did, via a
// `static std::array<std::once_flag, N>` local to the `Func`-templated function)
// silently merges those distinct kernels' guards: only the first kernel address
// ever passed for that shared type gets configured, and every other
// same-signature kernel launches with the untouched default 48 KiB smem limit,
// later failing with cudaErrorInvalidValue. Keying on the actual function pointer
// value (which is what cudaFuncSetAttribute itself operates on) makes distinct
// kernels distinct keys regardless of type collisions.

#include "core/device.h" // CUDA_CHECK

#include <mutex>
#include <set>
#include <utility>

namespace ninfer::ops::detail {

namespace kernel_attr_once_detail {

// Single process-wide registry of (kernel address, device) pairs that have
// already had the attribute applied. Deliberately NOT templated: it must be
// shared across every call site regardless of the kernel's C++ function type,
// which is exactly what a `Func`-templated static could not guarantee (see
// header comment above). `apply_once` is `inline` with external linkage, so
// its function-local statics are the same single instance across every
// translation unit that includes this header (ODR-guaranteed, standard
// "Meyer's singleton in a header" behavior) -- one registry, one mutex, for
// the whole process.
//
// This is a cold path: it only does real work the first time a given kernel
// is launched on a given device, so a single mutex held across the
// check-and-set is simpler than anything lock-free and costs nothing in
// steady state. Holding the mutex across the cudaFuncSetAttribute call itself
// (rather than just the registry check) also means a concurrent caller for
// the same (kernel, device) blocks until the first caller's attribute-set has
// actually completed, matching std::call_once's guarantee.
inline void apply_once(const void* func, int device, cudaFuncAttribute attr, int value) {
    static std::mutex mutex;
    static std::set<std::pair<const void*, int>> applied;

    std::lock_guard<std::mutex> lock(mutex);
    if (applied.insert({func, device}).second) {
        CUDA_CHECK(cudaFuncSetAttribute(func, attr, value));
    }
}

} // namespace kernel_attr_once_detail

// Applies `cudaFuncSetAttribute(func, attr, value)` exactly once per (kernel
// address, CUDA device) pair, keyed by cudaGetDevice() at call time. Safe to
// call on every launch: the underlying cudaFuncSetAttribute (and its
// CUDA_CHECK) only run the first time a given kernel reaches this call site on
// a given device.
template <typename Func>
void ensure_func_attr_per_device(Func* func, cudaFuncAttribute attr, int value) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    kernel_attr_once_detail::apply_once(reinterpret_cast<const void*>(func), device, attr, value);
}

} // namespace ninfer::ops::detail
