#pragma once

// ninfer::ops::detail - host-side geometry selection for the GQA launchers.
//
// Every launcher picks its Geometry through these two helpers, and both are generated from the
// registry lists in ops/kernel/gqa_attention_geometry.cuh. Registering a head geometry there
// therefore reaches every launcher at once; there is no per-launcher switch that a new geometry
// can be left out of, and an unregistered head count throws instead of silently falling through to
// whichever geometry happened to be last.

#include "ops/kernel/gqa_attention_geometry.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// --- registry coherence -----------------------------------------------------------------------
//
// src/ops/wrapper/gqa_attention.cpp's kv_heads_for_q_heads() mirrors this registry as three
// literal (q_heads, kv_heads) pairs, because that TU is host-only and this header is device-side
// (the repository's .cuh files are included from .cu only). These assertions are that mirror's
// compile-time tether.
//
// ADDING a geometry without updating the wrapper is already safe: the wrapper rejects an unknown
// Q-head count before any launcher is reached. EDITING a registered geometry's head counts is NOT:
// halve Gqa27Tp2Geometry::KVHeads, say, and the wrapper would keep validating a 2-head pool while
// the kernels stride by 4 through `paged_kv_element_offset<256, Geometry::KVHeads>` -- out-of-
// bounds cache reads with no diagnostic anywhere. These make that edit a build failure instead.
static_assert(Gqa27Geometry::QHeads == 24 && Gqa27Geometry::KVHeads == 4,
              "gqa_attention.cpp's kv_heads_for_q_heads() mirrors 24|4; update it and this assert "
              "together");
static_assert(Gqa35Geometry::QHeads == 16 && Gqa35Geometry::KVHeads == 2,
              "gqa_attention.cpp's kv_heads_for_q_heads() mirrors 16|2; update it and this assert "
              "together");
static_assert(Gqa27Tp2Geometry::QHeads == 12 && Gqa27Tp2Geometry::KVHeads == 2,
              "gqa_attention.cpp's kv_heads_for_q_heads() mirrors 12|2; update it and this assert "
              "together");

// Q-head counts must stay pairwise distinct, or selection by Q-head count is not a function.
static_assert(Gqa27Geometry::QHeads != Gqa35Geometry::QHeads &&
              Gqa27Geometry::QHeads != Gqa27Tp2Geometry::QHeads &&
              Gqa35Geometry::QHeads != Gqa27Tp2Geometry::QHeads);

// Every registered geometry's KV-head count must be served by an append representative.
constexpr bool gqa_kv_head_count_is_represented(int kv_heads) {
#define NINFER_GQA_REPRESENTATIVE_COVERS(Geometry)                                                 \
    if (kv_heads == Geometry::KVHeads) { return true; }
    NINFER_GQA_KV_REPRESENTATIVES(NINFER_GQA_REPRESENTATIVE_COVERS)
#undef NINFER_GQA_REPRESENTATIVE_COVERS
    return false;
}
#define NINFER_GQA_REQUIRE_REPRESENTATIVE(Geometry)                                                \
    static_assert(gqa_kv_head_count_is_represented(Geometry::KVHeads),                             \
                  "this registered GQA geometry's KV-head count has no append representative -- "  \
                  "add one to NINFER_GQA_KV_REPRESENTATIVES");
NINFER_GQA_GEOMETRIES(NINFER_GQA_REQUIRE_REPRESENTATIVE)
#undef NINFER_GQA_REQUIRE_REPRESENTATIVE

// --- selectors ---------------------------------------------------------------------------------

// Selects the registered geometry whose Q-head count matches. Q-head counts are pairwise distinct
// (asserted above), so this selection is unambiguous and independent of registry order.
template <typename Body>
decltype(auto) dispatch_gqa_geometry(std::int32_t q_heads, Body&& body) {
#define NINFER_GQA_DISPATCH_BY_Q_HEADS(Geometry)                                                   \
    if (q_heads == Geometry::QHeads) { return body.template operator()<Geometry>(); }
    NINFER_GQA_GEOMETRIES(NINFER_GQA_DISPATCH_BY_Q_HEADS)
#undef NINFER_GQA_DISPATCH_BY_Q_HEADS
    throw std::invalid_argument("gqa_attention: unsupported Q-head geometry");
}

// Selects the append representative for a KV-head count, for A2 (`gqa_kv_append`), which has no
// Q heads at all. The representatives carry pairwise-distinct KV-head counts by construction, so
// this too is order-independent; a registered geometry whose KV-head count is already represented
// (Gqa27Tp2Geometry, whose 2 KV heads Gqa35Geometry already covers) deliberately does NOT appear
// here, because a second entry with the same KV-head count would make this selection ambiguous.
// The kernels themselves are still instantiated for that geometry -- the batched prompt path
// calls `gqa_kv_append_launch_for<Geometry>` under `dispatch_gqa_geometry` above.
template <typename Body>
decltype(auto) dispatch_gqa_kv_geometry(std::int32_t kv_heads, Body&& body) {
#define NINFER_GQA_DISPATCH_BY_KV_HEADS(Geometry)                                                  \
    if (kv_heads == Geometry::KVHeads) { return body.template operator()<Geometry>(); }
    NINFER_GQA_KV_REPRESENTATIVES(NINFER_GQA_DISPATCH_BY_KV_HEADS)
#undef NINFER_GQA_DISPATCH_BY_KV_HEADS
    throw std::invalid_argument("gqa_kv_append: unsupported KV-head geometry");
}

} // namespace ninfer::ops::detail
