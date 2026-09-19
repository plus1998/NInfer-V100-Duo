// tp2 per-device KV pool sizing. Host-only: no artifact, no device, no Engine
// construction. Mirrors tests/test_kv_capacity.cpp's conventions and exercises
// runtime::kv_capacity in isolation.
//
// The table this test proves:
//   KV per token per device (tp2, INT8 group-64), MAIN K/V PLANES ONLY =
//       16 attn layers x 2 heads x 256 head_dim x 2 (K+V) x 1 byte  = 16 KiB/token/device
//   -> at    262144 logical tokens: 4096 pages x 1 MiB/page  =  4 GiB/device
//   -> at 1,048,576 logical tokens: 16384 pages x 1 MiB/page = 16 GiB/device
// BF16 doubles the per-token cost (32 KiB/token/device), so the same 1,048,576-token request
// needs 32 GiB/device and must be rejected on a device that does not have that much free.
//
// CAVEAT: the 16 KiB/token/device figure above covers the two main
// K/V data planes ONLY. It deliberately EXCLUDES the FP16 group-64 quant-scale planes that the
// real INT8 PagedKVPool also carries (src/targets/qwen3_6/impl/state/decoder_state.cpp:34-46,
// `plan_cache`: each quantized layer adds a K-scale and a V-scale plane, `DType::FP16`,
// `head_dim / quant_group` elements per head). With head_dim=256 and quant_group=64 that is
// 4 groups x 2 bytes x 2 (K+V) = 16 bytes/head/layer of scale, i.e. +0.5 KiB/token/device for a
// tp2 device's 16 layers x 2 heads -- real cost is ~16.5 KiB/token/device, ~3.1% above the
// main-planes-only number this test asserts. The real per-device curve (built from this same
// `plan_cache` geometry, with `kv_heads` halved for tp2) is target-layer work and is out of this
// test's scope, which is `kv_capacity.cpp` plus request memory only. The cross-check that curve
// owes this table: its `runtime_reservation_bytes` at 1,048,576 tokens should land at ~16.5
// GiB/device (+~3.1%), NOT exactly 16 GiB -- landing exactly on 16 would mean the quant-scale
// planes were dropped from the curve, and landing far from ~16.5 (double, or missing the FP16
// scale entirely) is a sizing bug.
//
// Page geometry (group-64 INT8, 64-token pages) is unchanged per head pair -- only the byte cost
// per page halves, because a tp2 device holds half the logical KV heads (2 of 4). The curves
// below encode exactly that: same `main_page_tokens = 64` as tp1, half the tp1 bytes-per-page.
//
// IMPORTANT semantics under test: `--kv-capacity N` (logical tokens) is the SAME N on every
// device -- not N/2. The *bytes* halve because the heads halve, not the token count. `auto` mode
// must read EACH device's own free VRAM independently and size the shared plan to the tightest
// device (the bottleneck), never averaging or assuming symmetric free memory.

#include "runtime/engine/kv_capacity.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

constexpr std::uint32_t kPageTokens = 64;

// 16 attn layers x 2 heads x 256 head_dim x 2 (K+V) x 1 byte(INT8) = 16 KiB/token/device.
// Main K/V data planes only -- excludes the FP16 group-64 quant-scale planes the real
// PagedKVPool also stores (see the file header caveat). Real cost is ~16.5 KiB/token/device.
constexpr std::size_t kBytesPerTokenInt8Tp2 = 16ULL * 1024ULL;
constexpr std::size_t kBytesPerPageInt8Tp2  = kBytesPerTokenInt8Tp2 * kPageTokens; // 1 MiB

// BF16 KV storage doubles the per-token byte cost relative to INT8 group-64.
constexpr std::size_t kBytesPerTokenBf16Tp2 = kBytesPerTokenInt8Tp2 * 2ULL;
constexpr std::size_t kBytesPerPageBf16Tp2  = kBytesPerTokenBf16Tp2 * kPageTokens; // 2 MiB

constexpr std::size_t kGiB = 1024ULL * 1024ULL * 1024ULL;

// A per-device curve whose reservation is exactly `main_page_groups * bytes_per_page` -- minimum
// page group 1 costs exactly one page's bytes, and every additional page costs one more page's
// bytes, so there is no base-reservation offset to account for in the assertions below.
ninfer::runtime::SequenceCapacityCurve make_tp2_curve(std::size_t bytes_per_page,
                                                       std::uint32_t maximum_main_page_groups) {
    return ninfer::runtime::SequenceCapacityCurve{
        .main_page_tokens                     = kPageTokens,
        .minimum_main_page_groups             = 1,
        .maximum_main_page_groups             = maximum_main_page_groups,
        .minimum_device_reservation_bytes     = bytes_per_page,
        .bytes_per_additional_main_page_group = bytes_per_page,
    };
}

} // namespace

int main() {
    int failures = 0;

    // -----------------------------------------------------------------------------------------
    // 1. Explicit capacity table: per-device pool bytes for 262144 and 1,048,576 logical tokens,
    //    INT8 group-64, MAIN K/V DATA PLANES ONLY (excludes FP16 quant-scale planes -- see file
    //    header caveat; real cost is ~3.1% higher). Same logical token count is requested on both
    //    devices; only the resulting BYTES are asserted per device, and they are identical because
    //    the curve (2-of-4 heads) is identical on both devices in a symmetric tp2 split.
    // -----------------------------------------------------------------------------------------
    {
        const auto curve = make_tp2_curve(kBytesPerPageInt8Tp2, 16384);

        const auto quarter_m = ninfer::runtime::resolve_kv_capacity(
            ninfer::KvCapacityPolicy::explicit_capacity(262144), curve, 64ULL * kGiB);
        failures += check(quarter_m.resolved_tokens == 262144,
                          "262144-token request did not resolve to the exact logical token count");
        failures += check(quarter_m.main_page_groups == 262144 / kPageTokens,
                          "262144-token request did not resolve to 4096 pages");
        failures += check(quarter_m.runtime_reservation_bytes == 4ULL * kGiB,
                          "262144-token INT8 tp2 pool did not size to 4 GiB/device");

        // Both devices see the SAME per-device reservation for the SAME requested token count --
        // this is the "N tokens on each device, not N/2" semantics this test guards against.
        const auto quarter_m_dev1 = ninfer::runtime::resolve_kv_capacity(
            ninfer::KvCapacityPolicy::explicit_capacity(262144), curve, 64ULL * kGiB);
        failures += check(quarter_m_dev1.runtime_reservation_bytes ==
                              quarter_m.runtime_reservation_bytes &&
                          quarter_m_dev1.resolved_tokens == quarter_m.resolved_tokens,
                          "identical per-device curves resolved 262144 tokens to different bytes");

        const auto one_m = ninfer::runtime::resolve_kv_capacity(
            ninfer::KvCapacityPolicy::explicit_capacity(1048576), curve, 64ULL * kGiB);
        failures += check(one_m.resolved_tokens == 1048576,
                          "1,048,576-token request did not resolve to the exact logical token count");
        failures += check(one_m.main_page_groups == 1048576 / kPageTokens,
                          "1,048,576-token request did not resolve to 16384 pages");
        failures += check(one_m.runtime_reservation_bytes == 16ULL * kGiB,
                          "1,048,576-token INT8 tp2 pool did not size to 16 GiB/device (spec "
                          "table, main K/V planes only -- real curve is ~16.5 GiB/device)");
    }

    // -----------------------------------------------------------------------------------------
    // 2. BF16-at-1M infeasibility: doubling the per-token byte cost to 32 KiB/token/device makes
    //    a 1,048,576-token request require 32 GiB/device. A device with less than that free must
    //    still reject the request -- the existing tp1 rejection behavior must hold per device.
    // -----------------------------------------------------------------------------------------
    {
        const auto curve = make_tp2_curve(kBytesPerPageBf16Tp2, 16384);

        // A realistic tp2 device (e.g. 32 GiB card, ~28 GiB free after weights) cannot fit it.
        bool rejected = false;
        try {
            (void)ninfer::runtime::resolve_kv_capacity(
                ninfer::KvCapacityPolicy::explicit_capacity(1048576), curve, 28ULL * kGiB);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected,
                          "BF16 tp2 pool at 1M tokens (32 GiB/device) was not rejected against a "
                          "28 GiB/device budget");

        // The exact 32 GiB/device requirement, confirmed by giving it exactly enough.
        const auto exact = ninfer::runtime::resolve_kv_capacity(
            ninfer::KvCapacityPolicy::explicit_capacity(1048576), curve, 32ULL * kGiB);
        failures += check(exact.runtime_reservation_bytes == 32ULL * kGiB,
                          "BF16 tp2 pool at 1M tokens did not size to exactly 32 GiB/device");
    }

    // -----------------------------------------------------------------------------------------
    // 3. tp1 curve (4 heads, no tp2 halving) at the same page geometry needs double the tp2 INT8
    //    bytes for the same token count -- pins that the halving is really from the head count,
    //    not from some other page-size change.
    // -----------------------------------------------------------------------------------------
    {
        const auto tp2_curve = make_tp2_curve(kBytesPerPageInt8Tp2, 16384);
        const auto tp1_curve = make_tp2_curve(kBytesPerPageInt8Tp2 * 2ULL, 16384);

        const auto tp2_resolution = ninfer::runtime::resolve_kv_capacity(
            ninfer::KvCapacityPolicy::explicit_capacity(1048576), tp2_curve, 64ULL * kGiB);
        const auto tp1_resolution = ninfer::runtime::resolve_kv_capacity(
            ninfer::KvCapacityPolicy::explicit_capacity(1048576), tp1_curve, 64ULL * kGiB);

        failures += check(tp2_resolution.resolved_tokens == tp1_resolution.resolved_tokens,
                          "tp1 and tp2 curves resolved a different logical token count for the "
                          "same explicit request (logical tokens must not be halved)");
        failures += check(tp1_resolution.runtime_reservation_bytes ==
                              2ULL * tp2_resolution.runtime_reservation_bytes,
                          "tp2 per-device bytes were not exactly half the tp1 (4-head) bytes for "
                          "the same logical token count");
    }

    // -----------------------------------------------------------------------------------------
    // 4. resolve_kv_capacity_symmetric: automatic mode reads EACH device's own free VRAM and
    //    sizes the single shared plan to the tightest device, never averaging.
    // -----------------------------------------------------------------------------------------
    {
        const auto curve = make_tp2_curve(kBytesPerPageInt8Tp2, 16384);

        // Device 0 has plenty of headroom; device 1 is the bottleneck at 20 GiB free.
        const std::array<std::size_t, 2> per_device_bytes{40ULL * kGiB, 20ULL * kGiB};
        const auto resolution = ninfer::runtime::resolve_kv_capacity_symmetric(
            ninfer::KvCapacityPolicy::automatic(1ULL * kGiB), curve, per_device_bytes);

        // Budget after the 1 GiB automatic headroom is 19 GiB on the bottleneck device ->
        // 19 GiB / 1 MiB-per-page = 19456 pages, capped at the curve's maximum of 16384.
        failures += check(resolution.main_page_groups == 16384,
                          "automatic tp2 resolution did not cap at the curve's maximum page count");
        failures += check(resolution.runtime_reservation_bytes == 16ULL * kGiB,
                          "automatic tp2 resolution did not size the capped plan to 16 GiB/device");

        // Symmetry check: resolving explicitly against ONLY the bottleneck device's budget must
        // give the identical answer resolve_kv_capacity_symmetric gave across both devices.
        const auto bottleneck_only = ninfer::runtime::resolve_kv_capacity(
            ninfer::KvCapacityPolicy::automatic(1ULL * kGiB), curve, 20ULL * kGiB);
        failures += check(bottleneck_only.main_page_groups == resolution.main_page_groups &&
                              bottleneck_only.runtime_reservation_bytes ==
                                  resolution.runtime_reservation_bytes,
                          "resolve_kv_capacity_symmetric did not reduce to the bottleneck device's "
                          "own resolution");

        // Reversing which device is the bottleneck must give the same answer -- order-independent.
        const std::array<std::size_t, 2> per_device_bytes_reordered{20ULL * kGiB, 40ULL * kGiB};
        const auto resolution_reordered = ninfer::runtime::resolve_kv_capacity_symmetric(
            ninfer::KvCapacityPolicy::automatic(1ULL * kGiB), curve, per_device_bytes_reordered);
        failures += check(resolution_reordered.main_page_groups == resolution.main_page_groups &&
                              resolution_reordered.runtime_reservation_bytes ==
                                  resolution.runtime_reservation_bytes,
                          "resolve_kv_capacity_symmetric was not order-independent across devices");
    }

    // -----------------------------------------------------------------------------------------
    // 5. resolve_kv_capacity_symmetric with a single-element span must reduce EXACTLY to
    //    resolve_kv_capacity -- this is the tp1 byte-identity guarantee: a tp1 caller that chose
    //    to route through the symmetric entry point would see no behavior change.
    // -----------------------------------------------------------------------------------------
    {
        const auto curve = make_tp2_curve(kBytesPerPageInt8Tp2, 16384);
        const std::array<std::size_t, 1> one_device{24ULL * kGiB};

        const auto via_symmetric = ninfer::runtime::resolve_kv_capacity_symmetric(
            ninfer::KvCapacityPolicy::explicit_capacity(262144), curve, one_device);
        const auto direct = ninfer::runtime::resolve_kv_capacity(
            ninfer::KvCapacityPolicy::explicit_capacity(262144), curve, 24ULL * kGiB);
        failures +=
            check(via_symmetric.main_page_groups == direct.main_page_groups &&
                      via_symmetric.runtime_reservation_bytes == direct.runtime_reservation_bytes &&
                      via_symmetric.resolved_tokens == direct.resolved_tokens,
                  "single-device resolve_kv_capacity_symmetric did not match resolve_kv_capacity "
                  "exactly (tp1 byte-identity)");
    }

    // -----------------------------------------------------------------------------------------
    // 6. resolve_kv_capacity_symmetric rejects an empty device list.
    // -----------------------------------------------------------------------------------------
    {
        const auto curve = make_tp2_curve(kBytesPerPageInt8Tp2, 16384);
        bool rejected = false;
        try {
            (void)ninfer::runtime::resolve_kv_capacity_symmetric(
                ninfer::KvCapacityPolicy::explicit_capacity(262144), curve, {});
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "resolve_kv_capacity_symmetric accepted an empty device list");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
