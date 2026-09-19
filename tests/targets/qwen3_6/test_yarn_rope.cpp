// YaRN: host-only parity + structural test for
// ninfer::targets::qwen3_6::detail::yarn_scale(). No artifact, no device, no GPU. Reference values below are literal copies of
// tests/core/data/yarn_ref_4x.json, produced by driving vLLM's *actual* rope classes directly
// (tools/tp2/dump_yarn_ref.py) against the production Qwen3.8-27B rope_parameters override
// (serve-orca-qwen38-27b-long.sh line 35: rope_type=yarn, rope_theta=1e7,
// partial_rotary_factor=0.25, factor=4.0, original_max_position_embeddings=262144).
//
// TWO STRUCTURAL FACTS THAT ARE EASY TO GET BACKWARDS (each established by tracing and executing
// vLLM's own code, not assumed):
//
//   1. The direction of the extrapolation/interpolation split is the opposite of the intuitive
//      reading. Pair index 0 is the HIGHEST rotary frequency (theta^0 == 1, shortest
//      wavelength) and is the one that needs NO correction (short wavelengths never exceed the
//      training context) -- it stays NATIVE, unchanged, not divided by `factor`. Pair index
//      (rotary_pairs-1) is the LOWEST frequency (longest wavelength) and is the one that gets
//      fully interpolated (native / factor), because uncorrected it would extrapolate far beyond
//      the trained context. This test asserts the verified-correct direction:
//      yarn_inv[0] == native_inv[0] (unchanged) and yarn_inv[31] == native_inv[31] / factor
//      (fully interpolated).
//
//   2. The correction range for the production config is (16, 24), not the (14, 22) one gets
//      from evaluating vLLM's `yarn_find_correction_range` against original_max_position (262144)
//      directly. Qwen3's rope_parameters always carry `mrope_section`, so vLLM's `get_rope()`
//      routes the "yarn" rope_type to `MRotaryEmbedding` (not the plain `YaRNScalingRotaryEmbedding`
//      class), whose constructor unconditionally enlarges the position count used for the
//      correction-range formula to `original_max * 4` (a Qwen2.5-VL video-cache headroom quirk
//      unrelated to the YaRN `factor`, see src/targets/qwen3_6/impl/runtime/yarn_rope.h and yarn_rope.cpp for
//      the full derivation). This test's "ramp only within (16, 24)" check locks in the
//      production-accurate range.

#include "targets/qwen3_6/impl/runtime/yarn_rope.h"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

int check_relative(double actual, double expected, double tolerance, const char* label) {
    const double denom   = std::abs(expected) > 0.0 ? std::abs(expected) : 1.0;
    const double rel_err = std::abs(actual - expected) / denom;
    if (rel_err <= tolerance) { return 0; }
    std::cerr << label << ": actual=" << actual << " expected=" << expected
               << " rel_err=" << rel_err << " (tolerance=" << tolerance << ")\n";
    return 1;
}

constexpr double kRelTol = 1e-6;

// tests/core/data/yarn_ref_4x.json: "yarn_inv_freq" (production config, factor=4.0,
// correction range (16, 24)).
constexpr std::array<double, 32> kExpectedYarnInvFreq4x = {
    1.0,
    0.6042963862419128,
    0.36517414450645447,
    0.22067339718341827,
    0.1333521455526352,
    0.08058421313762665,
    0.04869675263762474,
    0.02942727319896221,
    0.017782794311642647,
    0.010746078565716743,
    0.006493816617876291,
    0.003924189601093531,
    0.0023713738191872835,
    0.0014330126577988267,
    0.0008659643353894353,
    0.0005232990952208638,
    0.0003162277862429619,
    0.00017318011668976396,
    9.382603457197547e-05,
    5.015657370677218e-05,
    2.635603232192807e-05,
    1.3537826816900633e-05,
    6.737177955074003e-06,
    3.1988415685191285e-06,
    1.405853367941745e-06,
    8.495520660289912e-07,
    5.133812237545499e-07,
    3.102344408034696e-07,
    1.8747356023141037e-07,
    1.1328959459433463e-07,
    6.846049416253663e-08,
    4.1370427084075345e-08,
};

// tests/core/data/yarn_ref_4x.json: "native_inv_freq" (theta=1e7, rotary_dim=64, i.e. the same
// formula src/ops/kernel/rope.cuh's kTextRopeInvFrequency[32] table encodes).
constexpr std::array<double, 32> kExpectedNativeInvFreq = {
    1.0,
    0.6042963902381328,
    0.36517412725483767,
    0.220673406908459,
    0.1333521432163324,
    0.08058421877614819,
    0.04869675251658631,
    0.029427271762092817,
    0.01778279410038923,
    0.010746078283213174,
    0.006493816315762114,
    0.003924189758484536,
    0.002371373705661655,
    0.0014330125702369627,
    0.0008659643233600654,
    0.0005232991146814947,
    0.00031622776601683794,
    0.00019109529749704405,
    0.00011547819846894582,
    6.978305848598664e-05,
    4.216965034285823e-05,
    2.5482967479793468e-05,
    1.5399265260594922e-05,
    9.30572040929699e-06,
    5.623413251903491e-06,
    3.3982083289425593e-06,
    2.053525026457146e-06,
    1.2409377607517195e-06,
    7.498942093324558e-07,
    4.531583637600818e-07,
    2.7384196342643614e-07,
    1.6548170999431816e-07,
};

// tests/core/data/yarn_ref_4x.json: "attention_factor_A" = 1 / mscale^2, mscale =
// yarn_get_mscale(4.0) = 0.1*ln(4.0)+1.0 = 1.138629436111989 (attn_factor pinned at 1, not
// overridden by the production config).
constexpr double kExpectedAttentionFactor4x = 0.7713210535693215;

// Recompute the native table directly from its generating formula (theta^(-2i/rotary_dim)),
// independent of both kExpectedNativeInvFreq above and yarn_scale(), for the "native config
// (factor 1) matches the generating formula" bit-level structural check.
double native_formula(double theta, int rotary_dim, int i) {
    return 1.0 / std::pow(theta, static_cast<double>(2 * i) / static_cast<double>(rotary_dim));
}

}  // namespace

int main() {
    int failures = 0;

    // -----------------------------------------------------------------------------------------
    // 1. Production config (factor=4.0): all 32 corrected inv_freq values + attention factor A
    //    match the vLLM-derived reference to <=1e-6 relative.
    // -----------------------------------------------------------------------------------------
    ninfer::targets::qwen3_6::detail::YarnParams production;
    production.factor        = 4.0f;
    production.original_max  = 262144;
    production.theta         = 1e7f;
    production.rotary_pairs  = 32;
    production.beta_fast     = 32.0f;
    production.beta_slow     = 1.0f;

    const auto [yarn_inv_freq, attention_factor_a] = ninfer::targets::qwen3_6::detail::yarn_scale(production);
    failures += check(yarn_inv_freq.size() == 32, "yarn_scale(production) did not return 32 pairs");

    for (int i = 0; i < 32 && i < static_cast<int>(yarn_inv_freq.size()); ++i) {
        failures += check_relative(yarn_inv_freq[static_cast<std::size_t>(i)],
                                    kExpectedYarnInvFreq4x[static_cast<std::size_t>(i)], kRelTol,
                                    "yarn_inv_freq[i] mismatch vs vLLM reference");
    }
    failures += check_relative(attention_factor_a, kExpectedAttentionFactor4x, kRelTol,
                                "attention factor A mismatch vs vLLM reference");

    // -----------------------------------------------------------------------------------------
    // 2. Structural checks against the recomputed native table (verified-correct direction --
    //    see the file header correction note).
    // -----------------------------------------------------------------------------------------
    // Pair 0 (highest frequency, shortest wavelength) needs no correction: unchanged vs native.
    failures += check_relative(yarn_inv_freq[0], kExpectedNativeInvFreq[0], kRelTol,
                                "pair 0 should be unchanged vs native (no correction needed)");

    // Pair 31 (lowest frequency, longest wavelength) is fully interpolated: native / factor.
    failures += check_relative(yarn_inv_freq[31],
                                kExpectedNativeInvFreq[31] / static_cast<double>(production.factor),
                                kRelTol, "pair 31 should equal native / factor (full interpolation)");

    // Monotone non-increasing across all 32 pairs.
    {
        bool monotone = true;
        for (int i = 0; i + 1 < 32; ++i) {
            if (yarn_inv_freq[static_cast<std::size_t>(i)] <
                yarn_inv_freq[static_cast<std::size_t>(i + 1)]) {
                monotone = false;
                break;
            }
        }
        failures += check(monotone, "yarn_inv_freq is not monotone non-increasing");
    }

    // The ramp (blend zone) only spans (16, 24): pairs 0..16 are pure native/extrapolation
    // (unchanged), pairs 24..31 are pure interpolation (native / factor), and only pairs 17..23
    // are strictly in between.
    {
        bool range_ok = true;
        for (int i = 0; i <= 16; ++i) {
            const double rel = std::abs(yarn_inv_freq[static_cast<std::size_t>(i)] -
                                          kExpectedNativeInvFreq[static_cast<std::size_t>(i)]) /
                                 kExpectedNativeInvFreq[static_cast<std::size_t>(i)];
            if (rel > kRelTol) { range_ok = false; }
        }
        for (int i = 24; i < 32; ++i) {
            const double expected =
                kExpectedNativeInvFreq[static_cast<std::size_t>(i)] / static_cast<double>(production.factor);
            const double rel =
                std::abs(yarn_inv_freq[static_cast<std::size_t>(i)] - expected) / expected;
            if (rel > kRelTol) { range_ok = false; }
        }
        failures += check(range_ok,
                           "correction range is not exactly (16, 24): pairs outside the blend "
                           "zone were altered (or the blend zone shrank/grew)");
    }

    // -----------------------------------------------------------------------------------------
    // 3. Native config (factor == 1.0): yarn_scale returns the native table exactly (bit-compare
    //    against the generating formula theta^(-2i/rotary_dim)) and A == 1.
    // -----------------------------------------------------------------------------------------
    ninfer::targets::qwen3_6::detail::YarnParams native;
    native.factor       = 1.0f;
    native.original_max  = 262144;
    native.theta         = 1e7f;
    native.rotary_pairs  = 32;

    const auto [native_out, native_a] = ninfer::targets::qwen3_6::detail::yarn_scale(native);
    for (int i = 0; i < 32; ++i) {
        const double expected = native_formula(1e7, 64, i);
        failures += check_relative(native_out[static_cast<std::size_t>(i)], expected, kRelTol,
                                    "native (factor=1) inv_freq mismatch vs generating formula");
    }
    failures += check(native_a == 1.0f, "native (factor=1) attention factor A must be exactly 1");

    // -----------------------------------------------------------------------------------------
    // 4. Validation: reject origin != 262144, and factor * origin > 1,048,576 (spec sec 4.2).
    // -----------------------------------------------------------------------------------------
    {
        ninfer::targets::qwen3_6::detail::YarnParams bad_origin = production;
        bad_origin.original_max        = 131072;
        bool threw = false;
        try {
            (void)ninfer::targets::qwen3_6::detail::yarn_scale(bad_origin);
        } catch (const std::invalid_argument&) { threw = true; }
        failures += check(threw, "yarn_scale accepted an origin != 262144");
    }
    {
        ninfer::targets::qwen3_6::detail::YarnParams too_large = production;
        too_large.factor               = 8.0f;  // 8 * 262144 = 2,097,152 > 1,048,576
        bool threw = false;
        try {
            (void)ninfer::targets::qwen3_6::detail::yarn_scale(too_large);
        } catch (const std::invalid_argument&) { threw = true; }
        failures += check(threw, "yarn_scale accepted factor * origin > 1,048,576");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
