// RoPE frequency-override (YaRN) correctness.
//
// The Op under test is `ninfer::ops::rope`'s RopeFrequencyOverride form: the per-pair inverse
// frequency comes from a device buffer instead of the built-in theta table, and both cos and sin
// are scaled by `mscale`. Three independent claims are checked:
//
//   1. Native identity. With the buffer holding the frequencies the native table encodes and
//      mscale == 1, the override output is BIT-IDENTICAL to the ordinary call. The table is not
//      copied from the kernel: it comes from `yarn_scale({.factor = 1})`, which is the corrected
//      table degenerating to extrapolation, so this leg also pins "factor 1 == native".
//   2. YaRN rotation. With the factor-4 table and its mscale, the output matches an independent
//      FP64 evaluation of `mscale * R(position * inv_freq[i]) * x_rot` over positions spanning
//      the 1,048,576-token context, and dimensions [64,256) are untouched bit-for-bit.
//   3. Per-device buffers. Claim 2 again, once per visible GPU, each reading a table resident on
//      the device its q/k live on — the tp2 shape, where each rank ropes its head-local q/k
//      (12Q:2KV) against its own copy of the table.
//
// The oracle is naive FP64 from the represented BF16 inputs and the exact float frequencies that
// were uploaded. It does not reproduce the kernel's shared-memory cache, its block split, or its
// float phase product; the phase product is instead paid for in the criterion (see
// `rotation_limit`).
//
// --- Rotary-subspace mscale verification (no attention-level factor) ---------------------------
//
// An earlier design sketch proposed an "attention soft-cap factor A" applied to the softmax scale
// in yarn mode. The shipped design does not do that: mscale is applied ENTIRELY inside this rope
// path (both cos and sin, as already checked above), and the attention softmax
// `scale` argument is untouched in yarn mode -- see the note next to `RopeFrequencyOverride` in
// include/ninfer/ops/rope.h and the one next to `rope_frequency_` in
// src/targets/qwen3_6/impl/runtime/text_context.h for where `scale` is computed and why it does
// not change. What follows checks the algebra that substitutes for the abandoned attention
// factor: with the SAME (native) inverse-frequency table, moving mscale from 1 to M multiplies
// the rotated OUTPUT of both q and k by M (mscale multiplies cos and sin, i.e. the whole rotated
// vector, not the phase), so for any two vectors rotated at arbitrary -- possibly different --
// positions:
//
//   q'_M . k'_M  ==  q'_1 . k'_1  +  (M^2 - 1) * (q'_1_rot . k'_1_rot)
//
// where q'_1/k'_1 is the mscale=1 (plain native) rotation of the same inputs through the same
// table, and the trailing `_rot` dot is restricted to the rotary subspace [0,rotary_dim). Two
// legs check this: (a) at the dot-product level, directly from two rope Op outputs (mscale=1 and
// mscale=M, same native table) -- both an exact "zeroed pass-through" case and a general case;
// (b) at the attention level, feeding the SAME mscale=M rope output into the real
// `ops::gqa_attention` Op and comparing its output against a softmax+weighted-V
// oracle whose PRE-softmax score is built from the mscale=1 output's decomposed dot product via
// the formula above -- i.e. the prediction is derived from mscale=1 data and the formula, not
// from re-running the standard oracle on already-yarn-roped inputs.

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/rope.h"
#include "ops/gqa_attention_fixture.h"
#include "ops/op_tester.h"
#include "targets/qwen3_6/impl/runtime/yarn_rope.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

namespace yarn = ninfer::targets::qwen3_6::detail;
namespace tgqa = ninfer::test::gqa;

constexpr int kHeadDim = 256;
constexpr int kRotary  = 64;
constexpr int kPairs   = kRotary / 2;
constexpr float kTheta = 1.0e7F;

// The BF16 pointwise profile the production RoPE Op is qualified against (tests/ops/test_rope.cpp):
// either member of a rotated pair can cancel, so the bound is relative to the FP64 norm of the
// input pair rather than to the cancelled output.
constexpr double kRopePointwisePairRtol = 6.9e-3;

// Positions spanning the YaRN-extended context, including both ends of 1,048,576.
const std::vector<int> kSweepPositions{0, 1, 1023, 262'144, 524'288, 1'048'575};

struct Table {
    std::vector<float> inverse_frequency;
    float mscale = 1.0F;
};

Table yarn_table(float factor) {
    yarn::YarnParams params;
    params.factor       = factor;
    params.rotary_pairs = kPairs;
    params.theta        = kTheta;
    Table table;
    table.inverse_frequency = yarn::yarn_scale(params).first;
    table.mscale            = yarn::yarn_rope_mscale(params);
    return table;
}

std::size_t dense_index(int heads, int token, int head, int dim) {
    return (static_cast<std::size_t>(token) * static_cast<std::size_t>(heads) +
            static_cast<std::size_t>(head)) *
               static_cast<std::size_t>(kHeadDim) +
           static_cast<std::size_t>(dim);
}

std::vector<float> make_input_range(int heads, int tokens, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(static_cast<std::size_t>(kHeadDim) * heads * tokens);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(lo, hi);
    for (float& value : values) { value = bf16_to_f32(f32_to_bf16(distribution(generator))); }
    return values;
}

std::vector<float> make_input(int heads, int tokens, std::uint32_t seed) {
    return make_input_range(heads, tokens, seed, -4.0F, 4.0F);
}

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    std::transform(values.begin(), values.end(), bits.begin(),
                   [](float value) { return f32_to_bf16(value); });
    return bits;
}

// [T] for axes == 1, [T,3] for MRoPE. Axis 0 walks the swept positions; the other two axes are
// offset so a mis-selected axis cannot pass by coincidence.
std::vector<int> make_positions(int axes, int tokens) {
    constexpr int kContextCeiling = 1'048'575;
    const int sweep_size          = static_cast<int>(kSweepPositions.size());
    std::vector<int> positions(static_cast<std::size_t>(axes) * tokens);
    for (int axis = 0; axis < axes; ++axis) {
        for (int token = 0; token < tokens; ++token) {
            const int sweep = kSweepPositions[static_cast<std::size_t>(token % sweep_size)];
            const int step  = token / sweep_size;
            // Walk away from each swept anchor so long cases carry distinct positions, without
            // ever leaving [0, 1048575].
            int base = sweep + step > kContextCeiling ? sweep - step : sweep + step;
            base     = std::clamp(base, 0, kContextCeiling);
            positions[static_cast<std::size_t>(axis) * tokens + token] =
                axis == 0 ? base : std::clamp(base - 7 * axis, 0, kContextCeiling);
        }
    }
    return positions;
}

// One rope launch. `frequency` selects native (default-constructed) or override mode. Returns the
// full BF16 storage of q and k so both rotary and pass-through dimensions can be inspected.
struct Launch {
    std::vector<std::uint16_t> q;
    std::vector<std::uint16_t> k;
    int guard_failures = 0;
};

Launch run_rope(const std::vector<int>& positions, int axes, int tokens, int q_heads, int k_heads,
                const std::vector<std::uint16_t>& q_bits, const std::vector<std::uint16_t>& k_bits,
                const Table* table, const std::string& label) {
    GuardedDeviceBuffer q_device(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer position_device(positions.size() * sizeof(int));
    q_device.copy_from_host(q_bits.data(), q_device.bytes());
    position_device.copy_from_host(positions.data(), position_device.bytes());

    Tensor position_tensor(position_device.data(), DType::I32, {tokens, axes});
    Tensor q_tensor(q_device.data(), DType::BF16, {kHeadDim, q_heads, tokens});

    ops::RopeFrequencyOverride frequency;
    GuardedDeviceBuffer frequency_device((table == nullptr ? 1 : table->inverse_frequency.size()) *
                                         sizeof(float));
    if (table != nullptr) {
        frequency_device.copy_from_host(table->inverse_frequency.data(),
                                        table->inverse_frequency.size() * sizeof(float));
        frequency.inv_frequency = static_cast<const float*>(frequency_device.data());
        frequency.mscale        = table->mscale;
    }

    Launch result;
    if (k_heads == 0) {
        ops::rope(position_tensor, kRotary, kTheta, q_tensor, frequency, nullptr);
        cuda_synchronize();
        result.q = from_device<std::uint16_t>(q_device.data(), q_bits.size());
    } else {
        GuardedDeviceBuffer k_device(k_bits.size() * sizeof(std::uint16_t));
        k_device.copy_from_host(k_bits.data(), k_device.bytes());
        Tensor k_tensor(k_device.data(), DType::BF16, {kHeadDim, k_heads, tokens});
        ops::rope(position_tensor, kRotary, kTheta, q_tensor, k_tensor, frequency, nullptr);
        cuda_synchronize();
        result.q = from_device<std::uint16_t>(q_device.data(), q_bits.size());
        result.k = from_device<std::uint16_t>(k_device.data(), k_bits.size());
        result.guard_failures += k_device.verify_guards(label + " k guards");
    }
    result.guard_failures += q_device.verify_guards(label + " q guards");
    result.guard_failures += position_device.verify_guards(label + " position guards");
    result.guard_failures += frequency_device.verify_guards(label + " frequency guards");
    return result;
}

// FP64 oracle: mscale * R(phase) applied to the rotary half-pairs, pass-through untouched.
std::vector<double> rope_oracle(const std::vector<float>& input, const std::vector<int>& positions,
                                int axes, int tokens, int heads, const Table& table) {
    std::vector<double> output(input.begin(), input.end());
    const double mscale = static_cast<double>(table.mscale);
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            for (int pair = 0; pair < kPairs; ++pair) {
                const int axis     = axes == 3 ? pair % 3 : 0;
                const double phase = static_cast<double>(
                                         positions[static_cast<std::size_t>(axis) * tokens + token]) *
                                     static_cast<double>(
                                         table.inverse_frequency[static_cast<std::size_t>(pair)]);
                const double cosine  = std::cos(phase) * mscale;
                const double sine    = std::sin(phase) * mscale;
                const std::size_t lo = dense_index(heads, token, head, pair);
                const std::size_t hi = dense_index(heads, token, head, pair + kPairs);
                const double first   = static_cast<double>(input[lo]);
                const double second  = static_cast<double>(input[hi]);
                output[lo]           = first * cosine - second * sine;
                output[hi]           = second * cosine + first * sine;
            }
        }
    }
    return output;
}

// Error budget for one rotated component.
//
//   * `kRopePointwisePairRtol * mscale * pair_norm` — the qualified BF16 profile of the rotation
//     itself, scaled because the override makes the output mscale times larger.
//   * `phase_slack * mscale * pair_norm` — the kernel forms the angle as the float product
//     `float(position) * inv_freq[i]`, whose rounding is bounded by 2^-24 of the angle. A phase
//     error d rotates the output by d, so it costs at most |x| * d. This term is what makes the
//     comparison honest at position 1,048,575 (where an angle near 1e6 rad carries ~0.06 rad of
//     float rounding) without loosening the check for the small-angle pairs, which stay at the
//     BF16 profile. vLLM's cos/sin cache is built with the same float32 product (and then cast to
//     the query dtype), so this is reference behavior, not a defect of this kernel.
double rotation_limit(double pair_norm, double phase, double mscale) {
    const double phase_slack = std::ldexp(1.0, -24) * std::abs(phase);
    return mscale * pair_norm * (kRopePointwisePairRtol + phase_slack);
}

int verify_rotation(const std::string& label, const std::vector<std::uint16_t>& got,
                    const std::vector<double>& expected, const std::vector<float>& input,
                    const std::vector<int>& positions, int axes, int tokens, int heads,
                    const Table& table) {
    int violations           = 0;
    double worst_ratio       = 0.0;
    const double mscale      = static_cast<double>(table.mscale);
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            for (int dim = 0; dim < kRotary; ++dim) {
                const int pair       = dim < kPairs ? dim : dim - kPairs;
                const int axis       = axes == 3 ? pair % 3 : 0;
                const double phase   = static_cast<double>(
                                         positions[static_cast<std::size_t>(axis) * tokens + token]) *
                                     static_cast<double>(
                                         table.inverse_frequency[static_cast<std::size_t>(pair)]);
                const std::size_t index = dense_index(heads, token, head, dim);
                const std::size_t lo    = dense_index(heads, token, head, pair);
                const std::size_t hi    = dense_index(heads, token, head, pair + kPairs);
                const double value      = static_cast<double>(bf16_to_f32(got[index]));
                if (!std::isfinite(value) || !std::isfinite(expected[index])) {
                    std::cerr << label << ": non-finite output at index=" << index << '\n';
                    return 1;
                }
                const double pair_norm =
                    std::hypot(static_cast<double>(input[lo]), static_cast<double>(input[hi]));
                const double error = std::abs(value - expected[index]);
                const double limit = rotation_limit(pair_norm, phase, mscale);
                worst_ratio        = std::max(worst_ratio, limit == 0.0 ? (error == 0.0 ? 0.0 : 1e9)
                                                                        : error / limit);
                if (error > limit) {
                    ++violations;
                    if (violations == 1) {
                        std::cerr << label << ": rotation mismatch at token=" << token
                                  << " head=" << head << " dim=" << dim << " got=" << value
                                  << " expected=" << expected[index] << " error=" << error
                                  << " limit=" << limit << '\n';
                    }
                }
            }
        }
    }
    if (error_stats_enabled()) {
        std::printf("OP_ERROR_STATS kind=scaled_pointwise case=%s worst_limit_ratio=%.17g\n",
                    label.c_str(), worst_ratio);
    }
    if (violations != 0) {
        std::cerr << label << ": " << violations << " rotated values exceed the YaRN profile\n";
        return 1;
    }
    return 0;
}

int verify_passthrough(const std::string& label, const std::vector<std::uint16_t>& got,
                       const std::vector<std::uint16_t>& before, int tokens, int heads) {
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            for (int dim = kRotary; dim < kHeadDim; ++dim) {
                const std::size_t index = dense_index(heads, token, head, dim);
                if (got[index] != before[index]) {
                    std::cerr << label << ": pass-through dimension changed at token=" << token
                              << " head=" << head << " dim=" << dim << '\n';
                    return 1;
                }
            }
        }
    }
    return 0;
}

// --- leg 1: native table through the buffer == the built-in table, exactly -------------------
int run_native_identity_case(const char* label, int axes, int tokens, int q_heads, int k_heads) {
    const Table native      = yarn_table(1.0F);
    const auto positions    = make_positions(axes, tokens);
    const auto q            = make_input(q_heads, tokens, 0x51D1U);
    const auto k            = make_input(std::max(k_heads, 1), tokens, 0x51D2U);
    const auto q_bits       = to_bf16_bits(q);
    const auto k_bits       = to_bf16_bits(k);
    const std::string name  = label;

    if (native.mscale != 1.0F) {
        std::cerr << name << ": factor 1 must yield mscale 1, got " << native.mscale << '\n';
        return 1;
    }

    const Launch reference =
        run_rope(positions, axes, tokens, q_heads, k_heads, q_bits, k_bits, nullptr, name + " ref");
    const Launch buffered = run_rope(positions, axes, tokens, q_heads, k_heads, q_bits, k_bits,
                                     &native, name + " buffer");

    int failures = reference.guard_failures + buffered.guard_failures;
    failures += verify_exact((name + " q bit-identity").c_str(), buffered.q, reference.q);
    if (k_heads != 0) {
        failures += verify_exact((name + " k bit-identity").c_str(), buffered.k, reference.k);
    }
    // A rope that did nothing would also be "identical": require the rotation to have happened.
    if (buffered.q == q_bits) {
        std::cerr << name << ": q storage unchanged — the rope did not run\n";
        ++failures;
    }
    return failures;
}

// --- legs 2 and 4: the factor-4 table against the FP64 oracle --------------------------------
int run_yarn_case(const char* label, int axes, int tokens, int q_heads, int k_heads,
                  bool drop_mscale = false) {
    Table table = yarn_table(4.0F);
    if (drop_mscale) {
        table.mscale = 1.0F;
        std::cout << "control: the diagnostics below are EXPECTED — the oracle deliberately "
                     "drops mscale while the kernel keeps it\n";
    }
    const auto positions   = make_positions(axes, tokens);
    const auto q           = make_input(q_heads, tokens, 0x7A21U);
    const auto k           = make_input(std::max(k_heads, 1), tokens, 0x7A22U);
    const auto q_bits      = to_bf16_bits(q);
    const auto k_bits      = to_bf16_bits(k);
    const std::string name = label;

    // The kernel is driven with the real mscale even in the negative control; only the oracle's
    // copy is neutered, which is exactly the "someone forgot to scale cos/sin" failure.
    Table launch_table = yarn_table(4.0F);
    const Launch got   = run_rope(positions, axes, tokens, q_heads, k_heads, q_bits, k_bits,
                                  &launch_table, name);

    const auto q_expected = rope_oracle(q, positions, axes, tokens, q_heads, table);
    int failures          = got.guard_failures;
    failures += verify_rotation(name + " q", got.q, q_expected, q, positions, axes, tokens, q_heads,
                                table);
    failures += verify_passthrough(name + " q", got.q, q_bits, tokens, q_heads);
    if (k_heads != 0) {
        const auto k_expected = rope_oracle(k, positions, axes, tokens, k_heads, table);
        failures += verify_rotation(name + " k", got.k, k_expected, k, positions, axes, tokens,
                                    k_heads, table);
        failures += verify_passthrough(name + " k", got.k, k_bits, tokens, k_heads);
    }
    return failures;
}

// --- domain: an override outside the Text D256/R64 domain must be refused, never ignored -----
int run_rejection_case() {
    constexpr int kVisionDim = 72;
    constexpr int kTokens    = 4;
    const Table table        = yarn_table(4.0F);
    std::vector<int> positions(static_cast<std::size_t>(2) * kTokens, 5);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(kVisionDim) * 16 * kTokens, 0);

    GuardedDeviceBuffer x_device(x.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer position_device(positions.size() * sizeof(int));
    GuardedDeviceBuffer frequency_device(table.inverse_frequency.size() * sizeof(float));
    x_device.copy_from_host(x.data(), x_device.bytes());
    position_device.copy_from_host(positions.data(), position_device.bytes());
    frequency_device.copy_from_host(table.inverse_frequency.data(), frequency_device.bytes());

    Tensor position_tensor(position_device.data(), DType::I32, {kTokens, 2});
    Tensor x_tensor(x_device.data(), DType::BF16, {kVisionDim, 16, kTokens});
    ops::RopeFrequencyOverride frequency{static_cast<const float*>(frequency_device.data()),
                                         table.mscale};
    try {
        ops::rope(position_tensor, kVisionDim, 10'000.0F, x_tensor, frequency, nullptr);
    } catch (const std::invalid_argument&) {
        return 0;
    }
    cuda_synchronize();
    std::cerr << "vision override: a frequency override outside the Text domain was accepted\n";
    return 1;
}

int run_per_device_cases() {
    int devices = 0;
    cuda_check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    int failures = 0;
    for (int device = 0; device < devices; ++device) {
        cuda_check(cudaSetDevice(device), "cudaSetDevice");
        const std::string label = "tp2 shard rope on device " + std::to_string(device);
        // The tp2 head-local attention geometry: each rank ropes 12 Q and 2 KV heads against the
        // table resident on its own device.
        failures += run_yarn_case(label.c_str(), 1, 6, 12, 2);
    }
    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    if (devices < 2) {
        std::cout << "NOTE: only " << devices
                  << " CUDA device(s) visible; the per-device buffer leg ran on what is here\n";
    }
    return failures;
}

// === Leg (a): dot-product level mscale^2 algebra ============================================
//
// Both cases rope the SAME q/k through the SAME (native, factor==1) inverse-frequency table
// twice -- once at mscale==1 (the "native" run) and once at mscale==M (the factor-4 mscale,
// reused here purely as a nontrivial M != 1; the inverse-frequency table stays the native one
// in BOTH runs, isolating mscale from the table correction the frequency-table legs above check).
// The check is the dot product of one rope Op output against the other, evaluated on their own
// terms -- no attention, no softmax.

// Sum of q[base_q+d]*k[base_k+d] over d in [lo,hi), decoding both sides from BF16 storage.
double bf16_dot_range(const std::vector<std::uint16_t>& q, std::size_t base_q,
                      const std::vector<std::uint16_t>& k, std::size_t base_k, int lo, int hi) {
    double sum = 0.0;
    for (int d = lo; d < hi; ++d) {
        sum += static_cast<double>(bf16_to_f32(q[base_q + static_cast<std::size_t>(d)])) *
               static_cast<double>(bf16_to_f32(k[base_k + static_cast<std::size_t>(d)]));
    }
    return sum;
}

// Conservative dot-product-level bound: two INDEPENDENT bf16-rounded rope outputs (the mscale=1
// run and the mscale=M run) each carry up to `kRopePointwisePairRtol` relative error per rotated
// component (the qualified rope profile above), so the predicted-vs-actual dot product -- built
// from one run and checked against the other -- can carry up to twice that before even accounting
// for the sum over kRotary/kHeadDim components. This is deliberately generous: it is a documented
// verification bound for a cross-run algebraic identity, not the rope kernel's own qualified
// tolerance (that is `kRopePointwisePairRtol`, checked bit-for-bit against the FP64 oracle above).
constexpr double kDotAlgebraRtol = 3.0e-2;
constexpr double kDotAlgebraAtol = 5.0e-4;

int check_dot_prediction(const std::string& label, double actual, double predicted,
                         double reference_magnitude) {
    const double limit = kDotAlgebraAtol + kDotAlgebraRtol * std::abs(reference_magnitude);
    const double error  = std::abs(actual - predicted);
    if (error > limit) {
        std::cerr << label << ": mscale^2 dot prediction mismatch actual=" << actual
                  << " predicted=" << predicted << " error=" << error << " limit=" << limit
                  << '\n';
        return 1;
    }
    return 0;
}

// `zero_passthrough`: q/k carry zeros outside the rotary dims, so native_dot == rotary_dot and
// the prediction degenerates to the EXACT relation M^2 * native_dot (still bounded only by BF16
// rounding of the products, since the rotation itself was already qualified above). Otherwise a
// general case: native_dot != rotary_dot, and the prediction is native_dot + (M^2-1)*rotary_dot.
int run_mscale_dot_case(const char* label, bool zero_passthrough) {
    constexpr int kTokens = 3;
    // Distinct positions spanning small, mid, and YaRN-extended context so the rotation phase
    // itself is nontrivial; q and k are cross-checked at every (query token, key token) pair; a
    // shared position array is required by `run_rope`'s call shape, but nothing else pairs q[i]
    // with k[i] specifically -- the dot products below deliberately also use i != j.
    const std::vector<int> positions{37, 262'200, 1'000'000};

    auto q = make_input(1, kTokens, 0x9101U);
    auto k = make_input(1, kTokens, 0x9102U);
    if (zero_passthrough) {
        for (int t = 0; t < kTokens; ++t) {
            for (int d = kRotary; d < kHeadDim; ++d) {
                q[dense_index(1, t, 0, d)] = 0.0F;
                k[dense_index(1, t, 0, d)] = 0.0F;
            }
        }
    }
    const auto q_bits = to_bf16_bits(q);
    const auto k_bits = to_bf16_bits(k);

    Table native_table = yarn_table(1.0F); // mscale == 1, table degenerates to the native one
    Table scaled_table  = native_table;    // SAME table
    scaled_table.mscale = yarn_table(4.0F).mscale; // M != 1, table left untouched (native)

    const Launch native_run =
        run_rope(positions, 1, kTokens, 1, 1, q_bits, k_bits, &native_table, std::string(label) + " native");
    const Launch scaled_run =
        run_rope(positions, 1, kTokens, 1, 1, q_bits, k_bits, &scaled_table, std::string(label) + " scaled");

    int failures = native_run.guard_failures + scaled_run.guard_failures;
    const double m       = static_cast<double>(scaled_table.mscale);
    const double m2m1    = m * m - 1.0;
    bool saw_nonzero_gap = false;

    for (int qi = 0; qi < kTokens; ++qi) {
        for (int ki = 0; ki < kTokens; ++ki) {
            const std::size_t base_q = dense_index(1, qi, 0, 0);
            const std::size_t base_k = dense_index(1, ki, 0, 0);
            const double native_dot =
                bf16_dot_range(native_run.q, base_q, native_run.k, base_k, 0, kHeadDim);
            const double rotary_dot =
                bf16_dot_range(native_run.q, base_q, native_run.k, base_k, 0, kRotary);
            const double predicted = native_dot + m2m1 * rotary_dot;
            const double actual =
                bf16_dot_range(scaled_run.q, base_q, scaled_run.k, base_k, 0, kHeadDim);

            const std::string pair_label = std::string(label) + " q_tok=" + std::to_string(qi) +
                                           " k_tok=" + std::to_string(ki);
            failures += check_dot_prediction(pair_label, actual, predicted, native_dot);

            if (zero_passthrough) {
                // Exact leg: native_dot must equal rotary_dot (nothing lives outside rotary).
                failures += check_dot_prediction(pair_label + " (native==rotary)", native_dot,
                                                 rotary_dot, native_dot);
            }
            if (std::abs(m2m1 * rotary_dot) > 1.0e-3 * std::max(1.0, std::abs(native_dot))) {
                saw_nonzero_gap = true;
            }
        }
    }

    if (!saw_nonzero_gap) {
        std::cerr << label
                  << ": the mscale term never moved the dot product enough to test anything\n";
        ++failures;
    }

    // Negative control: an M==1 prediction (i.e. pretending mscale was never applied) must be
    // REJECTED at at least one pair -- otherwise this check has no power to catch a regression
    // that dropped the mscale multiply from the kernel.
    int control_failures  = 0;
    bool control_rejected = false;
    for (int qi = 0; qi < kTokens; ++qi) {
        for (int ki = 0; ki < kTokens; ++ki) {
            const std::size_t base_q = dense_index(1, qi, 0, 0);
            const std::size_t base_k = dense_index(1, ki, 0, 0);
            const double native_dot =
                bf16_dot_range(native_run.q, base_q, native_run.k, base_k, 0, kHeadDim);
            const double actual =
                bf16_dot_range(scaled_run.q, base_q, scaled_run.k, base_k, 0, kHeadDim);
            const double limit =
                kDotAlgebraAtol + kDotAlgebraRtol * std::abs(native_dot);
            if (std::abs(actual - native_dot) > limit) { control_rejected = true; }
        }
    }
    if (!control_rejected) {
        std::cerr << label
                  << ": mscale=1 negative control was NOT rejected -- the check cannot "
                     "distinguish an unscaled kernel from a correctly-scaled one\n";
        ++control_failures;
    } else {
        std::cout << "control: " << label
                  << " mscale=1 prediction was correctly rejected (expected)\n";
    }
    return failures + control_failures;
}

// === Leg (b): attention-level mscale^2 prediction ============================================
//
// The SAME algebraic identity as leg (a), now checked through the real `ops::gqa_attention` Op
// (A1 append+attend): q/k are yarn-roped (native table, mscale=M) and fed through the
// production Op exactly as text_context_impl.h's forward path does; the prediction is built not
// by re-running the standard ideal-attention oracle on those yarn-roped values (which would only
// prove the Op matches its own contract, already covered by test_gqa_attention.cpp), but from the
// mscale=1 rope output's dot product, decomposed into native and rotary-subspace parts per the
// formula above, then pushed through the identical softmax + weighted-V reduction. Agreement
// between the two demonstrates that the mscale^2 rotary-subspace effect survives the real fused
// attention kernel exactly where the algebra predicts, with no separate attention-side factor.

// One causal self-attention pass's pre-softmax scores are `attn_scale * dot_pred(t,s)` with
// `dot_pred` built from the mscale=1 rope output via the mscale^2 formula; probabilities and the
// output are the ordinary softmax + weighted-V reduction over q_native/k_native's OWN geometry.
std::vector<double> predict_yarn_attention(const std::vector<std::uint16_t>& q_native,
                                           const std::vector<std::uint16_t>& k_native,
                                           const std::vector<std::uint16_t>& v_bits,
                                           const tgqa::Geometry& geometry, int tokens, double m2m1,
                                           float attn_scale) {
    std::vector<double> output(static_cast<std::size_t>(kHeadDim) *
                               static_cast<std::size_t>(geometry.q_heads) *
                               static_cast<std::size_t>(tokens));
    std::vector<double> scores(static_cast<std::size_t>(tokens));
    std::vector<double> probabilities(scores.size());
    const int group = geometry.q_heads / geometry.kv_heads;

    for (int t = 0; t < tokens; ++t) {
        const int visible = t + 1; // causal self-attention over token indices 0..tokens-1
        for (int head = 0; head < geometry.q_heads; ++head) {
            const int kv_head       = head / group;
            const std::size_t base_q = tgqa::q_index(geometry, head, 0, t);
            double max_score         = -std::numeric_limits<double>::infinity();
            for (int s = 0; s < visible; ++s) {
                const std::size_t base_k = tgqa::kv_input_index(geometry, kv_head, 0, s);
                const double native_dot =
                    bf16_dot_range(q_native, base_q, k_native, base_k, 0, kHeadDim);
                const double rotary_dot =
                    bf16_dot_range(q_native, base_q, k_native, base_k, 0, kRotary);
                const double score = static_cast<double>(attn_scale) *
                                     (native_dot + m2m1 * rotary_dot);
                scores[static_cast<std::size_t>(s)] = score;
                max_score                            = std::max(max_score, score);
            }
            double sum = 0.0;
            for (int s = 0; s < visible; ++s) {
                const double probability =
                    std::exp(scores[static_cast<std::size_t>(s)] - max_score);
                probabilities[static_cast<std::size_t>(s)] = probability;
                sum += probability;
            }
            for (int s = 0; s < visible; ++s) { probabilities[static_cast<std::size_t>(s)] /= sum; }

            for (int d = 0; d < kHeadDim; ++d) {
                double value = 0.0;
                for (int s = 0; s < visible; ++s) {
                    const std::size_t base_v = tgqa::kv_input_index(geometry, kv_head, d, s);
                    value += probabilities[static_cast<std::size_t>(s)] *
                            static_cast<double>(bf16_to_f32(v_bits[base_v]));
                }
                output[tgqa::q_index(geometry, head, d, t)] = value;
            }
        }
    }
    return output;
}

int run_attention_mscale_case() {
    const tgqa::Geometry geometry{"12|2", 12, 2}; // registered head-local geometry (24|4's tp2 half)
    constexpr std::int32_t kTokens      = 8;
    constexpr std::int32_t kMaxContext  = 64;
    const std::string label              = "attention mscale^2 12q/2k";

    // Q/K use the same well-conditioned magnitude gqa_attention_fixture.h's own cases do
    // (make_bf16_values(..., -0.25, 0.25)): with kHeadDim=256 dims, a wider range makes the raw
    // dot products (and hence the softmax) numerically saturated -- tiny, otherwise-honest BF16
    // rounding differences between the two independent rope evaluations would then get amplified
    // by an almost one-hot softmax, which is a conditioning artifact of the test setup, not a
    // defect in the mscale^2 prediction. V keeps the fixture's wider -1..1 range since it never
    // goes through rope or the softmax's exponential.
    const auto q_native = make_input_range(geometry.q_heads, kTokens, 0xA201U, -0.25F, 0.25F);
    const auto k_native = make_input_range(geometry.kv_heads, kTokens, 0xA202U, -0.25F, 0.25F);
    const auto v_float  = make_input_range(geometry.kv_heads, kTokens, 0xA203U, -1.0F, 1.0F);
    const auto q_bits   = to_bf16_bits(q_native);
    const auto k_bits   = to_bf16_bits(k_native);
    const auto v_bits   = to_bf16_bits(v_float);

    std::vector<int> positions(static_cast<std::size_t>(kTokens));
    for (std::int32_t t = 0; t < kTokens; ++t) {
        positions[static_cast<std::size_t>(t)] = t; // sequential, causal self-attention
    }

    Table native_table = yarn_table(1.0F);
    Table scaled_table  = native_table;
    scaled_table.mscale = yarn_table(4.0F).mscale;
    const double m       = static_cast<double>(scaled_table.mscale);
    const double m2m1    = m * m - 1.0;

    const Launch native_run = run_rope(positions, 1, kTokens, geometry.q_heads, geometry.kv_heads,
                                       q_bits, k_bits, &native_table, label + " native rope");
    const Launch scaled_run = run_rope(positions, 1, kTokens, geometry.q_heads, geometry.kv_heads,
                                       q_bits, k_bits, &scaled_table, label + " yarn rope");
    int failures            = native_run.guard_failures + scaled_run.guard_failures;

    // --- run the real Op (A1 append+attend) on the yarn-roped (mscale=M) q/k -------------------
    const tgqa::HostCache initial =
        tgqa::make_cache(geometry, DType::BF16, kMaxContext, 0x6D6E75u);
    tgqa::DeviceCache cache(initial, tgqa::MappingPattern::Identity);

    GuardedDeviceBuffer dq(scaled_run.q.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(scaled_run.k.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_row(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(scaled_run.q.size() * sizeof(std::uint16_t));
    dq.copy_from_host(scaled_run.q.data(), scaled_run.q.size() * sizeof(std::uint16_t));
    dk.copy_from_host(scaled_run.k.data(), scaled_run.k.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    const std::int32_t table_row = 0;
    dtable_row.copy_from_host(&table_row, sizeof(table_row));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, kTokens});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, kTokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, kTokens});
    Tensor tp(dp.data(), DType::I32, {kTokens});
    Tensor ttable_row(dtable_row.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, kTokens});

    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(kTokens),
                                             static_cast<std::uint32_t>(kMaxContext)};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, DType::BF16, envelope, 1, kTokens, kTokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention(tq, tk, tv, tp, Tensor{}, ttable_row, tgqa::kAttentionScale,
                      cache.batch_view(), envelope, workspace, tout, nullptr);
    cuda_synchronize();

    const auto actual_bits = tgqa::copy_from_guarded<std::uint16_t>(dout, scaled_run.q.size());
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    failures += cache.verify_guards(label);

    // --- oracle: predict the yarn-mode output from the mscale=1 rope output ------------------
    const std::vector<double> predicted =
        predict_yarn_attention(native_run.q, native_run.k, v_bits, geometry, kTokens, m2m1,
                               tgqa::kAttentionScale);
    const std::vector<double> actual = tgqa::bf16_bits_to_double(actual_bits);
    failures += tgqa::verify_attention(label + " vs mscale^2 prediction", actual, predicted,
                                       tgqa::attention_criterion(DType::BF16));

    // Negative control: an M==1 prediction (m2m1==0, i.e. plain native rope reused as the
    // "yarn" prediction) must be REJECTED against the actual yarn-mode Op output -- otherwise
    // this leg could not tell a correctly-scaled kernel from one that dropped mscale entirely.
    const std::vector<double> unscaled_prediction =
        predict_yarn_attention(native_run.q, native_run.k, v_bits, geometry, kTokens, 0.0,
                               tgqa::kAttentionScale);
    int control_probe = tgqa::verify_attention(label + " vs mscale=1 control (EXPECT mismatch)",
                                               actual, unscaled_prediction,
                                               tgqa::attention_criterion(DType::BF16));
    if (control_probe == 0) {
        std::cerr << label
                  << ": mscale=1 attention-level control was NOT rejected -- the check cannot "
                     "distinguish an unscaled kernel from a correctly-scaled one\n";
        ++failures;
    } else {
        std::cout << "control: " << label
                  << " mscale=1 attention-level prediction was correctly rejected (expected)\n";
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;

    // Leg 1 — exact native identity, across both position layouts, both whole-model geometries,
    // the pair and single-tensor forms, and the block-size branches (<=6 tokens, <=1020, more).
    failures += run_native_identity_case("native identity 24q/4k 1-D", 1, 6, 24, 4);
    failures += run_native_identity_case("native identity 24q/4k mrope", 3, 6, 24, 4);
    failures += run_native_identity_case("native identity 24q/4k prefill", 1, 512, 24, 4);
    failures += run_native_identity_case("native identity 16q/2k chunk", 1, 1100, 16, 2);
    failures += run_native_identity_case("native identity single 4k", 1, 6, 4, 0);
    failures += run_native_identity_case("native identity single 24q mrope", 3, 6, 24, 0);

    // Leg 2 — the factor-4 table and its mscale against the FP64 oracle.
    failures += run_yarn_case("yarn 24q/4k 1-D", 1, 6, 24, 4);
    failures += run_yarn_case("yarn 24q/4k mrope", 3, 6, 24, 4);
    failures += run_yarn_case("yarn 24q/4k prefill", 1, 512, 24, 4);
    failures += run_yarn_case("yarn single 4k", 1, 6, 4, 0);

    // Sensitivity control. The same kernel output, judged against an oracle that forgot to scale
    // cos and sin, must be REJECTED — otherwise leg 2 would pass with the mscale multiply deleted
    // from the kernel and would be proving nothing about it.
    if (run_yarn_case("mscale sensitivity control", 1, 6, 24, 4, /*drop_mscale=*/true) == 0) {
        std::cerr << "mscale sensitivity control: an oracle without mscale was accepted — the "
                     "YaRN profile is too loose to prove the kernel applies it\n";
        ++failures;
    }

    failures += run_rejection_case();

    // Leg 4 — one table buffer per device, read by that device's rope.
    failures += run_per_device_cases();

    // Leg (a) — dot-product level mscale^2 algebra: exact (zeroed pass-through) and
    // general cases, both against two independent rope Op outputs (mscale=1 and mscale=M,
    // same native table), with a negative control that an mscale=1 prediction is rejected.
    failures += run_mscale_dot_case("mscale^2 dot exact (zeroed pass-through)",
                                    /*zero_passthrough=*/true);
    failures += run_mscale_dot_case("mscale^2 dot general", /*zero_passthrough=*/false);

    // Leg (b) — attention-level: the same identity, checked through the real
    // ops::gqa_attention Op, with the same negative control.
    failures += run_attention_mscale_case();

    std::cout << (failures == 0 ? "OK" : "FAIL") << " rope yarn correctness\n";
    return failures == 0 ? 0 : 1;
}
