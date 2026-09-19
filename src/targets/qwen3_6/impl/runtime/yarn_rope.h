#pragma once

// Host-side YaRN (Yet another RoPE extensioN) frequency correction, computed once at load time
// (never on the hot path). Consumed by the rope kernel wiring, which places
// `yarn_scale(...).first` into the device buffer the kernels in src/ops/kernel/rope.cuh read when
// `rope_mode == Yarn`, replacing the fixed `kTextRopeInvFrequency[32]` table, and
// `yarn_rope_mscale(...)` into `RopeFrequencyOverride::mscale` (see include/ninfer/ops/rope.h).
// docs/maintainer/tp2-yarn-1m.md is the maintainer-level account of the same math, including the
// (16, 24) correction range this file reproduces and why mscale lives inside rope.
//
// DESIGN NOTE -- mscale is a rope-path effect only: there is NO attention-level scale change.
// mscale is applied ENTIRELY inside the rope path above (to both cos and sin, i.e. to the whole
// rotated q/k output, not to the phase) -- see `RopeFrequencyOverride::mscale` and its kernel
// wiring. The attention softmax `scale` argument (ninfer::targets::qwen3_6::kAttentionScale /
// `Variant::attention_scale`, a compile-time constant derived from rsqrt(head_dim), e.g. 0.0625
// for the 27B head_dim=256 variant) is computed identically in native and yarn mode and is never
// touched by anything in this file:
// `src/ops/wrapper/gqa_attention.cpp`'s `validate_attention_tensors` hard-rejects any `scale`
// that is not exactly rsqrt(head_dim) (`kExpectedScale`, tolerance 1e-6), which by itself rules
// out a runtime attention-side factor; `src/ops/kernel/gqa_attention_prefill_bf16.cuh` folds
// `scale` into `exp2` (`scale_l2 = scale * Log2E`) and the decode launchers
// (`src/ops/launcher/gqa_attention_decode.cu`) pass it straight through as `float scale` -- in
// neither place does yarn mode branch. `attention_factor_a` (this header's `yarn_scale()` second
// return value, == `1/mscale^2`) is retained ONLY as a diagnostic/test value (it mirrors vLLM's
// `1/mscale^2` bookkeeping and is asserted in `tests/targets/qwen3_6/test_yarn_rope.cpp` and
// dumped by `tools/tp2/dump_yarn_ref.py`); no production code path reads it. What substitutes
// for an attention-side factor is checked in `tests/ops/test_rope_yarn.cpp`: with the SAME
// (native) inverse-frequency table, mscale 1->M multiplies the ROTATED OUTPUT of q and k by M,
// so `q'_M . k'_M == q'_1 . k'_1 + (M^2-1) * (q'_1_rot . k'_1_rot)` -- checked both at the
// dot-product level and through the real `ops::gqa_attention` Op.
//
// Reference: vLLM's YaRN implementation is the numerical reference, specifically
// `vllm/model_executor/layers/rotary_embedding/{common.py,yarn_scaling_rope.py,mrope.py}` in the
// serve environment (/home/pc/Projects/vllm/unsloth-nvfp4-env/lib/python3.13/site-packages/vllm).
// The committed dump `tools/tp2/dump_yarn_ref.py` reproduces the exact values below by importing
// and driving those vLLM classes directly (not a reimplementation) against the production Qwen3
// rope_parameters override (serve-orca-qwen38-27b-long.sh line 35); its output is
// `tests/core/data/yarn_ref_4x.json`, which `tests/targets/qwen3_6/test_yarn_rope.cpp` checks this
// implementation against to <=1e-6 relative (max observed error was of order 1e-7, i.e. float32
// rounding noise).
//
// IMPORTANT traced finding (do not "obviously" reimplement the textbook YaRN formula without
// this): Qwen3's `rope_parameters` always carries `mrope_section` (even for pure text), so
// vLLM's `get_rope()` dispatches the `rope_type == "yarn"` branch to `MRotaryEmbedding`, NOT the
// plain `YaRNScalingRotaryEmbedding` class. `MRotaryEmbedding.__init__` unconditionally enlarges
// the position count used to build the rope module (a Qwen2.5-VL video-position headroom quirk,
// unrelated to the YaRN `factor`):
//     self.cache_max_position_num = max_position_embeddings * 4
//     super().__init__(head_size, rotary_dim, self.cache_max_position_num, base, ...)
// `get_rope()`'s yarn branch passes `original_max_position_embeddings` as
// `max_position_embeddings`, so this sets the base class's `self.max_position_embeddings` (what
// `yarn_find_correction_range` actually reads) to `original_max * 4`, NOT `original_max` as one
// would naively assume. For the production config (original_max=262144) this shifts the
// correction range from (14, 22) [naive, using 262144 directly] to (16, 24) [what vLLM's serve
// process actually computes]. `yarn_scale()` below reproduces the *production* (16, 24)
// behavior: the correction range is computed against `original_max * 4`. Everything else in the
// inv_freq/mscale formula is identical between `MRotaryEmbedding` and the plain YaRN class for
// this config (`MRotaryEmbedding._compute_inv_freq`/`._compute_cos_sin_cache` literally delegate
// to `YaRNScalingRotaryEmbedding`'s methods bound to the `MRotaryEmbedding` instance); the
// mrope_section/mrope_interleaved fields only affect how cos/sin are assembled across T/H/W
// position axes for 2-D (multimodal) positions and never run for text mode's 1-D positions.
//
// --- Formula (vLLM `common.py` + `yarn_scaling_rope.py`, all in double precision here; vLLM
// itself computes in float32 -- the <=1e-6 relative tolerance absorbs that difference) ---
//
// Given `rotary_dim = 2 * rotary_pairs`, `theta`, YaRN `factor` F, `original_max` P,
// `beta_fast`/`beta_slow`:
//
//   1. Per-pair native ("extrapolation") and interpolated frequencies, i = 0 .. rotary_pairs-1:
//        pos_freq[i]      = theta ^ (2*i / rotary_dim)
//        extrapolation[i] = 1 / pos_freq[i]                    // == native inv_freq[i]
//        interpolation[i] = 1 / (F * pos_freq[i])               // == extrapolation[i] / F
//
//   2. Correction range (`yarn_find_correction_dim` / `yarn_find_correction_range`):
//        dim(num_rotations) = rotary_dim * ln(effective_max / (num_rotations * 2*pi))
//                             / (2 * ln(theta))
//        low  = clamp(floor(dim(beta_fast)), 0, rotary_dim - 1)
//        high = clamp(ceil (dim(beta_slow)), 0, rotary_dim - 1)
//        effective_max = original_max * 4   // the MRotaryEmbedding video-cache quirk, see above
//        if low == high: high += 0.001      // vLLM's singularity guard
//      `low`/`high` are dimension-scale values (0..rotary_dim-1) but only ever land in
//      [0, rotary_pairs-1] for realistic configs, since they index into the rotary_pairs-length
//      per-pair arrays below.
//
//   3. Linear ramp mask (`yarn_linear_ramp_mask`), i = 0 .. rotary_pairs-1:
//        ramp[i] = clamp((i - low) / (high - low), 0, 1)
//        mask[i] = 1 - ramp[i]              // extrapolation_factor pinned at 1 (not overridden)
//      mask[i] == 1 (pure extrapolation/native) for i <= low (short wavelength, high frequency
//      -- no correction needed); mask[i] == 0 (pure interpolation, i.e. native/F) for i >= high
//      (long wavelength, low frequency -- needs interpolation to stay in-distribution); linear
//      blend for low < i < high.
//
//   4. Corrected inverse frequency:
//        inv_freq[i] = interpolation[i] * (1 - mask[i]) + extrapolation[i] * mask[i]
//      At F == 1 (native/no-yarn), interpolation[i] == extrapolation[i] for all i, so
//      inv_freq[i] == extrapolation[i] == native inv_freq[i] regardless of the mask -- this is
//      the "native config" structural guarantee the test checks.
//
//   5. mscale (`yarn_get_mscale`, `YaRNScalingRotaryEmbedding.mscale`):
//        mscale = (F <= 1) ? 1.0 : 0.1 * ln(F) + 1.0            // yarn_get_mscale(F)
//        mscale *= attn_factor                                   // attn_factor pinned at 1
//      vLLM scales BOTH the cos and sin cache by `mscale` before rotating q and k identically
//      (`cos = freqs.cos() * mscale; sin = freqs.sin() * mscale`), so the rotary-subspace
//      contribution to `q . k` picks up one factor of `mscale` from each of q_rot and k_rot,
//      i.e. `mscale**2` relative to a rotation using unscaled cos/sin. This is the ENTIRE
//      YaRN "attention temperature" effect: it lives inside the rope path
//      (`RopeFrequencyOverride::mscale`), not as a separate attention-side factor -- see the
//      design note above. `attention_factor_a = 1/mscale^2` below is kept only as a diagnostic
//      (it mirrors vLLM's own `1/mscale^2` bookkeeping) and a test/reference value; no
//      production code multiplies anything by it. At F == 1, mscale == 1 and
//      attention_factor_a == 1 (no-op), matching native behavior exactly.
//
//      Note vLLM applies `mscale` only to the first `rotary_dim` head components (the rotated
//      ones); the remaining `head_dim - rotary_dim` "pass-through" components are untouched --
//      exactly what `RopeFrequencyOverride::mscale`'s kernel wiring reproduces. So the effect
//      on the *full* head_dim q.k dot product is NOT a uniform mscale^2 factor; it is
//      `native_dot + (mscale^2 - 1) * rotary_dot`, checked directly in
//      `tests/ops/test_rope_yarn.cpp` (dot-product and real-attention-Op legs).
//      docs/maintainer/tp2-yarn-1m.md carries the full derivation.

#include "ninfer/types.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

struct YarnParams {
    float factor       = 4.0f;    // --yarn-factor; 1.0 == native/no-op
    int original_max    = 262144;  // --yarn-origin; must equal the artifact's registered native max
    float theta         = 1e7f;    // from the artifact's TextConfig (rope_theta)
    int rotary_pairs    = 32;      // = rotary_dim / 2 = (head_dim * partial_rotary_factor) / 2
    float beta_fast     = 32.0f;   // YaRN default (vLLM reference; not a CLI flag)
    float beta_slow     = 1.0f;    // YaRN default (vLLM reference; not a CLI flag)
};

// Returns { yarn-corrected inverse frequencies (length == p.rotary_pairs), attention_factor_a ==
// 1/mscale^2 }. The second member is a DIAGNOSTIC/TEST value only (see the design note at the top
// of this file): no production code multiplies attention scores by it or its reciprocal. A == 1
// for native/factor==1.
//
// Throws std::invalid_argument if `p.original_max != 262144` (origins other than the artifact's
// registered 262144 are rejected) or if `p.factor * p.original_max > 1,048,576` (the product's
// max context ceiling).
std::pair<std::vector<float>, float> yarn_scale(const YarnParams& p);

// The rotary cos/sin multiplier itself, i.e. vLLM's
// `YaRNScalingRotaryEmbedding.mscale = yarn_get_mscale(factor) * attn_factor` (attn_factor pinned
// at 1). This is the number `_compute_cos_sin_cache` multiplies BOTH the cos and the sin cache by
// before rotating q and k, so it belongs inside the rope path, not in the attention scale: the
// rope kernels' buffer mode consumes it directly (`RopeFrequencyOverride::mscale`) alongside
// `yarn_scale(...).first`. It is exposed separately because `yarn_scale`'s second member is the
// derived attention divisor A == 1 / mscale^2, from which recovering mscale would mean an
// avoidable round trip through a square root. Returns exactly 1.0 for factor <= 1 (native),
// 1.138629436111989 for factor == 4.
//
// Pure function of `p.factor`; the artifact/context-ceiling validation lives in `yarn_scale`.
float yarn_rope_mscale(const YarnParams& p);

// --- Rope-option admission and the effective context ceiling -----------------------------------
//
// One variant's rope domain, as the family runtime sees it. `native_max` is
// `Variant::maximum_context` (the artifact's registered native position capacity) and
// `supports_yarn` is `Variant::supports_yarn_rope` -- false for a variant whose text attention is
// not the D256/R64 partial-rotary geometry the YaRN table is computed for (35B-A3B: D128/R128 with
// its own DFlash frequency tables, and no published YaRN rope_parameters for that checkpoint).
struct RopeDomain {
    std::uint32_t native_max = 262144;
    bool supports_yarn       = false;
};

// Validates `options`' rope fields against `domain` and returns the EFFECTIVE context ceiling:
//
//   RopeMode::Native -> domain.native_max
//   RopeMode::Yarn   -> yarn_origin * yarn_factor
//
// This value, not `Variant::maximum_context`, is what `options.max_context` must fit inside; it is
// the single definition every context-derived extent in the sequence plan follows (page tables,
// KV-plan capacity, graph profile boundaries and the request-memory reservation all size from
// `options.max_context` and inherit the ceiling through it).
//
// Throws std::invalid_argument with an actionable message when:
//   - max_context is 0 or exceeds the effective ceiling;
//   - yarn is requested on a variant whose rope domain does not carry a YaRN table;
//   - yarn is requested together with Vision (the vision encoder ropes 2-D image-grid positions
//     through an 18-entry D72/R72 table that the 32-entry text YaRN buffer does not describe;
//     the Op wrapper rejects the override for that domain outright);
//   - yarn is requested together with the DFlash speculative backend (D128/R128 double table);
//   - yarn_origin != domain.native_max, or yarn_factor is not finite/positive, or
//     yarn_origin * yarn_factor exceeds kMaximumYarnContext.
//
// Native mode ignores yarn_factor/yarn_origin entirely: leaving them at their defaults or setting
// them to anything else cannot change a native run.
[[nodiscard]] std::uint32_t rope_effective_max_context(const EngineOptions& options,
                                                       const RopeDomain& domain);

}  // namespace ninfer::targets::qwen3_6::detail
