#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Replaces the theta-derived rotary frequencies of the 1-D/MRoPE Text mode with an externally
 * computed table, and scales the rotation coefficients uniformly.
 *
 * `inv_frequency` is a device buffer of rotary_dim/2 float32 inverse frequencies replacing
 * `theta^(-2*i/rotary_dim)` for pair i; `theta` is then unused. `mscale` multiplies both the
 * cosine and the sine of every pair, so within one head the rotary subspace is rotated *and*
 * scaled by `mscale` while dimensions [rotary_dim,head_dim) stay bit-exact:
 *
 *   out[i]        = mscale * (x[i] * cos(phi) - x[i+R/2] * sin(phi))
 *   out[i+R/2]    = mscale * (x[i+R/2] * cos(phi) + x[i] * sin(phi)),  phi = p(t) * f[i].
 *
 * A default-constructed override (`inv_frequency == nullptr`) selects the ordinary theta-derived
 * behavior documented below, bit-for-bit; `mscale` is then ignored. A non-null override is
 * accepted only for the Text D256/R64 domain with 1-D or 3-D positions, requires a finite
 * positive `mscale`, and requires 4-byte-aligned Q/K storage; anything else throws. The buffer
 * must stay resident and at a fixed address for as long as the launch (or the CUDA graph that
 * captured it) can run, and holds the same values for q and k.
 *
 * This is the YaRN path: `inv_frequency` carries the interpolation/extrapolation-blended
 * frequencies and `mscale` the attention temperature vLLM folds into its cos/sin cache.
 *
 * `mscale` is the ONLY yarn-mode scaling factor anywhere in the attention pipeline: there is no
 * separate attention-softmax scale for yarn mode. `ops::gqa_attention`/`gqa_attention_cached`'s
 * `scale` argument (rsqrt(head_dim), a compile-time per-variant constant validated exactly by
 * `src/ops/wrapper/gqa_attention.cpp`) is computed and applied identically whether or not this
 * override is in effect. See `src/targets/qwen3_6/impl/runtime/yarn_rope.h` for the full account
 * of why (mscale multiplies the rotated q/k here, so `q'.k'` already carries
 * `mscale^2` on its rotary-subspace contribution without any attention-side change) and
 * `tests/ops/test_rope_yarn.cpp` for the verification.
 */
struct RopeFrequencyOverride {
    const float* inv_frequency = nullptr;
    float mscale               = 1.0F;
};

/**
 * Applies split-half NeoX RoPE in place. For pair i in [0,rotary_dim/2), angle phi(i,t), and
 * each head:
 *
 *   ideal[i]              = x[i] * cos(phi) - x[i+R/2] * sin(phi)
 *   ideal[i+rotary_dim/2] = x[i+R/2] * cos(phi) + x[i] * sin(phi).
 *
 * Dimensions [rotary_dim,head_dim) are unchanged. Supported modes are:
 *
 * - Text 1-D: positions I32 [T], either head_dim=256 with even 0<rotary_dim<=256, or the
 *   DFlash full-head domain head_dim=rotary_dim=128; phi=positions[t]*theta^(-2*i/rotary_dim).
 * - Text MRoPE: positions I32 [T,3], head_dim=256, rotary_dim=64; pair i uses axis i%3 with
 *   the same frequency as Text 1-D.
 * - Vision 2-D: positions I32 [T,2], head_dim=rotary_dim=72; pairs 0..17 use axis 0 and pairs
 *   18..35 use axis 1, each with local frequency theta^(-2*(i%18)/36).
 *
 * positions is contiguous and theta is positive and finite. Q/K tensors are BF16
 * [head_dim,heads,T] with positive head counts, contiguous head features and heads, and an optional
 * padded token stride. The registered optimized domains are D256/R64 Text Q/K head geometries
 * 24/4 and 16/2, D128/R128 1-D Text geometry 32/8, plus Vision geometry 16/16. q and k must not
 * overlap one another or positions. The Op mutates only dimensions [0,rotary_dim) of the supplied
 * Q/K tensor storage. The oracle evaluates the rotated dimensions naively in FP64 from the
 * represented inputs. The updated BF16 values are promoted and compared directly with that result;
 * output storage rounding belongs to the Op's numerical criterion, not the oracle. Unrotated
 * dimensions remain bit-exact. Private kernel arithmetic is implementation-defined. The Op uses no
 * workspace or persistent state.
 */
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          cudaStream_t stream);

// Single-tensor form with the same formula and storage contract. The head count comes directly
// from x; Q versus K role does not change the transformation.
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x, cudaStream_t stream);

// Frequency-override forms. Identical to the two above when `frequency` is default-constructed;
// see RopeFrequencyOverride for the substituted formula and its admitted domain.
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          const RopeFrequencyOverride& frequency, cudaStream_t stream);

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
          const RopeFrequencyOverride& frequency, cudaStream_t stream);

} // namespace ninfer::ops
