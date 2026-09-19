#!/usr/bin/env python3
"""Dump the vLLM-reference YaRN inverse-frequency table and attention soft-cap factor for the
production Qwen3.8-27B rope configuration.

This script imports vLLM's *actual* rope classes/functions from the serve environment (vLLM is
not installed in ninfer's own Python env) and instantiates the rope module exactly the way
`vllm/model_executor/models/qwen3.py::Qwen3Attention` does in production: it calls
`get_rope(head_size=config.head_dim, rope_parameters=config.rope_parameters, ...)` where
`rope_parameters` is the `--hf-overrides text_config.rope_parameters` dict below (verbatim from
`serve-orca-qwen38-27b-long.sh` line 35).

Traced call path:
  Qwen3Attention.__init__ -> get_rope(head_size=256, rope_parameters=<dict below>)
  -> get_rope() sees rope_type=="yarn" AND "mrope_section" in rope_parameters
  -> dispatches to `MRotaryEmbedding` (NOT the plain `YaRNScalingRotaryEmbedding`!)
  -> `MRotaryEmbedding._compute_inv_freq` / `._compute_cos_sin_cache` delegate to
     `YaRNScalingRotaryEmbedding`'s methods bound to the MRotaryEmbedding instance -- so the
     underlying inv_freq/mscale *formula* is identical to the plain YaRN class.

CRITICAL finding (do not assume -- verified by direct execution, see debug traces in the task
report): `MRotaryEmbedding.__init__` unconditionally enlarges the cache position count for
Qwen2.5-VL-style video inputs:

    self.cache_max_position_num = max_position_embeddings * 4   # mrope.py, fixed x4, not the
                                                                  # YaRN `factor`
    super().__init__(head_size, rotary_dim, self.cache_max_position_num, ...)

`get_rope()`'s yarn branch passes `original_max_position_embeddings` (262144) as
`max_position_embeddings` into `MRotaryEmbedding.__init__`, so this line sets
`self.max_position_embeddings = 262144 * 4 = 1,048,576` (a base-class attribute, set via the
`super().__init__` call). `yarn_find_correction_range` (called from
`YaRNScalingRotaryEmbedding._compute_inv_freq`, which `MRotaryEmbedding` delegates to) reads
`self.max_position_embeddings`, so the production correction range is computed against
1,048,576, NOT the raw 262,144 one would naively assume from the artifact's registered native
max. This is an incidental quirk of `MRotaryEmbedding` (written for Qwen2.5-VL video position
headroom) that Qwen3's *text* rope inherits only because `get_rope()` routes any
"yarn" + "mrope_section" config through `MRotaryEmbedding` -- there is no video content in play
here. It shifts the correction range from (14, 22) [what you get from
`yarn_find_correction_range(..., max_position_embeddings=262144)`, e.g. as computed against the
plain YaRN class in isolation] to (16, 24) [what the actual deployed rope module uses]. This
script proves both numbers and dumps the *production-accurate* (16, 24) table, since that is
what vLLM's serve process literally computes.

The mrope wrapper's mrope_section/mrope_interleaved fields do NOT otherwise affect inv_freq or
mscale -- they only affect how cos/sin get selected across T/H/W position axes for multimodal
(2-D) positions; text mode uses 1-D positions and never touches that code path.

mscale enters attention via `YaRNScalingRotaryEmbedding._compute_cos_sin_cache`, which scales
BOTH the cos and sin cache by `self.mscale = yarn_get_mscale(factor) * attn_factor` before they
are used to rotate q and k identically (`ApplyRotaryEmb.forward_static`: q' = q*cos - ~q*sin,
etc.). Since both q_rot and k_rot pick up one factor of mscale from the (cos, sin) scaling, the
rotary-subspace contribution to `q . k` is scaled by `mscale**2` relative to a rotation that used
unscaled cos/sin. `attn_factor` is not overridden by the production config (no `attn_factor` key
in the override dict), so it keeps vLLM's default of 1, and `mscale == yarn_get_mscale(factor)`.

Run with the *serve* env's python (vLLM is not importable from ninfer's own env):
    /home/pc/Projects/vllm/unsloth-nvfp4-env/bin/python tools/tp2/dump_yarn_ref.py \
        > tests/core/data/yarn_ref_4x.json
"""

from __future__ import annotations

import json
import sys

VLLM_SITE_PACKAGES = (
    "/home/pc/Projects/vllm/unsloth-nvfp4-env/lib/python3.13/site-packages"
)
if VLLM_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, VLLM_SITE_PACKAGES)

# Drift guard: the C++ ctest (tests/targets/qwen3_6/test_yarn_rope_drift.cpp) runs this script
# unmodified in the serve env and expects "vLLM (or a dependency) is not importable here" to be a
# clean, distinguishable SKIP (ctest code 77), never conflated with a real failure. Do not add a
# second script for this -- this is the one script the drift guard reuses, so the skip signal
# lives here.
try:
    import torch  # noqa: E402
    import vllm  # noqa: E402

    from vllm.config.vllm import VllmConfig, set_current_vllm_config  # noqa: E402
    from vllm.model_executor.layers.rotary_embedding import get_rope  # noqa: E402
    from vllm.model_executor.layers.rotary_embedding.common import (  # noqa: E402
        yarn_find_correction_range,
        yarn_get_mscale,
    )
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding  # noqa: E402
except ImportError as exc:  # pragma: no cover - exercised only when vLLM is unavailable
    print(f"SKIP: vLLM (or a dependency) is not importable in this Python: {exc}", file=sys.stderr)
    sys.exit(77)

# Exact production override, verbatim from serve-orca-qwen38-27b-long.sh line 35
# (--hf-overrides '{"text_config": {"rope_parameters": {...}}}').
HEAD_DIM = 256
ROTARY_DIM = 64  # head_dim * partial_rotary_factor = 256 * 0.25
ROTARY_PAIRS = ROTARY_DIM // 2  # 32
ROPE_THETA = 10_000_000
FACTOR = 4.0
ORIGINAL_MAX_POSITION_EMBEDDINGS = 262144
BETA_FAST = 32
BETA_SLOW = 1

ROPE_PARAMETERS_PRODUCTION = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "rope_type": "yarn",
    "rope_theta": ROPE_THETA,
    "partial_rotary_factor": 0.25,
    "factor": FACTOR,
    "original_max_position_embeddings": ORIGINAL_MAX_POSITION_EMBEDDINGS,
}


def main() -> None:
    torch.set_default_dtype(torch.float32)

    # Build the rope module exactly as Qwen3Attention does in production.
    with set_current_vllm_config(VllmConfig()):
        rope_mod = get_rope(
            head_size=HEAD_DIM,
            max_position=1_048_576,
            is_neox_style=True,
            rope_parameters=ROPE_PARAMETERS_PRODUCTION,
        )
    assert isinstance(rope_mod, MRotaryEmbedding), (
        f"expected get_rope() to dispatch yarn+mrope_section to MRotaryEmbedding, got "
        f"{type(rope_mod).__name__}"
    )
    assert rope_mod.rotary_dim == ROTARY_DIM
    assert rope_mod.scaling_factor == FACTOR

    # `_compute_cos_sin_cache` (the only thing vLLM's forward path actually calls) invokes
    # `self._compute_inv_freq(self.scaling_factor)` -- reproduce that call exactly (NOT
    # `self.base`; `YaRNScalingRotaryEmbedding._compute_inv_freq`'s parameter is literally named
    # `scaling_factor` and is used as the interpolation-branch scale).
    inv_freq_production = rope_mod._compute_inv_freq(rope_mod.scaling_factor).double().tolist()
    mscale = rope_mod.mscale
    attn_factor = rope_mod.attn_factor

    # The x4 video-cache quirk: rope_mod.max_position_embeddings is the *effective* value
    # (original_max * 4) that MRotaryEmbedding.__init__ passed to the RotaryEmbeddingBase
    # constructor, and yarn_find_correction_range uses this effective value.
    effective_max_position = rope_mod.max_position_embeddings
    assert effective_max_position == ORIGINAL_MAX_POSITION_EMBEDDINGS * 4

    low_production, high_production = yarn_find_correction_range(
        BETA_FAST, BETA_SLOW, ROTARY_DIM, float(ROPE_THETA), effective_max_position
    )
    # For the record: what you'd get if you (incorrectly) used the raw original_max directly,
    # ignoring MRotaryEmbedding's x4 video-cache enlargement.
    low_naive, high_naive = yarn_find_correction_range(
        BETA_FAST, BETA_SLOW, ROTARY_DIM, float(ROPE_THETA), ORIGINAL_MAX_POSITION_EMBEDDINGS
    )

    # Native (theta^(-2i/d)) table for the same theta/rotary_dim -- what `factor == 1` yarn
    # collapses to, and the basis for the structural checks in the C++ test.
    native_inv_freq = (
        1.0
        / (
            float(ROPE_THETA)
            ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float64) / ROTARY_DIM)
        )
    ).tolist()

    mscale_factor_only = yarn_get_mscale(FACTOR)
    # vLLM scales both cos and sin (used identically for q and k) by `mscale`, so the
    # rotary-subspace contribution to q.k is scaled by mscale**2 relative to an unscaled
    # rotation. By the interface convention -- scores multiplied by 1/A, A = 1 for native --
    # A = 1 / mscale**2.
    effective_attention_multiplier = mscale**2
    attention_factor_a = 1.0 / effective_attention_multiplier

    out = {
        "_comment": (
            "vLLM-reference YaRN dump for ninfer's host-side yarn_scale(). "
            "Reproduce with: unsloth-nvfp4-env/bin/python tools/tp2/dump_yarn_ref.py"
        ),
        "vllm_version": vllm.__version__,
        "generator_invocation": (
            "VLLM_LOGGING_LEVEL=WARNING "
            "/home/pc/Projects/vllm/unsloth-nvfp4-env/bin/python tools/tp2/dump_yarn_ref.py "
            "> tests/core/data/yarn_ref_4x.json"
        ),
        "config": ROPE_PARAMETERS_PRODUCTION,
        "head_dim": HEAD_DIM,
        "rotary_dim": ROTARY_DIM,
        "rotary_pairs": ROTARY_PAIRS,
        "beta_fast": BETA_FAST,
        "beta_slow": BETA_SLOW,
        "class_used_by_get_rope": type(rope_mod).__name__,
        "mrope_wrapper_changes_inv_freq_formula": False,
        "mrope_wrapper_changes_effective_max_position_via_x4_video_cache_quirk": True,
        "effective_max_position_embeddings_used_for_correction_range": effective_max_position,
        "correction_range_production": [low_production, high_production],
        "correction_range_naive_no_x4_quirk": [low_naive, high_naive],
        "attn_factor_param_default": attn_factor,
        "yarn_get_mscale_factor_only": mscale_factor_only,
        "mscale_effective": mscale,
        "effective_attention_multiplier_mscale_sq": effective_attention_multiplier,
        "attention_factor_A": attention_factor_a,
        "native_inv_freq": native_inv_freq,
        "yarn_inv_freq": inv_freq_production,
    }

    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
