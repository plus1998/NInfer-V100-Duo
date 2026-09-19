# Dual-GPU (TP2) execution and YaRN 1M context

This document records the design decisions, numerical contracts, and qualification evidence behind
two additions to the 27B execution package: tensor-parallel execution across two RTX 5090s
(`--tp 2 --devices A,B`) and YaRN positional scaling (`--rope yarn`) for contexts up to 1,048,576
tokens.

It is a maintainer reference, not a user guide. User-facing option contracts live in
[CLI](../cli.md) and [HTTP serving](../serving.md); the measured throughput, memory and retrieval
tables live in [Performance](../performance.md) and the project [README](../../README.md). What is
here is the material a maintainer needs before changing any of it: why the transport is shaped the
way it is, why the YaRN constants are what they are, and what each correctness gate actually
proves.

Scope: the 27B execution package (`qwen3.6-27b` and `qwen3.8-27b`, either weight profile).
`qwen3.6-35b-a3b` has no tensor-parallel path. Every measurement quoted here was taken on the
Qwen3.8-27B NVFP4 artifact, and every timing carries a per-GPU power limit — see
[Measurement conditions](#7-measurement-conditions-and-open-items).

---

## 1. Execution model

One process, one resident model, two CUDA devices, no NVLink and no distributed serving. TP2 is a
**capacity** feature first: it halves per-card weight and KV residency, which is what makes a
1,048,576-token window reachable at all. That it is also faster at long context is a consequence of
halving per-card weight and KV traffic, not the goal.

Rank 0 is the primary device. It owns tokenization, sampling, the retained MTP target hidden state,
and the request-visible state; rank 1 executes its half of every sharded Op and holds no
request-visible state of its own.

Prefix reuse supports suffix prefill and exact hits, including MTP. Rank 0 remains the authority
for retained target hidden and checkpoint hidden; resume stages the selected hidden into rank 1's
existing prefill buffer with stream ordering before either rank consumes it. Each rank captures
and restores its own sharded GDN checkpoint. Exact hits run the two vocabulary shards and gather
the target logits; the MTP bridge then uses the same two-rank MTP schedule as ordinary prefill.
This does not add a second persistent hidden ledger or change the decode hot path. Reusable
positions remain the resident frontier and complete typed checkpoints, as specified in
[prefix reuse](concurrent-inference-architecture.md#64-prefix-reuse).

---

## 2. Transport: no peer-to-peer on these cards

`cudaDeviceCanAccessPeer` reports **0 in both directions** between two RTX 5090s (GeForce driver
restriction; PCIe topology PHB). This is a measured property of the hardware, not a configuration
choice, and it is the single fact that shapes the collective layer.

The collectives are therefore **host-staged asynchronous copies over PCIe**, measured at
22.55/23.16 GiB/s bulk and 8.50/8.73 µs per 10 KiB transfer. `cudaMemcpyPeerAsync` is the same API
with or without peer access, so the absence of peer access changes the cost, not the code shape.

`allreduce_sum` / `allgather` live in `include/ninfer/ops/allreduce.h` and
`src/ops/common/allreduce.cu`. The design is **pull-based with four events per call**. A two-event
version was tried first and rejected: it carried a cross-call write-after-read hazard, where a
second call could overwrite a staging buffer a previous call's peer read had not yet consumed.

Measured costs:

| Quantity | Value |
|---|---|
| 10 KiB `allreduce_sum` | ~16 µs |
| Collectives per decode token | 128 reduces plus one logit all-gather |
| Whole collective set per token, under CUDA Graphs | ~0.2 ms |

(Both at the 400 W per-GPU cap.)

**`cudaMemcpyPeerAsync` is not stream-capturable.** The collective uses `cudaMemcpyAsync` in
device-to-device/UVA form instead, which is capturable and is what allows the whole dual-device
decode step to live inside one graph.

---

## 3. One cross-device CUDA graph

The decode step for `--tp 2` is captured as a **single fork/join graph spanning both devices**, not
as two per-device graphs joined by cross-device event edges. Two live captures cannot be linked:
attempting it fails with `cudaErrorStreamCaptureMerge`. There is exactly one graph mechanism in the
tree — the tp1 and tp2 paths share it — and no second transport is grown inside a Program.

| Quantity | Value |
|---|---|
| tp2 decode graph nodes | 1888 |
| tp1 decode graph nodes | 640 |
| Decode gain from capture | +15% |
| Graph residency at 1M | 2.00–3.44 MiB per device (20.00 MiB allowance) |

That residency is not deterministic, and the allowance has been observed to be too tight: one
1,048,576-token boot aborted with `CUDA Graph preparation consumed 44433408 bytes on device 0,
exceeding the planned per-device allowance of 20971520 bytes` where the identical command line
had succeeded minutes earlier and succeeded again on the immediate retry. See §7.1.

The captured graph's token output is **exactly identical** to the eager dual-device path; the eager
path remains as a fallback and as the identity reference for capture.

---

## 4. Shard map and per-device geometry

`plan_for(object, tp, config)` is a pure host computation over the target's compile-time
dimensions — no artifact, no device, no kernel. `tests/targets/test_shard_map.cpp` tables it
against the real object shapes and orders bound in
`src/targets/qwen3_6_27b/impl/load/bindings.cpp`.

**Fused row orders are the binder's, not the obvious ones.** Three of them differ from what a
naive reading of the model definition suggests, and every split form depends on getting them right:

- attention input projection: **Q | K | Gate | V**;
- GDN input projection: **Q(2048) | K(2048) | V(6144) | Z(6144)**;
- the GDN `out_proj` contraction dimension is **6144**.

**No kernel was rewritten for TP2.** The pattern throughout is that a shard is a standalone
shard-shaped tensor pushed through the existing kernel, plus a registry entry and a thin per-rank
wrapper (`src/ops/common/split_launch.h`). Where a kernel family is templated on a Geometry, the
shard is a second instantiation at the halved extent; where a family is runtime-dimensioned, the
same instantiation already serves the shard.

**GDN core: a head split needs no kernel change.** In the GDN chunked kernels every stage is one
CTA per value head with no cross-head reduction, so splitting heads across devices is a pure
launch-geometry change. The real geometry is **H_qk = 16, H_v = 48, G = 3**.

**NVFP4 row slicing is safe by construction.** In `blockscale-k16-m128x4-v1` the 128-row tile is
the outermost axis, so a shard's row range is contiguous in the stored layout and can be bound
directly without repacking.

**Full-attention geometry at tp2** is 12 Q heads and 2 KV heads per device (`Gqa27Tp2Geometry`),
head-local by construction: a device's head indices are its own, and nothing downstream renumbers
them.

### 4.1 Memory model

Per device, Qwen3.8-27B NVFP4, `--kv-dtype int8 --kv-capacity auto`:

- **Weights: 10.08 GiB** with MTP off, 10.46 GiB with MTP3. The replicated token embedding is part
  of this and is easy to omit from an estimate.
- **KV: 16.5 KiB per token per device.** 16 full-attention layers × 2 KV heads/device × 256
  head_dim × 2 (K+V) × 1 byte = 16 KiB/token, plus 0.5 KiB of FP16 group-64 quantization-scale
  planes. Verified against the allocator rather than only derived: at 8,192 tokens the pool is
  132.00 MiB and at 32,768 it is 528.00 MiB, both exactly 16,896 bytes per token, so at 1,048,576
  tokens the pool is **16.50 GiB per device**.
- **MTP surcharge is not a constant.** It is a measured +1.49 GiB of reserved memory per device at
  1M and +0.65 GiB at 262k. The two dominant terms are **0.38 GiB of head weights**, fixed at any
  window, and **1.03 GiB of MTP KV per 1M tokens of window**; the remainder of each measured delta
  is workspace and sequence-arena rounding. The KV term is one full extra attention layer's worth
  of KV over the window — one sixteenth of the text pool — which is what a one-layer MTP head
  costs.
- **Reserved versus resident.** `nvidia-smi` per-process memory runs about 0.48 GiB above the CLI
  load summary's reserved row: the CUDA context and driver-side allocations the planner does not
  count. The summary's `planned slack` therefore over-reports free memory by roughly that much.
  Residency was flat across every sample of every run — a 1M-token prefill does not push it above
  what was chosen at load.
- **Workspace is planned at tp1 extents at tp2**, which over-reserves tens of MiB per device. This
  is a known, unclaimed saving, not a correctness issue.

---

## 5. YaRN

### 5.1 The constants match vLLM *as deployed*, which is not the bare formula

Qwen3's `rope_parameters` carry `mrope_section`, so vLLM's `get_rope` returns `MRotaryEmbedding`,
whose constructor multiplies `max_position_embeddings` by 4 **before** the correction range is
computed. The production correction range is therefore **(16, 24)**, not the bare-formula
(14, 22). NInfer computes (16, 24) deliberately, to match the reference as it actually runs.

Parity against vLLM's own tensors: `inv_freq` agrees to ≤1e-6 relative (observed 1e-7 to 1e-9), and
the native (unscaled) table to 2.99e-10.

The `mscale` for ×4 scaling is **1.13863**.

### 5.2 `mscale` is applied entirely inside the rope path

There is **no attention-level scale change**. `mscale` multiplies `cos` and `sin` — that is, the
whole rotated q/k output, not the phase — for both q and k, over the 64 rotary dimensions of each
256-dimension head. The rotary-subspace contribution to `q·k` is therefore scaled by `mscale²`, and
that is the entire effect.

The attention softmax `scale` argument (`ninfer::targets::qwen3_6::kAttentionScale` /
`Variant::attention_scale`) is a compile-time constant derived from `rsqrt(head_dim)` — 0.0625 for
the 27B `head_dim = 256` variant — computed identically in native and YaRN mode.
`validate_attention_tensors` in `src/ops/wrapper/gqa_attention.cpp` hard-rejects any `scale` that
is not exactly `rsqrt(head_dim)` (`kExpectedScale`, tolerance 1e-6), which by itself forecloses an
attention-level YaRN factor.

Host-side computation lives in `src/targets/qwen3_6/impl/runtime/yarn_rope.h` and runs once at load
time, never on the hot path. `yarn_scale(...).first` becomes the device frequency buffer the
kernels in `src/ops/kernel/rope.cuh` read when `rope_mode == Yarn`, replacing the fixed
`kTextRopeInvFrequency[32]` table; `yarn_rope_mscale(...)` becomes
`RopeFrequencyOverride::mscale` (`include/ninfer/ops/rope.h`).

### 5.3 Rope never enters the GDN layers — in either engine

Audited on both sides. In NInfer, `gdn_mix` builds q/k through conv1d and linear projections, and
the recurrence has no position-dependent softmax. In vLLM, `QwenGatedDeltaNetAttention` has no
`rotary_emb` at all; only `Qwen3NextAttention` applies RoPE.

**YaRN therefore affects the 16 full-attention layers and the MTP attention only.**

### 5.4 YaRN is a separate kernel, not a runtime branch

`rope_yarn_text_kernel` is a distinct kernel. `cuobjdump -sass` of `rope.cu` is **byte-identical
for all 21 native kernels** before and after the YaRN work, so `--rope native` is provably the
pre-YaRN path rather than a path that merely tests a flag.

The native tp2 text rope shape (12 Q / 2 KV) is not a registered fixed domain and falls to the
generic rope kernel. This is a pre-existing shape gap; correctness is unaffected.

### 5.5 The visible-key ceiling had to be widened, and the old bound fails loudly

`ops::kGqaAttentionMaximumVisibleKeys` was **262,144**. The split-KV decode kernels derive
`PageIds = 64` from it, and shared memory overran beyond roughly 348k keys — 1M was not reachable
without changing it. The constant is now **1,048,576**, single-sourced (there is one ceiling
constant, not one per kernel family), with a compile-time shared-memory residency guard and a
negative control: the old bound produces an illegal memory access at 400k keys.

### 5.6 Float32 phase noise at 1M is real, and is parity with the reference

`pos × inv_freq` in float32 carries **≈0.06 rad** of phase error at position ~1e6 for the
highest-frequency pairs. This is **identical in vLLM and Hugging Face** (same float32 cos/sin
cache), and the HF training regime already carries 0.016 rad at 262k.

It is documented rather than fixed, deliberately, to keep bit-faithfulness to the deployed
reference. A double-precision angle path already exists in the repository (the DFlash rope
precedent) and is the lever to pull if 1M retrieval ever disappoints.

### 5.7 What the >262k legs of the real-weights YaRN test do and do not prove

`ninfer_qwen3_8_27b_yarn_real_test` retrieves a needle from prompts past the native 262,144-token
ceiling. That is a test of **position addressability** — that positions beyond the registered
ceiling are representable, reach the kernels, and round-trip through attention, MTP and the graph —
**not** of YaRN's quality.

The control that establishes this: with the YaRN descriptors nulled, a **270k needle still
resolved**. Roughly 10% extrapolation past the trained ceiling is tolerated by the model on its own,
so a passing needle just past 262k does not distinguish correct YaRN from no YaRN at all. What
demonstrates YaRN's value is the retrieval at genuine multiples of the native window: the **653k and
1,046k** rows, at 2.5x and 4.0x the 262,144-token native ceiling, far past anything unscaled
positional handling reaches.

Separately, that the corrected frequency table actually reaches the kernels is shown by direct
comparison rather than by the needle: **241,275 of 248,320** compared logits differ between the
corrected and the bare-formula tables. The needle legs would pass either way; the logit count is
what proves the table in use is the corrected one.

### 5.8 Drift guard

`tests/core/data/yarn_ref_4x.json` holds the reference values recorded against vLLM 0.25.1.
`ninfer_qwen3_6_yarn_rope_drift_test` regenerates them from the installed vLLM via
`tools/tp2/dump_yarn_ref.py` and fails, naming both versions, if either the numbers or the vLLM
version move. Without a vLLM environment it skips (exit 77). `NINFER_VLLM_PYTHON` selects the
interpreter; `NINFER_YARN_REF_JSON` overrides the compared file.

---

## 6. Numerical qualification

### 6.1 The tp1↔tp2 parity gate is comparative, because no absolute bar is attainable

The obvious gate — ≥99.9% token agreement and ≥0.9999 logit cosine between tp1 and tp2 — was
measured to be **unattainable by this model against itself**, before any tp2 result was judged
against it:

| Reference | Positions | Argmax agreement | Mean cosine | Mean KL |
|---|---:|---:|---:|---:|
| Same engine, tp1 prefill vs tp1 **own decode** | 1,532 | **96.61%** | — | — |
| tp1 vs tp1 with `prefill_chunk` 1024 → 128 (benign perturbation control) | 328 | 93.90% | 0.99280 | 0.01785 |
| **tp1 vs tp2** (the question) | 328 | **95.73%** | **0.99328** | **0.01566** |
| tp1 on device 0 vs **tp1 on device 1** (null control) | 384 | **100.00%** | 1.00000 | 0.00000 |

Three things follow, and they define the gate:

1. **A free-running greedy comparison is meaningless as a gate.** One near-tie flip diverges every
   subsequent token. The metric is **teacher-forced per-position** comparison: both engines see the
   prompt plus tp1's generated stream, and are compared position by position.
2. **The bar is comparative, not absolute.** tp2's perturbation must lie inside tp1's own
   benign-perturbation envelope, on matched positions, on three terms: argmax agreement, mean KL,
   and (1 − cosine). tp2 is a *smaller* perturbation than the chunk-size control on every axis.
3. **The null control proves the second GPU contributes nothing of its own** — tp1 on device 0
   versus tp1 on device 1 is bit-identical.

Margins at the last run: KL 0.01566 against a 0.07142 budget; argmax 0.9573 against a 0.9290 floor;
(1 − cos) 0.006719 against 0.028784. All **72 disagreements are near-ties**: tp1's own top-2 gap at
those positions has median 0.25 against 5.75 at agreeing positions (23× larger), and none of the 72
exceeds the local max|Δlogit|. Nine of tp1's twenty chunk-control flips coincide with tp2's — an
~11× enrichment over base rate — so the knife edges are a property of the model, not of the split.

The gate is a **green opt-in ctest** (`ninfer_qwen3_8_27b_tp2_parity_test`, driven by
`tools/tp2/parity.cpp`); a permanently-red test would be worthless. It skips (exit 77) unless the
artifact is named and at least two CUDA devices are visible.

**The needle-retrieval suite remains the definitive acceptance**, and it is unanimous.

### 6.2 A BF16 output-storage floor makes relative-L2 uninformative at 1M

At ~1M keys a near-uniform softmax over zero-mean rows collapses the reference RMS to ≈5.5e-4,
below BF16 output granularity. Relative-L2 therefore has a **storage floor of ≈1.7e-3 at any N**,
and the registered *gross* arm becomes vacuous too (its INT8 absolute term is ≈60% of max|ref| at
1M). Conditioning the queries was tried and **measured to fail**.

The remedy is a **self-calibrating** criterion, defined as a suite-owned named criterion in
`tests/ops/gqa_attention_fixture.h`: a BF16-rounded-FP64-oracle leg measures the floor in the same
shape (**1.65e-3 / 1.77e-3** measured), and the long-window kernel is gated at **≤2.5× that
measured floor** (≈4.14e-3 — *tighter* than the shipped 6.3e-3). The kernel sits at **1.79× /
1.73×** the floor; a one-page-short negative control is rejected at 2.01× / 1.92×.

**Judge 1M attention numerics on gross terms and tokens, or on a floor-calibrated relative-L2 —
never on a raw relative-L2 against a fixed limit.**

### 6.3 MTP is output-equivalent up to near-tie argmax flips, not bit-identical

A verify round evaluates the target model over `K+1` columns at once and an ordinary round over
one, which selects different GEMM shapes. Greedy MTP-on and MTP-off streams can therefore diverge
on a near-tie token. This is a **single-device** property, measured with a tp1 control before any
tp2 criterion was chosen, and it is now a committed asserted fixture: tp1 MTP and tp1 MTP-off first
differ at **position 65 of 96**, and tp2's *ordinary* path drifts from tp1's at position 28.

Losslessness is proven **per position, not statistically**. A teacher-forced oracle checks that
every token a tp2 speculative round commits equals the target model's own argmax given tp2's own
prefix: **64/64** on one probe against 62/64 for tp1's own MTP run; a 96-token confound control at
92/96 against 93/96; batched lanes 63/64 and 62/64 against 63/64.

The gate carries an **absolute acceptance floor of 0.25** plus a 0.8× relative floor where
trajectories coincide. A relative-only form was rejected on evidence: one probe's 0.468 against
0.741 would have failed it while being correct. The batched leg is acceptance-and-structure gated
only; its draft-side sensitivity is genuinely lower than the single-lane legs'.

At 1M the practical consequence is small: on the 1,046k needles the MTP3 and MTP-off answers are
identical token for token (one bundle bit-identical over 24 token ids), and on a 949,885-token soak
stream they first differ at generated index 45, on a near-tie, after which the streams lock step
for 18 tokens on a shift.

`ProgramImplCore::check_peer_mtp_egress` (debug-only, off by default, reachable via
`Engine::debug_enable_peer_egress_check`) reads rank 1's MTP egress back after every round and
compares licensed counts, accepted drafts, next extents and the licensed token prefix, returning
{rounds compared, field mismatches}. It is wired into `ninfer_qwen3_8_27b_mtp_tp2_real_test` on both
probes, with a tp1 negative control asserting the check stays inert without a peer.

### 6.4 Peer ingress is nulled, not inherited

Rank 1 uploads its **own** pinned ingress record (`publish_peer_ordinary_ingress`) with the sampling
counter pointers nulled, and `ordinary_batch_body` throws if that record is absent. Nulled rather
than repointed, because rank 1 never samples in the ordinary round. The invariant to preserve: every
rank-0 device pointer that reaches rank 1's ingress is either repointed at rank 1's own lane or
nulled, with the reason stated at the site.

### 6.5 The `--tp 1` golden-token gate, and exactly what it covers

`scripts/tp1-regression.sh` compares this build's greedy output against token streams recorded from
upstream `ninfer` at base commit `feaf4dd`. Upstream was built from a detached worktree of this
repository with the same generator, build type and toolchain, and both binaries ran the same three
greedy cases back to back on one GPU with nothing else resident: a short chat (23 prompt tokens), an
instruction (2,191) and a document (28,677), all at
`--max-new 128 --greedy --no-thinking --print-token-ids`, the fork at its `--tp 1` / `--rope native`
defaults.

Result: **byte-identical on all three — token ids, generated text, and the deterministic summary
rows.** `diff -r` between the two recorded directories is empty. The goldens, the generator and the
exact commands are in `tests/data/tp1-golden/`. The gate adds about 35 s.

**Its scope is narrower than "byte-identity" suggests**, and this is the statement of record:
greedy text decode, `--tp 1`, `--rope native`, NVFP4 weights, `qwen3.8-27b`, MTP off, concurrency 1.
One summary row, `CUDA Graph memory`, is excluded as non-deterministic — it moved across three
consecutive identical runs of a single binary, so it is not a difference between builds.

### 6.6 Shard-extent oracle coverage is partial

Independent FP64-oracle conformance for shard extents exists only for the **FP8 A8 row extents**
`[5120,3072]` and `[5120,8704]`. The NVFP4, Q4, Q5, W8 and vocabulary shard geometries are
qualified **pairwise** — the split kernels on the shards against the tp1 kernel on the whole
weight — plus the model-level parity, needle and MTP-argmax evidence above.

This is the largest open item in the numerical qualification. Closing it is per-format oracle work
at ten-plus new shapes; it is Op work, not integration work. See
[op-development.md](op-development.md) for the admission rules such work must meet.

---

## 7. Measurement conditions and open items

**Every timing carries a per-GPU power limit, and there are three of them.**

| Condition | What it covers |
|---|---|
| **400 W** per GPU | The main campaign. The minimum settable limit on these cards; sampled draw sat at 346–353 W, so the limit was binding |
| **575 W** per GPU | The re-measurement of the publishable subset |
| **500 W** per GPU | The cross-engine comparison against vLLM (see [Performance](../performance.md)) |

Vendor defaults are 600 W and 575 W; the maximum is 600 W on both cards. Do not quote a figure
without its condition, do not compare rows across conditions, and do not compare any of them
against the single-GPU campaigns elsewhere in [Performance](../performance.md), which were taken
under none of the three.

Lifting the cap helps TP1 more than TP2: one card running the whole model saturates its limit (peak
575.5 W) while two cards sharing it peak at 391 and 406 W, so the TP2-over-TP1 decode advantage
narrows from 1.44× at 400 W to 1.40× at 575 W.

Power *draw* is retained per tick (`power.limit` and `power.draw`) by the current samplers, so newer
CSVs carry their own power condition; the earliest campaign CSVs do not.

### 7.1 Open items

| Item | Status |
|---|---|
| Shard-extent FP64-oracle conformance beyond the FP8 A8 row extents | Open — see §6.6. Currently covered pairwise plus model-level evidence |
| vLLM-NVFP4 like-for-weights throughput comparison | Measured, at a **500 W** per-GPU cap — a third power condition distinct from the 400 W campaign and the 575 W re-measurement. The table and its caveats are in the cross-engine section of [Performance](../performance.md); §7.3 below records what vLLM can serve and why the weights are not identical. Still open within it: no vLLM MTP-off row, and the prefill-chunk difference was not swept |
| CUDA Graph per-device allowance at 1M | Open. The 20 MiB allowance is too tight against a capture pool that is not deterministic: successful boots capture 2.00–3.44 MiB per device, but one 1,048,576-token boot claimed 44,433,408 bytes and aborted, then booted normally on the immediate retry with no change to the command line. Workaround: retry. Fix candidates: raise the allowance, or size it from a measured reservation rather than a fixed per-request multiplier. Log retained at `eval/results/cross-engine-nvfp4/ninfer-500w-700000-mtp0-cudagraph-allowance-failure.txt`, and the successful boots' capacity summaries at `ninfer-boot-summaries.txt` in the same directory |
| Concurrency C=2 at 575 W | Not re-measured; C=1 and C=4 were |
| A seeded temperature > 0 soak | Not run. The soak is greedy, which is what makes its two passes hash-comparable; a sampled soak needs a seeded run and a different determinism criterion |
| Decode split policy at 1M | Tuned at 262k. Smooth rather than pathological at 1M, but the last graph profile spans [32768, 1,048,575] with a fixed `gridDim.y = 85` and has not been swept at that length |
| Attention `DecodeSplitScale = 2` at tp2 | Derived, not benchmarked |
| `kernel_attr_once`'s per-launch cost on tp1 | Estimated negligible against an ~8 ms decode step; never measured |

### 7.2 Deferred performance items

None is a regression against tp1 and none affects correctness:

1. W8, BF16 and Q5 shard decode fall-offs at T = 1 — the tuned launchers are not registered for the
   shard shapes, so the composed route runs instead.
2. No MMA route for the BF16 gating shard (structural).
3. Q4/Q5 `gdn_input` at small T, below the T ≥ 17 bit-exact regime.
4. The composed conv-snapshot route for the fused GDN projection+conv pair: the fused per-format
   snapshot kernels bake the tp1 row profile into compile-time constants, so the shard always takes
   the composed route — the same arithmetic in the same order at one extra kernel launch and one
   BF16 staging round.
5. Workspace planned at tp1 extents at tp2 (§4.1).

### 7.3 Extended context in vLLM, for comparison

vLLM does serve NVFP4 Qwen3.8-27B beyond 262,144 tokens. Extended context there is read from the
checkpoint's `config.json` `rope_parameters` rather than from a serving flag, so an NVFP4
repackaging that ships without a YaRN block is capped at its `max_position_embeddings` (262,144)
*until* one is injected at load — which `--hf-overrides` does:

```
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve unsloth/Qwen3.8-27B-NVFP4 \
  --max-model-len 750000 --hf-overrides '{"text_config": {"rope_parameters": {
    "mrope_interleaved": true, "mrope_section": [11,11,10], "rope_type": "yarn",
    "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0,
    "original_max_position_embeddings": 262144}}}'
```

The measurement host's own `serve-qwen38-27b-long.sh` runs exactly that shape at 750,000 tokens
with FP8 KV, TP2 and MTP3. The difference from NInfer is therefore *where the configuration lives*
— a serving flag on an unmodified artifact here, a checkpoint-config override there — not a
capability difference.

Any throughput row against vLLM also compares **two quantizations**, not one: the vLLM side uses an
independent NVFP4 quantization of the same base fine-tune rather than NInfer's own conversion. The
cross-engine numbers, taken at a **500 W** per-GPU cap, live in
[Performance](../performance.md).

**Freeing the GPUs after a vLLM run needs `SIGKILL`.** `kill -TERM` to the `vllm serve` process
group leaves its `VLLM::Worker_TP*` processes re-parented to init and still holding about 14.6 GiB
each, and they have been observed to stay there for more than ten minutes. A NInfer server launched
into that state fails at boot with a free-memory error and its `--kv-capacity auto` figure would be
wrong even if it did start. Send `SIGTERM` to the group, wait, then `SIGKILL` any surviving
`VLLM::Worker` or `vllm serve` process, and confirm that
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` is empty before launching NInfer.

---

## 8. Evaluation tooling notes

These are properties of the external evaluation stack that will bite anyone reproducing the
retrieval numbers.

- **EvalScope's stock needle prompt and its `judge_strategy: rule` contradict each other.** The
  stock prompt asks for a paraphrase; the rule metric is strict exact match after normalization, so
  a *correct* retrieval scores 0. Every needle result in this tree uses a `prompt_template`
  override that makes the pair coherent, and that override is strictly **stricter** than the stock
  prompt with the default LLM judge. The in-tree
  `eval/configs/qwen3_6_35b_needle_haystack.yaml` inherits the same flaw and its scores are likely
  understated. This is an upstream EvalScope issue.
- **EvalScope tiles a corpus shorter than the requested context** (`_get_context_tokens`),
  silently. Any tier above the corpus's own token count measures retrieval over repeated text. The
  stock corpus is 2.7 MB, so any English tier past roughly 150k tokens repeats; the 653k and 1,046k
  haystacks in this tree are purpose-built distinct text, verified at 200 of 200 sampled windows
  occurring exactly once.
- **`chat_template_kwargs.enable_thinking` is rejected by `ninfer-serve`**; the documented top-level
  `enable_thinking` extension works. vLLM is the other way round, which is why both spellings appear
  in the needle configs.
- **The structured log reports both ranks.** `server_start`'s `engine` block carries `tp` and the
  resolved `devices` list (`src/serve/request_log.cpp`, asserted in `tests/test_request_log.cpp`,
  documented in [serving.md](../serving.md)). Per-device memory rows remain a console table, not a
  log field.

The needle corpus builder, prompt verifier, YaRN reference dumper, parity harness and the P2P,
capture and transport probes all live in `tools/tp2/`. The 1M needle, soak and power runners live in
`eval/`, and their measured outputs are committed under `eval/results/`.

---

## 9. Functional limitations

- **Vision is `--tp 1` only.** The Vision encoder runs on the primary device against replicated
  weights and has no split path, so `--tp 2 --vision` is rejected at startup. YaRN is likewise
  rejected together with `--vision`, because the encoder ropes 2-D image-grid positions.
- **DFlash is rejected at `--tp 2`.** It remains a 35B-A3B text-only backend, and that target has no
  tensor-parallel path at all.
- **`--tp 2` requires an explicit `--devices A,B`** naming two distinct devices of the same compute
  capability.
- **1,048,576 tokens is a one-slot configuration**, by arithmetic rather than policy: the per-slot
  sequence cost is 16.66 GiB per device, so a second slot cannot fit on a 32 GiB card. Concurrency
  at extended context requires coming down from the ceiling — roughly 500k for two slots at INT8 KV.

---

## 10. Related documents

- [Performance](../performance.md) — the measured throughput, memory and latency tables, with their
  power conditions and reproduction commands.
- [CLI](../cli.md) and [HTTP serving](../serving.md) — the `--tp`, `--devices`, `--rope` and
  `--yarn-*` option contracts.
- [Op admission, contracts, ownership, qualification, and performance rules](op-development.md) —
  binding on any change to the split Op families.
- [Paged KV context storage, ownership, and capacity model](paged-kv-cache.md) — the KV substrate
  the per-device pools are carved from.
- [Small-scale concurrent inference architecture](concurrent-inference-architecture.md) —
  scheduling, CUDA Graph and speculative-concurrency contracts.
