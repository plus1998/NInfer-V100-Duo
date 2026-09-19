# Cross-engine long-context throughput: NInfer TP2+YaRN vs vLLM TP2+YaRN (NVFP4, 500 W)

Single-request prefill and decode on **byte-identical Needle-in-a-Haystack prompts** at three
context tiers, through two engines on the same pair of RTX 5090s.

All rows in the comparison table were measured on **2026-08-29 between 06:28 and 07:56 UTC** with
**both GPUs capped at 500 W**. An earlier attempt at 575 W/575 W ended when the host crashed and
rebooted mid-prefill; the caps were then set to 500 W and *every* row below — NInfer's included —
was re-measured under that one condition. The pre-crash partials are kept in `pre-crash-575w/`
and produced no result.

```
# power condition, captured before every server launch (vllm.power.txt, ninfer-500w-*.power.txt)
index, name, power.limit [W], power.default_limit [W], power.max_limit [W]
0, NVIDIA GeForce RTX 5090, 500.00 W, 600.00 W, 600.00 W
1, NVIDIA GeForce RTX 5090, 500.00 W, 575.00 W, 600.00 W
```

Power draw and VRAM were sampled at 3 s throughout every run (`*.vram.csv`, columns
`power_limit_w`, `power_draw_w`); the peaks below come from those samples.

---

## 1. The comparison table (500 W, one request, 512 output tokens, temperature 0)

`prefill tok/s` = `prompt_tokens / TTFT`. `decode tok/s` is client-side over the streamed window
(`(completion_tokens - 1) / (last_delta - first_delta)`). "Peak VRAM/GPU" is the peak this
engine's own process held on each device.

| Engine | Window served | Tier | MTP | Prompt tokens | TTFT (s) | Prefill tok/s | Decode tok/s | Total (s) | Peak VRAM/GPU | Peak draw (GPU0/GPU1) | Cap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vLLM 0.25.1 | 750,000 | 250k | MTP3 (always on) | 249,955 | **79.14** | **3,158.5** | 110.78 | 83.75 | 28.90 GiB | 443.9 / 467.6 W | 500 W |
| vLLM 0.25.1 | 750,000 | 653k | MTP3 (always on) | 652,955 | **353.84** | **1,845.3** | 41.94 | 366.03 | 29.00 GiB | 473.6 / 496.4 W | 500 W |
| vLLM 0.25.1 | 750,000 | 700k | MTP3 (always on) | 699,955 | **402.56** | **1,738.8** | 41.01 | 415.02 | 28.90 GiB | 475.3 / 497.7 W | 500 W |
| NInfer TP2 | 1,048,576 | 250k | off | 249,955 | 92.56 | 2,700.5 | 73.35 | 99.51 | 27.41 GiB | 385.2 / 396.8 W | 500 W |
| NInfer TP2 | 1,048,576 | 250k | MTP3 | 249,955 | 91.66 | 2,726.9 | **155.80** | 94.93 | 28.84 GiB | 380.9 / 373.6 W | 500 W |
| NInfer TP2 | 1,048,576 | 653k | off | 652,955 | 465.82 | 1,401.7 | 56.70 | 474.81 | 27.41 GiB | 414.1 / 440.0 W | 500 W |
| NInfer TP2 | 1,048,576 | 653k | MTP3 | 652,955 | 471.01 | 1,386.3 | **118.74** | 475.28 | 28.84 GiB | 475.3 / 488.1 W | 500 W |
| NInfer TP2 | 1,048,576 | 700k | off | 699,955 | 529.26 | 1,322.5 | 54.84 | 538.55 | 27.41 GiB | 415.2 / 458.8 W | 500 W |
| NInfer TP2 | 1,048,576 | 700k | MTP3 | 699,955 | 532.42 | 1,314.7 | **103.42** | 537.33 | 28.84 GiB | 465.0 / 463.0 W | 500 W |

Prompt token counts are **identical between the engines at every tier** (249,955 / 652,955 /
699,955), counted independently by each server from the same prompt string. That is the check
that the two engines really were given the same input.

### Ratios on identical input

| Tier | Prefill: vLLM / NInfer | Decode: NInfer MTP3 / vLLM | Decode: NInfer MTP-off / vLLM |
|---|---|---|---|
| 250k | **1.17x faster (vLLM)** | **1.41x faster (NInfer)** | 0.66x (vLLM faster) |
| 653k | **1.32x faster (vLLM)** | **2.83x faster (NInfer)** | **1.35x faster (NInfer)** |
| 700k | **1.32x faster (vLLM)** | **2.52x faster (NInfer)** | **1.34x faster (NInfer)** |

### Where the crossover lies

With only 512 output tokens the request is almost entirely prefill, so vLLM finishes first at
every tier. The number of output tokens at which NInfer's decode advantage repays its slower
prefill (NInfer MTP3 vs vLLM):

| Tier | TTFT gap (s) | Per-token decode gain (s/tok) | Break-even output tokens |
|---|---|---|---|
| 250k | 12.5 | 0.00261 | ~4,800 |
| 653k | 117.2 | 0.01542 | ~7,600 |
| 700k | 129.8 | 0.01472 | ~8,800 |

Below those lengths vLLM returns the whole answer sooner; above them NInfer does. The Qwen3.8
card's own guidance for the 1M window (budget up to 262k tokens of reasoning and 131k of final
response on agentic tasks) sits far above every one of these break-even points.

---

## 2. The finding that dominates the decode column: vLLM's MTP stops working past its native window

Speculative acceptance, read from each engine's own counters for exactly the requests in the
table (vLLM: `/metrics` deltas across a freshly booted server; NInfer: the `request_done` record
in `*.requests.jsonl`):

| Tier | vLLM draft tokens | vLLM accepted | vLLM acceptance | NInfer draft tokens | NInfer accepted | NInfer acceptance |
|---|---|---|---|---|---|---|
| 250k | 606 | 311 | **51.32 %** (per-position 152/96/63 of 202) | 567 | 322 | **56.79 %** (145/105/72 of 189) |
| 653k | 1,533 | **0** | **0.00 %** (0/0/0 of 511) | 547 | 328 | **59.96 %** (140/107/81 of 183) |
| 700k | 1,533 | **0** | **0.00 %** (0/0/0 of 511) | 602 | 310 | **51.50 %** (155/95/60 of 201) |

vLLM still *drafts* 3 tokens per step past 262,144 and has **every one of them rejected**, so it
pays the drafter's cost and gets nothing: its decode rate falls from 110.78 tok/s at 250k to
41.94 tok/s at 653k while the MTP-on cost stays on the bill. NInfer's MTP acceptance is
essentially flat across the same range (56.8 % -> 60.0 % -> 51.5 %), and that is the whole of its
2.5-2.8x decode lead at 653k and 700k. Note that NInfer's MTP-*off* decode at 653k/700k
(56.70 / 54.84 tok/s) already beats vLLM's *speculative* decode there (41.94 / 41.01).

This is a behaviour of the vLLM deployment as the measurement host's production script
configures it (`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` together with the YaRN x4
`--hf-overrides`). It was observed, not investigated further.

---

## 3. Context ceiling and memory

| | vLLM (as configured by `serve-qwen38-27b-long.sh`) | NInfer TP2 |
|---|---|---|
| Configured ceiling | 750,000 (`--max-model-len`) | **1,048,576** (`--max-context`, YaRN x4 from 262,144) |
| KV pool capacity measured at boot | **759,297 tokens** (12.73 GiB, fp8, `--gpu-memory-utilization 0.85`) | 1,048,576 tokens (int8, `--kv-capacity auto`) |
| Max concurrency at the configured ceiling | 1.01x (1 slot) | 1 (`--max-concurrency 1`) |
| Peak process VRAM per GPU | 28.90-29.00 GiB | **27.41 GiB** (MTP off) / 28.84 GiB (MTP3) |

NInfer holds a **38 % larger context window in less memory per GPU** (27.41 GiB for 1,048,576
tokens vs 28.90 GiB for 759,297). The 700k tier is near vLLM's ceiling — the largest prompt that
fits its window alongside 512 output tokens is 749,488 tokens — and is comfortably inside
NInfer's.

An earlier boot of the same vLLM script measured 760,847 tokens rather than 759,297; the KV pool
size varies by ~0.2 % between boots with the free-memory profile at launch. Both figures are in
the retained logs.

---

## 4. What was measured, and how

**Prompt.** Every prompt comes from `eval/tp2_needle_throughput_probe.py`'s `build_prompt` — the
EvalScope `needle_haystack` adapter, English subset, the corpus at
`~/.cache/modelscope/datasets/AI-ModelScope--Needle-in-a-Haystack-Corpus`, tokenizer
`/home/pc/models/ninfer-38/unsloth-nvfp4`, depth 50, the `summary` template (needle question plus
"summarise in at least 400 words", so the decode phase is long enough to time). Nominal haystack
sizes 250,000 / 653,000 / 700,000 tokens land at 249,955 / 652,955 / 699,955 prompt tokens. The
250k tier is the same prompt as the published NInfer rows.

**Client.** NInfer rows were taken with the project probe verbatim. vLLM rows were taken with
`xprobe.py` in this directory, which imports `build_prompt` and `SYSTEM_PROMPT` from the project
probe and keeps its timing logic unchanged, but differs in two ways the project probe cannot
express against this server:

* vLLM runs with `--reasoning-parser qwen3`, which routes the model's output to
  `delta.reasoning_content`. The project probe reads only `delta.content`, sees an empty stream
  and reports `ttft = null`. `xprobe.py` times the first delta on *either* channel and records
  which channel carried the tokens (`stream_channels`).
* vLLM ignores a top-level `enable_thinking` field; the equivalent is
  `chat_template_kwargs={"enable_thinking": false}`. `xprobe.py --vllm` sends that, so both
  engines ran in the same non-thinking mode. All vLLM rows in the table report
  `stream_channels: ["content"]`, i.e. thinking was actually off.

**Cold prefills, proven not assumed.** Both engines cache prefixes. Each context tier therefore
ran on a **freshly booted server** on both sides — three vLLM boots, six NInfer boots.

* vLLM: `vllm:prefix_cache_hits_total` was **0** after each measured request
  (0/249,955, 0/652,955, 0/699,955). Recorded in `vllm-spec-and-cache.txt`.
* NInfer: every `request_done` record carries `prefix_cache_hit_tokens = 0`,
  `prefix_reuse_path = "full_reset"`, and `computed_prefill_tokens = prompt_tokens`. See
  `ninfer-server-records.txt`.

**The warm repeat.** One extra vLLM run replayed the identical 250k prompt on the same server:
248,000 of 249,955 tokens served from the prefix cache (99.22 %), TTFT **1.80 s** instead of
79.14 s, decode unchanged at 108.75 tok/s. It is reported as a cache measurement, not as a
comparison row.

**Server-side agreement.** NInfer's own `--request-log-jsonl` timings agree with the client
throughout: e.g. 653k MTP3 client TTFT 471.01 s / decode 118.74 tok/s vs server
`timings_seconds.ttft` 470.98 s / 118.95 tok/s.

**Commands.**

```
# vLLM: the measurement host's production script, unmodified, one boot per tier
cd /home/pc/Projects/vllm && ./serve-qwen38-27b-long.sh
#   unsloth/Qwen3.8-27B-NVFP4, --tensor-parallel-size 2, --max-model-len 750000,
#   --kv-cache-dtype fp8, --gpu-memory-utilization 0.85, --max-num-batched-tokens 16768,
#   --max-num-seqs 4, --enable-prefix-caching, --async-scheduling, FlashInfer,
#   --speculative-config '{"method":"mtp","num_speculative_tokens":3}',
#   --hf-overrides rope_type yarn / factor 4.0 / original_max_position_embeddings 262144

# NInfer: one server per (tier x MTP)
build/apps/ninfer-serve <artifact> --host 127.0.0.1 --port 18080 --model-id qwen3.8-27b \
  --tp 2 --devices 0,1 --rope yarn --yarn-factor 4.0 --yarn-origin 262144 \
  --max-context 1048576 --kv-capacity auto --kv-dtype int8 --max-concurrency 1 \
  --max-pending-requests 8 --pending-timeout-ms 86400000 --prefill-chunk 1024 \
  --log-stats-interval-ms 5000 --request-log-jsonl <prefix>.requests.jsonl \
  [--spec mtp --draft-tokens 3 --lm-head-draft]
```

---

## 5. Caveats — read these before quoting any number above

1. **Different weights.** vLLM served `unsloth/Qwen3.8-27B-NVFP4`, an NVFP4 quantization of the
   *base* Qwen3.8-27B fine-tune. NInfer served its `.ninfer` conversion of the **huihui
   abliterated** fine-tune (`/home/pc/models/ninfer-38/huihui-nvfp4/qwen3_8_27b_nvfp4.ninfer`).
   These are different quantizations of different fine-tunes. The comparison is
   *engine + quantization pipeline*, not a controlled same-weights benchmark. Architecture, layer
   count and hidden sizes are identical, so the prefill/decode arithmetic is the same shape, but
   nothing here isolates the engine from the checkpoint.
2. **Different KV dtypes.** vLLM `fp8`, NInfer `int8`. This affects both memory (section 3) and
   attention bandwidth, and therefore both the prefill and the decode columns.
3. **Different prefill chunking.** vLLM batches up to **16,768** tokens per prefill step
   (`--max-num-batched-tokens`); NInfer uses `--prefill-chunk 1024`. This is the most likely
   single explanation for vLLM's 1.17-1.32x prefill advantage, and it is a tunable, not a
   ceiling — the NInfer prefill numbers here are for the 1,024-token chunk the 1M configuration
   ships with, and were not swept.
4. **vLLM could not be measured with MTP off.** Speculative decoding is fixed at launch; turning
   it off needs a restart with a different `--speculative-config`. That run was not made, so vLLM
   has no MTP-off row. Its 653k/700k rows are effectively MTP-off *behaviour* at MTP-on *cost*
   (0 % acceptance), which is worse than a true MTP-off run would be.
5. **Sampling path.** Both engines were driven at `temperature 0`, `seed 42`, 512 max tokens. The
   vLLM server additionally carries `--override-generation-config '{"temperature": 1.0, "top_p":
   0.95, "top_k": 20}'` from the measurement host's script; the explicit per-request `temperature: 0`
   overrides it, but the two engines' rejection-sampling paths under speculative decoding are not
   guaranteed identical, and no token-level equivalence was checked. This is a throughput
   comparison only — **no quality, retrieval or correctness claim is made**. (Both engines did
   return the correct needle sentence at every tier; that is an observation, not a measurement.)
6. **Prefix cache state.** Handled by booting a fresh server per tier and verifying zero hits on
   both sides (section 4). The single warm row is labelled as such.
7. **Concurrency.** One request at a time (`--max-concurrency 1`; vLLM `--max-num-seqs 4` but
   never more than one in flight). Nothing here says anything about batched serving.
8. **Power.** 500 W cap on both cards, recorded before every launch and sampled every 3 s. Peak
   draw never reached the cap in any NInfer run (max 488.1 W) and came within 3 W of it in vLLM's
   653k/700k runs (496.4 / 497.7 W). Brief excursions to ~510 W were seen in the sampler earlier
   in the session; a 3 s sampler undercounts short transients in any case.
9. **Binary provenance.** All six NInfer rows used one `build/apps/ninfer-serve` binary,
   md5 `8fbf08f4a3584d17a93a7b67c3da6ae8`, verified unchanged before and after the phase. It is
   **not** the binary that produced the 575 W reference rows in section 6 — a concurrent session
   rebuilt the tree at 08:24 local time, before this phase started.
10. **One transient boot failure.** The first NInfer 700k attempt died at startup with
    `CUDA Graph preparation consumed 44433408 bytes on device 0, exceeding the planned
    per-device allowance of 20971520 bytes`. The identical command line had booted successfully
    eight minutes earlier and booted successfully on the immediate retry that produced the table
    rows. The failed boot is retained verbatim as
    `ninfer-500w-700000-mtp0-cudagraph-allowance-failure.txt`. Worth a follow-up; it is not
    reflected in any number above.
11. **vLLM shutdown needs SIGKILL.** `kill -TERM` to the `vllm serve` process group left the
    `VLLM::Worker_TP{0,1}` processes re-parented to init and holding ~14.6 GiB each for over ten
    minutes, which starved the next boot (it failed with a free-memory error). The driver now
    escalates to SIGKILL and refuses to launch until `nvidia-smi --query-compute-apps` is empty.
12. **n = 1 per cell.** Each row is a single request.

---

## 6. Reference: the published NInfer rows at 575 W (NOT part of the comparison)

From `eval/results/final-rerun/stock-power-probes.jsonl`, measured earlier on this branch with
both GPUs at 575 W. Listed for continuity with the published documentation only: they were taken
at a different power cap, with a different binary, and — for the 250k rows — in a different
engine configuration (`--rope native`, `--max-context 262144`, not YaRN at 1,048,576).
Do not mix them into the table in section 1.

| Config | Prompt tokens | TTFT (s) | Decode tok/s | Cap | Window / rope |
|---|---|---|---|---|---|
| TP2, MTP off | 249,955 | 90.04 | 75.16 | 575 W | 262,144, native |
| TP2, MTP off (repeat, warm) | 249,955 | 0.38 | 75.14 | 575 W | 262,144, native |
| TP2, MTP3 | 249,955 | 90.45 | 159.04 | 575 W | 262,144, native |
| TP1, MTP off | 249,955 | 100.97 | 53.84 | 575 W | 252,928, native |
| TP1, MTP3 | 249,955 | 101.22 | 113.37 | 575 W | 252,928, native |
| TP2, MTP off | 1,045,954 | 1,078.16 | 46.01 | 575 W | 1,048,576, YaRN x4 |
| TP2, MTP3 | 1,045,954 | 1,074.17 | 100.33 | 575 W | 1,048,576, YaRN x4 |

Against these, the 500 W + YaRN 250k rows in section 1 (73.35 / 155.80 tok/s decode, 92.56 /
91.66 s TTFT) are 2.0-2.4 % slower. That gap combines the 75 W lower cap with the switch from the
native 262,144 window to the YaRN x4 1,048,576 window; this run does not separate the two.

---

## 7. Files in this directory

| File | What it is |
|---|---|
| `summary.md` | this document |
| `vllm-probes.jsonl` | raw vLLM probe results (3 tiers + 1 warm repeat) |
| `ninfer-probes.jsonl` | raw NInfer probe results (6 rows) |
| `ninfer-server-records.txt` | NInfer `request_done` records: server timings, prefix-cache proof, MTP acceptance |
| `vllm-spec-and-cache.txt` | vLLM `/metrics` speculative + prefix-cache counters per tier |
| `ninfer-peaks.txt` | per-configuration VRAM and power-draw peaks |
| `ninfer-500w-*.requests.jsonl` | NInfer server request logs (authoritative timings) |
| `ninfer-boot-summaries.txt` | each NInfer server's boot capacity summary, extracted from its server log |
| `ninfer-500w-*.power.txt`, `vllm.power.txt` | power condition captured before each launch |
| `ninfer-500w-*.vram.csv`, `vllm.vram.csv` | 3 s VRAM + power-limit + power-draw samples |
| `ninfer-500w-700000-mtp0-cudagraph-allowance-failure.txt` | the transient boot failure (caveat 10), verbatim |
| `xprobe.py`, `run_ninfer_probes.sh`, `vllm_phase.sh`, `sample_gpu.sh`, `peaks.py` | the harness |
| `pre-crash-575w/` | partial artifacts from the aborted 575 W attempt; no results |
| `ninfer-serve.md5` | binary identity for the NInfer rows |
| `superseded-run_vllm_probes.sh` | the first 500 W vLLM attempt's driver, before the reasoning-channel fix; produced no usable timing |

The `.log` files each phase wrote -- the two engines' server logs and the run driver logs -- are
held on the measurement host and are not tracked here, under the repository-wide `*.log` ignore
rule. Everything quoted above is in the tracked `.txt`, `.jsonl` and `.csv` files.
