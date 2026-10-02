# NInfer V100 Duo

Qwen3.8-27B inference on **2 × Tesla V100-SXM2 16 GB (NVLink)**, TP2, CUDA 12.8:
the **official NVFP4** artifact (executed in software on Volta) and the
**GSQ-RCO IQ3_S GGUF-blocks** artifact. Both contain Text, Vision, MTP and a
DFlash2 companion; MTP3 is the faster measured choice for short decode.
Based on
[Neroued/ninfer](https://github.com/Neroued/ninfer) and
[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100).

## Start

Build with `tools/v100/build.sh` and download one of these **registered
`.ninfer` files** (not the source checkpoint or a raw GGUF):

| Model | Download | Local filename used below | Strength on 2 × 16 GB |
|---|---|---|---|
| Official Qwen3.8-27B NVFP4 | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `qwen3_8_27b_nvfp4.ninfer` | Faster short MTP3 decode; smaller maximum context |
| GSQ-RCO IQ3_S NInfer v3 | [Hugging Face](https://huggingface.co/WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3) | `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` | Faster long-prompt prefill; native 262K context |

All commands below use TP2 on two V100s, INT8 KV, MTP3 with an optimized draft
head, and CUDA Graph. The server listens on `127.0.0.1:8080`; stop it before
starting another profile.

On the tested **2 × 16 GB** setup, choose a profile by workload (commands and
capacity limits are detailed below):

| Model | Workload | Startup flags after `model=...` |
|---|---|---|
| NVFP4 | [Text, one request](#vision-one-request) | Launcher defaults: `--max-context 180224 --prefill-chunk 4096 --max-concurrency 1` |
| NVFP4 | [Vision, one request](#vision-one-request) | `--vision --vision-max-tokens 2048 --max-context 155648 --prefill-chunk 1024 --max-concurrency 1 --kv-capacity 155648` |
| NVFP4 | [Text concurrency, four slots](#concurrent-text-requests) | `--max-context 131072 --prefill-chunk 1024 --max-concurrency 4 --kv-capacity 131072` |
| NVFP4 | [Vision + concurrency, two slots](#vision-with-concurrency) | `--vision --vision-max-tokens 2048 --max-context 155648 --prefill-chunk 1024 --max-concurrency 2 --kv-capacity 155648` |
| NVFP4 | [Maximum observed text context, **near-OOM**, not recommended](#vision-one-request) | `--max-context 200000 --prefill-chunk 1024 --max-concurrency 1` |
| GSQ-RCO | [Text, one request / native context](#gsq-rco-iq3_s-gguf-blocks) | `--max-context 262144 --kv-capacity 262144 --max-concurrency 1` |
| GSQ-RCO | [Vision, one request / native context](#gsq-rco-iq3_s-gguf-blocks) | `--vision --vision-max-tokens 2048 --max-context 262144 --prefill-chunk 1024 --max-concurrency 1 --kv-capacity auto` |
| GSQ-RCO | [Text concurrency, three slots](#gsq-rco-iq3_s-gguf-blocks) | `--max-context 262144 --prefill-chunk 1024 --max-concurrency 3 --kv-capacity auto` |
| GSQ-RCO | [Vision + concurrency, three slots](#gsq-rco-iq3_s-gguf-blocks) | `--vision --vision-max-tokens 2048 --max-context 262144 --prefill-chunk 1024 --max-concurrency 3 --kv-capacity auto` |

Four text slots favor aggregate short-prompt throughput; use the [two-slot
text profile](#concurrent-text-requests) for a 155K per-request ceiling. Slots
share a KV pool, so these profiles do not guarantee simultaneous full-context
requests.

**Adjusting to available VRAM.** Begin with the row for your artifact, then
check the *smaller* of the two GPUs' free memory after startup. For more margin,
lower `--kv-capacity` first (but keep it at least `--max-context` if one full
length request must fit); next lower `--prefill-chunk`, `--max-concurrency`,
`--vision-max-tokens`, or `--max-context` as the workload permits. Larger
prefill chunks usually improve long-prompt prefill but reserve more scratch.
`--kv-capacity auto` maximizes the **shared** main-KV pool under its planned
headroom; it does not reserve one full `--max-context` per concurrent slot.
For simultaneous near-limit requests budget roughly the *sum* of their
occupied contexts, not just one context. Memory headroom and the native
262,144-token ceiling do not by themselves guarantee long-context speed or
multiple full-context completions. Vision weights live mainly on GPU 0, so
check both GPUs rather than assuming the same margin.

| Parameter | What it changes |
|---|---|
| `--max-context N` | Per-request token ceiling, including input, media expansion and generated tokens; larger values also enlarge planned per-request state/graphs. |
| `--kv-capacity N\|auto` | Shared physical main-KV token pool; explicit `N` makes memory usage predictable; `auto` chooses an admissible pool with planned margin. |
| `--max-concurrency N` | Startup-fixed active request slots and decode graph shapes; slots without KV entitlement wait for admission. |
| `--prefill-chunk N` | Maximum text prefill chunk; lowering it saves transient workspace at the cost of more/smaller prefill passes. |
| `--vision --vision-max-tokens N` | Enables image/video and caps merged visual tokens *per prompt*; raises primary-GPU allocations. |
| `draft-tokens=3\|4\|5` | MTP draft window in the launcher; compare output quality as well as rate. |

These flags go **after** `model=...`; the launcher applies them after its MTP3
defaults. For DFlash2 experiments, override `--spec dflash2 --draft-tokens K
--lm-head-draft` explicitly; the profiles and tables below use MTP, not DFlash2.

### Vision, one request

Recommended on 2 × 16 GB (155,648 context tokens):

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer \
  --vision --vision-max-tokens 2048 \
  --max-context 155648 --prefill-chunk 1024 \
  --max-concurrency 1 --kv-capacity 155648
```

This leaves about **881 MiB free on the primary GPU** after startup. The
2,048-token Vision budget is the total merged image/video token limit per
prompt, not a text-context limit; text, media, and output together count
toward 155,648. Requests exceeding the visual budget are rejected. The same
155,648-token context has also passed a real 154,771-token image prefill in
the two-slot profile; see [Vision verification](docs/performance.md#v100-duo-vision-and-concurrency).

**Text-only default**, one request, 180,224 context tokens and 4,096-token
prefill chunks:

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer
```

**Text-only 200,000-token context**, one request (smaller prefill chunks):

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer \
  --max-context 200000 --prefill-chunk 1024
```

The text-only default leaves approximately 233 MiB of planned per-GPU
headroom. A fresh 200K startup left only **153 MiB free on the primary GPU**; treat it as a
capacity boundary, **not** a reliable serving recommendation. The 200K variant **must set both flags**; 200K with a 4,096-token
chunk does not fit on these 16 GB cards and uses slower prefill. The model's
262,144-token native limit does not fit this setup. Do not simply add
`--vision` to either text-only command: Vision adds GPU allocations.
See [capacity measurements](docs/performance.md#v100-duo-prefill-investigation).

To change the MTP draft window in any command, put `draft-tokens=4` or
`draft-tokens=5` immediately after `model=...`; the default is `3`. MTP5 was
fastest in the measured decode task but generated different output.

## Prefill

### Two-model comparison after rebuild

Rebuilt `build-v100-duo` on **2 × V100-SXM2 16 GB / CUDA 12.8** and measured
both explicit artifacts through the public Engine on October 2, 2026.
Both use TP2, INT8 KV, CUDA Graph, optimized MTP3, **131,072** context
capacity and 4,096-token chunks. For each code-chat prompt, the two artifacts
produced identical input token IDs; each rate below averages **two fresh,
uncached requests** with one first token from prefill and **512 measured
decode tokens**. Model loading and graph preparation are excluded. This fixed
budget disables model stops and is a speed workload, not an answer-quality
test. The 512-token point uses a different, shorter source excerpt than the
10K/30K points.

| Input length | NVFP4 prefill | GSQ prefill | NVFP4 decode | GSQ decode |
|---:|---:|---:|---:|---:|
| 512 tokens | 878 tok/s | **1,358 tok/s** | **128.60 tok/s** | 110.03 tok/s |
| 10,000 tokens | 1,123 tok/s | **1,857 tok/s** | **120.12 tok/s** | 107.61 tok/s |
| 30,000 tokens | 1,062 tok/s | **1,683 tok/s** | **119.22 tok/s** | 104.55 tok/s |

These measurements show GSQ's prefill advantage and NVFP4's decode advantage
on the *same* inputs/capacity. They are not an old/new binary A/B and cannot
prove zero speed regression; the earlier [context-decay](#context-decay-上下文衰减)
and [GSQ 262K](#gsq-rco-iq3_s-gguf-blocks) numbers use different capacities,
source snapshots or output windows. See the [exact commands and output](docs/performance.md#v100-duo-october-2026-rebuild-two-artifact-comparison).

### MTP3/4/5: peak and average decode

The same rebuilt server, TP2, INT8 KV, Graph, optimized proposal, **131,072**
capacity, one active request, no thinking or prefix reuse. For the same
74-token pelican-HTML user prompt, each model/window ran **three naturally
completed greedy requests**. Peak is the fastest logged five-second decode
interval; average is the arithmetic mean of three full-request decode rates
(`(output tokens − 1) / decode seconds`, excluding prefill and loading).

| Model | Window | Outputs per request | Peak 5 s | Average whole-request decode |
|---|---|---:|---:|---:|
| NVFP4 | MTP3 | 9,668 | 148.2 tok/s | 138.06 tok/s |
| NVFP4 | MTP4 | 9,324 | 156.4 tok/s | 142.87 tok/s |
| NVFP4 | MTP5 | 9,881 | **176.0 tok/s** | **151.45 tok/s** |
| GSQ-RCO | MTP3 | 6,986 | **121.0 tok/s** | **109.61 tok/s** |
| GSQ-RCO | MTP4 | 5,887 | 110.4 tok/s | 95.16 tok/s |
| GSQ-RCO | MTP5 | 6,531 | 119.6 tok/s | 93.75 tok/s |

**Different artifacts and MTP windows produce different text and lengths**;
neither the peak nor the average establishes answer-quality parity. MTP3 is
the default for both models: it is fastest of the measured GSQ windows and
the earlier NVFP4 pelican response better fulfilled the requested animation
than NVFP4 MTP4/5. The lower 131K capacity also differs from the older
NVFP4 180K table below; do not compare their rates as a code regression.
See [measurement details](docs/performance.md#v100-duo-october-2026-rebuild-two-artifact-comparison).

### Three Agent first requests

One uncached Engine request per captured *first SDK turn* after the same rebuild,
with **131,072** context, 4,096-token chunks, MTP3 and eight decode tokens;
the benchmark primes the decode Graph separately, but does not warm up the
measured Agent prompt:

| Captured SDK | Qwen input tokens | NVFP4 TTFT | GSQ TTFT | NVFP4 prefill | GSQ prefill |
|---|---:|---:|---:|---:|---:|
| Pi | 1,298 | 1.248 s | **0.912 s** | 1,040 tok/s | **1,424 tok/s** |
| Codex | 11,720 | 10.452 s | **6.213 s** | 1,121 tok/s | **1,886 tok/s** |
| Claude Code | 16,368 | 14.759 s | **8.978 s** | 1,109 tok/s | **1,823 tok/s** |

The SDKs send requests to a localhost capture server; their rendered payloads
are tokenized independently with each NInfer artifact (the IDs matched).
**TTFT** is from Engine request start to its first generated token, not model
load, SDK startup, network latency or an Agent's visible answer. This new
Codex SDK capture has 11,720 tokens, not the 11,718-token capture below;
it is not an identical-input rerun of the earlier table.

**Text-only default above**, using the official NVFP4 artifact, INT8 KV and
captured **first-turn API requests** from the three Agent SDKs:

| Agent | Prompt (Qwen tokens) | 首 token 时间 (TTFT) | Prefill 速度 |
|---|---:|---:|---:|
| Pi | 1,298 | **1.246 s** | 1,041 tok/s |
| Codex | 11,718 | **10.398 s** | 1,127 tok/s |
| Claude Code | 16,368 | **14.712 s** | 1,113 tok/s |

Each row is **one cold Engine request** on 2 × V100, with no warmup and an
eight-token MTP3 continuation. TTFT is the Engine's `first_token_seconds` from
request start, not browser/HTTP streaming latency; it excludes model load and
Agent startup. The prompt comes from a locally captured SDK
request, rendered and tokenized with the artifact's Qwen tokenizer; it is **not**
the provider's hidden prompt or an interactive Agent workload. The rates cover
prefill only, not decode or model load. For repeat measurements, the 1K/10K/20K
length-proxy sweep and discarded alternatives, see
[prefill investigation](docs/performance.md#v100-duo-prefill-investigation) and
the [SDK capture procedure](tools/v100/agent_prefill/README.md).

## Decode

**Text-only default above, varying only the MTP draft window**, thinking
off, on the English prompt to create a
standalone animated SVG/HTML pelican riding a bicycle (74 prompt tokens). Three
complete requests per MTP window, with no prefix reuse:

| Draft window | Output tokens | Peak decode (5 s) | Average decode (whole request) | MTP acceptance |
|---|---:|---:|---:|---:|
| **MTP3 (default)** | 9,668 | 132.0 tok/s | 122.43 tok/s | 87.56% |
| MTP4 | 9,324 | 142.4 tok/s | 128.42 tok/s | 81.62% |
| MTP5 | 9,881 | **159.2 tok/s** | **135.89 tok/s** | 77.73% |

Peak is the fastest 5-second server throughput interval across three requests;
average is the mean of their complete-request decode rates. Decode excludes
the first token (emitted during prefill) and model load. Acceptance counts
accepted draft tokens divided by proposed draft tokens.
Outputs **differ** between windows: MTP4/5 animate the wheels and crank but
leave the pelican's legs static, unlike MTP3. The default stays MTP3 rather than
trading the requested animation for higher throughput. See the
[prompt, quality check and measurements](docs/performance.md#v100-duo-decode-pelican-html).

## Context decay (上下文衰减)

Same 180,224-token capacity, TP2, INT8 KV, 4,096-token prefill chunks, and
**MTP3** as the default above. Two cold Engine requests per length, each with
**1,024 measured decode tokens** (plus one first token from prefill):

| Prompt context | First token | Prefill | Decode (1,024 tokens) |
|---:|---:|---:|---:|
| 30K | 28.47 s | 1,054 tok/s | 115.88 tok/s |
| 100K | 114.60 s | 873 tok/s | 82.98 tok/s |
| 150K | 196.67 s | 763 tok/s | 77.14 tok/s |

Rates and first-token times are two-run means, excluding model load, measured
with the one-shot NVLink all-reduce (2026-10-01). This
fixed-length code-prompt benchmark **disables stopping** to keep the decode
window identical; continuations repeat after a natural stopping point, so it
measures long-context speed, **not useful-answer quality or Agent latency**. See
[method and individual results](docs/performance.md#v100-duo-context-decay).

## GSQ-RCO IQ3_S (GGUF-blocks)

The [NInfer v3 conversion](https://huggingface.co/WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3)
of ISTA-DASLab's GSQ-RCO IQ3_S GGUF
(`Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer`, 15.0 GB file, 5.66 GiB of
weights per GPU for Text/MTP) runs without runtime repacking: every Text tensor
keeps its own ggml block type (IQ4_XS, IQ3_S, IQ3_XXS, IQ2_XS/XXS/S, IQ1_M,
Q2_K, Q4_K, Q6_K). The artifact also includes quantized Vision weights. The
full native **262,144-token** context
fits with INT8 KV and leaves **2.90 GiB** free per GPU:

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer \
  --max-context 262144 --kv-capacity 262144
```

For **two concurrent text slots while preserving the native 262,144-token
per-request ceiling**, add `--max-concurrency 2` to this command. The
262,144-token KV pool is **shared**: two short requests can decode together,
but two full-context requests cannot. Two simultaneous 40-token prompts with
513 output tokens each formed actual two-row decode batches and completed in
10.67 s / 10.02 s (96.2 / 102.4 aggregate output tok/s). To enlarge the shared
pool without changing the native context ceiling, `--kv-capacity auto` resolved
to **358,016 tokens** with 1 GiB planned headroom. Explicit
`--kv-capacity 409600` also started and ran two short requests, but left only
**277 MiB free per GPU**; use `auto` when headroom matters. The observed
near-OOM limit was **417,856** tokens (137 MiB free); **417,920** failed memory
admission on the next page. This is a measured boundary, **not** a safe default
or a two-long-request qualification. For **two separate
200,000-token KV entitlements** instead, use:

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer \
  --max-context 200000 --kv-capacity 400000 --max-concurrency 2
```

This started with 811 MiB free per GPU and ran two-row decode batches on
short prompts; generation with two fully occupied 200K contexts has **not**
been tested. The native 262K ceiling with a 524,288-token KV pool for two
full-context entitlements failed startup memory admission. These are tested
capacity profiles, **not an exhaustive search for the maximum feasible shared
KV pool**. See [concurrent GSQ measurements](docs/performance.md#v100-duo-gsq-rco-iq3_s-gguf-blocks).

For **three concurrent text slots** at the same native 262,144-token ceiling,
use `--max-concurrency 3 --kv-capacity 400000`. This sits below the observed
404,096-token text-only startup limit, but leaves little GPU headroom; for a
1 GiB planning margin use `--kv-capacity auto` (344,256 tokens measured).
The shared pool cannot hold two, let alone three, full 262K requests.

For **Vision with three concurrent slots** on 2 × 16 GB, reduce the prefill
chunk and bound the per-prompt visual tokens (2,048 is not a text limit):

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer \
  --vision --vision-max-tokens 2048 \
  --max-context 262144 --prefill-chunk 1024 \
  --max-concurrency 3 --kv-capacity 400000
```

This profile started with **283 MiB free per GPU**, correctly described test
images and formed three-row Vision decode batches. It preserves the *single*
request's 262K context ceiling, not three full-context KV entitlements. It
has little memory margin: `--kv-capacity auto` instead resolved **344,832**
tokens with 1 GiB planned headroom and also passed Vision checks. At 4,096-token
prefill chunks, the 400K Vision configuration failed memory admission.
See [real-image and capacity checks](docs/performance.md#v100-duo-gsq-rco-iq3_s-gguf-blocks).

Same method as the context-decay table above (TP2, INT8 KV, 4,096-token
chunks, MTP3, two cold requests, 1,024 measured decode tokens), with
262,144-token capacity:

| Prompt context | First token | Prefill | Decode (1,024 tokens) | MTP accepted |
|---:|---:|---:|---:|---:|
| 30K | 17.90 s | 1,676 tok/s | 98.88 tok/s | 80.3% |
| 100K | 79.34 s | 1,260 tok/s | 76.19 tok/s | 73.7% |
| 150K | 144.97 s | 1,035 tok/s | 70.24 tok/s | 78.1% |
| 250K | 341.33 s | 734 tok/s | 57.84 tok/s | 79.6% |

Against NVFP4 at the same occupied context, prefill is 36–59% faster (fewer
weight bytes through the FP16 tensor-core GEMM) and decode 8–15% slower: its
i-quant decoders cost more instructions per byte at the 4-column MTP3
verification width. Decode projections use int8 activations (llama.cpp's
q8_1 model), so output is not bit-identical to an FP32-activation reference.
MTP4/MTP5 measured slower (81.4 / 65.3 tok/s at 30K). See
[method, kernels and verification](docs/performance.md#v100-duo-gsq-rco-iq3_s-gguf-blocks).

## Concurrent text requests

The launcher's default is **one active request**. `--max-concurrency` enables
batched decode but does **not** split `--max-context` across slots. Main Text KV
is a shared pool: with `--kv-capacity` equal to `--max-context`, one request can
reach its full context ceiling, while shorter requests can overlap if their
combined capacity reservations fit. Requests that cannot be admitted wait;
setting four slots does not guarantee four full-length requests at once.

For **two text requests** on 2 × 16 GB, keep 4,096-token prefill chunks and
allow up to 155,648 tokens in either request:

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer \
  --max-context 155648 --prefill-chunk 4096 \
  --max-concurrency 2 --kv-capacity 155648
```

This started with **703 MiB free on GPU 0** and **709 MiB on GPU 1**. Two
simultaneous 384-token text completions formed actual two-row decode batches.

For **four text requests**, retain a 131,072-token *per-request* ceiling with
1,024-token chunks and one full-length shared KV entitlement:

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer \
  --max-context 131072 --prefill-chunk 1024 \
  --max-concurrency 4 --kv-capacity 131072
```

For more context in two-slot mode, `--max-context 180224 --prefill-chunk 1024
--max-concurrency 2 --kv-capacity 180224` **started** with only **449 MiB** free
on the primary card; 200,000 with those settings failed memory admission. With
4,096-token chunks, 180,224/two-slot did **not** fit; 176,128/two-slot started
with **231 MiB** free. These are startup boundaries, not safe defaults or
full-context generation qualifications. Tune context, concurrency (`1..8`),
chunk size and KV pool together for your load and inspect startup free memory.

With **identical 512-token prompts, 513 output tokens/request**, 131,072-token
context/KV and 1,024-token chunks, one discarded warmup and five measured
public-Engine runs per slot count on 2 × 16 GB gave:

| Simultaneous requests | Mean set time | Aggregate output | Mean decode batch |
|---:|---:|---:|---:|
| 1 | 7.523 s | 68.2 tok/s | 1.00 |
| 2 | 9.146 s | 112.2 tok/s | 1.99 |
| 4 | 14.381 s | 142.7 tok/s | 3.57 |

Aggregate rate counts all generated tokens and includes prefill, not model
load. Four requests give **2.09×** the one-request aggregate rate, but each
set takes longer. These short-prompt throughput figures do not measure
131K-token occupied-context performance or guarantee identical generated
text across batch sizes. See [text concurrency measurements](docs/performance.md#v100-duo-text-concurrency).

For **2 × 32 GB**, the following text-only four-slot setting is an **untested
starting point**; tune context and concurrency for actual prompt lengths and
check the resolved KV capacity and free memory at startup:

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer \
  --max-context 200000 --prefill-chunk 1024 \
  --max-concurrency 4 --kv-capacity auto
```

`auto` sets a shared pool based on available memory; four simultaneous
200K-token requests require **800,000** shared KV tokens and are not implied
by four slots. The native per-request context limit is 262,144 tokens.

## Vision with concurrency

Vision also supports TP2 batching. On 2 × 16 GB, add one Vision slot without
halving the single-request 155K context ceiling:

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer \
  --vision --vision-max-tokens 2048 \
  --max-context 155648 --prefill-chunk 1024 \
  --max-concurrency 2 --kv-capacity 155648
```

The primary GPU had **715 MiB** free at startup. A real image request with
154,771 occupied tokens completed correctly; two shorter visual requests formed
decode batch size **2.00**. This is not two simultaneous 155K requests. For
more headroom, lower `--max-context` to `131072` and use `--kv-capacity auto`:
that resolved **137,920 shared tokens** and left **1.14 GiB** free on the
primary GPU. `auto` reserves 1 GiB of planned sizing headroom and cannot admit
the 155K Vision setting here. Increase `--vision-max-tokens` only when larger
media is required, then recheck available memory; Vision weights and encoding
live on the primary card. The default visual budget without the flag is 32K.

For **2 × 32 GB**, this Vision/four-slot command is a **starting point only**;
no 32 GB GPU was available to verify it:

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer \
  --vision --vision-max-tokens 2048 --max-context 200000 --prefill-chunk 1024 \
  --max-concurrency 4 --kv-capacity auto
```

Adjust `--max-context` and `--max-concurrency` for your workload and check the
resolved shared KV/free memory at startup. Native context is capped at 262,144;
**YaRN remains incompatible with Vision**. See [serving capacity semantics](docs/serving.md#execution-behavior)
and [real-image/concurrency checks](docs/performance.md#v100-duo-vision-and-concurrency).

Requirements: Linux x86_64, two V100-SXM2 cards with NVLink (16 GB tested; 32 GB
profiles require local verification), CUDA 12.8,
CMake 3.28+, Ninja, a C++20 compiler, zlib development headers and pkg-config.
See [build and benchmark commands](bench/README.md). Licensed under Apache 2.0.
