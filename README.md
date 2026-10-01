# NInfer V100 Duo

Qwen3.8-27B inference on **2 × Tesla V100-SXM2 16 GB (NVLink)**, TP2, CUDA 12.8:
the **official NVFP4** artifact (executed in software on Volta) and the
**GSQ-RCO IQ3_S GGUF-blocks** artifact (Text + MTP, [below](#gsq-rco-iq3_s-gguf-blocks)).
Based on
[Neroued/ninfer](https://github.com/Neroued/ninfer) and
[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100).

## Start

Build with `tools/v100/build.sh` and download the
[official Qwen3.8-27B NVFP4 artifact](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer).
All commands below use TP2 on two V100s, INT8 KV, MTP3 with an optimized draft
head, and CUDA Graph. The server listens on `127.0.0.1:8080`; stop it before
starting another profile.

**Recommended single-request Vision** on 2 × 16 GB (155,648 context tokens):

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
headroom. The 200K variant **must set both flags**; 200K with a 4,096-token
chunk does not fit on these 16 GB cards and uses slower prefill. The model's
262,144-token native limit does not fit this setup. Do not simply add
`--vision` to either text-only command: Vision adds GPU allocations.
See [capacity measurements](docs/performance.md#v100-duo-prefill-investigation).

To change the MTP draft window in any command, put `draft-tokens=4` or
`draft-tokens=5` immediately after `model=...`; the default is `3`. MTP5 was
fastest in the measured decode task but generated different output.

## Prefill

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

The NInfer v3 conversion of ISTA-DASLab's GSQ-RCO IQ3_S GGUF
(`Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer`, 15.0 GB file, 5.66 GiB of
weights per GPU) runs unconverted: every tensor keeps its own ggml block type
(IQ4_XS, IQ3_S, IQ3_XXS, IQ2_XS/XXS/S, IQ1_M, Q2_K, Q4_K, Q6_K). **Text and
MTP only** — `--vision` is rejected. The full native **262,144-token** context
fits with INT8 KV and leaves **2.90 GiB** free per GPU:

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer \
  --max-context 262144 --kv-capacity 262144
```

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
