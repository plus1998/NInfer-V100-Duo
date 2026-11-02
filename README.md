# NInfer V100 Duo

Qwen3.8-27B inference on **2 × Tesla V100-SXM2 16 GB (NVLink)**, TP2 and
CUDA 12.8. One build supports all four artifacts below; choose one at server
startup. The primary artifact is the official **Qwen3.8-27B NVFP4** release
with Text, Vision, MTP and a DFlash2 companion; the other three are
alternative artifacts of the same family. The launcher defaults to **MTP3**.
Based on [Neroued/ninfer](https://github.com/Neroued/ninfer) and
[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100).

## Models

Download the `.ninfer` artifact, not the source checkpoint or raw GGUF:

| Model | Download | Artifact filename | Features |
|---|---|---|---|
| Official Qwen3.8-27B NVFP4 | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `qwen3_8_27b_nvfp4.ninfer` | Text, Vision and MTP |
| GSQ-RCO IQ3_S NInfer v3 | [Hugging Face](https://huggingface.co/WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3) | `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` | IQ3_S weights; native 262,144-token context |
| Swift-1.5 Qwen3.8-27B OracleRouter DFlash2 NVFP4 | [Hugging Face](https://huggingface.co/kvnxiao/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4-ninfer) | `swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4.ninfer` | Fine-tuned NVFP4 with DFlash2 |
| Qwen3.8-27B Coder390 EfficientThink Q6_K MTP | [Hugging Face](https://huggingface.co/nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-GGUF-NInfer) | `Qwen3.8-27B-Coder390-EfficientThink-Q6_K-MTP.ninfer` | Coder fine-tune with native Q6_K weights and full-vocabulary MTP |

All four load as the `qwen3.8-27b` family. The first three use an indexed
proposal head; Coder390 requires `proposal-head=full`. Swift-1.5 and
Coder390 are separately fine-tuned models, so their completions differ from
the official artifact's.

Build with `tools/v100/build.sh`. To start the official NVFP4 artifact:

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer
```

The launcher defaults to TP2 on GPUs 0/1, INT8 KV, CUDA Graph, MTP3 with the
optimized proposal head, one active request, and `127.0.0.1:8080`. Run any
other artifact by replacing `model=...` with **one** downloaded artifact
path; put profile flags *after* it. Stop the previous server before
switching models.

## NVFP4 Performance

Measured November 2, 2026 with the official Qwen3.8-27B NVFP4 artifact on
**2 × V100-SXM2 16 GB**, CUDA 12.8, TP2, INT8 KV, CUDA Graph and optimized
MTP3. Each point uses the fixed token corpus, a 2,048-token prefill chunk,
one warmup and three measured single-request runs. Values are means; request
time excludes model loading.

| Input tokens | Context capacity | Prefill | Decode | Total request |
|---:|---:|---:|---:|---:|
| 2,048 | 4,096 | 2,131.6 tok/s | 144.0 tok/s | 2.739 s |
| 16,384 | 32,768 | 1,970.2 tok/s | 142.9 tok/s | 10.108 s |

Each run generates 257 tokens, with the first produced during prefill and
256 counted in decode. Model stops are disabled for this fixed-length test;
these numbers do not describe long-answer quality or end-to-end Agent latency.
For these runs, generated token IDs matched the same workload on the build
before the prefill fix.

Reproduce from a CUDA build, replacing the executable and weight paths:

```bash
BENCH=/path/to/build/bench/ninfer_bench
WEIGHTS=/path/to/qwen3_8_27b_nvfp4.ninfer
CORPUS=bench/fixtures/bench_corpus.ids
"$BENCH" --weights "$WEIGHTS" --corpus "$CORPUS" --tp 2 --devices 0,1 \
  --kv-dtype int8 --max-ctx 4096 --prefill-chunk 2048 \
  --mtp-draft-tokens 3 --lm-head-draft -pg 2048,256 --warmup 1 -r 3 -o json
"$BENCH" --weights "$WEIGHTS" --corpus "$CORPUS" --tp 2 --devices 0,1 \
  --kv-dtype int8 --max-ctx 32768 --prefill-chunk 2048 \
  --mtp-draft-tokens 3 --lm-head-draft -pg 16384,256 --warmup 1 -r 3 -o json
```

## Startup Profiles

On **2 × 16 GB**, select a profile for the artifact and workload. Extra
slots allow multiple simultaneous requests but do not make a single request
faster. Memory headroom depends on context, prefill chunk and concurrency.

**NVFP4 — text speed (one request):**

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer \
  draft-tokens=5 --max-context 180224 --prefill-chunk 4096 \
  --max-concurrency 1 --kv-capacity 180224
```

This profile trades context capacity for a larger prefill chunk. The launcher
defaults to MTP3; `draft-tokens=5` selects a different speculative window.
Check output quality and free memory for your workload, and lower context and
KV capacity if needed.

**Swift-1.5 — example text profile:**

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4.ninfer \
  draft-tokens=3 --max-context 131072 --prefill-chunk 4096 \
  --max-concurrency 1 --kv-capacity 131072
```

Swift-1.5 loads as the same `qwen3.8-27b` target. Check free memory on both
GPUs when changing context or KV capacity.

**GSQ-RCO — text default (one request or up to three concurrent):**

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer \
  --max-context 262144 --prefill-chunk 4096 \
  --max-concurrency 3 --kv-capacity auto
```

The `auto` KV pool is shared across requests; concurrent requests cannot all
necessarily occupy the full per-request context at the same time.

**Choose another profile only when the workload requires Vision, NVFP4 text
concurrency, or NVFP4's absolute maximum context.** Flags go after `model=...`
and override the launcher's MTP3 defaults. These alternatives prioritize
features or aggregate throughput over single-request speed:

| Model | Use | Flags after `model=...` | Tradeoff |
|---|---|---|---|
| NVFP4 | Text concurrency | `--max-context 155648 --prefill-chunk 4096 --max-concurrency 2 --kv-capacity 155648` | Two-row decode for overlapping requests; smaller per-request ceiling. |
| NVFP4 | Vision | `--vision --vision-max-tokens 2048 --max-context 155648 --prefill-chunk 1024 --max-concurrency 2 --kv-capacity 155648` | Two Vision slots, tested image requests; reserves space for visual tokens. |
| GSQ-RCO | Vision and native context | `--vision --vision-max-tokens 2048 --max-context 262144 --prefill-chunk 1024 --max-concurrency 3 --kv-capacity auto` | Three Vision slots share the automatically sized KV pool. |

**NVFP4 text-only maximum context (separate, single-request option):**

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer \
  --max-context 200000 --prefill-chunk 1024 \
  --max-concurrency 1 --kv-capacity 200000
```

This is a **previously observed NVFP4 startup ceiling**, measured before the
latest prefill change. It trades a larger prefill chunk and concurrency for
200K capacity and leaves little memory headroom. Treat it as a near-OOM experimental
capacity setting, not a reliably qualified 200K-occupied completion; reduce
`--max-context` if more memory margin is needed. NVFP4 cannot serve a native
262,144-token request on these 16 GB cards. GSQ-RCO *can* retain its native
262K per-request ceiling with three slots, so no single-slot GSQ profile is
recommended by default.

The concurrent profiles were selected from existing startup/short-request
evidence; no matched mixed-workload A/B has established a universal fastest
concurrency setting. `--max-context` is a
*per-request ceiling*, whereas `--kv-capacity` is a **shared pool**: three
GSQ slots do not imply three simultaneous full-length requests. See the
[text concurrency](docs/performance.md#v100-duo-text-concurrency),
[Vision](docs/performance.md#v100-duo-vision-and-concurrency) and
[GSQ-RCO](docs/performance.md#v100-duo-gsq-rco-iq3_s-gguf-blocks) evidence.

**Coder390 — text speed (one request):**

```bash
tools/v100/ninfer-v100-duo.sh \
  model="$HOME/models/Qwen3.8-27B-Coder390-EfficientThink-Q6_K-MTP.ninfer" \
  draft-tokens=3 proposal-head=full \
  --max-context 131072 --prefill-chunk 4096 \
  --max-concurrency 1 --kv-capacity 131072
```

Coder390 requires the full-vocabulary proposal head. For multiple active
requests, the KV capacity is shared; verify free memory and reduce context or
prefill chunk as needed.

### Adjusting to Available VRAM

Check **both** GPUs' free memory after startup; Vision puts more pressure on
the primary GPU. If memory is tight, reduce the shared `--kv-capacity`, then
lower `--prefill-chunk`, `--max-concurrency`, `--vision-max-tokens` or
`--max-context` to match the workload. Lower chunks cost prefill throughput.
Use `--kv-capacity auto` for the largest admissible pool with planned headroom;
for an explicit value, keep it at least `--max-context` if **one** full-length
request must fit. Several near-limit requests need a pool covering their
combined occupied contexts, plus startup memory margin.

| Flag | Meaning |
|---|---|
| `--max-context N` | Maximum input (including expanded media) **plus output** tokens for each request. |
| `--kv-capacity N\|auto` | Physical main-KV tokens shared across active requests; `auto` reserves planned headroom. |
| `--max-concurrency N` | Startup-fixed maximum number of active requests and decode batch slots. |
| `--prefill-chunk N` | Largest text prefill chunk; larger chunks use more scratch memory. |
| `--vision --vision-max-tokens N` | Enable media and reserve room for up to `N` merged visual tokens per prompt. |

See [serving options](docs/serving.md) and [benchmarks](bench/README.md) for
further detail. Requirements: Linux x86_64, two V100-SXM2 GPUs with NVLink
(16 GB tested), CUDA 12.8, CMake 3.28+, Ninja, C++20, zlib headers and
pkg-config. Licensed under Apache 2.0.
