# NInfer V100 Duo

Qwen3.8-27B inference on **2 × Tesla V100-SXM2 16 GB (NVLink)**, TP2 and
CUDA 12.8. One build supports all three artifacts below; choose one at server
startup. All contain Text, Vision, MTP and a DFlash2 companion; the comparison
and recommended profiles below use **MTP3**.
Based on [Neroued/ninfer](https://github.com/Neroued/ninfer) and
[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100).

## Models

Download the `.ninfer` artifact, not the source checkpoint or raw GGUF:

| Model | Download | Artifact filename | Main advantage on 2 × 16 GB |
|---|---|---|---|
| Official Qwen3.8-27B NVFP4 | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `qwen3_8_27b_nvfp4.ninfer` | Faster MTP decode |
| GSQ-RCO IQ3_S NInfer v3 | [Hugging Face](https://huggingface.co/WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3) | `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` | Faster prefill; full native 262,144-token context |
| Swift-1.5 Qwen3.8-27B OracleRouter DFlash2 NVFP4 | [Hugging Face](https://huggingface.co/kvnxiao/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4-ninfer) | `swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4.ninfer` | Fastest measured MTP decode of the three |

All three load as the same `qwen3.8-27b` target and take the same profile
flags; only the artifact path changes. Swift-1.5 is a separately fine-tuned
model, so its completions differ from the official artifact's.

Build with `tools/v100/build.sh`. For the configurations below, run:

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/model.ninfer
```

The launcher defaults to TP2 on GPUs 0/1, INT8 KV, CUDA Graph, MTP3 with the
optimized proposal head, one active request, and `127.0.0.1:8080`. Replace
`model=...` with **one** downloaded artifact; put profile flags *after* it.
Stop the previous server before switching models.

## Performance Comparison

### MTP3, MTP4, MTP5

Same rebuilt servers, 131,072-token capacity and a 74-token pelican/SVG/HTML
prompt. Each row has three naturally completed, single-request, greedy runs
with no thinking or prefix reuse. **Peak** is the fastest logged five-second
decode interval; **average** is the mean of the three whole-request decode
rates. Both exclude prefill and loading.

| Model | Window | Output tokens per run | Peak 5 s | Average decode |
|---|---|---:|---:|---:|
| NVFP4 | MTP3 | 9,668 | 148.2 tok/s | 138.06 tok/s |
| NVFP4 | MTP4 | 9,324 | 156.4 tok/s | 142.87 tok/s |
| NVFP4 | MTP5 | 9,881 | **176.0 tok/s** | **151.45 tok/s** |
| GSQ-RCO | MTP3 | 6,986 | **121.0 tok/s** | **109.61 tok/s** |
| GSQ-RCO | MTP4 | 5,887 | 110.4 tok/s | 95.16 tok/s |
| GSQ-RCO | MTP5 | 6,531 | 119.6 tok/s | 93.75 tok/s |
| Swift-1.5 | MTP3 | 7,995 | 153.4 tok/s | 143.13 tok/s |
| Swift-1.5 | MTP4 | 8,830 | 176.2 tok/s | 157.83 tok/s |
| Swift-1.5 | MTP5 | 7,778 | **183.6 tok/s** | **163.13 tok/s** |

Different artifacts and draft windows generate **different text and lengths**:
these rates do not establish output-quality parity. GSQ is fastest with MTP3
on this task. Swift-1.5 leads every window on this prompt (measured October 5,
2026 on the same build and recipe as the rows above), but it is a different
fine-tune, so this is same-prompt throughput only; its completions run
7,778–8,830 tokens against NVFP4's 9,324–9,881. NVFP4 MTP5 is faster, but an
earlier inspection found its pelican animation less faithful than MTP3's;
therefore the launcher stays on MTP3 by default. Select MTP4/5 by adding
`draft-tokens=4` or `draft-tokens=5` **immediately after** `model=...`, and
review the resulting output.

### Three Agent First Requests

The Pi, Codex and Claude Agent SDKs sent their first requests to a **local
capture server**, not a model provider. Each captured payload was rendered
and tokenized by both artifacts; corresponding input token IDs matched.
The table is one uncached Engine request per capture with MTP3, 131,072-token
capacity, 4,096-token chunks and eight output tokens. Decode Graph is primed
separately; the Agent prompt has no measured-request warmup.

| Captured SDK | Input tokens | NVFP4 first token | GSQ first token | NVFP4 prefill | GSQ prefill |
|---|---:|---:|---:|---:|---:|
| Pi | 1,298 | 1.248 s | **0.912 s** | 1,040 tok/s | **1,424 tok/s** |
| Codex | 11,720 | 10.452 s | **6.213 s** | 1,121 tok/s | **1,886 tok/s** |
| Claude Code | 16,368 | 14.759 s | **8.978 s** | 1,109 tok/s | **1,823 tok/s** |

First-token time starts at the Engine request, **not** model load, SDK startup,
network transport, Agent tool execution or the first visible UI character.
These are captured prompt measurements, not interactive Agent benchmarks.
Methods and reproduction: [two-artifact measurements](docs/performance.md#v100-duo-october-2026-rebuild-two-artifact-comparison).

### Long-Context Decay (Earlier Campaign)

The October 1, 2026 campaign measured **occupied** code-chat prompt lengths
up to 250K on 2 × V100-SXM2 16 GB, with TP2, INT8 KV, Graph, optimized MTP3,
4,096-token prefill chunks and two uncached requests per point. Each request
generated one prefill token and **1,024 measured decode tokens**, with model
stops disabled. NVFP4 used 180,224-token *capacity*; GSQ used 262,144. They
are comparable by occupied length but **not** a same-capacity or current-build
A/B with the Agent and MTP measurements above.

| Occupied prompt | NVFP4 prefill | GSQ prefill | NVFP4 decode | GSQ decode |
|---:|---:|---:|---:|---:|
| 30,000 | 1,054 tok/s | **1,676 tok/s** | **115.88 tok/s** | 98.88 tok/s |
| 100,000 | 873 tok/s | **1,260 tok/s** | **82.98 tok/s** | 76.19 tok/s |
| 150,000 | 763 tok/s | **1,035 tok/s** | **77.14 tok/s** | 70.24 tok/s |
| 250,000 | — | 734 tok/s | — | 57.84 tok/s |

NVFP4's 250K entry is absent because that configuration does not admit a
250K request. These fixed-output continuations can extend past a natural
stop, so the table measures long-context throughput, not useful-answer or
Agent latency. Details: [context decay](docs/performance.md#v100-duo-context-decay)
and [GSQ long context](docs/performance.md#v100-duo-gsq-rco-iq3_s-gguf-blocks).

## Startup Profiles

On **2 × 16 GB**, start with the **text profile** for your model. NVFP4 needs
one slot to preserve its fast prefill chunk and useful context on 16 GB;
GSQ-RCO can keep that chunk and its native context ceiling with three slots.
Extra slots allow aggregate throughput under multiple simultaneous requests;
they do not make a single request faster.
For NVFP4, the measured 4,096-token chunk is the best practical prefill
choice for the tested agent-size prompts; a larger chunk is not automatically
faster. GSQ uses the same fast measured baseline, without a separate chunk
size sweep.

**NVFP4 — text speed (one request):**

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer \
  draft-tokens=5 --max-context 180224 --prefill-chunk 4096 \
  --max-concurrency 1 --kv-capacity 180224
```

This deliberately gives up 19,776 context tokens versus the 200K capacity
option to retain the faster 4,096-token prefill. MTP5 is the **raw decode-speed
choice on the measured short-prompt workload** (151.45 versus 138.06 tok/s
with MTP3 after the rebuild), not a quality-equivalent replacement: an older
HTML response inspection favored MTP3. Use `draft-tokens=3` when response
quality is more important than that measured decode gain. On the rebuilt
server, this exact MTP5 command started with **297 MiB free on GPU 0**;
if that margin is too small for your host, lower both context and KV capacity
to 163,840 without giving up the 4,096-token prefill chunk.

**Swift-1.5 — the NVFP4 profiles with its own path, in the measured
configuration:**

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4.ninfer \
  draft-tokens=3 --max-context 131072 --prefill-chunk 4096 \
  --max-concurrency 1 --kv-capacity 131072
```

It loads as the same `qwen3.8-27b` target, so every NVFP4 flag above and below
applies with only the artifact path changed; this block is the exact
configuration of its three measurement rows in the comparison above. It is the
fastest of the three artifacts on that prompt (143.13 tok/s average with MTP3,
163.13 tok/s with MTP5), but its completion quality was not inspected, so keep
MTP3 unless you have reviewed MTP5's output. Its startup free memory at
180,224 tokens has **not** been measured; if you use the NVFP4 180,224 profile
above, check both GPUs after startup and reduce context/KV as described below
when the margin is small.

**GSQ-RCO — text default (one request or up to three concurrent):**

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer \
  --max-context 262144 --prefill-chunk 4096 \
  --max-concurrency 3 --kv-capacity auto
```

This keeps its native per-request context ceiling and the measured 4,096-token
prefill baseline; MTP3 beats MTP4/5 on measured GSQ whole-request decode speed.
The three-slot `auto` profile has started with about **1.20 GiB free** and
resolved to 344,256 shared KV tokens in the text-only test. A single request
can occupy 262,144 tokens; three requests cannot all do so simultaneously.
No matched C1-versus-C3 single-request prefill/decode benchmark establishes
that a separate one-slot profile would be faster. Neither command claims to
be universally fastest at every occupied context or prompt; the cited speed
measurements use
the workloads in [the prefill investigation](docs/performance.md#v100-duo-prefill-investigation)
and [MTP comparison](docs/performance.md#v100-duo-october-2026-rebuild-two-artifact-comparison).

**Choose another profile only when the workload requires Vision, NVFP4 text
concurrency, or NVFP4's absolute maximum context.** Flags go after `model=...`
and override the launcher's MTP3 defaults. These alternatives prioritize
features or aggregate throughput over single-request speed:

| Model | Use | Flags after `model=...` | Tradeoff |
|---|---|---|---|
| NVFP4 | Text concurrency | `--max-context 155648 --prefill-chunk 4096 --max-concurrency 2 --kv-capacity 155648` | Two-row decode for overlapping requests; 703 MiB free in the earlier startup test; smaller per-request ceiling. |
| NVFP4 | Vision | `--vision --vision-max-tokens 2048 --max-context 155648 --prefill-chunk 1024 --max-concurrency 2 --kv-capacity 155648` | Two Vision slots, tested image requests; 713 MiB free after the rebuilt-server startup. |
| GSQ-RCO | Vision and native context | `--vision --vision-max-tokens 2048 --max-context 262144 --prefill-chunk 1024 --max-concurrency 3 --kv-capacity auto` | Tested three-row real-image decode; `auto` resolved to 344,832 shared tokens and 1.20 GiB free after rebuilt-server startup. |

**NVFP4 text-only maximum context (separate, single-request option):**

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer \
  --max-context 200000 --prefill-chunk 1024 \
  --max-concurrency 1 --kv-capacity 200000
```

This is the **largest observed NVFP4 startup ceiling**, not the everyday
speed recommendation. It trades away the faster 4,096-token prefill chunk
and concurrency to reach 200K, and the rebuilt-server startup left only
**153 MiB free on the primary GPU**. Treat it as a near-OOM experimental
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
