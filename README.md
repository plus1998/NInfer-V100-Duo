# NInfer V100 Duo

Qwen3.8-27B **official NVFP4** inference on **2 × Tesla V100-SXM2 16 GB (NVLink)**,
TP2, CUDA 12.8. NVFP4 is executed in software on Volta. Based on
[Neroued/ninfer](https://github.com/Neroued/ninfer) and
[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100).

## Configuration and context

| Profile | Context limit (prompt + output) | Prefill chunk | Other settings |
|---|---:|---:|---|
| **Text-only launcher default** | **180,224 tokens** | **4,096 tokens** | TP2, INT8 KV, MTP3, optimized draft head, CUDA Graph, one concurrent request |
| Larger context (explicit override) | 200,000 tokens | 1,024 tokens | Same settings; slower prefill |

The default leaves approximately 233 MiB of planned per-GPU headroom. The 200K
override **must set both flags**; 200K with a 4,096-token chunk does not fit on
these 16 GB cards. The model's 262,144-token native limit does not fit this setup.
See [capacity measurements](docs/performance.md#v100-duo-prefill-investigation).

## Prefill

**Default configuration above**, using the official NVFP4 artifact, INT8 KV and
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

**Production configuration above, varying only the MTP draft window**, thinking
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
| 30K | 28.05 s | 1,070 tok/s | 102.30 tok/s |
| 50K | 49.57 s | 1,009 tok/s | 97.07 tok/s |
| 100K | 113.16 s | 884 tok/s | 81.07 tok/s |
| 150K | 192.37 s | 780 tok/s | 70.70 tok/s |

Rates and first-token times are two-run means, excluding model load. This
fixed-length code-prompt benchmark **disables stopping** to keep the decode
window identical; continuations repeat after a natural stopping point, so it
measures long-context speed, **not useful-answer quality or Agent latency**. See
[method and individual results](docs/performance.md#v100-duo-context-decay).

## Run

Build with `tools/v100/build.sh`. Download the
[official Qwen3.8-27B NVFP4 artifact](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer),
then start the text-only server:

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer
```

The launcher binds to `127.0.0.1:8080`. Choose one MTP window at startup
(stop the current server before launching another):

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer draft-tokens=3  # default; animated legs in this test
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer draft-tokens=4
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer draft-tokens=5  # fastest measured, output differs
```

All three retain the 180,224-token default profile. To restore the larger context:

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer \
  --max-context 200000 --prefill-chunk 1024
```

### Vision and concurrency

The commands above are **text-only**. TP2 now supports `--vision`, including MTP
and concurrent generation. Vision encoding and its weights live on the primary
GPU; both GPUs execute the text model with the composed visual embeddings.
Vision needs additional memory: **do not simply add `--vision` to the 180,224-
or 200,000-token text-only profile on 16 GB cards**. The maximum visual-token
budget is independent of the maximum text context; media is still counted in
each request's total context.

Verified **2 × 16 GB Vision, 155K context per request, two concurrent slots**:

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer \
  --vision --vision-max-tokens 2048 \
  --max-context 155648 --prefill-chunk 1024 \
  --max-concurrency 2 --kv-capacity 155648
```

**155,648 is the per-request ceiling, not half of a two-request allocation.**
The explicitly sized **155,648-token shared KV pool** supports one full-length
request or two shorter requests decoding together; a second request waits if
the combined reservation does not fit. It does **not** reserve 155K per lane.
Startup left about **715 MiB on the primary GPU** and **1,021 MiB on the peer**;
this is a tighter memory profile and needs otherwise idle GPUs. A real image
request with **154,771 occupied prompt tokens** completed correctly, as did the
two-image and batched concurrent checks. Image/video tokens, text history and
output all count toward the same
per-request ceiling. `--vision-max-tokens 2048` separately limits **aggregate
merged visual tokens per prompt** (8,192 raw patches); media beyond that budget
is rejected, not silently truncated. Increase this budget only if your media
workload needs it, then recheck startup capacity; reduce context if other GPU
processes consume memory. Stop other model servers first.

For more headroom, use the **131K auto-sized** variant instead:

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer \
  --vision --vision-max-tokens 2048 \
  --max-context 131072 --prefill-chunk 1024 \
  --max-concurrency 2 --kv-capacity auto
```

`auto` keeps **1 GiB of planned sizing headroom** and resolved a 137,920-token
pool, leaving **1.14 GiB actual free** on the primary card. That reserve is why
`auto` rejects 155K even though the explicit profile fits. Startup at 164K,
172K and 180K with explicit KV was possible, but primary free memory shrank to
about **525, 337 and 147 MiB** respectively; those points were **not** qualified
with near-ceiling Vision requests and are not recommended.

Previously, enabling Vision reserved scratch for up to 32K visual tokens
whenever text context exceeded 32K, making the 16K/24K profiles look like a
text-context limit. The independent visual budget removes that coupling. The
default `--vision-max-tokens` remains 32K for the original broad media envelope,
so **the long-context recommendation requires the explicit 2048 setting**.

For **2 × 32 GB**, use this larger-context/four-slot **starting point** and
adjust it yourself for the workload:

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer \
  --vision --vision-max-tokens 2048 --max-context 200000 --prefill-chunk 1024 \
  --max-concurrency 4 --kv-capacity auto
```

The 32 GB profile is **not hardware-tested here**; this machine has two 16 GB
cards. More context and more slots both consume memory. Tune `--max-context`
and `--max-concurrency` (`1..8`) together and inspect startup's resolved KV
capacity/free memory. The KV pool is shared: to admit all slots at their full
context ceiling simultaneously, it needs at least `max_context × max_concurrency`
tokens; `auto` is memory-limited and need not reach that product. Otherwise
large requests wait for capacity. Native context is capped at 262,144 tokens;
**YaRN remains incompatible with Vision**. See [serving capacity semantics](docs/serving.md#execution-behavior).

The real-image and simultaneous-stream checks, including actual multi-row decode
batch evidence, are documented under
[Vision verification](docs/performance.md#v100-duo-vision-and-concurrency).

Requirements: Linux x86_64, two V100-SXM2 cards with NVLink (16 GB tested; 32 GB
profiles require local verification), CUDA 12.8,
CMake 3.28+, Ninja, a C++20 compiler, zlib development headers and pkg-config.
See [build and benchmark commands](bench/README.md). Licensed under Apache 2.0.
