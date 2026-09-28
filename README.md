# NInfer V100 Duo

Qwen3.8-27B **official NVFP4** inference on **2 × Tesla V100-SXM2 16 GB (NVLink)**,
TP2, CUDA 12.8. NVFP4 is executed in software on Volta. Based on
[Neroued/ninfer](https://github.com/Neroued/ninfer) and
[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100).

## Configuration and context

| Profile | Context limit (prompt + output) | Prefill chunk | Other settings |
|---|---:|---:|---|
| **Default, recommended** | **180,224 tokens** | **4,096 tokens** | TP2, INT8 KV, MTP3, optimized draft head, CUDA Graph, one concurrent request |
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

Requirements: Linux x86_64, two V100-SXM2 16 GB cards with NVLink, CUDA 12.8,
CMake 3.28+, Ninja, a C++20 compiler, zlib development headers and pkg-config.
See [build and benchmark commands](bench/README.md). Licensed under Apache 2.0.
