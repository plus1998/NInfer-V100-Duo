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

**Default configuration above**, thinking off: the English prompt to generate a
self-contained animated SVG pelican riding a bicycle (spinning wheels and pedaling),
as a standalone HTML file without external resources, produced **9,668 output
tokens** from a 74-token prompt. Two complete generations measured **122.30 and
122.31 decode tok/s** (mean **122.31 tok/s**); both stopped naturally. The
decode rate excludes the first token (emitted during prefill) and model load. See the
[exact prompt and reproduction details](docs/performance.md#v100-duo-decode-pelican-html).

## Run

Build with `tools/v100/build.sh`. Download the
[official Qwen3.8-27B NVFP4 artifact](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer),
then start the text-only server:

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer
```

The launcher binds to `127.0.0.1:8080` and uses the default profile in the table.
To restore the larger context:

```bash
tools/v100/ninfer-v100-duo.sh model=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer \
  --max-context 200000 --prefill-chunk 1024
```

Requirements: Linux x86_64, two V100-SXM2 16 GB cards with NVLink, CUDA 12.8,
CMake 3.28+, Ninja, a C++20 compiler, zlib development headers and pkg-config.
See [build and benchmark commands](bench/README.md). Licensed under Apache 2.0.
