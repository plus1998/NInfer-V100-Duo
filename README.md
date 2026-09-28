# NInfer V100 Duo

C++/CUDA inference engine for the official **Qwen3.8-27B NVFP4** artifact on
**2x Tesla V100-SXM2 16 GB NVLink**. The NVFP4 arithmetic is implemented in software for Volta;
the model is tensor-parallel across both cards and communicates over direct NVLink peer access.

Based on [Neroued/ninfer](https://github.com/Neroued/ninfer) and
[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100).

## Platform

| | |
|---|---|
| GPU | 2 x Tesla V100-SXM2 16 GB (`sm_70`) |
| Interconnect | NVLink 6-link |
| CUDA | 12.8 |

## Model

| Artifact | Source | Size |
|---|---|---:|
| `qwen3_8_27b_nvfp4.ninfer` | [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | 23.7 GB |

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir ~/models/Qwen3.8-27B-nvfp4-NInfer
```

NInfer uses the official v3 artifact.

## Quick Start

Vision-off launch scheme: text-only Engine, no `--vision` passed (and `--tp 2` rejects it at
startup), with a 180,224-token fast-prefill context capacity:

```bash
tools/v100/build.sh
tools/v100/ninfer-v100-duo.sh \
  model="$HOME/models/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer"
```

The launcher applies `--tp 2 --devices 0,1 --max-context 180224 --prefill-chunk 4096
--kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft --max-concurrency 1
--host 127.0.0.1 --port 8080`; any option passed after `model=` overrides that default.
This profile has about 233 MiB of *planned* per-rank runtime slack after reservation on the
local two-V100 machine. The prior 200K/1,024 context configuration remains available with
`--max-context 200000 --prefill-chunk 1024` together; it cannot fit the larger default chunk.

## Prefill

We captured one first-turn model request from the **Pi, Codex, and Claude Agent SDKs** using
a local-only rejecting endpoint. The system instructions, tools, and user request were rendered
as text and tokenized with the official NInfer artifact's Qwen3.8 tokenizer. These are real SDK
request payloads, **not** vendor-reported token counts, interactive agent task latencies, or
three different models running inference. The agents perform no tool calls; NInfer processes
each extracted payload as a cold, raw-token Engine request. Two V100-SXM2 16 GB cards, TP2,
INT8 KV, 32,768-token capacity, one output token, two measured runs after one warmup:

| Captured first turn | Qwen prompt tokens | Chunk 1,024 | Chunk 4,096 | Chunk 5,120 |
|---|---:|---:|---:|---:|
| Pi | 1,298 | 938.1 | **1,056.0** | 1,056.7 |
| Codex | 11,718 | 1,004.0 | **1,132.9** | 1,129.0 |
| Claude | 16,368 | 999.7 | **1,116.7** | 1,113.6 |

Rates are prompt tokens/s from `GenerationTimings.prefill_seconds`. The same first-token ID
was produced by all three chunk settings within each captured prompt. See the
[local capture harness](tools/v100/agent_prefill/README.md) for SDK versions, limitations,
raw-request handling, and reproduction. It cannot reveal extra prompts a provider inserts
after the request reaches its own servers.

The *fixed-length proxies* below additionally stand in for **Pi (1K)**, **Codex (10K)**, and
**Claude Code (20K)**. All three use the same checked-in token-ID corpus, with exact lengths
1,024, 10,240, and 20,480; no agent-specific prompts or tools enter this length-only sweep.

On the official Qwen3.8-27B NVFP4 artifact, two V100-SXM2 16 GB cards, TP2, INT8 KV, 32,768
context capacity, and one output token, the 1,024/2,048/4,096-token prefill chunk sweep gave
the following prompt tokens/s (two measured requests per point, after one warmup):

| Length proxy | Prompt tokens | Chunk 1,024 | Chunk 2,048 | Chunk 4,096 |
|---|---:|---:|---:|---:|
| Pi agent | 1,024 | 1,042.3 | 1,041.1 | 1,041.3 |
| Codex | 10,240 | 1,018.8 | 1,101.7 | **1,132.7** |
| Claude Code | 20,480 | 992.3 | 1,069.2 | **1,104.1** |

The larger chunk improves the 10K and 20K proxies by 11.2% and 11.3% against 1,024; at exactly
1,024 tokens it provides no benefit. The captured Pi prompt is longer than 1,024 tokens, so
avoiding a second chunk helps it too. A 5,120 chunk is the narrow synthetic peak (1,146.8 at
10K, 1,107.6 tok/s at 20K), but its small extra benefit does not hold for Codex or Claude's
captured prompts, and it reserves another ~142 MiB per GPU compared with 4,096. **Keep 4,096**
as the stable fast-prefill choice; it needs about 426 MiB more workspace than 1,024 at 32K.
The production fast-prefill profile uses 180,224 context and 4,096 chunk; it retains most of
the previous 200K capacity while leaving about 233 MiB of planned slack. On the captured
Claude request at 180,224 with MTP3, measured prefill was 1,109.2 versus 994.0 tok/s with
1,024 (two repetitions each); 5,120 scored 1,107.5 tok/s and left less memory. For an
explicit 32K profile, use `--max-context 32768 --prefill-chunk 4096`. If restoring 200K,
pass `--max-context 200000 --prefill-chunk 1024` together: with MTP3, even 2,048 fails
its reservation by about 17 MiB. Benchmark details, rejected options, and the
1Cat-vLLM operator comparison are in [the V100 prefill notes](docs/performance.md#v100-duo-prefill-investigation).

An independent V100 TPX flash-tile experiment found no worthwhile prefill improvement at
the 1K/10K/20K agent-proxy lengths: the same-capacity 8-warp tile regressed, and a
64-column tile required sacrificing at least 16K of context for a gain confined to
an 85K code prompt. Keep the production 180,224/4,096 configuration; see
[the measured tradeoff](docs/performance.md#v100-tpx-attention-experiments).

Reproduce the three sizes through the public Engine. After the [build](#build), enable and build
the optional benchmark (see [`bench/README.md`](bench/README.md)):

```bash
PKG_CONFIG_PATH="$PWD/build/_deps/install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
  cmake -S . -B build-v100-duo -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build-v100-duo --target ninfer_bench ninfer_v100_corpus -j
export LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for chunk in 1024 2048 4096 5120; do
  build-v100-duo/bench/ninfer_bench \
    --weights /path/to/qwen3_8_27b_nvfp4.ninfer \
    --corpus bench/fixtures/bench_corpus.ids \
    --tp 2 --devices 0,1 --kv-dtype int8 --max-ctx 32768 \
    --prefill-chunk "$chunk" -p 1024,10240,20480 -r 2 --warmup 1
done
```

## Decode

Official Qwen3.8-27B NVFP4 v3 artifact, TP2, NVLink, INT8 group-64 KV, CUDA Graphs, optimized
MTP3, greedy decoding, and a 196,608-token context capacity.

The deterministic synthetic continuation measures the high-acceptance ceiling. It uses
`PP6144+TG256`, a 1,024-token prefill chunk, one discarded warmup, and three measured repetitions:

| Prefill | Decode | MTP accepted | Tokens/round |
|---:|---:|---:|---:|
| 1,023.68 +/- 0.30 tok/s | **133.04 +/- 0.07 tok/s** | **576 / 576 (100%)** | 4.00 |

On this workload, routing the FP8 GDN and attention TP2 small-token projections through the Volta
QPN tensor-core kernels improved decode from 109.27 to 123.97 tok/s (+13.4%) on the same artifact
and hardware. The three measured continuations matched the pre-optimization tokens exactly; the
TP2 projections also pass independent sampled FP64 checks at the four-token decode shape.
Balancing the FP8 QPN K-block assignment and tuning the NVFP4 TP2 down projection then raised
decode from 123.97 to 128.73 tok/s (+3.8%). Prefill kernels were not changed in this round;
prefill measurements can vary between runs.
Prepacking the decode-only FP8 output head and tuning its split-K schedule further raised decode
to 129.33 tok/s without adding a persistent weight copy. Increasing the INT8 GQA attention work
per split at this decode shape reduced partial and reduction overhead, reaching 132.75 tok/s.
Fusing the NVFP4 SwiGLU activation into the up-projection epilogue removed one FP32 scratch plane
and the separate combine launch while preserving the tuned projection kernels, reaching 133.04
tok/s.

For the 85K code-chat corpus, a TP2-only 8-warp key-split INT8 attention kernel raised
MTP3 decode from 76.25 to 90.87 tok/s in two local Engine measurements. Draft acceptance
also changed (82.57% to 85.85%); 8K graph/eager outputs agree exactly, and the
independent attention FP64 oracle passes at 85K. The non-MTP path is unchanged.
See [the workload and rejected alternatives](docs/performance.md#v100-tpx-attention-experiments);
this is not a universal or quality-parity claim.

The following real programming behavior was measured before this decode optimization through one
persistent `ninfer-serve` process with prefix reuse disabled. It has not been remeasured on the
current build. The three repository code fixtures use the same fixed seed and request up to 4,096
completion tokens:

| Fixture | Prompt | Completion | Finish | Decode | MTP accepted | Tokens/round |
|---|---:|---:|---|---:|---:|---:|
| CUDA/C++ | 144 | 4,096 | output limit | 89.8 tok/s | 2,805 / 3,868 (72.5%) | 3.17 |
| Python | 122 | 4,096 | output limit | 95.9 tok/s | 2,890 / 3,612 (80.0%) | 3.40 |
| TypeScript | 122 | 200 | stop token | 98.8 tok/s | 144 / 174 (82.8%) | 3.48 |
| **Request mean** | | | | **94.8 +/- 4.6 tok/s** | **78.4% +/- 5.3%** | **3.35 +/- 0.16** |

These measurements used two V100-SXM2 16 GB cards and CUDA 12.8. Decode is committed output-token
throughput and excludes the first token produced by prefill. The synthetic row is an acceptance
ceiling, not expected application throughput; the code row is a three-request workload sample, not
a quality evaluation. Each card holds 10.46 GiB of weights. The 196,608-token configuration used
for these measurements leaves about 235 MiB startup headroom per card; the *previous* production
200,000-token allocation left about 155 MiB. The native 262,144-token model ceiling does not fit
this two-card 16 GiB profile: the largest context this profile admits is 203,776 tokens, which
leaves 71 MiB free per card.

## Build

For a fresh checkout, the equivalent manual dependency, configure, and build steps are:

```bash
git clone https://github.com/plus1998/NInfer-V100-Duo.git
cd NInfer-V100-Duo
tools/v100/build_dependencies.sh

PKG_CONFIG_PATH="$PWD/build/_deps/install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
cmake -S . -B build-v100-duo -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF
cmake --build build-v100-duo --target ninfer ninfer-serve -j
```

## Run

Recommended production configuration:

| Option | Value |
|---|---|
| Tensor parallelism | `--tp 2 --devices 0,1` |
| Context capacity / prefill chunk | `--max-context 180224 --prefill-chunk 4096` |
| KV cache | `--kv-dtype int8` |
| Speculative decoding | `--spec mtp --draft-tokens 3 --lm-head-draft` |
| Concurrency | `--max-concurrency 1` |
| Vision | off; `--vision` is unsupported at `--tp 2` |

The launcher requires the model path, binds to `127.0.0.1:8080`, and starts one request slot by
default. Additional server options override those defaults:

```bash
tools/v100/ninfer-v100-duo.sh \
  model=/path/to/model.ninfer \
  --host 0.0.0.0 --port 8081 --max-concurrency 1
```

`draft-tokens=N` selects the MTP window (default 3, valid 1..5). Measured on the three repository
code fixtures with thinking off, `draft-tokens=4` raises decode speed and tokens per round and
lowers the acceptance rate; its launch-time context ceiling is 203,712 instead of 203,776 tokens,
so the 180,224-token default capacity above is unchanged.

```bash
tools/v100/ninfer-v100-duo.sh model=/path/to/model.ninfer draft-tokens=4
```

## Requirements

- Linux x86_64
- 2x V100-SXM2 16 GB with NVLink
- CUDA 12.8
- CMake 3.28+, C++20, Ninja
- zlib development headers, pkg-config

## License

Apache 2.0.
