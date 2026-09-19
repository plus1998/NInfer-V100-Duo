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

After downloading the model, build and start NInfer:

```bash
tools/v100/build.sh
tools/v100/ninfer-v100-duo.sh \
  model="$HOME/models/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer"
```

## Performance

Official Qwen3.8-27B NVFP4 v3 artifact, TP2, NVLink, INT8 group-64 KV, CUDA Graphs, optimized
MTP3, greedy decoding, and the production 196,608-token context capacity.

The deterministic synthetic continuation measures the high-acceptance ceiling. It uses
`PP6144+TG256`, a 1,024-token prefill chunk, one discarded warmup, and three measured repetitions:

| Prefill | Decode | MTP accepted | Tokens/round |
|---:|---:|---:|---:|
| 1,012.85 +/- 0.14 tok/s | **109.39 +/- 0.05 tok/s** | **576 / 576 (100%)** | 4.00 |

Real programming behavior was measured through one persistent `ninfer-serve` process with prefix
reuse disabled. The three repository code fixtures use the same fixed seed and request up to 4,096
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
a quality evaluation. Each card holds 10.46 GiB of weights. The production context allocation
leaves about 235 MiB startup headroom per card. A 201,024-token diagnostic configuration has also
run, but leaves only about 133 MiB free per card. The native 262,144-token model ceiling does not
fit this two-card 16 GiB profile.

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
| Context capacity | `--max-context 196608` |
| KV cache | `--kv-dtype int8` |
| Speculative decoding | `--spec mtp --draft-tokens 3 --lm-head-draft` |
| Concurrency | `--max-concurrency 1` |

The launcher requires the model path, binds to `127.0.0.1:8080`, and starts one request slot by
default. Additional server options override those defaults:

```bash
tools/v100/ninfer-v100-duo.sh \
  model=/path/to/model.ninfer \
  --host 0.0.0.0 --port 8081 --max-concurrency 1
```

## Requirements

- Linux x86_64
- 2x V100-SXM2 16 GB with NVLink
- CUDA 12.8
- CMake 3.28+, C++20, Ninja
- zlib development headers, pkg-config

## License

Apache 2.0.
