# Ninfer V100 Duo

C++/CUDA inference engine for the official **Qwen3.8-27B NVFP4** artifact on
**2x Tesla V100-SXM2 16 GB NVLink**. The NVFP4 arithmetic is implemented in software for Volta;
the model is tensor-parallel across both cards and communicates over direct NVLink peer access.

Based on [Neroued/ninfer](https://github.com/Neroued/ninfer) and
[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100). See [NOTICE](NOTICE).

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
| `qwen3_8_27b_tturbo_q4_k_m.ninfer` | [DavidAU/Qwen3.8-27B-TWIN-TURBO...GGUF](https://huggingface.co/DavidAU/Qwen3.8-27B-TWIN-TURBO-Fable-Cold-Fusion-709-L-Uncensored-NM-DAU-NEO-MTP-GGUF) Q4_K_M (optional local profile) | 18 GB |

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir ~/models/Qwen3.8-27B-nvfp4-NInfer
```

The official v3 artifact is read directly. No conversion, downgrade, or runtime weight repacking is
performed. To build the optional Q4_K_M profile instead:

```bash
python3 -m tools.convert.qwen3_8_27b.convert_gguf \
  --model <path-to-Q4_K_M.gguf> \
  --mmproj <path-to-mmproj-BF16.gguf> \
  --out ~/models/qwen3_8_27b_tturbo_q4_k_m.ninfer
```

## Performance

Official Qwen3.8-27B NVFP4 artifact, TP2, NVLink, INT8 KV, CUDA Graphs, MTP3, 32K context:

| Workload | Throughput | MTP accept |
|---|---:|---:|
| PP1K | 997.27 tok/s | n/a |
| PP10K | 997.93 tok/s | n/a |
| PP20K | 975.24 tok/s | n/a |
| TG128 | 55.38 tok/s | 30.65% |

Measured on two V100-SXM2 16 GB cards with CUDA 12.8. Each card holds 10.46 GiB of weights. The
published single-V100 result is about 219 tok/s, but used a different prompt with about 99% MTP
acceptance; it is a reference point, not a hardware scaling ratio for this 30.65%-acceptance run.
The production context capacity is 196,608 tokens; a 201,024-token diagnostic configuration has
also run, but leaves only about 133 MiB free per card. The native 262,144-token model ceiling does
not fit this two-card 16 GiB profile.

Optional Q4_K_M profile, TP2, NVLink, INT8 KV, CUDA Graphs, MTP3, 262K context:

| Prompt | Prefill (tok/s) | Decode (tok/s) | MTP accept | TTFT |
|---:|---:|---:|---:|---:|
| ~1K | 1,286 | 55 | 53% | 0.74 s |
| ~10K | 1,409 | 54 | 53% | 6.40 s |
| ~20K | 1,376 | 54 | 57% | 13.04 s |
| ~100K | 1,090 | 42 | 55% | 81.87 s |
| ~200K | 830 | 31 | 46% | 214.88 s |

### GPU Memory

Dual V100-SXM2, TP2, 256K context:

| | Per GPU | 2 GPUs |
|---|---:|---:|
| Weights | 8.4 GiB | 16.8 GiB |
| KV cache (INT8) | 3.2 GiB | 6.3 GiB |
| Other | 3.4 GiB | 6.9 GiB |
| **Used** | **15.0 GiB** | **30.0 GiB** |
| **Free** | **1.0 GiB** | **2.0 GiB** |

## Build

```bash
git clone https://github.com/tuxKOH/ninfer-V100-Duo.git
cd ninfer-V100-Duo
tools/v100/build_dependencies.sh

PKG_CONFIG_PATH="$PWD/build/_deps/install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
cmake -S . -B build-v100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build-v100 -j
```

## Run

```bash
# CLI
NINFER_V100_DUO_ARTIFACT=~/models/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer \
tools/v100/ninfer-v100-duo.sh \
  --prompt "Hello" --max-new 128 --greedy --no-thinking

# HTTP Server
NINFER_V100_DUO_EXECUTABLE="$PWD/build-v100/apps/ninfer-serve" \
NINFER_V100_DUO_ARTIFACT=~/models/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer \
tools/v100/ninfer-v100-duo.sh \
  --host 127.0.0.1 --port 8080 --max-concurrency 1
```

## Requirements

- Linux x86_64
- 2x V100-SXM2 16 GB with NVLink
- CUDA 12.8
- CMake 3.28+, C++20, Ninja
- FFmpeg dev libs, libcurl, pkg-config

## License

Apache 2.0. See [NOTICE](NOTICE).
