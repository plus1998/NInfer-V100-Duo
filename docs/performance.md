# Serving performance

## V100 Duo measurement and acceptance

The active port uses two Tesla V100-SXM2 16 GB cards, CUDA 12.8 (`sm_70`), and the local
`qwen3.8-27b/gguf-q4-k-m` artifact converted from LM Studio's Qwen3.8-27B Q4_K_M GGUF.
On the fixed code-generation workload below, three valid runs at exactly 85,000 occupied prompt
tokens measured **53.4075 ± 0.0639 committed decode tok/s** (mean ± sample standard deviation),
**18.68% above** the user's approximately 45 tok/s baseline. Each run measures 512 decode tokens
with 180,000-token capacity. This establishes the speed result for this prompt and output window;
it does not guarantee the same speed on every prompt. The RTX 5090 tables later in this document
are inherited results for different hardware and weight profiles.

Startup disables direct P2P for Linux IOMMU `DMA` and `DMA-FQ` domains, verifies the selected
transfer route with exact byte comparisons in both directions, and rejects startup if verification
fails. The current `DMA-FQ` host uses CUDA's host-staged copies. This route passes the collective
suite, including different tensor sizes, guards and 64 consecutive rounds, and the three public
Engine CUDA Graph measurements below. The INT8 attention test also passes its independent FP64
oracle at 85K occupied keys for all four queries, all 24 heads and every visible key, with TP1/TP2 comparisons
and exact cache checks. Real-model MTP/non-MTP teacher forcing and graph/eager regression gates
pass on their short-prompt fixtures. The source Q4_K/Q6_K codes and scales remain unchanged;
these numerical and state checks support the tested route without asserting universal quality
parity for all prompts.

| Setting | V100 Duo comparison profile |
|---|---|
| GPUs | 2 x Tesla V100-SXM2 16 GB, TP2, devices 0 and 1 |
| CUDA compile/runtime | 12.8 / 12.8 |
| Artifact | `qwen3.8-27b/gguf-q4-k-m`; original Q4_K/Q6_K blocks and scales |
| Request mode | One active request |
| Context capacity | 180,000 tokens, native RoPE |
| Primary occupied context | Exactly 85,000 actual prompt tokens |
| KV cache | INT8 group-64; compared with LM Studio's Q8 KV configuration |
| CUDA Graph | Enabled |
| Prefill chunk | 1,024 tokens |
| NInfer MTP | Fixed draft window of three; optimized proposal head (`--lm-head-draft`) |
| Measured output window | 512 decode tokens plus the first token from prefill; three repetitions |
| TP2 transport | Verified CUDA host-staged copies on the current IOMMU host |

NInfer's `--mtp-draft-tokens 3` uses a fixed three-token proposal window, shortened when the remaining
output or context budget requires it. Verification may accept zero drafts. This is distinct from
LM Studio's maximum-three/minimum-zero draft configuration: a minimum draft count of zero permits
its proposal policy to vary how many drafts it attempts, whereas zero accepted drafts describes
the verification result. The two engines therefore use related, but different, MTP schedules.

| Repetition | Committed decode wall tok/s |
|---|---:|
| 1 | 53.3498 |
| 2 | 53.3966 |
| 3 | 53.4761 |
| Mean ± sample standard deviation | **53.4075 ± 0.0639** |

Every repetition produced the same 513 token IDs, with no EOS/EOG token in the captured window,
and accepted **364 / 444 drafts (81.98%)**. Prefill took **298.132–298.154 s** per repetition and
is excluded from decode throughput. Host CPU use was approximately one of 32 logical cores;
observed GPU memory use was **13,768 / 13,454 MiB**, including about 313 MiB of desktop use on GPU 0.

A separate short-input code-chat measurement used 512 prompt tokens, the same 180,000-token
capacity and execution settings, one warmup, and three 256-token decode windows. Its wall decode
rates were **59.9644, 60.0509 and 60.0721 tok/s**, or **60.0291 ± 0.0571 tok/s** (mean ± sample
standard deviation), with **65.89%** draft acceptance. All three 257-token outputs were identical
and EOS/EOG-free. This exceeds the reported 57 tok/s peak numerically, but the unspecified
occupancy of that peak prevents a matched comparison; it is not the 85K acceptance result.

The user-reported LM Studio baseline is approximately **45 committed decode tok/s at 85K occupied
context**, with a reported **57 tok/s peak** whose context occupancy was not specified. The
approximately 40 tok/s figure at longer context is an estimate. These observations are the
acceptance reference; the 18.68% gain is relative to the reported 45 tok/s value, rather than the
matched-engine measurement below. The 57 tok/s peak has not been exceeded by this 85K result and lacks
the context occupancy needed for a like-for-like comparison.

A matched diagnostic run of LM Studio's CUDA backend **2.33.0** used its automatic two-GPU split,
the same source GGUF and exact 85,000 prompt IDs, Q8 KV, a requested 180,000-token capacity
(rounded by the backend to 180,224), and maximum-three/minimum-zero MTP. Both engines used greedy
sampling and generated 513 output tokens: one from prefill and 512 in the measured decode interval.

| Engine | Decode tok/s | Prefill seconds | Accepted / drafted |
|---|---:|---:|---:|
| NInfer V100 Duo, mean of three runs | **53.4075** | 298.144 | 364 / 444 per run (81.98%) |
| LM Studio CUDA 2.33.0, one run | **35.4977** | 185.374 | 365 / 440 (82.95%) |

The LM run decoded for **14.42347 s**, evaluated all 85,000 prompt tokens without cache reuse,
and stopped at the output limit without any EOS/EOG token. NInfer's decode rate is **50.45% higher**
in this comparison, with similar MTP acceptance. Its prefill is **1.61 times as long**, so this is
a decode improvement, not a reduction in cold-request completion latency. The single LM run is
a diagnostic, not a stable average, and does not replace the user's approximately 45 tok/s
acceptance baseline.

Measure committed output tokens per decode second, excluding the first token produced by prefill.
Rejected draft tokens do not count as output. Report repeated measurements and context occupancy;
a short-prompt result at `--max-context 180000` cannot establish performance at 85K occupied tokens.
Use the same prompt content and sampling for the final LM Studio comparison. Record GPU memory,
power and aggregate CPU use; keep CPU below the user's approximately 85% ceiling.

The public Engine benchmark provides a reproducible greedy diagnostic. The code-chat corpus below
uses the artifact's embedded tokenizer and chat template with thinking disabled, distinct repository
source excerpts, and a final bounded blocking task-queue implementation request. The tool trims the
source excerpt body to make the complete prompt exactly 85,000 tokens, preserving the task and
assistant prefix; it does not repeat excerpts to fill the context. Feed the saved token IDs directly
to both engines, without applying another template or decoding and retokenizing them. Keep the
corpus command's `--output-tokens 1024`: this is part of the fixed prompt's wording, independent of
the measured 512-token decode window. The bundled 65,536-token benchmark corpus is too short for
this case:

```bash
cmake -S . -B build-v100 -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build-v100 --target ninfer_bench ninfer_v100_corpus -j

LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
build-v100/bench/ninfer_v100_corpus \
  /Models/ninfer-V100-Duo/qwen3_8_27b_q4_k_m.ninfer \
  /tmp/v100-code-85000.ids --code-chat 85000 --output-tokens 1024 \
  src/core/host_worker_pool.h \
  src/core/host_worker_pool.cpp \
  src/runtime/engine/concurrent_executor.h \
  src/targets/qwen3_6/impl/runtime/program_impl.h \
  src/targets/qwen3_6/impl/runtime/text_context_impl.h \
  src/targets/qwen3_6/impl/runtime/layouts_impl.h \
  src/targets/qwen3_6/impl/runtime/mtp_impl.h \
  src/ops/kernel/gqa_attention_decode_i8.cuh \
  src/ops/kernel/gqa_attention_decode_i8_tc_volta.cuh

LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
build-v100/bench/ninfer_bench \
  --weights /Models/ninfer-V100-Duo/qwen3_8_27b_q4_k_m.ninfer \
  --tp 2 --devices 0,1 --max-ctx 180000 --kv-dtype int8 \
  --prefill-chunk 1024 --mtp-draft-tokens 3 --lm-head-draft \
  --corpus /tmp/v100-code-85000.ids \
  -pg 85000,512 --warmup 0 -r 3 --capture-generation -o json \
  --output-file /tmp/ninfer-v100-duo-85000.json
```

`-pg 85000,512` fixes the output window at 513 tokens: one from prefill and 512 from decode.
Each measured decode follows the full 85K-token prefill, and the Engine primes its CUDA Graphs
before use; `--warmup 0` omits an additional benchmark warmup generation. For a cross-engine
wall-time decode comparison, compute each repetition's rate as
`512 / (timings.total_seconds - timings.first_token_seconds)`. Both timestamps include the same
prompt preparation and submission origin, so subtraction leaves the interval from first-token
commit to request completion, including work between decode rounds. The separately reported
`decode_output_tok_s` uses accumulated Program decode-phase time and excludes some Engine work
between rounds.

`--capture-generation` retains each measured repetition's raw text and token IDs under
`reps[].generation`. NInfer's fixed-budget benchmark disables model-default stopping but can emit
EOS tokens. An accepted repetition must contain no EOS within its entire 513-token output window;
also inspect the captured text for repetition before counting its rate as useful answer throughput.
The matched 512-token LM comparison uses `compare_llama.py` with `ignore_eos=false`, without
EOG-token logit biases. It requires `tokens_predicted=513` and a non-EOS ending; otherwise it saves
the returned output and rejects the repetition instead of calculating a complete-window rate.

The comparison used this local LM Studio backend command, leaving GPU splitting automatic:

```bash
LD_LIBRARY_PATH=/home/z/.lmstudio/extensions/backends/vendor/linux-llama-cuda-vendor-v1 \
/home/z/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.33.0/llama-server \
  --model /Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
  --ctx-size 180000 --parallel 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 0 \
  --threads 16 --threads-batch 24 \
  --host 127.0.0.1 --port 18081 --no-webui
```

With that backend ready, run the matched single-round diagnostic using Python 3.11:

```bash
.venv/bin/python3 tools/v100/compare_llama.py \
  --url http://127.0.0.1:18081 \
  --corpus /tmp/v100-code-85000.ids \
  --prompt-tokens 85000 --decode-tokens 512 --repetitions 1 \
  --output /tmp/llama-v100-duo-85000.json
```

This one-round LM Studio diagnostic does not establish a stable average. The NInfer comparison
against the user's 45 tok/s baseline uses the three measured repetitions above.

These are fixed-window code-generation throughput measurements, not evaluations of whether the
generated code correctly solves the requested programming task.

Quality checks are separate from timing. Preserving the source quantization blocks is an exact
conversion claim; control-tensor transformations and floating-point operators require numerical
oracles. Real-model MTP/non-MTP teacher forcing, graph/eager comparisons and cross-device state
checks qualify the execution path. A plausible answer or a faster kernel alone is insufficient to
claim unchanged model quality or an end-to-end speedup.

## V100 Duo maximum-context capacity sweep

The capacity comparison in the README varies only the requested maximum context from 1,024
through 65,536 tokens, doubling at each step. Every request uses the same 512-token code-chat
prompt and generates 257 tokens: one from prefill and 256 in the measured decode interval.
It measures the effect of the capacity setting, not inference with that many occupied tokens.

Each engine starts with a fresh resident model at each capacity, discards one complete warmup
request, and measures three complete requests with prompt-cache reuse disabled. Both consume
the exact saved prompt IDs, use greedy sampling, and retain their configured MTP3 policies.
NInfer uses TP2, INT8 group-64 KV, CUDA Graphs and the optimized proposal head; LM Studio uses
backend 2.33.0, automatic GPU splitting, Q8 KV and maximum-three/minimum-zero MTP as above.
Rates exclude the first token and loading time. The tool rejects EOS/EOG, incomplete windows,
truncated input and unexpected LM prompt-cache reuse. Reported deviations are sample standard
deviations across three repetitions.

Reproduce the prompt and comparison using the existing artifact and Python 3.11 environment:

```bash
LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
build-v100/bench/ninfer_v100_corpus \
  /Models/ninfer-V100-Duo/qwen3_8_27b_q4_k_m.ninfer \
  /tmp/v100-code-512.ids --code-chat 512 --output-tokens 1024 \
  src/core/host_worker_pool.h

.venv/bin/python3 tools/v100/bench_capacity.py \
  --corpus /tmp/v100-code-512.ids --output-dir /tmp/v100-capacity
```

The runner executes the engines sequentially and stops only its own temporary LM server.
Its output directory contains raw JSON responses, engine logs, requested and actual capacities,
per-repetition timings, MTP counts, and `summary.json` / `summary.md`. The `--engine ninfer`
and `--engine llama` options allow running the two sides separately in that same directory.

## V100 Duo prefix cache

The same two V100-SXM2 16 GB cards and GGUF-derived Qwen3.8-27B Q4_K_M artifact support
retained-prefix reuse with TP2 and optimized MTP3. The real-model gate uses INT8 group-64 KV,
4,096-token capacity, 256-token prefill chunks, greedy sampling and 32 output tokens. A rendered
code prompt contains 3,274 tokens; `preserve_thinking=true` saves its complete response boundary.

Three warmed cold/cache pairs measured median Engine time to first token of **11.9119 s cold**
and **0.0173602 s cached** (686×). Model loading, prompt rendering and HTTP transport are excluded.
Diagnostic logit/peer copies are disabled for these pairs. Cached requests report 3,274 reused
tokens and zero computed prefill tokens. Two exact-history follow-up turns each prefill only 30
tokens, with roughly 0.20 s time to first token. These are prompt-work savings, not a decode-rate
increase or a cache-enabled comparison against LM Studio.

The gate covers response-checkpoint replay, repeated append, zero-suffix sampling, changed-prefix
reset, rewritten response suffixes and stopping inside an accepted MTP round before continuing.
Repeated checkpoint execution, direct continuation versus checkpoint restore with identical
prefill partitions, and CUDA Graph versus eager execution must agree exactly in generated tokens,
captured logits and MTP acceptance. Both ranks' speculative egress must agree.

Cold re-prefill uses a different BF16 GEMM/GDN partition from retained decode and suffix state.
Two cold comparisons first diverged after 26 and 25 identical output tokens respectively; at each
shared history, a fresh single-output target evaluation assigned the cached choice exactly the
same logit as its selected winner (deficit 0). The test checks this first divergence against the
existing TP2 0.5-logit near-tie bound; it does not claim bit-identical free-running output across
different prefill partitions. Other cold output comparisons remain exact.

Reproduce the Engine gate and the actual HTTP turn/response-checkpoint smoke with the existing
artifact:

```bash
export LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
NINFER_V100_DUO_ARTIFACT=/Models/ninfer-V100-Duo/qwen3_8_27b_q4_k_m.ninfer \
  build-v100/tests/ninfer_qwen3_8_27b_v100-duo_prefix_real_test

.venv/bin/python3 tools/smoke/serve_thinking_preservation.py \
  --artifact /Models/ninfer-V100-Duo/qwen3_8_27b_q4_k_m.ninfer \
  --server-bin build-v100/apps/ninfer-serve --backend mtp \
  --tp 2 --devices 0,1 --kv-dtype int8
```

The HTTP smoke uses a temporary local server with a 1,024-token capacity and 128-token chunks.
The cache remains process-local and can resume only the current frontier or its saved complete
turn/response checkpoint; see [serving cache behavior](serving.md#execution-behavior).

## Inherited RTX 5090 campaigns

Tested Git revisions for the inherited campaigns:

- Qwen3.8-27B NVFP4 MTP0 context-length serving:
  `f08597d6eaafce5b875934aaa85854fcd5426df8`;
- Qwen3.8-27B NVFP4 MTP3 single-request and concurrent fixed-corpus serving:
  `32c9881b6783949df4999422a764b3dcaa111b13`;
- Concurrent MTP3 decode saturation for the three measured Qwen3.6 artifact profiles:
  `26da9df7c1b3d3c04ea7bbd730271aa01d00742a`;
- Refreshed Qwen3.6-35B-A3B and Qwen3.6-27B NVFP4 MTP3:
  `f4f21cc36bd1a83cbc046f668719d591dc9c1e2e`;
- Qwen3.6-35B-A3B stored MTP3 response audit:
  `b1a220f028aa750f75bceb3522ac00bbaab7e42d`;
- Qwen3.6-35B-A3B DFlash block=8 (`k=7`):
  `0dc94097e8ec5c5bcf59b9e13e9d1852f504eb61`;
- Qwen3.6-27B NVFP4 accuracy and MTP0:
  `b3d4d0f50b868711c62432bbd68e746217a2f49a`;
- Qwen3.6-27B groupwise-int MTP3: `5ea3242a206cdb0c4c1beaeb9d8a3048e6248423`;
- Qwen3.6-35B-A3B MTP0 and Qwen3.6-27B groupwise-int MTP0:
  `0795169393cab0f2c16246d4bac20dee735dc2a4`.

The Qwen3.6 measurements characterize its three registered artifact profiles independently on one
NVIDIA GeForce RTX 5090. They cover long-context prefill and baseline decode with speculative
decoding disabled, plus long-reasoning and cross-scenario decode with MTP and DFlash. The Qwen3.6
concurrent decode-saturation campaign measures all three profiles at C=1, 2, 4, and 8. The
Qwen3.8-27B NVFP4 campaign covers the MTP0 long-context profile and the complete MTP3
speculative-decode corpus at C=1, 2, 4, and 8; its C=1 point also supplies the single-request MTP3
results below. The registered Qwen3.8-27B `groupwise-int` profile remains outside the published
benchmark campaign.

Every inherited campaign is single-GPU except **Dual-GPU (TP2) and YaRN 1M context**, which
is measured across two RTX 5090s and under a power limit the single-GPU campaigns were not; the two
sets are not comparable to each other.

The single-request corpus requests were submitted serially to a persistent `ninfer-serve` process
over the loopback OpenAI-compatible HTTP endpoint. Each reported corpus fixture used five fixed
seeds. Values are arithmetic mean ± sample standard deviation, and server warm-up completes before
the measured requests. The concurrent campaign has its own sustained-wave method below.

## Single-request serving performance method

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GiB |
| CUDA compile/runtime | 13.1 / 13.1 |
| CUDA driver API | 13.3 for NVFP4 and refreshed 35B MTP3; 13.1 for the remaining single-request campaigns |
| Request mode | One active request, `stream=false` |
| Maximum context | 262,144 tokens; 131,072 for refreshed NVFP4 MTP3 |
| Prefill chunk | 1,024 tokens |
| KV cache | INT8 group-64 |
| CUDA Graph | Enabled |
| Prefix reuse | Disabled |
| Sampling | Temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0 |
| Greedy profile | Exact argmax (`--sampling greedy` in the corpus runner) |
| MTP0 | no `--spec` |
| MTP3 | `--spec mtp --draft-tokens 3 --lm-head-draft` |
| DFlash block=8 | `--spec dflash --draft-tokens 7 --lm-head-draft` |

The MTP0 profile uses four Long NIAH prompts with approximately 8K, 64K, 128K, and 256K tokens.
Thinking is disabled and the output budget is 128 tokens. These runs measure prefill throughput,
server-internal time to first token, and baseline decode throughput at each context length. Content
scenarios are not repeated with MTP disabled because they do not change the baseline decode path.

The speculative-decode corpus contains three long-reasoning fixtures with thinking enabled and a
65,536-token output limit, followed by twelve fixtures covering code, story, translation, and
structured output. The cross-scenario fixtures disable thinking and use a 4,096-token output limit.
The tables report actual completion lengths rather than assuming that every request reaches its
limit.

Metrics are computed from the server's unrounded phase timings and speculative-decode counters:

```text
prefill_tok_s = prompt_tokens / prefill_seconds
server_ttft_ms = 1000 * (prepare_seconds + vision_seconds + prefill_seconds)
decode_tok_s = (completion_tokens - 1) / decode_seconds
spec_acceptance = accepted_tokens / drafted_tokens
spec_tokens_per_round = 1 + accepted_tokens / speculative_rounds
```

Decode throughput is a transport/execution measurement, not a correctness score. The response text,
finish reason, and fixture-level structural requirements are audited separately below. A request
that exhausts its output budget or enters a repetition loop remains useful as a sustained-decode
stress sample, but is not presented as a successfully completed task.

## Qwen3.8-27B NVFP4 concurrent MTP3 corpus makespan

This campaign uses the complete speculative-decode corpus described above: three long-reasoning
fixtures and twelve cross-scenario fixtures, each with five fixed seeds, for 75 requests. The
runner shuffles that fixed request set once with seed `20260811` and preserves the same ordered HTTP
send sequence at every concurrency. Exactly C persistent client workers each submit their next
request only after receiving the current response. C=1 is therefore a serial single-request corpus
on one persistent server and supplies the per-fixture Qwen3.8 results in the final section.

Each point starts a fresh server on an RTX 5090 with CUDA 13.1 compile/runtime, CUDA driver API
13.3, stochastic sampling, INT8 group-64 KV, a 1,024-token prefill chunk, CUDA Graphs, prefix reuse
disabled, a 131,072-token per-request context ceiling, `--kv-capacity auto`, and
`--spec mtp --draft-tokens 3 --lm-head-draft`. Makespan begins when all client workers are released
and ends when the final complete HTTP response has been read. Prefill and decode rates divide the
corresponding server token totals by that full makespan; average batch includes the entire run,
including workload transitions and drain.

| C | Requests | Computed prefill tokens | Decode tokens | Makespan (s) | Requests/s | Prefill tok/s | Decode tok/s | Avg batch | MTP acceptance | Speedup vs. C1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75 | 15,460 | 752,160 | 4,670.27 | 0.0161 | 3.3 | 161.1 | 1.00 | 60.8% | 1.00× |
| 2 | 75 | 15,460 | 739,951 | 2,510.78 | 0.0299 | 6.2 | 294.7 | 1.98 | 59.2% | 1.86× |
| 4 | 75 | 15,460 | 713,384 | 1,647.74 | 0.0455 | 9.4 | 432.9 | 3.29 | 58.0% | 2.83× |
| 8 | 75 | 15,460 | 723,602 | 2,164.90 | 0.0346 | 7.1 | 334.2 | 2.36 | 57.6% | 2.16× |

All 300 requests completed without a request, CUDA, or out-of-memory failure. C=4 gives the
shortest complete-corpus makespan. C=8 is limited by memory pressure, which constrains effective
batching and makes the end-to-end result slower than C=4. Sampling is stochastic: prompts, seeds,
and send order are fixed, but concurrency-specific numerical routes can change sampled
continuations and their lengths. The makespan speedup is therefore a fixed-workload serving result
rather than a fixed-token normalization; the exact decode-token totals are retained in the table.

## Concurrent MTP3 decode saturation

The concurrent campaign uses the `long_decode_aime26_15` fixture with thinking enabled. The
rendered prompt is 293 tokens, and every request has an 8,192-token output budget. For each
concurrency C, the runner starts a fresh `ninfer-serve` process with `max_concurrency=C`, releases
C non-stream requests together using distinct fixed seeds, and waits for every HTTP response.
Startup and server warmup occur before the measured wave.

All points use an RTX 5090, CUDA 13.1 compile/runtime, CUDA driver API 13.3, stochastic sampling
(temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0), INT8 group-64 KV, a 1,024-token
prefill chunk, CUDA Graphs, prefix reuse disabled, and
`--spec mtp --draft-tokens 3 --lm-head-draft`. Each request has a 16,384-token context ceiling.
`--kv-capacity auto` resolved to exactly `C * 16,384` tokens at every point.

Saturated throughput uses only complete one-second server intervals satisfying all of the following:

- computed prefill tokens are zero;
- `running=C`, `prefilling=0`, and `decode_ready=C`;
- at least one decode round completed;
- every decode round had exactly C rows.

Ramp-up, prefill, and drain intervals are excluded. The reported aggregate rate is:

```text
steady_decode_tok_s = sum(committed_decode_tokens) / sum(interval_seconds)
```

Wave makespan starts when the client threads are released and ends after the last complete HTTP
response. MTP acceptance is aggregated over the complete wave. Each row below is one sustained
wave rather than a repeated-sample mean.

| Model profile | C | Steady (s) | Avg batch | Aggregate decode tok/s | MTP acceptance | Speedup vs. C1 | Wave makespan (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` | 1 | 43.01 | 1.00 | 185.8 | 68.2% | 1.00× | 44.23 |
| Qwen3.6-27B `groupwise-int` | 2 | 65.01 | 2.00 | 247.0 | 69.0% | 1.33× | 66.67 |
| Qwen3.6-27B `groupwise-int` | 4 | 102.02 | 4.00 | 309.5 | 68.4% | 1.67× | 107.49 |
| Qwen3.6-27B `groupwise-int` | 8 | 118.02 | 8.00 | 535.0 | 68.3% | 2.88× | 125.20 |
| Qwen3.6-27B `nvfp4` | 1 | 39.01 | 1.00 | 202.4 | 69.3% | 1.00× | 40.46 |
| Qwen3.6-27B `nvfp4` | 2 | 39.01 | 2.00 | 399.7 | 71.4% | 1.97× | 41.82 |
| Qwen3.6-27B `nvfp4` | 4 | 44.01 | 4.00 | 699.7 | 69.3% | 3.46× | 47.92 |
| Qwen3.6-27B `nvfp4` | 8 | 55.01 | 8.00 | 1,146.9 | 68.6% | 5.67× | 58.57 |
| Qwen3.6-35B-A3B `groupwise-int` | 1 | 12.00 | 1.00 | 593.0 | 67.2% | 1.00× | 13.75 |
| Qwen3.6-35B-A3B `groupwise-int` | 2 | 17.00 | 2.00 | 877.7 | 68.2% | 1.48× | 18.87 |
| Qwen3.6-35B-A3B `groupwise-int` | 4 | 26.01 | 4.00 | 1,166.0 | 69.8% | 1.97× | 28.43 |
| Qwen3.6-35B-A3B `groupwise-int` | 8 | 48.01 | 8.00 | 1,313.8 | 67.3% | 2.22× | 50.20 |

All 45 requests reached their output limit, producing 368,640 completion tokens. The campaign
contained 608 complete full-batch steady intervals and had no request, CUDA, or out-of-memory
failure. At C=8, available device memory after startup was 2.66 GiB for 27B groupwise-int,
2.18 GiB for 27B NVFP4, and 4.38 GiB for 35B-A3B.

## Dual-GPU (TP2) and YaRN 1M context

This inherited campaign was measured on the Qwen3.8-27B
NVFP4 artifact across two RTX 5090s with `--tp 2 --devices 0,1`, INT8 group-64 KV, CUDA Graphs
enabled, and greedy decoding. The extended-context rows additionally use
`--rope yarn --yarn-factor 4.0 --yarn-origin 262144`.

**Power condition.** The campaign was measured with both GPUs held at a **400 W per-GPU cap** --
the minimum settable limit on these cards; the vendor defaults on the measurement host are 600 W
and 575 W, and the maximum is 600 W on both. Sampled draw sat at 346-353 W against the cap, so the
limit was binding. The publishable subset was then **re-measured with both cards at 575 W**, and
the tables below carry both conditions. Lifting the cap helps `--tp 1` considerably more than
`--tp 2`: one card running the whole model saturates its limit (peak sampled draw 575.5 W) while
two cards sharing it peak at 391 and 406 W, so the TP2-over-TP1 decode advantage narrows from
1.44x to 1.40x. Never quote a figure from this section without its power condition. The single-GPU
campaigns above were measured under neither condition and are not comparable to these rows.

### Method

| Setting | Value |
|---|---|
| GPUs | 2 x NVIDIA GeForce RTX 5090, 32 GiB, no NVLink, peer-to-peer unavailable |
| Power limit | 400 W per GPU (campaign) and 575 W per GPU (re-measurement); vendor defaults 600 W / 575 W, maximum 600 W on both cards |
| Artifact | Qwen3.8-27B NVFP4 |
| Tensor parallel | `--tp 2 --devices 0,1` |
| KV cache | INT8 group-64 |
| CUDA Graph | Enabled |
| Prefill chunk | 1,024 tokens |
| Sampling | Greedy (exact argmax) unless a row states otherwise |
| Rope | `native` at 262k; `yarn` factor 4.0, origin 262,144 above it |
| MTP3 | `--spec mtp --draft-tokens 3 --lm-head-draft` |

### Single request, matched 249,955-token prompt

Byte-identical prompt on both widths, 512 generated tokens. TP1 ran at `--max-context 252928`,
the largest window that fits one card after weights; TP2 ran at the full `262144`.

| Metric | TP1 @400 W | TP2 @400 W | TP2/TP1 | TP1 @575 W | TP2 @575 W | TP2/TP1 |
|---|---:|---:|---:|---:|---:|---:|
| Prefill tok/s | 2,269.8 | 2,680.1 | 1.18x | 2,484.2 | 2,787.0 | 1.12x |
| Decode tok/s, MTP off | 52.35 | 75.18 | 1.44x | 53.95 | 75.32 | 1.40x |
| Decode tok/s, MTP3 | 101.7 | 152.1 | 1.50x | 113.60 | 159.39 | 1.40x |
| MTP3 draft acceptance | 50.83% | 57.96% | 1.14x | 50.83% | 57.96% | 1.14x |
| Time to first token, s | 111.0 | 93.7 | 0.84x | 101.0 | 90.0 | 0.89x |
| Per-GPU resident memory | 27.90 GiB (one card) | 15.04 GiB (each card) | | 27.90 GiB | 15.04 GiB | |
| Peak sampled draw, MTP off | — | — | | 575.5 W | 390.9 / 405.7 W | |
| Peak sampled draw, MTP3 | — | — | | 575.8 W | 483.8 / 444.5 W | |

Draft acceptance is identical to four decimal places across the two power conditions (0.5083 and
0.5796), which is the expected result: power changes timing, not arithmetic.

On a 536-token reasoning prompt the same comparison is 152.0 to 189.5 decode tok/s and 56.61% to
58.06% acceptance. TP1 wins short-prompt prefill (7,582 versus 5,451 tok/s at 8,147 tokens), where
the cross-device collectives are not amortized by a long chunked prefill. **Both of these
short-prompt comparisons were measured at the 400 W per-GPU cap only** and were not re-measured at
575 W, so they must not be read against the 575 W rows above.

### Saturated concurrent decode at a 262,144-token window

One server per concurrency point, `--decode-tokens 8192`, stochastic sampling, aggregate committed
decode tok/s over complete full-batch intervals.

| Concurrency | MTP off @400 W | Speedup | MTP3 @400 W | Speedup | MTP off @575 W | MTP3 @575 W |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 93.5 | 1.00x | 172.4 | 1.00x | 94.1 | 177.8 |
| 2 | 183.0 | 1.96x | 287.0 | 1.67x | not re-measured | not re-measured |
| 4 | 314.3 | 3.36x | 466.0 | 2.70x | 320.3 (3.40x) | 475.2 (2.67x) |

MTP raises absolute throughput at every point while scaling less steeply from batching, which is
consistent with its single-lane throughput already sitting closer to the ceiling batching pushes
toward. Engine-accounted per-GPU memory at C=4 was 14.97 GiB (MTP off) and 15.64 GiB (MTP3).

### Extended context, single request

Served from one `ninfer-serve` process at `--max-context 1048576 --max-concurrency 1`. Prefill and
decode are the server's own phase timings.

| Prompt tokens | MTP | Power | Prefill tok/s | Prefill wall | Decode tok/s |
|---:|---|---|---:|---:|---:|
| 8,146 | off | 400 W | 5,517.5 | 1.5 s | 97.89 |
| 652,954-652,955 | off | 400 W | 1,348.4 | 484.2 s | 58.22 |
| 652,954-652,955 | off | 400 W (re-run) | 1,379.5 | 470.7-476.0 s | 58.69 |
| 1,045,954-1,045,955 | off | 400 W | 928.9 | 1,126.0 s (18.8 min) | 46.39 |
| 1,045,954-1,045,955 | off | 575 W | 975.1 | 1,072.6 s (17.9 min) | 48.08 |
| 1,045,954 | off, 512 generated | 575 W | 971.4 | 1,076.7 s | 46.11 |
| 1,045,954 | MTP3, 512 generated | 575 W | 975.0 | 1,072.7 s | 100.54 at 56.41% acceptance |
| 949,885 | off | 400 W | 1,011.89 | 938.7 s | 45.49 |
| 949,885 | MTP3 | 400 W | 1,010.19 | — | 99.51 at 58.65% acceptance |

The two 512-generated-token rows are a **like-for-like** MTP measurement: same prompt, same window,
same token budget, so their ratio -- **2.18x** -- needs none of the cross-window caveat the
949,885-token pair in the same table carries. The 24-token rows are needle requests, whose decode
average starts and ends at a lower position. The 400 W and "400 W (re-run)" rows at 652,954 tokens
are two measurements of the same configuration under the same power condition, taken three days
apart; the 653k tier was not re-measured at 575 W.

Ten 653k requests spread 0.3% in prefill rate and ten 1M requests spread 1.4% over a 4.5-hour run:
no thermal fade and no drift. Decode degrades smoothly with context rather than falling off a
cliff, but the decode split policy was tuned at 262k and has not been swept at 1M.

At ~950k tokens MTP3 reaches 99.51 tok/s at 58.65% acceptance on non-repeating greedy text
(400 W). That comparison against 45.49 tok/s was cross-window -- its denominator averaged a decode
from ~950k to the 1,048,576 ceiling while its numerator decoded only ~950k to ~962k -- and it was
quoted as "about 2.2x" for that reason. The like-for-like pair in the table above settles it:
**100.54 against 46.11 tok/s = 2.18x**, same prompt, same window, same 512-token budget, at 575 W.
Acceptance is flat across context: 57.96% at 250k, 58.65% at ~950k, 56.41% at 1,046k.

Two figures from the same run are reported separately and must not be quoted as the headline: a
repetitive greedy stream reaches 93.54% acceptance and 135.78 tok/s, and a
temperature-0.8 sampled stream 80.36% and 122.21 tok/s. Sampled and greedy acceptance are different
algorithms, and the repetitive figure is a loop artifact. Two different degeneration mechanisms
produce those loops: the MTP-off soak stream repeats whole turns because `--ignore-eos` suppresses
its end-of-turn token, while the MTP3 greedy stream contains no end-of-turn token at all and
collapses into a 187-token content-level loop whose first repeat begins at generated index 1,201.
(The 1,341-token acceptance row above is a suffix-period cut, `12,000 - 57 x 187`, not the loop
onset; about 140 of its tokens sit inside the loop's first block, so 58.65% is a mild upper bound
on the novel-text figure.) Sampled decoding at temperature 0.8 does not loop.

### Memory, per GPU

`Resident` is `nvidia-smi` per-process memory. It was flat across every sample of every run: a 1M
prefill adds nothing to the residency chosen at load, and the workspace peaked at 112.29 MiB inside
a 182.81 MiB reservation during a 949,863-token prefill.

| Context | MTP | Weights | Sequence | Workspace | Reserved | Resident |
|---:|---|---:|---:|---:|---:|---:|
| 262,144 | off | 10.08 GiB | 4.28 GiB | 0.18 GiB | — | 15.04 GiB |
| 262,144 | MTP3 | 10.46 GiB | 4.54 GiB | 0.19 GiB | — | 15.69 GiB |
| 1,048,576 | off | 10.08 GiB | 16.66 GiB | 182.81 MiB | 26.93 GiB | 27.41 GiB |
| 1,048,576 | MTP3 | 10.46 GiB | 17.69 GiB | 192.93 MiB | 28.42 GiB | 28.84 GiB |

`Reserved` is the CLI load summary's per-device row; the 262,144-token rows were measured through
`ninfer-serve`'s startup record and `nvidia-smi` instead. The gap of about 0.48 GiB between
`Reserved` and `Resident` is the CUDA context and driver-side allocations the planner does not
count, so the summary's `planned slack` over-reports free memory by that much. CUDA Graph residency
at 1M was 2.00-3.00 MiB per device against a 20.00 MiB allowance.

Turning MTP3 on costs a measured 1.49 GiB of reserved memory per device at 1,048,576 tokens and
0.65 GiB at 262,144. Its two dominant terms are 0.38 GiB of head weights, fixed at any window, and
1.03 GiB of MTP KV per 1M tokens of window; the remainder of each measured delta is workspace and
sequence-arena rounding. At 1M with MTP3 the margin to a 30 GiB per-device budget is 1.16 GiB.

### Reproduction

The dual-GPU campaign is driven by the same concurrency runner as the single-GPU tables, with the
tensor-parallel flags added:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp3 --suite decode-saturation --concurrency 1 --concurrency 2 --concurrency 4 \
  --tp 2 --devices 0,1 --device 0 \
  --max-context 262144 --kv-capacity 262144 \
  --output profiles/bench/tp2_decode_saturation
```

The extended-context rows are single requests against a server started with the 1M configuration:

```bash
./build/apps/ninfer-serve out/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 \
  --rope yarn --yarn-factor 4.0 --yarn-origin 262144 \
  --max-context 1048576 --kv-capacity auto --kv-dtype int8 \
  --max-concurrency 1 --prefill-chunk 1024 \
  --request-log-jsonl run.requests.jsonl
```

Read prefill and decode from that log's `request_done.timings_seconds`, and confirm the power
condition with `nvidia-smi --query-gpu=power.limit,power.default_limit,power.draw --format=csv`
before quoting any figure.

### Cross-engine comparison against vLLM (NVFP4, 500 W per GPU)

This comparison is a **third power condition**: both cards capped at **500 W per GPU**, distinct
from the 400 W campaign and the 575 W re-measurement above. Rows from the three conditions are not
comparable with each other.

One request at a time, byte-identical needle-in-a-haystack prompts at three context tiers, 512
output tokens, temperature 0, thinking off, and a freshly booted server per tier on both sides so
that every prefill is genuinely cold -- zero prefix-cache hits, verified from each engine's own
counters. `Prefill tok/s` is `prompt_tokens / TTFT`; `decode tok/s` is measured client-side over
the streamed window. `Peak VRAM/GPU` is the peak the engine's own process held on each device, and
peak draw comes from a 3 s sampler. Every row below was taken at the 500 W per-GPU cap.

| Engine | Window served | Tier | MTP | Prompt tokens | TTFT (s) | Prefill tok/s | Decode tok/s | Total (s) | Peak VRAM/GPU | Peak draw (GPU0 / GPU1) |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| vLLM 0.25.1 | 750,000 | 250k | MTP3 | 249,955 | 79.14 | **3,158.5** | 110.78 | 83.75 | 28.90 GiB | 443.9 / 467.6 W |
| vLLM 0.25.1 | 750,000 | 653k | MTP3 | 652,955 | 353.84 | **1,845.3** | 41.94 | 366.03 | 29.00 GiB | 473.6 / 496.4 W |
| vLLM 0.25.1 | 750,000 | 700k | MTP3 | 699,955 | 402.56 | **1,738.8** | 41.01 | 415.02 | 28.90 GiB | 475.3 / 497.7 W |
| NInfer TP2 | 1,048,576 | 250k | off | 249,955 | 92.56 | 2,700.5 | 73.35 | 99.51 | 27.41 GiB | 385.2 / 396.8 W |
| NInfer TP2 | 1,048,576 | 250k | MTP3 | 249,955 | 91.66 | 2,726.9 | **155.80** | 94.93 | 28.84 GiB | 380.9 / 373.6 W |
| NInfer TP2 | 1,048,576 | 653k | off | 652,955 | 465.82 | 1,401.7 | 56.70 | 474.81 | 27.41 GiB | 414.1 / 440.0 W |
| NInfer TP2 | 1,048,576 | 653k | MTP3 | 652,955 | 471.01 | 1,386.3 | **118.74** | 475.28 | 28.84 GiB | 475.3 / 488.1 W |
| NInfer TP2 | 1,048,576 | 700k | off | 699,955 | 529.26 | 1,322.5 | 54.84 | 538.55 | 27.41 GiB | 415.2 / 458.8 W |
| NInfer TP2 | 1,048,576 | 700k | MTP3 | 699,955 | 532.42 | 1,314.7 | **103.42** | 537.33 | 28.84 GiB | 465.0 / 463.0 W |

vLLM 0.25.1 served `unsloth/Qwen3.8-27B-NVFP4` at `--tensor-parallel-size 2` with FP8 KV, YaRN x4
injected through `--hf-overrides`, `--max-model-len 750000`, `--max-num-batched-tokens 16768`,
FlashInfer, and `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`. Its
speculative configuration is fixed at launch, so all three of its rows are MTP3. NInfer served its
own NVFP4 artifact at `--tp 2 --rope yarn --yarn-factor 4.0 --yarn-origin 262144
--max-context 1048576 --kv-dtype int8 --prefill-chunk 1024`.

The two engines counted **identical prompt token totals at every tier** -- 249,955 / 652,955 /
699,955, each server tokenizing the same prompt string independently. That is the evidence that
they were given the same input.

**Prefill goes to vLLM at every tier**, by 1.17x at 250k, 1.32x at 653k and 1.32x at 700k:
3,158.5 / 1,845.3 / 1,738.8 tok/s against NInfer's 2,700.5 / 1,401.7 / 1,322.5. The most likely
single cause is prefill chunking -- vLLM batches up to 16,768 tokens per prefill step, NInfer
1,024 -- and that is a tunable rather than a ceiling. It was not swept.

**Decode goes to NInfer, and the cause is vLLM's speculative acceptance past its native window.**
That deployment's native window is 262,144 tokens; beyond it, its MTP head still drafts 3 tokens
per step and has every one rejected. Acceptance measured from each engine's own counters, for
exactly the requests in the table:

| Tier | vLLM drafted / accepted | vLLM acceptance | NInfer drafted / accepted | NInfer acceptance |
|---|---:|---:|---:|---:|
| 250k | 606 / 311 | 51.3% | 567 / 322 | 56.8% |
| 653k | 1,533 / **0** | **0.0%** | 547 / 328 | 60.0% |
| 700k | 1,533 / **0** | **0.0%** | 602 / 310 | 51.5% |

So vLLM pays the drafter's cost for nothing at 653k and 700k, and its decode falls to 41.94 and
41.01 tok/s. NInfer's acceptance is flat across the same range, and its MTP3 decode is **2.83x and
2.52x** vLLM's there (118.74 and 103.42 tok/s). NInfer's MTP-*off* decode at those tiers (56.70 and
54.84 tok/s) already beats vLLM's speculative decode. At 250k, where vLLM's speculation still
works, vLLM decodes 110.78 tok/s against NInfer's 155.80 with MTP3 and 73.35 with MTP off.

**Context ceiling and memory.** vLLM's KV pool measured 759,297 tokens at boot (12.73 GiB, FP8,
`--gpu-memory-utilization 0.85`), with 28.90-29.00 GiB held per GPU. NInfer holds 1,048,576 tokens
-- 38% more window -- in 27.41 GiB per GPU with MTP off and 28.84 GiB with MTP3. The 700k tier sits
near vLLM's ceiling and comfortably inside NInfer's. A second boot of the same vLLM configuration
measured 760,847 tokens; the pool varies by about 0.2% between boots with the free-memory profile
at launch.

**End to end at 512 output tokens, vLLM finishes first at every tier**, because a request of that
shape is almost entirely prefill. NInfer's decode advantage repays its slower prefill beyond
roughly **4,800 output tokens at 250k, 7,600 at 653k and 8,800 at 700k** (NInfer MTP3 against
vLLM). The Qwen3.8 card's own guidance for a 1M window -- up to 262k tokens of reasoning and 131k
of final response on agentic tasks -- sits far above all three break-even points.

#### Caveats on the cross-engine rows

- **Different weights.** vLLM served `unsloth/Qwen3.8-27B-NVFP4`, an NVFP4 quantization of the base
  Qwen3.8-27B fine-tune; NInfer served its own conversion of the huihui abliterated fine-tune.
  These are different quantizations of different fine-tunes. The comparison is *engine plus
  quantization pipeline*, not a controlled same-weights benchmark. Architecture, layer count and
  hidden sizes are identical, so the prefill and decode arithmetic has the same shape, but nothing
  here isolates the engine from the checkpoint.
- **Different KV dtypes.** vLLM FP8, NInfer INT8. That affects both the memory rows and attention
  bandwidth, so it is present in both the prefill and the decode columns.
- **Different prefill chunking, unswept.** vLLM `--max-num-batched-tokens 16768` against NInfer
  `--prefill-chunk 1024`, the value the 1M configuration ships with. Neither was swept.
- **No vLLM MTP-off row.** Speculative decoding is fixed at launch and turning it off needs a
  restart with a different `--speculative-config`; that run was not made. vLLM's 653k and 700k rows
  are therefore MTP-off *behaviour* at MTP-on *cost*, which is worse than a true MTP-off run would
  be.
- **No quality claim.** Both engines ran at `temperature 0`, `seed 42`, 512 max tokens, but their
  rejection-sampling paths under speculative decoding are not guaranteed identical and no
  token-level equivalence was checked. This is a throughput comparison only.
- **n = 1 per cell.** Each row is a single request.
- **The 500 W NInfer 250k rows are not a like-for-like re-run of the 575 W 250k rows.** These ran
  YaRN at a 1,048,576-token window, to match vLLM's YaRN deployment; the 575 W rows ran native rope
  at 262,144. The roughly 2% difference between them combines the lower cap with the window change
  and does not separate the two.
- **The vLLM client needed a wrapper.** That server runs `--reasoning-parser qwen3`, which routes
  output to `delta.reasoning_content`, and it ignores a top-level `enable_thinking` field (its
  equivalent is `chat_template_kwargs`). The project probe reads `delta.content` and sends the
  top-level field, so the vLLM rows were taken with a wrapper client that times the first delta on
  either channel and sends `chat_template_kwargs={"enable_thinking": false}`. Every vLLM row
  reports its tokens arriving on the `content` channel, so thinking was off on both sides.

Two further points about vLLM's extended context, independent of the table:

- **Extended context in vLLM is a checkpoint property, not a serving flag.** An NVFP4 repackaging
  that ships without a YaRN block in `config.json` `rope_parameters` is capped at its
  `max_position_embeddings` (262,144) until one is injected at load. `--hf-overrides` does exactly
  that, and vLLM then serves this checkpoint far beyond 262,144 tokens. The difference from NInfer
  is **where the configuration lives** -- a serving flag on an unmodified artifact here, a
  checkpoint-config override there -- not a capability difference.
- **A prefix-cache measurement, not a comparison row.** One extra vLLM run replayed the identical
  250k prompt on the same server: 248,000 of 249,955 tokens served from the prefix cache (99.22%),
  TTFT 1.80 s instead of 79.14 s, decode unchanged at 108.75 tok/s.

The raw probe records, per-request server logs, power and VRAM samples, speculative counters and
the full methodology are committed under
[`eval/results/cross-engine-nvfp4/`](../eval/results/cross-engine-nvfp4/README.md).

### Measurement gaps in this campaign

- **Concurrency C=2 was measured at the 400 W cap only**; the 575 W re-measurement covered C=1 and
  C=4, and the 653k extended-context tier was not re-measured at 575 W.
- **The soak is greedy only.** A seeded temperature > 0 soak, as the decode-coverage complement, has
  not been run.
- **The cross-engine comparison has no vLLM MTP-off row**, because speculative decoding is fixed at
  vLLM's launch and that restart was not made, and **the prefill-chunk difference was not swept** on
  either engine (`--prefill-chunk 1024` against `--max-num-batched-tokens 16768`).

The design decisions and correctness gates behind these numbers are in
[Dual-GPU (TP2) execution and YaRN 1M context](maintainer/tp2-yarn-1m.md).

## Reproduction

Build `ninfer-serve` and prepare the registered `.ninfer` artifacts. The refreshed per-target
serving tables use:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --mode mtp3 --suite corpus-makespan --concurrency 1 \
  --max-context 262144 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_35b_mtp3_20260811

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --mode mtp3 \
  --output profiles/bench/serve_corpus_27b_mtp3_20260724

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp0 --sampling stochastic \
  --output profiles/bench/serve_corpus_27b_nvfp4_w8_20260731

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan --concurrency 1 \
  --max-context 131072 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_27b_nvfp4_mtp3_20260811

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp0 --sampling stochastic \
  --output profiles/bench/serve_corpus_qwen3_8_27b_nvfp4_mtp0_20260817

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --max-context 131072 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_qwen3_8_27b_nvfp4_mtp3_20260817
```

The concurrent decode-saturation campaigns use:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_27b_mtp3_20260811

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_27b_nvfp4_mtp3_20260811

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_35b_mtp3_20260811
```

Use `--mode dflash7` for the corresponding DFlash block=8 campaign; add `--sampling greedy` for
the exact-argmax profile.

Omit `--mode` and supply the two measured Qwen3.6 groupwise-int artifacts to run the complete
published Qwen3.6 MTP0/MTP3 campaign:

```bash
python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --output profiles/bench/serve_corpus_20260720
```

For the 27B NVFP4 accuracy run, start the model service with:

```bash
build/apps/ninfer-serve out/qwen3_6_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 18080 \
  --max-context 262144 --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

Then run the repository's full 27B reasoning suite in a separate shell:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m ninfer_eval run \
  --config eval/configs/qwen3_6_27b_reasoning.yaml \
  --suite reasoning_full
```

## `qwen3_6_35b_a3b`

### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 15,544.3 ± 242.4 | 500.2 ± 7.8 | 271.1 ± 3.6 |
| 64,512 | 5 | 10,809.0 ± 95.3 | 6,009.9 ± 52.6 | 242.9 ± 1.3 |
| 130,048 | 5 | 7,828.4 ± 34.1 | 16,693.3 ± 71.2 | 219.4 ± 1.6 |
| 260,096 | 5 | 5,157.1 ± 52.4 | 50,598.8 ± 519.7 | 188.2 ± 2.1 |

### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 8,223.0 ± 2,224.1 | 726.2 ± 22.9 | 82.8% ± 3.4% | 3.48 ± 0.10 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 620.3 ± 8.1 | 72.7% ± 1.4% | 3.18 ± 0.04 |
| `long_decode_aime26_30` | 5 | 52,977.8 ± 11,849.6 | 671.9 ± 8.8 | 80.1% ± 2.7% | 3.40 ± 0.08 |

### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 657.6 ± 34.3 | 70.3% ± 5.5% | 3.11 ± 0.16 |
| Story | 15 | 456.2 ± 36.6 | 38.0% ± 6.0% | 2.14 ± 0.18 |
| Translation | 15 | 649.7 ± 33.0 | 67.6% ± 5.1% | 3.03 ± 0.15 |
| Structured | 15 | 770.9 ± 29.3 | 89.1% ± 4.9% | 3.67 ± 0.15 |

### DFlash block=8 (`k=7`), stochastic sampling

The fixtures, five seeds, sampling parameters, and output limits are identical to MTP3. Different
speculative backends consume random values differently, so this is a fixed-workload comparison
rather than a token-identical paired-output comparison.

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 8,495.4 ± 2,221.2 | 764.1 ± 55.6 | 65.2% ± 5.4% | 5.56 ± 0.38 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 584.0 ± 33.3 | 51.1% ± 3.7% | 4.58 ± 0.26 |
| `long_decode_aime26_30` | 5 | 53,330.4 ± 11,198.5 | 638.3 ± 15.8 | 56.4% ± 2.5% | 4.95 ± 0.17 |

#### Cross-scenario decode

| Category | Samples | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 562.3 ± 36.2 | 43.0% ± 3.7% | 4.01 ± 0.26 |
| Story | 15 | 261.7 ± 51.1 | 12.1% ± 5.3% | 1.85 ± 0.37 |
| Translation | 15 | 490.8 ± 62.6 | 34.8% ± 6.3% | 3.44 ± 0.44 |
| Structured | 15 | 786.4 ± 124.7 | 66.5% ± 13.5% | 5.66 ± 0.94 |

#### Decode throughput versus MTP3

| Workload | MTP3 tok/s | DFlash tok/s | DFlash change |
|---|---:|---:|---:|
| `long_decode_aime26_01` | 726.2 | 764.1 | +5.2% |
| `long_decode_aime26_15` | 620.3 | 584.0 | -5.9% |
| `long_decode_aime26_30` | 671.9 | 638.3 | -5.0% |
| Code | 657.6 | 562.3 | -14.5% |
| Story | 456.2 | 261.7 | -42.6% |
| Translation | 649.7 | 490.8 | -24.5% |
| Structured | 770.9 | 786.4 | +2.0% |

### DFlash block=8 (`k=7`), greedy sampling

Greedy uses exact argmax; all other corpus and server settings remain unchanged. The five seeds
repeat the same deterministic generation path, so within-fixture standard deviation measures
runtime variation rather than output variation.

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 6,692.0 ± 0.0 | 872.4 ± 3.3 | 74.4% ± 0.0% | 6.21 ± 0.00 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 651.6 ± 0.6 | 58.6% ± 0.0% | 5.10 ± 0.00 |
| `long_decode_aime26_30` | 5 | 65,536.0 ± 0.0 | 994.9 ± 3.4 † | 98.0% ± 0.0% | 7.86 ± 0.00 |

† The generation is a deterministic repetition loop, not a valid AIME response. The raw rate is
retained to describe what was measured, but is excluded from performance comparisons.

#### Cross-scenario decode

| Category | Samples | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 599.8 ± 12.3 | 46.4% ± 1.4% | 4.25 ± 0.10 |
| Story | 15 | 291.5 ± 55.6 | 14.9% ± 5.7% | 2.04 ± 0.40 |
| Translation | 15 | 475.5 ± 50.6 | 33.0% ± 5.1% | 3.31 ± 0.36 |
| Structured | 15 | 869.0 ± 120.2 | 74.5% ± 13.1% | 6.21 ± 0.92 |

#### Decode throughput versus stochastic DFlash

| Workload | Stochastic tok/s | Greedy tok/s | Greedy change |
|---|---:|---:|---:|
| `long_decode_aime26_01` | 764.1 | 872.4 | +14.2% |
| `long_decode_aime26_15` | 584.0 | 651.6 | +11.6% |
| `long_decode_aime26_30` | 638.3 | 994.9 † | not comparable † |
| Code | 562.3 | 599.8 | +6.7% |
| Story | 261.7 | 291.5 | +11.4% |
| Translation | 490.8 | 475.5 | -3.1% |
| Structured | 786.4 | 869.0 | +10.5% |

### Speculative-decode output audit

The audit covers all 225 stored July responses from the 35B-A3B MTP3 stochastic-sampler, DFlash
stochastic-sampler, and DFlash greedy campaigns. It checks termination, exact repetition, and
fixture-specific mechanical constraints. AIME 1 was checked algebraically; the AIME 30 answer
(`393`) was checked by independent enumeration. This audit does not attempt to assign a subjective
quality score to prose or translations.

#### Long-reasoning answers

| Fixture | MTP3 stochastic sampler | DFlash stochastic sampler | DFlash greedy |
|---|---|---|---|
| `long_decode_aime26_01` | 5/5 correct, natural stop | 5/5 correct, natural stop | 5/5 correct, natural stop |
| `long_decode_aime26_15` | 0/5 answers; all reach 65,536-token limit | 0/5 answers; all reach 65,536-token limit | 0/5 answers; all reach 65,536-token limit |
| `long_decode_aime26_30` | 3/5 correct, 1 wrong, 1 no answer | 2/5 correct, 1 wrong, 2 no answer | 0/5 answers; all enter the same repetition loop |

The greedy AIME 30 response has an empty final-content field and fills its 65,536-token reasoning
budget. The exact line `Wait, $x_7 x_1 x_3$ is $x_7 x_1 x_3$.` occurs 2,406 times among 2,538
non-empty reasoning lines. Its 98.0% acceptance and 994.9 tok/s therefore characterize a highly
predictable pathological loop, not normal reasoning performance.

AIME 15 is also not a valid completion in any of the three campaigns: every sample exhausts the
budget without a boxed answer. Its output is long, non-convergent reasoning rather than the short
exact cycle seen in greedy AIME 30. The AIME 15 rates may be read only as sustained long-decode
throughput.

#### Cross-scenario outputs

| Category | MTP3 stochastic sampler | DFlash stochastic sampler | DFlash greedy |
|---|---|---|---|
| Code | 1/15 natural stops; 0/15 prompt-complete | 2/15 natural stops; 0/15 prompt-complete | 0/15 natural stops |
| Story | 9/15 natural stops; the nine Chinese outputs pass requested division and minimum length | 8/15 natural stops; the eight Chinese outputs pass requested division and minimum length | 10/15 natural stops; five Chinese dialogue outputs are under length |
| Translation | 15/15 natural stops; 15/15 pass structural checks | 15/15 natural stops; 15/15 pass structural checks | 15/15 natural stops; 15/15 pass structural checks |
| Structured | 0/15 satisfy the requested complete record/script contract | 0/15 satisfy the requested complete record/script contract | 0/15 satisfy the requested complete record/script contract |

The code prompts require complete runnable multi-file deliverables, but almost all outputs end at the
4,096-token limit. The three natural-stop exceptions also contain decisive contract failures: the
MTP3 CUDA response substitutes CUDA 12.8 and an older architecture list; the DFlash CUDA response
copies FP32 input into a half-sized 16-bit allocation and passes raw `unsigned short` values to BF16
intrinsics; and the DFlash Python response never writes its advertised JSONL event stream to the
configured log file. Code throughput is therefore a truncated-generation stress result, not
successful code-generation throughput.

All English mystery samples reach the output limit with an unfinished ending. The naturally stopped
Chinese stories have the requested chapter/act counts; the MTP3 and stochastic-DFlash samples also
meet their requested Chinese-character minima. Greedy's five dialogue stories contain 3,239 Chinese
characters each, below the requested 3,500. Story results are consequently a mixed normal/truncated
workload.

All translation outputs stop naturally. Each plain-document result preserves six sections and
provides at least twenty glossary entries; each Markdown result preserves heading levels, the
six-line table, all required inline identifiers, and the exact fenced JSON object. Translation is
the cleanest cross-scenario normal-completion comparison in this corpus.

The structured prompts intentionally exceed what these generations fit into 4,096 tokens. MTP3,
stochastic DFlash, and greedy DFlash produce only 49–60, 49–58, and 57 valid JSONL records,
respectively, versus the requested 160. Their complete-width CSV ranges are 122–139, 121–143, and
133 rows versus the requested 220. No SQL output satisfies all four tables, two views, at least 80
rows, and six final analytical queries. These high-acceptance results describe predictable partial
record generation only.

The exact-line and repeated-token scan found no other response with a short-cycle collapse comparable
to greedy AIME 30. Output-limit and prompt-compliance failures above remain material even when no
repetition loop is present.

## `qwen3_6_27b`

### EvalScope reasoning accuracy

Both weight profiles were evaluated through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP=3, and a 262,144-token context limit. EvalScope 1.9.0 used 0-shot prompts, rule-based
scoring, and one sample per problem with temperature 0.6, top-p 0.95, top-k 20, presence penalty
1.0, and seed 42. All 258 samples completed and were scored for each profile.

| Weights ID | AIME 2025 | AIME 2026 | GPQA-Diamond |
|---|---:|---:|---:|
| `groupwise-int` | 86.67% (26 / 30) | 93.33% (28 / 30) | 86.87% (172 / 198) |
| `nvfp4` | 93.33% (28 / 30) | 93.33% (28 / 30) | 84.34% (167 / 198) |

These are single-sample results under the stated evaluation profile, not pass@k scores. Each
benchmark remains independently reportable; no combined score is computed.

### `groupwise-int`

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 3,218.1 ± 4.3 | 2,392.4 ± 3.0 | 77.6 ± 0.1 |
| 64,512 | 5 | 2,655.9 ± 2.9 | 24,335.7 ± 25.2 | 70.7 ± 0.1 |
| 130,048 | 5 | 2,185.3 ± 0.3 | 59,590.3 ± 8.9 | 64.5 ± 0.1 |
| 260,096 | 5 | 1,614.8 ± 0.6 | 161,221.8 ± 62.5 | 54.8 ± 0.1 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 10,686.2 ± 553.8 | 175.4 ± 1.0 | 77.9% ± 0.9% | 3.34 ± 0.03 |
| `long_decode_aime26_15` | 5 | 61,604.2 ± 5,677.9 | 161.9 ± 2.8 | 73.4% ± 1.7% | 3.20 ± 0.05 |
| `long_decode_aime26_30` | 5 | 47,339.8 ± 9,162.2 | 172.2 ± 0.9 | 78.8% ± 0.8% | 3.36 ± 0.02 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 167.0 ± 5.4 | 72.3% ± 3.5% | 3.17 ± 0.11 |
| Story | 15 | 112.6 ± 9.4 | 37.8% ± 5.9% | 2.13 ± 0.18 |
| Translation | 15 | 161.5 ± 11.3 | 68.3% ± 7.2% | 3.05 ± 0.22 |
| Structured | 15 | 193.0 ± 18.8 | 88.7% ± 11.7% | 3.66 ± 0.35 |

### `nvfp4`

The fixtures, seeds, sampling parameters, output limits, and runtime options are identical to the
groupwise-int serving campaign. Quantization can change sampled tokens, so the MTP3 results are a
fixed-workload comparison rather than a token-identical output comparison.

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 11,191.5 ± 70.2 | 692.5 ± 4.3 | 86.4 ± 0.5 |
| 64,512 | 5 | 6,298.5 ± 97.6 | 10,288.6 ± 159.3 | 78.0 ± 1.2 |
| 130,048 | 5 | 4,204.7 ± 14.1 | 31,012.5 ± 104.6 | 71.2 ± 0.2 |
| 260,096 | 5 | 2,510.6 ± 16.8 | 103,761.1 ± 698.8 | 59.9 ± 0.3 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 12,053.4 ± 820.9 | 231.0 ± 3.0 | 80.2% ± 1.2% | 3.41 ± 0.04 |
| `long_decode_aime26_15` | 5 | 63,109.0 ± 5,426.9 | 213.1 ± 4.2 | 76.3% ± 2.0% | 3.29 ± 0.06 |
| `long_decode_aime26_30` | 5 | 57,166.4 ± 9,204.9 | 223.3 ± 1.8 | 81.1% ± 1.5% | 3.43 ± 0.04 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 220.3 ± 8.2 | 74.2% ± 4.0% | 3.23 ± 0.12 |
| Story | 15 | 148.8 ± 11.6 | 39.2% ± 5.7% | 2.18 ± 0.17 |
| Translation | 15 | 213.6 ± 12.2 | 70.5% ± 6.0% | 3.12 ± 0.18 |
| Structured | 15 | 252.2 ± 16.3 | 89.8% ± 8.0% | 3.69 ± 0.24 |

The baseline and speculative-decode suites intentionally measure different supported workloads.
No per-scenario baseline/speculative speedup is reported.

## `qwen3_8_27b`

### `nvfp4`

The MTP0 table comes from the serial Long NIAH campaign described by the single-request method. The
MTP3 tables come from the C=1 point of the fixed concurrent-corpus campaign, which serially runs the
same three long-reasoning and twelve cross-scenario fixtures. Each fixture has five fixed seeds. The
tables report arithmetic mean ± sample standard deviation from the server's per-request phase
timings and speculative counters.

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 8,340.4 ± 13.0 | 931.6 ± 1.6 | 71.2 ± 0.1 |
| 64,512 | 5 | 5,297.9 ± 259.2 | 12,281.1 ± 561.5 | 65.7 ± 0.8 |
| 130,048 | 5 | 3,544.7 ± 25.3 | 36,853.5 ± 259.4 | 59.6 ± 0.9 |
| 260,096 | 5 | 2,203.1 ± 13.4 | 118,354.8 ± 717.2 | 52.9 ± 2.3 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 1,465.4 ± 417.3 | 195.2 ± 4.6 | 76.0% ± 2.4% | 3.28 ± 0.07 |
| `long_decode_aime26_15` | 5 | 65,414.4 ± 271.9 | 151.4 ± 2.0 | 56.2% ± 1.1% | 2.69 ± 0.03 |
| `long_decode_aime26_30` | 5 | 50,023.4 ± 14,839.1 | 167.5 ± 23.7 | 64.6% ± 14.9% | 2.94 ± 0.45 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 194.3 ± 6.1 | 76.4% ± 3.9% | 3.29 ± 0.12 |
| Story | 15 | 126.1 ± 10.9 | 37.4% ± 5.8% | 2.12 ± 0.17 |
| Translation | 15 | 192.3 ± 11.9 | 75.0% ± 6.5% | 3.25 ± 0.19 |
| Structured | 15 | 219.8 ± 8.6 | 90.8% ± 5.1% | 3.72 ± 0.15 |
