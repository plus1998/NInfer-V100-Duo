# Tests

The retained tests protect current `.ninfer`, numerical operator, target, runtime-transaction,
benchmark-report, and external protocol behavior. Repository verification principles are defined in
[`../AGENTS.md`](../AGENTS.md); Op contract and CUDA implementation guidance is in
[`../docs/maintainer/op-development.md`](../docs/maintainer/op-development.md).

## Organization

- `artifact/` — Python container, registered layout, quantization, and resource behavior;
- `ops/` — one identifiable qualification suite per semantic Op or closely related overload group,
  using independent numerical/state-transition oracles at real supported shapes;
- `ops/linear/` — weight/activation-profile-specific public Linear conformance tests plus their
  one shared input generator, FP64 GEMM oracle, tolerance registry, and output/effects mechanics;
- `ops/linear_add/`, `ops/linear_pair/`, `ops/linear_swiglu/` — fused-Op suites split by registered
  weight/activation profile, each evaluating its complete formula rather than composing production
  Ops;
- `ops/test_allreduce.cpp` — the two-device `allreduce_sum`/`allgather_rows` collectives; it needs
  two CUDA devices visible to one process and reports the shared skip code when fewer are present;
- `targets/qwen3_6/` — shared tokenizer/template, multimodal preprocessing, MRoPE, prepared-prompt,
  stop/output decoding, hybrid topology, decoder/GDN and round-state layouts/views, shifted-MTP
  alignment, Vision control, and family runtime mechanisms;
- `targets/qwen3_6_27b/` — registered inventory, converter recipe, source verifier, artifact
  bindings, reference diagnostics, family Program/multimodal/MTP behavior, and the opt-in real-Engine
  prefix test;
- `targets/qwen3_6_35b_a3b/` — registered inventory/converter contracts, artifact-native diagnostic
  reference, MoE oracle, typed binding, selected-expert row access, 256K INT8 memory calculation,
  and the opt-in real public-Engine route;
- `test_ninfer_artifact_reader.cpp` — C++ framing, directory, encoded-size, payload-span, and
  geometry behavior against a self-contained C++ fixture, plus the version-3 acceptance rule:
  a synthetic file with a foreign `metadata.name` reaches the projection while another layout is
  rejected;
- `test_request_memory.cpp` — startup-frozen request-transient capacity, stable address,
  activation alignment, rejection, and peak semantics;
- `test_openai_schema.cpp`, `test_responses_schema.cpp`, `test_response_store.cpp`,
  `test_anthropic_schema.cpp`, and `test_tool_call_parser.cpp` — current protocol translation,
  Responses Item/state/SSE behavior, and incremental tool-call behavior;
- `test_request_log.cpp` and `test_http_error_handler.cpp` — generation lifecycle records,
  preparation rejections, protocol-shaped payload-limit errors, and application-error preservation;
- `test_ninfer_bench_support.cpp` — product benchmark CLI, timing boundary, and schema-v9 reports;
- `test_bench_matrix.py` — schema-v9 report consumption by the Python matrix summarizer;
- `test_serve_corpus.py` — serving request-log schema compatibility at the measurement consumer;
- device/tensor/arena tests — reusable lower-component behavior; KV tests cover the core physical
  container, family runtime tests cover dimension-driven GDN storage/view mechanics, and Op tests
  cover mathematical state transitions at their own boundary.

Tests are grouped by observable risk, not by mirroring every source file or class.
`ops/op_tester.h` and `ops/op_check.h` own only reusable device/guard and comparison mechanics.
Concrete numerical criteria remain named by the semantic Op suite; there are no cross-Op tolerance
presets.

`ops/quantized_weight.h` is the common packed-weight fixture for Q4/Q5/Q6/W8 and NVFP4 Op tests. It
owns deterministic payload generation, device `Weight` views, row views, and independent logical
weight decoding.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Run a focused target for a localized change:

```bash
cmake --build build --parallel --target ninfer_sampling_test
ctest --test-dir build -R ninfer_sampling_test --output-on-failure
```

Enable uniform floating-point error records when establishing or reviewing an Op criterion:

```bash
NINFER_OP_REPORT_STATS=1 \
  ctest --test-dir build -V -R '^ninfer_(rmsnorm|gqa_attention)_test$'
```

Every participating comparison emits one `OP_ERROR_STATS` record containing the stable case label,
actual error, active limit, and error-to-limit ratio. The switch changes reporting only; the same
statistics still drive the normal verdict. Passing tests remain quiet without it.

Linear tests are independently runnable by weight and activation-compute profile:

```bash
cmake --build build --parallel --target \
  ninfer_linear_q4_a16_test ninfer_linear_q5_a16_test \
  ninfer_linear_q6_a16_test ninfer_linear_w8_a16_test
ctest --test-dir build -R '^ninfer_linear_(q4|q5|q6|w8)_a16_test$' --output-on-failure
```

All Linear files use `ops/linear/linear_test_common.{h,cpp}` and the same
`ops/quantized_weight.h` fixture as the fused projection tests. The fixture produces the complete
packed GPU payload and exact-decodes the logical float rows used by the one
`cpu_linear_gemm_fp64()` reference. The reference performs naive double accumulation and never
reproduces a production route's activation quantization, staging, reduction tree, or BF16 output
rounding. Each activation compute path selects one centrally defined comparison tolerance for its
whole suite; private kernel, schedule, launcher, and T selection do not change it. Individual test
files call public `linear()` and contain no private selector, launcher, schedule, or kernel
assertions.

Run the native Python suites with the project Python environment:

```bash
python3 -m pytest \
  tests/artifact tests/targets/qwen3_6_27b tests/targets/qwen3_6_35b_a3b \
  tests/test_bench_matrix.py tests/test_serve_corpus.py
```

The Python binding tests use `NINFER_QWEN3_6_27B_ARTIFACT` when set, otherwise they look for
`out/qwen3_6_27b.ninfer`. They report a pytest skip when neither path provides the real
artifact. The 35B-A3B reference binding test follows the same rule with
`NINFER_QWEN3_6_35B_A3B_ARTIFACT` and `out/qwen3_6_35b_a3b.ninfer`. The remaining Python
target tests still run without either artifact.

The C++ prefix/MTP integration test is separately opt-in because it loads the full artifact and
runs the real engine:

```bash
NINFER_QWEN3_6_27B_WEIGHTS=$PWD/out/qwen3_6_27b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_27b_prefix_real_test --output-on-failure
```

For TP2 Vision on the official Qwen3.8 NVFP4 artifact, build
`ninfer_qwen3_6_27b_prefix_real_test` and run the retained two-device variant:

```bash
NINFER_QWEN3_6_27B_NVFP4_WEIGHTS=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build-v100-duo -R '^ninfer_qwen3_6_27b_vision_tp2_real_test$' --output-on-failure
```

It checks real multimodal prefill, cross-chunk image lifetime, shifted MTP alignment,
same-media reuse, changed/appended media, and visual prefix bridges. To verify actual concurrent
HTTP generation, start the README's Vision profile with `--port 18081 --no-thinking --greedy
--request-log-jsonl /tmp/ninfer-vision.jsonl --log-stats-interval-ms 500`, then run with Python 3.11:

```bash
/path/to/python3.11 tools/v100/check_vision_http.py --request-log /tmp/ninfer-vision.jsonl
```

The HTTP check generates exact red-circle/blue-square PPM scenes on white backgrounds, verifies
their reported colors and geometry (not exact wording), tests a 1,630-token cross-chunk prompt
and two images, and sends two simultaneous streaming requests. It requires overlapping content
streams and structured-log evidence of an actual multi-row decode batch, not just two HTTP 200s.
Use a dedicated idle server with a 1,024-token prefill chunk and at least two request slots.

For version-3 artifacts, `NINFER_QWEN3_8_27B_WEIGHTS=/absolute/path/to/qwen3_8_27b_nvfp4.ninfer`
and `NINFER_QWEN3_8_27B_SWIFT_WEIGHTS=/absolute/path/to/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4.ninfer`
each run the reader's projection checks for that file: registered inventory, tensor formats, and
chat-template projection. Swift-1.5 is a third-party conversion whose MLP layers are all NVFP4
rather than the official converter's NVFP4/FP8 split, so its projected inventory differs while its
identity stays `qwen3.8-27b/nvfp4`. Without either variable the test still checks version-3
acceptance on a synthetic fixture: a file declaring a foreign `metadata.name` reaches the
projection, and a different layout is rejected naming the offending field.

For the GSQ-RCO v3 artifact, `NINFER_QWEN3_8_27B_GSQ_WEIGHTS=/absolute/path/to/artifact.ninfer`
enables the artifact reader's Vision and DFlash2 companion projection checks and the
`ninfer_qwen3_8_27b_dflash2_binding_test` (66 tensors validated in TP1/TP2; the ordinary
Engine leaves them nonresident). With two CUDA devices, the binding test separately uploads
their TP2 slices and samples their bytes against the source artifact. It additionally selects
the companion without MTP, materializes the complete Vision + Text + DFlash2 load plan on two
GPUs and checks both rank-local model views.
It then exercises the two-rank feature projection, normalization, and five-layer context KV
materialization with the real companion across three concurrent state slots, checking matching
nonzero projections and the written/unwritten cyclic slots. It binds a planned BF16 K/FP16 V
cache, drives the TP2 feature-capture prefill consumer, and checks per-lane checkpoint save/restore.
The test also exercises the 27B Program's optional two-rank persistent layout and state
binding, including the disabled path and accounted cyclic KV bytes.
The same real-weight test checks partial and zero pending-prefix commits across three slots,
uncommitted cyclic positions, and a dual-device pending-materialization Graph replay.
It also runs the five-layer TP2 draft backbone and both GGUF proposal heads on the real weights,
followed by top-16 candidate merging, three-request greedy/stochastic selector walks, and
two GGUF full-head dual-device Graph replays. The proposal retains global candidate IDs
alongside their sparse probabilities and publishes drafts/IDs/probabilities to rank 1;
the real-weight test verifies the selected ID against its probability support and checks
both ranks' verifier inputs in eager execution and Graph replay. The standalone
`ninfer_linear_topk_test` checks W8/Q4 TP2 global ids and projected scores against independently
decoded FP64 dots, including the padded full-head tail, mapped shortlist ids and cross-device
Graph replay;
`ninfer_linear_bf16_a16_test` covers the selector's 256-by-5120 hidden projection on V100.
This is an integration smoke test, not an independent numerical oracle for the
five-layer draft. `ninfer_qwen3_8_27b_dflash2_real_test` exercises the real
Engine decode path. For first-position acceptance diagnosis, set
`NINFER_DFLASH2_INSPECT_FIRST=1` and `NINFER_DFLASH2_PROFILE=1` with the
explicit `NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS` artifact path, then run
`build-v100-duo/tests/ninfer_qwen3_8_27b_dflash2_real_test 3 1 1 1 int8 0`.
The same test accepts the explicit NVFP4 v3 artifact path with an embedded
companion and its optimized Q4 proposal head. Append `0 1` after `int8 0`
for three unprofiled 127-token decode intervals each for ordinary, MTP3 and
DFlash2; the ordinary baseline runs under the same Graph and KV settings.
For a three-repetition concurrent throughput probe, append `0 1 1` after
`int8 0` and set the fourth positional argument to `3`; the final `1` skips
the MTP3 control when its B=3 Graph memory allowance is insufficient. This
probe measures wall-clock aggregate output rate including prefill, not the
single-request decode-only rate.
Each `dflash2-first` stderr line reports one active round's lane, position,
target greedy argmax, selected draft, rank of target in the unary top-16
(`-1` if absent), accepted prefix length, BF16 target-logit difference
`target_minus_draft`, and candidate IDs. The test performs
a 16-token warmup before its 128-token probe; the number of warmup rounds
depends on the prompt, so do not include warmup lines in
the probe statistics. This diagnostic reads back device tensors and prints per
round; never use its elapsed time as a speed result.
For the first-position edge calculation, additionally set
`NINFER_DFLASH2_INSPECT_EDGE=1` and use eager mode (`3 0 1 1 int8 0` in
the invocation above). With batch size one and no CUDA Graph capture,
`dflash2-edge` reports the anchor, selected first token, and each unary
candidate's FP32 score and CPU FP64 recomputation of its edge from the
stored BF16 codebooks and projected hidden. The edge readback synchronizes
and copies device tensors per candidate; it does not run inside Graph
capture, qualify the five-layer draft against a checkpoint oracle, or
provide timing evidence.
`NINFER_DFLASH2_PROFILE_TEXT` optionally replaces the counting prompt in this
test's profiling fixture; compare only runs using identical text and settings.
The README's GSQ Vision three-slot server profile also supports the same real-image
HTTP checker, with
`--base-url` pointing to the selected port.

On V100 (`sm_70`), `ninfer_candidate_selector_test`,
`ninfer_dynamic_grouped_conv_prepare_test`, `ninfer_linear_dynamic_grouped_conv_add_test`,
`ninfer_dflash2_tp2_conv_finish_test`, `ninfer_rmsnorm_rope_test`, and
`ninfer_context_kv_materialize_test` qualify DFlash2 Ops
against independent numerical/state oracles; the context KV test covers both full eight-head
and TP2 four-head weights/caches. They do not qualify a DFlash2 Engine route;
the 27B Program still rejects this backend.
`ninfer_speculative_sparse_test` qualifies the DFlash2 target acceptance Op at the small and
248077-token domains, including rejection from positive sparse residual mass, p(d)/q(d)
acceptance (distinct from the one-hot MTP rule), mixed greedy/sampling rows, token counts,
partial extents, K=15/B=8 and CUDA Graph replay. The existing
`ninfer_speculative_round_test` continues to protect MTP's one-hot proposal route.
`ninfer_tp2_feature_capture_test` checks exact packed halves of the five target residual
layers, the split within the middle layer, independent per-rank positions, invalid peer
shapes, incomplete capture, and CUDA Graph replay. The TP2 Text prefill (including Vision
scatter) and target verify now expose these rank-local captures; incomplete capture includes
missing positions on either rank. The DFlash2 Program does
not yet supply their persistent feature buffers or context-consumer callback.
The TP2 conv finish suite checks W8 row shards against a common full matrix, each rank's
projection against independently decoded FP64 dots, and the two-device allreduce/finish/residual
against the complete FP64 formula at attention and MLP widths, including three-way concurrency
and the maximum draft width. It verifies both rank-local residuals agree and preserves inputs.
`ninfer_swa_dflash2_test` qualifies the TP2 16Q/4KV, 2048-window BF16/FP16 SWA route
against an independent FP64 attention oracle, window exclusion, mixed live columns and
CUDA Graph replay. `ninfer_swa_dflash2_bench` measures only this Op, not a complete draft round.
`ninfer_dflash2_tp2_attn_input_test` checks the V100 rank-local W8 QKV fused projection's
three physical sections against independently decoded weights and FP64 dot products at decode,
three-request block widths, the T=32 Tensor Core dispatch boundary, and T=48 CUDA Graph replay.
It also reconstructs both rank-local physical section orders from one full W8 parent and
compares Q/K/V against the independently decoded corresponding full-matrix rows.
`ninfer_linear_w8_a16_test` independently checks both TP2 DFlash2 row-shard
Linear geometries against packed-weight FP64 oracles; on two V100 devices,
`ninfer_linear_dflash2_split_test` exercises feature and attention-output row-parallel
allreduces against their full-matrix reference. The general split suite includes
sm_120a-only NVFP4 A4 execution and cannot complete on V100.

Run the peer 35B-A3B route independently:

```bash
NINFER_QWEN3_6_35B_A3B_WEIGHTS=$PWD/out/qwen3_6_35b_a3b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_35b_a3b_real_test --output-on-failure
```

Without the corresponding variable CTest marks each C++ integration test as skipped. Neither test
uses another numerical/execution path's generated tokens as a golden.

The capability-evaluation coordinator has its own environment and unittest entry point:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover \
  -s eval/tests -p 'test_*.py'
```

Run the serving contract manually after starting a resident server in another terminal:

```bash
./build/apps/ninfer-serve out/qwen3_6_27b.ninfer \
  --host 127.0.0.1 --port 18080
```

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 --model qwen3.6-27b
```

This smoke check is intentionally not a CTest: it needs the real artifact, a supported GPU, and a
server process that remains alive while the client exercises OpenAI Responses/Chat, Anthropic,
state, streaming, and multimodal requests.

The thinking-preservation fixture starts and stops its own server, submits a fixed two-step tool
history, compares restored and cold greedy output, compares stripped and preserved closed-turn
prompt lengths, and verifies turn/response rewrite-checkpoint reuse paths plus Responses
inheritance:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp

python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_35b_a3b.ninfer --backend dflash
```

The shared messages are in
[`fixtures/serve/qwen3_6_thinking_preservation.json`](fixtures/serve/qwen3_6_thinking_preservation.json).

## What belongs here

A permanent test should protect one current risk, such as:

- exact registered artifact bytes, geometry, object binding, or conversion transform;
- a numerical operator contract with an independent oracle;
- family Frontend or Program frontier, prefix, MTP, or multimodal behavior;
- generated-token commit/stop/cancel consistency;
- public benchmark or OpenAI/Anthropic observable behavior;
- a reproduced supported bug.

Performance-only assertions belong in benchmarks and profiler review. Source scans,
implementation-shape assertions, trivial getters/configuration, retired command surfaces, and
broad additions without a concrete regression risk do not belong in the permanent suite.
