# Capability Evaluation

`eval/` contains the repository-local capability evaluation coordinator. It can evaluate this
project's server, another local OpenAI-compatible service, or a remote online model. The inference
engine is only one possible target; its single-sequence limitation is represented by
`max_concurrency: 1`, not built into the framework.

EvalScope is the first real evaluation backend. The coordinator, configuration, logging, progress,
resume, and result contracts do not import or depend on EvalScope. The deterministic `mock` backend
can exercise those contracts without a model service or network access.

## Environment

Create the isolated environment with the repository's canonical Python:

```bash
python3 -m venv eval/.venv
eval/.venv/bin/python -m pip install -r eval/requirements.txt
```

The pinned stack is EvalScope 1.10.0 with its BFCL, IFBench, and Needle-in-a-Haystack extras,
`bfcl-eval==2025.10.27.1`, and the BFCL runtime dependency `soundfile==0.14.0`. Dataset and model
caches remain owned by their upstream libraries. Installing dependencies does not download the
Qwen model or create a `.ninfer` artifact.

## Configuration

See [`configs/capability-suite.yaml`](configs/capability-suite.yaml) for the initial AIME25,
AIME26, GPQA-Diamond, and BFCL-v4 suites, and [`configs/mock-suite.yaml`](configs/mock-suite.yaml)
for a network-free example.

[`configs/qwen3_6_35b_needle_haystack.yaml`](configs/qwen3_6_35b_needle_haystack.yaml)
defines the 35B-A3B Needle-in-a-Haystack profiles separately: `standard` preserves EvalScope's
1K--32K, ten-length, ten-depth English/Chinese matrix (200 samples), while `native_long` evaluates
the exact 64K, 128K, and safe 260K prompt profiles at eleven depths in both languages (66 samples).
The 260K profile uses the exact local 35B tokenizer and leaves more than 2K native context tokens
for chat framing and its bounded 512-token answer. All profiles use rule scoring and explicitly
disable thinking so the observable answer is the retrieved needle.

A target defines the model service:

```yaml
targets:
  model_api:
    protocol: openai_chat
    base_url: http://127.0.0.1:18080/v1
    model: qwen3.6-27b
    api_key_env: MODEL_API_KEY   # optional; omit for an unauthenticated endpoint
    max_concurrency: 1
    request:
      timeout_seconds: 3600
      retries: 2
```

API keys must come from environment variables. Literal `api_key`, `Authorization`, and
`x-api-key` configuration is rejected so secrets cannot enter saved effective configurations.

Concurrency has two levels:

- `runtime.max_parallel_jobs` controls concurrently active dataset jobs;
- target `max_concurrency` caps aggregate requests to that endpoint;
- optional job `max_concurrency` caps how many target slots one job may reserve.

For EvalScope, the granted job slots become `eval_batch_size`. Multiple jobs sharing a target can
never reserve more slots than the target capacity. Set the local `ninfer-serve` target to one; set a
larger explicit value for an online service that supports it.

Portable generation settings live under `generation`. Evaluator-specific controls live under
`backend_args`; unknown fields are rejected rather than silently ignored.

## Commands

Set `PYTHONPATH` because this is a repository-local package:

```bash
export PYTHONPATH="$PWD/eval"
```

Validate configuration and installed runtime dependencies:

```bash
eval/.venv/bin/python -m ninfer_eval validate \
  --config eval/configs/capability-suite.yaml --suite smoke
```

Show expected work without making model requests:

```bash
eval/.venv/bin/python -m ninfer_eval plan \
  --config eval/configs/capability-suite.yaml --suite reasoning
```

Add `--check-runtime` to resolve configured secret environment variables and check pinned backend
packages.

Run the network-free coordinator check:

```bash
eval/.venv/bin/python -m ninfer_eval run \
  --config eval/configs/mock-suite.yaml --suite all
```

Run the small real-endpoint matrix before a formal evaluation:

```bash
eval/.venv/bin/python -m ninfer_eval run \
  --config eval/configs/capability-suite.yaml --suite smoke
```

Then run the full reasoning and BFCL suites independently:

```bash
eval/.venv/bin/python -m ninfer_eval run \
  --config eval/configs/capability-suite.yaml --suite reasoning

SERPAPI_API_KEY=... eval/.venv/bin/python -m ninfer_eval run \
  --config eval/configs/capability-suite.yaml --suite bfcl_full
```

For the Qwen3.8-27B NVFP4 evaluation, first populate EvalScope's default ModelScope dataset cache:

```bash
eval/.venv/bin/python - <<'PY'
from modelscope import dataset_snapshot_download

for dataset_id in (
    'evalscope/ERQA',
    'allenai/IFBench_test',
    'lmms-lab/RealWorldQA',
):
    print(dataset_snapshot_download(dataset_id))
PY
```

The formal run is deliberately split into two independently resumable steps. Inspect the plans,
then run the text step (IFBench, AIME25, AIME26, and GPQA-Diamond) and the multimodal step (ERQA and
RealWorldQA):

```bash
eval/run_qwen3_8_27b_nvfp4_reasoning.sh --plan
eval/run_qwen3_8_27b_nvfp4_reasoning.sh
```

With no step argument the script runs the two steps back to back, restarting the server between
them. Pass `text` or `multimodal` to run a single step only, and `--plan` to preview the plan for
the selected step(s) without starting the server.

The script starts a fresh local server for each step. The text server uses a 252,928-token context
(the largest that fits the RTX 5090 after weights; 262,144 is rejected at startup) and omits
`--vision`, so Vision's fixed GPU allocations do not reduce the KV pool needed by long reasoning.
The multimodal server is restarted with `--vision` and an 81,920-token context. Across the cached
ERQA and RealWorldQA data, the largest fully rendered prompt is ERQA_75 at 12,394 tokens; combined
with the 65,536-token output bound, it leaves 3,990 tokens of context slack. Sampling is specified
only by each EvalScope request. The target and ordinary jobs use concurrency two; GPQA-Diamond runs
at concurrency one so its 245,760-token output budget can accommodate the observed long tail. AIME
uses 122,880 output tokens per request, while IFBench and both multimodal datasets use 65,536.

The completed formal run recorded these scores (run directories `eval/runs/20260818T132336Z-c16a8902`
and `eval/runs/20260818T223812Z-da6cdbce`):

| Benchmark | Accuracy | Correct / total |
|---|---:|---:|
| IFBench (prompt-level strict) | 77.00% | 231 / 300 |
| AIME 2025 | 96.67% | 29 / 30 |
| AIME 2026 | 96.67% | 29 / 30 |
| GPQA-Diamond | 90.40% | 179 / 198 |
| ERQA | 66.25% | 265 / 400 |
| RealWorldQA | 83.53% | 639 / 765 |

The Qwen3.8-27B groupwise-int profile runs the same protocol through
`eval/run_qwen3_8_27b_groupwise_reasoning.sh`. Its 16.96 GiB artifact leaves more GPU memory, so the
text step uses the full 262,144-token context and both steps run at concurrency four (run
directories `eval/runs/20260819T031655Z-078bd8e0` and `eval/runs/20260819T141750Z-531a236a`; the
multimodal step was resumed with `ninfer_eval resume` after a local proxy change interrupted
RealWorldQA at 618/765 samples):

| Benchmark | Accuracy | Correct / total |
|---|---:|---:|
| IFBench (prompt-level strict) | 77.67% | 233 / 300 |
| AIME 2025 | 96.67% | 29 / 30 |
| AIME 2026 | 96.67% | 29 / 30 |
| GPQA-Diamond | 87.37% | 173 / 198 |
| ERQA | 66.25% | 265 / 400 |
| RealWorldQA | 82.22% | 629 / 765 |

Prepare and inspect Needle-in-a-Haystack without issuing model requests:

```bash
eval/.venv/bin/python -m pip install -r eval/requirements.txt
eval/.venv/bin/python - <<'PY'
from modelscope import dataset_snapshot_download
print(dataset_snapshot_download(
    'AI-ModelScope/Needle-in-a-Haystack-Corpus',
    allow_file_pattern=['PaulGraham_Essays.txt', 'Journey_to_the_West.txt'],
))
PY
eval/.venv/bin/python -m ninfer_eval plan \
  --config eval/configs/qwen3_6_35b_needle_haystack.yaml --suite standard --check-runtime
eval/.venv/bin/python -m ninfer_eval plan \
  --config eval/configs/qwen3_6_35b_needle_haystack.yaml --suite native_long --check-runtime
```

Run the one-sample NIAH smoke only after the active model evaluation has released the single target
slot, then select `standard` or `native_long` as a separate formal run.

### Qwen3.8-27B NVFP4 needles across two GPUs (TP2)

[`configs/qwen3_8_27b_nvfp4_needle_haystack.yaml`](configs/qwen3_8_27b_nvfp4_needle_haystack.yaml)
is the dual-RTX-5090 long-context acceptance suite for the NVFP4 artifact.
`eval/run_qwen3_8_27b_nvfp4_needle_tp2.sh` owns one `ninfer-serve --tp 2 --devices 0,1
--max-context 262144 --kv-dtype int8` process per step and runs one suite against it:

```bash
eval/run_qwen3_8_27b_nvfp4_needle_tp2.sh --plan 262k
eval/run_qwen3_8_27b_nvfp4_needle_tp2.sh smoke     # one 8k needle, contract check
eval/run_qwen3_8_27b_nvfp4_needle_tp2.sh 64k       # 10 needles
eval/run_qwen3_8_27b_nvfp4_needle_tp2.sh 128k      # 10 needles
eval/run_qwen3_8_27b_nvfp4_needle_tp2.sh 262k      # 10 needles, ~18 minutes
eval/run_qwen3_8_27b_nvfp4_needle_tp2.sh           # the whole 64k -> 128k -> 262k ladder

# Retrieval-vs-memorization controls at 262k (one sample each, ~3 minutes)
eval/run_qwen3_8_27b_nvfp4_needle_tp2.sh control                 # novel needle, expect score 1
eval/run_qwen3_8_27b_nvfp4_needle_tp2.sh control-absent          # absent question, strict prompt
eval/run_qwen3_8_27b_nvfp4_needle_tp2.sh control-absent-optout   # absent question, NOT FOUND allowed
```

Every step samples GPU memory for as long as its suite runs and prints the peaks it saw, so the
per-GPU VRAM budget is a re-runnable measurement rather than a number someone watched go by. The
samples land in `eval/server-logs/<...>.vram.csv` as `timestamp_utc,kind,key,used_mib`, with one
`kind=gpu` row per device (which includes memory other processes hold) and one `kind=process` row
per `<pid>@gpu<index>` (this server's own residency on that card) per tick.
`NINFER_VRAM_SAMPLE_SECONDS` overrides the 3-second interval. The per-GPU key matters: a TP2 server
appears once per card per tick, and keying on the pid alone would collapse the two cards and hide
an asymmetric split. So the peaks a 262k step prints look like:

```
  peak gpu 0                       16353 MiB  (15.97 GiB, 40 samples)
  peak gpu 1                       15436 MiB  (15.07 GiB, 40 samples)
  peak process 3417924@gpu0        15396 MiB  (15.04 GiB, 40 samples)
  peak process 3417924@gpu1        15396 MiB  (15.04 GiB, 40 samples)
```

Each `native_*` tier is 5 depths (10/30/50/70/90 %) x 2 corpora (English, Chinese) = 10 needles.
The 262k tier builds a 260,000-token haystack, which becomes a 259,954-token prompt and leaves
2,190 tokens of the 262,144-token window for the bounded 512-token answer. Speculative decoding is
off: `--tp 2` and `--spec` are mutually exclusive in the engine.

Scoring is `judge_strategy: rule`, and the config overrides EvalScope's prompt template to ask for
the source sentence copied word for word. EvalScope's stock needle prompt asks for a paraphrase
while its rule metric is a normalized *exact* string match, so a correctly retrieved needle answered
in the model's own words scores 0; the benchmark's own default is an LLM judge, which this
repository does not run for a hardware acceptance gate. The override makes the rule metric a
faithful, judge-free measurement of retrieval. Corpus, needle text, retrieval question, and the
insertion algorithm are EvalScope's unmodified defaults.

`eval/tp2_needle_throughput_probe.py` issues one streamed request on a needle prompt of a chosen
length and reports prompt tokens, TTFT, and decode tokens/second, so the same prompt can be timed
against a TP2 server and a single-GPU server. `--task summary` keeps the identical haystack but
asks for a long answer, so the decode phase is long enough to time precisely. Point it at whichever
server is currently up:

```bash
# TP2, against a server started as above
eval/.venv/bin/python eval/tp2_needle_throughput_probe.py \
  --context-tokens 250000 --depth 50 --task summary --max-tokens 512 \
  --label tp2-250k-summary --out eval/server-logs/throughput-probe.jsonl

# TP1 baseline: restart ninfer-serve with `--device 0 --max-context 252928` (262,144 does not fit
# one card), then issue the byte-identical prompt.
eval/.venv/bin/python eval/tp2_needle_throughput_probe.py \
  --context-tokens 250000 --depth 50 --task summary --max-tokens 512 \
  --label tp1-250k-summary --out eval/server-logs/throughput-probe.jsonl
```

The server's own `--request-log-jsonl` record (`timings_seconds.prefill` / `.decode`) is the
authoritative split between the two phases; the probe's streamed timings agree within about 1 %.

`eval/mtp_reasoning_probe.py` is the MTP acceptance gate's probe. It sends one fixed
~512-token reasoning prompt -- a multi-part derivation, so the answer is long and its acceptance
rate is meaningful -- and reports the same client-side timings. Unlike the needle probe it builds
no corpus and needs no tokenizer, so plain `python3` runs it. The acceptance numbers themselves
come from the server's request log, whose `speculative` block carries `rounds`, `drafted_tokens`,
`accepted_tokens` and `accepted_per_position` per request:

```bash
# Start ninfer-serve with `--spec mtp --draft-tokens 3 --lm-head-draft`, then:
python3 eval/mtp_reasoning_probe.py --label tp2-mtp --max-tokens 512 \
  --out eval/server-logs/mtp-probe.jsonl
python3 -c "import json,sys; [print(json.loads(l)['speculative']) for l in open(sys.argv[1]) \
  if 'speculative' in l]" eval/server-logs/<server>.requests.jsonl
```

Recorded MTP3 results for `qwen3_8_27b_nvfp4.ninfer` (`--draft-tokens 3 --lm-head-draft`, INT8 KV,
CUDA graphs on), TP2 at `--max-context 262144` against TP1 at `--max-context 252928`. **All rows in
this subsection were taken at the 400 W per-GPU cap**; see `docs/performance.md` for the 575 W
re-measurement and for why figures from the two conditions must not be compared:

| Prompt | Acceptance TP1 | Acceptance TP2 | Decode tok/s TP1 | Decode tok/s TP2 |
|---|---:|---:|---:|---:|
| 536-token reasoning | 56.61 % | 58.06 % | 152.0 | 189.5 |
| 249,955-token needle summary | 50.83 % | 57.96 % | 101.7 | 152.1 |

Per-GPU process residency under TP2 with MTP is a flat 16,062 MiB (15.69 GiB); TP1 with the smaller
252,928-token window uses 29,858 MiB (29.16 GiB) on one card. Both were measured at the 400 W cap
alongside the table above (residency is a memory figure, so the power condition does not move it —
it is stated for provenance).

Recorded TP2 results for `qwen3_8_27b_nvfp4.ninfer` (INT8 KV, CUDA graphs on, speculation off):

| Tier | Prompt tokens | Needles | Run directory |
|---|---:|---:|---|
| 64k | 65,490 | 10 / 10 | `eval/runs/20260825T162206Z-e50f922c` |
| 128k | 131,025 | 10 / 10 | `eval/runs/20260826T064554Z-5fe0220d` |
| 262k | 259,954 | 10 / 10 | `eval/runs/20260826T065823Z-055cb1f2` |

Each tier's 10 samples are 5 depths across two haystacks. Note that `chinese` selects a different
*corpus* (Journey to the West), not a translated needle: EvalScope inserts the same English needle
and asks the same English question in both, so the pair is a second haystack condition rather than
a second language condition. The grid is also deliberately narrower than the 35B `native_long`
profile's eleven depths — five depths x two haystacks is what the TP2 acceptance gate specifies,
and at ~105 s per 262k prefill an eleven-depth grid would cost about 40 minutes per tier.

Retrieval-vs-memorization controls at 262k (`eval/runs/20260826T074339Z-f9f7c5cb`,
`20260826T074608Z-bb0da9f5`, `20260826T074907Z-b6f62525`): with a needle the model cannot have
memorized, it reproduced the invented sentence verbatim out of a 259,956-token prompt (score 1).
Asked about an entity absent from the prompt it returned the nearest matching sentence under the
strict prompt, but answered `NOT FOUND` once the prompt allowed it to decline -- so the
substitution is a prompt artifact, not a failure to discriminate.

BFCL-v4 full evaluation contains 5,106 samples. Multi-turn samples can make more than one model
request. Its Web Search subsets require `SERPAPI_API_KEY`; `memory_vector` may download an upstream
model, which the example explicitly acknowledges with `allow_network_downloads: true`.

Inspect and resume a run:

```bash
eval/.venv/bin/python -m ninfer_eval status --run eval/runs/<run-id>
eval/.venv/bin/python -m ninfer_eval resume --run eval/runs/<run-id>
eval/.venv/bin/python -m ninfer_eval summarize --run eval/runs/<run-id>
```

Resume rejects a changed effective configuration or backend version. Completed jobs are skipped;
an incomplete EvalScope job reuses its own prediction cache when available.

### Qwen3.8-27B NVFP4 needles at 655k and 1M (TP2 + YaRN x4), with a vLLM control

[`configs/qwen3_8_27b_nvfp4_needle_1m.yaml`](configs/qwen3_8_27b_nvfp4_needle_1m.yaml) is the
extended-context suite. It differs from the 262k suite above in three ways that matter.

**1. A distinct-text corpus.** EvalScope's needle adapter *tiles* a corpus shorter than the
requested context (`_get_context_tokens` concatenates the file to itself until it is long enough),
and the stock ModelScope corpus is only ~155k tokens, so every tier above that measures retrieval
over repeated text. This suite instead uses a purpose-built corpus of Project Gutenberg
public-domain English books:

```bash
eval/.venv/bin/python tools/tp2/build_needle_1m_corpus.py \
    --out-dir /home/pc/models/ninfer-38/needle-1m \
    --tokenizer /home/pc/models/ninfer-38/unsloth-nvfp4     # ~4 min, needs gutenberg.org

# Prove the benchmark never tiles it, and measure the prompt lengths it will produce:
eval/.venv/bin/python tools/tp2/verify_needle_1m_prompt.py \
    --context-lengths 8192 653000 1046000 \
    --json-out eval/results/needle-1m/corpus-no-tiling-verification.json
```

The builder writes two **disjoint** 6,000,000-character bundles, both English. Their file names are
dictated by the adapter, which hardcodes `english -> PaulGraham_Essays.txt` and
`chinese -> Journey_to_the_West.txt`: bundle A goes into the first slot, bundle B into the second.
Neither file contains what its name says, and neither subset is Chinese. Running both subsets is
what gives **two independent samples per depth**. `provenance.json` in the output directory lists
every book and both bundle checksums.

**2. Two engines, never at the same time.** Each needs both GPUs.

```bash
eval/run_qwen3_8_27b_nvfp4_needle_1m.sh --plan 1m
eval/run_qwen3_8_27b_nvfp4_needle_1m.sh smoke        # one 8k needle, ~1 min
eval/run_qwen3_8_27b_nvfp4_needle_1m.sh 655k         # 10 needles, ~1.4 h
eval/run_qwen3_8_27b_nvfp4_needle_1m.sh 1m           # 10 needles, ~3.2 h
eval/run_qwen3_8_27b_nvfp4_needle_1m.sh ninfer-all   # all three on ONE server, ~4.6 h
eval/run_qwen3_8_27b_nvfp4_needle_1m.sh control-1m   # novel-needle retrieval control, ~21 min

# The vLLM control: starts and stops /home/pc/Projects/vllm/serve-orca-qwen38-27b-long.sh
eval/run_qwen3_8_27b_nvfp4_needle_1m.sh vllm-smoke
eval/run_qwen3_8_27b_nvfp4_needle_1m.sh vllm-655k    # ~1.1 h
eval/run_qwen3_8_27b_nvfp4_needle_1m.sh vllm-all
```

The script asserts `nvidia-smi --query-compute-apps` is empty **before** launching and **after**
exiting, writing both to `eval/server-logs/needle1m-<step>-<stamp>.nvidia-smi.txt`, and refuses to
start alongside a job it did not start. vLLM is launched under `setsid` and stopped by signalling
its process group, because `vllm serve` forks engine-core and worker children that would otherwise
keep holding GPU memory.

Every step's sampler also records `power.limit` and `power.draw` per GPU per tick, and the step
prints a `power gpu N limit ... W, draw peak ... W, mean ... W` line beside the memory peaks. This
matters on this host: the 1M needle numbers were all measured with both RTX 5090s at a **400 W**
power limit against 600 W / 575 W stock defaults, and the original artifacts did not record it.
`report_vram_peaks` accepts sampler CSVs both with and without the two power columns.

`NINFER_NEEDLE_1M_CORPUS` overrides the haystack directory. The corpus is a `local_path` inside the
YAML and `ninfer_eval` does not expand environment variables in configs, so the script renders a
copy of the config with the `local_path: &corpus_path` anchor rewritten (to
`eval/server-logs/needle1m-<step>-<stamp>.config.yaml`) and runs that; unset, the shipped config is
used verbatim. The rewrite fails loudly if the anchor is missing or the directory does not exist.

The NInfer steps serve both tiers from one `--max-context 1048576 --rope yarn --yarn-factor 4.0
--yarn-origin 262144 --max-concurrency 1` process. The YaRN table depends on the factor and origin,
not on `--max-context`, so the 655k tier is rope-identical to the 1M tier and to the control.
`--max-concurrency 1` is arithmetic, not policy: the per-slot sequence cost at 1M is 16.66
GiB/device.

**3. The tiers are haystack lengths, not prompt lengths.** 653,000 produces a 652,954-token prompt
that fits the control's `--max-model-len 655360` with the 512-token answer and 1,893 tokens to
spare; 1,046,000 produces 1,045,954 and leaves 2,109 tokens of the 1,048,576-token window. Both
tokenizers (the served artifact's and the vLLM checkpoint's) were verified to agree on those counts.

Consolidate the runs into one small table:

```bash
eval/.venv/bin/python tools/tp2/summarize_needle_1m.py \
    --run eval/runs/<ninfer-655k-run> --run eval/runs/<ninfer-1m-run> \
    --run eval/runs/<vllm-655k-run> \
    --requests eval/server-logs/needle1m-ninfer-all-<stamp>.requests.jsonl \
    --out-json eval/results/needle-1m/results.json \
    --out-md   eval/results/needle-1m/results.md
```

Scoring and the `prompt_template` override are the 262k suite's, unchanged.

## Progress And Logs

TTY runs use a live display with dataset phase, completed/total units, elapsed time, rate, and ETA.
Non-TTY runs print periodic heartbeats without ANSI cursor control. Unknown totals remain `?`; the
framework does not invent a percentage or ETA.

Every run is stored below `eval/runs/<timestamp>-<config-hash>/`:

| Artifact | Purpose |
|---|---|
| `effective-config.yaml` | validated, secret-free effective configuration |
| `manifest.json` | git state, environment, backend versions, target and concurrency provenance |
| `state.json` | atomically updated operational and resume state |
| `events.jsonl` | append-only structured progress and lifecycle events |
| `run.log` | human-readable timestamps, progress, retries, and failures |
| `backends/<job>/` | unchanged backend-native predictions, logs, cache, and reports |
| `summary.json` | versioned normalized result contract |
| `summary.md` | compact human-readable score table |

The sample-retention policy is recorded in the manifest. API keys and known secret values are
redacted from coordinator events and task snapshots.

## Scores

Each benchmark remains independently reportable. The framework does not average AIME, GPQA, and
BFCL into an invented cross-benchmark score.

- AIME25 and AIME26 report rule-scored accuracy over 30 samples each.
- GPQA-Diamond reports accuracy over 198 samples.
- IFBench reports prompt- and instruction-level strict and loose adherence over 300 samples; its
  primary metric is `prompt_level_strict`.
- ERQA reports accuracy over 400 multimodal samples across eight reasoning subsets.
- RealWorldQA reports accuracy over 765 multimodal samples.
- BFCL-v4 reports its official `agentic`, `multi_turn`, `live`, `non_live`, `hallucination`, and
  `overall` values when the full score-bearing suite is complete.

A partial or failed job makes the run `partial` or `failed`; an incomplete BFCL run is never labeled
as the official full BFCL score.

## Adding Evaluations

An ordinary EvalScope dataset needs only another configured job:

```yaml
- id: new_dataset
  backend: evalscope
  dataset: evalscope_dataset_name
  target: model_api
  generation:
    temperature: 0
  backend_args:
    subset_list: [subset_name]
```

An evaluator that does not use EvalScope implements the four-method backend protocol in
`ninfer_eval/backends/base.py`, registers one stable name in `backends/registry.py`, retains its raw
artifacts, and returns the normalized `DatasetResult`. The coordinator and summary writer do not
need benchmark-specific changes.

## Exit Status

| Code | Meaning |
|---:|---|
| 0 | completed successfully, or status query for an active run |
| 2 | invalid configuration or missing configured secret |
| 3 | missing/incompatible backend dependency |
| 4 | partial evaluation |
| 5 | failed evaluation or missing run artifact |
| 6 | cancelled evaluation |

## Verification

```bash
PYTHONPATH=eval eval/.venv/bin/python -m py_compile $(rg --files eval/ninfer_eval -g '*.py')
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover -s eval/tests -p 'test_*.py'
PYTHONPATH=eval eval/.venv/bin/python -m ninfer_eval run \
  --config eval/configs/mock-suite.yaml --suite all
```
