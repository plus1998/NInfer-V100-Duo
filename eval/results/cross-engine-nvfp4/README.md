# Cross-engine NVFP4 long-context throughput artifacts (500 W per GPU)

The measured artifacts behind the cross-engine section of
[`docs/performance.md`](../../../docs/performance.md): NInfer `--tp 2` with YaRN x4 against vLLM
0.25.1 `--tensor-parallel-size 2`, on the same pair of RTX 5090s, at three context tiers
(250k / 653k / 700k tokens) on byte-identical prompts.

**Power condition.** Every row here was taken with **both GPUs capped at 500 W**. That is a third
power condition, distinct from the 400 W campaign and the 575 W re-measurement that produce the
other numbers in `docs/performance.md`. Rows from the three conditions are not comparable with each
other, and no figure from this directory should be quoted without its 500 W label. The cap was
captured from `nvidia-smi` before every server launch (`*.power.txt`) and sampled every 3 s
throughout each run (`*.vram.csv`).

**Read the caveats first.** [`summary.md`](summary.md) section 5 is the binding caveat list, and
`docs/performance.md` repeats it. In short: the two engines ran different NVFP4 quantizations of
different Qwen3.8-27B fine-tunes and different KV dtypes (FP8 against INT8), the prefill chunk sizes
differ and were not swept, vLLM has no MTP-off row, there is one measurement per cell, and no
quality or correctness claim is made — this is throughput only.

## Files

| File | What it is |
|---|---|
| `summary.md` | the full method, tables, ratios, crossover analysis and caveats |
| `ninfer-probes.jsonl` | raw NInfer client probe results (6 rows) |
| `vllm-probes.jsonl` | raw vLLM client probe results (3 tiers plus one warm-cache repeat) |
| `ninfer-server-records.txt` | NInfer `request_done` records: server-side timings, prefix-cache proof, MTP acceptance |
| `vllm-spec-and-cache.txt` | vLLM `/metrics` speculative and prefix-cache counters per tier |
| `ninfer-peaks.txt` | per-configuration VRAM and power-draw peaks |
| `ninfer-500w-*.requests.jsonl` | NInfer structured request logs (the authoritative timings) |
| `ninfer-boot-summaries.txt` | each NInfer server's boot capacity summary — resolved KV capacity, per-device slack, captured CUDA Graph bytes |
| `ninfer-500w-*.power.txt`, `vllm.power.txt` | the power condition captured before each launch |
| `ninfer-500w-*.vram.csv`, `vllm.vram.csv` | 3 s VRAM, power-limit and power-draw samples |
| `vllm-500w-*.vllm-log-path.txt` | where each tier's vLLM server log was written on the measurement host |
| `ninfer-500w-700000-mtp0-cudagraph-allowance-failure.txt` | the one transient CUDA Graph allowance boot failure, verbatim (`summary.md` caveat 10) |
| `ninfer-serve.md5` | the identity of the single `ninfer-serve` binary that produced all six NInfer rows |
| `run_ninfer_probes.sh`, `vllm_phase.sh`, `xprobe.py`, `sample_gpu.sh`, `peaks.py` | the harness |
| `superseded-run_vllm_probes.sh` | the first vLLM attempt's driver, before the reasoning-channel client fix; produced no usable timing |
| `pre-crash-575w/` | partial artifacts from an aborted 575 W attempt that the host ended by crashing; no results |

The `.log` files these runs wrote — both engines' server logs and the run driver logs — stay on the
measurement host and are not tracked here, under the repository-wide `*.log` ignore rule. Every
figure quoted in `summary.md` and in `docs/performance.md` comes from the tracked `.txt`, `.jsonl`
and `.csv` files above.

## Reproducing

`vllm_phase.sh` and `run_ninfer_probes.sh` each boot a fresh server per tier, so every measured
prefill is a genuine cache miss; both sides verify that (zero `vllm:prefix_cache_hits_total`, and
`prefix_cache_hit_tokens = 0` in every NInfer `request_done`). Both scripts refuse to launch while
`nvidia-smi --query-compute-apps` reports anything holding GPU memory — vLLM's `VLLM::Worker_TP*`
processes survive `SIGTERM` to the serve process group and can hold about 14.6 GiB each for over
ten minutes, so `vllm_phase.sh` escalates to `SIGKILL` and waits for the devices to come back empty.

The prompts come from `eval/tp2_needle_throughput_probe.py`'s `build_prompt`. `xprobe.py` exists
only because the vLLM deployment runs `--reasoning-parser qwen3`, which routes output to
`delta.reasoning_content`, and ignores a top-level `enable_thinking` field; it reuses the project
probe's prompt construction and timing logic unchanged and differs only in reading both stream
channels and sending `chat_template_kwargs={"enable_thinking": false}`.
