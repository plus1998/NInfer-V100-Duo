# Final re-run set and 575 W re-measurement

Committed summary of results whose raw logs are excluded by `.gitignore` (`*.log`, `eval/runs/`,
`eval/server-logs/`). Method, power conditions and caveats:
[`docs/performance.md`](../../../docs/performance.md) and
[`docs/maintainer/tp2-yarn-1m.md`](../../../docs/maintainer/tp2-yarn-1m.md). The re-run set was
taken on the pre-publication development branch, at the point that hardened the peer-ingress copy
and added the cross-rank MTP egress check.

## 1. Correctness re-run set

| Check | Result | Prior |
|---|---|---|
| `scripts/tp1-regression.sh` | PASS (`identity OK`, target `qwen3_8_27b`, weights `nvfp4`); log retained at `tp1-regression.txt` | PASS at every commit on the branch |
| `ninfer_qwen3_8_27b_tp2_real_test` | PASS, 49.15 s | PASS |
| `ninfer_qwen3_8_27b_mtp_tp2_real_test` (incl. the per-position MTP oracle) | PASS, 284.94 s | PASS |
| `ninfer_qwen3_8_27b_graph_tp2_test` | PASS, 34.51 s | PASS |
| `ninfer_qwen3_8_27b_tp2_parity_test` | PASS, 351.85 s | PASS |
| full host `ctest` | 114/115, 10 opt-in skips | same single pre-existing failure throughout |

The one failure is `ninfer_qwen3_6_frontend_test`, pre-existing and environment-only: it opens the
hard-coded upstream-author path `/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16/tokenizer.json`.

Cross-rank MTP egress check (new in the commit under test): rank 1's MTP egress matched rank 0's
in every compared field over every round of both synthetic probes, with a tp1 negative control
asserting the check stays inert without a peer.

## 2. Retrieval re-run

Rule-scored (strict exact match after normalization), thinking disabled, greedy.

| Tier | Depths x bundles | Requests | Prompt tokens | Retrieved | Prior | Run |
|---|---|---:|---:|:-:|:-:|---|
| 262k, native rope, tiled corpus | 10/30/50/70/90 x A+B | 10 | 259,953-259,954 | 10/10 | 10/10 | `20260828T200240Z-055cb1f2` |
| 653k, YaRN x4, distinct text | 50 x A+B | 2 | 652,954-652,955 | 2/2 | 10/10 | `20260828T201948Z-e3ec9281` |

The 262k and 653k tiers ran entirely at the 400 W cap; the caps were raised 13.5 minutes into the
653k tier, so only the last ~2 minutes of its second request saw 575 W. Three of the four 1,046k
requests ran wholly at 575 W.
| 1,046k, YaRN x4, distinct text | 10, 90 x A+B | 4 | 1,045,954-1,045,955 | 4/4 | 10/10 | `20260828T203551Z-189e3d80` |

Every individual cell scored 1.0. Suite wall times 1,014.2 s, 960.5 s, 4,307.2 s.
Per-GPU residency: 15,396 MiB (15.04 GiB) at 262k and 28,070 MiB (27.41 GiB) at 1M — identical to
every earlier run at those windows.

## 3. Re-measurement at 575 W per GPU

Campaign figures were taken at a 400 W cap (the minimum settable limit). Both cards were then set
to 575 W. Per-run power conditions are captured in `eval/server-logs/stock-perf-*.power.txt`.

### Matched 249,955-token prompt, 512 generated tokens

| Metric | TP1 @400 W | TP1 @575 W | TP2 @400 W | TP2 @575 W |
|---|---:|---:|---:|---:|
| Prefill tok/s | 2,269.8 | 2,484.2 | 2,680.1 | 2,787.0 |
| Decode tok/s, MTP off | 52.35 | 53.95 | 75.18 | 75.32 |
| Decode tok/s, MTP3 | 101.7 | 113.60 | 152.1 | 159.39 |
| MTP3 draft acceptance | 50.83% | 50.83% | 57.96% | 57.96% |
| Time to first token (client-side) | 111.0 s | 101.0 s | 93.7 s | 90.0 s |
| Per-GPU resident, MTP off | 27.90 GiB | 27.90 GiB | 15.04 GiB | 15.04 GiB |
| Per-GPU resident, MTP3 | 29.16 GiB | 29.16 GiB | 15.69 GiB | 15.69 GiB |
| Peak sampled draw, MTP off | — | 575.5 W | — | 390.9 / 405.7 W |
| Peak sampled draw, MTP3 | — | 575.8 W | — | 483.8 / 444.5 W |

Draft acceptance is identical to four decimal places across power conditions. TP1 saturates its
limit and TP2 does not, so lifting the cap helps TP1 more: the TP2/TP1 decode ratio narrows from
1.44x to 1.40x.

### Extended context, one request each

| Measurement | Prompt | Generated | Power | Prefill tok/s | Decode tok/s | Acceptance |
|---|---:|---:|---|---:|---:|---:|
| needle, MTP off | 1,045,954 | 24 | 400 W | 928.9 | 46.39 | — |
| needle re-run, MTP off | 1,045,954 | 24 | 575 W | 975.1 | 48.08 | — |
| probe, MTP off | 1,045,954 | 512 | 575 W | 971.4 (1,076.7 s) | 46.11 | — |
| probe, MTP3 | 1,045,954 | 512 | 575 W | 975.0 (1,072.7 s) | 100.54 | 56.41% |

The two 512-token rows are like-for-like (same prompt, window and budget): MTP3 is **2.18x**
MTP off, which replaces an earlier cross-window estimate of "about 2.2x". Residency 28,070 MiB
(MTP off) and 29,528 MiB (MTP3) per device, identical to every earlier run at this window.

### Saturated concurrent decode, 262,144-token window

| Concurrency | MTP off @400 W | MTP off @575 W | MTP3 @400 W | MTP3 @575 W |
|---:|---:|---:|---:|---:|
| 1 | 93.5 | 94.1 | 172.4 | 177.8 |
| 4 | 314.3 (3.36x) | 320.3 (3.40x) | 466.0 (2.70x) | 475.2 (2.67x) |

## 4. vLLM on the same-family NVFP4 weights — configuration finding

`sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4` is an independent NVFP4 quantization of the
same abliterated base fine-tune NInfer's artifact was converted from.

**Extended context in vLLM comes from the checkpoint, not a serving flag.** This repackaging
carries `max_position_embeddings = 262144` and no YaRN block in `config.json` `rope_parameters`, so
a bare `--max-model-len 655360` is rejected at startup. That is a configuration-location
difference, not a capability limit: injecting a YaRN block at load with `--hf-overrides` (plus
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`) does serve this checkpoint far beyond 262,144 tokens — this
host's `serve-qwen38-27b-long.sh` targets 750,000 with FP8 KV, TP2 and MTP3. NInfer reaches
1,048,576 on an unmodified artifact via `--rope yarn`.

**No throughput comparison was obtained in this re-run set.** vLLM did not complete startup during
this session in three configurations tried (fp8 KV + FlashInfer; bf16 KV + FlashAttention; and the
host's own production script verbatim). The cross-engine comparison was taken separately, at a
**500 W** per-GPU cap — a different power condition from the 400 W and 575 W rows above, and
therefore not comparable to them. See the cross-engine section of
[`docs/performance.md`](../../../docs/performance.md). `eval/run_qwen3_8_27b_nvfp4_stock_power_perf.sh vllm-nvfp4`
reproduces the row against NInfer's 2,787.0 / 75.32 tok/s at the same prompt and the same
condition.
