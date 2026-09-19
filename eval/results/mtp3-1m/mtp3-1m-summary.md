# MTP3 at 1M under TP2 + YaRN ×4

Window: `--max-context 1048576`. MTP3 = `--spec mtp --draft-tokens 3 --lm-head-draft`.
Both RTX 5090s power-capped at 400 W for every measurement below.

| step | prompt tok | gen tok | prefill | decode | gpu0 reserved | gpu1 reserved | nvidia-smi peak/GPU |
|---|---|---|---|---|---|---|---|
| boot | 15 | 10 | 447.56 tok/s | 213.84 tok/s | 28.42 GiB | 28.42 GiB | 29526 MiB |
| needle-a-mtp3 | 1045954 | 24 | 930.41 tok/s | 132.60 tok/s | 28.42 GiB | 28.42 GiB | 29528 MiB |
| needle-a-off | 1045954 | 24 | 930.92 tok/s | 44.58 tok/s | 26.93 GiB | 26.93 GiB | 28070 MiB |
| needle-b-mtp3 | 1045955 | 24 | 929.57 tok/s | 132.80 tok/s | 28.42 GiB | 28.42 GiB | 29528 MiB |
| decode-greedy | 949885 | 12000 | 1007.98 tok/s | 135.78 tok/s | 28.42 GiB | 28.42 GiB | 29528 MiB |
| decode-sampled | 949885 | 12000 | 1008.17 tok/s | 122.21 tok/s | 28.42 GiB | 28.42 GiB | 29528 MiB |
| decode-novel | 949885 | 1341 | 1010.19 tok/s | 99.51 tok/s | 28.42 GiB | 28.42 GiB | 29528 MiB |

## What MTP3 costs at this window

Both rows are the same 1,045,954-token prompt; the runs differ only in `--spec mtp --draft-tokens 3 --lm-head-draft`.

| device-0 arena | MTP off | MTP3 | delta |
|---|---|---|---|
| weights | 10.08 GiB | 10.46 GiB | +0.38 GiB |
| kv pool | 16.50 GiB | 17.53 GiB | +1.03 GiB |
| gdn state | 146.81 MiB | 146.81 MiB | +0.00 GiB |
| sequence | 16.66 GiB | 17.69 GiB | +1.03 GiB |
| workspace | 182.81 MiB | 192.93 MiB | +0.01 GiB |
| reserved | 26.93 GiB | 28.42 GiB | +1.49 GiB |
| nvidia-smi per-process peak | 28070 MiB | 29528 MiB | +1458 MiB (+1.42 GiB) |

The same delta measured +0.65 GiB/device at `--max-context 262144`. The fixed part is the MTP head's weights (+0.38 GiB); the rest is its own KV pages, which scale with the window: 1.03 GiB at 1,048,576 tokens is 0.26 GiB at 262,144, and 0.38 + 0.26 = 0.64. So the MTP surcharge is **0.38 GiB + 1.03 GiB per 1M tokens of context**, not a constant.

## Speculative statistics

| step | rounds | drafted | accepted | acceptance | tok/round | fallback |
|---|---|---|---|---|---|---|
| boot | 3 | 9 | 7 | 77.78% | 3.33 tok/round | 0 |
| needle-a-mtp3 | 6 | 18 | 18 | 100.00% | 4.00 tok/round | 0 |
| needle-b-mtp3 | 6 | 18 | 17 | 94.44% | 3.83 tok/round | 0 |
| decode-greedy | 3153 | 9457 | 8846 | 93.54% | 3.81 tok/round | 0 |
| decode-sampled | 3518 | 10552 | 8480 | 80.36% | 3.41 tok/round | 1 |
| decode-novel | 486 | 1456 | 854 | 58.65% | 2.76 tok/round | 0 |

## Needle: MTP3 vs MTP-off

- bundle A token ids identical: **True** (24 vs 24 tokens, first divergence None)
- bundle A answer matches the needle campaign's MTP-off answer: **True**
- bundle B answer matches the needle campaign's MTP-off answer: **True**

## Decode streams

- `decode-greedy`: 12000 tokens, 308 distinct (ratio 0.0257), repeating period 187, repeats 57
- `decode-sampled`: 12000 tokens, 740 distinct (ratio 0.0617), repeating period None, repeats None
- `decode-novel`: 1341 tokens, 304 distinct (ratio 0.2267), repeating period None, repeats None
