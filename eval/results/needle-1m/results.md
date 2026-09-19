# Needle-in-a-haystack results, 653k and 1,046k tokens

Scores are EvalScope `judge_strategy: rule` (strict exact match after normalisation); 1 = the inserted needle was reproduced verbatim.
`bundle A` / `bundle B` are the two disjoint English Gutenberg book sets, carried by the adapter's `english` / `chinese` subset slots.


## needle_1m_ninfer_1m -- model `qwen3.8-27b`, haystack 1,046,000 tokens

| depth % | bundle A | bundle B | depth total |
|---:|:-:|:-:|:-:|
| 10 | 1 | 1 | 2 / 2 |
| 30 | 1 | 1 | 2 / 2 |
| 50 | 1 | 1 | 2 / 2 |
| 70 | 1 | 1 | 2 / 2 |
| 90 | 1 | 1 | 2 / 2 |
| **total** | | | **10 / 10** |

## needle_1m_ninfer_1m_novel -- model `qwen3.8-27b`, haystack 1,046,000 tokens

| depth % | bundle A | bundle B | depth total |
|---:|:-:|:-:|:-:|
| 50 | 1 | - | 1 / 1 |
| **total** | | | **1 / 1** |

## needle_1m_ninfer_655k -- model `qwen3.8-27b`, haystack 653,000 tokens

| depth % | bundle A | bundle B | depth total |
|---:|:-:|:-:|:-:|
| 10 | 1 | 1 | 2 / 2 |
| 30 | 1 | 1 | 2 / 2 |
| 50 | 1 | 1 | 2 / 2 |
| 70 | 1 | 1 | 2 / 2 |
| 90 | 1 | 1 | 2 / 2 |
| **total** | | | **10 / 10** |

## needle_1m_vllm_655k -- model `my-model`, haystack 653,000 tokens

| depth % | bundle A | bundle B | depth total |
|---:|:-:|:-:|:-:|
| 10 | 1 | 1 | 2 / 2 |
| 30 | 1 | 1 | 2 / 2 |
| 50 | 1 | 1 | 2 / 2 |
| 70 | 1 | 1 | 2 / 2 |
| 90 | 1 | 1 | 2 / 2 |
| **total** | | | **10 / 10** |

## Server-side throughput (ninfer-serve phase timings)

| prompt tokens (bucket) | n | prefill s (mean) | prefill tok/s (mean) | decode tok/s (mean) | gen tokens (mean) |
|---:|---:|---:|---:|---:|---:|
| ~8,000 | 1 | 1.5 | 5517.5 | 97.89 | 24.0 |
| ~653,000 | 10 | 484.2 | 1348.4 | 58.22 | 24.0 |
| ~1,046,000 | 10 | 1126.0 | 928.9 | 46.39 | 24.0 |
