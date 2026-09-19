# 1,048,576-token soak (TP2 + YaRN x4, NVFP4, 2x RTX 5090 @ 400 W)

| | pass A | pass B |
|---|---|---|
| artifact prefix | qwen3_8_27b_nvfp4-1m-soak-pass-a-20260828T144651Z | qwen3_8_27b_nvfp4-1m-soak-pass-b-20260828T153852Z |
| prompt tokens | 949885 | 949885 |
| generated tokens | 98692 | 98692 |
| finish reason | context-capacity | context-capacity |
| prefill speed | 1011.89 tok/s | 1009.97 tok/s |
| decode speed | 45.49 tok/s | 45.39 tok/s |
| sampling | greedy (temperature 0) | greedy (temperature 0) |
| rope | yarn factor 4.0000 origin 262144, ceiling 1048576, mscale 1.1386 | yarn factor 4.0000 origin 262144, ceiling 1048576, mscale 1.1386 |
| max context | 1048576 | 1048576 |
| KV capacity | 1048576 | 1048576 |
| workspace peak | 112.29 MiB / 182.81 MiB | 112.29 MiB / 182.81 MiB |
| token-stream sha256 | 96203636fb503bad882b9a27aa39c97f7d921cd70aad4b982ce8dd722090fcb7 | 96203636fb503bad882b9a27aa39c97f7d921cd70aad4b982ce8dd722090fcb7 |
| generate/text prefill (s) | 938.725 | 940.508 |
| generate/decode (s) | 2169.388 | 2174.476 |
| generate/total (s) | 3109.521 | 3116.378 |
| prepare/render/preprocess (s) | 1.162 | 1.152 |
| load/engine construction (s) | 10.022 | 10.7 |

Prompt file `soak-prompt-950000.messages.json` sha256 `8ee18ba41f6c306507160f93f03e4b06dceceeb36e17d0aef15ec07fa41b5d31` - both passes read the same bytes.

## Logit-sanity windows (pass A, 10000 generated tokens each)

| # | first idx | tokens | distinct | distinct ratio | top-1 id | top-1 share | longest run | out-of-domain | out-of-vocab | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| 0 | 0 | 10000 | 373 | 0.0373 | 13 | 0.0643 | 2 | 0 | 0 | PASS |
| 1 | 10000 | 10000 | 371 | 0.0371 | 13 | 0.0653 | 2 | 0 | 0 | PASS |
| 2 | 20000 | 10000 | 371 | 0.0371 | 13 | 0.0656 | 2 | 0 | 0 | PASS |
| 3 | 30000 | 10000 | 371 | 0.0371 | 13 | 0.0653 | 2 | 0 | 0 | PASS |
| 4 | 40000 | 10000 | 371 | 0.0371 | 13 | 0.0644 | 2 | 0 | 0 | PASS |
| 5 | 50000 | 10000 | 371 | 0.0371 | 13 | 0.0653 | 2 | 0 | 0 | PASS |
| 6 | 60000 | 10000 | 371 | 0.0371 | 13 | 0.0657 | 2 | 0 | 0 | PASS |
| 7 | 70000 | 10000 | 371 | 0.0371 | 13 | 0.0650 | 2 | 0 | 0 | PASS |
| 8 | 80000 | 10000 | 371 | 0.0371 | 13 | 0.0647 | 2 | 0 | 0 | PASS |
| 9 | 90000 | 8692 | 371 | 0.0427 | 13 | 0.0660 | 2 | 0 | 0 | PASS |

Worst window by top-1 share: #9 at 0.0660 (limit 0.95). Worst by longest constant run: #0 at 2 (limit 64). Pass B's window table is in the JSON; the two streams are byte-identical when the determinism gate passes.

## Generated-stream structure (pass A)

`--ignore-eos` keeps decode running past the model's end-of-turn token, so the stream is a repeating turn cycle rather than one answer. Pass A holds 52 matches of the opening 24-token signature: 51 complete episodes of [1901, 1903, 1904, 1905] tokens plus a 1635-token truncated tail cut off by the context ceiling. Each steady episode is one substantive answer followed by a short run of empty turn frames — 4 end-of-turn tokens (id 248046) per episode, 203 in the whole stream: one closes the answer, the rest close empty `<|im_start|>…<|im_end|>` frames before the next answer begins.

The longest run of **exactly identical consecutive episodes** is 46 episodes, starting at generated index 9519. That run is the sharpest numerical-stability evidence here: the same argmax sequence is reproduced over tens of thousands of additional KV positions without a single token flipping. It is also why the distinct-token ratio is low (373 distinct ids over 98692 tokens) — the vocabulary is bounded by the repeating turn, not by a collapsing logit distribution.

**What this does not cover.** Because the cycle locks in early, sanity windows 1-9 are one probe repeated, not nine independent probes: they establish that decode at positions 950k-1,048,576 is numerically stable and reproducible, and they do not exercise token-space diversity at all — 373 of 248,077 head rows are ever selected. A seeded `temperature > 0` soak is the coverage complement: it would walk a far wider slice of the vocabulary at the cost of the byte-identical determinism check this run relies on. Neither run substitutes for the other, and only the greedy one was performed here.

Gate note: `top-1 share` (limit 95%) and `longest run` (limit 64) and `out-of-domain` (limit 0) are all gated per window; `distinct` and `distinct ratio` are **reported, not gated** — there is no defensible threshold for them that a legitimate repeating turn cycle would pass.

## VRAM (nvidia-smi, sampled every 2 s for the life of each pass)

| pass | series | samples | min MiB | max MiB | max GiB | samples at peak | plateau samples | plateau span MiB | growth MiB | flat |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| pass-a | gpu 0 | 1516 | 1008 | 29098 | 28.42 | 2 | 274 | 17 | 9 | NO |
| pass-a | gpu 1 | 1516 | 32 | 28110 | 27.45 | 1510 | 1511 | 2 | 2 | yes |
| pass-a | process 1478955@gpu0 | 1515 | 10824 | 28070 | 27.41 | 1510 | 1511 | 2 | 2 | yes |
| pass-a | process 1478955@gpu1 | 1515 | 10824 | 28070 | 27.41 | 1510 | 1511 | 2 | 2 | yes |
| pass-b | gpu 0 | 1520 | 982 | 29094 | 28.41 | 15 | 287 | 15 | -2 | NO |
| pass-b | gpu 1 | 1520 | 32 | 28110 | 27.45 | 1513 | 1514 | 4 | 4 | yes |
| pass-b | process 1524819@gpu0 | 1519 | 504 | 28070 | 27.41 | 1513 | 1514 | 4 | 4 | yes |
| pass-b | process 1524819@gpu1 | 1519 | 504 | 28070 | 27.41 | 1513 | 1514 | 4 | 4 | yes |

Plateau = from the first sample within 16 MiB of the peak to the last one; the samples outside it are the startup ramp and the final teardown tick. `growth MiB` is the last plateau sample minus the first.

Only the `process ...` rows are gated: a `gpu N` row also counts memory other processes hold (GPU 0 carries ~1 GiB of desktop allocation on this machine) and drops to that baseline the moment the run exits.

## Gates

| gate | verdict |
|---|:-:|
| no_crash | PASS |
| window_full | PASS |
| logit_sanity | PASS |
| vram_within_gate | PASS |
| vram_flat | PASS |
| deterministic | PASS |
