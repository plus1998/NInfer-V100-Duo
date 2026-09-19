# tp1 golden-token identity gate

`--tp 1` must stay bit-identical to upstream `ninfer` at base commit `feaf4dd`. Until this
directory existed, `scripts/tp1-regression.sh` only checked that a single-GPU run still produced a
plausible counting sequence and the right identity rows — it had no upstream baseline and could not
have detected a token-level drift. These files are that baseline.

## What was compared

Both binaries were built from the same source tree layout with the same toolchain and run on the
same GPU, back to back, with no other compute process resident:

| | |
|---|---|
| upstream | `feaf4dd` ("docs(eval): publish qwen3.8 groupwise-int results"), built in a detached `git worktree` at `-DCMAKE_BUILD_TYPE=Release` |
| fork | the pre-publication development branch, run with its `--tp 1` default and its `--rope native` default |
| artifact | `/home/pc/models/ninfer-38/huihui-nvfp4/qwen3_8_27b_nvfp4.ninfer` (`qwen3_8_27b` / `nvfp4`, 18.99 GiB as read) |
| hardware | one RTX 5090 (`sm_120a`), CUDA 13.1, driver 580.178.04 |
| date | 2026-08-29 |

Three greedy cases, chosen to cover three prefill regimes rather than three topics:

| case | prompt tokens | flags beyond the common set |
|---|---|---|
| 1 short chat | 23 | `--max-context 4096` |
| 2 instruction | 2,191 | `--max-context 4096` |
| 3 document | 28,677 | `--max-context 32768 --kv-dtype int8` |

Common set: `--messages <case>.json --max-new 128 --greedy --no-thinking --print-token-ids`, with
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`. `--greedy` forces temperature 0, so the
streams are deterministic and a single differing logit would show as a differing token id.
The exact command both binaries ran is the loop in `record.sh`:

```bash
tests/data/tp1-golden/record.sh \
  <ninfer-binary> \
  /home/pc/models/ninfer-38/huihui-nvfp4/qwen3_8_27b_nvfp4.ninfer \
  <output-dir>
```

## Result

**Byte-identical on every compared file, for all three cases.** `diff -r` between the two recorded
directories is empty: the same 72 / 128 / 98 generated token ids (case 1 and case 3 stop on the
end-of-turn token before the 128-token limit), the same generated text, and the same summary rows.

Upstream `feaf4dd` accepts `--print-token-ids`, so this is a token-id comparison, not a
text-only one.

## Files

`case-N.ids` — the CLI's `tokens generated ids` row, ids only.
`case-N.txt` — the generated content exactly as it reached stdout.
`case-N.summary` — the summary rows that are deterministic for a fixed binary, artifact, and
workload *and* printed by both builds. `record.sh` documents the exclusions: wall-clock rows, the
three free-memory rows (they move with whatever else is resident on the GPU), `CUDA Graph memory`
(the driver's own accounting, observed at 2.12 / 2.00 / 2.00 MiB over three consecutive identical
runs of one binary), and the rows this fork added that upstream does not print (`tensor parallel`,
`rope`, `gpu0 *`).

`make_prompts.py` generates the prompts; they are not checked in because case 3 is ~160 KB. The
generator is pure — a fixed LCG over a fixed word list, ASCII, no clock, no locale — so any Python 3
reproduces these three files:

```text
161d7a84d7c669970a32fd8b69d3a05a470b79d0011d6e9012759cb819b00912  case-1-short.json
ba605bd2a2d937988bcbd83f062c5e6e1753bdcac11d6531be646035addaddb5  case-2-instruction.json
c68f2a85889f5c4884ba2f58b86e196a92180bad1e324b81eb5d4ac7e160c94d  case-3-document.json
```

sha256 of the recorded goldens, identical for both builds:

```text
58237734967d10002b75e6f14a1201f1286e33b01513d9b05a574c12061a9d7d  case-1.ids
3d73be06a308cbab6bec8ba8d88ec24a9906668d00c81731745221f4aba40362  case-2.ids
d86c0dd3091b7ab5e20a5d033a7d078fec20d917d905807544437067f7e55d9c  case-3.ids
a74f46f88d6816f88e1b4b3399ec698b5c7d78c9c64d8017f5f1dc5ecb9b8490  case-1.txt
c7fe7033b2c4d06d3371150e82d2d7def36a663e2636c52663be135c8b66b5e7  case-2.txt
0ae12756aa67a540438f31cd747eec6f7b210f622a54e9377f183b56e57022f1  case-3.txt
110ad2c9161cdbf5f82c31a117a2db674eb6dfa7065ac0d9244b272ee4aed566  case-1.summary
c880f07d6070f323078d5b538701e82a5dc5d50c4cb964a559040619c0bcb4c4  case-2.summary
d4b1979f0cf035238fdfb520f5345c3e263133d9c5a14af4665edf5dbfb92a1c  case-3.summary
```

## Scope, honestly stated

This gate covers **greedy text decode at `--tp 1`, `--rope native`, NVFP4, MTP off, on the
`qwen3.8-27b/nvfp4` identity**, over one short, one 2k, and one 29k prefill. It does not cover
sampled decoding, Vision, MTP, the groupwise-int profile, the 35B-A3B target, or concurrency > 1;
for those, tp1 identity rests on the instruction-level `cuobjdump -sass` comparison showing that
every native rope kernel emits identical SASS with the YaRN path added, and on the tp2 parity
harness's null control, not on this file. It is a regression gate against future drift, and
it is re-run by `scripts/tp1-regression.sh` on the fork's binary alone — re-establishing the
baseline needs the upstream build again, which `record.sh` makes a one-command job.
