# NInfer CLI

`build/apps/ninfer` runs one request against one registered `.ninfer` artifact. Build NInfer and
download an artifact using the [project README](../README.md) before following this guide.

## Text input

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 16384 \
  --max-new 256
```

Exactly one of `--prompt` and `--messages` is required.

Answer content is streamed to stdout. Reasoning, model loading (including the registered target and
canonical `weights_id`), timings, throughput, GPU memory, and speculative-decoding statistics are
written to stderr, so stdout can be redirected independently:

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Return one sentence." --max-new 64 \
  > answer.txt 2> run.log
```

Thinking is enabled by default. If the chat template embedded in the loaded artifact exposes
reasoning effort, `--reasoning-effort low|medium|xhigh` selects it; omitting the option uses the
template's default. An artifact whose template does not expose effort rejects the option. Add
`--no-thinking` for direct-response prompt rendering; it cannot be combined with
`--reasoning-effort`. `--greedy` selects exact argmax decoding independently.

## Startup memory profile

GPU residency is frozen when the Engine starts:

- no `--spec` omits MTP/DFlash weights and state and the optimized proposal head;
- `--spec mtp` loads only MTP, while `--spec dflash` loads only the 35B-A3B text-only DFlash
  backend;
- a speculative backend with the full proposal head omits the optimized proposal head;
- Vision is disabled by default, omitting its weights, Vision scratch phase, and frozen
  request-transient allocation;
- `--vision` loads those allocations and enables image/video input.

The complete `.ninfer` inventory is still validated. These choices are not lazy loading: a
text-only Engine rejects media and cannot enable Vision later. DFlash and Vision are mutually
exclusive. The default speculative and Vision settings produce the smallest resident profile.

## Structured messages

`--messages` accepts either a non-empty JSON message array or an object containing `messages`
and an optional `tools` array.

```json
[
  {
    "role": "system",
    "content": "Answer concisely."
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image",
        "image": "examples/cli/media/visual_chart.png"
      },
      {
        "type": "text",
        "text": "Describe the chart."
      }
    ]
  }
]
```

Run message files from the repository root when they contain repository-relative media paths:

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --vision
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`.
System and developer messages retain their array positions; the Qwen family frontend renders both
as system-class ChatML turns rather than moving later instructions to the beginning.

Message content may be a string or an ordered array containing:

| Content type | Source field | Accepted source |
|---|---|---|
| text | `text` | string |
| image / image_url | `image` or `image_url` | local path, HTTP(S) URL, or base64 data URI |
| video / video_url | `video` or `video_url` | local path, HTTP(S) URL, or base64 data URI |

`image_url` and `video_url` may be strings or objects containing a string `url`. Assistant
history may include `reasoning_content` and `tool_calls`; a tool result uses role `tool` and
`tool_call_id`.

See [`examples/cli/`](../examples/cli/) for committed text, image, video, mixed-media, thinking,
long-decode, and long-context inputs.

## Speculative decoding

Speculative decoding is disabled by default. Select MTP with one to five draft positions, or the
35B-A3B text-only DFlash backend with one to fifteen. `--lm-head-draft` selects the optimized
proposal head and requires a selected backend:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 \
  --max-new 512 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For DFlash:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --spec dflash --draft-tokens 7 --lm-head-draft
```

MTP and DFlash cannot be enabled together. The published [performance results](performance.md)
use MTP with three draft tokens and DFlash with seven draft tokens (block length eight), both with
the optimized proposal head. DFlash accepts up to fifteen draft tokens; seven is the current
measured recommendation rather than a semantic limit.

## Common options

| Option | Meaning | Default |
|---|---|---:|
| `--max-context N` | per-sequence logical context ceiling | `2048` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `2048` |
| `--rope native\|yarn` | rotary regime; `yarn` applies YaRN frequency correction and raises the `--max-context` ceiling to `--yarn-origin` x `--yarn-factor` | `native` |
| `--yarn-factor F` | YaRN scaling factor, in `[1.0, 64.0]`; only read under `--rope yarn`; `origin x factor` must be a whole token count not exceeding `1048576` | `4.0` |
| `--yarn-origin O` | YaRN origin window; must equal the artifact's registered native context capacity (`262144`) | `262144` |
| `--prefill-chunk N` | positive text-prefill chunk, in multiples of 128 | `1024` |
| `--max-new N` | requested output-token limit | `128` |
| `--device N` | CUDA device index | `0` |
| `--tp 1\|2` | tensor-parallel width; `2` splits the model across two GPUs | `1` |
| `--devices A,B` | one CUDA device index per `--tp` rank; required for `--tp 2` | `--device` |
| `--kv-dtype bf16\|int8` | KV-cache storage | `bf16` |
| `--spec mtp\|dflash` | speculative backend | off |
| `--draft-tokens N` | MTP `1..5`; DFlash `1..15` | unset |
| `--lm-head-draft` | optimized proposal head | off |
| `--vision` | enable image/video input and load Vision GPU allocations | off |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-thinking` | disable thinking in prompt rendering | thinking on |
| `--reasoning-effort low\|medium\|xhigh` | select an effort exposed by the loaded chat template | template default |
| `--greedy` | exact argmax decoding | off |
| `--ignore-eos` | drop the checkpoint's own end-of-turn token ids from the request stop policy | off |
| `--temperature F` | sampling temperature override | registered model/mode default |
| `--top-p F` | nucleus-threshold override | registered model/mode default |
| `--top-k N` | top-k-threshold override | registered model/mode default |
| `--min-p F` | min-p-threshold override | registered model/mode default |
| `--presence-penalty F` | presence-penalty override | registered model/mode default |
| `--frequency-penalty F` | frequency-penalty override | registered model/mode default (`0`) |
| `--seed N` | sampling seed | `0` |

When a sampling flag is omitted, Engine selects the official general-task preset registered for
the loaded model and the rendered prompt mode. The current presets are:

| Model | Prompt mode | Temperature | Top-p | Top-k | Min-p | Presence penalty |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.6-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.8-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.8-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | thinking | `1.0` | `0.95` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |

Frequency penalty is `0` in every registered preset. Qwen's separate precise-coding recommendation
is task-specific and is therefore an explicit override rather than an inferred Engine default.

Repeat `--stop-token-id`, `--stop`, or `--reasoning-stop` to add stop conditions. Use
`--raw-output` to expose the frontend's raw output stream and `--print-token-ids` to include
generated token IDs in diagnostics.

`--ignore-eos` removes the checkpoint's registered `eos_token_id` set from the request stop policy,
so decode continues until `--max-new` is exhausted or the remaining context capacity is reached
(`finish reason` `output-limit` or `context-capacity`) instead of ending on the model's end-of-turn
token. Stop conditions added with `--stop-token-id`, `--stop`, and `--reasoning-stop` are
unaffected. This exists for fixed-length decode work such as sustained long-context soak and
throughput measurement; normal use should leave it off, because generation past the end-of-turn
token is off-distribution and its text is not a product output.

Run `./build/apps/ninfer --help` for the exact option contract.

## Dual-GPU execution

`--tp 2` splits one resident model across two CUDA devices and requires an explicit `--devices A,B`
naming two distinct devices of the same compute capability. `--device` alone selects the single
device used at the default `--tp 1`; when both are given they must agree on the primary device.

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 \
  --max-context 262144 --kv-dtype int8 --kv-capacity auto \
  --messages long_prompt.json --max-new 256 --no-thinking
```

Tensor-parallel execution is implemented for the 27B execution package (`qwen3.6-27b` and
`qwen3.8-27b`, either weight profile). `qwen3.6-35b-a3b` has no tensor-parallel path and rejects
`--tp 2` at startup, as do `--spec dflash` and `--vision`. `--spec mtp` is supported at `--tp 2`,
including compatible-prefix reuse in a resident Engine. Both suffix prefill and exact-frontier
sampling restore the complete state on both devices. The HTTP server enables reuse by default;
separate CLI processes do not share a cache.

The load summary reports weights, KV pool, GDN state, sequence, workspace, CUDA Graph and reserved
bytes per device, plus a free/total row for each. `--no-cuda-graph` runs decode eagerly; at `--tp 2`
that is the same two-stream forward pass without a captured graph, and the tokens are identical.
`--tp 1` is bit-identical to single-device execution.

Speculative decoding at `--tp 2` is output-equivalent to non-speculative decoding rather than
bit-identical: a verify round evaluates the target model over `draft_tokens + 1` columns at once
while an ordinary round evaluates one, which selects different GEMM shapes, so greedy streams can
diverge on a near-tie token. Every committed token is still one the target model's own argmax
selected.

## Context and memory

The registered model IDs have a native context limit of 262,144 tokens, which is the ceiling
`--max-context` is checked against under the default `--rope native`. `--rope yarn` applies YaRN
frequency correction and raises that ceiling to `--yarn-origin` x `--yarn-factor`, up to 1,048,576
tokens; `--yarn-origin` must equal the registered native capacity (262,144); `--yarn-factor`
accepts a finite value in [1.0, 64.0] and defaults to 4.0. YaRN works at either `--tp` width and is
rejected together with `--vision`, `--spec dflash`, or a target with no YaRN rope domain
(`qwen3.6-35b-a3b`). The load summary's
`rope` row reports the resolved mode, factor, origin, effective ceiling and `mscale`. The practical
allocation on one RTX 5090 depends on the selected artifact, media workload, output budget, and
KV-cache type.
Use `--kv-dtype int8` for large context allocations; at `--max-context 1048576` it is mandatory,
because a BF16 pool at that window does not fit two RTX 5090s. The 1,048,576-token configuration is
`--tp 2 --devices A,B --rope yarn --yarn-factor 4.0 --yarn-origin 262144 --kv-dtype int8
--kv-capacity auto`; it reserves 26.93 GiB per device without a speculative backend and 28.42 GiB
with `--spec mtp --draft-tokens 3`. A full-ceiling prefill takes about 19 minutes on the measured
host. The prepared prompt must fit
`--max-context`; generation stops at the remaining context capacity when necessary.
`--kv-capacity N` controls the shared physical Main Text KV pool independently and is rounded up to
the 64-token page size. `--kv-capacity auto` loads the selected weights, measures the remaining GPU
memory, and directly chooses the largest legal page capacity for the complete enabled runtime
layout. This includes the selected speculative backend, fixed sequence state, workspace, Vision
request transient, and CUDA Graph allowance, while leaving the default 1 GiB automatic headroom
unallocated. It does not probe allocations or resize the pool at request time. The single-request
CLI normally leaves the option omitted so it follows
`--max-context`; the distinction matters primarily to a concurrent Engine or server.

At Engine startup NInfer reserves model weights, persistent sequence state, one phase-reused
Program scratch arena, the maximum Vision request-transient buffer when Vision is enabled, and a
separate CUDA Graph driver allowance. Scratch is the maximum of the enabled Text, MTP, DFlash, and
Vision phases, not their sum. Its prefill bound uses
`min(--prefill-chunk,--max-context)`. The request-transient buffer is also frozen at startup; a
media request activates only the needed prefix and performs no project-owned device allocation or
growth.

All weight, sequence, workspace, request-transient, and graph allocations are released when the
Engine is destroyed.
