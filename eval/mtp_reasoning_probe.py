#!/usr/bin/env python3
"""MTP acceptance + decode-throughput probe on a short reasoning prompt.

The MTP acceptance gate is "a 512-token reasoning prompt with --spec mtp
--draft-tokens 3 --lm-head-draft; acceptance rate >= 80% of the tp1 run's rate". This is
the durable form of that measurement: it sends one fixed reasoning prompt (the same bytes
to every server under comparison, so acceptance is measured on identical input), streams
the completion, and reports the client-side decode rate. The server's own
`--request-log-jsonl` record is the authoritative source for BOTH the phase timings and
the `speculative` block (`rounds` / `drafted_tokens` / `accepted_tokens` /
`accepted_per_position`); this script prints the client-side view so a run can be sanity
checked without opening the log.

The prompt is deliberately a chain-of-reasoning task rather than a retrieval one: MTP
acceptance depends on how predictable the continuation is, so a prompt whose answer is a
long derivation is what makes the tp1-vs-tp2 comparison meaningful. It carries no needle
corpus and no tokenizer dependency, so unlike tp2_needle_throughput_probe.py it needs
nothing from the eval virtualenv.

Example:

    python3 eval/mtp_reasoning_probe.py --label tp2-mtp --max-tokens 512
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

# ~512 tokens of prompt. The task is a multi-part derivation so the model produces a long,
# structured answer: acceptance measured over 24 decode steps would be noise.
REASONING_PROMPT = """You are given a scheduling problem and must reason about it carefully, step by step, showing every intermediate quantity.

A small inference cluster has two GPUs. A single 27-billion-parameter model is split across both of them, so every transformer layer performs two all-reduce collectives: one after the attention output projection and one after the MLP down projection. The model has 64 layers, of which 16 use full attention and 48 use a gated linear-attention recurrence. Each all-reduce moves one hidden-sized activation block of 5120 BF16 values per token in each direction, and the two cards are connected by a host-staged path rather than a direct peer link, so each direction costs roughly 6.5 microseconds of fixed overhead plus the transfer time at an effective 11 gigabytes per second.

The decode step for a single token, with tensor parallelism disabled, takes 13.6 milliseconds on one card. With the model split, the per-card compute halves, but the collectives are added and the two cards must synchronise twice per layer.

Speculative decoding is then enabled. A small draft head proposes three tokens ahead of the committed frontier; the full model verifies all four positions in a single batched forward pass, which costs 1.35 times a single-token forward pass rather than four times it. A proposed token is committed only if the full model's own greedy choice at that position agrees with the proposal.

Work through the following, in order, and show the arithmetic for each:

1. The total number of all-reduce collectives executed per decoded token, and the total bytes moved across the link per token.
2. The wall-clock cost of those collectives per token, separating the fixed overhead from the transfer time.
3. The decode step time with the model split across both cards, assuming compute halves exactly.
4. The speed-up, or slow-down, of the split configuration against the single-card baseline.
5. The expected number of committed tokens per verification round, if each of the three proposals is accepted independently with probability 0.8, and a proposal is only considered once every earlier proposal in the same round has been accepted.
6. The effective tokens per second of the split, speculative configuration, and how it compares to the single-card, non-speculative baseline.

State every assumption you rely on, and finish with a one-paragraph summary of which of the two effects -- the split or the speculation -- contributes more to the final number."""


def stream_completion(base_url: str, model: str, prompt: str, max_tokens: int,
                      timeout: float) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 42,
            "stream": True,
            "stream_options": {"include_usage": True},
            "enable_thinking": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    text_parts: list[str] = []
    usage: dict = {}
    first_token_at: float | None = None
    last_token_at: float | None = None
    chunks = 0
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {}).get("content")
                if not delta:
                    continue
                now = time.monotonic()
                if first_token_at is None:
                    first_token_at = now
                last_token_at = now
                chunks += 1
                text_parts.append(delta)
    finished = time.monotonic()

    completion_tokens = int(usage.get("completion_tokens") or chunks)
    decode_seconds = (
        (last_token_at - first_token_at)
        if (first_token_at is not None and last_token_at is not None)
        else 0.0
    )
    # The first streamed token is produced by prefill, so the decode phase covers the
    # remaining completion_tokens - 1 tokens.
    decode_tokens = max(completion_tokens - 1, 0)
    return {
        "text": "".join(text_parts),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "ttft_seconds": (first_token_at - started) if first_token_at is not None else None,
        "decode_seconds": decode_seconds,
        "total_seconds": finished - started,
        "decode_tokens_per_second": (decode_tokens / decode_seconds) if decode_seconds > 0 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=86400.0)
    parser.add_argument("--label", default="probe")
    parser.add_argument("--out", help="append the JSON result to this file")
    args = parser.parse_args()

    result = stream_completion(args.base_url, args.model, REASONING_PROMPT, args.max_tokens,
                               args.timeout)
    record = {"label": args.label, **{k: v for k, v in result.items() if k != "text"}}
    print(json.dumps(record, indent=2))
    if args.out:
        with open(args.out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({**record, "text": result["text"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
