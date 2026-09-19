#!/usr/bin/env python3
"""Cross-engine wrapper around eval/tp2_needle_throughput_probe.py.

Prompt construction is IMPORTED from the project probe, so the haystack, needle, depth,
tokenizer, system prompt and template are byte-identical to the rows already published for
NInfer. Only the transport differs, in two ways that the project probe cannot express:

  * vLLM is started by the measurement host's production script with `--reasoning-parser qwen3`, which
    routes the model's thinking channel to `delta.reasoning_content` instead of
    `delta.content`. The project probe only reads `delta.content`, so against that server it
    sees an empty stream and cannot time anything. This wrapper times the FIRST delta on
    EITHER channel and reports which channel carried the tokens.
  * vLLM ignores a top-level `enable_thinking` field; the equivalent is
    `chat_template_kwargs={"enable_thinking": false}`. `--vllm` sends both, so both engines
    run in the same non-thinking mode. NInfer honours the top-level field natively and is
    sent exactly the body the project probe sends.

Timing semantics are the project probe's, unchanged: monotonic clock, TTFT = first streamed
delta, decode window = last delta - first delta over completion_tokens - 1 tokens.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

sys.path.insert(0, "/home/pc/Projects/ninfer-1m/eval")
from tp2_needle_throughput_probe import SYSTEM_PROMPT, build_prompt  # noqa: E402


def stream(base_url, model, prompt, max_tokens, timeout, vllm):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": False,
    }
    if vllm:
        del body["enable_thinking"]
        body["chat_template_kwargs"] = {"enable_thinking": False}
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    parts, reasoning_parts, usage = [], [], {}
    first = last = None
    chunks = 0
    channels = set()
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
                delta = choice.get("delta", {}) or {}
                text = delta.get("content")
                reasoning = delta.get("reasoning_content")
                if not text and not reasoning:
                    continue
                now = time.monotonic()
                if first is None:
                    first = now
                last = now
                chunks += 1
                if text:
                    parts.append(text)
                    channels.add("content")
                if reasoning:
                    reasoning_parts.append(reasoning)
                    channels.add("reasoning_content")
    finished = time.monotonic()
    completion_tokens = int(usage.get("completion_tokens") or chunks)
    decode_seconds = (last - first) if (first is not None and last is not None) else 0.0
    decode_tokens = max(completion_tokens - 1, 0)
    return {
        "text": "".join(parts)[:400],
        "reasoning_text": "".join(reasoning_parts)[:400],
        "stream_channels": sorted(channels),
        "stream_deltas": chunks,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "ttft_seconds": (first - started) if first is not None else None,
        "decode_seconds": decode_seconds,
        "total_seconds": finished - started,
        "decode_tokens_per_second": (decode_tokens / decode_seconds) if decode_seconds > 0 else None,
        "prefill_tokens_per_second": (usage.get("prompt_tokens") / (first - started))
        if (first is not None and usage.get("prompt_tokens")) else None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--context-tokens", type=int, required=True)
    p.add_argument("--depth", type=int, default=50)
    p.add_argument("--subset", default="english")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--task", default="summary")
    p.add_argument("--timeout", type=float, default=86400.0)
    p.add_argument("--vllm", action="store_true")
    p.add_argument("--label", default="probe")
    p.add_argument("--out")
    p.add_argument("--prompt-only", action="store_true")
    a = p.parse_args()
    prompt = build_prompt(a.context_tokens, a.depth, a.subset, a.task)
    if a.prompt_only:
        print(json.dumps({"chars": len(prompt)}))
        return 0
    result = stream(a.base_url, a.model, prompt, a.max_tokens, a.timeout, a.vllm)
    result.update(
        label=a.label, base_url=a.base_url, model=a.model,
        context_tokens=a.context_tokens, depth=a.depth, subset=a.subset,
        task=a.task, max_tokens=a.max_tokens, vllm_mode=a.vllm,
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    line = json.dumps(result, ensure_ascii=False)
    print(line)
    if a.out:
        with open(a.out, "a", encoding="utf-8") as h:
            h.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
