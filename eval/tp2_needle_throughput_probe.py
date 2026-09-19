#!/usr/bin/env python3
"""Single-request decode-throughput probe on a Needle-in-a-Haystack prompt.

The needle suite measures retrieval; this measures speed on exactly the same kind of
prompt, so the TP2 vs TP1 comparison is made on identical input.

The prompt is built with EvalScope's own `needle_haystack` adapter (same corpus, same
tokenizer, same insertion algorithm) and with the same system prompt and prompt template
that `eval/configs/qwen3_8_27b_nvfp4_needle_haystack.yaml` sets, so the probe prompt is
token-for-token what the suite would send at that context length and depth.

It streams the completion so time-to-first-token and the decode phase can be separated
client-side. The server's own `--request-log-jsonl` record (`timings_seconds.prefill` /
`timings_seconds.decode`) remains the authoritative measurement; the two agree closely.

Example:

    eval/.venv/bin/python eval/tp2_needle_throughput_probe.py \
        --context-tokens 250000 --depth 50 --max-tokens 256 --label tp2

Requires the eval virtualenv (EvalScope + modelscope) and a running ninfer-serve.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

CORPUS = (
    "/home/pc/.cache/modelscope/datasets/"
    "AI-ModelScope--Needle-in-a-Haystack-Corpus/snapshots/master"
)
TOKENIZER = "/home/pc/models/ninfer-38/unsloth-nvfp4"
SYSTEM_PROMPT = (
    "You are a helpful AI bot that answers questions for a user. "
    "Keep your response short and direct."
)
PROMPT_TEMPLATE = """Please read the following text and answer the question below.

<text>
{context}
</text>

<question>
{question}
</question>

Answer with the single sentence from the text that answers the question, copied
word for word. Output only that sentence and nothing else."""

# The needle answer is one sentence, so a needle prompt only exercises ~24 decode steps.
# `--task summary` keeps the identical haystack, needle, and framing but replaces the final
# instruction so the decode phase is long enough to time precisely. Both servers under
# comparison receive the byte-identical prompt.
SUMMARY_TEMPLATE = """Please read the following text and answer the question below.

<text>
{context}
</text>

<question>
{question}
</question>

Answer the question, then summarise the text above in as much detail as you can. Write at
least 400 words."""

TEMPLATES = {"needle": PROMPT_TEMPLATE, "summary": SUMMARY_TEMPLATE}


def build_prompt(context_tokens: int, depth: int, subset: str, task: str) -> str:
    from evalscope import TaskConfig
    from evalscope.api.registry import get_benchmark
    import evalscope.benchmarks  # noqa: F401  (registers the benchmark)

    config = TaskConfig(
        model="probe",
        datasets=["needle_haystack"],
        dataset_args={
            "needle_haystack": {
                "local_path": CORPUS,
                "subset_list": [subset],
                "system_prompt": SYSTEM_PROMPT,
                "prompt_template": TEMPLATES[task],
                "extra_params": {
                    "context_lengths_min": context_tokens,
                    "context_lengths_max": context_tokens,
                    "context_lengths_num_intervals": 1,
                    "document_depth_percent_min": depth,
                    "document_depth_percent_max": depth,
                    "document_depth_percent_intervals": 1,
                    "tokenizer_path": TOKENIZER,
                },
            }
        },
    )
    adapter = get_benchmark("needle_haystack", config)
    dataset, _ = adapter.load()
    return adapter.format_prompt_template(dataset[subset][0])


def stream_completion(
    base_url: str, model: str, prompt: str, max_tokens: int, timeout: float
) -> dict:
    body = json.dumps(
        {
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
            payload = line[len("data:") :].strip()
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
        "decode_tokens_per_second": (decode_tokens / decode_seconds)
        if decode_seconds > 0
        else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--subset", default="english", choices=["english", "chinese"])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=86400.0)
    parser.add_argument(
        "--task",
        default="needle",
        choices=sorted(TEMPLATES),
        help="needle: the suite's one-sentence answer. summary: same haystack, long answer,\n"
        "so the decode phase is long enough to time precisely.",
    )
    parser.add_argument("--label", default="probe")
    parser.add_argument("--out", help="append the JSON result to this file")
    args = parser.parse_args()

    prompt = build_prompt(args.context_tokens, args.depth, args.subset, args.task)
    result = stream_completion(
        args.base_url, args.model, prompt, args.max_tokens, args.timeout
    )
    result.update(
        label=args.label,
        base_url=args.base_url,
        model=args.model,
        context_tokens=args.context_tokens,
        depth=args.depth,
        subset=args.subset,
        task=args.task,
        max_tokens=args.max_tokens,
    )
    line = json.dumps(result, ensure_ascii=False)
    print(line)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
