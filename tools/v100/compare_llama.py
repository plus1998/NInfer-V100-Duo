#!/usr/bin/env python3
"""Measure LM Studio's llama-server on the exact token corpus used by ninfer_bench."""
import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path


# End-of-generation tokens for this exact Qwen3.8 checkpoint.
EOG_TOKENS = {248044, 248046, 248063, 248064, 248065}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18081")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=85000)
    parser.add_argument("--decode-tokens", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0,
                        help="discarded full requests before measured repetitions")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corpus = [int(token) for token in args.corpus.read_text().split()]
    if args.prompt_tokens < 1 or len(corpus) < args.prompt_tokens:
        parser.error("corpus must contain the requested positive prompt length")
    if args.decode_tokens < 1 or args.repetitions < 1 or args.warmup < 0:
        parser.error("decode tokens and repetitions must be positive; warmup must be nonnegative")
    payload = {
        "prompt": corpus[:args.prompt_tokens],
        "n_predict": args.decode_tokens + 1,
        "temperature": 0,
        "ignore_eos": False,
        "cache_prompt": False,
        "stream": False,
        "return_tokens": True,
    }
    report = {"url": args.url, "corpus": str(args.corpus),
              "prompt_tokens": args.prompt_tokens, "decode_tokens": args.decode_tokens,
              "warmup": args.warmup,
              "request": {k: v for k, v in payload.items() if k != "prompt"},
              "warmup_reps": [], "reps": []}
    for index in range(args.warmup + args.repetitions):
        request = urllib.request.Request(args.url.rstrip("/") + "/completion",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=7200) as response:
            result = json.load(response)
        timings = result["timings"]
        row = {"wall_seconds": time.monotonic() - start, "timings": timings,
               "tokens_predicted": result.get("tokens_predicted"),
               "tokens_evaluated": result.get("tokens_evaluated"),
               "tokens_cached": result.get("tokens_cached"),
               "content": result.get("content"),
               "tokens": result.get("tokens"),
               "stop_type": result.get("stop_type"),
               "stopped_eos": result.get("stopped_eos"),
               "stopped_limit": result.get("stopped_limit"),
               "stopped_word": result.get("stopped_word"),
               "stopping_word": result.get("stopping_word"),
               "truncated": result.get("truncated"),
               "generation_settings": result.get("generation_settings")}
        # Store the server's actual token count and timing convention, including
        # whether it counts the prefill-produced token. Do not silently compare
        # predicted_per_second with NInfer's first-token-excluded decode rate.
        count = int(timings["predicted_n"])
        tokens = result.get("tokens")
        row["fixed_window_complete"] = (
            result.get("tokens_predicted") == args.decode_tokens + 1
            and count in (args.decode_tokens, args.decode_tokens + 1)
            and isinstance(tokens, list)
            and len(tokens) == args.decode_tokens + 1
            and not EOG_TOKENS.intersection(tokens)
            and timings.get("prompt_n") == args.prompt_tokens
            and timings.get("cache_n") == 0
            and result.get("tokens_evaluated") == args.prompt_tokens
            and not result.get("truncated")
            and not result.get("stopped_eos")
            and not result.get("stopped_word")
            and result.get("stop_type") == "limit"
            and timings["predicted_ms"] > 0
        )
        is_warmup = index < args.warmup
        report["warmup_reps" if is_warmup else "reps"].append(row)
        if not row["fixed_window_complete"]:
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            raise RuntimeError("invalid fixed-window completion (output/EOG, prompt/cache, "
                               "truncation or stop mismatch); saved diagnostic output")
        row["decode_excluding_first_tok_s"] = args.decode_tokens * 1000 / timings["predicted_ms"]
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        phase = "warmup" if is_warmup else "rep"
        phase_index = index + 1 if is_warmup else index - args.warmup + 1
        print(f"{phase}={phase_index} predicted_n={count} "
              f"decode_excluding_first_tok_s={row['decode_excluding_first_tok_s']:.3f}", flush=True)
    values = [r["decode_excluding_first_tok_s"] for r in report["reps"]]
    report["decode_tok_s_mean"] = statistics.mean(values)
    report["decode_tok_s_stdev"] = statistics.stdev(values) if len(values) > 1 else 0
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
