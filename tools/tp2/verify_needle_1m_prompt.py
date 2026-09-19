#!/usr/bin/env python3
"""Prove that the 1M needle haystack is distinct text, not a tiled corpus, and measure it.

This runs EvalScope's *own* haystack-construction code -- `NeedleHaystackAdapter`'s unbound
`_get_context_tokens` and `_insert_needles`, imported from the installed package -- against a
minimal stand-in object carrying only the four attributes those two methods touch. So the
context it builds is byte-for-byte the context the benchmark builds, without needing the rest
of the EvalScope evaluator plumbing.

It reports, per bundle and per tier:

  * whether the corpus was tiled. `_get_context_tokens` concatenates the corpus to itself
    while `len(tokens) < max(context_lengths)`; this script compares the corpus token count
    against the tier and states the number of repetitions (1 = no tiling).
  * a window-duplication census: N random 120-character windows are drawn from the built
    context and counted with `str.count`. On a tiled corpus every window recurs once per
    tile (a check against a tiled 262k corpus dump found exactly 2 occurrences of each window);
    on distinct text every window should occur exactly once.
  * the prompt length in tokens after the suite's prompt template, system prompt, and the
    HF chat template are applied -- an estimate of what the server will count, used to place
    the tier's `context_lengths_*` under the serving window.

The prompt template, system prompt, corpus directory and tokenizer are read out of the suite
config (`eval/configs/qwen3_8_27b_nvfp4_needle_1m.yaml`), and the needle and retrieval question
come from EvalScope's registered `needle_haystack` defaults, so nothing about the prompt is
restated here and this check cannot drift from what the suite actually sends.

Usage:
    eval/.venv/bin/python tools/tp2/verify_needle_1m_prompt.py \
        --context-lengths 8192 653000 1046000 \
        --json-out eval/results/needle-1m/corpus-no-tiling-verification.json

    # defaults come from the config; override either path explicitly if needed
    eval/.venv/bin/python tools/tp2/verify_needle_1m_prompt.py \
        --config eval/configs/qwen3_8_27b_nvfp4_needle_1m.yaml --suite ninfer_1m \
        --corpus-dir /home/pc/models/ninfer-38/needle-1m \
        --tokenizer /home/pc/models/ninfer-38/unsloth-nvfp4
"""

from __future__ import annotations

import argparse
import json
import random
import sys

DEFAULT_CONFIG = "eval/configs/qwen3_8_27b_nvfp4_needle_1m.yaml"
DEFAULT_SUITE = "ninfer_1m"

# The adapter's subset -> file mapping, which `NeedleHaystackAdapter.load()` hardcodes as a local
# dict and therefore cannot be imported. Everything else this script needs -- the prompt template,
# the system prompt, the corpus directory, the tokenizer -- is read from the suite config, and the
# needle and question come from the benchmark's registered defaults, so no wording is duplicated
# here and the script cannot drift from what the suite actually sends.
FILES = {"english": "PaulGraham_Essays.txt", "chinese": "Journey_to_the_West.txt"}


def load_suite_settings(config_path: str, suite: str) -> dict:
    """Pull the needle settings out of one suite's first job in the ninfer_eval config."""
    import yaml

    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    try:
        job = config["suites"][suite]["jobs"][0]
    except (KeyError, IndexError) as exc:
        raise SystemExit(f"{config_path}: no job found for suite {suite!r}") from exc
    dataset_args = job.get("backend_args", {}).get("dataset_args", {})
    extra = dataset_args.get("extra_params", {})

    from evalscope.api.registry import BENCHMARK_REGISTRY
    import evalscope.benchmarks.needle_haystack.needle_haystack_adapter  # noqa: F401  (registers)

    meta = BENCHMARK_REGISTRY.get("needle_haystack")
    if meta is None:
        raise SystemExit("evalscope has no registered `needle_haystack` benchmark")

    missing = [k for k in ("system_prompt", "prompt_template", "local_path") if k not in dataset_args]
    if missing:
        raise SystemExit(f"{config_path}: suite {suite!r} is missing dataset_args {missing}")

    return {
        "system_prompt": dataset_args["system_prompt"],
        "prompt_template": dataset_args["prompt_template"],
        "corpus_dir": dataset_args["local_path"],
        "tokenizer": extra.get("tokenizer_path"),
        # The gate suites do not override these, so they are the benchmark's own defaults; a suite
        # that does override them (the novel-needle control) is honoured.
        "needles": extra.get("needles", meta.extra_params["needles"]["value"]),
        "question": extra.get(
            "retrieval_question", meta.extra_params["retrieval_question"]["value"]
        ),
    }


class _Stub:
    """Carries exactly the attributes `_get_context_tokens` / `_insert_needles` read."""

    def __init__(self, tokenizer, needles, context_lengths):
        self.tokenizer = tokenizer
        self.needles = needles
        self.context_lengths = context_lengths
        self.insertion_percentages: list[float] = []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="the ninfer_eval suite config")
    ap.add_argument("--suite", default=DEFAULT_SUITE, help="which suite's settings to read")
    ap.add_argument("--corpus-dir", default=None, help="override the config's local_path")
    ap.add_argument("--tokenizer", default=None, help="override the config's tokenizer_path")
    ap.add_argument("--context-lengths", type=int, nargs="+", default=[653000, 1046000])
    ap.add_argument("--depth", type=int, default=50)
    ap.add_argument("--windows", type=int, default=200, help="random 120-char windows to census")
    ap.add_argument("--window-chars", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--json-out", default=None, help="write the findings to this JSON file")
    args = ap.parse_args()

    from pathlib import Path
    from transformers import AutoTokenizer
    from evalscope.benchmarks.needle_haystack.needle_haystack_adapter import (
        NeedleHaystackAdapter as Adapter,
    )

    settings = load_suite_settings(args.config, args.suite)
    corpus_dir = args.corpus_dir or settings["corpus_dir"]
    tokenizer_path = args.tokenizer or settings["tokenizer"]
    if not tokenizer_path:
        raise SystemExit(
            f"{args.config}: suite {args.suite!r} has no tokenizer_path; pass --tokenizer"
        )
    prompt_template = settings["prompt_template"]
    system_prompt = settings["system_prompt"]
    needles = settings["needles"]
    question = settings["question"]
    print(
        f"settings from {args.config} [suite {args.suite}]: corpus={corpus_dir} "
        f"tokenizer={tokenizer_path} needle={needles[0].strip()!r}"
    )

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    rng = random.Random(args.seed)
    findings: list[dict] = []

    for subset, filename in FILES.items():
        path = Path(corpus_dir) / filename
        text = path.read_text(encoding="utf-8")
        corpus_tokens = len(tok.encode(text, add_special_tokens=False))
        print(f"\n=== subset `{subset}` <- {filename} ===")
        print(f"corpus: {len(text):,} chars, {corpus_tokens:,} tokens")

        for n in args.context_lengths:
            stub = _Stub(tok, needles, [n])
            tokens_context = Adapter._get_context_tokens(stub, text)
            tiles = 1
            grown = corpus_tokens
            # Reproduce the adapter's loop arithmetic for reporting (the call above already
            # performed it); a corpus at least as long as the tier is never grown.
            while grown < n:
                tiles += 1
                grown = corpus_tokens * tiles
            context = Adapter._insert_needles(stub, tokens_context, args.depth, n)

            prompt = prompt_template.format(context=context, question=question)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            rendered = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prompt_tokens = len(tok.encode(rendered, add_special_tokens=False))

            # Window-duplication census.
            w = args.window_chars
            counts: dict[int, int] = {}
            usable = len(context) - w
            for _ in range(args.windows):
                i = rng.randrange(0, usable)
                occurrences = context.count(context[i:i + w])
                counts[occurrences] = counts.get(occurrences, 0) + 1
            max_occ = max(counts)

            needle_body = needles[0].strip()
            needle_occurrences = context.count(needle_body)

            # Two independent conditions, reported separately because they have different
            # causes. `adapter tiles` is the corpus-repetition the benchmark itself performs
            # (the thing this corpus exists to eliminate). `max window occurrences` can
            # exceed 1 even with tiles=1 when the underlying books contain internally
            # repetitive passages -- e.g. War and Peace's table of contents, ~10 KB at the
            # head of bundle A, whose 120-char windows recur a dozen times. That matters only
            # for context lengths small enough to be drawn entirely from such a passage.
            tiling_verdict = "NO TILING" if tiles == 1 else f"TILED x{tiles}"
            window_verdict = "all windows unique" if max_occ == 1 else \
                f"max window occurrences {max_occ} (intra-corpus repetition)"
            verdict = "NO TILING" if tiles == 1 and max_occ == 1 else \
                f"{tiling_verdict}; {window_verdict}"
            print(
                f"  tier {n:>9,}: adapter tiles={tiles}  context={len(context):,} chars  "
                f"prompt={prompt_tokens:,} tokens (HF template)  "
                f"window occurrences {counts}  needle x{needle_occurrences}  -> {verdict}"
            )
            findings.append(
                {
                    "subset": subset,
                    "file": filename,
                    "corpus_chars": len(text),
                    "corpus_tokens": corpus_tokens,
                    "context_length_param": n,
                    "adapter_tiles": tiles,
                    "context_chars": len(context),
                    "prompt_tokens_hf_template": prompt_tokens,
                    "window_chars": w,
                    "windows_sampled": args.windows,
                    "window_occurrence_histogram": {str(k): v for k, v in sorted(counts.items())},
                    "needle_occurrences": needle_occurrences,
                    "max_window_occurrences": max_occ,
                    "verdict": verdict,
                }
            )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(findings, fh, indent=2)
            fh.write("\n")
        print(f"\nwrote {args.json_out}")
    # The exit status keys on the property this corpus exists to guarantee: the benchmark
    # never repeats the corpus. Intra-corpus window repetition is reported, not asserted.
    return 0 if all(f["adapter_tiles"] == 1 for f in findings) else 1


if __name__ == "__main__":
    sys.exit(main())
