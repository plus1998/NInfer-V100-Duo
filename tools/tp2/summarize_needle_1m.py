#!/usr/bin/env python3
"""Summarize the needle runs into one small table (Markdown + JSON).

Reads the per-job EvalScope reports out of one or more `eval/runs/<stamp>-<id>/` directories
and, optionally, the `ninfer-serve --request-log-jsonl` files, and emits:

  * a per-tier x per-depth x per-bundle retrieval grid (1 = needle retrieved), and
  * per-tier prefill / decode throughput from the server's own phase timings.

Bundle naming: EvalScope's needle adapter hardcodes its two subset ids (`english`,
`chinese`); the needle corpus writes bundle A into the `english` slot and bundle B into
the `chinese` slot, and BOTH are English Gutenberg text (see
eval/configs/qwen3_8_27b_nvfp4_needle_1m.yaml). This script relabels them A/B so the table
does not read as a language comparison, which it is not.

Usage:
    eval/.venv/bin/python tools/tp2/summarize_needle_1m.py \
        --run eval/runs/<ninfer-655k-and-1m-run> \
        --run eval/runs/<vllm-655k-run> \
        --requests eval/server-logs/needle1m-ninfer-all-<stamp>.requests.jsonl \
        --out-json eval/results/needle-1m/results.json \
        --out-md   eval/results/needle-1m/results.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys

SUBSET_LABEL = {"english": "bundle A", "chinese": "bundle B"}
METRIC_RE = re.compile(r"^Context#(\d+) Depth#(\d+)$")


def load_reports(run_dir: str) -> list[dict]:
    """One entry per (job, context length, depth) with the two per-bundle scores."""
    rows: list[dict] = []
    pattern = os.path.join(run_dir, "backends", "*", "reports", "*", "needle_haystack.json")
    for path in sorted(glob.glob(pattern)):
        job = path.split(os.sep)[-4]
        model = path.split(os.sep)[-2]
        with open(path, encoding="utf-8") as fh:
            report = json.load(fh)
        for metric in report.get("metrics", []):
            m = METRIC_RE.match(metric["name"])
            if not m:
                continue
            context_length, depth = int(m.group(1)), int(m.group(2))
            subsets: dict[str, float] = {}
            for category in metric.get("categories", []):
                for subset in category.get("subsets", []):
                    subsets[subset["name"]] = float(subset["score"])
            rows.append(
                {
                    "run_dir": run_dir,
                    "job": job,
                    "model": model,
                    "context_length_param": context_length,
                    "depth_percent": depth,
                    "scores": subsets,
                    "mean": float(metric["score"]),
                }
            )
    return rows


def load_requests(paths: list[str]) -> list[dict]:
    out: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("event") != "request_done":
                    continue
                result = rec.get("result", {})
                timings = rec.get("timings_seconds", {})
                prompt = int(result.get("prompt_tokens", 0))
                prefill_s = float(timings.get("prefill", 0.0))
                decode_s = float(timings.get("decode", 0.0))
                completion = int(result.get("completion_tokens", 0))
                out.append(
                    {
                        "source": os.path.basename(path),
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "prefill_seconds": prefill_s,
                        "decode_seconds": decode_s,
                        "ttft_seconds": float(timings.get("ttft", 0.0)),
                        "total_seconds": float(timings.get("total", 0.0)),
                        "prefill_tok_s": prompt / prefill_s if prefill_s > 0 else None,
                        # Decode rate over the generated tokens after the first, which is the
                        # rate the engine reports for the decode phase.
                        "decode_tok_s": completion / decode_s if decode_s > 0 else None,
                        "prefix_reuse_path": result.get("prefix_reuse_path"),
                        "prefix_cache_hit_tokens": result.get("prefix_cache_hit_tokens"),
                        "finish_reason": result.get("finish_reason"),
                    }
                )
    return out


def bucket(prompt_tokens: int) -> int:
    """Group requests whose prompt lengths differ only by the needle insertion point."""
    return round(prompt_tokens, -3)


def render_markdown(rows: list[dict], requests: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Needle-in-a-haystack results, 653k and 1,046k tokens\n")
    lines.append(
        "Scores are EvalScope `judge_strategy: rule` (strict exact match after "
        "normalisation); 1 = the inserted needle was reproduced verbatim.\n"
        "`bundle A` / `bundle B` are the two disjoint English Gutenberg book sets, carried "
        "by the adapter's `english` / `chinese` subset slots.\n"
    )
    keys = sorted({(r["job"], r["context_length_param"]) for r in rows})
    for job, ctx in keys:
        subset_rows = [r for r in rows if r["job"] == job and r["context_length_param"] == ctx]
        subset_rows.sort(key=lambda r: r["depth_percent"])
        model = subset_rows[0]["model"]
        total = sum(sum(r["scores"].values()) for r in subset_rows)
        count = sum(len(r["scores"]) for r in subset_rows)
        lines.append(f"\n## {job} -- model `{model}`, haystack {ctx:,} tokens\n")
        lines.append("| depth % | bundle A | bundle B | depth total |")
        lines.append("|---:|:-:|:-:|:-:|")
        for r in subset_rows:
            a = r["scores"].get("english")
            b = r["scores"].get("chinese")
            fmt = lambda v: "-" if v is None else f"{v:.0f}"
            got = sum(v for v in (a, b) if v is not None)
            n = sum(1 for v in (a, b) if v is not None)
            lines.append(f"| {r['depth_percent']} | {fmt(a)} | {fmt(b)} | {got:.0f} / {n} |")
        lines.append(f"| **total** | | | **{total:.0f} / {count}** |")

    if requests:
        lines.append("\n## Server-side throughput (ninfer-serve phase timings)\n")
        lines.append(
            "| prompt tokens (bucket) | n | prefill s (mean) | prefill tok/s (mean) | "
            "decode tok/s (mean) | gen tokens (mean) |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|")
        buckets: dict[int, list[dict]] = {}
        for req in requests:
            buckets.setdefault(bucket(req["prompt_tokens"]), []).append(req)
        for key in sorted(buckets):
            group = buckets[key]
            pf = [r["prefill_tok_s"] for r in group if r["prefill_tok_s"]]
            dec = [r["decode_tok_s"] for r in group if r["decode_tok_s"]]
            lines.append(
                f"| ~{key:,} | {len(group)} | "
                f"{statistics.mean(r['prefill_seconds'] for r in group):.1f} | "
                f"{statistics.mean(pf):.1f} | "
                f"{statistics.mean(dec):.2f} | "
                f"{statistics.mean(r['completion_tokens'] for r in group):.1f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="an eval/runs/<...> directory")
    ap.add_argument("--requests", action="append", default=[], help="a ninfer-serve request log")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    rows: list[dict] = []
    for run_dir in args.run:
        found = load_reports(run_dir)
        if not found:
            print(f"warning: no needle_haystack.json under {run_dir}", file=sys.stderr)
        rows.extend(found)
    requests = load_requests(args.requests)

    md = render_markdown(rows, requests)
    print(md)
    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as fh:
            fh.write(md)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        payload = {
            "subset_labels": SUBSET_LABEL,
            "cells": rows,
            "requests": requests,
        }
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
