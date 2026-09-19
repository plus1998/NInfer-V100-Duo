#!/usr/bin/env python3
"""Summarize the 1M soak: per-window logit-sanity table, VRAM stats, determinism.

Reads the artifacts `eval/run_qwen3_8_27b_nvfp4_1m_soak.sh` leaves for each pass
(`<prefix>.cli.log`, `<prefix>.tokens.txt`, `<prefix>.vram.csv`) and writes a small committed
summary to `--out-dir`. Exits nonzero when a gate fails, so the script's own exit status is the
verdict rather than something a reader has to reconstruct from prose.

The generated stream is the only sanity probe the CLI can supply: it has no per-token logit dump,
and adding one would mean streaming ~98k x 248k floats. What the stream does show is the shape a
NaN/inf-poisoned logit vector takes under argmax -- top-1 collapses onto a constant -- so the gate
is "no window is dominated by one token id" plus the longest constant run, with the distinct-token
ratio reported next to it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter


def read_tokens(path: str) -> list[int]:
    with open(path, encoding="utf-8") as handle:
        return [int(line) for line in handle if line.strip()]


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vocab_bound(tokenizer_json: str | None) -> int | None:
    """Largest valid token id + 1, read straight out of tokenizer.json (no tokenizers import)."""
    if not tokenizer_json or not os.path.isfile(tokenizer_json):
        return None
    with open(tokenizer_json, encoding="utf-8") as handle:
        root = json.load(handle)
    ids = [int(value) for value in root.get("model", {}).get("vocab", {}).values()]
    ids += [int(entry["id"]) for entry in root.get("added_tokens", [])]
    return max(ids) + 1 if ids else None


def special_token_id(tokenizer_json: str | None, content: str) -> int | None:
    """Id of a named special token, read from tokenizer.json. Never hardcode 248046."""
    if not tokenizer_json or not os.path.isfile(tokenizer_json):
        return None
    with open(tokenizer_json, encoding="utf-8") as handle:
        root = json.load(handle)
    for entry in root.get("added_tokens", []):
        if entry.get("content") == content:
            return int(entry["id"])
    return None


def parse_cli_log(path: str) -> dict:
    metrics: dict[str, str] = {}
    stages: dict[str, float] = {}
    metric_re = re.compile(r"^summary\s{2,}(\S.*?)\s{2,}(.*\S)\s*$")
    stage_re = re.compile(r"^(prepare|generate|load)\s{2,}(\S.*?)\s{2,}([0-9.]+) s\s*$")
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = metric_re.match(line)
            if match:
                metrics.setdefault(match.group(1), match.group(2))
                continue
            match = stage_re.match(line)
            if match:
                stages.setdefault(f"{match.group(1)}/{match.group(2)}", float(match.group(3)))
    return {"metrics": metrics, "stages": stages}


def window_table(tokens: list[int], window: int, share_limit: float, domain: int, run_limit: int,
                 vocab: int | None) -> list[dict]:
    rows = []
    for start in range(0, len(tokens), window):
        chunk = tokens[start:start + window]
        counts = Counter(chunk)
        top_id, top_count = counts.most_common(1)[0]
        run_id, run_len, best_run, best_run_id = None, 0, 0, None
        for token in chunk:
            if token == run_id:
                run_len += 1
            else:
                run_id, run_len = token, 1
            if run_len > best_run:
                best_run, best_run_id = run_len, run_id
        # Hard gate: the LM head's own width (TextConfig::token_domain). Anything outside it is
        # not a token the model can have produced.
        out_of_domain = sum(1 for t in chunk if t < 0 or t >= domain)
        # Informational: ids above the tokenizer's largest assigned id are unused head rows.
        out_of_vocab = 0 if vocab is None else sum(1 for t in chunk if t >= vocab)
        share = top_count / len(chunk)
        rows.append({
            "window": len(rows),
            "first_token_index": start,
            "tokens": len(chunk),
            "distinct": len(counts),
            "distinct_ratio": len(counts) / len(chunk),
            "top1_id": top_id,
            "top1_count": top_count,
            "top1_share": share,
            "longest_run_id": best_run_id,
            "longest_run": best_run,
            "out_of_domain": out_of_domain,
            "out_of_vocab": out_of_vocab,
            "pass": bool(share <= share_limit and out_of_domain == 0 and best_run <= run_limit),
        })
    return rows


def cycle_stats(tokens: list[int], end_of_turn: int | None = None, signature: int = 24) -> dict:
    """Period of the generated stream, and how many consecutive periods are exactly identical.

    Under `--ignore-eos` the model finishes its turn, the engine ignores the end-of-turn token, and
    a fresh turn is decoded from a context that now also holds the previous answer. Greedy decoding
    settles into a fixed-length episode. That accident is a sharp stability probe: an identical
    episode repeated across tens of thousands of growing KV positions means no argmax anywhere in
    the block flipped as the context grew.
    """
    if len(tokens) < signature * 2:
        return {"signature_len": signature, "starts": [], "identical_consecutive_max": 0}
    key = tuple(tokens[:signature])
    starts = [i for i in range(len(tokens) - signature) if tuple(tokens[i:i + signature]) == key]
    episodes = [tokens[a:b] for a, b in zip(starts, starts[1:])]
    # `best` counts EPISODES in the longest all-identical consecutive run, not comparisons.
    best, run = 0, 0
    best_start = None
    for index in range(1, len(episodes)):
        if episodes[index] == episodes[index - 1]:
            run = run + 1 if run else 2
            if run > best:
                best, best_start = run, starts[index - run + 1]
        else:
            run = 0
    lengths = sorted({len(e) for e in episodes})
    steady = episodes[-1] if episodes else []
    return {
        "signature_len": signature,
        "signature_matches": len(starts),
        "complete_episodes": len(episodes),
        "trailing_partial_tokens": len(tokens) - starts[-1] if starts else 0,
        "end_of_turn_id": end_of_turn,
        "end_of_turn_total": tokens.count(end_of_turn) if end_of_turn is not None else None,
        "end_of_turn_per_steady_episode": (steady.count(end_of_turn)
                                           if end_of_turn is not None and steady else None),
        "episodes": len(episodes),
        "first_start": starts[0] if starts else None,
        "episode_lengths": lengths,
        "identical_consecutive_max": best,
        "identical_block_first_index": best_start,
    }


def vram_stats(path: str, tolerance_mib: int = 16, startup_samples: int = 60) -> dict:
    """Per-series residency, and whether it is FLAT once the engine has finished starting up.

    "Flat" is deliberately not "never changes after the global peak": the KV pool settles by a few
    MiB in the first seconds after the weights land, so the strict form fails on a 4 MiB dip that
    happens before the request is even submitted and reports nothing about a leak. The gate here is
    the one the soak actually cares about -- once residency is inside `tolerance_mib` of its peak
    (which must happen within `startup_samples` ticks, i.e. during startup, not mid-decode), the
    whole rest of the run stays inside that band. A leak over 36 minutes of decode would blow the
    band open; a 4 MiB settle at t+10 s would not.
    """
    series: dict[str, list[tuple[str, int]]] = {}
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split(",")
            if len(parts) != 4 or not parts[3].isdigit():
                continue
            series.setdefault(f"{parts[1]} {parts[2]}", []).append((parts[0], int(parts[3])))
    out = {}
    for key, rows in sorted(series.items()):
        values = [value for _, value in rows]
        counts = Counter(values)
        peak = max(values)
        # The plateau: from the first sample on it to the last. What falls outside is the startup
        # ramp (weights still loading) and, on the final tick, the process tearing its context down
        # while nvidia-smi still lists it. Neither is a residency the run ever held while working.
        on_plateau = [i for i, v in enumerate(values) if v >= peak - tolerance_mib]
        steady_from, steady_to = on_plateau[0], on_plateau[-1]
        steady = values[steady_from:steady_to + 1]
        out[key] = {
            "samples": len(values),
            "first_sample_utc": rows[0][0],
            "last_sample_utc": rows[-1][0],
            "min_mib": min(values),
            "max_mib": peak,
            "max_gib": round(peak / 1024.0, 2),
            "distinct_values": [{"mib": mib, "samples": n} for mib, n in sorted(counts.items())],
            "samples_at_peak": counts[peak],
            "steady_from_index": steady_from,
            "steady_from_utc": rows[steady_from][0],
            "steady_to_index": steady_to,
            "steady_to_utc": rows[steady_to][0],
            "samples_before_plateau": steady_from,
            "samples_after_plateau": len(values) - 1 - steady_to,
            "steady_samples": len(steady),
            "steady_min_mib": min(steady),
            "steady_max_mib": max(steady),
            "steady_span_mib": max(steady) - min(steady),
            "growth_mib": steady[-1] - steady[0],
            "flat": bool(steady_from <= startup_samples
                         and len(values) - 1 - steady_to <= 5
                         and max(steady) - min(steady) <= tolerance_mib),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass-a", required=True, help="artifact prefix of pass A")
    parser.add_argument("--pass-b", required=True, help="artifact prefix of pass B")
    parser.add_argument("--prompt-json", required=True)
    parser.add_argument("--window", type=int, default=10000)
    parser.add_argument("--degenerate-share", type=float, default=0.95)
    parser.add_argument("--window-tokens", type=int, default=1048576)
    parser.add_argument("--gate-mib", type=int, default=30 * 1024)
    parser.add_argument("--max-constant-run", type=int, default=64,
                        help="longest run of one repeated token id allowed in a window; a NaN/inf "
                             "logit vector under argmax produces runs in the thousands")
    parser.add_argument("--token-domain", type=int, default=248077,
                        help="LM head width (TextConfig::token_domain)")
    parser.add_argument("--tokenizer-json",
                        default="/home/pc/models/ninfer-38/unsloth-nvfp4/tokenizer.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-stem", default="soak-summary",
                        help="basename stem for the two summary files; the run script keys it on "
                             "the window size so a small-window validation cannot overwrite the "
                             "tracked 1M summary")
    args = parser.parse_args()

    vocab = vocab_bound(args.tokenizer_json)
    end_of_turn = special_token_id(args.tokenizer_json, "<|im_end|>")
    passes = {}
    for name, prefix in (("pass-a", args.pass_a), ("pass-b", args.pass_b)):
        tokens = read_tokens(f"{prefix}.tokens.txt")
        log = parse_cli_log(f"{prefix}.cli.log")
        passes[name] = {
            "prefix": os.path.basename(prefix),
            "tokens_sha256": sha256_file(f"{prefix}.tokens.txt"),
            "generated_tokens": len(tokens),
            "prompt_tokens": int(log["metrics"].get("prompt tokens", "0")),
            "finish_reason": log["metrics"].get("finish reason", "?"),
            "prefill_speed": log["metrics"].get("prefill speed", "?"),
            "decode_speed": log["metrics"].get("decode speed", "?"),
            "sampling": log["metrics"].get("sampling", "?"),
            "kv_capacity": log["metrics"].get("KV capacity", "?"),
            "max_context": log["metrics"].get("max context", "?"),
            "rope": log["metrics"].get("rope", "?"),
            "workspace_peak": log["metrics"].get("gpu workspace peak", "?"),
            "stages": log["stages"],
            "vram": vram_stats(f"{prefix}.vram.csv"),
            "windows": window_table(tokens, args.window, args.degenerate_share,
                                    args.token_domain, args.max_constant_run, vocab),
            "cycle": cycle_stats(tokens, end_of_turn),
            "distinct_overall": len(set(tokens)),
            "_tokens": tokens,
        }

    a, b = passes["pass-a"], passes["pass-b"]
    identical = a["_tokens"] == b["_tokens"]
    for entry in passes.values():
        entry.pop("_tokens")

    gates = {
        "no_crash": all(p["finish_reason"] in ("context-capacity", "output-limit")
                        for p in passes.values()),
        "window_full": all(p["prompt_tokens"] + p["generated_tokens"] - 1 == args.window_tokens
                           for p in passes.values()),
        "logit_sanity": all(row["pass"] for p in passes.values() for row in p["windows"]),
        "vram_within_gate": all(series["max_mib"] <= args.gate_mib
                                for p in passes.values()
                                for key, series in p["vram"].items() if key.startswith("process ")),
        "vram_flat": all(series["flat"]
                         for p in passes.values()
                         for key, series in p["vram"].items() if key.startswith("process ")),
        "deterministic": bool(identical and a["tokens_sha256"] == b["tokens_sha256"]),
    }

    report = {
        "task": "7.2 1M soak",
        "prompt_json": args.prompt_json,
        "prompt_json_sha256": sha256_file(args.prompt_json),
        "window_tokens": args.window_tokens,
        "sanity_window": args.window,
        "degenerate_share_limit": args.degenerate_share,
        "max_constant_run_limit": args.max_constant_run,
        "vocab_bound": vocab,
        "token_domain": args.token_domain,
        "end_of_turn_id": end_of_turn,
        "gate_mib": args.gate_mib,
        "passes": passes,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f"{args.out_stem}.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
        handle.write("\n")

    lines = ["# 1,048,576-token soak (TP2 + YaRN x4, NVFP4, 2x RTX 5090 @ 400 W)", ""]
    lines.append("| | pass A | pass B |")
    lines.append("|---|---|---|")
    for label, key in (("artifact prefix", "prefix"), ("prompt tokens", "prompt_tokens"),
                       ("generated tokens", "generated_tokens"), ("finish reason", "finish_reason"),
                       ("prefill speed", "prefill_speed"), ("decode speed", "decode_speed"),
                       ("sampling", "sampling"), ("rope", "rope"),
                       ("max context", "max_context"), ("KV capacity", "kv_capacity"),
                       ("workspace peak", "workspace_peak"),
                       ("token-stream sha256", "tokens_sha256")):
        lines.append(f"| {label} | {a[key]} | {b[key]} |")
    for stage in ("generate/text prefill", "generate/decode", "generate/total",
                  "prepare/render/preprocess", "load/engine construction"):
        lines.append(f"| {stage} (s) | {a['stages'].get(stage, '-')} | {b['stages'].get(stage, '-')} |")
    lines.append("")
    lines.append(f"Prompt file `{os.path.basename(args.prompt_json)}` sha256 "
                 f"`{report['prompt_json_sha256']}` - both passes read the same bytes.")
    lines.append("")

    lines.append(f"## Logit-sanity windows (pass A, {args.window} generated tokens each)")
    lines.append("")
    lines.append("| # | first idx | tokens | distinct | distinct ratio | top-1 id | top-1 share | "
                 "longest run | out-of-domain | out-of-vocab | verdict |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|")
    for row in a["windows"]:
        lines.append(
            f"| {row['window']} | {row['first_token_index']} | {row['tokens']} | {row['distinct']} "
            f"| {row['distinct_ratio']:.4f} | {row['top1_id']} | {row['top1_share']:.4f} "
            f"| {row['longest_run']} | {row['out_of_domain']} | {row['out_of_vocab']} "
            f"| {'PASS' if row['pass'] else 'FAIL'} |")
    lines.append("")
    worst = max(a["windows"], key=lambda r: r["top1_share"])
    worst_run = max(a["windows"], key=lambda r: r["longest_run"])
    lines.append(f"Worst window by top-1 share: #{worst['window']} at {worst['top1_share']:.4f} "
                 f"(limit {args.degenerate_share}). Worst by longest constant run: "
                 f"#{worst_run['window']} at {worst_run['longest_run']} "
                 f"(limit {args.max_constant_run}). Pass B's window table is in the JSON; the two "
                 f"streams are byte-identical when the determinism gate passes.")
    lines.append("")

    cyc = a["cycle"]
    lines.append("## Generated-stream structure (pass A)")
    lines.append("")
    lines.append(
        f"`--ignore-eos` keeps decode running past the model's end-of-turn token, so the stream is "
        f"a repeating turn cycle rather than one answer. Pass A holds {cyc['signature_matches']} "
        f"matches of the opening 24-token signature: {cyc['complete_episodes']} complete episodes "
        f"of {cyc['episode_lengths']} tokens plus a {cyc['trailing_partial_tokens']}-token "
        f"truncated tail cut off by the context ceiling. Each steady episode is one substantive "
        f"answer followed by a short run of empty turn frames — "
        f"{cyc['end_of_turn_per_steady_episode']} end-of-turn tokens "
        f"(id {cyc['end_of_turn_id']}) per episode, {cyc['end_of_turn_total']} in the whole "
        f"stream: one closes the answer, the rest close empty `<|im_start|>…<|im_end|>` frames "
        f"before the next answer begins.")
    lines.append("")
    lines.append(
        f"The longest run of **exactly identical consecutive episodes** is "
        f"{cyc['identical_consecutive_max']} episodes, starting at generated index "
        f"{cyc['identical_block_first_index']}. That run is the sharpest numerical-stability "
        f"evidence here: the same argmax sequence is reproduced over tens of thousands of "
        f"additional KV positions without a single token flipping. It is also why the "
        f"distinct-token ratio is low ({a['distinct_overall']} distinct ids over "
        f"{a['generated_tokens']} tokens) — the vocabulary is bounded by the repeating turn, not "
        f"by a collapsing logit distribution.")
    lines.append("")
    lines.append(
        f"**What this does not cover.** Because the cycle locks in early, sanity windows 1-9 are "
        f"one probe repeated, not nine independent probes: they establish that decode at positions "
        f"950k-1,048,576 is numerically stable and reproducible, and they do not exercise "
        f"token-space diversity at all — 373 of 248,077 head rows are ever selected. A seeded "
        f"`temperature > 0` soak is the coverage complement: it would walk a far wider slice of the "
        f"vocabulary at the cost of the byte-identical determinism check this run relies on. "
        f"Neither run substitutes for the other, and only the greedy one was performed here.")
    lines.append("")
    lines.append(
        f"Gate note: `top-1 share` (limit {args.degenerate_share:.0%}) and `longest run` (limit "
        f"{args.max_constant_run}) and `out-of-domain` (limit 0) are all gated per window; "
        f"`distinct` and `distinct ratio` are **reported, not gated** — there is no defensible "
        f"threshold for them that a legitimate repeating turn cycle would pass.")
    lines.append("")
    lines.append("## VRAM (nvidia-smi, sampled every 2 s for the life of each pass)")
    lines.append("")
    lines.append("| pass | series | samples | min MiB | max MiB | max GiB | samples at peak | "
                 "plateau samples | plateau span MiB | growth MiB | flat |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:-:|")
    for name, entry in passes.items():
        for key, series in entry["vram"].items():
            lines.append(f"| {name} | {key} | {series['samples']} | {series['min_mib']} | "
                         f"{series['max_mib']} | {series['max_gib']} | {series['samples_at_peak']} "
                         f"| {series['steady_samples']} | {series['steady_span_mib']} "
                         f"| {series['growth_mib']} | {'yes' if series['flat'] else 'NO'} |")
    lines.append("")
    lines.append("Plateau = from the first sample within 16 MiB of the peak to the last one; the "
                 "samples outside it are the startup ramp and the final teardown tick. `growth "
                 "MiB` is the last plateau sample minus the first.")
    lines.append("")
    lines.append("Only the `process ...` rows are gated: a `gpu N` row also counts memory other "
                 "processes hold (GPU 0 carries ~1 GiB of desktop allocation on this machine) and "
                 "drops to that baseline the moment the run exits.")
    lines.append("")

    lines.append("## Gates")
    lines.append("")
    lines.append("| gate | verdict |")
    lines.append("|---|:-:|")
    for gate, ok in gates.items():
        lines.append(f"| {gate} | {'PASS' if ok else 'FAIL'} |")
    lines.append("")

    md_path = os.path.join(args.out_dir, f"{args.out_stem}.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    for gate, ok in gates.items():
        print(f"  gate  {gate:<20} {'PASS' if ok else 'FAIL'}")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
