#!/usr/bin/env python3
"""Compare V100 Duo context capacities with fixed 512-token code input and 256-token decode.

Runs the two engines sequentially. Each capacity uses a fresh resident model, one
discarded request and three measured requests. Capacity is not prompt occupancy.
"""
import argparse
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from compare_llama import EOG_TOKENS


ROOT = Path(__file__).resolve().parents[2]
PROMPT_TOKENS = 512
DECODE_TOKENS = 256
REPETITIONS = 3
WARMUP = 1
DEFAULT_CAPACITIES = [1024, 2048, 4096, 8192, 16384, 32768, 65536]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def summary_row(engine, capacity, actual_capacity, rates, prefill, accepted, drafted, raw):
    return {
        "engine": engine,
        "requested_capacity": capacity,
        "actual_capacity": actual_capacity,
        "prompt_tokens": PROMPT_TOKENS,
        "decode_tokens": DECODE_TOKENS,
        "repetitions": REPETITIONS,
        "warmup": WARMUP,
        "decode_wall_tok_s_mean": statistics.mean(rates),
        "decode_wall_tok_s_stdev": statistics.stdev(rates),
        "decode_wall_tok_s_reps": rates,
        "prefill_seconds_mean": statistics.mean(prefill),
        "prefill_tok_s_mean": statistics.mean(PROMPT_TOKENS / seconds for seconds in prefill),
        "accepted_tokens": sum(accepted),
        "drafted_tokens": sum(drafted),
        "acceptance_rate": sum(accepted) / sum(drafted) if sum(drafted) else 0,
        "raw_report": raw.name,
    }


def summarize_ninfer(raw, capacity):
    report = json.loads(raw.read_text())
    config = report["config"]
    require(config["max_context"] == capacity, "NInfer context capacity mismatch")
    require(config["repetitions"] == REPETITIONS and config["warmup"] == WARMUP,
            "NInfer repetition/warmup mismatch")
    require(len(report["tests"]) == 1, "expected exactly one NInfer workload")
    test = report["tests"][0]
    require(test["n_prompt"] == PROMPT_TOKENS and test["n_gen"] == DECODE_TOKENS,
            "NInfer prompt/output window mismatch")
    require(len(test["reps"]) == REPETITIONS, "incomplete NInfer repetitions")
    rates, prefill, accepted, drafted = [], [], [], []
    for rep in test["reps"]:
        generation = rep["generation"]
        tokens = generation["token_ids"]
        require(rep["generated_output_tokens"] == DECODE_TOKENS + 1
                and rep["decode_output_tokens"] == DECODE_TOKENS
                and len(tokens) == DECODE_TOKENS + 1
                and not EOG_TOKENS.intersection(tokens)
                and generation["finish_reason"] == "output_limit",
                "invalid NInfer fixed-window output; inspect " + str(raw))
        timing = rep["timings"]
        elapsed = timing["total_seconds"] - timing["first_token_seconds"]
        require(elapsed > 0 and timing["prefill_seconds"] > 0, "invalid NInfer timing")
        rates.append(DECODE_TOKENS / elapsed)
        prefill.append(timing["prefill_seconds"])
        accepted.append(rep["speculative"]["accepted_tokens"])
        drafted.append(rep["speculative"]["drafted_tokens"])
    row = summary_row("ninfer", capacity, report["memory"]["max_context"], rates,
                      prefill, accepted, drafted, raw)
    row["allocated_kv_capacity"] = report["memory"]["kv_capacity"]
    row["outputs_identical_across_repetitions"] = all(
        rep["generation"]["token_ids"] == test["reps"][0]["generation"]["token_ids"]
        for rep in test["reps"])
    return row


def summarize_llama(raw, capacity, server_log):
    report = json.loads(raw.read_text())
    require(report["prompt_tokens"] == PROMPT_TOKENS
            and report["decode_tokens"] == DECODE_TOKENS
            and report["warmup"] == WARMUP,
            "LM Studio workload mismatch")
    require(len(report["reps"]) == REPETITIONS
            and len(report["warmup_reps"]) == WARMUP,
            "incomplete LM Studio repetitions")
    require(all(rep["fixed_window_complete"] for rep in report["warmup_reps"] + report["reps"]),
            "invalid LM Studio fixed-window output; inspect " + str(raw))
    matches = re.findall(r"n_ctx_slot\s*=\s*(\d+)", server_log.read_text())
    require(bool(matches), "LM Studio server log lacks actual slot context capacity")
    actual_capacity = int(matches[-1])
    require(actual_capacity >= capacity, "LM Studio actual capacity below requested capacity")
    reps = report["reps"]
    row = summary_row(
        "llama", capacity, actual_capacity,
        [rep["decode_excluding_first_tok_s"] for rep in reps],
        [rep["timings"]["prompt_ms"] / 1000 for rep in reps],
        [rep["timings"]["draft_n_accepted"] for rep in reps],
        [rep["timings"]["draft_n"] for rep in reps], raw)
    row["outputs_identical_across_repetitions"] = all(
        rep["tokens"] == reps[0]["tokens"] for rep in reps)
    return row


def library_env(path):
    env = os.environ.copy()
    previous = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(path) + (":" + previous if previous else "")
    return env


def run_logged(command, log, env):
    print("Running:", " ".join(map(str, command)), flush=True)
    with log.open("w") as stream:
        subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def run_ninfer(args, capacity):
    raw = args.output_dir / f"ninfer-{capacity}.json"
    command = [args.ninfer, "--weights", args.weights, "--tp", "2", "--devices", "0,1",
               "--max-ctx", capacity, "--kv-dtype", "int8", "--prefill-chunk", "1024",
               "--mtp-draft-tokens", "3", "--lm-head-draft", "--corpus", args.corpus,
               "-pg", "512,256", "--warmup", WARMUP, "-r", REPETITIONS,
               "--capture-generation", "-o", "json", "--output-file", raw]
    write_json(args.output_dir / f"ninfer-{capacity}-command.json", list(map(str, command)))
    run_logged(command, args.output_dir / f"ninfer-{capacity}.log",
               library_env(f"{ROOT}/build/_deps/install/lib:/usr/local/cuda-12.8/lib64"))
    return summarize_ninfer(raw, capacity)


def stop_server(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_llama(args, capacity):
    # Refuse an occupied port: never query or terminate somebody else's server.
    with socket.socket() as connection:
        require(connection.connect_ex(("127.0.0.1", args.port)) != 0,
                f"port {args.port} is already occupied")
    raw = args.output_dir / f"llama-{capacity}.json"
    server_log = args.output_dir / f"llama-{capacity}-server.log"
    command = [args.llama_server, "--model", args.gguf, "--ctx-size", capacity,
               "--parallel", "1", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
               "--flash-attn", "on", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
               "--spec-draft-n-min", "0", "--threads", "16", "--threads-batch", "24",
               "--host", "127.0.0.1", "--port", args.port, "--no-webui"]
    write_json(args.output_dir / f"llama-{capacity}-command.json", list(map(str, command)))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Starting own LM Studio backend for capacity {capacity}", flush=True)
    with server_log.open("w") as stream:
        process = subprocess.Popen(list(map(str, command)), cwd=ROOT,
                                   env=library_env(args.llama_library_path),
                                   stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 600
            while True:
                require(process.poll() is None, f"LM Studio server exited; inspect {server_log}")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=2) as response:
                        if json.load(response).get("status") == "ok":
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                require(time.monotonic() < deadline,
                        f"LM Studio startup timed out; inspect {server_log}")
                time.sleep(0.5)
            run_logged(
                [sys.executable, ROOT / "tools/v100/compare_llama.py", "--url", url,
                 "--corpus", args.corpus, "--prompt-tokens", PROMPT_TOKENS,
                 "--decode-tokens", DECODE_TOKENS, "--warmup", WARMUP,
                 "--repetitions", REPETITIONS, "--output", raw],
                args.output_dir / f"llama-{capacity}.log", os.environ.copy())
        finally:
            stop_server(process)
    return summarize_llama(raw, capacity, server_log)


def publish_summary(args):
    rows = []
    for capacity in args.capacities:
        for engine in ("ninfer", "llama"):
            result = args.output_dir / f"{engine}-{capacity}-summary.json"
            if result.exists():
                rows.append(json.loads(result.read_text()))
    report = {"workload": "fixed code input; capacity sweep, not occupancy sweep",
              "corpus": str(args.corpus), "prompt_tokens": PROMPT_TOKENS,
              "decode_tokens": DECODE_TOKENS, "repetitions": REPETITIONS,
              "warmup": WARMUP, "rows": rows}
    write_json(args.output_dir / "summary.json", report)
    lines = ["| Capacity | Engine | Actual capacity | Decode tok/s (mean ± SD) | Prefill s | MTP acceptance |",
             "|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['requested_capacity']} | {row['engine']} | {row['actual_capacity']} | "
                     f"{row['decode_wall_tok_s_mean']:.2f} ± {row['decode_wall_tok_s_stdev']:.2f} | "
                     f"{row['prefill_seconds_mean']:.3f} | {row['acceptance_rate']:.2%} |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("ninfer", "llama", "both"), default="both")
    parser.add_argument("--capacities", nargs="+", type=int, default=DEFAULT_CAPACITIES)
    parser.add_argument("--corpus", type=Path, default=Path("/tmp/v100-code-512.ids"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ninfer", type=Path, default=ROOT / "build-v100/bench/ninfer_bench")
    parser.add_argument("--weights", type=Path,
                        default=Path("/Models/ninfer-V100-Duo/qwen3_8_27b_q4_k_m.ninfer"))
    parser.add_argument("--gguf", type=Path, default=Path(
        "/Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"))
    parser.add_argument("--llama-server", type=Path, default=Path(
        "/home/z/.lmstudio/extensions/backends/"
        "llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.33.0/llama-server"))
    parser.add_argument("--llama-library-path", type=Path, default=Path(
        "/home/z/.lmstudio/extensions/backends/vendor/linux-llama-cuda-vendor-v1"))
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()
    if any(capacity < 1024 for capacity in args.capacities):
        parser.error("capacity must be at least 1024 for the fixed P512/G256 workload")
    if len(set(args.capacities)) != len(args.capacities):
        parser.error("capacities must not contain duplicates")
    for name in ("corpus", "output_dir", "ninfer", "weights", "gguf", "llama_server",
                 "llama_library_path"):
        setattr(args, name, getattr(args, name).resolve())
    corpus = [int(token) for token in args.corpus.read_text().split()]
    if len(corpus) != PROMPT_TOKENS:
        parser.error("use the exact 512-token code corpus for both engines")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    for capacity in args.capacities:
        for engine, runner in (("ninfer", run_ninfer), ("llama", run_llama)):
            if args.engine in (engine, "both"):
                row = runner(args, capacity)
                write_json(args.output_dir / f"{engine}-{capacity}-summary.json", row)
                publish_summary(args)
                print(f"{engine} capacity={capacity}: "
                      f"{row['decode_wall_tok_s_mean']:.3f} ± "
                      f"{row['decode_wall_tok_s_stdev']:.3f} decode tok/s; "
                      f"prefill={row['prefill_seconds_mean']:.3f}s", flush=True)
    print(f"Summary: {args.output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
