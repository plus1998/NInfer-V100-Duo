#!/usr/bin/env bash
# MTP3 speculative decoding at 1,048,576 tokens under TP2 + YaRN x4.
#
# MTP3 at tp2 was first measured at 262,144 tokens: +0.65 GiB/device over MTP-off, 57.96 %
# acceptance at 250k. The boot gate then booted 1M at 27.41 GiB/device, and the soak ran there.
# This script asks the one question those left open: does MTP3 still fit inside the 30 GiB/device
# budget at 1M, and what does it cost or buy there.
#
# Everything is driven through the CLI (`build/apps/ninfer`), not the server, for two reasons:
# the CLI prints the per-device load summary the memory gate reads, and `--print-token-ids` makes
# the MTP-on / MTP-off comparison an exact token-id comparison rather than a string comparison.
#
# Steps
#   boot            1M + MTP3, "Say hello.", 16 tokens. The memory gate. Honours
#                   NINFER_1M_MAX_CONTEXT so the same step is the probe for a ceiling bisect
#                   (multiples of 32,768) if 1M does not fit.
#   prompts         materialise the three prompts replayed here:
#                     needle-a / needle-b -- the needle campaign's 1,046,000-token haystack at
#                       depth 50, bundles A and B, lifted VERBATIM out of that campaign's own
#                       prediction records (system + user turns; the record's `assistant` turn is
#                       the expected answer and is dropped). Replaying those bytes is what makes
#                       the MTP-on comparison a comparison and not a new experiment.
#                     soak -- the soak's ~950,000-token novel-continuation prompt, reused as
#                       written (same file, same sha256).
#   needle-a-mtp3   bundle A, depth 50, MTP3, greedy, 512 tokens, token ids captured.
#   needle-a-off    the same prompt with MTP off. Greedy MTP verification accepts a draft only
#                   when it equals the token argmax would have produced, so these two id streams
#                   must be identical; this step is the reference half of that claim.
#   needle-b-mtp3   bundle B, depth 50, MTP3. Compared against the recorded needle answer text.
#   decode-greedy   the soak prompt, greedy, --ignore-eos, 12,000 tokens. The soak showed
#                   greedy + --ignore-eos settles into a repeating ~1,903-token answer turn, so
#                   this is the REPETITIVE stream: its acceptance rate is an optimistic bound.
#   decode-sampled  the same prompt and length with temperature sampling, which does not fall
#                   into the identical-turn loop. Caveat recorded in the report: sampled
#                   acceptance and greedy acceptance are different algorithms, so the two rows
#                   bound rather than bracket each other. Measured at 25.4 % novel 16-grams, this
#                   stream is less repetitive than the greedy one but is not novel text either.
#   decode-novel    the SAME greedy configuration stopped at the token where the loop begins.
#                   Measured from `decode-greedy`'s id stream: its first 1,341 tokens are 100 %
#                   novel 16-grams and everything after index 1,341 is a 187-token block repeated
#                   57 times. Cutting there is the only greedy, non-repeating acceptance number
#                   this prompt can yield -- shorter than the 10,000 tokens originally targeted,
#                   because at 1M-class context greedy + --ignore-eos has no 10,000-token novel
#                   stretch to measure.
#   analyze         token-id diff, repetition statistics, acceptance/throughput table ->
#                   eval/results/mtp3-1m/.
#
# GPU discipline: every step refuses to launch while any compute process holds GPU memory,
# asserts the list is empty again after exit, and never signals a process it did not
# start. The VRAM sampler carries the power columns added in commit 58d6566: both cards
# on this host are capped at 400 W against 600/575 W stock, and every timing below is a
# power-limited measurement.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cli_bin="${repo_dir}/build/apps/ninfer"
artifact="${NINFER_QWEN3_8_27B_NVFP4_ARTIFACT:-/home/pc/models/ninfer-38/huihui-nvfp4/qwen3_8_27b_nvfp4.ninfer}"
log_dir="${repo_dir}/eval/server-logs"
work_dir="${repo_dir}/eval/runs/mtp3-1m"
results_dir="${repo_dir}/eval/results/mtp3-1m"

# The 1M needle run. Its prediction records are the source of the two needle prompts.
needle_run_dir="${NINFER_MTP3_NEEDLE_RUN:-${repo_dir}/eval/runs/20260828T095453Z-7e01c791}"
needle_pred_dir="${needle_run_dir}/backends/needle_1m_ninfer_1m/predictions/qwen3.8-27b"
# The soak prompt, reused byte for byte.
soak_prompt="${NINFER_MTP3_SOAK_PROMPT:-${repo_dir}/eval/runs/soak-1m/soak-prompt-950000.messages.json}"

max_context="${NINFER_1M_MAX_CONTEXT:-1048576}"
gate_mib=$((30 * 1024))
decode_tokens="${NINFER_MTP3_DECODE_TOKENS:-12000}"
# The novel prefix of the greedy stream, measured from `decode-greedy`'s own ids (see analyze's
# `repetition`): tokens 0..1340 are 100 % novel 16-grams, index 1341 onward is a 187-token loop.
novel_tokens="${NINFER_MTP3_NOVEL_TOKENS:-1341}"
needle_tokens="${NINFER_MTP3_NEEDLE_TOKENS:-512}"
vram_sample_seconds="${NINFER_VRAM_SAMPLE_SECONDS:-2}"

tokenizer_python="${NINFER_1M_TOKENIZER_PYTHON:-${repo_dir}/eval/.venv/bin/python}"

needle_a_prompt="${work_dir}/needle-1046k-depth50-bundleA.messages.json"
needle_b_prompt="${work_dir}/needle-1046k-depth50-bundleB.messages.json"

# The 1M boot-gate configuration, verbatim.
base_flags=(
    --tp 2
    --devices "${NINFER_1M_DEVICES:-0,1}"
    --rope yarn
    --yarn-factor 4.0
    --yarn-origin 262144
    --max-context "${max_context}"
    --kv-dtype int8
    --kv-capacity "${NINFER_1M_KV_CAPACITY:-auto}"
)
if [[ -n "${NINFER_1M_PREFILL_CHUNK:-}" ]]; then
    base_flags+=(--prefill-chunk "${NINFER_1M_PREFILL_CHUNK}")
fi

# The MTP3 configuration, verbatim.
mtp_flags=(--spec mtp --draft-tokens "${NINFER_MTP3_DRAFT_TOKENS:-3}" --lm-head-draft)

usage() {
    cat >&2 <<'EOF'
usage: run_qwen3_8_27b_nvfp4_1m_mtp3.sh <step>

steps
  boot            1M + MTP3 boot and 16-token generation; the <=30 GiB/device memory gate.
                  NINFER_1M_MAX_CONTEXT overrides the window, so this step doubles as the
                  probe for a ceiling bisect if 1M does not fit.
  prompts         build the two needle prompts from the needle records; check the soak prompt.
  needle-a-mtp3   bundle A, depth 50, 1,046k, MTP3, greedy.
  needle-a-off    bundle A, depth 50, 1,046k, MTP off, greedy  (the id-identity reference).
  needle-b-mtp3   bundle B, depth 50, 1,046k, MTP3, greedy.
  decode-greedy   ~950k prompt, greedy, --ignore-eos, 12,000 tokens  (repetitive stream).
  decode-sampled  ~950k prompt, sampled,  --ignore-eos, 12,000 tokens (less repetitive stream).
  decode-novel    ~950k prompt, greedy,  --ignore-eos, 1,341 tokens -- the greedy stream's novel
                  prefix, i.e. acceptance before the answer turn starts looping.
  analyze         write eval/results/mtp3-1m/{mtp3-1m-summary.json,mtp3-1m-summary.md}.
EOF
}

if [[ $# -ne 1 ]]; then usage; exit 2; fi
step="$1"
case "${step}" in
    boot|prompts|needle-a-mtp3|needle-a-off|needle-b-mtp3|decode-greedy|decode-sampled|decode-novel|analyze) ;;
    *) usage; exit 2 ;;
esac

for required in "${cli_bin}" "${artifact}"; do
    [[ -f "${required}" ]] || { echo "missing required file: ${required}" >&2; exit 1; }
done
[[ -x "${cli_bin}" ]] || { echo "${cli_bin} must be executable" >&2; exit 1; }
mkdir -p -- "${log_dir}" "${work_dir}" "${results_dir}"

sampler_pid=""
cleanup() {
    if [[ -n "${sampler_pid}" ]] && kill -0 "${sampler_pid}" 2>/dev/null; then
        kill -TERM "${sampler_pid}" 2>/dev/null || true
        wait "${sampler_pid}" 2>/dev/null || true
    fi
    sampler_pid=""
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# CSV: timestamp_utc,kind,key,used_mib[,power_limit_w,power_draw_w]
#   kind=gpu      key=<index>          includes other processes' memory; carries the power columns.
#   kind=process  key=<pid>@gpu<index> this run's own residency on that card.
sample_vram() {
    local out="$1" interval="$2" uuid_map now
    printf 'timestamp_utc,kind,key,used_mib,power_limit_w,power_draw_w\n' >"${out}"
    uuid_map="$(nvidia-smi --query-gpu=uuid,index --format=csv,noheader,nounits 2>/dev/null \
        | tr -d '[:blank:]')"
    while true; do
        now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        nvidia-smi --query-gpu=index,memory.used,power.limit,power.draw \
                --format=csv,noheader,nounits \
            | awk -v ts="${now}" -F', *' '{print ts ",gpu," $1 "," $2 "," $3 "," $4}' \
            >>"${out}" 2>/dev/null || true
        nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits \
            | awk -v ts="${now}" -v map="${uuid_map}" -F', *' '
                BEGIN {
                    rows = split(map, line, "\n")
                    for (i = 1; i <= rows; ++i) {
                        if (split(line[i], field, ",") == 2) { index_of[field[1]] = field[2] }
                    }
                }
                { gpu = ($2 in index_of) ? index_of[$2] : $2
                  print ts ",process," $1 "@gpu" gpu "," $3 }' >>"${out}" 2>/dev/null || true
        sleep "${interval}"
    done
}

# Prints memory peaks and the power condition; nonzero when a per-process peak exceeds the gate.
report_vram_peaks() {
    local out="$1"
    if [[ ! -s "${out}" ]]; then echo "  (no vram samples collected)"; return 0; fi
    awk -F, -v gate="${gate_mib}" 'NR > 1 && $4 != "" && $4 + 0 == $4 {
        key = $2 " " $3
        if ($4 + 0 > peak[key]) { peak[key] = $4 + 0 }
        ticks[key]++
        if ($2 == "gpu" && NF >= 6 && $5 + 0 == $5 && $6 + 0 == $6) {
            limit[$3] = $5 + 0
            if ($6 + 0 > draw_peak[$3]) { draw_peak[$3] = $6 + 0 }
            draw_sum[$3] += $6 + 0
            draw_n[$3]++
        }
    }
    END {
        over = 0
        for (key in peak) {
            printf "  peak %-24s %8d MiB  (%6.2f GiB, %4d samples)\n", key, peak[key],
                   peak[key] / 1024, ticks[key]
            if (key ~ /^process / && peak[key] > gate) { over = 1 }
        }
        for (g in draw_n) {
            printf "  power gpu %-18s limit %.0f W, draw peak %.0f W, mean %.0f W (%d samples)\n",
                   g, limit[g], draw_peak[g], draw_sum[g] / draw_n[g], draw_n[g]
        }
        exit over
    }' "${out}" | sort
    return "${PIPESTATUS[0]}"
}

report_load_summary() {
    local err_log="$1"
    grep -E '^summary +(rope|tensor parallel|gpu[01] )' "${err_log}" || true
    awk -v gate="${gate_mib}" '
        $1 == "summary" && $2 ~ /^gpu[01]$/ && $3 == "reserved" {
            value = $4 + 0
            mib = ($5 == "GiB") ? value * 1024 : value
            printf "  gate  %s reserved %8.2f GiB  %s (bound 30.00 GiB)\n", $2, mib / 1024,
                   (mib <= gate) ? "PASS" : "FAIL"
            if (mib > gate) { over = 1 }
        }
        END { exit over }
    ' "${err_log}"
}

summary_value() { grep -E "^summary +$1" "$2" | head -1 | sed -E "s/^summary +$1 +//"; }

compute_app_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | tr -d '[:blank:]' | grep -E '^[0-9]+$' || true
}

record_compute_apps() {
    local label="$1" out="$2" rows power
    rows="$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv 2>&1)"
    power="$(nvidia-smi --query-gpu=index,power.limit,power.max_limit,power.draw --format=csv 2>&1)"
    {
        printf '=== compute processes %s (%s) ===\n' "${label}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '%s\n' "${rows}"
        printf '=== power condition %s ===\n' "${label}"
        printf '%s\n' "${power}"
    } >>"${out}"
    printf 'compute processes %s:\n' "${label}"; printf '%s\n' "${rows}" | sed 's/^/  /'
    printf 'power %s:\n' "${label}";             printf '%s\n' "${power}" | sed 's/^/  /'
}

# -------------------------------------------------------------------------------------------
# prompts
# -------------------------------------------------------------------------------------------
build_prompts() {
    if [[ ! -s "${soak_prompt}" ]]; then
        echo "missing soak prompt: ${soak_prompt}" >&2
        echo "rebuild it with: eval/run_qwen3_8_27b_nvfp4_1m_soak.sh prompt" >&2
        return 1
    fi
    echo "soak prompt (reused verbatim): ${soak_prompt}"
    echo "  sha256 $(sha256sum "${soak_prompt}" | cut -d' ' -f1)"

    if [[ -s "${needle_a_prompt}" && -s "${needle_b_prompt}" ]]; then
        echo "needle prompts already built:"
    else
        [[ -d "${needle_pred_dir}" ]] || {
            echo "missing needle prediction directory: ${needle_pred_dir}" >&2
            echo "override the run with NINFER_MTP3_NEEDLE_RUN" >&2
            return 1
        }
        "${tokenizer_python:-python3}" - "${needle_pred_dir}" "${needle_a_prompt}" \
            "${needle_b_prompt}" <<'PY'
import json
import sys

pred_dir, out_a, out_b = sys.argv[1], sys.argv[2], sys.argv[3]

# EvalScope's subset ids: `english` is bundle A, `chinese` is bundle B (both English; the ids are
# the adapter's hardcoded file slots, see eval/configs/qwen3_8_27b_nvfp4_needle_1m.yaml).
for subset, out_path, label in (("english", out_a, "A"), ("chinese", out_b, "B")):
    record = None
    with open(f"{pred_dir}/needle_haystack_{subset}.jsonl", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if (item.get("metadata") or {}).get("depth_percent") == 50:
                record = item
                break
    if record is None:
        raise SystemExit(f"no depth-50 record in subset {subset}")

    # The record carries system + user + assistant. The assistant turn is the EXPECTED answer the
    # scorer compares against, not part of the request: replaying it would hand the model its own
    # answer. Keep the request turns only, in order.
    messages = [
        {"role": message["role"], "content": message["content"]}
        for message in record["messages"]
        if message["role"] in ("system", "user")
    ]
    if [m["role"] for m in messages] != ["system", "user"]:
        raise SystemExit(f"unexpected request shape in subset {subset}: {[m['role'] for m in messages]}")

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(messages, handle)

    served = record["model_output"]["choices"][0]["message"]["content"]
    usage = record["model_output"]["usage"]
    print(
        f"bundle {label} ({subset}): depth 50, record reports {usage['input_tokens']} input tokens, "
        f"{usage['output_tokens']} output tokens"
    )
    print(f"  recorded answer (MTP off): {served!r}")
    with open(out_path + ".mtp-off-answer.txt", "w", encoding="utf-8") as handle:
        handle.write(served)
PY
    fi
    for prompt in "${needle_a_prompt}" "${needle_b_prompt}"; do
        echo "  $(sha256sum "${prompt}" | tr -s ' ' | cut -d' ' -f1)  ${prompt}"
    done
}

# -------------------------------------------------------------------------------------------
# a CLI run
# -------------------------------------------------------------------------------------------
run_cli() {
    local name="$1"; shift
    local run_stamp prefix err_log out_log vram_log smi_log tok_log rc=0 started ended

    run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    prefix="${log_dir}/qwen3_8_27b_nvfp4-1m-mtp3-${name}-${run_stamp}"
    err_log="${prefix}.cli.log"; out_log="${prefix}.reply.txt"
    vram_log="${prefix}.vram.csv"; smi_log="${prefix}.nvidia-smi.txt"
    tok_log="${prefix}.tokens.txt"

    echo
    echo "=== step ${name} (max-context ${max_context}) ==="
    echo "cli log:   ${err_log}"
    echo "reply:     ${out_log}"
    echo "vram log:  ${vram_log}"
    echo "smi log:   ${smi_log}"
    echo "command:   ${cli_bin} ${artifact} ${base_flags[*]} $*"

    : >"${smi_log}"
    record_compute_apps "before launch" "${smi_log}"
    if [[ -n "$(compute_app_pids)" ]]; then
        echo "  gate  no compute process before launch  FAIL" | tee -a "${smi_log}"
        echo "another process is already using a GPU; this script will not start alongside it" >&2
        echo "(it never signals a process it did not start -- stop the other job first)" >&2
        return 1
    fi
    echo "  gate  no compute process before launch  PASS" | tee -a "${smi_log}"

    sample_vram "${vram_log}" "${vram_sample_seconds}" &
    sampler_pid=$!

    started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    set +e
    "${cli_bin}" "${artifact}" "${base_flags[@]}" "$@" >"${out_log}" 2>"${err_log}"
    rc=$?
    set -e
    ended="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    if [[ -n "${sampler_pid}" ]] && kill -0 "${sampler_pid}" 2>/dev/null; then
        kill -TERM "${sampler_pid}" 2>/dev/null || true
        wait "${sampler_pid}" 2>/dev/null || true
    fi
    sampler_pid=""

    echo "cli exit: ${rc}   (${started} -> ${ended})"
    if [[ "${rc}" -ne 0 ]]; then
        echo "--- last 40 log lines ---"; tail -40 "${err_log}"; return "${rc}"
    fi

    grep -E '^tokens +generated ids' "${err_log}" \
        | sed -E 's/^tokens +generated ids +//' | tr ' ' '\n' | grep -E '^[0-9]+$' >"${tok_log}" || true
    echo "  token ids captured: $(wc -l <"${tok_log}")"

    echo "--- load summary (per device) ---"
    local summary_ok=0
    report_load_summary "${err_log}" || summary_ok=1

    echo "--- nvidia-smi peaks / power during the run ---"
    local sampler_ok=0
    report_vram_peaks "${vram_log}" || sampler_ok=1
    if [[ "${sampler_ok}" -eq 0 ]]; then
        echo "  gate  every per-process peak <= 30.00 GiB  PASS"
    else
        echo "  gate  a per-process peak exceeded 30.00 GiB  FAIL"
    fi

    echo "--- generation ---"
    local field
    for field in 'prompt tokens' 'generated tokens' 'finish reason' 'prefill speed' 'decode speed'; do
        printf '  %-18s %s\n' "${field}" "$(summary_value "${field}" "${err_log}")"
    done
    grep -E '^generate +(text prefill|decode)' "${err_log}" | sed 's/^/  /' || true
    echo "--- speculative ---"
    grep -E '^summary +mtp ' "${err_log}" | sed 's/^/  /' || echo "  (none reported: MTP off)"

    echo "--- clean exit ---"
    record_compute_apps "after exit" "${smi_log}"
    local leftover
    leftover="$(compute_app_pids)"
    if [[ -z "${leftover}" ]]; then
        echo "  gate  no compute process after exit  PASS" | tee -a "${smi_log}"
    else
        echo "  gate  compute processes still resident after exit: $(echo "${leftover}" | tr '\n' ' ') FAIL" \
            | tee -a "${smi_log}"
        return 1
    fi

    ln -sfn "$(basename "${prefix}")" "${log_dir}/qwen3_8_27b_nvfp4-1m-mtp3-${name}.latest"
    if [[ "${summary_ok}" -ne 0 || "${sampler_ok}" -ne 0 ]]; then
        echo "step ${name}: MEMORY GATE FAILED"; return 1
    fi
    echo "step ${name}: OK"
}

require_prompt() {
    [[ -s "$1" ]] || { echo "prompt missing; run '$0 prompts' first: $1" >&2; return 1; }
}

# -------------------------------------------------------------------------------------------
# analyze
# -------------------------------------------------------------------------------------------
analyze() {
    "${tokenizer_python:-python3}" - "${log_dir}" "${results_dir}" "${work_dir}" \
        "${max_context}" <<'PY'
import json
import re
import sys
from pathlib import Path

log_dir, results_dir, work_dir, max_context = (
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
)

STEPS = ["boot", "needle-a-mtp3", "needle-a-off", "needle-b-mtp3",
         "decode-greedy", "decode-sampled", "decode-novel"]


def resolve(step):
    """The `<step>.latest` symlink points at a run PREFIX, not at a file.

    `Path.exists()` follows the link and would therefore always say no; the link's own existence
    is what has to be tested.
    """
    link = log_dir / f"qwen3_8_27b_nvfp4-1m-mtp3-{step}.latest"
    if not link.is_symlink():
        return None
    return log_dir / link.readlink()


def summary_rows(text):
    """Parse the CLI's summary table.

    `print_metric` writes `setw(12) "summary"` then `setw(26) label` then the value, so the
    columns are fixed and a label that happens to fill its field still parses. Falls back to a
    whitespace split for any line that is shorter than the value column.
    """
    rows = {}
    for line in text.splitlines():
        if not line.startswith("summary"):
            continue
        label, value = line[12:38].strip(), line[38:].strip()
        if not label:
            match = re.match(r"^summary\s+(.+?)\s\s+(.*)$", line)
            if not match:
                continue
            label, value = match.group(1).strip(), match.group(2).strip()
        rows.setdefault(label, value)
    return rows


def vram_peaks(path):
    peaks, power = {}, {}
    if not path.exists():
        return peaks, power
    with path.open() as handle:
        next(handle, None)
        for line in handle:
            field = line.rstrip("\n").split(",")
            if len(field) < 4 or not field[3].strip().isdigit():
                continue
            key = f"{field[1]} {field[2]}"
            peaks[key] = max(peaks.get(key, 0), int(field[3]))
            if field[1] == "gpu" and len(field) >= 6:
                try:
                    limit, draw = float(field[4]), float(field[5])
                except ValueError:
                    continue
                entry = power.setdefault(field[2], {"limit_w": limit, "draw_peak_w": 0.0,
                                                    "draw_sum": 0.0, "n": 0})
                entry["draw_peak_w"] = max(entry["draw_peak_w"], draw)
                entry["draw_sum"] += draw
                entry["n"] += 1
    for entry in power.values():
        entry["draw_mean_w"] = round(entry.pop("draw_sum") / max(entry.pop("n"), 1), 1)
        entry["draw_peak_w"] = round(entry["draw_peak_w"], 1)
    return peaks, power


def token_ids(prefix):
    path = Path(str(prefix) + ".tokens.txt")
    if not path.exists():
        return []
    return [int(x) for x in path.read_text().split()]


def repetition(ids, min_period=16):
    """Longest exactly-repeating suffix cycle, plus the distinct-token ratio.

    Greedy + --ignore-eos re-answers the same prompt over and over, so the stream
    is a repeated block. Find the smallest period p for which the tail is p-periodic over at least
    two full blocks, which is what distinguishes a looping stream from a novel one.
    """
    n = len(ids)
    stats = {
        "tokens": n,
        "distinct": len(set(ids)),
        "distinct_ratio": round(len(set(ids)) / n, 4) if n else None,
        "period": None,
        "repeats": None,
    }
    for period in range(min_period, n // 2 + 1):
        if all(ids[i] == ids[i - period] for i in range(n - 2 * period, n)):
            stats["period"] = period
            repeats = 0
            index = n - period
            while index - period >= 0 and ids[index - period:index] == ids[index:index + period]:
                repeats += 1
                index -= period
            stats["repeats"] = repeats + 1
            break
    return stats


report = {"max_context": max_context, "steps": {}}
for step in STEPS:
    prefix = resolve(step)
    if prefix is None:
        continue
    log = Path(str(prefix) + ".cli.log")
    rows = summary_rows(log.read_text(errors="replace")) if log.exists() else {}
    peaks, power = vram_peaks(Path(str(prefix) + ".vram.csv"))
    ids = token_ids(prefix)
    entry = {
        "log": str(prefix),
        "prompt_tokens": rows.get("prompt tokens"),
        "generated_tokens": rows.get("generated tokens"),
        "finish_reason": rows.get("finish reason"),
        "prefill_speed": rows.get("prefill speed"),
        "decode_speed": rows.get("decode speed"),
        "gpu0_reserved": rows.get("gpu0 reserved"),
        "gpu1_reserved": rows.get("gpu1 reserved"),
        "gpu0_breakdown": {k[5:]: v for k, v in rows.items() if k.startswith("gpu0 ")},
        "workspace_peak": rows.get("gpu workspace peak"),
        "mtp": {k[4:]: v for k, v in rows.items() if k.startswith("mtp ")},
        "vram_peak_mib": {k: v for k, v in sorted(peaks.items())},
        "power": power,
        "generated_ids": len(ids),
    }
    if step.startswith("decode-"):
        entry["stream"] = repetition(ids) if ids else None
    report["steps"][step] = entry

# The identity claim: MTP3 vs MTP-off on the same bundle-A prompt, token id for token id.
a_on, a_off = resolve("needle-a-mtp3"), resolve("needle-a-off")
if a_on and a_off:
    ids_on, ids_off = token_ids(a_on), token_ids(a_off)
    first = next((i for i, (x, y) in enumerate(zip(ids_on, ids_off)) if x != y), None)
    report["needle_identity"] = {
        "bundle": "A",
        "mtp3_tokens": len(ids_on),
        "mtp_off_tokens": len(ids_off),
        "identical": ids_on == ids_off and bool(ids_on),
        "first_divergence_index": first,
    }

# The text claim: both bundles against the needle campaign's own recorded answers.
answers = {}
for bundle, step, prompt in (("A", "needle-a-mtp3", "needle-1046k-depth50-bundleA.messages.json"),
                             ("B", "needle-b-mtp3", "needle-1046k-depth50-bundleB.messages.json")):
    prefix = resolve(step)
    if prefix is None:
        continue
    reply = Path(str(prefix) + ".reply.txt")
    expected_path = work_dir / (prompt + ".mtp-off-answer.txt")
    got = reply.read_text().strip() if reply.exists() else None
    expected = expected_path.read_text().strip() if expected_path.exists() else None
    answers[bundle] = {
        "mtp3_answer": got,
        "needle_answer_mtp_off": expected,
        "match": (got == expected) if (got is not None and expected is not None) else None,
    }
report["needle_answers"] = answers

results_dir.mkdir(parents=True, exist_ok=True)

lines = [
    "# MTP3 at 1M under TP2 + YaRN ×4",
    "",
    f"Window: `--max-context {max_context}`. MTP3 = `--spec mtp --draft-tokens 3 --lm-head-draft`.",
    "Both RTX 5090s power-capped at 400 W for every measurement below.",
    "",
    "| step | prompt tok | gen tok | prefill | decode | gpu0 reserved | gpu1 reserved | nvidia-smi peak/GPU |",
    "|---|---|---|---|---|---|---|---|",
]
for step, entry in report["steps"].items():
    proc = [v for k, v in entry["vram_peak_mib"].items() if k.startswith("process ")]
    peak = f"{max(proc)} MiB" if proc else "n/a"
    lines.append(
        "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            step, entry["prompt_tokens"], entry["generated_tokens"], entry["prefill_speed"],
            entry["decode_speed"], entry["gpu0_reserved"], entry["gpu1_reserved"], peak)
    )

# What MTP3 costs at this window, taken from the two runs that differ only in the MTP flags.
def gib(text):
    if not text:
        return None
    value, unit = text.split()[0], text.split()[1]
    return float(value) * (1.0 if unit == "GiB" else 1.0 / 1024.0)


on_entry, off_entry = report["steps"].get("needle-a-mtp3"), report["steps"].get("needle-a-off")
if on_entry and off_entry:
    lines += ["", "## What MTP3 costs at this window", "",
              "Both rows are the same 1,045,954-token prompt; the runs differ only in "
              "`--spec mtp --draft-tokens 3 --lm-head-draft`.", "",
              "| device-0 arena | MTP off | MTP3 | delta |", "|---|---|---|---|"]
    deltas = {}
    for field in ("weights", "kv pool", "gdn state", "sequence", "workspace", "reserved"):
        off_v, on_v = off_entry["gpu0_breakdown"].get(field), on_entry["gpu0_breakdown"].get(field)
        delta = gib(on_v) - gib(off_v) if (off_v and on_v) else None
        deltas[field] = delta
        lines.append("| {} | {} | {} | {} |".format(
            field, off_v, on_v, "{:+.2f} GiB".format(delta) if delta is not None else "—"))
    off_peak = max((v for k, v in off_entry["vram_peak_mib"].items() if k.startswith("process ")),
                   default=None)
    on_peak = max((v for k, v in on_entry["vram_peak_mib"].items() if k.startswith("process ")),
                  default=None)
    if off_peak and on_peak:
        lines.append("| nvidia-smi per-process peak | {} MiB | {} MiB | {:+d} MiB ({:+.2f} GiB) |".format(
            off_peak, on_peak, on_peak - off_peak, (on_peak - off_peak) / 1024.0))
    report["mtp_cost_gib"] = {k: (round(v, 3) if v is not None else None) for k, v in deltas.items()}
    report["mtp_cost_gib"]["nvidia_smi_process_peak"] = (
        round((on_peak - off_peak) / 1024.0, 3) if (off_peak and on_peak) else None)
    lines += ["",
              "The same configuration measured +0.65 GiB/device at `--max-context 262144`. The "
              "fixed part is the MTP head's weights (+0.38 GiB); the rest is its own KV pages, "
              "which scale with the "
              "window: 1.03 GiB at 1,048,576 tokens is 0.26 GiB at 262,144, and 0.38 + 0.26 = 0.64. "
              "So the MTP surcharge is **0.38 GiB + 1.03 GiB per 1M tokens of context**, not a "
              "constant."]

lines += ["", "## Speculative statistics", "",
          "| step | rounds | drafted | accepted | acceptance | tok/round | fallback |",
          "|---|---|---|---|---|---|---|"]
for step, entry in report["steps"].items():
    mtp = entry["mtp"]
    if not mtp:
        continue
    lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
        step, mtp.get("rounds"), mtp.get("drafted tokens"), mtp.get("accepted tokens"),
        mtp.get("acceptance rate"), mtp.get("acceptance length"), mtp.get("fallback steps")))

if "needle_identity" in report:
    identity = report["needle_identity"]
    lines += ["", "## Needle: MTP3 vs MTP-off", "",
              "- bundle A token ids identical: **{}** ({} vs {} tokens, first divergence {})".format(
                  identity["identical"], identity["mtp3_tokens"], identity["mtp_off_tokens"],
                  identity["first_divergence_index"])]
for bundle, entry in answers.items():
    lines.append("- bundle {} answer matches the recorded MTP-off answer: **{}**".format(
        bundle, entry["match"]))

lines += ["", "## Decode streams", ""]
for step in ("decode-greedy", "decode-sampled", "decode-novel"):
    entry = report["steps"].get(step)
    if not entry or not entry.get("stream"):
        continue
    stream = entry["stream"]
    lines.append(
        "- `{}`: {} tokens, {} distinct (ratio {}), repeating period {}, repeats {}".format(
            step, stream["tokens"], stream["distinct"], stream["distinct_ratio"],
            stream["period"], stream["repeats"]))

# The JSON is written LAST, after every section above has had its chance to add to `report`.
# Written any earlier it silently loses whatever the later sections compute -- `mtp_cost_gib` was
# missing from the artifact for exactly that reason.
(results_dir / "mtp3-1m-summary.json").write_text(json.dumps(report, indent=2) + "\n")
(results_dir / "mtp3-1m-summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print()
print(f"wrote {results_dir}/mtp3-1m-summary.json")
print(f"wrote {results_dir}/mtp3-1m-summary.md")
PY
}

case "${step}" in
    boot)
        run_cli boot "${mtp_flags[@]}" --greedy --no-thinking --prompt "Say hello." --max-new 16
        ;;
    prompts)
        build_prompts
        ;;
    needle-a-mtp3)
        require_prompt "${needle_a_prompt}"
        run_cli needle-a-mtp3 "${mtp_flags[@]}" --greedy --no-thinking --print-token-ids \
            --messages "${needle_a_prompt}" --max-new "${needle_tokens}"
        ;;
    needle-a-off)
        require_prompt "${needle_a_prompt}"
        run_cli needle-a-off --greedy --no-thinking --print-token-ids \
            --messages "${needle_a_prompt}" --max-new "${needle_tokens}"
        ;;
    needle-b-mtp3)
        require_prompt "${needle_b_prompt}"
        run_cli needle-b-mtp3 "${mtp_flags[@]}" --greedy --no-thinking --print-token-ids \
            --messages "${needle_b_prompt}" --max-new "${needle_tokens}"
        ;;
    decode-greedy)
        require_prompt "${soak_prompt}"
        run_cli decode-greedy "${mtp_flags[@]}" --greedy --no-thinking --ignore-eos \
            --print-token-ids --messages "${soak_prompt}" --max-new "${decode_tokens}"
        ;;
    decode-sampled)
        require_prompt "${soak_prompt}"
        run_cli decode-sampled "${mtp_flags[@]}" --no-thinking --ignore-eos --print-token-ids \
            --temperature "${NINFER_MTP3_TEMPERATURE:-0.8}" --top-p "${NINFER_MTP3_TOP_P:-0.95}" \
            --seed "${NINFER_MTP3_SEED:-42}" \
            --messages "${soak_prompt}" --max-new "${decode_tokens}"
        ;;
    decode-novel)
        require_prompt "${soak_prompt}"
        run_cli decode-novel "${mtp_flags[@]}" --greedy --no-thinking --ignore-eos \
            --print-token-ids --messages "${soak_prompt}" --max-new "${novel_tokens}"
        ;;
    analyze)
        analyze
        ;;
esac
