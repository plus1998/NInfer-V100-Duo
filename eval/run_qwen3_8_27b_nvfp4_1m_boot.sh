#!/usr/bin/env bash
# Boot gate: 1,048,576-token context boot + short generation under TP2 (YaRN x4).
#
# Unlike the needle scripts next door, this one drives the CLI (`build/apps/ninfer`) rather than
# `ninfer-serve`: the gate is about one process booting a 1M-token KV pool on two devices and
# emitting a coherent reply, and the CLI prints the per-device load summary that the gate asserts.
#
# Steps
#   boot       the gate command itself (--max-new 16, thinking on). At 1M the reply budget
#              is spent inside the reasoning channel, so this step gates BOOT + VRAM, not prose.
#   coherence  the same engine with --no-thinking, so the 16-token budget lands in the answer
#              channel and the reply text itself can be read.
#   long       a synthetic >=900k-token haystack with a five-digit needle at ~50% depth, prefilled
#              to completion plus 16 greedy tokens. Records prefill/decode tok/s and the VRAM peak
#              during prefill (where the workspace peaks), and reports whether the needle came back.
#              Retrieval QUALITY is the needle suite's; this step only asks whether 1M prefill runs.
#
# Every step samples nvidia-smi for as long as the run lasts and prints the observed peaks per GPU
# and per (process, GPU), because the gate is a memory gate and a number nobody can re-measure is
# not evidence. `kind=gpu` rows include memory other processes hold (GPU 0 carries ~1 GiB of
# desktop/display allocation on this machine); `kind=process` rows are this run's own residency.
#
# The clean-exit gate is an assertion, not a printout. Each step writes `nvidia-smi`'s compute-app
# list both before launch and after exit into `<stamp>.nvidia-smi.txt` alongside the other logs, and
# computes a PASS/FAIL from it: a step REFUSES to launch while any compute process is resident, and
# fails after the run if any is left behind. The script never signals a GPU process it did not start
# -- when it finds one, it stops and says so.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cli_bin="${repo_dir}/build/apps/ninfer"
artifact="${NINFER_QWEN3_8_27B_NVFP4_ARTIFACT:-/home/pc/models/ninfer-38/huihui-nvfp4/qwen3_8_27b_nvfp4.ninfer}"
log_dir="${repo_dir}/eval/server-logs"

# The gate's memory bound, per device, in MiB.
gate_mib=$((30 * 1024))

# Long-probe size. The fallback target is 2x native (524,288); the target used here is >=900k.
long_target_tokens="${NINFER_1M_LONG_TOKENS:-950000}"
long_secret="${NINFER_1M_LONG_SECRET:-74812}"

# Tokenizer used only to SIZE the synthetic haystack. The authoritative token count is the one
# ninfer reports back ("summary prompt tokens"); this just gets us into the window in one shot.
tokenizer_json="${NINFER_1M_TOKENIZER_JSON:-/home/pc/models/ninfer-38/unsloth-nvfp4/tokenizer.json}"
tokenizer_python="${NINFER_1M_TOKENIZER_PYTHON:-${repo_dir}/eval/.venv/bin/python}"

vram_sample_seconds="${NINFER_VRAM_SAMPLE_SECONDS:-2}"

# Flags shared by every step: the 1M gate configuration verbatim.
engine_flags=(
    --tp 2
    --devices "${NINFER_1M_DEVICES:-0,1}"
    --rope yarn
    --yarn-factor 4.0
    --yarn-origin 262144
    --max-context 1048576
    --kv-dtype int8
    --kv-capacity "${NINFER_1M_KV_CAPACITY:-auto}"
    --greedy
)
if [[ -n "${NINFER_1M_PREFILL_CHUNK:-}" ]]; then
    engine_flags+=(--prefill-chunk "${NINFER_1M_PREFILL_CHUNK}")
fi
if [[ -n "${NINFER_1M_NO_CUDA_GRAPH:-}" ]]; then
    engine_flags+=(--no-cuda-graph)
fi

usage() {
    echo "usage: $0 [boot|coherence|long|all]" >&2
    echo "       default: all" >&2
}

case "${1:-all}" in
    boot)      steps=(boot) ;;
    coherence) steps=(coherence) ;;
    long)      steps=(long) ;;
    all)       steps=(boot coherence long) ;;
    *)         usage; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then usage; exit 2; fi

for required in "${cli_bin}" "${artifact}"; do
    if [[ ! -f "${required}" ]]; then
        echo "missing required file: ${required}" >&2
        exit 1
    fi
done
if [[ ! -x "${cli_bin}" ]]; then
    echo "${cli_bin} must be executable" >&2
    exit 1
fi
mkdir -p -- "${log_dir}"

sampler_pid=""
work_dir=""
cleanup() {
    if [[ -n "${sampler_pid}" ]] && kill -0 "${sampler_pid}" 2>/dev/null; then
        kill -TERM "${sampler_pid}" 2>/dev/null || true
        wait "${sampler_pid}" 2>/dev/null || true
    fi
    sampler_pid=""
    if [[ -n "${work_dir}" && -d "${work_dir}" ]]; then rm -rf -- "${work_dir}"; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# One CSV row per GPU and per (compute process, GPU) per tick: timestamp_utc,kind,key,used_mib.
# nvidia-smi's compute-app query reports a GPU uuid and emits one row per GPU a process touches, so
# a TP2 run appears twice per tick; keying on the pid alone would hide an asymmetric split.
sample_vram() {
    local out="$1" interval="$2" uuid_map now
    printf 'timestamp_utc,kind,key,used_mib\n' >"${out}"
    uuid_map="$(nvidia-smi --query-gpu=uuid,index --format=csv,noheader,nounits 2>/dev/null \
        | tr -d '[:blank:]')"
    while true; do
        now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | awk -v ts="${now}" -F', *' '{print ts ",gpu," $1 "," $2}' >>"${out}" 2>/dev/null || true
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

# Prints the peaks and returns nonzero when any per-process peak exceeds the gate.
report_vram_peaks() {
    local out="$1"
    if [[ ! -s "${out}" ]]; then
        echo "  (no vram samples collected)"
        return 0
    fi
    awk -F, -v gate="${gate_mib}" 'NR > 1 && $4 != "" && $4 + 0 == $4 {
        key = $2 " " $3
        if ($4 + 0 > peak[key]) { peak[key] = $4 + 0 }
        ticks[key]++
    }
    END {
        over = 0
        for (key in peak) {
            printf "  peak %-24s %8d MiB  (%6.2f GiB, %3d samples)\n", key, peak[key], peak[key] / 1024, ticks[key]
            if (key ~ /^process /) { if (peak[key] > gate) { over = 1 } }
        }
        exit over
    }' "${out}" | sort
    return "${PIPESTATUS[0]}"
}

# The per-device rows the gate asserts, straight out of the load summary, plus a pass/fail line.
report_load_summary() {
    local err_log="$1"
    grep -E '^summary +(rope|tensor parallel|gpu[01] )' "${err_log}" || true
    awk -v gate="${gate_mib}" '
        $1 == "summary" && $2 ~ /^gpu[01]$/ && $3 == "reserved" {
            value = $4 + 0
            mib = ($5 == "GiB") ? value * 1024 : (($5 == "MiB") ? value : value)
            printf "  gate  %s reserved %8.2f GiB  %s (bound 30.00 GiB)\n", $2, mib / 1024,
                   (mib <= gate) ? "PASS" : "FAIL"
            if (mib > gate) { over = 1 }
        }
        END { exit over }
    ' "${err_log}"
}

summary_value() { grep -E "^summary +$1" "$2" | head -1 | sed -E "s/^summary +$1 +//"; }

# The pids nvidia-smi currently reports as holding GPU memory, one per line, empty when none do.
# `--format=csv,noheader` prints nothing at all when the list is empty, so "no rows" IS the clean
# state; the grep guards against a driver that emits a placeholder instead of staying silent.
compute_app_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | tr -d '[:blank:]' | grep -E '^[0-9]+$' || true
}

record_compute_apps() {
    local label="$1" out="$2" rows
    rows="$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv 2>&1)"
    {
        printf '=== compute processes %s (%s) ===\n' "${label}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '%s\n' "${rows}"
    } >>"${out}"
    printf 'compute processes %s:\n' "${label}"
    printf '%s\n' "${rows}" | sed 's/^/  /'
}

# Builds a digit-free haystack with one five-digit needle at ~50% depth, sized with the real
# tokenizer, and writes it as a CLI --messages document.
build_long_prompt() {
    local out_json="$1" target="$2" secret="$3"
    if [[ ! -x "${tokenizer_python}" || ! -f "${tokenizer_json}" ]]; then
        echo "long probe needs a tokenizer to size the haystack:" >&2
        echo "  python:    ${tokenizer_python}" >&2
        echo "  tokenizer: ${tokenizer_json}" >&2
        echo "override with NINFER_1M_TOKENIZER_PYTHON / NINFER_1M_TOKENIZER_JSON" >&2
        return 1
    fi
    "${tokenizer_python}" - "${out_json}" "${target}" "${secret}" "${tokenizer_json}" <<'PY'
import json
import sys

from tokenizers import Tokenizer

out_json, target, secret, tokenizer_json = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
tok = Tokenizer.from_file(tokenizer_json)

# Digit-free filler, so the needle is the only run of digits anywhere in the prompt.
FILLER = [
    "The archive room stays quiet through the afternoon and the shelves hold folders of ordinary correspondence.",
    "A librarian walks the aisle, checking that each box sits square against the rail before moving on.",
    "Outside the window the courtyard trees move a little, and someone wheels a cart across the gravel.",
    "The catalogue is kept by hand, in a script that has grown steadily neater over the years.",
    "Every folder carries a short summary on its cover, written so a stranger could follow it.",
    "Nothing in this wing is urgent; the urgent material lives one floor down, behind a different door.",
    "The heating pipes tick as they warm, and the smell of paper and dust settles over the tables.",
    "A visitor signs the register, takes a pencil from the tray, and finds a seat near the far wall.",
    "The reading lamps are old but bright, and their shades throw neat circles onto the worn wood.",
    "By evening the light shifts and the room takes on the colour of the brick across the street.",
]
NEEDLE = (
    "IMPORTANT ARCHIVE NOTE: The secret access code for the Meridian vault is {}. "
    "Remember this code; it appears exactly once in this document.".format(secret)
)
QUESTION = (
    "\n\nThe document above is an archive log. What is the secret access code for the Meridian "
    "vault? Answer with the digits only."
)
PREAMBLE = "Read the following archive log carefully, then answer the question at the end.\n\n"

# The chat template wraps the user text; leave room for it rather than overshoot the 1,048,576 ceiling.
budget = target - 128 - len(tok.encode(PREAMBLE + NEEDLE + QUESTION, add_special_tokens=False).ids)
per_line = [len(tok.encode(line + "\n", add_special_tokens=False).ids) for line in FILLER]
cycle = sum(per_line)

cycles = max(1, budget // cycle)
lines = []
count = 0
for _ in range(cycles):
    for line, n in zip(FILLER, per_line):
        lines.append(line)
        count += n
i = 0
while count + per_line[i % len(FILLER)] <= budget:
    lines.append(FILLER[i % len(FILLER)])
    count += per_line[i % len(FILLER)]
    i += 1

middle = len(lines) // 2
lines.insert(middle, NEEDLE)
text = PREAMBLE + "\n".join(lines) + QUESTION

measured = len(tok.encode(text, add_special_tokens=False).ids)
depth_tokens = len(tok.encode(PREAMBLE + "\n".join(lines[:middle]), add_special_tokens=False).ids)
with open(out_json, "w", encoding="utf-8") as handle:
    json.dump([{"role": "user", "content": text}], handle)

print(
    "haystack: {} raw tokens, {} chars, needle at raw token ~{} (depth {:.1f}%)".format(
        measured, len(text), depth_tokens, 100.0 * depth_tokens / measured
    )
)
PY
}

run_step() {
    local step="$1" run_stamp err_log out_log vram_log smi_log rc=0
    local -a extra=()

    run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    err_log="${log_dir}/qwen3_8_27b_nvfp4-1m-${step}-${run_stamp}.cli.log"
    out_log="${log_dir}/qwen3_8_27b_nvfp4-1m-${step}-${run_stamp}.reply.txt"
    vram_log="${log_dir}/qwen3_8_27b_nvfp4-1m-${step}-${run_stamp}.vram.csv"
    smi_log="${log_dir}/qwen3_8_27b_nvfp4-1m-${step}-${run_stamp}.nvidia-smi.txt"

    case "${step}" in
        boot)      extra=(--prompt "Say hello." --max-new 16) ;;
        coherence) extra=(--prompt "Say hello." --max-new 16 --no-thinking) ;;
        long)
            work_dir="$(mktemp -d)"
            echo "building the long prompt (target ${long_target_tokens} tokens)"
            build_long_prompt "${work_dir}/messages.json" "${long_target_tokens}" "${long_secret}"
            extra=(--messages "${work_dir}/messages.json" --max-new 16 --no-thinking)
            ;;
    esac

    echo
    echo "=== step: ${step} ==="
    echo "cli log:   ${err_log}"
    echo "reply:     ${out_log}"
    echo "vram log:  ${vram_log}"
    echo "smi log:   ${smi_log}"

    # The clean-exit gate item is an assertion, not a printout: both halves are persisted next to
    # the other logs so the verdict can be re-derived from the artifacts after the fact.
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

    set +e
    "${cli_bin}" "${artifact}" "${engine_flags[@]}" "${extra[@]}" >"${out_log}" 2>"${err_log}"
    rc=$?
    set -e

    if [[ -n "${sampler_pid}" ]] && kill -0 "${sampler_pid}" 2>/dev/null; then
        kill -TERM "${sampler_pid}" 2>/dev/null || true
        wait "${sampler_pid}" 2>/dev/null || true
    fi
    sampler_pid=""
    if [[ -n "${work_dir}" && -d "${work_dir}" ]]; then rm -rf -- "${work_dir}"; work_dir=""; fi

    echo "cli exit: ${rc}"
    if [[ "${rc}" -ne 0 ]]; then
        echo "--- last 40 log lines ---"
        tail -40 "${err_log}"
        return "${rc}"
    fi

    echo "--- load summary (per device) ---"
    local summary_ok=0
    report_load_summary "${err_log}" || summary_ok=1

    echo "--- nvidia-smi peaks during the run ---"
    local sampler_ok=0
    report_vram_peaks "${vram_log}" || sampler_ok=1
    if [[ "${sampler_ok}" -eq 0 ]]; then
        echo "  gate  every per-process peak <= 30.00 GiB  PASS"
    else
        echo "  gate  a per-process peak exceeded 30.00 GiB  FAIL"
    fi

    echo "--- generation ---"
    printf '  prompt tokens     %s\n' "$(summary_value 'prompt tokens' "${err_log}")"
    printf '  generated tokens  %s\n' "$(summary_value 'generated tokens' "${err_log}")"
    printf '  finish reason     %s\n' "$(summary_value 'finish reason' "${err_log}")"
    printf '  prefill speed     %s\n' "$(summary_value 'prefill speed' "${err_log}")"
    printf '  decode speed      %s\n' "$(summary_value 'decode speed' "${err_log}")"
    grep -E '^generate +text prefill' "${err_log}" | sed 's/^/  /' || true
    grep -E '^generate +decode' "${err_log}" | sed 's/^/  /' || true
    printf '  workspace peak    %s\n' "$(summary_value 'gpu workspace peak' "${err_log}")"

    echo "--- reply (stdout) ---"
    sed 's/^/  | /' "${out_log}"
    if [[ -z "$(tr -d '[:space:]' <"${out_log}")" ]]; then
        echo "  (the answer channel is empty: the whole --max-new budget was spent inside the"
        echo "   reasoning channel, which ninfer streams to stderr. Reasoning text follows.)"
        # Everything the CLI writes to stderr that is not one of its own table rows is reasoning.
        grep -vE '^(phase|load|summary|prepare|generate) ' "${err_log}" | sed 's/^/  ~ /'
    fi

    if [[ "${step}" == "long" ]]; then
        if grep -q "${long_secret}" "${out_log}"; then
            echo "  needle: FOUND (${long_secret})"
        else
            echo "  needle: not in the reply (retrieval quality is the needle suite's, not this)"
        fi
    fi

    echo "--- clean exit ---"
    record_compute_apps "after exit" "${smi_log}"
    local leftover clean_ok=0
    leftover="$(compute_app_pids)"
    if [[ -z "${leftover}" ]]; then
        echo "  gate  no compute process after exit  PASS" | tee -a "${smi_log}"
    else
        clean_ok=1
        echo "  gate  compute processes still resident after exit: $(echo "${leftover}" | tr '\n' ' ') FAIL" \
            | tee -a "${smi_log}"
    fi

    if [[ "${summary_ok}" -ne 0 || "${sampler_ok}" -ne 0 ]]; then
        echo "step ${step}: MEMORY GATE FAILED"
        return 1
    fi
    if [[ "${clean_ok}" -ne 0 ]]; then
        echo "step ${step}: CLEAN-EXIT GATE FAILED"
        return 1
    fi
    echo "step ${step}: OK"
}

for step in "${steps[@]}"; do
    run_step "${step}"
done
