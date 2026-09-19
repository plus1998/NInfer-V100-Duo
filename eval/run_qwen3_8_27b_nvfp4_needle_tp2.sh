#!/usr/bin/env bash
# Needle-in-a-Haystack for the Qwen3.8-27B NVFP4 artifact on two RTX 5090s (TP2).
#
# Mirrors run_qwen3_8_27b_nvfp4_reasoning.sh: this script owns one local ninfer-serve
# process per step, waits for /health, runs one ninfer_eval suite against it, and stops the
# server again. It never touches a server it did not start.
#
# TP2 facts this encodes:
#   * `--tp 2` requires `--devices 0,1`, and the engine rejects it together with `--spec`
#     (speculative decoding across devices is still guarded) and `--vision`. The needle
#     suite therefore runs with MTP OFF.
#   * The full 262,144-token context fits at TP2 because the ~20 GiB of weights are split
#     across the two cards; a single card only reaches 252,928 (see the reasoning script).
#
# Every step also samples GPU memory for as long as the suite runs, so the per-GPU VRAM gate
# is re-runnable rather than a number someone once watched go by. The sampler writes one CSV
# row per GPU and per compute process per tick (default every 3 s, NINFER_VRAM_SAMPLE_SECONDS
# overrides) and the step prints the peaks it observed.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
server_bin="${repo_dir}/build/apps/ninfer-serve"
artifact="${NINFER_QWEN3_8_27B_NVFP4_ARTIFACT:-/home/pc/models/ninfer-38/huihui-nvfp4/qwen3_8_27b_nvfp4.ninfer}"
config="${repo_dir}/eval/configs/qwen3_8_27b_nvfp4_needle_haystack.yaml"
eval_python="${repo_dir}/eval/.venv/bin/python"
log_dir="${repo_dir}/eval/server-logs"

usage() {
    echo "usage: $0 [--plan] [smoke|64k|128k|262k|ladder|control|control-absent|control-absent-optout]" >&2
    echo "       with no step argument, the 64k -> 128k -> 262k ladder runs" >&2
}

# Emits tab-separated fields: suite, description.
step_config() {
    case "$1" in
        smoke)  printf 'smoke\t%s\n' "one 8k needle (contract check, not a gate)" ;;
        64k)    printf 'native_64k\t%s\n' "64k haystack, depths 10/30/50/70/90, en+zh" ;;
        128k)   printf 'native_128k\t%s\n' "128k haystack, depths 10/30/50/70/90, en+zh" ;;
        262k)   printf 'native_262k\t%s\n' "260k haystack in the 262,144-token window, depths 10/30/50/70/90, en+zh" ;;
        ladder) printf 'native_long\t%s\n' "64k, 128k, and 262k tiers back to back" ;;
        control)
            printf 'control_262k_novel_needle\t%s\n' \
                "262k retrieval-vs-memorization control: one novel needle the model cannot have memorized" ;;
        control-absent)
            printf 'control_262k_absent_needle\t%s\n' \
                "262k negative control: the question asks about something absent from the prompt" ;;
        control-absent-optout)
            printf 'control_262k_absent_needle_optout\t%s\n' \
                "262k negative control, with an explicit NOT FOUND opt-out in the prompt" ;;
        *)      return 2 ;;
    esac
}

plan_only=0
if [[ "${1:-}" == "--plan" ]]; then
    plan_only=1
    shift
fi
if [[ $# -gt 1 ]]; then
    usage
    exit 2
fi
case "${1:-}" in
    "")                                steps=(ladder) ;;
    smoke | 64k | 128k | 262k | ladder | control | control-absent | control-absent-optout)
                                       steps=("$1") ;;
    *)                                 usage; exit 2 ;;
esac

if [[ "${plan_only}" -eq 1 ]]; then
    for step in "${steps[@]}"; do
        IFS=$'\t' read -r suite _ <<<"$(step_config "${step}")"
        PYTHONPATH="${repo_dir}/eval" "${eval_python}" -m ninfer_eval plan \
            --config "${config}" --suite "${suite}" --check-runtime
    done
    exit 0
fi

for required_file in "${server_bin}" "${artifact}" "${config}" "${eval_python}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "missing required file: ${required_file}" >&2
        exit 1
    fi
done
if [[ ! -x "${server_bin}" || ! -x "${eval_python}" ]]; then
    echo "ninfer-serve and the evaluation Python must be executable" >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required for the server health check" >&2
    exit 1
fi

mkdir -p -- "${log_dir}"
server_pid=""
sampler_pid=""
vram_sample_seconds="${NINFER_VRAM_SAMPLE_SECONDS:-3}"

# One CSV row per GPU and per (compute process, GPU) per tick:
#   timestamp_utc,kind,key,used_mib
# `kind=gpu` keys on the GPU index and includes memory other processes hold. `kind=process` keys
# on `<pid>@gpu<index>` and is this server's own residency, which is what the per-GPU gate is
# about. The index matters: nvidia-smi's compute-app query reports a GPU uuid and emits one row
# per GPU a process touches, so a TP2 server appears twice per tick. Keying on the pid alone would
# average or overwrite the two cards and silently hide an asymmetric split -- this run happened to
# be symmetric, but a future one need not be. Resolve uuid -> index once, up front.
sample_vram() {
    local out="$1" interval="$2" uuid_map
    printf 'timestamp_utc,kind,key,used_mib\n' >"${out}"
    uuid_map="$(nvidia-smi --query-gpu=uuid,index --format=csv,noheader,nounits 2>/dev/null \
        | tr -d '[:blank:]')"
    while true; do
        local now
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

report_vram_peaks() {
    local out="$1"
    if [[ ! -s "${out}" ]]; then
        return
    fi
    echo "vram samples: ${out}"
    awk -F, 'NR > 1 && $4 != "" && $4 + 0 == $4 {
        key = $2 " " $3
        if ($4 + 0 > peak[key]) { peak[key] = $4 + 0 }
        ticks[key]++
    }
    END {
        for (key in peak) {
            printf "  peak %-24s %8d MiB  (%.2f GiB, %d samples)\n", key, peak[key], peak[key] / 1024, ticks[key]
        }
    }' "${out}" | sort
}

cleanup() {
    if [[ -n "${sampler_pid}" ]] && kill -0 "${sampler_pid}" 2>/dev/null; then
        kill -TERM "${sampler_pid}" 2>/dev/null || true
        wait "${sampler_pid}" 2>/dev/null || true
    fi
    sampler_pid=""
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill -TERM "${server_pid}"
        wait "${server_pid}" || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_step() {
    local step="$1"
    local suite description run_stamp server_log request_log vram_log

    IFS=$'\t' read -r suite description <<<"$(step_config "${step}")"

    run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    server_log="${log_dir}/qwen3_8_27b_nvfp4-needle-tp2-${step}-${run_stamp}.server.log"
    request_log="${log_dir}/qwen3_8_27b_nvfp4-needle-tp2-${step}-${run_stamp}.requests.jsonl"
    vram_log="${log_dir}/qwen3_8_27b_nvfp4-needle-tp2-${step}-${run_stamp}.vram.csv"

    if curl --fail --silent --show-error --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1; then
        echo "port 18080 already has a healthy service; stop it before running this script" >&2
        exit 1
    fi

    "${server_bin}" "${artifact}" \
        --host 127.0.0.1 \
        --port 18080 \
        --model-id qwen3.8-27b \
        --tp 2 \
        --devices 0,1 \
        --max-context 262144 \
        --kv-capacity auto \
        --kv-dtype int8 \
        --max-concurrency 1 \
        --max-pending-requests 4 \
        --pending-timeout-ms 86400000 \
        --prefill-chunk 1024 \
        --log-stats-interval-ms 5000 \
        --request-log-jsonl "${request_log}" \
        >"${server_log}" 2>&1 &
    server_pid=$!

    ready=0
    for ((attempt = 1; attempt <= 180; ++attempt)); do
        if curl --fail --silent --show-error --max-time 2 \
            http://127.0.0.1:18080/health >/dev/null 2>&1; then
            ready=1
            break
        fi
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            wait "${server_pid}" || true
            echo "ninfer-serve exited before becoming ready; see ${server_log}" >&2
            exit 1
        fi
        sleep 1
    done
    if [[ "${ready}" -ne 1 ]]; then
        echo "ninfer-serve did not become ready within 180 seconds; see ${server_log}" >&2
        exit 1
    fi

    sample_vram "${vram_log}" "${vram_sample_seconds}" &
    sampler_pid=$!

    echo "server log: ${server_log}"
    echo "request log: ${request_log}"
    echo "vram log: ${vram_log}"
    echo "running ${description}"

    PYTHONPATH="${repo_dir}/eval" "${eval_python}" -m ninfer_eval run \
        --config "${config}" --suite "${suite}"

    cleanup
    server_pid=""
    report_vram_peaks "${vram_log}"
}

for step in "${steps[@]}"; do
    run_step "${step}"
done
