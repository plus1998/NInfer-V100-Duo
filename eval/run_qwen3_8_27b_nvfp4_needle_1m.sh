#!/usr/bin/env bash
# 1M needle-in-a-haystack for Qwen3.8-27B NVFP4 under TP2 + YaRN x4, plus the
# vLLM control at 655k.
#
# Two engines, one at a time. Each takes BOTH RTX 5090s, so this script never runs them
# concurrently and refuses to start either one while any compute process holds GPU memory.
# It never signals a process it did not start.
#
#   ninfer-*   starts ./build/apps/ninfer-serve with the 1M boot-gate command line
#              (--tp 2 --devices 0,1 --rope yarn --yarn-factor 4.0 --yarn-origin 262144
#               --max-context 1048576 --kv-dtype int8 --kv-capacity auto
#               --max-concurrency 1) and runs one or more suites against it.
#              1M is a one-slot configuration by arithmetic: the per-slot sequence cost is
#              16.66 GiB/device, so a second slot cannot fit on a 32 GiB card.
#   vllm-*     starts the user's own serve script,
#              /home/pc/Projects/vllm/serve-orca-qwen38-27b-long.sh
#              (orcarouter/Qwen3.8-27B-Uncensored-FP8, TP2, --max-model-len 655360, the same
#               YaRN rope_parameters), runs the control suite against it, then stops it by
#              signalling the process group this script created.
#
# Both the 655k tier and the 1M tier are served by the SAME 1,048,576-token NInfer server.
# The rope table is a function of --yarn-factor and --yarn-origin only, not of --max-context,
# so the 655k row is rope-identical to the 1M row and to the vLLM control; serving both tiers
# from one process avoids a second 45 s model load.
#
# Environment overrides:
#   NINFER_QWEN3_8_27B_NVFP4_ARTIFACT   the .ninfer artifact
#   NINFER_NEEDLE_1M_CORPUS             the haystack corpus directory (the config's
#                                       `local_path: &corpus_path` anchor is rewritten into a
#                                       rendered copy at <prefix>.config.yaml)
#   NINFER_VLLM_LONG_SCRIPT             the vLLM control serve script
#   NINFER_1M_MAX_CONTEXT               the NInfer serving window (default 1048576)
#   NINFER_VRAM_SAMPLE_SECONDS          sampler interval (default 5)
#
# Power: every number below was measured with both RTX 5090s at a 400 W power limit (stock
# defaults are 600 W on GPU 0 and 575 W on GPU 1). The sampler records power.limit and power.draw
# per tick so a future artifact carries its own power condition rather than assuming stock.
#
# Timing (from the 1M boot gate's fit, t(n) = n/5.48e3 + n^2/1.16e9 at TP2):
#   653,000-token prefill  ~8 min   -> the 10-needle 655k tier is ~1.4 h
#   1,046,000-token prefill ~19 min -> the 10-needle 1M tier is ~3.2 h
# Run it under nohup and poll the log; do not sit on it.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
server_bin="${repo_dir}/build/apps/ninfer-serve"
artifact="${NINFER_QWEN3_8_27B_NVFP4_ARTIFACT:-/home/pc/models/ninfer-38/huihui-nvfp4/qwen3_8_27b_nvfp4.ninfer}"
config="${repo_dir}/eval/configs/qwen3_8_27b_nvfp4_needle_1m.yaml"
eval_python="${repo_dir}/eval/.venv/bin/python"
log_dir="${repo_dir}/eval/server-logs"
vllm_script="${NINFER_VLLM_LONG_SCRIPT:-/home/pc/Projects/vllm/serve-orca-qwen38-27b-long.sh}"
# The haystack corpus directory. It is a `local_path` inside the YAML rather than a command-line
# flag, and ninfer_eval does not expand environment variables in configs, so an override is
# applied by rendering a copy of the config with that one anchor line rewritten (render_config
# below). Unset -> the config is used verbatim, byte for byte.
corpus_override="${NINFER_NEEDLE_1M_CORPUS:-}"

ninfer_port=18080
vllm_port=8000
max_context="${NINFER_1M_MAX_CONTEXT:-1048576}"

usage() {
    cat >&2 <<'EOF'
usage: run_qwen3_8_27b_nvfp4_needle_1m.sh [--plan] <step>

steps (NInfer, one ninfer-serve process per step):
  smoke        one 8k needle on the new corpus (contract check, ~1 min incl. load)
  655k         653,000-token haystack, depths 10/30/50/70/90, both bundles  (~1.4 h)
  1m           1,046,000-token haystack, depths 10/30/50/70/90, both bundles (~3.2 h)
  ninfer-all   smoke -> 655k -> 1m, all on ONE server process (~4.6 h)
  control-1m   one 1M needle the model cannot have memorized (retrieval control, ~21 min)
  rerun        regression subset: 653k depth 50 then 1M depths 10/90, both bundles,
               on one server process (2 + 4 requests, ~1.8 h)

steps (vLLM control, starts and stops the user's serve script):
  vllm-smoke   one 8k needle through the vLLM control
  vllm-655k    the 653,000-token tier through the vLLM control (~30-60 min)
  vllm-all     vllm-smoke -> vllm-655k on one vLLM process
EOF
}

# step -> tab-separated: engine, space-separated suite list, description
step_config() {
    case "$1" in
        smoke)      printf 'ninfer\tsmoke_ninfer\t%s\n' "one 8k needle on the distinct-text corpus" ;;
        655k)       printf 'ninfer\tninfer_655k\t%s\n' "653,000-token haystack, depths 10/30/50/70/90, bundles A+B" ;;
        1m)         printf 'ninfer\tninfer_1m\t%s\n' "1,046,000-token haystack, depths 10/30/50/70/90, bundles A+B" ;;
        ninfer-all) printf 'ninfer\tsmoke_ninfer ninfer_655k ninfer_1m\t%s\n' "smoke, 655k and 1M tiers on one server" ;;
        control-1m) printf 'ninfer\tcontrol_1m_novel_needle\t%s\n' "1M retrieval-vs-memorization control: one invented needle at depth 50 (~21 min)" ;;
        rerun)      printf 'ninfer\tninfer_655k_rerun_d50 ninfer_1m_rerun_extremes\t%s\n' "re-run subset: 653k depth 50 + 1M depths 10/90, bundles A+B (~1.8 h)" ;;
        vllm-smoke) printf 'vllm\tsmoke_vllm\t%s\n' "one 8k needle through the vLLM control" ;;
        vllm-655k)  printf 'vllm\tvllm_655k\t%s\n' "653,000-token haystack through the vLLM control" ;;
        vllm-all)   printf 'vllm\tsmoke_vllm vllm_655k\t%s\n' "vLLM smoke and 655k control on one vLLM process" ;;
        *)          return 2 ;;
    esac
}

plan_only=0
if [[ "${1:-}" == "--plan" ]]; then
    plan_only=1
    shift
fi
if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi
step="$1"
if ! step_config "${step}" >/dev/null; then
    usage
    exit 2
fi
IFS=$'\t' read -r engine suites description <<<"$(step_config "${step}")"

# Echoes the config path to use. With NINFER_NEEDLE_1M_CORPUS set, writes a rendered copy to $2
# with the `local_path: &corpus_path ...` anchor rewritten, and fails loudly if that line is not
# found (a config edit that renamed the anchor must not silently fall back to the built-in path).
render_config() {
    local src="$1" dst="$2"
    if [[ -z "${corpus_override}" ]]; then
        printf '%s\n' "${src}"
        return 0
    fi
    if [[ ! -d "${corpus_override}" ]]; then
        echo "NINFER_NEEDLE_1M_CORPUS=${corpus_override} is not a directory" >&2
        return 1
    fi
    awk -v dir="${corpus_override}" '
        /^[[:space:]]*local_path:[[:space:]]*&corpus_path[[:space:]]/ {
            match($0, /^[[:space:]]*/)
            printf "%slocal_path: &corpus_path %s\n", substr($0, 1, RLENGTH), dir
            replaced = 1
            next
        }
        { print }
        END { if (!replaced) { exit 3 } }
    ' "${src}" >"${dst}" || {
        echo "could not find the 'local_path: &corpus_path' anchor in ${src}" >&2
        return 1
    }
    printf '%s\n' "${dst}"
}

if [[ "${plan_only}" -eq 1 ]]; then
    # The rendered plan config is scratch (the real run writes its copy next to the results under
    # "${prefix}"), so it is removed on every exit path -- success, plan failure, or signal. This
    # trap is installed and cleared entirely inside the plan-only branch, which returns before the
    # run path installs its own EXIT trap below.
    plan_config_tmp="$(mktemp -t needle1m-plan-XXXXXX.yaml)"
    trap 'rm -f "${plan_config_tmp}"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    plan_config="$(render_config "${config}" "${plan_config_tmp}")" || exit 1
    for suite in ${suites}; do
        PYTHONPATH="${repo_dir}/eval" "${eval_python}" -m ninfer_eval plan \
            --config "${plan_config}" --suite "${suite}" --check-runtime
    done
    exit 0
fi

for required_file in "${config}" "${eval_python}"; do
    [[ -f "${required_file}" ]] || { echo "missing required file: ${required_file}" >&2; exit 1; }
done
if [[ "${engine}" == "ninfer" ]]; then
    for required_file in "${server_bin}" "${artifact}"; do
        [[ -f "${required_file}" ]] || { echo "missing required file: ${required_file}" >&2; exit 1; }
    done
    [[ -x "${server_bin}" ]] || { echo "${server_bin} must be executable" >&2; exit 1; }
else
    [[ -x "${vllm_script}" ]] || { echo "${vllm_script} must be an executable script" >&2; exit 1; }
fi
command -v curl >/dev/null 2>&1 || { echo "curl is required for the health check" >&2; exit 1; }

mkdir -p -- "${log_dir}"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
prefix="${log_dir}/needle1m-${step}-${run_stamp}"
server_log="${prefix}.server.log"
request_log="${prefix}.requests.jsonl"
vram_log="${prefix}.vram.csv"
gpu_log="${prefix}.nvidia-smi.txt"
config_in_use="$(render_config "${config}" "${prefix}.config.yaml")" || exit 1

server_pid=""
sampler_pid=""
vram_sample_seconds="${NINFER_VRAM_SAMPLE_SECONDS:-5}"

# --- GPU discipline -------------------------------------------------------------------
# Both engines need both cards. Refuse to launch alongside anything else, and never signal a
# process this script did not start. Both halves are assertions, not printouts.
compute_app_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:blank:]' | sort -u
}

record_compute_apps() {
    local label="$1"
    {
        printf '=== compute processes %s (%s) ===\n' "${label}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv 2>/dev/null || true
    } | tee -a "${gpu_log}"
}

assert_gpus_free_before_launch() {
    record_compute_apps "before launch"
    if [[ -n "$(compute_app_pids)" ]]; then
        echo "  gate  no compute process before launch  FAIL" | tee -a "${gpu_log}"
        echo "another process is already using a GPU; this script will not start alongside it" >&2
        echo "(it never signals a process it did not start -- stop the other job first)" >&2
        exit 1
    fi
    echo "  gate  no compute process before launch  PASS" | tee -a "${gpu_log}"
}

assert_gpus_free_after_exit() {
    # Give the driver a moment to reap the process group.
    local waited=0
    while [[ -n "$(compute_app_pids)" && "${waited}" -lt 120 ]]; do
        sleep 2
        waited=$((waited + 2))
    done
    record_compute_apps "after exit"
    if [[ -n "$(compute_app_pids)" ]]; then
        echo "  gate  no compute process after exit  FAIL: $(compute_app_pids | tr '\n' ' ')" \
            | tee -a "${gpu_log}"
        return 1
    fi
    echo "  gate  no compute process after exit  PASS" | tee -a "${gpu_log}"
}

# --- VRAM / power sampler (uuid->index joined so the two ranks stay distinct) -----------
# CSV: timestamp_utc,kind,key,used_mib[,power_limit_w,power_draw_w]
#   kind=gpu      key=<device index>      includes memory other processes hold; carries the two
#                                         power columns, so every artifact records the per-GPU
#                                         limit it was measured under (400 W here, against
#                                         600/575 W stock).
#   kind=process  key=<pid>@gpu<index>    this server's own residency on that card; 4 columns.
# `report_vram_peaks` reads both this layout and the older 4-column one.
sample_vram() {
    local out="$1" interval="$2" uuid_map
    printf 'timestamp_utc,kind,key,used_mib,power_limit_w,power_draw_w\n' >"${out}"
    uuid_map="$(nvidia-smi --query-gpu=uuid,index --format=csv,noheader,nounits 2>/dev/null \
        | tr -d '[:blank:]')"
    while true; do
        local now
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

report_vram_peaks() {
    local out="$1"
    [[ -s "${out}" ]] || return 0
    echo "vram samples: ${out}"
    # Tolerant of both CSV layouts: the power columns are optional, so a sampler CSV that
    # predates them (4 columns) reports memory peaks and simply omits the power line.
    awk -F, 'NR > 1 && $4 != "" && $4 + 0 == $4 {
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
        for (key in peak) {
            printf "  peak %-24s %8d MiB  (%.2f GiB, %d samples)\n", key, peak[key], peak[key] / 1024, ticks[key]
        }
        for (g in draw_n) {
            printf "  power gpu %-18s limit %.0f W, draw peak %.0f W, mean %.0f W (%d samples)\n", \
                g, limit[g], draw_peak[g], draw_sum[g] / draw_n[g], draw_n[g]
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
        if [[ "${engine}" == "vllm" ]]; then
            # `vllm serve` forks engine-core and worker children. The script was started under
            # setsid, so the whole tree shares one process group; signal the group so no worker
            # is left holding a GPU. Read the group id from the process rather than assuming it
            # equals the pid -- that only holds when setsid did not have to fork.
            local pgid
            pgid="$(ps -o pgid= -p "${server_pid}" 2>/dev/null | tr -d '[:blank:]')"
            if [[ -n "${pgid}" ]]; then
                kill -TERM -- "-${pgid}" 2>/dev/null || true
            else
                kill -TERM "${server_pid}" 2>/dev/null || true
            fi
            for _ in $(seq 1 180); do
                kill -0 "${server_pid}" 2>/dev/null || break
                sleep 1
            done
            if kill -0 "${server_pid}" 2>/dev/null; then
                echo "vLLM did not stop on SIGTERM within 180 s; sending SIGKILL to the group" >&2
                if [[ -n "${pgid}" ]]; then
                    kill -KILL -- "-${pgid}" 2>/dev/null || true
                else
                    kill -KILL "${server_pid}" 2>/dev/null || true
                fi
            fi
        else
            kill -TERM "${server_pid}"
        fi
        wait "${server_pid}" 2>/dev/null || true
    fi
    server_pid=""
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_health() {
    local url="$1" attempts="$2"
    for ((attempt = 1; attempt <= attempts; ++attempt)); do
        if curl --fail --silent --show-error --max-time 5 "${url}" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            wait "${server_pid}" 2>/dev/null || true
            echo "server exited before becoming ready; see ${server_log}" >&2
            exit 1
        fi
        sleep 2
    done
    echo "server did not become ready in time; see ${server_log}" >&2
    exit 1
}

start_ninfer() {
    if curl --fail --silent --max-time 2 "http://127.0.0.1:${ninfer_port}/health" >/dev/null 2>&1; then
        echo "port ${ninfer_port} already has a healthy service; stop it first" >&2
        exit 1
    fi
    "${server_bin}" "${artifact}" \
        --host 127.0.0.1 \
        --port "${ninfer_port}" \
        --model-id qwen3.8-27b \
        --tp 2 \
        --devices 0,1 \
        --rope yarn \
        --yarn-factor 4.0 \
        --yarn-origin 262144 \
        --max-context "${max_context}" \
        --kv-capacity auto \
        --kv-dtype int8 \
        --max-concurrency 1 \
        --max-pending-requests 4 \
        --pending-timeout-ms 86400000 \
        --prefill-chunk 1024 \
        --log-stats-interval-ms 30000 \
        --request-log-jsonl "${request_log}" \
        >"${server_log}" 2>&1 &
    server_pid=$!
    wait_for_health "http://127.0.0.1:${ninfer_port}/health" 300
}

start_vllm() {
    if curl --fail --silent --max-time 2 "http://127.0.0.1:${vllm_port}/health" >/dev/null 2>&1; then
        echo "port ${vllm_port} already has a healthy service; stop it first" >&2
        exit 1
    fi
    setsid "${vllm_script}" >"${server_log}" 2>&1 &
    server_pid=$!
    # vLLM compiles and loads FP8 weights; 900 x 2 s = 30 minutes of patience.
    wait_for_health "http://127.0.0.1:${vllm_port}/health" 900
}

echo "step:        ${step} (${engine})"
echo "suites:      ${suites}"
echo "description: ${description}"
echo "server log:  ${server_log}"
echo "gpu log:     ${gpu_log}"
echo "vram log:    ${vram_log}"
[[ "${engine}" == "ninfer" ]] && echo "request log: ${request_log}"

assert_gpus_free_before_launch

if [[ "${engine}" == "ninfer" ]]; then
    start_ninfer
else
    start_vllm
fi

sample_vram "${vram_log}" "${vram_sample_seconds}" &
sampler_pid=$!

status=0
for suite in ${suites}; do
    echo "=== suite ${suite} ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
    if ! PYTHONPATH="${repo_dir}/eval" "${eval_python}" -m ninfer_eval run \
        --config "${config_in_use}" --suite "${suite}"; then
        echo "suite ${suite} FAILED" >&2
        status=1
        break
    fi
done

cleanup
report_vram_peaks "${vram_log}"
assert_gpus_free_after_exit || status=1
echo "step ${step} finished with status ${status} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "${status}"
