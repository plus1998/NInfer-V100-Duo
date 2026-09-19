#!/usr/bin/env bash
# Re-measure the publishable TP2 performance set at DEFAULT (unlifted) GPU power
# limits, and compare against vLLM serving an NVFP4 quantization of the SAME base checkpoint.
#
# Why this script exists
# ----------------------
# Every earlier performance figure on this branch was measured with both
# RTX 5090s held at a 400 W per-GPU cap -- the minimum settable limit on these cards. Those
# figures are lower bounds, not the hardware's behaviour. This script re-runs the publishable
# subset with the caps lifted, and records the power condition INTO EVERY ARTIFACT rather than
# assuming it: `nvidia-smi --query-gpu=index,power.limit,power.default_limit,power.max_limit` is
# captured to `<prefix>.power.txt` before each step, and the VRAM sampler records `power.limit`
# and `power.draw` per tick.
#
# Read the power file next to any number this script produces. Do not assume "stock" means the
# vendor default: GPU 1's vendor default on this host is 575 W and it can be raised to its 600 W
# maximum, so the two cards may or may not match.
#
# Steps
# -----
#   matched-250k   The headline comparison: a byte-identical 249,955-token needle
#                  prompt through four engine configurations (tp2/tp1 x MTP-off/MTP3), 512
#                  generated tokens. One ninfer-serve process per configuration.
#   concurrency    Saturated concurrent decode at a 262,144-token window, C=1 and C=4, MTP off
#                  and MTP3, through tools/bench/run_serve_concurrency.py.
#   long-1m        The 1,048,576-token configuration: one ~1,046k-token request with MTP off and
#                  one with MTP3, on one server process each.
#   vllm-nvfp4     vLLM TP2 serving sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4 -- an
#                  independent NVFP4 quantization of the SAME abliterated base fine-tune that
#                  NInfer's artifact was converted from -- at the context given by
#                  NINFER_VLLM_NVFP4_MAX_LEN (default 262144), then the same prompt through
#                  NInfer at the same context. This removes the different-checkpoint confound of
#                  the earlier FP8 control; the residual difference is the QUANTIZER, not the
#                  fine-tune.
#   ninfer-at      NInfer alone at NINFER_VLLM_NVFP4_MAX_LEN (for pairing with a vLLM run taken
#                  separately).
#
# Both engines take BOTH GPUs, so nothing here ever runs two servers at once, and every step
# refuses to start while any compute process holds GPU memory.
#
# Environment overrides:
#   NINFER_QWEN3_8_27B_NVFP4_ARTIFACT   the .ninfer artifact
#   NINFER_VLLM_NVFP4_MAX_LEN           vLLM's --max-model-len for the comparison (default 262144)
#   NINFER_VLLM_NVFP4_KV_DTYPE          vLLM --kv-cache-dtype (default fp8)
#   NINFER_VLLM_NVFP4_GPU_UTIL          vLLM --gpu-memory-utilization (default 0.92)
#   NINFER_VLLM_ENV                     the vLLM virtualenv (default the unsloth-nvfp4-env one)
#   NINFER_VRAM_SAMPLE_SECONDS          sampler interval (default 3)
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
server_bin="${repo_dir}/build/apps/ninfer-serve"
artifact="${NINFER_QWEN3_8_27B_NVFP4_ARTIFACT:-/home/pc/models/ninfer-38/huihui-nvfp4/qwen3_8_27b_nvfp4.ninfer}"
eval_python="${repo_dir}/eval/.venv/bin/python"
log_dir="${repo_dir}/eval/server-logs"
results_dir="${repo_dir}/eval/results/final-rerun"
vllm_env="${NINFER_VLLM_ENV:-/home/pc/Projects/vllm/unsloth-nvfp4-env}"
vllm_model="sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4"
vllm_max_len="${NINFER_VLLM_NVFP4_MAX_LEN:-262144}"
# bfloat16, not fp8: an fp8 KV cache on this hybrid GDN model with the FlashInfer backend hung
# this checkpoint's startup indefinitely (weights resident, no further log output, both workers
# near-idle for 20 minutes). bf16 KV is the configuration the host's own production serve script
# uses for this checkpoint, and it is what these numbers were taken with.
vllm_kv_dtype="${NINFER_VLLM_NVFP4_KV_DTYPE:-bfloat16}"
vllm_gpu_util="${NINFER_VLLM_NVFP4_GPU_UTIL:-0.85}"
vram_sample_seconds="${NINFER_VRAM_SAMPLE_SECONDS:-3}"
ninfer_port=18080
vllm_port=8000

# The matched comparison prompt. 250,000 nominal haystack tokens land at a 249,955-token prompt
# with this tokenizer and template -- the same prompt the 400 W TP1/TP2 row used, so the
# 400 W and default-limit columns are the same measurement under two power conditions.
probe_context=250000
probe_depth=50
probe_tokens=512
# The extended-context probe. 1,046,000 nominal lands at ~1,045,954, the needle suite's 1M tier.
long_context=1046000

usage() {
    cat >&2 <<'EOF'
usage: run_qwen3_8_27b_nvfp4_stock_power_perf.sh [--plan] <step>

steps:
  matched-250k   tp2/tp1 x MTP-off/MTP3 on one byte-identical 249,955-token prompt
  concurrency    saturated decode at 262,144, C=1 and C=4, MTP off and MTP3
  long-1m        1,048,576-token configuration, ~1,046k prompt, MTP off then MTP3
  vllm-nvfp4     vLLM TP2 on the NVFP4 checkpoint, then NInfer at the same context
  ninfer-at      NInfer alone at NINFER_VLLM_NVFP4_MAX_LEN
  all            matched-250k -> concurrency -> long-1m  (NInfer only)
EOF
}

plan_only=0
if [[ "${1:-}" == "--plan" ]]; then
    plan_only=1
    shift
fi
if [[ $# -ne 1 ]]; then usage; exit 2; fi
step="$1"
case "${step}" in
    matched-250k|concurrency|long-1m|vllm-nvfp4|ninfer-at|all) ;;
    *) usage; exit 2 ;;
esac

for required in "${server_bin}" "${artifact}" "${eval_python}"; do
    if [[ ! -e "${required}" ]]; then
        echo "missing: ${required}" >&2
        exit 1
    fi
done
mkdir -p "${log_dir}" "${results_dir}"

server_pid=""
sampler_pid=""
cleanup() {
    if [[ -n "${sampler_pid}" ]]; then kill "${sampler_pid}" 2>/dev/null || true; wait "${sampler_pid}" 2>/dev/null || true; sampler_pid=""; fi
    if [[ -n "${server_pid}" ]]; then
        kill "${server_pid}" 2>/dev/null || true
        for ((i = 0; i < 60; ++i)); do kill -0 "${server_pid}" 2>/dev/null || break; sleep 1; done
        kill -9 "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
        server_pid=""
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Refuses to start while anything else holds GPU memory. Never signals a foreign process.
require_free_gpus() {
    local busy
    busy="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true)"
    if [[ -n "${busy}" ]]; then
        echo "GPUs are busy; not starting a server:" >&2
        echo "${busy}" >&2
        exit 1
    fi
}

# The power condition, captured rather than assumed. Written next to every artifact.
record_power() {
    local out="$1"
    {
        echo "# captured $(date -u +%Y-%m-%dT%H:%M:%SZ) by ${BASH_SOURCE[0]##*/} step=${step}"
        nvidia-smi --query-gpu=index,name,pci.bus_id,power.limit,power.default_limit,power.max_limit \
            --format=csv
    } >"${out}"
    echo "power condition: ${out}"
    sed -n '2,$p' "${out}" | sed 's/^/  /'
}

sample_vram() {
    local out="$1" interval="$2"
    echo "timestamp_utc,kind,key,used_mib,power_limit_w,power_draw_w" >"${out}"
    while true; do
        local now
        now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        nvidia-smi --query-gpu=index,memory.used,power.limit,power.draw --format=csv,noheader,nounits |
            while IFS=, read -r idx used lim draw; do
                printf '%s,gpu,%s,%s,%s,%s\n' "${now}" "${idx// /}" "${used// /}" "${lim// /}" "${draw// /}" >>"${out}"
            done
        nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits |
            while IFS=, read -r pid uuid used; do
                printf '%s,process,%s@%s,%s,,\n' "${now}" "${pid// /}" "${uuid// /}" "${used// /}" >>"${out}"
            done
        sleep "${interval}"
    done
}

report_vram_peaks() {
    local csv="$1"
    "${eval_python}" - "$csv" <<'PY'
import csv, sys, collections
peak = collections.defaultdict(int)
draw = collections.defaultdict(float)
limit = {}
with open(sys.argv[1]) as handle:
    for row in csv.DictReader(handle):
        key = (row["kind"], row["key"])
        peak[key] = max(peak[key], int(row["used_mib"] or 0))
        if row["kind"] == "gpu":
            if row.get("power_draw_w"):
                draw[key] = max(draw[key], float(row["power_draw_w"]))
            if row.get("power_limit_w"):
                limit[key] = row["power_limit_w"]
for key in sorted(peak):
    extra = ""
    if key in draw:
        extra = f"  peak draw {draw[key]:.1f} W of {limit.get(key, '?')} W"
    print(f"  {key[0]} {key[1]}: {peak[key]} MiB ({peak[key] / 1024:.2f} GiB){extra}")
PY
}

wait_healthy() {
    local url="$1" attempts="$2"
    for ((attempt = 1; attempt <= attempts; ++attempt)); do
        if curl --fail --silent --show-error --max-time 2 "${url}" >/dev/null 2>&1; then return 0; fi
        if [[ -n "${server_pid}" ]] && ! kill -0 "${server_pid}" 2>/dev/null; then
            echo "server exited before becoming ready" >&2
            return 1
        fi
        sleep 2
    done
    return 1
}

# start_ninfer <prefix> <tp> <max-context> <max-concurrency> [extra serve flags...]
start_ninfer() {
    local prefix="$1" tp="$2" max_context="$3" concurrency="$4"
    shift 4
    require_free_gpus
    record_power "${prefix}.power.txt"
    local devices=()
    if [[ "${tp}" == "2" ]]; then devices=(--tp 2 --devices 0,1); else devices=(--device 0); fi
    "${server_bin}" "${artifact}" \
        --host 127.0.0.1 --port "${ninfer_port}" --model-id qwen3.8-27b \
        "${devices[@]}" \
        --max-context "${max_context}" \
        --kv-capacity auto --kv-dtype int8 \
        --max-concurrency "${concurrency}" \
        --max-pending-requests 8 --pending-timeout-ms 86400000 \
        --prefill-chunk 1024 --log-stats-interval-ms 5000 \
        --request-log-jsonl "${prefix}.requests.jsonl" \
        "$@" >"${prefix}.server.log" 2>&1 &
    server_pid=$!
    if ! wait_healthy "http://127.0.0.1:${ninfer_port}/health" 300; then
        echo "ninfer-serve did not become ready; see ${prefix}.server.log" >&2
        exit 1
    fi
    sample_vram "${prefix}.vram.csv" "${vram_sample_seconds}" &
    sampler_pid=$!
}

# probe <label> <context-tokens> <base-url> <model>
probe() {
    local label="$1" context="$2" base_url="$3" model="$4"
    PYTHONPATH="${repo_dir}/eval" "${eval_python}" \
        "${repo_dir}/eval/tp2_needle_throughput_probe.py" \
        --base-url "${base_url}" --model "${model}" \
        --context-tokens "${context}" --depth "${probe_depth}" \
        --max-tokens "${probe_tokens}" --task summary \
        --label "${label}" --out "${results_dir}/stock-power-probes.jsonl"
}

stamp="$(date -u +%Y%m%dT%H%M%SZ)"

run_matched_250k() {
    local config
    for config in tp2-mtp0 tp2-mtp3 tp1-mtp0 tp1-mtp3; do
        local prefix="${log_dir}/stock-perf-matched250k-${config}-${stamp}"
        local tp=2 ctx=262144 extra=()
        [[ "${config}" == tp1-* ]] && { tp=1; ctx=252928; }
        [[ "${config}" == *-mtp3 ]] && extra=(--spec mtp --draft-tokens 3 --lm-head-draft)
        echo "=== matched 250k :: ${config} (tp${tp}, max-context ${ctx}) ==="
        start_ninfer "${prefix}" "${tp}" "${ctx}" 1 "${extra[@]+"${extra[@]}"}"
        probe "stock-${config}" "${probe_context}" "http://127.0.0.1:${ninfer_port}/v1" qwen3.8-27b
        # One repeat on the primary configuration only: enough to show the measurement is stable
        # without paying a second ~100 s prefill on every configuration.
        if [[ "${config}" == "tp2-mtp0" ]]; then
            probe "stock-${config}-repeat" "${probe_context}" "http://127.0.0.1:${ninfer_port}/v1" qwen3.8-27b
        fi
        cleanup
        report_vram_peaks "${prefix}.vram.csv"
    done
}

run_concurrency() {
    # `--mode` is run_serve_concurrency.py's own spelling: mtp0 means MTP off.
    local mode
    for mode in mtp0 mtp3; do
        local prefix="${log_dir}/stock-perf-concurrency-${mode}-${stamp}"
        require_free_gpus
        record_power "${prefix}.power.txt"
        local out="${repo_dir}/eval/runs/stock-power-concurrency-${mode}"
        local spec_mode="${mode}"
        echo "=== concurrency :: --mode ${spec_mode} ==="
        python3 "${repo_dir}/tools/bench/run_serve_concurrency.py" \
            --serve "${server_bin}" \
            --artifact "qwen3_8_27b=${artifact}" \
            --mode "${spec_mode}" --suite decode-saturation \
            --concurrency 1 --concurrency 4 \
            --tp 2 --devices 0,1 --device 0 \
            --max-context 262144 --kv-capacity 262144 \
            --output "${out}" 2>&1 | tee "${prefix}.bench.log"
    done
}

run_long_1m() {
    local config
    for config in mtp0 mtp3; do
        local prefix="${log_dir}/stock-perf-1m-${config}-${stamp}"
        local extra=()
        [[ "${config}" == "mtp3" ]] && extra=(--spec mtp --draft-tokens 3 --lm-head-draft)
        echo "=== 1M :: ${config} ==="
        start_ninfer "${prefix}" 2 1048576 1 \
            --rope yarn --yarn-factor 4.0 --yarn-origin 262144 \
            "${extra[@]+"${extra[@]}"}"
        probe "stock-1m-${config}" "${long_context}" "http://127.0.0.1:${ninfer_port}/v1" qwen3.8-27b
        cleanup
        report_vram_peaks "${prefix}.vram.csv"
    done
}

run_ninfer_at() {
    local prefix="${log_dir}/stock-perf-ninfer-at-${vllm_max_len}-${stamp}"
    local ctx="${vllm_max_len}"
    local rope=()
    if (( ctx > 262144 )); then rope=(--rope yarn --yarn-factor 4.0 --yarn-origin 262144); fi
    # The comparison prompt is sized to the served window with the same 5,000-token head-room
    # the needle configs leave for the answer and the template.
    local probe_ctx=$(( ctx - 5000 ))
    echo "=== NInfer at ${ctx} (probe context ${probe_ctx}) ==="
    start_ninfer "${prefix}" 2 "${ctx}" 1 "${rope[@]+"${rope[@]}"}"
    probe "stock-ninfer-at-${ctx}" "${probe_ctx}" "http://127.0.0.1:${ninfer_port}/v1" qwen3.8-27b
    cleanup
    report_vram_peaks "${prefix}.vram.csv"
}

run_vllm_nvfp4() {
    local prefix="${log_dir}/stock-perf-vllm-nvfp4-${vllm_max_len}-${stamp}"
    require_free_gpus
    record_power "${prefix}.power.txt"
    echo "=== vLLM NVFP4 TP2 :: max-model-len ${vllm_max_len}, kv ${vllm_kv_dtype}, util ${vllm_gpu_util} ==="
    # No speculative config and one sequence: the point of comparison is single-request
    # prefill and decode against NInfer's single-request rows, not vLLM's batching.
    (
        # shellcheck disable=SC1091
        source "${vllm_env}/bin/activate"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export HF_HUB_DISABLE_XET=1
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        exec vllm serve "${vllm_model}" \
            --served-model-name my-model \
            --host 127.0.0.1 --port "${vllm_port}" \
            --tensor-parallel-size 2 \
            --max-num-seqs 1 \
            --max-model-len "${vllm_max_len}" \
            --kv-cache-dtype "${vllm_kv_dtype}" \
            --gpu-memory-utilization "${vllm_gpu_util}" \
            --max-num-batched-tokens 16768 \
            --enable-prefix-caching \
            --async-scheduling \
            --reasoning-parser qwen3
    ) >"${prefix}.server.log" 2>&1 &
    server_pid=$!
    if ! wait_healthy "http://127.0.0.1:${vllm_port}/health" 900; then
        echo "vLLM did not become ready; see ${prefix}.server.log" >&2
        echo "--- last 40 lines ---" >&2
        tail -40 "${prefix}.server.log" >&2
        exit 1
    fi
    sample_vram "${prefix}.vram.csv" "${vram_sample_seconds}" &
    sampler_pid=$!
    # The SAME 250,000-token haystack prompt the NInfer matched-250k step sends, so the two
    # engines are compared on byte-identical input. vLLM's window may be larger; the window it
    # actually booted with is recorded in its server log and quoted alongside the numbers.
    probe "stock-vllm-nvfp4-${vllm_max_len}" "${probe_context}" "http://127.0.0.1:${vllm_port}/v1" my-model
    cleanup
    report_vram_peaks "${prefix}.vram.csv"
}

if [[ "${plan_only}" -eq 1 ]]; then
    echo "step: ${step}"
    echo "artifact: ${artifact}"
    echo "probe: ${probe_context} haystack tokens, depth ${probe_depth}, ${probe_tokens} generated"
    record_power "/dev/stdout" >/dev/null || true
    nvidia-smi --query-gpu=index,power.limit,power.default_limit,power.max_limit --format=csv
    exit 0
fi

case "${step}" in
    matched-250k) run_matched_250k ;;
    concurrency)  run_concurrency ;;
    long-1m)      run_long_1m ;;
    vllm-nvfp4)   run_vllm_nvfp4 ;;
    ninfer-at)    run_ninfer_at ;;
    all)          run_matched_250k; run_concurrency; run_long_1m ;;
esac

echo "=== step ${step} complete ==="
