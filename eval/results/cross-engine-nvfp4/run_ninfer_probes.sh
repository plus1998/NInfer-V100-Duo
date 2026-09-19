#!/usr/bin/env bash
# NInfer TP2 + YaRN x4 1M-window cross-engine rows, one fresh server per configuration
# (so every prefill is cold -- prefix_cache_hit_tokens is asserted 0 in the request log).
# usage: run_ninfer_probes.sh <context>:<depth>:<mtp0|mtp3>[:<max-context>[:<rope>]] ...
set -uo pipefail
repo=/home/pc/Projects/ninfer-1m
out=$repo/eval/results/cross-engine-nvfp4
py=$repo/eval/.venv/bin/python
server=$repo/build/apps/ninfer-serve
artifact=${NINFER_QWEN3_8_27B_NVFP4_ARTIFACT:-/home/pc/models/ninfer-38/huihui-nvfp4/qwen3_8_27b_nvfp4.ninfer}
port=18080
server_pid=""; sampler_pid=""

cleanup() {
  [[ -n $sampler_pid ]] && { kill "$sampler_pid" 2>/dev/null; wait "$sampler_pid" 2>/dev/null; sampler_pid=""; }
  if [[ -n $server_pid ]]; then
    kill "$server_pid" 2>/dev/null
    for ((i=0;i<120;++i)); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$server_pid" 2>/dev/null; wait "$server_pid" 2>/dev/null; server_pid=""
  fi
}
trap cleanup EXIT INT TERM

for spec in "$@"; do
  IFS=: read -r ctx depth mtp maxctx rope <<<"$spec"
  maxctx=${maxctx:-1048576}
  rope=${rope:-yarn}
  label="ninfer-500w-${ctx}-${mtp}-ctx${maxctx}-${rope}"
  prefix="$out/$label"
  echo "### START $label $(date -u +%Y-%m-%dT%H:%M:%SZ)"

  busy="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)"
  if [[ -n $busy ]]; then echo "GPUs busy, refusing to launch:"; echo "$busy"; exit 1; fi
  { echo "# captured $(date -u +%Y-%m-%dT%H:%M:%SZ) :: $label"
    nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,power.max_limit --format=csv
  } > "$prefix.power.txt"

  ropeflags=()
  [[ $rope == yarn ]] && ropeflags=(--rope yarn --yarn-factor 4.0 --yarn-origin 262144)
  specflags=()
  [[ $mtp == mtp3 ]] && specflags=(--spec mtp --draft-tokens 3 --lm-head-draft)

  "$server" "$artifact" --host 127.0.0.1 --port $port --model-id qwen3.8-27b \
    --tp 2 --devices 0,1 \
    --max-context "$maxctx" --kv-capacity auto --kv-dtype int8 \
    --max-concurrency 1 --max-pending-requests 8 --pending-timeout-ms 86400000 \
    --prefill-chunk 1024 --log-stats-interval-ms 5000 \
    --request-log-jsonl "$prefix.requests.jsonl" \
    "${ropeflags[@]}" "${specflags[@]+"${specflags[@]}"}" \
    > "$prefix.server.log" 2>&1 &
  server_pid=$!

  ready=0
  for ((i=0;i<300;++i)); do
    curl --fail --silent --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ready=1; break; }
    kill -0 "$server_pid" 2>/dev/null || { echo "server died during boot; see $prefix.server.log"; tail -20 "$prefix.server.log"; exit 1; }
    sleep 2
  done
  [[ $ready -eq 1 ]] || { echo "server not ready"; tail -20 "$prefix.server.log"; exit 1; }
  echo "### READY $label $(date -u +%Y-%m-%dT%H:%M:%SZ)"

  "$out/sample_gpu.sh" "$prefix.vram.csv" 3 >/dev/null 2>&1 &
  sampler_pid=$!

  PYTHONPATH=$repo/eval "$py" "$repo/eval/tp2_needle_throughput_probe.py" \
    --base-url "http://127.0.0.1:$port/v1" --model qwen3.8-27b \
    --context-tokens "$ctx" --depth "$depth" --max-tokens 512 --task summary \
    --label "$label" --out "$out/ninfer-probes.jsonl"
  rc=$?
  echo "### PROBE-RC $label $rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cleanup
  echo "### PEAKS $label"
  "$py" "$out/peaks.py" "$prefix.vram.csv"
  echo "### END $label $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  [[ $rc -ne 0 ]] && exit $rc
done
echo "### ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
