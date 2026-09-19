#!/usr/bin/env bash
# Whole vLLM phase: one FRESH server per context tier, so every prefill is a genuine
# prefix-cache miss (the measurement host's production script enables --enable-prefix-caching).
# usage: vllm_phase.sh <label>:<context>:<depth>[:warmrepeat] ...
set -uo pipefail
repo=/home/pc/Projects/ninfer-1m
out=$repo/eval/results/cross-engine-nvfp4
py=$repo/eval/.venv/bin/python
tmp=/home/pc/scratch/cross-engine-nvfp4

metrics() {
  curl -s --max-time 10 http://127.0.0.1:8000/metrics \
    | grep -E '^vllm:(prefix_cache_queries_total|prefix_cache_hits_total|spec_decode_num_draft_tokens_total|spec_decode_num_accepted_tokens_total|spec_decode_num_drafts_total|spec_decode_num_accepted_tokens_per_pos_total)' | sed 's/^/    /'
}

stop_vllm() {
  # vLLM's TP workers (VLLM::Worker_TPn) are re-parented to init and have survived SIGTERM to
  # the serve process group on this host, holding ~14.6 GiB each and starving the next boot.
  # So: TERM the group, wait, then SIGKILL anything still holding GPU memory that is
  # identifiably ours (a vllm serve / VLLM::Worker process), and only then declare the GPUs free.
  local pgid pids p
  pgid="$(ps -eo pid,pgid,args | grep -E '[v]llm serve' | awk '{print $2}' | sort -u | head -1)"
  if [[ -n $pgid ]]; then
    echo "  stopping vllm pgid=$pgid (TERM)"
    kill -TERM -"$pgid" 2>/dev/null
  fi
  for ((i=0;i<60;++i)); do
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && break
    sleep 1
  done
  pids="$(ps -eo pid,args | grep -E '[V]LLM::|[v]llm serve|unsloth-nvfp4-env.*resource_tracker' | awk '{print $1}')"
  if [[ -n $pids ]]; then
    echo "  survivors after TERM: $(echo $pids | tr '\n' ' ') -> KILL"
    for p in $pids; do kill -KILL "$p" 2>/dev/null; done
  fi
  for ((i=0;i<120;++i)); do
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && break
    sleep 1
  done
  sleep 3
  local left
  left="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"
  echo "  compute apps after stop: [$left]"
  if [[ -n $left ]]; then echo "  REFUSING to continue: GPUs still busy"; return 1; fi
  return 0
}

start_vllm() {
  local log="$tmp/vllm-long-$(date -u +%H%M%S).log"
  echo "  starting vllm -> $log"
  ( cd /home/pc/Projects/vllm && nohup setsid ./serve-qwen38-27b-long.sh > "$log" 2>&1 & )
  for ((i=0;i<200;++i)); do
    grep -q "Application startup complete" "$log" 2>/dev/null && { echo "  ready after $((i*5))s"; VLLM_LOG="$log"; return 0; }
    sleep 5
  done
  echo "  vLLM did not become ready in 1000s"; tail -20 "$log"; return 1
}

for spec in "$@"; do
  IFS=: read -r label ctx depth warm <<<"$spec"
  echo "=== TIER $label ctx=$ctx depth=$depth $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  stop_vllm || exit 1
  start_vllm || exit 1
  grep -E "GPU KV cache size|Available KV cache memory|Maximum concurrency" "$VLLM_LOG" | sed 's/^/  /'
  echo "$VLLM_LOG" > "$out/$label.vllm-log-path.txt"
  echo "  metrics-before:"; metrics
  echo "### START $label $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$py" "$out/xprobe.py" --base-url http://127.0.0.1:8000/v1 --model my-model --vllm \
    --context-tokens "$ctx" --depth "$depth" --max-tokens 512 --task summary \
    --label "$label" --out "$out/vllm-probes.jsonl"
  rc=$?
  echo "### END $label rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "  metrics-after:"; metrics
  if [[ -n ${warm:-} ]]; then
    echo "### START $label-warm $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "$py" "$out/xprobe.py" --base-url http://127.0.0.1:8000/v1 --model my-model --vllm \
      --context-tokens "$ctx" --depth "$depth" --max-tokens 512 --task summary \
      --label "$label-warm" --out "$out/vllm-probes.jsonl"
    echo "### END $label-warm $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "  metrics-after-warm:"; metrics
  fi
  [[ $rc -ne 0 ]] && exit $rc
done
echo "=== VLLM PHASE COMPLETE, stopping server"
stop_vllm
echo "### ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
