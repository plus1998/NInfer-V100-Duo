#!/usr/bin/env bash
# Drive the needle throughput probe against a running vLLM server, capturing the
# prefix-cache counters from /metrics on either side of each request.
# usage: run_vllm_probes.sh <label>:<context>:<depth> ...
set -uo pipefail
repo=/home/pc/Projects/ninfer-1m
out=$repo/eval/results/cross-engine-nvfp4
py=$repo/eval/.venv/bin/python
metrics() {
  curl -s --max-time 10 http://127.0.0.1:8000/metrics \
    | grep -E '^vllm:(prefix_cache_queries_total|prefix_cache_hits_total|spec_decode_num_draft_tokens_total|spec_decode_num_accepted_tokens_total|spec_decode_num_drafts_total|spec_decode_num_accepted_tokens_per_pos)' \
    | sed 's/^/    /'
}
for spec in "$@"; do
  IFS=: read -r label ctx depth <<<"$spec"
  echo "### START $label ctx=$ctx depth=$depth $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "  metrics-before:"; metrics
  PYTHONPATH=$repo/eval "$py" "$repo/eval/tp2_needle_throughput_probe.py" \
    --base-url http://127.0.0.1:8000/v1 --model my-model \
    --context-tokens "$ctx" --depth "$depth" --max-tokens 512 --task summary \
    --label "$label" --out "$out/vllm-probes.jsonl"
  rc=$?
  echo "  metrics-after:"; metrics
  echo "### END $label rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  [[ $rc -ne 0 ]] && exit $rc
done
echo "### ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
