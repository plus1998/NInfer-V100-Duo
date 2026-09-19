#!/usr/bin/env bash
# GPU VRAM + power sampler. usage: sample_gpu.sh <out.csv> <interval_s>
out="$1"; interval="${2:-3}"
echo "timestamp_utc,kind,key,used_mib,power_limit_w,power_draw_w" >"$out"
while true; do
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-gpu=index,memory.used,power.limit,power.draw --format=csv,noheader,nounits |
    while IFS=, read -r idx used lim draw; do
      printf '%s,gpu,%s,%s,%s,%s\n' "$now" "${idx// /}" "${used// /}" "${lim// /}" "${draw// /}" >>"$out"
    done
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits |
    while IFS=, read -r pid uuid used; do
      printf '%s,process,%s@%s,%s,,\n' "$now" "${pid// /}" "${uuid// /}" "${used// /}" >>"$out"
    done
  sleep "$interval"
done
