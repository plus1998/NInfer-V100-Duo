#!/usr/bin/env bash
set -euo pipefail

# Production server launcher for the two-card Volta profile.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/../.." && pwd)

readonly executable="${repo_dir}/build-v100-duo/apps/ninfer-serve"
readonly runtime_lib_dir="${repo_dir}/build/_deps/install/lib"
readonly cuda_lib_dir=/usr/local/cuda-12.8/lib64

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
usage: ${BASH_SOURCE[0]} model=PATH [draft-tokens=N] [ninfer-serve options]

Starts the HTTP server with the dual-V100 production defaults:
  --tp 2 --devices 0,1 --max-context 180224 --prefill-chunk 4096 --kv-dtype int8
  --spec mtp --draft-tokens 3 --lm-head-draft
  --max-concurrency 1 --host 127.0.0.1 --port 8080

model=PATH and draft-tokens=N are read from the leading key=value arguments; draft-tokens defaults
to 3 and takes any MTP window in 1..5 (N=4 is the faster measured window: acceptance rate down,
decode speed and tokens per round up; the 180224-token default remains within its capacity).
Any remaining arguments are passed to ninfer-serve after these defaults and therefore override them.

For the previous long-context configuration, pass --max-context 200000 --prefill-chunk 1024
together: 200000 plus the 4096-token default chunk exceeds the 16 GB reservation.

Vision stays off by default. For the recommended single-request vision profile on 16 GB cards,
pass --vision --vision-max-tokens 2048 --max-context 155648 --prefill-chunk 1024
--max-concurrency 1 --kv-capacity 155648 (about 881 MiB free on the primary GPU).
For text-only concurrency, use --max-context 155648 --prefill-chunk 4096
--max-concurrency 2 --kv-capacity 155648 (about 703 MiB free on the primary GPU).
The KV pool is shared: each request can reach its context ceiling individually, but two
full-length requests need not fit together. Increase the visual-token budget only if needed;
the text-only defaults do not fit with Vision. See README.md for four-slot and vision-concurrent
profiles and measured throughput.

For the GSQ-RCO IQ3_S v3 artifact on 2 x 16 GB, native context and three Vision/text slots,
pass --vision --vision-max-tokens 2048 --max-context 262144 --prefill-chunk 1024
--max-concurrency 3 --kv-capacity 400000 (only about 283 MiB free per GPU at startup).
Use --kv-capacity auto for about 1 GiB of planned margin; the shared KV pool does not
provide three full 262144-token entitlements.
EOF
    exit 0
fi

artifact=
draft_tokens=3
while [[ $# -gt 0 && "${1}" == ?*=?* ]]; do
    key=${1%%=*}
    value=${1#*=}
    case "${key}" in
        model)
            artifact=${value} ;;
        draft-tokens)
            if [[ ! "${value}" =~ ^[1-5]$ ]]; then
                echo "draft-tokens=N must be an integer in 1..5 (see --help)" >&2
                exit 2
            fi
            draft_tokens=${value} ;;
        *)
            break ;;
    esac
    shift
done
readonly artifact draft_tokens
if [[ -z "${artifact}" ]]; then
    echo "first argument must be model=PATH (see --help)" >&2
    exit 2
fi

if [[ ! -x "${executable}" ]]; then
    echo "ninfer executable is missing: ${executable}" >&2
    echo "build it with tools/v100/build.sh" >&2
    exit 1
fi
if [[ ! -f "${artifact}" ]]; then
    echo "V100 Duo artifact is missing: ${artifact}" >&2
    exit 1
fi

# A source build keeps FFmpeg and CUDA beside the build tree rather than installing them system
# wide.  Make the launcher self-contained for the normal build-v100-duo layout while preserving any
# caller-provided library path (and without forcing a path when a custom executable has its own
# rpath).  This does not change CPU scheduling: the executor remains event/condition-variable
# driven and no artificial affinity or OMP limit is installed.
runtime_ld_parts=()
if [[ -d "${runtime_lib_dir}" ]]; then runtime_ld_parts+=("${runtime_lib_dir}"); fi
if [[ -d "${cuda_lib_dir}" ]]; then runtime_ld_parts+=("${cuda_lib_dir}"); fi
if [[ ${#runtime_ld_parts[@]} -gt 0 ]]; then
    runtime_ld_path=$(IFS=:; echo "${runtime_ld_parts[*]}")
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        export LD_LIBRARY_PATH="${runtime_ld_path}:${LD_LIBRARY_PATH}"
    else
        export LD_LIBRARY_PATH="${runtime_ld_path}"
    fi
fi

# The worker uses condition-variable blocking while CUDA performs the decode.  Do not set
# OMP_NUM_THREADS or an artificial CPU affinity here: that needlessly leaves host capacity idle.
# If a deployment has an external CPU quota, it can apply that quota to this process without
# changing the inference defaults.
exec "${executable}" "${artifact}" \
    --tp 2 --devices 0,1 \
    --max-context 180224 --prefill-chunk 4096 \
    --kv-dtype int8 \
    --spec mtp --draft-tokens "${draft_tokens}" --lm-head-draft \
    --max-concurrency 1 \
    --host 127.0.0.1 --port 8080 \
    "$@"
