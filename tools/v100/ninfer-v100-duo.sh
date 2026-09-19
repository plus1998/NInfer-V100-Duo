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
usage: ${BASH_SOURCE[0]} model=PATH [ninfer-serve options]

Starts the HTTP server with the dual-V100 production defaults:
  --tp 2 --devices 0,1 --max-context 196608 --kv-dtype int8
  --spec mtp --draft-tokens 3 --lm-head-draft
  --max-concurrency 1 --host 127.0.0.1 --port 8080

Additional options are passed to ninfer-serve after these defaults.
EOF
    exit 0
fi

if [[ "${1:-}" != model=* || -z "${1#model=}" ]]; then
    echo "first argument must be model=PATH (see --help)" >&2
    exit 2
fi
readonly artifact=${1#model=}
shift

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
    --max-context 196608 \
    --kv-dtype int8 \
    --spec mtp --draft-tokens 3 --lm-head-draft \
    --max-concurrency 1 \
    --host 127.0.0.1 --port 8080 \
    "$@"
