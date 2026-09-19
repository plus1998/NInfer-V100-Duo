#!/usr/bin/env bash
set -euo pipefail

# Convenience launcher for the two-card Volta profile.  Runtime defaults are deliberately kept
# here (rather than in the Engine) so the public CLI remains target-agnostic and the same binary
# can still be used for a single card or for a different context budget.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/../.." && pwd)

executable=${NINFER_V100_DUO_EXECUTABLE:-"${repo_dir}/build-v100/apps/ninfer"}
artifact=${NINFER_V100_DUO_ARTIFACT:-~/models/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer}
devices=${NINFER_V100_DUO_DEVICES:-0,1}
max_context=${NINFER_V100_DUO_MAX_CONTEXT:-196608}
kv_dtype=${NINFER_V100_DUO_KV_DTYPE:-int8}
draft_tokens=${NINFER_V100_DUO_DRAFT_TOKENS:-3}
proposal_head=${NINFER_V100_DUO_PROPOSAL_HEAD-optimized}
runtime_lib_dir=${NINFER_V100_DUO_RUNTIME_LIBDIR:-"${repo_dir}/build/_deps/install/lib"}
cuda_lib_dir=${NINFER_V100_DUO_CUDA_LIBDIR:-/usr/local/cuda-12.8/lib64}

proposal_args=()
case "${proposal_head}" in
    full) ;;
    optimized) proposal_args+=(--lm-head-draft) ;;
    *)
        echo "NINFER_V100_DUO_PROPOSAL_HEAD must be full or optimized (got '${proposal_head}')" >&2
        exit 2
        ;;
esac

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
usage: ${BASH_SOURCE[0]} (--prompt TEXT | --messages FILE) [ninfer options]

Defaults: artifact=${artifact}
          devices=${devices} max-context=${max_context} kv-dtype=${kv_dtype}
          spec=mtp draft-tokens=${draft_tokens} proposal-head=${proposal_head}

Environment overrides: NINFER_V100_DUO_EXECUTABLE, NINFER_V100_DUO_ARTIFACT,
NINFER_V100_DUO_DEVICES, NINFER_V100_DUO_MAX_CONTEXT, NINFER_V100_DUO_KV_DTYPE,
NINFER_V100_DUO_DRAFT_TOKENS, NINFER_V100_DUO_PROPOSAL_HEAD (full|optimized),
NINFER_V100_DUO_RUNTIME_LIBDIR, NINFER_V100_DUO_CUDA_LIBDIR.
EOF
    exit 0
fi
if [[ $# -eq 0 ]]; then
    echo "use --prompt TEXT or --messages FILE (see --help)" >&2
    exit 2
fi

if [[ ! -x "${executable}" ]]; then
    echo "ninfer executable is missing: ${executable}" >&2
    echo "build it with tools/v100/build_dependencies.sh and the README's sm_70 CMake command" >&2
    exit 1
fi
if [[ ! -f "${artifact}" ]]; then
    echo "V100 Duo artifact is missing: ${artifact}" >&2
    echo "set NINFER_V100_DUO_ARTIFACT to the official qwen3_8_27b_nvfp4.ninfer" >&2
    exit 1
fi

# A source build keeps FFmpeg and CUDA beside the build tree rather than installing them system
# wide.  Make the launcher self-contained for the normal build-v100 layout while preserving any
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
    --tp 2 --devices "${devices}" \
    --max-context "${max_context}" \
    --kv-dtype "${kv_dtype}" \
    --spec mtp --draft-tokens "${draft_tokens}" \
    "${proposal_args[@]}" \
    "$@"
