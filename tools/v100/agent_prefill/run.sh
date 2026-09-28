#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 ARTIFACT.ninfer OUTPUT_DIR [BENCH_BUILD_DIR]" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/../../.." && pwd)
artifact=$(realpath -- "$1")
output_dir=$(realpath -m -- "$2")
build_dir=${3:-${repo_dir}/build-v100-duo}

if [[ ! -f "${artifact}" ]]; then
    echo "missing artifact: ${artifact}" >&2
    exit 1
fi
if [[ "${output_dir}" == "${repo_dir}" || "${output_dir}" == "${repo_dir}/"* ]]; then
    echo 'captured prompts must be written outside the repository' >&2
    exit 2
fi
mkdir -p -- "${output_dir}"

export LD_LIBRARY_PATH="${repo_dir}/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${repo_dir}"
node "${script_dir}/capture.mjs" "${output_dir}"

for agent in pi codex claude; do
    "${build_dir}/bench/ninfer_v100_corpus" "${artifact}" \
        "${output_dir}/${agent}.ids" "${output_dir}/${agent}.prompt.txt"
    count=$(wc -l < "${output_dir}/${agent}.ids")
    if (( count > 32768 )); then
        echo "${agent}: ${count} tokens exceeds the 32K benchmark context" >&2
        exit 1
    fi
    for chunk in 1024 4096 5120; do
        "${build_dir}/bench/ninfer_bench" --weights "${artifact}" \
            --corpus "${output_dir}/${agent}.ids" --tp 2 --devices 0,1 \
            --kv-dtype int8 --max-ctx 32768 --prefill-chunk "${chunk}" \
            -p "${count}" -r 2 --warmup 1 --output json --capture-generation \
            --output-file "${output_dir}/${agent}-c${chunk}.json"
    done
done
