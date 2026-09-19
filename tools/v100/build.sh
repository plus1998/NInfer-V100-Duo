#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/../.." && pwd)

readonly deps_dir="${repo_dir}/build/_deps"
readonly build_dir="${repo_dir}/build-v100-duo"

NINFER_DEPS_DIR="${deps_dir}" "${script_dir}/build_dependencies.sh"

PKG_CONFIG_PATH="${deps_dir}/install/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}" \
    cmake -S "${repo_dir}" -B "${build_dir}" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build "${build_dir}" -j
