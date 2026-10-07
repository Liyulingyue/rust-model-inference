#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { echo 'usage: build.sh SD_CPP_CHECKOUT' >&2; exit 2; }
reference_dir=$(cd "$1" && pwd -P)
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/rmi-longcat-oracle.XXXXXX")
source_dir="$work_dir/stable-diffusion.cpp"
git clone --no-checkout "$reference_dir" "$source_dir" >&2
git -C "$source_dir" checkout --detach 3f8527a46c54ecf4cb4ed6003da8e8982283c73c >&2
git -C "$source_dir" submodule update --init ggml >&2
script_dir=$(cd "$(dirname "$0")" && pwd -P)
[[ $(git -C "$source_dir" rev-parse HEAD) == 3f8527a46c54ecf4cb4ed6003da8e8982283c73c ]] || { echo 'wrong Oracle commit' >&2; exit 1; }
[[ $(git -C "$source_dir/ggml" rev-parse HEAD) == 89c4413f5da6fb20cc796f16033d37f129be81fd ]] || { echo 'wrong ggml commit' >&2; exit 1; }
# Applies checkpoint instrumentation and selects existing generic CPU kernels only.
python3 "$script_dir/instrument.py" "$source_dir"
cp "$script_dir/oracle.cpp" "$source_dir/longcat-oracle.cpp"
cmake -S "$source_dir" -B "$source_dir/build-longcat" \
    -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF \
    -DSD_METAL=OFF -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF \
    -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_CPU_KLEIDIAI=OFF -DGGML_CPU_REPACK=OFF \
    -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
    '-DCMAKE_C_FLAGS=-ffp-contract=off -fno-vectorize -fno-slp-vectorize -U__ARM_NEON -U__ARM_NEON__ -U__ARM_FEATURE_MATMUL_INT8 -U__ARM_FEATURE_DOTPROD' \
    '-DCMAKE_CXX_FLAGS=-ffp-contract=off -fno-vectorize -fno-slp-vectorize -U__ARM_NEON -U__ARM_NEON__ -U__ARM_FEATURE_MATMUL_INT8 -U__ARM_FEATURE_DOTPROD' >&2
cmake --build "$source_dir/build-longcat" --target longcat-oracle --parallel "${RMI_BUILD_JOBS:-4}" >&2

printf '%s\n' "$source_dir/build-longcat/bin/longcat-oracle"
