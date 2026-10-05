#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 STABLE_DIFFUSION_CPP_CHECKOUT_OR_URL" >&2; exit 2; }
script_dir=$(cd "$(dirname "$0")" && pwd -P)
repo_root=$(cd "$script_dir/../../.." && pwd -P)
mkdir -p "$repo_root/target"
clone=$(mktemp -d "$repo_root/target/ernie-oracle-source.XXXXXX")
pin=3f8527a46c54ecf4cb4ed6003da8e8982283c73c
ggml_pin=89c4413f5da6fb20cc796f16033d37f129be81fd
git clone --no-checkout "$1" "$clone" >&2
git -C "$clone" cat-file -e "$pin^{commit}" || git -C "$clone" fetch origin "$pin" >&2
git -C "$clone" checkout --detach "$pin" >&2
git -C "$clone" submodule update --init --recursive >&2
[[ $(git -C "$clone/ggml" rev-parse HEAD) == "$ggml_pin" ]]
git -C "$clone" apply --check "$script_dir/stable-diffusion-trace.patch"
git -C "$clone/ggml" apply --check "$script_dir/ggml-trace.patch"
git -C "$clone" apply "$script_dir/stable-diffusion-trace.patch"
git -C "$clone/ggml" apply "$script_dir/ggml-trace.patch"
cmake -S "$clone" -B "$clone/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS='-U__ARM_NEON -U__ARM_NEON__ -ffp-contract=off' \
    -DCMAKE_CXX_FLAGS='-U__ARM_NEON -U__ARM_NEON__ -ffp-contract=off' \
    -DSD_METAL=OFF -DSD_VULKAN=OFF -DSD_CUDA=OFF -DSD_HIPBLAS=OFF \
    -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DGGML_CUDA=OFF \
    -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DGGML_CPU_KLEIDIAI=OFF \
    -DGGML_NATIVE=OFF -DGGML_CPU_REPACK=OFF \
    -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF \
    -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_SSE42=OFF \
    -DSD_SERVER_BUILD_FRONTEND=OFF >&2
cmake --build "$clone/build" --target sd-cli --parallel "${RMI_BUILD_JOBS:-8}" >&2
printf '%s\n' "$clone/build/bin/sd-cli"
