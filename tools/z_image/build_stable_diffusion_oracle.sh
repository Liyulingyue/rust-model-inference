#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 STABLE_DIFFUSION_CPP_CHECKOUT" >&2; exit 2; }
source_dir=$(cd "$1" && pwd -P)
git -C "$source_dir" rev-parse --git-dir >/dev/null
origin=$(git -C "$source_dir" remote get-url origin)
root=$(mktemp -d "${TMPDIR:-/tmp}/rmi-z-image-oracle.XXXXXX")
clone="$root/stable-diffusion.cpp"
script_dir=$(cd "$(dirname "$0")" && pwd -P)
pin=97d2990807fe6d558e395f8764198d7c7e7b411c

git clone --no-checkout "$origin" "$clone" >&2
git -C "$clone" fetch origin "$pin" >&2
git -C "$clone" checkout --detach "$pin" >&2
git -C "$clone" submodule update --init --recursive >&2
git -C "$clone" apply --check "$script_dir/stable-diffusion-z-image-trace.patch" >&2
git -C "$clone" apply "$script_dir/stable-diffusion-z-image-trace.patch" >&2
cmake -S "$clone" -B "$clone/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSD_METAL=OFF \
    -DGGML_METAL=OFF \
    -DGGML_ACCELERATE=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_CUDA=OFF \
    -DGGML_VULKAN=OFF \
    -DSD_BUILD_EXAMPLES=ON >&2
cmake --build "$clone/build" --target sd-cli --config Release --parallel "${RMI_BUILD_JOBS:-4}" >&2
bin="$clone/build/bin/sd-cli"
[[ -x "$bin" ]] || { echo "oracle build did not produce $bin" >&2; exit 1; }
printf '%s\n' "$bin"
