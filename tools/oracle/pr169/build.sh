#!/bin/sh
set -eu

pin=11fe02151f79c41d0d4af7da708755d73b9c0da6
if [ "$#" != 2 ]; then
    echo "usage: $0 TEMPORARY_LLAMA_CHECKOUT BUILD_DIRECTORY" >&2
    exit 2
fi
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ "$(git -C "$1" rev-parse HEAD)" != "$pin" ]; then
    echo "llama.cpp must be pinned to $pin" >&2
    exit 1
fi
patch="$script_dir/llama-trace.patch"
if git -C "$1" apply --check "$patch" 2>/dev/null; then
    git -C "$1" apply "$patch"
elif ! git -C "$1" apply --reverse --check "$patch" 2>/dev/null; then
    echo "Oracle patch neither applies nor is already applied" >&2
    exit 1
fi
flags=
llamafile=ON
if [ "${RMI_ORACLE_SCALAR:-0}" = 1 ]; then
    flags="-U__ARM_NEON -U__ARM_NEON__ -ffp-contract=off"
    llamafile=OFF
fi
# CPU numerical reference; quantized activations and approximate ops are not bitwise checks.
cmake -S "$1" -B "$2" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_C_FLAGS="$flags" -DCMAKE_CXX_FLAGS="$flags" \
    -DGGML_LLAMAFILE="$llamafile" \
    -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF \
    -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=OFF \
    -DGGML_CPU_KLEIDIAI=OFF -DGGML_OPENMP=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_SERVER=OFF
cmake --build "$2" --target llama-eval-callback llama-mtmd-debug llama-tokenize \
    --parallel "${RMI_BUILD_JOBS:-4}"
