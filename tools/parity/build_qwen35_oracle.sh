#!/bin/sh
set -eu

pin=b96806d96061049a5b574269b049bf6241d63d46
if [ "$#" -ne 2 ]; then
    echo "usage: $0 LLAMA_CPP_CHECKOUT WORK_DIR" >&2
    exit 2
fi

source_dir=$(CDPATH= cd -- "$1" && pwd)
work_dir=$2
actual=$(git -C "$source_dir" rev-parse HEAD)
if [ "$actual" != "$pin" ]; then
    echo "llama.cpp must be $pin, got $actual" >&2
    exit 1
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
clone_dir=$work_dir/llama.cpp
build_dir=$work_dir/build
git clone --shared --no-checkout "$source_dir" "$clone_dir"
git -C "$clone_dir" checkout --detach "$pin"
git -C "$clone_dir" apply "$script_dir/qwen35-llama-trace.patch"
cmake -S "$clone_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_ACCELERATE=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_CCACHE=OFF \
    -DGGML_METAL=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_TESTS=OFF
cmake --build "$build_dir" --target llama-eval-callback --parallel "${RMI_BUILD_JOBS:-4}"
printf '%s\n' "$build_dir/bin/llama-eval-callback"
