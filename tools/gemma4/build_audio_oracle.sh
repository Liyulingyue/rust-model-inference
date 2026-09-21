#!/bin/sh
set -eu

pin=b96806d96061049a5b574269b049bf6241d63d46
if [ "$#" -ne 1 ]; then
    echo "usage: $0 LLAMA_CPP_CHECKOUT" >&2
    exit 2
fi

source_dir=$(CDPATH= cd -- "$1" && pwd)
actual=$(git -C "$source_dir" rev-parse HEAD)
if [ "$actual" != "$pin" ]; then
    echo "llama.cpp must be $pin, got $actual" >&2
    exit 1
fi
if [ -n "$(git -C "$source_dir" status --porcelain)" ]; then
    echo "llama.cpp checkout must be clean: $source_dir" >&2
    exit 1
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
build_root=$(mktemp -d "${TMPDIR:-/tmp}/gemma4-audio-oracle.XXXXXX")
clone_dir=$build_root/llama.cpp
build_dir=$build_root/build
git clone --shared --no-checkout "$source_dir" "$clone_dir" >&2
git -C "$clone_dir" checkout --detach "$pin" >&2
git -C "$clone_dir" apply "$script_dir/gemma4ua-trace.patch" >&2
cmake -S "$clone_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_ACCELERATE=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_CCACHE=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DGGML_METAL=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_OPENSSL=OFF >&2
cmake --build "$build_dir" --target llama-mtmd-cli --parallel "${RMI_BUILD_JOBS:-4}" >&2

binary=$build_dir/bin/llama-mtmd-cli
if [ ! -x "$binary" ]; then
    echo "Oracle build did not produce $binary" >&2
    exit 1
fi
printf '%s\n' "$binary"
