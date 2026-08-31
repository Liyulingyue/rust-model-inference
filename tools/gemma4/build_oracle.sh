#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 LLAMA_CPP_CHECKOUT" >&2
    exit 2
fi

source_dir=$(cd "$1" 2>/dev/null && pwd -P) || {
    echo "not a directory: $1" >&2
    exit 2
}
if ! git -C "$source_dir" rev-parse --git-dir >/dev/null 2>&1; then
    echo "not a git checkout: $source_dir" >&2
    exit 2
fi

pin=3173a56471c1753650cd806694145ffd6dcace67
actual=$(git -C "$source_dir" rev-parse HEAD)
if [[ "$actual" != "$pin" ]]; then
    echo "llama.cpp must be pinned to $pin, got $actual" >&2
    exit 1
fi

script_dir=$(cd "$(dirname "$0")" && pwd -P)
patch="$script_dir/llama-gemma4-trace.patch"
build_root=$(mktemp -d "${TMPDIR:-/tmp}/gemma4-oracle.XXXXXX")
build_dir="$build_root/llama.cpp"

git clone --no-hardlinks "$source_dir" "$build_dir" >&2
git -C "$build_dir" checkout --detach "$pin" >&2
git -C "$build_dir" apply --check "$patch" >&2
git -C "$build_dir" apply "$patch" >&2
cmake -S "$build_dir" -B "$build_dir/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF >&2
cmake --build "$build_dir/build" --target llama-gemma4-trace --parallel "${RMI_BUILD_JOBS:-4}" >&2

binary="$build_dir/build/bin/llama-gemma4-trace"
if [[ ! -x "$binary" ]]; then
    echo "oracle build did not produce $binary" >&2
    exit 1
fi
printf '%s\n' "$binary"
