#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 LLAMA_CPP_CHECKOUT" >&2
    exit 2
fi

oracle_dir=$(cd "$1" 2>/dev/null && pwd -P) || {
    echo "not a directory: $1" >&2
    exit 2
}
if ! git -C "$oracle_dir" rev-parse --git-dir >/dev/null 2>&1; then
    echo "not a git checkout: $oracle_dir" >&2
    exit 2
fi
pinned=201e50cc2076a20adc460c41598593c7cd7b0813
script_dir=$(cd "$(dirname "$0")" && pwd -P)
patch="$script_dir/llama-qwen3-tts-trace.patch"
origin=$(git -C "$oracle_dir" remote get-url origin)
build_root=$(mktemp -d "${TMPDIR:-/tmp}/qwen3-tts-oracle.XXXXXX")
build_dir="$build_root/llama.cpp"

git clone --no-checkout "$origin" "$build_dir" >&2
git -C "$build_dir" fetch origin "$pinned" >&2
git -C "$build_dir" checkout --detach "$pinned" >&2
git -C "$build_dir" apply --check "$patch" >&2
git -C "$build_dir" apply "$patch" >&2
cmake -S "$build_dir" -B "$build_dir/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF >&2
cmake --build "$build_dir/build" --target llama-tts -j 4 >&2

binary="$build_dir/build/bin/llama-tts"
if [[ ! -x "$binary" ]]; then
    echo "oracle build did not produce $binary" >&2
    exit 1
fi
printf '%s\n' "$binary"
