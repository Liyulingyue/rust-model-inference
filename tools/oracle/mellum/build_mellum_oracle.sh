#!/bin/sh
# Build the pinned Mellum bit-parity oracle.
#
# Replaces `examples/eval-callback/eval-callback.cpp` with the Mellum replay
# harness and builds it with the scalar flags the repo's parity contract
# requires: no FMA contraction, no NEON, no BLAS/Accelerate, one thread,
# F32 KV cache, flash attention off, `n_batch = n_ubatch = 1`.
#
# The `mellum` graph already labels every checkpoint we compare through
# `llm_graph_context::cb(...)` in `src/models/mellum.cpp`, so this patch only
# touches the example program — no model source is modified.
#
# usage: tools/oracle/mellum/build_mellum_oracle.sh LLAMA_CPP_CHECKOUT [BUILD_DIR]
set -eu

pin=8e2d31e0eb658d0382d3106a61369dff81f0c080
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 LLAMA_CPP_CHECKOUT [BUILD_DIR]" >&2
    exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
llama_dir=$(CDPATH= cd -- "$1" && pwd)
build_dir=${2:-"$llama_dir/build-rmi-mellum"}

# A tarball checkout has no git metadata, so verify by blob hash instead: the
# four files below pin the `mellum` model, the shared graph helpers it calls
# (build_attn / build_moe_ffn / softmax / rope / mul_mat) and the vocab loader.
# A git checkout is additionally checked against the commit id.
if git -C "$llama_dir" rev-parse HEAD >/dev/null 2>&1; then
    actual=$(git -C "$llama_dir" rev-parse HEAD)
    if [ "$actual" != "$pin" ]; then
        echo "note: checkout HEAD is $actual, not $pin; falling back to blob verification" >&2
    fi
fi

check_blob() {
    want=$1
    file=$2
    got=$(git hash-object "$llama_dir/$file")
    if [ "$got" != "$want" ]; then
        echo "$file: expected blob $want, got $got (checkout is not $pin)" >&2
        exit 1
    fi
}
check_blob 3ab79747af276a93631ebe2d052edb63277e8971 src/models/mellum.cpp
check_blob 1514b8aebe8c3b121c46c968060be3105e0e4d0b src/llama-graph.cpp
check_blob b3a0cf5d536f72f52a9bec6928ed50c8be89dbaa src/llama-model.cpp
check_blob dbadaf100b64827003e9c5c289f4bfd4f9aee173 src/llama-vocab.cpp

patch="$script_dir/mellum-eval-callback.patch"
if git -C "$llama_dir" apply --check "$patch" 2>/dev/null; then
    git -C "$llama_dir" apply "$patch"
elif git -C "$llama_dir" apply --reverse --check "$patch" 2>/dev/null; then
    :
else
    echo "oracle patch neither applies cleanly nor is already applied" >&2
    exit 1
fi

cmake -S "$llama_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-U__ARM_NEON -U__ARM_NEON__ -ffp-contract=off" \
    -DCMAKE_CXX_FLAGS="-U__ARM_NEON -U__ARM_NEON__ -ffp-contract=off" \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_ACCELERATE=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_CCACHE=OFF \
    -DGGML_CUDA=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DGGML_METAL=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_VULKAN=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TOOLS=OFF
cmake --build "$build_dir" --target llama-eval-callback --parallel "${RMI_BUILD_JOBS:-4}"

echo "$build_dir/bin/llama-eval-callback"
