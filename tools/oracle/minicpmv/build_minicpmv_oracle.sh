#!/bin/sh
# Build the pinned MiniCPM-V 4.7 oracle.
#
# The repo's own `references/llama.cpp` predates MiniCPM-V 4.7: its
# `tools/mtmd/clip-impl.h` registers `minicpmv4_6` but not `minicpmv4_7`, so it
# cannot load this mmproj at all. The oracle is therefore the merge commit of
# upstream PR #29416 `[Model] Support MiniCPM-V 4.7`.
#
# A tarball checkout has no git metadata, so the pin is verified by blob hash
# instead of by commit id.
#
# usage: tools/oracle/minicpmv/build_minicpmv_oracle.sh [BUILD_PARENT]
set -eu

pin=69f201a2051ea9b9e9b50c3cc56afd9e2ae64414
build_parent=${1:-/tmp/opencode/oracle}

# Files that carry the minicpmv4_7 contract: the shared 4.6/4.7 graph, the
# CANVAS tokenizer/position logic, and the projector + hparams tables.
expect_minicpmv_cpp=0e0fc0381148d1e62ce35fdc55107e89211730a8
expect_mtmd_cpp=2c2b069361d5df81e35db59bfbe63d51aa011f38
expect_clip_cpp=fccb00c3aa7002bbcdbc6e988bb9a747bc7de864

blob() { git hash-object "$1" 2>/dev/null || printf '%s\n' "$1"; }

mkdir -p "$build_parent"
src="$build_parent/llama.cpp-$pin"
if [ ! -d "$src" ]; then
    archive="$build_parent/llama-$pin.tar.gz"
    [ -f "$archive" ] || curl -sSL -o "$archive" \
        "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$pin"
    tar xzf "$archive" -C "$build_parent"
fi

check() {
    got=$(blob "$src/$1")
    if [ "$got" != "$2" ]; then
        echo "$1: expected blob $2, got $got" >&2
        exit 1
    fi
}
check tools/mtmd/models/minicpmv.cpp "$expect_minicpmv_cpp"
check tools/mtmd/mtmd.cpp           "$expect_mtmd_cpp"
check tools/mtmd/clip.cpp           "$expect_clip_cpp"

# The whole point of the pin: this projector type only exists from here on.
grep -q 'PROJECTOR_TYPE_MINICPMV4_7, *"minicpmv4_7"' \
    "$src/tools/mtmd/clip-impl.h" || {
    echo "clip-impl.h does not register minicpmv4_7; wrong checkout" >&2
    exit 1
}

build="$src/build-rmi-minicpmv"
cmake -B "$build" -S "$src" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF
cmake --build "$build" --target llama-mtmd-cli -j "$(nproc 2>/dev/null || echo 8)"
echo "oracle ready: $build/bin/llama-mtmd-cli"
