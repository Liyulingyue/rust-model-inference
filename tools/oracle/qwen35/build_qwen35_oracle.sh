#!/bin/sh
set -eu

# Build the oracle against no user configuration at all. Two reasons, and the
# second is the important one: under MSYS/Git-Bash git cannot even read the
# global config (the home path does not survive conversion), and more generally
# an oracle that can be changed by someone's core.autocrlf or filters.apply is
# not a reference. Harmless on Linux, where these are /dev/null anyway.
GIT_CONFIG_GLOBAL=${GIT_CONFIG_GLOBAL:-/dev/null}
GIT_CONFIG_SYSTEM=${GIT_CONFIG_SYSTEM:-/dev/null}
export GIT_CONFIG_GLOBAL GIT_CONFIG_SYSTEM

pin=b96806d96061049a5b574269b049bf6241d63d46
# Same three files the minicpmv oracle pins by blob, because a tarball checkout
# has no git metadata to ask. `qwen35.cpp` and `qwen35moe.cpp` carry the SSM and
# MoE formulas; `llama-arch.cpp` carries the arch registration, so a tree that
# matches all three cannot be a different architecture or a different pin.
pin_qwen35=0b9210981d3b3d714f26e3fd9675cf709def83ee
pin_qwen35moe=9bf4ea43287b45fb1ef5c25d8d6c756d0d12b280
pin_arch=7adb87411abd0f383ab5eee396e1f3f527bf94c6
if [ "$#" -ne 2 ]; then
    echo "usage: $0 LLAMA_CPP_CHECKOUT WORK_DIR" >&2
    exit 2
fi

source_dir=$(CDPATH= cd -- "$1" && pwd)
work_dir=$2
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# Two ways in. A git checkout is asked for its HEAD; a tarball export of the
# same commit has no git metadata, so it is asked for its content instead. Both
# must land on the same tree -- the point of the check is that the oracle is the
# pinned reference, not that git happened to be available.
#
# The `.git` test is not optional. `git -C <dir> rev-parse HEAD` walks up to the
# nearest enclosing repository, so a checkout kept inside another repo (here,
# references/llama.cpp inside the harness) silently reports the *parent's* HEAD
# and passes the comparison against whatever commit that happens to be.
if [ -e "$source_dir/.git" ] &&
        actual=$(git -C "$source_dir" rev-parse HEAD 2>/dev/null); then
    if [ "$actual" != "$pin" ]; then
        echo "llama.cpp must be $pin, got $actual" >&2
        exit 1
    fi
    clone_dir=$work_dir/llama.cpp
    build_dir=$work_dir/build
    git clone --shared --no-checkout "$source_dir" "$clone_dir"
    git -C "$clone_dir" checkout --detach "$pin"
else
    echo "no git metadata at $source_dir; verifying the pinned blobs instead" >&2
    # Run from inside the tree and use relative paths: under MSYS/Git-Bash a
    # leading /c/... is not something `git hash-object` resolves.
    for pair in \
        "src/models/qwen35.cpp:$pin_qwen35" \
        "src/models/qwen35moe.cpp:$pin_qwen35moe" \
        "src/llama-arch.cpp:$pin_arch"
    do
        path=${pair%%:*}
        want=${pair##*:}
        if [ ! -f "$source_dir/$path" ]; then
            echo "missing $path in $source_dir" >&2
            exit 1
        fi
        # `hash-object` needs no configuration, and under MSYS git fails to read
        # the user's global config at all (the home path does not survive the
        # conversion), so point it at /dev/null rather than let it fail there.
        got=$(cd -- "$source_dir" && GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null \
                git hash-object "$path") || {
            echo "could not hash $path" >&2; exit 1; }
        if [ "$got" != "$want" ]; then
            echo "$path is $got, expected $want ($pin)" >&2
            exit 1
        fi
        echo "ok $path $got" >&2
    done
    clone_dir=$work_dir/llama.cpp
    build_dir=$work_dir/build
    rm -rf "$clone_dir"
    mkdir -p "$(dirname -- "$clone_dir")"
    cp -R "$source_dir" "$clone_dir"
fi

# A tarball nested inside a checkout must not resolve the parent's .git and
# silently skip every patch outside that repository-relative path.
GIT_CEILING_DIRECTORIES=$(CDPATH= cd -- "$(dirname -- "$clone_dir")" && pwd)
export GIT_CEILING_DIRECTORIES
git -C "$clone_dir" apply "$script_dir/qwen35-llama-trace.patch"
# Match Rust's precise softmax without changing the production implementation.
git -C "$clone_dir" apply "$script_dir/qwen35-scalar-softmax.patch"
set --
if [ "${RMI_QWEN35_SCALAR_KQUANT:-0}" = 1 ]; then
    # Match the existing ARM64 Rust Q4_K/Q5_K/Q6_K scalar dot products.
    git -C "$clone_dir" apply "$script_dir/qwen35-scalar-kquant.patch"
    set -- -DGGML_CPU_REPACK=OFF -DCMAKE_C_FLAGS=-ffp-contract=off
fi
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
    -DGGML_SSE42=OFF \
    -DGGML_AVX=OFF \
    -DGGML_AVX2=OFF \
    -DGGML_FMA=OFF \
    -DGGML_F16C=OFF \
    -DGGML_AVX512=OFF \
    -DGGML_AVX512_VBMI=OFF \
    -DGGML_AVX512_VNNI=OFF \
    -DGGML_AVX512_BF16=OFF \
    -DGGML_AVX_VNNI=OFF \
    -DGGML_AMX=OFF \
    -DGGML_CPU_ALL_VARIANTS=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DCMAKE_CXX_FLAGS=-DRMI_QWEN35_SCALAR_SOFTMAX "$@"
# The ISA block above is what makes this a scalar oracle on x86. `GGML_NATIVE=OFF`
# only stops cmake passing -march=native; ggml's own defaults still compile the
# `#if defined(__AVX2__)` branches and, with GGML_CPU_ALL_VARIANTS on, build a
# runtime-selected AVX2/FMA copy of every kernel. That drifts roughly 2 ULP per
# dot product, which is enough to move a router argmax -- the thing this oracle
# exists to settle. See tools/oracle/shared/build_llama_oracle.sh for the same
# flags and the longer argument.
# SSE4.2 also defines __SSE3__, enabling GGML_SIMD float reductions without AVX.
cmake --build "$build_dir" --target llama-eval-callback --parallel "${RMI_BUILD_JOBS:-4}"
printf '%s\n' "$build_dir/bin/llama-eval-callback"
