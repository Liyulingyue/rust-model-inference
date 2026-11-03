#!/bin/sh
set -eu

pin=749f688fcaa4c472ec034b08cb8a907c45cfaa02
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 LLAMA_CPP_CHECKOUT [BUILD_DIR]" >&2
    exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
llama_dir=$(CDPATH= cd -- "$1" && pwd)
build_dir=${2:-"$llama_dir/build-rmi-parity"}
actual=$(git -C "$llama_dir" rev-parse HEAD)
if [ "$actual" != "$pin" ]; then
    echo "llama.cpp must be pinned to $pin, got $actual" >&2
    exit 1
fi

patch="$script_dir/llama-scalar-trace.patch"
if git -C "$llama_dir" apply --check "$patch" 2>/dev/null; then
    git -C "$llama_dir" apply "$patch"
elif git -C "$llama_dir" apply --reverse --check "$patch" 2>/dev/null; then
    :
else
    echo "oracle patch neither applies cleanly nor is already applied" >&2
    exit 1
fi

# `GGML_NATIVE=OFF` on its own is NOT enough to get a scalar oracle.
#
# It only stops cmake from passing -march=native. It does not change ggml's own
# defaults, so at this pin:
#   * GGML_CPU_ALL_VARIANTS stays ON, which compiles a separate AVX2 / AVX512 /
#     FMA copy of every kernel and picks one at runtime, each with its own ISA
#     flags regardless of the baseline;
#   * GGML_AVX2 / GGML_FMA / GGML_F16C / GGML_AVX512* stay ON, so ggml's
#     `#if defined(__AVX2__)` branches in arch/x86/quants.c compile in and Q4_0
#     dots accumulate through `_mm256_fmadd_ps`.
#
# A "scalar" oracle built that way differs from the reference implementation by
# roughly 2 ULP per dot product, which is enough to move a greedy argmax. Every
# arch that claims bit-exact parity through this script has to turn these off
# explicitly. -ffp-contract=off additionally stops the compiler from fusing
# multiply-add in the scalar paths.
cmake -S "$llama_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-U__ARM_NEON -U__ARM_NEON__ -ffp-contract=off" \
    -DCMAKE_CXX_FLAGS="-U__ARM_NEON -U__ARM_NEON__ -ffp-contract=off" \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_ACCELERATE=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_CCACHE=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DGGML_METAL=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
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
    -DLLAMA_BUILD_TESTS=OFF
cmake --build "$build_dir" --target llama-eval-callback --parallel "${RMI_BUILD_JOBS:-4}"

echo "$build_dir/bin/llama-eval-callback"
