#!/bin/sh
set -eu

# Bit-exact parity oracle build for `general.architecture = "lfm2moe"`
# (LFM2.5-24B-A2B). Materialises the pinned llama.cpp tree, applies the
# checkpoint-label patch and builds `llama-eval-callback` with the scalar
# settings the parity contract requires.
#
#   tools/oracle/lfm2moe/build_lfm2moe_oracle.sh LLAMA_CPP_CHECKOUT [BUILD_DIR]
#
# MSVC notes (the reference machine for this model):
#   * `/fp:precise` is MSVC's default; it keeps the FP order and does not
#     contract `a*b+c` into FMA, which is what the Rust scalar side does.
#   * `GGML_NATIVE=OFF` + `GGML_LLAMAFILE=OFF` are the two flags that matter:
#     the first keeps ggml's runtime-dispatched AVX2 vec_dot kernels out of the
#     build, the second keeps tinyBLAS (which accumulates differently) out.
#   * `--parallel` needs a generator that supports it (Visual Studio / Ninja).

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

patch="$script_dir/lfm2moe-eval-callback.patch"
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
    -DCMAKE_C_FLAGS_RELEASE="/fp:precise" \
    -DCMAKE_CXX_FLAGS_RELEASE="/fp:precise" \
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
cmake --build "$build_dir" --config Release --target llama-eval-callback \
    --parallel "${RMI_BUILD_JOBS:-4}"

echo "$build_dir/bin/Release/llama-eval-callback"
