#!/bin/sh
# Build the upstream llama.cpp reference used to check the
# `nemotron_h_moe` forward pass.
#
# The source tree is read-only: only the example harness is replaced.
#
# On x86 the build is scalar-only (no AVX2 / FMA auto-vectorization) so
# the reference matches this repository's scalar path bit for bit.
#
# On aarch64 that is not enough: NEON is part of the base ISA and cannot
# be switched off, so ggml always takes its ARM kernels. The repository
# picks its kernels at *runtime* (it detects `dotprod` and uses the native
# `vdotq_s32`), while ggml picks them at *compile* time. If the two run
# different instructions the comparison is meaningless: ggml's own
# fallback for `ggml_vdotq_s32` produces "the same total sum as native
# vdotq_s32 but with different per-lane grouping", and for the quantized
# dot products those lanes are scaled and accumulated separately in F32,
# so the grouping decides the rounding.
#
# So on aarch64 the build enables `dotprod` explicitly -- the same
# extension the repository detects at runtime -- and leaves `i8mm` off so
# that ggml's Q4_K kernel stays on the NEON path the repository mirrors.
set -eu
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 LLAMA_CPP_SOURCE [WORK_DIR]" >&2
    exit 2
fi
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
work_dir=${2:-"$script_dir/../../../target/nemotron3-nano-oracle"}
mkdir -p "$work_dir"
work_dir=$(CDPATH= cd -- "$work_dir" && pwd)
source_dir="$work_dir/source"
if [ ! -d "$source_dir" ]; then
    mkdir -p "$source_dir"
    tar xzf "$1" -C "$source_dir" --strip-components=1
fi
if [ -n "$(git -C "$source_dir" status --porcelain --untracked-files=normal -- src ggml include common 2>/dev/null || true)" ]; then
    echo "refusing a modified Oracle graph or backend" >&2
    exit 1
fi
cp "$script_dir/oracle.cpp" "$source_dir/examples/eval-callback/eval-callback.cpp"
arch_flags=""
if [ "$(uname -m)" = "aarch64" ]; then
    arch_flags="-march=armv8.2-a+dotprod"
fi
cmake -S "$source_dir" -B "$work_dir/build" \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_C_FLAGS=-ffp-contract=off -fno-builtin-sinf -fno-builtin-cosf $arch_flags" \
    "-DCMAKE_CXX_FLAGS=-ffp-contract=off -fno-builtin-sinf -fno-builtin-cosf $arch_flags" \
    -DGGML_NATIVE=OFF \
    -DGGML_SSE42=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF \
    -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_F32C=OFF \
    -DBUILD_SHARED_LIBS=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF \
    -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DGGML_CCACHE=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF
cmake --build "$work_dir/build" --target llama-eval-callback --parallel "${RMI_BUILD_JOBS:-8}"
echo "$work_dir/build/bin/llama-eval-callback"
