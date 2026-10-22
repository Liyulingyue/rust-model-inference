#!/bin/sh
# Clone the read-only reference before replacing its example harness.
set -eu
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 LLAMA_CPP_SOURCE [WORK_DIR]" >&2
    exit 2
fi
pin=98a74c97ea09ac72f6ceda1559808fc0a2f447ce
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
work_dir=${2:-"$script_dir/../../../target/dory-oracle"}
mkdir -p "$work_dir"
work_dir=$(CDPATH= cd -- "$work_dir" && pwd)
source_dir="$work_dir/source"
if [ ! -d "$source_dir/.git" ]; then
    git clone --quiet --no-checkout "$1" "$source_dir"
    git -C "$source_dir" checkout --quiet --detach "$pin"
fi
test "$(git -C "$source_dir" rev-parse HEAD)" = "$pin"
if [ -n "$(git -C "$source_dir" status --porcelain --untracked-files=normal -- src ggml include common)" ]; then
    echo "refusing a modified Oracle graph or backend" >&2
    exit 1
fi
cp "$script_dir/oracle.cpp" "$source_dir/examples/eval-callback/eval-callback.cpp"
cmake -S "$source_dir" -B "$work_dir/build" \
    -DCMAKE_BUILD_TYPE=Release \
    '-DCMAKE_C_FLAGS=-U__ARM_NEON -U__ARM_NEON__ -ffp-contract=off -fno-builtin-sinf -fno-builtin-cosf' \
    '-DCMAKE_CXX_FLAGS=-U__ARM_NEON -U__ARM_NEON__ -ffp-contract=off -fno-builtin-sinf -fno-builtin-cosf' \
    -DGGML_NATIVE=OFF \
    -DGGML_SSE42=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF \
    -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF \
    -DBUILD_SHARED_LIBS=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF \
    -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF -DGGML_CCACHE=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF
cmake --build "$work_dir/build" --target llama-eval-callback --parallel "${RMI_BUILD_JOBS:-4}"
echo "$work_dir/build/bin/llama-eval-callback"
