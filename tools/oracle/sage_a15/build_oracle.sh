#!/bin/sh
set -eu
if [ "$#" -ne 2 ]; then
    echo "usage: $0 PINNED_LLAMA_CPP WORK_DIR" >&2
    exit 2
fi
scripts=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
sh "$scripts/qwen35/build_qwen35_oracle.sh" "$1" "$2"
work=$(CDPATH= cd -- "$2" && pwd)
export GIT_CEILING_DIRECTORIES="$work"
# macOS Make can miss patches applied in the same second as the first build.
sleep 1
git -C "$work/llama.cpp" apply "$scripts/qwen35/qwen35moe-ffn-dump.patch"
git -C "$work/llama.cpp" apply "$scripts/sage_a15/sequential-trace.patch"
# GGML_NATIVE=OFF leaves baseline NEON/SSE2 activation kernels enabled.
cmake -S "$work/llama.cpp" -B "$work/build" \
    -DCMAKE_C_FLAGS="-ffp-contract=off -U__ARM_NEON -U__SSE2__" \
    -DCMAKE_CXX_FLAGS="-DRMI_QWEN35_SCALAR_SOFTMAX -ffp-contract=off -U__ARM_NEON -U__SSE2__"
cmake --build "$work/build" --target llama-eval-callback llama-tokenize --parallel "${RMI_BUILD_JOBS:-4}"
printf '%s\n' "$work/build/bin/llama-eval-callback"
