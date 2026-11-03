#!/bin/sh
set -eu
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null

pin=11fe02151f79c41d0d4af7da708755d73b9c0da6
converter_pin=b96806d96061049a5b574269b049bf6241d63d46
if [ "$#" -ne 2 ]; then
    echo "usage: $0 LLAMA_CPP_CHECKOUT FRESH_WORK_DIR" >&2
    exit 2
fi
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source_dir=$(CDPATH= cd -- "$1" && pwd)
work_dir=$2
if [ ! -e "$source_dir/.git" ] || [ "$(git -C "$source_dir" rev-parse HEAD)" != "$pin" ]; then
    echo "llama.cpp must be a git checkout at $pin" >&2
    exit 1
fi
if [ -e "$work_dir" ]; then
    echo "refusing to overwrite $work_dir" >&2
    exit 1
fi
git clone --shared --no-checkout "$source_dir" "$work_dir/llama.cpp"
git -C "$work_dir/llama.cpp" checkout --detach "$pin"
work_dir=$(CDPATH= cd -- "$work_dir" && pwd)
clone_dir=$work_dir/llama.cpp

# Reuse the existing trace harness; no model formulas are patched.
git -C "$clone_dir" show "$converter_pin:examples/eval-callback/eval-callback.cpp" \
    > "$clone_dir/examples/eval-callback/eval-callback.cpp"
git -C "$clone_dir" apply "$script_dir/../qwen35/qwen35-llama-trace.patch"
git -C "$clone_dir" apply "$script_dir/../qwen35/qwen35-scalar-softmax.patch"
git -C "$clone_dir" diff --exit-code --quiet -- src
cat >> "$clone_dir/tools/mtmd/CMakeLists.txt" <<EOF

add_executable(rmi-ovisocr3-vision "$script_dir/../holo4/27b/vision.cpp")
target_link_libraries(rmi-ovisocr3-vision PRIVATE mtmd)
target_compile_features(rmi-ovisocr3-vision PRIVATE cxx_std_17)
EOF
cmake -S "$clone_dir" -B "$work_dir/build" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DGGML_METAL=OFF \
    -DGGML_CCACHE=OFF -DGGML_LLAMAFILE=OFF -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF -DGGML_CPU_REPACK=OFF -DGGML_CPU_ALL_VARIANTS=OFF \
    -DGGML_SSE42=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF \
    -DGGML_F16C=OFF -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF \
    -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF -DGGML_AVX_VNNI=OFF \
    -DGGML_AMX=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DCMAKE_CXX_FLAGS=-DRMI_QWEN35_SCALAR_SOFTMAX
cmake --build "$work_dir/build" --target llama-eval-callback rmi-ovisocr3-vision \
    --parallel "${RMI_BUILD_JOBS:-4}"
printf '%s\n' "$work_dir/build/bin/llama-eval-callback" "$work_dir/build/bin/rmi-ovisocr3-vision"
