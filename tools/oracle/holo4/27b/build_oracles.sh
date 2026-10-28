#!/bin/sh
set -eu

# Build both precision references from the same pinned, disposable CPU tree.
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ "$#" -ne 2 ]; then
    echo "usage: $0 PINNED_LLAMA_CPP FRESH_WORK_DIR" >&2
    exit 2
fi
RMI_QWEN35_SCALAR_KQUANT=1 sh "$script_dir/../../qwen35/build_qwen35_oracle.sh" "$1" "$2"
work_dir=$(CDPATH= cd -- "$2" && pwd)
source_dir=$work_dir/llama.cpp
build_dir=$work_dir/build
cp "$build_dir/bin/llama-eval-callback" "$build_dir/bin/llama-eval-callback-scalar"
cmake_file=$source_dir/tools/mtmd/CMakeLists.txt
for probe in vision image; do
    cat >> "$cmake_file" <<EOF

add_executable(rmi-holo4-$probe "$script_dir/$probe.cpp")
target_link_libraries(rmi-holo4-$probe PRIVATE mtmd)
target_compile_features(rmi-holo4-$probe PRIVATE cxx_std_17)
EOF
done
cmake -S "$source_dir" -B "$build_dir"
cmake --build "$build_dir" --target rmi-holo4-image --parallel "${RMI_BUILD_JOBS:-4}"
cp "$build_dir/bin/rmi-holo4-image" "$build_dir/bin/rmi-holo4-image-scalar"
# Only the K-quant dot product changes; flags, softmax and graph instrumentation stay fixed.
git -C "$source_dir" apply -R "$script_dir/../../qwen35/qwen35-scalar-kquant.patch"
# Force relinking even when Make considers a same-second archive update unchanged.
rm "$build_dir/bin/llama-eval-callback" "$build_dir/bin/rmi-holo4-image"
cmake --build "$build_dir" --target llama-eval-callback rmi-holo4-image --parallel "${RMI_BUILD_JOBS:-4}"
test "$(git -C "$source_dir" rev-parse HEAD)" = b96806d96061049a5b574269b049bf6241d63d46
for backend in ACCELERATE BLAS METAL; do
    grep -q "^GGML_$backend:BOOL=OFF$" "$build_dir/CMakeCache.txt" || {
        echo "Oracle must disable GGML_$backend" >&2
        exit 1
    }
done
cmake --build "$build_dir" --target rmi-holo4-vision llama-tokenize --parallel "${RMI_BUILD_JOBS:-4}"
printf '%s\n' "$build_dir/bin/rmi-holo4-vision"
