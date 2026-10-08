#!/usr/bin/env bash
set -euo pipefail

# Builds the pinned Qwen-Image-2.1 parity oracle: stable-diffusion.cpp with a
# CPU-only configuration plus the trace patch and the standalone DiT harness.
#
# usage: build_oracle.sh
# prints the path of the built harness binary on stdout.

script_dir=$(cd "$(dirname "$0")" && pwd -P)
sd_commit=2f886889e6e8b78738d6b87f7191f6018557c551
ggml_commit=4bf5f6000653b7881d00963cd6ddb665ccd62a8d
root=${RMI_QI21_ORACLE_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/rmi-qi21-oracle.XXXXXX")}
mkdir -p "$root"
work="$root/src"

fetch() {
    # fetch REPO COMMIT DEST_TARBALL
    local repo=$1 commit=$2 dest=$3
    if [[ ! -s "$dest" ]]; then
        curl -sL --max-time 900 "https://codeload.github.com/${repo}/tar.gz/${commit}" -o "$dest"
    fi
}

extract() {
    # extract TARBALL INTO_DIR (python tarfile: the sandbox blocks tar-written sources)
    python3 - "$1" "$2" << 'PYEOF'
import sys, tarfile
with tarfile.open(sys.argv[1]) as archive:
    archive.extractall(sys.argv[2], filter="data")
PYEOF
}

if [[ ! -d "$work" ]]; then
    fetch leejet/stable-diffusion.cpp "$sd_commit" "$root/sd.tar.gz"
    fetch leejet/ggml "$ggml_commit" "$root/ggml.tar.gz"
    extract "$root/sd.tar.gz" "$root"
    extract "$root/ggml.tar.gz" "$root"
    mv "$root/stable-diffusion.cpp-$sd_commit" "$work"
    rmdir "$work/ggml"
    mv "$root/ggml-$ggml_commit" "$work/ggml"
    (cd "$work" && git apply --check "$script_dir/qwen-image-2-1-trace.patch")
    (cd "$work" && git apply "$script_dir/qwen-image-2-1-trace.patch")
fi

cp "$script_dir/components.cpp" "$work/examples/qwen_image_2_1_trace/components.cpp"
if ! rg -q 'qwen-image-2-1-components' "$work/examples/qwen_image_2_1_trace/CMakeLists.txt"; then
    cat >> "$work/examples/qwen_image_2_1_trace/CMakeLists.txt" <<'CMAKE'
add_executable(qwen-image-2-1-components components.cpp)
target_include_directories(qwen-image-2-1-components PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(qwen-image-2-1-components PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(qwen-image-2-1-components PUBLIC cxx_std_17)
if(APPLE)
    sd_set_macos_rpaths(qwen-image-2-1-components)
endif()
CMAKE
fi

# Disable vector reductions, architecture kernels, contraction, and fused trig
# together. RMI_SCALAR=1 uses the same scalar arithmetic contract.
flags='-ffp-contract=off -fno-builtin-sinf -fno-builtin-cosf -U__ARM_NEON -U__ARM_FEATURE_FP16_VECTOR_ARITHMETIC -U__ARM_FEATURE_SVE -U__ARM_FEATURE_FMA'
if "${CXX:-c++}" --version | rg -qi clang; then
    flags+=' -fno-vectorize -fno-slp-vectorize'
else
    flags+=' -fno-tree-vectorize'
fi
build="$work/build-scalar"
cmake -S "$work" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="$flags" \
    -DCMAKE_CXX_FLAGS="$flags" \
    -DSD_METAL=OFF \
    -DGGML_METAL=OFF \
    -DGGML_ACCELERATE=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_CCACHE=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_SSE42=OFF \
    -DGGML_AVX=OFF \
    -DGGML_AVX2=OFF \
    -DGGML_AVX512=OFF \
    -DGGML_AVX512_VBMI=OFF \
    -DGGML_AVX512_VNNI=OFF \
    -DGGML_FMA=OFF \
    -DGGML_F16C=OFF \
    -DGGML_CUDA=OFF \
    -DGGML_VULKAN=OFF \
    -DSD_BUILD_EXAMPLES=ON >&2
cmake --build "$build" --target qwen-image-2-1-oracle qwen-image-2-1-components --config Release --parallel "${RMI_BUILD_JOBS:-4}" >&2

bin="$build/bin/qwen-image-2-1-oracle"
[[ -x "$bin" ]] || { echo "oracle build did not produce $bin" >&2; exit 1; }
printf '%s\n' "$bin"
