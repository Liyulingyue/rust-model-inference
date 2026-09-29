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

cmake -S "$work" -B "$work/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-fno-vectorize -fno-slp-vectorize" \
    -DCMAKE_CXX_FLAGS="-fno-vectorize -fno-slp-vectorize" \
    -DSD_METAL=OFF \
    -DGGML_METAL=OFF \
    -DGGML_ACCELERATE=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_CUDA=OFF \
    -DGGML_VULKAN=OFF \
    -DSD_BUILD_EXAMPLES=ON >&2
cmake --build "$work/build" --target qwen-image-2-1-oracle --config Release --parallel "${RMI_BUILD_JOBS:-4}" >&2

bin="$work/build/bin/qwen-image-2-1-oracle"
[[ -x "$bin" ]] || { echo "oracle build did not produce $bin" >&2; exit 1; }
printf '%s\n' "$bin"
