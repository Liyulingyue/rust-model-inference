#!/bin/sh
set -eu
pin=b96806d96061049a5b574269b049bf6241d63d46
test "$#" -eq 2 || { echo "usage: $0 LLAMA_CPP_CHECKOUT WORK_DIR" >&2; exit 2; }
source_dir=$(CDPATH= cd -- "$1" && pwd)
test "$(git -C "$source_dir" rev-parse HEAD)" = "$pin" || { echo "expected llama.cpp $pin" >&2; exit 1; }
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
clone_dir=$2/llama.cpp
build_dir=$2/build
git clone --shared --no-checkout "$source_dir" "$clone_dir"
git -C "$clone_dir" checkout --detach "$pin"
cp "$script_dir/oracle.cpp" "$clone_dir/examples/eval-callback/eval-callback.cpp"
python3 - "$clone_dir/src/models/bert.cpp" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
source = path.read_text()
marker = '        // input for next layer\n'
assert source.count(marker) == 1
source = source.replace(marker, '        cb(cur, "layer_out", il);\n' + marker)
for old, new in [("inp_embd", "rmi_embedding"), ("inp_norm", "rmi_embedding_norm"),
                 ("Qcur", "rmi_q"), ("Kcur", "rmi_k"), ("Vcur", "rmi_v"),
                 ("kqv_out", "rmi_attention"), ("ffn_inp", "rmi_ffn_input"),
                 ("layer_out", "rmi_layer_output")]:
    source = source.replace('"' + old + '"', '"' + new + '"')
path.write_text(source)
PY
flags='-U__ARM_NEON -U__ARM_NEON__ -ffp-contract=off -fno-vectorize -fno-slp-vectorize'
cmake -S "$clone_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="$flags" -DCMAKE_CXX_FLAGS="$flags" \
    -DBUILD_SHARED_LIBS=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF \
    -DGGML_CCACHE=OFF -DGGML_LLAMAFILE=OFF -DGGML_METAL=OFF -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF -DGGML_CPU_REPACK=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build "$build_dir" --target llama-eval-callback --parallel "${RMI_BUILD_JOBS:-4}"
