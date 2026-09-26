#!/bin/sh
set -eu

pin=b96806d96061049a5b574269b049bf6241d63d46
if [ "$#" != 1 ]; then
    echo "usage: $0 LLAMA_CPP_CHECKOUT" >&2
    exit 2
fi
llama_dir=$(CDPATH= cd -- "$1" && pwd)
if [ "$(git -C "$llama_dir" rev-parse HEAD)" != "$pin" ]; then
    echo "llama.cpp must be pinned to $pin" >&2
    exit 1
fi
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/nanbeige-oracle.XXXXXX")
git -C "$llama_dir" archive "$pin" | tar -x -C "$work"
printf '%s\n' "$pin" > "$work/ORACLE_COMMIT"

# Reuse the shared trace writer; specialize only Nanbeige's checkpoint names.
python3 - "$script_dir/../shared/llama-scalar-trace.patch" "$work" <<'PY'
import sys
from pathlib import Path

patch, work = Path(sys.argv[1]), Path(sys.argv[2])
section = patch.read_text().split('diff --git ')[1]
body, in_hunk = [], False
for line in section.splitlines():
    if line.startswith('@@'):
        in_hunk = True
    elif in_hunk and line.startswith(('+', ' ')):
        body.append(line[1:])
source = '\n'.join(body) + '\n'
source = source.replace('"q_norm-", "k_norm-",',
    '"Qcur-", "Kcur-", "Vcur-", "attn_out-", "ffn_inp-", "l_out-", "loop_norm-",')
source = source.replace('get("v_proj" + suffix)', 'get("Vcur" + suffix)')
source = source.replace('get("q_norm" + suffix).ne[0]', 'q_proj.ne[0]')
start = source.index('            write_tensor("q_norm"')
end = source.index('            write_tensor("attn_values"', start)
source = source[:start] + '''            write_tensor("q_rope", layer, step, {n_embd_q}, get("Qcur" + suffix).values);
            write_tensor("k_rope", layer, step, {n_embd_gqa}, get("Kcur" + suffix).values);
''' + source[end:]
start = source.index('            write_tensor("ffn_gate"')
end = source.index('            write_tensor("ffn_silu_gate"', start)
source = source[:start] + source[end:]
for old, new in [('attn_proj', 'attn_out'), ('post_attn_residual', 'ffn_inp'), ('post_ffn_residual', 'l_out')]:
    source = source.replace(f'get("{old}" + suffix)', f'get("{new}" + suffix)')
source = source.replace('        }\n        write_tensor("result_norm"',
    '            if (tensors_.count("loop_norm" + suffix)) { write_tensor("loop_norm", layer, step, {n_embd}, get("loop_norm" + suffix).values); }\n        }\n        write_tensor("result_norm"')
source = source.replace('index < max_tokens;', 'index < max_tokens - 1;')
source = source.replace('    trace.write_tokens("prompt_ids", prompt);',
    '    trace.write_tokens("prompt_ids", prompt);\n    if (std::getenv("RMI_TOKENIZE_ONLY")) { return true; }')
for kind in ('k', 'v'):
    source = source.replace(f'params.cache_type_{kind} = GGML_TYPE_F32;',
        f'params.cache_type_{kind} = std::getenv("RMI_NANBEIGE_F16") ? GGML_TYPE_F16 : GGML_TYPE_F32;')
(work / 'examples/eval-callback/eval-callback.cpp').write_text(source)
model = work / 'src/models/nanbeige.cpp'
source = model.read_text()
needle = '                    n_embd_head, n_head, n_head_kv, il);'
assert source.count(needle) == 1
source = source.replace(needle, needle + '\n            cb(Qcur, "q_proj", il);\n            cb(Kcur, "k_proj", il);')
source = source.replace('cb(cur, "ffn_out", il);', 'cb(cur, "ffn_down", il);', 1)
model.write_text(source)
PY

cpu_flags="-U__ARM_NEON -U__ARM_NEON__ -ffp-contract=off"
if [ "${RMI_ORACLE_SIMD:-0}" = 1 ]; then
    cpu_flags=""
fi
cmake -S "$work" -B "$work/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="$cpu_flags" \
    -DCMAKE_CXX_FLAGS="$cpu_flags" \
    -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_ACCELERATE=OFF \
    -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF -DGGML_CCACHE=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build "$work/build" --target llama-eval-callback --parallel "${RMI_BUILD_JOBS:-4}"
echo "$work/build/bin/llama-eval-callback"
