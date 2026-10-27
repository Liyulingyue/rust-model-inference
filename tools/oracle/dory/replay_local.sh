#!/bin/sh
# Replay the recorded Dory CPU parity cases on this host.
# Usage: sh tools/oracle/dory/replay_local.sh [case ...]
set -eu

cd "$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"

MODEL="$PWD/models/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"
ORACLE="$PWD/target/dory-oracle-strict/build/bin/llama-eval-callback"
export RMI_DORY_Q5_MODEL="$MODEL"
export RMI_DORY_THREADS=4
export RMI_DORY_N_PREDICT=8
PY="$PWD/.venv/bin/python"

run_case() {
    name=$1; kv=$2; scalar=$3; swa=$4; shift 4
    case_dir="$PWD/target/dory-parity/$name"
    rm -rf "$case_dir"
    mkdir -p "$case_dir"
    printf '%s' "$1" > "$case_dir/prompt.txt"
    export RMI_DORY_PROMPT_FILE="$case_dir/prompt.txt"
    export RMI_DORY_KV="$kv"
    if [ "$swa" != "0" ]; then export RMI_DORY_TEST_SWA="$swa"; else unset RMI_DORY_TEST_SWA || true; fi
    echo "=== $name (kv=$kv scalar=$scalar swa=$swa) ==="
    RMI_PARITY_TRACE="$case_dir/oracle.jsonl" \
        "$ORACLE" "$RMI_DORY_Q5_MODEL" "$RMI_DORY_PROMPT_FILE" "$RMI_DORY_N_PREDICT" \
        > "$case_dir/oracle.log" 2>&1
    if [ "$scalar" = "1" ]; then export RMI_SCALAR=1; else unset RMI_SCALAR || true; fi
    RMI_PARITY_TRACE="$case_dir/rust.jsonl" \
        cargo test --profile release-fast --features parity-trace --test dory_q5_k_m \
        dory_q5_reference_trace -- --ignored --exact --nocapture \
        > "$case_dir/rust.log" 2>&1
    "$PY" tools/oracle/dory/check.py "$case_dir/oracle.jsonl" "$case_dir/rust.jsonl" \
        --report "$case_dir/report.json" > "$case_dir/check.log" 2>&1
    "$PY" - "$case_dir/report.json" "$name" <<'PYEOF'
import json, sys
report, name = sys.argv[1], sys.argv[2]
d = json.load(open(report))
w = d.get("worst_relative_l2", {})
print(f"{name}: passed={d['passed']} checked={d['checked']} "
      f"relL2={w.get('relative_l2')} scaledAbs={d.get('worst_max_abs',{}).get('scaled_max_abs')} "
      f"failures={len(d.get('failures',[]))}")
sys.exit(0 if d["passed"] else 1)
PYEOF
}

run_case default-hello-f32   f32 0 0 'Hello'
run_case default-hello-f16   f16 0 0 'Hello'
run_case scalar-hello-f32    f32 1 0 'Hello'
run_case scalar-hello-f16    f16 1 0 'Hello'
run_case default-math-f32    f32 0 0 'What is 2+2?'
run_case default-math-f16    f16 0 0 'What is 2+2?'
run_case default-unicode-f32 f32 0 0 '你好
café	🙂 <s> [INS'
run_case default-window2-f32  f32 0 2 'Hello'
run_case default-window2-f16  f16 0 2 'Hello'
