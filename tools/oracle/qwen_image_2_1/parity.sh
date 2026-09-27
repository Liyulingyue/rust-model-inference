#!/usr/bin/env bash
set -euo pipefail

# Runs one Qwen-Image-2.1 forward on both sides of the parity check and prints
# the trace paths for tests/qwen_image_2_1_reference.rs.
#
# usage: parity.sh [RUST_BIN]
# env:   QWEN_IMAGE_2_1_DIT (default: models/Qwen-Image-2.1-GGUF/qwen-image-2.1-Q8_0.gguf)

script_dir=$(cd "$(dirname "$0")" && pwd -P)
repo=$(cd "$script_dir/../../.." && pwd -P)
dit=${QWEN_IMAGE_2_1_DIT:-$repo/models/Qwen-Image-2.1-GGUF/qwen-image-2.1-Q8_0.gguf}
rust_bin=${1:-$repo/target/release-fast/rust-model-inference}
threads=${RMI_QI21_THREADS:-1}

if [[ $# -eq 0 ]]; then
    (cd "$repo" && cargo build --profile release-fast --features parity-trace --bin rust-model-inference) >&2
fi
[[ -x "$rust_bin" ]] || { echo "rust binary not executable: $rust_bin" >&2; exit 1; }
[[ -f "$dit" ]] || { echo "missing dit gguf: $dit" >&2; exit 1; }

oracle_bin=$("$script_dir/build_oracle.sh")

trace_root=$(mktemp -d "${TMPDIR:-/tmp}/rmi-qi21-parity.XXXXXX")
oracle_trace="$trace_root/oracle.jsonl"
rust_trace="$trace_root/rust.jsonl"

QWEN_IMAGE_2_1_ORACLE_TRACE="$oracle_trace" "$oracle_bin" \
    --diffusion-model "$dit" --threads "$threads" --latent-w 16 --latent-h 16 \
    --context-len 128 --timestep 500 > "$trace_root/oracle.log" 2>&1

trace_filter='qwen.pe,qwen.time_embed,qwen.modulation,qwen.txt_in,qwen.joint,qwen.joint_final,qwen.scale,qwen.norm_out,qwen.out,qwen.output'
for layer in {0..31}; do
    trace_filter+=",qwen.block.${layer}"
done
RMI_PARITY_FILTER="$trace_filter" RMI_PARITY_TRACE="$rust_trace" "$rust_bin" \
    --model "$dit" --out "$trace_root/rust.bin" --threads "$threads" \
    > "$trace_root/rust.log" 2>&1
[[ -s "$rust_trace" ]] || { echo "Rust binary produced no parity trace; build it with --features parity-trace" >&2; exit 1; }

echo "QWEN_IMAGE_2_1_ORACLE_TRACE=$oracle_trace"
echo "QWEN_IMAGE_2_1_RUST_TRACE=$rust_trace"
echo "then: cargo test --profile release-fast --test qwen_image_2_1_reference -- --ignored"
