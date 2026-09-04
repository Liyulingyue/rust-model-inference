#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "$0")/.." && pwd)
shader_names=(
    q8_matmul
    q8_matmul_dp4a
    quantize_q8_0
    quantize_q8_k
    q8_matmul_grouped
    q4_0_matmul
    q4_1_matmul
    q4_k_matmul
    q6_k_matmul
    f16_matmul
    bf16_matmul
    rms_norm
    qk_norm_rope
    kv_write
    attention_scores
    softmax
    attention_values
    qwen35_dense_prepare
    qwen35_attention
    qwen35_recurrent_conv
    qwen35_recurrent_ssm
    silu_mul
    add
)
manifest="$root_dir/shaders/manifest.sha256"

for tool in glslangValidator spirv-dis spirv-val; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "missing required tool: $tool" >&2
        exit 1
    fi
done

compile_shader() {
    glslangValidator -V --target-env vulkan1.1 \
        "$root_dir/shaders/glsl/$1.comp" -o "$2"
}

hash_files() {
    local files=()
    for name in "${shader_names[@]}"; do
        files+=("shaders/glsl/$name.comp" "shaders/bin/$name.spv")
    done
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${files[@]}"
    else
        shasum -a 256 "${files[@]}"
    fi
}

check_hashes() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -c shaders/manifest.sha256
    else
        shasum -a 256 -c shaders/manifest.sha256
    fi
}

check_workgroup_limit() {
    local shader="$1"
    spirv-dis "$shader" -o - | awk -v shader="$shader" '
        $1 == "OpExecutionMode" && $3 == "LocalSize" {
            found = 1
            invocations = $4 * $5 * $6
            if (invocations > 64) {
                printf "%s: workgroup %sx%sx%s requires %s invocations; baseline limit is 64\n", \
                    shader, $4, $5, $6, invocations > "/dev/stderr"
                failed = 1
            }
        }
        END {
            if (!found) {
                printf "%s: missing LocalSize execution mode\n", shader > "/dev/stderr"
                exit 1
            }
            if (failed) exit 1
        }
    '
}

case "${1:-check}" in
    update)
        for name in "${shader_names[@]}"; do
            compile_shader "$name" "$root_dir/shaders/bin/$name.spv"
        done
        (cd "$root_dir" && hash_files) >"$manifest"
        ;;
    check)
        (cd "$root_dir" && check_hashes)
        temp_dir=$(mktemp -d)
        trap 'rm -rf "$temp_dir"' EXIT
        for name in "${shader_names[@]}"; do
            checked_in="$root_dir/shaders/bin/$name.spv"
            rebuilt="$temp_dir/$name.spv"
            spirv-val --target-env vulkan1.1 "$checked_in"
            check_workgroup_limit "$checked_in"
            compile_shader "$name" "$rebuilt"
            spirv-val --target-env vulkan1.1 "$rebuilt"
            cmp "$rebuilt" "$checked_in"
        done
        ;;
    *)
        echo "usage: $0 [check|update]" >&2
        exit 2
        ;;
esac
