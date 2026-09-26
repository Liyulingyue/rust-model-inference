#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "$0")/.." && pwd)
shader_names=(
    q8_matmul
    q8_matmul_dp4a
    q8_matmul_grouped_dp4a
    quantize_q8_0
    quantize_q8_k
    q8_matmul_grouped
    q4_0_matmul
    q4_1_matmul
    q4_k_matmul
    q5_k_matmul
    q6_k_matmul
    f16_matmul
    bf16_matmul
    f32_matmul
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

# Compile `name` into `out`, keeping the already checked-in SPIR-V when the
# local glslangValidator lacks GL_EXT_integer_dot_product. The dot-product
# shaders are assembled with spirv-as, so the checked-in binary is
# authoritative for them and a rebuild must not fail the run.
compile_shader_or_keep() {
    local out="$2" log rebuilt
    log=$(mktemp)
    rebuilt="$out.rebuilt"
    if compile_shader "$1" "$rebuilt" >"$log" 2>&1; then
        mv "$rebuilt" "$out"
        rm -f "$log"
        return 0
    fi
    rm -f "$rebuilt"
    if grep -Fq "extension not supported: GL_EXT_integer_dot_product" "$log"; then
        echo "$1: rebuild skipped (compiler lacks GL_EXT_integer_dot_product); keeping checked-in SPIR-V" >&2
        rm -f "$log"
        return 0
    fi
    cat "$log" >&2
    rm -f "$log"
    return 1
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
            compile_shader_or_keep "$name" "$root_dir/shaders/bin/$name.spv"
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
            if ! compile_shader_or_keep "$name" "$rebuilt" 2>"$temp_dir/$name.log"; then
                cat "$temp_dir/$name.log" >&2
                exit 1
            fi
            # A skipped rebuild leaves no `rebuilt` file: the checked-in binary
            # is the one under test, so there is nothing to byte-compare.
            if [[ -f "$rebuilt" ]]; then
                spirv-val --target-env vulkan1.1 "$rebuilt"
                cmp "$rebuilt" "$checked_in"
            fi
        done
        ;;
    *)
        echo "usage: $0 [check|update]" >&2
        exit 2
        ;;
esac
