# Task 10: Vulkan Q4_0 / Q4_1 / Q6_K inference

## Scope delivered

- Added Vulkan format dispatch for `Q4_0`, `Q4_1`, and `Q6_K`; Qwen3 binds each
  uploaded tensor using its GGML type and rejects unsupported or heterogeneous
  grouped weights before recording Vulkan work.
- Added Q4_0, Q4_1, Q6_K matvec shaders and a Q8_K activation shader for Q6_K.
- Added the Q4_1 activation-block auxiliary required by the CPU
  `forward_prepared` contract: `f16_round(sum(q8) * (amax(raw) / 127))`.
  It is generated once by the Q8_0 quantizer and reused by every Q4_1 output
  row.

## Investigation and TDD evidence

The initial plan assumed Q6_K consumed Q8_0 activations. The actual CPU path
uses Q8_K, so the implementation uses the same Q8_K block layout and records
`quantize_q8_k` before Q6_K matvec.

Before Task 10 implementation, the format gate was RED:

```text
unsupported Vulkan weight format q4_0,q4_1,q6_k
```

The command was:

```bash
cargo run --release --locked --features vulkan --example vk_ops_check -- \
  --formats q4_0,q4_1,q6_k
```

Q6_K initially reduced an entire 256-element block to one scalar. CPU instead
accumulates eight F32 lanes across Q8_K blocks and reduces only at the end.
The Vulkan Q6_K shader now preserves that order with `precise` temporaries.
The real-width synthetic result is:

```text
operator=q6_k max_abs=0.000e0 max_rel=0.000e0 first_bad=None
operator=quantize_q8_k bytes_exact=true max_scale_ulps=1
```

The Q4_1 min-term RED intentionally added an arena region and read it before
the quantizer wrote it. It failed as expected:

```text
Q4_1 input sum mismatch at 0: gpu=0 cpu=-4.2304688
```

The fixture also verifies that raw-scale and stored-F16-scale terms differ.
After the shader writes the raw-scale, F16-rounded sum, it is bit-exact and the
3072-element Q4_1 matvec is exact:

```text
operator=q4_1_input_sums exact=true
operator=q4_1 max_abs=0.000e0 max_rel=0.000e0 first_bad=None
```

Finally, Q4_0's shader had implicit multiply-add contraction while the CPU
scalar kernel is left-associated. Marking the block term and accumulator
`precise` resolved the remaining synthetic discrepancy:

```text
operator=q4_0 max_abs=0.000e0 max_rel=0.000e0 first_bad=None
```

Temporary per-layer snapshots used to identify this path were removed; no
diagnostic arena or trace plumbing is included in the final change.

## Final verification (Apple M3 Max)

```bash
bash scripts/vulkan-shaders.sh check

cargo run --release --locked --features vulkan --example vk_ops_check -- \
  --formats q4_0,q4_1,q6_k

cargo run --release --locked --features vulkan --example vk_model_check -- \
  qwen3 --model /Users/gouzi/Documents/git/rust-model-inference/models/Qwen3-0.6B/Qwen3-0.6B-Q4_0.gguf

cargo run --release --locked --features vulkan --example vk_model_check -- \
  qwen3 --model /Users/gouzi/Documents/git/rust-model-inference/models/Qwen3-0.6B-Q8_0/Qwen3-0.6B-Q8_0.gguf

git diff --check
```

All commands exited zero. Shader integrity rebuilt and byte-compared every
SPIR-V artifact against its GLSL source and manifest. The combined synthetic
gate passed Q4_0, Q4_1, and Q6_K; Q8_K activation bytes were exact and scale
error was at most one ULP.

The real Q4_0 model reported `prefill_logits max_abs=0.000e0`, generated 32
matching greedy tokens, and used 36 submissions (four prompt tokens plus 32
decode tokens). The Q8_0 regression likewise reported zero prefill error, 32
matching tokens, and 37 submissions (five prompt tokens plus 32 decode tokens).

The project emits pre-existing Rust warnings during these builds; no warning
was introduced by this task, and all focused gates above passed.
