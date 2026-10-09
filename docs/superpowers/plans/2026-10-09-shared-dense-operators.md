# Shared dense operator expression

## Scope and contract

Qwen3 and eligible standard Llama express a dense layer once, including tensor
connections: RMSNorm → grouped Q/K/V → QK normalization/RoPE → KV append →
attention → output projection → residual → RMSNorm → grouped gate/up →
SiLU(gate) × up → down → residual. CPU executes operators; Vulkan records them.
Model adapters provide weights, scratch buffers and model-specific attention,
RoPE, MoE, trace and deepstack behavior. GPU kernels remain backend implementations.

Preserve Qwen's per-row approximate SiLU and Llama's single-row approximate /
multi-row exact SiLU, prepared activation reuse, unequal CPU K/V dimensions,
all trace checkpoints, skipped vocabulary projection, and chunk-level commit.
No operation submits or reads GPU results. No new Auto admission rules.

Weight storage and projection semantics are independent. Models request a
numerical mode; the Vulkan operator owns shader selection. VAE retains rounded
F16 input and F32 accumulation. Dot contracts retain architecture eligibility.

## Implementation

1. Replace step tags with operand-bearing operators and a shared expression.
   Add operand/dataflow and failure tests. Reuse CPU norm/linear/add/SiLU kernels
   through one scratch view without allocations or new unsafe code.
2. Migrate Llama, Qwen3 and Vulkan adapters. Keep specialized attention/RoPE/KV
   operations and MoE explicit; move tracing and deepstack around shared calls.
   Give the existing SiLU shader an output operand, preserving old callers.
3. Separate storage from numerical mode in Vulkan bindings/cache selection and
   migrate VAE/AuK/YuE callers. Reject unsupported modes before writes/recording.
4. Run focused CPU, scalar/trace, device and state tests, shader validation and
   appropriate suite comparison with the merged head. Review the whole change,
   update validation report, commit/push and update Draft PR #166.

## Acceptance

- Tensor wiring and dense FFN formula exist in one model expression.
- Qwen and Llama use the same CPU primitive implementation.
- GPU recording consumes that same expression; no per-op host synchronization.
- Existing numerical and failure recovery tests retain their assertions.
- Storage cannot silently substitute a different numerical reduction.
- Report baseline failures and hardware coverage separately from passing checks.

## Delivery status

Implemented all four steps. Focused 50/50, scalar 40/40; physical-device suite
68 pass with one inherited AuK ARM failure. Full suites retain baseline failures.
The Qwen trace fixture now includes upstream raw Q/K checkpoints and preserves
B=1/64 binary equality. One independent whole-change review found no important
or critical issue. Evidence and proof boundaries are in
[the validation report](../../develop/UNIFIED_COMPUTE_VALIDATION.md).
