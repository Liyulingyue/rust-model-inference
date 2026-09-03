# Vulkan Inference Completion Design

Date: 2026-09-04

## Goal

Turn the existing experimental Q8_0 matmul offload into an explicit, testable
Vulkan inference path. The work is complete only when CLI and server entry
points enable the same backend, supported devices are selected by required
capabilities, checked-in shaders have one authoritative source, and supported
models can keep a full token's activations and KV state on the GPU with one
submission boundary.

The first end-to-end target is Qwen3-0.6B-Q8_0 on the local Apple M3 Max via
MoltenVK. Format coverage then expands using the models already present under
`/Users/gouzi/Documents/git/rust-model-inference/models`.

## Current State

- `--features vulkan --gpu` routes Q8_0 matmul through `VulkanContext` in the
  CLI, but the server parses `--gpu` without enabling the backend.
- macOS Loader discovery and Vulkan portability enumeration are implemented in
  `src/vulkan.rs`; the local integration check finds `Apple M3 Max` without
  Vulkan-related environment variables.
- Every matmul uploads the quantized activation, submits one command buffer,
  waits for one fence, and copies the output back to CPU memory. Model-level
  norm, RoPE, attention, activation, residual, and KV operations remain on CPU.
- `shaders/src/q8_matmul.comp` and `shaders/glsl/q8_matmul.comp` are conflicting
  implementations. The embedded `shaders/bin/q8_matmul.spv` is not tied to a
  source revision by an automated check.
- Device selection ranks device type before checking actual shader
  requirements, then unconditionally requests Vulkan 1.3, `shaderInt64`, and
  integer dot product even though the baseline shader does not require them.
- `docs/VULKAN.md` says `vk_check` covers five shapes, while the example
  currently runs one 1024 x 1024 case.

## Verified Model Matrix

| End-to-end model | Architecture | Relevant tensor formats |
|---|---|---|
| `Qwen3-0.6B-Q8_0.gguf` | qwen3 | 197 Q8_0, 113 F32 |
| `Qwen3-0.6B-Q4_0.gguf` | qwen3 | 193 Q4_0, 3 Q4_1, 1 Q6_K, 113 F32 |
| `Qwen3-0.6B-Q4_K_M.gguf` | qwen3 | 168 Q4_K, 29 Q6_K, 113 F32 |
| `Qwen3-Embedding-0.6B-f16.gguf` | qwen3 | 197 F16, 113 F32 |
| `Qwen3.5-0.8B-BF16.gguf` | qwen35 | 187 BF16, 133 F32 |

No inspected model contains Q5_K. Q5_K can receive synthetic kernel parity
coverage, but it must not be reported as end-to-end validated until a matching
model is available.

## Chosen Approach

Implement a Vulkan-specific model executor incrementally, starting with Qwen3.
It owns device buffers, pipelines, command recording, and GPU KV state, while
the existing CPU model remains the fallback and correctness oracle. This keeps
the change local to the only working GPU backend and avoids a speculative
cross-backend trait.

Two alternatives were rejected:

- Extending only the current per-matmul path cannot remove the dominant submit,
  fence, and CPU round-trip overhead.
- Introducing a generic graph or backend abstraction before a complete Vulkan
  path exists would change every model without proving the abstraction.

## Phase 1: Availability and Reproducibility

1. Make server startup call the same GPU enablement path as the CLI when
   `CliOptions.gpu` is true. Keep Vulkan opt-in; CPU behavior is unchanged when
   `--gpu` is absent or the feature is not compiled.
2. Correct README commands to use `--gpu` and document automatic Loader/ICD
   discovery before optional environment overrides.
3. Keep `shaders/glsl/` as the only Vulkan shader source tree and remove the
   conflicting `shaders/src/q8_matmul.comp`.
4. Add one regeneration/check script using `glslangValidator` and `spirv-val`.
   A small manifest records both source and SPIR-V hashes so CI detects source,
   artifact, or manifest drift. The binary remains checked in; normal Rust
   builds do not acquire a shader compiler dependency.
5. Add a CI job that compiles the Vulkan feature and validates shader artifacts
   without requiring a GPU. Real-hardware checks remain a separate runnable
   command because GitHub-hosted runners do not provide the required vendor
   matrix.

## Phase 2: Capability-Driven Context Creation

`VulkanContext` will enumerate compute-capable devices and build candidates
from properties, queue families, extensions, and features. Candidates are
ranked only after unsupported devices are removed. The baseline shader requests
only features it uses; integer dot product selects an optional dp4a pipeline
instead of being a hard requirement.

Initialization tries candidates in rank order. A device-specific creation or
pipeline failure is recorded and the next candidate is attempted. If none work,
the error contains the rejected device names and reasons before the existing
CPU fallback is used. Portability enumeration/subset remains conditional on
advertised extensions.

## Phase 3: Qwen3 Full-Token Vulkan Path

Add a Vulkan-only Qwen3 executor rather than changing the public `Kernel`
interface. Model construction uploads supported weights once. Session creation
allocates activation scratch buffers and KV buffers sized to the requested
context.

For each token, one command buffer records:

1. embedding/input staging;
2. RMS norm;
3. fused Q/K/V matvec;
4. Q/K norm, RoPE, and KV write;
5. attention score, softmax, and value accumulation;
6. output projection and residual;
7. FFN norm;
8. fused gate/up matvec, SiLU multiply, down projection, and residual;
9. final norm and logits.

One fence is waited at the token boundary and only logits plus the new KV delta
are read back. The current per-matmul path remains available until this vertical
slice passes correctness and performance gates, then becomes a compatibility
fallback rather than the preferred Qwen3 path.

### Failure Recovery

The CPU session remains usable at every committed token boundary. After a
successful GPU token, its new K/V delta is copied into the CPU shadow cache.
If recording, submission, timeout, or readback fails mid-token, the GPU context
is marked broken, the uncommitted GPU KV length is discarded, and that token is
recomputed on CPU from the last committed shadow state. A failed context is not
reused during the process lifetime.

This preserves the existing transparent fallback without copying the complete
KV cache after every token.

## Phase 4: Format and Architecture Expansion

GPU weight decoders are added behind the same internal matvec pipeline in this
order, each gated by synthetic kernel parity before model testing:

1. Q4_0, Q4_1, and Q6_K using `Qwen3-0.6B-Q4_0.gguf`;
2. Q4_K and Q6_K using `Qwen3-0.6B-Q4_K_M.gguf`;
3. F16 using `Qwen3-Embedding-0.6B-f16.gguf`;
4. BF16 plus qwen35-specific operations using `Qwen3.5-0.8B-BF16.gguf`;
5. Q5_K synthetic parity, followed by end-to-end validation only when a Q5_K
   model is supplied.

Each model reports which tensor formats ran on Vulkan and which fell back. A
mixed model is never labelled fully GPU-backed while a required tensor silently
runs on CPU.

## Testing and Acceptance

### Always-runnable checks

- Unit tests for device filtering/ranking and optional pipeline selection.
- Shader manifest hash validation and `spirv-val` when the tool is installed.
- Vulkan-feature compilation for library, CLI, server, and examples.
- CPU fallback tests with no Loader and with an intentionally rejected device
  description.
- Synthetic GPU-versus-scalar parity for every weight decoder and fused op.

### Local Apple M3 Max checks

- Initialize MoltenVK with no `DYLD_*`, `VK_ICD_FILENAMES`, or
  `VK_DRIVER_FILES` overrides.
- Restore representative `vk_check` shapes, including vocabulary projection
  and the `n_in = 16384` boundary; compare every output against the CPU scalar
  reference.
- For each available text model, compare fixed-prompt prefill logits, greedy
  token IDs, and 32 decode steps against the CPU path. For the embedding model,
  compare the complete output vector and ranking on a fixed input. GPU reduction
  order need not be bitwise identical, but greedy tokens and embedding rankings
  must match and numeric tolerances must be stated by each test.
- Benchmark CPU and Vulkan with identical model, prompt, context, thread count,
  warmup, and generation length. Use at least five alternating runs and report
  medians. Do not claim acceleration unless the end-to-end Vulkan median beats
  the CPU median; retain the measurements when it does not.

### Vendor matrix

Provide the same hardware test command for MoltenVK, Intel ANV, AMD RADV, and
NVIDIA. A vendor is listed as validated only after its command output records
device/driver versions, correctness results, and performance medians. CI
compilation is not hardware validation.

## Non-Goals

- No CUDA, native Metal, ROCm, wgpu, or generic backend abstraction in this
  work; MoltenVK remains the macOS Vulkan implementation.
- No model-file mutation or conversion.
- No claim that every repository model is GPU-backed; support is reported per
  architecture and tensor-format matrix.
- No unrelated cleanup of the repository's existing warnings or stale
  integration tests.

## Delivery Order

Each phase must leave a runnable check and keep CPU inference working. Phase 1
and Phase 2 land before the model executor. Qwen3 Q8_0 is completed and measured
before another format is added. Format additions are one independently
reviewable change each. Documentation is updated from planned to supported only
after the corresponding local or vendor validation succeeds.
