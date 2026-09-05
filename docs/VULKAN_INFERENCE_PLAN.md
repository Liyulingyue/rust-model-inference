# Vulkan Inference Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current Q8_0 matmul experiment into an opt-in, capability-driven Vulkan path that executes one complete Qwen3 token with GPU-resident activations and KV state, then extend the proven path to the tensor formats present in the supplied local models.

**Architecture:** Keep the existing CPU model as the correctness oracle and fallback. Extend `VulkanContext` only with Vulkan runtime primitives, place model-specific command recording in `src/vulkan/qwen3.rs`, and capture the GPU executor inside each `Qwen3Session`; no repository-wide backend trait is introduced. Each token is recorded into one command buffer and waits on one fence, while a CPU shadow KV cache is advanced only after the GPU token commits.

**Tech Stack:** Rust 2021, `ash` 0.37, Vulkan compute, GLSL 450, SPIR-V, GGUF tensor storage, existing scalar/SIMD CPU kernels.

**Spec:** `docs/VULKAN_INFERENCE_DESIGN.md`

## Global Constraints

- Vulkan remains opt-in through `--features vulkan --gpu`; a build or invocation without both keeps the existing CPU behavior.
- The first full-token target is qwen3 `Qwen3-0.6B-Q8_0.gguf` on Apple M3 Max through MoltenVK.
- A successful token uses one queue submission and one fence wait; activations and full GPU KV state do not round-trip between individual operators.
- CPU shadow KV receives only the committed token's K/V delta; a failed GPU token is recomputed on CPU from the previous committed state.
- `shaders/glsl/` is the only Vulkan shader source directory; checked-in SPIR-V is reproducible and validated in CI.
- The baseline path must not request `shaderInt64` or integer-dot-product features; integer dot product is an optional pipeline.
- No new dependency, generic backend trait, CUDA, native Metal, ROCm, wgpu rewrite, or model-file conversion is part of this work.
- Hardware claims require recorded output from that hardware; compile-only CI is not vendor validation.
- Stage only the paths named by each task. Preserve `.codex/` and all unrelated worktree changes.

## File Map

- `src/vulkan.rs`: Loader discovery, instance/device selection, low-level buffers, pipelines, command buffers, submission, and the compatibility Q8_0 matmul API.
- `src/vulkan/ops.rs`: Vulkan-only arena layout and operator recording helpers shared by the Qwen3 executor and synthetic checks.
- `src/vulkan/qwen3.rs`: Qwen3 eligibility, weight upload table, per-session GPU buffers/KV, full-token recording, commit, and reset.
- `src/models/qwen3/trunk/session.rs`: Select GPU versus CPU once per session, keep the CPU token implementation as fallback, and commit GPU logits/KV deltas.
- `src/ops/float.rs`: Process-wide opt-in/failed state and lazy `VulkanContext` construction.
- `src/bin/server.rs`, `README.md`, `docs/VULKAN.md`: User entry points and truthful support documentation.
- `shaders/glsl/*.comp`, `shaders/bin/*.spv`, `shaders/manifest.sha256`: Authoritative kernels, checked-in binaries, and their hashes.
- `scripts/vulkan-shaders.sh`: One command for shader regeneration and one command for deterministic validation.
- `examples/vk_check.rs`: Synthetic matvec shape and scalar-parity gate.
- `examples/vk_ops_check.rs`: Synthetic parity gate for GPU-resident elementwise/attention operations and every weight decoder.
- `examples/vk_model_check.rs`: Fixed-model logits, greedy-token, embedding, and benchmark acceptance command.
- `.github/workflows/ci.yml`: GPU-free Vulkan compile and SPIR-V validation job.

---

### Task 1: Finish Loader and Portability Discovery

**Files:**
- Modify: `src/vulkan.rs:101-948`

**Interfaces:**
- Produces: `fn load_entry() -> Result<ash::Entry, VulkanError>`
- Produces: `fn extension_available(&[vk::ExtensionProperties], &CStr) -> bool`
- Preserves: `pub fn VulkanContext::new() -> Result<VulkanContext, VulkanError>`

- [ ] **Step 1: Add the macOS initialization regression test**

```rust
#[cfg(all(test, feature = "vulkan", target_os = "macos"))]
#[test]
fn initializes_with_homebrew_moltenvk() {
    let installed = [
        "/opt/homebrew/lib/libvulkan.dylib",
        "/usr/local/lib/libvulkan.dylib",
    ]
    .iter()
    .any(|path| std::path::Path::new(path).exists());
    if !installed {
        eprintln!("skipping: no Homebrew Vulkan loader installed");
        return;
    }
    let context = VulkanContext::new().expect("Homebrew MoltenVK should initialize");
    assert!(!context.device_name().is_empty());
}
```

- [ ] **Step 2: Run the regression against the design commit**

Run: `cargo test --lib --locked --features vulkan vulkan::tests::initializes_with_homebrew_moltenvk -- --exact --nocapture`

Expected before the loader/portability patch: FAIL on the local Mac with `Vulkan init failed` or `No compute device found` when no `DYLD_*`, `VK_ICD_FILENAMES`, or `VK_DRIVER_FILES` override is set.

- [ ] **Step 3: Extract Loader fallback and conditionally enable portability extensions**

```rust
unsafe fn load_entry() -> Result<ash::Entry, VulkanError> {
    ash::Entry::load()
        .or_else(|error| {
            #[cfg(target_os = "macos")]
            for path in [
                "/opt/homebrew/lib/libvulkan.dylib",
                "/usr/local/lib/libvulkan.dylib",
            ] {
                if let Ok(entry) = ash::Entry::load_from(path) {
                    return Ok(entry);
                }
            }
            Err(error)
        })
        .map_err(|error| VulkanError::InitFailed(error.to_string()))
}
```

Use `VK_KHR_portability_enumeration` plus `ENUMERATE_PORTABILITY_KHR` only when the instance advertises it, and request `VK_KHR_portability_subset` only when the chosen physical device advertises it.

- [ ] **Step 4: Verify on the local Loader and compile the non-runtime path**

Run:

```bash
env -u DYLD_LIBRARY_PATH -u VK_ICD_FILENAMES -u VK_DRIVER_FILES \
  cargo test --lib --locked --features vulkan \
  vulkan::tests::initializes_with_homebrew_moltenvk -- --exact --nocapture
cargo check --locked --features vulkan --bin rust-model-inference
git diff --check
```

Expected: PASS and the test prints `Apple M3 Max` through `VulkanContext::new()`.

- [ ] **Step 5: Commit only the Loader fix**

```bash
git add src/vulkan.rs
git commit -m "fix: discover Homebrew Vulkan on macOS"
```

### Task 2: Wire `--gpu` Through Server Startup and Correct User Commands

**Files:**
- Modify: `src/ops/float.rs:6-14`
- Modify: `src/bin/server.rs:1250-1324`
- Modify: `README.md:128-166`

**Interfaces:**
- Produces: `pub fn gpu_requested() -> bool`
- Produces: `fn configure_gpu(options: &CliOptions)` in the server binary
- Consumes: existing `pub fn enable_gpu()`

- [ ] **Step 1: Add a server wiring test**

```rust
#[cfg(all(test, feature = "vulkan"))]
#[test]
fn gpu_flag_reaches_shared_switch() {
    let options = CliOptions {
        gpu: true,
        ..CliOptions::default()
    };
    configure_gpu(&options);
    assert!(rust_model_inference::ops::float::gpu_requested());
}
```

- [ ] **Step 2: Run it and observe the missing wiring**

Run: `cargo test --locked --features vulkan --bin server tests::gpu_flag_reaches_shared_switch -- --exact`

Expected: FAIL because `configure_gpu` and `gpu_requested` do not exist.

- [ ] **Step 3: Add the minimal shared switch and call it before backend construction**

```rust
pub fn gpu_requested() -> bool {
    GPU_ENABLED.load(Ordering::Relaxed)
}
```

```rust
fn configure_gpu(options: &CliOptions) {
    if options.gpu {
        rust_model_inference::ops::enable_gpu();
    }
}
```

Call `configure_gpu(&options);` after CLI validation and before `build_backend(&options)`. Replace the README's `USE_GPU=1` example with:

```bash
cargo run --release --features vulkan -- \
  --gpu \
  --model /Users/gouzi/Documents/git/rust-model-inference/models/Qwen3-0.6B-Q8_0/Qwen3-0.6B-Q8_0.gguf \
  --prompt "法国的首都是"
```

Document that Loader/ICD discovery is automatic and environment variables are troubleshooting overrides, not mandatory setup.

- [ ] **Step 4: Verify CLI and server builds**

Run:

```bash
cargo test --locked --features vulkan --bin server tests::gpu_flag_reaches_shared_switch -- --exact
cargo check --locked --features vulkan --bin rust-model-inference
cargo check --locked --features vulkan --bin server
git diff --check
```

Expected: all checks PASS.

- [ ] **Step 5: Commit the entry-point fix**

```bash
git add src/ops/float.rs src/bin/server.rs README.md
git commit -m "fix: honor gpu flag in server"
```

### Task 3: Make Vulkan Shaders Reproducible

**Files:**
- Delete: `shaders/src/q8_matmul.comp`
- Modify: `shaders/glsl/q8_matmul.comp`
- Modify: `shaders/glsl/q8_matmul_dp4a.comp`
- Modify: `shaders/bin/q8_matmul.spv`
- Create: `shaders/bin/q8_matmul_dp4a.spv`
- Create: `shaders/manifest.sha256`
- Create: `scripts/vulkan-shaders.sh`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `scripts/vulkan-shaders.sh update`
- Produces: `scripts/vulkan-shaders.sh check`
- Produces: baseline and optional-dp4a SPIR-V artifacts embedded by Rust

- [ ] **Step 1: Add a check mode that initially detects source/artifact drift**

Create the script with this exact behavior:

```bash
#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
names=(q8_matmul q8_matmul_dp4a)
hash_file="$root/shaders/manifest.sha256"

compile() {
  glslangValidator -V --target-env vulkan1.1 \
    "$root/shaders/glsl/$1.comp" -o "$2"
}

hashes() {
  cd "$root"
  shasum -a 256 \
    shaders/glsl/q8_matmul.comp \
    shaders/glsl/q8_matmul_dp4a.comp \
    shaders/bin/q8_matmul.spv \
    shaders/bin/q8_matmul_dp4a.spv
}

case "${1:-check}" in
  update)
    for name in "${names[@]}"; do
      compile "$name" "$root/shaders/bin/$name.spv"
    done
    hashes > "$hash_file"
    ;;
  check)
    (cd "$root" && shasum -a 256 -c shaders/manifest.sha256)
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    for name in "${names[@]}"; do
      spirv-val --target-env vulkan1.1 "$root/shaders/bin/$name.spv"
      compile "$name" "$tmp/$name.spv"
      cmp "$tmp/$name.spv" "$root/shaders/bin/$name.spv"
    done
    ;;
  *)
    echo "usage: $0 [check|update]" >&2
    exit 2
    ;;
esac
```

- [ ] **Step 2: Run check before regeneration**

Run: `bash scripts/vulkan-shaders.sh check`

Expected: FAIL because the manifest and dp4a binary do not exist.

- [ ] **Step 3: Remove the duplicate source and make both GLSL files compile**

Delete `shaders/src/q8_matmul.comp`. In the dp4a shader, name the integer accumulator `dot_sum` so it cannot shadow the GLSL `dot` overload. Change the baseline row mapping to support vocabulary projections larger than `maxComputeWorkGroupCount[0]`:

```glsl
uint row = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
if (row >= dims.y) return;
```

Make the same two-dimensional row mapping in the dp4a shader.

- [ ] **Step 4: Regenerate, validate, and add a GPU-free CI job**

Run:

```bash
bash scripts/vulkan-shaders.sh update
bash scripts/vulkan-shaders.sh check
```

Add this job to `.github/workflows/ci.yml`:

```yaml
  vulkan:
    name: Vulkan compile and shaders
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: sudo apt-get update && sudo apt-get install -y glslang-tools spirv-tools
      - run: bash scripts/vulkan-shaders.sh check
      - run: cargo check --locked --features vulkan --lib
      - run: cargo check --locked --features vulkan --bin rust-model-inference
      - run: cargo check --locked --features vulkan --bin server
      - run: cargo check --locked --features vulkan --examples
```

- [ ] **Step 5: Verify and commit shader provenance**

Run: `git diff --check && bash scripts/vulkan-shaders.sh check`

Expected: both commands PASS.

```bash
git add .github/workflows/ci.yml scripts/vulkan-shaders.sh \
  shaders/glsl/q8_matmul.comp shaders/glsl/q8_matmul_dp4a.comp \
  shaders/bin/q8_matmul.spv shaders/bin/q8_matmul_dp4a.spv \
  shaders/manifest.sha256
git add -u shaders/src/q8_matmul.comp
git commit -m "build: validate Vulkan shader artifacts"
```

### Task 4: Select Devices by Actual Shader Requirements

**Files:**
- Modify: `src/vulkan.rs:101-620`

**Interfaces:**
- Produces: `DeviceCandidate { physical_device, queue_family, name, device_type, api_version, limits, portability_subset, integer_dot_product }`
- Produces: `fn rejection_reason(&DeviceCandidate) -> Option<String>`
- Produces: `fn dispatch_grid(n_out: usize, limits: &vk::PhysicalDeviceLimits) -> Result<(u32, u32), VulkanError>`
- Produces: `PipelineVariant::{Baseline, IntegerDotProduct}`
- Produces: `fn DeviceCandidate::pipeline_variant(&self) -> PipelineVariant`
- Preserves: `VulkanContext::new()` fallback contract

- [ ] **Step 1: Add pure selection and dispatch-grid tests**

```rust
fn candidate_for_test(
    device_type: vk::PhysicalDeviceType,
    integer_dot_product: bool,
    shared_bytes: u32,
) -> DeviceCandidate {
    let mut limits = vk::PhysicalDeviceLimits::default();
    limits.max_compute_work_group_invocations = 64;
    limits.max_compute_work_group_size[0] = 64;
    limits.max_compute_work_group_count = [65_535, 65_535, 65_535];
    limits.max_compute_shared_memory_size = shared_bytes;
    DeviceCandidate {
        physical_device: vk::PhysicalDevice::null(),
        queue_family: 0,
        name: match device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => "test-discrete",
            _ => "test-integrated",
        }
        .into(),
        device_type,
        api_version: vk::API_VERSION_1_1,
        limits,
        portability_subset: false,
        integer_dot_product,
    }
}

#[test]
fn baseline_accepts_compute_without_int64_or_dot_product() {
    let candidate = candidate_for_test(vk::PhysicalDeviceType::INTEGRATED_GPU, false, 32 * 1024);
    assert_eq!(rejection_reason(&candidate), None);
    assert_eq!(candidate.pipeline_variant(), PipelineVariant::Baseline);
}

#[test]
fn unsupported_discrete_is_removed_before_ranking() {
    let bad = candidate_for_test(vk::PhysicalDeviceType::DISCRETE_GPU, true, 8 * 1024);
    let good = candidate_for_test(vk::PhysicalDeviceType::INTEGRATED_GPU, false, 32 * 1024);
    let names: Vec<_> = rank_supported(vec![bad, good]).into_iter().map(|value| value.name).collect();
    assert_eq!(names, ["test-integrated"]);
}

#[test]
fn vocabulary_projection_uses_two_dimensional_dispatch() {
    let mut limits = vk::PhysicalDeviceLimits::default();
    limits.max_compute_work_group_count = [65_535, 65_535, 65_535];
    assert_eq!(dispatch_grid(151_936, &limits).unwrap(), (65_535, 3));
}
```

- [ ] **Step 2: Run the tests and observe missing capability types**

Run: `cargo test --lib --locked --features vulkan vulkan::tests -- --nocapture`

Expected: FAIL because the candidate, ranking, and grid helpers do not exist.

- [ ] **Step 3: Enumerate and reject candidates before sorting**

The baseline rejection function checks only requirements used by the checked-in shader:

```rust
const Q8_SHARED_BYTES: u32 = (4096 + 64) * 4;

fn rejection_reason(candidate: &DeviceCandidate) -> Option<String> {
    let limits = &candidate.limits;
    if limits.max_compute_work_group_invocations < 64 {
        return Some("maxComputeWorkGroupInvocations is below 64".into());
    }
    if limits.max_compute_work_group_size[0] < 64 {
        return Some("maxComputeWorkGroupSize[0] is below 64".into());
    }
    if limits.max_compute_shared_memory_size < Q8_SHARED_BYTES {
        return Some(format!(
            "maxComputeSharedMemorySize {} is below {Q8_SHARED_BYTES}",
            limits.max_compute_shared_memory_size
        ));
    }
    None
}
```

Sort supported candidates by device type (`DISCRETE_GPU`, `INTEGRATED_GPU`, `VIRTUAL_GPU`, `CPU`, other) and retain enumeration order as the tie breaker. Do not reject on API 1.3, `shaderInt64`, or integer dot product.

- [ ] **Step 4: Try candidates in rank order and choose the optional pipeline**

Query `vk::PhysicalDeviceShaderIntegerDotProductFeatures` through `get_physical_device_features2`. Request it only for a candidate that reports the feature, and enable `VK_KHR_shader_integer_dot_product` only when it is extension-provided rather than core. Request portability subset independently. Build `q8_matmul_dp4a.spv` when integer dot product is available; if that optional pipeline fails, build the baseline pipeline on the same candidate. If device creation or the baseline pipeline fails, destroy that candidate's partial resources and continue to the next candidate. Aggregate failures as `device name: reason` in the final `VulkanError::InitFailed`.

Negotiate the instance version instead of requesting 1.3 unconditionally:

```rust
let loader_version = entry
    .try_enumerate_instance_version()
    .map_err(|error| VulkanError::InitFailed(error.to_string()))?
    .unwrap_or(vk::API_VERSION_1_0);
let api_version = loader_version.min(vk::API_VERSION_1_3);
```

On Vulkan 1.0, query optional integer-dot-product support only when `VK_KHR_get_physical_device_properties2` is advertised; its absence disables the optional pipeline, not the baseline device.

Use the physical-device limits in `matmul_q8_0`:

```rust
let (groups_x, groups_y) = dispatch_grid(n_out, &self.limits)?;
self.device.cmd_dispatch(self.command_buffer, groups_x, groups_y, 1);
```

- [ ] **Step 5: Verify pure tests and the real MoltenVK candidate**

Run:

```bash
cargo test --lib --locked --features vulkan vulkan::tests -- --nocapture
cargo run --release --locked --features vulkan --example vk_check
```

Expected: tests PASS, initialization names `Apple M3 Max`, and `vk_check` reports no bad row.

- [ ] **Step 6: Commit capability-driven selection**

```bash
git add src/vulkan.rs
git commit -m "fix: select capable Vulkan devices"
```

### Task 5: Restore the Documented Matvec Correctness Gate

**Files:**
- Modify: `examples/vk_check.rs`
- Modify: `docs/VULKAN.md`

**Interfaces:**
- Produces: deterministic cases `(1024, 1024)`, `(1024, 3072)`, `(3072, 1024)`, `(1024, 151936)`, `(16384, 32)`
- Produces: nonzero exit on Vulkan error, non-finite output, or tolerance failure

- [ ] **Step 1: Turn the existing one-shape example into a failing five-shape assertion**

```rust
const CASES: &[(usize, usize)] = &[
    (1024, 1024),
    (1024, 3072),
    (3072, 1024),
    (1024, 151_936),
    (16_384, 32),
];

fn within_tolerance(gpu: f32, cpu: f32) -> bool {
    gpu.is_finite() && (gpu - cpu).abs() <= 1e-4 + 1e-4 * cpu.abs()
}
```

Return `ExitCode::FAILURE` on the first mismatching row and print `device`, shape, maximum absolute error, maximum relative error, and first mismatching row for every case.

- [ ] **Step 2: Run before the 2D dispatch fix**

Run: `cargo run --release --locked --features vulkan --example vk_check`

Expected before Task 4: FAIL or a Vulkan validation/dispatch error on `(1024, 151936)`.

- [ ] **Step 3: Run the completed gate and align documentation with output**

Run: `cargo run --release --locked --features vulkan --example vk_check`

Expected: five passing shapes and process exit 0.

Update `docs/VULKAN.md` with the five exact shapes and the implemented tolerance `abs <= 1e-4 + 1e-4 * abs(cpu)`; remove the old blanket `rel <= 3e-7` claim.

- [ ] **Step 4: Commit the shape gate**

```bash
git add examples/vk_check.rs docs/VULKAN.md
git commit -m "test: cover Vulkan model matvec shapes"
```

### Task 6: Add Reusable Buffer, Pipeline, and Token Command Recording Primitives

**Files:**
- Modify: `src/vulkan.rs`
- Create: `src/vulkan/ops.rs`

**Interfaces:**
- Produces: `GpuBuffer { buffer, memory, size, mapped }`
- Produces: `ArenaLayout::for_dims(n_embd, n_ff, n_head, n_head_kv, head_dim) -> Result<ArenaLayout, VulkanError>`
- Produces: `ArenaLayout::qwen3(&Qwen3Config, capacity) -> Result<ArenaLayout, VulkanError>`
- Produces: `TokenDispatchPlan::qwen3_dense(layer_count) -> TokenDispatchPlan`
- Produces: `TokenCommands::{begin, bind, barrier, dispatch, submit_and_wait}`
- Produces: `VulkanContext::{upload_static, allocate_session_buffer, create_pipeline}`
- Produces: `VulkanContext::submission_count() -> u64`, incremented once per `queue_submit`

- [ ] **Step 1: Add deterministic arena and dispatch tests**

```rust
#[test]
fn qwen3_arena_regions_are_aligned_and_disjoint() {
    let layout = ArenaLayout::for_dims(1024, 3072, 16, 2, 64).unwrap();
    let regions = layout.regions();
    assert!(regions.iter().all(|region| region.offset % 16 == 0));
    assert!(regions.windows(2).all(|pair| pair[0].end() <= pair[1].offset));
}

#[test]
fn token_command_has_one_submit_boundary() {
    let plan = TokenDispatchPlan::qwen3_dense(28);
    assert_eq!(plan.queue_submissions, 1);
    assert_eq!(plan.fence_waits, 1);
    assert!(plan.dispatches > 28);
}
```

- [ ] **Step 2: Run tests to establish RED**

Run: `cargo test --lib --locked --features vulkan vulkan::ops::tests -- --nocapture`

Expected: FAIL because the module and types do not exist.

- [ ] **Step 3: Move ownership, not behavior, into reusable primitives**

Rename the current `BufferInfo` to `GpuBuffer` and retain its persistently mapped host-visible allocation policy. Add 16-byte-aligned arena regions for `x`, `normed`, `q`, `k`, `v`, `attn`, `projection`, `gate`, `up`, `down`, `logits`, `q8`, `q8_scales`, `scores`, `kv_k`, `kv_v`, `kv_delta_k`, and `kv_delta_v`. Use checked `usize` arithmetic and convert to `vk::DeviceSize` only after the complete layout succeeds.

The command wrapper must record barriers without submitting:

```rust
pub(crate) unsafe fn compute_barrier(&self, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    self.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        std::slice::from_ref(&barrier),
        &[],
        &[],
    );
}
```

`submit_and_wait` is the only token-path method that calls `queue_submit` or `wait_for_fences`.

- [ ] **Step 4: Keep the compatibility matmul working through the new primitives**

Run:

```bash
cargo test --lib --locked --features vulkan vulkan::ops::tests
cargo run --release --locked --features vulkan --example vk_check
```

Expected: arena tests and all five matvec cases PASS.

- [ ] **Step 5: Commit runtime primitives**

```bash
git add src/vulkan.rs src/vulkan/ops.rs
git commit -m "refactor: add Vulkan token command primitives"
```

### Task 7: Add GPU-Resident Qwen3 Operators

**Files:**
- Modify: `src/vulkan/ops.rs`
- Create: `examples/vk_ops_check.rs`
- Create: `shaders/glsl/quantize_q8_0.comp`
- Create: `shaders/glsl/q8_matmul_grouped.comp`
- Create: `shaders/glsl/rms_norm.comp`
- Create: `shaders/glsl/qk_norm_rope.comp`
- Create: `shaders/glsl/kv_write.comp`
- Create: `shaders/glsl/attention_scores.comp`
- Create: `shaders/glsl/softmax.comp`
- Create: `shaders/glsl/attention_values.comp`
- Create: `shaders/glsl/silu_mul.comp`
- Create: `shaders/glsl/add.comp`
- Create: corresponding `shaders/bin/*.spv`
- Modify: `scripts/vulkan-shaders.sh`
- Modify: `shaders/manifest.sha256`

**Interfaces:**
- Produces: `record_quantize_q8_0`, `record_rms_norm`, `record_q8_matvec`, `record_q8_matvec_group`, `record_qk_norm_rope`, `record_kv_write`, `record_attention`, `record_silu_mul`, and `record_add`
- Consumes: one `TokenCommands` command buffer and `ArenaLayout` offsets

- [ ] **Step 1: Add one synthetic parity program covering every operator**

The program creates deterministic inputs and compares complete output slices against existing CPU functions. Use these gates:

```rust
check("rms_norm", &gpu, &cpu, 2e-5, 2e-5)?;
check("qk_norm_rope", &gpu, &cpu, 3e-5, 3e-5)?;
check("attention_scores", &gpu, &cpu, 3e-5, 3e-5)?;
check("softmax", &gpu, &cpu, 3e-5, 3e-5)?;
check("attention_values", &gpu, &cpu, 4e-5, 4e-5)?;
check("silu_mul", &gpu, &cpu, 3e-5, 3e-5)?;
check("residual_add", &gpu, &cpu, 1e-6, 1e-6)?;
```

Also assert the context's test-only submission counter increases by exactly one for the complete synthetic operator chain.

- [ ] **Step 2: Run the operator program before shaders exist**

Run: `cargo run --release --locked --features vulkan --example vk_ops_check`

Expected: compile failure for the missing recording methods.

- [ ] **Step 3: Implement operators with explicit layouts and bounds**

Use F32 activation and GPU KV storage. `quantize_q8_0.comp` writes signed bytes packed into `uint` words plus one F32 scale per 32 values. `q8_matmul_grouped.comp` binds three fixed weight/output slots and uses `gl_WorkGroupID.z` to execute Q/K/V in one dispatch or gate/up in one dispatch; unused slot lengths are zero. `rms_norm.comp` uses one workgroup per vector and the existing `eps`. `qk_norm_rope.comp` applies optional Q/K RMS weights and Qwen3 Neox rotation. `kv_write.comp` writes `[layer][token][kv_stride]` and also copies that token into contiguous delta buffers. Attention scores use `1 / sqrt(head_dim)`, softmax subtracts the maximum, and value accumulation maps grouped-query heads with `kv_head = head / group_size`.

Each recorder validates offsets and lengths on the host, binds one pipeline, dispatches, then records `compute_barrier`; none submit or wait.

- [ ] **Step 4: Regenerate shaders and run scalar parity**

Run:

```bash
bash scripts/vulkan-shaders.sh update
bash scripts/vulkan-shaders.sh check
cargo run --release --locked --features vulkan --example vk_ops_check
```

Expected: all named operator gates PASS and `submissions=1`.

- [ ] **Step 5: Commit the operator slice**

```bash
git add src/vulkan/ops.rs examples/vk_ops_check.rs scripts/vulkan-shaders.sh \
  shaders/glsl/quantize_q8_0.comp shaders/glsl/q8_matmul_grouped.comp \
  shaders/glsl/rms_norm.comp \
  shaders/glsl/qk_norm_rope.comp shaders/glsl/kv_write.comp \
  shaders/glsl/attention_scores.comp shaders/glsl/softmax.comp \
  shaders/glsl/attention_values.comp shaders/glsl/silu_mul.comp \
  shaders/glsl/add.comp shaders/bin/quantize_q8_0.spv \
  shaders/bin/q8_matmul_grouped.spv \
  shaders/bin/rms_norm.spv shaders/bin/qk_norm_rope.spv \
  shaders/bin/kv_write.spv shaders/bin/attention_scores.spv \
  shaders/bin/softmax.spv shaders/bin/attention_values.spv \
  shaders/bin/silu_mul.spv shaders/bin/add.spv shaders/manifest.sha256
git commit -m "feat: add Vulkan Qwen3 operators"
```

### Task 8: Execute a Complete Dense Qwen3 Q8_0 Token

**Files:**
- Create: `src/vulkan/qwen3.rs`
- Modify: `src/vulkan.rs`
- Modify: `src/models/qwen3/trunk/session.rs:35-1000`
- Modify: `src/models/qwen3/trunk/weights.rs:25-568`

**Interfaces:**
- Produces: `EligibilityFacts { architecture, has_moe, n_deepstack_layers, has_qkv_bias, rope, weight_formats }`
- Produces: `fn check_eligibility(&EligibilityFacts) -> Result<(), String>`
- Produces: `Qwen3VulkanSession::try_new(&Qwen3Model, capacity, &'static VulkanContext) -> Result<Option<Self>, VulkanError>`
- Produces: `Qwen3VulkanSession::forward_token<'a>(&'a mut self, input: &[f32], position: usize) -> Result<GpuTokenResult<'a>, VulkanError>`
- Produces: `GpuTokenResult<'a> { logits: &'a [f32], k_delta: &'a [f32], v_delta: &'a [f32] }` backed by buffers allocated at session creation
- Produces: `TokenCommitState::{new(committed_len), begin(position), commit, abort, committed_len}`
- Produces: `Qwen3VulkanSession::{commit_token, reset}`
- Preserves: CPU `Qwen3Session` output and fallback

- [ ] **Step 1: Add eligibility and commit-state tests**

```rust
#[test]
fn qwen3_q8_dense_is_eligible() {
    let facts = EligibilityFacts {
        architecture: "qwen3".into(),
        has_moe: false,
        n_deepstack_layers: 0,
        has_qkv_bias: false,
        rope: Qwen3Rope::Neox,
        weight_formats: vec![GGMLType::Q8_0; 198],
    };
    assert_eq!(check_eligibility(&facts), Ok(()));
}

#[test]
fn unsupported_architecture_stays_on_cpu() {
    let facts = EligibilityFacts {
        architecture: "qwen3vl".into(),
        has_moe: false,
        n_deepstack_layers: 4,
        has_qkv_bias: true,
        rope: Qwen3Rope::Interleaved {
            sections: [16, 24, 24, 0],
            n_dims: 64,
        },
        weight_formats: vec![GGMLType::Q8_0; 198],
    };
    assert!(check_eligibility(&facts).is_err());
}

#[test]
fn failed_token_does_not_advance_committed_kv() {
    let mut state = TokenCommitState::new(7);
    state.begin(7).unwrap();
    state.abort();
    assert_eq!(state.committed_len(), 7);
}
```

- [ ] **Step 2: Run tests to establish RED**

Run: `cargo test --lib --locked --features vulkan vulkan::qwen3::tests -- --nocapture`

Expected: FAIL because the Qwen3 Vulkan module does not exist.

- [ ] **Step 3: Upload exactly the tensors needed by dense qwen3**

Eligibility requires architecture `qwen3`, `moe.is_none()`, `n_deepstack_layers == 0`, `has_qkv_bias == false`, `Qwen3Rope::Neox`, and Q8_0 for every matvec/output tensor used by the token. F32 norm tensors are supported as raw F32 buffers. The token embedding lookup remains the existing CPU lookup followed by one input staging copy. Look up weight bytes by the existing GGUF names (`blk.{layer}.attn_q.weight`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`, and `output.weight` with `token_embd.weight` as tied-output fallback) through `model.source`; do not add raw-byte methods to `Kernel`.

- [ ] **Step 4: Record the full token without an inner submission**

`forward_token` stages the CPU embedding vector once, begins one command buffer, and records for every layer:

```rust
ops.record_rms_norm(x, layer.attn_norm, normed, eps)?;
ops.record_quantize_q8_0(normed, q8, scales)?;
ops.record_q8_matvec_group(
    &[(layer.wq, q), (layer.wk, k), (layer.wv, v)],
    q8,
    scales,
)?;
ops.record_qk_norm_rope(q, k, layer.q_norm, layer.k_norm, position)?;
ops.record_kv_write(layer_index, position, k, v, kv_k, kv_v, delta_k, delta_v)?;
ops.record_attention(q, kv_k, kv_v, position + 1, attn)?;
ops.record_quantize_q8_0(attn, q8, scales)?;
ops.record_q8_matvec(layer.wo, q8, scales, projection)?;
ops.record_add(x, projection)?;
ops.record_rms_norm(x, layer.ffn_norm, normed, eps)?;
ops.record_quantize_q8_0(normed, q8, scales)?;
ops.record_q8_matvec_group(&[(layer.w_gate, gate), (layer.w_up, up)], q8, scales)?;
ops.record_silu_mul(gate, up)?;
ops.record_quantize_q8_0(gate, q8, scales)?;
ops.record_q8_matvec(layer.w_down, q8, scales, down)?;
ops.record_add(x, down)?;
```

Then record final RMS norm, quantization, and output matvec. Call `submit_and_wait` once and copy only logits and contiguous K/V delta buffers into host mirrors allocated by `Qwen3VulkanSession::try_new`; the hot token path allocates no `Vec`.

- [ ] **Step 5: Refactor the current loop into CPU-token and dispatch wrappers**

Move the existing per-token body without numerical changes into `forward_token_cpu`. Add `gpu: Option<Qwen3VulkanSession>` to `Qwen3Session` under the Vulkan feature. Construct it once in `new_with_kv_state` only when `get_vulkan_context()` succeeds. The wrapper is:

```rust
match self.gpu.as_mut().map(|gpu| gpu.forward_token(input, position[0])) {
    Some(Ok(result)) => {
        commit_shadow_kv(&mut self.kv_state, step, &result.k_delta, &result.v_delta)?;
        self.scratch.logits.copy_from_slice(&result.logits);
        self.gpu.as_mut().unwrap().commit_token();
        Ok(())
    }
    Some(Err(error)) => {
        crate::ops::mark_gpu_broken(&error.to_string());
        self.gpu = None;
        self.forward_token_cpu(step, position, input)
    }
    None => self.forward_token_cpu(step, position, input),
}
```

`commit_shadow_kv` converts to F16 when `KvState` uses F16 and copies directly for F32. `reset_kv()` resets both CPU and GPU state. Do not mark GPU broken for an ineligible model; log its explicit CPU fallback reason once.

- [ ] **Step 6: Verify CPU regression and GPU submission count**

Run:

```bash
cargo test --lib --locked --features vulkan models::qwen3::trunk::tests
cargo run --release --locked --features vulkan --example vk_ops_check
```

Expected: CPU tests PASS and the synthetic full-token fixture reports one submission and one fence wait.

- [ ] **Step 7: Commit the Q8_0 vertical slice**

```bash
git add src/vulkan.rs src/vulkan/qwen3.rs \
  src/models/qwen3/trunk/session.rs src/models/qwen3/trunk/weights.rs
git commit -m "feat: execute Qwen3 Q8 tokens on Vulkan"
```

### Task 9: Gate Qwen3 Q8_0 with the Supplied Model

**Files:**
- Create: `examples/vk_model_check.rs`
- Modify: `src/models/qwen3/trunk/session.rs`
- Modify: `docs/VULKAN.md`

**Interfaces:**
- Produces: `pub fn Qwen3Session::last_logits(&self) -> &[f32]`
- Produces: `vk_model_check qwen3 --model PATH [--benchmark]`
- Consumes: `/Users/gouzi/Documents/git/rust-model-inference/models/Qwen3-0.6B-Q8_0/Qwen3-0.6B-Q8_0.gguf`

- [ ] **Step 1: Add the acceptance program with a deliberately unmet backend assertion**

Use prompt `法国的首都是`, temperature `0`, F16 CPU shadow KV, 32 decode tokens, and the same thread count for both paths. Create and run the CPU session before calling `enable_gpu()`, then create the GPU session. Snapshot `VulkanContext::submission_count()` before and after the GPU run and compare the delta. Compare the complete prefill-logit vector with:

```rust
const LOGIT_ABS: f32 = 2e-3;
const LOGIT_REL: f32 = 2e-3;
assert_close("prefill_logits", gpu_logits, cpu_logits, LOGIT_ABS, LOGIT_REL)?;
if gpu_tokens != cpu_tokens {
    return Err(format!("greedy token mismatch: gpu={gpu_tokens:?} cpu={cpu_tokens:?}"));
}
if gpu_submission_count != prompt_tokens + 32 {
    return Err(format!("expected one submission per token, got {gpu_submission_count}"));
}
```

- [ ] **Step 2: Run before the full-token executor is selected**

Run:

```bash
cargo run --release --locked --features vulkan --example vk_model_check -- \
  qwen3 \
  --model /Users/gouzi/Documents/git/rust-model-inference/models/Qwen3-0.6B-Q8_0/Qwen3-0.6B-Q8_0.gguf
```

Expected before Task 8: FAIL because the observed submission count is per matmul rather than per token.

- [ ] **Step 3: Require both logits and greedy-token parity**

Run the same correctness command until the complete prefill-logit vector satisfies `LOGIT_ABS`/`LOGIT_REL`, all 32 greedy token IDs are identical, and the submission-count delta is exactly one per evaluated token. The program reports the first logit index exceeding tolerance with CPU value, GPU value, absolute error, and relative error.

- [ ] **Step 4: Run alternating benchmark medians**

With `--benchmark`, run one warmup plus five alternating CPU/GPU generation runs using identical prompt, 32 generated tokens, capacity, and thread count. Print all samples and median prompt/decode tokens per second. Exit successfully even when Vulkan is slower, but print `acceleration=false`; documentation must not claim acceleration in that case.

Run the command twice: once for correctness and once with `--benchmark`.

- [ ] **Step 5: Document measured facts and commit the model gate**

Record the model path, file size/hash, device/driver from `vulkaninfo --summary`, exact command, tolerances, token match, submission count, five timings, and medians in `docs/VULKAN.md`.

```bash
git add examples/vk_model_check.rs src/models/qwen3/trunk/session.rs docs/VULKAN.md
git commit -m "test: validate Qwen3 Q8 Vulkan inference"
```

### Task 10: Add Q4_0, Q4_1, and Q6_K Weight Decoders

**Files:**
- Modify: `src/vulkan/qwen3.rs`
- Modify: `src/vulkan/ops.rs`
- Modify: `src/models/qwen3/trunk/forward.rs`
- Modify: `examples/vk_ops_check.rs`
- Modify: `examples/vk_model_check.rs`
- Create: `shaders/glsl/q4_0_matmul.comp`
- Create: `shaders/glsl/q4_1_matmul.comp`
- Create: `shaders/glsl/q6_k_matmul.comp`
- Create: corresponding `shaders/bin/*.spv`
- Modify: `scripts/vulkan-shaders.sh`
- Modify: `shaders/manifest.sha256`
- Modify: `docs/VULKAN.md`

**Interfaces:**
- Extends: `GpuWeightFormat::{Q4_0, Q4_1, Q6_K}`
- Consumes: existing GGML block layouts and CPU scalar decoders as oracle
- Validates: `Qwen3-0.6B-Q4_0.gguf`

- [ ] **Step 1: Add deterministic decoder parity cases**

For each format, generate two complete GGML blocks with edge nibble/scale values, run at least two rows, and compare every GPU output to the corresponding CPU kernel with `abs <= 2e-3 + 2e-3 * abs(cpu)`. Include negative values, zero scale, maximum signed quant, and a non-multiple of workgroup-count row count.

- [ ] **Step 2: Run and observe unsupported-format failures**

Run: `cargo run --release --locked --features vulkan --example vk_ops_check -- --formats q4_0,q4_1,q6_k`

Expected: FAIL with explicit `unsupported Vulkan weight format` for each named format.

- [ ] **Step 3: Implement decoders from the repository's existing layouts**

Use `src/ops/kernel/q4_0/scalar.rs`, `src/ops/kernel/q4_1/scalar.rs`, and `src/ops/quant/q6_k.rs` as the byte-layout oracle. Keep activation quantization and output layout identical to the Q8_0 recorder. Select a pipeline by `Weight::ggml_type`; do not convert model weights to F32 or copy them into `QTensorOwned`.

- [ ] **Step 4: Validate the supplied mixed model**

Run:

```bash
bash scripts/vulkan-shaders.sh update
bash scripts/vulkan-shaders.sh check
cargo run --release --locked --features vulkan --example vk_ops_check -- --formats q4_0,q4_1,q6_k
cargo run --release --locked --features vulkan --example vk_model_check -- \
  qwen3 \
  --model /Users/gouzi/Documents/git/rust-model-inference/models/Qwen3-0.6B/Qwen3-0.6B-Q4_0.gguf
```

Expected: synthetic parity passes, prefill logits meet the documented tolerance, and all 32 greedy tokens match CPU. Output must report Q4_0, Q4_1, and Q6_K as Vulkan formats with no required tensor silently falling back.

- [ ] **Step 5: Commit the format family**

```bash
git add src/vulkan/qwen3.rs src/vulkan/ops.rs examples/vk_ops_check.rs \
  examples/vk_model_check.rs scripts/vulkan-shaders.sh \
  shaders/glsl/q4_0_matmul.comp shaders/glsl/q4_1_matmul.comp \
  shaders/glsl/q6_k_matmul.comp shaders/bin/q4_0_matmul.spv \
  shaders/bin/q4_1_matmul.spv shaders/bin/q6_k_matmul.spv \
  shaders/manifest.sha256 docs/VULKAN.md
git commit -m "feat: add Vulkan Q4_0 model support"
```

### Task 11: Add Q4_K and F16 Qwen3 Paths

**Files:**
- Modify: `src/vulkan/qwen3.rs`
- Modify: `src/vulkan/ops.rs`
- Modify: `examples/vk_ops_check.rs`
- Modify: `examples/vk_model_check.rs`
- Create: `shaders/glsl/q4_k_matmul.comp`
- Create: `shaders/glsl/f16_matmul.comp`
- Create: corresponding `shaders/bin/*.spv`
- Modify: `scripts/vulkan-shaders.sh`
- Modify: `shaders/manifest.sha256`
- Modify: `docs/VULKAN.md`

**Interfaces:**
- Extends: `GpuWeightFormat::{Q4_K, F16}`
- Extends: `Qwen3VulkanSession::forward_hidden_token` for `text_encode`
- Reuses: Q6_K decoder from Task 10
- Validates: `Qwen3-0.6B-Q4_K_M.gguf` and `Qwen3-Embedding-0.6B-f16.gguf`

- [ ] **Step 1: Add Q4_K and F16 synthetic parity**

Build deterministic blocks matching `BLOCK_Q4K_SIZE`/`QK_K`, plus F16 rows containing normal, subnormal, signed zero, and finite maximum values. Compare Q4_K matvec with `abs/rel = 3e-3/3e-3` and F16 with `abs/rel = 2e-4/2e-4`.

- [ ] **Step 2: Run and observe explicit unsupported formats**

Run: `cargo run --release --locked --features vulkan --example vk_ops_check -- --formats q4_k,f16`

Expected: FAIL naming Q4_K and F16.

- [ ] **Step 3: Implement the two decoders without requiring 16-bit storage**

Decode Q4_K according to `src/ops/quant/q4_k.rs`. Load F16 payload as packed `uint`, extract each 16-bit lane, and use the same finite F16 decode used by the baseline shader; this keeps the baseline device contract free of `storageBuffer16BitAccess`. Route qwen3 `text_encode` through `forward_hidden_token` when the complete embedding model is eligible; read back each final normalized hidden row and retain the existing CPU path for every ineligible model.

- [ ] **Step 4: Validate text and embedding models**

Run:

```bash
bash scripts/vulkan-shaders.sh update
bash scripts/vulkan-shaders.sh check
cargo run --release --locked --features vulkan --example vk_model_check -- \
  qwen3 \
  --model /Users/gouzi/Documents/git/rust-model-inference/models/Qwen3-0.6B/Qwen3-0.6B-Q4_K_M.gguf
cargo run --release --locked --features vulkan --example vk_model_check -- \
  embedding \
  --model /Users/gouzi/Documents/git/rust-model-inference/models/qwen-embedding/Qwen3-Embedding-0.6B-f16.gguf
```

For embedding, compare the entire vector with `abs/rel = 2e-3/2e-3` and require identical ordering for the fixed three-text ranking fixture. Report the exact Vulkan format set for each model.

- [ ] **Step 5: Commit each independently validated decoder**

```bash
git add src/vulkan/qwen3.rs src/vulkan/ops.rs src/models/qwen3/trunk/forward.rs \
  examples/vk_ops_check.rs \
  examples/vk_model_check.rs scripts/vulkan-shaders.sh shaders/glsl/q4_k_matmul.comp \
  shaders/bin/q4_k_matmul.spv shaders/manifest.sha256 docs/VULKAN.md
git commit -m "feat: add Vulkan Q4_K model support"

git add src/vulkan/qwen3.rs src/vulkan/ops.rs src/models/qwen3/trunk/forward.rs \
  examples/vk_ops_check.rs \
  examples/vk_model_check.rs scripts/vulkan-shaders.sh shaders/glsl/f16_matmul.comp \
  shaders/bin/f16_matmul.spv shaders/manifest.sha256 docs/VULKAN.md
git commit -m "feat: add Vulkan F16 embedding support"
```

### Task 12: Add BF16 Qwen3.5 and Close the Support Matrix

**Files:**
- Create: `src/vulkan/qwen35.rs`
- Modify: `src/vulkan.rs`
- Modify: `src/vulkan/ops.rs`
- Modify: `src/models/qwen35/trunk/session.rs`
- Modify: `examples/vk_ops_check.rs`
- Modify: `examples/vk_model_check.rs`
- Create: `shaders/glsl/bf16_matmul.comp`
- Create: corresponding `shaders/bin/bf16_matmul.spv`
- Create: `shaders/glsl/q5_k_matmul.comp`
- Create: corresponding `shaders/bin/q5_k_matmul.spv`
- Modify: `scripts/vulkan-shaders.sh`
- Modify: `shaders/manifest.sha256`
- Modify: `docs/VULKAN.md`

**Interfaces:**
- Extends: `GpuWeightFormat::BF16`
- Produces: `Qwen35VulkanSession` only for the operations actually present in `Qwen35Session`
- Validates: `Qwen3.5-0.8B-BF16.gguf`

- [ ] **Step 1: Add BF16 synthetic parity and qwen35 eligibility tests**

Decode BF16 with `f32::from_bits((bits as u32) << 16)` semantics in both test oracle and shader. Include signed zero, subnormal, finite extremes, and mixed-sign dot products. Gate matvec at `abs/rel = 2e-4/2e-4`. The qwen35 test rejects any model operation that lacks a recorder rather than mixing it silently with CPU.

- [ ] **Step 2: Run before BF16 support**

Run: `cargo run --release --locked --features vulkan --example vk_ops_check -- --formats bf16`

Expected: FAIL with `unsupported Vulkan weight format BF16`.

- [ ] **Step 3: Add the BF16 decoder and a qwen35-specific executor**

Keep qwen35 position handling, attention layout, and any architecture-specific normalization in `src/vulkan/qwen35.rs`; reuse only low-level recorders from `ops.rs`. Do not route qwen35 through Qwen3 assumptions. Capture the executor in `Qwen35Session` and use the same token commit/abort and CPU shadow-KV rules as Task 8.

- [ ] **Step 4: Validate the supplied BF16 model**

Run:

```bash
bash scripts/vulkan-shaders.sh update
bash scripts/vulkan-shaders.sh check
cargo run --release --locked --features vulkan --example vk_model_check -- \
  qwen35 \
  --model /Users/gouzi/Documents/git/rust-model-inference/models/qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf
```

Expected: prefill logits meet the declared tolerance, 32 greedy tokens match, one submission occurs per token, and output reports BF16 plus F32 as Vulkan-backed.

- [ ] **Step 5: Commit the independently validated BF16 path**

```bash
git add src/vulkan.rs src/vulkan/ops.rs src/vulkan/qwen35.rs \
  src/models/qwen35/trunk/session.rs examples/vk_ops_check.rs \
  examples/vk_model_check.rs scripts/vulkan-shaders.sh \
  shaders/glsl/bf16_matmul.comp shaders/bin/bf16_matmul.spv \
  shaders/manifest.sha256 docs/VULKAN.md
git commit -m "feat: add Vulkan BF16 inference support"
```

- [ ] **Step 6: Add Q5_K synthetic coverage without an end-to-end claim**

Create `shaders/glsl/q5_k_matmul.comp` from the byte layout in `src/ops/quant/q5_k.rs`, add it to the shader manifest, and add deterministic parity at `abs/rel = 3e-3/3e-3`:

```bash
bash scripts/vulkan-shaders.sh update
cargo run --release --locked --features vulkan --example vk_ops_check -- --formats q5_k
```

The support table labels Q5_K `synthetic kernel parity only`; it remains absent from the end-to-end model matrix until a Q5_K model is supplied.

```bash
git add src/vulkan/ops.rs examples/vk_ops_check.rs scripts/vulkan-shaders.sh \
  shaders/glsl/q5_k_matmul.comp shaders/bin/q5_k_matmul.spv \
  shaders/manifest.sha256 docs/VULKAN.md
git commit -m "feat: add Vulkan Q5_K kernel parity"
```

- [ ] **Step 7: Provide one vendor-neutral hardware acceptance command**

Add a documented command sequence that records `vulkaninfo --summary`, shader validation, five-shape matvec parity, operator parity, model parity, and alternating benchmark medians. Use the same sequence for MoltenVK, Intel ANV, AMD RADV, and NVIDIA; only rows backed by captured output are marked validated.

- [ ] **Step 8: Run the final local matrix and repository checks**

Run:

```bash
bash scripts/vulkan-shaders.sh check
cargo fmt --check
cargo check --locked --features vulkan --lib
cargo check --locked --features vulkan --bin rust-model-inference
cargo check --locked --features vulkan --bin server
cargo check --locked --features vulkan --examples
cargo run --release --locked --features vulkan --example vk_check
cargo run --release --locked --features vulkan --example vk_ops_check
```

Then run `vk_model_check` for the five supplied models named in Tasks 9-12. Run the repository's full test command and report pre-existing `parity_trace`, `q8_0_parallel_matmul`, or `quantized_inference` compilation failures separately rather than describing focused passing checks as a full-suite pass.

- [ ] **Step 9: Commit the final measured support matrix**

```bash
git add docs/VULKAN.md
git commit -m "docs: record Vulkan hardware matrix"
```
