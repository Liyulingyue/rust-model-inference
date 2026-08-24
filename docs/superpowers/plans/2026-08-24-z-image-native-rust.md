# Z-Image Turbo Native Rust Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate deterministic 512×512 Z-Image Turbo PNGs from the three supplied GGUFs using a CPU-only, native Rust pipeline.

**Architecture:** Keep the three GGUFs mmap-backed through `TensorSource`; classify their roles by tensor signatures because each declares `general.architecture=pig`. A focused `models::diffusion::z_image` module owns Qwen3 conditioning, Z-Image DiT, Flux VAE, deterministic RNG/sampler, and image result. The CLI validates the complete request, and the app atomically writes `--out`.

**Tech Stack:** Rust stable; existing `TensorSource`/GGUF mmap loader, `ComputePool`, F16/F32/Q8_0 CPU kernels, `image`, `half`, `rayon`, `serde_json`, and `parity-trace`. No new dependency or runtime executable.

**Spec:** `docs/superpowers/specs/2026-08-24-z-image-native-rust-design.md`

## Global Constraints

- Runtime inference is native Rust and CPU-only; `stable-diffusion.cpp` is test-only.
- Support only the supplied Turbo DiT, Qwen3-4B text encoder, and Flux VAE mixtures; no Base, GPU, img2img, additional quantization, or generic framework.
- Keep model-sized matrices/convolutions as `TensorSource` views. Do not clone model bytes or dequantize whole matrices to `Vec<f32>`.
- Reuse `f16_to_f32`, `F16Kernel`, Q8_0 kernels, `ComputePool`, BPE scanner, `image`, and `parity_trace`.
- Defaults: 512×512 square, 8 Euler steps, discrete flow shift 3.0, CFG 1.0, 16 latent channels, 2×2 patches, VAE scale/shift 0.3611/0.1159.
- Errors are fatal: no zero text context, partial VAE, synthetic pixels, or truncated output.
- Oracle: `stable-diffusion.cpp@97d2990807fe6d558e395f8764198d7c7e7b411c`; one CPU thread; exact token/noise/schedule values; intermediate `max_abs <= 1e-4`, final latent `max_abs <= 1e-3`, channel delta <= 1 and >=99.9% exact.
- Preserve `.omo/` and existing untracked `docs/superpowers/` entries; stage explicit paths only.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/app/cli.rs` | Strict `--seed` parsing and complete Z-Image request validation. |
| `src/main.rs` | Explicit Z-Image dispatch before generic `pig` handling. |
| `src/app/image.rs` | Native pipeline orchestration and atomic PNG publication. |
| `src/models/diffusion/{mod.rs,z_image/mod.rs}` | Focused pipeline API, exact component checks, borrowed linear dispatch, seed/schedule helpers. |
| `src/models/diffusion/z_image/{text.rs,qwen_merges.txt}` | Embedded Qwen BPE and fixed Qwen3-4B layer-35 conditioning. |
| `src/models/diffusion/z_image/{dit.rs,vae.rs}` | DiT/sampler and complete decoder-only Flux VAE. |
| `tests/z_image_reference.rs` | Opt-in trace comparison and prompt-sensitivity check. |
| `tools/z_image/{build_stable_diffusion_oracle.sh,stable-diffusion-z-image-trace.patch}` | Pinned read-only oracle builder and trace patch. |
| `src/models/diffusion/pig.rs` | Delete after native route is wired; it has invalid F16, DiT, VAE, and fallback logic. |

```rust
pub struct ZImageOptions { pub steps: usize, pub resolution: usize, pub seed: i64 }
pub struct ZImageRgb { pub width: u32, pub height: u32, pub bytes: Vec<u8> }
pub struct ZImagePipeline {
    dit: ZImageDit,
    text: Qwen3TextEncoder,
    vae: FluxVae,
}

impl ZImagePipeline {
    pub fn load(diffusion: Arc<dyn TensorSource>, text: Arc<dyn TensorSource>, vae: Arc<dyn TensorSource>, pool: Arc<ComputePool>) -> Result<Self, String>;
    pub fn generate_rgb(&self, prompt: &str, options: ZImageOptions) -> Result<ZImageRgb, String>;
}
```

No trait, factory, cache, or configurable backend is introduced.

### Task 1: CLI seed and explicit three-component request

**Files:**
- Modify: `src/app/cli.rs`
- Test: `src/app/cli.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Add `CliOptions::seed: Option<i64>`.
- Add `ZImageCliOptions { steps: usize, resolution: usize, seed: i64, out: PathBuf }`.
- Add `z_image_cli_options(&CliOptions) -> Result<Option<ZImageCliOptions>, String>`; it returns `Some` if either `--text-encoder` or `--vae` is present, which requires both components, a non-empty prompt, valid square resolution divisible by 16, positive steps, and `--out`.

- [ ] **Step 1: Write the failing CLI tests**

```rust
#[test]
fn z_image_cli_requires_all_components_prompt_and_out() {
    let complete = parse_cli_options(&args(&[
        "rmi", "--model", "dit.gguf", "--text-encoder", "text.gguf",
        "--vae", "vae.gguf", "--prompt", "fox", "--out", "fox.png", "--seed", "42",
    ])).unwrap();
    assert_eq!(z_image_cli_options(&complete).unwrap().unwrap().seed, 42);
    for argv in [
        ["rmi", "--model", "dit.gguf", "--text-encoder", "text.gguf"].as_slice(),
        ["rmi", "--model", "dit.gguf", "--vae", "vae.gguf", "--prompt", "fox"].as_slice(),
    ] {
        assert!(z_image_cli_options(&parse_cli_options(&args(argv)).unwrap()).is_err(), "{argv:?}");
    }
}

#[test]
fn seed_requires_a_signed_i64_value() {
    assert!(parse_cli_options(&args(&["rmi", "--seed"])).is_err());
    assert!(parse_cli_options(&args(&["rmi", "--seed", "nan"])).is_err());
}
```

- [ ] **Step 2: Run the tests and confirm the expected missing-symbol failure**

Run: `cargo test --lib app::cli::tests::z_image_cli_requires_all_components_prompt_and_out app::cli::tests::seed_requires_a_signed_i64_value -- --nocapture`

Expected: compile failure because `seed`, `ZImageCliOptions`, and `z_image_cli_options` do not exist.

- [ ] **Step 3: Implement strict parsing and validation**

```rust
pub fn z_image_cli_options(options: &CliOptions) -> Result<Option<ZImageCliOptions>, String> {
    if options.text_encoder.is_none() && options.vae.is_none() { return Ok(None); }
    options.text_encoder.as_ref().filter(|p| !p.as_os_str().is_empty()).ok_or("Z-Image requires --text-encoder")?;
    options.vae.as_ref().filter(|p| !p.as_os_str().is_empty()).ok_or("Z-Image requires --vae")?;
    let out = options.out.clone().filter(|p| !p.as_os_str().is_empty()).ok_or("Z-Image requires --out")?;
    if options.prompt.as_deref().is_none_or(|p| p.trim().is_empty()) { return Err("Z-Image requires a non-empty --prompt".into()); }
    let steps = options.steps.unwrap_or(8);
    let resolution = options.resolution.unwrap_or(512);
    if steps == 0 || resolution == 0 || resolution % 16 != 0 { return Err("Z-Image requires positive --steps and --resolution divisible by 16".into()); }
    Ok(Some(ZImageCliOptions { steps, resolution, seed: options.seed.unwrap_or(0), out }))
}
```

Parse `--seed` with `args.get(i + 1).ok_or("Missing value for --seed")?.parse::<i64>().map_err(|error| format!("Invalid --seed value: {error}"))?`; do not use `unwrap_or`. Reject `--seed` if `z_image_cli_options` returns `None`.

- [ ] **Step 4: Run focused and pre-existing CLI tests**

Run: `cargo test --lib app::cli::tests -- --nocapture`

Expected: PASS; existing TTS, ASR, and normal inference parsing remain unchanged.

- [ ] **Step 5: Commit the CLI contract**

```bash
git add src/app/cli.rs
git commit -m "feat: validate Z-Image CLI inputs"
```

### Task 2: Fixed component signatures and borrowed matrix dispatch

**Files:**
- Create: `src/models/diffusion/z_image/mod.rs`
- Modify: `src/models/diffusion/mod.rs`
- Test: `src/models/diffusion/z_image/mod.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Add `pub(crate) enum Component { Text, Dit, Vae }` and `validate_component(&dyn TensorSource, Component) -> Result<(), String>`.
- Add `Q8Scratch` and `linear_into(source, name, n_in, n_out, input, output, q8, pool) -> Result<(), String>`.
- `linear_into` permits only F16 or Q8_0 model matrices, borrows `tensor_slice(name)` only for the call, and overwrites output.

```rust
pub(crate) struct Q8Scratch {
    values: Vec<u8>,
    scales: Vec<f32>,
}

impl Q8Scratch {
    pub(crate) fn new(n_in: usize) -> Self {
        Self { values: vec![0; n_in], scales: vec![0.0; n_in.div_ceil(32)] }
    }

    fn prepare(&mut self, input: &[f32], n_in: usize) -> Result<(), String> {
        if input.len() != n_in { return Err("Invalid linear input length".into()); }
        self.values.resize(n_in, 0);
        self.scales.resize(n_in.div_ceil(32), 0.0);
        crate::ops::quantize_q8_0_into(input, n_in, &mut self.values, &mut self.scales);
        Ok(())
    }
}
```

- [ ] **Step 1: Write failing signature/F16 tests**

```rust
#[test]
fn pig_metadata_cannot_identify_a_component() {
    let source = TestSource::with_metadata("general.architecture", "pig")
        .with_tensor("x_embedder.weight", &[64, 3840], GGMLType::F16);
    assert!(validate_component(&source, Component::Dit).is_err());
    assert!(validate_component(&source, Component::Text).is_err());
    assert!(validate_component(&source, Component::Vae).is_err());
}

#[test]
fn f16_linear_uses_little_endian_half_values() {
    let source = TestSource::f16_matrix("w", &[2, 2], [1.0, 2.0, 3.0, 4.0]);
    let mut out = [99.0, 99.0];
    linear_into(&source, "w", 2, 2, &[5.0, 6.0], &mut out, &mut Q8Scratch::new(2), &ComputePool::new(1)).unwrap();
    assert_eq!(out, [17.0, 39.0]);
}
```

- [ ] **Step 2: Run the tests and confirm the module is absent**

Run: `cargo test --lib models::diffusion::z_image::tests -- --nocapture`

Expected: compile failure because `z_image` is not exported.

- [ ] **Step 3: Implement exact signatures and zero-copy dispatch**

```rust
pub(crate) fn linear_into(
    source: &dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
    input: &[f32],
    output: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
) -> Result<(), String> {
    let info = source.tensor_info(name).ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != [n_in as u64, n_out as u64] { return Err(format!("Invalid {name} dimensions")); }
    let bytes = source.tensor_slice(name).ok_or_else(|| format!("Missing tensor data: {name}"))?;
    match info.ggml_type {
        GGMLType::F16 => F16Kernel::new(bytes).forward(input, output, n_in, n_out),
        GGMLType::Q8_0 => {
            q8.prepare(input, n_in)?;
            crate::ops::matmul_q8_0_quantized_dynamic(
                bytes, &q8.values, &q8.scales, output, n_in, n_out, pool,
            );
        }
        kind => return Err(format!("Unsupported matrix type {kind:?} for {name}")),
    }
    Ok(())
}
```

Validate all 36 Qwen layers, both 2-layer DiT refiner families plus `layers.0..29`, and the full VAE decoder by exact names/dimensions/types. Use `TensorInfo::checked_nbytes()` and checked products. Keep only matrix names/dimensions and small F32 norm/bias vectors in structs; do not call `to_vec()` on any model-scale tensor.

- [ ] **Step 4: Run focused checks**

Run: `cargo test --lib models::diffusion::z_image::tests -- --nocapture && cargo fmt --check`

Expected: PASS; this establishes the correct shared F16 boundary without reusing `pig.rs::load_f16_as_f32`.

- [ ] **Step 5: Commit model-loading foundations**

```bash
git add src/models/diffusion/mod.rs src/models/diffusion/z_image/mod.rs
git commit -m "feat: add validated Z-Image tensor views"
```

### Task 3: Embedded Qwen tokenizer and layer-35 text conditioning

**Files:**
- Create: `src/models/diffusion/z_image/qwen_merges.txt`
- Create: `src/models/diffusion/z_image/text.rs`
- Modify: `src/core/tokenizer.rs`
- Modify: `src/models/diffusion/z_image/mod.rs`
- Test: `src/core/tokenizer.rs` and `src/models/diffusion/z_image/text.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Add `BPETokenizer::from_qwen3_embedded_merges() -> Result<BPETokenizer, String>`.
- Add `Qwen3TextEncoder::load(Arc<dyn TensorSource>, Arc<ComputePool>) -> Result<Self, String>`.
- Add `Qwen3TextEncoder::encode_layer_35(&self, prompt: &str) -> Result<Vec<f32>, String>` returning row-major `[token_count, 2560]` after block 35 and before an output head.

- [ ] **Step 1: Write failing fixed-ID and prompt-wrapper tests**

```rust
#[test]
fn embedded_qwen3_tokenizer_matches_z_image_chatml() {
    let tokenizer = BPETokenizer::from_qwen3_embedded_merges().unwrap();
    assert_eq!(tokenizer.vocab_size(), 151_936);
    assert_eq!(tokenizer.encode("hello   world", EncodeOptions::default()), vec![14_990, 256, 1_879]);
    assert_eq!(tokenizer.encode(
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        EncodeOptions { add_special: false, parse_special: true },
    ), vec![151_644, 872, 198, 9_707, 151_645, 198, 151_644, 77_091, 198]);
}

#[test]
fn z_image_prompt_enables_special_token_parsing() {
    assert_eq!(z_image_prompt("Hello"), "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n");
}
```

- [ ] **Step 2: Run the tests and confirm the constructor/API is absent**

Run: `cargo test --lib core::tokenizer::tests::embedded_qwen3_tokenizer_matches_z_image_chatml models::diffusion::z_image::text::tests::z_image_prompt_enables_special_token_parsing -- --nocapture`

Expected: compile failure because `from_qwen3_embedded_merges` and `z_image_prompt` do not exist.

- [ ] **Step 3: Add the exact tokenizer and narrow Qwen3 graph**

Copy `src/tokenizers/vocab/qwen_merges.hpp` from `stable-diffusion.cpp@97d2990807fe6d558e395f8764198d7c7e7b411c` into `qwen_merges.txt`, one UTF-8 merge pair per line in oracle order. `from_qwen3_embedded_merges` builds the 256 byte symbols, ordered merge vocabulary, and the oracle `Qwen2Tokenizer` special-token list, assigns sequential IDs, and errors unless the total is exactly 151936. Reuse the existing BPE byte encoder/decoder, `scan_qwen_ranges`, and merge-rank logic; do not add another tokenizer type.

```rust
pub(crate) fn z_image_prompt(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

pub(crate) fn encode_layer_35(&self, prompt: &str) -> Result<Vec<f32>, String> {
    let ids = self.tokenizer.encode(&z_image_prompt(prompt), EncodeOptions { add_special: false, parse_special: true });
    if ids.is_empty() { return Err("Z-Image prompt produced no tokens".into()); }
    self.forward_to_block(&ids, 35)
}
```

Implement only the supplied Qwen3-4B format: Q8_0 `model.embed_tokens.weight`, 36 HF-named blocks, F32 RMS/QK norms, 32 query heads, 8 KV heads, head width 128, causal attention, and SwiGLU FFN width 9728. Reuse `rms_norm`, `rope_neox`, `softmax_inplace`, F16 KV scratch, and Task 2 Q8 dispatch. Reject wrong types, missing layer-35 tensors, empty/non-finite output, and every output length other than `token_count * 2560`.

- [ ] **Step 4: Run tokenizer/text checks**

Run: `cargo test --lib core::tokenizer::tests::embedded_qwen3_tokenizer_matches_z_image_chatml models::diffusion::z_image::text::tests && cargo test --test tokenizer_reference -- --nocapture`

Expected: non-ignored tests PASS; existing ignored llama.cpp vocabulary fixtures remain opt-in.

- [ ] **Step 5: Commit tokenizer and Qwen conditioning**

```bash
git add src/core/tokenizer.rs src/models/diffusion/z_image/mod.rs \
  src/models/diffusion/z_image/text.rs src/models/diffusion/z_image/qwen_merges.txt
git commit -m "feat: add Z-Image Qwen3 conditioning"
```

### Task 4: Deterministic noise, schedule, patching, and three-axis RoPE

**Files:**
- Create: `src/models/diffusion/z_image/dit.rs`
- Modify: `src/models/diffusion/z_image/mod.rs`
- Test: `src/models/diffusion/z_image/dit.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Add `TorchMt19937::new(seed: u64)` and `fill_normal(&mut self, output: &mut [f32])`.
- Add `z_image_sigmas(steps: usize) -> Result<Vec<f32>, String>`, `z_image_rope(text_tokens, image_width, image_height) -> Result<Vec<f32>, String>`, `patchify_latent`, and `unpatchify_latent`.

- [ ] **Step 1: Write failing numeric primitive tests**

```rust
#[test]
fn torch_mt19937_recomputes_the_final_sixteen_values() {
    let mut values = vec![0.0; 20];
    TorchMt19937::new(42).fill_normal(&mut values);
    assert_eq!(values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected_seed_42_20_bits());
}

#[test]
fn discrete_flow_schedule_has_eight_steps_and_a_zero_tail() {
    let sigmas = z_image_sigmas(8).unwrap();
    assert_eq!(sigmas.len(), 9);
    assert_eq!(sigmas[0].to_bits(), time_snr_shift(3.0, 1.0).to_bits());
    assert_eq!(sigmas[8].to_bits(), 0.0f32.to_bits());
}

#[test]
fn z_image_rope_axes_sum_to_the_128_wide_head() {
    assert_eq!(z_image_rope(32, 64, 64).unwrap().len(), (32 + 1024) * 128);
}
```

Use this fixed oracle fixture in `expected_seed_42_20_bits()` before implementation:

```rust
[
    0x3dad_8137, 0x3eae_451f, 0x3f89_ace8, 0xbf97_8007, 0xbe8e_8300,
    0x3e5a_3468, 0xbf0d_d3d3, 0x3e06_f34e, 0xbfa9_6ec8, 0x3f59_e564,
    0x3df6_1a0c, 0xbe68_f632, 0x3f83_62b5, 0xbd11_3ede, 0x3f1e_1f8f,
    0x3f9b_38b7, 0x3ebf_f7bc, 0x3f0b_72d1, 0xbf08_e9c1, 0xbe56_5c99,
]
```

This captures the oracle's overlapping final-16 recomputation and makes a changed normal stream fail before a model run.

- [ ] **Step 2: Run the tests and confirm symbols are absent**

Run: `cargo test --lib models::diffusion::z_image::dit::tests -- --nocapture`

Expected: compile failure because the RNG, schedule, patching, and RoPE helpers do not exist.

- [ ] **Step 3: Port the bounded oracle primitives exactly**

```rust
pub(crate) fn time_snr_shift(alpha: f32, t: f32) -> f32 {
    if alpha == 1.0 { t } else { alpha * t / (1.0 + (alpha - 1.0) * t) }
}

pub(crate) fn z_image_sigmas(steps: usize) -> Result<Vec<f32>, String> {
    if steps == 0 { return Err("Z-Image steps must be positive".into()); }
    if steps == 1 { return Ok(vec![time_snr_shift(3.0, 1.0), 0.0]); }
    let stride = 999.0 / (steps - 1) as f32;
    let mut result = (0..steps).map(|i| time_snr_shift(3.0, (1000.0 - stride * i as f32) / 1000.0)).collect::<Vec<_>>();
    result.push(0.0);
    Ok(result)
}
```

Port `MT19937RNG::randn` from the pin: 624-word state, uint64 high/low draw ordering, `uniform_real`, 16-value Torch normal transform, tail recomputation, and short-vector double fallback. Do not use `rand`, `rand_distr`, `StdRng`, or a different Box–Muller order. Generate theta-256 positions over axes `[32, 48, 48]`; pad text/image rows independently to 32 before concatenating positions.

- [ ] **Step 4: Run primitives and 512 latent shape round-trip**

Run: `cargo test --lib models::diffusion::z_image::dit::tests -- --nocapture && cargo fmt --check`

Expected: PASS, including `[16, 64, 64] <-> [1024, 64]` without spatial transpose.

- [ ] **Step 5: Commit deterministic primitives**

```bash
git add src/models/diffusion/z_image/mod.rs src/models/diffusion/z_image/dit.rs
git commit -m "feat: add deterministic Z-Image sampling primitives"
```

### Task 5: Mixed-precision Z-Image DiT forward and Euler denoising

**Files:**
- Modify: `src/models/diffusion/z_image/{mod.rs,dit.rs}`
- Test: `src/models/diffusion/z_image/dit.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Add `ZImageDit::load(Arc<dyn TensorSource>, Arc<ComputePool>) -> Result<Self, String>`.
- Add `ZImageDit::predict_flow(&self, latent, latent_side, context, context_tokens, sigma, scratch) -> Result<(), String>`.
- Add `ZImageDit::denoise(&self, context, context_tokens, &ZImageOptions) -> Result<Vec<f32>, String>`.

- [ ] **Step 1: Write failing padding and Euler-step tests**

```rust
#[test]
fn padding_uses_the_learned_token_not_zero_context() {
    let mut values = vec![1.0; 31 * 3840];
    pad_rows_to_32(&mut values, 31, &[2.0; 3840]).unwrap();
    assert_eq!(&values[31 * 3840..32 * 3840], &[2.0; 3840]);
}

#[test]
fn euler_flow_update_consumes_the_next_sigma_once() {
    let mut latent = [2.0, -3.0];
    euler_flow_step(&mut latent, &[0.5, -0.25], 0.9, 0.4);
    assert_eq!(latent, [1.75, -2.875]);
}
```

- [ ] **Step 2: Run the tests and confirm the APIs are absent**

Run: `cargo test --lib models::diffusion::z_image::dit::tests::padding_uses_the_learned_token_not_zero_context models::diffusion::z_image::dit::tests::euler_flow_update_consumes_the_next_sigma_once -- --nocapture`

Expected: compile failure because `pad_rows_to_32` and `euler_flow_step` do not exist.

- [ ] **Step 3: Implement the fixed DiT order**

```rust
pub(crate) fn euler_flow_step(latent: &mut [f32], velocity: &[f32], sigma: f32, sigma_next: f32) {
    for (x, v) in latent.iter_mut().zip(velocity) {
        *x += (sigma_next - sigma) * *v;
    }
}
```

Load F16 `x_embedder`, `cap_embedder.1`, learned pad tokens, refiner QKV/out/FFN/AdaLN weights, and final matrices; load Q8_0 main `layers.0..29` matrices; load F32 biases/RMS/QK scales. Execute: text RMSNorm/project, image patch projection, timestep MLP, independent pad-to-32, context refiner 0..1, noise refiner 0..1, concatenate, joint layer 0..29, final AdaLN/linear, remove padded rows, unpatchify, and apply the reference output sign.

For attention, form row-major Q/K/V, apply per-head RMS Q/K norm and Task 4 RoPE, scale scores by `1/sqrt(128)`, softmax per query row, project values, then apply the reference AdaLN/SwiGLU residual order. `DitScratch` owns reusable token, QKV, attention, FFN, quantization, and output buffers; allocation inside the 30-layer loop is forbidden. For each adjacent sigma pair, record the sigma/timestep then call `predict_flow` and `euler_flow_step`.

- [ ] **Step 4: Run fast and supplied-model DiT load checks**

Run: `cargo test --lib models::diffusion::z_image::dit::tests -- --nocapture && Z_IMAGE_DIT=models/z-image-gguf/z-image-turbo-q8_0.gguf cargo test --lib models::diffusion::z_image::dit::tests::dit_loader_accepts_supplied_tensor_signature -- --ignored --nocapture`

Expected: PASS. The ignored check opens/validates the supplied DiT without running image generation.

- [ ] **Step 5: Commit DiT execution**

```bash
git add src/models/diffusion/z_image/mod.rs src/models/diffusion/z_image/dit.rs
git commit -m "feat: run Z-Image Turbo DiT in Rust"
```

### Task 6: Complete decoder-only Flux VAE

**Files:**
- Create: `src/models/diffusion/z_image/vae.rs`
- Modify: `src/models/diffusion/z_image/mod.rs`
- Test: `src/models/diffusion/z_image/vae.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Add `FluxVae::load(Arc<dyn TensorSource>) -> Result<Self, String>`.
- Add `FluxVae::decode_rgb(&self, diffusion_latent: &[f32], latent_side: usize) -> Result<ZImageRgb, String>`.
- `decode_rgb` requires 16 channels and maps values with `value / 0.3611 + 0.1159` before decoding.

- [ ] **Step 1: Write failing VAE primitive/validation tests**

```rust
#[test]
fn diffusion_latent_uses_flux_scale_and_shift() {
    assert_eq!(diffusion_to_vae(0.3611), 1.1159);
}

#[test]
fn learned_upsample_is_nearest_then_padded_conv() {
    let output = upsample_nearest_then_conv(&[1.0, 2.0, 3.0, 4.0], 1, 2, &identity_center_kernel(), None).unwrap();
    assert_eq!(output, vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0]);
}

#[test]
fn missing_mid_attention_is_a_load_error() {
    assert!(FluxVae::load(Arc::new(decoder_source_without("decoder.mid.attn_1.q.weight"))).is_err());
}
```

- [ ] **Step 2: Run the tests and confirm the decoder API is absent**

Run: `cargo test --lib models::diffusion::z_image::vae::tests -- --nocapture`

Expected: compile failure because `FluxVae`, `diffusion_to_vae`, and learned upsample helpers do not exist.

- [ ] **Step 3: Implement every decoder stage**

```rust
pub(crate) fn diffusion_to_vae(value: f32) -> f32 { value / 0.3611 + 0.1159 }
pub(crate) fn to_rgb_byte(value: f32) -> u8 {
    (((value.clamp(-1.0, 1.0) + 1.0) * 127.5).round()).clamp(0.0, 255.0) as u8
}
```

Implement padded stride-1 F16 convolution, GroupNorm(32, epsilon `1e-6`), SiLU, residual/shortcut blocks, one-head spatial attention at `decoder.mid.attn_1`, and nearest-neighbor then learned-convolution upsample. Load `conv_in`; `mid.block_1`, `mid.attn_1`, `mid.block_2`; `up.3` through `up.0` in decode order; `norm_out`; and `conv_out`. Borrow all F16 convolution/attention bytes during execution and copy only small F32 bias/norm vectors. `VaeScratch` retains two feature maps per resolution; the old nearest-only `pig.rs` path is not called.

- [ ] **Step 4: Run fast VAE tests and supplied-model load check**

Run: `cargo test --lib models::diffusion::z_image::vae::tests -- --nocapture && Z_IMAGE_VAE=models/z-image-gguf/pig_flux_vae_fp32-f16.gguf cargo test --lib models::diffusion::z_image::vae::tests::flux_vae_loader_accepts_complete_supplied_decoder -- --ignored --nocapture`

Expected: PASS. The ignored check validates all decoder tensors/types without generating an image.

- [ ] **Step 5: Commit the Flux VAE**

```bash
git add src/models/diffusion/z_image/mod.rs src/models/diffusion/z_image/vae.rs
git commit -m "feat: decode Z-Image latents with Flux VAE"
```

### Task 7: Native pipeline route, atomic PNG, and removal of `pig.rs`

**Files:**
- Modify: `src/{lib.rs,main.rs}`
- Modify: `src/app/{image.rs,mod.rs}`
- Modify: `src/models/diffusion/mod.rs`
- Delete: `src/models/diffusion/pig.rs`
- Test: `src/app/image.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Add `run_z_image_cli(diffusion, text, vae, prompt, ZImageCliOptions, n_threads) -> Result<(), String>`.
- Add `write_png_atomically(path: &Path, image: &ZImageRgb) -> Result<(), String>`.
- `main.rs` calls `z_image_cli_options` before generic dispatch; `Some` requires `Component::Dit` for `--model` and passes all three sources to `run_z_image_cli`.

- [ ] **Step 1: Write failing router and publication tests**

```rust
#[test]
fn failed_png_encoding_preserves_the_existing_output() {
    let dir = test_temp_dir();
    let output = dir.join("image.png");
    std::fs::write(&output, b"old").unwrap();
    let invalid = ZImageRgb { width: 2, height: 2, bytes: vec![0; 11] };
    assert!(write_png_atomically(&output, &invalid).is_err());
    assert_eq!(std::fs::read(&output).unwrap(), b"old");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn text_signature_cannot_be_dispatched_as_a_dit() {
    assert!(run_z_image_cli(Arc::new(qwen_signature_source()), valid_text(), valid_vae(), "fox", valid_options(), 1).is_err());
}
```

`test_temp_dir()` uses `std::env::temp_dir()` plus PID and `line!()`, creates exactly that directory, and removes it at test end; do not add the `tempfile` crate.

- [ ] **Step 2: Run tests and confirm missing route/writer APIs**

Run: `cargo test --lib app::image::tests -- --nocapture`

Expected: compile failure because `run_z_image_cli` and `write_png_atomically` do not exist.

- [ ] **Step 3: Wire the native path and safe output**

```rust
pub fn write_png_atomically(path: &Path, rgb: &ZImageRgb) -> Result<(), String> {
    let expected = usize::try_from(rgb.width).ok()
        .and_then(|w| usize::try_from(rgb.height).ok().and_then(|h| w.checked_mul(h)))
        .and_then(|pixels| pixels.checked_mul(3)).ok_or("Z-Image RGB size overflow")?;
    if rgb.bytes.len() != expected { return Err("Invalid Z-Image RGB length".into()); }
    let image = image::RgbImage::from_raw(rgb.width, rgb.height, rgb.bytes.clone()).ok_or("Invalid RGB image")?;
    let mut encoded = Vec::new();
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut std::io::Cursor::new(&mut encoded), image::ImageFormat::Png)
        .map_err(|e| format!("Encode PNG: {e}"))?;
    write_sibling_temp_then_rename(path, &encoded)
}
```

`write_sibling_temp_then_rename` creates a same-directory temp file through `OpenOptions::create_new(true)` with bounded PID/counter suffixes, writes plus `sync_all`, then renames. It only removes the temp path it created on error. Delete both old `arch == "pig"` branches, `Pig*` exports, and `pig.rs`; omitted components now fail rather than generating zero-context/fallback pixels.

- [ ] **Step 4: Run focused tests and a real functional smoke**

Run:

```bash
cargo test --lib app::image::tests app::cli::tests -- --nocapture
cargo run --release --bin rust-model-inference -- \
  --model models/z-image-gguf/z-image-turbo-q8_0.gguf \
  --text-encoder models/z-image-gguf/qwen3_4b_f32-q8_0.gguf \
  --vae models/z-image-gguf/pig_flux_vae_fp32-f16.gguf \
  --prompt "A red fox sleeping beneath a pine tree" \
  --steps 8 --resolution 512 --seed 42 --threads 1 --out /tmp/rmi-z-image-42.png
file /tmp/rmi-z-image-42.png
```

Expected: tests PASS and `file` reports 512×512 PNG. This is functional evidence only, not an oracle parity claim.

- [ ] **Step 5: Commit the user-visible path**

```bash
git add src/lib.rs src/main.rs src/app/cli.rs src/app/image.rs src/app/mod.rs \
  src/models/diffusion/mod.rs src/models/diffusion/z_image
git rm -- src/models/diffusion/pig.rs
git commit -m "feat: generate Z-Image PNGs in native Rust"
```

### Task 8: Pinned oracle trace, model-backed parity, and README

**Files:**
- Create: `tools/z_image/build_stable_diffusion_oracle.sh`
- Create: `tools/z_image/stable-diffusion-z-image-trace.patch`
- Create: `tests/z_image_reference.rs`
- Modify: `README.md`

**Interfaces:**
- Oracle builder accepts one existing stable-diffusion.cpp checkout, clones its `origin` to `mktemp -d`, detaches `97d2990807fe6d558e395f8764198d7c7e7b411c`, applies the patch, builds the `sd-cli` target CPU-only, and prints `$clone/build/bin/sd-cli`.
- Ignored integration test requires `Z_IMAGE_DIT`, `Z_IMAGE_TEXT`, `Z_IMAGE_VAE`, `Z_IMAGE_ORACLE_BIN`, and `Z_IMAGE_ORACLE_TRACE`.

- [ ] **Step 1: Write the ignored end-to-end test before its trace patch**

```rust
#[test]
#[ignore = "requires Z_IMAGE_DIT, Z_IMAGE_TEXT, Z_IMAGE_VAE, Z_IMAGE_ORACLE_BIN, and Z_IMAGE_ORACLE_TRACE"]
fn z_image_matches_pinned_oracle_and_changes_with_prompt() {
    let first = run_fixture(0, "A red fox sleeping beneath a pine tree");
    let second = run_fixture(1, "A blue ceramic fox beneath a pine tree");
    assert_ne!(first.text_layer_35, second.text_layer_35);
    assert_ne!(first.final_latent, second.final_latent);
    assert_ne!(first.rgb, second.rgb);
}
```

`run_fixture` invokes both binaries with `--steps 8 --resolution 512 --seed 42 --threads 1`, reads JSONL and little-endian F32 sidecars, compares token/noise/sigma/timestep exactly, checks intermediates at `1e-4`, final latent at `1e-3`, and computes RGB channel delta/exact-byte rate.

- [ ] **Step 2: Run without its environment and verify the safe boundary**

Run: `cargo test --release --features parity-trace --test z_image_reference -- --ignored --nocapture`

Expected: it reports missing explicit environment inputs or exits before comparison. It must not download, clean, or modify a user checkout.

- [ ] **Step 3: Implement a read-only oracle builder and trace patch**

```bash
#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 STABLE_DIFFUSION_CPP_CHECKOUT" >&2; exit 2; }
source_dir=$(cd "$1" && pwd -P)
git -C "$source_dir" rev-parse --git-dir >/dev/null
origin=$(git -C "$source_dir" remote get-url origin)
root=$(mktemp -d "${TMPDIR:-/tmp}/rmi-z-image-oracle.XXXXXX")
clone="$root/stable-diffusion.cpp"
git clone --no-checkout "$origin" "$clone" >&2
git -C "$clone" fetch origin 97d2990807fe6d558e395f8764198d7c7e7b411c >&2
git -C "$clone" checkout --detach 97d2990807fe6d558e395f8764198d7c7e7b411c >&2
git -C "$clone" apply --check "$(dirname "$0")/stable-diffusion-z-image-trace.patch" >&2
git -C "$clone" apply "$(dirname "$0")/stable-diffusion-z-image-trace.patch" >&2
```

Finish the script with `cmake -S "$clone" -B "$clone/build" -DSD_METAL=OFF -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DSD_BUILD_EXAMPLES=ON`, then `cmake --build "$clone/build" --target sd-cli --config Release`. Set `bin="$clone/build/bin/sd-cli"`, fail unless it is executable, and print exactly `$bin`. The patch writes prompt IDs, initial noise, sigmas/timesteps, Qwen block 35, selected DiT prelude/refiner/layer outputs, final latent, selected VAE stages, and raw RGB channels. Reuse JSONL/sidecar schema from `tests/qwen3_tts_reference.rs` and `parity_trace`; do not create another trace format.

- [ ] **Step 4: Run full verification**

Run:

```bash
cargo fmt --check
cargo test --lib models::diffusion::z_image::tests -- --nocapture
cargo test --lib app::cli::tests app::image::tests -- --nocapture
cargo test --release --features parity-trace --test z_image_reference z_image_matches_pinned_oracle_and_changes_with_prompt -- --ignored --nocapture
cargo build --release --bin rust-model-inference
```

Expected: fast tests and release build PASS; parity reports every required threshold. Report unrelated `cargo test --all-targets` baseline errors separately instead of changing unrelated code.

- [ ] **Step 5: Add concise usage documentation and commit**

```bash
git add tools/z_image tests/z_image_reference.rs README.md
git commit -m "test: verify Z-Image against pinned oracle"
```

README contains one native three-GGUF command, `--seed`, `--out`, 512×512 CPU scope, and unsupported Base/GPU/img2img modes.

## Plan Self-Review

| Spec requirement | Plan task |
| --- | --- |
| Native CPU Rust, no new runtime dependency | Global Constraints; Tasks 2-8 |
| Tensor-signature roles and strict types | Task 2 |
| Embedded Qwen BPE and layer-35 conditioning | Task 3 |
| Exact RNG/schedule/RoPE/padding/DiT | Tasks 4-5 |
| Complete Flux VAE and correct F16 | Task 6 |
| CLI seed/components/output and no fallbacks | Tasks 1 and 7 |
| mmap weights and scratch reuse | Global Constraints; Tasks 2, 5, and 6 |
| Pinned oracle thresholds and prompt sensitivity | Task 8 |
| Removal of provisional Pig path | Task 7 |

The only model-scale literal not printed inline is the 8.36 MiB merge table; Task 3 names its source file, commit, normalization, order, and vocabulary-size check. All introduced types are defined before their consumers: `ZImageOptions`/`ZImageRgb` feed DiT/VAE/app, and `ZImageCliOptions` feeds only `run_z_image_cli`.
