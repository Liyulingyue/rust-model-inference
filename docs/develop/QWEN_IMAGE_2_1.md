# Qwen-Image-2.1

The main CLI supports CPU text-to-image, single/multiple reference-image editing,
positive/negative Qwen3-VL conditioning, CFG, the Qwen-Image-2.1 flow schedule,
Euler sampling, dedicated VAE encoding/decoding and RGBA PNG output. The original
DiT-only F32 velocity interface remains available.

## Components

| Component | File | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| DiT | `qwen-image-2.1-Q8_0.gguf` | 7,640,860,384 | `c0ed4b2ffd56cbe9c3df1e4a4098045256484ebe94ba5a7e4338d35ec046baa5` |
| Text | `Qwen3VL-8B-Instruct-Q8_0.gguf` | 8,709,519,456 | `0d264b3941185d00a74f75c4245521dae088ff1efc90ab8d1754e83f5844adb0` |
| Vision | `mmproj-Qwen3VL-8B-Instruct-F16.gguf` | 1,159,029,824 | `ca524100ebf825c9a870db1c580d03879e0da0ab2541697e2458e64891cf9d38` |
| VAE | `Qwen-Image-2.1/vae/diffusion_pytorch_model.safetensors` | 1,350,989,512 | `a07a1b7c4ee2966a1b3bdc37de9b4f983d56937e46619f709a80b6e490675417` |
| VAE config | sibling `config.json` | 2,079 | `9785d527b278cb8b210e9f48a8028d92a4190968ce7e24c8fc85d9d1b82f6ba2` |

The DiT GGUF has no metadata; detection requires the complete architecture's
tensor signature. Its 265 tensors comprise 196 Q8_0, 65 F32 and four BF16 tensors,
with 32 transformer layers, width 4096, head dimension 128 and 64 latent channels.
The text encoder is explicitly registered Qwen3-VL-8B (36 layers, width 4096,
32 query/8 KV heads, MRoPE). Editing needs its 27-layer F16 vision projector and
three DeepStack levels. The `AutoencoderKLQwenImage21` VAE uses 64 channels,
16× spatial compression, encoder/decoder base widths 96/144 and RGBA images.
Older Wan/Qwen VAEs are rejected.

## CLI

```bash
cargo build --profile release-fast --bin rust-model-inference
RMI_SCALAR=1 target/release-fast/rust-model-inference \
  --model /path/to/qwen-image-2.1-Q8_0.gguf \
  --text-encoder models/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q8_0.gguf \
  --vae models/Qwen-Image-2.1/vae/diffusion_pytorch_model.safetensors \
  --prompt "A blue ceramic cat figurine on a wooden table." \
  --width 512 --height 512 --steps 30 --cfg 6 --seed 42 \
  --threads 8 --out cat.png
```

`RMI_SCALAR=1` selects the verified CPU scalar arithmetic in an ordinary build;
no trace feature or trace files are needed for inference. Omit it to use the
existing CPU SIMD kernels. SIMD execution works, but does not have bitwise
parity with the pinned Oracle; see the precision boundary below.

For editing, add `--mmproj .../mmproj-Qwen3VL-8B-Instruct-F16.gguf --image first.png`;
repeat `--reference another.png` for additional references. References preserve
their aspect ratio up to rounding each dimension to the nearest positive multiple
of 32. The VAE encodes all four channels; vision receives RGB composited on white.
References occupy image slots in the text sequence and are replaced with VAE
latent tokens in the DiT. Each segment attends its own and preceding segments;
reference/text prefixes use zero-timestep modulation.

Defaults: 512×512, 30 steps, CFG 6 and seed 42. Dimensions must be positive
multiples of 32. Tokenized conditioning is limited to 2048 tokens; aggregate reference
vision tokens are capped at 2000 and the completed prompt is checked separately.
CFG must be finite and at least one; steps are limited to 1–1000. `--negative-prompt`
is optional. `--noise file.f32` supplies fixed planar noise instead of CPU MT19937.
An existing output is protected unless `--overwrite` is set. Writing uses the
existing atomic output helper. GPU execution is rejected explicitly.

For explicit context generation use `--qwen-context-file context.f32 --cfg 1`,
`--vae`, image dimensions and a PNG output; omit prompts and the text encoder.
For the DiT-only interface use `--model ... --out velocity.bin` with optional
`--qwen-latent-file`, `--qwen-context-file`, `--qwen-latent-width`,
`--qwen-latent-height` and `--qwen-timestep`. Missing raw inputs use the existing
explicitly announced synthetic fixture.

## Verification

See the [reproducible full-pipeline runner](../../tools/oracle/qwen_image_2_1/README.md).
The Oracle is pinned stable-diffusion.cpp `2f886889e6e8b78738d6b87f7191f6018557c551`
and ggml `4bf5f6000653b7881d00963cd6ddb665ccd62a8d`, built in an isolated copy with
no external acceleration libraries. Scalar validation disables SIMD, FMA,
vector reductions and fused trig on both sides and compares every selected F32
as raw `u32` bits. It also checks actual tokenizer IDs and PNG pixels.

### Full scalar pipeline

Local evidence is retained in `target/qi21-validation-v10/summary.json` and its
checkpoint sidecars. All three cases used the actual main CLI, the component
hashes above, 64×32 RGBA output, three Euler steps, CFG 3.25, seed 42 and eight
CPU threads.

| Case | References | Checkpoints | F32 raw-bit comparisons | Rust seconds |
| --- | ---: | ---: | ---: | ---: |
| Text-to-image | 0 | 2,205 | 180,454,723 | 353.876 |
| Single-image edit | 1 | 2,688 | 260,759,666 | 748.797 |
| Multi-image edit | 2 | 3,169 | 393,831,840 | 1,166.149 |
| Total | | **8,062** | **835,046,229** | |

Every comparison passed with zero differing bits. The final PNG's 8,192 decoded
RGBA8 bytes also matched the independently computed Oracle pixels in each case.
This includes actual tokenizer IDs and all 36 raw text layers, all 27 vision
layers and three DeepStack levels, complete VAE encode/decode conv/norm/attention
checkpoints, all 32 DiT layers on every positive/negative evaluation, MT19937
noise, sigmas, CFG velocities and every Euler latent. Reference preprocessing is
also checked against independently normalized fixture pixels. The reference
images are 64×32 and 32×64 with alpha 160/255; cases include Chinese, emoji, an
empty negative prompt and different reference orientations.

Reproduce with the [pipeline runner](../../tools/oracle/qwen_image_2_1/README.md).
The runner rejects a missing checkpoint, a changed shape or any differing bit;
single-bit corruption and a removed checkpoint were checked to fail comparison.
The trace patch was also applied and compiled from a fresh pinned source copy.

The ordinary build was then run without tracing using `RMI_SCALAR=1` and 12
threads. All three cases again matched every RGBA8 pixel of the verified scalar
results (8,192 bytes each). These runs took 22.226, 22.045 and 23.863 seconds;
their commands, logs and hashes are in `target/qi21-production-v12/summary.json`.
This also checks that the scalar mode works without the `parity-trace` feature
and remains deterministic when the worker count changes from eight to twelve.

A final run with the current trace build and freshly rebuilt pinned Oracle
also checked the CFG=1, one-step path: 467 checkpoints and 46,864,676 F32 bits
matched, followed by all 8,192 PNG pixels. The runner's ordinary-build repeat
matched too. Evidence: `target/qi21-runner-v12/summary.json`; this exercises
`--production-rust-bin` without repeating the earlier three-case Oracle runs.

### Normal sampling smoke checks

The ordinary build also completed real 30-step generation and editing with
CFG 6, seed 42 and 12 CPU threads:

| Result | Size | Wall seconds | PNG SHA-256 |
| --- | --- | ---: | --- |
| Blue ceramic cat on a wooden table | 256×256 | 857.19 | `2396c9431ce51ff7fc116fea1b73dc4ddfa76109b92c4ab733f7e43c932416d2` |
| Reference edited to a red ceramic cat | 128×128 | 356.86 | `73b663283a3343cb1ab5804604aa197e95c3bb27e56773f2d7f966a71b30aef8` |

Files: `target/qi21-validation-v10/cat-256-30.png` and
`cat-red-edit-128-30.png`. Both were visually inspected: the generation follows
the prompt, and the edit changes blue to red while retaining the cat's shape,
wooden table and window light. Their alpha ranges are 249–255 and 253–255.
These two SIMD smoke checks establish usable images, not bitwise precision.

Hardware: Apple M3 Max, 16 CPU cores (12 performance/4 efficiency), 64 GB RAM,
macOS 27.0.1, Rust 1.98.1. Compiling and other verification ran concurrently,
so these timings are observations rather than uncontended performance benchmarks.

### Precision and engineering boundaries

Bitwise validation covers the pinned GGUF components' CPU **scalar** path. It
does not establish equivalence to unquantized original weights, other quantization
types, GPU execution or every image resolution. GPU is explicitly unsupported.
An independently optimized Oracle comparison failed at text block 0; diagnostic
traces show identical `attn_norm-0` inputs and the first differing raw Q/K
projections. At Q element 1, Rust is `0x3c29938e` and the Oracle `0x3c29938c`;
at K element 0 they are `0xbd6dc573` and `0xbd6dc572`. The SIMD accumulation
orders differ: reusing the existing Q8 NRC1 kernel for that diagnostic projection
matched all 135,168 Q values and 33,792 K values against the optimized Oracle;
the normal NRC4 dispatch did not. This isolates the projection difference, not
the rest of the optimized pipeline. SIMD bitwise parity is not claimed. Use
`RMI_SCALAR=1` when the verified numerical contract is required.

Focused model, pipeline, VAE and vision tests, ordinary/trace builds, formatting,
input rejection and output protection checks passed. The two integration targets
contain eight passing checks. The full ordinary-build library suite was also
run, followed by the same command on an isolated archive of original HEAD
`ce17cd5`:

| Source | Passed | Failed | Ignored |
| --- | ---: | ---: | ---: |
| Current implementation | 1,074 | 24 | 75 |
| Original HEAD | 1,072 | 24 | 74 |

Both runs failed on exactly the same 24 tests below. No new failing test was
observed. Logs are retained as `target/qi21-production-v12/full-lib-current.log`
and `full-lib-baseline.log`. The full suite does not pass; the isolated RoPE
failure also reproduces the same one-bit difference at the original HEAD.

```text
models::breeze::tests::main_matrices_require_original_dtype_exact_shape_and_length
models::breeze::tests::main_source_preserves_original_bf16_and_legacy_codec_f32
models::breeze::tests::main_source_rejects_non_bf16_heads_eoi_and_norms
models::clm::tests::matches_reference_golden_vectors
models::clm::tests::rejects_wrong_embedding_width
models::diffusion::z_image::text::tests::qwen_attention_softmax_matches_the_pinned_ggml_neon_reduction
models::diffusion::z_image::text::tests::silu_mul_inplace_matches_pinned_ggml_neon_activation
models::diffusion::z_image::vae::tests::silu_matches_pinned_ggml_neon_vector_path
models::funasr::encoder::parity_tests::fsmn_rounds_product_before_adding_like_ggml
models::funasr::encoder::parity_tests::layernorm_accumulates_variance_in_ggml_neon_groups
models::gemma4::trunk::tests::failed_later_gemma4_chunk_preserves_successful_prefix
models::gemma4::trunk::tests::per_layer_bf16_projection_matches_pinned_scalar_dot_bits
models::gemma4::trunk::tests::per_layer_projection_rejects_non_bf16_weight
models::gemma4::trunk::tests::per_layer_projection_rejects_wrong_bf16_storage_length
models::qwen35::trunk::config::tests::qwen35_config_rejects_invalid_tensor_dimensions
models::qwen35::vision::tests::layer_norm_stats_match_ggml_grouped_f32_variance
ops::argmax::tests::all_negative
ops::argmax::tests::large_random
ops::argmax::tests::middle_max
ops::argmax::tests::tail_handles_non_aligned_lengths
ops::argmax::tests::ties_pick_lowest_index
ops::matmul::neon_tests::neon_softmax_matches_ggml_vector_exp_and_f64_sum
ops::matmul::neon_tests::prepared_group_matches_mixed_format_sequential_bits
ops::rope::tests::rope_neox_inplace_simd_matches_scalar_fallback
```
