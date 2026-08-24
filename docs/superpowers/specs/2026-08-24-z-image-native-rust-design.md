# Z-Image Turbo Native Rust Support Design

## Status

Approved in chat on 2026-08-24. This design targets exactly the three local GGUF files under `models/z-image-gguf/`:

- `z-image-turbo-q8_0.gguf` — 453 tensors, mixed Q8_0/F16/F32 Z-Image DiT;
- `qwen3_4b_f32-q8_0.gguf` — 398 tensors, Qwen3-4B text encoder with Hugging Face tensor names and no tokenizer metadata;
- `pig_flux_vae_fp32-f16.gguf` — 244 tensors, mixed F16/F32 Flux VAE.

The numerical oracle is [`stable-diffusion.cpp@97d2990807fe6d558e395f8764198d7c7e7b411c`](https://github.com/leejet/stable-diffusion.cpp/tree/97d2990807fe6d558e395f8764198d7c7e7b411c). [`ComfyUI-GGUF@6ea2651e7df66d7585f6ffee804b20e92fb38b8a`](https://github.com/city96/ComfyUI-GGUF/tree/6ea2651e7df66d7585f6ffee804b20e92fb38b8a) is a compatibility reference for mixed GGUF tensor loading and model-role detection, not a numerical oracle.

The shipped inference path is entirely Rust. `stable-diffusion.cpp` is used only by an opt-in development parity harness and is never linked, invoked, or required at runtime.

## Problem

The repository has a provisional `pig` image path, but it cannot generate a valid Z-Image result from these files:

- all three components declare `general.architecture=pig`, so metadata alone cannot identify their roles;
- the text loader expects llama.cpp tensor names and tokenizer metadata that this text GGUF does not contain;
- the current F16 conversion does not decode IEEE 754 half precision correctly, and F16 refiner matrices are passed to Q8_0 math;
- the current positional encoding uses axes `[64, 96, 96]` with sum 256 instead of Z-Image axes `[32, 48, 48]` with sum 128;
- the current DiT forward is incomplete and substitutes zero context when text encoding is unavailable;
- the current VAE evaluates only its input/output convolutions and then uses nearest-neighbor enlargement, leaving most decoder tensors unused;
- VAE load failures are downgraded to warnings, and output is always written to `output.png` regardless of `--out`.

These are correctness failures, not optional quality improvements. The provisional `pig.rs` implementation will be replaced rather than extended.

## Goals

1. Generate a deterministic 512×512 PNG from the three supplied GGUFs using native Rust CPU inference.
2. Match the pinned oracle through tokenization, Qwen3 layer-35 conditioning, eight-step Z-Image Turbo denoising, Flux VAE decoding, and PNG serialization.
3. Reuse the repository's `TensorSource`, mmap loading, `ComputePool`, image encoder, and existing F32/F16/Q8_0 primitives where they are already correct.
4. Keep model-sized weights mmap-backed and bound transient memory with reusable activation buffers.
5. Fail on a missing, swapped, malformed, unsupported, or numerically invalid component instead of producing a fallback image.

## Non-goals

- Z-Image Base, other DiT/text/VAE files, or quantizations other than the exact supplied mixtures.
- GPU, Metal, Vulkan, WGPU, CUDA, Candle, external `sd-cli`, or another inference runtime.
- Img2img, inpainting, reference images, negative prompts, classifier-free guidance, rectangular output, batching, or a server API.
- A generic diffusion framework, general Hugging Face loader, general tokenizer registry, or GGUF conversion tooling.
- Shipping a fourth tokenizer model or requiring tokenizer files beside the GGUFs.
- Performance tuning beyond avoiding model copies and obviously unbounded/repeated allocations.

## CLI Contract

The existing image options remain the public interface, with one new deterministic seed:

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/z-image-gguf/z-image-turbo-q8_0.gguf \
  --text-encoder models/z-image-gguf/qwen3_4b_f32-q8_0.gguf \
  --vae models/z-image-gguf/pig_flux_vae_fp32-f16.gguf \
  --prompt "A red fox sleeping beneath a pine tree" \
  --steps 8 \
  --resolution 512 \
  --seed 42 \
  --out output.png
```

For this model, `--prompt`, `--text-encoder`, `--vae`, and `--out` are required and non-empty. `--steps` defaults to 8 and must be positive. `--resolution` defaults to 512 and must be a positive multiple of 16; only square output is in scope. `--seed` is an explicit signed 64-bit value with a fixed default so omitted and repeated runs are deterministic. `--out` is reused and the result is encoded as PNG.

Z-Image Turbo uses classifier-free guidance scale 1.0, so no negative-context pass or new CFG option is added. Existing unrelated CLI modes retain their current behavior.

CLI combinations and component signatures are validated before constructing model objects. Output is written only after the complete image has been decoded and encoded successfully.

## Component Detection and Loading

`general.architecture=pig` is compatibility metadata shared by all three files and is not a role discriminator. Roles are identified from required tensor signatures:

- text encoder: `model.embed_tokens.weight` plus complete `model.layers.0` through `model.layers.35` projections and norms;
- DiT: `x_embedder.weight`, both refiner families, `layers.0` through `layers.29`, and `final_layer.linear.weight`;
- VAE: `decoder.conv_in.weight`, complete `decoder.mid`, `decoder.up`, `decoder.norm_out`, and `decoder.conv_out` graphs.

Each loader validates the exact fixed architecture, every required tensor, dimensions, and permitted GGML type before inference. Representative fixed dimensions include:

- text: vocabulary 151936, width 2560, 36 layers, query width 4096, key/value width 1024, FFN width 9728;
- DiT: width 3840, 30 layers, 2 context refiners, 2 noise refiners, 30 heads, head width 128, FFN width 10240, 16 latent channels, patch size 2;
- VAE: 16 latent input channels, the supplied 512/256/128 decoder channel graph, three RGB output channels, and spatial factor 8.

The model structs retain their `Arc<dyn TensorSource>`. Matrix and convolution weights are validated byte views into the mmap rather than decoded into `Vec<f32>` or copied into `Vec<u8>`. Small bias/norm vectors may be decoded once when that is simpler and bounded. Weight dispatch follows each tensor's actual F32, F16, or Q8_0 type; type mismatch is an error.

## End-to-End Data Flow

### 1. Tokenization and prompt construction

The text GGUF has no tokenizer metadata. Add the fixed canonical Qwen merge table as a compile-time asset and build the exact byte-level BPE vocabulary in Rust from base byte symbols, ordered merges, and the fixed Qwen special tokens. The asset is approximately 8 MiB and is compiled into the binary; no tokenizer download or sidecar file is required.

Tokenize this exact wrapper with special-token recognition enabled:

```text
<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
```

Tokenizer construction asserts the exact 151936-row ID space. Fixed ASCII, UTF-8 Chinese, and special-token fixtures must match the oracle token-for-token.

### 2. Qwen3 text conditioning

Load the supplied Hugging Face names directly instead of renaming or converting the GGUF. Run the full prompt causally through the fixed 36-layer Qwen3-4B graph using existing Q8_0/F32 CPU kernels and correct Qwen3 RMSNorm, Q/K norm, and RoPE behavior.

Return the hidden state from zero-based transformer layer 35, as required by the pinned Z-Image conditioner, without applying the language-model output head. Its shape is `[token_count, 2560]`. Empty, non-finite, or wrongly shaped context is an error; there is no zero-context fallback.

### 3. Z-Image DiT

For a 512×512 image, create a seeded standard-normal latent with shape `[16, 64, 64]`, patch it at 2×2 into 1024 image tokens, and run:

1. text RMSNorm and projection, image patch projection, and timestep embedding;
2. text and image padding to a multiple of 32 using the learned `cap_pad_token` and `x_pad_token`;
3. two context-refiner blocks over text tokens;
4. two timestep-modulated noise-refiner blocks over image tokens;
5. concatenation followed by all 30 joint transformer blocks;
6. final AdaLN projection, extraction of real image tokens, unpatching, and the reference output sign.

RoPE uses theta 256 and axes `[32, 48, 48]`, whose sum equals the 128-wide head. Positional indices, text/image padding, modulation chunk order, residual order, RMS epsilon, attention scaling, and Q8_0/F16 accumulation follow the pinned oracle.

The Turbo sampler is the pinned reference's discrete flow schedule with flow shift 3.0 and Euler updates. Eight steps are the default. The model timestep transform, sigma sequence, seeded normal stream, update order, and all discrete boundary values are ported exactly rather than approximated with a visually similar scheduler.

### 4. Flux VAE and PNG

Convert the final diffusion latent to VAE space as:

```text
vae_latent = diffusion_latent / 0.3611 + 0.1159
```

Evaluate the complete supplied decoder: input convolution; mid residual block, attention block, and residual block; all decoder up stages with their residual/shortcut blocks and three learned upsample convolutions; output group norm, SiLU, and output convolution. F16 convolution/attention weights are decoded as IEEE 754 half values during the existing native float path, never treated as quantized bytes.

Map the decoder's three channels to PNG bytes with the oracle clamp and rounding order. The expected output is exactly `resolution × resolution × 3`; malformed or non-finite output is an error. The existing `image` dependency encodes PNG into memory first. After encoding succeeds, write a sibling temporary file and rename it over `--out`, preserving any existing output until the complete replacement is ready.

## Memory and Execution Rules

- All three GGUFs remain mmap-backed; loading must not allocate another model-sized copy.
- Each model stage owns a small set of reusable scratch buffers sized from validated dimensions. Layer loops reuse them rather than allocating per layer or per denoising step.
- Text conditioning may be released before VAE decoding except for the final context consumed by the DiT.
- Attention may use the existing CPU/thread-pool implementation. The one-thread path is the parity contract; deterministic multi-thread execution is not claimed until separately measured.
- Checked arithmetic guards shape products, strides, offsets, and allocation sizes before allocation or slicing.
- No cache, trait hierarchy, backend abstraction, or generic graph executor is introduced for this single fixed pipeline.

## Error Handling

- Reject missing CLI values, empty prompts, zero steps, invalid resolution, empty output paths, and conflicting image/audio/TTS modes before model loading.
- Report the detected role and first missing/mismatched tensor when files are swapped or malformed.
- Reject unsupported tensor types and exact dimension mismatches; do not infer a close architecture from filenames.
- Correct the shared F16 decode at its existing operation boundary and test normal, signed, zero, subnormal, infinity, and NaN encodings. Non-finite model activations remain fatal even though the decoder itself supports all valid finite F16 weights.
- Propagate text, DiT, VAE, allocation, and PNG errors. Remove the current warning-and-continue VAE behavior and every synthetic context/image fallback.
- Do not leave a truncated output file on failure.

## Testing and Oracle Parity

Implementation follows red-green-refactor. Fast tests use small in-memory tensor sources and fixed vectors; full model tests are ignored unless explicit model and oracle paths are supplied.

### Fast checks

- component detection rejects `general.architecture`-only matches and accepts only the three tensor signatures;
- IEEE F16 decoding and mixed F32/F16/Q8_0 dispatch;
- exact Qwen byte-BPE token IDs for ASCII, Chinese, whitespace, and ChatML special tokens;
- exact prompt wrapper and layer-35 output selection;
- Z-Image patch/unpatch, padding-to-32, timestep embedding, modulation chunking, and `[32, 48, 48]` RoPE vectors;
- fixed seed/noise sequence and eight-step sigma/timestep schedule;
- VAE group norm, residual/shortcut, attention, learned upsample, and channel-to-byte conversion on non-uniform miniature fixtures;
- CLI validation, output dimensions, deterministic seed behavior, and failure without all three valid components.

### Pinned reference comparison

Build `stable-diffusion.cpp` at `97d2990807fe6d558e395f8764198d7c7e7b411c` in a temporary checkout and apply a small trace patch there. The repository stores only the pin, trace patch, runner, and comparator; it does not vendor the checkout, executables, model files, traces, or generated PNGs.

Run both implementations CPU-only, one thread, 512×512, eight Euler steps, CFG 1.0, with the same fixed prompt and seed. Compare:

1. component roles, token IDs, shapes, sigma/timestep values, and seeded noise sequence exactly;
2. text layer-35 output and selected DiT/VAE intermediate tensors with `max_abs <= 1e-4`;
3. final diffusion latent with `max_abs <= 1e-3`;
4. final PNG channel bytes with absolute delta at most 1 and at least 99.9% exact channel-byte matches.

Trace comparison reports the first differing checkpoint, shape, flat index, Rust value, oracle value, and absolute error. A second prompt with the same seed must produce different text conditioning, final latent, and PNG, proving that prompt conditioning is active.

### Completion gates

- focused unit/integration tests pass;
- the native Rust binary builds on the current host; any unrelated ARM cfg/import baseline repair lands as a separate prerequisite commit and is excluded from Z-Image parity logic;
- the fixed 512×512 eight-step reference run satisfies every tolerance above;
- the second prompt proves prompt sensitivity;
- the PNG is valid, has the requested dimensions, and is written to `--out`;
- `cargo test --all-targets` passes, or unrelated pre-existing failures are reported separately with the focused passing checks.

## File Scope

Expected production changes are limited to:

- `src/app/cli.rs`, `src/main.rs`, and `src/app/image.rs` for seed, validation, dispatch, and `--out`;
- `src/models/diffusion/mod.rs` and `src/lib.rs` for the focused Z-Image exports;
- replacing `src/models/diffusion/pig.rs` with `src/models/diffusion/z_image/{mod.rs,text.rs,dit.rs,vae.rs}`;
- one embedded Qwen merge-table asset;
- focused tests, the opt-in parity harness, and concise README usage.

No GGUFs, generated images, oracle checkout/build, trace outputs, new inference dependency, generic framework, or unrelated refactor is committed.
