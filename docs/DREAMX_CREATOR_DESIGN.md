# DreamX-Creator GGUF Export and Native Rust CPU Inference

Date: 2026-09-07

## Goal

Add complete native CPU inference for the released DreamX-Creator pipeline:

1. The 7B joint image-to-audio-video generator.
2. The 5B causal 2x video refiner.
3. Every released refiner checkpoint path: Flash latent upsampling, causal 2D
   latent upsampling, the full Wan2.2 VAE, and LightVAE-NU decoding.

The runtime must not depend on llama.cpp, Python, libtorch, OpenBLAS, BLIS,
oneDNN, Accelerate, MKL, or another external compute backend. Python and
PyTorch are allowed only in the offline exporter because the source `.pt`
checkpoints use PyTorch serialization.

Upstream references:

- https://github.com/AMAP-ML/DreamX-Creator
- https://modelscope.cn/models/GD-ML/DreamX-Creator

## Scope

The first target is AArch64 CPU on the current 64 GiB development machine.
The implementation also retains a portable scalar path. GPU execution,
training, arbitrary external-video refinement, and sequence-parallel inference
are outside this change.

Functional output is a synchronized MP4 plus its intermediate video-only MP4
and WAV. `ffmpeg` remains an external packaging tool; it is not part of model
inference.

The official 5-second and 2K settings remain available, but initial automated
acceptance uses a short, low-resolution, reduced-step run. Full official CPU
runtime is reported rather than used as a test timeout.

## Artifact Layout

The exporter writes one matched GGUF pair.

### Main GGUF

Default name: `DreamX-Creator-Q8_0.gguf`.

It contains the compute-heavy denoisers:

- Creator video DiT.
- Creator audio DiT.
- Layers 15-29 A2V and V2A gated cross-attention weights.
- The 5B SR-DiT.

The GGUF architecture is `dreamx`. Tensor names use stable component prefixes:

- `dreamx.creator.video.*`
- `dreamx.creator.audio.*`
- `dreamx.creator.joint.*`
- `dreamx.refiner.dit.*`

### mmproj GGUF

Default name: `mmproj-DreamX-Creator-BF16.gguf`.

It contains the conditioning and codec components:

- UMT5-XXL encoder weights and tokenizer metadata.
- Wan2.2 video VAE encoder and decoder.
- CreatorDACVAE decoder.
- Flash latent upsampler.
- Causal 2D latent upsampler.
- LightVAE-NU decoder.

Tensor names use these prefixes:

- `dreamx.text.*`
- `dreamx.video_vae.*`
- `dreamx.audio_vae.*`
- `dreamx.refiner.upsampler.flash.*`
- `dreamx.refiner.upsampler.causal2d.*`
- `dreamx.refiner.lightvae.*`

Both files carry the same deterministic pair ID, source model identity,
exporter version, component inventory, architecture dimensions, source tensor
counts, and tokenizer configuration. The runtime rejects files with mismatched
pair IDs or component inventories.

## Precision

The exporter accepts `--outtype bf16` and `--outtype q8_0`.

For `q8_0`, large 2D Linear weights use row-wise GGML Q8_0. Norms, biases,
embeddings, modulation tensors, convolution weights, and tensors that cannot be
represented safely as Q8_0 remain BF16 or F32. Source F32 tensors are converted
streamingly so the exporter does not hold a complete checkpoint in memory.

The BF16 mode is the reference artifact. Q8_0 is the practical CPU artifact.
The runtime reads tensor precision from GGUF and does not infer it from file
names.

## Exporter

Add `tools/dreamx/convert_dreamx_creator.py`. It reuses the repository's GGUF
writer, safetensors reader, sharded-index reader, and Q8_0 quantizer instead of
adding a second serialization implementation.

The exporter reads:

- `creator/video_model/diffusion_pytorch_model*.safetensors`
- `creator/audio_model/diffusion_pytorch_model.safetensors`
- `creator/cross_attn_weights.safetensors`
- `wan2.2_ti2v_5b/models_t5_umt5-xxl-enc-bf16.pth`
- `wan2.2_ti2v_5b/Wan2.2_VAE.pth`
- `audio_vae/diffusion_pytorch_model.safetensors`
- `refiner/sr_dit_5b.pt`
- `refiner/latent_upsampler_flash.pt`
- `refiner/latent_upsampler_2d_causal.pt`
- `refiner/lightvae_nu_scheme3.pt`
- `wan2.2_ti2v_5b/google/umt5-xxl/tokenizer.json`

Before writing output, it validates every required path, sharded index entry,
tensor name, shape, and dtype against the supported released configuration.
Unsupported model revisions fail explicitly. Output is written to unique
temporary files, verified by reopening them, then atomically renamed. An error
never publishes a partial pair.

## Runtime Organization

Add `src/models/diffusion/dreamx/` with modules scoped to the actual pipeline:

- `config.rs`: metadata parsing and supported-shape validation.
- `text.rs`: UMT5 tokenizer integration and 24-layer encoder.
- `creator.rs`: video/audio DiTs and joint gated cross-attention.
- `video_vae.rs`: Wan2.2 encode/decode.
- `audio_vae.rs`: CreatorDACVAE decode.
- `refiner.rs`: SR-DiT, causal window attention, and truncated KV cache.
- `upsampler.rs`: bilinear, Flash, and causal 2D latent upsampling.
- `lightvae.rs`: LightVAE-NU decode.
- `mod.rs`: session orchestration and public API.

These are concrete model modules, not a new generic neural-network framework.
Shared low-level operations remain in the existing `ops` and `core` modules.

The Hugging Face `tokenizers` Rust crate reads the released tokenizer JSON. It
is a parsing dependency only and does not provide tensor compute.

## CPU Operators

No BLAS implementation is linked directly or transitively. New compute uses:

- Existing `ComputePool` worker scheduling.
- Hand-written AArch64 NEON kernels.
- Portable scalar fallbacks with identical public behavior.

Required kernels are:

- F32, BF16, and Q8_0 matrix-vector and matrix-matrix multiplication.
- Conv1D, Conv2D, Conv3D, depthwise convolution, and transposed convolution.
- LayerNorm, RMSNorm, GELU, SiLU, softmax, and elementwise gating.
- 1D temporal RoPE and Wan 3D RoPE.
- Blockwise dense attention for Creator.
- Causal block-grid window attention with a truncated KV cache for SR-DiT.

Attention processes score tiles and never allocates a full `sequence_length²`
matrix. Convolution and matmul scratch buffers are reused across layers.
Deliberately approximate activations are not used in the reference path.

`cargo tree` is part of verification and must show no OpenBLAS, BLIS, oneDNN,
Accelerate, MKL, or equivalent compute dependency.

## Base Generation Flow

1. Load and validate the matched GGUF pair.
2. Tokenize positive and negative prompts and run UMT5.
3. Resize the input image to the requested aligned token budget.
4. Encode the first frame with the Wan2.2 VAE.
5. Initialize deterministic video and audio noise latents.
6. Run the video and audio FlowMatch Euler schedules together.
7. In each Creator layer, run branch self-attention and text cross-attention.
8. In layers 15-29, update both branches from the same pre-update snapshot
   through gated A2V and V2A attention.
9. Apply text CFG and multimodal bridge CFG, then clamp the first-frame latent.
10. Decode video with Wan2.2 VAE and audio with CreatorDACVAE.
11. Write video-only MP4 and WAV, then mux the synchronized MP4.

## Refiner Flow

1. Reuse the generated low-resolution frames directly without decoding MP4.
2. Encode them with the full Wan2.2 VAE.
3. Upsample latent space with the selected `bilinear`, `flash`, or `causal2d`
   path.
4. Run the official four-step warped schedule chunk by chunk.
5. Use causal block-grid window attention and retain only the configured latent
   frame history in the KV cache.
6. Decode with the full Wan2.2 VAE or released LightVAE-NU decoder.
7. Encode the refined frames and stream-copy the generated WAV into the final
   MP4.

The full Wan VAE remains available when LightVAE is selected because refiner
input encoding still requires it.

## CLI

Architecture metadata dispatches DreamX through the existing `--model` and
`--mmproj` inputs. DreamX-specific arguments are limited to the real pipeline
controls:

```text
--image PATH
--prompt TEXT
--negative-prompt TEXT
--out PATH
--duration SECONDS
--fps FPS
--steps N
--seed N
--refine / --no-refine
--refiner-kv-len N
--latent-upsample bilinear|flash|causal2d
--refiner-decoder wan|lightvae
--dry-run
```

Defaults match the official base and refiner configurations. `--dry-run`
validates files, tensor shapes, derived latent shapes, and estimated peak memory
without running tensor compute.

## Memory Lifecycle

GGUF files remain memory-mapped, but only accessed pages become resident.
Scratch and caches are owned by a pipeline stage:

1. UMT5 scratch is released after prompt encoding.
2. Creator scratch is released before VAE decode.
3. Base frames are passed directly to refiner and released after VAE encoding.
4. Refiner KV cache is truncated by latent-frame count and released before
   final decode.
5. Frame encoding is streamed to `ffmpeg` so a second full RGB video copy is
   not retained.

Before compute, the runtime estimates weights, scratch, KV cache, latent, and
frame buffers. It rejects invalid dimensions and arithmetic overflow. It warns
when the estimate exceeds detected physical memory and requires an explicit
override to continue.

## Failure Behavior

- Missing or incompatible tensors name the component and tensor.
- A mismatched GGUF pair is rejected before model construction.
- NaN or Inf reports include stage, denoising step, layer, and tensor label.
- Missing optional-path weights never trigger a silent fallback.
- Existing output files are not overwritten unless explicitly requested.
- Base video and WAV remain available when refinement or muxing fails.
- Temporary outputs are removed only when their owning operation fails.

The runtime does not silently lower resolution, change precision, disable
refinement, change the selected upsampler, or replace LightVAE with Wan VAE.

## Verification

Development follows test-first slices.

1. Exporter contract tests use small synthetic safetensors and PyTorch
   checkpoints. Each new mapping is first represented by a failing test.
2. Every hand-written optimized operator is compared against a straightforward
   scalar reference over fixed edge cases and deterministic random inputs.
3. Tiny configurable DreamX fixtures exercise UMT5, Creator, both cross-modal
   directions, both VAEs, SR-DiT, all upsamplers, and LightVAE.
4. Exporting the supplied 51 GiB model verifies tensor counts, shapes, dtypes,
   metadata, pair IDs, and complete source coverage.
5. Rust `--dry-run` validates the real exported pair.
6. A short, low-resolution, reduced-step real-model run must produce a valid
   video-only MP4, 48 kHz WAV, base muxed MP4, and refined muxed MP4.
7. An upstream Python trace adapter records selected checkpoints when a usable
   CUDA environment is available. Until that comparison runs, verification is
   reported as structural, operator-level, and Rust end-to-end rather than full
   official-model numerical parity.
8. `cargo fmt --check`, focused tests, release compilation, `cargo tree`, and
   `git diff --check` must pass.

## Acceptance Criteria

- The supplied source directory exports successfully to a matched GGUF pair.
- The exporter does not load the complete model into memory at once.
- Rust loads both BF16 and Q8_0 main artifacts.
- Base Creator inference and the 2K refiner execute without Python or C++ model
  runtimes.
- Flash, causal 2D, Wan VAE, and LightVAE checkpoint paths are selectable and
  do not silently substitute for each other.
- No external CPU compute library is present in the runtime dependency tree.
- The smallest real-model end-to-end CPU check completes and produces valid
  media artifacts.
- Verification reports distinguish passing focused checks from unrun full-size
  official inference and unavailable CUDA Oracle comparison.
