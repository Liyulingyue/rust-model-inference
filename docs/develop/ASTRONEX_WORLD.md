# Astronex-World video generation

The explicit `--astronex` route loads the paired main/components GGUFs, UMT5,
Wan2.2 VAE, and the no-camera/no-action causal Wan graph. The implementation
follows the clean reference at `27584bf1a89da01ef35a03b48d83bc818d2adf2b`;
the weight contract remains `dec4ec502f7113a1ab07b78ae40a7ac1b88b2eeb`.

Sampling uses eight latent frames per block, a 20-frame raw-KV window and four
persistent sink frames. Repeated denoising calls overwrite the current block;
a final timestep-zero pass commits clean history before generating the next
block. CFG has separate positive/negative caches. Each block restarts the
8-step UniPC solver with shift 10 and the reference negative prompt. The head
uses the raw time embedding in both shift and scale; only the block modulation
passes through the second SiLU.

Widths/heights must be positive multiples of 16. The model pads internally to
32-pixel patch boundaries and crops after decoding. Arbitrary positive frame
counts are accepted: the model generates complete blocks and trims the decoded
sequence. For 720p, 24 FPS, 5 seconds this means 1280x736 and 32 latent frames,
then a spatial crop and temporal trim to 1280x720, 120 generated frames. A
15-second request uses the same geometry with 96 latent frames and 360 output
frames.
No upscaling, repeated frames or FPS change is used to manufacture the duration.

```sh
cargo build --profile release-fast --features vulkan --bin rust-model-inference
target/release-fast/rust-model-inference --gpu --astronex \
  --model models/Astronex-World/Astronex-World-BF16.gguf \
  --mmproj models/Astronex-World/mmproj-Astronex-World.gguf \
  --prompt 'A small wooden boat glides along a calm river, surrounded by green mountains, natural morning light, continuous gentle motion.' \
  --out artifacts/astronex-720p-5s.mp4 \
  --width 1280 --height 720 --video-frames 120 --fps 24 \
  --steps 8 --cfg-scale 3 --seed 42 --threads 1
```

Apple GPU execution reuses ash/Vulkan through MoltenVK. Projection uses BF16
weights with F32 activations/accumulation; online attention has bounded shared
workspace and no materialized N-squared matrix. The VAE's F32 3D convolutions
gather directly on GPU. GPU KV storage rounds to BF16 to fit shared memory;
scalar CPU KV remains F32. Text encoder allocations are dropped after encoding;
denoiser/CFG state is dropped after sampling and before VAE decoding. Projection
weights and GPU workspaces remain retained for reuse across blocks.

## Validation boundary

Focused model/CLI checks cover contract validation, complete-patch padding,
frame trimming, sink-preserving eviction, repeated-step overwrite/clean commit,
and the official head/time and UniPC timestep contracts. Independent CPU
scalar Oracle checks remain separate from GPU accuracy checks. SIMD/FMA,
GELU/transcendentals, solver arithmetic, GPU kernels and BF16 KV rounding are
excluded from the raw-bit gate. Real 5B end-to-end raw-bit parity is not claimed.
The actual components-GGUF tokenizer matches the official tokenizer's IDs,
512-position masks/padding and truncation for seven English/Chinese/Unicode,
whitespace, special-token and empty-input cases.
Twelve real-weight exact-operator cases compare original safetensors through an
independent no-FMA/no-SIMD scalar C Oracle against the actual paired GGUFs:
nine denoiser/UMT5 projections, two Conv3D operations and six embedding rows.
All 137,984 selected F32 output values match raw bits. These are isolated operator
checks; approximate activations and the full causal generation graph are outside
that result. Reproduction commands are in the Oracle README.

GPU tests exercise BF16/F32 range preservation, partial tiles, GQA/masks,
1024-channel VAE attention, and Conv3D stride/padding/dilation. They also cover
shape-specific transposed weight caching, rejection of that layout by the shared
row-major recorder, preservation of BF16Dot reduction, and workgroup-memory
limits that retain ordinary Vulkan operators. On the M3 Max, release-fast warm
native-size probes measured 7360x3072 -> 14336 BF16 projection at approximately
0.33 seconds, and 7360 queries / 18400 keys / 24 heads / head dimension 128
attention at 2.73-3.23 seconds. The attention arena is 633,077,760 bytes, independent
of total video duration. These are kernel probes, not full-video timings.

The current release binary completed real-GGUF 512x288, 29-frame, 24-FPS MP4
generation on the M3 Max. ffprobe confirms 1.208333 seconds and full ffmpeg
decoding confirms 29 distinct frame hashes. Six inspected frames show a boat,
water, mountains and continuous scene changes. The command took 826.11 seconds,
with 18,814,550,016 bytes maximum RSS and 18,249,758,144 bytes peak footprint.
It ran alongside the 720p task and caused memory paging; this is not an isolated
performance benchmark.

The release-fast Vulkan binary completed an isolated real-GGUF 1280x720,
120-frame, 24-FPS MP4 on the M3 Max. The model generated 1280x736 internally,
decoded four causal blocks, cropped to 1280x720, and wrote
`artifacts/astronex-720p-5s.mp4`. `ffprobe` reports H.264 yuv420p, 1280x720,
24/1 FPS, 120 frames and 5.000000 seconds. Full `ffmpeg` decoding passed, and
the framemd5 check found 120 unique frame hashes; sampled frames show the boat,
water and mountains with continuous motion. The single command took
37,815.30 seconds by `/usr/bin/time -l`, with 36,547,870,720 bytes maximum RSS
and 34,092,037,792 bytes peak footprint. No second inference, compile or preview
job was launched; no per-block cache cleanup was used. The run is a correctness
and production-size smoke check, not a throughput baseline because macOS paging
was observed.

Optional validation artifacts: `RMI_ASTRONEX_PROFILE=1` logs each layer;
`RMI_ASTRONEX_LATENT_OUT=path` saves C,T,H,W as four little-endian u64 values,
followed by CTHW F32 values, for independent decoding without repeating sampling.

PRoPE camera input, action conditioning/output, event prompts, retrieval-window
selection and non-eight-step sampling are unsupported. The no-control T2V and
first-frame I2V routes remain experimental; 15-second production-size generation
and full end-to-end raw-bit parity remain unvalidated.

## Local validation on the GB10 box (2026-10-21)

The branch was validated on an NVIDIA GB10 (128 GB unified memory, Vulkan 1.4,
NVIDIA GB10 device) that a second tenant holds at 65 GB and 79–92% utilisation,
so every timing below is a shared-device number, not an isolated benchmark.

### Source weights and conversion

The upstream release is `Astronex/Astronex-World` on ModelScope. It stores the
denoiser as `model-0000{1,2,3}-of-00003.safetensors` with
`model.safetensors.index.json`, the UMT5-XXL encoder under `text_encoder/`, the
Wan2.2 VAE under `vae/`, the tokenizer under `tokenizer/`, and the Astronex
`config.json` at the root. That is exactly the layout the converter expects.

```sh
.venv/bin/python -m tools.converter.astronex_world.convert_astronex_world \
  models/Astronex-World --out-dir models/Astronex-World
```

It writes `Astronex-World-BF16.gguf` (10 GB, 898 BF16 denoiser tensors) and
`mmproj-Astronex-World.gguf` (14 GB, 242 BF16 UMT5 tensors) and embeds the
source manifest (per-file SHA256, tensor counts) in the GGUF metadata.

The converter's shape contract comes from `references/astronex-world-contract`,
a separate pinned checkout at `dec4ec50…` that is not public, so the denoiser
and VAE header files used here were regenerated from the downloaded safetensors
themselves. The denoiser and VAE shape pins are therefore self-referential in
this run; the tensor counts (898/242/196), the dtypes (BF16/BF16/F32), the
`model.*` prefix rule and the hard-coded UMT5 shapes are still enforced.

### Gates

| gate | result |
| --- | --- |
| `RMI_SCALAR=1 cargo test --test astronex_checkpoint_operators -- --ignored` | 12/12 cases, 137,984 F32 values match raw bits |
| `cargo test --lib astronex_checkpoint_tokenizer -- --ignored` | token IDs, all 512 mask/padding positions and truncation match the official tokenizer |
| official reference checkout | `references/astronex-world` at `27584bf1a89da01ef35a03b48d83bc818d2adf2b`, cloned from `github.com/Astronex-Robotics/Astronex-World` |
| official graph | the pinned Python graph loads these weights and traces 1,438 checkpoints (the whole text encoder and most of the denoiser) before the scalar oracle rejects `linspace.default` in the solver path; it also needs the official `transformers==4.56.0`, `diffusers==0.35.0` and `huggingface-hub==0.34.0` pins, otherwise the oracle rejects `aten.where.ScalarOther` |

### End-to-end runs

Same prompt and `--seed 42` throughout, `--steps 8 --cfg-scale 3`.

| run | backend | wall clock | result |
| --- | --- | --- | --- |
| 256x144, 9 frames | Vulkan GPU | 215.7 s | 9 unique frames, 0.38 s at 24 FPS |
| 256x144, 9 frames | Vulkan GPU, repeat | 235.6 s | 9 unique frames, not bit-identical to the first run |
| 256x144, 9 frames | CPU | 4843.9 s | 9 unique frames, 22.4× the GPU wall clock |
| 512x288, 29 frames | Vulkan GPU | 594.8 s | 29 unique frames, 1.21 s at 24 FPS |

The video writer spawns `ffmpeg`, so the binary needs `ffmpeg` on `PATH`; this
box has none, so the runs above used the static build that `imageio-ffmpeg`
installs into the repository venv. Sampling dominates the GPU runs (123.7 s of
the 215.7 s at 256x144; 353.3 s of the 594.8 s at 512x288) and the text
encoder plus the VAE decode dominate the CPU run (4037 s of the 4843.9 s).

### CPU versus GPU

The GPU and the CPU run share the prompt, the seed, the noise and the solver,
so the only differences are numerical. Comparing the decoded frames of the two
256x144 runs:

| frame | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PSNR (dB) | 18.4 | 19.0 | 19.0 | 19.5 | 19.3 | 18.6 | 18.3 | 18.4 | 18.4 |
| correlation | 0.926 | 0.937 | 0.935 | 0.952 | 0.949 | 0.910 | 0.910 | 0.912 | 0.917 |

Frame means agree within 0.5–3.5 gray levels and the frame-to-frame motion
magnitudes match within 2 gray levels per step, so both backends produce the
same scene and the same motion; only the fine detail differs. That is the
expected outcome of BF16 KV rounding on the GPU against F32 KV on the CPU plus
different reduction orders, and it is why the raw-bit gate stays out of scope
for the full graph.

### Run-to-run reproducibility is still open

The same binary, the same prompt and the same seed produce different videos on
this box: two GPU runs of the 256x144 case differ by 18.4–19.2 dB PSNR, and
the sampled latents differ on 61,439 of 61,440 values (max |Δ| 1.78 on a
latent whose largest magnitude is 8.66). The divergence is already present
before the VAE, so the VAE and the encoder are not the source.

Narrowing it down:

- The online attention kernel and the BF16 projection kernel are each
  bit-deterministic when called twice with the same inputs in one process
  (`examples/astronex_gpu_determinism.rs`).
- The text encoder is the first place where the divergence appears, and the
  first differing operation is always a GPU projection: the same input hash
  and the same weight produce a different output hash.
- Disabling the GPU matmul (`RUST_GPU_MIN_OUTPUTS`) or using a single-threaded
  `ComputePool` makes the text encoder reproducible, so the CPU path and the
  thread pool's own reductions are not the source.
- The divergence is intermittent: some encodes match bit for bit, and the first
  differing operation moves between runs.
- `src/vulkan.rs` records its host-write barrier in `begin_commands`, before the
  host writes that the following dispatch reads. Moving the barrier to the
  dispatch (host write first, barrier afterwards) reduces the rate but does not
  remove it, so the missing dependency is not the whole story.

`examples/astronex_text_determinism.rs` reproduces the text-encoder case
without touching the model code. Nothing in this section changes the runtime;
it is recorded so the next pass can start from a measured baseline.
