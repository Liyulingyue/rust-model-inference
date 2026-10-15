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
scalar CPU KV remains F32. Text encoder allocations are dropped after encoding,
and no-longer-needed text/denoiser CFG caches are released only after sampling,
before VAE decoding; projection weights and GPU workspaces remain retained for
reuse across blocks.

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

The same release-fast binary then completed an isolated real-GGUF 1280x720,
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
