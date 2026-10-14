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
sequence. For 720p, 24 FPS, 15 seconds this means 1280x736 and 96 latent frames,
then a spatial crop and temporal trim to 1280x720, 360 generated frames.
No upscaling, repeated frames or FPS change is used to manufacture the duration.

```sh
cargo build --release --features vulkan --bin rust-model-inference
target/release/rust-model-inference --gpu --astronex \
  --model models/Astronex-World/Astronex-World-BF16.gguf \
  --mmproj models/Astronex-World/mmproj-Astronex-World.gguf \
  --prompt 'A small wooden boat glides along a calm river, surrounded by green mountains, natural morning light, continuous gentle motion.' \
  --out artifacts/astronex-720p-15s.mp4 \
  --width 1280 --height 720 --video-frames 360 --fps 24 \
  --steps 8 --cfg-scale 3 --seed 42 --threads 1
```

Apple GPU execution reuses ash/Vulkan through MoltenVK. Projection uses BF16
weights with F32 activations/accumulation; online attention has bounded shared
workspace and no materialized N-squared matrix. The VAE's F32 3D convolutions
gather directly on GPU. GPU KV storage rounds to BF16 to fit shared memory;
scalar CPU KV remains F32. Text encoder allocations are dropped after encoding,
and denoiser/CFG caches are dropped before VAE decoding.

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

GPU tests exercise BF16/F32 range preservation, partial tiles, GQA/masks,
1024-channel VAE attention, and Conv3D stride/padding/dilation. They also cover
shape-specific transposed weight caching, rejection of that layout by the shared
row-major recorder, preservation of BF16Dot reduction, and workgroup-memory
limits that retain ordinary Vulkan operators. On the M3 Max, release-fast warm
native-size probes measured 7360x3072 -> 14336 BF16 projection at approximately
0.33 seconds, and 7360 queries / 18400 keys / 24 heads / head dimension 128
attention at 2.73-3.23 seconds. The attention arena is 633,077,760 bytes, independent
of total video duration. These are kernel probes, not full-video timings.

The actual M3 Max
GGUF CLI completed 32x32, five-frame MP4 generation in 135.20 seconds, including
an internally generated eight-latent-frame block. That result verifies the media
chain, not production-resolution visual quality. Native 720p full generation
is still under validation; do not infer its runtime from that tiny clip.

Optional validation artifacts: `RMI_ASTRONEX_PROFILE=1` logs each layer;
`RMI_ASTRONEX_LATENT_OUT=path` saves C,T,H,W as four little-endian u64 values,
followed by CTHW F32 values, for independent decoding without repeating sampling.

PRoPE camera input, action conditioning/output, event prompts, retrieval-window
selection and non-eight-step sampling are unsupported. The no-control T2V and
first-frame I2V routes remain experimental until production-size visual and
reference comparisons are complete.
