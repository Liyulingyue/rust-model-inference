# DreamX-Creator Native Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the released DreamX-Creator checkpoints as a matched GGUF/mmproj pair and run the complete base audio-video generator plus 2K refiner in native Rust on CPU.

**Architecture:** Extend the existing Python GGUF writer with streaming tensor payloads, then add a DreamX-specific exporter that preserves source tensor names under stable component prefixes. Add a concrete `models::diffusion::dreamx` runtime that reuses `TensorSource`, `ComputePool`, and existing quantized kernels while supplying hand-written scalar and AArch64 NEON kernels for missing transformer, convolution, VAE, and streaming-window operations.

**Tech Stack:** Python 3 exporter, PyTorch only for offline `.pt` loading, GGUF v3, Rust 2021, `memmap2`, `half`, `rayon`, existing `ComputePool`, AArch64 NEON intrinsics, Hugging Face `tokenizers`, and system `ffmpeg` for media encoding/muxing.

**Spec:** `docs/DREAMX_CREATOR_DESIGN.md`

## Global Constraints

- First runtime target: AArch64 CPU with a portable scalar fallback.
- Do not link OpenBLAS, BLIS, oneDNN, Accelerate, MKL, llama.cpp, libtorch, or another external compute backend into Rust.
- Python and PyTorch are offline-export dependencies only.
- Main artifact architecture is `dreamx`; auxiliary architecture is `clip` with `clip.projector_type=dreamx_creator`.
- Q8_0 applies only to eligible large 2D Linear weights; other tensors remain BF16/F32.
- Preserve bilinear, Flash, causal 2D latent upsampling, full Wan2.2 VAE decode, and LightVAE-NU decode.
- Never silently lower resolution, change precision, disable refinement, or substitute an unavailable component.
- Keep `.codex/` and unrelated worktree changes out of every commit.
- Every non-trivial production change begins with a focused failing test.

---

### Task 1: Stream GGUF Tensor Payloads

**Files:**
- Modify: `tools/dots/convert_dots_tts.py`
- Modify: `tools/dots/test_convert_dots_tts.py`
- Verify: `tools/vibevoice/test_convert_vibevoice_asr.py`

**Interfaces:**
- Consumes: existing `GgufWriter`, `_tensor_nbytes`, and GGUF readback helpers.
- Produces: `GgufWriter.add_tensor_chunks(name, ggml_type, gguf_dims, nbytes, chunks)` where `chunks` returns a fresh byte iterator.

- [ ] **Step 1: Write failing streamed-writer tests**

```python
def test_streamed_tensor_round_trips(self):
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "stream.gguf"
        writer = GgufWriter(out)
        writer.add_meta("general.architecture", "test")
        writer.add_tensor_chunks(
            "weight", GGML_F32, (2,), 8,
            lambda: iter((b"\x00\x00\x80?", b"\x00\x00\x00@")),
        )
        writer.write()
        self.assertEqual(
            read_gguf_tensor_bytes(out, "weight"),
            b"\x00\x00\x80?\x00\x00\x00@",
        )

def test_streamed_tensor_rejects_declared_size_mismatch(self):
    writer = GgufWriter(Path("unused.gguf"))
    with self.assertRaisesRegex(ValueError, "expected 8 bytes"):
        writer.add_tensor_chunks(
            "weight", GGML_F32, (2,), 7, lambda: iter((b"x" * 7,))
        )
```

- [ ] **Step 2: Verify RED**

Run `python3 -m unittest tools/dots/test_convert_dots_tts.py -v`.

Expected: failure because `add_tensor_chunks` does not exist.

- [ ] **Step 3: Add bounded tensor payload entries**

```python
@dataclass(frozen=True)
class TensorPayload:
    nbytes: int
    chunks: Callable[[], Iterable[bytes]]

def add_tensor_chunks(self, name, ggml_type, gguf_dims, nbytes, chunks):
    expected = _tensor_nbytes(ggml_type, gguf_dims)
    if nbytes != expected:
        raise ValueError(f"{name}: expected {expected} bytes, got {nbytes}")
    self._add_tensor_payload(name, ggml_type, gguf_dims, TensorPayload(nbytes, chunks))

def add_tensor(self, name, ggml_type, gguf_dims, raw):
    payload = bytes(raw)
    self.add_tensor_chunks(
        name, ggml_type, gguf_dims, len(payload), lambda: iter((payload,))
    )
```

Update header sizing, offsets, file writing, and readback validation to use `payload.nbytes`. Count bytes and SHA-256 while writing, then hash the tensor range from the completed file in bounded chunks and require the same digest.

- [ ] **Step 4: Verify existing exporters remain green**

Run:

```bash
python3 -m unittest tools/dots/test_convert_dots_tts.py -v
python3 -m unittest tools/vibevoice/test_convert_vibevoice_asr.py -v
```

Expected: all tests pass and existing tensor bytes remain unchanged.

- [ ] **Step 5: Commit**

```bash
git add tools/dots/convert_dots_tts.py tools/dots/test_convert_dots_tts.py
git commit -m "refactor(export): stream GGUF tensor payloads"
```

---

### Task 2: Define the DreamX Source Inventory and Pair Contract

**Files:**
- Create: `tools/dreamx/convert_dreamx_creator.py`
- Create: `tools/dreamx/test_convert_dreamx_creator.py`

**Interfaces:**
- Consumes: `GgufWriter`, `Tensor`, `open_safetensors`, `gguf_dims`, `ShardedSafetensors`, and `quantize_q8_0` from current exporters.
- Produces: `build_inventory`, `pair_id`, `output_paths`, and `export_model`.

- [ ] **Step 1: Write failing inventory tests**

```python
def test_output_paths_are_precision_explicit(self):
    root = Path("out")
    self.assertEqual(
        output_paths(root, "q8_0"),
        (
            root / "DreamX-Creator-Q8_0.gguf",
            root / "mmproj-DreamX-Creator-BF16.gguf",
        ),
    )

def test_inventory_requires_every_released_component(self):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_minimal_layout(root, omit="refiner/lightvae_nu_scheme3.pt")
        with self.assertRaisesRegex(FileNotFoundError, "lightvae_nu_scheme3.pt"):
            build_inventory(root)

def test_pair_id_is_canonical(self):
    left = pair_id({"video": {"count": 825, "bytes": 20}})
    right = pair_id({"video": {"bytes": 20, "count": 825}})
    self.assertEqual(left, right)
```

- [ ] **Step 2: Verify RED**

Run `python3 -m unittest tools/dreamx/test_convert_dreamx_creator.py -v`.

Expected: import failure because the converter does not exist.

- [ ] **Step 3: Add the immutable inventory**

```python
@dataclass(frozen=True)
class DreamXInventory:
    video_dir: Path
    audio_model: Path
    joint: Path
    t5: Path
    video_vae: Path
    audio_vae: Path
    sr_dit: Path
    upsampler_flash: Path
    upsampler_causal2d: Path
    lightvae: Path
    tokenizer_json: Path
    video_config: dict
    audio_config: dict
```

Require video dimensions `3072/14336/24/30/48`, audio dimensions `1536/8960/12/30/128`, text length `512`, and joint layers `15..=29`.

- [ ] **Step 4: Add deterministic pair identity**

Canonicalize config JSON, tokenizer SHA-256, source relative paths, file sizes, and safetensors header SHA-256:

```python
def pair_id(manifest: dict) -> str:
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()
```

Both outputs receive `dreamx.pair_id`, `dreamx.exporter_version=1`, `dreamx.components`, and per-component tensor counts.

- [ ] **Step 5: Open `.pt` checkpoints one at a time**

```python
state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
state = state.get("model", state.get("ema", state))
```

Reject undocumented non-tensor entries. Delete each mapped dictionary before opening the next checkpoint.

- [ ] **Step 6: Verify GREEN and commit**

```bash
python3 -m unittest tools/dreamx/test_convert_dreamx_creator.py -v
git add tools/dreamx/convert_dreamx_creator.py tools/dreamx/test_convert_dreamx_creator.py
git commit -m "feat(export): define DreamX checkpoint contract"
```

---

### Task 3: Export the Matched GGUF Pair

**Files:**
- Modify: `tools/dreamx/convert_dreamx_creator.py`
- Modify: `tools/dreamx/test_convert_dreamx_creator.py`

**Interfaces:**
- Consumes: Task 2 inventory and Task 1 streaming writer.
- Produces: the stable tensor prefixes in the design and `--outtype bf16|q8_0`.

- [ ] **Step 1: Write failing mapping and precision tests**

```python
self.assertEqual(
    map_name("creator.video", "blocks.0.self_attn.q.weight"),
    "dreamx.creator.video.blocks.0.self_attn.q.weight",
)
self.assertTrue(should_quantize(
    "dreamx.creator.video.blocks.0.ffn.0.weight", (14336, 3072)
))
self.assertFalse(should_quantize(
    "dreamx.video_vae.decoder.conv1.weight", (160, 48, 3, 3, 3)
))
```

Add a coverage test requiring every synthetic source tensor to appear exactly once in one output.

- [ ] **Step 2: Verify RED**

Run `python3 -m unittest tools/dreamx/test_convert_dreamx_creator.py -v`.

- [ ] **Step 3: Register reproducible tensor directories**

Sort by output tensor name. Preserve source suffixes and add only these prefixes: `dreamx.creator.video`, `dreamx.creator.audio`, `dreamx.creator.joint`, `dreamx.refiner.dit`, `dreamx.text`, `dreamx.video_vae`, `dreamx.audio_vae`, `dreamx.refiner.upsampler.flash`, `dreamx.refiner.upsampler.causal2d`, and `dreamx.refiner.lightvae`.

- [ ] **Step 4: Stream precision conversion**

Process complete GGML rows in bounded chunks. Q8_0 chunk element counts must be divisible by 32. F32 source norms and biases remain F32; BF16 convolution/embedding payloads remain BF16; BF16 mode converts F32 Linear weights with round-to-nearest-even.

- [ ] **Step 5: Publish the pair atomically**

Write two unique sibling files, validate metadata/tensor directories/byte counts/SHA-256, then create the requested names. If publishing the second output fails, remove only the first output created by this invocation.

- [ ] **Step 6: Verify and commit**

```bash
python3 -m unittest tools/dreamx/test_convert_dreamx_creator.py -v
python3 tools/dreamx/convert_dreamx_creator.py --help
git add tools/dreamx/convert_dreamx_creator.py tools/dreamx/test_convert_dreamx_creator.py
git commit -m "feat(export): write DreamX GGUF pair"
```

---

### Task 4: Add DreamX Metadata and CLI Validation

**Files:**
- Create: `src/models/diffusion/dreamx/config.rs`
- Create: `src/models/diffusion/dreamx/mod.rs`
- Modify: `src/models/diffusion/mod.rs`
- Modify: `src/app/cli.rs`
- Modify: `src/app/mod.rs`

**Interfaces:**
- Produces: `DreamXConfig::from_sources`, `DreamXOptions`, `DreamXRefinerOptions`, `DreamXCliOptions`, and `dreamx_cli_options`.

- [ ] **Step 1: Write failing pair and CLI tests**

```rust
#[test]
fn pair_validation_rejects_different_ids() {
    let main = source("dreamx", "pair-a");
    let aux = source("clip", "pair-b")
        .with("clip.projector_type", "dreamx_creator");
    assert!(DreamXConfig::from_sources(&main, &aux)
        .unwrap_err().contains("pair ID mismatch"));
}

#[test]
fn dreamx_cli_requires_model_mmproj_image_prompt_and_out() {
    let options = parse_cli_options(&args(&[
        "rmi", "--dreamx", "--model", "dreamx.gguf",
        "--mmproj", "mmproj.gguf", "--image", "first.png",
        "--prompt", "scene", "--out", "scene.mp4",
    ])).unwrap();
    assert!(dreamx_cli_options(&options).unwrap().is_some());
}
```

- [ ] **Step 2: Verify RED**

Run `cargo test dreamx_cli -- --nocapture` and `cargo test pair_validation -- --nocapture`.

- [ ] **Step 3: Add exact option types**

```rust
pub struct DreamXOptions {
    pub duration_seconds: f32,
    pub fps: usize,
    pub steps: usize,
    pub seed: i64,
    pub target_spatial_tokens: usize,
    pub refine: bool,
    pub refiner: DreamXRefinerOptions,
}

pub struct DreamXRefinerOptions {
    pub kv_len: usize,
    pub latent_upsample: LatentUpsampleKind,
    pub decoder: RefinerDecoderKind,
}
```

Add strict parsing for `--dreamx`, `--duration`, `--fps`, `--target-spatial-tokens`, `--refine`, `--no-refine`, `--refiner-kv-len`, `--latent-upsample`, `--refiner-decoder`, `--dry-run`, `--overwrite`, and `--allow-memory-overcommit`.

- [ ] **Step 4: Validate the pair contract**

Require main `general.architecture=dreamx`; auxiliary `general.architecture=clip`; projector type `dreamx_creator`; equal 64-character hex pair IDs; all component flags; and the exact released dimensions from Task 2.

- [ ] **Step 5: Verify and commit**

```bash
cargo test dreamx_cli -- --nocapture
cargo test pair_validation -- --nocapture
git add src/models/diffusion/dreamx/config.rs src/models/diffusion/dreamx/mod.rs src/models/diffusion/mod.rs src/app/cli.rs src/app/mod.rs
git commit -m "feat(dreamx): validate model pair and CLI"
```

---

### Task 5: Add DreamX Scalar and AArch64 CPU Kernels

**Files:**
- Create: `src/models/diffusion/dreamx/kernels.rs`
- Modify: `src/models/diffusion/dreamx/mod.rs`

**Interfaces:**
- Consumes: `TensorSource`, `GGMLType`, `Weight`, `QuantizedTensor`, and `ComputePool`.
- Produces: checked Linear, normalization, activation, RoPE, attention, Conv1D/2D/3D, depthwise, and transposed-convolution functions.

- [ ] **Step 1: Write scalar-reference comparison tests**

```rust
#[test]
fn conv3d_matches_scalar_reference() {
    let input = deterministic_values(2 * 3 * 4 * 5);
    let weight = deterministic_values(4 * 2 * 3 * 3 * 3);
    let expected = conv3d_scalar(
        &input, [2, 3, 4, 5], &weight, [4, 2, 3, 3, 3], [1, 1, 1],
    ).unwrap();
    let actual = conv3d(
        &pool(), &input, [2, 3, 4, 5], &weight, [4, 2, 3, 3, 3], [1, 1, 1],
    ).unwrap();
    assert_close(&actual, &expected, 1e-5);
}

#[test]
fn online_attention_matches_materialized_softmax() {
    let (q, k, v, spec) = tiny_qkv();
    let expected = attention_scalar(&q, &k, &v, spec).unwrap();
    let actual = attention_online(&q, &k, &v, spec).unwrap();
    assert_close(&actual, &expected, 1e-5);
}
```

- [ ] **Step 2: Verify RED**

Run `cargo test models::diffusion::dreamx::kernels -- --nocapture`.

- [ ] **Step 3: Implement scalar correctness paths**

Use checked shape multiplication and explicit NCTHW/NCT layouts. Online attention maintains a running maximum, normalization sum, and value accumulator per query tile:

```rust
let new_max = running_max.max(score);
let old_scale = (running_max - new_max).exp();
let new_scale = (score - new_max).exp();
normalizer = normalizer * old_scale + new_scale;
for d in 0..head_dim {
    accumulator[d] = accumulator[d] * old_scale + new_scale * value[d];
}
running_max = new_max;
```

- [ ] **Step 4: Add AArch64 NEON inner loops**

Use `std::arch::aarch64` behind `#[cfg(target_arch = "aarch64")]` for F32/BF16 dot products, convolution channel accumulation, vector arithmetic, norm reductions, and activation batches. Dispatch once outside hot loops and use scalar code elsewhere.

- [ ] **Step 5: Verify kernels and dependency boundary**

```bash
cargo test models::diffusion::dreamx::kernels -- --nocapture
test -z "$(cargo tree | rg -i 'openblas|blas-src|blis|mkl|accelerate|onednn' || true)"
```

- [ ] **Step 6: Commit**

```bash
git add src/models/diffusion/dreamx/kernels.rs src/models/diffusion/dreamx/mod.rs
git commit -m "feat(dreamx): add native CPU kernels"
```

---

### Task 6: Implement UMT5 Tokenization and Text Encoding

**Files:**
- Modify: `Cargo.toml`
- Modify: `Cargo.lock`
- Create: `src/models/diffusion/dreamx/text.rs`
- Modify: `src/models/diffusion/dreamx/mod.rs`

**Interfaces:**
- Consumes: tokenizer JSON metadata and `dreamx.text.*` tensors.
- Produces: `DreamXTextEncoder::load` and `encode(prompt, negative_prompt) -> Result<TextConditioning, String>`.

- [ ] **Step 1: Write failing tokenizer and encoder tests**

```rust
#[test]
fn released_tokenizer_encodes_and_pads_to_512() {
    let encoder = tiny_text_encoder();
    let encoded = encoder.tokenize("a person speaking").unwrap();
    assert_eq!(encoded.ids.len(), 512);
    assert!(encoded.ids[encoded.real_len..].iter().all(|&id| id == 0));
}

#[test]
fn text_encoder_returns_positive_and_negative_context() {
    let out = tiny_text_encoder().encode("rain", "static").unwrap();
    assert_eq!(out.positive.len(), 512 * TINY_TEXT_DIM);
    assert_eq!(out.negative.len(), 512 * TINY_TEXT_DIM);
}
```

- [ ] **Step 2: Verify RED**

Run `cargo test dreamx::text -- --nocapture`.

- [ ] **Step 3: Add the tokenizer parser**

Run `cargo add tokenizers --no-default-features`. Load the exact tokenizer JSON stored in GGUF metadata and set truncation/padding to 512 explicitly. This dependency parses text only and must not add a compute backend.

- [ ] **Step 4: Port the released 24-layer encoder**

Load shared embedding, relative-position buckets, self-attention, gated GELU FFN, RMSNorm, and final norm tensors. Use Task 5 kernels. Zero rows after each real sequence length to match the reference wrapper.

- [ ] **Step 5: Verify and commit**

```bash
cargo test dreamx::text -- --nocapture
git add Cargo.toml Cargo.lock src/models/diffusion/dreamx/text.rs src/models/diffusion/dreamx/mod.rs
git commit -m "feat(dreamx): add UMT5 text encoder"
```

---

### Task 7: Implement Wan2.2 Video VAE

**Files:**
- Create: `src/models/diffusion/dreamx/video_vae.rs`
- Modify: `src/models/diffusion/dreamx/mod.rs`

**Interfaces:**
- Produces: `Wan22Vae::load`, `encode_first_frame`, `encode_frames`, and `decode_frames` using public `[C,T,H,W]` latent layout.

- [ ] **Step 1: Write failing normalization and shape tests**

```rust
#[test]
fn wan_latent_normalization_round_trips() {
    let raw = deterministic_values(48 * 2);
    let normalized = normalize_latent(&raw).unwrap();
    assert_close(&denormalize_latent(&normalized).unwrap(), &raw, 1e-6);
}

#[test]
fn first_frame_encode_has_48_channels_and_spatial_ratio_16() {
    let latent = tiny_vae().encode_first_frame(&rgb_fixture(64, 96)).unwrap();
    assert_eq!(latent.shape(), [48, 1, 4, 6]);
}
```

- [ ] **Step 2: Verify RED**

Run `cargo test dreamx::video_vae -- --nocapture`.

- [ ] **Step 3: Port the concrete Wan2.2 graph**

Implement causal Conv3D caches, residual blocks, attention blocks, spatial `/16`, temporal `/4`, 48-channel normalization, tiled encode/decode, and RGB conversion. Load only `dreamx.video_vae.*` tensors and include missing tensor names in errors.

- [ ] **Step 4: Verify and commit**

```bash
cargo test dreamx::video_vae -- --nocapture
git add src/models/diffusion/dreamx/video_vae.rs src/models/diffusion/dreamx/mod.rs
git commit -m "feat(dreamx): add Wan2.2 video VAE"
```

---

### Task 8: Implement CreatorDACVAE Audio Decode

**Files:**
- Create: `src/models/diffusion/dreamx/audio_vae.rs`
- Modify: `src/models/diffusion/dreamx/mod.rs`

**Interfaces:**
- Produces: `CreatorDacVae::load` and `decode(latent, frames) -> Result<Vec<f32>, String>` for 48 kHz mono.

- [ ] **Step 1: Write failing latent-length and decoder tests**

```rust
#[test]
fn five_seconds_maps_to_250_audio_latent_frames() {
    assert_eq!(CreatorDacVae::latent_frames(5.0).unwrap(), 250);
}

#[test]
fn tiny_decoder_produces_hop_length_samples_per_frame() {
    let waveform = tiny_audio_vae()
        .decode(&vec![0.0; 128 * 2], 2)
        .unwrap();
    assert_eq!(waveform.len(), 2 * 960);
}
```

- [ ] **Step 2: Verify RED**

Run `cargo test dreamx::audio_vae -- --nocapture`.

- [ ] **Step 3: Port the released continuous DAC decoder**

Implement initial Conv1D, residual units, Snake activations, transposed-convolution rates `[8,5,4,3,2]`, final convolution, and waveform clamp. Do not load discrete quantizer tensors because generated data enters the released continuous 128-channel latent directly.

- [ ] **Step 4: Verify and commit**

```bash
cargo test dreamx::audio_vae -- --nocapture
git add src/models/diffusion/dreamx/audio_vae.rs src/models/diffusion/dreamx/mod.rs
git commit -m "feat(dreamx): add Creator audio VAE"
```

---

### Task 9: Implement the Joint Creator Denoiser

**Files:**
- Create: `src/models/diffusion/dreamx/creator.rs`
- Modify: `src/models/diffusion/dreamx/mod.rs`

**Interfaces:**
- Consumes: `TextConditioning`, first-frame latent, and Task 5 kernels.
- Produces: `CreatorModel::load` and `denoise(video, audio, conditioning, options)`.

- [ ] **Step 1: Write failing temporal and joint-update tests**

```rust
#[test]
fn joint_block_uses_the_same_pre_update_snapshot_both_ways() {
    let block = tiny_joint_block();
    let (video, audio) = block.forward(&[1.0], &[2.0], tiny_joint_args()).unwrap();
    assert_eq!(video, vec![3.0]);
    assert_eq!(audio, vec![3.0]);
}

#[test]
fn first_frame_timestep_is_zero_at_every_step() {
    let times = video_token_timesteps(750.0, 4, 16).unwrap();
    assert_eq!(&times[..4], &[0.0; 4]);
    assert!(times[4..].iter().all(|&value| value == 750.0));
}
```

- [ ] **Step 2: Verify RED**

Run `cargo test dreamx::creator -- --nocapture`.

- [ ] **Step 3: Port video and audio branches**

Video uses `[1,2,2]` patches, 3072 hidden, 24 heads, Wan 3D RoPE, text cross-attention, and 14336 FFN. Audio uses 128-channel frames, 1536 hidden, 12 heads, 1D RoPE, text cross-attention, and 8960 FFN.

- [ ] **Step 4: Port gated A2V/V2A layers 15-29**

Normalize both branches, project Q/K/V, apply temporal RoPE in audio-token units, compute both directions from immutable pre-update slices, apply per-head hidden/context sigmoid gates, and add both residuals before either branch FFN.

- [ ] **Step 5: Add multimodal CFG and FlowMatch Euler**

Use the reference logical branches and bridge scales. Keep video/audio schedules distinct but require equal counts. After every video scheduler step, overwrite the first latent frame with the encoded input frame.

- [ ] **Step 6: Verify and commit**

```bash
cargo test dreamx::creator -- --nocapture
git add src/models/diffusion/dreamx/creator.rs src/models/diffusion/dreamx/mod.rs
git commit -m "feat(dreamx): add joint audio video denoiser"
```

---

### Task 10: Write Base Audio and Video Outputs

**Files:**
- Create: `src/models/diffusion/dreamx/media.rs`
- Modify: `src/models/diffusion/dreamx/mod.rs`

**Interfaces:**
- Produces: `encode_wav`, `write_wav_atomic`, `FfmpegVideoWriter`, and `mux_audio_atomic`.

- [ ] **Step 1: Write failing WAV and ffmpeg argument tests**

```rust
#[test]
fn wav_header_declares_mono_48khz_pcm16() {
    let bytes = encode_wav(&[0.0, 1.0, -1.0], 48_000).unwrap();
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 1);
    assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 48_000);
}

#[test]
fn video_writer_uses_raw_rgb24_and_requested_fps() {
    let args = ffmpeg_video_args(96, 64, 24, Path::new("out.mp4"));
    assert!(args.windows(2).any(|values| values == ["-pix_fmt", "rgb24"]));
    assert!(args.windows(2).any(|values| values == ["-r", "24"]));
}
```

- [ ] **Step 2: Verify RED**

Run `cargo test dreamx::media -- --nocapture`.

- [ ] **Step 3: Add atomic media writes**

Encode saturated PCM16, stream RGB24 frames into `ffmpeg` stdin, require a successful child exit, and rename sibling temporary outputs only after close. Mux with `-map 0:v:0 -map 1:a:0 -c copy -shortest`.

- [ ] **Step 4: Verify and commit**

```bash
cargo test dreamx::media -- --nocapture
git add src/models/diffusion/dreamx/media.rs src/models/diffusion/dreamx/mod.rs
git commit -m "feat(dreamx): write synchronized media"
```

---

### Task 11: Implement Refiner Upsamplers and LightVAE

**Files:**
- Create: `src/models/diffusion/dreamx/upsampler.rs`
- Create: `src/models/diffusion/dreamx/lightvae.rs`
- Modify: `src/models/diffusion/dreamx/mod.rs`

**Interfaces:**
- Produces: `LatentUpsampler::load/upsample` and `LightVae::load/decode_frames`.

- [ ] **Step 1: Write failing selection and shape tests**

```rust
#[test]
fn unavailable_selected_upsampler_does_not_fallback() {
    let error = LatentUpsampler::load(
        LatentUpsampleKind::Flash, &empty_source(), pool(),
    ).unwrap_err();
    assert!(error.contains("dreamx.refiner.upsampler.flash"));
}

#[test]
fn every_upsampler_doubles_spatial_shape() {
    for kind in [
        LatentUpsampleKind::Bilinear,
        LatentUpsampleKind::Flash,
        LatentUpsampleKind::Causal2d,
    ] {
        assert_eq!(
            tiny_upsampler(kind).upsample(&tiny_latent()).unwrap().shape(),
            [48, 3, 8, 12],
        );
    }
}
```

- [ ] **Step 2: Verify RED**

Run `cargo test dreamx::upsampler -- --nocapture` and `cargo test dreamx::lightvae -- --nocapture`.

- [ ] **Step 3: Port all latent upsampling paths**

Implement bilinear interpolation, eight released Flash blocks with replicated temporal memory initialization, and the causal 2D network with 12 input/output residual blocks, group norm, and left-padded temporal convolutions.

- [ ] **Step 4: Port LightVAE-NU scheme3**

Load `ema` weights, validate checkpoint-derived channel widths, denormalize the shared 48-channel latent, decode one latent frame at a time with causal feature caches, and unpatchify by 2.

- [ ] **Step 5: Verify and commit**

```bash
cargo test dreamx::upsampler -- --nocapture
cargo test dreamx::lightvae -- --nocapture
git add src/models/diffusion/dreamx/upsampler.rs src/models/diffusion/dreamx/lightvae.rs src/models/diffusion/dreamx/mod.rs
git commit -m "feat(dreamx): add refiner codecs"
```

---

### Task 12: Implement the Causal SR-DiT Refiner

**Files:**
- Create: `src/models/diffusion/dreamx/refiner.rs`
- Modify: `src/models/diffusion/dreamx/mod.rs`

**Interfaces:**
- Consumes: base RGB frames, UMT5 conditioning, Wan VAE, selected upsampler, and selected decoder.
- Produces: `DreamXRefiner::load` and `refine(frames, conditioning, options)`.

- [ ] **Step 1: Write failing schedule, cache, and window tests**

```rust
#[test]
fn refiner_uses_four_warped_steps() {
    assert_eq!(refiner_timesteps(), [1000.0, 750.0, 500.0, 250.0]);
}

#[test]
fn kv_cache_keeps_only_requested_latent_frames() {
    let mut cache = TinyKvCache::new(9);
    cache.push(frames(12));
    assert_eq!(cache.frames(), 9);
    assert_eq!(cache.first_frame_index(), 3);
}

#[test]
fn causal_window_never_reads_future_chunks() {
    let keys = window_key_indices(WindowSpec::released(), QueryChunk::new(6, 3));
    assert!(keys.iter().all(|&frame| frame <= 8));
}
```

- [ ] **Step 2: Verify RED**

Run `cargo test dreamx::refiner -- --nocapture`.

- [ ] **Step 3: Load and run the 30-layer 5B SR-DiT**

Use 3072 hidden, 14336 FFN, 24 heads, `[1,2,2]` patches, text cross-attention, Wan RoPE, and the released output projection. Process three latent frames per chunk with four denoise passes and a clean `t=0` KV refresh.

- [ ] **Step 4: Add block-grid causal window attention**

Use spatial blocks `[4,4]`, radius `[3,3]`, the current chunk plus nine cached latent frames, and Task 5 online-softmax tiles. Cache only K/V rows belonging to retained frames.

- [ ] **Step 5: Connect encode, upsample, denoise, and decode**

The full Wan VAE always encodes low-resolution frames. Decoder selection changes only final decode. Audio remains outside this module.

- [ ] **Step 6: Verify and commit**

```bash
cargo test dreamx::refiner -- --nocapture
git add src/models/diffusion/dreamx/refiner.rs src/models/diffusion/dreamx/mod.rs
git commit -m "feat(dreamx): add causal video refiner"
```

---

### Task 13: Integrate the Pipeline, Dry Run, and Main Dispatch

**Files:**
- Modify: `src/models/diffusion/dreamx/mod.rs`
- Create: `src/app/dreamx.rs`
- Modify: `src/app/mod.rs`
- Modify: `src/main.rs`
- Modify: `src/lib.rs`
- Create: `tests/dreamx_reference.rs`

**Interfaces:**
- Produces: `DreamXPipeline::load`, `DreamXPipeline::estimate`, `DreamXPipeline::generate`, and `run_dreamx_cli`.

- [ ] **Step 1: Write failing dispatch and dry-run tests**

```rust
#[test]
fn dreamx_mode_dispatches_before_generic_llm_loading() {
    assert_eq!(dispatch_mode(&dreamx_cli_fixture()), DispatchMode::DreamX);
}

#[test]
fn dry_run_reports_each_stage_without_tensor_compute() {
    let estimate = tiny_pipeline().estimate(&tiny_request()).unwrap();
    assert!(estimate.creator_scratch_bytes > 0);
    assert!(estimate.refiner_kv_bytes > 0);
    assert_eq!(estimate.audio_sample_rate, 48_000);
}
```

- [ ] **Step 2: Verify RED**

Run `cargo test --test dreamx_reference -- --nocapture` and `cargo test dreamx_mode -- --nocapture`.

- [ ] **Step 3: Add orchestration with explicit stage lifetimes**

```rust
pub fn generate(&self, request: &DreamXRequest) -> Result<DreamXArtifacts, String> {
    let conditioning = self.text.encode(
        &request.prompt, &request.negative_prompt,
    )?;
    let first = self.video_vae.encode_first_frame(&request.image)?;
    let (video_latent, audio_latent) = self.creator.denoise(
        first, &conditioning, &request.options,
    )?;
    let base_frames = self.video_vae.decode_frames(&video_latent)?;
    let waveform = self.audio_vae.decode_latent(&audio_latent)?;
    let refined = request.options.refine
        .then(|| self.refiner.refine(
            &base_frames, &conditioning, &request.options.refiner,
        ))
        .transpose()?;
    self.media.write_all(base_frames, waveform, refined, &request.output)
}
```

Use nested scopes so UMT5 scratch, Creator scratch, refiner KV, and decoded-frame scratch drop at the documented boundaries.

- [ ] **Step 4: Add checked physical-memory estimation**

Estimate active-stage weights, scratch, KV, latent, and frame buffers. If the estimate exceeds detected physical memory, fail unless `--allow-memory-overcommit` is supplied.

- [ ] **Step 5: Dispatch before generic LLM inference**

Open main with `ComponentRole::Llm`, mmproj with `ComponentRole::Mmproj`, validate the pair, run `run_dreamx_cli`, and return. Never route `dreamx` through the generic Qwen path.

- [ ] **Step 6: Verify and commit**

```bash
cargo test --test dreamx_reference -- --nocapture
cargo test dreamx_mode -- --nocapture
cargo fmt --check
git add src/models/diffusion/dreamx/mod.rs src/app/dreamx.rs src/app/mod.rs src/main.rs src/lib.rs tests/dreamx_reference.rs
git commit -m "feat(dreamx): integrate native pipeline"
```

---

### Task 14: Export Real Weights and Run CPU Acceptance

**Files:**
- Create: `tools/dreamx/dreamx_oracle_trace.py`
- Modify: `tests/dreamx_reference.rs`
- Modify: `README.md`
- Modify: `docs/SUPPORTED_MODELS.md`
- Modify: `docs/REFERENCE_IMPLEMENTATIONS.md`

**Interfaces:**
- Consumes: `/Users/gouzi/Documents/git/rust-model-inference/models/DreamX-Creator/`.
- Produces: verified GGUF files in that directory, an ignored real-model test, and exact usage documentation.

- [ ] **Step 1: Write the ignored real-pair preflight test**

```rust
#[test]
#[ignore = "requires exported DreamX-Creator GGUF pair"]
fn dreamx_real_pair_preflight_and_dry_run() {
    let root = dreamx_root();
    let main = open(&root.join("DreamX-Creator-Q8_0.gguf"));
    let aux = open(&root.join("mmproj-DreamX-Creator-BF16.gguf"));
    let pipeline = DreamXPipeline::load(main, aux, thread_count()).unwrap();
    pipeline.estimate(&small_request()).unwrap();
}
```

- [ ] **Step 2: Verify RED because artifacts are absent**

Run:

```bash
cargo test --test dreamx_reference dreamx_real_pair_preflight_and_dry_run -- --ignored --nocapture
```

- [ ] **Step 3: Export the supplied model**

```bash
python3 -m venv /tmp/rmi-dreamx-export
/tmp/rmi-dreamx-export/bin/pip install torch numpy
/tmp/rmi-dreamx-export/bin/python tools/dreamx/convert_dreamx_creator.py \
  /Users/gouzi/Documents/git/rust-model-inference/models/DreamX-Creator \
  --out-dir /Users/gouzi/Documents/git/rust-model-inference/models/DreamX-Creator \
  --outtype q8_0
```

Record exact file sizes, SHA-256 values, tensor counts, and elapsed time.

- [ ] **Step 4: Run real dry-run and reduced CPU generation**

Use an official case image copied to a stable temporary path, then run:

```bash
cargo run --release --bin rust-model-inference -- \
  --dreamx \
  --model /Users/gouzi/Documents/git/rust-model-inference/models/DreamX-Creator/DreamX-Creator-Q8_0.gguf \
  --mmproj /Users/gouzi/Documents/git/rust-model-inference/models/DreamX-Creator/mmproj-DreamX-Creator-BF16.gguf \
  --image /Users/gouzi/Documents/git/rust-model-inference/models/DreamX-Creator/dreamx-creator_teaser.png \
  --prompt "A man speaking while seated on a yellow couch." \
  --duration 0.2 --fps 5 --steps 1 --target-spatial-tokens 4 \
  --refine --latent-upsample flash --refiner-decoder lightvae \
  --out /tmp/dreamx-smoke.mp4 --threads 8
```

Require a probeable base video-only MP4, 48 kHz WAV, base muxed MP4, and refined muxed MP4.

- [ ] **Step 5: Add the official Python trace adapter**

Capture tokenizer IDs, text context, first-frame latent, Creator layer 0/15/29 outputs, final base latents, SR first/last block outputs, final SR latent, and decoder outputs. Exit with a clear CUDA requirement when the upstream runtime cannot execute.

- [ ] **Step 6: Run final focused checks**

```bash
python3 -m unittest tools/dots/test_convert_dots_tts.py -v
python3 -m unittest tools/vibevoice/test_convert_vibevoice_asr.py -v
python3 -m unittest tools/dreamx/test_convert_dreamx_creator.py -v
cargo test dreamx -- --nocapture
cargo test --test dreamx_reference --no-run
cargo fmt --check
cargo build --release --bin rust-model-inference
test -z "$(cargo tree | rg -i 'openblas|blas-src|blis|mkl|accelerate|onednn' || true)"
git diff --check
```

- [ ] **Step 7: Document exact support and limits**

README usage names both generated files and every refiner switch. Model documentation distinguishes structural, operator, and reduced end-to-end CPU validation from an unrun full-size official inference or unavailable CUDA Oracle comparison.

- [ ] **Step 8: Commit verification and documentation**

```bash
git add tools/dreamx/dreamx_oracle_trace.py tests/dreamx_reference.rs README.md docs/SUPPORTED_MODELS.md docs/REFERENCE_IMPLEMENTATIONS.md
git commit -m "test(dreamx): verify exported CPU pipeline"
```

---

## Spec Coverage

- Artifact pair, precision, streaming conversion, and publication: Tasks 1-3.
- Metadata identity, unsupported-revision rejection, and CLI boundaries: Task 4.
- Hand-written scalar/NEON compute with no BLAS dependency: Task 5.
- UMT5 conditioning: Task 6.
- Full Wan2.2 VAE encode/decode: Task 7.
- CreatorDACVAE audio decode: Task 8.
- Base video/audio denoising, gated cross-modal updates, CFG, and scheduling: Task 9.
- Atomic WAV/MP4 generation and ffmpeg muxing: Task 10.
- Bilinear, Flash, causal 2D, and LightVAE paths: Task 11.
- Causal 5B SR-DiT, window attention, and truncated KV cache: Task 12.
- Stage memory lifecycle, dry-run estimates, complete pipeline, and dispatch: Task 13.
- Real 51 GiB export, reduced CPU end-to-end run, Oracle hooks, and support docs: Task 14.

No approved design requirement is deferred to an unspecified task.

---

## Completion Review

- [ ] Confirm all task commits descend from approved design commit `599b7d0`.
- [ ] Confirm `.codex/` and generated model artifacts are not tracked by Git.
- [ ] Read the exact changed-file list with `git diff --name-only 5fb8672...HEAD`.
- [ ] Report focused passing checks separately from full-size official inference and CUDA Oracle status.
- [ ] Do not claim official numerical parity unless Python and Rust checkpoints were actually compared.
