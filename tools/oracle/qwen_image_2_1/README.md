# Qwen-Image-2.1 CPU oracle

Pinned sources: stable-diffusion.cpp `2f886889e6e8b78738d6b87f7191f6018557c551`,
ggml `4bf5f6000653b7881d00963cd6ddb665ccd62a8d`.
The script downloads isolated source copies, applies the trace patch, and builds
without Metal, Accelerate, BLAS, CUDA or Vulkan. It never edits a reference checkout.

## Full pipeline parity

```bash
cargo build --profile release-fast --features parity-trace --bin rust-model-inference
oracle=$(tools/oracle/qwen_image_2_1/build_oracle.sh)
python3 tools/oracle/qwen_image_2_1/pipeline_parity.py \
  --dit /path/to/qwen-image-2.1-Q8_0.gguf \
  --text models/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q8_0.gguf \
  --mmproj models/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-F16.gguf \
  --vae models/Qwen-Image-2.1/vae/diffusion_pytorch_model.safetensors \
  --oracle-bin "$(dirname "$oracle")/qwen-image-2-1-components" \
  --out target/qi21-validation
```

The VAE's sibling `config.json` is required. Use a fresh `--out` directory for
each run: traces append sidecars, and the runner refuses existing case directories.
`--cases text edit1 edit2` is the default. All cases use three Euler steps, CFG
3.25, seed 42, eight threads, and a 64×32 output. The two editing fixtures are
64×32 and 32×64 RGBA images with partially transparent pixels. The second edit
also tests an empty negative prompt. Positive prompts include English, Chinese
and emoji. Override `--steps`, `--cfg` and `--threads` when needed.

The runner compares independently computed Oracle/Rust tokenizer IDs, all 36
text layers before output RMSNorm, all 27 vision layers and three DeepStack
levels, every VAE conv/norm/attention result, all 32 DiT blocks on every positive
and negative evaluation, seeded CPU MT19937 noise, sigmas, CFG velocities, Euler
latents, decoded RGBA, and the actual PNG pixel bytes including alpha.
Oracle conditioning and reference latents feed the Oracle sampler; Rust results
are never substituted for them. The VAE encoder receives independently normalized
fixture pixels, which are also checked against Rust's actual preprocessed input.

Comparison uses little-endian F32 `u32` bits, with no tolerance. Missing or extra
selected checkpoints, order/count differences, inconsistent shapes/sidecar sizes,
non-finite values, or a different bit fail the run. GGML's removed unit batch/time
axes are normalized; token-major Rust text checkpoints are assembled in checked
token/layer order. The VAE's duplicate QKV trace alias is omitted because the
identical tensor is already checked at its convolution checkpoint. PNG compressed
bytes need not match between encoders; all decoded RGBA8 bytes must match.

Successful runs save `summary.json`, commands' logs, checkpoint JSONL/F32 files
and PNGs under `--out`. Capturing checkpoints requires `parity-trace`.
`RMI_SCALAR=1` also works in ordinary builds for inference without tracing.
To additionally check an ordinary build, compile it into a separate target
directory and pass `--production-rust-bin /path/to/ordinary/rust-model-inference`;
the runner repeats each CLI case without tracing and compares every RGBA8 byte
to the output already checked against the independent Oracle.
An optimized comparison must use `--optimized` and an independently built
optimized Oracle; that comparison also rejects every differing bit. The current
optimized comparison differs at text block 0, so only scalar bitwise parity is
claimed. See the [validation report](../../../docs/develop/QWEN_IMAGE_2_1.md).

```bash
cargo build --profile release-fast --bin rust-model-inference \
  --target-dir target/qi21-production
# Add to the full-pipeline command above:
# --production-rust-bin target/qi21-production/release-fast/rust-model-inference
```

## DiT-only parity

```bash
QWEN_IMAGE_2_1_DIT=/path/to/qwen-image-2.1-Q8_0.gguf \
  tools/oracle/qwen_image_2_1/parity.sh
```

This retains the original deterministic 16×16 latent, 128-token context,
timestep-500 regression. It covers all 32 blocks and the F32 velocity output.
For a complete unfiltered trace, the comparator also accepts:

```bash
python3 tools/oracle/qwen_image_2_1/compare.py dit rust.jsonl oracle.jsonl
```

## Component harness

`qwen-image-2-1-components` supports:

```text
noise SEED LATENT_W LATENT_H
schedule STEPS IMAGE_TOKENS
text TEXT_GGUF PROMPT THREADS [MMPROJ PLANAR_RGBA_F32 PIXEL_W PIXEL_H ...]
vae_encode VAE_SAFETENSORS PLANAR_MINUS1_PLUS1_RGBA_F32 PIXEL_W PIXEL_H THREADS
vae VAE_SAFETENSORS PLANAR_LATENT_F32 LATENT_W LATENT_H THREADS
sample DIT_GGUF NOISE_F32 POSITIVE_F32 NEGATIVE_F32 LATENT_W LATENT_H STEPS CFG THREADS
       [POSITIVE_IMAGE_SLOTS_F32 NEGATIVE_IMAGE_SLOTS_F32 REF_LATENT_F32 REF_W REF_H ...]
```

Set `QWEN_IMAGE_2_1_ORACLE_TRACE` to a fresh JSONL path. Vision input is planar
RGBA in `[0,1]`; VAE input is planar RGBA in `[-1,1]`. Image slots hold exact
integer IDs represented as F32. Text uses F32 KV and the last transformer layer's
raw hidden state. The sampler calls the pinned implementation's discrete flow
timestep conversion, Flux schedule, CFG and Euler integrator without prefix caching.
