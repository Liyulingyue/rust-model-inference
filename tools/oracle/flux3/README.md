# FLUX 3 Image Klein CPU adaptation and Oracle

This adapter targets the **distilled BF16** release only. It registers `flux3`
explicitly, uses the Qwen3-VL-4B hidden layers 4/8/12/16/20/24/28/32, and owns
its three-stream DiT and flow-map sampler. Base, FP8 and NVFP4 are rejected.

The runtime uses the existing Rust CPU operators with F32 activations and
accumulation, BF16 DiT/text weights and F16 VAE convolutions. Text features,
reference features and initial noise are rounded to BF16. This is a numerical
CPU implementation; it does not promise the CUDA BF16 activation bits.

## Fixed inputs

- Hub: `black-forest-labs/flux-3-image-klein`
- Revision: `337337621b8526e120f2275d250242f4a6bedd75`
- Official source: <https://github.com/black-forest-labs/flux-3-image-klein>
- Oracle commit: `bfc8889a4497f846416f253a3972a1941ac698b9`
- Source file hashes are enforced by `reference.py`.
- Weight hashes and complete tensor/payload contracts are enforced by
  `verify_conversion.py` (359 DiT, 398 text, 250 inference VAE tensors).
- Respect the model's license and gated access requirements.

Use the pinned official code defaults: **alpha=16, text max=4096**, right padding
to a multiple of 80, 16 sampling steps by default. The Hub's `config.json` lists
alpha=6.93 and max_tokens=8192; those values are intentionally not silently mixed
with the pinned official pipeline.

## Convert and run

Run Python modules from the repository root, using a repository-local venv.
`MODEL` below is the directory containing `model.safetensors`, `ae.safetensors`
and `text_encoder/`. For example, use `models/flux-3-image-klein` in a checkout with enough disk space.

```sh
python -m tools.converter.flux3.convert_flux3 "$MODEL" "$MODEL/gguf"
python -m tools.oracle.flux3.verify_conversion "$MODEL" "$MODEL/gguf" --output target/flux3-conversion.json
cargo build --offline --profile release-fast --features parity-trace --bin rust-model-inference

target/release-fast/rust-model-inference \
  --model "$MODEL/gguf/flux3-dit.gguf" \
  --text-encoder "$MODEL/gguf/flux3-text.gguf" \
  --vae "$MODEL/gguf/flux3-vae.gguf" \
  --prompt 'A red cube on a white table.' \
  --resolution 64 --steps 2 --seed 42 --threads 8 --out target/flux3.png
```

The converter refuses to replace existing output files. The CLI supports
`--overwrite` explicitly. Use `--image` and repeated `--reference` for image
conditioning (up to ten). Reference pixels are encoded through the deterministic
32-channel mean, quant_conv, patchify and the checkpoint's BN statistics; they
are never posterior samples. This initial VAE integration accepts square images
and references on the 16-pixel grid, 64..4096 pixels. It rejects rectangular
inputs and `--gpu`. Supply a full scene caption; the separate prompt upsampler
is not part of this adapter.

## Reproduce alignment

Check out the official source at the commit above into an ignored directory,
for example `target/flux3-oracle`. Do not alter its graph. CPU Python needs
PyTorch, Transformers, NumPy, safetensors and Pillow in the repository venv;
there is no need to install the official GPU/UI dependencies.

```sh
python tools/oracle/flux3/reference.py "$MODEL" target/flux3-reference \
  --oracle-source target/flux3-oracle --side 64 --steps 2 --seed 42 --threads 8
```

Run the same Rust command above with `RMI_PARITY_TRACE=target/flux3-rust.jsonl`,
then compare:

```sh
python tools/oracle/flux3/compare.py \
  target/flux3-reference/trace.jsonl target/flux3-rust.jsonl \
  --output target/flux3-comparison.json
```

Use fresh trace/output paths for each run. `--reference IMAGE` on the Oracle
matches `--reference IMAGE` on the CLI. `--dtype bf16` provides a second official
CPU BF16 reference; the comparator selects its numerical bound from `run.json`.
The F32 reference widens the exact released weights and executes the official
graph, separating graph mistakes from BF16 rounding differences.

The comparison requires canonical checkpoint order, names, occurrence counts
and shapes, including every DiT block and every sampling step. Token IDs and
BF16 initial noise use raw-bit comparison. Floating calculations use relative
L2 <= 0.005, and VAE encode/decode <= 0.03; CPU BF16 comparison uses <= 0.06.
All comparisons reject non-finite values and report max absolute errors. Extra
legacy VAE debugging checkpoints are not part of this canonical trace. The
shared VAE RGB trace has a legacy H,W,C shape label with planar payload; the
Oracle preserves that exact trace schema.

## Validation evidence

The supported numerical path is **F32 CPU activations/accumulation**, with the
released BF16 DiT/text weights and F16 VAE convolutions. The primary Oracle
widens the released weights and runs the pinned official graph in F32.

On macOS arm64, 8 threads, seed 42, 64x64 and two flow-map steps:

| Case | Canonical checkpoints | Text relative L2 | Maximum velocity relative L2 | RGB relative L2 |
| --- | ---: | ---: | ---: | ---: |
| Text to image | 89 | 0.0000154383 | 0.0000978136 | 0.000585576 |
| Two prepared reference images | 103 | 0.0000154383 | 0.00105131 | 0.000157232 |

Both cases pass every predeclared F32 bound, produce CLI PNGs, and preserve
canonical checkpoint order/shape/count. Token IDs and initial noise are exact.

- All source SHA256 hashes match the fixed revision. All **1007** converted
  tensors are checked: 935 payloads retain source bits; 72 VAE convolution
  payloads match the specified F16 cast (weight relative L2 0.000207374).
- Five released tokenizer fixtures cover English, Chinese/Unicode/NFC,
  whitespace, special literals and empty prompt; padding/positions and
  4096-token truncation are checked.
- Separate real VAE encode/decode: reference features relative L2 0.00135860,
  RGB 0.000483546 on arm64. On Linux x86_64 (EPYC 9334, container limit 55 GiB,
  16 CPUs), RGB is 0.000476291. Both runs pass all four checkpoints.
- A clean, isolated build passes all six FLUX3 tests, including the real released
  tokenizer, text encoder and VAE. The separate text-only Oracle comparison
  passes both checkpoints (text relative L2 0.0000154383).
- Eleven shared Qwen regression tests pass locally with `parity-trace` (one
  ignored), including hidden-layer capture and key masking. All eight
  converter/comparator checks and asymmetric F16/BF16 downsampling pass.
  The binary dispatches `flux3` explicitly.
- Broader arm64 VAE regression: 25 passed, 1 ignored, 1 existing SiLU bitwise
  assertion failed by 1 ULP. The same values/failure reproduce on unmodified
  initial HEAD `973210b6`.
- The Linux build and three FLUX3 unit tests pass. Its existing Rust 1.94.1
  needs temporary `RUSTC_BOOTSTRAP=1` and
  `RUSTFLAGS='-C target-cpu=native -Zcrate-attr=feature(if_let_guard)'` for
  pre-existing syntax in `src/prompt/jinja_compat.rs`; no source workaround or
  global environment change is required.

### Cross-precision diagnostic

The native CPU BF16-activation Oracle is also measured. Against the supported
F32 runtime it **fails** the fixed 0.06 bound: the first failure is
`single_blocks.18`, occurrence 1; second velocity relative L2 is 0.0992463,
final RGB 0.310745. Token IDs/noise remain exact. This is recorded as a failed
cross-precision diagnostic, never as native BF16 activation parity. The F32
primary comparisons above are the supported accuracy claim.

The diagnostic includes the official BF16 timestep/embedding arithmetic,
intermediate rounding and BF16 text encoder. Controlled runs with rounded
timestep inputs alone and with widened text features still exceed the BF16
bound; the difference is not resolved by changing a comparison threshold.

A 256x256, default 16-step CLI smoke produces a valid PNG depicting the requested
red cube, table and shadows. This larger image has no numerical Oracle comparison.
The complete Linux CLI run is still in progress.
No GPU, rectangular image, Base/FP8/NVFP4 or native BF16 activation parity claim
is made. This is an experimental CPU adapter.
