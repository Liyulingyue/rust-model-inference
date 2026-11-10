# Idol: image → 3D Gaussian PLY

Native Rust implementation of the released [speridlabs/idol](https://modelscope.cn/models/speridlabs/idol) graph. The model produces binary little-endian 3DGS PLY with 17 float properties per Gaussian and a predicted camera. No Python process or external acceleration library is used by inference.

The implementation uses the original safetensors directly: FP16 denoiser/octree/Gaussian blocks, FP32 DINOv3 (the official `dinov3_dtype="fp32"` option), BF16 FLUX.2 VAE encoder, and FP32 BiRefNet. CPU and Vulkan are supported; Auto selects a usable Vulkan device or falls back to CPU. Vulkan accelerates projection, attention and convolution through existing repository operators. Normalization, elementwise work, patch preparation and geometry postprocessing still use CPU. Strict Vulkan rejects unavailable or failed devices. This is experimental until broader quality and device coverage are established; see [actual validation](validation.md).

The original RTX 5090 five-step preview wrote 262144 finite Gaussians in **24 min 44.6 s**, with 17.24 GiB peak RSS and 6.03 GiB sampled device memory. The two avocado halves are recognizable but contain floating/displaced geometry. That CLI then crashed during exit (SIGSEGV); [performance-validation.json](performance-validation.json) preserves the failed process result. The review follow-up below documents a process-local workaround for this host's driver. Native 50-step quality and production throughput are unverified.

## Download on the inference host

Run these commands **on the remote Linux machine**, so weights and caches never touch the Mac. The adaptation downloaded directly from ModelScope; an SSH streaming relay was unnecessary.

```bash
mkdir -p models
python3 -m venv models/.venv
models/.venv/bin/pip install modelscope==1.40.2
python3 tools/oracle/idol/download.py --root models
python3 tools/oracle/idol/download.py --root models --verify-only
```

[manifest.json](manifest.json) pins repository revisions, explicit filenames, byte lengths and SHA256 hashes (about 6.8 GB decimal). It downloads only these components, without optimizer/training files:

```text
models/idol/{config.json,denoiser.safetensors,decoder.safetensors,decoder_gs.safetensors,birefnet.safetensors}
models/dinov3-vith16plus/{config.json,model.safetensors}
models/FLUX.2-dev/vae/{config.json,diffusion_pytorch_model.safetensors}
```

The primary config and each required tensor's shape/type/finite values are checked before inference. Quantized GGUF and other Idol configurations are unsupported. The reader represents safetensors rank-zero scalars as one-element vectors; existing non-scalar layouts remain unchanged.

## CLI and HTTP

```bash
cargo build --locked --release --features vulkan
# Omit --steps/--cfg/--num-gaussians for the official 50-step/3/262144 defaults.
target/release/rust-model-inference \
  --image-to-3d --model models/idol --components models \
  --image object.png --out object.ply \
  --geometry-mode synthetic --seed 42 --compute vulkan --threads 16

# Image and generation options are supplied per request in server mode.
target/release/rust-model-inference \
  --serve --image-to-3d --model models/idol --components models \
  --host 127.0.0.1 --port 18089 --compute auto --threads 16
curl --fail-with-body http://127.0.0.1:18089/v1/3d/generations \
  -H 'Content-Type: application/json' \
  -d '{"image":"object.png","options":{"steps":1,"cfg_strength":0,"num_gaussians":32768,"seed":42,"mode":"synthetic"}}' \
  -o object.ply
```

The standalone `rust-model-server` binary accepts the same server flags without `--serve`. `--components` defaults to the model directory's parent. `--geometry-mode real` uses the official 1.7 crop margin; `synthetic` uses 1.2. A nonopaque RGBA image supplies its own alpha; opaque images run native BiRefNet. Output creation is atomic and refuses overwrites without `--overwrite`. CLI prints the camera JSON to stderr; HTTP returns it in `x-camera` and the PLY body as `application/octet-stream`.

HTTP `image` follows the existing image-input policy (local path, data URL, or explicitly enabled remote URL). Requests are limited to 16 MiB; one generation runs at a time (429 when busy). Invalid JSON/options/image gives 400; a request the generator cannot process (for example a picture without an object) gives 422; a device or server fault gives 500. The backend exposes no text, chat, embedding or JEV routes. Text/image JEV has no semantic counterpart for this generator and returns 404.

Native seeds are reproducible within this implementation on the CPU path: the same binary, image and seed produce the same Gaussians. They do **not** promise the same random draw stream as official PyTorch: VAE/flow noise uses the repository's Torch-MT generator and octree resampling uses seeded `StdRng`. Numerical oracle tests therefore inject exactly the same noise, latent, features and query points in both implementations. Weights remain in their released precision boundaries; GPU reductions use F32 accumulation and tolerance-based validation. Device weights use VRAM on discrete GPUs, with the existing shared-memory fallback when VRAM cannot be allocated.

The Vulkan path is **not** bit-reproducible: repeated runs of the same binary, image and seed on the device have produced two different Gaussian sets. The divergence is intermittent, and it starts in the device projections rather than in the attention, which is checked against the CPU kernel on a sample of rows. The octree resampling then turns a small numerical difference into a different set of anchors, so the whole result differs. This is the same device-level nondeterminism that `docs/develop/ASTRONEX_WORLD.md` records for the shared projections, and it is a property of the device reductions, not of this backend. Use the CPU path when a run has to be reproduced bit for bit.

### Target-host NVIDIA shutdown workaround

On the Ubuntu 22.04 / RTX 5090 host with NVIDIA **570.153.02**, intermittent shutdown failures jump into the unloaded `libnvidia-eglcore.so.570.153.02`. Preload that existing library for the individual Vulkan command:

```bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libnvidia-eglcore.so.570.153.02 \
  RUST_GPU_MIN_OUTPUTS=0 target/release/rust-model-inference \
  --image-to-3d --model models/idol --components models \
  --image object.png --out object.ply --geometry-mode synthetic \
  --steps 5 --cfg 3 --num-gaussians 262144 --seed 42 --compute vulkan --threads 16
```

The library path and version are specific to the tested host. This does not change the installed driver or system configuration, and is not a portable driver fix. Use the same process-local prefix for Vulkan tests or the server on this host. With this prefix, **50/50** repeated-exit tests and **31 Idol tests** pass, and the same five-step preview exits **0** in **24 min 25.6 s** (PLY written at 24 min 22.9 s). Its PLY SHA256 and camera match the original output, so the preview quality limitations above still apply. This observed identity is not a required floating-point bitwise gate. [review-validation.json](review-validation.json) and [validation.md](validation.md) record the evidence; numerical gates are unchanged.

## Reproduce numerical alignment

The oracle is the read-only official [idol-sdk](https://github.com/speridlabs/idol-sdk) commit `4702a7f00850e34b6e83a2170cc123c3e9570744`. [source-manifest.json](source-manifest.json) also hashes the imported Python sources, so source archives can be verified without Git metadata. Transformers 4.56.0 and Diffusers 0.36.0 provide the official conditioning models.

Prepare a repository-local `.venv` with PyTorch 2.9.1 with matching torchvision, NumPy, Pillow, safetensors, `transformers==4.56.0`, `diffusers==0.36.0`, tensordict, tqdm and plyfile. The remote validation reused its preinstalled PyTorch through a `.venv`-local `.pth`; no global Python/toolchain packages were changed. Optional CUDA attention packages are disabled in the CPU oracle process.

```bash
git clone https://github.com/speridlabs/idol-sdk reference/idol
git -C reference/idol checkout 4702a7f00850e34b6e83a2170cc123c3e9570744
.venv/bin/python tools/oracle/idol/oracle.py \
  --oracle-source reference/idol --models models --output artifacts/idol --threads 8
IDOL_ROOT=models IDOL_FIXTURES=artifacts/idol IDOL_PYTHON=.venv/bin/python \
  RAYON_NUM_THREADS=8 cargo test --locked --profile release-fast \
  --features parity-trace --lib models::idol::tests::same_input_oracle_probe \
  -- --ignored --nocapture
# Repeat the same pinned fixture through native Vulkan (no Python inference).
IDOL_ROOT=models IDOL_FIXTURES=artifacts/idol IDOL_PYTHON=.venv/bin/python \
  IDOL_COMPUTE=vulkan RUST_GPU_MIN_OUTPUTS=0 RAYON_NUM_THREADS=16 \
  cargo test --locked --profile release-fast --features vulkan,parity-trace \
  --lib models::idol::tests::same_input_oracle_probe -- --ignored --nocapture
# GPU-only primitive checks include production attention shapes and sampled CPU queries.
RUST_GPU_MIN_OUTPUTS=0 cargo test --locked --profile release-fast \
  --features vulkan --lib models::idol::gpu::tests:: -- --ignored --nocapture --test-threads=1
python3 tools/oracle/idol/compare.py --self-test
python3 tools/oracle/idol/download.py --self-test
```

Use a disk-backed `TMPDIR=target/tmp` when `/tmp` is a small tmpfs. Generated tensor fixtures, traces, weights and PLYs stay in ignored `artifacts/` and `models/`; only scripts, hashes and measured summaries belong in Git.

The fixture traverses every released block with short diagnostic shapes: 32 latent tokens, 9 conditioning tokens, 12 decoder queries, and 64×64 images. It isolates the PLY transform with official neural features. Production inference always uses 8192 latent tokens and 1024×1024 images. Checkpoint names, order, count, shape and finite values must match. [gates.json](gates.json) defines max absolute error, relative RMSE and cosine gates (including the documented FP16 joint-state calibration) for each precision class. FP16/BF16/SIMD reductions are assessed numerically, without a bitwise floating-point requirement; the integer preprocessing fixture checks exact output pixels. Small fixtures do not establish default-step image/3D quality or full-resolution accumulated numerical error.

To reproduce the accumulated-state diagnosis:

```sh
TMPDIR="$PWD/target/tmp" .venv/bin/python tools/oracle/idol/diagnose_denoiser.py \
  --oracle-source reference/idol --models models --fixtures artifacts/idol \
  --report artifacts/idol/denoiser-drift-diagnostic.json --threads 8
```

The diagnostic reruns 13 selected official blocks on both saved native and oracle inputs. The oracle-input control must reproduce each saved oracle output exactly. On this fixture, the final joint block has local max error 0.0625 / relative RMSE 0.00010977 with identical input, while end-to-end accumulated max error is 3.5 / relative RMSE 0.0055598. At the worst accumulated element, local error is 0; the difference comes from propagated input state. The calibrated peak gate applies to every joint FP16 block, with the original aggregate and final-head gates retained. `denoiser-calibration.json` preserves a small source-backed summary of the original failure and diagnostic; full raw reports stay under artifacts/idol.
