# SenseNova Loop 9B Preview

Native `sensenova_loop` inference reuses the Qwen3.5 attention, GDN, FFN,
vision and tokenizer kernels. It loads 32 physical layers once, executes
`0..4 + (4..28) × 2 + 28..32`, and gives all 56 visits separate KV,
convolution and recurrent state. mHC routing stays F32. MTP is not executed.

The supported export is **F32 text + F32 mmproj**, matching the actual source
tensors (the source config's BF16 declaration does not describe their storage).
Other text quantizations are rejected. The two temporal patch matrices must stay
F32; manually converted encodings of the other projector matrices may load but
are unverified. The Vulkan decoder shares physical weights and keeps independent
visit state, with mHC routing, HF GDN head pairing and interleaved M-RoPE on
the device. Physical GPU numerical/performance validation remains pending:
the assigned RADV device also fails the unmodified baseline operator check
with `device lost / hard recovery`. Software-ICD shader checks are not GPU
performance evidence. Auto falls back to CPU on a device failure; strict
Vulkan propagates the failure. Vision encoding stays on CPU. The HTTP strict
Vulkan runtime retains its text-only contract; use Auto for image requests.
The image path supports one image in the final user message, retaining text
history. Historical images and video are rejected explicitly. Expanded image
context is checked before HTTP streaming and before vision inference.
Audio/video, multi-image, MTP and long-context quality remain unsupported or
unvalidated.

## Pinned reference

- [ModelScope checkpoint](https://modelscope.cn/models/SenseNova/SenseNova-Loop-9B-Preview),
  revision `4945891aba37ed2acfdb0119f5e0db224d04e20d`.
- [source-manifest.json](source-manifest.json) records source SHA256 and sizes.
- [config.json](config.json) is the pinned configuration fixture.
- The official `modeling_sensenova_loop.py` is loaded read-only with
  `trust_remote_code=True`, eager attention and F32 inference.
- Comparisons require exact tokenizer IDs and greedy IDs. Floating-point
  checkpoints use `max_abs <= 1e-3 + 1e-4 * ref_abs_max` and
  `RMSE <= 1e-5 + 1e-4 * ref_RMS`, with finite/shape/count checks. SIMD,
  exp, sigmoid, normalization and GPU reductions are not claimed bitwise identical to PyTorch.

## Remote-only weights

Run the following **on the validation host**, from the repository root. No
weight or GGUF needs to pass through the Mac filesystem. The manifest also
includes `mtp.safetensors`, required by the published shard index for official
loading, although export skips its tensors.

```sh
python3 -m venv .venv --system-site-packages
.venv/bin/pip install transformers==5.12.0 safetensors numpy pillow
# Install/reuse a CPU PyTorch build in this repository's venv.
.venv/bin/python - <<'PY'
import hashlib, json, subprocess
from pathlib import Path
manifest = json.loads(Path('tools/oracle/sensenova_loop/source-manifest.json').read_text())
root = Path('models/SenseNova-Loop-9B-Preview')
root.mkdir(parents=True, exist_ok=True)
for entry in manifest['files']:
    path = root / entry['Path']
    if not path.exists():
        partial = path.with_name(path.name + '.download')
        url = f"https://modelscope.cn/models/{manifest['repo']}/resolve/{manifest['revision']}/{entry['Path']}"
        subprocess.run(['curl', '-fL', '--retry', '4', '-o', str(partial), url], check=True)
        partial.rename(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert path.stat().st_size == entry['Size'] and digest == entry['Sha256'], path
PY
.venv/bin/python -m tools.converter.sensenova_loop.convert \
  models/SenseNova-Loop-9B-Preview \
  --outfile models/SenseNova-Loop-F32.gguf \
  --mmproj-outfile models/mmproj-SenseNova-Loop-F32.gguf
```

The converter refuses existing final/partial files, checks the complete tensor
inventory, preserves F32 mHC, applies zero-centered norm/A_log transforms, and
emits physical weights only. The projector carries a matching source-config
identity; unrelated projectors fail before inference. Allow roughly 80 GiB of
disk for source and exported weights. The reviewed text fixture run used about
30 GiB peak process RSS with borrowed F32 weights; allow more for image and
longer-context workloads. Do not infer resource limits from host RAM alone.

## Reproduce numerical checks

Use a release build and run the official and native models **sequentially**.
Caches and build artifacts stay inside the repository.

```sh
export HF_MODULES_CACHE="$PWD/target/hf-modules"
.venv/bin/python tools/oracle/sensenova_loop/oracle.py \
  models/SenseNova-Loop-9B-Preview target/loop-oracle
cargo build --locked --profile release-fast --features parity-trace,vulkan \
  --bin rust-model-inference --bin rust-model-server --example sensenova_loop_check
target/release-fast/examples/sensenova_loop_check \
  models/SenseNova-Loop-F32.gguf target/loop-oracle/fixtures.json target/loop-native-cpu cpu
.venv/bin/python tools/oracle/sensenova_loop/compare.py \
  target/loop-oracle target/loop-native-cpu --output target/loop-cpu-comparison.json
# Repeat with a fresh output directory and policy auto or vulkan.
# Strict vulkan must report backend=Vulkan and nonzero submission counters;
# a successful Auto fallback is CPU evidence only.
RMI_COMPUTE_TRACE=1 target/release-fast/examples/sensenova_loop_check \
  models/SenseNova-Loop-F32.gguf target/loop-oracle/fixtures.json target/loop-native-vulkan vulkan
```

The fixture runner checks Unicode/NFC, whitespace, special tokens, English
8-step decode, Chinese and a mixed Unicode prompt. Every physical visit is
captured independently, including both loop iterations.

For one image, create an already aligned fixture and run:

```sh
.venv/bin/python - <<'PY'
from PIL import Image
Image.new('RGB', (256, 256), (0, 0, 255)).save('target/loop-blue.png')
PY
.venv/bin/python tools/oracle/sensenova_loop/oracle.py \
  models/SenseNova-Loop-9B-Preview target/loop-image-oracle \
  --image target/loop-blue.png --steps 3
target/release-fast/examples/sensenova_loop_check \
  models/SenseNova-Loop-F32.gguf target/loop-image-oracle/fixtures.json \
  target/loop-image-native cpu models/mmproj-SenseNova-Loop-F32.gguf
.venv/bin/python tools/oracle/sensenova_loop/compare.py \
  target/loop-image-oracle target/loop-image-native --output target/loop-image-comparison.json
```

## User interfaces

```sh
target/release-fast/rust-model-inference --model models/SenseNova-Loop-F32.gguf \
  --prompt 'Hello!' --jinja --temperature 0 --max-tokens 8 --threads 16 --max-context 512
# Image generation requires --jinja and the matching mmproj.
target/release-fast/rust-model-inference --model models/SenseNova-Loop-F32.gguf \
  --mmproj models/mmproj-SenseNova-Loop-F32.gguf --image target/loop-blue.png \
  --prompt 'What color is this image?' --jinja --temperature 0 --max-tokens 3 --threads 16 --max-context 512
target/release-fast/rust-model-server --model models/SenseNova-Loop-F32.gguf \
  --mmproj models/mmproj-SenseNova-Loop-F32.gguf --jinja --threads 16 \
  --host 127.0.0.1 --port 18081 --max-context 512
```

HTTP chat accepts `enable_thinking:false` and image data URLs. Text and image
JEV use `/v1/jev/score`, `/v1/jev/grouped`, `/v1/jev/image` and
`/v1/jev/image_grouped`. CLI text supports `--jev`/`--jev-multi`; CLI image JEV
supports single-question mode and explicitly rejects `--jev-multi --image`.
For measured coverage, metrics and remaining limits see [validation.md](validation.md).

## Regression commands

```sh
.venv/bin/python -m unittest tools.converter.sensenova_loop.test_convert tools.oracle.sensenova_loop.test_compare
cargo test --locked --profile release-fast --features parity-trace,vulkan --lib sensenova
cargo test --locked --profile release-fast --features parity-trace,vulkan --lib qwen35::
cargo test --locked --profile release-fast --features parity-trace,vulkan --lib borrowed_f32
cargo test --locked --profile release-fast --features parity-trace,vulkan --lib mhc
cargo test --locked --profile release-fast --features parity-trace,vulkan --lib recurrent_ssm_gpu_repeat_interleave
# CPU software ICD checks exercise shader logic only, not acceleration.
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json cargo test --locked --profile release-fast \
  --features parity-trace,vulkan --lib full_graph_shader_prefill_decode_cache_and_reset
cargo test --locked --profile release-fast --lib
cargo check --locked --profile release-fast --features parity-trace,vulkan --examples
```

The default library suite has 16 failures already present at the unmodified
base; compare the failure names listed in the validation record.
