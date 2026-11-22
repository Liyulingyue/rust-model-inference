# SenseNova Loop 9B

The explicit `sensenova_loop` architecture supports the Preview contract and the
2026-11-17 release's YaRN/BF16 contract. Both reuse Qwen3.5 attention, GDN, FFN,
vision and tokenizer kernels: 32 physical layers execute
`0..4 + (4..28) × 2 + 28..32`, with independent KV/conv/SSM state for 56 visits.
The release's training-only `loop_gate.logits` is validated and omitted because
the pinned official implementation does not use it. mHC routing stays F32.

**Experimental / partial adaptation.** On the earlier host, before the mapped-memory
correction, release F32 text passed 570 official
state/logit comparisons on each of CPU, Auto and strict Vulkan, with exact
tokenizer/greedy IDs. F16/BF16 text and image pass 570/172 checks on all three.
F32 single-image greedy and final logits pass, but some intermediate states
exceed the original gates. Block formats still fail full-model gates; some cases
also diverge in greedy IDs. Sampled quantized projection and
embedding checks pass; those diagnostics do not clear the full-model failures.
The remote precision/interface matrix records every combination and failure.
See [release-validation.md](release-validation.md), the per-combination
[coverage matrix](release-matrix.csv), and [timings](performance.json).
[validation.md](validation.md) retains historical Preview evidence.

The port-33162 continuation rebuilt the CLI/server and fixed mapped-memory atom
alignment found by the Vulkan validation layer. Its tiny F32 probe improves from
six API errors to zero but still reports `device lost`; unmodified-baseline Qwen
fails as well. The retained interface snapshot remains 14/27 complete groups.
Independent integer-write and buffer-fill probes also fail outside this runtime
under system Mesa 25.2.8 and an isolated Mesa 24.0.5. A dedicated compute queue
passes once but fails repeat checks; that attempted selection change was reverted.
ROCm mul/add/readback still passes. Kernel logs are restricted in this container,
so the underlying kernel/firmware/hardware cause is unresolved.
The numeric matrix and old-model GPU regression precede this cache-maintenance
correction; post-correction full-model GPU validation remains incomplete.

The [2026-11-22 Q4_0 image follow-up](release-validation.md#q4_0-image-review-follow-up-2026-11-22)
records device-only snapshot `a4a5ed7` on a healthy RADV NAVI31 host. CPU answers `The image is blue.`
and Auto/strict Vulkan answer `Blue.`: the reported token divergence is real,
but both answers correctly describe the blue fixture. A hardware regression
fixes Q4_0 block-cancellation accuracy on AVX2/FMA hosts; it does **not** resolve
the remaining full-model numerical failures. Independent re-encoding of captured
first-layer GDN outputs shows sensitivity to Q8 activation-rounding boundaries;
the device codes were not read back in that diagnostic.
Original gates, before/after failures and CLI/HTTP/JEV checks are retained in
[q4-image-review.json](q4-image-review.json). The PR remains Experimental.
The later [main integration](release-validation.md#main-integration-and-cpu-q4_0-correction-2026-11-22)
also corrects Q4_0 AVX2 zero-point lane grouping. CPU/Auto/strict Vulkan now all
answer `Blue.`, but both the independent reference and CPU/GPU numerical gates
still fail. [Integration results](q4-image-integration.json) remain separate
from the device-only snapshot above.

The converter exports F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K text
weights, with F32 norms, recurrent parameters, mHC routers and projector.
Conversion and independent tiny decode checks cover all nine formats; these do
not establish full-model precision/backend support. Newly exported Q4_K/Q6_K
payloads include the shared encoder corrections described in the validation
record. Existing GGUF files are not rewritten.

The current image adapter accepts one image in the final user message while
retaining text history. Vision runs on CPU; Auto/strict Vulkan may run the text
decoder on the device. Historical images, multi-image requests, audio/video and
MTP are rejected or not implemented. Long-context quality is unverified.

## Pinned references

| Checkpoint | Revision | Assets and evidence |
| --- | --- | --- |
| [Release](https://modelscope.cn/models/SenseNova/SenseNova-Loop-9B) | `3fa2faf34a8ed0dba63d320e2312e2b900a7e8be` | [manifest](release-source-manifest.json), [config](release-config.json), [YaRN fixture](release-yarn.json), [validation](release-validation.md) |
| [Preview](https://modelscope.cn/models/SenseNova/SenseNova-Loop-9B-Preview) | `4945891aba37ed2acfdb0119f5e0db224d04e20d` | [manifest](source-manifest.json), [config](config.json), [historical validation](validation.md) |

The official `modeling_sensenova_loop.py` SHA256 is
`26e11d3522aa730c0ba031340ce42632bc2834da8cc0b6804006bb207d979a5b` for both.
Release RoPE uses YaRN factor 2, original context 262144 and maximum 524288.
Its parameters and interleaved axis selection stay local to SenseNova; existing
shared CPU/GPU rotation kernels consume the generated F32 coefficients.
Preview and other models keep their original default coefficient path.

## Remote-only assets

Run these commands **on the validation host**, from the repository root.
Do not download weights to the Mac. Allow at least 150 GiB free disk for the
source, nine text exports, projector and validation artifacts. Check the
filesystem containing `models`: the original numeric host's `/workspace` was a
separate 98 GiB volume. During validation, this task's `models` was moved to
`/root/rmi-sensenova-loop-9b-models` and symlinked back; verify its availability
after reconnecting.

```sh
python3 -m venv .venv --system-site-packages
# On the tested host, reuse its ROCm runtime before resolving dependencies.
.venv/bin/python - <<'RUNTIME'
from pathlib import Path
import sysconfig
runtime = Path('/opt/venv/lib/python3.12/site-packages')
assert runtime.is_dir()
(Path(sysconfig.get_paths()['purelib']) / 'runtime.pth').write_text(str(runtime) + '\n')
RUNTIME
.venv/bin/python -m pip install transformers==5.12.0 accelerate safetensors numpy pillow gguf
.venv/bin/python - <<'DOWNLOAD'
import hashlib, json, subprocess
from pathlib import Path
manifest = json.loads(Path('tools/oracle/sensenova_loop/release-source-manifest.json').read_text())
root = Path('models/SenseNova-Loop-9B')
root.mkdir(parents=True, exist_ok=True)
for entry in manifest['files']:
    path = root / entry['Path']
    if not path.exists():
        partial = path.with_name(path.name + '.download')
        url = f"https://modelscope.cn/models/{manifest['repo']}/resolve/{manifest['revision']}/{entry['Path']}"
        subprocess.run(['curl', '-fL', '--retry', '4', '-o', str(partial), url], check=True)
    source = path if path.exists() else partial
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert source.stat().st_size == entry['Size'] and digest == entry['Sha256'], source
    if source != path:
        source.rename(path)
DOWNLOAD
.venv/bin/python -m tools.converter.sensenova_loop.convert \
  models/SenseNova-Loop-9B --outfile models/SenseNova-Loop-9B-F32.gguf \
  --mmproj-outfile models/mmproj-SenseNova-Loop-9B-F32.gguf
```

If the remote cannot reach ModelScope, a `curl -fL URL | ssh ... 'cat > FILE.download'`
pipe forwards bytes without storing a weight file on the Mac. Enable shell
`pipefail`, then verify the remote size/SHA256 against the manifest before
renaming the partial file. Direct remote downloading was used for this run.

The converter refuses existing final/partial outputs and checks tensor names,
shapes, supported source precisions and metadata. It accepts `--quant FORMAT`
for a single low-precision export. To reuse one F32 export for all formats:

```sh
.venv/bin/python - <<'EXPORT'
from pathlib import Path
from tools.converter.sensenova_loop.convert import LINEAR_PROJECTIONS
from tools.converter.utils.gguf import _read_gguf
from tools.converter.utils.quantize_gguf import BLOCK_FORMATS, quantize_gguf
source = Path('models/SenseNova-Loop-9B-F32.gguf')
_, tensors = _read_gguf(source)
keep = tuple(name for name in tensors if not name.endswith(LINEAR_PROJECTIONS))
for fmt in BLOCK_FORMATS:
    output = Path(f'models/SenseNova-Loop-9B-{fmt.upper()}.gguf')
    quantize_gguf(source, output, target_format=fmt, keep=keep)
EXPORT
```

Quantization streams 16 MiB source chunks and publishes only a completed output,
without replacing a destination created concurrently.
The successful BF16 export used about 265 MiB peak RSS, compared with about
42.5 GiB for the interrupted, previously eager path. Conversion and oracle jobs
must respect the container memory limit, independently of advertised host RAM.

## Numerical reproduction

The reference is the pinned official model in F32/eager mode. For a low-precision
GGUF, `--gguf` replaces every converted projection with independent gguf-py
decoding, preserving the original F32 forward and non-projection parameters.
It rejects unmapped, nonfinite, meta and wrong-shaped projections. Block formats
also require `--activation-quantization` for the equivalent Q8_0/Q8_K input
contract; Q4_1 includes its separately rounded Q8_1 block sum. The independent
NumPy QDQ precedes official PyTorch Linear math; embeddings and F32 mHC are
untouched. This compares implementation error against the same stored weights
and activation contract. Weight and activation loss versus the original F32
checkpoint is reported separately.

```sh
export HF_MODULES_CACHE="$PWD/target/hf-modules"
.venv/bin/python tools/oracle/sensenova_loop/oracle.py \
  models/SenseNova-Loop-9B target/loop-oracle --device cuda --host-inputs
cargo build --locked --profile release-fast --features parity-trace,vulkan \
  --bin rust-model-inference --bin rust-model-server --example sensenova_loop_check
RMI_COMPUTE_TRACE=1 target/release-fast/examples/sensenova_loop_check \
  models/SenseNova-Loop-9B-F32.gguf target/loop-oracle/fixtures.json target/loop-native-cpu cpu
.venv/bin/python tools/oracle/sensenova_loop/compare.py \
  target/loop-oracle target/loop-native-cpu --output target/loop-comparison.json
```

Repeat with fresh output directories and `auto`/`vulkan`. A successful device
run must report Vulkan and nonzero execution submissions; Auto fallback alone
is CPU evidence. For each low-precision file, first run the official oracle
with `--gguf models/SenseNova-Loop-9B-FORMAT.gguf`; add
`--activation-quantization` for Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K. Compare
its native trace with `--lossy`. BF16 preserves F32 activations through the
existing SIMD kernel; F16 retains the family's prepared-input F16 rounding.
The oracle also accepts `--device cuda` for an installed ROCm/
CUDA PyTorch environment. Add `--host-inputs` to keep vision, embedding lookup
and the F32 output projection on CPU when shared VRAM cannot hold the full model.
Its non-overlapping device map prevents a GPU ancestor from temporarily moving
the CPU modules onto CUDA during initialization. It removes offload hooks,
materializes CPU parameters, then replaces GGUF weights. `execution.json`
records actual module devices, mapping and peak CUDA allocation. Earlier F32
validation runs predate CPU-head placement; preserve their recorded settings
when reproducing those exact observations.
Tokenizer and greedy IDs are exact; floating-point gates and their limits are
listed in the validation record. No bitwise cross-framework claim is made.

For the single-image comparison fixture:

```sh
.venv/bin/python - <<'IMAGE'
from PIL import Image
Image.new('RGB', (256, 256), (0, 0, 255)).save('target/loop-blue.png')
IMAGE
.venv/bin/python tools/oracle/sensenova_loop/oracle.py \
  models/SenseNova-Loop-9B target/loop-image-oracle --image target/loop-blue.png \
  --steps 3 --device cuda --host-inputs
target/release-fast/examples/sensenova_loop_check \
  models/SenseNova-Loop-9B-F32.gguf target/loop-image-oracle/fixtures.json \
  target/loop-image-native cpu models/mmproj-SenseNova-Loop-9B-F32.gguf
.venv/bin/python tools/oracle/sensenova_loop/compare.py \
  target/loop-image-oracle target/loop-image-native --output target/loop-image-comparison.json
```

This fixture does not validate arbitrary image resizing or long-context quality.

## User interfaces and checks

```sh
target/release-fast/rust-model-inference --model models/SenseNova-Loop-9B-F32.gguf \
  --compute cpu --prompt 'Hello!' --jinja --temperature 0 --max-tokens 8 --threads 16 --max-context 512
target/release-fast/rust-model-inference --model models/SenseNova-Loop-9B-F32.gguf \
  --mmproj models/mmproj-SenseNova-Loop-9B-F32.gguf --image target/loop-blue.png \
  --compute cpu --prompt 'What color is this image?' --jinja --temperature 0 \
  --max-tokens 3 --threads 16 --max-context 512
target/release-fast/rust-model-server --model models/SenseNova-Loop-9B-F32.gguf \
  --mmproj models/mmproj-SenseNova-Loop-9B-F32.gguf --compute cpu --jinja --threads 16 \
  --host 127.0.0.1 --port 18186 --max-context 512
```

HTTP chat accepts `enable_thinking:false` and image data URLs. Check streaming
and non-streaming chat separately. Text/image JEV routes are `/v1/jev/score`,
`/v1/jev/grouped`, `/v1/jev/image`, `/v1/jev/image_grouped`; compare their
probabilities with CLI `--jev --jev-output json` / `--jev-multi`. Both single
and grouped CLI image JEV accept `--mmproj` and `--image`. Decoder/projector source-config
identities must match in CLI, runtime startup and the shared JEV helper.
SenseNova HTTP image generation honors the selected Jinja/custom template,
preserves text/image order and requires one image marker per attachment. Marker
expansion and context-budget validation happen before streaming headers.

```sh
.venv/bin/python -m pytest tools/converter/sensenova_loop/test_convert.py \
  tools/converter/utils/test_quantize_gguf.py tools/converter/utils/test_kquants.py \
  tools/oracle/sensenova_loop/test_compare.py tools/oracle/sensenova_loop/test_quantized_reference.py
cargo test --locked --profile release-fast --features parity-trace --lib sensenova
cargo test --locked --profile release-fast --features parity-trace,vulkan --lib sensenova
# Run this process-global trace test alone; timeout catches a staging-lock regression.
cargo test --locked --profile release-fast --features parity-trace,vulkan \
  --lib traced_graph_releases_the_staging_submission_lock --no-run
timeout 30 cargo test --locked --profile release-fast --features parity-trace,vulkan \
  --lib traced_graph_releases_the_staging_submission_lock -- --ignored --test-threads=1
cargo test --locked --profile release-fast --features parity-trace,vulkan --lib qwen35::
cargo test --locked --profile release-fast --features parity-trace,vulkan --lib
```

Compare full-suite failure names to the baseline; do not classify pre-existing
or hardware failures as a successful regression run. Preserve the nine
existing binary targets and keep source/weights/dependencies on their intended
filesystems.
