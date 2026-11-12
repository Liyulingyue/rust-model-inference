# P-JiT 21K Oracle

This adapter targets the released still-image pixel-space DiT. It needs the
Qwen3.5-4B caption encoder, with raw block outputs 7/15/27 concatenated to 7680
features per token. There is no VAE. Rust uses existing SIMD/thread-pool kernels
and `LinearExecutor`; floating-point results are compared numerically.

**Experimental / partial adaptation.** Float DiT single-forward checks and the
post-fix seven-step F32 and default 1024px/50-step F16 sampling replays pass.
Quantized end-to-end numerical gates remain unresolved. The caption encoder
runs on CPU even when DiT uses Vulkan.
All 27 format/backend CLI and HTTP combinations have real transport evidence;
see [validation](validation.md) for their separate numerical results.

Pinned inputs:

- ModelScope `Linum-AI/pjit-21k`, revision `695b80dd382c45a049930c86a7a7590a81a1e918`.
- `model.safetensors`: 8,736,419,856 bytes; SHA256 `8605f442a6278698481bd36b87013644354652a7647974b8f19764f5113a971c`.
- Official code: <https://github.com/Linum-AI/pjit-21k>, commit `95403933f3727abe947977c75915a53292e1d841`.
- Qwen ModelScope `Qwen/Qwen3.5-4B`, revision `fcb1a040bb418b0b8add6f6f6c475386abc2cb97`.
- Text converter llama.cpp commit `b96806d96061049a5b574269b049bf6241d63d46`.

All weights were downloaded directly on the authorized Linux host. The checkout
is `/workspace/rmi-pjit`; `models` points to `/root/rmi-pjit-storage/models` on
the large root filesystem. `/workspace` is a separate 98 GB volume. Do not put
these weights on the Mac. Source archives can be piped through SSH; they contain
no model weights.

When the download endpoint is reachable only from the laptop, relay the response
body through SSH without a local temporary file. For example, after creating the
remote directory, with `REMOTE` set to an SSH host alias:

```sh
set -o pipefail
curl --fail --location \
 'https://modelscope.cn/api/v1/models/Linum-AI/pjit-21k/repo?Revision=695b80dd382c45a049930c86a7a7590a81a1e918&FilePath=model.safetensors' \
 | ssh "$REMOTE" 'cat > /data/pjit-21k/model.safetensors.partial'
# Only after the pipeline succeeds, verify SHA256 remotely against the pin above
# and rename the partial file there. Never treat a partial download as complete.
```

## Convert and run

Use a repository-local Python venv with NumPy, safetensors, gguf and the pinned
llama.cpp converter's dependencies. The Oracle additionally needs PyTorch and
Transformers. No external accelerator library is used by the Rust runtime.

```sh
python -m tools.converter.pjit.convert_pjit models/pjit-21k models/pjit-21k/pjit-f32.gguf --format f32
python -m tools.converter.pjit.convert_text models/Qwen3.5-4B models/Qwen3.5-4B/text-bf16.gguf --llama-cpp target/llama.cpp
# Additional DiT formats: f16 bf16 q8_0 q4_0 q4_1 q4_k q5_k q6_k.
# Additional text formats: export f32/f16 with convert_text, or quantize BF16:
python -m tools.converter.utils.quantize_gguf models/Qwen3.5-4B/text-bf16.gguf models/Qwen3.5-4B/text-q8_0.gguf --format q8_0
cargo build --locked --profile release-fast --features vulkan,parity-trace --bin rust-model-inference --bin rust-model-server

target/release-fast/rust-model-inference \
  --model models/pjit-21k/pjit-f16.gguf \
  --text-encoder models/Qwen3.5-4B/text-bf16.gguf \
  --prompt 'A red cube on a white table.' --resolution 1024 \
  --steps 50 --seed 42 --threads 8 --compute cpu --out target/pjit.png
```

`--noise` accepts already scaled planar CHW little-endian F32 noise for exact
input matching. The built-in CPU Torch MT19937 noise is reproducible locally;
it does not reproduce the official H100-specific noise kernel. `--cfg` (or
`--cfg-scale`) and `--negative-prompt` control APG. Output is an atomic PNG;
`--overwrite` is required to replace one. Non-multiple-of-32 sizes are padded
and cropped. Bounds: 2048 per side, 2,097,152 pixels, 1..1000 steps.

HTTP uses the existing server binary and `/v1/images/generations`:

```sh
target/release-fast/rust-model-server --model models/pjit-21k/pjit-f32.gguf \
  --text-encoder models/Qwen3.5-4B/text-bf16.gguf --threads 8 --compute auto --port 18121
curl http://127.0.0.1:18121/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A red cube on a white table.","size":"1024x1024","steps":50,"seed":42,"response_format":"b64_json"}'
```

Only `n=1`, `b64_json` responses are supported; unknown fields and incompatible
components/task options are rejected. Request concurrency is one; cancellation
retains the permit until computation finishes.

## Reproduce numerical checks

```sh
python tools/oracle/pjit/reference.py --oracle target/pjit-oracle --weights models/pjit-21k \
  --qwen models/Qwen3.5-4B --output target/pjit-fixture --prepare-only
python tools/oracle/pjit/reference.py --oracle target/pjit-oracle --weights models/pjit-21k \
  --fixture target/pjit-fixture --output target/pjit-reference
# Add --gguf PATH for independently decoded same-weight reference;
# add --native-bf16 to measure the official autocast precision separately.
RMI_PJIT_GGUF=models/pjit-21k/pjit-f32.gguf RMI_PJIT_FIXTURE=target/pjit-fixture \
 RMI_PJIT_COMPUTE=cpu RMI_PJIT_OUTPUT=target/pjit-actual.f32 \
 cargo test --locked --profile release-fast --lib pjit_released_dit_matches_fixture --features vulkan,parity-trace -- --ignored --nocapture
python tools/oracle/pjit/compare.py target/pjit-actual.f32 target/pjit-reference/pjit.prediction.f32
```

For quantized DiT, add `--gguf PATH --activation-quant` to the official reference
command and set `PYTHONPATH=.`. Without activation QDQ this measures weight-only
quantization, not the prepared integer kernels. To distinguish local projection
error from accumulated drift, capture an unfiltered parity trace and run:

```sh
PYTHONPATH=. python tools/oracle/pjit/linear_reference.py \
  models/pjit-21k/pjit-q8_0.gguf target/pjit.jsonl target/pjit-linears.json
PYTHONPATH=. python tools/oracle/pjit/text_reference.py --oracle target/pjit-oracle \
  --qwen models/Qwen3.5-4B --gguf models/Qwen3.5-4B/text-bf16.gguf \
  --output target/pjit-caption.f32
# Add --activation-quant for quantized caption files.
```

Both reference helpers accept `--batch-size 2`: duplicate the identical input
and compare the first sample with batch one to measure the Oracle's own
accumulation/quantization sensitivity. This diagnostic keeps the GGUF, device
and activation QDQ fixed. It does not replace or relax a failed runtime gate.

Capture a complete CLI run with
`RMI_PARITY_FILTER=pjit.text,pjit.initial_noise,pjit.euler,pjit.prediction` and
replay its identical inputs with `sample_reference.py --oracle PATH --weights PATH
--trace TRACE --output METRICS`. Add `--gguf PATH` for the exact stored float
weights. The helper also compares DiT and Euler/APG on each traced Rust input,
separating accumulated BF16-boundary drift from a local operation error. It does
not claim equality with native BF16 autocast. Quantized sampling needs activation
QDQ as well; the sampling helper alone is not a quantized-kernel oracle.
`--batch-one-baseline` also measures independent batch-one/batch-two Oracle
chains. Metrics are saved after every step so long replays retain progress.
Use a fresh trace path for each run: JSONL traces append records, and the replay
helper expects exactly one complete invocation.

`RMI_PARITY_TRACE=target/pjit.jsonl` records patch, block and final output
checkpoints; `RMI_PARITY_FILTER` can select exact names. The reference uses hooks
without editing the pinned source. The caption fixture records original token
IDs; `--text-f32` measures F32 computation over the original BF16 text weights.
See [validation](validation.md) for thresholds, results and remaining limits,
and [performance](performance.json) for same-host measurements.

Exported GGUF hashes/types are in [gguf_hashes.json](gguf_hashes.json), exact
tokenizer IDs in [tokenizer_cases.json](tokenizer_cases.json), and the final
format/backend commands and metrics in [matrix_results.json](matrix_results.json).
