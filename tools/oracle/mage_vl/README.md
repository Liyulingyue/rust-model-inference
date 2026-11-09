# Mage-VL-Cowork

Native `mage_vl` text generation and Mage-ViT image understanding, using the
repository's Qwen3 trunk, ComputePool, SIMD dot products, attention, LayerNorm
and GELU-erf. No external acceleration library is used for Rust inference.

The release is pinned to ModelScope `microsoft/Mage-VL-Cowork`, revision
`287c5fdc8c83e43c4d207d21bc51024a52fdff38`. The official Python source is kept
unmodified and checked by SHA256 before Oracle execution:

| File | Bytes | SHA256 |
|---|---:|---|
| model-00001-of-00002.safetensors | 4563695792 | `86424086d1a35e416ae6475248856b517933aaf87b8e2418b63e74ac0f920241` |
| model-00002-of-00002.safetensors | 4919972344 | `3ec6ed805a4664b0e39f57615f02e81989fbe08ae95ac96e3be6af309115e048` |
| modeling_llava_onevision2.py | — | `20b43f36ad15623d5a16cbc99847ec1b32dcb1e5ee81716682da40933fc2c7f9` |

## Download and convert on the weight host

Run these commands **on the remote machine**. `download.py` streams responses
to `.part` files there, verifies sizes/SHA256, then renames them. The Mac only
transfers source code; it never downloads or stages weights.

```sh
python3 tools/oracle/mage_vl/download.py models/Mage-VL-Cowork
.venv/bin/python -m tools.converter.mage_vl.convert_mage_vl \
  models/Mage-VL-Cowork --out-dir models/Mage-VL-Cowork-GGUF --text-type bf16
```

Python conversion needs NumPy. Oracle checks additionally need PyTorch,
Transformers 4.57.6, torchvision and Pillow, installed in the repository venv
or read from the host's existing environment. Do not install system packages.
Text exports support `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_k`, `q5_k` and
`q6_k`, using the shared converter quantizers and native runtime kernels.
Vision and norm vectors remain F32. Use a separate output directory for each
export; existing GGUFs are not overwritten.
`source-tensors.json` records all 399 text and 297 vision mappings and shapes.
For example, repeat the command with `--text-type q8_0` or `--text-type q4_k`
and a new `--out-dir`. These are homogeneous matrix formats, not Q4_K_M mixes.

## CLI and HTTP

```sh
cargo build --locked --profile release-fast --bin rust-model-inference
target/release-fast/rust-model-inference \
  --model models/Mage-VL-Cowork-GGUF/Mage-VL-Cowork-bf16.gguf \
  --mmproj models/Mage-VL-Cowork-GGUF/mmproj-Mage-VL-Cowork-F32.gguf \
  --image image.png --prompt 'Describe this image.' \
  --compute cpu --threads 8 --kv-cache f32 --max-context 2048 --max-tokens 64
```

Omit `--image` for text chat. An explicitly supplied projector is still checked
on text requests. CLI uses the official Jinja template automatically, including
its default `<think>\n` prefix. `--kv-cache f16|f32`, `--max-context`,
`--prefill-batch-size`, temperature and repetition penalty are honored.
Use `--serve --host 127.0.0.1 --port 8080` for the shared chat endpoints.
HTTP supports text/image turns, function schemas and tool-result history;
`enable_thinking:false` selects the official closed empty-think prefix.
The server returns tool calls; it does not execute them.

`--compute auto` selects shared Vulkan linear projections when a device is
available and the projection reaches the shared output-size threshold; otherwise
it uses CPU kernels. `--compute vulkan` requires the text and vision projections
to execute on a real device and returns device errors without silently retrying
them on CPU. This is a mixed path: Mage-ViT RoPE/attention, text attention/KV,
norms, activations and output-head projection remain on CPU. Both F16/F32 KV
are supported. `RMI_COMPUTE_TRACE=1` reports linear backend and device counters.
The complete-device whitelist intentionally does not include `mage_vl`.
CLI generation requires `--prompt`; interactive mode,
audio, video, embedding, bench/profile, dump-logits and built-in template
presets are not supported. Image inputs require the matching Mage-ViT
projector. Image tokens stay at their original positions across turns and
inside interleaved content arrays. The total visual budget is 2048 tokens;
context/output budgets are checked before vision encoding.

Text/image JEV use the existing `--jev` CLI and `/v1/jev/score`, `/v1/jev/image`
routes. They read label logits with fresh ephemeral KV (F16 for text, F32 for
images) and disable thinking.

## Reproduce numerical checks

```sh
.venv/bin/python tools/oracle/mage_vl/reference.py \
  models/Mage-VL-Cowork artifacts/mage_vl/reference
cargo build --locked --profile release-fast --features parity-trace \
  --example mage_vl_oracle --bin rust-model-inference
for case in text square wide resize; do
  target/release-fast/examples/mage_vl_oracle \
    models/Mage-VL-Cowork-GGUF/Mage-VL-Cowork-bf16.gguf \
    models/Mage-VL-Cowork-GGUF/mmproj-Mage-VL-Cowork-F32.gguf \
    artifacts/mage_vl/reference/$case artifacts/mage_vl/native/$case
done
.venv/bin/python tools/oracle/mage_vl/compare.py \
  artifacts/mage_vl/reference artifacts/mage_vl/native
.venv/bin/python -m tools.oracle.mage_vl.check_export \
  models/Mage-VL-Cowork \
  models/Mage-VL-Cowork-GGUF/Mage-VL-Cowork-bf16.gguf \
  models/Mage-VL-Cowork-GGUF/mmproj-Mage-VL-Cowork-F32.gguf --text-type bf16
.venv/bin/python tools/oracle/mage_vl/smoke.py \
  models/Mage-VL-Cowork-GGUF/Mage-VL-Cowork-bf16.gguf \
  models/Mage-VL-Cowork-GGUF/mmproj-Mage-VL-Cowork-F32.gguf \
  artifacts/mage_vl/reference/square/image.png artifacts/mage_vl/smoke
.venv/bin/python tools/oracle/mage_vl/review_regression.py \
  models/Mage-VL-Cowork-GGUF/Mage-VL-Cowork-bf16.gguf \
  models/Mage-VL-Cowork-GGUF/mmproj-Mage-VL-Cowork-F32.gguf \
  artifacts/mage_vl/reference/square artifacts/mage_vl/review-fix
```

Repeat the native checks with the F32 text export; append `f16` to the probe
command and `--kv-format f16` to `compare.py` to check F16 cache continuation.
Language diagnostic traces use a
separate F32 session. The official Oracle uses the original BF16 weights
widened to F32, CPU attention, explicit raster patch positions and absolute
1D decode positions. Loading must report no missing/unexpected/mismatched
weights. No change is made to the downloaded official implementation.

Token IDs, template IDs and lossless weight mappings must be exact. SIMD
reductions, transcendental functions and F16 KV use numerical error gates.
Pillow bicubic is reused for resizing: the released torchvision Fast processor
can differ by 1–2 uint8 levels. That lossy case has a separate pixel/relative
activation gate and a strict final-logit gate; it is not claimed bitwise equal.
See [validation.md](validation.md) for measured evidence and limits.
The original full-weight evidence below covers BF16/F32 CPU only. New compressed
formats and mixed Vulkan need fresh full-weight CLI/HTTP/JEV and Oracle runs;
the original SSH host is shut down. Local synthetic hardware checks are recorded
separately in [MAGE_VL.md](../../../docs/develop/MAGE_VL.md) and do not establish
model quality or full-model performance.

On a hardware Vulkan weight host, build the same binaries with `--features vulkan`,
run `smoke.py ... --vulkan` and `review_regression.py ... --compute vulkan`, then
repeat the entry checks with Auto. Default `smoke.py` is the CPU/no-device check.
`check_export.py --text-type q8_0|q4_k` checks deterministic re-encoding and
storage, not an independent numerical quantization Oracle.

Text JEV single/multi/block scoring checks the configured context cap before KV
allocation; HTTP rejects that limit violation with 400. An explicit projector
is validated even for text-only CLI JEV. Mage image usage and count-token
responses include expanded visual tokens in all three HTTP protocols,
including streaming (the pinned 64x64 square fixture is 26 input tokens, not
its 23 unexpanded template IDs). The image-logits pipeline lives in
`models/mage_vl`; app code loads inputs and adapts protocol messages.
