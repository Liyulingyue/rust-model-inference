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

This checkpoint is a computer-use policy. In the current ordinary-chat probes,
both BF16 and Q4_K repeatedly emitted thinking text and exhausted a 512-token
budget; the red/blue fixture's blue square was described as a rectangle.
Do not treat numerical parity as a general chat-quality pass. For agent use,
provide the pinned model card's hybrid system prompt and declare its functions
in the HTTP `tools` field with `tool_choice:"auto"`. A system prompt naming
functions alone does not enable API tool calls.
In one synthetic Save-button request per BF16/Q4_K format, the responses
describe a click but actually return `computer_use {"action":"screenshot"}`;
neither completes the save task. Full responses and timing scope are in
[real-weight-results.json](real-weight-results.json).

`--compute auto` selects shared Vulkan linear projections when a device is
available and the projection reaches the shared output-size threshold; otherwise
it uses CPU kernels. `--compute vulkan` requires the text and vision projections
to execute on a real device and returns device errors without silently retrying
them on CPU. This is a mixed path: Mage-ViT RoPE/attention, text attention/KV,
norms and activations remain on CPU. The output head retains the shared
prepared-kernel dispatcher and can also offload through its legacy Vulkan
wrapper; it is outside the explicit projection executor's strict guarantee.
Both F16/F32 KV
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
  --example mage_vl_oracle
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
separate CPU F32 session. The probe accepts optional `[f32|f16] [cpu|auto|vulkan]`
arguments; only its vision encoding and logits exercise the selected policy.
Hidden language checkpoints remain CPU diagnostics. Build production CLI/server
binaries without `parity-trace`, which changes shared diagnostic rounding.
The official Oracle uses the original BF16 weights
widened to F32, CPU attention, explicit raster patch positions and absolute
1D decode positions. Loading must report no missing/unexpected/mismatched
weights. No change is made to the downloaded official implementation.

Token IDs, template IDs and lossless weight mappings must be exact. SIMD
reductions, transcendental functions and F16 KV use numerical error gates.
Pillow bicubic is reused for resizing: the released torchvision Fast processor
can differ by 1–2 uint8 levels. That lossy case has a separate pixel/relative
activation gate and a strict final-logit gate; it is not claimed bitwise equal.
See [validation.md](validation.md) for measured evidence and limits.
The current EPYC/RADV host completed BF16/Q8_0/Q4_K exports entirely remotely.
BF16 CPU production logits pass all four fixtures with F32/F16 KV, and BF16
mixed GPU logits pass text/square probes with actual device submissions.
Q8_0/Q4_K remain **experimental**: eight greedy choices agree in every fixture,
but independent same-weight F32-activation logit gates fail. Quantized image
CPU/GPU gates also fail; text-only CPU/GPU logits agree. The gates were not
widened. See [real-weight-results.json](real-weight-results.json),
[validation.md](validation.md) and the per-format
[coverage matrix](../../../docs/develop/MAGE_VL.md). The earlier port-31599
device-loss/disconnection attempt is retained as history, not a current blocker.

On a hardware Vulkan weight host, build the same binaries with `--features vulkan`,
run `smoke.py ... --vulkan` and `review_regression.py ... --compute vulkan`, then
repeat the entry checks with Auto. Default `smoke.py` is the CPU/no-device check.
`check_export.py --text-type q8_0|q4_k` checks deterministic re-encoding and
storage, not an independent numerical quantization Oracle.

For an independent same-weight quantized check, install upstream `gguf==0.18.0`
in the remote repository venv (or reuse the host's existing package). Run:

```sh
.venv/bin/python tools/oracle/mage_vl/reference.py \
  models/Mage-VL-Cowork artifacts/mage_vl/reference-q8_0 \
  --text-gguf models/Mage-VL-Cowork-q8_0/Mage-VL-Cowork-q8_0.gguf
cargo build --locked --profile release-fast --features vulkan --example mage_vl_oracle
for case in text square wide resize; do
  target/release-fast/examples/mage_vl_oracle \
    models/Mage-VL-Cowork-q8_0/Mage-VL-Cowork-q8_0.gguf \
    models/Mage-VL-Cowork-q8_0/mmproj-Mage-VL-Cowork-F32.gguf \
    artifacts/mage_vl/reference-q8_0/$case artifacts/mage_vl/native-q8_0/$case f32 cpu
done
.venv/bin/python tools/oracle/mage_vl/compare.py \
  artifacts/mage_vl/reference-q8_0 artifacts/mage_vl/native-q8_0 --quantized-logits
```

Repeat with Q4_K. The upstream GGUF package decodes all 399 exported text tensors
into the unchanged official F32 model; it does not call this converter's encoder
or Rust kernels. Named quantized-logit gates are max_abs `<0.5`, RMSE `<0.05`,
cosine `>0.9999`, with F32 KV. These gates were declared before the remote runs
and **failed for both Q8_0 and Q4_K**; activation quantization is not sufficient
evidence to waive a failed gate. An additional upstream Q8 activation diagnostic
reduced the error but did not pass, so the numerical root cause remains unresolved.
Eight logit artifacts are required per fixture. If greedy tokens diverge, only
logits with a common input prefix are compared, including the first divergent
choice; the report states that reduced coverage. This checks implementation
error for the same weights, not the quality loss from BF16 to Q4/Q8.
Oracle `timing.json` records CPU diagnostic timings including checkpoint hooks;
it is not a benchmark against a clean production binary. The native probe also
records `parity_trace_build`, compute policy, KV format and prefill/decode time.
Its prefill timer covers the text session and excludes image encoding, model
loading and diagnostic hidden capture. Use the ordinary HTTP timings in
[performance.json](performance.json) for request-level performance.

Text JEV single/multi/block scoring checks the configured context cap before KV
allocation; HTTP rejects that limit violation with 400. An explicit projector
is validated even for text-only CLI JEV. Mage image usage and count-token
responses include expanded visual tokens in all three HTTP protocols,
including streaming (the pinned 64x64 square fixture is 26 input tokens, not
its 23 unexpanded template IDs). The image-logits pipeline lives in
`models/mage_vl`; app code loads inputs and adapts protocol messages.
