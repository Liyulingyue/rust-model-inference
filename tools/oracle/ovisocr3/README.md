# OvisOCR3 validation

[ATH-MaaS/OvisOCR3](https://modelscope.cn/models/ATH-MaaS/OvisOCR3/files)
declares `Qwen3_5ForConditionalGeneration`, not a new model architecture.
Its 24-layer hybrid GDN/attention decoder and 12-layer vision encoder use the
existing `qwen35` and `qwen3vl_merger` implementations. This adapter changes
conversion and input metadata, without adding an architecture alias or changing
production operators.

## Input contract and required adaptation

The decoder has hidden size 1024, FFN 3584, 8 query / 2 KV heads of width 256,
and full attention at layers 3, 7, 11, 15, 19 and 23. MRoPE uses sections
`[11,11,10,0]`, rotary dimension 64 and frequency base 10000000.
Vision has hidden size 768, FFN 3072, 12 heads, patch 16, temporal patch 2,
merge 2, output width 1024 and no DeepStack layers.

The converter validates 473 source tensor names, shapes and BF16 dtypes.
It preserves the tokenizer's NFC normalizer and exports the image processor's
saved `size.shortest_edge=65536` / `size.longest_edge=16777216` as
`clip.vision.image_min_pixels` / `clip.vision.image_max_pixels`.
The runtime already reads these keys. A 32x32 image must become 256x256 with
64 image tokens; omitting the metadata incorrectly selects 96x96 and 9 tokens.
The regression also checks the upper boundary: 8192x8192 becomes 4096x4096.
Image mean/std are both `[0.5,0.5,0.5]`.

The model README's `448*448` / `2880*2880` values are vLLM request overrides.
They are not the saved defaults and are not silently baked into this adapter.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Source `model.safetensors` | 1706030528 | `aa2311c6b22b07cf3afe9aac6e8bcb02a2b19dd8671b2a91d0b144071577e600` |
| Converted text GGUF | 1516744704 | `6e080ba3bcab9cdb81b600b9af714d20d2b1c30875f622de839c0592c89ed60a` |
| Converted mmproj GGUF | 207346528 | `b49907ed4f82896f16f8844bdada616932b46647ea0b15e4c4c4eebd54958831` |
| Widened text reference | 3020533760 | `bb07f1a96e84e9aaf7565fe1e30666e3fef0c11a85426396fd3e1679ab86508e` |

The GGUF text file contains 187 BF16 and 133 F32 tensors; the mmproj contains
50 BF16 and 104 F32 tensors. The widened reference stores the same weight
values in F32; the widening tool checks their raw F32 words before returning.
It is a diagnostic artifact, not a new published weight format.

## Reference versions and precision

Conversion and runtime validation share one pin, llama.cpp
`b96806d96061049a5b574269b049bf6241d63d46`.

That was not always the case. The runtime Oracle used to sit on
`11fe02151f79c41d0d4af7da708755d73b9c0da6`, which builds the GDN q/k
normalization as `x / sqrt(sum(x*x) + eps)` (RMSNorm plus scale); upstream
moved to `x / max(sqrt(sum(x*x)), eps)` (`ggml_l2_norm`) between `11fe0215`
and `b96806d`. Because this was the last Oracle on the old formula, the shared
Rust implementation had been pinned to the old formula to match it, which in
turn pulled every model whose Oracle was already `b96806d` off its reference.
The runtime Oracle now uses the conversion pin, so both sides run
`ggml_l2_norm` and the exception is gone.

With both sides on `b96806d` the recurrent checkpoints come back to noise
level: `q_conv_predelta-0` `7.8e-8`, `state_predelta-0` `1.1e-7`,
`conv_output_raw-0` exactly `0`, `final_output-0` `1.7e-7`. Switching only the
GDN formula back (`RMI_QWEN35_GDN_LEGACY_NORM=1`) puts them at `1.18e-5` and
`4.7e-5` again, which is the formula difference the old pin used to hide. The
switch is kept as a diagnostic -- no Oracle needs it any more -- and it is off
by default on the CPU path only.

The builder clones the fixed runtime reference into a disposable directory,
reuses the existing callback/precise-softmax instrumentation, and verifies that
no `src/` model formulas were patched. BLAS, Accelerate and Metal are disabled.

Text validation uses `RMI_SCALAR=1`, one CPU thread, F32 KV, prompt `你好`,
the embedded non-thinking chat template and four greedy steps. Both references
and Rust produce IDs `[109266,6115,103724,97319]`.
All 325 selected trace records have matching order, names, layer/step/occurrence,
shapes and structural values. The numerical checkpoints sample recurrent layer
0, dense layers 3 and 23, final normalization and logits over 16 forwards;
this does not trace every intermediate of all 24 layers.

Token IDs, layer flags, MRoPE positions, embedding lookup and initial zero state
are exact checks. The numerical checkpoints fall into two classes:

* Single-block checkpoints (`*_predelta-0`, `conv_output_raw-0`,
  `final_output-0`) are bounded against the Oracle's own BF16-versus-widened-F32
  spread, `min(0.05, max(1e-5, 1.5 * spread))`. They are effectively lossless
  and sit three to five orders of magnitude below that bound.
* Accumulated checkpoints (deep `Qcur`/`Kcur`/`attn_norm`, `layer_output-*`,
  `result_norm`, `result_output`) are reported and capped at 0.05 relative L2,
  the same cap the vision check uses. A spread multiplier carries no
  information there: the deep stack is chaotic in BF16, and moving
  `state_predelta-0` by the 1e-5 formula difference swings `layer_output-23`
  between `0.0117` and `0.0195` while leaving `result_output` at `4.4x` the
  Oracle spread. The greedy ids are the functional gate for those layers, and
  they are compared exactly.

The Oracle spread itself must be at most 0.05, and all values must be finite.
Widening changes activation rounding, so its output is not expected to be
bitwise identical to the BF16 path.

Vision validation sends identical normalized 256x256 RGB values to both
encoders. All 14 checkpoints (embedding, 12 blocks, projection) must agree in
order and shape, remain finite, and have relative L2 below the existing 0.05
vision bound. This isolates encoder arithmetic from lossy image resize.

## Measured CPU results

Linux aarch64 (NVIDIA GB10), `release-fast --features parity-trace`,
`RMI_SCALAR=1`, one thread, with the runtime Oracle on the conversion pin
`b96806d`:

| Selected checkpoint | Max Rust/BF16 relative L2 | Max Oracle F32/BF16 spread |
| --- | ---: | ---: |
| Recurrent Q, layer 0 (`q_conv_predelta-0`) | `7.771642e-8` | `0.001711939` |
| Recurrent state, layer 0 | `1.101880e-7` | `0.001460709` |
| Conv output, layer 0 | `0` (exact) | `0.000582941` |
| Layer 0 output | `9.421110e-5` | `0.003262944` |
| Layer 3 output | `0.004289493` | `0.005203881` |
| Layer 23 output | `0.019521800` | `0.009581441` |
| Final logits (`result_output`) | `0.014644887` | `0.003310246` |

With the legacy GDN formula the two recurrent rows read `1.179861e-5` and
`4.697174e-5` instead, which is the formula difference the old runtime pin
used to require.

Vision's largest relative L2 is `0.014742355` (block 11); final projection
relative L2 is `0.013866748`. The fixture produces 64 tokens of width 1024.
These are numerical results, not bitwise parity for accumulated arithmetic.
The earlier table on this page was measured on an Apple M3 Max against the
`11fe0215` runtime Oracle; those rows are superseded by the ones above.

Converter regressions pass (6 tests), as do the focused Qwen3.5 library tests
(81 passed / 4 ignored), three reference-helper tests and both real-weight
OvisOCR3 reference tests. Rustfmt and `git diff --check` pass. The full
repository suite was not run.

The default CPU CLI with eight threads resizes a 32x32 red PNG to 256x256 /
64 tokens and answers `Red`. A separate 448x448 bitmap containing `HELLO 123`
produces 196 image tokens and the exact OCR output `HELLO 123`. These are
smoke tests, not an OCR accuracy benchmark.

## Reproduce

Prepare one read-only llama.cpp checkout at the pin above (both the conversion
and the runtime Oracle use it). Use a repository-local Python environment with
the converter's dependencies; no system packages or external acceleration
libraries are needed.

```sh
.venv/bin/python -m tools.converter.ovisocr3.convert_ovisocr3 models/OvisOCR3 \
  --llama-cpp /path/to/llama-b96806d \
  --outfile target/ovisocr3/OvisOCR3-BF16.gguf \
  --mmproj-outfile target/ovisocr3/mmproj-OvisOCR3-BF16.gguf
sh tools/oracle/ovisocr3/build_oracles.sh /path/to/llama-b96806d \
  target/ovisocr3/oracle
PYTHONPATH=/path/to/llama-b96806d/gguf-py .venv/bin/python \
  tools/oracle/holo4/widen_projector.py target/ovisocr3/OvisOCR3-BF16.gguf \
  target/ovisocr3/OvisOCR3-F32-reference.gguf

RMI_SCALAR=1 \
RMI_LLAMA_CPP="$PWD/target/ovisocr3/oracle/llama.cpp" \
RMI_OVISOCR3_MODEL="$PWD/target/ovisocr3/OvisOCR3-BF16.gguf" \
RMI_OVISOCR3_F32_MODEL="$PWD/target/ovisocr3/OvisOCR3-F32-reference.gguf" \
RMI_OVISOCR3_ORACLE="$PWD/target/ovisocr3/oracle/build/bin/llama-eval-callback" \
RMI_OVISOCR3_MMPROJ="$PWD/target/ovisocr3/mmproj-OvisOCR3-BF16.gguf" \
RMI_OVISOCR3_VISION_ORACLE="$PWD/target/ovisocr3/oracle/build/bin/rmi-ovisocr3-vision" \
cargo test --locked --profile release-fast --features parity-trace \
  --test qwen35_reference ovisocr3_ -- --ignored --nocapture --test-threads=1
```

The model weights, build products and trace payloads stay in ignored local
directories. The tests print their artifact directories and numerical maxima.

## Coverage limits

Validation covers the fixed CPU text sample, normalized-image encoder fixture,
pixel-budget boundaries, and real CLI image/OCR smoke tests. It does not measure
full-page OCR accuracy, cover other quantizations, validate Vulkan, or establish
long-context or cross-platform parity. The configured maximum pixel budget is
not a memory guarantee: the current vision attention scratch grows quadratically
with patch count. Large-page memory behavior has not been tested.
