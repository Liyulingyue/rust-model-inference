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

Conversion remains pinned to llama.cpp
`b96806d96061049a5b574269b049bf6241d63d46`.
Runtime validation is independently pinned to
`11fe02151f79c41d0d4af7da708755d73b9c0da6`, matching the existing Rust GDN
normalization: `x / sqrt(sum(x*x) + eps)`, implemented as RMSNorm plus scale.
The older version uses `x / max(sqrt(sum(x*x)), eps)`. These formulas are
different; the original comparison first diverges at `q_conv_predelta-0`
and fails the old strict recurrent contract. That result remains a failure,
and the existing Qwen3.5 tests and bounds are unchanged.

The new builder clones the fixed runtime reference into a disposable directory,
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
are exact checks. Accumulated arithmetic is numerical: for each checkpoint,
measure the maximum Oracle BF16-versus-widened-F32 relative L2 over all
forwards, and use `min(0.05, max(1e-5, 1.5 * spread))` for Rust-versus-BF16.
The Oracle spread itself must be at most 0.05. This follows the existing Holo4
calibration method with a smaller cap, without changing other models' gates.
All values must be finite. Widening changes activation rounding, so its output
is not expected to be bitwise identical to the BF16 path.

Vision validation sends identical normalized 256x256 RGB values to both
encoders. All 14 checkpoints (embedding, 12 blocks, projection) must agree in
order and shape, remain finite, and have relative L2 below the existing 0.05
vision bound. This isolates encoder arithmetic from lossy image resize.

## Measured CPU results

Apple M3 Max, Rust 1.98.1 / `release-fast`, with the fixed runtime reference:

| Selected checkpoint | Max Rust/BF16 relative L2 | Max Oracle F32/BF16 spread |
| --- | ---: | ---: |
| Recurrent Q, layer 0 | `7.856063e-8` | `0.001711933` |
| Recurrent state, layer 0 | `1.028575e-7` | `0.001460699` |
| Layer 0 output | `2.698261e-7` | `0.003260384` |
| Layer 23 output | `0.008075446` | `0.011696423` |
| Final logits | `0.008448516` | `0.006233681` |

Vision's largest relative L2 is `0.016452702` (block 11); final projection
relative L2 is `0.016178738`. The fixture produces 64 tokens of width 1024.
These are numerical results, not bitwise parity for accumulated arithmetic.

Converter regressions pass (6 tests), as do the focused Qwen3.5 library tests
(81 passed / 4 ignored), three reference-helper tests and both real-weight
OvisOCR3 reference tests. Rustfmt and `git diff --check` pass. The full
repository suite was not run.

The default CPU CLI with eight threads resizes a 32x32 red PNG to 256x256 /
64 tokens and answers `Red`. A separate 448x448 bitmap containing `HELLO 123`
produces 196 image tokens and the exact OCR output `HELLO 123`. These are
smoke tests, not an OCR accuracy benchmark.

## Reproduce

Prepare separate read-only llama.cpp checkouts at the conversion and runtime
pins above. Use a repository-local Python environment with the converter's
dependencies; no system packages or external acceleration libraries are needed.

```sh
.venv/bin/python -m tools.converter.ovisocr3.convert_ovisocr3 models/OvisOCR3 \
  --llama-cpp /path/to/llama-conversion-pin \
  --outfile target/ovisocr3/OvisOCR3-BF16.gguf \
  --mmproj-outfile target/ovisocr3/mmproj-OvisOCR3-BF16.gguf
sh tools/oracle/ovisocr3/build_oracles.sh /path/to/llama-runtime-pin \
  target/ovisocr3/oracle
PYTHONPATH=/path/to/llama-conversion-pin/gguf-py .venv/bin/python \
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
