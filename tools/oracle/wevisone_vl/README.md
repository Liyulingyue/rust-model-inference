# WeVisOne-VL-4B: CPU, Auto and alpha decoder

The released model is Qwen3-VL with an expanded coordinate vocabulary and a
separate 9.6M-parameter FP32 alpha decoder. It uses the explicit `qwen3vl`
text architecture, `clip`/`qwen3vl_merger` vision projector and `wevisone_vl`
pixel decoder. No external acceleration dependency was added.

Pinned inputs:

- Tencent source `https://github.com/Tencent/WeVisOne-VL`, commit
  `9a6bcd85404505df0d05cfb5d97512ccacf35dee` (unmodified Oracle).
- ModelScope `tencent-community/WeVisOne-VL-4B`, revision
  `855d2a73ac0b929f668e6ac4f5deccc667aba99b`.
- `model.safetensors`, 9,661,250,408 bytes, SHA256
  `002c3ac0783af7c86dc28b282c84dd4bcd0c29dcb3b5e0d584492fe3212acb4a`.
- `mask_decoder.pt`, 38,534,741 bytes, SHA256
  `8528460481bd09d0780c53559b08194c7abbd29dff0d51783b619158acbe7b46`.

## Convert and run

Download the pinned model's config, tokenizer, template, main weights and
`mask_decoder.pt` into `models/WeVisOne-VL-4B`. Use the repository Python
venv with numpy and torch for conversion/reference checks.

```sh
.venv/bin/python -m tools.converter.wevisone_vl.convert_wevisone_vl \
  models/WeVisOne-VL-4B --out-dir models/WeVisOne-VL-GGUF --text-type bf16
cargo build --profile release-fast --features vulkan --bin rust-model-inference
```

Text export types: `f32`, `bf16`, `f16`, `q8_0`, `q4_k`, `q6_k`. Norm vectors,
vision and pixel decoder remain F32. An output directory must be new; existing
files are never overwritten. The vocabulary has 152679 tokenizer entries but
152680 embedding/output rows: the last GGUF entry is an empty unused row, not
an invented token. `<OBJ>` remains 152678.

`request.json`:

```json
{"image":"image.png","task":"segmentation","query":"the person","max_new_tokens":512}
```

```sh
target/release-fast/rust-model-inference \
  --model models/WeVisOne-VL-GGUF/WeVisOne-VL-4B-bf16.gguf \
  --mmproj models/WeVisOne-VL-GGUF/mmproj-WeVisOne-VL-4B-F32.gguf \
  --wevisone-decoder models/WeVisOne-VL-GGUF/WeVisOne-VL-decoder-F32.gguf \
  --wevisone-request request.json --output result.json --compute auto --threads 8
```

Tasks: detection, referring, point, count_box, count_point, segmentation,
matting, ocr, ocr_grounding, visual_prompt, freeform. Detection labels are
newline-separated. `visual_prompt` takes `reference_boxes` as pixel-space xyxy
arrays. The shared prediction route uses the released system/task prompts,
greedy generation and repetition penalty 1.05 by default. CLI generation
settings override the defaults; `--max-tokens` also caps a JSON token budget.
`--max-context`, `--prefill-batch-size` and generation `--kv-cache` are honored;
mask teacher replay uses fresh F32 KV. Unrelated CLI modes are rejected.

The response contains `response`, generated `token_ids`, normalized OBJ
`boxes`, original `width`/`height`, and one grayscale PNG data URI in
`alpha_png` per box. PNG alpha is rounded to 8 bits. Sigmoid happens before
resizing, matching the official engine. Other object outputs remain in the
model's response markup. Any task that emits OBJ needs the decoder; missing
components produce an error. There is no batch prediction API.

Start the same command with `--serve --host 127.0.0.1 --port 8080` instead of
`--wevisone-request`/`--output`. POST the same JSON to
`/v1/wevisone/predict`. HTTP accepts image data URIs/local paths; remote URLs
require the existing `--allow-remote-images` option. Ordinary
`/v1/chat/completions`, text/image JEV and CLI `--jev --image` use the existing
Qwen3-VL routes. Chat returns markup; use the dedicated prediction endpoint
for alpha masks.

## Dispatch and limits

- CPU uses existing precision/SIMD dispatch. BF16 retains checkpoint weights;
  F16 activation rounding and quantized formats are approximate.
- `--gpu` aliases `--compute auto`. In a Vulkan build, Auto offloads the dense
  backbone projections through the existing LinearExecutor, while preserving
  host attention/KV, DeepStack and raw OBJ states. Unsupported/unavailable GPU
  operations fall back to existing CPU kernels. Trace with
  `RMI_COMPUTE_TRACE=1`; a CPU host trace is not a full-GPU execution claim.
- `--compute vulkan` is explicitly rejected: the resident Qwen3-VL decoder
  does not implement the complete DeepStack/raw-hidden contract. Auto is the
  supported mixed CPU/GPU policy.
- Pixel-decoder tensors/computation remain F32, including when text is quantized.
  Vision is preserved in F32 instead of the older implicit Q8 precompute.
- Prediction is single-image, at most 4096 generated tokens, 64 masks, and
  64M total original-size alpha pixels. CLI JSON is at most 1MiB. The configured
  total context cap is enforced before embedding/KV allocation and on replay.

## Reproduce alignment

Floating point reductions, SIMD/FMA, F16 and quantization use numerical
metrics; they are not required or claimed to be bit-identical. Token IDs,
positions, tensor layouts and greedy IDs on the BF16 fixture are exact.

```sh
.venv/bin/python -m tools.oracle.wevisone_vl.run_oracle \
  --source reference/WeVisOne-VL --model models/WeVisOne-VL-4B \
  --out artifacts/wevisone/oracle
WEVISONE_FIXTURE=artifacts/wevisone/oracle \
  cargo test --profile release-fast --lib official_decoder_fixture -- --ignored --nocapture
.venv/bin/python -m tools.oracle.wevisone_vl.backbone \
  --source reference/WeVisOne-VL --model models/WeVisOne-VL-4B \
  --out artifacts/wevisone/backbone
WEVISONE_BACKBONE=artifacts/wevisone/backbone \
WEVISONE_MODEL=models/WeVisOne-VL-GGUF/WeVisOne-VL-4B-bf16.gguf \
WEVISONE_VISION=models/WeVisOne-VL-GGUF/mmproj-WeVisOne-VL-4B-F32.gguf \
WEVISONE_DECODER=models/WeVisOne-VL-GGUF/WeVisOne-VL-decoder-F32.gguf \
WEVISONE_MASKS=1 \
  cargo test --profile release-fast --lib official_backbone_fixture -- --ignored --nocapture
```

For F16/quantized text, set `WEVISONE_APPROXIMATE=1` and select that text
GGUF. For Auto add `WEVISONE_AUTO=1`; use `--features vulkan` to exercise
available GPU kernels, or omit it to test CPU fallback. Metrics print max
absolute error, RMSE, relative RMSE and cosine. GPU presence alone is not
proof of execution: inspect `backend=Vulkan` and submission counters.

Measured coverage and limitations are recorded in [validation.md](validation.md).
