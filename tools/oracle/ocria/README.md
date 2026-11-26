# OCRia-nano-1

Page OCR with the pinned [ModelScope release](https://modelscope.cn/models/llamaindex/OCRia-nano-1):
a text-line detector, a script router and four CTC recognizers, packed into one
`ocria` GGUF and run on the repository's own convolution operators. Rust
inference uses no Python, ONNX Runtime or external acceleration library.

`sources.json` records the per-file revision, byte count and SHA256 of the
release, the pinned LiteParse commit that defines the pipeline around the
graphs, and the license. `graph_notes.md` is the op-by-op reading of the six
graphs the Rust implementation was written against.

## Convert and run

```sh
python3 tools/oracle/ocria/download.py /tmp/OCRia-nano-1
python3 -m venv /tmp/ocria-venv
/tmp/ocria-venv/bin/pip install -r tools/oracle/ocria/requirements.txt
/tmp/ocria-venv/bin/python -m tools.converter.ocria.convert_ocria \
  /tmp/OCRia-nano-1 models/OCRia-nano-1/ocria-f32.gguf
cargo build --locked --profile release-fast --bin rust-model-inference --bin rust-model-server
target/release-fast/rust-model-inference --model models/OCRia-nano-1/ocria-f32.gguf \
  --image page.png --out lines.json --threads 4 --compute cpu
```

The CLI writes the page as JSON: `width`, `height` and one entry per line with
`text`, `bbox` (`[x1, y1, x2, y2]` in page pixels), `confidence` (the mean of
the kept CTC character confidences), `polygon` and `faint`. `--out` is
optional; without it the JSON goes to stdout.

HTTP serves the same document:

```sh
target/release-fast/rust-model-server --model models/OCRia-nano-1/ocria-f32.gguf \
  --host 127.0.0.1 --port 8080
curl -f http://127.0.0.1:8080/v1/ocr -H 'Content-Type: application/json' \
  -d "{\"image\":\"data:image/png;base64,$(base64 -i page.png)\"}"
```

`image` accepts a `data:` URL or bare base64; remote URLs are refused. The
input is a page image — this repository has no PDF renderer, so PDF input,
the release's PDF text layer and LayoutNet are the second delivery layer (see
[Limits](#limits) and [layout-plan.md](layout-plan.md)).

## Weight storage

| Format | Command | File size | Usable |
| --- | --- | --- | --- |
| `ocria-nano-1-f32-v1` | default | 15,571,808 bytes | yes |
| `ocria-nano-1-q8_0-v1` | `--quant q8_0` | 4,561,728 bytes | yes, with the measured weight error below |

The `f32` export is lossless: every tensor equals the ONNX initializer bit for
bit (the `rec_latin` int8 QDQ weights are dequantized with the ONNX
`y = (x - zero_point) * scale` arithmetic, which is what ONNX Runtime's own
dequantized weights hold).

The `q8_0` export stores the two-dimensional matrices whose row length is a
whole number of 32-value blocks — the pointwise convolutions, the recognizers'
`mix` `fc` convolutions, the head convolutions and the router's `Gemm`
matrices — in Q8_0, and keeps the depthwise 3x3 kernels (row length 9), the
9-tap `mix` kernels and every bias in F32. OCRia computes in F32 whatever the
file stores. See [validation.md](validation.md) for the measured effect on the
line output.

## Numerical oracle

Two layers of evidence, both pinned to the release and to LiteParse
`44ceb6ab160c64cc8b43aab4ed85251719495629`:

1. **Per-graph fixtures** — `trace.py` runs each of the six graphs in ONNX
   Runtime on a deterministic input and records the network input tensor and
   the graph's own outputs. `src/models/ocria/tests.rs` asserts on the same
   tensors through `Detector::forward_prob`, `Router::forward_logits` and
   `Recognizer::forward_logits`, i.e. before any CTC collapse or page policy.
   The two detector cases are a square 256x256 page and a non-square 96x160 one
   (the FPN pairs a 2x upsample with `ceil(n/2)` strides, which a square page
   alone can hide); the recognizer cases cover four widths per group so both
   timestep families (`W/4`, `W/8`) and the padded-width bucket are exercised.
   `rec_latin`'s released graph is ONNX-Runtime-quantized and snaps its
   activations inside the graph, which this crate does not reproduce, so its
   cases carry **two** answers: `ids_f32`/`conf_f32` from the faithful F32 graph
   (the target the Rust code must hit exactly) and `ids`/`conf` from the pinned
   QDQ graph (the official runtime's answer). The measured difference between
   the two is `latin_qdq_effect.json`, summarized in [validation.md](validation.md).
2. **Page fixtures** — `fixtures/pages/` holds three deterministic synthetic
   pages and, next to each, the line list the pinned LiteParse engine produced
   for it (`receipt`, `two_column`, `sideways`). `page_read_matches_liteparse`
   compares this crate's `Model::run` against them: text and line count
   exactly, boxes within 1 px, confidence within 1e-4.

```sh
OCRIA_GGUF=models/OCRia-nano-1/ocria-f32.gguf \
OCRIA_FIXTURES=tools/oracle/ocria/fixtures \
OCRIA_PAGES=tools/oracle/ocria/fixtures/pages \
cargo test --locked --profile release-fast --lib models::ocria::tests \
  -- --ignored --nocapture
```

The page images are rendered by `make_pages.py` and are byte-identical when
regenerated; the oracle JSON next to each was recorded with a small example
added to a temporary copy of the pinned LiteParse checkout (`ocria_dump.rs`),
which calls the engine directly so the coordinates are image pixels and not
PDF points.

Measured values, the gates and the q8_0 result are in [validation.md](validation.md).

## Limits

- **CPU only.** The release is a CPU pipeline and the graphs are
  depthwise-separable convolutions the repository's Vulkan kernels do not
  cover (they serve linear projections), so `--compute auto` runs on the host
  and `--compute vulkan` is refused rather than silently downgraded. There is
  no device path to measure, so no performance comparison against one.
- **Layer 1 of 2.** This delivers the OCR stack: detector, router, four
  recognizers and the page pipeline around them. LayoutNet — the release's
  table and layout-block model (`tablenet.onnx`, `tablenet.bin`, 4.27M
  parameters) — is **not** implemented; `layout-plan.md` records what it
  needs, what of that this repository already has, and the measured size of
  the remaining work.
- **Images only.** No PDF rendering, so no PDF text layer and no
  multi-page documents; one request reads one page image.
- **No device or precision claims beyond the measured ones.** The recognizers
  compute in F32; `rec_latin`'s activation quantization is not reproduced (the
  release's own graph quantizes activations at run time), and the effect on
  `ids` is measured rather than assumed.

## Attribution

The weights are released under the [OCRia-nano-1 Model License](https://modelscope.cn/models/llamaindex/OCRia-nano-1/resolve/master/LICENSE)
(AI Pubs Open RAIL-M based). LiteParse is Apache-2.0 and is used here only as a
read-only numerical oracle; no LiteParse code is vendored into this
repository.
