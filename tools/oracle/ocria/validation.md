# OCRia-nano-1: oracle, conversion and parity validation

What was pinned, what was converted, what was checked, and what is still unverified.
Commands are the ones that ran; they were run on the remote Linux box (128 cores, no
GPU) and, after that box went down, again locally on macOS with the same Python
package versions.

## 1. Pinned inputs

| item | pin |
| --- | --- |
| model | `llamaindex/OCRia-nano-1` on ModelScope, `master` @ `ca290e82458d32aa8b605dbf3248d7aaf895ea7e` |
| files | 18 downloaded: 16 layer-1 (6 graphs, 2 metas, 4 charsets, licence, README, 2 configs) + 2 layer-2 (`tablenet.onnx`, `tablenet.bin`), each with its own last-commit revision, byte count and SHA256 in [`sources.json`](sources.json) |
| reference runtime | LiteParse `44ceb6ab160c64cc8b43aab4ed85251719495629`, `crates/liteparse/src/ocr/ocria/` |
| licence | `LICENSE` (OCRia-nano-1 Model License, AI Pubs Open RAIL-M), embedded in the GGUF as `ocria.licenses` |

The ModelScope files API supplied the revision/size/SHA256 of every file;
`download.py` re-downloads only what does not match and refuses a file whose size or
hash differs. The six ONNX hashes were re-computed on the box and match the API's
byte for byte, so the pins are the files' own content hashes and not LFS pointer
hashes.

## 2. Environment

Python 3.12.3 (box) and 3.14.7 (local macOS fallback), both with `numpy==2.5.5`,
`onnx==1.23.2`, `onnxruntime==1.32.0` (CPU execution provider), per
[`requirements.txt`](requirements.txt). Fixture sessions run with
`intra_op_num_threads=1`, `inter_op_num_threads=1`, ONNX Runtime's default
optimisation level.

## 3. Conversion

```bash
python -m tools.converter.ocria.convert_ocria /path/to/OCRia-nano-1 out-f32.gguf --quant f32
python -m tools.converter.ocria.convert_ocria /path/to/OCRia-nano-1 out-q8_0.gguf --quant q8_0
```

| file | size | tensors | F32 | Q8_0 | wall time |
| --- | --- | --- | --- | --- | --- |
| `ocria-nano-1-f32.gguf` | 15 571 808 B | 342 | 342 | 0 | 0.24 s |
| `ocria-nano-1-q8_0.gguf` | 4 561 728 B | 342 | 257 | 85 | 0.35 s |

Tensor counts per component: `det` 74, `router` 34, `rec.latin` 58, `rec.cjk` 60,
`rec.rtl` 58, `rec.world` 58 — one weight per Conv/Gemm plus a bias wherever the node
has a third input (`det.lat.*`, `det.squeeze`, `det.lat2` have none).

`q8_0` stores a tensor in Q8_0 only when it is 2-D and its row length is a whole
number of 32-value blocks. That is a single uniform rule, applied by `quantizable`,
mirrored by `load_tensor` in `src/models/ocria/mod.rs`, and the converter **raises**
if a quantizable tensor would be left in F32 rather than silently keeping it. The 85
Q8_0 tensors are the pointwise convs with `in % 32 == 0`, the `mix` `fc1`/`fc2`
matrices, the four heads and the router's two `Gemm`s. The 257 F32 tensors are:

* every bias (1-D) — 168 of them;
* every depthwise 3x3 kernel (`[C,9]`, row length 9) and the 9-tap `mix` `dw` kernels
  (`[192,9]`) — 77 of them;
* 12 pointwise convs whose input channels are not a multiple of 32:
  `det.s4.s4.0.pw` `[24,16]`, `det.s4.s4.1.pw` `[24,24]`, `det.s8.s8.0.pw` `[48,24]`,
  `det.s8.s8.1.pw` `[48,48]`, `det.s8.s8.2.pw` `[48,48]`, `det.s16.s16.0.pw` `[96,48]`,
  `det.lat.0` `[32,24]`, `det.lat2` `[12,16]`, `det.head.head.0.head.0.3` `[16,12]`,
  `det.head.head.3` `[1,16]`, `router.body.body.0.pw` `[32,16]`, `det.lat.1` `[32,48]`.

## 4. What the round-trip check proved

`tools/converter/ocria/check_ocria.py` reads the produced file back with the shared
GGUF reader and compares it against the ONNX graphs the converter just read
(`numpy_helper.to_array`, or the QDQ chain for `rec_latin`):

```bash
python -m tools.converter.ocria.check_ocria /path/to/OCRia-nano-1 out-f32.gguf --quant f32
# out-f32.gguf: 342 tensors, F32 342, Q8_0 0, 15571808 bytes
python -m tools.converter.ocria.check_ocria /path/to/OCRia-nano-1 out-q8_0.gguf --quant q8_0
# out-q8_0.gguf: 342 tensors, F32 257, Q8_0 85, 4561728 bytes
```

* **set equality both ways** — the file holds exactly the 342 keys of
  `src/models/ocria/contract.json`, no missing and no extra tensor;
* **dimensions** — every tensor's GGUF `dims` equal the contract shape reversed, so
  the loader's `dims == shape.reversed()` check passes for all 342;
* **F32 payloads** — byte-identical to the ONNX array, in both files, including the
  257 F32 tensors of the `q8_0` file;
* **Q8_0 payloads** — byte-identical to `quantize_q8_0` of the same array. This
  compares the stored blocks (scale and int8 payload), not dequantized values, so a
  change in block layout or rounding cannot pass;
* **storage rule** — in the `q8_0` file every 2-D tensor with a 32-multiple row is
  Q8_0 and every other tensor is F32, so no tensor is silently kept at higher
  precision;
* **metadata** — `general.architecture=ocria`, `ocria.format` the documented string
  for the format, `ocria.config` byte-equal to
  `{"det":{"thresh":0.55,"merge":0.6,"long_side":1600},"rec":{"groups":["latin","cjk","rtl","world"],"visual_rtl":[false,false,true,false]}}`,
  the four `ocria.rec.<group>.charset` strings parsing to `classes - 1` entries each
  (791/6213/323/1711 against 792/6214/324/1712), and the 13 `ocria.source.files`
  hashes equal to `sources.json`.

Contract-level checks the converter enforces, and which the run exercised:
`det.meta.json`'s `thresh` equals the contract's, `rec_rtl.meta.json`'s `visual_rtl`
matches the contract's `[false,false,true,false]`, every charset is a non-empty array
of non-empty strings, each recognizer's head has exactly `len(charset) + 1` classes,
every tensor shape equals `contract.json`, and every value is finite. The tensor map
was also cross-checked independently against the raw node listings in
`tools/oracle/ocria/dumps/`: every Conv/Gemm weight and bias there appears in the
contract with the flattened shape and none is missing.

## 5. Fixtures

```bash
python -m tools.oracle.ocria.trace /path/to/OCRia-nano-1 tools/oracle/ocria/fixtures
# tools/oracle/ocria/fixtures: 2 det, 2 router, 16 rec cases, 1958619 bytes
```

`manifest.json` plus raw little-endian blobs, read by `src/models/ocria/tests.rs`.
The manifest's `note` says which half is which:

* `det` — two cases. `lines` is a 256x256 synthetic page (six text-like lines of
  fixed strokes; about 18 % of the resulting `prob` map is above the 0.55 threshold,
  so it exercises the threshold and component path, not just the arithmetic) with
  the normalised `(1,3,256,256)` network input and the `(1,1,128,128)` `prob` map.
  `narrow` is 96x160, both odd multiples of 32: the FPN pairs a 2x nearest upsample
  with a `ceil(n/2)` stride, and a square page alone can hide a wrong pairing.
  `--with-intermediates` additionally writes the stem Relu and the deepest FPN sum
  (1.5 MB at 256x256, so they are not committed).
* `router` — two cases (`latin`, `cjk`), two 32x512 crops each, with the raw `group`
  `(2,4)` and `flip` `(2,2)` **logits** (no softmax).
* `rec` — every group at widths 64, 256, 384 and 512, one crop each, with the graph's
  own `ids` (i64) and `conf` (f32). 64 and 512 are the bucket ends, 256 and 384 sit
  on the timestep boundary of the two stride families. The counts confirm the
  per-group stride: cjk `W/8` (8/32/48/64), latin/rtl/world `W/4` (16/64/96/128).
  **`latin` also carries `ids_f32`/`conf_f32`**, the output of the F32 graph Rust
  must reproduce exactly (§6); its `ids`/`conf` are the pinned QDQ graph's answer.

Every input is deterministic arithmetic on a fixed pattern — no RNG, no clock — and
`onnx_sha256` in the manifest pins the six graphs the fixtures came from, so a
fixture cannot silently drift from `sources.json`. The nets fixtures total
1 958 619 B; with the three page PNGs the directory is 2 064 013 B (1.97 MiB).

`trace.py` writes only its own files and preserves any manifest key it does not own,
so regenerating it never disturbs `fixtures/pages/`.

## 6. `rec_latin`'s activation quantization: measured effect on `ids`/`conf`

`rec_latin.onnx` is the one quantized graph. Its weights are dequantized into the
GGUF (bit-exact per §4), but its activation `QuantizeLinear`/`DequantizeLinear` pairs
are not reproduced: Rust computes in F32. `tools/oracle/ocria/qdq_effect.py` builds
the faithful F32 reference — the same dequantized weights, the activation pairs
rewired out, the Relus that the quantizer folded into them put back (see
`graph_notes.md` §8.1) — and compares the two graphs' `ids` and `conf`:

```bash
python -m tools.oracle.ocria.qdq_effect /path/to/OCRia-nano-1 rec_latin_qdq.json \
    --page receipt.png
```

| input | timesteps | `ids` differing | max abs Δ`conf` | decoded text |
| --- | --- | --- | --- | --- |
| fixtures `w64` | 16 | 1 (t=6) | 0.15 | `I- 1LI` vs `I 1LI` |
| fixtures `w256` | 64 | 3 (t=6, 20, 34) | 0.19 | `H- 1LI- 1LI- …` vs `H- ILI- ILI- …` |
| fixtures `w384` | 96 | 6 | 0.19 | `ILI- ILI- …` vs `ILI ILI- …` |
| fixtures `w512` | 128 | 8 | 0.19 | `ЦI- 1LI- …` vs `ЦI 1LI- …` |
| **fixtures total** | **304** | **18 (5.92 %)** | | |
| real receipt text, 10 bands of 32 px | 1600 | **9 (0.56 %)** | 0.43 | identical on 7 of 10 bands |

The per-case table, with the exact differing timesteps and both decoded texts, is
committed as [`latin_qdq_effect.json`](latin_qdq_effect.json); the fixture
`ids_f32` blobs differ from the QDQ `ids` at exactly the timesteps it lists.

The fixtures are a hostile case on purpose: a repeated stroke pattern leaves many
near-ties (the top-2 logit gap at the differing timesteps is 0.04–0.56, inside or
next to one 0.333 logit bin), so the argmax follows the perturbation 5.2 % of the
time. Real glyphs have wide margins: `CORNER STORE` and `Coffee beans 2 12.50` read
identically on both sides, and the differing bands differ by one character
(`Thank yourshoping` vs `Thank yufrshoping`). Activation quantization moves the
logits by up to ~9 in the tail (it compounds through 29 layers), but the argmax
rarely follows.

Consequences, stated plainly: `cjk`, `rtl` and `world` are plain F32 graphs and must
match their fixtures exactly. `latin` matches **except** for those timesteps, so its
parity target should be the F32 reference emitted by this script, not the QDQ graph.
The precision loss is in `ids`/`conf`, not in the weights, and no other part of the
pipeline is affected.

## 7. Rust parity against these fixtures

`src/models/ocria/tests.rs` (the `nets` teammate) reads this fixture set and compares
the crate's own operators against it. Measured on the committed files, 1064
recognizer timesteps over four groups and four widths:

| comparison | result |
| --- | --- |
| `ids` vs `ids_f32` (the F32 target), all four groups | **0 / 1064 timesteps differ** |
| `ids` vs the pinned QDQ graph, cjk + rtl + world | **0** |
| `ids` vs the pinned QDQ graph, latin | **18 / 304**, exactly the timesteps `latin_qdq_effect.json` lists |
| `conf` vs `conf_f32`, latin / cjk / rtl / world | 4.2e-6 / 2.3e-5 / 2.4e-6 / 7.6e-6 |
| `conf` vs the QDQ graph, latin | max 0.192, mean 4.3e-2 |
| det `prob`, `lines` (256x256) / `narrow` (96x160) | 7.2e-7 / 6.9e-7 |
| router `group` / `flip`, both cases | 2.3e-6 / 2.9e-6, 4.2e-7 / 4.8e-7 |

So the F32 implementation reproduces this oracle exactly on every recognizer
timestep, and the only `ids` differences anywhere are the ones the activation
quantization of §6 explains. The detector's `prob` tolerance has to be ~1e-5, not
1e-7: the crate's convolution order differs from ONNX Runtime's, and 7.2e-7 is the
measured floor.

## 8. Not verified here

* The end-to-end page test (`fixtures/pages/`): the page images and the LiteParse
  line lists next to them are the Lead's; the numbers are in `README.md` and
  `layout-plan.md`.
* `q8_0` numerics in Rust. The file loads (shape/type checks pass) but no inference
  was run from it; OCRia computes in F32 either way, so the expected difference is
  the Q8_0 weight rounding alone.
* GPU. OCRia has no device path: the graphs are depthwise-separable convolutions the
  repository's Vulkan kernels do not cover, and the upstream release is a CPU
  pipeline. `ComputePolicy::Vulkan` is refused by `mod.rs` rather than downgraded.

## 8. End-to-end page parity (Rust vs the pinned LiteParse engine)

Recorded by the Lead on 2026-11-26 with the same command the test documents:

```bash
cargo build --locked --profile release-fast --bin rust-model-inference
for p in receipt two_column sideways; do
  target/release-fast/rust-model-inference --model ocria-f32.gguf \
    --image tools/oracle/ocria/fixtures/pages/$p.png --out /tmp/$p.json --threads 4 --compute cpu
done
OCRIA_GGUF=ocria-f32.gguf OCRIA_PAGES=tools/oracle/ocria/fixtures/pages \
  cargo test --locked --profile release-fast --lib models::ocria::tests -- --ignored --nocapture
```

Host: Apple M3 Max (16 cores), macOS 27.2, `rustc 1.99.1`. The oracle JSON next to each
page was produced by the pinned LiteParse engine (`44ceb6ab`) through a small example added
to a temporary copy of that checkout; it calls `OcriaEngine::run` directly, so its boxes are
image pixels and not PDF points.

| page | what it exercises | lines | text | boxes | worst line confidence | page mean confidence |
| --- | --- | ---: | --- | --- | ---: | --- |
| `receipt.png` | single column, faint tokens | 19 | exact | ≤1 px | 6.20e-2 | 0.986419 vs 0.988068 |
| `two_column.png` | column-gutter split | 25 | exact | ≤1 px | 2.64e-2 | 0.987348 vs 0.985923 |
| `sideways.png` | sideways-page retry (two det passes) | 17 | exact | ≤1 px | 6.20e-2 | 0.985115 vs 0.987098 |

Text and line counts are exact on all three; the confidence difference is the `rec_latin`
QDQ effect from §7 (one character on a near tie moves a short line's mean by a few
hundredths, while the page mean stays within 2.0e-3).

**Q8_0 export, same pages** (`page_read_q8_0_keeps_the_text`): text and line count are
identical on all three pages; worst line confidence delta 6.96e-2, page means 0.986791 /
0.986888 / 0.985523. The quantized file's per-graph drift against the F32 file is measured
by `q8_0_weights_stay_within_the_measured_quantization_bounds`, which asserts the drift the
committed fixtures measure: detector `prob` max abs 1.85e-2 (gate 3e-2), router logits
6.08e-2 (gate 1e-1), recognizer `ids` 16/1064 timesteps = 1.5 % (gate 5 %) and a per-timestep
`conf` delta of 2.08e-1 (gate 3e-1, a sanity bound: a timestep's softmax maximum moves a lot
when two classes are nearly tied, while `ids` is what reaches the text). The F32 fixture gates
above are not applied to the Q8_0 file.

**Performance, same host** (Linux x86_64, 128 cores, no GPU; `release-fast`; wall clock
from a cold process to the finished JSON, so model load is included on both sides):

| page | Rust `--threads 4` | Rust `--threads 16` | LiteParse + ONNX Runtime |
| --- | ---: | ---: | ---: |
| `receipt.png` (19 lines) | 5.67 s | 2.00 s | 0.17 s |
| `two_column.png` (25 lines) | 9.74 s | 4.12 s | 0.15 s |
| `sideways.png` (17 lines, two detector passes) | 7.28 s | 2.78 s | 0.20 s |

On the macOS host (Apple M3 Max, `--threads 4`) the same three pages take 3.40 / 4.84 /
3.07 s, `--threads 1` 10.36 s and `--threads 8` 1.48 s for `receipt.png`; Q8_0 is 2.65 s.
The convolutions scale ~7x over eight threads.

The 12-27x gap to ONNX Runtime is attributed, not unexplained: `ops::nn::conv3d_cpu`
reduces a 1x1 pointwise convolution to one `dot_f32` of length 1 per input channel (and a
depthwise 3x3 to nine dots of length 3), so per-element call overhead dominates, while ONNX
Runtime runs NCHWc-packed specialized pointwise/depthwise kernels. Fixing it means adding
those paths to the shared `src/ops/nn.rs`, which every convolution caller in the repository
depends on; that is a separate change with its own caller-by-caller regression matrix, so it
is recorded here as the follow-up rather than folded into this adaptation.

## 9. Layer 2: LayoutNet

Weights: `tools/converter/ocria/convert_ocria_layout.py` packs `tablenet.onnx` (110
initializers, 4 268 063 f32) into its own GGUF.

| file | size | tensors | F32 | Q8_0 |
| --- | ---: | ---: | ---: | ---: |
| `ocria-layout-f32.gguf` | 17 098 496 B | 110 | 110 | 0 |
| `ocria-layout-q8_0.gguf` | 4 934 976 B | 110 | 75 | 35 |

The `f32` file is bit-exact against the ONNX initializers **and** against the release's own
`tablenet.bin` (35 matrices transposed, 4 conv kernels and biases under the f32 BN fold
`sc = g / sqrt(var + 1e-5)`, the rest direct); the `q8_0` blocks are byte-identical to
`quantize_q8_0` of the same arrays. Both are reproducible (re-running gives the same SHA256).

Numerical checks against the pinned reference:

| check | scope | result |
| --- | --- | --- |
| unit features `[N,68]` | 60 random pages + a degenerate page, all five feature groups | max abs **0.0**, 0 exact mismatches |
| pair features `[N,N,2]` | same | max abs **0.0** |
| `detect_rules` | 96 raster × scale combinations | bit-identical `h`/`v` vectors |
| `process_raster` / `ImgInput.canvas` | 4 pages | bytes identical / 0.0 |
| decode: `avg_link`, `reading_order`, `decode_tables`, `decode_grid`, classes, blocks, the three keep/drop predicates | 213 cases (200 random + 13 synthetic/degenerate) | **2585 checks, 0 failures** |
| network against ONNX Runtime, per stage | 3 cases (`page`, `clip` with `valid=(0.78,1.0)`, `blank`), gate 1e-4 | `unit_mlp` 9.5e-7, `embed` 9.5e-6, `enc_layer0` 9.5e-6, `enc_layer5` 1.1e-5, `hidden_final` 1.5e-5, `node` 6.7e-6, `pair` 7.6e-6, `cls` 8.1e-6; per-unit and per-pair argmax 8/8 and 64/64 |
| end to end on `fixtures/layout/table.png` | one ruled 3-column table | 1 table, **4x3 grid, `head_rows [0]`, 10 cells**, matching the reference's Markdown table |

The end-to-end run is the test `layout_finds_the_ruled_table`; the decode and feature
comparisons were run with throwaway harnesses that compile the pinned reference code beside
ours (not checked in — the committed evidence is the fixture below).

`layout_fixtures/` (2.27 MB) holds two pages' graph inputs, the three outputs and eleven
intermediates recorded from the pinned graph, so a network change can be localised to a stage
(`embed` → `enc_layer0` → `enc_layer5` → `hidden_final` → logits, with `grid`, `unit_samples`
and `unit_mlp` on the image side). The raster is committed as 8-bit ink plus the exact
`ink/255` decode, which keeps the set inside its budget; `--with-intermediates` regenerates
the three large tensors.

### The one bug the fixtures caught

The first port matched the image branch (`unit_mlp` 1.2e-6) and the encoder input (`embed`
7.6e-6) but was 2-3 logits off on `node`/`pair`/`cls`. Staged comparison put the first
divergence at the first cross-attention: `cross_attention` took the **token key** projection
from the packed `q|k|v` output's *query* block (`row[lo..hi]` instead of
`row[D + lo..D + hi]`), so every unit attended with `Q·K_q` while mixing the correct `V`. The
encoder's self-attention was unaffected because it indexes all three blocks explicitly, which
is why layers 0-4 matched and only the cross-attention and everything downstream did not.
A hand-built copy of that block, matched against the graph's own intermediate blobs at
<=2e-5 while the production function was 2.27 off on the same input, localised it to the
function; the final logits then matched at 6.7e-6.

One residual difference, deliberate: `pos2d` is computed from one frequency table, which
matches the graph's *runtime* table (unit positions) and differs from the graph's *baked*
token table by 1 ulp on 50 of 64 entries (max 1.5e-5) -- inside the gate by an order of
magnitude, and pinning both spellings would mean carrying two tables for a 1.5e-5 effect.
