# LayoutNet (layer 2) — scouting plan

Scouting report for the second delivery layer of the OCRia-nano-1 adaptation: the
table-structure / layout model (`tablenet.onnx` + `tablenet.bin`). No production code is
written here; the goal is that an engineer can pick this up and execute without repeating
the investigation, and that the open decisions are visible before anyone starts.

Reference runtime, read-only oracle: LiteParse at commit `44ceb6ab160c64cc8b43aab4ed85251719495629`
(mirrored locally at `/tmp/liteparse`). Every reference line number below is that mirror:
`table_model/mod.rs` 1026 lines, `units.rs` 874, `features.rs` 977, `net.rs` 1175,
`decode.rs` 1296 — 5348 lines total.

Notation used throughout: **[measured]** = a command was run and the output is quoted;
**[derived]** = arithmetic from measured numbers or from reference code, with the derivation
shown; **[reference]** = a fact read from the oracle source.

---

## 0. What was measured, and with what

All ONNX/weight measurements were taken on the remote box while
`/workspace/models/OCRia-nano-1/` still existed. **The box was re-provisioned at 01:39
(2026-11-26) and `/workspace/models` and `/workspace/ocria` are now gone** — see §7. The
numbers below are from before that; the byte sizes I read (`ls -la`) match the size fields
of the SHA-256 pins in `tools/oracle/ocria/sources.json`, so the artefacts measured are the
pinned ones.

```
# graph IO, metadata, op counts
./.remote/ssh.sh '/workspace/ocria/.venv/bin/python -c "import onnx, collections; ..."'
# initializers with shapes/dtypes + total elements
./.remote/ssh.sh '/workspace/ocria/.venv/bin/python -c "from onnx import numpy_helper; ..."'
# every node with attributes
./.remote/ssh.sh '/workspace/ocria/.venv/bin/python /workspace/ocria/onn2.py /workspace/models/OCRia-nano-1/tablenet.onnx'
# .bin header (magic, JSON header, tensor table, element count)
./.remote/ssh.sh '/workspace/ocria/.venv/bin/python -c "import json,struct; ..."'
# weight parity: BN-fold + transpose-aware sha256 bijection
./.remote/ssh.sh '/workspace/ocria/.venv/bin/python -c "import json,struct,numpy as np,onnx,hashlib; ..."'
```

Pinned artefacts (`tools/oracle/ocria/sources.json`):

| file | bytes | sha256 |
|---|---|---|
| `tablenet.bin` | 17 084 482 | `27b5f3ff2d6842d397a21954026b1e6411e8b35085b0b32c616e67e33852d1b4` |
| `tablenet.onnx` | 17 182 635 | `3b913866067d6e43f73a4e4c9c1eace955b7959447043cc39ea1d30e4c552a95` |

---

## 1. What LayoutNet is and what it needs

### 1.1 The model

One model, three heads, one forward pass over the page's **text units** — not over pixels
of text, and not over PDF objects. Configuration comes from the ONNX metadata key
`liteparse.tablenet` and, identically, from the `.bin` header **[measured, both]**:

```json
{"cfg":{"d":256,"layers":6,"heads":4,"pair_d":48,"stack_feats":true,"rule_feats":true,
        "layout":true,"n_cls":13,"img":{"c":64,"xattn":2}},"feat_dim":68}
```

- encoder input MLP `inp.0` 66→256 (ReLU), `inp.2` 256→256; the image branch's per-unit
  vector is added here (`net.rs:935-947`).
- 6 pre-norm transformer layers, 4 heads, ReLU feed-forward 256→512→256, **no final norm**
  (`net.rs:950-970`).
- node head `node` 256→2: `(in_table, header)` per unit.
- pair head `qk` 256→480 = 5 heads × (q|k) × 48: bilinear `q_h·k_hᵀ/√48`, symmetrised,
  plus a linear term over 15 pairwise geometry features (`net.rs:972-1024`,
  `pair_geometry` at `net.rs:1040-1084`). Five pair heads: `same_table`, `same_row`,
  `same_col`, `same_cell`, `same_block` — the fifth exists only because `layout: true`.
- class head `cls` 256→13 over `BLOCK_CLASSES` (`mod.rs:181-195`):
  Text, Heading, Title, PageHeader, PageFooter, List, Table, Caption, Footnote, Formula,
  Code, FigureText, KeyValue.
- optional image branch: patchify CNN (48ch @128², 48ch @128², 64ch @64², 64ch @64² from a
  1024² canvas, stride 16 total), 2×6 bilinear samples per unit box → 768→256 MLP added to
  the encoder input, plus 32×32 stride-32 tokens cross-attended by the top 2 layers
  (`net.rs:290-336`, `705-908`).

`N_IDS = 2` (`net.rs:14`): `feat_dim` is 68 but the encoder only ever sees **66** —
`forward` copies `feats[r*fd .. r*fd+fin]` with `fin = fd - 2` (`net.rs:929-934`). The last
two columns (block id, leaf id) are consumed **only** by `pair_geometry` as the
`same_blk` / `same_leaf` bits (`net.rs:1052-1062`). This matters for §2.2.

### 1.2 The unit list

A unit is a run of text the writer put down together, cut wherever a gap is wide enough to
separate two cells (`units.rs:1-7`). Fields (`units.rs:18-34`):

| field | meaning | used by |
|---|---|---|
| `bx` | box in the **reading frame**, points, top-left origin | features, decode (`DUnit.bx`), reading order |
| `pbox` | box on the **rendered page** (differs from `bx` only for sideways text) | rule features, image-branch boxes, block boxes |
| `text` | the run's text, whitespace-collapsed | features (chars, words, numeric…), decode (`is_caption`, `is_number`, folds) |
| `h` | median glyph height | feature 8 (`h/hm`), rule-feature yardstick, decode's `median_h` |
| `size` | median font size | **nothing** — carried but never read |
| `bold` | majority bold of the run's fonts | features 19, 20 |
| `q` | reading direction 0..3 | `main_rotation` (`mod.rs:525`) and `units.rs`'s own heuristics — dead on the OCR path |
| `minor` | runs against the page's main direction | `mask_minor`, block/table filtering |
| `words` | `(x0, x1, text)` per word | `join_touching` + `split_aligned` only — both text-layer |

Two producers:

1. **Text layer** — `units::pdf_units` (`units.rs:734-874`): PDFium characters → runs →
   `cut` on `CELL_GAP = 1.8` glyph heights / dot leaders / two adjacent numbers / a
   `gutter_gap` (`units.rs:210-349`), then `join_touching` (407-475), `split_aligned`
   (482-659), `flag_letter_stacks` (664-699).
2. **OCR** — `ocr_units` (`mod.rs:100-161`): one unit per **projected OCR item**, box
   `[x, y, x+w, y+h]` scaled from the projection's pixel space to page points by
   `wh / (sig.width, sig.height)`, `h = (bx[3]-bx[1]) / 1.4`, `size = 0`, `bold = None`,
   `q = 0`, `minor = false`, `words = []`.

**This repository can only take path 2** (§2).

### 1.3 The 68 per-unit features

`FEAT_DIM = 21 + 4*8 + 8 + 5 + 2 = 68` (`features.rs:38-39`), built by `unit_features`
(`features.rs:177-302`). Rows are `[N, 68]`, row-major.

| idx | feature | definition |
|---|---|---|
| 0 | `cx` | centre x / page width |
| 1 | `cy` | centre y / page height |
| 2, 3 | `w`, `h` | box width / page width, box height / page height |
| 4, 5, 6, 7 | `x0, x1, y0, y1` | edges / page width, height |
| 8 | `h/hm` | unit height / **median** unit height |
| 9 | `ln(len+1)/5` | character count |
| 10 | `ln(words+1)/3` | whitespace-split word count |
| 11, 12, 13 | digit / alpha / upper | fractions (upper over alphabetic) |
| 14 | `numeric` | whole-run regex `^[\s(\[$€£¥+\-–−±<>≤≥~]*[\d.,]+[%)\]*†‡§a-z]*$` (`features.rs:56-77`) |
| 15 | `endp` | ends in `.` `:` `;` `,` |
| 16 | `ln(n_row+1)/3` | units on the same line (centres within half the taller height) |
| 17, 18 | `gap_l`, `gap_r` | nearest same-line neighbour gap / median height, capped at 20, /20 |
| 19, 20 | `bold`, `bold_known` | `bold == Some(true)`, `bold.is_some()` |
| 21–52 | 32 Fourier | `q = 0..8`: `sin(2^q π cx)`, `cos(2^q π cx)`, `sin(2^q π cy)`, `cos(2^q π cy)` |
| 53–60 | 8 projection | present flag, `flow`, `ln(lc+1)/3`, `ln(rc+1)/3`, `ln(cc+1)/3`, `snap=='L'`, `snap=='R'`, `snap=='C'` — all **zero when the unit has no `Proj`** |
| 61–65 | 5 ruling-line | rule above / below / left / right (0/1), present flag |
| 66, 67 | block id, leaf id | `-1` when absent; **encoder does not read these** |

Note `t` — the text features read `proj.feat_text` when present, else `unit.text`
(`features.rs:224-228`). That is how the OCR faint-token read ("1", "€1") is masked to
`"x"*len` so LayoutNet does not fuse a table's rows around a recovered value
(`mod.rs:163-169`, `faint_mask`).

### 1.4 The pairwise features

`pair_geometry` builds 15 values per ordered pair (`net.rs:1040-1084`), consumed by
`pair_geo` `[5, 15]`:

| idx | value |
|---|---|
| 0 | `abs(cy_i − cy_j) / h_i`, capped 20, /20 |
| 1 | `abs(cx_i − cx_j) * 10`, capped 20, /20 |
| 2 | x-overlap `* 20`, clamped ±5, /5 |
| 3 | y-overlap `/ h_i`, clamped −5..2, /5 |
| 4, 5 | `(cx_i − cx_j) * 5`, `(cy_i − cy_j) * 5` |
| 6, 7 | `same_blk`, `same_leaf` (from the two id columns; 0 when either id < 0) |
| 8–12 | `stack_feats`: vertical gap, x-overlap fraction, left / right / centre offsets, all normalised by the smaller height |
| 13, 14 | `pairf[i][j]` — the two ruling-line pair features |

`pairf` itself is `[N, N, 2]` (`features.rs:122-174`): for each ordered pair, the number of
horizontal rules strictly between the two unit centres that span both boxes' x-ranges
(capped 3, /3), and the same for vertical rules over the y-ranges. Zeros when the page has
no rules (`mod.rs:621-624`).

### 1.5 The ruling-line (`Rules`) input

`Rules { h: Vec<[y, x0, x1]>, v: Vec<[x, y0, y1]> }` (`features.rs:32-36`), detected from a
**grayscale page render at 150 dpi** (`RULES_DPI = 150.0`, `features.rs:491`): ink is
`grey < 170`, an opening with a 1×L line kernel keeps runs ≥ `max(15, 3% of the page side)`,
8-connected components of runs on adjacent scanlines become rules, and anything thicker than
**2.5 pt** is discarded (`features.rs:304-488`, `detect_rules` / `rules_from_runs`).

`unit_rule_features` (`features.rs:85-119`) asks, per unit, whether a rule lies just above /
below (within −0.3…1.5 unit heights) / just left / right (−0.3…2.0 unit heights) and
overlaps the unit's span.

### 1.6 The CNN image branch

- `PageImage` (`features.rs:501-530`): grey page (0 = black), longest side `IMG_SIDE = 1024`
  (`net.rs:47`), area-resampled from the 150-dpi render (`resample_area`, OpenCV INTER_AREA
  when shrinking), plus `k` = pixels per PDF point in that image.
- `PageImage::input` (`features.rs:511-529`) puts ink on a `1024×1024` canvas, `(255 − px)/255`
  top-left aligned, maps each unit's `pbox` by `f = k / 1024` into canvas fractions, and
  returns `valid = (w/1024, h/1024)`.
- CNN: Conv 8×8/8 (48) → Conv 3×3/1 pad1 (48) → Conv 3×3/2 pad1 (64) → Conv 3×3/1 pad1 (64)
  → 64×64×64 map, BatchNorm folded into the convs, ReLU after each.
- `GridSample`: per unit, a 2×6 grid of fractional positions inside its box
  (`IMG_GH = 2`, `IMG_GW = 6`, `net.rs:300-301`), bilinear, `align_corners=False`, zeros
  outside (`bilinear`, `net.rs:318-336`), then `unit` MLP 768→256 (ReLU) →256, added to the
  encoder input.
- Tokens: 2×2 average pool of the 64×64 map → 32×32 stride-32 tokens, kept where
  `px ≤ valid.0 && py ≤ valid.1`, projected 64→256 by `tok`, plus `pos2d` of the token
  centre; the units get `pos2d` of their box centre. The top `xattn = 2` encoder layers
  cross-attend to those tokens (`cross_attention`, `net.rs:812-908`).
- `pos2d` (`net.rs:304-314`): `k = d/4 = 64`, `step = −ln(1000)/(k−1)`,
  `fr_i = exp(i·step)·π·64`, outputs `sin(x·fr)`, `cos(x·fr)`, `sin(y·fr)`, `cos(y·fr)`.

### 1.7 What `tablenet.onnx` actually takes and returns

**[measured]** `onnx.load` + graph IO:

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `feats` | `[1, N, 68]` | f32 | §1.3 |
| `mask` | `[1, N]` | **bool** | unit mask (all true for a page; `net.rs:372` passes `vec![true; n]`) |
| `pairf` | `[1, N, N, 2]` | f32 | §1.4 |
| `image` | `[1, 1, H, W]` | f32 | 1024² ink canvas (dims are dynamic in the graph) |
| `boxes` | `[1, N, 4]` | f32 | unit boxes as canvas fractions |
| `valid` | `[1, 2]` | f32 | `(w/1024, h/1024)` |
| **`node`** | `[1, N, 2]` | f32 | `(in_table, header)` |
| **`pair`** | `[1, 5, N, N]` | f32 | 5 pair heads (the file declares the first two dims as dynamic symbols; `net.rs:394` reads `n_pair` from `shape[1]`) |
| **`cls`** | `[1, N, 13]` | f32 | block classes |

opset 17, `ir_version` 8, producer `pytorch 2.14.0`, 1092 nodes, 110 initializers,
**4 268 063 f32 parameters** **[measured]**.

The Rust reference feeds exactly these six tensors and reads exactly these three
(`net.rs:370-396`), so the ONNX signature *is* the module boundary — a GGUF implementation
must reproduce this contract, not the `.bin` loader's.

---

## 2. What of that is reachable from this repository

### 2.1 The entry point is a page image, so exactly one reference path is open

This repository has no PDFium and no PDF text layer. Layer 1 takes a page image and produces
`Page { width, height, lines: Vec<Line{ text, bbox, confidence, polygon, faint }> }`
(`src/models/ocria/mod.rs`), so in the reference's terms every page here is an **OCR page**:

- `page_input` (`mod.rs:74-95`) would set `ocr = true` (text-layer characters < 20) and
  return **no units**; the units are built later by `ocr_units`.
- `run_page_framed` (`mod.rs:583-596`) then takes the `input.ocr` branch and, if
  `sig.is_none()`, returns an **empty** `PageModel` (`mod.rs:585-587`). The reference
  refuses to run the table model on an OCR page without projection signals.
- `main_rotation` returns `None` for OCR pages (`mod.rs:519-529`), so the whole rotation
  apparatus — `to_frame`/`from_frame`, `rotate_rules`, `rotate_image`, the box-mapping
  back at `mod.rs:483-513` — is dead code for us (~120 lines not to port).
- `minor` is `false` on every OCR unit (`mod.rs:146`), so `mask_minor` (`mod.rs:13-28`) and
  the `minor` filters in `run_page_framed` are no-ops.
- `units::pdf_units` (874 lines) is **never called on this path**; only the `Unit` struct
  survives. Confirmed by grep: `units::` in `mod.rs` appears at the struct definition, the
  `ocr_units` construction site, and the rotation branch only.

So the reachable reference is: `ocr_units` → `FUnit` → `unit_features` + `pair_rule_features`
→ `TableNet::forward` → `decode_tables` → `decode_grid` → `Grid::{is_toc,is_prose,is_chart}`,
`to_span_rows`/`to_html`. Plus `block_groups`/`split_*` if the block (layout) output is
wanted.

### 2.2 `ProjSignals` — the one real gap, and it is quantifiable

`ocr_units` (`mod.rs:100-161`) and `join_signals` (`mod.rs:358-442`) both need
`crate::projection::ProjSignals` (`projection.rs:1212-1231`):

```rust
pub struct ProjSignals {
    pub width: f32, pub height: f32,
    pub items: Vec<ProjSignalItem>,        // text, bx [x,y,w,h], flow, lc, rc, cc, snap, block, faint
    pub leaves: Vec<([f32; 4], Vec<u16>)>, // XY-cut leaf membership
}
```

`items` are not the raw OCR lines: they are the output of LiteParse's **grid projection**
pass, which merges/splits items by inferred column structure and records, per item, the
counts of left / right / centre anchors it shares with the page's other items, the snap
kind, whether its block rendered as flowing prose, which block it belongs to, and (via
XY-cut) which leaf of the page's recursive cut tree it sits in.

Cost: `projection.rs` is **7047 lines** (5761 before `#[cfg(test)]`), of which the parts that
produce these fields are `form_lines` (491-1126), `segment_blocks` / `extract_block_anchors`
/ `delta_min_filter` / `intercept_filter` / `try_align_floating` (1426-1692),
`is_flowing_text_block` / `detect_and_render_flowing_lines` (1726-2010), `project_to_grid`
(2010-3140), and the XY-cut family `xy_*` (3577-5231). That is roughly **3 500–4 500 lines**
of porting, more than the rest of layer 2 combined. It is text-item-only — `projection.rs`
has **zero** references to `pdfium` **[measured: `grep -c pdfium projection.rs` → 0]** — so
it is *portable*, just large. Nothing like it exists in this repository
(`grep -rn "projection\|ProjSignal" src/` finds only unrelated "linear projection" comments).

What is actually lost if it is skipped (`Proj::None` / `proj = None`):

- 8 of the **66** encoder features (idx 53–60) are zero, including the `present` flag;
- 2 of the 15 pair-geometry features (`same_blk`, `same_leaf`) are zero;
- block id / leaf id are `-1`, which is also what makes `same_blk`/`same_leaf` zero.

Everything else — all 21 base features, all 32 Fourier terms, all 5 rule features, all other
pair features — is derivable from the OCR lines plus the render. **Whether that is
acceptable is an accuracy question nobody has measured yet** (§5, checkpoint C0b). The
reference itself never runs this configuration: it returns an empty page instead.

### 2.3 Rules: reachable, with one unit caveat

`detect_rules(gray, w, h, scale)` needs only a grayscale render and pixels-per-point
(`features.rs:308-311`, `584-596`). Layer 1 already has the page pixels (`img.rs::Img`), so
this is a straight port of `InkBits`/`runs`/`rules_from_runs` (313-488) plus
`resample_area` (533-580).

Units: the caller's image is in *pixels*, not points, and its dpi is unknown. Check every
constant for unit dependence **[derived, by inspection]**:

- `unit_rule_features` and `pair_rule_features` are yardsticked by unit heights and counts
  → scale-free;
- run length uses 3 % of the page side → scale-free;
- **`thick_max_pt = 2.5 * scale` is the only absolute length**. With `scale = 150/72 =
  2.083` px/pt a rule is dropped above 5.2 px. If we pass `scale = 1.0` (1 px = 1 pt) the cap
  becomes 2.5 px, so 3–5 px rules that the model saw as real rules would be dropped. Pick
  the scale from the page: `scale = px_per_point = max(w,h) / assumed_page_long_side_pt`,
  and record the assumption in the GGUF contract.

Also unit-free, and worth stating because it removes a worry: `h = (bx[3]-bx[1]) / 1.4` in
`ocr_units` (`mod.rs:142`). The 1.4 cancels everywhere it could matter — feature 8 is
`h[i] / median(h)` (`features.rs:186, 254`), and every decode threshold is
`k * median_h` or `k * box_height` (`decode.rs:62-66, 392, 405, 640`). So the OCR path's
padded line box is not a problem, as long as the *same* box feeds features and decode.

### 2.4 The image branch: reachable, and scale-free

`PageImage::input` (`features.rs:511-529`) maps `pbox` by `f = k / IMG_SIDE` with
`k = (150/72) · s` and `s = IMG_SIDE / max(render_w, render_h)` — so
`f = (150/72) / max_px = 1 / max_pt` and therefore **`f · pbox_pt = pbox_pt / max_pt`**: the
canvas coordinate is simply the fraction of the page's long side, independent of the unit
system **[derived]**. So an image-only caller can pass `wh = (W, H)` in pixels and
`k = IMG_SIDE / max(W, H)`; no dpi is needed. (Rules still need §2.3's scale.)

The render must be resampled to longest side 1024 with area averaging — **not** a naive
nearest/bilinear resize: `resample_area` is a fractional-coverage box filter that the
reference pins against a bit-identical reference implementation
(`features.rs:683-977`, test `faster_raster_is_byte_identical`).

### 2.5 What genuinely does not exist here

| reference need | status in this repo |
|---|---|
| PDF text layer (`units::pdf_units`, `size`, `bold`, `q`, `words`) | absent; OCR path sets them to `0.0`/`None`/`0`/`[]` |
| `ProjSignals` (8 features + 2 pair features + block/leaf ids) | absent, ~3 500–4 500 lines to port (§2.2) |
| page render at a known dpi | layer 1 has page pixels; no dpi, handled in §2.3/§2.4 |
| a 2-D `GridSample` operator | only model-local, in `qwen_drive` (§3.4) |
| `sin`/`cos` as shared ops | only `rope::mrope::sin_cos` (rope-specific) |
| markdown table writer | `Grid::to_html` (46 lines) is the smallest useful output; the reference's `markdown_layout/tables.rs` is 8 665 lines and out of scope |

### 2.6 Minimal input contract for layer 2

The shape a first implementation should commit to — this is what §4's M1 builds:

```rust
// Everything layer 2 may read. No PDFium, no ProjSignals.
pub struct PageInput {
    pub width: usize, pub height: usize,      // page image pixels; used as "points"
    pub gray: Vec<u8>,                        // page render, row-major, 0 = black
    pub lines: Vec<Line>,                     // layer 1's output, page-space quads
    pub scale: f32,                           // px per point, for the 2.5 pt rule cap only
}
// unit := { bx = pbox = line bbox, text = line.text, h = (y1-y0)/1.4,
//           size = 0.0, bold = None, q = 0, minor = false, words = [] }
// proj := None                     (see §2.2 — the open decision)
```

---

## 3. Weight export

### 3.1 `tablenet.onnx` and `tablenet.bin` carry the same weights — verified

**[measured]** A transpose- and BN-fold-aware SHA-256 bijection over the two files:
**all 110 ONNX initializers are reproduced exactly** (bit-identical f32) by the `.bin`'s 126
tensors:

```
onnx initializers 110 matched 110 unmatched []
bin tensors used 110 of 134
```

The 24 unused `.bin` tensors (126 real tensors, 102 of which matched; the other 8 of the
"134" above are the synthetic BN-folded kernels) are exactly the four raw conv kernels plus
their BatchNorm statistics, which the ONNX export has already folded away:

```
img.cnn.{0,3.0,4.0,5.0}.weight                       (folded into onnx::Conv_*)
img.cnn.{1,3.1,4.1,5.1}.{weight,bias,running_mean,running_var,num_batches_tracked}
```

The fold itself was checked numerically, not just structurally:

```
img.cnn.0.weight weight maxabsdiff 0.0 bias maxabsdiff 0.0
img.cnn.3.0.weight weight maxabsdiff 0.0 bias maxabsdiff 0.0
img.cnn.4.0.weight weight maxabsdiff 0.0 bias maxabsdiff 0.0
img.cnn.5.0.weight weight maxabsdiff 0.0 bias maxabsdiff 0.0
```

(65 bias / LayerNorm / token / unit vectors matched directly; 33 `MatMul` weights matched
under the `[in,out]` ↔ `[out,in]` transpose; the two `img.attn.*.in_proj_weight` matched
directly.)

Element accounting **[measured]**, and it closes exactly:

| | tensors | f32 elements |
|---|---|---|
| `tablenet.onnx` initializers | 110 | 4 268 063 |
| `tablenet.bin` tensors | 126 | 4 268 739 |
| difference | +16 (4 BN × 4 arrays) +4 `num_batches_tracked` −4 conv biases | +676 = 900 − 224 |

`.bin` layout: `b"TNET"`, u32 LE header length (9 518), JSON header
`{cfg, feat_dim, tensors:[{name, shape, offset}]}`, then raw LE f32 at `offset * 4`;
`17 084 482 = 8 + 9 518 + 4 268 739 × 4` **[measured]**.

Parameter budget by block **[derived from the initializer shapes]**:

| block | params | share |
|---|---|---|
| encoder (6 layers incl. norms) | 3 162 624 | 74.1 % |
| image branch (CNN + unit MLP + tok + 2 cross-attn + norms) | 895 200 | 21.0 % |
| input MLP + node + qk + cls + pair_geo | 210 239 | 4.9 % |
| **total** | **4 268 063** | |

So layer 2 is a **17 MB, 4.27 M-parameter** model — 3 % of the 46 MB OCR stack, but 21 % of
its parameters sit in a branch that only exists to look at pixels.

### 3.2 Per-tensor inventory

Names and shapes below are **as stored in `tablenet.bin`** — torch orientation, `[out, in]`
for matrices, which is what the Rust loader expects (`Linear { n_out: ws[0], n_in: ws[1] }`,
`net.rs:170-179`). The ONNX file stores every `MatMul` weight **transposed** (`[in, out]`)
and its conv kernels under generated names (`onnx::Conv_1427`…). All f32.

```
inp.0.weight [256,66]        inp.2.weight [256,256]       node.weight [2,256]
qk.weight [480,256]          cls.weight [13,256]          pair_geo.weight [5,15]
enc.layers.{0..5}.self_attn.in_proj_weight [768,256]       ...self_attn.out_proj.weight [256,256]
enc.layers.{0..5}.linear1.weight [512,256]                 ...linear2.weight [256,512]
enc.layers.{0..5}.norm{1,2}.{weight,bias} [256]
img.attn.{0,1}.in_proj_weight [768,256]                    ...out_proj.weight [256,256]
img.norm.{0,1}.{weight,bias} [256]                         img.tok.weight [256,64]
img.unit.0.weight [256,768]                                img.unit.2.weight [256,256]
img.cnn.0.weight [48,1,8,8]   img.cnn.1.{weight,bias,running_mean,running_var} [48] +1
img.cnn.3.0.weight [48,48,3,3] img.cnn.3.1.{...} [48] +1   (3×3/1 pad1)
img.cnn.4.0.weight [64,48,3,3] img.cnn.4.1.{...} [64] +1   (3×3/2 pad1)
img.cnn.5.0.weight [64,64,3,3] img.cnn.5.1.{...} [64] +1   (3×3/1 pad1)
```

`qk` is `[480, 256] = 2·P·pair_d × d` with `P = 5` heads (q and k concatenated) — five, not
four, because `layout: true` (`net.rs:972-977`). `img.unit.0.weight`'s input width 768 is the
2×6×64 sampled feature vector (§1.6), which is the cross-check that pins the `GridSample`
output shape: `64 ch × 2 × 6 = 768`.

### 3.3 Can it go to GGUF by the same converter pattern? Yes.

`tools/converter/utils/gguf.py` is weight-format-agnostic: `GgufWriter.add_tensor(name,
ggml_type, gguf_dims(shape), raw_bytes)` with `gguf_dims` = torch dims reversed
(`gguf.py:576-586`), and `GGML_F32` needs no quantisation blocks. Rank is unbounded, so the
4-D conv kernels go in unchanged. The `pivo` converter is the pattern to copy
(`tools/converter/pivo/convert_pivo.py`: read `src/models/<m>/contract.json`, verify source
hashes, walk a name map, write `f32` and `q8_0` variants, `FORMATS = {'f32': ...}`).

Two differences from every existing converter:

1. **No converter in this repository reads ONNX today** — `grep -rln "import onnx" tools/`
   finds only `tools/oracle/pivo/`. The layer-2 converter would be the first. This is
   cheap, not risky: `onnx.load` + `numpy_helper.to_array` hands back numpy arrays, so no
   torch is needed, and the names are already the export names.
2. **The source of truth should be `tablenet.onnx`, not `tablenet.bin`** — the ONNX file
   already has BatchNorm folded, so the Rust side needs no `conv_bn` folding step
   (`net.rs:181-206` is then unnecessary). The `.bin` remains the parity oracle for the
   fold, which §3.1 has already discharged.

`q8_0` is a policy question: the encoder matmuls have row lengths 256 / 512 (both multiples
of the 32-value Q8_0 block), the conv kernels 48·9 = 432 and 64·9 = 576 (multiples of 32),
and `[66,256]`/`[768,256]` are fine too — so a Q8_0 export is mechanically possible. Given
the model is 17 MB, F32-only is the sane default for v1; the reference runs it in f32 in
both of its own backends.

### 3.4 ONNX ops the graph needs, versus repository operators

**[measured]** 39 distinct op types over 1092 nodes:

```
Constant 393  Unsqueeze 138  Add 77   MatMul 58  Concat 47  Reshape 47  Transpose 44
Div 38  Gather 35  Cast 30  Shape 26  Mul 22  Slice 20  Sub 15  LayerNormalization 14
Relu 12  Clip 11  Softmax 8  Split 6  Abs 5  Conv 4  Equal 4  Expand 4  Sin 4  Cos 4
Min 4  Range 3  Max 3  ConstantOfShape 2  Where 2  Greater 2  GreaterOrEqual 2  And 2
Not 1  GridSample 1  AveragePool 1  Or 1  Reciprocal 1  Exp 1
```

The layer-2 implementation is **not** an ONNX interpreter: like every other model here it
gets a hand-written forward. So the useful split is "does a kernel exist", not "is the op
listed".

| ONNX op | repository operator today | verdict |
|---|---|---|
| `Conv` ×4 | `ops::nn::conv2d` (`src/ops/nn.rs:1628`, stride/pad/dilation/groups; used by `hy_simulmt/audio.rs:179`, `wevisone_vl/decoder.rs:289`) | **have** |
| `MatMul` ×58 | `ops::nn::Linear` (`src/ops/nn.rs:24,175`) + the f32 gemm path | **have** |
| `LayerNormalization` ×14 | `ops::norm::layer_norm` (`src/ops/norm.rs:35`), `nn::layer_norm_rows` (`:731`) | **have** |
| `Softmax` ×8 | `ops::softmax::softmax_inplace` (`src/ops/softmax.rs:64`) | **have** |
| `Relu` ×12 | `ops::activation::relu` | **have** |
| `Exp` ×1 | `ops::math::exp::exp_inplace` (`src/ops/math/exp.rs:16`) | **have** |
| `Add/Mul/Sub/Div` | elementwise loops, per model | **have** |
| **`GridSample` ×1** | **no shared op.** Model-local `grid_sample_bilinear_aligned` (`src/models/qwen_drive/perception/ops.rs:134`), 4-D `[B,C,H,W]` in / `[B,Ho,Wo,2]` grid, `align_corners=false` variant exists and its padding/clamp semantics match — but it rounds through bf16 (`round_bf16`), so it is a *reference to copy*, not a callable op | **new kernel (~90 lines)** |
| **`AveragePool` 2×2/2** | no shared op; trivial. Model-local precedents: `gemma4/vision/mod.rs:743` (`average_pool_3x3`), `qwen3/omni.rs:936` (`average_pool_pairs`) | **new (~15 lines)** |
| **`Sin`/`Cos` ×4 each** | only `rope::mrope::sin_cos` (`pub(crate)`, rope-specific). Needs a `pos2d` helper anyway | **new (~10 lines)** |
| `Range` ×3, `ConstantOfShape` ×2, `Where` ×2, `Equal` ×4, `Greater`/`GreaterOrEqual`/`And`/`Or`/`Not`, `Split` ×6, `Slice` ×20, `Gather` ×35, `Shape` ×26, `Reshape` ×47, `Transpose` ×44, `Unsqueeze` ×138, `Concat` ×47, `Cast` ×30, `Clip` ×11, `Min`/`Max`/`Abs`/`Reciprocal`, `Constant` ×393 | no runtime analogue **by design** — these are graph plumbing the reference expresses as loops | **free, but each is a semantic decision** |

The plumbing is where the silent bugs live. Concretely, from the graph and the reference:

- `Range` + `Cast` + `Div` + `Exp` + `Reciprocal` build the `pos2d` frequency table: three
  `Range`s (token y index, token x index, frequency index `0..d/4`), one `Exp`, and 4
  `Sin`/`Cos` — i.e. exactly `net.rs:304-314`, including the `k.max(2)-1` denominator.
- `Split` ×6 = q|k|v inside the 6 encoder layers; the 2 cross-attentions instead `Slice`
  `in_proj_weight`/`in_proj_bias` into q/k/v (`/Slice_8..11`), which is why
  `img.attn.*.in_proj_weight` is stored untransposed.
- `Equal`+`Where`+`ConstantOfShape`+`Expand` ×2 = the box-padding mask: grid rows for units
  that do not exist are replaced with a fill value so `GridSample` samples nothing there.
- `Greater`+`Or`+`Cast` on `valid` = the token validity mask; the Rust reference instead
  *drops* invalid tokens and pads with `false` (`net.rs:765-790`). Same maths, different
  tensor shapes — so parity fixtures for the image branch must compare **logits**, not
  intermediate tensors.
- `Clip` ×11 = coordinate clamping into `[-1, 1]` for `GridSample` and the token grid.
- `Concat_22` = `[13 geometry features | pairf]` → `[15]` into `pair_geo`
  (`/Concat_21` gathers exactly 13 unsqueezed scalars), matching `pair_geo` `[5,15]`.

---

## 4. Effort and risk

Sizes are for the **image-only, no-`ProjSignals`** scope (§2.6) unless stated. Line counts
are production Rust excluding tests; the reference column is the code being ported, so the
ratio is a sanity check rather than a target.

| # | milestone | lines | reference | must be verified |
|---|---|---|---|---|
| **M0** | converter `tools/converter/ocria_layout/` (or a second component in `tools/converter/ocria/`) + contract + hash pins | 200–280 py | — | every GGUF tensor bit-equal to the ONNX initializer; 110 tensors / 4 268 063 f32; loads through `format::ggufrs` |
| **M1** | page input + units: `Line` → `Unit`, page raster → gray/scale, `rules` + `pairf`, `PageImage` + `resample_area` | 300–400 | `ocr_units` 62, raster+resample ~130, `detect_rules` ~185 | unit text/box/h against a reference dump; `Rules.h/v` exact; canvas bytes exact; `valid` exact |
| **M2** | features: `unit_features` (21 + 32 + 8 + 5 + 2), `pair_rule_features`, `unit_rule_features` | 250–350 | 126 + 53 + 35 (+ helpers) | `feats [N,68]` max abs diff ≤ 1e-6; `pairf` exact; `proj=None` path documented |
| **M3** | net: input MLP, 6 pre-norm layers, 5 pair heads + `pair_geometry`, node + cls heads, image branch (CNN, `GridSample`, unit MLP, tokens, `pos2d`, 2 cross-attentions) | 700–900 | `net.rs` 1175 (minus `.bin` loader 111, ONNX backend ~100, gemm shim 81 → ~880 of maths) | logits `node`/`pair`/`cls` within tolerance; per-stage fixture for `pos2d`, the 2×6 grid, and the token mask |
| **M4** | decode: `avg_link`, `bridges`, `span_link`, `decode_tables` (merge/absorb/`head_band`), `decode_grid` (forced links, owner slots, `squeeze_cols`, `fold_wrapped_rows`, `fold_header_wraps`) | 1 100–1 300 | `decode.rs` 1145 non-test | table unit-sets **exact**; grid `(r0,c0,r1,c1,text)`, `n_rows`, `n_cols`, `head_rows` **exact** |
| **M5** | output: `Grid::to_html` + `to_span_rows`, `is_toc`/`is_prose`/`is_chart`, block grouping (`block_groups`, `split_at_gaps`, `split_paragraphs`) | 300–400 | `decode.rs:1100-1143`, `mod.rs:784-979`, `mod.rs:222-346` | byte-identical HTML/GFM on fixtures; the keep/drop decision per table identical |
| **M6** | CLI (`--layout`) + HTTP route + contract/docs | 150–250 | `src/models/ocria/cli.rs` pattern | end-to-end on a real page; `--layout` off by default |
| | **total (M0–M6)** | **≈ 2 800–3 600 Rust + 200–280 py**, plus 1 000–1 500 of fixtures/tests | | |
| **M7** | *optional* `ProjSignals`: projection pass (lines → blocks → anchors → snap/flow → grid → XY-cut leaves) | **3 500–4 500** | `projection.rs` 5 761 non-test | 8 proj features + 2 pair features + block/leaf ids exact per unit |

Ordering: M0 and M1 are independent; M2 needs M1; M3 needs M2; M4 needs M3 (it is pure
logic on logits, so it can be written against recorded logits before M3 is finished — the
fixture-driven order M4-early is recommended). M5 needs M4. M7 is a separate decision.

### High-risk parts, worst first

**R1 — the `ProjSignals` dependency is unquantified, and the reference treats it as
mandatory** (§2.2). 8 of 66 encoder inputs and 2 of 15 pair inputs are zero, and the
reference *refuses* to produce tables at all without them on an OCR page
(`mod.rs:585-587`). Two very different projects hide here: 3 500–4 500 lines of projection
port (M7), or an accuracy regression nobody has measured. **This is the decision that should
be made first**, and it is measurable before any porting: record the same pages with and
without `proj` and diff the tables (§5, C0b). If the ablation shows the tables survive,
M1–M6 is the whole job; if it does not, M7 is not optional and layer 2 roughly triples.

**R2 — the decode's span logic** (`decode.rs` 613-847 and 884-1080). `span_link` splits units
into "bridging" and "core" by a 30 %-negative-neighbour rule (`bridges`, 151-173), links the
core by average linkage, then re-attaches bridges to *every* group they average positive
with. `decode_grid` then places cells in `(rmin, cmin)` order into an `owner` slot grid,
and when a span collides with an already-claimed slot it either collapses to a single free
slot or **appends the text to the occupant** (786-806). On top of that, `squeeze_cols` folds
near-empty columns into a neighbour *by text concatenation*, and two wrap-folding passes
mutate `r0/c0/r1/c1` in place while renumbering rows. Every step is order-sensitive
(argmax ties, `sort_unstable`, insertion order) and the reference says so explicitly
(`decode.rs:5-8`: "so parity fixtures compare byte for byte"). One different tie-break
changes the grid. **Mitigation:** port it as a mechanical transcription, keep the same data
layout (flat `Vec<f32>` logits indexed `[(k*n+i)*n+j]`), and drive it from recorded logits
before the net is finished.

**R3 — the image branch's `GridSample`** (`net.rs:318-336`, `features.rs:511-529`). Three
conventions have to be right simultaneously: (a) `x = fx·full_w − 0.5` with
`align_corners=false`, (b) zeros outside the map, (c) the *full* map size passed separately
from the crop — the reference crops the canvas to `crop(valid ∪ boxes) + 96 px margin` for
speed (`net.rs:717-730`) and then samples with `full = IMG_SIDE/16 = 64`. The crop is a pure
optimisation: `bilinear` takes the full size so that the same fractions land on the same
pixels whether or not the map was cropped. **Recommendation: skip the crop.** The full
1024²→64² map is ~50 M MACs in the first conv, far below the encoder, and it removes the
crop's `full_h/full_w` bookkeeping entirely (`full == h == w`). Also: the grid is 2×6 samples
per box at cell *centres* (`(gh+0.5)/2`, `(gw+0.5)/6`), and the token path is a 2×2 average
pool followed by a validity filter. A half-pixel or `align_corners` error shifts every image
feature and degrades tables silently rather than failing loudly. **Mitigation:** fixture the
64×64 map and the `[N, 768]` sampled vector, not just the logits.

**R4 — relative position encoding (`pos2d`)** (`net.rs:304-314`). `k = d/4 = 64`,
`step = −ln(1000)/(k−1)`, `fr_i = exp(i·step)·π·64`, four interleaved sin/cos blocks of 64.
Three easy ways to get it wrong: `k` vs `k−1` in the denominator, `i` vs `i+1` in the
exponent, and π·64 vs a bare π. It appears **twice** (token centres and unit centres) and
is added *after* the linear projections, so an error is invisible in shapes and only shows
as a small accuracy loss. **Mitigation:** fixture the `pos2d` matrix itself
(`[1024, 256]` for the token grid) and compare to 1e-6.

**R5 — the layout-class heuristics** (`mod.rs:784-923`). `is_toc` (≤ 3 columns, ≥ 3 rows
ending in a page number, arabic ones non-decreasing), `is_prose` (median ≥ 4 words, < 30 %
numeric, ≥ 3 adjacent same-column pairs with ≥ 40 % "unterminated → lowercase" flow) and
`is_chart` (sparse, majority numeric, a column of ≥ 3 evenly-stepped values) decide whether
a decoded grid is **reported at all**. They are threshold soup tuned against the reference's
own OCR text; this repo's OCR segments lines differently (different `Line` boxes, different
`text`), so the same thresholds may fire differently. Faithful transcription is necessary
but not sufficient — these need their own fixture set of *negative* pages (a TOC, a chart,
two-column prose) or the layer will emit tables where the reference emits text.

**R6 — cost and the `N²` ceiling.** `MAX_UNITS = 1500` (`mod.rs:32`); above it the reference
skips the model entirely. Memory: `pairf` is `N²·2·4` B (18 MB at N=1500), `pair` is
`5·N²·4` B (45 MB), and the Rust path materialises a `n×n` f32 `scores` per attention head
on top. Time: 6 layers × O(N²·d) plus 5 pair heads × O(N²·pair_d) plus `pair_geometry` at
O(N²·15). This is fine for text-layer pages but an OCR page can legitimately produce more
lines than a text layer has units, so the ceiling needs a policy (skip, or truncate) and a
measured number.

---

## 5. Parity strategy

The reference already has a stage-by-stage parity design — `bench/tables/dump_parity.py`,
referenced at `mod.rs:1-4` — so mirror its checkpoints rather than inventing new ones.

### Fixtures to record from the reference

Per page, one directory:

| fixture | content | produced by |
|---|---|---|
| `units.json` | `wh`, and per unit `{bx, pbox, text, h, size, bold, q, minor}`, plus per-unit `proj{flow, lc, rc, cc, snap, block, leaf, feat_text}` (null when absent) | dump inside `run_page_framed` before `unit_features` |
| `rules.json` | `{h: [[y,x0,x1]…], v: [[x,y0,y1]…]}` | `features::page_raster` |
| `img.bin` + `img.json` | the `PageImage` grey buffer, `w/h/k`, and `ImgInput{canvas, boxes, valid}` | `PageImage::input` |
| `feats.bin` | `[N,68]` f32 LE, shape in the JSON sidecar | `unit_features` |
| `pairf.bin` | `[N,N,2]` f32 LE | `pair_rule_features` |
| `logits.bin` | `node [N,2]`, `pair [5,N,N]`, `cls [N,13]` f32 LE | `TableNet::forward` |
| `tables.json` | `PageModel`: per table `bbox` + `grid{cells[{r0,c0,r1,c1,text,bx}], head_rows, n_rows, n_cols}`; `classes`; `blocks` | `run_page` |
| `golden.html` | `Grid::to_html` per table | `decode.rs:1100` |
| `pages/` | the source page images (rasterised, so the fixture set is self-contained) | — |

`units.json` + `feats.bin` are the important ones: they let M2 be validated without M3, and
M3 without M4.

### Checkpoints and acceptance

| checkpoint | gate | tolerance |
|---|---|---|
| **C0a** inputs | `rules`, `img.bin` byte-identical to the reference for every fixture page | exact |
| **C0b** *ablation* | run the reference with `proj = None` on the same pages; record `tables.json` twice | report: how many tables/cells change; this is R1's number |
| **C1** features | `feats.bin`, `pairf.bin` | max abs ≤ 1e-6 (both sides f32, same formula) |
| **C2** net | `logits.bin` | `node`/`cls` max abs ≤ 1e-4; `pair` ≤ 1e-3 relative (5 heads × N², gemm reassociation) — and see the sensitivity gate below |
| **C2b** net, staged | `pos2d` matrix; the 64×64 CNN map; the `[N,768]` sampled vector; the token mask | `pos2d` ≤ 1e-6; map/vector ≤ 1e-5 |
| **C3** tables | set of unit-index groups per page, after merges/absorption/`head_band` | **exact** (integer) |
| **C4** grids | per table `n_rows`, `n_cols`, `head_rows`, and the cell list `(r0,c0,r1,c1,text)` | **exact** |
| **C5** output | `to_html` / GFM, whitespace-normalised | **exact** |
| **C6** report set | which grids survive `is_toc`/`is_prose`/`is_chart` | **exact**, and the negative-page set must include a TOC, a chart, a two-column prose page, a 1-column list |
| **C7** blocks/classes | `classes[i].probs`, `blocks[]` | probs ≤ 1e-4; blocks exact |

**The tolerance question must be answered by measurement, not chosen.** C2→C4 are
thresholds on f32 logits (0.0 for `avg_link`, `table_thresh`, `node > -8/-1.5/-4`,
`hdr > 0.5`), so what matters is not the logit error but the *decision margin*. Record the
reference's margin distribution per page (min `|logit|` over pairs that were compared
against 0, and the distance of each `node` logit from its threshold) and then perturb
recorded logits by ±ε to find the ε at which no decode changes. That ε is C2's real budget;
if it is below what a different gemm order produces, the fix is to decode from
`logits.bin`-equivalent arithmetic, not to loosen C4.

### What the fixture set should cover

Small but adversarial: ≥ 20 pages. Suggested composition — a ruled financial table (rules +
spans), a borderless table, a table with a colspan header, a table with a rowspan, a table
continued across a page break (stresses the stacked-merge at `decode.rs:390-423`), a
two-column-prose page (must be rejected), a chart (must be rejected), a TOC (must be
rejected), a single-column list (must be rejected), a page with a rotated table, a dense
page above `MAX_UNITS`, a page whose OCR text contains a lone `1` between cells (the
`faint_mask` path, `mod.rs:163-169`), and a nearly-empty page.

Acceptance for the whole layer: **C3/C4/C5/C6 exact on all fixtures**, plus a perf gate
(the reference notes ONNX Runtime is 1.4–2× faster than its own Rust kernels on x86 Linux,
`net.rs:1005-1007`; budget ≤ 2× ORT single-thread on the same box).

---

## 6. Verdict

**Layer 2 is a self-contained follow-up of size ≈ 2 800–3 600 lines of Rust + 200–280 lines
of Python converter, plus 1 000–1 500 lines of fixtures — and it needs no capability this
repository lacks, provided `ProjSignals` is allowed to be absent.**

Evidence:

- The ONNX signature is a clean module boundary (`feats`, `mask`, `pairf`, `image`, `boxes`,
  `valid` → `node`, `pair`, `cls`, §1.7), and the reference itself uses exactly that boundary
  (`net.rs:370-396`), so the GGUF implementation has a fixed, small API to hit.
- The weights are the same in both shipped files, verified bit-exact (§3.1), so one GGUF
  serves both and the `.bin` needs no separate loader.
- Of the 39 ONNX op types, the ones needing a kernel that does not exist are `GridSample`
  (~90 lines, with a model-local precedent at `src/models/qwen_drive/perception/ops.rs:134`),
  `AveragePool` 2×2 (~15) and `sin`/`cos` (~10). Everything else is `ops::nn::conv2d`,
  `ops::nn::Linear`, `ops::norm::layer_norm`, `ops::softmax`, or graph plumbing that a
  hand-written forward expresses as index arithmetic.
- The largest single risk is **not** in the network. It is `ProjSignals`: the reference
  refuses to run LayoutNet on an OCR page without it (`mod.rs:585-587`), and porting the
  projection pass is 3 500–4 500 lines — comparable to, or larger than, M0–M6 combined. That
  is a scoping decision, and §5's C0b ablation answers it before any code is written.

Two caveats that would change the verdict:

1. If C0b shows tables degrade materially with `proj = None`, layer 2 needs the projection
   pass (M7) and becomes **≈ 6 300–8 100 lines** — still self-contained, but a much larger
   follow-up, and it should be split into its own PR (projection is also useful to the OCR
   layer for column ordering).
2. Layer 2 has **no usable input in this repository until layer 1 lands**: it consumes
   `Vec<Line>` with page-space quads and the page pixels. Layer 1's CLI shell already exists
   (`src/models/ocria/cli.rs`), but until `Model::run` returns real lines layer 2 can only be
   developed against recorded fixtures.

Not in scope for layer 2, and should stay out: the reference's `markdown_layout/` (10 079
lines across `tables.rs` + `blocks.rs`, with colspan-flattening and header-band heuristics),
`resegment.rs`, `table_fix.rs`, and PDF text-layer units.

---

## 7. Environment note (blocking for anyone starting now)

While this report was being written the remote box was re-provisioned: `/workspace/models`
and `/workspace/ocria` (including the pinned LiteParse checkout, the `.venv` with
`onnx`/`onnxruntime`, and `onn2.py`/`onn3.py`) **no longer exist**; `/workspace` now holds
only `bonsai25/`, `nariq/`, `lost+found/`, and an empty `bootstrap.sh`/`bootstrap.log`
(`ssh … 'ls -la /workspace'`, uptime 149 days, load ~21). The mirror at `/tmp/liteparse` is
unaffected.

To restart layer 2 work on the box:

1. `tools/oracle/ocria/download.py <dir>` restores the six OCR graphs from the ModelScope
   pins — **but it only iterates `sources.json`'s `files` list, and `tablenet.onnx` /
   `tablenet.bin` are under `unused_files`** (reason: "table-structure model, a separate
   component the OCR pipeline does not run"). Layer 2 must move those two entries into
   `files` (hashes are already recorded there) or the downloader will silently skip them.
2. Recreate the venv with `onnx` + `onnxruntime` and restore `onn2.py`/`onn3.py`
   (`onn2.py` dumps every node with attributes; `onn3.py` dumps the attribute-bearing ops).

---

## Appendix A — reference line index (for the port)

| topic | location |
|---|---|
| `PageInput`, deferred render | `mod.rs:36-95` |
| `ocr_units` (the reachable unit producer) | `mod.rs:100-161` |
| `faint_mask` | `mod.rs:167-169` |
| `BLOCK_CLASSES` (13) | `mod.rs:181-195` |
| `join_signals` (text-layer only) | `mod.rs:358-442` |
| `run_page` / `run_page_framed` | `mod.rs:449-515`, `576-779` |
| `main_rotation` (dead for OCR pages) | `mod.rs:519-529` |
| `is_toc` / `is_prose` / `is_chart` | `mod.rs:784-811`, `817-866`, `870-923` |
| `to_span_rows` | `mod.rs:927-979` |
| block grouping | `mod.rs:222-346` |
| `Proj`, `FUnit`, `Rules`, `FEAT_DIM` | `features.rs:6-39` |
| `unit_rule_features` / `pair_rule_features` | `features.rs:85-119`, `122-174` |
| `unit_features` | `features.rs:177-302` |
| `detect_rules` / `rules_from_runs` | `features.rs:304-488` |
| `PageImage` / `input` / `resample_area` | `features.rs:501-580` |
| `process_raster` / `RawRaster` | `features.rs:584-681` |
| `NetConfig` / `ImgCfg` / `IMG_SIDE` | `net.rs:16-56` |
| `pos2d` / `bilinear` | `net.rs:304-336` |
| ONNX input contract | `net.rs:370-377` |
| `from_bytes` (.bin loader — not needed) | `net.rs:449-559` |
| `image_features` | `net.rs:705-808` |
| `cross_attention` | `net.rs:812-908` |
| `forward` | `net.rs:913-1037` |
| `pair_geometry` | `net.rs:1040-1084` |
| `avg_link` / `bridges` / `span_link` | `decode.rs:75-217` |
| `head_band` / `decode_tables` | `decode.rs:298-496` |
| `squeeze_cols` / `fold_col` / `decode_grid` | `decode.rs:519-847` |
| wrap folds | `decode.rs:884-1080` |
| `to_html` | `decode.rs:1100-1143` |
| `ProjSignals` (the gap) | `projection.rs:1212-1231`, pipeline `projection.rs:1-5761` |

## Appendix B — decisions needed before M0

1. **`ProjSignals`: port (M7) or skip?** Needs the C0b ablation first. Owner: whoever owns
   layer 2's accuracy budget.
2. **Where does the layout GGUF live?** A second GGUF file (`ocria-layout.gguf`) with its own
   `general.architecture` (e.g. `ocria-layout`) is cleanest and matches how every other model
   here is dispatched; a second component inside `ocria.gguf` avoids a second `--model` flag
   but couples layer 2's release to layer 1's converter.
3. **Output format.** `to_html` (46 lines, already written in the reference) is the cheapest
   faithful target. GFM pipe tables lose rowspan, so a `to_span_rows` → GFM path needs a
   documented flattening rule. The reference's `markdown_layout/tables.rs` is out of scope.
4. **Rules scale.** What page long side (in points) is assumed for `scale = px/pt`? Affects
   only the 2.5 pt thickness cap (§2.3). Suggest recording it in the GGUF contract.
5. **`N` ceiling.** Skip above 1500 units as the reference does, or truncate? Needs a
   measured page-count distribution from layer 1 first.
