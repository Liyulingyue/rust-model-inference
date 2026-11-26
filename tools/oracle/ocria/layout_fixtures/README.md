# LayoutNet ONNX Runtime fixtures

Recorded from the pinned `tablenet.onnx` with ONNX Runtime, for the Rust port in
`src/models/ocria/layout/`. `manifest.json` is the index: for every blob it lists
the shape, dtype, byte count, sha256 and where the value comes from, plus the
graph's SHA256 and its `liteparse.tablenet` config.

**Size budget: 2.4 MB for all three cases, committed.** The page raster is stored as
8-bit ink (1 MB per case, the one blob worth its size) and the three big
intermediates are opt-in, because everything here is regenerable from the pinned
graph in seconds. If this set ever grows past ~3 MB, something that should be
opt-in has been committed.

Three cases, one synthetic ruled table (N = 8 units, 2 columns x 4 rows, 1024²
canvas):

| case | `valid` | raster | what it pins |
| --- | --- | --- | --- |
| `page` | `(1.0, 1.0)` | the ruled table | the ordinary full-page pass; the token mask is all zero |
| `clip` | `(0.78, 1.0)` | the table, cut at 0.78 | 224 masked tokens: the graph's `-1e9` additive mask vs the Rust port's "drop the invalid tokens" |
| `blank` | `(1.0, 1.0)` | **all white** | the image branch on a raster with no ink: the conv biases alone still give a non-zero `unit_samples` / `unit_mlp`, so "the raster decode or `GridSample` is wrong" separates from "the conv stack or the unit MLP is wrong". Same `feats`/`pairf`/`boxes`/`valid` as `page` (byte-identical blobs), so it is also a second, independent sample of the encoder path |

The `blank` case stores **no raster file**: its `image` entry has `"file": null` and
`"fill": 0`, and the canvas is `zeros[1,1,1024,1024]` (the entry still carries
`decoded_sha256`). Its `grid`, `pos2d_*` and `token_mask` are functions of
`boxes`/`valid` alone, so they are byte-identical to the `page` case's and are not
duplicated; `unit_samples` *is* recorded, because it is the blob that tells the two
failure modes apart.

Each case directory holds the six graph inputs, the three outputs and (for
`page`/`clip`) 11 intermediates, 6 for `blank` — `embed`, `enc_layer0`, `enc_layer5`, `hidden_final`, the `[N,768]`
GridSample vector, the image branch's `[N,256]` vector, the GridSample grid, both
`pos2d` spellings and the token mask — so a mismatch can be localised to a stage
instead of just "the logits are off". `canvas.png` is a grey preview of the
raster, for humans only.

Blobs are raw little-endian: `.f32` is row-major float32, `.u8` is one byte per
element (bool, or the 8-bit raster). The raster is the one blob that needs a
conversion — the manifest's `decode` field spells it out:

```python
ink = np.frombuffer((dir / "page/image.u8").read_bytes(), dtype=np.uint8).reshape(1024, 1024)
image = (ink.astype(np.float32) / 255.0).reshape(1, 1, 1024, 1024)  # == decoded_sha256
node = np.frombuffer((dir / "page/node.f32").read_bytes(), dtype="<f4").reshape(1, 8, 2)
# a blob with `fill` set has no file: the array is that value everywhere
image_blank = np.zeros((1, 1, 1024, 1024), dtype=np.float32)
```

Regenerate (deterministic — re-running reproduces byte-identical blobs):

```sh
# the committed set (2.4 MB)
python -m tools.oracle.ocria.layout_fixtures.make_fixtures /path/to/OCRia-nano-1 \
  tools/oracle/ocria/layout_fixtures --force
# plus cnn_map, avgpool and tokens (6.8 MB) in a scratch directory
python -m tools.oracle.ocria.layout_fixtures.make_fixtures /path/to/OCRia-nano-1 \
  /tmp/layout-fixtures-full --with-intermediates
```

The op-by-op reading of the graph these fixtures pin is
`../layout_graph_notes.md`.
