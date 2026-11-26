# LayoutNet (`tablenet.onnx`) — the graph, op by op

Reference for the layer-2 implementation in `src/models/ocria/layout/`. Everything
below was read off the pinned graph itself (`onnx.load` + a full node/attribute dump,
1092 nodes), not from the release notes or from `bench/tables/tablenet.py`.

* Model: `llamaindex/OCRia-nano-1`, ModelScope `master` @ `ca290e82458d32aa8b605dbf3248d7aaf895ea7e`;
  per-file revision + size + SHA256 in `sources.json` (`Role: layer-2`).
* `tablenet.onnx` 17 182 635 B, sha256 `3b913866067d6e43f73a4e4c9c1eace955b7959447043cc39ea1d30e4c552a95`.
* `tablenet.bin` 17 084 482 B, sha256 `27b5f3ff2d6842d397a21954026b1e6411e8b35085b0b32c616e67e33852d1b4`
  — LiteParse's own export; **same weights**, used here only as the naming and
  orientation oracle (§2).
* Oracle runtime: LiteParse `44ceb6ab160c64cc8b43aab4ed85251719495629`,
  `crates/liteparse/src/ocr/table_model/{net.rs,features.rs,decode.rs,mod.rs}`.
* Measured with `onnx` 1.23.2, `onnxruntime` 1.32.0, `numpy` 2.5.5.

Two abbreviations used throughout: **[measured]** = a command was run on the box and
the number is quoted; **[graph]** = read off a node's attributes or a constant's
value. `x` is always the current hidden state `[1,N,256]`.

## 1. The module contract

**[measured]** `onnx.load` graph IO — this *is* the Rust module boundary
(`net.rs:370-396` feeds exactly these six tensors and reads exactly these three):

| tensor | shape | dtype | meaning |
| --- | --- | --- | --- |
| `feats` | `[1,N,68]` | f32 | 68 per-unit features; only `0..66` reach the encoder (`/Slice`, §3.1) |
| `mask` | `[1,N]` | **bool** | `true` = real unit. Only used as a self-attention key mask (§3.3) |
| `pairf` | `[1,N,N,2]` | f32 | two ruling-line pair features; only read by `pair_geo` (§3.7) |
| `image` | `[1,1,H,W]` | f32 | ink canvas, `(255-px)/255`, `0` = white. `H`/`W` dynamic; production `1024²` |
| `boxes` | `[1,N,4]` | f32 | `(x0,y0,x1,y1)` as fractions of the canvas |
| `valid` | `[1,2]` | f32 | `(w/1024, h/1024)` of the real page inside the canvas |
| **`node`** | `[1,N,2]` | f32 | `(in_table, header)` |
| **`pair`** | `[1,5,N,N]` | f32 | five pair heads, §4 |
| **`cls`** | `[1,N,13]` | f32 | block classes, §4 |

`pair`'s and `cls`'s leading dims are *symbolic* in the file (`Addpair_dim_0`,
`Addnode_dim_0`) because the exporter derived them from arithmetic; `net.rs:394` reads
`n_pair` from `shape[1]` for exactly that reason. Opset 17, IR 8, producer
`pytorch 2.14.0`, **1092 nodes, 110 initializers, 4 268 063 f32 parameters** [measured].

Metadata, one key, verbatim:

```
liteparse.tablenet = {"cfg":{"d":256,"layers":6,"heads":4,"pair_d":48,"stack_feats":true,
                      "rule_feats":true,"layout":true,"n_cls":13,"img":{"c":64,"xattn":2}},
                      "feat_dim":68}
```

`src/models/ocria/layout/contract.json` carries that string's parse as `config`
(`config["cfg"]` is the `NetConfig`, `config["feat_dim"]` the feature width) and the
110-tensor name→shape map as `tensors`. The GGUF carries the same JSON in
`ocria-layout.config`.

Op histogram [measured, 39 types]:

```
Constant 393  Unsqueeze 138  Add 77   MatMul 58  Concat 47  Reshape 47  Transpose 44
Div 38  Gather 35  Cast 30  Shape 26  Mul 22  Slice 20  Sub 15  LayerNormalization 14
Relu 12  Clip 11  Softmax 8  Split 6  Abs 5  Conv 4  Equal 4  Expand 4  Sin 4  Cos 4
Min 4  Range 3  Max 3  ConstantOfShape 2  Where 2  Greater 2  GreaterOrEqual 2  And 2
Not 1  GridSample 1  AveragePool 1  Or 1  Reciprocal 1  Exp 1
```

Everything except `Conv`, `MatMul`, `LayerNormalization`, `Softmax`, `Relu`, `Split`,
`GridSample`, `AveragePool`, `Sin`/`Cos`, `Exp` is shape plumbing: the reference
expresses it as index arithmetic, so only those ten need a kernel (§6 of
`layout-plan.md` already sized them).

## 2. Weights: names, orientation, BatchNorm

### 2.1 Why the names need a rule at all

The export names initializers **inconsistently**: biases and norms keep their PyTorch
parameter name (`m.enc.layers.0.self_attn.in_proj_bias`, `m.img.norm.0.weight`), while
every weight matrix and conv kernel is anonymous (`onnx::MatMul_1443`,
`onnx::Conv_1427`). The *node* names are lossy in the other direction: the exporter
flattened the `m.`, `enc.` and `img.` scopes out, so the encoder's layer 0 is
`/layers.0/MatMul` and the image branch's first conv is `/cnn/cnn.0/Conv`.

`tablenet.bin` has a semantic name for all 126 of its tensors, so it is the authority.
The GGUF therefore uses **the `.bin` names verbatim** (torch parameter names, which is
also what `net.rs` reads: `t.linear("enc.layers.0.linear1")`, `t.conv_bn("img.cnn.0",
"img.cnn.1", …)`). That is the same target the layer-1 rule
(`<node-path>.<weight|bias>`, `/` → `.`) produces when the export preserves scopes; here
the scopes have to be recovered from the consumer node, and the `.bin` name is the
verification that they were recovered correctly.

The mapping is mechanical and checked, not hand-written:

* **67 tensors keep their own name** (strip the leading `m.`): every bias, every norm,
  `img.attn.*.in_proj_{weight,bias}`, `img.unit.*.bias`, `img.tok.bias`.
* **43 are anonymous** and are named from the node that consumes them:
  * the 35 `MatMul` weights from the `MatMul` node, `<module>.weight`;
  * the 4 conv kernels from the `Conv` node (input 1), `img.cnn.<n>.weight`;
  * the 4 conv biases from the same `Conv` node (input 2) — the **BN-folded** bias,
    `img.cnn.<n>.bias`, which the `.bin` does not carry at all.

Node → name, exhaustively (the only places a scope had to be re-attached are `enc.`
for `/layers.*` and `img.` for the image branch, both pinned by the sibling tensors
that still have their `m.` names):

| consuming node | GGUF name |
| --- | --- |
| `/inp/inp.{0,2}/MatMul` + `/Add` | `inp.{0,2}.weight` / `.bias` |
| `/layers.{l}/MatMul` + `/Add` | `enc.layers.{l}.self_attn.in_proj_weight` / `in_proj_bias` |
| `/layers.{l}/out_proj/MatMul` + `/Add` | `enc.layers.{l}.self_attn.out_proj.weight` / `.bias` |
| `/layers.{l}/linear{1,2}/MatMul` + `/Add` | `enc.layers.{l}.linear{1,2}.weight` / `.bias` |
| `/layers.{l}/norm{1,2}/LayerNormalization` | `enc.layers.{l}.norm{1,2}.weight` / `.bias` |
| `/cnn/cnn.0/Conv` | `img.cnn.0.weight` / `.bias` |
| `/cnn/cnn.{3,4,5}/cnn.{3,4,5}.0/Conv` | `img.cnn.{3,4,5}.0.weight` / `.bias` |
| `/unit/unit.{0,2}/MatMul` + `/Add` | `img.unit.{0,2}.weight` / `.bias` |
| `/tok/MatMul` + `/Add` | `img.tok.weight` / `.bias` |
| `/norm.{0,1}/LayerNormalization` | `img.norm.{0,1}.weight` / `.bias` |
| `/out_proj/MatMul` + `/Add` | `img.attn.0.out_proj.weight` / `.bias` |
| `/out_proj_1/MatMul` + `/Add` | `img.attn.1.out_proj.weight` / `.bias` |
| `/node/MatMul` + `/Add` | `node.weight` / `.bias` |
| `/qk/MatMul` + `/Add` | `qk.weight` / `.bias` |
| `/pair_geo/MatMul` + `/Add` | `pair_geo.weight` / `.bias` |
| `/cls/MatMul` + `/Add` | `cls.weight` / `.bias` |

`/out_proj` vs `/out_proj_1` is torch's export de-duplication: both cross-attention
modules are named `out_proj` inside their own `img.attn.{i}` scope, and the flattened
scope collides, so the second becomes `/out_proj_1`. The `.bin` distinguishes them as
`img.attn.0.out_proj` / `img.attn.1.out_proj`; the *order* is pinned by their biases,
which still carry the full names (`m.img.attn.0.out_proj.bias` feeds `/out_proj/Add`).

`img.attn.{0,1}.in_proj_{weight,bias}` are the one pair of weights with **no single
consuming node**: the graph slices them into q/k/v (`/Slice_8`…`/Slice_19`) and only
the slices reach a `MatMul`, so they keep their own (already semantic) names.

### 2.2 Orientation

The contract stores **torch orientation** — `[out, in]` for every matrix,
`[cout, cin, kh, kw]` for every conv, 1-D for every bias and norm. That is the `.bin`'s
shape and what `net.rs`'s `Linear { n_out: ws[0], n_in: ws[1] }` and
`ops::nn::conv2d(.., weight_shape=[cout,cin,kh,kw], ..)` want.

What the ONNX file stores:

| ONNX | count | GGUF |
| --- | --- | --- |
| `MatMul` weight `[in, out]` | 35 | transposed to `[out, in]` |
| `MatMul` weight `[out, in]` | 2 (`img.attn.{0,1}.in_proj_weight`, `[768,256]`) | **kept** — see below |
| `Conv` kernel `[cout,cin,kh,kw]` | 4 | unchanged |
| `Conv` bias `[C]` (already BN-folded) | 4 | unchanged |
| bias / norm `[C]` | 65 | unchanged |

(35 + 2 + 4 + 4 + 65 = 110; 4 of them are 4-D, 37 are 2-D, 69 are 1-D.)

The four `img.attn.*.in_proj_*` tensors (2 weights + 2 biases) are `[768,256]` =
`(q|k|v, d)` and the graph slices **rows** (`/Slice_8` = `0:256`, `/Slice_10` = `256:512`, `/Slice_12` =
`512:768`) and then transposes each slice (`/Transpose_9`, `perm=[1,0]`) before the
`MatMul`. So the file's orientation already matches the contract; transposing it would
be a silent, shape-preserving bug. This is the only matrix that must *not* be
transposed, and it is the reason the converter transposes per tensor rather than by op
type.

### 2.3 The BatchNorm fold is already in the file

`img.cnn.{0,3.0,4.0,5.0}` are followed by no `BatchNorm` node: the exporter folded it
into the kernel and gave the `Conv` a bias. So `img.cnn.<n>.weight` is the **folded**
kernel and `img.cnn.<n>.bias` the folded bias; `net.rs:181-206` (`Tensors::conv_bn`) has
no work to do on the Rust side.

**[measured]** the fold reproduces the `.bin`'s raw kernel bit-for-bit:

```
sc[o]      = bn.weight[o] / sqrt(bn.running_var[o] + 1e-5)        (f32)
kernel     = bin.img.cnn.<n>.0.weight * sc[:, None, None, None]    (f32)  max|Δ| = 0.0
bias       = bn.bias[o] - bn.running_mean[o] * sc[o]               (f32)  max|Δ| = 0.0
```

for all four convs, where `bn` = `img.cnn.{1,3.1,4.1,5.1}`. (Element accounting:
the `.bin`'s 126 tensors are the ONNX 110, minus these 4 folded biases, plus the 20
BatchNorm arrays `{weight,bias,running_mean,running_var,num_batches_tracked}` of the
four `bn` modules — 110 − 4 + 20 = 126, and 4 268 739 − 4·(48+48+64+64) + … closes to
4 268 063 as in `layout-plan.md` §3.1.)

**[measured]** the whole bijection: all **110** initializers map onto **110** distinct
`.bin` names and every array is bit-identical (the 4 folded conv tensors under the
`sc`/`bias` formulas above, the 35 transposed matrices under `.T`, the rest directly).
`check_ocria_layout.py` re-runs this check against the `.bin` as a second opinion on
top of the primary GGUF-vs-ONNX bit comparison.

## 3. The forward pass

### 3.1 Input slice: 66 of the 68 columns

`/Slice(feats, starts=[0], ends=[-2], axes=[2])` → `[1,N,66]`. The last two columns
(block id, leaf id) are never seen by the encoder; they are read only by `pair_geo`
(§3.7). This is `net.rs:929-934` (`fin = feat_dim - N_IDS`).

### 3.2 Input MLP, then the image vector is added

```
h = Relu(x66 @ W[66,256] + b) @ W2[256,256] + b2        # /inp/inp.0, /inp/inp.1/Relu, /inp/inp.2
x = h + unit_mlp                                          # /Add_2
```

`/Add_2` is the **encoder input** and the fixture's `embed.f32`. The image term comes
from §3.5; when the caller has no raster it is zero and this reduces to the MLP
(`net.rs:941-947`).

### 3.3 One encoder layer

Six identical layers, `l = 0..5`, pre-norm, no final norm. For layer `l` with input `x`:

```
h    = LayerNorm(x, norm1_l)                       # eps 1e-5, axis -1
qkv  = h @ W_in[256,768] + b_in                    # /layers.l/MatMul, /layers.l/Add
q,k,v = Split(qkv, axis=-1, [256,256,256])         # q = 0:256, k = 256:512, v = 512:768
for head hh in 0..4:                               # head_dim = 64
    s   = q_hh @ k_hh^T / 8.0                      # 8.0 = sqrt(64), a Constant
    s  += mask_term                                # [1,1,1,N], -1e9 where mask = false
    p   = Softmax(s, axis=-1)                      # over the key axis
    ctx_hh = p @ v_hh
ctx  = Concat(ctx_0..3)                            # [1,N,256]
x    = x + (ctx @ W_out[256,256] + b_out)          # /layers.l/Add_2
h2   = LayerNorm(x, norm2_l)
x    = x + (Relu(h2 @ W1[256,512] + b1) @ W2[512,256] + b2)   # /layers.l/Add_3
```

Shapes and the exact transposes (all `[graph]`): q is reshaped `[1,N,4,64]` then
`perm=[0,2,1,3]` → `[1,4,N,64]`; k `[1,N,4,64]` → `perm=[0,2,3,1]` → `[1,4,64,N]`;
v `[1,N,4,64]` → `perm=[0,2,1,3]` → `[1,4,N,64]`; the context comes back
`perm=[0,2,1,3]` → `[1,N,4,64]` → `[1,N,256]`. Head `hh` therefore owns columns
`hh*64..hh*64+64` of the 256-wide `q`, `k` and `v` — i.e. the packed `in_proj` is
**head-major inside each of q/k/v**, and q/k/v are contiguous 256-blocks.

`mask_term` = `-1e9 * (1 - mask)` broadcast over the last (key) axis, computed once
outside the layers (`/Not`, `/Cast`, `/Mul`) and reused by all six. It masks keys only;
padded query rows are computed but never read. With the all-true mask the Rust side
passes (`net.rs:372`) the term is exactly `0.0`, so a port that skips it is correct for
this repo's call sites — but the fixture pins the general form.

### 3.4 `pos2d` — two spellings of the same table

`d = 256`, `k = d/4 = 64`, `step = -ln(1000)/(k-1)`:

```
freq_i = exp(i * step) * pi * 64          i = 0..63
pos2d(x,y) = [ sin(x*freq_0..63), cos(x*freq_0..63), sin(y*freq_0..63), cos(y*freq_0..63) ]
```

i.e. four contiguous 64-blocks in the order **sin x, cos x, sin y, cos y** — not
interleaved. `step` is negative, so `freq` runs from `64*pi = 201.06194` (i = 0) down to
`0.001*64*pi = 0.20106` (i = 63) [measured: both ends].

The graph builds it **twice, differently**:

* **Units** (`/Range_2` → `/Reciprocal` → `/Mul_10` → `/Exp` → `/Mul_12` → `/Mul_13`):
  computed at run time in f32, with `step = f32(1/63) * f32(-6.9077554)`, and the
  frequency applied as `x * freq` then `Sin`/`Cos` (`/Sin_2`, `/Cos_2`, `/Sin_3`,
  `/Cos_3` → `/Concat_11`). `x = (box.x0+box.x1)/2`, `y = (box.y0+box.y1)/2`, the box
  centre as a canvas fraction.
* **Image tokens** (`/Constant_76`, `/Constant_78`): the same 64 values baked into the
  file as a `Constant` tensor and multiplied in (`/Mul_7`, `/Mul_8` → `/Sin`, `/Cos`,
  `/Sin_1`, `/Cos_1` → `/Concat_9`). The constant is a *float64* evaluation rounded
  once, so it does **not** agree bit-for-bit with the graph's own run-time f32 table:
  **[measured]** 50 of the 64 entries differ by 1 ulp, max |Δ| = 1.526e-05 (on the
  smallest entries, e.g. `0.20106192` baked vs `0.20106183` run time; the two ends
  agree exactly, `201.06194` and `0.25036177` vs `0.25036165`). A Rust port that
  computes the table once in f32 and uses it for both branches is therefore within
  ~1.5e-5 of the graph on the **token** branch while staying bit-exact on the unit
  branch; the fixture records the constant verbatim as `pos2d_freq.f32` and the
  run-time table as `pos2d_freq_runtime.f32` so the two can be pinned rather than
  argued about.

`k.max(2)` in `net.rs:306` never binds (`k = 64`), and `i` starts at 0, not 1.

### 3.5 The image branch

**CNN** (`/cnn/…`), all NCHW, all with folded bias + `Relu`, all `dilation=1`:

| # | node | kernel | stride | pad | out |
| --- | --- | --- | --- | --- | --- |
| 0 | `/cnn/cnn.0/Conv` | 8×8 | 8 | 0 | `[1,48,128,128]` |
| 1 | `/cnn/cnn.3/cnn.3.0/Conv` | 3×3 | 1 | 1 | `[1,48,128,128]` |
| 2 | `/cnn/cnn.4/cnn.4.0/Conv` | 3×3 | 2 | 1 | `[1,64,64,64]` |
| 3 | `/cnn/cnn.5/cnn.5.0/Conv` | 3×3 | 1 | 1 | `[1,64,64,64]` |

The last output is the fixture's `cnn_map.f32` (written only with
`--with-intermediates`, §7); the graph runs it on the **full** canvas (no crop), so
`64 = IMG_SIDE/16` always.

**Per-unit samples** (`/GridSample`, `mode=bilinear`, `padding_mode=zeros`,
`align_corners=0` [graph]):

* x positions per unit: `x0 + (x1-x0) * (gw+0.5)/6`, `gw = 0..5`
  (`/Constant_25 = [0.0833, 0.25, 0.4167, 0.5833, 0.75, 0.9167]`);
* y positions: `y0 + (y1-y0) * (gh+0.5)/2`, `gh = 0..1` (`/Constant_37 = [[0.25],[0.75]]`);
* grid value `= 2*f - 1` (`/Mul_5` by 2, `/Sub_2` by 1), last axis `(x, y)`, reshaped to
  `[1, 2N, 6, 2]` — the `(unit, gh)` pair is flattened into the `Ho` axis;
* `GridSample` → `[1, 64, 2N, 6]` → `/Reshape_3` `[1,64,N,12]` → `perm=[0,2,1,3]`
  `[1,N,64,12]` → `/Reshape_4` `[1,N,768]`.

So the MLP input index is `c*12 + gh*6 + gw` — channel-major, then the 2×6 sample grid
row-major. `align_corners=0` makes the sample position `f*64 - 0.5` in map pixels,
zeros outside — exactly `net.rs:319-336`. Then `/unit/unit.0` 768→256 ReLU,
`/unit/unit.2` 256→256, and `/Add_2` adds it to the encoder input (§3.2).

**Tokens** (`xattn = 2`):

* `/AveragePool` on the 64×64 map: 2×2, stride 2 → `[1,64,32,32]` (fixture `avgpool.f32`,
  `--with-intermediates`);
* `/Reshape_11` → `[1,64,1024]` → `perm=[0,2,1]` → `[1,1024,64]` → `/tok` 64→256 + bias;
* `pos2d` of each token centre `((tx+0.5)/32, (ty+0.5)/32)` (§3.4, constant table) added
  → `[1,1024,256]` (`/Add_5`, fixture `tokens.f32`, `--with-intermediates`). **All 1024
  tokens are kept**: the grid is `H/2 × W/2` of the CNN map, not filtered by `valid`;
* token mask: `px > valid[0] OR py > valid[1]` (`/Greater`, `/Greater_1`, `/Or`,
  `/Cast_9`) → `[1024]` f32, `1.0` = masked (the `Cast` target is f32, not bool), then
  `× -1e9` and added to the cross-attention scores. `net.rs:765-790` instead *drops* the invalid tokens and pads
  with `false`; identical maths, different tensor shapes — parity must be compared on
  `tokens`/logits, not on a "kept token" list.

### 3.6 Cross-attention (two of them, after layers 4 and 5)

```
q_in = LayerNorm(x_after_layer_l, img.norm.{i}) + pos2d(box centre)     # /Add_8, /Add_14
q    = q_in @ Wq^T + bq        # Wq = in_proj_weight[0:256]   (sliced, then perm=[1,0])
k    = tok @ Wk^T + bk         # Wk = in_proj_weight[256:512]
v    = tok @ Wv^T + bv         # Wv = in_proj_weight[512:768]
s    = q k^T / 8.0 + mask_term                                          # [1,4,N,1024]
x    = x + (Softmax(s) v) @ W_out^T + b_out                            # /Add_13, /Add_19
```

`i = 0` runs on the output of layer 4 (`/layers.4/Add_3`), `i = 1` on layer 5's
(`/layers.5/Add_3`); `xattn = 2` means "the last two layers". Both use the *same*
tokens `/Add_5`. The `pos2d` is added to the normalised query, **not** to the
projection — `net.rs:826-829` does the same. `/Add_19` is the final hidden state; the
three heads read it.

### 3.7 Heads

**node** — `/node`: `x @ W[256,2] + b` → `[1,N,2]` = `(in_table, header)`.

**qk → pair** — `/qk`: `x @ W[256,480] + b`, then `/Reshape_22` → `[1,N,2,5,48]`:
axis 2 is **q then k**, axis 3 the head, axis 4 `pair_d = 48`. Then

```
q_h = Gather(axis=2, 0)[:,:,h,:]   → perm [0,2,1,3] → [1,5,N,48]
k_h = Gather(axis=2, 1)[:,:,h,:]   → perm [0,2,3,1] → [1,5,48,N]
bilinear_h = (q_h k_h^T) / sqrt(48)                     # 6.928203 = sqrt(48), a Constant
pair_h     = (bilinear_h + bilinear_h^T) / 2            # /Transpose_18, /Add_20, /Div_10
```

**pair_geo** — `/Concat_21` builds **13** scalars per ordered pair, `/Concat_22` appends
`pairf[i][j][0..2]` → 15, then `pair_geo.weight [5,15]` (`[15,5]` in the file) gives
`[1,N,N,5]`, `perm=[0,3,1,2]` → `[1,5,N,N]`, added to `pair` (`/Add_23`). The 13, in
the graph's order (each `[graph]`, and each matching `net.rs:1040-1084`):

| idx | value |
| --- | --- |
| 0 | `min(|cy_i-cy_j| / h_i, 20) / 20` |
| 1 | `min(|cx_i-cx_j| * 10, 20) / 20` |
| 2 | `clamp((min(x1_i,x1_j) - max(x0_i,x0_j)) * 20, -5, 5) / 5` |
| 3 | `clamp((min(y1_i,y1_j) - max(y0_i,y0_j)) / h_i, -5, 2) / 5` |
| 4 | `(cx_i - cx_j) * 5` |
| 5 | `(cy_i - cy_j) * 5` |
| 6 | `same_blk` = `blk_i == blk_j && blk_i >= 0` |
| 7 | `same_leaf` = `leaf_i == leaf_j && leaf_i >= 0` |
| 8 | `clamp(max(y0_j-y1_i, y0_i-y1_j) / min(h_i,h_j), -3, 5) / 5` |
| 9 | `clamp((min(x1_i,x1_j) - max(x0_i,x0_j)) / min(w_i,w_j), -2, 1)` |
| 10 | `min(|x0_i-x0_j| / min(h_i,h_j), 10) / 10` |
| 11 | `min(|x1_i-x1_j| / min(h_i,h_j), 10) / 10` |
| 12 | `min(|(x0_i+x1_i) - (x0_j+x1_j)| / 2 / min(h_i,h_j), 10) / 10` |
| 13,14 | `pairf[i][j][0], pairf[i][j][1]` |

with `h_i = max(y1_i-y0_i, 1e-4)`, `w_i = max(x1_i-x0_i, 1e-4)`, all features read from
`feats` columns 0,1,4,5,6,7 and the two id columns 66,67. Note **`pair` is not
symmetric**: the symmetrisation happens before the geometry term is added, and the
geometry features are directional (`/h_i`, not `/h_j`).

**cls** — `/cls`: `x @ W[256,13] + b` → `[1,N,13]` over `BLOCK_CLASSES`
(`mod.rs:181-195`): Text, Heading, Title, PageHeader, PageFooter, List, Table, Caption,
Footnote, Formula, Code, FigureText, KeyValue.

## 4. Output layout

* `node [1,N,2]`: index 0 = `in_table` logit, 1 = `header` logit. Thresholds live in
  `decode.rs` (`node > -8 / -1.5 / -4`), not in the graph.
* `pair [1,5,N,N]`: index `k` is

  | k | head | used by |
  | --- | --- | --- |
  | 0 | `same_table` | `decode_tables` average linkage (`decode.rs:382`) |
  | 1 | `same_row` | `decode_grid` stacked-pair forcing (`decode.rs:645`) |
  | 2 | `same_col` | same |
  | 3 | `same_cell` | `decode_grid` cell grouping (`decode.rs:650`) |
  | 4 | `same_block` | `block_groups`; present only because `layout: true` |

  Element `pair[0,k,i,j]` is row-major `(k*N + i)*N + j` — the same layout
  `Logits::pair` (`net.rs:421-424`) indexes.
* `cls [1,N,13]`: softmax is *not* in the graph; `decode`/`mod.rs` apply it.

## 5. Where the graph and `net.rs` differ (port notes)

Same maths, five differences worth knowing before writing the Rust forward:

1. **No crop.** `net.rs:717-730` crops the canvas to `valid ∪ boxes + 96 px` and
   passes `full = IMG_SIDE/16` to `bilinear`; the graph runs the full canvas, so
   `full == h == w`. Dropping the crop (as `layout-plan.md` §R3 recommends) reproduces
   the graph exactly and removes the `full_h/full_w` bookkeeping.
2. **Tokens are never dropped.** The graph keeps all `H/2 × W/2` tokens and masks with
   `-1e9`; `net.rs` keeps only valid ones. Equal up to the softmax's max-shift, so
   compare logits, not token counts.
3. **`mask` exists.** `net.rs` never reads its `mask`; the graph adds `-1e9` on masked
   *keys* in all six self-attentions. All-true ⇒ zero, so ignoring it is correct for
   this repo's only call site.
4. **BatchNorm is pre-folded** (§2.3), and the conv `bias` is a tensor the `.bin`
   doesn't have.
5. **`pos2d` is spelled twice** (§3.4): run-time f32 for units, a baked constant for
   tokens, differing by ≤ 1.5e-5.

## 6. Constants

| constant | value | where |
| --- | --- | --- |
| `d`, `layers`, `heads`, `pair_d`, `n_cls` | 256, 6, 4, 48, 13 | metadata `cfg` |
| `feat_dim` / encoder input | 68 / 66 | metadata / `/Slice` |
| `IMG_SIDE` | 1024 | `net.rs:47`, not in the graph (dims dynamic) |
| CNN stride total | 16 → 64×64 map | §3.5 |
| `IMG_GH`, `IMG_GW` | 2, 6 | `/Constant_37`, `/Constant_25` |
| LayerNorm `eps` | `1e-5` (`9.9999997e-06`) | 14 `LayerNormalization` nodes |
| self-attn scale | `1/sqrt(64) = 1/8` | `/layers.l/Constant_8 = 8.0` |
| cross-attn scale | `1/8` | `/Div_8 = 8.0` |
| pair scale | `1/sqrt(48) = 1/6.9282031` | `/Constant_186` |
| attn mask value | `-1e9` | `/Constant_12`, `/Constant_95` |
| `pos2d` `k`, `step` | 64, `-ln(1000)/63 = -0.1096468` | §3.4 |
| `pair_geo` clips | 20/20, 5/5, 5/5, 3/5, 2/1, 10/10 | §3.7 |

## 7. Fixtures

`tools/oracle/ocria/layout_fixtures/` — ONNX Runtime, one deterministic synthetic
ruled table (N = 8 units, 2 columns × 4 rows), regenerated by
`python -m tools.oracle.ocria.layout_fixtures.make_fixtures <model-dir> <out-dir>`
(re-running it reproduces byte-identical blobs; the manifest carries a sha256 per
blob). Each case lives in its own subdirectory (`page/`, `clip/`, `blank/`),
`manifest.json` is the index, and `canvas.png` is a grey preview of the raster for
humans — not an input, and absent for `blank` (there is nothing to look at). The committed set is **2.4 MB for all three cases**: the raster is stored
as 8-bit ink (1 MB) and the three big intermediates are opt-in
(`--with-intermediates`, ~7 MB), because they are regenerable from the pinned graph.
The cases share one page and one unit list:

| case | `valid` | raster | purpose |
| --- | --- | --- | --- |
| `page` | `(1.0, 1.0)` | the ruled table, all boxes inside | the ordinary full-page pass |
| `clip` | `(0.78, 1.0)` | the table cut at the page edge, right column `x1` = 950 px = 0.928 | 224 masked tokens; boxes past `valid` |
| `blank` | `(1.0, 1.0)` | **all white**, no file (`fill: 0`) | the image branch with no ink: the conv biases still give a non-zero `unit_samples`/`unit_mlp`, so a raster/`GridSample` bug separates from a conv/MLP bug; same `feats`/`pairf`/`boxes`/`valid` as `page` (byte-identical), so it is a second encoder sample too |

Blobs (raw little-endian; every entry is also in `manifest.json` with shape, dtype,
file name and sha256, alongside the ONNX sha256 and the config). An entry with `fill`
set has no file — the array is that value everywhere — which is how `blank`'s raster
costs nothing:

| blob | shape | dtype | source |
| --- | --- | --- | --- |
| `feats` | `[1,8,68]` | f32 | input, real `unit_features` with `proj = None` |
| `mask` | `[1,8]` | bool | input, all `true` |
| `pairf` | `[1,8,8,2]` | f32 | input, real `pair_rule_features` (2 horizontal, 1 vertical rule) |
| `image` | `[1024,1024]` | u8 | input raster, 8-bit ink (0 = white, 255 = black); `canvas[1,1,1024,1024] = ink as f32 / 255.0`, exact for the two levels used (191, 255). `decoded_sha256` in the manifest pins the f32 canvas. `blank` has no file: `fill: 0`, i.e. `zeros` |
| `boxes` | `[1,8,4]` | f32 | input |
| `valid` | `[1,2]` | f32 | input |
| `node` | `[1,8,2]` | f32 | output |
| `pair` | `[1,5,8,8]` | f32 | output |
| `cls` | `[1,8,13]` | f32 | output |
| `grid` | `[1,16,6,2]` | f32 | `/Reshape_2_output_0` |
| `unit_samples` | `[1,8,768]` | f32 | `/Reshape_4_output_0` |
| `unit_mlp` | `[1,8,256]` | f32 | `/unit/unit.2/Add_output_0` |
| `embed` | `[1,8,256]` | f32 | `/Add_2_output_0` (encoder input) |
| `pos2d_units` | `[1,8,256]` | f32 | `/Concat_11_output_0` |
| `pos2d_freq` | `[64]` | f32 | `/Constant_76` (token `pos2d` table, §3.4) |
| `pos2d_freq_runtime` | `[64]` | f32 | `/Mul_13_output_0` (unit `pos2d` table, §3.4) |
| `token_mask` | `[1024]` | f32 | `/Cast_9_output_0` (`1.0` = masked; the node casts the bool to f32) |
| `enc_layer0` | `[1,8,256]` | f32 | `/layers.0/Add_3_output_0` |
| `enc_layer5` | `[1,8,256]` | f32 | `/layers.5/Add_3_output_0` |
| `hidden_final` | `[1,8,256]` | f32 | `/Add_19_output_0` (what the heads read) |

Not committed, written only by `--with-intermediates` (regenerable, and the sizes are
why): `cnn_map [1,64,64,64]` = `/cnn/cnn.5/cnn.5.2/Relu_output_0` (1 MB),
`avgpool [1,64,32,32]` = `/AveragePool_output_0` (256 KB) and `tokens [1,1024,256]` =
`/Add_5_output_0` (1 MB). The `blank` case records only the six inputs, the three
outputs and `unit_samples`, `unit_mlp`, `embed`, `enc_layer0`, `enc_layer5`,
`hidden_final` — its `grid`, `pos2d_*` and `token_mask` are functions of
`boxes`/`valid` alone and are byte-identical to `page`'s.

Intermediates are captured by adding the tensor names to `graph.output` on an in-memory
copy of the model — the shipped file is untouched. `embed`, `enc_layer0`, `enc_layer5`
and `hidden_final` localise a failure to "input side / layer 0 / last layer /
cross-attention"; `unit_samples` and `unit_mlp` localise the image branch (plus
`cnn_map`/`avgpool`/`tokens` in the full set), and
`pos2d_units`/`pos2d_freq`/`pos2d_freq_runtime` the two position spellings.

## 8. Decisions recorded here

1. **Tensor names = the `.bin`'s**, 110 of them, `contract.json` is the authority and
   the GGUF uses the same strings. The ONNX file's own names are unusable (43 of 110
   are anonymous) and its node scopes are lossy (§2.1).
2. **Convs stay 4-D** (`[cout,cin,kh,kw]`) rather than flattened to `[cout,cin*kh*kw]`
   as in layer 1, because `ops::nn::conv2d` takes `weight_shape: [usize; 4]`. Consequence
   for `--quant q8_0`: the four kernels are not 2-D, so they stay **F32** under the
   layer-1 rule ("2-D rows whose length is a multiple of 32"); everything else that
   qualifies is Q8_0. The f32 file is the lossless default and is what the parity
   fixtures are recorded against.
3. **`config` is the metadata JSON** (`{"cfg": …,"feat_dim":68}`); `config["cfg"]` is the
   `NetConfig`. The GGUF writes the graph's `liteparse.tablenet` string verbatim, while
   the loader (`LayoutNet::from_source`) compares it to `contract.json` by **value**, so
   key order and whitespace are not part of the contract.
