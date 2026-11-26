# OCRia-nano-1: the six ONNX graphs, op by op

Reference for the Rust implementation in `src/models/ocria/`. Everything here was
read off the pinned graphs themselves (`tools/oracle/ocria/dumps/*.txt`, regenerate
with `python -m tools.oracle.ocria.dump_graphs`), not from the release notes.

* Model: `llamaindex/OCRia-nano-1`, ModelScope `master` @ `ca290e82458d32aa8b605dbf3248d7aaf895ea7e`;
  per-file revision + size + SHA256 in `sources.json`.
* Oracle runtime: LiteParse `44ceb6ab160c64cc8b43aab4ed85251719495629`,
  `crates/liteparse/src/ocr/ocria/` (`det.rs`, `rec.rs`, `mod.rs`).
* ONNX opset 17, IR 8, all six produced by PyTorch 2.14.0 except `rec_latin.onnx`
  (`onnx.quantize 0.1.0`, QDQ — see §8).

## 1. Conventions used below

| item | meaning |
| --- | --- |
| `w[out,in,kh,kw]` | ONNX Conv weight, channel-first, as it sits in the file |
| `[out,in]`, `[C,9]` | the **contract** shape: the same weight flattened to 2-D (§9) |
| `k3 s2 p1 g1` | kernel 3x3, stride 2, pad 1 on every side, `group=1`; dilations are 1 everywhere |
| `dw` / `pw` | depthwise (`group=C`, `w[C,1,3,3]`) / pointwise (`k1`, `w[out,in,1,1]`) |
| `HxW` | spatial size of the activation entering that op |
| `+x` | residual `Add` of the named earlier activation, then `Relu` |

All activations are **NCHW**, `float32`. Every Conv has a bias. Every Conv is followed
by `Relu` unless the table says otherwise. There is no BatchNorm, no pooling, no
attention and no Softmax inside any graph except the ones named in §7.

Padding is always symmetric explicit `pads`, never `auto_pad`. A `k3 s2 p1` conv maps
`n -> ceil(n/2)`; a `k1 s1 p0` conv keeps the size; `Resize` with `scales=[1,1,2,2]`
exactly doubles.

**Resize is a plain 2x pixel duplication.** All 10 `Resize` nodes in `det.onnx` carry
`mode=nearest`, `coordinate_transformation_mode=asymmetric`, `nearest_mode=floor`,
`scales=[1,1,2,2]` (a `Constant`), so `out[2i + a] = in[i]` for `a in {0,1}` on both
axes. No cubic, no alignment, no `sizes` input.

## 2. Inputs and outputs

| graph | input | output |
| --- | --- | --- |
| `det.onnx` | `x` F32 `(N,3,H,W)`, `H`/`W` multiples of 32 | `prob` F32 `(N,1,H/2,W/2)` |
| `router.onnx` | `x` F32 `(N,1,32,512)` | `group` F32 `(N,4)`, `flip` F32 `(N,2)` |
| `rec_latin.onnx` | `x` F32 `(N,1,32,W)` | `ids` I64 `(N,T)`, `conf` F32 `(N,T)` |
| `rec_cjk.onnx` | `x` F32 `(N,1,32,W)` | `ids` I64 `(N,T)`, `conf` F32 `(N,T)` |
| `rec_rtl.onnx` | `x` F32 `(N,1,32,W)` | `ids` I64 `(N,T)`, `conf` F32 `(N,T)` |
| `rec_world.onnx` | `x` F32 `(N,1,32,W)` | `ids` I64 `(N,T)`, `conf` F32 `(N,T)` |

`T` is the recognizer's timestep count: `T = W/8` for **cjk** and `T = W/4` for
**latin, rtl, world** (§6). The output height is always 1.

The four recognizers' class counts, and how they line up with the charsets
(`id 0` = CTC blank, `id i` = `charset[i-1]`; every charset entry is exactly one
character, verified):

| group | classes | charset entries | head |
| --- | --- | --- | --- |
| latin | 792 | 791 | one `k1` conv |
| cjk | 6214 | 6213 | `k1` conv to 96, then `k1` conv to 6214, **no Relu between** |
| rtl | 324 | 323 | one `k1` conv |
| world | 1712 | 1711 | one `k1` conv |

## 3. What the pipeline feeds the graphs

From LiteParse (the authority on preprocessing; the graphs assume it):

* **det**: page resized (area average, never enlarged) so the long side is 1600, both
  sides rounded to a multiple of 32 with a floor of 32; input is RGB, each channel
  `v/127.5 - 1` from the uint8 pixel. **The graph converts to luma itself** (`Mul` by
  `[0.299, 0.587, 0.114]`, `ReduceSum` over the channel axis), so the first conv takes
  1 channel. Do the same three multiplies and the sum in F32 on the normalised
  channels rather than folding them into the uint8 values: `Σw_c` is 1.00000000745
  in F32, not 1, and that difference is inside the detector's output.
* **router**: crop → grey → resize to height 32 (width `round(w*32/h)`, at least 8) →
  **tile** the grey row-major pattern across 512 px (`x[y*512 + xx] = g[y][xx % g.w]`)
  → `v/127.5 - 1`. So a short line repeats; nothing is padded with white.
* **recognizer**: crop → grey → resize to height 32, width
  `round(w*32/h)` clamped to `[8,1280]` (area average when shrinking, bilinear when
  growing) → `v/127.5 - 1` → **right-pad with `+1.0` (white) to a multiple of 128**.
  Lines are batched by width, so the padded width a line is run at changes its
  output; the Rust side must reproduce the same bucketing to match bit-for-bit.
  Timesteps read out are `min(w / stride, T)`; a line is *not* read out to the
  padded width.

## 4. `det.onnx` — line detector

`x (N,3,H,W)` → `Mul` (luma weights) → `ReduceSum(axes=[1], keepdims=1)` → `(N,1,H,W)`.

### 4.1 Encoder (stride /32, five stages, residual only on the `s1` blocks)

| stage | op | weight | stride | out |
| --- | --- | --- | --- | --- |
| stem | `Conv /m/stem/stem.0/Conv` | `w[16,1,3,3]` | s2 p1 g1 | `16 x H/2` |
| | `Relu` | | | |
| s4.0 | `dw /m/s4/s4.0/dw/Conv` | `w[16,1,3,3]` | s2 p1 g16 | `16 x H/4` |
| | `Relu` → `pw /m/s4/s4.0/pw/Conv` `w[24,16,1,1]` → `Relu` | | | `24 x H/4` |
| s4.1 | `dw /m/s4/s4.1/dw/Conv` `w[24,1,3,3]` g24 | | s1 p1 | |
| | `Relu` → `pw w[24,24,1,1]` → **`Add` + s4.0 `act_1`** → `Relu` | | | `24 x H/4` |
| s8.0 | `dw w[24,1,3,3]` g24 | | s2 p1 | `24 x H/8` |
| | `Relu` → `pw w[48,24,1,1]` → `Relu` | | | `48 x H/8` |
| s8.1 | `dw w[48,1,3,3]` g48 → `Relu` → `pw w[48,48,1,1]` → `Add`(s8.0) → `Relu` | | s1 p1 | `48 x H/8` |
| s8.2 | `dw w[48,1,3,3]` g48 → `Relu` → `pw w[48,48,1,1]` → `Add`(s8.1) → `Relu` | | s1 p1 | `48 x H/8` |
| s16.0 | `dw w[48,1,3,3]` g48 → `Relu` → `pw w[96,48,1,1]` → `Relu` | | s2 p1 | `96 x H/16` |
| s16.1 | `dw w[96,1,3,3]` g96 → `Relu` → `pw w[96,96,1,1]` → `Add`(s16.0) → `Relu` | | s1 p1 | `96 x H/16` |
| s16.2 | `dw w[96,1,3,3]` g96 → `Relu` → `pw w[96,96,1,1]` → `Add`(s16.1) → `Relu` | | s1 p1 | `96 x H/16` |
| s32.0 | `dw w[96,1,3,3]` g96 → `Relu` → `pw w[128,96,1,1]` → `Relu` | | s2 p1 | `128 x H/32` |
| s32.1 | `dw w[128,1,3,3]` g128 → `Relu` → `pw w[128,128,1,1]` → `Add`(s32.0) → `Relu` | | s1 p1 | `128 x H/32` |

Residuals are `Add(pw_out, <the previous block's post-pw Relu>)`; the blocks whose
first conv has `stride 2` (`s4.0`, `s8.0`, `s16.0`, `s32.0`) have none, so the rule
is "add the block input when the stride is 1". Note `s8` has three blocks and `s16`
three, `s4` two, `s32` two.

### 4.2 FPN top-down (`k1` convs, no bias on any `lat.*`)

```
lat.3 = Conv(/m/lat.3/Conv, w[32,128,1,1]) on s32.1            -> 32 x H/32
lat.2 = Conv(/m/lat.2/Conv, w[32, 96,1,1]) on s16.2            -> 32 x H/16
Add   = lat.2 + Resize2x(lat.3)                                 -> 32 x H/16
lat.1 = Conv(/m/lat.1/Conv, w[32, 48,1,1]) on s8.2             -> 32 x H/8
Add_1 = lat.1 + Resize2x(Add)                                   -> 32 x H/8
lat.0 = Conv(/m/lat.0/Conv, w[32, 24,1,1]) on s4.1             -> 32 x H/4
Add_2 = lat.0 + Resize2x(Add_1)                                 -> 32 x H/4
```
`Add_2` is the only FPN output the head uses directly; the other three taps are
`Add_1`, `Add`, and `lat.3` itself.

### 4.3 Head (four taps, all at `H/4`)

```
smooth.N = Relu(pw(dw(tap)))      # dw: w[32,1,3,3] g32, k3 s1 p1; pw: w[16,32,1,1]; 16 ch
  smooth.0 ← Add_2  (H/4)              -> 16 x H/4          (no resize)
  smooth.1 ← Add_1  (H/8)  -> Resize2x -> 16 x H/4
  smooth.2 ← Add    (H/16) -> Resize2x -> Resize2x -> 16 x H/4
  smooth.3 ← lat.3  (H/32) -> Resize2x -> Resize2x -> Resize2x -> 16 x H/4
Concat(axis=1)                                                  -> 64 x H/4
fuse.0: dw w[64,1,3,3] g64 k3 s1 p1 -> Relu -> pw w[32,64,1,1] -> Relu   -> 32 x H/4
squeeze: Conv(/m/squeeze/Conv, w[12,32,1,1]) no bias            -> 12 x H/4
Resize2x                                                        -> 12 x H/2
lat2 = Conv(/m/lat2/Conv, w[12,16,1,1]) no bias, on stem (16 x H/2) -> 12 x H/2
Add_3 = Resize2x(squeeze) + lat2                                -> 12 x H/2
head.0: dw w[12,1,3,3] g12 k3 s1 p1 -> Relu -> pw w[16,12,1,1] -> Relu  -> 16 x H/2
head.3: Conv(w[1,16,1,1], bias[1]) k1                           ->  1 x H/2  logits
Sigmoid -> prob  (N,1,H/2,W/2)
```

The **Sigmoid is in-graph**: `prob` is already the segmentation probability, and the
detector's `thresh` (0.55 from `det.meta.json`) compares against it directly. The
prob map is at **half** the network input resolution, i.e. `H/2 x W/2`, which is why
LiteParse's `stride = h / ph` is 2.

Requiring `H`/`W` to be multiples of 32 is not cosmetic: every `Add` in §4.2/§4.3
pairs a `2x` nearest upsample with a `ceil(n/2)`-strided activation, and the two are
only guaranteed equal when the sizes are exact powers of two. Feed multiples of 32.

## 5. `router.onnx` — script router

`x (N,1,32,512)`; all convs `k3 s? p1` depthwise or `k1`, all with bias, all followed
by `Relu`.

```
stem.0     Conv w[16,1,3,3] g1   s(2,2)  -> 16 x 16 x 256   Relu
body.0     dw g16  w[16,1,3,3]   s(2,2)  -> 16 x  8 x 128   Relu
           pw w[32,16,1,1]                 -> 32 x  8 x 128   Relu
body.1     dw g32  w[32,1,3,3]   s(1,1)                    Relu
           pw w[32,32,1,1]  + body.0 act_1  -> 32 x 8 x 128   Relu
body.2     dw g32  w[32,1,3,3]   s(2,2)  -> 32 x  4 x  64   Relu
           pw w[64,32,1,1]                 -> 64 x  4 x  64   Relu
body.3     dw g64  w[64,1,3,3]   s(1,1)                    Relu
           pw w[64,64,1,1]  + body.2 act_1  -> 64 x 4 x 64    Relu
body.4     dw g64  w[64,1,3,3]   s(2,2)  -> 64 x  2 x  32   Relu
           pw w[96,64,1,1]                 -> 96 x  2 x  32   Relu
body.5     dw g96  w[96,1,3,3]   s(1,1)                    Relu
           pw w[96,96,1,1]  + body.4 act_1  -> 96 x 2 x 32    Relu
body.6     dw g96  w[96,1,3,3]   s(2,1)  -> 96 x  1 x  32   Relu
           pw w[96,96,1,1]                 -> 96 x  1 x  32   Relu
Squeeze(axes=[2])                          -> 96 x 32
ReduceMean(axes=[2], keepdims=0)           -> 96
ReduceMax (axes=[2], keepdims=0)           -> 96
Concat(axis=1)                             -> 192
Gemm /group/Gemm  W[4,192] b[4] transB=1 alpha=beta=1  -> group (N,4)
Gemm /flip/Gemm   W[2,192] b[2] transB=1 alpha=beta=1  -> flip  (N,2)
```

Residuals only on the stride-1 blocks (1, 3, 5), each `Add(pw_out, previous block's
post-pw Relu)` then `Relu`. The pooled feature is `concat(mean_w, max_w)` over the 32
remaining width positions — **not** over height (height is already 1).

**No Softmax in-graph.** LiteParse softmaxes `group` (4-way) and takes
`softmax(flip)[1]` as the upside-down probability; both stay F32 in Rust.

## 6. `rec_{latin,cjk,rtl,world}.onnx` — CTC recognizers

All four share the stem, the nine-block body and the three `mix` blocks; they differ
in the **last downsampling stride** and in the **head**. `x (N,1,32,W)`.

```
stem.0     Conv w[32,1,3,3] g1  s(2,2) -> 32 x 16 x W/2   Relu
body.0     dw g32  s(2,1)  -> 32 x  8 x W/2   Relu;  pw w[64,32,1,1] -> 64 x 8 x W/2  Relu
body.1     dw g64  s(1,1)  Relu; pw w[64,64,1,1]; Add(+body.0 act_1); Relu   -> 64 x 8 x W/2
body.2     dw g64  s(2,2)  -> 64 x  4 x W/4   Relu;  pw w[128,64,1,1] -> 128 x 4 x W/4 Relu
body.3     dw g128 s(1,1)  Relu; pw w[128,128,1,1]; Add(+body.2 act_1); Relu -> 128 x 4 x W/4
body.4     dw g128 s(1,1)  Relu; pw w[128,128,1,1]; Add(+body.3 act_1); Relu -> 128 x 4 x W/4
body.5     dw g128 s(2,1)  -> 128 x 2 x W/4  Relu;  pw w[192,128,1,1] -> 192 x 2 x W/4 Relu
body.6     dw g192 s(1,1)  Relu; pw w[192,192,1,1]; Add(+body.5 act_1); Relu -> 192 x 2 x W/4
body.7     dw g192 s(1,1)  Relu; pw w[192,192,1,1]; Add(+body.6 act_1); Relu -> 192 x 2 x W/4
body.8     dw g192 s(?,?)  -> 192 x 1 x W/8 or W/4  Relu; pw w[192,192,1,1] -> 192 x 1 x ... Relu
Squeeze(axes=[2]) -> (N,192,W')
```

`body.8` is the one divergence — its **width** stride:

| group | `body.8/dw/Conv` strides | `W'` | `T` | head |
| --- | --- | --- | --- | --- |
| latin | `[2,1]` | `W/4` | `W/4` | `Conv w[792,192,1]` |
| cjk | `[2,2]` | `W/8` | `W/8` | `Conv w[96,192,1]` → `Conv w[6214,96,1]` |
| rtl | `[2,1]` | `W/4` | `W/4` | `Conv w[324,192,1]` |
| world | `[2,1]` | `W/4` | `W/4` | `Conv w[1712,192,1]` |

Every `body.N/dw` is `k3 s? p1` with `w[C,1,3,3]`, `group=C`; every `pw` is `k1` with
`w[C_out,C_in,1,1]`; residuals on the stride-1 blocks 1, 3, 4, 6, 7 (blocks 0, 2, 5,
8 change the size and have none), each `Add(pw_out, previous block's post-pw Relu)`
then `Relu`. Note `body.0` and `body.5` stride only in **height**.

### 6.1 `mix` blocks (three, chained, 1-D over width, 192 channels)

Each block takes the previous block's output (`x0 = Squeeze output` for `mix.0`) and
returns the same shape `(N,192,W')`:

```
dw   = Conv1d  w[192,1,9] g192  k=9 pad=4 stride=1   -> Relu        (depthwise 9-tap)
Add  = dw + block_input                              (no Relu)
fc1  = Conv1d  w[384,192,1] k=1  bias[384]           -> Relu
fc2  = Conv1d  w[192,384,1] k=1  bias[192]
Add_1= Add + fc2                                     (no Relu; feeds the next block)
```
Two adds per block: `Add` mixes the depthwise output with the **block input**, `Add_1`
mixes that with the `fc2` output. `mix.0`'s block input is the `Squeeze` output,
`mix.1`'s is `mix.0/Add_1`, `mix.2`'s is `mix.1/Add_1`. There is **no** `Relu` after
either add.

The three `mix` blocks are 1-D convolutions: in ONNX they are `Conv` with
`kernel_shape=[9]` / `[1]` on a `(N,192,W')` tensor, i.e. width-only, and they are
stored in the contract as `[192,9]` (dw) and `[384,192]` / `[192,384]` (fc), see §9.

### 6.2 Head and where argmax / softmax-max happen

```
logits = head(...)        # (N,C,T) -- 3-D, the height axis is gone since Squeeze;
                          # cjk: Relu-free 96-wide bottleneck then C=6214;
                          # latin/rtl/world: a single k1 conv
ReduceMax       axes=[1] keepdims=0   ->  m_t   (N,T)     max logit over classes
ArgMax axis=1   keepdims=0            ->  ids   (N,T) I64 argmax over classes
ReduceLogSumExp axes=[1] keepdims=0   ->  lse_t (N,T)     log Σ_c exp(logit_c)
Sub                                   ->  m_t - lse_t
Exp                                   ->  conf  (N,T)     = softmax(logits)[argmax]
```

So **the CTC argmax and the softmax-max both happen inside the graph**, at the very
end, on the same `(N,C,T)` logits:

* `ids[t] = argmax_c logits[c,t]`, first index on ties (ONNX `select_last_index=0`);
* `conf[t] = exp(m_t - lse_t)`, i.e. the softmax probability of that argmax.

Rust must reproduce exactly that pair and nothing else: `conf = 1 / Σ_c exp(x_c - m)`
is the numerically stable form of `exp(m - lse)` (ORT computes `log(Σ exp(x))`
naively, which only differs for absurd logits). The CTC collapse (blank `id 0`,
drop consecutive repeats, then map `id-1` into the charset) happens **outside** the
graph, in the decoder — see `rec.rs` in the pinned LiteParse for the exact loop, and
note `conf` is averaged over the *kept* characters only (empty reading → 0.0).

## 7. Everything else the graphs do, in one place

| op | where | what to reproduce |
| --- | --- | --- |
| `Mul` + `ReduceSum(axes=[1])` | det, first two nodes | luma `0.299R+0.587G+0.114B` on the normalised channels |
| `Relu` | everywhere except between cjk's two head convs and after `mix`'s `Add`/`Add_1` | `max(x,0)` |
| `Add` | det encoder residuals, FPN, det head `Add_3`, router/rec residuals, `mix` | plain elementwise add, see the per-graph rules |
| `Resize` (10x, det only) | FPN + head | nearest 2x pixel duplication, `out[2i+a]=in[i]` |
| `Concat(axis=1)` | det head (4 taps → 64ch), router (`mean‖max` → 192) | channel concat |
| `Squeeze(axes=[2])` | router, rec | drop the height-1 axis |
| `ReduceMean/ReduceMax(axes=[2])` | router | mean/max over width (32 positions) |
| `Gemm` transB=1 | router ×2 | `y = W x + b`, `W[out,in]` |
| `Sigmoid` | det last node | `1/(1+exp(-x))` |
| `ArgMax`/`ReduceMax`/`ReduceLogSumExp`/`Sub`/`Exp` | rec last five nodes | §6.2 |
| `QuantizeLinear`/`DequantizeLinear` | `rec_latin.onnx` only | §8 — Rust skips the activation half |

## 8. `rec_latin.onnx` is ONNX-Runtime-quantized (QDQ)

Metadata `onnx.infer=onnxruntime.quant`, `onnx.quant.pre_process=onnxruntime.quant`;
producer `onnx.quantize 0.1.0`. Its topology is identical to §6/§6.1 with the head of
latin, but every conv input and weight is quantized.

### 8.1 There are no `Relu` nodes in this graph — they are folded into the quantizers

`rec_latin.onnx` contains **zero `Relu` nodes**. Its op set is `Conv`, `Add`,
`Squeeze`, `ReduceMax`, `ArgMax`, `ReduceLogSumExp`, `Sub`, `Exp`, plus 100
`DequantizeLinear` and 42 `QuantizeLinear`. ONNX Runtime's quantizer deleted every
Relu and relies on the following `uint8` activation `QuantizeLinear`, whose zero
point 0 clamps negatives to 0: `QuantizeLinear` then *is* the Relu (its output is
still named `.../act/Relu_output_0`). The 27 quantizers with `zp = 0` are exactly
the architecture's Relu outputs, plus `Squeeze` (a Relu output) and `mix.0/Add` (a
sum of two non-negative tensors).

**The Rust implementation must therefore apply the same Relus as `rec_rtl` /
`rec_world`, even though latin's node list shows none:** after the stem conv, after
every body block's `dw` and `pw` (and after the residual `Add` of blocks 1, 3, 4, 6,
7), after each `mix` `dw` and each `mix` `fc1` — 25 in all — and not after `mix`
`Add`/`Add_1`, not between cjk's two head convs, and not after the head conv. A port
that reads latin's node list literally and omits them produces activations of order
1e5 and gets every timestep wrong.

### 8.2 Weights and activations

* **Weights** — each Conv weight is an `int8` `*_quantized` initializer plus a
  per-output-channel `float32` `*_scale` (length `C_out`) and an `int8` `*_zero_point`
  (length `C_out`, **all zeros** here), reaching the Conv through a `DequantizeLinear`
  node (`axis=0` on the `[C_out,C_in,kh,kw]` weight). The converter dequantizes them
  with the ONNX QDQ arithmetic `y = (x - zp) * scale` and stores F32, so Rust's F32
  arithmetic holds exactly the values ORT's dequantized weights hold.
* **Activations** — `uint8` `QuantizeLinear`/`DequantizeLinear` pairs sit
  **after every Relu** (i.e. on every conv input), **after every `pw`/`conv` output
  that feeds a residual `Add`**, after each `mix` `Add`/`Add_1`, on the graph input
  `x`, and after the head conv. Per-tensor (scalar) scales. 42 of them, plus 58
  weight `DequantizeLinear` nodes (100 QDQ nodes in all).

Their scales are `max_abs / 255` of a calibration run, so they depend on the
calibration data, not on the input: `x_scale = 0.007843137718737125` (that is
`2/255`, zero point 128 → the input range is `[-1,1]`, matching the preprocessing in
§3), `Relu` outputs have `zero_point = 0`, the pre-residual and head tensors have
non-zero zero points (62…181), and the head logits carry
`scale = 0.3328447639942169`, `zero_point = 181`.

**Rust computes in F32 and does not reproduce activation quantization.** That is an
accepted precision loss, measured by `tools/oracle/ocria/qdq_effect.py` on `ids` and
`conf` (never on raw logits). It builds the faithful F32 reference — the same
dequantized weights the GGUF carries, the activation pairs rewired out, the folded
Relus of §8.1 put back — and runs both graphs on the same inputs:

| input | timesteps | `ids` differing | max abs Δ`conf` |
| --- | --- | --- | --- |
| fixtures `w64` / `w256` / `w384` / `w512` | 16 / 64 / 96 / 128 | 1 / 3 / 6 / 8 | 0.15-0.19 |
| fixtures, all four | 304 | **18 (5.92 %)** | |
| real receipt text, 10 bands of 32 px | 1600 | **9 (0.56 %)** | 0.43 |

The per-case table, with the exact differing timesteps and both decoded texts, is
`tools/oracle/ocria/latin_qdq_effect.json`; the fixtures' `ids_f32` blobs differ from
the QDQ `ids` at exactly the timesteps it lists.

The fixtures are a deliberately hostile case: a repeated stroke pattern leaves many
near-ties (the top-2 logit gap at the differing timesteps is 0.04-0.56, inside or
next to one 0.333 logit bin), so the argmax flips 5.9 % of the time. Real glyphs have
wide margins and read identically on most bands (`CORNER STORE`, `Coffee beans
2 12.50`); the bands that differ are off by one character. Activation quantization
moves the logits by up to ~9 in the tail (it compounds through 29 layers), but the
argmax rarely follows.

So: `cjk`, `rtl` and `world` are plain F32 and must match the fixtures exactly;
`latin`'s parity target is the F32 reference (`ids_f32`/`conf_f32` in its fixture
cases), not the QDQ graph, and the QDQ numbers stay recorded next to it so the size
of the loss remains measurable. Nothing else in the pipeline is affected.

## 9. Tensor names and shapes in the GGUF (the loader contract)

`src/models/ocria/contract.json` holds the authoritative `{name: shape}` map (342
tensors); the loader reads exactly those keys and rejects anything else.

Name = `<component>.<node-path>.<weight|bias>`:

* `component` is `det`, `router`, or `rec.<group>`;
* `node-path` is the ONNX node name with the leading `/` stripped, a leading `m`
  segment dropped, the trailing op segment (`Conv`/`Gemm`/`MatMul`) dropped, and the
  rest joined with `.`;
* `bias` only exists when the node has a third input (no bias on `det.lat.*`,
  `det.squeeze`, `det.lat2`).

Examples (all real keys of the contract):

| ONNX node | key | shape |
| --- | --- | --- |
| `/m/stem/stem.0/Conv` (det) | `det.stem.stem.0.weight` | `[16,9]` |
| `/m/s4/s4.0/dw/Conv` | `det.s4.s4.0.dw.weight` | `[16,9]` |
| `/m/s4/s4.0/pw/Conv` | `det.s4.s4.0.pw.weight` | `[24,16]` |
| `/m/lat.3/Conv` | `det.lat.3.weight` | `[32,128]` (no bias) |
| `/m/head/head.0/head.0.0/Conv` | `det.head.head.0.head.0.0.weight` | `[12,9]` |
| `/body/body.0/pw/Conv` (router) | `router.body.body.0.pw.weight` | `[32,16]` |
| `/group/Gemm` | `router.group.weight` / `.bias` | `[4,192]` / `[4]` |
| `/m/mix/mix.0/dw/Conv` (rec) | `rec.cjk.mix.mix.0.dw.weight` | `[192,9]` |
| `/m/mix/mix.0/fc1/Conv` | `rec.cjk.mix.mix.0.fc1.weight` | `[384,192]` |
| `/m/head/head.1/Conv` | `rec.cjk.head.head.1.weight` | `[6214,96]` |
| `/m/head/Conv` (latin) | `rec.latin.head.weight` | `[792,192]` |

Some stems look redundant because the ONNX node names are
(`det.smooth.0.smooth.0.0.smooth.0.0.0.weight`, `det.fuse.fuse.0.fuse.0.0.weight`,
`det.head.head.0.head.0.0.weight`): the rule is mechanical, so the keys are too. Use
`grep` on `contract.json` rather than guessing.

**Shapes are flattened to 2-D, C-order, nothing transposed.** Every Conv weight is
stored as the `[out, fan_in]` matrix obtained by flattening the *last three* axes in
C order, so row `o` is output channel `o` and

```
column = ((c_in * K_h) + ky) * K_w + kx      holds  w[o, c_in, ky, kx]
column = c_in * K + k                        holds  w[o, c_in, k]   (1-D convs)
```

Concretely: `[out,in,1,1] -> [out,in]`; `[C,1,3,3] -> [C,9]` with column
`c_in*9 + ky*3 + kx` (that is the depthwise tap order); `[C,1,9] -> [C,9]` with
column `c*9 + k`; `[out,in,1] -> [out,in]`. `Gemm`/`MatMul` are stored `[out,in]`
(a `transB=0` `Gemm` or a `MatMul` is transposed to that, which is the only
transpose anywhere). Biases stay 1-D `[out]`.

Read it back with `w[o][((c*K_h)+ky)*K_w + kx]`, or in Rust
`w[o * fan_in + c * K_h * K_w + ky * K_w + kx]`; a `[C,9]` depthwise kernel is
`w[c * 9 + ky * 3 + kx]`. The converter asserts this shape against `contract.json`
and against each node's `kernel_shape`, so a silent transpose fails the build of the
GGUF rather than the Rust model. This is what lets Q8_0 quantize the matrices whose
row length is a multiple of 32 (§10) and gives the Rust side a plain matmul for
every kernel.

## 10. Weight storage and Q8_0

`f32` is lossless. `q8_0` stores a tensor in Q8_0 only when it is 2-D and its row
length is a whole number of 32-value blocks; everything else stays F32. Under §9 that
means: pointwise convolutions with `in % 32 == 0`, the `mix` `fc1`/`fc2` matrices, the
head convolutions (`[C,192]`), and the router's two `Gemm` matrices. The depthwise
kernels (row length 9), the 9-tap `mix` `dw` kernels and every bias stay F32.
`src/models/ocria/mod.rs::load_tensor` enforces the same rule.
