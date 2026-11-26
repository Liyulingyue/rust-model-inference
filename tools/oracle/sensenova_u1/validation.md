# SenseNova-U1.6-8B-MoT — validation record

Measured on the AMD Radeon Cloud box (`36.150.116.206`, 128 cores, cgroup
`memory.max` ≈ 55 GB, ROCm 7.2.4 / 1 × gfx1100 visible), **2026-11-26**.
Oracle: `github.com/OpenSenseNova/SenseNova-U1` @ `d2d9bd1`.
Weights: `SenseNova/SenseNova-U1.6-8B-MoT-GGUF`, `Q8_0`,
20,695,981,472 B, sha256 `097ca40a942211c89fe0e945add3d159839e904a16feb5f9d0d14ba8e6644806`
(verified on the box before any run).

## Status: partial adaptation — numeric alignment NOT yet passing

The port loads, runs end-to-end and produces a PNG through the real CLI, but the
generated image does not yet match the oracle. Everything below is measured, not
inferred; the open divergence is described in §4.

## 1. What is verified

| Check | Result | Evidence |
| --- | --- | --- |
| Tensor contract (names) | **1115 / 1115 exact** | `gguf_dump.py` on the real GGUF vs the names `trunk.rs`/`vision.rs`/`fm.rs` build. Only `language_model.lm_head.weight` is unread (T2I never emits logits). |
| Tensor contract (shapes) | **0 mismatches / 1115** | Same dump, per-tensor `dims` compared against `config.rs`. |
| Unit tests | **36 passed, 0 failed** | `cargo test --profile release-fast --lib sensenova_u1` |
| Tokenizer | **matches the oracle exactly** | Conditional prompt 266 ids, unconditional 9 ids, pinned in `tokenizer.rs` tests and cross-checked against `added_tokens.json`; `merges.txt` byte-identical to the HF copy. |
| Chat template | **exact** | `neo1_0` MPT render including the "empty system message emits no system block" quirk. |
| `timesteps` | **bit-identical** | Rust vs oracle `timesteps.npy`: `[0, 0.10000002, 0.25, 0.5, 1.0]`, `max_abs = 0`. |
| Initial noise (`TorchMt19937`) | **matches** | `noise_init`: `max_abs 4.4e-6`, rel `4.6e-7` — the repo's torch-compatible MT19937 reproduces `torch.randn(Generator('cpu').manual_seed(42))`. |
| Vision tower + timestep/noise embedders | **matches** | `step0_image_embeds`: rel `1.4e-6`, `max_abs 1.8e-3`. Proves patch-embed conv, 2-D interleaved RoPE, 2×2 downsample conv, exact-erf GELU, both MLP embedders and patchify ordering. |
| ConvDecoder (pixel head) | **exact** | Fed the same input to the Rust decoder and a torch rebuild from the GGUF (`probe_decoder.py`): every stage — ps1, conv1, GELU, ps2, conv2, ps3 — `corr = 1.00000000`, `max_abs ≤ 2.2e-5`. |
| Prefix pass (understanding branch) | **correct** | `prefix_kv_layer{00,01,21,41}_{k,v}` `corr ≥ 0.9996`, rel ≤ 2.5e-2; `prefix_last_hidden_state_cond` `corr 0.99965`. |
| End-to-end CLI | **runs** | `rust-model-inference --model …Q8_0.gguf --prompt … --width 512 --height 512 --steps 4 --cfg 4.0 --seed 42 --out …` → PNG in **152 s** (wall), exit 0. |

## 2. Performance (same machine, same weights, same input)

| Implementation | Wall clock, 512×512, 4 steps | Peak RSS |
| --- | --- | --- |
| Python oracle (CPU, f32, dequant-on-forward) | **1057.2 s** | 33.8 GiB |
| Rust port (CPU, f32, `Weight` Q8_0 kernels) | **152–190 s** | ~17 GiB |

≈ **6–7× faster than the reference** on the CPU path. No GPU numbers: there is no
Vulkan path for this model yet (`--compute vulkan` is rejected explicitly, it does
not silently fall back).

## 3. Coverage matrix

| Weight format | Load | CPU forward | Vulkan | CLI | HTTP |
| --- | --- | --- | --- | --- | --- |
| `Q8_0` | 已实现 / 通过 | 已实现 / 数值失败 | 未实现 | 已实现 / 已运行 | 不适用¹ |
| `Q6_K` | 已实现 / 未运行 | 已实现 / 未运行 | 未实现 | 已实现 / 未运行 | 不适用¹ |
| `Q4_0` | 已实现 / 未运行 | 已实现 / 未运行 | 未实现 | 已实现 / 未运行 | 不适用¹ |
| `F32` / `F16` | 已实现 / 未运行 | 已实现 / 未运行 | 未实现 | 已实现 / 未运行 | 不适用¹ |

¹ HTTP is **not wired for this model**. `main` now serves
`POST /v1/images/generations` for P-JiT / Z-Image / ERNIE-Image, but the backend
is selected by the model's weights and SenseNova-U1.6 is not registered on it —
so the capability exists in the repo and this model is simply not connected.
That is a real remaining gap, not "not applicable".

`Q6_K` / `Q4_0` were downloaded on an earlier instance of the box but that
instance was reclaimed before they could be exercised; only `Q8_0` was run.

## 4. Open divergence (the blocker) — localised to a ~1e-2 error from layer 0

The generated image is structured but wrong (`psnr 7.83 dB`, `pixels_differing
99.5 %`). This session bisected it with per-layer captures on both sides
(`probe_trunk.py` hooks the oracle's decoder layers; the Rust side dumps
`denoise_layer{NN}_{cond,uncond}` under `RMI_SENSENOVA_TRACE`).

**Everything downstream of the trunk is correct.** Feeding the *same* input to
the Rust ConvDecoder and a torch rebuild from the GGUF gives `corr 1.00000000`
at every stage. Feeding the oracle's own `norm_mot_gen`-normalised hidden to
torch reproduces the oracle's `x_pred` to `corr 1.00000000` (so recovering
`x_pred = v_pred * denom + z` is valid). And the decoder is *not* hypersensitive:
torch on the Rust `decoder_in` vs torch on the oracle `decoder_in` is
`corr 0.99995799` (`max_abs 0.0099`).

**The trunk carries a small systematic error, present from layer 0:**

| checkpoint | corr | rel (max_abs / max) |
| --- | --- | --- |
| `image_embeds` (trunk input) | — | 1.4e-6 |
| `denoise_layer00_cond` | 0.99997370 | 1.27e-2 |
| `denoise_layer01_cond` | 0.99998171 | 1.07e-2 |
| `denoise_layer02_cond` | 0.99998522 | 9.87e-3 |
| `denoise_layer21_cond` | 0.99999170 | 7.32e-3 |
| `denoise_layer41_cond` | 0.99998487 | 1.59e-2 |
| post-`norm_mot_gen` hidden | 0.99999854 | 5.35e-3 |

The relative error is **already ~1e-2 at layer 0 and does not grow with depth** —
so this is not accumulation, it is a small algorithmic difference in the first
layer's forward (the same signature appears on the understanding branch: layer-0
prefix K/V is `rel 3.7e-3`, and the final prefix hidden is `rel 1.7e-2`).

It then **compounds across denoise steps**, which is what wrecks the image:

| step | `z_next` corr | rel |
| --- | --- | --- |
| 1 | 0.99521690 | 5.9e-2 |
| 2 | 0.96898049 | 2.05e-1 |
| 3 | 0.11303292 | 1.86 |

CFG with `cfg_scale = 4.0` multiplies the branch difference by 4 every step, so
a 1e-2 seed error collapses by step 3.

**Ruled out this session:** precision/SIMD. `RMI_SCALAR=1` (which forces the
scalar `expf` softmax instead of the x86 `exp_approx_avx2` path) gives
`psnr 7.83 dB` again — identical to 2 decimals. So the gap is not the
approximate exponential, not reduction order, and not f32 rounding.

### Root cause (this session): activation quantization, not a porting bug

Bisecting the *prefix* (understanding) branch layer 0 — where there is no prefix
cache, a plain causal mask, and the input is provably exact — pins it down:

| point | corr | rel |
| --- | --- | --- |
| `prefix_embed` (embed_tokens output) | **1.00000000** | 0 |
| `prefix_l0_normed` (input_layernorm) | **1.00000000** | 7.97e-8 |
| `prefix_l0_q` (q_proj output) | 0.99999654 | **1.96e-3** |
| `prefix_l0_k` (k_proj output) | 0.99999721 | 9.43e-4 |
| `prefix_l0_v` (v_proj output) | 0.99998262 | 1.12e-3 |

Embedding lookup and RMSNorm are **bit-exact**; the error appears exactly at the
first projection. The port multiplies through `PreparedRows`
(`Weight::quantize_and_matmul_with_scratch`), which **quantizes the activation to
Q8_0** before the matmul — the llama.cpp-style fast path. The oracle dequantizes
the Q8_0 weights and multiplies by **f32 activations**. A Q8_0 activation has
~1/256 resolution, so ~1e-3 relative error per projection is expected and
matches the measurement.

That ~1e-3/layer is not a bug, but it is fatal here: 42 layers carry it to
~1e-2, and the 4-step flow loop with `cfg_scale = 4.0` multiplies the branch
difference by 4 every step, so the image collapses by step 3. Other models in
this repo tolerate the same error because they are verified with error metrics,
not by image parity against a torch oracle.

**Correction (verified, this session): the f32-activation switch does NOT fix it.**
`RMI_F32_ACTIVATIONS=1` was implemented and run against the real weights: the
projection error is **bit-identical** to the quantised path
(`prefix_l0_q` rel `1.958e-3`, `max_abs 2.287958e-03` both ways). The reason is
that the Q8_0 kernel quantises the activation **inside** `Kernel::forward` — the
llama.cpp `vec_dot_q8_0_q8_0` contract requires a Q8_0 activation, so handing it
f32 rows just moves the same rounding one call deeper. `prepare_f32` is therefore
a no-op for quantised weights (it stays useful for F32/F16 weights, where it is
bit-exact against the quantised path).

**Real fix path:** dequantise the Q8_0 weights to f32 once at load and run an
F32 kernel, i.e. a genuine "dequantised weights + f32 activations" mode, not just
unquantised activations. That is a weight-side transform, not an activation-side
one.

**Fix path:** give this model an f32-activation matmul. `Weight::matmul(&[f32])`
already exists and runs the kernel with f32 activations (weights dequantized on
the fly), and `QuantizedLinear::forward_dequant` does the same in `ops::linear`.
The contained change is inside `PreparedRows` in `trunk.rs`: add a passthrough
mode that skips `quantize_q8_0_into` and calls `Weight::matmul` per row, selected
by a parity flag. That trades speed for oracle parity; the fast quantized path
should stay the default.

**Next step (narrowed):** the first layer's *understanding* branch is the
cheapest place to bisect further, because it is the same code with a causal mask
and no prefix. Compare, at layer 0 of the prefix forward: `input_layernorm`
output, `q/k/v` after projection, `q_norm`/`q_norm_hw` output, post-RoPE
`q/k`, and the attention output before `o_proj`. The candidate list is short —
RMSNorm epsilon, the 64-wide `q_norm_hw` applied *before* the h/w split, or the
RoPE segment theta — and each is a one-line experiment against a
`prefix_layer00_*` capture added to `probe_trunk.py`.

## 5. How to reproduce

```bash
# 0. weights (hash-verified, single process — see the warning in download_quant.sh)
bash tools/oracle/sensenova_u1/download_quant.sh Q8_0

# 1. oracle reference (≈18 min on CPU)
bash tools/oracle/sensenova_u1/bootstrap_remote.sh
bash tools/oracle/sensenova_u1/run_reference.sh

# 2. Rust port with checkpoint tracing
RMI_SENSENOVA_TRACE=/root/trace \
  rust-model-inference --model <dir>/SenseNova-U1.6-8B-MoT-Q8_0.gguf \
  --prompt "A studio photograph of a red apple on a white table, soft daylight." \
  --width 512 --height 512 --steps 4 --cfg 4.0 --seed 42 --out rust.png

# 3. compare
python3 tools/oracle/sensenova_u1/compare_trace.py /root/trace <ref512>
python3 tools/oracle/sensenova_u1/compare_png.py rust.png oracle.png
```

## 6. Notes / gotchas found along the way

- **Q8_0 is 20.7 GB, not ~9 GB**: only 589 of 1116 tensors are quantized; 527
  (embeddings, norms, convs, timestep MLPs) stay full precision. Budget ~21 GB
  RSS for weights alone.
- **`noise_scale` at 512² is 2.0**, not 4.0 — the formula divides by
  `merge_size²` (`sqrt(32·32 / 4 / 64) = 2`).
- **`_apply_time_schedule` hardcodes `time_schedule = "standard"`**, so the
  `dynamic` / `base_shift` / `max_shift` / `time_shift_type` values in
  `config.json` are dead code in this release.
- **The unconditional prompt has no system block**: `get_prompt()` emits nothing
  when `system_message == ''`, but the empty *user* message still emits
  `role + sep`.
- **Vision RoPE is interleaved** (`apply_rotary_emb_1d` pairs `(2i, 2i+1)`),
  while the LLM trunk is half-split (`rotate_half`). Same frequencies, different
  pairing; they diverge from position 1.
- **GGUF conv dims are the reversed torch shape but the payload is already
  torch-order** `[out][in][kh][kw]`; permuting the data scrambles the kernel.
- **Never run two `wget` downloads of the same GGUF at once**: `-c -O` on a
  shared path interleaves and produced a 22,692,593,723-byte file instead of
  20,695,981,472. `download_quant.sh` now enforces one process plus a sha256 gate.
- The Radeon Cloud Template was reclaimed **four times** during this work
  (`/root` is wiped; `/workspace` persists), which is why the reference run and
  the port run were eventually done on a fifth instance.
