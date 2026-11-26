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

## 4. Open divergence (the blocker)

`step0_v_pred_cond` differs from the oracle by rel `1.4e-1` (`rmse 0.32`), and the
final PNG is `psnr 7.8 dB` against the oracle's — a structured but wrong image
(horizontal banding with a repeating 3×3 motif) rather than noise.

Localisation so far, by first divergence in pipeline order:

```
noise_init                 ok   (rel 4.6e-07)
timesteps                  ok   (bit-identical)
step0_z                    ok   (rel 4.6e-07)
step0_image_embeds         ok   (rel 1.4e-06)   <- vision tower + embedders correct
prefix_last_hidden_state   ok   (corr 0.99965)
prefix_kv (layers 0/1/21/41) ok (corr >= 0.9996)
ConvDecoder (same input)   ok   (corr 1.00000000, max_abs 2.2e-05)
step0_v_pred_cond          FAIL (rel 1.4e-01)   <- generation-branch forward
```

Because the decoder is exact and the prefix (understanding branch: same
attention / RoPE / RMSNorm / MLP code) is correct, the defect is confined to the
**generation branch** of `Trunk::denoise` — i.e. the `*_mot_gen` projection set,
the bidirectional prefix+block attention, or the h/w M-RoPE positions used only
on that path. The trunk's own `tiny_trunk_matches_a_naive_reference` test passes,
so the naive transcription and the optimised path agree; the divergence is
between that transcription and the real oracle, which means the contract itself
is wrong somewhere on this path, not that the kernel is buggy.

Next steps, in order:
1. Dump the oracle's generation-branch hidden states (`_t2i_predict_v` return
   value) by adding one capture point to `make_reference.py`, and diff against
   `step0_hidden_cond` (already emitted by `RMI_SENSENOVA_TRACE`).
2. With that, split the remaining error into "attention over prefix+block" vs
   "`*_mot_gen` MLP/norm", which is a one-layer experiment.
3. Re-check `attention_value_reduce`'s two-call accumulation in `attend`
   (`src/models/diffusion/sensenova_u1/trunk.rs`, the `prefix_len != 0` branch) —
   this branch is *only* reachable from `denoise`, so the prefix pass cannot
   catch a defect in it.

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
