# Validation record — 2026-10-31

Host: Apple M3 Max, macOS arm64, Rust 1.99.1, torch 2.9.1 CPU F32.
The source/checkpoint pins and input SHA256s are in [README.md](README.md).
Generated artifacts are intentionally ignored under `artifacts/wevisone/`.

## Reference comparisons

The deterministic 128×128 image is a red rectangle on white. The official
segmentation instruction returns:

```text
<ref>the red rectangle</ref><box><258><170><773><799><OBJ></box>
```

The 12 generated IDs (excluding EOS) match exactly for CPU BF16, CPU F16,
CPU Q8_0, CPU-only Auto fallback, and the real Vulkan Auto CLI/HTTP run.
Tokenizer samples also cover Chinese, whitespace, emoji, accented text and
coordinate/OBJ markup. Full prompt IDs and all three M-RoPE axes are exact.

CPU BF16 versus unmodified Transformers F32 with original weights:

| Checkpoint | Maximum absolute error | RMSE |
|---|---:|---:|
| raw ViT layer 5 | 8.20e-5 | 2.35e-6 |
| raw ViT layer 11 | 2.90e-4 | 2.12e-6 |
| raw ViT layer 17 | 3.20e-4 | 4.54e-6 |
| raw ViT layer 23 | 1.294e-2 | 1.427e-4 |
| merged visual features | 1.271e-4 | 1.126e-5 |
| DeepStack | 4.20e-5 | 2.05e-6 |
| raw LM sequence | 5.37e-3 | 1.63e-4 |
| final prompt logits | 5.44e-5 | 1.12e-5 |
| raw OBJ hidden | 1.10e-3 | 3.74e-5 |
| full 1024² mask logits | 8.85e-5 | 1.10e-5 |
| original-size float alpha | 1.32e-5 | 4.27e-7 |

The largest ViT difference is on an activation near 936.58; the large raw LM
activation at the maximum difference is near 7141.66. Both max and RMSE are
checked rather than interpreting large absolute activations as bitwise parity.
The independent decoder-only fixture checks all four encoded levels, query
features, pixel fusion, premerge order and two mask logits; final logits
max_abs is 5.72e-5.

CPU lower-precision text against the same F32 original-weight reference:

| Text type | Raw hidden relative RMSE | Logits max abs | Logits relative RMSE | Logits cosine |
|---|---:|---:|---:|---:|
| F16 | 0.0006002 | 0.005841 | 0.0003376 | 0.99999995 |
| Q8_0 | 0.04732 | 0.37676 | 0.01901 | 0.99983762 |

Q8_0 has measurable quantization error (raw hidden max_abs 100.53); it is not
claimed equivalent to BF16. The fixture's greedy sequence remains unchanged.
Approximate checks require raw relative RMSE ≤6%, cosine ≥0.998, and logits
relative RMSE ≤3%, cosine ≥0.999. These are fixture checks, not a task-quality
benchmark. BF16/F32-weight calculations use tighter per-checkpoint absolute
and RMSE gates. F16 rounds activations at its existing kernel boundaries.

## Real entry points and dispatch

- Dedicated CLI Auto returned one normalized box and a 128×128 grayscale alpha PNG.
- HTTP `/v1/wevisone/predict`: 200 and the same response as CLI; oversized
  context and an invalid task each returned 400.
- Final CLI run with a 69-token context (57 prompt + 12 generated tokens)
  completed mask replay and matched the earlier prediction, including PNG.
- Ordinary HTTP image chat produced the exact reference markup/coordinates.
  Plain text Auto CLI returned `OK` for `Reply with exactly OK.`; text JEV
  CLI/HTTP agreed on yes with probability 0.9997690 for the stated red color.
  The WeVisOne assistant prefix omits empty thinking markers. Fresh text JEV
  sessions allocate KV for the actual prompt, not the declared 262144-token cap.
- Real MoltenVK trace identifies Apple M3 Max and `scope=host_linear
  backend=Vulkan`; counters reached 3542 submissions for generation and
  hidden replay. CPU host/attention and pixel-decoder work remains.
- The Auto PNG alpha differs from reference float alpha by at most 0.001961,
  including the unavoidable rounding to 8 bits (RMSE 0.000325).
- CPU-only Auto fallback matches BF16 reference metrics and greedy output.
- Auto Vulkan numerical fixture: logits max_abs 6.20e-5, RMSE 1.08e-5;
  raw hidden relative RMSE 9.92e-6; greedy IDs exact.
- CLI CPU JEV: red rectangle → yes (0.9953311), white → no (0.9994597).
  HTTP Auto JEV: red → yes (0.9953179), white → no (0.9994609). This is
  image-dependence/route verification, not calibration or accuracy evidence.
  Standard padded PNG base64 is accepted, and CLI label scoring no longer
  reaches the visual scorer's former unreachable tokenizer method.

## Engineering checks

- Converter unit tests: 5 passed, covering mapping, unknown architecture,
  coordinate/OBJ contract, temporal split and the unused vocabulary row.
- CPU build and Vulkan build/check succeeded. Scoped model, prompt/media,
  dispatch and parser checks are included in the Rust library suite.
- Final full library suite after rebasing onto `94b4b565`, serial:
  1296 passed / 20 failed / 86 ignored, with no filters.
  Comparison run at original base `92d2eca0`: 1270 passed /
  the same 20 failed / 80 ignored / 14 filtered out. Failure names match
  exactly. Existing failures span Breeze,
  CLM, Z-Image, FunASR, Gemma4, argmax and NEON softmax. This PR does not
  establish a green full repository baseline.
- Real CLI checks retain explicit rejection of unsupported plain-text
  `--dump-logits`, `--bench` and `--profile` rather than silently ignoring them.

## Linux/x86 re-validation — 2026-11-01

Host: Linux x86_64, 22 GiB RAM, Intel(R) Graphics (MTL) Vulkan device,
Rust release-fast profile. Source pins unchanged; `model.safetensors`
(9,661,250,408 B, SHA256 `002c3ac0783af7c86dc28b282c84dd4bcd0c29dcb3b5e0d584492fe3212acb4a`)
and `mask_decoder.pt` (38,534,741 B, SHA256
`8528460481bd09d0780c53559b08194c7abbd29dff0d51783b619158acbe7b46`) were
re-downloaded from ModelScope and verified. All six text exports were
converted; the six GGUF files share `general.architecture=qwen3vl`,
`vocab_size=152680`, `obj_token_id=152678` and
`rope.dimension_sections=[24,20,20,0]`.

Deterministic 128x128 red rectangle, CLI segmentation on 8 threads:

| Text type | Greedy IDs vs BF16 | Response | Alpha PNG vs BF16 | Wall / peak RSS |
|---|---|---|---|---|
| F32 | same | same | identical | 5m26s / 21.7 GB |
| BF16 | reference | `<ref>the red rectangle</ref><box><258><170><773><799><OBJ></box>` | reference | 5m24s / 13.0 GB |
| F16 | same | same | max 1 LSB on 4/16384 px | 5m13s / 11.0 GB |
| Q8_0 | same | same | max 4 LSB on 349/16384 px | 4m59s / 7.0 GB |
| Q6_K | differs by 1 coordinate token (`<799>`→`<798>`) | same box | max 34 LSB on 612/16384 px | 15m42s / 9.5 GB |
| Q4_K | differs by 2 coordinate tokens (`<773><799>`→`<777><798>`) | same box | max 38 LSB on 776/16384 px | 9m57s / 8.5 GB |

F32 needs ~21.7 GB resident, so it does not fit a 16 GiB machine.

Device and route coverage on the same host:

- `--compute auto` (Vulkan build) matched the CPU Q8_0 run exactly: same
  greedy IDs, same box and a byte-identical alpha PNG, in 3m15s with 3582
  device submissions (`scope=linear backend=Vulkan`).
- HTTP `/v1/wevisone/predict` returned 200 with the same response, IDs and
  a byte-identical alpha PNG; an invalid task returned 400.
- CLI and HTTP image JEV agreed on the answer: red rectangle -> yes
  (0.9969 CLI, 0.9987 HTTP), white -> no (0.0063 CLI, 0.0072 HTTP).
- All nine task prompts ran on Q8_0 (detection, ocr, ocr_grounding,
  count_box, count_point, point, referring, freeform, matting); only
  matting and segmentation emitted `<OBJ>` and therefore an alpha PNG.
- `--compute vulkan` is not rejected by validation: it prints one
  `compute requested=Vulkan backend=partial arch=qwen3vl` warning and
  then stops on the runtime capability gate (`Qwen3 Vulkan currently
  requires F16 KV`; with `--kv-cache f16` it reports `Qwen3 configuration
  is unsupported for full-model Vulkan`).
- `cargo test --locked --lib`: 1299 passed / 17 failed / 88 ignored. The 17
  failures are the pre-existing ones (Breeze, CLM, DOTS, Gemma4, Qwen-Drive,
  YUE2, argmax, dot BF16, NEON RoPE, vision RoPE, sparse GGUF segment);
  the WeVisOne branch adds 8 passing tests and no new failures.

### M-RoPE correction

The branch's interleaved M-RoPE fix (`rope_mrope_interleaved` now maps the
last three frequency slots back to the T axis) plus HF-exact text positions
`[next, next, next, 0]` are the actual repair for the "empty output at
>=256x256" symptom recorded in `docs/usage/qwen3.md` §3. Re-verified with
Qwen3-VL-4B-Instruct Q4_K_M (two overlapping yellow circles on light
cyan-blue): the current implementation describes 128/256/512 correctly,
while the previous combination (slots 61-63 on the E axis plus text
positions `[next,0,0,0]`) still truncates the 256x256 answer to 400 bytes.
`rope_mrope_interleaved` now matches HF `apply_interleaved_mrope` slot by
slot, and `build_qwen3_media_positions` matches HF `get_rope_index`, whose
`current_pos` advances by `max(grid_h, grid_w)//merge` after an image.

### CLI dispatch fixes found while re-validating

Two CLI defects surfaced on the same host and are fixed on this branch:

- `--wevisone-decoder` without `--wevisone-request` was silently ignored
  and the run died later with an unrelated error. It now fails with
  `--wevisone-decoder requires --wevisone-request`.
- A bare `--mmproj` with a text-only prompt captured the multimodal branch,
  which requires media; the request then fell through to the hybrid trunk
  and failed with `Failed to parse hybrid text model: Unsupported hybrid
  architecture: Some("qwen3vl")`. Qwen3-VL now takes the text path when no
  image/video/audio is given. The Qwen3-VL text path always runs with F16
  KV, so the `--kv-cache f32` guard now only fires when the user asked for
  F32 explicitly; before, it also rejected the CLI default and blocked
  plain-text Qwen3-VL entirely.

Re-verified after the fix: Qwen3-VL-4B-Instruct text with and without
`--mmproj` generates text instead of erroring, WeVisOne text with
`--mmproj` answers `OK`, the Qwen3-VL image path still describes the
256x256 fixture, and the WeVisOne segmentation run reproduces the
pre-fix token IDs and alpha PNG byte for byte.

Note on plain-text Qwen3-VL: the greedy answer is not stable across runs
(same arguments produced "I'm sorry, but I can't", "Wait, I need to reply
with exactly" and an empty answer). The shared Qwen3-VL text path always
uses F16 KV, so `--kv-cache` does not cause it; multi-threaded float
reductions move the first-token logits, and `docs/usage/qwen3.md` §3
already records the near-tie between `<|im_end|>` and real text as the
cause. This is pre-existing and unrelated to the changes above.

## Coverage limits

Only the stated deterministic image and decoder fixtures were compared;
broad detection/OCR/matting quality, long context, video, multi-image/batch,
discrete GPUs, and sustained server load were not validated.
The Linux/x86 re-validation above added the six text exports end to end and
the CLI/HTTP/JEV routes on one integrated GPU, but it did not widen the
image set beyond the deterministic 128x128 fixture (plus 256x256/512x512
re-runs of the same red rectangle). The pure Python K-quant converter can be
slow. No new external acceleration libraries were introduced.
The CPU 1024² decoder is expensive (the full backbone/mask fixture took
204s while other validation jobs ran); no performance improvement is claimed.

## Generated GGUF SHA256

```text
7ec2b9e896685048831c9bfda2747dbb81a79944b10a28b8e43e07403d1c4ab6  models/WeVisOne-VL-GGUF/mmproj-WeVisOne-VL-4B-F32.gguf
36c1e7509488ba5eb902cdce2881755b90c8f254d7ea547a8027343d3f5da031  models/WeVisOne-VL-GGUF/WeVisOne-VL-4B-bf16.gguf
ac3a684bf6f18e323f7982387a3fa83f5135f152ee83d43e2845134c5b1acc7c  models/WeVisOne-VL-GGUF/WeVisOne-VL-4B-f16.gguf
f1f93f1bb57ac81ad2ec600f4e7e7aed66270d6b9f4296f5ca072063abeb94f2  models/WeVisOne-VL-GGUF/WeVisOne-VL-4B-q8_0.gguf
f3136e06b979e81a339e024912c262a2277fa788c4462c2783a8ee7f8c580818  models/WeVisOne-VL-GGUF/WeVisOne-VL-decoder-F32.gguf
```

The Linux/x86 re-validation reproduced those five digests byte for byte
from the same pinned sources, so the converter is deterministic across the
two hosts. The Q6_K and Q4_K exports were not hashed here because their
digests are not recorded above.
