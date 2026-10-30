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

## Coverage limits

Only the stated deterministic image and decoder fixtures were compared;
broad detection/OCR/matting quality, long context, video, multi-image/batch,
Linux/x86, discrete GPUs, and sustained server load were not validated.
Full-weight F32/Q4_K/Q6_K exports were not executed end to end; their conversion
and existing kernel routes are available. The pure Python K-quant converter
can be slow. No new external acceleration libraries were introduced.
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
