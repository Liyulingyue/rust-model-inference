# FastFrames-S oracle validation — 2026-11-23

FF-S (`fastframes_s`, the `fastframes_l` variant of the 1.3B Causal-Forcing++
recipe) reuses the FF-L oracle facility: pinned source files, streaming
weights, an independent GGUF effective-weight reader and the same comparator.
The S reference script is `run_reference.py` in this directory; the shared
wrappers live in `../fastframes_l/`.

Host: aarch64 Linux, NVIDIA GPU (Vulkan), 121 GiB RAM, 20-thread cgroup.
FFmpeg ships from `target/fastframes/` (no system install). All weights stay
inside the repository tree.

## Oracle construction and the FastOps decision

The published FF-S recipe is `noise="sequential"`, fractional timesteps
(937.5 / 833.3333129882812), no autocast, no fast ops and an F32 decoder with
no bridge. The **controlled** recipe mirrors RMI's documented compute recipe
instead — GEMM patch embedding, fused QKV, F32 real RoPE — exactly as FF-L's
oracle does; the **published** recipe keeps the reference's own no-fast paths
(Conv3d patch, per-projection QKV, float64-complex RoPE). This is a
numerics-alignment choice, not a claim about the official release path.

Two wrapper bugs were found and fixed while establishing the S oracle; both
are shared with FF-L and neither changes FF-L behavior:

* `StreamingWeights.value()` and `effective_weights.bind()` now fall back to
  the `decoder.pixel_head.` prefix for FF-S's plain decoder keys (FF-L's
  bridged decoder keys hit the direct name, so the fallback is inert for L).
* `attach_blocks()` accepts `fast=None` so the shared streamer can serve
  recipes without fast ops.

The first controlled run also revealed a calibration mistake of our own, kept
here as the root-cause record: comparing RMI's BF16 GGUF against an oracle
loaded from the F32 source `.pt` produced `fastframes.patch` relative RMSE
`1.6e-3` — exactly the BF16 weight-rounding scale — and 29/321 failing
checkpoints concentrated in the refiner and decoder. The fix was to run the
controlled oracle in its GGUF mode (`--gguf models/FastFrames-S-bf16.gguf`),
dequantizing the same effective weights RMI reads. Thresholds were never
relaxed.

Sequential shared-noise layout: value count = `count + (2 + frames) * frame_count`
with `count = latent_frames*16*(h/8)*(w/8)` and `frame_count = 16*(h/8)*(w/8)`;
frame 0 consumes three single-frame transition blocks, each later frame one.

## CPU / Vulkan / CLI / HTTP acceptance (16×16, 5 frames, seed 17)

| Case | Checkpoints | Worst relative RMSE | Min cosine | Result |
|---|---|---|---|---|
| BF16 CPU CLI (`--noise` shared) | 321/321 | 7.05e-06 | 0.999999999975 | pass |
| BF16 strict Vulkan CLI (`--compute vulkan`) | 321/321 | 6.67e-06 | 0.999999999978 | pass |
| F32 CPU CLI (`--noise` shared) | 321/321 | 7.02e-06 | 0.999999999976 | pass |

Noise tensors match bit-for-bit (`fastframes.noise` relative RMSE 0); token IDs
match exactly. Strict Vulkan recorded 1,690 `backend=Vulkan` projection events
and 0 `backend=Cpu` events under `RMI_COMPUTE_TRACE=1`, with cumulative device
submissions/static uploads in `target/fastframes/s-vulkan-proof.log`; CPU and
Vulkan decoded videos are pixel-identical (0/3,840 bytes differ).

CLI↔HTTP: `POST /v1/videos/generations` returns `200 video/mp4` and the MP4 is
byte-identical to the CLI output. Error paths verified: 400 for non-multiple-of-16
sizes and frames>81, 422 for unknown/missing JSON fields, 429 while a
generation holds the single slot. A concurrent request completing after a 429
still reproduces the byte-identical MP4.

## Per-format CPU matrix

All nine formats were converted (`models/FastFrames-S-*.gguf`), each validated
against its own GGUF-mode controlled oracle, RMI CLI parity trace and decoded
video. Thresholds were never relaxed; the F32/BF16 video column compares
against the F32 run's decoded pixels (BF16 itself is the RMI-side base of the
comparison chain).

| Format | Checkpoint failures | Worst relative RMSE | Min cosine | Checkpoint gate | Decoded-video RMSE / cosine (normalized) | Video gate |
|---|---|---|---|---|---|---|
| bf16 | 0/321 | 7.050e-06 | 0.999999999975 | PASS | 0 / 1（对 F32 解码像素零差异） | PASS |
| f32 | 0/321 | 7.020e-06 | 0.999999999976 | PASS | 0 / 1（基准） | PASS |
| f16 | 0/321 | 2.003e-03 | 0.9999980 | PASS | 0.01594 / 0.9997442 | FAIL |
| q8_0 | 290/321 | 7.150e-02 | 0.9975186 | FAIL | 0.02871 / 0.9991610 | FAIL |
| q4_0 | 295/321 | 6.181e-02 | 0.9980892 | FAIL | 0.08914 / 0.9932836 | FAIL |
| q4_1 | 289/321 | 7.095e-02 | 0.9974836 | FAIL | 0.08113 / 0.9932548 | FAIL |
| q4_k | 298/321 | 1.070e-01 | 0.9943015 | FAIL | 0.07406 / 0.9944404 | FAIL |
| q5_k | 296/321 | 1.077e-01 | 0.9941810 | FAIL | 0.05574 / 0.9970998 | FAIL |
| q6_k | 296/321 | 1.098e-01 | 0.9939559 | FAIL | 0.03620 / 0.9986936 | FAIL |

The failure pattern matches FF-L: F16 passes every checkpoint (worst 2.0e-03,
within 0.005) but H.264 amplifies the small input differences past the
decoded-video gate; the six block-quantized formats fail both numeric and video
gates, first failing inside the umT5 text tower (`fastframes.text.*`). Raw
report: `target/fastframes/formats-cpu.json`.

## Auto dispatch and published recipe

`--compute auto` passes 321/321 checkpoints (worst 6.81e-06) with recorded
mixed execution: 128 `backend=Vulkan` and 1,562 `backend=Cpu` projection events
in `target/fastframes/s-auto-proof.log`.

The published FF-S recipe (no-fast paths, BF16 weights, own autocast-free
arithmetic) was run in GGUF mode with the same shared noise: controlled-vs-
published raw-RGB gap inside the official forwards is RMSE 1.06 / cosine
0.99998384 (max abs 5/255); re-encoding the published raw RGB through the same
ffmpeg command and comparing decoded pixels to RMI's MP4 gives normalized RMSE
0.01776 / cosine 0.9996778, within the separate published-recipe drift limit
0.08. RMI's own video is bit-identical between the no-`--noise` and shared-noise
BF16 runs (same-seed RNG draws match bitwise), but acceptance never relies on
that.

## Performance

Same-host diagnostic sampling (16×16 / 5 frames, one warmup + three measured
fresh processes, 20-thread cgroup, GPU shared): CPU median **4.08 s**
(4.06/4.08/4.10), strict Vulkan median **27.40 s** (27.90/26.83/27.40). At this
diagnostic size the device path pays per-projection upload/download on tiny
matrices; larger resolutions remain unmeasured and no isolated-performance
claim is made. Details in [performance.json](performance.json).

## Rebase note

After the shared-kernel commits #269 (AVX-512 zmm matmul) and #270 (Vulkan
Q2_K/Q3_K shaders) landed on `main`, the branch was rebased onto `3a1e3b4`
and the numerical gates re-run on that base: CPU 321/321 (worst `7.05e-06`)
and strict Vulkan 321/321 (worst `6.67e-06`, 1,690 `backend=Vulkan` / 0
`backend=Cpu` events) — identical to the pre-rebase numbers, now proven rather
than assumed on the merged kernel state.

## Reproduction

```bash
# controlled reference reading the GGUF's effective weights
.venv/bin/python -m tools.oracle.fastframes_s.run_reference \
  --reference target/fastframes/reference \
  --model-dir target/fastframes/checkpoints \
  --text-dir models/Wan2.1-T2V-14B \
  --output target/fastframes/s-oracle-bf16 \
  --gguf models/FastFrames-S-bf16.gguf \
  --width 16 --height 16 --frames 5 --seed 17 --threads 24 \
  --recipe controlled --device cpu
# RMI side with shared noise and the parity trace
RMI_PARITY_TRACE=target/fastframes/s-parity-cpu.jsonl \
PATH="$PWD/target/fastframes:$PATH" \
target/release-fast/rust-model-inference --model models/FastFrames-S-bf16.gguf \
  --prompt 'A red ball rolls on a table.' --width 16 --height 16 \
  --video-frames 5 --fps 16 --seed 17 --threads 16 --compute cpu \
  --noise target/fastframes/s-oracle-bf16/noise.f32 \
  --out target/fastframes/s-cli-cpu-shared.mp4
.venv/bin/python -m tools.oracle.fastframes_l.compare \
  --trace target/fastframes/s-parity-cpu.jsonl \
  --reference target/fastframes/s-oracle-bf16 \
  --output target/fastframes/s-compare-cpu.json
```

Build for traces with `cargo build --profile release-fast --bin rust-model-inference
--features parity-trace` (add `vulkan` for `--compute vulkan`). `fastframes.pixels`
is excluded by the comparator and noise must match bitwise; every other
checkpoint uses `relative_rmse <= 0.005` and `cosine >= 0.9999`.
