# FastFrames-L oracle

**Experimental: BF16/F32 CPU/Auto/Vulkan pass CLI/HTTP checkpoints and
decoded MP4 comparison at 16×16 / 5 frames. F16 passes checkpoints but fails
decoded-video gates. Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K have failed numeric gates. All 54 transports ran (12 passed, 42 failed). The 480p checkpoints pass,
but decoded-video cosine and published-BF16 drift fail their quality gates.** See [validation](validation.md), [matrix](matrix.json),
[observed results](observed-results.json) and [performance](performance.json).

Read-only reference: `apple-aiml-research/ml-fastframes` commit
`6044a56f44bffec500d04f2c21be3c5066891431`. The runner verifies 20 pinned
source/license files, source checkpoints and tokenizer files; records GGUF/wrapper
hashes and dependency versions. Python is test infrastructure only. Modules are
constructed on meta and text/DiT blocks loaded/released on demand, retaining the
pinned official forwards without requiring an ~84 GB anonymous F32 model copy.

## Arithmetic and input

`--recipe controlled` uses F32 activation/KV storage and accumulation with PyTorch
SDPA. `--gguf` independently decodes effective weights with `gguf` and reconstructs
source grouped/convolution layouts. Prepared projections reproduce the native
operator contract: F16 inputs round to F16, Q formats use independent Q8_0/Q8_K
activation QDQ, and Q4_1 retains the separately rounded Q8_1 sum term. These last
format-specific paths have regression checks; per-format full-model results and
failed gates are recorded separately in the matrix.

`--recipe published` retains BF16 casts while substituting an available PyTorch
SDPA backend for NVIDIA-only cuDNN on AMD. Its drift is separate; this is not an
exact reproduction of cuDNN arithmetic. The PyTorch dispatcher selects SDPA's
actual kernel; the runner does not claim to force its math backend.

`thresholds.json` was fixed before results: implementation relative RMSE ≤ 0.005,
cosine ≥ 0.9999. Token IDs/counts and **input noise** match exactly; lossy casts,
SIMD reductions and GPU arithmetic use numeric tolerances. Do not relax thresholds
after a failure. Locate the first checkpoint divergence instead. Storage-quality
loss versus raw source and published-BF16 drift are distinct comparisons.

Controlled default noise is F32; published default noise is BF16-rounded. Same
seed alone does not prove shared input across RNG implementations. Supply `--noise`
for comparison. For HTTP, retain its `fastframes.noise` binary and pass that file to
the oracle, because HTTP accepts a seed rather than a test-only noise override.

## Run on remote Linux

Keep all weights remote. Download the pinned reference into
`target/fastframes/reference`; install dependencies in the repository `.venv`.
Tokenizer auxiliary files are required by `AutoTokenizer`.

```bash
.venv/bin/python -m tools.oracle.fastframes_l.run_reference \
  --reference target/fastframes/reference \
  --model-dir models/FastFrames-L --text-dir models/Wan2.1-T2V-14B \
  --gguf /root/rust-model-inference-fastframes/models/FastFrames-L-bf16.gguf \
  --recipe controlled --device cuda --threads 8 \
  --width 16 --height 16 --frames 5 --seed 17 \
  --output target/fastframes/reference-bf16
```

Build Rust with `--features vulkan,parity-trace`; run the same prompt/geometry
with `--noise target/fastframes/reference-bf16/noise.f32`,
`RMI_PARITY_TRACE=/absolute/path/native-bf16.jsonl` and `RMI_COMPUTE_TRACE=1`.
The repository tracer writes every full F32 binary alongside the JSONL.

```bash
.venv/bin/python -m tools.oracle.fastframes_l.compare \
  --trace target/fastframes/native-bf16.jsonl \
  --reference target/fastframes/reference-bf16 \
  --output target/fastframes/comparison-bf16.json
```

Use `--quantized` for Q formats. Repeat all nine formats × CPU/Auto/strict Vulkan
with real CLI and HTTP independently; preflight rejection is not inference.
Checkpoints cover text, patches, each DiT block, flow/x0, repaired K/V and pixel
tokens. Raw reference pixels are retained but skipped by the comparator because
native traces contain pixel tokens. Independent transform tests cover that layout;
final RGB/MP4 quality remains a separate gate (the final BF16 CPU diagnostic
CLI/HTTP case passes it; the completed 480p case fails decoded cosine and
published-BF16 drift). Benchmark mode has no checkpoints
and cannot be used as parity evidence.

## Timing

`--benchmark` disables checkpoint copies/writes. The timed region still includes
**on-demand block decoding/uploads**, generation and final RGB transfer/write;
initial hashes/eager initialization are excluded. Normal traced runs also include
checkpoint I/O. These boundaries are recorded in each manifest. Do not report
traced timings as compute-only latency or infer a native speedup from them.
Synchronize GPU, warm up and repeat clean native/reference measurements at least
three times on the same host, accounting for the different weight-loading boundary.
Record CPU/GPU split, submissions and peak RSS/VRAM in `performance.json`.
