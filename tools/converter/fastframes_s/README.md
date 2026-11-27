# FastFrames-S converter

Run on Linux. Download directly to remote storage; no model files on the Mac.
Use `models/.venv` for download tools and repository `.venv` for converter
dependencies. Keep package/hub caches inside the remote repository; do not
install system packages.

FF-S shares FF-L's text side: the umT5 encoder and tokenizer files of
`Wan-AI/Wan2.1-T2V-1.3B` and `Wan2.1-T2V-14B` are byte-identical (SHA256
pinned in `contract_s.json`), so an existing `models/Wan2.1-T2V-14B` snapshot
can be reused with no second download.

```bash
models/.venv/bin/modelscope download --model apple/FastFrames-S \
  --revision aa099efcedf1b65447106bf9004cde75821f7387 \
  --local_dir models/FastFrames-S \
  fastframes-s/kv_refiner.safetensors fastframes-s/pixel_decoder.safetensors \
  SHA256SUMS
# zhuhz22/Causal-Forcing is not mirrored on ModelScope; huggingface.co is
# unreachable from this host, so the frozen CF++ generator comes from
# hf-mirror.com's resolve endpoint at the pinned revision.
curl -L -o models/Causal-Forcing/causal-forcing++/framewise-2step.pt \
  https://hf-mirror.com/zhuhz22/Causal-Forcing/resolve/2f8eb8bb6eeb1238da9d13e5420d342a74d634a6/causal-forcing%2B%2B/framewise-2step.pt
.venv/bin/python -m tools.converter.fastframes_s.convert_fastframes_s \
  --model-dir target/fastframes/checkpoints \
  --text-dir models/Wan2.1-T2V-14B \
  --output models/FastFrames-S-bf16.gguf \
  --format bf16
```

`--model-dir` must contain `causal-forcing++/framewise-2step.pt`,
`fastframes-s/kv_refiner.safetensors` and `fastframes-s/pixel_decoder.safetensors`
(a symlink layout works and keeps the originals in their pinned snapshots;
this repo used `target/fastframes/checkpoints` linking into
`models/FastFrames-S` and `models/Causal-Forcing`).

Dependencies: `torch`, `numpy`, `safetensors`, and existing shared quantizers.
Formats: `f32 f16 bf16 q8_0 q4_0 q4_1 q4_k q5_k q6_k`.
Source hashes and exact inventories must match `contract_s.json`. The CF++
generator is a torch pickle whose tensors sit under `generator_ema` with an
`model._fsdp_wrapped_module.` prefix and F32 storage (the released BF16 note
describes a post-load cast); the converter strips both and records the actual
dtype. The plain pixel decoder's bare keys are normalized with a `pixel_head.`
prefix so the single GGUF namespace stays unambiguous. Grouped refiner heads
are split/transposed; patch/pointwise convolutions are flattened. Biases,
norms, relative-bias tables, depthwise kernels and unaligned matrices stay F32.
Matrix encoding streams bounded row chunks through the shared GGUF writer;
sources are mmap-backed. Output publication never replaces an existing file.
A `.gguf.json` sibling records format, size and SHA256.

Use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for converter runs to bound
incidental Python dependency threads; Rust inference uses no external BLAS.
Numerical validation and per-format results are recorded in
[oracle validation](../../oracle/fastframes_s/validation.md).
