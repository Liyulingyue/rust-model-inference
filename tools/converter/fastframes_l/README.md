# FastFrames-L converter

Run on Linux. Download directly to remote storage; no model files on the Mac.
Use `models/.venv` for download tools and repository `.venv` for converter dependencies.
Keep package/hub caches inside the remote repository; do not install system packages.

```bash
models/.venv/bin/modelscope download --model apple/FastFrames-L \
  --revision 651fb104ea405268a4c7400b01c1859718090c5d \
  --local_dir models/FastFrames-L \
  fastframes-l/dit.safetensors fastframes-l/kv_refiner.safetensors \
  fastframes-l/pixel_decoder.safetensors
models/.venv/bin/modelscope download --model Wan-AI/Wan2.1-T2V-14B \
  --revision 349065d9c04b2608b85121f4b76f5fcbd03f7dab \
  --local_dir models/Wan2.1-T2V-14B \
  models_t5_umt5-xxl-enc-bf16.pth google/umt5-xxl/tokenizer.json \
  google/umt5-xxl/tokenizer_config.json google/umt5-xxl/special_tokens_map.json \
  google/umt5-xxl/spiece.model
.venv/bin/python -m unittest tools.converter.fastframes_l.test_convert
.venv/bin/python -m tools.converter.fastframes_l.convert_fastframes_l \
  --model-dir models/FastFrames-L --text-dir models/Wan2.1-T2V-14B \
  --output /root/rust-model-inference-fastframes/models/FastFrames-L-bf16.gguf \
  --format bf16
```

Dependencies: `torch`, `numpy`, `safetensors`, and existing shared quantizers.
Formats: `f32 f16 bf16 q8_0 q4_0 q4_1 q4_k q5_k q6_k`.
Source hashes and exact inventories must match `contract.json`. Grouped refiner
heads are split/transposed; patch/pointwise convolutions are flattened. Biases,
norms, relative-bias tables, depthwise kernels and unaligned matrices stay F32.
Matrix encoding streams bounded row chunks through the shared GGUF writer;
sources are mmap-backed. Output publication never replaces an existing file.
A `.gguf.json` sibling records format, size and SHA256.

Four converter tests passed remotely, including relative-bias retention and
parallel/serial encoded-byte equality. K formats use at most 16 shared-quantizer
workers and 16 queued chunks. BF16/F32/F16 full conversions completed; remaining
formats were interrupted before completion could be observed. See
[validation](../../oracle/fastframes_l/validation.md) for hashes and limits.
Use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for converter runs to bound
incidental Python dependency threads; Rust inference uses no external BLAS.
