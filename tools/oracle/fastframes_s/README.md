# FastFrames-S oracle

**Experimental / partial: BF16 and F32 pass CPU, strict Vulkan and auto
checkpoints at 16×16 / 5 frames (321/321 each, worst relative RMSE `7.05e-06`),
CLI↔HTTP MP4 byte-identical, error paths 400/422/429 verified. F16 passes
checkpoints but fails decoded-video gates. Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K fail
numeric gates.** See [validation](validation.md) and
[performance](performance.json).

The S runner (`run_reference.py` in this directory) reuses the FF-L oracle
facility: pinned source verification, meta-constructed modules, streamed
text/DiT blocks and the independent GGUF effective-weight reader
(`../fastframes_l/streaming_weights.py`), with the S-specific differences —
the frozen CF++ generator pickle (`generator_ema`, F32 storage), sequential
shared-noise layout (`count + (2 + frames) * frame_count` values), fractional
timesteps and the plain (bridge-less) pixel decoder whose GGUF keys carry the
`pixel_head.` prefix.

Read-only reference: `apple-aiml-research/ml-fastframes` commit
`6044a56f44bffec500d04f2c21be3c5066891431` — the same pin FF-L uses. The
runner verifies the pinned oracle source files, the S/CF++ checkpoints, the
reused 14B tokenizer assets (byte-identical to the 1.3B pins) and the GGUF
hash; records wrapper hashes and dependency versions.

`--recipe controlled` mirrors RMI's documented compute recipe (GEMM patch
embedding, fused QKV, F32 real RoPE) exactly as FF-L's oracle does;
`--recipe published` keeps the reference's own no-fast paths. `--gguf` makes the
oracle read the same effective weights RMI reads — the acceptance runs use this
mode. Python is test infrastructure only.

## Commands

```bash
.venv/bin/python -m tools.oracle.fastframes_s.run_reference \
  --reference target/fastframes/reference \
  --model-dir target/fastframes/checkpoints \
  --text-dir models/Wan2.1-T2V-14B \
  --output target/fastframes/s-oracle-bf16 \
  --gguf models/FastFrames-S-bf16.gguf \
  --width 16 --height 16 --frames 5 --seed 17 --threads 24 \
  --recipe controlled --device cpu
.venv/bin/python -m tools.oracle.fastframes_l.compare \
  --trace target/fastframes/s-parity-cpu.jsonl \
  --reference target/fastframes/s-oracle-bf16 \
  --output target/fastframes/s-compare-cpu.json
```

The comparator, thresholds and streaming wrappers are shared with FF-L; the
shared files gained `fast=None` compatibility and the decoder `pixel_head.`
prefix fallback, both covered by `../fastframes_l/test_streaming.py` (10 tests)
and a 412-checkpoint L binary-identity rerun (zero regressions).
