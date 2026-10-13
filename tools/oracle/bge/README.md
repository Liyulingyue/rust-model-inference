# BGE scalar parity

This directory compares converted BGE GGUFs against a fixed
`llama.cpp` Oracle. It covers tokenizer IDs, the pooled row before L2
normalization, and the final normalized embedding. Use F32 weights, one CPU
thread, F32 KV, and `RMI_SCALAR=1`; Q8_0 and SIMD/FMA paths are inference
smoke checks only and are outside the bit-level claim.

The reference used for the checked-in evidence is `llama.cpp`
`11fe02151f79c41d0d4af7da708755d73b9c0da6` (`11fe02151`). Build it with
external acceleration disabled and FMA disabled, then point `--oracle` at its
`llama-debug` binary. The Rust binary must be built with the `parity-trace`
feature:

The macOS ARM64 Oracle build used the following options with a clean checkout
at that commit:

```bash
bge_scalar_flags='-U__ARM_NEON -U__ARM_NEON__ -U__ARM_FEATURE_FMA -ffp-contract=off -fno-vectorize -fno-slp-vectorize'
cmake -S /tmp/bge-llama-scalar-11fe -B /tmp/bge-llama-scalar-11fe-build \
  -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_NATIVE=OFF \
  -DCMAKE_C_FLAGS="$bge_scalar_flags" -DCMAKE_CXX_FLAGS="$bge_scalar_flags"
cmake --build /tmp/bge-llama-scalar-11fe-build --target llama-debug -j4
```

```bash
cargo build --profile release-fast --features parity-trace --bin rust-model-inference
.venv/bin/python tools/oracle/bge/compare.py \
  --model /tmp/bge-small-en-v1.5-F32.gguf \
  --oracle /tmp/bge-llama-scalar-11fe-build/bin/llama-debug \
  --rust target/release-fast/rust-model-inference \
  --work-dir /tmp/bge-parity
```

`--work-dir` must be a new directory; the checker refuses to overwrite existing
results. These are output-checkpoint comparisons, not a comparison of every
encoder layer's intermediate state.

The default pooling is `cls` for BERT/XLM-R. Use `--pooling last` for decoder
models. The checker supplies F32 KV to both runtimes.

The reference fixture was converted from ModelScope `BAAI/bge-small-en-v1.5`
`model.safetensors` (133,466,304 bytes,
SHA-256 `3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad`).
The resulting F32 GGUF is 133,487,744 bytes with SHA-256
`ca21049b1ec3f76f0e2b53ecf39c72b27e3cd309735c72a1cc74b19a70e89448`.

The fixed four-text check is:

```text
hello
Hello, 世界! 123
What is the capital of China?
中国的首都是北京。
```

This Oracle check is for the BERT encoder path. Decoder-only BGE models use
last-token pooling and the ordinary decoder session; all four architectures'
Q8_0/F32 CLI inference is covered by synthetic converter/runtime fixtures.
Separate scalar F32 checks of those fixtures (256 dimensions, one layer or
two layers for Gemma2, prompt `abca`) match token IDs, pooled and final F32
bits against the same Oracle. They do not establish parity for real decoder
weights or every intermediate layer. Reproduce the synthetic checks with:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from tempfile import mkdtemp
import subprocess
from tools.converter.bge.test_convert_bge import _decoder_files
from tools.converter.bge.convert_bge import convert

root = Path(mkdtemp(prefix="bge-decoder-parity-"))
for kind in ("mistral", "qwen2", "qwen3", "gemma2"):
    model = root / kind
    _decoder_files(model, model_type=kind, hidden=256)
    gguf = root / f"{kind}-F32.gguf"
    convert(model, gguf, "F32")
    subprocess.run([
        ".venv/bin/python", "tools/oracle/bge/compare.py",
        "--model", str(gguf),
        "--oracle", "/tmp/bge-llama-scalar-11fe-build/bin/llama-debug",
        "--rust", "target/release-fast/rust-model-inference",
        "--work-dir", str(root / f"{kind}-parity"),
        "--pooling", "last", "--prompt", "abca",
    ], check=True)
PY
```

Final local validation: 12 BGE converter tests (including eight decoder CLI
format/architecture combinations), 18 existing converter tests for the shared
writer, and nine real BGE-small/M3 integration tests passed. The scalar GeGLU
golden-bit test passes with `RMI_SCALAR=1`; Rustfmt and the CLI build pass.
The serial `parity-trace` library suite has 1,159 passed, 25 failed and 81
ignored. Its failure set exactly matches the unmodified `b6599fd4` baseline
(1,155 passed, 25 failed, 81 ignored); the full suite is not green.
