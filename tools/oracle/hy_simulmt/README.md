# Hy-SimulMT alignment

Targets Tencent's `Hy-SimulMT-30B-A3B-GGUF` and
`Hy-SimulMT-30B-A3B-Interleave-GGUF`, using their bundled `hyv3_speech`
audio tower. The decoder architecture is explicitly `hy_v3`.

The read-only Oracle is
[lihy715/llama.cpp.Hyv3Speech](https://github.com/lihy715/llama.cpp.Hyv3Speech)
at **e41ca4212f786815faa37606ee2e6fcf8611f9e4** (`hyv3-speech`).
`oracle.cpp` uses its public APIs, without editing the reference implementation.
Both engines use CPU, four threads, F32 KV, no flash attention, greedy decoding,
and the same prompt and per-turn audio schedule.

Tokenizer IDs, special-token behavior, checkpoint names/order/shapes/counts and
generated token IDs are checked exactly. Floating point comparisons use numerical
tolerances: ggml's convolution lowers patches to **F16**, while Rust's existing
convolution retains **F32**; quantized projections, SIMD reductions, GELU and
SiLU also round differently. Bitwise equality is not an appropriate requirement
for those operations. `compare.py` checks finite values, relative RMS error,
cosine and maximum absolute error. It does not treat plausible text as parity.

Whole-model Q4 logits need a separate error budget from individual projections.
`--lossy-baseline` compares a **third, independent run of the pinned Oracle**
with CPU weight repacking enabled. The two Oracle paths must generate the same
turns, and all three runs must have the same checkpoint names, shapes and counts.
The logit budget is twice the maximum measured Oracle-path difference, capped
at 20% relative RMS error and a cosine floor of 0.98. Without this explicit
baseline, the checker keeps its strict 0.5% / 0.99998 logit limits. The baseline
cannot be the reference or Rust output directory. This is a bounded comparison
of a lossy quantized model, **not a claim of close F32 whole-model equivalence**.

## Weights

Both files contain all 1170 tensors and are 19,746,809,728 bytes.

| File | SHA256 |
|---|---|
| `Hy-SimulMT-30B-A3B-Q4_K_M.gguf` | `350344d24911ab93fdaee8f9b78e6c06041bff942dc7bc578d8c78d185e4f281` |
| `Hy-SimulMT-30B-A3B-Interleave-Q4_K_M.gguf` | `26c87f0a5f6409f3a57f8c4d2a49d58ad0f5c0753141643789e73564339cedbc` |

Download these individual files from the ModelScope repositories into `models/`.
Use a repo-contained Python environment; no global Python packages are needed.
Q4_K_M contains Q4_K/Q6_K/Q8_0 decoder weights and an F32 audio tower.

## Reproduce

Build an isolated reference checkout with all external acceleration disabled:

```sh
git clone --branch hyv3-speech https://github.com/lihy715/llama.cpp.Hyv3Speech references/llama-hy-simulmt
git -C references/llama-hy-simulmt checkout e41ca4212f786815faa37606ee2e6fcf8611f9e4
cmake -S references/llama-hy-simulmt -B target/llama-hy-simulmt \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
  -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON
a=$(pwd)/target/llama-hy-simulmt/bin
cmake --build target/llama-hy-simulmt --target mtmd llama-mtmd-cli -j4
c++ -std=c++17 -O2 tools/oracle/hy_simulmt/oracle.cpp \
  -I references/llama-hy-simulmt/include -I references/llama-hy-simulmt/ggml/include \
  -I references/llama-hy-simulmt/tools/mtmd -I references/llama-hy-simulmt/vendor \
  -L "$a" -Wl,-rpath,"$a" -lmtmd -lllama -lggml -lggml-base -lggml-cpu -o target/hy-oracle
cargo build --profile release-fast --features parity-trace --example hy_simulmt_parity
```

The speech fixture is the first 51,200 samples (3.2 s) of
[whisper.cpp's public JFK sample](https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav).
The full WAV's SHA256 is
`59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`.
Convert PCM16 to little-endian F32 by dividing by 32768, without resampling.
This covers three turns: 25,600 + 12,800 + 12,800 samples, producing 11 + 5 + 5
audio tokens, including a partial 100-mel-frame grid block and odd FSK padding.

With the downloaded sample saved as `artifacts/jfk.wav`, create the raw inputs:

```sh
python - <<'PY'
from pathlib import Path
import struct, wave
with wave.open('artifacts/jfk.wav') as w:
    assert (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (1, 16000, 2)
    raw = w.readframes(51200)
x = [v / 32768 for v in struct.unpack('<51200h', raw)]
for name, values in [('jfk-3.2s', x), ('jfk-tail', x[:13017]), ('silence', [0.] * 16000)]:
    Path(f'artifacts/{name}.f32').write_bytes(struct.pack(f'<{len(values)}f', *values))
with wave.open('artifacts/jfk-3.2s.wav', 'wb') as w:
    w.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
    w.writeframes(raw)
PY
```

Choose **new, empty output directories** for every run; parity traces append.
The final argument selects the Oracle's Interleave prompt (0/1); Rust reads
`general.name` from the GGUF. The intermediate `audio` mode uses a vocab-only
text model in the Oracle; it does not pretend to validate the decoder.

```sh
model=models/Hy-SimulMT-30B-A3B-GGUF/Hy-SimulMT-30B-A3B-Q4_K_M.gguf
target/hy-oracle tokens "$model" tools/oracle/hy_simulmt/tokenizer.json artifacts/hy-tokens-ref 4 64 0
target/release-fast/examples/hy_simulmt_parity tokens "$model" tools/oracle/hy_simulmt/tokenizer.json artifacts/hy-tokens-rust 4 64 0
python tools/oracle/hy_simulmt/compare.py artifacts/hy-tokens-ref artifacts/hy-tokens-rust

target/hy-oracle translate "$model" artifacts/jfk-3.2s.f32 artifacts/hy-ref 4 64 0
HY_ORACLE_REPACK=1 target/hy-oracle translate "$model" artifacts/jfk-3.2s.f32 artifacts/hy-ref-repacked 4 64 0
mkdir -p artifacts/hy-rust
RMI_PARITY_TRACE="$PWD/artifacts/hy-rust/trace.jsonl" \
RMI_PARITY_FILTER=hy.enc_inp,hy.after_transformer,hy.after_proj2,hy.projected,hy.logits \
  target/release-fast/examples/hy_simulmt_parity translate "$model" artifacts/jfk-3.2s.f32 artifacts/hy-rust 4 64 0
python tools/oracle/hy_simulmt/compare.py artifacts/hy-ref artifacts/hy-rust \
  --lossy-baseline artifacts/hy-ref-repacked
```

Repeat with the Interleave GGUF and final argument `1`. `compare.py` requires
NumPy in an existing or repo-contained environment. It refuses mismatched greedy
tokens before comparing logits: after a divergent token, later logits no longer
share the same input context.

Check the actual quantized projections independently, with **identical F32
inputs** so activation rounding cannot accumulate across layers:

```sh
python tools/oracle/hy_simulmt/make_probes.py artifacts/hy-probes
target/hy-oracle projections "$model" artifacts/hy-probes/projections.json artifacts/hy-projections-ref 4 0 0
mkdir -p artifacts/hy-projections-rust
RMI_PARITY_TRACE="$PWD/artifacts/hy-projections-rust/trace.jsonl" \
  target/release-fast/examples/hy_simulmt_parity projections "$model" artifacts/hy-probes/projections.json artifacts/hy-projections-rust 4 0 0
python tools/oracle/hy_simulmt/compare.py artifacts/hy-projections-ref artifacts/hy-projections-rust
python -m unittest discover -s tools/oracle/hy_simulmt -p test_compare.py
```

These 21 checks cover Q4_K, Q6_K and F32 weights: the first/last MoE blocks,
experts 0/127, shared experts, the router, a query projection and a 4096-wide
attention output. They require max absolute error ≤ 2e-5, relative RMS ≤ 2e-6
and cosine ≥ 1−2e-11. Audio checks require relative RMS ≤ 0.003,
cosine ≥ 0.99999 and max absolute error ≤ `0.002 + 0.001 * max(abs(reference))`.
Repeat `audio` mode for `jfk-tail.f32` and `silence.f32` to cover hop padding
and the mel floor. All values must be finite; no floating point stage uses
raw-bit equality.

## Measured validation

The [machine-readable record](verification.json) fixes the weights, inputs,
reference revision, actual errors and acceptance bounds. Real-weight inference
ran on an existing Linux x86_64 Xeon Gold 6138, four threads, Rust 1.98.1,
default SIMD, F32 KV and no flash attention. These results are experimental
CPU/Q4_K_M coverage, not a guarantee for other quantizations or long recordings.

| Check | Result |
|---|---|
| Tokenizer | All 60 cases match exactly for each variant, including protocol literals, parse-special flags, multilingual text and digit/combining-mark boundaries. |
| Identical-input projections | 21/21 per variant pass; translation-only max absolute error 1.67e-6, max relative RMS 2.40e-7. Both variants use the same strict limits. |
| Speech encoder, three turns | 12/12 checkpoints pass; max relative RMS 5.80e-4. |
| Non-hop tail / silence | 4/4 checkpoints each pass; max relative RMS 4.78e-4 / 2.00e-4. |
| Translation-only, three turns | All 10 generated IDs match; max logit relative RMS 8.02%, min cosine 0.99679. Oracle's own path variation reaches 7.40%. |
| Interleave, three turns | All 18 generated IDs match, including the source/target marker; max logit relative RMS 14.30%, min cosine 0.99060. Oracle's own path variation reaches 15.72%. |
| Main CLI | Both variants process the same three-turn WAV successfully in 37.5 s / 56.9 s on four CPU threads; Interleave emits source text and its translation separator. |
| Additional CLI checks | Translation-only completes the full 11 s JFK clip in 154.2 s: 13 turns, 70 audio tokens, 35 output tokens. A global limit of 4 tokens truncates in turn 2. The full clip is a CLI smoke test, not an additional Oracle parity case. |

The first material decoder difference was traced to a few rows of the initial
dense FFN after Q8_K activation rounding. Before that, all 46 prompt rows have
identical attention-normalized inputs; raw Q/K and attention weighted-value outputs differ by
less than 6.4e-8. Small rounding differences can change quantized activations and
expert choices over 47 MoE blocks. The strict projection tests and independent
Oracle-path calibration distinguish this effect from a layout or routing error.

## Scope

CPU speech-to-text translation via the library and main CLI. The first turn is
1.6 s and later turns are 0.8 s; each turn is encoded separately and EOT is
retained in the decoder KV. The output-token limit is global across the file.
Interleave emits source and translation, separated by the reserved target marker.
This does not add microphone capture, a speech server, TTS, FP8 loading or GPU
execution. Other quantizations and long recordings need their own validation.
