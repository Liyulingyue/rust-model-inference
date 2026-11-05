# Hy-SimulMT alignment

Latest: [2026-11-04 review fixes](validation.md#review-fixes-2026-11-04), following the [main integration validation](validation.md#main-integration-2026-11-04). Earlier dated measurements remain below.

Targets Tencent's `Hy-SimulMT-30B-A3B-GGUF` and
`Hy-SimulMT-30B-A3B-Interleave-GGUF`, using their bundled `hyv3_speech`
audio tower. The decoder architecture is explicitly `hy_v3`.

The read-only Oracle is
[lihy715/llama.cpp.Hyv3Speech](https://github.com/lihy715/llama.cpp.Hyv3Speech)
at **e41ca4212f786815faa37606ee2e6fcf8611f9e4** (`hyv3-speech`).
`oracle.cpp` uses its public APIs, without editing the reference implementation.
The historical CPU comparisons use four threads and AVX2/FMA/F16C; the local
follow-up uses native M3 Max NEON. Both use no flash attention, greedy decoding,
and the same prompt, KV format and per-turn audio schedule. Hy owns its decoder and KV
session; it shares operators with other models through `ops`, without entering
the Qwen3 trunk. Standard Auto uses Vulkan for eligible linear groups and
convolutions while attention/KV stay on the CPU. Interleave Auto keeps audio
offload but resolves its decoder to CPU because its GPU logits still exceed
the calibrated cosine bound.

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

Build an isolated x86_64 reference checkout with all external acceleration disabled.
Fix the reference ISA to the Rust CPU path rather than using `-march=native`:

```sh
git clone --branch hyv3-speech https://github.com/lihy715/llama.cpp.Hyv3Speech references/llama-hy-simulmt
git -C references/llama-hy-simulmt checkout e41ca4212f786815faa37606ee2e6fcf8611f9e4
cmake -S references/llama-hy-simulmt -B target/llama-hy-simulmt \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DGGML_AVX_VNNI=OFF \
  -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
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

For the M3 Max follow-up, the same pinned Oracle is built with `GGML_NATIVE=ON`
and native NEON rather than the x86 ISA options above. Keep Accelerate, BLAS,
Metal, Vulkan, CUDA and OpenMP disabled. Build `mtmd`, then link the same
`oracle.cpp` companion against that build's libraries. Do not compare an
AVX2 calibration run with a native NEON reference run: the independent
weight-repacking calibration must use the same Oracle build as its reference.

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

The [2026-11-01 merge review and local follow-up](validation.md) record fresh
entry, dispatch, whole-model numerical and process-cleanup checks separately
from the older numerical measurements below.

The [machine-readable record](verification.json) records the independent Hy
decoder measurements below from before the JEV removal in `fdb0f239`. These checks used the
Xeon Gold 6138, four threads, Rust 1.98.1, default SIMD and the pinned AVX2
reference. Speech translation uses F32 KV; the text companion tests both cache
formats. Results cover the two complete Q4_K_M files and these short fixtures.

| Recorded check (before `fdb0f239`) | Result |
|---|---|
| Tokenizer | 60/60 exact cases for each variant, including protocol literals, parse-special flags, multilingual text and digit/combining-mark boundaries. |
| Identical-input projections | 21/21 per variant pass; maximum absolute error 1.91e-6, relative RMS 2.40e-7. |
| Speech encoder, three turns | 12/12 checkpoints per variant pass; maximum relative RMS 8.82e-4. |
| Non-hop tail / silence | 4/4 checkpoints for each fixture and variant pass; maximum relative RMS 4.92e-4 / 2.08e-4. Audio-only mode also runs with a zero token limit. |
| Standard translation, three turns | All 10 generated IDs match; maximum logit relative RMS 7.29%, minimum cosine 0.99746. Passes the unchanged independent-baseline bounds. |
| Interleave translation, three turns | All 18 generated IDs match; maximum logit relative RMS 15.40%, minimum cosine 0.98928. Passes the unchanged independent-baseline bounds. |
| Standard native text, F32 / F16 KV | Greedy ID 32 matches; relative RMS 8.26% / 8.10%. Both pass the bounded check. |
| Interleave native text, F32 / F16 KV | Greedy ID 31168 matches; **fails** the 20% / 0.98 hard bounds: relative RMS 28.69% / 22.96%, cosine 0.96412 / 0.97466. |
| Standard / Interleave Auto, three turns | All 10 / 18 generated IDs match. Standard passes (6.64% relative RMS); Interleave **fails** (25.84% relative RMS, cosine 0.97505). Both processes exit normally. |
| Historical CLI / HTTP / JEV | Three real-weight entry tests per variant passed, covering both upload formats, concurrency admission, the former single/grouped JEV paths and the final-EOT context boundary. Main CPU CLI text equaled the library output: 41.4 s / 42.4 s for three turns. JEV was subsequently removed; these measurements do not establish current entry support. |

Interleave's independent Oracle paths differ by 39.56% / 25.15% relative RMS on
that text fixture. The hard caps remain unchanged, so this does not establish
text-score numerical parity. Native text mode is an internal decoder diagnostic;
Hy JEV is unsupported. No whole-model F32 equivalence is claimed.

The historical Auto runs used real Vulkan submissions and CPU kernels: 1004 / 1012 submissions for
the two three-turn runs, with peak allocations of 2,197,545,472 bytes. Both
one-token main CLI Auto checks exit normally. Its measured functional runs
are slower than CPU; they establish no GPU speedup. A focused Vulkan operator
regression finished its assertions and then exited with SIGSEGV on three
isolated reruns using NVIDIA 570.153.02. GDB caught a Vulkan background thread
while the main thread was in NVIDIA's process-exit callback. The root cause and
a fix were unconfirmed at that time. The local follow-up fixes missing shared
resource cleanup and validation-feature pointer lifetimes, and passes repeated
M3 Max child-process exit checks. It does not establish the cause or resolution
of the NVIDIA crash. The local follow-up also corrects Hy MoE's SiLU path;
Interleave Auto uses the CPU decoder with GPU audio encoding. Fresh complete
weight comparisons pass, while the unguarded GPU decoder's calibrated cosine
failure remains recorded in `verification.json`.

The original 11-second JFK CLI run belongs to revision `a2db408b`, before the
independent decoder. It is retained as historical evidence, not a current
whole-recording or Oracle comparison. Current entry and device results are
recorded separately in `verification.json`.

## Scope

Speech-to-text translation via the library, main CLI and HTTP. Single/grouped
JEV is unsupported. The first turn is 1.6 s and later turns are 0.8 s; each turn
is encoded separately and EOT is retained in the decoder KV when another audio
turn follows. The final EOT needs no unused KV slot. The output-token limit is global across the file.
Interleave emits source and translation, separated by the reserved target marker.
Standard Auto uses the shared CPU/GPU operator policy, including the existing
minimum output-size threshold for linear groups; attention/KV remain on CPU.
Interleave Auto keeps audio offload and uses the CPU decoder, with an explicit
startup reason, until GPU decoder logits pass the unchanged Oracle bounds. Forced full
Vulkan decoding is rejected. Microphone capture, TTS, FP8/other quantizations and
long recordings need their own integration or validation.

## Decoder and entry regressions

The synthetic HySession tests cover router-selected dense blocks, routed/shared MoE,
GQA, external embeddings, invalid-input recovery, CPU/Auto fallback, and both
F32/F16 caches. Each cache format compares 129-token prefill with token-by-token
execution, several batch sizes, and continuation. ARM Q6_K additionally compares
NEON DotProd to the scalar reference across block boundaries and signed scales.
F32/F16 attention use shared dot operators with the reference's AVX accumulator
layout; cancellation and vector-tail tests cover the reduction kernels.
`has_avx2_fma()` and `has_neon()` already honor scalar mode; a direct regression
also checks both ggml dot entry points against the scalar accumulation.

```sh
cargo test --profile release-fast --lib hy_
RMI_SCALAR=1 cargo test --profile release-fast --lib hy_
cargo test --profile release-fast --test hy_simulmt --test hy_simulmt_entries
RMI_HY_SIMULMT_MODEL="$model" RMI_HY_SIMULMT_WAV=artifacts/jfk-3.2s.wav \
  cargo test --profile release-fast --test hy_simulmt_entries -- --ignored --test-threads=1
```

Hy's output projection now has a separate one-row executor, so the vocabulary
width does not inflate the 128-row decoder arena. Descriptor ranges are checked
against the device limit before allocation/upload; Auto can fall back on an
unsupported size. Group descriptors are bounded and recycled only after device
completion, while weight uploads remain cached. F32 groups retain each single
projection's kernel, including groups mixing widths above and below 4096.

Loader compatibility follows the pinned reference: each layer's router tensor
selects dense/MoE; missing `output.weight` ties to the token embedding; optional
gating/norm/scale/shared-width metadata use reference defaults. Expert width
accepts a scalar or layer array (the pin uses the first array element for all
blocks); zero selects `feed_forward_length / expert_used_count`. The expert
width key itself and RMS epsilon remain required. RoPE's missing-key default is
10000. Unsupported non-sigmoid routers still fail explicitly.

`hy_mt2_token_ids_match_official_tokenizer` uses a small vocabulary/merge subset
of Tencent-Hunyuan/Hy-MT2-1.8B, preserving official IDs and merge ranks. The
fixture records its source SHA256 and seven `tokenizers 0.23.2` reference cases;
it does not require model downloads in CI. GPU group regressions run explicitly
on Lavapipe in CI and on the local physical device during this review.

```sh
cargo test --locked --profile release-fast --lib hy_mt2_
cargo test --locked --profile release-fast --features vulkan --lib \
  batched_linear_device_review_ -- --ignored --test-threads=1
```

F16 KV still converts the attention probability input to F16 before P·V. This
matches ggml's ordinary F16 matmul input conversion; multiplication/accumulation
remain F32 SIMD (with the existing scalar F64 fallback). The independent
[`f16_matmul_probe.cpp`](f16_matmul_probe.cpp) confirms that F16×F32 and explicit
F16×F16 inputs give identical ggml outputs, distinct from unrounded F32×F32.
On the local reference build it can be reproduced with:

```sh
clang++ -std=c++17 tools/oracle/hy_simulmt/f16_matmul_probe.cpp \
  -I references/llama-hy-simulmt/ggml/include \
  -L references/llama-hy-simulmt/build-hy/bin -lggml-cpu -lggml-base \
  -Wl,-rpath,"$PWD/references/llama-hy-simulmt/build-hy/bin" \
  -o target/hy-f16-matmul-probe
target/hy-f16-matmul-probe
```

Run the real-weight Auto regression separately for each complete Hy GGUF.
It compares all three turns with CPU and requires actual GPU submissions.
For Interleave those submissions come from audio encoding. Oracle comparisons
above independently check logits and audio checkpoints:

```sh
RMI_HY_SIMULMT_MODEL="$model" RMI_HY_SIMULMT_WAV=artifacts/jfk-3.2s.wav \
  cargo test --locked --profile release-fast --features vulkan --test hy_simulmt_entries \
  real_weights::hy_auto_offloads_and_matches_cpu -- --exact --ignored
```

The ignored tests cover real CPU/Auto outputs and GPU submissions,
the real HTTP binary's multipart/JSON uploads with a
target language, concurrent audio admission and invalid request options, plus
the library's final EOT in an exactly-sized context. The non-ignored entry tests
assert that single/grouped JEV is rejected. Run complete models sequentially on
a shared host. Main CLI audio smoke checks must be run separately.

The companions also accept `text` mode with `text.txt` as a native no-think
prompt. It records tokenizer IDs, final logits and the greedy token without
initializing the audio tower. `HY_KV=f16` selects F16 for this reference mode
and the Rust companion; the default is F32. `HY_COMPUTE=auto` selects the Rust
hybrid path (requires `--features parity-trace,vulkan`), and
`RMI_COMPUTE_TRACE=1` records actual backend choices and device counters. Use
a new output directory for each run.

```sh
target/hy-oracle text "$model" tools/oracle/hy_simulmt/text.txt artifacts/hy-text-ref 4 1 0
HY_ORACLE_REPACK=1 target/hy-oracle text "$model" tools/oracle/hy_simulmt/text.txt artifacts/hy-text-repack 4 1 0
mkdir -p artifacts/hy-text-rust
RMI_PARITY_TRACE="$PWD/artifacts/hy-text-rust/trace.jsonl" RMI_PARITY_FILTER=hy.logits \
  target/release-fast/examples/hy_simulmt_parity text "$model" tools/oracle/hy_simulmt/text.txt artifacts/hy-text-rust 4 1 0
python tools/oracle/hy_simulmt/compare.py artifacts/hy-text-ref artifacts/hy-text-rust \
  --lossy-baseline artifacts/hy-text-repack
```

The verification record separates the current independent decoder checks from
historical whole-recording and tokenizer/projection measurements. GPU offload is
a functional hybrid path, not a performance guarantee.
