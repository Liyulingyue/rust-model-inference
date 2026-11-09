# PiVo-1.4M

Native English TTS using the pinned [ModelScope release](https://modelscope.cn/models/empero-ai/PiVo-1.4M).
`sources.json` records the exact per-file revision, byte count and SHA256 for
the checkpoints, ONNX graphs, reference Python frontend/runtime and licenses.
Rust inference uses no Python, ONNX runtime or external acceleration library.

## Convert and run

```sh
python3 tools/oracle/pivo/download.py models/PiVo-1.4M
python3 -m venv .venv
.venv/bin/python -m pip install -r tools/oracle/pivo/requirements.txt
.venv/bin/python -m tools.converter.pivo.convert_pivo models/PiVo-1.4M models/PiVo-1.4M/pivo-F32.gguf
cargo build --locked --profile release-fast --bin rust-model-inference --bin rust-model-server
target/release-fast/rust-model-inference --tts --model models/PiVo-1.4M/pivo-F32.gguf \
  --prompt 'NASA is ready for the next mission.' --voice Pi --speed 1 --out pi.wav --threads 2 --compute cpu
target/release-fast/rust-model-inference --tts --model models/PiVo-1.4M/pivo-F32.gguf \
  --prompt 'Hello, Zzvo.' --voice Vo --pronunciations '{"zzvo":["Z","IY1","V","OW0"]}' --out vo.wav
```

One `pivo` GGUF includes 109 F32 tensors (1,401,562 parameters plus 40 feature
normalization buffer values), the CMU dictionary and all upstream license/notice
texts. The converter checks original hashes before loading checkpoints, checks
the exact tensor/config contract, and refuses to overwrite an existing output.
No `--mmproj` is used.
Retained upstream attribution and licenses are in [licenses/](licenses/).

## Weight storage

| Format | Command | File size | Usable |
| --- | --- | --- | --- |
| `pivo-f32-v1` | default | 9,252,704 bytes | yes |
| `pivo-q8_0-v1` | `--quant q8_0` | 6,988,192 bytes | **no — the audio is not usable** |

PiVo always computes in F32; the quantized export only changes how the weights
are stored. GGML quantizes a row at a time and a Q8_0 block holds 32 values, so
only the 52 matrices whose row length is a whole number of blocks are
quantized; the one-dimensional norms, biases and statistics, and the
convolution kernels with a kernel axis of 3 or 5, stay F32.

**Do not use the Q8_0 export.** PiVo's FARGAN vocoder feeds its own output
back into the next step, so the 1/127 weight grid compounds into a carrier phase
drift: the complete waveform ends up 0.82 of full scale away from the official
graph, and up to 23.23% of the pitch periods stop matching. The loader reads
the format, the CLI prints `format=pivo-q8_0-v1` and warns, and the measurement
is in [validation.md](validation.md#q80-export-the-audio-is-not-usable). The F32
export is the one to use.

`Pi` and `Vo` are the two built-in voices. Speed must be finite and in `[0.5,2]`.
The frontend uses NFKC, deterministic normalization and the dictionary's first
pronunciation; unknown words require ARPAbet overrides. Input limits match the
release: 2,000 characters, 512 tokens and 3,000 acoustic frames (30 seconds).
Empty/punctuation-only input, unsupported characters and ambiguous slash dates
are rejected. English only; reference-audio cloning and sampling controls do not
apply. Output is mono, 16 kHz PCM16 WAV, using the repository's WAV encoder.
Unicode decimal digits are normalized like the Python frontend. Conflicting
case-insensitive override keys are rejected to keep requests deterministic.

## HTTP

```sh
target/release-fast/rust-model-server --tts --model models/PiVo-1.4M/pivo-F32.gguf \
  --host 127.0.0.1 --port 8080 --threads 2 --compute auto
curl -f http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"Hello, my name is Vo.","voice":"Vo","speed":1,"response_format":"wav"}' --output vo.wav
```

`model` may be omitted; when provided it must match `/v1/models`.
`response_format` is `wav` (default) or raw little-endian PCM16 `pcm`.
`overrides` is an optional word-to-ARPAbet dictionary. CLI `--voice`, `--speed`
and `--pronunciations` also set server defaults; an HTTP value overrides its
corresponding default. Invalid requests return 400 (schema errors 422); concurrent
synthesis returns 429. JEV/chat/embedding routes are not mounted on this backend.
Numerical inference failures return 500.

CPU and Auto both use existing Rust SIMD operators on the CPU. Strict Vulkan is
rejected; there is no GPU speedup claim. Every utterance creates independent
acoustic/vocoder/DSP state.

Same machine, same weights, measured from a cold process to the finished audio
(Intel Core Ultra 5 125H, x86_64, `release-fast`):

| Case | Rust `--compute cpu` | Official ONNX Runtime, one thread |
| --- | --- | --- |
| "NASA is ready for the next mission." (1.87 s audio) | 192.4 ms | 192.3 ms |
| "The quick brown fox jumps over the lazy dog." (2.57 s audio) | 219.9 ms | 240.2 ms |

RTF is about 0.09–0.10 on both sides. `--threads 1`, `2` and `4` make no
measurable difference: the model is 1.4M parameters, so the per-frame vocoder
loop dominates and the thread pool has nothing to parallelise. The full record
is in [performance.json](performance.json).

## Numerical oracle

```sh
.venv/bin/python tools/oracle/pivo/trace.py models/PiVo-1.4M artifacts/pivo/fixtures.json
PIVO_GGUF=models/PiVo-1.4M/pivo-F32.gguf PIVO_FIXTURES=artifacts/pivo/fixtures.json \
  RAYON_NUM_THREADS=2 cargo test --locked --profile release-fast --lib \
  models::pivo::tests::official_onnx_checkpoints_and_waveform -- --ignored --nocapture
```

Repeat with `RMI_SCALAR=1 RAYON_NUM_THREADS=1` for the scalar operator dispatch,
or `PIVO_AUTO=1` for Auto. `trace.py` wraps the unchanged official ONNX sessions
to record actual encoder outputs, decoder features, conditioning and recurrent
step inputs/outputs. It also records the official complete waveform. IDs and
integer durations must match exactly; floating-point stages use absolute/RMS
error gates, because SIMD, transcendental functions and recurrent feedback lose
precision. WAV serialization can differ from the official NumPy writer by one
PCM16 unit because the shared repository encoder truncates instead of rounding.

Actual measurements, coverage and regression status are in [validation.md](validation.md).

The fixture generator also records a separate official-vocoder sensitivity probe
with each decoder output moved one F32 ULP toward positive infinity. This does not
replace the unmodified reference fixtures or the tighter teacher-forced checkpoints.

After building both binaries, exercise the real interfaces with:

```sh
.venv/bin/python tools/oracle/pivo/check_interfaces.py models/PiVo-1.4M/pivo-F32.gguf artifacts/pivo/fixtures.json
```
