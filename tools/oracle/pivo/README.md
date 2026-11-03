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
Only the F32 export is supported; other tensor types and missing components fail
at load time. No `--mmproj` is used.
Retained upstream attribution and licenses are in [licenses/](licenses/).

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
