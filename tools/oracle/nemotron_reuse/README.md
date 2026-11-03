# NVIDIA NemotronLabs RE-USE

Native Rust speech enhancement for the published SEMamba checkpoint. **Experimental:** the network and inverse STFT agree numerically on identical spectral inputs, but independent FFT rounding can cross the raw phase input's ±π branch cut and produce material end-to-end waveform differences. This is not a claim of strict waveform equivalence or measured enhancement quality. See [measured results](validation.md).

## Contract

- [ModelScope source](https://modelscope.cn/models/nv-community/NVIDIA-NemotronLabs-RE-USE/files), revision `d66af5dfc20db53a49f885eb32e7e7f1fcb70af2`.
- `model.safetensors`: 38,592,940 bytes, SHA256 `06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1`; 1,416 F32 tensors. The loader checks every name, shape, dtype, storage length and finite value against the fixed architecture.
- Published `config.json`: SHA256 `e69986483774f01893d2727503addc49ef201d19e6e1a266b8d4e971e28c478d`; 30 bidirectional time/frequency Mamba blocks, hidden 64, inner 256, state 16, convolution 4. The published `model_cfg` and `stft_cfg` must match.
- Original F32 safetensors only. No GGUF architecture/converter, tokenizer or additional component is involved; other dtypes are rejected.
- PCM16 WAV, integer sample rates 8,000–48,000 Hz, 1–8 channels processed independently. More than half an FFT window (approximately 20 ms), at most 10 minutes. Sample rate, channel count and frame count are preserved. HTTP additionally caps the body at 64 MiB.
- Rate-invariant STFT: base FFT/window 320 and hop 40 at 8 kHz, scaled then rounded up to an even size. Periodic Hann window and centered reflection padding. Inputs longer than 2 seconds use fixed 2-second chunks with 50% Hann overlap and the original shifted final-chunk schedule. The inverse STFT's incomplete final hop is padded with `1e-8`, matching the reference.
- `--compute cpu` uses existing SIMD dot products and CPU parallelism. `auto` uses the same CPU implementation; strict `vulkan` is rejected. JEV and text generation do not apply to this audio-to-audio model.

## Run

Download into the ignored model directory with the repository-local downloader:

```sh
models/.venv/bin/modelscope download nv-community/NVIDIA-NemotronLabs-RE-USE \
  model.safetensors config.json models/generator_SEMamba_time_d4.py \
  models/codec_module_time_d4.py models/mamba_block2_SEMamba.py models/stfts.py \
  noisy_audio/mic_test2.wav \
  --revision d66af5dfc20db53a49f885eb32e7e7f1fcb70af2 \
  --local-dir models/NVIDIA-NemotronLabs-RE-USE

cargo +1.98.1 build --locked --profile release-fast \
  --bin rust-model-inference --bin rust-model-server

target/release-fast/rust-model-inference --enhance \
  --model models/NVIDIA-NemotronLabs-RE-USE --audio input.wav --out enhanced.wav \
  --threads 4 --compute cpu

target/release-fast/rust-model-server --enhance \
  --model models/NVIDIA-NemotronLabs-RE-USE --host 127.0.0.1 --port 8080 \
  --threads 4 --compute auto

curl --fail -H 'Content-Type: audio/wav' --data-binary @input.wav \
  http://127.0.0.1:8080/v1/audio/enhancements --output enhanced-http.wav
```

`--model` also accepts the `model.safetensors` file beside `config.json`. CLI output uses the existing atomic writer; an existing output requires `--overwrite`. Do not combine `--enhance` with generation, ASR/TTS, embedding, JEV or another task's payload options. The server accepts audio in each request rather than startup `--audio`/`--out`.

The HTTP endpoint returns a binary `audio/wav` body. Invalid media/audio returns 400, another active enhancement returns 429, and an oversized body returns 413. Chat and text/image JEV routes are absent (404).

## Reproduce numerical checks

Use a repository-local `.venv`. The recorded oracle environment is Python 3.13, Torch 2.9.1, NumPy 2.5.4, einops 0.8.2, safetensors 0.8.0 and huggingface_hub 2.4.0. These packages are oracle dependencies, not Rust runtime dependencies.

`reference.py` verifies source/model/sample hashes. It loads the original SEMamba and original [Mamba v2.2.6](https://github.com/state-spaces/mamba/tree/d7b1ceb3c367ec9022925e812f507bcf706937c6) at `d7b1ceb3c367ec9022925e812f507bcf706937c6`. The original `selective_scan_ref` function is compiled verbatim from its AST to avoid importing CUDA extensions. No handwritten replacement network or scan is used. Mamba source files are downloaded from the pinned GitHub commit on first use and checked by SHA256.

```sh
.venv/bin/python tools/oracle/nemotron_reuse/reference.py \
  --model-dir models/NVIDIA-NemotronLabs-RE-USE \
  --output models/NVIDIA-NemotronLabs-RE-USE/oracle-pcm

# Normal build: measure independent end-to-end drift, verify CPU/Auto WAV equality.
.venv/bin/python tools/oracle/nemotron_reuse/parity.py \
  --model-dir models/NVIDIA-NemotronLabs-RE-USE \
  --oracle models/NVIDIA-NemotronLabs-RE-USE/oracle-pcm \
  --output models/NVIDIA-NemotronLabs-RE-USE/parity-default \
  --waveform-only --mode cpu --mode auto

# Production network/ISTFT supplied the official spectral features.
RMI_REUSE_MODEL=models/NVIDIA-NemotronLabs-RE-USE \
RMI_REUSE_ORACLE=models/NVIDIA-NemotronLabs-RE-USE/oracle-pcm \
cargo +1.98.1 test --locked --profile release-fast --lib \
  official_spectral_inference_matches_with_numerical_tolerances -- --ignored --nocapture

# HTTP checks launch and stop their own loopback servers (CPU and Auto).
.venv/bin/python tools/oracle/nemotron_reuse/http_smoke.py \
  --model-dir models/NVIDIA-NemotronLabs-RE-USE \
  --oracle models/NVIDIA-NemotronLabs-RE-USE/parity-default \
  --output models/NVIDIA-NemotronLabs-RE-USE/http-default

# Diagnostic checkpoints: this feature changes shared LayerNorm rounding.
cargo +1.98.1 build --locked --profile release-fast --features parity-trace \
  --bin rust-model-inference
.venv/bin/python tools/oracle/nemotron_reuse/parity.py \
  --model-dir models/NVIDIA-NemotronLabs-RE-USE \
  --oracle models/NVIDIA-NemotronLabs-RE-USE/oracle-pcm \
  --output models/NVIDIA-NemotronLabs-RE-USE/parity \
  --mode cpu --mode auto --mode scalar
```

Each chunk/channel records 36 checkpoints: STFT magnitude/phase, encoder, 30 Mamba blocks, decoder magnitude/phase, waveform. Names, shapes, counts and finite values are checked exactly. Floating-point reductions, transcendental functions, normalization and FFT use numerical tolerances. `scalar` sets `RMI_SCALAR=1` and one thread; SIMD/CPU and Auto use four threads. CPU/Auto WAV identity is checked separately from oracle tolerances.

The trace comparator feeds the **identical native spectral features** to the unmodified official network. Its passing assertions cover network/ISTFT arithmetic, not independent end-to-end parity. `--waveform-only` reports end-to-end error without declaring an oracle pass. Numerical bounds and the initial failed checks are recorded in [validation.md](validation.md).

Upstream SEMamba/config/STFT sources are Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, [OpenMDW-1.1](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1); Mamba sources retain their original license in the downloaded files. Weights, source snapshots, traces and audio are not committed.
