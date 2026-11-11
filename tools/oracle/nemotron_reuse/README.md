# NVIDIA NemotronLabs RE-USE

Native Rust speech enhancement for the published SEMamba checkpoint. **Experimental:** the network and inverse STFT agree numerically on identical spectral inputs, but independent FFT rounding can cross the raw phase input's ±π branch cut and produce material end-to-end waveform differences. Independent waveform equivalence remains unresolved. A bounded paired-speech PESQ/STOI evaluation and sweep-suppression ablation are recorded in [measured results](validation.md).

## Contract

- [ModelScope source](https://modelscope.cn/models/nv-community/NVIDIA-NemotronLabs-RE-USE/files), revision `d66af5dfc20db53a49f885eb32e7e7f1fcb70af2`.
- `model.safetensors`: 38,592,940 bytes (36.8 MiB), SHA256 `06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1`; 1,416 F32 tensors. The loader checks every name, shape, dtype, storage length and finite value against the fixed architecture.
- Published `config.json`: SHA256 `e69986483774f01893d2727503addc49ef201d19e6e1a266b8d4e971e28c478d`; 30 bidirectional time/frequency Mamba blocks, hidden 64, inner 256, state 16, convolution 4. The published `model_cfg` and `stft_cfg` must match.
- Original F32 safetensors only. No GGUF architecture/converter, tokenizer or additional component is involved; other dtypes are rejected.
- PCM16 WAV, integer sample rates 8,000–48,000 Hz, 1–8 channels processed independently. More than half an FFT window (approximately 20 ms), at most 10 minutes. Sample rate, channel count and frame count are preserved. HTTP additionally caps the body at 64 MiB.
- Rate-invariant STFT: base FFT/window 320 and hop 40 at 8 kHz, scaled then rounded up to an even size. Periodic Hann window and centered reflection padding. Inputs longer than 2 seconds use fixed 2-second chunks with 50% Hann overlap and the original shifted final-chunk schedule. The inverse STFT's incomplete final hop is padded with `1e-8`, matching the reference.
- Before inverse STFT, the model applies the published sweep-artifact postprocessing: if **more than half** of a frame's `expm1(relu(log_magnitude))` bins are zero, zero that frame's log magnitudes. Both pinned `inference.py` and `inference_chunk.py` apply it; `models/stfts.py::mag_phase_istft` itself does not. Rust follows this separation in `suppress_sweep_frames`, outside `signal::istft`. See the ablation below.
- `--compute cpu` uses existing SIMD dot products and CPU parallelism. `auto` uses the same CPU implementation; strict `vulkan` is rejected. JEV and text generation do not apply to this audio-to-audio model.

## Coverage and remaining layers

| Speech-enhancement weights | Request backend | CLI | HTTP | Implementation / numerical validation |
|---|---|---|---|---|
| Original F32 safetensors, F32 compute, no KV | cpu | `--enhance` | `/v1/audio/enhancements` | Implemented / component checks pass; independent waveform remains Experimental |
| Original F32 safetensors, F32 compute, no KV | auto | Same command, `--compute auto` | Same route, Auto server | Implemented on CPU / bytes equal CPU |
| Original F32 safetensors | vulkan | Rejected before loading | Rejected at startup | Unimplemented device path / rejection checked; no GPU numerical run |
| F16 safetensors | cpu / auto / vulkan | Unsupported | Unsupported | Unimplemented / not run; weight-format layer pending |
| BF16 safetensors | cpu / auto / vulkan | Unsupported | Unsupported | Unimplemented / not run; weight-format layer pending |
| F32/F16/BF16 GGUF | cpu / auto / vulkan | Unsupported | Unsupported | Unimplemented / not run; GGUF tensor mapping and loader pending |
| Q8_0 GGUF | cpu / auto / vulkan | Unsupported | Unsupported | Unimplemented / not run; converter and runtime quantized projections pending |
| Q4_K GGUF | cpu / auto / vulkan | Unsupported | Unsupported | Unimplemented / not run; block eligibility, converter and runtime pending |

JEV, text/vision generation, a tokenizer, streaming responses and external model components have no contract in this checkpoint. This model has no complete Vulkan graph; CPU/Auto identity does not demonstrate GPU acceleration. The next-layer format/device plan is described after the quality reproduction commands below.

## Run

Download into the ignored model directory with the repository-local downloader:

```sh
models/.venv/bin/modelscope download nv-community/NVIDIA-NemotronLabs-RE-USE \
  model.safetensors config.json models/generator_SEMamba_time_d4.py \
  models/codec_module_time_d4.py models/mamba_block2_SEMamba.py models/stfts.py \
  inference.py inference_chunk.py noisy_audio/mic_test2.wav \
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

`--model` also accepts the `model.safetensors` file beside `config.json`. CLI output uses the existing atomic writer; an existing output requires `--overwrite`. The parser allows only `--enhance`, `--model`, `--audio`, `--out`, `--threads`, `--compute`/`--gpu`, `--overwrite` and `--serve` in this mode. It rejects all other switches, including unknown future flags and explicitly supplied default-valued options. The server pre-parser additionally handles `--host`/`--port`. The server accepts audio in each request rather than startup `--audio`/`--out`.

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

## Paired-speech quality and suppression ablation

```sh
.venv/bin/python -m pip install scipy==1.17.0 pystoi==0.4.1 pesq==0.0.4 soundfile==0.13.1 einops
.venv/bin/python -m unittest discover -s tools/oracle/nemotron_reuse -p test_quality.py
# Rebuild without parity-trace after the diagnostic commands above.
cargo +1.98.1 build --locked --profile release-fast \
  --bin rust-model-inference --bin rust-model-server
.venv/bin/python tools/oracle/nemotron_reuse/quality.py \
  --model-dir models/NVIDIA-NemotronLabs-RE-USE \
  --corpus models/NVIDIA-NemotronLabs-RE-USE/quality-corpus \
  --output models/NVIDIA-NemotronLabs-RE-USE/quality
```

`quality.py` downloads and verifies three LibriSpeech clean narration clips and their attribution files using the SHA256 entries from [librosa 0.11.0's registry](https://github.com/librosa/librosa/blob/0.11.0/librosa/util/example_data/registry.txt). The clips are Garth Comira (`5703-47212-0000`), Anders Lankford (`3436-172162-0000`) and Heather Barnett (`198-209-0000`); [LibriSpeech SLR12](https://www.openslr.org/12/), CC BY 4.0. Audio remains in the ignored model directory.

Each clean clip is cropped to [1, 4) seconds, resampled to 8/16 kHz, and mixed with seed-17 stationary Gaussian noise at 0/10 dB SNR. Common gain prevents clipping; clean/noisy waveforms are quantized to PCM16 before evaluation. These 12 three-second pairs exercise the 2-second overlap path. PESQ-NB (8 kHz), PESQ-WB (16 kHz) and STOI use the **clean speech** as reference for noisy, official and native outputs. The oracle computes its own STFT and network inputs using the pinned official CPU `selective_scan_ref`; the published CUDA driver was not run. Both oracle variants use the native PCM16 output encoding; no extra time alignment or gain normalization is applied outside the metrics.

The script records per-case quality, native-minus-official deltas, independent waveform errors, input/source/binary hashes, versions and timings. The predeclared quality-delta limits are absolute PESQ <= 0.05 and STOI <= 0.005; these are bounded quality checks, separate from the unchanged strict waveform tolerances. It checks normal-build CPU/Auto WAV identity and rejects a traced binary. Three speakers with stationary synthetic noise cannot establish general perceptual equivalence or remove the Experimental status.

Recorded paired-speech results are available for [macOS/arm64](quality_results.json) and [Linux/x86_64](quality_results_linux.json), with [Linux environment/binary hashes](environment_linux.json). Both runs pass all 12 fixed quality-delta checks; detailed scope and independent waveform failures are in [validation.md](validation.md). The Linux run uses Python 3.12.3 and Torch 2.9.1+gitff65f5b; its package versions are recorded in the report.

OGG decoding has platform-dependent floating-point rounding even with the same SoundFile/libsndfile versions. The cross-host fixture audit found at most one PCM16 unit of difference after pairing; each host records its own clean/input hashes and supplies exactly the same local PCM to the official and native paths. See the measured counts and decoded-crop replay check in [validation.md](validation.md).

For suppression ablation, each official chunk is synthesized twice from the **same** predicted magnitude/phase, with the published postprocess enabled and disabled. No second network run is needed. `suppress_frames`, `without_suppress` and `suppress_waveform` report the count and quality/waveform impact. The original synthetic fixture ablation is also recorded in [validation.md](validation.md).

This PR is an experimental partial adaptation. The published download is 36.8 MiB and contains 9,607,747 parameters. F16/BF16, GGUF and Q8_0/Q4_K export/runtime support remain unimplemented. A subsequent weight-format layer must map this exact tensor contract to GGUF and validate the existing block-compatible quantizers (keeping ineligible convolution/norm tensors in floating point); successful conversion alone will not count as native low-precision support. A separate device layer needs shared bidirectional selective scan, the dilated 2D convolution/normalization path and model dispatch before Vulkan can run. Those pending layers require CLI/HTTP numerical and same-host performance evidence; Auto currently executes CPU.

Upstream SEMamba/config/STFT sources are Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, [OpenMDW-1.1](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1); Mamba sources retain their original license in the downloaded files. Weights, source snapshots, traces and audio are not committed.
