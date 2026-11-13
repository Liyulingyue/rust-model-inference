# Underdog Woof oracle and conversion

This adapter is derived from **Underdog Woof**, Copyright 2026 Conway.
The pinned model includes an adapted Kyutai Mimi decoder. The upstream
[LICENSE](licenses/LICENSE) and [NOTICE](licenses/NOTICE) are retained unchanged.
The upstream license covers its code, weights, configuration and voices;
the repository's MIT license does not replace it. Changes here are a Rust
runtime/text frontend, GGUF conversion and independent validation tools.

Non-commercial use requires credit. Published audio needs “Underdog Woof” or
“underdog.ai” in its title/description; applications offering this feature
must display [“Voice by Underdog Woof”](https://underdog.ai). Commercial use
requires the upstream commercial license. See the full license for all terms.

## Pinned contract

- ModelScope: [ConwayResearch/Underdog-Woof](https://modelscope.cn/models/ConwayResearch/Underdog-Woof),
  revision `bfdcdcbb002ce26fd86ca07c13845a78130aaec4`.
- `woof_int8.npz`: 40,595,426 bytes, SHA256
  `d895fbbe84b8f4f461e3e94ab48b1164353b1988ce8d365f5aa8263d0de1203e`.
- Official NumPy runtime SHA256:
  `e22d1e2a92f4d28c3784abbb889d77609dcecd1856bf6e65f03432b803a8d220`.
- Full source hashes, 175 source arrays, 127 exported tensors and configuration:
  [`contract.json`](../../../src/models/underdog_woof/contract.json).
- English TTS, `juno`/`mira`, 24 kHz mono; eight-layer speech transformer,
  one-step flow head and two-layer causal/windowed Mimi transformer + SEANet.

F32 preserves the **effective published int8/grouped-int8/FP16 weights**.
It cannot recover the unavailable weights preceding upstream quantization.
The converter verifies source hashes without executing the downloaded code;
only the explicitly invoked oracle executes the pinned official implementation.

## Run on the remote machine

All commands below run inside the repository **on the target machine**.
`download` streams HTTP chunks straight into a temporary file there, checks
SHA256, then publishes it. No model is cached on the Mac or intermediary host.
Dependencies stay in the repository's `.venv`; Rust caches/toolchains can stay
in `target/`. No external acceleration library is used by the Rust runtime.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r tools/oracle/underdog_woof/requirements.txt
.venv/bin/python -m tools.oracle.underdog_woof.download models/Underdog-Woof
.venv/bin/python -m tools.converter.underdog_woof.convert_underdog_woof \
  models/Underdog-Woof models/Underdog-Woof/woof-f32.gguf --format f32
cargo build --release --features vulkan --locked \
  --bin rust-model-inference --bin rust-model-server
```

Repeat conversion with `--format f16|bf16|q8_0|q4_0|q4_1|q4_k|q5_k|q6_k`
and a different output filename. Vectors, embeddings and voices stay F32;
unaligned projection rows also stay F32. K formats quantize 16 decoder
matrices; the speech transformer's 448/1216/320-wide rows remain F32.
These names describe mixed tensor storage, not whole-model arithmetic/KV precision.
The nine formats are the applicable shared `quantize_gguf.py` export set;
no new IQ or other quantizer is introduced.

## Reproduce the independent alignment

```bash
.venv/bin/python -m unittest tools.converter.underdog_woof.test_convert_underdog_woof
.venv/bin/python -m tools.oracle.underdog_woof.export_reference \
  models/Underdog-Woof target/woof-source --text 'Hi.' --voice juno --seed 0 --frames 16
.venv/bin/python -m tools.oracle.underdog_woof.export_reference \
  models/Underdog-Woof target/woof-f32 --text 'Hi.' --voice juno --seed 0 --frames 16 \
  --gguf models/Underdog-Woof/woof-f32.gguf --force-latents target/woof-source/latents.f32
RMI_WOOF_FIXTURE=target/woof-f32 RMI_WOOF_COMPUTE=cpu \
  cargo test --profile release-fast --locked --features vulkan,parity-trace \
  --lib official_pipeline_fixture -- --ignored --nocapture
```

Run the same commands separately for every exported format. `--gguf` uses
upstream `gguf.quants.dequantize`, reverses the exported layouts, and feeds
those exact effective weights into the official NumPy runtime. It does not
call the Rust dequantizer or the converter's quantizer to compute references.
Teacher forcing and fixed decoder input separate implementation error from
autoregressive amplification. Checkpoints cover prefix, speech layers,
noise/head/EOS, latents, decoder transformer and convolution stages.
`RMI_PARITY_TRACE` enables the repository's diagnostic checkpoint mechanism.

For natural EOS use `--frames 750 --eos-threshold 0.9`, omit `--force-latents`,
and set `RMI_WOOF_GGUF` when using an original-weight reference. For the
official public normalization/splitting/retry selection use `--public`:

```bash
.venv/bin/python -m tools.oracle.underdog_woof.export_reference \
  models/Underdog-Woof target/underdog-woof/public-f32-juno-short \
  --text 'Hi.' --voice juno --gguf models/Underdog-Woof/woof-f32.gguf --public
.venv/bin/python -m tools.oracle.underdog_woof.check_interfaces \
  models/Underdog-Woof target/woof-entries --bin-dir target/release
```

Before `check_interfaces`, export the public fixture for both voices and
both `short` (`Hi.`) and `split`
(`Your order costs $12.50. It ships on March 3rd.`) cases, named
`target/underdog-woof/public-f32-{voice}-{case}`. The checker runs both
binaries, WAV/PCM, native request controls, request reset, invalid controls,
route isolation and concurrency. It records failures rather than counting
strict-device errors or Auto CPU fallback as GPU verification.

The interface checks pass on CPU and Auto's CPU fallback for all nine formats.
The original strict F32 public waveform gate **fails** for the long Mira case
(relative RMSE `0.0463415` versus `0.002`). Teacher forcing isolates this from
the decoder: a 110-frame Mira probe and a 260-frame decoder-window probe pass.
The strict natural gate is retained; the adapter remains Experimental.

Reproduce the first divergent checkpoints and an independent official-runtime
ULP sensitivity experiment, then measure the same-host performance:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python \
  -m tools.oracle.underdog_woof.analyze_sensitivity \
  models/Underdog-Woof models/Underdog-Woof/woof-f32.gguf \
  target/underdog-woof/sensitivity --test-binary PATH_TO_PARITY_ENABLED_LIB_TEST
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python \
  -m tools.oracle.underdog_woof.benchmark \
  models/Underdog-Woof target/underdog-woof/performance.json --bin-dir target/release
```

The benchmark warms the implementations and records three CPU repetitions.
NumPy's timing excludes model construction; the Rust CLI includes startup and
loading. Auto/strict-device attempts are recorded separately, including errors.

See [validation](validation.md), [coverage/results](results.json),
[performance](performance.json) and the [model guide](../../../docs/develop/UNDERDOG_WOOF.md).
