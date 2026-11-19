# Sophon-Realtime-32B-Tag

**Experimental, partial adaptation.** F32/F16/BF16 thinker, the F32 vision encoder, F32/BF16 audio and short speech components have independent numerical passes; quantized failures remain recorded. The new lossless vision export passes all nine CPU CLI/HTTP image entrypoint checks with Q8_0 thinker/F16 KV. See [validation.md](validation.md), [coverage.json](coverage.json) and the [usage guide](../../../docs/develop/SOPHON_REALTIME.md) for entrypoint status and remaining layers. Conversion or a readable reply alone does not establish speech quality, full-duplex, native FP8 or complete Vulkan support.

Read-only references: Sophon `541d83c29ca68dd5235d2e344a2b72d29e6299c2` and DuplexJev `b851ea57ba48b351bb9381d6c16d6d10619a9d6b`. ModelScope file identities are in [modelscope-files.json](modelscope-files.json), revision `990d6a74bbf15c1765a6f7736fa839f423fd5b59`.

All nine thinker exports have CPU/F32-KV numerical records with separately identified artifacts: F32/F16/BF16 pass; Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K fail the original block gates. The replacement-host Q6_K run has greedy agreement 8/8 but seven numerical failures, max rRMSE 0.1524552110 and minimum cosine 0.9883425540. This fresh result does not determine the retired host's uncollected Q6 outcome, or establish HTTP/device acceptance for those formats.

An additional Q6_K/F16-KV CPU CLI completes eight prediction rows, but its independent reference result and exit remain **UNKNOWN** after SSH transport loss. The matching Rust token IDs are execution evidence only. This pending attempt does not replace the completed F32-KV numerical failure. Exact-head CI for code checkpoint `ef0a969` passes all three jobs; see [regression evidence](shared-regression-results.json).

## Asset locations

Download on the inference host. The source supports direct remote ModelScope HTTPS downloads, so neither a Mac model cache nor an intermediate Mac weight file is necessary. Check the filesystem backing `models/`, not just `/`: the validation host's workspace has a separate capacity cap, so GGUF exports use a `models/` symlink backed by its larger remote filesystem.

If only the laptop can reach an individual file, stream bytes over SSH with shell `pipefail` and verify the manifest's size and hash on the remote host **before** renaming the remote `.partial` file. For example, after filling in the pinned URL and destination:

```sh
set -o pipefail
curl --fail --location "$MODELSCOPE_PINNED_URL" |
  ssh "$REMOTE" 'cat > /path/to/checkout/models/source/file.partial'
ssh "$REMOTE" 'sha256sum /path/to/checkout/models/source/file.partial'
```

The initial validation downloaded 64 selected files directly to the remote host (42,318,478,677 bytes); redundant ONNX flow/batch-tokenizer copies and the documentation image were not required. After that host was retired, the Q6_K recovery downloaded the 16 thinker shards and six small files, then the separately indexed `thinker/visual.safetensors` required by the existing reader: 23 files, 35,935,130,322 bytes, all size/SHA256 checks passing on replacement port 31344. The first missing-visual conversion failure remains recorded separately. Historical component reports retain their original host/artifact identities. Those historical remote runs stored no model weights on the laptop. For the authorized 2026-11-19 M3 Max follow-up, source and converted files use `/Users/gouzi/Documents/git/rust-model-inference/models/`, with pinned revision and size/SHA256 checks.

## Independent thinker oracle

The reference uses upstream `gguf` to decode the actual exported file and the Transformers Qwen3-VL implementation to execute it, with F32 arithmetic, independent input QDQ when explicitly requested, and either F32 or explicitly rounded F16 KV. It evaluates weight-only exports; it does not emulate the release's native dynamic FP8 activation quantization.

```sh
.venv/bin/python -m unittest tools.oracle.sophon_realtime.test_oracle
.venv/bin/python -m tools.oracle.sophon_realtime.oracle \
  --model-dir models/Sophon-Realtime-32B-Tag \
  --gguf models/Sophon-Realtime-32B-Tag-GGUF/thinker-F32.gguf \
  --output target/sophon/oracle-f32 --steps 8 --threads 8 --kv f32
```

For a Rust `parity-trace`, pass `--rust-trace target/sophon/cpu.jsonl`. Prompt IDs must match the official tokenizer/template exactly. Eight steps compare the prompt-end prediction and seven teacher-forced continuation predictions. With `--reference-execution full-prefix`, one 22-token causal forward supplies those eight rows for the retained 15-token prompt; it does not recompute eight prefixes. Use the predeclared gates in `validation.md`, retain greedy margins, and locate the first checkpoint divergence if they fail. Output directories must be new so old and new evidence cannot be mixed.

## Independent F32 vision oracle

The converter now writes the generic boolean `clip.vision.preserve_f32=true`, so the shared encoder's existing lossless path retains F32 linear weights and activations. The historical `mmproj-F32.gguf` stored F32 but used Q8-prepacked weights and dynamic Q8 inputs for aligned linears, with F32 fallback for the 4304-wide FFN-down input. Its CLI/HTTP image passes remain historical functional evidence. Missing/false metadata retains that default path; the existing WeVisOne behavior is preserved.

[vision-results.json](vision-results.json) records the separately named `vision-lossless-mmproj-F32.gguf` (`38c73644…`). Its 352 tensor payloads and all original metadata match the historical asset (`75a9be7a…`); the only added metadata is the opt-in. The official Transformers **4.57.6 `Qwen3VLVisionModel`**, without a thinker or custom scalar forward, passes **32/32** checkpoints with the same exported F32 weights on CPU with four threads. The fixed 32×32 red image resizes to 256×256: 64 rows of width 5120, from the 1152-wide, 27-layer tower and DeepStack layers 8/16/24. The unchanged gates are rRMSE <= 0.001 and cosine >= 0.999999; worst rRMSE is 1.26997018e-5 and minimum cosine is 0.9999999999195905, with no divergence.

For the retained remote Rust capture, rerun the official reference into a fresh directory:

```sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
PYTHONPATH=/opt/venv/lib/python3.12/site-packages .venv/bin/python \
  -m tools.oracle.sophon_realtime.vision_oracle \
  --model-dir models/Sophon-Realtime-32B-Tag \
  --gguf models/Sophon-Realtime-32B-Tag-GGUF/vision-lossless-mmproj-F32.gguf \
  --image target/sophon/entrypoints-final-reviewed-q8-cpu-vision-33367/red.png \
  --rust-trace target/sophon/vision-lossless-33367/vision-lossless-rust.jsonl \
  --rust-binary target/sophon/vision-lossless-33367/vision-lossless-lib-test \
  --source-manifest target/sophon/vision-lossless-33367/vision-lossless-source-manifest.json \
  --output target/sophon/vision-reference-fresh
```

The report's `reproduction` section also records the existing ignored Rust image fixture and its 32 exact trace names; `RMI_PARITY_FILTER` uses exact names rather than prefixes. Raw arrays and weights stay remote. The historical shared-default asset produces bit-identical projected/DeepStack plane hashes before and after the opt-in change; other models' real vision assets were unavailable. **New lossless CPU CLI/HTTP image chat and single/grouped JEV pass all nine checks** with Q8_0 thinker/F16 KV; both CLI/HTTP probability differences are zero. Its Auto/Vulkan vision numerical paths remain untested. Recorded concurrent-build timings are not an isolated performance comparison.

## Audio, speech and integration evidence

Audio grouped scoring uses `/v1/jev/audio_grouped` with raw base64 `audio`, `context`, `questions:[{text,groups:[{label,options}]}]` and optional `mode` (`multi_select` by default or `block_choice`). It shares bounded decoding, resident scoring and per-group normalization with the CLI. Nonfinite `description:value` scores are rejected before inference in both entrypoints; the existing single-mode `/v1/jev/audio` protocol is retained. Measured combinations and artifact identities remain separate from this schema.

Replacement host 31344 passes real CPU single, block-choice and default multi-select CLI/HTTP checks with **zero probability differences** and seven HTTP 400 cases. The distinct CLI/server (`e842da75…` / `88206ac9…`) use Q6_K thinker + F32 audio and a 250ms synthetic tone; this establishes protocol consistency, not speech quality or a matching-format/GPU numerical pass. Server tests pass 112/112; the same-feature CPU-only full suite retains the baseline's identical 16 failures (1566 to 1568 passes, 106 ignored). The original harness health-response parse failure remains recorded. See the artifact-bound [entrypoint](entrypoint-results.json) and [regression](shared-regression-results.json) reports.

- [audio-results.json](audio-results.json): fourteen F32/BF16 boundary cases, explicit F16/runtime-QDQ comparison, and retained six-format quantized failures.
- [talker.md](talker.md) and [precision-results.json](precision-results.json): exact tokenizer/schedule checks, 29 independent precision jobs, export/runtime separation and extended F32 fixtures.
- [thinker-results.json](thinker-results.json): eight-step thinker metrics and subsequent format results.
- [entrypoint-results.json](entrypoint-results.json): artifact-bound CPU text/image/JEV, mixed-format audio/S2S, F32 TTS and strict device failures.
- [performance.json](performance.json): same-host validation timings and paging/profile limitations.

The audio, talker and code2wav references use pinned official component code with independent GGUF decoders. All floating-point/quantized comparisons use declared numerical gates; exact checks are reserved for discrete contracts. Commands and component contracts are in [the converter guide](../../converter/sophon_realtime/README.md). CLI and HTTP compose the same model-local encoder/talker/flow/HiFT implementations.

```sh
PYTHONPATH=/opt/venv/lib/python3.12/site-packages .venv/bin/python -m unittest \
  tools.oracle.sophon_realtime.test_oracle \
  tools.oracle.sophon_realtime.test_vision_oracle \
  tools.oracle.sophon_realtime.test_audio_oracle \
  tools.oracle.sophon_realtime.test_talker_oracle \
  tools.oracle.sophon_realtime.test_code2wav_oracle
```

Use new output directories for every real comparison. Keep checkpoint binaries, models and generated audio on the inference host; only small JSON reports and logs belong in the review checkout.


## Mixed Vulkan follow-up (2026-11-19)

Audio projections, block attention and frontend; Talker projection offload; Flow projections/convolutions/attention; and HiFT convolutions/STFT/iSTFT now execute on Vulkan. Host layout, normalization, pointwise operations and sampling remain explicit; Talker also retains biased Qwen2 attention/RoPE/F32 KV and its output head on CPU. Strict mode propagates device errors. Auto discards a failed operation, recomputes it on CPU and disables that component's device for the session.

All 81 format/component/backend combinations execute; F32/F16/BF16 pass all 27 short component gates. These fixtures run Flow and HiFT independently, with a supplied HiFT mel, and cannot establish connected synthesis acceptance. After convolution review, F32/F16 complete nonstream speech still fails the unchanged CPU/device waveform gate; BF16 nonstream and all three streaming comparisons pass on this one short input. Same-input real HiFT replay and official-reference replay isolate the nonlinear amplification of the small Flow mel differences. Block-quantized failures, including greedy divergence, remain failures. See [vulkan-results.json](vulkan-results.json), [coverage.json](coverage.json), and the latest section of [validation.md](validation.md).

The new mixed Q4_0 thinker/F32 audio/F16-KV CPU/Auto fixture also covers audio CLI, HTTP batch/SSE and single JEV. Same-backend CLI/HTTP probability differences are zero; CPU/Auto probabilities differ by `0.00033009495` and fail the `1e-7` comparison. The final real Auto JEV CLI retest confirms pure JSON stdout after moving runtime contract diagnostics to stderr, with unchanged probabilities. Original invalid stdout remains a failed RED record; these interface probes do not establish backend numerical acceptance or speech quality.

Reproduce device tests with the existing library test target (no new executable):

```sh
cargo test --locked --profile release-fast --features vulkan,parity-trace \
  --lib models::sophon_realtime -- --include-ignored --skip real_
RUST_GPU_MIN_OUTPUTS=0 cargo test --locked --profile release-fast \
  --features vulkan,parity-trace --lib \
  projection_session_preserves_qkv_bias_and_f32_kv_across_decode -- --ignored
```

For real components, set `RMI_SOPHON_COMPUTE=cpu|auto|vulkan` and the existing `RMI_SOPHON_AUDIO`, `RMI_SOPHON_TALKER`, `RMI_SOPHON_FLOW`, `RMI_SOPHON_HIFT`, `RMI_SOPHON_WAV` and `RMI_SOPHON_CODE2WAV_INPUT` paths; run `real_audio_checkpoints`, `real_talker_checkpoints` or `real_code2wav_checkpoints` with `parity-trace`, then use their matching precision Oracle commands above. F16 Flow requires the existing `--activation f32` reference mode. A full CPU speech trace filtered to `sophon.flow.mel,sophon.hift.source,sophon.hift.spectrum,sophon.hift.waveform` can be replayed by `real_hift_device_replays_identical_inputs` using `RMI_SOPHON_HIFT_TRACE` and `RMI_SOPHON_HIFT`.
