# Sophon-Realtime-32B-Tag

**Experimental, partial adaptation.** F32 thinker, F32/BF16 audio and short speech components have independent numerical passes; quantized failures remain recorded. See [validation.md](validation.md), [coverage.json](coverage.json) and the [usage guide](../../../docs/develop/SOPHON_REALTIME.md) for entrypoint status and remaining layers. Conversion or a readable reply alone does not establish speech quality, full-duplex, native FP8 or complete Vulkan support.

Read-only references: Sophon `541d83c29ca68dd5235d2e344a2b72d29e6299c2` and DuplexJev `b851ea57ba48b351bb9381d6c16d6d10619a9d6b`. ModelScope file identities are in [modelscope-files.json](modelscope-files.json), revision `990d6a74bbf15c1765a6f7736fa839f423fd5b59`.

## Remote assets

Download on the inference host. The source supports direct remote ModelScope HTTPS downloads, so neither a Mac model cache nor an intermediate Mac weight file is necessary. Check the filesystem backing `models/`, not just `/`: the validation host's workspace has a separate capacity cap, so GGUF exports use a `models/` symlink backed by its larger remote filesystem.

If only the laptop can reach an individual file, stream bytes over SSH with shell `pipefail` and verify the manifest's size and hash on the remote host **before** renaming the remote `.partial` file. For example, after filling in the pinned URL and destination:

```sh
set -o pipefail
curl --fail --location "$MODELSCOPE_PINNED_URL" |
  ssh "$REMOTE" 'cat > /path/to/checkout/models/source/file.partial'
ssh "$REMOTE" 'sha256sum /path/to/checkout/models/source/file.partial'
```

The initial validation downloaded 64 selected files directly to the remote host (42,318,478,677 bytes); redundant ONNX flow/batch-tokenizer copies and the documentation image were not required. No source weights or converted GGUFs were stored on the laptop.

## Independent thinker oracle

The reference uses upstream `gguf` to decode the actual exported file and the Transformers Qwen3-VL implementation to execute it, with F32 arithmetic, independent input QDQ when explicitly requested, and either F32 or explicitly rounded F16 KV. It evaluates weight-only exports; it does not emulate the release's native dynamic FP8 activation quantization.

```sh
.venv/bin/python -m unittest tools.oracle.sophon_realtime.test_oracle
.venv/bin/python -m tools.oracle.sophon_realtime.oracle \
  --model-dir models/Sophon-Realtime-32B-Tag \
  --gguf models/Sophon-Realtime-32B-Tag-GGUF/thinker-F32.gguf \
  --output target/sophon/oracle-f32 --steps 8 --threads 8 --kv f32
```

For a Rust `parity-trace`, pass `--rust-trace target/sophon/cpu.jsonl`. Prompt IDs must match the official tokenizer/template exactly. Compare logits at each teacher-forced step using the predeclared gates in `validation.md`, retain greedy margins, and locate the first checkpoint divergence if they fail. Output directories must be new so old and new evidence cannot be mixed.

## Audio, speech and integration evidence

- [audio-results.json](audio-results.json): fourteen F32/BF16 boundary cases, explicit F16/runtime-QDQ comparison, and retained six-format quantized failures.
- [talker.md](talker.md) and [precision-results.json](precision-results.json): exact tokenizer/schedule checks, 29 independent precision jobs, export/runtime separation and extended F32 fixtures.
- [thinker-results.json](thinker-results.json): eight-step thinker metrics and subsequent format results.
- [performance.json](performance.json): same-host validation timings and paging/profile limitations.

The audio, talker and code2wav references use pinned official component code with independent GGUF decoders. All floating-point/quantized comparisons use declared numerical gates; exact checks are reserved for discrete contracts. Commands and component contracts are in [the converter guide](../../converter/sophon_realtime/README.md). CLI and HTTP compose the same model-local encoder/talker/flow/HiFT implementations.

```sh
PYTHONPATH=/opt/venv/lib/python3.12/site-packages .venv/bin/python -m unittest \
  tools.oracle.sophon_realtime.test_oracle \
  tools.oracle.sophon_realtime.test_audio_oracle \
  tools.oracle.sophon_realtime.test_talker_oracle \
  tools.oracle.sophon_realtime.test_code2wav_oracle
```

Use new output directories for every real comparison. Keep checkpoint binaries, models and generated audio on the inference host; only small JSON reports and logs belong in the review checkout.
