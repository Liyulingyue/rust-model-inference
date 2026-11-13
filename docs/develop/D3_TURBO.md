# Standalone d3-turbo

Status: **Experimental — partial adaptation**. This is the standalone decision
checkpoint, separately registered as `d3_turbo`; it does not load Vela task
adapters or provide a causal language-model generation head. See the
[validation record](../../tools/oracle/d3_turbo/validation.md) before choosing a
format or backend.

The reference is [vllm-sr/d3-turbo](https://modelscope.cn/models/vllm-sr/d3-turbo)
at `0c9281c53740765406dcec7f20140179838a6fa5`. Keep the original checkpoint's
configuration, tokenizer, template, visual weights and `readout.safetensors`
together. The original BF16 storage uses F32 activations/accumulation in this
adapter; this is not native BF16 arithmetic acceleration.

## Convert and run

Download directly on the model host. Conversion streams bounded chunks and
never needs a second full F32 model in memory:

```bash
python -m tools.converter.d3_turbo.convert models/d3-turbo \
  models/d3-turbo-q8_0.gguf --type q8_0
```

Formats: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `q4_k`, `q5_k`, `q6_k`.
Only compatible language-model matrices use the selected format; vision,
normalization, convolution auxiliaries and the readout remain F32. K formats
accept `--workers 8` for bounded parallel conversion. Existing outputs and
partial files are never overwritten. A supplied ModelScope manifest is verified
before its source revision is embedded in the GGUF.

Save a request such as:

```json
{
  "state": "The parcel arrived, but the glass was broken.",
  "questions": {
    "refund": {
      "type": "noul",
      "instructions": "Should this customer receive a refund?"
    }
  }
}
```

Use either the original directory or a converted GGUF:

```bash
rust-model-inference --model models/d3-turbo \
  --decision-request request.json --compute cpu --threads 8
rust-model-inference --model models/d3-turbo-q8_0.gguf \
  --system-one-request request.json --compute cpu --threads 8
```

Both request flags accept `-` for stdin. Use `--decision-request` with
`--max-context` when setting an explicit context limit. `--system-one-request`
retains the existing CLI's task-option restrictions. Native decisions are
nonstreaming and produce probabilities, scores or spatial outputs, with zero
generated output tokens. `choice` criteria use an object of option labels and
descriptions; `score` criteria use an array. Invalid questions return individual
question errors without discarding other valid questions.

## HTTP and JEV

```bash
rust-model-server --model models/d3-turbo-q8_0.gguf \
  --host 127.0.0.1 --port 8000 --compute cpu --threads 8
curl --fail-with-body http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @request.json
```

`POST /v1/decide` is an alias. `/v1/decide/batch` accepts
`{"requests":[... ]}`. JEV uses the existing `--jev` CLI and
`/v1/jev/score`, `/v1/jev/grouped`, `/v1/jev/image`,
`/v1/jev/image_grouped` routes, all through the same model engine.

HTTP image inputs and named `assets` accept inline image data URLs or remote
image URLs when the server's existing remote-image policy permits them. Local
files are CLI-only. HTTP videos accept inline MP4, WebM, QuickTime or Matroska
data URLs, at most 32,000,000 decoded bytes; the request body limit is 64 MiB.
CLI videos can also use files or URLs. Video decoding requires `ffprobe` and
`ffmpeg` on `PATH`; media demuxing is restricted to local file/pipe protocols.

## Media and context

Images precede videos and state text. Named image assets can appear in the state,
instructions and option descriptions; the longest reference wins, and object
keys are not replaced. The native image budget is 65,536–1,638,400 pixels per
image. Video sampling uses 2 fps with up to 32 frames and temporal frame pairs,
at most 200,704 pixels per frame and 16,384 visual tokens per request.

An image `box` question takes `type` and `instructions`. For video boxes add a
finite, nonnegative `at` time. `track` takes `instructions` and optional integer
`rate` 1 or 2, returning timestamped boxes, confidence and visibility. Spatial
video requests require exactly one clip of at most 60 seconds. General videos
are limited to 300 seconds and decoded frames to 8,294,400 pixels. These bounds
are input validation, not evidence that every size/duration has been tested.

The checkpoint has a 262,144-token configured context and a 32,768-token
validated protocol limit. `long_context: true` opts into the configured limit,
subject to the CLI/server context cap. Large-context hardware/memory acceptance
has not been established. The implementation recomputes the prefix per
question; usage counts shared state once but runtime cost is not cached.

## Remaining layers

1. Current layer: standalone contract, converter, common CLI/HTTP/JEV routing,
   CPU/shared SIMD and mixed-device plumbing, fixed-reference diagnostics.
2. Numerical/media acceptance: resolve remaining intermediate-checkpoint gates,
   extend the format × media × entry matrix, and validate large contexts and
   affected legacy Vela models with real assets.
3. Shared Vulkan acceptance: repair the independently reproduced shared
   projection/device-loss failure, then rerun strict Vulkan and Auto numerical
   and same-host performance checks. The shared Vulkan/kernel maintainers own
   this dependency; the adapter owner owns the subsequent D3 matrix rerun.

Auto may fall back to CPU after a device failure. Strict Vulkan errors instead
of reporting a successful CPU execution. Neither fallback nor correct rejection
counts as GPU support passing.

## Same-host diagnostic timings

Eight-thread, 117-token text-choice runs on the authorized EPYC 9334 / NAVI31
host measured 52.14 s end to end for native CPU F32 computation, 35.87 s for
the official F32 graph without trace payload I/O, and 13.08 s for the official
default BF16 CPU runtime. These include startup/loading and use different
activation precision in the BF16 row; they are not throughput measurements.
The default BF16 CPU product result passed the fixed probability/confidence
tolerances with identical choice and token usage.

Strict native Vulkan failed before a completed forward, so it has no GPU
inference timing. The separate official `cuda:0` load was stopped at the
container memory bound before inference, without a recorded OOM event. Exact
commands, timings and setup/interruption evidence are in
[performance.json](../../tools/oracle/d3_turbo/performance.json).
