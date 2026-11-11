# THX-02 official and same-weight Oracle

`oracle.py` executes the pinned original THX-02 decision model, protocol and
runtime, or the native Transformers causal thinker. It verifies the downloaded
official Python sources against revision
`f17dca262dcd1d115e591424ce9cd012b51fc0db`. The optional SDK initializer is
bypassed in an isolated package namespace because its unrelated `client.py` is
not in the model download. The original model/protocol/runtime modules are not
edited.

Run from the repository root with the existing environment. Keep downloaded
weights, requests, logs and generated numerical evidence together:

```sh
/Users/gouzi/Documents/git/rust-model-inference/.venv/bin/python \
  -m tools.oracle.thx02.oracle \
  --model-dir /Users/gouzi/Documents/git/rust-model-inference/models/THX-02 \
  --request tools/oracle/thx02/requests.json \
  --output /Users/gouzi/Documents/git/rust-model-inference/models/THX-02/oracle-original.json \
  --threads 4 --skip-runtime
```

Without `--gguf`, original BF16 source weights are widened to CPU F32. Add
`--gguf /absolute/path/to/THX-02-Q8_0.gguf` for a same-weight reference. The reader
independently decodes F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K with NumPy and
restores the original parameter layouts in memory. Only the GGUF header reader
is shared with the exporter; neither its quantizers nor its tensor transform
functions execute in the Oracle. No additional Hugging Face checkpoint is
written. The main reference verifies the complete backbone/readout tensor set.

`--activation-quantization` additionally applies independent input
quantize/dequantize hooks to native Linear modules whose GGUF weights are
quantized. Q8_0/Q4_0 use 32-value Q8 blocks and F16 scales; K formats use
256-value signed-scale Q8_K blocks and F32 scales. The SSM output input is
temporarily reordered to runtime value-head order before QDQ, which matters
for 256-value groups. Embedding lookups, norms, convolution, and F32/F16/BF16
Linear weights receive no Q8 input hook. The host NEON/AVX2 rounding contract
is recorded by the platform provenance. A CPU hook does not establish a
different device's activation contract.

Retain both references: float activations measure weight quantization drift;
native QDQ helps isolate the runtime's input quantization from implementation
error. Small floating point differences can cross Q8 rounding boundaries, so
passing probabilities alone does not establish a predeclared logit tolerance.
Keep failed comparisons and investigate them rather than widening a tolerance
after observing the result.

The recorded Q8_0 CPU case-0 first-divergence check compares token-major
`[165,2560]` arrays from native checkpoints with Rust parity-trace binaries.
Embedding values are bitwise identical. Layer 0 first differs at max-absolute
`0.00109053`, RMSE `5.7811e-5`; layer 31 reaches `1.75075` / `0.0291811`, and
the output norm reaches `1.07066` / `0.0684132`. This localizes the divergence
to layer computation but does not prove its cause. The complete progression is
`models/THX-02/oracle-q8_0-native-qdq-first-divergence.json`. Q8_0's predeclared
logit acceptance gate remains failed; this evidence does not justify changing
production SIMD accumulation order to imitate PyTorch. Trace timings include
heavy binary I/O and are not performance measurements.

The Q4_K/Q5_K/Q6_K two-case float-input and native-QDQ references also retain
failed predeclared logit comparisons on both CPU and Vulkan. Some QDQ
probability comparisons fail as well. See
`models/THX-02/oracle-k-comparison-summary.json`; weight-only probability drift
is recorded separately in `oracle-weight-quality-drift.json`.

The input is one request object or a nonempty array. Outputs retain request and
question order, exact prompts, input IDs, actual mRoPE positions, marker
positions, all 255 readout logits, responses, source/GGUF hashes, thread count
and environment versions. Text and vision layer checkpoints are NumPy arrays
under the output's `.checkpoints` directory. Each completed joint case is also
saved immediately. `chunks` contains only the plan's selected sequences;
`planning_encodings` additionally retains rejected oversized candidates, and
`runtime_encodings` starts after joint planning. Existing outputs are never
overwritten.

Omit `--skip-runtime` to additionally execute the original `THX02.decide`,
including its joint-read, per-question and TTA plan. `--max-context 117` applies
the same reduced per-sequence limit to joint and runtime planning; by default
the limit is the backbone's 262144 tokens. `--tokenizer-only` validates protocol
and preprocessing without loading weights and cannot be combined with GGUF
execution.

The thinker uses the actual Transformers `Qwen3_5ForConditionalGeneration`
with native cached greedy generation and independently restored merged GGUF
parameters. Its tied output embedding is verified and no MTP acceleration head
is used:

```sh
/Users/gouzi/Documents/git/rust-model-inference/.venv/bin/python \
  -m tools.oracle.thx02.oracle \
  --model-dir /Users/gouzi/Documents/git/rust-model-inference/models/THX-02 \
  --request /Users/gouzi/Documents/git/rust-model-inference/models/THX-02/request-case2.json \
  --output /Users/gouzi/Documents/git/rust-model-inference/models/THX-02/oracle-thinker.json \
  --thinker-gguf /Users/gouzi/Documents/git/rust-model-inference/models/THX-02/THX-02-Thinker-F16.gguf \
  --steps 8 --threads 4
```

The official runtime constructs the thinker body using the decision encoder;
the thinker processor applies its original thinking-enabled chat template.
Both GGUF files therefore carry the decision encoder's 10-item
`thx02.escape_tokens` protocol set, even though the thinker tokenizer's own
special-token configuration contains 18 items. In particular, the body keeps
`<|im_start|>` as the original protocol does. The full thinker tokenizer JSON
and its special-token flags remain intact.
Multiple questions become ordered single-question jobs with source indices.
Every generated step, including a stopping step, records all vocabulary logits
in a `.npy` file. Stops are the original config EOS, tokenizer EOS and official
`</think>` token: `[248044,248046,248069]` for the pinned assets.

The pinned math example's actual F16 same-weight thinker run matches Rust CPU
on prompt text, all 105 input IDs, mRoPE positions and all eight greedy tokens:
`[1206,1423,279,2222,4016,11,31192,279]`. Across eight full 248320-logit rows,
maximum absolute error is `0.00581312` and maximum RMSE is `0.00124490`.
Evidence: `models/THX-02/oracle-thinker-f16-vs-rust-cpu.json`. This validates
that sample's eight steps; it does not establish long generation coverage.
CPU compute dtype is explicitly F32 even when original config declares BF16;
the source JSON files are not modified.

Additional eight-step references cover Chinese JSON state containing
`<|im_start|>` and an image. Both backends match the prompts, input IDs,
positions and greedy IDs. The image case originally failed the F16 full-logit
gate at max_abs `0.143157`. A preprocessing audit found identical native decision/thinker pixel
arrays and all 298 F16 vision tensor payloads identical between their GGUFs;
an isolated intervention then showed that rounding only the 24 native vision
GELU inputs/outputs to F16 closely reproduces the shared Rust discrepancy.
The opt-in F32 GELU method corrects THX-02 while retaining existing callers'
default path. The corrected CPU image check passes all eight greedy IDs and
full-logit gates: max_abs `0.005537`, maximum RMSE `0.001070`, with the original
`0.04 / 0.01` limits. This supersedes the earlier failure, which is retained.
Evidence: `models/THX-02/oracle-thinker-image-preprocess-audit.json`,
`vision-gelu-audit/summary.json`,
`thinker-f16-image-f32-gelu-cpu-vs-official.json` and the CPU/Vulkan image reruns
under `f32-gelu-correction/`. These paths are relative to the model asset root.

`seconds` / `joint_seconds` measure native forward wall time; decision tracing
performs checkpoint writes inside that interval and reports their cost
separately. Thinker logit dumps occur after generation and have separate I/O
timing. Use `--timing-note` to record concurrent jobs or verified isolation.
Timing collected during conversion, builds or other inference is numerical
evidence only, not an isolated performance benchmark.

Focused independent block/layout, hook, native generation and CLI checks:

```sh
/Users/gouzi/Documents/git/rust-model-inference/.venv/bin/python \
  -m unittest tools.oracle.thx02.test_oracle tools.converter.thx02.test_convert_thx02 tools.oracle.thx02.test_compare tools.oracle.thx02.test_verify_thinking
```

Actual coverage, tolerances and unresolved failures belong in `validation.md`;
exporter or tiny native-model tests alone do not validate real-model inference.

Real entry checks reuse the existing binaries:

```sh
/Users/gouzi/Documents/git/rust-model-inference/.venv/bin/python \
  -m tools.oracle.thx02.verify_entries \
  --model-dir /Users/gouzi/Documents/git/rust-model-inference/models/THX-02 \
  --formats F32 F16 BF16 Q8_0 Q4_0 Q4_K Q5_K Q6_K \
  --computes cpu auto vulkan

/Users/gouzi/Documents/git/rust-model-inference/.venv/bin/python \
  -m tools.oracle.thx02.verify_thinking \
  --model-dir /Users/gouzi/Documents/git/rust-model-inference/models/THX-02 \
  --computes cpu auto vulkan
```

`verify_entries` writes canonical `entry-validation/<format>-<compute>` paths;
copy that directory before rerunning to preserve older evidence. The F32 GELU
rerun preserves its predecessor under `public-entries-before-f32-gelu/`, with
binary/source hashes and backup mapping in `final-public-entries/summary.json`.
Thinking uses timestamped evidence directories. Endpoint agreement is separate
from a same-weight numerical gate; failed quantized logits remain failed.

The continuation BF16 thinker run uses the same controller with `--formats BF16`.
Its artifact hash, endpoint records and eight-step numerical comparisons are
summarized in
`/Users/gouzi/Documents/git/rust-model-inference/models/THX-02/continuation-bf16/summary.json`.
