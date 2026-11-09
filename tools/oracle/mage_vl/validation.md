# Mage-VL-Cowork validation

Current status: the 2026-11-09 changes add compressed text exports and mixed
Vulkan projection offload. The CPU-only/strict-Vulkan-rejection statements
below describe the **2026-11-03 implementation**, and are superseded by the
[current coverage matrix](../../../docs/develop/MAGE_VL.md). The replacement
port-32276 host completed real-weight tests: BF16 passes the recorded CPU and
mixed-GPU numerical gates; Q8_0/Q4_K independent Oracle and image CPU/GPU
gates fail. Those formats remain experimental. No weights were downloaded to
the Mac. See the completed replacement-host section below for current results.

Validated on 2026-11-03 (Asia/Shanghai), on the requested SSH host. All source
weights, exports and activation artifacts stayed on that host. ModelScope
downloads streamed directly into remote `.part` files; only code and small
configuration/template fixtures were transferred from the Mac.

## Fixed inputs and environment

- ModelScope revision: `287c5fdc8c83e43c4d207d21bc51024a52fdff38`.
- Original shard and official source hashes: [README.md](README.md).
- AMD EPYC 9334, x86_64, AVX2/FMA/F16C; 128 logical CPUs, cgroup quota 16 CPUs.
- Linux `6.8.0-147-generic`, Rust `1.94.1`, 8 inference threads, `release-fast`.
- Oracle: torch `2.9.1+gitff65f5b`, Transformers `4.57.6`, torchvision
  `0.24.1+d801a34`, Pillow `11.0.0`, NumPy `2.1.2`.
- Rust uses existing repository CPU/SIMD kernels. PyTorch is an independent
  Oracle, not an inference dependency. The official source was unmodified;
  the runner selected its built-in CPU SDPA fallback and supplied explicit
  raster patch positions / absolute cache positions.
- Oracle checkpoint loading: zero missing, unexpected or mismatched tensors.

| Export | SHA256 |
|---|---|
| BF16 text | `166b5a6852aa4ab9050e12fbf0b9f9a56993057fb9d5d3ca49ca5123016288ff` |
| F32 text | `4ebf4ca222ba95c156ba9a263bddaf57e5ff5eed96449b937364ae31a045763b` |
| F32 Mage-ViT projector (same in both directories) | `0f9e19c2803570cb1d6672781679931e4788b240fe4b0e66474bd8b91dbf8009` |

`check_export.py` checked all **399 text + 297 vision tensors**, for both
exports: names, dimensions, storage types and every payload byte match the
original BF16 tensors or their exact F32 widening. This is a lossless check;
it is separate from floating-point inference accuracy.

## Numerical evidence

The runner checks four tokenizer cases (including Unicode, whitespace and
special controls), official tool-schema/history templates, expanded image
token IDs, and eight greedy decode steps. Tool templates include positive
and negative scientific-notation exponents. No tokenizer/template tolerance
is permitted.

The four inference fixtures are text, 64×64 square, 96×64 wide, and 83×57
requiring resize. Each image compares normalized patches, patch embedding,
pre-LayerNorm, all 24 vision blocks and the merger. Every fixture compares
language blocks 0/17/35, final normalization, and all eight 151936-way logits.
There are 12 checkpoints for text and 40 for each image.

BF16 text weights with F32 KV, worst over the eight logits:

| Fixture | max_abs | RMSE | minimum cosine | Greedy IDs |
|---|---:|---:|---:|---|
| text | 4.697e-5 | 1.577e-5 | 0.999999999973 | 8/8 exact |
| square | 4.876e-5 | 1.757e-5 | 0.999999999992 | 8/8 exact |
| wide | 5.198e-5 | 8.790e-6 | 0.999999999994 | 8/8 exact |
| resize | 2.691e-3 | 9.658e-4 | 0.999999969439 | 8/8 exact |

The F32 text export passed the same four fixtures and all checkpoints. Its
worst logit metrics and all greedy IDs matched the BF16/F32-KV table above;
both exports represent the same original BF16 weight values.

BF16 text with F16 KV, worst over the eight logits:

| Fixture | max_abs | RMSE | minimum cosine | Greedy IDs |
|---|---:|---:|---:|---|
| text | 0.011645 | 0.002273 | 0.999998982281 | 8/8 exact |
| square | 0.013012 | 0.004789 | 0.999996611954 | 8/8 exact |
| wide | 0.013408 | 0.004246 | 0.999995863574 | 8/8 exact |
| resize | 0.011437 | 0.003873 | 0.999998667646 | 8/8 exact |

The initial F16 cosine gate of 0.999999 rejected these cache-rounding errors.
An isolated native F16-versus-F32 comparison reproduced them (worst max_abs
0.013420, RMSE 0.004782, relative RMSE 0.002898, minimum cosine 0.999995859867),
while all diagnostic language/vision stages remained within their F32 gates.
The calibrated F16 gate below keeps absolute and relative error bounds plus
exact eight-step greedy checks; F32 gates were not relaxed.

Gates in `compare.py`:

- Ordinary F32 checkpoints: max_abs < 0.005, RMSE < 0.0005, cosine > 0.999999.
- Resize: relative RMSE < 0.005 and cosine > 0.99998; final F32 logits also
  require max_abs < 0.005 and RMSE < 0.002.
- Pixels without resize: max_abs < 1e-6. Resize permits at most 2.01 uint8
  levels after normalization. The reused Pillow bicubic and official
  torchvision Fast resize differ by 1–2 uint8 levels; vision activations
  amplify this difference. This is explicitly a lossy path.
- F16 KV logits: max_abs < 0.03, RMSE < 0.005, relative RMSE < 0.005, cosine
  > 0.99999, and all eight greedy tokens must still match. Diagnostic language
  traces use F32 KV.

SIMD reductions, transcendental functions, image resize and F16 cache are
evaluated numerically; there is no claim of bitwise inference equivalence.
For example, the resize fixture's final vision block has max_abs 1.166901
but relative RMSE 0.001462 and cosine 0.999998963596; the projector has
max_abs 0.006114 and relative RMSE 0.001325. These deviations begin at resized
pixels (max_abs 0.028441 after normalization), rather than weight conversion
or text decoding.

## Engineering checks and review

Reproduction commands are in [README.md](README.md). Remote raw evidence is
under `artifacts/mage_vl/`: `reference/`, `native-final/`, `native-f32/`,
`native-f16/`, `comparison-*.json`, `check-export.log`, `smoke-reviewed/`, and
the build/test logs. Large activation artifacts are deliberately not in Git.

| Review matrix | Observed result |
|---|---|
| CLI / HTTP / JEV | Final `smoke.py`: 29 checks pass. CPU/Auto text/image outputs match; explicit F16 cache and an additional explicit F32 CLI image run succeed. Chat/Responses/Anthropic agree on interleaved content. Tool schemas/history work; invalid histories/placeholders/output budgets return 400. |
| JEV | Both text and image CLI/HTTP probability maximum differences are **0.0** (required < 1e-6). Image JEV rejects a two-token configured context before vision encoding. |
| CPU SIMD | Existing ComputePool, attention, normalization, GELU-erf and SIMD dot/matmul paths; all three 132-checkpoint configurations pass their documented gates. |
| Weight dispatch | BF16/F32 text plus F32 vision, all 696 tensor payloads checked for each export; missing/mismatched projector and invalid tensor contracts rejected. |
| CPU / Auto / Vulkan | CPU and Auto outputs agree, Auto logs the uncalibrated-device fallback, strict Vulkan rejects the model. No real GPU execution is claimed. |
| Shared code | Full final `cargo test --locked --profile release-fast --lib`: **1319 passed / 16 failed / 88 ignored**; exact same 16 failure names as the measured baseline. New Mage/JSON tests and existing protocol/Laya tests pass. |

Converter unittest: **3 passed**. The remote pinned-download recheck succeeds.
Changed Rust files pass `rustfmt --check`; the staged diff passes
`git diff --check`. The final CLI/Oracle build with `parity-trace` succeeds.
`cargo check --locked --profile release-fast --features vulkan --lib
--bin rust-model-inference --bin rust-model-server --example mage_vl_oracle`
also succeeds after streaming the repository's checked-in shaders to the
remote checkout (the CPU-only checkout initially omitted them).

The independent `sha256sum -c shaders/manifest.sha256` prerequisite check
reports **8 baseline mismatches**: the GLSL and SPIR-V files for `q4_k_matmul`,
`q5_k_matmul`, `q6_k_matmul`, and `f16_matmul`. All eight files and the manifest
are unchanged from base commit `afed6d48b1886c10ce785b56f10d9427830a0681`
(`git diff --quiet HEAD -- shaders` succeeds). Shader regeneration/validation
was not run because this host has no glslang/spirv tools; this is not a clean
shader-manifest check or evidence of working GPU inference.

Self-review and an independent read-only review led to fixes for:

1. Same-turn interleaved image/text order, including the independent Anthropic
   parser; images also retain their position across turns.
2. Image context plus requested output budget checked before vision encoding.
3. Image JEV respecting the configured CLI/server context cap.
4. Official Jinja final-newline handling, insertion order, JSON spacing and
   Python float exponent spelling. Existing Laya JSON serialization is kept.
5. Strict tensor shape/type/storage checks before the shared Qwen loader.
6. No extra system instruction for the HTTP default `tool_choice=none` when
   there are no tools: this previously changed the prompt and image answer
   relative to the CLI / official template.

The clean baseline required one equivalent stable-Rust compatibility fix in
`jinja_compat.rs` (remove an experimental if-let guard). On that baseline:
**1311 passed / 16 failed / 88 ignored**. The 16 failures were compared by
name, not just by count:

```text
models::breeze::tests::main_matrices_require_original_dtype_exact_shape_and_length
models::breeze::tests::main_source_preserves_original_bf16_and_legacy_codec_f32
models::breeze::tests::main_source_rejects_non_bf16_heads_eoi_and_norms
models::clm::tests::matches_reference_golden_vectors
models::clm::tests::rejects_wrong_embedding_width
models::dots::vocoder::tests::q8_lstm_matches_independent_gate_order_bias_and_recurrent_state
models::gemma4::trunk::tests::failed_later_gemma4_chunk_preserves_successful_prefix
models::gemma4::trunk::tests::per_layer_bf16_projection_matches_pinned_scalar_dot_bits
models::gemma4::trunk::tests::per_layer_projection_rejects_non_bf16_weight
models::gemma4::trunk::tests::per_layer_projection_rejects_wrong_bf16_storage_length
models::qwen_drive::tests::planner_scalar_fixture_matches_every_f32_word
models::yue2::tests::nar_noise_matches_torch_2_10_cpu_bits_across_tail_boundaries
ops::argmax::tests::large_random
ops::dot::tests::dot_bf16_f32_is_bitwise_stable_against_ascending_reference
ops::matmul::neon_tests::rope_neox_inplace_matches_pinned_ggml_recurrence_and_fused_rotation
ops::rope::tests::vision_rope_matches_ggml_libc_trig_bits
```

## Coverage boundaries

CPU text/images, BF16/F32 text exports and F32/F16 cache are the supported
scope. `Auto` explicitly falls back to CPU; strict Vulkan is rejected.
The visual budget is capped at 2048 tokens. Only the small fixtures above
have numerical Oracle evidence; multi-image/history support is checked for
ordering, not full multi-image numerical accuracy.

No real GPU execution, Mac/other-platform inference, long-context stress,
large-image accuracy, OSWorld/desktop-agent success rate or performance
benchmark was validated. Audio/video and interactive CLI are unsupported.

## Follow-up review fixes (2026-11-03)

The original 29-check smoke covered the image JEV context cap but missed the
text-only single/grouped path. A fresh repro returned success with
`--max-context 2` for CLI text JEV and HTTP `/v1/jev/score` and `/v1/jev/grouped`.
The shared Qwen scorer now checks the effective configured/model context before
KV allocation; single/multi/block CLI and both HTTP text scorers pass that cap.
Limit violations are HTTP 400; model loading/compute faults remain HTTP 500.
Explicit text-only CLI JEV projectors are also validated, including grouped mode.

The original HTTP image preflight correctly checked expanded image tokens but
then discarded the expanded count. Its usage and count-token response reported
23 tokens for the official 64x64 square fixture's 26-token input. `PromptResult`
now retains the expanded count for Chat/Responses/Anthropic, their streams, and
`/v1/messages/count_tokens`. Generation still receives the original placeholder
IDs, so image expansion happens once. Other backends retain their raw ID count.

The shared Mage image-logits model construction, prompt, prepare and forward
pipeline moved from `app/text/multimodal.rs` to `models/mage_vl::image_logits`.
App code loads media/components and adapts HTTP messages/tools to the official
schema; it no longer owns this Mage forward pipeline. No shared numerical
operator or weight conversion changed.

On the same requested Linux/x86_64 EPYC host, the fresh pre-fix library baseline
is **1319 passed / 16 failed / 88 ignored**; after the fixes it is **1320 passed /
16 failed / 88 ignored**. All 16 failure names are identical, compared as lists
in `artifacts/mage_vl/review-fix-lib-comparison.json`. The new unit test covers
context-limit versus model-fault HTTP status classification. Raw logs are
`fix-baseline.log` and `review-fix-lib.log` in the same remote artifact directory.

Final `review_regression.py`: **40 assertions pass**. Single/multi/block CLI
accepts the exact prompt length and rejects one token below it. CLI/HTTP normal
probabilities have maximum absolute difference **0.0** against a **1e-6** gate.
All three protocols, their streams (including a Responses `response.incomplete`
terminal event at the one-token output cap), and count_tokens report **26**
input tokens for the square fixture. The original **29-check smoke passes**;
text/image JEV probability differences against `smoke-reviewed` from before
these fixes are both **0.0**. CPU/Auto, F16 KV, components, interleaving, history,
image JEV and output-budget failures remain covered by that smoke.

CLI and standalone server `release-fast` builds pass. The Vulkan-feature
library/CLI/server/Oracle example `cargo check` passes; this is compile evidence,
not GPU execution. Changed-file rustfmt, Python syntax and `git diff --check`
pass. Raw follow-up evidence: `review-fix-green/`, `review-fix-green.log`,
`smoke-review-fix/`, `smoke-review-fix.log`, `review-fix-final-build.log`, and
`review-fix-vulkan-check.log` under the same remote artifact directory. Original
numerical-operator/weight-conversion gates and existing failure/shader boundaries
above still apply; performance and real GPU execution were not rerun for these
request-validation/accounting fixes. All model weights remain on the remote host.

## Quantization and GPU review follow-up (2026-11-09)

Merged upstream main through `532d051a0`; tested the unchanged merge baseline
`81551447e`. Text conversion/validation now accepts the existing F32/F16/BF16,
Q8_0/Q4_0/Q4_K/Q5_K/Q6_K formats, with F32 norms and vision. Reused shared
quantizers rather than introducing another encoding implementation.
`check_export.py` can check all formats but re-encoding a quantized payload is
only a storage check, not an independent quantization-accuracy Oracle.

Text and vision projections now use shared Vulkan linear execution. Auto can
fall back; strict Vulkan propagates errors. Attention/KV and Mage-ViT's per-lane
RoPE stay on CPU. The complete-device whitelist remains unchanged. Shared
Qwen session changes only opt Mage into its existing projection executor;
Qwen resident defaults are preserved. HTTP image validation permits a mixed
runtime while retaining text-only checks for complete resident decoders.

On Apple M3 Max hardware Vulkan, both new synthetic device tests pass. The
text test covers F32/BF16/Q8_0/Q4_K, F32/F16 KV, Auto/Vulkan and prefill plus
two decode steps; worst max_abs is `9.536743e-6` against the existing
`0.002 + 0.002 * abs(cpu)` gate. The one-block vision fixture submits seven
projections under each policy and has max_abs `0` against `<0.002`. No bitwise
alignment is required for these floating/quantized paths. See the matrix for
the fixture limits and diagnostic CPU/GPU timings.

The unchanged baseline fails the shared batched-linear hardware test because
the newly merged Nemotron FP64 shaders cannot compile on M3 Max. The optional
pipeline capability gate fixes that initialization failure, and calls to the
unsupported FP64 operators return an explicit error. The same shared test now
passes all nine supported GPU formats, and both existing Qwen hidden-row and
capture device tests pass. FP64-capable hardware and old-model real weights
were not available for regression.

Current checks:

- Python converter/shared quantizer tests: **38 passed, 15 subtests passed**.
- Full ordinary CPU library baseline: **1510 passed / 13 failed / 96 ignored**;
  after changes: **1512 passed / 13 failed / 96 ignored**. All 13 failure names
  match exactly in `target/mage-review-20261109/regression-comparison.json`.
  Existing failures concern media, missing Breeze/CLM assets, Z-Image/FunASR/
  softmax NEON parity, and Gemma4 validation; this is not a green full suite.
- Vulkan-feature Qwen trunk: **12 passed / 3 ignored**; explicitly running its
  three device tests: **3 passed**. Text runtime tests: **6 passed**.
- Smoke script command construction: 14 CPU/Vulkan commands checked; the
  duplicate `--compute` regression found during review is fixed. This is a
  mocked argument check, not a real CLI/HTTP integration result.
- Cargo metadata preserves all nine existing binary targets.
- Vulkan-feature `cargo check` passes for the library, existing CLI/server
  binaries and `mage_vl_oracle` example. Changed-file rustfmt, Python syntax
  and `git diff --check` pass.

Full-weight CLI/HTTP/JEV, per-format independent quantization Oracle, full-model
performance, long-context and broad task quality were **not rerun**. The model
therefore remains partially adapted, with these validation gaps explicit.

### Replacement-host attempt

On the replacement EPYC 9334 / 48 GiB RADV NAVI31 host (SSH port 31599), the
Vulkan CLI/server/Oracle `release-fast` build completed. Both source shards
passed the pinned SHA256 checks; BF16 and Q8_0 exports completed remotely.
Q4_K conversion and four-fixture official/independent quantized Oracle runs
were started but not confirmed complete before every SSH session disconnected.
Weights were never staged on the Mac. No fresh full-model parity, interface,
quality or performance pass is claimed from this interrupted attempt.

Both Mage synthetic hardware tests returned Vulkan device loss. An untouched
archive of main `532d051a0` reproduced RADV hard recovery in the existing
`linear_gpu_group_preserves_mixed_formats_across_batches` test, which received
CPU fallback backends instead of Vulkan. The vision test also failed with the
LLVM RADV compiler. Current AMD GPU parity and speed remain blocked.
The cause of the later SSH disconnection is unknown.

The probe now accepts CPU/Auto/Vulkan explicitly; hidden-language captures stay
CPU diagnostics. An optional upstream `gguf==0.18.0` decoder loads all 399
exported text tensors into the official F32 model for a separate same-weight
Oracle. Quantized-logit comparison requires eight artifacts per fixture and
compares only common-prefix logits after a greedy divergence. Its declared
F32-KV gates are max_abs `<0.5`, RMSE `<0.05`, cosine `>0.9999`; these gates have
not yet been established as passing on this model. Two comparator regressions
cover prefix divergence/missing artifacts and rejection of excessive error.

## Completed replacement-host runs (2026-11-09, port 32276)

The next instance is reachable and its RADV GPU executes the Mage tests without
reproducing the previous instance's device loss. This supersedes the interrupted
attempt's **current-blocker** status; the port-31599 observations above remain
historical facts. Hardware: EPYC 9334, 16-CPU cgroup quota, 59,055,800,320-byte
container RAM limit, RADV NAVI31/gfx1100 with 49,136 MiB GPU memory. Other tasks
share the instance. Builds use existing Rust 1.94.1, `release-fast`, Vulkan,
eight inference threads, and no `RMI_SCALAR` or `RMI_PARITY_TRACE` override.

Source is `dfb150398` plus timing-only changes to `mage_vl_oracle`. Both source
shards passed the pinned SHA256 checks. BF16, Q8_0 and Q4_K exports completed
on the remote filesystem. BF16 and Q8_0 `check_export` each pass all **696**
shape/type/payload checks. All three F32 projector files have SHA256
`0f9e19c2803570cb1d6672781679931e4788b240fe4b0e66474bd8b91dbf8009`.
The independent GGUF decoder checks all 399 Q4_K text shapes and finite values;
a full deterministic Q4_K re-encoding was not rerun. No weights or F32 trace
arrays were downloaded to the Mac.

### Numerical evidence and failed gates

- Official BF16 widened-to-F32 Oracle: no missing/unexpected/mismatched weights.
  The diagnostic native runs pass **132 checkpoints each** for F32/F16 KV.
  Worst F32-KV final-logit max_abs/RMSE are **0.00269055/0.00096573** (resize's
  existing separate gate); F16 KV: **0.01340723/0.00478817**.
- A separately built **non-parity-trace** production probe passes all **64**
  logit comparisons (four fixtures × eight steps × two KV formats), with the
  original BF16/F16/resize gates and identical greedy IDs. Diagnostic hidden
  captures remain CPU F32 and are not evidence of device hidden execution.
- Full-weight BF16 CPU↔Auto/Vulkan: text and square with F32 KV, plus square
  with F16 KV. All six runs submit Vulkan work, retain eight greedy IDs and
  pass max_abs `<0.03`, RMSE `<0.005`, cosine `>0.99999`. F32-KV worst
  max_abs/RMSE: **0.00011564/0.00003732**. F16-KV worst:
  **0.01382494/0.00381627**. The BF16 bound accommodates cache sensitivity;
  it is separate from the tighter CPU↔official-Oracle gates above.
- Q8_0/Q4_K text-only CPU↔Auto/Vulkan logits are numerically identical over
  eight steps. All quantized square runs retain the eight greedy IDs, but
  **fail** the previously declared `<0.5` max_abs, `<0.05` RMSE, `>0.9999`
  cosine gates. Q8_0 image worst max_abs/RMSE/cosine:
  **0.49679983/0.17498699/0.99913106**. Q4_K:
  **1.35604859/0.55691394/0.97540338**. F16 quantized runs compare to CPU F32
  KV and therefore include both cache and backend error. All 18 device probes
  completed with actual Vulkan trace evidence; only **10/18** pass their
  numerical gates. Successful execution is not a numerical pass.
- Independent same-weight Oracle uses upstream `gguf==0.18.0` to decode all
  399 text tensors into the official F32 model. Four fixtures × eight steps
  agree in greedy IDs for each format. Both formats **fail** the original
  quantized-logit gates: Q8_0 worst max_abs/RMSE/minimum cosine are
  **0.46366143/0.15020036/0.99558480**; Q4_K:
  **1.13677478/0.43955483/0.98222714**. This compares F32 HF activations with
  native quantized activations; it does not measure BF16→quantized task quality.

A separate one-step diagnostic quantized inputs to all 253 HF text linears
using upstream Q8_0 quantize/dequantize. Text RMSE fell from **0.15020** to
**0.11148**, but still failed. This does not establish activation quantization
as the sole cause: SIMD rounding/reduction differ and quantizer boundaries can
amplify small upstream errors. No shared numerical kernel was changed to force
bitwise agreement, and no gate was widened. Quantized formats and their image
GPU paths therefore remain **experimental**, with unresolved numerical gaps.
Per-fixture values, contracts and statuses are committed in
[real-weight-results.json](real-weight-results.json).
For context, a separate descriptive softmax-probability comparison has maximum
total variation **0.034261** for Q8_0 and **0.104831** for Q4_K over these same
32 steps per format. Some logit error is a common offset, but centered RMSE
still reaches **0.077197/0.233500**. These numbers neither replace the failed
gates nor establish broad task quality.

Both real-GPU synthetic tests pass on this instance: quantized text prefill and
decode across F32/BF16/Q8_0/Q4_K × F32/F16 KV × Auto/Vulkan, and seven vision
projections with max_abs **0**. These constructed fixtures do not override the
full-model image failures above. Attempting to compile untouched main
`532d051a0` on Rust 1.94.1 stops at its experimental if-let guard in
`src/prompt/jinja_compat.rs:260` (E0658). No baseline GPU pass is claimed from
that build; the earlier M3 baseline and unchanged failure-list comparison remain
the available shared-code regression evidence.

Raw remote evidence is under `target/mage-remote-32276/`; large traces remain
under `artifacts/mage_vl/`. Examples of the completed numerical commands:

```sh
.venv/bin/python tools/oracle/mage_vl/compare.py artifacts/mage_vl/reference artifacts/mage_vl/diagnostic-f32
.venv/bin/python tools/oracle/mage_vl/compare.py artifacts/mage_vl/reference artifacts/mage_vl/diagnostic-f16 --kv-format f16
# The following intentionally returns failure; do not widen gates to make it green.
.venv/bin/python tools/oracle/mage_vl/compare.py artifacts/mage_vl/reference-q8_0 artifacts/mage_vl/production-q8_0-f32 --quantized-logits
target/mage-remote-32276/mage_vl_oracle.production models/Mage-VL-Cowork-bf16/Mage-VL-Cowork-bf16.gguf models/Mage-VL-Cowork-bf16/mmproj-Mage-VL-Cowork-F32.gguf artifacts/mage_vl/reference/square artifacts/mage_vl/production-bf16-f32-vulkan/square f32 vulkan
```

Self-review corrected an earlier documentation error: the output head is **not
always CPU**. `prefill.rs::matmul_rows` calls `PreparedRows::matmul`, which can
use the weight's legacy Vulkan wrapper under Auto/Vulkan. Device logs show
`scope=linear backend=Vulkan` after the explicitly registered block projections.
The head is outside `LinearExecutor`'s strict guarantee and can fall back through
its existing dispatcher. This correction changes no kernels or measured runs.

The **BF16** current-instance integration runs pass the original **29-check
CLI/HTTP/JEV smoke** with hardware Vulkan and the **40-assertion CPU review
regression** (context limits, expanded usage, three protocols and streams).
Q8_0 and Q4_K smoke runs each fail at CPU/Auto image-output equality, before
later strict-Vulkan/HTTP/JEV checks. For the smoke prompt, Q8_0 begins
“I’m continuing from the previous step,” on CPU and “I’m continuing from the
earlier step where” on Auto. These are generated-text differences, not just
log differences. Later independently executed request/JEV results are recorded
separately; they must not be counted as those two smoke runs passing.

### Standalone interface and effect evidence

The standalone JEV matrix passes **16 text/image CLI↔HTTP checks**: BF16 under
CPU/Auto, and Q8_0/Q4_K under CPU/Auto/Vulkan. Each maximum probability
difference is **0**, against `<1e-6`. BF16 strict-Vulkan text/image JEV is
already covered by its passing smoke. These establish consistency between
interfaces under the same policy, not accuracy across policies or formats.
Four separate Q8_0/Q4_K strict-Vulkan CLI text/image generation requests exit
zero and report `scope=host_linear backend=Vulkan`. Ordinary non-trace HTTP
generation completes for all nine format/policy combinations in
[performance.json](performance.json). Quantized image CPU/Auto answers still
differ, so these completed requests do not supersede the failed smoke.

Ordinary BF16 and Q4_K CPU chat samples (arithmetic and a red/blue square image,
temperature zero, 512-token cap) all exhaust the cap with repeated thinking.
Both image answers identify the blue square as a rectangle. Text HTTP content
can be empty at the shorter 128-token cap because reasoning is filtered. This
checkpoint is a computer-use policy; these ordinary prompts do not establish
its performance in the official agent harness.

A separate synthetic 384×256 note-editor screenshot asks to save the note.
The Save button occupies pixels `[260,190,362,238]`. Requests use the pinned
model card's hybrid system prompt, explicitly declare `computer_use`/`bash`
in HTTP `tools`, select `tool_choice:"auto"`, CPU/F32 KV, context 4096,
temperature zero and a 512-token cap. Each format has one valid request:

| Format | Prompt / completion tokens | Request seconds | Actual returned tool call |
|---|---:|---:|---|
| BF16 | 1964 / 164 | 256.732 | `computer_use {"action":"screenshot"}` |
| Q4_K | 1964 / 169 | 257.944 | `computer_use {"action":"screenshot"}` |

Both describe the Save button and say to click it in their text, but neither
returns a click or coordinates. **The save task is not completed.** No tool
action was executed. These shared-host observations were not isolated timing
benchmarks, and no OSWorld success rate or localization pass is claimed. An
initial fixture omitted API tools and received the expected “tool call while
tools are disabled” error; the table contains only the corrected requests.
The small JEV, strict-CLI and complete GUI responses are preserved in
[real-weight-results.json](real-weight-results.json); weights and large traces
remain on the remote host.

Between weight formats, Q8_0 matches BF16's eight greedy choices in all four
Oracle fixtures. Q4_K changes the third text choice and the first image choice.
Agreement with the **same Q4_K weights** in HF does not mean agreement with
BF16 or preserved task quality. All native IDs are in the results JSON.
