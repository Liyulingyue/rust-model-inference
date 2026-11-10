# THX-02

Status: **Experimental / partial adaptation**. Native text/image decisions, the
optional causal thinker and CLI/HTTP/JEV wiring are implemented. The exact
verification state belongs to the [validation record](../../tools/oracle/thx02/validation.md),
not to this implementation inventory.

The source is ModelScope `HAL-X/THX-02` at
`f17dca262dcd1d115e591424ce9cd012b51fc0db`. This is a decision model: 255-way
readout logits at `<|fim_suffix|>` are converted to choice, boolean probability
or ordered scores. It is not a causal chat endpoint. The 32-layer Qwen3.5-4B
backbone alternates three causal gated-delta layers and one **noncausal** full
attention layer. A causal decoder with the same weights gives different answers.

`src/models/thx02/` owns validation, the embedded tokenizer, templates, image
preprocessing, M-RoPE, sessions, TTA and self-gated thinking. The app decision
adapters own files/stdin, HTTP image security and JEV result layouts. HTTP routes
are mounted only on `Backend::Decision`: `/v1/decide`, `/v1/decide/batch` and the
four applicable `/v1/jev/*` routes. The existing executables are reused; Cargo's
binary inventory is unchanged. See the [usage guide](../usage/thx02.md).

The converter bundles vision and readout weights in `general.architecture=thx02`.
Thinking uses a separate `qwen35` GGUF with `thx02.component=thinker`; its source
revision and base weight hash must match. LoRA is merged in F32 then rounded to
BF16 like the official builder, before the requested storage conversion. MTP is
an optional acceleration head, so ordinary greedy inference does not load it.

Important compatibility details:

- Qwen's adjacent repeated recurrent heads are permuted at conversion time to
  the shared trunk's modulo layout. This includes the value segments of QKV and
  convolution, gates, alpha/beta, A/dt and the output's input columns.
- The configured HF special-token map controls escaping. It differs from the
  tokenizer JSON's `added_tokens[].special`; in particular `<|im_start|>` and
  `<think>` are not escaped by this checkpoint's official processor. Thinking
  bodies use the decision encoder's ten escapes; the thinker tokenizer retains
  its complete, different special-token map.
- Image resizing uses the published 65,536–1,638,400 pixel limits, factor 32 and
  Python ties-to-even rounding. Shared llama.cpp resizing has different rounding,
  so this small preprocessing adapter remains local to THX-02.
- Joint facet grouping, odd question splits and Python `str()` in the thinking
  gate are learned input behavior. Changing grouping changes noncausal attention.
- Native vision retains F32 GELU activations. THX-02 explicitly selects
  `VisionEncoder::encode_image_f32_gelu`; the existing Qwen3.5 image/pair methods
  continue to use their original F16-rounded GELU.

CPU execution reuses the shared SIMD kernels and thread pool. Buffers,
accumulation, recurrent state and thinker KV use F32. Shared prepared F16
projections round their input to F16; other F16 dispatch paths can retain F32
inputs. Q8_0/Q4_0 use Q8_0 activation blocks; K formats use Q8_K blocks.
Storage, per-operator input rounding and accumulation are separate contracts.
Comparisons retain both a float-activation reference and an independent native
activation QDQ reference to expose quantization drift separately.

Auto and strict Vulkan use existing projection kernels and verified attention.
The gated-delta scan, norms and ineligible small projections run on CPU; attention
can also use the shared verified CPU fallback. This is a **mixed graph**, not a
resident device implementation. Strict Vulkan requires actual device submissions.
The existing Qwen3.5 resident decoder cannot express noncausal decision attention
through a stable model-neutral interface. Follow-up L2 belongs to shared hybrid
GPU dispatch/operators: expose that interface, add a gated-delta device scan and
validate quantized CPU/device precision before claiming full device coverage.

The [converter](../../tools/converter/thx02/README.md) exports F32, F16, BF16,
Q8_0, Q4_0, Q4_K, Q5_K and Q6_K. Q4_0/K files keep vision in F32 because the
shared vision loader cannot consume those block formats. Q4_1 remains a shared
exporter gap (follow-up L3, converter/quantization owner); it is not marked N/A.
No external acceleration library or dependency was added to Rust inference.

The merged BF16 causal thinker is now exported at
`models/THX-02/THX-02-Thinker-BF16.gguf` and has passed real CPU, Auto and strict
Vulkan CLI/HTTP checks. Its pinned eight-step CPU text, structured-edge and
image comparisons, plus Auto/Vulkan text comparisons, are recorded in
`models/THX-02/continuation-bf16/summary.json`; the largest recorded image
full-logit error is `0.00128603` max-absolute / `0.00021376` RMSE against the
BF16 Oracle limits `0.10 / 0.025`.

Cross-model changes expose `HybridTrunk::forward_recurrent_layer` and
`forward_ffn_parallel` without changing their bodies, and add the opt-in F32
GELU image method to the shared Qwen3.5 vision encoder. Its layer loop is private;
supporting native activation precision locally would otherwise duplicate that
encoder. A function argument selects the existing SIMD GELU implementation;
old APIs, configuration and default arithmetic remain unchanged. Shared callers
include text/multimodal, omni, diffusion, Qwen-Drive and Vela3/d3. The loader
explicitly registers THX-02 and its 256-wide attention head.

The original F16 thinker image check failed at logits max_abs `0.143157`.
An isolated native GELU-rounding intervention identified the shared precision
mismatch. With F32 GELU, the CPU eight-step check passes at max_abs `0.005537`
and RMSE `0.001070`; Vulkan passes at `0.005391 / 0.001267`. Both retain exact
prompts, IDs, positions and greedy tokens;
the unchanged F16 limits are `0.04 / 0.01`. Old failed outputs are retained.
See the validation record for the device and public-entry reruns.

The real legacy Qwen3.5-0.8B BF16 + F16 mmproj CPU sample retained its input
IDs and printed first five visual values. The saved baseline used sampling
without a fixed seed, so generated words cannot establish deterministic parity.
Full legacy visual arrays/logits and every shared caller were not compared;
this is a regression limitation, not a numerical pass claim.

Same-host timings and their limitations are recorded in
[performance.json](../../tools/oracle/thx02/performance.json). Cold upload and
warm operator-cache probe times are kept separate; probe warmup is not an HTTP
latency claim. Full context, quality benchmarks and other devices require separate
validation.

Observed Q8_0 timings on the M3 Max, four threads (seconds, nonisolated):

| Input | Rust CPU | Rust Vulkan | Official CPU with native input QDQ |
| --- | ---: | ---: | ---: |
| 151 text tokens, warm | 6.279 | 3.721 | 4.135 |
| 123 text tokens, warm | 4.410 | 2.875 | 3.959 |
| 168 tokens / one image, cold vision, corrected F32 GELU | 8.945 | 16.260 | 7.416 |

The corrected image call submitted 26,084 device operations. It is the first
case in separate processes, while the reference image followed text cases;
cache differences and concurrent work limit this comparison. Historical image
timings are retained in `performance.json`. Dispatch/transfer overhead in
the shared vision path needs work; these timings also exclude Engine construction
and are not isolated benchmark results. Q8_0's logit gate failed, independently
of these performance observations.
