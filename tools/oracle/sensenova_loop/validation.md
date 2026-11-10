# SenseNova Loop validation

Validation date: 2026-11-03 (Asia/Shanghai). Base: `afed6d48b`.
The original adaptation at `5937066` was an experimental F32 CPU
implementation: CPU and Auto deliberately used the same CPU kernels, and
strict Vulkan was rejected. The GPU extension below supersedes that routing
behavior; the original numerical records remain CPU evidence.

## Source and environment

- ModelScope `SenseNova/SenseNova-Loop-9B-Preview`, revision
  `4945891aba37ed2acfdb0119f5e0db224d04e20d`.
- Every downloaded source file was checked against the sizes and SHA256 in
  [source-manifest.json](source-manifest.json), including the MTP shard needed
  by the official loader. MTP is not exported or executed.
- Config SHA256:
  `3069b497e0a9ba70b1f74561831cbab0df0ccc4247f3ae106054c1632360ef51`.
- Official modeling source SHA256:
  `26e11d3522aa730c0ba031340ce42632bc2834da8cc0b6804006bb207d979a5b`.
- AMD EPYC 9334, x86_64 Linux 6.8.0-147-generic, 16 inference threads.
  The shared container has a 16 CPU quota and 59,055,800,320-byte memory limit;
  the host's advertised 503 GiB is not the usable memory allowance.
- Rust 1.99.1, PyTorch `2.9.1+gitff65f5b`, transformers 5.12.0,
  numpy 2.5.4, safetensors 0.8.0, Pillow 13.0.0.
- Rust tools/dependencies and Python environment live under the remote
  repository's `target/` and `.venv/`. No system package or shell-rc change.
- Source and GGUF weights stayed on the SSH host. Initial downloads used a
  streaming SSH pipe; later downloads went directly from ModelScope to the
  host. No weight file was written on the Mac.

The actual source matrices are F32 despite the config's BF16 declaration.
The converter preserves that precision and produces an F32 text GGUF and F32
projector. Text quantization/BF16 export is intentionally unsupported.

## Numerical method

The reference is the pinned, unmodified official remote-code model, evaluated
with F32, eager CPU attention and KV caching. The fixture runner captures all
56 layer visits independently. The native implementation shares the 32
physical weight sets but allocates independent attention, convolution and SSM
state for every visit.

Tokenizer IDs (plain and rendered prompts) and greedy token IDs must match
exactly. Floating-point states and logits must be finite, have the expected
shape/count, and pass both predeclared gates:

```text
max_abs <= 1e-3 + 1e-4 * max(abs(reference))
RMSE    <= 1e-5 + 1e-4 * RMS(reference)
```

No bitwise PyTorch claim is made for SIMD reductions, exp/sigmoid, Sinkhorn,
normalization or other floating-point arithmetic. The owned-versus-borrowed
F32 storage regression does check exact bits because it runs identical native
arithmetic over identical values.

Text fixtures cover an eight-token English decode, a Chinese prompt and
Unicode combining characters, whitespace and an explicit special token.
The image fixture is a 256 × 256 solid-blue PNG, followed by three greedy
tokens; its visual projection is compared separately. This image needs no
resize, so it does not establish PIL/native interpolation equivalence for
arbitrary image sizes.

| Run | Floating-point checks | Failed | Tokenizer/greedy |
| --- | ---: | ---: | --- |
| Text, CPU | 570 | 0 | Exact |
| Text, Auto | 570 | 0 | Exact |
| Single image, CPU | 172 | 0 | Exact |
| Single image, Auto | 172 | 0 | Exact |

Both policies produced `[9419, 0, 2500, 513, 488, 3604, 3242, 30]`
(`Hello! How are you doing today?`), `[109430]` (`您好`) and `[34]` (`C`).
All 3,258 text CPU/Auto binary trace records are identical to each other. This
does not imply bitwise equivalence to the official PyTorch implementation.

For both policies the largest logits max-absolute error is
`4.3392181396484375e-5`. The checkpoint with largest relative RMSE is English
step 0 logits: RMSE `7.859274640026298e-6`, relative RMSE
`3.756516677552462e-6`, cosine `0.9999999999929948`. Across text layer states
the largest absolute error is `4.6539306640625e-4`; every checkpoint passes
both scale-aware gates.

Measured CPU/Auto peak RSS was 31,447,020 / 31,445,312 KiB (about 30 GiB).
The measurement uses Python `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`.
These are fixture-run observations, not general memory limits or benchmarks.

Exported artifacts (remote only):

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `SenseNova-Loop-F32.gguf` | 35,838,774,368 | `593418b76e009467ff6fac197fe034a05c0fdfbeb4fcd98ea1f78d33d9d6904e` |
| `mmproj-SenseNova-Loop-F32.gguf` | 1,824,061,376 | `0e80bc1260d84e8b8821bdea983d85495c018c93f61ff90f9108ef25c1820856` |

The image run generated `[1919, 2099, 369]` (`The image is`) on both sides.
Its vision projection has max-absolute error `2.17437744140625e-4`, RMSE
`1.6856149465683227e-6`, and relative RMSE `1.0483531997574643e-5`.
Across its 56 visits and three generation steps, the largest state error is
`0.019773483276367188`; the largest relative RMSE is
`7.608416791464315e-5` at visit 54 during prefill. These large-magnitude mHC
states pass the same predeclared scale-aware gates. Final logits max-absolute
error is at most `5.030632019042969e-5`. All 5,023 image CPU/Auto native binary
trace records match each other. Image fixture peak RSS was 31,551,200 /
31,553,768 KiB for CPU/Auto.

## Interface checks

The CLI/server commands in README were run with 16 threads and a 512-token
context limit. HTTP used `127.0.0.1:18081`, non-streaming JSON, temperature 0
and `enable_thinking:false`. The image chat included a system turn
`Answer concisely.` before the image-bearing user turn.

| Entry | Observed result |
| --- | --- |
| CLI text, `Hello!`, 8 tokens | `Hello! How are you doing today?` |
| CLI single image, 3 tokens | Successful generation |
| Strict `--compute vulkan` | Nonzero exit with the SenseNova CPU-only reason |
| HTTP text chat, 8 tokens | HTTP 200; same text as CLI |
| HTTP image chat, 3-token cap | HTTP 200; `Blue.`, then EOS after 2 tokens |
| CLI/HTTP ordinary text JEV | Both pass; maximum probability difference 0 |
| CLI/HTTP grouped text JEV | Both pass; maximum probability difference 0 |
| CLI/HTTP single-image JEV | Both pass; maximum probability difference 0 |
| HTTP grouped-image JEV | HTTP 200; finite normalized probabilities, blue selected |
| CLI grouped-image JEV | Existing explicit single-question-only rejection |

All JEV requests used context `The image is blue.`, question `What color?`,
and options `blue`, `red`. CLI grouped text used `--jev --jev-multi`.
HTTP ordinary requests used
`{"context":"The image is blue.","questions":[{"text":"What color?","options":["blue","red"]}]}`;
grouped requests used `mode:"multi_select"` and one group labeled `pair_1`
with the same options. Image requests added the PNG as an `image_url` data URL.
The tested endpoints were `/v1/jev/score`, `/v1/jev/grouped`, `/v1/jev/image`
and `/v1/jev/image_grouped`. The grouped-image probability for blue was
`0.9999896287918091`; this is an interface smoke result, not a quality score.

## Regression checks

The default release-fast library suite was run before and after the change:

| Revision | Passed | Failed | Ignored |
| --- | ---: | ---: | ---: |
| Unmodified base | 1311 | 16 | 88 |
| Reviewed implementation | 1322 | 16 | 88 |

The failure-name sets are identical. The existing failures are:

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

The converter/comparator Python suite has eight passing tests. These exercise
the pinned execution contract, tensor naming, invalid config,
floating-point rejection and chunked-prefill logits
selection. Rust regressions cover shared-weight/independent-cache visits,
Sinkhorn/merge, tensor shape/storage validation, projector identity, strict
Vulkan policy, context limits and borrowed F32 storage.

The CLI, server and fixture example build with
`--features parity-trace,vulkan` passed. The same feature combination's
`--lib sensenova` tests passed 8/8 (including media and vision regressions).
The shared `--lib qwen35::` regressions passed 104 tests, with 19 ignored, and
the borrowed-F32 storage regression passed. Commands and numerical reproduction
are in [README.md](README.md).
All examples also pass `cargo check --locked --profile release-fast
--features parity-trace,vulkan --examples` after completing the remote source
copy; the first attempt lacked an existing tracked example, `vk_ops_check.rs`.

`cargo +1.98.1 fmt --check` still reports formatting drift in nine existing
files (server/WeVisOne, Audio8, NEON quantization and Vulkan code). Formatting
each corresponding unmodified-base source reproduces the drift. New files and
the changed implementation hunks are formatted; unrelated base formatting was
left unchanged. `git diff --check` passes.

## Review and fixes

A self-review and a separate read-only review were completed before the final
validation. The fixes include:

- Preserve the official HF GDN `repeat_interleave` key/value-head pairing;
  existing GGUF Qwen3.5 keeps its prior modulo pairing. The first incorrect
  mapping failed all 570 text checkpoints; the corrected mapping passed them.
- Separate physical weight indices from visit cache indices.
- Check matching projector identity and fail unsupported model loading at
  startup, rather than deferring it to an unsupported HTTP runtime.
- Use the actual user turn for HTTP image insertion and the M-RoPE text
  continuation position for decoding.
- Select the existing interleaved T/H/W M-RoPE operator for SenseNova's
  attention. The previous contiguous layout first diverged at physical
  attention layer 3 for images; equal text axes had hidden this difference.
- Honor CLI Jinja templates and configured context limits.
- Borrow aligned little-endian F32 mmap weights, with checked owned fallback,
  instead of copying the entire model for each runtime/JEV view.
- Preserve F32 patch weights and GELU in the vision encoder for SenseNova.
  The shared legacy F16 GELU path initially failed the image projection gate
  (max absolute error `0.144288`, RMSE `8.495799e-4`). The fix leaves the
  existing GGML model arithmetic unchanged. A separate official vision-only
  run reproduced the original full-image reference projection bit for bit.
- Pack patch weights and pixels in source C/T/H/W order and contract both
  frames in one existing F32 dot kernel, reducing the roundoff propagated
  through the repeated language layers.
- Compare the final prompt logits at the cumulative row boundary, ignoring
  intermediate logits emitted by chunked prefill; attach each chunk's logits
  to its last row when flushing token-major traces.

The independent reviewer reported no remaining critical/important findings;
the reviewer did not independently rerun the remote numerical checks.

## Limits

Only the pinned Preview checkpoint, source F32 text and exported F32 projector
are covered. Both temporal patch matrices must stay F32; manually encoded
dtypes for other projector matrices may be accepted by the shared loader but
are unverified. Physical GPU calibration/performance, other text quantizations,
video/audio, multiple images, MTP, long-context quality, broad image/task
quality, other platforms and performance targets have not been validated.
Resource contention from unrelated remote jobs makes these runs unsuitable as
performance benchmarks.
HTTP `usage.prompt_tokens` follows the existing server's text-token count
before native image-placeholder expansion, rather than the effective image
context length. Actual-weight regressions for other architectures were not run.


## Vulkan extension and review fixes (2026-11-03)

The decoder now records the physical execution plan on Vulkan, uploading each
physical weight once and giving every visit independent dense KV, convolution
and SSM state. mHC expansion/routing/log-Sinkhorn/merge/reduction run on device;
router weights remain F32. SenseNova uses the HF repeat-interleave GDN head
mapping and interleaved T/H/W coefficients. Existing Qwen3.5 keeps its default
head mapping and contiguous coefficients. Vision encoding remains on CPU;
Auto image requests can use the GPU decoder. The HTTP strict Vulkan runtime
retains its existing text-only contract.

The assigned container exposes only `/dev/dri/renderD130`, RADV NAVI31,
48 GiB VRAM. Strict native fixture runs failed before the first checkpoint
with `device lost / hard recovery`. An isolated source archive of the
**unmodified `5937066`** reproduced the same error with `vk_ops_check` on
ordinary Q8 operators. An existing small F32 device test also failed.
This is not a successful physical GPU run: full-size GPU parity, throughput,
VRAM residency and GPU failure/reset replay remain unverified. No GPU reset,
host reboot or another tenant's GPU/device mapping was changed.

The independent mHC test covers streams 2/4, three rows, identity and
nonidentity routing, all four modes and a separate f64 Sinkhorn reference.
The GDN test compares both default and repeat-interleave layouts against an
independent f64 scan. Both passed under llvmpipe (LLVM 20.1.2); these are
software shader correctness checks, not acceleration measurements.
A direct full-graph llvmpipe test also passed: a tiny dense+GDN model with
nonidentity mHC, shared physical layers, four independent visits, unequal
position axes, 3+2-row prefill, one-row decode and reset/replay. It compares
logits and the complete committed KV/conv/SSM shadow against CPU using
`abs_error <= 2e-4 + 2e-4 * abs(CPU)` for this small synthetic test. This is
separate from, and does not widen, the real-model oracle gates above.
Loop no longer runs the unrelated Qwen3 F16 attention-chain calibration:
its recorded dense attention is the separate Qwen3.5 F32 shader. Existing
Qwen3.5 routing retains its old gate, and activation contracts remain checked
according to weight format. Physical-visit validation, arena disjointness and
CPU-equivalent interleaved coefficient tests also passed. The final shared
Qwen3.5 regressions passed 107 tests with 19 ignored under the same software
ICD.

Auto's real-weight text fixture run encountered the queue failure, logged
`backend=Cpu`, and passed 570/570 unchanged oracle gates with exact greedy
IDs. The real-weight single-image Auto fallback passed 172/172 checks,
including the visual projection, and generated `[1919, 2099, 369]` exactly.
Text largest logits error remained `4.3392181396484375e-5`. This proves CPU
fallback parity and does not establish GPU parity. No tolerances were widened.

The three follow-up review findings are fixed:

- HTTP hybrid image input is accepted only in the final user message;
  historical images now fail explicitly instead of moving to another turn.
  System and history text are retained.
- SenseNova video generation fails before media/projector loading.
- Runtime preflight checks image count, projector presence, visual budget and
  expanded context before SSE headers and before vision encoder execution.
  Streaming and non-streaming invalid requests both return HTTP 400.

The real server was also started with Auto, the matching F32 projector and a
128-token context. Both streaming modes returned JSON HTTP 400 for an image
request requiring 181 expanded tokens and for a historical image (four
requests total); the test server was stopped afterwards. A real CLI strict
Vulkan request exited 1 with the device-loss error rather than falling back.

The related HTTP suite passed 77 tests. Dedicated expanded-context and early
video rejection tests passed. The default library baseline was 1322 passed /
16 failed / 88 ignored; after the fixes it is 1326 / 16 / 88, with exactly the
same failure-name set and no added or removed failures. The Python
converter/comparator suite passed 8 tests. Physical GPU completion remains a
required pre-merge check for this Draft PR.
The CLI, server and fixture example built with `parity-trace,vulkan`, and all
examples passed `cargo check` with the same features. Independent read-only
review found no further blocking issue; it did not independently run the
numerical checks and is not physical GPU evidence.

Existing CI maintenance was also verified: the four unchanged q4_k/q5_k/q6_k/
f16 shader binaries reproduce byte-for-byte with glslang 15.1.0, and their
stale manifest entries were refreshed without changing those binaries.
The shader `check` command passes with the CI-matching glslang 15.1.0 toolchain
on the remote host, and repository formatting passes. Mac glslang 16.6.0
accepts every manifest hash but produces different bytes for the unchanged
`softmax` shader; existing calibrated binaries were not regenerated with it.
