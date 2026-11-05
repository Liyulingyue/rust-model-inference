# Hy-SimulMT merge review — 2026-11-01

Reviewed with `.agents/skills/reviewing-model-changes/SKILL.md` on Apple M3 Max,
MoltenVK 1.4.2, Rust 1.98.1, `release-fast`, and the repository's native CPU flags.
The merge joins `fdb0f239` with main `b93f43d9`. The review covers the complete
Hy change relative to that main, including the moved operators and new upstream
callers. JEV is explicitly outside this model's supported contract.

## Findings and fixes

| Priority | Location | Reproduction and disposition |
| --- | --- | --- |
| P1, fixed | `src/vulkan/ops.rs:3191`, `:3277` | Group output widths `[1,65,129]`, input width 4, all-one weights/input: Auto returned 0 instead of 4 at column 64 of the 65-wide output. Dispatch used only the first projection's width. Both float recorders now dispatch the maximum width. The new device regression covers F32/BF16/F16, CPU fallback versus actual GPU selection, and changing row counts. The ARM F16 path uses its ordinary recorder; x86 tiled F16 execution was not rerun locally. |
| P2, fixed | `src/compute/linear.rs:359` | A four-projection F32/BF16 group returned four CPU backends in Auto; forced Vulkan would reject it. Main's LFM3 attention actually calls `run_group` with four projections (`src/models/lfm3/trunk/forward.rs:58`). Groups larger than three or with different numerical contracts now retain the previous sequential GPU path; eligible homogeneous groups retain one submission. The new device test also covers two mixed-format projections. |
| P2, fixed | `src/app/server/mod.rs:2211` | Main changed the generic Vulkan mode validator to warn. The test expected its old error before even identifying a Hy model. It now supplies a minimal `hy_v3` GGUF and checks either the missing-feature error or the model's explicit hybrid-only error. No global validator behavior was reverted. |
| P2, fixed | `tools/oracle/hy_simulmt/README.md`, `verification.json` | The guide still advertised single/grouped Hy JEV after `fdb0f239` removed it. Current support now says unsupported; old entry measurements remain explicitly historical. |

Before the operator fixes, both new device tests failed. Afterward both passed,
and all six focused device tests passed. The comparisons use absolute error
`< 1e-5` for representable integer-valued projection fixtures; no new lossy
whole-model bitwise gate was added.

## Layering

The model graph, audio encoder, independent mixed dense/MoE decoder, prompt
protocol and KV session live in `src/models/hy_simulmt/`. Hy does not enter the
Qwen3 trunk; `tests/hy_simulmt.rs` explicitly checks that `qwen3_arch_knobs`
rejects `hy_v3`. The CLI lives in the existing `src/app/asr/mod.rs`, with no
model-specific ASR file. It only loads input/options and calls the model.

HTTP uses a dedicated `Backend::HySimulMt` and capability-named audio routes.
The enum matches and literal `if let`/`matches!` backend checks were audited:
the generic text/JEV handlers are not mounted on this backend. The real Auto
server check verifies the text, grouped and image JEV routes return 404.

## Five coverage matrices

### 1. CLI / HTTP / JEV

| Entry | Fresh evidence | Boundary |
| --- | --- | --- |
| Main CLI | Complete Standard Q4_K_M, JFK 3.2 s, target English, greedy, one output token, four threads, F32 KV: CPU and two Auto runs exit 0 and return `And`. | A smoke test, not long-recording accuracy or throughput. |
| HTTP CPU | Both real-weight ignored tests pass: JSON/multipart equality, target language, 200/429 concurrent admission, invalid language 422, unsupported format/temperature 400; final EOT in an exactly-sized F16 KV context. | HTTP is CPU-tested on this complete Standard checkpoint. |
| HTTP Auto | Real JSON upload returns the same `And`; device counters prove execution. | One short request, no long-running GPU server soak. |
| JEV | `hy_text_jev_is_rejected` and `hy_grouped_jev_is_rejected` pass; actual Hy server JEV routes return 404. | Unsupported by design; neither language nor image JEV was added. |

### 2. CPU SIMD

| Area | Evidence | Boundary |
| --- | --- | --- |
| Hy sessions/contract | `cargo test --lib hy_` participates in the full suite; fresh `RMI_SCALAR=1 cargo test --lib hy_`: 20 passed. | Scalar and SIMD synthetic session consistency; no fresh whole-model Oracle run. |
| Q6_K NEON DotProd | The signed-scale/block-boundary regression passes in the full ARM suite; tolerance is `2e-5 * max(abs(reference),1)`. | Other models' real Q6_K weights were not exhaustively rerun. |
| Shared F32/F16 dots | Reduction and tail regressions pass; new ggml-layout helpers are used by Hy attention. | New x86 AVX2 execution relies on CI/historical evidence, not this ARM host. |

### 3. Components and multiple weights

| Contract | Evidence | Boundary |
| --- | --- | --- |
| Bundled components | Strict decoder metadata, tensor shapes/types/lengths, speech projector metadata, F32 audio tower and 2048-wide decoder validation; missing/wrong/truncated tensor regressions pass. CLI/server reject extra `--mmproj`. | Self-contained single GGUF; no external speech-tower pairing. |
| CPU groups | QKV/gate-up share activation preparation through `LinearExecutor`; synthetic sessions cover dense and routed/shared experts, batch sizes, F32/F16 KV and continuation. | Arithmetic equivalence is bounded, not universally bitwise. |
| GPU groups | Six device tests pass, including mixed widths, two mixed formats, four projections, thresholds, error fallback and existing F16/BF16 numerical modes. | The real model checked here is Standard Q4_K_M with Q4_K/Q6_K/Q8_0 decoder matrices and F32 audio. |
| Other exports / Interleave | Earlier complete Standard/Interleave Q4_K_M numerical records are retained in `verification.json`. | No local Interleave GGUF or other complete quantization export; fresh per-export token/logit parity is unverified. |

### 4. CPU / Auto / Vulkan

| Policy | Fresh evidence |
| --- | --- |
| CPU | Complete model CLI returns `And`; CPU HTTP and F16 KV boundary tests pass. |
| Auto | Two complete-model CLI runs and an HTTP request match that CPU output. Each records 401 actual submissions, 298 Vulkan projection reports and peak device allocations of 2,321,449,472 bytes. |
| Missing/failing device | Tiny Auto CPU selection and row-threshold switching pass; injected command failure falls back only for Auto. A fresh-process `RMI_SCALAR=1` test proves Auto and Vulkan selection avoid device initialization under scalar mode. |
| Forced Vulkan | Hy constructor/session explicitly reject a full-device request. The server mode regression passes in both feature configurations. Generic shared linear execution still preserves forced-device behavior for supported projections. |

### 5. Shared impact

| Shared change | Callers and proof |
| --- | --- |
| `dreamx/kernels.rs` → `ops/nn.rs` | DreamX, Mage-Flow, Astronex-World, Flux3, Qwen-Drive, Unlimited-OCR, WeVisOne and Vulkan kernel helpers keep the compatibility re-export. Comparing the complete moved file against current main yields only the new import plus `binding`/`forward_group`; existing implementations, including main's retry-cache/BF16-dot fixes, match. |
| Audio frontend | Qwen3 ASR/Omni/TTS, Audio8, Gemma4 ASR, shared media readers and ASR/TTS app callers keep the old re-export. The original mel path still uses the exact `2^-24` floor and does not drop a frame. Hy alone selects the new Whisper floor/frame behavior. |
| Audio transformer helpers | Qwen3 ASR/Omni and Hy share the moved helpers. Nine function bodies match main exactly after excluding a removed unused `maximum` local; validation and reduction order are retained. |
| `LinearExecutor` | Hy, Qwen3 dense prefill/MoE, Gemma4, LFM3, Flux3 and Auk use the shared executor. The review fixes preserve non-groupable GPU calls through the original staging path. Full unit suites exercise synthetic callers; real-weight parity of each model remains unverified. |
| Tokenizer / Q6_K | Hunyuan-Dense gets its pinned three-pass pretokenizer; `hunyuan` keeps its original scanner through a separate enum variant. Other pretokenizers' code is unchanged. Q6_K adds an ISA-gated operator implementation with scalar fallback; real-model numerical bounds outside Hy were not re-established. |

## Commands and regression baseline

Commands below use `RUSTUP_TOOLCHAIN=1.98.1`, `CARGO_TARGET_DIR=target/rmi`,
`--locked --profile release-fast`. Full-suite baselines use an isolated
`git archive fdb0f239` under `target/hy-merge-review`, with the same model fixtures.

| Command / configuration | Before merge | Reviewed merge |
| --- | --- | --- |
| `cargo test --lib -- --test-threads=1` | 1304 passed / 20 failed / 84 ignored | 1340 passed / 20 failed / 88 ignored; **identical failure names** |
| Same with `--features vulkan` | 1409 passed / 24 failed / 193 ignored | 1445 passed / 24 failed / 199 ignored; **identical failure names** |
| `cargo test --test hy_simulmt --test hy_simulmt_entries` | — | 5 passed, 2 real-weight tests normally ignored; both additionally run and pass |
| `RMI_SCALAR=1 cargo test --lib hy_` | — | 20 passed |
| Vulkan `--lib linear_gpu_group_ -- --ignored --test-threads=1` | Two new tests fail before fixes | 2 passed |
| Focused Vulkan linear ignored device tests, excluding scalar-only test | — | 6 passed; scalar-only test separately passes with `RMI_SCALAR=1` |
| `cargo build --features parity-trace,vulkan --bin rust-model-inference --bin rust-model-server --example hy_simulmt_parity` | — | Passed |
| `.venv/bin/python -m unittest discover -s tools/oracle/hy_simulmt -p test_compare.py` | — | 4 passed |

Logs, the full failed-name sets, source-body comparisons and fresh smoke records
are stored locally under `target/hy-merge-review/`. Fmt/diff/conflict checks pass
before the merge commit. Incidental upstream rustfmt differences in
three newly merged app files were normalized with the same toolchain.

## Documentation and remaining limits

`docs/MODEL_LIST.md`, `docs/develop/SUPPORTED_MODELS.md`, the usage guide and this
Oracle guide describe audio translation, an independent decoder and experimental
hybrid Auto. Main's other new models and `lfm3` registry entry are preserved.
The older measurements remain dated historical records; JEV claims were corrected.

- **Needs numerical follow-up before stable Auto support:** historical Interleave
  three-turn Auto logits have 25.84% relative RMS and cosine 0.97505, outside the
  unchanged 20% / 0.98 hard limits, despite matching greedy IDs. Native text
  diagnostic failures are also retained; they do not add a supported JEV mode.
  Reproduce with the two complete variants and pinned independent Oracle paths
  in `README.md`; local Standard one-token agreement does not resolve this.
- **Needs driver investigation:** historical NVIDIA 570.153.02 operator processes
  segfaulted in an exit callback after assertions. M3 Max device tests and CLI
  exits do not prove that driver issue fixed.
- Long recordings, microphone input, other complete quantization exports,
  x86 tiled F16 execution, fresh whole-model logits and other models' full
  weight matrices were not revalidated. Fresh smoke wall times include setup
  and concurrent builds, so they are not a performance comparison.

## Local follow-up — 2026-11-01

The following results supersede the corresponding merge-review boundaries above.
They use the same M3 Max / MoltenVK host, four threads, F32 KV, complete Q4_K_M
weights, and the pinned Oracle at `e41ca421`. The local Oracle uses native NEON
with Accelerate, BLAS, Metal and Vulkan disabled. Each comparison retains the
independent Oracle weight-repacking calibration and the original numerical caps.

### Shared Vulkan lifetime

Rust's static `OnceLock` did not run the shared context's destructor. A new
isolated-process regression registers an audit before Vulkan initialization:
before the fix, its computation passed but the audit aborted because device
allocations remained live. This reproduces missing cleanup locally; it does
**not** establish the cause of the historical NVIDIA SIGSEGV.

The shared context now registers process cleanup after warmup. Cleanup releases
the global attention, convolution and Q4_K runtimes before the context, waits
for device idle, and destroys cached pipelines before destroying the device.
The loader stays loaded for the driver's own finalizers. Context shutdown is
idempotent, rejects later command submissions, and retains resources if idle
completion cannot be established. Validation-feature arrays now outlive
`vkCreateInstance`; constructor errors also release the debug messenger.

Cleanup also exposed a dangling context in the pre-existing Q4_K cache: it
extended any borrowed context to a static lifetime, including a private stack
context used by the smoke test. A local macOS crash report traced the exit
SIGSEGV to that cache's buffer teardown. Only the singleton now uses the global
cache; private contexts get a call-scoped runtime. The smoke test fails before
the fix with 7,952 bytes retained and passes afterward with no retained bytes.
Two complete Vulkan suite processes then finish with ordinary test-failure
exit 101, rather than SIGSEGV, and identical baseline failure names.

All nine focused device regressions pass, including the new three-child exit
audit, explicit double-shutdown rejection, uncertain-resource retention and the
existing group/format/threshold/fallback cases. Run the exit checks with:

```sh
cargo test --locked --profile release-fast --features vulkan --lib \
  ops::float::tests::shared_vulkan_context_cleans_up_before_driver_exit \
  -- --exact --ignored --nocapture
cargo test --locked --profile release-fast --features vulkan --lib \
  vulkan::tests::vulkan_shutdown_drains_device_and_rejects_later_submissions \
  -- --exact --ignored --nocapture
```

### Standard numerical and entry checks

| Configuration | Generated IDs | Maximum logits relative RMS | Minimum logits cosine |
| --- | --- | --- | --- |
| CPU | All 10 match | 8.0043% | 0.997359 |
| Auto, two separate processes | All 10 match on both runs | 7.9228% | 0.997254 |

All 12 audio checkpoints also pass (maximum relative RMS 0.07532%). Before the
subsequent MoE correction below, the lifetime-only change preserved every CPU
and Auto checkpoint payload byte-for-byte against its respective pre-cleanup
run. Those separate measurements remain in `verification.json`; the table
above records the final MoE implementation using numerical tolerances.

The main Auto CLI and both HTTP upload formats produce the same complete
three-turn Chinese text as the library. The CLI records 760 device submissions,
555 Vulkan projection reports and 2,321,449,472 peak allocation bytes. The HTTP
JEV endpoints still return 404. The real CPU multipart/JSON/concurrency/error
regression and the F16 final-EOT context-boundary regression both pass.

### Interleave numerical correction and device gate

The complete Interleave GGUF was completed from the approved official HF mirror
at revision `2aae4719eb91c1e6990c9386f70b2610f106b97e` and its full SHA256 was
verified as `26c87f0a5f6409f3a57f8c4d2a49d58ad0f5c0753141643789e73564339cedbc`.
The pinned Oracle and independent repacked run set the unchanged logit bounds:
max absolute error 4.955625, relative RMS 0.20 and cosine 0.9852752325.

Fresh pre-fix CPU logits failed that cosine floor (0.983804), and unguarded
Auto failed relative RMS (20.1244%). Hy MoE used exact per-lane SiLU while the
pinned Oracle uses vector SwiGLU. It now uses the same existing SIMD SwiGLU
entry as Hy's dense FFN. A four-lane Oracle fixture uses an absolute tolerance
of `1e-7`; whole-model approximate math is not required to match bitwise.

The unguarded GPU decoder still fails the calibrated cosine floor after that
correction (0.982959). Interleave Auto therefore uses a CPU decoder and retains
GPU audio encoding, with an explicit startup reason. Standard's hybrid Auto
policy is unchanged. Both retain SIMD, grouped weights and normal device-failure
fallback; full Vulkan decoding and JEV remain unsupported.

| Final configuration | Generated IDs | Maximum logits relative RMS | Minimum logits cosine |
| --- | --- | --- | --- |
| Interleave CPU, two separate processes | All 18 match | 14.3136% | 0.989940 |
| Interleave Auto, two separate processes | All 18 match | 13.6124% | 0.992466 |
| Interleave scalar CPU | All 18 match | 18.6624% | 0.986624 |

All 12 Interleave audio checkpoints pass (maximum relative RMS 0.09087%). Auto
records 163 submissions and 150 Vulkan dispatch reports for audio, with peak
device allocation 1,324,639,680 bytes. The real-weight Auto regression compares
CPU and Auto using each actual variant, without a metadata override. The main
CLI and both HTTP upload formats are also checked against library output.

The final full default suite has 1341 passed / 20 failed / 88 ignored; Vulkan
has 1446 passed / 24 failed / 201 ignored. Both failed-name sets exactly match
the corresponding merge baseline, with no added or removed failures. The
Vulkan process exits normally with 101 for those known assertions. The extra
passing test is the MoE fixture. Detailed local logs and numerical artifacts
are in `target/hy-fix-review/`.

NVIDIA 570.153.02 was not retested, at the user's request to validate locally.
The Khronos validation layer is absent on this host; a validation-enabled
initialization reports that missing layer, so validation-layer coverage is not
claimed. No GPU speedup or long-recording coverage is established.


## Main integration 2026-11-04

Merged main `f4aa9d15526ec3af828c8c93f9bb943b45069e08` into Hy head
`247e8fe2d8f638a4c9e233c71c3d63ca484bb5e2` on the same local M3 Max / MoltenVK
host. Main's Audio8, AuK and Timestral fixes remain included. Both restored
Q4_K_M weight files were rechecked against their recorded complete SHA256s.

### Resolution and shared contracts

- Resolve the five conflicts in shared linear/Vulkan, ARM quantization and
  tokenizer code. Keep both tokenizer regressions and main's Q4/Q5/Q6 kernels,
  including the Q6 signed-scale/truncated-block regression.
- Unify the output-slot constructor while retaining checked allocation/slot
  arithmetic and descriptor capacity for grouped bindings. Preserve Hy's
  homogeneous grouped shader, main's mixed-format single-submit recording,
  maximum-output-width dispatch and existing error/cleanup behavior. Larger
  groups reuse eight slots in batches; device regressions cover 2/4/7/9
  projections and nonuniform F32/BF16 storage.
- Main's NEON F16 shader optimization reassigned inputs among accumulators;
  the existing 256-wide regression failed by three ULPs. Restore its original
  lane stride while keeping main's multiple-output-row workgroups. The original
  CPU reduction contract passes without changing the test or its expectation.
- Main's tiled F16 shader now accepts transposed weights. Preserve row-major
  matmul/convolution callers too: only the explicit transposed recorder selects
  that storage layout. A device test covers both layouts, 33 token rows,
  65 outputs, 130/258 inputs and partial tiles with the same numerical bound
  `abs <= 1e-4 + 2e-5 * abs(reference)`.
- Keep fault injection on the new batch recorder and validate the whole linear
  group before any upload/output write. Auto fallback, forced-device errors,
  shared/private-context lifetime and exit cleanup regressions all pass.
- Main added an optional AuK source to `build_backend`; update the two Hy server
  mode tests to pass `None`. Hy remains in its independent model backend and
  existing ASR entry, with no Qwen3-trunk integration or new model-named app file.

### Complete model and entry validation

Fresh three-turn traces use the same independent pinned native-NEON Oracle and
repacking artifacts as the November 1 follow-up. No calibration or hard cap was
changed. Each run used a new trace directory, four threads and F32 KV.

| Configuration | Generated IDs | Maximum logits relative RMS | Minimum logits cosine |
| --- | --- | --- | --- |
| Standard CPU | All 10 match | 8.6197% | 0.996390 |
| Standard Auto | All 10 match | 8.3869% | 0.997445 |
| Interleave CPU | All 18 match | 11.4664% | 0.993425 |
| Interleave Auto | All 18 match | 15.4427% | 0.989140 |

All 12 audio checkpoints per run also pass. Standard's logit bounds remain
max-abs 2.977287, relative RMS 0.1514963 and cosine 0.9942769; Interleave's remain
4.955625, 0.20 and 0.9852752325. Quantized/SIMD arithmetic uses these bounds,
while generated IDs, checkpoint names/order/counts/shapes are exact checks.

Both variants pass their three real-weight regressions: actual CPU/Auto output
and GPU submissions, HTTP concurrency/uploads/options, and final-EOT F16 KV
capacity. Each Auto CLI and Auto HTTP JSON/multipart response matches the full
library translation. Separate route probes check `/v1/jev/score`, `/grouped`,
`/image` and `/image_grouped` under `/v1/jev`: all return 404 for both variants.
The Interleave startup still names its CPU-decoder precision gate.

Standard CLI records 568 submissions, 555 Vulkan backend reports and peak
allocation 2,631,209,472 bytes. Interleave records 163 submissions, 150 Vulkan
reports and peak allocation 1,325,069,760 bytes, all for its audio tower. These
are functional counters; no timing/speedup claim is made. The wider shared
output workspace increases Standard peak allocation from the preceding
2,321,449,472-byte run.

### Regression and remaining boundaries

Commands use `RUSTUP_TOOLCHAIN=1.98.1 CARGO_TARGET_DIR=target/rmi`,
`--locked --profile release-fast`, and one test thread for full/device suites.

| Check | Result |
| --- | --- |
| Default full lib, before → after | 1341 passed / 20 failed / 88 ignored → 1369 / 21 / 89 |
| Vulkan full lib, before → after | 1446 passed / 24 failed / 201 ignored → 1474 / 25 / 204 |
| Vulkan linear ignored tests, excluding scalar-only selection | 6 passed |
| `vulkan::ops::tests::batched_linear_device_` ignored tests | 7 passed |
| Shared exit audit / explicit shutdown / private-context smoke | 1 passed each; exit audit checks three child processes |
| `RMI_SCALAR=1` Hy / scalar-only linear selection | 19 passed / 1 passed |
| Normal Hy integration/entry tests | 5 passed; 3 real-weight tests normally ignored |
| Real-weight entry tests, Standard and Interleave | 3 passed each |
| Both binaries and parity companion | Built with `parity-trace,vulkan` |
| Oracle comparison-checker tests | 4 passed |
| Shader manifest / SPIR-V validation | All hashes pass; all 54 binaries pass `spirv-val --target-env vulkan1.1` |

Failed-name comparison retains every prior failure and adds only main's new
`app::media::tests::stereo_uploads_are_refused`. A separate untouched
`git archive` of `f4aa9d15` reproduces that exact assertion: FFmpeg is installed
and the supported upload path mixes stereo down, while the test expects an
error. `src/app/media.rs` is byte-identical to main; this is not a merge
regression. Both full suites exit normally with 101 for those assertions.

The complete shader rebuild/byte-comparison script also fails on unchanged
`softmax.spv` on both untouched main and the merged tree with this local
compiler. Its hash and SPIR-V validation pass. The two edited F16 shaders were
rebuilt and validated; a full byte-rebuild pass is not claimed.

Full records and failed-name sets are in `verification.json`; local logs and
traces are under `target/hy-main-2026-11-04/`. No NVIDIA, other complete
quantizations, long-recording or controlled-performance rerun was performed.
Interleave GPU decoding and JEV remain unsupported under the existing policy.

## Review fixes 2026-11-04

Validated locally on Apple M3 Max (arm64, NEON and DotProd), using Rust 1.98.1
and `release-fast`. The reference remains `e41ca4212f786815faa37606ee2e6fcf8611f9e4`.
Both complete GGUF SHA256 values were rechecked. No new calibration, wider
precision threshold, Qwen3 decoder dependency or Hy JEV implementation was added.

### Review disposition

| Item | Change or evidence |
| --- | --- |
| M1, scalar dispatch | The existing `has_avx2_fma()` / `has_neon()` gates already consult scalar mode. Added direct F32/F16 ggml-dot scalar regression and ran the Hy suite with process-start `RMI_SCALAR=1`; no duplicate dispatch branch was added. |
| M2, legacy GPU opt-in | Restored `gpu_requested()` before legacy context initialization. A child-process regression proves an unenabled legacy Auto scope does not initialize Vulkan. Explicit `ComputePolicy::context()` selection remains available. |
| M3, layer kind | Validation and loading both choose dense/MoE from each layer's router tensor, including multiple leading dense blocks. |
| M4, tied output | Missing output weights reuse the embedding. An existing malformed output still fails; regression compares tied and explicitly copied heads. |
| M5, metadata | Added reference defaults for optional sigmoid gating, norm, scale and shared width. Norm=false and scale=0 have their actual reference behavior. Expert width accepts scalar/array and zero fallback; **the key remains required** in the pin's `get_key_or_arr` call. Its loader uses element zero for every block, including nonuniform arrays. RMS epsilon is required; missing RoPE base defaults to 10000. |
| M6, arena | The output head now uses a separate one-row executor. Arena and weight descriptor ranges are checked against `max_storage_buffer_range` before binding, and batched weights before upload. No general arena allocator redesign was needed. |
| M7, descriptors | Added a bounded grouped-set cache: recover/confirm device completion before freeing group sets, retaining single-weight sets and uploaded buffers. Failure to confirm completion retains the cache. The device regression traverses all ordered pairs of four weights twice with a seven-set budget. |
| M8, LayerNorm loading | Removed F32-only restriction, restoring `load_f32_tensor`'s decodable types and preserving finite-value/length validation. F32 and BF16 load regressions pass. F16 is not in that helper's pre-existing accepted set and is not newly claimed as supported. |
| M9, F32 grouping | Choose small/tiled kernels per projection, including mixed `[65,4097,67]` widths. On-device grouped outputs equal the same single projections bitwise at 1/17/33 rows. This tests dispatch invariance, not universal bitwise floating-point parity. |
| M10, group bounds | Reject more than three grouped weights and more than three descriptor extras before allocation/upload, without changing the larger independent-dispatch batching API. |
| M11, Hy-MT2 | Added seven official 1.8B tokenizer ID cases covering Han/Kana, combining marks, long digits and English punctuation/whitespace. The sparse fixture preserves 418 vocabulary IDs and 176 original merge ranks; source hash and oracle method are recorded in the fixture. |
| M12, F16 P·V | Retained input narrowing: the pin's F16 matmul converts F32 src1 to F16 (`ggml-cpu.c` type traits and mul_mat conversion). The independent graph probe gives F16×F32 = F16×F16 = -1.14203799, versus unrounded F32×F32 = -1.1419822. It does not accumulate in F16. |
| M13, hygiene | Renamed the trimmed log-mel entry and its test module, removed unused `erff`, cached successful mel filters and replaced DreamX-specific shared errors. Shared frontend traces now use `audio.*`. Public WAV types require the frontend module to stay public; a glob `nn` re-export would conflict with existing exports. Falcon's dot retains its distinct F64 tail/non-x86 behavior, documented at the function instead of silently changing its arithmetic. |

The ARM F32 reduction objection was also checked against the pin's
`ggml-cpu/simd-mappings.h`: its reduction combines `(acc0+acc2)+(acc1+acc3)`,
as the existing Rust NEON path does. Local DotProd is present, and the existing
Q6_K NEON/DotProd regressions actually execute on this host. No unsupported
hardware path was substituted for these checks.

### Complete model comparison

The same independent native-NEON Oracle and repacked baseline directories from
the preceding follow-up are reused, with fresh actual traces, four threads,
F32 KV, and the complete three-turn audio fixture. All 12 audio checkpoints per
run pass. Standard's 10 generated IDs and Interleave's 18 IDs match exactly.

| Configuration | Maximum logits relative RMS | Minimum logits cosine |
| --- | --- | --- |
| Standard CPU | 8.6197% | 0.996390 |
| Standard Auto | 7.7182% | 0.997224 |
| Interleave CPU | 11.4664% | 0.993425 |
| Interleave Auto | 13.4625% | 0.991932 |

The Standard gates remain max-abs 2.977287, relative RMS 0.1514963 and cosine
0.9942769; Interleave remains 4.955625, 0.20 and 0.9852752325. The CPU runs also
retain all 22 Standard and 30 Interleave checkpoint payloads byte-for-byte from
the preceding main-integration run. Interleave GPU decoding stays gated to CPU;
its Auto audio tower still runs on Vulkan.

Both variants pass all three real-weight regressions: CPU/Auto translation and
actual submissions, HTTP uploads/concurrency/options, and the F16 final-EOT KV
boundary. Ordinary Auto CLI and both HTTP upload formats match library output;
the four JEV routes return 404. Standard still records 568 submissions and 555
Vulkan reports, with peak device allocation reduced from 2,631,209,472 to
2,168,470,528 bytes (441.3 MiB saved). Interleave retains 163 submissions, 150
Vulkan reports and 1,325,069,760 peak bytes for its audio tower. These are
functional counters, not a controlled speed benchmark.

### Regression scope

- Default lib: 1369 passed / 21 failed / 89 ignored before, 1377 / 21 / 89 after.
- Vulkan lib: 1474 passed / 25 failed / 204 ignored before, 1484 / 25 / 207 after.
  Both final failed-name sets exactly match this review's fresh baseline.
- Explicit physical-device tests: 10 batched-runtime and 6 linear-executor
  regressions pass, including all three new review device tests. Scalar Hy:
  24 pass; scalar dot and device-free scalar selection: one pass each.
- Normal Hy contract/entry tests: 5 pass, with the three real-weight tests
  explicitly run separately. The numerical Oracle runs use `parity-trace,vulkan`;
  CLI/HTTP and real entry checks use ordinary `vulkan` binaries with scalar and
  trace environment variables cleared.
- CI now explicitly runs the new group regressions on Lavapipe, plus scalar dot,
  Hy metadata/decoder, Hy-MT2 tokenizer, audio and descriptor-bound tests.
- All shader hashes and all 54 SPIR-V validation checks pass. The local full
  rebuild check still differs at unchanged `softmax.spv`, identically reproduced
  on the untouched main archive with glslang 16.6.0; a local full byte-rebuild
  pass is not claimed. This review changes no shaders or SPIR-V.

Non-16 kHz mono WAV remains an explicit rejection, now checked separately from
stereo rejection; automatic resampling was not introduced. Other complete
quantizations, long audio, NVIDIA/AMD hardware and controlled speed benchmarks
were not rerun. The model remains Experimental under its existing error budget.
Detailed machine-readable evidence is appended to `verification.json`; local
logs and traces are under `target/hy-review-2026-11-04/`.
