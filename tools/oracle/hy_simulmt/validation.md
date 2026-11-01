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
