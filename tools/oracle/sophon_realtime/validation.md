# Sophon-Realtime-32B-Tag validation

Status: **Experimental, partial adaptation**. F32 thinker, F32/BF16 audio and short F32/BF16 speech components have numerical passes. Standalone F32 CLI TTS has emitted a valid waveform. Later integration fixes and remaining precision/entrypoint checks are recorded separately; none of those component passes establishes native FP8, full-duplex or complete GPU support.

ModelScope revision: `990d6a74bbf15c1765a6f7736fa839f423fd5b59`.
Official Sophon reference: `541d83c29ca68dd5235d2e344a2b72d29e6299c2`.
Official DuplexJev reference: `b851ea57ba48b351bb9381d6c16d6d10619a9d6b`.
Run host: authorized Linux/x86-64 EPYC 9334, AMD gfx1100 / 51,522,830,336 VRAM bytes, 2026-11-11.
Model assets and generated weights stay on the remote host under `/workspace/rust-model-inference/models/`.

## Contract and boundaries

The thinker is the released 64-layer Qwen3-VL decoder, with 5120 hidden values, 64 query heads, 8 KV heads, 128 head dimensions, 25600 intermediate values and 151936 vocabulary rows. Its vision tower has 27 layers and DeepStack layers 8/16/24. The explicit Sophon source adapter validates the thinker contract and then exposes it to the unchanged Qwen3-VL implementation. The model-local adapter must not alter other architectures or decode weight types.

The source is compressed FP8 E4M3FN with channel scales and dynamic token-wise activation quantization. Converted GGUFs use standard weight-only kernels. Widening source weights is not a claim of native FP8 execution; source FP8 activation loss, export rounding/quantization and implementation error must be reported separately.

The audio component uses an 18-layer portable Qwen3-ASR encoder and an Ultravox stack-2 projector. Audio positions advance as ordinary text positions; they do not receive visual DeepStack values. The talker is CosyVoice3 with tag-aware speech tokens, a four-step distilled flow (no CFG), HiFT and per-voice prompts. No new executable target is permitted.

## Acceptance gates (declared before inference)

Tokenizer IDs, tensor shapes, source pins, special tokens, component matching and stream ordering are exact contracts. Floating-point reductions, FMA/SIMD, F16/BF16 storage and block quantization are assessed numerically, without a bitwise equivalence requirement.

For the thinker, compare every vocabulary logit after prefill and eight cached decode steps against independent PyTorch execution of the **same exported weights**, with identical token IDs/positions and teacher-forced continuation. Record max absolute error, RMSE, relative RMSE, cosine, and greedy IDs/margins. F32: relative RMSE <= 0.001, cosine >= 0.999999. F16/BF16: relative RMSE <= 0.005, cosine >= 0.99998. Block-quantized kernels: relative RMSE <= 0.03, cosine >= 0.999. These approximate gates follow the adjacent Qwen3-VL WeVisOne validation precedent; they are not claims of task quality. Greedy divergences require a first-divergence checkpoint/margin and a separate task-quality assessment. Evaluate export degradation against the widened source in addition to same-file implementation error.

The audio reference executes the pinned standalone portable Qwen3-ASR encoder and the pinned Ultravox projector operations with the same independently dequantized GGUF. Before the first Rust comparison, mel gates are max absolute error <= 0.001, relative RMSE <= 0.0001 and cosine >= 0.99999999; encoder and projector gates use the thinker precision table above. These gates allow the existing FFT/SIMD reduction differences and separate quantized activation error from export degradation. Compare a short clip, the 100-frame CNN boundary, the 800-frame attention boundary and the 3000-frame processor boundary. Remove only per-chunk reference padding when comparing packed Rust checkpoints; do not remove real silence or tail samples. These component checks do not establish speech understanding or speech quality.

Flow/HiFT gates were fixed before their first Rust comparison: every intermediate has relative RMSE <= 0.001 and cosine >= 0.999999; waveform has relative RMSE <= 0.01, cosine >= 0.9999 and max_abs <= 0.01. These same-file component gates are retained for all requested precisions. Task/perceptual quality needs separate evaluation. No threshold was relaxed after a failure.

## Numerical evidence

The previous opening statement claiming no numerical or end-to-end result is superseded by these measured results. The earlier WIP entries were never acceptance claims.

| Scope | Measured result | Gate/status | Evidence |
| --- | --- | --- | --- |
| Thinker F32, 15 prompt tokens, eight cached steps | max rRMSE 5.26640777e-6; max_abs 6.49094582e-5; min cosine 0.999999999986; eight greedy IDs equal | PASS, same weight-only GGUF, F32 KV | [thinker-results.json](thinker-results.json) |
| Audio F32/BF16, short and both sides of CNN/attention/processor boundaries | 14/14 cases; max mel rRMSE 7.819e-6; encoder 1.28043e-4; projector 1.01639e-4 | PASS | [audio-results.json](audio-results.json) |
| Audio F16 | runtime-QDQ encoder rRMSE 0.001481; projector 0.001094 | PASS for runtime-QDQ; plain F32-activation reference FAIL remains visible | [audio-results.json](audio-results.json) |
| Quantized audio | encoder/projector final gates fail for all six block formats | FAIL; do not infer acceptance from same-input projections | [audio-results.json](audio-results.json) |
| Talker F32 | all 81 logits rows/controls/greedy decisions; 77 speech tokens and EOS; max rRMSE 2.66328396e-6 | PASS | [talker.md](talker.md) |
| Short Flow/HiFT F32 | all 15 checkpoints; flow mel rRMSE 9.44775e-6; waveform 1.32167e-5 | PASS | [precision-results.json](precision-results.json) |
| BF16 talker and short Flow/HiFT | same-file component checks pass; autonomous talker matches F32 | PASS in covered fixture | [precision-results.json](precision-results.json) |
| F16 talker | numerical rows pass; step 51 greedy mismatch, runtime-QDQ reference margin 0.00071907 | Overall FAIL under unchanged discrete gate | [precision-results.json](precision-results.json) |
| Quantized Flow/talker | six block Flow formats fail at velocity checkpoint 0; talker results vary | FAIL/Experimental; Q4_K talker strict fixture passes but autonomous output reaches cap without EOS | [precision-results.json](precision-results.json) |
| Extended F32 nonfinal | 15 checkpoints pass; max rRMSE 1.46377e-4 | PASS | [precision-results.json](precision-results.json) |
| Extended F32 64-frame fixture | HiFT stage0 rRMSE 0.00110145034 > 0.001; final waveform rRMSE 0.00360077 / max_abs 0.000727881 passes waveform gate | Overall FAIL; final waveform does not hide intermediate failure | [precision-results.json](precision-results.json) |
| Standalone F32 CLI TTS, brand_zh | 77 speech tokens, EOS, 73,920 samples at 24 kHz; WAV SHA256 9a1e2129d2922916e0c596e9ac5e52d9f9fad7c7e9d89fecf53c185b48a9e80d | Executed; initial nonstream binary before subsequent fixes | [performance.json](performance.json) |

Q8 same-input projection probes pass 12 audio operations (maximum rRMSE 4.9271e-7) and the real talker first-layer Q/K operations. They narrow the diagnosis to accumulated activation/nonlinear/rounding boundaries, but do not establish every kernel or end-to-end quality. The Q8 audio export also fails its original per-tensor conversion budget at layer0.fc2 (rRMSE 0.0105698 > 0.01). Implementation error, export degradation and source FP8-activation semantics remain separate.

## Coverage matrix

[coverage.json](coverage.json) enumerates each capability × format × compute policy × supported KV format, with independent CLI/HTTP/numerical states. `Implemented / not run` is not a verification pass. `strict rejected` is error handling, not device support. Component checkpoint tests do not stand in for CLI/HTTP inference.

| Capability | CLI | HTTP | Implementation | Verification boundary |
| --- | --- | --- | --- | --- |
| Text generation | ordinary prompt/Jinja path | chat/completions, messages, responses; SSE and batch | Implemented | F32 eight-step independent Oracle passes; final HTTP matrix pending |
| Vision | --image / --mmproj | ordinary image chat / image JEV | Existing Qwen3-VL runtime with strict Sophon component validation | Final real image matrix pending |
| Audio understanding | --audio / --mmproj audio component | /v1/audio/chat, batch and SSE | Implemented | Component boundaries pass; final end-to-end matrix pending |
| Text/image candidate scoring | --jev | /v1/jev/score, /v1/jev/image | Explicit thinker adapter; shared scorer | Final CLI/HTTP probability comparisons pending |
| Audio candidate scoring | --jev --audio, single and grouped scoring | /v1/jev/audio, single schema | CLI single/grouped and HTTP single implemented; shared audio encoder and resident logits | Local prompt/probability/error checks pass; real matrix pending |
| Tagged speech output | --tts, packaged --voice | /v1/audio/speech; WAV/PCM, batch and PCM SSE | Implemented | F32 CLI and short component results above; final stream/HTTP checks pending |
| Speech-to-speech | --audio plus --tts-model/--tts-mmproj | /v1/audio/chat with output_audio:true | Implemented as audio-to-text followed by synthesis | Final real entrypoint verification pending |
| Combined audio+image/video | explicit rejection | separate audio/image requests | Not implemented, layer 2 | Not run |
| Live full-duplex input/output, text starvation timing, interruption | no session protocol yet | no duplex session protocol yet | Not implemented, layer 2 | Not run |
| Native dynamic FP8 activations | standard weight-only GGUF | same | Not implemented, layer 2 | Not run; widened FP8 weights are not native FP8 execution |
| Grouped candidate scoring | --jev-multi / --jev-block for text/image/audio | /v1/jev/grouped and /v1/jev/image_grouped; audio HTTP remains single schema | Implemented with shared grouped core, resident runtime and pinned template; audio HTTP grouped remains layer 2 | Local real handler tests pass; final model matrix pending |
| Complete Vulkan | runtime capability gate | same gate at construction/request | Not implemented, shared capability layer 3 | Audio/speech strict rejection does not establish support; text/image resident decoder must be measured with F16 KV |

All nine requested formats are F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K. Eligible matrices use the repository's existing SIMD/prepared-row kernels; accumulation is F32, with per-kernel input QDQ. HiFT stays F32 in every requested export because its tensors do not meet the shared block-quantization policy. Vision is the source's F32 projector export. Thinker KV supports F32/F16; talker KV is F32. No new format, backend library or binary target was added.

## Shared changes and review fixes

- F32 static weight borrowing is explicit opt-in on `TensorSource`, default false. Only the Sophon thinker adapter enables it, and the unchanged Qwen3 model retains its adapter/source Arc. Owned behavior remains the default for FunASR/FSMN and other existing callers. A lifetime/drop regression test covers the default path; a real Qwen3 F32 baseline/current comparison is recorded separately.
- Shared audio upload decoding drains stdin/stdout/stderr concurrently, acquires request admission before expansion, and limits decoded samples before allocation grows beyond the budget. The no-ffmpeg PCM16 fallback now checks the same budget before reserving decoded storage. The existing unbounded public WAV decoder retains its behavior.
- `RealFft` adds spectrum/inverse operations for HiFT without altering the existing power-spectrum path.
- Shared Q4_K scale initialization fixes a nonzero-constant block reconstructed near 0.06669 instead of 1.00036. Independent payload tests cover the defect; old-model conversion/inference regression is tracked separately. Q5_K/Q6_K sampled encodings are unchanged.
- Streaming commits mel rows once; a later Flow recomputation cannot overwrite already emitted history. HiFT still replays committed history and emits only new PCM, a documented performance shortcut pending a bounded phase-carry implementation.
- AudioChat uses the common text-backend accessor for ordinary chat and JEV. Single text/image/audio and grouped text/image scoring share resident logits, template and context/KV options, with invalid input (400) separated from runtime failures (500). Sophon text/image request preflight counts expanded image tokens before the encoder or KV allocation. Streaming and batch synthesis check cancellation per speech token, Flow block/Euler step and HiFT stage/resblock. AudioChat batch carries the existing disconnect guard through text callbacks and subsequent speech; audio encoder/prefill internals are not newly interruptible.

## Checks and artifact identity

The pristine same-host library baseline was 1625 passed / 22 failed / 222 ignored. The first implementation run had 1637 passed / the identical 22 failed names / 223 ignored. Historical failures are not new passes. Final-source results and the final artifact identities must be consulted before accepting later integration changes.

After an unrelated full `/tmp` caused additional file-creation failures, both builds were rerun with repository-local `TMPDIR` and `--test-threads=1`: pristine **1624 passed / 23 failed / 222 ignored**, reviewed build **1658 passed / the identical 23 failed names / 225 ignored**. No failure was hidden by comparing totals. This result is tied to the build before the final grouped-JEV/batch-cancellation follow-up.

Local final follow-up checks include the server suite (109/109), audio grouped/single scoring (2/2), grouped templates (2/2), audio upload regressions (15/15), F32 ownership behavior, streaming mel/noise history, CLI component-pair validation, audio-JEV prompt/probability/nonfinite handling and HTTP 400/500 mapping. The Python Oracle suite was 29/29, then 31/31 after explicit thinker runtime-QDQ and cached/full-prefix calibration coverage. `cargo check` for both existing application binaries, formatting and diff-whitespace checks pass in the covered source revision.

A baseline checkout and the current checkout briefly shared a Cargo target directory: an apparently successful current build reused the baseline CLI SHA. That artifact is explicitly rejected as current-source evidence. Final builds use an isolated target directory, verify the binary hash/source markers, and keep protected baseline/current copies. No extra Cargo executable target is introduced.

## Remaining layers and order

1. **This layer:** strict component contracts, all nine converters, weight-only CPU paths, standalone and composed CLI/HTTP entrypoints, independent references, precision/error evidence and review fixes. Experimental status remains while failed and pending rows exist.
2. **Model/session follow-up, adaptation owner:** combined media, audio HTTP grouped schema, live duplex/starvation/cancellation scheduling, bounded HiFT phase state, native FP8 activation support and broader speech/perceptual quality. Preserve the pinned scheduler/noise contracts and repeat the same numerical gates.
3. **Shared runtime/backend owner, after layer 1 contracts:** audio block-attention/frontend kernels, causal convolution, STFT/iSTFT and HiFT device orchestration. The existing Qwen3-VL resident decoder already implements interleaved M-RoPE and DeepStack with F16 KV; model-specific validation and any failures are separate from missing audio kernels. Exercise CPU/Auto/strict Vulkan with actual submission evidence and same-host reference timing, then repeat the full precision and entrypoint matrix.

These layers describe remaining work, not a reduction of the intended model scope. The model is only fully adapted after all applicable layers and acceptance rows pass. [performance.json](performance.json) contains measured validation timings; paging/concurrent work and release-fast builds prevent deployment-capacity or GPU speedup conclusions.

The earlier draft incorrectly classified shared Qwen3-VL/DeepStack Vulkan support as missing. Source review confirmed the existing resident decoder implements both and requires F16 KV; the matrix now separates that implemented path from the still-missing audio/HiFT device orchestration. Hardware visibility or eligibility alone is not a validation pass.
