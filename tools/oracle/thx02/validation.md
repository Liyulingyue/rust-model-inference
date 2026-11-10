# THX-02 validation

**Experimental / partial adaptation.** Native Rust decisions, bundled vision,
optional causal thinking, CLI, HTTP and JEV are implemented. F32/F16/BF16 sample
readout checks passed. Every tested block format has a failed text logit gate;
Q6_K CPU, Q5_K CPU and Q4_K Vulkan also fail text probability gates. After the
F32 GELU correction, Q8_0/Q4_0 pass the single-image sample while K images still
fail logits. F16 thinker text and image pass eight-step CPU/Vulkan checks.
A working endpoint is separate from numerical acceptance. The GPU graph is
mixed CPU/device execution; unrun combinations remain unverified.

## Pinned inputs and reference

- ModelScope `HAL-X/THX-02`, revision
  `f17dca262dcd1d115e591424ce9cd012b51fc0db`.
- Unmodified official `thx02/{model,protocol,runtime,engine}.py` at that revision,
  loaded through `oracle.py`; native Transformers CPU F32 math. The isolated
  package loader bypasses only the unrelated SDK initializer.
- `model.safetensors`: 9,078,616,200 bytes,
  SHA256 `2e6585fe35a7c00afde5407b813f60f13cb7dbe0262a7a40e93bff12fb78976a`.
- `readout.safetensors`: 1,305,712 bytes,
  SHA256 `e8b24b4b9124c075fe939cb8cbbeb2da5ab15d0228c96b423dbbb9cf050f5dee`.
- All downloaded weights, tokenizer/config assets, converted GGUFs, requests,
  raw numerical outputs and device logs are under
  `/Users/gouzi/Documents/git/rust-model-inference/models/THX-02` (called
  `$assets` below). No weights are checked into Git.
- `converter-manifest.json` in that directory records completed file sizes,
  SHA256, tensor types and payload digests. `converter-image-bounds-update.json`
  records metadata-only repacking: existing tensor bytes and absolute offsets
  are unchanged, tokenizer JSON only loses insignificant whitespace. Reference
  provenance retains the actual historical GGUF hash used at execution time;
  it is not rewritten to imply execution of a newer hash.
  `converter-ftype-verified.json` verifies standard uint32 GGUF FTYPE and
  unchanged tensor payloads. `converter-thinker-escape-update.json` fixes only
  thinker protocol-body escapes to the decision encoder's ten tokens; both
  complete thinker tokenizer JSON documents remain intact.

The main checkpoint has 723 BF16 backbone tensors. Text uses 32 layers, width
2560, FFN 9216, Q/KV heads 16/4, head width 256, rotary width 64, interleaved
sections `[11,11,10,0]`, theta 1e7. Three causal gated-delta layers precede every
**noncausal** full-attention layer. The 255-way bias-free readout runs at
`<|fim_suffix|>`, calibrated temperature `1.5539092221462472`.

Vision has 24 layers, width 1024, FFN 4096, 16 heads, patch 16 / temporal patch 2,
merge 2 and learned 48x48 positions. Inputs accept up to four images, resized in
multiples of 32 to 65,536–1,638,400 pixels with Python ties-to-even rounding.
Thinking merges rank-32/alpha-32 LoRA, uses a causal native Qwen3.5 session and
F32 KV. MTP is an optional acceleration head and is not needed for greedy output.

## Acceptance rules fixed before inference

| Format | Logits max_abs | Logits RMSE | Probability max_abs |
| --- | ---: | ---: | ---: |
| F32 | 0.02 | 0.005 | 0.005 |
| F16 | 0.04 | 0.01 | 0.01 |
| BF16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K | 0.10 | 0.025 | 0.025 |

Prompt, IDs, mRoPE positions, readout markers and option/question order must be
exact. `compare.py` also checks answer types and labels. Thinker greedy IDs are
compared over eight cached steps, with full-vocabulary logit metrics and the
first divergent step reported. Limits are not widened after failed runs.

The gates above decide whether a format may be called numerically equivalent to
the reference. They are not a usability verdict: a block format can miss the
full-vocabulary logits gate and still return the same label in the same order,
which is what the caller sees. `compare.py` therefore also emits a decision
report (`usability` in its JSON): per answer, whether the argmax label moved
and how far the probabilities drifted, with a separate declared usability
criterion (`--usable-probability-drift`, default 0.01) that is not the
acceptance gate and never relaxes it. On the current evidence Q8_0 and Q4_0
keep the exact contract and stay inside the probability limit while missing
the text logits RMSE, so they are usable checkpoints that cannot carry the
"numerically validated" label; the K formats miss the probability limit
itself, which is the band where a decision can actually move.

Every precision has a same-weight reference. The independent NumPy decoder
restores source layouts without using exporter quantizers/transforms. Block
formats retain both a float-activation reference and a native input-QDQ
reference; the latter approximates the shared Q8_0/Q8_K activation contracts.
Neither probability agreement nor a CPU QDQ hook proves a different device's
kernel precision. `oracle-weight-quality-drift.json` reports weight quantization
loss separately from Rust implementation error.

Storage differs from computation: Rust uses F32 buffers, accumulation and
recurrent state. Shared prepared F16 projections round inputs to F16; other
F16 dispatch paths may retain F32 inputs. Block projections use shared Q8_0
or Q8_K activation blocks. BF16 text uses the existing F32-input SIMD kernel.
Neither buffer type nor storage type alone describes each operator's arithmetic.
There is no cross-request decision KV cache. Q4_0/K files keep vision in F32
because the shared vision loader lacks those block formats.

## Host and execution

2026-11-08, Apple M3 Max, 64 GiB RAM, 16 cores, macOS arm64; MoltenVK 1.4.2.
Rust uses `--profile release-fast`, four threads, existing SIMD/ComputePool;
Vulkan validation builds enable `--features vulkan,parity-trace`. Python uses the existing
repository `.venv` (torch 2.9.1, transformers 5.19.0). No Rust dependency or
external acceleration library was added.

CPU runs use shared operators. Auto/Vulkan submit eligible projections and
verified attention to the device. Gated-delta scan, norms, small projections and
shared attention fallback run on CPU. Strict Vulkan rejects requests with no
actual device submissions, but this check does **not** prove a resident GPU
implementation. Full-device execution remains **not implemented**, pending L2
shared hybrid dispatch/operator work.

Device counts are end-to-end evidence, not proof that the vision encoder is on
device. The corrected image probes submit 208 operations for F32/F16/BF16,
608 for Q4_0/K (F32 vision storage), and 26,084 for Q8_0. Vision offload depends
on the shared storage/kernel dispatch; per-component resident GPU coverage is
not claimed.

Timings collected alongside conversion, other inference or builds are
nonisolated observations. See [performance.json](performance.json) for same-host
CPU/Vulkan/official numbers, cold/warm differences, measured submissions and
limitations. Warm probe caches are not public HTTP latency: `Engine::decide`
reconstructs borrowed trunk/GPU weight wrappers and can upload again per call.

## Numerical evidence

Text cases cover choice/noul/score, structured JSON, Chinese/Unicode,
special-token escaping and multi-readout ordering. F32 and Q8_0 have three
text references; other formats have one. Each format has a fixed RGB image.
Text has no vision call and retains its previous numerical evidence. Every
image was rerun after the opt-in F32 GELU correction against its fixed
same-weight reference. Old full reports remain unchanged; combined current
reports are `f32-gelu-correction/combined-<format>-<compute>-vs-oracle.json`.

| Format | Text CPU / Vulkan | Text max_abs / RMSE | One image CPU / Vulkan | Image max_abs / RMSE |
| --- | --- | --- | --- | --- |
| F32 | Pass / Pass | 0.0000191 / 0.000005942 | Pass / Pass | 0.0001145 / 0.00004883 |
| F16 | Pass / Pass | 0.002688 / 0.0008861 | Pass / Pass | 0.001803 / 0.0004861 |
| BF16 | Pass / Pass | 0.00004864 / 0.00000796 | Pass / Pass | 0.0001147 / 0.00004775 |
| Q8_0, native QDQ | Fail / Fail | 0.099622 / 0.042006 | Pass / Pass | 0.079166 / 0.024571 |
| Q4_0, native QDQ | Fail / Fail | 0.152861 / 0.032746 | Pass / Pass | 0.077131 / 0.022255 |
| Q4_K, native QDQ | Fail / Fail | 0.187792 / 0.063015 | Fail / Fail | 0.225979 / 0.062837 |
| Q5_K, native QDQ | Fail / Fail | 0.208994 / 0.061132 | Fail / Fail | 0.171211 / 0.057117 |
| Q6_K, native QDQ | Fail / Fail | 0.327330 / 0.065512 | Fail / Fail | 0.166616 / 0.081075 |

Metrics are maxima across the stated samples/backends, rounded upward; exact
values and per-case pass/fail belong to the JSON reports. Auto public outputs
and device submissions are checked separately. No independent Auto full-logit
Oracle was run; the matrix keeps that numerical cell unrun and references the
explicit Vulkan counterpart for context.

| F16 thinker, eight cached steps | CPU | Vulkan | Largest max_abs / RMSE |
| --- | --- | --- | --- |
| Math + Chinese/special-token text | Pass | Pass | 0.006770 / 0.001430 |
| One image, corrected F32 GELU | Pass | Pass | 0.005538 / 0.001267 |

Q8_0 probabilities max_abs are <=0.003303 CPU and <=0.010233 Vulkan. Q4_0
probabilities max_abs are <=0.000526 CPU and <=0.002424 Vulkan. These pass their
probability limit but do not override the failed logits tests. Q4_0 also changes
one sample's score distribution by about 0.152 from full precision in the
independent float-activation Oracle; this is separate weight quantization loss,
not evidence of a Rust bug or a quality benchmark. Text K native-QDQ probability
max_abs failures are Q6_K CPU `0.054052`, Q5_K CPU `0.031347`, Q4_K Vulkan
`0.034261`, versus `0.025`. Same-weight float and QDQ reports are both retained.

Q8_0 checkpoint evidence localizes the first difference: embedding `[165,2560]`
is bitwise exact; layer 0 max_abs/RMSE is `0.00109053 / 5.7811e-5`, layer 31 is
`1.75075 / 0.0291811`, final norm `1.07066 / 0.0684132`. Full progression is in
`oracle-q8_0-native-qdq-first-divergence.json`. Input-quantizer rounding
boundaries and floating point accumulation are hypotheses, not a uniquely
established cause. Production SIMD was not replaced with scalar Torch math.

K-format case-0 embeddings are also bitwise exact. Their first saved difference
is layer 0: Q6_K `0.00227237 / 7.71167e-5`, Q5_K `0.00130260 / 4.57533e-5`,
Q4_K `0.00179553 / 4.21664e-5` (max_abs / RMSE). Full 32-layer growth is in
`oracle-<format>-native-qdq-first-divergence-final.json`; this is localization,
not proof that one particular rounding operation caused the error.

Thinker math, structured Chinese/special-token and image cases have exact
prompts, IDs, positions and eight greedy IDs on CPU and Vulkan (248320 logits
per step). The original image result failed at max_abs `0.143157`, despite
exact greedy IDs. Identical native pixel arrays and 298 matching vision tensors
ruled out preprocessing and weight payload differences. The earliest saved
visual output at `[66,2560]` differed by max_abs `0.02303457`, RMSE `0.000299868`.

A controlled native intervention changed only the 24 vision GELU inputs and
outputs to F16 rounding. The original native output exactly reproduced the
saved Oracle, while the rounded intervention closely reproduced Rust's error.
THX-02 now opts into the existing SIMD F32 GELU without changing other callers'
default. CPU visual error drops to max_abs `0.00196719`, RMSE `0.0000280871`.
The final full-logit image checks pass at `0.00553703 / 0.00106971` CPU and
`0.00539101 / 0.00126686` Vulkan. This overturns the old failed image conclusion;
the acceptance limits remain `0.04 / 0.01`, and the old failure files are retained.

Evidence: `oracle-thinker-image-preprocess-audit.json`,
`oracle-thinker-f16-image-visual-divergence-final.json`,
`thinker-f16-{cpu,vulkan}-vs-official.json` (historical),
`vision-gelu-audit/summary.json`,
`thinker-f16-image-f32-gelu-cpu-vs-official.json`,
`f32-gelu-correction/thinker-f16-image-vulkan-vs-oracle.json`.

An earlier F32 image probe used 130 tokens because the shared resize helper had
different bounds/rounding. It is retained as negative evidence. The local
preprocessor and GGUF image bounds were corrected; the accepted image has 168
tokens and exact reference positions. Golden resizing includes
`230x129 -> 352x192`, `208x400 -> 192x384`, `1920x1080 -> 1696x960`.
The four-image Vulkan smoke completed at 372 tokens, but has no four-image
numerical reference yet.

## Independent decision-level run (2026-11-10, Linux x86_64)

The acceptance gates above decide whether a format may be called numerically
equivalent to the reference. They are not a usability verdict, so the same
question was asked again at the decision level on this host, with the real
ModelScope checkpoint (`f17dca26…`, SHA256 verified), the F32 export as the
reference and three quantized exports measured against it.

The reference is the repository's own F32 export, not the official Python
runtime: this run answers "does the quantized checkpoint still decide the same
way", which is a weaker and more operational question than "is it numerically
equivalent to the official implementation".

| Export | Size | Decision flips | Largest probability drift | Reads as |
|---|---:|---:|---:|---|
| F32 | 18,190,652,320 B | reference | reference | reference |
| Q8_0 | 4,869,984,512 B | 0 of 3 | 0.009093 | usable |
| Q4_0 | 3,734,443,520 B | 1 of 3 | 0.193776 | changes a decision |
| Q4_1 | 3,997,283,680 B | 1 of 3 | 0.260803 | changes a decision |

Three native decision requests (a choice/noul/score request over a refund
message, a Chinese noul request over a structured state, and a choice request
over an arithmetic check) were run per export on CPU with eight threads. The
flipped decision is the same one in both 4-bit exports: the `urgency` score
moves from `High` to `Normal`. F32 gives `Low/Normal/High` =
0.0338/0.3268/0.6394 and Q8_0 gives 0.0316/0.3199/0.6485, both still `High`;
Q4_0 gives 0.0687/0.4857/0.4456 and Q4_1 gives 0.1018/0.5196/0.3786, both
`Normal`. The choice and noul answers of the same request move by at most
0.0011 in the 4-bit exports. Q8_0 keeps every decision and moves every
probability by at most 0.0091.

This matches the acceptance matrix rather than contradicting it: Q8_0's
probability gate passes and only its text full-vocabulary logits RMSE misses,
while the 4-bit formats miss the probability limit itself. The operational
reading is that the Q8_0 export is a usable checkpoint that cannot carry the
"numerically validated" label, and the 4-bit exports are not.

Wall clock on the same host (the three requests, 8 threads, CPU): F32
51.2 / 24.9 / 20.7 s, Q8_0 9.9 / 6.0 / 4.9 s, Q4_0 22.4 / 12.9 / 10.4 s.
Q8_0 is about four times faster than F32; the 4-bit exports are slower than
Q8_0 because their dequantization costs more and their vision weights stay
F32.

### Device path on this host

The same Q8_0 export was run on an NVIDIA L20 (46 GiB, driver-reported int8
dot product and shader f16 support), pinned to card 1 with `RMI_VK_DEVICE=1`,
with `--compute vulkan` and with `--compute auto`. Both compute modes print
`backend=partial arch=thx02`: the eligible projections and the verified
attention run on the device, while the gated-delta scan, the norms, the small
projections and the shared attention fallback stay on the CPU. A strict
`--compute vulkan` request that submits no device operator is rejected, and
these requests submitted 5 to 7.

| Request | CPU | GPU (Vulkan) | Decision agreement | Largest probability drift |
|---|---:|---:|---|---:|
| 0 | 9.91 s | 10.95 s | same labels | 0.00326311 |
| 1 | 5.97 s | 6.83 s | same labels | 0.00016376 |
| 2 | 4.86 s | 6.70 s | same labels | 0.00010793 |

`--compute auto` and `--compute vulkan` produce identical output. The device
counters show 41 MiB of peak device allocation for a 4.87 GB checkpoint, so
the offloadable share of this graph is small, and on this host the device path
is 10% to 38% slower than the CPU path for the same decisions. A full-device
path remains the L2 item below; this run does not change that.

The Q4_1 export additionally exposed a shared-kernel gap: the Q4_1 kernel had
no `embedding_lookup`, so a GGUF with a Q4_1 `token_embd` panicked with
"embedding_lookup not implemented for this kernel type" while every other
matmul-capable block type had one. `ops::embedding::embedding_lookup_q4_1`
and the Q4_1 kernel's implementation now exist and are unit-tested against
the block layout.

## Coverage matrix and entrypoint evidence

Implementation and verification are separate. Pass means the stated check ran;
Fail retains the failing metric/output; Not run is not a pass. Q4_1 now has a
shared exporter (see the L3 item below), but no checkpoint has been converted or
measured with it, so its row stays unrun rather than N/A. Other ungenerated
thinker precisions remain in scope.
Detailed per-component/format/requested-backend rows are in [coverage.csv](coverage.csv),
with CLI, HTTP, numerical and actual-backend evidence. [results.json](results.json)
retains file manifests, individual numerical gates and actual entry summaries.
The matrix separates public endpoint checks from numerical acceptance.

Public thinker checks exercise `think:true` text/image (up to 320 generated
tokens), `think:"auto"` (up to 96), thought return and the decision reread.
They compare CLI with real HTTP responses. The full-vocabulary numerical
Oracle covers eight cached steps; later public-generation tokens have no
independent full-logit reference.

The corrected F16 thinker public checks passed on CPU, Auto and strict Vulkan,
with 21 command records. `think:"auto"` actually generated a thought in all
three compute modes (96 output tokens / four sequences); direct text/image
thinking also returned thoughts and completed the decision reread. Across
eight CPU-versus-Auto/Vulkan comparisons, thought text and usage match exactly
and the largest probability difference is `0.0000846481`, below the fixed F16
limit `0.01`. Evidence:
`f32-gelu-correction/public-thinker-backend-comparisons.json` and the three
current `thinking-validation/summary-F16-<compute>.json` files.

`verify_entries.py` runs the actual existing CLI, starts the actual server and
saves each command, status, JSON response, timing and stderr. Every combination
checks native text, native image, JEV score and JEV image, then matches CLI with
HTTP batch and JEV probabilities (1e-5 tolerance, latency excluded). Auto and
Vulkan require nonzero device-submission evidence. Saved summaries belong to
`$assets/entry-validation/summary-<format>-<compute>.json`.

All 24 main format/backend endpoint combinations were rerun after the vision
correction and passed, with 192 command records. Each current summary carries
`vision_gelu: "F32"`. The complete prior directories, including old thinker
CPU results, are retained under `$assets/public-entries-before-f32-gelu/`.
The controller's `$assets/final-public-entries/summary.json` records the backup
mapping, source/binary hashes and completed combinations. It finished with all
24 main and three thinker combinations passing. The final consistency check
`$assets/final-validation-integrity.json` verifies the 213 command records,
current hashes, matrix states and retained numerical failures.

The first controller stopped on ENOSPC while saving Q4_0 CPU evidence after
12 completed combinations. Those completed results and the interrupted partial
directory were retained. Removing only this worktree's regenerable Cargo
`target/release-fast/deps` cache recovered space; both public executable hashes
remained identical and all model assets/numerical evidence were retained.
The resumed controller ran only unfinished combinations. Evidence:
`final-public-entries.log`, `final-public-entries-resumed.log` and
`final-public-entries/cargo-cache-cleanup.json`.

Historical reports may name the original canonical entry directories; use the
backup mapping to resolve those paths to the retained pre-correction outputs.
Current native image CLI and HTTP probabilities are also compared directly
with each same-weight Oracle in
`f32-gelu-correction/public-same-weight-probability-comparisons.json`.
All 54 comparisons pass: 48 image checks across eight formats/three backends
and six F16 text checks against the official runtime. All 32 native text/image
CPU-versus-Auto/Vulkan probability checks also pass their per-format limits.
Neither result overrides the failed quantized full-logit gates.

Continuation validation generated the merged BF16 thinker at
`$assets/THX-02-Thinker-BF16.gguf` (9,122,447,712 bytes,
SHA256 `6b52560957c379b92dbda2035d5bc94659e28c57db24d022b72a4482a783be38`).
The real CLI/HTTP thinker checks passed on CPU, Auto and strict Vulkan; each
record includes direct text/image thinking, auto-generated thought, HTTP batch,
HTTP auto-thinking and nonzero device submissions for Auto/Vulkan. The BF16
Oracle has three eight-step cases. CPU covers the text, structured-edge and
image cases; Auto/Vulkan cover the text case. Prompts, input IDs, mRoPE
positions and all greedy IDs are exact. The largest full-vocabulary error is
`0.00128603` max-absolute / `0.00021376` RMSE on the CPU image case, below the
predeclared BF16 limits `0.10 / 0.025`. Evidence is retained under
`$assets/continuation-bf16/summary.json` and the timestamped
`$assets/thinking-validation/BF16-*` directories.

Before the GELU correction, focused Q8_0/Vulkan checks passed 19 checks / 44
command records: stdin/local images, grouped JEV text/image, TTA, joint facets
with an odd context split, schema/media errors, 429 busy, context overflow and
four-image smoke. Evidence: `extra-validation/Q8_0-vulkan-20261108-023825/`.
These are historical functional checks; they do not establish corrected
four-image/grouped-image numerics. Forced missing-ICD text checks passed five
records: Auto matched CPU and strict Vulkan rejected the CLI/server invocation.
Evidence: `extra-validation/no-device-text-20261108-024501/`; no-device images
and thinker were not run.
At context 117 the pinned official tokenizer plans `[q0]` (91 tokens) and
`[q1,q2]` (117 tokens), with markers 81 / 81,107.

No streaming protocol is implemented because the model's decision contract is
nonstreaming. Shared validation rejects every supplied `stream` value other
than `false` for both CLI and HTTP, before model loading/forward. Audio/video are not part
of this checkpoint's accepted inputs. Chat routes are not mounted. These
contract absences do not remove applicable weight/device combinations.

## Engineering and change boundary

Baseline commit `1dd27a12ca534c1f4730bece6aa97149f76987cb`:
`cargo test --locked --profile release-fast --lib` gave **1475 pass / 21 fail /
96 ignored**. The final full lib run after the F32 GELU correction gave
**1499 pass / 21 fail / 98 ignored**.
All 21 failure names are identical; no new failures or hidden cancellations.
The full failure lists are in `$assets/regression-results.json` and
[engineering.json](engineering.json), with raw logs
`/tmp/thx02-baseline-tests.log` and `$assets/final-engineering-logs/`.
Focused Vulkan THX-02 tests: **14 pass / 2 ignored**. Post-correction shared
vision checks: **22 pass / 1 ignored**. The final `decision`-filtered lib run
passed **15** tests, including HTTP boundary checks; CLI integration passed
**9**. CLI streaming and HTTP truncated-image validation each executed
RED -> GREEN. Python checks: **54 pass** total, comprising **23 converter,
28 Oracle and 3 comparison** tests. These unit counts do not replace
real-weight runs.

The BF16 continuation rerun passes **58** Python tests (23 converter,
28 Oracle, 4 comparison and 3 thinker-controller tests), **15** decision-filtered
Rust library tests and **9** CLI integration tests. These results are recorded
separately in `engineering.json::continuation_validation`; the earlier test
counts and source/binary hashes remain historical evidence.

Changes are local to `models/thx02`, its converter/Oracle and capability-based
app decision adapters, plus explicit loader/module/backend/CLI registration.
The two Qwen3.5 forward helpers have visibility-only changes. The shared vision
encoder also gains an opt-in F32 GELU method: its private layer loop otherwise
prevents composing native GELU precision locally without duplicating the
encoder. It calls the existing shared SIMD `gelu_inplace`; original image/pair
APIs retain their original `gelu_ggml_f16_inplace` default. No model-specific
configuration/state field or new numeric kernel was introduced.

Shared vision callers include text/multimodal, omni, diffusion, Qwen-Drive,
THX-02 and Vela3/d3. A real legacy Qwen3.5-0.8B BF16 + F16 mmproj CPU
before/after sample retained all input IDs and the five printed visual values.
The baseline omitted greedy/seed controls, so different generated words cannot
establish a deterministic regression. Full legacy visual arrays/logits and
every shared caller were not compared; they are not claimed numerically
verified. Evidence: `$assets/qwen35-vision-regression/{before,after}.json`.
Cargo metadata was rechecked after the correction and still contains the same
nine binary targets; no model-specific executable was added.

## Remaining layers

- **L1, THX-02 owner:** complete the remaining thinker asset and validation
  matrix: thinker F32 and block-format files were not generated. Original
  checkpoints, eight main exports, F16 thinker and BF16 thinker are retained; free space is
  also needed for observed system swapping. This is an environment/asset gap,
  not a reduction of scope or a passed result. The initial asset-generation
  audit recorded 12,275,159,040 free bytes and an 8 GiB swap/work reserve in
  `$assets/remaining-thinker-disk-audit.json`. That reserve subsequently proved
  insufficient: public checks encountered ENOSPC while swap reached 18 GiB.
  The observed available-space drop rounds up to a 12 GiB reserve. The updated
  `$assets/remaining-thinker-disk-audit-after-enospc.json` records free space
  at that audit and each remaining export's estimate; the smallest additional thinker
  is 3,739,713,984 bytes, requiring 16,624,615,872 bytes with this reserve.
  No remaining format fit that audited headroom. BF16 was subsequently generated
  and validated in the continuation above; the old disk audit is retained as
  historical evidence rather than a current capacity check.
- **L2, shared hybrid GPU / quantization owners:** stable model-neutral
  noncausal resident-device interface, gated-delta device scan, batched vision
  dispatch and persistent weight wrappers. Resolve block-format logit divergence
  in the shared precision paths (F16 thinker image amplification was resolved
  by the opt-in F32 GELU correction),
  with checkpoint evidence before acceptance. Owners are the shared
  `src/ops/kernel`, Qwen3.5 vision and Vulkan hybrid dispatch modules.
  The existing resident causal Qwen3.5 decoder cannot express this graph through
  a stable reusable interface; copying it would fork another model's machinery.
- **L3, shared converter/quantization owner:** the missing Q4_1 export facility
  is now implemented in `tools/converter/utils/gguf.py` (`quantize_q4_1`) and
  registered in the shared `BLOCK_FORMATS`, with the GGUF writer taught the
  20-byte block size. Its layout follows the repository's runtime kernel
  (`ops::kernel::q4_1`): F16 scale, F16 minimum and unsigned 0..15 nibbles read
  back as `d * q + m`; the encoder is unit-tested against an independent
  decoder, and the THX-02 converter now accepts `--format q4_1`. What is still
  missing is the checkpoint side: no Q4_1 export has been converted or run
  through the numerical gates, and its logits are expected to sit in the same
  band as the other block formats.

Maximum context, broad task-quality datasets, other devices, no-device
image/thinker fallback, and thinker formats beyond the generated F16/BF16 assets
remain unverified. A partial result does not imply those checks passed.
