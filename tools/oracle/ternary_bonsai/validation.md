# Bonsai 2.5 27B validation

Acceptance is set before the full-model run. Compare all 248,320 logits at the
last prefill position and after three forced continuations. Require finite
values, cosine ≥ 0.995 and relative L2 ≤ 0.10 at each position, and compare all
four greedy IDs. The bounds cover native quantized reductions on the shared
64-layer Qwen3.5 trunk (see the [Holo4 calibration](../holo4/27b/README.md)).
Check the packed codes, FP16 scales, head permutation and tokenizer IDs exactly.
For a single normalized Hadamard block, require max absolute error ≤ 1e-5 on
bounded test inputs against a dense signed reference.

The matrix includes PTQ1_0, PQ2_0 and Q2_0, each with and without the optional
Q4_0 MTP head, and BF16/Q8_0 vision projectors. Exercise text, image and JEV
through CLI and HTTP, streaming and non-streaming generation, and CPU, Auto
and strict Vulkan. Record the actual execution backend separately.

## Short-context runs on 2026-11-26

The full-vocabulary comparisons and generation runs used commit `9cb45549`, an EPYC 9334, RADV NAVI31
and eight inference threads. All three base files were compared to the
publisher's CPU implementation using the same weights, five prompt IDs
(`760 6511 314 9338 369`), 256-token context, eight-row Rust prefill chunks
and F32 KV. Rust and the publisher predicted `11751 13 248046 198` at
all four positions in every run. Each decode step consumed the reference
continuation. The tokenizer also matched on empty input, mixed whitespace,
Unicode and explicit chat delimiters. [Per-position metrics, timings and exact IDs](performance.json)

| Decoder | CPU min cosine / max relative L2 | Auto min cosine / max relative L2 | Strict Vulkan min cosine / max relative L2 |
|---|---:|---:|---:|
| PTQ1_0 | 0.99998510 / 0.00559489 | 0.99998565 / 0.00538422 | 0.99998565 / 0.00538422 |
| PQ2_0 | 0.99997931 / 0.00678090 | 0.99998389 / 0.00587226 | 0.99998389 / 0.00587226 |
| Q2_0 | 0.99997824 / 0.00670976 | 0.99998600 / 0.00531991 | 0.99998600 / 0.00531991 |

Auto and strict Vulkan submitted 62 times at each text position. The CPU
submitted zero times. A 64×64 RGB red image produced nine projected tokens.
With the PTQ1_0 decoder, every projector/backend pair predicted the
publisher's first token `5844` (red). The image probe measured submissions
over preprocessing, vision encoding and target prefill.

| Projector | CPU cosine / relative L2 | Auto cosine / relative L2 | Strict Vulkan cosine / relative L2 | Auto / strict submissions |
|---|---:|---:|---:|---:|
| BF16 | 0.99998320 / 0.00579736 | 0.99997970 / 0.00693371 | 0.99997970 / 0.00693371 | 1,348 / 1,348 |
| Q8_0 | 0.99998083 / 0.00643347 | 0.99995590 / 0.01190903 | 0.99995590 / 0.01190903 | 4,282 / 4,282 |

For each format, all 851 target tensors in the ordinary and MTP files
matched by name, shape, type and payload bytes. Each MTP file adds 15
tensors in block 64. The named-payload digests in the results are SHA-256
over the sorted tensor names, each followed by a NUL byte and the SHA-256
of that tensor's payload.

## CLI, HTTP and device coverage

Each of the six decoder files ran through the shipped CLI and a loopback
HTTP server with `--compute cpu`, `auto` and `vulkan`. All 18 runs produced
`Paris` for the capital question and `Red` for the image. Each answer was
collected separately from CLI stdout, a JSON response and SSE deltas
terminated by `[DONE]`. The base files used the BF16 projector. The files
with a draft head used the Q8_0 projector. [Requests, answers and scores](entries.json)

| File | CPU | Auto | Strict Vulkan | Exercised |
|---|---|---|---|---|
| PTQ1_0 | Passed | Passed | Passed | Text and image generation, single and grouped JEV |
| PQ2_0 | Passed | Passed | Passed | Text and image generation, single and grouped JEV |
| Q2_0 | Passed | Passed | Passed | Text and image generation, single and grouped JEV |
| PTQ1_0-mtp | Passed | Passed | Passed | Speculative text and target-only image generation |
| PQ2_0-mtp | Passed | Passed | Passed | Speculative text and target-only image generation |
| Q2_0-mtp | Passed | Passed | Passed | Speculative text and target-only image generation |

With `--mtp-draft 2`, the CLI accepted three of six proposals in three
rounds for the one-word capital prompt in all nine file/backend pairs. A
second eight-token counting prompt on CPU gave the same output with ordinary
and speculative decoding for each format. All six proposals were accepted
in three rounds. The image CLI used each MTP file with speculation disabled.
A direct image request with the draft flag failed with
`speculative decoding requires a text-only prompt`. Servers started with the
draft flag served their image requests through the target model.

The final JEV runs used commit `30379492`. Single-choice JEV compared both
label probabilities and the selected label for text and image. Grouped JEV
compared hue and brightness separately through `/v1/jev/grouped` and
`/v1/jev/image_grouped`. The nine Bonsai base-file/backend combinations, a
PTQ1_0 CPU template override and a dense Qwen3.5-0.8B BF16 CPU regression
all passed. Across these 11 runs the largest CLI/HTTP probability difference
was 0 at the serialized precision (44 single-choice and 88 grouped
probabilities, tolerance 0.01). On PTQ1_0 CPU, prepending a fixed string
to the checkpoint template changed the grouped image brightness probability
from 0.93938524 to 0.95891809 in both interfaces. [Per-case results](entries.json)

With an unavailable ICD, Auto ran all three decoders with zero GPU
submissions and met the same oracle bounds at all four positions. Strict
Vulkan exited with `Unable to find a Vulkan driver`. On RADV, text and image
probes recorded the nonzero counts above. [Fallback measurements](performance.json)

Run `entries.py BIN_DIR MODEL MMPROJ RED.png LOG_DIR cpu|auto|vulkan`.
Add `--mtp` for a draft file, `--jev` to include the four scoring endpoints,
or `--scores-only` to repeat scoring. `--template-file PATH` applies the same
explicit template on both sides. Results are saved per response.

## Regression and review

Packed-code round trips cover all three block layouts, including PTQ1_0's
final two high-trit bytes. Separate tests check the signed GDN head gather
and the normalized Walsh transform against a dense float64 matrix. A
single-coefficient matrix checks the inverse embedding and both forward
activation quantizers against the expected signed Walsh column. A header
fixture rejects a missing transform and a tied v1 output before reading
any tensor. The loader shares the recurrent gather across all 48 GDN layers.
The projection test also checks that row lookup and a Vulkan upload leave the CPU
cache empty. After a CPU projection, an upload borrows the existing cache.
An image speculative-prefill test rejects token-only input before reading
the projector or decoder. Shader validation uses Vulkan 1.1.

On the Linux host, `cargo test --locked --profile release-fast --features
vulkan --lib FILTER` at `30379492` passed 126 tests for `qwen35` (24 marked
ignored), two for `app::jev::grouped::tests` and four for
`app::server::api::tests`.

At `a43c2d4a` the same filters passed 126 (25 ignored), two and four tests respectively. The `vulkan_q4_0_matvec_and_tiled_prefill_match_the_scalar_shader` hardware test passed on RADV with packed integer dots and again with `RUST_GPU_DP4A=0`. It covers ordinary and folded Q4_0, including an incomplete 32-token tile and weight rows containing 17 blocks. The long attention test also passed with a nonzero cache layer, causal row masking and the output gate.

`cargo metadata --locked --no-deps --format-version 1` reported the same nine binary targets and source paths at the PR base and the delivered runtime.

A real dense Qwen3.5-0.8B BF16 and its BF16 projector were run before and
after against base `7dfd5f15` on Apple M3 Max. CPU text (18 prompt IDs, six
generated tokens) and image (29 prompt IDs, red response) produced identical
inputs and output text. Before the FP64 capability fix, ordinary BF16
strict Vulkan failed during pipeline initialization on both revisions.
MoltenVK identified the `double local_sum` in PiVo acoustic attention.
Loading Qwen now skips that optional pipeline on devices without
`shaderFloat64`. On the M3 Max the original six-token strict-Vulkan run
returned the same text as CPU. The text/image CLI, HTTP JSON and SSE
returned `Paris` and `Red`. All single and grouped CLI/HTTP score
probabilities agreed (single-choice P(A) 0.84985191 and 0.97221237).
The physical-device PiVo test checked its explicit FP64 requirement.
The full Mac shader
rebuild also hit the same pre-existing binary difference in
`nemotron_moe_route.spv`; the new shader rebuilt byte-for-byte and passed
`spirv-val`. No 27B weight files were downloaded to the Mac.

The non-folded BF16 decoder SHA-256 was
`cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5`.
On the Mac with explicit `--no-thinking --jinja`, the CLI, HTTP and SSE returned
`Paris` for the short-answer text prompt and `Red` for the red image.
CLI and HTTP JEV returned A with probabilities 0.85150671 (text) and
0.97095829 (image). With an unavailable Vulkan ICD the Auto text result
matched CPU. On Linux at `30379492`, the same 0.8B decoder returned
`Paris` and `Red` across all three interfaces. The CLI and HTTP single-choice
probabilities were 0.85438418 (text) and 0.97294360 (image). The grouped
results are in the `dense_qwen35_0_8b_bf16` entry. [Linux results](entries.json)

## 8,192-token retrieval

The fixed archive prompt gives a recovery phrase near the beginning and requests it at the end. All three publisher decoders returned `The capital of France is Paris.` before the first end-of-message token. The comparison uses the final prefill position and 15 forced continuations, including the chat delimiters after that answer. It applies the same cosine and relative L2 limits to each full-vocabulary row. The probes also check every input token ID against the publisher. [Reproduction commands](README.md#8192-token-retrieval-and-forced-continuation)

The production CLI and `/v1/chat/completions` ran the same prompt with PTQ1_0 on strict Vulkan. The embedded chat template produced 8,204 input tokens with thinking disabled. Both returned `The capital of France is Paris.` at temperature zero. HTTP reported 8,204 prompt tokens and seven completion tokens. These runs used a context of 8,320, a prefill batch of 128, F32 KV and eight threads.

Each cell gives the minimum cosine and maximum relative L2 over 16 positions. All 144 greedy token IDs matched. The CPU runs used `ad24c11b`. The Auto and strict Vulkan runs used `a43c2d4a`. Auto and strict Vulkan recorded 3,968 device submissions for 64 prefill batches and 62 at each decode position. CPU recorded zero. [Per-position logits metrics, token IDs, timings and submissions](performance.json)

| Decoder | CPU | Auto | Strict Vulkan |
|---|---:|---:|---:|
| PTQ1_0 | 0.999404 / 0.034935 | 0.996380 / 0.087993 | 0.996380 / 0.087993 |
| PQ2_0 | 0.999355 / 0.036049 | 0.999859 / 0.016775 | 0.999859 / 0.016775 |
| Q2_0 | 0.999866 / 0.016382 | 0.998060 / 0.064680 | 0.998060 / 0.064680 |

At `0532ffa4`, strict Vulkan chose the same 16 tokens as the PTQ1_0 reference, but position 14 had relative L2 0.125917. In a trace at that position, the first inspected layer output differed by 0.000132 relative L2. The difference grew from 0.01644 at layer 39 to 0.23702 at layer 47. As a separate check, changing only the publisher's attention implementation gave a maximum logit relative L2 of 0.092659 on the same continuation. Its fixed reference reproduced exactly. With eight F32 accumulators per folded projection at `a43c2d4a`, the PTQ1_0 maximum fell to 0.087993 on strict Vulkan.

The eight-stream shader also passed the short text probe on Auto and strict Vulkan for all three formats. The 24 greedy IDs matched, with a minimum cosine of 0.999957729 and a maximum relative L2 of 0.009307787. Across all three decoders, both vision projectors and all three compute policies, each of the 18 red-image comparisons selected token 5844. Their minimum cosine was 0.999928177 and their maximum relative L2 was 0.015193841. The CPU image measurements use `ad24c11b`.

The delivered CLI, HTTP JSON and SSE returned `Paris` for the text question and `Red` for the image with PTQ1_0/BF16 on strict Vulkan, PQ2_0/Q8_0 on Auto, and Q2_0/BF16 on strict Vulkan. Single and grouped CLI/HTTP JEV matched for all 36 text and image probabilities in these runs. All three MTP files returned `Paris` on strict Vulkan, with three of six drafts accepted in each run. The Q2_0 MTP server with the Q8_0 projector also returned the expected text and image answers in JSON and SSE.

## Single-token Vulkan decode

The delivered folded projection at `a43c2d4a` accumulates eight F32 partial dots across the full input width. Ordinary dense Q4_0 weights use the packed matvec from `ad24c11b` for single-token decode on supported devices. The RADV run compared the folded projection to `79af30ed` with the same PTQ1_0 file, F32 KV, one-token input and eight publisher-forced decode steps. Both revisions matched all nine greedy IDs, with 62 submissions at each position. The delivered path had minimum cosine 0.99989178 and maximum relative L2 0.01948461.

| Nine single-token positions | Baseline | Eight-stream folded projection |
|---|---:|---:|
| Q4_0 GPU time for 2,745 projections | 2,589.40 ms | 821.11 ms |
| Sum of timed GPU dispatches | 2,911.78 ms | 1,149.47 ms |
| Wall-clock decode, three-run median | 2.359 tok/s | 4.431 tok/s |

Three alternating uninstrumented runs gave 2.359, 2.336 and 2.409 tok/s on the baseline and 4.410, 4.440 and 4.431 tok/s on the delivered path. Each rate divides eight decode evaluations by their combined elapsed time, excluding model loading and prefill. The shared host had a different load during the earlier measurements below. Compare wall time within each paired run.

### Earlier Q4_0 matvec comparison

On the EPYC 9334 / RADV NAVI31, runtime `ad24c11b` and baseline `79af30ed` ran the same PTQ1_0 file, F32 KV, eight threads and a one-token prompt (`1`, ID 16). Both evaluated the prefill position and eight publisher-forced decode positions. All nine greedy IDs agreed with the publisher. The lowest full-vocabulary cosine was 0.99996835 and the highest relative L2 was 0.00908625.

The timestamp run bracketed each dispatch with Vulkan queries, collected results after fence completion and multiplied the raw ticks by the device's 10 ns timestamp period. The compute queue reported 64 valid timestamp bits. Matching the pipeline fingerprint to the checked-in SPIR-V identified 2,745 Q4_0 projection dispatches in each run.

| Nine single-token positions | Baseline | Q4_0 matvec |
|---|---:|---:|
| Q4_0 projection GPU time | 2,591.70 ms | 204.20 ms |
| Sum of timed GPU dispatches | 2,919.22 ms | 531.58 ms |
| Q4_0 share of timed dispatches | 88.78% | 38.41% |
| Wall-clock decode, three-run median | 0.353 tok/s | 0.473 tok/s |

Wall-clock runs disabled the timestamp layer and alternated baseline/new, new/baseline, baseline/new. Each run divided eight decode evaluations by their total elapsed time. The host carried other inference workloads. Individual baseline runs ranged from 0.352 to 0.409 tok/s and the new runs from 0.468 to 0.553 tok/s. Both runtimes submitted 62 times at every position.

On Apple M3 Max, three strict-Vulkan runs per revision generated the same text in 24 tokens on each dense Q4_0 model. Median decode throughput rose from 12.88 to 16.65 tok/s with Qwen3.5-4B and from 40.8 to 57.2 tok/s with Qwen3-0.6B. Forcing scalar int8 with `RUST_GPU_DP4A=0` retained the Qwen3 output. A device test compares the scalar projection with the selected kernel at one, two, three and five input rows, including a 17-block input and 129 output rows. It allows absolute and relative FP32 error.

On the same Mac, `cargo test --locked --profile release-fast --features vulkan --lib -- --test-threads=4` reported 1,841 passed, 22 failed and 261 ignored on both revisions. The 22 failing test names were identical. The explicit Q4_0 physical-device test passed.

The full library suite at the PR base `7dfd5f15` reported 1,818 passed, 39 failed and 261 ignored. Once the optional FP64 pipeline was fixed, running the server GPU flag test before either the Yue2 CPU convolution or the Q8 CPU partition test reproduced a change of backend through the process-wide GPU switch. Pinning the CPU tests to their intended policy gave 1,845 passed, 18 failed and 262 ignored at `a43c2d4a`. All 18 failures also occurred at the PR base.

Using the same retrieval prompt on an M3 Max with an ordinary Qwen3.5-0.8B BF16 decoder, the embedded chat template produced 8,204 input tokens. Both CPU and strict Vulkan returned `The capital of France is Paris.`. With F32 KV, 128-token batches and six threads, CPU prefill took 2,715.21 s and Vulkan prefill took 256.77 s. The long-context attention, MTP head, speculative decode and SenseNova loop cache tests also passed on that device.
