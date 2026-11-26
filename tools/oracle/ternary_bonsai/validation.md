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

## Full-model runs on 2026-11-26

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

A real dense Qwen3.5-0.8B BF16 and its BF16 projector were run before and
after against base `7dfd5f15` on Apple M3 Max. CPU text (18 prompt IDs, six
generated tokens) and image (29 prompt IDs, red response) produced identical
inputs and output text. The ordinary BF16 strict-Vulkan run reported the
same shader initialization failure on both revisions. The full Mac shader
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
