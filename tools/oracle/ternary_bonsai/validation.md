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

Both runs used an EPYC 9334, RADV NAVI31 and eight inference threads. The
operator probes use a 256-token text context, eight rows per prefill chunk
and F32 KV. The five input IDs for `The capital of France is` were
`760 6511 314 9338 369` on both sides. The publisher and Rust produced
`11751 13 248046 198` at all four positions for each of the six runs below.
The decode steps consume the reference continuation even if a prediction
differs. [Step-level results and timings](performance.json)

| Decoder | CPU min cosine / max relative L2 | Vulkan min cosine / max relative L2 | Vulkan submissions per step |
|---|---:|---:|---:|
| PTQ1_0 | 0.99998510 / 0.00559489 | 0.99998565 / 0.00538422 | 62 |
| PQ2_0 | 0.99997931 / 0.00678090 | 0.99998389 / 0.00587226 | 62 |
| Q2_0 | 0.99997824 / 0.00670976 | 0.99998600 / 0.00531991 | 62 |

A 64×64 RGB red image produced nine projected tokens. Both projector files
were compared against the publisher's CPU mtmd implementation with the PTQ1_0
decoder. The first predicted token was `5844` (red) on all four runs.

| Projector | CPU cosine / relative L2 | Vulkan cosine / relative L2 |
|---|---:|---:|
| BF16 | 0.99998320 / 0.00579736 | 0.99997970 / 0.00693371 |
| Q8_0 | 0.99998083 / 0.00643347 | 0.99995590 / 0.01190903 |

## Entry and device coverage

| Input | Implementation | Observed on the 27B |
|---|---|---|
| Text, three base files | CLI and HTTP use the existing hybrid runtime | CPU and strict Vulkan numerical probes above. Auto completed the three four-step probes on RADV. The Auto dumps were not compared before the instance disconnected. |
| MTP, three draft files | Shared NextN loader and text-only greedy speculation | PTQ1_0-MTP strict Vulkan CLI accepted 4/4 drafts in two rounds and returned the same six tokens as ordinary generation. PQ2_0-MTP and Q2_0-MTP entry runs are pending. |
| Image, BF16 and Q8_0 | Shared vision encoder and hybrid decoder. CLI accepts an explicit thinking setting. | Both CPU and strict Vulkan passed the full-vocabulary probe. Image CLI ran on PTQ1_0. The original CLI ignored the explicit thinking setting for media. A corrected 27B entry run is pending. |
| JEV, text and image | Shared scorers with the server's configured template | PTQ1_0 CPU text CLI selected A (red), probability 0.99988544. The 27B HTTP score and image-score comparisons are pending. |
| HTTP chat and SSE | Existing OpenAI-compatible server | Q2_0 served its model ID. The inference host disconnected before the generation and streaming responses could be collected. |

`entries.py BIN_DIR MODEL MMPROJ RED.png LOG_DIR cpu|auto|vulkan [--mtp] [--jev]`
runs the two executables serially. It compares CLI, HTTP JSON and SSE text,
checks the draft count when requested, and compares JEV choices and label
probabilities (absolute difference below 0.01). The server is bound to
loopback. Each completed result is written before the next request.

## Regression and review

Packed-code round trips cover all three block layouts, including PTQ1_0's
final two high-trit bytes. Separate tests check the signed GDN head gather
and the normalized Walsh transform against a dense float64 matrix. A
single-coefficient matrix checks the inverse embedding and both forward
activation quantizers against the expected signed Walsh column. A header
fixture rejects a missing transform and a tied v1 output before reading
any tensor. The loader shares the recurrent gather across all 48 GDN layers.
An image speculative-prefill test rejects token-only input before reading
the projector or decoder. Shader validation uses Vulkan 1.1.

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
With explicit `--no-thinking --jinja`, the CLI, HTTP and SSE returned
`Paris` for the short-answer text prompt and `Red` for the red image.
CLI and HTTP JEV returned A with probabilities 0.85150671 (text) and
0.97095829 (image). With an unavailable Vulkan ICD the Auto text result
matched CPU. These are shared-entry regression checks on the 0.8B decoder.
