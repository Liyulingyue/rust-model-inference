# NariQ v1 validation

On 2026-11-26 the AMD EPYC 9334 / RADV NAVI31 host downloaded the three GGUFs and the BF16 projector directly from the pinned ModelScope revision. All four SHA-256 digests matched. The comparisons below use the same file on both sides, with F32 accumulation in the trellis projection and F32 KV in Rust. The image encoder runs on CPU. The Auto and Vulkan sessions offload the decoder. [File identities and reproduction commands](README.md)

## Tokenization

The text prompt `The capital of France is` produced `[760, 6511, 314, 9338, 369]` in Rust and the Nari Labs fork. The official Qwen tokenizer and Rust also matched on all six fixtures at each tier, including empty input, whitespace, Chinese, emoji, digits, special tokens and composed/decomposed accents. The model declares NFC. For decomposed `Cafe\u0301`, both the official tokenizer and Rust emitted `[34, 2492, 933]`. The fork's BPE emitted `[34, 5468, 52033]` for the decomposed string and `[34, 2492, 933]` after NFC. Its NFC implementation is used by its unigram tokenizer. Official `tokenizer.json` SHA-256 is `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`. [Qwen tokenizer revision](https://modelscope.cn/models/Qwen/Qwen3.8-27B/resolve/1098534ab5d7220ea0f4a6b9f07bb03729a79c1d/tokenizer.json) · [Reference tokenizer](https://github.com/nari-labs/nari-llama.cpp/blob/2644f7de461c27675eef373faa1776f20b393004/src/llama-vocab.cpp)

## Full-vocabulary comparisons

Each text cell covers the last prefill position and two teacher-forced decode positions, 248320 logits per position. Each image cell compares the first prediction on the fixed 64×64 red image, resized to 96×96 and represented by nine visual tokens. The gates require finite logits, exact greedy IDs and KL(reference ‖ Rust) ≤ 0.001. Relative L2 and maximum absolute error are diagnostics. The three slash-separated KL values are **CPU / Auto / strict Vulkan**. Error columns give the worst completed observation across policies and, for text, positions.

| Weight | Input | Reference token IDs | Maximum KL by policy | Worst relative L2 | Worst absolute error |
| --- | --- | --- | --- | ---: | ---: |
| 2.0 bpw | Text | `11751, 13, 198` | 1e-08 / 1e-08 / 1e-08 | 7.987e-05 | 0.001157 |
| 2.0 bpw | Image | `5844` | 2.117e-06 / 2.102e-06 / 2.102e-06 | 0.00289543 | 0.035659 |
| 2.5 bpw | Text | `11751, 13, 198` | 7e-09 / 7e-09 / 7e-09 | 7.988e-05 | 0.000883 |
| 2.5 bpw | Image | `6301` | 3.834e-06 / 3.786e-06 / 3.786e-06 | 0.00228845 | 0.026694 |
| 3.0 bpw | Text | `11751, 13, 198` | 1.5e-08 / 1.5e-08 / 1.5e-08 | 9.355e-05 | 0.000862 |
| 3.0 bpw | Image | `3605` | 2.2753e-05 / 2.3186e-05 / 2.3186e-05 | 0.00135995 | 0.024562 |

The text probe recorded 231 device submissions and the image probe 1778 under both Auto and strict Vulkan. CPU recorded zero. With the Vulkan ICD hidden, an actual 2.0 bpw Auto CLI request generated `Hello` and reported `backend=Cpu`; strict Vulkan exited with status 2 and reported the missing driver. Reference text uses the fork's default KV storage and reference vision uses F32 KV. [Wall times on the same host](performance.json)

The reference CPU kernel reuses each decoded tile for several prompt tokens. The current Rust CPU kernel decodes the tile per input row, and permuted output rows can cause several workers to evaluate the same Hadamard block. The Vulkan path runs input transform, coded matmul and output transform per band. The timings include host contention and do not isolate the cost of individual kernels.

## Image alignment repair

With contiguous RoPE sections, the 3.0 bpw image prediction kept the expected token but exceeded the KL gate (0.002448, strict Vulkan). Supplying Rust's projected image embeddings to the reference decoder reduced KL to 0.00002235. Conversely, supplying the reference embeddings to the original Rust decoder gave KL 0.00208956, using both a single 34-row step and the reference's 4 / 9 / 21 chunk boundaries. The image position grid and text IDs agreed. The NariQ reference graph selected IMRoPE (type 40). Enabling its interleaved temporal, row and column frequency layout in both Rust decoder backends brought image KL under the existing gate. Equal text coordinates had hidden the layout error.

## CLI, HTTP, SSE and JEV

The raw chat probes used greedy generation, a two-token limit, `Hi` for text, and a 32×32 red PNG with `Color?` for images. Each SSE check reconstructs content and requires the terminal `[DONE]`. The JEV request supplies context `Red`, question `Color?` and candidates A = Red, B = Blue. The table counts CLI and HTTP separately. Probabilities are for A from the matching image CLI / HTTP probes on the corrected layout.

| Weight | Policy | Text CLI / HTTP / SSE | Image CLI / HTTP / SSE | JEV text CLI / HTTP | JEV image CLI / HTTP | P(A), image CLI / HTTP |
| --- | --- | --- | --- | --- | --- | --- |
| 2.0 bpw | cpu | pending / pass / pending | pending / pending / pending | pending / pending | pending / pending | — / — |
| 2.0 bpw | auto | pass / pass / pass | pass / pass / pass | pass / pass | pass / pass | 0.9999219 / 0.9999219 |
| 2.0 bpw | vulkan | pass / pass / pass | pass / pass / pass | pass / pass | pass / pass | 0.9999219 / 0.9999219 |
| 2.5 bpw | cpu | pending / pending / pending | pending / pending / pending | pending / pending | pending / pending | — / — |
| 2.5 bpw | auto | pass / pass / pass | pass / pass / pass | pass / pass | pass / pass | 0.9999686 / 0.9999686 |
| 2.5 bpw | vulkan | pass / pass / pass | pass / pass / pass | pass / pass | pass / pass | 0.9999686 / 0.9999686 |
| 3.0 bpw | cpu | pass / pass / pass | pass / pass / pending | pending / pass | pending / pending | — / — |
| 3.0 bpw | auto | pass / pass / pass | pass / pass / pass | pass / pass | pass / pass | 0.9999878 / 0.9999878 |
| 3.0 bpw | vulkan | pass / pass / pass | pass / pass / pass | pass / pass | pass / pass | 0.9999878 / 0.9999878 |

Text-only entry captures include runs before the rotary-layout correction. Those text positions use equal temporal and spatial coordinates. Image entries above require a process started with the corrected binary. The two-token chat probes exercise transport and routing. The full-vocabulary image comparison above measures numerical accuracy. The published release is tagged image-text-to-text. Video and long-context benchmark scores were not evaluated.

## Operator and regression checks

- Pinned C++ integer-grid fixture for TRL1–TRL6, permuted dense oracle, frequency-band composition, logical worker partition, and validation of missing scales, duplicate indices and the GGUF version marker. Rust scalar or SIMD floating-point dots use a tolerance.
- Apple M3 Max, Metal via MoltenVK. One device projection used six bands of different widths, nontrivial gathers and 11 token rows across an eight-row scratch buffer. All outputs were finite, maximum absolute error against CPU was 0.00012207031 (limit 0.0005), and a device submission completed.
- The M3 Max exposed an existing device initialization failure. The shared pipeline table attempted to compile PiVo attention, which uses FP64, even for Qwen. On devices lacking FP64, the PiVo constructor now reports the missing capability and the ordinary operator table stays usable. NariQ registers its own shader only when a coded weight is bound.
- Existing Qwen3.5-0.8B BF16, CPU, eight threads, `Hi`, greedy, two output tokens. Both the base and reviewed runtime returned `Hello!`. All 14 full-vocabulary `result_output` captures were byte-identical. With its original BF16 projector, the 32×32 red PNG and `Color?`, all 25 image-prompt logit rows also matched the base byte for byte. The textual architecture, tokenizer, CLI and server use their existing registrations.
- Four existing CPU-specific fixtures became sensitive to a previously initialized Vulkan device once the shared pipeline could run on Apple. The three YuE2 CPU convolution fixtures and the Q8_0 per-worker partition fixture now select the CPU policy explicitly.
- The Ubuntu 24.04 shader check rebuilt and validated the checked-in SPIR-V with the CI toolchain. The host Homebrew glslang also rebuilt NariQ identically, but its full-repository reproducibility check differed on the pre-existing `nemotron_moe_route.spv`.
- Serial full-library tests on this Mac against base `7dfd5f15`. With default features, base was 1691 passed / 24 failed / 119 ignored and reviewed code 1697 / 24 / 119, with the same failures by name. With `vulkan,parity-trace`, base was 1815 / 53 / 264 and reviewed code 1842 / 32 / 265. All 32 failures already occurred on the base. The seven focused loader, kernel and device tests passed. Cargo exposes the same nine binary targets.
