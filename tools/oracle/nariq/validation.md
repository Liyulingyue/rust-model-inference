# NariQ v1 validation

2026-11-26. The EPYC 9334 / RADV NAVI31 host downloaded the three GGUFs and BF16 projector directly from the pinned ModelScope release. All four SHA-256 values matched. The unmodified base loader stopped at the fork-specific GGML type. The reference fork generated separate logit captures for all three files on that host. [File identities, commands and reference pin](README.md)

## Same-file text parity

Input `The capital of France is` encoded to `[760, 6511, 314, 9338, 369]`. All three reference files produced `[11751, 13, 198]`. One prefill and two subsequent decode predictions span 248320 logits each. The gates are exact token IDs and maximum KL ≤ 0.001. The table reports the largest error across the three positions. Reference CPU uses its default KV storage, Rust uses the Qwen session's F32 KV.

| Release | Rust policy | Greedy agreement | Max KL | Max relative L2 | Max absolute | Device evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2.0 bpw | CPU | 3/3 | 1.0e-8 | 7.987e-5 | 0.001157 | CPU |
| 2.0 bpw | Auto | 3/3 | 1.0e-8 | 7.970e-5 | 0.001141 | Numerical comparison passed. The first capture used a legacy counter before enabling it, so it cannot establish Auto placement. |
| 2.0 bpw | Vulkan | 3/3 | 1.0e-8 | 7.970e-5 | 0.001141 | Repeated with the CLI compute scope active. The strict session submitted 231 times. |
| 2.5 bpw | CPU | 3/3 | 7.0e-9 | 7.988e-5 | 0.000883 | CPU |
| 2.5 bpw | Auto / Vulkan | pending | — | — | — | No complete comparison retrieved |
| 3.0 bpw | CPU / Auto / Vulkan | pending | — | — | — | Reference complete, Rust comparisons outstanding |

These remote measurements precede the final changes to CPU worker ownership and on-demand Vulkan pipeline creation. The SSH instance stopped accepting connections during the remaining jobs. The 27B end-to-end recheck on the reviewed code and the full image-logit comparisons remain outstanding. [Measured times](performance.json)

## Public entry points

The implementation loads all three released files through the existing text CLI and server. With the released BF16 projector, the image encoder is the existing Qwen vision path and CPU image embedding feeds the decoder. Below are the responses retrieved from the remote runs. An empty cell is an outstanding real-file check.

| Release | Policy | Text CLI | Chat HTTP | SSE | Image chat CLI / HTTP | JEV CLI / HTTP |
| --- | --- | --- | --- | --- | --- | --- |
| 2.0 bpw | CPU / Auto / Vulkan | Startup recorded | — | — | — | — |
| 2.5 bpw | CPU / Auto / Vulkan | Startup recorded | — | — | — | — |
| 3.0 bpw | CPU | Startup recorded | `Hi` → `Hello!`, 13 input / 2 output tokens | — | — | — |
| 3.0 bpw | Auto | — | — | — | — | — |
| 3.0 bpw | Vulkan | Startup recorded | — | — | — | HTTP text and image chose A. P(A) 0.9995383 and 0.9999881. CLI comparison outstanding. |

The early GPU HTTP chat probe sent the model ID `local` and returned 404. The replay uses the ID from `/v1/models`. A 404 supplies no chat or SSE inference evidence. Video inputs share the base Qwen visual path and have not been evaluated on these releases. The published files omit an MTP head.

## Operator and regression checks

- Pinned C++ integer-grid fixture for TRL1–TRL6, permuted dense oracle, frequency-band composition, logical worker partition, and validation of missing scales, duplicate indices and the GGUF version marker. Rust scalar or SIMD floating-point dots use a tolerance.
- Apple M3 Max, Metal via MoltenVK. One device projection used six bands of different widths, nontrivial gathers and 11 token rows across an eight-row scratch buffer. All outputs were finite, maximum absolute error against CPU was 0.00012207031 (limit 0.0005), and a device submission completed.
- The M3 Max exposed an existing device initialization failure. The shared pipeline table attempted to compile PiVo attention, which uses FP64, even for Qwen. On devices lacking FP64, the PiVo constructor now reports the missing capability and the ordinary operator table stays usable. NariQ registers its own shader only when a coded weight is bound.
- Existing Qwen3.5-0.8B BF16, CPU, eight threads, `Hi`, greedy, two output tokens. Both the base and reviewed runtime returned `Hello!`. All 14 full-vocabulary `result_output` captures were byte-identical. The textual architecture, tokenizer, CLI and server use their existing registrations.
- Four existing CPU-specific fixtures became sensitive to a previously initialized Vulkan device once the shared pipeline could run on Apple. The three YuE2 CPU convolution fixtures and the Q8_0 per-worker partition fixture now select the CPU policy explicitly.
- The Ubuntu 24.04 shader check rebuilt and validated the checked-in SPIR-V with the CI toolchain. The host Homebrew glslang also rebuilt NariQ identically, but its full-repository reproducibility check differed on the pre-existing `nemotron_moe_route.spv`.
- Serial full-library tests on this Mac against base `7dfd5f15`. With default features, base was 1691 passed / 24 failed / 119 ignored and reviewed code 1697 / 24 / 119, with the same failures by name. With `vulkan,parity-trace`, base was 1815 / 53 / 264 and reviewed code 1842 / 32 / 265. All 32 failures already occurred on the base. The seven focused loader, kernel and device tests passed. Cargo exposes the same nine binary targets.
