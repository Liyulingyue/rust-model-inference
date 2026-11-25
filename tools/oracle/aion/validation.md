# Aion Instruct validation

Measured on 2026-11-26 on Apple M3 Max, macOS 27.2, with eight Rust compute threads and MoltenVK 1.4.2. The Rust build used `release-fast`. The reference was Transformers 5.19.0 and PyTorch 2.9.1 with eager F32 attention on the same host.

## Checkpoint contract

ModelScope file revision `2e7155f6388a09858c6f6d99b9e100979d4ca1fc`. The BF16 source `model.safetensors` is 4,577,698,688 bytes with SHA-256 `6c779881b539d9074f356964ad3089459e9e837da7649d6f271859135256984e`. The model and config Python files have SHA-256 `b48ac50a21700e8dcbbe6643e8de22f4894a59a2a4a2969913f3eacba7d51ed4` and `6f0cb6d9a8657d4ae2cb23f7f10038a88a0c15548081b4ee87cae49365e66736`. The tokenizer and template SHA-256 are `7ea8bdf68c3e7549a3fb4342523288ce628f6ab56a618f9a4dfb234a0b4d46a8` and `36f20c222f9f7cc374987623008542e7c4d63d3b9c35fd6b05518a04af35df03`. Quantized weights were independently decoded with gguf-py from llama.cpp `7d7f0287530bc8981690f361d8c1627dbae1c99a`.

The seven full attention layers are 2, 3, 4, 9, 13, 15 and 16 (zero-based). The other 21 use a 2048-token causal window. The decoder uses 24 query heads, eight KV heads, head width 128 and per-head Q/K RMSNorm. The embedding and output share a 200029 × 2048 weight. Q is 3072 × 2048, K/V are each 1024 × 2048, O is 2048 × 3072 and SwiGLU uses width 8192. The generation stop IDs are 200020 and 199999. The independent tensor inventory checked all 310 tensors and 2,288,831,488 values. The 197 BF16 matrices preserved their source bytes and the F32 export widened every source value exactly.

| GGUF | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| F32 | 9,170,875,136 | `3f019d37ad5671541b1c048935e9d83d07e8f04f31a21334357097781267fe69` |
| F16 | 4,593,459,968 | `6dbb066a0eee0f2d66a3c2b7aa7dd7e9ce46dd34640598b197d675f35c76e71b` |
| BF16 | 4,593,459,968 | `f0dfbcd343103e71aa899069bf8e49d7864e15ae2efbee8ccdbab90df01fbd8d` |
| Q8_0 | 2,447,796,608 | `ab847849b5951c48bf21765e7271dba8ca12b4a65802fb062cedaec86bbc9655` |
| Q4_0 | 1,303,442,816 | `6048282f3f3fb1352da356914595ef399247c7e74746a48c8c51e2d1a88d3566` |
| Q4_1 | 1,446,487,040 | `f9e7c714e61f2e3be29bf59564c1f0025af7ed6989d96de3fd561c3d5551b9e7` |
| Q4_K | 1,303,442,816 | `e56a2b851431d818e6bb754fb561e72cd2198b0e3fc40c259116988d5c687ab9` |
| Q5_K | 1,589,531,264 | `b2c8b5f8011e486b2cdda644d207fac708d4d6cb7288ec331cea3ffc6df7aa49` |
| Q6_K | 1,893,500,256 | `a41c6e8ebf0682d6dae1ef4fd2465ac7aaf4de2bb543068522034d360a5a8c22` |

## Entrypoint coverage

All 27 storage format × policy combinations ran on the real model with F32 KV. Each cell reports CLI, non-streaming chat, and SSE in that order. All produced `OK` and HTTP recorded 44 input tokens and one completion token. Vulkan submitted 227 device operations per CLI invocation and 453 across the two HTTP requests. Auto submitted 115 and 229 respectively and ran small decode projections on the CPU. Strict Vulkan ran all linear projections on the device. The host retains normalization, rotation, attention, KV, and sampling.

| Storage | CPU | Auto | Vulkan |
| --- | --- | --- | --- |
| F32 | Pass / pass / pass | Pass / pass / pass | Pass / pass / pass |
| F16 | Pass / pass / pass | Pass / pass / pass | Pass / pass / pass |
| BF16 | Pass / pass / pass | Pass / pass / pass | Pass / pass / pass |
| Q8_0 | Pass / pass / pass | Pass / pass / pass | Pass / pass / pass |
| Q4_0 | Pass / pass / pass | Pass / pass / pass | Pass / pass / pass |
| Q4_1 | Pass / pass / pass | Pass / pass / pass | Pass / pass / pass |
| Q4_K | Pass / pass / pass | Pass / pass / pass | Pass / pass / pass |
| Q5_K | Pass / pass / pass | Pass / pass / pass | Pass / pass / pass |
| Q6_K | Pass / pass / pass | Pass / pass / pass | Pass / pass / pass |

Q8_0 also passed CLI, HTTP and SSE with F16 KV on strict Vulkan. The same Q8_0 request under Auto with a missing ICD used CPU for all 226 CLI and 452 HTTP projections. Strict Vulkan with that ICD returned an explicit device initialization failure. CLI rejected an image request and a user turn containing a reserved control token.

For JEV, BF16 and Q8_0 CPU matched CLI and HTTP in single and grouped modes. Q8_0 also matched on strict Vulkan. For the Q8_0 CPU single case, both returned A=0.9989032745361328 and B=0.00109669822268188. The grouped case returned A=0.9988780617713928 and B=0.001121907727792859. A real HTTP request with a required weather tool returned `weather({"city":"Hangzhou"})`.

## Numerical alignment

The pinned arithmetic, Unicode/whitespace and tool-history prompts matched the official template bytes and all 56, 62 and 193 input IDs respectively. Each case recorded prefill plus three teacher-forced decode steps, including EOS as a forced continuation where applicable. Each step compared all 200029 logits. Acceptance budgets for F32 KV were max abs ≤ 0.05, RMSE ≤ 0.008 and cosine ≥ 0.999999 for floating weights. The separate BF16/F16 KV run used 0.10, 0.015 and 0.99999. Packed tests used 0.8, 0.12 and 0.999 against independently dequantized weights and 0.20, 0.025 and 0.9999 against independently rounded activations. The rounded pass also checked the greedy token’s oracle margin ≤ 0.20.

| Floating storage | CPU max abs / RMSE | Auto max abs / RMSE | Vulkan max abs / RMSE |
| --- | ---: | ---: | ---: |
| F32 | 6.96182e-05 / 1.34951e-05 | 8.36253e-05 / 1.33824e-05 | 6.86646e-05 / 1.28686e-05 |
| F16 | 0.0224004 / 0.00325839 | 0.0224004 / 0.00325839 | 0.0224004 / 0.00325839 |
| BF16 | 5.55515e-05 / 1.00633e-05 | 6.86646e-05 / 1.12178e-05 | 7.05719e-05 / 1.41747e-05 |

All floating-weight runs had identical greedy IDs and passed their budget. BF16 with F16 KV on CPU had max abs 0.00696683 and RMSE 0.00138415. The separate 2094-token BF16 CPU probe crossed the sliding window edge, matched greedy IDs `[5175, 13]`, and had max abs 3.96967e-05 and RMSE 7.42449e-06. A unit test exercised ring wrap for batched appends.

The six packed formats passed same-input projection replay. Each backend checked 394 projections across prefill and decode, 4728 in total. The assertion normalized absolute error by the largest magnitude in the reference row (floor 1) with a budget of 2e-4, and used 5e-5 for scaled RMSE. The comparison used an independent GGUF decoder and PyTorch matmuls.

| Packed storage | Replay scaled max abs CPU / Vulkan | Whole-model max abs over CPU / Auto / Vulkan | Whole-model worst RMSE | Min cosine | Largest oracle top-1 margin lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | 1.12e-06 / 1.12e-06 | 0.8065 / 0.8065 / 0.8065 | 0.1369 | 0.999591 | 0.0000 |
| Q4_0 | 7.7e-07 / 7.7e-07 | 0.5398 / 0.5398 / 0.5398 | 0.1055 | 0.999703 | 0.0000 |
| Q4_1 | 9.41e-07 / 9.41e-07 | 0.7500 / 0.7500 / 0.7500 | 0.1239 | 0.999685 | 0.0000 |
| Q4_K | 5.28e-07 / 4.66e-07 | 2.1716 / 1.8295 / 2.0818 | 0.4387 | 0.996609 | 0.1217 |
| Q5_K | 6.43e-07 / 6.56e-07 | 2.7288 / 2.7066 / 3.2737 | 0.5535 | 0.993412 | 0.6454 |
| Q6_K | 3.59e-07 / 3.66e-07 | 2.3678 / 2.7015 / 2.9673 | 0.5184 | 0.994455 | 0.2054 |

The packed whole-model results exceeded the fixed budgets, including the earlier unrounded comparison. The Q8_0 layer-0 Q projection input matched the official computation on the 44-token prompt. At layer 1 the floating input max error was 0.001428 and the next Q8 round-trip error was 0.01207. At layer 21 the floating input max error was 0.1264. The matched-input replay verifies the projections with the input held fixed across the two implementations. It complements the whole-model measurements, which include downstream accumulation and rounding.

The two trace steps for the 44-token prompt selected `[5175, 200020]` on both CPU and Vulkan for all six formats. Native CPU↔Vulkan logits for that prompt matched for Q8_0, Q4_0 and Q4_1. The K-quants showed small reduction differences at individual projections and larger accumulated logits differences. The worst of these two-step comparisons was Q5_K with max abs 1.2251 and RMSE 0.2752. All six packed formats remain experimental for full-graph numerical alignment.

## Review and regression

- Fixed the extra newline after a tool result by selecting Transformers-compatible Jinja whitespace rules for Aion. The original renderer retains its existing behavior. Regression coverage checks whitespace, reserved-token errors and streaming tool delimiters inside quoted JSON.
- Fixed optional PiVo attention shader selection on devices without shaderFloat64. Qwen3-0.6B Q4_0 strict Vulkan reproduced the initialization failure before the fix. Afterward it submitted 29 operations and matched the CPU output for the same four greedy tokens.
- The default library suite changed from 1702 passed / 13 failed / 119 ignored to 1705 passed / 13 failed / 119 ignored. The 13 failure names matched the baseline exactly. Both application binaries built with Vulkan, and Cargo reported the same binary targets before and after.
- Both KV formats use the same model path. The full 27-cell CLI/HTTP matrix used F32 KV, with the F16 KV checks listed above. Only MoltenVK on Apple M3 Max was exercised. A fully resident GPU decoder would need per-layer YaRN/sliding-window KV dispatch in the shared device layer.

[Commands and oracle inputs](README.md) · [Same-host timings](performance.json)
