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

## Public entrypoints

Each of the nine GGUFs ran through the CLI, non-streaming HTTP chat and SSE in all three compute policies, with F32 and F16 KV (54 combinations). Every chat returned `OK` for the 44-token prompt, with one completion token. The [per-run receipts and timings](performance.json) include both storage and compute policy. A linear trace entry counts a dispatch group, which may contain multiple projections.

On the Q8_0 strict Vulkan CLI run, 226 linear groups and 56 attention calls ran on the GPU, with 283 device submissions. The matching chat and SSE requests reported 452 linear groups and 112 attention calls on Vulkan in total. On Auto, the CLI ran 114 linear groups and all 56 attention calls on Vulkan and 112 smaller linear groups on CPU. With a missing ICD, Auto completed the same request with CPU attention and projections. Strict Vulkan failed at initialization in both binaries.

F16 KV occupies two bytes per value on the GPU. Each local layer stores at most `min(capacity, window + batch_size - 1)` KV rows. Global layers retain the request history. A separate device test compared causal GQA with a direct host calculation across repeated wraps, mixed batch sizes, single-row decode and replaying an append. The same test exercised full attention and both KV formats.

Q8_0 single and grouped JEV matched between CLI and HTTP on CPU and strict Vulkan. JEV uses F32 KV. A required HTTP weather tool returned `weather({"city":"Hangzhou"})`. Chat also rejects image input and reserved control tokens.

## Full-model numerical alignment

The reference uses Microsoft's PyTorch model, eager F32 attention and the same GGUF weights independently unpacked with gguf-py. The fixed arithmetic, Unicode/whitespace and tool-history prompts matched the official template bytes and all 56, 62 and 193 token IDs. Each case compared every one of the 200029 logits for prefill and three teacher-forced decode steps. The continuation inputs were pinned to the BF16 reference so that different weight formats received the same histories.

F32-KV budgets for floating weights were max abs ≤ 0.05, RMSE ≤ 0.008 and cosine ≥ 0.999999. Packed same-weight budgets were 0.80, 0.12 and 0.999, with a top-1 reference margin allowance of 0.20. The separate floating F16-KV run used 0.10, 0.015 and 0.99999. Every greedy ID matched its same-weight reference on all 27 F32-KV runs (324 of 324 steps). All figures below use the worst step across the three prompts.

| Storage | CPU max abs | Auto max abs | Vulkan max abs | Worst RMSE | Min cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| F32 | 6.96182e-05 | 6.96182e-05 | 6.10352e-05 | 1.44099e-05 | 0.999999999996 |
| BF16 | 5.55515e-05 | 7.48634e-05 | 6.35386e-05 | 1.39895e-05 | 0.999999999996 |
| F16 | 0.0224004 | 0.0113773 | 0.0113773 | 0.00325839 | 0.999999766211 |
| Q8_0 | 5.91278e-05 | 6.00815e-05 | 5.43594e-05 | 1.11951e-05 | 0.999999999997 |
| Q4_0 | 4.55379e-05 | 3.8147e-05 | 3.8147e-05 | 9.45059e-06 | 0.999999999998 |
| Q4_1 | 7.20024e-05 | 7.39098e-05 | 7.39098e-05 | 1.08583e-05 | 0.999999999997 |
| Q4_K | 6.29425e-05 | 6.58035e-05 | 6.58035e-05 | 1.03892e-05 | 0.999999999997 |
| Q5_K | 5.53131e-05 | 5.57899e-05 | 5.24521e-05 | 1.04179e-05 | 0.999999999997 |
| Q6_K | 5.65052e-05 | 6.03199e-05 | 6.03199e-05 | 1.11862e-05 | 0.999999999997 |

The 2094-token BF16 prompt crossed the 2048-token sliding window. Strict Vulkan selected `[5175, 13]`, with max abs 4.57764e-05 and RMSE 7.67678e-06. The recorded CPU run selected the same tokens and had max abs 3.97e-5.
Q4_K with F16 KV on CPU matched all 12 greedy tokens, with max abs 0.00531727 and RMSE 0.000939801.
Q8_0 with F16 KV on strict Vulkan matched all 12 greedy tokens, with max abs 0.0047102 and RMSE 0.000859339.

### Weight quantization and activation precision

The original packed path rounded each projection input to Q8. On the captured Q8_0 prompt, the layer-1 Q input already differed by 0.001428 before activation packing and 0.01207 after packing. Across the decoder, this pushed the full-model logits outside the fixed budgets. Aion now preserves the F32 input activations on both backends. The CPU dequantizes a single row per worker and reuses it across input tokens. The Vulkan shader decodes packed weights while multiplying them. A separate same-input replay checked 394 projections per format/backend, covering every layer and the tied output head on prefill and decode (4,728 total). The largest scaled absolute errors were 8.66e-7 on CPU and 2.68e-7 on Vulkan, against a 2e-4 budget.

The following comparison uses PyTorch for both executions. It measures quality loss from storing the weights at a different precision, holding the input histories and F32 activation math fixed.

| GGUF vs original BF16 | Greedy IDs matching BF16 | Worst max abs | Worst RMSE | Min cosine |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 | 12 / 12 | 0.4336 | 0.07883 | 0.999856 |
| Q4_0 | 10 / 12 | 7.436 | 1.474 | 0.948027 |
| Q4_1 | 10 / 12 | 6.382 | 1.18 | 0.971393 |
| Q4_K | 10 / 12 | 7.26 | 1.242 | 0.970619 |
| Q5_K | 11 / 12 | 3.621 | 0.6245 | 0.991694 |
| Q6_K | 11 / 12 | 3.855 | 0.6434 | 0.991225 |

## Review and regression

- The default library suite had 1,706 passes, 13 failures and 119 ignored tests. The failure names matched the pinned pre-adaptation baseline. The real-checkpoint tests and device-only fixtures above were invoked explicitly.
- Qwen3-0.6B Q4_0 strict Vulkan kept the baseline two-token `OK.` output and 28 submissions. THX-02 Q8_0 strict Vulkan returned exactly the same probability (`0.9728000801073396`) and 409 device submissions before and after the follow-up.
- Review covered F16 cache packing, append retries, ring bounds and shared pipeline indices. Tests cover grouped packed projections and mixed-size GQA batches. Both added shader binaries matched the output of the CI shader toolchain.
- The initial adaptation also corrected the tool-history newline and optional PiVo FP64 shader initialization on MoltenVK. The checkpoint and binary targets are unchanged. Hardware tests used the Apple M3 Max.

[Reproduction commands](README.md) · [Measurements](performance.json)
