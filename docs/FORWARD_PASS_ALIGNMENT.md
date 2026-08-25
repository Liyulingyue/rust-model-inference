# Forward Pass Alignment Check

## Setup
- Prompt: `The capital of France is`
- Chat template applied → 16 prompt tokens
- Same model file (MiniCPM5-1B-Q8_0.gguf)
- llama.cpp: prints top-10 logits from `llama_get_logits(ctx)` after each `llama_decode`
- Our Rust: prints top-10 logits from `scratch.logits` (post-nominal sampling target)
- Note: rust log is gated by `RUST_LLAMA_DEBUG_LOGITS=1`

## Step 15 (first generated token)
```
llama.cpp: 33:10.95 130063:10.58 242:10.36 350:10.15 ...
Rust:      13656:17.07 608:16.68 59:16.23 9:16.18 ...
```
- All 10 tokens differ
- Values differ by ~+6 (our values higher)
- BUT ordering completely different - not a simple offset

## Step 16
```
llama.cpp: 5:13.36 24:13.15 49:12.78 11127:12.26 ...
Rust:      21128:20.81 14100:20.28 30731:19.90 ...
```
- Values differ by ~+7

## Step 17
```
llama.cpp: 80964:22.80 122895:22.76 608:22.37 18812:21.49 ...
Rust:      367:17.57 26888:17.34 788:17.29 545:17.22 ...
```
- Values differ by ~-5 (reversed sign!)

## Diagnosis
The forward pass is clearly broken. The values are not even
monotonically off. This suggests:
1. Different tensor operations executed
2. OR different tensor shapes (transposed)
3. OR different RMSNorm epsilon / scale
4. OR RoPE applied differently

## Next steps
1. Add prints **at each layer** in Rust to localize the divergence
2. Compare intermediate hidden states (e.g. after layer 0, layer 1, etc.)
3. The first divergence will tell us where it breaks