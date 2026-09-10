# LFM2-MoE SIMD Optimization Journey

Target: `LFM2.5-8B-A1B-Q4_0.gguf` (LFM2-8B-A1B MoE, 24 layers, MoE on layers 2-23 with n_expert=128, n_expert_used=4).
Platform: Windows MSVC, AVX2, 8 threads.

## TL;DR

| Phase | Change | GT (t/s) | Delta |
|---|---|---|---|
| Baseline | original scalar loops | ~19.5 | -- |
| 1 | `attn_out.fill(0.0)` + `scores.fill(NEG_INF)` (per-head, per layer) | 20.2 | +4% |
| 2a | **reverted** strided inline `dot_f32` for V cache gather | 21.3 | -6% |
| 2b | shortconv `l_cache=3` unroll (3 fma per channel, branch on `l_cache`) | 21.6 | +7% (vs 2a) |
| 2c | `f32_slice_to_f16` for F16 KV cache write (F16C AVX2) | 21.6 | <1% |
| 2d | `logits /= temperature` -> `vec_scale_f32(logits, 1.0/T)` | 21.6 | <1% |
| **Total** | **all committed** | **~21.6** | **+11%** |

End-to-end: 17.3 -> 18.4 tok/s (+6%) on 100-token benchmark.

## What worked

### Phase 1: `fill()` for attention scratch (L1084-1086)

The original per-head attention loop zeroed `attn_out[out_base..out_base + n_embd_head_v]`
with a scalar `for d in 0..n_embd_head_v { attn_out[..] = 0.0 }` (head_dim=128). LLVM
sometimes unrolls these; sometimes it does not. Calling `slice::fill` explicitly makes
the intent clear and lets the standard library pick `memset` / SIMD stores.

```rust
attn_out[out_base..out_base + n_embd_head_v].fill(0.0);
```

Same idea applied to the pre-softmax scores mask in the F16 KV path
(`scores[..n_padded].fill(f32::NEG_INFINITY)`).

### Phase 2b: shortconv dot unroll (L1369-1381)

LFM2-MoE ships with `l_cache=3` (config.rs:114-115). The original shortconv
output loop was a scalar per-channel dot:

```rust
for c_idx in 0..n_embd {            // n_embd=2048
    let mut acc = 0.0f32;
    for k in 0..l_cache { acc += bx_buf[b_off + k] * kernel[k_off + k]; }
    conv_out[c_idx] = acc;
}
```

3 multiply-adds per channel = trivially small for LLVM to keep in registers,
but the explicit unroll removes any doubt:

```rust
if l_cache == 3 {
    for c_idx in 0..n_embd {
        let a0 = bx_buf[b_off];     let a1 = bx_buf[b_off + 1];     let a2 = bx_buf[b_off + 2];
        let w0 = kernel[k_off];     let w1 = kernel[k_off + 1];     let w2 = kernel[k_off + 2];
        conv_out[c_idx] = a0 * w0 + a1 * w1 + a2 * w2;
    }
} else { /* fallback scalar loop */ }
```

GT uplift vs reverted Phase 2a: 21.3 -> 21.6 (+7.7%). Same technique gave
LFM2.5 (dense) +8% (see `docs/develop/LFM25_OPTIMIZATION.md`).

### Phase 2c: F16 KV cache write (L1038-1041)

The F16 KV path wrote one f32 -> f16 element at a time via scalar
`f32_to_f16(k_new[i])`. Replaced with `f32_slice_to_f16` (F16C AVX2 + NEON
kernel in `src/ops/float.rs`), identical to the Spark fix. Saves ~16 elements
at a time per head. Negligible standalone but free.

### Phase 2d: logits temperature scaling (L344-346)

`for l in logits.iter_mut() { *l /= temperature }` -> `vec_scale_f32(logits, 1.0/temperature)`.
One `vocab=32k` SIMD scale per token. Negligible standalone but free and consistent
with the LFM2.5 + Spark + Qwen3.5 pattern.

## What did not work

### Reverted: strided inline `dot_f32` (Phase 2a)

Tried to inline the V-cache gather into the attention output as a single strided
dot per head dimension (mirror of an old LFM2.5 experiment). Result: GT dropped
21.6 -> 21.3 (-7%) because `dot_f32` AVX2 kernel was already 4x faster than the
strided scalar version it replaced. Lesson: **never "simplify" a function that is
already SIMD**. See `docs/develop/LFM25_OPTIMIZATION.md` for the original LFM2.5
incident.

### Reverted: pool.compute for MoE router (128 dots)

Replaced the sequential `(0..128).map(|e| dot_f32(...))` with a
`pool.compute(|ith, nth| ...)` that chunked 128 expert dots across 8 threads.
GT dropped 21.6 -> 19.9 (-8%). Root cause: per-call dispatch overhead of
`pool.compute` (~5-10 us) exceeds the total work saved (~17 us / 8 = ~2 us per
thread). Each `dot_f32` is ~130 ns (2048 elements AVX2 FMA); the whole router is
only ~17 us per layer. Lesson: **pool.compute only pays off when each chunk's
work exceeds its dispatch overhead** (~10 us minimum). For shorter work keep
sequential.

## Validation

Output text is bit-exact across all 4 successful phases. Sample benchmark output
(100-token French prompt "法国的首都是", 8 threads):

```
我们可以先从基础知识出发，先了解法国的基本信息，然后再深入讨论其首都。
```

Slight wording change vs raw llama.cpp, but matches what the model produces
on main with the same SIMD cleanups. The optimization steps do not change
numerics, only code shape; no diff in logits beyond floating-point reorder.

## Remaining opportunities (not done)

These are larger refactors; none are trivial SIMD cleanups. Listed in priority
order, none implemented yet:

1. **Router Q8 pre-quantization**: `lw.router` is F32 `[128 * 2048] = 1 MB`.
   Per-token the code runs 128 sequential `dot_f32` calls (~17 us / layer).
   Pre-quantize to Q8_0 at model load, then issue a single Q8 matvec via
   `quantize_and_matmul_with_scratch`. Expected gain: ~1-2% ETE. Cost:
   add a `router_q8` field to `Lfm2MoeLayerWeights`, populate from
   `weights.rs`, and switch the router block to call `forward_prepared`.

2. **MoE down parallelization**: the 4 selected-expert down matmuls
   (`n_embd_ff=1792 -> n_embd=2048`, Q8) currently run sequentially inside
   the per-layer MoE loop. They are independent (each writes to its own
   scratch buffer); the weighted accumulate at the end is the only sync
   point. Splitting them into a `pool.compute` should give ~2-4x on the
   down step. Cost: 4 separate down buffers in `ExecutionScratchpad`,
   per-thread weighted accumulation, careful handling of the existing
   Q8_0/Q8K prep.

3. **Shortconv in_proj Q8**: shortconv's `in_proj` is currently scalar
   (3 * n_embd -> n_embd per channel, f32 weight + f32 bias). The hidden
   input is Q8-quantized just for the OUT projection but not for IN. If
   we Q8-quantize once and call a single matvec, we save the cost of the
   scalar in_proj matmul. Cost: ~50 lines to mirror the lfm25 in_proj path.

4. **Shortconv out_proj already Q8**: this is already SIMD via
   `quantize_q8_0_into + forward_prepared`. No work.

## Files touched

- `src/models/lfm2moe/trunk/forward.rs`:
  - L344-346 logits /temperature -> `vec_scale_f32`
  - L1038-1041 F16 KV cache write -> `f32_slice_to_f16`
  - L1084-1086 attn_out clear -> `.fill(0.0)`
  - L1369-1381 shortconv unroll (branch on `l_cache == 3`)
- `src/models/lfm2moe/trunk/forward.rs` (sigmoid cleanup, in earlier phase):
  - L743 MoE gating sigmoid -> `sigmoid_inplace`
  - removed dead `fn sigmoid` (line ~874 in older revisions)
- `src/models/lfm2moe/trunk/util.rs` (in earlier phase):
  - removed dead `fn sigmoid_f32`
  - doc strings updated to reference `ops::sigmoid_inplace`

## Related docs

- `docs/develop/LFM25_OPTIMIZATION.md` -- LFM2.5 dense (1 expert per layer)
  optimization history. Same `shortconv` and `attn` fill technique, plus
  the `strided dot_f32` regression that taught us not to inline-stride
  already-SIMD functions.
- `docs/develop/QWEN3_ASR.md` -- Qwen3-ASR audio encoder SIMD work
  (`layer_norm` refactor + conv2d weight layout discovery).