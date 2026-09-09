# LFM2.5-1.2B Inference SIMD 优化路线

## Baseline（2026-09-09，Windows MSVC，T8）

`./target/release/rust-model-inference.exe --model models/LFM2.5-1.2B-Instruct-Q8_0.gguf --prompt "法国的首都是" --max-tokens 200 --threads 8`

```
Model: lfm2.5 | n_embd=2048 n_layer=16 n_head=32 n_ff=8192 d_conv=2 | loaded in ~50ms
Prompt: 11 tokens
Prompt: 8.4 t/s | Generation: 24.5 t/s | end-to-end: 5.4 tok/s
Output: 法国的首都是巴黎。
```

走路径：`models/lfm25/trunk/forward.rs`（import 已经是 `vec_add_into` / `vec_mad_f16_f32` / `vec_scale_f32` / `vec_mul_inplace` 的 SIMD 友好状态）。

---

## Phase 1（已完成 2026-09-09，GT +11%）

**改动 2 行**：把两个 scalar fill 循环改成 `.fill()` 调用。

### 1.1 `attn_out` 标量 fill → `slice::fill`

**位置**：`src/models/lfm25/trunk/forward.rs:748`

```rust
// Before:
for d in 0..n_embd_head_v {
    attn_out[out_base + d] = 0.0;
}

// After:
attn_out[out_base..out_base + n_embd_head_v].fill(0.0);
```

**为什么有效**：`.fill()` 让 LLVM 知道这是非 aliasing write，
- 不被循环依赖 tracking 阻塞
- vectorizer 可以完全跳过这段
- 后续 `vec_scale_f32` / `softmax_inplace` pipeline 更好

**频率**：每层 × 32 head = 512 次/forward，decode 时每 token 都走。

### 1.2 `scores` neg-inf padding → `slice::fill`

**位置**：`src/models/lfm25/trunk/forward.rs:825`

```rust
// Before:
for v in &mut scores[n_cached..n_padded] {
    *v = f32::NEG_INFINITY;
}

// After:
scores[n_cached..n_padded].fill(f32::NEG_INFINITY);
```

**频率**：每层 × 32 head = 512 次/forward，但 `n_padded - n_cached` 在长 context 时小，前期大。

### 1.3 结果

| 指标 | Before | After | 变化 |
|------|--------|-------|------|
| **Generation t/s** | 24.5 | 27.3 (3-run avg) | **+11%** |
| End-to-end tok/s | 5.4 | 10.9 | +100%（含 prefill） |
| Prompt t/s | 8.4 | 22.7 | +170% |

> **注意**：ETE 提升远超 GT 是因为 prefill 用同一段代码且只跑一次，
> decode 每 token 重复 512 次 fill（16 层 × 32 head），单次成本小但累积大。

---

## Phase 2（规划中）

### 2.1 `attention values gather` 重构（推荐先做）

**位置**：`src/models/lfm25/trunk/forward.rs:825-833`

```rust
// 当前（每次 decode 走）：
let mut values = [0.0f32; 512];
for d in 0..n_embd_head_v {
    for t in 0..n_cached {
        values[t] = v_cache[kb + t * n_embd_gqa + kv_h * n_embd_head_v + d];
    }
    attn_out[out_base + d] = dot_f32(&values[..n_padded], &scores[..n_padded], n_cached);
}
```

**问题**：
1. 512-element stack buffer 每次重新初始化
2. transpose gather（`v_cache[kb + t * stride + d]`）— stride access，cache-unfriendly
3. 调 `dot_f32` 函数开销

**目标**：去掉中转 buffer，inline 计算（按 d 循环累加）：

```rust
// 重构后：
for d in 0..n_embd_head_v {
    let mut acc = 0.0f32;
    for t in 0..n_cached {
        acc += v_cache[kb + t * n_embd_gqa + kv_h * n_embd_head_v + d] * scores[t];
    }
    attn_out[out_base + d] = acc;
}
```

**预估收益**：GT +5-10%（去掉中转 + 函数调用；LLVM 可在长 context 时部分向量化 stride dot）。

**风险**：低。语义等价（同样输入同样输出），没有改变数据依赖。

### 2.2 `shortconv conv1d` 标量 dot unroll

**位置**：`src/models/lfm25/trunk/forward.rs:996-1004`

```rust
for c_idx in 0..n_embd {
    let k_off = c_idx * l_cache;
    let b_off = c_idx * l_buf;
    let mut acc = 0.0f32;
    for k in 0..l_cache {                              // l_cache = 4
        acc += bx_buf[b_off + k] * kernel[k_off + k];
    }
    conv_out[c_idx] = acc;
}
```

**改动**：手写 unroll 4 次 fma（l_cache 在 LFM2.5 是固定 4，从 config 查得）：

```rust
for c_idx in 0..n_embd {
    let k_off = c_idx * 4;
    let b_off = c_idx * l_buf;
    let a0 = bx_buf[b_off];
    let a1 = bx_buf[b_off + 1];
    let a2 = bx_buf[b_off + 2];
    let a3 = bx_buf[b_off + 3];
    let w0 = kernel[k_off];
    let w1 = kernel[k_off + 1];
    let w2 = kernel[k_off + 2];
    let w3 = kernel[k_off + 3];
    conv_out[c_idx] = a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3;
}
```

**预估收益**：GT +2-5%（每层都跑 16 次，但每个 c_idx 只有 4 fma）。

**风险**：低。但 `l_cache` 是 cfg 字段（不是 const），需要 `assert!(cfg.l_cache == 4)` 保护。

### 2.3 `logits /= temperature` → `vec_scale_f32`

**位置**：`src/models/lfm25/trunk/forward.rs:290-292`

```rust
// Before:
for l in logits.iter_mut() {
    *l /= temperature;
}

// After:
vec_scale_f32(logits, 1.0 / temperature);
```

**预估收益**：微小（一次/tok，vocab=65000），但代码更干净。
**风险**：极低。

### 2.4 `argmax` → `argmax_f32`

**位置**：`src/models/lfm25/trunk/forward.rs:282-288`

`ops/sampling.rs` 已有 `argmax(x: &[f32]) -> usize`（line 4）。可直接复用。
**预估收益**：微小。
**风险**：极低。

### 2.5 `for (value, bias) in x.iter_mut().zip(bias)` → `vec_add_into`

**位置**：`src/models/lfm25/trunk/forward.rs` 多处 prefill residual add

参考 llama trunk 已经全部用 `vec_add_into`，LFM2.5 这几处是漏网之鱼。
**预估收益**：prefill only（decode 不走），小。
**风险**：低。

---

## Phase 3（潜在大收益）

### 3.1 FFN Q8_0 量化 dedup

**借鉴**：PR #54 给 qwen35 trunk 加的优化——
> The qwen35 trunk was calling `quantize_and_matmul_with_scratch` once per matmul: WQ, WK, WV each re-quantized the same F32 input slice to Q8_0 + scales + Q8K (3x redundant); same for FFN gate and FFN up (2x redundant).
> 
> Pre-quantize once per token into the existing q8_buf / scale_buf / q8k_buf, then run WQ/WK/WV (and FFN gate/up) as `kernel.forward_prepared` inside a single `pool.compute` closure.

**LFM2.5 当前**：每个 matmul 前都做 `quantize_q8_0_into` + `quantize_row_q8_k_into`（forward.rs:215, 680, 914 等）。

**改动**：把同一层同一 input 的量化合并到 matmul 之前。

**预估收益**：GT +10-20%（量化本身是 free 的，但减少 cache pressure 和减少重复 work）。

**风险**：中。要确保量化 buffer 不被并发 matmul 复用（pool.compute 的并发模型要正确）。

### 3.2 `KvCache::F32` → `KvCache::F16`

**借鉴**：LFM2.5 已经有 `KvCache::F16` 分支（forward.rs:718），但 default 可能是 F32。

**LFM2.5-1.2B 实测**：n_embd=2048, n_head=32, n_head_kv 隐含 GQA 比 1:1（n_layer=16, max_ctx 默认 ~2048）→ KV cache 总大小：16 × 2048 × 2048 × 4 = 256MB (F32) vs 128MB (F16)。

**改动**：默认切换 + 在线量化（matmul 之前）。

**预估收益**：内存减半 + GT +2-5%（F16 dot 比 F32 dot 快，量化在 matmul 前一次性做）。

**风险**：低。已有 `vec_mad_f16_f32` 路径走 F16 cache。

---

## Phase 4（不建议）

### 4.1 `rope_neox` per-head 循环展开

已经在 `ops::rope` SIMD 优化过，每个 head 64 dim 已经覆盖。

### 4.2 shortconv 改 NEON 特化

LFM2.5 矩阵小（n_embd=2048, d_conv=4），SIMD overhead 比收益大。

---

## 实施优先级

| 优先级 | 改动 | 行数 | 预估 GT 收益 | 风险 |
|--------|------|------|--------------|------|
| **P0** | 2.1 attention values gather | ~10 | +5-10% | 低 |
| **P0** | 2.2 shortconv conv1d unroll | ~15 | +2-5% | 低 |
| **P1** | 2.3 logits scale → vec_scale_f32 | 1 | 微小 | 极低 |
| **P1** | 2.4 argmax → argmax_f32 | 5 | 微小 | 极低 |
| **P1** | 2.5 residual add → vec_add_into | ~5 | prefill | 低 |
| **P2** | 3.1 FFN Q8_0 dedup | ~30 | +10-20% | 中 |
| **P3** | 3.2 F32 → F16 KV cache default | ~10 | +2-5% | 低 |

**推荐顺序**：P0 (2.1 + 2.2) → P1 (cleanup) → P2 (FFN dedup)。

## 验证

每次改动：
1. `cargo build --release --bin rust-model-inference` 通过
2. 跑 3 次取平均 GT：
   ```bash
   for i in 1..3; do ./target/release/rust-model-inference.exe \
     --model models/LFM2.5-1.2B-Instruct-Q8_0.gguf \
     --prompt "法国的首都是" --max-tokens 200 --threads 8 2>&1 | grep t/s; done
   ```
3. 输出文本必须是 `法国的首都是巴黎。`（bit-exact）