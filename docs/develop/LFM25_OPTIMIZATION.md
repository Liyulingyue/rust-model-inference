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

| 指标 | Before | After (Phase 1) | After (Phase 2) | 总变化 |
|------|--------|-----------------|-----------------|--------|
| **Generation t/s** | 24.5 | 27.3 | **28.6** | **+17%** |
| End-to-end tok/s | 5.4 | 10.9 | 11.6 | +115%（含 prefill） |
| Prompt t/s | 8.4 | 22.7 | 23.9 | +184% |

> **注意**：ETE 提升远超 GT 是因为 prefill 用同一段代码且只跑一次，
> decode 每 token 重复 512 次 fill（16 层 × 32 head），单次成本小但累积大。

---

## Phase 2（已完成 2026-09-09）

### 2.1 `attention values gather` 重构 ❌ 失败（已 revert）

**位置**：`src/models/lfm25/trunk/forward.rs:825-833`

尝试把 `dot_f32` 调用替换成 inline strided dot：

```rust
// 试过的方案（已 revert）：
for d in 0..n_embd_head_v {
    let mut acc = 0.0f32;
    let base_d = kb_local + kv_h * n_embd_head_v + d;
    for t in 0..n_cached {
        acc += v_cache[base_d + t * n_embd_gqa] * scores[t];
    }
    attn_out[out_base + d] = acc;
}
```

**实测**：GT 从 27.3 跌到 21.3（-22%）。

**原因**：`dot_f32`（`ops/dot.rs`）是 AVX2 + FMA 向量化的（每次 8 floats FMA）。
inline strided dot 里 `v_cache[base_d + t * n_embd_gqa]` 是 stride access（stride=2048），
LLVM 不能向量化 → 比 AVX2 向量化的 `dot_f32` 慢 4×。

**Lesson**：当 `dot_f32` 已经 SIMD 时不要"简化"调用。这条路径在
F32 KV cache 模式下才走（`KvFormat::F32`），而 LFM2.5 默认 `KvFormat::F16`，
所以实际很少跑。**保留原 `dot_f32` 调用**。

### 2.2 `shortconv conv1d` 标量 dot unroll ✅ 成功（GT +7.7%）

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

## Phase 3（重新评估）

### 3.1 ~~FFN Q8_0 量化 dedup~~ ❌ 已经实现，无需做

**重要发现**：审计时误以为 LFM2.5 每个 matmul 前都做 quantize，实际**已经是 dedup 模式**——

参考 `src/models/lfm25/trunk/forward.rs`:
- **Attention** (line 600-650): 单次 `quantize_q8_0_into` + 单次 `pool.compute` 内跑 `wq` + `wk` + `wv` 三个 matmul，共享同一个 `input`/`q8`/`sc`/`q8k`
- **FFN** (line 461-520): 单次 quantize + 单次 pool.compute 内跑 `w_gate` + `w_up` + silu_mul

这正是 PR #54 给 qwen35 加的模式，**LFM2.5 已经具备**。

### 3.2 F32 KV cache path — 几乎不走

LFM2.5 默认 `KvFormat::F16`（`app/text.rs:176`）。F32 path 仅在用户传 `--kv-cache f32` 时走。
**不是 GT 优化目标**。

### 3.3 dead `.to_vec()` allocation — prefill only

`forward_layer` line 433/448 的 `let _cur_after_block = ... .to_vec()` 是 dead allocation（变量立即丢弃）。
每层 × 2 路径 × prefill tokens 次 = prefill 时 ~32 次冗余分配 + 拷贝。

**改动**：去掉 `.to_vec()`，直接 discard。约 5 行代码改动。
**预估收益**：prefill 提升（prefill only），GT 几乎不影响。

**实测**：已完成，GT 没变化（27.6-29.4 t/s vs Phase 2 的 28.2-30.8 t/s，噪声范围内），符合 prefill-only 的预期。

---

## Phase 4（不建议）

### 4.1 `rope_neox` per-head 循环展开

已经在 `ops::rope` SIMD 优化过，每个 head 64 dim 已经覆盖。

### 4.2 shortconv 改 NEON 特化

LFM2.5 矩阵小（n_embd=2048, d_conv=4），SIMD overhead 比收益大。

---

## 实施优先级（修订）

| 优先级 | 改动 | 行数 | 预估 GT 收益 | 状态 |
|--------|------|------|--------------|------|
| **P0** | Phase 1: fill() x 2 | 2 | +11% | ✅ 完成 |
| **P0** | Phase 2.2: shortconv conv1d unroll | ~15 | +7.7% | ✅ 完成 |
| **P1** | 2.1 attention values gather | ~10 | -22%（**失败**） | ❌ 已 revert |
| **P1** | 3.3 dead `.to_vec()` | ~5 | prefill only | ✅ 完成 |
| ~~P2~~ | ~~3.1 FFN dedup~~ | — | — | 已实现，无需做 |
| ~~P3~~ | ~~3.2 F32 KV path~~ | — | — | 几乎不走 |

**当前状态**：LFM2.5 GT 从 baseline 24.5 → 28.6 t/s (+17%)。剩余 GT 优化空间不大，
**真正大头是 matmul（已 SIMD）+ attention softmax（已 SIMD）+ shortconv（已 unroll）**。

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