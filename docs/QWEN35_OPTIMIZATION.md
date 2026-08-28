# Qwen3.5 推理路径优化记录（模块化后校准）

> 原版以 commit `6396b36` 之后的非模块化全内联 `qwen35.rs` 为基准。后来 forward/loader/scratchpad/session/vision 拆分为 `src/models/qwen35/` 目录，文档未同步。本次重写以 `src/` 当前实现为准，砍掉过时/失真条目，保留仍可执行的优化方向。

## 架构变更摘要（vs 原版）

| 原版引用 | 现状 | 备注 |
|---|---|---|
| `src/models/qwen35.rs`（单文件） | `src/models/qwen35/{mod,forward,loader,scratchpad,session,vision,util,positions,clip_config,tests}.rs` | 行号全部失效，下表已重定位 |
| `fuse_vstack` 仅 Q4K/Q5K/Q6K | `src/ops/quant/fuse.rs` 提供 `fuse_vstack_q4_k/q5_k/q6_k/q8_0` 四档 | 仍有零 caller |
| matmul 调用走 `quantize_and_matmul_with_scratch` per token | 同一入口（`src/ops/kernel/mod.rs:66`），调用者更广（Qwen3 llama 也走它） | batch-token 优化将惠及所有调用方 |
| `KvCache` 单结构体 | `core/scratchpad.rs` 中 `KvCache::F16/F32` + `KvState` 抽象 | `positions` 字段仍未加 |
| `Qwen35Scratchpad` 散在各 helper | `src/models/qwen35/scratchpad.rs` 集中 | `kv_cache_pos` / `kv_cache_store` 仍在此文件 |
| `forward_ffn_parallel` 走两次 matmul | 仍两次 matmul（forward.rs:573-580） | fused 权重未生成 |

## 实施状态（11 条 P0/P1/P2 对照）

| # | 原条目 | 文档结论 | 当前位置 | 状态 |
|---|---|---|---|---|
| 1 | P0：FFN gate+up Q8_0 fuse | fuse_vstack 加 Q8_0 分支 | `ops/quant/fuse.rs:25` 实现+单测 | ✅ 基础设施完成，❌ forward 未接通 |
| 2 | P0：batch-token matmul | 新增 `matmul_q8_0_quantized_batch_tokens` | 不存在 | ❌ 未落地 |
| 3 | P1：dense val_gather_dot 重构 | (s,d) 重排 + `vec_dot_f32` | `forward.rs:307-334` 仍逐 head × dim | ❌ 未落地 |
| 4 | P1：recr conv1d AVX2 + 并行 | 内层 AVX2 fma + `pool.compute` | `forward.rs:424-440` 仍纯标量 | ❌ 未落地 |
| 5 | P1：recr SSM 外层并行 + `ssm_outer_product_update` 向量化 | `for v_h` 改 `pool.compute`；outer 内层 AVX2 | `forward.rs:501-523` 串行；`ssm.rs:89-97` 仍逐元素 `vec_mad_f32`（无 AVX2 分支） | ❌ 未落地 |
| 6 | P1：dense score_dot GEMV | 预 pack K，按 head GEMV | `forward.rs:314-318` 仍逐 s `dot_f32` | ❌ 未落地 |
| 7 | P2：kv_cache_pos O(1) 计数器 | `KvCache` 加 `positions: Vec<usize>` | `core/scratchpad.rs:34-47` 无 positions 字段；`models/qwen35/scratchpad.rs:96-106` 仍线性扫零前缀 | ❌ 未落地 |
| 8 | P2：l2_norm/sigmoid/softplus 向量化 | AVX2/NEON | `models/qwen35/util.rs:29-40` 纯标量，无 SIMD 分支 | ❌ 未落地 |
| 9 | P2：scratchpad 零分配 | 复用 scratchpad 替换 `vec![0.0;…]` | `forward.rs:147, 347, 555` 三处仍 `vec![0.0f32; n_tokens * n_embd]`；两个 layer forward 都返回新 `Vec` | ❌ 未落地 |
| 10 | P2：kv_cache_store 单次大 copy | 一次 `copy_to_within` | `models/qwen35/scratchpad.rs:117-122` 仍逐 token `copy_from_slice(k_dim)` | ❌ 未落地 |
| 11 | P0：解码期零分配（隐含） | — | `forward.rs:155-162` 输出 logits 仍走 `quantize_and_matmul_with_scratch` + `vec![0.0; vocab_size]`，每 token 一次 alloc | ❌ 未落地 |

**结论**：11 条里只有 #1 的底层函数到位（且未接通），其余全部未实现。

## 仍可执行的优化方向（按当前代码重新排序）

### 🔴 P0-A：batch-token matmul（最高杠杆，覆盖所有调用方）

**核心痛点**：每层 forward 都是 `for t in 0..n_tokens { w.quantize_and_matmul_with_scratch(...) }`。`kernel/mod.rs:66` 内部对每个 token 跑一次 `pool.compute` + 完整重量化（`quantize_q8_0_into` + `quantize_row_q8_k_into`）。45-token prefill 估算：

| 路径 | 每层 × 每 token matmul 数 | 45-token 总 enqueue |
|---|---|---|
| dense attn ×12 | 4（wq/wk/wv/wo） | 2160 |
| recr attn ×12 | 5（wqkv/wqkv_gate/ssm_beta/ssm_alpha/ssm_out） | 2700 |
| FFN ×24 | 3（gate/up/down） | 3240 |
| **合计** | — | **≈ 8100 次 `pool.compute` 调度** |

**改动**：在 `src/ops/kernel/q8_0/parallel.rs` 新增：

```rust
pub fn matmul_q8_0_quantized_parallel_batch_tokens(
    weight: &[u8], input_flat: &[f32],
    q8_buf: &mut [u8], scale_buf: &mut [f32],
    output: &mut [f32], n_tokens: usize, n_in: usize, n_out: usize,
    ith: usize, nth: usize,
)
```

输入按 `[n_tokens, n_in]` 行优先排布；单次 `quantize_q8_0_into` 跑整块（`n_tokens × n_in`），输出 `[n_tokens, n_out]`。`pool.compute` 按 `(token, row_out)` 二维分块。`Q8_K` 路径同理。

**调用点替换**（forward.rs 内 12 处 `quantize_and_matmul_with_scratch` 调用，对应 9 个 matmul × 3 个 forward）：

- `forward_dense_attn_layer`:206-217（QKV）、349-353（WO）
- `forward_recurrent_layer`:395-413（4 个 matmul）、557-561（ssm_out）
- `forward_ffn_parallel`:573-580（gate/up）、584-588（down）
- `forward`:155-162（output）

**预估**：prefill -25~35%（385 → ~270 ms），对应文字 prefill `2.5 → 3.4 t/s`、576-token vision prefill `0.1 → 0.3 t/s`。Decode（单 token）走原路径，影响为零。

### 🔴 P0-B：FFN gate+up 真正接通 fuse

**现状**：`ops/quant/fuse.rs:25` 有 `fuse_vstack_q8_0` + 单测，但 loader 没生成 fused 权重，`forward_ffn_parallel:573-580` 始终两次 matmul。

**改动**：

1. `src/models/qwen35/loader.rs`：在加载完 `ffn_gate`/`ffn_up` 后，若两者 `ggml_type == Q8_0` 且 `n_in/n_rows` 一致，调 `fuse_vstack_q8_0` 生成 `ffn_gate_up: Option<Weight<'a>>`。
2. `src/models/qwen35/mod.rs:48-69` `Qwen35LayerWeights` 增加 `pub ffn_gate_up: Option<Weight<'a>>`。
3. `forward_ffn_parallel`:573-580 优先用 fused，单次 matmul 写 `ffn_gate_buf` + `ffn_up_buf`（仍要两个输出缓冲区，因为后续 `silu_mul_approx_inplace` 需要逐元素乘）。

**预估**：FFN -33%（48 → 32 ms / prefill），约 -8% 总耗时。

### 🟡 P1-A：dense val_gather_dot 重构

**现状**：`forward.rs:307-334` 每个 (head, dim) 都重走 `v_cache` 的 stride-dim 访问。

**改动**：先把当前 token 涉及的所有 `v_cache[s * v_dim + kv_h * n_embd_head + d]` 重排到 `attn_value_buf[s * n_embd_head + d]`（一次性 `s × d` 内层循环），再 `vec_dot_f32(score_buf, values_packed)` 拿所有 dim。等价于 softmax(QK^T) @ V 的 inner-product 形式。

**预估**：dense attn -30%，即 val_gather_dot 从 48 → 34 ms（prefill 节省 ~14 ms ≈ 4%）。

### 🟡 P1-B：recr conv1d AVX2 + 并行

**现状**：`forward.rs:424-440` 纯标量 fma，无 SIMD，无 `pool.compute`。

**改动**：

1. 内层 `k` 循环 `_mm256_fmadd_ps`（per token 一次 state shift + 一次 d_conv=4 tap 求和）。d_conv=4 直接走标量也无妨，关键是 SIMD 后整段可以并行。
2. 外层 `c` 维（conv_dim=4608）用 `pool.compute` 拆 8 线程，每个 c 块独立（state 切片不同 dim）。

**预估**：conv1d 从 24 → 10 ms（prefill 节省 ~14 ms ≈ 4%）。

### 🟡 P1-C：recr SSM 外层并行 + ssm_outer_product_update 向量化

**现状**：

- `forward.rs:501-523`：`for t in 0..n_tokens { for v_h in 0..num_v_heads { ... } }` 串行。`v_h` 间 state slice 独立（`v_h * head_v_dim * head_v_dim`），可并行。
- `ops/ssm.rs:89-97`：`ssm_outer_product_update` 内层 `vec_mad_f32` 逐元素。无 AVX2 分支（对比 `ssm_state_decay:19-33` 和 `ssm_matvec:67-87` 已有 AVX2）。

**改动**：

1. `for v_h` 用 `pool.compute` 并行（每个 v_h 独立的 state slice，互不冲突）。`t` 维保持串行（state 跨 token 串行依赖）。
2. `ssm_outer_product_update` 加 `_mm256_fmadd_ps` 内层。`d * dim + i` 索引模式可一次算 8 个 i。

**预估**：SSM 4 ops 从 48 → 24 ms（prefill 节省 ~24 ms ≈ 6%）。

### 🟡 P1-D：dense score_dot GEMV 化（长上下文）

**现状**：`forward.rs:314-318` 逐 s `dot_f32`，每次 stride-k_dim 访问 cache。

**改动**：prefill 阶段把 K 一次性按 head 重排到 `k_packed[head, n_attend, n_embd_head]`（stride-1 内层），对每个 head 做一次 `vec_dot_f32(Q[head, dim], K_packed[head, s, dim])`。

**预估**：45-token 不明显；>512 token 时（vision 576 token 路径）显著。当前 `score_dot ~10 ms` 维持不变。

### 🟢 P2-A：kv_cache_pos O(1) 计数器

**现状**：`models/qwen35/scratchpad.rs:96-106` 每次 forward 都从 0 线性扫到 `k_len / k_dim`，逐块检查"是否全 0"。decode 路径每 forward 扫一遍。

**改动**：在 `Qwen35Scratchpad` 加 `kv_positions: Vec<usize>`（`n_layer` 长），`kv_cache_store` 时同步自增。`kv_cache_pos` 退化为查表。prefill 路径直接传 `pos = 0`。

**预估**：<1% 总耗时。顺带修一个潜在 bug（cache 满了之后 `pos + n_tokens > k_len` 会越界）。

### 🟢 P2-B：scratchpad 零分配

**现状**：

- `forward.rs:147`：`let mut normed = vec![0.0f32; n_tokens * n_embd]`
- `forward.rs:347`：`let mut result = vec![0.0f32; n_tokens * n_embd]`（dense）
- `forward.rs:555`：`let mut result = vec![0.0f32; n_tokens * n_embd]`（recr）
- `forward.rs:163`：`let mut result = vec![0.0f32; cfg.vocab_size]`（logits）

每次 forward 都 malloc；24 层 × N 次 prefill token = 大量 alloc。

**改动**：复用 `scratch.buf` / 新增 `scratch.attn_out_dense` / `scratch.attn_out_recr` 字段；layer forward 改为 `&mut [f32]` 输出而非返回 `Vec`。Logits 用 `scratch.matmul_out`（已存在）。

**预估**：profile 不可见，但减少 jemalloc 压力，长上下文 multi-stream 更友好。

### 🟢 P2-C：kv_cache_store 单次大 copy

**现状**：`models/qwen35/scratchpad.rs:117-122` 逐 token `copy_from_slice(k_dim)`。`n_tokens=45` × 12 dense 层 × 多次 = ~540 次小 copy。

**改动**：用 `c.k.copy_within(il*k_len + pos*k_dim, il*k_len + (pos+n_tokens)*k_dim, ...)` 一次到位（要求 src/dst 不重叠——prefill 是从 `scratch.k_buf` 写 `c.k`，decode 是就地更新——两种情况分别处理）。

**预估**：<1% 总耗时。

### 🟢 P2-D：散点向量化（l2_norm / sigmoid_f32 / softplus_f32）

**现状**：`models/qwen35/util.rs:29-40` 全标量。`forward.rs:405`（sigmoid_f32）、`411`（softplus_f32）、`461-462`（l2_norm）逐元素调用。

**改动**：util.rs 各函数加 AVX2/NEON path（参照 `ops/ssm.rs:19-33` 已有写法），保持 `f32` 位精确（数值必须通过现有 `qwen35_*_matches_pinned_llama_cpp_bits` 测试）。

**预估**：~1-2% 总耗时。

## 预估整体收益（重做后）

| 阶段 | 当前（推算） | 优化后 | 加速比 |
|---|---|---|---|
| 文字 prefill（45 tokens） | ~0.385 s（2.5 t/s） | ~0.21 s（~4.7 t/s） | 1.8× |
| 大图 prefill（576 vision + 17） | ~5.9 s（0.1 t/s） | ~2.5 s（~0.4 t/s） | 2.4× |
| Decode（1 token） | ~20 ms（35 t/s） | ~13 ms（~55 t/s） | 1.5× |

主要来自 P0-A + P0-B + P1-A/B/C。

## 移除 / 不再适用的条目

1. **"fuse_vstack 只支持 Q4K/Q5K/Q6K"** — 已过时。`fuse_vstack_q8_0` 已实现。改为 P0-B 的"接通 fuse"任务。
2. **"aarch64 parity-trace 不要跑"** — 已删。parity-trace 仍存在（`#[cfg(feature = "parity-trace")]`），但与 release 路径无关。
3. **"Persistent Workers 重写 pool.compute"** — 仍无此改动。建议先做 P0-A（batch-token matmul 自然摊薄调度开销），再做 persistent worker 才划算。
4. **原版行号引用（`qwen35.rs:522, 933-944, 1041-1062, 1154-1168, 1314-1335, 1362-1372, 1374-1386` 等）** — 全部失效。上表已用 `forward.rs / scratchpad.rs / util.rs` 新行号重定位。
5. **"45-token prefill 测量表"（attn=64.8%, ffn=34.7%）** — 拆分结构变了，但 matmul 占比分布大体一致。本表不再逐子项给百分比，避免被陈旧数字误导；改为按代码位置定位。

## 复现命令

> 测量脚本未随模块化重构更新。当前 profile 输出仍是 PROFILE_QWEN35 全局计时（forward.rs:41/182/369），不打印子步骤。

```bash
cargo build --release --bin rust-model-inference

# 文字 prefill + decode
PROFILE_QWEN35=1 target/release/rust-model-inference \
    --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf \
    --prompt "法国的首都是巴黎。巴黎是法国的首都，位于该国中北部，横跨塞纳河两岸，是欧洲大陆最重要的政治、文化和金融中心之一。" \
    --max-tokens 1 --temp 0 2>&1 | grep -E '^PROFILE|^\s+(dense_attn|recr)' | head -30

# 含图片的 VL prefill
PROFILE_QWEN35=1 target/release/rust-model-inference \
    --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf \
    --mmproj models/Qwen3.5-0.8B-GGUF/mmproj-F16.gguf \
    --image models/test768.png \
    --prompt "描述这张图片" \
    --max-tokens 1 --temp 0 2>&1 | grep -E '^PROFILE|^\s+(dense_attn|recr)|Vision tokens'
```

## 已知谨慎处理

1. **Q8_0 数值精度**：release x86_64 上与 llama.cpp md5 比对验证（logits 逐 bit 一致）。aarch64 跑 parity-trace 时存在已知不一致，见原版注释。
2. **FFN 融合边界**：P0-B fuse 的只是 stage1（gate + up 拼成一个 weight），后续 `silu_mul_approx_inplace` 仍是两次访存 + 一次乘。不要尝试把 down 也 fuse 进去（依赖前一步输出，shape 不匹配）。
3. **batch-token matmul 与 KV cache 的关系**：prefill 期间 KV cache 一次性写入，但 dense 注意力层内仍要 score over cached K/V。batch-token matmul 只解决 QKV/WO 自身的并行度，不解决 score_dot 的 cache 访问模式。