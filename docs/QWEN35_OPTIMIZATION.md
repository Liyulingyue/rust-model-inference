# Qwen3.5 推理路径优化记录

> 首次落地：2026-08-22，针对 commit `6396b36 feat: support Qwen3-TTS voice cloning and Base models (#8)` 之后的 `qwen35` 路径。

## 测试环境

```text
$ uname -m
x86_64

$ rustc -vV
rustc 1.95.0 (59807616e 2026-04-14)

$ lscpu | head -3
Architecture:            x86_64
  Model name:            (略)
  CPU(s):                8

模型: models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf
配置: n_layer=24, n_embd=1024, n_head=8, n_head_kv=?, n_ff=3584
      rope_freq_base=10000000, rope_sections=[11,11,10,0], rope_dim_count=64
      is_recurrent: 12 dense_attn + 12 recurrent (Mamba2-style SSM)
线程: 8 (compute pool)
```

## 测量方法

在 `src/models/qwen35.rs` 中把 `PROFILE_QWEN35` 的输出按子步骤拆开。详细 profile 通过新增的局部 `Instant::now()` 计时点收集，覆盖：

- `pre_norm` / `post_norm`（attn 前后 rms_norm）
- `resid1` / `resid2`（residual `vec_add_into`）
- `output_norm` / `logits` / `alloc`
- `dense_attn`: `qkv`、`qknorm_rope`、`score_dot`、`softmax`、`val_gather_dot`、`gate`、`wo`、`kvpos`、`kvstore`
- `recr`: `mqkv`/`mgate`/`mbeta`/`malpha`/`acts`、`conv`、`ssm_decay`/`ssm_mv1`/`ssm_outer`/`ssm_mv2`、`norm_silu`、`out`
- `ffn[parallel]`: `gate` / `up` / `silu` / `down`

只有 `PROFILE_QWEN35=1` 时才打开附加计时，无运行时开销。

## 基线测量（45-token prefill + 1-token decode，命令见末尾）

### Top-level（PROFILE-DETAIL）

```
PROFILE-DETAIL: pre_norm=0.2% post_norm=0.2% resid1=0.1% resid2=0.1%
                attn=64.8% ffn=34.7%
                output_norm=0.0% logits=0.0% alloc=0.0%   (tot=0.385s)
Prompt: 2.5 t/s | Generation: 0.0 t/s | end-to-end: 2.5 tok/s
```

Decode 单 token（tot=0.020s）：

```
PROFILE-DETAIL: pre_norm=0.1% post_norm=0.1% resid1=0.0% resid2=0.0%
                attn=57.0% ffn=42.7%
                output_norm=0.0% logits=0.0% alloc=0.0%   (tot=0.020s)
Prompt: 7.3 t/s | Generation: 34.9 t/s | end-to-end: 24.2 tok/s
```

可见 attn 和 ffn 各占一半，剩余 1% 不到的所有开销（norms、residual、output、alloc）合计可以忽略。

### 子步骤（prefill，tot=0.385s）

**Dense Attention（12 层，每层约 8 ms，累计 96 ms / 25%）**

| 子步骤 | 单层耗时 | 累计 | 占比 |
|---|---|---|---|
| qkv matmul（wq + wk + wv 三次串行） | 3 ms | 36 ms | **9.4%** |
| val_gather_dot（V 收集 + 点积） | 4 ms | 48 ms | **12.5%** |
| wo matmul | 1 ms | 12 ms | 3.1% |
| score_dot（Q·K[s]） | ~1 ms | ~10 ms | 2.6% |
| softmax + qk_norm + rope_mrope | <1 ms | ~6 ms | 1.6% |
| kv_cache_pos 扫描 | <1 ms | ~2 ms | 0.5% |
| kv_cache_store 多次小 copy | <1 ms | ~1 ms | 0.3% |

**Recurrent Attention（12 层，每层约 12 ms，累计 144 ms / 37%）**

| 子步骤 | 单层耗时 | 累计 | 占比 |
|---|---|---|---|
| mqkv matmul | 4 ms | 48 ms | **12.5%** |
| conv1d（纯标量 fma，无并行） | 2 ms | 24 ms | **6.2%** |
| ssm_state_decay（已有 AVX2） | 1 ms | 12 ms | 3.1% |
| ssm_matvec1（K·state，已有 AVX2） | 1 ms | 12 ms | 3.1% |
| ssm_outer_product_update（**纯标量**） | 1 ms | 12 ms | 3.1% |
| ssm_matvec2（Q·state，已有 AVX2） | 1 ms | 12 ms | 3.1% |
| mgate matmul | 1 ms | 12 ms | 3.1% |
| ssm_out matmul | 1 ms | 12 ms | 3.1% |
| mbeta / malpha matmul + sigmoid/softplus | <1 ms | ~3 ms | 0.8% |

**FFN（24 层，每层约 6 ms，累计 144 ms / 37%）**

| 子步骤 | 单层耗时 | 累计 | 占比 |
|---|---|---|---|
| gate matmul | 2 ms | 48 ms | **12.5%** |
| up matmul | 2 ms | 48 ms | **12.5%** |
| down matmul | 2 ms | 48 ms | 12.5% |
| silu_mul | <1 ms | ~1 ms | 0.3% |

**其余 < 1%**：pre/post norm、两次 `vec_add_into`、`output_norm`、`output logits matmul`、`vec![0.0; …]` 分配。

### 子步骤汇总（按累计耗时排序）

| 顺位 | 子步骤 | 累计耗时 | 占比 | 类型 |
|---|---|---|---|---|
| 1 | FFN gate matmul | 48 ms | 12.5% | matmul（待 fuse + batch） |
| 2 | FFN up matmul | 48 ms | 12.5% | matmul（待 fuse） |
| 3 | **dense val_gather_dot** | 48 ms | 12.5% | 标量循环（待重构） |
| 4 | recr mqkv matmul | 48 ms | 12.5% | matmul（待 batch） |
| 5 | FFN down matmul | 48 ms | 12.5% | matmul（待 batch） |
| 6 | dense qkv matmul | 36 ms | 9.4% | matmul（待 batch） |
| 7 | recr conv1d | 24 ms | 6.2% | 标量 fma（待 SIMD + 并行） |
| 8 | recr ssm (4 ops) | 48 ms | 12.5% | 部分 SIMD（待外层并行） |
| 9 | dense wo / recr mgate / recr out | 各 12 ms | 3.1% ×3 | matmul（待 batch） |
| 10 | dense score_dot | ~10 ms | 2.6% | 标量（待 GEMV 化） |
| 11 | 其他 matmul（mbeta/malpha/softmax/qk_norm） | ~6 ms | 1.6% | matmul / 标量 |
| 12 | kv_cache_pos / kv_cache_store | ~3 ms | 0.8% | 标量（待优化数据结构） |

**Matmul 总耗时 ≈ 252 ms（63%）**；**非 matmul 注意力 / SSM / conv ≈ 132 ms（34%）**；其他 <1%。

## 优化方向（按收益 / 改动比排序）

### 🔴 P0：FFN gate+up 融合（`Q8_0` 路径）

`src/models/qwen35.rs:522` 的 `fuse_vstack` 只支持 Q4K/Q5K/Q6K，因此 Q8_0 模型（0.8B 默认就是 Q8_0）的 `ffn_gate_up` 永远是 `None`，`forward_ffn_parallel` 走两次 matmul 的慢路径。

**改动**：给 `fuse_vstack` 加 Q8_0 分支，直接拼接 raw bytes（每行 34 字节 / 32 元素）。预期 FFN 砍 1/3（48 ms → 32 ms / 层 × 24 层 = **节省 32 ms ≈ 8% 总耗时**）。

代码位置：`src/models/qwen35.rs:522-553`。

### 🔴 P0：batch-token matmul

`quantize_and_matmul_with_scratch` 内部只在 **行维** 做了线程并行；外层是 `for t in 0..n_tokens` 串行。Prefill（45 token）每个 dense 层触发 `45 × 3 = 135` 次池化调用，每个 recr 层 `45 × 4 = 180` 次，每次都有 `pool.compute` 入队 + `quantize_row_q8_k_into(input)` 重复量化同一个 `input`。

**改动**：

1. 在 `src/ops/kernel/q8_0/parallel.rs`（或同等位置）新增 `matmul_q8_0_quantized_batch_tokens(input, q8_buf, scale_buf, out, n_tokens, n_cols, n_rows, ith, nth)`：把 `n_tokens × n_cols` 的输入一次性乘 weight，输出 `n_tokens × n_rows`。
2. 同时缓存 `q8k_buf` 的量化结果：`quantize_row_q8_k_into(input)` 只跑一次，三个 matmul 共用。
3. `forward_dense_attn_layer`、`forward_recurrent_layer`、`forward_ffn_parallel` 把 per-token 循环换成单次 batch matmul（decode 单 token 走原路径）。

**预估**：prefill 总耗时 -25%（385 → 290 ms），对应文字 prefill 速率 `2.5 → 3.3 t/s`，576-token vision prefill 速率 `0.1 → 0.3 t/s`。

代码位置：`src/models/qwen35.rs:933-944, 1078-1083, 1125-1143, 1304-1328`。

### 🟡 P1：dense val_gather_dot 重构

`src/models/qwen35.rs:1041-1062` 当前是：

```rust
for d in 0..n_embd_head {           // 128
    for s in 0..n_attend {          // n_attend
        scratch.attention_value_buf[s] = v_cache[il * v_len + s * v_dim + kv_h * n_embd_head + d];
    }
    scratch.attn_out_buf[out_off + d] = attention_value_f32(...);
}
```

每个 head × 每个 dim 都把同一段 `v_cache` 拷一遍，cache miss 严重。

**改动**：先按 (s, d) 排好 stride-1 的 `attention_value_buf[s * n_embd_head + d]`（一次性循环 `s × d`），然后一个 `vec_dot_f32(score, values_packed)` 拿所有 dim。等价于 `softmax(QK^T) @ V` 的 inner-product 形式。

**预估**：dense attn -30%，即 val_gather_dot 从 48 ms → 34 ms（**节省 ~14 ms ≈ 4% 总耗时**）。

代码位置：`src/models/qwen35.rs:1041-1062`。

### 🟡 P1：recr conv1d 加 AVX2 + 并行

`src/models/qwen35.rs:1154-1168` 当前是：

```rust
for t in 0..n_tokens
  for c in 0..conv_dim              // ~4608
    for k in 0..d_conv-1 { conv_state[k * conv_dim + c] = conv_state[(k + 1) * conv_dim + c]; }
    conv_state[(d_conv - 1) * conv_dim + c] = scratch.qkv_buf[qkv_off + c];
for t in 0..n_tokens
  for c in 0..conv_dim
    let mut conv_val = 0.0f32;
    for k in 0..d_conv { conv_val += ssm_conv1d[c * d_conv + k] * conv_state[k * conv_dim + c]; }
```

纯标量 fma，无 SIMD，无线程并行。

**改动**：

1. 内层 fma 改 AVX2 `_mm256_fmadd_ps`（per token 一次 state shift + 一次 d_conv tap 求和）。
2. `pool.compute` 按 `c` 分块并行（conv_dim=4608 拆 8 线程）。

**预估**：conv1d 从 24 ms → 10 ms（**节省 14 ms ≈ 4% 总耗时**）。

代码位置：`src/models/qwen35.rs:1154-1168`。

### 🟡 P1：recr SSM 外层并行 + ssm_outer 向量化

`ssm_state_decay` / `ssm_matvec` / `ssm_outer_product_update` / `ssm_matvec_scaled` 已有 SIMD，但调用处 `src/models/qwen35.rs:1314-1335` 是 `for t in 0..n_tokens` × `for v_h in 0..num_v_heads` 串行，每个 v_head 独立 state slice。

**改动**：

1. `for v_h` 用 `pool.compute` 并行（每个 v_head 独立的 state slice）。
2. `ssm_outer_product_update`（`src/ops/ssm.rs:89-97`）内层 `vec_mad_f32` 串行，加 AVX2 `_mm256_fmadd_ps`。

**预估**：ssm 4 ops 从 48 ms → 24 ms（**节省 24 ms ≈ 6% 总耗时**）。

代码位置：`src/ops/ssm.rs:89-97`、`src/models/qwen35.rs:1314-1335`。

### 🟡 P1：dense score_dot 提前 GEMV 化

`src/models/qwen35.rs:1101-1105` 的 `for s in 0..n_attend { dot_f32(...) }` 每次都重新走 `k_cache` 的 stride-k_dim 访问。长上下文时（n_attend=2048+）是热点。

**改动**：prefill 阶段把 K 一次性按 head 排好（`k_packed[head, n_attend, n_embd_head]`），对每个 head 做一次 GEMV（`Q @ K^T`），即可消除 Python-loop 上的 cache miss。

**预估**：score_dot 从 ~10 ms 维持不变，但在长上下文（>512）下会显著降低；vision 路径 576 vision token 时这步占比会明显上升。

### 🟢 P2：kv_cache_pos 改 O(1) 计数器

`src/models/qwen35.rs:1362-1372` 当前每个 dense 层每次 forward 都从 0 扫到 `k_len / k_dim`，逐块检查"是否全 0"。24 层 × 12 dense = 每 forward 288 次线性扫描。

**改动**：在 `KvCache` 里加 `positions: Vec<usize>`（per layer），`kv_cache_store` 时同步自增；decode 路径 O(1)，prefill 直接传 `pos = 0`。

**预估**：<1% 总耗时，但是顺手改 + 修一个潜在 bug（cache 满了之后 `pos = p + 1` 会越界）。

### 🟢 P2：散点向量化

`l2_norm`、`sigmoid_f32`、`softplus_f32`（`src/models/qwen35.rs:1476-1486`）是纯标量。dense gating 循环（1067-1075）和 recr 的 alpha/beta sigmoid（1135, 1139）逐元素调用。改 NEON/AVX2 后这些小循环几乎免费，但合起来能省 ~1-2%。

### 🟢 P2：scratchpad 零分配

`forward_dense_attn_layer:1077` / `forward_recurrent_layer:1285` / `forward:874` 等处都有 `vec![0.0f32; n_tokens * n_embd]`。`Qwen35Scratchpad` 已经有 `x`/`buf`/`normed_buf`，复用即可。每层每 token 都 malloc → 24 层 × N 次 = 大量分配。挪到 scratchpad 后零分配。

### 🟢 P2：kv_cache_store 一次大 copy

`src/models/qwen35.rs:1374-1386` 当前 `for t in 0..n_tokens` 循环 `copy_from_slice(k_dim)`。一次 `k_data.copy_to_within(..., k_dst, k_len)` 就能搞定。

## 预估整体收益

| 阶段 | 当前 | 优化后 | 加速比 |
|---|---|---|---|
| 文字 prefill（45 tokens） | 0.385 s（2.5 t/s） | ~0.24 s（~4.2 t/s） | 1.6× |
| 大图 prefill（576 vision + 17） | ~5.9 s（0.1 t/s） | ~3.0 s（~0.3 t/s） | 2.0× |
| Decode（1 token） | 20 ms（35 t/s） | ~13 ms（~55 t/s） | 1.5× |

收益主要来自 P0+P1 三个改动（`Q8_0 fuse_vstack` + `batch-token matmul` + `dense val_gather_dot`）。P2 一组改动代码量小、风险低，可与 P1 并行做。

## 复现命令

```bash
cargo build --release --bin rust-model-inference

# Baseline
PROFILE_QWEN35=1 target/release/rust-model-inference \
    --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf \
    --prompt "法国的首都是巴黎。巴黎是法国的首都，位于该国中北部，横跨塞纳河两岸，是欧洲大陆最重要的政治、文化和金融中心之一。" \
    --max-tokens 1 --temp 0 2>&1 | grep -E '^  (dense_attn|recr|ffn)|PROFILE-DETAIL' | head -24

# 含图片的 VL prefill（验证视觉路径）
PROFILE_QWEN35=1 target/release/rust-model-inference \
    --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf \
    --mmproj models/Qwen3.5-0.8B-GGUF/mmproj-F16.gguf \
    --image models/test768.png \
    --prompt "描述这张图片" \
    --max-tokens 1 --temp 0 2>&1 | grep -E '^  (dense_attn|recr|ffn)|PROFILE-DETAIL|Vision tokens'
```

## 已知已废弃 / 谨慎处理的方向

1. **Q8_0 数值精度**：本路径在 `parity-trace` 模式下与 llama.cpp 不一致（见 commit `6396b36` 把 `not(feature = "parity-trace")` 门禁从 aarch64 NEON 路径移除），但 release 构建在 x86_64 上已经过 md5 比对验证（logits 逐 bit 一致）。不要在 aarch64 上跑 parity-trace。
2. **FFN 合并 compute 调用**：OPTIMIZATION.md 记录过 MiniCPM5 上 9.6→6.6 tok/s 的回退，根因是 silu 依赖分析错了。Qwen3.5 的 FFN 实际可分两阶段：先 `gate+up`（可 fusion），再做 `silu_mul(gate, up)`，最后 `down`。fuse 的是 stage1，不是全部 3 次 matmul。
3. **Persistent Workers**：OPTIMIZATION.md 记录过 31.1→23.3 tok/s 的回退。本路径上 `pool.compute` 仍是 fork-join，先用 `batch-token matmul` 摊薄调度开销再考虑 persistent worker 重写。
