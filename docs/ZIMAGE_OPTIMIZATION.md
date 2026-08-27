# Z-Image 性能分析 & 优化方向

> 本文档基于 `src/models/diffusion/z_image/{mod,text,dit,vae}.rs` 的实测 profile + 代码审查得出。
> 测试条件：64×64 分辨率、5 步、seed=42、8 线程、`AVX2+FMA`。
> **真实场景将使用 1024×1024**：latent token 数从 64 → 16384（×256），所有 per-token 工作按线性放大。

---

## 1. Z-Image 管线架构

```
ZImagePipeline::generate_rgb(prompt)
  │
  ├─ text.encode_layer_35(prompt)            ← Qwen3 4B (35 层)
  │     16 tokens × 35 layers
  │     每 layer: rms_norm + 4×linear + rope + attention + 3×linear + swiglu
  │
  ├─ dit.denoise(context, ...)               ← 5 × 30 blocks, latent 64 tokens
  │     每 block: AdaLN + 30-layer DiT (含 qkv/out/w1/w3/w2 + rope + attention)
  │
  └─ vae.decode_rgb(latent, ...)            ← latent 8×8 → RGB 64×64
        conv_in + mid(2 resblock + 1 attn) + 4 upsample stage + final conv
```

### 1.1 当前实测耗时（64×64 baseline）

| 阶段 | 耗时 | 占比 |
|------|------|------|
| **text_encode** | 2,041 ms | 2.5% |
| **denoise** | 77,216 ms | **93.1%** |
| └─ linear ffn (w1+w3+w2) | 48,719 ms | (62% denoise) |
| └─ linear qkv | 18,402 ms | (24%) |
| └─ linear out | 5,645 ms | (7%) |
| └─ attention_into | 2,684 ms | (3.5%) |
| └─ other (rms/swiglu/add_residual) | 911 ms | (1.2%) |
| └─ modulation / rope / rms_norm | 252 ms | (0.3%) |
| **vae_decode** | 3,745 ms | 4.5% |
| └─ upsample stages | 3,602 ms | (96% VAE) |
| └─ mid (residual + attention) | 125 ms | (3%) |
| └─ conv_out + post | 17 ms | (0.5%) |
| **总耗时** | **83,003 ms** | 100% |

### 1.2 1024 分辨率下的预期

| 阶段 | 当前 (@64) | 预估 (@1024) | 备注 |
|------|-----------|--------------|------|
| text_encode | 2.0s | **~2.0s** | 与图像分辨率无关（仅依赖 prompt 长度） |
| denoise | 77s | **~5.5 小时**（latent ×256，每 op 计算/内存 ×256）| per-token 工作 |
| vae_decode | 3.7s | **~15 分钟**（像素 ×256）| O(pixels²) |
| 总计 | 83s | **~5.5 小时** | denoise 仍是绝对主导 |

---

## 2. 节点级 SIMD 状态总表

### 2.1 Text Encoder（`text.rs`）

| Node | 实现位置 | 当前 SIMD | @64 耗时 | @1024 耗时 | 优化方向 | Approx? |
|------|---------|----------|---------|------------|----------|---------|
| `forward_to_block` 主循环 | text.rs:119 | 串行 prefill | 2.0s | 不变 | - | - |
| `embedding_lookup` | text.rs:161 | `Weight` API（Q8_0 SIMD） | 微小 | 微小 | - | ❌ |
| `linear_into` (q,k,v,o,gate,up,down) | text.rs:169-272 | ✅ AVX2 NR=4, 持久线程池 | 主体 | 不变 | - | ❌ |
| `rms_norm` sum_sq | text.rs:163,236 | ❌ **标量** | ~50ms | ~50ms | 写 `sum_sq_f32_avx2` | ❌ |
| `rms_norm_inplace` sum_sq | text.rs:201,204 | ❌ **标量** | ~30ms | ~30ms | 写 `sum_sq_f32_avx2` | ❌ |
| `rope_neox` | text.rs:206,207 | ❌ **标量 sin/cos**（macOS 除外） | ~40ms | ~40ms | macOS 已有 `__sincosf`；其他平台需 AVX2 sincos | ❌ |
| `attention_dot` (Q@K) | text.rs:406 | ✅ AVX2 (`dot_f32`) | 微小 | 微小 | - | ❌ |
| `attention_softmax` | text.rs:395 | ❌ **标量**（`softmax_inplace` 精确版） | 微小 | 微小 | `softmax_approx_inplace_avx2` | ⚠️ 是 |
| **`qwen_attention_value`** | text.rs:372 | ⚠️ x86 路径退化到**标量 fold**！| 微小 | 微小 | **改用 `attention_value_f32`（已 SIMD）** | ❌ |
| `qwen_swiglu` | text.rs:385 | ❌ `silu_mul_inplace` 标量精确 | ~100ms | ~100ms | `silu_mul_approx_inplace_avx2` | ⚠️ 是 |

**Text Encoder 关键观察**：
- 几乎所有耗时在 7 个 `linear_into`（已 SIMD）。
- **与图像分辨率无关**——只受 prompt 长度影响（当前 16 tokens）。
- `qwen_attention_value` 在 x86 上有个**回归**：本来 `attention_value_f32` 已经有 AVX2 实现，但 text.rs:378 的 x86 分支写成了 `fold` 而不是 `attention_value_f32`，**这是 bug**。

---

### 2.2 DiT Core（`dit.rs`）

| Node | 实现位置 | 当前 SIMD | @64 耗时 | @1024 预估 | 优化方向 | Approx? |
|------|---------|----------|---------|-----------|----------|---------|
| `run_block` 主循环 | dit.rs:1066 | 复杂 | denoise 主体 | ×256 | - | - |
| **`linear_into` (qkv/out/w1/w3/w2)** | dit.rs | ✅ AVX2 NR=4, persistent pool | 94.5% | ×256 | - | ❌ |
| `quantize_q8_0_into` | q8_0.rs:176 | ✅ AVX2 | 含在 linear | - | - | ❌ |
| `q8.prepare()` input量化 | mod.rs:611 | ✅ AVX2 | 含在 linear | - | - | ❌ |
| `modulation` linear (AdaLN) | dit.rs:1097 | ✅ AVX2 NR=4 | 54ms | 54ms | - | ❌ |
| **`rms_norm` sum_sq** | dit.rs:1218 | ❌ **标量** | ~50ms | **~13s** | 写 `sum_sq_f32_avx2` | ❌ |
| **`rms_norm_inplace` sum_sq** (K/Q head, ffn_norm2) | dit.rs:1158,1168 | ❌ **标量** | ~30ms | **~7s** | 写 `sum_sq_f32_avx2` | ❌ |
| `scale_modulated_branch` | dit.rs:404 | ✅ SIMD FMA (`vec_mad_self_f32`) | 28ms | ~7s | - | ❌ |
| **`add_modulated_residual`** | dit.rs:417 | ❌ **标量** | ~500ms | **~128s** | 见下 | - |
| └─ 无 gates 路径 | dit.rs:432 | ❌ 标量 | ~300ms | ~77s | 改 `vec_add_into`（已 SIMD） | ❌ |
| └─ 有 gates 路径（含 `tanh`） | dit.rs:427 | ❌ 标量 + stdlib tanh | ~200ms | ~51s | 见下方 3 选项 | - |
| `attention_into` | dit.rs:535 | 部分 SIMD | 2.7s | **~12 min** | 见下 | - |
| └─ `dot_f32` (Q@K) | dit.rs:569 | ✅ AVX2 | 微小 | - | - | ❌ |
| └─ `softmax_inplace` | dit.rs:576 | ❌ **标量精确** | 估 ~50% attn | **~5.7 min** | `softmax_approx_inplace_avx2` | ⚠️ 是（NaN 风险）|
| └─ `attention_value_f32` (A@V) | dit.rs:584 | ✅ AVX2 | 微小 | - | - | ❌ |
| └─ 外层 query/key/head loop | dit.rs:554-586 | ❌ 串行 | - | - | rayon parallel per-head | ❌ |
| **`z_image_swiglu`** (= `silu_mul_inplace`) | dit.rs:592 | ❌ **标量精确** | ~500ms | **~128s** | `silu_mul_approx_inplace_avx2` | ⚠️ 是 |
| **`rotate_interleaved_inplace`** | dit.rs:522 | ❌ 标量 2-elem | ~50ms | **~13s** | AVX2 一次 8 对（16 f32） | ❌ |
| `layer_norm_no_affine` (final) | dit.rs:596 | aarch64 NEON，x86 **标量** | 微小 | 几秒 | x86 写 AVX2 mean+var+norm | ❌ |
| `force_f32_linear_into` (final_layer.linear) | dit.rs:257 | ⚠️ f16→f32 **标量** + dot AVX2 | 微小 | 微小 | 批量 f16→f32 + multi-row matmul | ❌ |
| `timestep_embedding` | dit.rs:504 | 标量 cos/sin | 微小（每 step 1 次） | 不变 | 预计算缓存 | ❌ |
| `z_image_rope_into` | dit.rs:1631 | 标量 cos/sin | 1 次预计算 | 不变 | 预计算缓存 | ❌ |
| `patchify_latent_into` | dit.rs:1495 | ❌ 标量复制 | 微小（1 次） | 微小 | AVX2 memcpy | ❌ |
| `unpatchify_latent_into` | dit.rs:1572 | ❌ 标量复制 | 微小 | 微小 | AVX2 memcpy | ❌ |
| `sign_and_unpatchify_image` (final sign+bias) | dit.rs:473 | ❌ 标量 | 微小 | 微小 | AVX2 | ❌ |
| `rng.normal_fill_16` | dit.rs:1391 | ❌ 标量 | 微小 | 微小 | 预生成 batch | ❌ |

**DiT 关键观察**：
- linear matmul 是绝对热点（94.5%），但**已 78% 内存带宽极限**，进一步优化空间 <3%。
- **周边 kernel 在 @1024 下变成主要瓶颈**：rms_norm/silu/swiglu/attention softmax 等。
- `add_modulated_residual` 有两条路径：
  - **无 gates**：直接用 `vec_add_into`（已 SIMD），改动 1 行即可获得 77s 收益。
  - **有 gates**：需要 SIMD tanh，可选 `(A) tanh_approx_avx2` 2e-3 误差 / `(B) 自己写 Padé SIMD` 1e-6 / `(C) 只 SIMD 化乘加，stdlib tanh`。

---

### 2.3 VAE Decoder（`vae.rs`）

| Node | 实现位置 | 当前 SIMD | @64 耗时 | @1024 预估 | 优化方向 | Approx? |
|------|---------|----------|---------|-----------|----------|---------|
| `vae_softmax_inplace` (= `softmax_inplace`) | vae.rs:986 | ❌ **标量** | 微小 | ~10s | `softmax_approx_inplace_avx2` | ⚠️ 是 |
| **`upsample_nearest_then_conv`**（3 stage ×2×）| vae.rs:1022 | 调用 `conv_f16_into` | **3.6s**（96% VAE） | **~15 min** | 见下 | - |
| └─ `upsample_nearest_into` | vae.rs:989 | ❌ 标量内存复制 | ~30% upsamples | ~4 min | AVX2 strided copy | ❌ |
| └─ `padded_conv_f16_into` (3×3 kernel) | vae.rs:775 | ❌ per-pixel 6 重循环 | ~70% upsamples | - | im2col + 批量 matmul | ❌ |
| **`conv_f16_into`**（3×3/1×1 kernel） | vae.rs:677 | ❌ per-pixel 6 重循环 | ~1.5s | **~6.4 min** | im2col + 批量 matmul | ❌ |
| └─ `dot_f16_f16_bytes` (inner dot) | vae.rs:759 | ✅ AVX2 F16C FMA | 微小 | - | - | ❌ |
| └─ `f32_to_f16` (patch 转换) | vae.rs:751 | ⚠️ 标量 | 部分 | - | 改用 `f32_slice_to_f16_avx2`（已存在）| ❌ |
| **`group_norm_32_into`** | vae.rs:798 | ❌ **标量 4-pass**（mean/var/scale/bias） | ~900ms | **~3.8 min** | 融合为 2-pass AVX2 | ❌ |
| `silu_inplace_checked` | vae.rs:886 | ❌ **标量** | 微小 | ~几秒 | `silu_approx_inplace_avx2` | ⚠️ 是 |
| **`run_residual_block`** | vae.rs:431 | 调用上面所有 | ~2.6s | **~11 min** | 依赖上述 kernel 优化 | - |
| └─ identity residual | vae.rs:509 | ❌ 标量 add | 微小 | ~几秒 | `vec_add_into` | ❌ |
| **`run_attention`** | vae.rs:520 | 部分 SIMD | ~125ms | **~32s** | 见下 | - |
| └─ `one_head_spatial_attention_into` | vae.rs:928 | ❌ **完全标量** | ~125ms | **~32s** | 用 `dot_f32_avx2` + AVX2 softmax + AVX2 matmul | ❌ |
| `add_shortcut_residual_into` | vae.rs:894 | ❌ conv + 标量 add | 估 ~15% VAE | ~几秒 | AVX2 add | ❌ |
| `diffusion_to_vae` | vae.rs:591 | 标量 × channels | 微小 | 几秒 | AVX2 | ❌ |
| `rgb_bytes_from_channels` | vae.rs:1051 | 标量 × channels | 微小 | 几秒 | AVX2 + clamp | ❌ |

**VAE 关键观察**：
- **VAE 几乎全标量**！除 `dot_f16_f16_bytes` 外没有任何优化。
- 内层 `dot_f16_f16_bytes` 用 F16C + AVX2 + FMA（最高效），但**外层 6 重 per-pixel 循环的调度开销巨大**。
- @1024 下 VAE 总耗时可能达 15-20 分钟——**这是当前最大的优化机会**。
- `conv_f16_into` 应改 **im2col + 单次大批量 matmul**（参考 llama.cpp SD 实现）。
- `group_norm_32_into` 是 4 趟独立循环（line 828-881），可以融合为 1-2 趟 AVX2。

---

### 2.4 Pipeline 顶层（`mod.rs` / `pig`）

| Node | 实现位置 | 当前 SIMD | @64 耗时 | 优化方向 | Approx? |
|------|---------|----------|---------|----------|---------|
| `ZImagePipeline::generate_rgb` | mod.rs:48 | 串行调度 | 100% | N/A | - |
| `tensor_info / tensor_slice` 重复查询 | mod.rs:633,651 | n/a | 占 linear ffn 总开销 ~5% | **缓存 weight 指针**（load 阶段一次性 load） | ❌ |

**注**：`pig.rs` 是独立的旧 backbone，**当前 Z-Image 路径不使用它**。`src/lib.rs:36` 仍导出，但 `z_image/{mod,text,dit,vae}.rs` 完全自包含。

---

## 3. 优化优先级分类

### 🥇 第一批：必须做（@1024 致命瓶颈，无 approx）

| 优化 | 改动范围 | @64 收益 | @1024 收益（估）| Approx |
|------|---------|---------|----------------|--------|
| **VAE `conv_f16_into` im2col + 批量 matmul** | 重写 conv（重构） | ~1.5s | **~6 min** | ❌ |
| **VAE `group_norm_32_into` 4-pass → 2-pass AVX2** | 改 norm | ~0.9s | **~3.8 min** | ❌ |
| **VAE `upsample_nearest_into` AVX2 复制** | 1 函数 | ~1s | **~4 min** | ❌ |
| **DiT `rms_norm` sum_sq AVX2** | 1 新函数 + 1 行调用 | ~50ms | **~13s** | ❌ |
| **DiT `rms_norm_inplace` sum_sq AVX2** | 1 新函数 + 2 行调用 | ~30ms | **~7s** | ❌ |

### 🥈 第二批：强烈推荐（@1024 显著收益，无 approx）

| 优化 | 改动范围 | @64 收益 | @1024 收益（估）| Approx |
|------|---------|---------|----------------|--------|
| DiT `attention_into` 内部 SIMD + softmax 优化 | 多处 | ~1s | **~4 min** | ❌ |
| VAE `one_head_spatial_attention_into` 改用 `dot_f32` + softmax SIMD | 重写 | ~125ms | ~32s | ❌ |
| DiT `add_modulated_residual` 无 gates 路径改 `vec_add_into` | **1 行** | ~300ms | **~77s** | ❌ |
| DiT `rotate_interleaved_inplace` AVX2 8-pair | 1 新函数 | ~40ms | ~10s | ❌ |
| **Text `qwen_attention_value` 修复（用 `attention_value_f32`）** | **1 行（修 bug）** | ~5ms | ~5ms | ❌ |
| DiT `layer_norm_no_affine` AVX2（x86 路径） | 1 新函数 | 微小 | ~几秒 | ❌ |
| VAE `f32_to_f16` patch 转换改 AVX2 | 1 行 | 微小 | ~几秒 | ❌ |
| VAE `add_shortcut_residual_into` AVX2 add | 1 行 | 微小 | ~几秒 | ❌ |
| VAE `rgb_bytes_from_channels` AVX2 + clamp | 1 新函数 | 微小 | ~几秒 | ❌ |
| DiT `timestep_embedding` 预计算缓存 | 1 cache | 微小 | 微小 | ❌ |
| Pipeline `tensor_info/tensor_slice` 缓存（load 阶段一次性）| Block 重构 | ~5% linear | ~5% linear | ❌ |
| DiT `patchify_latent_into`/`unpatchify_latent_into` AVX2 | 2 函数 | 微小 | 微小 | ❌ |
| DiT `sign_and_unpatchify_image` AVX2 | 1 函数 | 微小 | 微小 | ❌ |
| DiT `force_f32_linear_into` 批量 f16→f32 | 重构 | 微小 | 微小 | ❌ |
| VAE `diffusion_to_vae` AVX2 | 1 函数 | 微小 | ~几秒 | ❌ |

### 🥉 第三批：Approx 优化（需评估精度 vs 收益）

| 优化 | Approx 误差 | @1024 收益 | 风险评估 | 建议 |
|------|-----------|----------|----------|------|
| DiT `silu_mul_inplace` → `silu_mul_approx_inplace_avx2` | exp ~1e-5 | ~2 min | **低**（Qwen3 TTS 也用近似 exp，精度有保障） | ✅ 推荐 |
| DiT `add_modulated_residual` tanh → `tanh_approx_avx2` | 2e-3（AdaLN gate 范围） | ~51s | **中**（需像素 diff 测试；AdaLN gate 串联效应） | ⚠️ 先测试图差异 |
| DiT `softmax_inplace` → `softmax_approx_inplace_avx2` | exp | ~2.5 min | **高**（之前 Z-Image 因 approx softmax 触发 NaN，已修复） | ❌ 暂不建议 |
| VAE `silu_inplace` → `silu_approx_inplace_avx2` | exp ~1e-5 | 几秒 | **低**（收益小） | ✅ 可做 |
| VAE softmax → approx | exp | ~10s | **中**（与 DiT 同源风险） | ⚠️ 与 DiT 同步决策 |
| Text `silu_mul_inplace` → approx | exp | ~50ms | **低** | ✅ 可做 |
| Text `softmax_inplace` → approx | exp | 微小 | **中** | ⚠️ 与 DiT 同步 |

### ⚪ 不值得做

| 项 | 原因 |
|----|------|
| DiT linear matmul 进一步优化 | 已 78% 内存带宽极限 |
| DiT `rope_neox`（仅 text encoder 用）| 仅 ~40ms，与分辨率无关 |
| DiT `timestep_embedding` 标量 cos/sin | 每 step 仅 1 次，~微秒级 |
| DiT `z_image_rope_into` | 1 次预计算（10ms 内）|
| DiT `patchify/unpatchify_latent_into` | 每图 1 次 |
| DiT `rng.normal_fill_16` | 微小 |
| DiT `modulation` matmul | 仅 54ms（小 matmul）|

---

## 4. 当前 SIMD 实现参考

### 4.1 Q8_0 矩阵乘法（DiT 主力）

调用链（修复 `matmul_q8_0_quantized_dynamic` → `parallel_rows` 后）：

```
dit.rs:1224  linear_into(source, &block.w1, ...)
  ↓
mod.rs:617  linear_into_scaled_impl()
  ↓ (我修改前用 broken _dynamic, 现用 parallel_rows + pool.compute)
mod.rs:618  pool.compute(move |ith, nth| {
              matmul_q8_0_quantized_parallel_rows(bytes, &q8.values, &q8.scales,
 output, n_in, n_out, ith, nth) })
  ↓
parallel.rs:130  matmul_q8_0_quantized_range(...)    // per-worker row range
  ↓
dispatch.rs:72  matmul_q8_0_vs_q8_0_avx2(...)        // x86_64 + AVX2 检测
  ↓
avx2.rs:53  // NR=4 主循环 + 末尾 single-row fallback
```

**NR=4 含义**：内层循环一次同时累加 4 个输出行（共享 input Q8 load），独立 ymm 累加器避免行间数据依赖，让 CPU 乱序执行/ILP 最大化。

**当前状态**：8 线程 + persistent pool + NR=4 已达 ~78% 内存带宽极限（实测 48.7s/理论 38s 下限）。

### 4.2 已有的 SIMD 工具函数（可直接复用）

| 函数 | 路径 | 用途 |
|------|------|------|
| `vec_add_into(a, b)` | ops/activation/vector.rs:65 | b += a（精确 AVX2）|
| `vec_add(a, b, c)` | ops/activation/vector.rs:124 | c = a + b |
| `vec_mul(a, b)` | ops/activation/vector.rs | 逐元素乘 |
| `vec_mad_f32(y, x, v)` | ops/dot.rs:508 | y += x*v |
| `vec_mad_self_f32(y, x)` | ops/dot.rs:565 | y += y*x（AdaLN scale）|
| `vec_scale_f32(y, v)` | ops/dot.rs:411 | y *= v |
| `dot_f32(a, b, n)` | ops/dot.rs:349 | 标量 dot（AVX2 内核）|
| `dot_f16_f16_bytes(a, b, n)` | ops/dot.rs:121 | f16 dot（F16C+AVX2+FMA）|
| `attention_value_f32` | ops/attention.rs | weighted sum（AVX2）|
| `scale_mul_avx2(scale, w, x)` | ops/norm.rs:65 | x = x*scale*w |
| `silu_approx_inplace_avx2` | ops/activation/silu.rs:32 | SiLU（approx）|
| `silu_mul_approx_inplace_avx2` | ops/activation/silu.rs:100 | SwiGLU（approx）|
| `tanh_approx_avx2` | ops/math/tanh.rs:67 | tanh（2e-3 误差）|
| `exp_approx_avx2` | ops/math/exp.rs:44 | exp（约1e-5）|
| `softmax_approx_inplace_avx2` | ops/softmax.rs:44 | softmax（exp approx）|
| `quantize_q8_0_into_avx2` | ops/quant/q8_0.rs:176 | Q8 量化（AVX2）|

### 4.3 标量 fallback / 无 SIMD 的函数

| 函数 | 路径 | 现状 |
|------|------|------|
| `sum_sq_f32` | (无) | 需要新增——`rms_norm` / `rms_norm_inplace` 内部 |
| `tanh_inplace_avx2` | ops/math/tanh.rs:27 | **名义 AVX2 但内部仍是 stdlib 标量 tanh**——无意义，需重写 |
| `softmax_inplace`（精确）| ops/softmax.rs:4 | 标量 |
| `layer_norm_no_affine` x86 路径 | dit.rs:596 | aarch64 有 NEON，x86 是标量 |
| `rotate_interleaved_inplace` | dit.rs:522 | 标量 2-elem |
| `upsample_nearest_into` | vae.rs:989 | 标量复制 |
| `group_norm_32_into` | vae.rs:798 | 标量 4-pass |
| `conv_f16_into` 外层循环 | vae.rs:677 | 标量 6 重 per-pixel |

---

## 5. SIMD 设计建议

### 5.1 复用现有精确算子的路径（无 approx）

```rust
// add_modulated_residual 无 gates 路径（dit.rs:432）：
vec_add_into(residual, tokens);  // 1 行，已 SIMD

// rms_norm 系列：写 sum_sq_f32_avx2（一次性 reduce 8 elements）：
//   AVX2: loadu_ps 8 floats, fma self, horizontal add
//   + 现有 scale_mul_avx2 已 OK

// rotate_interleaved_inplace：一次处理 8 对 (16 f32)
//   AVX2 FMA: new_first = first*cos - second*sin
//             new_second = first*sin + second*cos

// layer_norm_no_affine：mean + variance + normalize
//   复用 sum_sq_avx2 思路
```

### 5.2 `add_modulated_residual` 有 gates 路径（涉及 tanh）

```rust
// 选项A：用 tanh_approx_avx2（2e-3 误差，最简单）
//   单行改动，但有精度损失

// 选项B：自己写 Padé[7/6] 近似 tanh（1e-6 误差，比 tanh_approx_avx2 紧）
//   对 |x| < 4 用有理逼近，|x| ≥ 4 直接 ±1
//   仍是 SIMD，但精度更高

// 选项C：stdlib tanh per-element + SIMD FMA 乘加
//   只 SIMD 化 y += a*tanh_buf，tanh 仍 stdlib 标量
//   改动小，无精度损失，但 tanh 部分不加速
```

### 5.3 VAE Conv im2col 改造（最大改动）

当前 `conv_f16_into` (vae.rs:677) 是 per-pixel 6 重循环，每次 inner dot 调 `dot_f16_f16_bytes`（已 SIMD）。

**改造方案**：
1. 用 im2col 把 (H, W, IC) 输入 + kernel 展开成 (H*W, IC*kH*kW) 的列矩阵。
2. 权重 reshape 为 (OC, IC*kH*kW)。
3. 调一次大批量 matmul（用现有 `dot_f16_f16_bytes` 矩阵化版本，或 `matmul_q8_0_vs_q8_0_avx2` 改 F16 input）。
4. 一次性写回 output。

参考 llama.cpp 的 `ggml_compute_forward_conv_2d` 实现。

---

## 6. 已经修复的关键 bug

### 6.1 `linear_into_scaled_impl` 用 broken `matmul_q8_0_quantized_dynamic`

**问题**（修复前）：每次 matmul 调用都 `std::thread::spawn` 7 个新线程（绕过 persistent pool）。2250 次 matmul/图 = ~15000 次线程创建。

**修复**：改用 `pool.compute(move |ith, nth| matmul_q8_0_quantized_parallel_rows(...))`。

**结果**：80s（之前 110s，加速 27%）。

### 6.2 `softmax_approx_inplace_avx2` 在 Z-Image NaN

**问题**：`exp_approx_avx2` 截断到 ±88.376，对 softmax 输入大正数 OK，但 Z-Image 的 attention 中存在大负数 `dot * scale` → `(x - max).exp()` 后需要处理 ~-88 以下的情形。

**当前解决**：dit.rs:576 使用 `softmax_inplace`（精确标量版），attention_into 不切到 AVX2。

---

## 7. 测试与验证流程

每次优化后必须验证：
1. **像素对比**：`--prompt "A red fox sleeping beneath a pine tree" --seed 42` 跑出新图，与 baseline 像素差 < 1e-5
2. **Z-Image 内部 parity-trace**：可选，开启 `parity-trace` feature 做逐层数值对比
3. **单元测试**：对应 op 已有 AVX2-vs-scalar 对比测试（如 `matmul_q8_0_avx2.rs:230-298`）

---

## 8. 总结

### 8.1 1024 分辨率下预测的优化收益

**Denoise 周边 kernel（无 approx）**：
- rms_norm ×2: ~20s
- attention_into 优化: ~4 min
- rotate_interleaved: ~10s
- add_modulated_residual (no gates): ~77s
- **小计**：~5-6 min 节省

**VAE 全面 SIMD 化（无 approx）**：
- conv im2col: ~6 min
- group_norm: ~3.8 min
- upsample: ~4 min
- attention 重写: ~32s
- **小计**：~14-15 min 节省

**Approx 优化（带精度风险）**：
- silu_mul approx: ~2 min
- add_modulated tanh approx: ~51s
- **小计**：~3 min 节省

**总潜在节省**：~25-30 min（在 5.5 小时基线上约 8-10% 提速）。

### 8.2 短期建议（立即可做，1-2 小时）

1. **`add_modulated_residual` 无 gates 路径 → `vec_add_into`**（1 行，77s @1024 收益）
2. **`qwen_attention_value` 修复**（1 行修 bug）
3. **`rms_norm` + `rms_norm_inplace` 写 `sum_sq_f32_avx2`**（1 函数 + 替换，~20s @1024 收益）
4. **`rotate_interleaved_inplace` 写 AVX2 8-pair**（1 新函数，~10s @1024 收益）
5. **`silu_mul_inplace` → `silu_mul_approx_inplace_avx2`**（1 行，~2 min @1024 收益；风险低）

### 8.3 中期建议（值得做的较大改动）

- **VAE `conv_f16_into` im2col 改造**（VAE 60% 提速，~6 min @1024）
- **VAE `group_norm_32_into` 2-pass AVX2**（~3.8 min @1024）
- **DiT `attention_into` 内部 SIMD**（~4 min @1024）

### 8.4 不建议（精度风险 / 收益小）

- `softmax_inplace` 改 approx（之前 NaN bug 来源）
- linear matmul 进一步优化（已 memory-bound）
- 各种 < 1% 收益的微优化

---

## 9. VAE Conv 并行优化实验（2026-08-27）

### 9.1 目标与初始预期

原 `conv_f16_into`（vae.rs:677, 旧版已删）是 per-pixel 6 重循环，**VAE up stage 占 3,602 ms**。
预期 im2col + 批量 matmul + pool 并行能拿到 60% 提速（~6 min @1024）。

### 9.2 三轮实验结果（@64 分辨率，5 步，8 线程）

| 阶段 | VAE 总耗时 | VAE up stage | 总耗时 | vs baseline |
|------|-----------|--------------|--------|------------|
| **baseline**（per-pixel + 单线程）| 3,745 ms | 3,602 ms | 83,003 ms | - |
| **实验 1**（im2col + pool）| 845 ms | 790 ms | 79,768 ms | -3.9% |
| **实验 2**（per-pixel + pool，无 im2col）| **575 ms** | **519 ms** | **73,069 ms** | **-12%** |

**关键发现：**

1. **im2col 拖慢了性能**（3,602 → 845 是 4.3× 加速，但**只来自 pool 并行**，不是 im2col）
2. **per-pixel + pool 比 im2col + pool 还快**（575 vs 845，**额外 30% 提速**）
3. **VAE up 实际是 6.7× speedup**（3,602 → 519 ms）

### 9.3 im2col 为什么反而慢

我的初始推理（"im2col 共享 patch 准备，提升 cache 复用"）**是错的**。实测：

| 因素 | im2col | per-pixel + pool |
|------|--------|-----------------|
| im2col 矩阵大小 | 18M f16 = **37 MB**（超 L3）| N/A（每线程9 KB patch，in L1）|
| f32→f16 转换次数 | 18.9M 次（不变） | 18.9M 次 |
| f16→cache reload | 每个 dot reload 4608 f16×1024 oc | 每线程 patch 常驻 L1 |
| L3 miss | 高（37MB 不在 L3）| 低（patch 9KB 在 L1） |
| Worker 同步 | 必须等 im2col 完成后才能 dot | 无（每 worker 独立） |

**37 MB im2col buffer 远超典型 L3 cache（8-30 MB）**——每个 dot 都要重新从内存读，反而拖慢。

### 9.4 教训

#### A. im2col 不总适用于小规模问题

im2col 在 **大型 matmul**（如全连接层、K²×IC=4608×OC=1024）中可能有帮助，因为 dot 工作远大于 patch 准备开销。但在 VAE 这种**中等规模 matmul**（patch 9 KB，dot 4.7M ops per worker），per-pixel + L1 cache 更优。

#### B. 共享 buffer 反而拖慢并行

im2col 的"共享"语义是**单线程顺序时**的优势。但 **8 线程并行**时：
- 所有 worker 等一个共享 buffer 填充完成
- worker 之间争抢 L3 读带宽
- 每 worker 都需要自己处理一个 4608 元素 patch 段

**per-worker 独立 patch buffer**反而更快（每个 patch 9 KB 永远在 L1）。

#### C. profile > 直觉

我最初预估 im2col 收益"显著"，实际**反优化 30%**。教训：**先 profile 实测，再优化**——尤其是涉及 memory layout 的优化。

### 9.5 最终实现（保留）

`src/models/diffusion/z_image/vae.rs` 中的 `conv_f16_parallel_into`：

```rust
fn conv_f16_parallel_into(
    input: &[f32],
    input_channels: usize,
    side: usize,
    weights: &[u8],
    output_channels: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    output: &mut [f32],
    pool: &Arc<ComputePool>,
) -> Result<(), String> {
    // ... validate ...

    let patch_len = kernel * kernel * input_channels;
    let pixel_count = side * side;

    // 准备 weight/bias 数据（共享）
    let weight_rows: Vec<&[u8]> = ...;
    let bias_values: Vec<f32> = ...;

    pool.compute(move |ith, nth| {
        let per_thread = (pixel_count + nth - 1) / nth;
        let start = ith * per_thread;
        let end = (start + per_thread).min(pixel_count);
        let mut patch = vec![0u16; patch_len];  // 每 worker 独立
        for pixel in start..end {
            // 1. 填充 patch（每个 pixel 一次）
            // 2. OC 个 dot products
        }
    });
}
```

**关键点**：
- `pool.compute` 通过 `Arc<ComputePool>` 从 `ZImagePipeline::load` 传入，**与 DiT/text encoder 共享**（不新建线程）
- 每 worker 独立的 `Vec<u16>` patch（9 KB，常驻 L1）
- 不预先构建 im2col 矩阵（避免 L3 压力）

### 9.6 同步改动（保留）

| 改动 | 位置 | 影响 |
|------|------|------|
| `FluxVae` 加 `pool: Arc<ComputePool>` 字段 | vae.rs:168 | 接收共享 pool |
| `FluxVae::load(source, pool)` 签名 | vae.rs:175 | 显式传 pool |
| `ZImagePipeline::load` 传 pool 给所有3个组件 | mod.rs:42-44 | pool 共享 |
| `run_conv` / `padded_conv_f16_into` / `add_shortcut_residual_into` 签名加 `&Arc<ComputePool>` | vae.rs:668, 818, 942 | 调用传入 pool |
| `upsample_nearest_then_conv` 加 pool 参数 | vae.rs:1057 | 测试也更新 |

### 9.7 GroupNorm 实测（2026-08-27）

在 VAE 617 ms（总）中，group_norm **仅 21.5 ms**（3.5%）。

| 操作 | 耗时 | 占比 |
|------|------|------|
| upsample conv（已并行化）| ~570 ms | 92% |
| **group_norm** | **21.5 ms** | **3.5%** |
| 其他 | ~25 ms | 4% |

**结论：group_norm 不是高价值目标**。即使完全优化为0 ms，也只省21.5 ms。

**修正原文档假设**：第 3 章 "VAE group_norm 4-pass AVX2 → ~3.8 min @1024" **不成立**。实际节省估 < 100ms @1024（×4-5 = ~85ms）。

### 9.8 下一步建议（修正版）

按实测数据，VAE 当前状态：

| 阶段 | 耗时 | 占比 | 优化空间 |
|------|------|------|----------|
| upsample conv | ~570 ms | 92% | **已是 dominant** |
| group_norm | 21.5 ms | 3.5% | 跳过（收益小）|
| mid + out | ~40 ms | 6% | 跳过（占比小）|

**真正还能动的方向**：

1. **继续优化 upsample conv**（最大热点）：
   - 共享 kernel window（连续 output_pixel 的 patch 有大量重叠）
   - 多 pixel 共享 patch 准备（tile 4×4）
   - **预期**：20-40% 提速（额外 100-200 ms @64，对应 @1024 ~1-2 min）
2. **跳过 group_norm 优化**
3. **转向 DiT 周边 kernel 优化**（之前估算 @1024 数分钟）

**不要做的**：
- ❌ im2col（已验证是反优化）
- ❌ group_norm AVX2（占比太小）
- ❌ 大型重构（除非有 profile 数据支持）

### 9.9 Tile 优化实验（2026-08-27，已撤回）

#### 思路

实现 4×4 tile 共享输入区域：
- 16 个输出像素共用 (4+K-1)² = 36 个输入位置（每 channel）
- 把 36×IC = 18K f32 读到 contiguous buffer
- AVX2 bulk f32→f16（替代逐元素 stdlib 调用）
- 重组 patches，跑 16 × OC 个 dot products

#### 实测结果（@64 分辨率，5 步，8 线程）

| 配置 | VAE 总耗时 | 总耗时 |
|------|-----------|--------|
| per-pixel + pool（当前）| 575-663 ms | 73,069-83,870 ms |
| **4×4 tile + AVX2 f32→f16** | **622 ms** | **76,373 ms** |

**没有提速，反而略慢**。

#### 为什么 tile 没收益

**理论节省**（4×4 tile = 16 pixels）：
- 输入 reads: 16 × 9 × IC = 144 × IC → tile: 36 × IC（**节省 75%**）
- f32→f16 转换: 73,728 → 18,432（**节省 75%**）

**但实际**：
- f32→f16 本身不是瓶颈（VAE 中估 < 2 ms 总开销）
- 输入 reads 在 NHWC layout 是 strided（ic * spatial 步长），cache miss 在 per-pixel 也部分 cache 友好
- **真正瓶颈是 OC × 4608 个 dot products**（每个 conv ~12M FMA ops）

**f32→f16 SIMD 节省 < 0.5%**，被以下 overhead 抵消：
1. 每 tile copy input region 到 local buffer（额外 ~73 KB 写入）
2. `Vec<Vec<u16>>` 数组的 heap allocation 和间接寻址
3. 嵌套循环和 indexing 计算

#### 教训

#### D. 优化前要算 FLOPs/bytes 比例

`f32→f16` 是个**轻量操作**（每元素 ~1ns AVX2 ~0.1ns）。在 18.9M 次转换中：
- 总开销 ≈ 1.9 ms (scalar) → 0.2 ms (AVX2)
- VAE 总耗时 622 ms
- **优化 f32→f16 节省 1.7 ms = 0.27%**

#### E. NHWC + cache 行为复杂

即使有理论节省，实际 cache 行为取决于：
- L1 size（通常 32 KB）
- L2 size（256 KB - 1 MB）
- L3 size（8-30 MB）
- 内存预取策略

NHWC layout 的输入 `input[ic*spatial + y*side + x]` 在不同 ic 间步进 spatial 字节，**L1 cache line (64 bytes) 只覆盖 ~16 个 ic 值**——比想象的 cache-friendly。

#### F. 优化的真正方向

**真正的瓶颈是 dot products**（OC × patch_len × pixels）：
- 每个 conv: 512 × 4608 × pixels = **2.36M × pixels** 个 FMA ops
- VAE 25 个 convs × avg 4096 pixels = **~240 GFLOPs total**
- @30 GFLOPS/s AVX2 = 8 s theoretical @64；实测 622 ms = 12.8 GFLOPS/s achieved
- **AVX2 dot kernel 已经接近峰值**（dot_f16_f16_bytes 用 _mm256_fmadd_ps）

继续优化 conv 需要：**重写 dot kernel**（如 NR=4 tile + 共享 weight rows）或**完全避免 dot**（如量化中间结果到 int8）。

#### G. 不要为"看起来更高效"的方案花时间

本次 tile 实现代码量增加 **~120 行**，测试 + 调试 + 撤回耗时 ~30 分钟，**收益 = 0**。教训：**先做最小验证**（用 perf counter 单独测 f32→f16 开销）确认是真瓶颈再投入。

---

## 10. 最终建议（更新版）

### 已确认可优化（实测量化）

1. **`pool.compute` 用于 VAE conv per-pixel**（已实施，VAE -77%，保留）

### 已确认不可优化或反优化

1. ❌ **im2col + 大共享 buffer**（反优化 ~30%）
2. ❌ **4×4 tile + AVX2 f32→f16**（持平，f32→f16 不是瓶颈）
4. ❌ **VAE group_norm SIMD**（占比 3.5%，收益 < 22 ms）

### 未来可能的方向（需 profile 数据支持再投入）

1. **DiT 周边 kernel 优化**：
   - `rms_norm` sum_sq（@1024 ~13s 预估）
   - `silu_mul_inplace` → approx（@1024 ~2 min 预估，低风险）
   - `add_modulated_residual` 无 gates → `vec_add_into`（@1024 ~77s 预估）
   - `rotate_interleaved_inplace` AVX2 8-pair（@1024 ~10s 预估）
2. **VAE conv kernel 重写**（高风险，高潜在）：
   - 重写 `dot_f16_f16_bytes` 为 NR=4 multi-row（避免 weight reload）
   - 中间结果量化到 int8（量化噪声风险）
3. **VAE attention 重写**（小占比，3.5%，~32s @1024）

### 优化方法论（从这次实验学到的）

1. **profile 先于优化**：用最小化改动测量每个 op 的实际开销
2. **算 FLOPs/bytes 比例**：高 arithmetic intensity 的 op 才适合 SIMD 优化
3. **小改动验证**：先做最小版本（10-20 行）跑性能，再决定是否投入大改动
4. **记录每次实验**：即使反优化也要记录（避免后续重蹈覆辙）
5. **NHWC layout cache 行为难预测**：依赖 profile，不靠推理