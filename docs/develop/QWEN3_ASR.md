# Qwen3-ASR — 实现笔记与优化路线

Qwen3-ASR（`models/qwen3/asr/`）是仓库里第一个 ASR pipeline，结合了：
- audio encoder（`mel_encoder.rs`，log-Mel → 3×conv2d → 18 层 transformer → projector）
- text decoder（标准 Qwen3 trunk，`models/qwen3/trunk/`）

本文档记录两个主题：
1. **conv2d 权重布局发现**（一个非标准的 GGUF 维度排序，导致现有测试无法捕捉 layout 错误）
2. **优化路线图**（已做的 + 待做的 SIMD 机会）

---

## 1. conv2d 权重布局发现

### 现象

`mel_encoder.rs::conv2d_stride2_padding1` 第 1000 行附近：
```rust
for output_channel in 0..weights.output_channels {
    let weight_byte = output_channel * patch_len * 2;
    let sum = ... + dot_f16_f16_bytes(
        &patch,
        &weights.weight.bytes[weight_byte..weight_byte + patch_len * 2],
        patch_len,
    );
}
```

`patch` 用 `(ic * 3 + ky) * 3 + kx` 索引（`[ic, ky, kx]` 顺序），但 GGUF 报告的权重 dims 是 `[3, 3, ic, oc]`。

直觉上这意味着权重切片顺序应该是 `[ky, kx, ic]`，跟 patch 不一致 → 应该有 bug。

但 ASR 输出文本（`早上好，今天是二零二零年十月二十九日，最低温度是零下三度。`）完全正确。

### 验证

`mel_encoder.rs::tests::conv2d_layout_matches_oc_ic_ky_kx_byte_order` 用**非均匀权重** + **多通道**（`IC=1, OC=4` / `IC=2, OC=3` / `IC=3, OC=2`）构造了一个对照测试：

1. 在 `bytes[oc * patch_len + ic * 9 + ky * 3 + kx]` 位置写入 `(ky*9 + kx*3 + ic*7 + oc*11) * 0.125` 作为权重
2. 跑 `conv2d_stride2_padding1`
3. 用同样的字节布局做 brute-force 参考实现
4. 三组 `(IC, OC)` 都 bit-exact 通过。

### 结论

**实际 GGUF 字节 layout 是 `[oc, ic, ky, kx]`**（kx innermost），即使 dims attribute 写的是 `[kH, kW, ic, oc]`。

#### 什么是 dims attribute？

GGUF 文件格式里每个 tensor 的 header 包含一个 `ne_dimensions` 字段（`uint64[n_dims]`），**这是 tensor 的逻辑 shape**——模型作者/导出器填的"这个权重代表什么"。对 conv2d，约定俗成是 `[kH, kW, ic, oc]`。这是**逻辑维度声明**，不是字节存储顺序。

#### 实测确认（2026-09-08）

在 `load_conv2d` 加临时 debug eprintln，跑真实 ASR：

```
[conv2d-dbg] a.conv2d.1.weight file_dims=[3, 3, 1, 480]   constructed_dims=[3, 3, 1, 480]
[conv2d-dbg] a.conv2d.2.weight file_dims=[3, 3, 480, 480] constructed_dims=[3, 3, 480, 480]
[conv2d-dbg] a.conv2d.3.weight file_dims=[3, 3, 480, 480] constructed_dims=[3, 3, 480, 480]
```

`file_dims` 是 GGUF header 实际报的值，`constructed_dims` 是我们 `load_conv2d` 用来 `static_tensor` 校验的值——**两者一致，都写的是 `[kH, kW, ic, oc]`**。

#### 那 bytes 顺序怎么是 `[oc, ic, ky, kx]`？

llama.cpp / GGML 导出器在把 PyTorch conv2d 权重写进 GGUF 时，**把逻辑维度的最后两维 `[ic, oc]` 物理换序成了 `[oc, ic]`**（OC-first 让 SIMD GEMM kernel 可以对每行连续访问同一 output channel 的全部 weights），但**保留了 dims 字段没更新**。

#### 转化链（推荐记忆方式）

```
PyTorch 逻辑:  [kH=3, kW=3, ic, oc]
                    ↓ flatten kernel 到一维
合并的 shape:   [kH*kW=9, ic, oc]                ← 物理 stride: ic * oc * 9
                    ↓ OC-first swap
GGUF bytes:     [oc, ic, kH*kW=9]                ← 物理 stride: oc * (ic * 9)
                                                    每个 oc 段 = ic * 9 个 f16
                                                    ic 段内按 [ky, kx] 排 (kx innermost)
```

为什么是 OC-first 而不是 IC-first：GGML SIMD GEMM 内核（`ggml-cuda.cu::ggml_cuda_conv_2d`、`ggml-cuda.cu::conv2d_mul_mat_f16_f32`）对每个 output channel 做一次 matmul，OC 作为 outer dim 让每行/每段连续访问同一 oc 的全部权重，缓存局部性更好。

这是个 GGML 内部的隐性约定——所有走 `ggml_conv_2d` 的权重都是这个 layout。参考：
- llama.cpp `convert_hf_to_gguf.py` 里的 conv2d 写入逻辑（外部工具）
- `ggml-cuda/ggml-cuda.cu` 的 `ggml_cuda_conv_2d` 读取逻辑（OC-major）

#### dims 写得对吗？

作为**逻辑 shape**（PyTorch 视角）：✅ 正确——`[kH, kW, ic, oc]` 描述了 kernel size、输入通道数、输出通道数。
作为**bytes 索引器**：❌ 误导——正确的 dims 描述（跟 bytes 对齐）应该是 `[kH, kW, oc, ic]` 或者 `[oc, ic, kH, kW]`，但 GGML 没更新。

实操：
- 从 dims 读 `oc`（=`dims[3]`）✅ 正确
- 用 `bytes[oc * (ic * 9 * 2)..]` 访问 oc 的权重 ✅ 正确（stride 用了 ic * 9，没用 dims 直接算）

#### conv2d 代码为什么正确？

```rust
let weight_byte = output_channel * patch_len * 2;
let sum = ... + dot_f16_f16_bytes(
    &patch,
    &weights.weight.bytes[weight_byte..weight_byte + patch_len * 2],
    patch_len,
);
```

`bytes[oc * patch_len..]` 切片**恰好按 `[ic, ky, kx]` 顺序排**（每个 oc 一个 patch_len 大小的连续段），跟 patch buffer 索引 `(ic * 3 + ky) * 3 + kx` 完美对齐。代码隐式假设了这个 layout，没写注释——所以单通道测试用全 1.0 权重**任何 layout 都通过**，layout quirk 一直没被发现。

#### 推论

**优化时不能假设 dims attribute = 实际字节顺序**。任何 conv2d 的字节重排/转置 SIMD 优化，都必须保留 `[oc, ic, ky, kx]` 的字节访问模式，或者在加载时显式重排到一个新的、命名清晰的 layout（例如 `weight_oc_major`），并在注释里注明这是从 GGML 字节序转出来的。

---

## 1.5 conv2d SIMD 现状与优化路线

### 现状：部分 SIMD，但不是瓶颈

`conv2d_stride2_padding1` 内部结构：

```rust
for oy in 0..H {                              // ← 外层是 (oy, ox)
    for ox in 0..W {
        patch.fill(0.0);                      // ❌ 标量 fill（patch_len 个 f32）
        for ic in 0..IC {                     // ❌ 3×3 gather 带 if continue 分支
            for ky in 0..3 {
                if py == 0 || py > H { continue; }   // ❌ 分支
                for kx in 0..3 {
                    if px == 0 || px > W { continue; } // ❌ 分支
                    patch[(ic*3+ky)*3+kx] = input[...];
                }
            }
        }
        for oc in 0..OC {                     // ❌ 内层 oc 循环
            let weight_byte = oc * patch_len * 2;
            let sum = dot_f16_f16_bytes(      // ✅ AVX2 F16C + FMA（patch_len ≥ 16 时）
                &patch,
                &bytes[weight_byte..weight_byte + patch_len * 2],
                patch_len,
            );
            output[(oc * H + oy) * W + ox] = sum + bias[oc];
        }
    }
}
```

**已 SIMD**：`dot_f16_f16_bytes` 内层（`ops/dot.rs:159`）是 AVX2 F16C + FMA，8 elements/iter。Conv1/conv2 patch_len=4320 → 完整走 SIMD path。

**未 SIMD**：
- `patch.fill(0.0)`（patch_len 个 f32 store，conv1/conv2 每次 4320 floats = 17KB）
- 3×3 gather 带 `if continue` 分支
- 跨 oc 循环的权重读：每个 `(oy, ox)` 都重读全部 oc×ic×9 = 2.07M 个 f16 = 4.1MB 权重 → **cache thrashing**

### 真实瓶颈：权重 cache miss，不是 dot

实测 `3xconv2d=430ms`：

| 阶段 | 耗时 | 性质 |
|------|------|------|
| Conv0 (1→480, patch_len=9) | ~10ms | 标量 dot fallback（n < 16），循环开销 |
| Conv1 (480→480, patch_len=4320) | ~200ms | **AVX2 dot 跑得很快，但每个 (oy,ox) 重读 4MB 权重** |
| Conv2 (480→480, patch_len=4320) | ~220ms | 同上 |

按比例估：dot 计算本身（已 AVX2）只占 ~30%，**剩下 70% 是 patch buffer 反复 fill + 跨 oc 权重读取的 cache miss**。

### 优化方案（按收益/风险排序）

#### 方案 A：交换 oc / (oy, ox) 循环顺序（推荐首选）

```rust
// 当前：(oy, ox) 外，oc 内
for oy in 0..H { for ox in 0..W {
    patch.fill(0); gather_patch();
    for oc in 0..OC { dot(patch, bytes[oc*patch_len..]); }   // 每次跨 OC 重读权重
}}

// 改后：oc 外，(oy, ox) 内
for oc in 0..OC {                                    // 权重切片一次加载到 L1
    let weight_slice = &bytes[oc * patch_len * 2..][..patch_len * 2];
    for oy in 0..H {
        for ox in 0..W {                              // 权重一直在 L1 hot
            patch.fill(0); gather_patch();
            output[(oc * H + oy) * W + ox] = dot(patch, weight_slice) + bias[oc];
        }
    }
}
```

**收益**：
- 每个 oc 的权重切片 = `ic × kH × kW × 2` bytes
  - conv0: 1 × 9 × 2 = 18B
  - conv1/conv2: 480 × 9 × 2 = 8.6KB ← **装得进 L1d (32KB)**
- 全部 `(oy, ox)` 复用同一份权重 → cache hit rate 接近 100%
- 预估 **2-3× speedup** for conv1/conv2，**总 ~150-250ms 收益**

**风险**：低。layout 完全保留（`bytes[oc * patch_len..]` 切片语义不变），测试 `conv2d_layout_matches_oc_ic_ky_kx_byte_order` bit-exact 通过就行。输出位置 `output[(oc * H + oy) * W + ox]` 跟原来一样（oc 在外层，只是赋值时机不同）。

#### 方案 B：多 oc SIMD tile

```rust
for oc_tile in 0..OC step 4 {                       // 同时算 4 个 oc
    for (oy, ox) in pixels {
        patch.fill(0); gather_patch();
        // 4 oc × patch_len FMA，权重 contiguous
        let sums = dot4_oc(patch, &bytes[oc_tile..][..4*patch_len*2]);
        for k in 0..4 {
            output[((oc_tile+k) * H + oy) * W + ox] = sums[k] + bias[oc_tile+k];
        }
    }
}
```

**收益**：在方案 A 基础上再 +30-50%，但 layout 验证复杂（要保证 `bytes[oc_tile..(oc_tile+4)*patch_len..]` 是 `[4个 oc, ic, ky, kx]` 连续排，符合 OC-major）。

**风险**：中。要新写 `dot4_oc` 函数 + 验证权重连续性。

#### 方案 C：Conv0 (patch_len=9) inline FMA

patch_len=9 太小（< 16），`dot_f16_f16_bytes` 掉到 scalar fallback。可以 inline 9 个 `f32::mul_add` 替代函数调用。

**收益**：~5ms（微小），但代码简单。

### 推荐路径

1. **方案 A（必做）**：循环交换，~30 行代码改动，预估 150-250ms 收益，零风险。
2. **方案 C（顺手）**：conv0 inline FMA，5 行改动，~5ms 收益。
3. **方案 B（看情况）**：如果方案 A 后还是 LLM decode 阶段（4.3s）盖住了 audio encode（1.4s），且 conv2d 还是 200ms+ 才考虑。

### 验证

任何 conv2d 重构必须：
1. `cargo test --release --lib conv2d_layout_matches_oc_ic_ky_kx_byte_order` 三组 `(IC, OC)` bit-exact 通过
2. ASR 端到端推理输出文本与参考一致（`早上好，今天是二零二零年十月二十九日，最低温度是零下三度。`）
3. `cargo bench`（如果有）显示 conv2d 阶段耗时下降

---

## 2. ASR 性能基线（2026-09-08，Windows MSVC，T8）

测试用例：`models/001_16k.wav`（中文天气广播，~5.3s 音频），`--max-tokens 256`，`--threads 8`：

```
load_decoder=0.156s load_runtime=0.018s transcribe=5.0s
ASR: 97 prompt tokens, 78 audio tokens, 23 output tokens

decode_wav=0.000s
mel=0.029s                       ← log-Mel + FFT
  audio_encode=1.3s
    3xconv2d=0.43s               ← conv0(1→480) + conv1(480→480) + conv2(480→480)
    project_f16=0.015s           ← post-conv linear
    per-window conv=0.45s        ← 78 窗口 × 3 conv（每窗口 5.7ms）
    per-window layers=0.83s      ← 78 × 18 层（每层 46ms）
  llm_generate=3.7s               ← text decoder prefill + 23 decode steps
```

**总 wall time ~5s**，其中 audio encoder 占 26%，LLM decode 占 74%。

### ASR 阶段热点表

| 阶段 | 耗时 | 频率 | SIMD 状态 |
|------|------|------|-----------|
| `compute_log_mel`（FFT + Mel filter） | 30ms | 1 次 | 部分（4-way unroll，LLVM 兜底） |
| `apply_conv2d`（3 层 conv） | 430ms | 78+1 次 | AVX2 已就绪（`dot_f16_f16_bytes` n≥16 触发），但 patch buffer 重读权重 |
| `apply_gelu_erf`（`erff` extern） | < 5ms × 36 | 36 次 | 标量（libm `erff`） |
| `add_residual` / `add_position_embeddings` | < 1ms × 36 | 36 次 | 标量 element-wise |
| `flatten_conv_output`（transposition） | 5-10ms | 1 次 | 标量嵌套循环 |
| `layer_norm` | < 1ms × 36 | 36 次 | **✅ 已用 `ops::sum_f32` + `sum_sq_centered_f32`** |
| `AudioLinear::project`（FFN/attention） | ms 级 | 36 × 4 = 144 | **✅ Q8_0 SIMD 量化 matmul** |

---

## 3. 已完成

- ✅ Layer norm 用 `ops::sum_f32` + `ops::sum_sq_centered_f32`（AVX2/NEON），从手写 5 个 simd_* helper 改为 fused loop
- ✅ 移除 macOS Accelerate / vDSP 依赖（`audio_processor.rs`），改用 LLVM 自动向量化的 f32 helper
- ✅ `dots/speaker.rs::hypotf` → `f32::hypot`（Windows MSVC 兼容性 fix，顺手做的）
- ✅ `mel_encoder.rs` 死代码清理（删未使用的 `matmul_q8_0_quantized_range*` import）

---

## 4. 待做优化（按收益/风险排序）

### P0 — 收益大但需要小心

#### 4.1 `apply_conv2d` patch buffer 消除

**问题**：当前实现对每个 (output_pixel, output_channel) 都重建 patch buffer + 重读 4MB 权重。`dot_f16_f16_bytes` 本身是 AVX2，但每像素权重读 800 次 → cache thrashing。

**方案 A（最小改动）**：保持 patch 顺序，改成 inline fma（去掉 `dot_f16_f16_bytes` 函数调用开销）。Conv0（patch_len=9，scalar fallback）受益最大；conv1/2 已经有 AVX2 但函数调用开销省掉。

**方案 B（彻底改）**：output_channel 外层循环（权重一次读到 L1，所有像素复用）+ 每个 channel 独立 patch。改 layout 后可以做多通道 SIMD tile，但要保留 `[oc, ic, ky, kx]` 字节访问模式（或加载时转置到新 layout）。

**风险**：必须保留 layout（参见第 1 节），改完跑 `conv2d_layout_matches_oc_ic_ky_kx_byte_order` + ASR 推理输出对比。

### P1 — 安全收益

#### 4.2 `add_residual` → `ops::vec_add_f32_inplace`

```rust
// 当前 (mel_encoder.rs:1295)
for (hidden, update) in hidden.iter_mut().zip(update) {
    *hidden += *update;
    if !hidden.is_finite() { ... }
}

// 目标
crate::ops::vec_add_inplace(hidden, update);  // LLVM 自动向量化为 _mm256_add_ps
// 再用 zip().any(|v| !v.is_finite()) 检查有限性
```

36 次/ASR，每次 ~微秒级，零风险。

#### 4.3 `add_position_embeddings` → 行级 vec_add

```rust
// 当前 (mel_encoder.rs:1272)
for token in 0..hidden.tokens {
    let position = token % 13;
    for lane in 0..width {
        hidden.values[token * width + lane] += positions[position * width + lane];
    }
}

// 目标：position 是 token % 13，按 token 行做 vec_add
// 或预展开 13 个位置的 vec_add
```

1 次/ASR，~5ms 收益，零风险。

#### 4.4 `flatten_conv_output` → 行级 transpose

```rust
// 当前 (mel_encoder.rs:1033): 3 层嵌套
for time_index in 0..time {
    for channel in 0..channels {
        for mel in 0..mel_bins {
            output[time_index * features + feature] = input[(channel * mel_bins + mel) * time + time_index];
        }
    }
}

// 目标：改成 channel × mel 维度直接 copy + 时间维 SIMD memcpy
// 形状 (channels=480, mel=16, time=13) → output (time=13, channels=480, mel=16)
```

~5-10ms 收益，零风险。

### P2 — 收益小但代码量大

#### 4.5 `apply_gelu_erf` → SIMD-friendly 近似

`gelu_erf` 用 `erff()` extern（C libm 标量），无法 SIMD。

替代方案（按精度递减）：
- `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))` —— SIMD 友好
- Rational 近似（更便宜，~1e-3 误差）

36 次/ASR，每次 ~微秒级。LLM text decoder 的 SiLU 已经做了类似近似，对 ASR 文本输出来说 GELU 近似误差可接受。

#### 4.6 `compute_log_mel` Mel filter apply

4-way unroll 已经 LLVM-vectorized，再推 8-way 收益边际。mel 时间只 30ms（占总 0.6%），不值得优化。

---

## 5. 不要碰的地方

- **`apply_conv2d` 的字节切片模式 `bytes[oc * patch_len * 2..]`** —— 这是 `[oc, ic, ky, kx]` layout 的隐式依赖。任何 SIMD tile 重构都要么保留这个模式，要么显式加载时重排。
- **`f16_to_f32` / `f32_to_f16` 在 mel_encoder.rs 里的用法** —— 已用 `ops::` 公开 API，避免重复实现。
- **`mel_filters` / `periodic_hann_window`** —— one-time init，不在 hot path。

---

## 6. 验证测试

新加的 `mel_encoder.rs::tests::conv2d_layout_matches_oc_ic_ky_kx_byte_order` 是关键 regression 测试。任何 conv2d 重构必须保持这个测试 bit-exact 通过 + ASR 推理输出文本不变。

```bash
# 跑单测
cargo test --release --lib conv2d_layout

# 跑 ASR 端到端
./target/release/rust-model-inference.exe \
  --model models/Qwen3-asr-gguf/Qwen3-ASR-0.6B-Q8_0.gguf \
  --mmproj models/Qwen3-asr-gguf/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
  --audio models/001_16k.wav --language Chinese --max-tokens 256 --threads 8
# 期望: 早上好，今天是二零二零年十月二十九日，最低温度是零下三度。
```