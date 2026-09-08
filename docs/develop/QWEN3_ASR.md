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

- llama.cpp / GGML 导出器在导出 conv2d 权重时，**把最后两维 `[ic, oc]` 换成了 `[oc, ic]`**（OC-first 利于 SIMD GEMM 内核连续访问），但保留了 dims 字段没更新。
- conv2d 代码隐式利用了这个 layout：`bytes[oc * patch_len..]` 切片恰好按 `[ic, ky, kx]` 顺序排，跟 patch 完美对齐。
- 现有单输出通道测试（`conv2d_stride2_padding_and_layout_are_exact`）用全 1.0 权重，**任何 layout 都通过**，所以这个 layout quirk 一直没被发现。

**优化时不能假设 dims attribute = 实际字节顺序**。任何 conv2d 的字节重排/转置 SIMD 优化，都必须保留 `[oc, ic, ky, kx]` 的字节访问模式，或者在加载时重排到一个显式的新 layout。

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