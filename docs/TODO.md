# TODO

## TODO-001: Q4_0 AVX2 kernel 不使用 FMA，性能受限

### 现状

`src/ops/kernel/q4_0/avx2.rs` 的 Q4_0 matmul kernel 故意**不使用 FMA**（`fused multiply-add`）：

```rust
// q4_0/avx2.rs:14
//! 8. Multiply by d * scale in f32 with explicit `_mm256_mul_ps` + `_mm256_add_ps`
//!    (no `_mm256_fmadd_ps`) to match scalar's mul+add rounding exactly.
//!
//! **Precision contract**: bit-exact with the scalar implementation, including
//! for edge cases (all-zero Q8, all-127 Q8, all-0 nibble, all-15 nibble).
```

行累加采用 `acc += dc * d * scale` 三次 f32 运算（mul → mul → add），而不是 `fmadd(dc, d*scale, acc)` 一次融合运算。

### 影响

- **性能**：相比 Q4_K / Q8_0 kernel（已用 FMA）慢约 1.5-2×。E4B Q4_0 文件实测 5.8 t/s decode，E2B Q4_K_M 10 t/s（同样 embd 比例下 kernel 速度是主要差异）。
- **正确性**：与 scalar reference **bit-exact**（`assert_avx2_eq_scalar` 测试在 `q4_0/avx2.rs:241` 验证 `a.to_bits() == b.to_bits()`）。所有 Q4_0 模型通过 parity 测试。

### 何时触发

加载任何 Q4_0 权重时——目前主要是 Gemma 4 E4B Q4_0 导出文件。`require_tensor_any` 列表里 Q4_0 在 `src/models/gemma4/trunk/config.rs:215`。

### 选项

1. **加 FMA + 放松精度契约**（推荐 follow-up）：把 `acc += dc * d * scale` 改成 `acc = _mm256_fmadd_ps(prod_d_scale, ...)`，并把 `assert_avx2_eq_scalar` 改成 ~1 ULP 容忍。需要新增"near bit-exact"测试模式（参考 llama.cpp 的 GGML_FMA_DISABLED 编译开关做法）。
   - 预期：1.5-2× 加速
   - 风险：所有 Q4_0 模型需要重新跑 Oracle 验证（输出 logits 可能漂移 ±1 ULP，最终 token 偶发分叉）
   - 工作量：半天

2. **重导出模型到 Q4_K / Q8_0**（推荐立即做）：
   - `q4_k/avx2.rs` 已经用 FMA 写好 + 通过 parity
   - `q8_0/avx2.rs` 已经用 FMA 写好 + 通过 parity
   - 用 llama.cpp 的 `convert.py` 把 E4B 转成 Q4_K_M 或 Q8_0
   - 文件大 ~30%，但推理快 2-3×
   - 工作量：1 小时（含下载 + 转换 + 验证）

3. **写 FMA + 8-wide row unroll 版本**：保留旧 kernel，新增 `matmul_q4_0_vs_q8_0_avx2_fma`，编译时 feature flag 切换。
   - 预期：2-3× 加速
   - 工作量：1 天

### 推荐

短期：选选项 2（重导出 E4B），立即得 2-3× 加速，零代码风险。
中期：选选项 1（加 FMA kernel），其他 Q4_0 模型（Qwen3-0.6B-Q4_0 等）也受益。

### 关联文件

- `src/ops/kernel/q4_0/avx2.rs` — kernel 实现 + parity 测试
- `src/ops/kernel/q4_0/scalar.rs` — scalar reference
- `src/ops/kernel/q4_0/mod.rs` — kernel 分发
- `src/models/gemma4/trunk/config.rs:215` — Q4_0 类型在允许列表
