---
name: operator-internal-dispatch
description: Use when 在 rust-model-inference 中设计/重构算子（SIMD dispatch、kernel selection、严格 vs 近似精度切换）时。强调函数级 cfg + 运行时 feature 检测，调用方不应嗅探 target_arch / ggml_type / weight_bytes()。
---

# Operator-internal dispatch

## 核心原则

每个算子自己决定走 SIMD 还是 scalar、自己做 zero-fill、自己做精度协商。调用方只看到**一个统一入口**，传入 f32 输入拿到 f32 输出，不关心 target_arch、量化类型、SIMD feature、原始字节布局。

**永远不要为了对齐而把 type-sniffing / weight sniffing 推到调用方**。如果某条 SIMD 路径只为 parity oracle 而存在，应该在 trait method 或专用函数里 opt-in，而不是让 caller 检查 `ggml_type` 选不同入口。

## 反例（重构前）

```rust
fn linear_fwd(lin: &Linear, input: &[f32], t: usize, pool: &ComputePool) -> Vec<f32> {
    pool.compute(move |ith, nth| {
        for row in start..end {
            // ❌ caller 检查 ggml_type
            if weight.ggml_type == GGMLType::F16 {
                // ❌ caller 手动转 f32→f16
                let input_f16 = input_row.iter().map(|&v| f32_to_f16(v)).collect();
                // ❌ caller 嗅探 weight_bytes() 拿原始字节
                for (i, out) in output_row.iter_mut().enumerate() {
                    let row = &bytes[i * in_dim * 2..(i + 1) * in_dim * 2];
                    *out = unsafe { funasr_f16_dot_f16(row, &input_f16) };
                }
            } else {
                weight.kernel.forward(...);
            }
        }
    });
}
```

问题：
- 调用方要知道 `ggml_type` 才能选 strict F16×F16 路径
- 调用方要知道 `weight_bytes()` 才能拿原始 f16 字节
- 任意 kernel 想加严格语义都得复制这份样板

## 正解（重构后）

**Kernel trait 加 opt-in 方法，默认返回 false**：

```rust
pub trait Kernel: Send + Sync {
    /// Strict F16×F16 single-row matmul with f64 lane accumulator.
    /// F16 kernel opts in (returns true); others return false.
    fn forward_f16_strict(
        &self, _input: &[f32], _output: &mut [f32], _n_in: usize, _n_out: usize,
    ) -> bool { false }
    // ... 现有 forward / forward_prepared / ...
}
```

**F16 kernel 实现**：

```rust
impl Kernel for F16Kernel<'_> {
    fn forward_f16_strict(&self, input, output, n_in, _n_out) -> bool {
        let mut input_f16: Vec<u16> = input.iter().map(|&v| f32_to_f16(v)).collect();
        for (i, out_i) in output.iter_mut().enumerate() {
            let row = &self.weight[i * n_in * 2..(i + 1) * n_in * 2];
            *out_i = unsafe { dot_f16_f16_strict(row, &mut input_f16) };
        }
        true
    }
}
```

**调用方只有一行**：

```rust
if weight.kernel.forward_f16_strict(input_row, output_row, in_dim, out_dim) {
    // F16 kernel 自己处理；其它 type 默认返回 false 走下面
} else {
    weight.kernel.forward(input_row, output_row, in_dim, out_dim);
}
```

## 硬约束

| 规则 | 为什么 |
| --- | --- |
| 算子内部 `#[cfg(target_arch = "x86_64")]` + `is_x86_feature_detected!("avx2")` 选 SIMD/scalar fallback | cfg 限定代码生成，runtime 检测同型号 CPU 是否启用 |
| 调用方不写 `#[cfg(target_arch = "aarch64")]` 分支 | 调用方变成 target-aware，污染多个调用点 |
| 调用方不嗅探 `ggml_type`、`weight_bytes()`、内部字段 | 改动一个 kernel 不应让所有 caller 都重看一遍 |
| 严格/近似语义放到 trait method 而不是分支 if-else | 任意 kernel 可独立 opt-in，调用方只问"能不能" |
| helper 自己 zero-fill 输出 | 否则 caller 会预 zero + 重复累加 → double-counting bug |
| helper 自己负责量化（f32→f16 等） | caller 不应看到 `f16` / `u16` 等内部类型 |

## 函数级 dispatch 的标准样板

```rust
#[inline]
pub fn my_op(input: &[f32], output: &mut [f32], n: usize) {
    debug_assert_eq!(input.len(), n);
    output.fill(0.0);                  // 自包含
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe { my_op_avx2(input, output, n); return; }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon() {
            unsafe { my_op_neon(input, output, n); return; }
        }
    }
    for i in 0..n {                    // scalar fallback
        output[i] += scalar_compute(input[i]);
    }
}
```

`has_avx2_fma` / `has_neon` / `has_f16c` 等 `#[inline(always)]` runtime 检测器统一放在 `src/ops/` 下。

## 常见错误

- **double-counting**：helper 已经 `output.fill(0.0)` + 累加，caller 又写 `for j in 0..t { out += s * v[j] }`。
- **target_arch 嗅探**：调用方写 `#[cfg(target_arch = "aarch64")] use helper_a; #[cfg(not(target_arch = "aarch64"))] use helper_b;`，每次加架构都要改。
- **bit-level 兼容性伪装对齐**：用 `abs_diff <= 2` 当 parity oracle 的通过条件，但实际 oracle 期望 bit-exact——这是质量倒退，不是 dispatch 改进。
- **kernel 不知道自己的精度**：把 F16×F16 严格语义放在 caller 的 `if ggml_type == F16` 而不是 kernel 自己 opt-in。

## 检查清单（提交前）

- [ ] 调用方 grep `target_arch`、`ggml_type`、`weight_bytes` 都为空
- [ ] helper 自包含（zero-fill + accumulate），caller 不预填
- [ ] SIMD/scalar fallback 都至少有一个测试
- [ ] SIMD reduction 与 scalar 顺序不同时，测试用 ≤2 ULP 或 rel 容差而非 bit-exact
- [ ] 新 kernel 想加 opt-in 行为只动 impl 块，不动 caller