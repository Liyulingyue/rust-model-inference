---
name: operator-internal-dispatch
description: Use when 在 rust-model-inference 中设计/重构算子（SIMD dispatch、kernel selection、严格 vs 近似精度切换、是否引入 `_ggml` 后缀的对齐函数）时。强调函数级 cfg + 运行时 feature 检测，调用方不应嗅探 target_arch / ggml_type / weight_bytes()；`_ggml` 函数仅在无法绕开时才使用。
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
- **重新发明标准库操作**：手写标量 polynomial exp、分层 multi-lane dot product、4-level sum-of-squares 等 PyTorch ATEN 翻译。这些闭包/手写版本无法被编译器向量化，且代码量是标准库入口的 10-50 倍。见下方专节。

## 不要重新发明标准库操作

禁止手写标量 polynomial/multi-lane 实现来替代已有的标准库方法或仓库算子。
PyTorch ATEN kernel 的逐行翻译几乎总是错误的——它们是为特定 CPU 微架构
写的 C++ 代码，在 Rust 中不会获得同样的编译器优化。

| 禁止手写 | 替代方案 |
|---------|---------|
| polynomial exp 近似（magic constants） | `f32::exp()` 或 `value.exp()` |
| 手写分层 sum-of-squares（4-lane x 4-ILP x 4-level） | `iter().map(|v| v * v).sum()` |
| 手写 4-lane 标量 dot product | `crate::ops::dot_f32`（有 NEON/AVX2） |
| 手写 bf16xf32 dot via 闭包 | `crate::ops::dot_bf16_f32`（有 NEON/AVX2） |
| 闭包 `Fn(usize) -> f32` 做逐元素访问 | slice + 标量循环（编译器可自动向量化） |

判断标准：如果你写了一个超过 10 行的标量函数，检查 `crate::ops::` 和
Rust 标准库是否已有等价入口。PyTorch 的内部 kernel 不是"更快"的参考
——Rust 编译器对标量循环的自动向量化通常优于手写多累加器。

## `_ggml` 后缀函数的使用边界

仓库里有一批 `_ggml` 后缀的函数（`dot_f32_ggml`、`silu_ggml`、`softmax_ggml_inplace`、
`forward_prepared_ggml`、`ops/kernel/{f32,q8_0}/ggml.rs` 等）。它们复刻上游
llama.cpp / ggml 的**具体累加结构**（几个累加器、什么顺序 reduce、用哪个 `exp`），
目的是让结果与 ggml **逐位一致**。

**默认不新开这种函数。** 仓库的原则是「实现并采用 `dot_f32` 这样的函数，尽量不采用
`dot_f32_ggml` 这样的函数」。理由：

- 浮点加法**不满足结合律**，所以「换一种累加顺序」只是换了一种舍入，不是更正确；
- ggml 的累加结构是**其实现细节**，不是精度或速度上的最优解 —— 从两个方向都很难证明
  它最优，因此没有理由让仓库默认遵循它；
- 每多一条 `_ggml` 路径，就多一份要维护、要测试、要和上游同步的算术。

**只有以下四种情况可以引入 `_ggml` 函数**：

| 允许 | 例子 |
| --- | --- |
| 无法绕开的精度问题 | 上游用某结构算出的值与仓库结构差异过大，导致量化档位翻转、输出崩掉 |
| 行为对齐问题 | 参考实现的行为（不是数值）依赖该结构，例如 softmax 掩码列必须恰好贡献 0 |
| 行为对齐测试 | 需要逐位复现参考实现才能定位第一个分叉点 |
| 精确数值测试 | 断言两个实现在特定位模式上完全一致 |

引入时必须同时满足：

1. **注释写明为什么绕不开** —— 具体到「不用它会怎样」，而不是「为了对齐 ggml」；
2. **不改变默认路径的契约** —— 既有调用方继续走 `dot_f32` 等默认入口，`_ggml` 是
   opt-in 的并行入口（trait method 或独立函数）；
3. **命名与文档可发现** —— 新函数名带 `_ggml`，并在 `docs/develop/` 登记它为什么存在。

### 现状与收口方向

当前仓库里已有一批 `_ggml` 函数（`src/ops/dot.rs::dot_f32_ggml`、
`src/ops/kernel/f32/ggml.rs`、`src/ops/kernel/q8_0/ggml.rs`、`silu_ggml`、
`softmax_ggml_inplace`、`forward_prepared_ggml` 等）。它们是**历史既有**，
其中一部分是 nemotron_h_moe 逐位对拍时引入的。

**未来改动的收口方向**：

- 新增代码**不得**再引入 `_ggml` 函数，除非命中上表四种情况之一；
- 对既有 `_ggml` 函数，若其存在的理由已消失（例如对拍已完成、模型已验证），
  应评估把调用点迁回默认入口，让 `_ggml` 路径自然消亡；
- 无法迁移的（例如上游行为确实依赖该结构），在 `docs/develop/` 记下「为什么绕不开」，
  避免下一个人重新纠结。

### 同名不同义：`dot_f32_ggml` 的两处定义

仓库里有两个 `dot_f32_ggml`，**同名但签名与语义都不同**，读代码时注意区分：

| 位置 | 签名 | 语义 |
| --- | --- | --- |
| `src/ops/dot.rs` | `dot_f32_ggml(a: &[f32], b: &[f32], n: usize) -> f32` | 通用算子层的 ggml 归约顺序；aarch64 上转发到 `dot_f32_neon` |
| `src/ops/kernel/f32/ggml.rs` | `dot_f32_ggml(x: &[f32], y: &[f32]) -> f32` | 带 slice 长度；另有 `_stride` / `_stride_padded` 变体，供 nemotron trunk 用 |

**aarch64 上两者等价**（都走 `dot_f32_neon` 的 4 累加器结构）；**x86 上不等价**
（前者 4 个 256-bit 累加器，后者单累加器）。

## 检查清单（提交前）

- [ ] 调用方 grep `target_arch`、`ggml_type`、`weight_bytes` 都为空
- [ ] helper 自包含（zero-fill + accumulate），caller 不预填
- [ ] SIMD/scalar fallback 都至少有一个测试
- [ ] SIMD reduction 与 scalar 顺序不同时，测试用 ≤2 ULP 或 rel 容差而非 bit-exact
- [ ] 新 kernel 想加 opt-in 行为只动 impl 块，不动 caller