# Parallel Matmul Output Aliasing（Issue 4）

## 摘要

`ComputePool::compute` 上的并行 matmul 闭包有两类形态，其中一类**形式上违反** Rust 的别名规则（stacked borrows / tree borrows），但运行上正确：

```rust
// 形态 A：caller 端 pre-split（合法）
pool.compute(move |ith, nth| {
    let (start, end) = row_range(n_out, ith, nth);
    let my_out = unsafe { std::slice::from_raw_parts_mut(output_ptr.add(start), end - start) };
    kernel.forward_prepared(..., my_out, ...);   // 每个 worker 持有 disjoint &mut [f32]
});
```

```rust
// 形态 B：caller 端裸指针 + 全长（形式 UB / 实际正确）
let output_ptr = output.as_mut_ptr();
pool.compute(|ith, nth| {
    let output = unsafe { std::slice::from_raw_parts_mut(output_ptr, n_out) };   // ← 整段 &mut
    self.kernel.forward_prepared(..., output, n_in, n_out, ith, nth);            // kernel 内部按 (ith, nth) 只写 [start, end)
});
```

形态 B 实际写入 disjoint，kernel 内部按 `(ith, nth)` 用 `row_range` 算 `[start, end)`、只写 `output[start..end]`。**形式上 N 个 worker 同时持有指向重叠字节的 `&mut`，但实际写入 disjoint**——Miri / `-Zmiri-stacked-borrows` 会直接挂。

> **更正：** 初版文档提到 `pool.compute_unchecked(...)` 与 commit `a5b7711`。`compute_unchecked` 在 `src/core/thread_pool.rs` 中**不存在**（只有 `compute` / `compute_with_chunks` / `next_chunk`），`a5b7711` commit 也不在 `git log` 中。本文档以下不再引用这两个名字。

---

## 1. 受影响范围（精确定位到 5 文件 8 处）

仓库内 `unsafe { std::slice::from_raw_parts_mut(output_ptr, n_out_or_rows) }` 的实际站点：

| 文件 | 行 | 函数 | 调用场景 |
|---|---|---|---|
| `src/ops/kernel/mod.rs` | 384 | `Weight::quantize_and_matmul_with_scratch` | **LLM 主路径**（Qwen3 / Gemma4 / Llama / LFM2 / LFM2.5 / LFM2-MoE / Spark 等的 trunk 调用） |
| `src/ops/kernel/quantized_tensor.rs` | 640 | `QuantizedTensor::quantize_and_matmul_with_scratch`（K-quant 分支） | K-quant 模型的直接调用 |
| `src/ops/kernel/quantized_tensor.rs` | 663 | `QuantizedTensor::quantize_and_matmul_with_scratch`（Q8_0 分支） | Q8_0 模型的直接调用 |
| `src/ops/kernel/qtensor_owned.rs` | 768 | `QTensorOwned::quantize_and_matmul`（K-quant dispatch） | K-quant 经 `QTensorOwned` 的路径 |
| `src/models/gemma4/trunk/forward.rs` | 767 | `gemma4_bf16_input_matmul` 闭包 | Gemma4 BF16 路径 |
| `src/models/gemma4/trunk/forward.rs` | 996 | Gemma4 attention 路径 | 同上 |
| `src/models/diffusion/z_image/mod.rs` | 648 | Z-Image matmul | Diffusion 路径 |
| `src/models/diffusion/z_image/mod.rs` | 728 | Z-Image matmul（同函数另一调用） | 同上 |

**已采用形态 A（safe）的参考实现**（建议作为模板）：

| 文件 | 行 | 函数 |
|---|---|---|
| `src/ops/kernel/qtensor_owned.rs` | 660 | `matmul_into_buf_pooled`（K-quant 行分区） |
| `src/ops/kernel/qtensor_owned.rs` | 663 | 同上 |
| `src/ops/kernel/qtensor_owned.rs` | 696 | `matmul_into_buf_pooled`（F32/F16/BF16 行分区） |
| `src/ops/kernel/qtensor_owned.rs` | 800 | `quantize_and_matmul` 的 F32/F16/BF16 分支 |
| `src/ops/kernel/q4_0/scalar.rs` | 122 | **列分区**模式（见 §3） |

**其余 ~85 个 `pool.compute(...)` 站点**（`models/{llama,lfm2,lfm25,lfm2moe,gemma4,spark,qwen3,qwen35}/trunk/forward.rs`、各 `models/diffusion/*.rs`、`models/qwen3/{tts,omni,asr}/*.rs`、`models/vibevoice_asr/llm.rs`、`models/qwen_drive/weights.rs` 等）的闭包体是 `kernel.forward_prepared(..., output, ..., ith, nth)`——**没有 `from_raw_parts_mut` 包装**，闭包对每个 worker 是独立调用、独立 borrow 同一 `output` slice。它们不构成 §1 表里的形态 B。

---

## 2. 不存在显式安全契约

初版文档提到 "`src/core/thread_pool.rs` 中 `compute_unchecked` 的安全要求明文写着"——**该契约在 `compute` 上不存在**：

```rust
// src/core/thread_pool.rs:183
pub fn compute<F: Fn(usize, usize)>(&self, f: F) { /* ... */ }
```

* `compute` 签名是 `pub fn compute<F>(&self, f: F)`，**无 `unsafe` 标记、无 `///` docstring、无 `// SAFETY:` 注释**。
* 调用站点 `unsafe { std::slice::from_raw_parts_mut(...) }` 是调用方责任，不是 `compute` 自身的契约。
* §1 表中所有形态 B 站点**没有**任何 `// SAFETY:` 注释说明「kernel 只动 disjoint [start, end)」。
* §3 表中形态 A 站点也只 `q4_0/scalar.rs:111` 写了 `// SAFETY: this worker exclusively owns columns start..end.`，其余形态 A 站点也没写。

也就是说**当前没有任何机制把不变量在源码层固化**——依赖「每个 dtype kernel 已经过 audit 的事实」是隐式的。

---

## 3. 两种分区模式

文档初版只描述了行分区（每个 worker 写 `[row_start..row_end]`）。实际存在两种：

* **行分区（row partition）**：`q8_0/dispatch.rs`、`bf16/mod.rs`、`f32/mod.rs`、`f16/mod.rs`、`q4_0/mod.rs`、`q4_1/mod.rs`、`Kernel` trait 的 `forward_prequantized` / `forward_prepared` 默认形态。每个 worker 写 `output[start..end]`，`start = ith * per_thread`，`end = (start + per_thread).min(n_out)`。
* **列分区（column partition）**：`q4_0/scalar.rs:118-133`。每个 worker 写 `output[*][start..end]`，列区间在 worker 间 disjoint，行区间共享。caller 端 pre-split 成 `output.add(row * n_out + start)` 的 disjoint `&mut [f32]`。这条路径**形态 A 已经走通**（行 122）。

§1 表中形态 B 的 8 处全部走行分区。修复时不需要为列分区特殊处理。

---

## 4. 为什么数值仍然正确

每个 dtype kernel（无论行/列分区）的 SIMD/Scalar 实现都按 `(ith, nth)` 算 `[start, end)`、只写 `output[start..end]`。具体审计过的实现：

* `bf16/mod.rs::forward_f32_rows` + `Self::row_range`（`Self::row_range` 与 f32/d32 是同一套）
* `q4_0/{avx2,scalar}.rs`、`q4_1/mod.rs`、`q8_0/dispatch.rs::matmul_q8_0_quantized_range`
* `q2_k / q3_k / q4_k / q5_k / q6_k / iq4_nl / iq4_xs` 的 `forward_prepared`（间接调 `vec_dot_*_q8k`）

§1 表里形态 B 的 `unsafe { std::slice::from_raw_parts_mut(output_ptr, n_out_or_rows) }` 之后**直接传 `output` 给 `kernel.forward_prepared(..., ith, nth, ...)`**——kernel 在自己的 `row_range(n_out, ith, nth)` 后只动 `[start, end)`。

`PreparedRows::matmul_group`（`ops/kernel/mod.rs:155-256`）包含两条子路径，**两条都需要审计**：

* `batched_q4` 分支（行 197-216）：闭包内**多 weight 共享 `output_ptr`**（每个 weight 写入不同列区间，靠 `weight.n_out` 步进）。把 `*mut f32` 直接传给 `q4_0::scalar::matmul_q4_0_batched_scalar_range`（行 203-213），由该函数内部按列做 disjoint 写入。形式上仍共享 `*mut f32`，但函数体不构造跨 worker 的 `&mut [f32]`，**不构成 §1 表那种 alias UB**（`// SAFETY:` 在行 199-201 已写）。
* 非 batched 分支（行 218-251）：**形态 B 的变体**。每个 worker 顺序遍历 `self.rows`，每行用 `unsafe { std::slice::from_raw_parts_mut(*output_ptr, weight.n_out) }`（行 236）构造 per-row `&mut [f32]`，所有 worker 通过 closure capture 共享同一 `*output_ptr`。kernel 在该 slice 内按 `(ith, nth)` 行分区写入 disjoint `[start, end)`。**写入 disjoint、借用 alias**——与 §1 表同型。这条路径**没有 `// SAFETY:` 注释**（对比 `batched_q4` 分支行 199 有），是 §9 backlog 的一项。

---

## 5. 实际风险

| 风险 | 严重度 |
|---|---|
| Miri / `-Zmiri-stacked-borrows` 跑 §1 表中形态 B 站点会直接挂 | 高（前提是项目跑 sanitizer；目前 CI 没跑） |
| Kernel bug 导致越界写一行 → 静默破坏另一个 worker 的输出 | 中（没有交叉校验，难调试） |
| 编译器按 `&mut` 独占做 memory op 重排/消除 | 低（目前没有真实 codegen bug） |
| 未来 stacked borrows 收紧或换 codegen 后崩 | 不可知 |

---

## 6. 修复路径

### A. 推荐：caller 端 pre-split（形态 A）

不动 `Kernel` trait 签名。caller 闭包里按 `ith/nth` 切片后传入：

```rust
let (start, end) = row_range(n_out, ith, nth);
let my_out = unsafe { std::slice::from_raw_parts_mut(output_ptr.add(start), end - start) };
kernel.forward_prepared(..., my_out, ...);
```

每个 worker 持有自己独占的 `&mut [f32]`，stack 上 disjoint，符合 aliasing 规则。

**已有参考实现**：`qtensor_owned.rs:660, 663, 696, 800` 与 `q4_0/scalar.rs:122`——直接复制其写法即可。

工作量：§1 表中 5 文件 8 处 + `Weight::quantize_and_matmul_with_scratch` 的「文档注释 + 实际改动」打包成一个 PR。**预计 < 100 行 diff**。

### B. 改 Kernel API：接受 `*mut f32`

把 `Kernel::forward_prepared` 的 `output: &mut [f32]` 改成 `output: *mut f32`，把 `n_out` 保留。每个 kernel 内部自己构造 disjoint `&mut [f32]`。

工作量：Kernel trait 签名 + 13 个 dtype kernel + 每个 caller 站点调整。**跨 ~20 文件**，比 A 大一个数量级。

A 与 B 在借用规则上等价（都是 caller 或 kernel 内构造 disjoint `&mut [f32]`）。A 是「已经修了什么」，B 是「把 unsafe 推到 kernel 内层」。**没有 A 没解决的别名问题**。

### C. 不修：显式接受 UB + 加不变量注释

保留形态 B + 在 `Weight::quantize_and_matmul_with_scratch` 等函数顶部加 `// SAFETY:` 注释固化不变量 + 在 `compute` 上加 docstring 写明 caller 责任。

代价：
- 项目无法跑 Miri（CI 已没跑，加了也不会挂）
- 任何新加的 kernel 都必须遵守 partition 不变量，否则静默破坏
- 文档/代码 review 时新人容易忽略

---

## 7. 当前策略

**C 路线**：未加显式注释，仅依赖现有 dtype kernel 已经过 audit 的事实。`Weight::quantize_and_matmul_with_scratch` 顶部**没有 `// SAFETY:` 注释**——这是一个文档与代码不一致的点（见 §9）。

未来安排：单独立 PR 做 A，作为本仓库 LLM 路径的一次系统性 hardening。参考实现已经在 `qtensor_owned.rs`，工作量小。

---

## 8. 历史

* `2a78e56 spark: thread matmuls through ComputePool via quantize_and_matmul_with_scratch`（2026-09-03，作者 `liyulingyue`）：Spark 接入 pool，沿用 `Weight::quantize_and_matmul_with_scratch` 同一模板，没有引入新的违规面。
* `df27887 修复Spark模型上的问题`（2026-09-03）：同一时段 Spark 相关修复。
* 初版文档引用的 `a5b7711 修复compute pool bug`：**该 commit 不在 `git log` 中**——可能是被 squash 抹了 hash，或方案从未落库。无论哪种，`compute_unchecked` 在源码中**不存在**（已 grep 全仓验证），初版文档的论述前提不成立。

---

## 9. 代码开发缺失项（文档 → 代码的反向追踪）

按本文件 §6 路径 A 落地，下列代码层条目尚未补齐，列作 backlog：

### 9.1 已完成（2026-09-18）

下列 5 条全部以纯文档/属性方式落进 src/，无运行行为变化：

* ✅ `ComputePool::compute` docstring（`core/thread_pool.rs:182`）：写明 closure 必须在 worker 间 disjoint partition；引用本文件 §1。
* ✅ `Weight::quantize_and_matmul_with_scratch` docstring（`ops/kernel/mod.rs:336`）：写明 alias 模式 + 不变量依赖 kernel 行分区；引用本文件。
* ✅ `QuantizedTensor::quantize_and_matmul_with_scratch` docstring（`ops/kernel/quantized_tensor.rs:611`）：同上。
* ✅ `PreparedRows::matmul_group` 非 batched 分支 `// SAFETY:`（`ops/kernel/mod.rs:235`）：解释 `*output_ptr` 跨 worker 共享 + 每 worker 行内序列化 + kernel 按 `(ith, nth)` 写 disjoint `[start, end)`。
* ✅ `matmul_q8_0_quantized_dynamic` `#[deprecated]`（`ops/kernel/q8_0/parallel.rs:131`）：note 指向 `# Bug` 注释 + 推荐替代。
* 配套：`src/ops/matmul.rs` re-export 加 `#[allow(deprecated)]`，避免编译期 warning 噪声。

验证：`cargo build --lib`（default + `--features vulkan` + `--features parity-trace`）✓；`cargo test --lib` 失败数与改动前一致（33 个全是环境/精度类预存在失败，与注释无关）。

### 9.2 仍待处理

| 缺口 | 描述 | 影响 | 推荐落地方式 |
|---|---|---|---|
| **`gemma4_bf16_input_matmul` 闭包外的 `unsafe {}` 是误导** | `models/gemma4/trunk/forward.rs:761` 的 `pool.compute(\|thread, threads\| unsafe { ... })` 把整个闭包标 `unsafe`，但实际只在内部做 `from_raw_parts_*`，**闭包自身的 `pool.compute` 不是 unsafe 函数**。这个标记让人误以为闭包本身有特殊不安全责任。 | 阅读者误解 `compute` 的契约。 | 把 `unsafe { ... }` 缩到 `from_raw_parts_*` 那一行——这正好与 §1 表 gemma4:767 的 pre-split 修复同一 PR。 |
| **§1 表中 7 处 alias（其中 5 处涉及 GPU 协调）** | `kernel/mod.rs:384`、`quantized_tensor.rs:640`、`:663`、`qtensor_owned.rs:768`、`z_image/mod.rs:648`、`:728` 仍是 `from_raw_parts_mut(output_ptr, n_out)` 形态。 | Miri 跑会挂；任何新 kernel 改动 partition 公式都可能引入静默破坏。 | §6 路径 A 直接落地 2 处无 GPU 路径的（`quantized_tensor.rs:640` K-quant 分支、`gemma4:767` BF16 分支）；5 处涉及 `matmul_q8_0_quantized_parallel_rows` GPU 分支的需 GPU/CPU 分支并存或先做路径 B。 |
| **`Kernel::forward_prepared` 签名阻碍路径 B 推广** | `&mut [f32]` 参数是形态 B 的源头。要根除需要走 §6 路径 B，或在 caller 端 pre-split（路径 A）。 | 路径 B 工作量 ~20 文件；路径 A 工作量 ~5 文件。 | 短期走 A；长期等 Kernel trait 大重构时并入 B。 |
| **CI 未跑 Miri / stacked-borrows** | `.github/workflows/ci.yml` 不含 `cargo +nightly miri test`。 | §1 表 B 站点任何回归都不会被 sanitizer 拦截。 | 加 CI job。 |
| **新 dtype kernel 加进来没有 partition audit** | 没有 CI / lint 检查新 `forward_prepared` 是否遵守 `row_range` 不变量。 | §6 路径 C 提到的「新加 kernel 必须遵守 partition 不变量」无强制。 | PR template 加 checklist；长期 maturin-style 半自动 lint。 |