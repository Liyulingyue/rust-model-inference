# Parallel Matmul Output Aliasing（Issue 4）

## 摘要

`ComputePool` 上的 `pool.compute_unchecked` 调用点统一使用同一形状的并行 matmul 闭包：

```rust
unsafe { pool.compute_unchecked(|ith, nth| {
    let output = unsafe { std::slice::from_raw_parts_mut(output_ptr, n_out) };
    kernel.forward_prepared(..., output, ...);
}) }
```

每个 worker 都从**同一个** `output_ptr` 派生出一个覆盖整段 output 的 `&mut [f32]`。N 个线程同时持有指向重叠字节的 `&mut` 引用——形式上违反 Rust 的别名规则（stacked borrows / tree borrows）。

之所以目前没有炸，是因为每个 kernel 都按 `(ith, nth)` 算 `[start, end)`、只写入 `output[start..end]`。**实际写入 disjoint，但形式上 UB**。

---

## 受影响范围

`Weight::quantize_and_matmul_with_scratch` 的违规形态是仓库里所有并行 matmul 的标准模板，被每个 forward 复制粘贴：

| 路径 | 文件 |
|---|---|
| 核心 helper | `src/ops/kernel/mod.rs`（`Weight::quantize_and_matmul_with_scratch`） |
| 直接并行量化 matmul | `src/ops/kernel/quantized_tensor.rs`、`src/ops/kernel/qtensor_owned.rs` |
| LLM trunk | `src/models/llama/trunk/forward.rs`、`src/models/lfm2/trunk/forward.rs`、`src/models/lfm25/trunk/forward.rs`、`src/models/lfm2moe/trunk/forward.rs` |
| Qwen3 | `src/models/qwen3/trunk/session.rs`、`src/models/qwen3/embedding.rs`、`src/models/qwen3/asr/mel_encoder.rs`、`src/models/qwen3/omni.rs`、`src/models/qwen3/tts/{codec/dac.rs,codec/predictor.rs,codec/rvq.rs,codec/tfm.rs,talker.rs}` |
| Gemma4 | `src/models/gemma4/asr/mod.rs`、`src/models/gemma4/trunk/forward.rs`、`src/models/gemma4/vision/mod.rs` |
| Qwen35 | `src/models/qwen35/trunk/forward.rs` |
| Diffusion | `src/models/diffusion/pig.rs`、`src/models/diffusion/z_image/mod.rs`、`src/models/diffusion/z_image/text.rs`、`src/models/diffusion/z_image/dit.rs`、`src/models/diffusion/z_image/vae.rs` |
| Spark（最近加入） | `src/models/spark/trunk/forward.rs`（commit `2a78e56`，跟其他模型完全同模板） |

---

## 契约 vs 现状

`src/core/thread_pool.rs` 中 `compute_unchecked` 的安全要求明文写着：

> *No shared `&mut` references; partition writes must be expressed through disjoint raw pointers or pre-split `&mut` slices.*

但 `quantize_and_matmul_with_scratch` 在每个 worker 里：

```rust
let output = unsafe { std::slice::from_raw_parts_mut(output_ptr, n_out) };
```

`output` 是整段 buffer 的 `&mut [f32]`——不是 disjoint 切片也不是 disjoint 裸指针。**违反上面这条契约**。

`unsafe { pool.compute_unchecked(...) }` 的包装只是标注「调用方已手动验证 unsafety」，并不修复别名问题。

---

## 为什么数值仍然正确

每个 dtype kernel 都按 `(ith, nth)` 用 `row_range(n_out, ith, nth)` 算 `[start, end)`，N 个线程之间的 row 区间严格 disjoint。kernel 内部 SIMD 实现只写 `output[start..end]`。

- BF16：`src/ops/kernel/bf16/mod.rs::forward_f32_rows` + `Self::row_range`。
- Q4_K / Q5_K / Q6_K / Q8_0 / Q4_0：各自 SIMD/Scalar 路径都按 `(ith * per_thread, min((ith+1)*per_thread, n_out))` 切。
- Q4_1、IQ4_NL、IQ4_XS：同样模式。

所以**形式上 UB、数值上正确**。

---

## 实际风险

| 风险 | 严重度 |
|---|---|
| Miri / `-Zmiri-stacked-borrows` 跑会直接挂 | 高（前提是项目跑 sanitizer） |
| Kernel bug 导致越界写一行 → 静默破坏另一个线程的输出 | 中（没有交叉校验，难调试） |
| 编译器按 `&mut` 独占做 memory op 重排/消除 | 低（目前没有真实 codegen bug） |
| 未来 stacked borrows 收紧或换 codegen 后崩 | 不可知 |

---

## 修复路径

### A. 推荐：Kernel API 改用裸指针 + 让 kernel 内部切片

把 `Kernel::forward_prepared` 的 `output: &mut [f32]` 改成 `output: *mut f32`，把 `n_out` 保留。每个 kernel 在自己的 SIMD/Scalar 实现里：

```rust
let (start, end) = Self::row_range(n_out, ith, nth);
if end > start {
    let my_out = unsafe { std::slice::from_raw_parts_mut(output_ptr.add(start), end - start) };
    // ... 写入 my_out ...
}
```

每个 worker 持有自己独占的 `&mut [f32]`，stack 上是 disjoint 的，符合 aliasing 规则。Caller 端闭包只剩：

```rust
unsafe { pool.compute_unchecked(|ith, nth| {
    kernel.forward_prepared(..., output_ptr, n_in, n_out, ith, nth);
}) }
```

工作量：Kernel trait 签名 + 每个 dtype kernel 内部微调 + 每个 caller 站点调整（去掉 `output_ptr` 的 `as_mut_ptr()` 包装）。**跨约 30 个文件**，适合单独立项做一次彻底 cleanup。

### B. 中改：caller 处 pre-split + 改 BF16 内部

不动 Kernel API，caller 闭包里按 `ith/nth` 切片后传入：

```rust
let (start, end) = Self::row_range(n_out, ith, nth);
let my_out = unsafe { std::slice::from_raw_parts_mut(output_ptr.add(start), end - start) };
kernel.forward_prepared(..., my_out, n_in, n_out, ith, nth);
```

BF16 的 `forward_f32_rows` 内部又会做 `output[start..end]`——pre-split 之后这步会越界（`my_out[start..end]` 是空）。需要同步把 BF16 改成「信任 caller 已切片，只用 `output[0..len]` 寻址」。

其他 kernel 多数依赖 `n_out`+`ith`/`nth` 自己分区，把传给它们的 `output` 当成 per-thread 切片需要逐一 audit。

工作量比 A 小（不动 trait 签名），但仍然要逐个 kernel 看一遍。

### C. 不修：显式接受 UB

保留当前形态 + `unsafe { compute_unchecked }` 包装 + 一行注释说明「写入是 disjoint 的，kernel 永远只动自己的 `[start, end)`」。`a5b7711 修复compute pool bug` 实际上就是这条路线。

代价：

- 项目无法跑 Miri
- 任何新加的 kernel 都必须遵守 partition 不变量，否则静默破坏
- 文档/代码 review 时新人容易忽略

---

## 当前策略

接受 C：在 `Weight::quantize_and_matmul_with_scratch` 顶部加注释把不变量写明，依赖现有 dtype kernel 已经过 audit 的事实。后续一旦新增 dtype kernel 或修改 partition 公式，必须在 PR description 里复述不变量。

未来安排：单独立 PR 做 A，作为本仓库 LLM 路径的一次系统性 hardening。

---

## 历史

- `a5b7711 修复compute pool bug`（2026-09-03，作者 `liyulingyue`）：把全仓库 `pool.compute(...)` 站点改为 `unsafe { pool.compute_unchecked(...) }`。该 commit 的目的是把 pool 的安全契约形式化，但**没有**修复 helper 内部的别名违规。
- `2a78e56 spark: thread matmuls through ComputePool`（2026-09-03）：Spark 接入 pool，沿用同一模板，没有引入新的违规面。