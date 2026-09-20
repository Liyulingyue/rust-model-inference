# TODO — RustModelInference Roadmap

This document merges the legacy `docs/TODO.md` (deep-dive format with
TODO-001…TODO-010) and the roadmap-style `docs/develop/TODO.md`
(checklist of upcoming work). The bottom half carries the detailed
investigation notes; the top half carries the at-a-glance priority list.

CPU SIMD capability assumed throughout: see [SIMD extension inventory](#simd-extension-inventory)
at the bottom of this file (AVX2 + FMA + F16C + AVX-VNNI on Intel Core Ultra 5 125H;
NEON on aarch64; no AVX-512).

---

## High Priority

- [ ] **Q2_K / Q3_K SIMD 加速** — 当前 scalar 5-9 t/s。仿 `vec_dot_q4k_q8k_avx2` 写 `_avx2` AVX2 kernel。
      预期 5-10× 加速，目标 30-50 t/s。详见 `docs/OPTIMIZATION.md` § "Quant Kernel 补全"。
- [ ] **IQ2_XS / IQ3_S / IQ2_S scalar forward_prequantized stub 修复** — 现状：`src/ops/kernel/iq4_xs.rs`
      的 `iq_kernel_impl!` macro 给 IQ2_XXS / IQ2_S / IQ3_XXS / IQ3_S / IQ1_M / IQ1_S 生成了
      Kernel trait impl，`forward_prepared` 正确调 `vec_dot_*_q8k` scalar（生产走 Q8K 路径无 panic）；
      但 `forward_prequantized` 是 stub（写 0）。`uses_q8_k()` 包含所有 IQ 类型 → 生产永远走
      `forward_prepared` → stub 永不触发，但接口完整性差。
      选项 1：把 macro 里的 stub 改成完整 dequant-to-f32 + dot 路径（与 IQ4_NL / Q4_K 同款），低成本。
      选项 2：直接写 SIMD kernel（需要 IQ2_XS / IQ3_S 的 GGUF 测试模型 + llama.cpp Oracle；当前
      `models/qwen3-0.6b-gguf/` 没有这些格式，按 `adapting-new-models` skill 暂无法做精度验证）。
- [ ] **AVX-VNNI int8 dot 加速** — 见 [TODO-AVX-VNNI](#todo-avx-vnni-int8-dot-加速) 详细说明。
      当前 Q8_0 × Q8_0 matmul 已用 `_mm256_maddubs_epi16` (AVX2)，但 AVX-VNNI 的
      `_mm256_dpbssd_epi32`（带 saturate 的三操作数 int8 dot）在
      K2-Horizon / Breeze 等 Q8_0 路径上可省一次 saturate pass。
- [ ] **Q4_0 kernel 加 FMA + tiling** — 见 [TODO-001](#todo-001-q4_0-avx2-kernel-不使用-fma性能受限)。
      预期 1.5-2× 加速。

### JEV 决策评分 follow-ups

`--jev` 已在 `src/app/text.rs::run_jev_decision` 中按 `general.architecture` 自动路由
到 9 个 trunk 的 `forward_logits` / `run_forward_logits_*`（Qwen3 / Qwen3.5 /
Llama / Gemma4 / LFM2 / LFM2.5 / Spark / Nemotron-H / Hunyuan）。通用协议与
per-arch chat template 见 [`docs/usage/qwen3.md` §10](../usage/qwen3.md) 与
[`docs/develop/jev.md`](jev.md)。

- [ ] **JEV Multi-question KV 共享** — 当前多 question 模式（`--jev-question × N`）
      每个 question 都会新建 session 重新 prefill 一遍 context + question 文本，
      耗时 N × prefill_time。KV 共享方案：让 session 支持 `reset_kv(seq_len)`
      把 KV cache 截到 context 之后的长度，对每个 question 只 prefill
      `{"question": ..., "candidates": ...}` 部分复用同一份 context KV。
      预期 N 个 question 总耗时从 `N × t` 降到 `t + (N-1) × t'`，其中 `t'`
      是 question-only 部分。
      适用 trunk：qwen3 / qwen35 / gemma4 / spark（有 Session API）；
      llama / lfm2 / lfm25 / nemotron_h / hunyuan 需要先把 monolith
      拆出 session API。
- [ ] **JEV Shared prefix batching** — 多个 `--jev-question` 不仅共享 prefix，
      还可以把 K 个不同 question 的最后 token batch 成一次前向（拼接
      成 `[prefix; q1_suffix; q2_suffix; ...; qK_suffix]`，attention 阶段
      mask 让 K 个问题互不干扰）。当 K 个 question 文本相似度高时（典型
      客服路由、FAQ 分类场景），可以把 prefill 提速近 K×。
      需要先解决：不同 question 的 last-token 位置索引；padding 到统一
      长度；attention mask 的构造。当前架构（每 question 一个独立 session）
      完全不支持此模式，需要结构化改造。
- [ ] **JEV instruct-tuned GGUF 文档** — 现状：本地 `models/qwen3-0.6b-gguf/`
      只有 base model Qwen3-0.6B（IQ4_NL / Q4_0 / Q8_0 等量化），不是
      Instruct 变体。OpenJEV 锁定 Qwen3-4B-Instruct-2507。要让 JEV 输出
      真正准确的 label，需要：
      1. 找/下载 `Qwen3-0.6B-Instruct` 的 GGUF（huggingface 上有）
      2. 同 §1（用 base model + IQ4_NL 跑通作为 baseline）
      3. 加 `tests/qwen3_instruct_jev_reference.rs` 比对 base vs instruct
- [ ] **JEV 与生成模式共享 KV（暂留）** — 当 `--prompt` 后面接 `--jev` 时，
      可以复用生成模式 prefill 出来的 KV cache，避免重复编码同一段 context。
      当前架构下两者用不同的 session path，需要在 dispatch 入口统一
      KV lifecycle（`KvLifecycle::Shared`）。

## Medium Priority

- [ ] **讨论：MemoryArena 与 BlockAllocator 组合**
- [ ] **讨论：GPU 后端架构设计** — Vulkan / wgpu / CUDA 等多后端抽象
- [ ] **讨论：SIMD 扩展路线** — 当前 AVX2+FMA、NEON。后续可考虑 AVX-512 (高端 CPU)、ARM SVE、AVX-VNNI (int8 dot)
- [ ] **讨论：两套线程调度统一** — ComputePool vs rayon。暂不统一（LLM 热路径不应轻易改动）
- [ ] **Q8_0 与 Q8_K 量化路径按需量化（消除冗余计算，保留两份 buffer）** — dispatch 按 layer 权重格式，省一次量化 pass
- [ ] **Qwen3.5：借用权重与 FFN gate/up 输入量化复用的取舍** — 中期重构，不阻塞局部 FFN 优化

### K-quant multi-row tile（vec_dot_q4k_q8k_avx2 / vec_dot_q6k_q8k_avx2）

目标：Q4_K_M 从 ~76 t/s → 120-150 t/s。详见 TODO-001 关联。

## Low Priority

- [ ] 更多量化格式支持（Q4_K, Q5_K 等）
- [ ] 完善 GGUfRS 导出功能
- [ ] GGUF 导出支持

---

## Detailed TODOs (merged from legacy docs/TODO.md)

### TODO-001: Q4_0 AVX2 kernel 不使用 FMA，性能受限

#### 现状

`src/ops/kernel/q4_0/avx2.rs` 的 Q4_0 matmul kernel 故意**不使用 FMA**：

```rust
//! 8. Multiply by d * scale in f32 with explicit `_mm256_mul_ps` + `_mm256_add_ps`
//!    (no `_mm256_fmadd_ps`) to match scalar's mul+add rounding exactly.
//!
//! **Precision contract**: bit-exact with the scalar implementation.
```

行累加采用 `acc += dc * d * scale` 三次 f32 运算（mul → mul → add），
而不是 `fmadd(d, d*scale, acc)` 一次融合运算。

#### 影响

- **性能**：相比 Q4_K / Q8_0 kernel（已用 FMA）慢约 1.5-2×。
  E4B Q4_0 文件实测 5.8 t/s decode，E2B Q4_K_M 10 t/s（同 embd 比例下 kernel 速度是主要差异）。
- **正确性**：与 scalar reference **bit-exact**（`assert_avx2_eq_scalar` 测试在 `q4_0/avx2.rs:241`）。
  所有 Q4_0 模型通过 parity 测试。

#### 何时触发

加载任何 Q4_0 权重时——目前主要是 Gemma 4 E4B Q4_0 导出文件。

#### 选项

1. **加 FMA + 放松精度契约**（推荐 follow-up）：把 `acc += dc * d * scale` 改成
   `acc = _mm256_fmadd_ps(prod_d_scale, ...)`。需要新增"near bit-exact"测试模式
   （参考 llama.cpp 的 GGML_FMA_DISABLED 编译开关做法）。
   - 预期：1.5-2× 加速
   - 风险：所有 Q4_0 模型需要重新跑 Oracle 验证（logits 可能漂移 ±1 ULP，最终 token 偶发分叉）
   - 工作量：半天
2. **重导出模型到 Q4_K / Q8_0**（推荐立即做）：
   - `q4_k/avx2.rs` 已用 FMA 写好 + parity 通过
   - `q8_0/avx2.rs` 已用 FMA 写好 + parity 通过
   - 用 llama.cpp 的 `convert.py` 把 E4B 转成 Q4_K_M 或 Q8_0
   - 文件大 ~30%，但推理快 2-3×
   - 工作量：1 小时
3. **写 FMA + 8-wide row unroll 版本** — 保留旧 kernel，新增 `matmul_q4_0_vs_q8_0_avx2_fma`，编译时 feature flag 切换。
   - 预期：2-3× 加速
   - 工作量：1 天

#### 推荐

短期：选选项 2（重导出 E4B），立即得 2-3× 加速，零代码风险。
中期：选选项 1（加 FMA kernel），其他 Q4_0 模型（Qwen3-0.6B-Q4_0 等）也受益。

#### 关联文件

- `src/ops/kernel/q4_0/avx2.rs` — kernel 实现 + parity 测试
- `src/ops/kernel/q4_0/scalar.rs` — scalar reference
- `src/ops/kernel/q4_0/mod.rs` — kernel 分发

---

### TODO-002: SIMD GeGLU `tanh_approx` 未启用

#### 现状

`src/models/gemma4/trunk/forward.rs` 的 `ggml_geglu_fp16_inplace` 仍然走 scalar 路径。
AVX2+F16C SIMD 版本原型写过又回退了，注释详细记录在函数体内。

#### 影响

- **性能**：scalar GELU 算 10240 元素 × 2 个 GELU/层（FFN + per-layer）× 35 层 = ~717k scalar ops/token。
  scalar ~12ns/element = **~8.6 ms/token**（当前 E2B ~80ms/token 的 ~10%）。
- **正确性**：scalar 版本 bit-exact 同 llama.cpp。所有 gemma4 测试通过。

#### 选项

1. **接受 ±1-2 ULP drift 启用 SIMD**（推荐）：Padé [7/6] tanh 近似（~0.04% max error in |x| ≤ 4.5）；
   加 `#[cfg(feature = "experimental-geglu-simd")]` 默认关；
   用 `cargo run --features experimental-geglu-simd` 启用。
   预期：+5-8% ETE。风险：gemma4_reference.rs 中 token Oracle 偶尔 ±1 分叉。
2. **Schraudolph fast exp + tanh = (exp(2x)-1)/(exp(2x)+1)**：5-10% tanh error on |x|>2 → 必分叉。不推荐。
3. **保持 scalar**：零风险，零加速。

#### 推荐

短期：选项 3。中期：跑 gemma4_reference.rs Oracle 确认 Padé [7/6] 漂移可控后，切到选项 1。

#### 关联文件

- `src/models/gemma4/trunk/forward.rs:858` — 当前 scalar + 详细 doc-comment

---

### TODO-003: 统一 5 个 `load_weight` / `load_weight_any` 到单一 `core::load_weight_any`

#### 现状

仓库目前每个模型自维护一份 GGML weight loader，签名 / 白名单 / dims 拆解 / 错误返回都不一致：

| 实现 | 签名 | 接受类型 | dims 校验 | 错误返回 |
|---|---|---|---|---|
| `dots/weights.rs::load_weight` | `(source, name, dims)` | F32/F16/BF16/Q8_0 | 严格 | `Result<_, String>` |
| `qwen35/trunk/weights.rs::load_weight` | `(source, name)` | 8 种 | 不校 | `Option<Weight>` |
| `qwen35/trunk/weights.rs::load_weight_f32` | `(source, name) → Vec<f32>` | F32/F16/BF16 | n_elements | `Option<Vec<f32>>` |
| `gemma4/trunk/weights.rs::load_weight` | `(source, name, dims, ggml_type)` | 单一类型 | 严格 dims+type | `Result<_, String>` |
| `gemma4/trunk/weights.rs::load_weight_any` | `(source, name, dims, &[GGMLType])` | 白名单 | 严格 dims+白名单 | `Result<_, String>` |

`qwen3` 干脆没有 `load_weight` —— 直接展开 `Weight::from_quantized(QuantizedTensor::from_bytes(...))`，
天然支持 `QuantizedTensor::from_bytes` 接受的全部 20+ 类型。

**核心公共部分**（≈ 5 行）：

```rust
let info = source.tensor_info(name)?;
let bytes = source.tensor_slice(name)?;
let weight = Weight::from_quantized(QuantizedTensor::from_bytes(
    bytes, info.ggml_type, n_in, n_out,
));
```

#### 影响

- **未来加 IQ2/IQ4/Q6_K/Q5_K 等新量化类型**——每个 `load_weight` 都要扩白名单。
- **dims 拆解不一致**：`dims.last()` vs `dims[0]/dims[1]` vs `dims[..-1]` 累乘
- **白名单粒度不一**：写死 vs 参数化 vs 无
- **错误返回风格不一**：`Result<_, String>` vs `Option<Weight>`

#### 选项

1. **最小**：扩 `dots::load_weight` 接受 Q4_0/Q4_1。维持命名遗留。
2. **中度**：breeze 摆脱 dots 命名，改直接展开 `Weight::from_quantized(...)`（学 qwen3）。
3. **完整**：5 个 `load_weight` 全部迁到 `crate::core::tensor::load_weight_any`：
   - 签名：`(source, name, dims: &[u64], allowed: Option<&[GGMLType]>) -> Result<Weight, String>`
   - `dims` 校验可选；`allowed=None` 表示 `QuantizedTensor::from_bytes` 接受全部类型
   - 所有模型调用方改 import；`qwen35::load_weight_f32` 与 `core::load_f32_tensor` 合并去重
   - 工作量：~200 行；彻底统一

#### 推荐

**方案 3**。`load_weight` 不是 dots 私有的，它是项目 GGML weight loader 的事实标准入口——
`qwen3` 已经证明**不抽象也活得很好**，所以抽象必须带来明显收益。

#### 关联文件

- `src/models/dots/weights.rs:5-67`、`qwen35/trunk/weights.rs:80-124`、`gemma4/trunk/weights.rs:226-309`
- `src/core/tensor.rs:288-330` — `load_f32_tensor`

---

### TODO-004: Breeze Q4_0 不可用 — 走 per-tensor 混合精度路径

#### TL;DR

**Q4_0 单层全量化对 Breeze 不可用**（128 frames vs BF16 59 frames = 117% 偏差）。
短期路线不是"再换一种 4-bit 格式"，而是 **per-tensor 混合精度**：embedding / lm_head / codebook 保留 BF16 或 Q8_0，
hidden-attn / mlp 主矩阵走 Q4_0 或 Q4_K。长期再评估 Q4_K / Q6_K 全替换。

#### 现状

实测推理结果：

| 精度 | 帧数（prompt "你好。"，seed 42） | 文件大小 | 备注 |
|---|---|---|---|
| BF16 | 29 | 6.6 GB | 与原字节等价 |
| F16 | 35 | 6.6 GB | 同源 |
| F32 | 29 | 13 GB | md5 与 BF16 bit-exact |
| Q8_0 | 37 | 3.6 GB | **3.6× 加速，推荐部署** |
| **Q4_0** | **128** | 2.1 GB | **输出退化** |

#### 关联文件

- `tools/converter/breeze/convert_breeze.py` — `_must_keep_source` / `_source_ggml_type`
- `src/ops/kernel/quantized_tensor.rs:275-...` — `QuantizedTensor::from_bytes`
- `models/Breeze-TTS-2-gguf/breeze-tts-2-Q4_0.wav` — 当前 Q4_0 输出（退化证据）

详见原 `docs/TODO.md` §TODO-004 完整分析（256 行），含 per-tensor 风险表、推荐方案、实施步骤。

---

### TODO-005: F32 x86_64 matmul 缺 SIMD kernel — ✅ 已完成

`src/ops/kernel/f32/avx2.rs` 实现完成，F32 / F16 NEON kernel 完成，AVX2 matmul macro 共享代码。
BF16 / F16 / F32 三档精度 matmul 共享 `avx2_matmul_packed!` 宏。

实测：Breeze `--quant f32` scalar 234s → AVX2 54s (4.3×)，`--quant f16` 47s → 31s (1.5×)。
F32 与 BF16 md5 bit-exact (a7c5e3f5...)，三档精度 md5 都已记录 baseline。

---

### TODO-006: Q8_0 Breeze output drifts ~1 ULP under SIMD activation/matmul

**v3 baseline `b051f3c1...` / 29 frames 已选定**。BF16 / F16 / F32 bit-exact 是硬约束；
Q8_0 没有 ground truth（量化本身有损），后续每次 SIMD 改动会再次漂移——只要 ≤ 1 ULP 就接受，记录新 baseline。

详见 `tools/converter/README.md:40` Q8_0 行。

---

### TODO-007: Breeze `rope()` SIMD 路径精度破坏 — ✅ 已修复

`src/ops/rope/neox.rs::rope_neox_inplace_with_table` 新增，BF16 round-trip 严格按 scalar op 顺序
（mul → round → mul → round → add → round），AVX2 内核在每个乘加后立即 bf16 round。
`src/models/breeze/transformer.rs::rope` 一行调用。Breeze 三档精度 (BF16/F16/F32/Q8_0) bit-exact。

剩余：NEON 版本（aarch64）暂未实现，不阻塞 Breeze 路径。

---

### TODO-008: Breeze F16 合成听感更合理 / F32-BF16 需对齐

人工听审发现：Breeze TTS 2 在 BF16 / F16 / F32 下比特级输出一致（BF16 ↔ F32 md5 `c19502ff...`），
但听感上 F16 更合理（F16 md5 `4dd19b6d...`）。

数学上 BF16 = F32（高 16 位），两者 bit-exact 等价是数学正确；F16 与它们差异是 mantissa 精度 + exponent bias 不同。
这是**听感 vs 数值精度**的张力：Breeze 上游训练 / 参考实现大概率用 BF16 模拟精度，
F16 路径数学上不对但听感更"干净"——需要在 BF16 / F32 路径上找到听感偏差的根因。

详见 `models/Breeze-TTS-2-gguf/README.md` Alignment 段落。

---

### TODO-009: VibeVoice F16/F32 kernel panic + F32 matmul placeholder — ✅ 已修复

`f16::forward_prepared` / `f32::forward_prepared` 加 `input_f32.len() >= n_in` fallback；
`f32::forward_prequantized` 真实实现替换 `row_dot_range` 占位符；`QuantizedTensor::F32(Vec<f32>)` 改为 struct variant
存 `n_in / n_out`。全 4 个 VibeVoice LLM 精度 (BF16/F16/F32/Q8_0) zh.wav 2 chunks 输出 bit-equivalent
"Speaker 0: 我认为跑步最重要的就是给我带来了身体健康。"

---

### TODO-010: Prefill runtime 通用化，避免每个模型重复实现 chunk 调度

#### 现状

`src/core/prefill.rs` 目前只提供 `prefill_chunks()` 和 batch size 校验。
Qwen3、Qwen3.5、Gemma4 仍分别维护自己的 chunk loop、CPU/Vulkan fallback、KV/state snapshot、commit/rollback 和 logits 输出处理。

#### 目标

参考 llama.cpp 的 `llama_batch_allocr`、`llama_memory_context_i` 和 `process_ubatch` 分层：

- 公共 `PrefillRunner` 负责输入校验、chunk 切分、`base_position`、输出顺序、失败重试、backend fallback 和 chunk 级事务；
- 公共 `PrefillContext` / `ChunkTxn` 统一 row shape、状态快照、commit/rollback；
- 模型只实现 `PrefillKernel::forward_chunk`，保留各自的 attention、SSM/conv、MoE 和多模态 position 语义；
- 不支持多行 batch 的模型声明 singleton fallback，不阻塞模型接入。

#### 实施顺序

1. 抽 `PrefillRunner` 包裹现有 Qwen3/Qwen3.5/Gemma4 loop，不改变数值行为。
2. 抽 `PrefillContext` / `ChunkTxn`，统一 KV/state commit、rollback 和 CPU/GPU fallback。
3. 模型接口收敛为 `PrefillKernel` + capability；无 batch 能力时自动走 batch=1。
4. 复用 shape 兼容的 scratch/graph，补充 batch=1、跨 chunk、失败回滚和真实 GGUF parity/benchmark。

#### 关联文件

- llama.cpp：`src/llama-batch.h`、`src/llama-memory.h`、`src/llama-context.cpp`、`src/llama-graph.h`
- 当前 Rust：`src/core/prefill.rs`、`src/ops/kernel/mod.rs::PreparedRows`、`src/models/qwen3/trunk/prefill.rs`、`src/models/qwen35/trunk/session.rs`、`src/models/gemma4/trunk/forward.rs`

---

## Archived / completed work

### MiniCPM5-1B 输出对齐 Llama.cpp (2026-08) — ✅ 已解决

**根因：RoPE 风格用错。** `llama` GGUF 架构（含 MiniCPM5）使用 GGML `ROPE_TYPE_NORM`（interleaved 相邻对旋转），
而 Rust trunk 之前调用了 `rope_neox`（halves 风格）。

**修复**：`src/ops/rope.rs` 新增 `rope_norm`（interleaved 相邻对旋转），带 pinned-bits 单元测试；
`src/models/llama/trunk/forward.rs` Q/K RoPE 改用 `rope_norm`。

**验证**：MiniCPM5-1B-Q8_0 17 token prompt，Rust vs llama.cpp oracle 8 步贪心 top-1 完全一致。

### LFM2-8B-A1B (lfm2moe) MoE 支持 (2026-08) — ✅ 已完成

新增 `src/models/lfm2moe/`，实现 `lfm2moe` 架构的 Mixture-of-Experts FFN。
对齐 llama.cpp `build_moe_ffn(gating_op=sigmoid, norm_w=true)`：sigmoid 路由 + 偏置仅影响 top-k 选择 +
无偏 probs 归一化 + 加权求和。

排查过程发现并修复的既有 bug（**模式性 bug**，影响所有 trunk）：
1. **silu 与 matmul 的行分区竞争** — `n_ff % nth != 0` 时 silu 读到 matmul 尚未写入的行。
   闭包内统一为 ceil 分区：`llama`、`lfm2`、`lfm25`、`lfm2moe` 四处。
2. **shortconv conv 状态语义** — 解码期状态更新应为滑动窗口而非全行复制。
3. **lfm2/lfm25 conv 权重 2-D 形状** — 兼容 1-D 与 2-D 两种存储。

### LFM2.5-1.2B（dense）对齐 llama.cpp (2026-08) — ✅ 已完成

修复 conv 2-D 加载 + silu 分区竞争 + conv 状态滑动窗口后验证：
`What is the capital of France?` 13 token prompt，**8/8 步贪心 top-1 完全一致**（1098, 5706, 803, 4481, 856, 5242, 523, EOS=7），EOS 时机一致。

### Ornith-1.5-9B (qwen35 架构, 9B hybrid) 适配 (2026-08) — ✅ 已完成

33 blocks = 32 主层 + 1 MTP nextn 层；SSM 线性注意力 `full_attention_interval=4`，GQA 16/4 头，head_dim 256 + partial mrope 64/[11,11,10,0]。
适配点：
1. KV cache 按请求预算分配（原为 OOM 根因）：`(prompt_len + max_tokens).min(n_ctx)`。
2. nextn/MTP 层排除：新增 `config.n_nextn`，主栈层只遍历 n_layer_impl。
3. 上述封顶同时惠及 Qwen3.5-2B。

验证：llama.cpp oracle 8/8 步一致。封顶同时惠及 Qwen3.5-2B（原每次运行 KV cache 固定分配 12.9GB）。

### Spark-X2.5-1.7B / 4B-GGUF 适配 — ✅ 已完成（功能）+ ⚠️ 性能待优化

`arch=spark2_5`，Xunfei Spark 2.5 讯飞星火。适配点：
- **Fused QKV**：`attn_qkv [n_embd, n_embd_qkv]` 单一投影 → split 为 Q/K/V。
- **Per-layer heterogeneous RoPE**：full-attn `n_rot=64, freq_base=5M`；SWA `n_rot=256, freq_base=10K`。新增 `rope_neox_partial`。
- **Sliding window mask**：SWA 层只看最后 `sliding_window=512` 个位置。
- **Per-head gating**：`attn_gate` pre-attention norm → sigmoid → 逐 head 乘 attention output。
- **GeGLU FFN**：`gelu(gate(x)) * up(x)` 后 down。
- **Tied embeddings**：GGUF 无 `output.weight` → 复用 `token_embd.weight`。
- **Tokenizer pre=`spark2_5`**：与 qwen2 正则不同，新增 `PreTokenizer::Spark2_5`。

冒烟通过。1.7B ≈ 1.1 tok/s, 4B ≈ 0.4 tok/s（4 线程；生产路径未优化）。

待优化：
- [ ] Per-token prefill O(N²) 注意力 → batched prefill
- [ ] 每次 `decode_step` 全量 `Vec::new` → ExecutionScratchpad
- [ ] Generate 阶段没有 KV cache 预热
- [ ] FFN gate 输入源（pre-attention norm vs post-attention hidden）
- [ ] GeGLU 用 `0.5 * x * (1 + tanh(...))` 近似 vs llama.cpp 默认精确 GELU
- [ ] chat template 整段手写，未走 `tokenizer.chat_template` 的 Jinja 引擎
- [ ] 精度对齐 XFllama.cpp oracle
- [ ] `get_f32_tensor` 重复 6 处（暂不抽，等待 `RequiredTensor` trait 落地）

---

## SIMD extension inventory

This project targets the following SIMD instruction sets:

| Architecture | Extensions used | Detection | Files |
|---|---|---|---|
| **x86_64** (Intel Core Ultra 5 125H, AMD Zen 4, etc.) | SSE2 / SSE4.2 / AVX / **AVX2** / **FMA** / **F16C** / **BMI2** | `is_x86_feature_detected!` | `src/ops/simd_avx2.rs`, `src/ops/kernel/*/avx2*.rs` |
| x86_64 (Skylake-X, Ice Lake, Sapphire Rapids, Zen 5) | + **AVX-512F** / **AVX-512BW** / **AVX-512VNNI** | runtime gate | (not yet wired) |
| x86_64 (Intel Alder Lake+, Zen 5) | + **AVX-VNNI** (`vpdpbusd`/`vpdpwssd`) | CPUID `avx_vnni` | (see TODO-AVX-VNNI below) |
| **aarch64** | NEON / NEON-F16 / NEON-F32 | `is_aarch64_feature_detected!` | `src/ops/kernel/*/neon*.rs` |
| aarch64 (Apple M1+) | + NEON-DOT / SME | runtime | (not yet wired) |

### AVX-VNNI / AVX-512 background

**AVX-VNNI** (`vpdpbusd`/`vpdpwssd`, also written as `vpdpbusd_yx_yx` or `_mm256_dpbssd_epi32`)
is an Intel/AMD x86 extension that performs **saturating int8 dot-product with a
single FMA-like instruction**. First shipped on Intel Tiger Lake (2020) and AMD Zen 5;
the current dev box (Core Ultra 5 125H, Meteor Lake) supports it via the `avx_vnni`
CPUID bit. It computes `acc += (a - 128) * b` over 16 int8 lanes with saturating accumulation
to int32 — exactly the operation our Q8_0 × Q8_0 matmul does via three separate
AVX2 instructions (`_mm256_maddubs_epi16` → unpack → `_mm256_madd_epi16` → CVTPS).

**AVX-512 VNNI** (`vpdpbusd` with 64-lane / 32-int32 ops on zmm registers) is the server-class
extension of the same idea: 2× the throughput per instruction and the only way to reach
> 50 tok/s on Qwen3-0.6B decode. Sapphire Rapids, Zen 5, and Zen 6 (Medusa Ridge) support it;
our dev CPU does not.

### TODO-AVX-VNNI: int8 dot 加速

**目标**：在已有 `q8_0/avx2.rs` 的 3-instruction int8 dot (`_mm256_maddubs_epi16` →
unpack → `_mm256_madd_epi16` → CVTPS) 基础上加一条 AVX-VNNI 路径
(`_mm256_dpbsud_epi32` / `_mm256_dpbssd_epi32`)，把 saturating int8 dot
缩成 1 条指令。预期 1.3-1.5× Q8_0 matmul 加速。

**实现计划**：
1. 加 `#[target_feature(enable = "avx2", enable = "avxvnni")]` 的 kernel
   `matmul_q8_0_vs_q8_0_avxvnni(weight, q8, scales, output, n_in, n_out, start, end)`，
   内部用 `_mm256_dpbsud_epi32` (unsigned×signed saturating dot)。
2. 在 `q8_0/dispatch.rs::matmul_q8_0_quantized_range` 加 `is_x86_feature_detected!("avxvnni")` 分支。
3. 加 parity 测试 `q8_0_avxvnni_matches_scalar_q8_dots`，相对 scalar 1 ULP 内。
4. 在 `src/ops/kernel/mod.rs::Kernel` dispatch 表里 KV 缓存值的 int8 dot 也走 AVX-VNNI。

**风险**：
- `_mm256_dpbssd_epi32` saturate 到 int32 → 16 lanes per dot。K2-Horizon-4B / Breeze 等
  Q8_0 权重 dot 输出通常在 ±10⁴ 量级，不会触发 saturate，所以等价于无 saturate 的 int32 dot。
- AVX-VNNI 在 Meteor Lake 上只有 256-bit 寄存器（不是 512-bit）。所以加速上限是"少 2 条 unpack 指令"，
  不是 2× lane 宽。
- 加 `target_feature("avxvnni")` 意味着老的 CPU 不能跑这条路径（fallback 到 AVX2）。

**验收**：
- Q8_0 Q4_0 matmul parity 测试通过（1 ULP 内）
- Qwen3-0.6B-Q8_0 decode ≥ 36 t/s（vs 当前 ~28 t/s），用 `--max-context 8192` 默认 cap
- K2-Horizon-4B-Q8_0 decode ≥ 8 t/s（vs BF16 当前 4 t/s），关键前置：**必须** `--max-context 8192`
  否则默认走 524288 → 77 GB OOM（见 TODO-MAX-CONTEXT-CLI）

### TODO-LLAMA-PER-TOKEN-SIMD化

**目标**：扫描 `src/models/llama/trunk/forward.rs` 仍为 scalar 的 per-token 小循环，
逐个替换为 SIMD 版本。AVX2/FMA/NEON 已覆盖 `vec_scale_f32` / `vec_add_into` / `vec_mad_f32` 等基础算子。

**已审计的热点**（per token × per layer）：

| 操作 | 当前 | 行号 | 优化 |
|---|---|---|---|
| RMS norm grouped | ✅ SIMD via `rms_norm_grouped` → `scale_mul_avx2` | 462, 750 | - |
| Q8_0 quantize | ✅ SIMD via `quantize_q8_0_into_avx2` | 464, 766 | - |
| Q8_K quantize | ✅ SIMD via `quantize_row_q8_k_avx2` | 472, 772 | - |
| BF16 × Q8 matmul | ✅ AVX2 (`matmul_bf16_vs_q8_avx2`) | 484, 702, 804 | - |
| **Embedding scale** | ❌ scalar `for v in x.iter_mut() { *v *= scale; }` | 419-422 | ✅ **已替换为 `vec_scale_f32`**（commit 见 K2check 分支） |
| RoPE (k2-horizon neox) | ✅ SIMD `rope_neox_inplace_avx2` | 528-544 | - |
| KV cache write (F32) | ✅ `copy_from_slice` (memcpy → SIMD 路径) | 577-583 | - |
| KV cache write (F16) | ✅ `f32_slice_to_f16_avx2` (F16C) | 559-568 | - |
| Attention (F16) | ✅ SIMD `dot_f16_f32`, `vec_mad_f16_f32`, `vec_scale_f32` | 609-630 | - |
| Attention (F32) | ✅ SIMD `dot_f32`, `vec_mad_f32` | 636+ | - |
| Residual add | ✅ `vec_add_into`, `vec_mad_f32` | 720-724, 910-914 | - |
| silu_mul | ✅ SIMD `silu_mul_approx_inplace` | 831, 840 | - |
| **KV cache / scores buffer size** | ❌ hardcoded `512.min(cfg.n_ctx)` + `[0.0f32; 512]` | 117, 941 | ✅ **已替换为 `cfg.n_ctx.min(max_context)` + `vec![0.0f32; max_ctx]`**（见 TODO-MAX-CONTEXT-CLI） |

**结论**：除 `embedding_scale` 外 llama trunk 已是 SIMD 化。`embedding_scale` 修复已落地 K2check 分支
（每个 token 节省约 1.3 µs @ n_embd=1536，对总推理时间影响 < 0.1%）。

剩余 per-row attention 在 autoregressive decode 时是 sequential（依赖前面 token 的 KV），
SIMD 不能跨 token 加速——这部分要看 [TODO-010](#todo-010-prefill-runtime-通用化避免每个模型重复实现-chunk-调度) batched prefill。

### LLama trunk matmul 真实瓶颈（per-token）

解码时 n_out=1，所有 matmul kernel 的内并行（按 n_out 切 rows）失效——线程利用率 12.5%（1/8）。
**这是 autoregressive decode 慢的真凶**，不是 SIMD 缺位。

修复路径见 TODO-010（batched prefill）+ 自动接受单 token decode 慢的现实（llama.cpp 同样慢）。

### TODO-MAX-CONTEXT-CLI: `--max-context` 用户可调 KV cache 上限

**背景**：`src/models/llama/trunk/forward.rs` 和 `src/models/lfm2/trunk/forward.rs` 历史上都把
KV cache 上限硬编码到 512：

```rust
// llama trunk 历史硬编码
let max_ctx = 512usize.min(config.n_ctx);
// lfm2 trunk 历史硬编码
let max_ctx = 512usize.min(cfg.n_ctx);
// lfm2 attention `values` 数组
let mut values = [0.0f32; 512];  // attention 长生成时越界 panic
```

后果：512 token 以上的生成直接越界 panic，或更长生成被静默截断。多个用户的报告
（VibeVoice ASR 1.5B 长音频、K2-Horizon 中文翻译模型 chat 段、K2-Horizon-4B 192K 长输出）
都触及过同一类 bug。

**修复**（已落地 K2check 分支）：

- llama trunk / lfm2 trunk：`max_ctx = config.n_ctx.min(max_context)`，配套
  `run_inference[_stream]` 加 `max_context: usize` 参数
- lfm2 attention `values`：`vec![0.0f32; max_ctx]`
- 新 CLI 参数 `--max-context N (default 8192)`（`CliOptions::DEFAULT_MAX_CONTEXT`），
  `effective_max_context()` helper
- 全链路贯通：`src/app/cli.rs` / `src/app/text.rs` / `src/main.rs` /
  `src/models/lfm2/vision.rs`（multimodal LFM2-VL 也接 max_context）

**为什么用 CLI 而不是无脑用 model claim**：部分模型 GGUF `context_length` 字段是
"该 GGUF 可接受的 max"，而不是该模型实际有意义的长度。例如：

| 模型 | claim `context_length` | 512-tok panic | CLI `--max-context 8192` |
|------|------------------------|---------------|---------------------------|
| K2-Horizon-1B | 131072 | 是 | KV cache 0.29 GB |
| K2-Horizon-4B | **524288** | **77 GB OOM** | KV cache 0.86 GB |
| LFM2-1.2B | 32768 | 是 | KV cache 0.5 GB |
| Qwen3-0.6B | 32768 | 是 | KV cache 0.4 GB |

用户按需 `--max-context` 即可（绝大多数 chat 8K 够，长上下文特殊任务手动加）。

**后续**：

- `docs/usage/llama.md` / `docs/usage/qwen3.md` 等可加一行 "默认 KV cache cap 是 8K，可用
  `--max-context N` 覆盖"。
- TODO-AVX-VNNI / TODO-LLAMA-PER-TOKEN-SIMD 实现后，再跑一遍 K2-Horizon-4B
  对比 `--max-context 8192` vs `--max-context 32768` 的 prefill 时间（验证大 context
  不会因为 KV 随机访问模式变慢）。