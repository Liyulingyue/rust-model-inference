# 代码重构规划：rust-model-inference

> **文档用途：** 本文档规划对当前平铺式 `src/*.rs` 代码基础进行一次全面重构，目标是在**不改变任何模型行为**的前提下，将 31,450 行代码重组为可分目录、可单独单元测试、依赖边界清晰的分层结构。

---

## 1. 现状分析

### 1.1 文件规模（总计 31,450 行）

| 文件 | 行数 | 职责 | 问题 |
|------|-----:|------|------|
| `main.rs` | 4304 | CLI 解析 + 6 个 `run_*` 驱动 + 单元测试 | 入口、CLI、各任务驱动、测试全部混在一个文件 |
| `ggufrs.rs` | 4344 | `.ggufrs` 多组件容器格式（读写/校验） | 独立领域却与模型代码平铺并列 |
| `ops.rs` | 3592 | 计算内核：matmul/norm/rope/quantize/attention/ssm | 单一"上帝文件"，多类内核混杂 |
| `qwen3a.rs` | 3334 | Qwen3-Audio 模型 | 与 qwen3/qwen35 重复大量 Layer 结构 |
| `qwen3.rs` | 1827 | Qwen3 文本模型 + 最近的 `text_encode` | 模型本体与编码器逻辑混合 |
| `vision.rs` | 1820 | 视觉编码器 | 独立但挂在根目录 |
| `qwen35.rs` | 1832 | Qwen 3.5 模型 | 与 qwen3 相似度高，重复 |
| `asr.rs` | 1806 | ASR 运行时（转录驱动） | 属于应用层却与模型平铺 |
| `model.rs` | 1395 | GGUFLoader / TensorSource / MetaValue / GGMLType | 加载器与类型定义混合 |
| `load_plan.rs` | 1226 | 异构设备加载规划 | 独立领域 |
| `tokenizer.rs` | 1154 | BPE tokenizer | 独立领域 |
| `pig.rs` | 1127 | Z-Image（Pig）扩散模型 | 独立领域，当前含活跃 bug |
| `quant.rs` | 1038 | Q4K/Q5K/Q6K 量化与 matmul | 与 ops.rs 的 matmul 入口重复 |
| 其余 | ~1800 | thread_pool / clip_config / vulkan / wgpu / memory / prompt / traits / scratchpad / parity_trace | — |

### 1.2 明确问题清单

1. **main.rs 职责混杂（4304 行）**：`CliOptions` 解析、`run_asr_cli`、`run_embedding`、`run_dump_logits`、`run_shared_inference`、`run_pig_image`、`run_inference`、`run_interactive`、`run_multimodal`、`run_self_test` 全部在同一文件，另有 700+ 行单元测试。
2. **ops.rs 是"上帝文件"（3592 行）**：f16/f32 转换、norm、rope、激活、quantize、全部 matmul 变体、embedding lookup、采样、SSM 运算混在一起，对外 re-export 全部 `pub use ops::*`。
3. **matmul 入口重复**：`ops.rs`（`matmul_q8_0`, `matmul_q8_0_quantized*` 等）与 `quant.rs`（`matmul_q4k_q8k`, `matmul_q5k_q8k`, `matmul_q6k_q8k`）各自维护一套维度/调度逻辑。
4. **模型间重复**：qwen3 / qwen35 / qwen3a 有相似但复制粘贴的 Layer 权重访问、RoPE、scratchpad。
5. **lib.rs 平铺 re-export**：无目录结构，所有模块平级；公共 API 靠 `pub use xxx::*` 一把梭。
6. **F16 无统一 matmul 入口**：`ProcessedWeight` enum（ops.rs:1654）只覆盖 F32/Q8_0/Q6_K/Q4_0/Q4_1，F16 各模型用自己的 `matmul_f16_f32` 路径。

### 1.3 基线（重构零风险的前提）

- `cargo build --release` 通过（仅 warnings）。
- `cargo test --release`：163 通过，2 失败，6 ignored。
  - 失败 1：`ggufrs::tests::open_does_not_map_sparse_segment_region` — 环境相关（sparse 段映射，FileTooLarge），视环境而定。
  - 失败 2：`tokenizer::tests::rejects_missing_or_unknown_qwen_pre` — tokenizer 对 `default`/`qwen35` pre 的预期断言，与本次重构无关。
- 所有重构阶段必须维持此基线不破坏（上述 2 个失败不属于重构引入）。

---

## 2. 目标结构

指导思想：**按领域分层，物理拆分优先，行为重构放后。** 依赖方向只能从上向下：
`app → models → ops / core / format`。

```
src/
├── lib.rs                    # 仅声明模块 + 精选 re-export
├── main.rs                   # 薄层：解析 CLI → 分派到 app::* （目标 ~300 行）
│
├── app/                      # 应用驱动（拆自 main.rs）
│   ├── mod.rs                # main() 分派逻辑
│   ├── cli.rs                # CliOptions / 参数解析 / 校验
│   ├── text.rs               # run_inference / run_interactive / run_shared_inference
│   ├── image.rs              # run_pig_image
│   ├── audio.rs              # run_asr_cli
│   ├── embedding.rs          # run_embedding + EmbeddingActivationScratch + 相关测试
│   ├── logits.rs             # run_dump_logits
│   └── selftest.rs           # run_self_test
│
├── core/                     # 领域无关基础设施
│   ├── mod.rs
│   ├── tensor.rs             # TensorSource / TensorInfo / GGMLType / MetaValue  ← 拆自 model.rs
│   ├── loader.rs             # GGUFLoader                                        ← 拆自 model.rs
│   ├── model.rs              # QuantizedLinear / ModelGraph / ModelConfig        ← 拆自 model.rs
│   ├── tokenizer.rs          # BPETokenizer（原样搬移）
│   ├── memory.rs             # BlockAllocator / MemoryArena / KV（原样搬移）
│   ├── thread_pool.rs        # ComputePool（原样搬移）
│   ├── scratchpad.rs         # ExecutionScratchpad（原样搬移）
│   └── traits.rs             # ExecContext / Layer / ModelConfig（原样搬移）
│
├── ops/                      # 计算内核（拆自 ops.rs + quant.rs）
│   ├── mod.rs                # pub use 各子模块
│   ├── float.rs              # f16/f32 转换、SIMD 转换辅助
│   ├── norm.rs               # rms_norm / rms_norm_inplace
│   ├── rope.rs               # rope_neox / rope_mrope / rope_mrope_interleaved
│   ├── activation.rs         # silu / gelu / 相关融合
│   ├── quant.rs              # quantize_q8_0* / Q4K/Q5K/Q6K quantize/dequantize  ← 并入原 quant.rs
│   ├── matmul.rs             # 全部 matmul 内核 + MatmulTask + ProcessedWeight
│   ├── attention.rs          # softmax / attention_value / FlashAttention
│   ├── sampling.rs           # argmax / top_k / 采样
│   └── ssm.rs                # SSM 状态运算（qwen3a 专用）
│
├── format/                   # 文件格式领域
│   ├── mod.rs
│   ├── ggufrs.rs             # .ggufrs 容器读写（原样搬移）
│   ├── load_plan.rs          # 异构设备加载规划（原样搬移）
│   └── gguf.rs               # GGUF 低级解析（若从 loader.rs 中进一步拆出）
│
├── models/                   # 模型实现（依赖 core + ops）
│   ├── mod.rs                # 公共 trait：CommonLayer / ModelTrait 等（由 Step 4 引入）
│   ├── qwen3.rs              # Qwen3 文本模型
│   ├── qwen3_text_encode.rs  # text_encode 独立（pig 依赖）
│   ├── qwen35.rs             # Qwen 3.5
│   ├── qwen3a.rs             # Qwen3-Audio
│   ├── vision.rs             # VisionEncoder
│   ├── asr.rs                # AsrRuntime
│   └── diffusion/
│       ├── mod.rs
│       ├── pig.rs            # PigModel / PigConfig（原样搬移，bug 另文档跟踪）
│       └── vae.rs            # PigVAE
│
└── backend/                  # 硬件后端（可选，见决策点 4）
    ├── mod.rs                # Device 抽象 + CpuBackend
    ├── cpu.rs
    ├── vulkan.rs
    └── wgpu.rs
```

### 依赖约束

- `app/*` 可依赖 `core`、`ops`、`format`、`models`。
- `models/*` 可依赖 `core`、`ops`、`format`；**不得**依赖 `app`。
- `ops/*` 可依赖 `core`；不得依赖 `models`。
- `core/*` 不依赖任何上层。

---

## 3. 分阶段执行计划

原则：**每阶段结束必须 `cargo build --release` + `cargo test --release` 通过（维持 1.3 基线），且每阶段是一个独立、可 merge 的提交。**

### Phase 1 — 拆 main.rs（风险：低，收益：最大）

- 将 `CliOptions` / `parse_cli_options` / `validate_cli_options` → `app/cli.rs`。
- 将 6 个 `run_*` 驱动按职责拆分到 `app/{text,image,audio,embedding,logits,selftest}.rs`；`main.rs` 仅保留 `fn main()` 分派 + `resolve_*` 辅助。
- 将 main.rs 内嵌的 `cli_tests` 测试模块迁移到对应 app 子模块的 `#[cfg(test)]`。
- 产出：`main.rs` 4304 → ~300 行；`app/` 各文件 <800 行。

### Phase 2 — ops.rs 类型抽象 + 物理拆解（风险：低→中）

**核心理念：先定义接口类型，再迁移函数实现。** 即使某些函数暂时还没接入新类型，接口也要先预留，避免后期推翻重来。

#### Phase 2.1 — 骨架：定义类型抽象（风险：零）

在 `src/ops/mod.rs` 顶部新增一个 `pub mod kernel` 子模块，只放**类型定义 + trait + 接口声明**，不写实现细节。

- 定义 `Kernel` trait（内核统一接口）：
  ```rust
  pub trait Kernel {
      fn forward(&self, input: &[f32], output: &mut [f32], n_in: usize, n_out: usize);
      fn forward_batched(&self, /* ... */) { /* 默认调用 forward 单 token 循环 */ }
  }
  ```
- 定义 `QuantizedTensor<'a>` 枚举（**保留所有类型的入口**，即使某些类型暂未实现）：
  ```rust
  pub enum QuantizedTensor<'a> {
      F32(&'a [f32]),
      F16(&'a [u8]),   // 预留
      Q8_0(&'a [u8]),
      Q6_K(Q6_KWeight<'a>),
      Q4_0(Q4_0Weight<'a>),
      Q4_1(Q4_1Weight<'a>),
  }
  ```
- 保留现有 `ProcessedWeight` 公共函数签名；`QuantizedTensor` 仅作为新接口的**占位类型**（暂时不替换）。
- 验证：`cargo build + cargo test` 全绿，与基线一致。
- 产出：新增 `src/ops/kernel.rs` (~80 行类型 + trait + 占位 enum)；`ops.rs` 行为零变化。

#### Phase 2.2 — 迁移 F32 matmul 到 Kernel trait（风险：低）

最小动作：只动 F32 路径。

- `ops.rs` 提取 `matmul_f32_scalar_range` + `matmul_f32_parallel_rows` 到 `ops/kernel/f32.rs`。
- 实现 `Kernel for F32Kernel<'_>`。
- 在 `ops/mod.rs` 加 `pub use kernel::f32::F32Kernel`，原 `matmul_f32_scalar_range` 签名标记 `#[deprecated]` 或保留兼容 wrapper。
- 加 `#[cfg(test)]` 单测：`F32Kernel::forward(&[1; 32], &mut [0], 32, 1)` → `[32]`。
- 验证：190 passed, 2 failed (基线)。

#### Phase 2.3 — 迁移 Q8_0 matmul 到 Kernel trait（风险：低）

- 提取 Q8_0 内核到 `ops/kernel/q8_0.rs`。
- `Q8Kernel<'_>` 实现 `Kernel`。
- 加测试。

#### Phase 2.4 — 预留 F16 matmul 接口（风险：低）

- `ops/kernel/f16.rs` 定义 `F16Kernel` 结构 + `impl Kernel for F16Kernel` 的**接口骨架**。
- 实现部分暂时调用现有 `matmul_f16_f32`（pig.rs 已在用），不重写。
- 不改任何调用方。

#### Phase 2.5 — 迁移 Q6_K / Q4_0 / Q4_1 到 Kernel trait（风险：中）

- 提取 Q6_K → `ops/kernel/q6_k.rs`。
- 提取 Q4_0 / Q4_1 → `ops/kernel/q4.rs`。
- 复用现有 quant.rs 的 `BlockQ8K` 与 `vec_dot_q*k_q8k` 函数（**不动 quant.rs**，先物理搬移到 `ops/quant/`）。

#### Phase 2.6 — quant.rs 物理搬移到 ops/quant/（风险：低）

- `quant.rs` 全部代码并入 `ops/quant/` 子模块（block.rs、q4k.rs、q5k.rs、q6k.rs、q8k.rs、dequant.rs）。
- lib.rs 删 `pub mod quant`，所有引用从 `crate::quant::*` 改为 `crate::ops::quant::*`。

#### Phase 2.7 — 文件物理拆解 ops.rs（风险：中）

只有当前面 6 个子阶段全部跑过、接口稳定后，才动 `ops.rs` 文件拆分。

- 按 kernel/quant/norm/rope/activation/sampling/ssm/embedding 拆文件。
- 每拆一个文件，确保 `cargo test` 全绿再继续。
- 最终产出：`ops.rs` 3592 → 7-8 个 <700 行的子文件。

#### Phase 2.x 验收

每阶段后必须：
- `cargo build --release` 零 error；
- `cargo test --release` 与基线一致（190 passed, 2 failed, 8 ignored）；
- 关键命令验证：
  - `cargo run -- --model models/Qwen3-0.6B-Q8_0.gguf --prompt "Hello" --max-tokens 20`（Q8 文本）
  - `cargo run -- --model models/Qwen3-Embedding-0.6B-Q8_0.gguf --prompt "test" --embedding`（embedding）
  - `cargo run -- --model models/Qwen3-ASR-0.6B-Q8_0.gguf --mmproj ... --audio models/zh.wav --language Chinese`（ASR）
- cos 相似度与 llama.cpp 对比 ≥ 0.9999。

### Phase 3 — 拆 model.rs（风险：低）

- `TensorSource` / `TensorInfo` / `GGMLType` / `MetaValue*` → `core/tensor.rs`。
- `GGUFLoader` / `model_config_from_source` → `core/loader.rs`。
- `QuantizedLinear` / `ModelGraph` → `core/model.rs`。
- 更新全项目引用（grep 替换）。

### Phase 4 — 拆 qwen3.rs + 建立模型公共接口（风险：中）

- `text_encode` 及其 attention 辅助移入 `models/qwen3_text_encode.rs`（pig 依赖面收窄）。
- 为 qwen3 / qwen35 / qwen3a 抽取 `CommonLayer` trait（`models/mod.rs`），先收敛**读取权重 + 类型分派**部分（各模型已基本符合），不强制重构推理主循环。
- 产出：消除 qwen35 / qwen3a 中与 qwen3 重复的权重访问样板在 trait 层面的显式复用。

### Phase 5 — 搬移 format / core（风险：低）

- `ggufrs.rs` → `format/ggufrs.rs`，`load_plan.rs` → `format/load_plan.rs`。
- `tokenizer.rs` / `memory.rs` / `thread_pool.rs` / `scratchpad.rs` / `traits.rs` → `core/`。
- 建立 §2 的依赖约束，lib.rs 改为分层声明 + 精选 re-export（替换 `pub use xxx::*` 一把梭）。

### Phase 6 — 收尾清理（风险：低）

- 删除死代码（重复的 `sample_token` / `f16_to_f32` / `per_second` 等）。
- 合并 main.rs 与 lib 测试中重复的辅助函数。
- 更新 `ARCHITECTURE.md` 的 Key Files 段与 README 中的文件引用。

## 4. 后续（非本期，逐步评估）

| 项 | 说明 | 前置 |
|----|------|------|
| `Kernel` trait 全面落地 | 所有量化类型接入 `Kernel` 统一调度 | Phase 2.1-2.5 已铺垫 |
| `ProcessedWeight` → `QuantizedTensor` 迁移 | 引入 F16 之后，替换旧 enum | Phase 2.3 / 2.4 后可启动 |
| FFN / Gate-Up 融合 | 减少 pool.compute() 调用 | 独立性能任务 |
| qwen3a / qwen35 主循环重构 | 对齐 CommonLayer 推理路径 | 依赖 Phase 4 |
| backend/ 目录化 | vulkan / wgpu 归入 backend/ | 实时进行，见决策点 4 |

---

## 5. 决策点

1. **拆分粒度**：默认"物理拆分"（纯移动代码、不改逻辑），行为重构（如统一 matmul 调度）推后。是否接受？
2. **GPU 模块归属**：`vulkan.rs`/`wgpu.rs` 目前被 `ops.rs` 通过 feature 引入（`init_gpu` / `get_vulkan_context` 等）。可直接并入 Phase 2 的 `ops/`（含 backend 逻辑），或延后单独成 `backend/`。倾向：先并入 `ops/`，独立 `backend/` 作为后续演进。
3. **qwen3a 的行数**：`qwen3a.rs` 3334 行，Phase 4 只做 trait 收敛权重访问，不拆文件；是否在 phase 4 顺带物理拆 `qwen3a/{model,audio_processor}.rs`？倾向：拆，风险低。
4. **ggufrs.rs（4344 行）**：是否在 Phase 5 拆成 `format/ggufrs/{read,write,validate}.rs`？倾向：VOL 先整文件搬移，内容拆分放后续。

---

## 6. 验收标准

- 全部阶段完成后：`cargo build --release` 零 error（允许 warnings）。
- `cargo test --release`：与基线一致（163 通过，2 环境性失败，6 ignored），不新增失败。
- `wgpu`/`vulkan` feature 组合仍可编译：
  `cargo build --release --features vulkan` 与 `cargo build --release --features wgpu`。
- 关键路径烟测不受影响：
  - Qwen3 文本生成：`cargo run --release --bin rust-model-inference -- --model models/Qwen3-0.6B-Q8_0.gguf --prompt "Hello" --bench`
  - Pig 图像生成（当前为 debug 中状态，不作为本重构回归项，仅确认不因重构而恶化）。
- 目录规模目标：`main.rs` ≤ 400 行；根目录 rust 文件全部消失（全部归入子目录）。