# 模型组织规范：`models/{name}/` 目录结构

> **文档用途：** 本文档定义 `src/models/` 下每个模型实现的统一目录结构、命名约定与职责边界。所有新模型与现有模型重构必须遵守。配套的执行路线图见末尾 §5。

---

## 1. 核心洞察：LLM trunk + 多模态 sibling

每个推理模型都由两部分组成：

1. **trunk**（主干）：纯 transformer 解码器，把 token 序列变成下一个 token 的 logits（或中间 hidden states）。
2. **多模态 sibling**（可选）：前拼的编码器（audio/image → tokens）或后拼的解码器（logits → audio）。

数据流：

```text
  [AudioEncoder] → tokens ─┐
                            ├──→ trunk ──→ logits ──┬──→ [TextDecoder (BPE)]
  [ImageEncoder]  → tokens ─┘                       └──→ [AudioDecoder (codec)]
```

**历史包袱**：qwen3 起初是纯文本模型，逻辑都在 `base.rs`；后来陆续加入 ASR（audio 前拼）与 TTS（audio 后拼），代码直接追加到 `base.rs`，导致单文件膨胀到 1971 行，职责混杂。本规范把"主干"概念显式化，强制多模态模块作为 trunk 的 sibling 而非嵌套。

---

## 2. 标准目录结构

```text
src/models/{model_name}/
├── mod.rs              # 顶层 re-exports（保持向后兼容的 Model / Session 名）
├── trunk/              # 必含：纯 transformer 解码器
│   ├── mod.rs          # pub use + 内部 wiring
│   ├── config.rs       # 超参（n_layer, n_embd, n_head, n_embd_head, n_ff, eps, rope_freq_base…）
│   ├── weights.rs      # LayerWeight 结构 + load_layers + get_f32_tensor
│   ├── forward.rs      # forward_step / run_shared_inference（核心前向循环）
│   ├── session.rs      # Session（含 KV cache 管理）
│   ├── scratch.rs      # trunk 专属 scratch buffer（仅当 core::scratchpad 不够用时存在）
│   └── tests.rs        # trunk 单元测试
│
├── [可选] asr/         # AudioEncoder → text tokens（前拼）
├── [可选] tts/         # text tokens → AudioDecoder（后拼）
└── [可选] vision/      # ImageEncoder → tokens（前拼）
```

### 2.1 `trunk/` 内部规则

* **必须包含 `mod.rs`**：禁止出现 `base.rs`，杜绝"新模型是不是该有 base.rs"的歧义。
* **必含子文件**：`config.rs`、`weights.rs`、`forward.rs`、`session.rs`、`tests.rs`。
* **`scratch.rs` 可选**：当 trunk 需要 core::scratchpad 之外的临时 buffer（如 shortconv state、SSM state）时单独存在；否则省略。
* **依赖方向**：`forward.rs` 与 `session.rs` 可互相调用；二者只能依赖 `config.rs`、`weights.rs`、`scratch.rs`，不可反向。
* **公开 API 路径稳定**：`mod.rs` 用 `pub use` 把 `Qwen3Model`/`Qwen3Session`/`forward_step` 等重新导出到 `models::qwen3::` 命名空间，调用方代码不动。

### 2.2 sibling 模块规则

* **与 `trunk/` 平级**，不是 `trunk/asr/`。
* **单向依赖**：sibling 可调用 trunk；trunk 不可调用 sibling。
* **职责清晰**：sibling 只负责自己的编码/解码，不重写 trunk 已有的 LayerWeight 结构。

---

## 3. 命名约定

| 旧名 | 新名 | 理由 |
|------|------|------|
| `base.rs` | `trunk/mod.rs`（+ `trunk/{config, weights, forward, session, tests}.rs`） | `base` 一词含糊；`trunk` 精准描述"LLM 解码器主干" |
| `skeleton.rs` | `trunk/weights.rs` | `skeleton` 历史上指"权重骨架"，与 `weights` 同义；统一用 `weights` |
| `text.rs` | `trunk/forward.rs` 或 `app/text.rs` | `text` 是入口概念，不属于模型内部 |

**特例**：现有 `qwen3/text.rs` 同时承担 CLI 入口与文本推理，按职责拆到 `app/text.rs`（CLI 入口）与 `qwen3/trunk/forward.rs`（推理循环）。

---

## 4. 各模型落地清单

| 模型 | trunk/ 内容 | sibling 模块 | 备注 |
|------|-------------|--------------|------|
| **llama** | trunk/{config, weights, forward, session, tests} | 无 | 纯文本，最简结构 |
| **lfm2** | 同上 + trunk/scratch.rs | 无 | 需 shortconv state buffer |
| **lfm25** | 同上 + trunk/scratch.rs | 无 | 同上 |
| **qwen3** | 同 llama + trunk/scratch.rs | `asr/`、`tts/` | 当前 `base.rs` 1971 行需拆分 |
| **qwen35** | 同 llama + trunk/positions.rs（RoPE 工具单独） | `vision/`（含 clip_config） | 当前已按职责拆分，仅需对齐命名 |
| **diffusion** | 不适用 | — | 不是 LLM，维持现有 `pig.rs / dit.rs / vae.rs` |

---

## 5. 迁移路线图（按风险/收益排序）

### Step 1 — 抽 `load_f32_tensor` 到 `core::loader`（风险：零）

**目标**：消除 5 处重复的 F32/BF16 norm 加载逻辑（qwen3/base.rs + 4 个 skeleton）。

**步骤**：
1. 在 `src/core/loader.rs` 新增 `pub fn load_f32_tensor(source, name, expected_dims) -> Result<Vec<f32>, String>`，签名与 qwen3/base.rs:1862 现有版本一致（接受 F32 或 BF16）。
2. 删除 llama/skeleton.rs:23、lfm2/skeleton.rs:64、lfm25/skeleton.rs:38 三个 `get_f32_tensor` 中的 F32/BF16 分支，改为调用 `core::loader::load_f32_tensor`。
3. qwen3/base.rs:1862 改为 re-export 或直接调用核心版本。
4. `cargo build --lib` + `cargo test --lib ops::float::tests core::tensor::tests` 验证零回归。

**收益**：立刻消除本次 BF16 改动时暴露的 5 处重复，未来再加新类型（如 FP8）只需改 1 处。

### Step 2 — 重命名 llama/lfm2/lfm25（风险：低）

**目标**：把 `{base, skeleton}.rs` 改为 `trunk/{forward, weights}.rs`，加 `trunk/mod.rs`。

**步骤**：
1. 对每个模型：`base.rs` → `trunk/forward.rs` + `trunk/session.rs` + `trunk/config.rs`；`skeleton.rs` → `trunk/weights.rs`。
2. 新建 `trunk/mod.rs` 做 `pub use` 转发。
3. `mod.rs` 在 `pub use trunk::*;` 之外保留原 `LlamaModel`/`Lfm2Model` 等高层导出，调用方代码零改动。
4. `cargo build --lib` + 跑 Qwen3-Q8 推理验证文本路径未破坏。

### Step 3 — 拆分 qwen3/base.rs（风险：中）

**目标**：1971 行单文件拆到 `trunk/{config, weights, forward, session, scratch, tests}.rs`。

**步骤**：
1. 复制 `qwen3/base.rs:52 Qwen3Config` → `trunk/config.rs`。
2. 复制 `qwen3/skeleton.rs` 整文件 → `trunk/weights.rs`，改名 `get_f32_tensor` 为 `weights::get_f32_tensor`。
3. 复制 `qwen3/base.rs:243 Qwen3Model` 与 `:254 Qwen3Session` 的字段定义 → `trunk/mod.rs`（保留 `pub use`）。
4. 复制 `:1601 run_shared_inference` + `:1691 forward_step` → `trunk/forward.rs`。
5. 复制 `:1823 static_q8_matrix` / `:1839 static_q8_tensor` / `:1850 static_tensor` → `trunk/weights.rs`（它们本质是"静态权重"工具，与加载同源）。
6. 复制 `:1676 TestTensorSource` + `:1694 test_model` + trunk 相关测试 → `trunk/tests.rs`。
7. `qwen3/base.rs` 改为只做 `pub use trunk::*;` 与 ASR/TTS 跨模块 wiring。
8. 跑 `cargo test --lib` 与 Qwen3 BF16/Q8_0 双推理验证。

**收益**：单文件从 1971 行降到 < 800 行；qwen3::base::tests 死代码（TestTensorSource/test_model 标记为 `#[warn(dead_code)]`）自然消亡；前几轮 `cargo test --lib` 看到的 9 个 `src/models/qwen3/base.rs` warning 消失。

### Step 4 — qwen35 命名对齐（风险：低）

**目标**：把 `forward.rs`/`loader.rs`/`session.rs`/`scratchpad.rs` 收纳到 `trunk/`，vision 单独保留 `vision/`。

**步骤**：
1. `qwen35/forward.rs` → `trunk/forward.rs`；`loader.rs` → `trunk/weights.rs`；`session.rs` → `trunk/session.rs`；`scratchpad.rs` → `trunk/scratch.rs`；`positions.rs` 保留（RoPE 工具）；`util.rs` → `trunk/util.rs`；`tests.rs` → `trunk/tests.rs`。
2. `vision.rs` 移到 `vision/forward.rs` + `vision/weights.rs`；`clip_config.rs` → `vision/clip_config.rs`。
3. `qwen35/mod.rs` 改为 `pub use trunk::*; pub use vision::*;`。
4. `cargo build --lib` + 跑 Qwen3.5 推理（若有本地模型）。

### Step 5 — 文档与基线刷新（风险：零）

1. 更新本文件末尾的"已完成"清单。
2. 更新 `docs/ARCHITECTURE.md` 的 "Key Files" 段。
3. 删除已迁移的旧文件路径引用。

---

## 6. 单一事实源原则（贯穿所有 step）

跨模型共性的工具函数**必须**先抽到 `core/`，禁止在 model 内重复实现。当前已知的共享工具：

| 函数 | 当前所在 | 应去 |
|------|----------|------|
| `load_f32_tensor`（F32/BF16 norm 加载） | qwen3/base.rs:1862、llama/skeleton.rs:23、lfm2/skeleton.rs:64、lfm25/skeleton.rs:38 | `core::loader::load_f32_tensor` |
| `static_q8_matrix` / `static_q8_tensor` / `static_tensor` | qwen3/base.rs:1823-1860 | `core::loader`（Step 3 顺带） |
| `expect_supported_embedding` / `is_supported_embedding` | `ops/embedding.rs`（已统一） | — ✅ |
| `SUPPORTED_EMBEDDING_TYPES` | `ops/embedding.rs`（已统一） | — ✅ |

未来新增任何跨模型共性工具，按同样模式处理：**先入 `core/`，再被模型调用，不允许模型内部重新实现等价逻辑**。

---

## 7. 验收标准

每个 step 完成后必须满足：

1. `cargo build --lib` 零 error、warning 数不增。
2. `cargo test --lib` 通过数不减少（基线 335 passed / 29 failed；29 个 failed 均为与本规范无关的预存在失败）。
3. 至少 1 个模型推理冒烟测试：
   - Q8_0：`cargo run --release --bin rust-model-inference -- --model models/qwen3-0.6b-gguf/Qwen3-0.6B-Q8_0.gguf --prompt "The capital of France is" --max-tokens 12 --temp 0 --threads 4` → 期望 `**Paris**.`
   - BF16：同上但用 `Qwen3-0.6B-BF16.gguf`。
4. 本文件末尾"已完成"清单追加本次 step。
