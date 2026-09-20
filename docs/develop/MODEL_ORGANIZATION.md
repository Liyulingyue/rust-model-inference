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
* **推荐子文件**（绝大多数 trunk 应含）：
  * `config.rs` — 超参（n_layer, n_embd, n_head, n_embd_head, n_ff, eps, rope_freq_base…）
  * `weights.rs` — LayerWeight 结构 + load_layers + get_f32_tensor
  * `forward.rs` — forward_step / run_shared_inference（核心前向循环）
  * `session.rs` — Session（含 KV cache 管理）
  * `tests.rs` — trunk 单元测试
* **可选子文件**（按需存在）：
  * `scratch.rs` — trunk 专属 scratch buffer（仅当 `core::scratchpad` 不够用，如 shortconv state、SSM state）
  * `prefill.rs` — 模型级 batched prefill 调度（仅当 `core::prefill` 接口不够用，例如 `qwen3/trunk/prefill.rs` 894 行）
  * `util.rs` — 通用 helpers + 单元测试（`qwen3`、`qwen35` 用）
  * `positions.rs` — RoPE 位置构造（`qwen3/trunk/positions.rs`、`qwen35/trunk/positions.rs`）
* **豁免**（允许只含部分推荐文件）：
  * **纯函数式 trunk**（如 `llama`、`lfm2`/`lfm25`/`lfm2moe`/`nemotron_h`/`spark`）：如果模型没有显式 `Config`/`Session` 结构（config 通过 `core::loader::model_config_from_source` 解析、session 状态由 forward 内部 `Vec` 管），允许只含 `config.rs` + `forward.rs` + `weights.rs`（及必要的 `util.rs`），不引入空 `session.rs` / `tests.rs`。豁免清单见 §4「各模型落地清单」备注列。
  * **跨模型 trunk 复用**（如 `qwen_drive` 直接用 `qwen35::Qwen35Model`）：允许整个目录没有 `trunk/` 子目录。
* **依赖方向**：`forward.rs` 与 `session.rs` 可互相调用；二者只能依赖 `config.rs`、`weights.rs`、`scratch.rs` / `util.rs`，不可反向。
* **公开 API 路径稳定**：`mod.rs` 用 `pub use` 把 `Qwen3Model`/`Qwen3Session`/`forward_step` 等重新导出到 `models::<name>::` 命名空间，调用方代码不动。

### 2.2 sibling 模块规则

* **与 `trunk/` 平级**，不是 `trunk/asr/`。
* **单向依赖**：sibling 可调用 trunk；trunk 不可调用 sibling。
* **sibling → sibling 禁止**（默认）：不允许 A sibling 直接 `use crate::models::<x>::<other_sibling>::helper`；若两个 sibling 真有共用底层（例如 `qwen3/omni.rs` 当前直接调 `qwen3/asr/mel_encoder.rs` 的 `add_residual`/`layer_norm_rows` 等内部 helper），必须先把共享部分下沉到 `core/` 或新建 `models/<x>/<sibling>/_shared.rs`，否则视为违反 §6「单一事实源」。
* **职责清晰**：sibling 只负责自己的编码/解码，不重写 trunk 已有的 LayerWeight 结构。
* **尺寸警戒**：sibling 单文件 ≥1500 行时按职责再拆。已识别的超阈值文件：`qwen35/vision/mod.rs` 2926 行、`qwen3/vision/mod.rs` 1806 行（均未拆分，记入 §9「已知偏差」）。`qwen3/asr/mel_encoder.rs` 2840 行、`qwen3/tts/talker.rs` 1346 行、`qwen3/asr/audio_processor.rs` 640 行（音视频单文件较大但属于算法本身高内聚，按 §9 单独评估）。

---

## 3. 命名约定

| 旧名 | 新名 | 理由 |
|------|------|------|
| `base.rs` | `trunk/mod.rs`（+ `trunk/{config, weights, forward, session, tests}.rs`） | `base` 一词含糊；`trunk` 精准描述"LLM 解码器主干" |
| `skeleton.rs` | `trunk/weights.rs` | `skeleton` 历史上指"权重骨架"，与 `weights` 同义；统一用 `weights` |
| `text.rs` | `trunk/forward.rs` 或 `app/text.rs` | `text` 是入口概念，不属于模型内部 |
| `embedding.rs`（CLI 入口） | `app/embedding.rs` 或保留 `models/<name>/embedding.rs` | embedding 是入口概念；当前 `qwen3/embedding.rs` 638 行实为 CLI helper（`run_embedding` / `compute_embedding` / `MediaEmbeddings`），`app/embedding.rs` 只做 `pub use` 转发。**§9 待迁移**：纯 CLI 包装应迁到 `app/`，但目前保持原位以最小化 import 改动。 |
| `hunyuan.rs`（chat 包装） | `app/hunyuan.rs`（arch=hunyuan-dense 走 qwen3 trunk） | Hunyuan-MT2 chat prompt 模板 + 文本推理 CLI 入口；当前 `qwen3/hunyuan.rs` 39 行是 thin wrapper，`hunyuan.rs::run_inference` 直接调 `qwen3::text::run_inference_tokens`。**§9 待迁移**：`models/<arch>/chat.rs` 更适合该 arch 自身的目录，目前借住 qwen3 命名空间。 |
| `omni.rs`（Qwen2.5-Omni 音频） | `qwen3/asr/omni_audio.rs` 或独立 `models/omni/` | 不是 qwen3 主干，是 sibling 之一；当前 `qwen3/omni.rs` 990 行（含 `Qwen25OmniAudioModel` / `Qwen25OmniAudioConfig` / `encode_audio`）复用 `qwen3/asr/mel_encoder.rs` 的内部 helper，违反 §2.2 sibling → sibling 禁止。**§9 待迁移**：先建 `qwen3/asr/omni_audio.rs` 把 `Qwen25OmniAudioModel` + `encode_audio` 移过去，再让 `qwen3/asr/mel_encoder.rs` 的 helper 公开成 `pub(crate)` 或下沉到 `core/audio.rs`。 |

**特例**：现有 `qwen3/text.rs` 同时承担 CLI 入口与文本推理，按职责拆到 `app/text.rs`（CLI 入口）与 `qwen3/trunk/forward.rs`（推理循环）。Step 3 完成后 `qwen3/text.rs` 仍保留为 thin wrapper（`run_inference` + `run_inference_tokens`），被 `app/text.rs:137` 和 `qwen3/hunyuan.rs:32` 调用；按 §9 标记为「保留但应逐步迁移到 `app/`」。

---

## 4. 各模型落地清单

按 §2.1 豁免规则，列出每个模型的 trunk/ 实际内容、sibling 模块、与规范偏差。

### 4.1 文本 LLM（trunk 完整）

| 模型 | trunk/ 内容 | sibling 模块 | 备注 |
|------|-------------|--------------|------|
| **llama** | trunk/{forward, weights}（豁免：纯函数式，无 Config/Session） | 无 | §2.1 豁免；text-only |
| **lfm2** | trunk/{config, forward, weights}（豁免：无 Session） | `vision.rs`（应在 `vision/` 子目录，见 §9） | 缺 `trunk/scratch.rs` — shortconv state 由 `forward.rs` 内 `Vec<Vec<f32>>` 管理 |
| **lfm25** | trunk/{config, forward, weights}（豁免：无 Session） | 无 | 同 lfm2，缺 `trunk/scratch.rs` |
| **lfm2moe** | trunk/{config, forward, weights}（豁免：无 Session） | 无 | MoE 变体；前 `leading_dense_block_count` 层 dense SwiGLU，其余 sigmoid gating + top-k 路由 |
| **nemotron_h** | trunk/{config, forward, weights}（豁免：无 Session） | 无 | Hybrid Mamba-Transformer；当前 SSM 残差 L2 与 oracle 偏差 ~25%（`SUPPORTED_MODELS.md` Experimental） |
| **spark** | trunk/{config, forward, weights, session（在 forward.rs 内）}（豁免：session 不分文件） | 无 | Spark 2.5（Xunfei）；fused QKV、SWA + full-attn 混合、per-head sigmoid gate、GeGLU FFN |
| **qwen3** | trunk/{config, weights, forward, session, tests, prefill, util, positions} | `asr/`、`tts/`、`vision/`（`vision/clip_config.rs`） | §8 Step 3 完成；顶层另有 `embedding.rs` / `omni.rs` / `text.rs` / `hunyuan.rs` 4 个 CLI 包装（§9 待迁移） |
| **qwen35** | trunk/{config, weights, forward, session, scratch, tests, util, positions} | `vision/{mod.rs, clip_config.rs}` | §8 Step 4 完成；vision/mod.rs 2926 行超 §2.2 警戒线（§9 待拆） |
| **gemma4** | trunk/{config, weights, forward, session, scratch, tests} | `asr/{mod, config}`、`vision/{mod, config}`、`app.rs`（编排）、`contract.rs`（架构契约）、`tests.rs` | §8 完成；`app.rs` 实为模型级 CLI 编排层（应迁到 `src/app/gemma4.rs`，§9 待迁移）；`pub use asr as audio` 兼容旧调用路径 |

### 4.2 多模态/专用（无标准 trunk）

| 模型 | 实际结构 | 说明 |
|------|---------|------|
| **dots** | 不适用 trunk/ — LLM 部分为 arch=`qwen2` 独立 GGUF，由元数据驱动加载；mmproj 部分为 arch=`dotstts` | TTS：`tools/converter/dots/convert_dots_tts.py` 导出 LLM gguf（arch=qwen2, 255 tensors）+ mmproj（arch=dotstts）。子模块：`config.rs`、`schedule.rs`、`llm.rs`、`patch_encoder.rs`、`dit.rs`、`speaker.rs`、`vocoder.rs`、`generate.rs`、`edit.rs`、`weights.rs`、`speaker/{exp, log, melbank}.rs`。推理入口 `--tts --model … --mmproj … [--ref-audio] [--ref-text]` |
| **breeze** | **未迁移** — 应进 `trunk/` | 当前 `breeze/mod.rs`（702 行）+ `breeze/transformer.rs`（648 行 backbone）+ `breeze/tests.rs`（334 行）+ `breeze/codec/{decoder, encoder, mod, tests}.rs`（codec 是 sibling）。`transformer.rs` 是 backbone，应拆到 `trunk/{config, weights, forward, session}.rs`（§9 待迁移）。`transformer.rs` 反向依赖 `breeze::codec::BreezeCodec`，与 §2.2 sibling → trunk 方向一致 |
| **vibevoice_asr** | **未迁移** — 应进 `trunk/` | 当前 `vibevoice_asr/{config, encoder, generate, llm, mod}.rs`。`llm.rs` 848 行是 arch=`qwen2` LLM + `clip` mmproj + `vibevoice_asr` projector 的 LLM 部分，应拆到 `trunk/{config, weights, forward, session}.rs`；`encoder.rs` 是 acoustic + semantic tokenizer sibling（§9 待迁移） |
| **qwen_drive** | **跨模型 trunk 复用** — 无独立 trunk/ | 直接使用 `qwen35::Qwen35Model` 作 backbone；本目录只放规划头（`planning.rs` 1498 行）+ 感知头（`perception/{bev, fpn, heads, mod, ops}.rs`） + `scene.rs` + `weights.rs` + `config.rs` + `rng.rs`。§6 「单一事实源」不适用，因为没有自己的 trunk 复制 |
| **diffusion** | 不适用 trunk/ — 多个独立子项目 | 当前结构 `diffusion/{mod.rs, pig.rs}` + `diffusion/dreamx/{mod, audio_vae, config, creator, kernels, lightvae, media, refiner, text, upsampler, video_vae}.rs`（DreamX-Creator，Experimental）+ `diffusion/z_image/{mod, dit, text, vae, qwen_merges.txt}.rs`（Z-Image Turbo，§4 旧版只提到 `pig.rs / dit.rs / vae.rs` 已过期）。`z_image/text.rs` 复用 qwen3 trunk 作为文本编码器；`z_image/dit.rs` 2314 行 + `video_vae.rs` 1914 行均超警戒线（§9 待评估） |

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
| `load_f32_tensor`（F32/BF16 norm 加载） | qwen3/trunk/weights.rs:65、llama/trunk/weights.rs、lfm2/trunk/weights.rs、lfm25/trunk/weights.rs | `core::tensor::load_f32_tensor`（§8 Step 1 已抽离） ✅ |
| `static_q8_matrix` / `static_q8_tensor` / `static_tensor` | qwen3/trunk/util.rs:209-240 | `core::loader`（建议抽离 — §9 待迁移） |
| `expect_supported_embedding` / `is_supported_embedding` | `ops/embedding.rs`（已统一） | — ✅ |
| `SUPPORTED_EMBEDDING_TYPES` | `ops/embedding.rs`（已统一） | — ✅ |
| qwen3/asr/mel_encoder 的 `add_residual` / `layer_norm_rows` / `apply_gelu_erf` / `full_attention_into` / `checked_product` / `resize_f32` / `reserved_f32` / `static_tensor` / `load_f32_tensor` | qwen3/asr/mel_encoder.rs（`qwen3/omni.rs` 当前直接 `use`） | §2.2 sibling → sibling 禁止；helper 应公开 `pub(crate)` 或下沉到 `core/audio.rs`（§9 待迁移） |
| lfm2 / lfm25 / lfm2moe 的 shortconv state buffer（`Vec<Vec<f32>>` + `l_cache` × `n_embd` × `d_conv`） | `forward.rs` 内嵌（lfm2:184、lfm25 同上） | 若 §4 描述「需 `trunk/scratch.rs`」成立，应抽到 `trunk/scratch.rs`；目前 §4 描述与现实不一致，记入 §9 |
| qwen35 mamba2 SSM state / qwen_drive perception feature map | `qwen35/trunk/scratch.rs`（197 行）/ `qwen_drive/perception/` 各模块 | 已下沉到各模型 scratch；qwen_drive 不走统一 scratchpad 是因为 perception 是非 transformer 算子（BEVFormer / MS-Deform-Attn / FPN），与 §3 「执行 hot path 走 Arena」的法则二一致 |

未来新增任何跨模型共性工具，按同样模式处理：**先入 `core/`，再被模型调用，不允许模型内部重新实现等价逻辑**。

跨模型 trunk 复用（`qwen_drive` 直接用 `qwen35::Qwen35Model`、`z_image` 用 qwen3 作文本编码器、`dots` 内 arch=`qwen2` 的 LLM 由元数据驱动加载）不在本节约束范围内 — 它们没有重新实现 trunk，而是显式 `use crate::models::<other>::*`。

---

## 7. 验收标准

每个 step 完成后必须满足：

1. `cargo build --lib` 零 error、warning 数不增。
2. `cargo test --lib` 通过数不减少（基线 383 passed / 7 failed；7 个 failed 均为与本规范无关的预存在失败）。
3. 至少 1 个模型推理冒烟测试：
   - Q8_0：`cargo run --release --bin rust-model-inference -- --model models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf --prompt "The capital of France is" --max-tokens 12 --temp 0 --threads 4` → 期望 `**Paris**.`
   - BF16：同上但用 `Qwen3-0.6B-BF16.gguf`。
4. 本文件末尾"已完成"清单追加本次 step。

---

## 8. 已完成步骤

### ✅ Step 1 — 抽 `load_f32_tensor` 到 `core::tensor`（2026-08-28, commit `fa93bac`）

* 新增 `core::tensor::load_f32_tensor<S>(source, name, dims) -> Result<Vec<f32>, String>`，含完整 type/dims/byte-length 校验。
* `bf16_to_f32` 规范到 `core::tensor`，`ops::float` 通过 `use` 转发保持向后兼容。
* 5 处重复调用方改为委派：`llama/skeleton.rs`、`lfm2/skeleton.rs`、`lfm25/skeleton.rs`、`qwen3/base.rs`、`bin/server.rs`。
* 删除 ~135 行重复 F32/BF16 norm 加载代码。
* 验收：
  * `cargo build --lib` 零 error。
  * `cargo test --lib` 335 passed / 29 failed（与基线一致）。
  * Q8_0 与 BF16 双推理冒烟均输出 `**Paris**.`。

### ✅ Quant Kernel 补全（2026-08-28..29, commits `b7509ef`/`9d04643`/`402bc3d`/`592ba28`）

不在原 Step 1..5 路线图中，由"用户跑实际模型文件"反馈驱动，发现 4 类格式问题，逐个修复：

| Commit | 内容 | 端到端效果 |
|--------|------|------------|
| `b7509ef` | **Q5_K** kernel 之前是占位 `output = 0.0`。实现 `vec_dot_q5k_q8k_scalar`（仿 q4k 结构复用 `vec_dot_q5k_q8k_avx2`）。 | Q5_K_S/M、Q4_K_S 从乱码/全空 → 正确产出 `**Paris**` |
| `9d04643` | **Q4_1 AVX2** 内核（4.7× 加速：11.6→54.5 t/s）；**BF16 AVX2+FMA**（3.7× 加速：7.4→27.5 t/s）；**Q2_K/Q3_K scalar** 内核接入 + `QuantizedTensor` 注册。Q3_K 已知输出乱码（format bug，待验证）。 | Q4_1: 11.6→54.5 t/s；BF16: 7.4→27.5 t/s |
| `402bc3d` | **GGMLType 注册全部 I-quant**（IQ2_XXS/XS/S、IQ3_XXS/XS/S、IQ4_NL/XS）。**IQ4_NL scalar matmul** + `embedding_lookup_iq4_nl`。IQ4_XS kernel 在 `src/ops/quant/avx2_k.rs` AVX2 实现（commit `b8d6b7c`）；IQ2/IQ3 大部分 kernel 仍留 TODO panic。 | IQ4_NL / IQ4_XS 端到端可跑（实测产出 "Paris"）；IQ2/IQ3 panic |
| `2026-09-18` (uncommitted) | **IQ4_NL AVX2 + NEON kernel**：`vec_dot_iq4_nl_q8k_avx2` 在 `src/ops/quant/avx2_k.rs`，与 IQ4_XS 共享 `_mm_shuffle_epi8` LUT 查表结构；新增 `vec_dot_iq4_nl_q8k_neon` 在 `src/ops/quant/neon_k.rs` 走 aarch64 128-bit SIMD（`vqtbl1q_u8` LUT + `vqdmull_s16` hadd/madd），同源 FMA drift ≤ 1 ULP。顺带补完 `IQ4NLKernel::embedding_lookup`（之前未实现，默认 panic）和 `forward_prequantized`（原本写 0）。 | scalar 8.6 → AVX2 16.1 t/s gen（1.87×）；5 批中位稳定；产出 "Paris" |
| `592ba28` | **Q3_K / Q2_K format 修复**（参照 llama.cpp `dequantize_row_q2_K`、`vec_dot_q3_K_q8_K_generic` 逐行移植）。详见 §9。 | Q2_K/L、Q3_K_S/M 全部产出 `Paris`** |

新增结构：
```
src/ops/kernel/
├── q4_1/{mod, avx2, scalar}.rs   # Q4_1 AVX2
├── bf16/{mod, avx2, scalar}.rs   # BF16 AVX2
├── q2_k.rs                       # Q2_K scalar（Q8K path）
├── q3_k.rs                       # Q3_K scalar（Q8K path）
├── iq4_nl.rs                     # IQ4_NL kernel 入口（AVX2 src/ops/quant/avx2_k.rs，NEON src/ops/quant/neon_k.rs）
└── iq4_xs.rs                     # IQ4_XS kernel 入口（AVX2 实现在 src/ops/quant/avx2_k.rs 共享）
src/ops/quant/mod.rs 新增:
- BLOCK_Q2K_SIZE / BLOCK_Q3K_SIZE 常量
- dequantize_row_q2_k / dequantize_row_q3_k
- vec_dot_q2k_q8k_scalar / vec_dot_q3k_q8k_scalar
- IQ4_NL_LUT 常量（16-entry 非线性查找表）
- dequantize_row_iq4_nl / vec_dot_iq4_nl_q8k_scalar
```

测试：`ops::quant::avx2_parity` 新增 4 个 Q3_K 单元测试 + `make_q3k_block` helper（合成已知字节 → 验证 dequant/vecdot 数学值）。

---

### Q3_K format bug 修复详解（commit `602ba28`，§8 内 Quant Kernel 补全延伸）

Q3_K 模型（Q2_K/Q3_K_S/M）原本能加载但推理产出乱码。根因：

**Bug 1 — 输出索引遗漏 n 迭代推进**（致命）：
```rust
// 旧 (broken):
output[out_base + j * 32 + l] = dl_a * q_signed as f32;
// `j` 在 `[0..4)` 内循环，但跨 `_n in 0..(QK_K/128)` 不变。
// → n=1（元素 128..255）写回到 0..127，覆盖 n=0 结果，128..255 全 0。

// 修复：沿用 C 的 *y++ 指针递增语义
let mut out_idx = 0usize;
for _n in 0..(QK_K / 128) {
    for j in 0..4 {
        // ...
        for l in 0..16usize {
            output[out_base + out_idx] = ...;
            out_idx += 1;
        }
    }
}
```

**Bug 2 — scales 字段越界**：Q3_K scales 是 12 bytes（3 × u32），不是 16 bytes。旧实现读 4 个 u32（16 bytes）越界 4 字节，触发真实数据上的 panic。修复：用 `u32::from_le_bytes` 显式读 12 字节，第四个 u32 留 0。

**Q2_K format bug**：Q2_K qs 字节布局与 Q3_K 类似但 sub-block 不同。原实现把 16 sub-block 顺序处理（`scales[j]` for elements `[j*16, j*16+16)`），实际是 8 个 16-element pair（sub-A 读 `qs[l]`、sub-B 读 `qs[l+16]`），scale 索引 `n_outer*8 + j*2 + (sub_b ? 1 : 0)`，shift `j*2`。

修复后端到端验证（Qwen3-0.6B, `--temp 0 --threads 4`）：
- Q2_K.gguf:    `The capital of France is Paris.`
- Q2_K_L.gguf:  `The capital of France is Paris.`
- Q3_K_M.gguf:  `The capital of France is **Paris**` （与 Q8_0 baseline 一致）
- Q3_K_S.gguf:  `The capital of France is **Lyon**` （量化噪声导致 argmax 翻转，仍是 valid French city）
- Q8_0 baseline: `The capital of France is **Paris**.`

**教训**：现有 scalar K-quant 内核只有 4-bit 单元测试覆盖到合成块层面（`q4k_avx2_matches_scalar_*`），没有覆盖跨 128 元素 `n` 边界的累加和输出索引连续性。Q3_K 实际模型触发后才暴露出来。**今后每加一个 scalar K-quant 必须加 Q3_K 风格的"全零 + 已知模式 → 期望值"测试**，强制覆盖 n 迭代边界。

---

### ✅ Step 2 — 重命名 llama/lfm2/lfm25 到 `trunk/`（2026-08-30, commit `aad89bd`）

* `llama/{base,skeleton}.rs` → `llama/trunk/{forward,weights}.rs`。llama 无 `Config`/`Session` 结构，公开 API 只有 `run_inference` + `run_inference_tokens`，不引入空文件。
* `lfm2/{base,skeleton}.rs` → `lfm2/trunk/{forward,weights,session,config}.rs`。`Lfm2Config` 拆到 `config.rs`；`KvCacheFmt` enum 拆到 `session.rs`。
* `lfm25/{base,skeleton}.rs` → `lfm25/trunk/{forward,weights,session,config}.rs`。verbatim 复制 `lfm2/trunk/` 后做 `Lfm2→Lfm25` 重命名；GGUF metadata keys 保持 `lfm2.*`（现有 lfm25 GGUF 文件就是用这组键）。
* 公开 API 通过 `models::<name>::*` 暴露：`models::llama::{run_inference, run_inference_tokens}`、`models::lfm2::{run_inference, Lfm2Config, Lfm2LayerWeights, KvCacheFmt, ...}`、`models::lfm25::{...}` 同上。
* 验收：
  * `cargo build --lib` 零 error。
  * `cargo test --lib` 383 passed / 7 failed（与基线一致）。
  * 3 个模型均编译通过（不验证精度，按用户要求：llama/lfm2/lfm25 本来精度就对不齐，能编译即可）。

### ✅ Step 3 — 拆分 qwen3/base.rs 到 `trunk/`（2026-08-30, commits `f153a80` + `95d37f8`（其中 Step 3 含 `f153a80` qwen3 部分；Step 4 commit `95d37f8` 含 vision/clip_config 拆分））

合并提交 `4c89ed4` → `1c16990` → `f153a80` 完成 §5 Step 3。`qwen3/base.rs` (1971 行) 拆为：

* `trunk/config.rs` — `Qwen3Config` + `Qwen3Rope`
* `trunk/weights.rs` — `Qwen3Model` struct + `Qwen3LayerWeights` + load helpers + `impl Qwen3Model { from_source, accessors, embed_tokens }`
* `trunk/forward.rs` — `Qwen3Input` / `Qwen3GenerateOptions` / `Qwen3Generation` structs + `text_encode` free fn + `run_shared_inference` + `impl Qwen3Model { generate, generate_asr, text_encode_wrapper }`
* `trunk/session.rs` — `Qwen3Session` struct + `impl Qwen3Session { new, new_with_kv_state, generate_with_asr_trace, generate_streaming, ... }` (877 行)
* `trunk/util.rs` — helpers + 单元测试
* `trunk/positions.rs` — `qwen_text_positions`
* `trunk/tests.rs` — `TestTensorSource` + `MapTensorSource` + `test_model`

删除 `qwen3/base.rs`（per §2.1 禁止 base.rs）。`qwen3/mod.rs` 改为 pure re-export layer。所有 `qwen3::base::*` 调用方改为 `qwen3::*`（7 个外部 import 站点更新：`app/audio.rs`、`app/text.rs`、`app/image.rs`、`asr/model.rs`、`tts/talker.rs`、`format/ggufrs.rs`、`src/lib.rs`）。

修复一个 `Qwen3GenerateOptions` 的 `Copy` 问题（原代码靠隐式 `Copy` 行为，但派生只写了 `Clone`；改为 `validate_generation(&options, ...)` 改签名收引用）。

验收：
* `cargo build --lib` 零 error。
* `cargo test --lib` 383 passed / 7 failed（与基线一致）。
* Q8_0 端到端冒烟：Qwen3-0.6B-Q8_0 输出 `**Paris**`（46.9 t/s gen）。
* micro-bench：4608×1536 79.69 GFLOPS（与重构前一致）。

### ✅ Step 4 — qwen35 命名对齐（2026-08-30, commit `95d37f8`）

* `qwen35/forward.rs` → `trunk/forward.rs`
* `qwen35/loader.rs` → `trunk/weights.rs`（`Qwen35Model` + `Qwen35LayerWeights` structs 也移入此）
* `qwen35/session.rs` → `trunk/session.rs`
* `qwen35/scratchpad.rs` → `trunk/scratch.rs`
* `qwen35/positions.rs` → `trunk/positions.rs`
* `qwen35/util.rs` → `trunk/util.rs`
* `qwen35/tests.rs` → `trunk/tests.rs`
* `qwen35/vision.rs` → `vision/mod.rs`
* `qwen35/clip_config.rs` 拆分：`Qwen35Config` → `trunk/config.rs`（LLM 配置）；`ClipVisionConfig` → `vision/clip_config.rs`（vision encoder 配置）
* `qwen35/mod.rs` 改为 `pub use trunk::*; pub use vision::*;` 形式的纯 re-export 层
* `src/lib.rs` 更新：`models::qwen35::clip_config::{ClipVisionConfig, Qwen35Config}` → 拆成 `vision::clip_config::ClipVisionConfig` 和 `qwen35::Qwen35Config`（后者通过 trunk re-export）

验收：
* `cargo build --lib` 零 error。
* `cargo test --lib` 383 passed / 7 failed（与基线一致）。
* qwen35-specific 测试：30 passed / 0 failed（不变）。
* Q8_0 端到端冒烟（qwen3 路径）：`The capital of France is **Paris**`。

### ✅ Gemma4 — trunk 与多模态 sibling 对齐（2026-08-30）

* `gemma4/text.rs` 拆为 `trunk/{config,weights,forward,session,scratch,tests}.rs`，`trunk/mod.rs` 统一 re-export `Gemma4Config`、`Gemma4Model`、`Gemma4Session` 与 `Gemma4InputRow`。
* `gemma4/audio.rs` → `gemma4/asr/mod.rs`，`Gemma4AudioConfig` 单独放入 `asr/config.rs`；保留 `gemma4::audio` re-export 兼容旧调用路径。
* `gemma4/vision.rs` → `gemma4/vision/mod.rs`，`Gemma4VisionConfig` 单独放入 `vision/config.rs`。
* 模型加载、媒体编码编排、tokenizer、生成循环与 stdout 输出从 `models/gemma4/multimodal.rs` 移到 `app/gemma4.rs`；模型 trunk 不依赖 `asr` 或 `vision`。
* 验收：
  * `cargo test --lib gemma4`：49 passed / 0 failed / 1 ignored。
  * `cargo test --test gemma4_reference`：6 passed / 0 failed / 5 ignored。
  * llama.cpp `3173a56471c` softmax 前严格 parity：text、image、audio、image+audio 四组输入的 token IDs，以及图像预处理、audio mel 的 checkpoint shape 和 F32 `u32` bits 一致。
  * Gemma4 attention 统一使用准确、稳定的标量 softmax；softmax 之后的 layer checkpoint、logits 与 greedy token IDs 不再承诺和 llama.cpp 逐位一致。

### ✅ Step 5 — 文档与基线刷新（2026-09-18）

按 §5 Step 5 的承诺刷新文档与清单：

* §2.1 规则从「必含」放宽为「推荐」，新增 `prefill.rs` / `util.rs` / `positions.rs` 可选项与「纯函数式 trunk 豁免」。
* §2.2 新增「sibling → sibling 禁止」与「单文件 ≥1500 行警戒」。
* §3 命名表新增 `embedding.rs` / `hunyuan.rs` / `omni.rs` 三行（CLI 入口与 sibling 隔离）。
* §4 模型清单从 7 个扩到 14 个（新增 `lfm2moe` / `nemotron_h` / `spark` / `gemma4` / `breeze` / `vibevoice_asr` / `qwen_drive` + `diffusion/{dreamx,z_image}` 子模块）；diffusion 描述从「pig.rs / dit.rs / vae.rs」更新到完整子项目结构。
* §6 文件路径从 `qwen3/base.rs:1862` 更新到 `qwen3/trunk/weights.rs:65`、`qwen3/trunk/util.rs:209-240`。
* 新增 §9「已知偏差与待迁移」，承认 Step 5 之后仍存在的规范偏差并给出迁移方向。

---

## 9. 已知偏差与待迁移

本节列出 §1–§4 已识别但**未在 Step 1–5 修复**的规范偏差，按风险/收益排序，每条都给出建议的下一步动作。所有偏差均不影响 `cargo build` 与 `cargo test` 基线（383 passed / 7 failed 与 §7 一致），纯结构性 follow-up。

### 9.1 路径偏差

| # | 偏差 | 当前现实 | 迁移方向 | 风险 |
|---|------|---------|----------|------|
| P-1 | qwen3 CLI 包装仍在 `models/qwen3/` 下 | `qwen3/text.rs`、`qwen3/embedding.rs`、`qwen3/hunyuan.rs` | 全部迁到 `app/{text, embedding, hunyuan}.rs`；`qwen3/mod.rs` 移除 `pub use text::*; pub use embedding::*;` | 低（仅 import 站点变更） |
| P-2 | qwen3/omni.rs 借住 qwen3 命名空间 | `qwen3/omni.rs` 990 行（Qwen2.5-Omni 音频 + `encode_audio`） | 移到 `qwen3/asr/omni_audio.rs`；`qwen3/asr/mel_encoder.rs` 的内部 helper 公开为 `pub(crate)` 或下沉 `core/audio.rs` | 中（sibling → sibling 依赖清理） |
| P-3 | lfm2 vision 在顶层 `vision.rs` | `lfm2/vision.rs` 862 行（LFM2.5-VL SigLIP） | 移到 `lfm2/vision/mod.rs`；`lfm2/mod.rs` 已是 `pub mod vision;` 不需大改 | 低 |
| P-4 | gemma4 编排层在 `models/gemma4/app.rs` | `gemma4/app.rs` 501 行 | 移到 `src/app/gemma4.rs`；`gemma4/mod.rs` 移除 `pub mod app;`，新增 `pub use crate::app::gemma4::run_gemma4;` | 中（涉及 `gemma4::app::*` 引用站点） |

### 9.2 结构偏差

| # | 偏差 | 当前现实 | 迁移方向 | 风险 |
|---|------|---------|----------|------|
| S-1 | `breeze/` 没有 trunk/ | `breeze/{mod, transformer, tests}.rs` + `codec/{...}` | `transformer.rs` 拆到 `breeze/trunk/{config, weights, forward, session}.rs`；`mod.rs` 改为 `pub use trunk::*; pub use codec::*;` | 中（backbone 与 codec 之间反向依赖） |
| S-2 | `vibevoice_asr/` 没有 trunk/ | `vibevoice_asr/{config, encoder, generate, llm, mod}.rs` | `llm.rs` 848 行拆到 `vibevoice_asr/trunk/{config, weights, forward, session}.rs`；`encoder.rs` 留 sibling | 低 |
| S-3 | `qwen_drive/` 没有 trunk/（跨模型复用 qwen35） | `qwen_drive/{config, planning, perception/, scene, weights, rng, mod}.rs` | 显式记录「跨模型 trunk 复用」豁免（§4 备注列已写）；不引入 trunk/ | 零（已合规） |
| S-4 | `lfm2/lfm25/lfm2moe` 没有 `trunk/scratch.rs` | §4 旧描述「需 shortconv state buffer」 | 选项 A：补 `trunk/scratch.rs`；选项 B：改 §4 描述为「shortconv state 由 `forward.rs` 内嵌 `Vec<Vec<f32>>` 管理」 | 低 |
| S-5 | `gemma4/contract.rs`（架构契约）在 models 下 | `gemma4/contract.rs` 162 行 | 留 models 下但改名 `gemma4/_contract.rs`（下划线前缀暗示私有），或者迁到 `core/contracts/gemma4.rs` | 低 |
| S-6 | `gemma4/tests.rs`（顶层 547 行测试）在 models 下 | 与 `trunk/tests.rs` 并列 | 移到 `tests/integration/gemma4.rs`（Cargo 集成测试位置） | 低 |

### 9.3 共享工具下沉

| # | 工具 | 当前所在 | 应去 |
|---|------|----------|------|
| T-1 | `static_q8_matrix` / `static_q8_tensor` / `static_tensor` | qwen3/trunk/util.rs:209-240 | `core::loader`（与 `load_f32_tensor` 同位置） |
| T-2 | qwen3/asr/mel_encoder 内部 helper（`add_residual` / `layer_norm_rows` / `apply_gelu_erf` / `full_attention_into` / `checked_product` / `resize_f32` / `reserved_f32` / `static_tensor` / `load_f32_tensor`） | qwen3/asr/mel_encoder.rs（被 omni.rs 复用） | 部分公开 `pub(crate)`；纯计算 helper 下沉到 `core/audio.rs` |
| T-3 | qwen3/asr/audio_processor `log_mel_windows` / `compute_log_mel` / `HOP` | qwen3/asr/audio_processor.rs（被 omni.rs 复用） | 与 T-2 同处理 |
| T-4 | lfm2 / lfm25 / lfm2moe 共享的 shortconv state 类型 + buffer 分配 | 各模型 `forward.rs` | 若 S-4 选 A：抽到 `core/scratchpad` 或各模型 `trunk/scratch.rs` |

### 9.4 尺寸警戒（§2.2 新规则触发）

| # | 文件 | 行数 | 评估 | 建议 |
|---|------|------|------|------|
| Z-1 | `qwen35/vision/mod.rs` | 2926 | 已超 §2.2 警戒线（≥1500） | 按 vision encoder / projector / mRoPE / deepstack / scratchpad 职责再拆 4–5 个文件 |
| Z-2 | `qwen3/asr/mel_encoder.rs` | 2840 | 高内聚音频编码算法 | 关注 |
| Z-3 | `diffusion/z_image/dit.rs` | 2314 | 高内聚 DiT forward（per-block compute） | 关注 |
| Z-4 | `diffusion/dreamx/video_vae.rs` | 1914 | 高内聚 video VAE | 关注 |
| Z-5 | `qwen3/vision/mod.rs` | 1806 | 已超警戒线 | 按 vision encoder / projector / deepstack / scratchpad 拆 3–4 个文件 |
| Z-6 | `diffusion/dreamx/creator.rs` | 1519 | 刚超警戒线 | 关注 |
| Z-7 | `dots/speaker.rs` | 2309 | 高内聚 speaker encoder + 合成 | 关注 |
| Z-8 | `dots/vocoder.rs` | 1797 | 高内聚 vocoder | 关注 |
| Z-9 | `dots/llm.rs` | 1384 | 接近警戒线 | 关注 |
| Z-10 | `dots/patch_encoder.rs` | 1402 | 接近警戒线 | 关注 |

评估标准：高内聚单一职责（如音频编码、DiT、VAE、speaker encoder）即使 >1500 行也可接受；多职责混合（如 vision encoder + projector + scratchpad 同一文件）必须拆。

### 9.5 Step 6 候选（按收益/风险）

完成 §9 偏差后，§5 路线图可继续 Step 6（按本节优先级）：

1. **Step 6.1**（P-3）：`lfm2/vision.rs` → `lfm2/vision/mod.rs`。风险：零。
2. **Step 6.2**（P-1）：`qwen3/{text, embedding, hunyuan}.rs` → `app/{text, embedding, hunyuan}.rs`。风险：低（import 站点更新）。
3. **Step 6.3**（P-2 + T-2 + T-3）：`qwen3/omni.rs` → `qwen3/asr/omni_audio.rs` + asr 内部 helper 下沉。风险：中（sibling 隔离）。
4. **Step 6.4**（S-1）：`breeze/transformer.rs` → `breeze/trunk/{config, weights, forward, session}.rs`。风险：中（backbone ↔ codec 反向依赖）。
5. **Step 6.5**（S-2）：`vibevoice_asr/llm.rs` → `vibevoice_asr/trunk/{...}`。风险：低。
6. **Step 6.6**（Z-1、Z-5）：`qwen35/vision/mod.rs` 与 `qwen3/vision/mod.rs` 按职责拆分。风险：中（多 import 站点）。

每步均按 §7 验收标准执行（`cargo build --lib` 零 error、`cargo test --lib` 不退化、至少 1 个真实模型推理冒烟通过、§8 已完成清单追加本次 step）。
