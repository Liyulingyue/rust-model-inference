# GLiNER 系列适配 TODO

fastino 的 GLiNER 家族在 ModelScope 共 11 个公开 repo。已通过 `modelscope list --owner fastino --repo-type model --all --page-size 50` 确认无遗漏。

## ⚠️ 重大发现（推翻此前假设）

`custom_tag:gliner2` 标签只代表"GLiNER 家族"，**不意味着同一架构**。下载 config 之后实际拆出 **3 个互不兼容的架构族**，每条 TODO 都要先认清自己在哪个族再决定能不能复用现有代码：

| 族 | 架构 | encoder | tokenizer | 适配状态 |
|---|---|---|---|---|
| **SpanExtractor (pre-2.5)** | `SpanExtractor`（无 `architecture` 字段，tensor 命名同 Decide） | DeBERTa-v3 | `DebertaV2Tokenizer` (SPM) | 没适配，权重可能跟 Decide 不同 |
| **SpanExtractor (2.5 / 1B)** | `architecture="span"` | Ettin 1B（decoder-only-派生的 encoder） | **BPE ByteLevel**（HF fast tokenizer） | 没适配 |
| **BoundaryExtractor (2.5)** | `architecture="boundary"` | DeBERTa-v3-base / mDeBERTa-v3-base | `DebertaV2Tokenizer` (SPM) | 没适配，**forward 完全不同于 Decide** |

含义：
- **同族内**才有可能复用现有 GLiNER2 Decide 代码（pre-2.5 v1 与 Decide 同族）。
- **跨族**是独立的大实现：每个新族都需要新 forward、新 GGUF 转换、新端到端 oracle。
- 命名"Decide"后缀不可靠：`GLiNER2.5-multi-Decide` 是 `boundary` 不是 `span`。真正决定能不能复用代码的是 `architecture` 字段（如果存在）和 encoder 名（`model_name`），不是 repo 名。

每条 TODO 都要在动手前解决 **flag 放置问题**（共用 `--gliner2-decide` 还是另开 flag）、**路由问题**（`/v1/jev/score` 还是另起 endpoint）、**模型层位置问题**（`src/models/gliner/{family}.rs` 还是全新 `src/models/{family}/`）。**默认假定**共用 `--gliner2-decide` + `/v1/jev/score`，除非验证发现该变体 head 形状/契约不一致。

---

## ✅ 已完成

### GLiNER2.5-Decide — DeBERTa-v3-large, 2 层 ReLU MLP, 族 = `span`
- 路径：`--gliner2-decide` → `Backend::Gliner2` → `/v1/jev/score`
- 转换器：`tools/converter/gliner/convert_gliner.py`（DeBERTa-v3-large 硬编码）
- 推理：`src/models/gliner/{compute,mod,prompt,weights}.rs`
- 端到端 oracle：6 case, max logit delta **7.2e-6**（threshold 1e-4）

---

## 🟡 SpanExtractor pre-2.5 族（同 Decide 架构，只是权重不同）

这族 **encoder + tokenizer 与 Decide 完全相同**，理论上能用现有 GGUF 转换器 + 推理代码重新吃一份。差异只剩权重。开工前需要验证 schema prompt 是否兼容（2.5 引入了 `<|context_start|>` 等新增 token，pre-2.5 v1 可能没有）。

### ✅ 1. `fastino/gliner2-large-v1` — DeBERTa-v3-large（已完成，含 byte-exact oracle）
- **架构族**：SpanExtractor（无 `architecture` 字段），`max_width=8`、`counting_layer=count_lstm`、`model_type=extractor`、DeBERTa-v3-large
- **flag 决策**：✅ 共用 `--gliner2-decide`
- **路由**：✅ `/v1/jev/score`
- **架构位置**：✅ `src/models/gliner/`（同目录同代码）—— 零新代码
- **改动**：
  - `tools/converter/gliner/convert_gliner.py`：把 `EXPECTED_CONFIG` 拆成严格（`model_name`）和可选（`architecture`/`config_version`/`token_pooling`），pre-2.5 缺字段也算合法
  - `tests/gliner2_large_v1.rs`：6 个 env-gated 烟测覆盖合同 / refund / 多任务 / examples / 长文本 / marker 位置
  - `tests/gliner2_large_v1_parity.rs`：6 个 case 的 byte-exact oracle，最大 logit delta **2.193e-5**（threshold 1e-4 余量 ~4.5×）
  - `tests/fixtures/gliner2-large-v1/classify-golden.json`：dump_golden.py 用 `transformers==4.48.1` + 同 microsoft/deberta-v3-large config 生成（与 Decide oracle 完全同一条 oracle 链）
- **commit**：`f6619df` (smoke) + 后续 oracle commit

### 2. `fastino/gliner2-base-v1` — DeBERTa-v3-base
- 同上但 encoder 维度更小（768 hidden / 12 layers / 12 heads）
- flag / 路由 / 位置同上
- **额外工作**：`tools/converter/gliner/convert_gliner.py` 当前硬编码 DeBERTa-v3-large（1024 / 24 / 16），需要读 config 动态化
- 推理层：`src/models/gliner/weights.rs` 同理需要动态化
- **价值**：base 尺寸跑得快很多（吞吐 ~3×），生产部署友好
- **前置**：先 #1 完成验证流程，再做 #2 的尺寸动态化

### 3. `fastino/gliner2-multi-v1` — 多语 base
- 同 #2 但训练数据换了，权重不同，tokenizer 可能是 mDeBERTa-v3（多语 SPM）
- flag / 路由 / 位置同上

---

## 🟠 SpanExtractor 1B 族（**完全不同**：Ettin + BPE）

### 4. `fastino/GLiNER2.5-Decide-1B` — Ettin-1B + BPE
- **架构族**：SpanExtractor（`architecture="span"`, `config_version=3`，同 Decide head 契约）
- **encoder**：`jhu-clsp/ettin-enc-from-dec-1b`
  - hidden=**1792**, 28 层, fused QKV `attn.Wqkv.weight (5376,1792)`, MLP `mlp.Wi (7680,1792)` + `mlp.Wo (1792,3840)` → **SwiGLU** + 中间维 3840
  - **RMSNorm**（`mlp_norm.weight`、`embeddings.norm.weight`、`final_norm.weight`）
  - 预 norm → fused QKV → output proj → add → 预 norm → SwiGLU → output proj → add（标准 LLaMA-style decoder 布局，**不是** DeBERTa）
- **tokenizer**：`tokenizer.json` 是 HF `TokenizersBackend`（`model.type=BPE`, `normalizer=NFC`, `pre_tokenizer=ByteLevel`），**不是** SentencePiece
  - vocab=50280 + 98 added = 50378（决定 embedding 行数）
  - 特殊 token ID：`[CLS]=50281 [SEP]=50282 [PAD]=50283 [MASK]=50284 [SEP_STRUCT]=50368 [SEP_TEXT]=50369 [P]=50370 [C]=50371 [E]=50372 [R]=50373 [L]=50374 [EXAMPLE]=50375 [OUTPUT]=50376 [DESCRIPTION]=50377`
- **classifier head 同 Decide**：2 层 ReLU MLP `classifier.0 (3584,1792)` + `classifier.2 (1,3584)` —— 3584 = 2 × 1792
- **flag 决策**：❓ 三个选项
  - (a) 共用 `--gliner2-decide`，因为同样 SpanExtractor 同样 schema。代价：encoder 实现 + tokenizer 实现都要新增
  - (b) 新增 `--gliner2-decide-1b`（明确型号）。代价：split 不可持续，每来一个新变体都加 flag
  - (c) `--gliner2-decide` 根据 GGUF metadata 自动适配（推荐），与 OpenAI 风格的"模型驱动路由"一致
- **路由**：✅ 共用 `/v1/jev/score`
- **架构位置**：❓ 二选一
  - (a) 新建 `src/models/gliner/encoder/ettin.rs` + `src/models/gliner/tokenizer/bpe.rs`，现有 `src/models/gliner/{compute,prompt,weights}.rs` 抽象出 `Encoder` / `Tokenizer` trait
  - (b) 全新模块 `src/models/gliner_ettin/`（彻底独立），简单但难维护
- **预计工作量**：~2000 行新代码（Ettin forward ~1500 + BPE tokenizer ~500 + GGUF 转换器 ~300 + 测试 ~500 + oracle）
- **前置**：先定 flag 决策（(a)/(b)/(c)），再开工

---

## 🔴 BoundaryExtractor 族（**完全不同的架构**）

这是 fastino 推出的**多任务**结构化预测架构，**不是** Decide 的"size variant"。一个 forward pass 同时输出：
- **Boundary detection**（start/end 候选，128 维 boundary head，4 head × 2 layer boundary attention，top-K=128）
- **Relation extraction**（pair biaffine，32 head × relation type）
- **Record extraction**（key-value，128 维 record head）
- **Count prediction**（count head + count embed LSTM）
- **Abstention**（abstention_threshold=0.5）
- **Span content**（content_dim=64）

输出维度 60+ config 字段，**不能**靠复用 Decide 代码做任何部分。

### 5. `fastino/gliner2.5-base-v1` — DeBERTa-v3-base + BoundaryExtractor
- **架构族**：BoundaryExtractor（`architecture="boundary"`），encoder=DeBERTa-v3-base，tokenizer=`DebertaV2Tokenizer` (SPM)
- **flag 决策**：❌ 不应共用 `--gliner2-decide`——语义完全不一样（输出 spans/relations/records，不是 label-set 评分）
- **建议 flag**：新增 `--gliner2-boundary`，路由到 `/v1/gliner2/boundary`（或 `/v1/jev/boundary`）
- **架构位置**：❌ 新模块 `src/models/gliner_boundary/`（与 Decide 同级但独立）
- **预计工作量**：~3000 行新代码（boundary head forward + relation head + record head + 多个 loss-style 推理路径 + GGUF 转换 + 端到端 oracle）
- **优先级**：中。生产价值高（一次推理多任务），但实现成本高

### 🟡 5.1 `fastino/gliner2.5-base-v1` GGUF 打包（已完成 converter + smoke）
- **改动**：
  - `tools/converter/gliner/convert_boundary.py`（独立脚本，处理 `architecture="boundary"` 模型，硬编码 DeBERTa-v3-base dims）：生成 `gliner2.5-base-v1-f32.gguf`，747 MB，包含 202 encoder tensors + 132 bundled heads（boundary_head 102 + relation_scorer 12 + record_decoder 18）
  - `tests/gliner2_5_base_v1_smoke.rs`：5 个 env-gated 测试覆盖 metadata / encoder dims / encoder tensor shapes / bundled heads / tokenizer
- **GGUF metadata 写入**：`gliner2.variant=boundary`、`gliner2.classifier.last_layer_index=3`（区别于 Decide 的 2）、`gliner2.boundary.bundled_heads`、`gliner2.boundary.bundled_tensor_count`
- **bundled heads 命名空间**：保留原始 safetensors key（`boundary_head.boundary_encoder.bos_state` 等），未来 BoundaryExtractor Rust forward 可直接消费，不需要再做 name map
- **classifier 层索引差异**：base-v1 用 `classifier.0` + `classifier.3`（中间 GeLU + dropout），Decide 用 `classifier.0` + `classifier.2`（中间 ReLU）。GGUF metadata 标记 `last_layer_index=3` 让 Rust loader 区分
- **验证**：5/5 通过。**不验证 byte-exact**：Rust 还没有 BoundaryExtractor forward
- **未做**（明确范围）：boundary detection forward、pair scoring、relation decoding、record decoding——这些需要 Rust 实现 ≥1000 行才能输出第一组 logits。参考实现 `target/gliner2-oracle/gliner2/models/boundary/` 有 8149 行 Python

### 🟡 5.2 BoundaryExtractor Rust forward（待开工）
- **范围**：实现 `boundary_head.boundary_proposer` + `boundary_head.pair_scorer` + `boundary_head.shared_pool_scorer` 三个子模块的 forward；relations/records/counts/abstention 各自一模块
- **工作量**：~1000 行最小可用 forward（仅 boundary detection），~3000 行完整 forward（含 relations + records + count + abstention）
- **byte-exact oracle**：与 Decide 同一套机制（GLiNER2 参考实现 + `transformers==4.48.1`）生成 golden fixture，但 boundary forward 涉及 top-K / sparse sampling / rotary position embeddings，复杂得多
- **本会话不做**：留给下次或独立分支

### 6. `fastino/gliner2.5-multi-v1` — mDeBERTa-v3-base + BoundaryExtractor
- 同 #5，但 encoder 换成多语 mDeBERTa-v3-base
- 工作量同 #5（同一族代码，encoder 切换即可）

### 7. `fastino/GLiNER2.5-multi-Decide` — mDeBERTa-v3-base + BoundaryExtractor
- 注意 repo 名有"Decide"但架构是 `boundary`！命名不可信
- 同 #5/#6 一族
- **优先级**：低——只是多语 base 版本的另一种权重，跟 #6 重复价值不大

### 8. `fastino/gliner2.5-small-v1`
- 没下载，估计 `boundary` 族 + DeBERTa-v3-small
- 同 #5 一族，encoder 换 small 即可

---

## 🔵 专项 / 守门员家族（用途特殊）

需要先看每个的 config 才能判定属于哪个族 + 用什么 head。

### 9. `fastino/GLiNER2-Guardrails-PII-Multi`
### 10. `fastino/gliner2-privacy-filter-PII-multi`
### 11. `fastino/gliguard-LLMGuardrails-300M`
- 专项 guardrail，没下载 config，无法判定架构
- 与 #5 不同的特殊用途（PII detection / prompt attack），可能走 span 也可能走 boundary
- 建议先做 #5 再回来看这些

---

## ⚪ 行业垂直（不属于 GLiNER 家族）

### 12. `fastino/Fastino-Nemotron-3.5-Lightning-Healthcare`
### 13. `fastino/Fastino-Nemotron-3.5-Lightning-Finance`
- **架构**：Nemotron3.5 + LoRA 行业微调，chat/instruct 模型，**不是** GLiNER 变体
- **TODO 范围之外**：本文件只追踪 GLiNER；Nemotron 应该走 `MODEL_LIST.md` 的 chat trunk 流程，与本 TODO 无关

---

## 通用架构问题（任何一项开工前必答）

1. **架构族识别**：每个变体开工前必须读 `config.json` 里的 `architecture` 字段（如果存在）和 `model_name`。SpanExtractor / BoundaryExtractor / 其他？同族内才复用代码。
2. **Encoder 尺寸动态化**：每族内的 size variant 需要：
   - 转换器：读 `config.json` 拿 `hidden_size / intermediate_size / num_hidden_layers / num_attention_heads`
   - 推理层：把 hardcode 替换成动态读
   - Oracle fixture：每个尺寸生成自己的 golden.jsonl
3. **Tokenizer 多样性**：每族内的 tokenizer 也可能不同（SPM / BPE / 未来的 UGM / WPM）。需要：
   - 抽象 `Tokenizer` trait
   - 具体实现 `SPMTokenizer`（已有）/ `BpeTokenizer`（需新增）
   - converter 根据 config 选 tokenizer，嵌入 GGUF
4. **GGUF 自描述**：转换器要在 metadata 写 `gliner2.architecture` / `gliner2.encoder_name` / `gliner2.encoder_dim` / `gliner2.tokenizer_type`，让 `from_source` 不依赖文件名判断
5. **flag 命名空间**：共用 `--gliner2-decide` vs 按型号分。建议**模型驱动路由**——同一个 flag，由 GGUF metadata 决定走哪个 encoder + head + tokenizer；新变体只需替换 GGUF 不改 CLI
6. **路由粒度**：所有 SpanExtractor 共用 `/v1/jev/score`；BoundaryExtractor 共用 `/v1/jev/boundary`；专项 guard 各自独立

---

## 执行顺序建议（**修订后**）

> 原顺序假设"size variant"是简单适配，已推翻。新顺序按"工作量 / 价值"重排。

| # | 任务 | 族 | 工作量 | 价值 | 优先级 |
|---|---|---|---|---|---|
| ~~A~~ | ~~架构 trait 化~~ | 全部 | ~~~1天~~ | ~~后续所有变体的前置~~ | **取消**：抽象应在真有重复时再做，避免过早抽象 |
| ✅ B | `gliner2-large-v1` Decide 验证 | span pre-2.5 | 小（~200行） | 中（兼容性证据） | 完成 |
| C | `gliner2-base-v1` + multi-v1 | span pre-2.5 | 中（~500行 + 尺寸动态化） | **高**（生产可用） | **下一条** |
| D | BoundaryExtractor 任一基线 | boundary | 大（~3000行） | 中（多任务） | 中 |
| E | Ettin encoder + BPE tokenizer | span 1B | **巨大**（~2000行） | 中（1B 升级） | 低 |
| F | 专项 guardrail（待定） | 待定 | 中 | 低 | 低 |

每条开工前必须先 `modelscope download --model <repo> config.json tokenizer_config.json`（按 model-download skill），读 config 填本文件对应 TODO 的"flag 决策 / 路由 / 架构位置"三栏，再写代码。**不再做"看着像就动手"的盲改**。

## B 完成的方法论选择

按用户原话"按照工作量，由少到多"执行：**不做架构抽象，先 B 再 C**。理由：
- B 已经证明 pre-2.5 同型可以直接复用现有转换器 + 推理代码（只放宽了 config 校验）
- C 的"尺寸动态化"在做完 B 之后成为可观察的具体改动（之前的猜想需要被验证/否定）
- 抽象在做 C 时如果真的"DeBERTa-v3-base 的所有循环展开都跟 large 一样"，才有现实基础。否则 base 可能要求不同的 matmul kernel、不同的 layer_norm_eps 等，那时候 trait 形状会被实际数据决定，比猜的准

---

## 已下载并清理的证据（**不存盘**，看这里就够了）

| Repo | 大小 | 关键 config 字段 | 状态 |
|---|---|---|---|
| `GLiNER2.5-Decide-1B` | 4.5GB (safetensors) | `architecture=span`, `model_name=jhu-clsp/ettin-enc-from-dec-1b`, tokenizer=ByteLevel BPE | 已删 |
| `gliner2.5-base-v1` | 8MB (config+tokenizer only) | `architecture=boundary`, `model_name=microsoft/deberta-v3-base`, tokenizer=DebertaV2Tokenizer (SPM) | 留 8MB config |
| `GLiNER2.5-multi-Decide` | 1.1GB | `architecture=boundary`, `model_name=microsoft/mdeberta-v3-base` | 已删 |
| `gliner2.5-multi-v1` | 1.1GB | `architecture=boundary`, `model_name=microsoft/mdeberta-v3-base` | 已删 |
| `gliner2-large-v1` | 226B (config only) | 无 architecture 字段, `model_name=microsoft/deberta-v3-large`, `model_type=extractor` | 已删 |

---

## 更新规则

- 每条 TODO 完成时，状态从 🟡/🟠/🔴/🔵 改成 ✅，commit 信息里写明改动了哪条 flag / 路由 / 模块
- 任何"flag 共用 vs 另开"的决策都要在本文件写理由 + commit hash，半年后回头看不会懵
- "已下载模型但未完成适配"的中间状态用 ⏸ 标记，列在对应 TODO 下方
- **族分类写错比不写更糟**——每条开工前必读 config 复核 `architecture` 与 `model_name`