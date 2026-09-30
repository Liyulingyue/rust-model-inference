# GLiNER 系列适配 TODO

fastino 的 GLiNER 家族在 ModelScope 共 11 个公开 repo。已通过 `modelscope list --owner fastino --repo-type model --all --page-size 50` 确认无遗漏。所有变体 `tags` 都命中 `custom_tag:gliner2` + `model_type:extractor` + `task:token-classification` + `library:safetensors` + `library:pytorch`，即同一架构族（`SpanExtractor`），但 encoder 尺寸 / head / 用途 / 是否带 "Decide" head 决定每条要走哪条路径。

每条 TODO 都要在动手前解决 **flag 放置问题**（共用 `--gliner2-decide` 还是另开 flag）、**路由问题**（`/v1/jev/score` 还是另起 endpoint）、**模型层位置问题**（`src/models/gliner/{size}.rs` vs 全新 `src/models/gliner2_extractor/`）。**默认假定**共用 `--gliner2-decide` + `/v1/jev/score`，除非验证发现该变体 head 形状/契约不一致。

---

## ✅ 已完成

### GLiNER2.5-Decide — DeBERTa-v3-large, 2 层 ReLU MLP
- 路径：`--gliner2-decide` → `Backend::Gliner2` → `/v1/jev/score`
- 转换器：`tools/converter/gliner/convert_gliner.py`（DeBERTa-v3-large 硬编码）
- 推理：`src/models/gliner/{compute,mod,prompt,weights}.rs`
- 端到端 oracle：6 case, max logit delta **7.2e-6**（threshold 1e-4）

---

## 🟡 Decide 变体（同 Decide 头；改 encoder）

### 1. `fastino/GLiNER2.5-Decide-1B`
- **预期**：DeBERTa-v3 升级到 ~1B（可能是 `microsoft/deberta-v3-XLarge` 或类似，需看 `config.json`）
- **flag 决策**：✅ 共用 `--gliner2-decide`，因为带 "Decide" 后缀
- **路由**：✅ `/v1/jev/score`
- **架构位置**：✅ `src/models/gliner/`，跟现有代码并列；encoder 尺寸通过 `from_source` 动态探测
- **验证**：需 `RMI_AUDIO8_GGUF`-style 环境门（`RMI_GLINER2_1B_GGUF`）+ 重新跑 oracle
- **风险**：1B encoder 单次推理内存显著上升；如果 hidden size 是 1536 而非 1024，要把 2 层 MLP 的 `intermediate=2048` 动态调整
- **前置**：先下载 `config.json` 确认 hidden_size / num_layers；hf 网络在仓库不可达，只能走 ModelScope（`modelscope download`）

### 2. `fastino/GLiNER2.5-multi-Decide`
- **预期**：DeBERTa-v3-large + multilingual corpus 训练
- **flag 决策**：✅ 共用 `--gliner2-decide`
- **路由**：✅ `/v1/jev/score`
- **架构位置**：✅ `src/models/gliner/`，同一文件即可
- **验证**：multi-Decide 的 schema prompt/label 描述在非英文输入下应该对——oracle 用中文 + 阿拉伯文 fixture
- **风险**：很低（架构与现有 Decide 一致，只是训练数据换了）；真正的风险是 SentencePiece vocab 是不是相同，下载后立刻比 `spm.model` 字节哈希

### 3. `fastino/gliner2.5-multi-v1`
- **预期**：可能 *不是* "Decide" 头，而是 SpanExtractor 原始 NER/分类
- **flag 决策**：❓ 需先看 `config.json` 的 `architecture_version`、`span_head`、`classifier` 是否存在——若没有 classifier 字段，走 span 抽取路径，要新建 `Span` 任务 flag，不应混进 `--gliner2-decide`
- **路由**：❓ 决定于 flag——span-extraction 没有"label set 评分"语义，要么暴露 token 类别要么拒收
- **架构位置**：❓ 如果走 span，可能要新模块 `src/models/gliner2_extractor/`（架构上是 Detectree v1 风格，不同语义）
- **前置**：必须先看 config

---

## 🟢 v1 尺寸家族（同 v1 头；改 encoder）

### 4. `fastino/gliner2.5-base-v1`
- **预期**：DeBERTa-v3-base（hidden=768, layers=12）—— 比 large 快 ~2-3×，内存 ~1/3
- **flag 决策**：❓ 同 #3，未确认有没有 Decide 头；带 `Decide` 后缀的 #1/#2 是明确的；不带后缀的 v1 可能只走 span
- **路由**：❓ 同 #3
- **架构位置**：❓ 同 #3（如果走 span）/ 与现有 `src/models/gliner/` 同目录（如果走 Decide）
- **价值**：base 对生产部署最有价值（吞吐/延迟），优先级最高

### 5. `fastino/gliner2.5-small-v1`
- **预期**：DeBERTa-v3-small（hidden=384/512, layers=6）
- **flag 决策 / 路由 / 位置**：❓ 同 #3/#4

### 6. `fastino/gliner2-large-v1` / `fastino/gliner2-base-v1` / `fastino/gliner2-multi-v1`
- **预期**：gliner2 上一代，可能 schema prompt 与 2.5 略有不同（2.5 引入了 `<|context_start|>` 等新增 token）
- **flag 决策**：❓ 仍然未确认是不是 Decide 头
- **路由 / 位置**：❓ 同上

---

## 🔵 专项 / 守门员家族（用途特殊）

### 7. `fastino/GLiNER2-Guardrails-PII-Multi`
- **预期**：训练目标是 PII 类别（name/email/phone/...），可能仍然是 Decide 头但 class 数很少（~10 个）
- **flag 决策**：❓ 三选一：
  - (a) 共用 `--gliner2-decide`（最简单，把 PII 类内置到 schema prompt 的 label 集）
  - (b) 新增 `--gliner2-pii` 暴露为更窄的 API（直接接受 text，返回 PII spans）
  - (c) 放到 adapter 模式（`adapters/gliner2_pii.rs`），CLI flag 与 (b) 相同
- **路由**：❓ 三选一：
  - (a) `/v1/jev/score`，强制 label set 限定到 PII
  - (b) 新建 `/v1/pii/detect`
- **架构位置**：❓ 同 Decide 头 → 复用 `src/models/gliner/`；若是 span head → 新模块
- **建议**：先看 config；如果带 Decide 头，做方案 (a)；否则做方案 (b) + 新模块

### 8. `fastino/gliner2-privacy-filter-PII-multi`
- **预期**：和 #7 极相似，但可能 fine-tune 目标不同（filter 而非 detect）
- **flag / 路由 / 位置**：❓ 同 #7，但可能可以跟 #7 共用同一份代码 + 不同训练数据

### 9. `fastino/gliguard-LLMGuardrails-300M`
- **预期**：guardrail 安全分类（prompt attack detection、hallucinated, etc.）
- **flag 决策**：❌ 不建议共用 `--gliner2-decide`——这是给 LLM 输出打分的"打分器"，与 label-set 评分语义不同
- **建议**：新建 `--gliner-guard` flag + 新路由 `/v1/guard/score` + 新模块 `src/models/gliner_guard/`

---

## ⚪ 行业垂直（不属于 GLiNER 家族）

### 10. `fastino/Fastino-Nemotron-3.5-Lightning-Healthcare`
### 11. `fastino/Fastino-Nemotron-3.5-Lightning-Finance`
- **架构**：Nemotron3.5 + LoRA 行业微调，是 chat/instruct 模型，**不是** GLiNER 变体
- **TODO 范围之外**：本文件只追踪 GLiNER；Nemotron 应该走 `MODEL_LIST.md` 的 chat trunk 流程，与本 TODO 无关

---

## 通用架构问题（任何一项开工前必答）

1. **SpanExtractor vs Decide**：每个 v1 变体的 `config.json` 里 `classifier` 字段是否存在？存在 → Decide 头；不存在 → span 头。要先看 config。
2. **Encoder 尺寸动态化**：现有 `convert_gliner.py` 和 `src/models/gliner/weights.rs` 写死了 DeBERTa-v3-large（1024/24/16）。所有尺寸变体都要：
   - 转换器：用 `config.json` 读 `hidden_size / intermediate_size / num_hidden_layers / num_attention_heads`
   - 推理层：把 hardcode 替换成动态读，`Weight::n_in/n_out` 已经动态，但 kernel 调用链要核
   - Oracle fixture：每个尺寸生成自己的 golden.jsonl
3. **GGUF 自描述**：转换器要在 metadata 里写 `gliner2.encoder_dim` / `gliner2.encoder_layers` / `gliner2.encoder_heads` / `gliner2.span_mode` / `gliner2.has_classifier`，让 `from_source` 不依赖文件名判断
4. **flag 命名空间**：共用 `--gliner2-decide` 还是按尺寸分 (`--gliner2-decide-large/base/small`)？建议共用，flag 不绑尺寸，模型尺寸由 GGUF metadata 决定；只在输出 `models/<filename>.md` 写明
5. **路由粒度**：所有 Decide 系列共用 `/v1/jev/score`；Guard 系列单独 `/v1/guard/score`；Span 系列（如果有）需要新路由或开放 token 类别返回

---

## 执行顺序建议

| # | Repo | 优先级 | 理由 |
|---|---|---|---|
| 1 | `GLiNER2.5-Decide-1B` | 高 | 直接复用代码，确认架构动态化假设 |
| 2 | `gliner2.5-base-v1` 或 `multi-v1` | 高 | 生产可用尺寸，先看 config 决定是 Decide 还是 span |
| 3 | `GLiNER2.5-multi-Decide` | 中 | 多语 Decide，价值高，代码改动最小 |
| 4 | `gliner2.5-small-v1` | 中 | 边端场景 |
| 5 | 上一代 v1 | 低 | 兼容历史用户 |
| 6 | PII / Privacy | 低 | 专项，等 base 落地再做 |
| 7 | gliguard | 低 | 跨架构，独立模块 |

每条开工前必须先 `modelscope download --model <repo> config.json tokenizer.json spm.model model.safetensors --local_dir ./<repo>`（遵循 `model-download` skill），读 config 填本文件对应 TODO 的"flag 决策 / 路由 / 架构位置"三栏，再写代码。

---

## 更新规则

- 每条 TODO 完成时，状态从 🟡/🟢/🔵/❓ 改成 ✅，commit 信息里写明改动了哪条 flag / 路由 / 模块
- 任何"flag 共用 vs 另开"的决策都要在本文件写理由 + commit hash，半年后回头看不会懵
- "已下载模型但未完成适配"的中间状态用 ⏸ 标记，列在对应 TODO 下方