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
- **GGUF metadata 写入**：`gliner2.variant=boundary`、`gliner2.classifier.last_layer_index=3`（区别于 Decide 的 2）、`gliner2.boundary.bundled_heads`、`gliner2.boundary.bundled_tensor_count`，以及 `config.json` 里 `boundary_head` 整块转写出的 `gliner2.boundary.*` flag / dim / 温度 / 阈值（`boundary_dim` / `pair_dim` / `content_dim` / `use_inside_evidence` / `enable_span_content` / `content_soft_max_pool` / `query_conditioned_inside_weight` / `endpoint_difference_features` / `enable_rotary_endpoints` / `multihead_pair_compat_heads` / `rotary_base` / `candidate_pool` / 各 head 的 enable 与阈值）
- **bundled heads 命名空间**：保留原始 safetensors key（`boundary_head.boundary_encoder.bos_state` 等），未来 BoundaryExtractor Rust forward 可直接消费，不需要再做 name map
- **classifier 层索引差异**：base-v1 用 `classifier.0` + `classifier.3`（中间 GeLU + dropout），Decide 用 `classifier.0` + `classifier.2`（中间 ReLU）。GGUF metadata 标记 `last_layer_index=3` 让 Rust loader 区分
- **验证**：5/5 通过。**不验证 byte-exact**：Rust 还没有 BoundaryExtractor forward
- **未做**（明确范围）：boundary detection forward、pair scoring、relation decoding、record decoding——这些需要 Rust 实现 ≥1000 行才能输出第一组 logits。参考实现 `target/gliner2-oracle/gliner2/models/boundary/` 有 8149 行 Python

### ✅ 5.2.1 BoundaryEncoder.forward + byte-exact oracle（commit `d725de8` + `b678b95`）
- **范围**：`gliner2.models.boundary.encoding.BoundaryEncoder.forward` 的 Rust 端口（encoding.py:185-205）
- **实现**：
  - `src/models/gliner_boundary/loader.rs`：`BoundaryModel::from_source`，复用 `gliner::compute::encode`（DeBERTa forward），独立加载 classifier.0+3（GeLU 而非 ReLU）
  - `src/models/gliner_boundary/forward.rs`：shift left/right + 左/右 projection + concat + output_projection + LayerNorm + 2 attention blocks + 1 refinement block + mask
- **GGUF converter fix**：`convert_boundary.py` 的 vocab_size 计算少算 [MASK]（在 SPM vocab 内 id=128000），修了 `base = len(pieces) + 1`
- **byte-exact oracle**：通过 `tools/oracle/gliner_boundary/dump_boundary_encoder.py` 生成 golden，Rust 端 `tests/gliner2_5_base_v1_boundary_encoder_parity.rs` 比对
- **max delta**：**1.907e-6**（release-fast），与 Decide 的 7.2e-6 / large-v1 的 2.193e-5 同量级（F32 累加顺序差）
- **发现一个真 bug**：softmax 缺 validity mask。oracle 首次跑出 0.224 delta，加 mask + 对角线 OR 后通过
- **commit**：`d725de8` (loader + forward + smoke) + `b678b95` (oracle + fixture)

### ✅ 5.2.2a BoundaryProposer.score_explicit_pairs + PairScorer（全 feature，byte-exact）
- **范围**：`SparseBoundaryProposer.score_explicit_pairs` + `SparseBoundaryPairScorer.forward`（scoring.py:177）完整 feature 集合
- **实现**：
  - `proposer.rs`：`RotaryBoundaryEmbedding` + `score_explicit_pairs`（marginal-free compat prior），max delta **2.384e-7**（`6eafe4e`）
  - `marginals.rs`：`BoundaryQueryHead.forward`（start/end marginals + inside prefix），max delta start 3.338e-6 / end 1.431e-6 / prefix 2.861e-6（`f3d6643`）
  - `content_pooler.rs`（新）：`SpanContentPooler`，prefix-sum 均值池化 + LayerNorm
  - `pair_scorer.rs`：`SparseBoundaryPairScorer` 全 feature——endpoint compat、endpoint difference、start/end marginal、prior、span content（pooler + query projection + bias）、inside evidence（query-conditioned weight）、length features、`MASK_LOGIT`
  - `spans.rs`：`score_spans` 组合 API
- **feature flag 从 GGUF metadata 读**：`convert_boundary.py` 现在把 `config.json` 的 `boundary_head` 整块转写进 `gliner2.boundary.*`（flag / dim / 温度 / 阈值），并在转换时用 `check_pair_scorer_shapes` 交叉校验 tensor shape 与 setting 是否一致。Rust loader 缺 metadata 直接报错要求重转，不再猜默认值——之前 limited 版本就是靠默认值蒙混过关的
- **byte-exact oracle**：`tools/oracle/gliner_boundary/dump_score_explicit_spans_full.py` 直接从 checkpoint 的 config + safetensors 构造 reference `BoundaryHead`（`load_state_dict(strict=True)`）并调 `BoundaryHead.score_explicit_spans`，flag 全部来自 checkpoint
- **max delta**：**5.722e-6**
- **抓到的两个真 bug**：
  1. `endpoint_difference_projection` 的输入宽度是 `2 * pair_dim`（`cat(s-e, |s-e|)`），Rust 端按 `pair_dim` 取行，读取了拼接向量的前半段
  2. `content_pooler.build_prefix` 的原地 cumsum 写成了**倒序**；倒序时 row j-1 还没累加完，row j 只拿到最后两项。这个 bug 只在 span 不从 0 起始时暴露，delta 只有 ~0.2~1.3，非常容易误判成 F32 噪声
- **commit**：`1d98e37`（score_spans）+ 本次（全 feature + metadata 转写）
- **删除**：`dump_pair_scorer_limited.py` / `pair_scorer_limited_parity.rs` / `pair-scorer-limited-golden.json`。limited config 不是任何已发布 checkpoint 的真实配置，留着只会诱导「默认值够用」的错觉

### ✅ 5.2.2b 文档级推理路径：DocumentCandidatePool + SharedPoolScorer（主线）
- **关键发现**：`gliner2.5-base-v1` 的 `candidate_pool = "shared"`，所以**普通推理根本不走 `SparseBoundaryPairScorer`**
  - `model.py:396-466`：`DocumentCandidatePool`（`pool.py:107`）建候选 + `SharedPoolScorer`（`pool.py:446`）打分，然后 `pair_logits = pooled_logits.transpose(1, 2)`
  - `engine.py:78` 调的是 `self.boundary_head(...)`（即 `BoundaryHead.forward`），它按 `candidate_pool` 分支——所以这条就是推理主线
  - `SparseBoundaryPairScorer` 只在 `score_explicit_spans` 里被调用，也就是：entity 分类（`engine.py:692`，`choice_pairs = [(i, i+1)]`）、entity 属性（`engine.py:435`）、joint-IE。5.2.2a 做的是这条线
- **实现**：
  - `settings.rs`（新）：`gliner2.boundary.*` metadata → 单一 `BoundarySettings`，所有 head 的 feature flag / dim 只有一个来源
  - `pool.rs`（新）：`DocumentCandidatePool`（query union → top-k 边界 → 笛卡尔配对 → per-query quota → 去重截断）+ `SharedPoolScorer`（candidate 向量 → query dot + FiLM → marginals + inside evidence）
  - `spans.rs`：`score_document_candidates()` 顶层 API，返回 public `[B,Q,C]` 顺序（`to_candidate_batch` 的转置）
- **oracle**：
  - `dump_document_candidate_pool.py`：max proposal delta **2.622e-6**（3 cases，最长 24 token / 625 笛卡尔对 / 满 192 pool）
  - `dump_shared_pool_scorer.py`：max pair-logit delta **1.526e-5**；同时比对 `candidate` 采样行
  - 测试按**顺序**而非只比分数断言：pool 是离散算法，tie-break 不同就会选出不同 span，那种错看起来像数值噪声
- **抓到的三个真 bug**：
  1. **`apply_linear_full` 不能用于多行**。它从 `input.len()` 推 `n_in`，喂给它整个 `[rows, in]` 块会把块当成一个宽向量、按错误 stride 读权重行。已加 `apply_linear_rows` 并要求显式传 `n_out`
  2. **`Weight::n_out` 在 F32 上无意义**：`QuantizedTensor::n_rows()` 对 F32 返回 `usize::from(!data.is_empty())`，即 0 或 1。现有调用点全靠 slice 长度推形状，所以只在单行投影下侥幸正确
  3. **`film(query).chunk(2, -1)` 的 beta 偏移是 `qi * 2 * pair + pair`**，我先写成 `qi * 3 * pair + pair`，`qi=1` 时直接越界
- **`boundary_attention_window` 一直没被实现**（base-v1 = 128）。原代码注释写「metadata-only for now」，因为所有 fixture 都 <= 24 token，`|i-j| <= 128` 全真，删掉窗口也照样 byte-exact。已修 + 新增 `dump_boundary_attention_window.py`（n = 8 / 273，逐行记录 allowed key 集合）专门堵这个洞
- **commit**：`7fbc0d2` 之后的 boundary pool commit

### 🟡 5.2.3 relations + records + count + abstention（待开工）
- **范围**：relation_scorer（head/tail projection + biaffine + mlp）、record_decoder（candidate/field/instance projection + key/value attention）、count_head（scalar projection）、null_projection（scalar projection）
- **工作量**：~1000 行 Rust
- **byte-exact oracle**：与 5.2.1 同套机制，每个子模块单独 fixture
- **价值**：生产用途（一次推理多任务）；学术上 boundary + relation 是最常用的

### ✅ 5.2.4a 真实入口 + CLI flag（已可运行）
- **范围**：tokenizer → prompt builder（`[E]` marker）→ `compute::encode` → word/marker gather → `score_document_candidates` → `sigmoid` + threshold + 解码，加上 `--gliner2-boundary`
- **实现**：
  - `prompt.rs`：`build_boundary_prompt_with`（与 Decide 共用 token 流，只多返回 word 路由和 query 路由；`EncodedPrompt` 加了 `words` / `text_word_first_positions` / `query_positions` / `query_names`）
  - `extract.rs`（新）：`run_extraction` / `extract_spans` / `decode_spans` / `gather_states`，span 偏移是**词**下标
  - `adapters/gliner2_boundary.rs`：`parse_boundary_schema`（reference 的 `{"entities": [...]}` / `{"entities": {...}}` + `entity_descriptions`）+ CLI 输出
- **端到端 oracle**：`dump_extract_spans_end_to_end.py` 是本目录**第一个不从 synthetic `text_states` 起步**的 oracle。它跑 reference `SchemaTransformer` → **独立的** `transformers` DeBERTa-v3-base（权重取自 checkpoint 的 `encoder.*`，即微调后的）→ reference `BoundaryHead` → `decode_candidates`
  - max pair-logit delta **2.813e-5**，span 与 reference 完全一致
  - `input_ids` / word 路由 / query 路由**逐位相等**
- **三个只有真正跑起来才暴露的坑**：
  1. **encoder 必须用微调后的权重**。reference `from_pretrained` 会用 checkpoint 的 `encoder.*` 覆盖 `microsoft/deberta-v3-base`；用原始权重时所有 pair logit 都在 -15 附近，模型什么都抽不出来——而且**不会报错**，只是结果为空
  2. **预处理入口选错**。`transform_and_format` 看起来是"main preprocessing entry point"，但它**不**调 `_normalize_text`；真实推理走 `collate_fn_inference` → `_collate_batch`，会补句末 `.`。少一个词，后面所有下标全错
  3. **classification 的 schema parser 不能复用**。`parse_schema` 会把 `entity_descriptions` 当成第二个 task，query 数量翻倍且一半是描述
- **实测**（`--gliner2-boundary`，本机 1.8s/条）：
  - "Ada Lovelace worked with Charles Babbage in London." → person: `ada lovelace` / `charles babbage`，location: `london`
  - "Marie Curie moved to Paris and later to the Curie Institute." → person: `marie curie`，organization: `curie institute`，location: `paris`
  - "Apple Inc. ... Tim Cook ... Cupertino ..." → person: `tim cook`，organization: `apple inc .`，location: `cupertino`
  - 负例 "nothing here should extract cleanly" → 0 span
- **已知 artifact**（与 reference 一致，非移植问题）：`"Apple Inc."` 会被抽成 `apple inc .`，因为 reference 的 word splitter 把句末 `.` 当成一个独立的词

### ✅ 5.2.4b-1 分类头 + null_projection + count_head
- **范围**：`[L]` marker → `cls_marker_indices` → `classifier.0` + ReLU + `classifier.3`；`null_projection`（abstention）；`count_head`
- **实现**：
  - `prompt.rs`：`BoundaryTaskKind`（Entities=`[E]` / Classification=**`[L]`** / JsonStructure=`[C]` / Relation=`[R]`）+ `build_mixed_boundary_prompt`，两套路由分开返回
  - `extract.rs`：`classify_group` / `query_heads` / `apply_abstention`；`run_mixed_extraction` 统一返回 spans + classifications + query heads
  - `adapters/gliner2_boundary.rs`：`parse_boundary_schema` 支持 `classifications` 列表（含 `label_descriptions` / `multi_label` / `cls_threshold`），CLI 两组同时输出
- **oracle**：`dump_classification_and_query_heads.py`，max delta **9.060e-6**，chosen labels 完全一致
- **关键纠正**：**classification 用的是 `[L]` 不是 `[C]`**。`[C]` 属于 `json_structures`（processor.py:1124-1188）。搞反的话 classification 的 marker 会被路由进 document pool，变成"抽取"出一堆没人要的 span，而且**不报错**
- **自己引入又抓到的 regression**：`build_with_child_marker` 在重构时把 `child_marker` 参数忽略了，`kinds=None` 默认成 `Entities`，导致 **Decide 路径的 `[L]` 全部变成 `[E]`**。所有 boundary fixture 全绿（因为 boundary 路径显式传 `kinds`），只有跑 `gliner2_large_v1_parity` 才暴露。已修 + 加了 `the_two_prompt_families_use_their_own_child_marker` 单测锁住
- **性能坑**：`classify_state` 原本每次调用都新建 `ComputePool`，而 pool worker 空闲时是**忙等**（thread_pool.rs:382）。一次 extraction 两个 pool，并发跑就把机器打满（12 核上 48 个自旋线程）。改成直接算（1.2M MAC，不需要 pool），测试从 82s → 3.2s。**这是引擎的既有特性，不是 boundary 引入的**，但 server 阶段要注意并发请求的 oversubscription

### ✅ 5.2.4b-1a overlap_policy 解码
- **范围**：`resolve_overlaps`（`inference/overlap.py:56`）。base-v1 的 `overlap_policy = "flat"` → canonical `disallow` = **最大总分不重叠集**（加权区间调度），不是贪心
- **实现**：`overlap.rs`（新）+ `decode_spans` 接上 `boundary_overlap_policy`
- **oracle**：`dump_overlap_resolution.py` 29 个 case，纯函数不需要 GGUF，**全 case 逐位一致**
- **抓到的真问题**：**之前的 e2e oracle 用错了 reference 的解码阶段**。`decode_candidates` 只做 threshold + 排序，`_decode_entities` 之后才跑 `_resolve_spans(..., policy)`。也就是说我的 e2e fixture 比的是**中间结果**，而 Rust 端也没做 resolution——两边一起错，看起来是绿的
- **修法**：oracle 现在同时 dump `spans`（中间）和 `resolved_spans`（最终），并加 threshold=0.02 的 case 强制产生重叠（默认 threshold 下模型太自信，**同一个 field 不会有重叠 span**，resolution 阶段等于没被测到）。现在 29 个候选 → `flat` 收敛到 3，`allow` 保留 14
- **fixture 里专门留了一个贪心必错的 case**：三条交叉 span，中间那条分最高（0.5）但会挡住两边；贪心拿 0.5，最优是 0.55 + 0.45 = 1.0。除了这个 case 之外的所有 case 贪心都能过

### ✅ 5.2.4b-1b HTTP 路由 `/v1/jev/boundary`
- **后端**：`Gliner2Boundary`，`--gliner2-boundary`。flag 选 head，`is_boundary_gguf()`（`gliner2.variant`）在启动时确认变体，**不匹配就报错**
- **不缓存模型**：mapping 留在后端，每请求现建 `BoundaryModel`（零拷贝 + 一次 settings 解析），和 Decide 每请求重建 tokenizer 一致，**不需要 `'static` 泄漏**
- **自己的 request/response 形状**：`JevResult` 装不下 word offset 和 resolution 后的顺序，所以 body 直接吃 reference 的 schema 形状。**刻意不 alias `/v1/jev/score`**——JEV body 在那里会报"缺 schema 字段"，读起来像请求写错而不是路由不存在，404 才诚实
- **测试**：`tests/gliner2_5_base_v1_boundary_http.rs`，真起 server + curl，3 个测试。覆盖路由分发、raw schema、响应形状、mixed 抽取+分类、threshold override、`/v1/jev/score` 必须 404、四条 400 错误路径
- **threshold override 的测试用 0.02**：默认 threshold 下同 field 无重叠，threshold 被忽略也看不出来

### 🟢 5.2.4b-2 relations + records + json_structures（待开工）
- **范围**：
  1. relations（`relation_scorer`，`[R]` marker + directional head/tail states）
  2. records（`record_decoder`，需要 `candidate_states`——已经返回了）
  3. HTTP 路由 `/v1/jev/boundary`
  4. `json_structures`（`[C]` marker + 嵌套结构解码）
- **前置**：5.2.4b-1 ✅

### 📊 5.2 阶段总结
| Phase | 范围 | 状态 | commit |
|---|---|---|---|
| 5.1 | GGUF converter + smoke | ✅ | `b0f58d3` |
| 5.2.1 | BoundaryEncoder forward + oracle | ✅ | `d725de8` + `b678b95` |
| 5.2.1b | BoundaryQueryHead + oracle | ✅ | `f3d6643` |
| 5.2.2a | score_explicit_pairs + PairScorer（全 feature） | ✅ | `6eafe4e` + `1d98e37` + 本次 |
| 5.2.2b | DocumentCandidatePool + SharedPoolScorer（主线） | ✅ | 本次 |
| 5.2.3 | relations + records + count + abstention | 🟡 待开工 | — |
| 5.2.4a | 真实入口 + `--gliner2-boundary` | ✅ | 本次 |
| 5.2.4b-1 | 分类头 + null/count head | ✅ | 本次 |
| 5.2.4b-1a | overlap_policy 解码 | ✅ | 本次 |
| 5.2.4b-1b | HTTP 路由 `/v1/jev/boundary` | ✅ | 本次 |
| 5.2.4b-2 | relations + records + json_structures | 🟢 待开工 | — |

已完成：boundary encoder（含 attention window）、per-query marginals、显式 span 的 compat prior、完整 `SparseBoundaryPairScorer`、以及**主线** `DocumentCandidatePool` + `SharedPoolScorer`，10 个 boundary 测试文件 / 25 个测试全绿，delta 在 1e-6 ~ 1.5e-5。
`score_document_candidates()` 已经能从 `text_states` 走到 `[B,Q,C]` 的最终 logits。

**5.2.4a 已完成：模型可以真的跑了。** `--gliner2-boundary` 从 CLI 端到端出 span，输出与 reference 逐位一致（`input_ids` 相等、pair-logit delta 2.813e-5、span 完全相同）。

**现在支持的**：extractive spans（`[E]`）+ classification（`[L]`）+ abstention（`null_projection`）+ count log-rate（`count_head`），两组可以同时出现在一个 schema 里。

**仍然没有的**：
- relations（`[R]`）/ records（`record_decoder`）/ `json_structures`（`[C]`）
- per-field threshold override（`_query_thresholds` 读 `entity_metadata.<field>.threshold`）和 per-sample `_overlap_policy` override
- `adaptive_threshold`（base-v1 是 false，但 `count_head` 已经算出来了）

14 个 boundary 测试文件 / 37 个测试全绿。

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