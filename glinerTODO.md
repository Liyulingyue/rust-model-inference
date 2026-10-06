# GLiNER 系列适配 TODO

fastino 的 GLiNER 家族在 ModelScope 共 11 个公开 repo。已通过 `modelscope list --owner fastino --repo-type model --all --page-size 50` 确认无遗漏。

## ⚠️ 重大发现（推翻此前假设）

`custom_tag:gliner2` 标签只代表"GLiNER 家族"，**不意味着同一架构**。下载 config 之后实际拆出 **3 个互不兼容的架构族**，每条 TODO 都要先认清自己在哪个族再决定能不能复用现有代码：

| 族 | 架构 | encoder | tokenizer | 适配状态 |
|---|---|---|---|---|
| **SpanExtractor (pre-2.5)** | `SpanExtractor`（无 `architecture` 字段，tensor 命名同 Decide） | DeBERTa-v3 | `DebertaV2Tokenizer` (SPM) | 没适配，权重可能跟 Decide 不同 |
| **SpanExtractor (2.5 / 1B)** | `architecture="span"` | Ettin 1B（ModernBERT，decoder-only 派生） | **BPE ByteLevel**（HF fast tokenizer） | ✅ 完成（`src/models/gliner_ettin/`） |
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

### ✅ 2. `fastino/gliner2-base-v1` — DeBERTa-v3-base（已完成，**零 Rust 改动**）
- **架构族**：SpanExtractor pre-2.5，与 `gliner2-large-v1` 同族；768 hidden / 12 layers / 12 heads / 3072 FF
- **flag / 路由 / 位置**：✅ 全部沿用 `--gliner2-decide` + `/v1/jev/score` + `src/models/gliner/`
- **推翻的假设**：本 TODO 原先写"推理层 `weights.rs` 同理需要动态化"——**错的**。Rust 侧
  `n_embd` / `n_layer` / `n_head` / `n_ff` / `head_dim` / `eps` / bucket 全部从 GGUF metadata
  读，`src/` 一行没改。只有转换器把尺寸写死了
- **转换器改动**（`convert_gliner.py`）：
  - `ENCODER` 整块全局硬编码 → `ENCODER_SIZES`（按 `model_name`）+ `ENCODER_COMMON`，
    `resolve_encoder()` 遇到未知 `model_name` 直接报错而不回退到 large（回退会静默产出
    能过形状契约但解码成噪声的 GGUF）
  - `validate_config` 现在额外校验 `model_type == "extractor"`（同族的标志），并返回解析后的 dims
  - `general.name` 原来对**每个**模型都写死 `"gliner2.5-decide"`（对 large-v1 也是错的）→ 改为 `model_dir.name`
  - `tokenizer_class` 只接受 `DebertaV2Tokenizer`，但 base-v1 发的是 `DebertaV2TokenizerFast`
    （同一份 SPM vocab，类名只是包装差异）→ 精确接受这两个名字，不放宽到任意 `Deberta*`
- **尺寸表怎么来的**：从权重张量形状反推 + 用 `microsoft/deberta-v3-base` 的 config 取
  `num_attention_heads`（**唯一不出现在任何张量形状里的字段**，只有 oracle 能抓）。转换器原有的
  形状契约会重新校验另外三个，填错立刻抛错
- **`counting_layer` 差异无需处理**：base-v1 是 `count_lstm_v2`（`count_embed` 为 2 层
  transformer），large-v1 是 `count_lstm`（`count_embed.projector`）。两者都不进 GGUF——转换器
  本来就 drop `span_rep.*` / `count_embed.*` / `count_pred.*` 并且拒绝 drop 任何其它东西
- **oracle**（`dump_golden.py` 也去硬编码）：分类头 `Linear(hidden*2)` 现在从 encoder config 推导，
  `load_state_dict` 严格模式，宽度错了在生成时就炸。**max logit delta 3.433e-5**（1e-4 容差）
- **测试**：`tests/gliner2_base_v1.rs`（6 smoke）+ `tests/gliner2_base_v1_parity.rs`（2 byte-exact）
- **跨变体回归**：base-v1 与 Decide 的 `input_ids` / `marker_positions` **逐字节相同**（encoder 尺寸
  不参与 tokenization），已作为测试钉住
- **验证 large-v1 零回归**：用新转换器重建 large-v1，394 个 tensor 的 **1.74 GB 张量数据逐字节
  相同**；metadata 仅 3 处差异，都是上面有意改的（`general.name` / `source_architecture`
  补 `model_type` / `source_config` key 顺序）
- **价值**：base 尺寸跑得快很多（吞吐 ~3×），生产部署友好

### ✅ 3. `fastino/gliner2-multi-v1` — mDeBERTa-v3-base（已完成，抓到两个真 bug）
- **架构族**：SpanExtractor pre-2.5（同 #1/#2），`model_name: microsoft/mdeberta-v3-base`
- **flag / 路由 / 位置**：✅ 沿用 `--gliner2-decide` + `/v1/jev/score` + `src/models/gliner/`
- **encoder 侧零新代码**：mDeBERTa-v3 的 config 与 deberta-v3-base **逐字段相同**（只差
  `vocab_size` 251000 vs 128100），HF 用同一个 `DebertaV2Model`，且 torch 版用的是普通
  `softmax`（`XSoftmax` 只存在于 TF 实现）。所以 `ENCODER_SIZES` 里它就是 base 那一行
- **测试**：`tests/gliner2_multi_v1.rs`（6 smoke）+ `tests/gliner2_multi_v1_parity.rs`（1
  byte-exact）。**max logit delta 5.627e-5**（1e-4 容差）
- **故意没有**跨变体 `input_ids` 测试：250k 多语 SPM 的分段与 DeBERTa 不同，该断言不成立

**bug 1：schema marker 的 id 被硬编码在 128000 段**
- `prompt::ADDED_TOKENS` 原本是 `const [(&str, u32); 15]`，11 个 GLiNER2 special 写死
  128000..128010。那只对 128k 词表成立——mDeBERTa 的追加块从 **250101** 开始，于是每个
  `[P]`/`[L]`/`[E]` 都被编码成**不存在的 id**
- 症状特别隐蔽：周围文本分词完全正确，只有 marker 落到 embedding 表的错误行
- 修法：拆成 `BASE_SPECIALS`（SPM 约定固定 0..3）+ `APPENDED_SPECIALS`（只存名字），
  id 由 `added_tokens(spm.len())` 推导。基址取自**做编码的那个 tokenizer 自己的 piece
  数**，所以 id 不可能与词表漂移
- 验证中性：三个 deberta 模型的 `spm.piece_count` 都是 128000，推导出的 id 与原硬编码
  逐个相同；只有 multi-v1 是 250101

**bug 2：SentencePiece trie 的根节点与第一个子节点别名（跨模型、跨进程不确定性）**
- `PieceTrie` 用 `#[derive(Default)]`，`values` 初始为空，于是 `insert` 里
  `child = values.len()` 让**第一个子节点拿到索引 0，与根槽位重合**
- 后果：任何在根层走到"第一个 piece 的首字节"、并恰好在该处结束的 piece，都会把
  `values[0]` 覆写成自己的 id，根节点就此被污染
- **为什么是间歇性**：谁赢取决于 `normal_ids`（`HashMap`）的迭代顺序，而
  `RandomState` 每进程不同 → 同一输入在不同进程给出不同分段。实测
  `down. Can` 会切成 `666.` / `1111.` / `0000.` / `.` 之一
- 定位手法：先证明"同进程内 200 次稳定、跨进程变"（排除 DP 和 prompt 构造），再在
  `insert` 里加碰撞检测，一眼看到冲突全在 `node 0`；最后用"同进程重建 60 次"把
  间歇性变成**确定性复现**（60 次里 5~13 种结果）
- 修法：`PieceTrie::new()` 显式 `values: vec![u32::MAX]` 预分配根哨兵
- 回归测试：`piece_trie_root_never_holds_a_piece` +
  `piece_trie_build_is_order_independent`。**原有的 `piece_trie_reports_prefixes_in_order`
  在有 bug 时也通过**，所以这个洞一直没被发现——两个新测试退回修复后立刻失败
- 影响面不止 gliner：所有 SentencePiece 模型（BERT 家族等）都走这条路径
- 修复后 multi-v1 parity 连跑 15/15 通过（修复前约 2/10 失败）
- 其它既有模型的 golden 全部仍然通过——说明这个 bug 此前只在少数输入上触发

**顺带修的 tokenizer 差异**
- `tokenizer_class`：base-v1 发的是 `DebertaV2TokenizerFast`，另两个是 slow 类。精确接受
  这两个名字，不放宽到任意 `Deberta*`
- `added_tokens_decoder` 里可以有**已在 SPM vocab 内**的声明：4 个 base specials 加上
  mDeBERTa 的 100 个 `<extra_id_N>` sentinel。转换器现在按 id 划分，vocab 内的每个都必须
  与它声明的 piece 一致（比忽略它们更严），只有越过 SPM 边界的 11 个算追加 token
- oracle 的 `vocab_size` 原写死 128011，改为从 checkpoint 的 embedding 表读，并校验它
  覆盖 tokenizer 的最大 added-token id（mDeBERTa 发布 config 写 251000，是**上界**不是
  piece 数）

---

## 🟠 SpanExtractor 1B 族（**完全不同**：Ettin + BPE）

### ✅ 4. `fastino/GLiNER2.5-Decide-1B` — Ettin-1B + ByteLevel BPE（**已完成**）
- **架构族**：SpanExtractor（`architecture=span`, `config_version=3`），head 契约同 Decide
- commit `ac27379`（tokenizer）+ `4e58bfd`（forward）
- 6 个请求 `input_ids` 逐 token 一致，最大 logit delta **7.6e-4**

**⚠ 勘察阶段的六处猜测全部被实测推翻**——`models/ettin-enc-from-dec-1b/config.json`
的权威答案是 `ModernBertForMaskedLM`，不是 LLaMA 派生的纯 decoder：

| 勘察时猜的 | 实测 | 出处 |
|---|---|---|
| RMSNorm | **LayerNorm，`bias = False`**，eps 1e-5 | `norm_bias: False` |
| SwiGLU | **GeLU GLU**：`chunk(2)` + `act(first) * second` | `ModernBertMLP` |
| 16 头 | **28 头**，head_dim 64，`hidden 1792` | `num_attention_heads: 28` |
| 纯 LLaMA 式全局注意力 | **混合**：层 0,3,6… 全局，其余 128 宽窗口 | `global_attn_every_n_layers: 3` |
| QKV 可能连续 | **交错**：`view(seq, 3, heads, head_dim)`，role 轴跨 `heads*head_dim` | `ModernBertAttention.forward` |
| 无位置参数 → 纯 LLaMA | RoPE **有**，theta 160000；`position_embedding_type: sans_pos` | `rope_parameters` |

层 0 的 `attn_norm` 是 `nn.Identity()`，checkpoint 里**没有对应张量**（199 → 202 tensor）。
`mlp.Wi` [7680, 1792] = 2 × 3840 确实是 fused gate+up，但门控算子是 GeLU 不是 SwiGLU。

**实现**：`src/models/gliner_ettin/{mod,bpe}.rs`（全新模块，未复用 llama/qwen3——
见下）。转换器 `tools/converter/gliner/convert_ettin.py`，oracle `dump_ettin_golden.py`。

**关于"先评估能否复用 `llama`/`qwen3` 的 RMSNorm+SwiGLU+RoPE"**：
评估结论是**不可复用**，且理由比"接口不同"更硬——仓库里根本没有 RMSNorm+SwiGLU+RoPE
的现成组合。`llama`/`qwen3` 用 RMSNorm + SwiGLU + RoPE，ModernBERT 是 LayerNorm +
GeLU GLU + RoPE，两者只共享 RoPE。`src/ops/norm.rs` 的 `rms_norm` 和
`silu_mul_inplace` 都是**错误的算子**，模块注释里已写明这一点。强行复用的结果是
"形状对、数值全错"，而这正是本模型五个 forward bug 的共同特征。

**ByteLevel BPE 是一等实现**，不是薄封装。`src/core/tokenizer/` 里现有的三套
（`SPMTokenizer` linked-list trie、`UgmTokenizer` `NaiveTrie`）都是 SentencePiece /
Unigram，没有 ByteLevel 的 byte→unicode 映射与 GPT-2 预分词。对齐 `tokenizers`
需要四处规则，详见 `docs/GLINER_FAMILY.md`。

**已知限制**：`nfc()` 目前是恒等实现（std 没有 Unicode 归一化）。12 个 fixture case
都不含 combining mark，所以尚未暴露。`is_nfc_safe()` 提供前置检测，PR 里已写明。

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

### ✅ 5.2.4b-1c relation head（generator + scorer + oracle）
- **实现**：`relations.rs`（新）—— `TypedRelationPairGenerator`（typed+capped top-k）和 `SparseRelationScorer`（local feature MLP + biaffine content）
- **base-v1 设置**：`heads_per_relation=32`、`tails_per_relation=32`、`pair_cap=64`、**`argument_threshold=0.2`**（reference dataclass 默认是 0.0！）、`relation_temperature=1.0`、`directional_relation_states=true` → relation query 宽度 1536、`relation_biaffine_content=true`
- **converter 补了两处**：`relation_biaffine_content` 之前**没有**转写（loader 会读不到）；新增 `check_relation_scorer_shapes`，12 个 tensor 全部交叉校验。**单独一个函数**而不是塞进 `check_pair_scorer_shapes`——后者的局部 `d` 是 `boundary_dim`(128)，relation scorer 用的是 encoder hidden(768)，共用一个 scope 两者只差一个字符
- **`mlp.3` 不是 `mlp.2`**：`nn.Sequential(Linear, GELU, Dropout, Linear)` 里 Dropout 占一个 index
- **抓到的三个坑**：
  1. **argument 概率不除 temperature**：span decode 用 `sigmoid(logits/pair_temperature)`，但 generator 内部重新算 `sigmoid(pair_logits)`。base-v1 的 `pair_temperature=1.0` 两者相等，**任何 fixture 都测不出来**——仍然照抄，否则换 `pair_temperature≠1` 的 checkpoint 会静默地用不同阈值筛 argument
  2. **padding slot 会污染 pair top-k**：reference 把 `pair_valid = hvalid & tvalid` mask 掉，所以「真 head × padded tail」被丢弃；我第一版靠 score=0 推断 validity，会让这些 0 分 pair 在**文档 argument 不够时**挤掉真 pair。`valid` 是承重字段
  3. **mention 排序是 `(start, end)` 字典序**，不是分数序；它存在只是给后面的 score sort 一个确定 tie-break。两次 stable argsort（先 end 后 start）== 一次 stable sort on `(start,end)`
- **oracle**：`dump_relations.py`，4 个 case，pair 集合和 logit 都对齐（TOLERANCE 2.0e-4）。case 覆盖：同分 mention（同 span 不同 query / 相邻 span）、padding + threshold 边界（logit 恰好等于 `logitit(0.2)`，reference 用 `>=`）、same-span 自剔除、named/dead spec

### ✅ 5.2.4b-1d relation `[R]` prompt + schema + decode + CLI/HTTP
- **schema 形状**（reference `_process_relations`）：`{"relations": [{"founded_by": {"head": <span>, "tail": <span>}}], "relation_descriptions": {...}}`。value 是 gold span，**只有 field name 进 prompt**；第一个 field 是 head、第二个是 tail，所以顺序就是方向
- **`parse_boundary_schema` 的 group 顺序改对了**：之前是 classifications → relations → entities，reference 的 `_transform_record` 是 json_structures → entities → relations → classifications。entity query 的 id 必须在 relation role slot **之前**，否则 head/tail 拿到 0/1 而 entity 被挤到后面。e2e fixture 的 mixed case 专门盯这个
- **抓到一个 routing bug**：`yields_boundary_queries()` 之前只匹配 `Entities`，所以 `[C]`/`[R]` 的 child 被路由到 classifier，报错是 "4 classification choices routed but 0 consumed"——指向 classification head 而不指向 routing，很难查。reference 的判据是 `task_types == "classifications"`，即**其余全部**走 query 侧
- **relation type 字符串**：`_schema_group_name` 取的是 prompt-joined 形式（`"worked_in: who worked in which place"`），`_decode_relations` 的 alias 表再映射回裸名。所以 `RelationTypeSpec` 存 joined 形式，decode 时 `resolve_relation_type` 按 `": "` 切回。description 泄漏进输出只有 fixture 带 `relation_descriptions` 才看得见
- **`run_mixed_extraction` 加了 `relation_threshold: Option<f32>`**：reference 的 `_decode_relations` 收的是和 span 路径同一个 `threshold`，不是硬编码
- **oracle**：`dump_relations_end_to_end.py`，5 个 case（真实 text + relation schema → edges），含 mixed schema、threshold 0.02、负例。query routing / word list / edge 全部对齐（2.0e-4）

### 🟡 5.2.4b-2 records + json_structures（已勘察，未开工）

勘察结论（`records.py` 1421 行，但只有 ~310 行是推理路径）：

**推理只需要两个函数**（其余全是 loss）：
- `RecordHead.forward_group`（`records.py:572-698`，~130 行）
- `decode_group`（`records.py:714-893`，~180 行）

**18 个 tensor**，与 safetensors 完全对应（`record_dim=128`、`instance_queries=32`）：
`inst_proj` / `field_proj` / `cand_proj`（768→128）、`null_embed`(128)、
`object_head` / `latent_seed_head`（768→1）、`instance_embed`(32×768)、
`q_proj` / `k_proj`（768→128）、`v_proj`（768→768）。
（`record_decode.py` 只有 8 行，是 re-export；`RecordSetDecoder`、
`FieldAssignmentScorer`、`create_anchor_instances` **推理不调用**——文件头
自己写了 "low-level primitives"，只有训练/其他入口用。）

**三种 instance 模式**，由 `record_metadata.<parent>.mode` 选：
- `natural`：`inst_states` 就是 anchor field 的候选状态，`object_logits` 就是 anchor
  候选的 `pair_logits`（不经过 `object_head`！）
- `latent`：所有 field 的候选都进 `latent_seed_head` 评分当 seed
- `anchorless`：`instance_embed` 当 instance 状态，过 `object_head`；阈值用
  `object_threshold` 而不是 `anchor_threshold`

**最大的坑：`decode_group` 用匈牙利算法做 exclusive scalar field 的全局联合分配**，
不是贪心。reference 自己的注释说明为什么：贪心让 object 最高的 instance 先抢它最喜欢
的候选，会把后面的 instance 逼到无关 span 上。

**✅ LSA 已移植**（`a` 待补 commit 号，`matching.rs`）：
- **不需要自己从零写**：reference 自带一个确定性 O(n²m) Jonker-Volgenant solver
  (`training/matching.py:22`)，只在 scipy 存在时才走 scipy（且 scipy 那条路会加
  **sub-ULP 字典序 offset** 来 break tie）。我们移植**内部那个**，因为它才是被指定的
  那个：本身确定、无浮点扰动、结果不依赖「某个依赖装没装」。oracle venv 没有 scipy，
  所以 reference 走的就是内部路径
- **tie-break 是输出的一部分**：最优解**经常不唯一**（两个候选同分 → 两个同价最优解），
  返回哪个决定每个 record field 绑哪个 span。所以 oracle **只断言 `(row, col)` 对**，
  **不断言总代价**——任何 solver 都能通过总代价检查，包括返回明显不同 span 的
- `minv[j] < delta` 是**严格**比较 → 同分取最小列下标；最后按列扫描再按 row 排序
- 21 个 case，含全等矩阵（2x2/3x3/4x4）、**争夺同一列**、代价平台、相同行、
  两种矩形方向、empty、单元、**全 inf 行**（转成大哨兵而不是失败）、**NaN 直接报错**

**剩下的坑**（`decode_group` 内部，还没实现）：
- `allows_absent = false` 时 diagonal 是个标量 `max(candidate_cost) + 50.0`
  **broadcast 到所有行**（不是逐行 max），这个 `+50` 语义不能改成逐行
- `invalid_cost = max(candidate_cost.max(), diagonal.max()) + 1000.0`
- list field 走 sigmoid + 每候选取 argmax row（**不走** LSA）

**还没确认的**：
- `RecordSpec` 的编译逻辑（`query_id` 怎么分配、`anchor_query_id` 怎么定）
- `_process_json_structures`（`processor.py:921-1022`）的 `json_descriptions` /
  `record_metadata` schema 形状
- `candidate_states` 我们已经有（`DocumentCandidateBatch.candidate_states`，
  `[B, C, pair_dim]`）—— 但 records 要的宽度是 `hidden_size`(768)，
  **`pair_dim`(128) 不够**，需要确认 reference 的 `candidate_states` 宽度

### ✅ 5.2.4b-1e `candidate_encoder` / `candidate_states`（records 的前置）
- **新** `candidate_encoder.rs`：`Linear(2*boundary_dim, hidden_size)`，**无激活**
- **改名**：旧的 `candidate_states` → `pool_candidate_features`（scorer 内部特征）
- **oracle**：`dump_shared_pool_scorer.py` 新增 `candidate_state_rows`（768 宽），
  与 `candidate_rows`（128 宽）并列，两个都验
- **保持 `[B, C, H]` 而不是 `[B, Q, C, H]`**：reference 的 `to_candidate_batch` 用
  `expand` 得到的，值不依赖 `q`，物化那个轴要白花 `q_count` 倍内存
- **无效 slot 必须精确为 0**（不是「很小」）：reference 是
  `.masked_fill(~mask, 0.0)`，padding slot 的端点是池填充的残留值，真状态漏进去
  会污染 record head 的 assignment 分数

### ✅ 5.2.4b-1g `record_metadata` 归一化 + `RecordSpec` 编译
- **新** `record_spec.rs`：`FieldCardinality` / `default_cardinality` /
  `normalize_record_metadata` / `compile_record_specs`。纯函数，**不需要模型**
- **确认 `RECORD_TASK_TYPES = ("json_structures",)`**：`json_structures` group 带
  `mode` → record spec；不带 → legacy structure 路径。所以「未标注」是个**有意义
  的状态而不是默认值**，而且这是整个 feature 里**唯一的静默路径**——`_compile` 跳过
  而不报错。哪天它开始编译出 spec，所有未标注的 group 语义就静默变了，所以专门有
  fixture case + 独立测试盯着
- **validation 才是重点**：每条规则都拦下一个「否则会在更晚、更难懂的地方炸」的 schema。
  `natural` 缺 anchor / `latent` **带** anchor（不是忽略而是报错，因为设了它说明调用方
  以为自己写的是 natural record）/ cardinality 不在 enum 里 / anchor 指向不存在的
  field
- **cardinality 默认**：anchor 字段一律 `required_one`（没 anchor 的 instance 不算
  instance），`dtype == "str"` → `optional_one`，其余 → `zero_or_more`。
  `is_scalar` / `allows_absent` 是 decoder 分两条路的依据，所以单独测了 enum 定义
- **schema 形状**：`json_descriptions[parent]` 是 **field → description 的 map**
  （relations 那边的 group description 是纯字符串，这里不一样）。field 顺序是所有
  occurrence 的 key 并集按首次出现顺序——reference 特意没走 set，因为那样 schema prompt
  会依赖 `PYTHONHASHSEED`
- **oracle**：`dump_record_specs.py`，16 个 case（含 7 个 error case 记 message）

### ✅ 5.2.4b-1h `[C]` json_structures schema 解析 + prompt 路由
- **新** `parse_json_structure_groups`：`{"json_structures": [{"parent": [fields]}],
  "json_descriptions": {"parent": {"field": "desc"}}}`，输出 `[C]` Task
- **field 顺序 = 所有 occurrence 的 key 并集按首次出现顺序**。reference 特意没走
  set（那样会依赖 `PYTHONHASHSEED`，同时改掉 query 顺序和解码值）。两个 entry 命名
  同一个 parent 会**合并成一个 group**
- **`json_descriptions[parent]` 是 field → description 的 map**（relations 那边的 group
  description 是纯字符串）。当成字符串读会静默丢掉所有 description 并让后面所有 marker
  index 位移——单测专门盯这个
- **空 group 被跳过**（而不是产出一个没有 `[C]` child 的 group，那个 group 一个 query
  都没有）。若 schema 里只有空 group → 报「needs one of ...」而不是返回空成功
- **parser 不看 `record_metadata`**：同一个 schema 加不加 record 标注必须产出**完全相同**
  的 prompt，否则 query id 会变。单测断言 `bare == annotated`
- **抓到我自己的排序错误**：我先写了 entities → json_structures，但
  `_transform_record` 是 **json_structures 最先**。单测直接把这个抓出来了

### ✅ 5.2.4b-1i `RecordHead.forward_group` + `decode_group`（18 个 tensor）
- **新** `record_head.rs`：三种 mode（`natural` / `latent` / `anchorless`）+
  `_assign_logits`（instance/field 投影**先相加**再点积）+ `_anchorless_states`
  （`instance_embed` + 一次 attention over 全部候选）+ `decode_group`
- **`natural` 的 object logits 是 anchor 候选的 `pair_logits` 直通**，不过
  `object_head`。这是最容易漏的一处，单测直接断言 `object_logits == anchor pair_logits`
- **`anchorless` 用 `object_threshold` 而不是 `anchor_threshold`**（它没有 anchor）。
  case 里把两个阈值设成 0.99 / 0.1，用错必然选出不���的实例集
- **exclusive scalar field 走 LSA 全局分配**；`allows_absent = false`（`required_one`）
  时 absent 列是 `max(candidate_cost) + 50.0` 这个**标量 broadcast 到每一行**，
  不是逐行 max——改成逐行会让「候选便宜的行」逃掉「候选贵的行」被迫接的分配。
  `invalid_cost = max(...) + 1000.0`，再拼 `row_count` 个 absent 列保证每行有落点
- **list field 不走 LSA**：exclusive 的每个候选取 argmax row 归给那一个 instance；
  非 exclusive 的逐候选取 threshold
- **抓到的 bug**：
  1. **non-exclusive scalar 遇到 null 列时 reference 是 `continue` 不是 `break`**。
     `required_one` 字段要「跳过 null 列取下一个候选」，我 break 了 → 字段空掉，
     而这**正是 cardinality 禁止的 ABSENT 结果**，看起来像「没抽到东西」
  2. **LSA 返回的 (row, col) 是成对的**，我按循环下标去索引 `assigned_cols`，
     应该是 `assigned_rows[i]` 配 `assigned_cols[i]`
  3. **latent/anchorless 的记录顺序是 dict 插入序**，我返回 BTreeMap 的 key 序
- **`instance_embed` 的 dims 是 relabel 不是 transpose**：converter 只改了 shape
  标注（`(in, out)`），**字节仍是 checkpoint 的 row-major**。按 dims 读会得到一个
  32 行全被打乱的表；而表是 `randn * 0.02`，每行几乎一样，logit 只差 ~0.01——
  **松 tolerance 会盖住，且没有任何结构暗示这是置换**。所以专门加了
  `anchorless_without_candidates_isolates_instance_embed`（候选全 invalid → 跳过
  attention → 只剩 `object_head(instance_embed)`）
- **fixture 静默失效过一次**：synthetic candidate states 尺度不够时 11 个 case 里
  2 个解出 0 条记录，但**测试照样绿**——「head 和 reference 在「无」上达成一致」
  和「一致」长得一模一样。现在有测试守着不让任何 case 掉到 0

### ✅ 5.2.4b-1j records 接进 extract/CLI/HTTP —— 并修掉 LSA 的 cost 矩阵 bug
- `run_mixed_extraction` 多收一个 `record_metadata: Option<&Value>`（schema 顶层 key）。
  没有它 `compile_record_specs` 永远返回空，**每条 json_structures 都静默走 legacy 路径**
- `Extraction` 加 `records`；CLI 打文本 / `--jev-output json` 打 `records`；
  HTTP `/v1/jev/boundary` 响应加 `records`
- `BoundarySettings` 读 `record_anchor_threshold` / `record_anchor_proposal_threshold` /
  `record_field_threshold` / `record_temperature`；`BoundaryModel` 持有
  `record_head: Option<RecordHead>`（按 `enable_records` gate）
- **手工验证语义正确**：`Marie Curie worked with Pierre Curie in Paris.` →
  marie curie→paris，pierre curie **拿不到 paris**（exclusive，Paris 已分配）。这正是
  贪心会搞错、必须全局分配的场景

**🔴 抓到的真 bug（`decode_group` 的 cost 矩阵）**：
```
cost[row * width + row] = diagonal[row];          // ❌
cost[row * width + candidate_count + row] = ...;  // ✅
```
ABSENT 块的对角线要落在**第 `candidate_count + row` 列**，我写成了第 `row` 列 ——
**落进真实候选块里，静默覆盖了一个真实 cost**，同时该行自己的 absent 槽留在
`invalid_cost`。**不报错**，solver 只是老老实实解了另一个问题的最优解。

**定位方法值得记下来**：把 Rust 构造的矩阵 dump 出来，喂给 **reference 自己的
solver**：
- reference solver 在**我的**矩阵上给出**我的**答案 → 矩阵错，solver 对
- reference solver 在我的矩阵上给出 **reference** 的答案 → solver 错

一次跑就二分出了责任方。修之前矩阵 row0 = `[65.94, 15.94, ...]`，reference 是
`[0.0, 15.94, ..., 65.94, 1065, ...]`，一眼看出整体错位一列。

**为什么 21 个 solver oracle case 全绿**：它们全是手工小矩阵，**没有一个是这个
形状**（真实块 + `rows` 宽的 ABSENT 块，只有对角那个便宜）。已补 3 个
`record_*` case 进 `dump_assignment.py`，`record_absent_block_4x8` 就是抓到这个
bug 的那个矩阵本身。

顺带发现：修对角线时我引入的 `let width = candidate_count + rows` **shadow 了**已有的
softmax `width`，导致后面 `probs[matrix_row * width + ...]` 全部用错宽度 → 越界。
改名 `matrix_width`。

### ✅ 5.2.4b-1k records 端到端 oracle（补上唯一没验的接缝）
- **新** `dump_records_end_to_end.py`（9 case）+ `gliner2_5_base_v1_records_e2e_parity.rs`（2 测试）
- **验的是接缝**：head fixture 喂的是**我自己编的** `candidate_states` 公式，它验的是
  head 的算术，不是 pool→head 的交接。端到端这里状态来自 checkpoint 真实的
  `candidate_encoder`、候选来自真实 pool（这个长度下每 query 66 个有效槽）、field query
  id 来自真实 prompt 路由
- **为什么两个 oracle 都要**：一个一列的 cost 矩阵错位和一个 `width` shadowing 都是
  **过了 head fixture**、只在真实形状 + reference 下才暴露的
- case 覆盖：natural / latent / anchorless、`exclusive` 抢占（marie curie 拿 paris、
  pierre curie 拿不到）、`zero_or_more`、entities+records mixed（钉住 record field 的
  query id 在 entity 之后）、负例、threshold 0.02
- **额外断言**：exclusive 字段全局只绑定一次（不只是「和 reference 一致」，而是「分配本身对」）

**顺带修的 parser bug（e2e 才暴露）**：`json_structures[parent]` 的形状。Rust 侧原来
只接受「field 名列表」，但 reference 的 `_process_json_structures` 是
`for field_name in occ`——`occ` 是 `{field: span}` **dict**，字段名是它的 key
（value 是训练 target，不进 prompt）。现在两种都接受：dict（单次出现的简写）和
list-of-dict。同时 oracle 用 `error_policy="raise"` 而非 `"fallback"`——`fallback`
会把 malformed schema 静默替换成 dummy `[E] entity`，症状变成「没有 record」而不是
schema 错误本身。

### ✅ 5.2.4b-4 `[C]` legacy structure 路径（无 `record_metadata`）
- **新** `src/models/gliner_boundary/structure.rs`：`decode_legacy_structures` /
  `StructureSpan` / `StructureField`(Scalar|List) / `StructureInstance` /
  `LegacyStructureGroup`
- **新** `dump_structures_end_to_end.py`（7 case）+ `structures-e2e-golden.json` +
  `gliner2_5_base_v1_structures_e2e_parity.rs`（3 测试）
- **接进** `extract.rs`（`Extraction.structures` / `score_structures`）、
  `gliner2_boundary.rs`（`BoundarySchemaOptions` / CLI + JSON 输出）
- **reference 的四条 legacy 语义**（全部 oracle 覆盖）：
  1. 每个 group **恰好一个** instance——boundary 没有 count-slot 轴，所以没有「取第几行」
     的问题；这是它与 records 最大的结构差异
  2. `field_metadata["<group>.<field>"]["dtype"] == "str"` → scalar，**取 `spans[0]` 并丢弃
     其余**；缺失默认 `"list"`，全留
  3. 字段全空 → **整个 instance 丢弃**（不是输出空对象）
  4. scalar 遇 null 列是 `continue` 不是 `break`（`required_one` 字段因此不会静默变空）
- **两个承重排序**（`structure.rs` 模块注释里写了原因）：先按 score 降序取 scalar 的
  `spans[0]`，再按字段顺序输出 list。顺序反了 scalar 会拿到 list 的第一项
- **手动验证**：`Marie Curie worked in Paris with Pierre Curie in London.` →
  `name: marie curie`、`city: london | paris`、`colleague: pierre curie`
  （scalar 单值、list 多值）
- **已知未覆盖**：带 `choices` 的 `field_metadata` 会走 reference 的
  `_decode_choice_field` / `_record_local_choice_mentions` literal-enum 分支，当前只覆盖
  无 `choices` 的路径（oracle docstring 已写明此限制）

### 📊 5.2 阶段总结
| Phase | 范围 | 状态 | commit |
|---|---|---|---|
| 5.1 | GGUF converter + smoke | ✅ | `b0f58d3` |
| 5.2.1 | BoundaryEncoder forward + oracle | ✅ | `d725de8` + `b678b95` |
| 5.2.1b | BoundaryQueryHead + oracle | ✅ | `f3d6643` |
| 5.2.2a | score_explicit_pairs + PairScorer（全 feature） | ✅ | `6eafe4e` + `1d98e37` + 本次 |
| 5.2.2b | DocumentCandidatePool + SharedPoolScorer（主线） | ✅ | 本次 |
| 5.2.3 | relations + records + count + abstention | ✅ | `af503dc` ~ `ea0e1dd` |
| 5.2.4a | 真实入口 + `--gliner2-boundary` | ✅ | 本次 |
| 5.2.4b-1 | 分类头 + null/count head | ✅ | 本次 |
| 5.2.4b-1a | overlap_policy 解码 | ✅ | 本次 |
| 5.2.4b-1b | HTTP 路由 `/v1/jev/boundary` | ✅ | 本次 |
| 5.2.4b-1c | relation head（generator + scorer） | ✅ | 本次 |
| 5.2.4b-1d | relation `[R]` prompt + schema + decode + CLI/HTTP | ✅ | 本次 |
| 5.2.4b-1e | `candidate_encoder` / `candidate_states` | ✅ | `2cd425b` |
| 5.2.4b-1f | LSA（匈牙利）solver | ✅ | `b446a98` |
| 5.2.4b-1g | `record_metadata` + `RecordSpec` 编译 | ✅ | `ebf0c39` |
| 5.2.4b-1h | `[C]` json_structures schema + prompt 路由 | ✅ | 本次 |
| 5.2.4b-1i | `RecordHead.forward_group` + `decode_group` | ✅ | `07d808d` |
| 5.2.4b-1j | records 接进 extract/CLI/HTTP | ✅ | `226902a` |
| 5.2.4b-1k | records 端到端 oracle | ✅ | `ea0e1dd` |
| 5.2.4b-2/3 | json_structures（`[C]`）+ record head | ✅ | `7df4ed0` ~ `ea0e1dd` |
| 5.2.4b-4 | `[C]` legacy structure 路径 | ✅ | 本次 |

已完成：boundary encoder（含 attention window）、per-query marginals、显式 span 的 compat prior、完整 `SparseBoundaryPairScorer`、以及**主线** `DocumentCandidatePool` + `SharedPoolScorer`，10 个 boundary 测试文件 / 25 个测试全绿，delta 在 1e-6 ~ 1.5e-5。
`score_document_candidates()` 已经能从 `text_states` 走到 `[B,Q,C]` 的最终 logits。

**5.2.4a 已完成：模型可以真的跑了。** `--gliner2-boundary` 从 CLI 端到端出 span，输出与 reference 逐位一致（`input_ids` 相等、pair-logit delta 2.813e-5、span 完全相同）。

**现在支持的**：四类 query group 全部打通，可同时出现在一个 schema 里——
extractive spans（`[E]`）、classification（`[L]`）、relations（`[R]`）、
structures/records（`[C]`）；外加 abstention（`null_projection`）与 count log-rate（`count_head`）。
`[C]` 的两条分支都在：带 `record_metadata.mode` → record（LSA 分配），不带 → legacy structure。

**仍然没有的**：
- `relation_metadata.<type>.threshold` per-type override（现在统一用 caller 的 threshold）
- per-field threshold override（`_query_thresholds` 读 `entity_metadata.<field>.threshold`）和 per-sample `_overlap_policy` override
- `adaptive_threshold`（base-v1 是 false，但 `count_head` 已经算出来了）
- `field_metadata` 的 `choices` literal-enum 分支（`_decode_choice_field` / `_record_local_choice_mentions`）

22 个 boundary 测试文件 / 68 个测试全绿（**0 ignored**）。

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
| ✅ C | `gliner2-base-v1` + multi-v1 | span pre-2.5 | 中 | **高** | ✅ `9f71c78` / `8215fd8` |
| D | boundary 家族另三个变体（multi-v1 / multi-Decide / small-v1） | boundary | 小（尺寸表） | 中 | ✅ `b6bba85` |
| E | 三个 guardrail（Guardrails-PII / privacy-filter / gliguard） | span pre-2.5 | **≈0**（tokenizer 声明风格） | 中 | ✅ `499e7ec` |
| F | 架构融合评估（11 个变体实现后） | 全部 | 文档 | — | ✅ `ca40928` → `docs/GLINER_FAMILY.md` |
| G | `GLiNER2.5-Decide-1B`（Ettin + ByteLevel BPE） | span 1B | 大（~2000 行） | 中 | ✅ `ac27379` / `4e58bfd` |

每条开工前必须先 `modelscope download --model <repo> config.json tokenizer_config.json`（按 model-download skill），读 config 填本文件对应 TODO 的"flag 决策 / 路由 / 架构位置"三栏，再写代码。**不再做"看着像就动手"的盲改**。

## 特性层待办（模型已适配 ≠ 特性齐全）

12 个模型、2 个架构（`span` / `boundary`）都已 byte-exact，**但用户可见特性不等于模型**。
本节只记「reference 有、我们没有」的语义特性；框架层（批处理、`batch_size`、AMP/`quantize`、
`torch.compile`、设备管理、LoRA、训练循环）按用户判断**不在本 PR 范围**。

划界标准（用户原话）：**模型家族特性 = 用户在 schema 里声明、影响推理输出的语义**；
**框架/基础设施 = 批处理、张量并行、线程数、dtype、设备**。

开工前**必须先写 oracle**。这一节的所有条目都踩过同一个坑：不看 reference 源码、
只凭模型「应该支持某特性」去猜，会漏掉一半。已完成的 5 项全部先跑 oracle 拿真实答案。

### ✅ 已完成（分支 `gliner-features`）

| 特性 | commit | oracle | 备注 |
|---|---|---|---|
| `adaptive_threshold`（count head 引导准入） | `1c1466a` | `dump_adaptive_threshold.py` | 4 个 boundary config 全是 `false`，**不影响 12/12**，是给未来 checkpoint 的 |
| `word_splitter="char"` + 修 `\w` | `e9904c4` | `dump_word_splitter.py` | **修了既有 bug**，见下 |
| per-entity / per-field / per-relation threshold | `5a22717` | `dump_per_query_thresholds.py` | 三条**互不相通**的通道 |
| `validators` / `RegexValidator` | `3be43a0` | `dump_regex_validator.py` | 两处 engine 差异**显式记录**而非掩盖 |
| `choices` 全程（prefix + 查表 + 打分 + dtype/gate） | `34622d3` + 待提交 | `dump_choice_fields.py` + `dump_choice_decode.py` | ✅ 8 个测试（含 model-backed e2e） |
| `_record_local_choice_mentions`（文档级字面量归属） | `3fa6645` | `dump_record_choice_mentions.py` | ✅ 5 个测试；两处变异均被抓 |
| 长文本 `*_long` 的 chunk + merge | 待提交 | `dump_long_document.py` | ✅ 12 个测试；多数投票变异被抓 |

### ✅ F-1 `choices` literal-enum（**全程完成**）

**已验证（7 个单测 + 1 个 model-backed e2e + 2 个 oracle fixture）**：
- [x] prompt 侧 prefix 渲染 + 落在 text 流 `[SEP_TEXT]` 之后（`34622d3`）
- [x] `_find_choice_idx`：prefix 区内小写全等匹配，**整条目匹配不重分词**
      —— 所以多词字面量 `very happy` 是**一个**条目、能匹配、span 覆盖整个字面量
      （我一开始以为是两个 token 匹配不上，oracle 推翻了）
- [x] 重复字面量只打分一次（取首次出现）；prefix 里没有的字面量被丢弃
- [x] `dtype` 两个分支：`list` 给全部过阈值的、**按声明序**；scalar 给 `argmax`，
      **best 未过阈则什么都不给**（不退回第一个）。`uppercase_choices` 钉住声明序
      （第二个分数更高却排第二）
- [x] `field_metadata` 里 reported value **保留声明时的大小写**，只有查表折叠
- [x] 空列表结构被判为「无内容」而丢弃（与 span 字段同一规则）

- [x] **打分本身**：给 choice token 的 `(idx, idx+1)` 走 `score_spans`。
      途中修掉 `BoundaryProposer` 一个**只在 `enable_rotary_endpoints` 开启
      （base-v1 就是）时才触发**的既有 bug：gate 缓冲区与 stride 按全宽 `boundary_dim`
      算，而投影其实是半宽。span 路径走 `score_document_candidates` 从不调 proposer，
      唯一覆盖它的 `score_explicit_spans_full_parity` 又是 env-gated —— 所以一直没暴露。
      修完后该测试**由红转绿**（对着 reference 录制的 fixture 匹配）。
- [x] e2e 测试 `gliner2_5_choice_decode_parity::end_to_end` **通过**（model-backed，
      11 个字段）。分数容差 `1e-1` 而非 fixture 用的 `1e-4`：reference 在**整个**
      `[1, Q, C, 2]` batch 上算（`inside_prefix_mean` 与 proposer 兼容先验跨 query 池化），
      本 port 按字段切片单 query 改变了 f32 归约宽度，漂移 5e-3..5e-2。
      **逻辑仍精确**：谁在场、顺序、list/scalar、谁胜出、gate 判定全部逐位一致。
- [x] `decode_choice_fields` 在打分数量不足时**返回 Err 而不是少报**：
      静默少报会在唯一没有 model-backed 测试的路径上给出「空字段」这种错答案。

**ground truth 怎么解决的**（原以为是个死结）：12 个模型没有一个 schema 用过 `choices`，
但 reference 愿意解任何声明了 `choices` 的 schema —— 所以 oracle 直接**声明一个让
reference 自己解**，数字来自真 encoder + 真 head。已生成 `choice-decode-golden.json`。

**已知精度边界**（非 bug，是「按字段切片单 query」的设计取舍）：分数比 reference
低 5e-3..5e-2。下一个改进是**一次 `score_explicit_spans` 打完一个 group 的所有 choice**，
保持 batch 宽度以对齐归约顺序。

### 🟠 F-1b record 路径的 `choices` 前缀回退（**未做，F-1 剩下的最后一块**）

`_record_local_choice_mentions`（文档级字面量归属，`engine.py:595`）**已完成**，
见上表。但它只是**优先路径**。当文档里**没有**任何 choice 字面量时
（`has_literal_choices == false`），reference 回退到**给 schema prefix 里的 enum token
打分**（`engine.py:1080-1130`）。回退路径要求：

- [ ] record head 为 `choices` 字段把**候选集换成 prefix enum token**，而不是共用
      文档 candidate pool（`build_record_group` 现在对所有字段都用同一个 pool）
- [ ] 命中 token 的 surface 从 prefix token 解析（`prefix_choice_by_token`），
      **不能**走 `token_boundaries_to_character_offsets` —— 它只映射文档范围内的
      token，prefix token 落在 `offset` 之前，会解不出字符
- [ ] 概率取 `min(candidate_probability, assignment_probability)`（有 assignment 时）
- [ ] 再过 per-field threshold + `validators`（注意此处 validator 校验的是
      **choice 字面量本身**，不是文档 span）
- [ ] reference 自己的注释记录了这个回退为何重要：不修的话每个 record 都会退回
      `_decode_choice_field` 的文档级打分，导致**所有 record 拿到完全相同的
      choice 和 confidence**，与「哪个 record 真正提出了这个 choice」无关

**零 ground truth**：现有 12 个模型没有一个 schema 用过 `choices`，
且 record + choices 的组合没有 oracle。开工前必须先造 fixture。

**engine 坑（已踩）**：reference 的 `(?<!\w)choice(?!\w)` 用 `re.escape(choice)`
先转义，**且 `regex` crate 不支持 lookaround**，`\b` 也不等价
（choice 含标点时语义不同）。正确做法是「转义后的字面量 + 手工检查两侧非词字符」，
不能整词匹配 —— 那样 `a.b` 这种带标点的 choice 会漏。

### 🟠 F-2 `entity_attributes` / `AttributeGroup`（工作量最大，无 ground truth）

`schema.py:63-80`、`schema.py:330-392`。属性组用**独立 sigmoid 多标签**解码，
有 `qualify_labels` / `applies_to` / 独立 `threshold`。

**难在两个架构机制不同**（不是同一特性的两份实现）：
- span：复用同一 span 处已有的 `raw_logits`（`runtime.py:788-883`）
- boundary：只对**保留的** span 调 `score_explicit_spans`（`engine.py:379-485`）

校验规则也不轻：组名非空且不撞 `{text,confidence,start,end}`、label 组间不重复、
属性 label 不得与实体 label 相撞（否则要求 `qualify_labels=True`）。
`to_dict` **完全不序列化** `entity_attributes`，`from_dict` **没有这条路**。

**零 ground truth**，且要同时覆盖两套机制。先造 fixture 再动手。

### ✅ F-3 长文本 chunk + merge 的**内核**（纯函数部分完成）

`split_text_into_chunks` + `merge_chunk_results` + `remap_result_spans` +
`_strip_span_metadata` 全部完成（`long_document.rs`，12 个测试）。这些是
`chunking.py` 里与模型无关的纯函数，fixture 为真值表。

**尚未接线**：`extract_long` / `classify_text_long` 等 10 个 `*_long` 入口本身还没调用
这套内核 —— 它们在 `runtime.py` 里，把 chunk 文本逐个喂给 `extract` 再合并，
而本 port 的 `extract` 入口尚未接上长文档驱动。这是一个独立的接线步骤，
不涉及 kernel，但需要 CLI/HTTP 层的 10 个入口。

已钉的性质：多数投票（平票取**最早** chunk，不是最高分）、classification 取最大
confidence、span surface **从原文档重切**（不携带 chunk 里的 text）、非 span 项按
**忽略 confidence** 的 canonical key 去重、enum/choice 字段在无 confidence 时塌成裸字符串
（这是长文本路径独有的形状）、空文档仍产出一个 chunk。

`chunking.py` 整套。10 个 `*_long` 方法与短文本路径的**六处差异**：
1. 按**词**切窗（`chunk_size=384` / `chunk_overlap=64`）
2. 内部**强制** `include_confidence=True, include_spans=True`，与调用方 flag 无关
3. 强制 `format_results=True`（`batch_extract_long` 直接 `ValueError`）
4. char offset 加 `chunk.start_char`，且 surface 从**原文**重新切
5. merge 规则分类型：分类取 max confidence、裸字符串走**多数投票**（平票取最早的 chunk）、
   list 拼接去重、dict 递归
6. 非 span 项按**忽略 confidence** 的 canonical key 去重，保留高 confidence 的那个

依赖：char offset 需要 word→char 映射，`word_splitter="char"` 改变了窗口边界
（`runtime.py:1334` 把模型的 splitter 传进去）。

### 🟡 F-4 relation 4 阶段 dedup canonicalizer

`engine.py:899-1002` `_deduplicate_relation_edges`，`relations.rs` 里确认没有：
1. 单侧包含关系归一到最长 mention
2. 精确 `(h0,h1,t0,t1)` 去重，保留最高分
3. **case / 空白折叠后**的语义文本去重，平票按 token 距离再按分数
4. token 子集支配关系剔除
最终按 `(head_start, tail_start, -score)` 排序。

### ⚪ F-5 已确认**不做**的（记录理由，避免以后重复调查）

| 项 | 理由 |
|---|---|
| `occurrence_policy`（4 个取值） | **只在建 gold target 时用，推理零影响**。若「实现」它，那是新功能不是对齐 |
| `joint_ie/` 与 `classification/` 子系统 | 是独立公开库面（`Classifier.from_pretrained` / `JointIE.from_pretrained` → `AutoExtractor`），不是 `span`/`boundary` 架构。带完整约束 DSL + beam/exact 解码 + 温度标定，**建议另开 PR** |
| `Schema.to_dict()` / `from_dict()` 往返 | reference 自身就不对称（丢 `threshold`/`validators`/`cls_threshold`/`class_act`/`prompt`/`examples`，`entity_attributes` 完全无路径）。对齐它=复制缺陷 |
| 训练侧旋钮（`SamplingConfig` 15 个 knob、LoRA、`push_to_hub`） | 训练/框架层 |
| `span` 架构的 NER 头（`span_rep`/`count_embed`/`count_pred`） | 转换器既定范围，**本来就没转**（`weights.rs:6`），不是遗漏 |

### 本轮抓到的三个真 bug（都是「模型适配完成」也发现不了的）

1. **`\w` 语义不一致**（`e9904c4` 修）：Python 的 `\w` = `L*`+`N*`+`_`；
   rust `regex` 的是 `\p{Alphabetic}`+`\p{M}`+`\p{Nd}`+`\p{Join_Control}`+`\p{Pc}`。
   **两个方向都错**：`cafe`+U+0301 被误合成一个 token，U+2160/U+00BD 被误拆。
   存活原因：12 个模型的测试文本里**没有任何组合符**。
2. **structure validator key 错配**（`3be43a0` 修）：`structure.rs` 用裸字段名查，
   `score_structures` 用 `<group>.<field>` 建表 —— 永远查不到，且裸名会在两个 group
   声明同名字段时撞车。**编译通过、测试全绿、特性完全无效**。
3. **`sep_index` 在有 prefix 时算错**（本轮修）：`combined.len() - 1 - words.len()`
   这个公式假设 `[SEP_TEXT]` 紧邻 words。插进 prefix 后它落进 words 内部，
   于是 **prefix 的 11 行 + 前 11 个词被一起跳过**，`text_word_first_positions`
   长度直接腰斩（22 → 11），所有 span 索引平移。
   特征单测（只查 `input_ids` 顺序）抓不到，**model-backed e2e 才抓到**。
4. **`score_spans` 的 `batch` 是「样本数」不是「样本下标」**（本轮踩到）：
   返回 `0..batch * q_count * c` 个元素，传 `0` 得到 0 个候选 —— 而调用点读起来
   像是索引。参数名有歧义，值得改。
5. **`regex` crate 不支持 lookaround**（本轮踩到）：reference 的
   `(?<!\w)choice(?!\w)` 无法直译，`\b` **不是等价物** —— choice 以标点开头/结尾时
   语义不同。整词匹配 `[\p{L}\p{N}_]+` 再比较也错：那会让 `a.b` 这类带标点的
   choice 整个漏掉（`a.b` 里 `.` 打断词 run）。正解是「`re.escape` 后的字面量 +
   手工检查两侧非词字符」。第一版用整词匹配，`a_choice_is_matched_literally`
   立刻抓到了。
6. **`overlap.rs` 的 `usize` 下溢**（`1c1466a` 修）：reference 的
   `bisect_right(ends, start, 0, position) - 1` 对 `usize` 减 1。release 下靠二次回绕
   碰巧正确，**debug 的溢出检查会让 3 个 `overlap_resolution_parity` 测试失败**。
   之前记的「gliner 82/82」是 release 下测的。

---

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
- **特性层**（上方「特性层待办」一节）开工前**必须先写 oracle**，与模型适配同一条纪律：
  凭「模型应该支持某特性」去猜会漏掉一半。已完成的 5 项全部先跑 oracle 拿 reference 的真实答案，
  三个真 bug 也是这么抓到的 —— 它们在「模型 byte-exact 完成」的状态下完全不可见
- 任何"flag 共用 vs 另开"的决策都要在本文件写理由 + commit hash，半年后回头看不会懵
- "已下载模型但未完成适配"的中间状态用 ⏸ 标记，列在对应 TODO 下方
- **族分类写错比不写更糟**——每条开工前必读 config 复核 `architecture` 与 `model_name`