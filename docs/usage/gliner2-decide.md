# GLiNER2.5-Decide 适配侦察记录

状态：**侦察完成，未开始实现**。这篇记录架构事实和实现边界，避免下次重新扒。

模型：`fastino/GLiNER2.5-Decide`（ModelScope，safetensors F32 1.9GB）

## 它是什么

340M 英文分类器，底座 `microsoft/deberta-v3-large`（300M）+ 一个 MLP 分类头。
`classify_text(text, {head: [labels]})` 一次前向给出每个 label 的 logit。
不生成文本、不推理、不解释。README 明确说 "No prompt template. No generated tokens."

benchmark（fastino/fast-decisions，17 域各 300 样本 exact-match）：
GLiNER2.5-Decide 60.2% > GLiNER2.5-Decide-1B 59.6% > JevK5 57.6% > SemIf(Qwen3.5-4B) 56.4%

## 架构事实（已从 checkpoint 验证）

`config.json`：
- `architectures: ["SpanExtractor"]`, `architecture: "span"`, `config_version: 3`
- `model_name: "microsoft/deberta-v3-large"`
- `token_pooling: "first"`
- `max_width: 8`, `span_head.span_mode: "markerV0"`（NER 用，classify 不需要）
- `use_moe: false`

tensor（419 个，按前缀分组）：

| 前缀 | 数量 | classify 需要？ |
|---|---|---|
| `encoder.embeddings.word_embeddings.weight` [128011, 1024] | 1 | 是 |
| `encoder.embeddings.LayerNorm.{weight,bias}` [1024] | 各 1 | 是 |
| `encoder.encoder.rel_embeddings.weight` [512, 1024] | 1 | 是（disentangled attention 关键） |
| `encoder.encoder.layer.N.attention.self.{query,key,value}_proj.{weight,bias}` | 24 层 | 是 |
| `encoder.encoder.layer.N.attention.output.dense.{weight,bias}` + `output.LayerNorm` | 24 层 | 是（ST-transposed） |
| `encoder.encoder.layer.N.intermediate.dense.{weight,bias}` [4096, 1024] | 24 层 | 是（FFN up） |
| `encoder.encoder.layer.N.output.dense.{weight,bias}` + `output.LayerNorm` | 24 层 | 是（FFN down） |
| `encoder.encoder.LayerNorm.{weight,bias}` | 各 1 | 是（最终层范数） |
| `classifier.{0,2}.{weight,bias}` | 2 层 | 是 |
| `span_rep.*` | 6 | 否（NER span 头） |
| `count_embed.*` / `count_pred.*` | 若干 | 否（NER 计数头） |

维度：hidden 1024，24 层，vocab 128011，FFN 4096，rel embeddings 512 桶。

## 分类头形状

```
classifier.0: Linear(1024 -> 2048)   [2048, 1024]
classifier.1: 激活（未序列化，代码里是 Sequential 的中间层）
classifier.2: Linear(2048 -> 1)      [1, 2048]
```

推理路径（`gliner2/classification/scoring.py::batch_score`）：

```python
encoded = model.encoder(input_ids, attention_mask).last_hidden_state
_, schema_embs = processor.extract_embeddings_from_batch(encoded, input_ids, batch)
for each task:
    embs = schema_embs[t_idx]
    label_embs = embs[1:]                       # 丢掉 [P] prompt 行
    logits = model.classifier(torch.stack(label_embs)).squeeze(-1)
```

即：**encoder 前向 → 取每个 `[L]` marker 位置的隐状态 → 过 classifier → 每个 label 一个 logit**。
`span_rep` / `count_pred` / `count_embed` 在这个路径上完全不参与。

## 输入构造（已确定）

`token_pooling == "first"`，`extract_embeddings_from_batch` 走 gather 快路径：
- `batch.text_word_indices` —— text 各 word 的首 token 下标
- `batch.schema_special_indices` —— schema 特殊 token（`[P]`/`[L]`）的下标

调用链：`collate_fn_inference` → `_collate_batch` → `_transform_record` →
`_infer_from_json` → `_build_outputs` → `_format_input_with_mapping`。
源码位置：`/tmp/gliner2-src/gliner2/processor.py`。

### 1. text 侧（`WhitespaceTokenSplitter`，默认 `word_splitter="whitespace"`）

正则（`re.VERBOSE | re.IGNORECASE`，`lower=True` 只对 token 值小写）：

```python
r"""(?:https?://[^\s]+|www\.[^\s]+)
|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
|@[a-z0-9_]+
|\w+(?:[-_]\w+)*
|\S"""
```

`_normalize_text` 先补句末标点：空串→`.`；不以 `.`/`!`/`?` 结尾→追加 `.`。

### 2. schema 侧（`_transform_schema`）

```python
prompt_str = task                                  # 或 f"{task}: {prompt}"
for label, desc in label_descriptions:             # example_mode == "both"
    prompt_str += f" [DESCRIPTION] {label}: {desc}"
for inp, out in examples:                          # 只保留 out in labels
    prompt_str += f" [EXAMPLE] {inp} [OUTPUT] {out}"

tokens = ["(", "[P]", prompt_str, "("]
for field_name in fields:                          # 声明顺序，推理不 shuffle
    tokens.extend(["[L]", field_name])
tokens.extend([")", ")"])
```

推理时 `example_modes = ["both"]`（`is_training=False`），所以描述和 few-shot
**总是同时**进 prompt；`sampling=None` 时 label 顺序与 schema 声明一致。

### 3. 拼装（`_format_input_with_mapping`）

```python
combined = []
for struct in schema_tokens_list:
    combined.extend(struct)
    combined.append("[SEP_STRUCT]")
if combined: combined.pop()          # 去掉最后一个多余的 [SEP_STRUCT]
combined.append("[SEP_TEXT]")
combined.extend(text_tokens)
```

**注意：没有 `[CLS]` / `[SEP]`。** `input_ids = tokenizer.convert_tokens_to_ids(subwords)`
直接把 subword 序列送进 DeBERTa，不走 `build_inputs_with_special_tokens`。

marker 槽位（`schema_marker_orig_indices`）：每段里 `offset+1`（`[P]`）以及
`range(4, len(struct)-2, 2)`（全部 `[L]`）。记录的是 **subword 下标**，
顺序为 `[P], [L]_0, [L]_1, ...`；`embs[1:]` 丢掉 `[P]` 行后与 label 顺序一一对应。

### 4. 单个 token 的 subword 化

`sub_tokens = tokenizer.tokenize(token)`。对 10 个 GLiNER special token
（`[P]`/`[L]`/`[SEP_TEXT]`/`[SEP_STRUCT]`/…），`SchemaTransformer.__init__`
先 `add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})`，
所以 `PreTrainedTokenizer.tokenize` 的 `tokens_trie` 会整块切出，恒为 1 个 id。
`(`、`)`、`,`、`|` 不是 added token，走 SentencePiece。

`DebertaV2Tokenizer` 自身不实现 `tokenize`/`convert_tokens_to_ids`，落到
`PreTrainedTokenizer`：

- `tokenize(t)`：trie 切 added token → 否则 `spm.encode(t, out_type=str)`
  （`split_by_punct=False`，所以没有 DeBERTa 的数字+逗号特殊处理）
- `convert_tokens_to_ids(tok)`：先查 `_added_tokens_encoder`（`[P]`→128003 等），
  否则 `spm.PieceToId(tok)`

## 分类头激活：ReLU

`gliner2/models/span/model.py` 里 classifier 是
`create_mlp(input_dim=1024, intermediate_dims=[2048], output_dim=1, dropout=0.,
activation="relu", add_layer_norm=False)`，
而 `create_mlp` 的顺序是 `Linear → (LayerNorm) → act → (Dropout)`，所以：

```
classifier.0  Linear(1024 -> 2048)
classifier.1  ReLU
classifier.2  Linear(2048 -> 1)
```

与 safetensors 里只有 `classifier.0` / `classifier.2` 两组权重一致。

## 解码（`inference/runtime.py::_extract_classification_result`）

```python
logits = classifier(embs[1:]).squeeze(-1) / temperature   # temperature 默认 1.0
act = class_act or ("sigmoid" if multi_label else "softmax")
multi_label: 取所有 prob >= cls_threshold 的 label；空则回退 argmax
否则:        argmax 的 label + 它的 prob
```

`classify_text` 的入参形态（`runtime._classification_schema`）：

```python
{head: [labels]}                                   # single-label
{head: {"labels": [labels],
        "multi_label": False, "cls_threshold": 0.5}}
{head: {"labels": {name: description, ...}}}       # 带描述
{head: {"labels": [...], "prompt": "..."}}         # 附加指令
```

`{"labels": {"name": "desc"}}` 时 `label_names = dict.keys()`，描述进
`label_descriptions` 参与 prompt 拼接。

## DeBERTa-v3-large 编码器配置（已确认）

底座 `microsoft/deberta-v3-large` 的 `config.json`：

```json
{"model_type": "deberta-v2", "hidden_size": 1024, "num_hidden_layers": 24,
 "num_attention_heads": 16, "intermediate_size": 4096, "hidden_act": "gelu",
 "layer_norm_eps": 1e-7, "relative_attention": true, "position_buckets": 256,
 "max_position_embeddings": 512, "max_relative_positions": -1,
 "position_biased_input": false, "type_vocab_size": 0,
 "norm_rel_ebd": "layer_norm", "pos_att_type": "p2c|c2p", "share_att_key": true}
```

这解释了 checkpoint 的 tensor 形状与"缺件"：

- `position_biased_input=false` → **没有** `position_embeddings`
- `type_vocab_size=0` → **没有** `token_type_embeddings`
- `max_relative_positions=-1 → 512`，`pos_ebd_size = position_buckets*2 = 512`
  → `rel_embeddings.weight [512, 1024]`
- `norm_rel_ebd="layer_norm"` → `encoder.encoder.LayerNorm.{weight,bias}`
- `conv_kernel_size` 缺省 0 → **没有** `ConvLayer`（DeBERTa-v1 才有）

前向（transformers 4.48.1 `models/deberta_v2/modeling_deberta_v2.py`，
reference 在 `/tmp/tfdl/x/transformers/models/deberta_v2/`）：

1. `embeddings`：只查 `word_embeddings` → LayerNorm(eps=1e-7) → 乘 mask
2. 每层：
   - `rel_embeddings` 过 `LayerNorm` 得到 `rel_emb`（`norm_rel_ebd`）
   - `attn = Dense(softmax(Dense_self_attn(x) + x))`（**残差先加，再 LayerNorm**，
     即 ST-transposed）
   - `out = LayerNorm(Dense(GELU(Dense(attn))) + attn)`（同样是 ST-transposed）
3. `score_scale = 1 / sqrt(head_dim * scale_factor)`，`scale_factor = 1 + |pos_att_type| = 3`
   （content-content / c2p / p2c 三项共用同一个 scale，**不是** `1/sqrt(64)`）
4. 相对位置（`make_log_bucket_position`，bucket=256、max=512）：

```text
c2p_pos = clamp(relative_pos + 256, 0, 511)
p2c_pos = clamp(-relative_pos + 256, 0, 511)
pos_key   = transpose_for_scores(key_proj(rel_emb[:512]))     # share_att_key
pos_query = transpose_for_scores(query_proj(rel_emb[:512]))
c2p = gather(Q · pos_key^T, c2p_pos) / scale
p2c = gather(K · pos_query^T, p2c_pos)^T / scale
```

5. 推理是 `model.eval()`，两个 dropout 都是恒等。batch=1 时 `attention_mask`
   全 1，mask 只影响 padding 位置。

## 主要工作量 / 风险

1. **DeBERTa-v3 encoder（最大头）**。disentangled attention 是真正的新数学：
   score = content·content + content·position + position·content，
   靠 `rel_embeddings`（512×1024）做相对位置项。仓库现有的 attention（llama
   GQA / qwen / gemma4 / lfm2 / falcon / nemotron）都是标准 dot-product，
   没有可复用的 kernel。
2. **ST-transposed**：LayerNorm 在残差相加**之前**，不是之后。和我们已经习惯的
   pre-norm/post-norm 都不一样，是 DeBERTa 的特色。
3. **SentencePiece tokenizer**：仓库现在只有 BPE（qwen/llama 家）和 laya 那套。
   DeBERTa-v3-large  vocab 128011 是 SP unigram。`tokenizer.ggml.model` 会是新值。
4. **safetensors → GGUF 转换器**：从零写。仓库现有转换器（breeze / dots /
   dreamx / laya / yue2 / qwen_drive）都是 safetensors→GGUF，可以照 `laya` 的
   结构抄，laya 是最接近的先例（也是 encoder + 决策头）。

已解决：

- tokenizer 文件已下载到 `models/GLiNER2.5-Decide/`
- 无 conv 权重是**正常**的：v3 的 `conv_kernel_size` 缺省 0
- `classifier.1` = ReLU（见上）
- 精确 token 序列已确定（见上）


## ## Tokenizer (determined)

DebertaV2Tokenizer, vocab_type spm, do_lower_case false, split_by_punct
false.  SentencePiece unigram, case sensitive.  The checkpoint on
ModelScope carries only model.safetensors and config.json; the tokenizer
pack comes from base model fastino/gliner2-large-v1 and is already in
models/GLiNER2.5-Decide/:

- spm.model, 2.4 MB (the real vocab model)
- tokenizer_config.json / special_tokens_map.json

Special token ids:

| token | id | | token | id |
|---|---|---|---|---|
| [PAD] | 0 | | [SEP_STRUCT] | 128001 |
| [CLS] | 1 | | [SEP_TEXT] | 128002 |
| [SEP] | 2 | | [P] | 128003 |
| [UNK] | 3 | | [C] | 128004 |
| [MASK] | 128000 | | [E] | 128005 |
| | | | [R] | 128006 |
| | | | [L] | 128007 |
| | | | [EXAMPLE] | 128008 |
| | | | [OUTPUT] | 128009 |
| | | | [DESCRIPTION] | 128010 |

[L] is the label marker and [P] the prompt marker; scoring drops the [P]
row via embs[1:].  bos and cls are both [CLS], eos is [SEP].


实现的自然切分

建议分两步，各自可验收：

- **第一步**：DeBERTa-v3 encoder（24 层，含 SP tokenizer）+ F32/Q8_0 GGUF 转换。
  验收标准：给定同一段文本，embedding 和 HF 参考逐位对齐。
  这一步就是 laya PR 的等价工作量。
- **第二步**：SchemaTransformer prompt 构造 + classifier 头 + `classify_text`
  CLI/HTTP。验收标准：README 里的 customer support intent 等例子复现。

## TODO(clm-style)

- `[x]` `collate_fn_inference` 产出的精确 token 序列
- `[x]` `classifier.1` 激活函数 → ReLU
- `[x]` DeBERTa-v3 attention / conv 细节
- `[x]` 下载 tokenizer 文件
- `[ ]` 探针 `spm.model` protobuf：normalizer_spec、byte_fallback、piece score 类型
- `[ ]` `tools/converter/gliner/convert_gliner.py`
- `[ ]` Rust SentencePiece unigram 分词
- `[ ]` Rust DeBERTa-v3 encoder 前向
- `[ ]` Rust schema prompt 构造 + word splitter
- `[ ]` classifier 头 + softmax/sigmoid 解码
- `[ ]` CLI (`--gliner2-decide`) + HTTP endpoint
- `[ ]` golden 向量：抓 HF 参考的 input_ids / logits 落盘

