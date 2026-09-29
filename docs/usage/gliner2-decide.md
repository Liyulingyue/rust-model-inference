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

## 输入构造

`token_pooling == "first"`，`extract_embeddings_from_batch` 走 gather 快路径：
- `batch.text_word_indices` —— text 各 word 的首 token 下标
- `batch.schema_special_indices` —— schema 特殊 token（`[L]` 等）的下标

输入序列大致是 `[CLS] text tokens [SEP] [L] label1 [L] label2 ...`，
prompt 串由 `SchemaTransformer` 生成。**精确模板还没扒**，写转换器/推理前需要从
`gliner2/processor.py` 的 `collate_fn_inference` 和 `processing/layouts.py`
里抠出来。这是实现前必须先确定的第一个点。

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
5. **下载 tokenizer 文件**：ModelScope 的 `GLiNER2.5-Decide` 只有
   `model.safetensors` + `config.json`，tokenizer 要从底座
   `microsoft/deberta-v3-large` 或 `fastino/gliner2-large-v1` 取。
   `AutoTokenizer.from_pretrained(repo_or_dir)` 依赖那些文件。

## 实现的自然切分

建议分两步，各自可验收：

- **第一步**：DeBERTa-v3 encoder（24 层，含 SP tokenizer）+ F32/Q8_0 GGUF 转换。
  验收标准：给定同一段文本，embedding 和 HF 参考逐位对齐。
  这一步就是 laya PR 的等价工作量。
- **第二步**：SchemaTransformer prompt 构造 + classifier 头 + `classify_text`
  CLI/HTTP。验收标准：README 里的 customer support intent 等例子复现。

## TODO(clm-style)

代码还没落，所以只有文档戳。真正开始实现前必须先确定：

- `[ ]` `collate_fn_inference` 产出的精确 token 序列（`[L]` / `[P]` 的位置和 id）
- `[ ]` `classifier.1` 到底是 GELU 还是 ReLU（未序列化，要看模型代码）
- `[ ]` DeBERTa-v3 attention 的 rope/相对位置细节（`glm` 系无 RoPE，用
      `rel_embeddings`，但 `should_apply_ln` / `conv_kernel_size` 等开关要确认；
      DeBERTa-v3 的 embedding 段还有一层 conv，本 checkpoint 的 tensor 列表里
      **没有** conv 权重，需要确认是否被裁掉）
- `[ ]` 下载 tokenizer 文件并确认 SP unigram 参数
