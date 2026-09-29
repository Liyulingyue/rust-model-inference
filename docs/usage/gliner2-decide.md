# GLiNER2.5-Decide 用法

GLiNER2.5-Decide 不是生成模型。它是 DeBERTa-v3-large encoder 上面接一个两层
MLP 分类头，**一次前向给出每个 label 的 logit**，不生成 token、不推理、不解释。

```
logit(label) = classifier(hidden_at_[L]_marker)(label)
```

label 集合在**调用时**传进去，不是模型里烤死的。ModelScope 上的
`fastino/GLiNER2.5-Decide` 是 340M 参数、1.9GB safetensors。

## 1. 准备

```bash
# 1) 权重 + tokenizer
models/.venv/bin/modelscope download --model fastino/GLiNER2.5-Decide \
    model.safetensors config.json --local-dir ./GLiNER2.5-Decide
models/.venv/bin/modelscope download --model fastino/gliner2-large-v1 \
    spm.model tokenizer_config.json special_tokens_map.json \
    --local-dir ./GLiNER2.5-Decide

# 2) 转 GGUF（F32，1.75GB）
models/.venv/bin/python tools/converter/gliner/convert_gliner.py \
    models/GLiNER2.5-Decide models/GLiNER2.5-Decide/gliner2-decide-f32.gguf
```

SentencePiece 词表、piece score、piece type、`nmt_nfkc` charsmap 和三个
normalizer flag 全部**内嵌进 GGUF metadata**，所以跑起来只需要一个 `.gguf`，
不需要 `spm.model` 边车文件。

## 2. 命令行

没有专用 binary，挂在既有 JEV flag 家族下（和 CLM 同级）：

```bash
./target/release/rust-model-inference \
  --model models/GLiNER2.5-Decide/gliner2-decide-f32.gguf \
  --jev --gliner2-decide \
  --jev-context "My subscription renewed on April 15 for ¥5,400 after the service was already down. Can I get that charge refunded?" \
  --jev-question intent \
  --jev-option order_status --jev-option refund_request \
  --jev-option cancel_subscription --jev-option update_payment \
  --jev-option login_problem --jev-option shipping_delay \
  --jev-option bug_report --jev-option speak_to_human --jev-option other
```

实测输出（复现 ModelScope README 那个例子的 `{"intent": "refund_request"}`）：

```
GLiNER2: deberta-v3 24x1024x16, 1 task(s)
Q: intent
  A. order_status  p=0.0002  score=-4.8493
  B. refund_request  p=0.9971  score=3.5138
  C. cancel_subscription  p=0.0006  score=-3.8711
  ...
  -> choice: B
```

`--jev-question` 是**头的名字**，`--jev-option` 是它的 label 集合。多个
`--jev-question` 就是多个头，一次前向全出（对应 README 的 “Several decisions
at once”）：

```bash
  --jev-question intent   --jev-option maintenance --jev-option room_change --jev-option billing \
  --jev-question priority --jev-option low --jev-option normal --jev-option high --jev-option urgent \
  --jev-question needs_human --jev-option yes --jev-option no
```

每个 question 一行 `Q:`，各自带自己的概率和 choice。

### `--gliner2-schema`：直接吃参考实现的 `classify_text` 参数

`--jev-option` 只能表达“纯 label 列表的单标签头”。要多标签、阈值、label 描述、
指令、few-shot，用 `--gliner2-schema`，它就是参考实现 `classify_text(text, tasks)`
第二个参数的 JSON：

```bash
./target/release/rust-model-inference \
  --model models/GLiNER2.5-Decide/gliner2-decide-f32.gguf \
  --jev --gliner2-decide \
  --jev-context "Battery dies before lunch, but the keyboard and the screen are the best I have used on a laptop." \
  --gliner2-schema '{
    "aspects": {"labels": ["battery","keyboard","screen","camera","price","support"],
                "multi_label": true, "cls_threshold": 0.4}
  }'
```

支持的四种形态（与参考 `runtime._classification_schema` 一一对应）：

| 形态 | 含义 |
|---|---|
| `["a","b"]` | 单标签，softmax |
| `{"labels": [...], "multi_label": true, "cls_threshold": 0.4}` | 多标签，独立 sigmoid |
| `{"labels": {"a": "描述", "b": "描述"}}` | label 带描述，描述进 prompt |
| `{"labels": [...], "prompt": "...", "examples": [["in","out"]]}` | 附加指令 / few-shot |

还接受 `class_act`（`softmax` / `sigmoid` / `auto`，默认 `auto` = 多标签走
sigmoid、否则 softmax）和 `temperature`（默认 1.0，除在 logit 上）。

`--gliner2-schema` 优先于 `--jev-question` / `--jev-option`。两边都不给就报错。

## 3. 服务端模式

`--gliner2-decide` 是启动参数（和 `--clm-head` 同级），请求体不带模型路径：

```bash
./target/release/rust-model-server \
  --model models/GLiNER2.5-Decide/gliner2-decide-f32.gguf \
  --gliner2-decide --host 0.0.0.0 --port 8080 --threads 8
```

启动日志打 `mode=gliner2`。**只注册 `/v1/jev/score`**，其余一律 404。

请求体和 JEV logit 打分完全一致，一个 question 就是一个头：

```bash
curl http://127.0.0.1:8080/v1/jev/score -H 'Content-Type: application/json' -d '{
  "context": "Battery dies before lunch, but the keyboard and the screen are the best I have used on a laptop.",
  "questions": [{"text": "aspects",
                 "options": ["battery","keyboard","screen","camera","price","support"],
                 "multi_label": true, "cls_threshold": 0.4}]
}'
```

`multi_label` / `cls_threshold` / `prompt` / `descriptions` 是 GLiNER2 专有的
可选字段（`descriptions` 与 `options` 等长，按序对应）；其他 JEV 后端忽略它们。

```json
{
  "mode": "single",
  "results": [{
    "mode": "multi_select", "labels": ["A","B","C","D","E","F"],
    "descriptions": ["battery","keyboard","screen","camera","price","support"],
    "values": [1.6774151, 4.9915481, 4.0002327, -7.0071478, -6.8949332, -5.6953726],
    "probabilities": {"A": 0.8426, "B": 0.9933, "C": 0.9820,
                      "D": 0.0009, "E": 0.0010, "F": 0.0034},
    "choice": "B", "selected": ["battery", "keyboard", "screen"],
    "confidence": 0.5227, "entropy": 0.2013, "margin": 0.0112, "prefill_ms": 3068
  }]
}
```

`selected` 就是参考实现会返回的那个列表。`values` 现在对所有 JEV 模式都带
（以前只有 `score` 模式带），因为 GLiNER 每个 label 都有真实 logit 值。
`/v1/jev/grouped` 返回 404：grouped 做 per-group softmax，GLiNER 的每头独立
归一化没有对应物。

## 4. prompt 布局（不看这个会排错序）

这是最容易踩的坑，全部来自 `gliner2/processor.py`：

- **没有 `[CLS]` / `[SEP]`。** `input_ids` 直接是 subword 序列，
  `build_inputs_with_special_tokens` 根本没被调用。
- **单头时没有 `[SEP_STRUCT]`。** 拼装时每个 schema 后面都加一个
  `[SEP_STRUCT]`，然后 `pop()` 掉最后一个——所以只有多头 prompt 才带分隔符。
- **分类行就是 `[L]` marker 自己的隐状态**，`embs[1:]` 丢掉 `[P]` prompt 行。
- **描述和 few-shot 总是同时进 prompt。** 推理时 `example_mode == "both"`。
- **text 先补句末标点**（`_normalize_text`：空串→`.`，不以 `.!?` 结尾→追加 `.`），
  然后按 GLiNER 自己的 word splitter 切词并小写，再逐词送 SentencePiece。

单头、单 label 的完整序列长这样：

```
▁(  [P]  ▁intent  ▁(  [L]  ▁order _ status  [L]  ▁refund _ request  ...  ▁)  ▁)  [SEP_TEXT]  text…
```

对照实现（`src/models/gliner/prompt.rs`）：

```python
tokens = ["(", "[P]", prompt_str, "("]
for field_name in fields:          # 声明顺序，推理不 shuffle
    tokens.extend(["[L]", field_name])
tokens.extend([")", ")"])
```

`prompt_str` 是 `task` 或 `f"{task}: {prompt}"`，后面按声明顺序拼
`" [DESCRIPTION] {label}: {desc}"` 和 `" [EXAMPLE] {in} [OUTPUT] {out}"`。

### 会被拒绝的字符串

`_RESERVED`（`[P] [L] [C] [E] [R] [DESCRIPTION] [EXAMPLE] [OUTPUT] ( )`）
出现在 task 名、label 名、描述、指令或 few-shot 里会**直接报错**，不是静默
截断。原因是这些串原样进 prompt，一个多余的 `[L]` 会把后面的 logit 整体错位
一位——参考实现在 `gliner2/classification/schema.py` 里也是这么防的。

## 5. DeBERTa-v3 encoder 细节

`src/models/gliner/compute.rs`。四个和 `bert` / `jina-bert-v2` 不一样的地方，
每一个都会把结果改坏：

1. **Disentangled attention。** score = `content·content + content·position +
   position·content`，三项**共用同一个** `1/sqrt(head_dim * scale_factor)`，
   `scale_factor = 1 + |pos_att_type| = 3`。不是 `1/sqrt(64)`。
2. **位置向量来自本层自己的 `query_proj` / `key_proj`**（`share_att_key`），
   而 `rel_embeddings` 表由 encoder 统一 LayerNorm 一次后传给所有层。**不是**
   所有层共用第一层的投影——共用会让 layer 0 之后的每一层全错。
3. **ST-transposed 残差。** 子层是 `LayerNorm(f(x) + x)`，norm 看的是**和**。
4. **c2p 和 p2c 用同一张位置下标表。** 参考代码写的是
   `c2p_pos = clamp(rel + att_span)` 和 `p2p_pos = clamp(-r_pos + att_span)`，
   看着是两个表，但 p2c 那边多了一次 `gather(...).transpose(-1, -2)`，轴一换
   就变成 `clamp(-(s - t) + att_span)`，和 c2p 完全一样。照字面写会多取一次负号，
   每一层就开始漂。

相对位置分桶（`make_log_bucket_position`，bucket=256、max=512、att_span=256）：

```text
mid = 128
abs_pos = if -128 < rel < 128 { 127 } else { |rel| }
log_pos = ceil(ln(abs_pos/128) / ln(511/128) * 127) + 128
bucket  = if abs_pos <= 128 { rel } else { log_pos * sign(rel) }
c2p_pos = p2c_pos = clamp(bucket + 256, 0, 511)
```

其余配置（`microsoft/deberta-v3-large`）解释了 checkpoint 里缺哪些东西：
`position_biased_input=false` → 没有 position embedding；`type_vocab_size=0` →
没有 token_type embedding；v3 的 `conv_kernel_size` 缺省 0 → **没有** conv
（那是 v1 才有）；`norm_rel_ebd="layer_norm"` → `rel_norm` 作用在相对位置表上，
不是作用在输出上；`max_relative_positions=-1` + `position_buckets=256` →
`rel_embeddings` 是 `[512, 1024]`（`position_buckets * 2`）。

分类头是 `create_mlp(1024, [2048], 1, activation="relu")`，即
`Linear → ReLU → Linear`。所以 safetensors 里只有 `classifier.0` / `classifier.2`
两组权重，中间那层是 ReLU 而不是 LayerNorm。

## 6. Tokenizer

`DebertaV2Tokenizer` + SentencePiece **unigram**，大小写敏感，128000 个 piece。
Rust 侧实现在 `src/core/sentencepiece.rs`，包含：

- `nmt_nfkc` normalizer（Darts double-array 查表 + dummy prefix / 空白折叠 /
  空白转 `▁`）
- unigram Viterbi（`Lattice::Viterbi`，含 `has_single_node` 的 UNK 回退）
- byte fallback（`byte_fallback = true`，未知字符拆成 `<0xXX>`）
- `tokens_trie` 语义的 added-token 切分：**字符串内部的** `[DESCRIPTION]` 等
  也会被切出来。GLiNER2 把这些 marker 拼进 prompt 串，所以这一步不做就会
  拿到不同的 id。

两个坑：

- **lattice 的位置是字符位置，不是字节位置。** `▁`（U+2581）占 3 字节，
  其中间两个字节**不是** lattice 位置，不能在那里插 UNK 节点。Rust 侧按字节
  偏移建 lattice，所以只遍历字符起点。
- **`QuantizedTensor::n_rows()` 对 F32 返回 1**，所以 F32 权重的
  `Weight::n_out` 是 1。`quantize_and_matmul_with_scratch` 读的是
  `self.n_out`，直接调会在 F32 上只算一列——`compute::matmul_into` 因此自己
  拿 `kernel.f32_slice()` 驱动 SIMD 行 kernel 并传真实宽度。

## 7. 精度验证

```bash
models/.venv/bin/python /tmp/gliner_ref/dump_golden.py   # 重新生成 fixture
cargo test --profile release-fast --test gliner2_classify_parity
cargo test --profile release-fast --test gliner2_spm_parity
cargo test --profile release-fast --test gliner2_cli
```

`tests/fixtures/gliner2-decide/` 下两个 fixture 都由参考栈
（GLiNER2 的 `SchemaTransformer` + `transformers` 4.48.1 的 DeBERTa-v3 +
checkpoint 自己的头）生成：

- `classify-golden.json` — 6 个 case 的 `input_ids`、`[P]/[L]` 下标、每 label logit
- `spm-pieces.json` — 69 条字符串的 SentencePiece pieces / ids

实测 6 个 case 全部对齐，**最大 logit 偏差 7.2e-6**（F32 累加顺序差），所以
测试阈值定在 1e-4。

## 8. 性能

aarch64、8 线程、45 token 单头：**5.9s**（参考 PyTorch 同输入 2.3s）。
102 token 四头：8.2s。

大头是**每层都要重算** `pos_ebd_size × n_embd` 的两次位置投影
（24 层 × 2 × 512 × 1024 × 1024 ≈ 25.8 GMAC，比 token 侧重 4 倍）。这是
DeBERTa-v3 本身的形状，参考实现同样每层重算，跨层没有可缓存的东西。

`--threads 12` 在本机会退化到 70s+（`ComputePool` 自旋等 barrier 的既有
行为，和仓库里其他模型一致）；默认上限是 8，正常用不要手动调过。
