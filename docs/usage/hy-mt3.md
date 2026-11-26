# Hunyuan / Hy-MT3 用法

`Hy-MT3` 是腾讯混元第三代多语言翻译模型（102 种语言），2026-11-24 开源，三种规模：
**1.8B**、**7B**、**30B-A3B**（MoE，激活 3B）。

三个 GGUF 落在**两条架构**上，本仓库两条都已接入：

| 规模 | GGUF `general.architecture` | 本仓库路径 |
| --- | --- | --- |
| 1.8B | `hunyuan-dense` | `src/models/qwen3/hunyuan.rs`（复用 qwen3 trunk，与 Hy-MT2 同一条路径） |
| 7B | `hunyuan-dense` | 同上（但 tokenizer 不同，见 §3） |
| 30B-A3B | `hy_v3` | `src/models/hy_simulmt/text.rs`（`HyV3TextRuntime`），解码器在 `src/models/hy_simulmt/{config,weights,decoder,moe}.rs` |

`hy_v3` 同时是 **Hy-SimulMT** 语音翻译模型的解码器架构。两者靠
`clip.projector_type == "hyv3_speech"` 区分：带音频塔的 checkpoint 必须走
`--audio`，纯文本 checkpoint 走本文的文本入口。CLI 和 HTTP 都按这个键分流，
不会把文本模型塞进音频后端，也不会让语音模型在缺少音频时静默生成。

> 共用前置：`cargo build --release --bin rust-model-inference`。
> 30B-A3B Q8_0 约 32.8 GB，建议 `--threads` 取物理核数；1.8B/7B 用小得多的线程数即可。
> 三个模型当前都是 **CPU 路径**；`hy_v3` 没有 Vulkan 算子，严格 `--compute vulkan` 会明确报错而不是静默回退。

## 1. 翻译任务（主用途）

Hy-MT3 的用法是「翻译指令 + 源文本」一条 prompt 直接喂入。模型卡给出中英两套模板，
`target_lang` 用**语言全称**（中文 prompt 用中文名，英文 prompt 用英文名）：

```
将以下文本翻译为 {target_lang}，注意只需要输出翻译后的结果，不要额外解释：

{source_text}
```

```
Translate the following text into {target_lang}. Note that you should only output the translated result without any additional explanation:

{source_text}
```

模型卡另给术语表、风格、个性化、分隔符、结构化数据等模板，见
`models/Hy-MT3-*-GGUF/README.md`。

## 2. CLI 调用

**30B-A3B（`hy_v3` 文本入口）：**

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/Hy-MT3-30B-A3B-GGUF/Hy-MT3-30B-A3B-Q8_0.gguf \
  --prompt "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：

机器学习是人工智能的一个分支。" \
  --max-tokens 128 --threads 32 --temp 0
```

**1.8B / 7B（`hunyuan-dense`）：**

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/Hy-MT3-1.8B-GGUF/Hy-MT3-1.8B-Q8_0.gguf \
  --prompt "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：

机器学习是人工智能的一个分支。" \
  --max-tokens 128 --threads 8 --temp 0
```

**HTTP：**

```bash
cargo run --release --bin rust-model-server -- \
  --model models/Hy-MT3-30B-A3B-GGUF/Hy-MT3-30B-A3B-Q8_0.gguf --port 8080

curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role":"user","content":"将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n机器学习是人工智能的一个分支。"}],
  "max_tokens": 128, "temperature": 0
}'
```

`/v1/chat/completions`、`/v1/responses`、`/v1/messages` 与 SSE 流式都走同一个
`TextRuntime`，CLI 与 HTTP 共用 `HyV3TextRuntime`，不会各写一套解码循环。

## 3. Prompt 与 Tokenizer 契约

三个文件的 tokenizer 并**不相同**，且决定 prompt 形态的不是架构，而是词表里有没有
`<｜hy_User｜>` 这个 control token：

| 规模 | `tokenizer.ggml.pre` | 词表 | `<｜hy_User｜>` | prompt 形态 |
| --- | --- | --- | --- | --- |
| 1.8B | `hunyuan-dense` | 120818 | 有（120006） | `<｜hy_User｜>{prompt}<｜hy_Assistant｜>` |
| 7B | `hunyuan` | 128167 | **无** | 原样透传（可选 BOS `<｜hy_begin▁of▁sentence｜>`） |
| 30B-A3B | `hunyuan-dense` | 120832 | 有（120006） | `<｜hy_User｜>{prompt}<｜hy_Assistant｜>` |

`build_hunyuan_chat_prompt` 按 `special_token_id("hy_user")` 是否存在的自动选择就是上表；
`--jinja` 则直接渲染 GGUF 自带的 `tokenizer.chat_template`（30B 的模板 11 KB，会带
`<｜hy_begin▁of▁sentence｜>` 与空 system 轮）。

两处容易踩的细节：

- **`clean_spaces` 只影响 detokenize**。1.8B/7B 为 `true`、30B 为 `false`；llama.cpp 只在
  解码阶段用它去掉 `? ! . ,` 前的空格（`llama-vocab.cpp` 的 `clean_up_tokenization_spaces`），
  不参与分词。所以**token ID 对齐不受影响**，但两边解出来的文本可能差一个空格；
  比对文本时要容忍这一点，比对 token ID 时不能。
- **`pre = "hunyuan"` 的数字规则是 `\p{N}`**：每个数字各自成一个 piece。1.8B/30B 的
  `hunyuan-dense` 反而是 `\p{N}{1,3}` 三段一组。两者不能共用一条分支，否则
  `12345` 会从 5 个 piece 变成一个，BPE 之后再也拼不回来。本仓库此前把两者混用，
  已修正（`src/core/tokenizer/mod.rs`，回归测试
  `hy_scanner_isolates_each_digit_but_hy_dense_chunks_them`）。

## 4. 采样参数

模型卡建议（非 thinking 模式，三个模型都没有默认 system prompt）：

| 规模 | temperature | top_p | top_k | repetition_penalty | max_tokens |
| --- | --- | --- | --- | --- | --- |
| 1.8B / 7B | 0.7 | 0.6 | 20 | 1.05 | 4096 |
| 30B-A3B | 0.7 | 1.0 | -1 | 1.0 | 4096 |

本仓库 CLI 默认 `--temp 0`（greedy），这也是**唯一能对齐 Oracle 的模式**：`hy_v3`
文本运行时用 `LlamaSampler`，`temp > 0` 时会叠上 `general.sampling.*` 的
`top_k=40 / top_p=0.95`，而 oracle harness 只做 argmax。要做数值对齐就传 `--temp 0`。

## 5. 上下文

- GGUF `context_length = 262144`，但 KV 实际容量取 `min(n_ctx, --max-context)`，
  默认 `--max-context 8192`。30B-A3B 是 48 层 × 4 KV head × 128 维，
  8192 容量下 F32 KV 约 **1.6 GB**，F16 约 0.8 GB
  （`48 × 8192 × 4 × 128 × 2(K,V) × 4 B`）；开到 262144 会直接吃掉 50 GB 量级。
- `--kv-cache` 默认 F32。和 llama.cpp 对拍时显式传 `--kv-cache f32`（oracle 侧
  `type_k=type_v=GGML_TYPE_F32`），否则 attention 的舍入差会被逐层放大。

## 6. 限制

- **只有 CPU 路径**。`hy_v3` 没有 Vulkan 算子，`--compute vulkan` 明确报错；
  `auto` 在无设备机器上就是 CPU。性能与数值证据见
  [`tools/oracle/hy_mt3/`](../../tools/oracle/hy_mt3/README.md)。
- **JEV 不支持**。`hy_v3` 文本运行时只暴露 `generate()`，没有 logits 入口，
  `--jev` 会明确拒绝而不是给出无意义的分数。
- 30B-A3B 的 MTP/NextN 头**不在**这几个 GGUF 里（766 个张量中没有 `nextn.*`），
  投机解码不适用。
- 完整 262144 上下文、其他量化档（IQ2）与 30B 的 GPU 路径未验证。
