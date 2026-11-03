# Mellum 2.1 用法

Mellum 2.1 是 JetBrains 的 MoE 思考型文本模型：12B 总参数、2.5B 激活
（64 专家 top-8）。GGUF `general.architecture = mellum`，实现位于
`src/models/mellum/trunk/`。

> 前置：`cargo build --release --bin rust-model-inference`。
> 当前仅 CPU；仅 **Q8_0** 已验证（见下方「已验证范围」）。

## 1. 权重

```bash
mkdir -p models && cd models
[ -x .venv/bin/modelscope ] || (python3 -m venv .venv && .venv/bin/pip install modelscope)

.venv/bin/modelscope download \
  --model JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF \
  Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf README.md \
  --local_dir ./Mellum2.1-12B-A2.5B-Thinking-GGUF
```

同 repo 里的 `MXFP4_MOE` / `Q4_K_M` / `Q6_K` **未接入**：`MXFP4_MOE`
需要仓库没有的反量化路径，k-quant 另外还要复现 ggml 的 f16 累加点积。

已验证文件：

| | |
| --- | --- |
| 文件 | `Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf` |
| 大小 | 12 925 417 760 字节 |
| SHA-256 | `69e0ac02a16586c2c9e991f1e3b8a1eb403882e7ab9e69ea01eb23dbbbc6841f` |

## 2. 推理

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/Mellum2.1-12B-A2.5B-Thinking-GGUF/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf \
  --prompt "What is 2+2?" \
  --max-tokens 256 \
  --threads 8 \
  --kv-cache f32 \
  --temp 0
```

`--threads` 走仓库内置 ComputePool 并行 matmul。KV cache 建议 `--kv-cache f32`；
`--kv-cache f16` 可用但未做逐位对齐。

## 3. thinking 模式

GGUF 自带的 `tokenizer.chat_template` 是 ChatML，生成前缀为
`<|im_start|>assistant\n`；`--thinking` 决定是否由模型自行展开 `<think>`，
`--no-thinking` 则在 prompt 里写入一个空的 `<think>\n\n</think>\n\n`。

```bash
# 思考后再回答
... --prompt "What is 2+2?" --thinking

# 直接回答
... --prompt "What is 2+2?" --no-thinking
```

## 4. 模型结构要点

这些是从 GGUF metadata 与 pinned llama.cpp `src/models/mellum.cpp` 读出来的
契约，改动前请先对照：

| 项 | 值 |
| --- | --- |
| 层数 / `n_embd` | 28 / 2304 |
| 注意力 | 32 query head + 4 kv head，head_dim 128（取自 `attention.key_length`，**不是** `n_embd / n_head` = 72） |
| MoE | 64 专家 top-8，`expert_feed_forward_length = 896`，softmax 门控 + 权重归一 |
| 残差 | **并行**：`ffn_inp = attn_out + hidden`，`hidden = moe(ffn_inp) + ffn_inp` |
| 滑动窗口 | `sliding_window = 1024`，`sliding_window_pattern = [T,T,T,F] × 7`（21 个 SWA 层） |
| RoPE | NEOX；SWA 层用无缩放表，另外 7 个非 SWA 层用 YaRN（factor 16 / orig ctx 8192 / base 5e5） |
| Tokenizer | `tokenizer.ggml.model = gpt2`，`tokenizer.ggml.pre = mellum2` |
| 特殊 token | BOS `<\|endoftext\|>`=0，EOS `<\|im_end\|>`=28；`add_bos_token = false` |

`mellum` 之所以不复用 qwen3 trunk：它在 tokenizer、位置编码、Attention/KV
三处都与 qwen3moe 不一致，按「相近架构」路由会静默算错。

`mellum2` 预分词与仓库里其它 BPE dialect 的两点差别：

* 先按 `\p{N}` 拆一遍，所以 `12345` 是 5 个单数字 pre-token（`LlamaBpe` 会把
  `12345` 保持成一个）；
* 词正则里收缩词只写小写，所以 `'S` 拆成 `'` + `S`（Qwen 系会保持 `'S`）。

## 5. 已验证范围

对齐 pinned llama.cpp `8e2d31e0eb658d0382d3106a61369dff81f0c080` 的标量
oracle（`RMI_SCALAR=1`，`--kv-cache f32`）：

* `"What is 2+2?"` + 4 步 greedy：prompt/生成 token IDs 与 oracle 完全一致，
  **7425 个检查点 F32 逐位一致**；
* 1290 token prompt（跨 1024 滑动窗口边界）+ 2 步 greedy：生成 token 一致，
  边界前后（step 1022–1026）SWA 层与非 SWA 层检查点逐位一致。

复现命令、oracle 固定版本与踩过的坑见
[`tools/oracle/mellum/README.md`](../../tools/oracle/mellum/README.md)。

**未覆盖**：MXFP4_MOE / Q4_K_M / Q6_K；SIMD（非标量）路径；embedding、rerank 入口；
batched prefill（trunk 按单 token 推进）。

## 6. JEV 决策打分

单次前向打分候选标签，已接入单模式与 grouped 两种模式：

```bash
# 单模式
./target/release-fast/rust-model-inference --model <gguf> --jev \
  --jev-context "Extraction pipeline status: green" \
  --jev-question "What is the pipeline status?" \
  --jev-option "red" --jev-option "green"
# => choice: B (green), 0.9195

# grouped（multi_select）
./target/release-fast/rust-model-inference --model <gguf> --jev --jev-multi \
  --jev-context "Ticket 41 assigned to the payments team." \
  --jev-question "Is the ticket assigned?" \
  --jev-option "assigned to payments" --jev-option "unassigned"
```

HTTP 端点 `/v1/jev/score` 与 `/v1/jev/grouped` 同样可用，实测概率与 CLI 逐位一致
（`B: 0.9194594025611877`）。

prompt 用 checkpoint 自己的 ChatML（`role\n content\n` + `assistant\n`，无
`<|im_start|>` 标记，`add_bos_token = false`），不注入 `<think>` 块，保证被打分的位置不动。

## 7. 不支持的入口

* `--embedding` / `--rerank`：未接入。
* 图像 / 音频输入：`mellum` GGUF 无多模态组件，不支持（`/v1/jev/image*` 返回 400）。
