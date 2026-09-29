# CLM-v0.1-8B 用法

CLM（Contrastive Language Model）不是生成模型。它在冻结的 Qwen3-8B encoder 上
加两个 MLP 头（state head / action head），用双向 InfoNCE 训练，只做**给候选
打分**，不生成文本。

## 0. 它是什么

```
score(state, candidate) = logit_scale * cos(state_head(e_s), action_head(e_c))
```

其中 `e_s` / `e_c` 是 Qwen3-8B 的 **last-token pooling** 隐状态，两个头各自
L2-normalize 之后点乘。`logit_scale = min(exp(4.6132), 100) = 100`。

ModelScope 上的 `CLM_v0.1-8B.pt` 只有 **75 MB** —— 只含两个头，不含 encoder。
跑起来需要额外一个 Qwen3-8B 的 GGUF。

## 1. 准备

```bash
# 1) 头（75 MB）
modelscope download --model Contrastive-LM/CLM-v0.1-8B \
    CLM_v0.1-8B.pt --local-dir ./CLM-v0.1-8B

# 2) 转成 GGUF
PYTHONPATH=. models/.venv/bin/python tools/converter/clm/convert_clm.py \
    models/CLM-v0.1-8B/CLM_v0.1-8B.pt

# 3) encoder。CLM 是 encoder-locked（README 自己说的），
#    打分要尽量保真就用 BF16；要省内存 Q8_0 也能跑，分数会偏。
modelscope download --model ggml-org/Qwen3-8B-GGUF \
    Qwen3-8B-BF16.gguf --local-dir ./Qwen3-8B-GGUF
```

转换产物：`models/CLM-v0.1-8B/clm-v0.1-8B-heads-f32.gguf`
（17 个 tensor，18.9M 参数，F32）。

## 2. 打分

CLM 没有自己的 binary，走既有的 `rust-model-inference`，挂在 JEV flag 家族下：

```bash
./target/release/rust-model-inference \
  --model models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf \
  --jev --clm-head models/CLM-v0.1-8B/clm-v0.1-8B-heads-f32.gguf \
  --jev-context "Customer: my invoice was charged twice and nobody answers the phone!" \
  --jev-question "Which team should handle this?" \
  --jev-option "Charges, invoices, refunds" \
  --jev-option "Bugs and outages" \
  --threads 8
```

输出（README 那个例子的实测结果，排序与参考一致）：

```
1. 30.7022  Charges, invoices, refunds
2. 27.1489  Bugs and outages
```

`--jev-option` 可重复。QEV 的 26 个候选上限（A..Z）同样适用于这里。

## 3. prompt 布局（不看这个会排错序）

这是最容易踩的坑，`src/clm/schema.py` 里的 `state_text()`：

- **state 侧 = context + `\n\n` + question**（"上下文空行问题"）
- **candidate 侧 = 原文，什么都不加**

只喂 context 不喂 question，state head 的信息不足，排序会退化。实测过：

```
# 只喂 context
1. 24.8470  Restart the router      <-- 错
2. 23.4004  Escalate to billing
3. 21.8695  Offer a refund

# 补上 "Which team should handle this?"
1. 30.7022  Charges, invoices, refunds   <-- 对
2. 27.1489  Bugs and outages
```

## 4. 为什么没有 HTTP 端点

有意不做 `/v1/rank`。仓库已有的打分暴露策略是 JEV 一族
（`/v1/jev/score`、`/v1/jev/grouped`、`/v1/jev/image`、
`/v1/jev/image_grouped`），CLM 要接也应该接进这套，而不是另起一个
OpenAI 风格的 `/v1/rank`。

CLM 目前只有 CLI，因为接进 `/v1/jev/*` 不是改个路由名的事：

- JEV 的打分是**跑生成模型看 label logit**，CLM 是**embedding 余弦**，两套数；
- CLM 需要**额外的头文件**和 encoder 配对加载，JEV 的 scorer 接口
  （`build_prompt` + `forward_logits`）没有"第二个权重文件"这个维度；
- 按 `schema.py`，`action` 侧可按 candidate 缓存复用（~1k 候选时快 13x），
  这也需要在 backend 层做缓存，和 `Backend::Rerank` 那种一次性打分不同。

要做的话参照 `Backend::Rerank` 加一个 `Backend::Clm`，路由挂 `/v1/jev/...`
家族。

## 5. 精度

- **head 本身**逐位对过参考 torch forward（`models/CLM-v0.1-8B/golden.json`，
  `cargo test --lib models::clm`）。
- **端到端**排序与参考客户端一致，但绝对分数不完全对齐
  （README 0.94/0.06，本地同两个分数算出来 ~0.97/0.03）。参考客户端经 vLLM
  做 embedding，prompt 处理与 tokenizer 细节不同。CLM 是 encoder-locked 的，
  这部分差异要消掉得把 encoder 侧也逐位对齐，目前没做。

## 6. 源码索引

| 内容 | 位置 |
|---|---|
| 头加载 / forward / 打分 | `src/models/clm/mod.rs` |
| .pt -> GGUF 转换器 | `tools/converter/clm/convert_clm.py` |
| 转换器 round-trip 测试 | `tools/converter/clm/test_convert_clm.py` |
| CLI 分发（JEV 族肢） | `src/app/jev/clm.rs`（`run_clm_decision` / `run_clm_decision_data`） |
| encoder last-token 隐状态 | `Qwen3Session::forward_last_hidden`（`src/models/qwen3/trunk/session.rs`） |
