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

## 4. 服务端模式

CLM 没有自己的 binary，也不引入 OpenAI 风格的 `/v1/rank`。它复用仓库既有的
JEV flag 家族——打分这件事在仓库里已经有一套约定，CLM 只是它的另一种打分实现。

**启动**（`--clm-head` 是启动参数，和 `--mmproj` 同级）：

```bash
./target/release/rust-model-server \
  --model models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf \
  --clm-head models/CLM-v0.1-8B/clm-v0.1-8B-heads-f32.gguf \
  --host 0.0.0.0 --port 8080 --threads 8
```

启动日志会打 `mode=clm`。

**请求**（`/v1/jev/score`，请求体和 logit 打分的 JEV 完全一致，调用方
感知不到内部是 cosine 还是 logit）：

```bash
curl http://127.0.0.1:8080/v1/jev/score -H 'Content-Type: application/json' -d '{
  "context": "Customer: my invoice was charged twice and nobody answers the phone!\n\nWhich team should handle this?",
  "questions": [{"text": "", "options": ["Charges, invoices, refunds", "Bugs and outages"]}]
}'
```

```json
{
  "mode": "single",
  "results": [{
    "mode": "choice", "labels": ["A","B"],
    "descriptions": ["Charges, invoices, refunds","Bugs and outages"],
    "probabilities": {"A": 0.9721649289131165, "B": 0.027835026383399963},
    "choice": "A", "confidence": 0.47216495871543884,
    "entropy": 0.1271340698003769, "margin": 0.9443299174308777,
    "prefill_ms": 29213
  }]
}
```

`values` 字段（原始 score）在 JSON 里也会带上。

**只注册 `/v1/jev/score` 一个路由**，其余一律 404——这是模型能力边界，不是
接口缺功能：

| 路由 | CLM 后端 | 原因 |
|---|---|---|
| `/v1/jev/score` | yes | CLM 就是干这个的 |
| `/v1/jev/grouped` | 404 | grouped 做 per-group softmax，cosine 打分没有对应物 |
| `/v1/jev/image`、`/v1/jev/image_grouped` | 404 | 头是在文本 embedding 上训的，接不了图像 |
| `/v1/chat/completions` 等 | 404 | CLM 不生成文本 |

同一个 server 进程不会同时提供 CLM 打分和 JEV logit 打分——由启动时有没有
`--clm-head` 决定。这样调用方不需要协商模式，运维也只用看一个 flag。

### 性能：未做 candidate 缓存

参考实现把 action 侧的 embedding 按 candidate 缓存复用（~1k 候选时声称快
13x），这边没做，每次请求都重新 embed 所有 candidate。正确性优先，等真有
吞吐需求再加——加了之后要注意缓存 key 必须含 encoder + heads 的身份，
否则换模型后会读到别的头的投影。

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
| 分发（CLI 与 HTTP 共用） | `src/app/jev/clm.rs`（`run_clm_decision` / `run_clm_decision_data` / `run_clm_scoring`） |
| HTTP 后端 | `Backend::Clm` / `build_clm`（`src/app/server/mod.rs`）、`jev_score` 的 CLM 分支（`src/app/server/api.rs`） |
| encoder last-token 隐状态 | `Qwen3Session::forward_last_hidden`（`src/models/qwen3/trunk/session.rs`） |
