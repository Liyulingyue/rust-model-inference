# Spark-X2.5 用法

Spark 2.5 是 BF16 权重的国产文本模型，1.7B 与 4B 两个尺寸均已接入。
GGUF `general.architecture = spark2_5`，对应 `src/models/spark/trunk/`。
CLI 入口 `src/models/spark/trunk/forward.rs::run_inference`。

> 共用前置：构建 `cargo build --release --bin rust-model-inference`。
> 当前仅 CPU；BF16 路径通过仓库内置 AVX2 kernel 加速，已在 issue 9 修复后
> 接通 ComputePool 并行 matmul（4 线程相对单线程约 2.4× 加速）。

## 1. 推理

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/Spark-X2.5-1.7B.gguf \
  --prompt "法国的首都是" --max-tokens 64
```

加 `--threads 4` 启用并行 matmul；KV cache 默认 F16。

## 2. thinking 模式

Spark 2.5 chat template 在生成 prompt 末尾区分 `<|Bot|><think>` 与
`<|Bot|></think>`，分别对应「先推理再回答」与「直接回答」：

```bash
# reasoning + answer：显式传 --thinking
cargo run --release --bin rust-model-inference -- \
  --model models/Spark-X2.5-1.7B.gguf \
  --prompt "法国的首都是" --thinking

# 直接回答：默认行为（不传 --thinking）
cargo run --release --bin rust-model-inference -- \
  --model models/Spark-X2.5-1.7B.gguf \
  --prompt "法国的首都是"
```

CLI 当前**只暴露 `--thinking`**（默认 false）；没有 `--no-thinking` 标志。
要切换到直接回答，省略 `--thinking` 即可。

chat template 来源（`src/models/spark/trunk/forward.rs:423-428`）：

```
<sos><|System|>\nyou are a helpful assistant.<eos>
<sos><|User|>{prompt}<eos>
<sos><|Bot|><think>          # thinking=true
<sos><|Bot|></think>          # thinking=false（默认）
```

## 3. 4B 版本

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/Spark-X2.5-4B.gguf \
  --prompt "法国的首都是" --max-tokens 64 --threads 4
```

> 当前 CPU 路径对 4B 较慢；建议保留 `--threads` 显式给出。

## 4. CLI 路由速查

| GGUF `general.architecture` | 进入 trunk | Modes |
|---|---|---|
| `spark2_5` | `src/models/spark/trunk/forward.rs` | 文本（仅 `--thinking`） |

`src/app/text.rs:110-121` 把 `arch == "spark2_5"` 路由到 spark run_inference，
其余 arch 走默认 qwen3 fallback。

## 5. 与 llama.cpp 的对齐

参考实现：[XHToken/llama.cpp](https://github.com/XHToken/llama.cpp)（仓库内
旧称 `XFllama.cpp`），对应 `Spark2.5` 模型代码。

当前**未固定 commit**，状态 `Pending pin`（`docs/REFERENCE_IMPLEMENTATIONS.md`）。
`docs/SUPPORTED_MODELS.md` 明确：

- 1.7B：BF16 真实 GGUF 中英文和算术冒烟通过；**尚未完成 XFllama.cpp token 级 Oracle 对齐**
- 4B：BF16 真实 GGUF 冒烟通过；CPU 路径较慢，**尚未完成严格 Oracle 对齐**

要把这两行推进到 `Verified`，需要：

1. 选一个含完整 Spark2.5 实现的 llama.cpp commit 并 pin
2. 写 `tools/spark/build_oracle.sh`
3. 加 `tests/spark_reference.rs` 做 token 级对照

## 6. 已确认的限制 / 边界

| 范围 | 行为 |
|---|---|
| 量化 | 当前仅 BF16；其他量化格式未验证 |
| Oracle pin | `Pending pin`（XHToken/llama.cpp 未固定 commit） |
| thinking 切换 | 仅 `--thinking` 显式控制；默认 false（直接回答）。没有 `--no-thinking` 标志 |
| 计算性能 | 4B CPU 较慢；通过 ComputePool + BF16 AVX2 kernel 缓解但仍未与兄弟模型持平 |

## 7. 已知 bug 与修复历史

| Issue | 描述 | 修复 commit |
|---|---|---|
| Issue 7 | SWA mask 应在 softmax 之前置 `-inf`，而非 softmax 后清零 | `717fc42` |
| Issue 8 | prefill 末尾第一个生成 token 被跳过（生成循环固定再跑一次 decode_step） | `6d24aeb 修正早退问题` |
| Issue 9 | `--threads` 未真正生效——所有 matmul 都跑单线程 | `2a78e56` |

参考 [docs/PARALLEL_MATMUL_SAFETY.md](../../PARALLEL_MATMUL_SAFETY.md) 关于
并行 matmul `&mut` 别名问题。

## 8. 相关源码索引

- `src/models/spark/trunk/config.rs` — 配置加载（专用 config loader）
- `src/models/spark/trunk/forward.rs` — SparkSession + run_inference
- `src/models/spark/trunk/weights.rs` — 权重加载
- `src/app/text.rs:110-121` — CLI 路由
- `docs/REFERENCE_IMPLEMENTATIONS.md` — XHToken/llama.cpp 参考条目

## 9. JEV 决策评分

`--jev` 是 OpenJEV 风格的 single-forward-pass 决策评分模式。通用协议、
3 种 mode、JSON 输出、已知限制见 [`docs/develop/jev.md`](../develop/jev.md)
和 [`docs/usage/qwen3.md` §10](qwen3.md)。

### 9.1 Arch 路由

| Arch | JEV 路径 |
|---|---|
| `spark2_5` | `app/jev/single/spark.rs::run_jev_decision_spark` → `spark::SparkSession::forward_logits` |

Spark 2.5 的 chat template 用 `<｜start▁of▁sentence｜>` / `<｜end▁of▁sentence｜>`
包裹每个 role block。JEV 在 `run_jev_decision_spark` 中用对应格式拼
prompt（始终以 `</think>` 收尾，跳过 thinking block）。

### 9.2 示例

```bash
# Spark-X2.5-1.7B — Choice mode
rust-model-inference --model models/Spark-X2.5-1.7B.gguf \
  --jev --jev-context "用户咨询产品保修" \
  --jev-question "应该归到哪个类别？" \
  --jev-option "保修期内" --jev-option "保修期外" --jev-option "其他" \
  --threads 4

# Spark-X2.5-4B — Binary mode（首选 instruct-tuned 大模型，准确率更高）
rust-model-inference --model models/Spark-X2.5-4B.gguf \
  --jev --jev-context "用户已经提供了订单号" \
  --jev-question "客服可以立即处理吗？" \
  --jev-option "可以" --jev-option "不可以" --jev-positive A \
  --threads 4

# Score mode（带 value 评分）
rust-model-inference --model models/Spark-X2.5-1.7B.gguf \
  --jev --jev-context "投诉工单内容" \
  --jev-question "用户情绪强度？" \
  --jev-option "极低:1" --jev-option "低:2" --jev-option "中:3" \
  --jev-option "高:4" --jev-option "极高:5" \
  --threads 4
```

> Spark 2.5 是 hybrid（fused QKV + SWA + GeGLU + per-head gating），JEV
> prefill 复用 `SparkSession::forward_logits`（thin wrapper over
> `forward_step_logits`），与 `decode_step` 共享同一份 forward 路径，
> 但只跑 prefill 不做 sampling。