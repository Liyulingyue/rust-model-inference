# LFM 家族用法

本仓库对 LFM2 / LFM2.5 / LFM2-MoE / LFM2.5-VL / LFM2.5-Thinking 的端到端命令行示例。

> 通用前置：构建 `cargo build --release --bin rust-model-inference`。
> 所有 LFM 文本 / 视觉模型在 GGUF 里的 `general.architecture` 都是字符串 `"lfm2"`。
> LFM2 与 LFM2.5 文本的变体由 CLI 路由阶段通过 `general.basename` 含 `"2.5"` 区分
> （`src/app/text.rs:62-66`），分别进入 `src/models/lfm2/` 与 `src/models/lfm25/`。
> 详见 `docs/ISSUE.md` 的 LFM2 / LFM2.5 命名不一致条目。
> 
> 常用生成参数（适用于所有 LFM 路径）：
> 
> - `--max-context N`：KV cache 容量上限，默认 8192。LFM2.5 文本模型
>   `context_length=128000`，过大的 `--max-context` 会一次性占用 GB 级 KV 内存。
> - `--repetition-penalty α`：logit 级重复抑制，默认 1.0（禁用）；α > 1
>   抑制重复（与 llama.cpp / Hugging Face `repetition_penalty` 等价）。
>   对低质量量化（如 Q4_K_M）下陷入复读循环的模型尤其有用。

## 1. LFM2 文本（`src/models/lfm2/`）

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/lfm2/LFM2-XXXX-Q8_0.gguf \
  --prompt "法国的首都是" --max-tokens 30
```

## 2. LFM2.5 文本（`src/models/lfm25/`）

GGUF `general.basename` 含 `"2.5"` 时自动路由：

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/lfm2.5/LFM2.5-1.2B-Instruct-Q8_0.gguf \
  --prompt "法国的首都是" --max-tokens 30
```

如果 `general.basename` 不含 `"2.5"`，同样的 GGUF 仍能加载但会走 LFM2 trunk，
可能与 LFM2.5 实际架构不一致。建议显式选 basename 正确的 GGUF。

### 2.1 LFM2.5-Thinking（reasoning 模型）

`LFM2.5-1.2B-Thinking`（arch = `lfm2`,basename 含 `"2.5"`）走 `src/models/lfm25/`
trunk。模型默认输出会包含 ``…`` reasoning 段 + 答案文字。仓库代码
会在构造 chat prompt 时自动注入 LFM2.5 官方 system prompt：

> "You are a helpful assistant trained by Liquid AI. Your goal is to be helpful, accurate, and concise."

不带 system turn 也能加载但可能输出退化。建议显式走默认 chat 路径。

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q8_0.gguf \
  --prompt "What is 2 + 3?" --max-tokens 200 \
  --max-context 8192 --repetition-penalty 1.05
```

实测 `LFM2.5-1.2B-Thinking-Q8_0` 在 8 线程 Q8_0 下：

- arch 路由：`lfm2.5`（`src/models/lfm25/trunk/`）
- 文本生成：~28 t/s（短 prompt）
- 完整 reasoning + 答案：约 5–10 秒（typical reasoning 长度 50–150 tokens）

CLI 当前**未实现** `--thinking` / `--no-thinking` 切换标志。HTTP 端通过
`enable_thinking` 字段控制（详见 [§7.1](#71-enable_thinking-服务端开关)）：
- `enable_thinking: true`（默认）— 与训练时一致，模型输出 ``…`` reasoning 段
- `enable_thinking: false` — prompt 尾部追加 `\n\n`，让模型跳过 thinking 直接答
- 模型是 thinking-trained 时，`enable_thinking: false` 只是 hint（prompt 层的
  提示），不一定能完全阻止模型 emit think block；当前实现仍会发出一个短的
  ``…`` 段。ThinkFilter (`src/app/server/api/think.rs`) 会自动
  剥掉 leading `` 块再透传真正的答案。

## 3. LFM2-MoE

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/LFM2-8B-A1B-Q8_0.gguf \
  --prompt "2 + 3 =" --max-tokens 4 --temp 0
```

注意：

- 这是 **MoE** 路由，进入 `src/models/lfm2moe/`，CLI 路由条件为
  `arch == "lfm2moe"`（来自 GGUF metadata，非通用 `lfm2`）。
- 与 llama.cpp 前 6 个生成 token 一致；在 MoE 近平局处可能分叉。
- shared-expert 张量不被支持（会被 loader 显式拒绝）。

## 4. LFM2.5-VL（`src/models/lfm2/vision.rs`）

Vision 路径与文本路径**共用** `arch == "lfm2"` 分派，不走 basename。
代码里直接落到 `src/models/lfm2/vision.rs::run_multimodal`：

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/lfm2.5-vl/LFM2.5-VL-1.6B-Q8_0.gguf \
  --mmproj models/lfm2.5-vl/mmproj-F16.gguf \
  --image path/to/image.jpg \
  --prompt "Describe this image."
```

约束：

- 必须同时传入 `--mmproj` 与 `--image`，缺一即报错：
  - `LFM2.5-VL requires --mmproj`
  - `LFM2.5-VL requires --image`
- 仅支持**视觉**模态；传入 `--audio` 会被拒绝：
  `Only gemma4 architecture is supported for multimodal audio, got: lfm2`
  （见 `src/app/text.rs:803-806`）
- KV cache 固定 `F16`（`KvFormat::F16` 直接传给 `run_multimodal`）。
- mmproj 期望的 projector 元数据键：`clip.vision.{image_size, patch_size,
  projector.scale_factor, embedding_length, attention.head_count, block_count,
  feed_forward_length, attention.layer_norm_epsilon, projection_dim, image_mean,
  image_std}`，见 `src/models/lfm2/vision.rs:73-83`。

### 4.0 Prompt 模板（重要）

代码内部把 user / assistant turn 包成 ChatML：

```
<|startoftext|><|im_start|>user\n
<|image_start|><|img_row_*_col_*|>...<|img_thumbnail|>...<|image_end|>
{prompt}<|im_end|>
<|im_start|>assistant\n
```

CLI 不需要手动拼 `<|im_start|>` / `<|im_end|>`，但用户传入的 `--prompt`
应避免重复这些标记。**裸的 `user\n...assistant\n` 模板会让模型输出退化**
（实测：Q8_0 1.6B + 768×768 图 + 裸 `user/assistant` 模板 → 输出
"A / Answer: / A" 循环）。ChatML 模板 + 实际物体图片可正确识别
（实测：apple.png → "fresh-looking apple with a glossy red and yellow skin,
green leaf attached to its stem, plain white background"）。

### 4.1 图片大小与 vision token 数（CPU 实测，2026-09）

Vision encoder 把图切成 512×512 tile + 1 张 overview。tile 数和总 vision tokens
直接决定 prefill 时长：

| 原图分辨率 | Tile grid | Vision tokens | 备注 |
|---|---|---|---|
| ≤ 512（任一维） | 0×0 | ~64–128 | 单 overview，prefill ~1s |
| ~768×768 | 2×2 (4 tiles) + overview | ~1100+ | prefill 约 30s（8 线程 Q8_0）；后续生成 ~24 t/s；首 token 出得很慢但非卡死 |
| 401×287（实测 `references/apple.png`） | 0×0 | 117 | 端到端 8.9 tok/s @ 8 thread (LFM2.5-VL-1.6B Q8_0) |

**建议**：CPU 路径下使用 ≤ 512×512 的输入图。`references/apple.png`（401×287）
是个不错的示例尺寸。`models/test768.png`（768×768）CPU 上 prefill ~30s，
之后每 token 约 24 t/s，输出受 prompt template 影响大（见 §4.0）。

1.6B VL 模型 + 1024 vision tokens 的 CPU prefill 主要成本是 16 层 × 1024
tokens 的 matmul，不是 SIMD gap。如果要测大图，建议加 `--gpu`（Vulkan
未对 `lfm2` arch 完整覆盖，仅在分片 matmul 上生效——见 §6）。

## 5. CLI 路由规则速查

| 输入 GGUF `general.architecture` | `general.basename` | 进入 trunk | Modes |
|---|---|---|---|
| `lfm2` | 不含 `2.5` | `src/models/lfm2/`（文本） | 文本 |
| `lfm2` | 含 `2.5`（含 Instruct / Thinking） | `src/models/lfm25/`（文本） | 文本 |
| `lfm2` | 任意 | `src/models/lfm2/vision.rs`（VL，需 `--mmproj --image`） | 多模态 |
| `lfm2moe` | — | `src/models/lfm2moe/` | 文本（MoE） |
| `nanbeige` | — | `src/models/llama/` | 文本（Experimental） |

`LFM2.5-Thinking` 区别于 `LFM2.5-Instruct`：Thinking 模型默认在 `<think>...</think>`
内输出 reasoning；本仓库代码会用同一个 `lfm25` trunk，但**思考文本会直接流
到 stdout / 客户端**。HTTP 端可以通过 `enable_thinking: false` 字段抑制
（详见 §7.1），CLI 端当前**未实现** `--no-thinking` 标志 — 想关掉 CLI
上的 thinking，请走 HTTP `enable_thinking` 路径。

## 6. 与 llama.cpp 的对齐

`docs/REFERENCE_IMPLEMENTATIONS.md` 中**没有**任何 LFM 家族的 Pinned Oracle：

- 没有固定的 llama.cpp commit
- 没有 `tools/lfm2/...` 构建脚本- 没有 `tests/lfm*_reference.rs`

当前只能靠运行时 smoke test。建议在引入 LFM2.5-VL 的端到端用例之前先固定一个
llama.cpp commit + build 脚本。

## 7. 服务端模式

```bash
cargo run --release --bin server -- \
  --model models/lfm2.5/LFM2.5-1.2B-Instruct-Q8_0.gguf \
  --host 0.0.0.0 --port 8080 --threads 4
```

服务端对 LFM2 / LFM2.5 / LFM2-MoE 等纯文本架构按 CLI 选项暴露，无图像/音频模态。

### 7.1 `enable_thinking` 服务端开关

`/v1/chat/completions`（以及 Anthropic / Responses 协议）接受顶层
`enable_thinking: bool` 字段（同时也接受 jinja 风格的嵌套
`chat_template_kwargs: {"enable_thinking": ...}`，顶层优先）。`null` /
缺失 = 用模型训练时的默认（thinking-tuned 变体如 LFM2.5-Thinking 默认开）。

| arch | `None` 默认 | `enable_thinking: false` 的行为 |
|---|---|---|
| `lfm2moe`（LFM2.5-8B-A1B 等） | `true`（训练时是 thinking 模型） | prompt 尾部追加 `\n\n`，让模型尝试跳过 thinking |
| `lfm2`（LFM2 / LFM2.5 文本） | `true` | 同上 |
| `llama` / `nanbeige` / `granite` / `glm4` / `phi3` | `false`（HTTP 路径原本就走 `thinking=false`） | k2-horizon / MiniCPM5 才会显式生效；其它 arch 的 prompt 模板忽略此字段 |
| `qwen3` / `qwen35` / `qwen3vl` | `false` | 通过 `append_qwen_assistant_prefix(false)` 触发 non-thinking tail |

LFM2 / LFM2.5 实测（`LFM2.5-8B-A1B-Q8_0`，`max_tokens=150`，问 "What is
the capital of France?"）：

| 字段 | prompt tok | completion tok | 客户端收到 |
|---|---|---|---|
| `enable_thinking: false` | 33 | 104 | `'\nParis'`（think 段被 ThinkFilter 自动剥掉） |
| `enable_thinking: true` | 32 | 77 | `'\nParis'` |
| 缺失 | 32 | 77 | `'\nParis'` |

**注意**：LFM2.5-Thinking 是 thinking-trained 模型，即使
`enable_thinking: false`，模型仍可能 emit 一个短的 ``…`` reasoning
段（仅是 prompt 层的 hint，不是 jinja 层的开关）。`ThinkFilter`
（`src/app/server/api/think.rs`）自动剥掉 leading think 块，客户端看到
的内容仍是纯答案。如果模型 emit 中段的 `` 块（`...answer thinking ...answer`
 格式），则不会被剥掉 — 这是已知限制（详见 TODO）。

类型错误（`enable_thinking: "yes"`）返回 400：

```json
{"error":{"message":"enable_thinking must be a boolean", ...}}
```

CLI 上对应的 `--thinking` / `--no-thinking` 标志**尚未实现**；想关闭
thinking 请走 HTTP。

## 8. 已确认的限制 / 边界

| 范围 | 行为 |
|------|------|
| LFM2.5-VL + `--audio` | 拒绝（只支持视觉） |
| 缺 `--mmproj` / `--image` | 配置阶段报错 |
| LFM2.5-VL GGUF 的 `general.architecture` 是 `lfm2` | 与文本 LFM2 共用 arch，靠 mmproj + image 区分 |
| Dense-LFM2-v2 当前模型库没有对应 GGUF | 文档列入 `Experimental`，不可用 |

## 9. 相关源码索引

- `src/models/lfm2/trunk/` — LFM2 文本 trunk
- `src/models/lfm2/vision.rs` — LFM2.5-VL vision（text decoder 复用 lfm2/ trunk）
- `src/models/lfm25/` — LFM2.5 文本 trunk
- `src/models/lfm2moe/` — MoE 文本 trunk
- `src/app/text.rs:61-83` — LFM2 / LFM2.5 文本路由（basename 分流）
- `src/app/text.rs:809-824` — LFM2.5-VL 视觉路由
- `docs/ISSUE.md` — LFM2 / LFM2.5 命名不一致（arch vs 目录名）

## 10. JEV 决策评分

`--jev` 是 OpenJEV 风格的 single-forward-pass 决策评分模式。通用协议、
3 种 mode、JSON 输出、已知限制见 [`docs/develop/jev.md`](../develop/jev.md)
和 [`docs/usage/qwen3.md` §10](qwen3.md)。

### 10.1 Arch 路由

| Arch | JEV 路径 |
|---|---|
| `lfm2` | `app/jev/single/lfm2.rs::run_jev_decision_lfm2` → `lfm2::run_forward_logits_lfm2` |
| `lfm25` (含 `"2.5"` 的 basename) | `app/jev/single/lfm2.rs::run_jev_decision_lfm25` → `lfm25::run_forward_logits_lfm25` |
| `lfm2moe` | ❌ 暂未支持（JEV 路由只覆盖 `lfm2` / `lfm25`） |

LFM2 与 LFM2.5 的 chat template **完全相同**（`{role}\n{content}\n` 序列），
与 Qwen3 模板几乎一致。

### 10.2 示例

```bash
# LFM2-1.2B dense — Choice mode
rust-model-inference --model models/lfm2/LFM2-1.2B-Q8_0.gguf \
  --jev --jev-context "用户咨询账户安全问题" \
  --jev-question "这是哪种类型的请求？" \
  --jev-option "密码重置" --jev-option "2FA 启用" --jev-option "可疑活动" \
  --threads 4

# LFM2.5-1.2B-Instruct — Binary mode
rust-model-inference --model models/lfm2.5/LFM2.5-1.2B-Instruct-Q8_0.gguf \
  --jev --jev-context "用户已通过身份验证" \
  --jev-question "问题是否已解决？" \
  --jev-option "是" --jev-option "否" --jev-positive A \
  --threads 4

# LFM2-8B-A1B (MoE) — 暂不支持 JEV
# Will return: --jev is not yet supported for architecture "lfm2moe"
```

> 注意：LFM2 是 hybrid（attention + shortconv）架构，JEV prefill 走
> 完整 prefill 路径（不是 KV cache 共享）。每个 question 都是一次
> 独立 prefill，多 question 模式下耗时为 N × prefill_time。