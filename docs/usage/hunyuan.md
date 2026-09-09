# Hunyuan / Hy-MT2 用法

`Hy-MT2-1.8B` 是腾讯混元的多语言翻译模型（1.8B / 7B / 30B-A3B-MoE 三种规模），
仓库里目前接入并测试的是 **1.8B Q8_0**。GGUF `general.architecture = hunyuan-dense`，
CLI 路由到 `src/models/qwen3/hunyuan.rs`，内部委托 qwen3 trunk forward，
因此**完整继承 qwen3 的 SIMD 优化**（ExecutionScratchpad + F16 KV cache + dot_f16 +
silu_mul_approx_inplace + AVX2/NEON matmul）。

> 共用前置：构建 `cargo build --release --bin rust-model-inference`。
> KV cache 默认 F16；与 llama.cpp 位级对比时显式传 `--kv-cache f16`。
> 当前仅 CPU 路径；Vulkan dispatch 不在 `hunyuan-dense` 的覆盖范围内。

---

## 1. 翻译任务（主用途）

Hy-MT2 的核心用法是「翻译指令 + 源文本」一条 prompt 直接喂入模型，**不走对话模板构造**。
模型卡（`models/Hy-MT2-1.8B-GGUF/README.md`）给出的指令模板分中英两套，
`source_lang` / `target_lang` 字段使用**语言全称**（中文用中文名、英文用英文名）。

### 1.1 中文 prompt 模板

```
将以下文本翻译为 {target_lang}，注意只需要输出翻译后的结果，不要额外解释：

{source_text}
```

`target_lang` 用中文：`英语`、`日语`、`法语`、`德语`、`俄语`、`韩语` 等。

### 1.2 英文 prompt 模板

```
Translate the following text into {target_lang}. Note that you should only output the translated result without any additional explanation:

{source_text}
```

`target_lang` 用英文：`English`、`Japanese`、`French`、`German`、`Russian`、`Korean` 等。

### 1.3 实测样例

仓库已用 `models/Hy-MT2-1.8B-GGUF/Hy-MT2-1.8B-Q8_0.gguf` 跑过以下三组（CPU 8 线程）：

| 方向 | 源文本 | 输出 |
|---|---|---|
| 中 → 英 | 你好世界 | `Hello, World` |
| 英 → 中 | The quick brown fox jumps over the lazy dog | `敏捷的棕色狐狸跳过了懒狗` |
| 中 → 英 | 机器学习是人工智能的一个分支。 | `Machine learning is a branch of artificial intelligence.` |

### 1.4 CLI 调用

**中文 prompt（中 → 英）：**

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/Hy-MT2-1.8B-GGUF/Hy-MT2-1.8B-Q8_0.gguf \
  --prompt "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：

机器学习是人工智能的一个分支。" \
  --max-tokens 64 --threads 8
```

**英文 prompt（英 → 中）：**

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/Hy-MT2-1.8B-GGUF/Hy-MT2-1.8B-Q8_0.gguf \
  --prompt "Translate the following text into Chinese. Note that you should only output the translated result without any additional explanation:

The quick brown fox jumps over the lazy dog" \
  --max-tokens 64 --threads 8
```

> CLI 会把 `--prompt` 包进 Hunyuan chat template（`<hy_user>…<hy_assistant>`）。
> 这种「指令字符串直接当作 user message」的形式已经过实测，模型能正确响应。
> 不要手动拼 `<hy_user>` / `<hy_assistant>` 控制 token——CLI 已经做了。

---

## 2. 非翻译任务

Hy-MT2 也能跑通用 chat（毕竟底层是 Hunyuan Dense 1.8B），但仓库**没有针对
翻译以外的对话场景做验证**。如果使用 `--prompt "你好"` 这种闲聊 prompt，
会得到通用回答；模型卡对此行为没有承诺，请按需自测。

```bash
# 通用闲聊（未做翻译以外任务的回归验证）
cargo run --release --bin rust-model-inference -- \
  --model models/Hy-MT2-1.8B-GGUF/Hy-MT2-1.8B-Q8_0.gguf \
  --prompt "你好" --max-tokens 64 --threads 8
```

---

## 3. 支持的语言

Hy-MT2 支持 33 种主要语言 + 5 种方言（藏语、哈萨克语、蒙古语、维吾尔语、粤语）。
完整列表见 `models/Hy-MT2-1.8B-GGUF/README.md`。

prompt 里 `target_lang` 字段名要使用**对应语言的全称**：

| 目标语言 | 中文 prompt 写法 | 英文 prompt 写法 |
|---|---|---|
| 中文 | 中文 | Chinese |
| 英语 | 英语 | English |
| 日语 | 日语 | Japanese |
| 法语 | 法语 | French |
| 德语 | 德语 | German |
| 俄语 | 俄语 | Russian |
| 韩语 | 韩语 | Korean |
| 西班牙语 | 西班牙语 | Spanish |
| 阿拉伯语 | 阿拉伯语 | Arabic |
| 葡萄牙语 | 葡萄牙语 | Portuguese |

完整 33 + 5 项见模型 README。

---

## 4. 量化与性能

模型卡（README）提供的 GGUF 量化版本（均为 llama.cpp 原生工具生成）：

| 量化类型 | 文件大小（约） | 推荐场景 |
|---|---|---|
| **Q4_K_M** | 1.1 GB | **推荐**：精度与速度平衡 |
| Q3_K_M | 0.9 GB | 资源受限设备 |
| Q5_K_M | 1.3 GB | 高精度 |
| Q6_K | 1.5 GB | 接近原始质量 |
| Q8_0 | 1.9 GB | 近无损 |

仓库**已实测 Q8_0**（2026-09 验证）：

- 模型加载：~50–75 ms
- 单 batch 翻译（prompt 27–33 token）：8–9 tok/s（CPU 8 线程，无 GPU）
- 内存占用：≈ 模型权重 + scratch ≈ 2 GB 工作集

**未实测**的量化格式（Q4_K_M / Q3_K_M / Q5_K_M / Q6_K）应视为 `Supported`
而非 `Verified`，跑通后再升级状态。

---

## 5. CLI 路由速查

| GGUF `general.architecture` | 进入 trunk | 入口 |
|---|---|---|
| `hunyuan-dense` | qwen3 trunk（委托） | `src/models/qwen3/hunyuan.rs` → `qwen3::text::run_inference_tokens` |

`src/app/text.rs` 按 `arch == "hunyuan-dense"` 把 CLI 路由到 hunyuan.rs。
forward / matmul / KV cache / RMSNorm / RoPE 全部走 qwen3 的 SIMD 实现。

---

## 6. 与 llama.cpp 的对齐

Hy-MT2 是 Tencent 官方基于 [tencent/Hy-MT2-1.8B](https://www.modelscope.cn/models/Tencent-Hunyuan/Hy-MT2-1.8B) 的
llama.cpp GGUF 量化版本。`docs/REFERENCE_IMPLEMENTATIONS.md` 当前**未固定**该
模型对应的 llama.cpp commit 与 Oracle 脚本，因此状态从 `Supported` 升到
`Verified` 还差：

1. 选一个含 `hy-mt2` 实现的 llama.cpp commit 并 pin
2. 写 `tools/hunyuan/build_oracle.sh`（按 README 表格中的指令格式跑翻译）
3. 加 `tests/hunyuan_reference.rs` 做 token 级对照

当前 `docs/SUPPORTED_MODELS.md` 把 1.8B Q8_0 标为 `Verified` 是基于：

- 真实 GGUF 加载 + 端到端生成成功（无错误退出）
- 中→英 / 英→中 翻译结果内容正确（详见 §1.3）

不包含：与 llama.cpp 的位级 token 序列对照。

---

## 7. 已确认的限制 / 边界

| 范围 | 行为 |
|---|---|
| 量化 | 已验证 Q8_0；其他格式（Q4_K_M / Q5_K_M 等）未实测 |
| GPU | 不支持；`hunyuan-dense` 没有 Vulkan dispatch 入口（不像 `qwen3` / `qwen35`） |
| Oracle pin | `Pending pin`（llama.cpp 未固定 commit） |
| 语言覆盖 | 33 主语言 + 5 方言按 README 声明；实测仅中↔英 |
| 用途 | 仅翻译任务经过回归验证；通用 chat 未验证 |

---

## 8. 相关源码索引

- `src/models/qwen3/hunyuan.rs` — Hunyuan CLI 入口（委托 qwen3 trunk）
- `src/models/qwen3/text.rs` — `run_inference_tokens`（Qwen3Session 包装）
- `src/models/qwen3/trunk/session.rs` — 完整 SIMD forward loop
- `src/models/qwen3/trunk/forward.rs` — prefill 入口 + Vulkan dispatch（qwen3 部分）
- `src/prompt.rs:41` — `build_hunyuan_chat_prompt`（`<hy_user>…<hy_assistant>` 模板）
- `docs/REFERENCE_IMPLEMENTATIONS.md` — 参考实现清单
- `docs/SUPPORTED_MODELS.md` — 验证状态
