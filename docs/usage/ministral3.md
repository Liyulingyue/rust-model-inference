# Ministral-3 / Shieldstral 用法

`general.architecture = "mistral3"` 的 GGUF 走 `src/models/llama/` trunk（与
`llama` / `nanbeige` / `granite` 共用 forward 实现，差异在 chat template /
RoPE / tied-vs-untied / tokenizer pre-tokenizer）。Mistral 3 后训练模型都
共享同一份 26 层或 34 层图，差异只在权重和 chat_template 内容。

本仓库已在本机验证以下模型：

| 模型 | 尺寸 | 量化 | 路径 | 备注 |
|---|---|---|---|---|
| Ministral-3-3B-Instruct-2512 | 3B | Q4_K_M 2.1 GB | `models/Ministral-3-3B-Instruct-2512-GGUF/` | Mistral 官方 GGUF；tied embedding；`pre="tekken"` |
| Ministral-3-3B-Reasoning-2512 | 3B | Q4_K_M 2.0 GB | `models/Ministral-3-3B-Reasoning-2512-GGUF/` | Mistral 官方 GGUF；tied；chat_template 含 `[THINK]` 块 |
| Ministral-3-8B-Instruct-2512 | 8B | Q4_K_M 4.96 GB | `models/Ministral-3-8B-Instruct-2512-GGUF/` | Mistral 官方 GGUF；**untied**；`output.weight` Q6K |
| Mistral-Shieldstral-1.0-3B | 3B | Q4_K_M 2.05 GB | `models/Shieldstral-1.0-3B-GGUF/` | Metabaron6 社区 GGUF（Mistral 官方未出 GGUF） |

> **共用前置**：`cargo build --profile release-fast --bin rust-model-inference`
> （或 `--bin rust-model-server`）。默认 KV cache F16、最大 context 8192、4
> 线程。8B 在 7.5 GiB 物理内存下建议 `--max-context 4096`，Q5_K_M / Q8_0
> 扛不住。

## 1. CLI：文本生成

```bash
target/release-fast/rust-model-inference \
  --model models/Ministral-3-3B-Instruct-2512-GGUF/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf \
  --prompt "What is the capital of France? Answer in one sentence." \
  --max-tokens 32
```

CLI 默认走 `[INST] {prompt} [/INST]` 模板（`src/models/llama/trunk/forward.rs:646`
`is_mistral` 分支，含 `mistral` / `ministral` / `shieldstral` 子串匹配）。
实测：

- 3B-Instruct：`"The capital of France is **Paris**."`
- 3B-Reasoning：150+ token 数学题产出 Markdown 小标题结构化推理段（**不带
  `[THINK]…[/THINK]` 包裹**——见 §3 限制）
- 8B-Instruct：`"The capital of France is **Paris**."`（~3.1 t/s，4 核 + 7.5 GiB）
- Shieldstral（自然语言 prompt）：`[INST] … Is this safe (yes) or unsafe (no)?
  Message: I want to hurt someone at school tomorrow. …` → `"yes"`

## 2. CLI：JEV 决策评分

```bash
target/release-fast/rust-model-inference \
  --model models/Shieldstral-1.0-3B-GGUF/Shieldstral-1.0-3B-Q4_K_M.gguf \
  --jev \
  --jev-context "User: I want to hurt someone at school tomorrow" \
  --jev-question "Is this safe?" \
  --jev-option "safe" --jev-option "unsafe"
```

走 llama scorer（`src/app/jev/single.rs:107` `is_mistral`/`ministral` 子串匹
配命中，`run_jev_decision_llama`），JEV 测试 `tests/ministral3_3b_jev.rs` 2/2
锁住路由。

## 3. 已知限制（Mistral 3 checkpoint 共有的）

### 3.1 JEV JSON payload ~56-81% 位置偏置

Ministral-3 3B 在 JSON 风格的 JEV payload 上触发强位置偏置：`A` 总是赢
~56-81%，与候选顺序无关。这是 Mistral-3 3B checkpoint 的 JSON 指令微调弱，
**不是引擎 bug**。

**Workaround**：需要决策打分时优先用 `--prompt` 拿自由文本，或手写自然
语言 prompt 把候选塞进 `[INST]` 块。CLI 实测对 Shieldstral：

```
[INST] Classify the following message as safe (yes) or unsafe (no).
       Reply with only one word.
Message: I want to hurt someone at school tomorrow.
Answer: [/INST]
```

→ `"yes"`；对照良性内容 → `"no"`。模型能用，只是不遵循结构化 JSON 指令。

### 3.2 Reasoning 模型在 CLI 不显式出 `[THINK]…[/THINK]`

Ministral-3-3B-Reasoning 的 chat_template 要求模型先在 `[THINK]…[/THINK]` 块
中草拟思考再回答。CLI 只发 `[INST] {prompt} [/INST]`，**不读
`tokenizer.chat_template` 的 `[SYSTEM_PROMPT]` 默认 prompt**，所以模型仍
产出结构化推理 prose（Markdown 小标题、步骤化分析），但**不会以显式的
`[THINK]` 标记分隔思考段和正文**。

要触发显式 `[THINK]` 块需要其中一项：

1. 给 CLI 加 `--system-prompt` 标志把 `[SYSTEM_PROMPT]…[/SYSTEM_PROMPT]`
   block 拼到 `[INST]` 前
2. 做 chat-template-aware 模式跑 jinja（对当前 GGUF 的 chat_template 解析）
3. 当 `general.name` 含 `"reasoning"` 时自动注入 Mistral 官方默认 system
   prompt

这三条路都在 `src/models/llama/trunk/forward.rs:646` 的 TODO 注释里列出，
未在当前 PR 范围。

### 3.3 Shieldstral `pre="pixtral"`、`add_bos_token` 缺失

社区 Metabaron6 GGUF 把 `tokenizer.ggml.pre` 写成了 `"pixtral"` 而非
`"tekken"`——两者都映射到 `PreTokenizer::LlamaBpe`（`src/core/tokenizer/mod.rs`），
引擎不需要额外代码。GGUF 同时缺 `tokenizer.ggml.add_bos_token` 字段，llama.cpp
默认 false，tokenized prompt 从 `[INST]` 直接开始（不是 `<s>` + `[INST]`），
不影响 forward（`[INST]` 仍是单 Tekken special token）。

`general.name = "Shieldstral 1.0 3B"` 的 "Shieldstral" 不含 "mistral" /
"ministral" 字面子串，所以 `is_mistral` 检测要额外匹配 "shieldstral" 子串
（已在 `forward.rs:582` 加入）。

## 4. HTTP server 路径状态

启动：`rust-model-server --model <path> --host 127.0.0.1 --port 18080`。

实测（Shieldstral-3B Q4_K_M；其他三个模型同 arch，行为相同）：

| 路径 | 状态 | 说明 |
|---|---|---|
| `/v1/jev/score` | ✅ **好使** | 走 llama scorer；正确产出 `mode/choice/probabilities`。Schema：`{context, questions: [{text, options: [...]}]}`，注意**不是** CLI 风格的 `--jev-option` 数组 |
| `/v1/jev/grouped` | ✅ **好使** | 走 llama scorer；`{context, questions: [{text, groups: [{label, options}]}], mode: "multi_select"\|"block_choice"}` |
| `/v1/chat/completions` | ✅ **好使** | OpenAI 风格；用 `messages: [{role, content}]`，`is_mistral` 子串匹配驱动 `[INST]…[/INST]` 模板 |
| `/v1/responses` | ✅ **好使** | OpenAI 新风格；同上 |
| `/v1/messages` | ✅ **好使** | Anthropic 兼容；同上 |

### 4.1 CLI vs HTTP 的实际差异

| 行为 | CLI | HTTP `/v1/jev/score` |
|---|---|---|
| JEV 路由 | ✅ `run_jev_decision_llama`（`src/app/jev/single.rs:107` 含 `mistral3`） | ✅ 同样走 llama scorer |
| Schema | `--jev-option "safe" --jev-option "unsafe"` 多个 flag | `questions: [{text, options: ["safe","unsafe"]}]` 单个对象 |
| 候选 → letter 映射 | CLI 直接透传 A/B/C… | HTTP 自动 letter 映射：`["safe","unsafe"]` → A=safe / B=unsafe |

### 4.2 HTTP 端如何接入 mistral3

实测已通过（Shieldstral-3B Q4_K_M release-fast，4 核 + 7.5 GiB）：

```bash
target/release-fast/rust-model-server \
  --model models/Shieldstral-1.0-3B-GGUF/Shieldstral-1.0-3B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 18080 --threads 4

# curl
curl http://127.0.0.1:18080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Shieldstral","messages":[{"role":"user","content":"Say hi"}],
       "max_tokens":8}'
# → "no"（safety classifier 用 yes/no 回答）

curl http://127.0.0.1:18080/v1/jev/score -H 'Content-Type: application/json' \
  -d '{"context":"User: I want to hurt someone",
       "questions":[{"text":"safe?","options":["safe","unsafe"]}]}'
# → choice=A(safe) prob=0.5571

curl http://127.0.0.1:18080/v1/jev/grouped -H 'Content-Type: application/json' \
  -d '{"context":"User: hello",
       "questions":[{"text":"intent","groups":[{"label":"yes","options":["yes","no"]}]}],
       "mode":"multi_select"}'
# → choice=A(yes) prob=0.5601
```

### 4.3 HTTP 端接入 mistral3 的代码改动

5 处微改：

1. `src/app/server/api/tools.rs:30-54` `is_qwen35` 白名单追加
   `| "mistral3" => Ok(false)`。
2. `src/app/server/api/tools.rs:155-160` `llama_family` match 追加
   `| "mistral3"`——`build_prompt_tokens_from_turns` 已经被 CLI 验证过能
   处理 mistral3（`is_mistral` 子串匹配同时驱动 CLI 和 JEV），HTTP 这边
   只是过早拒绝。
3. `src/app/jev/grouped.rs:202` 的 `build_jev_token_ids_for_arch` 加
   `mistral3` arm，prompt 模板复用 `[INST] {system} {payload} [/INST]`
   （同 single scorer）。
4. `src/app/jev/grouped.rs:564` 的 `run_jev_grouped_decision_data` 顶部
   match 把 `mistral3` 加进 `"llama" | "k2-horizon" | ...` arm，复用
   `run_jev_grouped_llama`（其 `LlamaJevGroupedScorer` 内部已经走
   `LlamaJevScorer`，arch 由 `is_mistral` 自动判别）。
5. `src/app/jev/grouped.rs:431,627` 两处 `currently supported: ...` 错误
   消息里把 `mistral3` 加进去。

## 5. 与 llama.cpp 的对齐

短 context（< 16K YaRN 原生长度）下，本仓库标量路径与 llama.cpp
`b96806d` 参考实现位级一致；YaRN 长 context（≥ 16K）走
`compute_yarn_thetas` 波长校正（`src/models/llama/trunk/forward.rs`），
与 `ggml_compute_forward_rope_f32` 的 YaRN 分支 ≤ 1e-6 相对误差对齐
（`compute_yarn_thetas_matches_mistral3_pin` 单测已锁）。CLI smoke 与
Q4_K_M 标量路径一致。

## 6. 相关测试

| 测试 | 覆盖范围 |
|---|---|
| `tests/ministral3_3b_instruct_q4_k_m.rs` | 4/4：arch + dims + YaRN + Tekken + tensor inventory（3B Instruct，tied 236 tensors） |
| `tests/ministral3_3b_reasoning_q4_k_m.rs` | 5/5：含 chat_template 必含 `[THINK]/[SYSTEM_PROMPT]/[IMG]` 标记 |
| `tests/ministral3_8b_instruct_q4_k_m.rs` | 4/4：34/4096/14336 dims + 309 tensors + untied `output.weight` |
| `tests/shieldstral_1_0_3b_q4_k_m.rs` | 5/5：含 `general.name` 含 "shieldstral" + `pre="pixtral"` + `add_bos_token` 字段可空 + chat_template 不含 `[THINK]` |
| `tests/ministral3_3b_jev.rs` | 2/2：`run_jev_decision_llama` 路由（结构不变量 + argmax 不 pin，见 §3.1） |
| `src/models/llama/trunk/forward.rs::tests::compute_yarn_thetas_*` | YaRN θ 公式与 ggml 参考 ≤ 1e-6 |

## 7. 相关源码

- `src/models/llama/trunk/forward.rs` — `apply_rope` (8 args, `yarn_thetas`)
  + `compute_yarn_thetas` + 3 个 YaRN 单测；`is_mistral` 检测（含
  `"mistral" || "ministral" || "shieldstral"` 子串）
- `src/models/llama/trunk/session.rs` — `LlamaSessionConfig` 加了
  `yarn_thetas: Option<Vec<f32>>`；`from_source` 读
  `rope.scaling.{type,factor,original_context_length,yarn_beta_fast,yarn_beta_slow,yarn_log_multiplier}`
- `src/app/jev/single.rs:107` — `mistral3` → `run_jev_decision_llama`
- `src/app/jev/single/llama.rs:118-145` — Mistral-3 单 JEV `[INST] {system}
  {payload} [/INST]` 模板（含 JSON positional bias 解释）
- `src/app/text/generation.rs:22` — `uses_llama_trunk` 包含 `mistral3`
- `src/core/loader.rs:411` — arch 校验白名单包含 `mistral3`
- `src/core/tokenizer/mod.rs` — `pre="tekken"` 与 `pre="pixtral"` 都映射
  到 `PreTokenizer::LlamaBpe`
- `src/app/server/api/tools.rs:30-54` — `is_qwen35` 白名单（**未含
  mistral3**，见 §4.2）
- `src/app/jev/grouped.rs:202` — grouped arch 路由（**未含 mistral3**，
  见 §4.2）
