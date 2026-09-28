# 待接入模型清单（TODO）

> 数据来源：`references/llama.cpp/src/llama-model.cpp` 当前 **153** 个 `LLM_ARCH_*`
> 对照 `src/core/loader.rs` allowlist（18）与 `src/models/llama/trunk/forward.rs` 中
> 特殊 arch switch（14）。**缺口 = 128 个 arch 无推理支持。**
>
> MoE 模型给出「总参数 / 激活参数」两个口径——想跑低算力部署时激活参数才是有效指标。
> 本清单按 **激活参数 ≤ 32B** 收录。

## A. 代码已写好但 `MODEL_LIST.md` 没列（零成本，先补文档）

| arch | 实际状态 | 动作 |
|---|---|---|
| `lfm2moe` | **完整 trunk**（`src/models/lfm2moe/` 5 文件）+ `Lfm2MoeTextRuntime` + JEV dispatch | MODEL_LIST.md 加 LFM2-MoE 行 |
| `yue2` | **完整模块**（`src/models/yue2/` + `app/yue2.rs` CLI + 2 个测试） | MODEL_LIST.md 加 YuE2 行 |
| `pig` | = Z-Image 文生图路径 | MODEL_LIST.md 的 Z-Image 从「待核验」改为真实状态 |

已核实**不是**缺口的（此前误判，留档避免重复排查）：
- `hunyuan-dense` → 已有 `src/models/qwen3/hunyuan.rs` + `docs/usage/hunyuan.md`，MODEL_LIST.md 的 Hy-MT2 行即覆盖。

## B. 1 梯队：主流新架构，激活 ≤32B，GGUF 可获取

| arch | 模型 | 总/激活 | 接入要点 |
|---|---|---|---|
| `qwen3moe` | **Qwen3-30B-A3B** / Instruct | 30B/3B | 复用 qwen3 trunk，加 MoE 路由 |
| `mistral3` | **Mistral-Small-3.1/3.2-24B** | 24B dense | 标准 GQA，无新 op |
| `mistral4` | Mistral-Small-4 | ~24B | 同上 |
| `glm4` | **GLM-4-9B** / GLM-4-32B-0414 | dense | ChatGLM4-9B GGUF 多 |
| `hunyuan-moe` | **Hunyuan-A13B**（Hy-MT2 MoE 版） | 80B/13B | 与已有 hunyuan-dense 同源，加 MoE |
| `ernie4-5` | **ERNIE-4.5-21B-A3B** | 21B/3B | |
| `xverse` | XVERSE-**7B/13B** / MoE-A4.2B | 65B/4.2B | |
| `llama4` | **Llama-4-Scout** | 109B/17B | MoE + chunked attn + iRoPE（需先有 MoE） |
| `dots1` | 小红书 dots.llm1 | 142B/7B | |
| `cohere2` | **Cohere Command R7B** | 7B | Command A(111B) 排除 |
| `minicpm3` | **MiniCPM3-4B / MiniCPM4-8B** | dense | |
| `deepseek` | DeepSeek-LLM-7B | 7B | |
| `deepseek2` | **DeepSeek-V2-Lite-16B** | 16B/2.4B | MLA 是 V3/R1 的缩水前置 |
| `seed-oss` | Seed-OSS-36B | 36B/36B | 激活也超，**排除** |

> 说明：`hunyuan-moe` 对应 Hy-MT2 的 30B-A3B-MoE 变体，`docs/usage/hunyuan.md` 已提到该规模存在但未接入。
> `llama4`、`dots1`、`xverse` 总参数虽大但激活 ≤17B，走 GGUF Q4 量化后内存可控（Scout Q4 ≈ 55 GB）。

## C. Encoder / Embedding 家族（一次实现覆盖 5+ 模型，ROI 最高）

与已有 Qwen3-Embedding 同赛道，粒度小、无生成逻辑：

| arch | 模型与规格 |
|---|---|
| `bert` / `modern-bert` | BERT 110M–1.3B / ModernBERT 150M–1.4B |
| `jina-bert-v2` / `jina-bert-v3` | jina-embeddings-v2-base-zh(893M) / v3(570M) |
| `nomic-bert` / `nomic-bert-moe` | nomic-embed 137M / 440M / MoE-305M |
| `neo-bert` / `eurobert` | 436M / 210M–1.2B |
| `pangu-embed` | 华为盘古 embedding-7B |
| `gemma-embedding` | Google EmbeddingGemma-300M |
| `llama-embed` | Llama-Embed-Nemotron-3B |

## D. 经典 / SSM / 小模型（有兴趣再接）

- **dense 老牌**：`chameleon`-8B、`falcon`(TII-7B/11B，**与已有 falcon_h1 不同 arch**)、
  `bloom`、`starcoder`/`starcoder2`-15B、`gpt2`、`gptj`-6B、`mpt`-13B、
  `codeshell`-7B、`deci`-5.7B
- **中文老牌**：`baichuan`-7B/13B、`internlm2`-7B/20B、`spark2-5`
- **SSM / 混合（非标准 attention，需新 op）**：`mamba`(–7B)、`mamba2`、
  `rwkv6`/`rwkv7`(–14B)、`jamba`(52B，激活略超)
- **其它小模型**：`jais`/`jais2`、`olmo`/`olmo2`(-32B)、`olmoe`(7B/1B)、
  `orion`-14B、`apertus`-8B、`arcee`-7B、`refact`-1.6B、`smollm3`-3B、
  `mellum`-4B、`maincoder`-7B、`plamo`/`plamo2`/`plamo3`(8B–13B)
- **Granite 变体**（`granite` 已接入）：`granite-moe`(1B/3B)、`granite-hybrid`、
  `granite-swa`(-8B)、`granite-switch`

## E. 明确排除

- **激活参数也 >32B**：`minimax-01`/`minimax-m2`/`minimax-m3`、`arctic`、`dbrx`,
  `command-r`/`command-r`+、`step35`、`glm4-moe`(106B/**12B** — 激活其实 ≤32B，降级到 B 梯队可选)、
  `kimi-k3`/`kimi-linear`、`bailingmoe`/`bailingmoe2`/`bailingmoe3`、`afmoe`、`grok`、
  `grovemoe`、`openai-moe`、`llada`/`llada-moe`、`rwkv6qwen2`、`dream`、`eagle3`
- **音频 / 视觉专用，需非文本 trunk**：`clip`、`paddleocr`、`talkie`、
  `wavtokenizer-dec`、`pockettts`、`t5encoder`
- **llama.cpp 试验分支 / 内部占位**：`unknown`、`dflash`、`hrm-text`、`muse-glimmer`、`rnd1`
- **纯 decoder-only 编码任务需另做训练框架**：`t5`、`gpt2`(可作为 encoder 复用)

## 建议执行顺序

1. **补文档**（零代码）：A 区三项，特别是 `lfm2moe`（代码已完整）。
2. **`qwen3moe`（Qwen3-30B-A3B）** — 已有 qwen3 trunk，边际成本最低，且 MoE 路由是后续
   `llama4`/`hunyuan-moe`/`glm4-moe` 的共同前置。
3. **`mistral3`（Mistral-Small-24B）** — 标准架构，1 天量级，覆盖面大。
4. **Encoder/Embedding 家族（C 区）** — 一次实现覆盖 7+ 模型，和 Qwen3-Embedding 并列。
5. **`hunyuan-moe`（Hy-MT2 MoE 版）** — 复用已有 `src/models/qwen3/hunyuan.rs`，加 MoE 即完成。
6. 其余按 B → D 顺序视需求推进。

## 排查方法（下次更新用）

```bash
# llama.cpp 支持的全部架构
grep -oE 'LLM_ARCH_[A-Z0-9_]+' references/llama.cpp/src/llama-model.cpp \
  | sed 's/LLM_ARCH_//' | tr 'A-Z_' 'a-z-' | sort -u

# 本地 allowlist
grep -oE '^\s+\| "[a-z0-9_.-]+"' src/core/loader.rs | grep -oE '"[^"]+"' | tr -d '"'

# 本地 llama trunk 特殊 arch
grep -oE 'arch == "[a-z0-9_-]+"' src/models/llama/trunk/forward.rs | sort -u

# 确认某 arch 是否真有代码（避免把 tokenizer/chat_template 命中当成支持）
grep -rl "\"<arch>\"" src/ | grep -v chat_template | grep -v tokenizer
```
