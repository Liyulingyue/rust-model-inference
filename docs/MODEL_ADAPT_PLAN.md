# 待接入模型清单（TODO）

> 数据来源：`references/llama.cpp/src/llama-model.cpp` 当前 **153** 个 `LLM_ARCH_*`
> 对照 `src/core/loader.rs` allowlist 与 `src/models/llama/trunk/forward.rs` 中
> 特殊 arch switch。**缺口随每次适配递减，本文件按需刷新，不要在过期计数上做决策。**
>
> MoE 模型给出「总参数 / 激活参数」两个口径——想跑低算力部署时激活参数才是有效指标。
> 本清单按 **激活参数 ≤ 32B** 收录。

## A. 代码已写好但文档状态不实（2026-09-28 复核，多数已过期）

2026-09-28 逐项复核后，本区**只剩 `yue2` 一项真的缺文档**；另外三项此前误判：

- `lfm2moe` → **早已登记为 Verified**：`docs/develop/SUPPORTED_MODELS.md` 的 `LFM2-8B-A1B | lfm2moe`
  行（真实 GGUF 可完整生成、与 llama.cpp 前若干 token 对齐）。此处重复登记过一次。
- `pig` → **早已登记为 Verified**：同一张表的 `Z-Image Turbo | pig` 行（Q8_0 DiT + 文本编码器 + F16 VAE）。
- `hunyuan-dense` → 已有 `src/models/qwen3/hunyuan.rs` + `docs/usage/hunyuan.md`，
  MODEL_LIST.md 的 Hy-MT2 行即覆盖，`SUPPORTED_MODELS.md` 里也是 `Verified`。

剩余动作：

| arch | 实际状态 | 动作 |
|---|---|---|
| `yue2` | **完整模块**（`src/models/yue2/` 6 文件：ar/nar/vae/protocol/config）+ 专用 CLI `DispatchMode::Yue2`（`src/main.rs:147`） | 已补 MODEL_LIST.md / SUPPORTED_MODELS.md 行。**本地无 GGUF，因此只标 `Supported` 不标 `Verified`** |

> 经验：这个仓库的文档比代码慢，但比这里写的更快。**每次按本节动手前先 grep
> `docs/develop/SUPPORTED_MODELS.md`**，否则会重复补已登记的行（我这次就补错了三项）。

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

2026-09-29 状态：**C 区已兑现 4/7**。`nomic-bert-moe` 完成（含 UGM tokenizer + 偶奇层
dense/MoE 调度）。`src/models/bert_family/` 一份 graph 吃掉了 4 个 arch。

与已有 Qwen3-Embedding 同赛道，粒度小、无生成逻辑：

| arch | 模型与规格 | 状态 |
|---|---|---|
| `bert` | bge-small-en-v1.5 33M、**bert-base-uncased 110M**（均已验证） | **已接 2026-09-28 / 2026-09-29**（Verified，`tests/bge_small_en_v1_5.rs` 4/4 + `tests/bert_base_uncased.rs` 6/6；后者首次真正跑到 `position_embd.weight` 与 mean-pooling 默认值） |
| `jina-bert-v2` | jina-embeddings-v2-base-zh(893M) | **已接**（Verified，`tests/jina_v2_base_en.rs` 4/4） |
| `nomic-bert` | nomic-embed-text-v1.5 137M | **已接 2026-09-28**（Verified，`tests/nomic_embed_text_v1_5.rs` 5/5） |
| `nomic-bert-moe` | nomic-embed-text-v2-moe 512MB | **已接 2026-09-29**（Verified，`tests/nomic_embed_text_v2_moe.rs` 5/5；含 UGM tokenizer + 偶奇层 dense/MoE 调度） |
| `modern-bert` | ModernBERT 150M–1.4B | 待接：llama.cpp 有独立 `modern-bert.cpp`，**不在** `bert.cpp` 共享 graph 里 |
| `jina-bert-v3` | jina-embeddings-v3 570M | 待接：在 `bert.cpp` 共享 graph 内，需 RoPE + GELU SEQ + `token_types` |
| `neo-bert` / `eurobert` | 436M / 210M–1.2B | 待接 |
| `pangu-embed` | 华为盘古 embedding-7B | 待接 |
| `gemma-embedding` | Google EmbeddingGemma-300M | **已接**（Verified，`tests/embeddinggemma_300m.rs` 4/4） |
| `llama-embed` | Llama-Embed-Nemotron-3B | 待接 |

**已还清的架构差异**（`bert_family` 的 3 个变体开关，就是 `bert.cpp` 的 arch 分支表）：

| 开关 | `bert` | `jina-bert-v2` | `nomic-bert` |
|---|---|---|---|
| 位置编码 | 绝对 `pos_embd` | 无（ALiBi） | 无（**RoPE**） |
| QKV 打包 | q/k/v 分离 | q/k/v 分离 | **fused `attn_qkv`** |
| FFN | GELU SEQ | GEGLU | **SwiGLU** |
| 投影 bias | 4 组 | 4 组 | **全无** |
| rope freq_base | - | - | **1000** |

**权重可用性（2026-09-29 实测，别再凭印象写「下一个」）**：
ggml-org 在 ModelScope 上共 195 个 repo，encoder 相关的只有
`bert-base-uncased`(110M) / `bge-small-en-v1.5-Q8_0-GGUF` /
`bge-m3-Q8_0-GGUF`(635MB) / `gte-small-Q8_0-GGUF`(37MB) /
`e5-small-v2-Q8_0-GGUF`(37MB) / `jina-embeddings-v2-base-{en,code}-Q8_0-GGUF` /
`jina-reranker-v1-turbo-en-GGUF` / `embeddinggemma-300m-GGUF` / `Nomic-Embed-Text-V2-GGUF`。
**ggml-org 名下没有 jina-bert-v3，也没有 modern-bert。**

- `jina-bert-v3`（570M）：全站唯一 GGUF 是社区转换
  `fuyuantech/jina-embeddings-v3-Q4_K_M`（392MB，Q4_K_M，非 ggml-org）。
  它是唯一能验证 **fused QKV + 独立 Q/K/V bias**（`ed973d4` 刚补的路径）和
  `n_rot` 半旋转的样本；jina 系历来带 Q/K/V bias，而 nomic 系不带，
  所以这条路径目前**仍无模型验证过**。代价是第三方 Q4_K_M + 392MB。
- `bert-base-uncased`（110M，ggml-org 官方）：**已接**，见 C 区表格。
- `gte-small-q8_0` / `e5-small-v2-q8_0`（各 37MB，ggml-org 官方）：同为 `bert`
  arch，是给 `bert` 变体补充更多 pooling / 维度假例的最便宜来源。

`nomic-bert-moe` 已落地过一次 MoE 路由（router logits → top-k → softmax →
per-expert `expert @ x → gelu → @ down`），`qwen3moe` 等大权重 MoE 可复用同一形状，
届时主要差异在专家张量的切分方式。

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

## 建议执行顺序（2026-09-29 三次刷新）

1. ~~补文档~~ **已完成**：A 区只剩 `yue2`，已如实标 `Supported`（无 GGUF 不标 `Verified`）；
   `lfm2moe` / `pig` / `hunyuan-dense` 三项经复核早已登记为 `Verified`，是本文件写错了。
2. ~~`bert` 变体验证~~ **已完成**（bge-small-en-v1.5），并因此发现一个波及三个变体的
   attention 残差 bug。
3. ~~`nomic-bert-moe`~~ **已完成 2026-09-29**：UGM tokenizer + 偶奇层 dense/MoE 调度已
   跑通 `tests/nomic_embed_text_v2_moe.rs` 5/5。GGUF 489MB，CLI/HTTP 768 维输出 bit
   一致，跨语言语义排序正确。**同时落地了 MoE 路由的最小可用形状**（router logits →
   top-k → softmax → per-expert `expert @ x → gelu → @ down`），是后续大权重 MoE 的前置。
4. ~~`bert-base-uncased`（110M，ggml-org 官方 Q8_0 113MB）~~ **已完成 2026-09-29**：
   `arch="bert"` 变体首次用真权重验证，`position_embd.weight` 路径 +
   mean-pooling 默认值 + 四组投影 bias 全部跑到，`tests/bert_base_uncased.rs` 6/6。
5. **`jina-bert-v3`（570M）** — 见上面「权重可用性」：全站只有社区转换的
   `fuyuantech/jina-embeddings-v3-Q4_K_M`（392MB）。它验证的是 `ed973d4` 补的
   fused QKV + 独立 Q/K/V bias 与 `n_rot`，这两条路径**至今没有任何模型验证过**。
   是否接受「非 ggml-org 的第三方 Q4_K_M」需要先拍板。
6. **C 区其余** — `gte-small` / `e5-small-v2`（各 37MB，ggml-org 官方，
   给 `bert` 变体补更多 pooling / 维度假例）→ `modern-bert`（需新建独立模块，
   `modern-bert.cpp` 不在共享 graph 内）→ `llama-embed` → `neo-bert` / `eurobert`
   → `pangu-embed`。
7. **MoE 大模型**（权重规模大，接入前先确认有对应规模的验证环境）：`qwen3moe`
   （Qwen3-30B-A3B，已有 qwen3 trunk，边际成本最低）→ `hunyuan-moe`
   （复用 `src/models/qwen3/hunyuan.rs`）→ `glm4-moe` → `llama4`。
8. **`mistral3`（Mistral-Small-24B）** — 标准架构，覆盖面大。
9. 其余按 B → D 顺序视需求推进。

## TODO：BERT 家族位级 oracle 对齐（**暂缓，勿与他方核对工作并行**）

状态：**待办，未开始**。挂在这里是因为「另有人在跑核对」，两边同时动 llama.cpp
oracle 侧容易互相覆盖 / 撞同一份 patch 与构建产物。**动手前必须先确认那侧已经收尾
或者让对方知道**，否则不要开始。

动机：C 区已接的 5 个 encoder（`bert` / `jina-bert-v2` / `nomic-bert` /
`nomic-bert-moe` / `embeddinggemma`）目前只做到「语义排序看起来对」，**没有一个是
位级对齐过的**。这不是形式问题——`attn_proj` 从未加回残差那个 bug 就是"语义相似度
依然很高、但整层恒等"的典型，位级对比是唯一能稳定抓住这类问题的手段。

现成的基础设施（不必从零搭）：

- `tools/oracle/shared/build_llama_oracle.sh` — 把 llama.cpp pin 到
  `749f688fcaa4c472ec034b08cb8a907c45cfaa02`、打 `llama-scalar-trace.patch`
  （关 NEON / `-ffp-contract=off`，保证标量可复现）、构建
  `llama-eval-callback`。**注意当前 checkout 是另一个 commit，需要先切。**
- `llama-eval-callback` 是**通用 graph dumper**，会把图里每个 op 的名称与数值打出来，
  不限于 patch 里已挂钩子的 `qwen3.cpp`；encoder 走的是 `bert.cpp`，
  预期至少能拿到 `inp_embd` → 各层 attention/FFN 中间量 → pooling 后的输出。
- 5 个 GGUF 权重均已在本地，合计约 1.1GB。

建议的落地顺序（从小到大，先在最便宜的模型上把流程跑通）：

1. `bge-small-en-v1.5`（35MB，`bert` 变体，带 `position_embd` 与 CLS pooling，
   是最容易暴露 pooling / 残差问题的样本）
2. `jina-embeddings-v2-base-en`（140MB，ALiBi + geglu + QKV bias）
3. `nomic-embed-text-v1.5`（140MB，fused QKV + RoPE(interleaved) + SwiGLU）
4. `embeddinggemma-300m`（319MB，QK-norm + 4-norm sandwich + 对称 SWA，
   不属 `bert_family`，是独立模块）
5. `nomic-embed-text-v2-moe`（489MB，MoE 路由 + UGM tokenizer，
   重点看 router logits 与 per-expert 输出）

对齐口径：同一 prompt、同一 tokens 序列、对比 `llama-eval-callback` 打出的中间张量
与本仓库实现；超出 f32 末位 ulp 的差异要定位到具体算子，而不是调阈值糊过去。
任一项对齐通过后，再把 `docs/MODEL_LIST.md` / `docs/develop/SUPPORTED_MODELS.md`
里对应行的「未做 llama.cpp 位级 oracle 对齐」划掉。

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
