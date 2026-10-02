# GLiNER 家族：11/12 已实现，融合点评估

本文记录在 11 个 checkpoint 全部适配完成后，对"哪些模块应当融合"的评估。
评估的前提是：**先把所有变体实现出来，才知道哪些重复是真实需求、哪些是过早抽象**。
现在数据齐了。

## 实现现状

| # | repo | 架构族 | encoder | 新代码量 |
|---|---|---|---|---|
| 1 | `GLiNER2.5-Decide` | Span | DeBERTa-v3-large 1024 | 基准实现 |
| 2 | `gliner2-large-v1` | Span pre-2.5 | DeBERTa-v3-large 1024 | **0**（仅放宽 config 校验） |
| 3 | `gliner2-base-v1` | Span pre-2.5 | DeBERTa-v3-base 768 | 转换器尺寸表 |
| 4 | `gliner2-multi-v1` | Span pre-2.5 | mDeBERTa-v3-base 768 | 转换器尺寸表 + marker id 修复 |
| 5 | `gliner2.5-base-v1` | Boundary | DeBERTa-v3-base 768 | ~8000 行（BoundaryExtractor） |
| 6 | `gliner2.5-multi-v1` | Boundary | mDeBERTa-v3-base 768 | 转换器尺寸表 |
| 7 | `GLiNER2.5-multi-Decide` | Boundary | mDeBERTa-v3-base 768 | **0**（复用 6） |
| 8 | `gliner2.5-small-v1` | Boundary | DeBERTa-v3-**xsmall** 384 | 转换器尺寸表 |
| 9 | `GLiNER2-Guardrails-PII-Multi` | Span pre-2.5 | mDeBERTa-v3-base 768 | **0**（tokenizer 声明风格） |
| 10 | `gliner2-privacy-filter-PII-multi` | Span pre-2.5 | mDeBERTa-v3-base 768 | **0** |
| 11 | `gliguard-LLMGuardrails-300M` | Span pre-2.5 | DeBERTa-v3-base 768 | **0** |
| 12 | `GLiNER2.5-Decide-1B` | Span | **Ettin-1B 1792** | **~2000 行，未实现** |

**10 个变体只花了两处改动**：转换器的尺寸表，和 tokenizer 声明风格的兼容。
真正的新代码只有 BoundaryExtractor（#5，四个 head）和尚未开始的 Ettin（#12）。

## 结论一：Rust 推理层**不需要**融合

这是评估后最明确的一条，而且与直觉相反。

- `src/models/gliner/`（2511 行）= DeBERTa forward + prompt builder + schema 路由
- `src/models/gliner_boundary/`（8482 行）= BoundaryExtractor 四个 head
- 两者**已经分层**：`gliner_boundary` 直接 `use crate::models::gliner::prompt::{...}`，
  没有第二份 prompt builder

尺寸无关性是**已经成立**的事实，不是待办：
`n_embd` / `n_layer` / `n_head` / `n_ff` / `head_dim` / `eps` / 相对注意力 bucket
全部从 GGUF metadata 读。`src/models/gliner/` 里唯一的 `1024` 字面量在
`mod tests` 里。因此 #3/#4/#6/#8/#9/#10/#11 七个变体**一行 Rust 都没改**。

> 这条推翻了原 TODO 的两处判断。它当时写"推理层 `weights.rs` 同理需要动态化"和
> `gliner2.5-small-v1` 是 "DeBERTa-v3-small"。前者不需要做，后者是 **xsmall**
> （384 宽 / 6 头 / 1536 FF）。**没有验证就写进 TODO 的架构判断，两次都是错的。**

## 结论二：转换器层**确实**该融合，且收益明确

这是唯一真实的重复点，实测数据：

```
convert_gliner.py     485 行（312 实质行）
convert_boundary.py   752 行（447 实质行）
逐行完全相同的实质行：154  → 占较小者的 49%
```

重复的具体内容：

| 内容 | 两边都有 | 差异 |
|---|---|---|
| `parse_spm` | ✓ | 无 |
| `fast_tokenizer_pieces` | ✓ | 校验文案 |
| `resolve_encoder` | ✓ | boundary 多 3 个 encoder |
| `tensor_contracts` | ✓ | classifier 索引 2 vs 3 |
| 形状契约 + 校验逻辑 | ✓ | tensor 名单不同 |
| tokenizer 声明解析 | ✓ | boundary 兼容 `extra_special_tokens` |
| metadata 写入 | ✓ | boundary 多几十个 `boundary_head.*` |

**但要注意时机**。这次实现过程本身就是证据：融合要等到第 10 个变体才划算。

- 写 #1 时只有一份，没有重复
- 写 #5 时出现第二份，当时 154 行相同——但 #5 本身是 8000 行新架构，
  抽公共层会分散注意力
- 写 #6/#8 时重复开始**制造实际成本**（同一个 vocab bug 要在两个文件里各修一次；
  本次 `tokenizer.json` 词表长度、Unigram 校验、`extra_special_tokens` 风格
  都各踩了一遍）
- 写 #9/#10/#11 时确认：三个模型**零新代码**，全部落在转换器

所以建议是：**把 `parse_spm` / tokenizer 解析 / `resolve_encoder` / 形状契约
抽成 `tools/converter/gliner/common.py`，boundary 的 8000 行逻辑留在原文件。**
预期把 boundary 转换器从 752 行降到 ~400 行。

## 结论三：Ettin（#12）是另一个问题，不属于本次融合

`GLiNER2.5-Decide-1B` 用的是 Ettin-1B（LLaMA-style：28 层 / 1792 / fused QKV /
SwiGLU / RMSNorm / 无位置编码参数），tokenizer 是 ByteLevel BPE 而非
SentencePiece。它需要新 forward 和新 tokenizer。

**不要在这次 PR 里抽公共层。** 理由：

1. 它的 forward 与 DeBERTa 毫无共同结构，抽象不出东西
2. 真正的候选复用点是仓库**已有**的 `llama` / `qwen3` 系（RMSNorm + SwiGLU + RoPE），
   而不是 gliner 内部。这是一次跨模块的架构评估，不是 gliner 家族的收尾
3. 现在做会得出没有证据的结论——`llama` / `qwen3` / `qwen35` 三个模块之间是否
   该融合，需要先看清它们各自为战到什么程度

**建议单独一个 PR**，并在那个 PR 里回答"LLaMA-style forward 在本仓库应该
复用哪一份"。

## 顺带记录：三个 tokenizer 声明风格

实现过程中发现的家族内部不一致，都已兼容：

| 风格 | 出现于 | 声明方式 |
|---|---|---|
| A | #1–#4 | `tokenizer_config.added_tokens_decoder`（含 id） |
| B | #5–#11 | `extra_special_tokens`（仅名字）+ `tokenizer.json`（含 id） |

以及一个**外部不兼容**：`#9`–`#11` 的 `tokenizer_config.json` 让
`AutoTokenizer.from_pretrained` 直接抛异常（`extra_special_tokens` 是裸 list）。
transformers 的 fast 和 slow 类都失败——这说明 reference 只能从基座 encoder
取 tokenizer，oracle 因此新增了 `--base-encoder`（必填）。我们的转换器不受影响，
因为它直接读文件、只需要 id。
