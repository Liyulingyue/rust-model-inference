# Nemotron-3 Nano 30B-A3B (`nemotron_h_moe`) — 契约与适配清单

审阅日期：2026-11-05。契约全部从**真实 GGUF 头部**读出（不是从模型卡转述），
读取方式见文末「怎么复现这份契约」。

## 一句话结论

`nemotron_h_moe`（30B-A3B）**当前仓库完全不支持**：arch 不在白名单、config 前缀
不同、MoE 层分支不存在、并且它需要的 **Q5_1 量化格式在本仓库没有 dequant kernel**。
但基座（Mamba-2 SSM + attention 混合 trunk）已经在 `src/models/nemotron_h/trunk/`
里跑通了，MoE 的 3D 专家张量、路由归一化、共享专家这三块在 `src/models/qwen35/trunk/moe.rs`
里已经有一份**语义几乎同构**的实现 —— 所以这是一次「扩一个 arch + 换一套路由语义」
的增量适配，不是从零做。

## 1. 契约（从 `unsloth/Nemotron-3-Nano-30B-A3B-GGUF` 头部读出）

### 1.1 Metadata

| key | 值 |
|---|---|
| `general.architecture` | `nemotron_h_moe` |
| `nemotron_h_moe.block_count` | 52 |
| `nemotron_h_moe.context_length` | 1 048 576 |
| `nemotron_h_moe.embedding_length` | 2688 |
| `nemotron_h_moe.attention.head_count` | 32 |
| `nemotron_h_moe.attention.head_count_kv` | **逐层数组**：6 个 2，其余 0 |
| `nemotron_h_moe.feed_forward_length` | **逐层数组**：23 个 1856，其余 0 |
| `nemotron_h_moe.attention.key_length` / `value_length` | 128 / 128 |
| `nemotron_h_moe.attention.layer_norm_rms_epsilon` | 1e-5 |
| `nemotron_h_moe.rope.freq_base` | 10 000.0 |
| `nemotron_h_moe.rope.dimension_count` | 84（head_dim 是 128，RoPE 只覆盖 84） |
| `nemotron_h_moe.ssm.conv_kernel` | 4 |
| `nemotron_h_moe.ssm.state_size` | 128 |
| `nemotron_h_moe.ssm.group_count` | 8 |
| `nemotron_h_moe.ssm.inner_size` | 4096 |
| `nemotron_h_moe.ssm.time_step_rank` | 64 |
| `nemotron_h_moe.expert_count` | 128 |
| `nemotron_h_moe.expert_used_count` | 6 |
| `nemotron_h_moe.expert_group_count` / `group_used_count` | 1 / 1（**无分组路由**） |
| `nemotron_h_moe.expert_feed_forward_length` | 1856 |
| `nemotron_h_moe.expert_shared_feed_forward_length` | 3712 |
| `nemotron_h_moe.expert_shared_count` | 1 |
| `nemotron_h_moe.expert_weights_norm` | `true` |
| `nemotron_h_moe.expert_weights_scale` | 2.5 |
| `nemotron_h_moe.vocab_size` | 131 072 |

**没有** `nemotron_h_moe.moe_latent_size` → latent 分支不存在，专家维 = `n_embd` = 2688。
**没有** `nemotron_h_moe.nextn.*` → 不带 MTP draft head。

### 1.2 层模式

逐层数组给出的模式（与模型卡「23 Mamba-2 + 23 MoE + 6 attention」一致）：

- **MoE 层 23 个**（0-based）：1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 40, 43, 45, 47, 49, 51
- **attention 层 6 个**：5, 12, 19, 26, 33, 42
- **SSM（Mamba-2）层 23 个**：其余

判别规则与 llama.cpp 一致：**`head_count_kv != 0` → attention 层；`feed_forward_length == 0`
且 `head_count_kv == 0` → SSM 层；`feed_forward_length != 0` → MoE 层**。
仓库现有的 `nemotron_h` config 已经用同一套规则分三类层（`attention_layers` /
`ffn_layers` / 其余为 SSM），**可以直接复用**。

### 1.3 张量清单（401 个）

| 张量 | 形状 | 类型 | 出现次数 |
|---|---|---|---|
| `blk.N.attn_norm.weight` | 2688 | F32 | 52（**每一层都有**） |
| `blk.N.attn_q.weight` | 2688×4096 | Q5_0 | 6（attention 层） |
| `blk.N.attn_k.weight` | 2688×256 | Q5_0 | 6 |
| `blk.N.attn_v.weight` | 2688×256 | Q5_0 | 6 |
| `blk.N.attn_output.weight` | 4096×2688 | Q4_K | 6 |
| `blk.N.ssm_in.weight` | 2688×10304 | Q5_0 | 23（SSM 层） |
| `blk.N.ssm_conv1d.weight` / `.bias` | 4×6144 / 6144 | F32 | 23 |
| `blk.N.ssm_dt.bias` | 64 | F32 | 23 |
| `blk.N.ssm_a` / `ssm_d` | 1×64 | F32 | 23 |
| `blk.N.ssm_norm.weight` | 512×8 | F32 | 23 |
| `blk.N.ssm_out.weight` | 4096×2688 | Q4_K | 23 |
| `blk.N.ffn_gate_inp.weight` | 2688×128 | F32 | 23（MoE 路由器） |
| `blk.N.exp_probs_b.bias` | 128 | F32 | 23（**路由器偏置，注意名字不是 `ffn_gate_inp.bias`**） |
| `blk.N.ffn_up_exps.weight` | **2688×1856×128** | Q5_0 | 23（3D，expert 轴在最后） |
| `blk.N.ffn_down_exps.weight` | **1856×2688×128** | Q8_0 | 23（3D） |
| `blk.N.ffn_up_shexp.weight` | 2688×3712 | **Q5_1** | 23（共享专家 up） |
| `blk.N.ffn_down_shexp.weight` | 3712×2688 | Q8_0 | 23（共享专家 down） |
| `token_embd.weight` | 2688×131072 | Q5_0 | 1 |
| `output_norm.weight` | 2688 | F32 | 1 |
| `output.weight` | 2688×131072 | Q8_0 | 1 |

MoE 层**没有** `ffn_norm.weight` —— 复用该层的 `attn_norm`（与 4B 的处理一致）。
**没有** `ffn_gate_exps`（专家里没有 gate）、**没有** `ffn_latent_up/down`。

### 1.4 每层 MoE 的精确语义（llama.cpp `src/models/nemotron-h.cpp` + `llama-graph.cpp`）

```text
inp_latent = cur                       # 本模型没有 latent 投影
logits     = ffn_gate_inp @ cur         # [128]
probs      = sigmoid(logits)            # 路由用 sigmoid，不是 softmax
selection  = probs + exp_probs_b        # 偏置只参与「选哪几个专家」
selected   = topk(selection, 6)
weights    = probs[selected]            # 权重取**未加偏置**的 probs
if expert_weights_norm:                 # = true
    weights = weights / max(sum(weights), 2^-14)
if expert_weights_scale != 0 and != 1:  # = 2.5
    weights = weights * 2.5
up         = mul_mat_id(ffn_up_exps, cur, selected)    # [1856, 6]
act        = relu(up) ** 2             # squared ReLU，专家里没有 gate
down       = mul_mat_id(ffn_down_exps, act, selected)  # [2688, 6]
moe_out    = sum_i down_i * weights_i
shexp      = ffn_down_shexp @ (relu(ffn_up_shexp @ cur) ** 2)
out        = moe_out + shexp            # 共享专家没有 sigmoid 门
```

两个容易写错的点：

1. **权重取未加偏置的 probs**，偏置只用来排序（DeepSeek-V3 风格）。
2. **共享专家没有门控**，也没有乘 `weights`；`moe_out + shexp` 是唯一的加法。

## 1·5. 提示构造：`nemotron_h` / `nemotron_h_moe` 也有 CLI↔HTTP 分叉

这不是 MoE 特有的问题，是这一族模型的共同问题，适配 30B 之前就该修。

Nemotron-3 Nano 的 GGUF 自带一个 ~290 行的 Jinja 模板（4B 与 30B 同族），
它渲染出的是 `<|im_start|>system\n<|im_end|>\n<|im_start|>user\n…<|im_end|>\n
<|im_start|>assistant\n<think>…`。仓库把 `nemotron_h` 映射到
`ChatTemplate::ChatML`（`src/prompt/legacy.rs:119`），而 CLI 与 HTTP 各走一套
builder，于是同一条消息有三种提示（用仓库自己的 tokenizer 量 4B）：

| 路径 | token 数 |
|---|---|
| CLI 内置（`format_chat`） | **15** |
| HTTP 内置（`build_qwen_chat_prompt`） | **19**（多一个空 think 块 `<think>\n</think>`） |
| GGUF 自带模板（thinking=false / true） | **24 / 23**（多一个**空 system 轮**，还有一个前导 `\n`） |

差别不是 cosmetic：CLI 一个 think 标记都没有，HTTP 补一个**空** think 块，
模型自带模板**开着** think 块并且多一轮空 system。30B 的 `nemotron_h_moe` 会用
同一个模板家族，所以适配时必须让两条入口共用同一份渲染，否则
`nemotron_h_moe` 会带着这个分叉上线。

细节、证据与修法见 [DORY_REVIEW.md](DORY_REVIEW.md) 的 F1（同一族模型、同一
个模板、同一个问题）。

**BOS 的差别**：4B 的 GGUF 写 `add_bos_token = true`，所以它**有** BOS；
Dory 的 GGUF 写 `false`，而 llama.cpp 对 pixtral 预分词家族强制 `add_bos = true`
→ Dory 的提示比参考实现少一个 `<s>`（见 DORY_REVIEW.md 的 F2）。

## 2. 阻塞项（按必须做的顺序）

### B1. Q5_1 dequant kernel（硬前置）✅ 已完成

> **状态**：已实现并逐位对齐 ggml，见 §6。

原先 `src/ops/kernel/` 下有 `q5_0/` 但**没有 `q5_1/`**；`QuantizedTensor::from_bytes`
（`src/ops/kernel/quantized_tensor.rs`）的 match 里也**没有 `GGMLType::Q5_1`**。

而 `ffn_up_shexp.weight` 在**所有** unsloth 30B 量化档里都是 Q5_1：

| 量化档 | 张量类型分布 | 缺 Q5_1？ |
|---|---|---|
| `Q4_K_M` | Q8_0 37 / F32 237 / Q5_0 75 / Q4_K 29 / Q5_1 23 | ✗ 缺 |
| `IQ4_XS` | Q8_0 24 / F32 237 / Q5_0 1 / IQ4_NL 87 / IQ4_XS 29 / Q5_1 23 | ✗ 缺 |
| `Q4_0` | Q8_0 24 / F32 237 / Q4_0 115 / Q4_1 2 / Q5_1 23 | ✗ 缺 |

23 个 Q5_1 = 23 个 MoE 层的共享专家 up 投影，**换量化档躲不掉**。

Q5_1 与 Q5_0 的差异很小（ggml `block_q5_1`，24 字节/32 元素）：

```c
// Q5_0: y = d * (x - 16)              （22 字节，没有 m，level 是有符号的）
//   x0 = ((qs[j] & 0x0F) | xh_0) - 16
//   x1 = ((qs[j] >>  4)  | xh_1) - 16
// Q5_1: y = d * x + m                 （24 字节，多一个 fp16 的 m）
//   x0 = (qs[j] & 0x0F) | xh_0        ← 无 - 16，level 是 0..31 的无符号值
//   x1 = (qs[j] >>  4)  | xh_1
//   xh_0 = ((qh >> (j + 0)) << 4) & 0x10
//   xh_1 = ((qh >> (j + 12))     ) & 0x10
```

所以实现路径是：照着 `src/ops/kernel/q5_0/` 复制一份，改成 24 字节块、加上 `m` 的
`+ m` 项，并在 `quantized_tensor.rs` 的 `from_bytes` / `GGMLType` 映射里补 `Q5_1`。

**顺带**：`src/models/nemotron_h/trunk/weights.rs` 的 `is_supported_weight_type()`
白名单里也漏了 `Q5_0` 和 `Q5_1`（kernel 在，但白名单不认）——4B 用不到，30B 会撞上。

### B2. arch 与 config 前缀

- `src/core/loader.rs` 的 arch 白名单加 `nemotron_h_moe`；
- loader 里 `nemotron_h` 那个专用分支要能按 arch 前缀读
  `nemotron_h_moe.*`（现在硬编码 `nemotron_h.` 前缀）；
- `src/models/nemotron_h/trunk/config.rs` 的 `architecture != "nemotron_h"` 放行
  `nemotron_h_moe`，并且 metadata key 的前缀要跟着 arch 变。

> 现状是**响亮地失败**而不是静默出错：arch 检查先拒；即使放行，权重层会因为找不到
> `ffn_up.weight` 而报「Nemotron layer N tensors do not match layer metadata」。这点是对的。

### B3. MoE 层分支

`src/models/nemotron_h/trunk/` 现在只有「dense 2 层 FFN」（`ffn_up`/`ffn_down`）。
需要加第三条分支：MoE 层。

现成可抄的两块：

| 需要的东西 | 已经有的地方 | 差在哪 |
|---|---|---|
| 3D 专家张量加载（expert 轴在最后、连续） | `src/models/qwen35/trunk/weights.rs::load_moe_weights` | 命名一致（`ffn_up_exps` / `ffn_down_exps`），但 qwen35 额外有 `ffn_gate_exps`，Nemotron 没有 |
| 路由 top-k + 权重归一化 + 缩放 | `src/models/qwen35/trunk/moe.rs::route_experts` | qwen35 是 softmax；Nemotron 是 sigmoid + 偏置只参与排序 |

`route_experts` 的**归一化部分可以照抄**（`ggml_sum_rows` 用 F64 累加再降到 F32、
`2^-14` 下限 clamp、scale 放在归一化之后而不是折进除法 —— 这一点 qwen35 的注释已经
写清楚「不是逐位等价，会移动 router argmax」）。

### B4. 共享专家与平方 ReLU

- 专家激活：仓库现有的激活算子里有 SiLU / GeLU / ReLU，**没有 squared ReLU**
  → 要么加一个 `relu_sqr_inplace`，要么在 MoE 模块里内联。
- 共享专家：`ffn_up_shexp` → `relu²` → `ffn_down_shexp`，**没有 gate**。
  qwen35 的共享专家是 `down(silu(gate@x) * (up@x))` 并乘一个 sigmoid 门，
  **语义不同，不能直接复用**，只能复用权重加载那部分。

### B5. 数值对拍的 oracle

上游 llama.cpp **已经支持** `nemotron_h_moe`（PR #18058，2025-12-16 合并，
`src/models/nemotron-h-moe.cpp` 只有 159 行，主图在 `nemotron-h.cpp`）。
所以 oracle 用**上游 llama.cpp master**即可，不需要 Dory 那种专用 fork。

注意：llama.cpp 的 `nemotron_h` / `nemotron_h_moe` 都**不开 RoPE**
（`llama_rope_type` 返回 `LLAMA_ROPE_TYPE_NONE`）—— 这和仓库现有 `nemotron_h`
实现的 RoPE 处理需要核对一次（4B 的 `rope.dimension_count = 78`，30B 是 84，
都不是 head_dim 128，说明参考实现里 RoPE 覆盖的是一个子空间）。

## 3. 估算

| 工作项 | 规模 | 备注 |
|---|---|---|
| Q5_1 kernel + 白名单 | ~200 行 | ✅ 已完成，见 §6 |
| arch / config 前缀 | ~50 行 | loader + config |
| MoE 权重加载（3D + 共享专家） | ~250 行 | 抄 qwen35 的 `load_moe_weights` |
| MoE forward（sigmoid 路由 + relu² + 共享专家） | ~250 行 | 抄 qwen35 的 `moe.rs` 换路由语义 |
| 平方 ReLU 算子 | ~30 行 | |
| 对拍测试 + oracle 脚本 | ~300 行 | |
| **合计** | **~1100 行** | 不含调试 |

## 3·5. Q5_1 kernel 的实现要点（已完成）

`src/ops/kernel/q5_1/`（`mod.rs` / `scalar.rs` / `neon.rs` / `tests_neon.rs`）。

**块布局（24 字节 / 32 元素）**：

```
d    f16   偏移 0
m    f16   偏移 2
qh   u32   偏移 4    第 5 bit：bit j → 元素 j，bit j+16 → 元素 j+16
qs   16B   偏移 8    低 4 bit：qs[j] 低 nibble → 元素 j，高 nibble → 元素 j+16
```

**还原式**：`y = level * d + m`，`level ∈ [0, 31]`（**无符号**，**没有 `- 16`**）。
这是 Q5_1 与 Q5_0 最容易搞混的地方：Q5_0 是 `y = d * (level - 16)`。

**矩阵乘**：

```
sum += (d * scale) * dot + m * (scale * sum(q8))
```

`dot` 用原始 5-bit 层级（0..31），`m` 项需要块内激活和。`m` 项的缩放方式与
ggml 一致：ggml 的 `block_q8_1.s` 存的就是 `d * sum(qs)`（已乘激活 scale），
所以 `forward_prepared` 直接给 `input_sums`，`forward_prequantized` 则由 kernel
内部按 `scale * sum(q8)` 现算。

**对拍**：`tools/oracle/q5_1/q5_1_oracle.c` 是 ggml `dequantize_row_q5_1` 的
逐字转写，F16 转换用 ggml 原版 `ggml_compute_fp16_to_fp32`。
`tests/q5_1_reference.rs` 生成 4096 个随机块，喂给 C oracle 与 Rust 实现，
**逐位比较 F32 位模式**。

> 踩到的坑：最初手写了一份 half→float 的 C 转换，对拍立刻报出 6175/131072
> 个元素差 1~3 ULP。原因是 half subnormal 的指数算错了（ ggml 用 magic-number
> 技巧处理）。**教训：这类 oracle 必须逐字抄 ggml，不能自己重写等价实现。**

**NEON（aarch64，部署目标）**：与 `q4_1::neon` 同结构。第 5 bit 用 ggml 同一张
256 项查表展开（`table_b2b_0` 的 `(b) << 4` 语义），level 落在 0..31 可以直接
当 int8 交给 `UDOT`。`tests_neon.rs` 断言 NEON 与 scalar **逐位相等**。

**有意没做**：Q5_1 没有 Vulkan 路径。`src/compute/linear.rs` 的 GPU 白名单里
没有 `Q5_1`，GPU 模式会明确报「weight format has no Vulkan kernel」，而不是
静默算错。30B 的 GPU 支持留到 B2/B3 决定架构时一并评估。

## 4. 怎么复现这份契约

GGUF 的 magic、KV 块和张量信息表都在**文件头部**，所以不需要下载整个 24 GB 的文件：

```sh
# 只取前 64 MB（词表数组很大，头部比想象的大）
curl -sL -r 0-67108863 -o /tmp/n3_30b_head.bin \
  "https://hf-mirror.com/unsloth/Nemotron-3-Nano-30B-A3B-GGUF/resolve/main/Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf"
python3 tools/oracle/nemotron3_nano/gguf_header.py /tmp/n3_30b_head.bin
```

`gguf_header.py` 是一个纯标准库的 GGUF 头解析器（KV + 张量表），随本目录一起提交。

对照的参考实现：

- `llama.cpp` master：`src/models/nemotron-h.cpp`（主图 + MoE 层）、
  `src/models/nemotron-h-moe.cpp`（159 行，主要是 MTP draft head，本模型用不到）、
  `src/llama-graph.cpp::build_moe_ffn`（路由 / 归一化 / 激活的逐节点语义）。
