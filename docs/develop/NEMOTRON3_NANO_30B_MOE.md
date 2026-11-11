# Nemotron-3 Nano 30B-A3B (`nemotron_h_moe`) — 契约与适配清单

审阅日期：2026-11-05。契约全部从**真实 GGUF 头部**读出（不是从模型卡转述），
读取方式见文末「怎么复现这份契约」。

## 一句话结论

`nemotron_h_moe`（30B-A3B）**已适配**：CPU 路径与上游 llama.cpp 逐位一致，
CLI / HTTP / JEV 三个入口都可用，**Vulkan 设备路径也已落地**（2026-11-10，
阶段 3）。设备路径的 52 层（23 SSM / 23 MoE / 6 attention）全部在设备上，
路由留在设备上做（不回读 host），MoE 层不与 host 同步。
代价是**数值不再逐位一致**：设备 kernel 的求和顺序与 host 的 NEON kernel
不同，Mamba-2 递归把这一层差异放大，实测 logits 相对 L2 1.9%–5.8%
（见 §B7），因此设备路径按仓库惯例用**误差门限**验收，不声称逐位一致。

**状态口径（2026-11-11）**：按 `SUPPORTED_MODELS.md` 的四档定义，CPU 路径是
`Verified`（Q4_K_M 与 Q8_0 两档都对 llama.cpp 逐位），Vulkan 设备路径在
`Q8_0` 档上**逐位**（`relative_l2 = 0`），在 `Q4_K_M` 档上是 8% 误差门限。
22 个量化档里 20 个未实测。分层交付时用过的「部分适配」不是仓库的状态词，
已不再使用。

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
| `nemotron_h_moe.expert_group_count` / `expert_group_used_count` | 1 / 1（**无分组路由**） |
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

## 1·5. 提示构造：`nemotron_h` / `nemotron_h_moe` 也有 CLI↔HTTP 分叉 ✅ 已修（2026-11-09）

> **状态**：CLI↔HTTP 的分叉已修。两条入口现在都走
> `prompt::build_nemotron_chat_prompt`（渲染成 ChatML 文本，再用
> `add_special: true` 编码一次），于是 BOS 与 think 块都和 CLI 一致；
> `tests/cli_http_agreement.rs` 锁住 4B 与 30B 的逐 token 一致。
> 下面保留实现前的记录。

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

### 1.6 提示的 BOS：30B 与 Dory 不同

30B 的 GGUF：`tokenizer.ggml.pre = "pixtral"`，**没有** `tokenizer.ggml.add_bos_token` 这个 key。

llama.cpp 的规则是两步（`src/llama-vocab.cpp`）：

1. pre 分词器给出**默认值** —— `pixtral` / `tekken` → `add_bos = true`；
2. GGUF 的 `tokenizer.ggml.add_bos_token` **覆盖**默认值，连 `false` 都赢。

所以：

| GGUF | `pre` | `add_bos_token` | llama.cpp | 结论 |
|---|---|---|---|---|
| Dory 1.0 4B | `pixtral` | 有，`false` | 不加 BOS | 仓库（修正后）也不加 ✅ |
| **30B-A3B** | `pixtral` | **无** | **加 BOS** | 仓库（修正后）加 ✅ |

仓库原先写成 `bool_meta(add_bos_token)? || forced_add_bos` —— **强制值赢过 GGUF**，
而且只有 falcon-h1 设过强制值。结果是 30B 少了 `<s>`。已修（见
[DORY_REVIEW.md](DORY_REVIEW.md) F2 与 [TODO.md](TODO.md) TODO-054 F2）。

## 2. 阻塞项（按必须做的顺序）

### B1. Q5_1 dequant kernel（硬前置）✅ 已完成

> **状态**：已实现并逐位对齐 ggml，见 §6。

原先 `src/ops/kernel/` 下有 `q5_0/` 但**没有 `q5_1/`**；`QuantizedTensor::from_bytes`
（`src/ops/kernel/quantized_tensor.rs`）的 match 里也**没有 `GGMLType::Q5_1`**。

而 `ffn_up_shexp.weight` 在**大多数** unsloth 30B 量化档里都是 Q5_1（2026-11-10
重新核对：22 个档里 11 个含 Q5_1，其余是 Q5_0 或 Q8_0），23 个 Q5_1 = 23 个
MoE 层的共享专家 up 投影，**换量化档躲不掉**。

**22 个量化档的完整对照**（2026-11-10 从每个档的 GGUF 头部读出，
`unsloth/Nemotron-3-Nano-30B-A3B-GGUF` 的 22 个文件；F32 的 237 个张量
每个档都有，下表省略）：

| 量化档 | 张量类型 | CPU 路径 | Vulkan 设备路径 |
|---|---|---|---|
| `Q4_K_M` | Q4_K×29 / Q5_0×75 / Q5_1×23 / Q8_0×37 | ✅ **已验证逐位** | ✅ **已验证**（8% 门限） |
| `Q8_0` | Q8_0×164 | ✅ **已验证逐位** | ✅ **已验证逐位**（`relative_l2 = 0`） |
| `Q6_K` | Q6_K×29 / Q8_0×135 | ✅ | ✅ 类型齐全 |
| `Q5_K_M` | Q5_1×75 / Q5_K×29 / Q8_0×60 | ✅ | ✅ 类型齐全 |
| `Q5_K_S` | Q5_1×88 / Q5_K×29 / Q8_0×47 | ✅ | ✅ 类型齐全 |
| `Q4_K_S` | Q4_K×29 / Q5_0×82 / Q5_1×29 / Q8_0×24 | ✅ | ✅ 类型齐全 |
| `Q4_0` | Q4_0×115 / Q4_1×2 / Q5_1×23 / Q8_0×24 | ✅ | ✅ 类型齐全 |
| `Q4_1` | Q4_1×117 / Q5_1×23 / Q8_0×24 | ✅ | ✅ 类型齐全 |
| `UD-Q4_K_XL` | Q4_K×18 / Q5_0×86 / Q5_1×25 / Q5_K×1 / Q6_K×10 / Q8_0×24 | ✅ | ✅ 类型齐全 |
| `UD-Q5_K_XL` | Q5_0×3 / Q5_1×62 / Q5_K×29 / Q8_0×70 | ✅ | ✅ 类型齐全 |
| `UD-Q6_K_XL` | Q6_K×29 / Q8_0×135 | ✅ | ✅ 类型齐全 |
| `UD-Q8_K_XL` | F16×35 / Q8_0×129 | ✅ | ✅ 类型齐全 |
| `IQ4_NL` | IQ4_NL×116 / Q5_0×1 / Q5_1×23 / Q8_0×24 | ✅ | ❌ 缺 IQ4_NL |
| `IQ4_XS` | IQ4_NL×87 / IQ4_XS×29 / Q5_0×1 / Q5_1×23 / Q8_0×24 | ✅ | ❌ 缺 IQ4_NL / IQ4_XS |
| `Q2_K_L` | IQ4_NL×87 / Q2_K×23 / Q3_K×6 / Q5_0×24 / Q8_0×24 | ✅ | ❌ 缺 IQ4_NL / Q2_K / Q3_K |
| `Q3_K_M` | IQ4_NL×58 / Q3_K×23 / Q4_K×6 / Q5_0×50 / Q5_1×3 / Q8_0×24 | ✅ | ❌ 缺 IQ4_NL / Q3_K |
| `Q3_K_S` | IQ4_NL×87 / Q3_K×29 / Q5_0×24 / Q8_0×24 | ✅ | ❌ 缺 IQ4_NL / Q3_K |
| `UD-Q2_K_XL` | IQ4_NL×47 / Q2_K×15 / Q4_K×4 / Q5_0×64 / Q6_K×10 / Q8_0×24 | ✅ | ❌ 缺 IQ4_NL / Q2_K |
| `UD-Q3_K_XL` | IQ4_NL×47 / Q3_K×15 / Q4_K×3 / Q5_0×60 / Q5_1×4 / Q5_K×1 / Q6_K×10 / Q8_0×24 | ✅ | ❌ 缺 IQ4_NL / Q3_K |
| `UD-IQ2_M` | IQ2_S×15 / IQ4_NL×109 / IQ4_XS×4 / Q5_0×2 / Q6_K×10 / Q8_0×24 | ✅ | ❌ 缺 IQ2_S / IQ4_NL / IQ4_XS |
| `UD-IQ2_XXS` | IQ2_XXS×15 / IQ4_NL×109 / IQ4_XS×4 / Q5_0×2 / Q6_K×10 / Q8_0×24 | ✅ | ❌ 缺 IQ2_XXS / IQ4_NL / IQ4_XS |
| `UD-IQ3_XXS` | IQ3_XXS×15 / IQ4_NL×101 / IQ4_XS×3 / Q5_0×6 / Q5_1×4 / Q5_K×1 / Q6_K×10 / Q8_0×24 | ✅ | ❌ 缺 IQ3_XXS / IQ4_NL / IQ4_XS |

读法：

- **CPU 路径全部可用**：`is_supported_weight_type()` 白名单与
  `QuantizedTensor::from_bytes` 的 match 覆盖了上表出现的每一种类型，
  i-quant（IQ4_NL / IQ4_XS / IQ2_XXS / IQ2_S / IQ3_XXS）与 Q2_K / Q3_K
  的 kernel 都在 `src/ops/kernel/` 里。
- **设备路径取决于「该档的每一种张量类型是否都有设备 kernel」**：
  i-quant 与 Q2_K / Q3_K **没有**设备 kernel，所以那 10 个档
  （IQ4_NL / IQ4_XS / Q2_K_L / Q3_K_M / Q3_K_S 与 5 个 UD i-quant 档）
  整个 Nemotron session 建不起来 → `--compute auto` 打印
  `Nemotron-H Vulkan unavailable: …` 后**回退 CPU**，
  `--compute vulkan`（严格）则直接报错退出。
  其余 12 个档的类型组合在设备侧**类型齐全**，设备路径可以直接用。
- **`Q4_K_M` 与 `Q8_0` 两档真实下载并验证过**：`Q4_K_M` 的 CPU 路径逐位、
  设备路径 8% 门限；`Q8_0` 档**两条路径都逐位**（见 §B7.4）。
  其余 20 个档的类型分布是**从各档 GGUF 头部读出来的真实契约**，
  但没有下载权重，因此「能加载、能跑」只是类型层面的推断，不是实测。
- **`Q8_0` 档是唯一「设备也逐位」的档**：它的张量只有 Q8_0 / F32（头部的
  F16 计数是 0），而这两类设备 kernel 都是**逐位**的（Q8_0 的
  `max_abs = 0`，F32 的 `exact_bits = true`），路由 shader 用的也是
  ggml-exact 的 `exp_reference` + `divide_finite_normal`。实测
  `relative_l2 = 0`（1 / 2 / 3 token prefill），与 4B 相同。

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

### B2. arch 与 config 前缀 ✅ 已实现

- `src/core/loader.rs` 的 arch 白名单加 `nemotron_h_moe`；
- loader 里 `nemotron_h` 那个专用分支要能按 arch 前缀读
  `nemotron_h_moe.*`（现在硬编码 `nemotron_h.` 前缀）；
- `src/models/nemotron_h/trunk/config.rs` 的 `architecture != "nemotron_h"` 放行
  `nemotron_h_moe`，并且 metadata key 的前缀要跟着 arch 变。

> 现状是**响亮地失败**而不是静默出错：arch 检查先拒；即使放行，权重层会因为找不到
> `ffn_up.weight` 而报「Nemotron layer N tensors do not match layer metadata」。这点是对的。

**实现**：GGUF 把每个 arch 私有的 metadata key 放在 `{arch}.` 命名空间下，所以
`NemotronConfig::from_source` 用 `architecture` 本身当前缀，config 里新增
`NemotronMoeConfig`（`expert_count` / `expert_used_count` / `expert_group_count` /
`expert_group_used_count` / `expert_feed_forward_length` /
`expert_shared_feed_forward_length` / `expert_shared_count` / `expert_weights_norm` /
`expert_weights_scale` / `moe_latent_size`）。

> **key 名的一处订正**：分组路由已用的那个 key 实际是
> `nemotron_h_moe.expert_group_used_count`（不是 `group_used_count`）。契约表 §1.1
> 已按 GGUF 头部订正。

### B3. MoE 层分支 ✅ 已实现

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

**实现**（`src/models/nemotron_h/trunk/moe.rs` + `weights.rs`）：

- `NemotronMoeWeights` 持有一份借用的 `up_exps` / `down_exps` 原始字节 +
  每个专家的字节步长，而不是 128 个 eager `Weight`（23 层 × 128 专家 = 2944 个
  kernel，提前建好只用得上 6 个）。
- `route_experts` 与 llama.cpp 的 `build_moe_ffn` 逐节点对应：
  `probs = sigmoid(logits)`（ggml 的 `ggml_vec_sigmoid_f32` 就是
  `1/(1+expf(-x))`）→ `selection = probs + exp_probs_b`（**偏置只参与排序**）→
  `argsort_top_k` 取前 `n_expert_used` → 权重取**未加偏置**的 `probs` →
  `sum` 用 F64 累加降到 F32 → clamp 到 `2^-14` → 相除 → 乘
  `expert_weights_scale`。
- 专家**没有 gate**（本架构没有 `ffn_gate_exps`），所以专家激活是
  `relu(up)^2`；共享专家**也没有 gate**（没有 `ffn_gate_inp_shexp`），
  最终 `out = moe_out + ffn_shexp`。
- 层类型判定：`ffn_up.weight` 是 dense FFN，`ffn_up_exps.weight` 是 MoE 层，
  两者互斥，且必须与 metadata 的层模式一致，否则报
  「Nemotron layer N tensors do not match layer metadata」。

### B4. 共享专家与平方 ReLU ✅ 已实现

- 专家激活：仓库现有的激活算子里有 SiLU / GeLU / ReLU，**没有 squared ReLU**
  → 要么加一个 `relu_sqr_inplace`，要么在 MoE 模块里内联。
- 共享专家：`ffn_up_shexp` → `relu²` → `ffn_down_shexp`，**没有 gate**。
  qwen35 的共享专家是 `down(silu(gate@x) * (up@x))` 并乘一个 sigmoid 门，
  **语义不同，不能直接复用**，只能复用权重加载那部分。

**实现**：squared ReLU 直接内联在 `moe.rs`（`relu` 后 `x*x`，负数归零），
没有单独的算子 —— 它只在这个架构的 FFN 与 MoE 里出现。

### B5. 数值对拍的 oracle ✅ 已完成（逐位一致）

上游 llama.cpp **已经支持** `nemotron_h_moe`（PR #18058，2025-12-16 合并，
`src/models/nemotron-h-moe.cpp` 只有 159 行，主图在 `nemotron-h.cpp`）。
所以 oracle 用**上游 llama.cpp master**即可，不需要 Dory 那种专用 fork。

**结果**：全 52 层 × 6 步（prefill 2 token + 生成 4 token）**逐位一致**，
最终 logits 逐位一致（`worst logit disagreement 0.0000`，而 argmax 领先幅度是
2.2746），端到端 greedy 输出一致（两侧 `generated_ids = [22177, 22177, 22177, 22177]`，
文本 `HelloHelloHelloHello`）。

注意：llama.cpp 的 `nemotron_h` / `nemotron_h_moe` 都**不开 RoPE**
（`llama_rope_type` 返回 `LLAMA_ROPE_TYPE_NONE`）—— 仓库现有 `nemotron_h`
实现与之一致（4B 的 `rope.dimension_count = 78`，30B 是 84，都不是 head_dim 128，
参考实现里 RoPE 覆盖的是一个子空间）。

#### B5.1 对拍工具

| 文件 | 作用 |
|---|---|
| `tools/oracle/nemotron3_nano/oracle.cpp` | 上游 `llama-eval-callback` harness，dump 命名张量 |
| `tools/oracle/nemotron3_nano/build_oracle.sh` | 解包 pinned llama.cpp tarball 并构建 harness |
| `tools/oracle/nemotron3_nano/check_moe.py` | 把 oracle trace 与 Rust `RMI_PARITY_TRACE` 按 `(name, layer, step)` join，逐位比 |
| `tools/oracle/q5_0/q5_0_oracle.c` | ggml ARM `ggml_vec_dot_q5_0_q8_0` 的独立转写 |
| `tools/oracle/silu/silu_oracle.c` | ggml ARM `ggml_vec_silu_f32` 的独立转写 |

Rust 侧的 checkpoint 由 `src/models/nemotron_h/trunk/trace.rs` 发出，
只在 `--features parity-trace` 且 `RMI_PARITY_TRACE` 有值时生效。

#### B5.2 为什么需要 ggml-exact kernel（而不是「差不多就行」）

第一次对拍时 4B 的 SSM 分支只差 1 ULP，30B 的却差 ~1e-4。根因是**量化权重
kernel 的 F32 累加结构**不同，被下游的 Q4_K 出投影的 Q8_K 激活量化放大了约 7000×：
一个 1-ULP 的差足以让某个激活跨过 8-bit 的舍入边界，于是变成整整一个量化台阶。

所以仓库新增了三个 **ggml-exact** 入口，与默认 kernel 并存（默认契约不动）：

| 入口 | 复刻的 ggml 结构 |
|---|---|
| `ops/kernel/f32/ggml.rs` | `ggml_vec_dot_f32`：`GGML_F32_STEP=16` / `EPR=4` / `ARR=4`，4 个 `vfmaq_f32` 累加器 + pairwise-tree reduce + `vaddvq_f32` |
| `ops/kernel/q8_0/ggml.rs` | `ggml_vec_dot_q8_0_q8_0`：2 个 4-lane 累加器、每 block `vaddq_s32(vdotq_s32(0,x0,y0), vdotq_s32(0,x1,y1))`、`vmlaq_n_f32`、末尾 `vaddvq_f32` 相加 |
| `ops/kernel/q5_0/neon.rs` | 2 block/迭代 + 2 个 4-lane F32 累加器 + `table_b2b_1` 查表折 `-16` |
| `ops/kernel/q5_1/neon.rs` | 同上，另加独立标量 `m*s` 累加器（`summs0`/`summs1`），末尾 `((vaddvq(sumv0)+vaddvq(sumv1))+summs0)+summs1` |

**per-lane 分组是硬约束**。`ggml_vdotq_s32` 在 dotprod 可用时就是 native
`vdotq_s32`（`dst[i] = Σ_{k=0..3} a[4i+k]*b[4i+k]`）；它自己的 fallback 注释写着
「different per-lane grouping — do not use when individual lane values matter」。
因此 oracle 与 Rust 两侧都必须在 `-march=armv8.2-a+dotprod` 下编译，且**不能开 i8mm**
（那会走 `ggml_vec_dot_q8_0_q8_0` 的 MATMUL_INT8 分支，是另一套算术）。

另外两处不显眼但必须对齐的细节：

- **SiLU 用 ggml 的多项式 `exp`**（`ggml_v_expf`），不是 libm `expf`。
  `ggml_v_expf` 里的 `vdupq_n_u32(0x7f000000)` 是 **u32 字面量不是 float**：
  写成 `vdupq_n_f32(0x7f000000)` 会加 `0x4EFE0000`（2139095040.0f 的舍入），
  于是溢出路径的 `s1*s1` 变成巨大有限值而不是 0 —— 而 softmax 掩码列（`-inf`）
  必须贡献 0。这个 bug 曾**同时**存在于 Rust 与 silu oracle 两侧，所以参考测试
  发现不了，只有对拍真实模型才暴露。
- **Q4_K/Q5_K 的 `dmin` 修正必须分两次累加**（`total -= dmin*min_sum; … total += d*main_sum`），
  不能合并成 `total += main + min`。
- **Q5_1 没有 `-16` 偏移**；`block_q8_1.s = f16(sum*d)`，`d = amax/127`（现代 ggml，非负）。
- **softmax 用 `ggml_v_expf`（NEON 4-lane 组）+ f64 累加 + 末尾 libm `expf` 尾块**；
  `softmax_inplace` 用 libm `expf` 且 f64 累加，两者不同。
- **llama.cpp 的 KV cache view 是 padded 的**（`GGML_PAD(used, 256)`），softmax 与
  `V×softmax` 都在 padded key 数上跑。Rust 侧用 `n_kv = (position+1).div_ceil(16)*16`
  复刻（16 对齐就够，因为 dot 的分组边界是 16）。
- oracle harness 用 `RMI_NEMOTRON_KV=f32`，即 KV cache 是 F32。

### B6. 入口层（CLI / HTTP / JEV）✅ 已完成（2026-11-09）

- `src/models/nemotron_h/trunk/runtime.rs`：`nemotron_h`（4B）与 `nemotron_h_moe`
  （30B）共用一个 `TextRuntime` 适配器，`/v1/chat/completions` 从 501 变成可用。
- 设备路径在 HTTP 侧也接上了（4B）：适配器在启动时建一次设备会话（权重要上传到
  设备，所以不能每请求重建），提示超过 arena 行数的那次请求回退 CPU。
- **提示分叉已修**：CLI 用 `prompt::legacy::format_chat`（ChatML、`add_special:
  true`、无 think 块），HTTP 用 qwen 的 ChatML builder（**无 BOS**、**多一个空
  think 块**）—— 同一条消息 CLI 15 token、HTTP 19 token。两族现在都走
  `prompt::build_nemotron_chat_prompt`，`tests/cli_http_agreement.rs` 锁住两条
  入口逐 token 一致。
- JEV 的单次与分组判定本来就按 arch 分派，两个入口都覆盖，实测概率完全相同
  （单次 A 0.0307 / B 0.9693；分组 A 0.0640 / B 0.9360）。

### B7. Vulkan 设备路径 ✅ 已完成（2026-11-10，阶段 3）

分三层，全部落地：

1. ~~**Q5_0 与 Q5_1 的设备 kernel**~~ ✅ **已完成（2026-11-09）**。30B 的
   `attn_q/k/v`、`ssm_in`、`ffn_up_exps`、`token_embd` 是 **Q5_0**，共享专家的
   `ffn_up_shexp` 是 **Q5_1**，换量化档躲不掉；两个格式现在都有设备 kernel
   （见 §3·5）。
2. ~~**MoE 层的设备分支**~~ ✅ **已完成（2026-11-10）**。
3. **入口与自动分发**：`nemotron_h_moe` 已列入
   `ComputePolicy::has_complete_device_path`，`--compute vulkan` / `Auto`
   走设备路径；HTTP 侧与 CLI 同源（§B6）。

#### B7.1 MoE 设备分支的形状与做法

| 部件 | 做法 | 与 host 语义的关系 |
|---|---|---|
| 路由器 | `ffn_gate_inp`（F32 2688×128）在设备上出 128 个 logit | 与 CPU 的 `ffn_gate_inp` 同矩阵、同 F32 累加 |
| 路由 | 新 shader `nemotron_moe_route`：sigmoid → 加 `exp_probs_b.bias` 排序 → top-6 → 按**无偏**概率归一化 → ×2.5 | 与 `src/models/nemotron_h/trunk/moe.rs::route_experts` 逐条对齐（bias 只参与排序，不进权重；`SUM_FLOOR = 2^-14`） |
| 专家选择 | 选中的 id 与权重**写进 arena**，专家 kernel 自己读 id 并把权重偏移一个专家矩阵 | 路由**不回读 host**：一层不产生同步点 |
| 专家投影 | 每个 slot 一次 dispatch：`ffn_up_exps[e]`（Q5_0 2688×1856）→ `relu(x)^2` → `ffn_down_exps[e]`（Q8_0 1856×2688） | 激活是 `relu(x)^2`，与 llama.cpp 的 `LLM_FFN_RELU_SQR` 一致 |
| 共享专家 | `ffn_up_shexp`（Q5_1 2688×3712）→ `relu(x)^2` → `ffn_down_shexp`（Q8_0） | **没有门控、不乘权重** |
| 归约 | 新 shader `nemotron_moe_reduce`：加权求和 + 共享专家直加 + 残差 | — |

**3D 张量的绑定**：ggml 把专家轴放在最后且连续，所以「专家 e 的矩阵」就是
第 e 段。一个 MoE 层因此只需要**一个** descriptor set（整个 3D 张量整体绑定），
而不是每专家一个 —— 描述符预算仍是 `3 + 6 × n_layer`（norm + 5 个权重）。
专家 id 由 push constant 的第 22–25 个 word 传进 kernel
（`expert_count / expert_ids_word / expert_used / expert_slot`）。

**显存**：专家权重约 24.2 GB（23 层 × 128 专家 × (Q5_0 2688×1856 + Q8_0
1856×2688)），单层最大张量 428 MB（up）/ 620 MB（down），都在 GB10 的
`maxMemoryAllocationSize`（约 4 GiB）之下；整模 24.5 GB 的权重与 host 侧的
mmap 是同一份数据的两份拷贝，统一内存放得下。

#### B7.2 数值契约：为什么是 8% 门限而不是 1%

CPU 路径与 llama.cpp 逐位一致靠的是复刻 ggml 的 F32 累加结构；设备上的
int8 dot + F32 scale 累加与之**不逐位相同**。30B 比 4B 更敏感：

- 单个投影的差异就已经不小：Q5_0 的 `ssm_in`（2688×10304）设备 vs CPU
  `max_rel = 3.791e-3`（Q8_0 的 kernel 则是**逐位一致**，`max_abs = 0`）。
- Mamba-2 的递归把这个差异沿 52 层放大，router 还可能在近似平局上翻转。

实测（GB10，1 / 2 / 3 token 的 prefill，logits 相对 L2 / 缩放最大绝对误差）：

| 配置 | 1 token | 2 token | 3 token |
|---|---|---|---|
| 全部在设备 | 4.28% / 3.35% | 4.23% / 3.40% | 1.79% / 1.39% |
| 只有 SSM 在设备 | 5.51% / 4.57% | 4.14% / 2.57% | 1.93% / 1.57% |
| 只有 MoE 在设备 | 5.45% / 5.50% | 3.72% / 2.45% | 1.92% / 1.49% |
| 只有 attention 在设备 | 4.32% / 3.86% | 1.23% / 0.95% | 1.17% / 0.94% |
| **全部在 host**（只留最终归一化 + 输出投影在设备） | **0% / 0%** | **0% / 0%** | **0% / 0%** |

最后一行说明两件事：设备的**最终 RMS 归一化与 Q8_0 输出投影与 CPU 逐位一致**；
三个分支各自的贡献量级相当（1%–6%），而且**不可加** —— 放大是混沌的，
「全部在设备」并不比「只有 SSM 在设备」更差。整张表的最大值是 **5.51%**。

硬件测试的门限因此取 **8%**（相对 L2 与缩放最大绝对误差各一条），另加一条
更硬的**功能门限**：设备路径必须选出与 CPU 路径相同的 top-1 token
（1 / 2 / 3 token 的 prefill 实测都相同）。测试：
`RMI_NEMOTRON_MOE_MODEL=<gguf> cargo test --locked --profile release-fast
--features vulkan --test nemotron3_nano_30b_device -- --ignored --nocapture`。

> **踩坑一：设备路径里跑在 host 上的层必须锁在 CPU 语义上。**
> 设备 session 在场时，`ops::kernel::q4_k` 的 `forward_prepared` 会把
> **host 上的 Q4_K matmul 也 offload 到设备**（`gpu_matmul_active()` 为真）。
> 30B 的 `ssm_out` / `attn_output` 都是 Q4_K，于是「在 host 上跑的层」与
> 纯 CPU 运行**不等价**：一个投影就差 1e-7，再经 Mamba-2 递归放大，最后
> 一层 hidden state 差 18%。`RMI_NH_CPU_*` 这三个覆盖开关因此必须配一个
> `ComputePolicy::Cpu` scope，否则没法把差异归因到真正搬上设备的那个分支。
> 修好之后，「全部在 host」这一行是 **0**（逐位一致）。

> **踩坑二：路由权重的求和要用 F64。**
> CPU 的 `route_experts` 复刻的是 ggml 的 `ggml_sum_rows`：F64 累加器，
> 末尾舍入成 F32。shader 里若用 F32 累加，权重会差 1 ULP 量级，
> 经递归放大后 logits 的相对 L2 从 **4.28%** 变成 **5.84%**
> （`nemotron_moe_route.comp` 因此开了 `GL_ARB_gpu_shader_fp64`）。

#### B7.3 性能（GB10，8 线程，55 token 提示 + 16 token 生成）

| | prefill | 生成 | 端到端 |
|---|---|---|---|
| CPU | 5.9 t/s | 9.6 t/s | 1.4 tok/s |
| Vulkan 设备路径 | **28.7 t/s** | 9.0 t/s | **4.3 tok/s** |

prefill 快 **4.9×**（权重读一次、多行复用），逐 token 生成与 CPU 持平
（单 token 走的是共用 Q8_0 kernel 的标量 int8 路径，没有用上 DP4A 单元；
这与 4B 的情况一样，是共用 kernel 的性质）。

#### B7.4 Q8_0 档：设备路径逐位一致（2026-11-11 实测）

§B7.2 的 8% 是 **Q4_K_M 档**的性质，不是设备路径的性质：Q4_K_M 的
`ssm_in` / attention q/k/v 是 Q5_0、`ssm_out` / `attn_output` 是 Q4_K，
这两个设备 kernel 的归约顺序与 host 的 NEON kernel 不同。**Q8_0 档没有
这个问题**：该档 401 个张量里 164 个是 Q8_0、237 个是 F32（头部的 F16 计数
是 **0**），两类设备 kernel 都是逐位的（Q8_0 `max_abs = 0`，F32
`exact_bits = true`），路由 shader 用的也是 ggml-exact 的 `exp_reference`
+ `divide_finite_normal`。

实测（GB10，8 线程）：

| 检查 | 结果 |
|---|---|
| 设备 vs CPU（1 / 2 / 3 token prefill） | **`relative_l2 = 0`、`scaled_max_abs = 0`**，top-1 token 相同 |
| CPU vs 上游 llama.cpp（`Hello` + 4 token 生成，2574 个 join 到的 checkpoint） | 全部 `EXACT`；最终 logits `worst disagreement 0.0000`，argmax 1095 相同（margin 1.1368） |

命令：

```sh
GGUF=models/Nemotron-3-Nano-30B-A3B-GGUF/Nemotron-3-Nano-30B-A3B-Q8_0.gguf
# 设备 vs CPU（RMI_NEMOTRON_MOE_EXACT=1 把门限收紧到 0）
RMI_NEMOTRON_MOE_MODEL=$GGUF RMI_NEMOTRON_MOE_EXACT=1 \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  cargo test --locked --profile release-fast --features vulkan \
  --test nemotron3_nano_30b_device -- --ignored --nocapture
# CPU vs llama.cpp
RMI_PARITY_TRACE=/tmp/n3_q8/oracle.jsonl RMI_NEMOTRON_THREADS=1 RMI_NEMOTRON_KV=f32 \
  target/nemotron3-nano-oracle/build/bin/llama-eval-callback \
  $GGUF /tmp/n3_q8/prompt.txt 4
RMI_NEMOTRON3_MODEL=$GGUF RMI_PARITY_TRACE=/tmp/n3_q8/rust.jsonl \
  RMI_NEMOTRON3_PROMPT_FILE=/tmp/n3_q8/prompt.txt RMI_NEMOTRON3_N_PREDICT=4 \
  cargo test --locked --profile release-fast --features parity-trace \
  --test nemotron3_nano_moe -- --ignored --nocapture
python3 tools/oracle/nemotron3_nano/check_moe.py \
  /tmp/n3_q8/oracle.jsonl /tmp/n3_q8/rust.jsonl
```

文件 SHA-256 `433a9c0d39c6c3434204797b30991c4d147c0f25328690d60862ec39e2a92188`，
33,585,495,840 bytes（ModelScope 列表值与实际值一致，下载后校验过）。

**结论**：30B 的「完整设备路径」不是做不到，而是**取决于量化档** ——
Q8_0 档已经与 4B 同级（逐位），Q4_K_M 档受 Q5_0 / Q4_K 设备 kernel 的归约
顺序限制。要把 Q4_K_M 也做到逐位，需要让那三个 kernel 复刻 host NEON 的
`vdotq_s32` 4-lane 分组与 F32 累加顺序（取舍见 §B7.2：逐位要求每行串行
累加，会牺牲 prefill 的「权重读一次、多行复用」）。

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

**设备路径（2026-11-09 补上）**：Q5_1 起初没有 Vulkan 路径（GPU 模式会明确报
「weight format has no Vulkan kernel」，而不是静默算错）。现在有
`shaders/glsl/q5_1_matmul.comp` 与 `shaders/glsl/q5_0_matmul.comp` 两个设备
kernel：Q4_1 的设备 kernel 带 `m` 项，正好是 Q5_1 的模板；Q4_0 是 Q5_0 的模板。
两者与 CPU kernel 的数值在门限内一致（Q5_0 max_rel 6.5e-6，Q5_1 max_rel 3.1e-5），
且比 CPU 快 2-3×。

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
