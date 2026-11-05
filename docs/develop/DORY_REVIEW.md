# Dory 1.0 审阅（2026-11-05）

审阅对象：`src/models/dory/` + `src/vulkan/dory.rs` + `tools/oracle/dory/`，
对照的真实产物是 `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF` 的
`NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf`
（3,673,288,864 bytes，SHA-256 `96080be49ffd67b916018b7bdef8c484762576fb52780e10ab98ac4978a660e0`）。
权重在本次审阅里重新下载并**逐字节校验**过 SHA-256，与 `SUPPORTED_MODELS.md` 的 pin 一致。

审阅主机：**NVIDIA GB10**（集成 GPU，`PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`），
驱动 580.173.02，Vulkan API 1.4.312。

## 结论

Dory 1.0 的实现是**完整且正确的**，文档里列出的两个最大缺口（只有 AMD 验证过、
性能从未 benchmark）本次都补上了，并且补上之后没有发现任何数值或契约偏差。
剩下的发现只有两条是「应该改」，其余是未验收范围。

| 项 | 结论 |
|---|---|
| CPU 契约 | ✓ 结构 / token / embedding 精确检查通过 |
| 前向语义 | ✓ 与参考实现（200 行的 `dory.cpp`）逐项对拍一致，未发现正确性缺陷 |
| Vulkan 设备路径 | ✓ GB10 上 5 个硬件测试全过，GPU↔CPU **逐位一致** |
| 性能 | ✓ 拿到首个基线：设备路径比 CPU 快 **2.7×** |
| CLI / HTTP / JEV | ✓ 三条入口都实测通（见下表） |
| 文档一致性 | ✗ 三处与实测不符（CLI/HTTP 的提示分叉、JEV 的 backend 提示、缺 BOS），已记在发现里 |
| 未验收 | 完整 262 144 上下文；chat template 本身 |

## 一、已验证（带证据）

### 1.1 设备覆盖：从「只有 RADV」变成「两个厂商」

`vulkaninfo`（NVIDIA ICD）实测：

```
deviceName                = NVIDIA GB10
apiVersion                = 1.4.312
shaderFloat64             = true
shaderInt64               = true
maxStorageBufferRange     = 4294967295   (4 GiB)
```

Dory 的设备路径硬性要求 `shaderFloat64`（`src/vulkan/dory.rs:372`），
GB10 满足；Dory 初始化需要 2,996,551,680 bytes ≈ 3.0 GB，落在 4 GiB 的
`maxStorageBufferRange` 内但余量不大 —— **换到 storage-buffer range 更小的设备时
Dory 会直接被拒**，这是既有设计，不是新问题。

```sh
RMI_DORY_Q5_MODEL="$PWD/models/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf" \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  cargo test --locked --profile release-fast --features vulkan \
  --test dory_q5_k_m -- --ignored --nocapture --test-threads=1
```

| 测试 | 结果 |
|---|---|
| `dory_q5_contract_and_tokenizer` | ok（CPU） |
| `dory_q5_single_token_logits_are_finite` | ok（CPU） |
| `dory_q5_prefill_chunking_and_reset` | ok |
| `dory_q5_vulkan_executes_and_matches_cpu` | ok，8 次比较全部 `relative_l2=0`、`scaled_max_abs=0` |
| `dory_batched_modes_and_packed_kv` | ok，`relative_l2=0` |
| `dory_reset_rebuilds_changed_config` | ok，`relative_l2=0` |
| `dory_q5_reference_trace` | ok，`generated_ids=[1897, 1429, 22177, 5325, 1897, 1429, 22177, 5325]` |

5 个被 `#[ignore]` 的硬件测试，179.87 s，零失败。

设备自报 `NVIDIA GB10 (int8 dot product: supported, shader f16: supported)`，
Q8_0 matmul 选了 packed int8 dot product —— **走的是与 RADV 同一套 device path，
不是回退**。

注意门限没被放松：README 的 1% 门限原样保留，**0 是观测值**，不是新门限。

### 1.2 性能基线（此前完全没有）

`Hello` + 16 token，8 线程，同一台机器上跑三轮：

| 路径 | prefill | 生成 | 端到端 |
|---|---|---|---|
| CPU | 2.3–3.2 t/s | 2.8–2.9 t/s | 2.5–3.0 tok/s |
| **Vulkan（GB10）** | **12.0 t/s** | **5.9–6.0 t/s** | **8.1 tok/s** |

→ 设备路径端到端 **2.7×**；prefill **3.9×**，生成只有 **2.0×**。

这个差距是设计使然，不是 bug：README 写明 Vulkan wrapper **故意保留 CPU 契约**
（不 offload 投影，Q5/Q6 用与参考一致的归约顺序，BF16 输入按 BF16 舍入）。
decode 阶段每 token 只有一行激活，权重搬运被摊薄，所以设备侧收益主要体现在
prefill（多行并行）。这也解释了为什么 Dory 的设备路径在数值上是**逐位**一致而不是
「在 1% 门限内」：它压根没改算术。

### 1.3 三条入口

| 入口 | 方式 | 结果 |
|---|---|---|
| CLI 文本 | `--compute cpu` | `Hello! How can I help you today?` |
| CLI 文本 | `--compute vulkan` | 同上；prefill 12.0 t/s |
| CLI JEV | `--jev --jev-context … --jev-question … --compute vulkan` | Blue 0.9309 / Green 0.0166 / Red 0.0505，prefill 5886 ms |
| CLI JEV | 同上 `--compute cpu` | **概率完全相同**，prefill 24646 ms（设备快 4.2×） |
| HTTP | `/v1/chat/completions` | 200，`finish_reason=length`，24 completion tokens |
| HTTP | `/v1/messages` | 200，Anthropic 形状的回复 |
| HTTP | `/v1/messages/count_tokens` | 200，`input_tokens=19` |
| HTTP | `/v1/jev/score` | 200，`Blue` 0.9461，`confidence=0.9461` |

服务端与 JEV 的实测值和 `SUPPORTED_MODELS.md` 里记的 0.9157 不同 —— 那是**不同的
提示格式**（文档里那条走的是 CLI 的 `--jev-context` + 选项路径，这里走的是 HTTP 的
`context` + `questions` 字段），不是回归。

## 二、发现清单

### [必须改] F1：CLI 与 HTTP 给 Dory 的默认提示**不一致**，而且都不等于模型自带模板

**位置**：
- CLI：`src/models/dory/forward.rs:773` → `prompt::legacy::format_chat("dory", …)`；
- HTTP：`src/app/server/api/tools.rs:554` → `prompt::build_qwen_chat_prompt(…)`；
- 两条路都在 `src/prompt/legacy.rs:124` 把 `dory` 映射到 `ChatTemplate::ChatML`。

**事实**（用仓库自己的 tokenizer 把每条路径的 token id 全部打出来）：

| 路径 | 结尾形态 | token 数 |
|---|---|---|
| **CLI** 内置 | `<\|im_start\|>user\n{q}<\|im_end\|>\n<\|im_start\|>assistant\n` | **15** |
| **HTTP** 内置 | `<\|im_start\|>user\n{q}<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n</think>\n` | **19** |
| 模型自带模板（`enable_thinking=false`） | `<\|im_start\|>system\n<\|im_end\|>\n<\|im_start\|>user\n{q}<\|im_end\|>\n<\|im_start\|>assistant\n<think></think>` | 22 |
| 模型自带模板（`enable_thinking=true`） | …`<\|im_start\|>assistant\n<think>\n` | 22 |

**差在哪**：

1. **CLI 与 HTTP 差一个 think 块**。HTTP 走 `build_qwen_chat_prompt` →
   `append_qwen_assistant_prefix(…, enable_thinking=false)`，它会补一个**空 think 块**
   `<think>\n</think>\n`（id 12, 1267, 13, 1267）；CLI 走 `ChatTemplate::ChatML::render`，
   **一个 think 标记都没有**。
2. **模型自带模板还多一个空 system 轮** `<\|im_start\|>system\n<\|im_end\|>\n`
   （模板里 `{%- set system_message = "" %}` 让 `{%- if system_message is defined %}`
   成立，于是**总是**输出这一轮）。
3. **模型自带模板的 think 块没有换行**（`<think></think>`），HTTP 的有
   （`<think>\n</think>\n`）。

**行为差异**（同一权重、同一问题、贪心、8 线程，两次跑都稳定）：

| 入口 | 输出 |
|---|---|
| CLI 内置 | `The sky appears blue during the day due to Rayleigh scattering` |
| HTTP `/v1/chat/completions` | `The sky is most commonly perceived as blue during the day because the Earth's atmosphere scatteres sunlight (` |
| CLI `--chat-template-file`（渲染等价的用户轮） | `The sky is typically blue during the day due to the scattering` |

**为什么是问题**：

- **CLI ↔ HTTP 不一致**，而仓库自己的 `tests/cli_http_agreement.rs` 只覆盖
  llama / qwen3 / qwen35，**没有 dory** —— 这条分叉正好落在覆盖矩阵的盲区里。
- Dory 是 reasoning 模型，`<think>` 是它训练时的「开始思考」信号。CLI 给零个
  think 标记（模型直接开始答），HTTP 给一个**空** think 块，模型自带模板给一个
  **开着的** think 块 —— 三种都不是对方。
- 参考实现（模型自带模板）还会加 **BOS**，见 F2。

**建议修法**：让 `dory` 的两条入口都走模型自带模板（最省事：`default_template`
对 `dory` 改成「渲染 GGUF 模板」，CLI 与 HTTP 共用同一份渲染），并把 dory 加进
`tests/cli_http_agreement.rs` 的覆盖列表，锁住两条入口逐 token 一致。

### [已修] F2：强制 BOS 的语义与 llama.cpp 不一致 —— **Dory 本身是对的，错的是别的 arch**

> **更正（2026-11-05，在 Nemotron-3-Nano 30B 适配时重新核对）**：本节原先的结论
> 「Dory 的提示比参考实现少一个 `<s>`」是**错的**。真正的问题是仓库把
> `add_bos` 的**优先级**写反了。修法见下面的「实际修法」，Dory 的行为**不变**。

**位置**：`src/core/tokenizer/mod.rs:557-624`（pre 分词器映射）与
`mod.rs:705-708`（`add_bos` 的取值）。

代码里的注释写着：

> llama.cpp forces `add_bos = true` for the llama3 pre family
> (llama3 / falcon3 / falcon-h1 / lfm2 / **pixtral** ...) even when the GGUF
> omits `tokenizer.ggml.add_bos_token`.

但实现里 `force_add_bos = true` **只给 `falcon-h1` 一个 arch**（`mod.rs:589-592`），
而且取值写成

```rust
let add_bos = bool_meta(get_meta("tokenizer.ggml.add_bos_token"), ...)? || forced_add_bos;
```

也就是**强制值赢过 GGUF**。

#### llama.cpp 的真实语义

`src/llama-vocab.cpp`（上游 master 与 NVIDIA 的 Dory fork **完全一致**）分两步：

1. pre 分词器分支给一个**默认值**：`pixtral` / `tekken` 落在 `LLAMA_VOCAB_PRE_TYPE_TEKKEN`
   分支里 `add_bos = true`（Dory fork 里 pixtral 落在 `LLAMA_VOCAB_PRE_TYPE_LLAMA3`
   分支，同样 `add_bos = true`）；
2. 之后 GGUF 的 key **覆盖**这个默认值：

```c
// Handle add_bos, add_eos and add_sep
{
    bool temp = true;
    if (ml.get_key(LLM_KV_TOKENIZER_ADD_BOS, temp, false)) {
        add_bos = temp;      // ← GGUF 赢，连 false 都赢
    }
    ...
}
```

所以**优先级是「GGUF 覆盖默认值」**，仓库写成了「默认值覆盖 GGUF」。

#### 结论：Dory 没有问题，缺 BOS 的是别的 arch

| GGUF | `pre` | `add_bos_token` | llama.cpp | 修正前仓库 |
|---|---|---|---|---|
| Dory 1.0 4B | `pixtral` | **有**，`false` | `false`（GGUF 覆盖） | `false` ✅ |
| Nemotron-3-Nano 30B | `pixtral` | **没有这个 key** | `true`（默认值） | `false` ❌ |
| Nemotron-3-Nano 4B | — | `add_bos_token = true` | `true` | `true` ✅ |

Dory 的 GGUF 明写 `add_bos_token = false`，参考实现因此**不加** BOS，仓库也**不加**
——两边一致。真正出问题的是 **key 缺失**的情况：此时该用 pre 分词器的默认值，
而仓库一律当 `false`。Nemotron-3-Nano 30B 正是这一类（`pre = "pixtral"` 且没有
`add_bos_token`），少了 `<s>` 提示就与参考实现不同。

顺带一提，修正前的写法对 **falcon-h1** 也是错的：仓库无条件强制 BOS，
而参考实现里 GGUF 的 `add_bos_token = false` 会把它盖掉。

#### 实际修法

1. `add_bos` 改成「**key 存在则用 key，key 缺失则用 pre 分词器默认值**」：
   ```rust
   let add_bos = match get_meta("tokenizer.ggml.add_bos_token") {
       Some(value) => bool_meta(Some(value.clone()), "tokenizer.ggml.add_bos_token")?,
       None => forced_add_bos,
   };
   ```
2. `forced_add_bos` 补齐 llama.cpp 的整张表（本仓库支持的 pre 里是
   `llama-bpe` / `pixtral` / `tekken` / `lfm2` / `falcon-h1`；
   `exaone` / `exaone4` / `exaone-moe` **不在**表里，参考实现对它们
   保持 `add_bos = false`）。
3. 回归测试 `core::tokenizer::tests`：
   - `pixtral_defaults_to_adding_bos`（30B 的情况：key 缺失 → 加 BOS）
   - `gguf_add_bos_token_overrides_the_pretokenizer_default`（Dory 的情况：
     key 说 false → 不加 BOS）
   - `gguf_add_bos_token_true_adds_bos_without_a_default`

**为什么没被 oracle 抓到**：Dory 的 oracle 对拍是**喂 token id** 的
（`tests/dory_q5_k_m.rs` 自己 encode 出 `[22177]` 再喂进去），oracle 验的是
**前向**，不是**提示构造**。BOS 属于提示构造，所以整条对拍链路都看不见它。
30B 这边同样的道理：oracle 验前向，提示构造要单独对 token id。

### [建议改] F3：`--compute vulkan --jev` 对 Dory 打出的「没有完整设备路径」提示是错的

**位置**：`src/app/cli/types.rs:240`（`vulkan_partial_mode()`）与
`src/app/server/mod.rs:1911`。

**现象**：`--compute vulkan --jev` 打印

```
compute requested=Vulkan backend=partial mode=jev note=no complete device path for this mode; only the offloadable parts run on the device
```

**但 Dory 的 JEV 前向确实整条在设备上**：

| 路径 | prefill | 概率 |
|---|---|---|
| Vulkan | 5886 ms | A 0.9309 / B 0.0166 / C 0.0505 |
| CPU | 24646 ms | A 0.9309 / B 0.0166 / C 0.0505 |

4.2× 的差距和完全一致的概率说明整条前向走了设备（`run_forward_logits_dory` 用
`ComputePolicy::legacy()`，而 `legacy()` 读的是 `--compute` 设进去的 legacy scope）。

**为什么是问题**：`vulkan_partial_mode()` 只看**模式**（jev / rerank / tts …），
不看 **arch**；而 `ComputePolicy::has_complete_device_path()` 里 **`dory` 是在列的**
（`src/compute/mod.rs:91`）。两个判断各管一个轴，于是对 Dory 打出了一句事实相反的
提示。服务端那条注释（"the projections move to the device, the rest stays on the
host"）描述的是 qwen3 那种部分 offload 的模型，对 Dory 也不成立。

**建议修法**：打印前先问 `has_complete_device_path(arch)`；Dory 这类有整图设备路径的
arch 不该打这句提示，或者把提示改成按 arch 措辞。

### [观察] F4：无效请求也会付设备初始化的时间

`--jev` 少给 `--jev-question` 时，错误是在模型加载**之前**抛的（实测总耗时
0.408 s，模型没被读进来），但 Vulkan 设备已经初始化并预热过了 —— 因为
`main()` 里 `configure_cli()`（`src/main.rs:112`）排在 JEV 的参数校验之前。

这不是 bug（设备初始化的 279 ms 本来就是为了和模型加载重叠），只是记一笔：
**「空跑一次 0.4 s」** 是当前 `--compute vulkan` 的固定地板成本。

### [已填补] F5：只在 AMD RADV NAVI31 验证过 → 现在有第二个厂商

见 §1.1。

### [已填补] F6：性能从未 benchmark → 现在有基线

见 §1.2。

## 三、代码对拍：前向语义与参考实现逐项一致

参考实现（NVIDIA 的 Dory llama.cpp fork，`src/models/dory.cpp`，**只有 200 行**）
与 `src/models/dory/forward.rs` 逐项对照，**没有发现语义偏差**：

| 参考实现 | 本仓库 | |
|---|---|---|
| `build_gdn_l2_norm(x, 1e-6f)` —— **L2 归一化，不是 RMSNorm** | `DoryModel::unit_norm`（按 GGML 的 `RMSNorm(eps/n)·1/√n` 写法展开） | ✓ |
| `theta = profile[p] == 2 ? theta_2 : theta_1` | `attention_is_sliding(layer)` 选 `rope_freq_base_2 / rope_freq_base` | ✓（profile 2 ⟺ `'S'` ⟺ theta_2，仓库还在 config 层**校验**了这个不变式，比参考更严） |
| `sq = sqr(scale(sqk, √hd / init_std))`，逐 head 广播 | `q[d] *= (qk_scale[d]·q_factor)²` | ✓ |
| `gate = sigmoid(gate_proj · sqr(scale(gate, √gate_init / init_std)))` | `1/(1+exp(-x))`，scale 先平方 | ✓ |
| attention scale `1/√hd` | `scale = 1/√head_dim` | ✓ |
| `cur = norm(cur · gate)` → `wo` | `unit_norm(attn_out·gate)` → `wo` | ✓ |
| MLP：`g·(suv_gate·√n_embd)`、`u·(suv_up·√n_embd)`、`norm(silu(g)·u)`、`ffn_down` | 同 | ✓ |
| `alpha = abs(scale(alpha, alpha_init / init_std))`，`cur = norm(a + alpha·(b−a))` | `slerp_residual` 里 `alpha[i].abs()` | ✓ |
| recurrent 组首 `cur = norm(cur + input_h)`（`input_h` 取自 `il == n_input`） | `prefill_chunk` 每轮开头 `h += input_h` 后 `unit_norm` | ✓ |
| recurrent 组尾 `cur = norm(cur)` | 最后一层 recurrent 后 `unit_norm` | ✓ |
| `logits = output(h) · (logit_scale / init_std)` | `*logit *= scale * (1/init_std)` | ✓ |
| 张量名 `output_scale` / `blk.%d.residual_alpha` / `attn_qk_scale` / `attn_gate_scale` / `ffn_gate_scale` / `ffn_up_scale` / `attn_gate` | 同名 | ✓ |

结构上也是一一对应的：参考实现里 `'S'` 是**滑窗注意力**层、`'*'` 是全注意力层、
`'-'` 是 MLP 层，仓库用同一套 `layer_pattern` 规则分类；`per_loop` / `last_loop`
只影响 recurrent 层的 KV 落点（参考实现 `cache_layer.push_back(… last_loop ? p : il)`）。

**没找到前向路径的正确性缺陷。** 剩下的风险在别处：提示构造（F1、F2）与设备路径的
覆盖（F3、§一）。

## 四、仍未验收的范围

| 范围 | 状态 | 说明 |
|---|---|---|
| 完整 262 144 上下文 | 未验收 | GGUF 声明 `dory.context_length = 262144`。要验收需要一条 262k token 的提示；设备侧还要看 KV 区域能不能放进 `maxStorageBufferRange`（GB10 上是 4 GiB，Dory 初始化就要 3.0 GB）。**这是当前最大的未覆盖面。** |
| GGUF 自带 chat template 本身 | 未验收 | F1 说明了默认路径与它不同；模板本身（含 tools / `<think>` / `medium_effort` 分支）在本仓库只被 `--jinja` 走过，没有对拍。 |
| Q6_K | 仅为计算路径 | Q6_K 文件是由 Q5_K 重新量化得到的测试文件，不代表官方原生 Q6 的质量。 |
| 其它 GPU / 驱动 | 部分 | 现在有 AMD RADV NAVI31 与 NVIDIA GB10 两家；软件 ICD（lavapipe 等）会被 `is_software_icd()` 拒绝。 |

## 五、复现

```sh
# 权重（ModelScope 上没有这个 repo，走 HF 镜像）
HF_ENDPOINT=https://hf-mirror.com models/.venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF",
                "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf",
                local_dir="models/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF")
PY
sha256sum models/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF/*.gguf
# 期望 96080be49ffd67b916018b7bdef8c484762576fb52780e10ab98ac4978a660e0

# 硬件测试
RMI_DORY_Q5_MODEL="$PWD/models/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf" \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  cargo test --locked --profile release-fast --features vulkan \
  --test dory_q5_k_m -- --ignored --nocapture --test-threads=1

# 性能基线
M="$PWD/models/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  ./target/release-fast/rust-model-inference --model "$M" --prompt "Hello" \
  --max-tokens 16 --compute vulkan --threads 8
```

更多证据见 [oracle README](../../tools/oracle/dory/README.md) 的
「Second GPU device: NVIDIA GB10 (2026-11-05)」一节。
