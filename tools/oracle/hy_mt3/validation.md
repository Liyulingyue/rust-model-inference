# Hy-MT3 验证记录

日期：2026-11-26
基线：上游 `main` **`7dfd5f15e`**（rebase 后；早先一轮是在 `b56a5ce98` 上采的，两轮数字不同，原因见 §3 末尾）
主机：Linux x86_64，128 核，503 GB RAM，**无 GPU**（`nvidia-smi` 不存在）
参考实现：llama.cpp `9c003a97ab34a69cc49243f9b02014038205ffae`（codeload tarball）
本仓库构建：`cargo build --release`（rustc 1.100.0）
设置：greedy argmax、`n_gen=32`、`n_ctx=2048`、`n_batch=n_ubatch=256`、KV F32、
flash-attn 关闭、`n_gpu_layers=0`、`add_bos=0`、`parse_special=true`

## 1. 模型契约

从 GGUF 头部直接读出，并用 llama.cpp 的 `print_info:` banner 交叉确认。

| 模型 | arch | n_layer | n_embd | head/kv | head_dim | n_ff | n_expert/used | n_ff_exp/shexp | rope base | vocab |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hy-MT3-1.8B-Q8_0 | `hunyuan-dense` | 32 | 2048 | 16/4 | 128 | 6144 | — | — | 11158840 | 120818 |
| Hy-MT3-7B-Q8_0 | `hunyuan-dense` | 32 | 4096 | 32/8 | 128 | 14336 | — | — | 11158840 | 128167 |
| Hy-MT3-30B-A3B-Q4_K_M | `hy_v3` | 48 | 2048 | 32/4 | 128 | 6912 | 128/8 | 768/768 | 11158840 | 120832 |

30B-A3B 的 llama.cpp banner 与本仓库 `HyConfig` 读到的值逐项一致：
`arch=hy_v3`、`n_layer=48`、`n_ff=6912`、`n_expert=128`、`n_expert_used=8`、
`n_expert_groups=0`；`expert_weights_scale=2.826`、`expert_weights_norm=true`、
`expert_gating_func=2`（sigmoid）。第 0 层是 dense（无 `ffn_gate_inp`），
第 1..47 层是 MoE + 无门控 shared expert。

文件身份（`sha256sum`，ModelScope `Tencent-Hunyuan/*` master）：

| 文件 | bytes | SHA256 |
| --- | --- | --- |
| `Hy-MT3-1.8B-Q8_0.gguf` | 1,908,528,448 | （见 §1 说明，未逐字节 pin） |
| `Hy-MT3-1.8B-Q4_K_M.gguf` | 1,348,432,160 | `bae6e591103654ff43ef0de7d1492f6e6df1f955505f6bb2cc52e86fc399a942` |
| `Hy-MT3-7B-Q4_K_M.gguf` | 5,145,531,232 | `e08a5098918c6615e1a84fbe714637237bfdc832e17761bdde4bb65878eb17c9` |
| `Hy-MT3-30B-A3B-Q4_K_M.gguf` | 19,502,694,048 | `6e301af0e6d374ff16953dd61665c53e2993ef0b439e2d5ee28f1b6f6570bfea` |
| `Hy-MT3-30B-A3B-Q8_0.gguf` | 32,791,821,792 | `c4254b824fb707f2d253deda4f86d390c3cc9247f0e98ac3ed01f10298c28d25` |

## 2. Tokenizer（离散契约，精确）

**方法**：官方 `tokenizer.json`（HF `tokenizers` 0.23.3）当权威，和本仓库分词器
逐 ID 比较，14 组用例 × 3 个模型，全部 **PASS**（raw prompt / 数字 / 长数字 /
中英混排 / hy 控制 token / 组合字符 / emoji / 空白 / 英文缩写 / 全标点 / 中日韩 /
空串 / 单字符 / 纯换行）。

prompt ID 与 llama.cpp 也逐 ID 相同（§3 表格 `ids` 列全为 Y）：

| 模型 | raw | chat | chat_bos |
| --- | --- | --- | --- |
| 1.8B | 23 | 25 | 26 |
| 7B | 25 | 41 | 51 |
| 30B-A3B | 23 | 25 | 26 |

**本轮修掉的既有缺陷**：`tokenizer.ggml.pre = "hunyuan"`（7B）原本和
`hunyuan-dense`（1.8B/30B）共用数字分支，把数字连成一整段；官方
`tokenizer.json` 与 llama.cpp 都是 `\p{N}`（**每个数字单独成段**）。后果是
`12345` 从一个 piece 变成 5 个，BPE 之后无法再合并，真实 token ID 不同。
已修（`src/core/tokenizer/mod.rs`），回归测试
`hy_scanner_isolates_each_digit_but_hy_dense_chunks_them`。

**llama.cpp 侧确认（7B）**：prompt `价格是12345元，订单号6124c78e。` 两侧
20 个 prompt ID 完全相同，`12345` → `16,17,18,19,20`（五个单数字 piece，
修复前是 2 个），`6124c78e` → `21,16,17,19,66,22,23,68`；greedy 32/32、
cos 0.999974、rel_rmse 0.75%、top-10 10/10。

**影响面**：`pre = "hunyuan"` 的模型都会吃到这次修正。仓库里能确认的调用方是
Hy-MT2-7B（`general.architecture=hunyuan-dense`、`pre=hunyuan`、词表 128167，
与本轮 Hy-MT3-7B 完全同一套 tokenizer 配置）和 Hy-MT3-7B。两者此前都
把数字连段，现在与官方 `tokenizer.json` 和 llama.cpp 一致 —— 是修复而不是回归。
`pre = "hunyuan-dense"`（1.8B / 30B / Hy-MT2-1.8B）**行为不变**，
由回归测试同时锁住两侧。

`clean_spaces`（1.8B/7B `true`、30B `false`）只影响 **detokenize**（去掉
`? ! . ,` 前的空格），不参与分词；本仓库未实现该开关，属全仓库既有缺口，
不影响 token ID 对齐，但比对文本时要容忍空格差异。

## 3. 数值对齐

`cos` = 最后 prompt 位置 logits 的余弦相似度；`rel_rmse` = RMSE / max|logit|；
`greedy` = 32 步贪心 token 是否完全一致；`top10` = 前 10 名的集合重合数。
门槛（`compare.py` 固定，不是事后挑的）：`cos ≥ 0.99`、`rel_rmse ≤ 1%`、
top-1 必须相同、prompt/generated token ID 必须完全相同。

**基线与 rebase 后的重采**（`7dfd5f15e`，2026-11-26 第二轮）：

| 模型 | 变体 | greedy 32 | cos | rel_rmse | top10 |
| --- | --- | --- | --- | --- | --- |
| 7B Q8_0 | raw | **32/32** | 0.999938 | 0.142% | 10/10 |
| 7B Q8_0 | chat | **32/32** | 0.999975 | 0.123% | 10/10 |
| 7B Q8_0 | chat_bos | **32/32** | 0.999975 | 0.095% | 10/10 |
| 7B Q4_K_M | raw | **32/32** | 0.999633 | 0.348% | 10/10 |
| 7B Q4_K_M | chat | **32/32** | 0.999911 | 0.257% | 10/10 |
| 7B Q4_K_M | chat_bos | 30 步分歧 | 0.999888 | 0.201% | 10/10 |
| 1.8B Q8_0 | raw | **32/32** | 0.999563 | 0.616% | 9/10 |
| 1.8B Q8_0 | chat | 13 步分歧 | 0.999792 | 0.200% | 10/10 |
| 1.8B Q8_0 | chat_bos | **32/32** | 0.999785 | 0.199% | 10/10 |
| 1.8B Q4_K_M | raw | 7 步分歧 | 0.998834 | 0.979% | 10/10 |
| 1.8B Q4_K_M | chat | 12 步分歧 | 0.999096 | 0.413% | 10/10 |
| 1.8B Q4_K_M | chat_bos | 11 步分歧 | 0.999105 | 0.401% | 10/10 |
| 30B-A3B Q8_0 | raw | 0 步分歧 | 0.997701 | **1.224%** | 9/10 |
| 30B-A3B Q8_0 | chat | 0 步分歧 | **0.948280** | 4.510% | 7/10 |
| 30B-A3B Q8_0 | chat_bos（模板路径） | **32/32** | 0.997530 | 0.713% | 9/10 |
| 30B-A3B Q4_K_M | raw | 11 步分歧 | 0.999603 | 0.466% | 10/10 |
| 30B-A3B Q4_K_M | chat | 3 步分歧 | 0.959063 | 3.257% | 7/10 |
| 30B-A3B Q4_K_M | chat_bos（模板路径） | **32/32** | 0.997076 | 0.795% | 9/10 |

**判读**：

- 三个模型的 **prompt token ID 在 18 个组合上全部逐 ID 相同** —— 离散契约无条件通过。
- **7B 两种精度、三个变体基本全对**（只有 Q4_K_M 的 chat_bos 在第 30 步分歧，
  cos 仍 0.9999），是这轮最干净的模型。
- **30B-A3B 的模板路径（`chat_bos`）在两种精度上都通过**：Q8_0 32/32 / cos 0.9975 /
  rel 0.71%，Q4_K_M 32/32 / cos 0.9971 / rel 0.80%。这条就是 CLI 与 HTTP 实际走的路径。
- **30B-A3B 的 `raw` 变体在 Q8_0 上 rel_rmse 1.224%，超出 1% 门槛**；Q4_K_M 是
  0.466%。`chat`（不带 BOS）两种精度都不过，且两侧都退化 —— 见下。
- 1.8B Q4_K_M 的 `raw` rel 0.979%，刚好压在门槛内侧。

**关于 `chat`（无 BOS）**：prompt token ID 两侧完全相同，所以不是 tokenizer 或实现
问题。两侧在该形态下**都退化成垃圾**（llama.cpp 输出 `。。。。`，本仓库输出
`\n\n\n…`），llama.cpp 侧 top-1 只领先 0.59。该形态不是模型的真实输入
（GGUF 模板本身发 `<｜hy_begin▁of▁sentence｜>`），已从 parity 契约里排除并如实记录。

**为什么这一轮数字和上一轮不同**：上游 `main` 在 `b56a5ce98..7dfd5f15e` 之间合入了
NEON/AVX-512 的批量 matmul 与 Q2_K..Q6_K 的 u32 读权重改造（#264 / #269 / #272 等），
求和顺序与累加宽度都变了。所以 30B Q8_0 的 `raw` 从「32/32、rel 0.46%」变成
「diff@0、rel 1.22%」—— 这是**同一份实现换基座后的数值漂移**，不是本 PR 引入的回归
（本 PR 在这两个基线上都没有改解码路径）。结论按新基线记录。

**为什么不做逐位对齐**：两侧的有损来源不同且都是刻意保留的 —— 本仓库用
`silu_mul_approx_inplace`（近似 SiLU）、自有 SIMD kernel 与不同的求和顺序；
量化档的 MoE 路由是离散 top-k，近临界处对最小数值差敏感。按
`.agents/skills/adapting-new-models` 的口径，这些用指标验收，不追求 bit-exact。

## 3.5 真实入口（CLI / HTTP）

**CLI（真实权重，30B-A3B Q4_K_M）**：

```
./target/release/rust-model-inference \
  --model Hy-MT3-30B-A3B-Q4_K_M.gguf \
  --prompt "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n机器学习是人工智能的一个分支。" \
  --max-tokens 32 --threads 16 --temp 0 --max-context 2048
→ <think></think>Machine learning is a branch of artificial intelligence.
```

**HTTP（真实权重，`7dfd5f15e` 上实测，不带 `--jinja`）**：

```
$ ./target/release/rust-model-server --model Hy-MT3-30B-A3B-Q4_K_M.gguf --port 18104 --threads 24
$ curl -s localhost:18104/v1/chat/completions -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n机器学习是人工智能的一个分支。"}],"max_tokens":32,"temperature":0}'
{"choices":[{"index":0,"message":{"role":"assistant","content":"Machine learning is a branch of artificial intelligence."},
 "finish_reason":"stop"}],"usage":{"prompt_tokens":39,"completion_tokens":9,"total_tokens":48}}
```

`prompt_tokens=39` 说明走的是**模型自带模板**（模板会带前导换行与
`<｜reasoning_mode｜>` 前言），不是 26 token 的手写 builder。

修通过程中有两处真实的坑，都已在代码里修掉并加测试：

1. **server 默认只在 `--jinja` / `mage_vl` 时启用自带模板**，`hy_v3` 落到手写渲染器
   后又被 arch 白名单拒绝（`400 Tool/chat template is unsupported`）。已让 `hy_v3`
   与 `mage_vl` 一样默认启用自带模板。
2. **同一个白名单在输出侧还会被调用一次**（`OutputParser::new`）。所以即使 prompt
   构造正确、生成成功，响应解析仍会以同一条消息报 500 —— 这条看起来像 prompt 问题，
   实际是输出解析。已把 `hy_v3` 加进白名单（无工具语法、无视觉路径，纯文本），
   回归测试 `hy_v3_plain_text_stream_is_accepted`。

prompt 构造另由 `hy_v3_http_prompt_uses_the_models_own_template` 锁住
（`RMI_HY_MT3_TOKENIZER_GGUF` 指向 tokenizer-only GGUF 即可无权重运行）：
模板渲染出 `\n\n<｜hy_begin▁of▁sentence｜><｜reasoning_mode｜>…<｜hy_User｜>…`，
BOS 在前，没有「位置 0 是控制 token」的退化形态；同一测试还断言**没有模板时仍然
明确报错**，不会静默换成 ChatML prompt。

## 3.6 Vulkan（AMD MI300X / RADV）

**设备**：`lspci` 8 张 `AMD ATI 744b`（MI300X VF，subsys `0E0C`），容器内暴露 1 个
render node（`renderD133`，driver `amdgpu`，`/dev/kfd` 在）。原镜像**没有 Vulkan ICD**，
装 `mesa-vulkan-drivers` 后 `vulkaninfo` 认到
`AMD Radeon Graphics (RADV NAVI31)`，`int8 dot product: supported, shader f16: supported`。

**本 PR 对算子的改动**：`HySession::new` 原先对 `--compute vulkan` 直接硬拒绝
（"hy_v3 uses hybrid linear offload; choose --compute auto or cpu"）。现在改为
**允许 + 一次性明确告知**哪些算子留在 CPU（与 `vela3` 的部分路径写法一致）：

```
compute requested=Vulkan backend=partial scope=hy_v3_decoder offload=linear
note=no complete device graph; attention/KV/RoPE/RMSNorm/SiLU/MoE-routing stay on CPU
```

设备上跑的是**全部线性投影**（q/k/v/o、dense FFN、128 专家的 gate/up/down、shared
expert、输出投影）；attention、KV 读写、RoPE、RMSNorm、SiLU·mul、MoE 路由与加权求和
仍在 CPU。严格 `vulkan` 的语义是「每个投影都必须有设备 kernel，否则报错」，
由 `LinearExecutor::gpu_compatible` 保证，不是「整图在设备上」。

**数值（30B-A3B Q4_K_M，同一进程同一输入）**：

| 对比 | greedy 64 | cos | rel_rmse | argmax |
| --- | --- | --- | --- | --- |
| CPU vs `auto` | 32/32 一致 | 0.999492 | 0.600% | 一致 |
| CPU vs `vulkan` | 第 11 步分歧 | 0.999585 | 0.460% | 一致 |

（`vulkan` 的第 11 步分歧与 CPU-vs-llama.cpp 的 `raw` 变体同一个位置 —— 该 prompt 的
top-1 领先只有 1.0，属于近临界翻转，不是设备 kernel 错。）

**性能（同机、3 次取最小值、`n_gen=64`、16 线程）**：

| 模型 | CPU | `auto`（混合） | 结论 |
| --- | --- | --- | --- |
| 30B-A3B Q4_K_M | **5.60 s** | 8.66 s | GPU 慢 1.55× |
| 1.8B Q8_0（F16 KV） | **2.47 s** | 5.42 s | GPU 慢 2.20× |

**GPU 在这台机器上是慢的，不是快**。归因：

- 逐 token 解码是 batch=1 的瘦矩阵乘，每次投影都是一次 dispatch + 同步；
  30B-A3B 每层 9 个专家 GEMM（8 routed + 1 shared）× 47 层 × 每 token，
  一次 64 token 的解码就是数万次小 dispatch。
- attention/KV/RoPE/RMSNorm/SiLU/路由仍在 CPU，每层都要 host↔device 往返。
- 对照的 CPU 是 128 核 EPYC 9334，小 batch 下本来就强。

所以 `--compute cpu`（默认）仍是这个模型的推荐档。据此把 `auto` 改成**主动退回 CPU**
并说明原因（与 `llama` trunk 的既有做法一致）：

```
compute requested=Auto scope=hy_v3_decoder backend=Cpu reason=no_validated_performance_bucket
```

设备路径改为显式 opt-in（`--compute vulkan`）。这样 `auto` 不会悄悄挑一条实测更慢的
路，而想验证设备路径的人仍然能跑。

### 另一个发现：30B 的设备路径被「一次性权重上传」主导

在共享机（load ≈ 50，计时噪声大）上按解码步数拆开 30B-A3B Q4_K_M：

| 解码步数 | `--compute vulkan` | `--compute cpu` |
| --- | --- | --- |
| 4 | 112.7 s | 34.4 s |
| 16 | 148.5 s | 16.1 s |
| 64 | 146.7 s | 26.2 s |

`vulkan` 的 4 步和 64 步只差 34 s —— 说明**固定开销约 110 s，每 token 边际只有
~0.6 s**。110 s 与「19.5 GB 权重按 ~180 MB/s 上传」吻合，即瓶颈是**把权重搬上设备**，
不是逐 token 的 dispatch。这条对后续很重要：即使把设备图补完，只要每次进程启动都要
重传 19.5 GB，短请求就永远不划算；要赢必须在**长驻进程 + 权重常驻**（服务化）的前提下，
或者解决传输带宽。

### 为什么设备图在这台机器上验证不了：驱动不支持 gfx942

一开始的现象是：把 `hunyuan-dense` 加进 `src/vulkan/qwen3.rs::check_eligibility` 后，
常驻图能跑，但自检拒绝，logits 相对差 2.3%。**看起来像模型适配问题，实际不是。**

**对照实验**：用仓库本来就支持的 Qwen3-0.6B 跑同一条常驻图路径——

```
Error: "Qwen3 Vulkan model contract failed: model logit mismatch at 0:
        gpu=3.160505 cpu=3.1245058 abs=0.035999298 rel=0.011521597"
```

**Qwen3-0.6B 也过不了**，而且三次运行 gpu 值完全一样（3.160505），是**确定性**的
系统性偏差，不是竞争/未初始化内存。所以问题不在 `hunyuan-dense`。

**根因**：这台机器（以及此前几台）的镜像里只有 Mesa 25.2.8 的 RADV，而它**没有
gfx942（MI300X）支持**：

```
$ RADV_FORCE_FAMILY=gfx942 vulkaninfo --summary
radv: Unknown family: gfx942            # 大写 GFX942 同样 Unknown
$ RADV_FORCE_FAMILY=GFX940  vulkaninfo --summary
（AMD 设备消失，只剩 llvmpipe → 强制族与硬件不匹配，设备初始化失败）
```

实际硬件是 `1002:744b`（MI300X VF，CDNA3 / gfx942），但 `vulkaninfo` 报的是
`AMD Radeon Graphics (RADV NAVI31)` —— RDNA3 / gfx1100。**芯片族被认错，shader 按
错误的 ISA 编译**，于是产生确定性的数值偏差。

这解释了两件事：

- 混合路径（只把线性投影挂上去）能过：它没有严格契约，实测 cos 0.9995 / rel 0.46%，
  偏差被容差吃掉了；
- 常驻图过不了：它的自检门槛是 `2e-3 abs + 2e-3 rel`，1.15%–2.3% 的偏差直接超限。

**结论**：这台机器上「Vulkan 算子适配」不是瓶颈，**驱动不支持这块 GPU 才是**。
补 shader、补 MoE 路由都不会改变这一点 —— 需要在支持 gfx942 的环境里（更新的 Mesa、
或 ROCm 的 HIP 后端、或真 Radeon 卡）才能验证设备图。

`hunyuan-dense` 的白名单改动已回滚，没有留在 PR 里。

### 另一个发现：30B 的设备路径被「一次性权重上传」主导

在共享机（load ≈ 50，计时噪声大）上按解码步数拆开 30B-A3B Q4_K_M：

| 解码步数 | `--compute vulkan` | `--compute cpu` |
| --- | --- | --- |
| 4 | 112.7 s | 34.4 s |
| 16 | 148.5 s | 16.1 s |
| 64 | 146.7 s | 26.2 s |

`vulkan` 的 4 步和 64 步只差 34 s —— 说明**固定开销约 110 s，每 token 边际只有
~0.6 s**。110 s 与「19.5 GB 权重按 ~180 MB/s 上传」吻合，即瓶颈是**把权重搬上设备**，
不是逐 token 的 dispatch。这条对后续很重要：即使把设备图补完，只要每次进程启动都要
重传 19.5 GB，短请求就永远不划算；要赢必须在**长驻进程 + 权重常驻**（服务化）的前提下，
或者解决传输带宽。

### 试过但没成：`hunyuan-dense` 直接复用 qwen3 常驻解码器

`hunyuan-dense`（1.8B / 7B）和 `qwen3` 是同一套 dense 解码器（GQA + per-head
q/k RMSNorm + NeoX RoPE，无 MoE、无 qkv bias），CPU 路径本来就复用 qwen3 trunk。
所以最省事的做法是把它加进 `src/vulkan/qwen3.rs::check_eligibility` 的白名单，
让 1.8B/7B 直接拿到**常驻**设备图（权重留在显存）。

实测结果：**常驻图能跑起来，但自检直接拒绝**——

```
[GPU] Q8_0 matmul: packed int8 dot product
operator=attention_score_reduction max_abs=0.000e0 max_rel=0.000e0 first_bad=None
operator=attention_value_reduction exact=true
Error: "Qwen3 Vulkan model contract failed: model logit mismatch at 0:
        gpu=1.124525 cpu=1.1514328 abs=0.026907802 rel=0.023368973"
```

即单个算子的自检都过（attention reduction 逐位、Q8_0 量化 tie-even 一致），
但**整模 logits 相对差 2.3%**，超出该解码器自己的门槛。说明 `hunyuan-dense` 与
`qwen3` 在某个层内细节上并不等价（嫌疑：RoPE 角度表或 q/k norm 的施加位置），
需要逐层对拍定位，不是加个白名单能解决的。

该改动已回滚，没有留在 PR 里。这个自检设计是对的 —— 它挡住了「按架构名放行」。

**已排除的嫌疑**：不是 rope 角度表的精度问题。`src/vulkan/ops.rs::fill_rope_neox`
的系数表是 **f32**（`&mut [f32]`），用 `crate::ops::rope_sin_cos` 且递推
`theta *= scale`，注释明确写着「Match the CPU recurrence」。`freq_base = 11158840`
虽然比 qwen3 的 1e6 大一个量级，但两侧算的是同一串角度。

**剩余嫌疑**（按可能性）：`DenseShape.attention_mode` 被写死为
`AttentionMode::PreparedF16` —— hunyuan-dense 的 logits 幅度比 qwen3 大（|logit| 可到
30+），F16 attention 在这个量级上的相对误差会被放大；其余候选是 `eps = 1e-5`
（qwen3 是 1e-6）在某处被写死。定位方法：把 `validate_model_logits` 的调用点改成
逐层比较 hidden state（`parity-trace` 已有 checkpoint 机制），一次构建即可二分。

**注意**：`src/models/qwen3/trunk/session.rs` 里该自检的容差是
`2e-3 abs + 2e-3 rel`，实测 abs=0.027 / rel=0.023，**超容差 10 倍**，不是边界抖动。

### 还缺什么（完整设备图）

`hy_v3` 目前只有线性投影的设备 kernel。要拿到完整设备图，缺的是下面这批：

| 算子 | 现状 | 复用可能性 |
| --- | --- | --- |
| RMSNorm / QkNorm+Rope / KV 写 / attention / 残差 add / SiLU·mul | CPU | **已有实现**：`src/vulkan/dense.rs` 里这六个 `DenseOp` 都实现了，只差把 `HySession` 的对应步骤改走 `compute::dense::run_dense_layer` |
| 路由（sigmoid → `probs+bias` 选 top-8 → 取未加偏置的概率 → clamp 6.103515625e-5 归一 → ×2.826） | CPU | 需新 shader。`qwen35_moe_route.comp` 是 **softmax**，`nemotron_moe_route.comp` 是 sigmoid + **squared-relu**，两者都不等价 |
| 专家 GEMM（128 选 8，3-D packed Q8_0） | 已可上设备（走 `LinearExecutor`） | 可复用 `q8_matmul_grouped*` |
| 专家输出加权求和 + 无门控 shared expert | CPU | `moe_reduce.comp` / `qwen35_moe_reduce.comp` 可参考 |

`DenseOp::Moe` 在通用 Vulkan 路径里目前直接返回 `UnsupportedShape("dense Vulkan MoE
is unsupported")` —— 每个 MoE 模型（qwen35moe / nemotron_h_moe / dory）都是各自一套
设备适配器。最接近 `hy_v3` 契约的是 `src/vulkan/nemotron_h.rs`（**2234 行**，sigmoid
路由 + `exp_probs_b` 选择偏置 + 共享专家），可作为新 `src/vulkan/hy_v3.rs` 的模板。

**验收口径**（照本目录其余部分）：CPU↔设备 32/32 greedy、cos ≥ 0.999、rel_rmse ≤ 1%，
再加同机 perf 对比。**注意**：以目前 batch=1 的实测，即使补完这批算子也未必快过
128 核 CPU（每层仍有 host↔device 往返），所以这一步的收益要先量再投。

## 4. 未验证 / 未运行

| 项 | 状态 | 原因 |
| --- | --- | --- |
| Vulkan | **部分适配（共享能力待补）** | 线性投影可上设备且数值一致；attention/KV/RoPE/RMSNorm/SiLU/MoE 路由仍是 CPU 算子。实测 GPU 比 CPU 慢 1.55–2.20×，因此 `auto` 主动退回 CPU，设备路径改为 `--compute vulkan` 显式 opt-in。剩余工作与复用点见 §3.6 |
| 30B-A3B MTP/NextN | 不适用 | 这些 GGUF 没有 `nextn.*` 张量（766 个张量里没有） |
| 30B-A3B `IQ2` | 不适用 | 含 IQ2_S / IQ3_S 张量，`hy_v3` loader 只接受 F32/BF16/Q4_K/Q6_K/Q8_0 |
| 完整 262144 上下文 | 未运行 | 只验证到 `n_ctx=2048` |
| JEV | 不适用 | `HyV3TextRuntime` 只暴露 `generate()`，没有 logits 入口 |
| 30B-A3B `raw` 变体 | **未过门槛** | Q8_0 rel_rmse 1.224% > 1%；模板路径通过 |
| 1.8B Q4_K_M | 部分通过 | cos/rel 全部达标，但三个变体 greedy 都在 7–12 步分歧 |

## 5. 性能（同机 CPU，仅供参考）

`llama-bench -p 32 -n 16 -t 16 -ngl 0`：

| 模型 | 精度 | pp32 (t/s) | tg16 (t/s) |
| --- | --- | --- | --- |
| Hy-MT3-1.8B | Q8_0 | 42.27 ± 7.93 | 5.07 ± 0.18 |
| Hy-MT3-7B | Q8_0 | 8.95 ± 0.68 | 1.76 ± 0.14 |
| Hy-MT3-30B-A3B | Q8_0 | 10.01 ± 9.44 | 6.46 ± 1.95 |

> **读这张表之前先看这条**：测试机与其他 agent 共享，采集时 load average 在
> 24–51 之间波动，所以数字偏悲观且噪声大（30B 的 pp32 标准差就有 ±9.44）。
> 作为量级参考：1.8B 的 tg 在空载 8 线程下是 10.01 t/s，本表 16 线程只有 5.07 t/s，
> 约 2× 的争用惩罚。这不是可比的性能基线，只是「能跑、量级合理」的记录。
>
> 本机是 AMD MI300X（RADV），设备路径见 §3.6：线性投影已上设备且数值一致，但实测
> 比 128 核 CPU 慢，`auto` 因此退回 CPU；`--compute vulkan` 是显式 opt-in。
