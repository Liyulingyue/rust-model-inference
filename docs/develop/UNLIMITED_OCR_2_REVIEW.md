# Unlimited-OCR-2 审阅报告

审阅对象：`gouzil/codex/unlimited-ocr-2`（`76a54957`），2026-10-30 在 Linux x86_64
（Intel Meteor Lake 集显）上独立复验。

---

## 1. 结论

分支的实现是**扎实的**：官方权重与源码的 SHA 全部对上，官方 PyTorch 模块 unmodified
就能作为 oracle，本机 3040 个单页检查点全部通过（最大 peak-relative 0.00009388，
门槛 0.002），CLI / HTTP / SSE / Anthropic Messages / Responses 的 11 项 HTTP 检查
全过，六种 GGUF 格式都能转换并加载运行。

**两个需要写下来的事实**：

1. **whole-graph 的 MoE 路由边界不稳定，而且不是作者机器的环境问题。** 作者在 macOS
   上保留了 5 处 prefill attention 例外；本机换渲染字体、换机器后，双页用例在
   `uocr.attention` decoder layer 2 越界（0.010240 > 0.002），**同一签名、同一量级**。
   单页用例反而全部通过。结论：**「按 fixture 对齐」不能当成「按架构对齐」**。
2. **Auto 路径在集显上比 CPU 慢 2.03 倍**，而且**单行 decode 的投影根本没上设备**
   （门槛是 16384 个输出元素）。这不是 bug，但「GPU」这个字眼会误导用户。

---

## 2. 权重与来源

| 项 | 值 |
|---|---|
| 来源 | `PaddlePaddle/Unlimited-OCR-2`（ModelScope），revision `7eadb7cd7afe660602040524eab37c8a9b91fe53` |
| 下载方式 | `models/.venv/bin/modelscope download --model PaddlePaddle/Unlimited-OCR-2 --revision 7eadb7cd… --local_dir models/Unlimited-OCR-2` |
| 权重 | `model-00001-of-00002.safetensors` 4,998,430,312 B / `model-00002-of-00002.safetensors` 1,689,875,264 B |
| 官方源码 | 12 个 `.py`/`.json`（`modeling_unlimitedocr.py`、`dash_attention.py`、`_sparse_base_impl.py` 等） |

**12 个官方源码 SHA + 2 个权重 SHA 全部匹配** `tools/oracle/unlimited_ocr/verify.py`
里 pin 的值。oracle 从不改官方实现，只用运行时 hooks。

官方 `config.json` 声明 `model_type=unlimited-ocr`；24 层里 11 层是 attention
（4 个 KV head）、13 层是 shortconv，5 个 SWA 层，MoE 64 expert top-6 + shared
expert，hidden 1280，FFN 1792（expert）/ 6848（layer 0 dense）。

---

## 3. 多精度转换覆盖（README 只说验过两种，实际六种都能跑）

`tools/converter/unlimited_ocr/convert.py` 支持 `bf16/f16/f32/q8_0/q4_k/q6_k`。
本机逐一转换并加载运行：

| 格式 | 文件大小 | 张量类型分布 | 能跑 | 与 bf16 的输出 |
|---|---:|---|---|---|
| bf16 | 6698 MB | 2818 BF16 + 1 I64 | ✓ | 基线 |
| f16 | 6698 MB | 2818 F16 + 1 I64 | ✓ | **逐字节相同** |
| f32 | 13386 MB | 2818 F32 + 1 I64 | ✓ | 16-token 跑，**常驻 22 GB** |
| q8_0 | 3574 MB | 2402 Q8_0 + 416 BF16 + 1 I64 | ✓ | 文字同，1 个框位移 1 px |
| q4_k | 3082 MB | 同上（K-quant） | ✓ | 文字同，框位移 1 px |
| q6_k | — | 同上（K-quant） | 转换中 | — |

要点：

- **量化只作用于对齐的 2-D `.weight` 矩阵**（`convert.py:30`），convolution、position、
  向量、未对齐矩阵保持源类型 —— 所以量化导出是**混合精度**（q8_0 里 416 个张量仍是
  BF16，共 24 MB）。这是正确的做法，但文档里应说清楚「不是全量化」。
- **f16 与 bf16 逐字节相同**：BF16 的 8 位尾数落在 F16 的 10 位尾数里且不溢出，
  转换精确无损。这条可以作为「F16 档是 BF16 的严格替代」写进文档。
- **f32 导出在 22 GB 机器上常驻 22 GB**。README 已提到「F32 投影在构造请求视图时
  拷贝映射值，增加启动时间与内存」，实测印证了这一点，但没给数字。

**README 里的说法需要更新**：原文说「Real BF16 and Q8_0 exports have been run
through CLI; the other four whole-model formats have not been validated」——
现在六种都验过了。`docs/develop/SUPPORTED_MODELS.md` 已改成「六种整模实测」。

---

## 4. CLI 入口

| 检查 | 结果 |
|---|---|
| `--out` 拒绝已存在的目标 | ✓ `Publish output to …: File exists (os error 17)`，加 `--overwrite` 才写 |
| 多页 `--reference` | ✓ 第二页输出 `<PAGE><BLOCK>title<LEVEL2>…<FORK>PAGE TWO` |
| `--compute vulkan` | ✓ 原本被拒（`--compute vulkan supports plain text decoder generation only…`），**本轮放开**：现在真跑设备，输出与 CPU 逐字节相同（见 §7.5） |
| 页数上限 | 1..=128 |
| 每页 token 上限 | 20000，超出时强制结束该页 |
| 单页 128 token 输出 | 与作者记录的 110 个 token ID **逐字节相同**（含全部页面坐标） |
| 多页 256 token 输出 | 与作者记录的 212 个 token ID **逐字节相同** |

**注意**：上面两条「逐字节相同」是对**作者 commit 里那张已删除的 PNG** 跑的；换成生成
的页面后文字相同、坐标位移 1–3 px（见 §6）。

---

## 5. HTTP 面

作者自带的 `check_http.py` 本机跑 **11 项全 PASS**：

| 用例 | 状态 |
|---|---|
| `temperature` / `top_p` / `enable_thinking` / `max_tokens` / `no_image` | 400，消息逐条正确 |
| `concurrent` | 429，`Another generation is running; retry after it completes` |
| `chat` / `anthropic` / `chat_sse` / `responses` | 200 |
| `count_tokens` | 200，`input_tokens: 279` —— 与作者记录的 `prompt_tokens = 279` 一致 |

服务端 JEV 路由对 OCR 后端返回 unsupported-backend（§7 修过状态码）。

---

## 6. 页面 fixture：从提交的二进制改成合成脚本

**问题**：`tools/oracle/unlimited_ocr/fixtures/page-{1,2}.png` 是提交进仓的 28 KB /
26 KB 图片二进制，而且 `check_http.py` 直接读它。仓库的惯例是**不提交生成的资产**
——`src/app/server/api/fixtures.rs` 里就写着：`references/apple.png` 那棵树被 gitignore
了，把它的二进制拷进 `tests/` 与仓库的卫生不一致。

**已改**：加 `tools/oracle/unlimited_ocr/make_pages.py`，PNG 从 git 移除并 gitignore，
`check_http.py` 在缺图时自动调用生成器。

**一个绕不开的事实**：原来那两张图**在 Linux 上无法逐字节复现**。作者用的是 macOS
系统字体（按 ink 宽高比反推是 Helvetica 一类，比 Arial 窄约 6%）。本机对 1930 个系统
字体做了穷举（按 ink 框尺寸筛出 23 个候选，再逐像素比对），**零个精确命中**，最佳
候选仍差 2112 个像素、形状 IoU 只有 0.768 —— 是字形本身不同，不是光栅化差异。

因此生成器改用 **Liberation Sans**（Debian `fonts-liberation`，脚本按 SHA-256 固定
它）：两页的 ink 框与原图**完全一致**（page-1 连边界都相同），文字相同，模型预测的
页面坐标位移 1–3 px。

**后果**：`verification.json` 里作者的 case 数据是拿已删除的 PNG 跑出来的，**无法再
复现**。本机用生成的页面重算了 oracle，结果记在
[`verification-linux.json`](../../tools/oracle/unlimited_ocr/verification-linux.json)，
作者的 `verification.json` 保留为 macOS 那次的记录。

---

## 7. CPU/GPU 分发

### 7.1 设备上跑的是什么

只有**线性投影**在设备上；attention、路由、shortconv 都在 host。这与 LFM3 的分发
形状一样，但 LFM3 会无条件打印一行诊断（`LFM3 projections: requested=… backend=…`），
**OCR 这条路径没有等价的诊断** —— 不设 `RMI_COMPUTE_TRACE=1` 就看不出投影到底在
哪跑。建议补一行。

### 7.2 关键发现：单行 decode 的投影根本没上设备

`src/ops/kernel/vulkan.rs:40` 的 `min_gpu_matmul_outputs()` 默认 **16384**，
投影要 `rows × n_out ≥ 16384` 才离 CPU。OCR 的各投影：

| 投影 | n_out | decode(rows=1) | prefill(rows=279) |
|---|---:|---|---|
| attention q/k/v/o | 1280 | 1280 ✗ | 357 120 ✓ |
| MoE expert gate/up | 896 | 896 ✗ | 249 984 ✓ |
| MoE expert down | 1280 | 1280 ✗ | 357 120 ✓ |
| MoE router gate | 64 | 64 ✗ | 17 856 ✓ |
| shared expert | 1792 | 1792 ✗ | 499 968 ✓ |
| lm_head | 128000 | 128000 ✓ | ✓ |

**所以 prefill 大部分上设备，decode 只有 lm_head 上设备。** 实测印证：16 token 的
Auto 跑有 2517 次提交；把门槛降到 0（`RUST_GPU_MIN_OUTPUTS=0`）后升到 **7628** 次，
输出不变。

### 7.3 每个设备投影还要跟 CPU 抽样核对

`Linear::forward_verified`（`src/models/diffusion/dreamx/kernels.rs:139`）在用设备
结果之前会抽样跟 CPU kernel 对一遍（`GPU_CHECK_SAMPLES = 65_536` 个元素），不一致
就重试、再不一致就在 CPU 上重算。这是 TODO-044 那个「设备偶尔返回上一轮结果」的
对策，OCR 路径继承了它。**这是正确性上的正确取舍，但它是 Auto 路径额外的一笔 CPU
开销，应该在文档里写出来。**

### 7.4 性能

128 token、8 线程、Intel 集显：

| 路径 | wall | maxRSS | 输出 |
|---|---:|---:|---|
| CPU | 58.12 s | 6.39 GB | 基线 |
| Auto（Vulkan 投影） | **118.18 s** | 6.77 GB | **与 CPU 逐字节相同** |

**Auto 慢 2.03 倍。** 两个原因叠加：集显的设备内存就是系统内存（与 LFM3 同样的
问题）；以及每个设备投影的 CPU 抽样核对开销。**在集显上不应把 Auto 当作加速路径。**

### 7.5 `--compute vulkan`：去掉白名单，改成 warning

作者在 OCR 上留了两道拒绝：

1. `src/app/cli/types.rs` 的模式白名单 —— 含 `self.image.is_some()`，而 OCR 必带
   `--image`（页面图），所以 OCR 必然被挡；
2. `src/models/unlimited_ocr/mod.rs` 里手写的一条
   `Unlimited-OCR supports --compute auto hybrid offload; no complete Vulkan graph`。

再加上 `src/compute/mod.rs` 的 arch 白名单（`llama|qwen3|qwen35|qwen35moe|dory`），
OCR 有三道门。**但白名单挡不住真正的漏洞**：`Configure_cli()` 里**只有 Auto 会
`ops::enable_gpu()`**，而共享线性离 CPU 的 `offload_enabled()` 要求 `gpu_requested()`
—— 也就是说**就算把白名单全放开，`--compute vulkan` 也会静默跑成纯 CPU**
（实测 `backend=Vulkan` 行数为 0）。请求被拒反而比请求被静默降级更诚实。

**已改**（本轮）：

| 位置 | 改法 |
|---|---|
| `src/compute/mod.rs` | 拆出 `has_complete_device_path(arch)`；`validate_text_arch` 改成**只警告一次**（进程级 `AtomicBool`）；`configure_cli()` 对 Vulkan 也 `enable_gpu()`，并强制设备可用 |
| `src/app/cli/types.rs` | 模式白名单改成**警告**（每次请求只打一行）；`--gpu` + `--compute` 互斥仍是硬错误 |
| `src/models/unlimited_ocr/mod.rs` | 删掉那条手写拒绝；补一行设备路径诊断 |
| `src/app/server/mod.rs` | 删掉 `/v1/jev/*` 在 Vulkan 下的 blanket 400，改成启动警告（见 §7.6） |

警告长这样：

```
compute requested=Vulkan backend=partial mode=image note=no complete device path for this mode; only the offloadable parts run on the device
Unlimited-OCR projections: requested=Vulkan backend=Vulkan; attention/routing/shortconv=Cpu
```

第二行是给 OCR 补的**模型级诊断**，格式照抄 LFM3 的 `LFM3 projections: …`
（§7.1 提过 OCR 缺这一行）。第一行是 CLI 层的通用警告。

**严格性没有丢**：设备不可用时 `--compute vulkan` 硬失败，不静默降级：

```
$ VK_ICD_FILENAMES=/nonexistent.json … --compute vulkan
--compute vulkan requires a usable Vulkan device: compute device failure: Vulkan init failed: Unable to find a Vulkan driver
# exit=2
```

`--compute auto` 在同样环境下仍然回退 CPU（既有行为）。

**实测（Intel 集显，BF16 导出）**：

| 用例 | 结果 |
|---|---|
| OCR CLI `--compute vulkan`（单页，32 token） | **2442 次设备提交、2533 submissions、0 条 CPU 回退**；输出与 `--compute cpu` **逐字节相同** |
| OCR server `--compute vulkan` | 起得来；`check_http.py` **11 项全 PASS** |
| OCR CLI `--compute vulkan` 与 `--compute cpu` 的输出 | **逐字节相同** |

### 7.6 服务器里还有一道同类拦截：`/v1/jev/*`

`src/app/server/mod.rs` 里有一段 axum 中间件：policy 是 Vulkan 时，`/v1/jev/*` 一律
回 400 `JEV has no complete Vulkan execution path`。这是同一类白名单，**但实测它是
过度拦截**：

| 用例 | 结果 |
|---|---|
| llama-1B JEV，CPU vs Vulkan | **逐位相同**（`A=0.8926224708557129`、`B=0.10737749934196472`、entropy `0.34099701046943665`、margin `0.7852449417114258`） |
| HTTP `/v1/jev/score`（llama，Vulkan） | **200**（改动前是 400） |

**已改**：删掉那段中间件，改成启动时打一行警告
（`compute requested=Vulkan backend=partial mode=jev …`）。**正确性交给模型自己的
契约门禁** —— 它比中间件更准：

- `qwen3` + Vulkan 的 JEV → 门禁在 token 0 就拦下
  （`model logit mismatch at 0: gpu=5.6980243 cpu=3.4995377 abs=2.1984866`）→ **500**，
  而不是悄悄给出一个错的分数。
- OCR 后端 + Vulkan 的 JEV → **400 `/v1/jev/* requires a text backend (got unlimited-ocr)`**
  —— 理由是「这个后端不是文本后端」，而不是「Vulkan 没有完整路径」。

### 7.7 顺带查出来的既有问题（不是本轮引入的）

`qwen3` 在 arch 白名单里，`--compute vulkan` 本来就能跑，但它的 CPU 与 Vulkan 输出
**差一个 token**（CPU 多一个 emoji `😊`），用 `--prompt hello` 这种单 token prompt 时
差异更大。**stash 回改动前的 HEAD 重编译重跑，输出完全相同** —— 是既有的数值偏离，
记在 TODO-046。仓库里其实已经记过一次同类现象：`docs/develop/UNIFIED_COMPUTE_VALIDATION.md`
写的远程 AMD 上 `qwen3` **模型准入门禁在 token 0 就失败**
（GPU `4.476106` vs CPU `4.4538846`，abs `0.022221565`，门槛 `2e-3`）→ `--compute vulkan`
exit 1。**本机（Intel 集显）过得了 token 0 的门禁，却在后面对齐时翻了一个 token。**

---

## 8. Oracle 对拍结果（本机）

| 用例 | 检查点 | 结果 | 最大 peak-relative | 最小 cosine |
|---|---:|---|---:|---:|
| 单页 | 3040 | **通过** | 0.00009388 | 0.999999999 |
| 双页 | 5835 | **未通过** | 0.01024003 | 0.999999569 |

双页在 `uocr.attention` decoder layer 2 occurrence 2 处越界（门槛 0.002，cosine
门槛 0.99999）。**这与作者在 macOS 上保留的例外是同一签名**（attention，decoder
layer 2/3/5/7/8），量级也相近（作者 0.012268，本机 0.010240）。作者归因于 MoE top-6
的离散截止：decoder layer 1、图像第 173 行，官方第 6/7 专家的分只差 0.00000149。

也就是说：**那个不稳定不是作者机器的环境问题**，换机器、换渲染字体后照样出现。
反过来说，「4 组输入各 8 步 greedy 与固定 llama.cpp 一致」这类声明**不能推广到任意
文档** —— 路由落在截止边界上时，同一个模型也会因为 1e-6 的舍入差而选到不同的专家。

另外，官方模块在本机重算的 primitive 与仓库里 pin 的
`tests/fixtures/unlimited_ocr_oracle.json` 在 float32 的 **1–2 ULP** 上有差异
（`rope/ones_rotated`、`route/bias`）。那是环境差异（PyTorch/BLAS 版本），不是 Rust
实现差异 —— 该 fixture 存的是官方代码自己的输出，Rust 测试的容差覆盖得住，测试通过。

---

## 9. 本轮我改的东西

| commit | 内容 |
|---|---|
| `10e5944d` | 删掉已移除的 OCR JEV 集成留下的脚手架（两个 `_with_ids` 变体、`allocate_group_labels` 的可见性放宽）；`src/app/jev/` 现在与 main 零差异 |
| `6cd5c6d5` | `src/app/server/api/unlimited_ocr.rs` → `ocr.rs`（api/ 按能力命名，不按模型） |
| `5bfcec7c` | JEV 请求打非文本后端返回 400 而不是 500；删掉 4 段复制粘贴的 OCR 专用拒绝块 |
| `dca8bf06` | 页面 fixture 改成合成脚本，PNG 不入库；`check_http.py` 自动生成 |
| （本轮） | `--compute vulkan` 去掉白名单改成 warning；Vulkan 也 `enable_gpu()` 并强制设备可用；OCR 补设备路径诊断；六种格式实测；本机验证记录 |

### 9.1 JEV 状态码：一个既存的 500

作者的分支在 4 个 JEV 端点各复制了一段 7 行的 OCR 专用拒绝块。我把它去掉再跑真
server，发现它**唯一的作用是把 500 变成 400**：

```
HTTP 500  /v1/jev/* requires a text backend (got unlimited-ocr)
```

根因是「拿 JEV 的服务器不是文本后端」被映射成 `INTERNAL_SERVER_ERROR` —— 那是
**客户端错误**。而且这不是 OCR 特有的：Embedding / Asr / Audio8 / Tts / Rerank /
Gliner2 / Timestral 打 `/v1/jev/*` 拿到的都是 500。

修法是复用现有谓词（`text_source` / `text_tokenizer`）而不写白名单 —— 支持集合
很杂（score 端点服务 Text/Clm/Gliner2，image 端点只服务 Text，Gliner2 有自己的
提前返回分支），白名单一旦漏掉 Gliner2 就直接弄坏它。改完后 4 个端点对 OCR 后端
都返回 400，其它非文本后端也从 500 变成 400。

---

## 10. 未验证与边界

- **双页 whole-graph 在首个失败检查点之后没有继续比**（比较器遇错即停）。
- **组件级对比（vision / decoder 单独比）本机没跑** —— 需要官方视觉嵌入作为 decoder
  输入，即 `real_decoder_from_official_embeddings` 那条被 `--ignored` 的测试。
- **q6_k 导出在本报告写成时仍在转换**，未跑满 token 预算。
- **没有独显**。Auto 的性能结论只覆盖集显。
- **原仓库的全量测试没跑**（作者也没跑；作者记录的是 1292 passed / 20 failed，
  且那组数字早于其集成修复）。
- 动态切片 / gundam / PDF 栅格化 / 采样 / 恒定内存无限流：官方与本仓库都未实现。
