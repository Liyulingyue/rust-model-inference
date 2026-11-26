# OCRia-nano-1 适配记录

> 本页汇总契约、入口、覆盖矩阵、数值证据与限制。逐检查点的实测数字与复现命令在
> [oracle 记录](../../tools/oracle/ocria/README.md) 与
> [validation.md](../../tools/oracle/ocria/validation.md)；模型清单见
> [MODEL_LIST.md](../MODEL_LIST.md)。
>
> 更新于 2026-11-26。

## 模型与组件

`llamaindex/OCRia-nano-1` 是 LlamaIndex 发布的文档 OCR 流水线，原始权重是六个 ONNX 图，
不是一个自带的推理脚本：`det.onnx`（行检测）、`router.onnx`（脚本路由，四组 + 倒置投票）、
`rec_{latin,cjk,rtl,world}.onnx`（CTC 识别，图内直接输出 `ids`/`conf`），外加 LayoutNet
（`tablenet.onnx` / `tablenet.bin`）。本仓库把前六个图打进一个 `ocria` GGUF，用仓库自己的
卷积算子推理；LayoutNet 属于第二层，见 [分层交付](#分层交付)。

| 组件 | 张量数 | 说明 |
| --- | ---: | --- |
| `det` | 74 | luma 混合 → stem → 4 个 dw/pw 阶段 → 1x1 侧向 → 最近邻 2x FPN → 头部 → 图内 `Sigmoid` |
| `router` | 34 | stem → 7 个残差块 → `concat(mean_w, max_w)` → 两个 `Gemm(transB=1)` |
| `rec.latin` | 58 | stem + 9 个 dw/pw 块 + 3 个 `mix` 块 + 单头 `[792,192]`；原图是 QDQ int8 |
| `rec.cjk` | 60 | 同上，末端宽度也降采样（`T = W/8`），两个 head 卷积（中间无 Relu） |
| `rec.rtl` | 58 | 同上，`visual_rtl=true`，输出按 bidi 还原为逻辑序 |
| `rec.world` | 58 | 同上 |

共 342 个张量。卷积核按 C 序展平成二维 `[out, fan_in]` 存储，所以逐点卷积的核是一行，
Q8_0 的 32 值块能整块量化；偏置与 9 元素深度卷积核保持 F32。

## 入口

CLI（模型按 `general.architecture=ocria` 自动选中）：

```bash
rust-model-inference --model ocria-f32.gguf --image page.png --out lines.json \
  [--threads N] [--compute cpu|auto] [--overwrite] [--ocr-options '<json>']
```

`--out` 省略时 JSON 写到 stdout。`--mmproj` 会被明确拒绝（一个 GGUF 自带全部组件）。

`--ocr-options` 覆盖页面管线参数，缺省字段保留 GGUF 契约里的值（检测阈值/合并距离）或参考默认
（各开关）：

```json
{"long_side":1600,"threshold":0.55,"merge":0.6,"gutter":true,
 "orient":true,"rescue":true,"junk":true,"faint_threshold":0.1,"faint_max":24}
```

JSON 语法与字段名在参数解析阶段就校验（拼错不会等到加载模型），取值范围在套用到模型默认值时
校验（`threshold` ∈ (0,1)、`long_side` ∈ [32,8192]、`merge` ≥ 0、`faint_threshold` ∈ [0,1] 等）。
`--ocr-options` 用在非 `ocria` 模型上会明确报错，不会被静默忽略。

HTTP：

```bash
rust-model-server --model ocria-f32.gguf --host 127.0.0.1 --port 8080
curl -f http://127.0.0.1:8080/v1/ocr -H 'Content-Type: application/json' \
  -d "{\"image\":\"data:image/png;base64,$(base64 -i page.png)\"}"
```

`image` 接受 `data:` URL 或裸 base64，远端 URL 明确拒绝（400）。请求可带一个 `options` 对象，
字段与 `--ocr-options` 相同，用于覆盖这一页的参数（非法取值返回 400）。CLI 与 HTTP 返回同一份
JSON（`width`/`height`/`lines[text,bbox,confidence,polygon,faint]`），`page_read_matches_liteparse`
对同一页图像验证两者共用的 `Model::run`，`tests/ocria_cli.rs` 另外验证 CLI 的参数校验与真实出图。

JEV 不适用：这是行检测 + CTC 识别的几何输出流水线，没有候选标签上的概率/评分语义。

## 覆盖矩阵

| 能力/组件 | 权重格式 | 计算/KV 精度 | 请求后端 | CLI 命令与结果 | HTTP 路由、请求与结果 | 实际执行后端、数值指标与证据 | 实现/验证状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 整页 OCR（det + router + 4×rec + 页面管线） | `ocria-nano-1-f32-v1`（342 F32） | F32 权重、F32 计算、无 KV | `--compute cpu` / `auto`（Auto 走 CPU） | `--model ocria-f32.gguf --image page.png --out lines.json` → 三页 fixture 文本逐行一致 | `POST /v1/ocr` + base64 图 → 同构 JSON | CPU（`ops::nn::conv2d/conv1d`、`dot_f32`、`softmax_inplace`）；det `prob` max abs 7.2e-7、router logits 2.9e-6、rec `ids` 0/1064、rec `conf` ≤2.7e-6；三页端到端文本/行数精确一致、bbox ≤1 px、页均值置信度差 ≤2.0e-3 | 已实现 / 通过 |
| 同上 | `ocria-nano-1-q8_0-v1`（257 F32 + 85 Q8_0） | F32 计算（加载时反量化） | `--compute cpu` / `auto` | 同上，三页文本逐行一致 | 同上 | 同上；量化后 det `prob` max abs 1.9e-2、router logits 6.1e-2、rec `ids` 22/1064 差、行置信度差 ≤7.0e-2，**三页文本仍逐行一致** | 已实现 / 通过（文本一致；概率图与置信度按量化误差评估） |
| 严格设备模式 | 任一 | — | `--compute vulkan` | 启动即明确报错 | 启动即明确报错 | 无设备算子：发布方是 CPU 流水线，仓库 Vulkan 内核只覆盖线性投影，本模型的 dw/pw 卷积没有设备实现 | 不适用（契约依据见下）/ 未运行 |
| 组件缺失/不匹配 | 任一 | — | 任一 | 缺张量、形状不符、charset 与 head 类数不符、`ocria.config` 与契约不符都在加载期指名报错 | 同上 | `contract.json` 342 个名字+形状双向校验；转换器与加载器共用同一 Q8_0 行宽规则 | 已实现 / 通过 |
| 页面管线开关（faint / gutter / orient / rescue / junk） | 任一 | — | 任一 | 三页 fixture 分别覆盖单列、双列栏缝切分、四分之一转重试；`--ocr-options` 可覆盖 | 请求级 `options` 对象（非法值 400） | `sideways.png` 17 行、`two_column.png` 25 行与参考一致；`tests/ocria_cli.rs` 用 `faint_threshold:0` 验证开关生效 | 已实现 / 通过 |
| LayoutNet（表格 + 版面块） | — | — | — | — | — | 未实现：`tools/oracle/ocria/layout-plan.md` 给出契约、可达性与工作量（≈2 800–3 600 行 Rust + 转换器），并指出 OCR 页路径硬依赖 LiteParse 的 projection 信号 | 未实现 / 未运行（第二层） |
| PDF 输入 | — | — | — | — | — | 仓库无 PDF 渲染器；本模型只接受页面图像 | 不适用 / 未运行 |

`--compute vulkan` 判为不适用而非「未实现」的依据：发布方把本模型定义为 CPU 流水线
（模型卡 "CPU runtime, everywhere LiteParse runs"），参考实现也没有设备路径；仓库的
Vulkan 能力目前只覆盖线性投影，而本模型的算力全在深度可分离卷积上。严格模式请求会
在加载期报错退出，不静默降级。

## 数值对齐

固定输入：模型 `master` @ `ca290e82458d32aa8b605dbf3248d7aaf895ea7e`（逐文件 SHA256 见
`sources.json`），参考实现 LiteParse `44ceb6ab160c64cc8b43aab4ed85251719495629`，
ONNX Runtime 1.32.0 CPU，`intra_op_num_threads=1`。

| 检查点 | 门槛 | 实测 |
| --- | --- | --- |
| det `prob`（256×256 与 96×160 非方图） | max abs ≤1e-5 | 7.15e-7 / 6.86e-7 |
| router `group` / `flip` logits | max abs ≤1e-4 | 2.87e-6 / 4.77e-7 |
| rec `ids`（4 组 × 4 宽度，1064 timestep） | 全等 | 0 差 |
| rec `conf` | max abs ≤1e-3 | ≤2.68e-6 |
| 端到端三页 | 文本+行数全等、bbox ≤1 px、页均值置信度 ≤5e-3 | 19/25/17 行全等；bbox ≤1 px；页均值差 ≤2.0e-3 |
| 单行置信度 | ≤2e-1（近并列字符的 QDQ 效应） | ≤6.2e-2 |

**逐位对齐的边界**：浮点路径（SIMD 卷积归约顺序、`ReduceLogSumExp`、`exp`）按误差门限
验收，不做逐位比较；`ids` 这类离散契约要求全等。`rec_latin.onnx` 是 ONNX Runtime 量化
图，会在图内把**激活**也吸附到 uint8 网格，本仓库不复制这一层（权重按 ONNX QDQ 公式
反量化为 F32）：因此 latin 的 `conf` 与官方 QDQ 图有 ~2e-2/时间步的差，真实文本上
0.56% 的时间步 `ids` 会翻（`latin_qdq_effect.json`）。fixture 里 latin 的对齐目标是
忠实 F32 图的 `ids_f32`/`conf_f32`，QDQ 数字并列保留作记录——不把「激活量化」当成实现误差。

## 性能

同一台机器（Linux x86_64，128 核，无 GPU；`release-fast`）上冷进程到出结果，含模型加载：

| 输入 | 行数 | Rust `--threads 4` | Rust `--threads 16` | LiteParse + ONNX Runtime |
| --- | ---: | ---: | ---: | ---: |
| `receipt.png`（600×800，单列） | 19 | 5.67 s | 2.00 s | 0.17 s |
| `two_column.png`（600×800，双列） | 25 | 9.74 s | 4.12 s | 0.15 s |
| `sideways.png`（800×600，侧转，两次检测） | 17 | 7.28 s | 2.78 s | 0.20 s |

**差距 12–27 倍，归因明确**：瓶颈是共享算子 `ops::nn::conv3d_cpu`。它对 1×1 逐点卷积走的是
「每个输入通道一次长度为 1 的 `dot_f32`」——`kernel_width = 1` 时内层点积退化成标量乘法，
192 个通道就是 192 次函数调用；深度可分离 3×3 同理是 9 次长度为 3 的点积。ONNX Runtime 用
NCHWc 打包 + 专用 depthwise/pointwise 内核，所以差一个数量级。

这不是本模型代码能局部修的：正确修法是在 `src/ops/nn.rs` 里加 1×1 与 depthwise 的专用路径
（NHWC 打包后按通道做连续点积），属于共享算子改动，会影响 audio8、unlimited_ocr、mage_vae、
Gemma 视觉塔等所有调用方，需要各自的回归证据，因此不在本 PR 内。记录在
[validation.md](../../tools/oracle/ocria/validation.md) 的性能一节。

参考基线不是硬门槛，但差距必须归因到具体算子——上面就是那个算子。

## 分层交付

本 PR 是**第一层**：OCR 栈（det、router、四个识别器、页面管线）加 CLI/HTTP 入口。

**第二层（待合入）**：LayoutNet 表格与版面模型。范围、契约与工作量见
[layout-plan.md](../../tools/oracle/ocria/layout-plan.md)：4 268 063 参数、110 个 ONNX
初始化器与 `tablenet.bin` 已验证为同一组权重；新增内核只有 `GridSample`、`AveragePool 2x2`
和 `sin/cos`，其余复用 `ops::nn` / `ops::norm` / `ops::softmax`。该计划的主要风险是
OCR 页路径硬依赖 LiteParse 的 projection 信号（8/66 个特征 + `same_blk`/`same_leaf`，
对应 3 500–4 500 行），因此第二层开工前先用一次消融实验判定是否必须移植该层。

第一层合入后模型状态是**部分适配（分层交付中）**，不是「适配完成」。

## 限制

- 只接受页面图像：没有 PDF 渲染、没有 PDF 文本层、没有多页请求。
- LayoutNet 未实现，所以没有表格/版面块输出。
- `rec_latin` 的激活量化未复制（见上）；其余三个识别器是纯 F32。
- 单页行数上限、`MAX_W = 1280` 的识别宽度裁剪沿用参考实现的语义：超过 40:1 的行会被
  降采样读取（与参考一致），时间步不足时报错而不是截断。
