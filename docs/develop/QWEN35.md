# Qwen3.5 / Qwen3.8-27B：验证记录

> 本页汇总这个型号的验证证据、限制与实测数字。模型清单见
> [SUPPORTED_MODELS.md](SUPPORTED_MODELS.md)；MTP 投机解码的实现沿革与逐条命令见
> [TODO.md](TODO.md) 的 TODO-072；对拍测试见
> [`tests/qwen35_reference.rs`](../../tests/qwen35_reference.rs)。
>
> 更新于 2026-11-18。

## 入口

- GGUF：`general.architecture = qwen35`。**Qwen3.8-27B 是独立型号**，与 Qwen3-8B
  不是同一个。
- 图像：需要配套的 F16 mmproj；视觉塔始终在 CPU 上跑。
- MTP 投机解码：CLI `--mtp-draft N`（每轮起草 N 个 token；仅 greedy、仅纯文本
  prompt——图像路径的 prompt 行是投影后的 embedding，拿不到 token id）。
- DSpark 投机解码（Qwen3.5-2B 实测）：CLI `--dspark-model <sidecar.gguf>` 与
  HTTP `--serve --dspark-model <sidecar.gguf>` 双入口（同一
  `RuntimeOptions::with_dspark_model` 分发；可选 `--mtp-draft N` 指定每轮草稿
  数，默认 8；与 MTP 同样仅 greedy、仅纯文本 prompt、目标走 CPU 路径——非
  `qwen35` target、设备路径在启动时报错，temp>0 或带图请求在请求时报错，flag
  不会静默降级）。sidecar 是独立的 `dflash` 架构 GGUF（如
  `Qwen3.5-2B-DSpark`），起草 token 与 pinned llama.cpp `b96806d` 的
  `--spec-type draft-dspark` 逐 token 一致；CLI 与 HTTP 输出逐字节一致且与
  plain greedy 相同（哨兵 `cli_and_http_agree_on_dspark_speculative_decode`
  断言）。实现与验证记录见
  [SPECULATIVE_DECODING.md](SPECULATIVE_DECODING.md) 与
  [tools/oracle/dspark/README.md](../../tools/oracle/dspark/README.md)。

## CPU：与 pinned llama.cpp 的对拍

对 pinned llama.cpp `b96806d` 逐 checkpoint 对拍：

- **无损档（BF16）**：embedding 与 GDN 递推 state **逐位一致**，其余 checkpoint 在
  既有容差内。
- **Q8_0 档**：llama.cpp 把激活量化后再投影，与本仓库的 Q8_0 kernel 差约 1 ULP；
  深层栈把这个差放大（实测相对 L2：GDN 侧 2.2e-7…3.5e-7、第 3 层 1.8e-3、
  第 63 层 7.4e-2、logits 5.4e-2）。该档因此按 **GDN checkpoint 相对 L2 ≤ 1e-5**
  验收，并以 **greedy token 逐个相同**为功能门限：16 步生成里 Rust 与 pinned
  llama.cpp 的 15 个 token（含 EOS）逐个相同
  `[109266, 6115, 116577, 108888, 95933, 1710, 110450, 97375, 98691, 111454, 96598, 97319, 98179, 10992, 248046]`。
- `--mmproj` 图像端到端冒烟通过。

## 设备路径（Vulkan）

Q8_0 matmul 的 16384 宽度上限已随分块 kernel 放宽（#238）。逐位一致的部分：

- 设备 kernel 对 CPU 归约逐位一致；
- 16 步 greedy 的 15 个 token（含 EOS）CPU 与设备逐个相同，logits（248320 维）
  逐位一致（relative L2 = 0）；
- 372 与 892 token 的 64 行批次预填同样逐位一致；
- 图像（`--mmproj` + `--compute vulkan`）的 prompt id 与 greedy token 也与 CPU
  逐个相同，视觉塔仍在 CPU。

速度（这台机器，8 线程）：

| | 设备 | CPU |
| --- | --- | --- |
| prefill | 快约 5～12 倍（短/长 prompt） | – |
| 单行 decode（Q4_0 档，17 GB 权重） | 404 ms/token | 187 ms/token |
| 单行 decode（Q8_0 档，30 GB 权重） | 406 ms/token | 274 ms/token |
| 3 行前向 | 175 ms/行 | 187 ms/行 |
| 8 行前向 | 103 ms/行 | 187 ms/行 |

即**设备的优势在多行**（tiled kernel 让多行共享一次权重读取），**单行 decode 反而
比 CPU 慢**。所以设备路径的价值在批量（预填、投机解码的验证），不在单行解码。

## MTP 投机解码

GGUF 自带 block 64 就是 MTP 起草头，已实现并与 pinned llama.cpp `b96806d` 对拍：

- 第一轮的 4 个草稿与 `llama-server --spec-type draft-mtp --spec-draft-n-max 4`
  记的候选**逐 token 相同**；
- 接受率 13/24 = 0.54（llama.cpp 10/19 = 0.53，同一模型 / prompt / n_max）；
- 投机解码产出的 token 与 greedy **逐个相同**。

**CPU 上**：27B Q8_0 的前向是**访存**瓶颈（一遍前向要读 29 GB 的 Q8_0 权重），
2 个草稿的投机解码比 greedy 快约 1.3 倍（4.67 s vs 5.98 s，19 token prompt +
16 个生成 token，8 线程）。

**设备上**（#250）：目标前向跑在 Vulkan 上时投机解码也能跑。

- 设备把每一行的 hidden 与词表投影一次性交回来（新增 `logits_rows` 区，整批行
  一次词表投影，而不是每行读一遍 1.27 GB 的词表矩阵）。
- **起草头也跑在设备上**：它是主栈之后的一个稠密块——自己的注意力、自己的 FFN、
  自己的 norm；KV cache 是设备 arena 里主栈之后的那一层槽位；`eh_proj` 吃
  「归一化 embedding ‖ 归一化 hidden」，设备没有 concat，于是两个 norm 各写进
  同一个 buffer 的两半；词表投影复用设备上已有的 `output.weight`；catch-up 是一
  批行。起草头的权重如果设备放不下就留在 CPU 上，目标仍然跑在设备上。
- 设备的回滚是「只重放循环状态更新」：chunk 捕获每个循环层的**投影**（投影后的
  q/k/v、gate、beta/alpha），回滚时把保存的状态上传回去，只重放 conv + SSM
  scan——投影那一半要读 48 个循环层的权重（3.1 GB），重算它要 122 ms，捕获之后
  回滚只要 **20～26 ms**（真模型上整个回滚从 1.54 s 降到 0.02 s）。
- 真模型上投机解码产出的 token 与 greedy **逐个相同**（Q8_0 档接受率 0.54，与
  CPU 一致）。
- **Q8_0 与 Q4_0 档上设备的投机解码都比 greedy 快约 1.55 倍**（2 个草稿；Q8_0 的
  decode 段 8.1 s vs 12.6 s，Q4_0 是 7.8 s vs 12.1 s；18 token prompt + 32 个
  生成 token，`--compute vulkan`，交替跑取中位数）。

前提是**多行前向在设备上足够便宜**：Q8_0 与 Q4_0 都有 tiled kernel（权重跨行
共享，算术与逐行 kernel 逐位相同），所以验证那一遍的 3 行 ≈ 1 行；其余权重档位
（Q4_1 / Q5_0 / Q4_K / Q6_K …）还没有，那些档位的多行前向要付 N 倍权重流量，
投机解码在设备上没有收益。

## 已知限制

- **Q8_0 档与 llama.cpp 不是逐位一致**（约 1 ULP 的激活量化差，深层栈会放大），
  按上面的相对 L2 门限与 greedy token 门限验收。
- **设备的单行 decode 比 CPU 慢**（见上面的表）；设备路径要靠批量才划算。
- **设备上的投机解码目前只在 Q8_0 与 Q4_0 档有收益**（其余档位缺 tiled kernel）。
- 只有稠密 `qwen35` 装载 MTP 头；`qwen35moe` 的 nextn 走专家，是另一张图。
