# 默认 16 步生成与既有图像入口回归

2026-11-25 在 Apple M3 Max 上使用 `golden crown with blue gems`、seed 0、恒定 CFG 3、温度 1、位置选择温度 2 和空间 NMS。每种存储精度先由 gguf-py 0.19.0 独立反量化，再交给作者的 CLIP、Tessera 和 PyTorch CPU 采样器。固定版本与下载命令见 [README](README.md)。

```sh
for fmt in f32 f16 bf16 q8_0 q4_0 q4_1 q4_k q5_k q6_k; do
  .venv/bin/python tools/oracle/tessera/reference.py "$ASSETS" "$ASSETS/parity/$fmt" \
    --gguf "$ASSETS/tessera-small-$fmt.gguf" --default-sample --threads 3 || exit 1
done
.venv/bin/python tools/oracle/tessera/endpoints.py "$ASSETS" target/release-fast \
  "$ASSETS/parity/default-endpoints" --default-sample --repeat
```

## 权重精度对官方采样的影响

下表在同一官方实现内比较每档权重与原始 EMA + CLIP。记录每步提交后的 256 个 token ID，MASK ID 也参与比较。“首次不同”表示该步至少一个位置的 token ID 或提交状态不同。Q8_0 的轨迹短暂不同，最终图像相同。

| 权重 | 首次不同（共 16 步） | 最终相同像素 / 256 |
| --- | ---: | ---: |
| F32 | 全程相同 | 256 |
| F16 | 全程相同 | 256 |
| BF16 | 14 | 247 |
| Q8_0 | 8 | 256 |
| Q4_0 | 1 | 79 |
| Q4_1 | 4 | 182 |
| Q4_K | 3 | 150 |
| Q5_K | 1 | 124 |
| Q6_K | 6 | 222 |

## Rust 与同档参考的比较

每个组合运行 CLI 和真实 HTTP 请求。脚本精确比较二者解码后的 RGBA；再独立统计与本档 PyTorch 参考的相同像素数、合成到黑色背景后的 RGB RMSE（0–255 范围）、不透明像素区域的交并比（alpha IoU）。这是固定输入和 seed 的整条随机采样轨迹；各步的浮点校验见[组件矩阵](validation.md)。

27 组全部完成。每组由两个独立 CLI 进程和同一个 HTTP 服务中的两次请求组成；共 108 张图。每组两次 CLI、两次 HTTP、CLI 与 HTTP 三种比较均为 256/256 像素相同。各档 Auto 与严格 Vulkan 的 PNG 也相同。每次生成记录到的去噪器投影为 CPU 1840 次主机投影，Auto 和严格 Vulkan 均为 1840 次 GPU 投影。

每格依次列出相同像素数（满分 256）、RGB RMSE（0–255）及 alpha IoU。以下数值相对于本档同权重官方 PyTorch。

| 权重 | CPU | Auto | Vulkan |
| --- | --- | --- | --- |
| F32 | 256 / 0.000 / 1.0000 | 256 / 0.000 / 1.0000 | 256 / 0.000 / 1.0000 |
| F16 | 256 / 0.000 / 1.0000 | 256 / 0.000 / 1.0000 | 256 / 0.000 / 1.0000 |
| BF16 | 256 / 0.000 / 1.0000 | 256 / 0.000 / 1.0000 | 256 / 0.000 / 1.0000 |
| Q8_0 | 244 / 13.570 / 1.0000 | 254 / 0.629 / 0.9938 | 254 / 0.629 / 0.9938 |
| Q4_0 | 203 / 23.408 / 0.9647 | 198 / 27.950 / 0.9647 | 198 / 27.950 / 0.9647 |
| Q4_1 | 202 / 22.091 / 0.9355 | 244 / 15.709 / 1.0000 | 244 / 15.709 / 1.0000 |
| Q4_K | 233 / 24.215 / 1.0000 | 214 / 18.777 / 0.9848 | 214 / 18.777 / 0.9848 |
| Q5_K | 237 / 21.611 / 0.9930 | 234 / 17.349 / 0.9859 | 234 / 17.349 / 0.9859 |
| Q6_K | 134 / 57.521 / 0.8452 | 235 / 16.293 / 0.9815 | 235 / 16.293 / 0.9815 |

三个浮点档在本例的种子下与各自参考完全相同。块量化随机链可明显分叉，例如 Q6_K CPU 的相同像素数为 134/256；同档 CPU 的完整 PNG 与基线二进制逐字节相同。对全部九档 CPU 的两入口分别与旧版比对均逐字节一致；F32 的三后端、两入口六份 PNG 也分别与旧版逐字节一致。

同一设备上每个组合首次执行的墙钟时间如下，单位秒。CLI 包含模型加载、初始化和落盘；HTTP 从已启动的服务收到请求计时，包含推理与响应。PyTorch 列为已加载模型的 3 线程 CPU 采样时间，含逐步画布记录。Rust 入口使用 8 个线程。主机同时执行其他验证，时间未用于验收。

| 权重 | CPU CLI / HTTP | Auto CLI / HTTP | Vulkan CLI / HTTP | PyTorch CPU 16 步 |
| --- | ---: | ---: | ---: | ---: |
| F32 | 32.62 / 20.11 | 0.93 / 0.86 | 0.96 / 0.86 | 10.73 |
| F16 | 4.44 / 4.44 | 3.06 / 3.01 | 3.08 / 3.02 | 10.81 |
| BF16 | 8.53 / 8.45 | 1.03 / 0.98 | 1.05 / 0.98 | 10.91 |
| Q8_0 | 2.99 / 4.83 | 1.62 / 1.58 | 1.75 / 2.11 | 10.90 |
| Q4_0 | 27.58 / 8.78 | 1.69 / 1.66 | 1.74 / 1.76 | 11.14 |
| Q4_1 | 3.99 / 5.97 | 1.28 / 1.24 | 1.30 / 1.24 | 11.16 |
| Q4_K | 5.27 / 4.83 | 3.00 / 2.95 | 3.01 / 2.96 | 10.78 |
| Q5_K | 3.79 / 3.76 | 0.84 / 0.80 | 0.85 / 0.79 | 10.74 |
| Q6_K | 3.87 / 3.82 | 2.59 / 2.55 | 2.61 / 2.54 | 9.69 |

## 旧模型 CLI 与 HTTP 回归

基线为 PR 提交 `517fbfdf657c` 的 release-fast 二进制。候选与基线各运行同一个 CPU 请求，8 线程、1 步、seed 42、guidance 1。分别比较 CLI 与 `POST /v1/images/generations`，以及两者在新旧版本之间的 PNG 字节和解码后的 RGB。最终复测使用的 CLI 二进制 SHA256 为 `3fc007a6d1566ce34dc7764184cf41d750a9ce98557bbd51bd7126819549d41b`，服务器为 `9410f8b7032a8731c00d3b39d66ac4408c01fede41b1ce569870d227de82a540`。这组结果覆盖单步小图的入口行为及原有数值输出。

| 模型 | 实际组件及输入 | 旧/新、CLI/HTTP 的结果 | 解码 RGB SHA256 |
| --- | --- | --- | --- |
| ERNIE-Image | Q4_K_M DiT + Ministral-3-3B-Instruct-2512 Q4_K_M + Flux2 VAE；`a lovely cat`，64×64 | 四份 PNG 完全一致，4096/4096 像素 | `3f9666f299dd7331eb619e26536390b6d33fa807d6aa920da8e5845e88c98428` |
| P-JiT 21K | Q4_0 DiT + Qwen3.5-4B Q4_0；`A red cube on a white table.`，33×35（覆盖 pad/crop） | 四份 PNG 完全一致，1155/1155 像素 | `d317d69eaa4e71935012879ee9ab30c937ef10ae5556568979ce10ca83b9b75e` |

ERNIE 使用原[权重记录](../ernie_image/README.md)的三种组件：模型文件 SHA256 `ed43d36a…`，Mistral 官方文本 GGUF `9ed150d4…`，原始 VAE `d64f3a68…` 经仓库转换后为 `83556840…`。P-JiT 使用其[发布源与仓库转换器](../pjit/README.md)，实际导出的 Q4_0 SHA256 为 `a984b70a084202fb0d5fe6dbc095de145f46b7a3925c54685c590fa18faba86e`，与[既有导出记录](../pjit/gguf_hashes.json)一致。搭配的 `unsloth/Qwen3.5-4B-GGUF` 文本塔来自 ModelScope revision `167b4afc359863325cb4164418c715421b4e9118`，`Qwen3.5-4B-Q4_0.gguf` SHA256 `298fcb5fe7a77ccc79745ae24751560c5ac56874caff4bb39b1f2055bd72b8bb`。以下命令使用实际回归中的组件名，`MODELS` 指向本地模型目录。

```sh
e="$MODELS/ernie-image-regression"
p="$MODELS/pjit-21k-regression"
target/release-fast/rust-model-inference --compute cpu --threads 8 \
  --model "$e/ernie-image-Q4_K_M.gguf" --text-encoder "$e/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf" \
  --vae "$e/flux2-vae.gguf" --prompt 'a lovely cat' --resolution 64 \
  --steps 1 --seed 42 --cfg-scale 1 --out ernie-regression.png
target/release-fast/rust-model-inference --compute cpu --threads 8 \
  --model "$p/pjit-q4_0.gguf" --text-encoder "$p/Qwen3.5-4B-Q4_0.gguf" \
  --prompt 'A red cube on a white table.' --width 33 --height 35 \
  --steps 1 --seed 42 --cfg-scale 1 --out pjit-regression.png
```

HTTP 服务器使用同一组组件、线程与后端。请求指定相同 prompt、`size`、`steps: 1`、`seed: 42`、`guidance: 1` 和 `response_format: "b64_json"`；结果从 `data[0].b64_json` 解码。
