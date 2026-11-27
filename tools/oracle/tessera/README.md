# Tessera-small 对齐复现

Tessera by Jagrat Patel (TheJagStudio)。官方模型将 16×16 图像表示为 256 个调色板 ID，每个位置可取 512 种颜色（0 透明）。文本条件为冻结的 OpenAI CLIP ViT-B/32 的 24 个 token 隐藏态和 EOS 池化输出。转换后的单个 GGUF 包含文本塔、分词器、双分辨率去噪器和调色板。[模型卡](https://modelscope.cn/models/thejagstudio/tessera-small)

## 生成像素画

有单文件 GGUF 后即可通过现有 CLI 生成原生分辨率的 RGBA PNG。默认使用 16 步、CFG 3、颜色温度 1、位置选择温度 2 和 seed 0。

```sh
cargo build --locked --profile release-fast --features vulkan --bin rust-model-inference --bin rust-model-server
MODEL="$ASSETS/tessera-small-f16.gguf"
target/release-fast/rust-model-inference --model "$MODEL" --compute cpu \
  --prompt 'golden crown with blue gems' --seed 0 --out crown.png
target/release-fast/rust-model-server --model "$MODEL" --compute auto --host 127.0.0.1 --port 8080
```

在另一个终端发送单张图的请求，响应中的 `data[0].b64_json` 是带透明通道的 PNG。

```sh
curl -sS http://127.0.0.1:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"golden crown with blue gems","size":"16x16","seed":0}'
```

局部重绘在 CLI 通过 `--input options.json`，HTTP 通过 `pixel_options` 接收相同的对象。其 `canvas` 字段为 256 项行优先数组。用 `null` 标出需要重绘的像素，用 0–511 的调色板 ID 或 `[r,g,b,a]` 保留像素。ID 0 表示透明。RGBA 中 alpha 小于 128 的像素映射为透明，其他颜色取 OKLab 距离最近的不透明调色板项。每次生成一张 16×16 图。

## 固定资产

| 组件 | 来源 / revision | 校验 |
| --- | --- | --- |
| Tessera-small EMA、配置、调色板 | ModelScope `thejagstudio/tessera-small@f8663fba85fa71952b7727f373141df7d21fc9d7` | `tessera-small-fp16.pt` 95,254,426 字节，SHA256 `772a289de232950bc411bbfeaa4d60c320450b04cf5b7d8280c9c0b36880fec3` |
| OpenAI CLIP ViT-B/32 | ModelScope `openai-mirror/clip-vit-base-patch32@5587187bcb3a57301519ced320b42d3c4827897d` | `pytorch_model.bin` SHA256 `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`，与 [OpenAI 原版](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268) 一致 |
| Python 参考实现 | [作者 Space 源码](https://huggingface.co/spaces/thejagstudio/tessera-pixel-studio/tree/8de933771e65ab3931b456090d065a1044c9ff18/tessera) | `8de933771e65ab3931b456090d065a1044c9ff18` |

模型与作者参考实现使用 CC BY-NC 4.0；模型卡同时列明生成图像的非商业使用条件。CLIP 使用 OpenAI MIT 许可证。GGUF 将 Tessera 与 CLIP 的文本部分打包，矩阵可按需量化。以下步骤保留作者随模型发布的许可文件。[Tessera 许可](https://modelscope.cn/models/thejagstudio/tessera-small/file/view/master/LICENSE) · [OpenAI CLIP 许可](https://github.com/openai/CLIP/blob/main/LICENSE)

## 下载

从仓库根运行。此次验证将 `MODELS` 设为主 checkout 的模型目录；替换该变量可复用已有的 ModelScope 下载环境。

```sh
MODELS=/Users/gouzi/Documents/git/rust-model-inference/models
ASSETS="$MODELS/tessera-small"
MS="$MODELS/.venv/bin/modelscope"
if [ ! -x "$MS" ]; then
  test -x "$MODELS/.venv/bin/python" || uv venv --python 3.12 "$MODELS/.venv"
  uv pip install --python "$MODELS/.venv/bin/python" modelscope==1.40.1 modelscope-hub==0.4.5
fi
mkdir -p "$ASSETS/oracle/tessera"
"$MS" download --model thejagstudio/tessera-small \
  --revision f8663fba85fa71952b7727f373141df7d21fc9d7 --local_dir "$ASSETS" \
  tessera-small-fp16.pt model_config.json LICENSE README.md
"$MS" download --model openai-mirror/clip-vit-base-patch32 \
  --revision 5587187bcb3a57301519ced320b42d3c4827897d --local_dir "$ASSETS/clip" \
  pytorch_model.bin config.json tokenizer.json tokenizer_config.json \
  special_tokens_map.json merges.txt vocab.json

SPACE=https://huggingface.co/spaces/thejagstudio/tessera-pixel-studio/resolve/8de933771e65ab3931b456090d065a1044c9ff18
for file in LICENSE requirements.txt tessera/__init__.py tessera/config.py \
  tessera/model.py tessera/palette.py tessera/pipeline.py tessera/sampler.py \
  tessera/text_encoder.py tessera/utils.py tessera/viz.py; do
  curl -fLsS --retry 2 "$SPACE/$file" -o "$ASSETS/oracle/$file" || exit 1
done
```

## 转换及参考结果

实测环境为 Python 3.12、ModelScope 1.40.1。以下命令使用单独的仓库推理虚拟环境。转换器会校验 EMA、CLIP 权重和 6 个 CLIP 配置/分词文件的 SHA256 后再打包。

```sh
test -x .venv/bin/python || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.9.1 transformers==5.22.2 \
  tokenizers==0.23.3 numpy==2.5.5 pillow==13.0.0 gguf==0.19.0
PY=.venv/bin/python
"$PY" tools/oracle/tessera/reference.py "$ASSETS" "$ASSETS/parity/original" --default-sample
for fmt in f32 f16 bf16 q8_0 q4_0 q4_1 q4_k q5_k q6_k; do
  gguf="$ASSETS/tessera-small-$fmt.gguf"
  "$PY" -m tools.converter.tessera.convert "$ASSETS/tessera-small-fp16.pt" \
    "$ASSETS/clip" "$gguf" --format "$fmt" || exit 1
  "$PY" tools/oracle/tessera/reference.py "$ASSETS" "$ASSETS/parity/$fmt" --gguf "$gguf" || exit 1
done
```

浮点和块量化只作用于线性层与 CLIP 嵌入矩阵。Tessera 的 `tok_emb.weight`、宽 40 的首层颜色投影、偏置、归一化、卷积和调色板保留 F32。块格式要求每行维度满足块宽，因而宽 16 的 mask-ratio 投影在块格式中也保留 F32。

参考脚本直接调用 Space 的 `FrozenTextEncoder`、`Tessera` 和 `sample_iter`。`original` 使用原始发布权重；各 GGUF 档由独立的 gguf-py reader/反量化器恢复权重，再载入同一份 PyTorch 模型。每个档位的数值实现误差与相对原始发布的量化误差可分别计算。

| 输出文件 | 校验内容 |
| --- | --- |
| `tokenizer.json`, `ids.u32` | 四个 prompt 的精确 ID，覆盖空串、Unicode、空白和截断，主样例固定为 `golden crown with blue gems` |
| `states.f32`, `pooled.f32` | CLIP token 隐藏态 [24,512] 和 pooled [512] |
| `logits_{cond,uncond}.f32` | 全 mask 第一轮两个 CFG 分支，各 [256,512] |
| `sc-step-2.f32`, `logits_{cond,uncond}_2.f32` | 第一次提交后的自条件概率及第二轮两个分支，各 [256,512] |
| `greedy-step-{1..4}.u16`, `greedy.png` | CFG=3、颜色及位置选择温度均为 0 的四步画布与最终 RGBA；中间 MASK=512 |
| `default-step-{1..16}.u16`, `default.png` | `--default-sample` 输出官方默认 sampler 的 CPU seed=0 轨迹 |
| `timings.json` | 文本编码与采样的墙钟时间；四步记录包含 trace 写入 |

二进制片段采用 little endian 与 PyTorch 连续行序。离散 ID、张量形状精确检查；浮点中间值和有损权重使用 RMSE / cosine 等误差指标。[实测矩阵](validation.md)

## 真实入口

```sh
cargo build --locked --profile release-fast --bins --features vulkan
"$PY" tools/oracle/tessera/endpoints.py "$ASSETS" target/release-fast \
  "$ASSETS/parity/endpoints" --formats f32 --policies cpu
"$PY" tools/oracle/tessera/inpaint.py "$ASSETS" target/release-fast/rust-model-inference \
  "$ASSETS/parity/inpaint"
```

`endpoints.py` 会启动真实服务，请求 `/v1/images/generations`，逐像素比较 CLI、HTTP 和同档 PyTorch。`inpaint.py` 调用作者的 `TesseraPipeline.inpaint`，另用相同 RGBA 画布和采样设置运行 CLI，分别统计保留及重新生成的像素。

以下命令覆盖九种权重与 CPU、Auto、严格 Vulkan 的局部重绘。每档从原始权重的 16 步示例图抽出相同四个位置，用本档反量化 PyTorch 权重独立重绘一步。每行分别统计 252 个保留像素、4 个重绘像素、CLI / HTTP 一致数和各入口的 CPU / GPU 去噪器投影次数。保留像素变化或同一后端的两个入口不一致会令命令失败；同档浮点参考的重绘像素差异按实测值报告。

```sh
"$PY" tools/oracle/tessera/endpoints.py "$ASSETS" target/release-fast \
  "$ASSETS/parity/inpaint-endpoints" --inpaint > "$ASSETS/parity/inpaint-matrix.jsonl"
```

九种格式的 CLIP 和去噪器误差、设备分工与 16 步随机采样结果见[设备矩阵](device.md)和[默认 16 步矩阵](full_steps.md)。

## 批量生成和逐步预览

CLI 用同一提示词生成多张图。下例在每个采样步为两张图各写一张 RGBA 预览，最终输出 `sprite-000.png` 和 `sprite-001.png`。省略 `--num-samples` 时最终文件仍为 `sprite.png`。预览文件名中的步号和图像序号均从 0 开始。

```sh
target/release-fast/rust-model-inference --model "$ASSETS/tessera-small-f16.gguf" \
  --prompt 'golden crown with blue gems' --num-samples 2 --seed 17 \
  --out sprite.png --preview-dir previews --compute vulkan
```

HTTP 同时接受单个提示词和提示词数组。`n` 是每个提示词的图数，单次请求最多 16 张，`data` 按提示词顺序排列。批次共享同一个随机数流，每一步先为整批抽颜色，再为整批抽位置。非流式请求直接返回最终 PNG。

```sh
curl -N http://127.0.0.1:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":["golden crown","red apple"],"n":1,"seed":17,"stream":true}'
```

`image.preview` 事件包含 `step`、`total_steps`、`phase`、图像序号 `index` 和 base64 PNG。预览显示该步模型对每个像素的最高概率颜色，最终图保留按置信度逐步提交的采样颜色。全部掩码提前填满时会跳过余下的 decode 步。启用 polish 后，后续事件的 `phase` 为 `polish`。最后一个 `image.completed` 事件包含与非流式接口相同结构的 `data` 数组。流开始后若计算失败，会收到 `error` 事件。

以下脚本比较同提示词双图和不同提示词双图的预览与最终结果，同时检查断开 SSE 后计算槽可再次使用。

```sh
"$PY" tools/oracle/tessera/batch.py "$ASSETS" "$ASSETS/parity/batch" --reference
for mode in cpu auto vulkan; do
  "$PY" tools/oracle/tessera/batch.py "$ASSETS" "$ASSETS/parity/batch" \
    --bin-dir target/release-fast --policy "$mode" || exit 1
done
```
