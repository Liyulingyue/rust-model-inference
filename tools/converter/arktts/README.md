# Audio8-TTS-Realistic-Air 转换

对应 ModelScope [`Edge0/Audio8-TTS-Realistic-Air`](https://modelscope.cn/models/Edge0/Audio8-TTS-Realistic-Air)
的 ArkTTS DualAR 发布版本。它与 `Audio8-ASR-Infinite` 的架构、tokenizer 和 codec 不同。

在仓库内 Python 虚拟环境安装 `torch==2.9.1`、`numpy` 后，从仓库根目录执行：

```sh
MODEL_DIR=/Users/gouzi/Documents/git/rust-model-inference/models/Audio8-TTS-Realistic-Air
.venv/bin/python -m tools.converter.arktts.convert_arktts "$MODEL_DIR"
.venv/bin/python -m unittest tools.converter.arktts.test_convert_arktts
```

输入为官方 `config.json`、`tokenizer.json`、`model.safetensors` 和 `codec.pth`。
输出两份配套文件。`--format` 和 `--codec-format` 支持 `f32`、`f16`、`bf16`、`q8_0`、
`q4_0`、`q4_1`、`q4_k`、`q5_k`、`q6_k`；默认仍是兼容性最好的 BF16 主模型 + F32 codec：

```sh
.venv/bin/python -m tools.converter.arktts.convert_arktts "$MODEL_DIR" \
  --out-dir "$MODEL_DIR/multiq/q4_k" --format q4_k --codec-format q4_k
```

主模型的 norm、bias 保留 BF16；二维 embedding 和 projection 按所选格式导出。
Q4/Q5/Q6_K 仅量化能整除块宽的 FFN down projection，896 宽的 attention/embedding
矩阵保留 BF16。codec 的卷积、norm、codebook
以及量化敏感的 `quantizer.upsample.*` / `quantizer.downsample.*` ConvNeXt 投影保留 F32，
其余二维 dense projection 按所选格式导出。这样每个配对文件仍是原生块量化格式，同时避免
重采样通道的激活量化误差被放大到波形。

Q4_1 还将 `quantizer.post_module.*` 解码 Transformer 保留为 F32：x86 上的 affine
activation QDQ 分叉曾使固定码流波形超过既定误差门槛。Q4_1 主模型和 codec 编码器仍量化；
codec 为 60 个 Q4_1 + 395 个 F32 张量，文件约 798 MB。此策略对所有平台和后端一致。

每个输出都写入相同的 `arktts.bundle_id`，并在发布前完成目录、shape、payload 和回读校验。
例如默认输出为：

- `Audio8-TTS-Realistic-Air-BF16.gguf`：226 个主模型张量，BF16。
- `Audio8-TTS-Realistic-Air-codec-F32.gguf`：455 个 codec 张量，F32，并使用 PyTorch 的
  weight-norm kernel 展开权重。

转换器拒绝不匹配的主模型配置、张量清单、形状和类型，逐张量检查打包后的 payload，
并用两个原始权重文件的 SHA-256 生成共同的 bundle ID。两个组件必须一起导出和使用。
输出文件已存在时拒绝覆盖；可以用 `--out-dir` 指向新目录。

转换器能导出量化格式不等于每个组合都已通过质量验收；CPU、Auto、严格 Vulkan 和 CLI/HTTP
的实测矩阵见 [Oracle 说明](../../oracle/arktts/README.md) 与
`/Users/gouzi/Documents/git/rust-model-inference/models/Audio8-TTS-Realistic-Air/multiq/`。
