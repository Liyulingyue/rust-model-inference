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
输出两份配套文件：

- `Audio8-TTS-Realistic-Air-BF16.gguf`：226 个张量保留原始 BF16 payload，内嵌原始 tokenizer 和配置。
- `Audio8-TTS-Realistic-Air-codec-F32.gguf`：455 个 F32 张量，使用 PyTorch 的 weight-norm kernel 展开 codec 权重。

转换器拒绝不匹配的主模型配置、张量清单、形状和类型，逐张量检查打包后的 payload，
并用两个原始权重文件的 SHA-256 生成共同的 bundle ID。两个组件必须一起导出和使用。
输出文件已存在时拒绝覆盖；可以用 `--out-dir` 指向新目录。

当前仅导出这个 BF16 主模型 / F32 codec 组合；未提供 Q4/Q8 等量化导出。
运行命令、官方对齐过程和验证边界见 [Oracle 说明](../../oracle/arktts/README.md)。
