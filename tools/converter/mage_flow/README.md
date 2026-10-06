# Mage-Flow 无损 BF16 导出

六个 Microsoft Mage-Flow 型号共用 NR-MMDiT 张量契约，分别使用 `--variant base|flow|turbo|edit-base|edit|edit-turbo`。`--vae` 导出公共的确定性 MageVAE。

```sh
.venv/bin/python tools/converter/mage_flow/convert_mage_flow.py \
  models/Mage-Flow-Turbo --variant turbo
.venv/bin/python tools/converter/mage_flow/convert_mage_flow.py \
  models/Mage-Flow-Turbo --vae
.venv/bin/python -m unittest tools.converter.mage_flow.test_convert_mage_flow
```

转换保留源 BF16 字节，检查 config、张量名称/shape/dtype 和完整 payload；不覆盖已有文件。文本和 vision 使用固定 llama.cpp 转换脚本。模型下载、组件导出、专用 `mage-flow` CLI、哈希及标量逐位对齐见 [Oracle 说明](../../oracle/mage_flow/README.md)。
