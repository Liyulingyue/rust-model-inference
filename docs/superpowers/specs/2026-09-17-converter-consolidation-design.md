# Converter 代码归档到 `tools/converter`

## 目标和边界

在现有 `codex/cleanup-converter-duplicates` 分支及 Draft PR #73 上，把仓库中位于 `tools/converter` 外的 GGUF 转换实现、其单元测试和紧邻的必需数据文件集中到 `tools/converter`。这次只变更文件位置、命令路径和导入路径；不合并不同的导出算法，也不改变 GGUF 元数据、张量、量化字节或输出文件名。

旧转换器路径不保留薄包装、符号链接或兼容模块。Oracle、trace、构建脚本及模型推理代码不搬迁；它们若导入转换器，仅修正导入路径。已有未跟踪的 `.codex/` 不参与提交。

## 文件归属

| 当前路径 | 新路径 | 说明 |
|---|---|---|
| `tools/breeze/convert_breeze.py` 及测试 | `tools/converter/breeze/convert_breeze_plain.py` 及 `test_convert_breeze_plain.py` | 保留原来的未量化导出，与现有扩展量化版并存 |
| `tools/dots/convert_dots_tts.py` 及测试 | `tools/converter/dots/` 同名文件 | dots writer 和 Q8 语义原样保留 |
| `tools/vibevoice/convert_vibevoice_asr.py` 及测试 | `tools/converter/vibevoice/convert_vibevoice_asr_original.py` 及 `test_convert_vibevoice_asr_original.py` | 与现有扩展版并存，不假定两者字节等价 |
| `tools/dreamx/convert_dreamx_creator.py` 及测试 | `tools/converter/dreamx/` 同名文件 | 不搬迁 DreamX Oracle |
| `tools/neohorse/convert_neohorse.py` 及测试 | `tools/converter/neohorse/` 同名文件 | 保留 llama.cpp pin 和 NFC 契约 |
| `tools/qwen_drive/convert_qwen_drive.py`、测试、`source-tensors.json` | `tools/converter/qwen_drive/` 同名文件 | manifest 与调用它的脚本同目录 |

现有 `tools/converter/breeze`、`tools/converter/vibevoice` 的扩展转换器与测试保留当前名称。原 `tools/<model>/` 中的 README、Oracle、trace 和非转换器资产保留原位。两套 Breeze/VibeVoice 是否最终合并不属于本次搬迁。

## 调用和导入

全仓更新可执行命令、README、使用文档、测试导入、Oracle 导入及脚本内 `__file__` 相对路径。统一用 `tools.converter.<model>` 引用包内模块，避免同一文件同时以顶层模块和包模块加载；直接执行新路径的 CLI 时仅做必需的仓库根目录引导。Qwen-Drive 继续通过脚本相邻的 `source-tensors.json` 加载 manifest。无任何旧转换器路径仍作为可执行命令或有效导入留下；历史说明若提及旧路径，改为指向新文件。

## 验证

1. 在搬迁前固定当前分支的测试结果、CLI 命令和小型受控导出字节摘要；搬迁后重跑六组原版与两组扩展版转换器测试、`--help`、Oracle 导入检查、`git diff --check`，并扫描旧路径引用。
2. 用相同的受控输入比较搬迁前后的完整 GGUF 文件 SHA-256，覆盖 BF16/F16/F32/Q8/Q4 等各实现实际支持的路径；不把“单测通过”当作字节一致证据。
3. `/Users/gouzi/Documents/git/rust-model-inference/models` 只作为模型输入，绝不覆盖其中现有 GGUF。该卷当前约余 16 GiB；对可容纳的模型按空间预算在临时目录顺序验证并比较摘要，对 DreamX 等无法安全容纳完整输出的模型使用测试夹具、只读验证命令和静态路径检查，明确记录未做的真实导出。
4. 仅暂存本次迁移文件；验证后将后续提交推送到同一分支，并回读 PR #73 的实际文件列表、基线和 CI 状态。

## 成功条件

六组外部转换实现及其测试都已移入 `tools/converter`，原脚本路径不再存在；Oracle/trace 保留在原位且仍可使用。相关测试与 CLI 检查通过，受控导出逐字节一致，任何未验证的真实模型导出均明确标注，不声称全量模型字节一致。
