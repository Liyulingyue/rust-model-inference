---
name: model-download
description: Use when 用户要求下载模型/权重（GGUF、mmproj 等）到 rust-model-inference 的 models 目录时。规定 ModelScope 优先、models/.venv 环境准备与按文件下载的命令模式。
---

# 模型下载

## 核心原则

- **ModelScope 优先**：默认从 ModelScope（`modelscope download`）下载，不要默认走 HuggingFace。仅当用户明确要求或仓库在 ModelScope 上不存在时才换源，且换源前先和用户确认。
- **按需下载文件，不整仓克隆**：只列出需要的文件名（权重 gguf、mmproj、README 等），避免下载整个仓库的冗余大文件。
- **下载位置固定**：所有模型放在仓库根的 `models/` 下，每个 repo 一个从属于 `models` 的子目录（`--local_dir ./<repo-name>`）。

## 环境准备（每次下载前检查）

1. 确保 `models` 目录存在：

   ```bash
   mkdir -p models
   ```

2. 在 `models` 目录下创建并复用 `.venv`，安装 modelscope（已存在则跳过）：

   ```bash
   [ -x models/.venv/bin/modelscope ] || (python3 -m venv models/.venv && models/.venv/bin/pip install modelscope)
   ```

   - 之后直接用 `models/.venv/bin/modelscope` 调用，无需 source activate。
   - 网络慢时可给 pip 加国内镜像，如 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

## 下载命令模式

工作目录切到 `models/`，按下面的模板拼命令：

```bash
modelscope download --model {repo} {necessary files...} --local_dir {models 下的子目录}
```

实例（在 `models/` 目录下执行）：

```bash
../models/.venv/bin/modelscope download --model unsloth/Qwen3.5-0.8B-GGUF \
    Qwen3.5-0.8B-Q8_0.gguf mmproj-F16.gguf README.md \
    --local_dir ./Qwen3.5-0.8B-GGUF
```

- `--model`：ModelScope 上的 repo id。
- 中间的位置参数：**必要文件列表**，从 repo 的文件列表里挑出本次任务需要的（如主权重 gguf、`mmproj-*.gguf`、`README.md`/`config.json` 等），不要省略成整仓下载。
- `--local_dir`：`models` 下从属的子目录，习惯上用 repo 名（如 `./Qwen3.5-0.8B-GGUF`）。

## 下载后验证

```bash
ls -lh models/<repo-name>/
```

- 确认目标文件存在且体积合理（gguf 主权重通常是 GB 级；若只有几 KB，多半是 LFS 指针/断点文件，需要重下）。
- 分片权重（`*-00001-of-000XX.gguf`）要把分片全部列进命令，缺一片后续加载会失败。

## 反模式

- 不要 `modelscope download --model {repo}` 不带文件列表——会整仓下载。
- 不要把模型下到 `models/` 之外的散落路径，或把 `--local_dir` 写成仓库根目录。
- 不要往系统 Python 里 `pip install modelscope`——一律用 `models/.venv`。
- 不要静默改用 HuggingFace 源——ModelScope 找不到 repo 时先问用户。
