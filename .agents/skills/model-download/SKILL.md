---
name: model-download
description: Use when 用户要求下载模型/权重（GGUF、mmproj 等）到 rust-model-inference 的 models 目录时。规定 ModelScope 优先、models/.venv 环境准备、按文件下载的命令模式，以及浏览/枚举 ModelScope 仓库的命令（无需登录）。
---

# 模型下载

## 核心原则

- **ModelScope 优先**：默认从 ModelScope（`modelscope download`）下载，不要默认走 HuggingFace。仅当用户明确要求或仓库在 ModelScope 上不存在时才换源，且换源前先和用户确认。
- **按需下载文件，不整仓克隆**：只列出需要的文件名（权重 gguf、mmproj、README 等），避免下载整个仓库的冗余大文件。
- **下载位置固定**：所有模型放在仓库根的 `models/` 下，每个 repo 一个从属于 `models` 的子目录（`--local_dir ./<repo-name>`）。

## 浏览 / 枚举（找可适配的新模型）

ModelScope 网页是 SPA，curl 抓不到列表；`POST /api/v1/models` 又要登录（`user not logged in`）。直接用 CLI 的 `list` / `info`，**两个命令都无需登录**，从 `models/` 工作目录执行：

```bash
# 列出一个 org / 个人名下所有 repo（按 repo-type 过滤）
../models/.venv/bin/modelscope list \
    --repo-type model --owner fastino \
    --page 1 --page-size 50        # 或者 --all 一次性拿全

# 查看单个 repo 的元数据（license / downloads / tags / 描述）
../models/.venv/bin/modelscope info --repo-type model fastino/GLiNER2.5-Decide
```

- `--repo-type` 必填：`model` / `dataset` / `studio` / `skill` / `mcp`。
- `--owner` 是 org id 或用户 id（不是 repo id）。
- `--all` 会自动翻页到末尾，省心但量大时慢；用 `--page N --page-size M` 翻页可控。
- `info` 返回的 `tags` 字段包含 `custom_tag:*`（如 `custom_tag:gliner2`、`custom_tag:qwen3`），用来快速判断架构族。

输出列：`repo_id / visibility / downloads / likes / license`——`list` 末尾会打印 `page X / total Y`。

### 反模式

- 不要 `curl https://www.modelscope.cn/<org>?tab=model`——SPA 没 JS 拿不到列表。
- 不要 `curl -X POST /api/v1/models`——需要登录。
- 不要假设某个 repo 在 ModelScope 上——用 `list --owner` 先确认存在再下，缺源时跟用户确认是否换 HF。

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
