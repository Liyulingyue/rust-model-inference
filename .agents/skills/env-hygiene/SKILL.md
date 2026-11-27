---
name: env-hygiene
description: Use when 在 rust-model-inference 仓库调试/验证时，需要安装 Python/Rust/系统工具依赖，或要修改 shell rc、环境变量等。规定所有依赖必须落在仓库内的 .venv / target，绝不污染系统 Python、全局 cargo install、apt 或 shell rc。
---

# 环境卫生：调试不污染系统

## 核心原则

- 所有依赖必须落在**仓库内**，不写系统 Python、不 `apt install`、不动 `~/.bashrc`/`~/.profile`/任何 shell rc。
- 仓库自带两个 venv，职责分明：
  - **仓库根 `.venv/`**：通用 Python 工具链（openai SDK、requests、numpy、pytest 等）。
  - **`models/.venv/`**：仅服务于模型下载链路（modelscope、huggingface_hub 等）。
- Rust 工具链一律走 `cargo`（依赖进 `Cargo.toml` / `Cargo.lock`），不 `cargo install` 到 `~/.cargo/bin`。
- 系统级包管理（`apt` / `dnf` / `brew` / `pip --break-system-packages`）**默认禁用**，除非用户当场明确授权。

## 选择哪个 .venv

| 用途 | 用哪个 venv |
| --- | --- |
| 模型下载（ModelScope / HF） | `models/.venv/` |
| 通用 Python 客户端脚本（openai SDK、requests） | 仓库根 `.venv/` |
| 推理 / 服务相关的 Python 辅助脚本 | 仓库根 `.venv/` |
| 跟 `models/.venv` 已有依赖冲突时 | 仓库根 `.venv/` |

不确定时优先**仓库根 `.venv`**——它是仓库级 venv；`models/.venv` 只服务于下载链路。

## 装依赖的正确写法

```bash
# 仓库根 .venv 缺包时
./.venv/bin/python -m pip install <pkg>
# 或带幂等检查
[ -x ./.venv/bin/<pkg> ] || ./.venv/bin/pip install <pkg>

# models/.venv 缺包时
./models/.venv/bin/pip install <pkg>

# 不要 source activate，直接用绝对路径解释器，避免子 shell 失效
```

## 跑脚本的正确写法

```bash
./.venv/bin/python ./script.py
./models/.venv/bin/modelscope download --model X --local_dir ./models/X
```

## Rust 依赖

- 加 Rust 依赖必须改 `Cargo.toml`（或 `cargo add` 写 `Cargo.lock`），不要 `cargo install --force` 把二进制塞到 `~/.cargo/bin`。
- 调试时 `--profile release-fast` 已经够用，不要每次都 `--release`（见 `fast-profile-iteration` skill）。
- 临时工具（如 `cargo-watch`、`cargo-edit`、`mdbook`）如果要装，装到 `./.cargo/bin`，然后在当前命令前缀 `PATH="$(pwd)/.cargo/bin:$PATH"`，**不要写 shell rc**。

## 系统级操作的硬红线

除非用户**当场**明确授权，下列动作**一律不做**：

- `pip install --break-system-packages` / 任何写到系统 site-packages 的 `pip install`
- `sudo apt install` / `sudo dnf install` / `brew install`
- 改 `/etc/*` 任何配置
- 改 `~/.bashrc` / `~/.profile` / `~/.zshrc` / `~/.config/fish/config.fish`
- `cargo install` 不带 `--root`（会落到 `~/.cargo/bin`）
- `npm install -g` / `pnpm add -g` / `yarn global add`
- `go install` 不带 `GOBIN`
- 修改 git config（`git config --global`）
- 持久化 `export PATH=...` / `export PYTHONPATH=...` 到任何 rc 文件

## 反模式

- `pip install <pkg>` 直接跑：当前 shell 默认激活 `models/.venv`，会把包装到错误位置。**永远用绝对路径 `./.venv/bin/python -m pip install`**。
- `python3 -m pip install --user <pkg>`：在 venv 里会失败；解 venv 后会污染 `~/.local`——同样禁用。
- `apt install python3-<pkg>`：污染系统且版本老，禁用。
- `cargo install some-binary`：会落到 `~/.cargo/bin` 污染全局 PATH → 改用 vendored 依赖或 `./.cargo/bin` + 前缀 PATH。
- 写 `export PATH=...` 到 `~/.bashrc` 持久化一个临时调试变量 → 用 `PATH=... command` 一次性前缀。
- 用 `which python3` 判断"系统 Python"：仓库 shell 通常已经激活了 `models/.venv`，`which python3` 看起来像系统 Python 实际不是。**永远用绝对路径 `./.venv/bin/python`**。

## 自检 checklist

调试完记得过一遍：

1. `git status` 看仓库内有没有不该有的目录（`venv/`、`env/` 应当不存在；`.venv/` / `models/.venv/` / `target/` 应当被 `.gitignore` 忽略）。
2. 在每个 venv 里跑 `./<venv>/bin/pip list`，确认里面只装了职责范围内的包——比如 `models/.venv` 里出现 `openai` 就说明装错地方了。
3. `ls ~/.cargo/bin` 是否多了临时工具——有就卸掉或迁到 `./.cargo/bin`。
4. `git diff ~/.bashrc`（如果存在）应当为空。