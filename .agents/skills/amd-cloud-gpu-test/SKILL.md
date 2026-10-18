---
name: amd-cloud-gpu-test
description: Use 仅当且仅当用户**明确要求**在 AMD Radeon Cloud（48GB Radeon PRO W7900）上跑 rust-model-inference 的测试/benchmark 时。默认不在云端跑——本机或既有 Linux box 优先。云端机器**TCP 出网被墙**（apt / rustup / HF 都走不通），**默认 workflow 是「本机拉资源（Rust 离线包 / 模型权重）→ scp → 云端 `cargo build` + 跑」**，资源走本地、算力走远程；起步验证建议用 0.6B 量级的 GGUF。
---

# AMD Cloud GPU 远程测试

## 触发条件（重要）

**默认不要往 Radeon Cloud 跑测试。** 只有当用户明确说出诸如「在 AMD Cloud 上跑一下」「上云端跑」「在 W7900 上验证」等指令时才走本 skill。本地其它能用的机器（Linux box、已有 GPU 工作站）默认优先。

云端的隐性成本：

- 流量免费时长按墙钟计，编译 + 跑模型很烧券。
- 云端机器**没有公网出口/受限**：很多 apt 包、conda 包、HF 直连都失败。
- 杀进程/重启不会保留文件——重要产物（权重、二进制、结果）必须 `scp` 回本地。

## 硬件 & SSH 入口

- GPU：AMD Radeon PRO W7900，48GB VRAM（ROCm/HIP 后端）。
- SSH 别名（已配置在 `~/.ssh/config`）：`radeon-cloud`
- 主机端口：每次启动 Template 会变，**连接前先确认**。`~/.ssh/config` 里写的端口是最近一次的值，过期就连不上。

## 接入

下面这条命令在 PowerShell 里直接连，无需记 IP/port/user/key：

```powershell
ssh radeon-cloud
```

成功后进入 Linux Terminal，可直接 `rocm-smi` 看 W7900、`hipcc --version` 检查 ROCm 工具链。

连不上时先用 `ssh -v radeon-cloud` 区分三类故障：

- `Connection timed out` → Port 已过期，见下文《Port 轮换》。
- `Permission denied (publickey)` → 本机 `ssh-add` 一下 `id_ed25519_radeon_cloud`，或检查 `~/.ssh/config` 里 `IdentityFile` 路径。
- `REMOTE HOST IDENTIFICATION HAS CHANGED` → known_hosts 里旧指纹，按下文《Port 轮换》里 `ssh-keygen -R` 的步骤清掉。

### Port 轮换（Template 重启后）

`~/.ssh/config` 里只**改 Port**（HostName 是 `36.150.116.206` 固定），同时清掉 `known_hosts` 里旧的端口指纹。**助手默认不替用户改这两个文件**——env-hygiene 硬红线就是不动用户态配置，下面给的是人肉版命令，用户自己复制粘贴运行。

需要的三个动作：

1. 改 Port。先看清 `radeon-cloud` 这一段：

    ```powershell
    Get-Content "$HOME\.ssh\config" | Select-String -Pattern radeon-cloud -Context 5
    ```

    然后用 `sed` 原地替换 Port（`-i` 在 PowerShell 5.1 上不支持，统一走读-改-写）：

    ```powershell
    $cfg = "$HOME\.ssh\config"
    $newPort = Read-Host "新 Port"
    $text = Get-Content -LiteralPath $cfg -Raw
    $text = [regex]::Replace($text, '(?ms)(Host radeon-cloud\b.*?)Port\s+\d+', "`$1Port $newPort")
    Set-Content -LiteralPath $cfg -Value $text -Encoding UTF8
    ```

2. 清掉 known_hosts 里旧端口的指纹（OpenSSH 以 `[host]:port` 为键）：

    ```powershell
    $oldPort = Read-Host "旧 Port（要清的那个）"
    ssh-keygen -R "[radeon-cloud]:$oldPort"
    ```

3. 首次连走 `accept-new`，避免交互卡住：

    ```powershell
    ssh -o StrictHostKeyChecking=accept-new radeon-cloud
    ```

    之后无 flags 的 `ssh radeon-cloud` 就行。

已知坑：Windows 自带 `ssh-keyscan` 在这套 Radeon Cloud OpenSSH 上做不了 KEX 协商（`sntrup761x25519-sha512@openssh.com` 客户端不支持），所以「先 ssh-keyscan 预热 known_hosts」走不通——靠第 3 步的 `accept-new` 自然落地。`accept-new` 模式只会新增 fingerprint、不会覆盖；不同端口的 key 各管各的，安全。

### 全新配置（第一次起）

1. **Radeon Cloud 注册 & 领券**：`https://developer.amd.com.cn/login?source=WmhWQBvrH` → 我的权益 → 领取 → 兑换 → 在 Profile 的 Redeem Credit 粘贴兑换链。
2. **本机生成 SSH key 并贴到云端 Profile**：

    ```powershell
    New-Item -ItemType Directory -Force -Path "$HOME\.ssh" | Out-Null
    ssh-keygen -t ed25519 -C "amd-radeon-cloud" -f "$HOME\.ssh\id_ed25519_radeon_cloud"
    Get-Content "$HOME\.ssh\id_ed25519_radeon_cloud.pub" | Set-Clipboard
    ```

    把剪贴板里的公钥贴到 Radeon Cloud Profile → SSH Public Key。

3. **起 Template + 启动 SSH Server + 写别名**：
    - Profile → Add Template：填必填项，**开启 SSH Access**，Add → Launch。
    - 进 JupyterLab → Terminal，跑一键脚本：

        ```bash
        curl -fsSL https://modelscope.cn/models/AIwork4me/SSH-Radeon-Cloud/resolve/master/scripts/start-ssh-server.sh -o /tmp/start-ssh-server.sh && bash /tmp/start-ssh-server.sh
        ```

    - 本机跑：

        ```powershell
        irm https://modelscope.cn/models/AIwork4me/SSH-Radeon-Cloud/resolve/master/scripts/create-radeon-ssh-config.ps1 | iex
        ```

        按提示输入新的 IP/Port，之后 `ssh radeon-cloud` 即可。

## 环境约束（踩过的坑）

云端是个干净镜像，**别指望 `pip install` / `apt install` 都能跑通**：

- 没装 HIP/ROCm 完整 SDK——直接 `pip install torch` 不会带 ROCm wheel，要走 AMD 官方仓库；好在项目本身就走 Rust + 自带 ROCm bindings，这一步不踩。
- `cmake`/`gcc`/`pkg-config` 通常够用，但版本老——`cc-rs` / `cmake-rs` 跑 cargo build 时若报「找不到 gcc」，先 `which gcc && gcc --version`。
- `pip` 装大包（torch、transformers、vllm）经常超时或被限流——**别走这条路**。Rust 项目里就算有 Python 胶水，也得提前在本机 venv 里编好或干脆纯 Rust 实现。
- HuggingFace 默认走不到——`modelscope download` 也常失败。模型权重必须在本机先拉好，再 `scp -r` 上云。
- 没有持久化存储——重启 Template 后 `/root` 和 `/opt` 都清空，每次都得重推 Rust 工具链和源码。

## 离线工具链（云端无出网时怎么装 Rust）

云端到 `static.rust-lang.org` / `archive.ubuntu.com` / `compute-artifactory.amd.com` 的 TCP 出口全死——DNS 能解析但 IPv4 超时、IPv6 立即 "Network unreachable"。`rustup-init.sh` 默认走 `sh.rustup.rs` 拉 toolchain，会卡在中途；`apt update` 全 W: 拿不到索引。绕开办法：**本机预下载官方静态 tarball → scp → 云端解压**。单份 stable ≈ 200 MiB 压缩、≈ 1.5 GiB 解压，跑得动。

### 1. 拿到当前 stable tarball 的 URL

官方 channel 文件动态给出稳定版元数据。**注意 SHA256 是该 tarball 的，不是 nightly 增量包的**。

```powershell
$tmp = Join-Path $env:TEMP 'channel-rust-stable.toml'
curl.exe -sSL -o "$tmp" https://static.rust-lang.org/dist/channel-rust-stable.toml
# 锁定 [pkg.rust.target.<target>] 这一块——xz_url 和 xz_hash 必须在同一 subtable
$rustHdr = Select-String -Path $tmp -Pattern '^\[pkg\.rust\.target\.x86_64-unknown-linux-gnu\]$' | Select-Object -First 1
if (-not $rustHdr) { throw "rust target subtable not found" }
$nextHdr = (Select-String -Path $tmp -Pattern '^\[' | Where-Object { $_.LineNumber -gt $rustHdr.LineNumber } | Select-Object -First 1)
$endLine = if ($nextHdr) { $nextHdr.LineNumber } else { $rustHdr.LineNumber + 8 }
$blockLines = Get-Content -LiteralPath $tmp | Select-Object -Skip ($rustHdr.LineNumber - 1) -First ($endLine - $rustHdr.LineNumber)
$url  = ($blockLines | Select-String -Pattern '^xz_url\s*=\s*".*rust-\d+\.\d+\.\d+-x86_64-unknown-linux-gnu\.tar\.xz"' | Select-Object -First 1).Line -replace '^[^"]+"','' -replace '"$',''
$hash = ($blockLines | Select-String -Pattern '^xz_hash\s*=\s*"[0-9a-f]{64}"'                    | Select-Object -First 1).Line -replace '^[^"]+"','' -replace '"$',''
"URL:  $url"
"Hash: $hash"
```

例如当前（2026-10）：

```
URL:  https://static.rust-lang.org/dist/2026-10-15/rust-1.99.1-x86_64-unknown-linux-gnu.tar.xz
Hash: 231fcae47ee950621374725b5fc3b5441d2fbda2955ac45dd7dc816d8876cfe7
```

> **坑**：早期版本用 `Select-String '^xz_hash=...' | Select-Object -First 1` 取的是 channel-stable.toml **最早**出现的 `xz_hash`——落在 `pkg.cargo.target.aarch64-apple-darwin`（这包根本没有上层 `bin/cargo` 自洽安装脚本，走错全套），跟 `xz_url` 不在同一 subtable，校验必然 fail。`xz_url = "..."` 和 `xz_hash = "..."` **必须**从同一个 `[pkg.rust.target.<target>]` sub-block 里取。

### 2. 本机下载 + 哈希校验 + scp

tarball 落到 opencode 的暂存目录（**别写 `$HOME\Downloads\`**：那是用户可见目录，env-hygiene 也不让动）：

```powershell
$tmpDir  = Join-Path $env:TEMP 'opencode\rust-bootstrap'
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
$ver     = ([regex]::Match($url, 'rust-(\d+\.\d+\.\d+)-')).Groups[1].Value
$tarball = Join-Path $tmpDir ("rust-" + $ver + "-x86_64-unknown-linux-gnu.tar.xz")
curl.exe -sSL -o "$tarball" $url
$got = (Get-FileHash $tarball -Algorithm SHA256).Hash.ToLower()
if ($got -ne $hash) { throw "SHA256 mismatch: got=$got expect=$hash" }
"OK sha256=$got"
scp $tarball radeon-cloud:/root/
```

### 3. 云端解压 + install + 加 PATH

```bash
ssh radeon-cloud
ver=1.99.1            # 改成你选的 stable 版本
tarball=/root/rust-$ver-x86_64-unknown-linux-gnu.tar.xz
# 解压产物是 rustup 分组件布局：cargo/, rustc/, rust-std-x86_64-unknown-linux-gnu/ ...
# 没有顶层 bin/！必须跑 install.sh 把 components 装到一个 prefix 下。
mkdir -p /opt
tar -xJf "$tarball" -C /opt/

# 用 install.sh 装到一个干净的 prefix（避免污染原解压目录）
prefix=/opt/rust-$ver-root
cd /opt/rust-$ver-x86_64-unknown-linux-gnu
./install.sh --prefix="$prefix" --without=rust-docs --disable-ldconfig
ls "$prefix/bin/"         # 应看到 cargo / rustc / rustfmt / cargo-clippy / clippy-driver / ...

export PATH="$prefix/bin:$PATH"
# 持久化（Template 内有效，重启即清）
line="export PATH=\"$prefix/bin:\$PATH\""
grep -F "$line" /root/.bashrc >/dev/null 2>&1 || echo "$line" >> /root/.bashrc
cargo --version     # 应打印 $ver
rustc --version
```

> **坑**：单纯 `tar -xJf` + `export PATH=.../rust-<ver>-x86_64-unknown-linux-gnu/bin:$PATH` 会**立刻报 `cargo: command not found`**——官方 tarball 无顶层 `bin/`。必须 `./install.sh --prefix=<dest>` 由它生成 `bin/`。`--disable-ldconfig` 防止云端 ldconfig 不可写时脚本报错；不缺 `--without=rust-docs` 的话光 docs 就是 200+ MiB，没必要。

> **prefix 命名约定**：解压原始目录 `rust-<ver>-x86_64-unknown-linux-gnu/`（分组件）、安装落地目录 `rust-<ver>-root/`（含 bin/）——两者**不要**重名，否则 `tar -xJf` 后再跑 `install.sh --prefix=同目录` 会把 components 和 aggregated bin/ 揉到一起，难看难删。

### 4. 配套系统库（缺什么再补）

云端镜像工具齐全：`xz tar git gcc make pkg-config cmake curl wget`（`pkg-config` 和 `cmake` 版本可能老，但 cargo build 跑基础 crate 没问题）。RUST 要的链接器、libssl、libz 等 Ubuntu 默认装了。**别试 `apt install`**——上游不通；要补库只能走本机编好 `dpkg -i` 或干脆在 `Cargo.lock` 里选纯 rust 实现。

### 持久化

Template 重启 `/root` 和 `/opt` 都清空。要每次起 Template 不用重新推，把解压步骤塞进 Template 的「启动脚本」里（前提是该 Template 配了持久卷把 `/opt` 落盘）—— 这是 Template 配置层面的活，本 skill 不管。

## 离线 build（云端 build 要 vendored crate 源）

云端装好 Rust 工具链只解决了「能编译」，没解决「crate 源从哪来」——`cargo build` 默认要拉 crates.io，云端 TCP 出网死，**必然 hang**。修法：**本机 `cargo vendor` 把 `Cargo.lock` 锁定的所有 crate 源码打到 `vendor/`，随项目源码一起 scp，云端 `--offline` build**。

### 1. 本机 `cargo vendor`

```bash
cd /path/to/rust-model-inference
cargo vendor vendor      # 不接参数就是 ./vendor，相对路径
```

- 会在仓库根建 `vendor/`（约 500 MiB / 20k 文件），`git status` 多一条 `?? vendor/`。
- `.cargo-checksum.json` 每个 crate 一份，cloud 端 `--offline` build 会校验 hash。

> **坑 1**：`cargo vendor` **从不自动**修改 `.cargo/config.toml`——它**只**在 stdout 打三行建议，必须自己追加。stdout 最后那几行：
>
> ```
> To use vendored sources, add this to your .cargo/config.toml for this project:
>
> [source.crates-io]
> replace-with = "vendored-sources"
>
> [source.vendored-sources]
> directory = "vendor"
> ```

### 2. 手动把三行追加到 `.cargo/config.toml`

```toml
# 原本已存在
[build]
rustflags = ["-C", "target-cpu=native"]

# 手动追加
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "vendor"
```

> **坑 2**：`directory = "vendor"` 是**相对路径**，对所有引用 `.cargo/config.toml` 的 cargo 调用生效——本机 build 会优先用 vendor，本地 crates.io 拉取就被绕开。如果你想「本机 build 走 crates.io，云端才走 vendor」，**不要**给项目 commit `.cargo/config.toml` 的 source 段；改用环境变量 `CARGO_HOME=/some/dir` + 在该 home 下的 `config.toml` 里放 source replacement。**最低侵入做法**：在云端 cargo build 之前用 `--config` CLI flag 临时注入，不动 `.cargo/config.toml`：
>
> ```bash
> cargo build --offline \
>   --config 'source.crates-io.replace-with="vendored-sources"' \
>   --config 'source.vendored-sources.directory="vendor"'
> ```

### 3. 云端 build

```bash
cd /root/rust-model-inference
CARGO_TARGET_DIR=/root/target-amd cargo build --offline \
  --profile release-fast --features vulkan --bin rust-model-inference
```

`--offline` 强制走 vendor/`；没有它，cargo 还是会去 `crates.io` 探一下再 hang。`CARGO_TARGET_DIR` 是 env（**不是** `--release-target-dir`，cargo 没那个 flag——见「推荐的工作流 §0」坑）。

### 4. 测完还原

```bash
# 本机：去掉 vendor 和 .cargo/config.toml 的 source 段
git checkout -- .cargo/config.toml
rm -rf vendor/

# 现在 git status 应该回到 cargo vendor 之前的状态
git status
```

---

## 推荐的工作流（rust-model-inference）

**原则**：资源走本地，算力走远程。云端负责 build + 跑，本机负责拉取所有网络资源（Rust 工具链、模型权重、crate 缓存）。

### 0. 本机预检（不实际 build 目标 binary）

本机**只做轻量编译/lint smoke test**，不期望产出最终 binary：

```bash
# 用 release-fast 跑一遍做"我的改动 OK 吗"的初筛；产物可以丢
CARGO_TARGET_DIR=target-fast-amd cargo build --profile release-fast --bin <some-small-bin>
CARGO_TARGET_DIR=target-fast-amd cargo clippy --profile release-fast -- -D warnings
CARGO_TARGET_DIR=target-fast-amd cargo test --profile release-fast --lib --no-run   # 类型检查
```

> **坑**：cargo **没有** `--release-target-dir` 这个 flag（早期 cargo 草稿里有过，后来删了）。设 target 目录只能走 `CARGO_TARGET_DIR=…` 环境变量。`cargo build --target-dir X` 在 nightly 上有但不等价。

> **真正在云端跑的 binary 是不在本机 build 的**——目标是 ROCm 路径，本机没 W7900 没法测；本机 smoke test 只是为了省一次云端迭代。

### 1. 拉资源 + scp 到云端

拉两种东西：(a) Rust 离线工具链（首次会话或升级时），(b) 项目源码 + 模型权重。

**(a) Rust 工具链**（每个 Template 都要重推一次，详见「离线工具链」节）：

```powershell
# 已落到 opencode 暂存目录的 tarball
$tmpDir  = Join-Path $env:TEMP 'opencode\rust-bootstrap'
$tarball = Get-ChildItem -LiteralPath $tmpDir -Filter 'rust-*-x86_64-unknown-linux-gnu.tar.xz' |
           Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
scp $tarball radeon-cloud:/root/
```

**(b) 项目源码**（每次改完都得推）：

最朴素的做法是把源码打成 tarball 再 scp，免去 `rsync` server 之类配置：

```powershell
# ⚠ tar / rsync 的 --exclude=target **贪婪**：会吃掉任何子目录里叫 target 的文件
# （例如 vendor/cc/src/target/，cloud 端 cargo --offline build 时直接失败）。
# 修法：在项目 **父目录** 跑，path 加 `rust-model-inference/` 前缀。
$srcTar = Join-Path $env:TEMP 'opencode\rust-bootstrap\rust-model-inference-src.tar.gz'
Set-Location (Split-Path -Parent (Get-Location))    # 上移到 F:\PythonCodes\
tar -czf $srcTar `
    --exclude=rust-model-inference/target `
    --exclude=rust-model-inference/target-fast-amd `
    --exclude=rust-model-inference/target-amd `
    --exclude=rust-model-inference/.git `
    --exclude=rust-model-inference/models `
    --exclude=rust-model-inference/references `
    --exclude=rust-model-inference/out `
    --exclude='*/.venv' `
    --exclude='*/__pycache__' `
    --exclude='*.rs.bk' `
    --exclude='*.swp' `
    --exclude='*.swo' `
    rust-model-inference
scp $srcTar radeon-cloud:/root/rust-model-inference-src.tar.gz
```

云端解压用 **in-place**（**不要** `rm -rf /root/rust-model-inference`，会把已经传好的 `models/` 也清掉）：

```bash
ssh radeon-cloud
# 仅在第一次或更换项目时建空目录
test -d /root/rust-model-inference || mkdir -p /root/rust-model-inference
# 直接 in-place 解到 /root/，tarball 里的 rust-model-inference/* 会落到 /root/rust-model-inference/*
tar -xzf /root/rust-model-inference-src.tar.gz -C /root/
# 验证关键文件：vendor/.cargo-checksum.json 完整 + models/ 仍在
test -f /root/rust-model-inference/vendor/cc/src/target/apple.rs && echo "vendor OK"
ls /root/rust-model-inference/models/Qwen3-0.6B-GGUF/*.gguf
```

本机若装有 `rsync`（Git Bash 自带），可替代成增量推送。**同样要把 path 加 `rust-model-inference/` 前缀**：

```bash
# 等价但增量。父目录跑，path 用 rust-model-inference/target 这种完整前缀
rsync -avz --delete \
  --exclude=rust-model-inference/target \
  --exclude=rust-model-inference/target-fast-amd \
  --exclude=rust-model-inference/target-amd \
  --exclude=rust-model-inference/.git \
  --exclude='rust-model-inference/models/*' \
  --exclude=rust-model-inference/references \
  rust-model-inference/ radeon-cloud:/root/rust-model-inference/
```

**(c) 模型权重**（一次推完就别动）：

```bash
# 本机先 modelscope download 到 models/<repo>/
# 然后整目录推上去
scp -r models/<repo>/ radeon-cloud:/root/rust-model-inference/models/
```

0.6B 起步建议：

- `unsloth/Qwen3-0.6B-GGUF` 或 `unsloth/Qwen3.5-0.8B-GGUF`——Q8_0/Q4_K_M 都行，体积小，验证推理链路足够。
- 选 `Q8_0` 比 `Q4_K_M` 更稳：权重误差小，分布式/算子差异更容易暴露真 bug 而不是量化误差。

### 2. 云端 build

```bash
ssh radeon-cloud
```

```bash
# 在云端——前提是 1.(a) 已经把 tarball 推过来了；
# 完整解压+install 流程见「离线工具链」§3，这里只列关键一句
ver=$(ls /root/rust-*-x86_64-unknown-linux-gnu.tar.xz | head -1 | grep -oP '(?<=rust-)\d+\.\d+\.\d+')
prefix=/opt/rust-$ver-root
test -x "$prefix/bin/cargo" || {
    tar -xJf /root/rust-$ver-x86_64-unknown-linux-gnu.tar.xz -C /opt/
    cd /opt/rust-$ver-x86_64-unknown-linux-gnu
    ./install.sh --prefix="$prefix" --without=rust-docs --disable-ldconfig
}
export PATH="$prefix/bin:$PATH"

# 进入项目
cd /root/rust-model-inference

# 真正在云端跑 build——release-fast，省 ~80% 时间
CARGO_TARGET_DIR=/root/target-amd cargo build --profile release-fast --bin <your-bin>

# 验证产物
ls -lh /root/target-amd/release-fast/<your-bin>
ldd /root/target-amd/release-fast/<your-bin>     # ash 是运行时 dlopen libvulkan，不显式链 vulkan 是正常的
```

`/root/target-amd/` 是云端的 build target dir；别用 `target-fast-amd/`（那是本机 smoke test 用的）。

### 3. 云端跑

```bash
# 在云端
rocm-smi                                # 看 W7900 上线、显存空闲
HIP_VISIBLE_DEVICES=0 /root/target-amd/release/<your-bin> \
  --model /root/rust-model-inference/models/<repo>/*.gguf \
  --prompt "..."
```

第一次跑先只跑 **tokenizer + 一次 forward**，确认：

- ROCm 后端 init 没 panic。
- KV cache 分配没炸（48GB 充裕但要看上下文长度）。
- `rocm-smi` 看显存确实有占用、kernel 真在 GPU 上跑（不是 ROCm fallback）。

### 4. 结果回拉

```bash
scp radeon-cloud:/root/<result-file> ./results/
```

云端可以留 `/root/target-amd/`（在同一会话内复用，省得二次 build）；别的缓存（`~/.cargo`、`/tmp/cargo-install`）别在云端留——每次 Template 重启都从零起。

## 调试要点

- `rocm-smi` 看不到卡 → SSH 进的是 CPU-only Template，重启带 GPU 的 Template。
- 云端 build 出来的二进制段错误 → 多半是 ROCm 驱动/HIP runtime 版本对不上：先 `hipcc --version` / `rocminfo`，再 `ldd <binary>` 看 `libamdhip64.so` 是否能解析。
- 显存看着没用满但 OOM → 检查 `HIP_VISIBLE_DEVICES` 是否设；以及别在云端和本机 fork 同一个进程组。
- `cargo build` 报「找不到 gcc / cc」 → 先 `which gcc && gcc --version`；云端 `cc-rs` 可能挂着老的 build-essential 版本够用就别动。
- ssh 上去发现 `id_ed25519_radeon_cloud` 没带过来 → 本机 `ssh-add` 一下，或 `~/.ssh/config` 里 `IdentityFile` 路径对不对。

## 反模式

- **不要默认把跟 ROCm/GPU 无关的纯 CPU 验证搬到 Radeon Cloud**——本地 Linux box + `release-fast` 已经够用，且不烧云端流量费。
- **`cargo build` 烧券的是「无限增量」（每次都重新拉 + 重编），不是 build 本身**。把源码 release-fast 编一次缓存到 `/root/target-amd/` 内复用的成本可控；反复 wipe 重 build 才是浪费。
- 不要把模型权重从 HF 直连拉到云端——先在本机 `modelscope download` 拉到 `models/`，再 `scp -r` 过去。
- 不要用 `dev`/`main` profile 编（dev = debug，编译慢 5–10×）——一律 `release-fast`，省 80% 时间。
- 不要在云端留 `~/.cargo` / `models/` 当「持久存储」——Template 重启即清空；`/root/target-amd/` 同会话可复用，跨会话别指望。
- **不要在没有用户授权的情况下，直接改动用户态文件**（`~/.ssh/config`、`~/.ssh/known_hosts`、`$PROFILE` 等）。Template 重启后看到 stale Port/指纹，**只给用户看命令**，让他们自己跑或明确点头后再动；助手自己 `Set-Content ~/.ssh/config` 或 `ssh-keygen -R` 不算经过同意。

## 跨会话沉淀（Z-Image 11.5 GB / 3 档验证后新增）

Z-Image-Turbo Q8_0 三档（CPU 512 / GPU 512 / GPU 1024）端到端跑通后新增 5 条坑，前面章节都未覆盖。

### 1. cloud 没装 `/usr/bin/time`，`set -e` 脚本秒退

Template 默认不带 `timeutils` 包，`/usr/bin/time -v` 不存在；`set -e` 下脚本立即 exit 0 行——3 档都没跑。

```bash
# 替代：bash builtin time（无 -v），或直接 date +%s%N 取差
start_ns=$(date +%s%N); "$BIN" "$@"; end_ns=$(date +%s%N)
echo "wall=$(( (end_ns - start_ns) / 1000000 ))ms"

# peak RSS 从 /proc/$PID/status 的 VmHWM 读（替代 /usr/bin/time -v 唯一给的那个）
```

### 2. 大文件 scp / 11 GB 级 tarball

- **`-C` 不要用**：GGUF / 权重里是随机浮点，compression 纯浪费 CPU。本网络实测 ~33 MB/s 上限，**11 GB ≈ 35 min wall**。Opencode shell `timeout` 上限给 2700000 ms（45 min）。
- **scp 静默半截失败**：第一次传到 5.5 GB 时 SSH wrapper timeout，cloud 残留 5.5 GB 错误文件——下次重传覆盖前**始终 SHA256 校验两端**：
  ```powershell
  # local:
  Get-FileHash <file> -Algorithm SHA256
  # cloud:
  ssh radeon-cloud 'sha256sum /root/<file>'
  ```
- **不想冒险**：本机 `split -b 2G big.tar chunks/chunk_` 后逐块 scp，cloud `cat chunks/chunk_* | tar -xf -`。
- **别用 `-r` scp 整目录大文件**——`scp -r` 是多次 stream 拼接，断点续传零支持。

### 3. cgroup `memory.max` ≠ `free -h` 显示的 host RAM

`free -h` 的 `total` 给的是 host 物理 RAM；**容器内实际可用看 `cat /sys/fs/cgroup/memory.max`**。本网络实测：503 GB host → 55 GB cgroup。

- **预算大模型推理前先 cat 一次**——55 GB 对 12B Q8_0 DiT（7 GB）有富裕，但 4 个并发任务 + 1024 latent attention 可能 OOM。
- `free -h` 的 `available` 列是 cgroup 视图（不是 host 剩余），看起来 444 GB free 实际**根本拿不到**。

```bash
ssh radeon-cloud 'cat /sys/fs/cgroup/memory.max; free -h | head -2'
```

### 4. cloud 进程存活判定——`pgrep` 不一定够

Z-Image 实际跑了 5 min，但 `pgrep -f run-zimage` 三次只匹配脚本 PID 没匹配 `rust-model-inference`（cmdline 没 "run-zimage"）。判定 binary 还在不在：

```bash
# 1. binary 自己的 cmdline + 当前 %CPU/RSS
ps -o pid,pcpu,rss,etime,cmd -p $(pgrep -f rust-model-inference | head -1)
# → %CPU=799% RSS=10.9GB 表示真在跑（8 核满载）

# 2. /proc/<pid>/status 看 State
cat /proc/<PID>/status | head -5   # R=running, S=sleeping on syscall

# 3. top 5 秒实时
top -bn 2 -d 2 -p <PID> | tail -10
```

PowerShell 端串行「string not match 'foo'」判定 ssh 输出「process 消失」**不可靠**——直接 `LASTEXITCODE` 或 `pgrep` 返回空数组。

### 5. CPU vs GPU backend 一致性验证（构图守恒）

任何「把 matmul 搬上 Vulkan」类的改动都必须用同 prompt + 同 seed 同时跑 CPU 和 GPU，肉眼对比是端到端 V&V 最便宜一步（不需要 Python 评测脚本）：

```bash
# 同 prompt / 同 seed / 同 thread 数
... --threads 8 --seed 42 --out cpu.png
... --threads 8 --seed 42 --out gpu.png --gpu

# md5 不一定相等（RNG 在 matmul 顺序上的差异会被 diffusion 放大），
# 但 主体 / 构图 / 配色 应该肉眼一致。Z-Image 在 512² / seed=42 下肉眼一致 ✓
```

如果 CPU vs GPU 主体错位（主体对了但动作/朝向偏差大），多半是 DiT matmul 顺序在 attention / RoPE / adaln 一处悄悄引入 race。同 seed + 同 thread 数是基本盘。
