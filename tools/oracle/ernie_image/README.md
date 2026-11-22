# ERNIE-Image CPU 对齐与运行

固定参考：stable-diffusion.cpp `3f8527a46c54ecf4cb4ed6003da8e8982283c73c`，
ggml `89c4413f5da6fb20cc796f16033d37f129be81fd`。
插桩只复制张量、记录检查点，不修改参考算子的数值计算；构建脚本在仓库 `target/` 中创建独立副本。

## 权重

| 组件 | 来源 / 文件 | SHA256 |
| --- | --- | --- |
| DiT | ModelScope `unsloth/ERNIE-Image-GGUF` / `ernie-image-Q4_K_M.gguf` | `ed43d36ab45df0ef24e55d88e72bea241d0d54e4006ce0442386aba071b04c2a` |
| 文本 | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8` |
| 原始 VAE | ModelScope `Comfy-Org/ERNIE-Image` / `vae/flux2-vae.safetensors` | `d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5` |
| 转换 VAE | `flux2-vae.gguf` | `8355684003ce8727605cdd5285038e73c82d8521d62df7680d262bf0caaf47fb` |

DiT 有 409 个张量，36 层、4096 hidden、32×128 heads、FFN 12288，混合 BF16/Q4_K/Q5_K/Q6_K。
发布文件错标 `general.architecture=wan`，运行入口按 ERNIE 张量契约识别并校验。
文本编码器用原始 prompt + BOS，取 `hidden_states[-2]`：25 个 blocks 的输出，跳过最后一层和最终 RMSNorm。

VAE 导出 decoder/post_quant_conv 共 140 个张量，Conv 权重 F16、Attention 四个 Linear 和向量 F32，
与参考实现实际加载精度一致。128 个 packed channels 先按固定 mean/std 反归一化，再 pixel shuffle 到
32 channels、空间×2，随后 post_quant_conv 和共享 Flux decoder；DiT latent 分辨率是图片的 /16。

```bash
.venv/bin/python -m tools.converter.ernie_image.convert_vae \
  models/ERNIE-Image/vae/flux2-vae.safetensors models/ERNIE-Image/vae/flux2-vae.gguf
```

## 逐位复现

下面命令从仓库根运行；目录必须是本次新建的目录，避免混入旧检查点。
已验证的标量环境是 macOS arm64、单线程，双方禁用 SIMD/FMA/权重重排/算子融合和外部加速库。
ERNIE 默认噪声使用参考的 `--rng cpu` MT19937；参考的默认 `cuda` RNG 会产生不同输入。

```bash
oracle=$(bash tools/oracle/ernie_image/build_oracle.sh /path/to/stable-diffusion.cpp)
mkdir target/ernie-oracle-run
GGML_CPU_DISABLE_FUSION=1 RMI_ORACLE_TRACE="$PWD/target/ernie-oracle-run" "$oracle" \
  --diffusion-model models/ERNIE-Image-GGUF/ernie-image-Q4_K_M.gguf \
  --llm models/Ministral-3-3B-Instruct-2512-GGUF/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf \
  --vae models/ERNIE-Image/vae/flux2-vae.safetensors \
  -p 'a lovely cat' -W 64 -H 64 --steps 1 --cfg-scale 1 --seed 42 -t 1 \
  --rng cpu --sampling-method euler --scheduler discrete -o target/ernie-oracle-run.png

cargo build --profile release-fast --features parity-trace,vulkan --bin rust-model-inference
RMI_SCALAR=1 RMI_PARITY_TRACE="$PWD/target/ernie-rust-run.jsonl" \
  target/release-fast/rust-model-inference \
  --model models/ERNIE-Image-GGUF/ernie-image-Q4_K_M.gguf \
  --text-encoder models/Ministral-3-3B-Instruct-2512-GGUF/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf \
  --vae models/ERNIE-Image/vae/flux2-vae.gguf --prompt 'a lovely cat' \
  --resolution 64 --steps 1 --cfg-scale 1 --seed 42 --threads 1 --out target/ernie-rust-run.png
.venv/bin/python tools/oracle/ernie_image/compare.py target/ernie-rust-run.jsonl target/ernie-oracle-run
```

1 步、CFG 1：token IDs `[1,1097,45988,7990]`，93 个检查点、6,440,960 个 F32 值原始 u32 位一致。
覆盖 25 层文本特征、噪声、时间/调制、36 个 DiT blocks、velocity、Euler latent 和完整 VAE RGB F32。
比较器检查每次 forward 的完整层集合及 buffer 长度；VAE Linear 的 pixel-major 与 CHW 仅作排列转换。
另一次 64×64、2 步、CFG 5 对照：条件/空提示词两路 token IDs、全部文本/DiT 输出、两步 Euler latent
和完整 VAE RGB 共 261 个检查点、16,132,096 个 F32 原始位一致。复现时在双方命令中同时改为
`--steps 2 --cfg-scale 5`，使用新的 trace 和输出目录。
GGML 会覆盖最终文本节点的名字，因此第 25 层用 DiT context 输入检查点比对。
PNG 的整数化和元数据不作为 F32 对齐证据。

单组件故障定位测试需要真实权重和上述 fixtures；环境变量为
`RMI_ERNIE_ORACLE_FIXTURES`、`RMI_ERNIE_IMAGE_DIT_GGUF`、
`RMI_MINISTRAL_3_3B_INSTRUCT_Q4_K_MODEL`、`RMI_ERNIE_VAE_GGUF`。
运行 `cargo test --profile release-fast --features parity-trace,vulkan --lib oracle_vae_fixture -- --ignored --nocapture`；
另外两项分别过滤 `oracle_flow_fixture` 和 `oracle_text_fixture`。

## 远端使用

目标目录 `/root/rust-model-inference`，权重在 `models/` 根目录；CPU 实际 cgroup 配额 16 核。
离线依赖已放在 `target/ernie-vendor`，无系统 Python/apt/shell 配置改动。

```bash
cd /root/rust-model-inference
CARGO_HOME="$PWD/target/cargo" RUSTUP_HOME=/usr/local/rustup \
RUSTUP_TOOLCHAIN=stable PATH=/usr/local/cargo/bin:$PATH \
cargo build --offline --locked --profile release-fast --features parity-trace \
  --bin rust-model-inference \
  --config 'source.crates-io.replace-with="vendored-sources"' \
  --config 'source.vendored-sources.directory="target/ernie-vendor"'
target/release-fast/rust-model-inference \
  --model models/ernie-image-Q4_K_M.gguf \
  --text-encoder models/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf \
  --vae models/flux2-vae.gguf --prompt 'a lovely cat' \
  --resolution 256 --steps 32 --cfg-scale 5 --seed 42 --threads 16 --out target/ernie.png
```

普通 ERNIE 默认 32 步 / CFG 5；文件名包含 Turbo 时默认 8 步 / CFG 1。
本次远端 EPYC 9334、容器 16 核配额、503 GiB 主机内存，256×256 / 32 步 / CFG 5 / seed 42 / 16 threads，
Q4_K_M DiT/Text、VAE Conv F16 / Linear F32，新进程、未清理 OS 页缓存的一次运行：
文本编码 0.331 s、去噪 1023.162 s、VAE 3.848 s、CLI 总耗时 1027.489 s。
提示词为 `a lovely cat, detailed fur, sitting on a wooden table, soft daylight, photography`，
产物 `target/ernie-remote-256-final.png` 显示桌面与窗边场景，但没有猫；只能证明生成链路运行成功，质量验证未通过。

## HTTP 入口

服务端已接入 `POST /v1/images/generations`（与 P-JiT 共用路由，模型按权重签名判定：
发布的导出把 `general.architecture` 误标成 `wan`，所以按张量名
`layers.0.self_attention.to_q.weight` + `text_proj.weight` 识别）。

```bash
target/release/rust-model-inference --serve --host 127.0.0.1 --port 8099 \
  --model models/ERNIE-Image-GGUF/ernie-image-Q8_0.gguf \
  --text-encoder models/Ministral-3-3B-Instruct-2512-GGUF/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf \
  --vae models/ERNIE-Image/vae/flux2-vae.gguf

curl -s -X POST http://127.0.0.1:8099/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a red panda on a rock","size":"128x128","steps":2,"seed":42}'
```

请求字段：`prompt`（必填）、`size` 或 `width`/`height`（方形）、`steps`、`seed`、
`guidance`（映射到 CFG scale，必须为正）、`n`（只接受 1）、`response_format`（只接受 `b64_json`）。
返回 `{"created":…,"data":[{"b64_json":…}]}`。

实测（aarch64 Vulkan 主机，release 构建，128×128 / 2 步 / seed 42）：
Q8_0 DiT 与 Q4_K_M DiT 两档，HTTP 返回的 PNG 与同一权重的 CLI 输出**逐位相同**
（Q8_0 md5 `9be3e9f9809f98ae96b4458af32d3856`，Q4_K_M md5 `5e2c2ab771690a89dc54f06ee75dbe95`）。
K-quant 档在服务端同样归到 CPU（打印 `compute scope=ernie_image backend=Cpu`），
与 CLI 的行为一致 —— 同一份权重不会因为入口不同而给出不同的图。

目前是 CPU 实验支持，`--gpu` 明确报错。Turbo 独立权重、其他量化、SIMD/FMA 数值对齐和 512/1024 图片质量未验证。
相关单测与真实契约通过；共享 Z-Image 回归为 84 passed / 7 failed / 10 ignored，7 个失败均在 main 基线复现。
基线另有一个 VAE 卷积失败，被本次清零 padding 修复。完整仓库测试未运行。
