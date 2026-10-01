# Z-Image Turbo 用法

Z-Image Turbo 是文生图模型，分三个独立 GGUF 组件：

- DiT：`--model`
- Qwen3 文本编码器：`--text-encoder`
- Flux VAE：`--vae`

GGUF `general.architecture = pig`，对应 `src/models/diffusion/pig.rs` 与
`src/models/diffusion/z_image/`。CLI 入口 `src/app/diffusion.rs::run_z_image_cli`。

> 共用前置：构建 `cargo build --release --bin rust-model-inference`。
> 当前**仅 CPU**，**仅 512×512**，**仅 txt2img**。
> img2img、GPU、Z-Image Base 列为 `Unsupported`（SUPPORTED_MODELS.md）。

## 1. 三组件

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/z-image-gguf/z-image-turbo-q8_0.gguf \
  --text-encoder models/z-image-gguf/qwen3_4b_f32-q8_0.gguf \
  --vae models/z-image-gguf/pig_flux_vae_fp32-f16.gguf \
  --prompt "A red fox sleeping beneath a pine tree" \
  --steps 8 --resolution 512 --seed 42 --threads 1 --out fox.png
```

三个 GGUF 缺任一都会立即拒绝（`src/app/mod.rs:32-37`）：

```
Z-Image model requires --text-encoder, --vae, --prompt, and --out
```

`--prompt` / `--out` 同样必填。

### 实测耗时（20 核 DGX Spark，512×512，8 步）

| 阶段 | 耗时 | 占比 |
|---|---|---|
| 三组件加载 | 73 ms | — |
| 文本编码 | 677 ms | 0.06% |
| **去噪（8 步）** | **1133 s** | **97.2%** |
| VAE 解码 | 30 s | 2.6% |
| **总计** | **1166 s ≈ 19.4 min** | |

每步约 140 s，其中 `attention_into` 占 50%、`linear ffn` 36%、`linear qkv` 11%。

> **加速空间有限。** 每步要把 5.64 GB Q8_0 权重完整流过一遍，实测约
> 48 GB/s，已是单通道 DDR5 的上限。`examples/dit_vk_bench.rs` 测得
> GB10 的 Vulkan 在这些形状上只有 0.4×–1.36×（加权约 0.6×），因为
> 单图推理权重搬运量不变，而每次 dispatch 的固定开销被放大 2880 次。
> 线程数也非越多越好：16 线程首步 269.3 s，反而慢于 8 线程的 140.5 s。

## 2. 参数约束

来自 `src/app/cli.rs:447-449`：

- `--steps` 必须正整数
- `--resolution` 必须正整数且 **divisible by 16**

## 3. 从 PyTorch 权重转换

上游 `gguf-org/z-image-gguf` 只提供 Q8_0 一档 DiT。仓库自带的转换器可以从
原始 safetensors 重新导出：

```bash
models/.venv/bin/python tools/converter/z_image/convert_z_image.py \
  models/Z-Image-Turbo --out-dir models/z-image-gguf-mine --outtype q8_0
```

参数：

| 选项 | 说明 |
|---|---|
| `--outtype` | `q8_0`（默认）/ `f16` / `f32` |
| `--components` | `all`（默认）/ `dit` / `vae` / `text` |
| `--overwrite` | 覆盖已有输出 |

> **DiT 的 dtype 不随 `--outtype` 变化**。`validate_dit` 硬编码要求
> 30 个主层为 Q8_0、两个 refiner 栈为 F16 —— 主层走量化 matmul，refiner
> 每步只跑两次所以留在 F16。因此 `--outtype f16` 不会让推理更快：
> 实测 139.4 s/步 对 140.5 s/步。

VAE 只导出 `decoder.*`（txt2img 是 latent → 像素，编码器用不上）。

## 4. 已知范围

| 范围 | 状态 |
|---|---|
| Z-Image Turbo（CPU、512×512、txt2img） | `Verified`（[tests/z_image_reference.rs](../../tests/z_image_reference.rs) 覆盖 pinned Oracle） |
| Z-Image Base | `Unsupported` |
| img2img | `Unsupported` |
| GPU 后端 | `Unsupported`（仓库 CPU-only） |

## 5. 与 Oracle 的对齐

Pinned Oracle：[leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
@ `97d2990807fe6d558e395f8764198d7c7e7b411c`

构建 / 测试入口：

- `tools/oracle/z_image/build_stable_diffusion_oracle.sh`
- `tools/oracle/z_image/stable-diffusion-z-image-trace.patch`
- `tests/z_image_reference.rs`

> 仓库口头习惯说的 `Dif.cpp` 实际指这个 stable-diffusion.cpp fork，
> `docs/REFERENCE_IMPLEMENTATIONS.md` 顶端有说明。

## 6. CLI 选项速查

| 选项 | 用途 | 必填 |
|---|---|---|
| `--model` | DiT GGUF | ✓ |
| `--text-encoder` | Qwen3 文本编码器 GGUF | ✓ |
| `--vae` | Flux VAE GGUF | ✓ |
| `--prompt` | 文生图 prompt | ✓ |
| `--out` | 输出 PNG 路径 | ✓ |
| `--steps` | Euler 步数（NFE） | ✓ |
| `--resolution` | 输出分辨率（divisible by 16） | ✓ |
| `--seed` | RNG 种子 | — |
| `--threads` | ComputePool 线程数 | — |

## 7. 相关源码索引

- `src/app/diffusion.rs` — `run_z_image_cli` 主入口
- `src/app/mod.rs:32-37` — pig arch 三组件必填检查
- `src/app/cli.rs:447-449` — `--steps` / `--resolution` 校验
- `src/models/diffusion/pig.rs` — DiT 模型
- `src/models/diffusion/z_image/dit.rs` — DiT forward
- `src/models/diffusion/z_image/text.rs` — Qwen3 文本编码器
- `src/models/diffusion/z_image/vae.rs` — Flux VAE
- `tests/z_image_reference.rs` — pinned Oracle 对齐
- `docs/REFERENCE_IMPLEMENTATIONS.md` — Oracle pin 与构建脚本
- `tools/converter/z_image/convert_z_image.py` — safetensors → 三组件 GGUF
- `tools/converter/z_image/probe_layout.py` — 只读 header，打印张量名/维度分布