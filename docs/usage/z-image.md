# Z-Image Turbo 用法

Z-Image Turbo 是文生图模型，分三个独立 GGUF 组件：

- DiT：`--model`
- Qwen3 文本编码器：`--text-encoder`
- Flux VAE：`--vae`

GGUF `general.architecture = pig`，对应 `src/models/diffusion/pig.rs` 与
`src/models/diffusion/z_image/`。CLI 入口 `src/app/image.rs::run_z_image_cli`。

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

## 2. 参数约束

来自 `src/app/cli.rs:447-449`：

- `--steps` 必须正整数
- `--resolution` 必须正整数且 **divisible by 16**

## 3. 已知范围

| 范围 | 状态 |
|---|---|
| Z-Image Turbo（CPU、512×512、txt2img） | `Verified`（[tests/z_image_reference.rs](../../tests/z_image_reference.rs) 覆盖 pinned Oracle） |
| Z-Image Base | `Unsupported` |
| img2img | `Unsupported` |
| GPU 后端 | `Unsupported`（仓库 CPU-only） |

## 4. 与 Oracle 的对齐

Pinned Oracle：[leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
@ `97d2990807fe6d558e395f8764198d7c7e7b411c`

构建 / 测试入口：

- `tools/z_image/build_stable_diffusion_oracle.sh`
- `tools/z_image/stable-diffusion-z-image-trace.patch`
- `tests/z_image_reference.rs`

> 仓库口头习惯说的 `Dif.cpp` 实际指这个 stable-diffusion.cpp fork，
> `docs/REFERENCE_IMPLEMENTATIONS.md` 顶端有说明。

## 5. CLI 选项速查

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

## 6. 相关源码索引

- `src/app/image.rs` — `run_z_image_cli` 主入口
- `src/app/mod.rs:32-37` — pig arch 三组件必填检查
- `src/app/cli.rs:447-449` — `--steps` / `--resolution` 校验
- `src/models/diffusion/pig.rs` — DiT 模型
- `src/models/diffusion/z_image/dit.rs` — DiT forward
- `src/models/diffusion/z_image/text.rs` — Qwen3 文本编码器
- `src/models/diffusion/z_image/vae.rs` — Flux VAE
- `tests/z_image_reference.rs` — pinned Oracle 对齐
- `docs/REFERENCE_IMPLEMENTATIONS.md` — Oracle pin 与构建脚本