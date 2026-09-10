# LFM 家族用法

本仓库对 LFM2 / LFM2.5 / LFM2-MoE / LFM2.5-VL 的端到端命令行示例。

> 通用前置：构建 `cargo build --release --bin rust-model-inference`。
> 所有 LFM 文本 / 视觉模型在 GGUF 里的 `general.architecture` 都是字符串 `"lfm2"`。
> LFM2 与 LFM2.5 文本的变体由 CLI 路由阶段通过 `general.basename` 含 `"2.5"` 区分
> （`src/app/text.rs:62-66`），分别进入 `src/models/lfm2/` 与 `src/models/lfm25/`。
> 详见 `docs/ISSUE.md` 的 LFM2 / LFM2.5 命名不一致条目。

## 1. LFM2 文本（`src/models/lfm2/`）

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/lfm2/LFM2-XXXX-Q8_0.gguf \
  --prompt "法国的首都是" --max-tokens 30
```

## 2. LFM2.5 文本（`src/models/lfm25/`）

GGUF `general.basename` 含 `"2.5"` 时自动路由：

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/lfm2.5/LFM2.5-1.2B-Instruct-Q8_0.gguf \
  --prompt "法国的首都是" --max-tokens 30
```

如果 `general.basename` 不含 `"2.5"`，同样的 GGUF 仍能加载但会走 LFM2 trunk，
可能与 LFM2.5 实际架构不一致。建议显式选 basename 正确的 GGUF。

## 3. LFM2-MoE

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/LFM2-8B-A1B-Q8_0.gguf \
  --prompt "2 + 3 =" --max-tokens 4 --temp 0
```

注意：

- 这是 **MoE** 路由，进入 `src/models/lfm2moe/`，CLI 路由条件为
  `arch == "lfm2moe"`（来自 GGUF metadata，非通用 `lfm2`）。
- 与 llama.cpp 前 6 个生成 token 一致；在 MoE 近平局处可能分叉。
- shared-expert 张量不被支持（会被 loader 显式拒绝）。

## 4. LFM2.5-VL（`src/models/lfm2/vision.rs`）

Vision 路径与文本路径**共用** `arch == "lfm2"` 分派，不走 basename。
代码里直接落到 `src/models/lfm2/vision.rs::run_multimodal`：

```bash
cargo run --release --bin rust-model-inference -- \
  --model models/lfm2.5-vl/LFM2.5-VL-450M-Q8_0.gguf \
  --mmproj models/lfm2.5-vl/mmproj-F16.gguf \
  --image path/to/image.jpg \
  --prompt "描述这张图片"
```

约束：

- 必须同时传入 `--mmproj` 与 `--image`，缺一即报错：
  - `LFM2.5-VL requires --mmproj`
  - `LFM2.5-VL requires --image`
- 仅支持**视觉**模态；传入 `--audio` 会被拒绝：
  `Only gemma4 architecture is supported for multimodal audio, got: lfm2`
  （见 `src/app/text.rs:803-806`）
- KV cache 固定 `F16`（`KvFormat::F16` 直接传给 `run_multimodal`）。
- mmproj 期望的 projector 元数据键：`clip.vision.{image_size, patch_size,
  projector.scale_factor, embedding_length, attention.head_count, block_count,
  feed_forward_length, attention.layer_norm_epsilon, projection_dim, image_mean,
  image_std}`，见 `src/models/lfm2/vision.rs:73-83`。

### 4.1 图片大小与 vision token 数（CPU 实测，2026-09）

Vision encoder 把图切成 512×512 tile + 1 张 overview。tile 数和总 vision tokens
直接决定 prefill 时长：

| 原图分辨率 | Tile grid | Vision tokens | 备注 |
|---|---|---|---|
| ≤ 512（任一维） | 0×0 | ~64–128 | 单 overview，prefill ~1s |
| ~768×768 | 2×2 (4 tiles) + overview | ~1100+ | CPU prefill > 10 分钟（实际不要用） |
| 401×287（实测 `references/apple.png`） | 0×0 | 117 | 端到端 3.8 tok/s @ 8 thread |

**建议**：CPU 路径下使用 ≤ 512×512 的输入图。`references/apple.png`（401×287）
是个不错的示例尺寸。`models/test768.png`（768×768）这种分辨率在 CPU 上
prefill 阶段会卡死，需要 ≥ 10 分钟才能出第一个 token。

3B VL 模型 + 1024 vision tokens 的 CPU prefill 主要成本是 30 层 × 1024
tokens 的 matmul，不是 SIMD gap。如果要测大图，建议加 `--gpu`（Vulkan
未对 `lfm2` arch 完整覆盖，仅在分片 matmul 上生效——见 §6）。

## 5. CLI 路由规则速查

| 输入 GGUF `general.architecture` | `general.basename` | 进入 trunk | Modes |
|---|---|---|---|
| `lfm2` | 不含 `2.5` | `src/models/lfm2/`（文本） | 文本 |
| `lfm2` | 含 `2.5` | `src/models/lfm25/`（文本） | 文本 |
| `lfm2` | 任意 | `src/models/lfm2/vision.rs`（VL，需 `--mmproj --image`） | 多模态 |
| `lfm2moe` | — | `src/models/lfm2moe/` | 文本（MoE） |
| `nanbeige` | — | `src/models/llama/` | 文本（Experimental） |

## 6. 与 llama.cpp 的对齐

`docs/REFERENCE_IMPLEMENTATIONS.md` 中**没有**任何 LFM 家族的 Pinned Oracle：

- 没有固定的 llama.cpp commit
- 没有 `tools/lfm2/...` 构建脚本- 没有 `tests/lfm*_reference.rs`

当前只能靠运行时 smoke test。建议在引入 LFM2.5-VL 的端到端用例之前先固定一个
llama.cpp commit + build 脚本。

## 7. 服务端模式

```bash
cargo run --release --bin server -- \
  --model models/lfm2.5/LFM2.5-1.2B-Instruct-Q8_0.gguf \
  --host 0.0.0.0 --port 8080 --threads 4
```

服务端对 LFM2 / LFM2.5 / LFM2-MoE 等纯文本架构按 CLI 选项暴露，无图像/音频模态。

## 8. 已确认的限制 / 边界

| 范围 | 行为 |
|------|------|
| LFM2.5-VL + `--audio` | 拒绝（只支持视觉） |
| 缺 `--mmproj` / `--image` | 配置阶段报错 |
| LFM2.5-VL GGUF 的 `general.architecture` 是 `lfm2` | 与文本 LFM2 共用 arch，靠 mmproj + image 区分 |
| Dense-LFM2-v2 当前模型库没有对应 GGUF | 文档列入 `Experimental`，不可用 |

## 9. 相关源码索引

- `src/models/lfm2/trunk/` — LFM2 文本 trunk
- `src/models/lfm2/vision.rs` — LFM2.5-VL vision（text decoder 复用 lfm2/ trunk）
- `src/models/lfm25/` — LFM2.5 文本 trunk
- `src/models/lfm2moe/` — MoE 文本 trunk
- `src/app/text.rs:61-83` — LFM2 / LFM2.5 文本路由（basename 分流）
- `src/app/text.rs:809-824` — LFM2.5-VL 视觉路由
- `docs/ISSUE.md` — LFM2 / LFM2.5 命名不一致（arch vs 目录名）