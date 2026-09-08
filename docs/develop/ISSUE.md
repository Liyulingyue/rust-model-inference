# Issue Track — RustModelInference Debugging Log

## LFM2 / LFM2.5 命名不一致（arch vs trunk 目录名）

**状态**：已记录，暂不修。

**现象**：LFM2 和 LFM2.5 文本模型在 GGUF `general.architecture` 上都是字符串
`"lfm2"`（同一个值）。区分两者靠 `general.basename` 含不含 `"2.5"`（见
`src/app/text.rs:62-66`）。但 Rust 源码侧拆成两个目录：

- `src/models/lfm2/` — LFM2 文本
- `src/models/lfm25/` — LFM2.5 文本（这个 `lfm25` 是我们自己的目录名，**不是**
  GGUF 字段）

`Lfm25Config::from_source`（`src/models/lfm25/trunk/config.rs:55`）一行
`general.architecture` 检查都没有，直接读 `lfm2.*` 前缀的 metadata。

**为什么会迷惑**：

- 看 GGUF 文件，arch 是 `lfm2`，没法只靠 arch 区分 LFM2 / LFM2.5
- 看 Rust 源码，目录叫 `lfm25/`，让人以为 GGUF 字段也叫 `lfm25`
- LFM2.5-VL（vision 侧）走 `src/models/lfm2/vision.rs`，进一步打破
  「源码目录名 = 模型变体」的隐式假设

**影响**：

- 文档（`docs/MODEL_LIST.md`、`docs/SUPPORTED_MODELS.md`）必须同时说清楚
  「GGUF arch = lfm2 + basename 区分」和「源码目录 = lfm2 vs lfm25」
- 新人 review 容易误解（LFM2.5 是不是 arch = `lfm2.5`？答：不是）
- 任何「按 arch 分流」的工具/脚本都得改用 basename 判断

**修复方向**（按工作量排序）：

1. **改名**（最小代价）——`src/models/lfm25/` → `src/models/lfm2_5/` 或
   `src/models/lfm2/v5/`。改 import 即可，不改行为。但会动 git blame。
2. **加注释**——在 `lfm25/` 目录顶加一段 module-level doc，明确写
   「GGUF arch 与本目录名不一致」。不动代码。
3. **不动**——维持现状，靠文档和 reviewer 把关。

**决策**：暂不修，先记录。下次有新人因为这个困惑再处理。

## `extern "C"` 引入 C 标准库 `erff` 的隐式链接

**状态**：当前能跑，先记录；未来 musl-only target / 自包含发布时可能断链。

**现象**：仓库多处用 `unsafe extern "C" { fn erff(value: f32) -> f32; }` 引入
C 标准库 `<math.h>` 的 `erff`（float 误差函数）。涉及面：

- `src/models/vibevoice_asr/encoder.rs:36-38`
- `src/models/dots/blas.rs:26-50`
- `src/models/gemma4/asr/mod.rs:33+`
- `src/models/qwen3/tts/speaker.rs:14`
- `src/models/qwen3/tts/codec/dac.rs:28`
- `src/models/qwen3/asr/mel_encoder.rs:25`
- `src/models/qwen3/asr/audio_processor.rs:15`
- `src/ops/rope.rs:4`

**为什么必须用 `erff` 而不是 tanh 近似**：VibeVoice、Dots TTS、Gemma 4 ASR、
Qwen3-TTS 等组件的训练目标函数是基于精确 erf 形式的 GELU（transformers 的
`ACT2FN["gelu"]`，公式 `0.5x(1+erf(x/√2))`）。换成 tanh 近似会与权重不兼容。

**为什么 Rust 标准库没原生 `erf`**：Rust stable 没有 `f32::erf` / `f64::erf`；
nightly 有 `#![feature(float_erf)]`。

**为什么仓库用 `extern "C"` 而不是 `libm` crate**：

- 简化依赖（不引入 `libm = "0.2"`）
- 系统 libc 默认链接：Linux glibc 的 libm、macOS libSystem、Windows MSVC
  CRT 都暴露 `erff`

**风险**：

| 风险 | 严重度 |
|---|---|
| 当前 target（Linux glibc / MSVC / macOS）下都能跑 | — |
| 切到 musl-only target 时：musl 默认链 libm，但 `extern "C"` 没 `#[link]` 属性，依赖默认行为 | 中 |
| MinGW / `*-pc-windows-gnu` target：`erff` 在 `libmoldname.a`，未必自动拉 | 中 |
| 自包含 / 静态发布：必须保证 libm 在链接列表里 | 中 |
| 文档：新人看到 `unsafe extern "C" { fn erff(...) }` 容易误解为「我们自己写的 C 函数」 | 低 |

**修复方向**：

1. **加 `libm = "0.2"` 依赖**（推荐）——把 `unsafe extern "C" { fn erff... }`
   替换为 `use libm::erff;`，去掉所有 `unsafe { erff(...) }` 调用。libm 是
   musl 的纯 Rust 移植，跨平台一致。
2. **加 `#[link]` 属性**——在现有 `extern "C"` 块上补 `#[link(name = "m")]`（Linux
   musl）或 `#[link(name = "msvcrt")]`（MSVC）以显式声明。不改实现，但
   减少断链风险。
3. **不动**——维持现状，CI 跑通就行。

**决策**：暂不修。先记录。等到出现 musl / MinGW / 自包含发布需求时再处理。
