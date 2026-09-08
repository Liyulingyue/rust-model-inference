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
