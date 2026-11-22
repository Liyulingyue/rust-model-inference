# 重构残余项：rust-model-inference

> **文档用途：** 原《代码重构规划》已基本执行完毕（Phase 1-3、5、6 完成，物理拆分全部落地：
> `ops.rs` → `ops/{kernel,quant,math,...}` 目录化、`model.rs` → `core/` 三件套、`format/` 归位、
> `main.rs` 现 442 行）。模型目录结构由 [`MODEL_ORGANIZATION.md`](MODEL_ORGANIZATION.md) 接管。
> 本文档跟踪**当前仍存在的架构偏差**（GPU 模块归属、ggufrs.rs 单文件、lib.rs glob re-export、
> 根目录散文件），全部收敛后即可删除本文档。

## 1. GPU 模块归属（已落地 · 第三条路径）

**当前状态**（`wc -l` 实测 2026-11-19，`c4c1685`）：

| 文件 | 行数 | 角色 |
|---|---:|---|
| `src/vulkan.rs` | 2760 | 胶水层：声明 `pub(crate) mod {dense, dory, matmul_q4k, nemotron_h, ops, q8_layout, qwen3, qwen35, timestral, yue2}`（+ 测试内 `zimage_probe`），提供 `VulkanContext`、`gpu_broken()`、`mark_gpu_broken()`、`GPU_BROKEN`、`VulkanError`、`SHADER` 常量；`pub use ops::{run_batched_matmul_check, run_qwen3_operator_check}` 转发两条诊断入口 |
| `src/vulkan/ops.rs` | 15059 | Vulkan 计算内核（buffer / arena / matmul / shader dispatch） |
| `src/vulkan/qwen35.rs` | 4547 | Qwen35 trunk 专用 Vulkan session |
| `src/vulkan/nemotron_h.rs` | 2234 | Nemotron-H / Dory 系 Vulkan session |
| `src/vulkan/dense.rs` | 1160 | 稠密投影共享路径 |
| `src/vulkan/dory.rs` | 1068 | Dory 1.0 Vulkan session |
| `src/vulkan/yue2.rs` | 708 | YuE2 音频编解码 Vulkan path |
| `src/vulkan/timestral.rs` | 569 | Timestral Vulkan session |
| `src/vulkan/qwen3.rs` | 486 | Qwen3 trunk 专用 Vulkan session |
| `src/vulkan/q8_layout.rs` | 405 | Q8_0 权重布局 helpers |
| `src/vulkan/matmul_q4k.rs` | 230 | Q4_K matmul kernel |
| `src/wgpu.rs` | 372 | 单一文件，未拆分（feature-gated `--features wgpu`） |

布局形态是 Rust 2018+ 的 `foo.rs` + `foo/*.rs`，但**不是**原计划假设的两个方向（`并入 ops/` / `独立 backend/`）中的任何一个，而是第三条混合路径。

**依赖方向现状**（[`MODEL_ORGANIZATION.md`](MODEL_ORGANIZATION.md) 约束审视）：

- `lib.rs:14` `pub mod vulkan;` 把 `vulkan` 作为顶级公共模块暴露；引擎侧从 `models/qwen3/trunk/forward.rs`、`models/qwen35/trunk/forward.rs` 等多处直接 `use crate::vulkan::qwen3::*` / `qwen35::*`，与 trunk 形成紧耦合。
- `vulkan.rs` 把 GPU 视为设备抽象（非纯计算内核）是有理由的——`VulkanContext` 跨 matmul/launch/调度共享——所以不并入 `ops/` 是合理的。
- **真正的问题**是目录语义不统一：`wgpu.rs` 仍是单文件根模块；`vulkan` 走子目录混合形态。如果未来 wgpu 也要拆分，需要决定走同样的 `wgpu.rs` + `wgpu/*.rs` 形态还是新建 `src/backend/{vulkan,wgpu}/`。

**待决**：
1. 是否把 GPU 子目录统一改名为 `src/backend/{vulkan,wgpu}/`（`vulkan.rs` → `backend/vulkan/mod.rs`），让两个后端的目录形态对齐？
2. 是否把 `vulkan::ops` / `vulkan::qwen3` / `vulkan::qwen35` 之间的 `pub(crate)` 边界画得更清晰（避免 `models::qwen3::trunk` 直接 `use crate::vulkan::qwen3::*`）？

**验收**：`cargo build --release --features vulkan` 与 `--features wgpu` 仍可编译；`pub use` 路径不变（`format::ggufrs::xxx` 同样不破坏外部调用方）。

## 2. ggufrs.rs 内容拆分（风险：低）

`format/ggufrs.rs` 仍是 **4362 行**单文件（读写/校验混在一起；2026-09 实测 4107 行，
此后继续增长）。

**做法**：按职责拆为 `format/ggufrs/{read,write,validate}.rs`，纯物理搬移不改逻辑。
`lib.rs` 的 re-export 路径不变（`format::ggufrs::xxx` 通过 mod 转发）。

## 3. lib.rs 清理 glob re-export（风险：低）

Phase 5 的"精选 re-export 替换一把梭"未完成，残留 **三处** glob（`grep -n "^\s*pub use .*\*;" src/lib.rs` 实测，2026-11-19）：

- `lib.rs:39` `pub use models::qwen3::asr::model::*;`
- `lib.rs:40` `pub use models::qwen3::*;`
- `lib.rs:49` `pub use ops::*;`

**做法**：展开为显式 re-export 列表。`ops::*` 尤其要小心——`ops` 目录下的 `pub fn` 数量较多（`ops/float.rs`、`ops/dot.rs`、`ops/softmax.rs`、`ops/norm.rs`、`ops/sampling.rs`、`ops/rope.rs` 等），建议按"高频外部依赖 vs 内部 helper"两栏筛选后再展开。改完 `grep -rn 'crate::' src tests examples` 全仓库引用路径确认无遗漏。

## 4. 根目录散文件（新增登记）

第 1-3 节假设"根目录只放入口（`main.rs`、`lib.rs`）+ GPU 模块"，但实测还有两处根条目未在任何文档登记（2026-11-19 复核）：

| 条目 | 行数 | 门控 | 内容 |
|---|---:|---|---|
| `src/parity_trace.rs` | 471 | `#[cfg(feature = "parity-trace")]`（`lib.rs:10-12`） | llama.cpp 逐层精度对比 debug 模块 |
| `src/prompt/`（目录，`mod.rs` 462 + `jinja.rs` 1236 + `jinja_compat.rs` 683 + `legacy.rs` 265） | 2646 | 无门控（`lib.rs:13` `pub mod prompt;`，`lib.rs:50` `pub use prompt::{...}`） | Chat 模板 builder（Qwen / Hunyuan，含 Jinja 引擎） |

**倾向**：
- `parity_trace.rs` 迁入 `src/devtools/parity_trace.rs`（与"调试工具"语义一致）；或在 `MODEL_ORGANIZATION.md` §9「已知偏差」里登记为 feature-gated 例外。
- `src/prompt/` 已从单文件长成根目录子模块（带 `legacy.rs` 兼容层）；待决问题收敛为是否进一步迁入 `src/core/prompt/`（与 `core/tokenizer`、`core/scratchpad` 同属"输入处理"层），迁移时同步把 `pub use prompt::{...}` 在 `lib.rs` 里改成 `pub use core::prompt::{...}`。

**待决**：选一个方向统一处理，避免根目录继续累积散条目。

## 当前测试基线（残余项的回归判据）

`cargo test --profile release-fast --lib`（2026-11-19 实测，`c4c1685`）：
**1615 passed / 18 failed / 117 ignored**。任何残余项改动前后该基线不得变化；
改动后用以下命令现场复核：

```bash
git rev-parse HEAD
cargo test --profile release-fast --lib 2>&1 | tail -n 5
```

18 个失败均为预存平台/fixture 相关项（breeze / clm / gemma4 / funasr / qwen_drive /
yue2 / z_image / rope / matmul / ggufrs 等，需模型 env 或特定硬件的用例），与
重构无关；ignored 桶含大量需真实权重的重模型用例。新合入的模块测试
（pjit / underdog_woof / d3_turbo / decision 等 58 项）全部通过，计入 passed。
