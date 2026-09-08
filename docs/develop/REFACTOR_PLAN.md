# 重构残余项：rust-model-inference

> **文档用途：** 原《代码重构规划》已基本执行完毕（Phase 1-3、5、6 完成，物理拆分全部落地：
> main.rs 4304→261 行、ops.rs → `ops/{kernel,quant,math,...}` 目录化、model.rs → `core/` 三件套、
> `format/` 归位）。模型目录结构由 [`MODEL_ORGANIZATION.md`](MODEL_ORGANIZATION.md) 接管。
> 本文档只跟踪**尚未完成的 3 项残余重构**，全部完成即可删除本文档。

## 1. GPU 模块归属（决策点，风险：低）

`vulkan.rs`（750 行）与 `wgpu.rs`（335 行）仍在 `src/` 根目录，是根目录仅存的非入口文件。
原计划两个方向（并入 `ops/` 或独立 `backend/`）均未执行。

**待决**：选一个方向执行。
- 倾向：建 `src/backend/{vulkan.rs, wgpu.rs}`——两者是设备抽象而非计算内核，放 `ops/` 会破坏
  `ops 只依赖 core` 的约束（见 `MODEL_ORGANIZATION.md` 依赖方向）。
- 验收：`cargo build --release --features vulkan` 与 `--features wgpu` 仍可编译。

## 2. ggufrs.rs 内容拆分（风险：低）

`format/ggufrs.rs` 仍是 ~4300 行单文件（读写/校验混在一起）。

**做法**：按职责拆为 `format/ggufrs/{read,write,validate}.rs`，纯物理搬移不改逻辑。
`lib.rs` 的 re-export 路径不变（`format::ggufrs::xxx` 通过 mod 转发）。

## 3. lib.rs 清理 glob re-export（风险：低）

Phase 5 的"精选 re-export 替换一把梭"未完成，残留两处：

- `lib.rs` `pub use ops::*;`
- `lib.rs` `pub use models::qwen3::*;`

**做法**：展开为显式 re-export 列表。改完 grep 全仓库 `crate::` 引用路径确认无遗漏。

## 当前测试基线（残余项的回归判据）

`cargo test --lib`：383 passed / 7 failed / 13 ignored。
7 个失败均为环境相关（sparse mmap、需模型 env 的用例等），与重构无关；残余项改动前后该基线不得变化。
