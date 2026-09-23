---
name: fast-profile-iteration
description: Use when 在 rust-model-inference 中需要快速验证改动（编译、单测、clippy），且完整的 release 编译耗时不可接受时。覆盖迭代 loop 的常用命令和取舍。
---

# Fast profile for iteration

## 核心原则

仓库自定义了一个 `release-fast` Cargo profile，编译耗时约为默认 `release` 的 1/5~1/8（典型改动 ~30s vs ~5min），同时保留 LTO + 优化等级，足以运行单元测试和绝大多数验证。**默认使用 `--profile release-fast`**；只在用户明确要求发布产物或运行长时 benchmark 时切到 `release`。

## 什么时候用 fast

| 场景 | 命令 |
| --- | --- |
| 改完想确认能编译 | `cargo build --profile release-fast` |
| 跑 lib 单测 | `cargo test --profile release-fast --lib` |
| 跑特定模块单测 | `cargo test --profile release-fast --lib <module>::` |
| 跑特定测试（带输出） | `cargo test --profile release-fast --lib -- --nocapture <test_name>` |
| 看 clippy 警告 | `cargo clippy --profile release-fast --lib -- -D warnings`（按需） |
| 跑集成测试 | `cargo test --profile release-fast --test <test_name>` |

`release-fast` 的 baseline：786~789 passed / 14 failed / 60 ignored。**14 failed 是预存失败**（parity oracle 未跑、第三方依赖回归），不是本次回归。

## 什么时候**不**用 fast

- 用户要求发布构建、跑 perf benchmark、生成最终 binary → 用 `cargo build --release`
- 需要 dev profile（debug 断言 + 完整 backtrace）→ 用 `cargo build` 或 `cargo test`
- parity-oracle 测试需要严格浮点顺序对齐 → 仍走 release-fast 已经够，但 oracle 本身是 `#[ignore]` 的，需要显式 `cargo test --profile release-fast -- --ignored`

## 已知约束

- 编译 warning 数量很大（~400）几乎全部是历史遗留 `unused variable` / `dead_code`，与本次改动无关。**不要把 warning 当 error**（`-D warnings` 会刷屏）。
- 编译时会有几个 `vk_*` example crate 因为缺 `main` 报 hard error（line 54/102/144）。这是 pre-existing 仓库状态，**不是本次改动的回归**——忽略它们，看 lib 的 build success 即可。
- 缓存命中：重复编译 + 短改动通常 ~5s；冷启动或大改动 ~30~60s。
- 编译命令若在 bash 工具里 timeout，检查是否忘记加 `--profile release-fast` 走了默认 dev profile。

## 反模式

- 不要每次都 `cargo build --release`——会浪费 5 分钟等冷编译。
- 不要因为 release-fast 名字带"fast"就以为它不做优化——它仍然有 LTO 和高级别 opt，只是去掉了 codegen-units 拆分和符号剥离等发布用步骤。
- 不要在 fast 模式下编出二进制拿去发布——`release-fast` 不剥离 panic 符号、没有 panic=abort，不适合生产。