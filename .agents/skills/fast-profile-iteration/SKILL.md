---
name: fast-profile-iteration
description: Use when 在 rust-model-inference 中需要快速验证改动（编译、单测、clippy），且完整的 release 编译耗时不可接受时。覆盖迭代 loop 的常用命令和取舍。
---

# Fast profile for iteration

> **本 skill 是被引用项。** 它只回答「怎么编译/跑测试最快」，不描述任何任务流程。
> 工作流类 skill（`adapting-new-models`、`reviewing-model-changes`）在需要编译、
> 跑单测时**引用**它，不要在各自的 skill 里重复写编译命令——否则同一套命令会在
> 多个文件里各写一份，改 profile 时漏改一处。

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

`release-fast` 的 baseline：**1346 passed / 26 failed / 87 ignored**（2026-11-04，Linux/aarch64 实测）。**26 failed 是预存失败**，分两类，都不是本次回归：

- **平台相关（aarch64 上的 NEON parity）**：`ops::quant::neon_k`（8）、`models::funasr::encoder::parity_tests`（2）、`models::diffusion::z_image::text`（2）、`models::diffusion::z_image::vae`（1）、`ops::rope::tests` / `ops::rope::neox::tests` / `ops::matmul::neon_tests`（各 1）、`models::gemma4::asr`（1）——都是拿 ggml 的 ARM NEON 参考值比对，在 x86 上反而是过的。
- **平台无关**：`models::breeze`（3）、`models::clm`（2）、`models::yue2` / `models::qwen_drive` / `models::gemma4::trunk` / `format::ggufrs`（各 1）——parity oracle 未跑或第三方依赖回归。

x86 机器上的失败名单不同（少 NEON 那一组），所以**先跑一遍拿当次基线**再对比。

> 数字会随模型适配漂移。**每次改动前先跑一遍拿到当次基线，报告时逐条对比失败名单，不要只对比总数**——新增一个失败和少一个失败在总数上可能互相抵消。

## 什么时候**不**用 fast

- 用户要求发布构建、跑 perf benchmark、生成最终 binary → 用 `cargo build --release`
- 需要 dev profile（debug 断言 + 完整 backtrace）→ 用 `cargo build` 或 `cargo test`
- parity-oracle 测试需要严格浮点顺序对齐 → 仍走 release-fast 已经够，但 oracle 本身是 `#[ignore]` 的，需要显式 `cargo test --profile release-fast -- --ignored`

## **测速要看 profile**：dev 的 perf 数字没意义，release-fast 是最低门槛，要真实吞吐得 `release`

仓库里有三个 perf 等级：

| Profile | opt-level | LTO | codegen-units | 闭包/SIMD 跨函数优化 | 实测吞吐（1024×1280×3840 matmul，8 线程） |
| --- | --- | --- | --- | --- | --- |
| dev（默认） | 0 | — | 16 | 全无；`_mm256_fmadd_ps` 当普通函数调用 | **~2 GFLOPS**（1.94 GFLOPS，2026-10-11） |
| `release-fast` | 3 | thin | 16 | SIMD 算子本身能 inline；**但闭包/LTO 跨函数优化不到位** | **~55 GFLOPS**（55.51 GFLOPS，2026-10-11） |
| `release` | 3 | fat | 1 | 全开；闭包去糖化、跨 crate 内联 | 接近理论 400 GFLOPS |

**怎么选**：

- **dev 构建的 perf 数字没参考价值**——SIMD 都不发，怎么改都不会快。先确认自己有没有在 dev 上自欺欺人。
- **release-fast 是验证"加速路径是否生效"的最低门槛**——能区分"完全没 SIMD"和"SIMD 已就位但吞吐仍不理想"两种情况。如果 release-fast 已经把吞吐拉到 50%+ 理论峰值，说明 SIMD/内联路径都在工作，剩下的瓶颈通常是内存、缓存或分配。
- **要真实吞吐、做容量规划、判断"这个模型 e2e 要几分钟"——必须 `release`**。release-fast 的 55 GFLOPS 在闭包/LTO 缺失场景下可能还差 release 真实值一倍以上（典型 1.3~2×），用 release-fast 估的 ZDT 23s 在 release 下可能就是 12~15s，但容量估算时不能拍脑袋。

**触发条件**：dev 测出来的数字看起来"明显低于理论"，必须切到 release-fast 重测；如果 release-fast 已经验证了路径，**且**你需要估算 e2e 实际时间，再花 5 min 切到 release。

**绝对禁止**：用 dev 构建的耗时数倒推"算法没生效"、"SIMD 没起作用"、"有性能 bug"——绝大多数情况下只是 `-O0` 没启用。

**编译耗时控制**：

- dev：~3s 增量
- release-fast：~80s 增量（增量编译；冷启动 1~2 min）
- release：~5 min 冷启动

如果 release 编译耗时超本会话墙（默认 300 s），就**先 release-fast 验证路径**，然后告诉用户"需要 release 数据，但他们自己跑 `cargo build --release && <bench>`"。**不要**用 release-fast 的数字硬装 release 的结论。

## 已知约束

- 编译 warning 数量很大（lib 约 300 条，bin 2 条）几乎全部是历史遗留 `unused variable` / `dead_code` / `unused import`，与本次改动无关。**不要把 warning 当 error**（`-D warnings` 会刷屏）。
- 只跑 Vulkan 的 example（`astronex_gpu_determinism`、`astronex_text_determinism`、`mage_vae_conv_probe`、`spv_compile_probe`、`vk_mem_info` 等）在 `Cargo.toml` 里带 `required-features = ["vulkan"]`，不带该 feature 时 `cargo build --all-targets` 会自动跳过它们。**新增只跑 Vulkan 的 example 时记得补这条**，否则默认构建会报 "main function not found"。
- 缓存命中：重复编译 + 短改动通常 ~5s；冷启动或大改动 ~30~60s。
- 编译命令若在 bash 工具里 timeout，检查是否忘记加 `--profile release-fast` 走了默认 dev profile。

## 反模式

- 不要每次都 `cargo build --release`——会浪费 5 分钟等冷编译。
- 不要因为 release-fast 名字带"fast"就以为它不做优化——它仍然有 LTO 和高级别 opt，只是去掉了 codegen-units 拆分和符号剥离等发布用步骤。
- 不要在 fast 模式下编出二进制拿去发布——`release-fast` 不剥离 panic 符号、没有 panic=abort，不适合生产。