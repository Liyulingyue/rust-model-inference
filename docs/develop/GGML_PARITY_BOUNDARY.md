# `_ggml` 函数的使用边界

> 记录于 2026-11-06。这份文档回答一个具体问题：**仓库什么时候应该复刻 ggml 的算术，什么时候不应该。**

## 原则

仓库里有一批 `_ggml` 后缀的函数（`dot_f32_ggml`、`silu_ggml`、`softmax_ggml_inplace`、
`forward_prepared_ggml`、`ops/kernel/{f32,q8_0}/ggml.rs` 等）。它们复刻上游
llama.cpp / ggml 的**具体累加结构** —— 几个累加器、什么顺序 reduce、用哪个 `exp`
—— 目的是让结果与 ggml **逐位一致**。

**默认不新开这种函数。** 仓库的原则是：

> 实现并采用 `dot_f32` 这样的函数，尽量不采用 `dot_f32_ggml` 这样的函数，
> 仅当无法绕开时才使用。

### 为什么

1. **浮点加法不满足结合律。** 换一种累加顺序只是换了一种舍入，不是更正确。
   `(a+b)+c` 与 `a+(b+c)` 在 F32 下可以差 1 ULP，没有哪一边「更对」。
2. **ggml 的结构既不是精度最优，也不是速度最优。** 它是 ggml 在特定硬件、
   特定编译器、特定历史时期的选择。从精度和速度两个方向都很难证明它最优，
   因此没有理由让仓库默认遵循它。
3. **维护成本。** 每多一条 `_ggml` 路径，就多一份要维护、要测试、要和上游
   同步的算术。上游一改，这条路径就悄悄失效了。

## 允许引入的四种情况

| 情况 | 例子 |
| --- | --- |
| **无法绕开的精度问题** | 上游用某结构算出的值与仓库结构差异过大，导致量化档位翻转、输出崩掉 |
| **行为对齐问题** | 参考实现的**行为**（不是数值）依赖该结构，例如 softmax 掩码列必须恰好贡献 0 |
| **行为对齐测试** | 需要逐位复现参考实现才能定位第一个分叉点 |
| **精确数值测试** | 断言两个实现在特定位模式上完全一致 |

### 引入时必须同时满足

1. **注释写明为什么绕不开** —— 具体到「不用它会怎样」，而不是「为了对齐 ggml」。
   如果注释只写「对齐 ggml」，那它就不该存在。
2. **不改变默认路径的契约** —— 既有调用方继续走 `dot_f32` 等默认入口，`_ggml`
   是 opt-in 的并行入口（trait method 或独立函数）。
3. **命名与文档可发现** —— 新函数名带 `_ggml`，并在本文档登记它为什么存在。

## 现状：已有的 `_ggml` 函数

以下是**历史既有**，不是本次新增。它们的存在理由各不相同，收口方向也不同。

| 函数 | 位置 | 存在理由 | 收口方向 |
| --- | --- | --- | --- |
| `dot_f32_ggml` | `src/ops/dot.rs` | x86 上复刻 ggml 的 4 累加器归约；aarch64 上转发到 `dot_f32_neon`（与默认等价） | 保留：x86 对拍需要 |
| `dot_f32_ggml` / `_stride` / `_stride_padded` | `src/ops/kernel/f32/ggml.rs` | nemotron trunk 的 attention 需要 stride/padding 语义 | 随 nemotron 评估 |
| `matmul_q8_0_vs_q8_0_ggml` | `src/ops/kernel/q8_0/ggml.rs` | 挂在 `forward_prepared_ggml` 默认 trait 方法上 | 见下方「注意」 |
| `silu_ggml` / `silu_ggml_inplace` | `src/ops/activation/silu.rs` | ggml 的多项式 `exp`，影响 softmax 掩码列是否恰好为 0 | nemotron 专属 |
| `softmax_ggml_inplace` | `src/ops/softmax.rs` | 同上 | nemotron 专属 |
| `forward_prepared_ggml` | `src/ops/kernel/trait_.rs` | nemotron trunk 的 matmul 入口 | nemotron 专属 |
| `q5_0::neon::*` / `q5_1::neon::*` | `src/ops/kernel/q5_{0,1}/neon.rs` | **生产 kernel 本身**，被重写成 ggml 结构 | 见下方「注意」 |
| `exp_ggml4` / `silu_ggml4` | `shaders/glsl/exp_ggml.glsl` | ggml 的 `ggml_v_expf`，Mamba-2 的 conv/gate 的 SiLU 必须与 CPU 的 `silu_ggml_inplace` 同一位，否则 1 ULP 会被 Mamba-2 递归放大。仓库自带的 `exp_approx_neon.glsl`（求值顺序不同）与 `exp_accurate.glsl`（更高阶多项式）都不是 ggml 的那一个 | nemotron 专属 |

### 注意：Q5_0/Q5_1 的 NEON kernel 是生产代码

这两个**不是**「为对齐而存在的副本」，而是 Q5_0/Q5_1 在 aarch64 上的**唯一**
NEON 实现（默认 dispatch 就走它）。它们之所以长成 ggml 的样子，是因为对齐
过程中发现仓库原有的累加结构与 ggml 不同，会导致 30B 的输出分叉，于是把
**生产 kernel 本身**改成了 ggml 结构。

这意味着**其他模型也在用这份 kernel**。若其他模型原本有自己的数值基线，
这次重写可能改变了它们的结果。收口方向：

- 确认其他模型的既有 oracle 是否仍成立（跑一遍它们的 reference 测试）；
- 若某个模型因此分叉，评估是它该跟着变，还是该给它保留旧结构。

## 同名不同义：两处 `dot_f32_ggml`

仓库里有两个 `dot_f32_ggml`，**同名但签名与语义都不同**，读代码时注意区分：

| 位置 | 签名 | 语义 |
| --- | --- | --- |
| `src/ops/dot.rs` | `dot_f32_ggml(a: &[f32], b: &[f32], n: usize) -> f32` | 通用算子层的 ggml 归约顺序；aarch64 上转发到 `dot_f32_neon` |
| `src/ops/kernel/f32/ggml.rs` | `dot_f32_ggml(x: &[f32], y: &[f32]) -> f32` | 长度由 slice 决定；另有 `_stride` / `_stride_padded` 变体 |

**aarch64 上两者等价**（都走 `dot_f32_neon` 的 4 累加器结构）；**x86 上不等价**
（前者 4 个 256-bit 累加器 + pairwise reduce，后者单累加器）。

## GDN 归一化：两代 llama.cpp 的两个公式

这不是 `_ggml` 函数，但属于同一类问题：**上游改了结构，仓库必须选一个**，
所以登记在这里。

llama.cpp 的 GDN q/k 归一化换过公式：

| llama.cpp | 结构 | 公式 |
| --- | --- | --- |
| `11fe0215` 及更早 | `build_gdn_l2_norm` = `ggml_scale(ggml_rms_norm(x, eps/n), 1/sqrt(n))` | `x / sqrt(sum(x²) + eps)` |
| `b96806d` 起 | `ggml_l2_norm` | `x / max(sqrt(sum(x²)), eps)` |

差在 epsilon 的位置：旧公式把 `eps` 加进平方和里，新公式让它当下限。
`eps = 1e-5`、GDN q/k 范数约 0.14 时，两者相差 `eps / (2 * sum) ≈ 2.4e-6`
相对——正好是 Qwen3.8 对拍在 `q_conv_predelta-0` 上分叉的位置。

**默认取新公式**（`src/models/qwen35/trunk/util.rs` 的 `l2_norm`）：

- 仓库所有模型的**转换**都 pin 在 `b96806d`；
- `b96806d` 是较新的上游，新公式是上游当前行为，也是模型语义本身
  （HF 参考实现用新公式，旧的 llama.cpp 是近似）；
- `4a2ac0b`（Holo4 合并）曾把公式换成旧的，导致与 `b96806d` 对齐的
  Qwen3.8 / Holo4 / Sage-A1.5 / NeoHorse / MiniCPM-V 35B 全部失去对齐。
- 换回新公式后的复测（2026-11-12）：OvisOCR3 的 `state_predelta-0` 回到
  `1.1e-7`（旧公式 `4.7e-5`）；NeoHorse-1-4B 的 **BF16 档与 llama.cpp 逐位
  一致**，量化档的 GDN checkpoint `8.9e-8`…`5.4e-7`（旧公式 `8.6e-5`）。

**旧公式保留为开关，默认关闭**：`RMI_QWEN35_GDN_LEGACY_NORM=1`
（`src/models/qwen35/trunk/util.rs` 的 `legacy_gdn_norm`）。它是**诊断工具**，
不是逐模型代码路径：

- 对拍是**排 bug** 的手段，不是逐模型对齐的目标——误差合理就算过，所以默认
  走上游的新公式；
- 要在 CPU 上量一量两个公式差多少时再打开：只换这一处，OvisOCR3 的
  `state_predelta-0` 就从 `1.1e-7` 变成 `4.7e-5`（`q_conv_predelta-0` 从
  `7.8e-8` 变成 `1.18e-5`），是这类公式差最干净的观测点；
- 仓库里已经没有 Oracle 需要它了：OvisOCR3 的运行时 Oracle 已随 TODO-068
  重新 pin 到 `b96806d`，与它的转换 pin 一致，所以默认路径不再带
  `eps / (2 * sum)` 的**公式版本差**。

Vulkan 侧的 `shaders/glsl/qwen35_recurrent_conv.comp` 始终用上游的新公式；
开关只作用于 CPU 路径。

## 未来改动的收口方向

- **新增代码不得再引入 `_ggml` 函数**，除非命中上表四种情况之一。
- **对既有 `_ggml` 函数**，若其存在理由已消失（例如对拍已完成、模型已验证），
  评估把调用点迁回默认入口，让 `_ggml` 路径自然消亡。
- **无法迁移的**（例如上游行为确实依赖该结构），在上表补一行「为什么绕不开」，
  避免下一个人重新纠结。
- **新增 `_ggml` 函数时**，同步更新上表。

## 相关文档

- `.agents/skills/operator-internal-dispatch/SKILL.md` 的「`_ggml` 后缀函数的使用边界」
  —— 实现时的硬约束与检查清单
- `.agents/skills/adapting-new-models/SKILL.md` 第 5 步 —— 适配新模型时不要默认照搬 ggml
- `.agents/skills/reviewing-model-changes/SKILL.md` 的「数值验收标准」—— 审阅时怎么问「为什么绕不开」
- `docs/develop/REFERENCE_IMPLEMENTATIONS.md` —— Oracle 清单与固定版本
