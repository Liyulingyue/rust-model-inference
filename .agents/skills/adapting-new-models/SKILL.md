---
name: adapting-new-models
description: Use when 用户要求在 rust-model-inference 中新增或扩展 GGUF 模型架构、Tokenizer、推理入口，或要求与 llama.cpp、官方实现进行对齐。
---

# 适配新模型

## 核心原则

先从真实模型文件和固定版本的参考实现确定模型契约，再复用仓库已有能力实现最小接入；完成的标准是可复现的逐位对齐，不是“输出看起来正常”。

## 硬约束

- 不得新增、调用或建议接入任何外部加速库，包括 OpenBLAS、BLAS/LAPACK、MKL、Apple Accelerate、oneDNN、cuBLAS、rocBLAS。只能使用仓库已有的 Rust 算子、线程池、SIMD、量化内核和现有依赖；能力不足时报告阻塞，不得绕过限制。
- 不得把未知架构默认路由到相近模型。只有 Tokenizer、张量契约、位置编码、归一化、Attention/KV、FFN/MoE 和输出变换均一致时才能复用实现，并且仍需显式注册新架构。
- 将用户指定的 llama.cpp 或官方实现视为只读 Oracle。需要插桩时，在临时副本中固定 commit 后修改，不污染用户的参考仓库。
- 以实际提供的 GGUF、mmproj 和组件为范围边界；没有多模态组件就不扩展多模态支持。

## 工作流

1. **固定输入**：记录模型路径、SHA256、文件大小、量化类型，读取 `general.architecture`、全部相关 metadata、Tokenizer/chat template，以及张量名称、shape、类型和组件清单。
2. **固定 Oracle**：优先检查用户点名的本地参考仓库，确认其 commit、工作区状态、模型实现、Tokenizer 和转换脚本确实支持目标架构。找不到匹配 Oracle 时停止并说明缺口，不猜测实现。
3. **追踪现有路径**：结构查询优先使用 CodeGraph；定位架构注册、配置解析、Tokenizer、张量加载、模型构造、forward/session、CLI 调度、`parity-trace` 和相邻测试。先比较契约，再决定可复用部分。
4. **先写失败检查**：至少覆盖新架构不会落入旧模型、关键 metadata/shape 错误会被拒绝、Tokenizer 的 BOS/EOS/特殊 token/Unicode/空白与 Oracle 一致，以及 CLI 确实进入新路径。
5. **最小实现**：仅补齐必须的配置解析、Tokenizer 行为、权重映射、模型计算、状态/KV 管理和显式调度；复用仓库已有算子，不为未来模型建立抽象层。
6. **逐位对齐**：Rust 与 Oracle 使用同一模型、prompt、chat template、线程数、CPU 路径、KV 类型和 greedy 参数。依次比较 token IDs、checkpoint 名称/顺序/shape/次数、中间 F32 的 `to_bits()`/`u32`、最终 logits 位模式和多步 greedy token。出现差异时定位第一个分叉 checkpoint；不得改用容差、余弦相似度或可读文本作为替代结论。
7. **验证真实入口**：运行最小相关单测、格式/编译检查、release 模式真实 CLI 和逐位对齐命令。完整测试中的历史失败、环境失败与本次回归必须分开报告；没有执行的检查不得声称通过。
8. **更新模型清单**：适配完成后同步更新 `docs/MODEL_LIST.md` 和 `docs/develop/SUPPORTED_MODELS.md`；写明具体型号、GGUF architecture、已验证格式、验证范围和已知限制。

## 完成定义

| 层级 | 必须提供的证据 |
| --- | --- |
| 模型契约 | GGUF 哈希、架构、关键 metadata、张量名称/shape/type |
| Tokenizer | 输入文本及双方完全相同的 token IDs |
| 计算图 | 双方 checkpoint 顺序、shape、次数及首个差异位置 |
| 数值 | 选定中间结果和最终 F32 的原始 `u32` 位完全相同 |
| 生成 | 多步 greedy token 序列完全相同且可重复 |
| 工程验证 | 实际执行的命令、结果和明确的覆盖边界 |

## 最小示例

若 GGUF 声明 `general.architecture=newarch`，即使其大部分张量名称像 Qwen，也不能写成未知架构回退 Qwen。应先用 Oracle 证明全部关键契约一致，再添加显式 `newarch` 调度；任何 checkpoint 首次分叉都应修正根因后继续，而不是放宽比较条件。

## 常见错误

- 只看模型名称，不检查 GGUF metadata 和张量清单。
- 选用了“附近”的 llama.cpp checkout，而不是实际支持该架构的固定版本。
- 只比较最终文本，遗漏 Tokenizer、prefill、KV 续写或中间层差异。
- 为追求速度先接入 OpenBLAS 等外部库，改变浮点顺序并掩盖正确性问题。
- 把仓库原有编译故障算作适配回归，或把未运行的真实模型检查写成已通过。
