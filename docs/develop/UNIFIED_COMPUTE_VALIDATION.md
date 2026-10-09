# 统一计算执行：实现与验收记录

日期：2026-10-09。实施基线：`ce17cd5d04076e13f7e72b40bb58713630ac16c3`。
实施计划：[八阶段计划](../superpowers/plans/2026-10-08-unified-compute-dispatch.md)。

统一入口、两个 linear 消费者、共享 dense 执行、事务回退及 CLI/HTTP 策略已实现。
**尚未证明真实模型的自动 GPU 加速收益，因此没有开放新的 Auto workload。**
真实 Qwen 权重仍未通过既定 logits 门槛；原始版本能复现相同偏差。
全量测试也有基线失败，不能把本分支描述为“全绿”或“Vulkan 全面可用”。

## 共享算子表达验收（2026-10-09）

本轮基线：`409c2f19b240dd126f8cdb4b9246ff3b2b059cfc`。
[实施范围与验收条件](../superpowers/plans/2026-10-09-shared-dense-operators.md)。

`DenseOp` 携带张量输入、输出及权重角色，`run_dense_layer` 唯一表达 dense
前向的连接和 FFN 公式。Qwen3、标准 Llama 的 CPU 适配器共用 `DenseCpu` 的
RMSNorm、prepared grouped linear、residual 和 SiLU；Vulkan 消费同一表达式，
保留既有 arena、上传和 chunk 提交。各后端的 kernel 仍独立实现。

保留 Qwen 的反向 gate/up scratch 命名、逐行 approximate SiLU、MoE、deepstack、
不同 K/V head 维度和全部 trace；Llama 保留单行 approximate / 多行 exact SiLU。
SiLU shader 新增输出地址，原调用仍写 gate，共享表达式写 up，down 读取同一个逻辑 up。
没有新增每算子提交、读回、activation 拷贝、全局缓存或依赖。

`GpuWeightFormat` 现在只表示存储；`GpuMatmulMode` 独立表示 Prepared、
RoundedInputF32 或 Dot，`GpuMatmulSpec` 组合两者，由算子校验并选择 shader。
旧 `F16F32` 对应 `F16 + RoundedInputF32`，旧 `F16Dot` / `BF16Dot` 对应
相应存储加 Dot。缓存同时验证格式与 mode；非法组合及缓存 mode 变化保留输出。
VAE 继续舍入 F16 输入并 F32 累加；AuK/YuE 保留 Dot 的架构资格与原有归约。

设备：Apple M3 Max / ARM64 NEON / macOS / MoltenVK 1.4.2 / Rust 1.98.1。
构建统一使用 repo-local CARGO_HOME，`--offline --locked --profile release-fast`。

| 验证 | 结果与边界 |
| --- | --- |
| CPU 全库 | 1165 passed / 23 failed / 78 ignored；失败名称与合并基线完全相同 |
| Vulkan + parity-trace 全库 | 1263 passed / 29 failed / 137 ignored；原 Vulkan 28 项失败加独立基线也失败的 Gemma trace 用例 |
| 本轮相关 CPU/trace/shape | 50 passed / 0 failed / 8 ignored |
| RMI_SCALAR=1 相关回归 | 40 passed / 0 failed / 7 ignored |
| 真实 GPU/state/VAE | 68 passed / 1 failed；唯一失败为 main 已复现的 AuK ARM fixture，要求不支持的 AVX2/F16C Dot 提交 |
| 构建与格式 | CPU lib/bins、Vulkan + parity-trace lib/bins/examples check、rustfmt、diff whitespace 通过 |
| Shader | 62 项 manifest、31 个 SPIR-V 验证通过；使用 CI 的 glslang 15.1.0 完整重建，全部 31 个 shader 字节一致 |

新增 operand 表达式和 mode API 测试先编译失败，再实现并通过；CPU scratch 的别名拒绝、
组校验与错误后视图恢复有数值测试。设备用例验证独立 SiLU 输出地址不改输入、不额外提交，
以及数值 mode 变化不污染缓存或 host 输出。现有 Qwen B=1/3/64 logits/KV/提交、Llama
CPU bits/FFN 角色、失败前缀和 VAE 卷积数值断言保持原门槛。

组合 feature 全库最初有两个额外 trace 失败，在单独源码/target 构建的上述基线上
均复现。Qwen fixture 遗漏 main 新增的 raw Q/K checkpoint，本轮补齐预期，
B=1/64 的 token-major checkpoint、shape 和二进制逐位比较通过；Gemma trace 的缺失
checkpoint 是原有失败，没有放宽断言或声称已修复。独立全改动审查无重要/严重发现；
审查后按测试反馈补齐 Qwen fixture并验证 RED→GREEN。

PR CI 暴露本机 glslang 16.6 与 CI 15.1.0 的生成物差异（SiLU 的 SPIR-V
header byte 13）。在仓库 `target/shader-tools` 内构建 15.1.0，重新生成 SiLU
并更新 manifest，完整 `scripts/vulkan-shaders.sh check` 通过。未修改 CI
或降低字节一致性门槛；此前 softmax 重建差异也由同一工具链版本解释。

本轮证明受支持模型的前向表达可以由两后端共用，没有重新测量真实模型速度、
x86/独显或外部 oracle，没有扩大模型资格或开放新的 Auto workload。
历史真实 Qwen 的误差与 GPU 较慢结果仍是原有门禁，不能把本轮结构复用说成已测加速。
原始日志保留在本轮忽略的 `.superpowers/sdd/2026-10-09-shared-dense-operators/`。

## Z-Image refiner 数值回归修复（2026-10-09）

统一分支的 Z-Image DiT refiner 仍把 F16 权重绑定为默认 Prepared，导致 ARM
设备从原有 F32 累加切换成 CPU prepared 的半精度归约，并退出原有 tiled 路径。
这是遗漏的数值契约迁移。现在 `DitGpuSession::bind_weight_as` 为 F16 明确绑定
RoundedInputF32，保留 F16 输入舍入和 F32 累加；Q8_0 继续 Prepared。
未关闭 refiner、改用 CPU 或修改默认开关，Qwen/Llama 的 Prepared 契约保持不变。

设备回归使用实际 W2 形状 `10240 → 3840`、32 行、F16 权重 1，输入
`-256.0625` 舍入为 `-256`：修复前输出 `-inf`，修复后 grouped 和 tiled
输出均精确为 `-2621440`。该用例显式初始化真实设备，不允许无设备静默跳过。

完整权重复现使用 Apple M3 Max / MoltenVK、`release-fast` + Vulkan + parity-trace，
prompt `A red fox sleeping beneath a pine tree`、seed 42、512×512、8 线程、`--gpu`。
为了定位首次坏值，先把用户的 8 步缩到 2 步；其余参数相同。

| 2 步真实 CLI | 修复前 `78c79b70` | 修复后 |
| --- | --- | --- |
| 首次非有限边界 | 第二步 sigma=0.003，noise refiner 0 中 387 个 NaN，首个索引 492758 | 两步的全部 14 个边界 checkpoint 有限 |
| 后续传播 | noise refiner 1 全部 3932160 个 NaN，flow 全部 65536 个 NaN，退出 1 | 正常退出并写出 512×512 PNG |
| 实际 GPU 范围 | 每步 34 block / 170 projection / 300 dispatch | 相同，没有 CPU 回退 |

原错误日志的 `-inf=65536` 是 NaN 的负号也被计入 infinity；原始二进制 trace
确认 flow 的实际 infinity 数为 0。诊断现在只对 `is_infinite()` 分正负计数，
并以正/负 NaN、正/负 infinity 测试完整错误文本，保留非有限值拒绝。

| 当前回归 | 结果 |
| --- | --- |
| CPU 全库 | 1165 passed / 23 failed / 78 ignored；失败名称不变 |
| Vulkan + parity-trace 全库 | 1264 passed / 28 failed / 138 ignored；没有新增失败，修复原有诊断 fixture 失败 |
| 共享执行、事务、VAE、refiner 设备套件 | 70 passed / 1 failed；仍为原有 AuK ARM fixture |
| Z-Image 原有五项 GPU 正确性 | 5 passed；AdaLN、QKV、W2、融合 norm/modulation 与 ignored attention |
| 构建与格式 | Vulkan + parity-trace lib/bins/examples check、rustfmt、diff check 通过 |

复现模型均使用本机已有文件，未下载或转换权重：

| GGUF | SHA-256 |
| --- | --- |
| z-image-turbo-q8_0.gguf | `39674cec3b98e737276443cbf02f29b0aea616164e465c731f19903b00363cfd` |
| qwen3_4b_f32-q8_0.gguf | `aeaeb1222b858fc98fa01f31bdfdceb8a5b1719a9b91a3909774a85ecf8930fe` |
| pig_flux_vae_fp32-f16.gguf | `7e9b2072ef8d8bde202804362b273a96233e54e4b52c820662cdc70b3e08b27a` |

日志、完整 trace、失败名称对照和 PNG 保留在上述本轮忽略目录的 `zimage-*` 文件。
本次证明数值回归已修复；未据此开放 Auto workload，未做像素 oracle/PSNR 或跨设备
性能准入。用户完整 8 步的 release 命令另行验证，不能把上述 2 步结果算作 8 步。

## main 合并验收（2026-10-09）

合并前分支为 `4994bd22a56a127a7c23199db30f0e979baf0a8a`，同步的 main 为
`9fb52fc9802a5ae8da8ccb4e11bc43de3ccafbcf`（共同祖先为上述实施基线）。
解决 12 个冲突文件，保留 main 的 Jinja、DiffusionPipeline、Qwen-Image-2.1、
AuK 20 层结构和新增 Vulkan projection/VAE 卷积，同时保留共享执行与事务策略。

合并验证修复了两个接口/数值兼容点：

- `VulkanKernel` 转发 `supports_f16_strict`，避免包装器误拒绝原本支持的严格 F16。
  已有 rounding 测试先失败，补上转发后通过。
- F16 shader 同时保留 ARM prepared half 累加与 main 的 AVX2/F16C dot reduction。
  合并时 VAE 明确使用 `F16F32` 契约（本轮拆为 `F16 + RoundedInputF32`），保留 F16 输入舍入及 F32 累加；卷积不继承 ARM prepared
  half 累加。合并后的 VAE 设备测试先失败，修复后通过，继续使用 tiled/direct GPU 路径。

新增 raw hidden-sequence 集成测试，验证逐行输出 RMSNorm 的 raw-bit 对应关系、KV 提交长度，
以及 raw/normalized 两个 API 对强制 Vulkan 的共同拒绝行为。

| 当前合并结果检查 | 结果 |
| --- | --- |
| CPU 库全量 | 1164 pass / 23 fail / 78 ignored |
| Vulkan 库全量（沙箱） | 1251 pass / 28 fail / 132 ignored |
| 严格 F16 与 raw hidden API 定向检查 | 2/2 pass |
| `RMI_SCALAR=1`、parity-trace 定向检查 | 38/38 pass |
| 原有设备套件与 main 新增 VAE 卷积 | 65/65 pass |
| main 新增 AuK projection/cache ignored 测试 | 1 fail，在 main 原始源码同设备复现 |
| CPU lib/bins；Vulkan lib/bins/examples 编译 | pass |
| rustfmt、diff 空白、完整 manifest、全部已提交 SPIR-V | pass |
| 合并 F16 shader 字节重建 | pass |
| 完整 shader 重编译 | fail：未改动的 `softmax.spv` 在 byte 13 不同 |

全量失败名称均在合并前的基线对照中出现，没有新增失败名称。
main 的 AuK ignored 用例无条件要求 F16Dot 提交，但该模式目前要求 AVX2/F16C；
本机 ARM 会回退 CPU，提交数断言失败。没有放宽断言或以 CPU 回退宣称 GPU 验收通过。
VAE 同一用例在 main 原始源码通过，在合并结果也通过。
shader 全量检查已越过原有 grouped-dp4a 差异，当前失败点为 main 原样保留的 softmax。

对照源码由 `git archive 9fb52fc9` 导出。一次共用 target 的对照构建污染了测试缓存，
该轮结果已作废；清除冲突 fingerprint 后重新编译当前工作树，确认最终二进制包含
本分支的 compute 和 raw hidden 回归用例，才运行上述最终 Vulkan/设备检查。

main 带入已锁定的 minijinja/memo-map 依赖；为验证下载的依赖放在仓库
`target/merge-cargo-home`，未修改系统环境。合并记录与日志在本地
`.superpowers/sdd/2026-10-08-unified-compute-dispatch/merge-main/`。
复现上述 Cargo 检查时使用 `CARGO_HOME="$PWD/target/merge-cargo-home"` 与 `--offline --locked`。

**以下既有章节中的真实模型数值、CLI/HTTP 与 release 性能数据采集于合并前；
本次没有重新测量真实模型性能，也没有据旧数据开放新 Auto workload。**
当前设备证据限于 Apple M3 Max/MoltenVK；未验证新增 AVX2/F16C shader 模式的 x86 实机行为。

## 实现范围

| 层次 | 当前行为 |
| --- | --- |
| 策略 | `ComputePolicy::{Cpu,Auto,Vulkan}`；默认 CPU；`--gpu` 是 Auto 别名 |
| CPU 隔离 | 作用域覆盖兼容入口、请求线程和 `ComputePool` worker，退出恢复原策略 |
| Linear | `LinearExecutor` 借用权重、缓存上传，区分 Prepared/Forward/F16Strict；Auto 原子回退整组 |
| 消费者 | Gemma4 公共 projection；AuK 每 render 会话，移除静态缓存与多余 F16→Q8 转换 |
| Dense | Qwen3 与标准 Llama 共享 12 步 recipe；GPU 共用上传、arena、录制、提交、KV delta 与 reset |
| 状态 | 成功后提交 shadow；Auto 整 chunk 重算，强制 Vulkan 返回错误；双重失败保留两个原因 |
| Qwen3.5 | 保持独立 conv/SSM/四轴位置语义，接入相同策略和事务要求 |
| 观测 | `RMI_COMPUTE_TRACE=1` 报实际后端/范围；设备计数涵盖提交、上传、逻辑读写及 buffer 分配 |

权重格式与设备选择分层处理，没有将 Q4/F16/BF16 统一重新量化为 Q8。
局部 linear 卸载不能称作完整 GPU decoder；embedding lookup、tokenizer、采样仍在主机。
强制 CLI/HTTP Vulkan 只接受声明的纯文本 decoder；Qwen3/Llama 要求显式 F16 KV。
不改变原有 F32 KV、采样、线程、batch 或 context 默认值。

新增 Llama Auto 当前固定留在 CPU；原有 Auto 路径保留实验性行为。
标准 Llama 子集拒绝 SWA、softcap、特殊 scale、bias/postnorm、部分/YaRN RoPE 等未实现配置。
ARM prepared F16 shader 匹配 32 路 half 累加，包括半精度 FMA tie；不支持的 half-prefix/F64-tail
在上传前拒绝。Forward/F16Strict 不会冒充 Prepared 语义。

## 环境与真实权重

- Apple M3 Max，macOS，aarch64 NEON；Rust 1.98.1。
- MoltenVK 1.4.2，Vulkan API 1.4.357；真实设备运行需解除执行沙箱的设备访问限制。
- 功能迭代使用 `release-fast`；性能与 CLI/HTTP 使用相同配置的 `release` 构建。
- 无新增依赖；没有下载权重或修改系统环境。

| 文件 | SHA-256 |
| --- | --- |
| Qwen3-0.6B-Q4_0.gguf | `33bcc57074ec7b6eada5a90651ee546ec0c2b271002c22baf9f1b2dd1e8f75cb` |
| Qwen3-0.6B-Q4_K_M.gguf | `ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a` |
| Qwen3.5-0.8B-BF16.gguf | `cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5` |

## 数值、状态和路由检查

真实短 prompt 为“法国的首都是”；Qwen3 输入 IDs 为 `[104328,9370,59975,100132]`，
4 线程、F16 KV、batch 64、context 37，检查 prefill logits 与 32 个 greedy token。
门槛保持 `abs <= 2e-3 + 2e-3 * abs(CPU)`。

| 模型 | 本分支首个失败（logit 0：GPU / CPU） | 原始版本 |
| --- | --- | --- |
| Qwen3 Q4_0（混合 Q4_1/Q6_K） | `2.0631144 / 2.1942825` | 完全相同 |
| Qwen3 Q4_K_M（混合 Q6_K） | `3.221686 / 3.0801702` | 完全相同 |
| Qwen3.5 BF16 | `2.2660627 / 2.25888` | 完全相同 |

这些运行在 logits 门禁失败，**不算 32-token 端到端验收通过**。
原始版本由 `git archive ce17cd5d` 的独立源码副本构建；仅在诊断 example 添加 CPU 计时入口，
未修改其生产路径。

Qwen3 两组真实权重分别通过 CPU 内部与 Vulkan 内部的 batch 1/3/64 比较：132-token prompt，
logits、prompt/decode KV 及 32 个 greedy token 均 raw-bit 相同。
Vulkan prefill 提交数分别为 132/44/3，加 decode 后为 164/76/35。
Qwen3.5 的 80-token prompt 在 CPU 三种 batch 下保持 logits、dense KV、conv/SSM 和 decode 状态完全一致；
初次 GPU batch 1 在提交数检查失败（预期 80，实际 560），原始版本完全复现。
原因是原有 executor 每 16 个 dispatch 分段 flush 以规避驱动错误，而 example 仍假定一次提交。
诊断已改用强制 Vulkan（不允许 fallback），要求 GPU 每 chunk 至少一次提交、CPU 零提交，报告实际计数；
未删除安全 flush，也未改变数值/状态比较门槛。
修正诊断后，Qwen3.5 CPU/Vulkan 各自的 B=1/3/64 全矩阵通过，GPU prefill 提交数
560/189/14，加 32 步 decode 后为 784/413/238；logits、dense KV、conv/SSM、greedy token、decode 状态逐位一致。
这些同后端分块检查与 CPU/GPU 跨后端比较分别记账。

集中真实设备 fixture 共 **64 项通过**，覆盖九格式上传缓存、整组 fallback、F16 raw bits、
Qwen3/Llama dense、Gemma4/AuK、非零前缀/第二 chunk 失败、双错误保留、reset、
CPU/Auto 会话隔离及不确定资源保留。直接 F16 检查覆盖输入宽度 32/256/512/2048 × rows 1/3/64。
Llama 的纯 F16 与混合 Q8_0/Q4_0/Q4_K fixture 通过 32 步 greedy、KV 和 reset；属于合成验证。
另以相同探针分别链接原始和当前无 feature 的 `release-fast` 库：两组真实 Qwen3 权重的
33 个 greedy token、32 步 decode 后的完整 logits raw-bit SHA-256、整个 F16 KV raw-bit SHA-256
及 seq_len=36 均完全相同。探针源码和输出保存在本地日志目录，不作为生产 example 保留。

CLI/HTTP 使用 Qwen3 Q4_K_M、相同 16-token ChatML prompt、4 线程、F16 KV、batch 64、
context 128、temperature 0、关闭 thinking、最多 32 输出 token。
CPU 与强制 Vulkan 各自的 CLI/HTTP 均得到同样的 8-token 文本“法国的首都是**巴黎**。”；
trace 分别确认 CPU 和 resident Vulkan。该检查验证路由一致性，不替代上面的数值门禁。
临时服务只监听 `127.0.0.1`，检查后关闭。

`vk_check` 五种 shape 通过。`vk_ops_check --all-formats --rows 3/64` 均在 quantize tie-even
检查失败（GPU `-21`、CPU `-20`）；基线 rows 3 完全复现。九格式缓存 fixture 的通过
不能覆盖这个完整算子检查的失败。

## 全量与构建检查

| 检查 | 原始版本 | 本分支 | 结论 |
| --- | --- | --- | --- |
| 无 feature 库测试 | 1072 pass / 24 fail / 74 ignored | 1096 / 23 / 74 | 无新增失败名称 |
| Vulkan 库测试（沙箱内） | 1150 / 35 / 110 | 1178 / 30 / 120 | 无新增失败名称 |
| app 定向测试 | — | 155 pass / 3 ignored | 通过 |
| CLI 参数表 | — | 42 pass | 通过 |
| `RMI_SCALAR=1` + parity-trace + compute | — | 30 pass | 独立 scalar 进程通过 |
| 真实设备集中 fixture | — | 64 pass | 通过 |
| benchmark/数值检查 example 单测 | — | 6 pass | 包含非有限 CPU 参照拒绝及分段提交 |

Vulkan 的 RoPE coefficient raw-bit 与五项额外 Z-Image 失败都已在原始版本复现。
沙箱内的设备初始化失败不能代表真实设备不可用；同测试在真实设备通过。
原有 `prepared_group_matches_mixed_format_sequential_bits` 这次没有失败，未认定已修复。

无 feature 库/bins 与 Vulkan 库/bins/examples 编译通过；格式及 diff 空白检查通过。
shader manifest 哈希及所有 SPIR-V 的 `spirv-val` 通过；修改的两个 shader 重编译字节一致。
完整 shader rebuild check 在**未修改**的 `q8_matmul_grouped_dp4a.spv` 第 9 字节失败，
属于本机工具链复现差异，未通过重生成无关二进制掩盖。
无 feature 全 examples 编译仍被原有 `vk_mem_info` 未加 feature guard 的 import 阻断。

## 最终复审与收尾修复

独立全分支复审覆盖 `ce17cd5d..afffc3bd`，未发现新的 Critical/Important 问题，
列出两项 Minor；复审另行通过 34 项无 feature 定向测试和 diff 空白检查。
作者按性能目标将其中的 Llama 中间 chunk 多算 logits 提升为 Important 并修复。

`project_logits` 现在贯通 Llama 共享 CPU、兼容 CPU 与 resident Vulkan 路径，
普通 chunked prefill 只在最后一个 chunk 做输出 norm/投影和 logits 回读；
逐 token 调试入口继续为每个 token 产生 logits。
跳过投影时不再检查或覆盖旧 logits，仍验证本 chunk 的 hidden/KV 有限性才提交进度。

两项 CPU 回归先复现失败：关闭投影仍覆盖 logits；坏的输出权重让第一块就失败，
未保留本应成功的非末尾前缀。修复后 Llama CPU 定向 **6/6 通过**，
覆盖 rows 1/3、标准/带 residual scale 的兼容路径、最终 logits/KV raw-bit 一致、
输出头失败保留前缀，以及不投影时拒绝非有限 hidden。

修复后的真实设备集中测试 **64/64 通过**，其中 Llama 的 11 项全部通过，
包含 Vulkan 不投影/最后投影、失败前缀、F16/混合量化、32 步 decode 与 reset。
全量测试对照原始失败名称：CPU 全量 1096 pass / 23 fail / 74 ignored，
Vulkan 全量 1178 / 30 / 120，均无新增失败；这是基线回归比较通过，并非全量全绿。

**Deferred minor：** Qwen3.5 CPU 直接 single-token decode 在输出 `hybrid_decoder` trace 前返回，
该快速路径缺少逐块后端 trace；推理与状态不受影响，本轮未修改。

性能数据采集于 `afffc3bd`；最后的投影修复仅影响 Llama，未重跑不受影响的 Qwen 性能样本。
由于没有真实 Llama 权重，不能给此修复附上实测加速幅度，也没有据此开放 Auto。
按照执行流程完成一次作者修复与回归，不再派发第二轮复审。

## 性能判据与测量方式

`vk_model_check qwen3|llama --model PATH --benchmark` 预热后交替 CPU→GPU / GPU→CPU，
五对样本报告 prefill、decode、reset+generation 墙钟及设备计数。
正确性失败仍可采集诊断耗时，但输出 `correctness=false`、`threshold_passed=false`，
最终返回失败；任何速度数字都不能据此开放 Auto。

模型哈希扫描在加载前执行，故 cold 项仅表示本进程模型/设备/会话初始化，**没有测 OS 冷页缓存**。
`/usr/bin/time -l` 的 max RSS 是整进程高水位，包含并存的 CPU/GPU 会话；
设备内存项是 Vulkan buffer allocation 高水位，非驱动总显存；UMA 两者不可相加。
host read/write 是逻辑 arena 字节数，不等同 PCIe 传输。计数为设备 context 全局，测量时无并发任务。

新增 Auto 要同时满足正确性、至少五对样本、墙钟中位数改善 ≥10%，并复测相邻形状；
本轮没有组合准入，因此没有性能阈值硬编码到生产选择器。
默认 CPU 与原始版本同配置比较，退化 >5% 需解释并阻止合入，不能用 GPU 数字抵消。

## 实测结果（相同 release 构建配置）

CPU 基线比较：每模型五对独立进程，每进程预热后取五次样本，25 个有效样本/版本。
Q4_0 初始一对可能与 CPU 正确性探针重叠，已整对剔除并补测；最终统计仅用其余五对。

| 权重 | 原始 CPU 墙钟中位数 ms | 当前 CPU ms | 变化 | 原始范围 ms | 当前范围 ms |
| --- | ---: | ---: | ---: | --- | --- |
| q4_0 | 697.517 | 702.132 | +0.66% | 694.815–700.041 | 699.986–716.801 |
| q4_k | 676.144 | 678.382 | +0.33% | 672.251–689.439 | 676.745–683.583 |

两组已测 CPU 均未触及 5% 回归门槛；结论限定于这些固定输入与配置。

CPU/GPU 配对：每组预热、五对交替样本；4-token prefill + 32 步 decode，
batch 上限 64（本例有效 prompt rows=4），context 37，4 线程，F16 KV。

| 权重/后端 | prefill tok/s 中位数 | decode tok/s 中位数 | 墙钟 ms 中位数 |
| --- | ---: | ---: | ---: |
| q4_0 / cpu | 35.830 | 54.927 | 700.096 |
| q4_0 / vulkan | 16.807 | 8.286 | 4106.633 |
| q4_k / cpu | 98.940 | 48.898 | 701.550 |
| q4_k / vulkan | 3.904 | 2.895 | 12084.388 |

GPU 在这两组小 prompt/decode 工作负载分别约为 CPU 的 **5.87 倍 / 17.23 倍耗时**；
正确性也未通过，故不能称作模型加速。没有继续为新 Auto bucket 外推相邻形状。

每个 warm GPU 样本均为 33 次主计算提交、0 transfer 提交、0 次静态上传；
逻辑 host write=165,888 bytes、host read=28,595,712 bytes。CPU 每样本 GPU 提交为 0。

| 权重 | 模型加载 ms | CPU 会话 ms | context 初始化 ms | GPU 会话 ms | 静态上传 bytes / 次数 | buffer 峰值 bytes | 进程 max RSS bytes |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| q4_0 | 46.003 | 0.050 | 23.192 | 48.903 | 376204288 / 310 | 399726560 | 917307392 |
| q4_k | 45.459 | 0.051 | 21.825 | 48.297 | 390753280 / 310 | 414275552 | 944734208 |

冷初始化与 warm 数据分别记录；不能把以上初始化加上 warm 中位数宣称 OS 冷启动端到端时间。
无真实 Llama 产物，所以新 Llama benchmark 入口未取得实机性能数据。

## 未覆盖与后续门禁

- 没有真实 Llama、F16 embedding、Qwen3 Q8_0 或 AuK 产物；这些模型未取得真实端到端证明。
- 没有 x86 AVX2、独显、ANV/RADV/NVIDIA 运行结果；不从 MoltenVK 外推。
- Llama 原有极端 attention 分数下的 CPU 下溢问题未处理；新增 fixture 与 gate/up 回归单独覆盖。
- 真实模型 logits、完整 quantizer 和 shader 重编译基线问题仍需单独修复，随后再运行 Auto 准入。
- 未运行全仓所有 ignored 模型测试、全部 integration/all-targets 或外部 llama.cpp oracle。
- 没有新增 CUDA/Metal/wgpu 生产后端、持久化 autotuner 或全模型图调度。

原始日志与逐阶段 RED/GREEN 证据保存在本地
`.superpowers/sdd/2026-10-08-unified-compute-dispatch/`；下方实施裁决保留理由与代价。

## 复现命令

```bash
cargo check --locked --profile release-fast --lib --bins
cargo check --locked --profile release-fast --features vulkan --lib --bins --examples
cargo test --locked --profile release-fast --lib
cargo test --locked --profile release-fast --features vulkan --lib
RMI_SCALAR=1 cargo test --locked --profile release-fast --features parity-trace --lib compute::
cargo test --locked --profile release-fast --features vulkan --example vk_model_check
cargo test --locked --profile release-fast --features vulkan --lib -- --include-ignored --test-threads=1 compute:: models::llama::trunk::compute_tests batched_linear_device_ qwen3_gpu_chunk_failure qwen3_gpu_second_chunk_failure qwen3_vulkan_prefill_batches_match_bits_kv_and_submissions qwen35_gpu_chunk_failure initializes_with_homebrew_moltenvk gemma4_vulkan_linear models::diffusion::auk::linear
cargo build --release --locked --features vulkan --example vk_model_check
/usr/bin/time -l target/release/examples/vk_model_check qwen3 --model "$MODEL" --benchmark
target/release/examples/vk_model_check qwen3 --model "$MODEL" --cpu-benchmark
target/release/examples/vk_model_check qwen3 --model "$MODEL" --compare-prefill-batches 1,3,64
target/release/examples/vk_model_check qwen35 --model "$QWEN35_BF16" --compare-prefill-batches 1,3,64
```

`MODEL`/`QWEN35_BF16` 指向上表已核对哈希的产物。所有命令按前述结果解释，失败不算通过。

## 继承的失败名称

当前 Vulkan 全量套件以下 30 个失败均在原始版本出现；无 feature 套件是其中 23 项。

```text
models::breeze::tests::main_matrices_require_original_dtype_exact_shape_and_length
models::breeze::tests::main_source_preserves_original_bf16_and_legacy_codec_f32
models::breeze::tests::main_source_rejects_non_bf16_heads_eoi_and_norms
models::clm::tests::matches_reference_golden_vectors
models::clm::tests::rejects_wrong_embedding_width
models::diffusion::z_image::dit::tests::attention_matches_the_ggml_neon_softmax_and_value_reduction
models::diffusion::z_image::dit::tests::default_z_image_rope_uses_the_oracle_paired_sincos_rounding
models::diffusion::z_image::dit::tests::dit_boundaries_reject_non_finite_values
models::diffusion::z_image::dit::tests::silu_mul_inplace_matches_ggml_neon_activation_dit
models::diffusion::z_image::dit::tests::torch_mt19937_recomputes_the_final_sixteen_values
models::diffusion::z_image::text::tests::qwen_attention_softmax_matches_the_pinned_ggml_neon_reduction
models::diffusion::z_image::text::tests::silu_mul_inplace_matches_pinned_ggml_neon_activation
models::diffusion::z_image::vae::tests::silu_matches_pinned_ggml_neon_vector_path
models::funasr::encoder::parity_tests::fsmn_rounds_product_before_adding_like_ggml
models::funasr::encoder::parity_tests::layernorm_accumulates_variance_in_ggml_neon_groups
models::gemma4::trunk::tests::failed_later_gemma4_chunk_preserves_successful_prefix
models::gemma4::trunk::tests::per_layer_bf16_projection_matches_pinned_scalar_dot_bits
models::gemma4::trunk::tests::per_layer_projection_rejects_non_bf16_weight
models::gemma4::trunk::tests::per_layer_projection_rejects_wrong_bf16_storage_length
models::qwen35::trunk::config::tests::qwen35_config_rejects_invalid_tensor_dimensions
models::qwen35::vision::tests::layer_norm_stats_match_ggml_grouped_f32_variance
ops::argmax::tests::all_negative
ops::argmax::tests::large_random
ops::argmax::tests::middle_max
ops::argmax::tests::tail_handles_non_aligned_lengths
ops::argmax::tests::ties_pick_lowest_index
ops::matmul::neon_tests::neon_softmax_matches_ggml_vector_exp_and_f64_sum
ops::rope::tests::rope_neox_inplace_simd_matches_scalar_fallback
vulkan::ops::tests::vulkan_rope_coefficients_match_cpu_dimension_formula
vulkan::tests::initializes_with_homebrew_moltenvk
```

## 实施裁决（Rulings I made）

保留逐阶段裁决原文，包括错误判断可能造成的代价。

- Ruling: Use the existing isolated worktree and create a branch when committing — no checkout duplication is needed — cost if wrong: branch naming changes only.

- Ruling: Establish F16 mode parity alongside the real LinearExecutor in Task 2 — Task 1 has no consumer for LinearMode — cost if wrong: semantics coverage arrives one task later, before any migration.

- Ruling: Correct QuantizedTensor::F32 row metadata at its source, retaining the legacy zero-dimension fallback — validated linear bindings otherwise reject valid F32 matrices — cost if wrong: callers that depended on truncated F32 output need adjustment; run full regression.

- Ruling: Gemma4 native F32 projections stay on the original CPU dot path — generic F32 GPU fallback would change its exact reduction — cost if wrong: fewer local GPU submissions, assess end-to-end before Auto enablement.

- Ruling: AuK creates borrowed F16 kernels once per render scope — storing a borrowed executor inside the model would be self-referential — cost if wrong: small per-render setup cost; no per-projection rebuild or static cache.

- Ruling: The task completion command explicitly excludes the four recorded preexisting Gemma4 failures and reports them separately — preserve unrelated baseline defects, verify all other consumer cases — cost if wrong: inherited BF16/per-layer/late-nonfinite defects remain until separately repaired.

- Ruling: DenseBlockOps carries its backend error as an associated type — CPU String and VulkanError keep their existing messages and failure classification without a lossy round trip — cost if wrong: callers needing one external error type must map once at the session boundary.

- Ruling: The existing injected GPU failure now submits after the first shared dense layer instead of immediately after its KV append — the fixture still writes the same tentative KV range before failing, while production step recording never submits — cost if wrong: injection also executes later stateless operations, so their failures could mask the injected error; current fixture passes.

- Ruling: Qwen3 generation decode now uses the same one-row prefill transaction — the old separate decode retry lost original GPU errors and bypassed CPU state validation — cost if wrong: minor one-row validation overhead; include decode in final paired timings.

- Ruling: Correct the standard Llama batched gate/up inversion while retaining B=1 semantics — otherwise the shared recipe would replicate a proven wrong FFN operation; scalar exp semantics remain batch-specific — cost if wrong: multi-row Llama outputs change, guarded by the B=1 numerical oracle and legacy-bit checks after the fix.

- Ruling: Share the resident arena, uploads and chunk execution by moving the existing Qwen3 implementation into vulkan/dense; retain Qwen-only eligibility and weight adaptation in qwen3 — a second Llama GPU forward would defeat the plan — cost if wrong: shared lifetime/resize regressions; rerun original device fixtures.

- Ruling: Match the ARM prepared F16 reduction in the public Vulkan operator, including half-FMA tie correction, and reject its unimplemented F64 tail before shared linear/dense uploads — a dtype-only choice changed greedy output — cost if wrong: slower F16 kernels and fewer eligible nonaligned widths; benchmark before Auto admission. Other CPU reductions retain tolerance validation, not a claim of universal raw-bit equality.

- Ruling: Keep the stable nonzero Llama GPU fixture and a separate stronger FFN regression — the old CPU F16 attention underflows on the original large-scale fixture, before this migration — cost if wrong: extreme-score CPU attention remains a preexisting limitation, no claim of real Llama-model proof. No Llama artifact is installed.

- Ruling: Record the unchanged grouped-dp4a shader compiler byte mismatch separately from modified shader validation — full shader check fails at original binary byte 9 on this toolchain — cost if wrong: CI/compiler reproducibility remains unresolved; manifest, spirv-val and changed shader rebuilds must pass.

- Ruling: Use a thread-bound, restoring legacy policy scope for existing CLI signatures, plus explicit RuntimeOptions and per-session constructors — threading a new argument through every legacy multimodal/sampling entry would expand scope without migrating them — cost if wrong: a new asynchronous entry must explicitly copy the policy into its worker; server blocking jobs and direct runtime consumers are covered.

- Ruling: Forced CLI Vulkan accepts plain text generation for llama/qwen3/qwen35 and rejects interactive, multimodal, JEV, embedding, profile and bench modes — those routes do not declare a complete migrated execution scope — cost if wrong: callers must use explicit diagnostics or CPU/Auto for those modes. HTTP text routes reject image input and forced JEV; constructor errors propagate.

- Ruling: Keep Llama CLI prompt/sampling code and use its shared session only for forced resident execution; original CPU CLI stays unchanged — changing generation defaults is out of scope — cost if wrong: legacy CPU CLI remains a separate driver, with the same operator recipe used by stateful sessions and GPU.

- Ruling: Counters measure device-wide accepted submissions, static uploads, logical host/arena bytes and buffer allocation peak; process RSS is measured externally — mapped UMA copies are not PCIe transfers and allocations are not total driver memory — cost if wrong: concurrent workloads contaminate per-run deltas; benchmark in isolation.

- Ruling: Preserve existing experimental Vulkan support and report the reproduced baseline logits/quantizer failures instead of broadening this refactor into a numerical backend repair — do not loosen tolerances or admit any new Auto workload — cost if wrong: explicit and legacy Auto Vulkan remain experimental and may differ from CPU on real weights.

- Ruling: Collect paired diagnostic timings even when correctness fails, but keep the failing exit status and force threshold_passed=false — this separates throughput evidence from eligibility — cost if wrong: timings must never be cited as validated model acceleration.

- Ruling: Update the Qwen35 diagnostic to use forced Vulkan and permit the existing safety flushes while requiring at least one submission per chunk and zero CPU submissions — do not remove the driver-correctness workaround to satisfy a stale one-submit assertion — cost if wrong: diagnostics report, rather than tightly cap, submission count; state/logit/greedy checks remain exact within each backend.

- Ruling: Keep the local validation workspace and raw failure logs after review — full-suite and hardware gates have inherited failures, and their exact baseline evidence remains useful for follow-up — cost if wrong: extra local disk use; no generated artifacts are committed.

- Ruling: Mark Task 8 implementation/measurement complete with the acceleration objective unmet — both tested GPU workloads are slower and fail baseline numerical gates; the plan explicitly permits retaining the unified API and forced diagnostics without Auto admission — cost if wrong: users gain execution consistency, not a promised speedup, until later kernel/numerical work is validated.

- Final: Ruling: Regrade the Llama unused intermediate logits finding as Important for this performance refactor — long prefill pays unnecessary vocabulary projection and GPU readback on every chunk; preserve the existing project_logits contract through all Llama paths — cost if wrong: added flag plumbing and final-hidden validation overhead; real-model timing benefit remains unmeasured.

- Final: Ruling: The review set aside reproduced logits, quantizer, shader rebuild and whole-suite baseline failures — keep their failing status and prohibit new Auto admission; this branch does not claim to repair them — cost if wrong: existing experimental GPU numerical differences and baseline test failures remain visible to users.

- Final: Ruling: The review set aside absent real Llama/AuK artifacts and x86/discrete/other-driver coverage — fixture results are bounded to the recorded Apple/MoltenVK environment — cost if wrong: portability and real-model failures may remain undiscovered; do not claim those gates passed.

- Final: Ruling: The review did not quantify the extra Llama projection cost — remove the proven redundant work and verify state/logit equivalence without publishing an inferred speedup — cost if wrong: the actual benefit may be small; no Auto rule can be enabled from this fix.
