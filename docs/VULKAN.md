# Vulkan GPU 后端（实验性）

状态：**实验性**。`--features vulkan` 编译，`--gpu` 启用；缺少任一项都保持原 CPU 行为。

macOS 会自动查找系统 Loader，以及 Homebrew 的 `/opt/homebrew/lib/libvulkan.dylib` 和
`/usr/local/lib/libvulkan.dylib`。`VK_ICD_FILENAMES`、`VK_DRIVER_FILES` 和
`DYLD_LIBRARY_PATH` 只用于排障，不是正常启动的必需配置。

## 支持范围

- dense、Neox RoPE、无 QKV bias 的 Qwen3 Q8_0、Q4_0、Q4_1、Q4_K、Q6_K 和 F16 模型支持完整 token Vulkan 执行。
- Qwen3.5 BF16 文本模型使用独立 executor，覆盖 dense attention、recurrent convolution/SSM、
  mRoPE、FFN 和 logits；BF16 matmul 权重与 F32 辅助张量均在 Vulkan 路径执行。
- 权重、F32 activation 和 GPU KV cache 常驻设备；每个 token 只提交一次 command buffer、
  等待一次 fence。embedding lookup 和 greedy sampling 仍在 CPU；提交成功后，Qwen3 同步 F16
  shadow KV，Qwen3.5 同步 F32 shadow KV 与 recurrent state。
- `text_encode` 对整模符合资格、标准递增位置的模型逐 token 返回最终 RMSNorm hidden row，
  不录制 logits matvec；初始化或执行失败时丢弃 GPU 结果并用原 CPU 路径重算完整序列。
- Vulkan token 失败时从上一个已提交 KV 状态在 CPU 重算；不符合资格的模型直接使用 CPU，
  不会静默混用不支持的 Vulkan 算子。
- Q5_K 目前只完成合成 kernel parity，尚未纳入端到端模型支持矩阵；同一组 gate/up 权重格式不一致，
  或 Qwen3.5 存在未录制算子时，模型整体回退 CPU。

## 架构

- **完整 token 提交**：每层的 RMSNorm、动态 Q8_0 activation 量化、Q/K/V、RoPE、KV 写入、
  attention、FFN 和 residual add 依次录入同一 command buffer，最终 logits 后统一提交。
- **F16 权重**：activation 先按 CPU contract 舍入为 F16，shader 从 `uint` storage buffer
  解包权重并复现 ARM64 FP16 累加/归约顺序，不要求 `storageBuffer16BitAccess`。
- **BF16 权重**：shader 从 `uint` storage buffer 解包 16-bit lane，并按
  `uintBitsToFloat(bits << 16)` 还原 BF16，不要求 16-bit storage feature。
- **Qwen3.5 token 事务**：dense KV delta、recurrent convolution state 和 SSM state 只在
  GPU token 成功后一起提交到 CPU shadow；失败 token 从上一个完整提交点在 CPU 重算。
- **常驻资源**：模型权重只上传一次；session arena、activation、完整 GPU KV 和 token delta
  在 session 创建时分配，算子之间不回传 activation。
- **设备优选**：先按 shader 的 workgroup / shared-memory 要求过滤，再按
  discrete > integrated > virtual > CPU 排序；候选初始化失败时继续尝试下一设备。
  baseline 不要求 Vulkan 1.3、`shaderInt64` 或整数点积，整数点积可用时选用 dp4a，
  否则使用 baseline pipeline。
- **预热**：上下文创建后立即跑一次 32×32 dummy matmul，吸收驱动首次 dispatch 的 JIT。
- **看门狗**：每次 GPU 调用有可配置超时（`RUST_GPU_TIMEOUT_MS`，默认 5 s；fence 等待
  内层 60 s）。超时/错误时标记 GPU broken → 该次 matmul 由线程 0 全量 CPU 重算
  （其余线程已返回，必须全量而非按行区间，否则留下未计算的行）→ 后续调用走 CPU。

## 跨厂商统一验收流程

MoltenVK、Intel ANV、AMD RADV 和 NVIDIA 使用同一套命令，不为某个驱动放宽误差门限或删减
case。先把下面五个变量指向与模型清单 SHA-256 一致的本地文件；驱动选择只通过运行环境完成，
不修改命令本身。

```bash
VULKAN_Q8_0_MODEL=/path/to/Qwen3-0.6B-Q8_0.gguf
VULKAN_Q4_0_MODEL=/path/to/Qwen3-0.6B-Q4_0.gguf
VULKAN_Q4_K_M_MODEL=/path/to/Qwen3-0.6B-Q4_K_M.gguf
VULKAN_F16_EMBED_MODEL=/path/to/Qwen3-Embedding-0.6B-f16.gguf
VULKAN_QWEN35_BF16_MODEL=/path/to/Qwen3.5-0.8B-BF16.gguf

vulkaninfo --summary
bash scripts/vulkan-shaders.sh check
cargo fmt --check
cargo check --locked --features vulkan --lib
cargo check --locked --features vulkan --bin rust-model-inference
cargo check --locked --features vulkan --bin server
cargo check --locked --features vulkan --examples

cargo run --release --locked --features vulkan --example vk_check
cargo run --release --locked --features vulkan --example vk_ops_check -- \
  --formats q4_0,q4_1,q4_k,q5_k,q6_k,f16,bf16

cargo run --release --locked --features vulkan --example vk_model_check -- \
  qwen3 --model "$VULKAN_Q8_0_MODEL"
cargo run --release --locked --features vulkan --example vk_model_check -- \
  qwen3 --model "$VULKAN_Q4_0_MODEL"
cargo run --release --locked --features vulkan --example vk_model_check -- \
  qwen3 --model "$VULKAN_Q4_K_M_MODEL"
cargo run --release --locked --features vulkan --example vk_model_check -- \
  embedding --model "$VULKAN_F16_EMBED_MODEL"
cargo run --release --locked --features vulkan --example vk_model_check -- \
  qwen35 --model "$VULKAN_QWEN35_BF16_MODEL"

cargo run --release --locked --features vulkan --example vk_model_check -- \
  qwen3 --model "$VULKAN_Q8_0_MODEL" --benchmark
```

`vk_check` 必须完成五种 shape，完整 `vk_ops_check` 必须覆盖上面列出的全部权重格式；Q5_K
在这里仍只代表合成 kernel parity。五个 `vk_model_check` 都成功且 benchmark 输出五轮交替样本
及中位数后，才可把对应硬件行标成“已验证”。仓库级
`cargo test --all-targets --locked --features vulkan` 也要运行，但其与硬件验收分开记账，避免既有
集成测试编译失败掩盖或冒充 Vulkan 结果。

验收模型清单：

| 模型 | bytes | SHA-256 |
|---|---:|---|
| Qwen3-0.6B-Q8_0.gguf | 639,446,688 | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` |
| Qwen3-0.6B-Q4_0.gguf | 382,156,480 | `33bcc57074ec7b6eada5a90651ee546ec0c2b271002c22baf9f1b2dd1e8f75cb` |
| Qwen3-0.6B-Q4_K_M.gguf | 396,705,472 | `ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a` |
| Qwen3-Embedding-0.6B-f16.gguf | 1,197,629,632 | `421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340` |
| Qwen3.5-0.8B-BF16.gguf | 1,516,744,736 | `cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5` |

## 硬件矩阵（2026-09-04）

| Vulkan 栈 | GPU | shader / 五 shape | 完整算子 | 五模型 | 交替基准中位数（CPU → GPU，prompt/decode） | 状态 |
|---|---|---|---|---|---|---|
| MoltenVK 1.4.2，driver 0.2.2210 | Apple M3 Max | 通过 / 5/5 | 全部通过；Q5_K 仅合成 | 5/5 | 20.704/20.401 → 6.048/6.031 tok/s | **已验证** |
| Intel ANV | 未采集 | 未运行 | 未运行 | 未运行 | 未运行 | 未验证 |
| AMD RADV | 未采集 | 未运行 | 未运行 | 未运行 | 未运行 | 未验证 |
| NVIDIA Vulkan | 未采集 | 未运行 | 未运行 | 未运行 | 未运行 | 未验证 |

Apple 行的 `vulkaninfo --summary` 为 Vulkan API 1.4.357、integrated GPU、driver ID
`DRIVER_ID_MOLTENVK`。本次五模型结果为：Q8_0、Q4_0、Q4_K_M 的 prefill 最大绝对误差均为
0，且各自 32/32 greedy token 相同（submission 分别为 37、36、36）；F16 embedding 的三个
向量最大绝对误差为 `4.267e-4`、`4.156e-4`、`6.245e-4`，排序一致且共 26 submissions；
Qwen3.5 BF16 prefill 最大绝对误差为 `2.956e-5`，32/32 token 相同，共 36 submissions。

## Qwen3 Q8_0 实机门禁（2026-09-04）

设备：Apple M3 Max；Vulkan Loader/API 1.4.357；MoltenVK 1.4.2，driver 0.2.2210。

模型：`/Users/gouzi/Documents/git/rust-model-inference/models/Qwen3-0.6B-Q8_0/Qwen3-0.6B-Q8_0.gguf`
（639,446,688 bytes，SHA-256
`9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`）。

```bash
cargo run --release --locked --features vulkan --example vk_model_check -- \
  qwen3 \
  --model /Users/gouzi/Documents/git/rust-model-inference/models/Qwen3-0.6B-Q8_0/Qwen3-0.6B-Q8_0.gguf
```

固定 prompt `法国的首都是`（5 tokens）、F16 shadow KV、4 CPU threads、temperature 0：
完整 prefill logits 在 `abs <= 2e-3 + 2e-3 * abs(cpu)` 门限内，实测最大绝对/相对误差均为
0；32/32 greedy token ID 相同；5 个 prompt token 加 32 个 decode token 共 37 次 Vulkan
submission。

## Qwen3 F16 embedding 实机门禁（2026-09-04）

模型：`/Users/gouzi/Documents/git/rust-model-inference/models/qwen-embedding/Qwen3-Embedding-0.6B-f16.gguf`。

```bash
cargo run --release --locked --features vulkan --example vk_model_check -- \
  embedding \
  --model /Users/gouzi/Documents/git/rust-model-inference/models/qwen-embedding/Qwen3-Embedding-0.6B-f16.gguf
```

三个固定文本先在 CPU 计算完整 hidden rows，再启用 Vulkan 通过同一 `text_encode` API 计算；
均取最后一行并按相同 F32/F64 contract 做 L2 归一化。全向量满足
`abs <= 2e-3 + 2e-3 * abs(cpu)`，查询对两个文档的 cosine 排序相同；26 个输入 token
对应 26 次 Vulkan submission。

## Qwen3.5 BF16 实机门禁（2026-09-04）

设备：Apple M3 Max（MoltenVK）。

模型：`/Users/gouzi/Documents/git/rust-model-inference/models/qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf`
（1,516,744,736 bytes，SHA-256
`cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5`）。

```bash
cargo run --release --locked --features vulkan --example vk_model_check -- \
  qwen35 \
  --model /Users/gouzi/Documents/git/rust-model-inference/models/qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf
```

固定 prompt 为 4 tokens；prefill logits 满足
`abs <= 2e-3 + 2e-3 * abs(cpu)`，实测最大绝对误差 `2.956e-5`、最大相对误差
`1.830e0`（相对误差峰值对应接近零的参考值）；32/32 greedy token ID 相同；4 个
prompt token 加 32 个 decode token 共 36 次 Vulkan submission。输出格式摘要为
`matmul={BF16};auxiliary={F32};backend=vulkan`。

## 交替基准

```bash
cargo run --release --locked --features vulkan --example vk_model_check -- \
  qwen3 \
  --model /Users/gouzi/Documents/git/rust-model-inference/models/Qwen3-0.6B-Q8_0/Qwen3-0.6B-Q8_0.gguf \
  --benchmark
```

一次 CPU/GPU warmup 后，按 CPU→GPU 交替采五轮；单位均为 tokens/s：

| 样本 | CPU prompt | CPU decode | Vulkan prompt | Vulkan decode |
|---:|---:|---:|---:|---:|
| 1 | 20.742 | 20.535 | 6.057 | 6.031 |
| 2 | 20.686 | 21.217 | 5.970 | 6.032 |
| 3 | 20.979 | 17.800 | 6.060 | 5.791 |
| 4 | 17.683 | 20.251 | 6.048 | 6.046 |
| 5 | 20.704 | 20.401 | 6.042 | 6.023 |
| **中位数** | **20.704** | **20.401** | **6.048** | **6.031** |

prompt speedup 0.292×，decode speedup 0.296×，`acceleration=false`。当前 M3 Max 上的
MoltenVK 路径是正确性后端，不宣称比 4-thread CPU 更快。

## 仓库测试边界（2026-09-04）

`cargo test --all-targets --locked --features vulkan` 在运行测试前以 101 退出，原因是四个既有
集成测试没有跟上当前公开接口：

- `tests/gemma4_reference.rs` 导入不存在的 `app::run_gemma4` 和 `Gemma4Request`；
- `tests/parity_trace.rs` 未启用 `parity-trace` feature，却直接引用受该 feature 保护的模块；
- `tests/q8_0_parallel_matmul.rs` 导入已不存在的 `ops::matmul_q8_0_quantized`。
- `tests/quantized_inference.rs` 仍使用已移除的 `Q4_KWeight`、旧版单参数 kernel 构造器和
  缺少 Q8_K 输入参数的 `forward_prepared` 调用，共产生 10 个编译错误。

因此不宣称全仓测试通过；该结果与本页明确列出的 shader、build、合成算子和五个实模门禁
分开报告。

## 已知问题与调试开关

- **ANV/Meteor Lake 偶发 wedge**：层 matmul dispatch 间歇性阻塞在驱动内部
  （`vkWaitForFences` 超时不生效）。看门狗超时后放弃该调用并 CPU 重算，进程不再挂死；
  被放弃的线程可能在驱动内持续自旋（占用 1 核直到进程退出）。
- `RUST_GPU_TRACE=1`：打印每次 dispatch 的序号/形状/耗时。
- `RUST_GPU_MAX_ROWS=<n>`：超过 n 行的 matmul 回退 CPU（0 = 全 CPU）。
- `RUST_GPU_TIMEOUT_MS=<n>`：单次 GPU 调用看门狗超时（默认 5000）。
- 正确性基准：`cargo run --release --features vulkan --example vk_check`。它逐行比较
  GPU 与 CPU 标量参考，覆盖 `(1024,1024)`、`(1024,3072)`、`(3072,1024)`、
  `(1024,151936)` 和 `(16384,32)`，判定条件为
  `abs(gpu - cpu) <= 1e-4 + 1e-4 * abs(cpu)`；Vulkan 错误、非有限输出或越界都会以
  非零状态退出。`vk_bench` 是独立吞吐基准。
- shader 唯一源码位于 `shaders/glsl/`；运行 `bash scripts/vulkan-shaders.sh update`
  重新生成，运行 `bash scripts/vulkan-shaders.sh check` 校验源码、SPIR-V 和 manifest。
- wgpu 后端（`--features wgpu`）当前未接入新分发路径，保持 CPU。
