# Vulkan GPU 后端（实验性）

状态：**实验性**。`--features vulkan` 编译，`--gpu` 启用；缺少任一项都保持原 CPU 行为。

macOS 会自动查找系统 Loader，以及 Homebrew 的 `/opt/homebrew/lib/libvulkan.dylib` 和
`/usr/local/lib/libvulkan.dylib`。`VK_ICD_FILENAMES`、`VK_DRIVER_FILES` 和
`DYLD_LIBRARY_PATH` 只用于排障，不是正常启动的必需配置。

## 支持范围

- dense、Neox RoPE、无 QKV bias 的 Qwen3 Q8_0、Q4_0、Q4_1、Q4_K、Q6_K 和 F16 模型支持完整 token Vulkan 执行。
- 权重、F32 activation 和 GPU KV cache 常驻设备；每个 token 只提交一次 command buffer、
  等待一次 fence。embedding lookup 和 greedy sampling 仍在 CPU，提交成功后同步 F16 shadow KV。
- `text_encode` 对整模符合资格、标准递增位置的模型逐 token 返回最终 RMSNorm hidden row，
  不录制 logits matvec；初始化或执行失败时丢弃 GPU 结果并用原 CPU 路径重算完整序列。
- Vulkan token 失败时从上一个已提交 KV 状态在 CPU 重算；不符合资格的模型直接使用 CPU，
  不会静默混用不支持的 Vulkan 算子。
- Q5_K 和 BF16 权重尚未接入完整 Vulkan 模型路径；同一组 gate/up 权重格式不一致时，模型整体回退 CPU。

## 架构

- **完整 token 提交**：每层的 RMSNorm、动态 Q8_0 activation 量化、Q/K/V、RoPE、KV 写入、
  attention、FFN 和 residual add 依次录入同一 command buffer，最终 logits 后统一提交。
- **F16 权重**：activation 先按 CPU contract 舍入为 F16，shader 从 `uint` storage buffer
  解包权重并复现 ARM64 FP16 累加/归约顺序，不要求 `storageBuffer16BitAccess`。
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
| 1 | 21.176 | 17.748 | 5.944 | 6.030 |
| 2 | 21.136 | 17.949 | 5.977 | 5.999 |
| 3 | 20.666 | 17.457 | 5.964 | 5.962 |
| 4 | 21.637 | 17.815 | 6.001 | 6.033 |
| 5 | 19.659 | 18.023 | 5.956 | 5.897 |
| **中位数** | **21.136** | **17.815** | **5.964** | **5.999** |

prompt speedup 0.282×，decode speedup 0.337×，`acceleration=false`。当前 M3 Max 上的
MoltenVK 路径是正确性后端，不宣称比 4-thread CPU 更快。

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
