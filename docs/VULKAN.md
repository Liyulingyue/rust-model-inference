# Vulkan GPU 后端（实验性）

状态：**实验性**。`--features vulkan` 编译，`--gpu` 启用（不传则纯 CPU，行为与历史版本一致）。
测试环境：Intel Meteor Lake-P 核显 + ANV（Mesa）驱动，Qwen3-0.6B-Q8_0。

## 支持范围

- **仅 Q8_0 matmul** 走 GPU：`Q8Kernel::forward_prepared` → `parallel_rows` → GPU
  （一次 dispatch 覆盖全部行）。embedding lookup、RMS norm、RoPE、attention、silu、采样均在 CPU。
- 其他量化类型（Q4_K/Q6_K/…）与非 Q8_0 模型自动走 CPU 路径。
- 形状限制：`n_in ≤ 16384`（shader 共享内存 16 KiB 输入暂存），超出自动 CPU 回退。

## 架构（v2 重写，2026-08）

- **权重常驻**：首次 matmul 按 `(data_ptr, len)` 把 mmap 权重切片上传到持久映射缓冲并缓存，
  之后全部命中（旧实现每次调用新建/销毁 4 个缓冲，是 0.7 t/s 的根因之一）。缓冲按 tensor
  字节长度 + 16 B 填充分配（shader 对末行末块的成对 word 预读需要）。
- **持久 IO 缓冲**：输入 Q8 / scale / 输出按需增长、常驻映射，每次调用只 memcpy。
- **单线程提交**：`pool.compute` 闭包中 `ith == 0` 提交 GPU dispatch，其余线程直接返回。
  *调用线程拥有 fence 完成点*——trunk 的 element-wise 后处理（silu 等）必须在拥有完成的
  线程上执行（见下"并发所有权"）。
- **设备优选**：先按 shader 的 workgroup / shared-memory 要求过滤，再按
  discrete > integrated > virtual > CPU 排序；候选初始化失败时继续尝试下一设备。
  baseline 不要求 Vulkan 1.3、`shaderInt64` 或整数点积，整数点积可用时选用 dp4a，
  否则使用 baseline pipeline。
- **预热**：上下文创建后立即跑一次 32×32 dummy matmul——驱动首次 dispatch 需要 JIT
  （Meteor Lake 实测 >5 s），不预热会触发看门狗误放弃首次真实 matmul。
  预热在 `OnceLock::get_or_init` 闭包**内部**执行（曾经放在外面，输掉
  `WARMED.swap` 的线程会在 JIT 完成前 dispatch，产生 8/8 全 wedge）。
- **看门狗**：每次 GPU 调用有可配置超时（`RUST_GPU_TIMEOUT_MS`，默认 5 s；fence 等待
  内层 60 s）。超时/错误时标记 GPU broken → 该次 matmul 由线程 0 全量 CPU 重算
  （其余线程已返回，必须全量而非按行区间，否则留下未计算的行）→ 后续调用走 CPU。

## 并发所有权（关键设计约束）

CPU 路径中，每个线程的 element-wise 后处理（silu）读自己行的 matmul 结果——数据依赖
天然成立。GPU 路径中 matmul 是单线程 fence 提交，没有"每行归属者"，因此：

- GPU 激活时，**silu 必须由线程 0 全量执行**（`ops::gpu_matmul_active()` 判断），
  其余线程跳过；GPU 关闭时保持原 per-thread 分区。
- 同理适用于所有"matmul 闭包内的后处理"——新增 GPU 后端时逐处检查。

## 性能（Qwen3-0.6B-Q8_0, MTL iGPU, 18 CPU / 8 线程池）

| 路径 | 生成速度 |
|---|---|
| CPU（AVX2, 8 线程） | ~47 t/s |
| GPU v1（旧实现：每行区间新建缓冲+上传+提交） | 0.7 t/s |
| GPU v2（权重常驻 + 单次 dispatch） | ~5.5 t/s |

剩余差距的全部来源是**每次 dispatch 的驱动固定开销**（ANV 提交 + fence ≈ 250-600 µs；
纯计算只需数 µs）。0.6B 每 token 约 197 次 dispatch → 开销 ~120 ms。加速路径：

1. 合并 dispatch：QKV 三合一（同输入）、gate/up 二合一 → 每层 7→4 次；
2. 整层/整 token 批量提交（一个 command buffer 多个 dispatch，一次 fence）；
3. element-wise 后处理上 GPU，消除 CPU↔GPU 数据往返；
4. dp4a（设备支持 `shaderIntegerDotProduct` 时启用）提升 shader 吞吐。

前三项完成后 GPU 才可能在 0.6B 上超过 CPU；更大的模型因开销占比下降会先受益。

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
