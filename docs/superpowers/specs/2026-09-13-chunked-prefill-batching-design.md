# 单请求 Chunked Prefill Batching 设计

## 状态

- 日期：2026-09-13
- 状态：设计已在对话中确认，等待书面审阅
- 目标模型：Qwen3、Qwen3.5、Gemma4
- 目标后端：CPU，以及各模型当前可用范围内的 Vulkan

## 背景

当前生成路径没有统一利用 prompt 的多行计算：

- Qwen3 把 prompt token 和 decode token 放在同一个逐 token 循环中，每个 prompt row 都执行完整 forward 和 vocab projection。
- Qwen3.5 的 CPU 接口能接收多个 token，但主要 projection 仍按 token 调用 matvec；Vulkan prefill 会退回 CPU。
- Gemma4 的 `forward_rows` 逐行调用完整 `forward_row`。
- `Kernel::forward_batched` 及其常用实现仍是逐行循环，没有减少量化次数或线程池同步次数。

因此，prompt processing 实际接近逐 token 推理。设计目标是在不改变 decode 数值语义、不增加外部加速库、不引入通用计算图 runtime 的前提下，把单个请求的 prompt 按固定大小分块执行。

## 范围

### 包含

- 单请求 prompt 的 chunked prefill。
- Qwen3、Qwen3.5、Gemma4 的 CPU 路径。
- Qwen3、Qwen3.5 已支持配置的完整 Vulkan prefill。
- Gemma4 对通用 Vulkan batched matmul 的复用；attention、KV 和模型控制流仍由 Gemma4 自身负责。
- CLI、server 启动配置及现有库级生成入口中的 batch size 传递。
- chunk 级原子状态提交、GPU 整块回退、逐位正确性验证和独立 pp/tg benchmark。

### 不包含

- 多请求 continuous batching、请求调度器、动态合批或 paged KV。
- 新的统一 tensor/graph runtime。
- BLAS、Accelerate、MKL、oneDNN 等外部加速库。
- 新增 Vulkan 模型资格范围，例如为当前不支持的 MoE、deepstack 或权重格式补齐完整 GPU 实现。
- 运行时自动调参；batch size 由一个显式参数控制。
- 新的 tiled SIMD GEMM。只有完成本设计后，benchmark 仍证明 dot kernel 是主瓶颈时再单独设计。

这里的“覆盖 Vulkan”指：Qwen3/Qwen3.5 在当前 Vulkan eligibility 内完成整模型 chunk prefill；Gemma4 复用通用 Vulkan batched matmul，不在本变更中复制一套 Gemma4 专属 GPU runtime。

## 方案选择

### 方案 A：只在模型外层按 chunk 包装现有逐行 forward

改动最小，但内部仍会逐 token 量化、逐 token 唤醒线程池，也不会减少 Vulkan submission。它只改变循环位置，不解决性能根因。

### 方案 B：引入统一计算图和通用 batched runtime

长期扩展性最高，但会同时重写模型执行、状态管理和后端接口。对当前三个模型而言，复杂度、验证面和回归风险都过大。

### 采用方案：共享 batched linear primitive + 模型专属 causal prefill

只抽取三个模型确实共享的热点：多行 projection。attention、位置编码、KV、Gemma4 shared-KV 和 Qwen3.5 recurrent state 继续留在模型实现中。

```text
prompt rows
  -> chunks of at most prefill_batch_size
  -> batched projections
  -> model-specific causal attention / recurrent scan
  -> atomically commit chunk state
  -> project only the final prompt row to vocab logits
  -> existing single-token decode loop
```

## 外部契约

新增：

```text
--prefill-batch-size <N>
```

规则：

- `N` 必须大于等于 1。
- 未指定时使用 `64`。
- `N=1` 强制 prompt 走现有单行基线路径。
- 最后一个不足 `N` 的 chunk 按实际长度执行，不 padding。
- 开始 prefill 前先校验整个输入能否放入剩余 KV capacity；不能时沿用现有容量错误且不提交任何 prompt row。容量足够时，最后一个 chunk 只按剩余 prompt 行数执行。

`CliOptions` 使用可选字段，以避免其现有 `Default` 调用点把有效值默认为 0；解析完成后在一处转换为有效的 `usize`。主 CLI 和 `rust-model-server` 共用该启动参数。server 不增加非标准的 OpenAI 请求字段；同一 server 实例的请求使用启动时配置。

有效值继续传入现有模型级入口：Qwen3 的 generate options/session、Qwen3.5 的生成/forward 调用以及 `Gemma4Request`。库调用者不经过 CLI 时也能显式选择 batch size；库级默认值同样是 64。

## CPU batched linear primitive

不修改公开 `Kernel` trait，也不依赖当前只是逐行循环的 `forward_batched`。新增 crate-private 的 prepared-rows 调用路径，继续复用现有：

- `ComputePool`
- `quantize_q8_0_into`
- Q8_K preparation
- `Kernel::forward_prepared`
- F32、F16、BF16 和既有量化 kernel

每个 projection 的执行顺序为：

1. 对 chunk 的输入 rows 一次性准备连续的 F32/Q8_0/Q8_K row-major buffers。
2. Q/K/V 共享同一份已准备输入；gate/up 同样共享。
3. 每个 projection 只调用一次 `pool.compute`。
4. 每个 worker 固定负责 weight output-row 分区，在该分区内依次处理所有 token rows。
5. 每个 output element 仍调用原有 `forward_prepared` dot 顺序，不能为 batch 改变单个 dot 的浮点累加顺序。

scratch 由 session 按 `prefill_batch_size` 分配或复用，大小只能与 chunk size 成正比，不能与完整 prompt 长度成正比。batch size 为 1 时直接使用原有单行 scratch 和调用路径。

## 模型执行

### Qwen3

prompt assembler 先产生与当前路径相同的 token/raw-embedding rows、绝对 position 和 deepstack 输入。CPU chunk forward 在每层执行：

1. 多行 RMSNorm。
2. batched Q/K/V projections，共用 prepared rows。
3. 逐 row 的 QK norm 和 RoPE，使用 `base_position + row_index`。
4. 将本 chunk 的 K/V 写入 tentative cache 区域。
5. 对每个 row 执行 causal attention；它可以看到已提交前缀以及本 chunk 中不晚于当前 row 的 K/V。
6. batched output、gate/up 和 down projections。

MoE、multimodal position/deepstack 等已有 CPU 语义保持不变；不满足当前 Vulkan eligibility 的输入继续在进入 chunk 前选择 CPU。只有 prompt 最后一个 row 执行最终 vocab projection。decode 保留现有单 token forward。

chunk 成功后才把 `KvState.seq_len` 从 `base_position` 推进 `rows`。tentative cache 物理内容无需清零；在逻辑长度未推进时不可见，同一 chunk 的 CPU 重试会覆盖它。

### Qwen3.5

dense attention 层使用与 Qwen3 相同的 batched projection 和 causal KV 规则。recurrent 层拆成：

- 可并行的 input projections：按 rows batch。
- 有顺序依赖的 conv/SSM scan：从 chunk 起始 state 开始，严格按 row 顺序推进到 chunk-local next-state。
- 可并行的 output projections：按 rows batch。

全 chunk 成功后同时提交 dense KV、conv state、SSM state 和 sequence length。失败时丢弃 chunk-local next-state，原 session state 不变。

### Gemma4

`forward_rows` 将 assembled rows 按 chunk 交给新的模型内 prefill 路径。每层保持既有 Gemma4 契约：

- per-layer token embedding 和 projection。
- shared base-KV 映射。
- SWA/full-attention 选择。
- Q/K/V norm、RoPE、post norms、per-layer gates 和 output scale。
- 最终 logit softcap。

base-KV layer 对本 chunk 追加 tentative K/V，依赖层仍通过 `kv_source_layer` 读取同一来源。任一步失败时，把所有本 chunk 修改过的 base-KV `keys`/`values` truncate 到进入 chunk 前的长度，并保持 `seq_len` 不变。

Gemma4 不新增完整专属 Vulkan executor。其 batched projection 通过通用 Vulkan matmul 执行，模型层顺序、attention 和回滚仍由 Gemma4 CPU 控制流负责。因此，Gemma4 的 Vulkan 验收关注每个 projection 从逐 row dispatch 降为逐 chunk dispatch，而不是要求整模型每个 chunk 只有一次 submission。

## Vulkan 执行

GLSL 和对应 SPIR-V 在现有 shader 体系内更新，不新增 shader framework。通用 batched matmul 的映射为：

```text
workgroup.x = output row
workgroup.z = token_row * grouped_weight_count + weight_slot
```

push constants 或等价参数增加实际需要的：

- `rows`
- `base_position`
- input/output row stride
- grouped weight count/slot

quantize、RMSNorm、QK norm/RoPE、KV write 和 attention shader 接收 rows 与 base position。Qwen3.5 recurrent shader 在一个 state lane 内按 token row 顺序扫描，不能跨 row 并行破坏状态依赖。

Qwen3/Qwen3.5 的完整 Vulkan session 为一个 chunk 录制并提交一个 command buffer。GPU 结果和 CPU shadow state 都成功后才提交逻辑长度。

`TokenCommitState` 从单个 `pending: bool` 扩展为 pending chunk：

- `begin(base_position, rows)` 校验没有未完成 chunk、position 等于 committed length、rows 大于 0 且不越 capacity。
- `commit()` 一次增加 pending rows。
- `abort()` 丢弃 pending rows，不改变 committed length。

GPU chunk 失败时，禁止保留部分 GPU 结果或从中间 token 继续。session 从相同 base position 用 CPU 重算整个 chunk；成功后再同步提交 CPU shadow 和逻辑状态。只输出一次明确的降级日志。

## 错误与原子性

每个 chunk 的事务边界是：

```text
capture base state
  -> compute tentative rows
  -> validate finite outputs and state sizes
  -> commit all logical state
```

约束：

- CPU 错误恢复 chunk 前状态并向上传递，不返回部分 logits。
- Vulkan 计算或 readback 错误先 abort，再整块 CPU 重试。
- 前面已成功提交的 chunks 保留，不重复计算。
- CPU 重试也失败时返回 CPU 错误，并附带原 Vulkan 错误作为上下文。
- 数值不一致不是自动 fallback 条件；它必须在测试中失败。
- 只在 `#[cfg(test)]` 下提供最小故障注入，以验证回滚，不增加生产 failpoint 配置。

## 正确性验证

### Chunk 边界

使用表驱动测试覆盖：

```text
1, 2, 3, 63, 64, 65, 127, 128
```

另覆盖：空输入拒绝、`N=0` 参数拒绝、恰好达到 KV capacity、最后一个 chunk 只剩一行和 prompt 超过 capacity。

### 模型矩阵

| 模型 | 必须覆盖的状态/分支 |
| --- | --- |
| Qwen3 | text、现有 multimodal positions/deepstack、causal KV、CPU/Vulkan eligibility 分支 |
| Qwen3.5 | dense attention、conv/SSM recurrent state、CPU/Vulkan |
| Gemma4 | shared KV、SWA/full attention、softcap、CPU/通用 Vulkan matmul |

### 对齐标准

在同一模型、同一 backend、同一设备上比较 `prefill_batch_size=1` 与 `16/32/64/128`：

- token IDs 完全一致。
- KV、conv/SSM state 的 F32/F16 原始位完全一致。
- 最终 prompt logits 的 F32 `u32` 完全一致。
- 固定 greedy 参数下多步生成 token 序列完全一致。

Vulkan 比较的是同一 Vulkan 路径的 batch=1 与 batch=N，不把 CPU/Vulkan 跨后端位一致作为前提。Oracle 验证固定真实 GGUF SHA256、llama.cpp commit、chat template、线程数、KV 格式和 greedy 参数，依次比较 token IDs、checkpoint 顺序/shape/次数、F32 `u32` 和多步 greedy token。

回滚测试分别在 Qwen3 KV、Qwen3.5 dense/recurrent state、Gemma4 shared base-KV 和 Vulkan GPU/CPU-shadow 之间注入 chunk 中途错误，确认逻辑长度和可见状态等于 chunk 前快照。

## 性能验证

固定生成 32 tokens，测试：

- prompt length：8、32、128、512
- prefill batch size：1、16、32、64、128
- Qwen3、Qwen3.5、Gemma4
- CPU 和适用的 Vulkan 配置

每个样本固定模型 SHA256、backend、设备、线程数、KV 格式、prompt 和采样配置；预热后记录多次样本的中位数。分别输出：

- `pp`：prefill tokens/s
- `tg`：decode tokens/s
- 首 token 和总耗时
- 峰值 scratch/内存
- Vulkan dispatch/submission 数

默认 64 的合入门槛：

- 512-token prefill 中位数至少提升 10%。
- 128-token prefill 中位数不得回退超过 3%。
- decode 中位数不得回退超过 3%。
- scratch 上限与 chunk size 成正比，不随完整 prompt 增长。
- Qwen3/Qwen3.5 完整 Vulkan session 的 prompt submission 数为 `ceil(prompt_len / chunk_size)`。
- Gemma4 通用 Vulkan matmul 的每个 projection dispatch 数从逐 token 降为逐 chunk；不套用整模型单 submission 指标。

任何目标模型/backend 未满足正确性或适用性能门槛，都不能靠静默降级掩盖，也不能宣称完整落地。

## 预计改动边界

实现计划应按可独立验证的顺序拆分，但不为这些步骤预建接口：

1. CLI/option 传播、正确性基线和 pp/tg benchmark harness。
2. crate-private CPU prepared-rows primitive。
3. Qwen3、Qwen3.5、Gemma4 的 CPU chunk prefill。
4. 通用 Vulkan batched operators 和 shader 参数。
5. Qwen3/Qwen3.5 完整 Vulkan chunk session。
6. Gemma4 对通用 Vulkan batched matmul 的接入。
7. 真实模型、固定 Oracle 和性能矩阵验证。

预计涉及现有 `src/app/`、三个模型目录、`src/ops/kernel/`、`src/vulkan/`、`shaders/glsl/`、`shaders/bin/` 和相关 tests/examples。具体文件清单由实现计划在读取当前符号影响范围后确定，避免提前制造空模块或通用接口。

## 完成定义

完成必须同时具备：

1. 三个模型的目标 CPU/Vulkan 范围均能由同一 batch-size 契约驱动。
2. chunk 状态提交和 GPU 整块回退通过故障测试。
3. batch=1 与 batch=N 满足同 backend 原始位和 greedy token 对齐。
4. 固定 Oracle 的真实模型验证可复现并记录模型/Oracle 哈希。
5. pp 与 tg 分开报告，达到上述默认启用门槛。
6. 未改变单 token decode、现有模型 eligibility 或不相关架构。
