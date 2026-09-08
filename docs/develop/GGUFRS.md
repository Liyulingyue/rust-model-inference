# GGUFRS v1

GGUFRS 是 RustModelInference 的模型管理和设备无关加载方案。它打包一个 LLM GGUF 和可选的一个 mmproj GGUF，保留组件元数据和原始 tensor 字节。它不可被 llama.cpp 读取，也不替代普通的 GGUF 交换格式。

所有整数均为小端序。偏移量和字节长度为 `u64`；计数和稳定 ID 为 `u32`。字符串为 `u64 字节长度` + UTF-8 字节。GGUF 元数据和 GGML tensor 类型数值码被复用。

## 物理布局

```text
128 字节超级块
组件表
组件级元数据表
段表
tensor 表
零对齐填充
64 KiB 对齐的 tensor 段
```

tensor 数据之后不追加任何组件或目录。最后一个段在声明的文件大小处结束。

## 超级块（Superblock）

| 偏移 | 大小 | 字段 |
|---:|---:|---|
| 0 | 8 | magic `b"GGUFRS\0\0"` |
| 8 | 4 | version，`1` |
| 12 | 4 | flags，v1 中为 `0` |
| 16 | 8 | 声明的文件大小 |
| 24 | 4 | 组件数量 |
| 28 | 4 | 元数据数量 |
| 32 | 4 | 段数量 |
| 36 | 4 | tensor 数量 |
| 40 | 8 | 组件表偏移 |
| 48 | 8 | 组件表长度 |
| 56 | 8 | 元数据表偏移 |
| 64 | 8 | 元数据表长度 |
| 72 | 8 | 段表偏移 |
| 80 | 8 | 段表长度 |
| 88 | 8 | tensor 表偏移 |
| 96 | 8 | tensor 表长度 |
| 104 | 8 | tensor 数据偏移 |
| 112 | 16 | 保留的零字节 |

读取方会拒绝：不支持的版本、非零 flags/保留字节、无序或不连续的表、非零表填充、无效范围、追加的数据、以及声明大小与实际文件大小不符的情况。

## 组件表（Component Table）

每个条目为：

```text
u32 component_id
u32 role                 # 1 = LLM, 2 = MMPROJ
string name              # 规范名为 "llm" 或 "mmproj"
u32 metadata_start
u32 metadata_count
u32 tensor_start
u32 tensor_count
u32 segment_start
u32 segment_count
```

V1 要求恰好一个 LLM 和最多一个 mmproj。组件按 role 和 UTF-8 名字字节排序；ID 即其表索引。

## 组件级元数据表（Scoped Metadata Table）

每个条目为：

```text
u32 component_id
string key
i32 GGUF value_type
typed GGUF value
```

数组编码为 `i32 element_type`、`u64 count`，然后是同构值。元数据按组件和 key 字节排序。一个组件内重复的 key 无效；不同组件中相同的 key 保持独立。

## 段表（Segment Table）

每个 72 字节条目为：

```text
u32 segment_id
u32 component_id
u32 kind                 # 1 = shared, 2 = layer, 3 = component
i32 layer                # layer 索引，或 -1
u64 absolute_offset
u64 stored_length
u32 tensor_start
u32 tensor_count
u8 sha256[32]
```

LLM 有一个共享段和每个 layer 一个段。mmproj 有一个组件段。段起始和存储长度是 64 KiB 的倍数且段是连续的。SHA-256 覆盖完整存储段，包括 tensor 间和尾部零填充。因此段可以独立验证、映射和释放。

## Tensor 表与字节（Tensor Table and Bytes）

每个条目为：

```text
u32 component_id
u32 segment_id
string tensor_name
i32 GGML type
u32 rank
u64 dims[rank]
u64 offset_within_segment
u64 exact_byte_length
```

Tensor 在每个段内按名字字节排序。偏移量使用该组件的 `max(32, general.alignment)`。映射前验证 shape、量化块大小、范围和重叠。

导出器直接复制 `GGUFLoader::tensor_slice(name)`。它从不反量化、再量化、重打包或通过浮点转换 tensor 数据。因此相同的源字节和选项产生字节完全相同的包；源路径、时间戳、主机设备和临时名称不被序列化。

## 导出与发布

```bash
cargo run --release --bin ggufrs -- \
  export \
  --llm model.gguf \
  --mmproj mmproj.gguf \
  --output model.ggufrs
```

`--mmproj` 是可选的。默认不会覆盖已有输出。`--overwrite` 请求原子替换。导出在输出目录写入唯一文件，保留并同步其 `create_new` 句柄，通过该句柄的克隆和生产者读者验证每个段，然后发布。不支持的原子发布返回错误，且从不先删除目标。

## 运行时与加载规划

`TensorSource` 是 GGUF 和已加载 GGUFRS 组件的通用只读接口。运行时格式选择使用文件 magic 而非扩展名。显式 `--mmproj` 覆盖打包的组件。

`LayerSplit` 保持每个 layer 段完整，并将连续 layer 范围分配给调用方提供的逻辑设备。共享和 mmproj tensor 保留在声明的主设备上。`TensorSplit` 只在完整行之间划分 tensor；量化行必须包含完整的量化块。容量只计算 tensor 载荷，不计算表或填充字节。

V1 只针对逻辑 CPU 设备执行计划，以验证确定性放置和映射生命周期。Metal、CUDA、NPU、传输和执行调度是未来后端；它们不改变此文件格式。

---

# 生态定位与演进认知（2026-08）

本节记录对 GGUFRS 战略定位的分析结论与决策依据，供后续演进（v2 role 扩展、直转路径）参考。

## 与 safetensors / GGUF 的正面对比

**"GGUFRS 的优势是 mmap 好做"——这个说法不成立。** safetensors 本身就是 mmap-first 设计
（8 字节头长度 + JSON 头 + 原始对齐字节），GGUF 同样 mmap 友好。三者在这点上打平。
GGUFRS 真正超出前两者的能力是：

| 能力 | GGUFRS | safetensors | GGUF |
|---|---|---|---|
| 段级生命周期（per-layer 独立 map/unmap） | ✅ 64KiB 对齐段 + 段级 SHA-256 | ❌ 单一大映射 | ❌ 无段概念 |
| 完整性 | ✅ 段级校验、独立 verify | ❌ 无 | ❌ 无 |
| 多组件打包 | ✅ 一个文件 + role 隔离的元数据 | ❌ 单模型 | ❌ 单模型 |
| 确定性打包 / 原子发布 | ✅ 字节级可复现 | ❌ | ❌ |
| 生态 | ❌ 仅本仓库 | ✅✅ HF 默认 | ✅✅ llama.cpp |

## 命名契约的三层模型

一个权重在流转中携带三层命名，转换脚本的职责边界由此确定：

1. **结构名（训练侧）**：`model.layers.0.self_attn.q_proj.weight` —— 训练代码模块树的路径，
   描述性、家族间不保证一致。safetensors 原样保存生产者的名字。
2. **规范名（格式侧）**：`blk.0.attn_q.weight` —— GGUF spec 定义的跨家族统一名，
   `general.architecture` 指定组装结构。**"出新架构要加命名配置"的成本在这一层**：
   纯更名/重排版的模型 = 加一张映射表；计算结构变化 = 引擎侧另写前向（与转换无关）。
3. **量化类型名（字节布局侧）**：`Q8_0`、`Q4_K_M` —— GGML 特有，规定字节块布局与解码 kernel，
   属于文件格式层特性。safetensors 世界没有这层（GPTQ/AWQ 量化是模型代码层的，打包方式各家自定义，
   非自描述）。

推论：**转换脚本永远不实现计算**，只做"结构名 → 规范名"的映射 + 排版重排（permute/融合/堆叠）。
排版规则跟引擎内核走（如 llama 权重 permute 源于 ggml 的 interleaved-rope 布局），不跟数学走。

## 依赖边界与双轨战略

当前对 llama.cpp 的真实依赖只有两条：HF→GGUF 转换脚本（新架构支持速度）、
隐性布局契约（内核排版约定，已通过 bit 级 parity 吸收）。引擎本身零依赖——`TensorSource`
是 GGUF 与 GGUFRS 组件的公共接口。

终局形态为**双轨**：

- **量化模型**：HF → [llama.cpp convert] → GGUF → [ggufrs export] → GGUFRS。
  上游生态的架构支持与量化工具链（imatrix 等）继续白拿，本仓库对 GGUF 永远保留导入路径。
- **直转路径（退路）**：HF safetensors → GGUFRS，绕开对上游转换节奏的依赖。
  代价是每自支持架构认领一份"HF 名 → 规范名"映射表（复用现有 trunk 已知的规范名，引擎零改动）
  以及可能的独立量化工具链（唯一的大工程，远期项）。

不建议做的事：往格式里加量化能力（字节拷贝原则是正确性卖点）、追求 HF/llama.cpp 互认、
松动确定性导出与原子发布纪律。

## 多模态组件：V2 role 扩展的依据

现状差距：V1 硬性限定"恰好一个 LLM + 最多一个 mmproj"，而本项目内已存在三类实际需求——
Z-Image 需要 diffusion / text-encoder / vae 三组件单文件分发；ASR/TTS 的 mmproj 组件已在
`models/` 中出现；Omni 类模型（视觉 + 语音 + 语音生成）是明确趋势。

参照 llama.cpp 的做法（源码核实，`tools/mtmd/clip.cpp`）：**一个 mmproj GGUF 可同时容纳
视觉/语音/语音生成编码器**（`loader.has_vision / has_audio / has_gen_audio` 分别建
`clip_ctx`），元数据冲突靠 **key 字符串前缀**解决（`clip.vision.n_embd` /
`clip.audio.n_embd` / `clip.gen.audio.*`），张量同理用 `mm.*` 等前缀。这是"单文件单表 +
字符串前缀命名空间"的方案——可行但无结构保证，新增模态要改 clip.cpp 的前缀清单。

GGUFRS 的**组件级元数据表**（每个 component 独立 scoped metadata）是该需求的结构性解法：
llm / vision-encoder / audio-encoder / audio-decoder 各自成组件，天然隔离、无前缀约定，
新增模态 = 新增一个 role 值。引擎侧按 role 取组件，加载逻辑不感知命名空间细节。

**V2 待办（按依赖顺序）**：
1. role 枚举扩展（diffusion / text-encoder / vae / audio-*），解除 mmproj 单组件限制；
2. `ggufrs` CLI 补 `verify` / `info` 子命令（发布侧校验与检视是分发格式基本功）；
3. release 流水线接入打包（脚本或直接调用 export）；
4. 远期：HF 直转原型（先选 qwen3 系验证映射表工作量）+ 视需要自研量化。
