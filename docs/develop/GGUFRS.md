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

## 加载时不变量

下列规则在 `validate_index` 中强制执行，违反任一项即拒收：

- **包至少含一个组件**；`role == Llm` 必须恰好 1 个；`role == Mmproj` 至多 1 个。
- **`general.alignment`**：必须是 2 的非零次幂；缺省视为 32。每个 tensor 的 `segment_offset` 必须按 `max(32, general.alignment)` 对齐。
- **段**：段表按表索引顺序连续写入；`absolute_offset` 与 `stored_len` 都是 64 KiB 的倍数；LLM 段数严格等于 `block_count + 1`（1 个 shared + `block_count` 个 layer），layer 索引恰好覆盖 `0..block_count` 无缺、无重。
- **段内零填充**：每个段在加载时通过 `mmap` 读取完整 `stored_len`，先校验 SHA-256，再要求 tensor 范围之外的所有字节必须为 0（段头对齐间隙与段尾填充同理）。任何非零 padding 即拒收。
- **段排序**：段在其所属组件内按 `(kind, layer)` 升序排序——`Shared < Layer < Component`、layer 段按 layer 索引升序。`segment_id` 是段在全表中的索引，由该排序导出。
- **元数据与张量排序**：每个组件内的 metadata 按 key 字节升序、tensor 按 name 字节升序；同一组件内 key/name 重复即拒收。不同组件允许持有同名 key（按 `(component_id, key)` 独立寻址）。
- **无组件重叠**：tensor 字节范围按偏移排序后，相邻对不得相交。

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

数组编码为 `i32 element_type`、`u64 count`，然后是同构值；嵌套数组被拒收。整张元数据表按 `(component_id, key 字节)` 全序排序——等价于先按组件表序，再按组件内 key 升序。一个组件内重复的 key 无效；不同组件允许持有同名 key（loader 按 `(component_id, key)` 寻址，互不干扰）。`TensorSource::metadata(key)` 只返回当前组件作用域内的值。

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

LLM 段数严格等于 `block_count + 1`：1 个 shared 段 + 每个 layer 一个 layer 段。mmproj 只有 1 个 component 段。段在其组件内按 `(kind, layer)` 升序排序——`Shared < Layer < Component`，layer 段按层索引升序。`segment_id` 是段在段表中的索引，由该排序导出。段起始和存储长度是 64 KiB 的倍数且段是连续的。SHA-256 覆盖完整存储段，包括 tensor 间和尾部零填充；加载时进一步要求所有非 tensor 字节必须为 0。因此段可以独立验证、映射和释放。

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

`--mmproj` 是可选的。默认不会覆盖已有输出。`--overwrite` 请求原子替换。

### 源 GGUF 校验

导出在搬运字节之前先校验源：

- **LLM**：要求 `general.architecture` ∈ `{qwen2, qwen2vl, qwen3, qwen3vl, qwen3vlmoe, qwen35, llama}` 之一，并要求 `{arch}.block_count` 存在且 ≥ 1；否则直接报错。其他架构需先经上游 `convert` 脚本走 GGUF 再打包。
- **mmproj**：按 `clip.has_audio_encoder` 分支——若为 `true`，要求 `clip.audio.projector_type = qwen3a` 或 `clip.projector_type = qwen2.5o` 之一并通过对应的音频配置校验（`mel_encoder::validate_qwen3a_source` 或 `Qwen25OmniAudioConfig::from_source`）；否则走 vision 路径，要求 `clip.vision.projection_dim / image_size / patch_size / embedding_length / feed_forward_length / block_count / attention.head_count` 与 `clip.vision.attention.layer_norm_epsilon` 全齐，并存在 `v.patch_embd.weight`、`mm.0.weight`、`mm.2.weight` 三个必备张量。

### 字节来源与生命周期

导出器通过 `GGUFLoader::tensor_slice(name)` 直接读取源 `.gguf` 的 mmap 切片，**不做反量化、再量化、重打包或浮点转换**。`ExportSource` 持有 loader 直到写盘结束，因此源文件全程不能删除或被截断（其 mmap 必须保持有效）。

### 原子发布

导出在目标目录的同一文件系统下创建唯一临时文件（`.ggufrs-{pid}-{id}.tmp`，由 `create_new(true)` 保证独占），写完后执行 `flush` + `sync_all`；随后从临时文件的克隆句柄重新打开并跑 `verify_all()`（对每个段重新计算 SHA-256 并与表头比对），最后做身份校验：

- **Unix**：`st_dev + st_ino` 比对，强度高。
- **Windows**：仅比较 `file.len()`（无 `dev/ino`），强度弱于 Unix。

身份校验失败即拒收，临时文件随 `Drop` 清理。**默认发布路径是 `hard_link` 临时文件到目标**——这要求目标不存在且文件系统支持 hard link；任一条件不满足即报错，**绝不先删除目标**。`--overwrite` 走 `rename(临时, 目标)` 的原地原子替换；若底层文件系统不支持 rename 跨设备/跨路径也会报错。两条路径都不留半成品给读者。

### 确定性前提

"相同源 + 相同选项 → 字节完全一致" 成立的前提：源 `.gguf` 的 mmap 视图稳定（不被截断/重写）、元数据 key 集合固定、临时文件名生成序列的全局计数器与并发无关（同进程内多次调用是确定的）。一旦源 GGUF 内容变化或选项变化，输出随之变化；SHA-256、段偏移、表长度都会变。

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

### 当前实际路径（V1 仍生效）

V1 硬性限定"恰好一个 LLM + 最多一个 mmproj"。项目里已有的多模型需求，**当前**都是用 **多个独立 `.gguf` 文件 + 多 CLI flag** 解决的——三者均以 `ComponentRole::Llm` 打开：

- **Z-Image（pig）**：`--model <diffusion>` + `--text-encoder <qwen3>` + `--vae <flux_vae>`，三个独立 `.gguf`，见 `src/main.rs:140-163` 与 `src/app/cli.rs:516-530`。
- **DreamX-Creator / Qwen-Drive / ASR / TTS / 多模态视觉问答**：LLM + mmproj 两个独立 `.gguf`，mmproj 以 `ComponentRole::Mmproj` 打开；这条路径**可以直接走 v1 打包**进单文件 `.ggufrs`，无需等 v2。

也就是说，V1 不是"够用但不够美"，而是**真正的多组件单文件分发尚未启用**——多文件分布既是当前态，也是 v2 role 扩展要收敛的目标。

### llama.cpp 的对照做法

参照 llama.cpp（源码核实，`tools/mtmd/clip.cpp`）：**一个 mmproj GGUF 可同时容纳视觉/语音/语音生成编码器**（`loader.has_vision / has_audio / has_gen_audio` 分别建 `clip_ctx`），元数据冲突靠 **key 字符串前缀**解决（`clip.vision.n_embd` / `clip.audio.n_embd` / `clip.gen.audio.*`），张量同理用 `mm.*` 等前缀。这是"单文件单表 + 字符串前缀命名空间"的方案——可行但无结构保证，新增模态要改 clip.cpp 的前缀清单。

### GGUFRS 的结构性解法

GGUFRS 的**组件级元数据表**（每个 component 独立 scoped metadata）天然就是该需求的解法：`llm / diffusion / text-encoder / vae / vision-encoder / audio-encoder / audio-decoder / omni` 各自成组件，**作用域隔离、无前缀约定**，新增模态 = 新增一个 role 值 + 引擎侧 `load_component(role)`。loader 完全不感知命名空间细节。

### V2 待办（按依赖顺序）

1. **role 枚举扩展**——`diffusion / text-encoder / vae / audio-encoder / audio-decoder / omni` 等新 role（omni 涵盖视觉+语音+语音生成的多模态编码器合一组件）。同时调整 V1 的"恰好一个 LLM + 最多一个 mmproj"为"包至少含一个 LLM，按 role 限额接受其它角色"。
2. **`ggufrs` CLI 补 `verify` / `info` 子命令**——发布侧校验与检视是分发格式基本功。底层 API 已就绪（`GgufrsFile::verify_all` + `components() / segments_for_component() / tensors_for_segment()`），CLI 只差胶水。
3. **release 流水线接入打包**——脚本或直接调用 `export_ggufrs`；附 `verify_all` 作为发版前门禁。
4. **远期**：HF 直转原型（先选 qwen3 系验证映射表工作量）+ 视需要自研量化。
