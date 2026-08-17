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
