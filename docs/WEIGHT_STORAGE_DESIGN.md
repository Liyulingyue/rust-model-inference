# 权重存储设计

## 目标

支持未来多个模型在同一套 Kernel 基础设施上运行，同时保留 GGUF/mmap 的零拷贝优势，并把需要生成新权重数据的变换隔离出来。

## 类型职责

### `QuantizedTensor<'a>`

借用型权重视图，直接引用 `TensorSource` 提供的 mmap 字节。它适合：

- 模型加载和推理期间由外层保证 source 生命周期的场景；
- 不需要融合、转置或 dequantize 的普通权重；
- `logits`、parity 等函数内完成的临时推理；
- 低启动延迟和低额外内存占用。

它不拥有底层数据，不能脱离 source 独立存活。

### `QTensorOwned`

拥有型权重，使用 `Vec<u8>` 或 `Vec<f32>` 保存数据。它适合：

- `fuse_vstack` 等会生成新权重布局的变换；
- dequantize、转置、重排等需要独立缓冲区的操作；
- 模型必须脱离原始 mmap 或跨越 source 生命周期长期保存的场景。

它不应成为所有模型的默认权重类型，因为加载大型模型时会产生额外副本。

## 模型选择规则

模型不按名称选择权重类型，而按权重操作选择：

| 权重用途 | 类型 |
| --- | --- |
| 原始只读权重 | `QuantizedTensor<'a>` |
| 临时函数内推理 | `QuantizedTensor<'a>` |
| 融合后的权重 | `QTensorOwned` |
| dequantize/转置后的权重 | `QTensorOwned` |
| 需要独立于 source 存活 | `QTensorOwned` |

因此，不做 FFN/QKV 融合的 Qwen3.5 可以全部使用 `QuantizedTensor<'a>`。如果未来重新启用融合，应只将融合结果放入 `QTensorOwned`，而不是复制所有原始权重。

## Kernel 层

两种权重都实现统一的 `Kernel` trait。上层推理逻辑只依赖 Kernel，不依赖具体存储类型。`QuantizedTensor::into_kernel()` 应在加载阶段调用一次，不能在每个 matmul 中重复创建视图或 Kernel。

## 生命周期规则

借用型模型必须保证 `TensorSource` 的生命周期长于所有权重和 Kernel。长期运行的服务如果无法表达这一关系，应使用拥有型权重，或引入带 `Arc<TensorSource>` 的 mmap-backed Kernel/WeightHandle；禁止用局部借用伪装成长期模型状态。

## 内存和性能取舍

- 借用型路径不复制 GGUF 权重，启动快、额外 RSS 低，适合超大模型。
- 拥有型路径会复制被转换的权重，代价是加载时间和峰值内存，但生命周期和变换更简单。
- 对 100 GB 级模型，默认保持 mmap 零拷贝，只复制确实需要融合或转换的张量。

## 演进方向

1. 保持 `QuantizedTensor<'a>` 和 `QTensorOwned` 两个清晰的所有权边界。
2. 继续让 Kernel dispatch 逻辑共享，避免两种类型复制 SIMD 算法。
3. 如果长期模型需要无生命周期的零拷贝存储，引入保存 `Arc<Mmap>`、offset、长度和形状的 `MmapWeight`，由它在调用时提供临时视图。
4. 只有在性能测量证明必要时，才把融合扩展到更多层或离线生成融合后的 GGUF。
