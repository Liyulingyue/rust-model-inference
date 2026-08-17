# Issue Track — RustModelInference Debugging Log

## Remaining Known Issues

- [ ] Q4_0 量化模型运行时 panic：代码硬编码使用 `embedding_lookup_q8_0`，但 Q4_0 使用 18 bytes/block（vs Q8_0 的 34 bytes/block），导致 offset 计算错误和越界访问
- [ ] UTF-8 multi-byte characters split across token boundaries may produce replacement characters (U+FFFD)
- [ ] `get_f32_tensor` 每次 forward 调用应缓存 norm weights
- [ ] Matmul per-row 分配应使用预分配 buffer
