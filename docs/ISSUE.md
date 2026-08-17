# Issue Track — RustModelInference Debugging Log

## Remaining Known Issues

- [ ] Q6_K embedding lookup 实现不完整：已添加 `embedding_lookup_q6_k` 函数，但运行时产生空输出或hang（模型能加载但无文字生成）
- [ ] UTF-8 multi-byte characters split across token boundaries may produce replacement characters (U+FFFD)
- [ ] `get_f32_tensor` 每次 forward 调用应缓存 norm weights
- [ ] Matmul per-row 分配应使用预分配 buffer
