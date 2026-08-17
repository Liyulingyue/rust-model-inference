# Issue Track — RustModelInference Debugging Log

## Remaining Known Issues

- [ ] UTF-8 multi-byte characters split across token boundaries may produce replacement characters (U+FFFD)
- [ ] `get_f32_tensor` 每次 forward 调用应缓存 norm weights
- [ ] Matmul per-row 分配应使用预分配 buffer
