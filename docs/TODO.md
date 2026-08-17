# TODO — RustModelInference Feature Roadmap

## High Priority

- [ ] Prompt 处理速度优化（当前远低于 llama.cpp）


## Medium Priority


- [ ] GPU/NPU/CPU+GPU 混合支持
- [ ] Row 切分支持（tensor parallelism across rows）
- [ ] Layer 切分支持（pipeline parallelism across layers）

## Low Priority
- [ ] 更多量化格式支持（Q4_K, Q5_K, Q6_K 等）
- [ ] 完善 GGUfRS 导出功能
- [ ] GGUF 导出支持
