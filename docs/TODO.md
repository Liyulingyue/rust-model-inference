# TODO — RustModelInference Feature Roadmap

## High Priority

- [ ] Prompt 处理速度优化（当前远低于 llama.cpp）
- [ ] **Q6_K embedding_lookup 调试** - 当前实现数值正确但模型挂起
- [ ] **统一 embedding_lookup 函数** - 当前 Qwen3Model 存储 embedding_type，每个模型有多处重复的 match 分支
  - 创建统一的 `embedding_lookup(weight, token_id, n_embd, embd_type, out)` 函数
  - 从 Qwen3Model/Qwen35Model/Qwen3AudioModel 中移除 embedding_type 字段
  - 避免在 qwen3.rs、qwen35.rs、qwen3a.rs 中重复 match 分支

## Medium Priority

- [ ] GPU/NPU/CPU+GPU 混合支持
- [ ] Row 切分支持（tensor parallelism across rows）
- [ ] Layer 切分支持（pipeline parallelism across layers）

## Low Priority
- [ ] 更多量化格式支持（Q4_K, Q5_K 等）
- [ ] 完善 GGUfRS 导出功能
- [ ] GGUF 导出支持
