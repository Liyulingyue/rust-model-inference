# Chunked prefill

单个请求的 prompt 以固定大小的 chunk 处理。默认 `--prefill-batch-size` 为 `64`；
设置为 `1` 可得到顺序诊断基线。该参数也可用于库级 session 入口。

## 支持范围

| 模型 | CPU | Vulkan |
|------|-----|--------|
| Qwen3 | chunked prefill | 在现有 Vulkan eligibility 内执行 chunked prefill |
| Qwen3.5 | chunked prefill | 在现有 Vulkan eligibility 内执行 chunked prefill |
| Gemma4 | chunked prefill | 批量线性投影；attention 与 KV 为模型控制的 CPU 路径 |

decode 保持一次一个 token。

## CLI

```bash
rust-model-inference --model model.gguf --prompt "Hello" --prefill-batch-size 64
rust-model-inference --model model.gguf --prompt "Hello" --prefill-batch-size 1
```

每个 chunk 的 KV、recurrent state 和 logits 在成功后一起提交。Vulkan 执行失败时，
运行时放弃该 chunk 的 GPU 结果，并从相同的已提交前缀在 CPU 重算完整 chunk；CPU 重试
失败则保持已提交前缀不变并返回错误。

## `prefill_bench`

基准命令形状如下；`qwen3` 也可替换为 `qwen35` 或 `gemma4`，后两者按模型要求
使用对应的模型文件和 KV 格式：

```bash
cargo run --release --example prefill_bench -- qwen3 \
  --model /path/to/model.gguf --backend cpu --threads 4 --kv f16 \
  --prompt-tokens 512 --batch 64 --samples 5 --generate 32
```

Vulkan 运行时增加 `--features vulkan` 并将 `--backend` 设为 `vulkan`。输出包含每个
样本及中位数的 prompt/decode 速率、首 token 与总耗时、scratch 大小，以及适用时的
Vulkan submission 计数。
