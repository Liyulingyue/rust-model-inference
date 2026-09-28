# Edge0-35B-A3B-preview 标量验证

输入是原始 4 个 safetensors shard、`lora_edge0_35b.safetensors` 和 tokenizer；参考代码固定为 Edge0 `fb4cd2c49ebe22bb230e1451ecb8fb4957ca62e6`。`prerouter_edge0_35b.safetensors` 服务于官方预测路径，Rust 使用模型的实际 router。原始文件含视觉配置但没有视觉权重，因此当前仅支持文本。

转换器保留全部 2377 个张量的原始字节（19,551,119,616 字节 payload），将 U32 packed words 放进 `general.architecture=edge0` 的 GGUF I32 张量；这是本仓库专用格式。已生成文件 `Edge0-35B-A3B-preview-lossless.gguf` 的 SHA-256 是 `50c6c1ce5faef36d5e72d565fa4a27a04801aa243a0d4a5c3f5c4337a408ec7d`。

```sh
MODEL=/path/to/Edge0-35B-A3B-preview
PYTHON=/path/to/repo/.venv/bin/python
"$PYTHON" -m tools.converter.edge0.convert_edge0 "$MODEL" --check
# 首次转换时执行下一行；已有输出文件时可跳过。
"$PYTHON" -m tools.converter.edge0.convert_edge0 "$MODEL" --out "$MODEL/Edge0-35B-A3B-preview-lossless.gguf"
RUSTFLAGS='-C no-vectorize-loops -C no-vectorize-slp' cargo build --profile release-fast --features parity-trace --bin rust-model-inference
TRACE=$(mktemp /tmp/edge0-trace-XXXXXX)
RMI_SCALAR=1 RMI_PARITY_TRACE="$TRACE" RMI_PARITY_FILTER=edge0.embedding,edge0.norm-0,edge0.qkv-0,qwen35.greedy_token_ids \
  target/release-fast/rust-model-inference --model "$MODEL/Edge0-35B-A3B-preview-lossless.gguf" \
  --prompt Hello --threads 1 --kv-cache f32 --prefill-batch-size 1 --max-tokens 4 --temp 0
"$PYTHON" tools/oracle/edge0/check_scalar.py "$MODEL" "$TRACE"
```

已验证：四组文本的 token IDs 一致，`Hello` 的四步 greedy IDs 与官方 Edge0 同为 `[9419, 0, 2500, 628]`；独立纯 Python 标量计算对第一条输入的 2048 个 embedding、2048 个首层 RMSNorm 和全部 8192 个 QKV F32 值逐位一致。检查器对单 bit 改动报错。官方 MLX 使用 BF16/Metal，只用于核对 token 行为，不用于 F32 位级对齐；尚无整模型逐层、完整 logits 的独立标量位级结论。
