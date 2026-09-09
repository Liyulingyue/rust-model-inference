# NeoHorse-1-9B

复用 `qwen35` 文本推理入口。原始目录是 BF16 safetensors；配置声明 MTP，但发布权重不含 MTP，需要跳过该组件。发布的 Tokenizer 使用 NFC，转换时保留为 `tokenizer.ggml.normalizer.nfc=true`。

## 转换和运行

固定本地 llama.cpp 到 `b96806d96061049a5b574269b049bf6241d63d46`。转换器调用该版本已有的 Qwen3.5 张量映射、归一化权重变换和 V head 重排。

```sh
cd /path/to/rust-model-inference

uv run --no-project --with torch --with transformers --with sentencepiece --with mistral-common --with gguf \
  python tools/neohorse/convert_neohorse.py \
  /path/to/models/NeoHorse-1-9B \
  --llama-cpp /path/to/llama.cpp \
  --outfile /path/to/models/NeoHorse-1-9B-BF16-NFC.gguf

cargo run --release --bin rust-model-inference -- \
  --model /path/to/models/NeoHorse-1-9B-BF16-NFC.gguf \
  --prompt '你好' --max-tokens 4 --temp 0 --threads 1 --kv-cache f32
```

请将 `/path/to` 替换为本机实际路径。转换器拒绝覆盖已有输出；已转换时可直接运行第二条命令。

## 模型契约

| 项目 | 值 |
| --- | --- |
| GGUF 架构 | `qwen35` |
| 文件大小 | 17,920,697,216 bytes |
| SHA256 | `58660a818b7f51fd089a0b5f3ed80a432672ff0a141a9f7e822bbc1f563211b8` |
| 张量 | 427：250 BF16、177 F32 |
| 层 | 32；24 循环层、8 完整注意力层；MTP=0 |
| hidden / FFN / vocab | 4096 / 12288 / 248320 |
| Q / KV heads / head dim | 16 / 4 / 256 |
| SSM state / key heads / value heads / conv | 128 / 16 / 32 / 4 |
| RoPE dim / base / sections | 64 / 10000000 / `[11,11,10,0]` |
| BOS / EOS | 不自动添加 BOS；`<|im_end|>`=248046 |

## 验证

```sh
python3 -m unittest discover -s tools/neohorse -v

RMI_NEOHORSE_MODEL=/path/to/models/NeoHorse-1-9B-BF16-NFC.gguf \
RMI_NEOHORSE_HF=/path/to/models/NeoHorse-1-9B \
  cargo test --release --features parity-trace --test qwen35_reference neohorse_tokenizer -- --ignored --nocapture

RMI_NEOHORSE_MODEL=/path/to/models/NeoHorse-1-9B-BF16-NFC.gguf \
RMI_LLAMA_CPP=/path/to/llama.cpp \
  cargo test --release --features parity-trace --test qwen35_reference neohorse_matches_pinned_llama_cpp_bitwise -- --ignored --nocapture
```

数值验证在临时目录构建 Oracle，不修改原始 llama.cpp checkout。双方使用同一 GGUF、ChatML、1 线程 CPU、F32 KV 和 4 步 greedy；Oracle 关闭 Flash Attention、LLAMAFILE、BLAS、Accelerate 和 Metal。Tokenizer 另与发布的 `tokenizer.json` 比较：固定 llama.cpp 的 GGUF Tokenizer 不识别 NFC metadata。

2026-09-09 在本机 macOS arm64 执行结果：

- NeoHorse 集成检查：2 passed（发布 Tokenizer 对照，包括 NFC metadata 类型校验；4 步 greedy 的 checkpoint 顺序、shape、次数及 F32 原始位比较）。
- Qwen3.5：51 passed、2 ignored；包括短缓存和完整缓存使用相同求和顺序的回归检查。
- BF16：16 passed；包括现有 Gemma4 投影检查。
- 转换源契约检查、拒绝 1 ULP 差异的比较器检查、格式检查和 `git diff --check` 均通过。

通用 Tokenizer 测试中的 `normal_control_looking_literal_has_no_chatml_semantic_name` 目前失败（`Some(1)` 与 `None`）；已用 HEAD 原版 Tokenizer 代码独立复现。本次未修改该特殊 token 名称行为。
