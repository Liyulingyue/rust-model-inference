# NeoHorse-1-9B / 4B

复用 `qwen35` 文本推理入口。9B 原始目录是 BF16 safetensors；配置声明 MTP，但发布权重不含 MTP，需要跳过该组件。发布的 Tokenizer 使用 NFC，转换时保留为 `tokenizer.ggml.normalizer.nfc=true`。

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

数值验证在临时目录构建 Oracle，不修改原始 llama.cpp checkout。双方使用同一 GGUF、ChatML、1 线程 CPU、F32 KV 和 4 步 greedy；Oracle 关闭 Flash Attention、LLAMAFILE、BLAS、Accelerate 和 Metal，并让 attention softmax 使用标量 `expf`，与 Rust 的精确 `f32::exp()` 路径对齐。Tokenizer 另与发布的 `tokenizer.json` 比较：固定 llama.cpp 的 GGUF Tokenizer 不识别 NFC metadata。

2026-09-09 在本机 macOS arm64 执行结果：

- NeoHorse 集成检查：2 passed（发布 Tokenizer 对照，包括 NFC metadata 类型校验；4 步 greedy 的 checkpoint 顺序、shape、次数及 F32 原始位比较）。
- Qwen3.5：51 passed、2 ignored；包括短缓存和完整缓存使用相同求和顺序的回归检查。
- BF16：16 passed；包括现有 Gemma4 投影检查。
- 转换源契约检查、拒绝 1 ULP 差异的比较器检查、格式检查和 `git diff --check` 均通过。

2026-09-10 将 Qwen3.5 dense attention 恢复为精确 softmax 后，9B BF16 在标量 `expf` Oracle 下重新完成 85 条 trace、3,928,576 个 F32 和 4 步 greedy token 的逐位比较。

通用 Tokenizer 测试中的 `normal_control_looking_literal_has_no_chatml_semantic_name` 目前失败（`Some(1)` 与 `None`）；已用 HEAD 原版 Tokenizer 代码独立复现。本次未修改该特殊 token 名称行为。

## NeoHorse-1-4B 官方 GGUF 对比

2026-09-10 使用 [TokenRhythm/NeoHorse-1-4B-GGUF](https://huggingface.co/TokenRhythm/NeoHorse-1-4B-GGUF) 的本地文件核验。4B 仍为 `qwen35`、32 层（24 循环层、8 dense 层）、vocab 248320；hidden=2560、FFN=9216，输出层复用 embedding，共 426 个张量。文件不含 MTP 层或 NFC metadata；这里验证的是相同 GGUF 在两个运行时的行为，不代表已与原始 HF Tokenizer 的 NFC 行为对齐。

双方使用固定的上述 llama.cpp commit、相同 GGUF 和 ChatML 输入 `你好`、单线程 ARM64 CPU、F32 KV、4 步 greedy。Oracle 的 attention softmax 使用标量 `expf`；Q4_K_M / Q5_K_M 还使用标量 K-quant 点积。每种格式比较 85 条 trace 记录，共 3,928,576 个 F32 值，包含选定中间 checkpoint、每步完整 logits，以及 token IDs；均逐位一致。这是固定输入的正确性检查，不是模型质量或性能评测。

| GGUF 格式 | Oracle 路径 | 结果 | SHA256 |
| --- | --- | --- | --- |
| BF16 | 标量 softmax；BF16 标量点积 | 逐位一致 | `b27b4cb2770673ab948a8db166f52f9c74da3df3c88d43f5aa564377f1b4ae50` |
| F16 | 标量 softmax | 逐位一致 | `6882ed8cbb847bca71144b8527dfce3ee7ce91573e9f3f411d13df53b3d738c7` |
| Q8_0 | 标量 softmax | 逐位一致 | `62c6b5d23e15e38f75e4d301ba3d9a16d48662bf48cebe8ec27a7c9c30b65dfb` |
| Q4_K_M | 标量 softmax；Q4_K / Q5_K / Q6_K 标量点积 | 逐位一致 | `7da8ac6219b0aaa21b58cb26c1ec6826b478449215a3b0448742e1cda60fa2b4` |
| Q5_K_M | 标量 softmax；Q4_K / Q5_K / Q6_K 标量点积 | 逐位一致 | `617f51ff6e639d0c5049cbacad3ca736523d50de8510b7afff7d5d0fa9e7fc02` |

Q4_K_M / Q5_K_M 的 Rust ARM64 路径已有标量点积。普通 Oracle 启用 CPU 权重重排时，首个 `conv_output_raw-0` 的第二个值出现 `0x3c592303` / `0x3c592302` 差异。量化标量模式使用原有 `*_generic` 点积，并关闭 `GGML_CPU_REPACK` 和 C 自动 FMA 融合；attention softmax 在所有 Oracle 构建中固定为标量参考路径。

复现 BF16、F16 或 Q8_0（替换实际路径和文件后缀）：

```sh
RMI_NEOHORSE_MODEL=/path/to/NeoHorse-1-4B-GGUF/NeoHorse-1-4B-BF16.gguf \
RMI_LLAMA_CPP=/path/to/llama.cpp \
  cargo test --release --features parity-trace --test qwen35_reference neohorse_matches_pinned_llama_cpp_bitwise -- --ignored --nocapture
```

复现 Q4_K_M / Q5_K_M 时显式选择标量量化 Oracle：

```sh
RMI_NEOHORSE_MODEL=/path/to/NeoHorse-1-4B-GGUF/NeoHorse-1-4B-Q4_K_M.gguf \
RMI_LLAMA_CPP=/path/to/llama.cpp \
RMI_QWEN35_SCALAR_KQUANT=1 \
  cargo test --release --features parity-trace --test qwen35_reference neohorse_matches_pinned_llama_cpp_bitwise -- --ignored --nocapture
```

两种 Oracle 都在临时副本中构建，保留原始 llama.cpp checkout。`RMI_QWEN35_ORACLE` 可复用已按对应模式构建的二进制；设置它时不会重新执行构建脚本。
