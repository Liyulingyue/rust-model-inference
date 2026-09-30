# CLM-v0.1-8B 标量逐位验证

CLM 只有 state/action 投影头；文本编码器另用 Qwen3-8B。参考源码固定为
[Contrastive-LM/CLM `bb42c6c`](https://github.com/Contrastive-LM/CLM/commit/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7)，
编码器 Oracle 固定为 llama.cpp `b96806d96061049a5b574269b049bf6241d63d46`。

| 文件 | SHA-256 |
|---|---|
| `CLM_v0.1-8B.pt` | `b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5` |
| `clm-v0.1-8B-heads-f32.gguf` | `d5bbef8f4c9edf0b5cfb38b4a9326a1ac776cc9b479345a56b2efc37de3c39be` |
| `Qwen3-8B-BF16.gguf` (`unsloth/Qwen3-8B-GGUF`, revision `c2f559f`) | `5e416a2020fe63e76ea13c8979be35fc6070aaf3578f7876400c55c2f5c3eb30` |

本仓库的转换器生成上表中的 GGUF：

```bash
models/.venv/bin/python -m tools.converter.clm.convert_clm \
  models/CLM-v0.1-8B/CLM_v0.1-8B.pt \
  -o models/CLM-v0.1-8B/clm-v0.1-8B-heads-f32.gguf
```

llama.cpp 构建使用 `GGML_NATIVE=OFF`、`GGML_CPU_ARM_ARCH=armv8-a`，关闭
Metal、BLAS、Accelerate、OpenMP；C/C++ flags 为
`-O2 -U__ARM_NEON -fno-vectorize -fno-slp-vectorize -ffp-contract=off`。
运行时为单线程、F32 KV、`-fa off -ngl 0 --no-repack`。
Rust 标量构建与验证：

```bash
RUSTFLAGS='-C no-vectorize-loops -C no-vectorize-slp' \
  cargo build --profile release-fast --features parity-trace --bin rust-model-inference
models/.venv/bin/python -m tools.oracle.clm.verify_text \
  models/CLM-v0.1-8B/CLM_v0.1-8B.pt \
  models/CLM-v0.1-8B/clm-v0.1-8B-heads-f32.gguf \
  models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf \
  ./target/release-fast/rust-model-inference \
  /path/to/scalar-llama.cpp/bin/llama-debug
```

`verify_text.py` 比较 `hello`、`hello world`、`你好，世界` 的 token IDs；
比较两组文本评分的 last-token pooled 与 L2-normalized embedding，
以及双头 `inp`、GELU、hidden、LayerNorm、out、单位向量和最终 logit 的原始 F32 位。
`scalar.py` 只用 PyTorch 反序列化 `.pt`，逐元素计算双头，不调用 BLAS。

2026-09-30 的结果：两组输入共 **32,768 个编码器值、34,818 个投影头值逐位一致**；
logit 分别为 `0x415a1e78`、`0x41b29eb4`。这个结论仅覆盖上述 BF16 编码器与
F32 双头的标量路径；官方 vLLM、NEON/FMA/BLAS/Accelerate、默认加速路径及其他量化
GGUF 未做逐位验证。
