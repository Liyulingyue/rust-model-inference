# GLiNER2.5-Decide F32 标量逐位对齐

## 固定输入

- 模型：`fastino/GLiNER2.5-Decide`；`model.safetensors` SHA-256 `40a5a23ff860dc3dff426cecd1048cacdd29c648c96db209dad818e9686dc997`，`tokenizer.json` SHA-256 `3ad87d9ffe669147063e70850927dd2da90249e2acc5c8527f1eb65df467bcc8`。
- 官方实现：`fastino-ai/GLiNER2` commit `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf`，只读 Oracle。
- 新 GGUF：`gliner2-decide-f32.gguf`，SHA-256 `093a67cc5e42afe26f43a55dea727040c2f4456f2120ff521c3809386135e755`，1,752,447,104 bytes，`general.architecture=gliner2`，`tokenizer.ggml.model=hf-json`。394 个 F32 张量、1,744,089,092 个载荷字节与 Safetensors 对应张量逐字节相同；转换器将原始 `tokenizer.json` 放进 metadata。

## 复现

把原始权重、`config.json`、`tokenizer.json`、`tokenizer_config.json` 和 `encoder_config/config.json` 放在同一目录。验证环境为 Python 3.12、PyTorch 2.14.0、Transformers 4.57.6、tokenizers 0.22.2、NumPy 2.5.3。所有 Python 依赖装在仓库根 `.venv`；`trace_official.py` 只读固定 commit 的官方实现。

```sh
git clone https://github.com/fastino-ai/GLiNER2.git target/gliner2-oracle
git -C target/gliner2-oracle checkout 55656fbfa01d3d4a77485e1a1eeeaf682990ccdf
./.venv/bin/python -m pip install -e target/gliner2-oracle 'torch==2.14.0' 'transformers==4.57.6' 'tokenizers==0.22.2' 'numpy==2.5.3' safetensors

MODEL=/Users/gouzi/Documents/git/rust-model-inference/models/GLiNER2.5-Decide
./.venv/bin/python -m tools.converter.gliner.convert_gliner "$MODEL" "$MODEL/gliner2-decide-f32.gguf"
RUSTFLAGS='-C no-vectorize-loops -C no-vectorize-slp' cargo build --profile release-fast --features parity-trace --bin rust-model-inference
./.venv/bin/python -m tools.oracle.gliner.run_parity \
  --model-dir "$MODEL" --gguf "$MODEL/gliner2-decide-f32.gguf" \
  --output target/gliner-parity-traces
```

转换器拒绝覆盖已有 GGUF，trace 目录也必须不存在。Oracle 将 linear、BMM、LayerNorm、softmax、GELU 换为单线程 C F32 内核，用 `-ffp-contract=off -fno-vectorize -fno-slp-vectorize` 编译。Rust 使用 `RMI_SCALAR=1` 和禁用自动向量化的构建。比较器严格检查 token IDs、检查点顺序与形状、全部 F32 原始位，并报告首个分叉。

| 请求 | token 数 | 检查点 | 逐位一致的 F32 值 |
|---|---:|---:|---:|
| `refund.json`：两标签 | 14 | 30 | 415,746 |
| `multitask.json`：多任务、多标签、描述 | 52 | 30 | 1,544,199 |
| `examples.json`：指令与示例 | 25 | 30 | 742,402 |
| `long-position.json`：跨越 128 token 相对位置桶边界 | 142 | 30 | 4,216,834 |

合计 120 个检查点、6,919,181 个 F32 值；包括 embedding、首层 Q/K/V 与 attention context、24 层输出及最终 logits。四例标签输出也与官方相同。**结论仅适用于 F32 标量路径**；SIMD、FMA、BLAS、Accelerate、量化路径未作为逐位对齐目标。GLiNER 是分类模型，没有 greedy 生成阶段。
