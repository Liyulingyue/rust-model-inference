# Edge0-35B-A3B-preview 标量验证

输入是原始 4 个 safetensors shard、`lora_edge0_35b.safetensors` 和 tokenizer；参考代码固定为 Edge0 `fb4cd2c49ebe22bb230e1451ecb8fb4957ca62e6`。`prerouter_edge0_35b.safetensors` 服务于官方预测路径，Rust 使用模型的实际 router。原始文件含视觉配置但没有视觉权重，因此当前仅支持文本。

Rust 的 `Edge0Model` 独立持有 MoE 权重，与 Qwen3.5 共用 attention/SSM 的 `HybridTrunk` 和 session 实现；`Qwen35Model::from_source` 拒绝 `edge0` 架构。

转换器保留全部 2377 个张量的原始字节（19,551,119,616 字节 payload），将 U32 packed words 放进 `general.architecture=edge0` 的 GGUF I32 张量；这是本仓库专用格式。已生成文件 `Edge0-35B-A3B-preview-lossless.gguf` 的 SHA-256 是 `50c6c1ce5faef36d5e72d565fa4a27a04801aa243a0d4a5c3f5c4337a408ec7d`。

**只有 `lossless` 模式可以跑下面的 oracle。** 转换器还支持 `--quant f32/f16/q8_0/q4_0`，
它们先把 affine group 展开成 F32 再重新编码成通用 GGML 类型，输出不再包含
`scales`/`biases`，Rust 端也无法加载（`Edge0Model` 只认 I32 + BF16 三元组）。
详见 `tools/converter/README.md#edge0-量化支持现状`。

```sh
MODEL=/path/to/Edge0-35B-A3B-preview
PYTHON=/path/to/repo/.venv/bin/python
"$PYTHON" -m tools.converter.edge0.convert_edge0 "$MODEL" --check
# 首次转换时执行下一行；已有输出文件时可跳过。
"$PYTHON" -m tools.converter.edge0.convert_edge0 "$MODEL" --out "$MODEL/Edge0-35B-A3B-preview-lossless.gguf"
RUSTFLAGS='-C no-vectorize-loops -C no-vectorize-slp' cargo build --profile release-fast --features parity-trace --bin rust-model-inference
TRACE=$(mktemp /tmp/edge0-trace-XXXXXX)
EDGE0_TRACE_FILTER=edge0.embedding,edge0.norm-0,edge0.qkv-0,edge0.z-0,edge0.beta-0,conv_output_raw-0,q_conv_predelta-0,k_conv_predelta-0,new_state-0,final_output-0,edge0.recurrent_projection-0,edge0.moe_input-0,edge0.router-0,edge0.chosen-0,edge0.routed-0,edge0.shared_gate-0,edge0.shared-0,layer_output-0,qwen35.greedy_token_ids
RMI_SCALAR=1 RMI_PARITY_TRACE="$TRACE" RMI_PARITY_FILTER="$EDGE0_TRACE_FILTER" \
  target/release-fast/rust-model-inference --model "$MODEL/Edge0-35B-A3B-preview-lossless.gguf" \
  --prompt Hello --threads 1 --kv-cache f32 --prefill-batch-size 1 --max-tokens 4 --temp 0
"$PYTHON" tools/oracle/edge0/check_scalar.py "$MODEL" "$TRACE"
```

完整文本路径使用原始 safetensors 的独立 C/Python 标量参考；trace 必须使用新的临时目录，以免旧文件的同名 checkpoint 序号混入本次结果：

```sh
mkdir -p target/oracle
cc -std=c11 -O2 -ffp-contract=off -fno-vectorize -fno-slp-vectorize \
  -shared -fPIC tools/oracle/edge0/scalar_affine.c \
  -o target/oracle/libedge0_scalar.dylib
TRACE_DIR=$(mktemp -d /tmp/edge0-parity-XXXXXXXX)
TRACE="$TRACE_DIR/trace"
FILTER=$("$PYTHON" -c 'print(",".join([
    "edge0.embedding", "edge0.norm-0", "edge0.qkv-0", "conv_output_raw-0",
    "q_conv_predelta-0", "k_conv_predelta-0", "state_predelta-0", "new_state-0",
    "final_output-0", "edge0.recurrent_projection-0", "Qcur_normed-3",
    "Kcur_normed-3", "result_norm", "result_output",
] + [f"layer_output-{i}" for i in range(40)]))')
RMI_SCALAR=1 RMI_PARITY_TRACE="$TRACE" RMI_PARITY_FILTER="$FILTER" \
  target/release-fast/rust-model-inference \
  --model "$MODEL/Edge0-35B-A3B-preview-lossless.gguf" \
  --prompt Hello --threads 1 --kv-cache f32 --prefill-batch-size 1 \
  --max-tokens 4 --temp 0
"$PYTHON" tools/oracle/edge0/token_scalar.py "$MODEL" "$TRACE" --layers 40 \
  --token-ids 248045,846,198,9419,248046,198,248045,74455,198,248068,271,248069,271,9419,0,2500
```

已验证：四组文本的 token IDs 一致，`Hello` 的四步 greedy IDs 与官方 Edge0 同为 `[9419, 0, 2500, 628]`。独立标量检查器从原始 safetensors 复算 13 个 prompt token 和后续 3 个生成输入：640 个层输出、16 组最终归一化、16 组各 248,320 个 logits，共 5,316,608 个 F32 值与真实 GGUF CLI 逐位一致；第 0 层后续 token 的 recurrent 状态及第 3 层 Q/K 归一化也逐位一致。首 token 的详细检查另覆盖 567,585 个 F32 值及 4 个专家 ID，包括 QKV/LoRA、卷积、加性 `1e-6` Q/K RMSNorm、beta、524,288 个 recurrent 状态值、输出投影、router、4 个选中专家及共享专家；所选专家为 `[129, 38, 132, 213]`。

数值对齐使用标量 C/Python 与 `RMI_SCALAR=1`，禁用自动向量化，并在 RoPE 标量路径避免 FMA；不调用 MLX、BLAS 等外部加速库。Qwen3.5 保留原有 L2 计算。官方 MLX 的 BF16/Metal 输出仅用于 token 行为对照，不用于 F32 位级对齐。当前模型不含视觉权重，本验证只覆盖固定的文本 prompt 和续写。
