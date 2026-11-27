# Laya F32 标量对齐

## 固定参考

- 官方 Laya commit：`4066d5d5fbf08b66c6757ddeedbd797bd7655bc0`，checkout 必须干净。
- HF `convaiinnovations/laya-multilingual` revision：`e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`。
- `model.safetensors` SHA256：`9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204`。
- `laya-multilingual-F32.gguf` SHA256：`5c1ee61b8312424819d16e751a17d1edc1bf77b2d6a8e9cf71a577e018e81dc4`。
- 已验证 macOS ARM64、PyTorch 2.8.0、Transformers 5.0.0。

`trace_laya.py` 保留官方模型、Tokenizer、请求处理和计算图。`Scalar`
在 TorchDispatch 入口替换全部浮点计算：独立 C 单线程 F32 内核、不使用
BLAS/Accelerate/oneDNN、关闭向量化及 FMA。Torch 仅负责布局、索引和复制；
未覆盖的浮点算子直接报错。RoPE 频率也从官方公式重新标量计算。
`environment.json` 记录版本、权重哈希、C 编译参数与内核调用次数。

数值契约：点积、求和均按元素顺序累加 F32；LayerNorm 使用两遍总体方差；
SDPA 先计算 QK 点积再缩放，随后标量 softmax 和 PV；GELU 使用 libm `erff`；
RoPE 使用独立 `sinf`/`cosf`，避免 macOS 合并 sincos 的舍入差异。
这是官方计算图的标量基线，验证范围不包括原始加速 PyTorch 内核的位模式。

## 复现

在仓库根目录运行；Python 依赖使用仓库 `.venv`。`target/laya-oracle` 是上述
commit 的官方 checkout。每次 Rust trace 使用新目录，因为 trace 文件会追加。

```sh
RUSTFLAGS='-C target-cpu=native -C no-vectorize-loops -C no-vectorize-slp -C llvm-args=-fp-contract=off' \
CARGO_TARGET_DIR=target/laya-scalar \
cargo build --profile release-fast --features parity-trace --bin rust-model-inference

model_dir=/Users/gouzi/Documents/git/rust-model-inference/models/laya-multilingual
run_dir=$(mktemp -d "$PWD/target/laya-validation/run.XXXXXX")
for fixture in laya-request laya-scalar-edge; do
  mkdir -p "$run_dir/$fixture/native"
  .venv/bin/python tools/oracle/laya/trace_laya.py trace \
    "$model_dir" target/laya-oracle "tests/fixtures/$fixture.json" "$run_dir/$fixture/oracle"
  RMI_PARITY_TRACE="$run_dir/$fixture/native/trace.jsonl" \
    target/laya-scalar/release-fast/rust-model-inference \
    --model "$model_dir/laya-multilingual-F32.gguf" \
    --laya-request "tests/fixtures/$fixture.json" > "$run_dir/$fixture/native/result.json"
  .venv/bin/python tools/oracle/laya/trace_laya.py compare \
    "$run_dir/$fixture/oracle/trace.jsonl" "$run_dir/$fixture/native/trace.jsonl"
done

.venv/bin/python tools/oracle/laya/test_scalar.py
```

## 已验证范围

2026-09-27：两组 fixture，共 5 个问题、195 个 checkpoint、15,586,581 个 F32
值逐位一致。比较包含 token IDs、marker、checkpoint 顺序/shape、embedding、
所有 22 层编码器、2 层决策头、最终 logits 和 action logits；第 0 层额外细分
QKV、RoPE、attention 和 MLP。两组最终解码 JSON 也完全一致（概率保留四位小数）。

- 常规 fixture：中文 choice、score、noul，序列长度 62/58/55。
- 边界 fixture：单候选、结构化 instructions/criteria、会话数组左截断、Unicode、
  空白、mask 转义、自定义 noul labels，两个 160-token 序列覆盖局部窗口外的屏蔽。

验证仅覆盖 F32 标量 forward。Laya 是决策模型，没有自回归生成/KV 续写。
SIMD、FMA、外部加速库、量化权重和其他平台不在本次逐位验证范围内。
