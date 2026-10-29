# LFM3-1.7B CPU 对齐

适配 [atomic.chat/LFM3-1.7B-GGUF](https://modelscope.cn/models/atomic.chat/LFM3-1.7B-GGUF/files)
的 `LFM3-1.7B-Q8_0.gguf`。验证环境为 macOS ARM64，单线程 CPU、F32 KV、greedy。

## 固定契约

- 文件大小：`1,846,238,336` bytes；SHA-256：
  `4dc04caea646076de9502b41b6c6db52203945374246d928d91e70ef5ec8fdcb`。
- `general.architecture=lfm3`；24 层、2048 hidden、16 query heads、128 head dim、7680 FFN。
- KV heads：`[0,0,4,0,4,0,4,0,4,0,4,0,4,0,4,0,4,0,4,0,4,0,4,0]`。
  零表示 shortconv，kernel 长度 7、历史长度 6；注意力层使用 sigmoid head gate。
- 0-based SWA 层 `[4,8,12,16,20]`，窗口 256、RoPE base `10000`；
  全局层 RoPE base `1000000`，并除以 `rope_freqs.weight` 的 per-pair 因子
  （前 32 个为 1，后 32 个约为 `1e30`）。
- `post_attention_norm` / `post_ffw_norm` 必须成对出现；本模型两者都有。
  缺少 `output.weight`，输出头与 embedding 绑定。
- GPT2 BPE、`tokenizer.ggml.pre=lfm2`；BOS `124894`、EOS `124900`，
  `add_bos_token=false` / `add_eos_token=false`。

CLI 为 `lfm3` 默认渲染 GGUF 自带 Jinja 模板，支持 `--thinking` 和
`--chat-template-file`；CPU model API 的 session 保留卷积历史及 KV，支持续写与 reset。

## 参考实现与复现

Oracle 固定为 [llama.cpp `19cce6919992fdffd92e0ac152caaf1e2b8f254e`](https://github.com/ggml-org/llama.cpp/commit/19cce6919992fdffd92e0ac152caaf1e2b8f254e)，
实现为 `src/models/lfm3.cpp` / `src/models/lfm-base.cpp`。
在独立临时 checkout 构建，禁用 Accelerate、BLAS、Metal、llamafile 和 OpenMP；
实现及验证均未引入外部加速库。以下命令在仓库根执行：

```bash
git clone https://github.com/ggml-org/llama.cpp /tmp/llama-cpp-lfm3
git -C /tmp/llama-cpp-lfm3 checkout 19cce6919992fdffd92e0ac152caaf1e2b8f254e
# 只添加 Q/K RoPE 检查点，不改变计算。
git -C /tmp/llama-cpp-lfm3 apply "$PWD/tools/oracle/lfm3/checkpoints.patch"
cmake -S /tmp/llama-cpp-lfm3 -B /tmp/llama-cpp-lfm3/build-lfm3-cpu \
  -DCMAKE_BUILD_TYPE=Release -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF \
  -DGGML_METAL=OFF -DGGML_LLAMAFILE=OFF -DGGML_OPENMP=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build /tmp/llama-cpp-lfm3/build-lfm3-cpu --target llama -j 8
c++ -std=c++17 -O2 tools/oracle/lfm3/oracle.cpp \
  -I /tmp/llama-cpp-lfm3/include -I /tmp/llama-cpp-lfm3/ggml/include \
  -L /tmp/llama-cpp-lfm3/build-lfm3-cpu/bin \
  -Wl,-rpath,/tmp/llama-cpp-lfm3/build-lfm3-cpu/bin \
  -lllama -lggml -lggml-base -o /tmp/lfm3-oracle

RMI_LFM3_FORWARD_SMOKE=1 RMI_LFM3_ORACLE=/tmp/lfm3-oracle \
  cargo +1.98.1 test --profile release-fast \
  --test lfm3_reference --test lfm3_synthetic -- --nocapture
RMI_LFM3_ORACLE=/tmp/lfm3-oracle RMI_LFM3_LONG_CONTEXT=1 \
  cargo +1.98.1 test --profile release-fast \
  --test lfm3_reference lfm3_oracle_alignment -- --nocapture
```

默认模型路径为 `models/LFM3-1.7B-GGUF/LFM3-1.7B-Q8_0.gguf`，可用
`RMI_LFM3_MODEL` 指定其他副本。Oracle runner 接受模型、UTF-8 prompt 文件、
输出目录、生成数量；输出 prompt/generated IDs 和每一步完整 F32 logits。
`N_GENERATE=0` 仅检查 tokenizer。数值测试每次实际调用 Oracle，不使用历史 logits。

保存逐层检查点并核对名称、shape、次数、embedding 原始位模式及首个浮点差异：

```bash
RMI_LFM3_ORACLE=/tmp/lfm3-oracle RMI_LFM3_CASE=chat \
  RMI_LFM3_CHECKPOINTS=1 RMI_LFM3_ALIGN_DIR=/tmp/lfm3-trace \
  cargo +1.98.1 test --profile release-fast --features parity-trace \
  --test lfm3_reference lfm3_oracle_alignment -- --nocapture
python3 tools/oracle/lfm3/compare_checkpoints.py /tmp/lfm3-trace/chat
```

Rust 仅在 prefill 最后一步和每次 decode 投影输出头，所以检查点比较器只在这些
位置核对 `result_norm` / `result_output`；其余共同检查点覆盖每个输入位置。

## 2026-10-29 结果

| 输入 | Prompt token 数 | Greedy 对齐 | 8 步中最大 logits abs error | 最大逐步 RMSE |
| --- | ---: | --- | ---: | ---: |
| `Hello` | 1 | 8/8 | 0.243883 | 0.048465 |
| Jinja thinking chat | 17 | 8/8 | 1.115660 | 0.171233 |
| `你好，LFM3!\n\t café  🦀` | 13 | 8/8 | 0.533533 | 0.095196 |
| 混合英文重复文本，跨过 SWA 窗口 | 265 | 8/8 | 0.452057 | 0.068492 |

四组 prompt IDs 全部完全一致。另有空文本、标点、空白、Unicode、BOS/EOS/control
静态 tokenizer fixtures，期望值由固定 Oracle 得到。

聊天 prompt 为 `Reply with exactly one short sentence.`；Jinja 结果为：

```text
[124894,124899,5922,207,57922,415,8043,734,2789,12683,22,124900,207,124899,63514,207,124901]
```

双方 greedy IDs 为 `[597,4695,20589,34,496,57922,415,8043]`。
实际 CPU CLI 自动模板入口也得到相同 prompt/generated IDs：

```bash
cargo +1.98.1 run --profile release-fast --bin rust-model-inference -- \
  --model models/LFM3-1.7B-GGUF/LFM3-1.7B-Q8_0.gguf \
  --prompt 'Reply with exactly one short sentence.' --max-tokens 8 \
  --temperature 0 --threads 1 --max-context 512 --kv-cache f32 --thinking
```

聊天过程覆盖 24 个输入位置、5,176 个共同检查点，名称、shape、次数符合预期；
24 次 embedding 逐位一致。第一个浮点位差异位于位置 4、layer 0 的 shortconv
累加，最大 `3.7252903e-9`。这种浮点累加差异会在后续 Q8_0 激活重新量化时放大。
短卷积使用 FMA；SWA/full RoPE、gate、sandwich norm 和状态写回均按参考图实现。

按用户要求，有损的量化、浮点累加、softmax/exp、RoPE 和 SIMD 路径使用数值及
多步 greedy 检查，不声明完整 logits 逐位一致。回归上限为每步完整 logits
`max_abs <= 1.25`、`RMSE <= 0.20`，同时要求有限值和 8 个 greedy IDs 完全一致；
这些是上述固定模型/样例的经验界限，不是任意输入或硬件的误差保证。

无需模型的测试覆盖非法 metadata/shape/字节长度、post norm 配对、RoPE 因子、
上下文边界、卷积超过六步的历史、分块/增量/reset 一致性和 F16/F32 KV。

## 范围

已验证 CPU CLI/model API 和真实 Q8_0、F32 KV；F16 KV 仅有合成状态回归。
HTTP TextRuntime、多模态、其他权重量化档及 GPU 未纳入支持范围。
分块 prefill 仍按 token 因果执行，未做矩阵批处理优化；未运行全仓测试套件。
