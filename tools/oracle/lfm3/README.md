# LFM3-1.7B 适配与验证

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

CLI 和 HTTP 为 `lfm3` 默认渲染 GGUF 自带 Jinja 模板，支持 thinking 和模板覆盖；
JEV 单项/分组打分使用同一模板并关闭 thinking。model API 的 session 保留卷积历史及 KV，
支持续写与 reset。共享 `LinearExecutor` 负责所有投影，Q/K/V/head gate 和 FFN gate/up
成组执行并复用激活量化；CPU 精度与 SIMD 分发留在共享算子内部。

`--compute cpu` 固定 CPU；`--gpu` / `--compute auto` 尝试 Vulkan 投影并允许 CPU 回退。
attention、短卷积和 KV 仍在 CPU，`--compute vulkan` 的全图要求明确拒绝。

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
| Jinja thinking chat | 17 | 8/8 | 1.007810 | 0.154821 |
| `你好，LFM3!\n\t café  🦀` | 13 | 8/8 | 0.520325 | 0.144917 |
| 混合英文重复文本，跨过 SWA 窗口 | 265 | 8/8 | 0.398313 | 0.067313 |

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
24 次 embedding 逐位一致。当前第一个浮点位差异位于位置 1、layer 2 的 Q RoPE，
最大 `4.7683716e-7`；位置 4、layer 0 的 shortconv 仍有 `3.7252903e-9` 的累加差异。
这些局部舍入差异会在后续 Q8_0 激活重新量化时放大，见下面的归因探针。
短卷积使用 FMA；SWA/full RoPE、gate、sandwich norm 和状态写回均按参考图实现。

按用户要求，有损的量化、浮点累加、softmax/exp、RoPE 和 SIMD 路径使用数值及
多步 greedy 检查，不声明完整 logits 逐位一致。回归上限为每步完整 logits
`max_abs <= 1.25`、`RMSE <= 0.20`，同时要求有限值和 8 个 greedy IDs 完全一致；
这些是上述固定模型/样例的经验界限，不是任意输入或硬件的误差保证。

无需模型的测试覆盖非法 metadata/shape/字节长度、post norm 配对、RoPE 因子、
上下文边界、卷积超过六步的历史、分块/增量/reset 一致性和 F16/F32 KV。

## 范围

已验证真实 Q8_0/F32 KV 的 CPU CLI、HTTP、JEV 单项/分组和 model API，
以及 M3 Max/MoltenVK 的 Auto 投影。F16 KV 仅有合成状态回归。
CPU 合成测试覆盖 F32/F16/BF16/Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K 投影及混合权重；
Q4_1 embedding 不在共享 lookup 支持表中，加载时拒绝，测试使用合法的 F32 embedding。
其他真实权重量化档、其他 GPU、全图 Vulkan、多模态和工具调用未验证或不支持。
分块 prefill 仍按 token 因果执行，未做矩阵批处理优化；未运行全仓测试套件，
不声称 Auto 投影带来端到端加速。

## Review 修复与误差归因

已合入上游 main `8d18565b05c76897e46752aca98e83f29e24b452`。
入口、状态、数值和设备证据汇总在 [review-fixes.json](review-fixes.json)。

同一聊天样例、固定原生 CPU Oracle、每次只改变 Rust 侧路径：

| Rust 路径 | max abs | 最大逐步 RMSE |
| --- | ---: | ---: |
| 原 fused sin/cos | 1.115660 | 0.171233 |
| 独立 sin/cos（当前路径） | 1.007810 | 0.154821 |
| 独立 sin/cos + `RMI_SCALAR=1` | 0.730175 | 0.141612 |

第一项实验只改变 RoPE，证实它会影响最终误差，但不是唯一来源。scalar 控制同时改变
SIMD 归约和激活量化的舍入，不能把结果单独归于 attention dot。未为了逐位一致而
禁用默认 SIMD 或替换共享算子。

审阅对 softmax 的判断需要修正：固定 Oracle 的 `ggml_vec_soft_max_f32` 在 NEON 上使用
向量 exp 近似、四路 F32 横向求和，再累加到 `ggml_float`；Rust 使用 libm exp 和逐值
F64 求和。不是简单的“Oracle 全程 F32 求和”。两侧原生 NEON Q8_0 激活量化都使用
ties-to-even，不能把原生路径的差异直接解释成 tie 规则错误。

比较器可让固定 Oracle 的同一个 Q8_0 量化函数分别处理双方的 norm 检查点，隔离
“输入变化”与“量化实现变化”：

```bash
python3 tools/oracle/lfm3/compare_checkpoints.py /tmp/lfm3-trace/chat \
  /tmp/llama-cpp-lfm3/build-lfm3-cpu/bin/libggml-cpu.dylib
```

实测首个发生 Q8 字节变化的 **norm 输入**在位置 4、layer 13 的 operator norm：
输入最大差 `0.00017584115`，量化后 10 个字节不同。它证实了重新量化的非连续放大；
各算子的最终贡献不能按上表简单相加。

探针现在覆盖**每一个量化投影的输入**，而不只是两个 norm：`attn_norm`（q/k/v/head-gate
投影的输入）、`ffn_norm`（gate/up 的输入）、`conv_conv`（短卷积输出投影的输入）、
`result_norm`（输出头的输入），以及 `attn_values`（**sigmoid head gate 缩放后**、
输出投影的输入）。`attn_values` 需要 Oracle 侧多一个检查点，
`tools/oracle/lfm3/checkpoints.patch` 已在 `src/models/lfm3.cpp` 里加上
`cb(attn_out, "lfm3_attn_gated", il)`；不打这个补丁时该输入被跳过而不是误判。
`ffn_swiglu`（down 投影的输入）在共享的 `build_ffn()` 内部，暴露它要改所有模型的图，
因此保持未探测。

报告字段：`q8_amplification_points` 列出全部发生 Q8 字节变化的投影输入
（含 `input_max_abs` 与 `quantized_bytes_differ`），
`projection_inputs_available` / `projection_inputs_without_oracle_checkpoint`
说明哪些输入可测、哪些缺 Oracle 检查点，`first_norm_q8_change` 保留首个以兼容旧记录。

这个探针量的是**数值裕度**，不是比特对齐：它回答“哪里的输入差异容易被量化放大”，
而不是“哪里还没对齐”。残差本身是量化固有的损失，不是待修的缺陷。

`1.25 / 0.20` 保留为固定模型/样例的经验回归报警线，并非推导出的全图误差上界或
“精确对齐”证明。真实运行状态与数值精确性分别描述；有损路径继续检查有限值、完整
logits 误差和多步 greedy，Tokenizer/embedding/检查点结构保留精确检查。

## CLI / HTTP / JEV / Auto 复现

```bash
cargo +1.98.1 build --profile release-fast --features 'vulkan parity-trace'
RMI_COMPUTE_TRACE=1 target/release-fast/rust-model-inference \
  --model models/LFM3-1.7B-GGUF/LFM3-1.7B-Q8_0.gguf \
  --prompt 'Reply with exactly one short sentence.' --thinking \
  --max-tokens 8 --temperature 0 --threads 1 --max-context 512 --kv-cache f32 --gpu

RMI_LFM3_GPU=1 RMI_COMPUTE_TRACE=1 RMI_LFM3_CASE=chat \
  RMI_LFM3_ORACLE=/tmp/lfm3-oracle \
  cargo +1.98.1 test --profile release-fast --features 'vulkan parity-trace' \
  --test lfm3_reference lfm3_oracle_alignment -- --nocapture

target/release-fast/rust-model-inference \
  --model models/LFM3-1.7B-GGUF/LFM3-1.7B-Q8_0.gguf \
  --jev --jev-context 'An apple is red.' --jev-question 'What color is the apple?' \
  --jev-option red --jev-option blue --compute auto --threads 1

RMI_COMPUTE_TRACE=1 target/release-fast/rust-model-inference \
  --model models/LFM3-1.7B-GGUF/LFM3-1.7B-Q8_0.gguf \
  --serve --host 127.0.0.1 --port 8080 --compute auto --kv-cache f32 --threads 1
```

真实 HTTP 验证包括 `/v1/chat/completions`（thinking 开/关、SSE）、`/v1/jev/score`、
`/v1/jev/grouped` 和超限请求 400；CPU/Auto 的文本、token usage 和 JEV 概率一致。
CLI/HTTP runtime 对照覆盖 thinking 开/关；另修复了只有完整 HTTP 链路才能发现的
LFM3 输出解析器未注册导致的 500。

聊天 Auto Oracle 检查得到同样的 8 步 greedy 和上表当前 CPU 误差；实测 2,312 次
Vulkan 投影分发、3,681 次设备提交、155 次静态上传（1,836,440,640 bytes）。这些计数
证明投影确实上 GPU，不表示 attention/短卷积也在设备端。设置无效 `VK_DRIVER_FILES`
还验证了同一二进制 Auto 回退 CPU 后正常生成，并打印失败原因。
