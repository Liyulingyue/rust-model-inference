# Timestral-209M CPU / Vulkan F32

原生 Rust 预测，不依赖 Python 推理或外部加速库。显式 `general.architecture=timestral`，仅接受本目录对应的官方 209M 配置和 F32 权重；其他尺寸/量化格式会被拒绝。

固定输入：

- HF `mistralai/Timestral-209M` revision `9874d252ab0aae16f7f7c999f98f98d6ad7d8365`。
- `model.safetensors`：837,271,648 bytes，SHA-256 `1456effb9927e3daa8c7234f10497e3424233869245a9c72b8ecebaf7ad2b7fa`。
- 203 个 F32 张量、209,312,256 个参数；24 层，d=768，12 heads，head_dim=64，FFN=2048，patch_size=32。每四层有一次变量 attention。
- GGUF：837,265,632 bytes，SHA-256 `7628bcbfe43c8fec82bfbeace0ebbeb7dcd49facec6a785a51f430995d704e10`。名称、shape、全部 F32 payload 与官方权重精确一致；不重排 QKV、不量化。
- Oracle：[mistralai/timestral](https://github.com/mistralai/timestral/tree/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff)，commit `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`；PyTorch 2.9.1，NumPy 2.5.3，safetensors 0.8.0，CPU 单线程/F32。只使用运行时 hooks，源代码保持干净。

## 转换与运行

所有权重统一保存在主仓库的 models 下。本地需要配置和权重两个文件，不需要 tokenizer、chat template 或 mmproj。

```sh
MODEL_DIR=/Users/gouzi/Documents/git/rust-model-inference/models/Timestral-209M
# config.json 和 model.safetensors 从固定 HF revision 按文件下载到 MODEL_DIR。
.venv/bin/python -m tools.converter.timestral.convert_timestral \
  "$MODEL_DIR" "$MODEL_DIR/Timestral-209M-F32.gguf"
cargo build --profile release-fast --bin rust-model-inference
./target/release-fast/rust-model-inference \
  --model "$MODEL_DIR/Timestral-209M-F32.gguf" \
  --timestral-request series.json --threads 4
```

`series.json`：

```json
{"context":[1,2,null,4,5,6,7,8],"prediction_length":96}
```

`null` 表示缺失值。`context` 支持 `[time]` 或 `[variate][time]`；二维输入表示同一个序列的多变量联合预测。输出 JSON 为 `quantile_levels`（21 个分位点）和 `quantiles[variate][horizon][quantile]`，中位数在索引 10。

可选字段：

- `observation_mask`：`[target variate][context time]` 布尔矩阵。
- `past_covariates`：`[covariate][context time]` 数值矩阵。
- `known_covariates`：`[covariate][context time + prediction_length]` 数值矩阵；均允许 `null`。
- `symmetric_averaging` / `clip_nonnegative`：默认 `true`，与官方一致。
- `max_context`：默认 32768，范围 1..32768，只保留最近的历史并向左补齐 patch；非负裁剪的资格仍基于完整输入历史。
- `per_shot_horizon_patches`：默认 32，范围 1..32。超过一个 block 时使用中位数反馈，已知未来协变量使用真实值，past-only 协变量使用缺失值。每个 block 都应用 advancing refine frame。

不接受未知 JSON 字段。暂不提供 categorical target encoding、批次 API 或 HTTP；categorical 协变量不会被静默当作数值协变量处理。其他 Timestral 尺寸和量化格式未经验证。

## GPU 运行

```sh
cargo build --profile release-fast --features vulkan --bin rust-model-inference
./target/release-fast/rust-model-inference \
  --model "$MODEL_DIR/Timestral-209M-F32.gguf" \
  --timestral-request series.json --compute vulkan --threads 4
```

`--compute vulkan` 要求真实硬件 Vulkan；缺少编译 feature、设备初始化或执行失败时返回错误。`--gpu` / `--compute auto` 优先使用 Vulkan，无法使用时明确记录 CPU 回退；`--compute cpu` 强制 CPU。

24 层 Transformer 的全部 F32 投影、RMSNorm、partial RoPE、时间/变量 attention、sigmoid gate、SwiGLU 和残差在 GPU 执行。复用仓库 Vulkan F32 matmul/RMSNorm/add；Timestral shader 处理 QKV head 布局、双轴 mask 和在线 softmax，不构造平方大小的 attention 分数矩阵。权重按模型缓存；块变大时重建 arena 并重新上传。输入缩放、patch/head 投影、分位数变换和预测反馈仍在 CPU；这是 CPU/GPU 混合预测入口。

GPU 与同一个官方 CPU/F32 Oracle 比较，沿用下面的误差界，不要求浮点逐位一致。复现时在下节 parity 命令追加 `--mode vulkan`；可同时添加 `--mode simd --mode auto`。GPU 验收还检查每个块实际报告 Vulkan 后端和不少于 252 个投影/Timestral dispatch，CPU 回退不算通过。`RMI_COMPUTE_TRACE=1` 可输出后端、dispatch 和设备累计计数。普通运行不读取中间激活；parity trace 为比较而增加同步和读回。

GPU 对齐构建和命令：

```sh
cargo build --profile release-fast --features vulkan,parity-trace --bin rust-model-inference
cp target/release-fast/rust-model-inference target/timestral-gpu-cli
.venv/bin/python -m tools.oracle.timestral.parity \
  --oracle reference/timestral --model-dir "$MODEL_DIR" \
  --gguf "$MODEL_DIR/Timestral-209M-F32.gguf" \
  --binary target/timestral-gpu-cli --output-dir target/timestral-gpu-parity \
  --mode simd --mode vulkan --mode auto
```

Shader 二进制使用与 CI 相同的 glslang 15.1.0 生成；16.6.0 的输出字节不同。`scripts/vulkan-shaders.sh check` 的本地复现需要把 15.1.0 的 `glslangValidator` 放在当前命令的 PATH 前端，无需修改全局环境。

## 对齐复现

在仓库 `.venv` 安装 Oracle 所需依赖，依赖不进入 Rust 运行时：

```sh
uv --cache-dir target/uv-cache pip install --python .venv/bin/python \
  torch==2.9.1 numpy==2.5.3 safetensors==0.8.0 huggingface_hub

git clone https://github.com/mistralai/timestral.git reference/timestral
git -C reference/timestral checkout a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff
cargo build --profile release-fast --features parity-trace --bin rust-model-inference
cp target/release-fast/rust-model-inference target/timestral-cli
.venv/bin/python -m tools.oracle.timestral.parity \
  --oracle reference/timestral --model-dir "$MODEL_DIR" \
  --gguf "$MODEL_DIR/Timestral-209M-F32.gguf" \
  --binary target/timestral-cli --output-dir target/timestral-parity
```

脚本先验证 Oracle commit/干净状态、官方权重哈希、GGUF 张量集合、shape/type 和每个 payload 的原始字节，再运行真实 CLI 的 `RMI_SCALAR=1` 单线程及默认 SIMD 四线程路径。

逐项检查 checkpoint 顺序、shape、次数、输入缺失通道、时间/变量 mask、位置；后者精确比较 F32 位模式。没有 tokenizer，故不做 token ID/greedy 文本对齐。连续值预测以完整 21 分位数数组为最终结果。

浮点归约顺序、SIMD/FMA、RMSNorm、SiLU/softmax、asinh/sinh 和 attention 与 PyTorch 不保证逐位一致。每个中间张量和最终输出同时要求：

- `max_abs_error <= 3e-5 + 3e-4 * max_abs(reference)`；
- `RMSE <= 3e-5 + 5e-5 * RMS(reference)`。

绝对阈值同时用于最大误差和 RMS，避免接近零的张量仅因相对误差变大而误报；原始 NRMSE 仍记录为 `RMSE / max(RMS(reference), 1e-6)`。比较器自身覆盖错误顺序/shape、位模式、数值和缺失 checkpoint 的拒绝，以及近零数值的验收测试。`report.json` 记录每个 case/执行模式的 checkpoint 数、最差归一化误差和最终误差。全缺失输入可能有极大的中间 quantile tails，需结合相对误差和官方 cap 后的最终输出评估。

用例覆盖缺失值/非整 patch、联合多变量、全缺失变量、numeric past/known covariates（含 0/1 指示变量）、常量/全缺失历史、显式观测掩码、窗口裁剪、分块反馈、1025 步 horizon，以及负值历史被裁剪后的非负裁剪资格。

验证命令：

```sh
cargo fmt --all -- --check
cargo test --profile release-fast --lib timestral
cargo test --profile release-fast --test timestral_cli
.venv/bin/python -m unittest tools.converter.timestral.test_convert_timestral tools.oracle.timestral.test_parity
```

CPU 验证使用 ARM64/F32；没有对 32768 点最大上下文做性能或预测精度 benchmark，不据合成数据对齐结果声称达到官方数据集指标。

实测摘要见 [verification.json](verification.json)：10 个用例、20 次真实权重 CLI 执行、6,788 个检查点。最终预测最差 NRMSE `7.50e-6`；最大绝对误差 `0.0962` 出现在长 horizon 的大幅值尾部分位数。近零用例的中间 NRMSE 可达 `7.01e-5`，对应绝对偏差仅 `8.1e-8`，满足上述混合误差界。权重和结构检查精确一致。

工程验证：8 个模型单测、2 个 CLI 测试、2 个 Python 检查通过；最新优化 CLI 再验数值协变量及近零裁剪用例。默认全量 `cargo test` 在未修改的 `vk_mem_info` 示例编译失败；排除示例后的库测试为 1224 passed / 23 failed / 79 ignored，全部失败名称保存在验证记录中，未继续宣称其余集成测试通过。失败位于本次未修改的模型/算子路径；不据此声明全库通过。


GPU 实测见 [gpu_verification.json](gpu_verification.json)：M3 Max / MoltenVK 1.4.2，最终 glslang 15.1.0 shader；10 个用例在 SIMD / 严格 Vulkan / Auto 下执行 30 次，比较 10,182 个检查点。其中 20 次 GPU 执行、6,788 个检查点全部通过同一误差界，每个 Transformer 块实际记录 252 个投影/Timestral dispatch。GPU 最差最终 NRMSE 为 `5.84e-6`；最大最终绝对误差 `0.0703125` 出现在长 horizon 的大幅值尾部分位数，中间尾部分位数绝对误差可达 `4736`，不能据此声称所有绝对误差都很小。Vulkan 与 Auto 的最终预测逐位一致，但它们与 Oracle 的浮点计算按数值界验收。

另有 4 次普通 GPU CLI（严格 Vulkan / `--gpu`、关闭 parity trace）与插桩 GPU 的预测精确一致；设备不可用时 CPU/Auto 输出精确一致，严格 Vulkan 返回错误。8 个模型单测、3 个 CLI 测试、3 个 Python 检查、CPU-only 库编译、Vulkan 构建和完整 shader 检查通过。GPU 扩展后未重复全量测试，原有失败见上文记录。

仅实测 M3 Max/MoltenVK；其他 GPU、最大上下文性能和数据集预测指标未验证，不声明加速倍数。构建工具和依赖均保留在仓库 .venv / target，模型文件始终从上述主仓库 models 目录读取。
