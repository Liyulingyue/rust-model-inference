# LongCat Image Edit / Edit Turbo：Q8_0 Transformer 对齐

## 范围

新增 `LongCatTransformer`，由测试专用的 `tests/longcat_reference.rs` 读取用户提供的两个 GGUF 并对照 Oracle。
此处的逐位对齐只支持 **batch=1、CPU 标量、Q8_0 主干 + BF16 输入/时间/输出投影**。
输入是已经编码的文本特征和已经打包的 latent，可包含目标图和多个参考图。
本机已经补齐 Edit 的 Qwen2.5-VL-7B 文本/视觉权重、Tokenizer/processor 和 Flux VAE；Turbo 目录只含配置，按官方发布契约复用 Edit 的这些组件。
两个版本共用 Transformer 架构，使用各自权重；另有实验性 Rust CLI 串接图文编码、采样、VAE 和最终 PNG。

**Transformer 的逐位证据不覆盖完整图片编辑入口**。实验 CLI 读取 Qwen2.5-VL-7B / 3584 维编码器、Tokenizer/processor 和 Flux VAE safetensors；本轮已把同一真实 Edit 输入的 Qwen2.5-VL 视觉编码输出与官方 Oracle 逐位对齐（196×3584，702,464 个 F32 words）。文本 hidden state、VAE、采样过程及最终 PNG 尚未通过完整官方 Oracle 数值对齐。
Tokenizer、图像预处理、latent packing/unpacking、scheduler、最终图片不在本次 Transformer 对齐范围。
已有 Qwen2.5-VL-3B 或 YuE2 VAE 不能替代这些组件。通用 `--image` 入口不会把任意 `flux` 权重当作 LongCat。

## 模型契约

| 版本 | 文件 | 大小 | SHA256 |
|---|---|---|---|
| Edit | `LongCat-Image-Edit-GGUF/LongCat-Image-Edit-Q8_0.gguf` | 6,725,439,840 bytes | `a998d17d51943ab15bf3ba3ca1a509be3792987395c3bcb8d6bcbfafa7030a42` |
| Edit Turbo | `LongCat-Image-Edit-Turbo-GGUF/LongCat-Image-Edit-Turbo-Q8_0.gguf` | 6,725,439,840 bytes | `ac4ae6172eea6dc892ba8c1cb63ec5b1232a294d0c430299ec26006f956e437b` |

两者均为 414 个同名、同 shape、同 dtype 张量，metadata 仅含 `general.architecture=flux`、`general.quantization_version=2`、`general.file_type=7`，没有 Tokenizer 或架构维度 metadata。
`LongCatTransformer::load` 验证完整必要张量的名称、shape、dtype、字节数，并拒绝其他 Flux 的 vector/guidance embedding 和额外层；测试按 `edit` / `turbo` 显式选用对应 GGUF。

- hidden=3072，24 heads × 128，text=3584，packed image=64。
- 10 double blocks，20 single blocks，MLP=12288。
- RoPE axes `[16,56,56]`，theta=10000。
- timestep 在 `[0,1]`，内部乘 1000，256 维 cos/sin，SiLU MLP。
- `norm_out.linear` 与 `final_layer.adaLN_modulation.1` 同时存在。参考 loader 对带 `model.diffusion_model.` 前缀的原始名称排序后转换，Q8_0 `norm_out.linear` 覆盖 BF16 duplicate，**包括 bias**。Rust 使用相同有效权重，直接按 `[shift,scale]` 拆分，不额外 swap。

## 固定参考与精度设置

- stable-diffusion.cpp：`3f8527a46c54ecf4cb4ed6003da8e8982283c73c`。
- ggml：`89c4413f5da6fb20cc796f16033d37f129be81fd`。
- 官方架构/位置规则审计：LongCat-Image `f0e4c43c5ef74b011ff71570fbfc2bdffbc9ab06`；数值 Oracle 使用上述 GGUF-aware sd.cpp，不能将 Q8_0 对齐结论推广到官方非量化权重。

`build.sh` 从参考仓库创建临时 clone，对用户参考 checkout 不写入。
插桩只添加 checkpoint copies；数学图直接调用 upstream `Flux::forward_orig`。
选择 ggml 现有 generic CPU kernel 和现有 **F32 GELU 分支**，关闭默认 FP16 GELU 查表。
禁用 SIMD、FMA、自动向量化、repack、KleidiAI、OpenMP、Accelerate、BLAS、Metal、CUDA、Vulkan；attention 使用普通 F32 softmax/matmul，关闭 flash/sage。
Rust 复用本仓库 BF16/Q8_0 scalar kernel、RMSNorm、exact softmax、F32 dot，并保持 ggml F32 GELU 运算顺序。
不对齐会改变精度的加速路径，也不引入外部加速库。

## 运行

在仓库根目录，macOS ARM64 / Clang：

```bash
oracle=$(bash tools/oracle/longcat/build.sh /tmp/stable-diffusion.cpp-longcat)
python3 tools/oracle/longcat/check.py "$oracle" \
  /Users/gouzi/Documents/git/rust-model-inference/models
```

`check.py` 检查两份完整 SHA256，给每个模型运行两个不同输入/位置/timestep fixture（`Ni=2,Nt=2,t=0.375`；`Ni=3,Nt=1,t=0.875`），保存到独立 `target/longcat-parity-*`。脚本用禁用自动向量化的 `RUSTFLAGS`、`RMI_SCALAR=1` 调用被忽略的集成测试，不生成正式 LongCat 可执行文件。
每次严格比较 44 条 checkpoint 名称、顺序、shape 和全部 little-endian F32 原始位；任一差异立即失败并报告首个位置，未提供 tolerance 选项。
可以独立重放已有 trace：

```bash
python3 tools/oracle/longcat/compare.py ORACLE_DIR/trace.jsonl RUST_TRACE.jsonl
```

`RMI_SCALAR=1` 必须在测试进程启动前设置；非标量路径明确拒绝。
`parity-trace` 编译时可设置 `RMI_PARITY_TRACE=/absolute/trace.jsonl` 输出 checkpoint。

输入目录中的原始文件为 little-endian、row-major、F32：

| 文件 | shape | 说明 |
|---|---|---|
| `img.f32` | `[Ni,64]` | 目标图 token 在前，参考图 token 在后 |
| `txt.f32` | `[Nt,3584]` | 预先计算的文本/视觉上下文 |
| `positions.f32` | `[Nt+Ni,3]` | 文本位置在前，然后图像；每行 `[modality,row,col]` |

文本位置通常为 `[0,i,i]`，目标图 `[1,row+Nt,col+Nt]`，参考图 modality 从 2 起。
输出为 `[Ni,64]`，保留参考 token；调用方取目标 token 后再解包。

## 本次实测结果

macOS ARM64，以上两个完整模型 SHA256，Rust `release-fast` + 禁止自动向量化：

| 模型 | fixture | checkpoint 数 | 完全相同的 F32 words |
|---|---|---:|---:|
| Edit | 2 image / 2 text，t=0.375 | 44 | 384128 |
| Edit | 3 image / 1 text，t=0.875 | 44 | 384192 |
| Turbo | 2 image / 2 text，t=0.375 | 44 | 384128 |
| Turbo | 3 image / 1 text，t=0.875 | 44 | 384192 |

合计 176 条记录、1,536,640 个 F32 words；模型、输入和输出哈希见 [verification.json](verification.json)。
另外，从干净临时 clone 使用 `build.sh` 构建的 Oracle 重放普通版 fixture 也通过 44 条记录的逐位检查。
比较器实测能拒绝首 checkpoint 的单 bit 修改；测试入口拒绝非法型号。

工程检查通过：带 `parity-trace` 的 `longcat_reference` 集成测试构建及上述四组 Oracle 用例、LongCat 契约单测、默认跳过该集成测试的检查、rustfmt、`git diff --check`、Python 和 shell 语法检查。
未运行全仓测试；编译产生仓库既存 warnings，不将其视为本次精度回归。

## 本机组件审计

本机 `models/LongCat-Image-Edit` 已包含完整的 Edit 组件。五个文本编码器分片和 VAE 文件的大小、SHA256 与 [components.json](components.json) 中的固定记录一致；仅读取 safetensors header 还确认了文本编码器 729 个 BF16 tensor、VAE 244 个 BF16 tensor。

`models/LongCat-Image-Edit-Turbo` 目前只有配置和独立 scheduler。它没有重复下载 text encoder/VAE；按官方 model index，这些文件应与 Edit 共用。两个 scheduler 的本地 SHA256 和参数已经记录在 [components.json](components.json)。

官方两个 model repo 的固定 revision、组件契约和本地审计保存在 [components.json](components.json)：

- Edit：`meituan-longcat/LongCat-Image-Edit` @ `7b54ef423aa7854be7861600024be5c56ab7875a`。
- Turbo：`meituan-longcat/LongCat-Image-Edit-Turbo` @ `6a7262de5549f0bf0ec54c08ef7d283ef41f3214`。
- 两者的五份 `text_encoder/model-0000N-of-00005.safetensors` 和 `vae/diffusion_pytorch_model.safetensors` 在 Hub 上的 Git LFS SHA256 完全相同；本机 Edit 文件已逐字节计算并核对这些 SHA256，可以共用一套。
- 文本/视觉 encoder 符合该 checkpoint 的 Qwen2.5-VL-7B 契约：3584 hidden、28 层、28 heads、4 KV heads；vision 1280 hidden、32 层，输出 3584。Rust 后续可直接使用已核验的 safetensors 和 Tokenizer/processor，或使用匹配的 GGUF + mmproj。
- VAE 必须含 encoder 和 decoder：16 latent channels，`scale=0.3611`、`shift=0.1159`，无 quant/post-quant conv。原始 VAE 文件在 Hub 标记为 167,666,902 bytes。
- **scheduler 两版不同**：Edit 动态 shifting 从 `base_shift=0.5` 到 `max_shift=1.15`；Turbo 两者均为 `1.15`。不能只换 Transformer 权重并复用同一份 schedule。官方推荐 Edit 为 50 步 / CFG 4.5，Turbo 为 8 步 / CFG 1。

组件文件已到位并接入实验 CLI。Tokenizer token ID、图像预处理、视觉编码和 packing/scheduler 位模式已做局部检查；文本 hidden state、VAE encoder/decoder 中间值、采样及最终图片仍需与官方 Oracle 逐位核对。当前完整链路的首个未对齐检查点是 VAE reference latent，故入口仍为实验状态。

## 实验图片编辑入口

入口要求方形输入，并缩放为 `--side` 指定的方形画布，尺寸须为 16 的倍数。Edit 与 Turbo 共用 `--components` 目录中的 Qwen2.5-VL-7B、Tokenizer 和 Flux VAE；`--model` 使用各自的 GGUF。默认 Edit 为 50 步 / CFG 4.5，Turbo 为 8 步 / CFG 1。

```bash
RMI_SCALAR=1 cargo run --profile release-fast --features parity-trace --bin longcat-image-edit -- \
  --kind turbo \
  --model /Users/gouzi/Documents/git/rust-model-inference/models/LongCat-Image-Edit-Turbo-GGUF/LongCat-Image-Edit-Turbo-Q8_0.gguf \
  --components /Users/gouzi/Documents/git/rust-model-inference/models/LongCat-Image-Edit \
  --input input.png --output edited.png \
  --instruction 'Change the blue area to green.' --side 1024 --seed 42 --threads 8
```

这个入口当前标记为 `Experimental`。`--side` 缩放与官方按原图宽高比生成画布的规则不同；完整链路的数值和生成质量尚未核验。视觉编码和 Transformer 的对齐只针对标量路径，不把 SIMD、FMA、BLAS、Accelerate 等影响浮点顺序的加速实现纳入结论。
