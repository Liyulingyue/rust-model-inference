# Mage-Flow 原生 CPU 适配与标量对齐

生成和编辑使用主 CLI `cargo run --profile release-fast --bin rust-model-inference -- ...`，按 GGUF 的 `general.architecture=mage_flow` 和 `mage_flow.variant` 自动选择。支持 Microsoft 的 Mage-Flow-Base、Mage-Flow、Mage-Flow-Turbo、Mage-Flow-Edit-Base、Mage-Flow-Edit 和 Mage-Flow-Edit-Turbo。已接入主 CLI 和现有 app/diffusion；服务端尚无图像生成接口。

## 精度契约

DiT、MageVAE 和文本矩阵保留原始 BF16 权重；文本 norm 与 vision 以 F32 无损 widening 保存。激活和文本 KV 为 F32，不做 Q8 重量化。DiT 是 12 层、3072 hidden、24 heads、128 latent channels、2560 context；MageVAE 下采样 16 倍并使用 posterior mean。文本和图像编码器复用发布的 Qwen3-VL 4B，支持最多三张参考图及三层 DeepStack。

Oracle 固定 [microsoft/Mage](https://github.com/microsoft/Mage) 的 `76bec2bb3818863f470de7e867c2dc7f1d0bfd83`，Qwen3-VL 图使用 `transformers==4.57.6`，scheduler 使用 `diffusers==0.38.0`。已验证的本地和远端 Python 依赖版本记录在 `verification.json`。`trace.py` 要求 Oracle checkout 干净且 commit 一致，执行官方 `.float()` 图；Torch 只提供布局和数据搬运，数值算子由独立 C 标量实现替代，未知浮点算子直接报错。C 禁止向量化和 FMA，未调用 BLAS、Accelerate、cuBLAS 或 rocBLAS。

比较检查每条记录的名称、shape、顺序、occurrence 和全部 F32 原始 `u32` 位，不使用容差。BF16 的无损 widening 不代表官方默认 BF16 GPU 计算与本路径逐位相同；不同操作系统的 libm 也可能产生位差，必须在目标机器上运行同机 Oracle。

## 下载与导出

在仓库的 `models/.venv` 使用 ModelScope。每个型号至少下载 `transformer/config.json` 和 `transformer/diffusion_pytorch_model.safetensors`；公共组件的 SHA 在六个源仓库一致，可只下载一份。文件哈希见 [verification.json](verification.json)。

```sh
models/.venv/bin/modelscope download --model microsoft/Mage-Flow-Turbo \
  transformer/config.json transformer/diffusion_pytorch_model.safetensors \
  vae/config.json vae/diffusion_pytorch_model.safetensors \
  text_encoder/config.json text_encoder/tokenizer.json \
  text_encoder/tokenizer_config.json text_encoder/merges.txt text_encoder/vocab.json \
  text_encoder/preprocessor_config.json text_encoder/model.safetensors.index.json \
  text_encoder/model-00001-of-00002.safetensors text_encoder/model-00002-of-00002.safetensors \
  --local_dir models/Mage-Flow-Turbo

.venv/bin/python tools/converter/mage_flow/convert_mage_flow.py \
  models/Mage-Flow-Turbo --variant turbo
.venv/bin/python tools/converter/mage_flow/convert_mage_flow.py \
  models/Mage-Flow-Turbo --vae
```

其余型号的 `--variant` 为 `base`、`flow`、`edit-base`、`edit`、`edit-turbo`。转换检查完整张量名称、shape、BF16 dtype、连续 offset 和文件长度，保留原始 BF16 payload；已存在输出不会覆盖。VAE 跳过官方加载器也不使用的训练用 Flux encoder。

Qwen3-VL 文本/vision GGUF 使用 llama.cpp `11fe02151f79c41d0d4af7da708755d73b9c0da6` 的转换脚本，依赖放在仓库 `.venv`：

```sh
.venv/bin/python /path/to/llama.cpp/convert_hf_to_gguf.py \
  models/Mage-Flow-Turbo/text_encoder --outtype bf16 \
  --outfile models/mage-flow/qwen3vl-text-BF16.gguf
.venv/bin/python /path/to/llama.cpp/convert_hf_to_gguf.py \
  models/Mage-Flow-Turbo/text_encoder --mmproj --outtype f32 \
  --outfile models/mage-flow/vision-F32.gguf
```

## 运行

```sh
RUSTFLAGS='-C no-vectorize-loops -C no-vectorize-slp' \
  cargo build --profile release-fast --features parity-trace \
    --bin rust-model-inference --example mage_flow_trace

RMI_SCALAR=1 target/release-fast/rust-model-inference \
  --model models/mage-flow/mage-flow-turbo-dit-BF16.gguf \
  --vae models/mage-flow/mage-vae-BF16.gguf \
  --text-encoder models/mage-flow/qwen3vl-text-BF16.gguf \
  --prompt '蓝色的猫。' --height 16 --width 16 \
  --noise target/mage-parity/latent.f32 --steps 4 --cfg 1 \
  --threads 8 --out generated.png

RMI_SCALAR=1 target/release-fast/rust-model-inference \
  --model models/mage-flow/mage-flow-edit-turbo-dit-BF16.gguf \
  --vae models/mage-flow/mage-vae-BF16.gguf \
  --text-encoder models/mage-flow/qwen3vl-text-BF16.gguf \
  --mmproj models/mage-flow/vision-F32.gguf \
  --image input.png --prompt '改成蓝色。' \
  --height 16 --width 16 --noise target/mage-parity/latent.f32 \
  --steps 1 --cfg 1 --threads 8 --out edited.png
```

默认图像为 1024×1024；Turbo 默认 4 步/CFG 1，Base 30 步/CFG 5，Flow/Edit 20 步/CFG 5。这些默认分辨率和完整步数尚无端到端精度记录。上面的小尺寸命令是实际验证规模，不能代表大分辨率的性能或成图质量。`--image` 为第一张参考图，`--reference` 可重复追加，合计最多三张；每张参考的 VAE latent 在采样时保持不变。`--noise` 是 target token-major F32 初始噪声；不提供时使用 Rust 的 seed/Box–Muller，未对齐官方 GaussianShading RNG。

仅用于对齐的原始组件 example（`target/release-fast/examples/mage_flow_trace`）：`dit`、`sample`、`vae-encode`、`vae-decode`、`vision`、`text`。`dit/sample --shapes` 是 `[target, ref1, ...]` 的 latent 高宽，如 `1x2,2x1,1x1,1x1`；DiT 图像/文本分别为 token-major 128/2560 维。VAE 输入输出是 CHW。Vision 输入是已归一化 HWC RGB，宽高为 32 的倍数且每边不超过 512；`--deepstack` 可保存三层特征。Text 的 `--input`、`--deepstack`、`--reference-count` 可直接载入固定参考特征。输出原子发布且禁止覆盖。

## 复现标量比较

生成固定输入：

```sh
.venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np
p=Path('target/mage-parity');p.mkdir(parents=True,exist_ok=True)
for name,n,mod,scale in [('latent',128,23,10),('pixels',3*16*16,41,20),
                         ('context',2*2560,29,14),('packed',6*128,23,10),
                         ('negative',2560,13,6),('vision-pixels',32*32*3,31,15)]:
    ((np.arange(n,dtype=np.float32)%mod)/np.float32(scale)-np.float32(1)).tofile(p/(name+'.f32'))
PY

RMI_SCALAR=1 RMI_PARITY_TRACE=target/mage-parity/check-rust.jsonl \
  target/release-fast/examples/mage_flow_trace dit \
  --model models/mage-flow/mage-flow-turbo-dit-BF16.gguf \
  --input target/mage-parity/packed.f32 --context target/mage-parity/context.f32 \
  --shapes 1x2,2x1,1x1,1x1 --sigma 0.375 --threads 8 \
  --output target/mage-parity/check-rust.f32

.venv/bin/python tools/oracle/mage_flow/trace.py dit \
  --oracle /path/to/clean/Mage --model models/Mage-Flow-Turbo \
  --input target/mage-parity/packed.f32 --context target/mage-parity/context.f32 \
  --shapes 1x2,2x1,1x1,1x1 --sigma 0.375 \
  --output target/mage-parity/check-oracle
.venv/bin/python tools/oracle/mage_flow/trace.py compare \
  target/mage-parity/check-oracle/trace.jsonl target/mage-parity/check-rust.jsonl
```

每次使用新的 trace 名和 Oracle 输出目录。其他组件使用同名 `trace.py` 子命令：VAE 指定 `--height/--width`；vision 使用已归一化输入；text 使用 `--prompt`，参考模式另传 `--references`、`--reference-embeddings` 和 `--reference-deepstack`。采样用 `sample --steps N --cfg N`，CFG > 1 还需 `--negative-context`。

主 CLI 接入的工程检查与历史逐位对齐分开记录在 [verification.json](verification.json) 的 `cli_integration`。用户确认远端已关机，本次尚未上传新源码或复验主 CLI 的真实权重输出；历史记录不作为这次接入的新数值证据。

历史已验证结果以 [verification.json](verification.json) 为准，包括六个真实 DiT、共享 VAE/vision/text、固定参考特征、四步采样与最小生成。实际参考图片到 VL fast processor 的 resize/归一化尚未逐位验证；编辑 PNG 当前是入口 smoke。GPU、默认大图、SIMD/FMA 和其他量化不在精度结论内。
