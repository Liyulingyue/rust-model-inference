# FastFrames-S（Experimental）

`fastframes_s` 是 `fastframes_l` 架构内的 S 变体：1.3B 文生视频，DiT 为冻结的
Causal-Forcing++ 权重（`Wan2.1-T2V-1.3B` 结构），显式接入现有 CLI 和
`POST /v1/videos/generations`，复用原生 Rust 算子与 `LinearExecutor`，不新增可
执行文件或加速库。GGUF 的 `general.architecture` 区分 `fastframes_l` / `fastframes_s`
两 Variant，同一实现按 Variant 分发规格与噪声布局；CLI/HTTP 入口零结构改动。

当前是**部分验收 / Experimental**：16×16 / 5 帧 BF16 的 CPU 与严格 Vulkan CLI 各
321 个检查点全部通过（最差相对 RMSE `7.05e-6` / `6.67e-6`，最低 cosine
`0.999999999975`），CPU↔Vulkan 解码像素零差异；F32 CPU 同过。CLI↔HTTP MP4 字节
级一致；错误路径 400/422/429 已验证。F16 检查点通过但压缩视频门禁失败；六档块量
化数值门禁失败（逐项见 [验证记录](../../tools/oracle/fastframes_s/validation.md)）。

见 [验证记录](../../tools/oracle/fastframes_s/validation.md)。

## 与 FastFrames-L 的配方差异

| 项 | FF-S | FF-L |
|---|---|---|
| DiT | 冻结 CF++ 1.3B（torch pickle，`generator_ema` 下 F32 存储） | 专用 14B safetensors（BF16） |
| 文本侧 | 复用 `Wan2.1-T2V-14B` 资产（SHA256 与 1.3B 逐一相同，零下载） | `Wan2.1-T2V-14B` |
| 时间表 | 首帧 `1000/937.5/833.3333129882812/625`，后续 `1000/833.3333129882812`，refine `833.3333129882812`（分数步） | 整数步 `1000/937/833/625` 与 `1000/833` |
| 噪声 | `sequential`：initial 全帧 + 每次转移抽单帧块（值数 `count + (2+frames)*frame_count`） | `pregenerated`：3 组全帧转移块（`4*count`） |
| autocast / fast_ops | 无 / 无（published 路径） | 有 / 有 |
| pixel decoder | F32 无 bridge（DiT 已是 1536 宽），GGUF 键带 `pixel_head.` 前缀 | BF16 带 bridge，键直接命中 |
| refiner | `dim=1536, heads=12, blocks=30` | `dim=5120, heads=40, blocks=40` |

## 契约

- FastFrames-S：ModelScope `apple/FastFrames-S` @ `aa099efcedf1b65447106bf9004cde75821f7387`。
- CF++ DiT：`zhuhz22/Causal-Forcing` @ `2f8eb8bb6eeb1238da9d13e5420d342a74d634a6`
  （ModelScope 无此 repo，经 hf-mirror `resolve/` 下载，SHA256 对官方 pin）。
- umT5：`Wan-AI/Wan2.1-T2V-1.3B` @ `37ec512624d61f7aa208f7ea8140a131f93afc9a`；
  资产与本地 14B 快照逐字节相同，直接复用。
- 只读 Oracle：`apple-aiml-research/ml-fastframes` @ `6044a56f44bffec500d04f2c21be3c5066891431`。
- 单个 GGUF 包含 umT5、30 层 DiT、KV refiner、pixel decoder 和 tokenizer。
  源文件哈希和 tensor schema 固定在 `src/models/diffusion/fastframes_l/contract_s.json`
  （L 的 `contract.json` 零改动）。两变体共享全部实现与算子路径——同仓库 `flux3/`
  目录承载 `flux3`/`flux3_vae` 双 arch 的先例——S 的契约随实现同目录存放，
  `include_str!` 编译期内嵌。契约身份不依赖目录名：GGUF 以
  `general.architecture: fastframes_s` 与 `fastframes_s.contract_sha256` 自描述，
  加载时按 Variant 分发校验，不存在读错契约的可能。
- 九档导出：F32/F16/BF16/Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K。norm、bias、
  relative position bias、depthwise kernel 和不满足块对齐的矩阵保留 F32。
- 激活/KV 存储及累加为 F32；F16 投影输入按共享 Prepared 语义先舍入到 F16，
  块量化投影使用共享 Q8 prepared activation 路径。
- `cpu`、`auto`、严格 `vulkan` 共用计算图；GPU 复用共享投影和 attention 分发，
  文本 relative-bias attention、norm、RoPE、调制、布局和 decoder depthwise
  convolution 使用 CPU。严格 Vulkan 投影失败返回错误。用 `RMI_COMPUTE_TRACE=1`
  取证，设备初始化不是执行证据。
- 首 latent frame 4 步，其余 2 步；首帧 clean pass 重写 KV，后续由 refiner 修复 KV。
- 宽高是 16 的倍数，最大 832×480；1–81 帧、1–60 fps。小尺寸仅用于诊断。
  JEV、文本 token streaming 不适用于该视频生成契约。

## 下载与转换

仅在 Linux 主机执行 [下载和转换](../../tools/converter/fastframes_s/README.md)。
CF++ `.pt` 约 5.7 GB、S 附属权重约 1 GB、文本侧复用本地 14B；各档 GGUF 另需空间
（F32 约 30 GB，BF16/F16 约 15 GB，量化档更小）。

```bash
.venv/bin/python -m tools.converter.fastframes_s.convert_fastframes_s \
  --model-dir target/fastframes/checkpoints --text-dir models/Wan2.1-T2V-14B \
  --output models/FastFrames-S-bf16.gguf --format bf16
cargo build --profile release-fast --bins --features vulkan,parity-trace
```

## CLI 与 HTTP

以下命令可复现小尺寸诊断，要求 `ffmpeg` 在 PATH：

```bash
target/release-fast/rust-model-inference \
  --model models/FastFrames-S-bf16.gguf \
  --prompt 'A red ball rolls on a table.' \
  --width 16 --height 16 --video-frames 5 --fps 16 \
  --seed 17 --threads 16 --compute cpu --out target/fastframes/s-cpu.mp4
target/release-fast/rust-model-server \
  --model models/FastFrames-S-bf16.gguf \
  --threads 16 --compute cpu --host 127.0.0.1 --port 8080
curl --fail-with-body http://127.0.0.1:8080/v1/videos/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A red ball rolls on a table.","width":16,"height":16,"frames":5,"fps":16,"seed":17}' \
  --output target/fastframes/s-http.mp4
```

`--compute auto|vulkan` 切换策略。CLI `--noise` 接受 Oracle 的 little-endian F32，
布局按 Variant 分发：S 为 initial 全帧噪声 + `(2+latent_frames)` 个单帧转移块，
每块 `[16,h/8,w/8]`。共享 noise 避免要求跨设备 RNG 逐位相同；本机实测同 seed 下
RMI 自产噪声与 oracle 逐位一致（无 `--noise` 与 shared-noise 的 BF16 输出字节相同），
但验收不依赖这一点。默认不覆盖输出，显式 `--overwrite` 才替换。
`--steps`、KV 类型覆盖及其他模型的控制项会拒绝。

HTTP 成功返回 `video/mp4`、`Cache-Control: no-store`。可选 JSON 字段为
`model,width,height,frames,fps,seed`，未知字段拒绝。忙时返回 429；客户端取消后，
计算结束前仍占用任务槽。server 启动参数不能包含 CLI prompt/output/noise。

## 内存与限制

1.3B DiT 使 S 的权重与 KV 远小于 L；F32 GGUF 约 30 GB。max 尺寸/帧数的完整
显存与内存峰值未单独验证。Vulkan 沿用每八层释放上传缓存策略。

性能与验证边界见 [验证记录](../../tools/oracle/fastframes_s/validation.md)
与 [性能对标](../../tools/oracle/fastframes_s/performance.json)。
