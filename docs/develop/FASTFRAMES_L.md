# FastFrames-L（Experimental）

`fastframes_l` 显式架构接入现有 CLI 和 `POST /v1/videos/generations`，复用原生
Rust 算子与 `LinearExecutor`，不新增可执行文件或加速库。当前是**部分验收 / Experimental**：
16×16 / 5 帧 BF16 CPU 的 411 个检查点全部对齐，实际 HTTP 已返回 MP4；其余精度、
完整 GPU/HTTP 对齐及 480p 质量仍待验证。
见 [验证记录](../../tools/oracle/fastframes_l/validation.md) 和
[逐项矩阵](../../tools/oracle/fastframes_l/matrix.json)。

## 契约

- FastFrames-L：ModelScope `apple/FastFrames-L` @ `651fb104ea405268a4c7400b01c1859718090c5d`。
- umT5：`Wan-AI/Wan2.1-T2V-14B` @ `349065d9c04b2608b85121f4b76f5fcbd03f7dab`。
- 只读 Oracle：`apple-aiml-research/ml-fastframes` @ `6044a56f44bffec500d04f2c21be3c5066891431`。
- 单个 GGUF 包含 umT5、40 层 DiT、KV refiner、pixel decoder 和 tokenizer。
  源文件哈希和 tensor schema 固定在 `src/models/diffusion/fastframes_l/contract.json`。
- 九档导出：F32/F16/BF16/Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K。norm、bias、
  relative position bias、depthwise kernel 和不满足块对齐的矩阵保留 F32。
- 激活/KV 存储及累加为 F32；F16 投影输入按共享 Prepared 语义先舍入到 F16，
  块量化投影使用共享 Q8 prepared activation 路径。BF16/F16 是权重存储类型，
  不代表原生低精度激活加速。CPU 使用共享 SIMD/线程池。
- `cpu`、`auto`、严格 `vulkan` 共用计算图。GPU 复用共享投影和 attention
  分发；文本 relative-bias attention、norm、RoPE、调制、布局和 decoder depthwise
  convolution 使用 CPU。共享 Vulkan Conv3D 目前只接受 groups=1，不能声称此处卷积已上 GPU。
  严格投影失败返回错误。用 `RMI_COMPUTE_TRACE=1` 取证，设备初始化不是执行证据。
- 首 latent frame 4 步，其余 2 步；首帧 clean pass 重写 KV，后续由 refiner 修复 KV。
- 宽高是 16 的倍数，最大 832×480；1–81 帧、1–60 fps。默认 832×480/81 帧/16 fps。
  小尺寸仅用于诊断。JEV、文本 token streaming 不适用于该视频生成契约。

## 下载与转换

仅在 Linux 主机执行 [下载和转换](../../tools/converter/fastframes_l/README.md)。
源权重约 42 GB，各档 GGUF 还需额外空间。此次远端 `/workspace` 为 98 GB，
GGUF 输出选用 `/root/rust-model-inference-fastframes/models/`；Mac 仅传代码和小型证据。

```bash
.venv/bin/python -m tools.converter.fastframes_l.convert_fastframes_l \
  --model-dir models/FastFrames-L --text-dir models/Wan2.1-T2V-14B \
  --output /root/rust-model-inference-fastframes/models/FastFrames-L-bf16.gguf \
  --format bf16
cargo build --profile release-fast --bins --features vulkan,parity-trace
```

## CLI 与 HTTP

以下命令可复现小尺寸诊断，要求 `ffmpeg` 在 PATH。BF16 CPU 已成功生成；
完整九档入口/后端验收见矩阵：

```bash
target/release-fast/rust-model-inference \
  --model /root/rust-model-inference-fastframes/models/FastFrames-L-bf16.gguf \
  --prompt 'A red ball rolls on a table.' \
  --width 16 --height 16 --video-frames 5 --fps 16 \
  --seed 17 --threads 24 --compute cpu --out target/fastframes/cpu.mp4
target/release-fast/rust-model-server \
  --model /root/rust-model-inference-fastframes/models/FastFrames-L-bf16.gguf \
  --threads 24 --compute cpu --host 127.0.0.1 --port 8080
curl --fail-with-body http://127.0.0.1:8080/v1/videos/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A red ball rolls on a table.","width":16,"height":16,"frames":5,"fps":16,"seed":17}' \
  --output target/fastframes/http.mp4
```

`--compute auto|vulkan` 切换策略。CLI `--noise` 接受 Oracle 的 little-endian F32，
为 initial noise 和三组 transition noise，各组 `[latent_frames,16,height/8,width/8]`。
共享 noise 避免要求跨设备 RNG 逐位相同。默认不覆盖输出，显式 `--overwrite` 才替换。
`--steps`、KV 类型覆盖及其他模型的控制项会拒绝。

HTTP 成功返回 `video/mp4`、`Cache-Control: no-store`。可选 JSON 字段为
`model,width,height,frames,fps,seed`，未知字段拒绝。忙时返回 429；客户端取消后，
计算结束前仍占用任务槽。server 启动参数不能包含 CLI prompt/output/noise。

## 内存与限制

最大尺寸/帧数的 F32 KV 约 54 GB，另需权重、特征及缓冲。Vulkan 每八层释放上传缓存，
避免整个 F32 DiT 常驻 48 GiB 显存；最终峰值尚未实测。F32 矩阵使用仓库已有的
mmap 借用接口，避免复制约 84 GB 权重。实际容器内存上限约 55 GiB，不能以主机
503 GiB 物理内存推断进程可用内存。

2026-11-18 在替换 SSH 端口后完成 BF16/F32/F16 转换、BF16 CPU Oracle 和实际 HTTP；
随后远端再次拒绝连接，剩余任务最终状态和原始证据未能取回。新容器上已重下全部源权重，
最终 Linux 模型测试 13 项、真实 tokenizer 测试及 11 项 Python 检查均通过，包含 F32
mmap 修复的九个 Vulkan/parity 二进制构建成功；但 BF16 转换期间连接又被关闭，尚未观察到
新的完整模型结果，不能据此判断为 OOM。GitHub CI 三项作业已通过。BF16 CPU 最大相对 RMSE
为 `4.240734428259513e-5`，最低 cosine 为 `0.9999999991360493`，优于预设
`0.005 / 0.9999` 门槛。Vulkan 仅有部分设备执行与前缀对齐证据。最终 F32 加载路径的完整推理、
其余格式/入口数值、480p 多帧质量和无插桩性能仍需远端复测，保持 Draft/Experimental。
