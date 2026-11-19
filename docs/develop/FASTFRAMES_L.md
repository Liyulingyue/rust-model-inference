# FastFrames-L（Experimental）

`fastframes_l` 显式架构接入现有 CLI 和 `POST /v1/videos/generations`，复用原生
Rust 算子与 `LinearExecutor`，不新增可执行文件或加速库。当前是**部分验收 / Experimental**：
16×16 / 5 帧 BF16 的 CPU/Auto/严格 Vulkan 在 CLI/HTTP 各 411 个检查点和视频像素通过；
F32 CPU/Auto/Vulkan 两入口也通过。F16 各后端两入口的检查点通过，但压缩后视频比较失败；Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K 数值对齐失败，54 个格式/后端/入口组合已跑完，12 个通过、42 个失败；480p 检查点通过，但压缩视频和 BF16 配方漂移超阈值。
见 [验证记录](../../tools/oracle/fastframes_l/validation.md) 和
[逐项矩阵](../../tools/oracle/fastframes_l/matrix.json)。

上述 54 项为首次完整矩阵。新机上修复了参考脚本的量化除法舍入，12 项 Python 回归通过；
Q8 文本仍超出原容差。相同原生输入的 28 个独立投影复算相对误差均低于 `5e-7`，
首层归一化的微小误差已导致一个激活量化码变化。
[新机复测记录](../../tools/oracle/fastframes_l/replacement-host-results.json) 单独保存修复后的
BF16/F16/Q8_0 CLI/HTTP 验收，不覆盖历史失败；生产计算和阈值未调整。

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
  共享 attention 对健康 GPU 结果也会完整计算 CPU 参考以执行既有 TODO-044 校验，
  大尺寸保留显著 CPU 开销。严格投影失败返回错误。用 `RMI_COMPUTE_TRACE=1` 取证，设备初始化不是执行证据。
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
避免整个 F32 DiT 常驻 48 GiB 显存；小样本设备分配器峰值已记录，完整尺寸进程峰值尚未验证。F32 矩阵使用仓库已有的
mmap 借用接口，避免复制约 84 GB 权重。实际容器内存上限约 55 GiB，不能以主机
503 GiB 物理内存推断进程可用内存。

2026-11-19 在新的授权主机上重下并校验全部源文件，完成最终代码的 BF16 CPU CLI/HTTP
对齐：双方各 411 个检查点通过，最大相对 RMSE 为 `2.794884733482099e-5`，最低 cosine
为 `0.9999999997176138`，优于预设 `0.005 / 0.9999` 门槛。视频经相同编码再解码后像素
RMSE 为 0。F32 的 mmap 加载路径也已在 CLI/HTTP 通过同样的 411 个检查点和视频像素检查。
此前 Linux 模型测试 13 项、真实 tokenizer 及 11 项 Python 检查均通过，九个 Vulkan/parity
二进制构建成功，GitHub CI 三项作业通过。严格 Vulkan 两入口最大相对 RMSE 为
`2.5552589342221195e-5`，Auto 为 `2.5236295358634716e-5`，视频像素 RMSE 均为 0。
无设备时 Auto 的 CPU 回退也通过，严格 Vulkan 明确报错。

32×32 / 9 帧 CPU 用例通过 578 个检查点，覆盖非零空间位置和后续帧使用修复的 KV；
视频 RMSE 为 `0.00759218`，cosine 为 `0.99992699`，均通过原门槛；相对官方 BF16 配方的
像素 RMSE 为 `0.01125140`，也通过独立的 `0.08` 门槛。
F16 CPU 两入口均通过 411 个检查点，但压缩后视频 RMSE 为 `0.01441015`、cosine 为
`0.99954839`，未通过 `0.01 / 0.9999` 门槛。编码前 RGB 最大只差 `1/255`、RMSE
为 `0.00074342`；独立重编码验证了 H.264 对微小差异的放大，原阈值和失败记录保留。
fat-LTO release 编译已完成。关闭插桩后，16×16 / 5 帧 CPU 的三次整进程耗时为
63.40 / 73.74 / 78.48 秒，中位数 73.74 秒；已完成的 CPU 和初次 Auto 样本的视频像素
也通过对齐。Auto / 严格 Vulkan 的中位整进程耗时为 106.49 / 553.27 秒；官方受控参考 /
发布 BF16 配方为 269.16 / 123.76 秒。五组各一次预热、三次测量均完成，12 段原生视频的
解码像素均通过。线程数、加载边界和共享负载有差异，不能据此宣称独占性能或加速比。
F32 Auto/严格 Vulkan 的最大相对 RMSE 分别为 `4.775628860976526e-5` /
`4.0885956235754525e-5`，各入口均通过 411 个检查点，解码视频像素 RMSE 为 0。
832×480 / 5 帧严格 Vulkan 的 411 个检查点也通过，最大相对 RMSE 为 `0.00375643`。
编码前 RGB 的 RMSE 为 `0.00038718`，独立重编码可精确重现原生解码像素；压缩后视频的
cosine 为 `0.99987585`，未过 `0.9999`。官方受控 / 发布 BF16 配方的像素 RMSE 为
`0.10040363`，也未过 `0.08`。这些失败均保留；原生带插桩 release-fast 进程耗时
10,845.66 秒，期间与其它验收任务共享主机，不能当作独占 release 延迟。
81 帧默认规格、完整进程峰值和独占性能未验证，保持 Draft/Experimental。
