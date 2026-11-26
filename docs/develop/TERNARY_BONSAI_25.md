# Ternary Bonsai 2.5 27B

模型使用现有 `qwen35` 文本和图像入口。将 `MODEL` 设为官方解码器 GGUF 的路径，`MMPROJ` 设为同一发布仓的 BF16 或 Q8_0 投影器。以下命令在存放权重的推理主机执行。[权重和校验值](../../tools/oracle/ternary_bonsai/README.md)

```sh
cargo build --locked --profile release-fast --features vulkan \
  --bin rust-model-inference --bin rust-model-server
bin=target/release-fast
"$bin/rust-model-inference" --model "$MODEL" --compute vulkan --jinja --no-thinking \
  --prompt 'Name the capital of France.' --temperature 0 --max-tokens 16
"$bin/rust-model-inference" --model "$MODEL" --mmproj "$MMPROJ" --image red.png \
  --compute vulkan --jinja --no-thinking --temperature 0 --max-tokens 8 \
  --prompt 'What color is the image? Answer with one word.'
"$bin/rust-model-server" --model "$MODEL" --mmproj "$MMPROJ" --compute vulkan \
  --jinja --host 127.0.0.1 --port 8080 --max-context 512 --prefill-batch-size 16
```

`--compute cpu` 使用多线程 CPU，`auto` 自动选择设备。文本贪心生成可对 `*-mtp.gguf` 启用 `--mtp-draft 2`。图像输入走目标模型，CLI 图像命令省略草稿参数。HTTP 支持 `/v1/chat/completions`，图像使用 `image_url` 内容块，流式请求加 `"stream":true`。JEV 的文本和图像路由分别为 `/v1/jev/score`、`/v1/jev/image`。分组评分使用 `/v1/jev/grouped` 或 `/v1/jev/image_grouped`，CLI 在相同的 `--jev-question` 后为每组传入 `--jev-block` 和对应的 `--jev-option`。

```sh
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name the capital of France."}],"temperature":0,"enable_thinking":false,"max_tokens":16}'
```

## 精度和性能

三种文件均直接加载。CPU 首次使用投影矩阵时，将三元码和原始 FP16 缩放无损重排为 Q4_0 并缓存，401 个 CPU 投影合计 13.41 GiB。词嵌入按 token 解包一行。Vulkan 逐矩阵重排并上传，最大的临时矩阵为 682.03 MiB 的输出头，上传完成即释放。设备侧保留投影权重、KV 和工作区。归一化 Hadamard、激活量化及矩阵乘法按误差阈值验收。

下表为提交 `9cb45549` 在同一台 EPYC 9334 / RADV NAVI31、8 个推理线程下的一次实测。每格先列 5 个输入 token 的首次 prefill 毫秒数，再列随后 3 次单 token decode 的平均吞吐。CPU 首次 prefill 含投影矩阵重排。Vulkan 的重排和上传发生在会话初始化期间，表中另列这段耗时。该主机同时运行其他推理任务，单次短提示的耗时会随设备负载变化。[全部逐步计时](../../tools/oracle/ternary_bonsai/performance.json)

| 权重 | Rust CPU，ms / tok/s | Rust 严格 Vulkan，ms / tok/s | Vulkan 初始化，s | 发布方 llama.cpp CPU，ms / tok/s |
|---|---:|---:|---:|---:|
| PTQ1_0 | 12,777 / 3.10 | 2,501 / 0.34 | 26.43 | 1,850 / 1.71 |
| PQ2_0 | 11,826 / 3.01 | 859 / 1.80 | 39.60 | 1,080 / 2.00 |
| Q2_0 | 11,241 / 2.76 | 3,781 / 0.38 | 31.61 | 1,469 / 1.95 |

提交 `a43c2d4a` 中，折叠投影用八路 FP32 累加，支持 packed int8 dot 的设备用一条指令计算每四对整数的点积。在 RADV NAVI31 上使用 PTQ1_0、一个输入 token 和八次续写与 `79af30ed` 对照。九个位置共执行 2,745 次 Q4_0 投影，GPU 时间从 2,589.40 ms 降至 821.11 ms。全部算子的 GPU 时间从 2,911.78 ms 降至 1,149.47 ms。

同一轮验证中另做三轮交替的无插桩测试。排除模型加载和首 token 后，实际解码吞吐中位数从 2.36 提升至 4.43 tok/s。每个位置有 62 次设备提交，9 个 greedy token 均与发布方一致。墙钟计时包含主机调度、同步和同机其他任务的等待。[对齐与性能记录](../../tools/oracle/ternary_bonsai/validation.md)

在固定的 8,192 token 检索提示上，三种权重分别覆盖 CPU、Auto 和严格 Vulkan。每组比较最后一个 prefill 位置和 15 个强制续写位置。144 个贪心 token 均与发布方一致，最低余弦为 0.996380，最大相对 L2 为 0.087993。三个解码器在首个结束标记前均输出 `The capital of France is Paris.`。[按权重和后端列出的结果](../../tools/oracle/ternary_bonsai/validation.md#8192-token-retrieval)

长提示可用 `--max-context 8256 --prefill-batch-size 128 --kv-cache f32` 为 8,192 个输入 token 和后续生成预留空间。27B 的 64 层中有 16 个稠密注意力层，Vulkan 为这 16 层分配持久 KV，带 MTP 的文件为草稿头再分配一份。序列超过 4,096 时，设备使用分块 online softmax。该路径要求 256 线程工作组和至少 27 KiB 工作组共享内存。
