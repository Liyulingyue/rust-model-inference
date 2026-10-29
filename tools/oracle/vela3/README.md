# Vela3 Oracle

参考为六个 ModelScope 包附带的 Python 实现及其固定 d3 版本。源码 SHA256、权重大小与哈希见 [snapshots.json](snapshots.json)，实际结果见 [verification.json](verification.json)。模型、临时 trace 和大数组不提交到 Git。

Native 只调用已有 Rust 算子。Python 依赖仅运行只读 Oracle，安装在仓库 `.venv`：Torch 2.9.1、Transformers 5.19.0、torchvision 0.24.1、numpy 2.5.3、Pillow 13.0.0、safetensors、accelerate、SciPy。`accelerate` 满足官方 CPU `device_map` 加载要求，SciPy 供官方 box 连通区域解码使用。

fast-vision 原包面向更新的 Transformers；Oracle 注册包内原始 `text/modeling_ultrabert.py`，仅兼容配置声明和类注册，保持计算实现。预处理匹配 torchvision 0.24.1 的 uint8 bicubic 回退，未验证更新版本的 Lanczos 路径。所有模型文件均须先按清单下载到本地；下列 Oracle 不联网换模型。

## 便携检查

以下检查不需要权重或 Python 推理依赖。验证机器使用 Rust 1.98.1、macOS arm64；普通测试跳过需要本地 fixture 的检查。

```bash
cargo +1.98.1 test --profile release-fast --lib models::vela3::
cargo +1.98.1 test --profile release-fast --lib format::safetensors::tests::
.venv/bin/python -m unittest discover -s tools/oracle/vela3 -p test_compare_requests.py
cargo +1.98.1 build --profile release-fast --bin vela3
```

## Fast 全路由

小型官方计算图包含九种适配器、15 个问题，以及独立的长窗口聚合检查。

```bash
.venv/bin/python tools/oracle/vela3/fast_oracle.py \
  --model models/vela3-fast --synthetic models/vela3-fast-tiny \
  --out models/vela3-fast-tiny-oracle.json
RMI_VELA3_FAST_ROOT=models/vela3-fast-tiny \
RMI_VELA3_FAST_FIXTURE=models/vela3-fast-tiny-oracle.json \
  cargo +1.98.1 test --profile release-fast --lib native_fast_matches_official \
  -- --ignored --nocapture

.venv/bin/python tools/oracle/vela3/fast_oracle.py \
  --model models/vela3-fast --synthetic models/vela3-fast-window \
  --span-window 64 --out models/vela3-fast-window-oracle.json
RMI_VELA3_FAST_ROOT=models/vela3-fast-window \
RMI_VELA3_FAST_FIXTURE=models/vela3-fast-window-oracle.json \
  cargo +1.98.1 test --profile release-fast --lib native_fast_matches_official \
  -- --ignored --nocapture

.venv/bin/python tools/oracle/vela3/fast_oracle.py \
  --model models/vela3-fast --request tools/oracle/vela3/requests/fast.json \
  --system-one --require-all-adapters --out models/vela3-fast-oracle.json
RMI_VELA3_FAST_ROOT=models/vela3-fast \
RMI_VELA3_FAST_FIXTURE=models/vela3-fast-oracle.json \
  cargo +1.98.1 test --profile release-fast --lib native_fast_matches_official \
  -- --ignored --nocapture
```

`--require-all-adapters` 拒绝缺失的可选适配器，避免把官方回退到共享底座误记成任务适配器已验证。

## Fast-vision 四种真实任务

每条任务自动生成同一张确定性测试图，逐层记录五个图像切片、SigLIP、pixel shuffle、connector、UltraBERT、选取 hidden、任务 logits，并比较完整公开 System One JSON。

```bash
for task in decision span ocr box; do
  .venv/bin/python tools/oracle/vela3/vision_oracle.py \
    --root models/vela3-fast-vision --task "$task" \
    --out "models/vela3-vision-$task"
  RMI_VELA3_VISION_FIXTURE="models/vela3-vision-$task" \
    cargo +1.98.1 test --profile release-fast --lib \
    models::vela3::vision::tests::official_fixture -- --exact --ignored --nocapture
done
```

省略 `--task` 可生成小型官方图；`--raw` 使用真实 raw 底座与确定性合成任务头。两者不能当成四个真实任务适配器的替代证据。

## d3 请求、任务头和完整计算图

```bash
.venv/bin/python tools/oracle/vela3/head_oracle.py \
  --models models/vela3-lite models/vela3-mini models/vela3-flash models/vela3-turbo \
  --out models/vela3-head-fixture.json
RMI_VELA3_HEAD_FIXTURE=models/vela3-head-fixture.json \
  cargo +1.98.1 test --profile release-fast --lib real_heads_match -- --ignored --nocapture

.venv/bin/python tools/oracle/vela3/request_oracle.py \
  --model models/vela3-lite --base models/d3-lite \
  --request tools/oracle/vela3/requests/lite.json --out models/vela3-request-oracle.json
RMI_VELA3_REQUEST_FIXTURE=models/vela3-request-oracle.json \
  cargo +1.98.1 test --profile release-fast --lib official_request_token_ids -- --ignored --nocapture
cargo +1.98.1 run --profile release-fast --bin vela3 -- \
  --model models/vela3-lite --base-model models/d3-lite \
  --request tools/oracle/vela3/requests/lite.json --threads 4 > models/vela3-request-native.json
.venv/bin/python tools/oracle/vela3/compare_requests.py \
  models/vela3-request-oracle.json models/vela3-request-native.json \
  --atol 0.01 --rtol 0.01 --out models/vela3-request-comparison.json
```

把请求换成 `requests/unicode.json` 可复现中文问题。`compare_requests.py` 精确比较字段、选项、整数位置与 usage，仅容忍浮点误差；错误诊断文字可不同，错误码仍须一致。

逐层 trace 需要 `parity-trace`。下面自动取得实际 lib-test 可执行文件路径，避免依赖 Cargo 生成的文件名。

```bash
cargo +1.98.1 test --profile release-fast --lib --features parity-trace \
  --no-run --message-format=json > models/vela3-test-build.jsonl
vela3_test_bin=$(.venv/bin/python -c 'import json; print(next(r["executable"] for line in open("models/vela3-test-build.jsonl") if (r := json.loads(line)).get("executable") and r["target"]["kind"] == ["lib"]))')

.venv/bin/python tools/oracle/vela3/d3_oracle.py suite \
  --native "$vela3_test_bin" --base models/d3-tiny --vela models/vela3-lite \
  --synthetic --dtype float32 --causal --generation-steps 4 \
  --atol 2e-6 --rtol 1e-6 --out models/vela3-d3-tiny-causal

.venv/bin/python tools/oracle/vela3/d3_oracle.py suite \
  --native "$vela3_test_bin" --base models/d3-lite --vela models/vela3-lite \
  --dtype bfloat16 --causal --generation-steps 4 \
  --out models/vela3-d3-lite-causal

.venv/bin/python tools/oracle/vela3/d3_oracle.py suite \
  --native "$vela3_test_bin" --base models/d3-lite --vela models/vela3-lite \
  --dtype float32 --adapters span decision \
  --out models/vela3-d3-lite-f32
```

Checkpoint Oracle 检查名称、顺序、shape、次数、Token IDs、mRoPE 和全部数值，并报告第一个不满足门槛的位置。`float32` 诊断保持真实 BF16 权重值、只关闭激活舍入；不另存一份整模 F32 权重。BF16 文本使用保存于报告中的逐层误差包络，logits 另外要求相同 argmax。`suite --images image.png --vision-detail` 可重现图像路径差异，该检查目前失败。

## 实际验证范围

- lite：14 个英文问题和 2 个中文问题的选项、非空 span 边界、usage 精确一致，浮点最大绝对误差分别为 `0.004634` / `0.004669`。四步 greedy IDs 与官方均为 `[2838, 682, 264, 220]`。
- d3 小型完整图：F32 最大误差 `4.77e-7`，BF16 最大误差 `0.00390625`；四个 d3 包共 15 个真实 span/NER/relation/image-halu 任务头最大误差 `6.68e-6`。
- lite 真实短序列 F32 激活诊断：所有 checkpoint 满足 `atol=5e-4, rtol=1e-4`。BF16 四步 logits 最大误差 `0.0703125`，各步 argmax 精确一致。
- fast：真实 380M 决策 hidden 最大误差 `4.58e-5`、logits `4.83e-6`，并已覆盖真实 ordinary span/evidence/span-set；全九适配器检查状态以验证 JSON 为准。
- fast-vision：四条真实任务各 84 个 checkpoint 和完整响应通过，task logits 最大误差 `8.35e-6`。Box/point 坐标精确一致，confidence 差约 `5.96e-7`。OCR 使用零阈值确保比较非空 span。真实 raw 底座完整 hidden max `0.247871`、NRMSE `0.001068`，采用明确的缩放门槛；小图仍使用严格容差。
- d3 图像：75 个展开 tokens 和三轴位置精确一致。F32 视觉塔相对官方 F32 投影 max `0.005106`、RMSE `0.000204`；对官方默认 BF16 塔则扩大为最终 hidden max `2.03125`、RMSE `0.10997`、logits max `0.08461`。样本 argmax 一致，但 **BF16 逐层门槛未通过**，未放宽门槛将其记为通过。

mini/turbo 的当前 ModelScope 底座与 Vela 固定版本不一致；完整固定版本验证需要相应文件，不能用当前镜像冒充固定 checkpoint。flash 整模状态以验证 JSON 为准。GPU、GGUF、HTTP server、视频、长上下文性能、跨平台与完整公开基准均未执行。
