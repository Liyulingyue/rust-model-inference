# Longer prompts, spatial images and short clips

The AMD EPYC 9334 / RADV NAVI31 run on 2026-11-26 used the four published files at their recorded SHA-256 values. Text uses the fork's default KV and media uses F32 reference KV. Rust uses F32 KV. All comparisons require the same greedy token and finite values across 248,320 logits per position, with KL(reference ‖ Rust) ≤ 0.001. Text also requires identical prompt IDs. [Weight hashes and reference revision](README.md)

## Inputs and reproduction

Run these Bash commands in the Rust repository on the weight host. Start with the pinned Nari fork and its CPU build. The clips require FFmpeg and ffprobe on PATH. [Reference build instructions](README.md)

```bash
out="$PWD/runs/nariq-extended"
mkdir -p "$out"
cmake -S oracle -B oracle/build -DLLAMA_SUBPROCESS=ON -DMTMD_VIDEO=ON
cmake --build oracle/build -j 8 --target mtmd
python3 tools/oracle/nariq/fixtures.py "$out" --video
export LD_LIBRARY_PATH="$PWD/oracle/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8
prefix=models/Qwen3.8-27B-NariQ-GGUF/Qwen3.8-27B-NariQ-
export NARIQ_MODEL="${prefix}3.0bpw.gguf" NARIQ_MMPROJ="${prefix}mmproj-BF16.gguf"
```

The upstream text probe bounds prompts at 64 tokens and allocates a 128-position context. Increase those two limits in a scratch copy. Tokenization, model evaluation and the saved rows use the original reference implementation.

```bash
sed -e 's/nt <= 64, "prompt must contain 1-64 tokens"/nt <= 4096, "extended probe accepts at most 4096 prompt tokens"/' \
    -e 's/cp.n_ctx = 128;/cp.n_ctx = std::max(128, nt + predict);/' \
    oracle/tests/test-trellis-model.cpp > "$out/context.cpp"
c++ -std=c++17 -O2 -Ioracle/include -Ioracle/ggml/include -Ioracle/vendor \
    "$out/context.cpp" -Loracle/build/bin -lllama -lggml -lggml-base -o "$out/text-oracle"
for sample in continuation ledger; do
    steps=32
    if [[ $sample == ledger ]]; then steps=8; fi
    # read -d '' keeps the final newline in the counting prompt.
    IFS= read -r -d '' question < "$out/$sample.txt" || true
    export NARIQ_ORACLE="$out/$sample-${NARIQ_MODEL##*/}"
    "$out/text-oracle" --model "$NARIQ_MODEL" --output "$NARIQ_ORACLE" \
        --prompt "$question" --predict "$steps" --threads 8 --ubatch 8 --gpu-layers 0
    NARIQ_COMPUTE=vulkan cargo test --locked --profile release-fast --features vulkan \
        --test nariq_reference nariq_prefill_and_decode_match_oracle -- --ignored --nocapture
done
```

The 1,434-token capture occupies about 1.43 GB on the weight host. The comparator seeks to the final prefill row and streams one vocabulary row at a time. It feeds each reference token back and checks all subsequent predictions against the independent greedy sequence.

The following patch lets the existing mtmd probe accept a PNG or MP4 and its question. It writes the input chunks, positions, top token and configured context to JSON. Use a different output prefix for each file and quantization.

```bash
patch -s -o "$out/media.cpp" tools/oracle/holo4/27b/image.cpp < tools/oracle/nariq/media.patch
c++ -std=c++17 -O2 -Ioracle/include -Ioracle/ggml/include -Ioracle/tools/mtmd \
    "$out/media.cpp" -Loracle/build/bin -lmtmd -lllama -lggml -lggml-base -o "$out/media-oracle"
for sample in left top red-blue blue-red; do
    unset NARIQ_IMAGE_INPUT NARIQ_VIDEO_INPUT
    case "$sample" in
        left|top)
            input="$out/$sample.png"
            export NARIQ_IMAGE_INPUT="$input"
            export NARIQ_MEDIA_PROMPT="Which color is on the $sample, red or blue? Answer with one color." ;;
        *)
            input="$out/$sample.mp4"
            export NARIQ_VIDEO_INPUT="$input"
            export NARIQ_MEDIA_PROMPT='What color fills the final frame of this clip? Answer with one color word.' ;;
    esac
    export NARIQ_IMAGE_ORACLE="$out/$sample-${NARIQ_MODEL##*/}"
    "$out/media-oracle" "$NARIQ_MODEL" "$NARIQ_MMPROJ" "$NARIQ_IMAGE_ORACLE.f32" \
        "$input" "$NARIQ_MEDIA_PROMPT" > "$NARIQ_IMAGE_ORACLE.json"
    NARIQ_COMPUTE=vulkan cargo test --locked --profile release-fast --features vulkan \
        --test nariq_reference nariq_image_logits_match_oracle -- --ignored --nocapture
done
```

Set `NARIQ_MODEL` to either other tier and select `cpu`, `auto` or `vulkan` in the comparison commands to reproduce the corresponding measurements below.

To verify media prompt IDs, compare the public CLI's `[RUST_TOKENS]` line on stderr with the reference JSON's `chunks` in order. Concatenate each text chunk's `text` array and expand each media chunk into `count` copies of the published placeholder ID, `248056` for images or `248057` for video.

## Full-vocabulary results

The counting prompt contains 24 tokens. The ledger contains 128 records, with `orchid` at index 017. It tokenizes to 1,434 tokens. The landscape image is 192×96 with red on the left. The portrait is 96×192 with blue on top. Each image projects to 18 tokens. Each 96×96 clip contains four frames at 2 fps. The reference wraps each pair of frames in its own span of nine visual tokens, preceded by `<0.2 seconds>` or `<1.2 seconds>`.

| bpw | Input | Positions | CPU max KL | Auto max KL | Strict Vulkan max KL |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2.0 | Counting | 32 | 6.6e-08 | — | 6.6e-08 |
| 2.0 | 128-record ledger | 8 | — | — | 4.47e-07 |
| 2.0 | Left | 1 | 4.415e-06 | — | 4.387e-06 |
| 2.0 | Top | 1 | 1.8643e-05 | — | 1.8689e-05 |
| 2.0 | Red → blue | 1 | 9.99e-07 | 1.012e-06 | 1.012e-06 |
| 2.0 | Blue → red | 1 | 2.961e-06 | 2.955e-06 | 2.955e-06 |
| 2.5 | Counting | 32 | 7.8e-08 | 7.7e-08 | 7.7e-08 |
| 2.5 | 128-record ledger | 8 | — | — | 1.08e-07 |
| 2.5 | Left | 1 | 2.1037e-05 | — | 2.0638e-05 |
| 2.5 | Top | 1 | 5.165e-05 | — | 5.1686e-05 |
| 2.5 | Red → blue | 1 | 3.492e-06 | 3.536e-06 | 3.536e-06 |
| 2.5 | Blue → red | 1 | 1.732e-06 | 1.731e-06 | 1.731e-06 |
| 3.0 | Counting | 32 | 7.9e-08 | — | 8.3e-08 |
| 3.0 | 128-record ledger | 8 | 7.93e-07 | — | 8.03e-07 |
| 3.0 | Left | 1 | 6.46e-06 | — | 6.774e-06 |
| 3.0 | Top | 1 | 0.000186183 | 0.000185076 | 0.000185076 |
| 3.0 | Red → blue | 1 | 1.5887e-05 | 1.5926e-05 | 1.5926e-05 |
| 3.0 | Blue → red | 1 | 1.554e-05 | 1.548e-05 | 1.548e-05 |

All 42 recorded comparisons passed across 287 prediction positions. Each number is the largest KL across the prediction positions for one input and backend. A dash indicates an unmeasured combination. Across all rows the largest relative L2 was 0.00483355 and the largest absolute logit difference was 0.097906. On the device, counting recorded 2,618 submissions, the ledger 14,399, the spatial images 1,855 and the clips 2,009. CPU comparisons recorded zero. The first eight ledger predictions contain `orchid` at 2.0 and 2.5 bpw. The 3.0 bpw reference starts a longer answer and has yet to emit the label at position eight.

At 3.0 bpw, the 1,434-token prefill took 12,320.3 s on Rust CPU, 649.6 s on strict Vulkan and 307.6 s on the reference CPU. Both Rust backends reproduced the eight reference token IDs `[271, 27775, 383, 279, 45366, 3766, 11, 279]`. The reference also saved every prompt logit row. These are single wall-time observations on a shared host.

## Public CLI

```bash
cli=target/release-fast/rust-model-inference
cargo build --locked --profile release-fast --features vulkan --bin rust-model-inference
"$cli" --model "$NARIQ_MODEL" --mmproj "$NARIQ_MMPROJ" --compute vulkan \
    --threads 8 --prefill-batch-size 8 --temp 0 --no-thinking --max-context 256 \
    --max-tokens 12 --video "$out/red-blue.mp4" \
    --prompt 'What color fills the final frame of this clip? Answer with one color word.'
"$cli" --model "$NARIQ_MODEL" --compute vulkan --threads 8 --prefill-batch-size 8 \
    --temp 0 --no-thinking --max-context 4096 --max-tokens 32 --prompt "$(cat "$out/ledger-360.txt")"
```

With 3.0 bpw, strict Vulkan answered the two spatial questions with `red` and `blue`. The 360-record ledger produced 4,000 chat prompt tokens and returned `orchid`. The measured prefill took 1,851.6 s (2.2 tokens/s). The 768-record version with `--max-context 12288` and 32 requested output tokens produced 8,488 prompt tokens. Strict Vulkan rejected capacity 8,520 at session creation. The shared Qwen3.5 attention shader supports session capacities up to 4,096 including the generation budget. The larger prompt has no measured answer. These synthetic prompts do not establish quality at the model's advertised 262,144-token context.

Before the repair, the video CLI reported 28 text tokens and zero visual placeholders. Both CPU and strict Vulkan answered `Black` for both clips. The repaired CLI's prompt IDs matched the reference chunks in order, with 48 positions for each spatial image and 62 for each clip. Each clip included 18 visual placeholders. At 3.0 bpw, CPU and strict Vulkan returned `blue` for red → blue and `red` for blue → red. The shipped Jinja template gave the same two answers on strict Vulkan with `--jinja --no-thinking`. The CLI samples at 2 fps with a 32-frame cap. These clips use four frames. HTTP chat returned 400 for `video_url`. The CLI rejected video JEV during argument validation.

The text and spatial grid began on the original NariQ runtime. The video comparisons use the corrected input preparation. Commit `69a54229e` reproduced both clip logits, both Jinja answers and the portrait logits on strict Vulkan, plus the red → blue clip logits on CPU. Commit `8080544b0` changed diagnostic output routing. Its JEV CLI parsed as one JSON document for every tested input, preserving both candidate probabilities from HTTP. [Entry-point regression](validation.md)
