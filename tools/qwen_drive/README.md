# Qwen-Drive-1.0-4B GGUF export

This exporter is fixed to the released Qwen-Drive component inventory in
`source-tensors.json` and llama.cpp commit
`b96806d96061049a5b574269b049bf6241d63d46`. It streams head payloads and
writes every output through a same-directory temporary file.

## Source checkpoints

| Component | Bytes | Tensors | SHA256 |
| --- | ---: | ---: | --- |
| VLM | 9,078,630,512 | 723 | `b9de4bf448f57485fdaa45c60b1eea8e41a4b6ae82ec0cee8855a1e0301caccc` |
| planner-sft | 2,079,739,550 | 358 | `dcf5989ed292799e77f539b21e5d8b701566676c1c5d13038135e97f96e2d7b4` |
| planner-rl | 2,079,739,550 | 358 | `74478eb7ec5dea0d8144372ed7b60e6f508da7d9eb1522e567b391cad6dbaf29` |
| perception | 500,368,384 | 827 | `e964ca945f028bbbfadfdf9c1e47d31cfc3fe502d205eca0bb5a3d2c5ab450ae` |

Regenerate and review the manifest:

```bash
python3 tools/qwen_drive/convert_qwen_drive.py inspect \
  /Users/gouzi/Documents/git/rust-model-inference/models/Qwen-Drive-1.0-4B \
  --write-manifest tools/qwen_drive/source-tensors.json
```

## Export and verify

At least 16 GiB must be free. The observed preflight on 2026-09-11 had 25 GiB
available.

```bash
python3 tools/qwen_drive/convert_qwen_drive.py export \
  /Users/gouzi/Documents/git/rust-model-inference/models/Qwen-Drive-1.0-4B \
  --llama-cpp /Users/gouzi/Documents/git/llama.cpp \
  --out-dir /Users/gouzi/Documents/git/rust-model-inference/models/Qwen-Drive-1.0-4B

python3 tools/qwen_drive/convert_qwen_drive.py verify \
  /Users/gouzi/Documents/git/rust-model-inference/models/Qwen-Drive-1.0-4B \
  --out-dir /Users/gouzi/Documents/git/rust-model-inference/models/Qwen-Drive-1.0-4B
```

`verify` checks the source manifest, GGUF architecture/name/type/shape
directories, and every planner/perception payload digest. The llama.cpp
conversion preserves BF16 values while promoting required norms, biases, and
vision patch tensors to exact F32 representations.

| Output | Bytes | Tensors | SHA256 |
| --- | ---: | ---: | --- |
| `Qwen-Drive-1.0-4B-BF16.gguf` | 8,424,390,752 | 426 | `1d5aabe7f02ef97fdf173bad8bcaaf783af4d85bd07ea56cc3dd42d1f28deac9` |
| `Qwen-Drive-1.0-4B-mmproj-BF16.gguf` | 675,569,216 | 298 | `d0ba72870cca4073c0e3aaef251e36c6ff1061c6b76fa725cb55d518a11a3d9f` |
| `Qwen-Drive-1.0-planner-sft-BF16.gguf` | 2,079,728,448 | 358 | `7ab6abef07523192aad877bc0186dcc6c82c854f951f00ba00c7ce06bcf89cd9` |
| `Qwen-Drive-1.0-planner-rl-BF16.gguf` | 2,079,728,448 | 358 | `05f205728c822382650b6ee43f8c343da09f2dce381e399c86f891b4411b9b5a` |
| `Qwen-Drive-1.0-perception-F32.gguf` | 500,350,464 | 827 | `ed3c415ac932f7c48e37ca6caea75e4fb3d399e4e66773dce23d9edc4e8bb0d3` |

The official Qwen-Drive source Oracle is fixed at
`28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0`.

Full CUDA end-to-end perception parity is unverified until the NVIDIA gate runs.
