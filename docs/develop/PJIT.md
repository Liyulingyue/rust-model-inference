# P-JiT 21K (Experimental / partial)

The released Linum-AI model generates still images directly in pixel space. It
combines a 2.184B-parameter DiT with Qwen3.5-4B raw block 7/15/27 caption features;
no VAE or image projector is required. The existing inference binary accepts a
`pjit` DiT GGUF plus `--text-encoder`; the existing server exposes
`POST /v1/images/generations`. Model computation is shared in
`src/models/diffusion/pjit`. See the [run instructions](../../tools/oracle/pjit/README.md).

Converters/loaders support F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K.
Unaligned K rows retain source floats. Storage, shared activation rounding and
F32 accumulation are recorded separately; BF16 storage does not imply native
BF16 activation arithmetic. All 18 exports have verified
[hash/type manifests](../../tools/oracle/pjit/gguf_hashes.json). Weights remain on
the authorized Linux host; only small validation records/images are on the Mac.

All 27 storage/backend combinations passed real CLI and HTTP requests, including
PNG shape and malformed-request checks. Corresponding CLI/HTTP PNG bytes match.
Float DiT single-forward CPU/Auto/Vulkan gates and the post-APG-fix seven-step F32
sampling chain pass. The chain's final NRMSE is 0.0197263 / cosine 0.99980654,
superseding the pre-fix 0.03215 failure. A real 1024×1024 / 50-step F16 CPU
generation also completed; its official same-feature/noise replay passes all
50 gates (final NRMSE 0.0196754, cosine 0.99980697). Decoded PNG RGB RMSE is
1.15923 on the 0–255 scale (PSNR 46.85 dB). This is one prompt, not a broad
perceptual-quality evaluation. Quantized end-to-end gates remain open;
independent same-input projection replay is reported separately, without hiding
full-forward failures or changing thresholds. Exact token IDs for eight Unicode,
whitespace, special-token and truncation cases match the official template.
See the [full matrix and measurements](../../tools/oracle/pjit/validation.md).

CPU uses existing SIMD kernels and the compute pool. Auto/Vulkan executes DiT
projections and supported attention on the device; norms, RoPE, elementwise work,
sampling and caption extraction stay on CPU. Caption extraction recomputes three
prefixes through public Qwen APIs. The resident device decoder exposes normalized
final states/logits, rather than raw features at several layers; a shared
compute/Qwen follow-up must add that API and validate the caption matrix. No
other model or shared kernel implementation was changed for this adapter.

Same-host release warm F32 single-forward baseline (33×35, 18 caption tokens,
batch 1, four repeats, median of the final three): CPU **0.771232 s**, Vulkan
**3.058170 s**, official PyTorch ROCm **0.197731 s**. The Vulkan path remains
slower. It includes synchronous per-projection transfers/submits and host work.
The container has a 16-CPU quota and 55 GiB RAM despite reporting 128 logical CPUs;
other work shares the host. Timing scopes, cold samples and limits are in
[performance.json](../../tools/oracle/pjit/performance.json).

The model batches up to 256 projection rows and reuses spatial RoPE phases across
tokens/heads. Default-size output, preserved 1024×1024 old/new forward checks and
remaining numerical limits are recorded in the validation matrix. This remains
an Experimental partial adaptation until the numerical and shared-device gaps
are closed.
