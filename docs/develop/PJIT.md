# P-JiT 21K (Experimental / partial)

The released Linum-AI P-JiT model generates still images directly in pixel space.
It combines a 2.184B-parameter DiT with Qwen3.5-4B caption features from raw blocks
7, 15 and 27. No VAE or image projector is required.

The existing inference binary accepts a `pjit` DiT GGUF and `--text-encoder` Qwen
GGUF. The existing server exposes `POST /v1/images/generations`. Model semantics,
caption extraction, padding/cropping, APG and Euler sampling are shared in
`src/models/diffusion/pjit`. See the [run instructions](../../tools/oracle/pjit/README.md).

Storage formats implemented by the converter and loader are F32, F16, BF16,
Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K. Unaligned K rows retain source float
storage. All arithmetic uses the repository's existing kernels; BF16 storage
does not mean native BF16 activation arithmetic. CPU uses shared SIMD and the
compute pool. Auto/Vulkan offloads DiT projections and supported attention;
normalization, RoPE, elementwise work and sampling remain on the host.

Caption extraction currently recomputes three CPU Qwen prefixes through the
public Qwen API. The resident GPU decoder does not expose raw intermediate
states at multiple layers. A follow-up in the shared compute/Qwen layer must
add that API and verify the caption matrix before this becomes a full device
pipeline. The new adapter does not change the existing Qwen implementation.

Numerical status: float DiT single-forward checks passed; quantized full-forward
and the recorded seven-step sampler did not meet the predeclared NRMSE gate.
Q8 same-input independent replay passed all 217 projections (worst NRMSE
5.41e-7); the end-to-end failure remains a separate open gate.
Tokenizer contracts passed. Small CLI/HTTP CPU and Auto runs passed for the
first six formats. Remaining device transport checks, K text checks, final
multi-step rerun, default-size quality and warm performance checks were blocked
when the authorized SSH host stopped accepting connections.
The replacement host also became unreachable during validation; its source
hash manifest and observed Q8 replay summary are preserved in the Oracle records.
This model is not
listed as fully verified. Exact results and the complete matrix are in
[validation.md](../../tools/oracle/pjit/validation.md).

Diagnostic same-host 33×35 / 18-caption-token F32 prediction timings were
7.803182 s Rust CPU and 44.780257 s Rust Vulkan (including cold session/upload),
versus 2.341076 s official PyTorch ROCm F32 (already loaded, checkpoint I/O).
These scopes do not support a speed ratio. The release/warm benchmark was
interrupted; see [performance.json](../../tools/oracle/pjit/performance.json).
