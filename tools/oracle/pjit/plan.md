# P-JiT 21K implementation and validation plan

Input: ModelScope Linum-AI/pjit-21k revision 695b80dd382c45a049930c86a7a7590a81a1e918;
official read-only Oracle Linum-AI/pjit-21k commit 95403933f3727abe947977c75915a53292e1d841.
Starting base: b0c193fab03d5b02b16ed33e01911809d5d86180. Branch: codex/pjit-21k.
Rebased onto upstream d16dc16d96126351b10c13c8458f968c5c57b2e7 before submission.

Architecture: local pixel DiT (no VAE), Qwen3.5-4B features at blocks 7/15/27,
APG/Euler sampler. Shared LinearExecutor, normalization, attention and RNG;
existing CLI/server binaries. No external acceleration library in Rust.
All weights remain on the authorized SSH host; its checkout's models symlink
points to a larger root-filesystem volume. No weights are stored on the Mac.

Changes: new models/diffusion/pjit, converter/pjit, oracle/pjit; necessary diffusion
module/main CLI and image-generation HTTP registration. No other model changes.
Text feature extraction composes existing public Qwen3.5 APIs locally; no trunk changes.

- [x] Contract/converter checks; explicit arch and exact tensor inventory.
- [x] DiT, padding/cropping, text features, APG and deterministic sampler checks.
- [x] CLI and HTTP share models implementation and validate incompatible options.
- [ ] Full real-weight matrix: see validation.md for failures and unfinished rows.
- [x] Self-review and repair findings; local checks and documentation.
- [ ] Remote follow-up after SSH availability is restored; release/default-size checks.

Coverage: DiT and Qwen text F32/F16/BF16/Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K;
CPU/Auto/Vulkan × CLI/HTTP. K formats retain source floats for unaligned rows.
IQ export: not applicable (no existing encoder); loader-only formats are not new exports.
JEV/streaming: not applicable to pixel image generation. No remote prompt upsampler.
Track implemented and verified states separately in validation.md. Never call CPU fallback GPU support.
Thresholds before measurement: discrete shapes/token IDs exact; F32 DiT reference
max_abs <= 0.002 and cosine >= .99999; lossy same-weight reference normalized RMSE <= .02,
cosine >= .999; quantization quality measured separately against F32, not an implementation gate.
Native BF16 sampling vs F32 arithmetic measured separately; no bitwise float acceptance gate.
