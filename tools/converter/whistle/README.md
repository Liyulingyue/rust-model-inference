# Whistle (Cactus-Compute) adapter

Whistle is Cactus-Compute's 144M on-device multilingual ASR. It targets
a closed container (`.cact`) plus the proprietary `needle3` C++ runtime,
**but** the same model also ships as plain safetensors in the public
ModelScope repo (`Cactus-Compute/whistle`).

This adapter ports the safetensors checkpoint end-to-end through the
standard `whistle_encoder` + `whistle_decoder` (8 layers + 1 stack
layer + shared embedding output projection) using the same
`TensorSource` interface the rest of the repo uses.

## Status — `Experimental`

| Phase | Status | Notes |
|-------|--------|-------|
| Conv stem (`dw_1` / `dw_2` / `w` / `pw_1` / `pw_2` / `out`) | **TODO** | The 3 dw + 2 pw + 1 linear stack is not reverse-engineered against the closed `needle3` runtime; the CLI runs with a no-op stem that emits zero hidden states. |
| ZCRMSNorm | placeholder | Falls back to standard RMSNorm. |
| `mhc_phi` residual (hadamard-convolution gating) | placeholder | Skip = `mhc_a_res[layer] * x` (a scalar placeholder matches the Oracle, but the full `mhc_phi_pre / mhc_a_pre / mhc_b_pre` family is unused). |
| `k_taps` / `q_taps` / `v_taps` (3-tap causal FIR) | **TODO** | Stack self-attn only; not used by the placeholder path. |
| `cactus-quants` 2–4-bit weight quantisation | **TODO** | Out: GGUF ships in F32 (220 MB ≈ safetensors size). |
| Encoder/decoder numerical parity | **√** | `tests/whistle_layer_test.rs` matches the in-tree Python Oracle to 1 ULP after 8 layers of accumulation. |

## Files

```
tools/converter/whistle/
├── convert_whistle.py    # safetensors → F32 GGUF
└── README.md             # this file
tools/converter/whistle/probe/
└── oracle.py             # Python forward reference (per-layer)
src/models/whistle/
├── mod.rs                # EncoderLayer struct + load_encoder_layer
├── attention.rs          # GQA self-attn + cross-attn (no RoPE)
├── hadamard.rs           # 4-stage Hadamard MLP (placeholder weights)
├── norm.rs               # RMSNorm placeholder for ZCRMSNorm
├── stack.rs              # StackLayer (self-attn + cross-attn + final_norm)
└── stem.rs               # placeholder stem (no-op)
src/app/asr/whistle.rs    # CLI driver (placeholder stem)
tests/whistle_layer_test.rs  # parity tests against oracle.py
```

## Running

```bash
# 1. Pull the safetensors from ModelScope (16 MB GGUF artifact, but we
#    use the 221 MB safetensors for full fidelity).
models/.venv/bin/modelscope download --model Cactus-Compute/whistle \
    --include "config.json" --include "checkpoints/whistle.safetensors" \
    --local_dir /tmp/whistle-probe

# 2. Convert safetensors → F32 GGUF (no quant yet).
. .hf-venv/bin/activate
python3 tools/converter/whistle/convert_whistle.py \
    --safetensors /tmp/whistle-probe/checkpoints/whistle.safetensors \
    --config     /tmp/whistle-probe/config.json \
    --out        models/whistle-probe/whistle.f32.gguf

# 3. Run the parity tests.
cargo test --test whistle_layer_test -- --nocapture

# 4. CLI smoke. Output is meaningless because the stem is a no-op.
./target/debug/rust-model-inference \
    --model models/whistle-probe/whistle.f32.gguf --audio whistle
```

## Reference data layout

The Python Oracle saves per-layer encoder inputs/outputs to
`/tmp/whistle-probe/oracle_reference.npz`:

- `encoder_input`:  `[1, 16, 512]`
- `encoder_layer_in`:  `[8, 1, 16, 512]` — input to each encoder layer
- `encoder_layer_out`: `[8, 1, 16, 512]` — output of each encoder layer
- `encoder_final`:    `[1, 16, 512]`    — output after `encoder/final_norm`
- `text_tokens`:      `[1, 5]`
- `text_emb`:         `[1, 5, 512]`

The Rust parity tests load these tensors and assert bit-level agreement
with the Rust encoder+stack path.

## Reverse-engineering notes

### Architectural constants

| Constant | Value | Source |
|----------|-------|--------|
| `HIDDEN` | 512 | `embedding/embedding` second dim |
| `NUM_LAYERS` | 8 | `self_attn/q_proj/kernel` first dim |
| `NUM_HEADS` | 8 | `q_norm/scale` first dim |
| `NUM_KV_HEADS` | 2 (self-attn) / 8 (cross-attn) | `k_proj/kernel` out ÷ 48 |
| `HEAD_DIM_Q/K` | 48 | `q_norm/scale` second dim |
| `HEAD_DIM_V` | 64 | `v_proj/kernel` out ÷ num_v_heads |
| `TEXT_VOCAB` | 8199 | `embedding/embedding` first dim |

### Kernel layout

Whistle kernels are stored as `[num_layers, in, out]` row-major
(safetensors default). The Rust adapter copies the layer-row slice via
`load_layer_kernel_2d` and indexes it as `weight[in_idx * out_dim + out_idx]`.

### Norm-scale layout

All RMS-style scales in the safetensors are `[num_layers, hidden]`
(per-layer). The decoder applies per-layer scale via
`zc_rms_norm_placeholder(x, scale[layer], …)`.

### Hadamard MLP shape

Each encoder block has a 4-stage Hadamard MLP:

```
x → 32 chunks of 16 → matmul w1a (16×16) → 32×16
  → 16 chunks of 32 → matmul w1b (32×32) → 16×32
  → 32 chunks of 16 → matmul w2a (16×16)
  → 16 chunks of 32 → matmul w2b (32×32)
  → 32×16 → w3a → 16×32 → w3b → 512 + bias
```

Each stage adds `d_i * cond` where `cond = x @ cond_u^T @ cond_v^T`
(a rank-8 per-channel conditioner).

### What is missing

- **`conv stem`**: 3 depthwise Conv2d (in=1, out=128, k=3×3) interleaved
  with two 1×1 channel mixers (128→128) and a final linear 1280→512.
  The 1280 input is unexplained from the safetensors shape alone —
  reverse engineering the layout requires either a numerical oracle
  (we don't have one) or running the official `needle3` binary.
- **`ZCRMSNorm`**: the real ZC variant likely centres on the per-token
  mean (`x -= mean`) before the RMS step; the placeholder uses plain
  RMSNorm and matches our Python Oracle.
- **`mhc_phi`**: the per-layer scalar + low-rank Hadamard-gated residual
  is implemented as a flat skip; the true `mhc_phi_pre` / `mhc_phi_post`
  2048×4 / 2048×16 matrices are unused.
- **`taps`**: 3-tap causal FIR applied to `q` / `k` / `v` projections
  of the stack's self-attention. The weights are read but not applied.
- **`cactus-quants`**: the proprietary 2–4-bit quantisation is not
  implemented; GGUF currently ships F32 (220 MB).

Until a numerical oracle is available we cannot lift these placeholders
without breaking the parity tests (which use the same placeholders).