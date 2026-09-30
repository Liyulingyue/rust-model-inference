"""Wrap Edge0-35B MLX-affine weights and LoRA in a GGUF container.

``--quant lossless`` (the default) stores the packed U32 words as GGUF I32
with identical bytes, keeping scales, biases and LoRA at BF16/F16.  This is
the format the Rust ``MlxAffineKernel`` consumes and the one the scalar oracle
verifies against, so it must stay byte-exact.

The other modes first expand the MLX affine groups back to F32 and then
re-encode the learned matrices into an ordinary GGML type, so the result
becomes readable by stock llama.cpp tooling.  These are a second, lossy
quantization pass: the source codes are 4-bit, so any output below F32 throws
away information the MLX checkpoint no longer has.  Norms, biases, the router
and the LoRA adapters always keep their source precision, matching how
``convert_breeze.py`` protects tensors the loader pins to a fixed type.

    lossless  pass the packed U32 words through unchanged (default)
    f32       expand the affine groups to F32
    f16       expand the affine groups to F16
    q8_0      expand to F32, then per-32 Q8_0 blocks
    q4_0      expand to F32, then per-32 Q4_0 blocks

This is an Edge0 GGUF, not a llama.cpp model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np

from tools.converter.edge0.mlx_affine import (
    MLX_GROUP_SIZE,
    bf16_to_f32,
    dequantize_matrix,
)
from tools.converter.utils.kquants import quantize_k
from tools.converter.utils.gguf import (
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_I32,
    GGML_Q4_0,
    GGML_Q4K,
    GGML_Q6K,
    GGML_Q8_0,
    GgufWriter,
    gguf_dims,
    open_safetensors,
    quantize_q4_0,
    quantize_q8_0,
)


LAYER = re.compile(r"language_model\.model\.layers\.(\d+)\.(.+)")
STEMS = {
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "post_attention_norm",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "self_attn.q_norm": "attn_q_norm",
    "self_attn.k_norm": "attn_k_norm",
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.in_proj_a": "ssm_alpha",
    "linear_attn.in_proj_b": "ssm_beta",
    "linear_attn.conv1d": "ssm_conv1d",
    "linear_attn.norm": "ssm_norm",
    "linear_attn.out_proj": "ssm_out",
    "mlp.gate": "ffn_gate_inp",
    "mlp.shared_expert_gate": "ffn_shared_gate",
    "mlp.shared_expert.gate_proj": "ffn_gate",
    "mlp.shared_expert.up_proj": "ffn_up",
    "mlp.shared_expert.down_proj": "ffn_down",
    "mlp.switch_mlp.gate_proj": "ffn_gate_exps",
    "mlp.switch_mlp.up_proj": "ffn_up_exps",
    "mlp.switch_mlp.down_proj": "ffn_down_exps",
}
SPECIAL = {
    "linear_attn.A_log": "ssm_a",
    "linear_attn.dt_bias": "ssm_dt.bias",
}
TOP = {
    "language_model.model.embed_tokens": "token_embd",
    "language_model.model.norm": "output_norm",
    "language_model.lm_head": "output",
}
DTYPES = {"U32": GGML_I32, "BF16": GGML_BF16, "F16": GGML_F16}

#: ``--quant`` choices, mapped to the GGML type the expanded matrices take.
#: ``None`` keeps the lossless I32 pass-through.  ``q4_k_m`` is a per-tensor mix
#: so it is handled separately by :func:`target_for`.
QUANT_MODES = {
    "lossless": None,
    "f32": GGML_F32,
    "f16": GGML_F16,
    "q8_0": GGML_Q8_0,
    "q6_k": GGML_Q6K,
    "q4_k": GGML_Q4K,
    "q4_0": GGML_Q4_0,
    "q4_k_m": None,
}
QUANT_SUFFIX = {
    "lossless": "lossless",
    "f32": "F32",
    "f16": "F16",
    "q8_0": "Q8_0",
    "q6_k": "Q6_K",
    "q4_k": "Q4_K",
    "q4_0": "Q4_0",
    "q4_k_m": "Q4_K_M",
}

#: Tensors ``q4_k_m`` promotes to Q6_K.  These carry the routed expert outputs
#: and the attention value projection, whose error propagates into every
#: downstream residual stream, so they get the extra 2.06 bits.  The choice
#: follows llama.cpp's LLM_TENSOR_MAP, which promotes the same two tensors.
Q6_K_STEMS = {
    "attn_v",
    "ffn_down",
    "ffn_down_exps",
}
#: Tensors ``q4_k_m`` keeps at 8 bits.  The routers pick which experts run, so
#: a flipped logit changes the whole trajectory rather than adding noise.
Q8_0_STEMS = {"ffn_gate_inp", "ffn_shared_gate"}
#: Files this converter always reads from the source checkpoint.
LORA_SHARD = "lora_edge0_35b.safetensors"


def gguf_name(name: str) -> str:
    for source, target in TOP.items():
        if name.startswith(source + "."):
            return target + name[len(source):]
    match = LAYER.fullmatch(name)
    if match is None:
        raise ValueError(f"unexpected Edge0 tensor: {name}")
    layer, tail = match.groups()
    if tail in SPECIAL:
        return f"blk.{layer}.{SPECIAL[tail]}"
    stem, sep, suffix = tail.rpartition(".")
    if not sep or stem not in STEMS or suffix not in {"weight", "scales", "biases", "lora_A", "lora_B"}:
        raise ValueError(f"unexpected Edge0 tensor: {name}")
    return f"blk.{layer}.{STEMS[stem]}.{suffix}"


def chunks(path: Path, offset: int, length: int):
    with path.open("rb") as source:
        source.seek(offset)
        while length:
            data = source.read(min(length, 8 << 20))
            if not data:
                raise ValueError(f"truncated tensor in {path}")
            yield data
            length -= len(data)


def read_bytes(path: Path, offset: int, length: int) -> bytes:
    return b"".join(chunks(path, offset, length))


def tensors(model_dir: Path):
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    if len(shard_names) != 4:
        raise ValueError(f"expected four Edge0 shards, got {shard_names}")
    seen = set()
    for shard_name in shard_names + [LORA_SHARD]:
        source = open_safetensors(model_dir / shard_name)
        for name, info in source.header.items():
            if name == "__metadata__":
                continue
            if shard_name in shard_names and index["weight_map"].get(name) != shard_name:
                raise ValueError(f"index mismatch for {name}")
            mapped = gguf_name(name)
            if mapped in seen:
                raise ValueError(f"duplicate tensor {mapped}")
            seen.add(mapped)
            shape = tuple(info["shape"])
            start, end = info["data_offsets"]
            dtype = info["dtype"]
            if dtype not in DTYPES or start < 0 or end < start or source.data_offset + end > source.file_size:
                raise ValueError(f"invalid tensor {name} in {shard_name}")
            nbytes = end - start
            if nbytes != (4 if dtype == "U32" else 2) * math.prod(shape):
                raise ValueError(f"invalid tensor size for {name}")
            yield mapped, dtype, shape, source.path, source.data_offset + start, nbytes
    if len(seen) != len(index["weight_map"]) + 620:
        raise ValueError("Edge0 tensor or LoRA inventory is incomplete")


class TensorEntry:
    """One source tensor, kept addressable so the affine triplets can pair up."""

    def __init__(self, mapped: str, dtype: str, shape: tuple[int, ...], path: Path, offset: int, nbytes: int):
        self.mapped = mapped
        self.dtype = dtype
        self.shape = shape
        self.path = path
        self.offset = offset
        self.nbytes = nbytes

    def read(self) -> bytes:
        return read_bytes(self.path, self.offset, self.nbytes)


def is_packed_matrix(entry: TensorEntry) -> bool:
    """True for the U32 ``*.weight`` tensors that carry MLX affine codes.

    Every other tensor (norms, SSM state params, the BF16 ``scales``/
    ``biases`` companions and the F16 LoRA adapters) keeps its source bytes.
    """
    return entry.dtype == "U32" and entry.mapped.endswith(".weight")


def target_for(name: str, mode: str) -> int | None:
    """Resolve the GGML type a given matrix takes under ``mode``.

    Most modes are a single type, but ``q4_k_m`` mirrors llama.cpp's mixed
    recipe: routers stay at 8 bits because a flipped logit reroutes the whole
    forward pass, the value and down projections are promoted to Q6_K because
    their error lands in every residual stream, and everything else takes Q4_K.
    """
    if mode != "q4_k_m":
        return QUANT_MODES[mode]
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    leaf = stem.rsplit(".", 1)[-1]
    if leaf in Q8_0_STEMS:
        return GGML_Q8_0
    if leaf in Q6_K_STEMS:
        return GGML_Q6K
    return GGML_Q4K


def encoded_nbytes(ggml_type: int, elements: int) -> int:
    """Byte length of ``elements`` values stored as ``ggml_type``."""
    if ggml_type == GGML_Q8_0:
        if elements % 32:
            raise ValueError(f"Q8_0 needs a multiple of 32 elements, got {elements}")
        return (elements // 32) * 34
    if ggml_type == GGML_Q4_0:
        if elements % 32:
            raise ValueError(f"Q4_0 needs a multiple of 32 elements, got {elements}")
        return (elements // 32) * 18
    if ggml_type in (GGML_Q4K, GGML_Q6K):
        if elements % 256:
            raise ValueError(f"k-quant needs a multiple of 256 elements, got {elements}")
        return (elements // 256) * {GGML_Q4K: 144, GGML_Q6K: 210}[ggml_type]
    return elements * {GGML_F32: 4, GGML_F16: 2, GGML_BF16: 2, GGML_I32: 4}[ggml_type]


def matrix_axes(entry: TensorEntry, scales: TensorEntry) -> tuple[int, int, int, int, int]:
    """Resolve ``(n_out, n_in, packed_cols, bits, experts)`` for one matrix.

    The safetensors axes run opposite to GGUF, so on a packed matrix the last
    axis is the packed width, the one before it the output rows, and an
    optional leading axis the expert index.  The BF16 companions follow the
    same row axis but end in the group count.
    """
    if entry.shape[-2] != scales.shape[-2]:
        raise ValueError(f"{entry.mapped}: row axis {entry.shape[-2]} != scales {scales.shape[-2]}")
    n_out = entry.shape[-2]
    groups = scales.shape[-1]
    n_in = groups * MLX_GROUP_SIZE
    packed_cols = entry.shape[-1]
    if packed_cols * 32 % n_in:
        raise ValueError(f"{entry.mapped}: packed width {packed_cols} does not match {n_in} inputs")
    bits = packed_cols * 32 // n_in
    experts = entry.shape[0] if len(entry.shape) == 3 else 1
    if bits not in (4, 8):
        raise ValueError(f"{entry.mapped}: unsupported MLX affine bit width {bits}")
    return n_out, n_in, packed_cols, bits, experts


def expand_matrix(
    packed: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    n_out: int,
    n_in: int,
    bits: int,
) -> np.ndarray:
    """Dequantize one MLX affine matrix into an F32 ``(n_out, n_in)`` array."""
    return dequantize_matrix(packed, scales, biases, (n_out, n_in), bits)


def reencode(values: np.ndarray, ggml_type: int, label: str = "") -> bytes:
    """Encode an F32 matrix into the GGML payload for ``ggml_type``.

    The k-quants run a 20-candidate scale search per value, which is orders of
    magnitude slower than the closed-form formats, so they report progress per
    chunk; ``label`` names the tensor being converted.
    """
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    if ggml_type == GGML_F32:
        return flat.tobytes()
    if ggml_type == GGML_F16:
        return flat.astype(np.float16).tobytes()
    if ggml_type == GGML_Q8_0:
        return quantize_q8_0(flat)
    if ggml_type == GGML_Q4_0:
        return quantize_q4_0(flat)
    if ggml_type in (GGML_Q4K, GGML_Q6K):
        name = "q4_k" if ggml_type == GGML_Q4K else "q6_k"
        return quantize_k(name, flat, _progress_reporter(label, name))
    raise ValueError(f"unsupported re-encode target {ggml_type}")


def _progress_reporter(label: str, name: str):
    """A ``progress(done, total)`` callback that prints a rate and ETA."""
    state = {"start": None}

    def report(done: int, total: int) -> None:
        if state["start"] is None:
            state["start"] = time.monotonic()
            print(
                f"  {label} [{name}]: {total} super-blocks",
                flush=True,
            )
            return
        elapsed = time.monotonic() - state["start"]
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (total - done) / rate if rate > 0 else float("inf")
        print(
            f"  {label} [{name}]: {done}/{total} blocks "
            f"({100.0 * done / total:5.1f}%) {rate:6.1f} blk/s eta {remaining / 60:6.1f} min",
            flush=True,
        )

    return report


def emit_packed(
    writer: GgufWriter,
    entry: TensorEntry,
    scales: TensorEntry,
    biases: TensorEntry,
    ggml_type: int | None,
) -> None:
    """Write one affine matrix in the requested mode.

    ``lossless`` streams the original U32 words.  Every other mode expands to
    F32, re-encodes, and drops the now-redundant ``scales``/``biases``
    companions.  Expert-stacked tensors are emitted one expert at a time so a
    500 MB F32 intermediate never materialises.
    """
    name = entry.mapped
    if ggml_type is None:
        writer.add_tensor_chunks(
            name, GGML_I32, gguf_dims(entry.shape), entry.nbytes,
            lambda e=entry: chunks(e.path, e.offset, e.nbytes),
        )
        return

    n_out, n_in, _packed_cols, bits, experts = matrix_axes(entry, scales)
    values_per_word = 32 // bits
    row_bytes = (n_in // values_per_word) * 4
    matrix_bytes = n_out * row_bytes
    # Each expert owns a contiguous slice of the companions, so the byte
    # stride is the element count times the BF16 element size.
    scale_values = n_out * (n_in // MLX_GROUP_SIZE)
    scale_stride = scale_values * 2
    per_expert = n_out * n_in

    if experts > 1:
        # GGUF keeps the expert axis last, matching the lossless layout.
        out_dims = (n_in, n_out, experts)
        payload = encoded_nbytes(ggml_type, per_expert) * experts

        def expert_chunks(
            e=entry, s=scales, b=biases, n=experts, mb=matrix_bytes, sb=scale_stride
        ):
            raw_packed = e.read()
            raw_scale, raw_bias = s.read(), b.read()
            for index in range(n):
                lo = index * mb
                so = index * sb
                yield reencode(
                    expand_matrix(
                        np.frombuffer(raw_packed[lo : lo + mb], dtype=np.uint8).reshape(n_out, row_bytes),
                        bf16_to_f32(raw_scale[so : so + sb]),
                        bf16_to_f32(raw_bias[so : so + sb]),
                        n_out,
                        n_in,
                        bits,
                    ),
                    ggml_type,
                    f"{name} expert {index}",
                )

        writer.add_tensor_chunks(name, ggml_type, out_dims, payload, expert_chunks)
        return

    matrix = expand_matrix(
        np.frombuffer(entry.read(), dtype=np.uint8).reshape(n_out, row_bytes),
        bf16_to_f32(scales.read()),
        bf16_to_f32(biases.read()),
        n_out,
        n_in,
        bits,
    )
    writer.add_tensor(name, ggml_type, (n_in, n_out), reencode(matrix, ggml_type, name))


def add_metadata(writer: GgufWriter, model_dir: Path, mode: str) -> None:
    config = json.loads((model_dir / "config.json").read_text())
    text = config["text_config"]
    quant = config["quantization"]
    if (config["model_type"], text["num_hidden_layers"], text["num_experts"],
        quant["group_size"], quant["bits"], quant["mode"]) != (
        "qwen3_5_moe", 40, 256, 64, 4, "affine"
    ):
        raise ValueError("unsupported Edge0-35B model contract")
    writer.add_meta("general.architecture", "edge0")
    writer.add_meta("general.name", "Edge0-35B-A3B-preview")
    writer.add_meta("edge0.quant.mode", mode)
    if mode == "lossless":
        writer.add_meta("edge0.quant.group_size", 64)
    writer.add_meta("edge0.expert_count", 256)
    writer.add_meta("edge0.expert_used_count", 4)
    writer.add_meta("edge0.expert_feed_forward_length", text["moe_intermediate_size"])
    writer.add_meta("edge0.shared_expert_feed_forward_length", text["shared_expert_intermediate_size"])
    writer.add_meta("edge0.lora.scale", 2.0)
    values = {
        "block_count": text["num_hidden_layers"],
        "context_length": text["max_position_embeddings"],
        "embedding_length": text["hidden_size"],
        "feed_forward_length": text["shared_expert_intermediate_size"],
        "attention.head_count": text["num_attention_heads"],
        "attention.head_count_kv": text["num_key_value_heads"],
        "attention.key_length": text["head_dim"],
        "attention.value_length": text["head_dim"],
        "attention.layer_norm_rms_epsilon": text["rms_norm_eps"],
        "rope.dimension_count": int(text["head_dim"] * text["partial_rotary_factor"]),
        "rope.dimension_sections": [*text["rope_parameters"]["mrope_section"], 0],
        "rope.freq_base": text["rope_parameters"]["rope_theta"],
        "ssm.conv_kernel": text["linear_conv_kernel_dim"],
        "ssm.state_size": text["linear_key_head_dim"],
        "ssm.group_count": text["linear_num_key_heads"],
        "ssm.time_step_rank": text["linear_num_value_heads"],
        "ssm.inner_size": text["linear_num_value_heads"] * text["linear_value_head_dim"],
        "full_attention_interval": text["full_attention_interval"],
        "vocab_size": text["vocab_size"],
    }
    for key, value in values.items():
        writer.add_meta(f"edge0.{key}", value)
    tokenizer = json.loads((model_dir / "tokenizer.json").read_text())
    vocab = {int(id): token for token, id in tokenizer["model"]["vocab"].items()}
    added = tokenizer["added_tokens"]
    vocab.update({entry["id"]: entry["content"] for entry in added})
    tokens = [vocab.get(i, f"<|reserved_{i}|>") for i in range(text["vocab_size"])]
    types = [1] * len(tokens)
    for entry in added:
        types[entry["id"]] = 3
    writer.add_meta("tokenizer.ggml.model", "gpt2")
    writer.add_meta("tokenizer.ggml.pre", "qwen2")
    writer.add_meta("tokenizer.ggml.tokens", tokens)
    writer.add_meta("tokenizer.ggml.token_type", types)
    writer.add_meta("tokenizer.ggml.merges", [" ".join(pair) for pair in tokenizer["model"]["merges"]])
    writer.add_meta("tokenizer.ggml.eos_token_id", 248046)
    writer.add_meta("tokenizer.ggml.bos_token_id", 248044)
    writer.add_meta("tokenizer.ggml.add_bos_token", False)
    writer.add_meta("tokenizer.ggml.add_eos_token", False)
    writer.add_meta("tokenizer.chat_template", (model_dir / "chat_template.jinja").read_text())


def convert(model_dir: Path, output: Path, check_only: bool, mode: str = "lossless") -> None:
    writer = GgufWriter(output.with_suffix(output.suffix + ".part"))
    add_metadata(writer, model_dir, mode)
    entries = [TensorEntry(*rest) for rest in tensors(model_dir)]
    by_name = {entry.mapped: entry for entry in entries}
    # The expanded modes fold each affine group into the matrix itself, so the
    # per-group scale/bias tensors are no longer part of the output.
    companions = {".scales", ".biases"}
    skipped = 0
    count = 0
    total = 0
    kinds: dict[int, int] = {}
    # The k-quants run a per-value scale search, so a full conversion takes tens
    # of minutes; report which tensor is in flight and how far along we are.
    packed_total = sum(1 for e in entries if is_packed_matrix(e))
    packed_done = 0
    started = time.monotonic()
    for entry in entries:
        ggml_type = target_for(entry.mapped, mode)
        if ggml_type is not None and entry.mapped.endswith(tuple(companions)):
            if entry.dtype == "BF16":
                skipped += 1
                continue
        if is_packed_matrix(entry):
            stem = entry.mapped[: -len(".weight")]
            try:
                scales = by_name[stem + ".scales"]
                biases = by_name[stem + ".biases"]
            except KeyError as exc:
                raise ValueError(f"{entry.mapped}: missing affine companion {exc}") from exc
            packed_done += 1
            if ggml_type in (GGML_Q4K, GGML_Q6K):
                print(
                    f"[{packed_done}/{packed_total}] {entry.mapped} "
                    f"({time.monotonic() - started:.0f}s elapsed)",
                    flush=True,
                )
            emit_packed(writer, entry, scales, biases, ggml_type)
            kinds[ggml_type] = kinds.get(ggml_type, 0) + 1
        else:
            writer.add_tensor_chunks(
                entry.mapped, DTYPES[entry.dtype], gguf_dims(entry.shape), entry.nbytes,
                lambda e=entry: chunks(e.path, e.offset, e.nbytes),
            )
        count += 1
        total += _payload_nbytes(entry, ggml_type, by_name)
    if mode == "lossless":
        print(f"validated {count} lossless tensors; payload {total:,} bytes", flush=True)
    else:
        names = {v: k for k, v in QUANT_MODES.items() if v is not None}
        mix = " ".join(f"{names.get(t, t)}x{n}" for t, n in sorted(kinds.items()))
        print(
            f"validated {count} tensors in {mode} [{mix}]; dropped {skipped} affine "
            f"companions; payload {total:,} bytes",
            flush=True,
        )
    if check_only:
        return
    if output.exists() or writer.path.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < total + (64 << 20):
        raise OSError("not enough free space for the Edge0 GGUF")
    try:
        writer.write()
        os.replace(writer.path, output)
    except BaseException:
        writer.path.unlink(missing_ok=True)
        raise
    print(output, flush=True)


def _payload_nbytes(entry: TensorEntry, ggml_type: int | None, by_name: dict[str, TensorEntry]) -> int:
    """Bytes this tensor contributes to the output, in the active mode."""
    if ggml_type is None or not is_packed_matrix(entry):
        return entry.nbytes
    stem = entry.mapped[: -len(".weight")]
    n_out, n_in, _packed_cols, _bits, experts = matrix_axes(
        entry, by_name[stem + ".scales"]
    )
    return encoded_nbytes(ggml_type, n_out * n_in) * experts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--quant",
        choices=sorted(QUANT_MODES),
        default="lossless",
        help="output precision; 'lossless' passes the MLX affine codes through unchanged",
    )
    args = parser.parse_args()
    if not args.check and args.out is None:
        parser.error("--out is required unless --check is set")
    suffix = QUANT_SUFFIX[args.quant]
    default_out = args.model_dir / f"Edge0-35B-A3B-preview-{suffix}.gguf"
    convert(args.model_dir, args.out or default_out, args.check, args.quant)
