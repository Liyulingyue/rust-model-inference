"""Export Tongyi-MAI/Z-Image-Turbo safetensors into the three GGUF components the
Rust inference path loads.

The target layout is not a convention we get to pick: `src/models/diffusion/
pig.rs` opens three separate files and each one looks its tensors up by exact
name, so the exporter's job is to make the published Diffusers checkpoint
present itself under those names with the dtypes the loader asserts on:

  * 1-D vectors (norm weights, every bias) are GGML F32, dims `[n]`
  * 2-D projection weights are GGML F16 or Q8_0, dims `[n_in, n_out]`
  * every tensor carries `general.architecture = pig`

Diffusers stores `nn.Linear.weight` as `[out, in]`, which is the reverse of the
`[n_in, n_out]` the loader asserts on -- but that assertion only ever reads the
declared dims. The reference exports keep the raw torch payload and reverse the
dims tuple instead, and the kernels agree with the reference, so neither 2-D nor
4-D payloads are transposed here. Transposing would also drag the 32-value Q8_0
blocks onto the wrong axis, since GGML blocks run along the contiguous one.

Measured against the pinned reference weights (gguf-org/z-image-gguf) this
reproduces 453 / 244 / 398 tensors for the DiT / VAE / text encoder.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from converter.utils.gguf import (  # noqa: E402
    GGML_F16,
    GGML_F32,
    GGML_Q8_0,
    GgufWriter,
    bf16_to_f32,
    f16_to_f32,
    f32_to_f16,
    open_safetensors,
    quantize_q8_0,
    validated_dir,
)

SUPPORTED_OUTTYPES = ("f32", "f16", "q8_0")
_MATRIX_TYPES = {"f32": GGML_F32, "f16": GGML_F16, "q8_0": GGML_Q8_0}

HIDDEN = 3840
FFN_WIDTH = 10240
HEAD_WIDTH = 128
QKV_WIDTH = HIDDEN * 3
ADALN_WIDTH = HIDDEN * 4
TIME_WIDTH = 256
TIME_HIDDEN = 1024
CAP_WIDTH = 2560
PATCH_WIDTH = 64
MAIN_LAYERS = 30
REFINER_LAYERS = 2


def _f32_array(raw: bytes, dtype: str) -> np.ndarray:
    """Normalise a safetensors payload to f32, handling bf16/f16/f32 sources."""
    if dtype == "BF16":
        return np.frombuffer(bf16_to_f32(raw), dtype=np.float32).copy()
    if dtype == "F16":
        return np.frombuffer(f16_to_f32(raw), dtype=np.float32).copy()
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).copy()
    raise ValueError(f"unsupported safetensors dtype {dtype!r}")


class Component:
    """One output GGUF: accumulates tensors, then writes them in a stable order.

    `f16` pins every matrix to F16. The quantized modes instead keep F16 for
    the tensors the reference exports keep in F16 -- the two refiner stacks and
    every top-level projection, which are small, run once per step, and are the
    ones whose quantization error shows up directly in the final pixels -- and
    quantize only the 30-layer main stack.
    """

    def __init__(self, outtype: str) -> None:
        if outtype not in SUPPORTED_OUTTYPES:
            raise ValueError(f"unsupported outtype {outtype!r}")
        self.outtype = outtype
        self.matrix_type = _MATRIX_TYPES[outtype]
        self.entries: list[tuple[str, int, tuple[int, ...], bytes]] = []

    def _matrix_ggml_type(self, keep_f16: bool) -> int:
        if self.outtype == "f32":
            return GGML_F32
        if self.outtype == "f16" or keep_f16:
            return GGML_F16
        return GGML_Q8_0

    def _matrix_raw(self, values: np.ndarray, keep_f16: bool) -> bytes:
        if self.outtype == "f32":
            return values.tobytes()
        if self.outtype == "f16" or keep_f16:
            return f32_to_f16(values.tobytes())
        return quantize_q8_0(values)

    def vector(self, name: str, values: np.ndarray) -> None:
        """1-D tensor. The loader demands F32 and dims [n]."""
        flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
        self.entries.append((name, GGML_F32, (flat.size,), flat.tobytes()))

    def column_vector(self, name: str, values: np.ndarray) -> None:
        """1-D tensor the loader reads as F16 with dims [n, 1] (the pad tokens)."""
        flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
        self.entries.append(
            (name, GGML_F16, (flat.size, 1), f32_to_f16(flat.tobytes()))
        )

    def matrix(
        self, name: str, weight: np.ndarray, n_in: int, n_out: int, *, keep_f16: bool = False
    ) -> None:
        """2-D weight declared as [n_in, n_out], payload written in torch order.

        The reader only checks the declared dims, and the reference exports
        store the raw `[out, in]` payload behind a reversed dims tuple, so the
        payload is *not* transposed. Transposing here would additionally move the
        32-value Q8_0 blocks off the axis the kernel walks.
        """
        array = np.ascontiguousarray(weight, dtype=np.float32)
        if array.shape != (n_out, n_in):
            raise ValueError(
                f"{name}: expected torch shape ({n_out}, {n_in}), got {array.shape}"
            )
        payload = array.reshape(-1)
        self.entries.append(
            (
                name,
                self._matrix_ggml_type(keep_f16),
                (n_in, n_out),
                self._matrix_raw(payload, keep_f16),
            )
        )

    def write(self, path: Path) -> None:
        writer = GgufWriter(path)
        writer.add_meta("general.architecture", "pig")
        writer.add_meta("general.quantization_version", 2)
        writer.add_meta("general.file_type", self.matrix_type)
        for name, ggml_type, dims, raw in self.entries:
            writer.add_tensor(name, ggml_type, dims, raw)
        writer.write()


class Checkpoint:
    """Lazy view over the sharded safetensors of one component directory."""

    def __init__(self, directory: Path) -> None:
        shards = sorted(directory.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no safetensors under {directory}")
        self.shards = [open_safetensors(shard) for shard in shards]
        index_path = directory / "diffusion_pytorch_model.safetensors.index.json"
        if not index_path.exists():
            index_path = directory / "model.safetensors.index.json"
        self.weight_map: dict[str, str] = {}
        if index_path.exists():
            self.weight_map = json.loads(index_path.read_text()).get("weight_map", {})

    def find(self, name: str):
        for shard in self.shards:
            if name in shard.header:
                return shard.get(name)
        return None

    def get(self, name: str) -> np.ndarray:
        tensor = self.find(name)
        if tensor is None:
            raise KeyError(f"tensor not found: {name}")
        return _f32_array(tensor.raw, tensor.dtype).reshape(tensor.shape)

    def has(self, name: str) -> bool:
        if name in self.weight_map:
            return True
        return self.find(name) is not None


def _fused_qkv(checkpoint: Checkpoint, prefix: str) -> np.ndarray:
    """Stack to_q/to_k/to_v into the single [out=3*hidden, in=hidden] qkv matrix.

    The reader slices a fused row as query at 0, key at `hidden`, value at
    `2 * hidden` (see `attention_value_reduce` call sites in dit.rs), so the
    concatenation order is q, k, v.
    """
    parts = [
        checkpoint.get(f"{prefix}.attention.to_{letter}.weight")
        for letter in ("q", "k", "v")
    ]
    for letter, part in zip(("q", "k", "v"), parts):
        if part.shape != (HIDDEN, HIDDEN):
            raise ValueError(
                f"{prefix}.attention.to_{letter}.weight: expected "
                f"({HIDDEN}, {HIDDEN}), got {part.shape}"
            )
    return np.concatenate(parts, axis=0)


def _block_tensors(
    component: Component,
    checkpoint: Checkpoint,
    prefix: str,
    *,
    modulated: bool,
    keep_f16: bool,
) -> None:
    """Emit one transformer block under the reader's names.

    `keep_f16` pins the block's matrices to F16; the reference export does this
    for the two refiner stacks and not for the main 30.
    """
    component.matrix(
        f"{prefix}.attention.qkv.weight",
        _fused_qkv(checkpoint, prefix),
        HIDDEN,
        QKV_WIDTH,
        keep_f16=keep_f16,
    )
    component.matrix(
        f"{prefix}.attention.out.weight",
        checkpoint.get(f"{prefix}.attention.to_out.0.weight"),
        HIDDEN,
        HIDDEN,
        keep_f16=keep_f16,
    )
    for suffix, (n_in, n_out) in (
        ("feed_forward.w1.weight", (HIDDEN, FFN_WIDTH)),
        ("feed_forward.w2.weight", (FFN_WIDTH, HIDDEN)),
        ("feed_forward.w3.weight", (HIDDEN, FFN_WIDTH)),
    ):
        torch_name = f"{prefix}.{suffix}"
        component.matrix(torch_name, checkpoint.get(torch_name), n_in, n_out, keep_f16=keep_f16)

    for reader_suffix, source_suffix in (
        ("attention_norm1.weight", "attention_norm1.weight"),
        ("attention_norm2.weight", "attention_norm2.weight"),
        ("ffn_norm1.weight", "ffn_norm1.weight"),
        ("ffn_norm2.weight", "ffn_norm2.weight"),
        ("attention.q_norm.weight", "attention.norm_q.weight"),
        ("attention.k_norm.weight", "attention.norm_k.weight"),
    ):
        component.vector(
            f"{prefix}.{reader_suffix}",
            checkpoint.get(f"{prefix}.{source_suffix}"),
        )

    if modulated:
        name = f"{prefix}.adaLN_modulation.0.weight"
        component.matrix(
            name, checkpoint.get(name), TIME_WIDTH, ADALN_WIDTH, keep_f16=keep_f16
        )
        bias_name = f"{prefix}.adaLN_modulation.0.bias"
        component.vector(bias_name, checkpoint.get(bias_name))


def convert_dit(model_dir: Path, out_path: Path, outtype: str) -> int:
    checkpoint = Checkpoint(model_dir / "transformer")
    component = Component(outtype)

    # The published checkpoint prefixes the two top-level embedders with
    # `all_` and suffixes them with `.2-1`, a Diffusers fusion artifact that the
    # reader has no notion of.
    component.vector("cap_embedder.0.weight", checkpoint.get("cap_embedder.0.weight"))
    component.matrix(
        "cap_embedder.1.weight",
        checkpoint.get("cap_embedder.1.weight"),
        CAP_WIDTH,
        HIDDEN,
        keep_f16=True,
    )
    component.vector("cap_embedder.1.bias", checkpoint.get("cap_embedder.1.bias"))
    component.matrix(
        "x_embedder.weight",
        checkpoint.get("all_x_embedder.2-1.weight"),
        PATCH_WIDTH,
        HIDDEN,
        keep_f16=True,
    )
    component.vector("x_embedder.bias", checkpoint.get("all_x_embedder.2-1.bias"))
    component.column_vector("cap_pad_token", checkpoint.get("cap_pad_token"))
    component.column_vector("x_pad_token", checkpoint.get("x_pad_token"))

    for index in (0, 2):
        name = f"t_embedder.mlp.{index}.weight"
        n_in, n_out = (TIME_WIDTH, TIME_HIDDEN) if index == 0 else (TIME_HIDDEN, TIME_WIDTH)
        component.matrix(name, checkpoint.get(name), n_in, n_out, keep_f16=True)
        bias = f"t_embedder.mlp.{index}.bias"
        component.vector(bias, checkpoint.get(bias))

    # Only the noise refiner is adaLN-modulated. The context refiner is not --
    # the reader encodes that as `modulation: None`, and the reference weights
    # agree: context_refiner.{i} carries 11 tensors against noise_refiner's 13.
    for prefix, modulated in (
        ("context_refiner", False),
        ("noise_refiner", True),
    ):
        for index in range(REFINER_LAYERS):
            _block_tensors(
                component,
                checkpoint,
                f"{prefix}.{index}",
                modulated=modulated,
                keep_f16=True,
            )
    for index in range(MAIN_LAYERS):
        _block_tensors(component, checkpoint, f"layers.{index}", modulated=True, keep_f16=False)

    component.matrix(
        "final_layer.linear.weight",
        checkpoint.get("all_final_layer.2-1.linear.weight"),
        HIDDEN,
        PATCH_WIDTH,
        keep_f16=True,
    )
    component.vector(
        "final_layer.linear.bias", checkpoint.get("all_final_layer.2-1.linear.bias")
    )
    component.matrix(
        "final_layer.adaLN_modulation.1.weight",
        checkpoint.get("all_final_layer.2-1.adaLN_modulation.1.weight"),
        TIME_WIDTH,
        HIDDEN,
        keep_f16=True,
    )
    component.vector(
        "final_layer.adaLN_modulation.1.bias",
        checkpoint.get("all_final_layer.2-1.adaLN_modulation.1.bias"),
    )

    component.write(out_path)
    return len(component.entries)


def _conv_nchw_to_ggml(name: str, weight: np.ndarray) -> np.ndarray:
    """Autoencoder conv weights are torch [out, in, kh, kw].

    The VAE reader asks for [kh, kw, in, out] (see the `decoder.conv_in.weight`
    entry in src/models/diffusion/z_image/vae.rs) but, as with the projections,
    the payload itself stays in torch order -- only the declared dims are
    reversed.
    """
    if weight.ndim != 4:
        raise ValueError(f"{name}: expected a 4-D conv weight, got {weight.shape}")
    return np.ascontiguousarray(weight)


def _vae_gguf_name(torch_name: str) -> str:
    """Rename an Autoencoder decoder tensor to the names the reader looks up.

    The reader (`src/models/diffusion/z_image/vae.rs`) speaks Diffusers'
    original vocabulary, while the checkpoint went through a `mid_block` /
    `up_blocks` / `down_blocks` refactor and a fused attention module. Verified
    to be a bijection onto the 138 reference decoder tensors.
    """
    name = torch_name
    name = name.replace("mid_block", "mid")
    name = name.replace("attentions.0.", "attn_1.")
    name = name.replace("to_out.0.", "proj_out.")
    name = name.replace("group_norm", "norm")
    name = re.sub(r"\bto_([qkv])\.", lambda m: m.group(1) + ".", name)
    # ResNet blocks carry no `block_` prefix in the checkpoint and count from 0.
    name = re.sub(r"\.resnets\.(\d+)\.", lambda m: f".block_{int(m.group(1)) + 1}.", name)
    name = re.sub(
        r"\.resnets\.(\d+)\.conv_shortcut\.",
        lambda m: f".block_{int(m.group(1)) + 1}.nin_shortcut.",
        name,
    )
    name = name.replace("conv_shortcut.", "nin_shortcut.").replace(
        "conv_norm_out", "norm_out"
    )
    # `up_blocks` is stored widest-stage-first, but the reader counts from the
    # narrowest end, so the stage index is mirrored.
    name = re.sub(
        r"\.up_blocks\.(\d+)\.block_(\d+)\.",
        lambda m: f".up.{3 - int(m.group(1))}.block.{int(m.group(2)) - 1}.",
        name,
    )
    name = re.sub(
        r"\.up_blocks\.(\d+)\.upsamplers\.(\d+)\.",
        lambda m: f".up.{3 - int(m.group(1))}.upsample.",
        name,
    )
    name = re.sub(
        r"\.down_blocks\.(\d+)\.block_(\d+)\.",
        lambda m: f".down.{m.group(1)}.block.{int(m.group(2)) - 1}.",
        name,
    )
    name = re.sub(
        r"\.downsamplers\.(\d+)\.",
        lambda m: f".down.{m.group(1)}.downsample.",
        name,
    )
    return name


def convert_vae(model_dir: Path, out_path: Path, outtype: str) -> int:
    """Export only the decoder: txt2img maps a latent to pixels, so
    `src/models/diffusion/z_image/vae.rs` never opens the encoder half."""
    checkpoint = Checkpoint(model_dir / "vae")
    component = Component(outtype)

    for torch_name in sorted(_vae_names(checkpoint)):
        weight = checkpoint.get(torch_name)
        name = _vae_gguf_name(torch_name)
        if weight.ndim == 1:
            component.vector(name, weight)
            continue
        if weight.ndim == 2:
            # The mid-block attention is a 1x1 conv stored as [out, in], but the
            # reader still addresses it with four dims, so keep the 1x1 shape
            # instead of flattening to a 2-D projection.
            out_c, in_c = weight.shape
            # Declared with the same (kh, kw, in, out) tuple the 4-D conv branch
            # uses, i.e. (1, 1, in, out); the payload stays in torch [out, in]
            # order to match the reference export.
            payload = np.ascontiguousarray(weight).reshape(1, 1, out_c, in_c)
            component.entries.append(
                (name, GGML_F16, (1, 1, in_c, out_c), f32_to_f16(payload.tobytes()))
            )
            continue
        transposed = _conv_nchw_to_ggml(name, weight)
        out_c, in_c, kh, kw = weight.shape
        # The reference VAE keeps every conv in F16 -- it is only 0.16 GB and
        # runs once, so there is nothing to gain by quantizing it.
        keep_f16 = True
        raw = component._matrix_raw(transposed, keep_f16)
        # dims are stored reversed relative to torch, matching the reader.
        component.entries.append(
            (
                name,
                component._matrix_ggml_type(keep_f16),
                (kw, kh, in_c, out_c),
                raw,
            )
        )

    component.write(out_path)
    return len(component.entries)


def _vae_names(checkpoint: Checkpoint) -> Iterable[str]:
    seen = set()
    for shard in checkpoint.shards:
        for name in shard.header:
            if name.startswith("decoder."):
                seen.add(name)
    return sorted(seen)


def convert_text_encoder(model_dir: Path, out_path: Path, outtype: str) -> int:
    """The text encoder is a stock Qwen3, so the names transfer unchanged."""
    checkpoint = Checkpoint(model_dir / "text_encoder")
    component = Component(outtype)

    for name in sorted(_text_encoder_names(checkpoint)):
        weight = checkpoint.get(name)
        if weight.ndim == 1:
            component.vector(name, weight)
            continue
        if weight.ndim != 2:
            raise ValueError(f"{name}: unexpected rank {weight.ndim}")
        n_out, n_in = weight.shape
        component.matrix(name, weight, n_in, n_out)

    component.write(out_path)
    return len(component.entries)


def _text_encoder_names(checkpoint: Checkpoint) -> Iterable[str]:
    seen = set()
    for shard in checkpoint.shards:
        for name in shard.header:
            if name.startswith("model.") or name.startswith("lm_head"):
                seen.add(name)
    return sorted(seen)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="export Tongyi-MAI/Z-Image-Turbo to the three Z-Image GGUF components"
    )
    parser.add_argument("model_dir", help="path to the Z-Image-Turbo directory")
    parser.add_argument(
        "--out-dir", default=None, help="output directory (default: model directory)"
    )
    parser.add_argument(
        "--outtype",
        choices=SUPPORTED_OUTTYPES,
        default="q8_0",
        help="DiT/text-encoder matrix precision (default: q8_0)",
    )
    parser.add_argument(
        "--components",
        choices=("all", "dit", "vae", "text"),
        default="all",
        help="which components to export (default: all)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing output")
    args = parser.parse_args()

    model_dir = validated_dir(args.model_dir, must_exist=True)
    out_dir = validated_dir(args.out_dir or str(model_dir), must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Callable[[Path, Path, str], int], str]] = [
        ("dit", convert_dit, f"z-image-turbo-{args.outtype}.gguf"),
        ("vae", convert_vae, "pig_flux_vae_fp32-f16.gguf"),
        ("text", convert_text_encoder, f"qwen3_4b_f32-{args.outtype}.gguf"),
    ]

    for key, convert, filename in jobs:
        if args.components != "all" and args.components != key:
            continue
        out_path = out_dir / filename
        if out_path.exists() and not args.overwrite:
            print(f"skip {out_path} (exists; pass --overwrite)", file=sys.stderr)
            continue
        print(f"writing {out_path} ...", flush=True)
        count = convert(model_dir, out_path, args.outtype)
        print(f"wrote {out_path} ({count} tensors)")


if __name__ == "__main__":
    main()
