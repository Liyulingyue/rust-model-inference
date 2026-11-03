"""Convert the Qwen3-VL vision tower of a Mage-Flow text encoder to an F32 mmproj.

Mage-Flow edits a reference image through a CLIP-style vision tower that this
engine reads from an mmproj GGUF.  The released tower is BF16 and the engine
requires lossless F32 tensors there, so the conversion widens the weights
exactly instead of re-quantising them.

The two contracts that have to line up:

* the patch embedding is a Conv3d over (channel, time, height, width) with a
  temporal patch size of two, while the engine projects one frame at a time
  with a matrix of 3 * patch_size * patch_size inputs, so the temporal axis
  splits into the two matrices ``v.patch_embd.weight`` and
  ``v.patch_embd.weight.1``;
* the merger and the deepstack mergers fold four patches together, so their
  first projection takes 4 * hidden inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from converter.utils.gguf import GGML_F32, GgufWriter, gguf_dims, open_safetensors, validated_dir

VISION = "model.visual."
LAYERS = 24
DEEPSTACK_INDEXES = (5, 11, 17)
EPS = 1e-6


def _bf16_to_f32(raw: bytes) -> bytes:
    """Widen BF16 to F32; the mapping is exact, so nothing is lost."""
    values = np.frombuffer(raw, dtype="<u2")
    if values.size == 0:
        return b""
    widened = values.astype(np.uint32) << np.uint32(16)
    return widened.view("<f4").tobytes()


def _f32(raw: bytes) -> bytes:
    return raw


def _shards(root: Path) -> list[Path]:
    index = root / "model.safetensors.index.json"
    if index.is_file():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [root / name for name in names]
    single = root / "model.safetensors"
    if single.is_file():
        return [single]
    raise FileNotFoundError(f"no safetensors under {root}")


class Reader:
    """Reads tensors across the shards of one text encoder."""

    def __init__(self, root: Path) -> None:
        self.files = [open_safetensors(path) for path in _shards(root)]
        self.owners: dict[str, int] = {}
        for index, source in enumerate(self.files):
            for name in source.header:
                if name != "__metadata__":
                    self.owners[name] = index

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.files[self.owners[name]].header[name]["shape"])

    def raw(self, name: str) -> bytes:
        return self.files[self.owners[name]].get(name).raw

    def dtype(self, name: str) -> str:
        return self.files[self.owners[name]].header[name]["dtype"]

    def f32(self, name: str) -> np.ndarray:
        raw = self.raw(name)
        dtype = self.dtype(name)
        if dtype == "F32":
            values = np.frombuffer(raw, dtype="<f4")
        elif dtype == "BF16":
            values = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            values = (values << np.uint32(16)).view("<f4")
        else:
            raise ValueError(f"{name}: unsupported dtype {dtype}")
        return values.reshape(self.shape(name))

    def has(self, name: str) -> bool:
        return name in self.owners


def _matrix(reader: Reader, name: str, out_dim: int, in_dim: int) -> bytes:
    values = reader.f32(name)
    if values.shape != (out_dim, in_dim):
        raise ValueError(f"{name}: shape {values.shape} != {(out_dim, in_dim)}")
    # GGUF keeps dims reversed against torch, so the rows stay as they are.
    return np.ascontiguousarray(values).tobytes()


def _vector(reader: Reader, name: str, length: int) -> bytes:
    values = reader.f32(name)
    if values.shape != (length,):
        raise ValueError(f"{name}: shape {values.shape} != {(length,)}")
    return np.ascontiguousarray(values).tobytes()


def _patch_embeddings(reader: Reader) -> list[tuple[str, tuple[int, int], bytes]]:
    """Split the temporal axis of the Conv3d patch embedding into two frames."""
    name = f"{VISION}patch_embed.proj.weight"
    values = reader.f32(name)
    if values.ndim != 5:
        raise ValueError(f"{name}: expected a 5D conv3d weight, got {values.shape}")
    out_dim, channels, frames, height, width = values.shape
    if frames != 2:
        raise ValueError(f"{name}: expected two temporal patches, got {frames}")
    patch_dim = channels * height * width
    out = []
    for frame in range(frames):
        matrix = np.ascontiguousarray(values[:, :, frame, :, :].reshape(out_dim, patch_dim))
        suffix = "" if frame == 0 else ".1"
        # The dims stay in torch order; gguf_dims reverses them on the way out.
        out.append((f"v.patch_embd.weight{suffix}", (out_dim, patch_dim), matrix.tobytes()))
    return out


def convert(text_encoder: Path, out_dir: Path) -> Path:
    root = validated_dir(str(text_encoder), must_exist=True)
    config = json.loads((root / "config.json").read_text())
    vision = config["vision_config"]
    if vision.get("model_type") != "qwen3_vl":
        raise ValueError(f"expected a Qwen3-VL vision tower, got {vision.get('model_type')}")
    if vision["deepstack_visual_indexes"] != list(DEEPSTACK_INDEXES):
        raise ValueError(
            f"unexpected deepstack indexes {vision['deepstack_visual_indexes']}"
        )
    hidden = vision["hidden_size"]
    inner = vision["intermediate_size"]
    heads = vision["num_heads"]
    patch = vision["patch_size"]
    merge = vision["spatial_merge_size"]
    positions = vision["num_position_embeddings"]
    out_dim = vision["out_hidden_size"]
    # The merger folds the four patches of a cell into one vector before it
    # projects, so its first layer takes 4 * hidden inputs.
    merged = hidden * merge * merge

    reader = Reader(root)
    preprocessor = json.loads((root / "preprocessor_config.json").read_text())
    mean = preprocessor.get("image_mean", [0.5, 0.5, 0.5])
    std = preprocessor.get("image_std", [0.5, 0.5, 0.5])
    size = preprocessor.get("size", {})
    min_pixels = size.get("shortest_edge", 65536)
    max_pixels = size.get("longest_edge", 16777216)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "mmproj-mage-flow-F32.gguf"
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    handle, temporary = tempfile.mkstemp(prefix=f".{out.name}.", dir=out_dir)
    os.close(handle)
    temporary = Path(temporary)
    writer = GgufWriter(temporary)
    writer.add_meta("general.architecture", "clip")
    writer.add_meta("general.name", "mmproj-mage-flow")
    writer.add_meta("general.file_type", 0)
    writer.add_meta("general.type", "mmproj")
    writer.add_meta("clip.vision.image_size", int(round(positions**0.5)) * patch)
    writer.add_meta("clip.vision.patch_size", patch)
    writer.add_meta("clip.vision.embedding_length", hidden)
    writer.add_meta("clip.vision.feed_forward_length", inner)
    writer.add_meta("clip.vision.block_count", LAYERS)
    writer.add_meta("clip.vision.attention.head_count", heads)
    writer.add_meta("clip.vision.attention.layer_norm_epsilon", EPS)
    writer.add_meta("clip.vision.spatial_merge_size", merge)
    writer.add_meta("clip.vision.projection_dim", out_dim)
    writer.add_meta("clip.vision.image_min_pixels", int(min_pixels))
    writer.add_meta("clip.vision.image_max_pixels", int(max_pixels))
    writer.add_meta("clip.vision.image_mean", [float(value) for value in mean])
    writer.add_meta("clip.vision.image_std", [float(value) for value in std])
    writer.add_meta("clip.use_gelu", True)
    writer.add_meta(
        "clip.vision.is_deepstack_layers",
        [index in DEEPSTACK_INDEXES for index in range(LAYERS)],
    )
    writer.add_meta("mage_flow.mmproj.source", "Qwen3-VL vision tower")
    writer.add_meta("mage_flow.mmproj.dtype", "F32")

    written: set[str] = set()

    def put(name: str, dims: tuple[int, ...], raw: bytes) -> None:
        if name in written:
            raise ValueError(f"duplicate tensor {name}")
        written.add(name)
        writer.add_tensor(name, GGML_F32, gguf_dims(dims), raw)

    for name, dims, raw in _patch_embeddings(reader):
        put(name, dims, raw)
    put("v.patch_embd.bias", (hidden,), _vector(reader, f"{VISION}patch_embed.proj.bias", hidden))
    put(
        "v.position_embd.weight",
        (positions, hidden),
        _matrix(reader, f"{VISION}pos_embed.weight", positions, hidden),
    )
    for layer in range(LAYERS):
        block = f"{VISION}blocks.{layer}."
        put(
            f"v.blk.{layer}.ln1.weight",
            (hidden,),
            _vector(reader, f"{block}norm1.weight", hidden),
        )
        put(
            f"v.blk.{layer}.ln1.bias",
            (hidden,),
            _vector(reader, f"{block}norm1.bias", hidden),
        )
        put(
            f"v.blk.{layer}.ln2.weight",
            (hidden,),
            _vector(reader, f"{block}norm2.weight", hidden),
        )
        put(
            f"v.blk.{layer}.ln2.bias",
            (hidden,),
            _vector(reader, f"{block}norm2.bias", hidden),
        )
        put(
            f"v.blk.{layer}.attn_qkv.weight",
            (3 * hidden, hidden),
            _matrix(reader, f"{block}attn.qkv.weight", 3 * hidden, hidden),
        )
        put(
            f"v.blk.{layer}.attn_qkv.bias",
            (3 * hidden,),
            _vector(reader, f"{block}attn.qkv.bias", 3 * hidden),
        )
        put(
            f"v.blk.{layer}.attn_out.weight",
            (hidden, hidden),
            _matrix(reader, f"{block}attn.proj.weight", hidden, hidden),
        )
        put(
            f"v.blk.{layer}.attn_out.bias",
            (hidden,),
            _vector(reader, f"{block}attn.proj.bias", hidden),
        )
        put(
            f"v.blk.{layer}.ffn_up.weight",
            (inner, hidden),
            _matrix(reader, f"{block}mlp.linear_fc1.weight", inner, hidden),
        )
        put(
            f"v.blk.{layer}.ffn_up.bias",
            (inner,),
            _vector(reader, f"{block}mlp.linear_fc1.bias", inner),
        )
        put(
            f"v.blk.{layer}.ffn_down.weight",
            (hidden, inner),
            _matrix(reader, f"{block}mlp.linear_fc2.weight", hidden, inner),
        )
        put(
            f"v.blk.{layer}.ffn_down.bias",
            (hidden,),
            _vector(reader, f"{block}mlp.linear_fc2.bias", hidden),
        )
    put(
        "mm.0.weight",
        (merged, merged),
        _matrix(reader, f"{VISION}merger.linear_fc1.weight", merged, merged),
    )
    put(
        "mm.0.bias",
        (merged,),
        _vector(reader, f"{VISION}merger.linear_fc1.bias", merged),
    )
    put(
        "mm.2.weight",
        (out_dim, merged),
        _matrix(reader, f"{VISION}merger.linear_fc2.weight", out_dim, merged),
    )
    put(
        "mm.2.bias",
        (out_dim,),
        _vector(reader, f"{VISION}merger.linear_fc2.bias", out_dim),
    )
    for index, layer in enumerate(DEEPSTACK_INDEXES):
        source = f"{VISION}deepstack_merger_list.{index}."
        # The engine keys the deepstack weights by the layer they come from.
        put(
            f"v.deepstack.{layer}.norm.weight",
            (merged,),
            _vector(reader, f"{source}norm.weight", merged),
        )
        put(
            f"v.deepstack.{layer}.norm.bias",
            (merged,),
            _vector(reader, f"{source}norm.bias", merged),
        )
        put(
            f"v.deepstack.{layer}.fc1.weight",
            (merged, merged),
            _matrix(reader, f"{source}linear_fc1.weight", merged, merged),
        )
        put(
            f"v.deepstack.{layer}.fc1.bias",
            (merged,),
            _vector(reader, f"{source}linear_fc1.bias", merged),
        )
        put(
            f"v.deepstack.{layer}.fc2.weight",
            (out_dim, merged),
            _matrix(reader, f"{source}linear_fc2.weight", out_dim, merged),
        )
        put(
            f"v.deepstack.{layer}.fc2.bias",
            (out_dim,),
            _vector(reader, f"{source}linear_fc2.bias", out_dim),
        )
    writer.add_meta("mage_flow.mmproj.tensor_count", len(written))
    try:
        writer.write()
        os.link(temporary, out)
    finally:
        temporary.unlink(missing_ok=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text_encoder", type=Path, help="Mage-Flow text_encoder directory")
    parser.add_argument("--out-dir", type=Path, default=Path("models/mage-flow"))
    args = parser.parse_args()
    out = convert(args.text_encoder, args.out_dir)
    print(out)


if __name__ == "__main__":
    main()
