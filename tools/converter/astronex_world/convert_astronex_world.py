#!/usr/bin/env python3
"""Export the Astronex-World Wan2.2 checkpoint as a lossless GGUF pair.

The exporter intentionally has no quantization or dtype conversion path.  The
released denoiser/text encoder are BF16 and the Wan VAE is F32; those payloads
are copied byte-for-byte into the GGUF files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import tempfile
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.converter.dots.convert_dots_tts import (  # noqa: E402
    GGML_BF16,
    GGML_F32,
    GgufWriter,
    gguf_dims,
    validated_dir,
)
from tools.converter.dreamx.convert_dreamx_creator import SourceTensor  # noqa: E402


ARCHITECTURE = "astronex_world"
EXPECTED_DENOISER_TENSORS = 898
EXPECTED_TEXT_TENSORS = 242
EXPECTED_VAE_TENSORS = 196
EXPECTED_DENOISER_DTYPE = "BF16"
EXPECTED_TEXT_DTYPE = "BF16"
EXPECTED_VAE_DTYPE = "F32"
GGUF_FILE_TYPE_BF16 = 32  # mostly BF16; GGML_BF16 (30) is the tensor type.
_DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2}
_TEXT_BLOCK = re.compile(r"^encoder\.block\.(\d+)\.layer\.(0|1)\.(.+)$")


@dataclass(frozen=True)
class AstronexInventory:
    root: Path
    config_path: Path
    transformer_config_path: Path
    text_config_path: Path
    vae_config_path: Path
    tokenizer_path: Path
    denoiser_index_path: Path
    denoiser_shards: tuple[Path, ...]
    text_index_path: Path
    text_shards: tuple[Path, ...]
    vae_path: Path


def _json_object(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON in {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    return _json_object(path.read_bytes(), path)


def _required(root: Path, relative: str) -> Path:
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"input path escapes model directory: {relative}") from error
    if not path.is_file():
        raise FileNotFoundError(f"missing required Astronex input: {path}")
    return path


def _first_required(root: Path, relatives: tuple[str, ...]) -> Path:
    for relative in relatives:
        path = root / relative
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing required Astronex input: {root / relatives[0]}")


def _index_shards(index_path: Path) -> tuple[dict[str, str], tuple[Path, ...]]:
    index = _load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path}: weight_map is missing or empty")
    if any(not isinstance(name, str) or not name for name in weight_map):
        raise ValueError(f"{index_path}: invalid tensor name in weight_map")
    if any(not isinstance(shard, str) or not shard for shard in weight_map.values()):
        raise ValueError(f"{index_path}: invalid shard name in weight_map")
    names = sorted(set(weight_map.values()))
    shards = tuple(_required(index_path.parent, name) for name in names)
    return dict(weight_map), shards


def build_inventory(model_dir: Path) -> AstronexInventory:
    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"missing Astronex model directory: {root}")
    return AstronexInventory(
        root=root,
        config_path=_required(root, "config.json"),
        transformer_config_path=_required(root, "transformer/config.json"),
        text_config_path=_required(root, "text_encoder/config.json"),
        vae_config_path=_required(root, "vae/config.json"),
        tokenizer_path=_first_required(root, ("tokenizer.json", "tokenizer/tokenizer.json")),
        denoiser_index_path=_required(root, "model.safetensors.index.json"),
        denoiser_shards=_index_shards(_required(root, "model.safetensors.index.json"))[1],
        text_index_path=_required(root, "text_encoder/model.safetensors.index.json"),
        text_shards=_index_shards(_required(root, "text_encoder/model.safetensors.index.json"))[1],
        vae_path=_required(root, "vae/diffusion_pytorch_model.safetensors"),
    )


def _file_range_chunks(path: Path, offset: int, length: int, chunk_bytes: int = 16 * 1024 * 1024) -> Iterable[bytes]:
    with path.open("rb") as handle:
        handle.seek(offset)
        remaining = length
        while remaining:
            chunk = handle.read(min(chunk_bytes, remaining))
            if not chunk:
                raise ValueError(f"truncated tensor payload in {path}")
            yield chunk
            remaining -= len(chunk)


def validate_safetensor_header(path: Path, header: dict[str, Any], raw_header_length: int) -> None:
    data_start = 8 + raw_header_length
    file_size = path.stat().st_size
    intervals: list[tuple[int, int, str]] = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise ValueError(f"{path}: invalid tensor entry {name}")
        try:
            dtype = info["dtype"]
            raw_shape = info["shape"]
            raw_offsets = info["data_offsets"]
            if not isinstance(dtype, str) or not isinstance(raw_shape, list) or not all(type(value) is int for value in raw_shape) or not isinstance(raw_offsets, list) or len(raw_offsets) != 2 or not all(type(value) is int for value in raw_offsets):
                raise ValueError
            shape = tuple(raw_shape)
            start, end = raw_offsets
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: invalid tensor entry {name}") from error
        if dtype not in _DTYPE_BYTES or not shape or any(value <= 0 for value in shape):
            raise ValueError(f"{path}: unsupported tensor {name}: {dtype!r} {shape}")
        expected = math.prod(shape) * _DTYPE_BYTES[dtype]
        if start < 0 or end < start or end - start != expected:
            raise ValueError(f"{path}: invalid data_offsets for {name}")
        absolute_start = data_start + start
        absolute_end = data_start + end
        if absolute_end > file_size:
            raise ValueError(f"{path}: tensor {name} lies outside file")
        intervals.append((absolute_start, absolute_end, name))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise ValueError(f"{path}: tensor offsets overlap ({previous[2]}, {current[2]})")


def read_safetensor_sources(component: str, paths: Iterable[Path]) -> list[SourceTensor]:
    sources: list[SourceTensor] = []
    names: set[str] = set()
    for path in paths:
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError(f"truncated safetensors header: {path}")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length > path.stat().st_size - 8:
                raise ValueError(f"invalid safetensors header length: {path}")
            raw_header = handle.read(header_length)
        header = _json_object(raw_header, path)
        validate_safetensor_header(path, header, header_length)
        data_start = 8 + header_length
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in names:
                raise ValueError(f"duplicate {component} tensor: {name}")
            names.add(name)
            dtype = info["dtype"]
            shape = tuple(info["shape"])
            start, end = info["data_offsets"]
            sources.append(SourceTensor(component, name, dtype, shape, lambda path=path, data_start=data_start, start=start, end=end: _file_range_chunks(path, data_start + start, end - start)))
    return sorted(sources, key=lambda source: source.name)


def _tensor_owners(paths: Iterable[Path], base: Path) -> dict[str, str]:
    owners: dict[str, str] = {}
    for path in paths:
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError(f"truncated safetensors header: {path}")
            length = struct.unpack("<Q", raw_length)[0]
            raw_header = handle.read(length)
        header = _json_object(raw_header, path)
        validate_safetensor_header(path, header, length)
        for name in header:
            if name == "__metadata__":
                continue
            if name in owners:
                raise ValueError(f"duplicate tensor across shards: {name}")
            owners[name] = path.relative_to(base).as_posix()
    return owners


def _safe_name(name: str) -> str:
    if not name or name.startswith(".") or ".." in name.split("."):
        raise ValueError(f"invalid tensor name: {name!r}")
    return name


def map_main_name(name: str) -> str:
    name = _safe_name(name)
    if not name.startswith("model.") or name == "model.":
        raise ValueError(f"denoiser tensor must start with model.: {name}")
    return "astronex." + name.removeprefix("model.")


def map_text_name(name: str) -> str:
    name = _safe_name(name)
    if name == "shared.weight":
        return "astronex.text.token_embedding.weight"
    if name == "encoder.final_layer_norm.weight":
        return "astronex.text.norm.weight"
    match = _TEXT_BLOCK.match(name)
    if match is None:
        raise ValueError(f"unsupported UMT5 tensor name: {name}")
    block, layer, suffix = int(match.group(1)), match.group(2), match.group(3)
    if layer == "0":
        mapping = {
            "SelfAttention.q.weight": "attn.q.weight",
            "SelfAttention.k.weight": "attn.k.weight",
            "SelfAttention.v.weight": "attn.v.weight",
            "SelfAttention.o.weight": "attn.o.weight",
            "SelfAttention.relative_attention_bias.weight": "pos_embedding.embedding.weight",
            "layer_norm.weight": "norm1.weight",
        }
    else:
        mapping = {
            "DenseReluDense.wi_0.weight": "ffn.gate.0.weight",
            "DenseReluDense.wi_1.weight": "ffn.fc1.weight",
            "DenseReluDense.wo.weight": "ffn.fc2.weight",
            "layer_norm.weight": "norm2.weight",
        }
    try:
        suffix = mapping[suffix]
    except KeyError as error:
        raise ValueError(f"unsupported UMT5 tensor name: {name}") from error
    return f"astronex.text.blocks.{block}.{suffix}"


def map_vae_name(name: str) -> str:
    name = _safe_name(name)
    if name.startswith("quant_conv."):
        return "astronex.vae.conv1." + name.removeprefix("quant_conv.")
    if name.startswith("post_quant_conv."):
        return "astronex.vae.conv2." + name.removeprefix("post_quant_conv.")
    match = re.match(r"^(encoder|decoder)\.(conv_in|conv_out)\.(weight|bias)$", name)
    if match:
        side, layer, leaf = match.groups()
        return f"astronex.vae.{side}.{('conv1' if layer == 'conv_in' else 'head.2')}.{leaf}"
    match = re.match(r"^(encoder|decoder)\.norm_out\.gamma$", name)
    if match:
        return f"astronex.vae.{match.group(1)}.head.0.gamma"
    match = re.match(r"^(encoder|decoder)\.mid_block\.resnets\.(0|1)\.(.+)$", name)
    if match:
        side, block, suffix = match.groups()
        middle = "0" if block == "0" else "2"
        return f"astronex.vae.{side}.middle.{middle}.residual.{_vae_residual_suffix(suffix)}"
    match = re.match(r"^(encoder|decoder)\.mid_block\.attentions\.0\.(norm\.gamma|to_qkv\.(?:weight|bias)|proj\.(?:weight|bias))$", name)
    if match:
        side, suffix = match.groups()
        return f"astronex.vae.{side}.middle.1.{suffix}"
    match = re.match(r"^(encoder)\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", name)
    if match:
        side, stage, block, suffix = match.groups()
        return f"astronex.vae.{side}.downsamples.{stage}.downsamples.{block}.residual.{_vae_residual_suffix(suffix)}"
    match = re.match(r"^(decoder)\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", name)
    if match:
        side, stage, block, suffix = match.groups()
        return f"astronex.vae.{side}.upsamples.{stage}.upsamples.{block}.residual.{_vae_residual_suffix(suffix)}"
    match = re.match(r"^(encoder)\.down_blocks\.(\d+)\.downsampler\.(resample\.1|time_conv)\.(weight|bias)$", name)
    if match:
        side, stage, layer, leaf = match.groups()
        return f"astronex.vae.{side}.downsamples.{stage}.downsamples.2.{layer}.{leaf}"
    match = re.match(r"^(decoder)\.up_blocks\.(\d+)\.upsampler\.(resample\.1|time_conv)\.(weight|bias)$", name)
    if match:
        side, stage, layer, leaf = match.groups()
        return f"astronex.vae.{side}.upsamples.{stage}.upsamples.3.{layer}.{leaf}"
    raise ValueError(f"unsupported Wan VAE tensor name: {name}")


def _vae_residual_suffix(suffix: str) -> str:
    mapping = {
        "norm1.gamma": "0.gamma",
        "conv1.weight": "2.weight",
        "conv1.bias": "2.bias",
        "norm2.gamma": "3.gamma",
        "conv2.weight": "6.weight",
        "conv2.bias": "6.bias",
        "conv_shortcut.weight": "shortcut.weight",
        "conv_shortcut.bias": "shortcut.bias",
    }
    try:
        return mapping[suffix]
    except KeyError as error:
        raise ValueError(f"unsupported Wan VAE residual tensor: {suffix}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest(inventory: AstronexInventory) -> dict[str, Any]:
    files = [
        inventory.config_path,
        inventory.transformer_config_path,
        inventory.text_config_path,
        inventory.vae_config_path,
        inventory.tokenizer_path,
        inventory.denoiser_index_path,
        *inventory.denoiser_shards,
        inventory.text_index_path,
        *inventory.text_shards,
        inventory.vae_path,
    ]
    return {
        "files": [
            {"path": path.relative_to(inventory.root).as_posix(), "size": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(set(files))
        ],
        "tensor_counts": {"denoiser": EXPECTED_DENOISER_TENSORS, "text": EXPECTED_TEXT_TENSORS, "vae": EXPECTED_VAE_TENSORS},
    }


def _contract_shapes(component: str) -> dict[str, tuple[int, ...]]:
    contract = Path(__file__).resolve().parents[3] / "references" / "astronex-world-contract"
    expected: dict[str, tuple[int, ...]] = {}
    if component == "denoiser":
        for path in sorted(contract.glob("model-*.safetensors.header.json")):
            header = _load_json(path)
            expected.update({name: tuple(int(value) for value in info["shape"]) for name, info in header.items() if name != "__metadata__"})
    elif component == "vae":
        header = _load_json(contract / "vae/diffusion_pytorch_model.safetensors.header.json")
        expected.update({name: tuple(int(value) for value in info["shape"]) for name, info in header.items() if name != "__metadata__"})
    elif component == "text":
        expected["shared.weight"] = (256384, 4096)
        expected["encoder.final_layer_norm.weight"] = (4096,)
        for block in range(24):
            prefix = f"encoder.block.{block}.layer."
            for suffix in ("SelfAttention.q.weight", "SelfAttention.k.weight", "SelfAttention.v.weight", "SelfAttention.o.weight"):
                expected[prefix + "0." + suffix] = (4096, 4096)
            expected[prefix + "0.SelfAttention.relative_attention_bias.weight"] = (32, 64)
            expected[prefix + "0.layer_norm.weight"] = (4096,)
            expected[prefix + "1.layer_norm.weight"] = (4096,)
            expected[prefix + "1.DenseReluDense.wi_0.weight"] = (10240, 4096)
            expected[prefix + "1.DenseReluDense.wi_1.weight"] = (10240, 4096)
            expected[prefix + "1.DenseReluDense.wo.weight"] = (4096, 10240)
    if not expected:
        raise FileNotFoundError(f"missing Astronex contract headers for {component}")
    return expected


def _validate_component_sources(component: str, sources: list[SourceTensor], expected_count: int, expected_dtype: str) -> None:
    if len(sources) != expected_count:
        raise ValueError(f"{component}: expected {expected_count} tensors, got {len(sources)}")
    expected = _contract_shapes(component)
    actual = {source.name: source for source in sources}
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:3]
        extra = sorted(set(actual) - set(expected))[:3]
        raise ValueError(f"{component}: tensor names differ (missing={missing}, extra={extra})")
    for name, source in actual.items():
        if source.dtype != expected_dtype or source.shape != expected[name]:
            raise ValueError(f"{component}.{name}: expected {expected_dtype} {expected[name]}, got {source.dtype} {source.shape}")


def collect_sources(inventory: AstronexInventory) -> tuple[list[SourceTensor], list[SourceTensor], list[SourceTensor]]:
    denoiser_map, _ = _index_shards(inventory.denoiser_index_path)
    denoiser = read_safetensor_sources("denoiser", inventory.denoiser_shards)
    denoiser_owners = _tensor_owners(inventory.denoiser_shards, inventory.denoiser_index_path.parent)
    if set(denoiser_map) != set(denoiser_owners) or set(denoiser_map) != {source.name for source in denoiser}:
        raise ValueError("denoiser index and shard tensor names differ")
    if any(denoiser_map[name] != owner for name, owner in denoiser_owners.items()):
        raise ValueError("denoiser index shard ownership differs from tensor headers")

    text_map, _ = _index_shards(inventory.text_index_path)
    text = read_safetensor_sources("text", inventory.text_shards)
    text_owners = _tensor_owners(inventory.text_shards, inventory.text_index_path.parent)
    if set(text_map) != set(text_owners) or set(text_map) != {source.name for source in text}:
        raise ValueError("text index and shard tensor names differ")
    if any(text_map[name] != owner for name, owner in text_owners.items()):
        raise ValueError("text index shard ownership differs from tensor headers")

    vae = read_safetensor_sources("vae", (inventory.vae_path,))
    _validate_component_sources("denoiser", denoiser, EXPECTED_DENOISER_TENSORS, EXPECTED_DENOISER_DTYPE)
    _validate_component_sources("text", text, EXPECTED_TEXT_TENSORS, EXPECTED_TEXT_DTYPE)
    _validate_component_sources("vae", vae, EXPECTED_VAE_TENSORS, EXPECTED_VAE_DTYPE)
    if any(not source.name.startswith("model.") for source in denoiser):
        raise ValueError("denoiser contains a tensor outside model.*")
    return denoiser, text, vae


def _config_contract(inventory: AstronexInventory) -> None:
    model = _load_json(inventory.config_path)
    transformer = _load_json(inventory.transformer_config_path)
    text = _load_json(inventory.text_config_path)
    vae = _load_json(inventory.vae_config_path)
    latent = model.get("latent", {})
    kwargs = model.get("model_kwargs", {})
    if not isinstance(latent, dict) or not isinstance(kwargs, dict):
        raise ValueError("Astronex config latent/model_kwargs must be objects")
    if (
        model.get("architecture") != "CausalWanModel"
        or model.get("backbone_variant") != "wan2.2"
        or model.get("dtype") != "bfloat16"
        or latent.get("channels") != 48
        or latent.get("compression") != [4, 16, 16]
        or kwargs.get("use_camera") is not True
        or kwargs.get("use_action") is not True
        or kwargs.get("action_dim") != 64
        or kwargs.get("action_output") is not False
        or kwargs.get("num_embodiments") != 32
        or kwargs.get("prope_num_layers") != 30
    ):
        raise ValueError("config is not Astronex CausalWanModel wan2.2")
    expected_transformer = {"in_dim": 48, "out_dim": 48, "dim": 3072, "ffn_dim": 14336, "num_heads": 24, "num_layers": 30, "text_dim": 4096, "text_len": 512}
    if any(transformer.get(key) != value for key, value in expected_transformer.items()) or transformer.get("patch_size") != [1, 2, 2] or transformer.get("qk_norm") is not True or transformer.get("cross_attn_norm") is not True:
        raise ValueError("transformer config does not match Wan2.2 TI2V-5B")
    expected_text = {"model_type": "umt5", "d_model": 4096, "d_ff": 10240, "num_heads": 64, "num_layers": 24, "vocab_size": 256384, "dense_act_fn": "gelu_new", "torch_dtype": "bfloat16"}
    if any(text.get(key) != value for key, value in expected_text.items()):
        raise ValueError("text config does not match UMT5-XXL")
    expected_vae = {"_class_name": "AutoencoderKLWan", "z_dim": 48, "patch_size": 2, "scale_factor_spatial": 16, "scale_factor_temporal": 4, "base_dim": 160, "decoder_base_dim": 256, "in_channels": 12, "out_channels": 12, "num_res_blocks": 2, "dim_mult": [1, 2, 4, 4], "temperal_downsample": [False, True, True], "is_residual": True, "attn_scales": [], "dropout": 0.0, "clip_output": False}
    try:
        latent_mean = [float(value) for value in vae.get("latents_mean", ())]
        latent_std = [float(value) for value in vae.get("latents_std", ())]
    except (TypeError, ValueError) as error:
        raise ValueError("VAE config latents_std must be numeric") from error
    if any(vae.get(key) != value for key, value in expected_vae.items()) or not isinstance(vae.get("latents_mean"), list) or not isinstance(vae.get("latents_std"), list) or len(latent_mean) != 48 or len(latent_std) != 48 or any(not math.isfinite(value) for value in latent_mean) or any(not math.isfinite(value) or value <= 0 for value in latent_std):
        raise ValueError("VAE config does not match AutoencoderKLWan")


def _metadata(inventory: AstronexInventory, manifest: dict[str, Any], pair: str) -> dict[str, list[tuple[str, Any]]]:
    common = [
        ("astronex.pair_id", pair),
        ("astronex.source_revision", "dec4ec502f7113a1ab07b78ae40a7ac1b88b2eeb"),
        ("astronex.oracle_commit", "27584bf1a89da01ef35a03b48d83bc818d2adf2b"),
        ("astronex.config.json", inventory.config_path.read_text()),
        ("astronex.transformer.config.json", inventory.transformer_config_path.read_text()),
        ("astronex.text.config.json", inventory.text_config_path.read_text()),
        ("astronex.vae.config.json", inventory.vae_config_path.read_text()),
        # Runtime loaders historically use the *_json spelling; retain both
        # spellings while the public contract uses dotted config names.
        ("astronex.text.config_json", inventory.text_config_path.read_text()),
        ("astronex.vae.config_json", inventory.vae_config_path.read_text()),
        ("astronex.tokenizer.json", inventory.tokenizer_path.read_text()),
        ("astronex.source_sha256_manifest", json.dumps(manifest, sort_keys=True, separators=(",", ":"))),
        ("astronex.source_manifest", json.dumps(manifest, sort_keys=True, separators=(",", ":"))),
        ("astronex.denoiser.tensor_count", EXPECTED_DENOISER_TENSORS),
        ("astronex.text.tensor_count", EXPECTED_TEXT_TENSORS),
        ("astronex.vae.tensor_count", EXPECTED_VAE_TENSORS),
    ]
    return {
        "main": [("general.architecture", ARCHITECTURE), ("general.name", "Astronex-World-BF16"), ("general.file_type", GGUF_FILE_TYPE_BF16), ("astronex.file_role", "main"), *common],
        "components": [("general.architecture", "clip"), ("general.name", "mmproj-Astronex-World"), ("general.file_type", GGUF_FILE_TYPE_BF16), ("general.type", "mmproj"), ("clip.projector_type", ARCHITECTURE), ("astronex.file_role", "components"), *common],
    }


def _write_pair_files(main_path: Path, components_path: Path, plan: Iterable[tuple[str, str, int, SourceTensor]], metadata: dict[str, list[tuple[str, Any]]]) -> None:
    writers = {"main": GgufWriter(main_path), "components": GgufWriter(components_path)}
    seen = {"main": set(), "components": set()}
    for target, name, ggml_type, source in plan:
        if target not in writers:
            raise ValueError(f"unknown output target {target}")
        if name in seen[target]:
            raise ValueError(f"duplicate output tensor {name}")
        seen[target].add(name)
        writers[target].add_tensor_chunks(name, ggml_type, gguf_dims(source.shape), source.nbytes, lambda source=source: source.chunks())
    for target, writer in writers.items():
        for key, value in metadata[target]:
            writer.add_meta(key, value)
        writer.write()


def output_paths(out_dir: Path) -> tuple[Path, Path]:
    return out_dir / "Astronex-World-BF16.gguf", out_dir / "mmproj-Astronex-World.gguf"


def _publish_pair(main_tmp: Path, components_tmp: Path, main_out: Path, components_out: Path, overwrite: bool) -> None:
    if not overwrite and (main_out.exists() or components_out.exists()):
        raise FileExistsError("Astronex output already exists")
    if not overwrite:
        os.link(main_tmp, main_out)
        try:
            os.link(components_tmp, components_out)
        except BaseException:
            main_out.unlink(missing_ok=True)
            raise
        return
    nonce = uuid.uuid4().hex
    backups = {path: path.with_name(f".{path.name}.{nonce}.backup") for path in (main_out, components_out) if path.exists()}
    moved: list[tuple[Path, Path]] = []
    try:
        for path, backup in backups.items():
            os.replace(path, backup)
            moved.append((path, backup))
        os.replace(main_tmp, main_out)
        os.replace(components_tmp, components_out)
    except BaseException:
        for path in (main_out, components_out):
            path.unlink(missing_ok=True)
        for path, backup in moved:
            if backup.exists():
                os.replace(backup, path)
        raise
    for backup in backups.values():
        backup.unlink(missing_ok=True)


def export_model(model_dir: Path, out_dir: Path, overwrite: bool = False) -> tuple[Path, Path]:
    inventory = build_inventory(model_dir)
    _config_contract(inventory)
    denoiser, text, vae = collect_sources(inventory)
    manifest = _source_manifest(inventory)
    manifest["component_counts"] = {"denoiser": len(denoiser), "text": len(text), "vae": len(vae)}
    pair = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    plan: list[tuple[str, str, int, SourceTensor]] = []
    for source in denoiser:
        plan.append(("main", map_main_name(source.name), GGML_BF16, source))
    for source in text:
        plan.append(("components", map_text_name(source.name), GGML_BF16, source))
    for source in vae:
        plan.append(("components", map_vae_name(source.name), GGML_F32, source))
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    main_out, components_out = output_paths(out_dir)
    with tempfile.TemporaryDirectory(prefix=".astronex-export-", dir=out_dir) as raw:
        temporary = Path(raw)
        metadata = _metadata(inventory, manifest, pair)
        _write_pair_files(temporary / main_out.name, temporary / components_out.name, sorted(plan, key=lambda entry: entry[1]), metadata)
        _publish_pair(temporary / main_out.name, temporary / components_out.name, main_out, components_out, overwrite)
    return main_out, components_out


def convert(model_dir: Path, out_dir: Path, overwrite: bool = False) -> tuple[Path, Path]:
    """Stable programmatic entry point used by converter tooling."""
    return export_model(model_dir, out_dir, overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(description="export Astronex-World Wan2.2 to lossless GGUF plus components")
    parser.add_argument("model_dir")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    model_dir = validated_dir(args.model_dir, must_exist=True)
    out_dir = validated_dir(args.out_dir or str(model_dir), must_exist=False)
    main_path, components_path = export_model(model_dir, out_dir, args.overwrite)
    print(f"wrote {main_path}")
    print(f"wrote {components_path}")


if __name__ == "__main__":
    main()
