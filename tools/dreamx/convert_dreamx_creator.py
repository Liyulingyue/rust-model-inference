#!/usr/bin/env python3
"""Export the released DreamX-Creator checkpoint as GGUF plus mmproj."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JOINT_LAYERS = tuple(range(15, 30))
SUPPORTED_OUTTYPES = {"bf16": "BF16", "q8_0": "Q8_0"}


@dataclass(frozen=True)
class DreamXInventory:
    root: Path
    video_dir: Path
    audio_model: Path
    joint: Path
    t5: Path
    video_vae: Path
    audio_vae: Path
    sr_dit: Path
    upsampler_flash: Path
    upsampler_causal2d: Path
    lightvae: Path
    tokenizer_json: Path
    video_config: dict[str, Any]
    audio_config: dict[str, Any]
    video_shards: tuple[Path, ...]


def output_paths(out_dir: Path, outtype: str) -> tuple[Path, Path]:
    try:
        precision = SUPPORTED_OUTTYPES[outtype.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported DreamX outtype: {outtype}") from error
    return (
        out_dir / f"DreamX-Creator-{precision}.gguf",
        out_dir / "mmproj-DreamX-Creator-BF16.gguf",
    )


def pair_id(manifest: dict[str, Any]) -> str:
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _require_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing required input path: {path}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require_config(
    component: str,
    config: dict[str, Any],
    expected: dict[str, int],
) -> None:
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"{component} {key}: expected {value}, got {config.get(key)!r}"
            )


def _read_safetensors_header(path: Path) -> tuple[bytes, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        length = struct.unpack("<Q", raw_length)[0]
        if length > path.stat().st_size - 8:
            raise ValueError(f"invalid safetensors header length: {path}")
        raw = handle.read(length)
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid safetensors header: {path}") from error
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors tensor directory: {path}")
    return raw, header


def _validate_joint_layers(path: Path) -> None:
    _raw, header = _read_safetensors_header(path)
    layers = {
        int(match.group(1))
        for name in header
        if name != "__metadata__"
        and (match := re.match(r"^joint_blocks\.(\d+)\.", name))
    }
    expected = set(JOINT_LAYERS)
    if layers != expected:
        raise ValueError(
            f"joint layers: expected {list(JOINT_LAYERS)}, got {sorted(layers)}"
        )


def _video_shards(video_dir: Path) -> tuple[Path, ...]:
    index_path = _require_file(
        video_dir,
        "diffusion_pytorch_model.safetensors.index.json",
    )
    index = _load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"missing weight_map in {index_path}")
    shard_names = sorted(set(weight_map.values()))
    if not all(isinstance(name, str) and name for name in shard_names):
        raise ValueError(f"invalid shard name in {index_path}")
    return tuple(_require_file(video_dir, name) for name in shard_names)


def build_inventory(model_dir: Path) -> DreamXInventory:
    root = model_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"missing DreamX model directory: {root}")

    video_dir = root / "creator/video_model"
    if not video_dir.is_dir():
        raise FileNotFoundError(f"missing required input path: {video_dir}")
    video_config = _load_json(_require_file(video_dir, "config.json"))
    audio_config_path = _require_file(root, "creator/audio_model/config.json")
    audio_config = _load_json(audio_config_path)
    _require_config(
        "video",
        video_config,
        {
            "dim": 3072,
            "ffn_dim": 14336,
            "num_heads": 24,
            "num_layers": 30,
            "in_dim": 48,
            "out_dim": 48,
            "text_len": 512,
        },
    )
    _require_config(
        "audio",
        audio_config,
        {
            "dim": 1536,
            "ffn_dim": 8960,
            "num_heads": 12,
            "num_layers": 30,
            "in_dim": 128,
            "out_dim": 128,
            "text_len": 512,
        },
    )

    joint = _require_file(root, "creator/cross_attn_weights.safetensors")
    _validate_joint_layers(joint)
    return DreamXInventory(
        root=root,
        video_dir=video_dir,
        audio_model=_require_file(
            root, "creator/audio_model/diffusion_pytorch_model.safetensors"
        ),
        joint=joint,
        t5=_require_file(
            root, "wan2.2_ti2v_5b/models_t5_umt5-xxl-enc-bf16.pth"
        ),
        video_vae=_require_file(root, "wan2.2_ti2v_5b/Wan2.2_VAE.pth"),
        audio_vae=_require_file(
            root, "audio_vae/diffusion_pytorch_model.safetensors"
        ),
        sr_dit=_require_file(root, "refiner/sr_dit_5b.pt"),
        upsampler_flash=_require_file(root, "refiner/latent_upsampler_flash.pt"),
        upsampler_causal2d=_require_file(
            root, "refiner/latent_upsampler_2d_causal.pt"
        ),
        lightvae=_require_file(root, "refiner/lightvae_nu_scheme3.pt"),
        tokenizer_json=_require_file(
            root, "wan2.2_ti2v_5b/google/umt5-xxl/tokenizer.json"
        ),
        video_config=video_config,
        audio_config=audio_config,
        video_shards=_video_shards(video_dir),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(inventory: DreamXInventory, path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(inventory.root).as_posix(),
        "size": path.stat().st_size,
    }
    if path.suffix == ".safetensors":
        raw, header = _read_safetensors_header(path)
        record["header_sha256"] = hashlib.sha256(raw).hexdigest()
        record["tensor_count"] = len(header) - int("__metadata__" in header)
    return record


def build_pair_manifest(inventory: DreamXInventory) -> dict[str, Any]:
    files = [
        inventory.audio_model,
        inventory.joint,
        inventory.t5,
        inventory.video_vae,
        inventory.audio_vae,
        inventory.sr_dit,
        inventory.upsampler_flash,
        inventory.upsampler_causal2d,
        inventory.lightvae,
        *inventory.video_shards,
    ]
    return {
        "exporter_version": 1,
        "video_config": inventory.video_config,
        "audio_config": inventory.audio_config,
        "joint_layers": list(JOINT_LAYERS),
        "tokenizer": {
            **_source_record(inventory, inventory.tokenizer_json),
            "sha256": _sha256_file(inventory.tokenizer_json),
        },
        "sources": sorted(
            (_source_record(inventory, path) for path in files),
            key=lambda record: record["path"],
        ),
    }


def load_pt_state_dict(path: Path, torch_module=None):
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as error:
            raise RuntimeError(
                "PyTorch is required only for exporting DreamX .pt checkpoints"
            ) from error
    state = torch_module.load(
        path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(state, dict):
        raise ValueError(f"{path}: expected checkpoint dictionary")
    state = state.get("model", state.get("ema", state))
    if not isinstance(state, dict):
        raise ValueError(f"{path}: expected model state dictionary")
    non_tensors = [
        name for name, value in state.items() if not torch_module.is_tensor(value)
    ]
    if non_tensors:
        raise ValueError(f"{path}: non-tensor checkpoint entries: {non_tensors[:5]}")
    return state
