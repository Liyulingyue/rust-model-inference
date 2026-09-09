#!/usr/bin/env python3
"""Export the released DreamX-Creator checkpoint as GGUF plus mmproj."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import struct
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR / "dots"))

import convert_dots_tts as _gguf  # noqa: E402
from convert_dots_tts import (  # noqa: E402
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_Q8_0,
    GgufWriter,
    gguf_dims,
    quantize_q8_0,
    validated_dir,
)


JOINT_LAYERS = tuple(range(15, 30))
SUPPORTED_OUTTYPES = {"bf16": "BF16", "q8_0": "Q8_0"}
COMPONENT_PREFIXES = {
    "creator.video": "dreamx.creator.video",
    "creator.audio": "dreamx.creator.audio",
    "creator.joint": "dreamx.creator.joint",
    "refiner.dit": "dreamx.refiner.dit",
    "text": "dreamx.text",
    "video_vae": "dreamx.video_vae",
    "audio_vae": "dreamx.audio_vae",
    "refiner.upsampler.flash": "dreamx.refiner.upsampler.flash",
    "refiner.upsampler.causal2d": "dreamx.refiner.upsampler.causal2d",
    "refiner.lightvae": "dreamx.refiner.lightvae",
}
MAIN_COMPONENTS = {
    "creator.video",
    "creator.audio",
    "creator.joint",
    "refiner.dit",
}
_DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2}


@dataclass(frozen=True)
class SourceTensor:
    component: str
    name: str
    dtype: str
    shape: tuple[int, ...]
    chunks: Callable[[], Iterable[bytes]]

    @property
    def nbytes(self) -> int:
        try:
            element_bytes = _DTYPE_BYTES[self.dtype]
        except KeyError as error:
            raise ValueError(
                f"{self.component}.{self.name}: unsupported dtype {self.dtype}"
            ) from error
        return math.prod(self.shape) * element_bytes


@dataclass(frozen=True)
class PlannedTensor:
    source: SourceTensor
    target: str
    output_name: str
    ggml_type: int
    gguf_shape: tuple[int, ...]
    nbytes: int
    chunks: Callable[[], Iterable[bytes]]


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


def map_name(component: str, source_name: str) -> str:
    try:
        prefix = COMPONENT_PREFIXES[component]
    except KeyError as error:
        raise ValueError(f"unsupported DreamX component: {component}") from error
    if not source_name or source_name.startswith(".") or ".." in source_name.split("."):
        raise ValueError(f"invalid tensor name for {component}: {source_name!r}")
    return f"{prefix}.{source_name}"


def should_quantize(output_name: str, shape: tuple[int, ...]) -> bool:
    if not any(
        output_name.startswith(f"{COMPONENT_PREFIXES[name]}.")
        for name in MAIN_COMPONENTS
    ):
        return False
    return (
        len(shape) == 2
        and output_name.endswith(".weight")
        and min(shape) >= 128
        and max(shape) >= 1024
        and shape[-1] % 32 == 0
    )


def f32_to_bf16_rne(raw: bytes) -> bytes:
    if len(raw) % 4:
        raise ValueError("F32 payload size is not divisible by 4")
    bits = np.frombuffer(raw, dtype="<u4")
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype("<u2").tobytes()


def _source_f32(raw: bytes, dtype: str) -> np.ndarray:
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4")
    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        return (bits << np.uint32(16)).view(np.float32)
    raise ValueError(f"cannot quantize source dtype {dtype}")


def _converted_chunks(
    source: SourceTensor,
    target_type: int,
) -> Iterable[bytes]:
    if target_type == GGML_BF16 and source.dtype == "F32":
        carry = bytearray()
        for chunk in source.chunks():
            carry.extend(chunk)
            size = len(carry) // 4 * 4
            if size:
                yield f32_to_bf16_rne(bytes(carry[:size]))
                del carry[:size]
        if carry:
            raise ValueError(f"{source.component}.{source.name}: incomplete F32 value")
        return

    if target_type == GGML_Q8_0:
        row_bytes = source.shape[-1] * _DTYPE_BYTES[source.dtype]
        carry = bytearray()
        for chunk in source.chunks():
            carry.extend(chunk)
            size = len(carry) // row_bytes * row_bytes
            if size:
                yield quantize_q8_0(_source_f32(bytes(carry[:size]), source.dtype))
                del carry[:size]
        if carry:
            raise ValueError(
                f"{source.component}.{source.name}: incomplete GGML row"
            )
        return

    yield from source.chunks()


def build_tensor_plan(
    sources: Iterable[SourceTensor],
    outtype: str,
) -> list[PlannedTensor]:
    if outtype.lower() not in SUPPORTED_OUTTYPES:
        raise ValueError(f"unsupported DreamX outtype: {outtype}")
    plan = []
    source_ids = set()
    output_names = set()
    for source in sources:
        source_id = (source.component, source.name)
        if source_id in source_ids:
            raise ValueError(f"duplicate source tensor: {source.component}.{source.name}")
        source_ids.add(source_id)
        output_name = map_name(source.component, source.name)
        if output_name in output_names:
            raise ValueError(f"duplicate output tensor: {output_name}")
        output_names.add(output_name)
        quantized = outtype.lower() == "q8_0" and should_quantize(
            output_name, source.shape
        )
        bf16_linear = outtype.lower() == "bf16" and source.dtype == "F32" and should_quantize(
            output_name, source.shape
        )
        if quantized:
            ggml_type = GGML_Q8_0
        elif bf16_linear:
            ggml_type = GGML_BF16
        else:
            ggml_type = {
                "F32": GGML_F32,
                "F16": GGML_F16,
                "BF16": GGML_BF16,
            }.get(source.dtype)
            if ggml_type is None:
                raise ValueError(
                    f"{source.component}.{source.name}: unsupported dtype {source.dtype}"
                )
        dims = gguf_dims(source.shape)
        nbytes = _gguf._tensor_nbytes(ggml_type, dims)
        plan.append(
            PlannedTensor(
                source=source,
                target="main" if source.component in MAIN_COMPONENTS else "mmproj",
                output_name=output_name,
                ggml_type=ggml_type,
                gguf_shape=dims,
                nbytes=nbytes,
                chunks=lambda source=source, ggml_type=ggml_type: _converted_chunks(
                    source, ggml_type
                ),
            )
        )
    return sorted(plan, key=lambda entry: entry.output_name)


def _publish_pair(
    main_tmp: Path,
    mmproj_tmp: Path,
    main_out: Path,
    mmproj_out: Path,
    overwrite: bool,
) -> None:
    if not overwrite:
        existing = next((path for path in (main_out, mmproj_out) if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"output already exists: {existing}")
        os.link(main_tmp, main_out)
        try:
            os.link(mmproj_tmp, mmproj_out)
        except BaseException:
            main_out.unlink(missing_ok=True)
            raise
        return

    nonce = uuid.uuid4().hex
    backups = {
        path: path.with_name(f".{path.name}.{nonce}.backup")
        for path in (main_out, mmproj_out)
        if path.exists()
    }
    moved_backups = []
    try:
        for path, backup in backups.items():
            os.replace(path, backup)
            moved_backups.append((path, backup))
    except BaseException:
        for path, backup in reversed(moved_backups):
            os.replace(backup, path)
        raise
    published = []
    try:
        os.replace(main_tmp, main_out)
        published.append(main_out)
        os.replace(mmproj_tmp, mmproj_out)
        published.append(mmproj_out)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        for path, backup in backups.items():
            if backup.exists():
                os.replace(backup, path)
        raise
    for backup in backups.values():
        backup.unlink(missing_ok=True)


def _write_pair_files(
    main_path: Path,
    mmproj_path: Path,
    plan: Iterable[PlannedTensor],
    metadata: dict[str, list[tuple[str, Any]]],
) -> None:
    writers = {
        "main": GgufWriter(main_path),
        "mmproj": GgufWriter(mmproj_path),
    }
    for target, writer in writers.items():
        for key, value in metadata[target]:
            writer.add_meta(key, value)
    for entry in plan:
        writers[entry.target].add_tensor_chunks(
            entry.output_name,
            entry.ggml_type,
            entry.gguf_shape,
            entry.nbytes,
            entry.chunks,
        )
    writers["main"].write()
    writers["mmproj"].write()


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


def _file_range_chunks(
    path: Path,
    offset: int,
    length: int,
    chunk_bytes: int,
) -> Iterable[bytes]:
    with path.open("rb") as handle:
        handle.seek(offset)
        remaining = length
        while remaining:
            chunk = handle.read(min(chunk_bytes, remaining))
            if not chunk:
                raise ValueError(f"truncated tensor payload in {path}")
            yield chunk
            remaining -= len(chunk)


def read_safetensor_sources(
    component: str,
    paths: Iterable[Path],
    *,
    chunk_bytes: int = 16 * 1024 * 1024,
) -> list[SourceTensor]:
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    sources = []
    names = set()
    for path in paths:
        raw_header, header = _read_safetensors_header(path)
        data_start = 8 + len(raw_header)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in names:
                raise ValueError(f"duplicate {component} tensor: {name}")
            names.add(name)
            try:
                dtype = info["dtype"]
                shape = tuple(int(value) for value in info["shape"])
                start, end = (int(value) for value in info["data_offsets"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid safetensors entry {path}: {name}") from error
            if dtype not in _DTYPE_BYTES or any(value <= 0 for value in shape):
                raise ValueError(f"unsupported tensor {component}.{name}: {dtype} {shape}")
            length = end - start
            expected = math.prod(shape) * _DTYPE_BYTES[dtype]
            if start < 0 or length != expected:
                raise ValueError(
                    f"{component}.{name}: expected {expected} source bytes, got {length}"
                )
            absolute = data_start + start
            if absolute + length > path.stat().st_size:
                raise ValueError(f"{component}.{name}: tensor lies outside {path}")
            sources.append(
                SourceTensor(
                    component=component,
                    name=name,
                    dtype=dtype,
                    shape=shape,
                    chunks=lambda path=path, absolute=absolute, length=length: _file_range_chunks(
                        path, absolute, length, chunk_bytes
                    ),
                )
            )
    return sorted(sources, key=lambda source: source.name)


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


def collect_source_tensors(
    inventory: DreamXInventory,
    spool_dir: Path,
    *,
    spool_checkpoint=None,
) -> list[SourceTensor]:
    spool_checkpoint = spool_checkpoint or _spool_pt_checkpoint
    sources = []
    sources.extend(
        read_safetensor_sources("creator.video", inventory.video_shards)
    )
    index = _load_json(
        inventory.video_dir / "diffusion_pytorch_model.safetensors.index.json"
    )["weight_map"]
    if {source.name for source in sources} != set(index):
        raise ValueError("creator.video sharded index does not match tensor directories")
    for component, path in (
        ("creator.audio", inventory.audio_model),
        ("creator.joint", inventory.joint),
        ("audio_vae", inventory.audio_vae),
    ):
        sources.extend(read_safetensor_sources(component, (path,)))

    for component, path in (
        ("text", inventory.t5),
        ("video_vae", inventory.video_vae),
        ("refiner.dit", inventory.sr_dit),
        ("refiner.upsampler.flash", inventory.upsampler_flash),
        ("refiner.upsampler.causal2d", inventory.upsampler_causal2d),
        ("refiner.lightvae", inventory.lightvae),
    ):
        spool_path = spool_dir / f"{component.replace('.', '_')}.safetensors"
        declared = spool_checkpoint(path, spool_path)
        component_sources = read_safetensor_sources(component, (spool_path,))
        if declared != len(component_sources):
            raise ValueError(
                f"{component}: spooled {declared} tensors, read {len(component_sources)}"
            )
        sources.extend(component_sources)
    return sources


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def build_pair_metadata(
    inventory: DreamXInventory,
    outtype: str,
    model_pair_id: str,
    component_counts: dict[str, int],
    manifest: dict[str, Any],
) -> dict[str, list[tuple[str, Any]]]:
    components = sorted(COMPONENT_PREFIXES)
    if sorted(component_counts) != components:
        raise ValueError("DreamX component counts do not match released inventory")
    common = [
        ("dreamx.pair_id", model_pair_id),
        ("dreamx.exporter_version", int(manifest.get("exporter_version", 1))),
        ("dreamx.source_model", "GD-ML/DreamX-Creator"),
        ("dreamx.components", components),
        ("dreamx.joint_layers", list(JOINT_LAYERS)),
        ("dreamx.source_manifest", _canonical_json(manifest)),
        ("dreamx.video.config", _canonical_json(inventory.video_config)),
        ("dreamx.audio.config", _canonical_json(inventory.audio_config)),
        ("dreamx.video.embedding_length", 3072),
        ("dreamx.video.feed_forward_length", 14336),
        ("dreamx.video.attention.head_count", 24),
        ("dreamx.video.block_count", 30),
        ("dreamx.video.in_channels", 48),
        ("dreamx.audio.embedding_length", 1536),
        ("dreamx.audio.feed_forward_length", 8960),
        ("dreamx.audio.attention.head_count", 12),
        ("dreamx.audio.block_count", 30),
        ("dreamx.audio.in_channels", 128),
        ("dreamx.text.context_length", 512),
        ("dreamx.text.embedding_length", 4096),
        ("dreamx.text.feed_forward_length", 10240),
        ("dreamx.text.attention.head_count", 64),
        ("dreamx.text.block_count", 24),
        ("dreamx.text.vocab_size", 256384),
        ("dreamx.tokenizer.sha256", _sha256_file(inventory.tokenizer_json)),
    ]
    common.extend(
        (f"dreamx.component.{component}.tensor_count", component_counts[component])
        for component in components
    )
    common.extend((f"dreamx.has_component.{component}", True) for component in components)
    main = [
        ("general.architecture", "dreamx"),
        ("general.name", f"DreamX-Creator-{SUPPORTED_OUTTYPES[outtype.lower()]}"),
        ("general.file_type", 7 if outtype.lower() == "q8_0" else 32),
        ("dreamx.file_role", "main"),
        ("dreamx.file_components", sorted(MAIN_COMPONENTS)),
        ("dreamx.outtype", outtype.lower()),
        *common,
    ]
    mmproj = [
        ("general.architecture", "clip"),
        ("general.type", "mmproj"),
        ("general.name", "mmproj-DreamX-Creator-BF16"),
        ("general.file_type", 32),
        ("clip.projector_type", "dreamx_creator"),
        ("dreamx.file_role", "mmproj"),
        (
            "dreamx.file_components",
            sorted(set(COMPONENT_PREFIXES) - MAIN_COMPONENTS),
        ),
        ("dreamx.outtype", "bf16"),
        ("dreamx.tokenizer.json", inventory.tokenizer_json.read_text()),
        *common,
    ]
    return {"main": main, "mmproj": mmproj}


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
    state = state.get("model", state.get("ema", state.get("generator", state)))
    if not isinstance(state, dict):
        raise ValueError(f"{path}: expected model state dictionary")
    non_tensors = [
        name for name, value in state.items() if not torch_module.is_tensor(value)
    ]
    if non_tensors:
        raise ValueError(f"{path}: non-tensor checkpoint entries: {non_tensors[:5]}")
    return state


def _spool_pt_checkpoint(
    source: Path,
    destination: Path,
    *,
    torch_module=None,
    save_file=None,
) -> int:
    if save_file is None:
        try:
            from safetensors.torch import save_file
        except ImportError as error:
            raise RuntimeError(
                "safetensors is required for bounded DreamX .pt export"
            ) from error
    state = load_pt_state_dict(source, torch_module)
    ordered = dict(sorted(state.items()))
    try:
        save_file(ordered, str(destination))
        return len(ordered)
    finally:
        del ordered
        del state
        gc.collect()


def export_model(
    model_dir: Path,
    out_dir: Path,
    outtype: str = "q8_0",
    overwrite: bool = False,
) -> tuple[Path, Path]:
    outtype = outtype.lower()
    if outtype not in SUPPORTED_OUTTYPES:
        raise ValueError(f"unsupported DreamX outtype: {outtype}")
    inventory = build_inventory(Path(model_dir))
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    main_out, mmproj_out = output_paths(out_dir, outtype)
    if not overwrite:
        existing = next(
            (path for path in (main_out, mmproj_out) if path.exists()),
            None,
        )
        if existing is not None:
            raise FileExistsError(f"output already exists: {existing}")

    with tempfile.TemporaryDirectory(prefix=".dreamx-export-", dir=out_dir) as raw:
        temporary = Path(raw)
        sources = collect_source_tensors(inventory, temporary)
        plan = build_tensor_plan(sources, outtype)
        component_counts = {
            component: sum(source.component == component for source in sources)
            for component in COMPONENT_PREFIXES
        }
        manifest = build_pair_manifest(inventory)
        manifest["component_tensor_counts"] = component_counts
        model_pair_id = pair_id(manifest)
        metadata = build_pair_metadata(
            inventory,
            outtype,
            model_pair_id,
            component_counts,
            manifest,
        )
        main_tmp = temporary / main_out.name
        mmproj_tmp = temporary / mmproj_out.name
        _write_pair_files(main_tmp, mmproj_tmp, plan, metadata)
        _publish_pair(
            main_tmp,
            mmproj_tmp,
            main_out,
            mmproj_out,
            overwrite,
        )
    return main_out, mmproj_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="export GD-ML/DreamX-Creator to a matched GGUF/mmproj pair"
    )
    parser.add_argument("model_dir", help="path to the DreamX-Creator model directory")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: model directory)",
    )
    parser.add_argument(
        "--outtype",
        choices=sorted(SUPPORTED_OUTTYPES),
        default="q8_0",
        help="main GGUF precision (default: q8_0)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing matched pair",
    )
    args = parser.parse_args()
    model_dir = validated_dir(args.model_dir, must_exist=True)
    out_dir = validated_dir(args.out_dir or str(model_dir), must_exist=False)
    main_path, mmproj_path = export_model(
        model_dir,
        out_dir,
        args.outtype,
        args.overwrite,
    )
    print(f"wrote {main_path}")
    print(f"wrote {mmproj_path}")


if __name__ == "__main__":
    main()
