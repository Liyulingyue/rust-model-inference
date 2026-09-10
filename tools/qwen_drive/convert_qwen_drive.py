#!/usr/bin/env python3
"""Export the released Qwen-Drive checkpoints without changing tensor precision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dots.convert_dots_tts import (
    GGML_BF16,
    GGML_F32,
    GgufWriter,
    _read_gguf,
    gguf_dims,
    read_gguf_directory,
)


COMPONENT_FILES = {
    "vlm": Path("model.safetensors"),
    "planner-sft": Path("planner-sft/model.safetensors"),
    "planner-rl": Path("planner-rl/model.safetensors"),
    "perception": Path("perception/model.safetensors"),
}
HEAD_PREFIXES = {
    "qwen_drive_planner": "planning_expert.",
    "qwen_drive_perception": "bev_modeling.",
}
DTYPE_BYTES = {"BF16": 2, "F32": 4}
GGML_TYPES = {"BF16": GGML_BF16, "F32": GGML_F32}
LLAMA_CPP_COMMIT = "b96806d96061049a5b574269b049bf6241d63d46"
OUTPUT_NAMES = (
    "Qwen-Drive-1.0-4B-BF16.gguf",
    "Qwen-Drive-1.0-4B-mmproj-BF16.gguf",
    "Qwen-Drive-1.0-planner-sft-BF16.gguf",
    "Qwen-Drive-1.0-planner-rl-BF16.gguf",
    "Qwen-Drive-1.0-perception-F32.gguf",
)


@dataclass(frozen=True)
class SourceTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_tensors(path: Path) -> list[SourceTensor]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{path}: truncated safetensors header")
        header_length = struct.unpack("<Q", raw_length)[0]
        raw_header = handle.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"{path}: truncated safetensors header")
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid safetensors header: {error}") from error
    data_start = 8 + header_length
    tensors = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        try:
            dtype = info["dtype"]
            shape = tuple(int(dim) for dim in info["shape"])
            start, end = (int(value) for value in info["data_offsets"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: invalid tensor entry {name!r}") from error
        element_size = DTYPE_BYTES.get(dtype)
        if element_size is None:
            raise ValueError(f"{path}: unsupported dtype {dtype} for {name}")
        if any(dim <= 0 for dim in shape) or start < 0 or end < start:
            raise ValueError(f"{path}: invalid shape or offsets for {name}")
        expected = math.prod(shape) * element_size
        if end - start != expected:
            raise ValueError(
                f"{path}: {name} payload is {end - start} bytes, expected {expected}"
            )
        absolute = data_start + start
        if data_start + end > file_size:
            raise ValueError(f"{path}: truncated tensor payload for {name}")
        tensors.append(SourceTensor(name, dtype, shape, absolute, expected))
    if not tensors:
        raise ValueError(f"{path}: no tensors")
    return tensors


def payload_chunks(
    path: Path,
    offset: int,
    nbytes: int,
    chunk_size: int = 1 << 20,
) -> Iterable[bytes]:
    with path.open("rb") as handle:
        handle.seek(offset)
        remaining = nbytes
        while remaining:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                raise ValueError(f"truncated tensor payload in {path}")
            remaining -= len(chunk)
            yield chunk


def _source_path(model_dir: Path, component: str, entry: dict) -> Path:
    relative = Path(entry["file"])
    direct = model_dir / relative
    if direct.exists():
        return direct
    nested = model_dir / component / relative
    return nested


def _manifest_tensors(entry: dict) -> dict[str, tuple[str, tuple[int, ...]]]:
    result = {}
    for item in entry.get("tensors", []):
        name = item["name"]
        if name in result:
            raise ValueError(f"duplicate manifest tensor {name}")
        result[name] = (item["dtype"], tuple(item["shape"]))
    return result


def _validate_component(path: Path, entry: dict) -> list[SourceTensor]:
    actual_size = path.stat().st_size
    if actual_size != entry["size"]:
        raise ValueError(f"{path}: size {actual_size}, expected {entry['size']}")
    actual_hash = sha256_file(path)
    if actual_hash != entry["sha256"]:
        raise ValueError(f"{path}: SHA256 {actual_hash}, expected {entry['sha256']}")
    tensors = source_tensors(path)
    expected = _manifest_tensors(entry)
    actual_names = {tensor.name for tensor in tensors}
    expected_names = set(expected)
    if extra := sorted(actual_names - expected_names):
        raise ValueError(f"unexpected tensor {extra[0]!r} in {path}")
    if missing := sorted(expected_names - actual_names):
        raise ValueError(f"missing tensor {missing[0]!r} in {path}")
    for tensor in tensors:
        dtype, shape = expected[tensor.name]
        if tensor.dtype != dtype:
            raise ValueError(f"{tensor.name}: dtype {tensor.dtype}, expected {dtype}")
        if tensor.shape != shape:
            raise ValueError(f"{tensor.name}: shape {tensor.shape}, expected {shape}")
    return tensors


def validate_source(model_dir: Path, manifest: dict) -> dict[str, list[SourceTensor]]:
    if manifest.get("version") != 1:
        raise ValueError("unsupported source manifest version")
    components = manifest.get("components")
    if not isinstance(components, dict) or not components:
        raise ValueError("source manifest has no components")
    return {
        name: _validate_component(_source_path(model_dir, name, entry), entry)
        for name, entry in components.items()
    }


def output_name(architecture: str, source_name: str) -> str:
    prefix = HEAD_PREFIXES.get(architecture)
    if prefix is None:
        raise ValueError(f"unsupported head architecture {architecture!r}")
    if not source_name.startswith(prefix):
        raise ValueError(f"unexpected tensor {source_name!r} for {architecture}")
    return f"{architecture}.{source_name.removeprefix(prefix)}"


def export_head(
    component_dir: Path,
    out_path: Path,
    architecture: str,
    manifest: dict,
    overwrite: bool = False,
) -> None:
    component = component_dir.name
    try:
        entry = manifest["components"][component]
    except KeyError as error:
        raise ValueError(f"manifest has no component {component!r}") from error
    path = _source_path(component_dir, component, entry)
    if not path.exists():
        path = _source_path(component_dir.parent, component, entry)
    tensors = _validate_component(path, entry)
    expected_dtype = "F32" if architecture == "qwen_drive_perception" else "BF16"
    writer = GgufWriter(out_path)
    writer.add_meta("general.architecture", architecture)
    writer.add_meta("general.name", component)
    writer.add_meta("general.file_type", 0 if expected_dtype == "F32" else 32)
    writer.add_meta(f"{architecture}.storage_dtype", expected_dtype.lower())
    writer.add_meta(
        f"{architecture}.compute_dtype",
        "bfloat16",
    )
    for tensor in tensors:
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"{tensor.name}: dtype {tensor.dtype}, expected {expected_dtype} for {architecture}"
            )
        writer.add_tensor_chunks(
            output_name(architecture, tensor.name),
            GGML_TYPES[tensor.dtype],
            gguf_dims(tensor.shape),
            tensor.nbytes,
            lambda tensor=tensor: payload_chunks(path, tensor.offset, tensor.nbytes),
        )
    writer.write(overwrite=overwrite)


def output_paths(out_dir: Path) -> tuple[Path, Path, Path, Path, Path]:
    return tuple(out_dir / name for name in OUTPUT_NAMES)  # type: ignore[return-value]


def _validate_llama_cpp(llama_cpp: Path) -> None:
    if not (llama_cpp / "convert_hf_to_gguf.py").is_file():
        raise FileNotFoundError(f"missing llama.cpp converter: {llama_cpp}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=llama_cpp, check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != LLAMA_CPP_COMMIT:
        raise ValueError(f"llama.cpp HEAD is {head}, expected {LLAMA_CPP_COMMIT}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=llama_cpp, check=True, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise ValueError(f"llama.cpp worktree is not clean: {llama_cpp}")


def _prepare_vlm_view(model_dir: Path, view: Path) -> None:
    config = json.loads((model_dir / "config.json").read_text())
    vlm_config = config.get("vlm_config")
    if not isinstance(vlm_config, dict):
        raise ValueError("config.json has no vlm_config")
    for source in model_dir.iterdir():
        if source.is_file() and source.name != "config.json":
            (view / source.name).symlink_to(source)
    (view / "config.json").write_text(json.dumps(vlm_config, indent=2) + "\n")


def _run_llama_converter(
    llama_cpp: Path, view: Path, output: Path, *, mmproj: bool
) -> None:
    bootstrap = r'''
import sys
from pathlib import Path

llama_cpp, model, output, mode = map(Path, sys.argv[1:])
sys.path.insert(0, str(llama_cpp / "gguf-py"))
sys.path.insert(0, str(llama_cpp))
from conversion.qwen import Qwen3_5TextModel
from conversion.qwen3vl import Qwen3VLVisionModel

text_filter = Qwen3_5TextModel.filter_tensors
vision_filter = Qwen3VLVisionModel.filter_tensors

@classmethod
def qwen_drive_text_filter(cls, item):
    name, generator = item
    return text_filter((name.removeprefix("vlm."), generator))

@classmethod
def qwen_drive_vision_filter(cls, item):
    name, generator = item
    return vision_filter((name.removeprefix("vlm."), generator))

Qwen3_5TextModel.filter_tensors = qwen_drive_text_filter
Qwen3VLVisionModel.filter_tensors = qwen_drive_vision_filter

import convert_hf_to_gguf
sys.argv = ["convert_hf_to_gguf.py", "--outfile", str(output), "--outtype", "bf16"]
if mode.name == "mmproj":
    sys.argv.append("--mmproj")
sys.argv.append(str(model))
convert_hf_to_gguf.main()
'''
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        "3.13",
        "--with",
        "numpy>=1.26.4,<3",
        "--with",
        "sentencepiece>=0.1.98,<0.3",
        "--with",
        "transformers==4.57.6",
        "--with",
        "protobuf>=4.21,<5",
        "--with",
        "torch>=2.6,<3",
        "python",
        "-c",
        bootstrap,
        str(llama_cpp),
        str(view),
        str(output),
        "mmproj" if mmproj else "text",
    ]
    try:
        subprocess.run(command, cwd=llama_cpp, check=True)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"llama.cpp conversion failed with exit {error.returncode}") from error


def _install_llama_output(
    llama_cpp: Path, view: Path, destination: Path, *, mmproj: bool, overwrite: bool
) -> None:
    fd, raw_temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary = Path(raw_temporary)
    temporary.unlink()
    try:
        _run_llama_converter(llama_cpp, view, temporary, mmproj=mmproj)
        metadata, tensors = read_gguf_directory(temporary)
        expected = "clip" if mmproj else "qwen35"
        if metadata.get("general.architecture") != expected or not tensors:
            raise ValueError(f"{temporary}: invalid {expected} GGUF readback")
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def _export_vlm_pair(
    model_dir: Path,
    llama_cpp: Path,
    vlm_path: Path,
    mmproj_path: Path,
    overwrite: bool,
) -> None:
    with tempfile.TemporaryDirectory(prefix="qwen-drive-vlm-") as raw_view:
        view = Path(raw_view)
        _prepare_vlm_view(model_dir, view)
        _install_llama_output(llama_cpp, view, vlm_path, mmproj=False, overwrite=overwrite)
        _install_llama_output(llama_cpp, view, mmproj_path, mmproj=True, overwrite=overwrite)


def export_model(
    model_dir: Path,
    llama_cpp: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    model_dir = model_dir.resolve()
    llama_cpp = llama_cpp.resolve()
    out_dir = out_dir.resolve()
    manifest = _load_manifest(Path(__file__).with_name("source-tensors.json"))
    validate_source(model_dir, manifest)
    _validate_llama_cpp(llama_cpp)
    paths = output_paths(out_dir)
    if not overwrite and (existing := next((path for path in paths if path.exists()), None)):
        raise FileExistsError(f"output already exists: {existing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"exporting {paths[0].name}")
    _export_vlm_pair(model_dir, llama_cpp, paths[0], paths[1], overwrite)
    print(f"exported {paths[1].name}")
    print(f"exporting {paths[2].name}")
    export_head(model_dir / "planner-sft", paths[2], "qwen_drive_planner", manifest, overwrite)
    print(f"exporting {paths[3].name}")
    export_head(model_dir / "planner-rl", paths[3], "qwen_drive_planner", manifest, overwrite)
    print(f"exporting {paths[4].name}")
    export_head(model_dir / "perception", paths[4], "qwen_drive_perception", manifest, overwrite)
    return paths


def _sha256_region(path: Path, offset: int, nbytes: int) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(offset)
        remaining = nbytes
        while remaining:
            chunk = handle.read(min(1 << 20, remaining))
            if not chunk:
                raise ValueError(f"{path}: truncated payload at offset {offset}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.digest()


def _verify_head_payloads(
    output: Path,
    source: Path,
    architecture: str,
    expected: list[SourceTensor],
) -> None:
    _metadata, directory = _read_gguf(output)
    expected_names = {output_name(architecture, tensor.name) for tensor in expected}
    if set(directory) != expected_names:
        raise ValueError(f"{output}: head tensor inventory mismatch")
    for tensor in expected:
        name = output_name(architecture, tensor.name)
        ggml_type, dims, nbytes, offset = directory[name]
        if (ggml_type, dims, nbytes) != (
            GGML_TYPES[tensor.dtype],
            gguf_dims(tensor.shape),
            tensor.nbytes,
        ):
            raise ValueError(f"{output}: tensor contract mismatch for {name}")
        if _sha256_region(source, tensor.offset, tensor.nbytes) != _sha256_region(
            output, offset, nbytes
        ):
            raise ValueError(f"{output}: payload mismatch for {name}")


def verify_outputs(
    paths: tuple[Path, ...],
    *,
    model_dir: Path | None = None,
    manifest: dict | None = None,
) -> dict[str, dict]:
    if len(paths) != len(OUTPUT_NAMES):
        raise ValueError(f"expected {len(OUTPUT_NAMES)} GGUF paths, got {len(paths)}")
    expected_architectures = (
        "qwen35",
        "clip",
        "qwen_drive_planner",
        "qwen_drive_planner",
        "qwen_drive_perception",
    )
    result = {}
    for path, name, architecture in zip(paths, OUTPUT_NAMES, expected_architectures):
        if path.name != name:
            raise ValueError(f"expected output name {name}, got {path.name}")
        metadata, tensors = read_gguf_directory(path)
        actual = metadata.get("general.architecture")
        if actual != architecture:
            raise ValueError(f"{path}: expected {architecture} architecture, got {actual}")
        if not tensors:
            raise ValueError(f"{path}: GGUF contains no tensors")
        if architecture == "clip" and metadata.get("clip.projector_type") != "qwen3vl_merger":
            raise ValueError(f"{path}: expected qwen3vl_merger projector")
        if architecture.startswith("qwen_drive_"):
            prefix = f"{architecture}."
            expected_type = GGML_F32 if architecture.endswith("perception") else GGML_BF16
            for tensor_name, (ggml_type, _dims, _nbytes) in tensors.items():
                if not tensor_name.startswith(prefix):
                    raise ValueError(f"{path}: unexpected tensor {tensor_name!r}")
                if ggml_type != expected_type:
                    raise ValueError(f"{path}: unexpected tensor type for {tensor_name}")
        counts: dict[str, int] = {}
        for ggml_type, _dims, _nbytes in tensors.values():
            key = str(ggml_type)
            counts[key] = counts.get(key, 0) + 1
        result[path.name] = {
            "architecture": architecture,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "tensors": len(tensors),
            "ggml_types": counts,
        }
    if model_dir is not None:
        model_dir = model_dir.resolve()
        manifest = manifest or _load_manifest(Path(__file__).with_name("source-tensors.json"))
        for index, component, architecture in (
            (2, "planner-sft", "qwen_drive_planner"),
            (3, "planner-rl", "qwen_drive_planner"),
            (4, "perception", "qwen_drive_perception"),
        ):
            entry = manifest["components"][component]
            source = _source_path(model_dir, component, entry)
            _verify_head_payloads(paths[index], source, architecture, source_tensors(source))
    return result


def inspect_model(model_dir: Path) -> dict:
    components = {}
    for name, relative in COMPONENT_FILES.items():
        path = model_dir / relative
        tensors = source_tensors(path)
        components[name] = {
            "file": str(relative),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "tensors": [
                {"name": tensor.name, "dtype": tensor.dtype, "shape": list(tensor.shape)}
                for tensor in tensors
            ],
        }
    return {"version": 1, "components": components}


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("model_dir", type=Path)
    inspect.add_argument("--write-manifest", required=True, type=Path)
    export = subparsers.add_parser("export")
    export.add_argument("model_dir", type=Path)
    export.add_argument("--llama-cpp", required=True, type=Path)
    export.add_argument("--out-dir", required=True, type=Path)
    export.add_argument("--overwrite", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("model_dir", type=Path)
    verify.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        manifest = inspect_model(args.model_dir)
        _write_json_atomic(args.write_manifest, manifest)
        for name, entry in manifest["components"].items():
            counts: dict[str, int] = {}
            for tensor in entry["tensors"]:
                counts[tensor["dtype"]] = counts.get(tensor["dtype"], 0) + 1
            print(f"{name}: {len(entry['tensors'])} tensors {counts} {entry['sha256']}")
        return 0
    if args.command == "export":
        paths = export_model(args.model_dir, args.llama_cpp, args.out_dir, args.overwrite)
        print(json.dumps(verify_outputs(paths, model_dir=args.model_dir), indent=2, sort_keys=True))
        return 0
    if args.command == "verify":
        validate_source(
            args.model_dir.resolve(),
            _load_manifest(Path(__file__).with_name("source-tensors.json")),
        )
        print(
            json.dumps(
                verify_outputs(output_paths(args.out_dir.resolve()), model_dir=args.model_dir),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
