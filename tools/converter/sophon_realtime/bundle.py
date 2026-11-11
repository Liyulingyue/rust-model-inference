"""Bundle converted flow, HiFT and the six pinned voice prompts for --mmproj."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .code2wav import CONFIGS
from .convert import REVISION
from tools.converter.utils.gguf import GgufWriter, _read_gguf, model_metadata_value


def chunks(path, offset, size):
    with Path(path).open("rb") as source:
        source.seek(offset)
        while size:
            block = source.read(min(size, 1 << 20))
            if not block:
                raise ValueError("truncated component payload")
            size -= len(block)
            yield block


def voice_tensors(state):
    if set(state) != {"speech_token", "speech_feat", "embedding", "hop"}:
        raise ValueError("voice prompt inventory mismatch")
    tokens, mel, speaker = (state[k] for k in ("speech_token", "speech_feat", "embedding"))
    if (
        tokens.dtype != torch.int32 or tokens.ndim != 2 or tokens.shape[0] != 1
        or not 0 < tokens.shape[1] <= 7500
        or mel.dtype != torch.float32 or tuple(mel.shape) != (1, tokens.shape[1] * 2, 80)
        or speaker.dtype != torch.float32 or tuple(speaker.shape) != (1, 192)
        or type(state["hop"]) is not int or state["hop"] != 10
        or torch.any(tokens < 0) or torch.any(tokens >= 6561)
        or not torch.isfinite(mel).all() or not torch.isfinite(speaker).all()
        or speaker.norm().item() == 0
    ):
        raise ValueError("invalid voice prompt shapes, dtypes or values")
    return {
        "tokens": tokens[0].numpy().astype("<f4"),
        "mel": mel[0].T.contiguous().numpy().astype("<f4", copy=False),
        "speaker": speaker[0].numpy().astype("<f4", copy=False),
    }


def convert_bundle(model_dir, flow, hift, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    writer = GgufWriter(output)
    writer.add_meta("general.architecture", "sophon_realtime_code2wav")
    writer.add_meta("general.name", "Sophon-Realtime-32B-Tag code2wav and voice prompts")
    writer.add_meta("sophon_realtime.component", "code2wav")
    writer.add_meta("sophon_realtime.source_revision", REVISION)
    for component, path in (("flow", Path(flow)), ("hift", Path(hift))):
        metadata, directory = _read_gguf(path)
        if (
            metadata.get("general.architecture") != "sophon_realtime_" + component
            or metadata.get("sophon_realtime.component") != component
            or metadata.get("sophon_realtime.source_revision") != REVISION
            or json.loads(metadata.get(f"sophon_realtime.{component}_config_json", "null")) != CONFIGS[component]
        ):
            raise ValueError(f"not the pinned {component} component")
        for key, value in metadata.items():
            writer.add_meta(component + "." + key, model_metadata_value(value))
        for name, (kind, dims, length, offset) in directory.items():
            writer.add_tensor_chunks(
                component + "." + name, kind, dims, length,
                lambda path=path, offset=offset, length=length: chunks(path, offset, length),
            )
    manifest = json.loads((Path(__file__).parents[2] / "oracle/sophon_realtime/modelscope-files.json").read_text())
    voices = [item for item in manifest["files"] if item["path"].startswith("voices/") and item["path"].endswith(".pt")]
    names = []
    for item in voices:
        path = Path(model_dir) / item["path"]
        raw = path.read_bytes()
        if len(raw) != item["size"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"voice source pin mismatch: {path.name}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        tensors = voice_tensors(state)
        name = path.stem
        names.append(name)
        writer.add_meta(f"voices.{name}.hop", 10)
        writer.add_meta(f"voices.{name}.source_sha256", item["sha256"])
        for suffix, values in tensors.items():
            writer.add_tensor(f"voices.{name}.{suffix}", 0, tuple(reversed(values.shape)), values.tobytes())
    writer.add_meta("sophon_realtime.voices_json", json.dumps(names))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer.write()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("flow", type=Path)
    parser.add_argument("hift", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert_bundle(args.model_dir, args.flow, args.hift, args.output)


if __name__ == "__main__":
    main()
