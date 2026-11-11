"""Export the pinned Sophon portable ASR encoder and Ultravox projector."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import tempfile

from tools.converter.sophon_realtime.convert import (
    FILE_TYPES,
    FORMATS,
    K_WORKERS,
    REVISION,
    Reader,
    add_streamed,
    common_metadata,
    target_type,
    validate_config,
)
from tools.converter.utils.gguf import GgufWriter, _read_gguf

PINNED_AUDIO = dict(
    d_model=896,
    downsample_hidden_size=480,
    encoder_attention_heads=14,
    encoder_ffn_dim=3584,
    encoder_layers=18,
    num_mel_bins=128,
    n_window=50,
    n_window_infer=800,
    activation_function="gelu",
    model_type="qwen3_asr_encoder_portable",
    scale_embedding=False,
)
PINNED_PREPROCESSOR = dict(
    feature_size=128,
    hop_length=160,
    n_fft=400,
    sampling_rate=16000,
    chunk_length=30,
    n_samples=480000,
    nb_max_frames=3000,
    padding_side="right",
    padding_value=0.0,
    dither=0.0,
    return_attention_mask=True,
    feature_extractor_type="WhisperFeatureExtractor",
)


def audio_specs(config):
    a = config["audio_config"]
    h = a["d_model"]
    c = a["downsample_hidden_size"]
    f = a["encoder_ffn_dim"]
    for i in range(1, 4):
        yield f"audio_tower.conv2d{i}.weight", f"audio_tower.conv2d{i}.weight", (
            c,
            1 if i == 1 else c,
            3,
            3,
        )
        yield f"audio_tower.conv2d{i}.bias", f"audio_tower.conv2d{i}.bias", (c,)
    yield "audio_tower.conv_out.weight", "audio_tower.conv_out.weight", (
        h,
        c * ((a["num_mel_bins"] + 7) // 8),
    )
    for i in range(a["encoder_layers"]):
        for name, shape in [
            ("fc1", (f, h)),
            ("fc2", (h, f)),
            ("final_layer_norm", (h,)),
            ("self_attn_layer_norm", (h,)),
        ] + [(f"self_attn.{p}_proj", (h, h)) for p in ("q", "k", "v", "out")]:
            for suffix in ("weight", "bias"):
                full = f"audio_tower.layers.{i}.{name}.{suffix}"
                yield full, full, shape if suffix == "weight" else (shape[0],)
    for suffix in ("weight", "bias"):
        name = "audio_tower.ln_post." + suffix
        yield name, name, (h,)
    stacked = config["stack_factor"] * h
    inner = config["hidden_size"] // 2
    for name, shape in [
        ("linear_1.weight", (config["hidden_size"], stacked)),
        ("linear_2.weight", (config["text_config"]["hidden_size"], inner)),
        ("ln_mid.weight", (inner,)),
        ("ln_pre.weight", (stacked,)),
    ]:
        name = "multi_modal_projector." + name
        yield name, name, shape


def audio_contract(root, *, validate_pin=True):
    if (root / "thinker" / "config.json").is_file():
        root = root / "thinker"
    thinker = json.loads((root / "config.json").read_text())
    validate_config(thinker, validate_pin=validate_pin)
    config = json.loads((root / "audio" / "config.json").read_text())
    pre = json.loads((root / "audio" / "preprocessor_config.json").read_text())
    if (
        config.get("model_type") != "ultravox"
        or config.get("projector_act") != "swiglu"
        or config.get("projector_ln_mid") is not True
        or config.get("stack_factor") != 2
        or config["hidden_size"] % 2
    ):
        raise ValueError("unsupported audio/projector config")
    if (
        config["audio_config"]["d_model"]
        % config["audio_config"]["encoder_attention_heads"]
    ):
        raise ValueError("invalid audio attention heads")
    if config["text_config"]["hidden_size"] != thinker["text_config"]["hidden_size"]:
        raise ValueError("audio projector/thinker hidden size mismatch")
    if validate_pin:
        for actual, expected in (
            (config["audio_config"], PINNED_AUDIO),
            (pre, PINNED_PREPROCESSOR),
        ):
            for name, value in expected.items():
                if actual.get(name) != value:
                    raise ValueError(f"pinned audio config {name} mismatch")
        if (
            config.get("hidden_size") != 4096
            or config.get("audio_token_index") != 151669
        ):
            raise ValueError("pinned Ultravox config mismatch")
    reader = Reader(root / "audio")
    specs = list(audio_specs(config))
    if set(reader.entries) != {name for name, _, _ in specs}:
        raise ValueError("missing/unknown audio component tensor")
    for name, _, shape in specs:
        if reader.shape(name) != shape or reader.dtype(name) not in ("BF16", "F32"):
            raise ValueError(f"{name}: invalid audio tensor shape/dtype")
    if validate_pin:
        pinned = json.loads(
            Path(__file__).with_name("auxiliary-inventory.json").read_text()
        )
        for name in reader.entries:
            if (
                name not in pinned
                or list(reader.shape(name)) != pinned[name]["shape"]
                or reader.dtype(name) != pinned[name]["dtype"]
            ):
                raise ValueError(f"{name}: pinned audio tensor mismatch")
    return thinker, config, pre, reader, specs


def convert_audio(model_dir: Path, output: Path, *, quant="f32", validate_pin=True):
    if quant not in FORMATS:
        raise ValueError(f"unknown format {quant}")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    thinker, config, pre, reader, specs = audio_contract(
        Path(model_dir), validate_pin=validate_pin
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    pool = None
    try:
        pool = (
            ProcessPoolExecutor(max_workers=K_WORKERS)
            if quant in ("q4_k", "q5_k", "q6_k") and K_WORKERS > 1
            else None
        )
        writer = GgufWriter(temporary)
        writer.add_meta("general.architecture", "sophon_realtime_audio")
        writer.add_meta("general.name", "Sophon-Realtime-32B-Tag audio")
        writer.add_meta("general.file_type", FILE_TYPES[quant])
        common_metadata(writer, thinker, "audio")
        writer.add_meta(
            "sophon_realtime.audio_config_json",
            json.dumps(config, separators=(",", ":"), ensure_ascii=False),
        )
        writer.add_meta(
            "sophon_realtime.audio_preprocessor_json",
            json.dumps(pre, separators=(",", ":"), ensure_ascii=False),
        )
        writer.add_meta(
            "sophon_realtime.audio_weight_source", "thinker/audio/model.safetensors"
        )
        writer.add_meta("sophon_realtime.requested_quant", quant)
        writer.add_meta(
            "sophon_realtime.tensor_types_json",
            json.dumps(
                {name: target_type(shape, quant) for name, _, shape in specs},
                sort_keys=True,
            ),
        )
        for source, target, shape in specs:
            add_streamed(writer, reader, source, target, shape, quant, pool)
        writer.write()
        metadata, directory = _read_gguf(temporary)
        if (
            set(directory) != {name for name, _, _ in specs}
            or metadata.get("sophon_realtime.source_revision") != REVISION
        ):
            raise ValueError("audio GGUF readback mismatch")
        os.link(temporary, output)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quant", choices=FORMATS, default="f32")
    args = parser.parse_args()
    convert_audio(args.model_dir, args.output, quant=args.quant)


if __name__ == "__main__":
    main()
