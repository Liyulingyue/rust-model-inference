"""Pinned CosyVoice3 flow and F32-materialized HiFT component conversion."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch

from tools.converter.sophon_realtime.convert import (
    FILE_TYPES,
    FORMATS,
    K_WORKERS,
    REVISION,
    add_streamed,
    target_type,
)
from tools.converter.sophon_realtime.talker import PtReader
from tools.converter.utils.gguf import (
    GgufWriter,
    _read_gguf,
    gguf_dims,
    model_metadata_value,
)

YAML_SHA256 = "f5a6b2c6f05139d0f18861a1fe506f751e787026b77c05f7e8fef9f8a4405965"
SOURCE_SHA256 = {
    "flow": "917a794287bb400f849b2c7883babf78a100bd85e57955776e02cea94e37c117",
    "hift": "b279d7641eb97ae55b3b540cfba4f953c26492a2df758328a89a4d007ab87a65",
}
MIN_MATRIX_ELEMENTS = 65536
WN_MARKER = ".parametrizations.weight.original"

# YAML values plus explicit defaults from the pinned CosyVoice3 implementation.
# The tag launcher overrides the YAML's CFG .7 with four steps and CFG 0.
CONFIGS = {
    "flow": dict(
        input_size=80,
        output_size=80,
        spk_embed_dim=192,
        vocab_size=6561,
        sample_rate=24000,
        input_frame_rate=25,
        token_mel_ratio=2,
        pre_lookahead_len=3,
        pre_lookahead_channels=1024,
        dim=1024,
        depth=22,
        heads=16,
        dim_head=64,
        ff_mult=2,
        time_embed_dim=256,
        conv_pos_groups=16,
        conv_pos_kernel=31,
        norm_eps=1e-6,
        ff_activation="gelu_tanh",
        rotary_layout="adjacent_pairs",
        static_chunk_size=50,
        num_decoding_left_chunks=-1,
        solver="euler",
        t_scheduler="cosine",
        sigma_min=1e-6,
        yaml_inference_cfg_rate=0.7,
        tag_default_steps=4,
        tag_default_cfg=0.0,
    ),
    "hift": dict(
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=24000,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_n_fft=16,
        istft_hop_len=4,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1,
        final_lrelu_slope=0.01,
        audio_limit=0.99,
        conv_pre_look_right=4,
        f0_cond_channels=512,
        f0_num_class=1,
        snake_alpha_logscale=False,
        snake_epsilon=1e-9,
        upsample_mode="nearest_then_causal_conv1d",
        conv_layout="OIK",
        stft_center=True,
        stft_pad_mode="reflect",
        stft_periodic_hann=True,
    ),
}


def _linear(prefix, out_features, in_features):
    yield prefix + ".weight", (out_features, in_features)
    yield prefix + ".bias", (out_features,)


def _conv(prefix, out_channels, in_channels, kernel, *, normalized=False):
    yield prefix + ".bias", (out_channels,)
    if normalized:
        yield prefix + WN_MARKER + "0", (out_channels, 1, 1)
        yield prefix + WN_MARKER + "1", (out_channels, in_channels, kernel)
    else:
        yield prefix + ".weight", (out_channels, in_channels, kernel)


def source_specs(component, config):
    c = config
    if component == "flow":
        h, mel = c["dim"], c["input_size"]
        yield "input_embedding.weight", (c["vocab_size"], mel)
        yield from _linear("spk_embed_affine_layer", mel, c["spk_embed_dim"])
        yield from _conv(
            "pre_lookahead_layer.conv1",
            c["pre_lookahead_channels"],
            mel,
            c["pre_lookahead_len"] + 1,
        )
        yield from _conv(
            "pre_lookahead_layer.conv2", mel, c["pre_lookahead_channels"], 3
        )
        prefix = "decoder.estimator."
        yield from _linear(prefix + "time_embed.time_mlp.0", h, c["time_embed_dim"])
        yield from _linear(prefix + "time_embed.time_mlp.2", h, h)
        yield from _linear(prefix + "input_embed.proj", h, 4 * mel)
        for i in (1, 2):
            yield from _conv(
                prefix + f"input_embed.conv_pos_embed.conv{i}.0",
                h,
                h // c["conv_pos_groups"],
                c["conv_pos_kernel"],
            )
        yield prefix + "rotary_embed.inv_freq", (c["dim_head"] // 2,)
        for i in range(c["depth"]):
            block = prefix + f"transformer_blocks.{i}."
            yield from _linear(block + "attn_norm.linear", 6 * h, h)
            for projection in ("to_q", "to_k", "to_v", "to_out.0"):
                yield from _linear(block + "attn." + projection, h, h)
            yield from _linear(block + "ff.ff.0.0", h * c["ff_mult"], h)
            yield from _linear(block + "ff.ff.2", h, h * c["ff_mult"])
        yield from _linear(prefix + "norm_out.linear", 2 * h, h)
        yield from _linear(prefix + "proj_out", mel, h)
    elif component == "hift":
        yield from _linear("m_source.l_linear", 1, c["nb_harmonics"] + 1)
        yield from _conv(
            "conv_pre",
            c["base_channels"],
            c["in_channels"],
            c["conv_pre_look_right"] + 1,
            normalized=True,
        )
        stages = len(c["upsample_rates"])
        for i, kernel in enumerate(c["upsample_kernel_sizes"]):
            channels = c["base_channels"] // 2 ** (i + 1)
            yield from _conv(
                f"ups.{i}", channels, channels * 2, kernel, normalized=True
            )
            rate = math.prod(c["upsample_rates"][i + 1 :])
            yield from _conv(
                f"source_downs.{i}",
                channels,
                c["istft_n_fft"] + 2,
                1 if rate == 1 else rate * 2,
            )
            yield from _resblock(
                f"source_resblocks.{i}",
                channels,
                c["source_resblock_kernel_sizes"][i],
                c["source_resblock_dilation_sizes"][i],
            )
            kernels = c["resblock_kernel_sizes"]
            for j, size in enumerate(kernels):
                yield from _resblock(
                    f"resblocks.{i * len(kernels) + j}",
                    channels,
                    size,
                    c["resblock_dilation_sizes"][j],
                )
        yield from _conv(
            "conv_post",
            c["istft_n_fft"] + 2,
            c["base_channels"] // 2**stages,
            7,
            normalized=True,
        )
        for i in range(5):
            yield from _conv(
                f"f0_predictor.condnet.{2 * i}",
                c["f0_cond_channels"],
                c["in_channels"] if i == 0 else c["f0_cond_channels"],
                4 if i == 0 else 3,
                normalized=True,
            )
        yield from _linear(
            "f0_predictor.classifier", c["f0_num_class"], c["f0_cond_channels"]
        )
    else:
        raise ValueError(f"unknown component {component}")


def _resblock(prefix, channels, kernel, dilations):
    for branch in (1, 2):
        for j in range(len(dilations)):
            yield from _conv(
                f"{prefix}.convs{branch}.{j}",
                channels,
                channels,
                kernel,
                normalized=True,
            )
            yield f"{prefix}.activations{branch}.{j}.alpha", (channels,)


def validate_config(component, config):
    if set(config) != set(CONFIGS[component]):
        raise ValueError("missing/unknown component config field")
    for key, value in config.items():
        expected = CONFIGS[component][key]
        if type(value) is not type(expected):
            raise ValueError(f"invalid component config {key}")
        if (
            type(value) is int
            and value <= 0
            and key not in ("nb_harmonics", "num_decoding_left_chunks")
        ):
            raise ValueError(f"invalid component config {key}")
        if type(value) is float and not math.isfinite(value):
            raise ValueError(f"nonfinite component config {key}")
        if isinstance(expected, (str, bool)) and value != expected:
            raise ValueError(f"unsupported component config {key}")
    if component == "flow":
        if (
            config["dim"] != config["heads"] * config["dim_head"]
            or config["dim"] % config["conv_pos_groups"]
            or config["dim_head"] % 2
            or config["conv_pos_kernel"] % 2 != 1
            or config["input_size"] != config["output_size"]
            or config["norm_eps"] <= 0
            or config["sigma_min"] <= 0
        ):
            raise ValueError("unsupported flow dimensions/config")
    else:
        rates = config["upsample_rates"]
        if (
            not rates
            or config["base_channels"] % 2 ** len(rates)
            or config["istft_n_fft"] % 2
            or config["nb_harmonics"] < 0
        ):
            raise ValueError("unsupported HiFT dimensions/config")
        for key in (
            "upsample_rates",
            "upsample_kernel_sizes",
            "resblock_kernel_sizes",
            "source_resblock_kernel_sizes",
        ):
            if not config[key] or any(
                type(v) is not int or v <= 0 for v in config[key]
            ):
                raise ValueError(f"invalid HiFT config {key}")
        for key in ("resblock_dilation_sizes", "source_resblock_dilation_sizes"):
            if not config[key] or any(
                not isinstance(row, list)
                or not row
                or any(type(v) is not int or v <= 0 for v in row)
                for row in config[key]
            ):
                raise ValueError(f"invalid HiFT config {key}")
        if not (
            len(rates)
            == len(config["upsample_kernel_sizes"])
            == len(config["source_resblock_kernel_sizes"])
            == len(config["source_resblock_dilation_sizes"])
            and len(config["resblock_kernel_sizes"])
            == len(config["resblock_dilation_sizes"])
        ):
            raise ValueError("mismatched HiFT stage config")


class ComponentReader(PtReader):
    def __init__(self, path):
        super().__init__(path)
        self.materialized = {}

    def read(self, name, first=0, rows=None):
        if name not in self.materialized:
            return super().read(name, first, rows)
        g_name, v_name = self.materialized[name]
        g, v = self.tensors[g_name], self.tensors[v_name]
        stop = None if rows is None else first + rows
        g, v = g[first:stop], v[first:stop]
        if not torch.isfinite(g).all() or not torch.isfinite(v).all():
            raise ValueError(f"{name}: nonfinite weight norm source")
        return torch._weight_norm(v, g, 0).detach().numpy()


def tensor_quant(shape, quant):
    return (
        quant if len(shape) == 2 and math.prod(shape) >= MIN_MATRIX_ELEMENTS else "f32"
    )


def component_contract(model_dir, component, *, validate_pin=True, config=None):
    if component not in CONFIGS:
        raise ValueError(f"unknown component {component}")
    if config is not None and validate_pin:
        raise ValueError("custom component config is only for synthetic fixtures")
    config = copy.deepcopy(CONFIGS[component] if config is None else config)
    validate_config(component, config)
    root = Path(model_dir)
    if (root / "talker" / "cosyvoice3.yaml").is_file():
        root = root / "talker"
    hashes = {}
    for name, expected in (
        (component + ".pt", SOURCE_SHA256[component]),
        ("cosyvoice3.yaml", YAML_SHA256),
    ):
        with (root / name).open("rb") as source:
            hashes[name] = hashlib.file_digest(source, "sha256").hexdigest()
        if validate_pin and hashes[name] != expected:
            raise ValueError(f"{name}: pinned {component} source SHA256 mismatch")
    raw_yaml = (root / "cosyvoice3.yaml").read_text()
    reader = ComponentReader(root / (component + ".pt"))
    specs = dict(source_specs(component, config))
    if set(specs) != set(reader.tensors):
        raise ValueError(f"missing/unknown {component} component tensor")
    for name, shape in specs.items():
        value = reader.tensors[name]
        if (
            tuple(value.shape) != shape
            or value.dtype != torch.float32
            or value.layout != torch.strided
            or not value.is_contiguous()
        ):
            raise ValueError(f"{name}: invalid component shape/dtype/layout")
    exported = {}
    for name, shape in specs.items():
        if WN_MARKER in name:
            prefix, part = name.split(WN_MARKER)
            if part == "0":
                continue
            target = prefix + ".weight"
            reader.materialized[target] = (prefix + WN_MARKER + "0", name)
        else:
            target = name
        exported[target] = shape
    return config, raw_yaml, hashes, reader, exported


def convert_component(
    model_dir, output, *, component, quant="f32", validate_pin=True, config=None
):
    if quant not in FORMATS:
        raise ValueError(f"unknown format {quant}")
    if component not in CONFIGS:
        raise ValueError(f"unknown component {component}")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    config, raw_yaml, hashes, reader, specs = component_contract(
        model_dir, component, validate_pin=validate_pin, config=config
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    pool = None
    try:
        pool = (
            ProcessPoolExecutor(max_workers=K_WORKERS)
            if quant in ("q4_k", "q5_k", "q6_k")
            and K_WORKERS > 1
            and any(tensor_quant(shape, quant) != "f32" for shape in specs.values())
            else None
        )
        writer = GgufWriter(temporary)
        metadata = {
            "general.architecture": "sophon_realtime_" + component,
            "general.name": "Sophon-Realtime-32B-Tag " + component,
            "general.file_type": FILE_TYPES[quant],
            "sophon_realtime.component": component,
            "sophon_realtime.source_revision": REVISION,
            "sophon_realtime.source_precision": "F32",
            "sophon_realtime.runtime_quantization": "GGUF weight-only",
            f"sophon_realtime.{component}_config_json": json.dumps(
                config, sort_keys=True, separators=(",", ":")
            ),
            "sophon_realtime.config_json": json.dumps(
                config, sort_keys=True, separators=(",", ":")
            ),
            "sophon_realtime.code2wav_yaml": raw_yaml,
            "sophon_realtime.source_hashes_json": json.dumps(hashes, sort_keys=True),
            "sophon_realtime.weight_source": "talker/" + component + ".pt",
            "sophon_realtime.requested_quant": quant,
            "sophon_realtime.quantization_min_matrix_elements": MIN_MATRIX_ELEMENTS,
            "sophon_realtime.tensor_layout": "original row-major PyTorch; Conv1d OIK; GGUF dimensions reversed",
            "sophon_realtime.tensor_types_json": json.dumps(
                {
                    name: target_type(shape, tensor_quant(shape, quant))
                    for name, shape in specs.items()
                },
                sort_keys=True,
            ),
            "sophon_realtime.weight_norm_materialization": (
                "F32 torch._weight_norm(v,g,dim=0); .weight only; no original g/v"
                if component == "hift"
                else "none"
            ),
            "sophon_realtime.weight_norm_mapping_json": json.dumps(
                reader.materialized, sort_keys=True
            ),
        }
        for key, value in config.items():
            if not isinstance(value, list) or all(
                not isinstance(item, list) for item in value
            ):
                metadata[f"sophon_realtime.{component}." + key] = value
        for key, value in metadata.items():
            writer.add_meta(key, model_metadata_value(value))
        for name, shape in specs.items():
            add_streamed(
                writer, reader, name, name, shape, tensor_quant(shape, quant), pool
            )
        writer.write()
        actual, directory = _read_gguf(temporary)
        if (
            actual.get("general.architecture") != "sophon_realtime_" + component
            or actual.get("sophon_realtime.component") != component
            or actual.get("sophon_realtime.source_revision") != REVISION
            or set(directory) != set(specs)
            or any(
                directory[name][:2]
                != (target_type(shape, tensor_quant(shape, quant)), gguf_dims(shape))
                for name, shape in specs.items()
            )
        ):
            raise ValueError("code2wav GGUF readback mismatch")
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
    parser.add_argument("--component", required=True, choices=tuple(CONFIGS))
    parser.add_argument("--quant", choices=FORMATS, default="f32")
    args = parser.parse_args()
    convert_component(
        args.model_dir, args.output, component=args.component, quant=args.quant
    )


if __name__ == "__main__":
    main()
