"""Export the ERNIE Flux2 VAE decoder to the runtime's F16/F32 GGUF contract.

Run from the repository root with .venv/bin/python -m
tools.converter.ernie_image.convert_vae INPUT.safetensors OUTPUT.gguf.
"""
import argparse
import re
from pathlib import Path

import numpy as np

from tools.converter.utils.gguf import GgufWriter, GGML_F16, GGML_F32, open_safetensors


def convert(source: Path, destination: Path) -> None:
    tensors = open_safetensors(source)
    for name, shape in [("decoder.conv_in.weight", [512, 32, 3, 3]), ("post_quant_conv.weight", [32, 32, 1, 1])]:
        if tensors.header[name]["shape"] != shape:
            raise ValueError(f"{name}: expected {shape}")
    writer = GgufWriter(destination)
    writer.add_meta("general.architecture", "flux2_vae")
    for name in sorted(tensors.header):
        if not name.startswith(("decoder.", "post_quant_conv.")):
            continue
        tensor = tensors.get(name)
        if tensor.dtype != "F32":
            raise ValueError(f"{name}: expected F32, got {tensor.dtype}")
        matrix = len(tensor.shape) > 1
        # The reference casts Conv2d weights to F16 but keeps Attention Linear F32.
        linear = len(tensor.shape) == 2
        kind = GGML_F16 if matrix and not linear else GGML_F32
        data = np.frombuffer(tensor.raw, dtype="<f4").astype("<f2" if kind == GGML_F16 else "<f4")
        mapped = name.replace("decoder.conv_norm_out.", "decoder.norm_out.")
        mapped = re.sub(r"decoder.up_blocks.(\d+).resnets.(\d+)", lambda m: f"decoder.up.{3 - int(m[1])}.block.{m[2]}", mapped)
        mapped = re.sub(r"decoder.up_blocks.(\d+).upsamplers.0", lambda m: f"decoder.up.{3 - int(m[1])}.upsample", mapped)
        mapped = re.sub(r"decoder.mid_block.resnets.(\d+)", lambda m: f"decoder.mid.block_{int(m[1]) + 1}", mapped)
        mapped = mapped.replace("decoder.mid_block.attentions.0.", "decoder.mid.attn_1.")
        for source_name, target_name in [("group_norm", "norm"), ("to_out.0", "proj_out"), ("to_q", "q"), ("to_k", "k"), ("to_v", "v"), ("conv_shortcut", "nin_shortcut")]:
            mapped = mapped.replace(f".{source_name}.", f".{target_name}.")
        writer.add_tensor(mapped, kind, tensor.shape[::-1], data.tobytes())
    writer.write()
    print(f"{len(writer.tensors)} tensors -> {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)
