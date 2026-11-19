"""Convert pinned Underdog-Woof effective weights to GGUF.

F32 preserves the published int8/grouped-int8 dequantization, not the
unavailable weights preceding upstream quantization. Lookups and vectors
stay F32; projection matrices use the requested shared GGUF format when
their contiguous row width meets that format's block alignment.
"""
import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from tools.converter.utils.gguf import GGML_F32, GgufWriter, gguf_dims
from tools.converter.utils.quantize_gguf import FILE_TYPES, quantize_tensor

CONTRACT = json.loads((Path(__file__).resolve().parents[3] /
                       "src/models/underdog_woof/contract.json").read_text())
FORMATS = ("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q4_k", "q5_k", "q6_k")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources(directory):
    for name, expected in CONTRACT["source"]["hashes"].items():
        if sha256(Path(directory) / name) != expected:
            raise ValueError(f"Underdog-Woof source hash mismatch: {name}")


def float_array(value):
    array = np.asarray(value)
    if array.dtype.kind != "f" or not array.size or any(d <= 0 for d in array.shape):
        raise ValueError("expected a nonempty floating-point tensor")
    result = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite Underdog-Woof tensor")
    return result


def dequantize(entry):
    """Official grouped-int8 layout, including partial groups and nonzero axes."""
    if not isinstance(entry, dict):
        return float_array(entry)
    if set(entry) != {"q8", "s", "shape", "axis", "g"}:
        raise ValueError("invalid grouped-int8 tensor fields")
    q = np.asarray(entry["q8"])
    scales = float_array(entry["s"])
    shape, axis, group = entry["shape"], entry["axis"], entry["g"]
    if (not isinstance(shape, (list, tuple)) or not shape or
            any(type(d) is not int or d <= 0 for d in shape) or
            type(axis) is not int or not 0 <= axis < len(shape) or
            type(group) is not int or group <= 0):
        raise ValueError("invalid grouped-int8 shape, axis or group")
    rows, width = shape[axis], math.prod(d for i, d in enumerate(shape) if i != axis)
    groups = (width + group - 1) // group
    if q.dtype != np.int8 or q.shape != (rows, width) or scales.shape != (rows, groups):
        raise ValueError("invalid grouped-int8 payload shape or type")
    padded = np.zeros((rows, groups * group), dtype=np.float32)
    padded[:, :width] = q
    effective = (padded.reshape(rows, groups, group) * scales[:, :, None]).reshape(rows, -1)[:, :width]
    rest = [d for i, d in enumerate(shape) if i != axis]
    return float_array(np.moveaxis(effective.reshape(rows, *rest), 0, axis))


def linear_weights(entry):
    """Speech linear weights stay in native [output, input] orientation."""
    if "q" in entry:
        if set(entry) - {"q", "s", "b"}:
            raise ValueError("invalid speech linear fields")
        q, scales = np.asarray(entry["q"]), float_array(entry["s"])
        if q.dtype != np.int8 or q.ndim != 2 or scales.shape != (q.shape[0],):
            raise ValueError("invalid speech row-int8 payload")
        weight = float_array(q.astype(np.float32) * scales[:, None])
    else:
        if set(entry) - {"w", "b"} or "w" not in entry:
            raise ValueError("invalid speech linear fields")
        weight = float_array(entry["w"])
    if weight.ndim != 2:
        raise ValueError("speech linear weight must have two dimensions")
    bias = None if "b" not in entry else float_array(entry["b"])
    if bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError("invalid speech linear bias")
    return weight, bias


def conv_taps(weight):
    weight = float_array(weight)
    if weight.ndim != 3:
        raise ValueError("convolution weight must have three dimensions")
    return [np.ascontiguousarray(weight[:, :, tap]) for tap in range(weight.shape[2])]


def conv_transpose_matrix(weight):
    weight = float_array(weight)
    if weight.ndim != 3:
        raise ValueError("transposed convolution weight must have three dimensions")
    cin, cout, kernel = weight.shape
    return np.ascontiguousarray(weight.transpose(0, 2, 1).reshape(cin, kernel * cout).T)


def unflatten(npz):
    def build(node):
        if isinstance(node, dict):
            if set(node) == {"@"}:
                return npz[node["@"]]
            return {key: build(value) for key, value in node.items()}
        if isinstance(node, list):
            return [build(value) for value in node]
        return node
    if set(npz.files) != set(CONTRACT["source_tensors"]) | {"__tree__"}:
        raise ValueError("Underdog-Woof source tensor inventory mismatch")
    for name, expected in CONTRACT["source_tensors"].items():
        value = npz[name]
        if list(value.shape) != expected["shape"] or value.dtype.str != expected["dtype"]:
            raise ValueError(f"Underdog-Woof source tensor shape/type mismatch: {name}")
    return build(json.loads(str(npz["__tree__"])))


def export_tensors(state):
    config = CONTRACT["config"]
    if (state["speech"]["cfg"] != config["speech"] or
            state["decoder"]["cfg"] != {k: v for k, v in config["decoder"].items() if k != "layers"} or
            state["max_frames"] != config["max_frames"] or state["fps"] != config["fps"]):
        raise ValueError("unsupported Underdog-Woof source configuration")
    tensors = {}

    def linear(name, entry):
        weight, bias = linear_weights(entry)
        tensors[name + ".weight"] = weight
        if bias is not None:
            tensors[name + ".bias"] = bias

    speech, decoder = state["speech"], state["decoder"]
    for name in ("ce", "ada", "lat_in", "h_inp", "h_out"):
        linear("speech." + name, speech[name])
    for i, layer in enumerate(speech["layers"]):
        if len(layer) != 4:
            raise ValueError("invalid speech transformer layer")
        for name, entry in zip(("qkv", "o", "fc1", "fc2"), layer):
            linear(f"speech.layers.{i}.{name}", entry)
    for i, layer in enumerate(speech["hblocks"]):
        if len(layer) != 2:
            raise ValueError("invalid speech head block")
        for name, entry in zip(("fc1", "fc2"), layer):
            linear(f"speech.hblocks.{i}.{name}", entry)
    tensors["speech.text_emb.weight"] = float_array(speech["text_emb"])
    tensors["speech.bos"] = float_array(speech["bos"])
    for name, voice in state["voices"].items():
        tensors["voices." + name] = float_array(voice)
    for name in ("mean", "std"):
        tensors["latent." + name] = float_array(state[name])
    tensors["decoder.inp.weight"] = dequantize(decoder["inp"][0])
    tensors["decoder.inp.bias"] = dequantize(decoder["inp"][1])
    tensors["decoder.inv_freq"] = float_array(decoder["inv_freq"])
    for i, layer in enumerate(decoder["tf"]):
        for name in ("qkv", "o", "fc1", "fc2"):
            tensors[f"decoder.tf.{i}.{name}.weight"] = dequantize(layer[name])
        for name in ("ln1", "ln2"):
            for part, value in zip(("weight", "bias"), layer[name], strict=True):
                tensors[f"decoder.tf.{i}.{name}.{part}"] = dequantize(value)

    def conv(name, entry):
        for tap, weight in enumerate(conv_taps(dequantize(entry["w"]))):
            tensors[f"{name}.tap.{tap}.weight"] = weight
        tensors[name + ".bias"] = float_array(entry["b"])

    if len(decoder["seanet"]) != len(CONTRACT["seanet"]):
        raise ValueError("invalid SEANet operation count")
    for i, ((kind, entry), expected) in enumerate(zip(decoder["seanet"], CONTRACT["seanet"])):
        if kind != expected["kind"]:
            raise ValueError(f"invalid SEANet operation: {i}")
        name = f"decoder.seanet.{i}"
        if kind == "conv":
            conv(name, entry)
        elif kind == "convt":
            if entry["stride"] != expected["stride"]:
                raise ValueError(f"invalid SEANet stride: {i}")
            tensors[name + ".weight"] = conv_transpose_matrix(dequantize(entry["w"]))
            tensors[name + ".bias"] = float_array(entry["b"])
        elif kind == "res":
            if len(entry) != 2:
                raise ValueError(f"invalid SEANet residual block: {i}")
            for subindex, sublayer in enumerate(entry):
                conv(f"{name}.{subindex}", sublayer)
        elif kind != "elu":
            raise ValueError(f"unsupported SEANet operation: {kind}")
    if tensors.keys() != CONTRACT["tensors"].keys():
        raise ValueError("Underdog-Woof exported tensor inventory mismatch")
    for name, value in tensors.items():
        if list(value.shape) != CONTRACT["tensors"][name]:
            raise ValueError(f"Underdog-Woof exported tensor shape mismatch: {name}")
    return tensors


def encode_tensor(name, value, target_format):
    if target_format not in FORMATS:
        raise ValueError(f"unsupported Underdog-Woof format: {target_format}")
    value = float_array(value)
    dims, raw = gguf_dims(value.shape), value.astype("<f4").tobytes()
    if target_format == "f32" or name == "speech.text_emb.weight" or name.startswith("voices."):
        return GGML_F32, dims, raw
    if target_format == "f16" and value.ndim == 2 and np.max(np.abs(value)) > np.finfo(np.float16).max:
        raise ValueError(f"tensor overflows F16: {name}")
    kind, payload = quantize_tensor(name, raw, GGML_F32, dims, target_format)
    return kind, dims, payload


def convert(directory, output, target_format="f32"):
    directory, output = Path(directory), Path(output)
    if target_format not in FORMATS:
        raise ValueError(f"unsupported Underdog-Woof format: {target_format}")
    if output.exists():
        raise FileExistsError(output)
    verify_sources(directory)
    with np.load(directory / "woof_int8.npz", allow_pickle=False) as npz:
        tensors = export_tensors(unflatten(npz))
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(path)
    try:
        writer = GgufWriter(temporary)
        writer.add_meta("general.architecture", CONTRACT["architecture"])
        writer.add_meta("general.name", "Underdog-Woof")
        writer.add_meta("general.file_type", FILE_TYPES[target_format])
        writer.add_meta("general.quantization_version", 2)
        writer.add_meta("general.license", "LicenseRef-Underdog-Woof")
        for name, value in {
            "format": target_format,
            "config": json.dumps(CONTRACT["config"], sort_keys=True),
            "seanet": json.dumps(CONTRACT["seanet"], sort_keys=True),
            "source_revision": CONTRACT["source"]["revision"],
            "source_hashes": json.dumps(CONTRACT["source"]["hashes"], sort_keys=True),
            "weight_semantics": CONTRACT["source"]["weight_semantics"],
            "lexicon": (directory / "underdog_woof/respell.tsv").read_text(encoding="utf-8"),
            "license": (directory / "LICENSE").read_text(encoding="utf-8"),
            "notice": (directory / "NOTICE").read_text(encoding="utf-8"),
        }.items():
            writer.add_meta("underdog_woof." + name, value)
        for name, value in sorted(tensors.items()):
            kind, dims, raw = encode_tensor(name, value, target_format)
            writer.add_tensor(name, kind, dims, raw)
        writer.write()
        # Hard-link publication is atomic and cannot replace a concurrently created destination.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--format", choices=FORMATS, default="f32")
    args = parser.parse_args()
    print(convert(args.directory, args.output, args.format))


if __name__ == "__main__":
    main()
