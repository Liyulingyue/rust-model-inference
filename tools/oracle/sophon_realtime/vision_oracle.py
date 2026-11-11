"""Official Transformers 4.57.6 vision-only reference using the exported F32 GGUF.

No thinker is constructed. Raw checkpoints stay beside the remote result JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import time
from pathlib import Path

import numpy as np

from .oracle import GATES, meets_gate, metrics, reference_weight, sha256


CONFIG_SHA256 = "b48dc9247bf75aa54ba2af1109ae9e2e5414db54ce0331dba8169ccd6f4405ba"
REVISION = "990d6a74bbf15c1765a6f7736fa839f423fd5b59"


def tensor_name(name, deepstack_layers):
    top = {
        "v.patch_embd.bias": "patch_embed.proj.bias",
        "v.position_embd.weight": "pos_embed.weight",
        **{f"v.post_ln.{s}": f"merger.norm.{s}" for s in ("weight", "bias")},
        **{f"mm.{i}.{s}": f"merger.linear_fc{j}.{s}"
           for i, j in ((0, 1), (2, 2)) for s in ("weight", "bias")},
    }
    if name in top:
        return top[name]
    block = re.fullmatch(r"v\.blk\.(\d+)\.(\w+)\.(weight|bias)", name)
    if block:
        layer, role, suffix = block.groups()
        role = {
            "ln1": "norm1", "ln2": "norm2", "attn_qkv": "attn.qkv",
            "attn_out": "attn.proj", "ffn_up": "mlp.linear_fc1", "ffn_down": "mlp.linear_fc2",
        }.get(role)
        if role:
            return f"blocks.{int(layer)}.{role}.{suffix}"
    stack = re.fullmatch(r"v\.deepstack\.(\d+)\.(norm|fc1|fc2)\.(weight|bias)", name)
    if stack:
        layer, role, suffix = stack.groups()
        if int(layer) in deepstack_layers:
            index = deepstack_layers.index(int(layer))
            role = {"norm": "norm", "fc1": "linear_fc1", "fc2": "linear_fc2"}[role]
            return f"deepstack_merger_list.{index}.{role}.{suffix}"
    raise ValueError(f"unexpected vision tensor: {name}")


def join_patch_weights(first, second, *, hidden, patch):
    frames = [np.asarray(frame).reshape(hidden, 3, patch, patch) for frame in (first, second)]
    return np.ascontiguousarray(np.stack(frames, axis=2))


def run(args):
    import gguf
    import torch
    import transformers
    from gguf.quants import dequantize
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

    if transformers.__version__ != "4.57.6":
        raise ValueError("this oracle requires Transformers 4.57.6")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    config_path = args.model_dir / "thinker/config.json"
    if sha256(config_path) != CONFIG_SHA256:
        raise ValueError("source config does not match the pinned Sophon release")
    source_config = json.loads(config_path.read_text())
    vision = dict(source_config["vision_config"])
    if vision.pop("rope_parameters") != {"rope_theta": 10000.0, "rope_type": "axial"}:
        raise ValueError("unexpected vision rotary contract")
    config = Qwen3VLVisionConfig(**vision)
    config._attn_implementation = "eager"
    pre_path = args.model_dir / "thinker/preprocessor_config.json"
    pre = json.loads(pre_path.read_text())

    reader = gguf.GGUFReader(str(args.gguf))
    def string(key):
        field = reader.fields[key]
        return bytes(field.parts[field.data[0]]).decode("utf-8")
    if (string("general.architecture"), string("sophon_realtime.component"),
        string("sophon_realtime.source_revision")) != ("clip", "vision", REVISION):
        raise ValueError("not the pinned Sophon vision component")
    flag = reader.fields.get("clip.vision.preserve_f32")
    if flag is None or flag.types != [gguf.GGUFValueType.BOOL] or not flag.parts[flag.data[0]].item():
        raise ValueError("GGUF must explicitly opt into F32 vision compute")
    if json.loads(string("sophon_realtime.config_json")) != source_config:
        raise ValueError("GGUF config differs from the pinned source")
    if json.loads(string("sophon_realtime.preprocessor_json")) != pre:
        raise ValueError("GGUF preprocessing differs from the source")
    if any(int(t.tensor_type) != 0 for t in reader.tensors):
        raise ValueError("this F32 oracle does not accept quantized vision weights")

    tensors = {t.name: t for t in reader.tensors}
    patch_names = ("v.patch_embd.weight", "v.patch_embd.weight.1")
    values = lambda name: reference_weight(tensors[name], dequantize, None)
    state = {
        tensor_name(name, config.deepstack_visual_indexes): torch.from_numpy(values(name))
        for name in tensors if name not in patch_names
    }
    patch = join_patch_weights(*(values(name) for name in patch_names),
                               hidden=config.hidden_size, patch=config.patch_size)
    state["patch_embed.proj.weight"] = torch.from_numpy(patch)
    with torch.device("meta"):
        model = hf.Qwen3VLVisionModel(config)
    model.load_state_dict(state, strict=True, assign=True)
    # inv_freq is non-persistent, so restore it with the official constructor after meta loading.
    model.rotary_pos_emb = hf.Qwen3VLVisionRotaryEmbedding(config.hidden_size // config.num_heads // 2)
    model.eval()
    if any(t.is_meta or t.dtype != torch.float32 for t in model.parameters()):
        raise ValueError("vision parameters must all be resident F32")

    image = Image.open(args.image).convert("RGB")
    if image.size != (32, 32) or not np.all(np.asarray(image) == [255, 0, 0]):
        raise ValueError("this fixture requires the fixed 32x32 red image")
    # 4.57.6's slow processor requires explicit pixel budgets to retain the source size values.
    processor = Qwen2VLImageProcessor(
        min_pixels=pre["size"]["shortest_edge"], max_pixels=pre["size"]["longest_edge"],
        patch_size=pre["patch_size"], temporal_patch_size=pre["temporal_patch_size"],
        merge_size=pre["merge_size"], image_mean=pre["image_mean"], image_std=pre["image_std"],
    )
    processed = processor(images=image, return_tensors="pt")
    pixels, grid = processed["pixel_values"], processed["image_grid_thw"]
    if grid.tolist() != [[1, 16, 16]] or list(pixels.shape) != [256, 1536]:
        raise ValueError("unexpected preprocessed patch grid")
    channels = pixels.reshape(256, 3, 2, 16, 16)
    if not all(torch.all(channels[:, c] == value).item() for c, value in enumerate((1, -1, -1))):
        raise ValueError("official preprocessing differs from the Rust red fixture")

    captures = {}
    def capture(name, value):
        if name in captures:
            raise ValueError(f"duplicate checkpoint: {name}")
        captures[name] = value.detach().float().cpu().contiguous().numpy().copy()
    handles = [model.blocks[0].norm1.register_forward_pre_hook(lambda _m, x: capture("input", x[0]))]
    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(lambda _m, _x, y, i=i: capture(f"blocks.{i}", y)))
    for i, merger in enumerate(model.deepstack_merger_list):
        handles.append(merger.register_forward_hook(lambda _m, _x, y, i=i: capture(f"deepstack.{i}", y)))
    started = time.perf_counter()
    with torch.inference_mode():
        projected, _deepstack = model(pixels, grid)
    seconds = time.perf_counter() - started
    capture("output", projected)
    for handle in handles:
        handle.remove()

    names = ["input"]
    for i in range(config.depth):
        names.append(f"blocks.{i}")
        if i in config.deepstack_visual_indexes:
            names.append(f"deepstack.{config.deepstack_visual_indexes.index(i)}")
    names.append("output")
    if list(captures) != names:
        raise ValueError("official hook order differs from the vision checkpoint contract")
    records = [json.loads(line) for line in args.rust_trace.read_text().splitlines()]
    if [r["name"] for r in records] != [f"mage_flow.vision.{name}" for name in names]:
        raise ValueError("Rust trace does not have the 32 ordered vision checkpoints")
    args.output.mkdir(parents=True, exist_ok=False)
    pixels.numpy().astype("<f4").tofile(args.output / "pixel-values.f32")
    results = []
    for i, (name, record) in enumerate(zip(names, records)):
        expected = captures[name]
        if record["shape"] != [1, expected.size] or record["occurrence"] != 0 or record["layer"] is not None:
            raise ValueError(f"Rust checkpoint contract mismatch at {name}")
        sidecar = Path(record["binary_path"])
        actual = np.fromfile(sidecar, dtype="<f4").reshape(expected.shape)
        reference_path = args.output / f"{i:02d}-{name.replace('.', '-')}.f32"
        expected.astype("<f4").tofile(reference_path)
        result = metrics(actual, expected)
        results.append({"name": name, "shape": list(expected.shape), **result,
                        "pass": meets_gate(result, "f32"), "rust_sha256": sha256(sidecar),
                        "reference_sha256": sha256(reference_path)})
    first = next((r for r in results if not r["pass"]), None)
    report = {
        "status": "PASS" if first is None else "FAIL", "first_divergence": first,
        "gates": {"relative_rmse_max": GATES["f32"][0], "cosine_min": GATES["f32"][1]},
        "vision_config": source_config["vision_config"],
        "reference": {"class": f"{type(model).__module__}.{type(model).__name__}", "transformers": transformers.__version__,
                      "torch": torch.__version__, "numpy": np.__version__, "attention": "eager",
                      "vision_module_sha256": sha256(Path(inspect.getfile(hf))),
                      "processor_class": "Qwen2VLImageProcessor",
                      "processor_module_sha256": sha256(Path(inspect.getfile(Qwen2VLImageProcessor))),
                      "threads": 4, "interop_threads": 1, "compute": "cpu", "dtype": "f32",
                      "vision_only": True, "parameters": sum(p.numel() for p in model.parameters()),
                      "inference_seconds": seconds},
        "inputs": {"gguf": str(args.gguf), "gguf_sha256": sha256(args.gguf),
                   "source_config_sha256": sha256(config_path), "preprocessor_sha256": sha256(pre_path),
                   "image_sha256": sha256(args.image), "image_size": [32, 32], "grid_thw": grid.tolist(),
                   "image_rgb_sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
                   "image_min_pixels": processor.min_pixels, "image_max_pixels": processor.max_pixels,
                   "normalized_channels": [1, -1, -1], "projected_shape": [64, config.out_hidden_size],
                   "deepstack_layers": config.deepstack_visual_indexes,
                   "rust_trace": str(args.rust_trace), "rust_trace_sha256": sha256(args.rust_trace),
                   "rust_binary": str(args.rust_binary), "rust_binary_sha256": sha256(args.rust_binary),
                   "source_manifest": str(args.source_manifest), "source_manifest_sha256": sha256(args.source_manifest),
                   "oracle_script_sha256": sha256(Path(__file__))},
        "results": results,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checkpoints": len(results), "first_divergence": first}))
    return first is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "gguf", "image", "rust-trace", "rust-binary", "source-manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    raise SystemExit(0 if run(parser.parse_args()) else 1)


if __name__ == "__main__":
    main()
