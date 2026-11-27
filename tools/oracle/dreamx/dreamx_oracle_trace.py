#!/usr/bin/env python3
"""Capture F32 checkpoints from the official DreamX-Creator CUDA pipelines."""

import argparse
import importlib.util
import json
import re
import runpy
import sys
from collections import defaultdict
from pathlib import Path


def require_cuda(torch_module) -> None:
    if not torch_module.cuda.is_available():
        raise RuntimeError(
            "DreamX Oracle tracing requires a CUDA-capable PyTorch runtime; "
            "the official Creator and refiner pipelines cannot run on CPU or MPS"
        )


class TraceSink:
    def __init__(self, root: Path, torch_module, stage: str, command: list[str]):
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(f"Trace output directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.torch = torch_module
        self.counts = defaultdict(int)
        self.manifest = {
            "format": "dreamx-oracle-trace-v1",
            "stage": stage,
            "command": command,
            "tensors": [],
        }

    def dump(self, name, value, integer=False):
        call = self.counts[name]
        self.counts[name] += 1
        self._dump_tree(f"{name}.call{call:03d}", value, integer)

    def _dump_tree(self, name, value, integer):
        if self.torch.is_tensor(value):
            self._dump_tensor(name, value, integer)
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                self._dump_tree(f"{name}.{key}", value[key], integer)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self._dump_tree(f"{name}.{index}", item, integer)

    def _dump_tensor(self, name, value, integer):
        dtype = self.torch.int64 if integer else self.torch.float32
        suffix = "i64" if integer else "f32"
        tensor = value.detach().to(device="cpu", dtype=dtype).contiguous()
        filename = re.sub(r"[^0-9A-Za-z_.-]+", "_", name) + f".{suffix}"
        path = self.root / filename
        array = tensor.numpy().astype("<i8" if integer else "<f4", copy=False)
        array.tofile(path)
        self.manifest["tensors"].append(
            {
                "name": name,
                "file": filename,
                "source_dtype": str(value.dtype),
                "storage": suffix,
                "shape": list(value.shape),
                "elements": value.numel(),
            }
        )

    def finish(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")
        print(f"DreamX Oracle trace: {path}")


class TracedTokenizer:
    def __init__(self, tokenizer, sink):
        self.tokenizer = tokenizer
        self.sink = sink

    def __call__(self, *args, **kwargs):
        result = self.tokenizer(*args, **kwargs)
        self.sink.dump("creator.token_ids", result.input_ids, integer=True)
        if hasattr(result, "attention_mask"):
            self.sink.dump(
                "creator.attention_mask", result.attention_mask, integer=True
            )
        return result

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def _wrap_method(target, name, sink, input_name, output_name):
    original = getattr(target, name)

    def traced(*args, **kwargs):
        if args:
            sink.dump(input_name, args[0])
        output = original(*args, **kwargs)
        sink.dump(output_name, output)
        return output

    setattr(target, name, traced)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load upstream entrypoint: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_creator(upstream_root: Path, upstream_args: list[str], sink: TraceSink):
    entry = upstream_root / "audio_video_generation" / "inference.py"
    if not entry.is_file():
        raise FileNotFoundError(f"DreamX Creator entrypoint not found: {entry}")
    sys.path.insert(0, str(entry.parent))
    module = _load_module(entry, "dreamx_creator_inference")

    original_setup = module.setup_models

    def setup_models(*args, **kwargs):
        models = original_setup(*args, **kwargs)
        models["tokenizer"] = TracedTokenizer(models["tokenizer"], sink)
        transformer = models["transformer"]
        creator = getattr(transformer, "model", transformer)
        blocks = creator.joint_blocks
        if len(blocks) < 30:
            raise RuntimeError(f"Expected 30 Creator joint blocks, got {len(blocks)}")
        for index in (0, 15, 29):
            blocks[index].register_forward_hook(
                lambda _module, _inputs, output, layer=index: sink.dump(
                    f"creator.joint_layer_{layer:02d}", output
                )
            )
        _wrap_method(
            models["video_vae"],
            "decode",
            sink,
            "creator.final_video_latent",
            "creator.video_decoder_output",
        )
        _wrap_method(
            models["audio_vae"],
            "decode",
            sink,
            "creator.final_audio_latent",
            "creator.audio_decoder_output",
        )
        return models

    original_encode_first_frame = module.encode_first_frame

    def encode_first_frame(*args, **kwargs):
        output = original_encode_first_frame(*args, **kwargs)
        sink.dump("creator.first_frame_latent", output)
        return output

    original_encode_prompt = module.encode_prompt

    def encode_prompt(*args, **kwargs):
        output = original_encode_prompt(*args, **kwargs)
        sink.dump("creator.text_context", output)
        return output

    module.setup_models = setup_models
    module.encode_first_frame = encode_first_frame
    module.encode_prompt = encode_prompt
    sys.argv = [str(entry), *upstream_args]
    module.main()


def run_refiner(upstream_root: Path, upstream_args: list[str], sink: TraceSink):
    root = upstream_root / "video_refiner"
    entry = root / "inference_sr.py"
    if not entry.is_file():
        raise FileNotFoundError(f"DreamX refiner entrypoint not found: {entry}")
    sys.path.insert(0, str(root))

    from pipeline_sr.causal_inference import CausalInferencePipeline

    original_init = CausalInferencePipeline.__init__

    def init(pipeline, *args, **kwargs):
        original_init(pipeline, *args, **kwargs)
        blocks = pipeline.generator.model.blocks
        if not blocks:
            raise RuntimeError("DreamX SR-DiT has no transformer blocks")
        for index in (0, len(blocks) - 1):
            blocks[index].register_forward_hook(
                lambda _module, _inputs, output, layer=index: sink.dump(
                    f"refiner.block_{layer:02d}", output
                )
            )
        _wrap_method(
            pipeline.vae,
            "decode_to_pixel",
            sink,
            "refiner.decoder_input",
            "refiner.decoder_output",
        )

    CausalInferencePipeline.__init__ = init
    for method_name in ("inference_sr", "inference_sr_df"):
        original = getattr(CausalInferencePipeline, method_name)

        def traced(pipeline, *args, _original=original, **kwargs):
            output = _original(pipeline, *args, **kwargs)
            sink.dump("refiner.final_sr_latent", output)
            return output

        setattr(CausalInferencePipeline, method_name, traced)

    sys.argv = [str(entry), *upstream_args]
    runpy.run_path(str(entry), run_name="__main__")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Capture DreamX-Creator official CUDA Oracle checkpoints",
        epilog="Pass official entrypoint arguments after a standalone --.",
    )
    parser.add_argument("stage", choices=("creator", "refiner"))
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    if "--" not in argv:
        parser.parse_args(argv)
        parser.error("official entrypoint arguments must follow --")
    split = argv.index("--")
    args = parser.parse_args(argv[:split])
    upstream_args = argv[split + 1 :]
    if not upstream_args:
        parser.error("official entrypoint arguments are required after --")
    return parser, args, upstream_args


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser, args, upstream_args = parse_args(argv)
    try:
        import torch
    except ImportError:
        parser.error("DreamX Oracle tracing requires PyTorch with CUDA support")
    try:
        require_cuda(torch)
    except RuntimeError as error:
        parser.error(str(error))

    sink = TraceSink(args.out_dir, torch, args.stage, upstream_args)
    try:
        if args.stage == "creator":
            run_creator(args.upstream_root, upstream_args, sink)
        else:
            run_refiner(args.upstream_root, upstream_args, sink)
    finally:
        sink.finish()


if __name__ == "__main__":
    main()
