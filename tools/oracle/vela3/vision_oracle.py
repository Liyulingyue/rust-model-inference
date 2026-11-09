#!/usr/bin/env python3
"""Export reproducible UltraVBERT CPU/F32 checkpoints from the shipped Python source.

  .venv/bin/python tools/oracle/vela3/vision_oracle.py --root models/vela3-fast-vision --out .codex/vela3/vision-tiny
  RMI_VELA3_VISION_FIXTURE=.codex/vela3/vision-tiny cargo test --lib vela3::vision::tests::official_fixture -- --ignored --nocapture

The fixture covers SigLIP patches/layers, pixel shuffle, RMS connector, merged
embeddings, bidirectional UltraBERT, LoRA and the task head. --task selects a
real task/backbone; --raw uses the real raw encoder and a synthetic readout. No snapshot
files are modified. The compatibility shim supplies only config declarations
and registers the package's own UltraBERT implementation in transformers 5.19.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModel, GPT2Tokenizer
from transformers.configuration_utils import PreTrainedConfig
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

SOURCE_SHA256 = {
    "configuration_ultravbert.py": "97b79ba8cf3d1e96b829441fa06fcc56ed83d3c0d43a89b04661c463cf9bf1cf",
    "modeling_ultravbert.py": "fdc8e5c3ace4336e0eee7f54366029e3bb7ab347b51cf01372eaa38e3ebae866",
    "image_processing_ultravbert.py": "6c41054315f105fabda36d9fcc9d84f1fce84bf0cb4489917360a5dc2544dac9",
    "uvb_core.py": "973a990aee3ec5968ca1f8d321edf6add2bd92e18cd4939eeb0b9c888fed5a58",
    "vela3v_encoder.py": "2398a40745281624c14598f486846999068de4ac1486fec36c3d8e868908c2ed",
    "text/configuration_ultrabert.py": "823d10bda19f5164da4f79f8f9c01e5ebd7a75569abce78c0a26b7674fc78cd9",
    "text/modeling_ultrabert.py": "6d86b56fbcc634dd2f7e2aa29047062289def1ba7e602b991be1af888dc82072",
    "config.json": "96c0a13f682092cb4b7289ec10d54179dd0085bbc988cba269a7b3d1ace1c079",
    "vela3v_config.json": "f8d159468b49ee268c29d29c99ec488226ef7524fd967caa34d8d71f333f71f2",
    "processor_config.json": "0657d91d3b77ff66812ae4527aa0fad027dfd532df237e73c5b794cc2fcbac5c",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def official_classes(root):
    for file, expected in SOURCE_SHA256.items():
        actual = hashlib.sha256((root / file).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"source snapshot changed: {file}: {actual} != {expected}")
    name = "transformers.models.ultrabert"
    package = types.ModuleType(name)
    package.__path__ = [str(root / "text")]
    sys.modules[name] = package
    config_class = load_module(name + ".configuration_ultrabert", root / "text/configuration_ultrabert.py").UltrabertConfig
    text_module = load_module(name + ".modeling_ultrabert", root / "text/modeling_ultrabert.py")
    AutoConfig.register("ultrabert", config_class, exist_ok=True)
    AutoModel.register(config_class, text_module.UltrabertModel, exist_ok=True)

    class UltraVBertConfig(PreTrainedConfig):
        model_type = "ultravbert"
        sub_configs = {"text_config": config_class, "vision_config": SiglipVisionConfig}

        def __init__(self, text_config=None, vision_config=None, **kwargs):
            super().__init__(**kwargs)
            text_config = dict(text_config or {})
            # The shipped old text module expresses the same per-layer contract as lists.
            if "layer_types" in text_config:
                kinds = text_config["layer_types"]
                rope = text_config["rope_parameters"]
                text_config["attention_windows"] = [text_config["sliding_window"] if k == "sliding_attention" else None for k in kinds]
                text_config["rope_theta"] = [rope[k]["rope_theta"] for k in kinds]
            self.text_config = config_class(**text_config)
            self.vision_config = SiglipVisionConfig(**(vision_config or {}))
            for k, default in dict(image_token_id=49191, pixel_shuffle_factor=4, initializer_range=0.02,
                                   initializer_cutoff_factor=2.0, connector_norm=True).items():
                setattr(self, k, kwargs.get(k, default))
            self._attn_implementation = "eager"
            self.text_config._attn_implementation = "eager"
            self.vision_config._attn_implementation = "eager"

    name = "rmi_ultravbert_oracle"
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    sys.modules[name] = package
    shim = types.ModuleType(name + ".configuration_ultravbert")
    shim.UltraVBertConfig = UltraVBertConfig
    sys.modules[shim.__name__] = shim
    model = load_module(name + ".modeling_ultravbert", root / "modeling_ultravbert.py").UltraVBertModel
    processor = load_module(name + ".image_processing_ultravbert", root / "image_processing_ultravbert.py").UltraVBertImageProcessor
    sys.path.insert(0, str(root))
    core = load_module("uvb_core", root / "uvb_core.py")
    return UltraVBertConfig, model, processor, core


def synthetic_image():
    y, x = np.indices((21, 37))
    return Image.fromarray(np.stack(((x * 13 + y * 7) % 256, (x * 5 + y * 19) % 256, (x * 23 + y * 3) % 256), -1).astype(np.uint8))


def dump(out, name, value):
    array = value.detach().float().cpu().contiguous().numpy() if isinstance(value, torch.Tensor) else np.asarray(value, dtype=np.float32)
    file = name + ".f32"
    array.astype("<f4").tofile(out / file)
    return {"name": name, "file": file, "shape": list(array.shape)}


def weight_provenance(path):
    path = Path(path)
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def assembly_cases(proc, core, image):
    proc.image_processor.size = {"longest_edge": 1024}
    proc.image_processor.do_image_splitting = True
    state = {"user": "详细描述", "answer": "Résumé: 中🙂 [42.00]."}
    ocr = {"invoice": {"total": "42.00", "count": 2}, "notes": ["é", "中"]}
    cases = [
        ("decision", "  hello 中é  ", {"type": "choice", "instructions": "Pick the subject", "criteria": {"A": "receipt", "B": None}},
         core.decision_example(proc, [image, image], "  hello 中é  ", "Pick the subject", ["A: receipt", "B"]), 2),
        ("span", state, {"type": "span"}, core.span_example(proc, image, "User request: " + state["user"], state["answer"]), 1),
        ("ocr", ocr, {"type": "span", "criteria": {"ocr_error": None}},
         core.span_example(proc, image, "User request: Check this OCR extraction against the scanned document.",
                           "invoice.total: 42.00\ninvoice.count: 2\nnotes.0: é\nnotes.1: 中"), 1),
        ("box", "", {"type": "box", "instructions": "the total amount"}, core.box_example(proc, image, "the total amount"), 1),
    ]
    out = []
    for task, state, question, example, count in cases:
        if task == "decision":
            selections = example["spans"]
        else:
            positions = example["img_pos"] if task == "box" else example["word_pos"]
            selections = [(int(i), int(i) + 1) for i in positions]
        out.append(dict(task=task, state=state, question=question, images=count, input_ids=example["input_ids"],
                        selections=[list(s) for s in selections],
                        offsets=example.get("w_offs", np.zeros((0, 2), dtype=np.int64)).tolist()))
    return out


def run(args):
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261030)
    Config, Model, Processor, C = official_classes(root)
    from transformers.models.idefics3.processing_idefics3 import Idefics3Processor
    proc = Idefics3Processor(Processor(**json.loads((root / "processor_config.json").read_text())["image_processor"]),
                             GPT2Tokenizer.from_pretrained(root), image_seq_len=64)
    image = synthetic_image()
    image.save(out / "image.png")
    layouts = []
    for edge in (512, 1024):
        proc.image_processor.size = {"longest_edge": edge}
        proc.image_processor.do_image_splitting = edge > 512
        ids, pixels, mask = C.image_prefix(proc, [image])
        _, _, grid = C.image_cells(proc, ids)
        layouts.append({"edge": edge, "input_ids": ids, "grid": list(grid),
                        "pixels": dump(out, f"preprocess_{edge}", pixels), "mask_sum": int(mask.sum())})
    assembly = assembly_cases(proc, C, image)

    config = json.loads((root / "config.json").read_text())
    if args.task or args.raw:
        task_config = json.loads((root / "vela3v_config.json").read_text())
        task_name = args.task or "decision"
        task = task_config["tasks"][task_name] if not args.raw else {"edge": 512}
        cfg = Config(**config)
        enc = Model(cfg).eval()
        backbone = root / task.get("backbone_file", "model.safetensors")
        enc.load_state_dict({k: v.float() for k, v in load_file(str(backbone)).items()}, strict=True)
        adapter = load_file(str(root / task["file"])) if not args.raw else None
        edge = task["edge"]
        proc.image_processor.size = {"longest_edge": edge}
        proc.image_processor.do_image_splitting = edge > 512
        C.Proc._cache[(str(root), edge)] = proc
        if task_name == "decision":
            e = C.decision_example(proc, [image], "An illustrated receipt.", "What is shown?", ["receipt", "landscape"])
        elif task_name == "box":
            e = C.box_example(proc, image, "the total amount")
        elif task_name == "ocr":
            e = C.span_example(proc, image, "User request: Check this OCR extraction against the scanned document.", "The receipt totals 42.00.")
        else:
            e = C.span_example(proc, image, "User request: Describe this image.", "The receipt totals 42.00.")
        backbone_path, adapter_path = str(backbone), str(root / task["file"]) if not args.raw else str(out / "adapter.safetensors")
        rank, alpha = task.get("lora_rank", 0), task.get("lora_alpha", 0)
    else:
        config.update(image_token_id=25, pixel_shuffle_factor=2)
        config["vision_config"].update(hidden_size=8, intermediate_size=12, num_attention_heads=2,
                                        num_hidden_layers=2, image_size=8, patch_size=2)
        config["text_config"] = dict(model_type="ultrabert", vocab_size=32, hidden_size=8, intermediate_size=12,
                                     num_attention_heads=2, num_key_value_heads=2, head_dim=4, num_hidden_layers=2,
                                     attention_windows=[4, None], rope_theta=[10000.0, 2560000.0], rms_norm_eps=1e-6,
                                     attention_bias=False, mlp_bias=False, hidden_act="silu")
        cfg, task_name, rank, alpha = Config(**config), "decision", 2, 4
        enc = Model(cfg).eval()
        save_file({k: v.contiguous() for k, v in enc.state_dict().items()}, str(out / "model.safetensors"))
        pixels = torch.linspace(-0.9, 0.8, 3 * 8 * 8).reshape(1, 3, 8, 8)
        e = dict(input_ids=[0, 25, 25, 25, 25, 2, 3, 4, 15], pv=pixels, pm=torch.ones(1, 8, 8, dtype=torch.bool), spans=[(5, 7), (7, 8)])
        adapter = None
        backbone_path, adapter_path = str(out / "model.safetensors"), str(out / "adapter.safetensors")

    net = C.TaskNet.__new__(C.TaskNet)
    nn.Module.__init__(net)
    net.enc, net.head = enc, C.Head(h=cfg.text_config.hidden_size)
    if rank:
        for layer in enc.text_model.layers:
            for _, module in list(layer.named_modules()):
                for target in C.TARGETS:
                    base = getattr(module, target, None)
                    if isinstance(base, nn.Linear):
                        setattr(module, target, C.LoRALinear(base, rank, alpha))
    if adapter is None:
        with torch.no_grad():
            for name, parameter in net.named_parameters():
                if name.endswith("lora_B"):
                    parameter.copy_(torch.randn_like(parameter) * 0.025)
        state = net.state_dict()
        adapter = {k: v.contiguous() for k, v in state.items() if k.startswith("head.") or "lora_" in k}
        # Exercise a directly trained norm replacement as well as a wrapped base matrix.
        for name in (("enc.text_model.norm.weight", "enc.text_model.layers.0.self_attn.q_proj.base.weight") if rank else ()):
            adapter[name] = (state[name] + torch.randn_like(state[name]) * 0.005).contiguous()
        save_file(adapter, adapter_path)
    state = net.state_dict()
    unexpected = set(adapter) - set(state)
    if unexpected:
        raise ValueError(f"unexpected adapter keys: {sorted(unexpected)[:10]}")
    with torch.no_grad():
        for name, value in adapter.items():
            state[name].copy_(value.float())
    net.eval()

    traces, handles = [], []
    # Capture one image frame at a time, in the same order as the native encoder.
    vision = enc.vision_model
    for frame, pixels in enumerate(e["pv"]):
        frame_traces = []
        def hook(name):
            return lambda _module, _inputs, result: frame_traces.append(dump(out, f"frame_{frame}_{name}", result))
        handles.append(vision.embeddings.register_forward_hook(hook("patch_embeddings")))
        for i, layer in enumerate(vision.encoder.layers):
            handles.append(layer.register_forward_hook(hook(f"vision_layer_{i}")))
        with torch.inference_mode():
            hidden = vision(pixel_values=pixels[None]).last_hidden_state
            frame_traces.append(dump(out, f"frame_{frame}_vision_hidden", hidden))
            shuffled = enc.connector.pixel_shuffle(hidden, cfg.pixel_shuffle_factor)
            frame_traces.append(dump(out, f"frame_{frame}_pixel_shuffle", shuffled))
            embeddings = enc.connector(hidden)
            frame_traces.append(dump(out, f"frame_{frame}_image_embeddings", embeddings))
        for handle in handles:
            handle.remove()
        handles.clear()
        traces.extend(frame_traces)

    with torch.inference_mode():
        ids = torch.tensor([e["input_ids"]])
        pixels = e["pv"][None]
        image_embeds = enc.get_image_features(pixel_values=pixels).pooler_output
        merged = enc.inputs_merger(ids, enc.text_model.embed_tokens(ids), image_embeds)
        traces.append(dump(out, "merged_embeddings", merged))
        h = enc.text_model(inputs_embeds=merged).last_hidden_state
        traces.append(dump(out, "text_hidden", h))
        if task_name == "decision":
            selections = e["spans"]
            selected = torch.stack([h[0, a:b].mean(0) for a, b in selections])
        else:
            positions = e["img_pos"] if task_name == "box" else e["word_pos"]
            selections = [(int(i), int(i) + 1) for i in positions]
            selected = h[0, positions]
        traces.append(dump(out, "selected_hidden", selected))
        logits = net.head(selected)[:, 0]
        traces.append(dump(out, "logits", logits))
    public = {}
    if args.task:
        if task_name == "decision":
            state, question = "An illustrated receipt.", {"type": "choice", "instructions": "What is shown?", "criteria": ["receipt", "landscape"]}
        elif task_name == "box":
            state, question = "", {"type": "box", "instructions": "the total amount"}
        elif task_name == "ocr":
            # A zero threshold exercises nonempty decoder output even on this synthetic image.
            state, question = "The receipt totals 42.00.", {"type": "span", "task": "ocr_check", "threshold": 0.0}
        else:
            state, question = {"user": "Describe this image.", "answer": "The receipt totals 42.00."}, {"type": "span"}
        encoder_class = load_module("vela3v_encoder", root / "vela3v_encoder.py").Vela3VEncoder
        encoder = encoder_class.__new__(encoder_class)
        encoder.path, encoder.cfg, encoder.name = root, task_config, task_config["name"]
        encoder.device, encoder.bf16 = "cpu", False
        encoder.nets = {task_name: net}
        if task_name == "ocr":
            encoder.nets["span"] = net
        def cached_logits(_task, examples, *_args):
            # Reuse the already captured forward only after proving the public wrapper built identical inputs.
            assert examples[0]["input_ids"] == e["input_ids"]
            assert torch.equal(examples[0]["pv"], e["pv"])
            return [logits.detach().cpu().numpy()]
        net.logits = cached_logits
        expected = encoder.system_one(state, {"q": question}, images=[image])
        public = {"request": {"state": state, "questions": {"q": question}, "images": [str(out / "image.png")]}, "expected": expected}
    manifest = dict(source_root=str(root), config=config, task=task_name, backbone=backbone_path, adapter=adapter_path,
                    lora_scale=alpha / rank if rank else 0, input_ids=e["input_ids"],
                    selections=[list(s) for s in selections], grid=list(e.get("grid", (0, 0))),
                    pixels=[dump(out, f"input_frame_{i}", p) for i, p in enumerate(e["pv"])],
                    checkpoints=traces, layouts=layouts, assembly=assembly, source_sha256=SOURCE_SHA256,
                    weights={"backbone": weight_provenance(backbone_path), "adapter": weight_provenance(adapter_path)},
                    runtime={"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                             "torchvision": __import__("torchvision").__version__, "dtype": "float32", "attention": "eager"},
                    preprocessing="Shipped processor declares Lanczos; torchvision <0.27 falls back to uint8 antialiased bicubic. Native matches that fallback.",
                    compatibility="Config declaration shim + shipped text module registration; modern config maps layer_types to explicit per-layer windows/thetas. Model math is unchanged.", **public)
    (out / "fixture.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"fixture": str(out / "fixture.json"), "task": task_name, "checkpoints": len(traces), "input_tokens": len(e["input_ids"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task", choices=("decision", "span", "box", "ocr"))
    selection.add_argument("--raw", action="store_true", help="real raw encoder with a synthetic head; no task adapter required")
    parser.add_argument("--threads", type=int, default=1)
    run(parser.parse_args())
