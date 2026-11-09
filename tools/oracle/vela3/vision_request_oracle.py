#!/usr/bin/env python3
"""Run the pinned fast-vision public System One wrapper, including its text side.

All weights are local. The existing vision config compatibility shim is reused;
the original public wrapper, text runtime, task modules and model math are unchanged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", default=[], help="override request images with local files")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel, GPT2Tokenizer
    from transformers.models.idefics3.processing_idefics3 import Idefics3Processor
    from vision_oracle import official_classes, load_module

    root = args.root.resolve()
    snapshot = json.loads(Path(__file__).with_name("snapshots.json").read_text(encoding="utf-8"))["vela3-fast-vision"]
    for name, expected in snapshot["source_sha256"].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"source snapshot changed: {name}")
    text_files = [entry for entry in snapshot["files"] if entry["Path"].startswith("text/")]
    text_weights = [entry for entry in text_files if entry["Path"].endswith(".safetensors")]
    if not text_weights:
        raise ValueError("the snapshot must pin the public wrapper's text weights")
    for entry in text_files:
        path = root / entry["Path"]
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if path.stat().st_size != entry["Size"] or actual != entry["Sha256"]:
            raise ValueError(f"text snapshot changed: {path}")

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    Config, Model, Processor, core = official_classes(root)
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    task_cfg = json.loads((root / "vela3v_config.json").read_text(encoding="utf-8"))
    processor_cfg = json.loads((root / "processor_config.json").read_text(encoding="utf-8"))["image_processor"]
    for edge in {int(task["edge"]) for task in task_cfg["tasks"].values()}:
        proc = Idefics3Processor(Processor(**processor_cfg), GPT2Tokenizer.from_pretrained(root), image_seq_len=64)
        proc.image_processor.size = {"longest_edge": edge}
        proc.image_processor.do_image_splitting = edge > 512
        core.Proc._cache[(str(root), edge)] = proc

    def load_base(path, **_kwargs):
        # Load the same model/weights with the existing config shim, avoiding newer config imports.
        if Path(path).resolve() != root:
            raise ValueError(f"unexpected backbone: {path}")
        model = Model(Config(**cfg)).eval()
        model.load_state_dict({k: v.float() for k, v in load_file(str(root / "model.safetensors")).items()}, strict=True)
        return model

    public = load_module("vela3_encoder", root / "vela3_encoder.py")
    with patch.object(AutoModel, "from_pretrained", side_effect=load_base):
        model = public.Vela3Encoder.load(root, device="cpu")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if args.image:
        request["images"] = [str(path.resolve()) for path in args.image]
    response = model.system_one(request.get("state"), request["questions"], images=request.get("images"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n")
    args.out.with_suffix(".request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    metadata = {"verified_at_utc": datetime.now(timezone.utc).isoformat(),
                "platform": platform.platform(), "threads": args.threads, "source_root": str(root),
                "torch": torch.__version__, "dtype": "float32", "attention": "eager",
                "request": request, "source_sha256": snapshot["source_sha256"],
                "text_weights": text_weights,
                "compatibility": "Existing vision config shim for AutoModel construction; unmodified public/text/task source and calculations."}
    args.out.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "usage": response["usage"],
                      "heads": {key: answer.get("head", answer.get("error", "decision"))
                                for key, answer in response["answers"].items()}}))


if __name__ == "__main__":
    main()
