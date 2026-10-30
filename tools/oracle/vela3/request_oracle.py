#!/usr/bin/env python3
"""Run the pinned official d3 Vela System One entry point, without editing it."""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys
from pathlib import Path


def load(path: Path):
    name = "vela_request_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threads", default=1, type=int)
    args = parser.parse_args()
    os.environ.setdefault("HF_MODULES_CACHE", str(args.out.parent.resolve() / "hf-modules"))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", D3_FAST="0")
    import torch
    torch.set_num_threads(args.threads)
    model_path, base_path = args.model.resolve(), args.base.resolve()
    sys.path.insert(0, str(base_path))
    executed_sources = {}
    original_exec_module = importlib.machinery.SourceFileLoader.exec_module

    def recording_exec_module(loader, module):
        path = Path(loader.path).resolve()
        if path.suffix == ".py" and (module.__name__.startswith("transformers_modules.") or
                any(path.is_relative_to(root) for root in (model_path, base_path))):
            entry = executed_sources.setdefault(str(path), dict(
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), modules=[]))
            entry["modules"].append(module.__name__)
        return original_exec_module(loader, module)

    importlib.machinery.SourceFileLoader.exec_module = recording_exec_module
    model = load(model_path / "modeling_vela3.py").Vela3MiniModel.from_pretrained(
        str(model_path), base_path=str(base_path), device="cpu", batch_size=1,
        verify="none", local_files_only=True,
    )
    request = json.loads(args.request.read_text())
    rt = model.runtime
    decisions = {k: q for k, q in request["questions"].items()
                 if not isinstance(q, dict) or q.get("type") != "span"}
    prepared = rt.prepare(request.get("state"), decisions, request.get("images"))
    decision_tokens = {k: prepared.sequences[k] for k in prepared.runnable}
    span_records = []
    adapter_calls, definition_generations = [], []
    original_active = model._A.active

    def recording_active(name, rows=None):
        adapter_calls.append(dict(adapter=name, rows=None if rows is None else rows.tolist()))
        return original_active(name, rows)

    model._A.active = recording_active
    original_generate = model.defs.generate

    def recording_generate(prompts):
        generated = original_generate(prompts)
        definition_generations.append(dict(
            writer=getattr(model.defs, "writer", None), prompts=prompts, definitions=generated))
        return generated

    model.defs.generate = recording_generate

    class RecordingRender:
        def __init__(self, original):
            self.original = original

        def __call__(self, parts, target, question, labels, text, **kwargs):
            rec = self.original(parts, target, question, labels, text, **kwargs)
            span_records.append(dict(parts=parts, target=target, question=question,
                labels=labels, text=text, images=kwargs.get("n_images", 0),
                ids=rec["ids"], word_positions=rec["w_pos"].tolist(),
                word_offsets=rec["w_offs"].tolist(), label_ranges=rec["ranges"]))
            return rec

    for name in ("span", "span_ner", "span_rel", "span_img"):
        engine = getattr(model, name, None)
        if engine is not None:
            engine.R = RecordingRender(engine.R)
    answer = model.system_one(**request)
    importlib.machinery.SourceFileLoader.exec_module = original_exec_module
    sources = {}
    for root in (model_path, base_path):
        for path in sorted(root.glob("*.py")):
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    import transformers
    result = dict(request=request, response=answer, decision_tokens=decision_tokens,
                  span_records=span_records, adapter_calls=adapter_calls,
                  definition_generations=definition_generations, provenance=dict(
        model=str(model_path), base=str(base_path), torch=torch.__version__,
        transformers=transformers.__version__, threads=args.threads, sources=sources,
        executed_source_modules=executed_sources, environment={"D3_FAST": os.environ["D3_FAST"]},
        vela_config=json.loads((model_path / "vela3_config.json").read_text()),
    ))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(answer, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
