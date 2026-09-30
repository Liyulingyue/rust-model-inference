"""Trace the pinned GLiNER2 official graph on CPU, optionally with scalar kernels."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import transformers
from gliner2 import AutoExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--scalar", action="store_true")
    arguments = parser.parse_args()
    request = json.loads(arguments.request.read_text())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = AutoExtractor.from_pretrained(arguments.model, use_flashdeberta=False)
    model.eval().cpu()
    if arguments.scalar:
        from tools.oracle.gliner.scalar import patch_model

        patch_model(model, arguments.trace)
    traces = []

    counts = {}
    classifier_outputs = []

    def capture(name):
        def hook(_module, _input, output):
            occurrence = counts.get(name, 0)
            counts[name] = occurrence + 1
            if occurrence and name in ("gliner.q", "gliner.k", "gliner.v"):
                return
            values = output[0] if isinstance(output, tuple) else output
            values = values.detach().contiguous().cpu().float().numpy()
            if name == "gliner.logits":
                classifier_outputs.append(values.copy())
                return
            raw = Path(f"{arguments.trace}.{name}.f32")
            raw.write_bytes(values.tobytes())
            traces.append({"name": name, "shape": list(values.shape), "binary_path": str(raw)})
        return hook

    handles = [model.encoder.embeddings.register_forward_hook(capture("gliner.embeddings"))]
    first = model.encoder.encoder.layer[0].attention.self
    for name in ("query_proj", "key_proj", "value_proj"):
        handles.append(getattr(first, name).register_forward_hook(capture({"query_proj": "gliner.q", "key_proj": "gliner.k", "value_proj": "gliner.v"}[name])))
    handles.append(first.register_forward_hook(capture("gliner.context")))
    handles += [layer.register_forward_hook(capture(f"gliner.layer.{index}")) for index, layer in enumerate(model.encoder.encoder.layer)]
    handles += [model.classifier.register_forward_hook(capture("gliner.logits"))]
    with torch.inference_mode():
        schema = model._classification_schema(request["tasks"])
        batch = model.processor.collate_fn_inference([(request["text"], schema)], error_policy="raise")
        token_ids = batch.input_ids[0, :batch.original_lengths[0]].tolist()
        result = model.classify_text(request["text"], request["tasks"], include_confidence=request.get("include_confidence", False))
    for handle in handles:
        handle.remove()
    if classifier_outputs:
        import numpy as np

        values = np.concatenate(classifier_outputs, axis=0)
        raw = Path(f"{arguments.trace}.gliner.logits.f32")
        raw.write_bytes(values.tobytes())
        traces.append({"name": "gliner.logits", "shape": list(values.shape), "binary_path": str(raw)})
    arguments.trace.write_text(json.dumps({"oracle": "fastino-ai/GLiNER2@55656fbfa01d3d4a77485e1a1eeeaf682990ccdf", "torch": torch.__version__, "transformers": transformers.__version__, "scalar_kernels": arguments.scalar, "token_ids": token_ids, "result": result, "checkpoints": traces}, indent=2))
    print(json.dumps({"token_ids": token_ids, "result": result, "trace": str(arguments.trace)}))


if __name__ == "__main__":
    main()
