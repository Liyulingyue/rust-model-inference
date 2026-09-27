"""Trace the pinned official Laya graph with scalar F32 kernels; compare raw bits."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

Laya_COMMIT = "4066d5d5fbf08b66c6757ddeedbd797bd7655bc0"


def compare(oracle, native):
    left = [json.loads(s) for s in Path(oracle).read_text().splitlines()]
    right = [json.loads(s) for s in Path(native).read_text().splitlines()]
    if len(left) != len(right):
        raise AssertionError(f"checkpoint counts differ: {len(left)} != {len(right)}")
    for a, b in zip(left, right):
        if (a["name"], a["shape"]) != (b["name"], b["shape"]):
            raise AssertionError(f"checkpoint contract differs: {a} != {b}")
        if "token_ids" in a:
            if a["token_ids"] != b["token_ids"]:
                raise AssertionError(f"{a['name']}: token IDs differ")
            continue
        if "values" in a:
            if a["values"] != b.get("values", b.get("usize_values")):
                raise AssertionError(f"{a['name']}: marker positions differ")
            continue
        x = np.fromfile(a["binary_path"], dtype="<u4")
        y = np.fromfile(b["binary_path"], dtype="<u4")
        if a["occurrence"] != b["occurrence"]:
            raise AssertionError(f"{a['name']}: checkpoint occurrences differ")
        if x.size != y.size or x.size != int(np.prod(a["shape"])):
            raise AssertionError(f"{a['name']}: binary sizes differ")
        mismatch = np.flatnonzero(x != y)
        if mismatch.size:
            i = int(mismatch[0])
            raise AssertionError(f"first divergence {a['name']} occurrence={a['occurrence']} "
                                 f"index={i}: oracle=0x{int(x[i]):08x}, native=0x{int(y[i]):08x}; "
                                 f"{mismatch.size}/{x.size} differing values")
    print(f"PASS: {len(left)} checkpoints, token IDs, markers and every F32 bit match")


def trace(model_dir, source_dir, request_file, output):
    commit = subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True).strip()
    if commit != Laya_COMMIT:
        raise ValueError(f"Expected official source {Laya_COMMIT}, got {commit}")
    dirty = subprocess.check_output(["git", "-C", str(source_dir), "status", "--porcelain"], text=True)
    if dirty:
        raise ValueError("Oracle source must be clean")
    sys.path.insert(0, str(source_dir.resolve()))
    import torch
    from scalar import Scalar, FLAGS
    from laya import Agent
    from laya.common import collate_items
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.mkldnn.enabled = False
    agent = Agent(str(model_dir.resolve()), device="cpu", fast=False, compile=False)
    agent.model.float().eval()
    agent.amp_enabled = False
    scalar = Scalar(output)
    with torch.inference_mode(), scalar:
        scalar.reset_rope(agent.model.encoder)
    request = json.loads(request_file.read_text())
    ids = list(request["questions"])
    for qid in ids:
        agent._check_question(qid, request["questions"][qid])
    internal = {qid: agent._to_internal(request["questions"][qid]) for qid in ids}
    items = agent._encode_state(request["state"], ids, internal,
                                request.get("max_len"), request.get("head_max_len"))
    output.mkdir(parents=True, exist_ok=True)
    records, counts = [], collections.Counter()

    def emit(name, value):
        if isinstance(value, tuple):
            value = value[0]
        data = value.detach().float().cpu().numpy()
        shape = list(data.shape)
        if data.ndim >= 2 and shape[0] == 1:
            shape = shape[1:]
        occurrence = counts[name]
        counts[name] += 1
        path = output / f"{name}.{occurrence}.f32"
        data.astype("<f4").tofile(path)
        records.append(dict(name=name, shape=shape, occurrence=occurrence, binary_path=str(path.resolve())))

    def hook(name):
        return lambda module, inputs, result: emit(name, result)

    handles = []
    scalar.attention_trace = emit
    for name, module in [("laya.embedding", agent.model.encoder.embeddings.tok_embeddings),
                         ("laya.embedding_norm", agent.model.encoder.embeddings.norm),
                         ("laya.encoder_norm", agent.model.encoder.final_norm)]:
        handles.append(module.register_forward_hook(hook(name)))
    for i, layer in enumerate(agent.model.encoder.layers):
        handles.append(layer.register_forward_hook(hook(f"laya.encoder.{i}")))
    layer = agent.model.encoder.layers[0]
    for name, module in [("qkv", layer.attn.Wqkv), ("attn_out", layer.attn.Wo),
                         ("mlp_norm", layer.mlp_norm), ("mlp_in", layer.mlp.Wi),
                         ("mlp_out", layer.mlp.Wo)]:
        handles.append(module.register_forward_hook(hook(f"laya.encoder.0.{name}")))
    for i, layer in enumerate(agent.model.head.layers):
        handles.append(layer.register_forward_hook(hook(f"laya.head.{i}")))
    handles.append(agent.model.head.layers[0].register_forward_pre_hook(
        lambda module, inputs: emit("laya.typed_hidden", inputs[0])))
    answers = {}
    with torch.inference_mode(), scalar:
        for j, item in enumerate(items):
            records.append(dict(name="laya.input_ids", shape=[len(item["ids"])], token_ids=item["ids"]))
            records.append(dict(name="laya.markers", shape=[len(item["markers"])], values=item["markers"]))
            batch = collate_items([[item]], agent.tok.pad_token_id)
            logits, act = agent.model(**{k: batch[k] for k in
                ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]})
            emit("laya.logits", logits)
            emit("laya.act_logits", act)
            decoded = agent._decode_answers(logits.numpy(), torch.softmax(act, -1).numpy(),
                                             [item], [ids[j]], {ids[j]: internal[ids[j]]}, 0)
            answers.update(decoded)
    for h in handles:
        h.remove()
    (output / "trace.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    weights = model_dir / "model.safetensors"
    with weights.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "result.json").write_text(json.dumps({"answers": answers}, ensure_ascii=False, indent=2))
    (output / "environment.json").write_text(json.dumps(dict(laya_commit=commit, weights_sha256=digest,
        torch=torch.__version__, transformers=__import__("transformers").__version__,
        dtype="float32", device="cpu", threads=1, arithmetic="independent scalar C F32",
        compiler_flags=FLAGS, scalar_calls=scalar.calls, mha_fastpath=False), indent=2))
    print(json.dumps({"answers": answers}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("trace")
    t.add_argument("model_dir", type=Path)
    t.add_argument("source_dir", type=Path)
    t.add_argument("request", type=Path)
    t.add_argument("output", type=Path)
    c = sub.add_parser("compare")
    c.add_argument("oracle", type=Path)
    c.add_argument("native", type=Path)
    args = parser.parse_args()
    if args.command == "trace":
        trace(args.model_dir, args.source_dir, args.request, args.output)
    else:
        compare(args.oracle, args.native)
