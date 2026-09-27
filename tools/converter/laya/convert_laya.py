"""Convert the pinned Laya multilingual checkpoint to lossless F32 GGUF."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools.converter.utils.gguf import GgufWriter, GGML_F32, gguf_dims, open_safetensors


def validate_config(encoder, agent):
    expected = dict(model_type="modernbert", hidden_size=768, num_hidden_layers=22,
                    num_attention_heads=12, intermediate_size=1152, vocab_size=256000,
                    local_attention=128, max_position_embeddings=8192,
                    layer_norm_eps=1e-5, global_attn_every_n_layers=3,
                    attention_bias=False, mlp_bias=False, norm_bias=False,
                    hidden_activation="gelu")
    for key, value in expected.items():
        if encoder.get(key) != value:
            raise ValueError(f"Unsupported encoder {key}: {encoder.get(key)!r}; expected {value!r}")
    if encoder.get("layer_types") != ["full_attention" if i % 3 == 0 else "sliding_attention"
                                      for i in range(22)]:
        raise ValueError("Unsupported encoder layer_types")
    for kind in ["full_attention", "sliding_attention"]:
        if encoder.get("rope_parameters", {}).get(kind, {}).get("rope_theta") != 160000:
            raise ValueError(f"Unsupported {kind} RoPE")
    if agent.get("encoder") != "jhu-clsp/mmBERT-base" or agent.get("head_layers") != 2:
        raise ValueError("Expected multilingual encoder and two head_layers")
    if agent.get("act_costs") != {"escalate": .5}:
        raise ValueError("Unsupported act_costs")
    if not 1 <= agent.get("max_len", 0) <= 8192 or not 1 <= agent.get("head_max_len", 0) <= agent["max_len"]:
        raise ValueError("Invalid sequence budgets")
    if len(agent.get("temperature", [])) != 3:
        raise ValueError("Expected three temperatures")


def tensor_contracts(encoder, agent):
    d, f = encoder["hidden_size"], encoder["intermediate_size"]
    shapes = {"temperature": (3,), "type_emb.weight": (3, d),
              "encoder.embeddings.tok_embeddings.weight": (encoder["vocab_size"], d),
              "encoder.embeddings.norm.weight": (d,), "encoder.final_norm.weight": (d,)}
    for i in range(encoder["num_hidden_layers"]):
        p = f"encoder.layers.{i}."
        shapes.update({p+"attn.Wqkv.weight": (3*d, d), p+"attn.Wo.weight": (d, d),
                       p+"mlp_norm.weight": (d,), p+"mlp.Wi.weight": (2*f, d),
                       p+"mlp.Wo.weight": (d, f)})
        if i:
            shapes[p+"attn_norm.weight"] = (d,)
    for i in range(agent["head_layers"]):
        p = f"head.layers.{i}."
        for name, out, inp in [("self_attn.in_proj", 3*d, d), ("self_attn.out_proj", d, d),
                               ("linear1", 4*d, d), ("linear2", d, 4*d)]:
            separator = "_" if name == "self_attn.in_proj" else "."
            shapes[p+name+separator+"weight"] = (out, inp)
            shapes[p+name+separator+"bias"] = (out,)
        for norm in ["norm1", "norm2"]:
            shapes[p+norm+".weight"] = (d,)
            shapes[p+norm+".bias"] = (d,)
    shapes.update({"scorer.0.weight": (d,), "scorer.0.bias": (d,),
                   "scorer.1.weight": (d, d), "scorer.1.bias": (d,),
                   "scorer.3.weight": (1, d), "scorer.3.bias": (1,),
                   "act_head.0.weight": (256, d+4), "act_head.0.bias": (256,),
                   "act_head.2.weight": (2, 256), "act_head.2.bias": (2,)})
    return shapes


def convert(model_dir: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    encoder = json.loads((model_dir / "encoder/config.json").read_text())
    agent = json.loads((model_dir / "rl_agent_config.json").read_text())
    validate_config(encoder, agent)
    source = open_safetensors(model_dir / "model.safetensors")
    contracts = tensor_contracts(encoder, agent)
    if set(source.header) - {"__metadata__"} != set(contracts):
        raise ValueError("Checkpoint tensor names do not match the Laya contract")
    for name, shape in contracts.items():
        info = source.header[name]
        a, b = info["data_offsets"]
        size = {"F16": 2, "F32": 4}.get(info["dtype"])
        if tuple(info["shape"]) != shape or size is None or b-a != math.prod(shape)*size or a < 0 or source.data_offset+b > source.file_size:
            raise ValueError(f"Invalid tensor {name}: {info}")
    tokenizer = (model_dir / "tokenizer/tokenizer.json").read_text()
    json.loads(tokenizer)
    writer = GgufWriter(output)
    writer.add_meta("general.architecture", "laya")
    writer.add_meta("general.name", "laya-multilingual")
    writer.add_meta("laya.protocol_version", 1)
    writer.add_meta("laya.encoder_config", json.dumps(encoder))
    writer.add_meta("laya.agent_config", json.dumps(agent))
    writer.add_meta("laya.tokenizer_json", tokenizer)
    with source.path.open("rb") as stream:
        writer.add_meta("laya.source_sha256", hashlib.file_digest(stream, "sha256").hexdigest())

    def chunks(name):
        info = source.header[name]
        start, end = info["data_offsets"]
        dtype = "<f2" if info["dtype"] == "F16" else "<f4"
        with source.path.open("rb") as stream:
            stream.seek(source.data_offset+start)
            remaining = end-start
            while remaining:
                raw = stream.read(min(remaining, 4*1024*1024))
                if not raw:
                    raise ValueError(f"Truncated tensor {name}")
                values = np.frombuffer(raw, dtype=dtype).astype("<f4")
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite tensor {name}")
                yield values.tobytes()
                remaining -= len(raw)

    for name, shape in contracts.items():
        writer.add_tensor_chunks(name, GGML_F32, gguf_dims(shape), math.prod(shape)*4,
                                 lambda name=name: chunks(name))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    writer.path = temporary
    try:
        writer.write()
        temporary.rename(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.model_dir, args.output)
