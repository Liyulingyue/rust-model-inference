#!/usr/bin/env python3
"""Separate accumulated FP16 state drift from same-input local block error."""
import argparse
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--oracle-source", required=True, type=Path)
parser.add_argument("--models", required=True, type=Path)
parser.add_argument("--fixtures", required=True, type=Path)
parser.add_argument("--report", type=Path)
parser.add_argument("--threads", type=int, default=8)
args = parser.parse_args()
if args.threads < 1:
    parser.error("threads must be positive")
import os
os.environ['ATTN_BACKEND'] = 'sdpa'
import json
import sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from compare import records as validated_records
from oracle import COMMIT, sha256, validate_source
ART = args.fixtures.resolve()
MODELS = args.models.resolve()
SOURCE = args.oracle_source.resolve()
validate_source(SOURCE)
metadata = json.loads((ART / "fixtures.json").read_text())
if (metadata["oracle_commit"] != COMMIT
        or metadata["models"]["idol/denoiser.safetensors"]["sha256"] != sha256(MODELS / "idol/denoiser.safetensors")
        or metadata["models"]["idol/config.json"]["sha256"] != sha256(MODELS / "idol/config.json")
        or metadata["trace_sha256"] != sha256(ART / "oracle.jsonl")):
    raise ValueError("Source, weights or oracle trace differs from fixture provenance")
sys.path.insert(0, str(SOURCE))
from idol.models import LatentSeqMMFlowModel

torch.set_num_threads(args.threads)
torch.set_float32_matmul_precision('highest')
config = json.loads((MODELS / 'idol/config.json').read_text())
model = LatentSeqMMFlowModel(**config['denoiser']['args'])
model.load_state_dict(load_file(str(MODELS / 'idol/denoiser.safetensors')), strict=True)
model.eval().requires_grad_(False)

def records(kind):
    rows = validated_records(ART / f'{kind}.jsonl')
    if any(row["occurrence"] != 0 for row in rows):
        raise ValueError("Denoiser diagnostic requires exactly one occurrence per checkpoint")
    return {row["name"]: row for row in rows}
RECORDS = {kind: records(kind) for kind in ['oracle', 'native']}
identities = lambda rows: [(row["name"], row["shape"]) for row in rows.values()]
if identities(RECORDS["oracle"]) != identities(RECORDS["native"]):
    raise ValueError("Oracle/native diagnostic checkpoint name, order or shape differs")
def value(kind, name):
    r = RECORDS[kind]['idol.denoiser.' + name]
    a = np.array(r['values'], dtype=np.float32).reshape(r['shape'])
    return torch.from_numpy(a)
def diff(a, b):
    a, b = a.float().double().reshape(-1), b.float().double().reshape(-1)
    e = a - b
    index = int(e.abs().argmax())
    rms = float(e.square().mean().sqrt())
    return dict(max_abs=float(e.abs().max()), rmse=rms,
                relative_rmse=rms / max(float(b.square().mean().sqrt()), 1e-30),
                index=index, actual=float(a[index]), reference=float(b[index]),
                nonzero=int((e != 0).sum()))

def get_input(kind, group, index):
    if index:
        return value(kind, f'{group}.{index-1}.out').half()
    if group == 'noise':
        return value(kind, 'position_added').half()
    if group == 'context':
        return value(kind, 'condition').half()
    return torch.cat([value(kind, 'noise.1.out'), value(kind, 'context.1.out'),
                      value(kind, 'camera.1').half()], dim=1).half()

results = []
selected = [('noise', 0), ('noise', 1), ('context', 0), ('context', 1)]
selected += [('joint', i) for i in [0, 1, 12, 15, 19, 20, 21, 22, 23]]
with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
    for group, index in selected:
        if group == 'noise': blocks, repos = model.noise_refiner, model.noise_repo_layers
        elif group == 'context': blocks, repos = model.context_refiner, model.context_repo_layers
        else: blocks, repos = model.blocks, model.repo_layers
        block, repo = blocks[index], repos[index]
        name = f'{group}.{index}.out'
        native_in = get_input('native', group, index)
        oracle_in = get_input('oracle', group, index)
        native_mod = None if group == 'context' else value('native', 'modulation').reshape(1, -1).half()
        oracle_mod = None if group == 'context' else value('oracle', 'modulation').reshape(1, -1).half()
        fresh_native = block(native_in, native_mod, repo(native_in))
        fresh_oracle = block(oracle_in, oracle_mod, repo(oracle_in))
        native_out = value('native', name)
        oracle_out = value('oracle', name)
        worst = (native_out - oracle_out).abs().reshape(-1).argmax()
        result = dict(name=name,
            reconstruction=diff(fresh_oracle, oracle_out),
            local=diff(native_out, fresh_native),
            propagated=diff(fresh_native, oracle_out),
            total=diff(native_out, oracle_out),
            worst_total_index=int(worst),
            local_at_worst_total=float((native_out-fresh_native).reshape(-1)[worst]),
            propagated_at_worst_total=float((fresh_native-oracle_out).reshape(-1)[worst]))
        if result["reconstruction"]["max_abs"] != 0:
            raise ValueError(f"Official block does not reproduce saved oracle at {name}")
        print(f"{name}: local_max={result['local']['max_abs']:.6g}, total_max={result['total']['max_abs']:.6g}", flush=True)
        results.append(result)
report = args.report or ART / "denoiser-drift-diagnostic.json"
report.write_text(json.dumps(dict(oracle_commit=COMMIT, seed=metadata["seed"], torch=torch.__version__,
                                 oracle_trace_sha256=metadata["trace_sha256"], native_trace_sha256=sha256(ART / "native.jsonl"),
                                 checkpoints=results), indent=2)+"\n")
