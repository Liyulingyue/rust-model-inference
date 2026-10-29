#!/usr/bin/env python3
"""Pinned official CPU/F32 oracle. No copied/reimplemented model arithmetic.

Creates exact preprocessing/token/page fixtures, or runs actual checkpoint inference
and compares the Rust parity-trace stream (including checkpoint order and shape).
"""
import argparse
import hashlib
import importlib
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

REVISION = "7eadb7cd7afe660602040524eab37c8a9b91fe53"
SOURCE_SHA256 = {
    "config.json": "607d130a70d5af5a35329aa237acd68651f45504c62cc7ced981066f4e781126",
    "_sparse_base_impl.py": "86da9c5638704bb9338575040568a97b3e10820bbeb426f51c07e866f3423ccd",
    "modeling_unlimitedocr.py": "f6df813cef23cc28b1daca40fd8cb20740d05a238e7b9dfa2334531fdcf39f1f",
    "modeling_deepseekv2.py": "24a15dfd47578ab24a7a4d4e977dc5d7a23dfef0a2a4c4bf73fcf3e2ce7acaf2",
    "deepencoder.py": "419b63388c1645b9b72c716070cdca36cd0be6047049424dc75941d3a00aaf1f",
    "dash_attention.py": "960b460acd816df8e561e73dccd5683acb95445eddc723c4f0d6ed1f462b2f77",
    "tokenizer.json": "c55c165e32c5a67830362a53e1f548c8930dcc8469efa30ff3252a982799c589",
    "configuration_deepseek_v2.py": "b8470dd616ba8745fce6e27b093aef73a098863cc891b2477dcf9326a36000f7",
    "packing_patch.py": "e60d842c210c9f1d6cd2ae328cf9127a75f7941a0cea2a8e9abc14662f770c6c",
    "conversation.py": "ec7b6ce89bcda643de1f43269ffa66a7b2e65dc3ed30e427958f776546b4ba03",
    "tokenizer_config.json": "ceba7d221f265cca4c98b5e0a1d48c10107270b7b29c12396610d6d0fd64b126",
    "special_tokens_map.json": "9e89493d7938e7710c524dbaff613c3648f0a018262417af78505128004e5cb4",
}
WEIGHT_SHA256 = {
    "model-00001-of-00002.safetensors": "4c40b740f209fd7682040d4ebf19409208acde5c401ccd2dde426d0f41542f18",
    "model-00002-of-00002.safetensors": "9490b82fa47f89e5110eb5577f8ab1b6694c12121905dbc086ddb10207a4ba0c",
}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def reference(root):
    for name, sha in SOURCE_SHA256.items():
        if digest(root / name) != sha:
            raise ValueError(f"Oracle source differs from pinned {REVISION}: {name}")
    package = types.ModuleType("uocr_reference")
    package.__path__ = [str(root.resolve())]
    sys.modules[package.__name__] = package
    return importlib.import_module("uocr_reference.modeling_unlimitedocr")


def primitives(mod, root, output):
    import torch
    from transformers import AutoTokenizer
    decoder = importlib.import_module("uocr_reference.modeling_deepseekv2")
    base = importlib.import_module("uocr_reference._sparse_base_impl")
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    cfg = mod.UnlimitedOCRConfig.from_pretrained(root, local_files_only=True)
    idx = base.DeepseekV2Indexer(cfg, 0)
    base._PAGE_SCOPE_PREFILL = True
    ids = [0, 128827] + [128815] * 273 + [128827] + [128815] * 273 + tokenizer.encode("Multi page parsing.", add_special_tokens=False)
    coords = idx._build_page_rope_pos(torch.tensor([ids]), len(ids))
    scores = torch.tensor([[(i * 7 % 31) / 17.0 - 0.9 for i in range(25)]], dtype=torch.float32)[:, None]
    allowed = torch.tensor([[[i % 3 != 0 for i in range(25)]]])
    params = dict(idx_scale=128.0 / (128 ** 0.5), floor=0.0001, sigma=1.0, K=12, topm=16)
    indices, bias, valid = decoder._infer_route_select(scores, allowed, params)
    rope_coords = torch.tensor([[0, 0], [273, 1], [549, 2], [4097, 5]])
    ones = torch.ones(1, 1, len(rope_coords), 128)
    cos, sin = idx.rotary_emb_2d(ones, rope_coords[:, 0][None], rope_coords[:, 1][None])
    rotated, _ = decoder._llama_apply_rotary_pos_emb(ones, ones, cos, sin)
    result = {
        "revision": REVISION,
        "tokenizer": [{"text": text, "ids": tokenizer.encode(text, add_special_tokens=False)} for text in ["", "Multi page parsing.", "中文 OCR αβ\n\t 42", "a  b\n\n", "<PAGE>", "<image>"]],
        "prompt_token_count": len(ids),
        "coordinate_samples": {str(i): [int(coords[0][0, i]), int(coords[1][0, i])] for i in [0, 1, 2, 274, 275, 276, 548, 549, 552]},
        "route": {"scores": scores.flatten().tolist(), "allowed": allowed.flatten().tolist(), "indices": indices.flatten().tolist(), "bias": bias.flatten().tolist(), "valid": valid.flatten().tolist()},
        "rope": {"coords": rope_coords.tolist(), "ones_rotated": rotated.flatten().tolist()},
    }
    import numpy as np
    from PIL import Image
    pixels = np.array([[(x * 11 % 256, y * 15 % 256, (x * 7 + y * 13) % 256) for x in range(23)] for y in range(17)], dtype=np.uint8)
    _, imgs, _, _ = mod._uocr_build_inputs(tokenizer, "<PAGE><image>Multi page parsing.", [Image.fromarray(pixels)], crop_mode=False)
    raw = imgs[1][0].float().permute(1, 2, 0).contiguous().numpy().astype("<f4").tobytes()
    result["pixel_sha256"] = hashlib.sha256(raw).hexdigest()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


class Trace:
    def __init__(self, root):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / "trace.jsonl").exists():
            raise FileExistsError(f"Use a fresh oracle output directory: {root}")
        self.counts = defaultdict(int)
        self.rows = []

    def add(self, name, layer, x):
        import numpy as np
        values = x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=np.float32)
        values = np.ascontiguousarray(values, dtype="<f4")
        occurrence = self.counts[name]
        self.counts[name] += 1
        path = self.root / f"{name}.{occurrence}.f32"
        values.tofile(path)
        row = dict(name=name, layer=layer, shape=list(values.shape), occurrence=occurrence, binary_path=str(path.resolve()))
        self.rows.append(row)
        with (self.root / "trace.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(name, layer, values.shape, flush=True)


def run(mod, root, output, images, max_tokens, threads, instruction="Multi page parsing.", prefill_only=False):
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_utils import no_init_weights
    torch.set_num_threads(threads)
    for name, sha in WEIGHT_SHA256.items():
        if digest(root / name) != sha:
            raise ValueError(f"Wrong real checkpoint: {name}")
    cfg = mod.UnlimitedOCRConfig.from_pretrained(root, local_files_only=True)
    cfg._attn_implementation = "eager"
    with no_init_weights():
        model = mod.UnlimitedOCRForCausalLM(cfg).eval()
    loaded = set()
    for name in WEIGHT_SHA256:
        state = load_file(root / name)
        loaded.update(state)
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        if unexpected:
            raise ValueError(f"Unexpected checkpoint keys: {unexpected}")
    if loaded != set(model.state_dict()):
        raise ValueError(f"Missing checkpoint tensors: {set(model.state_dict()) - loaded}")
    model = model.float()
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    pil = [Image.open(path).convert("RGB") for path in images]
    prompt = "<PAGE><image>" * len(pil) + instruction
    if pil:
        ids, imgs, mask, spatial = mod._uocr_build_inputs(tokenizer, prompt, pil, crop_mode=False)
    else:
        ids = torch.tensor([0] + tokenizer.encode(instruction, add_special_tokens=False))
        imgs = ([], [])
    trace = Trace(output)
    trace.add("uocr.token_ids", None, ids)
    sam, clip = model.model.sam_model, model.model.vision_model
    sam.patch_embed.register_forward_hook(lambda _, args, value: trace.add("uocr.sam_embed", None, (value + sam.pos_embed)[0]))
    for i, layer in enumerate(sam.blocks):
        layer.register_forward_hook(lambda _, args, value, i=i: trace.add("uocr.sam_layer", i, value[0]))
    sam.register_forward_hook(lambda _, args, value: trace.add("uocr.sam", None, value[0].permute(1, 2, 0)))
    clip.pre_layrnorm.register_forward_hook(lambda _, args, value: trace.add("uocr.clip_embed", None, value[0]))
    for i, layer in enumerate(clip.transformer.layers):
        layer.register_forward_hook(lambda _, args, value, i=i: trace.add("uocr.clip_layer", i, value[0]))
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.o_proj.register_forward_pre_hook(lambda _, args, i=i: trace.add("uocr.attention", i, args[0][0]))
        layer.register_forward_hook(lambda _, args, value, i=i: trace.add("uocr.layer", i, value[0][0]))
    model.model.norm.register_forward_hook(lambda _, args, value: trace.add("uocr.norm", None, value[0]))
    cache = DynamicCache()
    gen = []
    guard = dict(marker_cum=0, scanned=0, page_len=0)
    mod.configure_dash_inference(model.model, True, topm=16)
    model.config._ring_window = None
    model.config.sliding_window = None
    with torch.inference_mode():
        hidden = model.model.embed_tokens(ids[None])
        for page, image in enumerate(imgs[1]):
            trace.add("uocr.pixels", page, image.float().permute(1, 2, 0))
            sf = sam(image[None].float())
            cf = clip(image[None].float(), sf)
            features = model.model.projector(torch.cat((cf[:, 1:], sf.flatten(2).permute(0, 2, 1)), dim=-1))[0].view(16, 16, -1)
            features = torch.cat((features, model.model.image_newline[None, None].expand(16, 1, -1)), dim=1).reshape(-1, 1280)
            features = torch.cat((features, model.model.view_seperator[None]), dim=0)
            trace.add("uocr.visual", None, features)
            start = 2 + page * 274
            hidden[0, start:start + 273] = features
        for step in range(max_tokens):
            xids = ids[None] if step == 0 else torch.tensor([[gen[-1]]])
            trace.add("uocr.embeddings", None, hidden[0])
            out = model.model(input_ids=xids, inputs_embeds=hidden, past_key_values=cache, use_cache=True,
                              position_ids=(torch.arange(len(ids))[None] if step == 0 else torch.tensor([[len(ids) + step - 1]])), return_dict=True)
            cache = out.past_key_values
            logits = model.lm_head(out.last_hidden_state[:, -1:]).float()[0, -1]
            trace.add("uocr.logits", None, logits)
            if prefill_only:
                break
            mod._uocr_page_count_guard(logits, gen, len(pil), 128827, 1, 20000, guard)
            next_id = int(logits.argmax())
            if next_id == 1:
                break
            gen.append(next_id)
            hidden = model.model.embed_tokens(torch.tensor([[next_id]]))
        trace.add("uocr.generated", None, gen)
    import platform
    import transformers
    result = dict(revision=REVISION, dtype="F32 compute from original BF16 storage", threads=threads,
                  platform=platform.platform(), torch_version=torch.__version__, transformers_version=transformers.__version__,
                  image_sha256={str(path): digest(path) for path in images}, max_tokens=max_tokens,
                  instruction=instruction, prefill_only=prefill_only,
                  prompt_ids=ids.tolist(), generated_ids=gen, text=tokenizer.decode(gen, skip_special_tokens=False),
                  source_sha256=SOURCE_SHA256, weight_sha256=WEIGHT_SHA256)
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


def validate_trace(rows, scope):
    position = 0

    def expect(name, layer=None):
        nonlocal position
        if position >= len(rows) or (rows[position]["name"], rows[position]["layer"]) != (name, layer):
            raise AssertionError(f"Incomplete trace at {position}: expected {name} layer={layer}")
        position += 1

    expect("uocr.token_ids")
    pages = 0
    while position < len(rows) and rows[position]["name"] == "uocr.pixels":
        expect("uocr.pixels", pages)
        expect("uocr.sam_embed")
        for layer in range(12):
            expect("uocr.sam_layer", layer)
        expect("uocr.sam")
        expect("uocr.clip_embed")
        for layer in range(24):
            expect("uocr.clip_layer", layer)
        expect("uocr.visual")
        pages += 1
    if not pages and scope != "decoder":
        raise AssertionError("Incomplete trace: full per-page vision stages required")
    steps = 0
    while position < len(rows) and rows[position]["name"] == "uocr.embeddings":
        expect("uocr.embeddings")
        for layer in range(12):
            expect("uocr.attention", layer)
            expect("uocr.layer", layer)
        expect("uocr.norm")
        expect("uocr.logits")
        steps += 1
    expect("uocr.generated")
    if not steps or position != len(rows) or steps not in {max(1, rows[-1]["shape"][0]), rows[-1]["shape"][0] + 1}:
        raise AssertionError("Incomplete trace: decoder steps do not cover the generated sequence")
    occurrences = defaultdict(int)
    for row in rows:
        if row["occurrence"] != occurrences[row["name"]]:
            raise AssertionError(f"Wrong checkpoint occurrence: {row['name']}")
        occurrences[row["name"]] += 1


def compare(oracle, rust, output, scope="all"):
    import numpy as np
    a = [json.loads(x) for x in oracle.read_text().splitlines()]
    b = [json.loads(x) for x in rust.read_text().splitlines()]
    # Route traces are optional diagnostics, independently verified by the primitive fixture.
    b = [x for x in b if x["name"] not in {"uocr.routes", "uocr.moe_gate"}]
    for rows in [a, b]:
        validate_trace(rows, scope)
    if scope != "all":
        vision = {"uocr.pixels", "uocr.sam_embed", "uocr.sam_layer", "uocr.sam", "uocr.clip_embed", "uocr.clip_layer", "uocr.visual"}
        keep = lambda row: row["name"] in {"uocr.token_ids", "uocr.generated"} or ((row["name"] in vision) == (scope == "vision"))
        a, b = [row for row in a if keep(row)], [row for row in b if keep(row)]
        if scope == "vision" and not any(row["name"] == "uocr.visual" for row in a):
            raise AssertionError("Missing visual checkpoints")
    if [(x["name"], x["layer"], x["shape"], x["occurrence"]) for x in a] != [(x["name"], x["layer"], x["shape"], x["occurrence"]) for x in b]:
        raise AssertionError("Checkpoint order/count/shape differs")
    records = []
    for left, right in zip(a, b):
        x = np.fromfile(left["binary_path"], dtype="<f4")
        y = np.fromfile(right["binary_path"], dtype="<f4")
        if x.size != int(np.prod(left["shape"])) or y.size != x.size:
            raise AssertionError(f"Wrong checkpoint payload length: {left['name']}")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise AssertionError(f"Non-finite {left['name']}")
        exact = left["name"] in {"uocr.token_ids", "uocr.pixels", "uocr.generated"}
        diff = np.abs(x.astype(np.float64) - y)
        scale = max(float(np.max(np.abs(x), initial=0)), 1e-12)
        relative = float(np.max(diff, initial=0)) / scale
        norm = float(np.linalg.norm(x.astype(np.float64)) * np.linalg.norm(y.astype(np.float64)))
        cosine = 1.0 if not x.any() and not y.any() else float(np.dot(x.astype(np.float64), y) / max(norm, 1e-30))
        passed = np.array_equal(x.view("<u4"), y.view("<u4")) if exact else relative <= 0.002 and cosine >= 0.99999
        if not passed:
            raise AssertionError(f"First differing checkpoint: {left['name']} layer={left['layer']} occurrence={left['occurrence']} max_rel={relative} cosine={cosine}")
        records.append(dict(name=left["name"], layer=left["layer"], occurrence=left["occurrence"], bit_exact=bool(np.array_equal(x.view("<u4"), y.view("<u4"))), max_abs=float(np.max(diff, initial=0)), max_relative_to_peak=relative, cosine=cosine))
    output.write_text(json.dumps(dict(scope=scope, checkpoints=len(records), passed=True, records=records), indent=2) + "\n")
    print(f"PASS ({scope}): {len(records)} checkpoints; lossless transforms/tokens exact; floating point <=0.002 peak-relative and cosine >=0.99999")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--primitives", action="store_true")
    p.add_argument("--images", type=Path, nargs="+")
    p.add_argument("--instruction", type=Path, help="UTF-8 instruction file; page markers are automatic")
    p.add_argument("--prefill-only", action="store_true", help="record final logits without sampling")
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--compare", type=Path, nargs=2, metavar=("ORACLE_TRACE", "RUST_TRACE"))
    p.add_argument("--scope", choices=["all", "vision", "decoder"], default="all", help="comparison scope; bounds are identical for each scope")
    args = p.parse_args()
    if not args.compare and (args.model_root is None or (not args.primitives and not args.images and not (args.prefill_only and args.instruction))):
        p.error("--model-root and --images are required for real inference")
    if args.max_tokens <= 0 or args.threads <= 0:
        p.error("--max-tokens and --threads must be positive")
    if args.compare:
        compare(*args.compare, args.output, args.scope)
    else:
        mod = reference(args.model_root)
        if args.primitives:
            primitives(mod, args.model_root, args.output)
        else:
            run(mod, args.model_root, args.output, args.images or [], args.max_tokens, args.threads,
                args.instruction.read_text() if args.instruction else "Multi page parsing.", args.prefill_only)


if __name__ == "__main__":
    main()
