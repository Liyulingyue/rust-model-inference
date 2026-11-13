#!/usr/bin/env python3
"""Independent pinned-checkpoint oracle; run on the weight host, never on the Mac.

F32 executes the official Transformers graph with one layer temporarily cast at
a time. This preserves BF16 checkpoint values without a second 100 GB model.
The optional official mode measures the published BF16 System One API instead.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vela3"))
from d3_oracle import Trace, load_module
from stream_f32 import require_bf16_checkpoint, stream_f32, _float_tensors


def media_prompt(root, runtime, backbone, model, request, question, config, trace, clips=None, emit_video=True):
    import torch
    dc = load_module(root / "d3_compose.py")
    processor = runtime.load_processor(root)
    runtime.configure_video_processor(processor)
    assets = request.get("assets") or {}
    refs = dc.check_references(list(assets))
    front = list(request.get("images") or [])
    state, hoisted = dc.hoist_state(request["state"], refs, len(front))
    front += [assets[ref] for ref in hoisted]
    if clips is None:
        clips = [runtime.load_video(v) for v in request.get("videos") or []]
    spatial = question["type"] in ("box", "track")
    normalized = None if spatial else runtime.normalize_question(question)
    parts = dc.user_parts(state, question, config["codes"], len(front), len(clips), refs)
    text = dc.render(processor, parts, dc.suffix(question["type"] == "track") if spatial else "")
    own = [assets[item] for kind, item in parts if kind == dc.IMAGE and isinstance(item, str)]
    kwargs = dict(images=[runtime.load_image(v) for v in front + own] or None)
    if clips:
        kwargs.update(videos=[c.frames for c in clips], video_metadata=[c.metadata() for c in clips])
    inputs = processor(text=[text], return_tensors="pt", **kwargs)
    ids = inputs["input_ids"]
    plain = processor.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    state_count = runtime.state_token_count(plain["offset_mapping"], runtime.state_end_char(text, state))
    grids = inputs.get("image_grid_thw")
    own_tokens = sum(int(g.prod()) // processor.image_processor.merge_size**2 - 1 for g in grids[len(front):]) if grids is not None else 0
    state_count += ids.numel() - len(plain["input_ids"]) - own_tokens
    embeds = model.embed_tokens(ids)
    runtime.linearize_patch_embed(backbone)
    with _float_tensors(backbone.visual):
        if inputs.get("pixel_values") is not None:
            pooled = backbone.get_image_features(inputs["pixel_values"].float(), inputs["image_grid_thw"], return_dict=True).pooler_output
            for value in pooled:
                trace.emit("vela3.d3.vision.projected", value)
            features = torch.cat(pooled)
            slots, _ = backbone.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=features)
            embeds = embeds.masked_scatter(slots, features)
        if inputs.get("pixel_values_videos") is not None:
            pooled = backbone.get_video_features(inputs["pixel_values_videos"].float(), inputs["video_grid_thw"], return_dict=True).pooler_output
            if emit_video:
                for value, grid in zip(pooled, inputs["video_grid_thw"]):
                    for pair in value.chunk(int(grid[0])):
                        trace.emit("vela3.d3.vision.projected", pair)
            features = torch.cat(pooled)
            _, slots = backbone.get_placeholder_mask(ids, inputs_embeds=embeds, video_features=features)
            embeds = embeds.masked_scatter(slots, features)
    positions = backbone.compute_3d_position_ids(input_ids=ids, inputs_embeds=embeds,
        image_grid_thw=inputs.get("image_grid_thw"), video_grid_thw=inputs.get("video_grid_thw"),
        attention_mask=inputs["attention_mask"], mm_token_type_ids=inputs["mm_token_type_ids"])
    readouts = dc.readout_offsets(processor.tokenizer, text, question["type"] == "track") if spatial else [0]
    counts = ([2] if question["type"] == "track" else []) + [128]*4 if spatial else [len(normalized.keys)]
    return text, ids, embeds, positions, state_count, normalized, readouts, counts


def main(args):
    import torch
    import transformers
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    from safetensors.torch import load_file
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    os.environ["D3_FAST"] = "0"
    os.environ["D3_FUSED"] = "0"
    root = args.model.resolve()
    # The read-only official runtime lazily imports its sibling compose module.
    sys.path.insert(0, str(root))
    runtime = load_module(root / "d3_runtime.py")
    request = json.loads(args.request.read_text())
    started = time.perf_counter()
    if args.official:
        engine = runtime.D3(root, device=args.device, model_name="vllm-sr/d3-turbo")
        loaded = time.perf_counter()
        response = engine.system_one(**request)
        args.out.write_text(json.dumps(dict(response=response, load_seconds=loaded-started,
            inference_seconds=time.perf_counter()-loaded, dtype="bfloat16", device=args.device,
            torch=torch.__version__, transformers=transformers.__version__), indent=2))
        return
    require_bf16_checkpoint(root)
    tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
    backbone = Qwen3_5Model.from_pretrained(root, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).eval()
    model = backbone.language_model
    if args.device != "cpu":
        raise ValueError("streaming F32 diagnostics use CPU; --official supports the GPU")
    weights = None
    if args.gguf:
        from gguf_weights import Weights
        weights = Weights(args.gguf)
        if any(int(t.tensor_type) == 1 for t in weights.tensors.values()):
            args.activation_quantization = True
    head = weights.value("weight") if weights else load_file(root / "readout.safetensors")["weight"].float()
    config = json.loads((root / "decision_config.json").read_text())
    trace = Trace(args.out)
    trace.record(dict(kind="oracle", torch=torch.__version__, transformers=transformers.__version__,
        dtype="float32", storage_dtype="gguf_mixed" if weights else "bfloat16", D3_FAST=0, D3_FUSED=0, D3_STATE_ONCE=0,
        source_revision="0c9281c53740765406dcec7f20140179838a6fa5",
        gguf=str(args.gguf) if args.gguf else None,
        activation_quantization=args.activation_quantization,
        gguf_sha256=hashlib.file_digest(args.gguf.open("rb"), "sha256").hexdigest() if args.gguf else None,
        sources={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
            ["config.json","decision_config.json","d3_runtime.py","d3_format.py","d3_compose.py",
             "tokenizer.json","tokenizer_config.json","chat_template.jinja","readout.safetensors"]}))
    hooks = []
    if args.vision_only:
        hooks.append(backbone.visual.patch_embed.register_forward_hook(
            lambda m, a, y: trace.emit("omni.vision.patch_bias", y)))
        hooks.append(backbone.visual.blocks[0].register_forward_pre_hook(
            lambda m, a: trace.emit("omni.vision.inp_pos_emb", a[0])))
        for i, layer in enumerate(backbone.visual.blocks):
            hooks.append(layer.register_forward_hook(
                lambda m, a, y, i=i: trace.emit("omni.vision.layer_out", y, i)))
    for i, layer in enumerate(model.layers):
        hooks.append(layer.input_layernorm.register_forward_hook(
            lambda m, a, y, i=i: trace.emit("vela3.d3.input_norm", y, i)))
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(
            lambda m, a, i=i: trace.emit("vela3.d3.attention_residual", a[0], i)))
        hooks.append(layer.register_forward_hook(
            lambda m, a, y, i=i: trace.emit("vela3.d3.layer", y, i)))
    def evaluate(key, question, clips, emit_video):
        mrope = None
        if request.get("images") or request.get("videos") or request.get("assets") or question["type"] == "box":
            text, ids, embeds, mrope, state, normalized, readouts, counts = media_prompt(
                root, runtime, backbone, model, request, question, config, trace, clips, emit_video)
        else:
            normalized = runtime.normalize_question(question)
            text = runtime.render(tok, "d3", request.get("state"), normalized.rendered, config["codes"])
            encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            state = runtime.state_token_count(encoded["offset_mapping"], runtime.state_end_char(text, request.get("state")))
            ids = torch.tensor([encoded["input_ids"]])
            embeds = model.embed_tokens(ids)
            readouts, counts = [0], [len(normalized.keys)]
        trace.record(dict(kind="prompt", question=key, text=text, state_tokens=state))
        trace.record(dict(name="vela3.d3.tokens", shape=[ids.numel()], len=ids.numel(), token_ids=ids[0].tolist()))
        if mrope is not None:
            trace.emit("vela3.d3.positions", mrope[:,0,:].T)
        trace.emit("vela3.d3.embedding", embeds)
        if args.vision_only:
            return None, [], 0
        blocks = runtime.StateBlocks.of(ids.shape[-1], [ids.shape[-1]], [state])
        hidden = runtime.PromptEncoder(model)(embeds, torch.ones_like(ids), mrope, blocks=blocks)
        trace.emit("vela3.d3.hidden", hidden)
        values = []
        for offset, count in zip(readouts, counts):
            logits = (hidden[:, -1-offset].float() @ head.T)[0, :count] / config["temperature"]
            trace.emit("vela3.d3.logits", logits)
            values.append(logits.softmax(-1).tolist())
        if normalized:
            answer = runtime.product_answer(normalized, values[0])
        else:
            box = dc.decode(values[-4:])
            answer = dict(type="box", box=box, confidence=dc.calibrate(
                dc.confidence(values[-4:],box), config["systemone"]["box_calibration"]))
            if question["type"] == "track":
                answer["visible"] = values[0][1] / sum(values[0])
        return answer, ids[0, :state].tolist(), ids.numel() - state

    loaded = time.perf_counter()
    answers, prefixes, tokens = {}, {}, 0
    dc = load_module(root / "d3_compose.py")
    if weights:
        from gguf_weights import stream_gguf
    with (stream_gguf(model, weights, args.activation_quantization) if weights else stream_f32(model)):
        for key, question in request["questions"].items():
            work = [(question, None, "request", None)]
            tracking = question["type"] == "track"
            if tracking or question["type"] == "box" and question.get("at") is not None:
                assert len(request["videos"]) == 1
                rate = question.get("rate", 1) if tracking else 1
                windows = runtime.load_video_windows(request["videos"][0], rate)
                moments = [(w, t) for w, clip in enumerate(windows) for t in dc.group_times(clip.indices, clip.fps)]
                if not tracking:
                    moments = [min(moments, key=lambda m: abs(m[1] - question["at"]))]
                work = [(dict(type=question["type"], instructions=question["instructions"], at=t),
                         [windows[w]], ("window", rate, w), t) for w, t in moments]
            entries = []
            for moment, clips, state_key, at in work:
                answer, prefix, block = evaluate(key, moment, clips, state_key not in prefixes)
                if args.vision_only:
                    continue
                if state_key not in prefixes:
                    prefixes[state_key] = prefix
                    tokens += len(prefix)
                assert prefixes[state_key] == prefix
                tokens += block
                if at is not None:
                    answer["time"] = round(at, 3)
                entries.append(answer)
            if entries:
                if tracking:
                    for entry in entries:
                        del entry["type"]
                    answers[key] = dict(type="track", track=entries)
                else:
                    answers[key] = entries[0]

    for hook in hooks:
        hook.remove()
    result = dict(response=dict(model="vllm-sr/d3-turbo", answers=answers,
        usage=dict(input_tokens=tokens, output_tokens=0)), load_seconds=loaded-started,
        inference_seconds=time.perf_counter()-loaded, dtype="float32", device="cpu",
        vision_only=args.vision_only)
    args.out.with_suffix(".result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--device", default="cpu")
    p.add_argument("--official", action="store_true")
    p.add_argument("--gguf", type=Path, help="independent same-weight dequantized reference")
    p.add_argument("--activation-quantization", action="store_true", help="model CPU block activation rounding independently")
    p.add_argument("--vision-only", action="store_true", help="trace preprocessing and the visual tower without the text forward")
    main(p.parse_args())
