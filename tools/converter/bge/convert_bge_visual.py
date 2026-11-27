"""Convert BGE-VL and Visualized BGE weights to self-contained GGUF."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tools.converter.bge.convert_bge import (
    TensorStore,
    _add_meta,
    _chunks,
    _load_json,
    _source_sha256,
    _encoder_writer,
    _source_names,
    _decoder_writer,
    _decoder_mapping,
    _to_f32,
)
from tools.converter.utils.gguf import (
    GgufWriter,
    GGML_F32,
    GGML_Q8_0,
    MetaArray,
    gguf_dims,
)


def clip_layer_shapes(prefix: str, cfg: dict) -> dict[str, tuple[int, ...]]:
    hidden = int(cfg["hidden_size"])
    ff = int(cfg["intermediate_size"])
    heads = int(cfg["num_attention_heads"])
    layers = int(cfg["num_hidden_layers"])
    if min(hidden, ff, heads, layers) <= 0 or hidden % heads:
        raise ValueError("invalid CLIP dimensions")
    shapes = {}
    for i in range(layers):
        base = f"{prefix}.encoder.layers.{i}"
        for norm in ["layer_norm1", "layer_norm2"]:
            for suffix in ["weight", "bias"]:
                shapes[f"{base}.{norm}.{suffix}"] = (hidden,)
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            shapes[f"{base}.self_attn.{proj}.weight"] = (hidden, hidden)
            shapes[f"{base}.self_attn.{proj}.bias"] = (hidden,)
        for proj, shape in [("fc1", (ff, hidden)), ("fc2", (hidden, ff))]:
            shapes[f"{base}.mlp.{proj}.weight"] = shape
            shapes[f"{base}.mlp.{proj}.bias"] = (shape[0],)
    return shapes


def vision_shapes(prefix: str, cfg: dict) -> dict[str, tuple[int, ...]]:
    hidden = int(cfg["hidden_size"])
    side = int(cfg["image_size"])
    patch = int(cfg["patch_size"])
    if side <= 0 or patch <= 0 or side % patch:
        raise ValueError("invalid CLIP image/patch size")
    shapes = clip_layer_shapes(prefix, cfg)
    shapes[f"{prefix}.embeddings.class_embedding"] = (hidden,)
    shapes[f"{prefix}.embeddings.patch_embedding.weight"] = (hidden, 3, patch, patch)
    shapes[f"{prefix}.embeddings.position_embedding.weight"] = (
        (side // patch) ** 2 + 1,
        hidden,
    )
    for norm in ["pre_layrnorm", "post_layernorm"]:
        for suffix in ["weight", "bias"]:
            shapes[f"{prefix}.{norm}.{suffix}"] = (hidden,)
    return shapes


def clip_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    text = config["text_config"]
    vision = config["vision_config"]
    out = int(config["projection_dim"])
    shapes = vision_shapes("vision_model", vision)
    shapes.update(clip_layer_shapes("text_model", text))
    shapes["text_model.embeddings.token_embedding.weight"] = (
        int(text["vocab_size"]),
        int(text["hidden_size"]),
    )
    shapes["text_model.embeddings.position_embedding.weight"] = (
        int(text["max_position_embeddings"]),
        int(text["hidden_size"]),
    )
    for suffix in ["weight", "bias"]:
        shapes[f"text_model.final_layer_norm.{suffix}"] = (int(text["hidden_size"]),)
    shapes["text_projection.weight"] = (out, int(text["hidden_size"]))
    shapes["visual_projection.weight"] = (out, int(vision["hidden_size"]))
    return shapes


def _check_inventory(
    store: TensorStore, shapes: dict[str, tuple[int, ...]], allowed: set[str]
) -> None:
    missing = set(shapes) - store.names()
    extra = store.names() - set(shapes) - allowed
    if missing:
        raise ValueError(f"missing required visual tensors: {sorted(missing)[:5]}")
    if extra:
        raise ValueError(
            f"unsupported visual tensors would be dropped: {sorted(extra)[:5]}"
        )
    for name, shape in shapes.items():
        if store[name].shape != shape:
            raise ValueError(f"{name}: shape {store[name].shape} != expected {shape}")


def _write_tensors(writer: GgufWriter, store: TensorStore, names, quant: str) -> None:
    for name in names:
        tensor = store[name]
        quantized = (
            quant == "Q8_0"
            and len(tensor.shape) == 2
            and name.endswith("weight")
            and tensor.shape[-1] % 32 == 0
            and "position" not in name
            and "norm" not in name
            and "rope" not in name
        )
        mode = "Q8_0" if quantized else "F32"
        elements = math.prod(tensor.shape)
        writer.add_tensor_chunks(
            name,
            GGML_Q8_0 if quantized else GGML_F32,
            gguf_dims(tensor.shape),
            elements // 32 * 34 if quantized else elements * 4,
            lambda t=tensor, q=mode: _chunks(store, t, q),
        )


def convert_visual(
    model_dir: Path, output: Path, quant: str = "F32", *, text_model: Path | None = None
) -> None:
    if output.exists():
        raise FileExistsError(output)
    if quant not in {"F32", "Q8_0"}:
        raise ValueError(f"unsupported quantization {quant!r}")
    if model_dir.is_file():
        if text_model is None:
            raise ValueError(
                "Visualized BGE requires --text-model with the base encoder config and tokenizer"
            )
        writer = _visualized_writer(model_dir, text_model, output, quant)
        writer.write()
        writer.path.replace(output)
        print(output)
        return
    config = _load_json(model_dir, "config.json")
    if config.get("model_type") == "qwen2_5_vl":
        writer = _screenshot_writer(model_dir, output, config, quant)
        writer.write()
        writer.path.replace(output)
        print(output)
        return
    if config.get("model_type") == "llava_next":
        writer = _llava_writer(model_dir, output, config, quant)
        writer.write()
        writer.path.replace(output)
        print(output)
        return
    if config.get("model_type") != "clip":
        raise ValueError(f'unsupported visual model type {config.get("model_type")!r}')
    store = TensorStore(model_dir)
    shapes = clip_shapes(config)
    _check_inventory(
        store,
        shapes,
        {
            "logit_scale",
            "text_model.embeddings.position_ids",
            "vision_model.embeddings.position_ids",
        },
    )
    tokenizer = (model_dir / "tokenizer.json").read_text(encoding="utf-8")
    json.loads(tokenizer)
    processor = _load_json(model_dir, "preprocessor_config.json")
    writer = GgufWriter(output.with_suffix(output.suffix + ".tmp"))
    _add_meta(writer, "general.architecture", "bge-vl-clip")
    _add_meta(writer, "general.name", model_dir.name)
    _add_meta(writer, "general.file_type", 0 if quant == "F32" else 7)
    _add_meta(writer, "general.quantization_version", 2)
    _add_meta(writer, "general.source_sha256", _source_sha256(store))
    _add_meta(writer, "bge.config", json.dumps(config, separators=(",", ":")))
    _add_meta(writer, "bge.preprocessor", json.dumps(processor, separators=(",", ":")))
    _add_meta(writer, "bge.tokenizer_json", tokenizer)
    _write_tensors(writer, store, shapes, quant)
    writer.write()
    writer.path.replace(output)
    print(output)


def _visualized_writer(
    checkpoint: Path, text_model: Path, output: Path, quant: str
) -> GgufWriter:
    config = _load_json(text_model, "config.json")
    store = TensorStore(checkpoint)
    if config["model_type"] not in {"bert", "xlm-roberta"}:
        raise ValueError("Visualized BGE requires a BERT or XLM-R encoder")
    hidden = int(config["hidden_size"])
    depth = int(config["num_hidden_layers"])
    prefix = "model_visual.visual"
    patch = store[f"{prefix}.patch_embed.proj.weight"].shape[-1]
    n = store[f"{prefix}.pos_embed"].shape[1]
    grid = math.isqrt(n - 1)
    if grid * grid != n - 1:
        raise ValueError("EVA position table is not a square grid")
    ff = store[f"{prefix}.blocks.0.mlp.w1.weight"].shape[0]
    head_dim = store[f"{prefix}.blocks.0.attn.rope.freqs_cos"].shape[1]
    if hidden % head_dim:
        raise ValueError("invalid EVA head dimension")
    shapes = {
        f"{prefix}.cls_token": (1, 1, hidden),
        f"{prefix}.pos_embed": (1, n, hidden),
        f"{prefix}.patch_embed.proj.weight": (hidden, 3, patch, patch),
        f"{prefix}.patch_embed.proj.bias": (hidden,),
        "visual_proj.weight": (hidden, hidden),
        "visual_proj.bias": (hidden,),
    }
    for i in range(depth):
        b = f"{prefix}.blocks.{i}"
        for norm, width in [
            ("norm1", hidden),
            ("norm2", hidden),
            ("attn.inner_attn_ln", hidden),
            ("mlp.ffn_ln", ff),
        ]:
            for suffix in ["weight", "bias"]:
                shapes[f"{b}.{norm}.{suffix}"] = (width,)
        for proj in ["q_proj", "k_proj", "v_proj"]:
            shapes[f"{b}.attn.{proj}.weight"] = (hidden, hidden)
        for bias in ["q_bias", "v_bias"]:
            shapes[f"{b}.attn.{bias}"] = (hidden,)
        shapes[f"{b}.attn.proj.weight"] = (hidden, hidden)
        shapes[f"{b}.attn.proj.bias"] = (hidden,)
        for proj, shape in [
            ("w1", (ff, hidden)),
            ("w2", (ff, hidden)),
            ("w3", (hidden, ff)),
        ]:
            shapes[f"{b}.mlp.{proj}.weight"] = shape
            shapes[f"{b}.mlp.{proj}.bias"] = (shape[0],)
        for kind in ["cos", "sin"]:
            shapes[f"{b}.attn.rope.freqs_{kind}"] = (n - 1, head_dim)
    text_mapping = {
        name: (
            ("bge_embeddings." + name.removeprefix("embeddings."))
            if name.startswith("embeddings.")
            else "bge_" + name
        )
        for name in _source_names(
            config["model_type"], depth, int(config["type_vocab_size"])
        ).values()
    }
    allowed = (
        set(text_mapping.values())
        | {f"bge_pooler.dense.{suffix}" for suffix in ["weight", "bias"]}
        | {"model_visual.logit_scale"}
    )
    allowed |= {
        f"{prefix}.{name}.{suffix}"
        for name in ["norm", "head"]
        for suffix in ["weight", "bias"]
    }
    allowed |= {f"{prefix}.rope.freqs_{kind}" for kind in ["cos", "sin"]}
    _check_inventory(store, shapes, allowed)
    writer = _encoder_writer(
        text_model, output, config, quant, store.select(text_mapping)
    )
    _add_meta(writer, "bge.visualized", True)
    _add_meta(
        writer,
        "bge.config",
        json.dumps(
            dict(
                hidden_size=hidden,
                intermediate_size=ff,
                num_hidden_layers=depth,
                num_attention_heads=hidden // head_dim,
                image_size=grid * patch,
                patch_size=patch,
                layer_norm_eps=1e-6,
            )
        ),
    )
    _add_meta(
        writer,
        "bge.preprocessor",
        json.dumps(
            dict(
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711],
            )
        ),
    )
    _write_tensors(writer, store, shapes, quant)
    tensor = store["bge_embeddings.position_embeddings.weight"]
    writer.add_tensor_chunks(
        "bge.raw_position.weight",
        GGML_F32,
        gguf_dims(tensor.shape),
        math.prod(tensor.shape) * 4,
        lambda: _chunks(store, tensor, "F32"),
    )
    return writer


def screenshot_vision_config(config: dict) -> dict:
    # Qwen2_5_VLVisionConfig defaults from transformers v4.51.3, the source release.
    vision = dict(
        depth=32,
        hidden_size=3584,
        intermediate_size=3420,
        num_heads=16,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        window_size=112,
        fullatt_block_indexes=[7, 15, 23, 31],
        hidden_act="silu",
    )
    vision.update(config["vision_config"])
    vision.setdefault("out_hidden_size", config["hidden_size"])
    if (
        vision["hidden_act"] != "silu"
        or vision["temporal_patch_size"] != 2
        or vision["spatial_merge_size"] != 2
    ):
        raise ValueError("unsupported Screenshot vision activation/patch merge")
    full = vision["fullatt_block_indexes"]
    period = full[0] + 1 if full else 0
    if (
        not period
        or full != list(range(period - 1, int(vision["depth"]), period))
        or vision["window_size"] != 112
    ):
        raise ValueError("unsupported Screenshot window attention schedule")
    return vision


def screenshot_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    v = screenshot_vision_config(config)
    h = int(v["hidden_size"])
    ff = int(v["intermediate_size"])
    p = int(v["patch_size"])
    merged = h * 4
    out = int(v["out_hidden_size"])
    if min(h, ff, p, int(v["depth"]), int(v["num_heads"]), out) <= 0 or h % int(
        v["num_heads"]
    ):
        raise ValueError("invalid Screenshot vision dimensions")
    shapes = {
        "visual.patch_embed.proj.weight": (h, 3, 2, p, p),
        "visual.merger.ln_q.weight": (h,),
        "visual.merger.mlp.0.weight": (merged, merged),
        "visual.merger.mlp.0.bias": (merged,),
        "visual.merger.mlp.2.weight": (out, merged),
        "visual.merger.mlp.2.bias": (out,),
    }
    for i in range(int(v["depth"])):
        b = f"visual.blocks.{i}"
        shapes[f"{b}.norm1.weight"] = (h,)
        shapes[f"{b}.norm2.weight"] = (h,)
        for name, shape in [
            ("attn.qkv", (3 * h, h)),
            ("attn.proj", (h, h)),
            ("mlp.gate_proj", (ff, h)),
            ("mlp.up_proj", (ff, h)),
            ("mlp.down_proj", (h, ff)),
        ]:
            shapes[f"{b}.{name}.weight"] = shape
            shapes[f"{b}.{name}.bias"] = (shape[0],)
    return shapes


def _screenshot_writer(
    model_dir: Path, output: Path, config: dict, quant: str
) -> GgufWriter:
    store = TensorStore(model_dir)
    v = screenshot_vision_config(config)
    shapes = screenshot_shapes(config)
    text = dict(config, model_type="qwen2")
    mapping = _decoder_mapping("qwen2", int(text["num_hidden_layers"]))
    _check_inventory(store, shapes, set(mapping.values()) | {"lm_head.weight"})
    if int(v["out_hidden_size"]) != int(text["hidden_size"]):
        raise ValueError("Screenshot projector/text width mismatch")
    writer = _decoder_writer(
        model_dir,
        output,
        text,
        quant,
        store.select({name: name for name in mapping.values()}),
        architecture="qwen2vl",
    )
    sections = config.get("rope_scaling", {}).get("mrope_section")
    head = int(text["hidden_size"]) // int(text["num_attention_heads"])
    if (
        not isinstance(sections, list)
        or len(sections) != 3
        or any(not isinstance(x, int) or x < 0 for x in sections)
        or sum(sections) != head // 2
    ):
        raise ValueError("invalid Screenshot mrope_section")
    _add_meta(
        writer, "qwen2vl.rope.dimension_sections", MetaArray(5, tuple(sections + [0]))
    )
    _add_meta(writer, "bge.screenshot", True)
    _add_meta(writer, "bge.config", json.dumps(config, separators=(",", ":")))
    processor = _load_json(model_dir, "preprocessor_config.json")
    metadata = {
        "projection_dim": int(v["out_hidden_size"]),
        "image_size": 560,
        "patch_size": int(v["patch_size"]),
        "embedding_length": int(v["hidden_size"]),
        "feed_forward_length": int(v["intermediate_size"]),
        "block_count": int(v["depth"]),
        "attention.head_count": int(v["num_heads"]),
        "attention.layer_norm_epsilon": float(text["rms_norm_eps"]),
        "spatial_merge_size": 2,
        "n_wa_pattern": int(v["fullatt_block_indexes"][0]) + 1,
        "image_min_pixels": int(processor["min_pixels"]),
        "image_max_pixels": int(processor["max_pixels"]),
    }
    for key, value in metadata.items():
        _add_meta(writer, "clip.vision." + key, value)
    for key in ["image_mean", "image_std"]:
        _add_meta(writer, "clip.vision." + key, MetaArray(6, tuple(processor[key])))
    _add_meta(writer, "clip.use_silu", True)
    renamed = {"v.post_ln.weight": "visual.merger.ln_q.weight"}
    for layer in range(int(v["depth"])):
        b = f"visual.blocks.{layer}"
        out = f"v.blk.{layer}"
        for target, original in [
            ("ln1", "norm1"),
            ("ln2", "norm2"),
            ("attn_qkv", "attn.qkv"),
            ("attn_out", "attn.proj"),
            ("ffn_gate", "mlp.gate_proj"),
            ("ffn_up", "mlp.up_proj"),
            ("ffn_down", "mlp.down_proj"),
        ]:
            for suffix in ["weight", "bias"]:
                name = f"{b}.{original}.{suffix}"
                if name in shapes:
                    renamed[f"{out}.{target}.{suffix}"] = name
    for i in [0, 2]:
        for suffix in ["weight", "bias"]:
            renamed[f"mm.{i}.{suffix}"] = f"visual.merger.mlp.{i}.{suffix}"
    _write_tensors(writer, store.select(renamed), renamed, quant)
    tensor = store["visual.patch_embed.proj.weight"]
    values = np.frombuffer(
        _to_f32(store.read(tensor), tensor.dtype), dtype="<f4"
    ).reshape(tensor.shape)
    for index in range(2):
        raw = values[:, :, index].copy().tobytes()
        shape = (tensor.shape[0], 3, tensor.shape[3], tensor.shape[4])
        writer.add_tensor(
            "v.patch_embd.weight" + (".1" if index else ""),
            GGML_F32,
            gguf_dims(shape),
            raw,
        )
    return writer


def _llava_writer(
    model_dir: Path, output: Path, config: dict, quant: str
) -> GgufWriter:
    store = TensorStore(model_dir)
    text = dict(config["text_config"])
    if text.get("model_type") != "mistral":
        raise ValueError("BGE LLaVA-NeXT requires the verified Mistral decoder")
    embedding = store["language_model.model.embed_tokens.weight"]
    text["vocab_size"], text["hidden_size"] = embedding.shape
    text["num_hidden_layers"] = len(
        {
            n.split(".")[3]
            for n in store.names()
            if n.startswith("language_model.model.layers.")
        }
    )
    text.setdefault("num_attention_heads", 32)
    text.setdefault("num_key_value_heads", 8)
    text.setdefault("rms_norm_eps", 1e-5)
    text.setdefault("rope_theta", 1000000.0)
    text.setdefault("max_position_embeddings", 32768)
    mapping = {
        name: "language_model." + name
        for name in _decoder_mapping("mistral", text["num_hidden_layers"]).values()
    }
    vision = dict(config["vision_config"])
    vision.setdefault("hidden_act", "quick_gelu")
    vision.setdefault("layer_norm_eps", 1e-5)
    shapes = vision_shapes("vision_tower.vision_model", vision)
    h = int(text["hidden_size"])
    v = int(vision["hidden_size"])
    for name, shape in [("linear_1", (h, v)), ("linear_2", (h, h))]:
        shapes[f"multi_modal_projector.{name}.weight"] = shape
        shapes[f"multi_modal_projector.{name}.bias"] = (h,)
    shapes["image_newline"] = (h,)
    allowed = set(mapping.values()) | {"language_model.lm_head.weight"}
    allowed |= {name for name in store.names() if name.endswith("rotary_emb.inv_freq")}
    _check_inventory(store, shapes, allowed)
    writer = _decoder_writer(model_dir, output, text, quant, store.select(mapping))
    _add_meta(writer, "bge.llava_next", True)
    _add_meta(
        writer,
        "bge.config",
        json.dumps(dict(config, vision_config=vision, text_config=text)),
    )
    _add_meta(
        writer,
        "bge.preprocessor",
        json.dumps(_load_json(model_dir, "preprocessor_config.json")),
    )
    _write_tensors(writer, store, shapes, quant)
    return writer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quant", choices=["F32", "Q8_0"], default="F32")
    parser.add_argument("--text-model", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    convert_visual(args.model_dir, args.output, args.quant, text_model=args.text_model)


if __name__ == "__main__":
    main()
