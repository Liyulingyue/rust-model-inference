"""Convert a Microsoft/Mage-Flow transformer to a lossless BF16 GGUF.

All six Mage-Flow repositories publish the same NR-MMDiT tensor contract.  The
converter deliberately keeps the source BF16 bytes and only reverses the GGUF
dimension declaration required by this repository; it never quantizes or
upcasts a tensor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from converter.utils.gguf import GGML_BF16, GgufWriter, gguf_dims, open_safetensors, validated_dir

HIDDEN = 3072
CONTEXT = 2560
HEADS = 24
HEAD_DIM = 128
LAYERS = 12
IN_CHANNELS = 128
FFN = 12288
VARIANTS = {
    "base": (30, "text-to-image"),
    "flow": (20, "text-to-image"),
    "turbo": (4, "text-to-image"),
    "edit-base": (30, "image-to-image"),
    "edit": (20, "image-to-image"),
    "edit-turbo": (4, "image-to-image"),
}


def _find_transformer(root: Path) -> Path:
    candidates = sorted((root / "transformer").glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"no transformer safetensors under {root / 'transformer'}")
    if len(candidates) != 1:
        raise ValueError("Mage-Flow transformer must be a single safetensors file")
    return candidates[0]


def _validate_config(root: Path) -> dict:
    path = root / "transformer" / "config.json"
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid transformer/config.json: {exc}") from exc
    expected = {
        "in_channels": IN_CHANNELS,
        "out_channels": IN_CHANNELS,
        "context_in_dim": CONTEXT,
        "hidden_size": HIDDEN,
        "num_heads": HEADS,
        "depth": LAYERS,
        "axes_dim": [16, 56, 56],
        "patch_size": 1,
        "double_block_type": "double_stream",
        "param_dtype": "bfloat16",
        "mlp_ratio": 4.0,
        "depth_single_blocks": 0,
        "theta": 10000,
        "qkv_bias": True,
        "guidance_embed": False,
        "rope_type": "msrope",
        "time_type": "qwen_proj",
        "apply_text_rotary_emb": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"transformer config {key!r}: expected {value!r}, got {config.get(key)!r}")
    return config


def _expected_shapes() -> dict[str, tuple[int, ...]]:
    shapes = {"txt_norm.weight": (CONTEXT,)}

    def linear(name: str, n_in: int, n_out: int) -> None:
        shapes[f"{name}.weight"] = (n_out, n_in)
        shapes[f"{name}.bias"] = (n_out,)

    linear("img_in", IN_CHANNELS, HIDDEN)
    linear("txt_in", CONTEXT, HIDDEN)
    linear("norm_out.linear", HIDDEN, 2 * HIDDEN)
    linear("proj_out", HIDDEN, IN_CHANNELS)
    linear("time_text_embed.timestep_embedder.linear_1", 256, HIDDEN)
    linear("time_text_embed.timestep_embedder.linear_2", HIDDEN, HIDDEN)
    for i in range(LAYERS):
        p = f"transformer_blocks.{i}"
        for stream in ("img", "txt"):
            linear(f"{p}.{stream}_mod.1", HIDDEN, 6 * HIDDEN)
            linear(f"{p}.{stream}_mlp.net.0.proj", HIDDEN, FFN)
            linear(f"{p}.{stream}_mlp.net.2", FFN, HIDDEN)
        for suffix in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_out.0", "to_add_out"):
            linear(f"{p}.attn.{suffix}", HIDDEN, HIDDEN)
        for suffix in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            shapes[f"{p}.attn.{suffix}.weight"] = (HEAD_DIM,)
    return shapes


def _expected_names() -> set[str]:
    return set(_expected_shapes())


def _validate_tensors(source, expected=None) -> dict[str, tuple[int, ...]]:
    expected = _expected_shapes() if expected is None else expected
    names = set(source.header) - {"__metadata__"}
    if names != set(expected):
        missing, extra = sorted(set(expected) - names), sorted(names - set(expected))
        raise ValueError(f"transformer tensor contract mismatch: missing={missing[:3]} extra={extra[:3]}")
    spans = []
    for name, shape in expected.items():
        info = source.header[name]
        if info.get("dtype") != "BF16":
            raise ValueError(f"{name}: Mage-Flow source must remain BF16, got {info.get('dtype')}")
        if tuple(info.get("shape", ())) != shape:
            raise ValueError(f"{name}: expected shape {shape}, got {info.get('shape')}")
        offsets = info.get("data_offsets", [])
        if len(offsets) != 2 or any(type(x) is not int for x in offsets):
            raise ValueError(f"{name}: invalid data_offsets")
        start, end = offsets
        if start < 0 or end - start != 2 * math.prod(shape) or source.data_offset + end > source.file_size:
            raise ValueError(f"{name}: invalid or truncated BF16 payload")
        spans.append((start, end))
    position = 0
    for start, end in sorted(spans):
        if start != position:
            raise ValueError("transformer safetensors contains overlapping or gapped payloads")
        position = end
    if source.data_offset + position != source.file_size:
        raise ValueError("transformer safetensors contains trailing bytes")
    return expected


def _add_tensor(writer: GgufWriter, source, name: str) -> None:
    info = source.header[name]
    start, end = info["data_offsets"]

    def chunks():
        with source.path.open("rb") as stream:
            stream.seek(source.data_offset + start)
            remaining = end - start
            while remaining:
                chunk = stream.read(min(remaining, 8 * 1024 * 1024))
                if not chunk:
                    raise ValueError(f"{name}: truncated safetensors data")
                remaining -= len(chunk)
                yield chunk

    writer.add_tensor_chunks(name, GGML_BF16, gguf_dims(tuple(info["shape"])), end - start, chunks)


def convert(model_dir: Path, out_dir: Path, variant: str) -> Path:
    model_dir = validated_dir(str(model_dir), must_exist=True)
    config = _validate_config(model_dir)
    if variant not in VARIANTS:
        raise ValueError(f"unknown Mage-Flow variant {variant!r}; choose from {sorted(VARIANTS)}")
    source = open_safetensors(_find_transformer(model_dir))
    expected = _validate_tensors(source)
    with source.path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    steps, task = VARIANTS[variant]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"mage-flow-{variant}-dit-BF16.gguf"
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    fd, temporary = tempfile.mkstemp(prefix=f".{out.name}.", dir=out_dir)
    os.close(fd)
    temporary = Path(temporary)
    writer = GgufWriter(temporary)
    writer.add_meta("general.architecture", "mage_flow")
    writer.add_meta("general.file_type", 32)  # GGUF MOSTLY_BF16, distinct from tensor type 30.
    writer.add_meta("mage_flow.variant", variant)
    writer.add_meta("mage_flow.task", task)
    writer.add_meta("mage_flow.steps", steps)
    writer.add_meta("mage_flow.lossless_dtype", "BF16")
    writer.add_meta("mage_flow.config", json.dumps(config, sort_keys=True))
    writer.add_meta("mage_flow.source_sha256", digest)
    writer.add_meta("mage_flow.oracle_commit", "76bec2bb3818863f470de7e867c2dc7f1d0bfd83")
    try:
        for name in sorted(expected):
            _add_tensor(writer, source, name)
        writer.write()
        os.link(temporary, out)  # Publish without overwriting an existing export.
    finally:
        temporary.unlink(missing_ok=True)
    return out



def _vae_shapes():
    shapes = {}
    def linear(name, n_in, n_out, bias=True):
        shapes[name + ".weight"] = (n_out, n_in)
        if bias: shapes[name + ".bias"] = (n_out,)
    def conv(name, n_in, n_out, kernel=1, groups=1, bias=True):
        shapes[name + ".weight"] = (n_out, n_in // groups, kernel, kernel)
        if bias: shapes[name + ".bias"] = (n_out,)
    def norm(name, width, bias=True):
        shapes[name + ".weight"] = (width,)
        if bias: shapes[name + ".bias"] = (width,)
    def dico(name, width=384, adaptive=True):
        conv(name+".conv1", width,width)
        conv(name+".conv2", width,width,3,groups=width)
        conv(name+".conv3", width,width)
        conv(name+".ca.1", width,width)
        conv(name+".conv4", width,4*width)
        conv(name+".conv5", 4*width,width)
        if adaptive: linear(name+".adaLN_modulation.1",width,6*width)
        else:
            norm(name+".norm1",width); norm(name+".norm2",width)
    enc="student.dconv_encoder"
    conv(enc+".patch_cond_embed",3,768,16)
    for i in range(2): dico(f"{enc}.head_blocks.{i}",768,False)
    conv(enc+".proj_down",768,384);conv(enc+".z_proj",128,384)
    conv(enc+".fuse_proj",768,384)
    linear(enc+".t_embedder.mlp.0",256,384);linear(enc+".t_embedder.mlp.2",384,384)
    for i in range(21):dico(f"{enc}.blocks.{i}")
    norm(enc+".norm_out",384);conv(enc+".proj_out",384,256)
    dec="pipeline"
    linear(dec+".t_embedder.mlp.0",256,384);linear(dec+".t_embedder.mlp.2",384,384)
    conv(dec+".y_embedder_x",384,8192);linear(dec+".x_embedder.embedder.0",99,32)
    conv(dec+".s_embedder.proj1",3,128,16,bias=False);conv(dec+".s_embedder.proj2",512,384)
    for i in range(21):dico(f"{dec}.blocks.{i}")
    linear(dec+".dec_net.cond_embed",384,8192);linear(dec+".dec_net.input_proj",32,32)
    for i in range(3):
        p=f"{dec}.dec_net.res_blocks.{i}"
        norm(p+".in_ln",32);linear(p+".mlp.0",32,32);linear(p+".mlp.2",32,32)
        linear(p+".adaLN_modulation.1",32,96)
    norm(dec+".final_layer.norm",32,bias=False);linear(dec+".final_layer.linear",32,3)
    p=dec+".y_embedder.decoder"
    conv(p+".conv_in",128,384,3);norm(p+".norm_out",384);conv(p+".conv_out",384,384,3)
    for i in (0,2,4):
        b=f"{p}.block.{i}"
        norm(b+".norm1",384);conv(b+".conv1",384,384,3)
        norm(b+".norm2",384);conv(b+".conv2",384,384,3)
    for i in (1,3):
        b=f"{p}.block.{i}";norm(b+".norm",384)
        for n in ("q","k","v","proj_out"):conv(b+"."+n,384,384)
    return shapes


def convert_vae(model_dir: Path, out_dir: Path) -> Path:
    config=json.loads((model_dir/"vae/config.json").read_text())
    if config != {"_class_name":"MageVAE", "latent_channels":128, "downsample_factor":16, "sample_posterior":False}:
        raise ValueError("Unsupported MageVAE config (deterministic posterior mode required)")
    source=open_safetensors(model_dir/"vae/diffusion_pytorch_model.safetensors")
    expected=_vae_shapes()
    # The official MageVAE loader discards the training-only Flux encoder.
    ignored={name:tuple(info["shape"]) for name,info in source.header.items() if name.startswith("pipeline.y_embedder.encoder.")}
    _validate_tensors(source,expected | ignored)
    out_dir.mkdir(parents=True,exist_ok=True)
    out=out_dir/"mage-vae-BF16.gguf"
    if out.exists() or out.is_symlink():raise FileExistsError(out)
    fd,temporary=tempfile.mkstemp(prefix=".mage-vae.",dir=out_dir);os.close(fd)
    temporary=Path(temporary)
    try:
        writer=GgufWriter(temporary)
        writer.add_meta("general.architecture","mage_vae")
        writer.add_meta("general.file_type",32)
        writer.add_meta("mage_vae.sample_posterior",False)
        with source.path.open("rb") as stream:writer.add_meta("mage_vae.source_sha256",hashlib.file_digest(stream,"sha256").hexdigest())
        for name in sorted(expected):_add_tensor(writer,source,name)
        writer.write();os.link(temporary,out)
    finally:temporary.unlink(missing_ok=True)
    return out

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="ModelScope/Hugging Face Mage-Flow directory")
    parser.add_argument("--variant", choices=sorted(VARIANTS))
    parser.add_argument("--out-dir", type=Path, default=Path("models/mage-flow"))
    parser.add_argument("--vae", action="store_true", help="Export the shared deterministic MageVAE component")
    args = parser.parse_args()
    if not args.vae and not args.variant: parser.error("--variant is required for a DiT export")
    print(convert_vae(args.model_dir, args.out_dir) if args.vae else convert(args.model_dir, args.out_dir, args.variant))


if __name__ == "__main__":
    main()
