#!/usr/bin/env python3
"""Generate small same-input CPU fixtures from pinned official Idol; never downloads."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

COMMIT = "4702a7f00850e34b6e83a2170cc123c3e9570744"
SEED = 20261103
SIDE = 64


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(source):
    pinned = json.loads(Path(__file__).with_name("source-manifest.json").read_text())
    actual_files = {str(path.relative_to(source)) for path in source.rglob("*.py")}
    if pinned["commit"] != COMMIT or actual_files != set(pinned["files"]):
        raise ValueError("Oracle source file set does not match pinned archive")
    for relative, expected in pinned["files"].items():
        if sha256(source / relative) != expected:
            raise ValueError(f"Pinned oracle source hash mismatch: {relative}")
    if (source / ".git").exists():
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if commit != COMMIT:
            raise ValueError(f"Oracle source commit mismatch: {commit}")


class Trace:
    def __init__(self, directory):
        self.directory = directory.resolve()
        self.values = {}
        self.order = []

    def emit(self, name, value, shape=None, nhwc=False):
        if isinstance(value, tuple):
            value = value[0]
        data = value.detach().float().cpu()
        if nhwc:
            data = data.permute(0, 2, 3, 1)
        if shape:
            data = data.reshape(shape)
        data = data.contiguous().numpy().copy()
        if name in self.values:
            raise ValueError(f"Duplicate oracle checkpoint {name}")
        if not __import__("numpy").isfinite(data).all():
            raise ValueError(f"Nonfinite oracle checkpoint {name}")
        self.values[name] = data

    def hook(self, module, name, half=False, shape=None, nhwc=False, before=False):
        def capture(module, inputs, output=None):
            value = inputs[0] if before else output
            if half:
                value = value.half()
            self.emit(name, value, shape, nhwc)
        if before:
            return module.register_forward_pre_hook(capture)
        return module.register_forward_hook(capture)

    def finish(self):
        if len(self.order) != len(set(self.order)) or set(self.order) != set(self.values):
            raise ValueError(f"Oracle capture/order mismatch: missing={set(self.order)-set(self.values)}, extra={set(self.values)-set(self.order)}")
        records = []
        for name in self.order:
            data = self.values[name]
            path = self.directory / f"oracle.jsonl.{name}.f32"
            data.astype("<f4").tofile(path)
            records.append(dict(name=name, occurrence=0, shape=list(data.shape), len=int(data.size), finite=True, binary_path=str(path)))
        path = self.directory / "oracle.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        return path


def denoiser_probe(root, config, inputs, trace):
    import torch
    from safetensors.torch import load_file
    from tensordict import TensorDict
    from idol.models import LatentSeqMMFlowModel
    model = LatentSeqMMFlowModel(**config["denoiser"]["args"])
    model.load_state_dict(load_file(str(root / "idol/denoiser.safetensors")), strict=True)
    model.eval().requires_grad_(False)
    # Only diagnostic fixtures use a prefix of the stored Sobol positions.
    model.pos_pe = model.pos_pe[:, :32]
    h = lambda module, name, **kw: trace.hook(module, "idol.denoiser." + name, **kw)
    h(model.input_layer, "input", half=True)
    h(model.cond_embedder, "condition1")
    h(model.cond_embedder2, "condition2")
    h(model.context_repo_layers[0], "condition", before=True)
    h(model.t_embedder.mlp[0], "time_linear1", shape=(1, 1, 1024))
    h(model.t_embedder.mlp[1], "time_silu", shape=(1, 1, 1024))
    h(model.t_embedder, "time", shape=(1, 1, 1024))
    h(model.adaLN_modulation, "modulation", half=True, shape=(1, 6, 1024))
    h(model.noise_repo_layers[0], "position_added", before=True)
    for prefix, blocks in [("noise", model.noise_refiner), ("context", model.context_refiner), ("joint", model.blocks)]:
        for i, block in enumerate(blocks):
            h(block, f"{prefix}.{i}.out")
    h(model.cam_refiner.mlp[0], "camera_linear.0")
    h(model.cam_refiner.mlp[1], "camera.0")
    h(model.cam_refiner.mlp[2], "camera_linear.1")
    h(model.cam_refiner.mlp[2], "camera.1")
    h(model.out_layer, "final_modulated_latent", before=True)
    h(model.cam_out_layer, "final_modulated_camera", before=True)
    h(model.out_layer, "velocity_latent")
    h(model.cam_out_layer, "velocity_camera")
    x = TensorDict({"latent": inputs["latent"], "camera": inputs["camera"]}, batch_size=1)
    cond = TensorDict({"feature1": inputs["feature1"], "feature2": inputs["feature2"]}, batch_size=1)
    model(x, torch.tensor([500.]), cond)
    names = ["input", "condition1", "condition2", "condition", "time_linear1", "time_silu", "time", "modulation", "position_added"]
    names += [f"noise.{i}.out" for i in range(2)] + [f"context.{i}.out" for i in range(2)]
    names += ["camera_linear.0", "camera.0", "camera_linear.1", "camera.1"]
    names += [f"joint.{i}.out" for i in range(24)]
    names += ["final_modulated_latent", "final_modulated_camera", "velocity_latent", "velocity_camera"]
    trace.order.extend("idol.denoiser." + name for name in names)


def decoder_probe(root, config, inputs, trace):
    import torch
    from safetensors.torch import load_file
    from idol.models import GaussianFixedlenDecoder, OctreeProbabilityFixedlenDecoder
    from plyfile import PlyData
    for key, cls in [("decoder", OctreeProbabilityFixedlenDecoder), ("decoder_gs", GaussianFixedlenDecoder)]:
        name = "octree" if key == "decoder" else "gaussian"
        model = cls(**config[key]["args"])
        model.load_state_dict(load_file(str(root / f"idol/{key}.safetensors")), strict=True)
        model.eval().requires_grad_(False)
        if name == "octree":
            trace.hook(model.adaLN_modulation, "idol.octree.modulation", half=True)
            trace.order.append("idol.octree.modulation")
        trace.hook(model.input_layer, f"idol.{name}.input", half=True, shape=(12, 1024))
        trace.order.append(f"idol.{name}.input")
        for i, block in enumerate(model.blocks):
            trace.hook(block, f"idol.{name}.block.{i}", shape=(12, 1024))
            trace.order.append(f"idol.{name}.block.{i}")
        trace.hook(model.out_proj, f"idol.{name}.output", shape=(12, 8 if name == "octree" else 480))
        trace.order.append(f"idol.{name}.output")
        if name == "octree":
            model(inputs["points"], torch.tensor([16]), inputs["latent"])
        else:
            features = model(inputs["points"], inputs["latent"])
            inputs["representation_features"] = features.detach().float().contiguous()
            gaussian = model.to_representation(inputs["points"], features)[0]
            # Read the actual official PLY export, including inverse activations and y-up transform.
            with tempfile.TemporaryDirectory(dir=trace.directory) as temporary:
                path = Path(temporary) / "oracle.ply"
                gaussian.save_ply(str(path))
                vertices = PlyData.read(str(path))["vertex"].data
                data = __import__("numpy").stack([vertices[name] for name in vertices.dtype.names], axis=1)
                trace.emit("idol.ply", torch.from_numpy(data.copy()))
                trace.order.append("idol.ply")
        del model
        gc.collect()


def conditioning_probe(root, inputs, trace):
    import torch
    from diffusers.utils import import_utils as diffusers_imports
    # CPU oracle uses SDPA math; shared host packages may contain CUDA-only optional extensions.
    for optional in ["_flash_attn_available", "_flash_attn_3_available", "_aiter_available", "_sageattention_available", "_xformers_available"]:
        setattr(diffusers_imports, optional, False)
    from diffusers import AutoencoderKLFlux2
    from safetensors.torch import load_file
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    from idol.conditioning import ImageEncoder
    # Instantiate directly from local config/state; no from_pretrained or hub resolution.
    encoder = ImageEncoder.__new__(ImageEncoder)
    encoder.device = torch.device("cpu")
    dino = root / "dinov3-vith16plus"
    encoder.dinov3 = DINOv3ViTModel(DINOv3ViTConfig.from_json_file(str(dino / "config.json")))
    encoder.dinov3.load_state_dict(load_file(str(dino / "model.safetensors")), strict=True)
    encoder.dinov3.eval().requires_grad_(False)
    vae = root / "FLUX.2-dev/vae"
    encoder.flux2_vae = AutoencoderKLFlux2.from_config(json.loads((vae / "config.json").read_text()))
    encoder.flux2_vae.load_state_dict(load_file(str(vae / "diffusion_pytorch_model.safetensors")), strict=True)
    encoder.flux2_vae.to(torch.bfloat16).eval().requires_grad_(False)
    trace.hook(encoder.dinov3.embeddings, "idol.dino.embeddings")
    trace.order.append("idol.dino.embeddings")
    for i, layer in enumerate(encoder.dinov3.layer):
        trace.hook(layer, f"idol.dino.block.{i}")
        trace.order.append(f"idol.dino.block.{i}")
    trace.hook(encoder.dinov3.norm, "idol.dino.last_hidden_state")
    trace.order.extend(["idol.dino.last_hidden_state", "idol.condition.feature1"])
    vae = encoder.flux2_vae
    trace.hook(vae.encoder.conv_in, "idol.vae.conv_in", nhwc=True)
    trace.order.append("idol.vae.conv_in")
    for i, block in enumerate(vae.encoder.down_blocks):
        for j, resnet in enumerate(block.resnets):
            trace.hook(resnet, f"idol.vae.down.{i}.{j}", nhwc=True)
            trace.order.append(f"idol.vae.down.{i}.{j}")
    for module, name in [(vae.encoder.mid_block.resnets[0], "idol.vae.mid.resnet0"),
                         (vae.encoder.mid_block.attentions[0], "idol.vae.mid.attention"),
                         (vae.quant_conv, "idol.vae.moments_nhwc")]:
        trace.hook(module, name, nhwc=True)
        trace.order.append(name)
    trace.order.append("idol.condition.feature2")
    images = inputs["pixels"].permute(2, 0, 1)[None] / 255.
    trace.emit("idol.condition.feature1", encoder._dinov3_tokens(images))
    noise = inputs["vae_noise"].to(torch.bfloat16)
    with patch("diffusers.models.autoencoders.vae.randn_tensor", return_value=noise):
        tokens = encoder._flux2_tokens(images)
    trace.emit("idol.condition.feature2", tokens)




def background_probe(root, inputs, trace):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from idol.models import BiRefNet
    model = BiRefNet().eval().requires_grad_(False)
    model.load_state_dict(load_file(str(root / "idol/birefnet.safetensors")), strict=True)
    calls = [0]
    def backbone(module, arguments, maps):
        label = "full" if calls[0] == 0 else "half"
        calls[0] += 1
        for i, value in enumerate(maps):
            trace.emit(f"idol.background.{label}.stage.{i}.out", value, nhwc=True)
    model.bb.register_forward_hook(backbone)
    original = model._forward_enc
    def encoded(image):
        values = original(image)
        for i, value in enumerate(values):
            trace.emit(f"idol.background.multiscale.{i}", value if i < 3 else value[:, -3072:], nhwc=True)
        return values
    model._forward_enc = encoded
    trace.hook(model.squeeze_module, "idol.background.squeeze", nhwc=True)
    for i, block in enumerate([5,4,3,2]):
        trace.hook(getattr(model.decoder, f"ipt_blk{block}"), f"idol.background.image_patch.{i}", nhwc=True)
        trace.hook(getattr(model.decoder, f"decoder_block{4-i}"), f"idol.background.decoder.{i}", nhwc=True)
    trace.hook(model.decoder.conv_out1, "idol.background.logits", nhwc=True)
    trace.hook(model, "idol.background.alpha", nhwc=True)
    trace.emit("idol.background.input", inputs["background_normalized"])
    model(inputs["background_normalized"].permute(0,3,1,2).contiguous())
    names = ["input"]
    names += [f"{label}.stage.{i}.out" for label in ["full","half"] for i in range(4)]
    names += [f"multiscale.{i}" for i in range(4)] + ["squeeze"]
    names += [f"{part}.{i}" for i in range(4) for part in ["image_patch","decoder"]]
    names += ["logits","alpha"]
    trace.order.extend("idol.background."+name for name in names)

def preprocess_probe(inputs, trace):
    import numpy as np
    import torch
    from PIL import Image
    from idol.preprocess import preprocess_image
    rgba = inputs["rgba"].byte().numpy()
    result = preprocess_image(Image.fromarray(rgba, "RGBA"), background_remover=None,
                              image_size=64, crop_margin=1.2)
    trace.emit("idol.preprocess", torch.from_numpy(np.asarray(result).copy()))
    trace.order.append("idol.preprocess")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-source", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True, help="Parent containing idol/, dinov3-vith16plus/, FLUX.2-dev/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", default="denoiser,decoder,conditioning,background")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    components = args.components.split(",")
    if any(c not in {"denoiser", "decoder", "conditioning", "background"} for c in components) or len(components) != len(set(components)):
        parser.error("components must be unique denoiser,decoder,conditioning,background entries")
    if args.threads < 1:
        parser.error("threads must be positive")
    source = args.oracle_source.resolve()
    validate_source(source)
    versions = {name: importlib.metadata.version(name) for name in ["torch", "transformers", "diffusers", "safetensors"]}
    if versions["transformers"] != "4.56.0" or versions["diffusers"] != "0.36.0":
        raise ValueError(f"Expected transformers4.56.0/diffusers0.36.0, got {versions}")
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(source))
    import torch
    from safetensors.torch import save_file
    from torch.nn.attention import SDPBackend, sdpa_kernel
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True, exist_ok=True)
    model_root = args.models.resolve()
    config = json.loads((model_root / "idol/config.json").read_text())
    needed = {"idol/config.json"}
    if "denoiser" in components:
        needed.add("idol/denoiser.safetensors")
    if "decoder" in components:
        needed.update(["idol/decoder.safetensors", "idol/decoder_gs.safetensors"])
    if "conditioning" in components:
        needed.update(["dinov3-vith16plus/config.json", "dinov3-vith16plus/model.safetensors", "FLUX.2-dev/vae/config.json", "FLUX.2-dev/vae/diffusion_pytorch_model.safetensors"])
    if "background" in components:
        needed.add("idol/birefnet.safetensors")
    manifest = json.loads(Path(__file__).with_name("manifest.json").read_text())
    checked = {}
    for repository in manifest:
        for file in repository["files"]:
            relative = f"{repository['directory']}/{file['Path']}"
            if relative not in needed:
                continue
            path = model_root / relative
            digest = sha256(path)
            if path.stat().st_size != file["Size"] or digest != file["Sha256"]:
                raise ValueError(f"Pinned model hash/size mismatch: {relative}")
            checked[relative] = dict(sha256=digest, bytes=path.stat().st_size)
    if set(checked) != needed:
        raise ValueError(f"Missing pinned manifest entries: {needed - set(checked)}")
    rng = torch.Generator().manual_seed(SEED)
    inputs = {"latent": torch.randn((1, 32, 16), generator=rng), "camera": torch.randn((1, 1, 5), generator=rng),
              "feature1": torch.randn((1, 9, 1280), generator=rng), "feature2": torch.randn((1, 9, 128), generator=rng),
              "points": torch.rand((1, 12, 3), generator=rng),
              "pixels": torch.randint(0, 256, (SIDE, SIDE, 3), generator=rng).float(),
              "vae_noise": torch.randn((1, 32, SIDE // 8, SIDE // 8), dtype=torch.bfloat16, generator=rng).float()}
    rgba = torch.zeros((25, 37, 4), dtype=torch.float32)
    for y in range(25):
        for x in range(37):
            rgba[y, x] = torch.tensor([(7*x) % 256, (11*y) % 256, (5*(x+y)) % 256,
                                      (17*x+13*y) % 256 if 4 <= x < 33 and 3 <= y < 22 else 0])
    inputs["rgba"] = rgba
    inputs["background_normalized"] = ((inputs["pixels"][None] / 255. - torch.tensor([0.485,0.456,0.406])) / torch.tensor([0.229,0.224,0.225])).contiguous()
    trace = Trace(args.output)
    preprocess_probe(inputs, trace)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for component in ["denoiser", "decoder", "conditioning", "background"]:
            if component in components:
                print(f"Running pinned CPU {component} fixture", flush=True)
                if component == "background":
                    background_probe(model_root, inputs, trace)
                elif component == "conditioning":
                    conditioning_probe(model_root, inputs, trace)
                else:
                    {"denoiser": denoiser_probe, "decoder": decoder_probe}[component](model_root, config, inputs, trace)
                gc.collect()
    input_path = args.output / "inputs.safetensors"
    save_file(inputs, str(input_path))
    trace_path = trace.finish()
    metadata = dict(oracle_commit=COMMIT, seed=SEED, time=500., octree_resolution=16, image_side=SIDE,
                    components=components, versions=versions, torch_threads=args.threads,
                    attention="sdpa math; F16 decoder/denoiser blocks, F32 DINO, BF16 VAE",
                    models=checked, inputs_sha256=sha256(input_path), trace_sha256=sha256(trace_path),
                    gates_sha256=sha256(Path(__file__).with_name("gates.json")), source_manifest_sha256=sha256(Path(__file__).with_name("source-manifest.json")), checkpoint_count=len(trace.order))
    (args.output / "fixtures.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(trace.order)} checkpoints and exact input tensors to {args.output}")


if __name__ == "__main__":
    main()
