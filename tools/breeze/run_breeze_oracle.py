#!/usr/bin/env python3
"""Pinned official Breeze CPU eager Oracle; JSONL checkpoints and F32 sidecars.

Use --trace FILE.jsonl (or a directory). Provenance is FILE.jsonl.manifest.json.
The official greedy CFG path divides by temperature even without sampling, so
both stages use do_sample=False and temperature=1.0 rather than dividing by zero.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

ORACLE_COMMIT = "e2c5ac2f54fe15daa94237a7dbf31e446660a4c9"
PINNED = {"torch": "2.9.1", "torchaudio": "2.9.1", "transformers": "4.57.3", "qwen-tts": "0.1.1"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkout(checkout: Path) -> str:
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(checkout), *arguments], text=True).strip()
    commit = git("rev-parse", "HEAD")
    if commit != ORACLE_COMMIT:
        raise ValueError(f"Oracle commit is {commit}; expected {ORACLE_COMMIT}")
    dirty = git("status", "--porcelain", "--untracked-files=normal")
    if dirty:
        raise ValueError(f"Oracle checkout must be clean: {dirty}")
    return commit


def model_manifest(directory: Path) -> dict:
    files = {}
    # Only runtime inputs; generated GGUF/WAV files and README/assets are excluded.
    for base in (directory, directory / "audio_tokenizer"):
        for path in sorted(base.iterdir()):
            if path.is_file() and path.suffix in (".json", ".safetensors"):
                record = {"bytes": path.stat().st_size, "sha256": sha256(path)}
                if path.suffix == ".safetensors":
                    with path.open("rb") as source:
                        length = int.from_bytes(source.read(8), "little")
                        header = json.loads(source.read(length))
                    record["tensors"] = {name: value for name, value in header.items() if name != "__metadata__"}
                files[path.relative_to(directory).as_posix()] = record
    return files


def load_eager_codec(directory: Path):
    from qwen_tts import Qwen3TTSTokenizer
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config

    config = Qwen3TTSTokenizerV2Config.from_pretrained(directory)
    # Mimi chooses its attention class and caches the backend during construction.
    # Changing the outer model's backend after loading cannot replace that class.
    config._attn_implementation = "eager"
    codec = Qwen3TTSTokenizer.from_pretrained(
        str(directory), config=config, device_map="cpu", attn_implementation="eager")
    codec.model.eval()
    encoder = codec.model.encoder.encoder_transformer
    decoder = codec.model.decoder.pre_transformer
    state = {
        "encoder_cached_attention": encoder._attn_implementation,
        "encoder_layers": [{"class": type(layer.self_attn).__name__,
                            "attention": layer.self_attn.config._attn_implementation}
                           for layer in encoder.layers],
        "decoder_layers": [{"class": type(layer.self_attn).__name__,
                            "attention": layer.self_attn.config._attn_implementation}
                           for layer in decoder.layers],
    }
    if (state["encoder_cached_attention"] != "eager"
            or not all(layer == {"class": "MimiAttention", "attention": "eager"}
                       for layer in state["encoder_layers"])
            or not all(layer["attention"] == "eager" for layer in state["decoder_layers"])):
        raise ValueError(f"Codec did not construct eager attention: {state}")
    return codec, state


class Trace:
    """Forward hooks only observe outputs; the streamer provides the frame clock."""

    def __init__(self, path, torch, numpy):
        self.path, self.torch, self.np = path, torch, numpy
        self.output = path.open("w", encoding="utf-8")
        self.occurrences = Counter()
        self.frames = []
        self.prompt_seen = False
        self.depth_step = 0
        self.hooks = []
        self.previous_profile = None
        self.profile_installed = False
        self.attention_observers = []

    def write(self, record):
        self.output.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        self.output.flush()

    def ids(self, name, tensor):
        values = tensor.detach().cpu().reshape(-1).tolist()
        self.write({"name": name, "shape": [len(values)], "len": len(values), "token_ids": values})

    def checkpoint(self, name, tensor, layer=None, step=None):
        if not self.torch.is_tensor(tensor):
            raise TypeError(f"{name} did not return a tensor")
        value = tensor.detach().to(device="cpu", dtype=self.torch.float32)
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        values = value.contiguous().numpy().astype("<f4", copy=False).reshape(-1)
        if not self.np.isfinite(values).all():
            raise ValueError(f"{name}: non-finite checkpoint")
        occurrence = self.occurrences[name]
        self.occurrences[name] += 1
        suffix = f".{occurrence}" if occurrence else ""
        sidecar = Path(f"{self.path}.{name}{suffix}.f32")
        sidecar.write_bytes(values.tobytes())
        self.write({"name": name, "layer": layer, "step": step,
                    "shape": list(value.shape), "len": int(values.size), "finite": True,
                    "sum": sum(map(float, values)),
                    "min": float(values.min()) if values.size else None,
                    "max": float(values.max()) if values.size else None,
                    "head": values[:8].tolist(), "tail": values[-8:].tolist(),
                    "occurrence": occurrence, "binary_path": str(sidecar)})

    def attach(self, module, name, stage="text", layer=None, before=False, transform=None):
        def record(_module, arguments, kwargs, output=None):
            if before:
                value = arguments[0] if arguments else kwargs["hidden_states"]
            else:
                value = output[0] if isinstance(output, tuple) else output
            if transform is not None:
                value = transform(value)
            step = None if stage == "text" else (len(self.frames) if stage == "backbone" else self.depth_step)
            self.checkpoint(name, value, layer, step)
        if before:
            handle = module.register_forward_pre_hook(record, with_kwargs=True)
        else:
            handle = module.register_forward_hook(record, with_kwargs=True)
        self.hooks.append(handle)

    def install(self, model, fine_text_layer=None, fine_depth_layer=None, fine_backbone_layer=None):
        text = model.text_encoder
        self.attach(text.embed_tokens, "breeze.text.embedding")
        for index, layer in enumerate(text.layers):
            self.attach(layer, f"breeze.text.layer.{index}", layer=index)
        self.attach(text.norm, "breeze.text.norm")
        self.attach(model.text_encoder_proj, "breeze.text.projected")
        if fine_text_layer is not None:
            layer = text.layers[fine_text_layer]
            modules = {
                "pre_self_attn": layer.pre_self_attn_layernorm,
                "q_proj": layer.self_attn.q_proj, "k_proj": layer.self_attn.k_proj,
                "v_proj": layer.self_attn.v_proj, "q_norm": layer.self_attn.q_norm,
                "k_norm": layer.self_attn.k_norm, "attn_output": layer.self_attn,
                "post_self_attn": layer.post_self_attn_layernorm,
                "pre_ff": layer.pre_feedforward_layernorm, "gate": layer.mlp.gate_proj,
                "activation": layer.mlp.act_fn, "up": layer.mlp.up_proj,
                "ff_input": layer.mlp.down_proj, "ff": layer.mlp,
                "post_ff": layer.post_feedforward_layernorm,
            }
            for name, module in modules.items():
                self.attach(module, f"breeze.text.layer.{fine_text_layer}.{name}",
                            layer=fine_text_layer, before=name == "ff_input")
        backbone = model.backbone_model
        self.attach(backbone.layers[0], "breeze.backbone.input", "backbone", before=True)
        for index, layer in enumerate(backbone.layers):
            self.attach(layer, f"breeze.backbone.layer.{index}", "backbone", index)
        if fine_backbone_layer is not None:
            self.install_fine_causal(backbone.layers[fine_backbone_layer], fine_backbone_layer, "backbone")
        self.attach(backbone.norm, "breeze.backbone.norm", "backbone")
        self.attach(model.lm_head, "breeze.backbone.logits", "backbone")
        depth = model.depth_decoder.model

        def depth_clock(_module, arguments, kwargs):
            positions = kwargs.get("cache_position")
            if positions is not None:
                prediction = int(positions[-1])
            else:
                ids = kwargs.get("input_ids", arguments[0] if arguments else None)
                prediction = ids.shape[1] - 1
            if not 1 <= prediction < model.config.num_codebooks:
                raise ValueError(f"Unexpected depth position {prediction}")
            self.depth_step = len(self.frames) * (model.config.num_codebooks - 1) + prediction - 1

        self.hooks.append(model.depth_decoder.register_forward_pre_hook(depth_clock, with_kwargs=True))
        self.attach(depth.inputs_embeds_projector, "breeze.depth.input", "depth")
        for index, layer in enumerate(depth.layers):
            self.attach(layer, f"breeze.depth.layer.{index}", "depth", index)
        if fine_depth_layer is not None:
            self.install_fine_causal(depth.layers[fine_depth_layer], fine_depth_layer, "depth")
        self.attach(depth.norm, "breeze.depth.norm", "depth")
        self.attach(model.depth_decoder.codebooks_head, "breeze.depth.logits", "depth")

    def install_fine_causal(self, layer, index, stage):
        modules = {
            "pre_self_attn": layer.input_layernorm,
            "q_proj": layer.self_attn.q_proj, "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj, "attn_output": layer.self_attn,
            "pre_ff": layer.post_attention_layernorm, "gate": layer.mlp.gate_proj,
            "activation": layer.mlp.act_fn, "up": layer.mlp.up_proj,
            "ff_input": layer.mlp.down_proj, "ff": layer.mlp,
        }
        for name, module in modules.items():
            self.attach(module, f"breeze.{stage}.layer.{index}.{name}",
                        stage, index, before=name == "ff_input")
        if stage == "backbone":
            for name in ("q_norm", "k_norm"):
                self.attach(getattr(layer.self_attn, name), f"breeze.{stage}.layer.{index}.{name}",
                            stage, index, transform=lambda tensor: tensor.transpose(1, 2))
        self.attach(layer.self_attn.o_proj, f"breeze.{stage}.layer.{index}.attended", stage, index, before=True)
        self.observe_attention(layer.self_attn, index, stage)

    def observe_attention(self, attention, layer, stage):
        if stage == "backbone":
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward
        else:
            from models.breeze import apply_rotary_pos_emb, eager_attention_forward

        rope_code = apply_rotary_pos_emb.__code__
        eager_code = eager_attention_forward.__code__
        softmax_code = self.torch.nn.functional.softmax.__code__

        def record(name, tensor):
            step = len(self.frames) if stage == "backbone" else self.depth_step
            self.checkpoint(f"breeze.{stage}.layer.{layer}.{name}", tensor, layer, step)

        def observe(frame, event, result):
            if event != "return":
                return
            parent = frame.f_back
            if frame.f_code is rope_code and parent.f_locals.get("self") is attention:
                for name, tensor in zip(("q_rope", "k_rope"), result):
                    record(name, tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], -1))
            elif (frame.f_code is softmax_code and parent.f_code is eager_code
                  and parent.f_locals.get("module") is attention):
                record("attn_scores", frame.f_locals["input"][0])
            elif frame.f_code is eager_code and frame.f_locals.get("module") is attention:
                record("probabilities", result[1][0])

        # Observe unchanged Python functions; do not replace kernels or recompute evidence.
        self.attention_observers.append(observe)
        if not self.profile_installed:
            self.previous_profile = sys.getprofile()
            if self.previous_profile is not None:
                raise ValueError("Fine attention requires an unused Python profiler")

            def dispatch(frame, event, result):
                for observer in self.attention_observers:
                    observer(frame, event, result)

            sys.setprofile(dispatch)
            self.profile_installed = True

    def put(self, tensor):
        if not self.prompt_seen:
            self.prompt_seen = True  # GenerationMixin sends the input prompt first.
            return
        if tuple(tensor.shape) != (1, 16):
            raise ValueError(f"Unexpected streamed frame shape {tuple(tensor.shape)}")
        self.ids("breeze.frame", tensor)
        self.frames.append(tensor[0].tolist())

    def end(self):
        pass

    def close(self):
        if self.profile_installed:
            sys.setprofile(self.previous_profile)
        for handle in self.hooks:
            handle.remove()
        self.output.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--instruction")
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--ref-text")
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    fine = parser.add_mutually_exclusive_group()
    fine.add_argument("--fine-text-layer", type=int)
    fine.add_argument("--fine-text-layer0", dest="fine_text_layer", action="store_const", const=0)
    parser.add_argument("--fine-depth-layer", type=int)
    parser.add_argument("--fine-backbone-layer", type=int)
    args = parser.parse_args()
    if args.fine_text_layer is not None and not 0 <= args.fine_text_layer < 26:
        parser.error("--fine-text-layer must be in 0..25")
    if args.fine_depth_layer is not None and not 0 <= args.fine_depth_layer < 12:
        parser.error("--fine-depth-layer must be in 0..11")
    if args.fine_backbone_layer is not None and not 0 <= args.fine_backbone_layer < 28:
        parser.error("--fine-backbone-layer must be in 0..27")
    if args.frames < 1 or args.threads < 1:
        parser.error("--frames and --threads must be positive")
    if not math.isfinite(args.cfg_scale) or args.cfg_scale <= 0:
        parser.error("--cfg-scale must be finite and positive")
    if (args.ref_audio is not None) != bool(args.ref_text and args.ref_text.strip()):
        parser.error("--ref-audio and nonempty --ref-text must be provided together")
    if args.ref_audio is not None and not args.ref_audio.is_file():
        parser.error("--ref-audio does not exist")
    args.checkout, args.model_dir = args.checkout.resolve(), args.model_dir.resolve()
    commit = verify_checkout(args.checkout)
    versions = {name: importlib.metadata.version(name) for name in PINNED}
    if versions != PINNED:
        raise ValueError(f"Pinned dependencies required: {PINNED}; installed: {versions}")
    sys.path.insert(0, str(args.checkout))
    sys.dont_write_bytecode = True
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import numpy as np
    import torch
    import soundfile as sf
    from breeze_infer.runtime import load_runtime, set_all_seeds, update_generation_config_for_breeze
    from breeze_infer.templates import get_template, prepare_inputs, select_template_name

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    set_all_seeds(args.seed)
    print("Loading pinned CPU eager Oracle", flush=True)
    tokenizer, model, audio_tokenizer = load_runtime(args.model_dir, device="cpu", attn_implementation="eager")
    # The official runtime does not forward attn_implementation to its codec loader.
    del audio_tokenizer
    audio_tokenizer, codec_attention = load_eager_codec(args.model_dir / "audio_tokenizer")
    if model.dtype != torch.bfloat16 or audio_tokenizer.model.dtype != torch.float32:
        raise ValueError("Expected BF16 Breeze and F32 audio codec")
    update_generation_config_for_breeze(model, {
        "do_sample": False, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
        "max_new_tokens": args.frames, "repetition_penalty": 1.0,
        "depth_decoder_do_sample": False, "depth_decoder_temperature": 1.0,
        "depth_decoder_top_p": 1.0, "depth_decoder_top_k": 0,
        "depth_decoder_min_new_tokens": 15, "depth_decoder_max_new_tokens": 15,
    })
    request = {"id": "oracle", "text": args.text, "speaker": "S0"}
    if args.instruction and args.instruction.strip():
        request["instruction"] = args.instruction.strip()
    if args.ref_audio is not None:
        request.update(ref_audio_path=str(args.ref_audio.resolve()), ref_text=args.ref_text.strip())
    template_name = select_template_name(request)
    if args.cfg_scale != 1 and get_template(template_name).build_negative_segments is None:
        raise ValueError("CFG requires an instruction template with a negative prompt")
    trace_path = args.trace.resolve()
    if trace_path.is_dir() or not trace_path.suffix:
        trace_path = trace_path / "trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(f"{trace_path}.manifest.json")
    manifest = {"status": "running", "schema_version": 1,
                "oracle_commit": commit, "oracle_checkout": str(args.checkout),
                "python": {"executable": sys.executable, "version": platform.python_version()},
                "dependencies": versions, "model_dir": str(args.model_dir),
                "model_files": model_manifest(args.model_dir),
                "model_config": json.loads((args.model_dir / "config.json").read_text()),
                "codec_config": json.loads((args.model_dir / "audio_tokenizer/config.json").read_text()),
                "request": request, "template": template_name, "frames_requested": args.frames,
                "cfg_scale": args.cfg_scale, "threads": args.threads, "seed": args.seed,
                "fine_text_layer": args.fine_text_layer,
                "fine_depth_layer": args.fine_depth_layer, "codec_attention": codec_attention,
                "fine_backbone_layer": args.fine_backbone_layer,
                "device": "cpu", "attention": "eager", "model_dtype": "bfloat16", "codec_dtype": "float32",
                "do_sample": False, "temperature": 1.0, "depth_frames_per_backbone_frame": 15}
    if args.ref_audio:
        manifest["reference_audio"] = {"sha256": sha256(args.ref_audio), "bytes": args.ref_audio.stat().st_size}
    trace = Trace(trace_path, torch, np)
    try:
        with torch.inference_mode():
            inputs = prepare_inputs(tokenizer, audio_tokenizer, model, [request], get_template(template_name),
                                    guidance_scale=args.cfg_scale, guidance_scale_ref=None, guidance_scale_ins=None)
            trace.ids("breeze.prompt_ids", inputs["input_ids"])
            if inputs.get("cfg_negative_prompt_ids") is not None:
                trace.ids("breeze.cfg_negative_prompt_ids", inputs["cfg_negative_prompt_ids"])
            if inputs["input_values"] is not None:
                trace.ids("breeze.reference_codes", inputs["input_values"])
            trace.install(model, args.fine_text_layer, args.fine_depth_layer, args.fine_backbone_layer)
            # Keep every cfg_* field; filtering here would silently disable both CFG stages.
            result = model.generate(**inputs, max_new_tokens=args.frames, output_audio=False, streamer=trace)
            codes = result.sequences if hasattr(result, "sequences") else result
            if codes.ndim != 3 or codes.shape[0] != 1 or codes.shape[-1] != 16:
                raise ValueError(f"Unexpected generated codes shape: {tuple(codes.shape)}")
            if codes[0].cpu().tolist() != trace.frames or not trace.frames:
                raise ValueError("Streamed frames do not match generated codes")
            manifest["frames"] = trace.frames
            if args.out is not None:
                # Match official EOS handling: all-pad frames terminate generated audio.
                pad = (codes[0] == model.config.codebook_pad_token_id).all(dim=-1).nonzero()
                cutoff = int(pad[0]) if pad.numel() else codes.shape[1]
                if cutoff == 0:
                    raise ValueError("No audio frame was generated; cannot validate a WAV")
                wavs, sample_rate = audio_tokenizer.decode({"audio_codes": codes[0, :cutoff]})
                wav = np.asarray(wavs[0], dtype=np.float32)
                if len(wavs) != 1 or wav.ndim != 1 or not wav.size or not np.isfinite(wav).all():
                    raise ValueError("Codec returned invalid audio")
                trace.checkpoint("breeze.codec.audio", torch.from_numpy(wav))
                args.out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(args.out, wav, sample_rate, subtype="PCM_16")
                info = sf.info(args.out)
                if info.frames != wav.size or info.samplerate != sample_rate or info.channels != 1:
                    raise ValueError("Written WAV failed sample-count/rate/channel verification")
                manifest["audio"] = {"path": str(args.out.resolve()), "samples": int(wav.size),
                                     "sample_rate": sample_rate, "sha256": sha256(args.out)}
        verify_checkout(args.checkout)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        trace.close()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "complete", "trace": str(trace_path), "frames": len(trace.frames),
                      "manifest": str(manifest_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
