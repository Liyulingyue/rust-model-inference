#!/usr/bin/env python3
"""Pinned, read-only YuE2 reference trace entrypoint."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
import re
import sys
from pathlib import Path


# Apple Accelerate does not honor torch.set_num_threads() for BLAS calls.
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


WHEEL_SHA256 = "8801e2c0d969db02df78d2994150b4ccd86077d87c24fdb8509b1f6f31462641"


def bounded_int(name: str, lower: int, upper: int):
    def parse(value: str) -> int:
        parsed = int(value)
        if not lower <= parsed <= upper:
            raise argparse.ArgumentTypeError(f"{name} must be in [{lower}, {upper}]")
        return parsed

    return parse


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--wheel", type=Path, required=True)
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--vae-dir", type=Path)
    result.add_argument("--phase", choices=("tokenizer", "ar", "nar", "vae", "e2e"), required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--style", default="")
    result.add_argument("--lyrics", default="")
    result.add_argument("--seed", type=bounded_int("seed", 0, 2**63 - 1), default=831001)
    result.add_argument("--max-abc-tokens", type=bounded_int("max ABC tokens", 1, 4096), default=4096)
    result.add_argument("--max-music-tokens", type=bounded_int("max music tokens", 1, 9000), default=9000)
    result.add_argument("--latent-frames", type=bounded_int("latent frames", 1, 24_576), default=1)
    result.add_argument("--steps", type=bounded_int("steps", 1, 64), default=32)
    result.add_argument("--prefill-only", action="store_true")
    result.add_argument("--context", type=bounded_int("context", 1, 24_576), default=24_576)
    result.add_argument("--core", type=bounded_int("core", 1, 24_576), default=16)
    result.add_argument("--halo", type=bounded_int("halo", 16, 24_576), default=16)
    return result


def verify_wheel(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != WHEEL_SHA256:
        raise ValueError(f"YuE2 wheel SHA256 mismatch: {digest}")


def inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def configure_torch() -> None:
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def trace_tokenizer(model_dir: Path, text: str) -> dict[str, object]:
    from yue2.tokenization_yue2 import YuE2TextTokenizer

    tokenizer = YuE2TextTokenizer.from_pretrained(model_dir)
    ids = tokenizer.encode(text)
    return {
        "phase": "tokenizer",
        "ids": ids,
        "decoded_hex": tokenizer.decode(ids).encode("utf-8").hex(),
    }


class TraceSink:
    def __init__(self, out: Path, torch_module):
        self.out = out
        self.torch = torch_module
        self.step = 0
        self.active_layer = None
        self.rotary_call = 0
        self.counts = defaultdict(int)
        self.buffers = defaultdict(dict)
        if out.exists():
            raise FileExistsError(f"refusing to overwrite trace: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record: dict[str, object]) -> None:
        with self.out.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def tensor(self, name: str, value, shape: tuple[int, ...], layer=None) -> None:
        tensor = value.detach().to(device="cpu", dtype=self.torch.float32).contiguous().reshape(shape)
        occurrence = self.counts[name]
        self.counts[name] += 1
        filename = re.sub(r"[^0-9A-Za-z_.-]+", "_", name) + f".{occurrence}.f32"
        tensor.numpy().astype("<f4", copy=False).tofile(self.out.parent / filename)
        self._append({
            "name": name,
            "layer": layer,
            "step": self.step,
            "occurrence": occurrence,
            "shape": list(shape),
            "len": tensor.numel(),
            "path": filename,
        })

    def tokens(self, name: str, values: list[int]) -> None:
        occurrence = self.counts[name]
        self.counts[name] += 1
        self._append({
            "name": name,
            "layer": None,
            "step": None,
            "occurrence": occurrence,
            "shape": [len(values)],
            "len": len(values),
            "token_ids": values,
        })


def install_ar_hooks(model, modeling, sink: TraceSink):
    hidden = model.config.hidden_size
    q_heads = model.config.num_attention_heads
    kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim
    ffn = model.config.intermediate_size
    vocab = model.config.vocab_size

    model.model.embed_tokens.register_forward_hook(
        lambda _module, _inputs, output: sink.tensor("yue2.ar.embedding", output, (hidden,))
    )
    for layer_index, layer in enumerate(model.model.layers):
        buffer = sink.buffers[layer_index]

        def attn_norm(_module, _inputs, output, layer_index=layer_index):
            sink.active_layer = layer_index
            sink.rotary_call = 0
            sink.tensor("yue2.ar.attn_norm", output, (hidden,), layer_index)

        layer.input_layernorm.register_forward_hook(attn_norm)
        layer.self_attn.v_proj.register_forward_hook(
            lambda _module, _inputs, output, buffer=buffer: buffer.__setitem__("v", output)
        )
        layer.self_attn.q_norm.register_forward_hook(
            lambda _module, _inputs, output, buffer=buffer: buffer.__setitem__("q", output)
        )

        def k_norm(_module, _inputs, output, layer_index=layer_index, buffer=buffer):
            buffer["k"] = output
            sink.tensor("yue2.ar.q", buffer.pop("q"), (q_heads, head_dim), layer_index)
            sink.tensor("yue2.ar.k", buffer.pop("k"), (kv_heads, head_dim), layer_index)
            sink.tensor("yue2.ar.v", buffer.pop("v"), (kv_heads, head_dim), layer_index)

        layer.self_attn.k_norm.register_forward_hook(k_norm)
        layer.self_attn.o_proj.register_forward_pre_hook(
            lambda _module, inputs, layer_index=layer_index: sink.tensor(
                "yue2.ar.attn", inputs[0], (q_heads * head_dim,), layer_index
            )
        )
        layer.self_attn.o_proj.register_forward_hook(
            lambda _module, _inputs, output, layer_index=layer_index: sink.tensor(
                "yue2.ar.attn_output", output, (hidden,), layer_index
            )
        )
        layer.post_attention_layernorm.register_forward_pre_hook(
            lambda _module, inputs, layer_index=layer_index: sink.tensor(
                "yue2.ar.attn_residual", inputs[0], (hidden,), layer_index
            )
        )
        layer.post_attention_layernorm.register_forward_hook(
            lambda _module, _inputs, output, layer_index=layer_index: sink.tensor(
                "yue2.ar.ffn_norm", output, (hidden,), layer_index
            )
        )
        layer.mlp.gate_proj.register_forward_hook(
            lambda _module, _inputs, output, layer_index=layer_index: sink.tensor(
                "yue2.ar.ffn_gate", output, (ffn,), layer_index
            )
        )
        layer.mlp.up_proj.register_forward_hook(
            lambda _module, _inputs, output, layer_index=layer_index: sink.tensor(
                "yue2.ar.ffn_up", output, (ffn,), layer_index
            )
        )
        layer.mlp.down_proj.register_forward_hook(
            lambda _module, _inputs, output, layer_index=layer_index: sink.tensor(
                "yue2.ar.ffn_down", output, (hidden,), layer_index
            )
        )
        layer.register_forward_hook(
            lambda _module, _inputs, output, layer_index=layer_index: sink.tensor(
                "yue2.ar.ffn_residual", output, (hidden,), layer_index
            )
        )

    original_apply_rotary = modeling._apply_rotary

    def apply_rotary(value, cos, sin):
        output = original_apply_rotary(value, cos, sin)
        name = "yue2.ar.rope_q" if sink.rotary_call == 0 else "yue2.ar.rope_k"
        heads = q_heads if sink.rotary_call == 0 else kv_heads
        sink.tensor(name, output, (heads, head_dim), sink.active_layer)
        sink.rotary_call += 1
        return output

    modeling._apply_rotary = apply_rotary
    model.model.norm.register_forward_hook(
        lambda _module, _inputs, output: sink.tensor("yue2.ar.final_norm", output, (hidden,))
    )
    model.lm_head.register_forward_hook(
        lambda _module, _inputs, output: sink.tensor(
            "yue2.ar.logits", output, (vocab,)
        )
    )
    return lambda: setattr(modeling, "_apply_rotary", original_apply_rotary)


def trace_ar(args) -> None:
    import torch
    from yue2 import modeling_yue2
    from yue2.modeling_yue2 import StaticKVCache, YuE2ForCausalLM
    from yue2.protocol import Sampling, SongRequest, token_prefixes
    from yue2.sampling import distribution
    from yue2.tokenization_yue2 import YuE2TextTokenizer

    model = YuE2ForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to("cpu")
    tokenizer = YuE2TextTokenizer(args.model_dir / "qwen.tiktoken")
    request = SongRequest(style=args.style, lyrics=args.lyrics, seed=args.seed)
    sink = TraceSink(args.out, torch)
    restore_rotary = install_ar_hooks(model, modeling_yue2, sink)

    def generate(prefix: list[int], sampling, phase: str) -> list[int]:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        cache = StaticKVCache(
            num_layers=model.config.num_hidden_layers,
            batch_size=1,
            num_kv_heads=model.config.num_key_value_heads,
            max_seq_len=len(prefix) + sampling.max_tokens,
            head_dim=model.config.head_dim,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        logits = None
        for position, token in enumerate(prefix):
            sink.step = position
            logits = model(
                torch.tensor([[token]], dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            ).logits[:, -1, :]
        history = []
        end = 151848 if phase == "abc" else 151852
        for step in range(sampling.max_tokens):
            scores = distribution(logits, sampling, history, step, phase)
            if sampling.temperature == 0:
                token = int(scores.argmax(-1).item())
            else:
                token = int(torch.multinomial(scores.softmax(-1), 1, generator=generator).item())
            if token == end:
                break
            history.append(token)
            if step + 1 < sampling.max_tokens:
                sink.step = len(prefix) + len(history) - 1
                logits = model(
                    torch.tensor([[token]], dtype=torch.long),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                ).logits[:, -1, :]
        return history

    if args.phase == "e2e":
        abc_sampling = Sampling(.7, .9, 30, 1.005, 100, 32, args.max_abc_tokens)
        semantic_sampling = Sampling(1., .95, 100, 1.2, 50, 200, args.max_music_tokens)
    else:
        abc_sampling = Sampling(.0, .9, 30, 1.005, 100, args.max_abc_tokens, args.max_abc_tokens)
        semantic_sampling = Sampling(
            .0, .95, 100, 1.2, 50, args.max_music_tokens, args.max_music_tokens
        )
    try:
        with torch.inference_mode():
            abc_prefix = token_prefixes(request, tokenizer)
            if args.phase == "ar":
                sink.tokens("yue2.abc.prefix_ids", abc_prefix)
            abc_ids = generate(abc_prefix, abc_sampling, "abc")
            sink.tokens("yue2.abc.generated_ids", abc_ids)
            semantic_prefix = token_prefixes(request, tokenizer, abc_ids)
            if args.phase == "ar":
                sink.tokens("yue2.semantic.prefix_ids", semantic_prefix)
            semantic_ids = generate(semantic_prefix, semantic_sampling, "semantic")
            sink.tokens("yue2.semantic.generated_ids", semantic_ids)
    finally:
        restore_rotary()
    return semantic_prefix, [int(token) - 151_853 for token in semantic_ids], sink


def trace_nar(args, prefix=None, codec=None, sink=None):
    import torch
    import torch.nn.functional as F
    from yue2.modeling_yue2 import YuE2ForCausalLM
    from yue2.nar import CachedNAR, song_chunks

    model = YuE2ForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to("cpu")
    prefix = [151_643, 151_851] if prefix is None else prefix
    codec = list(range(args.latent_frames)) if codec is None else codec
    chunks = song_chunks(prefix, codec, args.seed, args.context)
    sink = TraceSink(args.out, torch) if sink is None else sink
    sink.step = None
    sink.tensor(
        "yue2.nar.noise",
        torch.cat([chunk.noise for chunk in chunks]),
        (len(codec), 64),
    )
    ranges = []
    start = 0
    output = []
    for chunk in chunks:
        end = start + len(chunk.noise)
        ranges.extend((start, end))
        start = end
    sink.tokens("yue2.nar.chunk_ranges", ranges)

    for chunk in chunks:
        engine = CachedNAR(model, chunk)
        for layer_index, (keys, values) in enumerate(engine.cache):
            sink.tensor(
                "yue2.nar.prefix_k",
                keys,
                (engine.ar_length, model.config.num_key_value_heads, model.config.head_dim),
                layer_index,
            )
            sink.tensor(
                "yue2.nar.prefix_v",
                values,
                (engine.ar_length, model.config.num_key_value_heads, model.config.head_dim),
                layer_index,
            )

        if args.prefill_only:
            engine.close()
            continue

        def velocity(state, raw_t):
            x_nar = F.pad(state, (0, 0, 1, 1))
            shifted = model._shift_t_value(raw_t, engine.device, engine.dtype)
            x = model.vae2llm(x_nar[None])
            time = model.time_embedder(shifted.expand(engine.nar_length))
            sink.tensor("yue2.nar.time_embedding", time[0], (model.config.hidden_size,))
            sink.tensor(
                "yue2.nar.position_embedding",
                engine.pos_emb,
                (engine.nar_length, model.config.hidden_size),
            )
            x = x + time[None]
            x = x + engine.pos_emb
            sink.tensor(
                "yue2.nar.input", x, (engine.nar_length, model.config.hidden_size)
            )
            for layer_index, (layer, (ar_k, ar_v)) in enumerate(
                zip(model.model.layers, engine.cache)
            ):
                normed = layer.nar_input_layernorm(x)
                sink.tensor(
                    "yue2.nar.attn_norm",
                    normed,
                    (engine.nar_length, model.config.hidden_size),
                    layer_index,
                )
                q, k, v = layer.nar_self_attn.project_qkv(normed, engine.cos, engine.sin)
                sink.tensor(
                    "yue2.nar.q",
                    q,
                    (engine.nar_length, model.config.num_attention_heads, model.config.head_dim),
                    layer_index,
                )
                sink.tensor(
                    "yue2.nar.k",
                    k,
                    (engine.nar_length, model.config.num_key_value_heads, model.config.head_dim),
                    layer_index,
                )
                sink.tensor(
                    "yue2.nar.v",
                    v,
                    (engine.nar_length, model.config.num_key_value_heads, model.config.head_dim),
                    layer_index,
                )
                all_k = torch.cat((ar_k, k[0]))
                all_v = torch.cat((ar_v, v[0]))
                attention = engine._attention(q[0], all_k, all_v)
                flattened = attention.flatten(1)
                sink.tensor(
                    "yue2.nar.attn",
                    flattened,
                    (engine.nar_length, model.config.num_attention_heads * model.config.head_dim),
                    layer_index,
                )
                projected = layer.nar_self_attn.o_proj(flattened[None])
                sink.tensor(
                    "yue2.nar.attn_output",
                    projected,
                    (engine.nar_length, model.config.hidden_size),
                    layer_index,
                )
                x = x + projected
                sink.tensor(
                    "yue2.nar.attn_residual",
                    x,
                    (engine.nar_length, model.config.hidden_size),
                    layer_index,
                )
                ffn_norm = layer.nar_pre_mlp_layernorm(x)
                sink.tensor(
                    "yue2.nar.ffn_norm",
                    ffn_norm,
                    (engine.nar_length, model.config.hidden_size),
                    layer_index,
                )
                gate = layer.nar_mlp.gate_proj(ffn_norm)
                up = layer.nar_mlp.up_proj(ffn_norm)
                sink.tensor(
                    "yue2.nar.ffn_gate",
                    F.silu(gate) * up,
                    (engine.nar_length, model.config.intermediate_size),
                    layer_index,
                )
                sink.tensor(
                    "yue2.nar.ffn_up",
                    up,
                    (engine.nar_length, model.config.intermediate_size),
                    layer_index,
                )
                down = layer.nar_mlp.down_proj(F.silu(gate) * up)
                sink.tensor(
                    "yue2.nar.ffn_down",
                    down,
                    (engine.nar_length, model.config.hidden_size),
                    layer_index,
                )
                x = x + down
                sink.tensor(
                    "yue2.nar.ffn_residual",
                    x,
                    (engine.nar_length, model.config.hidden_size),
                    layer_index,
                )
            normed = model.model.norm(x)
            sink.tensor(
                "yue2.nar.final_norm",
                normed,
                (engine.nar_length, model.config.hidden_size),
            )
            return model.llm2vae(normed)[0, 1:-1]

        state = chunk.noise.to(device=engine.device, dtype=engine.dtype)
        dt = 1.0 / args.steps
        for step in range(args.steps):
            time = 1.0 - step * dt
            raw = torch.logit(torch.tensor(time, dtype=torch.float64)).clamp(-20, 20).item()
            first = velocity(state, raw)
            sink.step = step
            sink.tensor("yue2.nar.velocity_first", first, tuple(first.shape))
            sink.step = None
            middle = state - first * (dt / 2)
            raw_midpoint = torch.logit(
                torch.tensor(time - dt / 2, dtype=torch.float64)
            ).clamp(-20, 20).item()
            midpoint = velocity(middle, raw_midpoint)
            sink.step = step
            sink.tensor("yue2.nar.velocity_midpoint", midpoint, tuple(midpoint.shape))
            sink.step = None
            state = state - midpoint * dt
        sink.tensor("yue2.nar.latents", state.float(), tuple(state.shape))
        output.append(state.float())
        engine.close()
    return torch.cat(output, dim=0) if output else None


def install_vae_hooks(vae, sink: TraceSink) -> None:
    for index, layer in enumerate(vae.decoder.layers[:9]):
        layer.register_forward_hook(
            lambda _module, _input, output, index=index: sink.tensor(
                "yue2.vae.decoder", output, tuple(output.shape), index
            )
        )
        if hasattr(layer, "layers") and type(layer).__name__ == "DecoderBlock":
            for child in layer.layers:
                child.register_forward_hook(
                    lambda _module, _input, output, index=index: sink.tensor(
                        "yue2.vae.decoder.block", output, tuple(output.shape), index
                    )
                )


def trace_vae(args) -> None:
    import torch
    from yue2.modeling_vae import YuE2VAE

    if args.vae_dir is None:
        raise ValueError("--vae-dir is required for VAE tracing")
    vae = YuE2VAE.from_pretrained(args.vae_dir, decoder_only=True, device="cpu")
    sink = TraceSink(args.out, torch)
    sink.step = None
    install_vae_hooks(vae, sink)
    frames = args.latent_frames
    values = (torch.arange(frames * 64, dtype=torch.float32) % 11) * 0.01
    latents = values.reshape(1, frames, 64).transpose(1, 2).contiguous()
    full = vae.decode(latents)
    sink.tensor("yue2.vae.full", full, tuple(full.shape))
    tiled = vae.decode_tiled(latents, core_frames=args.core, halo_frames=args.halo)
    sink.tensor("yue2.vae.tiled", tiled, tuple(tiled.shape))


def trace_e2e(args) -> None:
    import torch
    from yue2.modeling_vae import YuE2VAE

    if args.vae_dir is None:
        raise ValueError("--vae-dir is required for end-to-end tracing")
    prefix, codec, sink = trace_ar(args)
    latents = trace_nar(args, prefix=prefix, codec=codec, sink=sink)
    vae = YuE2VAE.from_pretrained(args.vae_dir, decoder_only=True, device="cpu")
    install_vae_hooks(vae, sink)
    channel_major = latents.T.unsqueeze(0).contiguous()
    audio = vae.decode_tiled(
        channel_major, core_frames=args.core, halo_frames=args.halo
    )
    if not torch.isfinite(audio).all():
        raise ValueError("YuE2 Oracle VAE produced non-finite audio")


def main() -> int:
    args = parser().parse_args()
    verify_wheel(args.wheel)
    for directory in (args.model_dir, args.vae_dir):
        if directory is not None and inside(args.out, directory):
            raise ValueError(f"refusing to write Oracle output inside model directory: {directory}")

    sys.path.insert(0, str(args.wheel.resolve()))
    import yue2  # noqa: F401

    if args.phase == "tokenizer":
        record = trace_tokenizer(args.model_dir, args.style)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    elif args.phase == "ar":
        configure_torch()
        trace_ar(args)
    elif args.phase == "nar":
        configure_torch()
        trace_nar(args)
    elif args.phase == "vae":
        configure_torch()
        trace_vae(args)
    elif args.phase == "e2e":
        configure_torch()
        trace_e2e(args)
    else:
        raise AssertionError(f"unhandled phase: {args.phase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
