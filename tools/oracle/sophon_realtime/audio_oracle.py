"""Pinned portable Qwen3-ASR + Ultravox reference, using the actual audio GGUF.

The encoder implementation is loaded read-only from the pinned DuplexJev source.
Projector operations follow vLLM ced6857afa0ea7b2e3f0846a62e1394e90f15607.
The default keeps F32 activations; --activation runtime independently emulates
the CPU PreparedRows input contract. Optional linear input/output checkpoints
isolate matmul errors using identical inputs without replacing the full audio gate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
import wave
from pathlib import Path

import numpy as np

from .oracle import meets_gate, metrics, sha256
from tools.oracle.thx02.oracle import activation_roundtrip


def runtime_activation_roundtrip(values, weight_kind):
    """Independent QDQ for audio.rs's PreparedRows CPU matmul contract.

    F16 rounds activations to F16. BF16 and F32 keep F32 inputs. Q4_1,
    Q4_0 and Q8_0 share Q8_0 inputs; K weights consume signed-scale Q8_K.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 0 or not values.size or not np.isfinite(values).all():
        raise ValueError("runtime activations must be nonempty finite F32 rows")
    if weight_kind in (0, 30):
        return values
    if weight_kind == 1:
        with np.errstate(over="ignore"):
            rounded = values.astype(np.float16).astype(np.float32)
        if not np.isfinite(rounded).all():
            raise ValueError("F16 activation rounding overflow")
        return rounded
    # The independently tested THX helper has no Q4_1 weight decoder, but
    # its Q8_0 input contract is identical to Q4_0/Q8_0.
    return activation_roundtrip(values, 8 if weight_kind == 3 else weight_kind)


class ActivationQuantization:
    """Hook native encoder linears and explicitly round functional projector inputs."""

    def __init__(self, encoder, kinds):
        import torch

        self.kinds, self.hooks, self.description = kinds, [], []
        self.recorded = set()
        for name, module in encoder.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            target = f"audio_tower.{name}.weight"
            kind = self.kinds[target]
            if kind not in (0, 30):
                self.hooks.append(module.register_forward_pre_hook(self.hook(target), with_kwargs=True))

    def hook(self, name):
        def quantize(module, args, kwargs):
            value = args[0] if args else kwargs["input"]
            rounded = self.apply(value, name)
            if args:
                return (rounded, *args[1:]), kwargs
            return args, {**kwargs, "input": rounded}
        return quantize

    def apply(self, value, name):
        import torch

        if value.dtype != torch.float32 or value.device.type != "cpu":
            raise ValueError("runtime activation Oracle requires native CPU/F32 inputs")
        kind = self.kinds[name]
        rounded = runtime_activation_roundtrip(value.detach().numpy(), kind)
        if kind in (0, 30):
            return value
        if name not in self.recorded:
            self.recorded.add(name)
            self.description.append({"tensor": name, "weight_type": kind,
                                     "activation": "f16" if kind == 1 else "q8_0" if kind in (2, 3, 8) else "q8_k"})
        return torch.from_numpy(np.ascontiguousarray(rounded))

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def checkpoint_values(mel, encoder, projected, mel_lengths, encoder_lengths):
    """Match Rust's chunk-by-chunk traces, excluding the reference's batch padding."""
    if mel.ndim != 3 or encoder.ndim != 3 or mel.shape[0] != encoder.shape[0]:
        raise ValueError("audio checkpoints require matching padded batches")
    for values, lengths in ((mel, mel_lengths), (encoder, encoder_lengths)):
        width = values.shape[2] if values is mel else values.shape[1]
        if len(lengths) != values.shape[0] or any(n < 1 or n > width for n in lengths):
            raise ValueError("invalid audio checkpoint lengths")
    return {
        "mel": np.concatenate([row[:, :int(n)].ravel() for row, n in zip(mel, mel_lengths)]),
        "encoder": np.concatenate([row[:int(n)].ravel() for row, n in zip(encoder, encoder_lengths)]),
        "projected": projected.ravel(),
    }


def compare_projection_checkpoints(trace, tensors, kinds, activation):
    """Isolate shared matmuls using the identical pre-quantization Rust inputs.

    This diagnostic keeps the F32 implementation gate even for quantized weights;
    the independent waveform-to-embedding comparisons remain separate and required.
    """
    import torch

    groups = {}
    prefix = "sophon.audio.linear."
    for record in trace:
        if not record["name"].startswith(prefix):
            continue
        stem, _, stage = record["name"][len(prefix):].rpartition(".")
        if stage not in ("input", "output"):
            raise ValueError("invalid audio projection checkpoint stage")
        key = (stem, record.get("occurrence", 0))
        group = groups.setdefault(key, {})
        if stage in group:
            raise ValueError(f"duplicate projection checkpoint: {key} {stage}")
        group[stage] = record
    results = {}
    for (stem, occurrence), pair in groups.items():
        if set(pair) != {"input", "output"}:
            raise ValueError(f"missing paired projection checkpoint: {stem}")
        def read(record):
            shape = record["shape"]
            values = np.fromfile(record["binary_path"], dtype="<f4")
            if len(shape) != 2 or min(shape) < 1 or values.size != np.prod(shape):
                raise ValueError(f"invalid projection checkpoint shape: {stem}")
            return values.reshape(shape)
        x, actual = read(pair["input"]), read(pair["output"])
        weight_name = f"{stem}.weight"
        weight = tensors[weight_name]
        if weight.ndim != 2 or x.shape[1] != weight.shape[1] or actual.shape != (x.shape[0], weight.shape[0]):
            raise ValueError(f"projection weight/checkpoint shape mismatch: {stem}")
        if activation == "runtime":
            x = runtime_activation_roundtrip(x, kinds[weight_name])
        elif activation != "f32":
            raise ValueError(f"unknown activation contract: {activation}")
        expected = torch.nn.functional.linear(torch.from_numpy(x), weight, tensors.get(f"{stem}.bias"))
        comparison = metrics(actual.ravel(), expected.detach().numpy().ravel())
        comparison["pass"] = bool(meets_gate(comparison, "f32"))
        results[f"{stem}#{occurrence}"] = comparison
    return results


def load_reference(directory):
    package = "_sophon_audio_reference"
    module = types.ModuleType(package)
    module.__path__ = [str(directory)]
    sys.modules[package] = module
    for name in ("configuration_qwen3_asr_encoder", "modeling_qwen3_asr_encoder", "processing"):
        spec = importlib.util.spec_from_file_location(f"{package}.{name}", directory / f"{name}.py")
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = loaded
        spec.loader.exec_module(loaded)
    return sys.modules[f"{package}.modeling_qwen3_asr_encoder"], sys.modules[f"{package}.processing"]


def run(args):
    import gguf
    import torch
    from gguf.quants import dequantize
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    args.output.mkdir(parents=True, exist_ok=False)
    reference, processing = load_reference(args.reference)
    config = json.loads((args.model_dir / "thinker/audio/config.json").read_text())
    encoder_config = reference.Qwen3ASREncoderConfig(**config["audio_config"])
    reader = gguf.GGUFReader(str(args.gguf))
    fields = {}
    for key in ("general.architecture", "sophon_realtime.component", "sophon_realtime.source_revision"):
        field = reader.fields[key]
        fields[key] = bytes(field.parts[field.data[0]]).decode()
    if fields != {"general.architecture": "sophon_realtime_audio", "sophon_realtime.component": "audio", "sophon_realtime.source_revision": "990d6a74bbf15c1765a6f7736fa839f423fd5b59"}:
        raise ValueError("not the pinned Sophon audio component")
    state, projector, kinds = {}, {}, {}
    for tensor in reader.tensors:
        kinds[tensor.name] = int(tensor.tensor_type)
        values = dequantize(tensor.data, tensor.tensor_type).reshape(tuple(reversed(tensor.shape)))
        weight = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
        if tensor.name.startswith("audio_tower."):
            state[tensor.name.removeprefix("audio_tower.")] = weight
        elif tensor.name.startswith("multi_modal_projector."):
            projector[tensor.name.removeprefix("multi_modal_projector.")] = weight
        else:
            raise ValueError(f"unknown audio tensor: {tensor.name}")
    with torch.device("meta"):
        encoder = reference.Qwen3ASREncoder(encoder_config)
    encoder.load_state_dict(state, strict=True, assign=True)
    # This nonpersistent buffer is not in the checkpoint.
    encoder.positional_embedding = reference.SinusoidsPositionEmbedding(13, 896)
    encoder.eval()
    if set(projector) != {"ln_pre.weight", "linear_1.weight", "ln_mid.weight", "linear_2.weight"}:
        raise ValueError("projector tensor inventory mismatch")

    if args.wav:
        with wave.open(str(args.wav), "rb") as source:
            if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 16000):
                raise ValueError("oracle WAV must be mono PCM16 at 16kHz")
            pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").copy()
    else:
        positions = np.arange(args.samples, dtype=np.float64)
        pcm = np.rint(8000 * np.sin(positions * (2 * np.pi * 220 / 16000)) + 2000 * np.sin(positions * (2 * np.pi * 497 / 16000))).astype("<i2")
    wav = args.output / "input.wav"
    with wave.open(str(wav), "wb") as target:
        target.setparams((1, 2, 16000, len(pcm), "NONE", "not compressed"))
        target.writeframes(pcm.tobytes())
    samples = pcm.astype(np.float32) / 32768
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir / "thinker", local_files_only=True)
    extractor = WhisperFeatureExtractor.from_pretrained(args.model_dir / "thinker/audio", local_files_only=True)
    processor = processing.DuplexJevProcessor(extractor, tokenizer, stack_factor=2, encoder_ds_factor=8)
    processed = processor(audio=[samples], sampling_rate=16000, include_audio_num_chunks=True)
    activation = ActivationQuantization(encoder, kinds) if args.activation == "runtime" else None
    try:
        hidden = encoder(processed.audio_values, audio_len=processed.audio_lens).last_hidden_state
        # DuplexJevAudioTower pads every batch to the shared padded mel length.
        rows = int(reference._get_feat_extract_output_lengths(torch.tensor(processed.audio_values.shape[-1])))
        hidden = torch.nn.functional.pad(hidden, (0, 0, 0, max(0, rows - hidden.shape[1])))[:, :rows]
        stacked = torch.nn.functional.pad(hidden, (0, 0, 0, hidden.shape[1] % 2)).reshape(hidden.shape[0], -1, 1792)
        def rms(x, weight):
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight
        def linear(x, name):
            if activation is not None:
                x = activation.apply(x, f"multi_modal_projector.{name}.weight")
            return torch.nn.functional.linear(x, projector[f"{name}.weight"])
        projected = linear(rms(stacked, projector["ln_pre.weight"]), "linear_1")
        first, second = projected.chunk(2, dim=-1)
        projected = first * torch.nn.functional.silu(second)
        projected = linear(rms(projected, projector["ln_mid.weight"]), "linear_2")
        projected = torch.cat([row[:int(n)] for row, n in zip(projected, processed.audio_token_len)])
    finally:
        if activation is not None:
            activation.close()
    arrays = {"mel": processed.audio_values.numpy(), "encoder": hidden.numpy(), "projected": projected.numpy()}
    for name, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite reference output: {name}")
        values.tofile(args.output / f"{name}.f32")
    result = {"semantics": "independent same-GGUF runtime activation QDQ with F32 accumulation" if activation is not None else "independent same-GGUF F32 compute", "activation": args.activation, "activation_quantization": activation.description if activation is not None else "float32", "torch": torch.__version__, "gguf_sha256": sha256(args.gguf), "source": fields, "reference_sha256": sha256(args.reference / "modeling_qwen3_asr_encoder.py"), "input_sha256": sha256(wav), "mel_lengths": processed.audio_lens.tolist(), "audio_token_lengths": processed.audio_token_len.tolist(), "shapes": {name: list(values.shape) for name, values in arrays.items()}, "precision": args.precision}
    if args.rust_trace:
        checkpoints = checkpoint_values(
            arrays["mel"], arrays["encoder"], arrays["projected"],
            processed.audio_lens.tolist(),
            reference._get_feat_extract_output_lengths(processed.audio_lens).tolist(),
        )
        trace = [json.loads(line) for line in args.rust_trace.read_text().splitlines()]
        result["comparisons"] = {}
        for name, expected in checkpoints.items():
            records = [record for record in trace if record["name"] == f"sophon.audio.{name}"]
            if not records:
                raise ValueError(f"missing Rust {name} checkpoint")
            actual = np.concatenate([np.fromfile(record["binary_path"], dtype="<f4") for record in records])
            comparison = metrics(actual, expected.ravel())
            comparison["pass"] = bool(
                comparison["max_abs"] <= 0.001
                and comparison["relative_rmse"] is not None
                and comparison["relative_rmse"] <= 0.0001
                and comparison["cosine"] >= 0.99999999
            ) if name == "mel" else bool(meets_gate(comparison, args.precision))
            result["comparisons"][name] = comparison
        result["pass"] = all(value["pass"] for value in result["comparisons"].values())
        tensors = {f"audio_tower.{name}": value for name, value in state.items()}
        tensors.update({f"multi_modal_projector.{name}": value for name, value in projector.items()})
        isolated = compare_projection_checkpoints(trace, tensors, kinds, args.activation)
        if isolated:
            result["linear_comparisons"] = isolated
            result["pass"] = result["pass"] and all(value["pass"] for value in isolated.values())
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if result.get("pass") is False:
        raise SystemExit("audio component parity gate failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rust-trace", type=Path)
    parser.add_argument("--wav", type=Path)
    parser.add_argument("--samples", type=int, default=20480)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--precision", default="f32")
    parser.add_argument("--activation", choices=("f32", "runtime"), default="f32",
                        help="F32 reference (default) or PreparedRows CPU activation QDQ for kernel parity")
    args = parser.parse_args()
    if args.threads < 1 or args.samples < 1:
        parser.error("threads and samples must be positive")
    run(args)


if __name__ == "__main__":
    main()
