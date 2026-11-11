"""Independent F32 flow / HiFT Oracle with serialized, shared explicit noise.

Pinned official CosyVoice3 modules execute independently decoded component
GGUFs. HiFT calls its F32 public submodules, avoiding vendored inference's F64
F0 cast. This validates same-file implementation error, not speech quality.
"""

from __future__ import annotations

import argparse
import ast
from functools import wraps
import json
from pathlib import Path
import sys
import tempfile
import time
import types

import numpy as np

from .oracle import GATES, metrics, reference_weight, sha256
from .audio_oracle import runtime_activation_roundtrip

REVISION = "990d6a74bbf15c1765a6f7736fa839f423fd5b59"
ROTARY_PIN = "0cc071b1120ac391455540200ed78ee2390281a97f301eb99710f042f6586602"
PINS = {
    "e2_model.py": "1f41c5f85ea01310e52201121e353aeff7cb545804daf3d693983a8e04d2acca",
    "cosyvoice_lite/activation.py": "1a1b7a6e511d8dfda9d1d083ebf0fb0db42b278ccd167c23e32a193fb5d2560b",
    "cosyvoice_lite/common.py": "ba98f2e6a769e4873ae3a3aade0cb95ac3f1642fef144c8abc698ccb507d8b4f",
    "cosyvoice_lite/convolution.py": "af14da3b74559754901ecf8d064cdf29edb2ed4a48964440b046fb4b8929c962",
    "cosyvoice_lite/dit_modules.py": "ef5d56d718e3f4db471b265f1769aa26d2cc1211d74719ccb8fe66a443f3423e",
    "cosyvoice_lite/dit.py": "a841f4d96900223bc08064462556ee0905f914da8979a6266fca74c8e876fda3",
    "cosyvoice_lite/f0_predictor.py": "ac5c583093611058cd187ef2eb605d33ce18990f6f539f697fe23d78d0ef4fb9",
    "cosyvoice_lite/flow_matching.py": "66f3ffc4ba58b20ab39bcca328ab78c7270b9354bebc3c556bbc0b7fc0d15d81",
    "cosyvoice_lite/generator.py": "232a5d1e83c6c430676ab6ec6c26ebf6d09951dda0d539823c378d6dbb8b9b80",
    "cosyvoice_lite/mask.py": "a4d0b57e838759cbfe2bff37eaa7a59c18bf6f19b5a4be0f76477cf40e668048",
    "cosyvoice_lite/prelookahead.py": "8f1ea2c7449a1c23fcaec166394bbeb80f3505e5bfcc9e9eb7ac9ebdd9ec497c",
}

# Declared before any Rust run. F32 intermediates reuse the thinker runtime gate;
# nonlinear waveform reconstruction has an additional absolute-error bound.
INTERMEDIATE_GATE = dict(maximum_relative_rmse=0.001, minimum_cosine=0.999999)
WAVEFORM_GATE = dict(
    maximum_relative_rmse=0.01, minimum_cosine=0.9999, maximum_absolute_error=0.01
)


def load_reference(directory, rotary_source):
    import torch
    from einops import rearrange

    for name, expected in PINS.items():
        if sha256(directory / name) != expected:
            raise ValueError(f"reference does not match pinned source: {name}")
    if sha256(rotary_source) != ROTARY_PIN:
        raise ValueError("x-transformers 2.31.7 source does not match pin")
    rotary_tree = ast.parse(rotary_source.read_text())
    wanted = {
        "exists",
        "disable_autocast",
        "RotaryEmbedding",
        "rotate_half",
        "apply_rotary_pos_emb",
    }
    nodes = [
        node
        for node in rotary_tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted
    ]
    rotary = {
        "torch": torch,
        "Module": torch.nn.Module,
        "arange": torch.arange,
        "stack": torch.stack,
        "is_tensor": torch.is_tensor,
        "cat": torch.cat,
        "rearrange": rearrange,
        "autocast": torch.amp.autocast,
        "wraps": wraps,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(rotary_source), "exec"),
        rotary,
    )
    package = "_sophon_code2wav_reference"
    parent = types.ModuleType(package)
    parent.__path__ = [str(directory / "cosyvoice_lite")]
    sys.modules[package] = parent
    result = dict(rotary)
    for name in [
        "common",
        "mask",
        "convolution",
        "activation",
        "prelookahead",
        "f0_predictor",
        "flow_matching",
        "dit_modules",
        "dit",
        "generator",
    ]:
        path = directory / f"cosyvoice_lite/{name}.py"
        tree = ast.parse(path.read_text())
        # MelSpec is unused: drop its optional torchaudio import. Supply the exact
        # pinned x-transformers functions without installing the unrelated package.
        tree.body = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.Import)
                and any(alias.name == "torchaudio" for alias in node.names)
                or isinstance(node, ast.ImportFrom)
                and node.module == "x_transformers.x_transformers"
            )
        ]
        module = types.ModuleType(f"{package}.{name}")
        module.__file__, module.__package__ = str(path), package
        module.__dict__.update(
            RotaryEmbedding=rotary["RotaryEmbedding"],
            apply_rotary_pos_emb=rotary["apply_rotary_pos_emb"],
        )
        sys.modules[module.__name__] = module
        exec(compile(tree, str(path), "exec"), module.__dict__)
        result[name] = module
    builders = ast.parse((directory / "e2_model.py").read_text())
    builders.body = [
        node
        for node in builders.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in {"_Cfg", "build_decoder", "build_hift"}
    ]
    namespace = {
        "DiT": result["dit"].DiT,
        "CausalConditionalCFM": result["flow_matching"].CausalConditionalCFM,
        "CausalHiFTGenerator": result["generator"].CausalHiFTGenerator,
        "CausalConvRNNF0Predictor": result["f0_predictor"].CausalConvRNNF0Predictor,
    }
    exec(compile(builders, str(directory / "e2_model.py"), "exec"), namespace)
    result.update(namespace)
    return result


def remove_weight_norm(model):
    import torch

    for module in model.modules():
        if torch.nn.utils.parametrize.is_parametrized(module, "weight"):
            torch.nn.utils.parametrize.remove_parametrizations(
                module, "weight", leave_parametrized=True
            )


class RuntimeActivations:
    """Independent PreparedRows QDQ on native CPU Linear module inputs."""

    def __init__(self, model, kinds, names=None):
        import torch

        self.kinds, self.names = kinds, names or {}
        self.hooks, self.description, self.recorded = [], [], set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                target = self.names.get(f"{name}.weight", f"{name}.weight")
                if kinds[target] not in (0, 30):
                    self.hooks.append(
                        module.register_forward_pre_hook(
                            self.hook(target), with_kwargs=True
                        )
                    )

    def hook(self, name):
        def quantize(_module, args, kwargs):
            rounded = self.apply(args[0] if args else kwargs["input"], name)
            return (
                ((rounded, *args[1:]), kwargs)
                if args
                else (args, {**kwargs, "input": rounded})
            )

        return quantize

    def apply(self, value, name):
        import torch

        if value.dtype != torch.float32 or value.device.type != "cpu":
            raise ValueError("runtime activation Oracle requires CPU F32")
        kind = self.kinds[name]
        if kind in (0, 30):
            return value
        rounded = runtime_activation_roundtrip(value.detach().numpy(), kind)
        if name not in self.recorded:
            self.recorded.add(name)
            self.description.append(
                {
                    "tensor": name,
                    "weight_type": kind,
                    "activation": (
                        "f16" if kind == 1 else "q8_0" if kind in (2, 3, 8) else "q8_k"
                    ),
                }
            )
        return torch.from_numpy(np.ascontiguousarray(rounded))

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def component_state(path, component, precision="f32", kinds=None, scratch_dir=None):
    import gguf
    import torch
    from gguf.quants import dequantize

    reader = gguf.GGUFReader(str(path), mode="c")
    for key, expected in {
        "general.architecture": f"sophon_realtime_{component}",
        "sophon_realtime.component": component,
        "sophon_realtime.source_revision": REVISION,
        "sophon_realtime.requested_quant": precision,
    }.items():
        field = reader.fields.get(key)
        if field is None or bytes(field.parts[field.data[0]]).decode() != expected:
            raise ValueError(f"GGUF is not the pinned {precision} {component}: {key}")
    state = {}
    with tempfile.TemporaryFile(
        prefix=f"{component}-reference-", dir=scratch_dir
    ) as scratch:
        for tensor in reader.tensors:
            if tensor.name in state or int(tensor.tensor_type) not in {
                0,
                1,
                30,
                2,
                3,
                8,
                12,
                13,
                14,
            }:
                raise ValueError("Oracle requires unique supported component tensors")
            if kinds is not None:
                kinds[tensor.name] = int(tensor.tensor_type)
            state[tensor.name] = torch.from_numpy(
                reference_weight(tensor, dequantize, scratch)
            )
    return state


def build_flow(reference, state):
    import torch

    with torch.device("meta"):
        model = torch.nn.Module()
        model.input_embedding = torch.nn.Embedding(6561, 80)
        model.spk_embed_affine_layer = torch.nn.Linear(192, 80)
        model.pre_lookahead_layer = reference["prelookahead"].PreLookaheadLayer(
            80, 1024, 3
        )
        model.decoder = reference["build_decoder"]()
    model.load_state_dict(state, strict=True, assign=True)
    model.decoder.inference_cfg_rate = 0.0
    model.eval()
    return model


def build_hift(reference, state):
    import torch

    with torch.device("meta"):
        model = reference["build_hift"]()
        remove_weight_norm(model)
    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    return model


def set_excitation(source, noise):
    import torch

    # Nonpersistent buffers are explicit inputs, not checkpoint weights. Initial
    # phase at sample zero is discarded by the official 480:1 linear downsample.
    source.l_sin_gen.rand_ini = torch.zeros((1, 9), dtype=torch.float32)
    source.l_sin_gen.sine_waves = noise
    source.uv = torch.zeros(noise.shape[:2] + (1,), dtype=torch.float32)


def write_fixture(output, seed, *, token_count=4, mel_frames=8, finalize=True):
    if (
        token_count < 1
        or mel_frames < (1 if finalize else 9)
        or (not finalize and token_count < 4)
    ):
        raise ValueError(
            "fixture requires output tokens and enough mel/lookahead frames"
        )
    rng = np.random.default_rng(seed)
    tokens = ([1, 34, 95, 256] + [(i * 37 + 1) % 6561 for i in range(4, token_count)])[
        :token_count
    ]
    flow_frames = (token_count + 2 - (0 if finalize else 3)) * 2
    f0_frames = mel_frames - (0 if finalize else 3)
    inputs = {
        "tokens": tokens,
        "prompt_tokens": [3, 55],
        "prompt_mel": rng.normal(-4.0, 1.0, (80, 4)).astype(np.float32),
        "speaker": rng.normal(size=192).astype(np.float32),
        "initial_noise": rng.normal(size=(80, flow_frames)).astype(np.float32),
        "mel": rng.normal(-4.0, 1.0, (80, mel_frames)).astype(np.float32),
        "excitation_noise": rng.random((f0_frames * 480, 9), dtype=np.float32),
        "mel_frames": mel_frames,
        "streaming": True,
        "finalize": finalize,
        "steps": 4,
    }
    serialized = {
        key: value.ravel().tolist() if isinstance(value, np.ndarray) else value
        for key, value in inputs.items()
    }
    (output / "input.json").write_text(json.dumps(serialized, separators=(",", ":")))
    for name in ["initial_noise", "excitation_noise"]:
        inputs[name].astype("<f4", copy=False).tofile(output / f"{name}.f32")
    return inputs


def load_inputs(path):
    inputs = json.loads(path.read_text())
    frames = int(inputs["mel_frames"])
    for name in ["prompt_mel", "speaker", "initial_noise", "mel", "excitation_noise"]:
        inputs[name] = np.asarray(inputs[name], dtype=np.float32)
        if not np.isfinite(inputs[name]).all():
            raise ValueError(f"nonfinite input: {name}")
    inputs["prompt_mel"] = inputs["prompt_mel"].reshape(80, -1)
    inputs["initial_noise"] = inputs["initial_noise"].reshape(80, -1)
    inputs["mel"] = inputs["mel"].reshape(80, frames)
    inputs["excitation_noise"] = inputs["excitation_noise"].reshape(-1, 9)
    if inputs["speaker"].shape != (192,) or any(
        type(token) is not int or not 0 <= token < 6561
        for token in inputs["tokens"] + inputs["prompt_tokens"]
    ):
        raise ValueError("invalid speaker/token input")
    return inputs


def flow_reference(model, inputs):
    import torch
    import torch.nn.functional as F

    final, streaming = inputs["finalize"], inputs["streaming"]
    token = torch.tensor([inputs["prompt_tokens"] + inputs["tokens"]])
    hidden = model.input_embedding(token)
    hidden = (
        model.pre_lookahead_layer(hidden)
        if final
        else model.pre_lookahead_layer(hidden[:, :-3], hidden[:, -3:])
    )
    mu = hidden.repeat_interleave(2, dim=1).transpose(1, 2)
    speaker = model.spk_embed_affine_layer(
        F.normalize(torch.from_numpy(inputs["speaker"])[None], dim=1)
    )
    cond = torch.zeros_like(mu)
    prompt = torch.from_numpy(inputs["prompt_mel"])
    cond[:, :, : prompt.shape[1]] = prompt
    noise = torch.from_numpy(inputs["initial_noise"])[None]
    if noise.shape != mu.shape:
        raise ValueError("initial noise does not match conditioned flow frames")
    model.decoder.rand_noise = noise
    arrays = {"flow.mu": mu[0], "flow.cond": cond[0], "flow.speaker": speaker[0]}
    velocities = []
    hook = model.decoder.estimator.register_forward_hook(
        lambda _m, _a, result: velocities.append(result[:1].detach().clone())
    )
    try:
        mel, _ = model.decoder(
            mu,
            torch.ones_like(mu[:, :1]),
            inputs["steps"],
            spks=speaker,
            cond=cond,
            streaming=streaming,
        )
    finally:
        hook.remove()
    arrays.update(
        {f"flow.velocity.{i}": velocity[0] for i, velocity in enumerate(velocities)}
    )
    arrays["flow.mel"] = mel[0, :, prompt.shape[1] :]
    return {
        name: value.detach().float().numpy().copy() for name, value in arrays.items()
    }


def hift_reference(model, inputs):
    import torch

    mel = torch.from_numpy(inputs["mel"])[None]
    final = inputs["finalize"]
    # Intentionally call the F32 module, not inference()'s vendored F64 policy.
    f0 = model.f0_predictor(mel, finalize=final)
    noise = torch.from_numpy(inputs["excitation_noise"])[None]
    if noise.shape != (1, f0.shape[1] * 480, 9):
        raise ValueError("HiFT excitation noise does not match F0 duration")
    set_excitation(model.m_source, noise)
    source, _, _ = model.m_source(model.f0_upsamp(f0[:, None]).transpose(1, 2))
    source = source.transpose(1, 2)
    arrays = {"hift.f0": f0[0], "hift.source": source[0, 0]}
    residuals, hooks = {}, []
    for index, block in enumerate(model.resblocks):
        hooks.append(
            block.register_forward_hook(
                lambda _m, _a, result, i=index: residuals.__setitem__(
                    i, result.detach().clone()
                )
            )
        )
    hooks.append(
        model.conv_post.register_forward_hook(
            lambda _m, _a, result: arrays.__setitem__(
                "hift.spectrum", result[0].detach().clone()
            )
        )
    )
    try:
        waveform = model.decode(
            (
                mel
                if final
                else mel[:, :, : -model.f0_predictor.condnet[0].causal_padding]
            ),
            source,
            finalize=final,
        )
    finally:
        for hook in hooks:
            hook.remove()
    for stage in range(3):
        arrays[f"hift.stage.{stage}"] = (
            (residuals[stage * 3] + residuals[stage * 3 + 1] + residuals[stage * 3 + 2])
            / 3
        )[0]
    arrays["hift.waveform"] = waveform[0]
    return {
        name: value.detach().float().numpy().copy() for name, value in arrays.items()
    }


def comparison(name, actual, expected):
    result = metrics(np.asarray(actual).ravel(), np.asarray(expected).ravel())
    gate = WAVEFORM_GATE if name == "hift.waveform" else INTERMEDIATE_GATE
    result["gate"] = gate
    result["pass"] = bool(
        result["relative_rmse"] is not None
        and result["relative_rmse"] <= gate["maximum_relative_rmse"]
        and result["cosine"] >= gate["minimum_cosine"]
        and result["max_abs"] <= gate.get("maximum_absolute_error", float("inf"))
    )
    return result


def compare_trace(path, arrays):
    trace = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    comparisons = {}
    for name, expected in arrays.items():
        records = [row for row in trace if row["name"] == f"sophon.{name}"]
        if len(records) != 1:
            raise ValueError(f"expected one Rust checkpoint sophon.{name}")
        if records[0]["shape"] != list(expected.shape):
            raise ValueError(f"Rust checkpoint shape differs: {name}")
        comparisons[name] = comparison(
            name, np.fromfile(records[0]["binary_path"], dtype="<f4"), expected
        )
    return comparisons


def run(args):
    import torch
    from gguf.quants import dequantize
    import inspect

    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = (
        load_inputs(args.input)
        if args.input
        else write_fixture(
            args.output,
            args.seed,
            token_count=args.fixture_tokens,
            mel_frames=args.fixture_mel_frames,
            finalize=not args.fixture_nonfinal,
        )
    )
    if args.input:
        (args.output / "input.json").write_text(args.input.read_text())
        for name in ["initial_noise", "excitation_noise"]:
            inputs[name].astype("<f4", copy=False).tofile(args.output / f"{name}.f32")
    reference = load_reference(args.reference, args.rotary_source)
    kinds = {"flow": {}, "hift": {}}
    models = {
        component: builder(
            reference,
            component_state(
                path, component, args.precision, kinds[component], args.output
            ),
        )
        for component, builder, path in [
            ("flow", build_flow, args.flow_gguf),
            ("hift", build_hift, args.hift_gguf),
        ]
    }
    activations = (
        {name: RuntimeActivations(model, kinds[name]) for name, model in models.items()}
        if args.activation == "runtime"
        else {}
    )
    started = time.perf_counter()
    arrays = {
        **flow_reference(models["flow"], inputs),
        **hift_reference(models["hift"], inputs),
    }
    elapsed = time.perf_counter() - started
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite reference output: {name}")
        value.astype("<f4", copy=False).tofile(args.output / f"{name}.f32")
    result = {
        "semantics": "same-GGUF F32 compute; four-step cosine Euler CFG=0; independent F32 HiFT input mel",
        "seed": args.seed,
        "threads": args.threads,
        "torch": torch.__version__,
        "seconds": elapsed,
        "precision": args.precision,
        "activation": args.activation,
        "activation_projections": {
            name: activation.description for name, activation in activations.items()
        },
        "reference_pins": PINS,
        "x_transformers_source_sha256": ROTARY_PIN,
        "gguf_decoder_source_sha256": sha256(Path(inspect.getfile(dequantize))),
        "flow_gguf_sha256": sha256(args.flow_gguf),
        "hift_gguf_sha256": sha256(args.hift_gguf),
        "input_sha256": sha256(args.output / "input.json"),
        "noise_sha256": {
            name: sha256(args.output / f"{name}.f32")
            for name in ["initial_noise", "excitation_noise"]
        },
        "shapes": {name: list(value.shape) for name, value in arrays.items()},
        "gates": {"intermediate": INTERMEDIATE_GATE, "waveform": WAVEFORM_GATE},
        "f0_range": [float(arrays["hift.f0"].min()), float(arrays["hift.f0"].max())],
    }
    if args.calibrate_threads:
        torch.set_num_threads(args.calibrate_threads)
        repeated = {
            **flow_reference(models["flow"], inputs),
            **hift_reference(models["hift"], inputs),
        }
        result["reference_thread_calibration"] = {
            name: comparison(name, repeated[name], values)
            for name, values in arrays.items()
        }
        result["calibration_threads"] = [args.threads, args.calibrate_threads]
    if args.rust_trace:
        result["rust_comparisons"] = compare_trace(args.rust_trace, arrays)
        result["parity_pass"] = all(
            row["pass"] for row in result["rust_comparisons"].values()
        )
    for activation in activations.values():
        activation.close()
    (args.output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result), flush=True)
    if result.get("parity_pass") is False:
        raise SystemExit("code2wav numerical gates failed; inspect result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-gguf", type=Path, required=True)
    parser.add_argument("--hift-gguf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("target/sophon/reference/sophon_expressive"),
    )
    parser.add_argument(
        "--rotary-source",
        type=Path,
        default=Path("target/sophon/tasks/x-transformers-2.31.7-reference.py"),
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--fixture-tokens", type=int, default=4)
    parser.add_argument("--fixture-mel-frames", type=int, default=8)
    parser.add_argument("--fixture-nonfinal", action="store_true")
    parser.add_argument("--rust-trace", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=list(GATES), default="f32")
    parser.add_argument("--activation", choices=["f32", "runtime"], default="f32")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--calibrate-threads", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
