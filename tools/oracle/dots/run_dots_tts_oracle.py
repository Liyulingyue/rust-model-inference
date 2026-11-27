#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import wave
from pathlib import Path

PINNED_SHA = "32407a55228630475c48ecdb2c4e2c0f9c09e030"
TARGET_PATCH_BUDGET = 2


def existing_path(value: str) -> Path:
    return Path(value).expanduser().resolve(strict=True)


def new_path(value: str) -> Path:
    path = Path(value).expanduser().resolve(strict=False)
    if path.exists():
        raise ValueError(f"output already exists: {path}")
    if not path.parent.is_dir():
        raise ValueError(f"output parent is not a directory: {path.parent}")
    return path


def base_max_generate_length(prompt_samples: int, samples_per_patch: int) -> int:
    return (prompt_samples + samples_per_patch - 1) // samples_per_patch + TARGET_PATCH_BUDGET


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--checkout", required=True)
    result.add_argument("--model-dir", required=True)
    result.add_argument("--mode", required=True, choices=("base", "edit"))
    result.add_argument("--text")
    result.add_argument("--ref-audio", required=True)
    result.add_argument("--ref-text")
    result.add_argument("--instruction")
    result.add_argument("--source-text")
    result.add_argument("--target-text")
    result.add_argument("--latent-noise", required=True)
    result.add_argument("--dit-noise", required=True)
    result.add_argument("--trace", required=True)
    result.add_argument("--out", required=True)
    return result


def validate_mode(args: argparse.Namespace) -> None:
    if args.mode == "base":
        if not args.text or not args.ref_text:
            raise ValueError("base mode requires --text and --ref-text")
        if args.instruction or args.source_text or args.target_text:
            raise ValueError("base mode rejects edit-only options")
    else:
        if args.text or args.ref_text:
            raise ValueError("edit mode rejects --text and --ref-text")
        if not args.instruction or not args.source_text or not args.target_text:
            raise ValueError(
                "edit mode requires --instruction, --source-text, and --target-text"
            )


def write_pcm16(path: Path, audio, sample_rate: int) -> None:
    import numpy as np

    array = audio.detach().to(device="cpu", dtype=None).float().numpy()
    if array.ndim == 3 and array.shape[:2] == (1, 1):
        array = array[0, 0]
    elif array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1 or sample_rate != 48_000:
        raise ValueError(
            f"oracle result must be 48-kHz mono, got shape={array.shape} rate={sample_rate}"
        )
    pcm = (np.clip(array, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def trace_first_dit_forward(runtime) -> None:
    from dots_tts.oracle_trace import tensor

    runtime.model._get_dit_solver(solver_mode=runtime.model.core.mode)
    dit = runtime.model.core.velocity_field_predictor

    def first_two(name: str, transform=None):
        seen = 0

        def hook(_module, _inputs, output):
            nonlocal seen
            if seen < 2:
                seen += 1
                tensor(name, output if transform is None else transform(output))

        return hook

    def first_two_inputs(name: str):
        seen = 0

        def hook(_module, inputs):
            nonlocal seen
            if seen < 2:
                seen += 1
                tensor(name, inputs[0])

        return hook

    dit.time_embedder.mlp[0].register_forward_pre_hook(
        first_two_inputs("dots.dit.time_embedding")
    )
    dit.time_embedder.mlp[0].register_forward_hook(first_two("dots.dit.time_linear0"))
    dit.time_embedder.mlp[1].register_forward_hook(first_two("dots.dit.time_silu"))
    dit.time_embedder.mlp[2].register_forward_hook(first_two("dots.dit.time"))
    dit.fused_adaln[-1].register_forward_pre_hook(
        first_two_inputs("dots.dit.mods_input")
    )
    dit.input_layer.register_forward_hook(first_two("dots.dit.projected_input"))
    block = dit.blocks[0]
    block.norm1.register_forward_hook(first_two("dots.dit.block0.norm1"))
    block.attn.register_forward_pre_hook(first_two_inputs("dots.dit.block0.attn_in"))
    block.attn.qkv_proj.register_forward_hook(first_two("dots.dit.block0.qkv"))
    heads_to_rows = lambda value: value.permute(0, 2, 1, 3)
    block.attn.q_norm.register_forward_hook(
        first_two("dots.dit.block0.q_norm", heads_to_rows)
    )
    block.attn.k_norm.register_forward_hook(
        first_two("dots.dit.block0.k_norm", heads_to_rows)
    )
    block.attn.o_proj.register_forward_pre_hook(
        first_two_inputs("dots.dit.block0.attn_out")
    )
    block.attn.o_proj.register_forward_hook(first_two("dots.dit.block0.attn_proj"))
    block.norm2.register_forward_hook(first_two("dots.dit.block0.norm2"))
    block.ffn.register_forward_pre_hook(first_two_inputs("dots.dit.block0.ffn_in"))
    block.ffn.fc1.register_forward_hook(first_two("dots.dit.block0.fc1"))
    block.ffn.act.register_forward_hook(first_two("dots.dit.block0.gelu"))
    block.ffn.fc2.register_forward_hook(first_two("dots.dit.block0.fc2"))
    dit.output_layer.norm.register_forward_hook(first_two("dots.dit.final.norm"))
    dit.output_layer.linear.register_forward_pre_hook(
        first_two_inputs("dots.dit.final.input")
    )
    dit.output_layer.linear.register_forward_hook(first_two("dots.dit.final.output"))


def main() -> None:
    try:
        args = parser().parse_args()
        validate_mode(args)
        checkout = existing_path(args.checkout)
        model_dir = existing_path(args.model_dir)
        ref_audio = existing_path(args.ref_audio)
        latent_noise = existing_path(args.latent_noise)
        dit_noise = existing_path(args.dit_noise)
        trace = new_path(args.trace)
        output = new_path(args.out)
        if trace.is_relative_to(checkout) or output.is_relative_to(checkout):
            raise ValueError(f"output must not be inside oracle checkout: {checkout}")
    except (OSError, ValueError) as error:
        parser().error(str(error))
    sha = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if sha != PINNED_SHA:
        raise ValueError(f"oracle checkout must be {PINNED_SHA}, got {sha}")
    for noise in (latent_noise, dit_noise):
        if noise.stat().st_size == 0 or noise.stat().st_size % 4:
            raise ValueError(f"noise must contain raw float32 values: {noise}")

    sys.path.insert(0, str(checkout / "src"))
    os.environ["DOTS_TTS_ORACLE_TRACE"] = str(trace)
    os.environ["DOTS_TTS_ORACLE_LATENT_NOISE"] = str(latent_noise)
    os.environ["DOTS_TTS_ORACLE_DIT_NOISE"] = str(dit_noise)

    if args.mode == "base":
        from dots_tts.runtime import DotsTtsRuntime

        runtime = DotsTtsRuntime.from_pretrained(
            str(model_dir),
            precision="float32",
            optimize=False,
            max_generate_length=500,
            warmup_on_optimize=False,
        )
        trace_first_dit_forward(runtime)
        prompt = runtime._load_prompt_audio(str(ref_audio))
        samples_per_patch = int(runtime.model.config.patch_size * runtime.model.hop_size)
        runtime.max_generate_length = base_max_generate_length(
            int(prompt.shape[-1]), samples_per_patch
        )
        result = runtime.generate(
            text=args.text,
            prompt_audio_path=str(ref_audio),
            prompt_text=args.ref_text,
            template_name="tts",
            speaker_scale=1.5,
            ode_method="euler",
            num_steps=10,
            guidance_scale=1.2,
        )
    else:
        from dots_tts.edit_runtime import DotsTtsEditRuntime

        runtime = DotsTtsEditRuntime.from_pretrained(
            str(model_dir),
            precision="float32",
            optimize=False,
            max_generate_length=500,
            warmup_on_optimize=False,
        )
        trace_first_dit_forward(runtime)
        source = runtime._load_edit_source_audio(str(ref_audio))
        samples_per_patch = int(runtime.model.config.patch_size * runtime.model.hop_size)
        source_patches = (int(source.shape[-1]) + samples_per_patch - 1) // samples_per_patch
        runtime.max_generate_length = source_patches + TARGET_PATCH_BUDGET
        result = runtime.generate_edit(
            source_audio_path=str(ref_audio),
            instruction=args.instruction,
            source_text=args.source_text,
            target_text=args.target_text,
            use_xvector=True,
            speaker_scale=1.5,
            ode_method="euler",
            num_steps=10,
            guidance_scale=1.2,
        )

    from dots_tts.oracle_trace import assert_noise_consumed

    assert_noise_consumed()
    write_pcm16(output, result["audio"], int(result["sample_rate"]))


if __name__ == "__main__":
    main()
