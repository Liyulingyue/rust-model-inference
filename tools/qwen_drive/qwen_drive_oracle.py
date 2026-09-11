#!/usr/bin/env python3
"""Small fixed-version PyTorch oracles for Qwen-Drive parity fixtures."""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def bits(values: torch.Tensor) -> list[int]:
    return [struct.unpack("<I", struct.pack("<f", float(value)))[0] for value in values.flatten()]


def operator_fixture() -> dict:
    if torch.__version__.split("+", 1)[0] != "2.8.0":
        raise RuntimeError(f"expected torch 2.8.0, got {torch.__version__}")

    times = torch.tensor([0.0, 0.1, 0.9], dtype=torch.float32)
    half = 64
    decay = math.log(10000) / (half - 1)
    freqs = torch.exp(torch.arange(half, dtype=torch.float32) * -decay)
    angles = 1000.0 * times.unsqueeze(1) * freqs.unsqueeze(0)
    time = torch.cat([angles.sin(), angles.cos()], dim=-1)

    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.25, -0.5, 1.0]], dtype=torch.bfloat16
    )
    fourier_freqs = torch.logspace(
        0, math.log10(16.0), steps=16, dtype=torch.bfloat16
    )
    fourier_angles = points.float().unsqueeze(-1) * fourier_freqs * (2 * math.pi)
    fourier = torch.cat([fourier_angles.sin(), fourier_angles.cos()], dim=-1)
    fourier = fourier.flatten(-2).to(torch.bfloat16)

    q = torch.arange(-16, 16, dtype=torch.float32).mul_(0.125).to(torch.bfloat16)
    q = q.reshape(1, 2, 2, 8)
    positions = torch.tensor(
        [[[258, 259]], [[259, 260]], [[260, 261]]], dtype=torch.int64
    )
    exponents = torch.arange(0, 6, 2, dtype=torch.float32)
    inv_freq = (1.0 / (10_000_000.0 ** (exponents / 6))).to(torch.bfloat16)
    rope_angles = positions.to(torch.bfloat16).unsqueeze(-1) * inv_freq
    merged = rope_angles[0].clone()
    merged[..., 1:3:3] = rope_angles[1][..., 1:3:3]
    merged[..., 2:3:3] = rope_angles[2][..., 2:3:3]
    emb = torch.cat([merged, merged], dim=-1).unsqueeze(2)
    cos, sin = emb.cos(), emb.sin()
    rotated, passthrough = q[..., :6], q[..., 6:]
    first, second = torch.chunk(rotated, 2, dim=-1)
    rotate_half = torch.cat([-second, first], dim=-1)
    rope = torch.cat([rotated * cos + rotate_half * sin, passthrough], dim=-1)

    history = torch.tensor(
        [[index * 0.25, index * -0.125, -3.5 + index * 0.5] for index in range(16)],
        dtype=torch.float32,
    )
    relative = history - history[0:1]
    heading = torch.remainder(relative[:, 2:3] + math.pi, 2 * math.pi) - math.pi
    normalized_history = torch.cat([relative[:, :2], heading], dim=-1)[1:]
    heading = (
        torch.remainder(normalized_history[:, 2:3] + math.pi, 2 * math.pi)
        - math.pi
    )
    normalized_history = torch.cat([normalized_history[:, :2], heading], dim=-1)
    normalized_history = normalized_history / torch.tensor(
        [165.0, 25.0, 1.5703125], dtype=torch.float32
    )

    x = torch.tensor([0.25, -0.5, 1.0, -1.5, 2.0, -2.5], dtype=torch.float32)
    endpoint = torch.tensor([1.0, 0.5, -1.0, 2.0, -2.0, 3.0], dtype=torch.float32)
    euler = x + (endpoint - x) / max(1.0 - 0.8, 0.1) * 0.1

    normal = {}
    for seed in (42, 43):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        normal[str(seed)] = bits(torch.randn(150, generator=generator, dtype=torch.float32))

    return {
        "time_inputs": bits(times),
        "time": bits(time),
        "point_inputs": bits(points.float()),
        "fourier": bits(fourier.float()),
        "rope_input": bits(q.float()),
        "rope": bits(rope.float()),
        "history_input": bits(history),
        "history": bits(normalized_history),
        "euler_input": bits(x),
        "euler_endpoint": bits(endpoint),
        "euler": bits(euler),
        "normal": normal,
    }


def _bf16(values: torch.Tensor) -> torch.Tensor:
    return values.to(torch.bfloat16).float()


def _multi_scale_deformable_attn_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    batch, _, heads, channels = value.shape
    _, queries, _, levels, points, _ = sampling_locations.shape
    value_list = value.split(
        [int(height * width) for height, width in value_spatial_shapes], dim=1
    )
    sampling_grids = 2 * sampling_locations - 1
    sampled = []
    for level, (height, width) in enumerate(value_spatial_shapes):
        level_value = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch * heads, channels, int(height), int(width))
        )
        level_grid = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled.append(
            F.grid_sample(
                level_value.float(),
                level_grid.float(),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    weights = attention_weights.float().transpose(1, 2).reshape(
        batch * heads, 1, queries, levels * points
    )
    output = (torch.stack(sampled, dim=-2).flatten(-2) * weights).sum(-1)
    return output.view(batch, heads * channels, queries).transpose(1, 2).contiguous()


def perception_operator_fixture() -> dict:
    if torch.__version__.split("+", 1)[0] != "2.8.0":
        raise RuntimeError(f"expected torch 2.8.0, got {torch.__version__}")

    resize_input = _bf16(
        torch.arange(-6, 6, dtype=torch.float32).reshape(1, 2, 2, 3) * 0.125
    )
    resize = _bf16(
        F.interpolate(
            resize_input,
            size=(3, 5),
            mode="bilinear",
            align_corners=False,
        )
    )

    grid_input = _bf16(
        torch.tensor(
            [
                -1.0,
                -0.75,
                -0.5,
                -0.25,
                0.0,
                0.25,
                0.5,
                0.75,
                1.0,
                1.25,
                1.5,
                1.75,
            ],
            dtype=torch.float32,
        ).reshape(1, 2, 2, 3)
    )
    grid = _bf16(
        torch.tensor(
            [
                [-1.25, -1.0],
                [-0.25, -0.5],
                [0.5, 0.0],
                [-1.0, 1.0],
                [0.25, 0.75],
                [1.2, 1.1],
            ],
            dtype=torch.float32,
        ).reshape(1, 2, 3, 2)
    )
    grid_output = _bf16(
        F.grid_sample(
            grid_input.float(),
            grid.float(),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
    )

    voxel_shape = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
    batch, sweeps, cameras, x_size, y_size, z_size, depth, height, width, channels = voxel_shape
    img_feats = _bf16(
        (torch.arange(batch * sweeps * cameras * channels * height * width, dtype=torch.float32) - 7)
        * 0.125
    )
    img_depth = _bf16(
        (torch.arange(batch * sweeps * cameras * depth * height * width, dtype=torch.float32) + 1)
        * 0.0625
    )
    point_indices = [0, 1, 3, 5, 8, 10, 15]
    coords = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
    ]
    voxel_output = torch.zeros(
        batch, cameras, x_size, y_size, z_size, channels, dtype=torch.float32
    )
    for coord, point_index in zip(coords, point_indices):
        point = point_index
        w = point % width
        point //= width
        h = point % height
        point //= height
        d = point % depth
        point //= depth
        camera = point % cameras
        point //= cameras
        sweep = point % sweeps
        image = (coord[0] * sweeps + sweep) * cameras + camera
        for channel in range(channels):
            voxel_output[coord[0], camera, coord[1], coord[2], coord[3], channel] += (
                img_feats.reshape(batch * sweeps * cameras, channels, height, width)[
                    image, channel, h, w
                ]
                * img_depth.reshape(batch * sweeps * cameras, depth, height, width)[
                    image, d, h, w
                ]
            )
    voxel_output = _bf16(voxel_output)

    deform_shape = [1, 2, 2, 2, 2]
    batch, queries, heads, channels, points = deform_shape
    spatial_shapes = [[2, 2], [1, 2]]
    level_start_index = [0, 4]
    spatial_size = sum(height * width for height, width in spatial_shapes)
    value = _bf16(
        (torch.arange(batch * spatial_size * heads * channels, dtype=torch.float32) - 11)
        * 0.09375
    ).reshape(batch, spatial_size, heads, channels)
    sampling_locations = _bf16(
        torch.tensor(
            [
                0.0, 0.0, 0.25, 0.75, 0.5, 0.5, 1.0, 1.0,
                0.125, 0.875, 0.75, 0.25, -0.1, 0.5, 1.1, 0.5,
                0.33, 0.66, 0.8, 0.2, 0.1, 0.9, 0.6, 0.4,
                0.45, 0.55, 0.9, 0.1, 0.2, 0.3, 0.7, 0.8,
            ],
            dtype=torch.float32,
        ).reshape(batch, queries, heads, len(spatial_shapes), points, 2)
    )
    attention_weights = _bf16(
        torch.tensor(
            [
                0.1, 0.2, 0.3, 0.4,
                0.4, 0.3, 0.2, 0.1,
                0.15, 0.35, 0.25, 0.25,
                0.05, 0.45, 0.4, 0.1,
            ],
            dtype=torch.float32,
        ).reshape(batch, queries, heads, len(spatial_shapes), points)
    )
    deform_output = _bf16(
        _multi_scale_deformable_attn_pytorch(
            value,
            torch.tensor(spatial_shapes, dtype=torch.int64),
            sampling_locations,
            attention_weights,
        )
    ).reshape(batch, queries, heads, channels)

    return {
        "resize": {
            "input": {"shape": list(resize_input.shape), "values": bits(resize_input)},
            "output_hw": [3, 5],
            "output": bits(resize),
        },
        "grid": {
            "input": {"shape": list(grid_input.shape), "values": bits(grid_input)},
            "grid_shape": [1, 2, 3],
            "grid": bits(grid),
            "output": bits(grid_output),
        },
        "voxel": {
            "img_feats": bits(img_feats),
            "img_depth": bits(img_depth),
            "coords": coords,
            "point_indices": point_indices,
            "shape": voxel_shape,
            "output": bits(voxel_output),
        },
        "deform": {
            "value": bits(value),
            "spatial_shapes": spatial_shapes,
            "level_start_index": level_start_index,
            "sampling_locations": bits(sampling_locations),
            "attention_weights": bits(attention_weights),
            "shape": deform_shape,
            "output": bits(deform_output),
        },
    }


class TraceSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.counts: dict[str, int] = defaultdict(int)
        selected = os.environ.get("RMI_PARITY_FILTER")
        self.selected = set(selected.split(",")) if selected else None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)

    def dump(
        self,
        name: str,
        value: torch.Tensor,
        *,
        layer: int | None = None,
        step: int | None = None,
    ) -> None:
        if self.selected is not None and name not in self.selected:
            return
        value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        occurrence = self.counts[name]
        self.counts[name] += 1
        suffix = f".{occurrence}" if occurrence else ""
        binary = Path(f"{self.path}.{name}{suffix}.f32")
        binary.write_bytes(value.numpy().astype("<f4", copy=False).tobytes())
        record = {
            "name": name,
            "layer": layer,
            "step": step,
            "shape": list(value.shape),
            "len": value.numel(),
            "finite": bool(torch.isfinite(value).all()),
            "occurrence": occurrence,
            "binary_path": str(binary),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def _require_commit(source: Path, expected: str) -> None:
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise RuntimeError(f"expected official commit {expected}, got {actual}")


def _planner_inputs(torch_module, dtype):
    history = torch_module.tensor(
        [[index * 0.25, index * -0.125, -3.5 + index * 0.5] for index in range(16)],
        dtype=torch_module.float32,
    ).unsqueeze(0)
    velocity = torch_module.tensor(
        [[index * 0.03125, index * -0.015625] for index in range(16)],
        dtype=torch_module.float32,
    ).unsqueeze(0)
    acceleration = torch_module.tensor(
        [[(index % 5 - 2) * 0.0625, (index % 3 - 1) * 0.03125] for index in range(16)],
        dtype=torch_module.float32,
    ).unsqueeze(0)
    nav = torch_module.tensor([1], dtype=torch_module.long)
    ego = torch_module.tensor(
        [[0.25, -0.5, 1.0, -1.5, 2.0, -2.5, 0.125, -0.0625]],
        dtype=torch_module.float32,
    )
    anchor = torch_module.tensor([[257], [258], [259]], dtype=torch_module.long)
    scene_cache = []
    for index in range(8):
        base = (
            (torch_module.arange(3 * 4 * 256, dtype=torch_module.float32) % 257 - 128)
            * 0.0078125
            + index * 0.03125
        )
        key = base.to(dtype).reshape(1, 3, 4, 256)
        value = (-base * 0.5 + index * 0.015625).to(dtype).reshape(1, 3, 4, 256)
        scene_cache.append((key, value))
    return history, velocity, acceleration, nav, ego, anchor, scene_cache


def planner_trace(args) -> None:
    if torch.__version__.split("+", 1)[0] != "2.8.0":
        raise RuntimeError(f"expected torch 2.8.0, got {torch.__version__}")
    _require_commit(args.source, args.expected_commit)
    sys.path.insert(0, str(args.source / "src"))
    from safetensors.torch import load_file
    from qwen_drive.configuration_qwen_drive import PlanningExpertConfig
    from qwen_drive.planning_expert import PlanningExpert
    from qwen_drive.trajectory import denormalize_trajectory, normalize_history

    torch.set_num_threads(args.threads)
    torch.backends.mkldnn.enabled = False
    planner_root = args.model_root / f"planner-{args.planner}"
    config = PlanningExpertConfig.from_pretrained(planner_root)
    expert = PlanningExpert(config, 50, 16, 3).to(dtype=torch.bfloat16)
    state = {
        name.removeprefix("planning_expert."): value
        for name, value in load_file(planner_root / "model.safetensors").items()
    }
    expert.load_state_dict(state, strict=True)
    expert.eval()

    sink = TraceSink(args.trace)

    def hook(name: str, layer: int | None = None):
        return lambda _module, _inputs, output: sink.dump(name, output, layer=layer)

    def pre_hook(name: str, layer: int):
        return lambda _module, inputs: sink.dump(name, inputs[0], layer=layer)

    def traced_attend(layer, index: int):
        attend = layer._attend

        def call(query, key, value):
            sink.dump("qwen_drive.planner.rotary_query", query, layer=index)
            sink.dump("qwen_drive.planner.joint_key", key, layer=index)
            sink.dump("qwen_drive.planner.joint_value", value, layer=index)
            heads_per_group = query.shape[2] // key.shape[2]
            scaled_query = query.float().transpose(1, 2) * (query.shape[-1] ** -0.25)
            scaled_key = (
                key.float()
                .transpose(1, 2)
                .repeat_interleave(heads_per_group, dim=1)
                * (query.shape[-1] ** -0.25)
            )
            scores = torch.matmul(scaled_query, scaled_key.transpose(-2, -1))
            probabilities = torch.softmax(scores, dim=-1)
            sink.dump(
                "qwen_drive.planner.attention_scores",
                scores.transpose(1, 2),
                layer=index,
            )
            sink.dump(
                "qwen_drive.planner.attention_probabilities",
                probabilities.transpose(1, 2),
                layer=index,
            )
            output = attend(query, key, value)
            sink.dump("qwen_drive.planner.attention_raw", output, layer=index)
            return output

        return call

    for mlp in [
        expert.history_encoder,
        expert.history_velocity_encoder,
        expert.history_acceleration_encoder,
        expert.fourier_encoder.net,
        expert.time_mlp,
        expert.query_fusion,
        expert.nav_mlp,
        expert.ego_mlp,
    ]:
        mlp[0].register_forward_hook(hook("qwen_drive.planner.mlp_linear"))
        mlp[1].register_forward_hook(hook("qwen_drive.planner.mlp_silu"))

    expert.history_encoder.register_forward_hook(hook("qwen_drive.planner.history"))
    expert.history_velocity_encoder.register_forward_hook(hook("qwen_drive.planner.velocity"))
    expert.history_acceleration_encoder.register_forward_hook(
        hook("qwen_drive.planner.acceleration")
    )
    expert.trajectory_proj.register_forward_hook(
        hook("qwen_drive.planner.trajectory_embedding")
    )
    expert.fourier_encoder.register_forward_hook(
        hook("qwen_drive.planner.fourier_embedding")
    )
    expert.time_mlp.register_forward_hook(hook("qwen_drive.planner.time_condition"))
    expert.query_fusion.register_forward_hook(hook("qwen_drive.planner.query"))
    for index, layer in enumerate(expert.layers):
        layer.adaln_modulation.register_forward_hook(
            hook("qwen_drive.planner.modulation", index)
        )
        layer.input_layernorm.register_forward_hook(
            hook("qwen_drive.planner.attention_norm", index)
        )
        layer.q_norm.register_forward_hook(hook("qwen_drive.planner.query_norm", index))
        layer.k_norm.register_forward_hook(hook("qwen_drive.planner.key_norm", index))
        layer._attend = traced_attend(layer, index)
        layer.qkv_proj.register_forward_pre_hook(
            pre_hook("qwen_drive.planner.qkv_input", index)
        )
        layer.qkv_proj.register_forward_hook(hook("qwen_drive.planner.qkv", index))
        layer.o_proj.register_forward_pre_hook(
            pre_hook("qwen_drive.planner.attention", index)
        )
        layer.register_forward_hook(hook("qwen_drive.planner.layer_output", index))
    expert.final_layernorm.register_forward_hook(hook("qwen_drive.planner.final_norm"))
    expert.out_proj.register_forward_hook(hook("qwen_drive.planner.endpoint"))

    history, velocity, acceleration, nav, ego, anchor, scene_cache = _planner_inputs(
        torch, expert.dtype
    )
    generator = torch.Generator(device="cpu").manual_seed(42)
    noise = torch.randn((1, 50, 3), generator=generator, dtype=torch.float32)
    sink.dump("qwen_drive.planner.noise", noise)
    scale = torch.tensor([165.0, 25.0, 1.5703125], dtype=torch.float32)
    normalized_history = normalize_history(history, scale)
    with torch.no_grad():
        history_queries = expert.encode_history(
            normalized_history, nav, velocity, acceleration
        )
        waypoints = noise.float()
        step = 1.0 / args.steps
        for index in range(args.steps):
            flow_time = torch.full((1,), index * step, dtype=torch.float32)
            endpoint = expert.predict_endpoint(
                waypoints,
                flow_time,
                history_queries,
                scene_cache,
                anchor,
                nav,
                ego,
            )
            remaining = max(1.0 - index * step, 0.1)
            waypoints = waypoints + (endpoint - waypoints) / remaining * step
            sink.dump("qwen_drive.planner.euler", waypoints, step=index)
        sink.dump("qwen_drive.planner.trajectory", denormalize_trajectory(waypoints, scale))


def planning_prompt_fixture(args) -> None:
    _require_commit(args.source, args.expected_commit)
    sys.path.insert(0, str(args.source / "src"))
    from transformers import AutoTokenizer
    from qwen_drive.benchmarks import read_scene_file
    from qwen_drive.configuration_qwen_drive import QwenDriveConfig
    from qwen_drive.scene import QwenDriveProcessor

    tokenizer = AutoTokenizer.from_pretrained(args.model_root)
    processor = QwenDriveProcessor(
        tokenizer, QwenDriveConfig.from_pretrained(args.model_root)
    )
    sample = next(
        read_scene_file(
            args.source / "data/demo/planning_scenes.jsonl",
            image_root=args.source / "data/demo",
            limit=1,
        )
    )
    _, _, token_counts = processor.encode_images(sample.scene)
    fixture = {
        "token_counts": token_counts,
        "direct": processor.build_input_ids(sample.scene, token_counts, False),
        "reasoning": processor.build_input_ids(sample.scene, token_counts, True),
    }
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["planner-operators", "perception-ops", "planning-prompt", "planner"],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--planner", choices=["sft", "rl"], default="sft")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if args.command == "planner-operators":
        if args.output is None:
            parser.error("planner-operators requires --output")
        fixture = operator_fixture()
        args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    elif args.command == "perception-ops":
        if args.output is None or args.source is None or args.expected_commit is None:
            parser.error("perception-ops requires --output, --source, and --expected-commit")
        _require_commit(args.source, args.expected_commit)
        fixture = perception_operator_fixture()
        args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    elif args.command == "planning-prompt":
        if (
            args.output is None
            or args.source is None
            or args.expected_commit is None
            or args.model_root is None
        ):
            parser.error(
                "planning-prompt requires --output, --source, --expected-commit, and --model-root"
            )
        planning_prompt_fixture(args)
    else:
        if args.source is None or args.expected_commit is None or args.model_root is None:
            parser.error("planner requires --source, --expected-commit, and --model-root")
        if args.trace is None:
            parser.error("planner requires --trace")
        if args.steps <= 0 or args.threads <= 0:
            parser.error("--steps and --threads must be greater than zero")
        planner_trace(args)


if __name__ == "__main__":
    main()
