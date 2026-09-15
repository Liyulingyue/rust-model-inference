#!/usr/bin/env python3
"""Small fixed-version PyTorch oracles for Qwen-Drive parity fixtures."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import struct
import subprocess
import sys
import types
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


def perception_view_fixture(source: Path) -> dict:
    if torch.__version__.split("+", 1)[0] != "2.8.0":
        raise RuntimeError(f"expected torch 2.8.0, got {torch.__version__}")

    norm_input = _bf16(
        (torch.arange(16, dtype=torch.float32) - 7) * 0.1875
    ).reshape(1, 4, 2, 2)
    norm_weight = _bf16(torch.tensor([0.75, 1.0, 1.25, -0.5]))
    norm_bias = _bf16(torch.tensor([0.125, -0.25, 0.5, -0.75]))
    norm_mean = norm_input.mean(1, keepdim=True)
    norm_var = (norm_input - norm_mean).pow(2).mean(1, keepdim=True)
    norm_output = _bf16(
        norm_weight[:, None, None]
        * ((norm_input - norm_mean) / torch.sqrt(norm_var + 1e-6))
        + norm_bias[:, None, None]
    )

    transpose_input = _bf16(
        (torch.arange(8, dtype=torch.float32) - 3) * 0.25
    ).reshape(1, 2, 2, 2)
    transpose_weight = _bf16(
        (torch.arange(24, dtype=torch.float32) - 11) * 0.0625
    ).reshape(2, 3, 2, 2)
    transpose_bias = _bf16(torch.tensor([0.125, -0.25, 0.375]))
    transpose_output = _bf16(
        F.conv_transpose2d(
            transpose_input,
            transpose_weight,
            transpose_bias,
            stride=2,
        )
    )

    aligned_input = _bf16(
        (torch.arange(12, dtype=torch.float32) - 5) * 0.125
    ).reshape(1, 2, 2, 3)
    aligned_output = _bf16(
        F.interpolate(
            aligned_input,
            size=(3, 5),
            mode="bilinear",
            align_corners=True,
        )
    )

    depth_input = _bf16(
        torch.tensor(
            [
                -1.0, 0.5, 1.25, -0.75,
                0.25, -0.5, 0.75, 1.5,
                1.0, 0.0, -1.25, 0.25,
            ],
            dtype=torch.float32,
        )
    ).reshape(1, 3, 2, 2)
    depth_output = _bf16(torch.softmax(depth_input, dim=1))

    frustum_range = [0.0, 0.0, 1.0, 2.0, 2.0, 3.0]
    frustum_size = [1.0, 1.0, 1.0]
    pc_range = [-1.0, -1.0, 0.0, 3.0, 3.0, 3.0]
    voxel_size = [1.0, 1.0, 1.0]
    voxel_shape = [4, 4, 3]
    identity = torch.eye(4, dtype=torch.float32)
    camera_two = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    lidar2img = torch.stack([identity, camera_two]).reshape(1, 2, 1, 4, 4)
    lidar2ego = identity.reshape(1, 4, 4)
    axes = [
        torch.arange(frustum_range[i], frustum_range[i + 3], frustum_size[i])
        for i in range(3)
    ]
    frustum = torch.stack(torch.meshgrid(axes, indexing="ij"), dim=-1)
    width, height, depth = frustum.shape[:-1]
    points = torch.cat([frustum, torch.ones_like(frustum[..., :1])], -1)
    points = points.flatten(0, 2).unsqueeze(0).unsqueeze(0)
    points[..., :2] *= points[..., 2:3]
    points = torch.matmul(
        torch.inverse(lidar2img.flatten(1, 2)).unsqueeze(2),
        points.unsqueeze(-1),
    ).squeeze(-1)
    points = torch.matmul(
        lidar2ego[:, None, None], points.unsqueeze(-1)
    ).squeeze(-1)
    voxel_coords = (
        (points[..., :3] - torch.tensor(pc_range[:3]))
        / torch.tensor(voxel_size)
    ).int()
    batch_index = torch.zeros_like(voxel_coords[..., :1])
    voxel_coords = torch.cat([batch_index, voxel_coords], dim=-1)
    voxel_coords = (
        voxel_coords.view(1, 2, 1, width, height, depth, 4)
        .permute(0, 1, 2, 5, 4, 3, 6)
        .contiguous()
    )
    mask = (
        (voxel_coords[..., 1] >= 0)
        & (voxel_coords[..., 1] < voxel_shape[0])
        & (voxel_coords[..., 2] >= 0)
        & (voxel_coords[..., 2] < voxel_shape[1])
        & (voxel_coords[..., 3] >= 0)
        & (voxel_coords[..., 3] < voxel_shape[2])
    )
    flat_mask = mask.reshape(-1)
    geometry_coords = voxel_coords.reshape(-1, 4)[flat_mask].tolist()
    geometry_indices = torch.arange(flat_mask.numel(), dtype=torch.int64)[flat_mask].tolist()

    bev_shape = [2, 2, 2, 2, 2]
    bev_values = _bf16(
        (torch.arange(math.prod(bev_shape), dtype=torch.float32) - 15) * 0.0625
    ).reshape(bev_shape)
    bev_weight = _bf16(
        (torch.arange(12, dtype=torch.float32) - 5) * 0.09375
    ).reshape(3, 4, 1, 1)
    bev_bias = _bf16(torch.tensor([0.125, -0.25, 0.5]))
    bev_maps = _bf16(
        F.conv2d(bev_values.reshape(2, 4, 2, 2), bev_weight, bev_bias)
    )
    bev_output = _bf16(bev_maps.mean(dim=0, keepdim=True))
    bev_output = bev_output.squeeze(0).permute(1, 2, 0).reshape(-1, 3)

    module_path = source / "src/qwen_drive_perception/fpn.py"
    spec = importlib.util.spec_from_file_location("qwen_drive_perception_fpn", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    SimpleFPN = module.SimpleFPN

    fpn = SimpleFPN(dim=4, out_channels=2).to(torch.bfloat16).eval()
    with torch.no_grad():
        for index, parameter in enumerate(fpn.parameters()):
            values = (
                torch.arange(parameter.numel(), dtype=torch.float32)
                - parameter.numel() // 2
            ) * (0.0078125 / (index + 1))
            parameter.copy_(values.reshape(parameter.shape).to(torch.bfloat16))
    fpn_input = _bf16(
        (torch.arange(24, dtype=torch.float32) - 11) * 0.0625
    ).reshape(1, 4, 2, 3).to(torch.bfloat16)
    with torch.no_grad():
        fpn_outputs = fpn(fpn_input)
    fpn_weights = [
        {"name": name, "shape": list(value.shape), "values": bits(value.float())}
        for name, value in fpn.state_dict().items()
    ]

    package = types.ModuleType("qwen_drive_perception")
    package.__path__ = [str(source / "src/qwen_drive_perception")]
    sys.modules[package.__name__] = package
    module_path = source / "src/qwen_drive_perception/view_transform.py"
    spec = importlib.util.spec_from_file_location(
        "qwen_drive_perception.view_transform", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    DepthNet = module.DepthNet
    Uni3DVoxelPoolDepth = module.Uni3DVoxelPoolDepth

    depth_net = DepthNet(
        in_channels=4,
        mid_channels=32,
        depth_channels=3,
        aspp_mid_channels=32,
    ).to(torch.bfloat16).eval()
    with torch.no_grad():
        for index, parameter in enumerate(depth_net.parameters()):
            values = (
                torch.arange(parameter.numel(), dtype=torch.float32)
                - parameter.numel() // 2
            ) * (0.0009765625 / (index + 1))
            parameter.copy_(values.reshape(parameter.shape).to(torch.bfloat16))
    depth_net_input = _bf16(
        (torch.arange(48, dtype=torch.float32) - 23) * 0.03125
    ).reshape(2, 4, 2, 3).to(torch.bfloat16)
    with torch.no_grad():
        depth_logits = depth_net(depth_net_input)
        depth_probabilities = depth_logits.softmax(1)
    depth_weights = [
        {"name": name, "shape": list(value.shape), "values": bits(value.float())}
        for name, value in depth_net.state_dict().items()
    ]

    view_features = _bf16(
        (torch.arange(32, dtype=torch.float32) - 15) * 0.03125
    ).reshape(2, 4, 2, 2).to(torch.bfloat16)
    view_depth_logits = _bf16(
        (torch.arange(16, dtype=torch.float32) - 7) * 0.0625
    ).reshape(2, 2, 2, 2).to(torch.bfloat16)
    view_depth = view_depth_logits.softmax(1)
    pooled = torch.zeros(1, 2, 4, 4, 3, 4, dtype=torch.float32)
    for coord, point_index in zip(geometry_coords, geometry_indices):
        point = point_index
        w = point % 2
        point //= 2
        h = point % 2
        point //= 2
        d = point % 2
        point //= 2
        camera = point % 2
        for channel in range(4):
            pooled[coord[0], camera, coord[1], coord[2], coord[3], channel] += (
                view_features[camera, channel, h, w].float()
                * view_depth[camera, d, h, w].float()
            )
    pooled = _bf16(pooled).to(torch.bfloat16)
    voxel_space = pooled.permute(0, 1, 5, 4, 3, 2).contiguous()
    view_transform = Uni3DVoxelPoolDepth(
        pc_range=pc_range,
        voxel_size=voxel_size,
        voxel_shape=voxel_shape,
        frustum_range=frustum_range,
        frustum_size=frustum_size,
        embed_dim=4,
    ).to(torch.bfloat16).eval()
    with torch.no_grad():
        for index, parameter in enumerate(view_transform.parameters()):
            values = (
                torch.arange(parameter.numel(), dtype=torch.float32)
                - parameter.numel() // 2
            ) * (0.00390625 / (index + 1))
            parameter.copy_(values.reshape(parameter.shape).to(torch.bfloat16))
        for index, layer in enumerate(view_transform.conv_layer):
            layer[1].running_mean.copy_(
                torch.arange(4, dtype=torch.float32).mul(0.0078125 * (index + 1)).to(torch.bfloat16)
            )
            layer[1].running_var.copy_(
                torch.arange(4, dtype=torch.float32).mul(0.015625).add(1.0).to(torch.bfloat16)
            )
        view_voxel = view_transform.feat_encoding(voxel_space)
    uvtr = torch.nn.Conv2d(12, 3, kernel_size=1).to(torch.bfloat16).eval()
    with torch.no_grad():
        uvtr.weight.copy_(
            (torch.arange(36, dtype=torch.float32) - 17)
            .mul(0.005859375)
            .reshape_as(uvtr.weight)
            .to(torch.bfloat16)
        )
        uvtr.bias.copy_(torch.tensor([0.125, -0.25, 0.375], dtype=torch.bfloat16))
        view_bev = uvtr(view_voxel.flatten(1, 2))
        view_bev = view_bev.mean(0, keepdim=True).squeeze(0).permute(1, 2, 0).reshape(-1, 3)
    view_weights = [
        {
            "name": f"view_trans.{name}",
            "shape": list(value.shape),
            "values": bits(value.float()),
        }
        for name, value in view_transform.state_dict().items()
    ] + [
        {
            "name": f"uvtr_query_proj.{name}",
            "shape": list(value.shape),
            "values": bits(value.float()),
        }
        for name, value in uvtr.state_dict().items()
    ]

    return {
        "fpn": {
            "input": {"shape": list(fpn_input.shape), "values": bits(fpn_input.float())},
            "weights": fpn_weights,
            "outputs": [
                {"shape": list(output.shape), "values": bits(output.float())}
                for output in fpn_outputs
            ],
        },
        "depth_net": {
            "input": {
                "shape": list(depth_net_input.shape),
                "values": bits(depth_net_input.float()),
            },
            "weights": depth_weights,
            "logits": {"shape": list(depth_logits.shape), "values": bits(depth_logits.float())},
            "probabilities": bits(depth_probabilities.float()),
        },
        "view_transform": {
            "features": {"shape": list(view_features.shape), "values": bits(view_features.float())},
            "depth": {"shape": list(view_depth.shape), "values": bits(view_depth.float())},
            "weights": view_weights,
            "voxel": {"shape": list(view_voxel.shape), "values": bits(view_voxel.float())},
            "bev": {"shape": list(view_bev.shape), "values": bits(view_bev.float())},
        },
        "layer_norm": {
            "input": {"shape": list(norm_input.shape), "values": bits(norm_input)},
            "weight": bits(norm_weight),
            "bias": bits(norm_bias),
            "epsilon": bits(torch.tensor([1e-6], dtype=torch.float32))[0],
            "output": bits(norm_output),
        },
        "transpose": {
            "input": {
                "shape": list(transpose_input.shape),
                "values": bits(transpose_input),
            },
            "weight_shape": list(transpose_weight.shape),
            "weight": bits(transpose_weight),
            "bias": bits(transpose_bias),
            "stride": [2, 2],
            "padding": [0, 0],
            "output_shape": list(transpose_output.shape),
            "output": bits(transpose_output),
        },
        "resize_aligned": {
            "input": {"shape": list(aligned_input.shape), "values": bits(aligned_input)},
            "output_hw": [3, 5],
            "output": bits(aligned_output),
        },
        "depth": {"shape": list(depth_input.shape), "values": bits(depth_input)},
        "depth_output": bits(depth_output),
        "geometry": {
            "frustum_range": bits(torch.tensor(frustum_range)),
            "frustum_size": bits(torch.tensor(frustum_size)),
            "pc_range": bits(torch.tensor(pc_range)),
            "voxel_size": bits(torch.tensor(voxel_size)),
            "voxel_shape": voxel_shape,
            "lidar2img": [bits(identity), bits(camera_two)],
            "lidar2ego": [bits(identity)],
            "coords": geometry_coords,
            "point_indices": geometry_indices,
        },
        "bev": {
            "values": bits(bev_values),
            "shape": bev_shape,
            "weight": bits(bev_weight),
            "weight_shape": [3, 4],
            "bias": bits(bev_bias),
            "output": bits(bev_output),
        },
    }


def perception_heads_fixture(source: Path) -> dict:
    if torch.__version__.split("+", 1)[0] != "2.8.0":
        raise RuntimeError(f"expected torch 2.8.0, got {torch.__version__}")

    package = types.ModuleType("qwen_drive_perception")
    package.__path__ = [str(source / "src/qwen_drive_perception")]
    sys.modules[package.__name__] = package
    from qwen_drive_perception.heads import BevFeatureSlicer, NMSFreeCoder
    from qwen_drive_perception.layers import inverse_sigmoid
    from qwen_drive_perception.perception_transformer import PerceptionTransformer

    references = _bf16(torch.tensor([0.0, 0.125, 0.5, 0.875, 1.0]))
    inverse = _bf16(inverse_sigmoid(references))

    refine_references = torch.tensor(
        [[0.125, 0.5, 0.875], [0.25, 0.75, 0.625]], dtype=torch.bfloat16
    )
    refine_regression = (
        (torch.arange(20, dtype=torch.float32) - 9) * 0.0625
    ).reshape(2, 10).to(torch.bfloat16)
    refined = torch.zeros_like(refine_references)
    refined[..., :2] = (
        refine_regression[..., :2] + inverse_sigmoid(refine_references[..., :2])
    )
    refined[..., 2:3] = (
        refine_regression[..., 4:5] + inverse_sigmoid(refine_references[..., 2:3])
    )
    refined = refined.sigmoid()
    scaled = refine_regression.clone()
    scaled[..., :2] += inverse_sigmoid(refine_references[..., :2])
    scaled[..., :2] = scaled[..., :2].sigmoid()
    scaled[..., 4:5] += inverse_sigmoid(refine_references[..., 2:3])
    scaled[..., 4:5] = scaled[..., 4:5].sigmoid()
    scaled[..., 0:1] = scaled[..., 0:1] * 102.4 - 51.2
    scaled[..., 1:2] = scaled[..., 1:2] * 102.4 - 51.2
    scaled[..., 4:5] = scaled[..., 4:5] * 10.4 - 5.0

    classes = torch.tensor([[0.25, 1.0], [-0.75, 0.5]]).to(torch.bfloat16)
    coordinates = (
        torch.tensor(
            [
                [2.0, -1.0, 0.0, math.log(2.0), 1.5, math.log(0.5), 0.0, 1.0, 0.25, -0.5],
                [-3.0, 4.0, math.log(1.5), 0.0, -0.5, math.log(2.0), 1.0, 0.0, -0.25, 0.75],
            ]
        ).to(torch.bfloat16)
    )
    coder = NMSFreeCoder(
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 5.4],
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_num=4,
        num_classes=2,
    )
    decoded = coder.decode_single(classes, coordinates)
    boxes = decoded["bboxes"].clone()
    boxes[:, 2] -= boxes[:, 5] * 0.5

    det_grid = {
        "xbound": [-2.0, 2.0, 1.0],
        "ybound": [-2.0, 2.0, 1.0],
        "zbound": [-1.0, 1.0, 2.0],
    }
    map_grid = {
        "xbound": [-1.0, 1.0, 1.0],
        "ybound": [-1.0, 1.0, 1.0],
        "zbound": [-1.0, 1.0, 2.0],
    }
    cropper = BevFeatureSlicer(det_grid, map_grid).to(torch.bfloat16).eval()
    crop_input = (
        (torch.arange(32, dtype=torch.float32) - 15) * 0.0625
    ).reshape(1, 2, 4, 4).to(torch.bfloat16)
    crop_output = cropper(crop_input)

    adaptor = PerceptionTransformer.__new__(PerceptionTransformer)
    adaptor.det_pc_range = [-2.0, -2.0, -1.0, 2.0, 2.0, 1.0]
    adaptor.occ_pillar_h = 2
    volume_input = _bf16(
        (torch.arange(64, dtype=torch.float32) - 31) * 0.03125
    ).reshape(1, 2, 2, 4, 4)
    volume_output = adaptor._adapt_volume_for_occ(
        volume_input,
        4,
        4,
        occ_pc_range=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        occ_voxel_size=[1.0, 1.0, 1.0],
    )

    logits = _bf16(torch.tensor([[-2.0, -0.5, 0.25], [1.0, 0.0, -1.0]]))
    softplus = _bf16(F.softplus(logits))

    return {
        "inverse_sigmoid": {
            "input": bits(references),
            "output": bits(inverse),
        },
        "reference_refine": {
            "references": bits(refine_references),
            "regression": bits(refine_regression),
            "refined": bits(refined),
            "scaled": bits(scaled),
        },
        "detection": {
            "classes": bits(classes),
            "coordinates": bits(coordinates),
            "boxes": bits(boxes.float()),
            "scores": bits(decoded["scores"].float()),
            "labels": decoded["labels"].tolist(),
        },
        "map_crop": {
            "input": {"shape": list(crop_input.shape), "values": bits(crop_input)},
            "grid": bits(cropper._grid(crop_input).to(torch.bfloat16)),
            "output": {"shape": list(crop_output.shape), "values": bits(crop_output)},
        },
        "occupancy_crop": {
            "input": {"shape": list(volume_input.shape), "values": bits(volume_input)},
            "output": {"shape": list(volume_output.shape), "values": bits(volume_output)},
        },
        "softplus": {
            "input": bits(logits),
            "output": bits(softplus),
            "argmax": softplus.argmax(-1).tolist(),
        },
    }


def perception_frame_fixture(source: Path, frame_dir: Path, model_root: Path) -> dict:
    import numpy as np
    from PIL import Image
    from transformers import AutoTokenizer

    frame = json.loads((frame_dir / "frame.json").read_text())
    calibration = np.load(frame_dir / "calib.npz")
    module_path = source / "src/qwen_drive_perception/geometry.py"
    spec = importlib.util.spec_from_file_location("qwen_drive_perception_geometry", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {module_path}")
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)

    target_width, target_height = 896, 512
    lidar2img = []
    for camera, intrinsic, rotation, translation in zip(
        frame["cam_order"],
        calibration["cam_intrinsic"],
        calibration["sensor2lidar_rotation"],
        calibration["sensor2lidar_translation"],
    ):
        with Image.open(frame_dir / "images" / f"{camera}.jpg") as image:
            width, height = image.size
        matrix = geometry.build_lidar2img(intrinsic, rotation, translation)
        matrix = geometry.apply_image_scale(
            matrix, target_width / width, target_height / height
        )
        lidar2img.append(matrix.astype(np.float32).tolist())

    tokenizer = AutoTokenizer.from_pretrained(model_root)
    token = lambda value: tokenizer.convert_tokens_to_ids(value)
    encode = lambda value: tokenizer.encode(value, add_special_tokens=False)
    body = []
    for item in frame["content"]:
        if "text" in item:
            body.extend(encode(item["text"]))
        else:
            body.append(token("<|vision_start|>"))
            body.extend([token("<|image_pad|>")] * (32 // 2 * 56 // 2))
            body.append(token("<|vision_end|>"))
    prompt_ids = (
        [token("<|im_start|>")]
        + encode("user")
        + encode("\n")
        + body
        + [token("<|im_end|>")]
        + encode("\n")
        + [token("<|im_start|>")]
        + encode("assistant")
        + encode("\n")
    )
    return {
        **frame,
        "image_shapes": [[target_height, target_width, 3] for _ in frame["cam_order"]],
        "lidar2img": lidar2img,
        "lidar2ego": calibration["lidar2ego"].astype(np.float32).tolist(),
        "box_coord_system": "ego",
        "prompt_ids": prompt_ids,
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
        choices=[
            "planner-operators",
            "perception-ops",
            "perception-view",
            "perception-heads",
            "perception-frame",
            "planning-prompt",
            "planner",
        ],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--frames", type=Path)
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
    elif args.command == "perception-view":
        if args.output is None or args.source is None or args.expected_commit is None:
            parser.error("perception-view requires --output, --source, and --expected-commit")
        _require_commit(args.source, args.expected_commit)
        fixture = perception_view_fixture(args.source)
        args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    elif args.command == "perception-heads":
        if args.output is None or args.source is None or args.expected_commit is None:
            parser.error("perception-heads requires --output, --source, and --expected-commit")
        _require_commit(args.source, args.expected_commit)
        fixture = perception_heads_fixture(args.source)
        args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    elif args.command == "perception-frame":
        if (
            args.output is None
            or args.source is None
            or args.expected_commit is None
            or args.frames is None
            or args.model_root is None
        ):
            parser.error(
                "perception-frame requires --output, --source, --expected-commit, --frames, and --model-root"
            )
        _require_commit(args.source, args.expected_commit)
        fixture = perception_frame_fixture(args.source, args.frames, args.model_root)
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
