#!/usr/bin/env python3
"""Numpy oracle for the VibeVoice ASR speech-frontend math.

Implements the TokenizerEncoder (ConvNeXt-style causal conv blocks) and the
SpeechConnector forward passes directly from the checkpoint safetensors with
numpy only (no torch), following the official microsoft/VibeVoice modeling
code. Writes f32 dumps consumed by tests/vibevoice_encoder_reference.rs:

  vibevoice_oracle_audio.f32       input chunk (83200 samples @ 24 kHz)
  vibevoice_oracle_acoustic.f32    acoustic tokenizer mean latents [26, 64]
  vibevoice_oracle_semantic.f32    semantic tokenizer mean latents [26, 128]
  vibevoice_oracle_acoustic_proj.f32   acoustic connector(mean) [26, 3584]
  vibevoice_oracle_semantic_proj.f32   semantic connector(mean) [26, 3584]
  vibevoice_oracle_combined.f32    projected sum [26, 3584]

Usage:
  python3 tools/vibevoice/vibevoice_oracle.py models/VibeVoice-ASR-Streaming-7B --out-dir /tmp
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# checkpoint access
# --------------------------------------------------------------------------- #


class Checkpoint:
    def __init__(self, model_dir: Path):
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self.model_dir = model_dir
        self._cache: dict[str, np.ndarray] = {}

    def get(self, name: str) -> np.ndarray:
        if name in self._cache:
            return self._cache[name]
        shard = self.weight_map[name]
        with open(self.model_dir / shard, "rb") as fh:
            header_len = int(np.frombuffer(fh.read(8), dtype="<u8")[0])
            header = json.loads(fh.read(header_len))
            info = header[name]
            count = int(np.prod(info["shape"]))
            assert info["dtype"] == "BF16"
            fh.seek(8 + header_len + info["data_offsets"][0])
            raw = np.frombuffer(fh.read(count * 2), dtype="<u2").astype(np.uint32)
            values = (raw << np.uint32(16)).view(np.float32).reshape(info["shape"])
        self._cache[name] = values
        return values


# --------------------------------------------------------------------------- #
# SConv1d (non-streaming, causal, constant pad) and ConvNeXt blocks
# --------------------------------------------------------------------------- #


def sconv1d_causal(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: int) -> np.ndarray:
    """x: [C_in, T]; weight: [C_out, C_in, K]; returns [C_out, T_out]."""
    c_out, c_in, k = weight.shape
    padding_total = k - 1 - (stride - 1)
    length = x.shape[1]
    n_frames = (length - k + padding_total) / stride + 1
    ideal = (math.ceil(n_frames) - 1) * stride + (k - padding_total)
    extra = ideal - length
    padded = np.concatenate(
        [np.zeros((c_in, padding_total), dtype=np.float32), x,
         np.zeros((c_in, extra), dtype=np.float32)],
        axis=1,
    )
    frames = 1 + (padded.shape[1] - k) // stride
    idx = np.arange(frames)[:, None] * stride + np.arange(k)[None, :]
    patches = padded[:, idx]  # [C_in, frames, K]
    # flatten to [frames, C_in*K] with (ci, kk) order matching the torch
    # weight layout [C_out, C_in, K]
    patches = np.transpose(patches, (1, 0, 2)).reshape(frames, c_in * k)
    w = weight.reshape(c_out, c_in * k)
    out = patches @ w.T + bias[None, :]
    return np.ascontiguousarray(out.T)


def depthwise_conv_causal(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Depthwise causal conv k=7, stride 1, constant zero pad. x: [C, T]."""
    c, t = x.shape
    k = weight.shape[2]
    padded = np.concatenate([np.zeros((c, k - 1), dtype=np.float32), x], axis=1)
    idx = np.arange(t)[:, None] + np.arange(k)[None, :]
    windows = padded[:, idx]  # [C, T, K]
    out = np.einsum("ctk,ck->ct", windows, weight[:, 0, :]) + bias[:, None]
    return out.astype(np.float32)


def conv_rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """ConvRMSNorm: x [C, T] -> per-frame RMS over channels, channel scale."""
    tc = x.T
    mean_sq = np.mean(tc * tc, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(mean_sq + eps)
    return ((tc * inv) * weight[None, :]).T.astype(np.float32)


def gelu(values: np.ndarray) -> np.ndarray:
    return (0.5 * values * (1.0 + np.vectorize(math.erf)(values / math.sqrt(2.0)))).astype(np.float32)


class Block:
    def __init__(self, cp: Checkpoint, base: str):
        self.norm = cp.get(f"{base}.norm.weight")
        self.mixer_w = cp.get(f"{base}.mixer.conv.conv.conv.weight")
        self.mixer_b = cp.get(f"{base}.mixer.conv.conv.conv.bias")
        self.gamma = cp.get(f"{base}.gamma")
        self.ffn_norm = cp.get(f"{base}.ffn_norm.weight")
        self.f1_w = cp.get(f"{base}.ffn.linear1.weight")
        self.f1_b = cp.get(f"{base}.ffn.linear1.bias")
        self.f2_w = cp.get(f"{base}.ffn.linear2.weight")
        self.f2_b = cp.get(f"{base}.ffn.linear2.bias")
        self.ffn_gamma = cp.get(f"{base}.ffn_gamma")

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = conv_rms_norm(x, self.norm, 1e-5)
        h = depthwise_conv_causal(h, self.mixer_w, self.mixer_b)
        x = (x + h * self.gamma[:, None]).astype(np.float32)

        h = conv_rms_norm(x, self.ffn_norm, 1e-5).T  # [T, C]
        h = gelu(h @ self.f1_w.T + self.f1_b[None, :])
        h = h @ self.f2_w.T + self.f2_b[None, :]
        return (x + (h * self.ffn_gamma[None, :]).T).astype(np.float32)


class Encoder:
    """VibeVoice TokenizerEncoder forward (mean only)."""

    def __init__(self, cp: Checkpoint, side: str):
        base = f"model.{side}_tokenizer.encoder"
        self.depths = [3, 3, 3, 3, 3, 3, 8]
        # encoder convs run over reversed ratios: [2, 2, 4, 5, 5, 8]
        self.ratios = [2, 2, 4, 5, 5, 8]
        self.downsamples = []
        for stage in range(len(self.depths)):
            w = cp.get(f"{base}.downsample_layers.{stage}.0.conv.conv.weight")
            b = cp.get(f"{base}.downsample_layers.{stage}.0.conv.conv.bias")
            stride = 1 if stage == 0 else self.ratios[stage - 1]
            self.downsamples.append((w, b, stride))
        self.blocks = [
            [Block(cp, f"{base}.stages.{stage}.{i}") for i in range(depth)]
            for stage, depth in enumerate(self.depths)
        ]
        self.head_w = cp.get(f"{base}.head.conv.conv.weight")
        self.head_b = cp.get(f"{base}.head.conv.conv.bias")

    def forward(self, audio: np.ndarray) -> np.ndarray:
        x = audio[None, :]  # [1, T] -> C=1
        x = sconv1d_causal(x, self.downsamples[0][0], self.downsamples[0][1], 1)
        traces = []
        for stage in range(len(self.depths)):
            if stage > 0:
                w, b, stride = self.downsamples[stage]
                x = sconv1d_causal(x, w, b, stride)
            for block in self.blocks[stage]:
                x = block.forward(x)
            traces.append(np.ascontiguousarray(x.T))  # token-major [T, C]
        x = sconv1d_causal(x, self.head_w, self.head_b, 1)
        return x.T, traces  # [T', D]


class Connector:
    def __init__(self, cp: Checkpoint, side: str):
        base = f"model.{side}_connector"
        self.fc1_w = cp.get(f"{base}.fc1.weight")
        self.fc1_b = cp.get(f"{base}.fc1.bias")
        self.norm = cp.get(f"{base}.norm.weight")
        self.fc2_w = cp.get(f"{base}.fc2.weight")
        self.fc2_b = cp.get(f"{base}.fc2.bias")

    def forward(self, features: np.ndarray) -> np.ndarray:
        x = features @ self.fc1_w.T + self.fc1_b[None, :]
        mean_sq = np.mean(x * x, axis=-1, keepdims=True)
        x = (x / np.sqrt(mean_sq + 1e-6)) * self.norm[None, :]
        x = x @ self.fc2_w.T + self.fc2_b[None, :]
        return x.astype(np.float32)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="VibeVoice ASR numpy oracle")
    parser.add_argument("model_dir", type=str)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--samples", type=int, default=83200,
        help="chunk length in samples (22+4 frames at 3200 = 83200)",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # deterministic test signal: two sines + decaying chirp envelope, f64 math
    # like torch would produce; dumped to f32 so Rust compares the same input
    t = np.arange(args.samples, dtype=np.float64) / 24000.0
    audio = (
        0.35 * np.sin(2 * np.pi * 180.0 * t)
        + 0.25 * np.sin(2 * np.pi * 950.0 * t + 0.7)
        + 0.12 * np.sin(2 * np.pi * 2900.0 * t + 1.3) * np.exp(-t * 0.8)
    ).astype(np.float32)

    cp = Checkpoint(Path(args.model_dir))
    print("encoding acoustic…")
    acoustic, acoustic_traces = Encoder(cp, "acoustic").forward(audio)
    print("encoding semantic…")
    semantic, _ = Encoder(cp, "semantic").forward(audio)
    print("projecting…")
    acoustic_proj = Connector(cp, "acoustic").forward(acoustic)
    semantic_proj = Connector(cp, "semantic").forward(semantic)
    combined = acoustic_proj + semantic_proj

    print(f"acoustic latents {acoustic.shape}, semantic {semantic.shape}")
    for name, values in [
        ("audio", audio.reshape(1, -1)),
        ("acoustic", acoustic),
        ("semantic", semantic),
        ("acoustic_proj", acoustic_proj),
        ("semantic_proj", semantic_proj),
        ("combined", combined),
    ]:
        path = out_dir / f"vibevoice_oracle_{name}.f32"
        values.astype(np.float32).tofile(path)
        print(f"  wrote {path} {values.shape} "
              f"[min {values.min():.5f}, max {values.max():.5f}]")
    for stage, trace in enumerate(acoustic_traces):
        path = out_dir / f"vibevoice_oracle_acoustic_stage{stage}.f32"
        trace.astype(np.float32).tofile(path)
        print(f"  wrote {path} {trace.shape}")


if __name__ == "__main__":
    main()
