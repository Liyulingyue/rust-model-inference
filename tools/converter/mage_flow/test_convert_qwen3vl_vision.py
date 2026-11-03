import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.converter.mage_flow.convert_qwen3vl_vision import (
    DEEPSTACK_INDEXES,
    LAYERS,
    _bf16_to_f32,
    _patch_embeddings,
    convert,
)
from tools.converter.utils.gguf import read_gguf_directory, read_gguf_tensor_bytes


class _Reader:
    """A reader that serves the tensors of one synthetic text encoder."""

    def __init__(self, tensors):
        self.tensors = tensors

    def f32(self, name):
        raw, shape = self.tensors[name]
        return np.frombuffer(raw, dtype="<f4").reshape(shape)

    def shape(self, name):
        return self.tensors[name][1]

    def dtype(self, name):
        return "F32"

    def raw(self, name):
        return self.tensors[name][0]

    def has(self, name):
        return name in self.tensors


def _bf16(value):
    # BF16 is the high half of an F32.
    return struct.unpack("<I", struct.pack("<f", value))[0] >> 16


class MageQwen3VlVisionConverterTest(unittest.TestCase):
    def test_bf16_widening_is_exact(self):
        # Every BF16 widens to the F32 with the same value, and the values
        # BF16 can hold exactly survive the round trip.
        for value in (0.0, 1.0, -2.5, 0.125, 1024.0):
            bits = _bf16(value)
            widened = np.frombuffer(_bf16_to_f32(struct.pack("<H", bits)), dtype="<f4")
            self.assertEqual(float(widened[0]), float(np.float32(value)))
        # A value BF16 cannot hold is rounded to the nearest BF16, and the
        # widening of that value is stable.
        bits = _bf16(3.14159)
        widened = float(np.frombuffer(_bf16_to_f32(struct.pack("<H", bits)), dtype="<f4")[0])
        self.assertNotEqual(widened, 3.14159)
        self.assertEqual(widened, 3.140625)
        again = _bf16(widened)
        self.assertEqual(again, bits)

    def test_patch_embedding_splits_the_temporal_axis(self):
        # A conv3d patch embedding keeps one matrix per frame.
        values = np.arange(2 * 3 * 2 * 4 * 4, dtype="<f4").reshape(2, 3, 2, 4, 4)
        reader = _Reader(
            {"model.visual.patch_embed.proj.weight": (values.tobytes(), values.shape)}
        )
        frames = {name: (dims, raw) for name, dims, raw in _patch_embeddings(reader)}
        self.assertEqual(sorted(frames), ["v.patch_embd.weight", "v.patch_embd.weight.1"])
        first = np.frombuffer(frames["v.patch_embd.weight"][1], dtype="<f4").reshape(2, 3 * 4 * 4)
        second = np.frombuffer(frames["v.patch_embd.weight.1"][1], dtype="<f4").reshape(2, 3 * 4 * 4)
        self.assertTrue(np.array_equal(first, values[:, :, 0, :, :].reshape(2, 48)))
        self.assertTrue(np.array_equal(second, values[:, :, 1, :, :].reshape(2, 48)))
        self.assertFalse(np.array_equal(first, second))

    def _synthetic_encoder(self, root: Path) -> None:
        hidden, inner, heads, patch, merge = 8, 16, 2, 4, 2
        merged = hidden * merge * merge
        out_dim = 12
        positions = 16
        config = {
            "vision_config": {
                "model_type": "qwen3_vl",
                "hidden_size": hidden,
                "intermediate_size": inner,
                "num_heads": heads,
                "patch_size": patch,
                "spatial_merge_size": merge,
                "num_position_embeddings": positions,
                "out_hidden_size": out_dim,
                "deepstack_visual_indexes": list(DEEPSTACK_INDEXES),
            }
        }
        (root / "config.json").write_text(json.dumps(config))
        (root / "preprocessor_config.json").write_text(
            json.dumps({"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]})
        )
        tensors = {}
        counter = [0]

        def put(name, shape, fill):
            # Every tensor gets its own values so a mis-indexed frame shows up.
            count = int(np.prod(shape))
            values = np.arange(count, dtype="<f4") / 1024.0 + fill + counter[0]
            counter[0] += 1
            tensors[name] = (values.tobytes(), shape)

        put("model.visual.patch_embed.proj.weight", (hidden, 3, 2, patch, patch), 0.5)
        put("model.visual.patch_embed.proj.bias", (hidden,), 0.25)
        put("model.visual.pos_embed.weight", (positions, hidden), 0.125)
        for layer in range(LAYERS):
            block = f"model.visual.blocks.{layer}."
            put(f"{block}norm1.weight", (hidden,), 1.0)
            put(f"{block}norm1.bias", (hidden,), 0.0)
            put(f"{block}norm2.weight", (hidden,), 1.0)
            put(f"{block}norm2.bias", (hidden,), 0.0)
            put(f"{block}attn.qkv.weight", (3 * hidden, hidden), 0.0625)
            put(f"{block}attn.qkv.bias", (3 * hidden,), 0.0)
            put(f"{block}attn.proj.weight", (hidden, hidden), 0.0625)
            put(f"{block}attn.proj.bias", (hidden,), 0.0)
            put(f"{block}mlp.linear_fc1.weight", (inner, hidden), 0.03125)
            put(f"{block}mlp.linear_fc1.bias", (inner,), 0.0)
            put(f"{block}mlp.linear_fc2.weight", (hidden, inner), 0.03125)
            put(f"{block}mlp.linear_fc2.bias", (hidden,), 0.0)
        put("model.visual.merger.linear_fc1.weight", (merged, merged), 0.015625)
        put("model.visual.merger.linear_fc1.bias", (merged,), 0.0)
        put("model.visual.merger.linear_fc2.weight", (out_dim, merged), 0.015625)
        put("model.visual.merger.linear_fc2.bias", (out_dim,), 0.0)
        for index in range(len(DEEPSTACK_INDEXES)):
            source = f"model.visual.deepstack_merger_list.{index}."
            put(f"{source}norm.weight", (merged,), 1.0)
            put(f"{source}norm.bias", (merged,), 0.0)
            put(f"{source}linear_fc1.weight", (merged, merged), 0.015625)
            put(f"{source}linear_fc1.bias", (merged,), 0.0)
            put(f"{source}linear_fc2.weight", (out_dim, merged), 0.015625)
            put(f"{source}linear_fc2.bias", (out_dim,), 0.0)
        header = {}
        offset = 0
        for name, (raw, shape) in tensors.items():
            header[name] = {
                "dtype": "F32",
                "shape": list(shape),
                "data_offsets": [offset, offset + len(raw)],
            }
            offset += len(raw)
        blob = json.dumps(header).encode()
        (root / "model.safetensors").write_bytes(
            struct.pack("<Q", len(blob)) + blob + b"".join(raw for raw, _ in tensors.values())
        )

    def test_export_carries_the_engine_contract(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            encoder = root / "text_encoder"
            encoder.mkdir()
            self._synthetic_encoder(encoder)
            out_dir = root / "out"
            out = convert(encoder, out_dir)
            metadata = read_gguf_directory(out)[0]
            self.assertEqual(metadata["general.architecture"], "clip")
            self.assertEqual(metadata["general.type"], "mmproj")
            self.assertEqual(metadata["clip.vision.embedding_length"], 8)
            self.assertEqual(metadata["clip.vision.block_count"], LAYERS)
            self.assertEqual(metadata["clip.vision.patch_size"], 4)
            self.assertEqual(metadata["clip.vision.spatial_merge_size"], 2)
            self.assertEqual(
                [bool(value) for value in metadata["clip.vision.is_deepstack_layers"]],
                [index in DEEPSTACK_INDEXES for index in range(LAYERS)],
            )
            directory = read_gguf_directory(out)
            names = set(directory[1])
            self.assertIn("v.patch_embd.weight", names)
            self.assertIn("v.patch_embd.weight.1", names)
            self.assertIn("v.position_embd.weight", names)
            self.assertIn("mm.0.weight", names)
            self.assertIn("mm.2.weight", names)
            for layer in DEEPSTACK_INDEXES:
                self.assertIn(f"v.deepstack.{layer}.fc1.weight", names)
            for layer in range(LAYERS):
                self.assertIn(f"v.blk.{layer}.attn_qkv.weight", names)
                self.assertIn(f"v.blk.{layer}.ffn_down.weight", names)
            # The projection is stored with the dims reversed against torch.
            self.assertEqual(directory[1]["mm.2.weight"][:2], (0, (32, 12)))
            # Every tensor is F32, which is what the engine demands.
            self.assertTrue(all(entry[0] == 0 for entry in directory[1].values()))

    def test_patch_embedding_reaches_the_engine_as_two_matrices(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            encoder = root / "text_encoder"
            encoder.mkdir()
            self._synthetic_encoder(encoder)
            out = convert(encoder, root / "out")
            first = np.frombuffer(read_gguf_tensor_bytes(out, "v.patch_embd.weight"), dtype="<f4")
            second = np.frombuffer(read_gguf_tensor_bytes(out, "v.patch_embd.weight.1"), dtype="<f4")
            self.assertEqual(first.size, 8 * 3 * 4 * 4)
            self.assertEqual(second.size, 8 * 3 * 4 * 4)
            self.assertEqual(read_gguf_directory(out)[1]["v.patch_embd.weight"][:2], (0, (48, 8)))
            self.assertFalse(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
