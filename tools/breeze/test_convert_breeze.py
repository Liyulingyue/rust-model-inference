#!/usr/bin/env python3
"""Run with: uv run --with numpy python tools/breeze/test_convert_breeze.py."""

import json
from pathlib import Path
import struct
import tempfile
import unittest

try:
    import convert_breeze
except ModuleNotFoundError:
    convert_breeze = None


def safetensors(path, tensors):
    header, payload = {}, bytearray()
    for name, dtype, shape, raw in tensors:
        start = len(payload)
        payload.extend(raw)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    raw_header = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)


def read_gguf(path):
    """Independent parser for the GGUF scalar metadata and unquantized test tensors."""
    with path.open("rb") as source:
        def number(fmt):
            return struct.unpack(fmt, source.read(struct.calcsize(fmt)))[0]

        def string():
            return source.read(number("<Q")).decode()

        assert source.read(4) == b"GGUF"
        assert number("<I") == 3
        count, metadata_count = number("<Q"), number("<Q")
        metadata = {}
        for _ in range(metadata_count):
            key, kind = string(), number("<I")
            metadata[key] = string() if kind == 8 else number({4: "<I", 10: "<Q"}[kind])
        directory = {}
        for _ in range(count):
            name = string()
            dims = tuple(number("<Q") for _ in range(number("<I")))
            directory[name] = (dims, number("<I"), number("<Q"))
        data_start = (source.tell() + 31) // 32 * 32
        tensors = {}
        for name, (dims, kind, offset) in directory.items():
            elements = 1
            for dim in dims:
                elements *= dim
            source.seek(data_start + offset)
            tensors[name] = (dims, kind, source.read(elements * {0: 4, 30: 2}[kind]))
    return metadata, tensors


class ConversionTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(convert_breeze, "Breeze converter has not been implemented")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name) / "model"
        self.model.mkdir()
        (self.model / "audio_tokenizer").mkdir()
        self.out = Path(self.tmp.name) / "output"
        self.config = '{\n "model_type": "breeze", "architectures": ["BreezeForConditionalGeneration"]\n}\n'
        self.tokenizer = '{ "version": "1.0", "model": {"type":"BPE", "vocab":{"你":0}, "merges":[]} }\n'
        self.audio_config = '{ "model_type": "qwen3_tts_tokenizer_12hz", "architectures":["Qwen3TTSTokenizerV2Model"] }\n'
        (self.model / "config.json").write_text(self.config)
        (self.model / "tokenizer.json").write_text(self.tokenizer)
        (self.model / "audio_tokenizer/config.json").write_text(self.audio_config)
        # Includes signed zero and a NaN payload to detect accidental float conversion.
        safetensors(self.model / "first.safetensors", [
            ("backbone_model.layers.0.weight", "BF16", [2, 3], bytes.fromhex("0000803f0080817f7fff80bf")),
            ("codec_model.legacy.initialized", "F32", [1], bytes.fromhex("0000803f")),
        ])
        safetensors(self.model / "second.safetensors", [
            ("text_encoder.weight", "BF16", [1], bytes.fromhex("003f")),
        ])
        safetensors(self.model / "audio_tokenizer/model.safetensors", [
            ("decoder.conv.weight", "F32", [1, 2, 1], bytes.fromhex("0000803f0100c07f")),
        ])
        self.index = {
            "metadata": {"total_size": 18},
            "weight_map": {
                "backbone_model.layers.0.weight": "first.safetensors",
                "codec_model.legacy.initialized": "first.safetensors",
                "text_encoder.weight": "second.safetensors",
            },
        }
        self.write_index()

    def write_index(self):
        (self.model / "model.safetensors.index.json").write_text(json.dumps(self.index))

    def reject(self, pattern):
        with self.assertRaisesRegex((ValueError, FileNotFoundError), pattern):
            convert_breeze.convert(self.model, self.out)
        self.assertFalse(self.out.exists() and any(self.out.iterdir()))

    def test_roundtrip_preserves_metadata_names_dims_types_and_raw_bytes(self):
        main, codec = convert_breeze.convert(self.model, self.out)
        self.assertEqual((main.name, codec.name), ("breeze-tts-2-BF16.gguf", "breeze-tts-2-codec-F32.gguf"))
        metadata, tensors = read_gguf(main)
        self.assertEqual(metadata["general.architecture"], "breeze")
        self.assertEqual(metadata["breeze.config"], self.config)
        self.assertEqual(metadata["breeze.tokenizer_json"], self.tokenizer)
        self.assertEqual(metadata["breeze.oracle.commit"], "e2c5ac2f54fe15daa94237a7dbf31e446660a4c9")
        self.assertEqual(tensors, {
            "backbone_model.layers.0.weight": ((3, 2), 30, bytes.fromhex("0000803f0080817f7fff80bf")),
            "codec_model.legacy.initialized": ((1,), 0, bytes.fromhex("0000803f")),
            "text_encoder.weight": ((1,), 30, bytes.fromhex("003f")),
        })
        metadata, tensors = read_gguf(codec)
        self.assertEqual(metadata["general.architecture"], "breeze_audio")
        self.assertEqual(metadata["breeze_audio.config"], self.audio_config)
        self.assertEqual(tensors, {"decoder.conv.weight": ((1, 2, 1), 0, bytes.fromhex("0000803f0100c07f"))})

    def test_rejects_missing_or_wrong_shard_membership_and_total_size(self):
        self.index["weight_map"]["text_encoder.weight"] = "first.safetensors"
        self.write_index()
        self.reject("index|shard")
        self.index["weight_map"]["text_encoder.weight"] = "second.safetensors"
        self.index["metadata"]["total_size"] = 20
        self.write_index()
        self.reject("total_size")

    def test_rejects_swapped_shard_mapping_and_external_metadata_symlink(self):
        self.index["weight_map"]["text_encoder.weight"] = "first.safetensors"
        self.index["weight_map"]["backbone_model.layers.0.weight"] = "second.safetensors"
        self.write_index()
        self.reject("index|shard")
        self.index["weight_map"]["text_encoder.weight"] = "second.safetensors"
        self.index["weight_map"]["backbone_model.layers.0.weight"] = "first.safetensors"
        self.write_index()
        config = self.model / "config.json"
        outside = Path(self.tmp.name) / "outside-config.json"
        outside.write_bytes(config.read_bytes())
        config.unlink()
        config.symlink_to(outside)
        self.reject("escape|outside")

    def test_rejects_nonfinite_metadata_and_noncontiguous_tensor_ranges(self):
        (self.model / "config.json").write_text('{"model_type":"breeze","rms_norm_eps":NaN}')
        self.reject("JSON")
        (self.model / "config.json").write_text(self.config)
        shard = self.model / "second.safetensors"
        header = json.dumps({"text_encoder.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [2, 4]}}).encode()
        shard.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
        self.reject("range|offset|gap")

    def test_rejects_traversal_absolute_and_symlink_input_paths(self):
        outside = Path(self.tmp.name) / "outside.safetensors"
        outside.write_bytes((self.model / "second.safetensors").read_bytes())
        (self.model / "link.safetensors").symlink_to(outside)
        for shard in ("../outside.safetensors", str(outside), "link.safetensors"):
            with self.subTest(shard=shard):
                self.index["weight_map"]["text_encoder.weight"] = shard
                self.write_index()
                self.reject("escape|relative|outside")

    def test_rejects_inconsistent_shape_truncated_data_and_codec_dtype(self):
        shard = self.model / "second.safetensors"
        safetensors(shard, [("text_encoder.weight", "BF16", [2], b"\x00\x00")])
        self.reject("byte|shape")
        safetensors(shard, [("text_encoder.weight", "BF16", [1], b"\x00\x00")])
        shard.write_bytes(shard.read_bytes()[:-1])
        self.reject("byte|offset|truncated|exceeds")
        safetensors(shard, [("text_encoder.weight", "BF16", [1], b"\x00\x00")])
        safetensors(self.model / "audio_tokenizer/model.safetensors", [("decoder.weight", "BF16", [1], b"\x00\x00")])
        self.reject("dtype")

    def test_rejects_unknown_architecture_and_invalid_metadata(self):
        config_path = self.model / "config.json"
        for config in ('{"model_type":"qwen3"}', '{"model_type":"breeze","model_type":"qwen3"}', '[]', '{"model_type":NaN}'):
            with self.subTest(config=config):
                config_path.write_text(config)
                self.reject("model_type|duplicate|object|JSON")

    def test_existing_output_is_unchanged_and_other_output_is_not_created(self):
        self.out.mkdir()
        existing = self.out / "breeze-tts-2-codec-F32.gguf"
        existing.write_bytes(b"user-owned")
        with self.assertRaises(FileExistsError):
            convert_breeze.convert(self.model, self.out)
        self.assertEqual(list(self.out.iterdir()), [existing])
        self.assertEqual(existing.read_bytes(), b"user-owned")


if __name__ == "__main__":
    unittest.main()
