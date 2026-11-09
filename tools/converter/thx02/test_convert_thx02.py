"""Small contract and streaming checks; no downloaded weights are required."""

from __future__ import annotations

import copy
import importlib
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from tokenizers import Tokenizer, models, normalizers

from tools.converter.utils.gguf import (
    GGML_BF16, GGML_F16, GGML_F32, GGML_Q4_0, GGML_Q4K, GGML_Q5K,
    GGML_Q6K, GGML_Q8_0, GgufWriter, Safetensors, _read_gguf,
    bf16_to_f32, f32_to_bf16, open_safetensors,
)

try:
    converter = importlib.import_module("tools.converter.thx02.convert_thx02")
except ModuleNotFoundError:
    converter = None

FIXTURES = Path(__file__).with_name("fixtures")


def safetensors(path, tensors):
    header, payload = {}, bytearray()
    for name, (values, dtype) in tensors.items():
        values = np.asarray(values, dtype="<f4")
        raw = f32_to_bf16(values.tobytes()) if dtype == "BF16" else values.tobytes()
        header[name] = dict(dtype=dtype, shape=list(values.shape),
                            data_offsets=[len(payload), len(payload) + len(raw)])
        payload.extend(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return open_safetensors(path)


class ConverterExists(unittest.TestCase):
    def test_converter_is_implemented(self):
        self.assertIsNotNone(converter, "THX-02 converter is not implemented")


@unittest.skipIf(converter is None, "converter not implemented yet")
class Thx02ConverterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("config.json", "decision_config.json", "joint_config.json"):
            (self.root / name).write_bytes((FIXTURES / name).read_bytes())
        self.config = json.loads((self.root / "config.json").read_text())
        decision = json.loads((self.root / "decision_config.json").read_text())
        vocab = dict(zip(decision["codes"], decision["token_ids"]))
        self.tokenizer = Tokenizer(models.BPE(
            vocab, [tuple(code) for code in decision["codes"] if len(code) == 2]))
        self.tokenizer.normalizer = normalizers.NFC()
        raw = json.loads(self.tokenizer.to_str())
        raw["added_tokens"] = [
            dict(id=token_id, content=token, single_word=False, lstrip=False,
                 rstrip=False, normalized=False, special=True)
            for token, token_id in {
                "<|endoftext|>": 248044, "<|im_start|>": 248045,
                "<|im_end|>": 248046, "<|vision_start|>": 248053,
                "<|vision_end|>": 248054, "<|image_pad|>": 248056,
                "<|fim_suffix|>": 248062,
            }.items()
        ]
        (self.root / "tokenizer.json").write_text(json.dumps(raw))
        (self.root / "tokenizer_config.json").write_text(json.dumps({
            "eos_token": "<|im_end|>", "chat_template": "{{ messages }}"}))
        self.contract = converter.read_contract(self.root)

    def spec(self, name="blk.0.ffn_gate.weight", source="weight", shape=(2, 256),
             kind="matrix", transform="none", vision=False, frame=None):
        return converter.TensorSpec(source, name, shape, kind, transform, vision, frame)

    def read_tensor(self, output, name):
        metadata, tensors = _read_gguf(output)
        kind, dims, size, offset = tensors[name]
        with output.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(size)
        return metadata, kind, dims, raw

    def emit(self, values, spec, target="f32", adapter=None, pair=None):
        source = safetensors(self.root / "tiny.safetensors", {spec.source: (values, "BF16")})
        output = self.root / "tiny.gguf"
        writer = GgufWriter(output)
        converter.add_tensor(writer, source, spec, target, adapter=adapter, pair=pair)
        writer.write()
        return self.read_tensor(output, spec.name)

    def test_metadata_namespaces_protocol_and_full_tokenizer(self):
        writer = GgufWriter(self.root / "unused.gguf")
        converter.write_metadata(writer, self.contract, "q4_k")
        meta = dict(writer.metadata)
        self.assertEqual(meta["general.architecture"], "thx02")
        self.assertEqual(meta["thx02.embedding_length"], 2560)
        self.assertEqual(meta["thx02.attention.key_length"], 256)
        self.assertEqual(meta["thx02.ssm.inner_size"], 4096)
        self.assertEqual(meta["thx02.ssm.group_count"], 16)
        self.assertEqual(meta["thx02.protocol_version"], 3)
        self.assertEqual(meta["thx02.component"], "decision")
        self.assertEqual(meta["thx02.recurrent_head_layout"], "qwen35-modulo-v1")
        self.assertEqual(meta["thx02.rope.dimension_sections"], [11, 11, 10, 0])
        self.assertEqual(meta["thx02.image_token_id"], 248056)
        self.assertEqual(meta["clip.vision.block_count"], 24)
        self.assertEqual(meta["clip.vision.spatial_merge_size"], 2)
        self.assertEqual(meta["clip.vision.image_min_pixels"], 65536)
        self.assertEqual(meta["clip.vision.image_max_pixels"], 1638400)
        self.assertEqual(meta["thx02.vision_storage"], "f32")
        self.assertEqual(json.loads(meta["thx02.config"]), self.config)
        self.assertEqual(json.loads(meta["thx02.joint_config"])["version"], 3)
        self.assertEqual(json.loads(meta["thx02.decision_config"])["codes"][-1], "JT")
        self.assertEqual(meta["thx02.tokenizer_json"], (self.root / "tokenizer.json").read_text())
        self.assertEqual(meta["thx02.source_revision"], "f17dca262dcd1d115e591424ce9cd012b51fc0db")
        self.assertEqual(len(meta["tokenizer.ggml.tokens"]), 248320)
        self.assertTrue(meta["tokenizer.ggml.normalizer.nfc"])

    def test_file_type_uses_gguf_ftype_not_tensor_type_or_format_string(self):
        for target, expected in {"f32": 0, "f16": 1, "bf16": 32, "q8_0": 7,
                                 "q4_0": 2, "q4_k": 14, "q5_k": 16, "q6_k": 18}.items():
            with self.subTest(target=target):
                writer = GgufWriter(self.root / "unused.gguf")
                converter.write_metadata(writer, self.contract, target)
                value = dict(writer.metadata)["general.file_type"]
                self.assertIs(type(value), int)
                self.assertEqual(value, expected)

    def test_escape_tokens_follow_tokenizer_config_instead_of_added_token_flags(self):
        self.contract.protocol_tokenizer_config = {
            "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "bos_token": None,
            "model_specific_special_tokens": {
                "audio_bos_token": "<|audio_start|>", "audio_eos_token": "<|audio_end|>",
                "audio_token": "<|audio_pad|>", "image_token": "<|image_pad|>",
                "video_token": "<|video_pad|>", "vision_bos_token": "<|vision_start|>",
                "vision_eos_token": "<|vision_end|>",
            },
            "added_tokens_decoder": {"248045": {"content": "<|im_start|>", "special": True}},
        }
        writer = GgufWriter(self.root / "unused.gguf")
        converter.write_metadata(writer, self.contract, "f32")
        tokens = dict(writer.metadata)["thx02.escape_tokens"]
        self.assertEqual(set(tokens), {
            "<|im_end|>", "<|endoftext|>", "<|audio_start|>", "<|audio_end|>",
            "<|audio_pad|>", "<|image_pad|>", "<|video_pad|>",
            "<|vision_start|>", "<|vision_end|>", "<|fim_suffix|>",
        })
        self.assertNotIn("<|im_start|>", tokens)
        self.assertEqual(tokens, sorted(tokens, key=lambda token: (-len(token), token)))

    def test_escape_tokens_ignore_tokenizer_behavior_flags(self):
        tokens = converter.escape_tokens({"eos_token": "<|im_end|>", "add_bos_token": False,
                                          "additional_special_tokens": ["<|im_start|>"],
                                          "extra_special_tokens": {"image_token": "<|image_pad|>"}})
        self.assertEqual(set(tokens), {"<|im_end|>", "<|im_start|>", "<|image_pad|>", "<|fim_suffix|>"})

    def test_thinker_body_uses_decision_escapes_and_keeps_thinker_tokenizer(self):
        protocol = {
            "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>",
            "model_specific_special_tokens": {
                "audio_bos_token": "<|audio_start|>", "audio_eos_token": "<|audio_end|>",
                "audio_token": "<|audio_pad|>", "image_token": "<|image_pad|>",
                "video_token": "<|video_pad|>", "vision_bos_token": "<|vision_start|>",
                "vision_eos_token": "<|vision_end|>",
            },
        }
        (self.root / "tokenizer_config.json").write_text(json.dumps(protocol))
        thinker_dir = self.root / "thinker" / "lm_config"
        thinker_dir.mkdir(parents=True)
        (thinker_dir / "config.json").write_text(json.dumps(self.contract.backbone))
        raw = (self.root / "tokenizer.json").read_text()
        (thinker_dir / "tokenizer.json").write_text(raw)
        thinker_config = {
            "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>",
            "chat_template": "thinker-template: {{ messages }}",
            "additional_special_tokens": [
                "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
                "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
                "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
                "<|image_pad|>", "<|video_pad|>",
            ],
            "extra_special_tokens": protocol["model_specific_special_tokens"],
        }
        (thinker_dir / "tokenizer_config.json").write_text(json.dumps(thinker_config))
        (thinker_dir.parent / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "r": 32, "lora_alpha": 32, "bias": "none",
            "fan_in_fan_out": False, "use_dora": False, "use_rslora": False,
            "use_qalora": False, "rank_pattern": {}, "alpha_pattern": {},
            "modules_to_save": None, "target_modules": sorted(converter.LORA_TARGETS),
        }))
        contract = converter.read_contract(self.root, thinker=True)
        writer = GgufWriter(self.root / "unused.gguf")
        converter.write_metadata(writer, contract, "f16", thinker=True)
        metadata = dict(writer.metadata)
        self.assertEqual(len(converter.escape_tokens(contract.tokenizer_config)), 18)
        self.assertEqual(metadata["thx02.escape_tokens"], converter.escape_tokens(protocol))
        self.assertEqual(len(metadata["thx02.escape_tokens"]), 10)
        self.assertNotIn("<|im_start|>", metadata["thx02.escape_tokens"])
        self.assertEqual(metadata["thx02.tokenizer_json"], raw)
        self.assertEqual(metadata["qwen35.tokenizer_json"], raw)
        self.assertEqual(metadata["tokenizer.ggml.token_type"][248045], 3)
        self.assertEqual(metadata["tokenizer.chat_template"], thinker_config["chat_template"])
        self.assertEqual(metadata["tokenizer.ggml.eos_token_id"], 248046)

    def test_pinned_config_rejects_shape_and_semantic_drift(self):
        changes = [
            ("model_type", "qwen3_5"), ("max_options", 256),
            ("backbone_config.text_config.hidden_size", 1024),
            ("backbone_config.text_config.attn_output_gate", False),
            ("backbone_config.text_config.head_dim", 160),
            ("backbone_config.text_config.linear_num_key_heads", 32),
            ("backbone_config.text_config.linear_value_head_dim", 64),
            ("backbone_config.text_config.rope_parameters.mrope_interleaved", False),
            ("backbone_config.vision_config.temporal_patch_size", 1),
            ("backbone_config.vision_config.deepstack_visual_indexes", [8]),
        ]
        for key, value in changes:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.config)
                node = bad
                parts = key.split(".")
                for part in parts[:-1]:
                    node = node[part]
                node[parts[-1]] = value
                (self.root / "config.json").write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, parts[-1]):
                    converter.read_contract(self.root)

    def test_codebook_order_ids_and_readout_marker_are_pinned(self):
        for filename, key, value in [
            ("decision_config.json", "codes", ["A"] * 255),
            ("decision_config.json", "token_ids", list(range(255))),
            ("decision_config.json", "attention_mode", "causal"),
            ("decision_config.json", "temperature", 0),
            ("joint_config.json", "marker", "<|im_end|>"),
            ("joint_config.json", "readout_offset", 1),
        ]:
            with self.subTest(key=key):
                original = (FIXTURES / filename).read_text()
                bad = json.loads(original)
                bad[key] = value
                (self.root / filename).write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    converter.read_contract(self.root)
                (self.root / filename).write_text(original)

    def test_tokenizer_code_and_special_ids_are_checked(self):
        raw = json.loads((self.root / "tokenizer.json").read_text())
        for kind in ("code", "special", "normalizer"):
            with self.subTest(kind=kind):
                bad = copy.deepcopy(raw)
                if kind == "code":
                    bad["model"]["vocab"]["JT"] = 20
                elif kind == "special":
                    bad["added_tokens"][-1]["id"] = 248063
                else:
                    bad["normalizer"] = None
                (self.root / "tokenizer.json").write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    converter.read_contract(self.root)

    def test_all_checkpoint_shapes_and_names_are_covered(self):
        specs = converter.tensor_specs(self.contract.backbone)
        self.assertEqual(len(specs), 725)
        expected = {spec.source: spec.shape for spec in specs if spec.component == "model"}
        self.assertEqual(len(expected), 723)
        by_name = {spec.name: spec for spec in specs}
        self.assertEqual(by_name["blk.3.attn_q.weight"].shape, (8192, 2560))
        self.assertEqual(by_name["blk.0.ssm_conv1d.weight"].output_shape, (8192, 4))
        self.assertEqual(by_name["cls.output.weight"].output_shape, (255, 2560))
        self.assertEqual(by_name["v.patch_embd.weight.1"].output_shape, (1024, 3, 16, 16))
        offset, header = 0, {}
        for name, shape in expected.items():
            end = offset + math.prod(shape) * 2
            header[name] = dict(shape=list(shape), dtype="BF16", data_offsets=[offset, end])
            offset = end
        source = Safetensors(self.root / "unread.safetensors", header, 8, offset + 8)
        converter.validate_source(source, expected)
        for name in expected:
            with self.subTest(name=name):
                header[name]["shape"][-1] += 1
                with self.assertRaisesRegex(ValueError, "shape"):
                    converter.validate_source(source, expected)
                header[name]["shape"][-1] -= 1
        name = next(iter(expected))
        saved = header.pop(name)
        with self.assertRaisesRegex(ValueError, "missing"):
            converter.validate_source(source, expected)
        header[name] = saved
        header["unexpected.weight"] = saved
        with self.assertRaisesRegex(ValueError, "unexpected"):
            converter.validate_source(source, expected)

    def test_malformed_offsets_dtype_and_truncation_are_rejected(self):
        source = safetensors(self.root / "source.safetensors", {"weight": ([1, 2], "BF16")})
        for field, value in (("data_offsets", [0, 2]), ("data_offsets", [-2, 2]),
                             ("dtype", "I32"), ("shape", [0])):
            with self.subTest(field=field, value=value):
                bad = copy.deepcopy(source)
                bad.header["weight"][field] = value
                with self.assertRaises(ValueError):
                    converter.validate_source(bad, {"weight": (2,)})
        source.file_size -= 1
        with self.assertRaisesRegex(ValueError, "truncated"):
            converter.validate_source(source, {"weight": (2,)})

    def test_norm_offset_excludes_gated_delta_norm_and_a_log_is_negative_exp(self):
        values = np.array([-1, 0, 1], dtype="<f4")
        for transform, expected in (("norm", [0, 1, 2]), ("none", [-1, 0, 1]),
                                    ("a_log", -np.exp(values))):
            with self.subTest(transform=transform):
                _, kind, dims, raw = self.emit(values, self.spec(
                    name=f"norm_{transform}", shape=(3,), kind="float", transform=transform))
                self.assertEqual((kind, dims), (GGML_F32, (3,)))
                np.testing.assert_allclose(np.frombuffer(raw, dtype="<f4"), expected, rtol=1e-7)

    def test_classifier_rows_are_not_transposed(self):
        values = np.arange(3 * 256, dtype="<f4").reshape(3, 256) / 256
        _, kind, dims, raw = self.emit(values, self.spec(name="cls.output.weight", shape=(3, 256)))
        self.assertEqual((kind, dims), (GGML_F32, (256, 3)))
        expected = np.frombuffer(bf16_to_f32(f32_to_bf16(values.tobytes())), dtype="<f4")
        np.testing.assert_array_equal(np.frombuffer(raw, dtype="<f4"), expected)

    def test_requested_formats_use_existing_encoders(self):
        cases = {"f32": (GGML_F32, 2048), "f16": (GGML_F16, 1024),
                 "bf16": (GGML_BF16, 1024), "q8_0": (GGML_Q8_0, 544),
                 "q4_0": (GGML_Q4_0, 288), "q4_k": (GGML_Q4K, 288),
                 "q5_k": (GGML_Q5K, 352), "q6_k": (GGML_Q6K, 420)}
        values = np.linspace(-1, 1, 512, dtype="<f4").reshape(2, 256)
        for target, (expected_type, size) in cases.items():
            with self.subTest(target=target):
                _, kind, dims, raw = self.emit(values, self.spec(), target)
                self.assertEqual((kind, dims, len(raw)), (expected_type, (256, 2), size))
                self.assertTrue(any(raw))
        with self.assertRaisesRegex(ValueError, "Q4_1.*exporter"):
            converter.tensor_type(self.spec(), "q4_1")

    def test_vision_low_bit_outputs_keep_float_weights_and_positions(self):
        matrix = self.spec(name="v.blk.0.attn_qkv.weight", vision=True)
        for target in ("q4_0", "q4_k", "q5_k", "q6_k"):
            self.assertEqual(converter.tensor_type(matrix, target), GGML_F32)
        self.assertEqual(converter.tensor_type(matrix, "q8_0"), GGML_Q8_0)
        for target in converter.FORMATS:
            for name in ("v.position_embd.weight", "v.blk.0.ln1.weight", "v.patch_embd.bias"):
                self.assertEqual(converter.tensor_type(self.spec(name=name, kind="float", vision=True), target), GGML_F32)

    def test_conv3d_temporal_frames_are_split_without_permuting_channels(self):
        values = np.arange(48, dtype="<f4").reshape(2, 3, 2, 2, 2)
        for frame in (0, 1):
            with self.subTest(frame=frame):
                spec = self.spec(name=f"v.patch_embd.weight.{frame}", shape=values.shape,
                                 kind="patch", vision=True, frame=frame)
                _, kind, dims, raw = self.emit(values, spec)
                self.assertEqual((kind, dims), (GGML_F32, (2, 2, 3, 2)))
                np.testing.assert_array_equal(np.frombuffer(raw, dtype="<f4"), values[:, :, frame].reshape(-1))

    def test_payload_is_streamed_in_bounded_row_chunks(self):
        values = np.arange(5 * 256, dtype="<f4").reshape(5, 256) / 256
        source = safetensors(self.root / "source.safetensors", {"weight": (values, "BF16")})
        with patch.object(converter, "CHUNK_BYTES", 1024):
            chunks = list(converter.tensor_chunks(source, self.spec(shape=values.shape), GGML_F32))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 2048 for chunk in chunks))
        expected = bf16_to_f32(f32_to_bf16(values.tobytes()))
        self.assertEqual(b"".join(chunks), expected)

    def test_recurrent_weight_layout_permutes_every_value_head_axis(self):
        head_order = [*range(0, 32, 2), *range(1, 32, 2)]
        value_order = np.concatenate([np.arange(h * 128, (h + 1) * 128) for h in head_order])
        qkv_order = np.concatenate([np.arange(4096), 4096 + value_order])
        for suffix, shape, order, columns in (
            ("in_proj_qkv.weight", (8192, 1), qkv_order, False),
            ("conv1d.weight", (8192, 1, 4), qkv_order, False),
            ("in_proj_z.weight", (4096, 1), value_order, False),
            ("in_proj_a.weight", (32, 1), head_order, False),
            ("in_proj_b.weight", (32, 1), head_order, False),
            ("A_log", (32,), head_order, False),
            ("dt_bias", (32,), head_order, False),
            ("out_proj.weight", (2, 4096), value_order, True),
        ):
            with self.subTest(suffix=suffix):
                values = np.arange(math.prod(shape), dtype="<f4").reshape(shape)
                rounded = np.frombuffer(bf16_to_f32(f32_to_bf16(values.tobytes())), dtype="<f4").reshape(shape)
                name = "blk.0.ssm_conv1d.weight" if suffix == "conv1d.weight" else "transformed"
                spec = self.spec(name=name, source="language_model.layers.0.linear_attn." + suffix,
                                 shape=shape, kind="float")
                _, _, _, raw = self.emit(values, spec)
                expected = rounded[:, order] if columns else rounded[order]
                np.testing.assert_array_equal(np.frombuffer(raw, dtype="<f4"), expected.reshape(-1))

    def test_modulo_layout_preserves_repeat_interleave_recurrence(self):
        rng = np.random.default_rng(42)
        order = converter.recurrent_order(2, 4, 1)
        np.testing.assert_array_equal(order, [0, 2, 1, 3])
        q, k = rng.normal(size=(2, 3)), rng.normal(size=(2, 3))
        values, gates = rng.normal(size=(5, 4, 2)), rng.normal(size=(5, 4, 2))
        alpha, beta = rng.uniform(0.1, 0.9, size=(2, 5, 4))
        projection = rng.normal(size=(3, 8))
        def run(v, z, decay, strength, output, modulo):
            state = np.zeros((4, 3, 2))
            results = []
            for t in range(5):
                hidden = []
                for h in range(4):
                    key = h % 2 if modulo else h // 2
                    state[h] *= decay[t, h]
                    error = v[t, h] - k[key] @ state[h]
                    state[h] += strength[t, h] * np.outer(k[key], error)
                    hidden.append((q[key] @ state[h]) * z[t, h])
                results.append(output @ np.asarray(hidden).reshape(-1))
            return results
        reference = run(values, gates, alpha, beta, projection, False)
        actual = run(values[:, order], gates[:, order], alpha[:, order], beta[:, order],
                     projection[:, converter.recurrent_order(2, 4, 2)], True)
        np.testing.assert_allclose(actual, reference, atol=1e-12)

    def test_lora_merge_rounds_back_to_backbone_bf16_before_export(self):
        spec = self.spec(source="language_model.layers.3.self_attn.q_proj.weight", shape=(3, 256))
        a = np.full((32, 256), 0.125, dtype="<f4")
        b = np.full((3, 32), 0.025, dtype="<f4")
        key = "base_model.model.language_model.layers.3.self_attn.q_proj"
        adapter = safetensors(self.root / "adapter.safetensors", {
            key + ".lora_A.weight": (a, "F32"), key + ".lora_B.weight": (b, "F32")})
        pairs = converter.lora_contract(adapter, [spec], {"r": 32, "lora_alpha": 32, "target_modules": ["q_proj"]})
        values = np.ones((3, 256), dtype="<f4")
        _, _, _, raw = self.emit(values, spec, adapter=adapter, pair=pairs[spec.source])
        expected = bf16_to_f32(f32_to_bf16((values + b @ a).tobytes()))
        self.assertEqual(raw, expected)
        del adapter.header[key + ".lora_B.weight"]
        with self.assertRaisesRegex(ValueError, "missing"):
            converter.lora_contract(adapter, [spec], {"r": 32, "lora_alpha": 32, "target_modules": ["q_proj"]})

    def test_missing_backbone_and_thinker_components_fail_before_output(self):
        output = self.root / "never.gguf"
        with self.assertRaisesRegex((ValueError, FileNotFoundError), "model.safetensors"):
            converter.convert(self.root, output)
        self.assertFalse(output.exists())
        with self.assertRaisesRegex((ValueError, FileNotFoundError), "thinker"):
            converter.read_contract(self.root, thinker=True)

    def test_failed_write_removes_partial_file_and_preserves_existing_output(self):
        output = self.root / "output.gguf"
        writer = GgufWriter(output)
        def broken():
            yield struct.pack("<f", 1)
            raise ValueError("truncated source")
        writer.add_tensor_chunks("weight", GGML_F32, (2,), 8, broken)
        with self.assertRaisesRegex(ValueError, "truncated"):
            converter.write_atomic(writer, output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob("*.partial")), [])
        output.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            converter.write_atomic(GgufWriter(output), output)
        self.assertEqual(output.read_bytes(), b"keep")

    def test_checksum_mismatch_is_rejected(self):
        source = self.root / "source.bin"
        source.write_bytes(b"abc")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            converter.checked_sha256(source, "0" * 64)


if __name__ == "__main__":
    unittest.main()
