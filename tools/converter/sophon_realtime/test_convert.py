"""Synthetic converter contracts; torch is an independent FP8 test oracle only."""

import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.converter.sophon_realtime import convert as c
from tools.converter.utils.gguf import GGML_F32, _read_gguf


def safefile(path, tensors):
    header, payload = {}, bytearray()
    for name, (dtype, array) in tensors.items():
        raw = array.tobytes()
        header[name] = dict(
            dtype=dtype,
            shape=list(array.shape),
            data_offsets=[len(payload), len(payload) + len(raw)],
        )
        payload.extend(raw)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def fixture(root):
    root.mkdir(exist_ok=True)
    config = {
        "model_type": "qwen3_vl",
        "text_config": dict(
            hidden_size=256,
            intermediate_size=256,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            num_hidden_layers=1,
            vocab_size=4,
            max_position_embeddings=32,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            attention_bias=False,
            rope_parameters=dict(
                rope_theta=5000000,
                mrope_interleaved=True,
                mrope_section=[24, 20, 20],
                rope_type="default",
            ),
            bos_token_id=0,
            eos_token_id=1,
        ),
        "vision_config": {},
        "tie_word_embeddings": False,
        "quantization_config": c.QUANTIZATION,
    }
    (root / "config.json").write_text(json.dumps(config))
    tensors = {}
    for source, _, shape in c.decoder_specs(config):
        if len(shape) == 2 and ".layers." in source:
            tensors[source] = ("F8_E4M3", np.full(shape, 0x38, dtype=np.uint8))
            tensors[source + "_scale"] = (
                "F32",
                np.full((shape[0], 1), 0.25, dtype="<f4"),
            )
        else:
            tensors[source] = ("F32", np.ones(shape, dtype="<f4"))
    safefile(root / "model.safetensors", tensors)
    tok = {
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": [["a", "b"]]},
        "normalizer": {"type": "NFC"},
        "added_tokens": [
            {
                "id": 2,
                "content": "<x>",
                "special": True,
                "normalized": False,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
            }
        ],
    }
    (root / "tokenizer.json").write_text(json.dumps(tok))
    (root / "chat_template.jinja").write_text("{{ messages }}\n")
    return config, tensors


def _worker_failure(*args):
    raise RuntimeError("worker failed")


class ConverterTests(unittest.TestCase):
    def test_fp8_every_code_matches_torch_and_channel_scales(self):
        import torch

        codes = np.arange(256, dtype=np.uint8)
        expected = torch.from_numpy(codes).view(torch.float8_e4m3fn).float().numpy()
        finite = np.isfinite(expected)
        np.testing.assert_array_equal(c.decode_fp8(codes[finite]), expected[finite])
        for scale in (0.125, 1.0, 3.25):
            np.testing.assert_array_equal(
                c.decode_fp8(codes[finite]) * np.float32(scale),
                expected[finite] * np.float32(scale),
            )
        for code in codes[~finite]:
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                c.decode_fp8(np.array([code], dtype=np.uint8))

    def test_all_formats_stream_and_keep_norm_f32(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config, _ = fixture(root)
            for fmt in c.FORMATS:
                out = root / (fmt + ".gguf")
                c.convert(root, out, quant=fmt, validate_pin=False)
                meta, ts = _read_gguf(out)
                self.assertEqual(meta["general.architecture"], "sophon_realtime")
                self.assertEqual(meta["sophon_realtime.source_revision"], c.REVISION)
                self.assertEqual(meta["tokenizer.ggml.tokens"][2], "<x>")
                self.assertTrue(meta["tokenizer.ggml.normalizer.nfc"])
                self.assertEqual(meta["tokenizer.chat_template"], "{{ messages }}\n")
                self.assertEqual(ts["output_norm.weight"][0], GGML_F32)
                self.assertEqual(
                    ts["blk.0.attn_q.weight"][0], c.target_type((256, 256), fmt)
                )
                with self.assertRaises(FileExistsError):
                    c.convert(root, out, validate_pin=False)

    def test_inventory_scales_shapes_and_nonfinite_fail_atomically(self):
        for problem in (
            "missing_scale",
            "bad_scale",
            "nonfinite_scale",
            "scale_shape",
            "nonfinite",
            "shape",
            "unknown",
            "missing_tensor",
            "missing_component",
            "config",
        ):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                config, tensors = fixture(root)
                key = "language_model.model.layers.0.self_attn.q_proj.weight"
                if problem == "missing_scale":
                    del tensors[key + "_scale"]
                elif problem == "bad_scale":
                    tensors[key + "_scale"] = ("F32", np.zeros((256, 1), dtype="<f4"))
                elif problem == "nonfinite_scale":
                    tensors[key + "_scale"] = (
                        "F32",
                        np.full((256, 1), np.nan, dtype="<f4"),
                    )
                elif problem == "scale_shape":
                    tensors[key + "_scale"] = ("F32", np.ones((256,), dtype="<f4"))
                elif problem == "nonfinite":
                    tensors[key][1][0, 0] = 0x7F
                elif problem == "shape":
                    tensors[key] = ("F8_E4M3", np.ones((255, 256), dtype=np.uint8))
                elif problem == "unknown":
                    tensors["surprise.weight"] = ("F32", np.ones(1, dtype="<f4"))
                elif problem == "missing_tensor":
                    del tensors["language_model.model.norm.weight"]
                elif problem == "missing_component":
                    (root / "tokenizer.json").unlink()
                elif problem == "config":
                    config["text_config"]["hidden_act"] = "gelu"
                    (root / "config.json").write_text(json.dumps(config))
                safefile(root / "model.safetensors", tensors)
                out = root / "bad.gguf"
                with self.assertRaises((ValueError, FileNotFoundError)):
                    c.convert(root, out, validate_pin=False)
                self.assertFalse(out.exists())
                self.assertEqual(list(root.glob(".bad.gguf.*")), [])

    def test_writer_failure_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            out = root / "bad.gguf"
            with patch.object(c.GgufWriter, "write", side_effect=RuntimeError("disk")):
                with self.assertRaisesRegex(RuntimeError, "disk"):
                    c.convert(root, out, validate_pin=False)
            self.assertFalse(out.exists())
            self.assertEqual(list(root.glob(".bad.gguf.*")), [])

    def test_release_root_selects_thinker_even_with_top_level_config(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "config.json").write_text('{"model_type":"sophon_realtime"}')
            fixture(root / "thinker")
            c.convert(root, root / "text.gguf", validate_pin=False)
            self.assertTrue((root / "text.gguf").is_file())

    def test_pin_rejects_synthetic_model(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            with self.assertRaisesRegex(ValueError, "pinned"):
                c.convert(root, root / "bad.gguf")

    def test_vision_temporal_split_norm_deepstack_and_pixel_budgets(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            vision = dict(
                depth=3,
                hidden_size=4,
                intermediate_size=8,
                num_heads=2,
                out_hidden_size=8,
                patch_size=2,
                temporal_patch_size=2,
                spatial_merge_size=2,
                num_position_embeddings=4,
                in_channels=3,
                deepstack_visual_indexes=[0, 1, 2],
            )
            config = {"vision_config": vision}
            tensors = {
                name: (
                    "F32",
                    np.arange(np.prod(shape), dtype="<f4").reshape(shape) / 100,
                )
                for name, _, shape in c.vision_specs(config)
            }
            safefile(root / "model.safetensors", tensors)
            pre = dict(
                size=dict(shortest_edge=16, longest_edge=64),
                patch_size=2,
                temporal_patch_size=2,
                merge_size=2,
                image_mean=[0.5] * 3,
                image_std=[0.5] * 3,
            )
            (root / "preprocessor_config.json").write_text(json.dumps(pre))
            out = root / "vision.gguf"
            c.vision_writer(out, c.Reader(root), root, config).write()
            meta, ts = _read_gguf(out)
            self.assertEqual(meta["clip.projector_type"], "qwen3vl_merger")
            self.assertEqual(meta["clip.vision.image_min_pixels"], 16)
            self.assertEqual(meta["clip.vision.is_deepstack_layers"], [True] * 3)
            self.assertEqual(ts["v.post_ln.weight"][1], (4,))
            self.assertIn("v.deepstack.2.fc1.weight", ts)
            original = tensors["model.visual.patch_embed.proj.weight"][1]
            for frame, name in enumerate(
                ("v.patch_embd.weight", "v.patch_embd.weight.1")
            ):
                _, _, length, offset = ts[name]
                with out.open("rb") as f:
                    f.seek(offset)
                    actual = np.frombuffer(f.read(length), dtype="<f4")
                np.testing.assert_array_equal(
                    actual, original[:, :, frame, :, :].reshape(-1)
                )

    def test_f32_payload_is_scaled_fp8_and_bf16_is_exact(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, tensors = fixture(root)
            name = "language_model.model.norm.weight"
            tensors[name] = ("BF16", np.full((256,), 0x3F80, dtype="<u2"))
            safefile(root / "model.safetensors", tensors)
            out = root / "f32.gguf"
            c.convert(root, out, validate_pin=False)
            _, ts = _read_gguf(out)
            for name, value in [
                ("blk.0.attn_q.weight", 0.25),
                ("output_norm.weight", 1.0),
            ]:
                _, _, length, offset = ts[name]
                with out.open("rb") as f:
                    f.seek(offset)
                    actual = np.frombuffer(f.read(length), dtype="<f4")
                np.testing.assert_array_equal(
                    actual, np.full(actual.shape, value, dtype="<f4")
                )

    def test_truncated_or_overlapping_source_and_index_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            source = root / "model.safetensors"
            source.write_bytes(source.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "truncated"):
                c.Reader(root)
            fixture(root)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"unknown": "model.safetensors"}})
            )
            with self.assertRaisesRegex(ValueError, "index mismatch"):
                c.Reader(root)

    def test_pair_failure_publishes_neither_component(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            out = root / "text.gguf"
            vision = root / "vision.gguf"
            with patch.object(c, "vision_writer", side_effect=ValueError("bad vision")):
                with self.assertRaisesRegex(ValueError, "bad vision"):
                    c.convert(root, out, mmproj=vision, validate_pin=False)
            self.assertFalse(out.exists())
            self.assertFalse(vision.exists())
            self.assertEqual(list(root.glob(".*.gguf.*")), [])

    def test_kquant_parallel_is_byte_identical_and_worker_failure_is_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, tensors = fixture(root)
            for name, (_, values) in tensors.items():
                if name.endswith("_scale"):
                    values[:, 0] = (
                        np.arange(values.shape[0], dtype=np.float32) / 512 + 0.125
                    )
            safefile(root / "model.safetensors", tensors)
            for quant in ("q4_k", "q5_k", "q6_k"):
                serial = root / (quant + "-serial.gguf")
                parallel = root / (quant + "-parallel.gguf")
                with patch.object(c, "K_WORKERS", 1), patch.object(
                    c, "STREAM_CHUNK_BYTES", 32768
                ):
                    c.convert(root, serial, quant=quant, validate_pin=False)
                with patch.object(c, "K_WORKERS", 2), patch.object(
                    c, "STREAM_CHUNK_BYTES", 32768
                ):
                    c.convert(root, parallel, quant=quant, validate_pin=False)
                self.assertEqual(serial.read_bytes(), parallel.read_bytes())
            out = root / "failed.gguf"
            with patch.object(c, "K_WORKERS", 2), patch.object(
                c, "_encode_chunk", new=_worker_failure
            ):
                with self.assertRaisesRegex(RuntimeError, "worker failed"):
                    c.convert(root, out, quant="q4_k", validate_pin=False)
            self.assertFalse(out.exists())
            self.assertEqual(list(root.glob(".failed.gguf.*")), [])

    def test_independent_export_verifier_and_corruption_detection(self):
        from tools.converter.sophon_realtime.verify import verify

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            for quant in c.FORMATS:
                out = root / (quant + ".gguf")
                c.convert(root, out, quant=quant, validate_pin=False)
                result = verify(root, out, quant, hash_output=False)
                self.assertEqual(result["status"], "PASS")
            out = root / "f32.gguf"
            _, directory = _read_gguf(out)
            with out.open("r+b") as stream:
                stream.seek(directory["output_norm.weight"][3])
                stream.write(struct.pack("<f", 100.0))
            with self.assertRaisesRegex(ValueError, "casting mismatch"):
                verify(root, out, "f32", hash_output=False)

    def test_quantized_export_rejects_changed_f32_tensor(self):
        from tools.converter.sophon_realtime.verify import verify

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            out = root / "q4_k.gguf"
            c.convert(root, out, quant="q4_k", validate_pin=False)
            self.assertEqual(
                verify(root, out, "q4_k", hash_output=False)["status"], "PASS"
            )
            _, directory = _read_gguf(out)
            with out.open("r+b") as stream:
                stream.seek(directory["output_norm.weight"][3])
                stream.write(struct.pack("<f", 2.0))
            with self.assertRaisesRegex(ValueError, "casting mismatch"):
                verify(root, out, "q4_k", hash_output=False)

    def test_low_magnitude_tensor_degradation_cannot_hide_in_global_average(self):
        from tools.converter.sophon_realtime.verify import verify

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, tensors = fixture(root)
            source = "language_model.model.layers.0.self_attn.q_proj.weight"
            tensors[source + "_scale"][1].fill(0.01)
            safefile(root / "model.safetensors", tensors)
            out = root / "q4_k.gguf"
            c.convert(root, out, quant="q4_k", validate_pin=False)
            self.assertEqual(
                verify(root, out, "q4_k", hash_output=False)["status"], "PASS"
            )
            _, directory = _read_gguf(out)
            _, _, length, offset = directory["blk.0.attn_q.weight"]
            with out.open("r+b") as stream:
                stream.seek(offset)
                stream.write(bytes(length))
            with self.assertRaisesRegex(
                ValueError, r"blk\.0\.attn_q\.weight:.*relative RMSE"
            ):
                verify(root, out, "q4_k", hash_output=False)

    def test_audio_all_formats_mixed_k_types_and_independent_payloads(self):
        from tools.converter.sophon_realtime.audio import audio_specs, convert_audio
        from tools.converter.sophon_realtime.verify import verify

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            audio = root / "audio"
            audio.mkdir()
            config = dict(
                model_type="ultravox",
                projector_act="swiglu",
                projector_ln_mid=True,
                stack_factor=2,
                hidden_size=512,
                text_config=dict(hidden_size=256),
                audio_config=dict(
                    d_model=128,
                    downsample_hidden_size=2,
                    encoder_attention_heads=2,
                    encoder_ffn_dim=256,
                    encoder_layers=1,
                    num_mel_bins=128,
                ),
            )
            (audio / "config.json").write_text(json.dumps(config))
            pre = dict(feature_size=128, hop_length=160, n_fft=400, sampling_rate=16000)
            (audio / "preprocessor_config.json").write_text(json.dumps(pre))
            tensors = {
                name: ("BF16", np.full(shape, 0x3F80, dtype="<u2"))
                for name, _, shape in audio_specs(config)
            }
            safefile(audio / "model.safetensors", tensors)
            for quant in c.FORMATS:
                out = root / ("audio-" + quant + ".gguf")
                convert_audio(root, out, quant=quant, validate_pin=False)
                metadata, directory = _read_gguf(out)
                self.assertEqual(
                    metadata["general.architecture"], "sophon_realtime_audio"
                )
                self.assertEqual(metadata["sophon_realtime.source_precision"], "BF16")
                self.assertEqual(
                    json.loads(metadata["sophon_realtime.audio_preprocessor_json"]), pre
                )
                self.assertEqual(directory["audio_tower.conv2d1.weight"][0], GGML_F32)
                if quant in ("q4_k", "q5_k", "q6_k"):
                    self.assertEqual(
                        directory["audio_tower.layers.0.self_attn.q_proj.weight"][0],
                        GGML_F32,
                    )
                    self.assertEqual(
                        directory["audio_tower.layers.0.fc2.weight"][0],
                        c.target_type((128, 256), quant),
                    )
                self.assertEqual(
                    verify(root, out, quant, hash_output=False)["status"], "PASS"
                )
                if quant == "q4_k":
                    with out.open("r+b") as stream:
                        stream.seek(
                            directory["audio_tower.layers.0.self_attn.q_proj.weight"][3]
                        )
                        stream.write(struct.pack("<f", 2.0))
                    with self.assertRaisesRegex(ValueError, "casting mismatch"):
                        verify(root, out, quant, hash_output=False)
                with self.assertRaises(FileExistsError):
                    convert_audio(root, out, validate_pin=False)
            tensors.pop("audio_tower.ln_post.bias")
            safefile(audio / "model.safetensors", tensors)
            with self.assertRaisesRegex(ValueError, "component tensor"):
                convert_audio(root, root / "bad-audio.gguf", validate_pin=False)
            self.assertFalse((root / "bad-audio.gguf").exists())

    def test_storage_symlinks_keep_validated_source_paths_working(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            (root / "model.safetensors").rename(root / "backing.safetensors")
            (root / "model.safetensors").symlink_to(root / "backing.safetensors")
            c.convert(root, root / "out.gguf", validate_pin=False)
            self.assertTrue((root / "out.gguf").is_file())

    def test_zero_source_nonzero_export_is_rejected(self):
        from tools.converter.sophon_realtime.verify import verify

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, tensors = fixture(root)
            for name, (_, values) in tensors.items():
                if not name.endswith("_scale"):
                    values.fill(0)
            safefile(root / "model.safetensors", tensors)
            out = root / "zero.gguf"
            c.convert(root, out, quant="q8_0", validate_pin=False)
            self.assertEqual(
                verify(root, out, "q8_0", hash_output=False)["relative_rmse"], 0
            )
            _, directory = _read_gguf(out)
            with out.open("r+b") as stream:
                stream.seek(directory["token_embd.weight"][3])
                stream.write(struct.pack("<e", 1.0) + bytes([1]))
            with self.assertRaisesRegex(ValueError, "degradation"):
                verify(root, out, "q8_0", hash_output=False)

    def test_unknown_format(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "format"):
                c.convert(Path(d), Path(d) / "x", quant="oops")


if __name__ == "__main__":
    unittest.main()
