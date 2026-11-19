"""Small synthetic flow/HiFT checkpoints and independent GGUF payload checks."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.utils.parametrizations import weight_norm

from tools.converter.sophon_realtime import code2wav as m
from tools.converter.sophon_realtime import convert as c
from tools.converter.utils.gguf import GGML_F32, _read_gguf


def fixture(root, component):
    root.mkdir(exist_ok=True)
    config = copy.deepcopy(m.CONFIGS[component])
    if component == "flow":
        config.update(
            input_size=4,
            output_size=4,
            spk_embed_dim=6,
            vocab_size=7,
            pre_lookahead_channels=16,
            pre_lookahead_len=1,
            dim=256,
            depth=1,
            heads=8,
            dim_head=32,
            conv_pos_kernel=3,
        )
    else:
        config.update(
            in_channels=4,
            base_channels=16,
            nb_harmonics=2,
            upsample_rates=[2],
            upsample_kernel_sizes=[4],
            istft_n_fft=4,
            resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1]],
            source_resblock_kernel_sizes=[3],
            source_resblock_dilation_sizes=[[1]],
            f0_cond_channels=16,
            conv_pre_look_right=1,
        )
    state = {}
    for name, shape in m.source_specs(component, config):
        value = torch.linspace(-0.25, 0.35, int(np.prod(shape))).reshape(shape)
        if name.endswith(".parametrizations.weight.original0"):
            value = torch.linspace(0.75, 1.25, shape[0]).reshape(shape)
        state[name] = value
    torch.save(state, root / (component + ".pt"))
    (root / "cosyvoice3.yaml").write_text("# synthetic fixture\n")
    return config, state


def convert(root, output, component, config, quant="f32"):
    with patch.object(m, "K_WORKERS", 1):
        return m.convert_component(
            root,
            output,
            component=component,
            quant=quant,
            validate_pin=False,
            config=config,
        )


def independent_weights(state):
    result = {}
    marker = ".parametrizations.weight.original"
    for name, value in state.items():
        if marker not in name:
            result[name] = value.numpy()
        elif name.endswith("original1"):
            prefix = name.split(marker)[0]
            out_channels, in_channels, kernel = value.shape
            module = weight_norm(torch.nn.Conv1d(in_channels, out_channels, kernel))
            module.load_state_dict(
                {
                    "parametrizations.weight.original0": state[prefix + marker + "0"],
                    "parametrizations.weight.original1": value,
                    "bias": state[prefix + ".bias"],
                }
            )
            result[prefix + ".weight"] = module.weight.detach().numpy()
    return result


class Code2WavTests(unittest.TestCase):
    def test_pinned_schema_and_native_layout(self):
        flow = dict(m.source_specs("flow", m.CONFIGS["flow"]))
        hift = dict(m.source_specs("hift", m.CONFIGS["hift"]))
        self.assertEqual((len(flow), len(hift)), (330, 328))
        self.assertEqual(flow["input_embedding.weight"], (6561, 80))
        self.assertEqual(flow["decoder.estimator.rotary_embed.inv_freq"], (32,))
        self.assertEqual(
            flow["decoder.estimator.transformer_blocks.21.attn_norm.linear.weight"],
            (6144, 1024),
        )
        self.assertEqual(
            hift["ups.0.parametrizations.weight.original1"], (256, 512, 16)
        )
        self.assertEqual(
            hift["conv_post.parametrizations.weight.original1"], (18, 64, 7)
        )
        self.assertEqual(
            hift["f0_predictor.condnet.0.parametrizations.weight.original0"],
            (512, 1, 1),
        )

    def test_all_formats_independently_decode_and_keep_convs_small_matrices_f32(self):
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize

        budgets = dict(
            f32=0,
            f16=0.0005,
            bf16=0.005,
            q8_0=0.01,
            q4_0=0.15,
            q4_1=0.15,
            q4_k=0.12,
            q5_k=0.08,
            q6_k=0.04,
        )
        for component in ("flow", "hift"):
            with self.subTest(component=component), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                config, state = fixture(root, component)
                expected = independent_weights(state)
                for quant, budget in budgets.items():
                    out = root / (quant + ".gguf")
                    convert(root, out, component, config, quant)
                    metadata, directory = _read_gguf(out)
                    self.assertEqual(
                        metadata["general.architecture"], "sophon_realtime_" + component
                    )
                    self.assertEqual(metadata["sophon_realtime.component"], component)
                    self.assertEqual(
                        metadata["sophon_realtime.source_revision"], c.REVISION
                    )
                    self.assertEqual(
                        metadata["sophon_realtime.source_precision"], "F32"
                    )
                    self.assertEqual(
                        json.loads(
                            metadata[f"sophon_realtime.{component}_config_json"]
                        ),
                        config,
                    )
                    self.assertEqual(set(directory), set(expected))
                    self.assertEqual(
                        json.loads(metadata["sophon_realtime.tensor_types_json"]),
                        {name: info[0] for name, info in directory.items()},
                    )
                    if component == "hift":
                        self.assertTrue(
                            all(info[0] == GGML_F32 for info in directory.values())
                        )
                        self.assertEqual(len(directory), 34)
                        self.assertIn(
                            "F32",
                            metadata["sophon_realtime.weight_norm_materialization"],
                        )
                    else:
                        self.assertEqual(len(directory), 36)
                        self.assertEqual(
                            directory["spk_embed_affine_layer.weight"][0], GGML_F32
                        )
                        self.assertEqual(
                            directory["pre_lookahead_layer.conv1.weight"][0], GGML_F32
                        )
                        self.assertEqual(
                            directory[
                                "decoder.estimator.transformer_blocks.0.attn.to_q.weight"
                            ][0],
                            c.target_type((256, 256), quant),
                        )
                    with out.open("rb") as stream:
                        for name, reference in expected.items():
                            kind, _, length, offset = directory[name]
                            stream.seek(offset)
                            actual = dequantize(
                                np.frombuffer(stream.read(length), dtype=np.uint8),
                                GGMLQuantizationType(kind),
                            ).reshape(reference.shape)
                            if kind == GGML_F32:
                                np.testing.assert_array_equal(actual, reference)
                            elif quant in ("f16", "bf16"):
                                dtype = (
                                    torch.float16 if quant == "f16" else torch.bfloat16
                                )
                                rounded = (
                                    torch.from_numpy(reference.copy())
                                    .to(dtype)
                                    .float()
                                    .numpy()
                                )
                                np.testing.assert_array_equal(actual, rounded)
                            relative = np.linalg.norm(
                                actual.astype(np.float64) - reference
                            ) / np.linalg.norm(reference)
                            self.assertLessEqual(relative, budget)

    def test_weightnorm_materialization_is_chunk_independent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config, state = fixture(root, "hift")
            expected = independent_weights(state)
            out = root / "chunked.gguf"
            with patch.object(c, "STREAM_CHUNK_BYTES", 32):
                convert(root, out, "hift", config)
            _, directory = _read_gguf(out)
            with out.open("rb") as stream:
                for name, reference in expected.items():
                    _, _, length, offset = directory[name]
                    stream.seek(offset)
                    actual = np.frombuffer(stream.read(length), dtype="<f4").reshape(
                        reference.shape
                    )
                    np.testing.assert_array_equal(actual, reference)

    def test_invalid_state_and_weightnorm_values_fail_before_publication(self):
        for problem in (
            "missing",
            "unknown",
            "shape",
            "dtype",
            "nan_g",
            "nan_v",
            "zero_v",
            "nested",
        ):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                config, state = fixture(root, "hift")
                g = "conv_pre.parametrizations.weight.original0"
                v = "conv_pre.parametrizations.weight.original1"
                if problem == "missing":
                    del state[g]
                elif problem == "unknown":
                    state["unknown"] = torch.ones(1)
                elif problem == "shape":
                    state[g] = state[g][:-1]
                elif problem == "dtype":
                    state[v] = state[v].half()
                elif problem == "nan_g":
                    state[g][0] = float("nan")
                elif problem == "nan_v":
                    state[v][0] = float("nan")
                elif problem == "zero_v":
                    state[v][0] = 0
                elif problem == "nested":
                    state = {"state_dict": state}
                torch.save(state, root / "hift.pt")
                out = root / "out.gguf"
                with self.assertRaises(ValueError):
                    convert(root, out, "hift", config)
                self.assertFalse(out.exists())
                self.assertEqual(list(root.glob(".out.gguf.*")), [])

    def test_bad_config_pin_component_and_format_fail(self):
        for component, key, value in (
            ("flow", "heads", 3),
            ("flow", "conv_pos_groups", 3),
            ("hift", "base_channels", 7),
            ("hift", "upsample_rates", []),
        ):
            with self.subTest(
                component=component, key=key
            ), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                config, _ = fixture(root, component)
                config[key] = value
                with self.assertRaises(ValueError):
                    convert(root, root / "out.gguf", component, config)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config, _ = fixture(root, "flow")
            for kwargs in (
                dict(component="bad"),
                dict(component="flow", quant="bad"),
                dict(component="flow"),
            ):
                with self.assertRaises(ValueError):
                    m.convert_component(root, root / "out.gguf", **kwargs)
            with self.assertRaises(ValueError):
                m.convert_component(
                    root, root / "out.gguf", component="flow", config=config
                )

    def test_atomic_failures_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config, _ = fixture(root, "flow")
            out = root / "out.gguf"
            out.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                convert(root, out, "flow", config)
            self.assertEqual(out.read_bytes(), b"existing")
            out.unlink()
            out.symlink_to(root / "missing")
            with self.assertRaises(FileExistsError):
                convert(root, out, "flow", config)
            out.unlink()
            with patch.object(
                m.GgufWriter, "write", side_effect=RuntimeError("failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    convert(root, out, "flow", config)
            self.assertFalse(out.exists())
            self.assertEqual(list(root.glob(".out.gguf.*")), [])

    def test_release_root_selects_talker_component(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config, _ = fixture(root / "talker", "flow")
            out = root / "out.gguf"
            convert(root, out, "flow", config)
            self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()
