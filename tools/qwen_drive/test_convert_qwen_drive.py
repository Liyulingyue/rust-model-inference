import hashlib
import json
import subprocess
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.dots.convert_dots_tts import (
    GGML_BF16,
    GGML_F32,
    GgufWriter,
    read_gguf_directory,
    read_gguf_tensor_bytes,
)
from tools.qwen_drive import convert_qwen_drive as converter
from tools.qwen_drive.convert_qwen_drive import (
    export_head,
    export_model,
    output_paths,
    validate_source,
    verify_outputs,
)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    header = {}
    payload = bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def component_manifest(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> dict:
    return {
        "file": path.name,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tensors": [
            {"name": name, "dtype": dtype, "shape": list(shape)}
            for name, (dtype, shape, _raw) in tensors.items()
        ],
    }


def write_head_configs(root: Path) -> None:
    planner = {
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "layers_per_kv": 4,
        "rms_norm_eps": 1e-5,
        "nav_command_classes": 3,
        "ego_status_dim": 8,
        "history_dynamics_dim": 2,
        "time_embed_dim": 128,
        "time_embed_scale": 1000.0,
        "fourier_num_features": 16,
        "fourier_max_frequency": 16.0,
        "mrope_section": [11, 11, 10],
        "rope_theta": 10_000_000.0,
    }
    (root / "config.json").write_text(
        json.dumps(
            {
                "expert_config": planner,
                "num_future_points": 50,
                "num_history_points": 16,
                "trajectory_point_dim": 3,
                "trajectory_scale": [165.0, 25.0, 1.5703125],
                "num_inference_steps": 10,
                "min_one_minus_t": 0.1,
                "noise_init_std": 1.0,
                "noise_seed": 42,
                "trajectory_hz": 10.0,
                "max_reasoning_tokens": 256,
            }
        )
    )
    for name in ("planner-sft", "planner-rl"):
        component = root / name
        component.mkdir(exist_ok=True)
        (component / "config.json").write_text(json.dumps(planner))
    perception = {
        "llm_dim": 2560,
        "vit_dim": 1024,
        "embed_dim": 256,
        "bev_h": 200,
        "bev_w": 200,
        "occ_pillar_h": 16,
        "occ_dim": 32,
        "occ_num_classes": 10,
        "det_num_classes": 7,
        "map_num_classes": 6,
        "num_query": 900,
        "code_size": 10,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "image_size": [896, 512],
        "det_pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 5.4],
        "det_voxel_size": [0.512, 0.512, 10.4],
        "nuscenes_occ_pc_range": [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        "nuscenes_occ_voxel_size": [0.4, 0.4, 6.4],
        "nuplan_occ_pc_range": [-50.0, -50.0, -4.0, 50.0, 50.0, 4.0],
        "nuplan_occ_voxel_size": [0.5, 0.5, 0.5],
        "map_xbound": [-30.0, 30.0, 0.15],
        "map_ybound": [-15.0, 15.0, 0.15],
        "frustum_range": [0.0, 0.0, 1.0, 896.0, 512.0, 60.0],
        "frustum_size": [16.0, 16.0, 0.5],
    }
    perception_dir = root / "perception"
    perception_dir.mkdir(exist_ok=True)
    (perception_dir / "config.json").write_text(json.dumps(perception))


class QwenDriveExportTest(unittest.TestCase):
    def test_head_exports_fixed_geometry_metadata(self):
        tensors = {
            "planning_expert.out_proj.weight": ("BF16", (3, 4), bytes(range(24))),
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_head_configs(root)
            component, manifest = self.make_component(root, "planner-sft", tensors)
            out = root / "planner.gguf"

            export_head(component, out, "qwen_drive_planner", manifest)

            metadata, _ = read_gguf_directory(out)
            self.assertEqual(metadata["qwen_drive_planner.hidden_size"], 1024)
            self.assertEqual(metadata["qwen_drive_planner.head_dim"], 256)
            self.assertEqual(metadata["qwen_drive_planner.mrope_section"], [11, 11, 10])
            self.assertEqual(
                metadata["qwen_drive_planner.trajectory_scale"],
                [165.0, 25.0, 1.5703125],
            )
    def test_vlm_view_preserves_the_source_config(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            model = root / "model"
            view = root / "view"
            model.mkdir()
            view.mkdir()
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "vlm_config": {
                            "text_config": {
                                "num_hidden_layers": 32,
                                "mtp_num_hidden_layers": 1,
                            }
                        }
                    }
                )
            )
            (model / "tokenizer.json").write_text("{}")

            converter._prepare_vlm_view(model, view)

            config = json.loads((view / "config.json").read_text())
            self.assertEqual(config["text_config"]["num_hidden_layers"], 32)
            self.assertEqual(config["text_config"]["mtp_num_hidden_layers"], 1)
            self.assertEqual(
                (view / "tokenizer.json").resolve(), (model / "tokenizer.json").resolve()
            )

    def test_vlm_text_conversion_disables_the_unshipped_mtp_layer(self):
        with mock.patch.object(subprocess, "run") as run:
            converter._run_llama_converter(
                Path("/tmp/llama.cpp"),
                Path("/tmp/model"),
                Path("/tmp/model.gguf"),
                mmproj=False,
            )
        command = run.call_args.args[0]
        bootstrap = command[command.index("-c") + 1]
        self.assertIn('sys.argv.append("--no-nextn")', bootstrap)
        self.assertIn('add_bool("tokenizer.ggml.normalizer.nfc", True)', bootstrap)

    def test_output_paths_use_the_five_fixed_source_precision_names(self):
        paths = output_paths(Path("/tmp/out"))
        self.assertEqual(
            [path.name for path in paths],
            [
                "Qwen-Drive-1.0-4B-BF16.gguf",
                "Qwen-Drive-1.0-4B-mmproj-BF16.gguf",
                "Qwen-Drive-1.0-planner-sft-BF16.gguf",
                "Qwen-Drive-1.0-planner-rl-BF16.gguf",
                "Qwen-Drive-1.0-perception-F32.gguf",
            ],
        )

    def test_export_model_validates_every_source_before_writing(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_head_configs(root)
            tensors = {
                "vlm": {"vlm.model.embed_tokens.weight": ("BF16", (1, 2), bytes(4))},
                "planner-sft": {"planning_expert.out.weight": ("BF16", (1, 2), bytes(4))},
                "planner-rl": {"planning_expert.out.weight": ("BF16", (1, 2), bytes(4))},
                "perception": {"bev_modeling.out.weight": ("F32", (1, 2), bytes(8))},
            }
            components = {}
            for name, values in tensors.items():
                component = root if name == "vlm" else root / name
                component.mkdir(exist_ok=True)
                source = component / "model.safetensors"
                write_safetensors(source, values)
                components[name] = component_manifest(source, values)
                components[name]["file"] = str(source.relative_to(root))
            manifest = {"version": 1, "components": components}

            def fake_vlm(_model_dir, _llama_cpp, vlm, mmproj, overwrite):
                for path, architecture in ((vlm, "qwen35"), (mmproj, "clip")):
                    writer = GgufWriter(path)
                    writer.add_meta("general.architecture", architecture)
                    if architecture == "qwen35":
                        writer.add_meta("tokenizer.ggml.normalizer.nfc", True)
                    if architecture == "clip":
                        writer.add_meta("clip.projector_type", "qwen3vl_merger")
                    writer.add_tensor("weight", GGML_BF16, (2, 1), bytes(4))
                    writer.write(overwrite=overwrite)

            out = root / "out"
            with mock.patch.object(converter, "_load_manifest", return_value=manifest), mock.patch.object(
                converter, "_validate_llama_cpp"
            ), mock.patch.object(converter, "_export_vlm_pair", side_effect=fake_vlm):
                paths = export_model(root, root / "llama.cpp", out)
            self.assertEqual(paths, output_paths(out.resolve()))
            self.assertTrue(all(path.is_file() for path in paths))

            bad = json.loads(json.dumps(manifest))
            bad["components"]["planner-rl"]["sha256"] = "0" * 64
            other = root / "other"
            with mock.patch.object(converter, "_load_manifest", return_value=bad), mock.patch.object(
                converter, "_validate_llama_cpp"
            ), mock.patch.object(converter, "_export_vlm_pair", side_effect=fake_vlm):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    export_model(root, root / "llama.cpp", other)
            self.assertFalse(other.exists())

    def test_verify_outputs_rejects_wrong_architecture(self):
        with tempfile.TemporaryDirectory() as raw:
            paths = output_paths(Path(raw))
            for path, architecture in zip(
                paths,
                ("qwen2", "clip", "qwen_drive_planner", "qwen_drive_planner", "qwen_drive_perception"),
            ):
                writer = GgufWriter(path)
                writer.add_meta("general.architecture", architecture)
                if architecture == "clip":
                    writer.add_meta("clip.projector_type", "qwen3vl_merger")
                writer.add_tensor("weight", GGML_F32, (1,), bytes(4))
                writer.write()
            with self.assertRaisesRegex(ValueError, "qwen35"):
                verify_outputs(paths)

    def test_verify_outputs_compares_head_payloads_with_sources(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            components = {}
            fixtures = {
                "planner-sft": {"planning_expert.out.weight": ("BF16", (1, 2), b"\x01\x02\x03\x04")},
                "planner-rl": {"planning_expert.out.weight": ("BF16", (1, 2), b"\x05\x06\x07\x08")},
                "perception": {"bev_modeling.out.weight": ("F32", (1, 2), bytes(range(8)))},
            }
            for name, tensors in fixtures.items():
                component, one = self.make_component(root, name, tensors)
                components[name] = one["components"][name]
                components[name]["file"] = str((component / "model.safetensors").relative_to(root))
            manifest = {"version": 1, "components": components}
            paths = output_paths(root)
            for path, architecture in zip(paths[:2], ("qwen35", "clip")):
                writer = GgufWriter(path)
                writer.add_meta("general.architecture", architecture)
                if architecture == "clip":
                    writer.add_meta("clip.projector_type", "qwen3vl_merger")
                writer.add_tensor("weight", GGML_BF16, (2, 1), bytes(4))
                writer.write()
            export_head(root / "planner-sft", paths[2], "qwen_drive_planner", manifest)
            export_head(root / "planner-rl", paths[3], "qwen_drive_planner", manifest)
            export_head(root / "perception", paths[4], "qwen_drive_perception", manifest)
            verify_outputs(paths, model_dir=root, manifest=manifest)

            source = root / "planner-rl/model.safetensors"
            write_safetensors(source, {"planning_expert.out.weight": ("BF16", (1, 2), bytes(4))})
            components["planner-rl"] = component_manifest(
                source, {"planning_expert.out.weight": ("BF16", (1, 2), bytes(4))}
            )
            components["planner-rl"]["file"] = "planner-rl/model.safetensors"
            with self.assertRaisesRegex(ValueError, "payload mismatch"):
                verify_outputs(paths, model_dir=root, manifest=manifest)

    def test_script_entrypoint_imports_repo_tools(self):
        repo = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, str(repo / "tools/qwen_drive/convert_qwen_drive.py"), "--help"],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("export", result.stdout)
        self.assertIn("verify", result.stdout)

    def make_component(
        self,
        root: Path,
        name: str,
        tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
    ) -> tuple[Path, dict]:
        write_head_configs(root)
        component = root / name
        component.mkdir(exist_ok=True)
        source = component / "model.safetensors"
        write_safetensors(source, tensors)
        return component, {
            "version": 1,
            "components": {name: component_manifest(source, tensors)},
        }

    def test_head_tensor_names_preserve_bytes_and_reject_unknowns(self):
        tensors = {
            "planning_expert.out_proj.weight": ("BF16", (3, 4), bytes(range(24))),
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            component, manifest = self.make_component(root, "planner-sft", tensors)
            out = root / "planner.gguf"
            export_head(component, out, "qwen_drive_planner", manifest, False)
            metadata, directory = read_gguf_directory(out)
            self.assertEqual(metadata["general.architecture"], "qwen_drive_planner")
            self.assertEqual(
                directory["qwen_drive_planner.out_proj.weight"],
                (GGML_BF16, (4, 3), 24),
            )
            self.assertEqual(
                read_gguf_tensor_bytes(out, "qwen_drive_planner.out_proj.weight"),
                bytes(range(24)),
            )

            manifest["components"]["planner-sft"]["tensors"] = []
            with self.assertRaisesRegex(ValueError, "unexpected tensor"):
                export_head(component, root / "bad.gguf", "qwen_drive_planner", manifest, False)

    def test_perception_preserves_f32_and_records_bf16_compute(self):
        tensors = {
            "bev_modeling.proj.weight": ("F32", (2, 2), bytes(range(16))),
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            component, manifest = self.make_component(root, "perception", tensors)
            out = root / "perception.gguf"
            export_head(component, out, "qwen_drive_perception", manifest, False)
            metadata, directory = read_gguf_directory(out)
            self.assertEqual(metadata["qwen_drive_perception.compute_dtype"], "bfloat16")
            self.assertEqual(
                directory["qwen_drive_perception.proj.weight"],
                (GGML_F32, (2, 2), 16),
            )

    def test_perception_scalar_tensors_use_one_element_gguf_shape(self):
        tensors = {
            "bev_modeling.norm.num_batches_tracked": (
                "F32",
                (),
                struct.pack("<f", 7.0),
            ),
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            component, manifest = self.make_component(root, "perception", tensors)
            out = root / "perception.gguf"

            export_head(component, out, "qwen_drive_perception", manifest)

            _, directory = read_gguf_directory(out)
            self.assertEqual(
                directory["qwen_drive_perception.norm.num_batches_tracked"],
                (GGML_F32, (1,), 4),
            )

    def test_manifest_rejects_hash_shape_dtype_missing_and_truncated_payload(self):
        tensors = {
            "planning_expert.out_proj.weight": ("BF16", (3, 4), bytes(range(24))),
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            component, manifest = self.make_component(root, "planner-sft", tensors)

            for field, value, message in (
                ("sha256", "0" * 64, "SHA256"),
                ("size", 1, "size"),
            ):
                bad = json.loads(json.dumps(manifest))
                bad["components"]["planner-sft"][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_source(root, bad)

            for field, value, message in (
                ("shape", [4, 3], "shape"),
                ("dtype", "F32", "dtype"),
            ):
                bad = json.loads(json.dumps(manifest))
                bad["components"]["planner-sft"]["tensors"][0][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    export_head(component, root / f"bad-{field}.gguf", "qwen_drive_planner", bad, False)

            bad = json.loads(json.dumps(manifest))
            bad["components"]["planner-sft"]["tensors"].append(
                {"name": "planning_expert.missing", "dtype": "BF16", "shape": [1]}
            )
            with self.assertRaisesRegex(ValueError, "missing tensor"):
                export_head(component, root / "bad-missing.gguf", "qwen_drive_planner", bad, False)

            source = component / "model.safetensors"
            source.write_bytes(source.read_bytes()[:-1])
            manifest["components"]["planner-sft"]["size"] -= 1
            manifest["components"]["planner-sft"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "truncated tensor payload"):
                export_head(component, root / "bad-truncated.gguf", "qwen_drive_planner", manifest, False)

    def test_output_is_atomic_and_requires_explicit_overwrite(self):
        tensors = {
            "planning_expert.out_proj.weight": ("BF16", (3, 4), bytes(range(24))),
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            component, manifest = self.make_component(root, "planner-sft", tensors)
            out = root / "planner.gguf"
            out.write_bytes(b"old")
            with self.assertRaises(FileExistsError):
                export_head(component, out, "qwen_drive_planner", manifest, False)
            self.assertEqual(out.read_bytes(), b"old")

            export_head(component, out, "qwen_drive_planner", manifest, True)
            self.assertTrue(out.read_bytes().startswith(b"GGUF"))
            self.assertEqual(list(root.glob(f".{out.name}.*")), [])


if __name__ == "__main__":
    unittest.main()
