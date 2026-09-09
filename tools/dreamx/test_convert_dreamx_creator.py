import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

from tools.dreamx import convert_dreamx_creator as converter
from tools.dots.convert_dots_tts import read_gguf_directory, read_gguf_tensor_bytes
from tools.dreamx.convert_dreamx_creator import (
    JOINT_LAYERS,
    SourceTensor,
    _publish_pair,
    _spool_pt_checkpoint,
    _write_pair_files,
    build_pair_metadata,
    collect_source_tensors,
    export_model,
    build_tensor_plan,
    build_inventory,
    f32_to_bf16_rne,
    load_pt_state_dict,
    map_name,
    output_paths,
    pair_id,
    read_safetensor_sources,
    should_quantize,
)


class DreamXOracleTraceTest(unittest.TestCase):
    def test_cuda_requirement_fails_before_upstream_load(self):
        from tools.dreamx.dreamx_oracle_trace import require_cuda

        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            require_cuda(torch)


class DreamXExporterIsolationTest(unittest.TestCase):
    def test_import_keeps_shared_q8_row_width_validation(self):
        with self.assertRaisesRegex(ValueError, "row width"):
            converter._gguf._tensor_nbytes(converter.GGML_Q8_0, (16, 2))


VIDEO_CONFIG = {
    "dim": 3072,
    "ffn_dim": 14336,
    "num_heads": 24,
    "num_layers": 30,
    "in_dim": 48,
    "out_dim": 48,
    "text_dim": 4096,
    "text_len": 512,
}

AUDIO_CONFIG = {
    "dim": 1536,
    "ffn_dim": 8960,
    "num_heads": 12,
    "num_layers": 30,
    "in_dim": 128,
    "out_dim": 128,
    "text_dim": 4096,
    "text_len": 512,
}


def write_safetensors(path: Path, names=("weight",)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {}
    offset = 0
    for name in names:
        header[name] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [offset, offset + 4],
        }
        offset += 4
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))


def make_minimal_layout(root: Path, omit: str | None = None) -> None:
    files = (
        "creator/audio_model/diffusion_pytorch_model.safetensors",
        "creator/cross_attn_weights.safetensors",
        "wan2.2_ti2v_5b/models_t5_umt5-xxl-enc-bf16.pth",
        "wan2.2_ti2v_5b/Wan2.2_VAE.pth",
        "audio_vae/diffusion_pytorch_model.safetensors",
        "refiner/sr_dit_5b.pt",
        "refiner/latent_upsampler_flash.pt",
        "refiner/latent_upsampler_2d_causal.pt",
        "refiner/lightvae_nu_scheme3.pt",
        "wan2.2_ti2v_5b/google/umt5-xxl/tokenizer.json",
    )
    for relative in files:
        if relative == omit:
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".safetensors":
            names = (
                tuple(f"joint_blocks.{layer}.weight" for layer in JOINT_LAYERS)
                if path.name == "cross_attn_weights.safetensors"
                else ("weight",)
            )
            write_safetensors(path, names)
        else:
            path.write_bytes(b"{}" if path.suffix == ".json" else b"")

    video = root / "creator/video_model"
    audio = root / "creator/audio_model"
    video.mkdir(parents=True, exist_ok=True)
    audio.mkdir(parents=True, exist_ok=True)
    (video / "config.json").write_text(json.dumps(VIDEO_CONFIG))
    (audio / "config.json").write_text(json.dumps(AUDIO_CONFIG))
    shard = "diffusion_pytorch_model-00001-of-00001.safetensors"
    write_safetensors(video / shard)
    (video / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": shard}})
    )


class DreamXInventoryTest(unittest.TestCase):
    def test_output_paths_are_precision_explicit(self):
        root = Path("out")
        self.assertEqual(
            output_paths(root, "q8_0"),
            (
                root / "DreamX-Creator-Q8_0.gguf",
                root / "mmproj-DreamX-Creator-BF16.gguf",
            ),
        )
        self.assertEqual(
            output_paths(root, "bf16"),
            (
                root / "DreamX-Creator-BF16.gguf",
                root / "mmproj-DreamX-Creator-BF16.gguf",
            ),
        )

    def test_inventory_requires_every_released_component(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_minimal_layout(root, omit="refiner/lightvae_nu_scheme3.pt")
            with self.assertRaisesRegex(FileNotFoundError, "lightvae_nu_scheme3.pt"):
                build_inventory(root)

    def test_inventory_accepts_only_released_dimensions_and_joint_layers(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_minimal_layout(root)
            inventory = build_inventory(root)
            self.assertEqual(inventory.video_config["dim"], 3072)
            self.assertEqual(inventory.audio_config["dim"], 1536)
            self.assertEqual(JOINT_LAYERS, tuple(range(15, 30)))

            config_path = root / "creator/audio_model/config.json"
            bad = dict(AUDIO_CONFIG, ffn_dim=8192)
            config_path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "audio ffn_dim"):
                build_inventory(root)

    def test_pair_id_is_canonical(self):
        left = pair_id({"video": {"count": 825, "bytes": 20}})
        right = pair_id({"video": {"bytes": 20, "count": 825}})
        self.assertEqual(left, right)
        self.assertEqual(len(left), 64)

    def test_pt_loader_uses_bounded_cpu_options_and_unwraps_ema(self):
        tensor = object()
        torch = Mock()
        torch.load.return_value = {"ema": {"weight": tensor}}
        torch.is_tensor.side_effect = lambda value: value is tensor

        state = load_pt_state_dict(Path("weights.pt"), torch)

        self.assertEqual(state, {"weight": tensor})
        torch.load.assert_called_once_with(
            Path("weights.pt"),
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )

    def test_pt_loader_rejects_non_tensor_entries(self):
        torch = Mock()
        torch.load.return_value = {"weight": object(), "epoch": 4}
        torch.is_tensor.side_effect = lambda value: not isinstance(value, int)
        with self.assertRaisesRegex(ValueError, "non-tensor.*epoch"):
            load_pt_state_dict(Path("weights.pt"), torch)

    def test_pt_loader_unwraps_released_sr_dit_generator(self):
        tensor = object()
        torch = Mock()
        torch.load.return_value = {"generator": {"weight": tensor}}
        torch.is_tensor.side_effect = lambda value: value is tensor
        self.assertEqual(
            load_pt_state_dict(Path("sr_dit_5b.pt"), torch),
            {"weight": tensor},
        )

    def test_tensor_names_keep_source_suffix_under_stable_prefix(self):
        self.assertEqual(
            map_name("creator.video", "blocks.0.self_attn.q.weight"),
            "dreamx.creator.video.blocks.0.self_attn.q.weight",
        )

    def test_only_large_main_linear_weights_are_quantized(self):
        self.assertTrue(
            should_quantize(
                "dreamx.creator.video.blocks.0.ffn.0.weight",
                (14336, 3072),
            )
        )
        self.assertFalse(
            should_quantize(
                "dreamx.video_vae.decoder.conv1.weight",
                (160, 48, 3, 3, 3),
            )
        )
        self.assertFalse(
            should_quantize(
                "dreamx.creator.joint.joint_blocks.15.gate_hidden.weight",
                (12, 1536),
            )
        )

    def test_f32_to_bf16_uses_round_to_nearest_even(self):
        raw = struct.pack("<II", 0x3F808000, 0x3F818000)
        self.assertEqual(
            struct.unpack("<HH", f32_to_bf16_rne(raw)),
            (0x3F80, 0x3F82),
        )

    def test_tensor_plan_covers_each_source_once(self):
        sources = [
            SourceTensor(
                "creator.video",
                "blocks.0.ffn.0.weight",
                "F32",
                (32, 32),
                lambda: iter((bytes(32 * 32 * 4),)),
            ),
            SourceTensor(
                "text",
                "token_embedding.weight",
                "BF16",
                (2, 32),
                lambda: iter((bytes(2 * 32 * 2),)),
            ),
        ]

        plan = build_tensor_plan(sources, "q8_0")

        self.assertEqual(
            sorted((entry.source.component, entry.source.name) for entry in plan),
            sorted((source.component, source.name) for source in sources),
        )
        self.assertEqual(
            {entry.target for entry in plan},
            {"main", "mmproj"},
        )
        self.assertEqual(len({entry.output_name for entry in plan}), len(sources))

    def test_pair_publish_removes_first_target_when_second_link_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            main_tmp, aux_tmp = root / "main.tmp", root / "aux.tmp"
            main_out, aux_out = root / "main.gguf", root / "aux.gguf"
            main_tmp.write_bytes(b"main")
            aux_tmp.write_bytes(b"aux")
            real_link = os.link
            calls = 0

            def fail_second(source, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("second publish failed")
                real_link(source, target)

            with patch("tools.dreamx.convert_dreamx_creator.os.link", fail_second):
                with self.assertRaisesRegex(OSError, "second publish failed"):
                    _publish_pair(main_tmp, aux_tmp, main_out, aux_out, False)

            self.assertFalse(main_out.exists())
            self.assertFalse(aux_out.exists())

    def test_pair_overwrite_restores_first_file_when_second_backup_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            main_tmp, aux_tmp = root / "main.tmp", root / "aux.tmp"
            main_out, aux_out = root / "main.gguf", root / "aux.gguf"
            main_tmp.write_bytes(b"new-main")
            aux_tmp.write_bytes(b"new-aux")
            main_out.write_bytes(b"old-main")
            aux_out.write_bytes(b"old-aux")
            real_replace = os.replace
            calls = 0

            def fail_second(source, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("second backup failed")
                real_replace(source, target)

            with patch("tools.dreamx.convert_dreamx_creator.os.replace", fail_second):
                with self.assertRaisesRegex(OSError, "second backup failed"):
                    _publish_pair(main_tmp, aux_tmp, main_out, aux_out, True)

            self.assertEqual(main_out.read_bytes(), b"old-main")
            self.assertEqual(aux_out.read_bytes(), b"old-aux")

    def test_safetensor_source_payload_is_read_in_fresh_chunks(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "weights.safetensors"
            write_safetensors(path, ("b.weight", "a.bias"))

            sources = read_safetensor_sources("creator.audio", (path,), chunk_bytes=3)

            self.assertEqual([source.name for source in sources], ["a.bias", "b.weight"])
            self.assertEqual(sources[0].shape, (1,))
            self.assertEqual(sources[0].dtype, "F32")
            self.assertEqual(b"".join(sources[0].chunks()), bytes(4))
            self.assertEqual(b"".join(sources[0].chunks()), bytes(4))

    def test_pt_checkpoint_is_spooled_as_sorted_safetensors(self):
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "spool.safetensors"
            first, second = object(), object()
            torch = Mock()
            torch.load.return_value = {"model": {"b": second, "a": first}}
            torch.is_tensor.return_value = True
            saved = {}

            def save_file(state, path):
                saved.update(state)
                Path(path).write_bytes(b"spooled")

            count = _spool_pt_checkpoint(
                Path("weights.pt"),
                out,
                torch_module=torch,
                save_file=save_file,
            )

            self.assertEqual(count, 2)
            self.assertEqual(list(saved), ["a", "b"])
            self.assertEqual(out.read_bytes(), b"spooled")

    def test_pair_writer_routes_metadata_and_payloads(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            main, mmproj = root / "main.gguf", root / "mmproj.gguf"
            sources = [
                SourceTensor(
                    "creator.joint",
                    "gate.bias",
                    "F32",
                    (1,),
                    lambda: iter((struct.pack("<f", 1.0),)),
                ),
                SourceTensor(
                    "text",
                    "norm.weight",
                    "BF16",
                    (1,),
                    lambda: iter((struct.pack("<H", 0x3F80),)),
                ),
            ]
            plan = build_tensor_plan(sources, "q8_0")

            _write_pair_files(
                main,
                mmproj,
                plan,
                {
                    "main": [("general.architecture", "dreamx"), ("dreamx.pair_id", "a" * 64)],
                    "mmproj": [("general.architecture", "clip"), ("dreamx.pair_id", "a" * 64)],
                },
            )

            main_meta, main_tensors = read_gguf_directory(main)
            aux_meta, aux_tensors = read_gguf_directory(mmproj)
            self.assertEqual(main_meta["general.architecture"], "dreamx")
            self.assertEqual(aux_meta["general.architecture"], "clip")
            self.assertEqual(main_meta["dreamx.pair_id"], aux_meta["dreamx.pair_id"])
            self.assertEqual(list(main_tensors), ["dreamx.creator.joint.gate.bias"])
            self.assertEqual(list(aux_tensors), ["dreamx.text.norm.weight"])
            self.assertEqual(
                read_gguf_tensor_bytes(main, "dreamx.creator.joint.gate.bias"),
                struct.pack("<f", 1.0),
            )

    def test_source_collection_includes_all_released_components(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "model"
            spool = Path(raw) / "spool"
            spool.mkdir()
            make_minimal_layout(root)
            inventory = build_inventory(root)

            def fake_spool(_source, destination):
                write_safetensors(destination)
                return 1

            sources = collect_source_tensors(
                inventory,
                spool,
                spool_checkpoint=fake_spool,
            )

            self.assertEqual(
                {source.component for source in sources},
                set(
                    (
                        "creator.video",
                        "creator.audio",
                        "creator.joint",
                        "refiner.dit",
                        "text",
                        "video_vae",
                        "audio_vae",
                        "refiner.upsampler.flash",
                        "refiner.upsampler.causal2d",
                        "refiner.lightvae",
                    )
                ),
            )

    def test_pair_metadata_is_shared_and_role_specific(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            make_minimal_layout(root)
            inventory = build_inventory(root)
            counts = {name: index + 1 for index, name in enumerate(sorted((
                "creator.video", "creator.audio", "creator.joint", "refiner.dit",
                "text", "video_vae", "audio_vae", "refiner.upsampler.flash",
                "refiner.upsampler.causal2d", "refiner.lightvae",
            )))}
            metadata = build_pair_metadata(
                inventory,
                "q8_0",
                "b" * 64,
                counts,
                {"exporter_version": 1},
            )
            main = dict(metadata["main"])
            mmproj = dict(metadata["mmproj"])

            self.assertEqual(main["general.architecture"], "dreamx")
            self.assertEqual(mmproj["general.architecture"], "clip")
            self.assertEqual(mmproj["clip.projector_type"], "dreamx_creator")
            self.assertEqual(main["dreamx.pair_id"], mmproj["dreamx.pair_id"])
            self.assertEqual(main["dreamx.components"], mmproj["dreamx.components"])
            self.assertEqual(main["dreamx.video.embedding_length"], 3072)
            self.assertEqual(main["dreamx.audio.embedding_length"], 1536)
            self.assertIn("dreamx.tokenizer.json", mmproj)

    def test_export_model_writes_and_publishes_a_matched_pair(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "model"
            out = Path(raw) / "out"
            make_minimal_layout(root)
            sources = []
            for component in (
                "creator.video",
                "creator.audio",
                "creator.joint",
                "refiner.dit",
            ):
                sources.append(
                    SourceTensor(
                        component,
                        "test.bias",
                        "F32",
                        (1,),
                        lambda: iter((struct.pack("<f", 1.0),)),
                    )
                )
            for component in (
                "text",
                "video_vae",
                "audio_vae",
                "refiner.upsampler.flash",
                "refiner.upsampler.causal2d",
                "refiner.lightvae",
            ):
                sources.append(
                    SourceTensor(
                        component,
                        "test.weight",
                        "BF16",
                        (1,),
                        lambda: iter((struct.pack("<H", 0x3F80),)),
                    )
                )

            with patch(
                "tools.dreamx.convert_dreamx_creator.collect_source_tensors",
                return_value=sources,
            ):
                main, mmproj = export_model(root, out, "q8_0", False)

            main_meta, main_tensors = read_gguf_directory(main)
            aux_meta, aux_tensors = read_gguf_directory(mmproj)
            self.assertEqual(main_meta["dreamx.pair_id"], aux_meta["dreamx.pair_id"])
            self.assertEqual(len(main_tensors), 4)
            self.assertEqual(len(aux_tensors), 6)


if __name__ == "__main__":
    unittest.main()
