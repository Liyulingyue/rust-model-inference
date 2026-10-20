import io
import json
import os
import pickle
import struct
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

from tools.converter.bge.convert_bge import TensorStore, convert
from tools.converter.utils.gguf import read_gguf_directory, read_gguf_tensor_bytes


def _write_safetensors(
    path: Path, tensors: dict[str, tuple[tuple[int, ...], bytes]]
) -> None:
    header = {}
    payload = bytearray()
    for name, (shape, raw) in tensors.items():
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _rewrite_safetensors(
    path: Path, *, remove: str | None = None, extra: str | None = None
) -> None:
    raw = path.read_bytes()
    header_len = struct.unpack_from("<Q", raw)[0]
    header = json.loads(raw[8 : 8 + header_len])
    payload = raw[8 + header_len :]
    tensors = {}
    for name, info in header.items():
        if name == remove:
            continue
        start, end = info["data_offsets"]
        tensors[name] = (tuple(info["shape"]), payload[start:end])
    if extra is not None:
        tensors[extra] = ((32,), np.zeros(32, dtype=np.float32).tobytes())
    _write_safetensors(path, tensors)


class _Storage:
    def __init__(self, key: str, raw: bytes, dtype: str):
        self.key = key
        self.raw = raw
        self.dtype = dtype


class _Tensor:
    def __init__(
        self, storage: _Storage, shape: tuple[int, ...], stride: tuple[int, ...]
    ):
        self.storage = storage
        self.shape = shape
        self.stride = stride

    def __reduce__(self):
        return (
            _rebuild_tensor_v2,
            (self.storage, 0, self.shape, self.stride, False, {}),
        )


def _rebuild_tensor_v2(storage, offset, shape, stride, *_rest):
    return _Tensor(storage, tuple(shape), tuple(stride))


_rebuild_tensor_v2.__module__ = "torch._utils"
_rebuild_tensor_v2.__name__ = "_rebuild_tensor_v2"
_rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"


class _FloatStorage:
    __module__ = "torch"


class _LongStorage:
    __module__ = "torch"


_FloatStorage.__name__ = "FloatStorage"
_LongStorage.__name__ = "LongStorage"
_FloatStorage.__qualname__ = "FloatStorage"
_LongStorage.__qualname__ = "LongStorage"


def _write_torch_bin(path: Path, state: dict[str, _Tensor]) -> None:
    torch = types.ModuleType("torch")
    torch.FloatStorage = _FloatStorage
    torch.LongStorage = _LongStorage
    utils = types.ModuleType("torch._utils")
    utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    old_torch, old_utils = sys.modules.get("torch"), sys.modules.get("torch._utils")
    sys.modules["torch"], sys.modules["torch._utils"] = torch, utils
    try:

        class Pickler(pickle.Pickler):
            def persistent_id(self, value):
                if isinstance(value, _Storage):
                    storage_type = (
                        _FloatStorage if value.dtype == "F32" else _LongStorage
                    )
                    return (
                        "storage",
                        storage_type,
                        value.key,
                        "cpu",
                        len(value.raw) // (4 if value.dtype == "F32" else 8),
                    )
                return None

        payload = io.BytesIO()
        Pickler(payload, protocol=2).dump(state)
    finally:
        if old_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = old_torch
        if old_utils is None:
            sys.modules.pop("torch._utils", None)
        else:
            sys.modules["torch._utils"] = old_utils

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", payload.getvalue())
        for tensor in state.values():
            archive.writestr(f"archive/data/{tensor.storage.key}", tensor.storage.raw)
        archive.writestr("archive/version", "3\n")


def _decoder_files(
    model_dir: Path,
    *,
    torch_bin: bool = False,
    model_type: str = "qwen2",
    hidden: int = 32,
) -> dict[str, bytes]:
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "hidden_size": hidden,
                "intermediate_size": hidden * 2,
                "num_hidden_layers": 2 if model_type == "gemma2" else 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 16,
                "vocab_size": 4,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "head_dim": hidden // 4,
                "query_pre_attn_scalar": hidden // 4,
                "sliding_window": 4 if model_type in {"gemma2", "mistral"} else None,
            }
        )
    )
    (model_dir / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "b": 1, "c": 2, "d": 3},
                    "merges": [],
                },
                "added_tokens": [],
            }
        )
    )
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"add_bos_token": False, "add_eos_token": False})
    )
    shapes = {
        "model.embed_tokens.weight": (4, 32),
        "model.norm.weight": (32,),
        "model.layers.0.input_layernorm.weight": (32,),
        "model.layers.0.self_attn.q_proj.weight": (32, 32),
        "model.layers.0.self_attn.k_proj.weight": (16, 32),
        "model.layers.0.self_attn.v_proj.weight": (16, 32),
        "model.layers.0.self_attn.o_proj.weight": (32, 32),
        "model.layers.0.self_attn.q_proj.bias": (32,),
        "model.layers.0.self_attn.k_proj.bias": (16,),
        "model.layers.0.self_attn.v_proj.bias": (16,),
        "model.layers.0.mlp.gate_proj.weight": (64, 32),
        "model.layers.0.mlp.up_proj.weight": (64, 32),
        "model.layers.0.mlp.down_proj.weight": (32, 64),
        "model.layers.0.post_attention_layernorm.weight": (32,),
        "lm_head.weight": (4, 32),
        "model.layers.0.self_attn.rotary_emb.inv_freq": (4,),
    }
    if model_type != "qwen2":
        for name in list(shapes):
            if name.endswith(".bias"):
                del shapes[name]
    if model_type == "qwen3":
        shapes["model.layers.0.self_attn.q_norm.weight"] = (8,)
        shapes["model.layers.0.self_attn.k_norm.weight"] = (8,)
    if model_type == "gemma2":
        shapes["model.layers.0.pre_feedforward_layernorm.weight"] = (32,)
        shapes["model.layers.0.post_feedforward_layernorm.weight"] = (32,)
        shapes.update(
            {
                name.replace("layers.0.", "layers.1."): shape
                for name, shape in list(shapes.items())
                if "layers.0." in name
            }
        )
    sizes = {32: hidden, 64: hidden * 2, 16: hidden // 2, 8: hidden // 4}
    shapes = {
        name: tuple(sizes.get(size, size) for size in shape)
        for name, shape in shapes.items()
    }
    raws = {
        name: np.arange(np.prod(shape), dtype=np.float32).reshape(shape).tobytes()
        for name, shape in shapes.items()
    }
    for name, shape in shapes.items():
        if "norm.weight" in name:
            raws[name] = np.ones(shape, dtype=np.float32).tobytes()
        elif hidden > 32:
            raws[name] = (
                np.frombuffer(raws[name], dtype=np.float32) * np.float32(1e-5)
            ).tobytes()
    if torch_bin:
        state = {}
        for index, (name, raw) in enumerate(raws.items()):
            shape = shapes[name]
            state[f"module.{name}"] = _Tensor(
                _Storage(str(index), raw, "F32"),
                shape,
                tuple(int(np.prod(shape[index + 1 :])) for index in range(len(shape))),
            )
        _write_torch_bin(model_dir / "pytorch_model.bin", state)
    else:
        _write_safetensors(
            model_dir / "model.safetensors",
            {
                name.removeprefix("model."): (shape, raws[name])
                for name, shape in shapes.items()
            },
        )
    return raws


def _bert_files(model_dir: Path) -> dict[str, bytes]:
    model_dir.mkdir()
    vocab = [f"v{index}" for index in range(100)]
    vocab.extend(["[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "max_position_embeddings": 4,
                "vocab_size": len(vocab),
                "type_vocab_size": 2,
                "layer_norm_eps": 1e-12,
            }
        )
    )
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "unk_token": "[UNK]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "pad_token": "v0",
                "mask_token": "[MASK]",
            }
        )
    )
    (model_dir / "vocab.txt").write_text("\n".join(vocab) + "\n")
    shapes = {
        "embeddings.word_embeddings.weight": (len(vocab), 4),
        "embeddings.position_embeddings.weight": (4, 4),
        "embeddings.LayerNorm.weight": (4,),
        "embeddings.LayerNorm.bias": (4,),
        "embeddings.token_type_embeddings.weight": (2, 4),
        "encoder.layer.0.attention.self.query.weight": (4, 4),
        "encoder.layer.0.attention.self.query.bias": (4,),
        "encoder.layer.0.attention.self.key.weight": (4, 4),
        "encoder.layer.0.attention.self.key.bias": (4,),
        "encoder.layer.0.attention.self.value.weight": (4, 4),
        "encoder.layer.0.attention.self.value.bias": (4,),
        "encoder.layer.0.attention.output.dense.weight": (4, 4),
        "encoder.layer.0.attention.output.dense.bias": (4,),
        "encoder.layer.0.attention.output.LayerNorm.weight": (4,),
        "encoder.layer.0.attention.output.LayerNorm.bias": (4,),
        "encoder.layer.0.intermediate.dense.weight": (8, 4),
        "encoder.layer.0.intermediate.dense.bias": (8,),
        "encoder.layer.0.output.dense.weight": (4, 8),
        "encoder.layer.0.output.dense.bias": (4,),
        "encoder.layer.0.output.LayerNorm.weight": (4,),
        "encoder.layer.0.output.LayerNorm.bias": (4,),
    }
    raws = {
        name: np.arange(np.prod(shape), dtype=np.float32).reshape(shape).tobytes()
        for name, shape in shapes.items()
    }
    for name in (
        "embeddings.LayerNorm.weight",
        "embeddings.token_type_embeddings.weight",
        "encoder.layer.0.attention.output.LayerNorm.weight",
        "encoder.layer.0.output.LayerNorm.weight",
    ):
        raws[name] = np.ones(shapes[name], dtype=np.float32).tobytes()
    _write_safetensors(
        model_dir / "model.safetensors",
        {name: (shape, raws[name]) for name, shape in shapes.items()},
    )
    return raws


def _xlm_roberta_files(model_dir: Path) -> dict[str, bytes]:
    model_dir.mkdir()
    vocab = [["<s>", 0.0], ["<pad>", 0.0], ["</s>", 0.0], ["<unk>", 0.0], ["▁a", -1.0]]
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "xlm-roberta",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "max_position_embeddings": 6,
                "vocab_size": len(vocab),
                "type_vocab_size": 1,
                "pad_token_id": 1,
                "layer_norm_eps": 1e-5,
            }
        )
    )
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "cls_token": "<s>",
                "sep_token": "</s>",
                "mask_token": "<mask>",
            }
        )
    )
    (model_dir / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "Unigram", "vocab": vocab}, "added_tokens": []})
    )
    shapes = {
        "embeddings.word_embeddings.weight": (len(vocab), 4),
        "embeddings.position_embeddings.weight": (6, 4),
        "embeddings.LayerNorm.weight": (4,),
        "embeddings.LayerNorm.bias": (4,),
        "embeddings.token_type_embeddings.weight": (1, 4),
        "encoder.layer.0.attention.self.query.weight": (4, 4),
        "encoder.layer.0.attention.self.query.bias": (4,),
        "encoder.layer.0.attention.self.key.weight": (4, 4),
        "encoder.layer.0.attention.self.key.bias": (4,),
        "encoder.layer.0.attention.self.value.weight": (4, 4),
        "encoder.layer.0.attention.self.value.bias": (4,),
        "encoder.layer.0.attention.output.dense.weight": (4, 4),
        "encoder.layer.0.attention.output.dense.bias": (4,),
        "encoder.layer.0.attention.output.LayerNorm.weight": (4,),
        "encoder.layer.0.attention.output.LayerNorm.bias": (4,),
        "encoder.layer.0.intermediate.dense.weight": (8, 4),
        "encoder.layer.0.intermediate.dense.bias": (8,),
        "encoder.layer.0.output.dense.weight": (4, 8),
        "encoder.layer.0.output.dense.bias": (4,),
        "encoder.layer.0.output.LayerNorm.weight": (4,),
        "encoder.layer.0.output.LayerNorm.bias": (4,),
    }
    raws = {
        name: np.arange(np.prod(shape), dtype=np.float32).reshape(shape).tobytes()
        for name, shape in shapes.items()
    }
    for name in (
        "embeddings.LayerNorm.weight",
        "embeddings.token_type_embeddings.weight",
        "encoder.layer.0.attention.output.LayerNorm.weight",
        "encoder.layer.0.output.LayerNorm.weight",
    ):
        raws[name] = np.ones(shapes[name], dtype=np.float32).tobytes()
    _write_safetensors(
        model_dir / "model.safetensors",
        {name: (shape, raws[name]) for name, shape in shapes.items()},
    )
    return raws


class BgeConverterTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("BGE_RUST_BIN"), "set BGE_RUST_BIN to run decoder CLI fixtures"
    )
    def test_decoder_cli_f32_and_q8(self):
        with tempfile.TemporaryDirectory() as directory:
            for model_type in ("mistral", "qwen2", "qwen3", "gemma2"):
                model = Path(directory) / model_type
                _decoder_files(model, model_type=model_type, hidden=256)
                for quant in ("F32", "Q8_0"):
                    with self.subTest(model_type=model_type, quant=quant):
                        output = Path(directory) / f"{model_type}-{quant}.gguf"
                        convert(model, output, quant)
                        result = subprocess.run(
                            [
                                os.environ["BGE_RUST_BIN"],
                                "--model",
                                str(output),
                                "--prompt",
                                "abca",
                                "--embedding",
                                "--embedding-output",
                                "raw",
                                "--threads",
                                "1",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        line = next(
                            line
                            for line in result.stdout.splitlines()
                            if line.startswith("embedding_raw:")
                        )
                        values = np.array([float(value) for value in line.split()[1:]])
                        self.assertEqual(values.size, 256)
                        self.assertTrue(np.isfinite(values).all())
                        self.assertAlmostEqual(
                            float(np.linalg.norm(values)), 1.0, places=5
                        )

    def test_bert_safetensors_f32_maps_wordpiece_and_absolute_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            _bert_files(model)
            output = Path(directory) / "model.gguf"
            convert(model, output, "F32")

            metadata, tensors = read_gguf_directory(output)
            self.assertEqual(metadata["general.architecture"], "bert")
            self.assertEqual(metadata["bert.pooling_type"], 2)
            self.assertEqual(metadata["tokenizer.ggml.model"], "bert")
            self.assertEqual(metadata["tokenizer.ggml.cls_token_id"], 101)
            self.assertEqual(metadata["tokenizer.ggml.seperator_token_id"], 102)
            self.assertEqual(metadata["tokenizer.ggml.tokens"][0], "▁v0")
            self.assertEqual(tensors["position_embd.weight"][1], (4, 4))
            self.assertEqual(tensors["token_types.weight"][1], (4, 2))

    def test_bert_respects_case_and_accent_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            _bert_files(model)
            path = model / "tokenizer_config.json"
            config = json.loads(path.read_text())
            for lowercase, strip_accents in (
                (False, None),
                (True, None),
                (False, True),
            ):
                with self.subTest(lowercase=lowercase, strip_accents=strip_accents):
                    config.update(do_lower_case=lowercase, strip_accents=strip_accents)
                    path.write_text(json.dumps(config))
                    output = Path(directory) / f"{lowercase}-{strip_accents}.gguf"
                    convert(model, output)
                    metadata, _ = read_gguf_directory(output)
                    self.assertEqual(
                        metadata["tokenizer.ggml.normalizer.lowercase"], lowercase
                    )
                    expected_accents = (
                        lowercase if strip_accents is None else strip_accents
                    )
                    self.assertEqual(
                        metadata["tokenizer.ggml.normalizer.strip_accents"],
                        expected_accents,
                    )

    def test_xlm_roberta_trims_reserved_position_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            raws = _xlm_roberta_files(model)
            output = Path(directory) / "model.gguf"
            convert(model, output, "F32")

            metadata, tensors = read_gguf_directory(output)
            self.assertEqual(metadata["bert.context_length"], 4)
            self.assertEqual(metadata["tokenizer.ggml.model"], "t5")
            self.assertEqual(metadata["tokenizer.ggml.padding_token_id"], 1)
            self.assertEqual(metadata["tokenizer.ggml.token_type"][:4], [3, 3, 3, 2])
            self.assertFalse(
                metadata.get("tokenizer.ggml.treat_whitespace_as_suffix", False)
            )
            self.assertEqual(tensors["position_embd.weight"][1], (4, 4))
            self.assertEqual(
                read_gguf_tensor_bytes(output, "position_embd.weight"),
                raws["embeddings.position_embeddings.weight"][2 * 4 * 4 :],
            )

    def test_decoder_safetensors_f32_and_q8(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            raws = _decoder_files(model)
            f32 = Path(directory) / "model-f32.gguf"
            q8 = Path(directory) / "model-q8.gguf"
            convert(model, f32, "F32")
            convert(model, q8, "Q8_0")

            metadata, tensors = read_gguf_directory(f32)
            self.assertEqual(metadata["general.architecture"], "qwen2")
            self.assertEqual(metadata["tokenizer.ggml.model"], "gpt2")
            self.assertEqual(metadata["tokenizer.ggml.tokens"], ["a", "b", "c", "d"])
            self.assertEqual(tensors["token_embd.weight"][0], 0)
            self.assertEqual(
                read_gguf_tensor_bytes(f32, "token_embd.weight"),
                raws["model.embed_tokens.weight"],
            )
            self.assertEqual(
                read_gguf_tensor_bytes(f32, "blk.0.attn_q.bias"),
                raws["model.layers.0.self_attn.q_proj.bias"],
            )
            _, q8_tensors = read_gguf_directory(q8)
            self.assertEqual(q8_tensors["token_embd.weight"][0], 8)
            self.assertEqual(q8_tensors["blk.0.attn_norm.weight"][0], 0)

    def test_torch_bin_uses_storage_bytes_and_strips_module_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            raws = _decoder_files(model, torch_bin=True)
            output = Path(directory) / "model.gguf"
            convert(model, output, "F32")
            self.assertEqual(
                read_gguf_tensor_bytes(output, "token_embd.weight"),
                raws["model.embed_tokens.weight"],
            )
            names = TensorStore(model).names()
            self.assertIn("model.embed_tokens.weight", names)
            self.assertNotIn("module.model.embed_tokens.weight", names)

    def test_torch_bin_nested_prefix_and_auxiliary_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            storage = _Storage("0", np.ones(4, dtype=np.float32).tobytes(), "F32")
            storage_position = _Storage(
                "1", np.arange(2, dtype=np.float32).tobytes(), "F32"
            )
            storage_head = _Storage("2", np.zeros(2, dtype=np.float32).tobytes(), "F32")
            state = {
                "module.roberta.embeddings.word_embeddings.weight": _Tensor(
                    storage, (2, 2), (2, 1)
                ),
                "module.roberta.embeddings.position_ids": _Tensor(
                    storage_position, (2,), (1,)
                ),
                "module.roberta.lm_head.bias": _Tensor(storage_head, (2,), (1,)),
            }
            _write_torch_bin(model / "pytorch_model.bin", state)
            names = TensorStore(model).names()
            self.assertEqual(names, {"embeddings.word_embeddings.weight"})

    def test_torch_bin_accepts_ordered_state_dict(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            raw = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tobytes()
            state = OrderedDict(
                {
                    "embeddings.word_embeddings.weight": _Tensor(
                        _Storage("0", raw, "F32"), (2, 2), (2, 1)
                    ),
                }
            )
            _write_torch_bin(model / "pytorch_model.bin", state)
            store = TensorStore(model)
            self.assertEqual(
                store.read(store["embeddings.word_embeddings.weight"]), raw
            )

    def test_decoder_architecture_transforms(self):
        with tempfile.TemporaryDirectory() as directory:
            for model_type, arch in (
                ("mistral", "llama"),
                ("qwen3", "qwen3"),
                ("gemma2", "gemma2"),
            ):
                with self.subTest(model_type=model_type):
                    model = Path(directory) / model_type
                    _decoder_files(model, model_type=model_type)
                    output = Path(directory) / f"{model_type}.gguf"
                    convert(model, output, "F32")
                    meta, tensors = read_gguf_directory(output)
                    self.assertEqual(meta["general.architecture"], arch)
                    self.assertEqual(meta[f"{arch}.rope.dimension_count"], 8)
                    if model_type == "mistral":
                        query = np.frombuffer(
                            read_gguf_tensor_bytes(output, "blk.0.attn_q.weight"),
                            dtype="<f4",
                        )
                        self.assertEqual(
                            query.reshape(32, 32)[:8, 0].tolist(),
                            [0, 128, 32, 160, 64, 192, 96, 224],
                        )
                        self.assertEqual(meta["llama.attention.sliding_window"], 4)
                    elif model_type == "gemma2":
                        self.assertNotIn("output.weight", tensors)
                        norm = np.frombuffer(
                            read_gguf_tensor_bytes(output, "output_norm.weight"),
                            dtype="<f4",
                        )
                        self.assertEqual(norm.tolist(), [2.0] * 32)
                        for target in (
                            "attn_norm",
                            "post_attention_norm",
                            "ffn_norm",
                            "post_ffw_norm",
                        ):
                            raw = read_gguf_tensor_bytes(
                                output, f"blk.0.{target}.weight"
                            )
                            self.assertEqual(
                                np.frombuffer(raw, dtype="<f4").tolist(), [2.0] * 32
                            )
                        self.assertAlmostEqual(
                            meta["gemma2.embedding_scale"], 32**0.5, places=5
                        )
                        self.assertAlmostEqual(
                            meta["gemma2.attention.scale"],
                            1.0 / (32 // 4) ** 0.5,
                            places=6,
                        )
                        self.assertEqual(
                            meta["gemma2.attention.sliding_window_pattern"],
                            [True, False],
                        )

    def test_encoder_wrong_shape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            _bert_files(model)
            path = model / "model.safetensors"
            raw = path.read_bytes()
            size = struct.unpack_from("<Q", raw)[0]
            header = json.loads(raw[8 : 8 + size])
            header["encoder.layer.0.attention.self.query.weight"]["shape"] = [2, 8]
            encoded = json.dumps(header).encode()
            path.write_bytes(
                struct.pack("<Q", len(encoded)) + encoded + raw[8 + size :]
            )
            output = Path(directory) / "model.gguf"
            with self.assertRaisesRegex(ValueError, "shape"):
                convert(model, output)
            self.assertFalse(output.exists())

    def test_decoder_wrong_shape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            _decoder_files(model)
            path = model / "model.safetensors"
            raw = path.read_bytes()
            size = struct.unpack_from("<Q", raw)[0]
            header = json.loads(raw[8 : 8 + size])
            header["layers.0.self_attn.q_proj.weight"]["shape"] = [16, 64]
            encoded = json.dumps(header).encode()
            path.write_bytes(
                struct.pack("<Q", len(encoded)) + encoded + raw[8 + size :]
            )
            with self.assertRaisesRegex(ValueError, "shape"):
                convert(model, Path(directory) / "model.gguf")

    def test_unknown_or_missing_decoder_tensor_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for remove, extra, message in [
                ("norm.weight", None, "missing required decoder tensors"),
                (None, "unexpected.weight", "unsupported BGE decoder tensors"),
            ]:
                model = Path(directory) / ("missing" if remove else "unknown")
                _decoder_files(model)
                _rewrite_safetensors(
                    model / "model.safetensors", remove=remove, extra=extra
                )
                with self.subTest(remove=remove, extra=extra), self.assertRaisesRegex(
                    ValueError, message
                ):
                    convert(model, Path(directory) / f"{model.name}.gguf")


if __name__ == "__main__":
    unittest.main()
