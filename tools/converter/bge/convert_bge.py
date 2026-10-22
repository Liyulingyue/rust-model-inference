"""Convert BAAI BGE checkpoints to self-contained GGUF.

The converter covers the BERT/XLM-R encoders and the decoder models currently
published under ``BAAI/bge-*`` (Mistral, Qwen2/Qwen3, and Gemma2).  The
emitted tensors use the names consumed by the corresponding Rust model paths
and can be kept as F32 or quantized to Q8_0.

Usage::

    .venv/bin/python -m tools.converter.bge.convert_bge models/BAAI-bge-small-en-v1-5 \
        models/bge-small-en-v1.5-F32.gguf --quant F32
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import pickle
import re
import struct
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from tools.converter.utils.gguf import (
    GGML_F32,
    GGML_Q8_0,
    GgufWriter,
    MetaArray,
    MetaScalar,
    bf16_to_f32,
    f16_to_f32,
    gguf_dims,
    open_safetensors,
    quantize_q8_0,
)


def _add_meta(writer: GgufWriter, key: str, value: object) -> None:
    """Write BGE metadata with the scalar types required by llama.cpp."""
    if isinstance(value, int) and not isinstance(value, bool):
        value = MetaScalar(4, value)  # GGUF_UINT32
    elif isinstance(value, float):
        value = MetaScalar(6, value)  # GGUF_FLOAT32
    writer.add_meta(key, value)


_DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2}
_TORCH_DTYPE_BYTES = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I8": 1,
    "U8": 1,
}
_LAYER = re.compile(r"^encoder\.layer\.(\d+)\.(.+)$")
_DROP = {
    "embeddings.position_ids",
    "pooler.dense.weight",
    "pooler.dense.bias",
    # XLM-Retromae publishes the masked-LM head together with the encoder.
    # It is not part of the embedding graph and is deliberately not copied.
    "lm_head.bias",
    "lm_head.dense.weight",
    "lm_head.dense.bias",
    "lm_head.layer_norm.weight",
    "lm_head.layer_norm.bias",
    "lm_head.decoder.weight",
    "lm_head.decoder.bias",
}


@dataclass(frozen=True)
class TokenizerSpec:
    kind: str
    tokens: list[str]
    scores: list[float]
    special_names: dict[str, str]
    charsmap: bytes | None
    merges: list[str]
    token_types: list[int] | None
    add_bos: bool
    add_eos: bool
    add_space_prefix: bool = True


@dataclass(frozen=True)
class SourceTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    path: Path
    start: int
    end: int
    data_offset: int
    raw: bytes | None = None

    @property
    def nbytes(self) -> int:
        return len(self.raw) if self.raw is not None else self.end - self.start


@dataclass(frozen=True)
class _TorchStorageRef:
    dtype: str
    key: str
    raw: bytes
    numel: int


@dataclass(frozen=True)
class _TorchTensorRef:
    storage: _TorchStorageRef
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


class _TorchStorageMarker:
    """Pickle placeholder for torch storage classes when torch is absent."""

    dtype: str


def _torch_dtype(name: str) -> str:
    return {
        "FloatStorage": "F32",
        "HalfStorage": "F16",
        "BFloat16Storage": "BF16",
        "LongStorage": "I64",
        "IntStorage": "I32",
        "CharStorage": "I8",
        "ByteStorage": "U8",
    }.get(name, "")


class _TorchUnpickler(pickle.Unpickler):
    """Read the tensor-only part of a torch.save ZIP without importing torch."""

    def __init__(self, payload: bytes, archive: zipfile.ZipFile, prefix: str):
        super().__init__(io.BytesIO(payload))
        self.archive = archive
        self.prefix = prefix
        self.storages: dict[str, _TorchStorageRef] = {}

    def persistent_load(self, pid):  # noqa: ANN001
        if not isinstance(pid, tuple) or len(pid) < 5 or pid[0] != "storage":
            raise ValueError(f"unsupported PyTorch persistent id: {pid!r}")
        storage_type, key, numel = pid[1], str(pid[2]), int(pid[4])
        dtype = getattr(storage_type, "dtype", "")
        if not dtype:
            raise ValueError(f"unsupported PyTorch storage type: {storage_type!r}")
        cached = self.storages.get(key)
        if cached is not None:
            if cached.dtype != dtype or cached.numel != numel:
                raise ValueError(f"inconsistent PyTorch storage metadata for {key}")
            return cached
        raw = self.archive.read(f"{self.prefix}data/{key}")
        item_bytes = _TORCH_DTYPE_BYTES[dtype]
        expected = numel * item_bytes
        if len(raw) != expected:
            raise ValueError(f"PyTorch storage {key}: {len(raw)} bytes != {expected}")
        value = _TorchStorageRef(dtype, key, raw, numel)
        self.storages[key] = value
        return value

    def find_class(self, module: str, name: str):  # noqa: ANN001, N802
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch" and name.endswith("Storage"):
            dtype = _torch_dtype(name)
            if not dtype:
                raise ValueError(f"unsupported PyTorch storage class {name}")
            marker = type(name, (_TorchStorageMarker,), {"dtype": dtype})
            return marker
        if module == "torch._utils" and name in {
            "_rebuild_tensor",
            "_rebuild_tensor_v2",
            "_rebuild_tensor_v3",
        }:

            def rebuild(storage, offset, size, stride, *rest):  # noqa: ANN001
                del rest
                return _TorchTensorRef(
                    storage,
                    int(offset),
                    tuple(int(value) for value in size),
                    tuple(int(value) for value in stride),
                )

            return rebuild
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda tensor, requires_grad, hooks: tensor
        raise ValueError(f"unsupported PyTorch pickle global {module}.{name}")


def _canonical_tensor_name(name: str) -> str:
    # Fine-tuning checkpoints commonly stack wrappers (for example
    # ``module.roberta.``); remove every known wrapper so the inventory check
    # sees the same names as a plain Hugging Face state dict.
    prefixes = ("module.", "base_model.model.", "base_model.", "roberta.", "bert.")
    while True:
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        else:
            return name


def _torch_tensor_bytes(tensor: _TorchTensorRef) -> tuple[str, tuple[int, ...], bytes]:
    storage = tensor.storage
    item_bytes = _TORCH_DTYPE_BYTES[storage.dtype]
    if any(value < 0 for value in tensor.shape) or any(
        value < 0 for value in tensor.stride
    ):
        raise ValueError("negative PyTorch tensor shape or stride")
    count = math.prod(tensor.shape)
    if count == 0:
        return storage.dtype, tensor.shape, b""
    max_index = tensor.offset
    if tensor.shape:
        max_index += sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride)
        )
    if tensor.offset < 0 or max_index >= storage.numel:
        raise ValueError(
            f"PyTorch tensor view exceeds storage: offset={tensor.offset} shape={tensor.shape}"
        )
    # HF checkpoints use contiguous tensors, but honour the recorded strides so
    # a transposed view cannot silently corrupt a converted checkpoint.
    if not tensor.shape:
        indices = [tensor.offset]
    elif tensor.stride == tuple(
        math.prod(tensor.shape[index + 1 :]) for index in range(len(tensor.shape))
    ):
        start = tensor.offset * item_bytes
        end = start + count * item_bytes
        return storage.dtype, tensor.shape, storage.raw[start:end]
    else:
        values = bytearray(count * item_bytes)
        for linear in range(count):
            remainder = linear
            element = tensor.offset
            for axis in range(len(tensor.shape) - 1, -1, -1):
                coordinate = remainder % tensor.shape[axis]
                remainder //= tensor.shape[axis]
                element += coordinate * tensor.stride[axis]
            start = element * item_bytes
            dest = linear * item_bytes
            values[dest : dest + item_bytes] = storage.raw[start : start + item_bytes]
        return storage.dtype, tensor.shape, bytes(values)
    start = indices[0] * item_bytes
    return storage.dtype, tensor.shape, storage.raw[start : start + item_bytes]


class TensorStore:
    """Index one safetensors file or a standard HF shard set."""

    def __init__(self, model_dir: Path) -> None:
        if model_dir.is_file():
            self._load_torch_bin(model_dir)
            return
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"{index_path}: missing weight_map")
            shard_names = sorted(set(weight_map.values()))
        else:
            candidates = sorted(model_dir.glob("*.safetensors"))
            if not candidates:
                bin_path = model_dir / "pytorch_model.bin"
                if bin_path.exists():
                    self._load_torch_bin(bin_path)
                    return
                raise FileNotFoundError(
                    f"{model_dir}: expected safetensors, model.safetensors.index.json, or pytorch_model.bin"
                )
            if len(candidates) != 1:
                raise ValueError(
                    f"{model_dir}: multiple safetensors files require model.safetensors.index.json"
                )
            weight_map = None
            shard_names = [candidates[0].name]

        self._files = {}
        self._tensors: dict[str, SourceTensor] = {}
        for shard_name in shard_names:
            path = model_dir / shard_name
            source = open_safetensors(path)
            self._files[shard_name] = source
            for name, info in source.header.items():
                if name == "__metadata__":
                    continue
                if weight_map is not None and weight_map.get(name) != shard_name:
                    raise ValueError(f"{index_path}: weight_map mismatch for {name}")
                if name in self._tensors:
                    raise ValueError(f"duplicate source tensor {name}")
                dtype = info.get("dtype")
                shape = tuple(info.get("shape", ()))
                offsets = info.get("data_offsets")
                if name in _DROP:
                    # Position ids and the optional pooler are registered in
                    # many HF checkpoints but never enter the embedding graph.
                    continue
                if dtype not in _DTYPE_BYTES or not offsets:
                    raise ValueError(f"invalid safetensors entry {name}: {info}")
                start, end = offsets
                if (
                    start < 0
                    or end < start
                    or end > source.file_size - source.data_offset
                ):
                    raise ValueError(f"invalid byte range for {name}: {offsets}")
                expected = math.prod(shape) * _DTYPE_BYTES[dtype]
                if end - start != expected:
                    raise ValueError(
                        f"{name}: payload {end - start} != {expected} for {dtype}{shape}"
                    )
                self._tensors[name] = SourceTensor(
                    name, dtype, shape, source.path, start, end, source.data_offset
                )
        expected_names = (
            set(weight_map) - _DROP if weight_map is not None else set(self._tensors)
        )
        if weight_map is not None and expected_names != set(self._tensors):
            missing = sorted(expected_names - set(self._tensors))
            extra = sorted(set(self._tensors) - expected_names)
            raise ValueError(
                f"safetensors index inventory mismatch: missing={missing[:3]} extra={extra[:3]}"
            )

    def _load_torch_bin(self, path: Path) -> None:
        """Index a torch.save ZIP using only stdlib pickle/zipfile support."""
        self._files = {path.name: path}
        self._tensors: dict[str, SourceTensor] = {}
        try:
            archive = zipfile.ZipFile(path)
        except zipfile.BadZipFile as exc:
            raise ValueError(
                f"{path}: legacy non-ZIP PyTorch checkpoints are unsupported"
            ) from exc
        try:
            pickle_name = next(
                name for name in archive.namelist() if name.endswith("/data.pkl")
            )
            prefix = pickle_name[: -len("data.pkl")]
            obj = _TorchUnpickler(archive.read(pickle_name), archive, prefix).load()
        finally:
            archive.close()

        tensors: dict[str, _TorchTensorRef] = {}

        def visit(value: object, prefix_name: str = "") -> None:
            if isinstance(value, _TorchTensorRef):
                if not prefix_name:
                    raise ValueError(f"{path}: unnamed tensor in PyTorch checkpoint")
                tensors[_canonical_tensor_name(prefix_name)] = value
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    name = str(key)
                    visit(item, f"{prefix_name}.{name}" if prefix_name else name)

        visit(obj)
        if not tensors:
            raise ValueError(f"{path}: no tensors found in PyTorch checkpoint")
        for name, tensor in tensors.items():
            if name in _DROP:
                continue
            dtype, shape, raw = _torch_tensor_bytes(tensor)
            if dtype not in _DTYPE_BYTES:
                raise ValueError(f"invalid PyTorch tensor {name}: {dtype}{shape}")
            self._tensors[name] = SourceTensor(
                name,
                dtype,
                shape,
                path,
                0,
                len(raw),
                0,
                raw,
            )
        if not self._tensors:
            raise ValueError(
                f"{path}: no usable tensors found after dropping auxiliary heads"
            )

    def select(self, mapping: dict[str, str]) -> TensorStore:
        selected = object.__new__(TensorStore)
        selected._files = self._files
        selected._tensors = {
            target: replace(self[source], name=target)
            for target, source in mapping.items()
        }
        return selected

    def __contains__(self, name: str) -> bool:
        return name in self._tensors

    def __getitem__(self, name: str) -> SourceTensor:
        try:
            return self._tensors[name]
        except KeyError as exc:
            raise ValueError(f"missing source tensor {name}") from exc

    def names(self) -> set[str]:
        return set(self._tensors)

    def read(self, tensor: SourceTensor) -> bytes:
        if tensor.raw is not None:
            return tensor.raw
        with tensor.path.open("rb") as source:
            source.seek(tensor.data_offset + tensor.start)
            raw = source.read(tensor.nbytes)
        if len(raw) != tensor.nbytes:
            raise ValueError(f"truncated source tensor {tensor.name}")
        return raw


def _load_json(model_dir: Path, name: str, *, required: bool = True) -> dict | None:
    path = model_dir / name
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _validate_config(config: dict) -> tuple[str, int, int, int, int, int, int, int]:
    model_type = config.get("model_type")
    if model_type not in {"bert", "xlm-roberta"}:
        raise ValueError(f"unsupported BGE encoder model_type {model_type!r}")
    hidden = config.get("hidden_size")
    layers = config.get("num_hidden_layers")
    heads = config.get("num_attention_heads")
    ff = config.get("intermediate_size")
    vocab = config.get("vocab_size")
    context = config.get("max_position_embeddings")
    type_vocab = config.get("type_vocab_size", 1 if model_type == "xlm-roberta" else 2)
    values = (hidden, layers, heads, ff, vocab, context, type_vocab)
    if any(not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError(f"invalid BGE encoder dimensions: {values}")
    if hidden % heads:
        raise ValueError(
            f"hidden_size {hidden} is not divisible by num_attention_heads {heads}"
        )
    if config.get("position_embedding_type", "absolute") != "absolute":
        raise ValueError("only absolute-position BGE encoders are supported")
    if config.get("hidden_act", "gelu") not in {"gelu", "gelu_new"}:
        raise ValueError(f"unsupported BGE hidden_act {config.get('hidden_act')!r}")
    return model_type, hidden, layers, heads, ff, vocab, context, type_vocab


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift >= 70:
            break
    raise ValueError("invalid protobuf varint")


def _protobuf_fields(data: bytes) -> Iterable[tuple[int, int, int | bytes]]:
    offset = 0
    while offset < len(data):
        key, offset = _varint(data, offset)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, offset = _varint(data, offset)
        elif wire == 1:
            end = offset + 8
            if end > len(data):
                raise ValueError("truncated protobuf fixed64 field")
            value, offset = data[offset:end], end
        elif wire == 2:
            length, offset = _varint(data, offset)
            end = offset + length
            if end > len(data):
                raise ValueError("truncated protobuf bytes field")
            value, offset = data[offset:end], end
        elif wire == 5:
            end = offset + 4
            if end > len(data):
                raise ValueError("truncated protobuf fixed32 field")
            value, offset = data[offset:end], end
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield field, wire, value


def _read_sentencepiece_model(path: Path) -> tuple[list[str], list[float], list[int]]:
    """Read the small protobuf subset used by SentencePiece ModelProto."""
    pieces: list[str] = []
    scores: list[float] = []
    types: list[int] = []
    for field, wire, value in _protobuf_fields(path.read_bytes()):
        if field != 1 or wire != 2 or not isinstance(value, bytes):
            continue
        piece = None
        score = 0.0
        token_type = 1
        for subfield, subwire, subvalue in _protobuf_fields(value):
            if subfield == 1 and subwire == 2 and isinstance(subvalue, bytes):
                piece = subvalue.decode("utf-8")
            elif subfield == 2 and subwire == 5 and isinstance(subvalue, bytes):
                score = struct.unpack("<f", subvalue)[0]
            elif subfield == 3 and subwire == 0 and isinstance(subvalue, int):
                token_type = subvalue
        if piece is None:
            raise ValueError(f"{path}: SentencePiece entry has no piece")
        pieces.append(piece)
        scores.append(score)
        types.append(token_type)
    if not pieces:
        raise ValueError(f"{path}: no SentencePiece entries")
    return pieces, scores, types


def _added_tokens(
    tokens: list[str],
    scores: list[float],
    token_types: list[int],
    added: object,
) -> None:
    if not isinstance(added, list):
        return
    for item in added:
        if not isinstance(item, dict) or "id" not in item or "content" not in item:
            raise ValueError(f"invalid added token {item!r}")
        index = int(item["id"])
        content = str(item["content"])
        special = bool(item.get("special", False))
        if index < len(tokens):
            if tokens[index] != content:
                raise ValueError(f"added token {index} disagrees with model vocabulary")
            if special:
                token_types[index] = 3
            continue
        if index != len(tokens):
            raise ValueError("added token ids must be contiguous")
        tokens.append(content)
        scores.append(0.0)
        token_types.append(3 if special else 4)


def _read_tokens(model_dir: Path, model_type: str) -> TokenizerSpec:
    """Read tokenizer JSON/ModelProto into GGUF tokenizer metadata."""
    tokenizer_json = _load_json(model_dir, "tokenizer.json", required=False)
    tokenizer_config = (
        _load_json(model_dir, "tokenizer_config.json", required=False) or {}
    )
    if tokenizer_json is None and (model_dir / "vocab.json").exists():
        vocab = _load_json(model_dir, "vocab.json")
        merges = [
            line
            for line in (model_dir / "merges.txt").read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        added = [
            dict(value, id=int(index))
            for index, value in sorted(
                tokenizer_config.get("added_tokens_decoder", {}).items(),
                key=lambda item: int(item[0]),
            )
        ]
        tokenizer_json = {
            "model": {"type": "BPE", "vocab": vocab, "merges": merges},
            "added_tokens": added,
        }
    specials = {}
    for semantic in ("unk", "bos", "eos", "pad", "cls", "sep", "mask"):
        value = tokenizer_config.get(f"{semantic}_token")
        if isinstance(value, dict):
            value = value.get("content")
        if isinstance(value, str):
            specials[semantic] = value

    add_bos = bool(tokenizer_config.get("add_bos_token", False))
    add_eos = bool(tokenizer_config.get("add_eos_token", False))

    # Llama/Mistral and Gemma checkpoints ship the authoritative SentencePiece
    # ModelProto beside tokenizer.json.  Reading it avoids translating the
    # BPE compatibility view in tokenizer.json back into SPM pieces/scores.
    sentencepiece_path = model_dir / "tokenizer.model"
    if sentencepiece_path.exists() and model_type in {"mistral", "gemma2"}:
        tokens, scores, token_types = map(
            list, _read_sentencepiece_model(sentencepiece_path)
        )
        if tokenizer_json is not None:
            _added_tokens(
                tokens, scores, token_types, tokenizer_json.get("added_tokens")
            )
        return TokenizerSpec(
            "llama",
            tokens,
            scores,
            specials,
            None,
            [],
            token_types,
            add_bos,
            add_eos,
            model_type != "gemma2",
        )

    if tokenizer_json is not None:
        model = tokenizer_json.get("model") or {}
        charsmap = None
        normalizer = tokenizer_json.get("normalizer")
        normalizers = (
            normalizer.get("normalizers", []) if isinstance(normalizer, dict) else []
        )
        if isinstance(normalizers, list):
            for item in normalizers:
                if isinstance(item, dict) and item.get("type") == "Precompiled":
                    encoded = item.get("precompiled_charsmap")
                    if not isinstance(encoded, str):
                        raise ValueError(
                            "tokenizer.json Precompiled normalizer has no charsmap"
                        )
                    charsmap = base64.b64decode(encoded, validate=True)
                    break
        model_kind = model.get("type")
        if model_kind == "WordPiece":
            vocab = model.get("vocab")
            if not isinstance(vocab, dict):
                raise ValueError("tokenizer.json WordPiece model has no vocab")
            by_id = sorted(((int(index), token) for token, index in vocab.items()))
            if [index for index, _ in by_id] != list(range(len(by_id))):
                raise ValueError("WordPiece tokenizer ids are not contiguous")
            # llama.cpp's BERT vocabulary uses the SentencePiece-style
            # phantom-space marker for a word-initial piece and stores
            # continuation pieces without Hugging Face's ``##`` prefix.
            # WPMTokenizer deliberately prepends ``▁`` while matching each
            # pre-tokenized word, so emit the corresponding GGUF spelling.
            tokens = []
            for _, token in by_id:
                if token.startswith("##"):
                    tokens.append(token[2:])
                elif token.startswith("[") and token.endswith("]"):
                    tokens.append(token)
                else:
                    tokens.append("▁" + token)
            return TokenizerSpec(
                "bert",
                tokens,
                [0.0] * len(tokens),
                specials,
                None,
                [],
                None,
                False,
                False,
            )
        if model_kind == "Unigram":
            vocab = model.get("vocab")
            if not isinstance(vocab, list) or any(
                not isinstance(item, list) or len(item) != 2 for item in vocab
            ):
                raise ValueError("tokenizer.json Unigram model has invalid vocab")
            tokens = [str(item[0]) for item in vocab]
            scores = [float(item[1]) for item in vocab]
            token_types = [1] * len(tokens)
            _added_tokens(
                tokens, scores, token_types, tokenizer_json.get("added_tokens")
            )
            return TokenizerSpec(
                "t5",
                tokens,
                scores,
                specials,
                charsmap,
                [],
                token_types,
                add_bos,
                add_eos,
            )
        if model_kind == "BPE":
            vocab = model.get("vocab")
            merges = model.get("merges")
            if not isinstance(vocab, dict) or not isinstance(merges, list):
                raise ValueError("tokenizer.json BPE model has invalid vocab or merges")
            by_id = sorted((int(index), token) for token, index in vocab.items())
            if [index for index, _ in by_id] != list(range(len(by_id))):
                raise ValueError("BPE tokenizer ids are not contiguous")
            tokens = [token for _, token in by_id]
            scores = [0.0] * len(tokens)
            token_types = [1] * len(tokens)
            _added_tokens(
                tokens, scores, token_types, tokenizer_json.get("added_tokens")
            )
            merge_strings = []
            for merge in merges:
                if isinstance(merge, str):
                    merge_strings.append(merge)
                elif isinstance(merge, list) and len(merge) == 2:
                    merge_strings.append(f"{merge[0]} {merge[1]}")
                else:
                    raise ValueError(f"invalid BPE merge {merge!r}")
            return TokenizerSpec(
                "gpt2",
                tokens,
                scores,
                specials,
                None,
                merge_strings,
                token_types,
                add_bos,
                add_eos,
            )
        raise ValueError(f"unsupported tokenizer.json model type {model_kind!r}")

    vocab_path = model_dir / "vocab.txt"
    if vocab_path.exists():
        tokens = [line.rstrip("\n\r") for line in vocab_path.read_text(encoding="utf-8").splitlines()]
        tokens = [
            (
                token[2:]
                if token.startswith("##")
                else (
                    token
                    if token.startswith("[") and token.endswith("]")
                    else "▁" + token
                )
            )
            for token in tokens
        ]
        return TokenizerSpec(
            "bert", tokens, [0.0] * len(tokens), specials, None, [], None, False, False
        )
    raise FileNotFoundError(
        f"{model_dir}: tokenizer.json, tokenizer.model, or vocab.txt is required"
    )


def _special_token_ids(
    tokens: list[str], names: dict[str, int | str]
) -> dict[str, int]:
    ids: dict[str, int] = {}
    for semantic, value in names.items():
        if isinstance(value, str):
            try:
                ids[semantic] = tokens.index(value)
            except ValueError:
                continue
    return ids


def _token_types(
    kind: str,
    tokens: list[str],
    special_ids: dict[str, int],
    source_types: list[int] | None = None,
) -> list[int]:
    if source_types is not None:
        if len(source_types) != len(tokens):
            raise ValueError("tokenizer token_type length does not match vocabulary")
    types = list(source_types) if source_types is not None else [1] * len(tokens)
    if kind == "bert":
        for semantic in ("pad", "cls", "sep", "mask", "bos", "eos"):
            index = special_ids.get(semantic)
            if index is not None:
                types[index] = 3
        if "unk" in special_ids:
            types[special_ids["unk"]] = 2
    else:
        # SentencePiece control/unknown IDs are part of the model vocabulary;
        # added schema tokens are user-defined and must remain parseable.
        for semantic in ("pad", "cls", "sep", "bos", "eos", "mask"):
            index = special_ids.get(semantic)
            if index is not None:
                types[index] = 3
        if "unk" in special_ids:
            types[special_ids["unk"]] = 2
        for index, token in enumerate(tokens):
            if token.startswith("<extra_id_"):
                types[index] = 4
    return types


def _source_names(
    model_type: str, layer_count: int, type_vocab_size: int
) -> dict[str, str]:
    names = {
        "token_embd.weight": "embeddings.word_embeddings.weight",
        "position_embd.weight": "embeddings.position_embeddings.weight",
        "token_embd_norm.weight": "embeddings.LayerNorm.weight",
        "token_embd_norm.bias": "embeddings.LayerNorm.bias",
    }
    if type_vocab_size:
        names["token_types.weight"] = "embeddings.token_type_embeddings.weight"
    for layer in range(layer_count):
        src = f"encoder.layer.{layer}"
        dst = f"blk.{layer}"
        names.update(
            {
                f"{dst}.attn_q.weight": f"{src}.attention.self.query.weight",
                f"{dst}.attn_q.bias": f"{src}.attention.self.query.bias",
                f"{dst}.attn_k.weight": f"{src}.attention.self.key.weight",
                f"{dst}.attn_k.bias": f"{src}.attention.self.key.bias",
                f"{dst}.attn_v.weight": f"{src}.attention.self.value.weight",
                f"{dst}.attn_v.bias": f"{src}.attention.self.value.bias",
                f"{dst}.attn_output.weight": f"{src}.attention.output.dense.weight",
                f"{dst}.attn_output.bias": f"{src}.attention.output.dense.bias",
                f"{dst}.attn_output_norm.weight": f"{src}.attention.output.LayerNorm.weight",
                f"{dst}.attn_output_norm.bias": f"{src}.attention.output.LayerNorm.bias",
                f"{dst}.ffn_up.weight": f"{src}.intermediate.dense.weight",
                f"{dst}.ffn_up.bias": f"{src}.intermediate.dense.bias",
                f"{dst}.ffn_down.weight": f"{src}.output.dense.weight",
                f"{dst}.ffn_down.bias": f"{src}.output.dense.bias",
                f"{dst}.layer_output_norm.weight": f"{src}.output.LayerNorm.weight",
                f"{dst}.layer_output_norm.bias": f"{src}.output.LayerNorm.bias",
            }
        )
    return names


def _decoder_mapping(model_type: str, layer_count: int) -> dict[str, str]:
    """Map HuggingFace decoder names to the llama.cpp GGUF contract."""
    if model_type not in {"mistral", "qwen2", "qwen3", "gemma2"}:
        raise ValueError(f"unsupported BGE decoder model_type {model_type!r}")
    mapping = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
    }
    for layer in range(layer_count):
        src = f"model.layers.{layer}"
        dst = f"blk.{layer}"
        mapping.update(
            {
                f"{dst}.attn_norm.weight": f"{src}.input_layernorm.weight",
                f"{dst}.attn_q.weight": f"{src}.self_attn.q_proj.weight",
                f"{dst}.attn_k.weight": f"{src}.self_attn.k_proj.weight",
                f"{dst}.attn_v.weight": f"{src}.self_attn.v_proj.weight",
                f"{dst}.attn_output.weight": f"{src}.self_attn.o_proj.weight",
                f"{dst}.ffn_gate.weight": f"{src}.mlp.gate_proj.weight",
                f"{dst}.ffn_up.weight": f"{src}.mlp.up_proj.weight",
                f"{dst}.ffn_down.weight": f"{src}.mlp.down_proj.weight",
            }
        )
        if model_type == "gemma2":
            mapping.update(
                {
                    f"{dst}.post_attention_norm.weight": f"{src}.post_attention_layernorm.weight",
                    f"{dst}.ffn_norm.weight": f"{src}.pre_feedforward_layernorm.weight",
                    f"{dst}.post_ffw_norm.weight": f"{src}.post_feedforward_layernorm.weight",
                }
            )
        else:
            mapping[f"{dst}.ffn_norm.weight"] = f"{src}.post_attention_layernorm.weight"
        if model_type == "qwen2":
            for projection in ("q", "k", "v"):
                mapping[f"{dst}.attn_{projection}.bias"] = (
                    f"{src}.self_attn.{projection}_proj.bias"
                )
        if model_type == "qwen3":
            mapping.update(
                {
                    f"{dst}.attn_q_norm.weight": f"{src}.self_attn.q_norm.weight",
                    f"{dst}.attn_k_norm.weight": f"{src}.self_attn.k_norm.weight",
                }
            )
    return mapping


def _decoder_arch_metadata(config: dict, model_type: str) -> dict[str, object]:
    hidden = config.get("hidden_size")
    layers = config.get("num_hidden_layers")
    heads = config.get("num_attention_heads")
    kv_heads = config.get("num_key_value_heads", heads)
    ff = config.get("intermediate_size")
    vocab = config.get("vocab_size")
    context = config.get("max_position_embeddings")
    values = (hidden, layers, heads, kv_heads, ff, vocab, context)
    if any(not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError(f"invalid BGE decoder dimensions: {values}")
    if hidden % heads or heads % kv_heads:
        raise ValueError(
            f"invalid decoder head shape: hidden={hidden}, heads={heads}, kv={kv_heads}"
        )
    head_dim = int(config.get("head_dim", hidden // heads))
    if head_dim <= 0:
        raise ValueError(f"invalid decoder head_dim {head_dim}")
    return {
        "hidden": hidden,
        "layers": layers,
        "heads": heads,
        "kv_heads": kv_heads,
        "ff": ff,
        "vocab": vocab,
        "context": context,
        "head_dim": head_dim,
        "rope_theta": float(config.get("rope_theta", 10000.0)),
        "eps": float(config.get("rms_norm_eps", 1e-5)),
    }


def _to_f32(raw: bytes, dtype: str) -> bytes:
    if dtype == "F32":
        return raw
    if dtype == "F16":
        return f16_to_f32(raw)
    if dtype == "BF16":
        return bf16_to_f32(raw)
    raise ValueError(f"unsupported source dtype {dtype}")


def _chunks(store: TensorStore, tensor: SourceTensor, quant: str) -> Iterable[bytes]:
    """Stream a tensor without retaining the complete model in memory."""
    chunk_bytes = 8 * 1024 * 1024
    unit = _DTYPE_BYTES[tensor.dtype]
    chunk_bytes -= chunk_bytes % unit
    remaining = tensor.nbytes
    offset = 0
    while remaining:
        length = min(remaining, chunk_bytes)
        # TensorStore.read is also the source of truth for torch ZIP tensors,
        # whose bytes do not live at ``data_offset + start`` in the archive.
        raw = store.read(
            SourceTensor(
                tensor.name,
                tensor.dtype,
                tensor.shape,
                tensor.path,
                tensor.start + offset,
                tensor.start + offset + length,
                tensor.data_offset,
                (
                    tensor.raw[offset : offset + length]
                    if tensor.raw is not None
                    else None
                ),
            )
        )
        if len(raw) != length:
            raise ValueError(f"truncated source tensor {tensor.name}")
        f32 = _to_f32(raw, tensor.dtype)
        if not np.isfinite(np.frombuffer(f32, dtype="<f4")).all():
            raise ValueError(f"non-finite values in {tensor.name}")
        if quant == "F32":
            yield f32
        else:
            values = np.frombuffer(f32, dtype="<f4")
            if values.size % 32:
                raise ValueError(
                    f"{tensor.name}: Q8_0 requires a multiple of 32 values"
                )
            yield quantize_q8_0(values)
        remaining -= length
        offset += length


def _tensor_shape(
    target: str, tensor: SourceTensor, hidden: int, type_vocab_size: int
) -> tuple[int, ...]:
    if target == "token_types.weight" and type_vocab_size == 1:
        if math.prod(tensor.shape) != hidden:
            raise ValueError(f"{target}: expected {hidden} values, got {tensor.shape}")
        return (hidden,)
    return tensor.shape


def _trim_position_rows(tensor: SourceTensor, offset: int) -> SourceTensor:
    """Drop XLM-R's pad/control rows from the learned position table.

    XLM-R stores rows for ``pad_token_id`` and the following control position
    before the positions used by normal input.  llama.cpp removes those rows
    before writing ``bert.context_length`` and ``position_embd.weight``.
    Keep the operation as a byte-range view for safetensors and as a sliced
    payload for PyTorch ZIP checkpoints so conversion remains streaming for
    large shards.
    """
    if offset < 0 or len(tensor.shape) != 2:
        raise ValueError(f"{tensor.name}: expected a 2-D position table")
    rows, width = tensor.shape
    if offset >= rows:
        raise ValueError(f"{tensor.name}: position offset {offset} exceeds {rows} rows")
    row_bytes = width * _DTYPE_BYTES[tensor.dtype]
    shape = (rows - offset, width)
    if tensor.raw is not None:
        start = offset * row_bytes
        raw = tensor.raw[start : start + shape[0] * row_bytes]
        return SourceTensor(
            tensor.name,
            tensor.dtype,
            shape,
            tensor.path,
            tensor.start,
            tensor.start + len(raw),
            tensor.data_offset,
            raw,
        )
    start = tensor.start + offset * row_bytes
    end = start + shape[0] * row_bytes
    if end > tensor.end:
        raise ValueError(
            f"{tensor.name}: position table payload is shorter than its shape"
        )
    return SourceTensor(
        tensor.name, tensor.dtype, shape, tensor.path, start, end, tensor.data_offset
    )


def _sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _source_sha256(store: TensorStore) -> str:
    return _sha256(
        source if isinstance(source, Path) else source.path
        for source in store._files.values()
    )


def _write_tokenizer_metadata(
    writer: GgufWriter,
    spec: TokenizerSpec,
    special_ids: dict[str, int],
    vocab: int,
    *,
    type_vocab_count: int | None = None,
) -> None:
    if len(spec.tokens) != vocab:
        raise ValueError(
            f"tokenizer vocabulary {len(spec.tokens)} != config vocab_size {vocab}"
        )
    token_types = _token_types(spec.kind, spec.tokens, special_ids, spec.token_types)
    _add_meta(writer, "tokenizer.ggml.model", spec.kind)
    _add_meta(writer, "tokenizer.ggml.tokens", spec.tokens)
    # llama.cpp expects tokenizer token types as a signed int32 array and SPM
    # scores as f32 values.
    _add_meta(writer, "tokenizer.ggml.token_type", MetaArray(5, tuple(token_types)))
    _add_meta(writer, "tokenizer.ggml.vocab_size", vocab)
    if type_vocab_count is not None:
        _add_meta(writer, "tokenizer.ggml.token_type_count", type_vocab_count)

    if spec.kind == "bert":
        _add_meta(writer, "tokenizer.ggml.pre", "jina-v2-en")
        _add_meta(
            writer, "tokenizer.ggml.unknown_token_id", special_ids.get("unk", 100)
        )
        _add_meta(
            writer, "tokenizer.ggml.seperator_token_id", special_ids.get("sep", 102)
        )
        _add_meta(writer, "tokenizer.ggml.padding_token_id", special_ids.get("pad", 0))
        _add_meta(writer, "tokenizer.ggml.cls_token_id", special_ids.get("cls", 101))
        _add_meta(writer, "tokenizer.ggml.mask_token_id", special_ids.get("mask", 103))
        # BERT's llama.cpp vocabulary branch wraps embedding prompts in
        # [CLS]…[SEP] by default.  Leave the add_* keys absent so the model
        # type default stays aligned with the official BGE GGUF files; the
        # Rust WPM tokenizer has the same model-type contract.
        return
    if spec.kind == "t5":
        _add_meta(writer, "tokenizer.ggml.pre", "default")
        _add_meta(writer, "tokenizer.ggml.scores", MetaArray(6, tuple(spec.scores)))
        _add_meta(
            writer,
            "tokenizer.ggml.bos_token_id",
            special_ids.get("bos", special_ids.get("cls", 0)),
        )
        _add_meta(
            writer,
            "tokenizer.ggml.eos_token_id",
            special_ids.get("eos", special_ids.get("sep", 2)),
        )
        _add_meta(writer, "tokenizer.ggml.unknown_token_id", special_ids.get("unk", 3))
        _add_meta(writer, "tokenizer.ggml.padding_token_id", special_ids.get("pad", 1))
        _add_meta(
            writer, "tokenizer.ggml.seperator_token_id", special_ids.get("sep", 2)
        )
        _add_meta(
            writer, "tokenizer.ggml.mask_token_id", special_ids.get("mask", vocab - 1)
        )
        _add_meta(writer, "tokenizer.ggml.add_bos_token", True)
        _add_meta(writer, "tokenizer.ggml.add_eos_token", True)
        _add_meta(writer, "tokenizer.ggml.add_sep_token", True)
        _add_meta(writer, "tokenizer.ggml.add_space_prefix", spec.add_space_prefix)
        _add_meta(writer, "tokenizer.ggml.remove_extra_whitespaces", True)
        _add_meta(writer, "tokenizer.ggml.treat_whitespace_as_suffix", False)
        if spec.charsmap is not None:
            _add_meta(
                writer,
                "tokenizer.ggml.precompiled_charsmap",
                MetaArray(0, tuple(spec.charsmap)),
            )
        return
    if spec.kind == "llama":
        _add_meta(writer, "tokenizer.ggml.pre", "default")
        _add_meta(writer, "tokenizer.ggml.scores", MetaArray(6, tuple(spec.scores)))
        _add_meta(writer, "tokenizer.ggml.bos_token_id", special_ids.get("bos", 1))
        _add_meta(writer, "tokenizer.ggml.eos_token_id", special_ids.get("eos", 2))
        _add_meta(writer, "tokenizer.ggml.unknown_token_id", special_ids.get("unk", 0))
        _add_meta(writer, "tokenizer.ggml.padding_token_id", special_ids.get("pad", 0))
        _add_meta(writer, "tokenizer.ggml.add_bos_token", spec.add_bos)
        _add_meta(writer, "tokenizer.ggml.add_eos_token", spec.add_eos)
        _add_meta(writer, "tokenizer.ggml.add_space_prefix", spec.add_space_prefix)
        return
    if spec.kind == "gpt2":
        _add_meta(writer, "tokenizer.ggml.pre", "qwen2")
        _add_meta(writer, "tokenizer.ggml.merges", spec.merges)
        _add_meta(writer, "tokenizer.ggml.bos_token_id", special_ids.get("bos", 0))
        _add_meta(writer, "tokenizer.ggml.eos_token_id", special_ids.get("eos", 0))
        _add_meta(writer, "tokenizer.ggml.unknown_token_id", special_ids.get("unk", 0))
        _add_meta(
            writer,
            "tokenizer.ggml.padding_token_id",
            special_ids.get("pad", special_ids.get("eos", 0)),
        )
        _add_meta(writer, "tokenizer.ggml.add_bos_token", spec.add_bos)
        _add_meta(writer, "tokenizer.ggml.add_eos_token", spec.add_eos)
        return
    raise ValueError(f"unsupported tokenizer kind {spec.kind!r}")


def _add_mapped_tensors(
    writer: GgufWriter,
    store: TensorStore,
    mapping: dict[str, str],
    quant: str,
    model_type: str,
    dims: dict[str, object],
    *,
    quantize_embeddings: bool = True,
) -> None:
    for target, source_name in mapping.items():
        tensor = store[source_name]
        if model_type == "gemma2" and "norm.weight" in source_name:
            values = np.frombuffer(
                _to_f32(store.read(tensor), tensor.dtype), dtype="<f4"
            ) + np.float32(1)
            raw = values.tobytes()
            tensor = replace(
                tensor, dtype="F32", start=0, end=len(raw), data_offset=0, raw=raw
            )
        elif model_type == "mistral" and target.endswith(
            ("attn_q.weight", "attn_k.weight")
        ):
            heads = int(
                dims["kv_heads"] if target.endswith("attn_k.weight") else dims["heads"]
            )
            values = np.frombuffer(
                _to_f32(store.read(tensor), tensor.dtype), dtype="<f4"
            )
            # HF splits each RoPE head in halves; GGUF llama interleaves its pairs.
            raw = (
                values.reshape(heads, 2, -1, tensor.shape[1])
                .swapaxes(1, 2)
                .reshape(tensor.shape)
                .tobytes()
            )
            tensor = replace(
                tensor, dtype="F32", start=0, end=len(raw), data_offset=0, raw=raw
            )
        quant_tensor = quant == "Q8_0" and (
            (quantize_embeddings and target == "token_embd.weight")
            or (
                target.startswith("blk.")
                and target.endswith(".weight")
                and "norm" not in target
            )
            or target == "output.weight"
        )
        tensor_quant = "Q8_0" if quant_tensor else "F32"
        ggml_type = GGML_Q8_0 if quant_tensor else GGML_F32
        elements = math.prod(tensor.shape)
        if quant_tensor and elements % 32:
            raise ValueError(f"{target}: Q8_0 requires a multiple of 32 values")
        nbytes = elements * 4 if not quant_tensor else elements // 32 * 34
        writer.add_tensor_chunks(
            target,
            ggml_type,
            gguf_dims(tensor.shape),
            nbytes,
            lambda tensor=tensor, tensor_quant=tensor_quant: _chunks(
                store, tensor, tensor_quant
            ),
        )


def _decoder_writer(
    model_dir: Path,
    output: Path,
    config: dict,
    quant: str,
    store: TensorStore,
    *,
    architecture: str | None = None,
) -> GgufWriter:
    model_type = str(config.get("model_type"))
    dims = _decoder_arch_metadata(config, model_type)
    layers = int(dims["layers"])
    vocab = int(dims["vocab"])
    spec = _read_tokens(model_dir, model_type)
    if len(spec.tokens) < vocab:
        # Qwen3 reserves a tail of embedding rows that is intentionally absent
        # from tokenizer.json. Keep the ids addressable in GGUF while leaving
        # them unused by normal text tokenization.
        start = len(spec.tokens)
        spec.tokens.extend(f"<|reserved_{index}|>" for index in range(start, vocab))
        spec.scores.extend([0.0] * (vocab - start))
        if spec.token_types is not None:
            spec.token_types.extend([5] * (vocab - start))
    if len(spec.tokens) != vocab:
        raise ValueError(
            f"tokenizer vocabulary {len(spec.tokens)} != config vocab_size {vocab}"
        )
    special_ids = _special_token_ids(spec.tokens, spec.special_names)
    mapping = _decoder_mapping(model_type, layers)
    if "embed_tokens.weight" in store:
        mapping = {
            target: name.removeprefix("model.") for target, name in mapping.items()
        }
    # A tied embedding is sufficient for embedding extraction; an LM head is
    # accepted when present but never required by the embedding entry point.
    lm_head = "lm_head.weight"
    if lm_head in store and model_type != "gemma2":
        mapping["output.weight"] = lm_head
    required = set(mapping.values())
    missing = sorted(required - store.names())
    if missing:
        raise ValueError(
            f"checkpoint is missing required decoder tensors: {missing[:5]}"
        )
    allowed = required | {
        name
        for name in store.names()
        if name.endswith(".rotary_emb.inv_freq")
        or name.endswith(".rotary_emb.cos_cached")
    }
    # Gemma2 GGUF always ties its output to token_embd; an independent LM
    # head would be an unused tensor rejected by llama.cpp's loader.
    if model_type == "gemma2":
        allowed.add(lm_head)
    unexpected = sorted(store.names() - allowed)
    if unexpected:
        raise ValueError(
            f"unsupported BGE decoder tensors would be dropped: {unexpected[:5]}"
        )

    hidden, ff, head_dim = (int(dims[key]) for key in ("hidden", "ff", "head_dim"))
    query, kv = int(dims["heads"]) * head_dim, int(dims["kv_heads"]) * head_dim
    shapes = {
        "token_embd.weight": (vocab, hidden),
        "output.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
        "attn_norm.weight": (hidden,),
        "post_attention_norm.weight": (hidden,),
        "ffn_norm.weight": (hidden,),
        "post_ffw_norm.weight": (hidden,),
        "attn_q.weight": (query, hidden),
        "attn_k.weight": (kv, hidden),
        "attn_v.weight": (kv, hidden),
        "attn_output.weight": (hidden, query),
        "attn_q.bias": (query,),
        "attn_k.bias": (kv,),
        "attn_v.bias": (kv,),
        "ffn_gate.weight": (ff, hidden),
        "ffn_up.weight": (ff, hidden),
        "ffn_down.weight": (hidden, ff),
        "attn_q_norm.weight": (head_dim,),
        "attn_k_norm.weight": (head_dim,),
    }
    for target, name in mapping.items():
        expected = shapes[".".join(target.split(".")[-2:])]
        if store[name].shape != expected:
            raise ValueError(
                f"{name}: shape {store[name].shape} != expected {expected}"
            )

    arch = architecture or ("llama" if model_type == "mistral" else model_type)
    writer = GgufWriter(output.with_suffix(output.suffix + ".tmp"))
    _add_meta(writer, "general.architecture", arch)
    _add_meta(writer, "general.name", model_dir.name)
    _add_meta(writer, "general.file_type", 0 if quant == "F32" else 7)
    _add_meta(writer, "general.quantization_version", 2)
    _add_meta(writer, "general.source_sha256", _source_sha256(store))
    _add_meta(writer, f"{arch}.block_count", layers)
    _add_meta(writer, f"{arch}.context_length", int(dims["context"]))
    _add_meta(writer, f"{arch}.embedding_length", int(dims["hidden"]))
    _add_meta(writer, f"{arch}.feed_forward_length", int(dims["ff"]))
    _add_meta(writer, f"{arch}.attention.head_count", int(dims["heads"]))
    _add_meta(writer, f"{arch}.attention.head_count_kv", int(dims["kv_heads"]))
    _add_meta(writer, f"{arch}.attention.layer_norm_rms_epsilon", float(dims["eps"]))
    _add_meta(writer, f"{arch}.rope.freq_base", float(dims["rope_theta"]))
    _add_meta(writer, f"{arch}.rope.dimension_count", head_dim)
    _add_meta(writer, f"{arch}.attention.causal", True)
    _add_meta(writer, f"{arch}.pooling_type", 3)
    _add_meta(writer, f"{arch}.vocab_size", vocab)
    _add_meta(writer, f"{arch}.attention.key_length", int(dims["head_dim"]))
    _add_meta(writer, f"{arch}.attention.value_length", int(dims["head_dim"]))
    if model_type == "gemma2":
        _add_meta(writer, "gemma2.embedding_scale", math.sqrt(hidden))
        _add_meta(
            writer,
            "gemma2.attention.scale",
            1.0 / math.sqrt(config.get("query_pre_attn_scalar", head_dim)),
        )
        _add_meta(
            writer,
            "gemma2.attention.sliding_window_pattern",
            MetaArray(7, tuple(layer % 2 == 0 for layer in range(layers))),
        )
        _add_meta(
            writer,
            "gemma2.attn_logit_softcapping",
            float(config.get("attn_logit_softcapping", 50.0)),
        )
        _add_meta(
            writer,
            "gemma2.final_logit_softcapping",
            float(config.get("final_logit_softcapping", 30.0)),
        )
        _add_meta(
            writer,
            "gemma2.attention.sliding_window",
            int(config.get("sliding_window", 4096)),
        )
    elif model_type == "mistral" and config.get("sliding_window"):
        _add_meta(
            writer, "llama.attention.sliding_window", int(config["sliding_window"])
        )
    _write_tokenizer_metadata(writer, spec, special_ids, vocab)
    _add_mapped_tensors(writer, store, mapping, quant, model_type, dims)
    return writer


def convert(model_dir: Path, output: Path, quant: str = "F32") -> None:
    if output.exists():
        raise FileExistsError(output)
    if quant not in {"F32", "Q8_0"}:
        raise ValueError(f"unsupported quantization {quant!r}")
    config = _load_json(model_dir, "config.json")
    store = TensorStore(model_dir)
    build = (
        _decoder_writer
        if config.get("model_type") in {"mistral", "qwen2", "qwen3", "gemma2"}
        else _encoder_writer
    )
    writer = build(model_dir, output, config, quant, store)
    writer.write()
    writer.path.replace(output)
    print(output)


def _encoder_writer(
    model_dir: Path, output: Path, config: dict, quant: str, store: TensorStore
) -> GgufWriter:
    model_type, hidden, layers, heads, ff, vocab, context, type_vocab_size = (
        _validate_config(config)
    )
    spec = _read_tokens(model_dir, model_type)
    if len(spec.tokens) != vocab:
        raise ValueError(
            f"tokenizer vocabulary {len(spec.tokens)} != config vocab_size {vocab}"
        )
    special_ids = _special_token_ids(spec.tokens, spec.special_names)
    if spec.kind == "bert":
        required = {"unk": 100, "cls": 101, "sep": 102, "pad": 0}
        for name, index in required.items():
            if special_ids.get(name, index) != index:
                raise ValueError(f"BERT special token {name} must be id {index}")

    mapping = _source_names(model_type, layers, type_vocab_size)
    source_required = set(mapping.values())
    missing = sorted(source_required - store.names())
    if missing:
        raise ValueError(f"checkpoint is missing required tensors: {missing[:5]}")
    unexpected = sorted(store.names() - source_required - _DROP)
    if unexpected:
        raise ValueError(f"unsupported BGE tensors would be dropped: {unexpected[:5]}")

    shapes = {
        "token_embd.weight": (vocab, hidden),
        "position_embd.weight": (context, hidden),
        "token_types.weight": (type_vocab_size, hidden),
        "token_embd_norm.weight": (hidden,),
        "token_embd_norm.bias": (hidden,),
        "attn_q.weight": (hidden, hidden),
        "attn_q.bias": (hidden,),
        "attn_k.weight": (hidden, hidden),
        "attn_k.bias": (hidden,),
        "attn_v.weight": (hidden, hidden),
        "attn_v.bias": (hidden,),
        "attn_output.weight": (hidden, hidden),
        "attn_output.bias": (hidden,),
        "attn_output_norm.weight": (hidden,),
        "attn_output_norm.bias": (hidden,),
        "ffn_up.weight": (ff, hidden),
        "ffn_up.bias": (ff,),
        "ffn_down.weight": (hidden, ff),
        "ffn_down.bias": (hidden,),
        "layer_output_norm.weight": (hidden,),
        "layer_output_norm.bias": (hidden,),
    }
    for target, name in mapping.items():
        expected = shapes[".".join(target.split(".")[-2:])]
        if (
            target == "token_types.weight"
            and type_vocab_size == 1
            and store[name].shape == (hidden,)
        ):
            continue
        if store[name].shape != expected:
            raise ValueError(
                f"{name}: shape {store[name].shape} != expected {expected}"
            )

    position_offset = 0
    if model_type == "xlm-roberta":
        # XLM-R reserves rows through pad_token_id + 1.  This is the same
        # adjustment made by llama.cpp's XLMRobertaModel converter; without it
        # every input position is shifted and the GGUF context is two tokens
        # larger than the learned table actually supports.
        pad_token_id = config.get("pad_token_id")
        if not isinstance(pad_token_id, int) or pad_token_id < 0:
            raise ValueError(
                f"XLM-R config must define a non-negative pad_token_id, got {pad_token_id!r}"
            )
        position_offset = 1 + pad_token_id
        position = store[mapping["position_embd.weight"]]
        if len(position.shape) != 2 or position.shape[1] != hidden:
            raise ValueError(
                "XLM-R position_embeddings.weight must have shape "
                f"[rows, {hidden}], got {position.shape}"
            )
        if position.shape[0] != context:
            raise ValueError(
                "XLM-R position_embeddings.weight rows must match "
                f"config max_position_embeddings={context}, got {position.shape[0]}"
            )
        context -= position_offset
        if context <= 0:
            raise ValueError(
                f"XLM-R position offset {position_offset} leaves no usable context"
            )

    writer = GgufWriter(output.with_suffix(output.suffix + ".tmp"))
    _add_meta(writer, "general.architecture", "bert")
    _add_meta(writer, "general.name", model_dir.name)
    _add_meta(writer, "general.file_type", 0 if quant == "F32" else 7)
    _add_meta(writer, "general.quantization_version", 2)
    _add_meta(
        writer,
        "general.source_sha256",
        _source_sha256(store),
    )
    _add_meta(writer, "bert.block_count", layers)
    _add_meta(writer, "bert.context_length", context)
    _add_meta(writer, "bert.embedding_length", hidden)
    _add_meta(writer, "bert.feed_forward_length", ff)
    _add_meta(writer, "bert.attention.head_count", heads)
    _add_meta(
        writer,
        "bert.attention.layer_norm_epsilon",
        float(config.get("layer_norm_eps", 1e-12)),
    )
    _add_meta(writer, "bert.attention.causal", False)
    _add_meta(writer, "bert.pooling_type", 2)
    _add_meta(writer, "bert.vocab_size", vocab)

    _write_tokenizer_metadata(
        writer, spec, special_ids, vocab, type_vocab_count=type_vocab_size
    )
    if spec.kind == "bert":
        tokenizer_config = (
            _load_json(model_dir, "tokenizer_config.json", required=False) or {}
        )
        lowercase = bool(tokenizer_config.get("do_lower_case", True))
        strip_accents = tokenizer_config.get("strip_accents")
        _add_meta(writer, "tokenizer.ggml.normalizer.lowercase", lowercase)
        _add_meta(
            writer,
            "tokenizer.ggml.normalizer.strip_accents",
            lowercase if strip_accents is None else bool(strip_accents),
        )
    # `_write_tokenizer_metadata` already emitted the XLM-R UGM charsmap and
    # score arrays above; no model-specific tokenizer fields remain here.

    for target, source_name in mapping.items():
        tensor = store[source_name]
        if target == "position_embd.weight" and position_offset:
            tensor = _trim_position_rows(tensor, position_offset)
        shape = _tensor_shape(target, tensor, hidden, type_vocab_size)
        # The source embedding/projection matrices use the conventional HF
        # [out, in] order; GGUF's reversed dimensions preserve the bytes.
        quant_tensor = quant == "Q8_0" and (
            target == "token_embd.weight"
            or (
                target.startswith("blk.")
                and target.endswith(".weight")
                and not target.endswith("_norm.weight")
            )
        )
        tensor_quant = "Q8_0" if quant_tensor else "F32"
        ggml_type = GGML_Q8_0 if quant_tensor else GGML_F32
        elements = math.prod(shape)
        if quant_tensor and elements % 32:
            raise ValueError(f"{target}: Q8_0 requires a multiple of 32 values")
        nbytes = elements * 4 if not quant_tensor else elements // 32 * 34
        writer.add_tensor_chunks(
            target,
            ggml_type,
            gguf_dims(shape),
            nbytes,
            lambda tensor=tensor, tensor_quant=tensor_quant: _chunks(
                store, tensor, tensor_quant
            ),
        )

    return writer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quant", choices=("F32", "Q8_0"), default="F32")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    convert(args.model_dir, args.output, args.quant)


if __name__ == "__main__":
    main()
