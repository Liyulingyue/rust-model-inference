"""Convert the pinned native CosyVoice3 talker checkpoint to a component GGUF."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import torch

from tools.converter.sophon_realtime.convert import (
    FILE_TYPES,
    FORMATS,
    K_WORKERS,
    REVISION,
    add_streamed,
    target_type,
)
from tools.converter.utils.gguf import (
    GgufWriter,
    _read_gguf,
    gguf_dims,
    model_metadata_value,
)

PINNED_TEXT = dict(
    model_type="qwen2",
    hidden_size=896,
    intermediate_size=4864,
    num_hidden_layers=24,
    num_attention_heads=14,
    num_key_value_heads=2,
    vocab_size=151936,
    max_position_embeddings=32768,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    attention_dropout=0.0,
    use_sliding_window=False,
    tie_word_embeddings=True,
)
PINNED_FILES = {
    "llm.pt": "334b6899848142c6ebc982e483c94843cee0cf78555f04b54a30586eb3ce2992",
    "cosyvoice3.yaml": "f5a6b2c6f05139d0f18861a1fe506f751e787026b77c05f7e8fef9f8a4405965",
    "CosyVoice-BlankEN/config.json": "168aa1bd401abc3bc262ba15ba4e499627a8b4e006e9d050b47c22de20660185",
    "CosyVoice-BlankEN/tokenizer_config.json": "482bd979881423375ca5414e4e0d94cd7c5349dbb17fffd46b4d36d71e62a1bc",
    "CosyVoice-BlankEN/vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "CosyVoice-BlankEN/merges.txt": "ac8ff86a72bee70828fbc1119bc4398c6f3a9a6e490d7b0dbe917be025478bd0",
}

# Ordered additions from vLLM-Omni 0ea4e383's CosyVoice3Tokenizer, without importing vLLM.
SPECIAL_TOKENS = """
<|im_start|> <|im_end|> <|endofprompt|> [breath] <strong> </strong> [noise]
[laughter] [cough] [clucking] [accent] [quick_breath] <laughter> </laughter>
[hissing] [sigh] [vocalized-noise] [lipsmack] [mn] <|endofsystem|>
[AA] [AA0] [AA1] [AA2] [AE] [AE0] [AE1] [AE2] [AH] [AH0] [AH1] [AH2]
[AO] [AO0] [AO1] [AO2] [AW] [AW0] [AW1] [AW2] [AY] [AY0] [AY1] [AY2]
[B] [CH] [D] [DH] [EH] [EH0] [EH1] [EH2] [ER] [ER0] [ER1] [ER2]
[EY] [EY0] [EY1] [EY2] [F] [G] [HH] [IH] [IH0] [IH1] [IH2]
[IT] [IT0] [IT1] [IT2] [JH] [K] [L] [M] [N] [NG]
[OW] [OW0] [OW1] [OW2] [OY] [OY0] [OY1] [OY2] [P] [R] [S] [SH] [T] [TH]
[UH] [UH0] [UH1] [UH2] [UW] [UW0] [UW1] [UW2] [V] [W] [Y] [Z] [ZH]
[a] [ai] [an] [ang] [ao] [b] [c] [ch] [d] [e] [ei] [en] [eng] [f] [g] [h]
[i] [ian] [in] [ing] [iu] [ià] [iàn] [iàng] [iào] [iá] [ián] [iáng] [iáo]
[iè] [ié] [iòng] [ióng] [iù] [iú] [iā] [iān] [iāng] [iāo] [iē] [iě]
[iōng] [iū] [iǎ] [iǎn] [iǎng] [iǎo] [iǒng] [iǔ] [j] [k] [l] [m] [n]
[o] [ong] [ou] [p] [q] [r] [s] [sh] [t] [u] [uang] [ue] [un] [uo]
[uà] [uài] [uàn] [uàng] [uá] [uái] [uán] [uáng] [uè] [ué] [uì] [uí]
[uò] [uó] [uā] [uāi] [uān] [uāng] [uē] [uě] [uī] [uō] [uǎ] [uǎi]
[uǎn] [uǎng] [uǐ] [uǒ] [vè] [w] [x] [y] [z] [zh]
[à] [ài] [àn] [àng] [ào] [á] [ái] [án] [áng] [áo] [è] [èi] [èn] [èng] [èr]
[é] [éi] [én] [éng] [ér] [ì] [ìn] [ìng] [í] [ín] [íng] [ò] [òng] [òu]
[ó] [óng] [óu] [ù] [ùn] [ú] [ún] [ā] [āi] [ān] [āng] [āo]
[ē] [ēi] [ēn] [ēng] [ě] [ěi] [ěn] [ěng] [ěr] [ī] [īn] [īng] [ō] [ōng] [ōu]
[ū] [ūn] [ǎ] [ǎi] [ǎn] [ǎng] [ǎo] [ǐ] [ǐn] [ǐng] [ǒ] [ǒng] [ǒu]
[ǔ] [ǔn] [ǘ] [ǚ] [ǜ]
""".split()


def talker_specs(config, speech_token_size):
    h, f, heads, kv_heads, vocab = (
        config[key]
        for key in (
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
        )
    )
    kv = kv_heads * (h // heads)
    speech_vocab = speech_token_size + 200
    yield "llm.model.model.embed_tokens.weight", "token_embd.weight", (vocab, h)
    # Official speech forward bypasses the original Qwen2 text output head.
    yield "llm.model.lm_head.weight", None, (vocab, h)
    yield "llm.model.model.norm.weight", "output_norm.weight", (h,)
    yield "llm_decoder.weight", "output.weight", (speech_vocab, h)
    yield "speech_embedding.weight", "speech_embd.weight", (speech_vocab, h)
    suffixes = [
        ("input_layernorm.weight", "attn_norm.weight", (h,)),
        ("post_attention_layernorm.weight", "ffn_norm.weight", (h,)),
        ("self_attn.q_proj.weight", "attn_q.weight", (h, h)),
        ("self_attn.q_proj.bias", "attn_q.bias", (h,)),
        ("self_attn.k_proj.weight", "attn_k.weight", (kv, h)),
        ("self_attn.k_proj.bias", "attn_k.bias", (kv,)),
        ("self_attn.v_proj.weight", "attn_v.weight", (kv, h)),
        ("self_attn.v_proj.bias", "attn_v.bias", (kv,)),
        ("self_attn.o_proj.weight", "attn_output.weight", (h, h)),
        ("mlp.gate_proj.weight", "ffn_gate.weight", (f, h)),
        ("mlp.up_proj.weight", "ffn_up.weight", (f, h)),
        ("mlp.down_proj.weight", "ffn_down.weight", (h, f)),
    ]
    for i in range(config["num_hidden_layers"]):
        for source, target, shape in suffixes:
            yield f"llm.model.model.layers.{i}.{source}", f"blk.{i}.{target}", shape


class PtReader:
    def __init__(self, path):
        self.tensors = torch.load(
            Path(path).resolve(strict=True),
            weights_only=True,
            mmap=True,
            map_location="cpu",
        )
        if not isinstance(self.tensors, dict) or not all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in self.tensors.items()
        ):
            raise ValueError("talker checkpoint must be a flat tensor state dict")

    def dtype(self, name):
        return "F32"

    def read(self, name, first=0, rows=None):
        value = self.tensors[name]
        value = value[first:] if rows is None else value[first : first + rows]
        return value.detach().numpy()


def _yaml_scalar(raw, name, *, nested=False):
    # Read only the pinned scalar subset; never execute HyperPyYAML constructors.
    indent = "    " if nested else ""
    found = re.findall(rf"(?m)^{indent}{re.escape(name)}:[ \t]*([^\n#]+)", raw)
    if len(found) != 1:
        raise ValueError(f"missing/ambiguous talker YAML {name}")
    try:
        return json.loads(found[0].strip())
    except ValueError as error:
        raise ValueError(f"unsupported talker YAML {name}") from error


def talker_contract(model_dir, *, validate_pin=True):
    root = Path(model_dir)
    if (root / "talker" / "cosyvoice3.yaml").is_file():
        root = root / "talker"
    digests = {}
    for name, expected in PINNED_FILES.items():
        with (root / name).open("rb") as source:
            digests[name] = hashlib.file_digest(source, "sha256").hexdigest()
        if validate_pin and digests[name] != expected:
            raise ValueError(f"{name}: pinned talker source SHA256 mismatch")
    config = json.loads((root / "CosyVoice-BlankEN" / "config.json").read_text())
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "num_hidden_layers",
        "vocab_size",
        "max_position_embeddings",
    ):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"invalid talker config {key}")
    if (
        config.get("model_type") != "qwen2"
        or config.get("hidden_act") != "silu"
        or config["hidden_size"] % config["num_attention_heads"]
        or config["num_attention_heads"] % config["num_key_value_heads"]
        or config.get("use_sliding_window") is not False
        or config.get("attention_dropout") != 0.0
        or not isinstance(config.get("rope_theta"), (int, float))
        or not np.isfinite(config["rope_theta"])
        or config["rope_theta"] <= 0
        or not isinstance(config.get("rms_norm_eps"), (int, float))
        or not np.isfinite(config["rms_norm_eps"])
        or config["rms_norm_eps"] <= 0
    ):
        raise ValueError("unsupported talker Qwen2 config")
    if validate_pin and any(
        config.get(key) != value for key, value in PINNED_TEXT.items()
    ):
        raise ValueError("pinned talker Qwen2 config mismatch")
    raw_yaml = (root / "cosyvoice3.yaml").read_text()
    settings = {
        key: _yaml_scalar(raw_yaml, key)
        for key in (
            "sample_rate",
            "llm_input_size",
            "llm_output_size",
            "token_frame_rate",
            "token_mel_ratio",
        )
    }
    speech = _yaml_scalar(raw_yaml, "speech_token_size", nested=True)
    mix = _yaml_scalar(raw_yaml, "mix_ratio", nested=True)
    if (
        settings["llm_input_size"] != config["hidden_size"]
        or settings["llm_output_size"] != config["hidden_size"]
        or any(type(value) is not int or value <= 0 for value in settings.values())
        or type(speech) is not int
        or speech <= 0
        or not isinstance(mix, list)
        or len(mix) != 2
        or any(type(value) is not int or value <= 0 for value in mix)
        or (validate_pin and speech != 6561)
    ):
        raise ValueError("unsupported talker speech config")
    settings.update(
        speech_token_size=speech,
        speech_vocab_size=speech + 200,
        sos_token_id=speech,
        eos_token_id=speech + 1,
        task_token_id=speech + 2,
        fill_token_id=speech + 3,
        mix_ratio=mix,
    )
    reader = PtReader(root / "llm.pt")
    specs = list(talker_specs(config, speech))
    if set(reader.tensors) != {name for name, _, _ in specs}:
        raise ValueError("missing/unknown talker component tensor")
    for name, _, shape in specs:
        value = reader.tensors[name]
        if (
            tuple(value.shape) != shape
            or value.dtype != torch.float32
            or value.layout != torch.strided
            or not value.is_contiguous()
        ):
            raise ValueError(f"{name}: invalid talker tensor shape/dtype/layout")
    return root, config, settings, raw_yaml, digests, reader, specs


def tokenizer_metadata(root, vocab_size):
    text = root / "CosyVoice-BlankEN"
    raw_vocab = (text / "vocab.json").read_text()
    raw_config = (text / "tokenizer_config.json").read_text()
    raw_merges = (text / "merges.txt").read_text()
    vocab = json.loads(raw_vocab)
    config = json.loads(raw_config)
    if (
        not isinstance(vocab, dict)
        or not vocab
        or config.get("tokenizer_class") != "Qwen2Tokenizer"
        or config.get("add_prefix_space", False)
        or config.get("split_special_tokens", False)
    ):
        raise ValueError("unsupported talker tokenizer config")
    tokens = [None] * vocab_size
    types = [5] * vocab_size
    for piece, index in vocab.items():
        if (
            not isinstance(piece, str)
            or type(index) is not int
            or not 0 <= index < vocab_size
            or tokens[index] is not None
        ):
            raise ValueError("invalid talker tokenizer vocab id")
        tokens[index], types[index] = piece, 1
    for index, token in config.get("added_tokens_decoder", {}).items():
        index = int(index)
        if (
            not 0 <= index < vocab_size
            or tokens[index] not in (None, token["content"])
            or any(
                token.get(flag, False)
                for flag in ("normalized", "single_word", "lstrip", "rstrip")
            )
        ):
            raise ValueError("unsupported talker added token id/flags")
        tokens[index] = token["content"]
        types[index] = 3 if token.get("special") else 4
    occupied = [i for i, value in enumerate(tokens) if value is not None]
    if occupied != list(range(len(occupied))):
        raise ValueError("noncontiguous talker tokenizer ids")
    indices = {piece: i for i, piece in enumerate(tokens) if piece is not None}
    if len(indices) != len(occupied):
        raise ValueError("duplicate talker tokenizer pieces")
    next_id = len(occupied)
    for piece in ("<|endoftext|>", *SPECIAL_TOKENS):
        if piece not in indices:
            if next_id >= vocab_size:
                raise ValueError("talker tokenizer exceeds text embedding vocabulary")
            indices[piece] = next_id
            tokens[next_id] = piece
            next_id += 1
        types[indices[piece]] = 3
    merges = [
        line for line in raw_merges.splitlines() if line and not line.startswith("#")
    ]
    if any(len(line.split(" ")) != 2 for line in merges) or len(set(merges)) != len(
        merges
    ):
        raise ValueError("invalid talker tokenizer merges")
    return {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": [
            piece if piece is not None else f"<|reserved_{i}|>"
            for i, piece in enumerate(tokens)
        ],
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.merges": merges,
        "tokenizer.ggml.normalizer.nfc": False,
        "tokenizer.ggml.add_bos_token": False,
        "tokenizer.ggml.add_eos_token": False,
        "tokenizer.ggml.eos_token_id": indices["<|endoftext|>"],
        "tokenizer.ggml.padding_token_id": indices["<|endoftext|>"],
        "tokenizer.chat_template": config.get("chat_template", ""),
        "sophon_realtime.talker_vocab_json": raw_vocab,
        "sophon_realtime.talker_tokenizer_config_json": raw_config,
        "sophon_realtime.talker_merges_txt": raw_merges,
        "sophon_realtime.talker_special_tokens_json": json.dumps(
            {piece: index for piece, index in indices.items() if types[index] == 3},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "sophon_realtime.talker.effective_text_token_count": next_id,
    }


def convert_talker(model_dir, output, *, quant="f32", validate_pin=True):
    if quant not in FORMATS:
        raise ValueError(f"unknown format {quant}")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    root, config, settings, raw_yaml, digests, reader, specs = talker_contract(
        model_dir, validate_pin=validate_pin
    )
    tokens = tokenizer_metadata(root, config["vocab_size"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    pool = None
    try:
        pool = (
            ProcessPoolExecutor(max_workers=K_WORKERS)
            if quant in ("q4_k", "q5_k", "q6_k") and K_WORKERS > 1
            else None
        )
        writer = GgufWriter(temporary)
        head_dim = config["hidden_size"] // config["num_attention_heads"]
        metadata = {
            "general.architecture": "sophon_realtime_talker",
            "general.name": "Sophon-Realtime-32B-Tag CosyVoice3 talker",
            "general.file_type": FILE_TYPES[quant],
            "sophon_realtime.component": "talker",
            "sophon_realtime.source_revision": REVISION,
            "sophon_realtime.source_precision": "F32",
            "sophon_realtime.runtime_quantization": "GGUF weight-only",
            "sophon_realtime.config_json": json.dumps(config, separators=(",", ":")),
            "sophon_realtime.talker_config_json": json.dumps(
                settings, separators=(",", ":")
            ),
            "sophon_realtime.talker_yaml": raw_yaml,
            "sophon_realtime.talker_source_hashes_json": json.dumps(
                digests, sort_keys=True
            ),
            "sophon_realtime.talker_weight_source": "talker/llm.pt",
            "sophon_realtime.requested_quant": quant,
            "sophon_realtime.tensor_types_json": json.dumps(
                {
                    target: target_type(shape, quant)
                    for _, target, shape in specs
                    if target is not None
                },
                sort_keys=True,
            ),
            "sophon_realtime.omitted_tensors_json": json.dumps(
                {
                    "llm.model.lm_head.weight": "unused original text LM head; official speech forward uses llm_decoder.weight"
                }
            ),
            "sophon_realtime.talker.has_qkv_bias": True,
            "sophon_realtime.talker.has_qk_norm": False,
            "qwen2.block_count": config["num_hidden_layers"],
            "qwen2.context_length": config["max_position_embeddings"],
            "qwen2.embedding_length": config["hidden_size"],
            "qwen2.feed_forward_length": config["intermediate_size"],
            "qwen2.attention.head_count": config["num_attention_heads"],
            "qwen2.attention.head_count_kv": config["num_key_value_heads"],
            "qwen2.attention.key_length": head_dim,
            "qwen2.attention.value_length": head_dim,
            "qwen2.attention.layer_norm_rms_epsilon": config["rms_norm_eps"],
            "qwen2.rope.dimension_count": head_dim,
            "qwen2.rope.freq_base": float(config["rope_theta"]),
            **tokens,
        }
        for key, value in settings.items():
            metadata["sophon_realtime.talker." + key] = value
        for key, value in metadata.items():
            writer.add_meta(key, model_metadata_value(value))
        for source, target, shape in specs:
            if target is not None:
                add_streamed(writer, reader, source, target, shape, quant, pool)
        writer.write()
        actual, directory = _read_gguf(temporary)
        if (
            actual.get("general.architecture") != "sophon_realtime_talker"
            or actual.get("sophon_realtime.component") != "talker"
            or actual.get("sophon_realtime.source_revision") != REVISION
            or set(directory)
            != {target for _, target, _ in specs if target is not None}
            or any(
                directory[target][:2] != (target_type(shape, quant), gguf_dims(shape))
                for _, target, shape in specs
                if target is not None
            )
        ):
            raise ValueError("talker GGUF readback mismatch")
        os.link(temporary, output)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quant", choices=FORMATS, default="f32")
    args = parser.parse_args()
    convert_talker(args.model_dir, args.output, quant=args.quant)


if __name__ == "__main__":
    main()
