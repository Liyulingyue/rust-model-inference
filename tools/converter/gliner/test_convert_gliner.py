"""Unit tests for the GLiNER converters.

Most of the interesting machinery now lives in ``common.py`` (the protobuf
``spm.model`` reader, the encoder shape contract, the metadata blocks), and it
is shared by both converters, so it is tested here once. What stays per-variant
is the size table, the encoder tensor *name* map, the boundary settings
transcription, and the added-token rules.
"""
import unittest

from tools.converter.gliner.common import (
    encoder_tensor_contracts, fast_tokenizer_pieces, parse_spm, resolve_encoder,
)
from tools.converter.gliner.convert_gliner import (
    SPECIAL_TOKENS, added_tokens, source_contracts, tensor_contracts, validate_config,
)

CONFIG = {"architecture": "span", "model_name": "microsoft/deberta-v3-large",
          "token_pooling": "first", "config_version": 3}

TOKENIZER_CONFIG = {
    "added_tokens_decoder": {
        "0": {"content": "[PAD]"}, "1": {"content": "[CLS]"},
        "2": {"content": "[SEP]"}, "3": {"content": "[UNK]"},
        **{str(128000 + i): {"content": token} for i, token in enumerate(SPECIAL_TOKENS)},
    }
}
SPM_PREFIX = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]


def spm_of(count: int) -> list[str]:
    return SPM_PREFIX + [f"piece{i}" for i in range(count - 4)]


def encoder_of(model_name: str) -> dict:
    return validate_config({"model_type": "extractor", **CONFIG, "model_name": model_name})


class ConfigTest(unittest.TestCase):
    def test_accepts_the_pinned_config(self):
        self.assertEqual(encoder_of("microsoft/deberta-v3-large")["hidden_size"], 1024)

    def test_resolves_each_supported_size(self):
        for name, width in (("microsoft/deberta-v3-large", 1024),
                            ("microsoft/deberta-v3-base", 768),
                            ("microsoft/mdeberta-v3-base", 768)):
            with self.subTest(name=name):
                self.assertEqual(encoder_of(name)["hidden_size"], width)

    def test_rejects_an_unknown_backbone(self):
        with self.assertRaisesRegex(ValueError, "model_name"):
            encoder_of("microsoft/deberta-v3-xsmall")

    def test_rejects_a_foreign_model_type(self):
        with self.assertRaisesRegex(ValueError, "model_type"):
            validate_config({"model_type": "boundary", **CONFIG})

    def test_rejects_a_drifted_optional_field(self):
        for key, bad in (("architecture", "boundary"), ("token_pooling", "mean"),
                         ("config_version", 2)):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    validate_config({"model_type": "extractor", **CONFIG, key: bad})


class SharedEncoderTest(unittest.TestCase):
    """`common.resolve_encoder` and `common.encoder_tensor_contracts`."""

    SIZES = {
        "microsoft/deberta-v3-base": {"hidden_size": 768, "num_hidden_layers": 12,
                                      "num_attention_heads": 12, "intermediate_size": 3072},
    }

    def test_common_fields_are_merged_under_the_size(self):
        encoder = resolve_encoder("microsoft/deberta-v3-base", self.SIZES, "test")
        # The relative-attention block is what resolves identically for every v3
        # size, so it is not in the size table.
        self.assertEqual(encoder["position_buckets"], 256)
        self.assertEqual(encoder["pos_att_type"], "p2c|c2p")
        self.assertEqual(encoder["hidden_act"], "gelu")

    def test_a_size_entry_cannot_override_a_common_field(self):
        sizes = {"m": {**self.SIZES["microsoft/deberta-v3-base"], "position_buckets": 8}}
        encoder = resolve_encoder("m", sizes, "test")
        self.assertEqual(encoder["position_buckets"], 8)

    def test_contract_covers_every_block_tensor(self):
        encoder = resolve_encoder("microsoft/deberta-v3-base", self.SIZES, "test")
        shapes = encoder_tensor_contracts(encoder, 128011, 2)
        d, f = 768, 3072
        # 10 non-block tensors + 16 tensors per layer.
        self.assertEqual(len(shapes), 10 + 12 * 16)
        self.assertEqual(shapes["token_embd.weight"], (128011, d))
        # pos_ebd_size = position_buckets * 2, not max_position_embeddings.
        self.assertEqual(shapes["rel_embeddings.weight"], (512, d))
        # classifier.0 is hidden * 2 wide, not the FFN width.
        self.assertEqual(shapes["classifier.0.weight"], (1536, d))
        self.assertEqual(shapes["blk.0.ffn_up.weight"], (f, d))
        self.assertEqual(shapes["blk.11.ffn_down.bias"], (d,))

    def test_the_classifier_index_is_the_only_difference(self):
        encoder = resolve_encoder("microsoft/deberta-v3-base", self.SIZES, "test")
        span = encoder_tensor_contracts(encoder, 128011, 2)
        boundary = encoder_tensor_contracts(encoder, 128011, 3)
        self.assertEqual(span["classifier.2.weight"], (1, 1536))
        self.assertEqual(boundary["classifier.3.weight"], (1, 1536))
        self.assertNotIn("classifier.2.weight", boundary)
        self.assertEqual(len(span), len(boundary))
        # Everything except the classifier tail is byte-identical.
        shared = {k: v for k, v in span.items() if not k.startswith("classifier.2")}
        self.assertEqual(shared, {k: v for k, v in boundary.items()
                                  if not k.startswith("classifier.3")})


class SpanContractTest(unittest.TestCase):
    def test_tensor_contract_matches_the_checkpoint(self):
        shapes = tensor_contracts(encoder_of("microsoft/deberta-v3-large"), 128011)
        self.assertEqual(len(shapes), 10 + 24 * 16)
        self.assertEqual(shapes["token_embd.weight"], (128011, 1024))
        self.assertEqual(shapes["classifier.2.weight"], (1, 2048))
        self.assertEqual(shapes["blk.23.ffn_down.bias"], (1024,))

    def test_sources_cover_every_tensor_once(self):
        encoder = encoder_of("microsoft/deberta-v3-large")
        sources = source_contracts(encoder, 128011)
        self.assertEqual(len(sources), len(tensor_contracts(encoder, 128011)))
        self.assertEqual(len(set(sources.values())), len(sources))
        self.assertEqual(sources["blk.0.attn_q.weight"],
                         "encoder.encoder.layer.0.attention.self.query_proj.weight")
        self.assertEqual(sources["blk.0.attn_k.bias"],
                         "encoder.encoder.layer.0.attention.self.key_proj.bias")
        self.assertEqual(sources["blk.5.attn_out_norm.weight"],
                         "encoder.encoder.layer.5.attention.output.LayerNorm.weight")
        self.assertEqual(sources["rel_norm.bias"], "encoder.encoder.LayerNorm.bias")
        # No conv / position / token_type embeddings: position_biased_input=false
        # and type_vocab_size=0 on the pinned base config.
        self.assertFalse(any("position" in key or "conv" in key for key in sources.values()))


class FastTokenizerTest(unittest.TestCase):
    def fast_of(self, count: int = 128000, offset: int = 0) -> dict:
        return {
            "model": {"type": "Unigram", "vocab": [[f"piece{i}", 0.0] for i in range(count)]},
            "added_tokens": [
                {"content": token, "id": count + offset + i}
                for i, token in enumerate(SPECIAL_TOKENS)
            ],
        }

    def test_span_sits_at_the_vocabulary_end(self):
        self.assertEqual(len(fast_tokenizer_pieces(self.fast_of(), SPECIAL_TOKENS, 0)),
                         128000)

    def test_a_drifted_special_id_is_rejected(self):
        fast = self.fast_of()
        fast["added_tokens"][7]["id"] += 1
        with self.assertRaisesRegex(ValueError, "tokenizer.json places"):
            fast_tokenizer_pieces(fast, SPECIAL_TOKENS, 0)

    def test_the_boundary_offset_is_one(self):
        # The boundary list omits [MASK], which ships inside the base vocab, so
        # its block starts one id later.
        boundary_tokens = SPECIAL_TOKENS[1:]
        fast = {
            "model": {"type": "Unigram", "vocab": [[f"piece{i}", 0.0] for i in range(128000)]},
            "added_tokens": [
                {"content": token, "id": 128000 + 1 + i}
                for i, token in enumerate(boundary_tokens)
            ],
        }
        self.assertEqual(len(fast_tokenizer_pieces(fast, boundary_tokens, 1)), 128000)
        # The same file read with the span offset is off by one and must fail.
        with self.assertRaisesRegex(ValueError, "tokenizer.json places"):
            fast_tokenizer_pieces(fast, boundary_tokens, 0)

    def test_rejects_a_non_unigram_vocabulary(self):
        with self.assertRaisesRegex(ValueError, "unsupported tokenizer.json"):
            fast_tokenizer_pieces({"model": {"type": "BPE", "vocab": [["a", 0.0]]}},
                                  SPECIAL_TOKENS, 0)


class AddedTokenTest(unittest.TestCase):
    def test_appends_after_the_spm_vocab(self):
        added = added_tokens(TOKENIZER_CONFIG, spm_of(128000))
        self.assertEqual(added["[P]"], 128003)
        self.assertEqual(added["[L]"], 128007)
        self.assertEqual(len(added), len(SPECIAL_TOKENS))

    def test_a_misaligned_id_block_is_rejected(self):
        broken = {**TOKENIZER_CONFIG,
                  "added_tokens_decoder": dict(TOKENIZER_CONFIG["added_tokens_decoder"])}
        broken["added_tokens_decoder"]["128010"] = {"content": "[DESCRIPTION]"}
        broken["added_tokens_decoder"]["128011"] = {"content": "[SEP_STRUCT]"}
        with self.assertRaisesRegex(ValueError, "contiguous"):
            added_tokens(broken, spm_of(128000))

    def test_a_base_special_that_drifted_is_rejected(self):
        broken = {**TOKENIZER_CONFIG,
                  "added_tokens_decoder": dict(TOKENIZER_CONFIG["added_tokens_decoder"])}
        broken["added_tokens_decoder"]["1"] = {"content": "[BOS]"}
        with self.assertRaisesRegex(ValueError, r"\[CLS\]"):
            added_tokens(broken, spm_of(128000))

    def test_the_two_declaration_styles_must_agree(self):
        # Same token, two different ids: `tokenizer_config.json` and
        # `tokenizer.json` are both authoritative when present, so a drift has
        # to be an error rather than one silently winning.
        fast = {"added_tokens": [{"content": "[MASK]", "id": 128001}]}
        config = {"added_tokens_decoder": {"128000": {"content": "[MASK]"}}}
        with self.assertRaisesRegex(ValueError, "disagree"):
            added_tokens(config, spm_of(128000), fast)

    def test_a_missing_special_is_rejected(self):
        config = {"added_tokens_decoder": {
            index: entry for index, entry
            in TOKENIZER_CONFIG["added_tokens_decoder"].items()
            if entry["content"] != "[EXAMPLE]"}}
        with self.assertRaisesRegex(ValueError, "missing added tokens"):
            added_tokens(config, spm_of(128000))


class SpmParseTest(unittest.TestCase):
    @staticmethod
    def varint(value: int) -> bytes:
        out = bytearray()
        while True:
            byte = value & 0x7F
            value >>= 7
            out.append(byte | (0x80 if value else 0))
            if not value:
                return bytes(out)

    @classmethod
    def tag(cls, field: int, wire: int) -> bytes:
        return cls.varint((field << 3) | wire)

    @classmethod
    def blob(cls, field: int, payload: bytes) -> bytes:
        return cls.tag(field, 2) + cls.varint(len(payload)) + payload

    def test_parses_a_synthetic_model_proto(self):
        piece = self.blob(1, b"\x01my") + self.tag(2, 5) + b"\x00\x00\x80?" + self.tag(3, 0) + self.varint(1)
        unknown = self.blob(1, b"[UNK]") + self.tag(3, 0) + self.varint(2)
        trainer = self.tag(35, 0) + self.varint(1) + self.tag(24, 0) + self.varint(0)
        norm = self.blob(1, b"nmt_nfkc") + self.blob(2, b"\x04\x00\x00\x00abcd") + self.tag(3, 0) + self.varint(1)
        model = self.blob(1, unknown) + self.blob(1, piece) + self.blob(2, trainer) + self.blob(3, norm)

        spm = parse_spm(model)
        self.assertEqual(spm["pieces"], ["[UNK]", "\x01my"])
        self.assertAlmostEqual(spm["scores"][1], 1.0)
        self.assertEqual(spm["types"], [2, 1])
        self.assertEqual(spm["normalizer"]["name"], "nmt_nfkc")
        self.assertEqual(spm["normalizer"]["charsmap"], b"\x04\x00\x00\x00abcd")
        self.assertTrue(spm["normalizer"]["byte_fallback"])
        self.assertFalse(spm["normalizer"]["treat_whitespace_as_suffix"])
        # Absent flags fall back to the protobuf defaults.
        self.assertTrue(spm["normalizer"]["remove_extra_whitespaces"])
        self.assertTrue(spm["normalizer"]["escape_whitespaces"])

    def test_rejects_a_model_without_pieces(self):
        with self.assertRaisesRegex(ValueError, "no pieces"):
            parse_spm(b"")


if __name__ == "__main__":
    unittest.main()
