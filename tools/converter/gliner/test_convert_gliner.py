import unittest

from tools.converter.gliner.convert_gliner import (
    ENCODER,
    added_tokens,
    parse_spm,
    source_contracts,
    tensor_contracts,
    validate_config,
)

CONFIG = {"architecture": "span", "model_name": "microsoft/deberta-v3-large",
          "token_pooling": "first", "config_version": 3}

TOKENIZER_CONFIG = {
    "added_tokens_decoder": {
        "0": {"content": "[PAD]"}, "1": {"content": "[CLS]"},
        "2": {"content": "[SEP]"}, "3": {"content": "[UNK]"},
        **{str(128000 + i): {"content": token} for i, token in enumerate(
            ["[MASK]", "[SEP_STRUCT]", "[SEP_TEXT]", "[P]", "[C]", "[E]", "[R]", "[L]",
             "[EXAMPLE]", "[OUTPUT]", "[DESCRIPTION]"])},
    }
}
SPM_PREFIX = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]


def spm_of(count: int) -> list[str]:
    return SPM_PREFIX + [f"piece{i}" for i in range(count - 4)]


class ConfigTest(unittest.TestCase):
    def test_accepts_the_pinned_config(self):
        validate_config(CONFIG)

    def test_rejects_a_foreign_backbone(self):
        for key, bad in (("model_name", "microsoft/deberta-v3-base"),
                         ("architecture", "boundary"),
                         ("token_pooling", "mean"),
                         ("config_version", 2)):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    validate_config({**CONFIG, key: bad})


class ContractTest(unittest.TestCase):
    def test_tensor_contract_matches_the_checkpoint(self):
        shapes = tensor_contracts(128011)
        self.assertEqual(len(shapes), 10 + 24 * 16)
        self.assertEqual(shapes["token_embd.weight"], (128011, 1024))
        # pos_ebd_size = position_buckets * 2, not max_position_embeddings.
        self.assertEqual(shapes["rel_embeddings.weight"], (512, 1024))
        # classifier.0 is hidden * 2 wide, not the FFN width.
        self.assertEqual(shapes["classifier.0.weight"], (2048, 1024))
        self.assertEqual(shapes["classifier.2.weight"], (1, 2048))
        self.assertEqual(shapes["blk.0.ffn_up.weight"], (4096, 1024))
        self.assertEqual(shapes["blk.23.ffn_down.bias"], (1024,))
        self.assertEqual(ENCODER["position_buckets"] * 2, shapes["rel_embeddings.weight"][0])

    def test_sources_cover_every_tensor_once(self):
        sources = source_contracts(128011)
        self.assertEqual(len(sources), len(tensor_contracts(128011)))
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


class AddedTokenTest(unittest.TestCase):
    def test_appends_after_the_spm_vocab(self):
        added = added_tokens(TOKENIZER_CONFIG, spm_of(128000))
        self.assertEqual(added["[P]"], 128003)
        self.assertEqual(added["[L]"], 128007)
        self.assertEqual(len(added), 11)

    def test_rejects_a_misaligned_id_block(self):
        broken = {**TOKENIZER_CONFIG}
        broken["added_tokens_decoder"] = dict(TOKENIZER_CONFIG["added_tokens_decoder"])
        broken["added_tokens_decoder"]["128010"] = {"content": "[DESCRIPTION]"}
        broken["added_tokens_decoder"]["128011"] = {"content": "[SEP_STRUCT]"}
        with self.assertRaisesRegex(ValueError, "contiguous"):
            added_tokens(broken, spm_of(128000))

    def test_rejects_a_base_special_that_drifted(self):
        broken = {**TOKENIZER_CONFIG}
        broken["added_tokens_decoder"] = dict(TOKENIZER_CONFIG["added_tokens_decoder"])
        broken["added_tokens_decoder"]["1"] = {"content": "[BOS]"}
        with self.assertRaisesRegex(ValueError, r"\[CLS\]"):
            added_tokens(broken, spm_of(128000))


class SpmParseTest(unittest.TestCase):
    def test_parses_a_synthetic_model_proto(self):
        def varint(value: int) -> bytes:
            out = bytearray()
            while True:
                byte = value & 0x7F
                value >>= 7
                out.append(byte | (0x80 if value else 0))
                if not value:
                    return bytes(out)

        def tag(field: int, wire: int) -> bytes:
            return varint((field << 3) | wire)

        def blob(field: int, payload: bytes) -> bytes:
            return tag(field, 2) + varint(len(payload)) + payload

        piece = blob(1, b"\x01my") + tag(2, 5) + b"\x00\x00\x80?" + tag(3, 0) + varint(1)
        unknown = blob(1, b"[UNK]") + tag(3, 0) + varint(2)
        trainer = tag(35, 0) + varint(1) + tag(24, 0) + varint(0)
        norm = blob(1, b"nmt_nfkc") + blob(2, b"\x04\x00\x00\x00abcd") + tag(3, 0) + varint(1)
        model = blob(1, unknown) + blob(1, piece) + blob(2, trainer) + blob(3, norm)

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
