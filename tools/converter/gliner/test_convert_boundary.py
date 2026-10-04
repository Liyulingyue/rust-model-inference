"""Unit tests for the boundary converter.

Two things here are worth a test and are not covered by the span-side tests in
``test_convert_gliner.py``: the ``boundary_head`` settings transcription, and
the two shape cross-checks that keep the emitted ``gliner2.boundary.*`` metadata
honest about the checkpoint it came from.

Where a real checkpoint is present the tests also assert against its published
``config.json`` and state dict, so a settings-drift upstream shows up here
instead of at inference time.
"""
import json
import unittest
from pathlib import Path

from tools.converter.gliner.convert_boundary import (
    BOUNDARY_FLAG_KEYS_REQUIRED, boundary_settings_metadata,
    check_pair_scorer_shapes, check_relation_scorer_shapes, encoder_source_contracts,
    resolve_boundary_encoder, tensor_contracts,
)

MODELS = Path(__file__).resolve().parents[3] / "models"
BASE_V1 = MODELS / "gliner2.5-base-v1"

# The smallest settings block that still satisfies the required-key list: the
# flags decide which optional sub-modules exist, so they are what the shape
# checks below turn into tensor expectations.
MINIMAL = {
    "boundary_dim": 128, "pair_dim": 256, "content_dim": 64, "record_dim": 32,
    "multihead_pair_compat_heads": 4, "candidate_budget": 64, "pool_size": 32,
    "start_top_k": 16, "end_top_k": 16, "ends_per_start": 4, "starts_per_end": 4,
    "pool_boundary_top_k": 8, "min_pool_per_query": 2, "boundary_top_k_max": 128,
    "use_inside_evidence": True, "relation_biaffine_content": True,
    "enable_span_content": True, "content_soft_max_pool": False,
    "query_conditioned_inside_weight": True, "endpoint_difference_features": True,
    "enable_rotary_endpoints": True, "reranker_endpoint_compat": True,
    "directional_relation_states": True, "rotary_base": 10000.0,
    "relation_pair_cap": 64, "relation_heads_per_type": 2,
    "relation_tails_per_type": 2, "overlap_policy": "longest",
}


class SettingsTest(unittest.TestCase):
    def test_transcribes_every_known_key(self):
        out = boundary_settings_metadata(MINIMAL)
        self.assertEqual(out["boundary_dim"], 128)
        self.assertIs(out["enable_span_content"], True)
        self.assertEqual(out["rotary_base"], 10000.0)
        self.assertIsInstance(out["rotary_base"], float)
        self.assertEqual(out["overlap_policy"], "longest")

    def test_a_non_object_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be an object"):
            boundary_settings_metadata(["not", "a", "dict"])

    def test_a_wrongly_typed_flag_is_rejected(self):
        # A bool where a bool belongs is fine; the check that matters is that a
        # number cannot stand in for a feature flag.
        with self.assertRaisesRegex(ValueError, "must be a bool"):
            boundary_settings_metadata({**MINIMAL, "enable_span_content": 1})

    def test_a_wrongly_typed_int_is_rejected(self):
        for bad in (1.5, "8", True):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "must be an int"):
                    boundary_settings_metadata({**MINIMAL, "boundary_dim": bad})

    def test_a_missing_required_setting_is_rejected(self):
        for key in BOUNDARY_FLAG_KEYS_REQUIRED:
            with self.subTest(key=key):
                settings = {k: v for k, v in MINIMAL.items() if k != key}
                with self.assertRaisesRegex(ValueError, "missing required settings"):
                    boundary_settings_metadata(settings)

    def test_a_pair_dim_that_does_not_divide_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not divisible"):
            boundary_settings_metadata({**MINIMAL, "multihead_pair_compat_heads": 7})


class PairScorerShapeTest(unittest.TestCase):
    """The flag-to-tensor correspondence the Rust loader relies on."""

    HIDDEN = 768

    def shapes_for(self, settings: dict, **overrides) -> dict:
        """The shape table the checkpoint *should* carry for `settings`."""
        d = settings["boundary_dim"]
        pair = settings["pair_dim"]
        content = settings["content_dim"]
        content_out = content * (2 if settings["content_soft_max_pool"] else 1)
        heads = settings["multihead_pair_compat_heads"]
        gate_out = pair // 2 if settings["enable_rotary_endpoints"] else pair
        shapes = {
            "boundary_head.pair_scorer.start_endpoint_projection.weight": (pair, d),
            "boundary_head.pair_scorer.end_endpoint_projection.weight": (pair, d),
            "boundary_head.pair_scorer.query_gate.weight": (gate_out, self.HIDDEN),
            "boundary_head.pair_scorer.compat_mix.weight": (1, heads),
            "boundary_head.pair_scorer.length_query_projection.weight": (3, self.HIDDEN),
        }
        if settings["query_conditioned_inside_weight"]:
            shapes["boundary_head.pair_scorer.inside_weight.weight"] = (1, self.HIDDEN)
        if settings["endpoint_difference_features"]:
            shapes["boundary_head.pair_scorer.endpoint_difference_projection.weight"] = (1, 2 * pair)
        if settings["enable_span_content"]:
            shapes["boundary_head.pair_scorer.content_pooler.value_projection.weight"] = (content, self.HIDDEN)
            shapes["boundary_head.pair_scorer.content_pooler.layer_norm.weight"] = (content_out,)
            shapes["boundary_head.pair_scorer.content_query_projection.weight"] = (content_out, self.HIDDEN)
            shapes["boundary_head.pair_scorer.content_bias.weight"] = (1, content_out)
        return shapes

    def test_a_matching_checkpoint_passes(self):
        settings = boundary_settings_metadata(MINIMAL)
        check_pair_scorer_shapes(settings, self.shapes_for(settings), self.HIDDEN)

    def test_a_missing_tensor_is_rejected(self):
        settings = boundary_settings_metadata(MINIMAL)
        shapes = self.shapes_for(settings)
        del shapes["boundary_head.pair_scorer.compat_mix.weight"]
        with self.assertRaisesRegex(ValueError, "does not carry it"):
            check_pair_scorer_shapes(settings, shapes, self.HIDDEN)

    def test_a_wrong_shape_is_rejected(self):
        settings = boundary_settings_metadata(MINIMAL)
        shapes = self.shapes_for(settings)
        shapes["boundary_head.pair_scorer.compat_mix.weight"] = (1, 5)
        with self.assertRaisesRegex(ValueError, "settings imply"):
            check_pair_scorer_shapes(settings, shapes, self.HIDDEN)

    def test_a_flag_that_disables_a_module_forbids_its_tensor(self):
        settings = boundary_settings_metadata({**MINIMAL, "enable_span_content": False})
        # The checkpoint still carries the content pooler: that is the mismatch.
        with self.assertRaisesRegex(ValueError, "enable_span_content is false"):
            check_pair_scorer_shapes(settings, self.shapes_for(MINIMAL), self.HIDDEN)

    def test_rotary_endpoints_halve_the_gate(self):
        with_gate = boundary_settings_metadata(MINIMAL)
        without = boundary_settings_metadata({**MINIMAL, "enable_rotary_endpoints": False})
        self.assertEqual(self.shapes_for(with_gate)["boundary_head.pair_scorer.query_gate.weight"],
                         (128, self.HIDDEN))
        self.assertEqual(self.shapes_for(without)["boundary_head.pair_scorer.query_gate.weight"],
                         (256, self.HIDDEN))

    def test_soft_max_pooling_doubles_the_content_width(self):
        settings = boundary_settings_metadata({**MINIMAL, "content_soft_max_pool": True})
        shapes = self.shapes_for(settings)
        self.assertEqual(shapes["boundary_head.pair_scorer.content_pooler.layer_norm.weight"], (128,))
        check_pair_scorer_shapes(settings, shapes, self.HIDDEN)


class RelationScorerShapeTest(unittest.TestCase):
    HIDDEN = 768

    def shapes_for(self, settings: dict) -> dict:
        h = self.HIDDEN
        relation_dim = 2 * h if settings["directional_relation_states"] else h
        shapes = {
            "relation_scorer.mlp.0.weight": (h, 4 * h + relation_dim + 2),
            "relation_scorer.mlp.0.bias": (h,),
            "relation_scorer.mlp.3.weight": (1, h),
            "relation_scorer.mlp.3.bias": (1,),
        }
        if settings["relation_biaffine_content"]:
            shapes["relation_scorer.head_content_projection.weight"] = (h, h)
            shapes["relation_scorer.tail_content_projection.weight"] = (h, h)
            shapes["relation_scorer.relation_content_gate.weight"] = (h, relation_dim)
            shapes["relation_scorer.content_linear.weight"] = (1, 2 * h + relation_dim)
        return shapes

    def test_a_matching_checkpoint_passes(self):
        settings = boundary_settings_metadata(MINIMAL)
        check_relation_scorer_shapes(settings, self.shapes_for(settings), self.HIDDEN)

    def test_the_dropout_index_is_three(self):
        # nn.Sequential(Linear, GELU, Dropout, Linear): Dropout takes an index,
        # so the output linear is mlp.3, not mlp.2.
        settings = boundary_settings_metadata(MINIMAL)
        shapes = self.shapes_for(settings)
        self.assertIn("relation_scorer.mlp.3.weight", shapes)
        self.assertNotIn("relation_scorer.mlp.2.weight", shapes)

    def test_directional_states_widen_the_relation_input(self):
        directional = boundary_settings_metadata(MINIMAL)
        averaged = boundary_settings_metadata({**MINIMAL, "directional_relation_states": False})
        self.assertEqual(self.shapes_for(directional)["relation_scorer.content_linear.weight"],
                         (1, 4 * self.HIDDEN))
        self.assertEqual(self.shapes_for(averaged)["relation_scorer.content_linear.weight"],
                         (1, 3 * self.HIDDEN))

    def test_a_disabled_flag_forbids_its_tensors(self):
        settings = boundary_settings_metadata({**MINIMAL, "relation_biaffine_content": False})
        with self.assertRaisesRegex(ValueError, "relation_biaffine_content is false"):
            check_relation_scorer_shapes(settings, self.shapes_for(MINIMAL), self.HIDDEN)

    def test_the_encoder_hidden_size_is_not_the_boundary_dim(self):
        # boundary_dim is 128 but the relation scorer is built on the encoder
        # hidden size; conflating the two is the bug this split guards against.
        settings = boundary_settings_metadata(MINIMAL)
        check_relation_scorer_shapes(settings, self.shapes_for(settings), self.HIDDEN)
        with self.assertRaises(ValueError):
            check_relation_scorer_shapes(settings, self.shapes_for(settings),
                                         settings["boundary_dim"])


class ContractTest(unittest.TestCase):
    def test_the_classifier_lands_at_index_three(self):
        encoder = resolve_boundary_encoder("microsoft/deberta-v3-base")
        shapes = tensor_contracts(encoder, 128012)
        self.assertIn("classifier.3.weight", shapes)
        self.assertNotIn("classifier.2.weight", shapes)
        self.assertEqual(shapes["classifier.3.weight"], (1, 1536))

    def test_each_boundary_encoder_resolves(self):
        for name, width in (("microsoft/deberta-v3-base", 768),
                            ("microsoft/mdeberta-v3-base", 768),
                            ("microsoft/deberta-v3-xsmall", 384)):
            with self.subTest(name=name):
                encoder = resolve_boundary_encoder(name)
                self.assertEqual(encoder["hidden_size"], width)
                self.assertEqual(encoder["hidden_act"], "gelu")

    def test_an_unknown_encoder_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "model_name"):
            resolve_boundary_encoder("microsoft/deberta-v3-large")

    def test_sources_cover_the_contract_once(self):
        encoder = resolve_boundary_encoder("microsoft/deberta-v3-base")
        sources = encoder_source_contracts(encoder)
        self.assertEqual(len(sources), len(tensor_contracts(encoder, 128012)))
        self.assertEqual(len(set(sources.values())), len(sources))
        self.assertEqual(sources["blk.0.attn_v.weight"],
                         "encoder.encoder.layer.0.attention.self.value_proj.weight")
        self.assertEqual(sources["classifier.3.weight"], "classifier.3.weight")

    def test_xsmall_is_a_genuinely_different_width(self):
        base = resolve_boundary_encoder("microsoft/deberta-v3-base")
        small = resolve_boundary_encoder("microsoft/deberta-v3-xsmall")
        base_shapes = tensor_contracts(base, 128012)
        small_shapes = tensor_contracts(small, 128012)
        self.assertEqual(base_shapes["token_embd.weight"][1], 768)
        self.assertEqual(small_shapes["token_embd.weight"][1], 384)
        self.assertEqual(base_shapes["blk.0.ffn_up.weight"], (3072, 768))
        self.assertEqual(small_shapes["blk.0.ffn_up.weight"], (1536, 384))


@unittest.skipUnless((BASE_V1 / "config.json").exists(), "gliner2.5-base-v1 not downloaded")
class PublishedCheckpointTest(unittest.TestCase):
    """Pin the transcription to the real checkpoint's published settings."""

    def setUp(self):
        self.config = json.loads((BASE_V1 / "config.json").read_text())
        self.settings = boundary_settings_metadata(self.config["boundary_head"])

    def test_the_published_block_transcribes(self):
        self.assertEqual(self.settings["boundary_dim"], 128)
        self.assertEqual(self.settings["pair_dim"], 128)
        self.assertEqual(self.settings["content_dim"], 64)
        self.assertEqual(self.settings["record_dim"], 128)
        self.assertEqual(self.settings["multihead_pair_compat_heads"], 8)
        self.assertEqual(self.settings["boundary_attention_window"], 128)
        self.assertEqual(self.settings["overlap_policy"], "flat")
        self.assertEqual(self.settings["rotary_base"], 10000.0)
        self.assertIs(self.settings["directional_relation_states"], True)
        self.assertIs(self.settings["relation_biaffine_content"], True)
        self.assertIs(self.settings["content_soft_max_pool"], False)

    def test_pair_dim_divides_by_the_compat_head_count(self):
        self.assertEqual(self.settings["pair_dim"] % self.settings["multihead_pair_compat_heads"], 0)

    def test_the_encoder_is_the_base_one(self):
        self.assertEqual(self.config["model_name"], "microsoft/deberta-v3-base")
        self.assertEqual(self.config["architecture"], "boundary")
        self.assertEqual(resolve_boundary_encoder(self.config["model_name"])["hidden_size"], 768)


@unittest.skipUnless((BASE_V1 / "model.safetensors").exists(), "gliner2.5-base-v1 not downloaded")
class PublishedStateDictTest(unittest.TestCase):
    """Run the cross-checks against the checkpoint's own tensor shapes.

    The synthetic tests above pin the *rule*; this one proves the published
    checkpoint actually satisfies it, which is the assertion that would have
    caught a settings drift at conversion time instead of at inference.
    """

    @classmethod
    def setUpClass(cls):
        import struct
        with (BASE_V1 / "model.safetensors").open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(length))
        cls.shapes = {k: tuple(v["shape"]) for k, v in header.items() if k != "__metadata__"}
        config = json.loads((BASE_V1 / "config.json").read_text())
        cls.settings = boundary_settings_metadata(config["boundary_head"])

    def test_the_pair_scorer_shapes_agree(self):
        check_pair_scorer_shapes(self.settings, self.shapes, 768)

    def test_the_relation_scorer_shapes_agree(self):
        check_relation_scorer_shapes(self.settings, self.shapes, 768)

    def test_the_hidden_size_comes_from_the_embedding_row(self):
        # The safetensors keys are the source names; the GGUF names come from
        # `encoder_source_contracts`.
        self.assertEqual(self.shapes["encoder.embeddings.word_embeddings.weight"][1], 768)


if __name__ == "__main__":
    unittest.main()
