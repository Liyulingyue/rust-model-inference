"""Pinned official causal graph smoke on a tiny deterministic configuration."""
from pathlib import Path
import tempfile
import unittest
import sys

import torch

from trace_official import (REFERENCE, Trace, _guard_reference, _trace_denoiser,
                            _trace_bidirectional_small, _trace_text_small,
                            _trace_vae_small)
from scalar import Scalar


class OfficialSmallGraph(unittest.TestCase):
    def test_cache_camera_action_path_without_weights(self):
        _guard_reference()
        sys.path.insert(0, str(REFERENCE))
        from models.action import add_action_parameters
        from models.causal_model import CausalWanModel
        from models.prope import add_prope_parameters

        # Smaller dimensions keep this as a quick graph/dispatch fixture. The
        # module classes and cache/camera/action control flow are the pinned
        # official classes; no Rust implementation or downloaded weight is used.
        torch.manual_seed(7)
        model = CausalWanModel(
            model_type="t2v", patch_size=(1, 2, 2), text_len=4,
            in_dim=4, dim=8, ffn_dim=16, freq_dim=4, text_dim=8,
            out_dim=4, num_heads=2, num_layers=1,
            local_attn_size=4, sink_size=1, qk_norm=True,
            cross_attn_norm=True,
        )
        add_prope_parameters(model, zero_init=False)
        add_action_parameters(model, action_dim=64, num_embodiments=2)
        model.add_action_head(64)
        model.float().eval()
        with tempfile.TemporaryDirectory() as tmp:
            trace = Trace(Path(tmp) / "trace")
            context = torch.zeros(1, 4, 8, dtype=torch.float32)
            with Scalar(Path(tmp) / "compile") as scalar:
                _trace_denoiser(model, trace, scalar, context, frames=2, latent_hw=(2, 2))
                _trace_bidirectional_small(trace, scalar)
                _trace_vae_small(trace, scalar)
                _trace_text_small(trace, scalar)
            self.assertGreaterEqual(len(trace.records), 2)
            names = [r["name"] for r in trace.records]
            self.assertIn("astronex.denoiser.input", names)
            self.assertIn("astronex.denoiser.output.cache0", names)
            self.assertIn("astronex.denoiser.output.cache1", names)
            self.assertIn("astronex.denoiser.output.bidirectional", names)
            self.assertIn("astronex.vae.latent", names)
            self.assertIn("astronex.vae.pixel", names)
            self.assertIn("astronex.text.output", names)


if __name__ == "__main__":
    unittest.main()
