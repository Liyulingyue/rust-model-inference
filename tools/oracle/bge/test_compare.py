import unittest
from tools.oracle.bge.compare import require_exact_output_path


class OutputScopeTest(unittest.TestCase):
    def test_lossy_operations_and_visual_graphs_are_excluded(self):
        for arch in ["bert", "gemma2", "bge-vl-clip", "qwen2vl"]:
            with self.assertRaisesRegex(ValueError, "whole-pipeline"):
                require_exact_output_path(
                    {"general.architecture": arch, "general.file_type": 0},
                    {"x": (0, (), 0, 0)},
                )
        for metadata, tensors in [
            (
                {"general.architecture": "qwen3", "general.file_type": 7},
                {"x": (8, (), 0, 0)},
            ),
            (
                {"general.architecture": "qwen3", "general.file_type": 0},
                {"x": (1, (), 0, 0)},
            ),
            (
                {
                    "general.architecture": "llama",
                    "general.file_type": 0,
                    "bge.llava_next": True,
                },
                {"x": (0, (), 0, 0)},
            ),
        ]:
            with self.assertRaisesRegex(ValueError, "whole-pipeline"):
                require_exact_output_path(metadata, tensors)

    def test_only_scalar_f32_text_decoder_outputs_are_eligible(self):
        for arch in ["llama", "qwen2", "qwen3"]:
            require_exact_output_path(
                {"general.architecture": arch, "general.file_type": 0},
                {"x": (0, (), 0, 0)},
            )


if __name__ == "__main__":
    unittest.main()
