import unittest
import numpy as np

from tools.converter.wevisone_vl.convert_wevisone_vl import (
    map_text_name, validate_config, validate_tokenizer, _patch_embeddings,
)


class ContractTests(unittest.TestCase):
    def test_text_mapping_rejects_pixel_and_unknown_tensors(self):
        self.assertEqual(map_text_name("model.language_model.layers.2.self_attn.q_proj.weight"),
                         "blk.2.attn_q.weight")
        self.assertEqual(map_text_name("lm_head.weight"), "output.weight")
        for name in ["decoder.weight", "model.visual.blocks.0.attn.qkv.weight",
                     "model.language_model.layers.2.bad.weight"]:
            with self.assertRaises(ValueError):
                map_text_name(name)

    def test_temporal_patch_weights_are_split_without_channel_reordering(self):
        values=np.arange(2*3*2*2*2,dtype=np.float32).reshape(2,3,2,2,2)
        class Reader:
            def f32(self,name): return values
        tensors=_patch_embeddings(Reader())
        for frame,(name,shape,raw) in enumerate(tensors):
            self.assertEqual(shape,(2,12))
            np.testing.assert_array_equal(np.frombuffer(raw,dtype='<f4').reshape(shape),values[:,:,frame].reshape(2,12))

    def test_padding_row_does_not_invent_a_token(self):
        tokenizer={"model":{"vocab":{str(i):i for i in range(152679)}},"added_tokens":[{"id":152678,"content":"<OBJ>"}]}
        for i in range(1001):tokenizer["model"]["vocab"][f"<{i}>"]=151669+i
        # Remove the synthetic aliases for coordinate IDs before checking the real contract.
        for i in range(151669,152670):tokenizer["model"]["vocab"].pop(str(i),None)
        tokens,types=validate_tokenizer(tokenizer)
        self.assertEqual((len(tokens),tokens[-1],types[-1]),(152680,"",5))

    def test_unknown_architecture_is_not_a_qwen_fallback(self):
        with self.assertRaises(ValueError):
            validate_config({"model_type": "unknown"})

    def test_obj_and_coordinate_ids_are_part_of_contract(self):
        with self.assertRaises(ValueError):
            validate_tokenizer({"model": {"vocab": {}}, "added_tokens": []})


if __name__ == "__main__":
    unittest.main()
