import json
import unittest
from tools.converter.timestral.convert_timestral import CONFIG, tensor_contracts, validate_config


class ContractTest(unittest.TestCase):
    def test_exact_209m_contract(self):
        config = json.loads(CONFIG.read_text())
        shapes = tensor_contracts(config)
        self.assertEqual(sum(__import__('math').prod(s) for s in shapes.values()), 209312256)
        self.assertEqual(shapes['transformer.layers.3.attns.1.wQKV.weight'], (2304, 768))
        self.assertNotIn('transformer.layers.2.attns.1.wQKV.weight', shapes)
        self.assertEqual(shapes['head.projection.output_layer.weight'], (672, 3072))
        for key, value in [('d_model', 1024), ('rope_partial_factor', 1.), ('use_asinh', False),
                           ('split_context_horizon_attn', False), ('extra', True)]:
            with self.assertRaisesRegex(ValueError, 'configuration'):
                validate_config(dict(config, **{key: value}))


if __name__ == '__main__':
    unittest.main()
