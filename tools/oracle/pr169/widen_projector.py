"""Widen F16/BF16 weights exactly so Oracle F32 matmul does not round activations."""
import argparse
from pathlib import Path

import gguf
import numpy as np


def widen(source, destination):
    if Path(destination).exists():
        raise FileExistsError(destination)
    reader = gguf.GGUFReader(source)
    writer = gguf.GGUFWriter(destination, 'clip')
    for key, field in reader.fields.items():
        if not key.startswith('GGUF.'):
            writer.add_key_value(key, field.contents(), field.types[0],
                                 field.types[1] if len(field.types) > 1 else None)
    for tensor in reader.tensors:
        data = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
        assert np.isfinite(data).all(), tensor.name
        writer.add_tensor(tensor.name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    widened = gguf.GGUFReader(destination)
    assert len(widened.tensors) == len(reader.tensors)
    for before, after in zip(reader.tensors, widened.tensors):
        expected = gguf.quants.dequantize(before.data, before.tensor_type)
        assert before.name == after.name and np.array_equal(expected, after.data), before.name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()
    widen(args.source, args.destination)
