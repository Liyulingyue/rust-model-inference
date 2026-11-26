"""Print the op sequence of each pinned OCRia graph, with the attributes that matter.

One text file per graph in `tools/oracle/ocria/dumps/`: graph inputs and outputs
with dtype and shape, then every node except `Constant` / `QuantizeLinear` /
`DequantizeLinear`. `Conv` and `Gemm`/`MatMul` rows carry the weight and bias
shapes (dequantized, for the QDQ graph) plus kernel, strides, pads and group;
every other row carries its attributes.

This is a reading aid for review and for `graph_notes.md`; the authoritative
tensor list is `src/models/ocria/contract.json`. Weight values are resolved
through the converter's own `Graph`, so the shapes printed here are exactly the
shapes the converter stores.

    python -m tools.oracle.ocria.dump_graphs /path/to/OCRia-nano-1 tools/oracle/ocria/dumps
"""
import argparse
from pathlib import Path

import onnx
from onnx import numpy_helper

from tools.converter.ocria.convert_ocria import GRAPHS, Graph


def attributes(node):
    out = {}
    for attribute in node.attribute:
        if attribute.type == onnx.AttributeProto.STRING:
            out[attribute.name] = attribute.s.decode()
        elif attribute.type == onnx.AttributeProto.INTS:
            out[attribute.name] = list(attribute.ints)
        elif attribute.type == onnx.AttributeProto.INT:
            out[attribute.name] = attribute.i
        elif attribute.type == onnx.AttributeProto.FLOAT:
            out[attribute.name] = attribute.f
        elif attribute.type == onnx.AttributeProto.TENSOR:
            value = numpy_helper.to_array(attribute.t)
            out[attribute.name] = list(value.shape) + value.flatten()[:6].tolist()
    return out


def describe(value_info):
    kind = value_info.type.tensor_type
    dims = [dim.dim_value if dim.HasField('dim_value') else (dim.dim_param or '?')
            for dim in kind.shape.dim]
    return f'{value_info.name} {onnx.TensorProto.DataType.Name(kind.elem_type)} {dims}'


def dump(graph, path):
    reader = Graph(path)
    model = onnx.load(str(path))
    lines = ['=' * 100, f'FILE {path.name}',
             f"producer: {model.producer_name} {model.producer_version} ir {model.ir_version} "
             f"opset {[(o.domain or 'ai.onnx', o.version) for o in model.opset_import]} "
             f"metadata {({p.key: p.value for p in model.metadata_props})}"]
    lines += [f'INPUT  {describe(value)}' for value in model.graph.input]
    lines += [f'OUTPUT {describe(value)}' for value in model.graph.output]
    for node in reader.nodes:
        if node.op_type in ('Constant', 'QuantizeLinear', 'DequantizeLinear'):
            continue
        attrs = attributes(node)
        if node.op_type in ('Conv', 'Gemm', 'MatMul'):
            weight = reader.value(node.input[1])
            bias = reader.value(node.input[2]) if len(node.input) > 2 and node.input[2] else None
            lines.append(
                '  %-6s %-46s w=%-18s bias=%-10s k=%s s=%s p=%s g=%s' % (
                    node.op_type, node.name, list(weight.shape),
                    None if bias is None else list(bias.shape),
                    attrs.get('kernel_shape'), attrs.get('strides'), attrs.get('pads'),
                    attrs.get('group', 1)))
        else:
            lines.append('  %-10s %-40s in=%s out=%s %s' % (
                node.op_type, node.name, list(node.input), list(node.output), attrs))
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for graph in GRAPHS:
        text = dump(graph, args.model_dir / f'{graph}.onnx')
        (args.output / f'{graph}.txt').write_text(text)
        print(f'{graph}.txt: {len(text.splitlines())} lines')


if __name__ == '__main__':
    main()
