"""Export the pinned OCRia-nano-1 ONNX graphs to one GGUF.

The release ships six ONNX graphs: `det.onnx` (line detector), `router.onnx`
(script router) and `rec_{latin,cjk,rtl,world}.onnx` (CTC recognizers). They all
go into a single GGUF, one tensor per Conv/Gemm weight and bias.

The `f32` export is lossless. The `q8_0` export stores the two-dimensional
weight matrices whose rows are a whole number of 32-value blocks -- pointwise
convolutions, the rec `mix` `fc` convolutions, the head convolutions and the
router's `Gemm` matrices -- in Q8_0 and keeps the depthwise kernels, the 9-tap
mix kernels and every bias in F32: OCRia computes in F32 whatever the file
stores.

Every Conv weight is stored as the 2-D `[out, fan_in]` matrix its kernel
flattens to (`[C, 1, 3, 3]` depthwise -> `[C, 9]`, `[out, in, 1, 1]` pointwise
-> `[out, in]`), so a kernel is a matrix row and Q8_0 can quantize it. Biases
stay one-dimensional. The loader in `src/models/ocria/mod.rs` checks the same
rule and reads the shapes from `src/models/ocria/contract.json`.

`rec_latin.onnx` is the one ONNX-Runtime-quantized graph: its weights are int8
`*_quantized` initializers that reach the Conv nodes through `DequantizeLinear`
nodes. They are dequantized here with the ONNX QDQ arithmetic,
`y = (x - zero_point) * scale` (per-axis where the graph says so), so the F32
arithmetic in Rust holds the values ONNX Runtime's dequantized weights hold.
The activation `QuantizeLinear`/`DequantizeLinear` pairs are *not* exported:
Rust computes in F32 throughout and the activation quantization is an accepted
precision loss (measured on `ids`/`conf` by `tools/oracle/ocria/trace.py`).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from tools.converter.utils.gguf import (
    GGML_F32,
    GGML_Q8_0,
    GgufWriter,
    gguf_dims,
    quantize_q8_0,
)

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = json.loads((ROOT / 'src/models/ocria/contract.json').read_text())
SOURCES = json.loads((ROOT / 'tools/oracle/ocria/sources.json').read_text())

#: The six graphs, in contract order.
GRAPHS = ('det', 'router', 'rec_latin', 'rec_cjk', 'rec_rtl', 'rec_world')
GROUPS = ('latin', 'cjk', 'rtl', 'world')
WEIGHT_OPS = ('Conv', 'Gemm', 'MatMul')
FORMATS = {'f32': 'ocria-nano-1-f32-v1', 'q8_0': 'ocria-nano-1-q8_0-v1'}
Q8_0_BLOCK = 32


def component_name(graph):
    """`det` -> `det`, `rec_cjk` -> `rec.cjk`: the component prefix of a graph."""
    return 'rec.' + graph[len('rec_'):] if graph.startswith('rec_') else graph


def tensor_stem(node_name):
    """The contract stem of a Conv/Gemm/MatMul node.

    Strip a leading `/`, split on `/`, drop the first segment when it is
    exactly `m`, drop the last segment when it is `Conv`, `Gemm` or `MatMul`,
    and join the rest with `.`. This is a contract with the Rust loader and
    with `contract.json`, so it is spelled out here and tested.

    `/m/stem/stem.0/Conv` -> `stem.stem.0`
    `/m/s4/s4.0/dw/Conv` -> `s4.s4.0.dw`
    `/body/body.0/pw/Conv` -> `body.body.0.pw`
    `/group/Gemm` -> `group`
    """
    segments = node_name[1:].split('/') if node_name.startswith('/') else node_name.split('/')
    if segments[:1] == ['m']:
        segments = segments[1:]
    if segments[-1:] and segments[-1] in WEIGHT_OPS:
        segments = segments[:-1]
    if not segments or not all(segments):
        raise ValueError(f'Unnamed OCRia weight node: {node_name}')
    return '.'.join(segments)


def quantizable(shape):
    """Whether a tensor may be stored in Q8_0.

    GGML quantizes a tensor row by row: the row is the fastest varying
    dimension and a Q8_0 block holds 32 values, so the row length has to be a
    whole number of blocks. Conv kernels are stored flattened to 2-D, so this
    keeps the pointwise convolutions, the mix `fc` convolutions, the head
    convolutions and the router's `Gemm` matrices, and leaves the depthwise
    kernels (row length 9), the 9-tap mix kernels and every bias in F32.

    The loader enforces the same rule in `load_tensor` in
    `src/models/ocria/mod.rs`; the two have to stay in step, because a file
    this function accepts but the loader rejects would fail to load.
    """
    return len(shape) == 2 and shape[-1] % Q8_0_BLOCK == 0


def flatten_weight(array, kernel_shape=None):
    """A conv/Gemm weight as the 2-D `[out, fan_in]` matrix the contract stores.

    The flatten is C-order over the trailing axes and nothing else: row `o` is
    the kernel of output channel `o`, and column `((c_in * Kh) + ky) * Kw + kx`
    holds `w[o, c_in, ky, kx]` (for the 1-D `mix` convolutions, column
    `c_in * K + k` holds `w[o, c_in, k]`). A `[C, 1, 3, 3]` depthwise kernel
    therefore becomes a `[C, 9]` row `c*9 + ky*3 + kx`, and a `[out, in, 1, 1]`
    pointwise kernel becomes `[out, in]`. Nothing is transposed: the resulting
    shape is checked against `contract.json` in `convert`, and
    `test_the_flatten_is_c_order_over_the_kernel_axes` pins the order (including
    the square case, where a transpose would keep the shape and only the values
    would move).
    """
    rows = int(array.shape[0])
    width = int(np.prod(array.shape[1:], dtype=np.int64))
    if kernel_shape is not None:
        # the trailing axes are the input channels followed by the kernel axes,
        # so a pointwise `[24,16,1,1]` kernel is `[1,1]` against `(1,1)` and a
        # depthwise `[32,1,3,3]` is `[3,3]` against `(3,3)`
        taps = int(np.prod(array.shape[2:], dtype=np.int64))
        if int(np.prod(kernel_shape, dtype=np.int64)) != taps:
            raise ValueError(
                f'OCRia weight {list(array.shape)} disagrees with kernel {list(kernel_shape)}')
    return np.ascontiguousarray(array.reshape(rows, width), dtype=np.float32)


def dequantize(values, scale, zero_point, axis=1):
    """ONNX `DequantizeLinear`: `y = (x - zero_point) * scale`, in F32.

    A one-dimensional `scale` (and `zero_point`) is per-axis: ONNX broadcasts
    it along `axis` of the quantized tensor, which is how per-channel weight
    quantization is expressed. The subtraction and the product are done in
    F32, the dtype the dequantized output has.
    """
    quantized = np.asarray(values)
    scale = np.asarray(scale, dtype=np.float32)
    zero = np.float32(0) if zero_point is None else np.asarray(zero_point, dtype=np.float32)
    if scale.ndim == 0:
        return (quantized.astype(np.float32) - zero) * scale
    if scale.ndim != 1:
        raise ValueError(f'Unsupported OCRia dequantize scale shape {list(scale.shape)}')
    if quantized.ndim == 0 or scale.size != quantized.shape[axis]:
        raise ValueError(f'Unsupported OCRia dequantize axis {axis} for {list(quantized.shape)}')
    shape = [1] * quantized.ndim
    shape[axis] = scale.size
    return (quantized.astype(np.float32) - zero.reshape(shape)) * scale.reshape(shape)


class Graph:
    """One ONNX graph with its initializers, `Constant` values and QDQ chains."""

    def __init__(self, path):
        self.path = path
        model = onnx.load(str(path), load_external_data=False)
        self.nodes = list(model.graph.node)
        self.arrays = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        self.producers = {}
        for node in self.nodes:
            for output in node.output:
                self.producers[output] = node

    def value(self, name):
        """A node input as a numpy array, dequantizing QDQ chains."""
        if name in self.arrays:
            return self.arrays[name]
        node = self.producers.get(name)
        if node is None:
            raise ValueError(f'{self.path.name}: cannot resolve {name}')
        if node.op_type == 'Constant':
            for attribute in node.attribute:
                if attribute.name == 'value':
                    return numpy_helper.to_array(attribute.t)
            raise ValueError(f'{self.path.name}: empty Constant {node.name}')
        if node.op_type == 'DequantizeLinear':
            axis = next((a.i for a in node.attribute if a.name == 'axis'), 1)
            zero = self.value(node.input[2]) if len(node.input) > 2 and node.input[2] else None
            return dequantize(self.value(node.input[0]), self.value(node.input[1]), zero, axis)
        raise ValueError(f'{self.path.name}: unsupported producer {node.op_type} for {name}')

    def weight(self, node):
        """A Conv/Gemm/MatMul weight as the 2-D `[out, fan_in]` matrix we store."""
        array = self.value(node.input[1])
        kernel = None
        if node.op_type == 'Conv':
            kernel = next((list(a.ints) for a in node.attribute
                           if a.name == 'kernel_shape'), None)
        elif node.op_type == 'Gemm':
            transposed = next((a.i for a in node.attribute if a.name == 'transB'), 0)
            if not transposed:
                array = array.T
        elif node.op_type == 'MatMul':
            array = array.T
        if array.ndim < 2:
            raise ValueError(f'{self.path.name}: {node.name} weight is {list(array.shape)}')
        return flatten_weight(array, kernel)

    def classes(self):
        """Output classes of a recognizer: the channels of the head Conv feeding ArgMax.

        The quantized graph puts a `DequantizeLinear` between the head Conv and
        `ArgMax`, so the chain is walked back to the Conv that made the logits.
        """
        for node in self.nodes:
            if node.op_type != 'ArgMax':
                continue
            source = node.input[0]
            while True:
                producer = self.producers.get(source)
                if producer is None:
                    raise ValueError(f'{self.path.name}: cannot resolve {source}')
                if producer.op_type in WEIGHT_OPS:
                    return int(self.value(producer.input[1]).shape[0])
                if producer.op_type in ('DequantizeLinear', 'QuantizeLinear', 'Relu'):
                    source = producer.input[0]
                    continue
                raise ValueError(
                    f'{self.path.name}: unexpected {producer.op_type} before ArgMax')
        raise ValueError(f'{self.path.name}: no ArgMax node')


def graph_tensors(graph, path):
    """The `{name: array}` weights and biases of one graph."""
    reader = Graph(path)
    component = component_name(graph)
    tensors = {}
    for node in reader.nodes:
        if node.op_type not in WEIGHT_OPS:
            continue
        stem = tensor_stem(node.name)
        weight = reader.weight(node)
        name = f'{component}.{stem}.weight'
        if name in tensors:
            raise ValueError(f'Duplicate OCRia tensor: {name}')
        tensors[name] = weight
        if len(node.input) > 2 and node.input[2]:
            bias = np.asarray(reader.value(node.input[2]), dtype=np.float32).reshape(-1)
            if bias.size != weight.shape[0]:
                raise ValueError(f'{name}: bias {bias.size} != rows {weight.shape[0]}')
            tensors[f'{component}.{stem}.bias'] = bias
    return tensors, reader


def tensor_map(directory):
    """Every tensor of the six graphs, and the class count of each recognizer."""
    tensors, classes = {}, {}
    for graph in GRAPHS:
        found, reader = graph_tensors(graph, directory / f'{graph}.onnx')
        overlap = tensors.keys() & found.keys()
        if overlap:
            raise ValueError(f'Duplicate OCRia tensors: {sorted(overlap)}')
        tensors.update(found)
        if graph.startswith('rec_'):
            classes[graph[len('rec_'):]] = reader.classes()
    return tensors, classes


def source_files():
    """The pinned files the converter reads: graphs, metadata, charsets, licence.

    `sources.json` also pins the layer-2 table model (`tablenet.onnx`,
    `tablenet.bin`); the OCR pipeline does not read those, so they are neither
    hashed here nor listed in the `ocria.source.files` metadata.
    """
    names = [f'{graph}.onnx' for graph in GRAPHS]
    names += ['det.meta.json', 'rec_rtl.meta.json', 'LICENSE']
    names += [f'rec_{group}.charset.json' for group in GROUPS]
    return names


def config_json(config):
    """The `ocria.config` string: the contract's `config` block, compactly."""
    return json.dumps(config, separators=(',', ':'))


def build_metadata(config, charsets, sources, quant, license_text):
    """The GGUF metadata pairs of a packed model, in file order."""
    rows = [
        ('general.architecture', 'ocria'),
        ('general.name', 'OCRia-nano-1'),
        ('general.license', sources['license']),
        ('general.license.link', f"https://modelscope.cn/models/{sources['repo']}"),
        ('ocria.format', FORMATS[quant]),
        ('ocria.config', config_json(config)),
    ]
    for group in GROUPS:
        rows.append((f'ocria.rec.{group}.charset', charsets[group]))
    rows += [
        ('ocria.source.repo', sources['repo']),
        ('ocria.source.revision', sources['revision']),
        ('ocria.source.commit', sources['commit']),
        ('ocria.source.files', json.dumps({name: pin['Sha256'] for name, pin in
                                           ((row['Path'], row) for row in sources['files'])
                                           if name in source_files()})),
        ('ocria.source.liteparse', sources['liteparse']['commit']),
        ('ocria.licenses', license_text),
    ]
    return rows


def read_charsets(directory, config):
    """The four charset JSON arrays as released, checked against the metadata files."""
    charsets = {}
    for group in GROUPS:
        text = (directory / f'rec_{group}.charset.json').read_text().strip()
        entries = json.loads(text)
        if not isinstance(entries, list) or not entries or \
                not all(isinstance(entry, str) and entry for entry in entries):
            raise ValueError(f'Invalid OCRia charset: rec_{group}.charset.json')
        charsets[group] = text
    # only the groups that ship a `rec_<group>.meta.json` opt into visual order
    visual_rtl = []
    for group in GROUPS:
        meta = directory / f'rec_{group}.meta.json'
        visual_rtl.append(bool(json.loads(meta.read_text())['visual_rtl'])
                          if meta.is_file() else False)
    if config['rec']['visual_rtl'] != visual_rtl:
        raise ValueError('OCRia visual_rtl disagrees with the rec_*.meta.json files')
    return charsets


def convert(directory, output, quant='f32'):
    if output.exists():
        raise FileExistsError(output)
    pins = {row['Path']: row['Sha256'] for row in SOURCES['files']}
    for name in source_files():
        if name not in pins:
            raise ValueError(f'OCRia source is not pinned: {name}')
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != pins[name]:
            raise ValueError(f'OCRia source hash mismatch: {name}')
    config = CONTRACT['config']
    thresh = json.loads((directory / 'det.meta.json').read_text())['thresh']
    if thresh != config['det']['thresh']:
        raise ValueError('OCRia detector threshold disagrees with det.meta.json')
    charsets = read_charsets(directory, config)
    tensors, classes = tensor_map(directory)
    for group, count in classes.items():
        if count != len(json.loads(charsets[group])) + 1:
            raise ValueError(f'OCRia {group} recognizer has {count} classes for its charset')
    if tensors.keys() != CONTRACT['tensors'].keys():
        raise ValueError('OCRia tensor inventory mismatch')
    for name, array in tensors.items():
        if list(array.shape) != CONTRACT['tensors'][name]:
            raise ValueError(f'Invalid OCRia tensor {name}: {list(array.shape)}')
        if not np.isfinite(array).all():
            raise ValueError(f'Non-finite OCRia tensor: {name}')
    license_text = (directory / 'LICENSE').read_text()
    writer = GgufWriter(output)
    for key, value in build_metadata(config, charsets, SOURCES, quant, license_text):
        writer.add_meta(key, value)
    quantized = 0
    payloads = []
    for name, array in tensors.items():
        if quant == 'q8_0' and quantizable(array.shape):
            payloads.append((name, GGML_Q8_0, gguf_dims(array.shape), quantize_q8_0(array)))
            quantized += 1
        else:
            payloads.append((name, GGML_F32, gguf_dims(array.shape),
                             array.astype('<f4').tobytes()))
    if quantized:
        writer.add_meta('ocria.quantized_tensors', quantized)
    for name, ggml_type, dims, raw in payloads:
        writer.add_tensor(name, ggml_type, dims, raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    writer.path = temporary
    try:
        writer.write()
        temporary.rename(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path, nargs='?')
    parser.add_argument('--quant', choices=sorted(FORMATS), default='f32',
                        help='weight storage: f32 (lossless) or q8_0 (smaller file)')
    parser.add_argument('--print-tensors', action='store_true',
                        help='print the contract `tensors` map of the graphs and exit')
    arguments = parser.parse_args()
    if arguments.print_tensors:
        tensors, _classes = tensor_map(arguments.directory)
        print(json.dumps({name: list(array.shape)
                          for name, array in sorted(tensors.items(), key=lambda row: row[0])},
                         indent=2))
    elif arguments.output is None:
        parser.error('an output path is required unless --print-tensors is given')
    else:
        convert(arguments.directory, arguments.output, arguments.quant)
