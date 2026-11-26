"""Export the pinned LayoutNet graph (`tablenet.onnx`) to one GGUF.

LayoutNet is layer 2 of the OCRia-nano-1 adaptation: a 4.27 M-parameter model
that reads a page's text units (68 features each), their pairwise ruling-line
features and the page raster, and emits `node [1,N,2]`, `pair [1,5,N,N]` and
`cls [1,N,13]` logits. It ships as its own GGUF (`general.architecture =
ocria-layout`) so the table model is a separate file from the OCR stack.

The source of truth is `tablenet.onnx`, not `tablenet.bin`: the ONNX export has
the image branch's BatchNorm already folded into the conv kernels and biases,
so the Rust side needs no fold (`net.rs:181-206` is dead code for us). The
`.bin` is only the naming and orientation oracle, and `check_ocria_layout.py`
re-runs the whole bijection against it.

## Names

The export names initializers inconsistently: biases and norms keep their
PyTorch parameter name (`m.enc.layers.0.self_attn.in_proj_bias`), while every
weight matrix and conv kernel is anonymous (`onnx::MatMul_1480`,
`onnx::Conv_1427`). Its *node* names are lossy in the other direction -- the
exporter flattened the `m.`, `enc.` and `img.` scopes, so layer 0 is
`/layers.0/MatMul` and the first conv is `/cnn/cnn.0/Conv`.

`tablenet.bin` names all 126 of its tensors semantically, and those names are
what the reference runtime (`net.rs`) reads, so the GGUF uses **the `.bin` names
verbatim**:

* 67 tensors keep their own name, minus the leading `m.`;
* 43 are anonymous and are named from the node that consumes them --
  `<module path>.<weight|bias>` with `/` -> `.` and the trailing op segment
  dropped, exactly the layer-1 rule, plus the branch prefix (`enc.` / `img.`)
  that the export dropped and the sibling tensors still carry. The 6
  `img.attn.*.in_proj_*` tensors are sliced into q/k/v before any `MatMul`, so
  they have no single consuming node and keep their own names;
* the four conv **biases** are a name this converter mints: the ONNX `Conv` has
  the BatchNorm-folded bias as its third input and the `.bin` carries the raw
  BN arrays instead.

`contract.json` is the authority for the resulting 110 names and shapes and is
checked against the graph here, so a drift between the two fails at conversion.

## Orientation

Every tensor is stored in torch orientation -- `[out, in]` for matrices,
`[cout, cin, kh, kw]` for convs, 1-D for biases and norms -- which is the `.bin`'s
shape, what `contract.json` lists, and what the Rust loader and
`ops::nn::conv2d` take. The ONNX file stores every `MatMul` weight as
`[in, out]`, so those are transposed; **the one exception** is
`img.attn.{0,1}.in_proj_weight`, which the file already stores as `[768, 256]`
(`q|k|v` rows) because the graph slices it before transposing each slice.

## Quantisation

The `f32` export is lossless. The `q8_0` export follows the layer-1 rule: a 2-D
tensor whose row length (the torch shape's last dim) is a whole number of
32-value blocks goes to Q8_0, everything else -- every bias, every norm, and the
four 4-D conv kernels, which `ops::nn::conv2d` wants as `[cout,cin,kh,kw]` --
stays F32. The loader in `src/models/ocria/layout/mod.rs` enforces the same rule
and reads the shapes from `src/models/ocria/layout/contract.json`.
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
CONTRACT = json.loads((ROOT / 'src/models/ocria/layout/contract.json').read_text())
SOURCES = json.loads((ROOT / 'tools/oracle/ocria/sources.json').read_text())

#: The `general.architecture` value; `src/models/ocria/layout/mod.rs` checks it.
ARCHITECTURE = 'ocria-layout'
FORMATS = {'f32': 'ocria-layout-f32-v1', 'q8_0': 'ocria-layout-q8_0-v1'}
Q8_0_BLOCK = 32
#: The pinned files the converter reads: the graph, its weight oracle, the licence.
SOURCE_FILES = ('tablenet.onnx', 'tablenet.bin', 'LICENSE')


def module_path(node_name):
    """The `.bin` module path of a `MatMul`/`Conv`/`Add` node.

    `/layers.0/MatMul`                 -> `enc.layers.0.self_attn`
    `/layers.0/out_proj/MatMul`        -> `enc.layers.0.self_attn.out_proj`
    `/layers.0/linear1/MatMul`         -> `enc.layers.0.linear1`
    `/layers.0/norm1/LayerNormalization` -> `enc.layers.0.norm1`
    `/inp/inp.0/MatMul`                -> `inp.0`
    `/cnn/cnn.0/Conv`                  -> `img.cnn.0`
    `/cnn/cnn.3/cnn.3.0/Conv`          -> `img.cnn.3.0`
    `/unit/unit.0/MatMul`              -> `img.unit.0`
    `/tok/MatMul`                      -> `img.tok`
    `/norm.0/LayerNormalization`       -> `img.norm.0`
    `/out_proj/MatMul`                 -> `img.attn.0.out_proj`
    `/out_proj_1/MatMul`               -> `img.attn.1.out_proj`
    `/node/MatMul`                     -> `node`

    The `enc.` and `img.` prefixes are the scopes the exporter dropped: they are
    recovered from the sibling tensors that still carry them (`m.enc.layers.0.
    norm1.weight`, `m.img.norm.0.weight`), which is why this table and
    `tablenet.bin`'s tensor table agree name for name.
    """
    segments = node_name[1:].split('/') if node_name.startswith('/') else node_name.split('/')
    segments = segments[:-1]  # drop the op segment (`MatMul`, `Conv`, `Add`, ...)
    if not segments or not all(segments):
        raise ValueError(f'Unnamed LayoutNet node: {node_name}')
    branch, _, index = segments[0].partition('.')
    if branch == 'layers':
        if len(segments) == 1:
            return f'enc.{segments[0]}.self_attn'
        leaf = segments[-1]
        if leaf == 'out_proj':
            return f'enc.{segments[0]}.self_attn.out_proj'
        if leaf in ('linear1', 'linear2', 'norm1', 'norm2'):
            return f'enc.{segments[0]}.{leaf}'
        raise ValueError(f'Unmapped LayoutNet layer node: {node_name}')
    if branch == 'inp':
        # `/inp/inp.0/MatMul`: the module name repeats the branch, `inp.0`
        return f"inp.{segments[-1].partition('.')[2]}"
    if branch == 'cnn':
        # `/cnn/cnn.3/cnn.3.0/Conv` repeats the module in the scope; the `.bin`
        # calls it `img.cnn.3.0`, so the last segment is the one to keep.
        return f'img.{segments[-1]}'
    if branch == 'unit':
        # `/unit/unit.0/MatMul` -> `img.unit.0`
        return f'img.{segments[-1]}'
    if branch == 'tok':
        return 'img.tok'
    if branch == 'norm':
        return f'img.norm.{index}'
    if branch == 'out_proj':
        return 'img.attn.0.out_proj'
    if branch == 'out_proj_1':
        return 'img.attn.1.out_proj'
    if branch in ('node', 'qk', 'pair_geo', 'cls'):
        return branch
    raise ValueError(f'Unmapped LayoutNet node: {node_name}')


def tensor_name(node, kind):
    """The `.bin` name of a weight or bias consumed by `node`."""
    path = module_path(node.name)
    if path.endswith('.self_attn'):
        # the packed q|k|v projection: `nn.MultiheadAttention`'s parameter names
        return f'{path}.in_proj_{kind}'
    return f'{path}.{kind}'


def quantizable(shape):
    """Whether a tensor may be stored in Q8_0 (the layer-1 rule).

    GGML quantises a tensor row by row and a Q8_0 block holds 32 values, so the
    row length -- the torch shape's last dim -- has to be a whole number of
    blocks. The four conv kernels are 4-D and stay F32, which is also what
    `load_tensor` in `src/models/ocria/layout/mod.rs` accepts.
    """
    return len(shape) == 2 and shape[-1] % Q8_0_BLOCK == 0


class Graph:
    """`tablenet.onnx`: its initializers and who consumes them."""

    def __init__(self, path):
        self.path = path
        model = onnx.load(str(path), load_external_data=False)
        self.initializers = {init.name: numpy_helper.to_array(init)
                             for init in model.graph.initializer}
        self.consumers = {}
        for node in model.graph.node:
            for slot, name in enumerate(node.input):
                if name in self.initializers:
                    self.consumers.setdefault(name, []).append((node, slot))
        self.metadata = {prop.key: prop.value for prop in model.metadata_props}

    def consumer(self, name):
        """The `(kind, node)` a weight or bias belongs to."""
        found = []
        for node, slot in self.consumers.get(name, []):
            if node.op_type in ('Conv', 'MatMul') and slot == 1:
                found.append(('weight', node))
            elif node.op_type == 'Conv' and slot == 2:
                found.append(('bias', node))
            elif node.op_type == 'Add' and slot == 0:
                found.append(('bias', node))
            else:
                raise ValueError(
                    f'{self.path.name}: unexpected {node.op_type}[{slot}] consumer of {name}')
        if len(found) != 1:
            raise ValueError(f'{self.path.name}: {name} has {len(found)} weight/bias consumers')
        return found[0]


def tensor_map(directory):
    """Every contract tensor as a torch-orientation F32 array."""
    graph = Graph(directory / 'tablenet.onnx')
    tensors = {}
    for name, array in graph.initializers.items():
        array = np.asarray(array, dtype=np.float32)
        if name.startswith('m.'):
            out = name[len('m.'):]
        else:
            kind, node = graph.consumer(name)
            out = tensor_name(node, kind)
            if node.op_type == 'MatMul':
                array = array.T  # the file stores [in, out]; the contract wants [out, in]
        if out in tensors:
            raise ValueError(f'Duplicate LayoutNet tensor: {out}')
        tensors[out] = np.ascontiguousarray(array, dtype=np.float32)
    return tensors, graph


def config_json(config):
    """The `ocria-layout.config` string: the contract's `config` block, compactly.

    The loader compares this against `serde_json::to_string` of the same value
    read from `contract.json`, so both sides have to agree on the key order:
    `serde_json` here is built with `preserve_order` (Cargo.toml), which keeps
    the file's order, and `json.dumps` on a `json.load`s dict keeps it too.
    """
    return json.dumps(config, separators=(',', ':'))


def source_pins():
    """The SHA256 pins of the files this converter reads, from `sources.json`."""
    pins = {row['Path']: row['Sha256'] for row in SOURCES['files']}
    for name in SOURCE_FILES:
        if name not in pins:
            raise ValueError(f'LayoutNet source is not pinned: {name}')
    return {name: pins[name] for name in SOURCE_FILES}


def build_metadata(config_text, quant, pins, license_text):
    """The GGUF metadata pairs of a packed model, in file order.

    `config_text` is the graph's own `liteparse.tablenet` string, written verbatim:
    the loader compares it to `contract.json`'s `config` by value, so this keeps the
    file independent of how the contract happens to be formatted.
    """
    rows = [
        ('general.architecture', ARCHITECTURE),
        ('general.name', 'OCRia-nano-1 LayoutNet'),
        ('general.license', SOURCES['license']),
        ('general.license.link', f"https://modelscope.cn/models/{SOURCES['repo']}"),
        ('ocria-layout.format', FORMATS[quant]),
        ('ocria-layout.config', config_text),
        ('ocria-layout.source.repo', SOURCES['repo']),
        ('ocria-layout.source.revision', SOURCES['revision']),
        ('ocria-layout.source.commit', SOURCES['commit']),
        ('ocria-layout.source.files', json.dumps(pins, separators=(',', ':'))),
        ('ocria-layout.source.liteparse', SOURCES['liteparse']['commit']),
        ('ocria-layout.licenses', license_text),
    ]
    return rows


def convert(directory, output, quant='f32'):
    if output.exists():
        raise FileExistsError(output)
    pins = source_pins()
    for name, sha in pins.items():
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError(f'LayoutNet source hash mismatch: {name}')
    tensors, graph = tensor_map(directory)
    # The graph's own metadata is the config the contract must carry. The loader
    # (`LayoutNet::from_source`) parses both sides into a `serde_json::Value` and
    # compares by value, so key order and whitespace are free; the GGUF still writes
    # the graph's string verbatim and `check_ocria_layout.py` pins that.
    config_text = graph.metadata['liteparse.tablenet']
    if json.loads(config_text) != CONTRACT['config']:
        raise ValueError('LayoutNet config disagrees with the graph metadata')
    if tensors.keys() != CONTRACT['tensors'].keys():
        raise ValueError('LayoutNet tensor inventory mismatch')
    for name, array in tensors.items():
        if list(array.shape) != CONTRACT['tensors'][name]:
            raise ValueError(f'Invalid LayoutNet tensor {name}: {list(array.shape)}')
        if not np.isfinite(array).all():
            raise ValueError(f'Non-finite LayoutNet tensor: {name}')
    license_text = (directory / 'LICENSE').read_text()
    writer = GgufWriter(output)
    for key, value in build_metadata(config_text, quant, pins, license_text):
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
        writer.add_meta('ocria-layout.quantized_tensors', quantized)
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
                        help='print the contract `tensors` map of the graph and exit')
    arguments = parser.parse_args()
    if arguments.print_tensors:
        tensors, _graph = tensor_map(arguments.directory)
        print(json.dumps({name: list(array.shape)
                          for name, array in sorted(tensors.items(), key=lambda row: row[0])},
                         indent=2))
    elif arguments.output is None:
        parser.error('an output path is required unless --print-tensors is given')
    else:
        convert(arguments.directory, arguments.output, arguments.quant)
