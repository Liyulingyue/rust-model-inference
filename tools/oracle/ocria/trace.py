"""Capture hash-pinned OCRia ONNX Runtime CPU inference as Rust parity fixtures.

One session per graph, one intra-op thread, ONNX Runtime's default optimisation
level, so the fixtures record what the runtime actually computes. Writes
`manifest.json` plus raw little-endian blobs into the directory given on the
command line; the Rust parity test in `src/models/ocria/tests.rs` reads exactly
that schema.

The inputs are the **network-boundary tensors**, not raw pixels: `det` gets the
normalised `(1,3,H,W)` float32 the detector session receives, `router` the
`(N,1,32,512)` and each recognizer the `(N,1,32,W)`. Preprocessing itself is
covered by the end-to-end page fixtures, so a failure here names the net.

Every input is a fixed deterministic pattern -- no randomness, no timestamps, no
seed -- and each case records the ONNX SHA256 it was produced from, so a fixture
cannot silently drift from `sources.json`.

`rec.latin` is the one ONNX-Runtime-quantized graph, and Rust deliberately does not
reproduce its activation quantization, so its cases carry both the pinned graph's
`ids`/`conf` and `ids_f32`/`conf_f32` from the equivalent F32 graph (`f32_graph`).
The latter is the target the Rust implementation must hit exactly; the manifest's
`note` says so, and `latin_qdq_effect.json` records the difference between them.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from tools.converter.ocria.convert_ocria import dequantize

HERE = Path(__file__).resolve().parent
SOURCES = json.loads((HERE / 'sources.json').read_text())

GRAPHS = ('det', 'router', 'rec_latin', 'rec_cjk', 'rec_rtl', 'rec_world')
GROUPS = ('latin', 'cjk', 'rtl', 'world')
#: Recognizer input height and the router's input width.
H = 32
ROUTER_W = 512
#: Recognizer widths the fixtures cover. 64 and 512 are bucket boundaries at both
#: ends; 256 and 384 sit on the timestep boundary of the two stride families
#: (cjk `W/8`, the other three `W/4`) and were added because a single square page
#: can hide a wrong stride pairing.
REC_WIDTHS = (64, 256, 384, 512)


def session(model):
    """One ONNX Runtime CPU session, single-threaded for reproducibility."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    source = model if isinstance(model, bytes) else str(model)
    return ort.InferenceSession(source, options, providers=['CPUExecutionProvider'])


def extended(model, extra):
    """The graph with `extra` tensors appended as outputs, as session bytes.

    ONNX Runtime only returns tensors that are graph outputs, so the detector
    activations `--with-intermediates` keeps are appended to an in-memory copy
    (typed by shape inference). The pinned file on disk is not touched, and `prob`
    is computed by both the plain and the extended session and must agree
    bit-for-bit -- that is what makes the extras trustworthy. `model` is a path or
    an already loaded `ModelProto`.
    """
    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(str(model))
    model = onnx.shape_inference.infer_shapes(model)
    known = {value.name: value
             for value in list(model.graph.value_info) + list(model.graph.output)}
    present = {output.name for output in model.graph.output}
    for name in extra:
        if name in present:
            continue
        if name not in known:
            raise ValueError(f'no value info for {name}')
        model.graph.output.append(known[name])
    return model.SerializeToString()


def blob(directory, name, array):
    """Write a little-endian f32/i64 blob and return its file name."""
    array = np.ascontiguousarray(array)
    if array.dtype == np.int64:
        suffix = '.i64'
        raw = array.astype('<i8').tobytes()
    else:
        suffix = '.f32'
        raw = array.astype('<f4').tobytes()
    assert name.endswith(suffix), (name, suffix)
    (directory / name).write_bytes(raw)
    return name


def normalize(grey):
    """uint8 grey -> the float32 the graphs are fed: `v/127.5 - 1`."""
    return grey.astype(np.float32) / np.float32(127.5) - np.float32(1.0)


def page(height, width):
    """A deterministic synthetic page: six text-like lines of strokes.

    Plain horizontal bars leave the detector's probability map under its
    threshold (it is trained on text lines), so each line here is a 32-px strip
    of the same strokes the recognizer fixtures use, at fixed vertical offsets
    with fixed gaps. With this page about 18 % of the `prob` map exceeds the
    detector's 0.55 threshold, which is what makes the fixture exercise the
    threshold and connected-component path as well as the arithmetic.
    """
    grey = np.full((height, width), 255, dtype=np.uint8)
    for index, top in enumerate((8, 48, 88, 128, 168, 208)):
        if top >= height:
            break
        strip = strokes(width - 20, index)
        rows = min(strip.shape[0], height - top)
        grey[top:top + rows, 10:10 + strip.shape[1]] = strip[:rows]
    return grey


def strokes(width, seed):
    """A deterministic 32-px-tall strip of dark strokes on white.

    Six hand-drawn cell shapes cycled by a fixed arithmetic rule, so a recognizer
    sees text-like strokes with real argmax margins instead of noise, and the
    same bytes on every run and every machine.
    """
    grey = np.full((H, width), 255, dtype=np.uint8)
    for index in range(width // 8):
        kind = (index * 5 + seed * 3) % 7
        left = index * 8 + 1
        if kind == 0:
            grey[6:26, left:left + 2] = 30
        elif kind == 1:
            grey[13:17, left:left + 5] = 30
        elif kind == 2:
            grey[8:24, left:left + 2] = 30
            grey[22:24, left:left + 6] = 30
        elif kind == 3:
            grey[8:24, left:left + 2] = 30
            grey[8:24, left + 4:left + 6] = 30
            grey[14:16, left:left + 6] = 30
        elif kind == 4:
            grey[7:25, left:left + 5] = 200
            grey[9:23, left + 1:left + 4] = 30
        elif kind == 5:
            grey[8:22, left:left + 4] = 30
            grey[20:22, left:left + 6] = 30
        # kind 6 stays blank: a word gap
    return grey


def f32_graph(path):
    """`rec_latin.onnx` as the F32 graph the Rust implementation must reproduce.

    Two rewrites, in memory, on the pinned file:

    * each weight `DequantizeLinear` becomes a `Constant` holding the values the
      converter writes into the GGUF (the same `(q - zp) * scale`, imported from
      `tools/converter/ocria/convert_ocria.py` so the two cannot drift);
    * every activation `QuantizeLinear`/`DequantizeLinear` pair is rewired out, so
      consumers read the unquantized activation. Where the quantizer's zero point
      is 0 it was clamping negatives, i.e. it *was* the Relu the quantizer deleted
      (see `graph_notes.md` section 8.1), so a `Relu` is put back in its place.
    """
    model = onnx.load(str(path))
    graph = model.graph
    initialisers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    producers = {output: node for node in graph.node for output in node.output}
    kept, rename, dropped = [], {}, set()
    for node in graph.node:
        if node.op_type == 'DequantizeLinear' and node.input[0] in initialisers:
            # a weight or a bias: freeze the dequantized values into a constant
            axis = next((a.i for a in node.attribute if a.name == 'axis'), 1)
            zero = initialisers[node.input[2]] if len(node.input) > 2 and node.input[2] else None
            values = dequantize(initialisers[node.input[0]], initialisers[node.input[1]],
                                zero, axis)
            kept.append(onnx.helper.make_node(
                'Constant', [], [node.output[0]], name=node.name,
                value=numpy_helper.from_array(values, name=node.output[0] + '_value')))
        elif node.op_type == 'DequantizeLinear':
            producer = producers[node.input[0]]
            if producer.op_type != 'QuantizeLinear':
                raise ValueError(f'Unexpected {producer.op_type} before {node.name}')
            source = producer.input[0]
            while source in rename:
                source = rename[source]
            zero = initialisers.get(producer.input[2]) if len(producer.input) > 2 else None
            if zero is not None and not zero.any():
                clamped = node.output[0] + '_relu'
                kept.append(onnx.helper.make_node('Relu', [source], [clamped],
                                                  name=node.name + '_relu'))
                source = clamped
            rename[node.output[0]] = source
            dropped.update((node.name, producer.name))
        elif node.op_type == 'QuantizeLinear':
            # every quantizer feeds exactly one dequantizer, handled above
            continue
        else:
            kept.append(node)
    for node in kept:
        for index, name in enumerate(node.input):
            if name in rename:
                node.input[index] = rename[name]
    del graph.node[:]
    graph.node.extend(kept)
    onnx.checker.check_model(model)
    return model


def capture_det(model_dir, out, intermediates):
    """Two detector cases: a 256x256 page and a 96x160 non-square one.

    The non-square case exists because the FPN pairs a 2x nearest upsample with a
    `ceil(n/2)` stride, and a single square page can hide a wrong pairing (96 and
    160 are odd multiples of 32, the harshest case). `--with-intermediates` also
    writes the stem Relu and the deepest FPN sum, which do not fit the committed
    fixture budget (1.5 MB at 256x256).
    """
    cases = [('lines', 256, 256), ('narrow', 96, 160)]
    entries = []
    for name, height, width in cases:
        grey = page(height, width)
        x = np.stack([normalize(grey)] * 3)[None]
        prob = session(model_dir / 'det.onnx').run(['prob'], {'x': x})[0]
        assert prob.shape == (1, 1, height // 2, width // 2), prob.shape
        entry = dict(name=name, height=height, width=width,
                     input=blob(out, f'det_{name}_input.f32', x),
                     prob=blob(out, f'det_{name}_prob.f32', prob))
        if intermediates:
            names = ['/m/stem/stem.2/Relu_output_0', '/m/Add_2_output_0']
            prob_plain, stem, fpn = session(extended(model_dir / 'det.onnx', names)).run(
                ['prob'] + names, {'x': x})
            if not np.array_equal(prob.view(np.uint32), prob_plain.view(np.uint32)):
                raise ValueError(
                    'the extended detector graph disagrees with the pinned one on prob')
            entry.update(stem=blob(out, f'det_{name}_stem.f32', stem),
                         fpn=blob(out, f'det_{name}_fpn.f32', fpn),
                         intermediate_shapes=dict(stem=list(stem.shape),
                                                  fpn=list(fpn.shape)))
        entries.append(entry)
    return entries


def capture_router(model_dir, out):
    """Two router cases, two crops each: the raw `group`/`flip` logits, no softmax."""
    entries = []
    reader = session(model_dir / 'router.onnx')
    for seed, name in enumerate(('latin', 'cjk')):
        count = 2
        lines = [normalize(strokes(ROUTER_W, seed * 2 + index)) for index in range(count)]
        x = np.stack(lines)[:, None]
        group, flip = reader.run(['group', 'flip'], {'x': x})
        assert group.shape == (count, 4) and flip.shape == (count, 2)
        entries.append(dict(
            name=name, count=count,
            input=blob(out, f'router_{name}_input.f32', x),
            group_logits=blob(out, f'router_{name}_group.f32', group),
            flip_logits=blob(out, f'router_{name}_flip.f32', flip)))
    return entries


def capture_rec(model_dir, out):
    """Every recognizer at four widths: `ids`/`conf` as the graph emits them.

    `rec.latin` is ONNX-Runtime-quantized, and Rust deliberately does not
    reproduce activation quantization, so its case also carries `ids_f32` /
    `conf_f32` from the faithful F32 graph (`f32_graph`) -- that is the target the
    Rust code must hit exactly. The QDQ `ids`/`conf` stay for the record, so the
    size of the precision loss remains measurable (see `latin_qdq_effect.json`).
    """
    cases = {}
    for index, group in enumerate(GROUPS):
        reader = session(model_dir / f'rec_{group}.onnx')
        plain = (session(f32_graph(model_dir / f'rec_{group}.onnx').SerializeToString())
                 if group == 'latin' else None)
        entries = []
        for step, width in enumerate(REC_WIDTHS):
            x = normalize(strokes(width, index * 4 + step))[None, None]
            ids, conf = reader.run(['ids', 'conf'], {'x': x})
            assert ids.dtype == np.int64 and conf.dtype == np.float32
            assert ids.shape == conf.shape == (1, width // (8 if group == 'cjk' else 4),)
            name = f'w{width}'
            entry = dict(name=name, count=1, width=width,
                         input=blob(out, f'rec_{group}_{name}_input.f32', x),
                         ids=blob(out, f'rec_{group}_{name}_ids.i64', ids),
                         conf=blob(out, f'rec_{group}_{name}_conf.f32', conf))
            if plain is not None:
                f32_ids, f32_conf = plain.run(['ids', 'conf'], {'x': x})
                assert f32_ids.shape == ids.shape
                entry.update(ids_f32=blob(out, f'rec_{group}_{name}_ids_f32.i64', f32_ids),
                             conf_f32=blob(out, f'rec_{group}_{name}_conf_f32.f32', f32_conf))
            entries.append(entry)
        cases[group] = entries
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path,
                        help='directory holding the six pinned ONNX graphs')
    parser.add_argument('output', type=Path, help='fixture directory to write')
    parser.add_argument('--with-intermediates', action='store_true',
                        help='also write the 1.5 MB detector stem/FPN activations')
    args = parser.parse_args()
    pins = {row['Path']: row['Sha256'] for row in SOURCES['files']}
    hashes = {}
    for graph in GRAPHS:
        path = args.model_dir / f'{graph}.onnx'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != pins[f'{graph}.onnx']:
            raise ValueError(f'Oracle source mismatch: {path}')
        hashes[f'{graph}.onnx'] = digest
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'manifest.json'
    # keep any key this script does not own (the page fixtures live in `pages/`
    # and may add their own), so regenerating the nets fixtures never drops them
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except ValueError:
            manifest = {}
    manifest.update({
        'note': 'Inputs are the post-preprocessing network tensors, not raw pixels. '
                '`ids`/`conf` (and `group`/`flip` logits) are ONNX Runtime\'s output of '
                'the pinned graph. `rec.latin` is the one ONNX-Runtime-quantized graph: '
                'its `ids_f32`/`conf_f32` come from the equivalent F32 graph the Rust '
                'implementation must reproduce exactly, while `ids`/`conf` are the '
                'official runtime\'s answer; `latin_qdq_effect.json` records the '
                'difference.',
        'onnx_sha256': hashes,
        'det': capture_det(args.model_dir, args.output, args.with_intermediates),
        'router': capture_router(args.model_dir, args.output),
        'rec': capture_rec(args.model_dir, args.output),
        'generated_by': {
            'onnxruntime': ort.__version__,
            'numpy': np.__version__,
            'intra_op_num_threads': 1,
            'inter_op_num_threads': 1,
            'optimization': 'onnxruntime default',
            'providers': ['CPUExecutionProvider'],
        },
    })
    # canonical key order, so regenerating the file is byte-stable whatever the
    # previous manifest looked like; keys this script does not own follow, sorted
    order = ['note', 'onnx_sha256', 'det', 'router', 'rec', 'generated_by']
    ordered = {key: manifest[key] for key in order if key in manifest}
    ordered.update({key: manifest[key] for key in sorted(manifest) if key not in ordered})
    manifest_path.write_text(json.dumps(ordered, indent=2) + '\n')
    size = sum(path.stat().st_size for path in args.output.iterdir() if path.is_file())
    print(f'{args.output}: {len(manifest["det"])} det, {len(manifest["router"])} router, '
          f'{sum(len(rows) for rows in manifest["rec"].values())} rec cases, {size} bytes')


if __name__ == '__main__':
    main()
