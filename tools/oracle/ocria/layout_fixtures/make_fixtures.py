"""Record ONNX Runtime fixtures for LayoutNet (`tablenet.onnx`).

The Rust port in `src/models/ocria/layout/` is written against the ONNX graph, so
the regression set has to come from ONNX Runtime on the same graph, not from the
reference's own kernels. Each case records the six graph inputs, the three
outputs and a handful of intermediates, so a mismatch can be localised instead of
just "the logits are off".

The page is synthetic and fully deterministic (integer coordinates, no RNG, no
fonts): a 4-row x 2-column ruled table, N = 8 units, drawn on the production
1024x1024 canvas. The features are the reference's own `unit_features` and
`pair_rule_features` with `proj = None` -- the configuration this repository can
reach (`layout-plan.md` §2.2) -- so features 53..60 are zero and the two id
columns are -1.

Three cases share the page and the unit list (`CASES` below):

* `page` -- `valid = (1, 1)`, every box inside the canvas. This is the ordinary
  full-page case; the token mask is all false.
* `clip` -- `valid = (0.78, 1)`, the right column's boxes reach x = 0.928, i.e.
  past the page's right edge. Tokens right of 0.78 are masked, which pins the
  graph's `-1e9` additive mask against the Rust port's "drop the invalid tokens"
  path (`net.rs:765-790`).
* `blank` -- the same inputs as `page` with an all-white raster, so the image
  branch runs on biases alone (`unit_mlp` is not zero). It separates "the raster
  decode or `GridSample` is wrong" from "the conv stack or the unit MLP is
  wrong", and gives the encoder a second, independent `unit_mlp` sample. Its
  raster is not stored: the manifest entry carries `fill: 0` instead of a file.

Intermediates are captured by appending the tensor names to `graph.output` on an
in-memory copy of the model; the pinned file is never modified.

Usage:

    python -m tools.oracle.ocria.layout_fixtures.make_fixtures /workspace/models/OCRia-nano-1 \\
        tools/oracle/ocria/layout_fixtures
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

IMG_SIDE = 1024
FEAT_DIM = 68
N_FOURIER = 8
#: Every tensor that may be recorded next to the outputs: name -> (graph tensor, why).
INTERMEDIATES = {
    'grid': ('/Reshape_2_output_0', 'the 2x6-per-box GridSample grid, (unit, gh) flattened'),
    'unit_samples': ('/Reshape_4_output_0', 'the [N,768] per-unit GridSample vector'),
    'unit_mlp': ('/unit/unit.2/Add_output_0', 'the image vector added to the encoder input'),
    'embed': ('/Add_2_output_0', 'the encoder input (input MLP + image vector)'),
    'pos2d_units': ('/Concat_11_output_0', 'pos2d of the unit box centres'),
    'pos2d_freq': ('/Constant_76_output_0', 'the token pos2d frequency table, as baked in the file'),
    'pos2d_freq_runtime': ('/Mul_13_output_0', 'the unit pos2d frequency table, computed at run time'),
    'token_mask': ('/Cast_9_output_0', '1.0 = masked (a token past the page extent)'),
    'enc_layer0': ('/layers.0/Add_3_output_0', 'the output of encoder layer 0'),
    'enc_layer5': ('/layers.5/Add_3_output_0', 'the output of encoder layer 5'),
    'hidden_final': ('/Add_19_output_0', 'after both cross-attentions: what the heads read'),
    # the three too big to commit (1 MB, 256 KB, 1 MB): `--with-intermediates` adds them
    'cnn_map': ('/cnn/cnn.5/cnn.5.2/Relu_output_0', 'the 64x64x64 stride-16 map'),
    'avgpool': ('/AveragePool_output_0', 'the 2x2/2 pool the tokens come from'),
    'tokens': ('/Add_5_output_0', 'img.tok + pos2d of the token centres'),
}
HEAVY_INTERMEDIATES = ('cnn_map', 'avgpool', 'tokens')
LIGHT_INTERMEDIATES = tuple(name for name in INTERMEDIATES if name not in HEAVY_INTERMEDIATES)

#: The cases, in manifest order. `valid_w` is the page's fraction of the canvas.
#: `blank` replaces the raster with an all-white canvas and keeps every other input,
#: so the image branch still runs (its biases alone give a non-zero `unit_mlp`) and the
#: encoder sees a second, independent `unit_mlp` sample. `intermediates` is the subset
#: of `INTERMEDIATES` worth recording for that case.
CASES = (
    {'name': 'page', 'valid_w': 1.0, 'blank': False, 'intermediates': LIGHT_INTERMEDIATES},
    {'name': 'clip', 'valid_w': 0.78, 'blank': False, 'intermediates': LIGHT_INTERMEDIATES},
    # `unit_samples` is the blob that separates "the raster decode / GridSample is
    # wrong" from "the conv stack or the unit MLP is wrong", so the blank case keeps
    # it. `grid`, `pos2d_*` and `token_mask` are functions of `boxes`/`valid` alone and
    # are byte-identical to the `page` case's, so they are not duplicated here.
    {'name': 'blank', 'valid_w': 1.0, 'blank': True,
     'intermediates': ('unit_samples', 'unit_mlp', 'embed', 'enc_layer0', 'enc_layer5',
                       'hidden_final')},
)
#: The rule set the page is drawn with: (y, x0, x1) horizontals, (x, y0, y1) verticals.
ROWS = ((100, 126), (160, 186), (220, 246), (280, 306))
INK_TEXT = 191  # 191/255 = 0.74902, a value the f32 canvas holds exactly
INK_RULE = 255
TEXTS = (('Item', 'Price'), ('Coffee', '12.50'), ('Bread', '4.20'), ('Olive oil', '8.90'))
NUMERIC = re.compile(r'^[\s(\[\u20ac\u00a3\u00a5+\-\u2013\u2212\u00b1<>\u2264\u2265~]*'
                     r'[\d.,]+[%)\]*\u2020\u2021\u00a7a-z]*$')


def canvas_boxes(valid_w):
    """The 8 unit boxes in canvas pixels, row-major, and the rules drawn under them."""
    left = (60, 400)
    right = (470, 920 if valid_w >= 1.0 else 950)
    boxes = []
    for y0, y1 in ROWS:
        boxes.append([left[0], y0, left[1], y1])
        boxes.append([right[0], y0, right[1], y1])
    right_edge = right[1]
    rules_h = [(145, left[0], right_edge), (325, left[0], right_edge)]
    rules_v = [(435, ROWS[0][0], ROWS[-1][1])]
    return boxes, rules_h, rules_v


def unit_features(boxes, texts, rules_h, rules_v):
    """`features.rs::unit_features` with `proj = None`, on a `(1024,1024)` page."""
    (w_pg, h_pg) = (float(IMG_SIDE), float(IMG_SIDE))
    n = len(boxes)
    h = [max(float(b[3] - b[1]), 0.5) for b in boxes]
    hs = [float(b[3] - b[1]) for b in boxes]  # the rule features use the unclamped height
    hm = float(np.median(np.array(h, dtype=np.float32)))
    mid = [(b[1] + b[3]) / 2.0 for b in boxes]
    feats = np.zeros((n, FEAT_DIM), dtype=np.float32)
    for i in range(n):
        n_row = 0.0
        gap_l = gap_r = 20.0
        lmax, rmin = -math.inf, math.inf
        has_l = has_r = False
        for j in range(n):
            same = abs(mid[j] - mid[i]) < 0.5 * max(h[j], h[i])
            if not same:
                continue
            n_row += 1.0
            if j == i:
                continue
            if boxes[j][2] <= boxes[i][0] + 0.5 * h[i]:
                has_l = True
                lmax = max(lmax, boxes[j][2])
            if boxes[j][0] >= boxes[i][2] - 0.5 * h[i]:
                has_r = True
                rmin = min(rmin, boxes[j][0])
        if has_l:
            gap_l = min(20.0, (boxes[i][0] - lmax) / hm)
        if has_r:
            gap_r = min(20.0, (rmin - boxes[i][2]) / hm)
        cx = (boxes[i][0] + boxes[i][2]) / 2.0 / w_pg
        cy = (boxes[i][1] + boxes[i][3]) / 2.0 / h_pg
        text = texts[i]
        length = float(len(text))
        n_alpha = float(sum(1 for c in text if c.isalpha()))
        dig = sum(1 for c in text if c.isdigit()) / max(length, 1.0)
        alpha = n_alpha / max(length, 1.0)
        upper = sum(1 for c in text if c.isupper()) / max(n_alpha, 1.0)
        words = float(len(text.split()))
        row = feats[i]
        row[0:21] = [
            cx, cy,
            (boxes[i][2] - boxes[i][0]) / w_pg,
            (boxes[i][3] - boxes[i][1]) / h_pg,
            boxes[i][0] / w_pg, boxes[i][2] / w_pg,
            boxes[i][1] / h_pg, boxes[i][3] / h_pg,
            h[i] / hm,
            math.log1p(length) / 5.0,
            math.log1p(words) / 3.0,
            dig, alpha, upper,
            1.0 if NUMERIC.match(text) else 0.0,
            1.0 if text.rstrip().endswith(('.', ':', ';', ',')) else 0.0,
            math.log1p(n_row) / 3.0,
            gap_l / 20.0, gap_r / 20.0,
            0.0, 0.0,  # bold, bold_known: the OCR path has no font information
        ]
        k = 21
        for q in range(N_FOURIER):
            fr = (2.0 ** q) * math.pi
            row[k], row[k + 1] = math.sin(fr * cx), math.cos(fr * cx)
            row[k + 2], row[k + 3] = math.sin(fr * cy), math.cos(fr * cy)
            k += 4
        k += 8  # proj = None: features 53..60 stay zero
        hh = max(hs[i], 0.5)
        for (y, x0, x1) in rules_h:
            if min(boxes[i][2], x1) - max(boxes[i][0], x0) <= 0.0:
                continue
            above = boxes[i][1] - y
            below = y - boxes[i][3]
            if -0.3 * hh <= above < 1.5 * hh:
                row[k + 0] = 1.0
            if -0.3 * hh <= below < 1.5 * hh:
                row[k + 1] = 1.0
        for (x, y0, y1) in rules_v:
            if min(boxes[i][3], y1) - max(boxes[i][1], y0) <= 0.0:
                continue
            left = boxes[i][0] - x
            right = x - boxes[i][2]
            if -0.3 * hh <= left < 2.0 * hh:
                row[k + 2] = 1.0
            if -0.3 * hh <= right < 2.0 * hh:
                row[k + 3] = 1.0
        row[k + 4] = 1.0  # rules present
        row[66] = -1.0  # block id: no projection signals on this path
        row[67] = -1.0  # leaf id
    return feats


def pair_features(boxes, rules_h, rules_v):
    """`features.rs::pair_rule_features`: `[N,N,2]`, rules strictly between the centres."""
    n = len(boxes)
    out = np.zeros((n, n, 2), dtype=np.float32)
    cap = 3.0
    cy = [(b[1] + b[3]) / 2.0 for b in boxes]
    cx = [(b[0] + b[2]) / 2.0 for b in boxes]
    hcov = [[r[1] <= b[2] and r[2] >= b[0] for r in rules_h] for b in boxes]
    vcov = [[r[1] <= b[3] and r[2] >= b[1] for r in rules_v] for b in boxes]
    for i in range(n):
        for j in range(n):
            lo, hi = min(cy[i], cy[j]), max(cy[i], cy[j])
            count = sum(1 for r, rule in enumerate(rules_h)
                        if lo < rule[0] < hi and hcov[i][r] and hcov[j][r])
            out[i, j, 0] = min(float(count), cap) / cap
            lo, hi = min(cx[i], cx[j]), max(cx[i], cx[j])
            count = sum(1 for r, rule in enumerate(rules_v)
                        if lo < rule[0] < hi and vcov[i][r] and vcov[j][r])
            out[i, j, 1] = min(float(count), cap) / cap
    return out


def canvas(boxes, rules_h, rules_v, valid_w):
    """The page raster as 8-bit ink: 0 = white, 255 = black, top-left aligned.

    The model's canvas is `(255-px)/255`, so the fixture stores the 8-bit raster and
    the reader divides: `canvas = ink as f32 / 255.0` is exact for every value here,
    which is why the two ink levels are `INK_TEXT = 191` and `INK_RULE = 255` rather
    than 0.75 and 1.0 -- 0.75 is not a multiple of 1/255 and would not survive the
    round trip.
    """
    img = np.zeros((IMG_SIDE, IMG_SIDE), dtype=np.uint8)
    for (y, x0, x1) in rules_h:
        img[y:y + 2, x0:x1] = INK_RULE
    for (x, y0, y1) in rules_v:
        img[y0:y1, x:x + 2] = INK_RULE
    for i, b in enumerate(boxes):
        (x0, y0, x1, y1) = (int(v) for v in b)
        # one ink bar per word, 2 px of white between them: a text line, not a block
        words = max(len(TEXTS[i // 2][i % 2].split()), 1)
        width = (x1 - x0) // words
        for word in range(words):
            start = x0 + word * width + 2
            stop = min(start + width - 4, x1)
            img[y0 + 6:y1 - 6, start:stop] = INK_TEXT
    if valid_w < 1.0:
        img[:, int(round(valid_w * IMG_SIDE)):] = 0
    return img


def blobs_for(model_bytes, case, names):
    """Run the graph once for `case` and return `[(name, array, description)]`."""
    valid_w = case['valid_w']
    boxes, rules_h, rules_v = canvas_boxes(valid_w)
    texts = [TEXTS[i // 2][i % 2] for i in range(len(boxes))]
    feats = unit_features(boxes, texts, rules_h, rules_v)
    pairf = pair_features(boxes, rules_h, rules_v)
    ink = np.zeros((IMG_SIDE, IMG_SIDE), dtype=np.uint8) if case['blank'] \
        else canvas(boxes, rules_h, rules_v, valid_w)
    n = len(boxes)
    feeds = {
        'feats': feats.reshape(1, n, FEAT_DIM),
        'mask': np.ones((1, n), dtype=bool),
        'pairf': pairf.reshape(1, n, n, 2),
        'image': (ink.astype(np.float32) / 255.0).reshape(1, 1, IMG_SIDE, IMG_SIDE),
        'boxes': np.asarray(boxes, dtype=np.float32).reshape(1, n, 4) / IMG_SIDE,
        'valid': np.asarray([valid_w, 1.0], dtype=np.float32).reshape(1, 2),
    }
    session = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
    wanted = ['node', 'pair', 'cls'] + [INTERMEDIATES[name][0] for name in names]
    outputs = session.run(wanted, feeds)
    named = dict(zip(wanted, outputs))
    rows = [(name, np.asarray(feeds[name]), 'graph input') for name in
            ('feats', 'mask', 'pairf', 'boxes', 'valid')]
    source = 'graph input raster: all white, the canvas is zeros' if case['blank'] \
        else 'graph input raster (8-bit ink); see `decode`'
    rows.insert(3, ('image', ink, source))
    rows += [('node', named['node'], 'graph output (in_table, header)'),
             ('pair', named['pair'], 'graph output, 5 pair heads'),
             ('cls', named['cls'], 'graph output, 13 block classes')]
    rows += [(name, named[INTERMEDIATES[name][0]], INTERMEDIATES[name][1]) for name in names]
    return rows


def dtype_name(array):
    if array.dtype == np.bool_:
        return 'bool'
    return str(array.dtype)


def write_case(directory, case, rows, blank):
    """Write one blob per row; returns the manifest entries.

    A blank raster is not written at all: the entry carries `fill: 0` and no file, so
    the all-white case costs nothing on disk and a reader builds `zeros` instead of
    decoding 1 MB of them.
    """
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, array, why in rows:
        if name == 'image' and blank:
            entries.append({
                'name': name,
                'file': None,
                'shape': list(array.shape),
                'dtype': dtype_name(array),
                'bytes': None,
                'sha256': None,
                'fill': 0,
                'decode': 'canvas[1,1,1024,1024] = zeros',
                'decoded_sha256': hashlib.sha256(
                    np.zeros((1, 1, IMG_SIDE, IMG_SIDE), dtype='<f4').tobytes()).hexdigest(),
                'source': why,
            })
            continue
        if array.dtype == np.bool_:
            suffix, raw = '.u8', array.astype(np.uint8).tobytes()
        elif array.dtype == np.uint8:
            suffix, raw = '.u8', np.ascontiguousarray(array, dtype=np.uint8).tobytes()
        else:
            suffix, raw = '.f32', np.ascontiguousarray(array, dtype='<f4').tobytes()
        path = directory / f'{name}{suffix}'
        path.write_bytes(raw)
        entry = {
            'name': name,
            'file': f'{case}/{path.name}',
            'shape': list(array.shape),
            'dtype': dtype_name(array),
            'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest(),
            'source': why,
        }
        if name == 'image':
            # the model's input is the f32 canvas; the committed blob is the 8-bit
            # raster, and this is the exact conversion plus a hash to check it by
            canvas = array.astype(np.float32) / 255.0
            entry['decode'] = 'canvas[1,1,1024,1024] = ink[1024,1024] as f32 / 255.0'
            entry['decoded_sha256'] = hashlib.sha256(
                canvas.astype('<f4').tobytes()).hexdigest()
        entries.append(entry)
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--force', action='store_true', help='overwrite existing blobs')
    parser.add_argument('--with-intermediates', action='store_true',
                        help='also write the three big intermediates (cnn_map 1 MB, '
                             'avgpool 256 KB, tokens 1 MB per case)')
    args = parser.parse_args()
    model_path = args.model_dir / 'tablenet.onnx'
    model_bytes = model_path.read_bytes()
    sha = hashlib.sha256(model_bytes).hexdigest()
    model = onnx.load_from_string(model_bytes)
    config = json.loads({p.key: p.value for p in model.metadata_props}['liteparse.tablenet'])
    # the intermediates are read out of an in-memory copy: the pinned file is untouched
    present = {o.name for o in model.graph.output}
    wanted = {INTERMEDIATES[name][0] for case in CASES for name in case['intermediates']}
    if args.with_intermediates:
        wanted |= {INTERMEDIATES[name][0] for name in HEAVY_INTERMEDIATES}
    for tensor in sorted(wanted):
        if tensor not in present:
            model.graph.output.append(onnx.helper.make_empty_tensor_value_info(tensor))
    extended = model.SerializeToString()
    manifest_path = args.out_dir / 'manifest.json'
    if manifest_path.exists() and not args.force:
        raise SystemExit(f'{manifest_path} exists; pass --force to regenerate')
    cases = []
    for case in CASES:
        names = case['intermediates']
        if args.with_intermediates:
            names = names + tuple(name for name in HEAVY_INTERMEDIATES if name not in names)
        rows = blobs_for(extended, case, names)
        entries = write_case(args.out_dir / case['name'], case['name'], rows, case['blank'])
        entry = {'name': case['name'], 'units': int(np.asarray(rows[0][1]).shape[1]),
                 'valid': [case['valid_w'], 1.0], 'blobs': entries}
        if not case['blank']:
            # the raster preview is for humans; a blank page has nothing to look at
            Image.fromarray(np.asarray(rows[3][1]).reshape(IMG_SIDE, IMG_SIDE)).save(
                args.out_dir / case['name'] / 'canvas.png')
            entry['preview'] = f"{case['name']}/canvas.png"
        cases.append(entry)
    manifest = {
        'source': {
            'file': 'tablenet.onnx',
            'sha256': sha,
            'bytes': len(model_bytes),
            'opset': model.opset_import[0].version,
            'config': config,
        },
        'generator': 'python -m tools.oracle.ocria.layout_fixtures.make_fixtures',
        'onnxruntime': ort.__version__,
        'encoding': {
            'float32': 'raw little-endian f32, row-major, shape as listed',
            'uint8': 'raw bytes, row-major; `image` is ink (0 = white, 255 = black)',
            'bool': 'one byte per element, 0 or 1',
        },
        'intermediates': 'full' if args.with_intermediates else 'light',
        'cases': [case['name'] for case in CASES],
        'cases': cases,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(manifest_path)


if __name__ == '__main__':
    main()
