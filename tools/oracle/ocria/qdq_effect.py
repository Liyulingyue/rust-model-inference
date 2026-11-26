"""Measure what `rec_latin`'s activation quantization costs, on `ids` and `conf`.

`rec_latin.onnx` is the one ONNX-Runtime-quantized graph: its weights are int8
(dequantized back to F32 by the converter) and every activation passes a
`QuantizeLinear` -> `DequantizeLinear` pair. Rust computes the same graph in F32
with the dequantized weights and no activation quantization, so
`trace.f32_graph` builds that F32 graph -- in memory, from the pinned file -- and
this script runs both graphs on the same inputs.

It writes `latin_qdq_effect.json` next to itself: per case, the timestep count,
how many timesteps' `ids` differ, and the text each side decodes to. That is the
acceptance record for the precision loss `graph_notes.md` §8 describes, and the
reason `rec.latin`'s fixtures carry both an F32 target and the QDQ answer.

The difference is explained, not chased: activation quantization perturbs the
logits (by up to ~8 in the tail, compounding through 29 layers), and the argmax
follows only where the top-2 gap is small against that perturbation. On the
synthetic stroke fixtures that happens 5 % of the time because the pattern leaves
many near-ties; on real receipt text it is 0.6 %.

    python -m tools.oracle.ocria.qdq_effect /path/to/OCRia-nano-1 \\
        --page receipt.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from tools.oracle.ocria.trace import (
    REC_WIDTHS,
    extended,
    f32_graph,
    normalize,
    session,
    strokes,
)

HERE = Path(__file__).resolve().parent
LOGITS = '/m/head/Conv_output_0_DequantizeLinear_Output'
F32_LOGITS = '/m/head/Conv_output_0'


def decode(ids, charset):
    """CTC collapse: drop the blank (id 0) and consecutive repeats, then map."""
    out, previous = [], 0
    for value in ids:
        if value != 0 and value != previous:
            out.append(charset[value - 1])
        previous = value
    return ''.join(out)


def cases(seed_base):
    """The same deterministic strips the recognizer fixtures use."""
    return [(f'w{width}', normalize(strokes(width, seed_base + step))[None, None])
            for step, width in enumerate(REC_WIDTHS)]


def page_cases(path):
    """Real-text cross-check: 32-px bands of a page image, skipping blank ones.

    The fixtures are synthetic strokes, a hostile case for the argmax (many
    near-ties). This measures the same loss on real glyphs with the recognizer
    preprocessing from `graph_notes.md` §3.
    """
    from PIL import Image

    page = Image.open(path).convert('L')
    bands = []
    for top in range(0, page.size[1] - 32, 48):
        crop = page.crop((20, top, page.size[0] - 20, top + 32))
        if min(crop.getdata()) > 240:
            continue
        width = crop.size[0]
        line = crop.resize((width, 32), Image.BOX)
        padded = Image.new('L', (max(128, (width + 127) // 128 * 128), 32), 255)
        padded.paste(line, (0, 0))
        array = np.asarray(padded, dtype=np.float32) / np.float32(127.5) - np.float32(1.0)
        bands.append((f'y{top}', array[None, None]))
    return bands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('report', type=Path, nargs='?',
                        default=HERE / 'latin_qdq_effect.json')
    parser.add_argument('--page', type=Path,
                        help='also measure on 32-px bands of a real page image')
    args = parser.parse_args()
    quantized_path = args.model_dir / 'rec_latin.onnx'
    quantized = session(extended(quantized_path, [LOGITS]))
    plain = session(extended(f32_graph(quantized_path), [F32_LOGITS]))
    # the head output's quantisation step: logits are snapped to this grid before
    # the argmax, so a top-2 gap under it can be flipped by the grid alone
    initialisers = {init.name: numpy_helper.to_array(init)
                    for init in onnx.load(str(quantized_path)).graph.initializer}
    scale = float(np.ravel(initialisers['/m/head/Conv_output_0_scale'])[0])
    charset = json.loads((args.model_dir / 'rec_latin.charset.json').read_text())

    def measure(name, x):
        q_ids, q_conf, q_logits = quantized.run(['ids', 'conf', LOGITS], {'x': x})
        p_ids, p_conf, p_logits = plain.run(['ids', 'conf', F32_LOGITS], {'x': x})
        mismatch = q_ids != p_ids
        margins = np.sort(p_logits[0], axis=0)[-2:]
        row = dict(
            case=name, timesteps=int(q_ids.size), ids_differing=int(mismatch.sum()),
            text_f32=decode(p_ids[0], charset), text_qdq=decode(q_ids[0], charset),
            differing_at=[int(i) for i in np.flatnonzero(mismatch.ravel())][:16],
            max_conf_delta=float(np.abs(q_conf - p_conf).max()),
            max_logit_delta=float(np.abs(q_logits - p_logits).max()),
            max_top2_gap_where_differing=float((margins[1] - margins[0])[mismatch.ravel()].max())
            if mismatch.any() else 0.0,
        )
        print(json.dumps(row))
        return row

    fixtures = [measure(name, x) for name, x in cases(seed_base=0)]
    real_text = [measure(name, x) for name, x in page_cases(args.page)] if args.page else []

    def total(rows):
        steps = sum(row['timesteps'] for row in rows)
        differing = sum(row['ids_differing'] for row in rows)
        return dict(timesteps=steps, ids_differing=differing,
                    percent=round(100.0 * differing / max(steps, 1), 3))

    report = dict(
        graph='rec_latin.onnx',
        head_output_scale=scale,
        note='F32 side: the dequantized weights the GGUF carries, activation '
             'quantization rewired out, the Relus it folded in put back. QDQ side: '
             'the pinned graph as ONNX Runtime runs it. `text_*` is the raw CTC '
             'decode before any clean-up.',
        fixtures=fixtures,
        real_text=real_text,
        totals=dict(fixtures=total(fixtures), real_text=total(real_text)),
    )
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(f'fixtures {report["totals"]["fixtures"]}, real text {report["totals"]["real_text"]}')
    print(args.report)


if __name__ == '__main__':
    main()
