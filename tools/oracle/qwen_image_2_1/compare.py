#!/usr/bin/env python3
"""Compare complete component checkpoints as little-endian F32 bits. No tolerance."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import struct
import zlib

if not __debug__:
    raise RuntimeError("Bitwise verification requires Python assertions (no -O/PYTHONOPTIMIZE)")


def records(path):
    result = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert result, f"empty trace: {path}"
    for row in result:
        assert math.prod(row["shape"]) == row["len"], row["name"]
    return result


def data(row):
    if 'parts' in row:
        return b''.join(data(r) for r in row['parts'])
    raw = Path(row["binary_path"]).read_bytes()
    assert len(raw) == row["len"] * 4, row["binary_path"]
    return raw


def shape(row):
    # GGML removes unit batch/time axes; inserting one never changes storage order.
    return [n for n in row["shape"] if n != 1] or [1]


def compare(rust, oracle):
    assert [r["name"] for r in rust] == [r["name"] for r in oracle], "checkpoint names/order/count"
    count = 0
    for r, o in zip(rust, oracle):
        assert shape(r) == shape(o), f"{r['name']} shape: {r['shape']} != {o['shape']}"
        a, b = data(r), data(o)
        assert len(a) == len(b), r["name"]
        if a != b:
            index = next(i for i in range(len(a)//4) if a[4*i:4*i+4] != b[4*i:4*i+4])
            bits = [struct.unpack_from('<I', x, 4*index)[0] for x in (a, b)]
            raise AssertionError(f"{r['name']} occurrence {r.get('occurrence', 0)}, first u32[{index}]: {bits[0]:08x} != {bits[1]:08x}")
        assert all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', a)), r["name"]
        count += len(a)//4
    return {"checkpoints": len(rust), "f32_bits": count}


def text_rows(rows, rust, references):
    names = ["qwen.text.tokens", "qwen.text.embeddings"] + [f"qwen.text.block.{i}" for i in range(36)] + ["qwen.text.context"]
    if references:
        names.append("qwen.text.image_slots")
    tokens = [r for r in rows if r["name"] == "qwen.text.tokens"]
    assert len(tokens) == 1, "one conditioning per text trace"
    n = tokens[0]["len"]
    if rust:
        hidden = [r for r in rows if r["name"].startswith("hidden_sequence.layer.")]
        assert [r["name"] for r in hidden] == [f"hidden_sequence.layer.{i}" for _ in range(n) for i in range(36)], "text token/layer order"
        for i in range(36):
            group = [r for r in hidden if r["name"] == f"hidden_sequence.layer.{i}"]
            assert all(r["len"] == 4096 for r in group)
            rows = rows + [{"name": f"qwen.text.block.{i}", "shape": [4096, n], "len": 4096*n, "parts": group}]
    selected = [r for r in rows if r["name"] in names]
    assert Counter(r["name"] for r in selected) == Counter(names), "all 36 text layers required"
    # Rust emits token-major, GGML layer-major. Check each native order above,
    # then compare the same contiguous [hidden, tokens] storage.
    return [next(r for r in selected if r["name"] == name) for name in names]


def vae_name(name):
    name = name.removeprefix("vae.conv.first_stage_model.").removeprefix("vae.norm.first_stage_model.")
    name = name.removesuffix(" (reshaped)").removesuffix(".weight").removesuffix(".gamma")
    name = re.sub(r"^(encoder|decoder)\.conv1$", r"\1.conv_in", name)
    name = re.sub(r"^(encoder|decoder)\.head\.0$", r"\1.norm_out", name)
    name = re.sub(r"^(encoder|decoder)\.head\.2$", r"\1.conv_out", name)
    name = re.sub(r"\.middle\.([02])", lambda m: ".mid_block.resnets." + str(int(m[1])//2), name)
    name = name.replace(".middle.1", ".mid_block.attentions.0")
    name = re.sub(r"decoder\.upsamples\.(\d+)\.upsamples\.3", r"decoder.up_blocks.\1.upsampler", name)
    name = re.sub(r"decoder\.upsamples\.(\d+)\.upsamples\.(\d+)", r"decoder.up_blocks.\1.resnets.\2", name)
    name = re.sub(r"encoder\.downsamples\.(\d+)\.downsamples\.2", r"encoder.down_blocks.\1.downsampler", name)
    name = re.sub(r"encoder\.downsamples\.(\d+)\.downsamples\.(\d+)", r"encoder.down_blocks.\1.resnets.\2", name)
    for old, new in [("0", "norm1"), ("2", "conv1"), ("3", "norm2"), ("6", "conv2")]:
        name = name.replace(".residual."+old, "."+new)
    name = name.replace(".shortcut", ".conv_shortcut")
    name = {"conv1": "quant_conv", "conv2": "post_quant_conv"}.get(name, name)
    return "qwen.vae." + name


def vae_rows(rust, oracle, encode, reference):
    chosen = []
    for row in oracle:
        if row["name"].startswith(("vae.conv.", "vae.norm.")):
            row = dict(row, name=vae_name(row["name"]))
        # qkv is already captured by its conv checkpoint; skip this duplicate alias.
        if row["name"].startswith("qwen.vae.") and row["name"] != "qwen.vae.attention.qkv":
            chosen.append(row)
    assert len(chosen) == (65 if encode else 85), "complete VAE conv/norm/attention/output checkpoints required"
    selected = []
    for o in chosen:
        matches = [r for r in rust if r["name"] == o["name"]]
        index = reference if encode or "attention.values" not in o["name"] else len([r for r in rust if r["name"] == "qwen.vae.encoded"])
        assert len(matches) > index, f"missing {o['name']} occurrence {index}"
        selected.append(matches[index])
    return selected, chosen


def png_rgba(path):
    raw = Path(path).read_bytes()
    assert raw[:8] == b'\x89PNG\r\n\x1a\n'
    pos, chunks = 8, []
    while pos < len(raw):
        n = struct.unpack_from('>I', raw, pos)[0]
        tag, payload = raw[pos+4:pos+8], raw[pos+8:pos+8+n]
        assert zlib.crc32(tag+payload) & 0xffffffff == struct.unpack_from('>I', raw, pos+8+n)[0]
        if tag == b'IHDR':
            w, h, depth, color, comp, filt, interlace = struct.unpack('>IIBBBBB', payload)
            assert (depth,color,comp,filt,interlace) == (8,6,0,0,0), "expected RGBA8 PNG"
        if tag == b'IDAT': chunks.append(payload)
        pos += n+12
    packed = zlib.decompress(b''.join(chunks))
    stride = w*4
    assert len(packed) == h*(stride+1)
    output, previous = bytearray(), bytearray(stride)
    for y in range(h):
        row = bytearray(packed[y*(stride+1)+1:(y+1)*(stride+1)])
        method = packed[y*(stride+1)]
        assert 0 <= method <= 4
        for x in range(stride):
            a, b, c = row[x-4] if x >= 4 else 0, previous[x], previous[x-4] if x >= 4 else 0
            p = a+b-c
            pa, pb, pc = abs(p-a), abs(p-b), abs(p-c)
            predictor = [0,a,b,(a+b)//2,a if pa <= pb and pa <= pc else b if pb <= pc else c][method]
            row[x] = (row[x]+predictor) & 255
        output.extend(row)
        previous = row
    return w, h, bytes(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['dit','sample','text','vision','vae','vae_encode'])
    p.add_argument('rust_trace')
    p.add_argument('oracle_trace')
    p.add_argument('--evaluations', type=int, default=1)
    p.add_argument('--references', type=int, default=0)
    p.add_argument('--reference', type=int, default=0)
    p.add_argument('--png')
    args = p.parse_args()
    rust, oracle = records(args.rust_trace), records(args.oracle_trace)
    if args.mode == 'text':
        rust, oracle = text_rows(rust, True, args.references), text_rows(oracle, False, args.references)
    elif args.mode == 'vision':
        required = Counter({'qwen.vision.embedding': args.references, 'qwen.vision.output': args.references})
        required.update({f'qwen.vision.block.{i}': args.references for i in range(27)})
        required.update({f'qwen.vision.deepstack.{i}': args.references for i in range(3)})
        rust = [r for r in rust if r['name'].startswith('qwen.vision.')]
        oracle = [r for r in oracle if r['name'].startswith('qwen.vision.') and r['name'] != 'qwen.vision.patch']
        assert Counter(r['name'] for r in rust) == required, 'all vision layers and DeepStack levels required'
    elif args.mode in ('vae','vae_encode'):
        rust, oracle = vae_rows(rust, oracle, args.mode == 'vae_encode', args.reference)
    else:
        def select(rows):
            return [r for r in rows if r['name'].startswith('qwen.') and not r['name'].startswith(('qwen.text.','qwen.vision.','qwen.vae.'))]
        rust, oracle = select(rust), select(oracle)
        required = ['pe','time_embed','modulation','txt_in','joint','joint_final','scale','norm_out','out','input.x','input.context','input.timesteps','output']
        counts = Counter(r['name'] for r in rust)
        assert all(counts['qwen.'+n] == args.evaluations for n in required)
        assert all(counts[f'qwen.block.{i}'] == args.evaluations for i in range(32))
        if args.mode == 'sample':
            steps = next(r['len']-1 for r in rust if r['name'] == 'qwen.sample.sigmas')
            assert counts['qwen.sample.noise'] == counts['qwen.sample.sigmas'] == 1
            assert counts['qwen.sample.velocity'] == counts['qwen.sample.latent'] == steps
    result = compare(rust, oracle)
    if args.png:
        w, h, pixels = png_rgba(args.png)
        values = [v[0] for v in struct.iter_unpack('<f', data(next(r for r in oracle if r['name'] == 'qwen.vae.output')))]
        # Reproduce each F32 rounding before sd.cpp's truncating byte conversion.
        f32 = lambda v: struct.unpack('<f', struct.pack('<f', v))[0]
        expected = bytes(int(max(0,min(255,f32(f32(f32(values[c*w*h+i]+1)*0.5)*255)))) for i in range(w*h) for c in range(4))
        assert pixels == expected, 'final RGBA PNG pixels (including alpha)'
        result['rgba_bytes'] = len(pixels)
    print(json.dumps(dict(mode=args.mode, **result)))


if __name__ == '__main__':
    main()
