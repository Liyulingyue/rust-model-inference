#!/usr/bin/env python3
"""Build a tokenizer-only GGUF from a full GGUF header (metadata section).

Usage: head_to_tokenizer_gguf.py <header-file> <out.gguf>

Keeps `general.*` and `tokenizer.ggml.*` keys (all that BPETokenizer needs),
drops everything else, writes zero tensors. Lets the Rust tokenizer be checked
against llama.cpp's token IDs without shipping the weights anywhere.
"""
import struct
import sys

T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL, \
    T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = range(13)


class R:
    def __init__(self, b):
        self.b, self.o = b, 0

    def take(self, n):
        v = self.b[self.o:self.o + n]
        if len(v) != n:
            raise EOFError(f"need {n} at {self.o}")
        self.o += n
        return v

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def s(self):
        return self.take(self.u64())


def read_value(r, t):
    """Return (python_value, raw_bytes) so we can re-emit byte-identically."""
    start = r.o
    if t == T_UINT8:
        r.take(1)
    elif t == T_INT8:
        r.take(1)
    elif t == T_UINT16:
        r.take(2)
    elif t == T_INT16:
        r.take(2)
    elif t == T_UINT32:
        r.take(4)
    elif t == T_INT32:
        r.take(4)
    elif t == T_FLOAT32:
        r.take(4)
    elif t == T_BOOL:
        r.take(1)
    elif t == T_STRING:
        r.s()
    elif t == T_UINT64:
        r.take(8)
    elif t == T_INT64:
        r.take(8)
    elif t == T_FLOAT64:
        r.take(8)
    elif t == T_ARRAY:
        et = r.u32()
        n = r.u64()
        for _ in range(n):
            read_value(r, et)
    else:
        raise ValueError(f"bad type {t}")
    return r.b[start:r.o]


def main(src, dst):
    b = open(src, "rb").read()
    r = R(b)
    assert r.take(4) == b"GGUF"
    ver = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()
    kept = []
    for _ in range(n_kv):
        key = r.s()
        t = r.u32()
        raw = read_value(r, t)
        name = key.decode("utf-8", "replace")
        if name.startswith("general.") or name.startswith("tokenizer.ggml."):
            kept.append((key, t, raw))
    out = bytearray(b"GGUF")
    out += struct.pack("<I", ver)
    out += struct.pack("<Q", 0)          # tensor_count
    out += struct.pack("<Q", len(kept))  # kv_count
    for key, t, raw in kept:
        out += struct.pack("<Q", len(key)) + key
        out += struct.pack("<I", t)
        out += raw
    out += b"\0" * ((-len(out)) % 32)     # alignment padding
    open(dst, "wb").write(out)
    print(f"{dst}: {len(kept)} kv, {len(out)} bytes (source had {n_tensors} tensors, {n_kv} kv)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
