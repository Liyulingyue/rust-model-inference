#!/usr/bin/env python3
"""Read a GGUF header (KV metadata + tensor table) from a partial file.

GGUF stores the magic, the KV block and the tensor-info table *before* the
tensor payload, so a byte range covering the first few megabytes is enough to
recover the whole contract: metadata keys, tensor names, shapes and types.

Usage:
    python3 gguf_header.py <local-path>            # parse a local file
    python3 gguf_header.py <local-path> --meta      # metadata only
"""

import json
import struct
import sys

GGUF_TYPE_NAMES = {
    0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16", 4: "UINT32", 5: "INT32",
    6: "FLOAT32", 7: "BOOL", 8: "STRING", 9: "ARRAY", 10: "UINT64", 11: "INT64",
    12: "FLOAT64",
}
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "IQ1_M",
    25: "BF16", 30: "TQ1_0", 31: "TQ2_0", 34: "MXFP4",
}


class Reader:
    def __init__(self, blob: bytes):
        self.blob = blob
        self.pos = 0

    def take(self, n: int) -> bytes:
        out = self.blob[self.pos:self.pos + n]
        if len(out) != n:
            raise EOFError(f"want {n} at {self.pos}, have {len(out)}")
        self.pos += n
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.take(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.take(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.take(8))[0]

    def boolean(self) -> bool:
        return self.take(1) != b"\x00"

    def string(self) -> str:
        length = self.u64()
        return self.take(length).decode("utf-8", errors="replace")

    def value(self, kind: int):
        if kind == 0:
            return self.u64()
        if kind == 1:
            return struct.unpack("<b", self.take(1))[0]
        if kind == 2:
            return struct.unpack("<H", self.take(2))[0]
        if kind == 3:
            return struct.unpack("<h", self.take(2))[0]
        if kind == 4:
            return self.u32()
        if kind == 5:
            return struct.unpack("<i", self.take(4))[0]
        if kind == 6:
            return self.f32()
        if kind == 7:
            return self.boolean()
        if kind == 8:
            return self.string()
        if kind == 9:
            item = self.u32()
            count = self.u64()
            return [self.value(item) for _ in range(count)]
        if kind == 10:
            return self.u64()
        if kind == 11:
            return self.i64()
        if kind == 12:
            return self.f64()
        raise ValueError(f"unknown gguf value type {kind}")


def parse(blob: bytes):
    reader = Reader(blob)
    if reader.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version = reader.u32()
    tensor_count = reader.u64()
    kv_count = reader.u64()

    meta = {}
    for _ in range(kv_count):
        key = reader.string()
        kind = reader.u32()
        meta[key] = reader.value(kind)

    tensors = []
    for _ in range(tensor_count):
        name = reader.string()
        n_dims = reader.u32()
        dims = [reader.u64() for _ in range(n_dims)]
        ggml_type = reader.u32()
        offset = reader.u64()
        tensors.append({"name": name, "dims": dims, "type": ggml_type, "offset": offset})
    return version, meta, tensors


def main() -> int:
    path = sys.argv[1]
    meta_only = "--meta" in sys.argv
    blob = open(path, "rb").read()
    version, meta, tensors = parse(blob)
    print(f"# gguf v{version}: {len(meta)} kv, {len(tensors)} tensors "
          f"(parsed {len(blob)} bytes)")
    print("\n## metadata")
    for key, value in meta.items():
        if key.startswith("tokenizer.ggml.tokens") or key.startswith("tokenizer.ggml.merges"):
            value = f"<{len(value)} entries>"
        if isinstance(value, list) and len(value) > 16:
            value = f"<{len(value)} entries> {value[:16]}..."
        if isinstance(value, str) and len(value) > 120:
            value = f"<{len(value)} chars>"
        print(f"{key} = {value!r}")
    if meta_only:
        return 0
    print("\n## tensors")
    for tensor in tensors:
        dims = "x".join(str(d) for d in tensor["dims"])
        print(f"{tensor['name']:60s} {dims:24s} {GGML_TYPE_NAMES.get(tensor['type'], tensor['type'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
