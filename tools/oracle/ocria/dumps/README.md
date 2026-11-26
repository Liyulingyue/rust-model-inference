# Raw op listings of the six OCRia-nano-1 graphs

One file per graph, in ONNX topological order: graph inputs/outputs with dtype and
shape, then every node that is not `Constant` / `QuantizeLinear` / `DequantizeLinear`.
`Conv` and `Gemm`/`MatMul` rows carry the weight and bias shapes, kernel, strides,
pads and group; every other row carries its attributes.

`rec_latin.txt` shows the **dequantized** weight shapes, because the weights are
resolved through the converter's own `Graph` — those are exactly the shapes the GGUF
stores. That graph has no `Relu` nodes at all (they are folded into its activation
quantizers); see §8.1 of `../graph_notes.md` before reading it.

These are a convenience for reading and review. The authoritative description of
what to implement is `../graph_notes.md`, and the authoritative tensor list is
`src/models/ocria/contract.json`. Regenerate them from the pinned graphs with:

```bash
python -m tools.oracle.ocria.dump_graphs /path/to/OCRia-nano-1 tools/oracle/ocria/dumps
```

The pinned SHA256 of each graph is in `../sources.json`, and
`tools/oracle/ocria/download.py` re-fetches them.
