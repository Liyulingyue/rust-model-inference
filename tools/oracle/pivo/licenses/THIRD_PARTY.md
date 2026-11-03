# Third-party materials

PiVo's original implementation and acoustic weights use Apache-2.0 (`LICENSE`). The model repository includes the trained PiVo acoustic checkpoint, converted upstream FARGAN weights, and their ONNX graphs. The GitHub repository contains inference and serving code; model files are obtained separately.

## FARGAN and Opus

The vocoder architecture, generation equations and converted upstream weights derive from Xiph/Opus FARGAN at revision `bafd40dcbd506e12da257b8cb05c0afe1ed44719`. Retain `LICENSES/Opus-BSD-3-Clause.txt` and the upstream patent notices when redistributing these components. Source: <https://github.com/xiph/opus/tree/bafd40dcbd506e12da257b8cb05c0afe1ed44719/dnn/torch/fargan>.

Changes include folded weights, explicit state inputs/outputs, deterministic floating-point evaluation, and a pitch-buffer lower-bound correction matching upstream C.

## Pronunciation dictionary

`pivo/resources/cmudict.dict` (or `src/pivo/resources/`) contains CMU pronunciation dictionary data, Copyright 1993–2015 Carnegie Mellon University. Its complete permissive notice accompanies the data and is duplicated in `LICENSES/CMUDICT_LICENSE`. Dictionary SHA256: `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22`. PiVo uses its own reader and does not distribute or depend on the separate GPL Python cmudict wrapper.

## Teacher and training materials

The synthetic-voice route uses pinned `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` and `Qwen/Qwen3-TTS-12Hz-1.7B-Base` models, both declaring Apache-2.0 in their pinned model cards. Teacher weights, alignment tools, ASR models, audio judges and training recordings are not used at inference and are not included. Teacher revisions and cached-card hashes are recorded in the model repository's evidence files.

Training text combines original scripts with excerpts from original works by Jane Austen, Arthur Conan Doyle, Charles Dickens, Lewis Carroll, Kenneth Grahame and Lucy Maud Montgomery. These original works are public domain in the United States and jurisdictions applying author-life-plus-70-year terms; no worldwide claim is made for every edition. Full text-source provenance stays with the research corpus. Listening-test texts and ratings are not included.

## Design inspiration and runtime dependencies

Mariusz Kurman's [Neuroblast](https://github.com/mkurman/neuroblast-v3) inspired the hybrid acoustic design. PiVo's acoustic implementation is original and uses no Neuroblast weights.

NumPy and ONNX Runtime are installed as dependencies under their own licenses. Optional FastAPI, Uvicorn and Hugging Face Hub dependencies retain their upstream licenses. These repositories do not bundle them; keep their notices if you distribute an application or environment that includes them.
