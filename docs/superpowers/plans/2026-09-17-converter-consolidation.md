# Converter Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every GGUF converter implementation, its converter tests, and its required adjacent data under `tools/converter` without changing exported bytes.

**Architecture:** `tools/converter/<model>/` becomes the only home for conversion code. Model-specific Oracle and trace tools remain under `tools/<model>/` and import converters through `tools.converter`; the dots converter remains the shared writer dependency until a separately verified consolidation is justified.

**Tech Stack:** Python 3.14, `unittest`, NumPy, Git, SHA-256, existing GGUF readers/writers

**Spec:** `docs/superpowers/specs/2026-09-17-converter-consolidation-design.md`

## Global Constraints

- Move paths and imports only; do not change metadata, tensor selection, quantization math, output names, or error contracts.
- Do not retain wrappers, symlinks, or compatibility modules at old converter paths.
- Keep Oracle, trace, README, and build scripts at their existing model directories unless they are converter tests or the Qwen-Drive manifest.
- Never write into `/Users/gouzi/Documents/git/rust-model-inference/models`; use it only as read-only input.
- Preserve the untracked `.codex/` directory and stage only named task files.
- Do not claim full-model byte parity without matching SHA-256 evidence from the same input and converter mode.

---

### Task 1: Record the pre-move contract

**Files:**
- Read: `tools/{breeze,dots,dreamx,neohorse,qwen_drive,vibevoice}/convert_*.py`
- Read: `tools/{breeze,dots,dreamx,neohorse,qwen_drive,vibevoice}/test_convert_*.py`
- Read: `tools/converter/{breeze,vibevoice}/convert_*.py`
- Create outside repository: `/tmp/converter-consolidation-baseline/`

**Interfaces:**
- Consumes: current commit `f78875c` and the eight existing converter test modules.
- Produces: passing test log, CLI `--help` log, and SHA-256 manifest for controlled GGUF outputs generated before any move.

- [ ] **Step 1: Run every converter test independently**

Run:

```bash
PYTHONPATH=.:tools python3 tools/breeze/test_convert_breeze.py
PYTHONPATH=.:tools python3 tools/dots/test_convert_dots_tts.py
PYTHONPATH=.:tools python3 tools/dreamx/test_convert_dreamx_creator.py
PYTHONPATH=.:tools python3 tools/neohorse/test_convert_neohorse.py
PYTHONPATH=.:tools python3 tools/qwen_drive/test_convert_qwen_drive.py
PYTHONPATH=.:tools python3 tools/vibevoice/test_convert_vibevoice_asr.py
PYTHONPATH=.:tools python3 tools/converter/breeze/test_convert_breeze.py
PYTHONPATH=.:tools python3 tools/converter/vibevoice/test_convert_vibevoice_asr.py
```

Expected: `8 + 20 + 21 + 1 + 13 + 10 + 17 + 11 = 101` tests pass. The Breeze extension may emit the existing NumPy F16 overflow/invalid warnings; no test may fail.

- [ ] **Step 2: Capture executable entry points**

Run each converter with `--help` and save stdout under `/tmp/converter-consolidation-baseline/help/`:

```bash
python3 tools/breeze/convert_breeze.py --help
python3 tools/dots/convert_dots_tts.py --help
python3 tools/dreamx/convert_dreamx_creator.py --help
python3 tools/neohorse/convert_neohorse.py --help
python3 tools/qwen_drive/convert_qwen_drive.py --help
python3 tools/vibevoice/convert_vibevoice_asr.py --help
PYTHONPATH=tools python3 tools/converter/breeze/convert_breeze.py --help
PYTHONPATH=tools python3 tools/converter/vibevoice/convert_vibevoice_asr.py --help
```

Expected: all commands exit 0 and describe the same flags later checked after relocation.

- [ ] **Step 3: Generate the controlled byte baseline**

Use a temporary script outside the repository to invoke existing test fixtures and emit deterministic GGUFs for dots BF16/Q8, Breeze BF16/F16/F32/Q8/Q4/Q4_MIXED, VibeVoice BF16/F16/F32/Q8/Q4, DreamX, and Qwen-Drive component writers. Record:

```bash
find /tmp/converter-consolidation-baseline/outputs -type f -name '*.gguf' \
  -exec shasum -a 256 {} + | sort \
  > /tmp/converter-consolidation-baseline/sha256.txt
```

Expected: every generated GGUF has one stable digest; retain the files until Task 9 if space permits, otherwise retain the digest manifest and delete only the exact temporary files with `unlink`.

### Task 2: Move the shared dots converter first

**Files:**
- Move: `tools/dots/convert_dots_tts.py` → `tools/converter/dots/convert_dots_tts.py`
- Move: `tools/dots/test_convert_dots_tts.py` → `tools/converter/dots/test_convert_dots_tts.py`
- Modify: `tools/dreamx/convert_dreamx_creator.py`
- Modify: `tools/dreamx/test_convert_dreamx_creator.py`
- Modify: `tools/qwen_drive/convert_qwen_drive.py`
- Modify: `tools/qwen_drive/test_convert_qwen_drive.py`
- Modify: `tools/vibevoice/convert_vibevoice_asr.py`
- Modify: `tools/vibevoice/test_convert_vibevoice_asr.py`
- Modify: `tools/converter/vibevoice/convert_vibevoice_asr.py`
- Modify: `tools/converter/vibevoice/test_convert_vibevoice_asr.py`

**Interfaces:**
- Consumes: public symbols currently imported from `tools.dots.convert_dots_tts` or top-level `convert_dots_tts`.
- Produces: identical symbols from `tools.converter.dots.convert_dots_tts` and a direct CLI at `tools/converter/dots/convert_dots_tts.py`.

- [ ] **Step 1: Change one consumer test to the new import and verify RED**

Change `tools/qwen_drive/test_convert_qwen_drive.py` imports to:

```python
from tools.converter.dots.convert_dots_tts import (
    GGML_BF16,
    GGML_F32,
    GgufWriter,
    read_gguf_directory,
    read_gguf_tensor_bytes,
)
```

Run:

```bash
PYTHONPATH=. python3 tools/qwen_drive/test_convert_qwen_drive.py
```

Expected: FAIL because `tools.converter.dots.convert_dots_tts` does not exist yet.

- [ ] **Step 2: Move dots and normalize imports**

Move the two files, change their usage strings to `tools/converter/dots/...`, and replace every dots import listed in **Files** with `tools.converter.dots.convert_dots_tts`. For directly executable scripts, insert the repository root only when `__package__ in (None, "")`:

```python
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
```

Remove model-directory `sys.path` insertion and top-level `import convert_dots_tts`; import the package module as `_dots` where private compatibility tables/readers are required.

- [ ] **Step 3: Run dependent tests and CLIs**

Run:

```bash
PYTHONPATH=. python3 tools/converter/dots/test_convert_dots_tts.py
PYTHONPATH=. python3 tools/dreamx/test_convert_dreamx_creator.py
PYTHONPATH=. python3 tools/qwen_drive/test_convert_qwen_drive.py
PYTHONPATH=. python3 tools/vibevoice/test_convert_vibevoice_asr.py
PYTHONPATH=. python3 tools/converter/vibevoice/test_convert_vibevoice_asr.py
python3 tools/converter/dots/convert_dots_tts.py --help
```

Expected: 75 tests pass and the new dots CLI exits 0.

- [ ] **Step 4: Commit the shared dependency move**

Stage only the moved dots files and import edits. Commit:

```bash
git commit -m "move dots converter under converter"
```

### Task 3: Move the original Breeze converter beside the extended converter

**Files:**
- Move: `tools/breeze/convert_breeze.py` → `tools/converter/breeze/convert_breeze_plain.py`
- Move: `tools/breeze/test_convert_breeze.py` → `tools/converter/breeze/test_convert_breeze_plain.py`
- Modify: `tools/breeze/README.md`
- Modify: `docs/usage/breeze.md`

**Interfaces:**
- Consumes: `tools.converter.dots.convert_dots_tts.GgufWriter` and its GGML constants.
- Produces: unquantized `convert(model_dir: Path, out_dir: Path) -> tuple[Path, Path]` at `tools.converter.breeze.convert_breeze_plain`; the extended quantized converter remains `tools.converter.breeze.convert_breeze`.

- [ ] **Step 1: Point the test at the new module and verify RED**

Change the test import to:

```python
from tools.converter.breeze import convert_breeze_plain as convert_breeze
```

Run:

```bash
PYTHONPATH=. python3 tools/breeze/test_convert_breeze.py
```

Expected: FAIL because `convert_breeze_plain` has not been moved yet.

- [ ] **Step 2: Move implementation and test**

Move and rename both files. Replace the dots path injection/import with:

```python
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.converter.dots.convert_dots_tts import GGML_BF16, GGML_F32, GgufWriter, gguf_dims
```

Update the test run docstring and the two Breeze documents to the new plain-converter path. Keep the quantized user documentation pointed at `tools/converter/breeze/convert_breeze.py` where its flags are required.

- [ ] **Step 3: Verify both Breeze implementations**

Run:

```bash
PYTHONPATH=. python3 tools/converter/breeze/test_convert_breeze_plain.py
PYTHONPATH=.:tools python3 tools/converter/breeze/test_convert_breeze.py
python3 tools/converter/breeze/convert_breeze_plain.py --help
PYTHONPATH=tools python3 tools/converter/breeze/convert_breeze.py --help
```

Expected: 25 tests pass; both CLIs exit 0 and expose their original flag sets.

- [ ] **Step 4: Commit the Breeze move**

```bash
git commit -m "move Breeze converters under converter"
```

### Task 4: Move the original VibeVoice converter beside the extended converter

**Files:**
- Move: `tools/vibevoice/convert_vibevoice_asr.py` → `tools/converter/vibevoice/convert_vibevoice_asr_original.py`
- Move: `tools/vibevoice/test_convert_vibevoice_asr.py` → `tools/converter/vibevoice/test_convert_vibevoice_asr_original.py`
- Modify: `tools/vibevoice/vibevoice_llm_oracle.py`
- Modify: `docs/usage/vibevoice.md`

**Interfaces:**
- Consumes: dots writer/types and shared Q4/dtype functions through `tools.converter` package imports.
- Produces: original converter at `tools.converter.vibevoice.convert_vibevoice_asr_original`; extended converter remains `tools.converter.vibevoice.convert_vibevoice_asr`.

- [ ] **Step 1: Change the original test imports and verify RED**

Use:

```python
from tools.converter.dots.convert_dots_tts import Tensor
from tools.converter.vibevoice import convert_vibevoice_asr_original as converter
```

Run the old test path and expect import failure because the new module does not yet exist.

- [ ] **Step 2: Move files and update Oracle import**

Move both files, replace top-level dots imports with package imports, update usage text, and change the Oracle to:

```python
from tools.converter.vibevoice.convert_vibevoice_asr_original import ShardedSafetensors
```

Update `docs/usage/vibevoice.md` to name the intended new converter command explicitly.

- [ ] **Step 3: Verify both VibeVoice implementations and Oracle import**

Run:

```bash
PYTHONPATH=. python3 tools/converter/vibevoice/test_convert_vibevoice_asr_original.py
PYTHONPATH=. python3 tools/converter/vibevoice/test_convert_vibevoice_asr.py
python3 tools/converter/vibevoice/convert_vibevoice_asr_original.py --help
PYTHONPATH=. python3 tools/converter/vibevoice/convert_vibevoice_asr.py --help
PYTHONPATH=. python3 -c 'import tools.vibevoice.vibevoice_llm_oracle'
```

Expected: 21 tests pass and all import/CLI checks exit 0.

- [ ] **Step 4: Commit the VibeVoice move**

```bash
git commit -m "move VibeVoice converters under converter"
```

### Task 5: Move DreamX converter and tests

**Files:**
- Move: `tools/dreamx/convert_dreamx_creator.py` → `tools/converter/dreamx/convert_dreamx_creator.py`
- Move: `tools/dreamx/test_convert_dreamx_creator.py` → `tools/converter/dreamx/test_convert_dreamx_creator.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `tools.converter.dots.convert_dots_tts` writer, types, dimensions, and Q8 quantizer.
- Produces: unchanged DreamX CLI and Python functions under `tools.converter.dreamx.convert_dreamx_creator`.

- [ ] **Step 1: Change test imports/patch targets and verify RED**

Replace `tools.dreamx.convert_dreamx_creator` with `tools.converter.dreamx.convert_dreamx_creator` in imports and `unittest.mock.patch` targets. Run the old test file; expect import failure.

- [ ] **Step 2: Move implementation and test**

Move both files. Add the direct-script repository-root bootstrap, use the package dots import, and update the top-level README command to `tools/converter/dreamx/convert_dreamx_creator.py`.

- [ ] **Step 3: Verify DreamX**

Run:

```bash
PYTHONPATH=. python3 tools/converter/dreamx/test_convert_dreamx_creator.py
python3 tools/converter/dreamx/convert_dreamx_creator.py --help
```

Expected: 21 tests pass and CLI exits 0.

- [ ] **Step 4: Commit DreamX move**

```bash
git commit -m "move DreamX converter under converter"
```

### Task 6: Move Qwen-Drive converter, test, and manifest together

**Files:**
- Move: `tools/qwen_drive/convert_qwen_drive.py` → `tools/converter/qwen_drive/convert_qwen_drive.py`
- Move: `tools/qwen_drive/test_convert_qwen_drive.py` → `tools/converter/qwen_drive/test_convert_qwen_drive.py`
- Move: `tools/qwen_drive/source-tensors.json` → `tools/converter/qwen_drive/source-tensors.json`
- Modify: `tools/qwen_drive/README.md`

**Interfaces:**
- Consumes: dots writer/readers and the adjacent `source-tensors.json` selected with `Path(__file__).with_name(...)`.
- Produces: unchanged `inspect`, `export`, and `verify` subcommands at the new path.

- [ ] **Step 1: Change test module imports and entrypoint expectation; verify RED**

Point imports to `tools.converter.qwen_drive.convert_qwen_drive` and the subprocess CLI assertion to `repo / "tools/converter/qwen_drive/convert_qwen_drive.py"`. Run the old test file; expect import failure.

- [ ] **Step 2: Move all three files**

Move implementation, test, and manifest as one unit. Change the direct-script root bootstrap from `parents[2]` to `parents[3]`; leave all `Path(__file__).with_name("source-tensors.json")` lookups intact. Update every command/import in `tools/qwen_drive/README.md`.

- [ ] **Step 3: Verify Qwen-Drive**

Run:

```bash
PYTHONPATH=. python3 tools/converter/qwen_drive/test_convert_qwen_drive.py
python3 tools/converter/qwen_drive/convert_qwen_drive.py --help
python3 tools/converter/qwen_drive/convert_qwen_drive.py verify \
  /Users/gouzi/Documents/git/rust-model-inference/models/Qwen-Drive-1.0-4B \
  --out-dir /Users/gouzi/Documents/git/rust-model-inference/models/Qwen-Drive-1.0-4B
```

Expected: 13 tests pass, CLI exits 0, and read-only verification accepts the five existing model outputs.

- [ ] **Step 4: Commit Qwen-Drive move**

```bash
git commit -m "move Qwen-Drive converter under converter"
```

### Task 7: Move NeoHorse converter and test

**Files:**
- Move: `tools/neohorse/convert_neohorse.py` → `tools/converter/neohorse/convert_neohorse.py`
- Move: `tools/neohorse/test_convert_neohorse.py` → `tools/converter/neohorse/test_convert_neohorse.py`
- Modify: `tools/neohorse/README.md`

**Interfaces:**
- Consumes: pinned llama.cpp commit `b96806d96061049a5b574269b049bf6241d63d46` only when doing a real export.
- Produces: unchanged source validation and CLI at the new path.

- [ ] **Step 1: Point the test at the new module and verify RED**

Change the import to `tools.converter.neohorse.convert_neohorse`; run the old test and expect import failure.

- [ ] **Step 2: Move files and update README**

Move implementation/test and change the documented command to `tools/converter/neohorse/convert_neohorse.py`. No new abstraction or wrapper is added.

- [ ] **Step 3: Verify NeoHorse**

Run:

```bash
PYTHONPATH=. python3 tools/converter/neohorse/test_convert_neohorse.py
python3 tools/converter/neohorse/convert_neohorse.py --help
```

Expected: one test passes and CLI exits 0. Do not run a 9B BF16 export unless the pinned llama.cpp checkout exists and free space exceeds the expected output plus 5 GiB safety margin.

- [ ] **Step 4: Commit NeoHorse move**

```bash
git commit -m "move NeoHorse converter under converter"
```

### Task 8: Update remaining documentation and remove stale path references

**Files:**
- Modify: `tools/converter/README.md`
- Modify: `README.md`
- Modify: `docs/TODO.md`
- Modify: `docs/usage/dots.md`
- Modify: `docs/develop/MODEL_ORGANIZATION.md`
- Modify: any non-historical source or test surfaced by the stale-reference scan

**Interfaces:**
- Consumes: final converter file layout from Tasks 2–7.
- Produces: current commands/imports that reference only `tools/converter` converter paths.

- [ ] **Step 1: Scan for stale paths**

Run:

```bash
rg -n 'tools/(breeze|dots|dreamx|neohorse|qwen_drive|vibevoice)/(convert_|test_convert_)|tools\.(breeze|dots|dreamx|neohorse|qwen_drive|vibevoice)\.(convert_|test_convert_)' \
  README.md docs tools tests .github
```

Expected: matches identify only paths requiring an update; the design/spec table may retain old paths because it documents the move mapping.

- [ ] **Step 2: Update live commands and architecture references**

Replace live paths with their exact new locations. Rewrite `tools/converter/README.md` as the authoritative directory map and test-command list. Keep historical statements only when they are explicitly framed as historical, not runnable commands.

- [ ] **Step 3: Require zero stale live references**

Run the scan again excluding `docs/superpowers/specs/2026-09-17-converter-consolidation-design.md` and this plan. Expected: no match.

- [ ] **Step 4: Commit documentation updates**

```bash
git commit -m "update converter paths"
```

### Task 9: Prove behavior and byte parity after relocation

**Files:**
- Test: all moved converter test modules
- Read-only input: `/Users/gouzi/Documents/git/rust-model-inference/models`
- Temporary output: one `mktemp -d /tmp/converter-model-check.XXXXXX` directory at a time

**Interfaces:**
- Consumes: pre-move test/CLI logs and `/tmp/converter-consolidation-baseline/sha256.txt`.
- Produces: final test result, controlled-output parity result, and explicit real-model verification matrix.

- [ ] **Step 1: Run the complete moved test set**

Run:

```bash
PYTHONPATH=. python3 -m unittest \
  tools.converter.breeze.test_convert_breeze_plain \
  tools.converter.breeze.test_convert_breeze \
  tools.converter.dots.test_convert_dots_tts \
  tools.converter.dreamx.test_convert_dreamx_creator \
  tools.converter.neohorse.test_convert_neohorse \
  tools.converter.qwen_drive.test_convert_qwen_drive \
  tools.converter.vibevoice.test_convert_vibevoice_asr_original \
  tools.converter.vibevoice.test_convert_vibevoice_asr
```

Expected: 101 tests pass.

- [ ] **Step 2: Recreate controlled outputs and compare exact bytes**

Run the same temporary fixture script from Task 1 against the new modules, then:

```bash
find /tmp/converter-consolidation-after/outputs -type f -name '*.gguf' \
  -exec shasum -a 256 {} + | sort \
  > /tmp/converter-consolidation-after/sha256.txt
diff -u /tmp/converter-consolidation-baseline/sha256.txt \
        /tmp/converter-consolidation-after/sha256.txt
```

Normalize only the temporary directory prefix before comparing manifests. Expected: `diff` exits 0; for retained files, `diff -qr` also exits 0.

- [ ] **Step 3: Perform space-safe real-model checks**

Before every export, run `df -Pk /tmp` and require expected output size plus 5 GiB free. Execute sequentially, hash outputs, then `unlink` only the generated files and `rmdir` their exact temporary directory:

- dots base BF16: expected about 5.2 GiB total; compare with existing base GGUFs when names/configuration correspond.
- Breeze BF16/F32 codec: expected about 7.2 GiB total; compare only with outputs produced by the same converter mode.
- VibeVoice Q8/BF16: expected about 8.8 GiB total; compare with existing same-mode files.
- Qwen-Drive: run `verify` against the five existing outputs without rewriting them.
- DreamX: skip full export while free space is below about 31 GiB (26 GiB output plus safety margin).
- NeoHorse 9B: skip full BF16 export while the pinned llama.cpp checkout or required safety margin is absent.

Record every attempted command, result, SHA-256 pair, and skipped reason. A pre-existing GGUF is not parity evidence unless its converter mode and inputs are confirmed identical.

- [ ] **Step 4: Run final repository checks**

Run:

```bash
git diff --check upstream/main...HEAD
git status --short
rg -n 'tools/(breeze|dots|dreamx|neohorse|qwen_drive|vibevoice)/(convert_|test_convert_)' \
  README.md docs tools tests .github \
  -g '!docs/superpowers/specs/2026-09-17-converter-consolidation-design.md' \
  -g '!docs/superpowers/plans/2026-09-17-converter-consolidation.md'
```

Expected: diff check exits 0; only `.codex/` remains untracked; stale-path scan returns no live reference.

### Task 10: Update Draft PR #73

**Files:**
- Commit: all scoped relocation, import, test, and documentation changes
- Remote: `gouzil:codex/cleanup-converter-duplicates`
- PR: `Liyulingyue/rust-model-inference#73`

**Interfaces:**
- Consumes: verified local branch and explicit verification matrix from Task 9.
- Produces: pushed commits and an updated Draft PR whose title/body/file list describe both duplicate cleanup and consolidation.

- [ ] **Step 1: Confirm local scope**

Run `git status -sb`, `git diff upstream/main...HEAD --name-status`, and `git log --oneline upstream/main..HEAD`. Expected: only converter migration, its docs/spec/plan, and the earlier duplicate cleanup are present; `.codex/` is untracked.

- [ ] **Step 2: Push without force**

```bash
git push origin codex/cleanup-converter-duplicates
```

Expected: origin advances to local HEAD without rewriting earlier commits.

- [ ] **Step 3: Update PR title and Chinese body**

Set the title to `[codex] 归档并清理 converter 实现`. The body must state the directory move, removed duplicates, behavioral boundary, 101-test result, controlled byte parity, each real-model check, and all explicit skips.

- [ ] **Step 4: Read back authoritative PR state**

Run:

```bash
gh pr view 73 --repo Liyulingyue/rust-model-inference \
  --json url,title,state,isDraft,baseRefName,headRefName,headRefOid,mergeable,files,statusCheckRollup
gh pr diff 73 --repo Liyulingyue/rust-model-inference --name-only
```

Expected: Draft/Open, base `main`, head `codex/cleanup-converter-duplicates`, mergeable unless GitHub reports a current conflict, and no `.codex/` file in the remote list.
