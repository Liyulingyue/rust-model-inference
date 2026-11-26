#!/usr/bin/env bash
# One-shot re-bootstrap of the SenseNova-U1 Python oracle on the Radeon Cloud box.
#
# The Radeon Cloud Template wipes /root and /opt on restart, so this script is
# idempotent and safe to re-run: it only redoes the steps whose outputs are
# missing. Run it ON THE REMOTE (over ssh), not locally.
#
#   scp -P <port> -i ~/.ssh/gouzi-remote tools/oracle/sensenova_u1/bootstrap_remote.sh \
#       root@36.150.116.206:/root/
#   ssh -p <port> -i ~/.ssh/gouzi-remote root@36.150.116.206 'bash /root/bootstrap_remote.sh'
#
# Env overrides: MODEL_DIR, VENV, ORACLE_SRC, ORACLE_COMMIT, GGUF_QUANT
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/root/models/SenseNova-U1.6-8B-MoT-GGUF}"
VENV="${VENV:-/root/sn_oracle_venv}"
ORACLE_SRC="${ORACLE_SRC:-/tmp/sn_oracle}"
ORACLE_REPO="${ORACLE_REPO:-https://github.com/OpenSenseNova/SenseNova-U1.git}"
ORACLE_COMMIT="${ORACLE_COMMIT:-d2d9bd1}"
GGUF_QUANT="${GGUF_QUANT:-Q8_0}"
BASE_PY="${BASE_PY:-/opt/venv/bin/python3}"
BASE_SP="${BASE_SP:-/opt/venv/lib/python3.12/site-packages}"
MS_BASE="https://www.modelscope.cn/models/SenseNova/SenseNova-U1.6-8B-MoT-GGUF/resolve/master"

log() { echo "=== $* ==="; }

# ---------------------------------------------------------------- 1. oracle src
# `git clone https://github.com/...` fails on this box: the HTTPS proxy answers
# the CONNECT tunnel with 503. `codeload.github.com` tarballs go through, so
# fetch the pinned commit that way and record the commit in a stamp file
# instead of relying on a .git directory.
if [ ! -f "$ORACLE_SRC/pyproject.toml" ]; then
  log "fetching oracle $ORACLE_COMMIT -> $ORACLE_SRC"
  rm -rf "$ORACLE_SRC"
  mkdir -p "$ORACLE_SRC"
  if ! curl -fsSL -o /tmp/oracle.tar.gz \
      "https://codeload.github.com/OpenSenseNova/SenseNova-U1/tar.gz/$ORACLE_COMMIT"; then
    log "pinned tarball failed, falling back to refs/heads/main"
    curl -fsSL -o /tmp/oracle.tar.gz \
      "https://codeload.github.com/OpenSenseNova/SenseNova-U1/tar.gz/refs/heads/main"
  fi
  tar -xzf /tmp/oracle.tar.gz -C "$ORACLE_SRC" --strip-components=1
  echo "$ORACLE_COMMIT" > "$ORACLE_SRC/.oracle-commit"
fi
log "oracle commit: $(cat "$ORACLE_SRC/.oracle-commit" 2>/dev/null || echo unknown) ($(ls "$ORACLE_SRC" | wc -l) entries)"

# ---------------------------------------------------------------- 2. weights
mkdir -p "$MODEL_DIR"
for f in config.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json \
         added_tokens.json README.md; do
  if [ ! -s "$MODEL_DIR/$f" ]; then
    log "fetch $f"
    wget -q -c -O "$MODEL_DIR/$f" "$MS_BASE/$f"
  fi
done

GGUF="$MODEL_DIR/SenseNova-U1.6-8B-MoT-$GGUF_QUANT.gguf"
if [ ! -s "$GGUF" ]; then
  log "downloading $GGUF_QUANT (~20.7 GB for Q8_0)"
  wget -q -c -O "$GGUF.part" "$MS_BASE/SenseNova-U1.6-8B-MoT-$GGUF_QUANT.gguf"
  mv "$GGUF.part" "$GGUF"
fi
ls -l "$GGUF"

# ---------------------------------------------------------------- 3. venv
# NOTE: `python -m venv --system-site-packages` from /opt/venv does NOT inherit
# torch (base_prefix is /usr, not /opt/venv). Bridge it with a .pth instead, so
# we reuse the ROCm torch build and never let pip replace it with a CPU wheel.
if [ ! -x "$VENV/bin/python" ]; then
  log "creating $VENV"
  "$BASE_PY" -m venv --system-site-packages "$VENV"
fi
SP="$VENV/lib/python3.12/site-packages"
echo "$BASE_SP" > "$SP/_opt_venv_bridge.pth"

"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "hip", torch.version.hip,
      "cuda_available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
assert torch.version.hip, "torch is not a ROCm build -- aborting"
PY

# ---------------------------------------------------------------- 4. deps
# --no-deps everywhere: the oracle pins torch==2.8.0 / torchvision==0.23.0 /
# pillow==12.0.0, and letting pip honour those would rip out the ROCm torch.
log "pip install oracle (editable, --no-deps)"
"$VENV/bin/pip" install -q -e "$ORACLE_SRC" --no-deps
log "pip install extras (--no-deps)"
"$VENV/bin/pip" install -q --no-deps gguf diffusers sentencepiece httpx tqdm packaging

"$VENV/bin/python" - <<'PY'
import torch, sensenova_u1, gguf, diffusers, sentencepiece
assert torch.version.hip, "torch got replaced by a CPU build!"
from diffusers.quantizers.gguf.utils import GGUFParameter, SUPPORTED_GGUF_QUANT_TYPES
from diffusers import GGUFQuantizationConfig
from diffusers.quantizers.gguf import GGUFQuantizer
print("oracle", sensenova_u1.__file__)
print("torch", torch.__version__, "diffusers", diffusers.__version__)
print("GGUF API OK")
PY

log "READY"
echo "smoke:"
echo "  cd $ORACLE_SRC && $VENV/bin/python examples/t2i/inference.py \\"
echo "    --model_path $GGUF --model_format gguf \\"
echo "    --prompt 'A studio photograph of a red apple on a white table, soft daylight.' \\"
echo "    --width 512 --height 512 --num_steps 4 --seed 42 \\"
echo "    --device cpu --dtype float32 --attn_backend sdpa \\"
echo "    --output /root/oracle_out/smoke.png --profile"
