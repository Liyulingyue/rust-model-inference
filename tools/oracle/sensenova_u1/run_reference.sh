#!/usr/bin/env bash
# Full reference-generation pipeline for the SenseNova-U1 oracle. RUN ON THE REMOTE.
#
#   0. gguf header dump (tensor inventory, no weights loaded)
#   1. official CLI smoke test  (examples/t2i/inference.py, the ground truth)
#   2. make_reference.py        (captured replay, verified against step 1's PNG)
#   3. collect + tar the artifacts
#
# Expects make_reference.py / gguf_dump.py to sit next to this script.
# Env: MODEL_DIR VENV ORACLE_SRC QUANT OUT TAG
set -uo pipefail

MODEL_DIR="${MODEL_DIR:-/root/models/SenseNova-U1.6-8B-MoT-GGUF}"
VENV="${VENV:-/root/sn_oracle_venv}"
ORACLE_SRC="${ORACLE_SRC:-/tmp/sn_oracle}"
QUANT="${QUANT:-Q8_0}"
OUT="${OUT:-/root/oracle_out}"
TAG="${TAG:-ref512}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GGUF="$MODEL_DIR/SenseNova-U1.6-8B-MoT-$QUANT.gguf"
PROMPT='A studio photograph of a red apple on a white table, soft daylight.'
PY="$VENV/bin/python"
mkdir -p "$OUT"

log() { echo; echo "=== $* @ $(date -u +%H:%M:%SZ) ==="; }

peak_rss() {  # $1=pid -> poll VmHWM, print max MB
  local pid=$1 hwm=0 v
  while kill -0 "$pid" 2>/dev/null; do
    v=$(awk '/VmHWM/{print $2}' "/proc/$pid/status" 2>/dev/null)
    [ -n "${v:-}" ] && [ "$v" -gt "$hwm" ] && hwm=$v
    sleep 2
  done
  echo "$((hwm / 1024))"
}

# ---------------------------------------------------------------- 0. gguf header
log "0. gguf header dump"
[ -f "$OUT/gguf_$QUANT.json" ] || \
  "$PY" "$HERE/gguf_dump.py" "$GGUF" --json "$OUT/gguf_$QUANT.json" --md "$OUT/gguf_$QUANT.md"

# ---------------------------------------------------------------- 1. CLI smoke
log "1. official CLI smoke test (cpu / float32 / sdpa / 512x512 / 4 steps)"
start=$(date +%s)
"$PY" "$ORACLE_SRC/examples/t2i/inference.py" \
  --model_path "$GGUF" --model_format gguf \
  --prompt "$PROMPT" \
  --width 512 --height 512 --num_steps 4 --seed 42 \
  --device cpu --dtype float32 --attn_backend sdpa \
  --output "$OUT/smoke.png" --profile > "$OUT/smoke.log" 2>&1 &
SMOKE_PID=$!
peak_rss "$SMOKE_PID" > /tmp/smoke_mb.txt & SMOKE_MON=$!
wait "$SMOKE_PID"; smoke_rc=$?
wait "$SMOKE_MON" 2>/dev/null; SMOKE_MB=$(cat /tmp/smoke_mb.txt 2>/dev/null || echo 0)
smoke_s=$(( $(date +%s) - start ))
echo "smoke rc=$smoke_rc wall=${smoke_s}s peakRSS=${SMOKE_MB}MB"
tail -25 "$OUT/smoke.log"
if [ $smoke_rc -ne 0 ]; then
  echo "!! CLI smoke failed -- see $OUT/smoke.log. Stopping so the reference run is not attempted blind."
  exit 1
fi

# ---------------------------------------------------------------- 2. reference
log "2. make_reference.py (captured replay, verified against smoke.png)"
start=$(date +%s)
"$PY" "$HERE/make_reference.py" \
  --model_path "$GGUF" --model_format gguf \
  --device cpu --dtype float32 \
  --out_dir "$OUT/$TAG" \
  --verify-png "$OUT/smoke.png" > "$OUT/ref.log" 2>&1 &
REF_PID=$!
peak_rss "$REF_PID" > /tmp/ref_mb.txt & REF_MON=$!
wait "$REF_PID"; ref_rc=$?
wait "$REF_MON" 2>/dev/null; REF_MB=$(cat /tmp/ref_mb.txt 2>/dev/null || echo 0)
ref_s=$(( $(date +%s) - start ))
echo "ref rc=$ref_rc wall=${ref_s}s peakRSS=${REF_MB}MB"
tail -30 "$OUT/ref.log"
[ $ref_rc -ne 0 ] && { echo "!! reference run failed"; exit 1; }

# ---------------------------------------------------------------- 3. collect
log "3. collect"
{
  echo "smoke_wall_seconds=$smoke_s"
  echo "smoke_peak_rss_mb=$SMOKE_MB"
  echo "ref_wall_seconds=$ref_s"
  echo "ref_peak_rss_mb=$REF_MB"
  echo "quant=$QUANT"
  echo "gguf=$(stat -c%s "$GGUF")"
  git -C "$ORACLE_SRC" rev-parse HEAD
} > "$OUT/$TAG/timings.txt"

du -sh "$OUT/$TAG"
ls -la "$OUT/$TAG" | head -40
cd "$OUT" && tar -czf "/root/$TAG.tar.gz" "$TAG" smoke.png smoke.log ref.log "gguf_$QUANT.json" "gguf_$QUANT.md" timings.txt 2>/dev/null
ls -l "/root/$TAG.tar.gz"
echo "=== PIPELINE OK ==="
