#!/bin/bash
# Single-process, hash-verified download of one SenseNova-U1.6 quant.
#
# The Radeon Cloud box gets wiped on every Template restart, so this is meant to
# be re-runnable: it downloads to `<file>.tmp`, checks the byte count AND the
# pinned sha256, and only then renames. Never run two of these at once for the
# same file -- `wget -c -O` on a shared path interleaves and produces a file that
# is larger than the original (observed: 22,692,593,723 B vs 20,695,981,472 B).
#
#   bash download_quant.sh Q8_0
#   bash download_quant.sh Q6_K
#   bash download_quant.sh Q4_0
set -euo pipefail

QUANT="${1:?usage: download_quant.sh Q8_0|Q6_K|Q4_0}"
DIR="${MODEL_DIR:-/root/models/SenseNova-U1.6-8B-MoT-GGUF}"
BASE="https://www.modelscope.cn/models/SenseNova/SenseNova-U1.6-8B-MoT-GGUF/resolve/master"
FILE="SenseNova-U1.6-8B-MoT-${QUANT}.gguf"

case "$QUANT" in
  Q8_0) WANT_SIZE=20695981472
        WANT_SHA=097ca40a942211c89fe0e945add3d159839e904a16feb5f9d0d14ba8e6644806 ;;
  Q6_K) WANT_SIZE=16620178848
        WANT_SHA=e538414f42201294471a5fb8ec58e54af57e260cac297daa529829be294d1b9b ;;
  Q4_0) WANT_SIZE=12281421216
        WANT_SHA=135ccdde87ebf92f76a6f7a17d604f91df6ba15718f1dd847e0dbb766e8ad28b ;;
  *) echo "unknown quant $QUANT" >&2; exit 2 ;;
esac

mkdir -p "$DIR"
cd "$DIR"

if [ -s "$FILE" ]; then
  GOT=$(stat -c%s "$FILE")
  if [ "$GOT" = "$WANT_SIZE" ]; then
    echo "already present with the expected size: $FILE ($GOT B)"
    exit 0
  fi
  echo "present but wrong size ($GOT != $WANT_SIZE), re-downloading"
  rm -f "$FILE"
fi

for f in config.json tokenizer_config.json vocab.json merges.txt \
         special_tokens_map.json added_tokens.json README.md manifest.json; do
  [ -s "$f" ] || wget -q -O "$f" "$BASE/$f"
done

echo "START $FILE $(date -u)"
rm -f "$FILE.tmp"
wget -q --show-progress=off -O "$FILE.tmp" "$BASE/$FILE"

GOT=$(stat -c%s "$FILE.tmp")
echo "SIZE $GOT (want $WANT_SIZE)"
if [ "$GOT" != "$WANT_SIZE" ]; then
  echo "SIZE MISMATCH" >&2
  exit 1
fi

SHA=$(sha256sum "$FILE.tmp" | cut -d' ' -f1)
echo "SHA $SHA"
if [ "$SHA" != "$WANT_SHA" ]; then
  echo "SHA MISMATCH (want $WANT_SHA)" >&2
  exit 1
fi

mv "$FILE.tmp" "$FILE"
echo "VERIFIED $FILE $(date -u)"
