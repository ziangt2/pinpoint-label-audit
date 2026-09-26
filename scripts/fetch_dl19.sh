#!/usr/bin/env bash
# Fetch the DL19 assets for the portability check (Sec. IV-D). Nothing is redistributed here.
set -euo pipefail
DEST="${1:-data/dl19}"; mkdir -p "$DEST"
PARRY_REV="12969fb4f189bd87fcfe3d4953d3ee0da110c379"   # Parry-Parry/sigir25-annotation
QRELS_SHA256="36e3687a5332f3cbdda17f8f3ed9ac6b1bf89c50dd54acebc6cbf8390f261686"  # 2019qrels-pass.txt
[ -d "$DEST/parry/.git" ] || git clone --quiet https://github.com/Parry-Parry/sigir25-annotation.git "$DEST/parry"
git -C "$DEST/parry" checkout --quiet "$PARRY_REV"
[ -d "$DEST/tilde/.git" ] || git clone --quiet --depth 1 https://github.com/ielab/TILDE.git "$DEST/tilde"
cp "$DEST/tilde/data/qrels/2019qrels-pass.txt" "$DEST/2019qrels-pass.txt"
echo "$QRELS_SHA256  $DEST/2019qrels-pass.txt" | sha256sum -c -
echo "DL19 assets ready in $DEST"
