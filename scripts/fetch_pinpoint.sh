#!/usr/bin/env bash
# Fetch the exact PinPoint artifacts analysed in the paper and verify them by git object ID.
# PinPoint data are NOT redistributed in this repository; this script retrieves them from the
# public upstream repository at pinned revisions. Check the upstream license before use.
set -euo pipefail

UPSTREAM="https://github.com/pinterest/pinpoint-dataset.git"
CODE_REV="059d6e4ae191d3f0775cbe2e9cf77baa0e714126"       # evaluator + standardized_results/
FIRST_REPLACEMENT="5fc977b40352205137584aaabf0aafab5c9576dc" # first positive-list replacement
OLD_LABEL_BLOB="aca644ff93242fe683bed67b6bacf17cb22241e2"    # earliest metadata release
CUR_LABEL_BLOB="54e2b3c3d572fe342e84e50a0374cecf43f65aaf"    # current licensed release
OLD_MANIFEST_BLOB="7483d669898555ac02abee2558da7995f507d32a"
CUR_MANIFEST_BLOB="7150b4ea9a990aa5f5579f27dd97aedeb0ff6230"

DEST="${1:-data/pinpoint}"
mkdir -p "$DEST"
if [ ! -d "$DEST/repo/.git" ]; then
  git clone --quiet "$UPSTREAM" "$DEST/repo"
fi
git -C "$DEST/repo" fetch --quiet origin
git -C "$DEST/repo" cat-file -e "$FIRST_REPLACEMENT^{commit}"
git -C "$DEST/repo" checkout --quiet "$CODE_REV"

# Extract label files and candidate manifests by blob ID (content-addressed, so they are exact).
# old = pinpoint_metadata.parquet / index_signatures.txt at 3f2cfb2 (earliest inspected release)
# current = pinpoint_licensed.parquet / index_signatures.txt at 059d6e4
git -C "$DEST/repo" cat-file -p "$OLD_LABEL_BLOB"      > "$DEST/labels_old.parquet"
git -C "$DEST/repo" cat-file -p "$CUR_LABEL_BLOB"      > "$DEST/labels_current.parquet"
git -C "$DEST/repo" cat-file -p "$OLD_MANIFEST_BLOB"   > "$DEST/candidates_old.txt"
git -C "$DEST/repo" cat-file -p "$CUR_MANIFEST_BLOB"   > "$DEST/candidates_current.txt"
# labels at the first replacement commit (used only for the split-at-first-replacement check)
git -C "$DEST/repo" show "$FIRST_REPLACEMENT:pinpoint_metadata.parquet" > "$DEST/labels_t0.parquet"
# upstream leaderboard numbers, used to check that our evaluator re-implementation reproduces them
git -C "$DEST/repo" show "$CODE_REV:metrics.csv" > "$DEST/metrics_upstream.csv"

# Cached runs used in the paper (standardized_results/ at CODE_REV).
mkdir -p "$DEST/runs"
for f in \
  bge_vl_mllm_s1_retrieval_results_licensed.json \
  bge_vl_mllm_s1_retrieval_results_licensed_reranked.json \
  bge_vl_mllm_s2_retrieval_results_licensed.json \
  metaclip2_retrieval_results_combined.json \
  metaclip2_retrieval_results_image_only.json \
  metaclip2_retrieval_results_text_only.json \
  metaclip2_slerp_retrieval_results_licensed.json; do
  cp "$DEST/repo/standardized_results/$f" "$DEST/runs/$f"
done

# Verify every extracted file against its git blob ID.
for pair in "labels_old.parquet:$OLD_LABEL_BLOB" "labels_current.parquet:$CUR_LABEL_BLOB" \
            "candidates_old.txt:$OLD_MANIFEST_BLOB" "candidates_current.txt:$CUR_MANIFEST_BLOB"; do
  f="${pair%%:*}"; want="${pair##*:}"
  got=$(git hash-object "$DEST/$f")
  [ "$got" = "$want" ] || { echo "HASH MISMATCH: $f ($got != $want)"; exit 1; }
done
echo "PinPoint artifacts fetched and verified in $DEST"
