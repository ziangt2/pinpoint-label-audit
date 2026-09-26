#!/usr/bin/env bash
# End-to-end reproduction from pinned public inputs.
#   bash scripts/reproduce_all.sh [OUT] [DATA] [DL19]
# 1. fetch inputs (verified by git object ID / sha256)       -> DATA, DL19
# 2. rebuild every result file                                -> OUT
# 3. compare OUT with the committed results/reproduced        (exact for counts/points; bitwise JSON)
# 4. redraw every figure from OUT                             -> OUT/figures (+ pixel comparison)
# 5. check every table cell / quoted number in paper/main.tex against OUT
# 6. compile the paper with OUT/figures if pdflatex + IEEEtran are available -> OUT/paper/main.pdf
# Environment, commands and outputs are written to OUT/REPRODUCTION_LOG.txt.
# Set SKIP_FETCH=1 to reuse already fetched inputs in DATA and DL19.
set -euo pipefail
OUT="${1:-results/run}"
DATA="${2:-data/pinpoint}"
DL19="${3:-data/dl19}"
PY="${PYTHON:-python}"
mkdir -p "$OUT"
LOG="$OUT/REPRODUCTION_LOG.txt"
exec > >(tee "$LOG") 2>&1

run() { echo "+ $*"; "$@"; }
echo "# reproduce_all.sh  $(date -u +%Y-%m-%dT%H:%MZ)"
echo "# repository commit: $(git rev-parse HEAD 2>/dev/null || echo 'n/a (not a git checkout)')"
echo "# working tree clean: $(git diff --quiet 2>/dev/null && echo yes || echo NO)"
echo "# $($PY --version 2>&1); $(uname -srm)"
$PY - <<'PY'
import importlib
for m in ("numpy", "pandas", "pyarrow", "matplotlib"):
    try:
        print(f"#   {m} {importlib.import_module(m).__version__}")
    except Exception as e:
        print(f"#   {m} not available ({type(e).__name__})")
PY
echo "# pdflatex: $(pdflatex --version 2>/dev/null | head -1 || echo 'not available')"

if [ "${SKIP_FETCH:-0}" != "1" ]; then
  run bash scripts/fetch_pinpoint.sh "$DATA"
  run bash scripts/fetch_dl19.sh "$DL19"
fi

run $PY -m src.audit.frame       --data "$DATA" --out "$OUT"
run $PY -m src.audit.score       --data "$DATA" --frame "$OUT/frame.json" --out "$OUT"
run $PY -m src.audit.bootstrap   --data "$DATA" --in "$OUT" --out "$OUT"
run $PY -m src.audit.diagnostics --data "$DATA" --in "$OUT" > /dev/null
run $PY -m src.audit.case        --data "$DATA" --in "$OUT" > /dev/null
run $PY -m src.audit.robustness  --data "$DATA" --in "$OUT" --out "$OUT" > /dev/null
run $PY -m src.audit.evaluator_sensitivity --data "$DATA" --out "$OUT" > /dev/null
run $PY -m src.audit.report      --scores "$OUT/query_scores.jsonl" --out "$OUT/four_cell_S1_vs_S1_reranked.json" > /dev/null
run $PY -m src.dl19.check        --parry "$DL19/parry" --qrels "$DL19/2019qrels-pass.txt" --out "$OUT"

echo "## compare with committed results/reproduced"
$PY - "$OUT" <<'PY'
import csv, json, sys
from pathlib import Path
new, ref = Path(sys.argv[1]), Path("results/reproduced")
for fn in ["summary.csv", "model_contrasts.csv", "identified_bounds.csv", "dl19_ndcg10.csv", "dl19_pairs.csv", "robustness.csv"]:
    A, B = list(csv.DictReader(open(new / fn))), list(csv.DictReader(open(ref / fn)))
    assert len(A) == len(B), fn
    worst = 0.0
    for a, b in zip(A, B):
        for k in a:
            try: worst = max(worst, abs(float(a[k]) - float(b[k])))
            except ValueError: assert a[k] == b[k], (fn, k, a[k], b[k])
    print(f"{fn}: max |diff| vs committed = {worst:.1e}")
    assert worst < 1e-9, fn
for fn in ["case.json", "diagnostics.json", "frame_summary.json", "robustness.json", "evaluator_sensitivity.json",
           "four_cell_S1_vs_S1_reranked.json"]:
    a, b = json.loads((new / fn).read_text()), json.loads((ref / fn).read_text())
    worst, bad = 0.0, []
    def walk(x, y, path=""):
        global worst
        if isinstance(x, dict) and isinstance(y, dict):
            if set(x) != set(y): bad.append(f"{path}: keys differ"); return
            for k in x: walk(x[k], y[k], f"{path}/{k}")
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y): bad.append(f"{path}: length {len(x)} vs {len(y)}"); return
            for i, (u, v) in enumerate(zip(x, y)): walk(u, v, f"{path}[{i}]")
        elif isinstance(x, float) or isinstance(y, float):
            d = abs(float(x) - float(y)); worst = max(worst, d)
            if d >= 1e-9: bad.append(f"{path}: {x} vs {y}")
        elif x != y:
            bad.append(f"{path}: {x!r} vs {y!r}")
    walk(a, b)
    for m in bad[:20]: print("  MISMATCH", m)
    assert not bad, fn
    print(f"{fn}: counts/strings identical, max float |diff| = {worst:.1e}")
PY
run $PY -m src.audit.compare_v10 --new "$OUT" > "$OUT/comparison_with_v10.json"
echo "comparison with v10 written to $OUT/comparison_with_v10.json"
run $PY -m tests.test_audit

echo "## figures from $OUT"
run $PY paper/code/make_figures.py --results "$OUT" --figdir "$OUT/figures"
$PY - "$OUT" <<'PY'
import sys
from pathlib import Path
import matplotlib.image as mpimg, numpy as np
new, ref = Path(sys.argv[1]) / "figures", Path("paper/figures")
for p in sorted(ref.glob("*.png")):
    a, b = mpimg.imread(new / p.name), mpimg.imread(p)
    same = a.shape == b.shape and np.array_equal(a, b)
    print(f"{p.name}: {'pixel-identical to paper/figures' if same else 'DIFFERS from paper/figures'}")
PY

echo "## numbers in paper/main.tex vs $OUT"
run $PY paper/code/verify_numbers.py --results "$OUT" --tex paper/main.tex

echo "## compile paper with $OUT/figures"
rm -rf "$OUT/paper"; mkdir -p "$OUT/paper"
cp paper/main.tex paper/references.bib "$OUT/paper/"; cp -r "$OUT/figures" "$OUT/paper/figures"
if [ -n "${IEEETRAN_DIR:-}" ]; then cp "$IEEETRAN_DIR"/IEEEtran.cls "$IEEETRAN_DIR"/IEEEtran.bst "$OUT/paper/"; fi
if command -v pdflatex >/dev/null && (kpsewhich IEEEtran.cls >/dev/null || [ -f "$OUT/paper/IEEEtran.cls" ]); then
  (cd "$OUT/paper" && pdflatex -interaction=nonstopmode main.tex >/dev/null; bibtex main >/dev/null;
   pdflatex -interaction=nonstopmode main.tex >/dev/null; pdflatex -interaction=nonstopmode main.tex >/dev/null)
  echo "pages: $(pdfinfo "$OUT/paper/main.pdf" 2>/dev/null | awk '/Pages/{print $2}')"
  grep -E "^!|Overfull|undefined" "$OUT/paper/main.log" || echo "LaTeX: no errors, overfull boxes or undefined references"
else
  echo "pdflatex or IEEEtran.cls not available; skipped (set IEEETRAN_DIR to a directory with IEEEtran.cls/.bst)"
fi
echo "# done"
