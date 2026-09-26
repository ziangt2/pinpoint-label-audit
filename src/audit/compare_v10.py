"""Compare a reproduction run with the earlier (v10) frozen outputs.

Point estimates, counts and component numbers are expected to match exactly (up to float
round-off). Bootstrap interval endpoints differ slightly because the original draw order of the
v10 run is not recoverable; the public implementation fixes it (see bootstrap.py).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(p, keys):
    return {tuple(r[k] for k in keys): r for r in csv.DictReader(open(p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="results/reproduced")
    ap.add_argument("--v10", default="results/v10_frozen")
    a = ap.parse_args()
    new, old = Path(a.new), Path(a.v10)
    rep = {}
    spec = [("summary.csv", ("domain", "stratum", "method", "metric"),
             ["old", "new", "delta", "groups", "queries", "components"], ["ci_low", "ci_high"]),
            ("model_contrasts.csv", ("domain", "stratum", "metric", "a", "b"),
             ["old_difference", "new_difference", "difference_change"],
             ["old_ci_low", "old_ci_high", "new_ci_low", "new_ci_high", "change_ci_low", "change_ci_high"]),
            ("identified_bounds.csv", ("domain", "stratum", "method", "metric"),
             ["old_low", "old_high", "new_low", "new_high", "delta_low", "delta_high"], [])]
    for fn, keys, exact, ci in spec:
        A, F = load(new / fn, keys), load(old / fn, keys)
        assert set(A) == set(F), fn
        r = {}
        for c in exact + ci:
            r[c] = max(abs(float(A[k][c]) - float(F[k][c])) for k in F)
        if fn == "identified_bounds.csv":
            ap_p = [k for k in F if k[3] in ("AP10", "P10")]
            r["max_abs_diff_AP10_P10_only"] = max(abs(float(A[k][c]) - float(F[k][c])) for k in ap_p for c in exact)
        rep[fn] = r
    fs = json.loads((new / "frame_summary.json").read_text())
    ref = {**json.loads((old / "frame_selection.json").read_text()), **json.loads((old / "frame344_summary.json").read_text())}
    rep["frame_counts_identical"] = all(fs[k] == v for k, v in ref.items() if k in fs)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
