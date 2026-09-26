"""Four-cell accounting report and label-free output diagnostics (analysis tool, Sec. III-A).

f_ij = f(R_i, Y_j, C). f_11 - f_00 = O_0 + L_0 + I with
O_0 = f_10 - f_00 (output change under old labels), L_0 = f_01 - f_00 (label change for the
original output), I = f_11 - f_10 - f_01 + f_00 (interaction). This is an accounting identity,
not a unique causal attribution; both substitution orders are reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import validate_ranking

K = 10


def four_cell(f00, f10, f01, f11):
    O0, L0, I = f10 - f00, f01 - f00, f11 - f10 - f01 + f00
    O1, L1 = f11 - f01, f11 - f10
    return {
        "cells": {"f00": f00, "f10": f10, "f01": f01, "f11": f11},
        "total": f11 - f00,
        "output_first": {"output_under_old_labels": O0, "label_under_new_output": L1},
        "label_first": {"label_under_old_output": L0, "output_under_new_labels": O1},
        "interaction": I,
        "semantic_robustness_claim": None,
    }


def output_diagnostics(r0, r1, k=K):
    """Ordered top-k mismatch, top-k set non-overlap, 1 - mean prefix overlap (depths 1..k)."""
    validate_ranking(r0, k)
    validate_ranking(r1, k)
    a, b = r0[:k], r1[:k]
    ordered_mismatch = float(a != b)
    set_nonoverlap = 1 - len(set(a) & set(b)) / k
    prefix = sum(len(set(a[:d]) & set(b[:d])) / d for d in range(1, k + 1)) / k
    return {"ordered_mismatch": ordered_mismatch, "set_nonoverlap": set_nonoverlap,
            "prefix_disagreement": 1 - prefix}


def main():
    ap = argparse.ArgumentParser(description="Four-cell report for run A (R0) vs run B (R1).")
    ap.add_argument("--scores", required=True, help="query_scores.jsonl from score.py")
    ap.add_argument("--r0", default="S1")
    ap.add_argument("--r1", default="S1 reranked")
    ap.add_argument("--metric", default="AP10")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.scores) if '"current_fixed"' in l]
    by = {(r["run"], r["query"]): r for r in rows}
    qs = sorted({r["query"] for r in rows})
    mean = lambda run, ver: sum(by[(run, q)][f"{a.metric}_{ver}"] for q in qs) / len(qs)
    rep = four_cell(mean(a.r0, "old"), mean(a.r1, "old"), mean(a.r0, "new"), mean(a.r1, "new"))
    diags = [output_diagnostics(by[(a.r0, q)]["top10"], by[(a.r1, q)]["top10"]) for q in qs]
    rep["output_diagnostics_mean"] = {k: sum(d[k] for d in diags) / len(diags) for k in diags[0]}
    rep["queries"] = len(qs)
    rep["runs"] = {"R0": a.r0, "R1": a.r1}
    rep["across_version_output_change"] = 0.0  # the same cached ranking is scored under both label versions
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
