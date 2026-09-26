"""The benchmark's own sensitivity metric under fixed outputs (paper Sec. IV-E).

(1) Replication: re-implements the public evaluator (evaluate.py at 059d6e4) and checks that it
    reproduces the upstream metrics.csv (mAP@10, precision@10, ling_sens_range) for the seven runs.
    The evaluator groups rows by (query_image_signature, query_image_signature2) and averages the
    within-group P@10 range over groups with more than one row.
(2) Fixed-output comparison: on all rows whose instruction and reference-image IDs are unchanged
    across releases, the evaluator's grouping and outputs are held fixed and only the positive
    lists (projected onto the current candidate domain) are switched between old and current.
    Paired intervals: 5,000 bootstrap draws over components of evaluator groups that share any
    reference image (seed 20260920).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from .data import METHOD_ID, RUN_FILES, default_paths, load_candidates, load_labels, load_runs, normalize_query_id
from .metrics import ap_at_k, p_at_k

SEED, DRAWS = 20260920, 5000
RUNS = list(RUN_FILES)


def evaluator_key(row):
    """Exactly as evaluate.py: raw signatures, (img1, img2) if img2 else (img1, None)."""
    img1, img2 = row.get("query_image_signature"), row.get("query_image_signature2")
    return (img1, img2) if img2 else (img1, None)


def replicate(raw_rows, cur, runs):
    out = {}
    for r in RUNS:
        groups, aps, ps = defaultdict(list), [], []
        for row in raw_rows:
            nq = normalize_query_id(row["query_id"])
            if nq not in runs[r]:
                continue
            ret, rel = runs[r][nq], cur[row["query_id"]]["positive_candidates"]
            aps.append(ap_at_k(ret, rel, 10)); pp = p_at_k(ret, rel, 10); ps.append(pp)
            if row.get("query_image_signature"):
                groups[evaluator_key(row)].append(pp)
        rs = [max(v) - min(v) for v in groups.values() if len(v) > 1]
        out[r] = {"mAP@10": float(np.mean(aps)), "precision@10": float(np.mean(ps)), "ling_sens_range": float(np.mean(rs))}
    return out


def components(group_keys):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k in group_keys:
        find(("g", k))
        for s in k:
            if s and s != "None":
                parent[find(("g", k))] = find(("s", s))
    comp = defaultdict(list)
    for k in group_keys:
        comp[find(("g", k))].append(k)
    return sorted((sorted(v, key=str) for v in comp.values()), key=lambda c: str(c[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics-csv", default=None, help="upstream metrics.csv at 059d6e4 (optional; else read from data/repo)")
    a = ap.parse_args()
    p = default_paths(a.data)
    old, cur, runs = load_labels(p["labels_old"]), load_labels(p["labels_current"]), load_runs(p["runs"])
    C = load_candidates(p["candidates_current"])
    raw_rows = [{**{k: v for k, v in r.items() if k in ("query_id", "query_image_signature", "query_image_signature2")}} for r in cur.values()]

    # ---- (1) replication of upstream metrics.csv
    rep = replicate(raw_rows, cur, runs)
    mpath = Path(a.metrics_csv) if a.metrics_csv else Path(a.data) / "metrics_upstream.csv"
    replication = {"available": mpath.exists()}
    if mpath.exists():
        up = {r["model"]: r for r in csv.DictReader(open(mpath))}
        diffs = {}
        for r in RUNS:
            u = up[METHOD_ID[r]]
            diffs[r] = {k: abs(round(rep[r][k], 4) - float(u[k])) for k in ("mAP@10", "precision@10", "ling_sens_range")}
        replication.update({"max_abs_diff_at_4dp": max(v for d in diffs.values() for v in d.values()),
                            "reproduced": rep, "cells_checked": 3 * len(RUNS)})

    # ---- (2) fixed-output comparison on unchanged rows, evaluator grouping
    qs = sorted(q for q in cur if old[q]["instruction"] == cur[q]["instruction"] and old[q]["ref"] == cur[q]["ref"])
    by_key = defaultdict(list)
    for q in qs:
        by_key[evaluator_key(cur[q])].append(q)
    G = {k: v for k, v in by_key.items() if len(v) > 1}
    comps = components(list(G))
    rng_ = np.random.default_rng(SEED)
    idx = rng_.integers(0, len(comps), size=(DRAWS, len(comps)))
    gcount = np.array([len(c) for c in comps], float)
    res = {"rows": len(qs), "groups": len(G), "components": len(comps), "runs": {}}
    per = {}
    for r in RUNS:
        for m, f in (("P10", p_at_k), ("AP10", ap_at_k)):
            for v, lab in (("old", old), ("new", cur)):
                s = {q: f(runs[r][normalize_query_id(q)], [x for x in lab[q]["positive_candidates"] if x in C], 10) for q in qs}
                per[(r, m, v)] = np.array([sum(max(s[q] for q in G[k]) - min(s[q] for q in G[k]) for k in c) for c in comps])

    def stat(x):
        pt = x.sum() / gcount.sum()
        lo, hi = np.percentile(x[idx].sum(1) / gcount[idx].sum(1), [2.5, 97.5])
        return [float(pt), float(lo), float(hi)]

    for r in RUNS:
        res["runs"][r] = {}
        for m in ("P10", "AP10"):
            res["runs"][r][m] = {"old": stat(per[(r, m, "old")]), "new": stat(per[(r, m, "new")]),
                                 "change": stat(per[(r, m, "new")] - per[(r, m, "old")])}
    order = {}
    for m in ("P10", "AP10"):
        for v in ("old", "new"):
            order[f"{m}_{v}"] = sorted(RUNS, key=lambda r: -res["runs"][r][m][v][0])
    res["sensitivity_order"] = order

    def kendall(a, b):
        pos = {x: i for i, x in enumerate(b)}
        conc = sum(1 if pos[x] < pos[y] else -1 for x, y in combinations(a, 2))
        return conc / (len(a) * (len(a) - 1) / 2)

    res["kendall_tau_old_vs_new"] = {m: kendall(order[f"{m}_old"], order[f"{m}_new"]) for m in ("P10", "AP10")}
    # pairwise sensitivity differences whose sign flips, with paired intervals of the new difference
    flips = []
    for m in ("P10", "AP10"):
        for x, y in combinations(RUNS, 2):
            do, dn = per[(x, m, "old")] - per[(y, m, "old")], per[(x, m, "new")] - per[(y, m, "new")]
            so, sn = stat(do), stat(dn)
            if np.sign(so[0]) != np.sign(sn[0]):
                flips.append({"metric": m, "a": x, "b": y, "old_diff": so, "new_diff": sn,
                              "both_intervals_exclude_zero": bool((so[1] > 0 or so[2] < 0) and (sn[1] > 0 or sn[2] < 0))})
    res["sign_flips"] = flips
    res["replication"] = replication
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "evaluator_sensitivity.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("rows", "groups", "components", "kendall_tau_old_vs_new", "sensitivity_order")}, indent=1))
    for r in RUNS:
        x = res["runs"][r]["P10"]; print(f"{r:22s} P10-range {x['old'][0]:.4f} -> {x['new'][0]:.4f}  change {x['change'][0]:+.4f} [{x['change'][1]:+.4f},{x['change'][2]:+.4f}]")
    print("flips", len(flips), sum(f["both_intervals_exclude_zero"] for f in flips))
    for f in flips:
        print(f["metric"], f["a"], "vs", f["b"], [round(v, 4) for v in f["old_diff"]], [round(v, 4) for v in f["new_diff"]], f["both_intervals_exclude_zero"])
    print("replication", {k: v for k, v in replication.items() if k != "reproduced"})


if __name__ == "__main__":
    main()
