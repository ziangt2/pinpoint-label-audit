"""TREC DL 2019 portability check (paper Sec. IV-D).

Fixed runs: the six `maik-froebe-*-run.txt` DL19 runs of Parry et al. (SIGIR 2025) at revision
12969fb4f189bd87fcfe3d4953d3ee0da110c379, selected as one published run family before metrics
were inspected. Official condition: DL19 passage qrels (TILDE mirror, 43 topics). Overlay
condition: for each main-round annotator of a topic, official qrels with that annotator's
judgments overwriting the judged pairs; the two overlay nDCG@10 values are averaged per topic.
Eligibility: exactly two main-round annotators and >= 10 results in all six runs.
nDCG@10 uses linear gain and the ideal ranking of the same (full) qrels condition.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS = ["colbert", "mono-t5-3b", "mono-t5-base", "rank-zephyr", "sparse-cross-encoder", "splade"]
ANNOTATORS = ["andrew-parry", "eugene-yang", "ferdinand-schlatt", "froebe", "guglielmo-faggioli",
              "harry-scells", "saber-zerhoudi", "sean-macavaney"]
SEED, DRAWS = 20260920, 5000


def read_qrels(path):
    q = defaultdict(dict)
    for line in open(path):
        p = line.split()
        if len(p) >= 4:
            q[p[0]][p[2]] = int(p[3])
    return q


def read_run(path):
    """Keep the file's rank order (not possibly tied scores)."""
    r = defaultdict(list)
    for line in open(path):
        p = line.split()
        r[p[0]].append((int(p[3]), p[2]))
    return {t: [d for _, d in sorted(v)] for t, v in r.items()}


def ndcg10(ranking, qrels):
    dcg = sum(qrels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranking[:10]))
    ideal = sorted(qrels.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(parry_dir, official_qrels):
    parry_dir = Path(parry_dir)
    official = read_qrels(official_qrels)
    ann = {a: read_qrels(parry_dir / "judgments/main/qrels" / f"{a}-qrels.txt") for a in ANNOTATORS}
    runs = {r: read_run(parry_dir / "runs/trec-dl-2019" / f"maik-froebe-{r}-run.txt") for r in RUNS}
    topics_by_n = defaultdict(list)
    for t in official:
        n = sum(t in ann[a] for a in ANNOTATORS)
        topics_by_n[n].append(t)
    two = sorted(topics_by_n[2])
    eligible = [t for t in two if all(len(runs[r].get(t, [])) >= 10 for r in RUNS)]
    excluded = {"not_two_annotators": {str(n): sorted(v) for n, v in topics_by_n.items() if n != 2},
                "short_runs": [t for t in two if t not in eligible]}
    per_topic = defaultdict(dict)
    for t in eligible:
        overlays = []
        for a in ANNOTATORS:
            if t in ann[a]:
                q = dict(official[t]); q.update(ann[a][t]); overlays.append(q)
        for r in RUNS:
            rk = runs[r][t]
            per_topic[r][t] = (ndcg10(rk, official[t]), float(np.mean([ndcg10(rk, q) for q in overlays])))
    return eligible, excluded, per_topic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parry", required=True, help="checkout of Parry-Parry/sigir25-annotation @ 12969fb")
    ap.add_argument("--qrels", required=True, help="2019qrels-pass.txt (TILDE mirror)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    eligible, excluded, pt = evaluate(a.parry, a.qrels)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(eligible), size=(DRAWS, len(eligible)))
    rows = []
    for r in RUNS:
        o = np.array([pt[r][t][0] for t in eligible]); n = np.array([pt[r][t][1] for t in eligible])
        lo, hi = np.percentile((n - o)[idx].mean(1), [2.5, 97.5])
        rows.append(dict(run=r, original=o.mean(), overlay=n.mean(), change=(n - o).mean(), ci_low=lo, ci_high=hi))
    # paired margin change for every run pair (e.g. mono-t5-3b minus rank-zephyr)
    pairs = []
    for i, r1 in enumerate(RUNS):
        for r2 in RUNS[i + 1:]:
            d = np.array([(pt[r1][t][1] - pt[r2][t][1]) - (pt[r1][t][0] - pt[r2][t][0]) for t in eligible])
            lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
            pairs.append(dict(a=r1, b=r2,
                              original_margin=float(np.mean([pt[r1][t][0] - pt[r2][t][0] for t in eligible])),
                              overlay_margin=float(np.mean([pt[r1][t][1] - pt[r2][t][1] for t in eligible])),
                              change=d.mean(), ci_low=lo, ci_high=hi))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "dl19_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
    with open(out / "dl19_ndcg10.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "dl19_topics.json").write_text(json.dumps({"eligible": eligible, "excluded": excluded}, indent=1))
    for x in rows:
        print(f"{x['run']:22s} {x['original']:.4f} {x['overlay']:.4f} {x['change']:+.4f} [{x['ci_low']:+.4f}, {x['ci_high']:+.4f}]")
    print(len(eligible), "eligible topics;", excluded)


if __name__ == "__main__":
    main()
