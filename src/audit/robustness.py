"""Robustness analyses for the reranking result (paper Sec. IV-E). Added after the main analysis
to address three reviewer-style questions; all are exploratory and use unadjusted intervals.

(1) Scaling. Is the smaller reranking advantage just the overall score drop?
    relative gain r_v = (A_v - S_v) / S_v, where A = S1 reranked, S = S1, v in {old, current};
    change in relative gain r_cur - r_old; and deviation from proportional scaling
    D = adv_cur - adv_old * S_cur / S_old (the current advantage expected if the advantage fell
    in proportion to the S1 score).
(2) Population. Do the results extend beyond the 344 constructed groups? Wider sample: every query
    present in both releases with unchanged instruction and reference-image IDs and complete
    caches for all seven runs (no group or shared-positive requirement). Reported for all such
    queries, for the 2,091 frame queries and for the other queries.
(3) Concentration. Is the reranking-change result driven by a few reference-image components?
    Per-component contributions, top-5 share of absolute contribution, and leave-one-component-out.

Intervals: 5,000 percentile-bootstrap draws over reference-image components (seed 20260920),
one draw matrix per population so all statistics are paired.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import RUN_FILES, default_paths, load_candidates, load_labels, load_runs, normalize_query_id
from .metrics import ap_at_k
from .score import project

SEED, DRAWS = 20260920, 5000
RUNS = list(RUN_FILES)
S, A = "S1", "S1 reranked"


def components(queries, ref):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for q in queries:
        find(("q", q))
        for s in ref[q]:
            if s:
                parent[find(("q", q))] = find(("s", s))
    comp = defaultdict(list)
    for q in queries:
        comp[find(("q", q))].append(q)
    return sorted((sorted(v) for v in comp.values()), key=lambda c: c[0])


class Pop:
    def __init__(self, comps, scores):
        self.comps = comps
        self.n = np.array([len(c) for c in comps], float)
        self.idx = np.random.default_rng(SEED).integers(0, len(comps), size=(DRAWS, len(comps)))
        self.sums = {k: np.array([sum(v[q] for q in c) for c in comps]) for k, v in scores.items()}

    def point_and_boot(self, f):
        """f maps a dict of total sums (+ 'n') to a statistic."""
        tot = {k: v.sum() for k, v in self.sums.items()}; tot["n"] = self.n.sum()
        bt = {k: v[self.idx].sum(1) for k, v in self.sums.items()}; bt["n"] = self.n[self.idx].sum(1)
        pt = f(tot)
        lo, hi = np.percentile(f(bt), [2.5, 97.5])
        return float(pt), float(lo), float(hi)


def stats(pop):
    m = lambda k: (lambda t: t[k] / t["n"])
    adv = lambda v: (lambda t: (t[(A, v)] - t[(S, v)]) / t["n"])
    rel = lambda v: (lambda t: (t[(A, v)] - t[(S, v)]) / t[(S, v)])
    out = {"queries": int(pop.n.sum()), "components": len(pop.comps)}
    for r in RUNS:
        for v in ("old", "new"):
            out[f"AP10_{v}|{r}"] = pop.point_and_boot(m((r, v)))
        out[f"AP10_delta|{r}"] = pop.point_and_boot(lambda t, r=r: (t[(r, "new")] - t[(r, "old")]) / t["n"])
    for v in ("old", "new"):
        out[f"adv_{v}"] = pop.point_and_boot(adv(v))
        out[f"rel_{v}"] = pop.point_and_boot(rel(v))
    out["adv_change"] = pop.point_and_boot(lambda t: adv("new")(t) - adv("old")(t))
    out["rel_change"] = pop.point_and_boot(lambda t: rel("new")(t) - rel("old")(t))
    out["adv_expected_if_proportional"] = pop.point_and_boot(lambda t: adv("old")(t) * t[(S, "new")] / t[(S, "old")])
    out["deviation_from_proportional"] = pop.point_and_boot(
        lambda t: adv("new")(t) - adv("old")(t) * t[(S, "new")] / t[(S, "old")])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--in", dest="inp", required=True, help="directory with frame.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = default_paths(a.data)
    old, cur, runs = load_labels(p["labels_old"]), load_labels(p["labels_current"]), load_runs(p["runs"])
    C = load_candidates(p["candidates_current"])
    frame = json.loads((Path(a.inp) / "frame.json").read_text())
    frame_q = {q for g in frame["groups"] for q in g["queries"]}
    frame_sorted = sorted(frame_q)  # deterministic summation order (set order varies between processes)

    # ---- wider unchanged-query sample
    shared = sorted(q for q in cur if q in old)
    wider = [q for q in shared if old[q]["instruction"] == cur[q]["instruction"] and old[q]["ref"] == cur[q]["ref"]
             and all(len(runs[r].get(normalize_query_id(q), [])) >= 10 for r in RUNS)]
    assert frame_q <= set(wider)
    scores = defaultdict(dict)
    for q in wider:
        for r in RUNS:
            rk = runs[r][normalize_query_id(q)]
            scores[(r, "old")][q] = ap_at_k(rk, project(old[q]["positive_candidates"], C), 10)
            scores[(r, "new")][q] = ap_at_k(rk, project(cur[q]["positive_candidates"], C), 10)
    ref = {q: old[q]["ref"] for q in wider}
    wide_comps = components(wider, ref)

    def restrict(comps, keep):
        return [[q for q in c if q in keep] for c in comps if any(q in keep for q in c)]

    frame_comps = components(sorted(frame_q), ref)  # identical to the main analysis' 312 components
    populations = {
        "frame_2091": frame_comps,
        "wider_all": wide_comps,
        "wider_non_frame": restrict(wide_comps, set(wider) - frame_q),
    }
    res = {"sample": {"shared_rows": len(shared), "wider_queries": len(wider), "frame_queries": len(frame_q),
                      "non_frame_queries": len(set(wider) - frame_q),
                      "wider_changed_positive_set": sum(set(old[q]["positive_candidates"]) != set(cur[q]["positive_candidates"]) for q in wider),
                      "non_frame_changed_positive_set": sum(set(old[q]["positive_candidates"]) != set(cur[q]["positive_candidates"]) for q in set(wider) - frame_q)}}
    for name, comps in populations.items():
        res[name] = stats(Pop(comps, scores))

    # ---- concentration / leave-one-component-out on the main frame
    d = {q: (scores[(A, "new")][q] - scores[(S, "new")][q]) - (scores[(A, "old")][q] - scores[(S, "old")][q]) for q in frame_sorted}
    N = len(frame_q)
    contrib = np.array([sum(d[q] for q in c) / N for c in frame_comps])
    order = np.argsort(-np.abs(contrib))
    tot = {k: sum(scores[k][q] for q in frame_sorted) for k in scores}
    loo_adv, loo_rel, loo_dev = [], [], []
    for c in frame_comps:
        n = N - len(c)
        t = {k: tot[k] - sum(scores[k][q] for q in c) for k in scores}
        ao, an = (t[(A, "old")] - t[(S, "old")]) / n, (t[(A, "new")] - t[(S, "new")]) / n
        loo_adv.append(an - ao)
        loo_rel.append((t[(A, "new")] - t[(S, "new")]) / t[(S, "new")] - (t[(A, "old")] - t[(S, "old")]) / t[(S, "old")])
        loo_dev.append(an - ao * t[(S, "new")] / t[(S, "old")])
    res["concentration_frame"] = {
        "components": len(frame_comps),
        "adv_change": float(contrib.sum()),
        "top5_share_of_abs_contribution": float(np.abs(contrib)[order[:5]].sum() / np.abs(contrib).sum()),
        "top5_sizes": [len(frame_comps[i]) for i in order[:5]],
        "top5_contributions": [float(contrib[i]) for i in order[:5]],
        "components_with_negative_contribution": int((contrib < 0).sum()),
        "components_with_positive_contribution": int((contrib > 0).sum()),
        "components_with_zero_contribution": int((contrib == 0).sum()),
        "loo_adv_change_min": float(min(loo_adv)), "loo_adv_change_max": float(max(loo_adv)),
        "loo_rel_change_min": float(min(loo_rel)), "loo_rel_change_max": float(max(loo_rel)),
        "loo_deviation_min": float(min(loo_dev)), "loo_deviation_max": float(max(loo_dev)),
        "loo_all_adv_change_negative": bool(max(loo_adv) < 0),
        "loo_all_deviation_negative": bool(max(loo_dev) < 0),
        "without_top5_adv_change": float(sum(d[q] for i in range(len(frame_comps)) if i not in set(order[:5]) for q in frame_comps[i])
                                         / sum(len(frame_comps[i]) for i in range(len(frame_comps)) if i not in set(order[:5]))),
    }
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "robustness.json").write_text(json.dumps(res, indent=1))
    rows = []
    for pop in populations:
        for k, v in res[pop].items():
            if isinstance(v, tuple):
                rows.append(dict(population=pop, statistic=k, estimate=v[0], ci_low=v[1], ci_high=v[2]))
    with open(out / "robustness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for pop in populations:
        r = res[pop]
        print(f"[{pop}] n={r['queries']} comps={r['components']}")
        for k in ("adv_old", "adv_new", "adv_change", "rel_old", "rel_new", "rel_change",
                  "adv_expected_if_proportional", "deviation_from_proportional"):
            e, lo, hi = r[k]; print(f"   {k:30s} {e:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(json.dumps(res["sample"])); print(json.dumps(res["concentration_frame"], indent=1))


if __name__ == "__main__":
    main()
