"""Paired component bootstrap and summary tables (paper Sec. III-D).

Resampling unit: connected components of queries that share any reference-image signature,
defined on the full matched frame and restricted to each analysed population. 5,000 draws, seed 20260920,
percentile intervals. All statistics in one population reuse the same draw matrix, so
contrasts are paired. Outputs follow the schema of results/frozen/*.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from .bounds import frame_bounds
from .data import METHOD_ID, RUN_FILES, default_paths, load_candidates, load_labels, load_runs, normalize_query_id
from .score import project

SEED, DRAWS = 20260920, 5000
RUNS = list(RUN_FILES)
STRATA = ("all344", "split121", "other223")


def components(groups):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g in groups:
        for q in g["queries"]:
            find(("q", q))
            for s in g["ref"]:
                if s:
                    parent[find(("q", q))] = find(("s", s))
    comp = defaultdict(list)
    for g in groups:
        comp[find(("q", g["queries"][0]))].append(g)
    # deterministic order: by smallest query id in the component
    return sorted(comp.values(), key=lambda gs: min(q for g in gs for q in g["queries"]))


class Population:
    """Per-component sums so that every bootstrap statistic is a ratio of resampled sums."""

    def __init__(self, groups, scores, frame_groups):
        # Component identity comes from the full 344-group frame; a stratum or subset keeps the
        # global components that intersect it (restricted to its own groups).
        keep = {g["group"] for g in groups}
        self.comps = [[g for g in c if g["group"] in keep] for c in components(frame_groups)]
        self.comps = [c for c in self.comps if c]
        self.n_groups = len(groups)
        self.n_queries = sum(len(g["queries"]) for g in groups)
        rng = np.random.default_rng(SEED)
        self.idx = rng.integers(0, len(self.comps), size=(DRAWS, len(self.comps)))
        self.qcount = np.array([sum(len(g["queries"]) for g in c) for c in self.comps], float)
        self.gcount = np.array([len(c) for c in self.comps], float)
        self.scores = scores  # {(run, metric, ver): {query: value}}

    def qmean(self, run, metric, ver):
        s = self.scores[(run, metric, ver)]
        return np.array([sum(s[q] for g in c for q in g["queries"]) for c in self.comps])

    def grange(self, run, metric, ver):
        s = self.scores[(run, metric, ver)]
        return np.array([sum(max(s[q] for q in g["queries"]) - min(s[q] for q in g["queries"]) for g in c)
                         for c in self.comps])

    def stat(self, comp_sums, weights):
        point = comp_sums.sum() / weights.sum()
        boot = comp_sums[self.idx].sum(1) / weights[self.idx].sum(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return point, lo, hi


def per_comp(pop, run, metric, ver):
    if metric in ("AP10", "P10"):
        return pop.qmean(run, metric, ver), pop.qcount
    return pop.grange(run, metric.split("_")[0] + "10", ver), pop.gcount


def analyse(pop, dom, st):
    summ, cont = [], []
    for metric in ("AP10", "P10", "AP_range", "P_range"):
        cache = {}
        for run in RUNS:
            o, w = per_comp(pop, run, metric, "old")
            n, _ = per_comp(pop, run, metric, "new")
            cache[run] = (o, n, w)
            po = o.sum() / w.sum(); pn = n.sum() / w.sum()
            _, lo, hi = pop.stat(n - o, w)
            summ.append(dict(domain=dom, stratum=st, method=METHOD_ID[run], metric=metric, old=po, new=pn,
                             delta=pn - po, ci_low=lo, ci_high=hi, groups=pop.n_groups,
                             queries=pop.n_queries, components=len(pop.comps)))
        for a, b in combinations(RUNS, 2):
            oa, na, w = cache[a]; ob, nb, _ = cache[b]
            od, olo, ohi = pop.stat(oa - ob, w)
            nd, nlo, nhi = pop.stat(na - nb, w)
            cd, clo, chi = pop.stat((na - nb) - (oa - ob), w)
            cont.append(dict(domain=dom, stratum=st, metric=metric, a=METHOD_ID[a], b=METHOD_ID[b],
                             old_difference=od, new_difference=nd, old_ci_low=olo, old_ci_high=ohi,
                             new_ci_low=nlo, new_ci_high=nhi, difference_change=cd, change_ci_low=clo,
                             change_ci_high=chi, point_sign_preserved=bool(np.sign(od) == np.sign(nd))))
    return summ, cont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--in", dest="inp", required=True, help="directory with frame.json and query_scores.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    inp, out = Path(a.inp), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = json.loads((inp / "frame.json").read_text())
    rows = [json.loads(l) for l in open(inp / "query_scores.jsonl")]

    by_dom = defaultdict(lambda: defaultdict(dict))
    inexact_groups = set()
    for r in rows:
        for m in ("AP10", "P10"):
            for v in ("old", "new"):
                by_dom[r["domain"]][(r["run"], m, v)][r["query"]] = r[f"{m}_{v}"]
        if r["domain"] == "common_fixed" and not r["exact"]:
            inexact_groups.add(r["group"])

    summary, contrasts = [], []
    for dom in ("current_fixed", "common_fixed"):
        for st in STRATA:
            gs = [g for g in frame["groups"] if st == "all344" or g["split"] == (st == "split121")]
            if dom == "common_fixed":
                gs = [g for g in gs if g["group"] not in inexact_groups]
            s, c = analyse(Population(gs, by_dom[dom], frame["groups"]), dom, st)
            summary += s; contrasts += c

    # identification bounds (all 344 groups, both domains)
    p = default_paths(a.data)
    old, cur, runs = load_labels(p["labels_old"]), load_labels(p["labels_current"]), load_runs(p["runs"])
    doms = {"current_fixed": load_candidates(p["candidates_current"])}
    doms["common_fixed"] = load_candidates(p["candidates_old"]) & doms["current_fixed"]
    bounds = []
    for dom, D in doms.items():
        for st in STRATA:
            gs = [g for g in frame["groups"] if st == "all344" or g["split"] == (st == "split121")]
            group_of = {q: g["group"] for g in gs for q in g["queries"]}
            for run in RUNS:
                rq = {q: ([x for x in runs[run][normalize_query_id(q)] if x in D],
                          project(old[q]["positive_candidates"], D), project(cur[q]["positive_candidates"], D))
                      for q in group_of}
                for metric in ("AP10", "P10"):
                    for m, v in frame_bounds(rq, metric, group_of).items():
                        bounds.append(dict(domain=dom, stratum=st, method=METHOD_ID[run], metric=m, **v))

    for name, data in (("summary.csv", summary), ("model_contrasts.csv", contrasts), ("identified_bounds.csv", bounds)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader(); w.writerows(data)
    print(f"wrote {len(summary)} summary rows, {len(contrasts)} contrasts, {len(bounds)} bound rows to {out}")


if __name__ == "__main__":
    main()
