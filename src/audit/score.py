"""Fixed-output rescoring under old and current labels (paper Sec. III-C/D).

Writes per-query scores for every run x label version x domain:
  current_fixed : candidate domain = current index IDs (primary analysis)
  common_fixed  : candidate domain = old ∩ current IDs; rankings filtered to the domain;
                  a query is 'exact' only if >= 10 filtered results remain
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import default_paths, load_candidates, load_labels, load_runs, normalize_query_id
from .metrics import METRICS, ap_at_k

K = 10


def project(labels, domain):
    """Project a positive list onto a candidate domain, keeping multiplicity."""
    return [x for x in labels if x in domain]


def score_all(frame, old, cur, runs, cand_old, cand_cur):
    common = cand_old & cand_cur
    domains = {"current_fixed": cand_cur, "common_fixed": common}
    rows = []
    for gr in frame["groups"]:
        for q in gr["queries"]:
            nq = normalize_query_id(q)
            for dom, D in domains.items():
                y = {"old": project(old[q]["positive_candidates"], D),
                     "new": project(cur[q]["positive_candidates"], D)}
                for run, R in runs.items():
                    ranking = R[nq]
                    if dom == "common_fixed":
                        ranking = [x for x in ranking if x in D]
                    exact = len(ranking) >= K
                    rec = {"group": gr["group"], "split": gr["split"], "query": q, "domain": dom,
                           "run": run, "exact": exact, "n_ranked": len(ranking),
                           "n_pos_old": len(y["old"]), "n_pos_new": len(y["new"])}
                    for ver in ("old", "new"):
                        for m, f in METRICS.items():
                            rec[f"{m}_{ver}"] = f(ranking, y[ver])
                        rec[f"AP10u_{ver}"] = ap_at_k(ranking, y[ver], K, denominator="unique")
                    rec["top10"] = ranking[:K]
                    rows.append(rec)
    return rows, len(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = default_paths(a.data)
    frame = json.loads(Path(a.frame).read_text())
    rows, n_common = score_all(frame, load_labels(p["labels_old"]), load_labels(p["labels_current"]),
                               load_runs(p["runs"]), load_candidates(p["candidates_old"]),
                               load_candidates(p["candidates_current"]))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "query_scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"scored {len(rows)} rows; common domain size {n_common}")


if __name__ == "__main__":
    main()
