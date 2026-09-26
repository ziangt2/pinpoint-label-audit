"""Additional checks quoted in the paper (Secs. IV-A and IV-B)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .data import default_paths, load_candidates, load_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    a = ap.parse_args()
    p = default_paths(a.data)
    inp = Path(a.inp)
    frame = json.loads((inp / "frame.json").read_text())
    cur = load_labels(p["labels_current"])
    common = load_candidates(p["candidates_old"]) & load_candidates(p["candidates_current"])
    out = {}
    import os
    order = ("json", "parquet") if os.environ.get("LABELS_FROM_JSON") == "1" else ("parquet", "json")
    t0_path = next((Path(a.data) / f"labels_t0.{e}" for e in order if (Path(a.data) / f"labels_t0.{e}").exists()),
                   Path(a.data) / "labels_t0.parquet")
    if t0_path.exists():  # labels at commit 5fc977b (first replacement)
        t0 = load_labels(t0_path)
        out["split_at_first_replacement"] = sum(
            1 for g in frame["groups"] if len({frozenset(t0[q]["positive_candidates"]) for q in g["queries"]}) > 1)
    out["split_groups_still_split_on_common_domain"] = sum(
        1 for g in frame["groups"] if g["split"] and
        len({frozenset(x for x in cur[q]["positive_candidates"] if x in common) for q in g["queries"]}) > 1)
    old = load_labels(p["labels_old"])
    Ccur = load_candidates(p["candidates_current"])
    projn = lambda lab, q: frozenset(x for x in lab[q]["negative_candidates"] if x in Ccur)
    fq = [q for g in frame["groups"] for q in g["queries"]]
    out["negatives_projected"] = {
        "rows_changed": sum(projn(old, q) != projn(cur, q) for q in fq), "rows": len(fq),
        "groups_sharing_old": sum(len({projn(old, q) for q in g["queries"]}) == 1 for g in frame["groups"]),
        "groups_sharing_current": sum(len({projn(cur, q) for q in g["queries"]}) == 1 for g in frame["groups"]),
        "split_groups_losing_shared_negatives": sum(g["split"] and len({projn(cur, q) for q in g["queries"]}) > 1 for g in frame["groups"]),
        "other_groups_losing_shared_negatives": sum((not g["split"]) and len({projn(cur, q) for q in g["queries"]}) > 1 for g in frame["groups"]),
    }
    u = defaultdict(list)
    for line in open(inp / "query_scores.jsonl"):
        r = json.loads(line)
        if r["domain"] == "current_fixed":
            u[r["run"]].append(r["AP10u_new"] - r["AP10_new"])
    out["unique_denominator_gain_current_AP10"] = {k: sum(v) / len(v) for k, v in u.items()}
    (inp / "diagnostics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
