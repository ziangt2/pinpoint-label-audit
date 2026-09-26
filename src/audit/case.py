"""Real case for Fig. 1b and Sec. IV-A, chosen by a fixed rule, plus release-level label statistics.

Selection rule (fixed before drawing the figure): among split groups in which the
MetaCLIP2 Image-only top-10 contains an item whose positive status differs between the two
label versions for some query, take the group whose current Image-only AP@10 range is the
median. With an even number of candidates the lower median is used, i.e. position
(n - 1) // 2 (0-based) after sorting by (range, group id) ascending; ties in range are broken
by group id. Image-only rankings are identical within every
matched group, so the case isolates the label change. The case documents what the public files
contain; it does not judge which label version is correct or whether the requests are
semantically equivalent.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from .data import default_paths, load_labels

IMG = "MetaCLIP2 Image-only"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    a = ap.parse_args()
    p = default_paths(a.data)
    old, cur = load_labels(p["labels_old"]), load_labels(p["labels_current"])
    inp = Path(a.inp)
    frame = json.loads((inp / "frame.json").read_text())
    by = defaultdict(dict)
    for line in open(inp / "query_scores.jsonl"):
        r = json.loads(line)
        if r["domain"] == "current_fixed":
            by[r["run"]][r["query"]] = r

    def rng(g, run, v):
        s = [by[run][q][f"AP10_{v}"] for q in g["queries"]]
        return max(s) - min(s)

    def status_change(g):
        for q in g["queries"]:
            o, n = set(old[q]["positive_candidates"]), set(cur[q]["positive_candidates"])
            if any((x in o) != (x in n) for x in by[IMG][q]["top10"]):
                return True
        return False

    groups = frame["groups"]
    split = [g for g in groups if g["split"]]
    assert all(len({tuple(by[IMG][q]["top10"]) for q in g["queries"]}) == 1 for g in groups)
    eligible = sorted([g for g in split if status_change(g)], key=lambda g: (rng(g, IMG, "new"), g["group"]))
    pos = (len(eligible) - 1) // 2
    g = eligible[pos]
    ranges = [round(rng(x, IMG, "new"), 12) for x in eligible]
    tie_at_median = ranges.count(ranges[pos]) > 1
    top10 = by[IMG][g["queries"][0]]["top10"]
    queries = []
    for q in g["queries"]:
        o, n = old[q]["positive_candidates"], cur[q]["positive_candidates"]
        queries.append({
            "query_id": q, "instruction": old[q]["instruction"],
            "instruction_unchanged": old[q]["instruction"] == cur[q]["instruction"],
            "positives_old": o, "positives_current": n,
            "top10_status": ["both" if (x in o and x in n) else "old_only" if x in o else "current_only" if x in n else "none"
                             for x in top10],
            "scores": {run: {"AP10_old": by[run][q]["AP10_old"], "AP10_new": by[run][q]["AP10_new"]}
                       for run in (IMG, "S1", "S1 reranked")},
        })
    changed = [q for gg in split for q in gg["queries"]
               if set(cur[q]["positive_candidates"]) != set(old[q]["positive_candidates"])]
    jac = [len(set(cur[q]["positive_candidates"]) & set(old[q]["positive_candidates"])) /
           len(set(cur[q]["positive_candidates"]) | set(old[q]["positive_candidates"])) for q in changed]
    dup = lambda rows: sum(len(r["positive_candidates"]) != len(set(r["positive_candidates"])) for r in rows)
    frame_q = [q for gg in groups for q in gg["queries"]]
    out = {
        "selection_rule": __doc__.split("Selection rule (fixed before drawing the figure): ")[1].split("\n\n")[0].replace("\n", " "),
        "n_eligible_groups": len(eligible),
        "median_position_1based": pos + 1,
        "tie_at_median": tie_at_median,
        "eligible_sorted": [{"group": x["group"], "range_current_image_only": r} for x, r in zip(eligible, ranges)],
        "eligible_range_median": st.median(rng(x, IMG, "new") for x in eligible),
        "case_group": g["group"], "reference": g["ref"], "top10_image_only": top10,
        "case_range_image_only": {"old": rng(g, IMG, "old"), "new": rng(g, IMG, "new")},
        "case_range_S1_reranked": {"old": rng(g, "S1 reranked", "old"), "new": rng(g, "S1 reranked", "new")},
        "queries": queries,
        "context": {
            "split_group_sizes": dict(Counter(len(x["queries"]) for x in split)),
            "distinct_current_sets_per_split_group": dict(Counter(len({frozenset(cur[q]["positive_candidates"]) for q in x["queries"]}) for x in split)),
            "changed_rows_per_split_group": dict(sorted(Counter(x["n_changed_rows"] for x in split).items())),
            "changed_rows_in_split_groups": len(changed),
            "median_jaccard_old_vs_current_changed_rows": st.median(jac),
            "changed_rows_with_zero_overlap": sum(j == 0 for j in jac),
            "rows_with_duplicate_positive_ids": {"old_release": dup(old.values()), "current_release": dup(cur.values()),
                                                 "old_frame": dup(old[q] for q in frame_q), "current_frame": dup(cur[q] for q in frame_q)},
            "rows_old_release": len(old), "rows_current_release": len(cur),
            "shared_rows_with_changed_positive_set": sum(set(cur[q]["positive_candidates"]) != set(old[q]["positive_candidates"]) for q in cur),
        },
    }
    (inp / "case.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("queries",)}, indent=1))


if __name__ == "__main__":
    main()
