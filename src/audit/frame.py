"""Matched-frame construction (paper Sec. III-B).

Groups: old rows sharing a normalized reference-image pair AND an identical positive set,
with >= 2 queries. Excluded: groups that lose any query in the current release, and groups
in which any query's reference pair changes. Selection never looks at model outputs.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .data import default_paths, load_labels


def build_frame(old: dict, cur: dict) -> dict:
    g = defaultdict(list)
    for qid, r in old.items():
        g[(r["ref"], frozenset(r["positive_candidates"]))].append(qid)
    multi = {k: sorted(v) for k, v in g.items() if len(v) >= 2}
    complete = {k: v for k, v in multi.items() if all(q in cur for q in v)}
    ref_changed = {k for k, v in complete.items() if any(cur[q]["ref"] != old[q]["ref"] for q in v)}
    eligible = {k: v for k, v in complete.items() if k not in ref_changed}

    groups = []
    for i, (k, qs) in enumerate(sorted(eligible.items(), key=lambda kv: kv[1][0])):
        cur_sets = {frozenset(cur[q]["positive_candidates"]) for q in qs}
        changed = [q for q in qs if set(cur[q]["positive_candidates"]) != set(old[q]["positive_candidates"])]
        groups.append({
            "group": f"g{i:04d}",
            "queries": qs,
            "ref": list(k[0]),
            "split": len(cur_sets) > 1,
            "n_changed_rows": len(changed),
        })
    instr_changes = sum(cur[q]["instruction"] != old[q]["instruction"] for gr in groups for q in gr["queries"])
    split = [gr for gr in groups if gr["split"]]
    summary = {
        "earliest_rows": len(old),
        "multirow_groups": len(multi),
        "groups_missing_any_query": len(multi) - len(complete),
        "complete_groups": len(complete),
        "complete_but_reference_changed": len(ref_changed),
        "eligible_groups": len(groups),
        "eligible_queries": sum(len(gr["queries"]) for gr in groups),
        "instruction_changes": instr_changes,
        "split": len(split),
        "split_with_some_unchanged": sum(1 for gr in split if gr["n_changed_rows"] < len(gr["queries"])),
        "split_all_rows_changed": sum(1 for gr in split if gr["n_changed_rows"] == len(gr["queries"])),
        "changed_rows_in_other_groups": sum(gr["n_changed_rows"] for gr in groups if not gr["split"]),
    }
    return {"summary": summary, "groups": groups}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = default_paths(a.data)
    fr = build_frame(load_labels(p["labels_old"]), load_labels(p["labels_current"]))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "frame.json").write_text(json.dumps(fr, indent=1))
    (Path(a.out) / "frame_summary.json").write_text(json.dumps(fr["summary"], indent=1))
    print(json.dumps(fr["summary"], indent=1))


if __name__ == "__main__":
    main()
