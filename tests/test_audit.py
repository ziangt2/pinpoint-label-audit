"""Unit tests for scoring and the four-cell report (no PinPoint download needed)."""
import math

from src.audit.bounds import query_bounds
from src.audit.metrics import ap_at_k, p_at_k, validate_ranking
from src.audit.report import four_cell, output_diagnostics
from src.audit.score import project

R = ["a", "n1", "b", "n2", "n3", "c", "n4", "n5", "n6", "n7"]


def test_ap_matches_upstream_convention():
    # hits at ranks 1, 3, 6 over 3 positives
    assert math.isclose(ap_at_k(R, ["a", "b", "c"]), (1 + 2 / 3 + 3 / 6) / 3)
    # duplicates kept in the denominator (raw list length), set membership for hits
    assert math.isclose(ap_at_k(R, ["a", "a", "b", "c"]), (1 + 2 / 3 + 3 / 6) / 4)
    assert math.isclose(ap_at_k(R, ["a", "a", "b", "c"], denominator="unique"), (1 + 2 / 3 + 3 / 6) / 3)
    assert ap_at_k(R, []) == 0.0 and p_at_k(R, []) == 0.0


def test_label_only_change_creates_group_range():
    # synthetic example of Fig. 1b: identical ranking, one paraphrase's positives replaced
    old = [ap_at_k(R, ["a", "b", "c"])] * 2
    new = [ap_at_k(R, ["a", "b", "c"]), ap_at_k(R, ["a", "d", "e"])]
    assert max(old) - min(old) == 0
    assert math.isclose(max(new) - min(new), 0.3888888888888889)


def test_output_only_change():
    rep = four_cell(0.2, 0.3, 0.2, 0.3)
    assert math.isclose(rep["interaction"], 0) and math.isclose(rep["label_first"]["label_under_old_output"], 0)


def test_label_only_change():
    rep = four_cell(0.2, 0.2, 0.1, 0.1)
    assert math.isclose(rep["output_first"]["output_under_old_labels"], 0) and math.isclose(rep["interaction"], 0)


def test_joint_change_identity_both_orders():
    f00, f10, f01, f11 = 0.26408, 0.30685, 0.19995, 0.22259
    rep = four_cell(f00, f10, f01, f11)
    o, l, i = rep["output_first"]["output_under_old_labels"], rep["label_first"]["label_under_old_output"], rep["interaction"]
    assert math.isclose(o + l + i, f11 - f00)
    assert math.isclose(sum(rep["output_first"].values()), f11 - f00)
    assert math.isclose(sum(rep["label_first"].values()), f11 - f00)
    assert math.isclose(i, -0.02013, abs_tol=1e-9)


def test_different_rankings_equal_quality():
    r2 = ["c", "n1", "b", "n2", "n3", "a", "n4", "n5", "n6", "n7"]
    assert ap_at_k(R, ["a", "b", "c"]) == ap_at_k(r2, ["a", "b", "c"])
    d = output_diagnostics(R, r2)
    assert d["ordered_mismatch"] == 1.0 and d["set_nonoverlap"] == 0.0 and d["prefix_disagreement"] > 0


def test_rejects_short_or_duplicate_rankings():
    for bad in (R[:9], ["a"] * 10):
        try:
            validate_ranking(bad, 10)
        except ValueError:
            continue
        raise AssertionError("invalid ranking accepted")


def test_projection_keeps_multiplicity():
    assert project(["a", "a", "x", "b"], {"a", "b"}) == ["a", "a", "b"]


def test_missing_tail_bounds():
    lo, hi = query_bounds(["a", "n1"], ["a", "b", "c"], "AP10")
    assert math.isclose(lo, 1 / 3) and math.isclose(hi, (1 + 2 / 3 + 3 / 4) / 3)
    lo, hi = query_bounds(R, ["a", "b", "c"], "AP10")
    assert lo == hi


def test_scaling_statistics_on_proportional_toy():
    """If reranked and base scores shrink by the same factor, deviation from proportional is 0."""
    from src.audit.robustness import Pop, stats
    comps = [["q1", "q2"], ["q3"], ["q4"]]
    base = {"q1": .2, "q2": .4, "q3": .3, "q4": .1}
    sc = {}
    from src.audit.robustness import RUNS
    for r in RUNS:
        mult = 1.2 if r == "S1 reranked" else 1.0
        sc[(r, "old")] = {q: v * mult for q, v in base.items()}
        sc[(r, "new")] = {q: v * mult * 0.5 for q, v in base.items()}
    out = stats(Pop(comps, sc))
    assert math.isclose(out["deviation_from_proportional"][0], 0, abs_tol=1e-12)
    assert math.isclose(out["rel_change"][0], 0, abs_tol=1e-12)
    assert math.isclose(out["rel_old"][0], 0.2)


if __name__ == "__main__":  # also runnable without pytest: python -m tests.test_audit
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
