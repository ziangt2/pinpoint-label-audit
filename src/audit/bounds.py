"""Missing-tail identification bounds on the common candidate domain (paper Sec. III-E).

When fewer than 10 results of a cached top-50 list survive filtering to the common domain,
ranks after the surviving ones are unknown. For each label version we bound a query score by
(low) assuming no unknown rank is relevant and (high) placing every still-unretrieved projected
positive at the first unknown ranks. Frame-level bounds average per-query bounds; the change
bounds are [new_low - old_high, new_high - old_low]. Group-range bounds use
[max(0, max_low - min_high), max_high - min_low] per group. These are identification bounds,
not confidence intervals.
"""
from __future__ import annotations

from collections import defaultdict

from .metrics import ap_at_k, p_at_k

K = 10


def query_bounds(ranking, positives, metric):
    f = (lambda r, y: ap_at_k(r, y, K)) if metric == "AP10" else (lambda r, y: p_at_k(r, y, K))
    lo = f(ranking, positives)
    if len(ranking) >= K:
        return lo, lo
    missing = [p for p in dict.fromkeys(positives) if p not in set(ranking)]
    filled = list(ranking) + missing[: K - len(ranking)]
    return lo, f(filled, positives)


def frame_bounds(rows_by_query, metric, group_of):
    """rows_by_query: {query: (ranking, pos_old, pos_new)} -> dict of bounds."""
    b = {}
    for q, (r, yo, yn) in rows_by_query.items():
        b[q] = (*query_bounds(r, yo, metric), *query_bounds(r, yn, metric))
    n = len(b)
    mean = lambda i: sum(v[i] for v in b.values()) / n
    ol, oh, nl, nh = mean(0), mean(1), mean(2), mean(3)
    groups = defaultdict(list)
    for q, v in b.items():
        groups[group_of[q]].append(v)

    def rng(lo_i, hi_i):
        lo = sum(max(0.0, max(v[lo_i] for v in L) - min(v[hi_i] for v in L)) for L in groups.values()) / len(groups)
        hi = sum(max(v[hi_i] for v in L) - min(v[lo_i] for v in L) for L in groups.values()) / len(groups)
        return lo, hi

    rol, roh = rng(0, 1)
    rnl, rnh = rng(2, 3)
    return {
        metric: dict(old_low=ol, old_high=oh, new_low=nl, new_high=nh, delta_low=nl - oh, delta_high=nh - ol),
        metric[:-2] + "_range": dict(old_low=rol, old_high=roh, new_low=rnl, new_high=rnh,
                                     delta_low=rnl - roh, delta_high=rnh - rol),
    }
