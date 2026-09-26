"""Metrics, matching the public PinPoint evaluator (utils/metrics.py at 059d6e4)."""
from __future__ import annotations


def ap_at_k(retrieved, relevant, k=10, denominator="raw"):
    """AP@k with set-membership hits.

    denominator="raw": min(k, len(relevant)) with duplicates retained (upstream behaviour).
    denominator="unique": min(k, len(set(relevant))) -- diagnostic only.
    """
    if not relevant:
        return 0.0
    rel = set(relevant)
    hits, s = 0, 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in rel:
            hits += 1
            s += hits / (i + 1)
    n = len(relevant) if denominator == "raw" else len(rel)
    return s / min(n, k)


def p_at_k(retrieved, relevant, k=10):
    if not relevant:
        return 0.0
    rel = set(relevant)
    return sum(1 for x in retrieved[:k] if x in rel) / k


def validate_ranking(retrieved, k):
    """Reject rankings the exact report cannot score."""
    if len(retrieved) < k:
        raise ValueError(f"ranking shorter than k={k}")
    if len(set(retrieved[:k])) != k:
        raise ValueError("duplicate result IDs in top-k")


METRICS = {"AP10": lambda r, y: ap_at_k(r, y, 10), "P10": lambda r, y: p_at_k(r, y, 10)}
