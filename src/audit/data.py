"""Loading pinned PinPoint artifacts.

Label files are the upstream parquet files; a JSON export with the same columns
(list of row objects) is also accepted so the pipeline can run where no parquet
reader is installed.
"""
from __future__ import annotations

import json
from pathlib import Path

RUN_FILES = {
    "S1": "bge_vl_mllm_s1_retrieval_results_licensed.json",
    "S1 reranked": "bge_vl_mllm_s1_retrieval_results_licensed_reranked.json",
    "S2": "bge_vl_mllm_s2_retrieval_results_licensed.json",
    "MetaCLIP2 Combined": "metaclip2_retrieval_results_combined.json",
    "MetaCLIP2 Image-only": "metaclip2_retrieval_results_image_only.json",
    "MetaCLIP2 Text-only": "metaclip2_retrieval_results_text_only.json",
    "MetaCLIP2 SLERP": "metaclip2_slerp_retrieval_results_licensed.json",
}
# Method identifiers used in results/frozen/*.csv
METHOD_ID = {k: v[:-5] for k, v in RUN_FILES.items()}


def _as_list(x):
    if x is None:
        return []
    if hasattr(x, "tolist"):
        return list(x.tolist())
    return list(x)


def norm_sig(x):
    """Normalize a reference-image signature ('None', '', NaN -> None)."""
    if x is None:
        return None
    if isinstance(x, float):  # NaN
        return None
    x = str(x)
    return None if x in ("", "None", "nan") else x


def load_labels(path: str | Path) -> dict:
    """Return {query_id: row} with list-typed positive/negative candidates."""
    path = Path(path)
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
    else:
        import pandas as pd  # requires pyarrow or fastparquet
        rows = pd.read_parquet(path).to_dict(orient="records")
    out = {}
    for r in rows:
        r = dict(r)
        r["positive_candidates"] = _as_list(r.get("positive_candidates"))
        r["negative_candidates"] = _as_list(r.get("negative_candidates"))
        r["ref"] = (norm_sig(r.get("query_image_signature")), norm_sig(r.get("query_image_signature2")))
        out[r["query_id"]] = r
    return out


def load_candidates(path: str | Path) -> set:
    return set(Path(path).read_text().split())


def normalize_query_id(qid) -> str:
    """Same rule as the upstream evaluator (utils/data_utils.py)."""
    if isinstance(qid, str):
        if qid.startswith("query_"):
            return qid[6:].zfill(5)
        return qid.zfill(5)
    return str(qid).zfill(5)


def load_runs(run_dir: str | Path) -> dict:
    """{display name: {normalized query id: [retrieved ids]}}"""
    run_dir = Path(run_dir)
    runs = {}
    for name, fn in RUN_FILES.items():
        raw = json.loads((run_dir / fn).read_text())
        runs[name] = {k: v["retrieved_items"] for k, v in raw.items()}
    return runs


def default_paths(data_dir: str | Path) -> dict:
    d = Path(data_dir)
    # Labels are read from the upstream parquet files. Where no parquet engine is installed, a JSON
    # export with identical rows can be used instead by setting LABELS_FROM_JSON=1 (see README).
    import os
    exts = (".json", ".parquet") if os.environ.get("LABELS_FROM_JSON") == "1" else (".parquet", ".json")

    def pick(stem):
        for ext in exts:
            if (d / f"{stem}{ext}").exists():
                return d / f"{stem}{ext}"
        raise FileNotFoundError(d / stem)
    return {
        "labels_old": pick("labels_old"),
        "labels_current": pick("labels_current"),
        "candidates_old": d / "candidates_old.txt",
        "candidates_current": d / "candidates_current.txt",
        "runs": d / "runs",
    }
