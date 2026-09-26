# pinpoint-label-audit

Reproduction package for:

> Ziang Tan. **Relevance-Label Revisions in PinPoint: A Fixed-Output Audit of Two Public Releases.** Prepared for submission to IEEE BigData 2026 (Special Session on Machine Learning on Big Data).

The audit rescores the *same* cached rankings of seven released PinPoint runs under the labels of the first public data commit and of the current release. Requests, candidate domain and metric code are held fixed. Every table, figure and quoted number in the paper is regenerated from pinned public artifacts by the code in this repository.

## Layout

```
scripts/fetch_pinpoint.sh     fetch PinPoint inputs at pinned revisions; verify git blob IDs
scripts/fetch_dl19.sh         fetch DL19 assets (Parry et al. SIGIR'25 repo + TILDE qrels mirror)
scripts/reproduce_all.sh      end-to-end: fetch -> rebuild -> compare -> figures -> number check
src/audit/frame.py            matched-frame construction (1,298 -> 1,164 -> 344 groups)
src/audit/score.py            fixed-output rescoring (current and common candidate domains)
src/audit/bootstrap.py        component bootstrap, summary / contrast / bound tables
src/audit/bounds.py           missing-tail identification bounds (common domain)
src/audit/report.py           four-cell accounting report + output diagnostics (analysis tool)
src/audit/diagnostics.py      split-at-first-replacement, common-domain split, unique-denominator check
src/audit/case.py             real case for Fig. 1b chosen by a fixed rule + release-level label statistics
src/audit/robustness.py       relative gain / proportional-scaling test, wider unchanged-query sample,
                              component concentration and leave-one-out (Sec. IV-C, Table III)
src/audit/evaluator_sensitivity.py  re-implementation of the public evaluator (reproduces upstream
                              metrics.csv exactly) and its sensitivity metric under fixed outputs (Sec. IV-E)
paper/code/make_figures.py    --results DIR --figdir DIR: draws every figure from a results directory
paper/code/verify_numbers.py  --results DIR --tex FILE: checks every table cell and quoted number
src/audit/compare_v10.py      comparison with the earlier (v10) frozen outputs
src/dl19/check.py             TREC DL 2019 portability check
tests/test_audit.py           unit tests (pytest, or: python -m tests.test_audit)
results/reproduced/           outputs of this code: the source of every number in the paper
results/v10_frozen/           outputs of the earlier private analysis run, kept for comparison
paper/                        LaTeX source, figures, figure script, number checker
```

## Quick check without downloading data

```sh
pip install -r requirements.txt
python -m tests.test_audit                 # 10 unit tests
python paper/code/make_figures.py          # regenerates paper/figures from results/reproduced
python paper/code/verify_numbers.py        # 173 checks of tables and quoted numbers, expects "errors: 0"
python -m src.audit.compare_v10            # reproduced vs v10 frozen outputs
```

## Full reproduction

```sh
pip install -r requirements.txt            # pyarrow is needed to read the upstream parquet files
bash scripts/reproduce_all.sh results/run  # fetch, rebuild, compare, redraw figures, check numbers, compile paper
```

The script fetches all inputs and rebuilds `results/run/`. It then asserts that the rebuilt tables equal `results/reproduced/`, runs the tests, redraws the figures and checks the paper's numbers.

## What matches the earlier (v10) analysis

`src.audit.compare_v10` reports the following:

- **Exact matches.** Frame counts, all 168 point estimates, all 504 pairwise contrasts, the AP@10/P@10 missing-tail bounds and bootstrap component counts are identical up to float round-off (≤ 4e-16).
- **Bootstrap endpoints.** Endpoints differ by at most 0.003 because the draw order of the v10 run could not be recovered. The paper reports the intervals produced by this code (`results/reproduced`). No conclusion in the paper changes between the two runs.
- **Range bounds.** Group-*range* missing-tail upper bounds differ by ≤ 0.0009 in three cells. They are not reported in the paper.

## Reproduction record

`results/reproduced/REPRODUCTION_LOG.txt` records a clean-room run: fresh clone of this repository, fresh fetch of all
upstream inputs, full rebuild, comparison with the committed results (identical), figure regeneration (pixel-identical),
173 number checks (all pass) and compilation of the paper (8 pages, no warnings). The environment and the one deviation
from the default path (parquet labels read via a JSON export because no parquet engine was installed) are listed at
the top of the log.

## Pinned inputs

| Item | Identifier |
|---|---|
| Upstream repository | `github.com/pinterest/pinpoint-dataset` |
| Old labels (`pinpoint_metadata.parquet`, first public data commit `1dd9d69`) | blob `aca644ff93242fe683bed67b6bacf17cb22241e2` |
| First label replacement (commit) | `5fc977b40352205137584aaabf0aafab5c9576dc` |
| Current labels (`pinpoint_licensed.parquet` at `059d6e4`) | blob `54e2b3c3d572fe342e84e50a0374cecf43f65aaf` |
| Old / current candidate manifests (`index_signatures.txt`) | blobs `7483d669…` / `7150b4ea…` |
| Evaluator + cached runs (`standardized_results/`) | revision `059d6e4ae191d3f0775cbe2e9cf77baa0e714126` |
| DL19 rejudgments: `Parry-Parry/sigir25-annotation` | revision `12969fb4f189bd87fcfe3d4953d3ee0da110c379` |
| DL19 official qrels: `ielab/TILDE` `data/qrels/2019qrels-pass.txt` | sha256 `36e3687a…261686` |

## Analysis protocol

- **Matched groups.** Old rows sharing a normalized reference-image pair (`'None'` → null) and an identical positive set, with at least 2 queries per group. Groups losing any row or changing any reference signature are excluded. Selection never uses model scores. These groups are not the paraphrase sets of the original PinPoint protocol.
- **Primary domain.** The 109,600 current index IDs. Positive lists are projected onto the domain with duplicates retained.
- **Metrics.** AP@10 follows the upstream evaluator: set-membership hits and a `min(10, raw list length)` denominator. P@10 is the hit count divided by 10. Zero-positive queries are kept and score 0.
- **Group range.** Within each group, max minus min of query scores. Ranges are averaged equally over groups.
- **Intervals.** 5,000 percentile-bootstrap draws over connected components of queries sharing reference images, seed 20260920. Components are defined on the full frame; strata keep the components they intersect. One draw matrix is used per population, so contrasts are paired. Intervals are unadjusted.
- **Common domain.** 30,535 shared IDs. The exact subset has 271 groups (at least 10 surviving results for every run). Missing-tail bounds cover all 344 groups.

## Data licensing

PinPoint data and cached runs are fetched from the upstream repository and are not redistributed; see its `DATA_LICENSE.TXT`. The DL19 assets are fetched from their public repositories and are not redistributed. `results/` contains only aggregate statistics. Code in this repository is MIT-licensed (`LICENSE`).
