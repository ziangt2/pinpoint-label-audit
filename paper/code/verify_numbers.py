"""Check every table cell and the numbers quoted in paper/main.tex against results/reproduced.

Each check recomputes the quoted value from the regenerated CSV/JSON files and asserts that the
exact string printed in the paper appears in main.tex. Run: python paper/code/verify_numbers.py
"""
import csv
import json
import statistics as st
from pathlib import Path

import argparse
R = Path(__file__).resolve().parents[2]
_ap = argparse.ArgumentParser()
_ap.add_argument("--results", default=str(R / "results/reproduced"), help="directory with regenerated results")
_ap.add_argument("--tex", default=str(R / "paper/main.tex"))
_args = _ap.parse_args()
RES = Path(_args.results)
S = list(csv.DictReader((RES / "summary.csv").open()))
C = list(csv.DictReader((RES / "model_contrasts.csv").open()))
B = list(csv.DictReader((RES / "identified_bounds.csv").open()))
D = list(csv.DictReader((RES / "dl19_ndcg10.csv").open()))
DP = list(csv.DictReader((RES / "dl19_pairs.csv").open()))
FR = json.loads((RES / "frame_summary.json").read_text())
DG = json.loads((RES / "diagnostics.json").read_text())
FC = json.loads((RES / "four_cell_S1_vs_S1_reranked.json").read_text())
tex = Path(_args.tex).read_text()
print(f"results: {RES}\ntex: {_args.tex}")

RUN_ORDER = ["S1", "S1 reranked", "S2", "MetaCLIP2 Combined", "MetaCLIP2 Image-only", "MetaCLIP2 Text-only", "MetaCLIP2 SLERP"]
KEY = {"S1 reranked": "bge_vl_mllm_s1_retrieval_results_licensed_reranked", "S1": "bge_vl_mllm_s1_retrieval_results_licensed",
       "S2": "bge_vl_mllm_s2_retrieval_results_licensed", "MC2 Combined": "metaclip2_retrieval_results_combined",
       "MC2 SLERP": "metaclip2_slerp_retrieval_results_licensed", "MC2 Text-only": "metaclip2_retrieval_results_text_only",
       "MC2 Image-only": "metaclip2_retrieval_results_image_only"}
errs = 0
checked = 0


def chk(cond, msg):
    global errs, checked
    checked += 1
    if not cond:
        errs += 1
        print("FAIL", msg)


def has(s, msg=None):
    chk(s in tex, msg or f"text not found: {s}")


def g(d, st_, m, me):
    return next(x for x in S if x["domain"] == d and x["stratum"] == st_ and x["metric"] == m and x["method"] == me)


def dot(v, n):  # .1234 / $-.1234$
    v = float(v)
    s = f"{abs(v):.{n}f}".replace("0.", ".", 1)
    return s if v >= 0 else f"$-{s}$"


def num(v, n, signed=False):
    v = float(v)
    s = f"{abs(v):.{n}f}"
    if v < 0:
        return "-" + s
    return ("+" + s) if signed else s


# ---- frame counts (Sec. III-B, IV-A)
for k, v in [("multirow_groups", "1,298"), ("earliest_rows", "7,846"), ("groups_missing_any_query", "134"),
             ("complete_but_reference_changed", "820"), ("complete_groups", "1,164"), ("eligible_groups", "344"),
             ("eligible_queries", "2,091"), ("split", "121"), ("split_with_some_unchanged", "119"),
             ("changed_rows_in_other_groups", "323")]:
    chk(f"{FR[k]:,}" == v, f"frame {k}={FR[k]} vs {v}")
    has(v)
chk(FR["instruction_changes"] == 0, "instructions unchanged")
chk(DG["split_at_first_replacement"] == 121, "split at 5fc977b")
chk(DG["split_groups_still_split_on_common_domain"] == 121, "split on common domain")

# ---- Table II (quality) and Table III (common domain)
for name, m in KEY.items():
    a, p = g("current_fixed", "all344", "AP10", m), g("current_fixed", "all344", "P10", m)
    has(f"{name} & {dot(a['old'],4)} & {dot(a['new'],4)} & {dot(a['delta'],4)} [{dot(a['ci_low'],3)}, {dot(a['ci_high'],3)}] & "
        f"{dot(p['old'],4)} & {dot(p['new'],4)}\\\\", f"Table II row {name}")
    c = g("common_fixed", "all344", "AP10", m)
    b = next(x for x in B if x["domain"] == "common_fixed" and x["stratum"] == "all344" and x["method"] == m and x["metric"] == "AP10")
    has(f"{name} & {dot(c['delta'],4)} & [{dot(c['ci_low'],3)}, {dot(c['ci_high'],3)}] & [{dot(b['delta_low'],4)}, {dot(b['delta_high'],4)}]\\\\",
        f"Table III row {name}")
    chk(float(b["delta_high"]) < 0, f"upper bound negative {name}")
    chk(c["groups"] == "271" and c["queries"] == "1636", "exact subset size")

# ---- orders, relative drops
for met, dom in (("AP10", "current_fixed"), ("P10", "current_fixed"), ("AP10", "common_fixed")):
    o = sorted(KEY.values(), key=lambda m: -float(g(dom, "all344", met, m)["old"]))
    n = sorted(KEY.values(), key=lambda m: -float(g(dom, "all344", met, m)["new"]))
    chk(o == n, f"{dom} {met} point order preserved")
ra = [float(g("current_fixed", "all344", "AP10", m)["new"]) / float(g("current_fixed", "all344", "AP10", m)["old"]) for m in KEY.values()]
rp = [float(g("current_fixed", "all344", "P10", m)["new"]) / float(g("current_fixed", "all344", "P10", m)["old"]) for m in KEY.values()]
chk((round((1 - max(ra)) * 100), round((1 - min(ra)) * 100)) == (24, 32), "AP 24-32%"); has("24--32\\%")
chk((round((1 - max(rp)) * 100), round((1 - min(rp)) * 100)) == (19, 20), "P 19-20%"); has("19--20\\%")
chk(f"{min(ra):.2f}--{max(ra):.2f}" == "0.68--0.76", "score ratios"); has("(0.68--0.76)")
dl = [-float(g("current_fixed", "all344", "AP10", m)["delta"]) for m in KEY.values()]
has(f"{min(dl):.3f}--{max(dl):.3f}")
cs = g("current_fixed", "all344", "AP10", KEY["MC2 Combined"]); sl = g("current_fixed", "all344", "AP10", KEY["MC2 SLERP"])
has(f"margin {float(cs['old'])-float(sl['old']):.4f} old, {float(cs['new'])-float(sl['new']):.4f} current")

# ---- margins
sig = lambda x: float(x["change_ci_low"]) > 0 or float(x["change_ci_high"]) < 0
for met, nshrink, nsig in (("AP10", 21, 19), ("P10", 20, 19)):
    cc = [x for x in C if x["domain"] == "current_fixed" and x["stratum"] == "all344" and x["metric"] == met]
    chk(len(cc) == 21 and all(x["point_sign_preserved"] == "True" for x in cc), met + " signs")
    chk(sum(abs(float(x["new_difference"])) < abs(float(x["old_difference"])) for x in cc) == nshrink, met + " shrink")
    chk(sum(map(sig, cc)) == nsig, met + " sig count")
    if met == "AP10":
        ns = {(x["a"], x["b"]) for x in cc if not sig(x)}
        chk(ns == {(KEY["S1"], KEY["S2"]), (KEY["MC2 Combined"], KEY["MC2 SLERP"])}, "non-significant AP pairs")
        med = st.median(abs(float(x["new_difference"])) / abs(float(x["old_difference"])) for x in cc)
        chk(f"{med:.2f}" == "0.75", "median 0.75"); has("to a median of 0.75 of their old size")
r = next(x for x in C if x["domain"] == "current_fixed" and x["stratum"] == "all344" and x["metric"] == "AP10"
         and x["a"] == KEY["S1"] and x["b"] == KEY["S1 reranked"])
adv_o, adv_n, chg = -float(r["old_difference"]), -float(r["new_difference"]), -float(r["difference_change"])
has(f"falls from {adv_o:.4f} (95\\% interval $[{-float(r['old_ci_high']):.3f}, {-float(r['old_ci_low']):.3f}]$)")
has(f"to {adv_n:.4f} ($[{-float(r['new_ci_high']):.3f}, {-float(r['new_ci_low']):.3f}]$)")
has(f"paired change is ${chg:.4f}$ ($[{-float(r['change_ci_high']):.4f}, {-float(r['change_ci_low']):.4f}]$)")
has(f"from {adv_o:.3f} to {adv_n:.3f} (paired change ${chg:.3f}$, 95\\% interval $[{-float(r['change_ci_high']):.3f},{-float(r['change_ci_low']):.3f}]$)", "abstract rerank")
has(f"{round((1-adv_n/adv_o)*100)}\\% relative reduction and a ratio of {adv_n/adv_o:.2f}")
rc = next(x for x in C if x["domain"] == "current_fixed" and x["stratum"] == "all344" and x["metric"] == "AP10"
          and x["a"] == KEY["S1"] and x["b"] == KEY["MC2 Combined"])
has(f"from {float(rc['old_difference']):.3f} to {float(rc['new_difference']):.3f}")
cells = FC["cells"]
has(f"$f_{{00}}={cells['f00']:.4f}$, $f_{{10}}={cells['f10']:.4f}$, $f_{{01}}={cells['f01']:.4f}$ and $f_{{11}}={cells['f11']:.4f}$")
has(f"$I={FC['interaction']:.4f}$")
chk(FC["output_diagnostics_mean"]["ordered_mismatch"] == 1.0, "all top-10 lists differ")
has(f"non-overlap {FC['output_diagnostics_mean']['set_nonoverlap']:.3f}")
u = DG["unique_denominator_gain_current_AP10"].values()
has(f"{min(u):.4f}--{max(u):.4f}")

# ---- ranges
for st_, frag in (("all344", "from {o} to {n} over all 344 groups ($\\Delta={d}$, $[{l},{h}]$)"),
                  ("split121", "from {o} to {n} in the 121 split groups (${d}$, $[{l}, {h}]$)"),
                  ("other223", "from {o} to {n} in the other 223 groups (${d}$, $[{l},{h}]$)")):
    x = g("current_fixed", st_, "AP_range", KEY["S1"])
    has(frag.format(o=num(x["old"], 3), n=num(x["new"], 3), d=num(x["delta"], 3, st_ == "split121"),
                    l=num(x["ci_low"], 3), h=num(x["ci_high"], 3)), f"S1 range {st_}")
x = g("current_fixed", "all344", "AP_range", KEY["MC2 Image-only"])
chk(float(x["old"]) == 0, "image-only old range 0")
has(f"becomes {float(x['new']):.3f} overall ($[{float(x['ci_low']):.3f}, {float(x['ci_high']):.3f}]$)")
x = g("current_fixed", "split121", "AP_range", KEY["MC2 Image-only"])
has(f"and {float(x['new']):.3f} in split groups ($[{float(x['ci_low']):.3f}, {float(x['ci_high']):.3f}]$)")
chk(float(g("current_fixed", "other223", "AP_range", KEY["MC2 Image-only"])["new"]) == 0, "image-only other stays 0")

# ---- DL19
dd = [-float(x["change"]) for x in D]
has(f"nDCG@10 by {min(dd):.3f}--{max(dd):.3f}")
p = next(x for x in DP if x["a"] == "mono-t5-3b" and x["b"] == "rank-zephyr")
has(f"(${float(p['change']):.3f}$, $[{float(p['ci_low']):.3f}, {float(p['ci_high']):.3f}]$)")
chk(float(p["ci_low"]) < 0 < float(p["ci_high"]), "DL19 margin CI crosses zero")
has(f"only by {-float(p['overlay_margin']):.3f}")
topics = json.loads((RES / "dl19_topics.json").read_text())
chk(len(topics["eligible"]) == 41, "41 DL19 topics"); has("41 topics remain")


# ---- real case (Sec. IV-A, Fig. 1b)
CA = json.loads((RES / "case.json").read_text())
ctx = CA["context"]
chk(CA["case_group"] == "g0039", "case group"); has("group \\texttt{g0039}")
has(f"among the {CA['n_eligible_groups']} split groups")
chk(CA["median_position_1based"] == (CA["n_eligible_groups"] - 1) // 2 + 1, "lower-median position")
has(f"the {CA['median_position_1based']}nd group"); chk(not CA["tie_at_median"], "no tie at median"); has("there is no tie at this position")
chk(CA["eligible_sorted"][CA["median_position_1based"] - 1]["group"] == CA["case_group"], "case is the lower-median group")
chk(all(q["instruction_unchanged"] for q in CA["queries"]), "instructions unchanged")
has("semantic equivalence within the group is not established")
qs = CA["queries"]
chg = [q for q in qs if q["positives_old"] != q["positives_current"]]
chk(len(qs) == 6 and len(chg) == 2 and all(q["instruction_unchanged"] for q in qs), "case: six requests, two replaced")
for q in chg:
    has(q["instruction"], f"case instruction {q['query_id']}")
    chk(q["top10_status"][3] == "old_only", "rank 4 old-only")
    kept = set(q["positives_old"]) & set(q["positives_current"]); new = set(q["positives_current"]) - set(q["positives_old"])
    chk(len(set(q["positives_old"])) == 5 and len(kept) == 1 and len(new) == 2, "kept one of five, two new IDs")
    chk(max(q["positives_current"].count(x) for x in new) == 3, "one new ID listed three times")
    si = q["scores"]["MetaCLIP2 Image-only"]
    chk(f"{si['AP10_old']:.2f}" == "0.09" and f"{si['AP10_new']:.2f}" == "0.02", "case image-only AP")
sr = sorted((f"{q['scores']['S1 reranked']['AP10_old']:.2f}", f"{q['scores']['S1 reranked']['AP10_new']:.2f}") for q in chg)
has(f"fall from {sr[1][0]} to {sr[1][1]} and from {sr[0][0]} to {sr[0][1]}")
has(f"group range rises from 0 to {CA['case_range_image_only']['new']:.2f}")
chk(CA["case_range_image_only"]["old"] == 0, "case old range 0")
chk(ctx["split_group_sizes"].get("6") == 116, "116 six-query groups"); has("116 contain six queries")
chk(ctx["distinct_current_sets_per_split_group"] == {"2": 121}, "two sets per split group")
chk(min(map(int, ctx["changed_rows_per_split_group"])) == 1 and max(map(int, ctx["changed_rows_per_split_group"])) == 6, "1-6 changed rows")
has(f"median Jaccard overlap of changed lists {ctx['median_jaccard_old_vs_current_changed_rows']:.2f}, "
    f"{ctx['changed_rows_with_zero_overlap']} of {ctx['changed_rows_in_split_groups']} with no overlap")
d = ctx["rows_with_duplicate_positive_ids"]
has(f"{d['current_release']:,} of the {ctx['rows_current_release']:,} current rows contain them, against none of the {ctx['rows_old_release']:,} old rows")
chk(d["old_release"] == 0, "no duplicates in old release")

# ---- robustness (Sec. IV-C, Table III)
RB = json.loads((RES / "robustness.json").read_text())
def sd(x, n=3):
    t = f"{abs(x):.{n}f}"; return ("$-$" if x < 0 else "") + t
for key, name in (("frame_2091", "Matched frame (344 groups)"), ("wider_non_frame", "Other unchanged queries"), ("wider_all", "All unchanged queries")):
    d = RB[key]; ch = d["adv_change"]; rc = d["rel_change"]; dv = d["deviation_from_proportional"]
    has(f"{name} & {d['queries']:,} & {d['components']} & {d['adv_old'][0]:.3f} & {d['adv_new'][0]:.3f} & {sd(ch[0])} [{sd(ch[1])}, {sd(ch[2])}] & "
        f"{100*d['rel_old'][0]:.1f} $\\rightarrow$ {100*d['rel_new'][0]:.1f} & {sd(100*rc[0],1)} [{sd(100*rc[1],1)}, {sd(100*rc[2],1)}] & "
        f"{sd(dv[0],4)} [{sd(dv[1],4)}, {sd(dv[2],4)}]\\\\", f"Table V row {key}")
fr_, nf, wa = RB["frame_2091"], RB["wider_non_frame"], RB["wider_all"]
chk(fr_["adv_change"][0] == fr_["adv_change"][0] and abs(fr_["adv_change"][0] - (-float(r["difference_change"]))) < 1e-12, "frame adv change equals main result")
chk(fr_["components"] == 312, "frame components")
has(f"falls from {100*fr_['rel_old'][0]:.1f}\\% to {100*fr_['rel_new'][0]:.1f}\\% (paired change ${100*fr_['rel_change'][0]:.1f}$ points, $[{100*fr_['rel_change'][1]:.1f}, {100*fr_['rel_change'][2]:.1f}]$)")
has(f"the current advantage would be {fr_['adv_expected_if_proportional'][0]:.3f}")
has(f"by ${fr_['deviation_from_proportional'][0]:.4f}$ ($[{fr_['deviation_from_proportional'][1]:.4f}, {fr_['deviation_from_proportional'][2]:.4f}]$)")
chk(fr_["rel_change"][2] < 0 and fr_["deviation_from_proportional"][2] < 0, "frame beyond-scaling intervals exclude 0")
chk(nf["rel_change"][1] < 0 < nf["rel_change"][2] and nf["deviation_from_proportional"][1] < 0 < nf["deviation_from_proportional"][2], "non-frame intervals include 0")
# (abstract no longer quotes the deviation; body checks above cover it)
sm = RB["sample"]
has(f"roughly halves across all {sm['wider_queries']:,} queries whose instructions and reference-image IDs are unchanged", "abstract wider")
chk(wa["adv_new"][0] / wa["adv_old"][0] < 0.55 and fr_["adv_new"][0] / fr_["adv_old"][0] < 0.55, "roughly halves in both populations")
has(f"{sm['wider_queries']:,} queries, the {sm['frame_queries']:,} frame queries plus {sm['non_frame_queries']:,} others ({sm['wider_changed_positive_set']:,} of the")
wr = [wa[f"AP10_new|{x}"][0] / wa[f"AP10_old|{x}"][0] for x in RUN_ORDER]
has(f"loses {round((1-max(wr))*100)}--{round((1-min(wr))*100)}\\% of its AP@10")
for k in (("AP10_old|",), ("AP10_new|",)):
    pass
oo = sorted(RUN_ORDER, key=lambda x: -wa[f"AP10_old|{x}"][0]); nn = sorted(RUN_ORDER, key=lambda x: -wa[f"AP10_new|{x}"][0])
chk(oo == nn, "wider point order preserved")
chk(all(wa[f"AP10_delta|{x}"][2] < 0 for x in RUN_ORDER), "wider all deltas negative")
has(f"overall ({wa['adv_old'][0]:.3f} to {wa['adv_new'][0]:.3f}) and outside the frame ({nf['adv_old'][0]:.3f} to {nf['adv_new'][0]:.3f})")
has(f"barely moves ({100*nf['rel_old'][0]:.1f}\\% to {100*nf['rel_new'][0]:.1f}\\%, change ${100*nf['rel_change'][0]:.1f}$ points, $[{100*nf['rel_change'][1]:.1f}, {100*nf['rel_change'][2]:.1f}]$)")
has(f"proportional scaling is ${nf['deviation_from_proportional'][0]:.4f}$ ($[{nf['deviation_from_proportional'][1]:.4f}, {nf['deviation_from_proportional'][2]:.4f}]$)")
cf = RB["concentration_frame"]
has(f"Of {cf['components']} components, {cf['components_with_zero_contribution']} contribute nothing")
has(f"account for {round(100*cf['top5_share_of_abs_contribution'])}\\% of the total")
chk(cf["loo_all_adv_change_negative"] and cf["loo_all_deviation_negative"], "LOO signs stable")
has(f"between ${cf['loo_adv_change_min']:.3f}$ and ${cf['loo_adv_change_max']:.3f}$")
has(f"between ${cf['loo_deviation_min']:.4f}$ and ${cf['loo_deviation_max']:.4f}$")
has(f"together gives ${cf['without_top5_adv_change']:.3f}$")

# ---- evaluator sensitivity (Sec. IV-E, Table IV) and negatives
EV = json.loads((RES / "evaluator_sensitivity.json").read_text())
chk(EV["replication"]["available"] and EV["replication"]["max_abs_diff_at_4dp"] == 0 and EV["replication"]["cells_checked"] == 21, "metrics.csv replicated exactly")
has(f"all {EV['replication']['cells_checked']} values")
has(f"to all {EV['rows']:,} rows whose instruction"); has(f"({EV['groups']} evaluator groups, {EV['components']} bootstrap components)")
SH = {"S1 reranked": "S1 reranked", "S1": "S1", "S2": "S2", "MetaCLIP2 Combined": "MC2 Combined", "MetaCLIP2 SLERP": "MC2 SLERP",
      "MetaCLIP2 Text-only": "MC2 Text-only", "MetaCLIP2 Image-only": "MC2 Image-only"}
def sd4(x, k=4):
    t = f"{abs(x):.{k}f}".replace("0.", ".", 1); return ("$-$" if x < 0 else ("+" if x > 0 else "")) + t
o, n = EV["sensitivity_order"]["P10_old"], EV["sensitivity_order"]["P10_new"]
for r_, sh in SH.items():
    x = EV["runs"][r_]["P10"]
    has(f"{sh} & {x['old'][0]:.4f}".replace("0.", ".", 1) + f" & {x['new'][0]:.4f}".replace("0.", ".", 1) +
        f" & {sd4(x['change'][0])} [{sd4(x['change'][1])}, {sd4(x['change'][2])}] & {o.index(r_)+1} $\\rightarrow$ {n.index(r_)+1}\\\\", f"Table IV row {sh}")
tau = EV["kendall_tau_old_vs_new"]
has(f"Kendall $\\tau={tau['P10']:.2f}$"); has(f"$\\tau={tau['AP10']:.2f}$ for the AP-based range")
tx = EV["runs"]["MetaCLIP2 Text-only"]["P10"]; im = EV["runs"]["MetaCLIP2 Image-only"]["P10"]
chk(o[0] == "MetaCLIP2 Text-only" and n.index("MetaCLIP2 Text-only") == 5, "Text-only most -> second least")
has(f"({tx['old'][0]:.3f}) to the second least sensitive ({tx['new'][0]:.3f}; change ${tx['change'][0]:.3f}$, $[{tx['change'][1]:.3f}, {tx['change'][2]:.3f}]$)")
chk(im["old"][0] == 0, "image-only evaluator sensitivity 0 old")
has(f"{im['new'][0]:.4f} ($[{im['new'][1]:.4f}, {im['new'][2]:.4f}]$) under current labels")
fl = [f for f in EV["sign_flips"] if f["metric"] == "P10"]; st_ = [f for f in fl if f["both_intervals_exclude_zero"]]
has(f"Of the {len(fl)} run pairs whose sensitivity difference changes sign, one has intervals")
chk(len(st_) == 1 and {st_[0]["a"], st_[0]["b"]} == {"MetaCLIP2 Combined", "MetaCLIP2 Text-only"}, "one robust flip")
NG = DG["negatives_projected"]
has(f"{NG['rows_changed']} of {NG['rows']:,} rows, and {NG['groups_sharing_old']-NG['groups_sharing_current']} of 344 groups no longer share one negative set")
chk(NG["groups_sharing_old"] == 344, "old: all groups share negatives")
has(f"and {NG['groups_sharing_current']} in the current one")

print(f"checked {checked} items; errors: {errs}")
raise SystemExit(1 if errs else 0)
