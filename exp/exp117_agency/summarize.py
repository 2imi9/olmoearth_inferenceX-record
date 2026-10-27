"""exp117's summary: every number the record quotes, read from the committed outputs (standard library only).

    python exp/exp117_agency/summarize.py > exp/out/exp117_summary.json

Run on the cluster (the record's rule: nothing runs on the laptop). It reads exp/out/exp117_*.json(l) and
exp/out/exp117_e2e/, computes nothing new beyond counts, medians and ranges, and writes one JSON."""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "exp", "out")


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


def span(values):
    v = [x for x in values if x is not None]
    return {"n": len(v), "min": min(v), "median": statistics.median(v), "max": max(v)} if v else {"n": 0}


S = {"sources": "exp/out/exp117_*; aicr jobs 1523908 (sizing), 1525456 (judge), 1531551 (box), 1534186 (grade end "
                "to end, 1.18.0 at 5d8b434; 1533266 at 0d9de25 printed the same numbers)"}

# 1. the chosen engine's exact self-tests (betting_widths.py, job 1525456)
st = load("exp117_betting_selftest.json")["selftest"]
S["selftest"] = {
    "dynamic_programme_vs_every_order_max_abs_difference": st["dynamic_programme_vs_every_order_max_abs_difference"],
    "largest_expected_evalue_without_replacement": st["largest_expected_evalue"]["without_replacement"],
    "largest_expected_evalue_point_frame_deflated": st["largest_expected_evalue"]["with_replacement_deflated"],
    "largest_expected_evalue_point_frame_plain": st["largest_expected_evalue"]["with_replacement_plain_NEGATIVE_CONTROL"],
    "linear_cells": len(st["linear_min_coverage"]),
    "linear_cells_holding": sum(bool(c["holds"]) for c in st["linear_min_coverage"].values()),
    "ratio_cells": len(st["ratio_min_coverage"]),
    "ratio_cells_holding": sum(bool(c["holds"]) for c in st["ratio_min_coverage"].values()),
    "linear_normal_min_coverage": min(c["normal_theory_nominal"] for c in st["linear_min_coverage"].values()),
}

# 2. the judge's exact enumerations (job 1525456): every form claimed valid at or above 1 - alpha in every cell
en = load("exp117_judge_enum.json")
cells, worst_slack, normal_min = [], None, None
for a in ("alpha_0.1", "alpha_0.3"):
    alpha = float(a.split("_")[1])
    for name, c in en[a].items():
        if not isinstance(c, dict) or "holds" not in c:
            continue
        claimed = {k: v for k, v in c.items() if isinstance(v, dict) and "min_coverage" in v
                   and "NOT_CLAIMED" not in k and "NOMINAL" not in k and "NOT_VALID" not in k}
        slack = min(v["min_coverage"] - (1 - alpha) for v in claimed.values()) if claimed else None
        nom = [v["min_coverage"] for k, v in c.items() if isinstance(v, dict) and "NOMINAL" in k]
        cells.append({"alpha": alpha, "cell": name, "holds": bool(c["holds"]), "claimed_forms": sorted(claimed),
                      "least_slack": slack, "normal_min_coverage": min(nom) if nom else None})
        if slack is not None:
            worst_slack = slack if worst_slack is None else min(worst_slack, slack)
        if nom:
            normal_min = min(nom) if normal_min is None else min(normal_min, min(nom))
S["judge_enumeration"] = {"cells": len(cells), "cells_holding": sum(c["holds"] for c in cells),
                          "least_slack_over_claimed_forms": worst_slack, "normal_min_coverage": normal_min,
                          "mm_engine_holds": bool(en["mm_engine"]["holds"]), "by_cell": cells}

# 3. the simulation at the agency designs' shapes (box_run.sh, job 1531551)
mc = load("exp117_box_mc.json")
cases = {k: v for k, v in mc.items() if isinstance(v, dict) and "holds" in v}
S["simulation"] = {
    "draws": mc["draws"], "gate": mc["gate"], "cases": len(cases), "cases_holding": sum(bool(v["holds"]) for v in cases.values()),
    "valid_coverage": span([v[m]["coverage"] for v in cases.values() for m in ("betting", "box") if m in v]),
    "normal_coverage": span([v["normal"]["coverage"] for v in cases.values() if "normal" in v]),
    "errors": sum(v["errors"] for v in cases.values()),
}

# 4. widths in the design study: betting against the fallback (boxed stratified Chernoff), the box against the scan
rows = [json.loads(line) for line in open(os.path.join(OUT, "exp117_judge_widths.jsonl"), encoding="utf-8")]
lin = []
for r in rows:
    q = r["result"]
    if q.get("family") == "linear" and q.get("betting_over_bsc") is not None:
        lin.append({"key": "/".join(r["key"]), "betting_over_bsc": q["betting_over_bsc"],
                    "betting_over_normal": q["x_normal"]["betting"]})
box = load("exp117_box_widths.json")
rat = []
for r in rows:
    if r["key"][1:2] != ["per_class"]:
        continue
    c, q = r["key"][2], r["result"]
    for what in ("user", "producer"):
        b = box.get(f"nlcd2016_B_2016/{what}/{c}")
        j = q.get(what, {})
        if not b or "box" not in b or not isinstance(j.get("bsc"), list):
            continue
        wb = b["box"][1] - b["box"][0]
        rat.append({"key": f"nlcd/{what}/{c}", "box_points": 100 * wb, "bsc_points": 100 * (j["bsc"][1] - j["bsc"][0]),
                    "scan_points": 100 * (j["betting"][1] - j["betting"][0]), "normal_points": 100 * (b["normal"][1] - b["normal"][0])})
S["design_study_widths"] = {
    "items_completed_before_the_time_limit": len(rows),
    "linear_betting_over_fallback": span([x["betting_over_bsc"] for x in lin]),
    "linear_by_item": lin,
    "ratio_box_over_fallback": span([x["box_points"] / x["bsc_points"] for x in rat if x["bsc_points"] > 0]),
    "ratio_box_over_scan": span([x["box_points"] / x["scan_points"] for x in rat if x["scan_points"] > 0]),
    "ratio_by_item": rat,
}
sel = load("exp117_judge_select.json")
S["selection_rule"] = {"rule": sel["rule"], "gates": sel["gates"], "verdict": sel["verdict"],
                       "shipped": {"linear": "betting", "ratio": "box of two betting intervals (not the scan the "
                                   "judge measured); user's accuracy by the pure-stratum rule where one pure "
                                   "stratum is declared"}}

# 5. the shipped command end to end (job 1534186)
cmp_ = load("exp117_e2e/compare.json")
E = os.path.join("exp117_e2e")
S["end_to_end"] = {
    "worst_abs_difference_to_the_design_study": cmp_["worst_abs_difference_to_judge"],
    "overall": [{k: r[k] for k in ("run", "map", "estimate", "interval", "normal", "width_over_normal", "published",
                                   "estimate_minus_published_points")} for r in cmp_["overall"]],
    "differences": [{k: r[k] for k in ("run", "difference", "estimate", "interval", "normal", "width_over_normal")}
                    for r in cmp_["differences"]],
    "width_over_normal_overall_and_differences": span([r["width_over_normal"] for r in cmp_["overall"] + cmp_["differences"]]),
}
alt = load(os.path.join(E, "nlcd2016_l2_alt.json"))["per_class"]["nlcd"]
pure_vs_box, pub = [], []
for r in cmp_["per_class"]:
    if r["run"] != "nlcd2016_l2_alt":
        continue
    b = box.get(f"nlcd2016_B_2016/user/{r['class']}")
    if r["user_method"] == "pure-stratum rule" and b and "box" in b:
        pure_vs_box.append({"class": r["class"], "pure_points": 100 * (r["user_interval"][1] - r["user_interval"][0]),
                            "box_points": 100 * (b["box"][1] - b["box"][0]),
                            "normal_points": 100 * (r["user_normal"][1] - r["user_normal"][0])})
    if "minus_published_points" in r:
        pub.append({"class": r["class"], "user_minus_published": r["minus_published_points"][0],
                    "producer_minus_published": r["minus_published_points"][1]})
S["end_to_end"]["pure_stratum_rule_vs_box"] = {
    "classes": len(pure_vs_box), "pure_narrower": sum(x["pure_points"] < x["box_points"] for x in pure_vs_box),
    "pure_over_box": span([x["pure_points"] / x["box_points"] for x in pure_vs_box]),
    "pure_over_normal": span([x["pure_points"] / x["normal_points"] for x in pure_vs_box]), "by_class": pure_vs_box}
S["end_to_end"]["nlcd_per_class_against_table4_as_transcribed"] = {
    "classes": len(pub), "largest_abs_user_points": max(abs(x["user_minus_published"]) for x in pub),
    "largest_abs_producer_points": max(abs(x["producer_minus_published"]) for x in pub),
    "within_1_5_points_both": sum(abs(x["user_minus_published"]) <= 1.5 and abs(x["producer_minus_published"]) <= 1.5
                                  for x in pub),
    "by_class": pub,
    "caveat": "Table 4 as transcribed in practitioner_sbui.py; read again from the PDF it is right, and the gap is "
               "the table's unstated rule (exp117_table4.json)"}
smallest = min(alt.items(), key=lambda kv: kv[1]["reference_share"]["estimate"])
S["end_to_end"]["smallest_class_producer"] = {"class": smallest[0], "share": smallest[1]["reference_share"]["estimate"],
                                              "producer_interval": smallest[1]["producer"]["interval"]}
json.dump(S, sys.stdout, indent=1)
sys.stdout.write("\n")
