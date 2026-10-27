"""exp118's summary: every number the record quotes, read from exp/out/exp118_second_route.json and
exp/out/exp118_coverage_B1.json (standard library only). Run on the cluster.

    python exp/exp118_summary.py > exp/out/exp118_summary.json"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out")


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


A, B = load("exp118_second_route.json"), load("exp118_coverage_B1.json")
S = {"plan": "docs/plan/grade_second_route_and_coverage.md (frozen 000bdf7)",
     "jobs": {"second_route": 1541331, "coverage_B1": 1541238, "coverage_B2": "1541240, cancelled after 7 minutes, "
              "before any result was read"}}

bars = {"linear": 1e-9, "difference": 1e-9, "box": 1e-9, "pure-stratum rule": 1e-6, "normal theory": 1e-9}
fam = {}
for k, v in A["max_by_family"].items():
    if isinstance(v, dict) and "max_abs_diff" in v:
        fam[k] = {"largest_abs_difference": v["max_abs_diff"], "where": v.get("where")}
S["second_route"] = {
    "by_family": fam,
    "P1_holds": all(fam[k]["largest_abs_difference"] <= b for k, b in bars.items() if k in fam)
    and set(bars) <= set(fam),
    "selftest": {k: A["selftest"][k] for k in ("order_average_vs_enumeration_max_abs_diff",
                                               "hypergeometric_tail_vs_exact_max_abs_diff",
                                               "hypergeometric_intervals_differing_from_exact")},
    "runs": sorted(A["runs"])}

Q = B["quantities"]
kind = lambda k: k.split("/")[1]
below = sorted(((k, q["normal_coverage"]) for k, q in Q.items() if q["normal_below_bar"]), key=lambda t: t[1])
least_valid = min(Q.items(), key=lambda kv: kv[1]["valid_coverage"])
least_normal = min(Q.items(), key=lambda kv: kv[1]["normal_coverage"])
ratio = {}
for k, q in Q.items():
    if q["normal_mean_width"] > 0:
        ratio.setdefault(kind(k), []).append(q["valid_mean_width"] / q["normal_mean_width"])
S["coverage_B1"] = {
    "package": B["package"], "draws": B["draws_per_block"], "quantities": len(Q), "errors": B["errors"],
    "bar": next(iter(Q.values()))["bar"],
    "P2_holds": B["P2_every_valid_interval_holds"],
    "valid_least": {"quantity": least_valid[0], "coverage": least_valid[1]["valid_coverage"]},
    "P3_holds": B["P3_normal_below_bar_somewhere"],
    "normal_below_bar": len(below), "normal_below_bar_quantities": below,
    "normal_least": {"quantity": least_normal[0], "coverage": least_normal[1]["normal_coverage"]},
    "normal_by_kind": {k: {"n": len([q for kk, q in Q.items() if kind(kk) == k]),
                           "below_bar": len([1 for kk, _ in below if kind(kk) == k]),
                           "least": min(q["normal_coverage"] for kk, q in Q.items() if kind(kk) == k),
                           "median": statistics.median(q["normal_coverage"] for kk, q in Q.items() if kind(kk) == k)}
                       for k in sorted({kind(kk) for kk in Q})},
    "overall_and_difference": {k: {"valid": q["valid_coverage"], "normal": q["normal_coverage"]}
                               for k, q in Q.items() if kind(k) in ("overall", "difference")},
    "width_over_normal_by_kind": {k: {"n": len(v), "median": statistics.median(v), "min": min(v), "max": max(v)}
                                  for k, v in sorted(ratio.items())}}
json.dump(S, sys.stdout, indent=1)
sys.stdout.write("\n")
