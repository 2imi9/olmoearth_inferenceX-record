"""exp117: NLCD 2016's per-class table (Wickham et al. 2021, Table 4) against the record's points (standard library).

    python exp/exp117_agency/table4.py > exp/out/exp117_table4.json

The published values below were read from the paper's PDF (page 4, Table 4, rendered on the cluster): level II, the
2016 map, user's (UA) and producer's (PA) accuracy by the primary reference label, and UAa, PAa "when agreement
included the alternate reference label"; overall accuracy 72.1% (primary), 86.9% in the caption and 86.4% in the text
(primary or alternate). The points are exp94's set B, whose labels equal the agency's own reference file (checked on
26 October 2026, aicr job 1540024: the same (stratum, map, primary, alternate) for all 4,629 points). Each point's 2011
primary label comes from its 2011 row (the same pid). Estimators: the paper's equations 1 and 4 (stratified, combined
ratio), stratum sizes the rounded sums of the weights. With exp/out/exp117_e2e/nlcd2016_l2_alt_p11.json present (the
package's `grade --alternate alt p11`), its user's accuracies are compared with this script's own."""
import csv
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "exp", "out")
CLASSES = [11, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]
PUB = {"UA": [92, 72, 43, 47, 63, 58, 82, 85, 52, 68, 65, 64, 89, 45, 54],
       "UAa": [99, 88, 74, 77, 75, 77, 90, 93, 75, 87, 91, 77, 93, 61, 69],
       "PA": [94, 42, 43, 47, 49, 63, 67, 75, 55, 81, 66, 54, 91, 86, 62],
       "PAa": [95, 61, 70, 84, 78, 77, 79, 85, 78, 95, 91, 76, 94, 96, 80]}
PUB = {k: dict(zip(CLASSES, v)) for k, v in PUB.items()}
PUB_OA = {"primary": 72.1, "alternate_caption": 86.9, "alternate_text": 86.4}


def load():
    rows = list(csv.DictReader(open(os.path.join(OUT, "exp94_nlcd_points.csv"), newline="", encoding="utf-8")))
    c = lambda r, k: int(float(r[k]))
    p11 = {r["pid"]: c(r, "ref_nlcd") for r in rows if r["set"] == "B" and r["year"] == "2011"}
    a11 = {r["pid"]: c(r, "ref_nlcd_alternate") for r in rows if r["set"] == "B" and r["year"] == "2011"}
    pts = [{"h": r["stratum"], "w": float(r["weight"]), "m": c(r, "nlcd"), "p": c(r, "ref_nlcd"),
            "a": c(r, "ref_nlcd_alternate"), "p11": p11[r["pid"]], "a11": a11[r["pid"]]}
           for r in rows if r["set"] == "B" and r["year"] == "2016"]
    wsum, n = defaultdict(float), defaultdict(int)
    for q in pts:
        wsum[q["h"]] += q["w"]
        n[q["h"]] += 1
    return pts, {h: round(v) for h, v in wsum.items()}, n


def ratio(pts, N, n, num, den):
    s_num, s_den = defaultdict(float), defaultdict(float)
    for q in pts:
        s_num[q["h"]] += num(q)
        s_den[q["h"]] += den(q)
    d = sum(N[h] * s_den[h] / n[h] for h in N)
    return 100 * sum(N[h] * s_num[h] / n[h] for h in N) / d


def hits(vals, pub):
    return {"matches_after_rounding": sum(round(vals[c]) == pub[c] for c in CLASSES),
            "largest_gap_points": max(abs(vals[c] - pub[c]) for c in CLASSES),
            "by_class": {str(c): [round(vals[c], 3), pub[c]] for c in CLASSES}}


def main():
    pts, N, n = load()
    one = lambda q: 1
    out = {"points": len(pts), "strata": len(N), "published": {k: {str(c): v[c] for c in CLASSES} for k, v in PUB.items()},
           "published_overall": PUB_OA}
    prim = lambda q: q["m"] == q["p"]
    out["primary"] = {"overall": ratio(pts, N, n, prim, one),
                      "user": hits({c: ratio(pts, N, n, lambda q, c=c: q["m"] == c and q["p"] == c,
                                             lambda q, c=c: q["m"] == c) for c in CLASSES}, PUB["UA"]),
                      "producer": hits({c: ratio(pts, N, n, lambda q, c=c: q["m"] == c and q["p"] == c,
                                                 lambda q, c=c: q["p"] == c) for c in CLASSES}, PUB["PA"])}
    rules = {"text: primary or alternate": ("a",),
             "primary, alternate or 2011 primary": ("a", "p11"),
             "primary, alternate or 2011 alternate": ("a", "a11"),
             "primary, alternate, 2011 primary or 2011 alternate": ("a", "p11", "a11")}
    out["rules"] = {}
    for name, extra in rules.items():
        ok = lambda q, extra=extra: q["m"] == q["p"] or any(q["m"] == q[k] for k in extra)
        ref = lambda q, ok=ok: q["m"] if ok(q) and q["m"] != q["p"] else q["p"]
        pa_defs = {
            "reference class the map's where an alternate matches, else the primary": (
                lambda q, c: q["m"] == c and ref(q) == c, lambda q, c: ref(q) == c),
            "primary class, right by the rule": (lambda q, c: q["p"] == c and ok(q), lambda q, c: q["p"] == c),
        }
        out["rules"][name] = {
            "overall": ratio(pts, N, n, ok, one),
            "user": hits({c: ratio(pts, N, n, lambda q, c=c: q["m"] == c and ok(q), lambda q, c=c: q["m"] == c)
                          for c in CLASSES}, PUB["UAa"]),
            "producer": {k: hits({c: ratio(pts, N, n, lambda q, c=c, f=f: f(q, c), lambda q, c=c, g=g: g(q, c))
                                  for c in CLASSES}, PUB["PAa"]) for k, (f, g) in pa_defs.items()}}
    path = os.path.join(OUT, "exp117_e2e", "nlcd2016_l2_alt_p11.json")
    if os.path.exists(path):
        g = json.load(open(path, encoding="utf-8"))["per_class"]["nlcd"]
        mine = out["rules"]["primary, alternate or 2011 primary"]["user"]["by_class"]
        out["grade_alternate_alt_p11"] = {
            "user_estimates_percent": {str(c): 100 * g[str(c)]["user"]["estimate"] for c in CLASSES},
            "largest_difference_to_this_script": max(abs(100 * g[str(c)]["user"]["estimate"] - mine[str(c)][0])
                                                     for c in CLASSES),
            "matches_table4_after_rounding": sum(round(100 * g[str(c)]["user"]["estimate"]) == PUB["UAa"][c]
                                                 for c in CLASSES)}
    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
