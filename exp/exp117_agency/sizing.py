"""Interval widths an agency sample gives, for the design of grading a map with one (run on aicr, not locally).

Overall accuracy and two-map differences under a stratified sample with strata that are not the map's classes:
Stehman's normal-theory interval against a valid union bound over strata (each stratum's exact interval at
1 - alpha_h, sum alpha_h = 0.05, alpha_h equal or proportional to the stratum's weight), and NLCD's per-class user's
accuracy with a union bound over the strata's two cell counts. Writes JSON to stdout."""
import csv, json, math, sys
from collections import defaultdict
import numpy as np
from oe_inferencex.estimate import hypergeom_interval

REC, AREAS = sys.argv[1], sys.argv[2]
Z = 1.959963984540054


def strat(rows, ok_key, size):
    """rows: dicts with 'h' and a 0/1 value under ok_key. size: {h: N_h}. Stehman normal theory and union bounds."""
    by = defaultdict(list)
    for r in rows:
        by[r["h"]].append(r[ok_key])
    hs = sorted(by)
    N = sum(size[h] for h in hs)
    W = {h: size[h] / N for h in hs}
    est = sum(W[h] * np.mean(by[h]) for h in hs)
    var = sum(W[h] ** 2 * (1 - len(by[h]) / size[h]) * np.var(by[h], ddof=1) / len(by[h]) for h in hs if len(by[h]) > 1)
    out = {"strata": len(hs), "n": sum(len(v) for v in by.values()), "estimate": est,
           "normal": [est - Z * math.sqrt(var), est + Z * math.sqrt(var)]}
    for name, alpha in (("union_equal", {h: 0.05 / len(hs) for h in hs}),
                        ("union_by_weight", {h: 0.05 * W[h] for h in hs})):
        lo = hi = 0.0
        for h in hs:
            k, n = int(sum(by[h])), len(by[h])
            Nh = max(int(round(size[h])), n)
            l, u = hypergeom_interval(k, n, Nh, conf=1 - alpha[h])
            lo += W[h] * l
            hi += W[h] * u
        out[name] = [lo, hi]
    for k in ("normal", "union_equal", "union_by_weight"):
        out[k + "_width"] = out[k][1] - out[k][0]
    return out


def diff(rows, a, b, size):
    """a - b accuracy difference: per stratum the two discordant counts' exact intervals at 1 - alpha/(2H)."""
    by = defaultdict(list)
    for r in rows:
        by[r["h"]].append((r[a], r[b]))
    hs = sorted(by)
    N = sum(size[h] for h in hs)
    W = {h: size[h] / N for h in hs}
    est = sum(W[h] * np.mean([x - y for x, y in by[h]]) for h in hs)
    var = sum(W[h] ** 2 * (1 - len(by[h]) / size[h]) * np.var([x - y for x, y in by[h]], ddof=1) / len(by[h])
              for h in hs if len(by[h]) > 1)
    lo = hi = 0.0
    for h in hs:
        n = len(by[h])
        Nh = max(int(round(size[h])), n)
        k10 = sum(1 for x, y in by[h] if x and not y)
        k01 = sum(1 for x, y in by[h] if y and not x)
        c = 1 - 0.05 / (2 * len(hs))
        l10, u10 = hypergeom_interval(k10, n, Nh, conf=c)
        l01, u01 = hypergeom_interval(k01, n, Nh, conf=c)
        lo += W[h] * (l10 - u01)
        hi += W[h] * (u10 - l01)
    return {"estimate": est, "normal": [est - Z * math.sqrt(var), est + Z * math.sqrt(var)],
            "normal_width": 2 * Z * math.sqrt(var), "union": [lo, hi], "union_width": hi - lo}


res = {}
# GFC2020: 149 strata, areas in ha; a 10 m pixel is 0.01 ha
area = {int(r["Strata"]): float(r["strata_ha"]) for r in csv.DictReader(open(AREAS))}
g = [r for r in csv.DictReader(open(f"{REC}/exp/out/exp94_gfc_points.csv"))]
rows = []
for r in g:
    h = int(r["stratum"])
    if h not in area:
        continue
    ref = int(r["ref_forest"])
    rows.append({"h": h, "gfc": int(int(r["gfc_v2"]) == ref), "wc": int(int(r["worldcover_forest"]) == ref)})
size_g = {h: area[h] / 0.01 for h in area}
res["gfc2020_gfc_v2_overall"] = strat(rows, "gfc", size_g)
res["gfc2020_worldcover_overall"] = strat(rows, "wc", size_g)
res["gfc2020_gfc_minus_worldcover"] = diff(rows, "gfc", "wc", size_g)

# NLCD 2016 AA (set B), year 2016: weights = N_h / n_h
n = [r for r in csv.DictReader(open(f"{REC}/exp/out/exp94_nlcd_points.csv")) if r["set"] == "B" and r["year"] == "2016"]
size_n = defaultdict(float)
for r in n:
    size_n[int(r["stratum"])] += float(r["weight"])
lvl1 = lambda c: int(c) // 10                       # Anderson level I
rows = [{"h": int(r["stratum"]), "l2": int(int(r["nlcd"]) == int(r["ref_nlcd"])),
         "l1": int(lvl1(r["nlcd"]) == lvl1(r["ref_nlcd"])), "m": int(r["nlcd"]), "r": int(r["ref_nlcd"])} for r in n]
res["nlcd2016_level2_overall"] = strat(rows, "l2", size_n)
res["nlcd2016_level1_overall"] = strat(rows, "l1", size_n)
# per-class user's accuracy, level II, a union bound over every stratum's two counts (map=c & ref=c, map=c & ref!=c)
hs = sorted(size_n)
classes = sorted({x["m"] for x in rows})
ua = {}
for c in classes:
    A_lo = A_hi = B_lo = B_hi = 0.0
    cnt = 0
    num = den = 0.0
    var_ok = True
    for h in hs:
        rh = [x for x in rows if x["h"] == h]
        nh = len(rh)
        Nh = max(int(round(size_n[h])), nh)
        a = sum(1 for x in rh if x["m"] == c and x["r"] == c)
        b = sum(1 for x in rh if x["m"] == c and x["r"] != c)
        conf = 1 - 0.05 / (2 * len(hs))
        la, ua_ = hypergeom_interval(a, nh, Nh, conf=conf)
        lb, ub = hypergeom_interval(b, nh, Nh, conf=conf)
        A_lo += Nh * la; A_hi += Nh * ua_; B_lo += Nh * lb; B_hi += Nh * ub
        num += Nh * a / nh; den += Nh * (a + b) / nh
        cnt += a + b
    ua[str(c)] = {"n_map_class": cnt, "estimate": num / den if den else None,
                  "union": [A_lo / (A_lo + B_hi) if A_lo + B_hi else 0.0, A_hi / (A_hi + B_lo) if A_hi + B_lo else 1.0]}
    ua[str(c)]["union_width"] = ua[str(c)]["union"][1] - ua[str(c)]["union"][0]
res["nlcd2016_user_accuracy_level2"] = ua
json.dump(res, sys.stdout, indent=1, default=float)
