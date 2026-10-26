#!/bin/bash
#SBATCH -J oeix-agency-sizing
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:30:00
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
export UV_CACHE_DIR=/scratch/qi_zim_neu/olmoearth_inferenceX/uv-cache
export PYTHONUNBUFFERED=1
D=/scratch/qi_zim_neu/olmoearth_inferenceX/agency/$SLURM_JOB_ID
mkdir -p "$D" || exit 1
cd "$D" || exit 1
git clone -q --depth 1 https://github.com/2imi9/olmoearth_inferenceX.git pkg || exit 1
git clone -q --depth 1 https://github.com/2imi9/olmoearth_inferenceX-record.git rec || exit 1
cat > strata_size.csv <<'CSV'
Strata,strata_ha
1001,18199738
1002,20978746
1003,222478929
1004,1019336791
1006,177268821
1008,27239648
1014,1368465
1015,8616809
1101,46741948
1102,91221495
1103,765162954
1104,206951698
1106,310518525
1114,6217819
1200,4899964
1210,22186007
1221,7224973
1222,27159363
1223,2512686
2001,248837
2002,412896
2003,12131857
2004,48469650
2006,210295297
2008,33488325
2014,42731903
2015,10360753
2025,330065716
2027,511252597
2029,194923270
2030,8795866
2101,2454939
2102,6763167
2103,61462759
2104,19071612
2106,47291256
2114,148930873
2125,154989894
2200,3684946
2210,258777480
2221,10410148
2222,100177481
2223,20090541
2225,51433694
4001,88495150
4002,3881870
4003,9312149
4004,278719706
4006,168962575
4008,54402776
4014,6543421
4015,401298
4025,15995277
4027,14113270
4029,25657290
4101,147213670
4102,71031530
4103,291283066
4104,447407601
4106,393994366
4114,302323312
4125,180161248
4200,41426256
4210,58305508
4221,36700717
4222,45744409
4223,4215300
4224,58379099
4226,12752897
5004,32453
5006,4634460
5008,53256056
5014,505189
5015,103229504
5025,3559792
5027,84437699
5029,12000744
5104,658361
5106,15920035
5114,24483528
5125,227547165
5200,5094508
5210,24379285
5221,13759882
5222,12365144
5223,3431287
5225,3678405
5226,2436873
5227,221966
6004,83234378
6006,178654915
6014,8068
6025,221229995
6027,862483136
6029,81824612
6030,1514071
6040,151154050
6104,437580
6106,19093034
6114,6178122
6125,259879639
6200,1666662
6210,38417027
6221,4052021
6222,67418645
6223,78292697
6224,57716552
6226,2808326
6227,20069962
7001,69667
7002,737654
7003,61845927
7004,351988233
7006,230518975
7008,21013743
7014,40039936
7015,54784644
7101,5244463
7102,987250
7103,1147552
7104,127951
7106,1313840
7114,7137182
7200,418344
7210,12572324
7222,2990168
7223,598730
7225,10339445
8001,216602905
8002,238809747
8003,357151992
8004,83827636
8006,106394804
8008,13361828
8014,78752669
8015,26720535
8029,5966862
8101,65281124
8102,72740572
8103,256411118
8104,30497009
8106,73915576
8114,139262097
8200,3113906
8210,33249600
8221,4898964
8222,19457237
8223,14318643
8226,3555721
CSV
cat > sizing.py <<'PY'
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
PY
cd pkg || exit 1
~/.local/bin/uv sync -q --frozen --python 3.12 || exit 1
~/.local/bin/uv run --frozen --no-sync python ../sizing.py "$D/rec" "$D/strata_size.csv" > "$D/sizing.json" || exit 1
echo "SIZING_OK $D/sizing.json"
cat "$D/sizing.json"
