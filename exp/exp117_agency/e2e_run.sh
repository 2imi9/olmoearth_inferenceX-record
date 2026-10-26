#!/bin/bash
#SBATCH -J oeix-grade-e2e
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
# `oe-inferencex grade` end to end on NLCD 2016 AA (set B) and GFC2020, as a user runs it, then every number set beside
# the agency's published one and beside the judge's (aicr jobs 1525456 and 1531551, written separately from the package).
set -u
SHA=5d8b4347202ff9db6bed09e70e704df8e35d37c3
J=/scratch/qi_zim_neu/olmoearth_inferenceX/agency
AREAS=$J/1525456/src/strata_size.csv
JUDGE=$J/1525456/judge_widths.partial.jsonl
BOX=$J/1531551/box_widths.json
for f in "$AREAS" "$JUDGE" "$BOX"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
export UV_CACHE_DIR=/scratch/qi_zim_neu/olmoearth_inferenceX/uv-cache
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=$J/e2e/$SLURM_JOB_ID
mkdir -p "$D" && cd "$D" || exit 1
git clone -q --branch cluster-suite https://github.com/2imi9/olmoearth_inferenceX.git pkg || exit 1
git clone -q --depth 1 https://github.com/2imi9/olmoearth_inferenceX-record.git rec || exit 1
( cd rec && git rev-parse HEAD > ../record_sha.txt )
cd pkg || exit 1
test "$(git rev-parse HEAD)" = "$SHA" || { echo "sha mismatch: $(git rev-parse HEAD)"; exit 1; }
~/.local/bin/uv sync -q --frozen --python 3.12 || exit 1
PY="$D/pkg/.venv/bin/python -I"
OX="$D/pkg/.venv/bin/oe-inferencex"
$PY -m pytest -q -p no:cacheprovider tests/test_agency.py 2>&1 | tail -3
cd "$D" || exit 1

# the point tables as an agency's user would write them: one row per point, codes as integers
$PY - "$D/rec" "$D" <<'PY' || { echo "tables failed"; exit 1; }
import csv, sys
rec, out = sys.argv[1], sys.argv[2]
T8 = {21: 1, 22: 1, 23: 1, 24: 1, 81: 2, 82: 2, 52: 3, 71: 3, 41: 4, 42: 4, 43: 4, 11: 5, 90: 6, 95: 6, 12: 7, 31: 8}
rows = list(csv.DictReader(open(f"{rec}/exp/out/exp94_nlcd_points.csv")))
# three alternate labels of set B lie outside the legend (50 twice, 91 once): no 8-class alternate, as the judge read them
for year in (2016, 2011):
    sel = [r for r in rows if r["set"] == "B" and r["year"] == str(year)]
    with open(f"{out}/nlcdB{year}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "stratum", "weight", "ref", "alt", "nlcd", "lcmap", "ref8", "alt8", "nlcd8", "ref1", "alt1",
                    "nlcd1"])
        for r in sel:
            c = lambda k: int(float(r[k]))
            w.writerow([r["pid"], r["stratum"], r["weight"], c("ref_nlcd"), c("ref_nlcd_alternate"), c("nlcd"),
                        c("lcmap"), T8[c("ref_nlcd")], T8.get(c("ref_nlcd_alternate"), ""), T8[c("nlcd")], c("ref_nlcd") // 10,
                        c("ref_nlcd_alternate") // 10, c("nlcd") // 10])
    print(year, len(sel), "points")
PY
cp "$D/rec/exp/out/exp94_gfc_points.csv" "$D/gfc.csv"

PURE="1=11 2=21 3=22 4=23 5=24 6=31 7=41 8=42 9=43 10=52 11=71 12=81 13=82 14=90 15=95"
PURE8="1=5 2=1 3=1 4=1 5=1 6=8 7=4 8=4 9=4 10=3 11=3 12=2 13=2 14=6 15=6"
PURE1="1=1 2=2 3=2 4=2 5=2 6=3 7=4 8=4 9=4 10=5 11=7 12=8 13=8 14=9 15=9"
NL="--stratum stratum --weight weight"
run() {   # name, then the arguments of grade
  local name=$1; shift
  local t0=$SECONDS
  $OX grade "$@" --out "$D/$name.json" > "$D/$name.txt" 2>&1
  local rc=$?
  echo "$name exit $rc in $((SECONDS - t0)) s"
  cat "$D/$name.txt"
  echo
}
run nlcd2016_l2_alt     nlcdB2016.csv $NL --reference ref  --alternate alt  --map nlcd  --pure $PURE
run nlcd2016_l2_primary nlcdB2016.csv $NL --reference ref                   --map nlcd  --pure $PURE
run nlcd2011_l2_alt     nlcdB2011.csv $NL --reference ref  --alternate alt  --map nlcd  --pure $PURE
run nlcd2016_l1_alt     nlcdB2016.csv $NL --reference ref1 --alternate alt1 --map nlcd1 --pure $PURE1
run nlcd2016_l1_primary nlcdB2016.csv $NL --reference ref1                  --map nlcd1 --pure $PURE1
run nlcd2016_8_alt      nlcdB2016.csv $NL --reference ref8 --alternate alt8 --map nlcd8 --map lcmap --pure $PURE8
run nlcd2016_8_primary  nlcdB2016.csv $NL --reference ref8                  --map nlcd8 --map lcmap --pure $PURE8
run gfc2020             gfc.csv --stratum stratum --strata "$AREAS" --sizes area --unit-area 1 --reference ref_forest \
                        --map gfc_v2 --map worldcover_forest

# every number beside the published one and the judge's
$PY - "$D" "$JUDGE" "$BOX" > "$D/compare.json" <<'PY' || { echo "compare failed"; exit 1; }
import json, sys
D, judge_path, box_path = sys.argv[1:4]
J = {tuple(r["key"]): r["result"] for r in map(json.loads, open(judge_path))}
B = json.load(open(box_path))
g = lambda name: json.load(open(f"{D}/{name}.json"))
# NLCD: Wickham et al. (2021) as quoted in the record (exp94) and transcribed in the judge script (Table 4, level II
# 2016, user's and producer's accuracy by the primary-or-alternate rule, percent); GFC2020: exp94's published value
PUB_OA = {("nlcd2016_l2_alt", "nlcd"): 86.4, ("nlcd2016_l2_primary", "nlcd"): 72.1, ("nlcd2011_l2_alt", "nlcd"): 86.5,
          ("nlcd2016_l1_primary", "nlcd1"): 79.0, ("nlcd2016_l1_alt", "nlcd1"): 90.6, ("gfc2020", "gfc_v2"): 91.0}
PUB_CLASS = {"11": (99, 95), "21": (88, 61), "22": (74, 70), "23": (77, 84), "24": (75, 78), "31": (77, 77),
             "41": (90, 79), "42": (93, 85), "43": (75, 78), "52": (87, 95), "71": (91, 91), "81": (77, 76),
             "82": (93, 94), "90": (61, 96), "95": (69, 80)}
JUDGE_OA = {("nlcd2016_l2_alt", "nlcd"): ("nlcd2016_B_2016", "oa_level2_primary_or_alternate"),
            ("nlcd2016_l2_primary", "nlcd"): ("nlcd2016_B_2016", "oa_level2_primary"),
            ("nlcd2011_l2_alt", "nlcd"): ("nlcd2016_B_2011", "oa_level2_primary_or_alternate"),
            ("nlcd2016_l1_primary", "nlcd1"): ("nlcd2016_B_2016", "oa_level1_primary"),
            ("nlcd2016_l1_alt", "nlcd1"): ("nlcd2016_B_2016", "oa_level1_primary_or_alternate_level2_alternate"),
            ("gfc2020", "gfc_v2"): ("gfc2020", "oa_gfc_v2"), ("gfc2020", "worldcover_forest"): ("gfc2020", "oa_worldcover")}
JUDGE_DIFF = {("nlcd2016_8_alt", "nlcd8 minus lcmap"): ("nlcd2016_B_2016", "nlcd_minus_lcmap_8class_primary_or_alternate"),
              ("nlcd2016_8_primary", "nlcd8 minus lcmap"): ("nlcd2016_B_2016", "nlcd_minus_lcmap_8class_primary"),
              ("gfc2020", "gfc_v2 minus worldcover_forest"): ("gfc2020", "gfc_minus_worldcover")}
out = {"overall": [], "differences": [], "per_class": [], "worst_abs_difference_to_judge": 0.0}


def diff(a, b):
    d = max(abs(x - y) for x, y in zip(a, b))
    out["worst_abs_difference_to_judge"] = max(out["worst_abs_difference_to_judge"], d)
    return d


for (name, m), pub in list(PUB_OA.items()) + [(k, None) for k in JUDGE_OA if k not in PUB_OA]:
    q = g(name)["overall_accuracy"][m]
    row = {"run": name, "map": m, "estimate": q["estimate"], "interval": q["interval"], "normal": q["normal_theory"],
           "width_over_normal": (q["interval"][1] - q["interval"][0]) / (q["normal_theory"][1] - q["normal_theory"][0]),
           "published": pub, "estimate_minus_published_points": None if pub is None else 100 * q["estimate"] - pub}
    j = J.get(JUDGE_OA.get((name, m)))
    if j:
        row["judge_betting"] = j["betting"]
        row["abs_difference_to_judge"] = diff(q["interval"], j["betting"])
        row["judge_estimate_difference"] = abs(q["estimate"] - j["estimate"])
    out["overall"].append(row)
for (name, k), jk in JUDGE_DIFF.items():
    q = g(name)["differences"][k]
    row = {"run": name, "difference": k, "estimate": q["estimate"], "interval": q["interval"], "normal": q["normal_theory"],
           "width_over_normal": (q["interval"][1] - q["interval"][0]) / (q["normal_theory"][1] - q["normal_theory"][0])}
    j = J.get(jk)
    if j:
        row["judge_betting"] = j["betting"]
        row["abs_difference_to_judge"] = diff(q["interval"], j["betting"])
    out["differences"].append(row)
for name, m, jname, bkey in (("nlcd2016_l2_alt", "nlcd", "nlcd2016_B_2016", "nlcd2016_B_2016/{w}/{c}"),
                             ("nlcd2016_l2_primary", "nlcd", "nlcd2016_B_2016", None),
                             ("gfc2020", "gfc_v2", "gfc2020", None), ("gfc2020", "worldcover_forest", "gfc2020", None)):
    for c, row in g(name)["per_class"][m].items():
        u, p, s = row["user"], row["producer"], row["reference_share"]
        r = {"run": name, "map": m, "class": c, "user": u["estimate"], "user_interval": u["interval"],
             "user_method": u["method"], "user_normal": u["normal_theory"], "producer": p["estimate"],
             "producer_interval": p["interval"], "producer_normal": p["normal_theory"], "share": s["estimate"],
             "share_interval": s["interval"], "labels_mapped": row["labels_mapped"]}
        if name == "nlcd2016_l2_alt" and c in PUB_CLASS:
            r["published_user_producer"] = PUB_CLASS[c]
            r["minus_published_points"] = [100 * u["estimate"] - PUB_CLASS[c][0], 100 * p["estimate"] - PUB_CLASS[c][1]]
        jc = J.get((jname, "per_class", c)) if jname.startswith("nlcd") else None
        if jc:
            ju = jc["user"] if name == "nlcd2016_l2_alt" else jc["user_primary"]
            if u["method"] == "pure-stratum rule" and "m2" in ju:
                r["judge_user_m2"] = ju["m2"]
                r["abs_difference_to_judge_user"] = diff(u["interval"], ju["m2"])
            if name == "nlcd2016_l2_alt":
                r["abs_difference_to_judge_share"] = diff(s["interval"], jc["reference_share"]["betting"])
        if name == "nlcd2016_l2_alt":
            bu, bp = B.get(f"nlcd2016_B_2016/user/{c}"), B.get(f"nlcd2016_B_2016/producer/{c}")
            if bp and "box" in bp:
                r["judge_producer_box"] = bp["box"]
                r["abs_difference_to_judge_producer"] = diff(p["interval"], bp["box"])
            if bu and "box" in bu and u["method"] != "pure-stratum rule":
                r["abs_difference_to_judge_user"] = diff(u["interval"], bu["box"])
        if name == "gfc2020":
            cname = {"1": "forest", "0": "nonforest"}[c]
            mname = "gfc_v2" if m == "gfc_v2" else "worldcover"
            for w, q in (("user", u), ("producer", p)):
                b = B.get(f"gfc2020/{w}_{cname}_{mname}")
                if b and "box" in b:
                    r[f"abs_difference_to_judge_{w}"] = diff(q["interval"], b["box"])
        out["per_class"].append(r)
json.dump(out, sys.stdout, indent=1)
PY
echo "compare exit $?"
$PY - "$D/compare.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
print("worst |interval end - judge's| over every number compared:", c["worst_abs_difference_to_judge"])
for r in c["overall"]:
    print(f"OA {r['run']:22s} {r['map']:18s} {100 * r['estimate']:6.2f} published {r['published']} "
          f"[{100 * r['interval'][0]:.2f}, {100 * r['interval'][1]:.2f}] normal [{100 * r['normal'][0]:.2f}, "
          f"{100 * r['normal'][1]:.2f}] x{r['width_over_normal']:.2f} judge diff {r.get('abs_difference_to_judge')}")
for r in c["differences"]:
    print(f"diff {r['run']:20s} {r['difference']:32s} {100 * r['estimate']:+6.2f} [{100 * r['interval'][0]:+.2f}, "
          f"{100 * r['interval'][1]:+.2f}] x{r['width_over_normal']:.2f} judge diff {r.get('abs_difference_to_judge')}")
for r in c["per_class"]:
    if r["run"] == "nlcd2016_l2_alt":
        print(f"class {r['class']:3s} UA {100 * r['user']:5.1f} [{100 * r['user_interval'][0]:.1f}, {100 * r['user_interval'][1]:.1f}] "
              f"{r['user_method'][:4]} PA {100 * r['producer']:5.1f} [{100 * r['producer_interval'][0]:.1f}, "
              f"{100 * r['producer_interval'][1]:.1f}] pub {r.get('published_user_producer')} judge diffs "
              f"{r.get('abs_difference_to_judge_user')}, {r.get('abs_difference_to_judge_producer')}")
PY
sha256sum "$D"/*.json "$D"/*.txt
echo "E2E DONE"
