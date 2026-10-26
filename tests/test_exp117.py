"""exp117 (exp/exp117_agency/): grading an agency's stratified sample with intervals valid at any sample size. The
shipped command's numbers against a second route written here from the points alone, its intervals against the
design study's own outputs, and the summary against the raw outputs. Standard library only."""
import csv
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out")
E2E = os.path.join(OUT, "exp117_e2e")


def _json(*parts):
    with open(os.path.join(OUT, *parts), encoding="utf-8") as f:
        return json.load(f)


def _judge():
    with open(os.path.join(OUT, "exp117_judge_widths.jsonl"), encoding="utf-8") as f:
        return {tuple(r["key"]): r["result"] for r in map(json.loads, f)}


def _stratified_mean(rows, stratum, size, right):
    """sum_h W_h (share right among stratum h's points), W_h = size_h / total."""
    acc = {}
    for r in rows:
        a = acc.setdefault(stratum(r), [0, 0])
        a[0] += 1
        a[1] += right(r)
    total = sum(size(h) for h in acc)
    return sum(size(h) / total * k / n for h, (n, k) in acc.items())


def test_overall_accuracy_by_a_second_route():
    """NLCD 2016 AA set B (weights: a stratum's size is the sum of its weights, rounded) and GFC2020 (sizes in ha):
    the overall accuracies grade printed, recomputed here without the package."""
    with open(os.path.join(OUT, "exp94_nlcd_points.csv"), newline="", encoding="utf-8") as f:
        pts = list(csv.DictReader(f))
    code = lambda r, k: int(float(r[k])) if r[k] not in ("", None) else None
    rules = {"nlcd2016_l2_alt": (2016, lambda m, p, a: m in (p, a)), "nlcd2016_l2_primary": (2016, lambda m, p, a: m == p),
             "nlcd2011_l2_alt": (2011, lambda m, p, a: m in (p, a)),
             "nlcd2016_l1_primary": (2016, lambda m, p, a: m // 10 == p // 10),
             "nlcd2016_l1_alt": (2016, lambda m, p, a: m // 10 in (p // 10, (a or 0) // 10))}
    got = {(r["run"], r["map"]): r["estimate"] for r in _json("exp117_e2e", "compare.json")["overall"]}
    for run, (year, right) in rules.items():
        rows = [r for r in pts if r["set"] == "B" and r["year"] == str(year)]
        wsum = {}
        for r in rows:
            wsum[r["stratum"]] = wsum.get(r["stratum"], 0.0) + float(r["weight"])
        mine = _stratified_mean(rows, lambda r: r["stratum"], lambda h: round(wsum[h]),
                                lambda r: right(code(r, "nlcd"), code(r, "ref_nlcd"), code(r, "ref_nlcd_alternate")))
        mapname = "nlcd1" if "_l1_" in run else "nlcd"
        assert abs(mine - got[(run, mapname)]) < 1e-9, run
    with open(os.path.join(OUT, "exp117_gfc2020_strata_size.csv"), newline="", encoding="utf-8") as f:
        area = {r["Strata"]: float(r["strata_ha"]) for r in csv.DictReader(f)}
    with open(os.path.join(OUT, "exp94_gfc_points.csv"), newline="", encoding="utf-8") as f:
        gfc = list(csv.DictReader(f))
    assert {r["stratum"] for r in gfc} == set(area)              # every stratum sampled: no bound for an empty one
    for col, mapname in (("gfc_v2", "gfc_v2"), ("worldcover_forest", "worldcover_forest")):
        mine = _stratified_mean(gfc, lambda r: r["stratum"], lambda h: area[h], lambda r: r[col] == r["ref_forest"])
        assert abs(mine - got[("gfc2020", mapname)]) < 1e-9, col


def test_the_shipped_intervals_equal_the_design_studys():
    """Every interval grade printed, read from its own JSON, against the design study's scripts' outputs (written
    separately; the package's interval code is a port of theirs): overall accuracies and differences against the
    judge's betting intervals, user's accuracies by the pure-stratum rule against the judge's, producer's accuracies
    against the box run's."""
    J, B = _judge(), _json("exp117_box_widths.json")
    g = lambda name: _json("exp117_e2e", f"{name}.json")
    pairs = [(g("nlcd2016_l2_alt")["overall_accuracy"]["nlcd"], J[("nlcd2016_B_2016", "oa_level2_primary_or_alternate")]["betting"]),
             (g("nlcd2016_l2_primary")["overall_accuracy"]["nlcd"], J[("nlcd2016_B_2016", "oa_level2_primary")]["betting"]),
             (g("nlcd2011_l2_alt")["overall_accuracy"]["nlcd"], J[("nlcd2016_B_2011", "oa_level2_primary_or_alternate")]["betting"]),
             (g("gfc2020")["overall_accuracy"]["gfc_v2"], J[("gfc2020", "oa_gfc_v2")]["betting"]),
             (g("gfc2020")["differences"]["gfc_v2 minus worldcover_forest"], J[("gfc2020", "gfc_minus_worldcover")]["betting"]),
             (g("nlcd2016_8_alt")["differences"]["nlcd8 minus lcmap"],
              J[("nlcd2016_B_2016", "nlcd_minus_lcmap_8class_primary_or_alternate")]["betting"])]
    alt = g("nlcd2016_l2_alt")["per_class"]["nlcd"]
    for c, row in alt.items():
        j = J.get(("nlcd2016_B_2016", "per_class", c))
        if j and "m2" in j["user"]:
            assert row["user"]["method"] == "pure-stratum rule"
            pairs.append((row["user"], j["user"]["m2"]))
        b = B.get(f"nlcd2016_B_2016/producer/{c}")
        if b and "box" in b:
            pairs.append((row["producer"], b["box"]))
    assert len(pairs) >= 6 + 8 + 15
    for q, ref in pairs:
        assert max(abs(x - y) for x, y in zip(q["interval"], ref)) < 1e-12


def test_the_summary_against_the_raw_outputs():
    S = _json("exp117_summary.json")
    mc = _json("exp117_box_mc.json")
    cases = [v for v in mc.values() if isinstance(v, dict) and "holds" in v]
    valid = [v[m]["coverage"] for v in cases for m in ("betting", "box") if m in v]
    assert S["simulation"]["valid_coverage"]["min"] == min(valid) and all(v["holds"] for v in cases)
    assert min(valid) >= mc["gate"]
    en = _json("exp117_judge_enum.json")
    n = sum(1 for a in ("alpha_0.1", "alpha_0.3") for c in en[a].values() if isinstance(c, dict) and "holds" in c)
    assert S["judge_enumeration"]["cells"] == n == S["judge_enumeration"]["cells_holding"]
    assert S["judge_enumeration"]["least_slack_over_claimed_forms"] >= -1e-12
    ratios = [r["width_over_normal"] for r in _json("exp117_e2e", "compare.json")["overall"]
              + _json("exp117_e2e", "compare.json")["differences"]]
    assert S["end_to_end"]["width_over_normal_overall_and_differences"]["median"] == statistics.median(ratios)
    # the design study's widths, from the judge's rows and the box run: betting over the fallback for linear
    # quantities, and the box over the fallback and over the scan for each user's and producer's accuracy
    J, B = _judge(), _json("exp117_box_widths.json")
    w = lambda iv: iv[1] - iv[0]
    lin = [q["widths_points"]["betting"] / q["widths_points"]["bsc"] for q in J.values()
           if q.get("family") == "linear" and "bsc" in q.get("widths_points", {})]
    over_bsc, over_scan = [], []
    for key, q in J.items():
        if key[1:2] != ("per_class",):
            continue
        for what in ("user", "producer"):
            b, j = B.get(f"nlcd2016_B_2016/{what}/{key[2]}"), q.get(what, {})
            if b and "box" in b and isinstance(j.get("bsc"), list):
                over_bsc.append(w(b["box"]) / w(j["bsc"]))
                over_scan.append(w(b["box"]) / w(j["betting"]))
    W = S["design_study_widths"]
    assert len(lin) == W["linear_betting_over_fallback"]["n"] == 12
    assert abs(statistics.median(lin) - W["linear_betting_over_fallback"]["median"]) < 1e-12
    assert len(over_bsc) == W["ratio_box_over_fallback"]["n"] == 30
    assert abs(statistics.median(over_bsc) - W["ratio_box_over_fallback"]["median"]) < 1e-12
    assert abs(statistics.median(over_scan) - W["ratio_box_over_scan"]["median"]) < 1e-12
