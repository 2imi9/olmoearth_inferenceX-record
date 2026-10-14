"""exp99: the transfer test. Ai2's FT-AWF model, run through the same olmoearth_run deployment configuration as exp98
but on 2017 imagery, over the 1024 x 1024-pixel windows that hold a plot of the East Africa TimeSync sample within
100 km of Ai2's AWF request geometry, graded at those plots, whose 2017 labels come from an independent simple random
sample (Bullock et al. 2021). Preregistered in docs/plan/awf_transfer.md; read that page first, this file is the run.

    python exp/exp99_transfer.py --smoke                                  # synthetic inputs end to end
    python exp/exp99_transfer.py --select --timesync CSV --request-geometry AI2.geojson [--boundaries NE.geojson]
        [--write-request DIR] [--record exp/out/exp99/e99_select.json]   # the area rule; counts only
    python exp/exp99_transfer.py --inventory --scores DIR [DIR ...] --timesync CSV --request-geometry AI2.geojson
        [--boundaries NE.geojson]                                         # counts only, allowed before freezing
    python exp/exp99_transfer.py --scores DIR [DIR ...] --timesync CSV --request-geometry AI2.geojson
        --boundaries NE.geojson [--exp98-scores DIR [DIR ...]]           # the run, once the page is frozen

The area rule, fixed before any map existed (9 October 2026, from the local copy of the sample): the plots of Kenya and
Tanzania with a 2017 row whose distance to Ai2's request geometry (EPSG:32737, the geometry's edges densified every
0.001 degree; 0 inside) is at most D, D the smallest multiple of 10 km giving at least 300 plots: D = 100 km, 309 plots
(219 in Kenya, 90 in Tanzania), 47 of them inside the geometry (33 and 14). The script refuses other counts. The region
is the D-km buffer of the geometry; each country's part of it is that country's stratum, with weight its share of the
region's area (Natural Earth's boundaries, in EPSG:6933).

What it reads: the from-olmoearth part directories of the 2017 run (exp/jobs/e99_read.sh; one directory per window,
read as one map as exp98 reads its parts), each plot at the 10 m score pixel holding its point. It writes
exp/out/exp99_summary.json and exp99_units.npz (per plot: country, label, covered, the deployed class and confidence,
the condition code, the 3 x 3 majority, the design weight and the errors under each rule; no identifier and no
position), exp99_inventory.json (--inventory), exp/out/exp99/e99_select.json (--select), exp99_summary_smoke.json
(--smoke). Nothing written holds a coordinate, a transform, a bound or a plot identifier; the request geometry of
squares around the plots is written only by --select --write-request, to a directory outside every git checkout (the
cluster's scratch).

Measures (the page's parts):
  A  the error rate of the 2017 map over the region, under each crosswalk rule: the countries as strata, each
     country's exact hypergeometric interval combined by the package's condition-design interval
     (oe_inferencex.estimate), the source's ratio estimate and the unweighted rate beside it; set beside Ai2's 89.5%,
     with the transfer-gap call fixed in advance (descriptive, P3);
  B  the ranking (graded: P1, P2 on STRICT, LENIENT reported): the design-weighted AUROC of the deployed confidence
     for the errors and the share of the gap from random to the ceiling that the least confident 10% closes, with
     bootstraps within countries;
  C  by TimeSync class, by condition code and the 3 x 3 majority (descriptive); the plots without input;
  D  where the plots sit among the windows' pixels (report-only; about 0.5 for a random sample);
  E  what plan says about certify on the region (report-only); certify itself is not applied (the page says why);
  F  the 47 plots inside Ai2's geometry on the 2017 map and, with --exp98-scores, on exp98's 2023 map (report-only).
"""
import argparse
import json
import math
import os
import sys
import time
import traceback

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
for _p in (ROOT, EXP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from oe_inferencex import estimate as est, metrics, plan, sequential as sq, stats   # noqa: E402
import exp89_finetuned_checkpoints as e89                                          # noqa: E402
import exp98_awf_deployment as e98                                                 # noqa: E402
import timesync_awf_crosswalk as tsx                                               # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
PLAN = os.path.join(ROOT, "docs", "plan", "awf_transfer.md")

YEAR = 2017
REQUEST_PERIOD = ("2017-01-01T00:00:00Z", "2017-12-31T00:00:00Z")   # Ai2's property names, oe_start_time/oe_end_time
PROPERTY_NAMES = ("oe_start_time", "oe_end_time")
MIN_PLOTS = 300
D_STEP_KM = 10
# fixed on 9 October 2026 from the local copy of the sample (sha256 in timesync_awf_crosswalk.TIMESYNC) against Ai2's
# geometry (sha256 AWF_GEOMETRY_SHA256), before any 2017 map existed; --select and the run refuse other counts
FIXED_RULE = {"d_km": 100, "n_selected": 309, "by_country": {"kenya": 219, "tanzania": 90},
              "inside_request_geometry": {"kenya": 33, "tanzania": 14}}
SQUARE_HALF_DEG = 0.00005          # each plot's request square, about 11 m on a side: it meets one 1024-px cell, or two
                                   # where the plot lies within about 6 m of a cell's edge
PILOT_N = 4
SEED = 99

PRIMARY = "strict"
THRESHOLDS = {"P1": 0.70, "P2": 0.25}          # confirmed by the owner on 9 October 2026, before the page was frozen
FLOORS = {"min_with_input": 250, "min_errors": 20, "min_correct": 20}
PREDICTIONS = tuple(THRESHOLDS)
AI2_ACCURACY = 0.895
GAP_MARGIN = 0.05                               # exp98's equivalence margin
BUDGETS = (0.05, 0.10, 0.20)
BOOT = 2000
SMOKE_BOOT = 200
EPS = 1e-12

PLAN_WIDTH = 0.10
PLAN_ALPHA = 0.10
PLAN_COVERAGES = (0.5, 0.8, 1.0)
PLAN_ZONE_DRAWS = plan.ZONE_DRAWS
ZONE_PLAN_MAX_N = 999_999_999      # numpy's multivariate hypergeometric sampler takes populations below 1e9
MAP_SAMPLE = 2_000_000
CANNOT = ("A certified zone needs the whole region's confidence ranking: certify's zones are the most confident shares "
          "of every window of the map, and a run over the windows that hold a plot does not give the region's "
          "ranking. plan's labels for certify are reported instead.",
          "The windows are a cluster sample of the region chosen through the plots: only the plots are a probability "
          "sample, and the windows' other pixels are described, never graded.",
          "2017 imagery read by a model fine-tuned on 2023 imagery, in places beyond its training area, against "
          "labels from another legend through a crosswalk: a gap between the error rate here and Ai2's 89.5% has "
          "all of these as causes, and this experiment does not separate them. Part F compares the 2017 and 2023 "
          "maps at the 47 plots inside Ai2's geometry; that difference mixes the imagery's year, satellite "
          "availability and processing, and land change after 2017 (which counts against the 2023 map), so it is "
          "descriptive and apportions no gap.",
          "The sample's design makes the plots of a country inside the region a simple random sample of its part of "
          "the region given their number; plots without input (a window with no 2017 imagery) are treated as missing "
          "at random within their country, and their labels are reported.")


# ----------------------------------------------------------------------------- helpers
def prereg_status(text=None):
    if text is None:
        with open(PLAN, encoding="utf-8") as f:
            text = f.read()
    return e89.prereg_status(text)


def bkey(b):
    return e89.bkey(b)


def finite(x):
    return e98.finite(x)


def utm_zone(lon):
    return np.floor((np.asarray(lon) + 180) / 6).astype(int) + 1


# ----------------------------------------------------------------------------- the area rule
def select(timesync, request_geometry, boundaries=None, geometry_sha256=tsx.AWF_GEOMETRY_SHA256, min_plots=MIN_PLOTS,
           fixed=FIXED_RULE):
    """(the selected plots, in memory; the counts record; the region's areas or None). Refuses counts other than
    `fixed` (None skips the check, for synthetic inputs)."""
    plots = tsx.read_plots(timesync, YEAR)
    geom = tsx.load_request_geometry(request_geometry, geometry_sha256)
    dist = tsx.distance_km(geom, plots["lon"], plots["lat"])
    rule = tsx.area_rule(dist, min_plots, D_STEP_KM)
    sel = dist <= rule["d_km"]
    P = tsx.subset(plots, sel)
    P["inside"] = dist[sel] == 0
    ctry = P["country"]
    rec = {"rule": f"plots of {' and '.join(tsx.COUNTRIES)} with a {YEAR} label within D km of the request geometry "
                   f"(distance in {tsx.DISTANCE_CRS}, 0 inside), D the smallest multiple of {D_STEP_KM} km giving at "
                   f"least {min_plots} plots",
           "d_km": rule["d_km"], "counts_by_d_km": {str(k): v for k, v in rule["counts_by_d_km"].items()},
           "n_selected": int(sel.sum()), "by_country": {c: int((ctry == c).sum()) for c in tsx.COUNTRIES},
           "inside_request_geometry": {c: int(((ctry == c) & P["inside"]).sum()) for c in tsx.COUNTRIES},
           "labels": {t: int((P["label"] == t).sum()) for t in tsx.TIMESYNC_CLASSES if (P["label"] == t).any()},
           "utm_zones": {str(z): int((utm_zone(P["lon"]) == z).sum()) for z in np.unique(utm_zone(P["lon"]))},
           "label_changes": {"two_or_more_labels_2015_2017": int((P["n_labels_recent"] > 1).sum()),
                             "two_or_more_labels_any_year": int((P["n_labels_all_years"] > 1).sum())},
           "timesync": {k: tsx.TIMESYNC[k] for k in ("repository", "sha256", "licence")},
           "request_geometry_sha256": geometry_sha256}
    if fixed is not None:
        got = {k: rec[k] for k in fixed}
        if got != fixed:
            raise ValueError(f"the area rule gives {got}, the preregistration fixed {fixed}: the sample or the "
                             "geometry is not the one the rule was fixed on")
        rec["matches_preregistration"] = True
    areas = None
    if boundaries:
        areas = tsx.region_areas(tsx.buffer_wgs84(geom, rule["d_km"]), boundaries)
        tot = sum(areas["by_country_km2"].values())
        rec["areas"] = {"region_km2": areas["region_km2"], "by_country_km2": areas["by_country_km2"],
                        "weights": {c: v / tot for c, v in areas["by_country_km2"].items()},
                        "share_outside_both_countries": areas["share_outside_countries"],
                        "country_km2": areas["country_km2"], "source": tsx.BOUNDARIES["repository"],
                        "area_crs": tsx.AREA_CRS}
        rec["plots_on_boundaries"] = tsx.plot_countries_on_boundaries(P["lon"], P["lat"], ctry, boundaries)
    return P, rec, areas


def request_squares(P, half=SQUARE_HALF_DEG, period=REQUEST_PERIOD):
    """The request geometry olmoearth_run reads: a FeatureCollection with ONE feature, a MultiPolygon of one small
    square around each plot, with Ai2's property names for the period, as Ai2's own file holds one feature.
    One feature per square multiplied the windows: olmoearth_run's GridPartitioner (partition_request_geometry, 1-degree
    grid, clip off) turns each feature into the whole 1-degree cell holding it, and each of those partitions then gets
    a window for every square in its cell, so k plots in a cell gave k squared windows (prepare job 1247349: 9,298
    windows for 309 plots, cancelled after 7 h). Coordinates: written only to scratch."""
    polys = []
    for lon, lat in zip(P["lon"], P["lat"]):
        ring = [[lon - half, lat - half], [lon + half, lat - half], [lon + half, lat + half], [lon - half, lat + half],
                [lon - half, lat - half]]
        polys.append([ring])
    feat = {"geometry": {"coordinates": polys, "type": "MultiPolygon"},
            "properties": dict(zip(PROPERTY_NAMES, period)), "type": "Feature"}
    return {"features": [feat], "type": "FeatureCollection"}


def n_squares(gj):
    """How many plot squares a request geometry holds (a MultiPolygon's parts, or one per Polygon feature)."""
    return sum(len(f["geometry"]["coordinates"]) if f["geometry"]["type"] == "MultiPolygon" else 1
               for f in gj["features"])


def structure(gj):
    """What must match Ai2's file: the keys at each level, a polygonal geometry, the property names. Polygon and
    MultiPolygon count as the same here: Ai2's file holds one Polygon, exp99's one MultiPolygon of the squares."""
    poly = {"Polygon": "polygonal", "MultiPolygon": "polygonal"}
    return {"top": sorted(gj), "feature": sorted({tuple(sorted(f)) for f in gj["features"]}),
            "geometry": sorted({(poly.get(f["geometry"]["type"], f["geometry"]["type"]), tuple(sorted(f["geometry"])))
                                for f in gj["features"]}),
            "properties": sorted({tuple(sorted(f["properties"])) for f in gj["features"]})}


def inside_a_checkout(path):
    p = os.path.realpath(path)
    while True:
        if os.path.exists(os.path.join(p, ".git")):
            return True
        parent = os.path.dirname(p)
        if parent == p:
            return False
        p = parent


def write_request(dest, P, ai2_geometry, pilot_n=PILOT_N, seed=SEED):
    """The full and pilot request geometries under `dest` (refused inside any git checkout), checked against the
    structure of Ai2's file. Returns counts and sha256s only."""
    if inside_a_checkout(dest):
        raise ValueError(f"refusing to write a request geometry inside a git checkout ({dest}): it holds positions")
    os.makedirs(dest, exist_ok=True)
    with open(ai2_geometry) as f:
        ai2 = json.load(f)
    full = request_squares(P)
    pick = np.sort(np.random.default_rng(seed).choice(len(P["lon"]), pilot_n, replace=False))
    pilot = request_squares(tsx.subset({k: v for k, v in P.items()}, pick))
    out = {}
    for name, gj in (("prediction_request_geometry.geojson", full), ("prediction_request_geometry_pilot.geojson", pilot)):
        s, a = structure(gj), structure(ai2)
        if s != a:
            raise ValueError(f"{name} does not have the structure of Ai2's request geometry: {s} against {a}")
        path = os.path.join(dest, name)
        with open(path + ".partial", "w") as f:
            json.dump(gj, f)
        os.replace(path + ".partial", path)
        out[name] = {"features": len(gj["features"]), "squares": n_squares(gj), "sha256": tsx.sha256_file(path)}
    out["period"] = dict(zip(PROPERTY_NAMES, REQUEST_PERIOD))
    out["square_half_deg"] = SQUARE_HALF_DEG
    out["pilot"] = {"n": pilot_n, "seed": seed, "rule": "numpy default_rng(seed).choice over the selected plots"}
    out["structure_matches_ai2"] = True
    return out


# ----------------------------------------------------------------------------- measures
def ranking(u, err, weights, strata, n_boot, seed=SEED):
    """The ranking at the plots: AUROC and capture at 5/10/20%, design-weighted (the region) and unweighted (the
    plots), the ceiling and the share of the gap from random to the ceiling closed, and bootstraps within the strata
    (countries) of the weighted AUROC and of the gap closed at 10%."""
    u, err, w = (np.asarray(a, dtype=np.float64) for a in (u, err, weights))
    n, k = err.size, int(err.sum())
    if k == 0 or k == n:
        return {"n": n, "n_errors": k, "note": "no ranking: no error or no correct plot"}

    def block(ww):
        theta = metrics.weighted_mean(err, ww)
        cap = metrics.weighted_capture_at_budget(u, err, ww, BUDGETS)
        ceil = {b: min(1.0, b / theta) for b in BUDGETS}
        return {"auroc": metrics.weighted_auroc(u, err, ww), "error_rate": theta,
                "capture": {bkey(b): cap[b] for b in BUDGETS}, "random": {bkey(b): b for b in BUDGETS},
                "ceiling": {bkey(b): ceil[b] for b in BUDGETS},
                "gap_closed": {bkey(b): (cap[b] - b) / (ceil[b] - b) if ceil[b] > b + EPS else None for b in BUDGETS},
                "lift": {bkey(b): cap[b] / b for b in BUDGETS},
                "aurc": metrics.weighted_aurc(u, err, ww), "excess_aurc": metrics.weighted_excess_aurc(u, err, ww)}
    out = {"n": n, "n_errors": k, "weighted": block(w), "unweighted": block(np.ones(n))}
    rng = np.random.default_rng(seed)
    st = np.asarray(strata)
    idx = {s: np.flatnonzero(st == s) for s in np.unique(st)}
    au, gc = [], []
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(ix, ix.size, replace=True) for ix in idx.values()])
        e_, u_, w_ = err[pick], u[pick], w[pick]
        if 0 < e_.sum() < e_.size:
            au.append(metrics.weighted_auroc(u_, e_, w_))
            th = metrics.weighted_mean(e_, w_)
            c = metrics.weighted_capture_at_budget(u_, e_, w_, (0.10,))[0.10]
            ce = min(1.0, 0.10 / th)
            if ce > 0.10 + EPS:
                gc.append((c - 0.10) / (ce - 0.10))
    for name, v in (("auroc_weighted_bootstrap", au), ("gap_closed_0.1_weighted_bootstrap", gc)):
        v = np.sort(np.asarray(v))
        out[name] = {"n_resamples": int(v.size)} if not v.size else {
            "n_resamples": int(v.size), "lo90": float(np.quantile(v, 0.05)), "hi90": float(np.quantile(v, 0.95)),
            "lower_one_sided_95": float(np.quantile(v, 0.05)), "resampling": "plots within each country"}
    return out


def gap_call(strict, lenient, ai2_accuracy=AI2_ACCURACY, margin=GAP_MARGIN):
    """P3's call, fixed in advance: a transfer gap when even LENIENT's whole interval lies above Ai2's error rate plus
    the margin; none shown when even STRICT's whole interval lies below it; otherwise not determined."""
    bar = 1 - ai2_accuracy + margin
    if lenient["low"] > bar + EPS:
        call = "transfer gap"
    elif strict["high"] < bar - EPS:
        call = "no gap shown"
    else:
        call = "not determined"
    return {"bar_error_rate": bar, "ai2_accuracy": ai2_accuracy, "margin": margin, "call": call,
            "rule": "a transfer gap when LENIENT's exact 95% interval lies wholly above Ai2's error rate (10.5%) plus 5 "
                    "points; no gap shown when STRICT's lies wholly below it; otherwise not determined. Descriptive: "
                    "it does not say which of year, place, legend or Ai2's point placement makes the gap"}


def plan_block(n_region, rates, zone_errors, draws=PLAN_ZONE_DRAWS, max_labels=plan.MAX_LABELS):
    """Part E, report-only: what plan says the region's 2017 map needs, at one 10 m pixel per window."""
    out = {"n_windows": int(n_region), "patch_px": 1, "width": PLAN_WIDTH, "alpha": PLAN_ALPHA,
           "note": plan.PLAN_NOTE, "error_rate": {}, "zone": {}}
    for key, r in rates.items():
        if r is not None and not (finite(r) and 0 < r < 1):
            continue
        res = plan.plan_error_rate(n_region, PLAN_WIDTH, r, max_labels=max_labels)
        out["error_rate"][key] = {"error_rate": res["error_rate"], "labels": res["labels"],
                                  "probability_at_labels": res["probability_at_labels"], "refusal": res.get("refusal")}
    # plan_zone's simulation draws ring counts with numpy's multivariate hypergeometric sampler, which refuses a
    # population of 1e9 or more (method "marginals"); the region at 10 m is about 1.08e9 pixels. The zone is planned on
    # just under 1e9 windows, which moves no budget: the finite-population correction at a few hundred labels is of
    # order 1e-6 either way. Recorded, and reported as a limit of the package (plan.zone_prefix_probability).
    n_zone = min(int(n_region), ZONE_PLAN_MAX_N)
    out["zone_population"] = {"n_windows_planned": n_zone, "n_region": int(n_region),
                              "capped": bool(n_zone < n_region),
                              "why": "numpy's multivariate_hypergeometric (method marginals) refuses a population of "
                                     "1e9 or more"}
    for rule, zs in zone_errors.items():
        for c in PLAN_COVERAGES:
            z = zs.get(c)
            res = plan.plan_zone(n_zone, c, PLAN_ALPHA, zone_error=z if finite(z) else None, draws=draws,
                                 max_labels=max_labels, seed=SEED)
            keep = {k: res.get(k) for k in ("coverage", "zone_error", "labels_to_test", "min_labels_inside", "refusal",
                                            "unplanned")}
            for k in ("bonferroni", "prefix_even"):
                if k in res:
                    keep[k] = {"labels": res[k]["labels"], "probability_at_labels": res[k]["probability_at_labels"]}
            out["zone"][f"{rule}:{c:g}"] = keep
    out["zone_error_source"] = ("the design-weighted error rate among the most confident share of the graded plots, "
                                "a guess at the zone's: the plots are a sample of the region, but the zone is defined "
                                "on the region's own confidence ranking, which this run does not have")
    out["certify_min_labels"] = est.min_labels_to_certify(PLAN_ALPHA, est.ZONE_DELTA)
    out["sequential_min_labels"] = sq.min_labels_sequential(PLAN_ALPHA, est.ZONE_DELTA)
    return out


def zone_error_guess(u, err, weights, coverage):
    """The weighted error rate among the most confident `coverage` share (in weight) of the plots."""
    order = np.argsort(np.asarray(u), kind="stable")
    w = np.asarray(weights, dtype=np.float64)[order]
    e = np.asarray(err, dtype=np.float64)[order]
    keep = np.cumsum(w) <= coverage * w.sum() + 1e-12
    keep[0] = True
    return float((e[keep] * w[keep]).sum() / w[keep].sum())


def read_map(scores):
    entries, names, meta = e98.find_outputs(scores)
    return [e98.open_grid(e) for e in entries], names, meta


def read_plots_on(grids, P):
    cands = e98.locate_all(e98.timesync_points(P), grids)
    gi, rows, cols, how, off, Pr, B, C, overlap = e98.read_points(grids, cands)
    return gi, Pr, B, C, overlap


# ----------------------------------------------------------------------------- the grades
def grade(m, thresholds=THRESHOLDS, floors=FLOORS):
    """P1 and P2 on the primary rule (STRICT) at their floors; P3 is the descriptive call."""
    ok = (m.get("n_with_input") or 0) >= floors["min_with_input"] and (m.get("n_errors") or 0) >= floors["min_errors"] \
        and (m.get("n_correct") or 0) >= floors["min_correct"]
    why = (f"fewer than {floors['min_with_input']} plots with input, {floors['min_errors']} errors or "
           f"{floors['min_correct']} correct plots under {PRIMARY.upper()}")
    rules = {"P1": f"design-weighted AUROC of the deployed confidence for the {PRIMARY.upper()} errors at the plots >= "
                   f"{thresholds['P1']}",
             "P2": f"design-weighted share of the gap from random to the ceiling closed by the least confident 10% of "
                   f"the plots ({PRIMARY.upper()} errors) >= {thresholds['P2']}"}
    values = {"P1": m.get("p1_auroc"), "P2": m.get("p2_gap_closed")}
    # LENIENT's values sit beside the verdict as the check on the crosswalk's grass/shrub boundary; never graded
    values_lenient = {"P1": m.get("p1_auroc_lenient"), "P2": m.get("p2_gap_closed_lenient")}
    out = {}
    for p in PREDICTIONS:
        v = values[p]
        row = {"rule": rules[p], "threshold": thresholds[p], "value": float(v) if finite(v) else None}
        if not finite(v):
            row.update(graded=False, reason="not computable on these inputs")
        elif not ok:
            row.update(graded=False, reason=why)
        else:
            row.update(graded=True, holds=bool(v >= thresholds[p] - EPS))
        lv = values_lenient[p]
        row["lenient_check"] = {"value": float(lv) if finite(lv) else None,
                                "meets_threshold": bool(lv >= thresholds[p] - EPS) if finite(lv) else None}
        if row.get("graded") and not row["holds"]:
            lm = row["lenient_check"]["meets_threshold"]
            row["reading"] = ("both rules fail: transfer is the likelier reading" if lm is False else
                              "only STRICT fails: transfer or the crosswalk's grass/shrub boundary, not determined")
        out[p] = row
    out["P3"] = {"rule": "descriptive: the exact 95% interval of the error rate under each rule beside Ai2's 89.5%, and "
                         "the transfer-gap call fixed in advance", "graded": False, "call": m.get("p3_call")}
    out["floors"] = dict(floors, n_with_input=m.get("n_with_input"), n_errors=m.get("n_errors"),
                         n_correct=m.get("n_correct"))
    return out


# ----------------------------------------------------------------------------- the analysis
def analyse(scores, timesync, request_geometry, boundaries=None, inventory=False, exp98_scores=None,
            geometry_sha256=tsx.AWF_GEOMETRY_SHA256, fixed=FIXED_RULE, min_plots=MIN_PLOTS, n_boot=BOOT,
            floors=FLOORS, thresholds=THRESHOLDS, zone_draws=PLAN_ZONE_DRAWS, max_labels=plan.MAX_LABELS,
            map_sample=MAP_SAMPLE, exp98_frozen=None, log=print):
    t0 = time.time()
    P, sel_rec, areas = select(timesync, request_geometry, boundaries, geometry_sha256, min_plots, fixed)
    grids, names, meta = read_map(scores)
    gi, Pr, B, C, overlap = read_plots_on(grids, P)
    cov = e98.covered(Pr)
    ctry, lab = np.asarray(P["country"]), np.asarray(P["label"])
    countries = tuple(tsx.COUNTRIES)
    log(f"{len(lab)} plots selected, {int((gi >= 0).sum())} in a grid, {int(cov.sum())} covered; {len(grids)} grid(s)")
    commit, dirty = e89.git_state()
    record = {"experiment": "exp99: the transfer test, Ai2's FT-AWF deployment configuration on 2017 imagery over the "
                            "windows holding East Africa TimeSync plots within 100 km of the AWF request geometry, "
                            "graded at those plots",
              "prereg_page": os.path.relpath(PLAN, ROOT), "utc": e89.utc_now(), "commit": commit, "dirty": dirty,
              "selection": sel_rec, "crosswalk": tsx.crosswalk_record(),
              "inputs": {"n_grids": len(grids), "crs": sorted({G["crs"].to_string() for G in grids}),
                         "from_olmoearth": {"parts": len(meta), "windows_read": sum((m.get("windows_read") or 0)
                                                                                    for m in meta)},
                         "condition_names": names, "request_period": dict(zip(PROPERTY_NAMES, REQUEST_PERIOD))},
              "plots": {c: {"selected": int((ctry == c).sum()), "in_a_grid": int(((ctry == c) & (gi >= 0)).sum()),
                            "covered": int(((ctry == c) & cov).sum())} for c in countries},
              "overlapping_grids": dict(overlap, rule="a plot in two grids takes the first whose pixel is covered"),
              "cannot_show": list(CANNOT)}
    record["plots"]["total"] = {"selected": int(len(lab)), "in_a_grid": int((gi >= 0).sum()), "covered": int(cov.sum())}
    if inventory:
        record["mode"] = "inventory: counts only; no class or probability at a plot is read out"
        record["seconds"] = round(time.time() - t0, 1)
        return record, None
    if areas is None:
        raise ValueError("the run needs --boundaries: each country's share of the region weights the strata")

    K = np.flatnonzero(cov)
    rd = e98.readings(Pr[K]) if K.size else None
    if K.size and rd["sum_dev"] > e98.SUM_TOLERANCE:
        raise ValueError(f"the probabilities at the plots sum to 1 +- {rd['sum_dev']:.3g}, not within "
                         f"{e98.SUM_TOLERANCE}: not the softmax; nothing is graded")
    pred = np.full(len(lab), -1, np.int64)
    u = np.full(len(lab), np.nan)
    p1 = np.full(len(lab), np.nan)
    if K.size:
        pred[K], u[K], p1[K] = rd["pred"], rd["u"], rd["p1"]
    maj = np.full(len(lab), -1, np.int64)
    maj[K] = e98.majority_3x3(B[K])
    n_by = {c: int((ctry[K] == c).sum()) for c in countries}
    W, w_plot = tsx.weights_from_areas(areas["by_country_km2"], n_by)
    weight = np.array([w_plot[c] if cov[i] else 0.0 for i, c in enumerate(ctry)])
    record["weights"] = {"by_country": W, "per_plot": w_plot,
                         "rule": "W_c, the country's share of the region's area, over n_c, its plots with input"}
    record["readings"] = {k: v for k, v in (rd or {}).items() if k not in ("pred", "u", "p1")}
    miss = np.flatnonzero(~cov)
    record["plots_without_input"] = {
        "n": int(miss.size), "by_country": {c: int((ctry[miss] == c).sum()) for c in countries},
        "by_label": {t: int((lab[miss] == t).sum()) for t in tsx.TIMESYNC_CLASSES if (lab[miss] == t).any()},
        "outside_every_grid": int((gi[miss] < 0).sum()), "in_a_grid_uncovered": int((gi[miss] >= 0).sum()),
        "assumption": CANNOT[3]}

    errs, rules = {}, {}
    for rule in ("strict", "lenient"):
        e = np.full(len(lab), np.nan)
        if K.size:
            e[K] = tsx.wrong(lab[K], pred[K], rule)
        errs[rule] = e
        k_by = {c: int(np.nansum(e[K][ctry[K] == c])) for c in countries}
        r = {"n": int(K.size), "errors": int(np.nansum(e[K])) if K.size else 0, "by_country": {
            c: {"n": n_by[c], "errors": k_by[c]} for c in countries}}
        if K.size:
            r["stratified"] = tsx.stratified_exact(k_by, n_by, areas["by_country_km2"])
            r["ratio_estimate_reported"] = tsx.ratio_estimate(k_by, n_by, areas["country_km2"])
            r["unweighted_rate"] = float(np.nanmean(e[K]))
            r["by_class"] = {t: {"n": int((lab[K] == t).sum()), "errors": int(np.nansum(e[K][lab[K] == t]))}
                             for t in tsx.TIMESYNC_CLASSES if (lab[K] == t).any()}
            cc = C[K]
            r["by_condition"] = {("unrecorded" if c == -1 else "no condition layer" if c == -2 else
                                  (names or {}).get(int(c), str(int(c)))): {
                "code": int(c), "n": int((cc == c).sum()), "errors": int(np.nansum(e[K][cc == c]))}
                for c in sorted(set(cc.tolist()))}
            r["three_by_three_majority_errors"] = int(tsx.wrong(lab[K], maj[K], rule).sum())
            r["ranking"] = ranking(u[K], e[K], weight[K], ctry[K], n_boot)
        rules[rule] = r
    record["rules"] = rules
    if K.size:
        record["p3_transfer_gap"] = gap_call(rules["strict"]["stratified"], rules["lenient"]["stratified"])
        pairs = {}
        for t, p in zip(lab[K], pred[K]):
            key = f"{t} -> {e98.CLASS_NAMES.get(int(p), f'channel {int(p)}')}"
            pairs[key] = pairs.get(key, 0) + 1
        record["confusion_counts"] = dict(sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0])))

    # part D (report-only): where the plots sit among the windows' pixels
    try:
        earlier, unmatched = e98.grid_overlaps(grids)
        q = min(1.0, map_sample / max(sum(G["shape"][0] * G["shape"][1] for G in grids), 1))
        rng = np.random.default_rng(SEED)
        samples, n_map = [], 0
        for G, ear in zip(grids, earlier):
            mp = e98.map_pass(G, q, rng, ear)
            samples.append(mp["sample"])
            n_map += mp["n_covered"]
        sample = np.sort(np.concatenate(samples)) if samples else np.zeros(0)
        d = {"n_window_pixels": int(n_map), "n_sampled": int(sample.size), "grid_overlaps_not_deduplicated": int(unmatched),
             "reading": "the share of the windows' pixels less confident than a plot, averaged over the plots; about "
                        "0.5 if the plots sit in their windows as random pixels do (exp98's P5 asks the same of Ai2's "
                        "expert-placed points)"}
        if K.size and sample.size:
            pct = e98.percentile_in_map(sample, u[K])
            d.update(mean_percentile=float(pct.mean()), mean_percentile_weighted=metrics.weighted_mean(pct, weight[K]),
                     windows_confidence_quantiles={f"{qq:g}": float(np.exp(-np.quantile(sample, 1 - qq)))
                                                   for qq in (0.1, 0.25, 0.5, 0.75, 0.9)},
                     plots_confidence_quantiles={f"{qq:g}": float(np.quantile(p1[K], qq))
                                                 for qq in (0.1, 0.25, 0.5, 0.75, 0.9)})
        record["where_the_plots_sit_reported"] = d
    except Exception as ex:  # noqa: BLE001  (report-only)
        record["where_the_plots_sit_reported"] = {"error": repr(ex), "traceback": traceback.format_exc()}

    # part E (report-only): plan on the region
    if K.size:
        n_region = int(round(areas["region_km2"] * tsx.PIXELS_PER_KM2))
        rates = {"strict": rules["strict"]["stratified"]["estimate"], "lenient": rules["lenient"]["stratified"]["estimate"],
                 "default_0.5": None}
        zones = {rule: {c: zone_error_guess(u[K], errs[rule][K], weight[K], c) for c in PLAN_COVERAGES}
                 for rule in ("strict", "lenient")}
        record["plan_reported"] = plan_block(n_region, rates, zones, zone_draws, max_labels)

    # part F (report-only): the plots inside Ai2's geometry, on this map and on exp98's 2023 map
    try:
        ins = np.flatnonzero(P["inside"] & cov)
        f = {"n_inside": int(P["inside"].sum()), "n_inside_covered": int(ins.size),
             "rules_2017": {rule: int(np.nansum(errs[rule][ins])) for rule in errs}}
        frozen98 = (e98.prereg_status() == "frozen") if exp98_frozen is None else bool(exp98_frozen)
        if exp98_scores and ins.size and not frozen98:
            f["note_2023"] = ("exp98's page is not frozen: its map is not read at the plots (its Part I reads them in "
                              "its own run)")
        elif exp98_scores and ins.size:
            g2, _, _ = read_map(exp98_scores)
            _, P2, _, _, _ = read_plots_on(g2, tsx.subset(P, ins))
            c2 = e98.covered(P2)
            pred23 = e98.readings(np.where(c2[:, None], P2, 0.0))["pred"]
            f["n_inside_covered_both"] = int(c2.sum())
            f["paired"] = {}
            for rule in errs:
                a = errs[rule][ins][c2]
                b = tsx.wrong(lab[ins][c2], pred23[c2], rule)
                f["paired"][rule] = {"both_right": int(((a == 0) & (b == 0)).sum()),
                                     "right_2017_only": int(((a == 0) & (b == 1)).sum()),
                                     "right_2023_only": int(((a == 1) & (b == 0)).sum()),
                                     "both_wrong": int(((a == 1) & (b == 1)).sum()),
                                     "sign_test_two_sided_p": stats.sign_test(int(((a == 0) & (b == 1)).sum()),
                                                                              int(((a == 1) & (b == 0)).sum()))}
            f["same_class_both_years"] = int((pred23[c2] == pred[ins][c2]).sum())
        record["inside_awf_reported"] = f
    except Exception as ex:  # noqa: BLE001  (report-only)
        record["inside_awf_reported"] = {"error": repr(ex), "traceback": traceback.format_exc()}

    s = rules[PRIMARY]
    rk = (s.get("ranking") or {}).get("weighted") or {}
    rl = (rules["lenient"].get("ranking") or {}).get("weighted") or {}
    m = {"n_with_input": int(K.size), "n_errors": s["errors"], "n_correct": int(K.size) - s["errors"],
         "p1_auroc": rk.get("auroc"), "p2_gap_closed": (rk.get("gap_closed") or {}).get(bkey(0.10)),
         "p1_auroc_lenient": rl.get("auroc"), "p2_gap_closed_lenient": (rl.get("gap_closed") or {}).get(bkey(0.10)),
         "p3_call": (record.get("p3_transfer_gap") or {}).get("call")}
    record["prereg"] = grade(m, thresholds, floors)
    record["seconds"] = round(time.time() - t0, 1)
    tidx = {t: i for i, t in enumerate(tsx.TIMESYNC_CLASSES)}
    cidx = {c: i for i, c in enumerate(countries)}
    units = {"country": np.array([cidx[c] for c in ctry], np.int64), "country_names": np.array(countries),
             "label": np.array([tidx[t] for t in lab], np.int64), "label_names": np.array(tsx.TIMESYNC_CLASSES),
             "covered": cov, "inside_awf": np.asarray(P["inside"]), "pred": pred, "suspicion": u, "p1": p1,
             "condition": C, "majority_3x3": maj, "weight": weight,
             "wrong_strict": errs["strict"], "wrong_lenient": errs["lenient"]}
    return record, units


# ----------------------------------------------------------------------------- the synthetic smoke
# A request rectangle of 0.06 x 0.06 degrees whose middle latitude is the border of two synthetic countries, "kenya"
# north and "tanzania" south. Plots: 8 inside (5 and 3), 8 at about 5 km (5, 3), 8 at about 15 km (5, 3), 6 at about
# 25 km, 2 at about 50 km, and 2 of a third country inside. With a floor of 22 plots the rule takes D = 20 km: 24 plots,
# 15 and 9. Each selected plot gets one window of 32 x 32 pixels at 10 m on a 320 m lattice (one part directory each),
# except one Kenya plot (no window) and one Tanzania plot (its pixel NaN); one window is written twice. The 22 plots with
# input are ranked by design: rank r has a top-1 trained probability 0.30 + 0.025 r, background pixels 0.6 (class 4).
SMOKE_RECT = (39.00, -5.03, 39.06, -4.97)
SMOKE_BORDER_LAT = -5.00
SMOKE_MIN_PLOTS = 22
SMOKE_FLOORS = {"min_with_input": 20, "min_errors": 3, "min_correct": 3}
SMOKE_MAX_LABELS = 300                    # plan on a 1.8e7-pixel region is slow; the smoke checks the path, not the budget
# rank: (country, label, predicted channel, ring); ring 0 inside, 1 at ~5 km, 2 at ~15 km
SMOKE_DESIGN = (("kenya", "Cropland", 4, 0), ("tanzania", "Cropland", 4, 0), ("kenya", "Cropland", 4, 1),
                ("tanzania", "Cropland", 4, 1), ("kenya", "Wooded Grassland", 4, 0), ("tanzania", "Wooded Grassland", 4, 1),
                ("kenya", "Wooded Grassland", 4, 1), ("kenya", "Open Grassland", 9, 1), ("kenya", "Open Grassland", 4, 0),
                ("tanzania", "Wooded Grassland", 2, 0), ("kenya", "Cropland", 5, 0), ("kenya", "Otherland", 4, 1),
                ("tanzania", "Open Forest", 2, 0), ("kenya", "Open Grassland", 4, 0), ("kenya", "Wooded Grassland", 2, 1),
                ("tanzania", "Cropland", 5, 1), ("kenya", "Otherland", 4, 2), ("kenya", "Open Forest", 2, 2),
                ("tanzania", "Open Grassland", 4, 2), ("kenya", "Wooded Grassland", 2, 2), ("kenya", "Cropland", 5, 2),
                ("tanzania", "Dense Forest", 6, 2))
SMOKE_NO_WINDOW = ("kenya", "Open Grassland", 2)          # ring 2, no window
SMOKE_NAN_PIXEL = ("tanzania", "Cropland", 2)             # ring 2, its pixel NaN
SMOKE_DUPLICATE_RANK = 10
SMOKE_EXPECTED = {"d_km": 20, "n_selected": 24, "by_country": {"kenya": 15, "tanzania": 9},
                  "inside": {"kenya": 5, "tanzania": 3}, "covered": {"kenya": 14, "tanzania": 8},
                  "strict": {"kenya": 5, "tanzania": 4}, "lenient": {"kenya": 3, "tanzania": 2},
                  "inside_strict_2017": 3, "inside_lenient_2017": 2}


def smoke_probabilities(rank, pred):
    t = 0.30 + 0.025 * rank
    p = np.full(e98.N_OUT, 0.0)
    if pred == tsx.UNTRAINED:                     # predicted as the untrained channel: the trained top-1 is still t
        p[:9] = (1 - t) / 8
        p[0] = t
        p = 0.5 * p
        p[9] = 0.5
    else:
        p[:9] = (1 - t) / 8
        p[pred] = t
    return p


def make_smoke_inputs(root):
    """Synthetic sample, geometry, boundaries, windows (one part directory each) and a '2023' map under root."""
    import pyproj
    import rasterio
    from rasterio.transform import from_origin
    x0, y0, x1, y1 = SMOKE_RECT
    paths = {"timesync": os.path.join(root, "timesync.csv"), "request_geometry": os.path.join(root, "request.geojson"),
             "boundaries": os.path.join(root, "countries.geojson")}
    ring_dlon = {0: None, 1: 0.045, 2: 0.135, 3: 0.225, 4: 0.45}

    def place(ring, k, north):
        lat = (y1 - 0.004 - 0.009 * (k // 2)) if north else (y0 + 0.004 + 0.009 * (k // 2))
        if ring == 0:
            lon = x0 + 0.004 + 0.012 * (k % 5) if north else x0 + 0.006 + 0.016 * (k % 3)
            lat = SMOKE_BORDER_LAT + (0.012 if north else -0.012) + (0.006 if north and k >= 5 else 0)
            return lon, lat
        east = k % 2 == 0
        return (x1 + ring_dlon[ring]) if east else (x0 - ring_dlon[ring]), lat
    plots, counter = [], {}

    def add(ctry, label, ring, rank=None, kind=None):
        north = ctry != "tanzania"
        key = (ring, north)
        k = counter.get(key, 0)
        counter[key] = k + 1
        lon, lat = place(ring, k, north)
        plots.append({"country": ctry, "label": label, "lon": lon, "lat": lat, "rank": rank, "kind": kind, "ring": ring})
    for r, (ctry, label, pred, ring) in enumerate(SMOKE_DESIGN):
        add(ctry, label, ring, rank=r)
    add(*SMOKE_NO_WINDOW, kind="no_window")
    add(*SMOKE_NAN_PIXEL, kind="nan_pixel")
    for k in range(6):
        add("kenya" if k < 3 else "tanzania", "Open Grassland", 3)
    add("kenya", "Cropland", 4)
    add("tanzania", "Cropland", 4)
    rows = []
    for pid, p in enumerate(plots, start=1):
        for y in (2015, 2016, 2017):
            lab = "Open Grassland" if (p["rank"] == 0 and y == 2015) else p["label"]
            rows.append(f"{pid},{y},{p['country']},0,{p['lat']:.10f},{p['lon']:.10f},{lab}")
    for k, pid in enumerate(range(len(plots) + 1, len(plots) + 3)):     # a third country, inside the rectangle
        for y in (2015, 2016, 2017):
            rows.append(f"{pid},{y},ethiopia,1,{SMOKE_BORDER_LAT + 0.02:.10f},{x0 + 0.02 + 0.02 * k:.10f},Cropland")
    with open(paths["timesync"], "w") as f:
        f.write("plotid,year,country,country_id,latitude,longitude,landcover\n" + "\n".join(rows) + "\n")

    def poly(a, b, c, d):
        return {"type": "Polygon", "coordinates": [[[a, b], [c, b], [c, d], [a, d], [a, b]]]}
    with open(paths["request_geometry"], "w") as f:
        json.dump({"features": [{"geometry": poly(x0, y0, x1, y1), "properties": {
            "oe_end_time": "2023-12-31T00:00:00Z", "oe_start_time": "2023-01-01T00:00:00Z"}, "type": "Feature"}],
            "type": "FeatureCollection"}, f)
    with open(paths["boundaries"], "w") as f:
        json.dump({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"ADM0_A3": "KEN"}, "geometry": poly(38.0, SMOKE_BORDER_LAT, 40.0, -4.0)},
            {"type": "Feature", "properties": {"ADM0_A3": "TZA"}, "geometry": poly(38.0, -6.0, 40.0, SMOKE_BORDER_LAT)},
            {"type": "Feature", "properties": {"ADM0_A3": "ETH"}, "geometry": poly(40.0, 5.0, 41.0, 6.0)}]}, f)

    to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32737", always_xy=True)
    bg = np.full(e98.N_OUT, 0.05, np.float32)
    bg[9] = 0.0
    bg[4] = 0.6

    def window(dirname, p, vec, nan=False):
        x, y = to_utm.transform(p["lon"], p["lat"])
        cx, cy = math.floor(x / 320.0) * 320.0, math.ceil(y / 320.0) * 320.0      # a 32-px cell's corner
        a = np.repeat(np.repeat(bg[:, None, None], 32, 1), 32, 2).astype(np.float32)
        j, i = int((x - cx) // 10), int((cy - y) // 10)
        a[:, i, j] = np.nan if nan else vec.astype(np.float32)
        os.makedirs(dirname, exist_ok=True)
        with rasterio.open(os.path.join(dirname, "scores_32737.tif"), "w", driver="GTiff", height=32, width=32,
                           count=e98.N_OUT, dtype="float32", nodata=float("nan"), crs="EPSG:32737",
                           transform=from_origin(cx, cy, 10, 10)) as dst:
            dst.write(a)
        cond = np.zeros((32, 32), np.int32) if p["country"] == "kenya" else np.ones((32, 32), np.int32)
        with rasterio.open(os.path.join(dirname, "condition_32737.tif"), "w", driver="GTiff", height=32, width=32,
                           count=1, dtype="int32", nodata=-1, crs="EPSG:32737", transform=from_origin(cx, cy, 10, 10)) as dst:
            dst.write(cond[None])
        with open(os.path.join(dirname, "olmoearth_conditions.json"), "w") as f:
            json.dump({"codes": {"0": "sentinel2:all:clear", "1": "sentinel2:most:some-cloud"}}, f)
    parts, parts23 = [], []
    for k, p in enumerate(plots):
        if p["rank"] is None and p["kind"] != "nan_pixel":
            continue
        d = os.path.join(root, "scores", f"w{k:03d}")
        vec = smoke_probabilities(p["rank"], SMOKE_DESIGN[p["rank"]][2]) if p["rank"] is not None else bg
        window(d, p, vec, nan=p["kind"] == "nan_pixel")
        parts.append(d)
        if p["rank"] == SMOKE_DUPLICATE_RANK:
            d2 = os.path.join(root, "scores", f"w{k:03d}_again")
            window(d2, p, vec)
            parts.append(d2)
        if p["ring"] == 0 and p["rank"] is not None:          # the '2023' map: right under STRICT at every inside plot
            d3 = os.path.join(root, "scores2023", f"w{k:03d}")
            window(d3, p, smoke_probabilities(p["rank"], tsx.AWF_INDEX[tsx.STRICT[p["label"]]]))
            parts23.append(d3)
    paths.update(scores=parts, exp98_scores=parts23)
    return paths


def smoke(out_dir=None, n_boot=SMOKE_BOOT, log=print):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_smoke_inputs(tmp)
        kw = dict(geometry_sha256=None, fixed=None, min_plots=SMOKE_MIN_PLOTS, log=log)
        _, sel, _ = select(paths["timesync"], paths["request_geometry"], paths["boundaries"], None, SMOKE_MIN_PLOTS,
                           None)
        req = os.path.join(tmp, "request_out")
        written = write_request(req, select(paths["timesync"], paths["request_geometry"], None, None, SMOKE_MIN_PLOTS,
                                            None)[0], paths["request_geometry"])
        inv, _ = analyse(paths["scores"], paths["timesync"], paths["request_geometry"], paths["boundaries"],
                         inventory=True, **kw)
        rec, units = analyse(paths["scores"], paths["timesync"], paths["request_geometry"], paths["boundaries"],
                             exp98_scores=paths["exp98_scores"], n_boot=n_boot, floors=SMOKE_FLOORS, zone_draws=40,
                             max_labels=SMOKE_MAX_LABELS, exp98_frozen=True, **kw)
    rec["mode"] = "smoke: synthetic inputs, floors lowered to " + json.dumps(SMOKE_FLOORS)
    rec["select_smoke"] = sel
    rec["request_written_smoke"] = written
    rec["inventory"] = {k: inv[k] for k in ("plots", "mode", "overlapping_grids")}
    out_dir = out_dir or OUT
    os.makedirs(out_dir, exist_ok=True)
    e98.dump(rec, os.path.join(out_dir, "exp99_summary_smoke.json"))
    np.savez_compressed(os.path.join(out_dir, "exp99_units_smoke.npz"), **units)
    return rec, units


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true", help="synthetic inputs end to end")
    ap.add_argument("--select", action="store_true", help="the area rule only (counts), and with --write-request the "
                    "request geometry of squares around the plots, outside every git checkout")
    ap.add_argument("--inventory", action="store_true", help="counts only; allowed before the page is frozen")
    ap.add_argument("--scores", nargs="+", help="the from-olmoearth part directories of the 2017 run (e99_read.sh)")
    ap.add_argument("--timesync", default=os.path.join(ROOT, "data", "breadth", tsx.TIMESYNC["file"]))
    ap.add_argument("--boundaries", default=None, help="Natural Earth's admin-0 countries (GeoJSON)")
    ap.add_argument("--request-geometry", default=None, help="Ai2's AWF prediction_request_geometry.geojson (the "
                    "pinned olmoearth_projects clone; its sha256 is checked)")
    ap.add_argument("--write-request", default=None, help="with --select: the directory for the request geometries")
    ap.add_argument("--record", default=os.path.join(OUT, "exp99", "e99_select.json"),
                    help="with --select: where the counts go")
    ap.add_argument("--exp98-scores", nargs="+", default=None, help="exp98's 2023 part directories, for Part F")
    ap.add_argument("--out-dir", default=OUT)
    a = ap.parse_args(argv)
    if a.smoke:
        rec, _ = smoke(a.out_dir)
        print(json.dumps(rec["prereg"], indent=1))
        return 0
    if not a.request_geometry:
        ap.error("--request-geometry is required outside --smoke")
    if a.select:
        P, rec, _ = select(a.timesync, a.request_geometry, a.boundaries)
        if a.write_request:
            rec["request_written"] = write_request(a.write_request, P, a.request_geometry)
        rec["utc"] = e89.utc_now()
        e98.dump(rec, a.record)
        print(json.dumps({k: rec[k] for k in ("d_km", "n_selected", "by_country", "inside_request_geometry")}))
        if "areas" in rec:
            print(json.dumps({"weights": rec["areas"]["weights"], "region_km2": round(rec["areas"]["region_km2"], 1),
                              "share_outside_both_countries": rec["areas"]["share_outside_both_countries"]}))
        if "request_written" in rec:
            print(json.dumps({k: v for k, v in rec["request_written"].items() if k.endswith(".geojson")}))
        return 0
    if not a.scores:
        ap.error("--scores is required for --inventory and the run")
    status = prereg_status()
    if not a.inventory and status != "frozen":
        print(f"refused: {os.path.relpath(PLAN, ROOT)} is {status}, not frozen; only --smoke, --select and --inventory "
              "run before it is frozen")
        return 3
    rec, units = analyse(a.scores, a.timesync, a.request_geometry, a.boundaries, inventory=a.inventory,
                         exp98_scores=a.exp98_scores)
    rec["prereg_status"] = status
    os.makedirs(a.out_dir, exist_ok=True)
    if a.inventory:
        e98.dump(rec, os.path.join(a.out_dir, "exp99_inventory.json"))
        print(json.dumps(rec["plots"], indent=1))
        return 0
    e98.dump(rec, os.path.join(a.out_dir, "exp99_summary.json"))
    np.savez_compressed(os.path.join(a.out_dir, "exp99_units.npz"), **units)
    print(json.dumps(rec["prereg"], indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001  (a job log needs the traceback, and the exit code must fail the job)
        traceback.print_exc()
        sys.exit(1)
