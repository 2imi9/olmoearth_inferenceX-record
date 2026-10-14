"""exp81: what a map user is owed per class. User's accuracy, producer's accuracy and the error-adjusted class share,
from the same labelled sample, graded by Monte Carlo on tasks where every unit is labelled.
Preregistered in docs/plan/per_class_assessment.md.

    python exp/exp81_per_class.py --stage estimate [--units DIR] [--encoder NAME] [--tasks ...] [--draws 2000]
    python exp/exp81_per_class.py --stage grade
    python exp/exp81_per_class.py --stage smoke

Numpy only. Reads per-unit exports that carry the reference class per unit (exp78's for the classification tasks;
exp79's for segmentation, whose exp78 export wrote no reference class). For every task, design in {random,
confidence} and budget B in {300, 1000}: R draws; on each draw `estimate.estimate_per_class` runs on the revealed
labels and every per-class interval is graded against the population truth. Coverage is reported conditional on
the class holding at least MIN_PER_CLASS labelled windows on that draw, because that is the condition under which
the package reports the interval without a warning; the share of draws meeting it is beside the coverage.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, EXP_DIR)
from oe_inferencex import estimate as est   # noqa: E402
import exp70_task_suite as e70              # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
SUMMARY = os.path.join(OUT, "exp81_summary.json")
R_DRAWS, BUDGETS, HEADLINE = 2000, (300, 1000), 300
DESIGNS = ("random", "confidence")
INTERVALS = ("wilson", "wald")   # the shipped form and the field's convention, both graded (amendment of 2026-09-23)
QUANTITIES = ("user_accuracy", "producer_accuracy", "reference_share")
# the preregistered bars, docs/plan/per_class_assessment.md
P1_COVER, P1_BIAS = 0.93, 0.02          # exp78's P1 bar; bias ratio within 2% where the class has >= 30 labels
P2_MIN_SHARE = 0.5                      # the confidence design widens the median per-class UA on at least half the tasks
P3_HIGH, P3_LOW, P3_SE_HI, P3_SE_LO = 0.80, 0.20, 3.0, 1.0


def load_units_with_reference(task, units_dir):
    d = np.load(os.path.join(units_dir, f"{task}.npz"), allow_pickle=False)
    y, dec, margin, p1 = d["y"].astype(int), d["dec"].astype(int), d["margin"].astype(float), d["p1"].astype(float)
    if np.unique(y).size < 2:
        raise ValueError(f"{task}: the export carries no reference class per unit (all {np.unique(y).tolist()})")
    return {"y": y, "dec": dec, "margin": margin, "p1": p1, "family": str(d["family"][0])}


def truth_per_class(y, dec, C):
    N = y.size
    N_map, N_ref = np.bincount(dec, minlength=C), np.bincount(y, minlength=C)
    both = np.bincount(dec[dec == y], minlength=C)
    with np.errstate(invalid="ignore", divide="ignore"):
        ua = np.where(N_map > 0, both / N_map, np.nan)
        pa = np.where(N_ref > 0, both / N_ref, np.nan)
    return {"user_accuracy": ua, "producer_accuracy": pa, "reference_share": N_ref / N, "map_share": N_map / N,
            "overall_accuracy": float((y == dec).mean())}


def run_cell(u, C, design, B, draws, seed=0, interval="wilson"):
    y, dec, margin, p1 = u["y"], u["dec"], u["margin"], u["p1"]
    T = truth_per_class(y, dec, C)
    N_map, wrong_in_map = np.bincount(dec, minlength=C), np.bincount(dec[dec != y], minlength=C)
    all_sums = {q: np.zeros(C) for q in QUANTITIES}
    all_n = {q: np.zeros(C) for q in QUANTITIES}
    cov = {q: np.zeros(C) for q in QUANTITIES}
    eligible = {q: np.zeros(C) for q in QUANTITIES}
    sums = {q: np.zeros(C) for q in QUANTITIES}
    halfw = {q: [[] for _ in range(C)] for q in QUANTITIES}
    overall_cov = 0
    for r in range(draws):
        s = est.sample_for_estimation(margin, B, design=design, p1=p1, seed=seed + r)
        out = est.estimate_per_class(s, y[s["indices"]], dec, n_classes=C, interval=interval)
        o = out["overall_accuracy"]
        overall_cov += o["low"] <= T["overall_accuracy"] <= o["high"]
        for c in range(C):
            row = out["per_class"][c]
            n_ok = {"user_accuracy": row["n_labelled_map_class"], "producer_accuracy": row["n_labelled_reference_class"],
                    "reference_share": row["n_labelled_reference_class"]}
            for q in QUANTITIES:
                v = row[q]
                if v is not None and np.isfinite(T[q][c]):          # bias over every draw, unconditional
                    all_sums[q][c] += v["estimate"]
                    all_n[q][c] += 1
                # the share's interval exists on every draw and its coverage is graded on every draw: conditioning
                # on the reference count conditions on the estimate itself (audit, 2026-09-23)
                gate = est.MIN_PER_CLASS if q != "reference_share" else 0
                if v is None or n_ok[q] < gate or not np.isfinite(T[q][c]):
                    continue
                eligible[q][c] += 1
                cov[q][c] += v["low"] <= T[q][c] <= v["high"]
                sums[q][c] += v["estimate"]
                halfw[q][c].append((v["high"] - v["low"]) / 2)
    per_class = {}
    for c in range(C):
        per_class[c] = {"map_share": float(T["map_share"][c]), "reference_share_true": float(T["reference_share"][c]),
                        "expected_labels": float(B * T["map_share"][c]), "n_map_true": int(N_map[c]),
                        "n_wrong_in_map_class_true": int(wrong_in_map[c])}
        for q in QUANTITIES:
            e = eligible[q][c]
            per_class[c][q] = {"truth": None if not np.isfinite(T[q][c]) else float(T[q][c]),
                               "eligible_share": float(e / draws), "coverage": float(cov[q][c] / e) if e else None,
                               "bias_ratio": float((all_sums[q][c] / all_n[q][c]) / T[q][c]) if all_n[q][c] and np.isfinite(T[q][c]) and T[q][c] > 0 else None,
                               "bias_ratio_conditional": float((sums[q][c] / e) / T[q][c]) if e and np.isfinite(T[q][c]) and T[q][c] > 0 else None,
                               "median_half_width": float(np.median(halfw[q][c])) if halfw[q][c] else None}
    return {"design": design, "budget": B, "interval": interval, "n_draws": draws, "overall_coverage": overall_cov / draws,
            "per_class": per_class}


def estimate_task(task, units_dir, draws, designs=DESIGNS):
    u = load_units_with_reference(task, units_dir)
    C = int(max(u["y"].max(), u["dec"].max()) + 1)
    N = int(u["y"].size)
    row = {"n_units": N, "n_classes": C, "family": u["family"], "cells": []}
    for design in designs:
        for B in BUDGETS:
            if B >= N:
                continue
            for interval in INTERVALS:
                t0 = time.time()
                cell = run_cell(u, C, design, B, draws, interval=interval)
                cell["seconds"] = round(time.time() - t0, 1)
                row["cells"].append(cell)
                covs = [cell["per_class"][c]["user_accuracy"]["coverage"] for c in range(C)
                        if cell["per_class"][c]["user_accuracy"]["coverage"] is not None]
                print(f"  {task:44s} {design:10s} B={B:5d} {interval:6s} overall cov {cell['overall_coverage']:.3f} "
                      f"UA cov min {min(covs) if covs else float('nan'):.3f} over {len(covs)} classes {cell['seconds']:.0f}s", flush=True)
    return row


# ----------------------------------------------------------------------------- verdicts, pure functions of the rows
def _cells(rows, design=None, budget=None, interval="wilson"):
    for t, row in rows.items():
        for c in row["cells"]:
            if (design is None or c["design"] == design) and (budget is None or c["budget"] == budget) \
                    and (interval is None or c.get("interval", "wald") == interval):
                yield t, c


def exact_ua_coverage(N, N_c, K_c, B, min_labels=est.MIN_PER_CLASS, max_class=5000):
    """A diagnostic, never an exemption (sixth amendment): the exact coverage of the shipped random-design user's
    accuracy interval (`est.hypergeom_interval`) for a class of N_c windows holding K_c errors, averaged over the
    hypergeometric count n_c of labelled windows the map calls c, conditional on n_c >= min_labels. It says whether
    the Monte Carlo agrees with the enumeration. It cannot pass a cell: a rule that passed cells matching their own
    exact coverage passed every deterministic defect, and hid the finite-population Wilson defect for a night.
    Classes above `max_class` windows return nan (the enumeration's cost grows with n_c times K_c)."""
    import math
    if N_c > max_class:
        return float("nan")
    lc = lambda n, r: math.lgamma(n + 1) - math.lgamma(r + 1) - math.lgamma(n - r + 1)
    lo, hi = max(min_labels, B - (N - N_c)), min(B, N_c)
    if hi < lo:
        return float("nan")
    w = np.array([math.exp(lc(N_c, m) + lc(N - N_c, B - m) - lc(N, B)) for m in range(lo, hi + 1)])
    cov = np.array([est.exact_coverage_srs(N_c, K_c, m, interval=est.hypergeom_interval) for m in range(lo, hi + 1)])
    return float((w * cov).sum() / w.sum())


def grade_p1(rows, cover=P1_COVER, bias=P1_BIAS, interval="wilson"):
    """P1: every per-class interval that the package reports without a warning covers at >= cover, and its
    estimate is unbiased to within `bias` over all draws, on every (task, design, budget, class, quantity) cell.
    Sixth amendment (2026-09-23): no cell passes by matching its own exact coverage; a failing random-design
    user's-accuracy cell carries that coverage as a diagnostic. The random-design user's accuracy is the exact
    hypergeometric interval under both interval options, so the Wald block does not grade it."""
    fails, n = [], 0
    for t, c in _cells(rows, interval=interval):
        N = rows[t]["n_units"]
        for k, pc in c["per_class"].items():
            for q in QUANTITIES:
                v = pc[q]
                if v["coverage"] is None or v["eligible_share"] < 0.5:
                    continue
                if interval == "wald" and c["design"] == "random" and q == "user_accuracy":
                    continue
                # the share's coverage is over every draw (second amendment); the cells graded are the ones the
                # package reports without a warning, i.e. an expected labelled count of at least MIN_PER_CLASS in
                # the reference class, since the third run showed that grading every class sweeps in classes the
                # map hardly ever predicts, whose share estimate is a handful of windows
                if q == "reference_share" and c["budget"] * pc["reference_share_true"] < est.MIN_PER_CLASS:
                    continue
                n += 1
                bad_bias = v["bias_ratio"] is not None and abs(v["bias_ratio"] - 1) > bias
                low = v["coverage"] < cover
                if low or bad_bias:
                    row = {"task": t, "design": c["design"], "budget": c["budget"], "class": int(k), "quantity": q,
                           "coverage": v["coverage"], "bias_ratio": v["bias_ratio"]}
                    if low and c["design"] == "random" and q == "user_accuracy" and "n_map_true" in pc:
                        row["exact_coverage"] = exact_ua_coverage(N, pc["n_map_true"], pc["n_wrong_in_map_class_true"], c["budget"])
                    fails.append(row)
    return {"holds": n > 0 and not fails, "n_cells": n, "interval": interval, "failing": fails}


def grade_p2(rows, budget=HEADLINE, min_share=P2_MIN_SHARE):
    """P2: the confidence design, allocated for the overall rate, widens the per-class user's accuracy interval:
    the median over classes of the half-width ratio confidence/random exceeds 1 on at least half the tasks."""
    per = {}
    for t, row in rows.items():
        # the shipped interval form only: the audit found this dict silently keeping the Wald cells (appended last)
        cr = {c["design"]: c for c in row["cells"] if c["budget"] == budget and c.get("interval", "wald") == "wilson"}
        if "random" not in cr or "confidence" not in cr:
            continue
        ratios = []
        for k in cr["random"]["per_class"]:
            a, b = cr["random"]["per_class"][k]["user_accuracy"], cr["confidence"]["per_class"][k]["user_accuracy"]
            if a["median_half_width"] and b["median_half_width"]:
                ratios.append(b["median_half_width"] / a["median_half_width"])
        if ratios:
            per[t] = float(np.median(ratios))
    n_wider = sum(1 for v in per.values() if v > 1)
    return {"holds": bool(per) and n_wider >= min_share * len(per), "n_tasks_wider": n_wider, "of": len(per), "median_ratio_per_task": per}


def grade_p3(rows, budget=HEADLINE, design="random", hi=P3_HIGH, lo=P3_LOW, se_hi=P3_SE_HI, se_lo=P3_SE_LO):
    """P3: the error-adjusted share tells the map's share from the truth exactly where the population says it
    should: where |map share - true share| exceeds se_hi expected standard errors, the interval excludes the map
    share on >= hi of draws; where it is below se_lo, on <= lo. Requires the per-draw exclusion rates, recorded
    in `share_excludes_map` by the estimate stage."""
    fails, n = [], 0
    for t, c in _cells(rows, design, budget):
        ex = c.get("share_excludes_map")
        if not ex:
            continue
        for k, v in ex.items():
            # fifth amendment (2026-09-23): the classes graded are the ones the package reports without a warning,
            # an expected reference count of at least MIN_PER_CLASS, the share's own P1 rule; the segmentation run
            # showed a class the reference never holds (cashew class 0, 1.8 expected labels) graded at 6,000
            # standard errors with a zero-variance expected error
            pc = c["per_class"].get(str(k), c["per_class"].get(k, {}))
            if budget * pc.get("reference_share_true", 1.0) < est.MIN_PER_CLASS:
                continue
            if v["disc_over_se"] >= se_hi:
                n += 1
                if v["exclusion_rate"] < hi:
                    fails.append({"task": t, "class": int(k), **v})
            elif v["disc_over_se"] < se_lo:
                n += 1
                if v["exclusion_rate"] > lo:
                    fails.append({"task": t, "class": int(k), **v})
    return {"holds": n > 0 and not fails, "n_cells": n, "failing": fails}


def verdicts(rows):
    out = {"P1_every_reported_interval_is_honest": grade_p1(rows), "P2_the_overall_design_widens_the_classes": grade_p2(rows),
           "P3_the_adjusted_share_moves_where_it_should": grade_p3(rows)}
    if any(c.get("interval") == "wald" for _, c in _cells(rows, interval=None)):
        w = grade_p1(rows, interval="wald")           # the field's convention, graded the same way, for the record
        out["P1_under_the_wald_form"] = {"holds": w["holds"], "n_cells": w["n_cells"], "n_failing": len(w["failing"]),
                                         "min_coverage": min((f["coverage"] for f in w["failing"]), default=None),
                                         "failing": w["failing"]}
    return out


def add_share_exclusion(u, C, row, draws, budget=HEADLINE, design="random"):
    """P3's ingredient: per class, how often the error-adjusted share's interval excludes the map's own share."""
    y, dec, margin, p1 = u["y"], u["dec"], u["margin"], u["p1"]
    N = y.size
    T = truth_per_class(y, dec, C)
    disc = np.abs(T["map_share"] - T["reference_share"])
    se_srs = np.sqrt(np.clip(T["reference_share"] * (1 - T["reference_share"]), 1e-9, None) / budget * max(0.0, 1 - budget / N))
    # the post-stratified share's expected standard error at the expected counts (the estimator's own variance
    # form on population values), which the audit found to be 0.5-0.7 of the simple-random one the page used
    N_map = np.bincount(dec, minlength=C).astype(float)
    W = N_map / N
    n_i = budget * W
    u = np.zeros((C, C))
    for i in range(C):
        if N_map[i] > 0:
            u[i] = np.bincount(y[dec == i], minlength=C) / N_map[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        var = np.where(n_i[:, None] > 1, W[:, None] ** 2 * (1 - n_i / N_map)[:, None] * u * (1 - u) / (n_i - 1)[:, None], 0.0).sum(0)
    se_ps = np.sqrt(np.clip(var, 1e-12, None))
    se = se_ps
    excl = np.zeros(C)
    for r in range(draws):
        s = est.sample_for_estimation(margin, budget, design=design, p1=p1, seed=r)
        out = est.estimate_per_class(s, y[s["indices"]], dec, n_classes=C)
        for c in range(C):
            v = out["per_class"][c]["reference_share"]
            excl[c] += not (v["low"] <= T["map_share"][c] <= v["high"])
    for c in row["cells"]:
        if c["design"] == design and c["budget"] == budget:
            c["share_excludes_map"] = {int(k): {"disc": float(disc[k]), "disc_over_se": float(disc[k] / se[k]),
                                                "disc_over_srs_se": float(disc[k] / se_srs[k]),
                                                "exclusion_rate": float(excl[k] / draws)} for k in range(C)}


# ----------------------------------------------------------------------------- stages
def cmd_estimate(args):
    import exp78_error_rate_estimation as e78
    units_dir = args.units or e78.UNITS
    tasks = args.tasks or (e70.TASKS_CLS + e70.TASKS_SEG)
    designs = tuple(d for d in DESIGNS if d in args.designs)
    rows, t0 = {}, time.time()
    for t in tasks:
        path = os.path.join(units_dir, f"{t}.npz")
        if not os.path.exists(path):
            print(f"  {t}: no export under {units_dir}", flush=True)
            continue
        try:
            rows[t] = estimate_task(t, units_dir, args.draws, designs)
        except ValueError as exc:
            print(f"  {t}: skipped, {exc}", flush=True)
            continue
        rows[t]["units"] = os.path.relpath(units_dir, ROOT)       # per task: a summary can merge exports (audit, 2026-09-23)
        u = load_units_with_reference(t, units_dir)
        if rows[t]["n_units"] > HEADLINE and "random" in designs:
            add_share_exclusion(u, rows[t]["n_classes"], rows[t], args.draws)
    # the graded run is OlmoEarth Base's; another encoder's export writes beside it, and a merge reads the file it
    # writes (until 2026-09-23 a merge read Base's summary whatever the encoder; latent, no other encoder was merged)
    path = SUMMARY if args.encoder == "olmoearth_base" else os.path.join(OUT, "exp81_per_class", f"{args.encoder}.json")
    prev = json.load(open(path)) if os.path.exists(path) and args.merge else {"tasks": {}}
    order = {d: i for i, d in enumerate(DESIGNS)}
    for t, row in rows.items():
        row = json.loads(json.dumps(row, default=float))        # string class keys, as a merged file carries them
        old = prev["tasks"].get(t)
        if old is not None and set(designs) != set(DESIGNS):
            # a partial rerun (sixth amendment): only the named designs' cells are replaced; the others, whose code
            # path the change did not touch, are carried over as run
            kept = [c for c in old["cells"] if c["design"] not in designs]
            row["cells"] = sorted(kept + row["cells"], key=lambda c: (order[c["design"]], c["budget"], INTERVALS.index(c["interval"])))
            row["carried_over_designs"] = sorted({c["design"] for c in kept})
        prev["tasks"][t] = row
    summary = {"experiment": "exp81 what a map user is owed per class", "encoder": args.encoder,
               "units": sorted({r.get("units", "unrecorded") for r in prev["tasks"].values()}),
               "config": {"draws": args.draws, "budgets": BUDGETS, "designs": DESIGNS, "min_per_class": est.MIN_PER_CLASS,
                          "seed": 0, "seconds_last_invocation": round(time.time() - t0)},
               "tasks": prev["tasks"], "prereg": verdicts(prev["tasks"])}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps({k: {kk: v[kk] for kk in v if kk in ("holds", "n_cells", "n_tasks_wider", "of")} for k, v in summary["prereg"].items()}, indent=1))
    print(f"wrote {path}")
    return 0


def cmd_grade(args):
    path = SUMMARY if args.encoder == "olmoearth_base" else os.path.join(OUT, "exp81_per_class", f"{args.encoder}.json")
    d = json.load(open(path))
    d["prereg"] = verdicts(d["tasks"])
    with open(path, "w") as f:
        json.dump(d, f, indent=1, default=float)
    print(json.dumps({k: v.get("holds") for k, v in d["prereg"].items()}, indent=1))
    print(f"wrote {path}")
    return 0


def cmd_smoke(args):
    rng = np.random.default_rng(0)
    N, C = 5000, 4
    margin = rng.random(N)
    dec = rng.integers(0, C, N)
    y = np.where(rng.random(N) < 0.5 + 0.4 * margin, dec, rng.integers(0, C, N))
    u = {"y": y, "dec": dec, "margin": margin, "p1": 0.5 + 0.5 * margin, "family": "synthetic"}
    row = {"n_units": N, "n_classes": C, "family": "synthetic", "cells": []}
    for design in DESIGNS:
        row["cells"].append(run_cell(u, C, design, 300, 200))
    add_share_exclusion(u, C, row, 200)
    v = verdicts({"synthetic": row})
    print(json.dumps({k: vv.get("holds") for k, vv in v.items()}, indent=1))
    # 200 draws grade nothing (SE 0.015); the smoke passes when every reported cell covers at >= 0.85 and is
    # unbiased to 5%, i.e. the code runs and nothing is grossly wrong. P1 is graded at 2,000 draws by the run.
    fails = v["P1_every_reported_interval_is_honest"]["failing"]
    gross = [f for f in fails if f["coverage"] < 0.85 or (f["bias_ratio"] is not None and abs(f["bias_ratio"] - 1) > 0.05)]
    ok = v["P1_every_reported_interval_is_honest"]["n_cells"] > 0 and not gross
    print("smoke", "ok" if ok else "FAILED", json.dumps(gross[:3], default=float))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("estimate", "grade", "smoke"), required=True)
    ap.add_argument("--units", default=None)
    ap.add_argument("--encoder", default="olmoearth_base")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--draws", type=int, default=R_DRAWS)
    ap.add_argument("--merge", action="store_true", help="estimate: keep tasks already in the summary (segmentation added later)")
    ap.add_argument("--designs", nargs="*", default=list(DESIGNS), choices=DESIGNS,
                    help="estimate: the designs to run; with --merge the other designs' cells are carried over (sixth amendment)")
    args = ap.parse_args(argv)
    return {"estimate": cmd_estimate, "grade": cmd_grade, "smoke": cmd_smoke}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
