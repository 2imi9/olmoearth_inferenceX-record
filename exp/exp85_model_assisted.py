"""exp85: does the map's own confidence sharpen the error rate once labels exist? The difference estimator and its
tuned (PPI++) and stratified forms beside exp78's arms, on the same export, the same draws and the same bars.
Preregistered in docs/plan/model_assisted_estimation.md.

    python exp/exp85_model_assisted.py --stage estimate [--units DIR] [--tasks ...] [--draws 2000]
    python exp/exp85_model_assisted.py --stage grade
    python exp/exp85_model_assisted.py --stage smoke

Numpy only. The predictor is g = 1 - p1 from the export, known on every window; labels are revealed for the
sampled windows only. Every arm records bias beside coverage.
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
import exp78_error_rate_estimation as e78   # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
SUMMARY = os.path.join(OUT, "exp85_summary.json")
R_DRAWS, B = 2000, 300
# the preregistered bars, docs/plan/model_assisted_estimation.md
P1_COVER, P1_BIAS = 0.93, 0.02
P2_RATIO = 1.00
P3_STRAT_MEDIAN, P3_CLEAN_RATIO, P3_CLEAN_TASKS = 0.95, 0.90, ("mados", "sen1floods11")
P4_MIN_WIDER = 2
P5_SAVING = 1.8


def wald_srs(k, n, N):
    p = k / n
    half = est.Z95 * np.sqrt((1 - n / N) * p * (1 - p) / (n - 1)) if n > 1 else float("nan")
    return max(0.0, p - half), min(1.0, p + half)


def run_arms(err, margin, p1, draws, rng, theta):
    N = err.size
    g = 1.0 - p1
    s = e78.strata_of(margin)
    sizes = np.bincount(s, minlength=e78.N_STRATA)
    q = np.array([(1 - p1[s == h]).mean() if (s == h).any() else 0.0 for h in range(e78.N_STRATA)])
    alloc = e78.neyman(sizes, np.sqrt(np.clip(q * (1 - q), 1e-9, None)), B)
    names = ("D1/E1", "D1/E1w", "D1/E3d", "D1/E3d++", "D2c/E1", "D2c/E3s")
    cov = {k: 0.0 for k in names}; wid = {k: 0.0 for k in names}; tot = {k: 0.0 for k in names}
    lams, lams_h = [], []
    for _ in range(draws):
        i = rng.choice(N, B, replace=False)
        k = int(err[i].sum())
        lo, hi = e78.wilson(k, B, N); r = {"D1/E1": (err[i].mean(), lo, hi)}
        lo, hi = wald_srs(k, B, N); r["D1/E1w"] = (err[i].mean(), lo, hi)
        e_, lo, hi, _ = est.model_assisted_interval(err, g, i, N, lam=1.0); r["D1/E3d"] = (e_, lo, hi)
        e_, lo, hi, lam = est.model_assisted_interval(err, g, i, N); r["D1/E3d++"] = (e_, lo, hi); lams.append(lam)
        j = e78.draw_stratified(rng, s, sizes, alloc)
        e_, lo, hi, _ = e78.stratified_interval(err, s, j, sizes, N); r["D2c/E1"] = (e_, lo, hi)
        e_, lo, hi, lh, _ = est.stratified_model_assisted_interval(err, g, s, j, sizes, N); r["D2c/E3s"] = (e_, lo, hi); lams_h.append(lh)
        for kname, (e_, lo, hi) in r.items():
            cov[kname] += lo <= theta <= hi; wid[kname] += (hi - lo) / 2; tot[kname] += e_
    arms = {k: {"coverage": cov[k] / draws, "half_width": wid[k] / draws, "mean_estimate": tot[k] / draws,
                "bias_ratio": (tot[k] / draws) / theta if theta > 0 else float("nan")} for k in names}
    corr_all = float(np.corrcoef(err, g)[0, 1]) if err.std() > 0 and g.std() > 0 else float("nan")
    within = [float(np.corrcoef(err[s == h], g[s == h])[0, 1]) for h in range(e78.N_STRATA)
              if (s == h).sum() > 2 and err[s == h].std() > 0 and g[s == h].std() > 0]
    lh = np.array([[d.get(h, np.nan) for h in range(e78.N_STRATA)] for d in lams_h])
    return {"arms": arms, "lambda_tuned_median": float(np.median(lams)), "lambda_tuned_iqr": [float(np.percentile(lams, 25)), float(np.percentile(lams, 75))],
            "lambda_per_stratum_median": [float(np.nanmedian(lh[:, h])) if np.isfinite(lh[:, h]).any() else None for h in range(e78.N_STRATA)],
            "corr_error_predictor": corr_all, "corr_within_strata": within,
            "saving_vs_D1E1": {k: float((arms["D1/E1"]["half_width"] / arms[k]["half_width"]) ** 2) for k in names}}


def estimate_task(task, units_dir, draws):
    t0 = time.time()
    u = e78.load_units(task, units_dir)
    err, theta = u["err"], float(u["err"].mean())
    rng = np.random.default_rng(0)
    row = run_arms(err, u["margin"], u["p1"], draws, rng, theta)
    row.update({"n_units": int(err.size), "error_rate": theta, "family": u["family"], "seconds": round(time.time() - t0, 1)})
    a = row["arms"]
    print(f"  {task:44s} theta {theta:.3f} | E1w cov {a['D1/E1w']['coverage']:.3f} hw {a['D1/E1w']['half_width']:.4f} | E3d {a['D1/E3d']['coverage']:.3f} "
          f"{a['D1/E3d']['half_width'] / a['D1/E1w']['half_width']:.3f} | E3d++ {a['D1/E3d++']['coverage']:.3f} {a['D1/E3d++']['half_width'] / a['D1/E1w']['half_width']:.3f} "
          f"lam {row['lambda_tuned_median']:.2f} | E3s/E1 {a['D2c/E3s']['half_width'] / a['D2c/E1']['half_width']:.3f} cov {a['D2c/E3s']['coverage']:.3f} r {row['corr_error_predictor']:.2f} {row['seconds']:.0f}s", flush=True)
    return row


# ----------------------------------------------------------------------------- verdicts
def _seg(rows):
    return {t: r for t, r in rows.items() if r["family"].startswith("seg")}


def grade_p1(rows):
    fails = [(t, k, round(r["arms"][k]["coverage"], 3), round(r["arms"][k]["bias_ratio"], 4)) for t, r in _seg(rows).items() for k in ("D1/E3d++", "D2c/E3s")
             if r["arms"][k]["coverage"] < P1_COVER or abs(r["arms"][k]["bias_ratio"] - 1) > P1_BIAS]
    return {"holds": len(_seg(rows)) == 7 and not fails, "n_tasks": len(_seg(rows)), "failing": fails}


def grade_p2(rows):
    ratios = {t: r["arms"]["D1/E3d++"]["half_width"] / r["arms"]["D1/E1w"]["half_width"] for t, r in _seg(rows).items()}
    return {"holds": len(ratios) == 7 and all(v <= P2_RATIO + 1e-12 for v in ratios.values()), "ratios": ratios, "max": max(ratios.values(), default=None)}


def grade_p3(rows):
    seg = _seg(rows)
    strat = {t: r["arms"]["D2c/E3s"]["half_width"] / r["arms"]["D2c/E1"]["half_width"] for t, r in seg.items()}
    clean = {t: seg[t]["arms"]["D1/E3d++"]["half_width"] / seg[t]["arms"]["D1/E1w"]["half_width"] for t in P3_CLEAN_TASKS if t in seg}
    med = float(np.median(list(strat.values()))) if strat else None
    return {"holds": len(seg) == 7 and med is not None and med <= P3_STRAT_MEDIAN and all(v <= P3_CLEAN_RATIO for v in clean.values()),
            "stratified_median_ratio": med, "stratified_ratios": strat, "clean_task_ratios": clean,
            "verdict_if_median_above_098": "no gain at this ranking quality; not shipped" if med is not None and med > 0.98 else None}


def grade_p4(rows):
    wider = [t for t, r in _seg(rows).items() if r["arms"]["D1/E3d"]["half_width"] > r["arms"]["D1/E1w"]["half_width"]]
    return {"holds": len(_seg(rows)) == 7 and len(wider) >= P4_MIN_WIDER, "n_wider": len(wider), "wider_on": wider}


def grade_p5(rows):
    worst = {t: max(v for k, v in r["saving_vs_D1E1"].items()) for t, r in rows.items()}
    return {"holds": bool(worst) and max(worst.values()) <= P5_SAVING, "max_saving": max(worst.values(), default=None), "per_task": worst}


def verdicts(rows):
    return {"P1_honest": grade_p1(rows), "P2_never_worse_than_classical": grade_p2(rows), "P3_gain_is_small": grade_p3(rows),
            "P4_why_lambda_is_tuned": grade_p4(rows), "P5_honesty_bound": grade_p5(rows)}


# ----------------------------------------------------------------------------- stages
def cmd_estimate(args):
    units_dir = args.units or e78.UNITS
    tasks = args.tasks or (e70.TASKS_SEG + e70.TASKS_CLS)
    rows, t0 = {}, time.time()
    for t in tasks:
        if not os.path.exists(os.path.join(units_dir, f"{t}.npz")):
            print(f"  {t}: no export", flush=True); continue
        u = e78.load_units(t, units_dir)
        if u["err"].size <= 3 * B:
            print(f"  {t}: {u['err'].size} units, too few for B = {B}", flush=True); continue
        rows[t] = estimate_task(t, units_dir, args.draws)
    summary = {"experiment": "exp85 does the map's confidence sharpen the error rate once labels exist", "units": os.path.relpath(units_dir, ROOT),
               "config": {"draws": args.draws, "budget": B, "seed": 0, "predictor": "1 - p1", "seconds": round(time.time() - t0)},
               "tasks": rows, "prereg": verdicts(rows)}
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps({k: {kk: v[kk] for kk in v if kk in ("holds", "max", "stratified_median_ratio", "n_wider", "max_saving")} for k, v in summary["prereg"].items()}, indent=1, default=str))
    print(f"wrote {SUMMARY}")
    return 0


def cmd_grade(args):
    d = json.load(open(SUMMARY))
    d["prereg"] = verdicts(d["tasks"])
    with open(SUMMARY, "w") as f:
        json.dump(d, f, indent=1, default=float)
    print(json.dumps({k: v.get("holds") for k, v in d["prereg"].items()}, indent=1))
    return 0


def cmd_smoke(args):
    rng = np.random.default_rng(0)
    N = 6000
    margin = rng.random(N)
    p1 = 0.5 + 0.5 * margin
    err = (rng.random(N) < 1.2 * (1 - p1)).astype(float)         # errors track the predictor, imperfectly
    row = run_arms(err, margin, p1, 300, rng, float(err.mean()))
    a = row["arms"]
    ok = all(a[k]["coverage"] > 0.9 for k in a) and a["D1/E3d++"]["half_width"] <= a["D1/E1w"]["half_width"] * 1.02 and 0.3 < row["lambda_tuned_median"] < 3
    print("smoke", "ok" if ok else "FAILED", {k: (round(v["coverage"], 3), round(v["half_width"], 4)) for k, v in a.items()}, "lambda", round(row["lambda_tuned_median"], 2))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("estimate", "grade", "smoke"), required=True)
    ap.add_argument("--units", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--draws", type=int, default=R_DRAWS)
    args = ap.parse_args(argv)
    return {"estimate": cmd_estimate, "grade": cmd_grade, "smoke": cmd_smoke}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
