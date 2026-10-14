"""exp80: which part of the map can be trusted, with a guarantee? Preregistered in docs/plan/trust_zone.md.

    python exp/exp80_trust_zone.py --stage estimate [--units DIR] [--encoder NAME] [--tasks ...] [--draws 2000]
    python exp/exp80_trust_zone.py --stage grade
    python exp/exp80_trust_zone.py --stage smoke

Numpy only, no torch, no cluster: it reads the per-unit exports (exp78's for OlmoEarth Base, exp79's for the other
encoders). For every task, alpha in {theta/2, 0.05} and budget B in {100, 300, 1000}: R draws of a simple random
sample of B windows; on each draw the three rules of oe_inferencex.estimate (prefix, Bonferroni, plug-in) pick a
zone from the grid above c_min(alpha, delta, B); the zone's TRUE error rate is read from the population and the
draw is a violation when it exceeds alpha. The verdicts are pure functions of the rows so tests can exercise them.
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
SUMMARY = os.path.join(OUT, "exp80_summary.json")
R_DRAWS, BUDGETS, HEADLINE, DELTA = 2000, (100, 300, 1000), 300, est.ZONE_DELTA
ALPHAS = ("half_theta", 0.05)
RULES = ("prefix", "bonferroni", "plugin")
# the preregistered bars, docs/plan/trust_zone.md
P1_TOL = 3.0                     # violation frequency <= delta + 3 Monte Carlo SEs
P2_STEPS = 2                     # median certified coverage within two grid steps of the arithmetic
P3_MIN_TASKS, P3_VIOL = 18, DELTA  # the plug-in violates on more than delta of draws on >= 18 of 24 tasks


def alpha_value(a, theta):
    return theta / 2 if a == "half_theta" else float(a)


def zone_risk(err_in_order, sizes):
    """The true error rate of every zone, from the population in zone order."""
    cum = np.cumsum(err_in_order)
    s = np.asarray(sizes, int)
    return cum[s - 1] / s


def monotone(risk):
    d = np.diff(np.asarray(risk, float))
    return {"nondecreasing": bool((d >= 0).all()), "n_falls": int((d < 0).sum()),
            "largest_fall": float(-d.min()) if (d < 0).any() else 0.0}


def arithmetic(risk, sizes, N, B, alpha, delta, rule):
    """The rule run once on the expected counts: b = B n_c / N, k = b R(c), both rounded to the nearest integer."""
    b = np.array([int(round(B * n / N)) for n in sizes])
    k = np.array([min(bb, int(round(bb * r))) for bb, r in zip(b, risk)])
    p = [est.zone_pvalue(kk, bb, n, alpha) for kk, bb, n in zip(k, b, sizes)]
    _, best = est.apply_zone_rule(p, b, k, alpha, delta, rule)
    return best


def run_cell(err_ordered, N, B, alpha, delta, draws, rng):
    """One (task, alpha, B): the Monte Carlo over draws, all three rules on the same draw."""
    cov, sizes, c_min = est.zone_levels(N, B, alpha, delta, cut="standard")      # recorded with the standard cut
    risk = zone_risk(err_ordered, sizes) if cov else np.array([])
    out = {"budget": B, "alpha": float(alpha), "c_min": float(c_min), "levels": cov, "zone_sizes": sizes,
           "zone_risk": [float(r) for r in risk], "monotone": monotone(risk) if cov else None,
           "oracle_coverage": float(max([c for c, r in zip(cov, risk) if r <= alpha], default=0.0)), "rules": {}}
    if not cov:
        for rule in RULES:
            out["rules"][rule] = {"note": "no level testable at this budget", "arithmetic": None,
                                  "violation_rate": 0.0, "no_zone_rate": 1.0, "median_coverage": None,
                                  "p10_coverage": None, "n_draws": draws}
        return out
    memo = [dict() for _ in sizes]
    picks = {rule: np.full(draws, -1, int) for rule in RULES}
    for r in range(draws):
        pos = rng.choice(N, B, replace=False)             # positions in zone order ARE a simple random sample
        b, k = est.zone_counts(pos, err_ordered[pos], sizes)
        p = np.empty(len(sizes))
        for j, (bb, kk, n) in enumerate(zip(b, k, sizes)):
            key = (int(bb), int(kk))
            if key not in memo[j]:
                memo[j][key] = est.zone_pvalue(kk, bb, n, alpha)
            p[j] = memo[j][key]
        for rule in RULES:
            _, best = est.apply_zone_rule(p, b, k, alpha, delta, rule)
            picks[rule][r] = -1 if best is None else best
    for rule in RULES:
        pk = picks[rule]
        got = pk >= 0
        viol = got & (risk[np.where(got, pk, 0)] > alpha)
        cs = np.array(cov)[pk[got]] if got.any() else np.array([])
        ar = arithmetic(risk, sizes, N, B, alpha, delta, rule)
        out["rules"][rule] = {"violation_rate": float(viol.mean()), "no_zone_rate": float((~got).mean()),
                              "median_coverage": float(np.median(cs)) if cs.size else None,
                              "p10_coverage": float(np.percentile(cs, 10)) if cs.size else None,
                              "modal_outcome": "zone" if got.mean() >= 0.5 else "no zone",
                              "arithmetic": None if ar is None else float(cov[ar]),
                              "arithmetic_outcome": "no zone" if ar is None else "zone", "n_draws": int(draws)}
    return out


def estimate_task(task, units_dir, draws, seed=0):
    u = e78.load_units(task, units_dir)
    err, margin = u["err"], u["margin"]
    order, _ = est.zone_order(margin)
    err_o = err[order]
    N, theta = int(order.size), float(err.mean())
    row = {"n_units": N, "error_rate": theta, "mean_top1": float(u["p1"].mean()),
           "calibration_gap": float(u["p1"].mean() - (1 - theta)), "family": u["family"], "cells": []}
    for a in ALPHAS:
        alpha = alpha_value(a, theta)
        for B in BUDGETS:
            if B >= N:
                continue
            rng = np.random.default_rng(seed)
            cell = run_cell(err_o, N, B, alpha, DELTA, draws, rng)
            cell["alpha_kind"] = a if isinstance(a, str) else "absolute"
            row["cells"].append(cell)
            c = cell["rules"]
            print(f"  {task:44s} a={alpha:.3f} B={B:5d} cmin {cell['c_min']:.2f} oracle {cell['oracle_coverage']:.2f} "
                  f"prefix viol {c['prefix']['violation_rate']:.3f} cov {c['prefix']['median_coverage']} | "
                  f"bonf viol {c['bonferroni']['violation_rate']:.3f} cov {c['bonferroni']['median_coverage']} | "
                  f"plugin viol {c['plugin']['violation_rate']:.3f}", flush=True)
    return row


# ----------------------------------------------------------------------------- verdicts, pure functions of the rows
def _cells(rows, alpha_kind=None, budget=None):
    for t, row in rows.items():
        for c in row["cells"]:
            if (alpha_kind is None or c["alpha_kind"] == alpha_kind) and (budget is None or c["budget"] == budget):
                yield t, c


def grade_p1(rows, delta=DELTA, draws=R_DRAWS, tol=P1_TOL):
    """P1: Bonferroni never violates above delta + tol SEs on any cell; the prefix rule the same on every cell of
    the tasks whose zone risk is nondecreasing on the grid. The prefix rule's violations on the other tasks are
    reported beside the largest fall in their risk curve, not graded."""
    bound = delta + tol * np.sqrt(delta * (1 - delta) / draws)
    bonf_fail, prefix_fail, prefix_uncovered = [], [], []
    for t, c in _cells(rows):
        if c["monotone"] is None:
            continue
        key = {"task": t, "alpha": c["alpha"], "budget": c["budget"]}
        if c["rules"]["bonferroni"]["violation_rate"] > bound:
            bonf_fail.append({**key, "violation_rate": c["rules"]["bonferroni"]["violation_rate"]})
        v = c["rules"]["prefix"]["violation_rate"]
        if c["monotone"]["nondecreasing"]:
            if v > bound:
                prefix_fail.append({**key, "violation_rate": v})
        elif v > delta:
            prefix_uncovered.append({**key, "violation_rate": v, "largest_fall": c["monotone"]["largest_fall"]})
    n_cells = sum(1 for _ in _cells(rows))
    return {"holds": not bonf_fail and not prefix_fail, "bound": float(bound), "n_cells": n_cells,
            "bonferroni_failing": bonf_fail, "prefix_failing_on_monotone_tasks": prefix_fail,
            "prefix_above_delta_on_non_monotone_tasks": prefix_uncovered,
            "max_violation": {rule: max((c["rules"][rule]["violation_rate"] for _, c in _cells(rows)), default=None)
                              for rule in RULES}}


def grade_p2(rows, budget=HEADLINE, steps=P2_STEPS, grid=est.ZONE_GRID):
    """P2: at the headline budget and alpha = theta/2, for prefix and Bonferroni, the modal outcome equals the
    arithmetic's on every task, and where both give a zone the median coverage is within `steps` grid steps."""
    step = grid[1] - grid[0]
    fails, per = [], {}
    for t, c in _cells(rows, "half_theta", budget):
        for rule in ("prefix", "bonferroni"):
            r = c["rules"][rule]
            outcome_ok = r.get("modal_outcome") == r.get("arithmetic_outcome")
            both = r["median_coverage"] is not None and r["arithmetic"] is not None
            within = (abs(r["median_coverage"] - r["arithmetic"]) <= steps * step + 1e-9) if both else True
            per[f"{t}/{rule}"] = {"modal": r.get("modal_outcome"), "arithmetic": r.get("arithmetic_outcome"),
                                  "median_coverage": r["median_coverage"], "arithmetic_coverage": r["arithmetic"]}
            if not (outcome_ok and within):
                fails.append({"task": t, "rule": rule, **per[f"{t}/{rule}"]})
    return {"holds": bool(per) and not fails, "n_tasks": len({k.split("/")[0] for k in per}), "failing": fails,
            "per_task_rule": per}


def grade_p3(rows, budget=HEADLINE, min_tasks=P3_MIN_TASKS, viol=P3_VIOL):
    """P3: the plug-in violates on more than delta of draws on at least min_tasks tasks at alpha = theta/2, B = 300."""
    rates = {t: c["rules"]["plugin"]["violation_rate"] for t, c in _cells(rows, "half_theta", budget)}
    n = sum(1 for v in rates.values() if v > viol)
    return {"holds": n >= min_tasks, "n_tasks_above_delta": n, "of": len(rates), "min_tasks": min_tasks,
            "plugin_violation_rates": rates}


def descriptive(rows):
    """Per (alpha kind, budget): median over tasks of certified coverage per rule, the no-zone share, and the
    prefix-to-Bonferroni ratio, the price of not assuming monotonicity."""
    out = {}
    for a in ALPHAS:
        ak = a if isinstance(a, str) else "absolute"
        for B in BUDGETS:
            cells = list(_cells(rows, ak, B))
            if not cells:
                continue
            d = {"n_tasks": len(cells), "oracle_median": float(np.median([c["oracle_coverage"] for _, c in cells]))}
            for rule in RULES:
                cov = [c["rules"][rule]["median_coverage"] or 0.0 for _, c in cells]
                d[rule] = {"median_of_median_coverage": float(np.median(cov)),
                           "tasks_with_a_zone_on_most_draws": int(sum(c["rules"][rule]["no_zone_rate"] < 0.5 for _, c in cells)),
                           "max_violation_rate": float(max(c["rules"][rule]["violation_rate"] for _, c in cells))}
            ratio = [((c["rules"]["prefix"]["median_coverage"] or 0.0) / c["rules"]["bonferroni"]["median_coverage"])
                     for _, c in cells if c["rules"]["bonferroni"]["median_coverage"]]
            d["prefix_over_bonferroni_median"] = float(np.median(ratio)) if ratio else None
            out[f"{ak}/B{B}"] = d
    return out


def verdicts(rows):
    return {"P1_validity": grade_p1(rows), "P2_agrees_with_the_arithmetic": grade_p2(rows),
            "P3_the_guarantee_is_worth_having": grade_p3(rows), "descriptive": descriptive(rows)}


# ----------------------------------------------------------------------------- stages
def cmd_estimate(args):
    units_dir = args.units or e78.UNITS
    tasks = args.tasks or (e70.TASKS_CLS + e70.TASKS_SEG)
    rows, t0 = {}, time.time()
    for t in tasks:
        if not os.path.exists(os.path.join(units_dir, f"{t}.npz")):
            print(f"  {t}: no export under {units_dir}", flush=True)
            continue
        rows[t] = estimate_task(t, units_dir, args.draws)
    summary = {"experiment": "exp80 which part of the map can be trusted, with a guarantee",
               "encoder": args.encoder, "units": os.path.relpath(units_dir, ROOT),
               "config": {"draws": args.draws, "budgets": BUDGETS, "alphas": [str(a) for a in ALPHAS], "delta": DELTA,
                          "grid": list(est.ZONE_GRID), "rules": RULES, "seed": 0, "seconds": round(time.time() - t0)},
               "tasks": rows, "prereg": verdicts(rows)}
    path = SUMMARY if args.encoder == "olmoearth_base" else os.path.join(OUT, "exp80_zones", f"{args.encoder}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps({k: {kk: v[kk] for kk in v if kk in ("holds", "n_cells", "n_tasks", "n_tasks_above_delta", "of", "max_violation")}
                      for k, v in summary["prereg"].items() if k != "descriptive"}, indent=1, default=str))
    print(f"wrote {path}")
    return 0


def cmd_grade(args):
    d = json.load(open(SUMMARY))
    d["prereg"] = verdicts(d["tasks"])
    with open(SUMMARY, "w") as f:
        json.dump(d, f, indent=1, default=float)
    print(json.dumps({k: v.get("holds") for k, v in d["prereg"].items() if k != "descriptive"}, indent=1))
    return 0


def cmd_smoke(args):
    """Synthetic: a map whose errors concentrate at low margin, so a zone exists and both guaranteed rules hold."""
    rng = np.random.default_rng(0)
    N = 6000
    margin = rng.random(N)
    err = (rng.random(N) < np.clip(0.6 - margin, 0, 1) ** 2).astype(float)
    order, _ = est.zone_order(margin)
    cell = run_cell(err[order], N, 300, float(err.mean()) / 2, DELTA, 300, rng)
    rows = {"synthetic": {"n_units": N, "error_rate": float(err.mean()), "cells": [dict(cell, alpha_kind="half_theta")]}}
    v = verdicts(rows)
    print(json.dumps({k: vv.get("holds") for k, vv in v.items() if k != "descriptive"}, indent=1))
    ok = v["P1_validity"]["holds"] and cell["rules"]["prefix"]["median_coverage"] is not None
    print("smoke", "ok" if ok else "FAILED", json.dumps(cell["rules"], default=float)[:400])
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("estimate", "grade", "smoke"), required=True)
    ap.add_argument("--units", default=None, help="directory of <task>.npz per-unit exports (default exp78's)")
    ap.add_argument("--encoder", default="olmoearth_base")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--draws", type=int, default=R_DRAWS)
    args = ap.parse_args(argv)
    return {"estimate": cmd_estimate, "grade": cmd_grade, "smoke": cmd_smoke}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
