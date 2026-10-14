"""exp93: a certified zone on a published product. Preregistered on 5 October 2026, before any draw was run.

    uv run --no-sync python exp/exp93_product_zone.py --stage run      # writes exp/out/exp93_summary.json
    uv run --no-sync python exp/exp93_product_zone.py --stage smoke    # synthetic population, no product values

**Question.** exp80 graded the certified zone (`oe_inferencex.estimate.certify_zone`, `oe-inferencex certify`) on
the record's probe maps. exp92 graded the error-rate interval, the comparison of two maps and the ranking on two
published products against LCMAP's random reference sample, and left the zone ungraded. This grades the zone on a
published product: which share of LCMAP Collection 1.3's 2018 map, taken from its most confident plots down by its
own confidence layer, is wrong at most alpha of the time, stated so that the statement fails on at most delta of a
reviewer's random draws, and how much of the map a budget of labels certifies.

**Population.** exp92's confidence subset (exp/out/exp92_plots.csv): of a random 5,000 of LCMAP's 25,000
simple-random reference plots, the 4,796 where the reference and LCMAP have a class and `lcpconf` is a confidence (1
to 100). LCMAP is wrong on 850 of them (theta = 17.72%). As in exp92 the plots stand in for the map, so every draw
is graded exactly against their truth. The values were read, and the ranking statistics recorded, in exp92 (AUROC
0.753). No zone, no draw and no risk curve beyond the facts disclosed below has been computed.

**Zone order.** `lcpconf` descending. The package breaks ties by index; the population is put in the order of a
random permutation with seed 93, fixed before any draw, so ties are broken at random. The CSV's own row order is not
used: within the largest tie block its first and second halves are wrong on 6.1% and 8.8% of plots, so that order
carries information about the errors (probably region) and would decide which tied plots enter a zone.

**Design.** Exactly exp80's: a simple random sample of B plots (positions drawn uniformly in zone order), the
package's exact hypergeometric p-value per level, delta = 0.1, alpha in {theta/2, 0.05}, B in {100, 300, 1000},
headline 300, R = 2,000 draws per (alpha, B), the generator reset to seed 0 per cell, so cells at one budget share
their draws. Each draw is passed through the package's `zone_levels`, `zone_counts`, `zone_pvalue` and
`apply_zone_rule` with the zone sizes given, as `certify_zone` calls them. Rules: prefix (fixed-sequence testing,
the default), Bonferroni (Learn then Test), plug-in (no guarantee). Two grids, both fixed before any label:

- G1, the package's grid: coverages 0.05, 0.10, ..., 1.00, cut at c_min = min_labels_to_certify / B.
- G2, the product's own thresholds: one level per value t that `lcpconf` takes, the set `lcpconf >= t`, with the
  same cut. Each G2 level is a statement in the product's terms ("plots with lcpconf of at least 98"); a G1 level
  inside a tie block is a set the tool returns, not a threshold.

**Disclosed population facts** (from the values and the permutation, before any draw; they fix expectations and
confirm nothing):

| fact | value |
|---|---|
| distinct `lcpconf` values | 68 |
| `lcpconf` = 100 | 118 plots (2.5%), 1.7% wrong |
| `lcpconf` = 99 | 2,646 plots (55.2%), 7.45% wrong; with 100, the first 57.6% of the order, 7.20% wrong |
| `lcpconf` = 98, 97 | 332 plots (18.1% wrong), 212 plots (18.9% wrong) |
| G1 zone error rate, 0.05 to 0.55 | 0.046, 0.060, 0.057, 0.064, 0.063, 0.063, 0.063, 0.064, 0.065, 0.070, 0.071; all inside the 100 and 99 blocks |
| G1 zone error rate, 0.60 to 1.00 | 0.077, 0.084, 0.091, 0.102, 0.114, 0.129, 0.140, 0.156, 0.177 |
| G1 monotone | no: falls of 0.0034 at 0.15 and at most 0.0002 at 0.25, 0.30, 0.35 |
| G1 oracle (largest level with true rate at most alpha) | 0.65 at alpha = theta/2 = 8.86%; 0.05 at alpha = 0.05 |
| G2 levels | 68; the 2.5% level (`lcpconf` = 100) is below c_min in every cell here, so 67 are tested, the first `lcpconf >= 99` |
| G2 oracle | `lcpconf >= 98` (64.6%, 8.37% wrong) at theta/2, the next level 9.04%; at 0.05 only the untestable 2.5% level |
| c_min at theta/2 | 0.25, 0.083, 0.025 at B = 100, 300, 1000 (25 labels) |
| c_min at 0.05 | 0.45, 0.15, 0.045 (45 labels) |

**Predictions.**

P1, validity, from the theorem. For the prefix and Bonferroni rules, on both grids and every (alpha, B) cell (24
rule-grid-cells), the frequency over draws of a certified zone whose true error rate exceeds alpha is at most delta
+ 3 sqrt(delta (1 - delta) / R) = 0.120. Why: both rules are valid on any map (fixed-sequence testing and
Bonferroni over levels fixed before the labels; tests/test_trust_zone.py enumerates small maps). What makes it fail:
a wrong p-value or a zone order that depends on the draw. A P1 failure is a bug and nothing else is graded until it
is fixed.

P2, the run agrees with the arithmetic. On G1 at B = 300 and alpha = theta/2, for the prefix and Bonferroni rules
separately, the modal outcome across draws (zone or no zone) equals the outcome of the rule run once on the expected
counts (exp80's `arithmetic`), and where both give a zone the median certified coverage is within two grid steps
(0.10) of the arithmetic's. Why and what makes it fail: as in exp80; here B/N is 0.06, far from the near-census
cells on which exp80's P2 failed.

P3, the guarantee is worth having. On G1 at B = 300 and alpha = theta/2 the plug-in rule certifies a zone whose true
error rate exceeds alpha on more than delta = 0.10 of draws. Why: the true rate crosses alpha between 0.65 (0.084)
and 0.70 (0.091) and rises slowly beyond (0.102 at 0.75), so the sample rate of a level past the crossing, with 200
or more labels and a standard error near 0.02, falls below alpha on a large share of draws. What makes it fail: a
plug-in that rarely accepts past the crossing.

P4, at the headline cell the shipped rule refuses. On G1 at B = 300 and alpha = theta/2 the prefix rule returns no
zone on more than half of the draws, and its no-zone share is within 3 Monte Carlo standard errors of 0.867. Why:
the first testable level (10% of the map, 480 plots, 29 wrong, 6.0%) holds about 30 labels; its exact p-value is at
most 0.1 only when none of them is wrong (25 or more labels) and the prefix rule stops at the first level it cannot
certify. Summed exactly over the hypergeometric distribution of the labels in that level and of the errors among
them, from the disclosed facts and before any draw, the level passes with probability 0.133, so the rule returns no
zone with probability 0.867. P4 is therefore a consequence of the facts, and the run must reproduce it. Not a
defect: the confident tenth of LCMAP is wrong 6% of the time, too close to 8.9% for 30 labels to tell apart. What
makes it fail: more labels in the first level than the cut implies, or counts taken from the wrong level.

**Descriptive, not predicted.** Per cell, grid and rule: violation share, no-zone share, median and 10th-percentile
certified coverage, the arithmetic, the oracle; for G2 the threshold of the median certified level and how often
each threshold is the one certified; the prefix rule against Bonferroni on each grid.

**Independent check before recording.** tests/test_exp93.py, written after the run: the population facts above
recomputed from the CSV by separate code; for every rule-grid-cell the exact probability that a level whose true
error rate exceeds alpha passes its own test (for the prefix rule, the first such level; for Bonferroni, the sum
over such levels at delta / J), an upper bound on the violation probability that the recorded share must respect
within Monte Carlo noise, and that must itself be at most delta; and the package's public `certify_zone` on draws of
the run, which must return the coverage the run recorded. Then an independent audit of the code and the text.

**What would invalidate the run.** A zone order changed after a draw; theta or a zone's error rate read from the
sample instead of the population; G2 thresholds chosen with the labels; calling a G1 zone inside the tie block a
threshold on `lcpconf`; reading the result as LCMAP's error outside the conterminous United States or 2018.
"""
import argparse
import csv
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
import exp80_trust_zone as e80              # noqa: E402

PLOTS = os.path.join(EXP_DIR, "out", "exp92_plots.csv")
SUMMARY = os.path.join(EXP_DIR, "out", "exp93_summary.json")
SEED_PERM, SEED_DRAWS, R_DRAWS, BUDGETS, HEADLINE, DELTA = 93, 0, 2000, (100, 300, 1000), 300, est.ZONE_DELTA
ALPHAS = ("half_theta", 0.05)
RULES = est.ZONE_RULES
GUARANTEED = ("prefix", "bonferroni")
P1_TOL, P2_STEPS = 3.0, 2


def population(path=PLOTS):
    """The 4,796 plots in the zone's tie order: (plot ids, confidence, wrong), permuted with SEED_PERM."""
    rows = list(csv.DictReader(open(path)))
    col = lambda k: np.array([int(r[k]) if r[k] else 0 for r in rows])
    ref, a, conf = col("ref"), col("lcmap"), col("lcpconf")
    sub = np.array([r["in_conf_subset"] == "1" for r in rows])
    g = np.flatnonzero(sub & (ref > 0) & (a > 0) & (conf >= 1) & (conf <= 100))
    idx = g[np.random.default_rng(SEED_PERM).permutation(g.size)]
    plotid = np.array([int(rows[i]["plotid"]) for i in idx])
    return plotid, conf[idx].astype(float), (a != ref)[idx].astype(float)


def threshold_grid(conf):
    """G2: one coverage per value the confidence takes, the share of the map at or above it, and the values."""
    values = np.unique(conf)[::-1]
    n_t = [int((conf >= t).sum()) for t in values]
    return tuple(n / conf.size for n in n_t), [float(t) for t in values], n_t


def _grid_setup(err_o, N, B, alpha, grid, values=None):
    cov, sizes, c_min = est.zone_levels(N, B, alpha, DELTA, grid, cut="standard")   # recorded with the standard cut
    risk = e80.zone_risk(err_o, sizes) if cov else np.array([])
    thr = None
    if values is not None:                       # G2: the level's threshold, and a check that its size is exact
        full = {round(c, 12): t for c, t in zip(grid, values)}
        thr = [full[round(c, 12)] for c in cov]
    return {"levels": list(cov), "sizes": sizes, "c_min": float(c_min), "risk": risk, "thresholds": thr}


def run_cell(err_o, N, B, alpha, grids, draws, seed=SEED_DRAWS):
    """One (alpha, B): the same draws through every grid and rule. grids: {name: (coverages, threshold values or None)}."""
    setups = {g: _grid_setup(err_o, N, B, alpha, cov, vals) for g, (cov, vals) in grids.items()}
    memo = {g: [dict() for _ in s["sizes"]] for g, s in setups.items()}
    picks = {g: {rule: np.full(draws, -1, int) for rule in RULES} for g in setups}
    rng = np.random.default_rng(seed)
    for r in range(draws):
        pos = rng.choice(N, B, replace=False)         # positions in zone order ARE a simple random sample
        for g, s in setups.items():
            if not s["levels"]:
                continue
            b, k = est.zone_counts(pos, err_o[pos], s["sizes"])
            p = np.empty(len(s["sizes"]))
            for j, (bb, kk, n) in enumerate(zip(b, k, s["sizes"])):
                key = (int(bb), int(kk))
                if key not in memo[g][j]:
                    memo[g][j][key] = est.zone_pvalue(kk, bb, n, alpha)
                p[j] = memo[g][j][key]
            for rule in RULES:
                _, best = est.apply_zone_rule(p, b, k, alpha, DELTA, rule, n=s["sizes"])
                picks[g][rule][r] = -1 if best is None else best
    out = {"budget": B, "alpha": float(alpha), "grids": {}}
    for g, s in setups.items():
        risk, cov = s["risk"], s["levels"]
        o = {"c_min": s["c_min"], "n_levels": len(cov), "levels": cov, "zone_sizes": s["sizes"],
             "zone_risk": [float(x) for x in risk], "thresholds": s["thresholds"],
             "monotone": e80.monotone(risk) if cov else None,
             "oracle_coverage": float(max([c for c, x in zip(cov, risk) if x <= alpha], default=0.0)), "rules": {}}
        for rule in RULES:
            pk = picks[g][rule]
            got = pk >= 0
            viol = got & (risk[np.where(got, pk, 0)] > alpha) if cov else np.zeros(draws, bool)
            cs = np.array(cov)[pk[got]] if got.any() else np.array([])
            ar = e80.arithmetic(risk, s["sizes"], N, B, alpha, DELTA, rule) if cov else None
            row = {"violation_rate": float(viol.mean()), "no_zone_rate": float((~got).mean()),
                   "median_coverage": float(np.median(cs)) if cs.size else None,
                   "p10_coverage": float(np.percentile(cs, 10)) if cs.size else None,
                   "modal_outcome": "zone" if got.mean() >= 0.5 else "no zone",
                   "arithmetic": None if ar is None else float(cov[ar]),
                   "arithmetic_outcome": "no zone" if ar is None else "zone", "n_draws": int(draws),
                   "pick_counts": {str(int(j)): int((pk == j).sum()) for j in np.unique(pk[got])}}
            if s["thresholds"] is not None and cs.size:
                med = int(np.sort(pk[got])[(cs.size - 1) // 2])
                row["median_threshold"] = s["thresholds"][med]
            o["rules"][rule] = row
        out["grids"][g] = o
    return out


# ----------------------------------------------------------------------------- verdicts, pure functions of the cells
def _alpha_kind(cell, theta):
    return "half_theta" if abs(cell["alpha"] - theta / 2) < 1e-12 else "absolute"


def grade_p1(cells, draws=R_DRAWS, tol=P1_TOL):
    bound = DELTA + tol * np.sqrt(DELTA * (1 - DELTA) / draws)
    checked, failing = 0, []
    for c in cells:
        for g, o in c["grids"].items():
            for rule in GUARANTEED:
                checked += 1
                v = o["rules"][rule]["violation_rate"]
                if v > bound:
                    failing.append({"alpha": c["alpha"], "budget": c["budget"], "grid": g, "rule": rule, "violation_rate": v})
    return {"holds": checked == 24 and not failing, "bound": float(bound), "n_checked": checked, "failing": failing,
            "max_violation": {rule: max(c["grids"][g]["rules"][rule]["violation_rate"] for c in cells for g in c["grids"])
                              for rule in RULES}}


def _headline(cells, theta, budget=HEADLINE):
    hits = [c for c in cells if c["budget"] == budget and _alpha_kind(c, theta) == "half_theta"]
    assert len(hits) == 1
    return hits[0]["grids"]["package"]


def grade_p2(cells, theta, steps=P2_STEPS):
    o = _headline(cells, theta)
    step = est.ZONE_GRID[1] - est.ZONE_GRID[0]
    per, ok = {}, True
    for rule in GUARANTEED:
        r = o["rules"][rule]
        same = r["modal_outcome"] == r["arithmetic_outcome"]
        both = r["median_coverage"] is not None and r["arithmetic"] is not None
        within = abs(r["median_coverage"] - r["arithmetic"]) <= steps * step + 1e-9 if both else True
        per[rule] = {"modal": r["modal_outcome"], "arithmetic": r["arithmetic_outcome"],
                     "median_coverage": r["median_coverage"], "arithmetic_coverage": r["arithmetic"]}
        ok = ok and same and within
    return {"holds": bool(ok), "per_rule": per}


def grade_p3(cells, theta):
    v = _headline(cells, theta)["rules"]["plugin"]["violation_rate"]
    return {"holds": v > DELTA, "plugin_violation_rate": v}


P4_NO_ZONE = 0.867          # 1 - P(the first G1 level passes at B = 300, alpha = theta/2), summed exactly before the run


def grade_p4(cells, theta, draws=R_DRAWS, tol=P1_TOL):
    nz = _headline(cells, theta)["rules"]["prefix"]["no_zone_rate"]
    se = np.sqrt(P4_NO_ZONE * (1 - P4_NO_ZONE) / draws)
    return {"holds": bool(nz > 0.5 and abs(nz - P4_NO_ZONE) <= tol * se), "prefix_no_zone_rate": nz,
            "expected": P4_NO_ZONE, "tolerance": float(tol * se)}


def verdicts(cells, theta):
    return {"P1": grade_p1(cells), "P2": grade_p2(cells, theta), "P3": grade_p3(cells, theta), "P4": grade_p4(cells, theta)}


# ----------------------------------------------------------------------------- stages
def evaluate(conf, err, draws, verbose=True):
    order, _ = est.zone_order(conf)
    err_o = err[order]
    N, theta = int(order.size), float(err.mean())
    g2, values, _ = threshold_grid(conf)
    grids = {"package": (est.ZONE_GRID, None), "thresholds": (g2, values)}
    cells = []
    for a in ALPHAS:
        alpha = theta / 2 if a == "half_theta" else float(a)
        for B in BUDGETS:
            cell = run_cell(err_o, N, B, alpha, grids, draws)
            cell["alpha_kind"] = a if isinstance(a, str) else "absolute"
            cells.append(cell)
            if verbose:
                for g, o in cell["grids"].items():
                    rr = o["rules"]
                    print(f"  a={alpha:.4f} B={B:5d} {g:10s} levels {o['n_levels']:2d} oracle {o['oracle_coverage']:.4f} | "
                          + " | ".join(f"{rule} viol {rr[rule]['violation_rate']:.3f} none {rr[rule]['no_zone_rate']:.3f} "
                                       f"med {rr[rule]['median_coverage']}" for rule in RULES), flush=True)
    return N, theta, cells


def cmd_run(args):
    t0 = time.time()
    plotid, conf, err = population()
    assert (conf.size, int(err.sum())) == (4796, 850), (conf.size, int(err.sum()))
    N, theta, cells = evaluate(conf, err, args.draws)
    summary = {"experiment": "exp93", "preregistered": True, "population": {"plots": N, "errors": int(err.sum()),
               "error_rate": theta, "distinct_confidence_values": int(np.unique(conf).size)},
               "config": {"draws": args.draws, "budgets": BUDGETS, "alphas": [str(a) for a in ALPHAS], "delta": DELTA,
                          "rules": RULES, "seed_permutation": SEED_PERM, "seed_draws": SEED_DRAWS,
                          "seconds": round(time.time() - t0)},
               "cells": cells, "prereg": verdicts(cells, theta)}
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps({k: v["holds"] for k, v in summary["prereg"].items()}))
    print(f"wrote {SUMMARY}")
    return 0


def cmd_smoke(args):
    """Synthetic, so no product value is seen: 4,796 plots with integer confidences tied as heavily as a product's,
    errors more frequent at low confidence. The code runs, the grids are built and both guaranteed rules hold."""
    rng = np.random.default_rng(1)
    conf = np.clip(np.round(100 - rng.exponential(6, 4796)), 1, 100)
    err = (rng.random(conf.size) < 0.04 + 0.5 * (100 - conf) / 100).astype(float)
    N, theta, cells = evaluate(conf, err, args.draws or 100)
    p1 = grade_p1(cells, draws=args.draws or 100, tol=5.0)
    print("smoke", "ok" if p1["n_checked"] == 24 else "FAILED", json.dumps(p1["max_violation"]))
    return 0 if p1["n_checked"] == 24 else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("run", "smoke"), required=True)
    ap.add_argument("--draws", type=int, default=None)
    args = ap.parse_args(argv)
    if args.stage == "run":
        args.draws = args.draws or R_DRAWS
    return {"run": cmd_run, "smoke": cmd_smoke}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
