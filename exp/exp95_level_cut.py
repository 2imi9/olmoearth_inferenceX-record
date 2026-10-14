#!/usr/bin/env python
"""exp95: certify's level cut, so that more labels do not make it certify less (preregistered 8 October 2026).

    python exp/exp95_level_cut.py run       # reads committed per-unit files, downloads nothing
    python exp/exp95_level_cut.py smoke     # synthetic units

**Why.** `plan` found (CHANGELOG, 7 October 2026) that in 1.7.0 the chance that `certify` certifies a zone can fall
when labels are added. A level of the grid is tested once budget x coverage reaches `min_labels_to_certify` (b_min,
the labels a zone needs to be certified with no error among them). It then holds about b_min labels, so it passes
only with almost no error among them: the prefix rule must pass it before any larger zone, and the Bonferroni rule
splits delta over one more level. On a 50% zone wrong 1% of the time (25,000 windows, alpha 0.05, errors spread
evenly) the prefix rule certified it with probability 0.49 at 400 labels and 0.40 at 500, after the 10% level entered
at 450. The candidate fix: test a level only once budget x coverage reaches kappa b_min. The cut depends on the
budget alone, so the rules keep their guarantee; a larger kappa admits small zones later, when they hold enough labels
to pass with a few errors.

**Cells.** 34 real error patterns, all committed: the 24 tasks of the OlmoEarth paper embedding suite with OlmoEarth
Base's frozen-probe errors (exp/out/exp78_units, the units exp80 graded, in exp80's zone order: margin, ties by
index), and the 10 product confidence cells of exp93 and exp94 (LCMAP lcpconf on its own plots and on the NLCD points
in five set-years, CGLS-LC100 proba in East Africa in three years, ODSE-LULC probability in Europe; ties broken by a
permutation, seed 95, as exp93 did). alpha is half the cell's error rate, exp80's and exp93's headline; delta 0.1; the
package's grid (0.05 to 1 in steps of 0.05).

**Draws.** For each cell, 1,000 draws. A draw is one random order of the units; the sample at budget n is its first n
units, so a larger budget adds labels to a smaller one, as a reviewer adds to a sample. Budgets: round(50 x 1.05^i)
from 50 to 3,000 and below half the cell's units. Every kappa and rule reads the same draws. Each rule is the
package's own: zone_counts, zone_pvalue, and the prefix and Bonferroni decisions with certify's exact tie rule.

**Measured, per cell, kappa in {1 (1.7.0), 1.5, 2, 3} and rule:** at each budget, the violation rate (certified zone
wrong more than alpha) and C(n), the mean certified coverage (0 when no zone); the largest fall, max over budgets n <
n' of C(n) - C(n'); the area, the mean of C over the budgets.

**Predictions.**
- P1: every kappa, rule, cell and budget has a violation rate at most delta + 3 sqrt(delta (1 - delta) / 1000) =
  0.128 (the guarantee; the cut depends on the budget only).
- P2: with kappa 1, the prefix rule's largest fall exceeds 0.05 of the map on at least half of the 34 cells (the
  defect is common on real maps, not only on the evenly spread zone plan simulated).
- P3: some kappa in {1.5, 2, 3} halves the prefix rule's largest fall, against kappa 1, on at least 75% of the cells
  where kappa 1's exceeds 0.05, while the median over all cells of (kappa 1's area minus its area) is at most 0.02.

**Decision, fixed now.** The package's default kappa becomes the smallest one meeting P3's two conditions, for both
rules; if none does, it stays 1 and the fall is documented. The Bonferroni rule is measured under the same kappa and
not decided on separately. exp80, exp93 and exp94 keep kappa 1 explicitly, so their records still reproduce.

**What it does not show.** Power on maps other than these 34; budgets above 3,000; alpha other than half the error
rate. A larger kappa means small zones are tested only once they hold more labels: at small budgets the package then
certifies small zones less often, which the area measures.
"""
import csv
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "exp"))
from oe_inferencex import estimate as est  # noqa: E402

OUT = os.path.join(ROOT, "exp", "out")
UNITS = os.path.join(OUT, "exp78_units")
SUMMARY = os.path.join(OUT, "exp95_summary.json")
SEED, DRAWS, DELTA = 95, 1000, est.ZONE_DELTA
KAPPAS = (1.0, 1.5, 2.0, 3.0)
RULES = ("prefix", "bonferroni")
BOUND = DELTA + 3 * math.sqrt(DELTA * (1 - DELTA) / DRAWS)
NLCD8 = {**dict.fromkeys((21, 22, 23, 24), 1), 81: 2, 82: 2, 52: 3, 71: 3, 41: 4, 42: 4, 43: 4, 11: 5, 90: 6, 95: 6,
         12: 7, 31: 8}


def budgets(N):
    out = sorted(set(int(round(50 * 1.05 ** i)) for i in range(200)))
    return [b for b in out if 50 <= b <= 3000 and b < N // 2]


# ----------------------------------------------------------------------------- cells
def _rows(name):
    return list(csv.DictReader(open(os.path.join(OUT, name))))


def _col(rows, k, missing=0):
    return np.array([int(float(r[k])) if r[k] not in ("", None) else missing for r in rows])


def _product_order(conf, wrong):
    """exp93's order: confidence descending, ties broken by a permutation fixed before any draw."""
    perm = np.random.default_rng(SEED).permutation(conf.size)
    c, w = conf[perm].astype(float), wrong[perm].astype(float)
    order, _ = est.zone_order(c)
    return w[order]


def cells():
    """{name: errors in zone order} for the 34 cells."""
    out = {}
    man = json.load(open(os.path.join(UNITS, "manifest.json")))
    for task in man["tasks"]:
        d = np.load(os.path.join(UNITS, f"{task}.npz"), allow_pickle=False)
        margin, err = d["margin"].astype(np.float64), d["err"].astype(np.float64)
        order, _ = est.zone_order(margin)                        # exp80's order: margin, ties by index
        out[f"suite:{task}"] = err[order]
    p92 = _rows("exp92_plots.csv")
    ref, lc = _col(p92, "ref"), _col(p92, "lcmap")
    sub, conf = np.array([r["in_conf_subset"] == "1" for r in p92]), _col(p92, "lcpconf")
    g = sub & (ref > 0) & (lc > 0) & (conf >= 1) & (conf <= 100)
    out["product:lcmap_2018_lcpconf"] = _product_order(conf[g], (lc != ref)[g])
    nl = _rows("exp94_nlcd_points.csv")
    for s, y in (("A", 2001), ("A", 2006), ("A", 2011), ("B", 2011), ("B", 2016)):
        rs = [r for r in nl if r["set"] == s and int(r["year"]) == y]
        r8 = np.array([NLCD8.get(v, 0) for v in _col(rs, "ref_nlcd")])
        lcm, c = _col(rs, "lcmap"), _col(rs, "lcpconf")
        g = (r8 > 0) & (lcm > 0) & (c >= 1) & (c <= 100)
        out[f"product:nlcd_{s}{y}_lcpconf"] = _product_order(c[g], (lcm != r8)[g])
    ea = _rows("exp94_eastafrica_points.csv")
    for y in (2015, 2016, 2017):
        rs = [r for r in ea if int(r["year"]) == y]
        r_, cg, pb = _col(rs, "ref"), _col(rs, "cgls"), _col(rs, "cgls_proba", 255)
        g = (r_ > 0) & (cg > 0) & (pb <= 100)
        out[f"product:eastafrica_{y}_cgls_proba"] = _product_order(pb[g], (cg != r_)[g])
    eu = _rows("exp94_europe_points.csv")
    r_, od, pr = _col(eu, "ref"), _col(eu, "odse"), _col(eu, "odse_prob", -1)
    g = (_col(eu, "in_subset") == 1) & (r_ > 0) & (od > 0) & (pr >= 0) & (pr <= 100)
    out["product:europe_2017_odse_prob"] = _product_order(pr[g], (od != r_)[g])
    return out


# ----------------------------------------------------------------------------- one cell
def levels(N, n, alpha, kappa, grid=est.ZONE_GRID):
    """The grid levels certify tests at budget n under the cut kappa: coverage c with n c >= kappa b_min, as
    zone_levels computes it (kappa 1 is zone_levels itself)."""
    b_min = est.min_labels_to_certify(alpha, DELTA)
    c_min = kappa * b_min / max(int(n), 1)
    return [j for j, c in enumerate(grid) if c >= c_min - 1e-12]


def run_cell(err_o, draws=DRAWS, seed=SEED, kappas=KAPPAS, grid=est.ZONE_GRID):
    N = err_o.size
    alpha = float(err_o.mean() / 2)
    sizes = np.array([max(1, int(round(c * N))) for c in grid])
    cum = np.cumsum(err_o)
    risk = cum[sizes - 1] / sizes
    bs = budgets(N)
    nmax = max(bs)
    rng = np.random.default_rng(seed)
    G = len(grid)
    # per draw and budget, labels and errors inside each grid zone: the first n units of one random order
    B = np.zeros((draws, len(bs), G), np.int32)
    K = np.zeros((draws, len(bs), G), np.int32)
    idx = np.array(bs) - 1
    for d in range(draws):
        pos = rng.choice(N, nmax, replace=False)
        inside = pos[:, None] < sizes[None, :]                     # nmax x G
        e = err_o[pos][:, None] * inside
        B[d] = np.cumsum(inside, axis=0)[idx]
        K[d] = np.cumsum(e, axis=0)[idx]
    # p-values for every (draw, budget, level): one cumulative distribution per level and label count
    pv = np.ones((draws, len(bs), G))
    for g in range(G):
        bg, kg, size = B[:, :, g], K[:, :, g], int(sizes[g])
        K0 = math.floor(alpha * size) + 1
        for b in np.unique(bg):
            if b == 0:
                continue                                           # no label in the zone: p = 1
            m = bg == b
            pv[:, :, g][m] = 0.0 if K0 > size else _cdf(size, K0, int(b), int(kg[m].max()))[kg[m]]

    def decide(t, g_list, level):
        """certify's decision for each draw at budget index t and each level in g_list: p <= level, decided in
        exact arithmetic where the p-value lies within 1e-6 of the level (est._at_most)."""
        lv = float(est._level(level))
        p = pv[:, t, g_list]
        acc = p <= lv
        near = np.argwhere(np.abs(p - lv) <= 1e-6 * lv)
        for d, i in near:
            g = g_list[i]
            b, k, size = int(B[d, t, g]), int(K[d, t, g]), int(sizes[g])
            acc[d, i] = est._at_most(est.zone_pvalue(k, b, size, alpha), level,
                                     lambda: est.zone_pvalue_exact(k, b, size, alpha), size)
        return acc
    out = {"N": int(N), "error_rate": float(err_o.mean()), "alpha": alpha, "budgets": bs,
           "risk_by_level": [float(r) for r in risk], "kappa": {}}
    for kappa in kappas:
        res = {}
        for rule in RULES:
            C, viol, none = np.zeros(len(bs)), np.zeros(len(bs)), np.zeros(len(bs))
            for t, n in enumerate(bs):
                lv = levels(N, n, alpha, kappa, grid)
                if not lv:
                    none[t] = 1.0
                    continue
                acc = decide(t, lv, DELTA if rule == "prefix" else est._level(DELTA) / len(lv))
                if rule == "prefix":
                    acc = np.cumprod(acc, axis=1).astype(bool)        # accepted while every level so far passes
                got = acc.any(axis=1)
                last = np.where(acc, np.arange(len(lv))[None, :], -1).max(axis=1)
                best = np.where(got, np.array(lv)[np.maximum(last, 0)], 0)
                cov = np.where(got, np.array(grid)[best], 0.0)
                C[t], none[t] = cov.mean(), 1 - got.mean()
                viol[t] = float((got & (risk[best] > alpha)).mean())
            falls = [C[i] - C[i + 1:].min() for i in range(len(bs) - 1)]
            res[rule] = {"mean_coverage": C.tolist(), "no_zone": none.tolist(), "violation": viol.tolist(),
                         "largest_fall": float(max(falls)) if falls else 0.0, "area": float(C.mean()),
                         "max_violation": float(viol.max())}
        out["kappa"][f"{kappa:g}"] = res
    return out


def _cdf(n, K, b, kmax):
    """P(X <= k) for k = 0..kmax, X ~ Hypergeom(b drawn from n of which K are marked), from the ratio of successive
    terms; zone_pvalue's value at each k up to rounding (decide redoes the ones near the level exactly)."""
    lo, hi = max(0, b - (n - K)), min(b, K)
    out = np.zeros(kmax + 1)
    if kmax < lo:
        return out
    lead = (math.lgamma(K + 1) - math.lgamma(lo + 1) - math.lgamma(K - lo + 1) + math.lgamma(n - K + 1)
            - math.lgamma(b - lo + 1) - math.lgamma(n - K - b + lo + 1)
            - (math.lgamma(n + 1) - math.lgamma(b + 1) - math.lgamma(n - b + 1)))
    top = min(kmax, hi)
    x = np.arange(lo, top, dtype=np.float64)
    r = np.log(K - x) + np.log(b - x) - np.log(x + 1) - np.log(n - K - b + x + 1)
    lp = lead + np.concatenate([[0.0], np.cumsum(r)])
    out[lo:top + 1] = np.minimum(1.0, np.cumsum(np.exp(lp)))
    out[top + 1:] = 1.0
    return out


def analyze(rows):
    k1 = {c: r["kappa"]["1"]["prefix"] for c, r in rows.items()}
    viol = max(r["kappa"][k][rule]["max_violation"] for r in rows.values() for k in r["kappa"] for rule in RULES)
    big = [c for c in rows if k1[c]["largest_fall"] > 0.05]
    cand = {}
    for kappa in KAPPAS[1:]:
        k = f"{kappa:g}"
        halved = [c for c in big if rows[c]["kappa"][k]["prefix"]["largest_fall"] <= k1[c]["largest_fall"] / 2]
        loss = float(np.median([k1[c]["area"] - rows[c]["kappa"][k]["prefix"]["area"] for c in rows]))
        cand[k] = {"halved": len(halved), "of": len(big), "share_halved": len(halved) / len(big) if big else None,
                   "median_area_loss": loss,
                   "meets_p3": bool(big) and len(halved) >= 0.75 * len(big) and loss <= 0.02}
    chosen = next((k for k in cand if cand[k]["meets_p3"]), None)
    return {"P1": {"holds": viol <= BOUND, "max_violation": viol, "bound": BOUND},
            "P2": {"holds": len(big) >= len(rows) / 2, "cells_with_fall_above_0.05": len(big), "of": len(rows)},
            "P3": {"holds": chosen is not None, "candidates": cand},
            "decision": {"default_kappa": float(chosen) if chosen else 1.0}}


def main(argv):
    if argv[:1] == ["smoke"]:
        rng = np.random.default_rng(0)
        err = (rng.random(6000) < np.linspace(0.005, 0.4, 6000)).astype(float)
        r = run_cell(err, draws=60)
        print({k: {rule: (round(v[rule]["largest_fall"], 3), round(v[rule]["area"], 3)) for rule in RULES}
               for k, v in r["kappa"].items()})
        print("smoke ok")
        return
    rows = {}
    for name, err_o in cells().items():
        rows[name] = run_cell(err_o)
        k = rows[name]["kappa"]
        print(f"{name:48s} N={rows[name]['N']:7d} alpha={rows[name]['alpha']:.3f} "
              + " ".join(f"k{kk}: fall {k[kk]['prefix']['largest_fall']:.3f} area {k[kk]['prefix']['area']:.3f}"
                         for kk in k), flush=True)
    out = {"experiment": "exp95", "preregistered": True, "draws": DRAWS, "kappas": list(KAPPAS), "delta": DELTA,
           "cells": rows, "prereg": analyze(rows)}
    json.dump(out, open(SUMMARY, "w"), indent=1, default=float)
    print(json.dumps(out["prereg"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
