#!/usr/bin/env python
"""exp96: a budget-aware level cut for certify, on maps exp95 did not use (preregistered 8 October 2026).

    python exp/exp96_level_cut_heldout.py run       # reads exp/out/exp79_units (local) and exp95's cells
    python exp/exp96_level_cut_heldout.py smoke

**Why.** exp95 (preregistered, 8baf436; result ad7b858) graded certify's level cut on 34 maps. With kappa 1 (1.7.0) a
grid level of coverage c is tested once budget n reaches b_min / c, b_min = `min_labels_to_certify`: at that budget its
expected labels equal b_min, so it holds fewer than b_min about half the time and cannot pass even with no error, and
the prefix rule must pass it before any larger zone. The chance of certifying then falls when labels are added (the
largest fall exceeded 0.05 of the map on 14 of 34 cells). kappa 3 (a level tested once it expects 3 b_min labels)
removed most large falls and certified more from about 300 labels, but certified less at about 100 labels on 17 of 34
cells, since no level then expects 3 b_min labels. The preregistered rule named kappa 3; found after the result, its
small-budget cost made the default unchanged.

**The cut tested here, designed after exp95 ('ramp').** The smallest coverage tested at budget n is
    c_min(n) = b_min / n          for n <= 3 b_min      (kappa 1, as 1.7.0)
             = 1 / 3              for 3 b_min <= n <= 9 b_min   (no level enters while labels accrue)
             = 3 b_min / n        for n >= 9 b_min      (kappa 3)
It is continuous and never rises with n, so a level once tested stays tested; it equals kappa 1 at small budgets by
construction. It depends on the budget only, so the rules keep their guarantee.

**Cells.** Held out: the frozen-probe maps of the 15 encoders of exp79 other than OlmoEarth Base (exp/out/exp79_units,
local, not committed; the summary records each file's sha256), 332 task-encoder cells, in exp80's zone order (margin,
ties by index). Not held out, reported apart: exp95's 34 cells. alpha half the cell's error rate, delta 0.1, the
package's grid; budgets round(50 x 1.05^i) from 50 to 3,000 and below half the cell's units; 400 nested draws per cell
(a draw is one random order of the units, the sample at budget n its first n); both rules as certify applies them.

**Predictions** (on the 332 held-out cells; SE = sqrt(0.09 / 400) = 0.015):
- P1: every cut, rule, cell and budget has a violation rate at most 0.1 + 5 SE = 0.175 (the guarantee; 5 SE because
  about 160,000 rates are read).
- P2: ramp's prefix and Bonferroni decisions equal kappa 1's at every budget up to 3 b_min, on every cell (a check of
  the construction).
- P3: on the cells where kappa 1's prefix largest fall exceeds 0.05, ramp halves it on at least 75%, and ramp's area
  is lower than kappa 1's by more than 0.02 on at most 10% of all cells.
- P4: at the budget nearest 100, ramp's mean certified coverage is at least kappa 3's minus 0.02 on at least 90% of the
  cells where that budget is checked.

**Decision, fixed now.** If P1 and P3 hold, the package's cut becomes ramp, for both rules; otherwise it stays kappa 1
and the fall is documented. exp80, exp93 and exp94 keep kappa 1 explicitly, so their records reproduce.

**What it does not show.** The held-out maps are other encoders on the same 24 tasks, so their errors share the
tasks' labels and structure; product maps were seen in exp95. Budgets above 3,000, alpha other than half the error
rate, and the threshold grid of exp93 are not graded.
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "exp"))
from oe_inferencex import estimate as est  # noqa: E402
import exp95_level_cut as e95  # noqa: E402

OUT = os.path.join(ROOT, "exp", "out")
UNITS79 = os.path.join(OUT, "exp79_units")
SUMMARY = os.path.join(OUT, "exp96_summary.json")
SEED, DRAWS, DELTA = 96, 400, est.ZONE_DELTA
RULES = ("prefix", "bonferroni")
SE = math.sqrt(DELTA * (1 - DELTA) / DRAWS)
BOUND = DELTA + 5 * SE


def c_min_k1(n, b):
    return b / n


def c_min_k3(n, b):
    return 3 * b / n


def c_min_ramp(n, b):
    if n <= 3 * b:
        return b / n
    if n <= 9 * b:
        return 1 / 3
    return 3 * b / n


CUTS = {"k1": c_min_k1, "k3": c_min_k3, "ramp": c_min_ramp}


def heldout_cells():
    """{encoder/task: errors in zone order} and {file: sha256} for the 15 encoders other than OlmoEarth Base."""
    cells, shas = {}, {}
    for enc in sorted(os.listdir(UNITS79)):
        if enc == "olmoearth_base" or not os.path.isdir(os.path.join(UNITS79, enc)):
            continue
        for f in sorted(os.listdir(os.path.join(UNITS79, enc))):
            if not f.endswith(".npz"):
                continue
            path = os.path.join(UNITS79, enc, f)
            shas[f"{enc}/{f}"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
            d = np.load(path, allow_pickle=False)
            order, _ = est.zone_order(d["margin"].astype(np.float64))
            cells[f"{enc}/{f[:-4]}"] = d["err"].astype(np.float64)[order]
    return cells, shas


def run_cell(err_o, draws=DRAWS, seed=SEED, cuts=CUTS, grid=est.ZONE_GRID):
    """exp95's harness with the level cut as a function of (budget, b_min)."""
    N = err_o.size
    alpha = float(err_o.mean() / 2)
    b_min = est.min_labels_to_certify(alpha, DELTA)
    sizes = np.array([max(1, int(round(c * N))) for c in grid])
    risk = np.cumsum(err_o)[sizes - 1] / sizes
    bs = e95.budgets(N)
    out = {"N": int(N), "error_rate": float(err_o.mean()), "alpha": alpha, "b_min": b_min, "budgets": bs, "cut": {}}
    if not bs:
        return out
    nmax, G = max(bs), len(grid)
    rng = np.random.default_rng(seed)
    B = np.zeros((draws, len(bs), G), np.int32)
    K = np.zeros((draws, len(bs), G), np.int32)
    idx = np.array(bs) - 1
    for d in range(draws):
        pos = rng.choice(N, nmax, replace=False)
        inside = pos[:, None] < sizes[None, :]
        B[d] = np.cumsum(inside, axis=0)[idx]
        K[d] = np.cumsum(err_o[pos][:, None] * inside, axis=0)[idx]
    pv = np.ones((draws, len(bs), G))
    for g in range(G):
        bg, kg, size = B[:, :, g], K[:, :, g], int(sizes[g])
        K0 = math.floor(alpha * size) + 1
        for b in np.unique(bg):
            if b == 0:
                continue
            m = bg == b
            pv[:, :, g][m] = 0.0 if K0 > size else e95._cdf(size, K0, int(b), int(kg[m].max()))[kg[m]]

    def decide(t, g_list, level):
        lv = float(est._level(level))
        p = pv[:, t, g_list]
        acc = p <= lv
        for d, i in np.argwhere(np.abs(p - lv) <= 1e-6 * lv):
            g = g_list[i]
            b, k, size = int(B[d, t, g]), int(K[d, t, g]), int(sizes[g])
            acc[d, i] = est._at_most(est.zone_pvalue(k, b, size, alpha), level,
                                     lambda: est.zone_pvalue_exact(k, b, size, alpha), size)
        return acc
    for name, cut in cuts.items():
        res = {}
        for rule in RULES:
            C, viol, cov_by = np.zeros(len(bs)), np.zeros(len(bs)), []
            for t, n in enumerate(bs):
                c0 = cut(n, b_min)
                lv = [j for j, c in enumerate(grid) if c >= c0 - 1e-12]
                if not lv:
                    cov_by.append(np.zeros(draws))
                    continue
                acc = decide(t, lv, DELTA if rule == "prefix" else est._level(DELTA) / len(lv))
                if rule == "prefix":
                    acc = np.cumprod(acc, axis=1).astype(bool)
                got = acc.any(axis=1)
                last = np.where(acc, np.arange(len(lv))[None, :], -1).max(axis=1)
                best = np.where(got, np.array(lv)[np.maximum(last, 0)], 0)
                cov = np.where(got, np.array(grid)[best], 0.0)
                cov_by.append(cov)
                C[t] = cov.mean()
                viol[t] = float((got & (risk[best] > alpha)).mean())
            falls = [C[i] - C[i + 1:].min() for i in range(len(bs) - 1)]
            res[rule] = {"mean_coverage": C.tolist(), "violation": viol.tolist(),
                         "largest_fall": float(max(falls)) if falls else 0.0, "area": float(C.mean()),
                         "max_violation": float(viol.max()), "_cov": cov_by}
        out["cut"][name] = res
    # P2's check: ramp's decisions equal kappa 1's, draw by draw, at every budget up to 3 b_min
    small = [t for t, n in enumerate(bs) if n <= 3 * b_min]
    out["ramp_equals_k1_small"] = all(np.array_equal(out["cut"]["ramp"][r]["_cov"][t], out["cut"]["k1"][r]["_cov"][t])
                                      for r in RULES for t in small)
    for name in out["cut"]:
        for r in RULES:
            del out["cut"][name][r]["_cov"]
    return out


def _near(r, cut, budget):
    bs = r["budgets"]
    i = int(np.argmin([abs(b - budget) for b in bs]))
    return bs[i], r["cut"][cut]["prefix"]["mean_coverage"][i]


def analyze(held):
    rows = {k: v for k, v in held.items() if v["budgets"]}
    viol = max(r["cut"][c][rule]["max_violation"] for r in rows.values() for c in r["cut"] for rule in RULES)
    over3 = sum(v > DELTA + 3 * SE for r in rows.values() for c in r["cut"] for rule in RULES
                for v in r["cut"][c][rule]["violation"])
    big = [k for k, r in rows.items() if r["cut"]["k1"]["prefix"]["largest_fall"] > 0.05]
    halved = [k for k in big if rows[k]["cut"]["ramp"]["prefix"]["largest_fall"]
              <= rows[k]["cut"]["k1"]["prefix"]["largest_fall"] / 2]
    losing = [k for k, r in rows.items() if r["cut"]["k1"]["prefix"]["area"] - r["cut"]["ramp"]["prefix"]["area"] > 0.02]
    near100 = [(k, _near(r, "ramp", 100)[1], _near(r, "k3", 100)[1]) for k, r in rows.items()
               if abs(_near(r, "ramp", 100)[0] - 100) <= 5]
    p4_ok = [k for k, a, b in near100 if a >= b - 0.02]
    return {
        "P1": {"holds": viol <= BOUND, "max_violation": viol, "bound": BOUND, "rates_above_3se": int(over3)},
        "P2": {"holds": all(r["ramp_equals_k1_small"] for r in rows.values())},
        "P3": {"holds": bool(big) and len(halved) >= 0.75 * len(big) and len(losing) <= 0.10 * len(rows),
               "cells_with_k1_fall_above_0.05": len(big), "halved": len(halved),
               "cells_losing_area_above_0.02": len(losing), "of": len(rows)},
        "P4": {"holds": bool(near100) and len(p4_ok) >= 0.9 * len(near100), "ok": len(p4_ok), "of": len(near100)},
    }


def main(argv):
    if argv[:1] == ["smoke"]:
        rng = np.random.default_rng(0)
        err = (rng.random(6000) < np.linspace(0.005, 0.4, 6000)).astype(float)
        r = run_cell(err, draws=40)
        print({c: (round(v["prefix"]["largest_fall"], 3), round(v["prefix"]["area"], 3)) for c, v in r["cut"].items()},
              "ramp = k1 at small budgets:", r["ramp_equals_k1_small"])
        print("smoke ok")
        return
    cells, shas = heldout_cells()
    held = {}
    for name, err_o in cells.items():
        held[name] = run_cell(err_o)
        c = held[name]["cut"]
        if held[name]["budgets"]:
            print(f"{name:52s} N={held[name]['N']:7d} " + " ".join(
                f"{k}: fall {c[k]['prefix']['largest_fall']:.3f} area {c[k]['prefix']['area']:.3f}" for k in c), flush=True)
    seen = {name: run_cell(err_o) for name, err_o in e95.cells().items()}
    out = {"experiment": "exp96", "preregistered": True, "draws": DRAWS, "delta": DELTA, "input_sha256": shas,
           "heldout": held, "exp95_cells": seen, "prereg": analyze(held), "exp95_cells_summary": analyze(seen)}
    json.dump(out, open(SUMMARY, "w"), default=float)
    print(json.dumps(out["prereg"], indent=1))
    print("exp95 cells (not held out):", json.dumps(out["exp95_cells_summary"]))


if __name__ == "__main__":
    main(sys.argv[1:])
