"""exp93 (the certified zone on LCMAP's own confidence) by a second route.

From the committed per-plot values (exp/out/exp92_plots.csv), with code written here and scipy's hypergeometric
distribution rather than the package's sums: the disclosed population facts; for every rule, grid and cell, the
exact probability that a zone wrong more than alpha of the time passes its own test (the first such level for the
prefix rule, the sum over such levels at delta / J for Bonferroni), which bounds the violation probability, must be
at most delta, and must hold the recorded Monte Carlo share within its noise; P4's 0.867 summed exactly; and the
package's public certify_zone, run on the experiment's own draws, certifying what the run recorded."""
import csv
import importlib.util
import json
import math
import os

import numpy as np
import pytest

hypergeom = pytest.importorskip("scipy.stats", reason="the second route counts with scipy's hypergeometric (the geo extra)").hypergeom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out")
DELTA, R = 0.1, 2000


def _summary():
    return json.load(open(os.path.join(OUT, "exp93_summary.json")))


def _population():
    rows = list(csv.DictReader(open(os.path.join(OUT, "exp92_plots.csv"))))
    keep = [i for i, r in enumerate(rows) if r["in_conf_subset"] == "1" and int(r["ref"]) > 0 and int(r["lcmap"]) > 0
            and r["lcpconf"] and 1 <= int(r["lcpconf"]) <= 100]
    keep = np.array(keep)[np.random.default_rng(93).permutation(len(keep))]
    conf = np.array([int(rows[i]["lcpconf"]) for i in keep])
    wrong = np.array([rows[i]["ref"] != rows[i]["lcmap"] for i in keep])
    order = sorted(range(len(keep)), key=lambda j: (-conf[j], j))      # confidence descending, ties by position
    return conf, wrong, np.array(order)


def _pass_probability(N, n, K, B, level, alpha):
    """P(a zone of n plots with K wrong passes its exact test at `level`) when B of the N plots are labelled at random:
    the labels inside the zone are Hypergeom(N, n, B); given b of them, the errors are Hypergeom(n, K, b); the test
    passes when P(X <= k) <= level for X ~ Hypergeom(n, floor(alpha n) + 1, b)."""
    K0 = math.floor(alpha * n) + 1
    if K0 > n:
        return 1.0
    pb = hypergeom(N, n, B)
    total = 0.0
    for b in range(max(0, B - (N - n)), min(B, n) + 1):
        w = pb.pmf(b)
        if w < 1e-16 or b == 0:
            continue
        p = hypergeom(n, K0, b).cdf(np.arange(b + 1))
        ok = np.flatnonzero(p <= level)
        if ok.size:
            total += w * hypergeom(n, K, b).cdf(ok.max())
    return total


def test_exp93_population_facts():
    conf, wrong, order = _population()
    s = _summary()
    assert (conf.size, int(wrong.sum())) == (s["population"]["plots"], s["population"]["errors"]) == (4796, 850)
    assert np.unique(conf).size == 68
    assert [(int((conf == t).sum()), int(wrong[conf == t].sum())) for t in (100, 99, 98, 97)] == \
        [(118, 2), (2646, 197), (332, 60), (212, 40)]
    w = wrong[order]
    cut = lambda c: int(round(c * conf.size))
    assert [round(float(w[:cut(c)].mean()), 3) for c in (0.05, 0.10, 0.55, 0.60, 0.65, 0.70, 0.75)] == \
        [0.046, 0.06, 0.071, 0.077, 0.084, 0.091, 0.102]
    assert cut(0.55) <= int((conf >= 99).sum())                         # every G1 level up to 0.55 is inside the tie block


def test_exp93_violation_is_bounded_exactly_by_delta():
    conf, wrong, order = _population()
    N, w, s = conf.size, wrong[order], _summary()
    worst = 0.0
    for cell in s["cells"]:
        a, B = cell["alpha"], cell["budget"]
        for g, o in cell["grids"].items():
            sizes, risk = o["zone_sizes"], o["zone_risk"]
            assert [round(float(w[:n].mean()), 12) for n in sizes] == [round(r, 12) for r in risk]   # the zone rates
            nulls = [j for j, r in enumerate(risk) if r > a]
            K = [int(w[:sizes[j]].sum()) for j in nulls]
            prefix = _pass_probability(N, sizes[nulls[0]], K[0], B, DELTA, a) if nulls else 0.0
            bonf = sum(_pass_probability(N, sizes[j], k, B, DELTA / len(sizes), a) for j, k in zip(nulls, K))
            for rule, bound in (("prefix", prefix), ("bonferroni", bonf)):
                assert bound <= DELTA + 1e-12, (g, a, B, rule, bound)
                v = o["rules"][rule]["violation_rate"]
                assert v <= bound + 3 * math.sqrt(max(bound, 1e-3) * (1 - bound) / R) + 1e-3, (g, a, B, rule, v, bound)
                worst = max(worst, bound)
    assert worst > 0.04                                                 # the bound is not vacuous somewhere


def test_exp93_p4_summed_exactly():
    conf, wrong, order = _population()
    N, w = conf.size, wrong[order]
    a = wrong.mean() / 2
    n = int(round(0.10 * N))
    passes = _pass_probability(N, n, int(w[:n].sum()), 300, DELTA, a)
    # exactly 0.86753; the preregistration wrote 0.867, truncated rather than rounded, against a tolerance of 0.023
    assert abs((1 - passes) - 0.86753) < 5e-6
    p4 = _summary()["prereg"]["P4"]
    assert p4["expected"] == 0.867 and abs(p4["prefix_no_zone_rate"] - (1 - passes)) < 3 * math.sqrt(passes * (1 - passes) / R)


def test_exp93_the_public_certify_zone_certifies_what_the_run_recorded():
    """certify_zone on the first draws of the headline cells, against run_cell on the same draws."""
    from oe_inferencex import estimate as est
    spec = importlib.util.spec_from_file_location("e93", os.path.join(ROOT, "exp", "exp93_product_zone.py"))
    e93 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e93)
    _, conf, err = e93.population()
    order, _ = est.zone_order(conf)
    N, alpha = conf.size, float(err.mean()) / 2
    g2, values, _ = e93.threshold_grid(conf)
    grids = {"package": (est.ZONE_GRID, None), "thresholds": (g2, values)}
    draws = 15
    for B in (300, 1000):
        cell = e93.run_cell(err[order], N, B, alpha, grids, draws)
        rng = np.random.default_rng(e93.SEED_DRAWS)
        got = {(g, rule): {} for g in grids for rule in est.ZONE_RULES}
        for _ in range(draws):
            idx = order[rng.choice(N, B, replace=False)]
            for g, (grid, _) in grids.items():
                for rule in est.ZONE_RULES:
                    z = est.certify_zone(conf, idx, err[idx], alpha, rule=rule, grid=grid, cut="standard")
                    key = "none" if z["coverage"] is None else str(cell["grids"][g]["levels"].index(z["coverage"]))
                    got[(g, rule)][key] = got[(g, rule)].get(key, 0) + 1
        for (g, rule), counts in got.items():
            rec = dict(cell["grids"][g]["rules"][rule]["pick_counts"])
            none = round(cell["grids"][g]["rules"][rule]["no_zone_rate"] * draws)
            if none:
                rec["none"] = none
            assert counts == rec, (B, g, rule, counts, rec)
