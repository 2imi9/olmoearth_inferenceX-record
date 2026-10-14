"""exp92 (the package on LCMAP and Esri land cover against LCMAP's reference sample) by a second route.

From the committed per-plot values (exp/out/exp92_plots.csv) and with code written here: the population counts and
accuracies; the exact coverage and decision rate of the which-map interval at 100 labels by enumerating every
outcome of the multivariate hypergeometric draw, which the recorded Monte Carlo must match within its noise; the
exact coverage of the error-rate interval; and the confidence ranking's AUROC and tie-aware capture by pair counting."""
import csv
import json
import math
import os

import numpy as np

from oe_inferencex import estimate as est

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out")


def _load():
    rows = list(csv.DictReader(open(os.path.join(OUT, "exp92_plots.csv"))))
    s = json.load(open(os.path.join(OUT, "exp92_summary.json")))
    col = lambda k: np.array([int(r[k]) for r in rows])
    return rows, s, col("ref"), col("lcmap"), col("esri")


def _lcomb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def test_exp92_by_a_second_route():
    rows, s, ref, a, b = _load()
    assert len(rows) == 25000 and set(np.unique(ref)) <= set(range(1, 9))
    V = (ref > 0) & (a > 0) & (b > 0)
    N = int(V.sum()); diff = V & (a != b); D = int(diff.sum())
    KA, KB = int(((a == ref) & diff).sum()), int(((b == ref) & diff).sum())
    q = s["Q1"]
    assert (N, D, KA, KB) == (q["N"], q["D"], q["K_A"], q["K_B"]) == (24968, 5387, 2728, 2062)
    assert abs(((a == ref) & V).sum() / N - q["acc_lcmap"]) < 1e-12 and abs(((b == ref) & V).sum() / N - q["acc_esri"]) < 1e-12
    delta = (KA - KB) / N

    # the disagree design at 100 labels, exactly: every (a, b) the draw can give, weighted by its probability
    n, c0 = 100, D - KA - KB
    tab = [est.hypergeom_interval(x, n, D, conf=0.975) for x in range(n + 1)]
    cover = decide_right = total = 0.0
    for x in range(n + 1):
        for y in range(n + 1 - x):
            z = n - x - y
            if x > KA or y > KB or z > c0:
                continue
            p = math.exp(_lcomb(KA, x) + _lcomb(KB, y) + _lcomb(c0, z) - _lcomb(D, n))
            lo, hi = (tab[x][0] - tab[y][1]) * D / N, (tab[x][1] - tab[y][0]) * D / N
            total += p
            cover += p * (lo <= delta <= hi)
            decide_right += p * (lo > 0)
    assert abs(total - 1) < 1e-9
    mc = q["cells"]["disagree_100"]
    se = lambda p: 3 * math.sqrt(p * (1 - p) / 2000) + 1e-3
    assert abs(mc["coverage"] - cover) < se(cover) and abs(mc["decides_right"] - decide_right) < se(decide_right)
    assert cover >= 0.95                                         # the guarantee itself, exactly, at this population

    # the error-rate interval at 300 labels, exactly, for each map
    for name, right in (("lcmap", a == ref), ("esri", b == ref)):
        K = int(N - (right & V).sum())
        exact = est.exact_coverage_srs(N, K, 300, interval=lambda k, n, NN: est.hypergeom_interval(k, n, NN))
        assert exact >= 0.95 and abs(s["Q2"][name]["coverage"] - exact) < se(exact), (name, exact)


def test_exp92_confidence_ranking_by_pair_counting():
    rows, s, ref, a, _ = _load()
    sub = np.array([r["in_conf_subset"] == "1" for r in rows])
    conf = np.array([int(r["lcpconf"]) if r["lcpconf"] else 0 for r in rows])
    g = sub & (ref > 0) & (a > 0) & (conf >= 1) & (conf <= 100)
    score, err = (101 - conf[g]).astype(float), (a != ref)[g]
    q = s["Q3"]["lcpconf"]
    assert (int(g.sum()), int(err.sum())) == (q["n"], q["errors"]) == (4796, 850)
    # AUROC: for each error, the share of correct plots it out-scores, ties counted half
    pos, neg = np.sort(score[err]), np.sort(score[~err])
    below = np.searchsorted(neg, pos, side="left"); ties = np.searchsorted(neg, pos, side="right") - below
    auroc = float((below + 0.5 * ties).sum() / (len(pos) * len(neg)))
    assert abs(auroc - q["auroc"]) < 1e-12
    # capture at 10%: the plots above the k-th score, plus the expected share of the tied block at it
    k = int(round(0.1 * len(score)))
    t = np.sort(score)[::-1][k - 1]
    above, at = score > t, score == t
    expected = err[above].sum() + (k - above.sum()) * err[at].sum() / at.sum()
    assert abs(expected / err.sum() - q["capture_10pct"]) < 1e-12
    assert s["prereg"] == {"P1": {"holds": True}, "P2": {"holds": True}, "P3": {"holds": True}, "P4": {"holds": True}}
