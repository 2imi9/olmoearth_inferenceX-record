"""exp94 (the package on four more reference samples) by a second route.

From the committed per-point values (exp/out/exp94_*_points.csv) and with the preregistered crosswalks restated here
rather than imported: the population counts of every which-map pair; the exact coverage of the disagree design at
100 labels by enumerating every outcome of the draw, which the recorded Monte Carlo must match within its noise; the
exact coverage of every error-rate interval; every confidence ranking's AUROC by pair counting; and that no output
holds a coordinate."""
import csv
import json
import math
import os

import numpy as np
import pytest

from oe_inferencex import estimate as est

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out")
SOURCES = ("nlcd", "gfc", "eastafrica", "europe")

# the preregistration's crosswalks (b27443e), written out again
NLCD8 = {**dict.fromkeys((21, 22, 23, 24), 1), 81: 2, 82: 2, 52: 3, 71: 3, 41: 4, 42: 4, 43: 4, 11: 5, 90: 6, 95: 6,
         12: 7, 31: 8}


def _rows(src):
    return list(csv.DictReader(open(os.path.join(OUT, f"exp94_{src}_points.csv"))))


def _col(rows, k, missing=0):
    return np.array([int(float(r[k])) if r[k] != "" else missing for r in rows])


def _summary():
    return json.load(open(os.path.join(OUT, "exp94_summary.json")))


def _lcomb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _pairs():
    """{summary key: (reference, product A, product B)} for every which-map pair, from the points."""
    out = {}
    nl = _rows("nlcd")
    for s, y in (("A", 2001), ("A", 2006), ("A", 2011), ("B", 2011), ("B", 2016)):
        rs = [r for r in nl if r["set"] == s and int(r["year"]) == y]
        ref = np.array([NLCD8.get(v, 0) for v in _col(rs, "ref_nlcd")])
        a = np.array([NLCD8.get(v, 0) for v in _col(rs, "nlcd")])
        out[f"nlcd_{s}{y}:nlcd_vs_lcmap"] = (ref, a, _col(rs, "lcmap"))
    g = _rows("gfc")
    out["gfc2020:gfc_vs_worldcover"] = (_col(g, "ref_forest") + 1, _col(g, "gfc_v2") + 1, _col(g, "worldcover_forest", -1) + 1)
    ea = [r for r in _rows("eastafrica") if r["year"] == "2017"]
    out["eastafrica_2017:cgls_vs_esri"] = (_col(ea, "ref"), _col(ea, "cgls"), _col(ea, "esri"))
    eu = _rows("europe")
    out["europe_2017:odse_vs_esri"] = (_col(eu, "ref"), _col(eu, "odse"), _col(eu, "esri"))
    return out


def test_no_output_holds_a_coordinate():
    for src in SOURCES:
        header = next(csv.reader(open(os.path.join(OUT, f"exp94_{src}_points.csv"))))
        bad = [h for h in header if h.lower() in ("x", "y") or any(w in h.lower() for w in ("lat", "lon", "coord", "geom",
                                                                                          "center", "point_"))]
        assert not bad, (src, bad)


@pytest.mark.parametrize("key", ["nlcd_A2001:nlcd_vs_lcmap", "nlcd_A2006:nlcd_vs_lcmap", "nlcd_A2011:nlcd_vs_lcmap",
                                 "nlcd_B2011:nlcd_vs_lcmap", "nlcd_B2016:nlcd_vs_lcmap", "gfc2020:gfc_vs_worldcover",
                                 "eastafrica_2017:cgls_vs_esri", "europe_2017:odse_vs_esri"])
def test_exp94_which_map_by_exact_enumeration(key):
    ref, a, b = _pairs()[key]
    q = _summary()["Q1"][key]
    V = (ref > 0) & (a > 0) & (b > 0)
    N = int(V.sum()); diff = V & (a != b); D = int(diff.sum())
    KA, KB = int(((a == ref) & diff).sum()), int(((b == ref) & diff).sum())
    assert (N, D, KA, KB) == (q["N"], q["D"], q["K_A"], q["K_B"])
    delta = (KA - KB) / N
    n, c0 = 100, D - KA - KB
    tab = [est.hypergeom_interval(x, n, D, conf=0.975) for x in range(n + 1)]
    cover = total = 0.0
    for x in range(min(n, KA) + 1):
        for y in range(min(n - x, KB) + 1):
            z = n - x - y
            if z > c0:
                continue
            p = math.exp(_lcomb(KA, x) + _lcomb(KB, y) + _lcomb(c0, z) - _lcomb(D, n))
            lo, hi = (tab[x][0] - tab[y][1]) * D / N, (tab[x][1] - tab[y][0]) * D / N
            total += p
            cover += p * (lo <= delta + 1e-12 and delta - 1e-12 <= hi)
    assert abs(total - 1) < 1e-9
    assert cover >= 0.95                                         # the guarantee itself, exactly, on this population
    assert abs(q["cells"]["disagree_100"]["coverage"] - cover) < 3 * math.sqrt(cover * (1 - cover) / 2000) + 1e-3


def test_exp94_error_rate_intervals_cover_exactly():
    s = _summary()["Q2"]
    for key, q in s.items():
        exact = est.exact_coverage_srs(q["N"], q["errors"], q["labels"],
                                       interval=lambda k, n, NN: est.hypergeom_interval(k, n, NN))
        assert exact >= 0.95, (key, exact)
        assert abs(q["coverage"] - exact) < 3 * math.sqrt(exact * (1 - exact) / 2000) + 1e-3, (key, exact)


def _confidences():
    """{summary key: (confidence, wrong)} for every Q3 cell, with the preregistered inclusion rules."""
    out = {}
    nl = _rows("nlcd")
    for s, y in (("A", 2001), ("A", 2006), ("A", 2011), ("B", 2011), ("B", 2016)):
        rs = [r for r in nl if r["set"] == s and int(r["year"]) == y]
        ref = np.array([NLCD8.get(v, 0) for v in _col(rs, "ref_nlcd")]); lc = _col(rs, "lcmap"); c = _col(rs, "lcpconf")
        g = (ref > 0) & (lc > 0) & (c >= 1) & (c <= 100)
        out[f"nlcd_{s}{y}:lcmap_lcpconf"] = (c[g], (lc != ref)[g])
    ea = _rows("eastafrica")
    for y in (2015, 2016, 2017):
        rs = [r for r in ea if int(r["year"]) == y]
        ref, cg, pb = _col(rs, "ref"), _col(rs, "cgls"), _col(rs, "cgls_proba", 255)
        g = (ref > 0) & (cg > 0) & (pb <= 100)
        out[f"eastafrica_{y}:cgls_proba"] = (pb[g], (cg != ref)[g])
    eu = _rows("europe")
    ref, od, pr = _col(eu, "ref"), _col(eu, "odse"), _col(eu, "odse_prob", -1)
    g = (_col(eu, "in_subset") == 1) & (ref > 0) & (od > 0) & (pr >= 0) & (pr <= 100)
    out["europe_2017:odse_prob"] = (pr[g], (od != ref)[g])
    return out


def test_exp94_confidence_rankings_by_pair_counting():
    s = _summary()
    cells = _confidences()
    assert set(cells) == set(s["Q3"]) == set(s["Q4"])
    for key, (conf, err) in cells.items():
        q = s["Q3"][key]
        assert (len(conf), int(err.sum())) == (q["n"], q["errors"]), key
        score = (101 - conf).astype(float)
        pos, neg = np.sort(score[err]), np.sort(score[~err])
        below = np.searchsorted(neg, pos, side="left"); ties = np.searchsorted(neg, pos, side="right") - below
        auroc = float((below + 0.5 * ties).sum() / (len(pos) * len(neg)))
        assert abs(auroc - q["auroc"]) < 1e-12, key
        assert q["auroc_ci95"][0] > 0.5, key                     # P4, cell by cell
        assert abs(s["Q4"][key]["alpha"] - err.mean() / 2) < 1e-12, key


def test_exp94_predictions_as_recorded():
    s = _summary()
    bound = 0.1 + 3 * math.sqrt(0.1 * 0.9 / 2000)
    assert all(q[r]["violation_rate"] <= bound for q in s["Q4"].values() for r in ("prefix", "bonferroni"))
    assert all(q["cells"][f"disagree_{n}"]["coverage"] >= 0.94 for q in s["Q1"].values() for n in (50, 100, 200))
    assert all(q["cells"]["disagree_100"]["median_width_points"] < q["cells"]["random_100"]["median_width_points"]
               for q in s["Q1"].values())
    assert s["prereg"] == {p: {"holds": True} for p in ("P1", "P2", "P3", "P4", "P5")}
