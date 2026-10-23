#!/usr/bin/env python
"""exp115: a per-class report in which every interval is exact (docs/plan/per_class_exact.md; the page is the
specification).

For each of 40 fully labelled maps and each class c: the exact probability, over simple random samples of n units, that
the class-share interval (hypergeometric, Bonferroni over the classes) covers the true share, that all share intervals
cover together, that the producer's-accuracy interval [A_lo / K_hi, min(1, A_hi / K_lo, M_c / K_lo)] covers, that the
share flag fires, and that the producer's-accuracy upper end falls to 0.25 or below. Every probability is a sum over the
sampling distribution (hypergeometric, or the three-category multivariate hypergeometric of the producer's accuracy),
computed in log space; there is no Monte Carlo.

Modes:
    run      every population; refused unless the page is frozen; writes exp/out/exp115_summary.json
    smoke    three populations at n = 100 only; nothing is graded or written to exp/out

    uv run python exp/exp115_per_class_exact.py run --e113-private DIR
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import time

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import binom

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
import oe_inferencex  # noqa: E402
from oe_inferencex import estimate as est  # noqa: E402

PLAN = os.path.join(ROOT, "docs", "plan", "per_class_exact.md")
OUT = os.path.join(EXP_DIR, "out")
BUDGETS = (100, 400, 1000)
GRADED_BUDGET = 400
ALPHA = 0.05
PA_BAR = 0.25
# bars (the page's)
P2_JOINT = 0.95
P3_SHARE, P3_RATIO, P3_POWER = 0.05, 0.5, 0.8
P4_DIFF, P4_FLAG = 0.02, 0.10
P5_SHARE, P5_PA, P5_POWER = 0.05, 0.10, 0.8
PLANTEUR = [f"loio:{t}" for t in ("Guadeloupe", "Martinique", "Guyane", "Réunion", "Mayotte")] + ["default"]
PLANTEUR_EMBS = ("oe13", "tessera2", "aef")
SEGMENTATION_WITH_TRUTH = ("sen1floods11",)        # binary: the truth follows from the map and err
SUITE = ("awf_landsat", "awf_sentinel1", "awf_sentinel2", "breizhcrops", "cropharvest_Peoples_Republic_of_China_6",
         "cropharvest_Peoples_Republic_of_China_6_sentinel1", "cropharvest_Peoples_Republic_of_China_6_sentinel1_sentinel2",
         "cropharvest_Togo_12_sentinel1", "cropharvest_Togo_12_sentinel2", "cropharvest_Togo_12_sentinel2_sentinel1",
         "m_brick_kiln", "m_eurosat", "m_forestnet", "m_so2sat", "nandi_landsat", "nandi_sentinel1", "nandi_sentinel2",
         "sen1floods11")
N_POPULATIONS = 39                                  # the page as amended before the run: 18 + 3 + 18
MASS_TOL = 1e-6                                     # gammaln at N near a million is good to about 1e-9 per term
P1_TOL = 1e-9                                       # floating-point noise allowed below a P1 bar (reported)


def log(*a):
    print(*a, flush=True)


def prereg_status(path=PLAN):
    with open(path, encoding="utf-8") as f:
        return "frozen" if "**Status: frozen" in f.read() else "draft"


# ----------------------------------------------------------------------------- populations
def populations(e113_private):
    """{name: (map class, reference class, number of classes)} for the page's 39 maps, in the page's order. A plot
    with no map class (exp92's code 0) is -1: it stays in the population and counts toward no class."""
    P = {}
    for cfg in PLANTEUR:
        for k in PLANTEUR_EMBS:
            d = np.load(os.path.join(e113_private, f"exp113_windows_{cfg.replace(':', '_')}_{k}.npz"))
            P[f"planteur|{cfg}|{k}"] = (d["pred"].astype(np.int64), d["true"].astype(np.int64), 8)
    with open(os.path.join(OUT, "exp92_plots.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ref = np.array([int(r["ref"]) for r in rows]) - 1
    for col, name in (("lcmap", "lcmap|collection_1_3"), ("esri", "lcmap|esri")):
        m = np.array([int(r[col]) for r in rows]) - 1
        P[name] = (m, ref.copy(), 8)
    with open(os.path.join(OUT, "exp112_plots.csv"), newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["in_fit_1000"] == "0" and r["has_embedding"] == "1"]
    P["lcmap|olmoearth_v1_3"] = (np.array([int(r["pred"]) for r in rows]) - 1, np.array([int(r["ref"]) for r in rows]) - 1, 8)
    for f in sorted(glob.glob(os.path.join(OUT, "exp78_units", "*.npz"))):
        task = os.path.basename(f)[:-4]
        d = np.load(f)
        fam, C = str(d["family"][0]), int(d["n_classes"][0])
        dec, err = d["dec"].astype(np.int64), d["err"].astype(bool)
        if fam == "classification":
            y = d["y"].astype(np.int64)
            if not np.array_equal(err, dec != y):
                raise SystemExit(f"exp115: {task}: err does not equal dec != y")
        elif task in SEGMENTATION_WITH_TRUTH:
            if C != 2:
                raise SystemExit(f"exp115: {task} is not binary")
            y = np.where(err, 1 - dec, dec)
        else:
            continue
        P[f"suite|{task}"] = (dec, y, C)
    for name, (m, r, C) in P.items():
        low = -1 if name in ("lcmap|collection_1_3", "lcmap|esri") else 0
        if m.shape != r.shape or m.min() < low or r.min() < 0 or m.max() >= C or r.max() >= C:
            raise SystemExit(f"exp115: {name}: classes outside 0..{C - 1} or misaligned arrays")
    if len(P) != N_POPULATIONS or sorted(n[6:] for n in P if n.startswith("suite|")) != sorted(SUITE):
        raise SystemExit(f"exp115: {len(P)} populations, not the page's {N_POPULATIONS}, or another suite list")
    return P


# ----------------------------------------------------------------------------- exact probabilities
def lcomb(n, k):
    """log C(n, k), -inf outside 0 <= k <= n; vectorised over k."""
    k = np.asarray(k, dtype=np.float64)
    ok = (k >= 0) & (k <= n)
    out = np.full(k.shape, -np.inf)
    out[ok] = gammaln(n + 1) - gammaln(k[ok] + 1) - gammaln(n - k[ok] + 1)
    return out


_TABLES = {}


def table(n, N, level):
    """The exact interval, as shares, for every count 0..n."""
    key = (n, N, round(level, 12))
    if key not in _TABLES:
        _TABLES[key] = np.array([est.hypergeom_interval(k, n, N, conf=level) for k in range(n + 1)])
    return _TABLES[key]


def hyper_pmf(N, K, n):
    """The hypergeometric pmf over 0..n, normalised: its raw sum is checked to MASS_TOL, then divided out."""
    k = np.arange(n + 1)
    lp = lcomb(K, k) + lcomb(N - K, n - k) - lcomb(N, n)
    tot = logsumexp(lp)
    if abs(math.exp(tot) - 1) > MASS_TOL:
        raise SystemExit(f"exp115: hypergeometric mass {math.exp(tot)} at N {N}, K {K}, n {n}")
    return np.exp(lp - tot)


def counts(tab, N):
    """The interval table as whole population counts (the package's ends are K/N): compared in integers."""
    T = np.rint(tab * N).astype(np.int64)
    if np.abs(tab * N - T).max() > 1e-6:
        raise SystemExit("exp115: an interval end is not a whole count")
    return T


def weighted_median(values, weights):
    o = np.argsort(values)
    c = np.cumsum(weights[o])
    return float(values[o][np.searchsorted(c, 0.5 * c[-1])])


def share_cell(N, K, M, n, tab):
    """Coverage of the share interval, the flag's probabilities and the median width, for one class; the interval's
    ends compared with K and M as whole counts."""
    pmf = hyper_pmf(N, K, n)
    T = counts(tab, N)
    return {"coverage": float(pmf[(T[:, 0] <= K) & (K <= T[:, 1])].sum()),
            "flag_too_little": float(pmf[M < T[:, 0]].sum()), "flag_too_much": float(pmf[M > T[:, 1]].sum()),
            "median_width": weighted_median(tab[:, 1] - tab[:, 0], pmf), "mass": float(pmf.sum())}


def joint_share_coverage(N, K, n, tab):
    """The exact probability that every class's share interval covers its share: the multivariate hypergeometric mass
    of the samples every class accepts. Each class's C(K_c, k) is weighted by q^k (1 - q)^(K_c - k), q = n / N, a factor
    common to every term of the sum (it multiplies to q^n (1 - q)^(N - n)), so the convolution of the classes' binomial
    pmfs over their accepted counts, divided by the binomial pmf of n in N, is the sum, without underflow (the pre-run
    review: the log-scaled form underflowed at N of several hundred thousand)."""
    q = n / N
    T = counts(tab, N)
    k = np.arange(n + 1)
    f = np.zeros(n + 1)
    f[0] = 1.0
    for Kc in K:
        acc = (T[:, 0] <= Kc) & (Kc <= T[:, 1])
        f = np.convolve(f, np.where(acc, binom.pmf(k, int(Kc), q), 0.0))[:n + 1]
    return float(f[n] / binom.pmf(n, N, q))


def pa_cell(N, K, A, M, n, tab2):
    """Coverage of the producer's-accuracy interval and the chance its upper end is at most PA_BAR, summed over the
    three-category multivariate hypergeometric of (a, b): a labelled units that are c and drawn c, b that are c and drawn
    otherwise. The interval's ends are compared in whole counts: lower end A_lo / K_hi, upper end min(1, A_hi / K_lo,
    M / K_lo), 1 when K_lo is 0."""
    a = np.arange(n + 1)[:, None]
    b = np.arange(n + 1)[None, :]
    valid = a + b <= n
    lp = lcomb(A, a) + lcomb(K - A, b) + lcomb(N - K, np.where(valid, n - a - b, -1)) - lcomb(N, n)
    lp = np.where(valid, lp, -np.inf)
    tot = logsumexp(lp)
    mass = float(math.exp(tot))
    if abs(mass - 1) > MASS_TOL:
        raise SystemExit(f"exp115: three-category mass {mass} at N {N}, K {K}, A {A}, n {n}")
    P = np.exp(lp - tot)
    T2 = counts(tab2, N)
    aa = np.broadcast_to(a, P.shape)
    ks = np.minimum(a + b, n)
    A_lo, A_hi = T2[aa, 0], T2[aa, 1]
    K_lo, K_hi = T2[ks, 0], T2[ks, 1]
    lower_ok = (K_hi == 0) | (A_lo * K <= A * K_hi)                      # A_lo / K_hi <= A / K
    upper_ok = (K_lo == 0) | ((A_hi * K >= A * K_lo) & (M * K >= A * K_lo))   # A / K <= min(A_hi, M) / K_lo
    bar = (K_lo > 0) & ((4 * A_hi <= K_lo) | (4 * M <= K_lo))             # min(1, A_hi / K_lo, M / K_lo) <= 0.25
    lo = np.divide(A_lo, K_hi, out=np.zeros(P.shape), where=K_hi > 0)
    hi = np.where(K_lo > 0, np.minimum(1.0, np.minimum(A_hi, M) / np.maximum(K_lo, 1)), 1.0)
    return {"coverage": float(P[valid & lower_ok & upper_ok].sum()), "upper_at_most_bar": float(P[valid & bar].sum()),
            "median_width": weighted_median(np.maximum(hi - lo, 0)[valid], P[valid]), "mass": mass}


def population(name, m, r, C, budgets):
    N = int(m.size)
    K = np.bincount(r, minlength=C)
    M = np.bincount(m[m >= 0], minlength=C)
    A = np.bincount(r[m == r], minlength=C)
    rec = {"N": N, "C": C, "unmapped": int((m < 0).sum()), "classes": {}, "joint_share_coverage": {}}
    for c in range(C):
        rec["classes"][str(c)] = {"K": int(K[c]), "M": int(M[c]), "A": int(A[c]), "p": float(K[c] / N), "m": float(M[c] / N),
                                  "pa": float(A[c] / K[c]) if K[c] else None, "by_n": {}}
    for n in budgets:
        if n >= N:
            rec["joint_share_coverage"][str(n)] = "census"
            continue
        tab = table(n, N, 1 - ALPHA / C)
        tab2 = table(n, N, 1 - ALPHA / (2 * C))
        rec["joint_share_coverage"][str(n)] = joint_share_coverage(N, K, n, tab)
        for c in range(C):
            if K[c] == 0:
                continue
            cell = {"share": share_cell(N, int(K[c]), int(M[c]), n, tab), "pa": pa_cell(N, int(K[c]), int(A[c]), int(M[c]), n, tab2)}
            rec["classes"][str(c)]["by_n"][str(n)] = cell
    return rec


def verdicts(S):
    out = {}
    worst_share, worst_pa, cells = 1.0, 1.0, 0
    for name, rec in S["maps"].items():
        bar = 1 - ALPHA / rec["C"]
        for c, cl in rec["classes"].items():
            for n, cell in cl["by_n"].items():
                cells += 1
                worst_share = min(worst_share, cell["share"]["coverage"] - bar)
                worst_pa = min(worst_pa, cell["pa"]["coverage"] - bar)
    out["P1"] = {"cells": cells, "min_share_coverage_minus_bar": worst_share, "min_pa_coverage_minus_bar": worst_pa,
                 "tolerance": P1_TOL, "holds": worst_share >= -P1_TOL and worst_pa >= -P1_TOL}
    joint = [v for rec in S["maps"].values() for v in rec["joint_share_coverage"].values() if v != "census"]
    out["P2"] = {"cells": len(joint), "min_joint_coverage": min(joint), "holds": min(joint) >= P2_JOINT}
    g = str(GRADED_BUDGET)
    p3, p4, p5 = {}, {}, {}
    for name, rec in S["maps"].items():
        if rec["N"] <= GRADED_BUDGET:
            continue
        for c, cl in rec["classes"].items():
            if g not in cl["by_n"]:
                continue
            cell = cl["by_n"][g]
            key = f"{name}|{c}"
            N, K, M, A = rec["N"], cl["K"], cl["M"], cl["A"]
            if 20 * K >= N and 2 * M <= K:                     # p >= 0.05 and m <= p / 2, in whole counts
                p3[key] = cell["share"]["flag_too_little"]
            if 50 * abs(M - K) <= N:                           # |m - p| <= 0.02
                p4[key] = cell["share"]["flag_too_little"] + cell["share"]["flag_too_much"]
            if 20 * K >= N and 10 * A <= K:                    # p >= 0.05 and PA <= 0.10
                p5[key] = cell["pa"]["upper_at_most_bar"]
    out["P3"] = {"classes": p3, "determined": bool(p3), "min": min(p3.values()) if p3 else None,
                 "holds": all(v >= P3_POWER for v in p3.values()) if p3 else None}
    out["P4"] = {"classes": len(p4), "determined": bool(p4), "max": max(p4.values()) if p4 else None,
                 "above_bar": {k: v for k, v in p4.items() if v > P4_FLAG},
                 "holds": all(v <= P4_FLAG for v in p4.values()) if p4 else None}
    out["P5"] = {"classes": p5, "determined": bool(p5), "min": min(p5.values()) if p5 else None,
                 "holds": all(v >= P5_POWER for v in p5.values()) if p5 else None}
    return out


def run(e113_private, smoke):
    if not smoke and prereg_status() != "frozen":
        raise SystemExit("exp115: the preregistration page is not frozen; refusing to grade")
    P = populations(e113_private)
    names = list(P)
    if smoke:
        names = [n for n in names if n in ("suite|awf_landsat", "lcmap|esri", "planteur|loio:Réunion|oe13")]
    S = {"page": os.path.relpath(PLAN, ROOT), "package": oe_inferencex.__version__, "smoke": smoke,
         "environment": {"numpy": np.__version__, "python": sys.version.split()[0]}, "maps": {}}
    for name in names:
        t0 = time.time()
        m, r, C = P[name]
        S["maps"][name] = population(name, m, r, C, (100, 1000) if smoke else BUDGETS)
        log(f"[exp115] {name}: N {m.size}, {C} classes, {time.time() - t0:.1f} s")
    if not smoke:
        S["verdicts"] = verdicts(S)
    return S


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("run", "smoke"))
    ap.add_argument("--e113-private", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    S = run(a.e113_private, a.mode == "smoke")
    if a.mode == "smoke":
        json.dumps(S, allow_nan=False)
        log(f"[exp115] smoke done in {time.time() - t0:.0f} s")
        return
    with open(os.path.join(OUT, "exp115_summary.json"), "w", encoding="utf-8") as f:
        json.dump(S, f, indent=1, ensure_ascii=False, allow_nan=False)
    log(json.dumps(S["verdicts"], indent=1, ensure_ascii=False)[:4000])
    log(f"[exp115] done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
