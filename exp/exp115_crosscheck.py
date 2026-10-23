#!/usr/bin/env python3
"""exp115, the independent second route: exact per-class share and producer's-accuracy intervals.

Written from docs/plan/per_class_exact.md alone, without the run's script or outputs, and without the package.
numpy, scipy and the standard library only. Every probability is a sum over the exact sampling distribution of a
simple random sample of n of the N units (hypergeometric, or the three-category multivariate hypergeometric, or the
multivariate hypergeometric over the reference classes); there is no Monte Carlo.

    python -I exp/exp115_crosscheck.py --e113-private DIR --repo DIR --out FILE [--smoke] [--expect-count 40]
    python -I exp/exp115_crosscheck.py --self-test DIR      # synthetic data only, written under DIR

The output JSON holds counts and probabilities per map and class; no unit-level list and no coordinate.
"""

import argparse
import csv
import glob
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
import unicodedata
import warnings
from fractions import Fraction

import numpy as np
import scipy
from scipy.stats import hypergeom

ALPHA = Fraction(1, 20)  # family-wise error per map, exact (shares; and separately the PA intervals)
BUDGETS = (100, 400, 1000)
FLAG_BUDGET = 400
P3_BAR = 0.8
P4_BAR = 0.10
P5_BAR = 0.8
P5_UPPER = Fraction(1, 4)
P2_BAR = 0.95
NUM_TOL = 1e-9  # P1/P2 only: a float shortfall below this is noise in a sum of exact probabilities
PLANTEUR_CONFIGS = ("default", "loio_Guadeloupe", "loio_Martinique", "loio_Guyane", "loio_Réunion", "loio_Mayotte")
PLANTEUR_EMBS = ("oe13", "tessera2", "aef")
PLANTEUR_N_CLASSES = 8
LCMAP_LEGEND = tuple(range(1, 9))
SMOKE_NAMES = ("planteur/default/oe13", "lcmap/exp92/lcmap", "suite/sen1floods11")

READINGS = [
    "Populations follow the page's description. PLANTEUR: 6 configurations x 3 embeddings of exp113's window arrays "
    "(pred = map class, true = reference). LCMAP: exp92's lcmap and esri columns at all its plots (reference ref), and "
    "exp112's pred at rows with in_fit_1000 == '0' and has_embedding == '1'. Suite: every exp78 unit file whose family "
    "is 'classification' (reference y; err is checked to equal dec != y), plus sen1floods11 (truth = dec where err == "
    "0, the other of its two classes where err == 1); the other segmentation files are left out. The script counts "
    "what this yields and refuses unless the count equals --expect-count (default 40, the page's number).",
    "C is the legend's class count (8 for PLANTEUR and LCMAP; n_classes of the unit file for the suite), not the number "
    "of classes present. Share intervals at level 1 - 0.05/C; the PA's two count intervals at 1 - 0.05/(2C); the P1 "
    "bar is 1 - 0.05/C for both.",
    "Interval for a population count from k of n: every K in 0..N with P_K(X >= k) > alpha/2 and P_K(X <= k) > "
    "alpha/2 (strict), alpha = 1 - level, reported as K/N; found by bisection on each tail (each monotone in K).",
    "Map codes outside the legend (LCMAP/Esri code 0; an empty exp112 pred, read as -1) stay in the population and "
    "count toward no class's M_c. A reference outside the legend refuses the run.",
    "PLANTEUR legend: the observed codes of pred and true must fit 0..7 or 1..8 (8 classes); otherwise refuse.",
    "Graded classes are those with K_c > 0 for every arm and prediction (P4 included); classes with K_c = 0 are "
    "neither reported nor graded.",
    "Budgets b in {100, 400, 1000}: n = min(b, N); graded when b < N; a census (b >= N) is computed at n = N and "
    "reported with graded = false. P3-P5 use b = 400 on maps with N > 400.",
    "Flag: too little when M_c < K_lo (m_c below the share interval), too much when M_c > K_hi, with the share "
    "interval at level 1 - 0.05/C. P4 adds the two disjoint probabilities.",
    "PA interval: lower A_lo/K_hi (0 when K_hi = 0); upper 1 when K_lo = 0, else min(1, A_hi/K_lo, M_c/K_lo). Its "
    "coverage of A_c/K_c and 'upper <= 0.25' are decided in integer arithmetic.",
    "Median width: the lower median (smallest w with P(W <= w) >= 1/2) under the exact sampling distribution; share "
    "width (K_hi - K_lo)/N; PA width max(0, upper - lower).",
    "Thresholds p_c >= 0.05, m_c <= p_c/2, |m_c - p_c| <= 0.02 and PA_c <= 0.10 are decided in integers "
    "(20 K >= N, 2 M <= K, 50 |M - K| <= N, 10 A <= K).",
    "P1 and P2 compare floating-point sums of exact probabilities with their bars: a shortfall below 1e-9 is counted "
    "as numerical noise (reported as strict_shortfalls), a larger one as a failure. P3-P5 compare without tolerance.",
    "Joint share coverage: a sequential convolution over the reference classes present (k_1 ~ Hyp(N, K_1, n); "
    "k_2 | k_1 ~ Hyp(N - K_1, K_2, n - k_1); ...), keeping only each k_j whose interval covers K_j.",
    "PA sums: P(a, k) = Hyp(k; N, K_c, n) x Hyp(a; K_c, A_c, k), which is the three-category multivariate "
    "hypergeometric of (a, k - a, n - k), summed over every (a, k) of its support.",
]

NOT_COMPUTED = [
    "No comparison with the run's script or outputs (by design of the second route).",
    "No designs other than simple random sampling (the confidence-stratified design is out of the page's scope).",
    "Census budgets (b >= N) are reported, not graded.",
    "Classes with K_c = 0 (reference absent) get no share or PA arm, even when the map draws them.",
    "The segmentation unit files without a reference class (all but sen1floods11) are not used.",
    "User's accuracy, overall accuracy and area-adjusted estimates are not computed.",
    "Flag and PA-bound probabilities at budgets other than 400 are reported but not graded.",
    "Interval widths are summarised by the median only (no other quantiles).",
]


# ----------------------------------------------------------------------------------------------------------------
# Hypergeometric primitives (scipy's parametrisation: hypergeom(M = population, n = successes, N = draws))
# ----------------------------------------------------------------------------------------------------------------

NEAR = 1e-7  # a float tail this close (relatively) to its threshold is decided again in exact integers


def _ints(*vals):
    return np.broadcast_arrays(*(np.asarray(v, dtype=np.int64) for v in vals))


def hpmf(x, npop, ksucc, ndraw):
    """P(X = x) for X ~ Hyp(npop, ksucc, ndraw); exact 0 outside the support, exact 1 on a one-point support."""
    x, npop, ksucc, ndraw = _ints(x, npop, ksucc, ndraw)
    lo = np.maximum(0, ndraw - (npop - ksucc))
    hi = np.minimum(ndraw, ksucc)
    inside = (x >= lo) & (x <= hi)
    point = lo == hi
    out = np.zeros(x.shape, dtype=float)
    gen = inside & ~point
    if gen.any():
        out[gen] = hypergeom.pmf(x[gen], npop[gen], ksucc[gen], ndraw[gen])
    out[inside & point] = 1.0
    return out


def hpmf_rows(npop, ksucc, draws, xmax):
    """Matrix P[i, x] = P(X_i = x) for X_i ~ Hyp(npop, ksucc, draws[i]) and x = 0..xmax (xmax >= every row's top).

    Each row is anchored at its mode by scipy's pmf and extended by the exact ratio
    P(x + 1) / P(x) = (ksucc - x)(d - x) / ((x + 1)(npop - ksucc - d + x + 1)), accumulated as a sum of logs
    (relative error about 1e-12 over 1000 steps). This replaces one slow pmf call per cell (scipy's pmf is slow for
    populations below about 10^5).
    """
    npop, ksucc = int(npop), int(ksucc)
    d = np.asarray(draws, dtype=np.int64).reshape(-1, 1)
    lo = np.maximum(0, d - (npop - ksucc))
    hi = np.minimum(d, ksucc)
    if d.size and int(hi.max()) > xmax:
        raise ValueError("xmax below the support")
    mode = np.clip(((d + 1) * (ksucc + 1)) // (npop + 2), lo, hi)
    anchor = hpmf(mode, npop, ksucc, d)
    y = np.arange(xmax, dtype=np.int64)[None, :]
    valid = (y >= lo) & (y < hi)
    with np.errstate(all="ignore"):
        num = (ksucc - y).astype(float) * (d - y).astype(float)
        den = (y + 1).astype(float) * (npop - ksucc - d + y + 1).astype(float)
        L = np.where(valid, np.log(num / den), 0.0)
    C = np.zeros((d.shape[0], xmax + 1))
    if xmax > 0:
        C[:, 1:] = np.cumsum(L, axis=1)
    Cm = np.take_along_axis(C, np.clip(mode, 0, xmax), axis=1)
    x = np.arange(xmax + 1, dtype=np.int64)[None, :]
    inside = (x >= lo) & (x <= hi)
    with np.errstate(all="ignore"):
        return np.where(inside, anchor * np.exp(C - Cm), 0.0)


def upper_tail(k, npop, ksucc, ndraw):
    """P(X >= k), floating point."""
    k, npop, ksucc, ndraw = _ints(k, npop, ksucc, ndraw)
    lo = np.maximum(0, ndraw - (npop - ksucc))
    hi = np.minimum(ndraw, ksucc)
    out = np.where(k <= lo, 1.0, 0.0)
    gen = (k > lo) & (k <= hi)
    if gen.any():
        out[gen] = hypergeom.sf(k[gen] - 1, npop[gen], ksucc[gen], ndraw[gen])
    return out


def lower_tail(k, npop, ksucc, ndraw):
    """P(X <= k), floating point."""
    k, npop, ksucc, ndraw = _ints(k, npop, ksucc, ndraw)
    lo = np.maximum(0, ndraw - (npop - ksucc))
    hi = np.minimum(ndraw, ksucc)
    out = np.where(k >= hi, 1.0, 0.0)
    gen = (k >= lo) & (k < hi)
    if gen.any():
        out[gen] = hypergeom.cdf(k[gen], npop[gen], ksucc[gen], ndraw[gen])
    return out


def exact_terms(N, K, n):
    """Integers T[x] = C(K, x) C(N - K, n - x) for x = 0..n (0 outside the support); they add to C(N, n)."""
    T = [0] * (n + 1)
    lo, hi = max(0, n - (N - K)), min(n, K)
    if lo > hi:
        return T
    cK, cR = math.comb(K, lo), math.comb(N - K, n - lo)
    T[lo] = cK * cR
    for x in range(lo, hi):
        cK = cK * (K - x) // (x + 1)
        cR = cR * (n - x) // (N - K - n + x + 1)
        T[x + 1] = cK * cR
    return T


_COMB = {}
EXACT_CALLS = [0]


def _exact_tail_gt(kind, k, N, K, n, t):
    """Exact integer decision of P_K(X >= k) > t (kind 'up') or P_K(X <= k) > t (kind 'low'); t a Fraction."""
    EXACT_CALLS[0] += 1
    if (N, n) not in _COMB:
        _COMB[(N, n)] = math.comb(N, n)
    T = exact_terms(N, K, n)
    s = sum(T[k:]) if kind == "up" else sum(T[: k + 1])
    return s * t.denominator > t.numerator * _COMB[(N, n)]


def tail_gt(kind, k, N, K, n, t):
    """Vectorised P_K(X >= k) > t or P_K(X <= k) > t: floats, with near-threshold cases decided in integers."""
    k, N_, K_, n_ = _ints(k, N, K, n)
    f = upper_tail(k, N_, K_, n_) if kind == "up" else lower_tail(k, N_, K_, n_)
    tf = float(t)
    out = f > tf
    near = np.abs(f - tf) <= NEAR * tf
    if near.any():
        out = out.copy()
        for i in zip(*np.nonzero(near)):
            out[i] = _exact_tail_gt(kind, int(k[i]), int(N_[i]), int(K_[i]), int(n_[i]), t)
    return out


_TABLES = {}


def interval_table(n, N, alpha):
    """(K_lo[k], K_hi[k]) for k = 0..n: the exact equal-tailed interval for a population count at level 1 - alpha.

    The interval is every K in 0..N with P_K(X >= k) > alpha/2 and P_K(X <= k) > alpha/2, alpha taken as an exact
    rational. P_K(X >= k) is nondecreasing in K and P_K(X <= k) nonincreasing, so each end is found by bisection,
    vectorised over k; each comparison is decided in floating point unless the tail is within NEAR of alpha/2, in
    which case it is decided in exact integers (ties at a rational threshold are decided exactly).
    """
    alpha = Fraction(alpha)
    key = (int(n), int(N), alpha)
    if key in _TABLES:
        return _TABLES[key]
    n, N = int(n), int(N)
    if not 0 <= n <= N:
        raise ValueError(f"need 0 <= n <= N, got n={n}, N={N}")
    t = alpha / 2
    ks = np.arange(n + 1, dtype=np.int64)
    # lower end: smallest K with P_K(X >= k) > t (true at K = N, since then X = n >= k)
    bad = np.full(n + 1, -1, dtype=np.int64)
    good = np.full(n + 1, N, dtype=np.int64)
    while True:
        idx = np.nonzero(good - bad > 1)[0]
        if idx.size == 0:
            break
        mid = (bad[idx] + good[idx]) // 2
        ok = tail_gt("up", ks[idx], N, mid, n, t)
        good[idx[ok]] = mid[ok]
        bad[idx[~ok]] = mid[~ok]
    k_lo = good.copy()
    # upper end: largest K with P_K(X <= k) > t (true at K = 0, since then X = 0 <= k)
    good = np.zeros(n + 1, dtype=np.int64)
    bad = np.full(n + 1, N + 1, dtype=np.int64)
    while True:
        idx = np.nonzero(bad - good > 1)[0]
        if idx.size == 0:
            break
        mid = (bad[idx] + good[idx]) // 2
        ok = tail_gt("low", ks[idx], N, mid, n, t)
        good[idx[ok]] = mid[ok]
        bad[idx[~ok]] = mid[~ok]
    k_hi = good.copy()
    k_lo.setflags(write=False)
    k_hi.setflags(write=False)
    _TABLES[key] = (k_lo, k_hi)
    return _TABLES[key]


def weighted_median(values, weights):
    """Lower median of a discrete distribution: smallest v with P(V <= v) >= 1/2."""
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    keep = weights > 0
    v, w = values[keep], weights[keep]
    if v.size == 0:
        return None
    order = np.argsort(v, kind="stable")
    v = v[order]
    c = np.cumsum(w[order])
    i = int(np.searchsorted(c, (0.5 - 1e-12) * c[-1], side="left"))
    return float(v[min(i, v.size - 1)])


# ----------------------------------------------------------------------------------------------------------------
# Exact probabilities for one class / one map
# ----------------------------------------------------------------------------------------------------------------

def share_stats(N, n, K, M, tab, alpha):
    k_lo, k_hi = tab
    kmin, kmax = max(0, n - (N - K)), min(n, K)
    ks = np.arange(kmin, kmax + 1, dtype=np.int64)
    pk = hpmf(ks, N, K, n)
    lo, hi = k_lo[ks], k_hi[ks]
    cover = (lo <= K) & (K <= hi)
    t = Fraction(alpha) / 2
    direct = tail_gt("up", ks, N, K, n, t) & tail_gt("low", ks, N, K, n, t)
    return {
        "share_coverage": float(pk[cover].sum()),
        "flag_too_little": float(pk[M < lo].sum()),
        "flag_too_much": float(pk[M > hi].sum()),
        "median_share_width": weighted_median((hi - lo) / N, pk),
        "share_mass": float(pk.sum()),
        "share_table_vs_direct_mismatches": int((cover != direct).sum()),
        "share_empty_intervals": int((lo > hi).sum()),
    }


def pa_stats(N, n, K, A, M, tab, rows=hpmf_rows):
    p_lo, p_hi = tab
    kmin, kmax = max(0, n - (N - K)), min(n, K)
    ks = np.arange(kmin, kmax + 1, dtype=np.int64)
    as_ = np.arange(0, min(n, A) + 1, dtype=np.int64)
    pk = hpmf(ks, N, K, n)
    pak = rows(K, A, ks, int(as_[-1]))  # P(a | k): a ~ Hyp(K, A, k)
    joint = pk[:, None] * pak
    lo_a, hi_a = p_lo[as_][None, :], p_hi[as_][None, :]
    lo_k, hi_k = p_lo[ks][:, None], p_hi[ks][:, None]
    cov_lower = lo_a * K <= A * hi_k  # A_lo/K_hi <= A/K (when K_hi = 0 the lower end is 0 and lo_a is 0)
    cov_upper = (lo_k == 0) | ((A * lo_k <= hi_a * K) & (A * lo_k <= M * K))
    cover = np.broadcast_to(cov_lower & cov_upper, joint.shape)
    up_le = np.broadcast_to((lo_k > 0) & ((4 * hi_a <= lo_k) | (4 * M <= lo_k)), joint.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        lower = np.where(hi_k > 0, lo_a / np.maximum(hi_k, 1), 0.0)
        upper = np.where(lo_k > 0, np.minimum(1.0, np.minimum(hi_a, M) / np.maximum(lo_k, 1)), 1.0)
    width = np.maximum(np.broadcast_to(upper - lower, joint.shape), 0.0)
    return {
        "pa_coverage": float(joint[cover].sum()),
        "pa_upper_le_0.25": float(joint[up_le].sum()),
        "median_pa_width": weighted_median(width, joint),
        "pa_mass": float(joint.sum()),
    }


def _grid_rows(npop, ksucc, draws, xmax):
    """Slow reference for hpmf_rows: scipy's pmf in every cell (used by the self-test)."""
    d = np.asarray(draws, dtype=np.int64)
    return hpmf(np.arange(xmax + 1)[None, :], npop, ksucc, d[:, None])


def joint_cover(N, n, Ks, covering_sets, rows=hpmf_rows):
    """P(every class's share interval covers) under the multivariate hypergeometric of the reference counts.

    Sequential convolution: after class j the state is the number of draws left for the classes after j.
    """
    p = np.zeros(n + 1)
    p[n] = 1.0
    remaining = int(N)
    for K, S in zip(Ks, covering_sets):
        rs = np.nonzero(p > 0)[0]
        ks = np.nonzero(S)[0]
        if rs.size == 0 or ks.size == 0:
            return 0.0
        mat = rows(remaining, K, rs, n)[:, ks]
        contrib = p[rs][:, None] * mat
        dest = rs[:, None] - ks[None, :]
        ok = (dest >= 0) & (contrib > 0)
        p = np.bincount(dest[ok], weights=contrib[ok], minlength=n + 1)[: n + 1]
        remaining -= int(K)
    if remaining != 0:
        raise RuntimeError("reference counts do not add to N")
    return float(p[0])


def analyse_population(pop, budgets, alpha_total=ALPHA):
    N, C = pop["N"], pop["C"]
    a_share = Fraction(alpha_total) / C
    a_pa = Fraction(alpha_total) / (2 * C)
    bar = float(1 - a_share)
    present = [c for c in pop["legend"] if pop["K"][c] > 0]
    classes = {}
    for c in present:
        K, M, A = pop["K"][c], pop["M"][c], pop["A"][c]
        classes[str(c)] = {"K": K, "M": M, "A": A, "p": K / N, "m": M / N, "PA": A / K, "by_budget": {}}
    joint = {}
    cache = {}
    for b in budgets:
        n = min(int(b), N)
        if n not in cache:
            tab_s = interval_table(n, N, a_share)
            tab_p = interval_table(n, N, a_pa)
            per, Ks, sets = {}, [], []
            for c in present:
                K, M, A = pop["K"][c], pop["M"][c], pop["A"][c]
                per[c] = {**share_stats(N, n, K, M, tab_s, a_share), **pa_stats(N, n, K, A, M, tab_p)}
                Ks.append(K)
                sets.append((tab_s[0] <= K) & (K <= tab_s[1]))
            jp = joint_cover(N, n, Ks, sets)
            covs = [per[c]["share_coverage"] for c in present]
            bonf = 1.0 - sum(1.0 - v for v in covs)
            cache[n] = (per, {
                "all_share_intervals_cover": jp,
                "bonferroni_lower_bound": bonf,
                "within_bounds": bool(bonf - 1e-9 <= jp <= min(covs) + 1e-9),
            })
        per, jt = cache[n]
        graded = int(b) < N
        for c in present:
            classes[str(c)]["by_budget"][str(b)] = {"n": n, "graded": graded, "census": n == N, **per[c]}
        joint[str(b)] = {"n": n, "graded": graded, "census": n == N, **jt}
    return {
        "name": pop["name"], "source": pop["source"], "N": N, "C": C, "legend": list(pop["legend"]),
        "classes_present": len(present), "units_map_outside_legend": pop["outside"],
        "share_level": float(1 - a_share), "pa_count_level": float(1 - a_pa), "p1_bar": bar,
        "inputs": pop.get("inputs", []), "classes": classes, "joint": joint,
    }


# ----------------------------------------------------------------------------------------------------------------
# Populations
# ----------------------------------------------------------------------------------------------------------------

class Refuse(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pop_from_counts(name, source, N, legend, K, M, A, outside=0, inputs=()):
    legend = list(legend)
    K = {c: int(K.get(c, 0)) for c in legend}
    M = {c: int(M.get(c, 0)) for c in legend}
    A = {c: int(A.get(c, 0)) for c in legend}
    if sum(K.values()) != N:
        raise Refuse(f"{name}: reference counts add to {sum(K.values())}, not N = {N}")
    for c in legend:
        if not 0 <= A[c] <= min(K[c], M[c]):
            raise Refuse(f"{name}: inconsistent counts for class {c}")
    return {"name": name, "source": source, "N": int(N), "C": len(legend), "legend": legend,
            "K": K, "M": M, "A": A, "outside": int(outside), "inputs": list(inputs)}


def pop_from_arrays(name, source, ref, pred, legend, inputs=()):
    ref = np.asarray(ref).astype(np.int64).ravel()
    pred = np.asarray(pred).astype(np.int64).ravel()
    if ref.shape != pred.shape or ref.size == 0:
        raise Refuse(f"{name}: reference and map arrays differ in length or are empty")
    legend = [int(c) for c in legend]
    if not np.isin(ref, legend).all():
        bad = sorted(set(np.unique(ref).tolist()) - set(legend))
        raise Refuse(f"{name}: reference codes outside the legend: {bad[:10]}")
    K = {c: int((ref == c).sum()) for c in legend}
    M = {c: int((pred == c).sum()) for c in legend}
    A = {c: int(((ref == c) & (pred == c)).sum()) for c in legend}
    outside = int((~np.isin(pred, legend)).sum())
    return pop_from_counts(name, source, int(ref.size), legend, K, M, A, outside, inputs)


def _find_planteur(d, config, emb):
    want = f"exp113_windows_{config}_{emb}.npz"
    for form in ("NFC", "NFD"):
        path = os.path.join(d, unicodedata.normalize(form, want))
        if os.path.isfile(path):
            return path
    target = unicodedata.normalize("NFC", want)
    for name in os.listdir(d):
        if unicodedata.normalize("NFC", name) == target:
            return os.path.join(d, name)
    return None


def _int_array(z, key, name):
    if key not in z.files:
        raise Refuse(f"{name}: array '{key}' missing")
    a = np.asarray(z[key])
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
        raise Refuse(f"{name}: array '{key}' is not a 1-D integer array ({a.dtype}, ndim {a.ndim})")
    return a.astype(np.int64)


def _code(s):
    s = s.strip()
    return int(s) if s != "" else -1


def build_populations(e113_dir, repo):
    pops, breakdown = [], {}
    # 1. PLANTEUR
    if not os.path.isdir(e113_dir):
        raise Refuse(f"--e113-private is not a directory: {e113_dir}")
    missing = []
    for config in PLANTEUR_CONFIGS:
        for emb in PLANTEUR_EMBS:
            path = _find_planteur(e113_dir, config, emb)
            name = f"planteur/{config}/{emb}"
            if path is None:
                missing.append(f"exp113_windows_{config}_{emb}.npz")
                continue
            with np.load(path, allow_pickle=False) as z:
                pred = _int_array(z, "pred", name)
                true = _int_array(z, "true", name)
            codes = set(np.unique(pred).tolist()) | set(np.unique(true).tolist())
            if codes <= set(range(0, 8)):
                legend = list(range(0, 8))
            elif codes <= set(range(1, 9)):
                legend = list(range(1, 9))
            else:
                raise Refuse(f"{name}: codes {sorted(codes)[:12]} fit neither 0..7 nor 1..8 (8 classes)")
            pops.append(pop_from_arrays(name, "PLANTEUR, exp113 windows", true, pred, legend,
                                        [{"path": os.path.basename(path), "sha256": sha256_file(path)}]))
    if missing:
        raise Refuse(f"PLANTEUR window arrays missing in {e113_dir}: {missing}")
    breakdown["planteur"] = 18
    # 2. LCMAP's reference plots
    p92 = os.path.join(repo, "exp", "out", "exp92_plots.csv")
    with open(p92, newline="") as f:
        rows = list(csv.DictReader(f))
    ref = [_code(r["ref"]) for r in rows]
    inp92 = [{"path": "exp/out/exp92_plots.csv", "sha256": sha256_file(p92)}]
    for col in ("lcmap", "esri"):
        pops.append(pop_from_arrays(f"lcmap/exp92/{col}", f"exp92 plots, map column '{col}'", ref,
                                    [_code(r[col]) for r in rows], LCMAP_LEGEND, inp92))
    p112 = os.path.join(repo, "exp", "out", "exp112_plots.csv")
    with open(p112, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["in_fit_1000"].strip() == "0" and r["has_embedding"].strip() == "1"]
    pops.append(pop_from_arrays("lcmap/exp112/oe13", "exp112 plots outside the fit set, map column 'pred'",
                                [_code(r["ref"]) for r in rows], [_code(r["pred"]) for r in rows], LCMAP_LEGEND,
                                [{"path": "exp/out/exp112_plots.csv", "sha256": sha256_file(p112)}]))
    breakdown["lcmap"] = 3
    # 3. the embedding suite
    units = os.path.join(repo, "exp", "out", "exp78_units")
    files = sorted(glob.glob(os.path.join(units, "*.npz")))
    if not files:
        raise Refuse(f"no unit files in {units}")
    used_cls, used_seg, left_out = [], [], []
    for path in files:
        task = os.path.splitext(os.path.basename(path))[0]
        name = f"suite/{task}"
        with np.load(path, allow_pickle=False) as z:
            family = str(np.asarray(z["family"]).ravel()[0])
            C = int(np.asarray(z["n_classes"]).ravel()[0])
            dec = _int_array(z, "dec", name)
            err = _int_array(z, "err", name)
            y = _int_array(z, "y", name) if family == "classification" else None
        inp = [{"path": f"exp/out/exp78_units/{task}.npz", "sha256": sha256_file(path)}]
        if family == "classification":
            if not np.array_equal(err != 0, dec != y):
                raise Refuse(f"{name}: err is not dec != y on every unit")
            ref = y
            used_cls.append(task)
        elif task == "sen1floods11":
            if C != 2 or not np.isin(dec, [0, 1]).all() or not np.isin(err, [0, 1]).all():
                raise Refuse(f"{name}: expected two classes coded 0/1 and err in 0/1")
            ref = np.where(err == 0, dec, 1 - dec)
            used_seg.append(task)
        else:
            left_out.append(task)
            continue
        if not np.isin(dec, np.arange(C)).all():
            raise Refuse(f"{name}: map codes outside 0..{C - 1}")
        pops.append(pop_from_arrays(name, f"exp78 units ({family})", ref, dec, list(range(C)), inp))
    breakdown["suite_classification_files"] = len(used_cls)
    breakdown["suite_sen1floods11"] = len(used_seg)
    breakdown["suite_left_out"] = left_out
    breakdown["suite_classification_tasks"] = used_cls
    breakdown["total"] = len(pops)
    return pops, breakdown


# ----------------------------------------------------------------------------------------------------------------
# Grading
# ----------------------------------------------------------------------------------------------------------------

def grade(results, smoke):
    v = {}
    # P1
    cells, fails, strict, worst = 0, [], 0, None
    for r in results:
        for c, cr in r["classes"].items():
            for b, br in cr["by_budget"].items():
                if not br["graded"]:
                    continue
                for arm, key in (("share", "share_coverage"), ("pa", "pa_coverage")):
                    cells += 1
                    margin = br[key] - r["p1_bar"]
                    if worst is None or margin < worst["margin"]:
                        worst = {"map": r["name"], "class": c, "budget": int(b), "arm": arm, "coverage": br[key],
                                 "bar": r["p1_bar"], "margin": margin}
                    if margin < 0:
                        strict += 1
                    if margin < -NUM_TOL:
                        fails.append({"map": r["name"], "class": c, "budget": int(b), "arm": arm,
                                      "coverage": br[key], "bar": r["p1_bar"]})
    v["P1"] = {"verdict": "holds" if cells and not fails else ("fails" if fails else "not determined"),
               "cells": cells, "failures": fails, "strict_shortfalls": strict, "smallest_margin": worst}
    # P2
    cells, fails, strict, worst = 0, [], 0, None
    for r in results:
        for b, jr in r["joint"].items():
            if not jr["graded"]:
                continue
            cells += 1
            margin = jr["all_share_intervals_cover"] - P2_BAR
            if worst is None or margin < worst["margin"]:
                worst = {"map": r["name"], "budget": int(b), "probability": jr["all_share_intervals_cover"],
                         "margin": margin}
            if margin < 0:
                strict += 1
            if margin < -NUM_TOL:
                fails.append({"map": r["name"], "budget": int(b), "probability": jr["all_share_intervals_cover"]})
    v["P2"] = {"verdict": "holds" if cells and not fails else ("fails" if fails else "not determined"),
               "cells": cells, "failures": fails, "strict_shortfalls": strict, "smallest_margin": worst}
    if smoke:
        for p in ("P1", "P2"):
            v[p]["scope"] = "smoke subset only (three maps, n = 100)"
        for p in ("P3", "P4", "P5"):
            v[p] = {"verdict": "not graded in smoke (needs n = 400)"}
        return v
    b4 = str(FLAG_BUDGET)
    p3, p4, p5 = [], [], []
    for r in results:
        N = r["N"]
        if not N > FLAG_BUDGET:
            continue
        for c, cr in r["classes"].items():
            K, M, A = cr["K"], cr["M"], cr["A"]
            br = cr["by_budget"][b4]
            assert br["graded"] and br["n"] == FLAG_BUDGET
            cell = {"map": r["name"], "class": c, "N": N, "K": K, "M": M, "A": A,
                    "p": cr["p"], "m": cr["m"], "PA": cr["PA"]}
            if 20 * K >= N and 2 * M <= K:
                p3.append({**cell, "flag_too_little": br["flag_too_little"],
                           "ok": br["flag_too_little"] >= P3_BAR})
            if 50 * abs(M - K) <= N:
                both = br["flag_too_little"] + br["flag_too_much"]
                p4.append({**cell, "flag_either_way": both, "ok": both <= P4_BAR})
            if 20 * K >= N and 10 * A <= K:
                p5.append({**cell, "pa_upper_le_0.25": br["pa_upper_le_0.25"],
                           "ok": br["pa_upper_le_0.25"] >= P5_BAR})

    def verdict(cells):
        if not cells:
            return "not determined"
        return "holds" if all(c["ok"] for c in cells) else "fails"

    v["P3"] = {"verdict": verdict(p3), "bar": P3_BAR, "qualifying": len(p3), "cells": p3,
               "smallest": min((c["flag_too_little"] for c in p3), default=None)}
    v["P4"] = {"verdict": verdict(p4), "bar": P4_BAR, "qualifying": len(p4),
               "failures": [c for c in p4 if not c["ok"]],
               "largest": max((c["flag_either_way"] for c in p4), default=None),
               "largest_cell": max(p4, key=lambda c: c["flag_either_way"]) if p4 else None}
    v["P5"] = {"verdict": verdict(p5), "bar": P5_BAR, "qualifying": len(p5), "cells": p5,
               "smallest": min((c["pa_upper_le_0.25"] for c in p5), default=None)}
    return v


def numerics(results):
    masses, mism, empty, bounds = [], 0, 0, 0
    for r in results:
        for cr in r["classes"].values():
            for br in cr["by_budget"].values():
                masses += [br["share_mass"], br["pa_mass"]]
                mism += br["share_table_vs_direct_mismatches"]
                empty += br["share_empty_intervals"]
        bounds += sum(not j["within_bounds"] for j in r["joint"].values())
    return {
        "largest_abs_mass_minus_1": max(abs(m - 1.0) for m in masses) if masses else None,
        "share_table_vs_direct_mismatches": mism,
        "share_empty_intervals": empty,
        "joint_outside_bonferroni_bounds": bounds,
    }


# ----------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------

def environment():
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform()}


def run(args):
    t0 = time.time()
    script = os.path.abspath(__file__)
    try:
        pops, breakdown = build_populations(args.e113_private, args.repo)
    except Refuse as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    if len(pops) != args.expect_count:
        print(f"REFUSED: built {len(pops)} populations, expected {args.expect_count}. Breakdown: "
              f"{json.dumps(breakdown, ensure_ascii=False)}", file=sys.stderr)
        return 2
    budgets = BUDGETS
    chosen = pops
    if args.smoke:
        budgets = (100,)
        by_name = {p["name"]: p for p in pops}
        chosen = [by_name[n] for n in SMOKE_NAMES if n in by_name]
        if len(chosen) < 3:
            seen = {p["name"] for p in chosen}
            chosen += [p for p in pops if p["name"] not in seen][: 3 - len(chosen)]
    results = []
    for p in chosen:
        t = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r = analyse_population(p, budgets)
        r["seconds"] = round(time.time() - t, 2)
        results.append(r)
        print(f"{p['name']}: N={p['N']} C={p['C']} {r['seconds']} s", file=sys.stderr, flush=True)
    page = os.path.join(args.repo, "docs", "plan", "per_class_exact.md")
    out = {
        "experiment": "exp115",
        "route": "independent second route (crosscheck), written from the frozen page without the run's script or "
                 "outputs and without the package",
        "page": "docs/plan/per_class_exact.md",
        "page_sha256": sha256_file(page) if os.path.isfile(page) else None,
        "script": "exp/exp115_crosscheck.py",
        "script_sha256": sha256_file(script),
        "environment": environment(),
        "smoke": bool(args.smoke),
        "settings": {"alpha": float(ALPHA), "alpha_exact": str(ALPHA), "budgets": list(budgets), "flag_budget": FLAG_BUDGET, "p2_bar": P2_BAR,
                     "p3_bar": P3_BAR, "p4_bar": P4_BAR, "p5_bar": P5_BAR, "p5_upper": float(P5_UPPER),
                     "p1_p2_numerical_tolerance": NUM_TOL},
        "count_expected": args.expect_count,
        "count_built": len(pops),
        "breakdown": breakdown,
        "verdicts": grade(results, args.smoke),
        "numerics": {**numerics(results), "tail_comparisons_decided_in_integers": EXACT_CALLS[0],
                     "near_threshold_window": NEAR},
        "readings": READINGS,
        "not_computed": NOT_COMPUTED,
        "maps": results,
        "seconds": round(time.time() - t0, 1),
    }
    tmp = args.out + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, args.out)
    print(json.dumps({k: out["verdicts"][k].get("verdict") for k in out["verdicts"]}), file=sys.stderr)
    print(f"wrote {args.out} in {out['seconds']} s", file=sys.stderr)
    return 0


# ----------------------------------------------------------------------------------------------------------------
# Self-test (synthetic data only)
# ----------------------------------------------------------------------------------------------------------------

def exact_scan(n, N, alphas):
    """Brute force: for every k, scan every K in 0..N with exact integer tails. {alpha: (lo list, hi list)}."""
    total = math.comb(N, n)
    lows, ups = [], []
    for K in range(N + 1):
        T = exact_terms(N, K, n)
        pre = list(itertools.accumulate(T))
        lows.append(pre)
        ups.append([total] + [total - pre[k - 1] for k in range(1, n + 1)])
    out = {}
    for alpha in alphas:
        fr = Fraction(alpha) / 2
        p, q = fr.numerator, fr.denominator
        thr = p * total
        lo_l, hi_l = [], []
        for k in range(n + 1):
            S = [K for K in range(N + 1) if ups[K][k] * q > thr and lows[K][k] * q > thr]
            assert S and S == list(range(S[0], S[-1] + 1)), (n, N, alpha, k, "empty or not contiguous")
            lo_l.append(S[0])
            hi_l.append(S[-1])
        out[alpha] = (lo_l, hi_l)
    return out


def enumerate_exact(ref, pred, legend, n, alpha_total):
    """Every sample of n of the N units, with intervals from the exact scan: exact probabilities as Fractions."""
    N, C = len(ref), len(legend)
    a_s, a_p = alpha_total / C, alpha_total / (2 * C)
    tabs = exact_scan(n, N, [a_s, a_p])
    s_lo, s_hi = tabs[a_s]
    q_lo, q_hi = tabs[a_p]
    present = [c for c in legend if sum(r == c for r in ref) > 0]
    K = {c: sum(r == c for r in ref) for c in present}
    M = {c: sum(m == c for m in pred) for c in present}
    A = {c: sum(r == c and m == c for r, m in zip(ref, pred)) for c in present}
    acc = {c: {"cov": 0, "low": 0, "high": 0, "pacov": 0, "up": 0, "sw": [], "pw": []} for c in present}
    joint, total = 0, 0
    for sample in itertools.combinations(range(N), n):
        total += 1
        allc = True
        for c in present:
            k = sum(ref[i] == c for i in sample)
            a = sum(ref[i] == c and pred[i] == c for i in sample)
            lo, hi = s_lo[k], s_hi[k]
            cov = lo <= K[c] <= hi
            allc &= cov
            d = acc[c]
            d["cov"] += cov
            d["low"] += M[c] < lo
            d["high"] += M[c] > hi
            d["sw"].append(Fraction(hi - lo, N))
            alo, ahi, klo, khi = q_lo[a], q_hi[a], q_lo[k], q_hi[k]
            lower = Fraction(alo, khi) if khi > 0 else Fraction(0)
            upper = Fraction(1) if klo == 0 else min(Fraction(1), Fraction(ahi, klo), Fraction(M[c], klo))
            pa = Fraction(A[c], K[c])
            d["pacov"] += lower <= pa <= upper
            d["up"] += upper <= P5_UPPER
            d["pw"].append(max(upper - lower, Fraction(0)))
        joint += allc

    def med(ws):
        ws = sorted(ws)
        for i, w in enumerate(ws):
            if Fraction(i + 1, len(ws)) >= Fraction(1, 2):
                return w

    res = {}
    for c in present:
        d = acc[c]
        res[c] = {"share_coverage": Fraction(d["cov"], total), "flag_too_little": Fraction(d["low"], total),
                  "flag_too_much": Fraction(d["high"], total), "pa_coverage": Fraction(d["pacov"], total),
                  "pa_upper_le_0.25": Fraction(d["up"], total), "median_share_width": med(d["sw"]),
                  "median_pa_width": med(d["pw"])}
    return res, Fraction(joint, total)


def _check(ok, msg, failures):
    if not ok:
        failures.append(msg)
        print("  FAIL:", msg, flush=True)


def write_synthetic(base, extra_classification):
    """A synthetic record checkout and exp113 directory with the real file names and layout."""
    rng = np.random.default_rng(11500)
    repo = os.path.join(base, "repo")
    e113 = os.path.join(base, "e113")
    units = os.path.join(repo, "exp", "out", "exp78_units")
    os.makedirs(units, exist_ok=True)
    os.makedirs(e113, exist_ok=True)
    os.makedirs(os.path.join(repo, "docs", "plan"), exist_ok=True)
    with open(os.path.join(repo, "docs", "plan", "per_class_exact.md"), "w") as f:
        f.write("synthetic stand-in page\n")

    def confuse(true, C, keep, collapse=None):
        pred = true.copy()
        flip = rng.random(true.size) > keep
        pred[flip] = rng.integers(0, C, flip.sum())
        if collapse is not None:
            src, dst, frac = collapse
            sel = (true == src) & (rng.random(true.size) < frac)
            pred[sel] = dst
        return pred

    shares8 = np.array([0.55, 0.14, 0.09, 0.07, 0.06, 0.05, 0.03, 0.01])
    for config in PLANTEUR_CONFIGS:
        for i, emb in enumerate(PLANTEUR_EMBS):
            N = 6000 + 500 * i
            true = rng.choice(8, size=N, p=shares8)
            pred = confuse(true, 8, 0.9, collapse=(1, 0, 0.95 if i == 0 else 0.3))
            np.savez(os.path.join(e113, unicodedata.normalize("NFC", f"exp113_windows_{config}_{emb}.npz")),
                     pred=pred.astype(np.int64), true=true.astype(np.int64), extra=np.zeros(3))
    # exp92
    N = 25000
    ref = rng.choice(np.arange(1, 9), size=N, p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.03, 0.02])
    lc = confuse(ref - 1, 8, 0.85) + 1
    lc[rng.random(N) < 0.01] = 0
    es = confuse(ref - 1, 8, 0.7, collapse=(2, 0, 0.9)) + 1
    es[rng.random(N) < 0.02] = 0
    with open(os.path.join(repo, "exp", "out", "exp92_plots.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plotid", "ref", "lcmap", "esri_raw", "esri", "in_conf_subset", "lcpconf"])
        for i in range(N):
            w.writerow([i + 1, ref[i], lc[i], 7, es[i], 0, ""])
    # exp112
    N = 12000
    ref = rng.choice(np.arange(1, 9), size=N, p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.03, 0.02])
    pr = confuse(ref - 1, 8, 0.8) + 1
    infit = np.zeros(N, int)
    infit[rng.choice(N, 1000, replace=False)] = 1
    with open(os.path.join(repo, "exp", "out", "exp112_plots.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plotid", "in_fit_1000", "ref", "has_embedding", "pred", "top_probability"])
        for i in range(N):
            w.writerow([i + 1, infit[i], ref[i], 1, "" if infit[i] else pr[i], 0.5])
    # exp78 units: 17 classification + 7 segmentation, as in the record (plus one extra if asked)
    cls = {"awf_landsat": (9, 200), "awf_sentinel1": (9, 200), "awf_sentinel2": (9, 200),
           "breizhcrops": (9, 5000), "cropharvest_Peoples_Republic_of_China_6": (2, 4397),
           "cropharvest_Peoples_Republic_of_China_6_sentinel1": (2, 4397),
           "cropharvest_Peoples_Republic_of_China_6_sentinel1_sentinel2": (2, 4397),
           "cropharvest_Togo_12_sentinel1": (2, 306), "cropharvest_Togo_12_sentinel2": (2, 306),
           "cropharvest_Togo_12_sentinel2_sentinel1": (2, 306), "m_brick_kiln": (2, 999), "m_eurosat": (10, 1000),
           "m_forestnet": (12, 993), "m_so2sat": (17, 986), "nandi_landsat": (6, 825),
           "nandi_sentinel1": (6, 825), "nandi_sentinel2": (6, 825)}
    if extra_classification:
        cls["zz_synthetic_extra"] = (4, 2500)
    seg = {"m_cashew_plant": (7, 3000), "m_sa_crop_type": (10, 3000), "mados": (15, 3000),
           "pastis_sentinel1": (19, 3000), "pastis_sentinel1_sentinel2": (19, 3000), "pastis_sentinel2": (19, 3000),
           "sen1floods11": (2, 8000)}
    for task, (C, N) in cls.items():
        y = rng.integers(0, C, N)
        dec = confuse(y, C, 0.8)
        np.savez(os.path.join(units, task + ".npz"), dec=dec.astype(np.uint8), y=y.astype(np.uint8),
                 err=(dec != y).astype(np.uint8), family=np.array(["classification"]), n_classes=np.array([C]))
    for task, (C, N) in seg.items():
        dec = rng.choice(C, size=N, p=None if C != 2 else [0.8, 0.2])
        err = (rng.random(N) < 0.1).astype(np.uint8)
        np.savez(os.path.join(units, task + ".npz"), dec=dec.astype(np.uint8), y=np.zeros(N, np.uint8), err=err,
                 family=np.array(["segmentation"]), n_classes=np.array([C]))
    with open(os.path.join(units, "manifest.json"), "w") as f:
        json.dump({"model": "synthetic"}, f)
    return repo, e113


def self_test(base):
    os.makedirs(base, exist_ok=True)
    failures = []
    warnings.simplefilter("ignore", RuntimeWarning)
    t0 = time.time()

    print("[A] interval tables vs a brute-force exact scan of every K (small N, every n and k)", flush=True)
    stress = [ALPHA / 8, ALPHA / 16, ALPHA / 2, Fraction(1, 5), Fraction(1, 2), Fraction(9, 10)]
    count = 0
    for N in list(range(1, 26)) + [33, 40]:
        for n in range(0, N + 1):
            ex = exact_scan(n, N, stress)
            for alpha in stress:
                lo, hi = interval_table(n, N, alpha)
                count += 1
                _check(list(lo) == ex[alpha][0] and list(hi) == ex[alpha][1],
                       f"A: table != exact scan at N={N} n={n} alpha={alpha}", failures)
    print(f"  {count} tables compared", flush=True)

    print("[B] interval tables vs exact scans at small-population sizes like the suite's (real levels)", flush=True)
    levels = [ALPHA / C for C in (2, 6, 9, 17)] + [ALPHA / (2 * C) for C in (2, 6, 9, 17)]
    for N, ns in ((200, (100,)), (306, (100,)), (825, (100, 400)), (999, (400,)), (1000, (400,))):
        for n in ns:
            ex = exact_scan(n, N, levels)
            for alpha in levels:
                lo, hi = interval_table(n, N, alpha)
                _check(list(lo) == ex[alpha][0] and list(hi) == ex[alpha][1],
                       f"B: table != exact scan at N={N} n={n} alpha={alpha}", failures)
    print("  done", flush=True)

    print("[C] exact boundary certificates at large N (each end checked on both sides in integers)", flush=True)
    for N in (4397, 122614, 820000):
        for n in (100, 400, 1000):
            for alpha in (ALPHA / 8, ALPHA / 16, ALPHA / 9):
                lo, hi = interval_table(n, N, alpha)
                _check(bool(np.all(np.diff(lo) >= 0) and np.all(np.diff(hi) >= 0) and np.all(lo <= hi)),
                       f"C: table not monotone or empty at N={N} n={n}", failures)
                fr = alpha / 2
                p, q = fr.numerator, fr.denominator
                total = math.comb(N, n)
                for k in sorted({0, 1, 2, n // 50, n // 7, n // 2, n - 3, n - 1, n}):
                    Klo, Khi = int(lo[k]), int(hi[k])
                    up = lambda K: sum(exact_terms(N, K, n)[k:])  # noqa: E731
                    low = lambda K: sum(exact_terms(N, K, n)[: k + 1])  # noqa: E731
                    _check(up(Klo) * q > p * total, f"C: up tail at K_lo not above t (N={N} n={n} k={k})", failures)
                    if Klo > 0:
                        _check(not up(Klo - 1) * q > p * total, f"C: K_lo - 1 also inside (N={N} n={n} k={k})",
                               failures)
                    _check(low(Khi) * q > p * total, f"C: low tail at K_hi not above t (N={N} n={n} k={k})", failures)
                    if Khi < N:
                        _check(not low(Khi + 1) * q > p * total, f"C: K_hi + 1 also inside (N={N} n={n} k={k})",
                               failures)
    print("  done", flush=True)

    print("[D] pmf accuracy against exact integers", flush=True)
    worst = 0.0
    for N, K, n in ((30, 7, 12), (1000, 400, 400), (4397, 1200, 1000), (122614, 3000, 1000), (820000, 410000, 1000),
                    (820000, 820, 1000), (592385, 50000, 400), (820000, 600000, 1000)):
        T = exact_terms(N, K, n)
        total = math.comb(N, n)
        xs = [x for x in range(n + 1) if T[x]]
        got = hpmf(np.array(xs), N, K, n)
        for x, g in zip(xs, got):
            e = T[x] / total  # correctly rounded big-integer division
            if e > 1e-250:
                worst = max(worst, abs(g - e) / e)
    _check(worst < 1e-9, f"D: pmf relative error {worst:.3g}", failures)
    print(f"  scipy pmf: largest relative error {worst:.3g}", flush=True)
    worst = 0.0
    for npop, ks_, draws in ((30, 7, (0, 1, 5, 12, 30)), (4397, 1200, (0, 3, 100, 400, 1000)),
                             (65600, 45920, (1, 100, 399, 400, 1000)), (65600, 0, (0, 400)), (65600, 65600, (0, 400)),
                             (820000, 1066, (100, 1000)), (820000, 410000, (1, 400, 1000)), (1000, 999, (400, 999))):
        xmax = min(max(draws), ks_)
        got = hpmf_rows(npop, ks_, list(draws), xmax)
        for i, d in enumerate(draws):
            T = exact_terms(npop, ks_, d)
            total = math.comb(npop, d)
            for x in range(xmax + 1):
                e = T[x] / total if x <= d else 0.0
                if e > 1e-250:
                    worst = max(worst, abs(got[i, x] - e) / e)
                elif e == 0.0:
                    _check(got[i, x] == 0.0, f"D: row pmf nonzero outside the support ({npop},{ks_},{d},{x})",
                           failures)
    _check(worst < 1e-9, f"D: row pmf relative error {worst:.3g}", failures)
    print(f"  row pmf (mode anchor + exact ratios): largest relative error {worst:.3g}", flush=True)
    worst = 0.0
    for N, n, K, A, M in ((820000, 400, 65600, 45920, 70000), (25000, 1000, 3750, 3000, 3500),
                          (122614, 1000, 9000, 900, 2000)):
        tp = interval_table(n, N, ALPHA / 16)
        f1 = pa_stats(N, n, K, A, M, tp)
        f2 = pa_stats(N, n, K, A, M, tp, rows=_grid_rows)
        for key in ("pa_coverage", "pa_upper_le_0.25", "pa_mass"):
            worst = max(worst, abs(f1[key] - f2[key]))
        _check(f1["median_pa_width"] == f2["median_pa_width"], f"D: median PA width differs ({N},{K})", failures)
        ts = interval_table(n, N, ALPHA / 8)
        Ks = [K, N - K - 5000, 5000]
        sets = [(ts[0] <= x) & (x <= ts[1]) for x in Ks]
        worst = max(worst, abs(joint_cover(N, n, Ks, sets) - joint_cover(N, n, Ks, sets, rows=_grid_rows)))
    _check(worst < 1e-10, f"D: fast rows vs scipy grid differ by {worst:.3g}", failures)
    print(f"  PA sums and joint convolution, fast rows vs scipy pmf in every cell: largest difference {worst:.3g}",
          flush=True)

    print("[E] every probability vs full enumeration of samples on tiny populations", flush=True)
    tiny = [
        ("T1", [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2], [0, 0, 0, 1, 0, 1, 1, 0, 3, 0, 0, 0], [0, 1, 2]),
        ("T2", [0] * 6 + [1] * 4, [0] * 10, [0, 1]),
        ("T3", [0, 0, 0, 1, 1, 1, 1, 2, 2], [0, 1, 2, 1, 1, 1, 0, 2, 2], [0, 1, 2, 3]),
        ("T4", [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5], [1, 1, 2, 2, 2, 2, 1, 3, 4, 4, 1], [1, 2, 3, 4, 5]),
    ]
    worst = 0.0
    cases = 0
    for name, ref, pred, legend in tiny:
        N = len(ref)
        for alpha_total in (ALPHA, Fraction(3, 5), Fraction(3, 2)):
            pop = pop_from_arrays(name, "synthetic", ref, pred, legend)
            fast = analyse_population(pop, list(range(1, N + 1)), alpha_total=alpha_total)
            for n in range(1, N + 1):
                ex, exj = enumerate_exact(ref, pred, legend, n, alpha_total)
                cases += 1
                fj = fast["joint"][str(n)]["all_share_intervals_cover"]
                worst = max(worst, abs(fj - float(exj)))
                for c, e in ex.items():
                    fb = fast["classes"][str(c)]["by_budget"][str(n)]
                    for key in ("share_coverage", "flag_too_little", "flag_too_much", "pa_coverage",
                                "pa_upper_le_0.25", "median_share_width", "median_pa_width"):
                        d = abs(fb[key] - float(e[key]))
                        worst = max(worst, d)
                        _check(d < 1e-12, f"E: {name} alpha={alpha_total} n={n} class={c} {key}: "
                                          f"{fb[key]} vs exact {float(e[key])}", failures)
                _check(abs(fj - float(exj)) < 1e-12, f"E: {name} alpha={alpha_total} n={n} joint {fj} vs "
                                                     f"{float(exj)}", failures)
    print(f"  {cases} (population, level, n) cases, largest difference {worst:.3g}", flush=True)

    print("[F] PA sums and the joint convolution vs scipy's multivariate hypergeometric (N = 200)", flush=True)
    from scipy.stats import multivariate_hypergeom
    worst = 0.0
    Kv, N = [90, 70, 40], 200
    Av, Mv = [80, 10, 0], [150, 30, 15]
    for n in (20, 60):
        for alpha_total in (ALPHA, Fraction(3, 5)):
            C = 3
            tab_s = interval_table(n, N, alpha_total / C)
            tab_p = interval_table(n, N, alpha_total / (2 * C))
            for K, A, M in zip(Kv, Av, Mv):
                fast = pa_stats(N, n, K, A, M, tab_p)
                cov = up = 0.0
                for a in range(0, min(A, n) + 1):
                    for b in range(0, min(K - A, n - a) + 1):
                        if n - a - b > N - K:
                            continue
                        pr = multivariate_hypergeom.pmf([a, b, n - a - b], [A, K - A, N - K], n)
                        k = a + b
                        alo, ahi = tab_p[0][a], tab_p[1][a]
                        klo, khi = tab_p[0][k], tab_p[1][k]
                        lower = Fraction(int(alo), int(khi)) if khi > 0 else Fraction(0)
                        upper = Fraction(1) if klo == 0 else min(Fraction(1), Fraction(int(ahi), int(klo)),
                                                                 Fraction(M, int(klo)))
                        cov += pr * (lower <= Fraction(A, K) <= upper)
                        up += pr * (upper <= P5_UPPER)
                worst = max(worst, abs(cov - fast["pa_coverage"]), abs(up - fast["pa_upper_le_0.25"]))
            sets = [(tab_s[0] <= K) & (K <= tab_s[1]) for K in Kv]
            fj = joint_cover(N, n, Kv, sets)
            ej = 0.0
            for k1 in range(0, n + 1):
                for k2 in range(0, n - k1 + 1):
                    k3 = n - k1 - k2
                    if sets[0][k1] and sets[1][k2] and sets[2][k3]:
                        ej += multivariate_hypergeom.pmf([k1, k2, k3], Kv, n)
            worst = max(worst, abs(fj - ej))
            full = joint_cover(N, n, Kv, [np.ones(n + 1, bool)] * 3)
            _check(abs(full - 1.0) < 1e-12, f"F: unrestricted convolution mass {full}", failures)
    _check(worst < 1e-10, f"F: largest difference {worst:.3g}", failures)
    print(f"  largest difference {worst:.3g}", flush=True)

    print("[G] the CLI end to end on synthetic files with the record's names and layout", flush=True)
    script = os.path.abspath(__file__)
    base39 = os.path.join(base, "synthetic_like_record")
    base40 = os.path.join(base, "synthetic_plus_one")
    repo39, e39 = write_synthetic(base39, extra_classification=False)
    repo40, e40 = write_synthetic(base40, extra_classification=True)
    cmd = [sys.executable, "-I", script]
    r = subprocess.run(cmd + ["--e113-private", e39, "--repo", repo39, "--out", os.path.join(base39, "o.json")],
                       capture_output=True, text=True)
    _check(r.returncode == 2 and "built 39 populations, expected 40" in r.stderr,
           f"G: record-like layout should refuse at 39, got rc={r.returncode}: {r.stderr[-300:]}", failures)
    print("  record-like layout:", (r.stderr.strip().splitlines() or [""])[-1][:160], flush=True)
    out_s = os.path.join(base40, "smoke.json")
    r = subprocess.run(cmd + ["--e113-private", e40, "--repo", repo40, "--out", out_s, "--smoke"],
                       capture_output=True, text=True)
    _check(r.returncode == 0, f"G: smoke failed: {r.stderr[-500:]}", failures)
    if r.returncode == 0:
        js = json.load(open(out_s))
        _check([m["name"] for m in js["maps"]] == list(SMOKE_NAMES), "G: smoke maps", failures)
        _check(js["verdicts"]["P1"]["verdict"] == "holds" and js["verdicts"]["P2"]["verdict"] == "holds",
               "G: smoke P1/P2", failures)
    out_f = os.path.join(base40, "full.json")
    t = time.time()
    r = subprocess.run(cmd + ["--e113-private", e40, "--repo", repo40, "--out", out_f], capture_output=True, text=True)
    _check(r.returncode == 0, f"G: full run failed: {r.stderr[-500:]}", failures)
    if r.returncode == 0:
        js = json.load(open(out_f))
        v = js["verdicts"]
        print("  full synthetic run:", {k: v[k]["verdict"] for k in v}, f"{time.time() - t:.1f} s", flush=True)
        _check(len(js["maps"]) == 40, "G: 40 maps", failures)
        _check(v["P1"]["verdict"] == "holds" and v["P2"]["verdict"] == "holds", "G: P1/P2 on synthetic", failures)
        _check(v["P3"]["qualifying"] > 0 and v["P5"]["qualifying"] > 0 and v["P4"]["qualifying"] > 0,
               "G: synthetic collapse should qualify for P3, P4 and P5", failures)
        nm = js["numerics"]
        _check(nm["largest_abs_mass_minus_1"] < 1e-9 and nm["share_table_vs_direct_mismatches"] == 0
               and nm["share_empty_intervals"] == 0 and nm["joint_outside_bonferroni_bounds"] == 0,
               f"G: numerics {nm}", failures)
        census = [m for m in js["maps"] if m["N"] <= 400]
        _check(all(not m["joint"]["400"]["graded"] and m["joint"]["400"]["n"] == m["N"] for m in census),
               "G: census budgets must be ungraded", failures)
        mreun = [m for m in js["maps"] if m["name"].startswith("planteur/loio_R")]
        _check(len(mreun) == 3, "G: Réunion files found", failures)

    print("[H] timing on one synthetic map of the largest size (N = 820,000, 8 classes, n = 100/400/1000)",
          flush=True)
    N = 820000
    shares = [0.60, 0.13, 0.08, 0.07, 0.05, 0.04, 0.02, 0.01]
    K = [int(s * N) for s in shares]
    K[0] += N - sum(K)
    A = [int(0.95 * K[0]), int(0.01 * K[1]), int(0.7 * K[2]), int(0.8 * K[3]), int(0.6 * K[4]), int(0.5 * K[5]),
         int(0.3 * K[6]), 0]
    M = [0, int(0.02 * K[1]), K[2], K[3], K[4], K[5], K[6], 50]
    M[0] = N - sum(M[1:])
    pop = pop_from_counts("timing", "synthetic", N, range(8), dict(enumerate(K)), dict(enumerate(M)),
                          dict(enumerate(A)))
    _TABLES.clear()
    t = time.time()
    res = analyse_population(pop, BUDGETS)
    secs = time.time() - t
    print(f"  {secs:.1f} s for one 820,000-unit map at three budgets; joint at 1000: "
          f"{res['joint']['1000']['all_share_intervals_cover']:.6f}", flush=True)

    print(f"self-test {'PASSED' if not failures else 'FAILED'} in {time.time() - t0:.0f} s; "
          f"{len(failures)} failure(s)", flush=True)
    return 0 if not failures else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e113-private", help="directory of exp113_windows_<config>_<emb>.npz")
    ap.add_argument("--repo", help="the record checkout (exp/out/exp92_plots.csv, exp112_plots.csv, exp78_units)")
    ap.add_argument("--out", help="output JSON")
    ap.add_argument("--smoke", action="store_true", help="three maps, n = 100")
    ap.add_argument("--expect-count", type=int, default=40,
                    help="refuse unless this many populations are built (the page says 40)")
    ap.add_argument("--self-test", metavar="DIR", help="run the self-test on synthetic data written under DIR")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test(args.self_test)
    if not (args.e113_private and args.repo and args.out):
        ap.error("--e113-private, --repo and --out are required")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
