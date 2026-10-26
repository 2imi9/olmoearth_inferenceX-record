#!/usr/bin/env python
"""Tighter valid intervals for an agency's stratified reference sample (design measurements; run on aicr, not locally).

    python agency_bounds.py REC AREAS real [COMBINED]  > real.json   # every construction on NLCD 2016 (set B) and GFC2020
    python agency_bounds.py REC AREAS enum             > enum.json   # exact coverage by enumeration, small populations
    python agency_bounds.py REC AREAS mc [R]           > mc.json     # Monte Carlo coverage at the real designs' shapes

REC is the research record's root (exp/out/exp94_nlcd_points.csv, exp94_gfc_points.csv), AREAS the GFC2020
strata_size.csv (hectares), COMBINED (optional) the GFC2020 combined_scenario_all_GFCV2.csv, read only to bound the 24
units with no assignment (its coordinates are never written). Written without being run, since the laptop stays idle:
read the `checks` block of `real` first; it compares the data reading with sizing.json and the optimiser with closed
forms.

Constructions, each a two-sided 95% interval:
  normal            Stehman (2014) / Cochran: the stratified estimate +/- 1.96 SE (nominal)
  union             each stratum's exact interval at 1 - 0.05/H, weighted (the package's union bound, as sizing.py)
  chernoff_global   stratified Chernoff test inverted: sup over every stratum composition consistent with the
                    hypothesised value, each tail at 0.025
  chernoff_boxed    the same with the compositions restricted to per-stratum boxes holding jointly at 1 - GAMMA
                    (Berger and Boos 1994), each tail at (0.05 - GAMMA)/2: the construction proposed for the package
  bennett2_wr       an upper confidence bound on the variance at 0.005, then Bennett's inequality (with-replacement model)
  ebmix_wr          predictable empirical-Bernstein e-value mixed over lambda (with-replacement model)
"""
import csv
import json
import math
import os
import sys
import time
from itertools import product

import numpy as np

from oe_inferencex.estimate import hypergeom_interval

Z = 1.959963984540054
ALPHA = 0.05
GAMMA = 0.01                 # the share of alpha the per-stratum boxes take
TMAX = 30.0                  # the largest |tilt| on one unit, lam W_h |v| / n_h, so exp() never overflows
METHODS = ("normal", "union", "chernoff_global", "chernoff_boxed", "chernoff_boxed_gamma_0.002",
           "chernoff_boxed_gamma_0.005", "chernoff_boxed_gamma_0.02", "bennett2_wr", "ebmix_wr")

# sizing.json (scratchpad/agency, run on aicr before this design): the data reading must reproduce these
SIZING = {
    "gfc_oa": (0.9150627376612703, (0.9107113928132743, 0.9194140825092664), (0.8330264385251692, 0.9566745600412357)),
    "gfc_diff": (0.0009360773645878584, (-0.003415911021792898, 0.005288065750968615),
                 (-0.09355880719469932, 0.09530105931867783)),
    "nlcd16_l2_primary": (0.7211334217077565, (0.7033254495268878, 0.7389413938886252),
                          (0.6298309978813539, 0.8008466179636884)),
    "nlcd16_ua52_primary_union": (0.6764787534540311, (0.5285539240483915, 0.7805659620586974)),
}
# the record (exp/out/exp94_summary.json, found_after_result): NLCD's published accuracies reproduced from the files
RECORD_ALT = {2016: 0.8638724418296603, 2011: 0.8650733895162801}
PUBLISHED = {
    "nlcd2016_B_2016_oa_level2_primary_or_alternate": 0.864,
    "nlcd2016_B_2011_oa_level2_primary_or_alternate": 0.865,
    "gfc2020_v2_oa": 0.91,
    "source": ("the research record, docs/results/comparisons.md (exp94): NLCD's published level II accuracies, which "
               "count a match with the primary or the alternate label, reproduced from the files as 86.4% (2016) and "
               "86.5% (2011); the published GFC2020 V2 figure, 91%. Per-class published values are to be transcribed "
               "from Wickham et al. (2021) and the JRC report (JRC141231) before they are compared."),
}
NLCD_TO_LCMAP = {21: 1, 22: 1, 23: 1, 24: 1, 81: 2, 82: 2, 52: 3, 71: 3, 41: 4, 42: 4, 43: 4, 11: 5, 90: 6, 95: 6,
                 12: 7, 31: 8}


# ============================================================================ the engine
def _seg(n, A, B, cp, cm, a0, b0, da, db, lo, hi):
    """max over s in [lo, hi] of F(a0 + s da, b0 + s db), F(a, b) = n log(1 + a A + b B) - cp a - cm b. F is concave
    along the segment, so its maximum is an end or the root of F' between them; -inf where the segment is empty."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        K = da * A + db * B
        C = da * cp + db * cm
        D0 = 1.0 + a0 * A + b0 * B
        dlo = n * K / (D0 + lo * K) - C
        dhi = n * K / (D0 + hi * K) - C
        root = (n * K / np.where(C == 0, 1.0, C) - D0) / np.where(K == 0, 1.0, K)
        s = np.where(dlo <= 0, lo, np.where(dhi >= 0, hi, np.minimum(np.maximum(root, lo), hi)))
        a = a0 + s * da
        b = b0 + s * db
        val = n * np.log1p(a * A + b * B) - cp * a - cm * b
    return np.where(lo <= hi + 1e-15, val, -np.inf)


def polymax(n, A, B, cp, cm, la, ua, lb, ub):
    """max over the polygon {la <= a <= ua, lb <= b <= ub, a + b <= 1} of n log(1 + a A + b B) - cp a - cm b. The
    function depends on (a, b) through a A + b B and a linear term, so along each line of constant a A + b B it is
    linear and its maximum over the polygon is reached on the boundary: the best of five edges, each in closed form."""
    e = (_seg(n, A, B, cp, cm, la, 0.0, 0.0, 1.0, lb, np.minimum(ub, 1 - la)),
         _seg(n, A, B, cp, cm, ua, 0.0, 0.0, 1.0, lb, np.minimum(ub, 1 - ua)),
         _seg(n, A, B, cp, cm, 0.0, lb, 1.0, 0.0, la, np.minimum(ua, 1 - lb)),
         _seg(n, A, B, cp, cm, 0.0, ub, 1.0, 0.0, la, np.minimum(ua, 1 - ub)),
         _seg(n, A, B, cp, cm, 0.0, 1.0, 1.0, -1.0, np.maximum(la, 1 - ub), np.minimum(ua, 1 - lb)))
    out = e[0]
    for v in e[1:]:
        out = np.maximum(out, v)
    return out


def linear_lower(kp, km, n, W, vp, vm, a_tail, box, census=None, iters=12, g=13):
    """Lower confidence bound, at 1 - a_tail when the box holds, for mu = sum_h W_h (vp P_h(+) + vm P_h(-)), y in
    {vp, 0, vm} with vm <= 0 < vp, from kp_h and km_h of n_h units drawn at random in each stratum (with or without
    replacement; strata independent). box = (la, ua, lb, ub), the shares P_h(+) and P_h(-) the null may take.

    For lam >= 0 and nu > 0, Markov's inequality on exp(lam mu_hat), Hoeffding (1963, Theorem 4: the sum of a sample
    drawn without replacement has an MGF at most that of one drawn with replacement) and weak duality give, for every
    composition in the box with mu <= mu0,
        P(mu_hat >= x) <= exp(-lam x + nu mu0 + sum_h phi_h(lam, nu)),
    phi_h the polygon maximum of n_h log(1 + a A_h + b B_h) - nu W_h (vp a + vm b), A_h = e^{lam W_h vp/n_h} - 1,
    B_h = e^{lam W_h vm/n_h} - 1. So every mu0 below (log a_tail + lam x - sum_h phi_h) / nu is rejected. Every
    (lam, nu) gives a valid bound; the largest found on a zooming grid is returned. A census stratum enters exactly."""
    kp, km, n, W = (np.asarray(v, dtype=float).ravel() for v in (kp, km, n, W))
    H = n.size
    census = np.zeros(H, bool) if census is None else np.asarray(census, bool).ravel()
    la, ua, lb, ub = (np.asarray(v, dtype=float).ravel()[:, None, None] for v in box)
    ybar = (vp * kp + vm * km) / n
    x = float(np.sum(W * ybar))
    floor = min(0.0, vm)
    if census.all():
        return x
    s2 = np.where(n > 1, (vp * vp * kp + vm * vm * km - n * ybar ** 2) / np.maximum(n - 1, 1), 0.0)
    var = float(np.sum(np.where(census, 0.0, W ** 2 * np.maximum(s2, 0.0) / n)))
    lam_cap = TMAX / float(np.max((W * max(abs(vp), abs(vm)) / n)[~census]))
    lam0 = min(math.sqrt(2 * math.log(1 / a_tail) / max(var, 1e-12)), lam_cap)
    cu, cr, hu, hr = math.log(lam0), 0.0, 3.0, 2.0
    nn, WW = n[:, None, None], W[:, None, None]
    cen, yc = census[:, None, None], ybar[:, None, None]
    loga, best = math.log(a_tail), -math.inf
    for _ in range(iters):
        U = cu + np.linspace(-hu, hu, g)
        R = cr + np.linspace(-hr, hr, g)
        lam = np.minimum(np.exp(U), lam_cap)
        nu = lam[:, None] * np.exp(R)[None, :]
        t = lam[None, :, None] * WW / nn
        phi = polymax(nn, np.expm1(t * vp), np.expm1(t * vm), nu[None] * WW * vp, nu[None] * WW * vm, la, ua, lb, ub)
        phi = np.where(cen, (lam[None, :, None] - nu[None]) * WW * yc, phi)
        L = (loga + lam[:, None] * x - phi.sum(0)) / nu
        L = np.where(np.isfinite(L), L, -np.inf)
        i, j = np.unravel_index(int(np.argmax(L)), L.shape)
        best = max(best, float(L[i, j]))
        cu, cr = float(U[i]), float(R[j])
        if 0 < i < g - 1 and 0 < j < g - 1:
            hu, hr = hu / 2.5, hr / 2.5
    return float(min(max(best - 1e-12, floor), x))


def two_point(k_lo, n, W, a_tail, box, census=None, k_hi=None, **kw):
    """(lower, upper) for sum_h W_h p_h, y in {0, 1}; box = (lo, hi) per stratum. k_lo gives the lower end and k_hi
    (default k_lo) the upper: units whose value is unknown count 0 at the one end and 1 at the other."""
    n = np.asarray(n, dtype=float).ravel()
    k_lo = np.asarray(k_lo, dtype=float).ravel()
    k_hi = k_lo if k_hi is None else np.asarray(k_hi, dtype=float).ravel()
    lo_b, hi_b = (np.asarray(v, dtype=float).ravel() for v in box)
    z = np.zeros(n.size)
    L = linear_lower(k_lo, z, n, W, 1.0, 0.0, a_tail, (lo_b, hi_b, z, z), census, **kw)
    Lc = linear_lower(n - k_hi, z, n, W, 1.0, 0.0, a_tail, (1 - hi_b, 1 - lo_b, z, z), census, **kw)
    return L, 1.0 - Lc


def three_point(kp_lo, km_lo, n, W, a_tail, box, census=None, kp_hi=None, km_hi=None, **kw):
    """(lower, upper) for sum_h W_h (P_h(+1) - P_h(-1)), y in {+1, 0, -1}; box = (la, ua, lb, ub)."""
    kp_hi = kp_lo if kp_hi is None else kp_hi
    km_hi = km_lo if km_hi is None else km_hi
    la, ua, lb, ub = box
    L = linear_lower(kp_lo, km_lo, n, W, 1.0, -1.0, a_tail, (la, ua, lb, ub), census, **kw)
    Ln = linear_lower(km_hi, kp_hi, n, W, 1.0, -1.0, a_tail, (lb, ub, la, ua), census, **kw)
    return L, -Ln


PIECES = 400                 # the ratio test's tilt direction is fixed on each of these equal pieces of [0, 1]


def ratio_lower(kt, kf, n, W, a_tail, box, census=None, K=PIECES, iters=8, g=9, bis=24):
    """Lower confidence bound for R = sum_h W_h P_h(t) / sum_h W_h (P_h(t) + P_h(f)) (a user's or producer's
    accuracy). R <= theta0 is (1 - theta0) tau - theta0 phi <= 0, tau and phi the two weighted shares.

    The test of R <= theta0 uses the one-dimensional statistic S = (1 - d) T_hat - d F_hat with a direction d FIXED IN
    ADVANCE as a function of theta0: the midpoint of the piece of [0, 1] (K equal pieces) that holds theta0. With
    tilts lt = lam (1 - d) on t and lf = -lam d on f and a dual nu = rho lam, it rejects when
        G(theta0) = -(lt T_hat + lf F_hat) + sum_h max_box [n_h log(1 + t A_h + f B_h) - nu W_h ((1 - theta0) t - theta0 f)]
    is at most log a_tail for some (lam, rho). At the true R the constraint term vanishes at the true composition, so a
    rejection there means the Chernoff p-value of S is at most a_tail: valid, since S's direction does not depend on
    the data (letting the data choose the direction would make it a test over many directions, not level a_tail).
    For a fixed (lam, rho) and piece, G rises with theta0 (only the cost term moves), so a rejection at a piece's top
    covers the whole piece. The bound is the scan from 0 upward: every piece wholly rejected, then the bisection
    inside the first piece that is not; an imperfect optimiser only stops the scan early, which is conservative."""
    kt, kf, n, W = (np.asarray(v, dtype=float).ravel() for v in (kt, kf, n, W))
    H = n.size
    census = np.zeros(H, bool) if census is None else np.asarray(census, bool).ravel()
    la, ua, lb, ub = (np.asarray(v, dtype=float).ravel()[:, None] for v in box)
    th_hat, fh_hat = kt / n, kf / n
    T, F = float(np.sum(W * th_hat)), float(np.sum(W * fh_hat))
    if T <= 0:
        return 0.0
    Rhat = T / (T + F)
    if census.all():
        return Rhat
    lam_cap = TMAX / float(np.max((W / n)[~census]))
    nn, WW, cen = n[:, None], W[:, None], census[:, None]
    tc, fc = th_hat[:, None], fh_hat[:, None]
    loga = math.log(a_tail)
    edges = np.linspace(0.0, 1.0, K + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])

    def scale(d):
        """the normal-theory tilt for direction d: sqrt(2 log(1/a) / Var(S_d)), capped"""
        d = np.atleast_1d(np.asarray(d, dtype=float))
        y1, y2 = (1 - d)[None, :], -d[None, :]
        ybar = (y1 * kt[:, None] + y2 * kf[:, None]) / nn
        s2 = np.where(nn > 1, (y1 ** 2 * kt[:, None] + y2 ** 2 * kf[:, None] - nn * ybar ** 2) / np.maximum(nn - 1, 1), 0.0)
        v = np.sum(np.where(cen, 0.0, WW ** 2 * np.maximum(s2, 0.0) / nn), 0)
        return np.minimum(np.sqrt(2 * math.log(1 / a_tail) / np.maximum(v, 1e-12)), lam_cap)

    def Gfun(d, th0, lam, rho):
        """G for arrays (broadcast to one length P) of direction, tested value, tilt and dual ratio"""
        d, th0, lam, rho = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, dtype=float)) for v in (d, th0, lam, rho)))
        lt, lf, nu = lam * (1 - d), -lam * d, lam * rho
        At = np.expm1(lt[None, :] * WW / nn)
        Bf = np.expm1(lf[None, :] * WW / nn)
        cp = nu[None, :] * WW * (1 - th0[None, :])
        cm = -nu[None, :] * WW * th0[None, :]
        phi = polymax(nn, At, Bf, cp, cm, la, ua, lb, ub)
        ce = (lt[None, :] * tc + lf[None, :] * fc) * WW - nu[None, :] * WW * ((1 - th0[None, :]) * tc - th0[None, :] * fc)
        return -(lt * T + lf * F) + np.where(cen, ce, phi).sum(0)

    def min_G(d, th0, lam0):
        """min over (lam, rho) on a zooming grid; any point is a valid test"""
        cu, cr, hu, hr = math.log(lam0), 0.0, 3.0, 2.0
        best = math.inf
        for _ in range(iters):
            U = cu + np.linspace(-hu, hu, g)
            R = cr + np.linspace(-hr, hr, g)
            Lm, Rm = np.meshgrid(np.minimum(np.exp(U), lam_cap), np.exp(R), indexing="ij")
            v = Gfun(d, th0, Lm.ravel(), Rm.ravel())
            k = int(np.argmin(np.where(np.isfinite(v), v, np.inf)))
            best = min(best, float(v[k]))
            if best <= loga:
                return best
            i, j = np.unravel_index(k, (g, g))
            cu, cr = float(U[i]), float(R[j])
            if 0 < i < g - 1 and 0 < j < g - 1:
                hu, hr = hu / 2.5, hr / 2.5
        return best

    j_hat = min(int(Rhat * K), K - 1)                  # the piece holding the estimate; pieces above are not rejected
    certified = np.zeros(K, bool)
    if j_hat > 0:                                        # a cheap pass over every piece below the estimate at once
        js = np.arange(j_hat)
        lam0s = scale(mids[js])
        for c in (0.25, 0.5, 1.0, 2.0, 4.0):
            for rho in (0.5, 1.0, 2.0):
                certified[js] |= Gfun(mids[js], edges[js + 1], np.minimum(lam0s * c, lam_cap), rho) <= loga
    L = 0.0
    for j in range(K):
        if certified[j]:
            L = float(edges[j + 1])
            continue
        lam0 = float(scale(mids[j])[0])
        if min_G(mids[j], edges[j + 1], lam0) <= loga:
            L = float(edges[j + 1])
            continue
        if min_G(mids[j], edges[j], lam0) > loga:
            break                                        # the scan stops: edges[j] (= L) is not rejected
        lo, hi = float(edges[j]), float(edges[j + 1])
        for _ in range(bis):
            mid = 0.5 * (lo + hi)
            if min_G(mids[j], mid, lam0) <= loga:
                lo = mid
            else:
                hi = mid
        L = lo
        break
    return float(max(0.0, min(L - 1e-12, Rhat)))


def ratio(kt, kf, n, W, a_tail, box, census=None, **kw):
    """(lower, upper) for R = tau / (tau + phi); the upper end is one minus the lower end of phi / (tau + phi)."""
    la, ua, lb, ub = box
    return (ratio_lower(kt, kf, n, W, a_tail, (la, ua, lb, ub), census, **kw),
            1.0 - ratio_lower(kf, kt, n, W, a_tail, (lb, ub, la, ua), census, **kw))


# ============================================================================ boxes
def kl(q, p):
    q = np.asarray(q, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-300, 1 - 1e-16)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(q > 0, q * np.log(np.where(q > 0, q, 1.0) / p), 0.0)
        b = np.where(q < 1, (1 - q) * np.log(np.where(q < 1, 1 - q, 1.0) / (1 - p)), 0.0)
    return a + b


def kl_bounds(k, n, c):
    """{p : n kl(k/n || p) <= c} per stratum, each end rounded outward (the bisection keeps the outside end)."""
    k, n = (np.asarray(v, dtype=float).ravel() for v in (k, n))
    q = k / n
    lo, hi = np.zeros(q.size), q.copy()
    for _ in range(80):
        m = 0.5 * (lo + hi)
        inside = n * kl(q, m) <= c
        hi, lo = np.where(inside, m, hi), np.where(inside, lo, m)
    lower = np.where(k == 0, 0.0, lo)
    lo2, hi2 = q.copy(), np.ones(q.size)
    for _ in range(80):
        m = 0.5 * (lo2 + hi2)
        inside = n * kl(q, m) <= c
        lo2, hi2 = np.where(inside, m, lo2), np.where(inside, hi2, m)
    upper = np.where(k == n, 1.0, hi2)
    return lower, upper


def box_for(k, n, N, delta):
    """Each stratum's share in an interval that holds with probability at least 1 - delta: exact hypergeometric where
    N_h counts the stratum's units, the two-sided Chernoff (KL) interval where only its area is known (valid for any
    number of units by Hoeffding 1963, Theorem 4); a census is its own value. Returns (lo, hi, census)."""
    k, n = (np.asarray(v, dtype=float).ravel() for v in (k, n))
    if N is None:
        lo, hi = kl_bounds(k, n, math.log(2 / delta))
        return lo, hi, np.zeros(k.size, bool)
    N = np.asarray(N).ravel()
    lo, hi = np.empty(k.size), np.empty(k.size)
    cen = n >= N
    for h in range(k.size):
        if cen[h]:
            lo[h] = hi[h] = k[h] / n[h]
        else:
            lo[h], hi[h] = hypergeom_interval(int(round(k[h])), int(n[h]), int(N[h]), conf=1 - delta)
    return lo, hi, cen


# ============================================================================ the constructions beside
def normal_linear(kp, km, n, W, vp, vm, N=None):
    """Stehman (2014) / Cochran: (estimate, low, high) for sum W ybar with SE^2 = sum W^2 (1 - n/N) s^2 / n."""
    kp, km, n, W = (np.asarray(v, dtype=float).ravel() for v in (kp, km, n, W))
    ybar = (vp * kp + vm * km) / n
    s2 = np.where(n > 1, (vp * vp * kp + vm * vm * km - n * ybar ** 2) / np.maximum(n - 1, 1), 0.0)
    fpc = 1.0 if N is None else 1 - n / np.asarray(N, dtype=float).ravel()
    est = float(np.sum(W * ybar))
    se = math.sqrt(max(float(np.sum(W ** 2 * fpc * np.maximum(s2, 0.0) / n)), 0.0))
    return est, est - Z * se, est + Z * se


def normal_ratio(kt, kf, n, W, N=None):
    """Stehman (2014): R = T / D with the linearised variance of z = (t - R (t + f)) / D; None when D = 0."""
    kt, kf, n, W = (np.asarray(v, dtype=float).ravel() for v in (kt, kf, n, W))
    T, D = float(np.sum(W * kt / n)), float(np.sum(W * (kt + kf) / n))
    if D <= 0:
        return None
    R = T / D
    zt, zf = (1 - R) / D, -R / D
    zbar = (zt * kt + zf * kf) / n
    s2 = np.where(n > 1, (zt * zt * kt + zf * zf * kf - n * zbar ** 2) / np.maximum(n - 1, 1), 0.0)
    fpc = 1.0 if N is None else 1 - n / np.asarray(N, dtype=float).ravel()
    se = math.sqrt(max(float(np.sum(W ** 2 * fpc * np.maximum(s2, 0.0) / n)), 0.0))
    return R, R - Z * se, R + Z * se


def union_share(k, n, W, Nint, alpha=ALPHA):
    k, n, W = (np.asarray(v, dtype=float).ravel() for v in (k, n, W))
    H = n.size
    lo = hi = 0.0
    for h in range(H):
        a, b = hypergeom_interval(int(k[h]), int(n[h]), int(Nint[h]), conf=1 - alpha / H)
        lo += W[h] * a
        hi += W[h] * b
    return lo, hi


def union_diff(kp, km, n, W, Nint, alpha=ALPHA):
    kp, km, n, W = (np.asarray(v, dtype=float).ravel() for v in (kp, km, n, W))
    H = n.size
    c = 1 - alpha / (2 * H)
    lo = hi = 0.0
    for h in range(H):
        l10, u10 = hypergeom_interval(int(kp[h]), int(n[h]), int(Nint[h]), conf=c)
        l01, u01 = hypergeom_interval(int(km[h]), int(n[h]), int(Nint[h]), conf=c)
        lo += W[h] * (l10 - u01)
        hi += W[h] * (u10 - l01)
    return lo, hi


def union_ratio(kt, kf, n, W, Nint, alpha=ALPHA):
    kt, kf, n, W = (np.asarray(v, dtype=float).ravel() for v in (kt, kf, n, W))
    H = n.size
    c = 1 - alpha / (2 * H)
    A_lo = A_hi = B_lo = B_hi = 0.0
    for h in range(H):
        la, ua = hypergeom_interval(int(kt[h]), int(n[h]), int(Nint[h]), conf=c)
        lb, ub = hypergeom_interval(int(kf[h]), int(n[h]), int(Nint[h]), conf=c)
        A_lo += W[h] * la
        A_hi += W[h] * ua
        B_lo += W[h] * lb
        B_hi += W[h] * ub
    return (A_lo / (A_lo + B_hi) if A_lo + B_hi > 0 else 0.0, A_hi / (A_hi + B_lo) if A_hi + B_lo > 0 else 1.0)


def bennett2_wr(k, n, W, alpha=ALPHA, gamma=0.005):
    """Two stages under the with-replacement model: an upper bound V_U on V = sum W^2 p (1 - p) / n holding with
    probability 1 - gamma (Hoeffding's U-statistic representation of s^2 and Maurer's lower-tail bound for non-negative
    variables, self-bounded as in Maurer and Pontil 2009), then Bennett's inequality at each tail (alpha - gamma) / 2."""
    k, n, W = (np.asarray(v, dtype=float).ravel() for v in (k, n, W))
    p = k / n
    est = float(np.sum(W * p))
    c = W ** 2 / n
    ok = n >= 2
    s2 = np.where(ok, n * p * (1 - p) / np.maximum(n - 1, 1), 0.0)
    m = np.floor(n / 2)
    vhat = float(np.sum(np.where(ok, c * s2, 0.0)))
    kappa = float(np.max(np.where(ok, c / (2 * np.maximum(m, 1)), 0.0)))
    L1 = math.log(1 / gamma)
    VU = (math.sqrt(kappa * L1 / 2) + math.sqrt(kappa * L1 / 2 + vhat)) ** 2 + float(np.sum(np.where(ok, 0.0, c / 4)))
    if VU <= 0:
        return est, est, est
    b = float(np.max(W / n))
    target = math.log(2 / (alpha - gamma)) * b * b / VU
    h = lambda u: (1 + u) * math.log1p(u) - u
    lo, hi = 0.0, 1.0
    while h(hi) < target:
        hi *= 2
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if h(mid) < target:
            lo = mid
        else:
            hi = mid
    d = hi * VU / b
    return est, max(0.0, est - d), min(1.0, est + d)


def ebmix_wr(k, n, W, alpha=ALPHA, seed=0, p_design=0.85, prior_sd=1.0):
    """Under the with-replacement model: for lam in [0, 1), exp((lam/s)(mu_hat - mu) - psi_E(lam) Q/s^2) has expectation
    at most 1 (Fan, Grama and Liu 2015's inequality exp(lam u - psi_E(lam) u^2) <= 1 + lam u for u >= -1, with
    u = c_h (y - yhat)/s and yhat predictable), Q = sum_h c_h^2 sum_i (y_hi - yhat_hi)^2, c_h = W_h/n_h, s = max c_h.
    A fixed log-normal mixture over lam (centred from the design and a design accuracy of 0.85, never from the labels)
    keeps it an e-value; units are read in a seeded random order within each stratum."""
    k, n, W = (np.asarray(v, dtype=float).ravel() for v in (k, n, W))
    rng = np.random.default_rng(seed)
    ys = [(rng.permutation(int(m)) < kk).astype(float) for kk, m in zip(k, n)]
    c = W / n
    s = float(c.max())
    a = alpha / 2
    vdes = float(np.sum(W ** 2 * p_design * (1 - p_design) / n))
    lam0 = min(0.9, s * math.sqrt(2 * math.log(1 / a) / vdes))
    lam = np.exp(np.linspace(math.log(1e-5), math.log(0.999), 600))
    lw = -0.5 * ((np.log(lam) - math.log(lam0)) / prior_sd) ** 2
    lw = lw - np.logaddexp.reduce(lw)
    psi = -np.log1p(-lam) - lam
    target = math.log(1 / a)

    def lower(yy):
        Q = 0.0
        for h, y in enumerate(yy):
            prev = np.concatenate(([0.0], np.cumsum(y)[:-1]))
            yhat = (0.5 + prev) / np.arange(1, y.size + 1)
            Q += c[h] ** 2 * float(np.sum((y - yhat) ** 2))
        mu = float(sum(W[h] * yy[h].mean() for h in range(len(yy))))

        def logE(th):
            return float(np.logaddexp.reduce(lw + lam / s * (mu - th) - psi * Q / s ** 2))
        if logE(0.0) < target:
            return 0.0
        lo, hi = 0.0, mu
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if logE(mid) >= target:
                lo = mid
            else:
                hi = mid
        return lo
    est = float(np.sum(W * k / n))
    return est, lower(ys), 1.0 - lower([1.0 - y for y in ys])


# ============================================================================ every construction on one quantity
def _finish(out):
    nw = None
    if "normal" in out and out["normal"] is not None:
        nw = out["normal"][1] - out["normal"][0]
    out["widths_points"] = {m: 100 * (out[m][1] - out[m][0]) for m in METHODS if out.get(m) is not None}
    if nw and nw > 0:
        out["x_normal"] = {m: (out[m][1] - out[m][0]) / nw for m in METHODS if out.get(m) is not None}
    est = out.get("estimate")
    lo_e, hi_e = (est, est) if not isinstance(est, (list, tuple)) else est
    if lo_e is not None:
        out["misses_its_estimate"] = [m for m in METHODS if out.get(m) is not None and m != "normal"
                                      and not (out[m][0] - 1e-9 <= lo_e and hi_e <= out[m][1] + 1e-9)]
    return out


def share_methods(k_lo, n, W, Nf, Nint, Nbox, k_hi=None, extra=False):
    """sum_h W_h p_h by every construction. Nf: sizes for the normal theory's finite-population correction (None: no
    correction); Nint: integer sizes for the union bound (None skips it); Nbox: counts for exact boxes (None: KL boxes,
    the areas case). k_hi (default k_lo) is the count at the upper end when some units' values are unknown."""
    n = np.asarray(n, dtype=float).ravel()
    k_lo = np.asarray(k_lo, dtype=float).ravel()
    bounded = k_hi is not None
    k_hi = k_lo if k_hi is None else np.asarray(k_hi, dtype=float).ravel()
    H = n.size
    out = {}
    e_lo, e_hi = normal_linear(k_lo, 0 * n, n, W, 1.0, 0.0, Nf), normal_linear(k_hi, 0 * n, n, W, 1.0, 0.0, Nf)
    out["estimate"] = [e_lo[0], e_hi[0]] if bounded else e_lo[0]
    out["normal"] = [e_lo[1], e_hi[2]]
    if Nint is not None and not bounded:
        out["union"] = list(union_share(k_lo, n, W, Nint))
    cen = np.zeros(H, bool) if Nbox is None else (n >= np.asarray(Nbox).ravel())
    z, o = np.zeros(H), np.ones(H)
    t = time.time()
    out["chernoff_global"] = list(two_point(k_lo, n, W, ALPHA / 2, (z, o), cen, k_hi=k_hi))
    out["seconds_global"] = time.time() - t
    L = max(int((~cen).sum()), 1)
    for gam, name in ((GAMMA, "chernoff_boxed"),) + (((0.002, "chernoff_boxed_gamma_0.002"),
                                                      (0.005, "chernoff_boxed_gamma_0.005"),
                                                      (0.02, "chernoff_boxed_gamma_0.02")) if extra else ()):
        t = time.time()
        lo1, hi1, _ = box_for(k_lo, n, Nbox, gam / L)
        lo2, hi2, _ = box_for(k_hi, n, Nbox, gam / L)
        out[name] = list(two_point(k_lo, n, W, (ALPHA - gam) / 2, (np.minimum(lo1, lo2), np.maximum(hi1, hi2)), cen,
                                   k_hi=k_hi))
        if name == "chernoff_boxed":
            out["seconds_boxed"] = time.time() - t
    if extra and not bounded:
        out["bennett2_wr"] = list(bennett2_wr(k_lo, n, W)[1:])
        t = time.time()
        out["ebmix_wr"] = list(ebmix_wr(k_lo, n, W)[1:])
        out["seconds_ebmix"] = time.time() - t
    return _finish(out)


def ratio_methods(kt, kf, n, W, Nf, Nint, Nbox):
    kt, kf, n = (np.asarray(v, dtype=float).ravel() for v in (kt, kf, n))
    nr = normal_ratio(kt, kf, n, W, Nf)
    if nr is None:
        return {"estimate": None, "note": "no labelled point in the denominator"}
    out = {"estimate": nr[0], "normal": [nr[1], nr[2]]}
    if Nint is not None:
        out["union"] = list(union_ratio(kt, kf, n, W, Nint))
    H = n.size
    cen = np.zeros(H, bool) if Nbox is None else (n >= np.asarray(Nbox).ravel())
    z, o = np.zeros(H), np.ones(H)
    t = time.time()
    out["chernoff_global"] = list(ratio(kt, kf, n, W, ALPHA / 2, (z, o, z, o), cen))
    out["seconds_global"] = time.time() - t
    L = max(int((~cen).sum()), 1)
    lt, ht, _ = box_for(kt, n, Nbox, GAMMA / (2 * L))
    lf, hf, _ = box_for(kf, n, Nbox, GAMMA / (2 * L))
    t = time.time()
    out["chernoff_boxed"] = list(ratio(kt, kf, n, W, (ALPHA - GAMMA) / 2, (lt, ht, lf, hf), cen))
    out["seconds_boxed"] = time.time() - t
    return _finish(out)


def diff_methods(kp_lo, km_lo, n, W, Nf, Nint, Nbox, kp_hi=None, km_hi=None):
    """sum W (P(A only right) - P(B only right)), the accuracy of A minus that of B; *_hi give the upper end where some
    of B's values are unknown (counted right at the lower end of A - B, wrong at the upper)."""
    n = np.asarray(n, dtype=float).ravel()
    kp_lo, km_lo = (np.asarray(v, dtype=float).ravel() for v in (kp_lo, km_lo))
    bounded = kp_hi is not None
    kp_hi = kp_lo if kp_hi is None else np.asarray(kp_hi, dtype=float).ravel()
    km_hi = km_lo if km_hi is None else np.asarray(km_hi, dtype=float).ravel()
    H = n.size
    e_lo, e_hi = normal_linear(kp_lo, km_lo, n, W, 1.0, -1.0, Nf), normal_linear(kp_hi, km_hi, n, W, 1.0, -1.0, Nf)
    out = {"estimate": [e_lo[0], e_hi[0]] if bounded else e_lo[0], "normal": [e_lo[1], e_hi[2]]}
    if Nint is not None and not bounded:
        out["union"] = list(union_diff(kp_lo, km_lo, n, W, Nint))
    cen = np.zeros(H, bool) if Nbox is None else (n >= np.asarray(Nbox).ravel())
    z, o = np.zeros(H), np.ones(H)
    t = time.time()
    out["chernoff_global"] = list(three_point(kp_lo, km_lo, n, W, ALPHA / 2, (z, o, z, o), cen, kp_hi=kp_hi,
                                              km_hi=km_hi))
    out["seconds_global"] = time.time() - t
    L = max(int((~cen).sum()), 1)
    boxes = []
    for a, b in ((kp_lo, kp_hi), (km_lo, km_hi)):
        l1, h1, _ = box_for(a, n, Nbox, GAMMA / (2 * L))
        l2, h2, _ = box_for(b, n, Nbox, GAMMA / (2 * L))
        boxes += [np.minimum(l1, l2), np.maximum(h1, h2)]
    t = time.time()
    out["chernoff_boxed"] = list(three_point(kp_lo, km_lo, n, W, (ALPHA - GAMMA) / 2, tuple(boxes), cen, kp_hi=kp_hi,
                                             km_hi=km_hi))
    out["seconds_boxed"] = time.time() - t
    lo, hi = out["chernoff_boxed"]
    out["verdict_boxed"] = "a" if lo > 0 else "b" if hi < 0 else "undetermined"
    return _finish(out)


# ============================================================================ data
def load_nlcd(rec, year):
    """NLCD 2016 AA (set B) for one year: weights are N_h / n_h, so a stratum's size is its points' summed weight."""
    path = os.path.join(rec, "exp", "out", "exp94_nlcd_points.csv")
    rows = [r for r in csv.DictReader(open(path)) if r["set"] == "B" and int(r["year"]) == year]
    col = lambda k, f=int: np.array([f(r[k]) if r[k] != "" else 0 for r in rows])
    st, w = col("stratum"), col("weight", float)
    ids = np.array(sorted(set(st.tolist())))
    idx = np.searchsorted(ids, st)
    H = ids.size
    n = np.bincount(idx, minlength=H).astype(float)
    Nf = np.bincount(idx, weights=w, minlength=H)
    spread = max(float(w[idx == h].max() / w[idx == h].min() - 1) for h in range(H))
    return {"ids": ids, "idx": idx, "H": H, "n": n, "Nf": Nf, "Nint": np.maximum(np.round(Nf), n).astype(np.int64),
            "W": Nf / Nf.sum(), "ref": col("ref_nlcd"), "nl": col("nlcd"), "alt": col("ref_nlcd_alternate"),
            "lc": col("lcmap"), "weight_spread_within_strata": spread, "points": len(rows)}


def load_gfc(rec, areas):
    """GFC2020: 149 continental strata with areas in hectares. sizing.py took 10 m pixels (0.01 ha) as the units for
    the finite-population correction and the union bound; the boxes here use only the areas (KL boxes)."""
    area = {int(r["Strata"]): float(r["strata_ha"]) for r in csv.DictReader(open(areas))}
    rows = list(csv.DictReader(open(os.path.join(rec, "exp", "out", "exp94_gfc_points.csv"))))
    st = np.array([int(r["stratum"]) for r in rows])
    keep = np.array([s in area for s in st])
    ids_all = np.array(sorted(area))
    n_all = np.bincount(np.searchsorted(ids_all, st[keep]), minlength=ids_all.size)
    A_all = np.array([area[s] for s in ids_all])
    empty = n_all == 0
    ids = ids_all[~empty]                               # as sizing.py: the strata that hold points
    idx = np.searchsorted(ids, st[keep])
    H = ids.size
    n = np.bincount(idx, minlength=H).astype(float)
    A = np.array([area[s] for s in ids])
    get = lambda k: np.array([int(r[k]) if r[k] != "" else -1 for r in rows])[keep]
    return {"ids": ids, "idx": idx, "H": H, "n": n, "A": A, "W": A / A.sum(), "Nf": A / 0.01,
            "Nint": np.maximum(np.round(A / 0.01), n).astype(np.int64), "ref": get("ref_forest"), "gfc": get("gfc_v2"),
            "wc": get("worldcover_forest"), "outside_area_table": int((~keep).sum()), "points": int(keep.sum()),
            "empty_strata": ids_all[empty].tolist(), "empty_area_share": float(A_all[empty].sum() / A_all.sum()),
            "area_total_ha": float(A_all.sum())}


def cnt(d, mask):
    return np.bincount(d["idx"], weights=np.asarray(mask, dtype=float), minlength=d["H"])


def describe(d):
    n, W = d["n"], d["W"]
    return {"points": d["points"], "strata": d["H"], "n_min": int(n.min()), "n_max": int(n.max()),
            "W_max": float(W.max()), "max_W_over_n": float((W / n).max()),
            "ratio_max_to_min_W_over_n": float((W / n).max() / (W / n).min())}


def _close(got, want, tol=1e-9):
    if isinstance(want, (list, tuple)):
        return {"got": list(got), "want": list(want), "ok": all(abs(a - b) <= tol for a, b in zip(got, want))}
    return {"got": got, "want": want, "ok": abs(got - want) <= tol}


def fieller_lower(kt, kf, n, W, a_tail, box, iters=30):
    """the Fieller-type bound with the direction tied continuously to the tested value (lt = lam (1 - theta0),
    lf = -lam theta0): a bisection on theta0 of the linear test of (1 - theta0) tau - theta0 phi <= 0, which assumes
    the rejected values are nested (not guaranteed; a check only). ratio_lower fixes the direction on pieces of width
    1/PIECES, so the two should agree to about that width (a check of its optimiser and its scan)."""
    T = float(np.sum(W * np.asarray(kt) / n))
    Fh = float(np.sum(W * np.asarray(kf) / n))
    lo, hi = 0.0, T / (T + Fh)
    for _ in range(iters):
        th = 0.5 * (lo + hi)
        if linear_lower(kt, kf, n, W, 1 - th, -th, a_tail, box) > 0:
            lo = th
        else:
            hi = th
    return lo


# ============================================================================ commands
def real(rec, areas, combined=None):
    out = {"published": PUBLISHED, "gamma": GAMMA, "alpha": ALPHA, "checks": {}, "data": {}}
    chk = out["checks"]
    for year in (2016, 2011):
        d = load_nlcd(rec, year)
        key = f"nlcd2016_B_{year}"
        res = out[key] = {}
        info = out["data"][key] = describe(d)
        info.update({"weight_spread_within_strata": d["weight_spread_within_strata"],
                     "alternate_equals_primary": int((d["alt"] == d["ref"]).sum()),
                     "reference_codes": sorted(set(d["ref"].tolist())), "map_codes": sorted(set(d["nl"].tolist())),
                     "alternate_codes": sorted(set(d["alt"].tolist()))})
        n, W, Nf, Ni = d["n"], d["W"], d["Nf"], d["Nint"]
        ap = d["nl"] == d["ref"]
        aa = ap | (d["nl"] == d["alt"])
        t = time.time()
        res["oa_level2_primary"] = share_methods(cnt(d, ap), n, W, Nf, Ni, Ni, extra=(year == 2016))
        res["oa_level2_primary_or_alternate"] = share_methods(cnt(d, aa), n, W, Nf, Ni, Ni, extra=(year == 2016))
        res["seconds_oa"] = time.time() - t
        chk[f"nlcd{year}_primary_or_alternate_vs_record"] = _close(res["oa_level2_primary_or_alternate"]["estimate"],
                                                                   RECORD_ALT[year])
        if year != 2016:
            continue
        o = res["oa_level2_primary"]
        s = SIZING["nlcd16_l2_primary"]
        chk["nlcd16_primary_estimate"] = _close(o["estimate"], s[0])
        chk["nlcd16_primary_normal"] = _close(o["normal"], s[1])
        chk["nlcd16_primary_union"] = _close(o["union"], s[2])
        # per class, primary or alternate; the reference class is the map's where it matches the alternate label
        rstar = np.where(d["nl"] == d["alt"], d["nl"], d["ref"])
        classes = sorted(set(d["nl"].tolist()) | set(rstar.tolist()))
        pc = res["per_class_primary_or_alternate"] = {}
        t = time.time()
        for c in classes:
            m, r = d["nl"] == c, rstar == c
            pc[str(c)] = {"n_mapped": int(m.sum()), "n_reference": int(r.sum()),
                          "user": ratio_methods(cnt(d, m & r), cnt(d, m & ~r), n, W, Nf, Ni, Ni),
                          "producer": ratio_methods(cnt(d, m & r), cnt(d, ~m & r), n, W, Nf, Ni, Ni),
                          "area_share_primary": share_methods(cnt(d, d["ref"] == c), n, W, Nf, None, Ni)}
        res["seconds_per_class"] = time.time() - t
        m, r = d["nl"] == 52, d["ref"] == 52                  # sizing.py's union for class 52, primary label
        chk["nlcd16_ua52_primary_union"] = _close(list(union_ratio(cnt(d, m & r), cnt(d, m & ~r), n, W, Ni)),
                                                  SIZING["nlcd16_ua52_primary_union"][1])
        m, r = d["nl"] == 52, rstar == 52
        z, one = np.zeros(d["H"]), np.ones(d["H"])
        kt, kf = cnt(d, m & r), cnt(d, m & ~r)
        chk["ratio_optimiser_vs_fieller_class52"] = {
            "ratio_lower": ratio_lower(kt, kf, n, W, ALPHA / 2, (z, one, z, one)),
            "fieller_lower": fieller_lower(kt, kf, n, W, ALPHA / 2, (z, one, z, one))}
        chk["ratio_optimiser_vs_fieller_class52"]["ok"] = abs(chk["ratio_optimiser_vs_fieller_class52"]["ratio_lower"]
                                                              - chk["ratio_optimiser_vs_fieller_class52"]["fieller_lower"]) <= 0.01
        # NLCD against LCMAP in LCMAP's eight classes (USGS's crosswalk, as exp94); LCMAP 0 is no value: bounded
        cw = lambda a: np.array([NLCD_TO_LCMAP.get(int(v), 0) for v in a])
        ref8, alt8, nl8, lc = cw(d["ref"]), cw(d["alt"]), cw(d["nl"]), d["lc"]
        miss = lc == 0
        info.update({"lcmap_missing": int(miss.sum()), "reference_outside_crosswalk": int((ref8 == 0).sum())})
        for rule in ("primary", "primary_or_alternate"):
            use_alt = rule == "primary_or_alternate"
            ok_a = (nl8 == ref8) | (use_alt & (nl8 == alt8))
            ok_b = ((lc == ref8) | (use_alt & (lc == alt8))) & ~miss
            b_lo, b_hi = ok_b | miss, ok_b
            res[f"nlcd_minus_lcmap_8class_{rule}"] = diff_methods(
                cnt(d, ok_a & ~b_lo), cnt(d, ~ok_a & b_lo), n, W, Nf, Ni, Ni,
                **({"kp_hi": cnt(d, ok_a & ~b_hi), "km_hi": cnt(d, ~ok_a & b_hi)} if miss.any() else {}))
    # ------------------------------------------------------------------ GFC2020
    g = load_gfc(rec, areas)
    res = out["gfc2020"] = {}
    info = out["data"]["gfc2020"] = describe(g)
    info.update({k: g[k] for k in ("outside_area_table", "empty_strata", "empty_area_share", "area_total_ha")})
    n, W, Nf, Ni = g["n"], g["W"], g["Nf"], g["Nint"]
    ok_g = g["gfc"] == g["ref"]
    miss_w = g["wc"] < 0
    ok_w = (g["wc"] == g["ref"]) & ~miss_w
    info["worldcover_missing"] = int(miss_w.sum())
    t = time.time()
    res["oa_gfc_v2"] = share_methods(cnt(g, ok_g), n, W, Nf, Ni, None, extra=True)
    res["oa_worldcover"] = share_methods(cnt(g, ok_w), n, W, Nf, None if miss_w.any() else Ni, None,
                                         k_hi=cnt(g, ok_w | miss_w) if miss_w.any() else None)
    res["seconds_oa"] = time.time() - t
    s = SIZING["gfc_oa"]
    chk["gfc_estimate"] = _close(res["oa_gfc_v2"]["estimate"], s[0])
    chk["gfc_normal"] = _close(res["oa_gfc_v2"]["normal"], s[1])
    chk["gfc_union"] = _close(res["oa_gfc_v2"]["union"], s[2])
    t = time.time()
    for c, name in ((1, "forest"), (0, "non_forest")):
        m, r = g["gfc"] == c, g["ref"] == c
        res[f"user_{name}"] = ratio_methods(cnt(g, m & r), cnt(g, m & ~r), n, W, Nf, Ni, None)
        res[f"producer_{name}"] = ratio_methods(cnt(g, m & r), cnt(g, ~m & r), n, W, Nf, Ni, None)
    res["seconds_per_class"] = time.time() - t
    fa = share_methods(cnt(g, g["ref"] == 1), n, W, Nf, None, None)
    total_mha = g["area_total_ha"] * (1 - g["empty_area_share"]) / 1e6
    fa["forest_area_Mha"] = {m: [v * total_mha for v in fa[m]] for m in METHODS if fa.get(m) is not None}
    res["forest_area_share"] = fa
    b_lo, b_hi = ok_w | miss_w, ok_w
    res["gfc_minus_worldcover"] = diff_methods(
        cnt(g, ok_g & ~b_lo), cnt(g, ~ok_g & b_lo), n, W, Nf, None if miss_w.any() else Ni, None,
        **({"kp_hi": cnt(g, ok_g & ~b_hi), "km_hi": cnt(g, ~ok_g & b_hi)} if miss_w.any() else {}))
    if not miss_w.any():
        s = SIZING["gfc_diff"]
        chk["gfc_diff_estimate"] = _close(res["gfc_minus_worldcover"]["estimate"], s[0])
        chk["gfc_diff_normal"] = _close(res["gfc_minus_worldcover"]["normal"], s[1])
        chk["gfc_diff_union"] = _close(res["gfc_minus_worldcover"]["union"], s[2])
    if combined:
        res["oa_gfc_v2_no_assignment_bounded"] = gfc_unassigned(combined, areas)
    # ------------------------------------------------------------------ closed-form checks of the optimiser
    one = np.ones(1)
    got = two_point(np.array([37.0]), np.array([100.0]), one, ALPHA / 2, (np.zeros(1), one))[0]
    want = float(kl_bounds(np.array([37.0]), np.array([100.0]), math.log(1 / (ALPHA / 2)))[0][0])
    chk["single_stratum_chernoff_equals_kl"] = {"got": got, "want": want, "ok": abs(got - want) <= 1e-6}
    # ------------------------------------------------------------------ the end-to-end table
    out["e2e"] = {
        "nlcd2016_oa_primary_or_alternate": {"published": PUBLISHED["nlcd2016_B_2016_oa_level2_primary_or_alternate"],
                                             **{k: out["nlcd2016_B_2016"]["oa_level2_primary_or_alternate"].get(k)
                                                for k in ("estimate", "normal", "chernoff_boxed")}},
        "nlcd2011_oa_primary_or_alternate": {"published": PUBLISHED["nlcd2016_B_2011_oa_level2_primary_or_alternate"],
                                             **{k: out["nlcd2016_B_2011"]["oa_level2_primary_or_alternate"].get(k)
                                                for k in ("estimate", "normal", "chernoff_boxed")}},
        "gfc2020_v2_oa": {"published": PUBLISHED["gfc2020_v2_oa"],
                          **{k: res["oa_gfc_v2"].get(k) for k in ("estimate", "normal", "chernoff_boxed")}},
        "per_class_published": "to transcribe (see published.source)"}
    viol = []

    def walk(node, path):
        if isinstance(node, dict):
            if node.get("misses_its_estimate"):
                viol.append({"at": path, "methods": node["misses_its_estimate"]})
            for k, v in node.items():
                walk(v, f"{path}/{k}")
    walk({k: out[k] for k in ("nlcd2016_B_2016", "nlcd2016_B_2011", "gfc2020")}, "")
    chk["intervals_missing_their_estimate"] = viol
    chk["all_ok"] = all(v.get("ok", True) for v in chk.values() if isinstance(v, dict)) and not viol
    return out


def gfc_unassigned(combined, areas):
    """GFC2020 V2's overall accuracy with the units of no assignment (forest_class_num 100) inside GAUL and a stratum
    kept and bounded both ways: counted wrong at the lower end, right at the upper (the package's `?` rule)."""
    area = {int(r["Strata"]): float(r["strata_ha"]) for r in csv.DictReader(open(areas))}
    rows = [r for r in csv.DictReader(open(combined)) if r["gaul"] == "1" and int(r["strata"]) in area]
    st = np.array([int(r["strata"]) for r in rows])
    fc = np.array([int(r["forest_class_num"]) for r in rows])
    gv = np.array([int(r["GFC_v2"]) for r in rows])
    ids = np.array(sorted(set(st.tolist())))
    idx = np.searchsorted(ids, st)
    H = ids.size
    n = np.bincount(idx, minlength=H).astype(float)
    A = np.array([area[s] for s in ids])
    W = A / A.sum()
    un = fc == 100
    k_lo = np.bincount(idx, weights=((fc == gv) & ~un).astype(float), minlength=H)
    k_hi = k_lo + np.bincount(idx, weights=un.astype(float), minlength=H)
    res = share_methods(k_lo, n, W, A / 0.01, None, None, k_hi=k_hi)
    res["units"] = int(n.sum())
    res["no_assignment"] = int(un.sum())
    res["other_codes"] = sorted(set(fc.tolist()) - {0, 1, 100})
    return res


# ---------------------------------------------------------------------------- enumeration
def _comps(m, parts):
    return [c for c in product(range(m + 1), repeat=parts) if sum(c) == m]


def _mvh(pop, out, Nh, nh):
    p = 1
    for P, o in zip(pop, out):
        p *= math.comb(P, o)
    return p / math.comb(Nh, nh)


def _coverage(P, iv, theta, defined=None):
    """P: (populations, outcomes) probabilities; iv: (outcomes, 2) intervals; theta: (populations,) truths."""
    cover = (iv[None, :, 0] <= theta[:, None] + 1e-12) & (theta[:, None] - 1e-12 <= iv[None, :, 1])
    cov = (P * cover).sum(1)
    if defined is not None:
        cov = np.where(defined, cov, np.inf)
    i = int(np.argmin(cov))
    return float(cov[i]), i


def enum_share_counts(N, n, areas_mode=False):
    """sum_h W_h K_h / N_h from a stratified sample without replacement: every population and every outcome. In
    areas_mode the intervals use only W (KL boxes, binomial MGFs) while the draws stay without replacement."""
    N = np.array(N, dtype=np.int64)
    n = np.array(n, dtype=float)
    H = N.size
    W = N / N.sum()
    Nbox = None if areas_mode else N
    Nf = None if areas_mode else N.astype(float)
    cen = np.zeros(H, bool) if areas_mode else (n >= N)
    L = max(int((~cen).sum()), 1)
    z, o = np.zeros(H), np.ones(H)

    def boxed(k):
        lo, hi, c = box_for(k, n, Nbox, GAMMA / L)
        return two_point(k, n, W, (ALPHA - GAMMA) / 2, (lo, hi), c)
    methods = {"normal": lambda k: normal_linear(k, 0 * k, n, W, 1.0, 0.0, Nf)[1:],
               "chernoff_global": lambda k: two_point(k, n, W, ALPHA / 2, (z, o), cen),
               "chernoff_boxed": boxed,
               "chernoff_global_coarse_optimiser": lambda k: two_point(k, n, W, ALPHA / 2, (z, o), cen, iters=2, g=5),
               "bennett2_wr": lambda k: bennett2_wr(k, n, W)[1:],
               "ebmix_wr": lambda k: ebmix_wr(k, n, W)[1:]}
    if not areas_mode:
        methods["union"] = lambda k: union_share(k, n, W, N)
    outs = list(product(*[range(int(m) + 1) for m in n]))
    pops = list(product(*[range(int(M) + 1) for M in N]))

    def pmf(K, k):
        p = 1.0
        for h in range(H):
            p *= (math.comb(K[h], k[h]) * math.comb(int(N[h]) - K[h], int(n[h]) - k[h])
                  / math.comb(int(N[h]), int(n[h])))
        return p
    P = np.array([[pmf(K, k) for k in outs] for K in pops])
    theta = np.array([float(np.sum(W * np.array(K) / N)) for K in pops])
    res = {"N": N.tolist(), "n": n.tolist(), "areas_mode": areas_mode, "populations": len(pops), "outcomes": len(outs)}
    for name, f in methods.items():
        t = time.time()
        iv = np.array([f(np.array(k, dtype=float)) for k in outs], dtype=float)
        cov, i = _coverage(P, iv, theta)
        res[name] = {"min_coverage": cov, "at_population": list(pops[i]), "mean_width": float(np.mean(iv[:, 1] - iv[:, 0])),
                     "seconds": time.time() - t}
    return res


def enum_share_binomial(W, n, grid=21):
    """sum_h W_h p_h from independent binomial strata (the areas case): every outcome, p on a grid."""
    W = np.array(W, dtype=float)
    n = np.array(n, dtype=float)
    H = W.size
    ps = np.linspace(0.0, 1.0, grid)
    pmf = [np.array([[math.comb(int(n[h]), k) * p ** k * (1 - p) ** (int(n[h]) - k) for p in ps]
                     for k in range(int(n[h]) + 1)]) for h in range(H)]
    mesh = np.meshgrid(*([ps] * H), indexing="ij")
    theta = sum(W[h] * mesh[h] for h in range(H))
    z, o = np.zeros(H), np.ones(H)

    def boxed(k):
        lo, hi, c = box_for(k, n, None, GAMMA / H)
        return two_point(k, n, W, (ALPHA - GAMMA) / 2, (lo, hi), c)
    methods = {"normal": lambda k: normal_linear(k, 0 * k, n, W, 1.0, 0.0, None)[1:],
               "chernoff_global": lambda k: two_point(k, n, W, ALPHA / 2, (z, o)),
               "chernoff_boxed": boxed,
               "chernoff_global_coarse_optimiser": lambda k: two_point(k, n, W, ALPHA / 2, (z, o), iters=2, g=5),
               "bennett2_wr": lambda k: bennett2_wr(k, n, W)[1:],
               "ebmix_wr": lambda k: ebmix_wr(k, n, W)[1:]}
    outs = list(product(*[range(int(m) + 1) for m in n]))
    res = {"W": W.tolist(), "n": n.tolist(), "grid": grid, "outcomes": len(outs)}
    for name, f in methods.items():
        t = time.time()
        cov = np.zeros([grid] * H)
        widths = []
        for k in outs:
            lo, hi = f(np.array(k, dtype=float))
            widths.append(hi - lo)
            prob = np.ones([grid] * H)
            for h in range(H):
                shape = [1] * H
                shape[h] = grid
                prob = prob * pmf[h][k[h]].reshape(shape)
            cov += prob * ((theta >= lo - 1e-12) & (theta <= hi + 1e-12))
        i = np.unravel_index(int(np.argmin(cov)), cov.shape)
        res[name] = {"min_coverage": float(cov[i]), "at_p": [float(ps[j]) for j in i],
                     "mean_width": float(np.mean(widths)), "seconds": time.time() - t}
    return res


def enum_three_category(N, n, kind):
    """kind 'ratio': categories (t, f, other), truth sum W t / sum W (t + f); kind 'difference': categories
    (A only right, B only right, other), truth sum W (P - M). Multivariate hypergeometric draws in each stratum."""
    N = np.array(N, dtype=np.int64)
    n = np.array(n, dtype=float)
    H = N.size
    W = N / N.sum()
    pops_h = [_comps(int(N[h]), 3) for h in range(H)]
    outs_h = [_comps(int(n[h]), 3) for h in range(H)]
    Ph = [np.array([[_mvh(p, o, int(N[h]), int(n[h])) for o in outs_h[h]] for p in pops_h[h]]) for h in range(H)]
    joint_outs = list(product(*[range(len(outs_h[h])) for h in range(H)]))
    joint_pops = list(product(*[range(len(pops_h[h])) for h in range(H)]))
    cen = n >= N
    L = max(int((~cen).sum()), 1)
    z, o = np.zeros(H), np.ones(H)

    def counts(jo):
        return (np.array([outs_h[h][jo[h]][0] for h in range(H)], dtype=float),
                np.array([outs_h[h][jo[h]][1] for h in range(H)], dtype=float))

    def boxes(a, b):
        la, ua, _ = box_for(a, n, N, GAMMA / (2 * L))
        lb, ub, _ = box_for(b, n, N, GAMMA / (2 * L))
        return (la, ua, lb, ub)
    if kind == "ratio":
        def normal(a, b):
            r = normal_ratio(a, b, n, W, N.astype(float))
            return (0.0, 1.0) if r is None else (r[1], r[2])
        methods = {"normal": normal,
                   "chernoff_global": lambda a, b: ratio(a, b, n, W, ALPHA / 2, (z, o, z, o), cen),
                   "chernoff_boxed": lambda a, b: ratio(a, b, n, W, (ALPHA - GAMMA) / 2, boxes(a, b), cen),
                   "chernoff_boxed_coarse_optimiser": lambda a, b: ratio(a, b, n, W, (ALPHA - GAMMA) / 2, boxes(a, b),
                                                                         cen, K=40, iters=2, g=3, bis=8)}
    else:
        methods = {"normal": lambda a, b: normal_linear(a, b, n, W, 1.0, -1.0, N.astype(float))[1:],
                   "union": lambda a, b: union_diff(a, b, n, W, N),
                   "chernoff_global": lambda a, b: three_point(a, b, n, W, ALPHA / 2, (z, o, z, o), cen),
                   "chernoff_boxed": lambda a, b: three_point(a, b, n, W, (ALPHA - GAMMA) / 2, boxes(a, b), cen),
                   "chernoff_boxed_coarse_optimiser": lambda a, b: three_point(a, b, n, W, (ALPHA - GAMMA) / 2,
                                                                               boxes(a, b), cen, iters=2, g=5)}
    theta, defined = [], []
    for jp in joint_pops:
        a = sum(W[h] * pops_h[h][jp[h]][0] / N[h] for h in range(H))
        b = sum(W[h] * pops_h[h][jp[h]][1] / N[h] for h in range(H))
        if kind == "ratio":
            defined.append(a + b > 0)
            theta.append(a / (a + b) if a + b > 0 else np.nan)
        else:
            defined.append(True)
            theta.append(a - b)
    theta, defined = np.array(theta), np.array(defined)
    P = np.array([[float(np.prod([Ph[h][jp[h], jo[h]] for h in range(H)])) for jo in joint_outs] for jp in joint_pops])
    res = {"kind": kind, "N": N.tolist(), "n": n.tolist(), "populations": len(joint_pops), "outcomes": len(joint_outs)}
    for name, f in methods.items():
        t = time.time()
        iv = np.array([f(*counts(jo)) for jo in joint_outs], dtype=float)
        cov, i = _coverage(P, iv, np.nan_to_num(theta, nan=-1.0), defined)
        res[name] = {"min_coverage": cov, "at_population": [pops_h[h][joint_pops[i][h]] for h in range(H)],
                     "mean_width": float(np.mean(iv[:, 1] - iv[:, 0])), "seconds": time.time() - t}
    return res


def enum_unassigned(N, n):
    """Overall accuracy with units the reference could not assign: per stratum (right, wrong, unassigned and truly
    right, unassigned and truly wrong); the reviewer sees (right, wrong, unassigned). The bounded rule counts the
    unassigned wrong at the lower end and right at the upper; the agency's drop rule leaves them out."""
    N = np.array(N, dtype=np.int64)
    n = np.array(n, dtype=float)
    H = N.size
    W = N / N.sum()
    pops_h = [_comps(int(N[h]), 4) for h in range(H)]
    outs_h = [_comps(int(n[h]), 3) for h in range(H)]

    def pr(pop, out, Nh, nh):
        R, X, U1, U0 = pop
        r, x, u = out
        return math.comb(R, r) * math.comb(X, x) * math.comb(U1 + U0, u) / math.comb(Nh, nh)
    Ph = [np.array([[pr(p, o, int(N[h]), int(n[h])) for o in outs_h[h]] for p in pops_h[h]]) for h in range(H)]
    joint_outs = list(product(*[range(len(outs_h[h])) for h in range(H)]))
    joint_pops = list(product(*[range(len(pops_h[h])) for h in range(H)]))
    L = H

    def obs(jo):
        r = np.array([outs_h[h][jo[h]][0] for h in range(H)], dtype=float)
        x = np.array([outs_h[h][jo[h]][1] for h in range(H)], dtype=float)
        u = np.array([outs_h[h][jo[h]][2] for h in range(H)], dtype=float)
        return r, x, u

    def bounded(jo, boxed=True):
        r, x, u = obs(jo)
        if not boxed:
            return two_point(r, n, W, ALPHA / 2, (np.zeros(H), np.ones(H)), k_hi=r + u)
        lo1, hi1, _ = box_for(r, n, N, GAMMA / L)
        lo2, hi2, _ = box_for(r + u, n, N, GAMMA / L)
        return two_point(r, n, W, (ALPHA - GAMMA) / 2, (np.minimum(lo1, lo2), np.maximum(hi1, hi2)), k_hi=r + u)

    def dropped(jo, boxed):
        r, x, u = obs(jo)
        m = r + x
        if (m == 0).any():
            return (0.0, 1.0)
        if not boxed:
            return normal_linear(r, 0 * r, m, W, 1.0, 0.0, None)[1:]
        lo, hi, _ = box_for(r, m, None, GAMMA / L)
        return two_point(r, m, W, (ALPHA - GAMMA) / 2, (lo, hi))
    methods = {"bounded_boxed": lambda jo: bounded(jo, True), "bounded_global": lambda jo: bounded(jo, False),
               "drop_normal": lambda jo: dropped(jo, False), "drop_boxed": lambda jo: dropped(jo, True)}
    theta = np.array([sum(W[h] * (pops_h[h][jp[h]][0] + pops_h[h][jp[h]][2]) / N[h] for h in range(H))
                      for jp in joint_pops])
    P = np.array([[float(np.prod([Ph[h][jp[h], jo[h]] for h in range(H)])) for jo in joint_outs] for jp in joint_pops])
    res = {"N": N.tolist(), "n": n.tolist(), "populations": len(joint_pops), "outcomes": len(joint_outs)}
    for name, f in methods.items():
        t = time.time()
        iv = np.array([f(jo) for jo in joint_outs], dtype=float)
        cov, i = _coverage(P, iv, theta)
        res[name] = {"min_coverage": cov, "at_population": [pops_h[h][joint_pops[i][h]] for h in range(H)],
                     "mean_width": float(np.mean(iv[:, 1] - iv[:, 0])), "seconds": time.time() - t}
    return res


def enum():
    return {"E1_share_counts": enum_share_counts((6, 9), (3, 4)),
            "E1b_share_counts_unequal": enum_share_counts((10, 4), (2, 3)),
            "E2_share_binomial_three_strata": enum_share_binomial((0.6, 0.3, 0.1), (3, 5, 8), grid=21),
            "E3_share_areas_mode_drawn_without_replacement": enum_share_counts((5, 7), (3, 4), areas_mode=True),
            "E4_ratio_counts": enum_three_category((4, 8), (3, 3), "ratio"),
            "E5_difference_counts": enum_three_category((5, 6), (3, 3), "difference"),
            "E6_unassigned": enum_unassigned((5, 5), (3, 3))}


# ---------------------------------------------------------------------------- Monte Carlo at the data's shapes
def _mc_one(arg):
    c, seed = arg
    rng = np.random.default_rng([94, int(seed)])
    n, W, Nbox = c["n"], c["W"], c["Nbox"]
    L = n.size
    if c["kind"] == "share":
        k = rng.binomial(n.astype(int), c["p"]).astype(float)
        lo, hi, cen = box_for(k, n, Nbox, GAMMA / L)
        b = two_point(k, n, W, (ALPHA - GAMMA) / 2, (lo, hi), cen)
        nt = normal_linear(k, 0 * k, n, W, 1.0, 0.0, None)[1:]
    else:
        draws = np.array([rng.multinomial(int(m), pr) for m, pr in zip(n, c["probs"])], dtype=float)
        a, bb = draws[:, 0], draws[:, 1]
        la, ua, cen = box_for(a, n, Nbox, GAMMA / (2 * L))
        lb, ub, _ = box_for(bb, n, Nbox, GAMMA / (2 * L))
        if c["kind"] == "ratio":
            b = ratio(a, bb, n, W, (ALPHA - GAMMA) / 2, (la, ua, lb, ub), cen)
            r = normal_ratio(a, bb, n, W, None)
            nt = (0.0, 1.0) if r is None else r[1:]
        else:
            b = three_point(a, bb, n, W, (ALPHA - GAMMA) / 2, (la, ua, lb, ub), cen)
            nt = normal_linear(a, bb, n, W, 1.0, -1.0, None)[1:]
    th = c["theta"]
    return (float(b[0] <= th <= b[1]), b[1] - b[0], float(nt[0] <= th <= nt[1]), nt[1] - nt[0])


def mc(rec, areas, R=200):
    import multiprocessing as mp
    d = load_nlcd(rec, 2016)
    g = load_gfc(rec, areas)
    cases = {}
    kg = cnt(g, g["gfc"] == g["ref"])
    cases["gfc_oa"] = {"kind": "share", "n": g["n"], "W": g["W"], "p": kg / g["n"], "Nbox": None,
                       "theta": float(np.sum(g["W"] * kg / g["n"]))}
    ka = cnt(d, (d["nl"] == d["ref"]) | (d["nl"] == d["alt"]))
    cases["nlcd2016_oa_primary_or_alternate"] = {"kind": "share", "n": d["n"], "W": d["W"], "p": ka / d["n"],
                                                 "Nbox": d["Nint"], "theta": float(np.sum(d["W"] * ka / d["n"]))}
    rstar = np.where(d["nl"] == d["alt"], d["nl"], d["ref"])
    for c in (52, 23):
        m, r = d["nl"] == c, rstar == c
        t, f = cnt(d, m & r) / d["n"], cnt(d, m & ~r) / d["n"]
        probs = np.stack([t, f, np.clip(1 - t - f, 0, 1)], 1)
        probs = probs / probs.sum(1, keepdims=True)
        cases[f"nlcd2016_user_{c}"] = {"kind": "ratio", "n": d["n"], "W": d["W"], "probs": probs, "Nbox": d["Nint"],
                                       "theta": float(np.sum(d["W"] * t) / np.sum(d["W"] * (t + f)))}
    okg, okw = g["gfc"] == g["ref"], (g["wc"] == g["ref"]) & (g["wc"] >= 0)
    pp, pm = cnt(g, okg & ~okw) / g["n"], cnt(g, ~okg & okw) / g["n"]
    probs = np.stack([pp, pm, np.clip(1 - pp - pm, 0, 1)], 1)
    cases["gfc_minus_worldcover"] = {"kind": "diff", "n": g["n"], "W": g["W"], "probs": probs / probs.sum(1, keepdims=True),
                                     "Nbox": None, "theta": float(np.sum(g["W"] * (pp - pm)))}
    procs = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    out = {"draws": R, "processes": procs}
    with mp.Pool(procs) as pool:
        for name, c in cases.items():
            t = time.time()
            rows = np.array(pool.map(_mc_one, [(c, s) for s in range(R)], chunksize=max(1, R // (4 * procs))), float)
            out[name] = {"theta": c["theta"], "bound_coverage": float(rows[:, 0].mean()),
                         "bound_mean_width_points": float(100 * rows[:, 1].mean()),
                         "normal_coverage": float(rows[:, 2].mean()),
                         "normal_mean_width_points": float(100 * rows[:, 3].mean()), "seconds": time.time() - t}
    return out


def _json(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    rec, areas, cmd = sys.argv[1], sys.argv[2], sys.argv[3]
    t0 = time.time()
    if cmd == "real":
        res = real(rec, areas, sys.argv[4] if len(sys.argv) > 4 else None)
    elif cmd == "enum":
        res = enum()
    elif cmd == "mc":
        res = mc(rec, areas, int(sys.argv[4]) if len(sys.argv) > 4 else 200)
    else:
        sys.exit(f"unknown command {cmd!r}: real, enum or mc")
    res["seconds_total"] = time.time() - t0
    json.dump(res, sys.stdout, indent=1, default=_json)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
