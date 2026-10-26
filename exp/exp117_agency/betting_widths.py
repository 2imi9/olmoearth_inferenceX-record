#!/usr/bin/env python
"""Stratified betting intervals for an agency's reference sample: a design study, run on aicr, never on the laptop.

    python -I betting_widths.py REC GFC_AREAS [SIZING_JSON] [--quick] [--selftest-only] [--fast] > betting_widths.json

REC        the research record's root; exp/out/exp94_nlcd_points.csv and exp/out/exp94_gfc_points.csv are read
GFC_AREAS  the package's data/breadth/gfc2020/strata_size.csv (Strata, strata_ha)
SIZING     optional: agency/sizing.json, whose union-bound widths are copied beside ours
--quick    self-tests, overall accuracies, the two-map differences and the forest share only (about 5 minutes);
           without it the per-class tables run too (about 1 to 2 hours on one core)
--fast     a mixture of 8 values of c instead of 12 (for the per-class scans)
numpy only (oe_inferencex is not imported). Progress goes to stderr, the JSON to stdout. One core, under 1 GB.

THE INTERVAL. Stratum h has weight W_h (its share of the map), n_h labels drawn at random without replacement from N_h
units (N_h = inf for a point frame), values in [0, 1]. For its mean mu_h it gets the e-value

    K_h(mu_h) = prod_t (1 + l_t (X_t - c_t)) * exp(l_t (c_t - m_t(mu_h))),   t = 0 .. n_h - 1

with m_t(mu) = (N mu - S_t) / (N - t) the mean of the units not yet drawn under the null (S_t the sum of the t values
drawn before), c_t a predictable centre (the running mean (1/2 + S_t) / (t + 1), clipped at KAPPA / l_t for a linear
quantity), and l_t = c W_h (N_h - t) / (N_h n_h). Each factor has conditional mean (1 + l (m - c)) exp(-l (m - c)) <= 1
(1 + y <= e^y: the second line of the proof of Theorem 3.2 of Waudby-Smith and Ramdas 2020, NeurIPS, before Fan et
al.'s inequality), and is nonnegative, so E K_h(true mu_h) <= 1. log K_h is linear in mu_h with slope
-sum_t l_t N/(N - t) = -c W_h, so the product over the (independent) strata depends on the mu_h only through
theta = sum_h W_h mu_h: the union-intersection minimum over every way of splitting theta among the strata is the same
at every split, and costs nothing. Averaged over every order of each stratum's labels (exactly, by dynamic programming
over the counts, so no random order is involved), mixed over a grid of c fixed before any label, it is an e-value for
theta. Each end of the interval is where it reaches 2/alpha (Markov). Ratios (user's and producer's accuracy) invert
theta_y - r theta_z = 0 over r; for each r the e-value is log-concave in r term by term, which certifies the scan
between the points evaluated.
"""
import csv
import itertools
import json
import math
import sys
import time
from collections import defaultdict

import numpy as np

Z = 1.959963984540054
NEG = -np.inf
KAPPA = 0.9          # a linear quantity's centre is clipped at KAPPA / lambda, so every factor stays at least 1 - KAPPA
K_MIX = 12           # values of c in the mixture
SPAN = 10.0          # c runs from c_lo (every stratum at variance 1/4) to SPAN * c_lo, geometrically
NLCD_TO_LCMAP = {21: 1, 22: 1, 23: 1, 24: 1, 81: 2, 82: 2, 52: 3, 71: 3, 41: 4, 42: 4, 43: 4, 11: 5, 90: 6, 95: 6, 12: 7,
                 31: 8}


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def logsumexp(x):
    x = np.asarray(x, float).ravel()
    m = float(x.max())
    if m == NEG:
        return NEG
    return m + math.log(float(np.exp(x - m).sum()))


def log_choose(n, k):
    if k < 0 or k > n:
        return NEG
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def compositions(total):
    """Every (i, j, k) of whole numbers adding up to total."""
    return [(i, j, total - i - j) for i in range(total + 1) for j in range(total + 1 - i)]


# ----------------------------------------------------------------------------------------------- one stratum, exactly
def stratum_logavg(counts, a, b, lam0, N, n, r=0.0, clip=True, bound=None):
    """log of the average, over every order of one stratum's n labels, of
        prod_t (1 + l_t (v_t - c_t)) exp(l_t (c_t + S_t / (N - t)))
    (the stratum's e-value without its -c W mu term), for each first-step lambda in the array lam0 (shape K):
    l_t = lam0 (N - t) / N (lam0 when N is inf). counts: the labels per category (at most three); category j's value is
    a[j] + b[j] r. With bound = (r_der, delta, direction): a lower bound valid for every r' between r and
    r + direction * delta, each factor's log taken at r and lowered by delta times its slope at r_der where that slope
    works against the direction; each log-factor is concave in r (clip must be False), so its slope over the interval
    is never more adverse than at r_der."""
    counts = [int(x) for x in counts] + [0] * (3 - len(counts))
    a = [float(x) for x in a] + [0.0] * (3 - len(a))
    b = [float(x) for x in b] + [0.0] * (3 - len(b))
    if sum(counts) != n:
        raise ValueError(f"counts {counts} do not add up to n = {n}")
    if bound is not None and clip:
        raise ValueError("the certified bound needs the unclipped centre")
    order = sorted(range(3), key=lambda j: -counts[j])            # the largest category is the implied one
    k0, k1, k2 = (counts[j] for j in order)
    av, bv = [a[j] for j in order], [b[j] for j in order]
    lam0 = np.asarray(lam0, float).reshape(-1, 1, 1)
    K = lam0.shape[0]
    finite = math.isfinite(N)
    v = [av[j] + bv[j] * r for j in range(3)]
    if bound is not None:
        r_der, delta, direction = bound
        vd = [av[j] + bv[j] * r_der for j in range(3)]
    if n == 0:
        return np.zeros(K)
    if k1 == 0 and k2 == 0:                                       # one category: a single order, summed at once
        t = np.arange(n, dtype=float).reshape(1, -1)
        lt = lam0.reshape(-1, 1) * ((N - t) / N if finite else 1.0)
        S = t * v[0]
        cen = (0.5 + S) / (t + 1)
        if clip:
            cen = np.minimum(cen, KAPPA / lt)
        tail = S / (N - t) if finite else 0.0
        lf = np.log1p(lt * (v[0] - cen)) + lt * (cen + tail)
        if bound is not None:
            cend = (0.5 + t * vd[0]) / (t + 1)
            dS = t * bv[0]
            dcen = dS / (t + 1)
            dtail = dS / (N - t) if finite else 0.0
            slope = lt * (bv[0] - dcen) / (1.0 + lt * (vd[0] - cend)) + lt * (dcen + dtail)
            lf = lf - delta * np.maximum(0.0, -direction * slope)
        return lf.sum(axis=1)
    s1 = np.arange(k1 + 1, dtype=float).reshape(1, -1, 1)
    s2 = np.arange(k2 + 1, dtype=float).reshape(1, 1, -1)
    L = np.full((K, k1 + 1, k2 + 1), NEG)
    L[:, 0, 0] = 0.0
    for t in range(n):
        s0 = t - s1 - s2
        valid = (s0 >= 0) & (s0 <= k0)
        lt = lam0 * ((N - t) / N) if finite else lam0
        S = s0 * v[0] + s1 * v[1] + s2 * v[2]
        cen = (0.5 + S) / (t + 1)
        if clip:
            cen = np.minimum(cen, KAPPA / lt)
        tail = S / (N - t) if finite else 0.0
        if bound is not None:
            cend = (0.5 + s0 * vd[0] + s1 * vd[1] + s2 * vd[2]) / (t + 1)
            dS = s0 * bv[0] + s1 * bv[1] + s2 * bv[2]
            dcen = dS / (t + 1)
            dtail = dS / (N - t) if finite else 0.0
        new = np.full_like(L, NEG)
        for j, kj in enumerate((k0, k1, k2)):
            if kj == 0:
                continue
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                lf = np.log1p(lt * (v[j] - cen)) + lt * (cen + tail)
                if bound is not None:
                    slope = lt * (bv[j] - dcen) / (1.0 + lt * (vd[j] - cend)) + lt * (dcen + dtail)
                    lf = lf - delta * np.maximum(0.0, -direction * slope)
            src = L + np.where(valid, lf, NEG)
            if j == 0:
                new = np.logaddexp(new, np.where(s0 + 1 <= k0, src, NEG))
            elif j == 1:
                new[:, 1:, :] = np.logaddexp(new[:, 1:, :], src[:, :-1, :])
            else:
                new[:, :, 1:] = np.logaddexp(new[:, :, 1:], src[:, :, :-1])
        L = new
    return L[:, k1, k2] - (math.lgamma(n + 1) - math.lgamma(k0 + 1) - math.lgamma(k1 + 1) - math.lgamma(k2 + 1))


# ----------------------------------------------------------------------------------------------- the mixture over c
def c_grid(strata, alpha, cap, K=K_MIX, span=SPAN):
    """The mixture's values of c, from the design alone (weights, labels per stratum, alpha), never the labels."""
    vmax = sum(s["W"] ** 2 / (4.0 * s["n"]) for s in strata)
    clo = math.sqrt(2.0 * math.log(2.0 / alpha) / vmax)
    chi = span * clo
    if cap:                                   # unclipped centres: lambda at most 1 keeps every factor nonnegative
        cmax = min(s["n"] / s["W"] for s in strata)
        clo, chi = min(clo, cmax), min(chi, cmax)
    if K == 1 or chi <= clo * (1 + 1e-12):
        return np.array([clo])
    return np.geomspace(clo, chi, K)


def deflation(strata, cs):
    """A stratum read as a point frame (N = inf) but drawn without replacement from at least Nlo units: the
    with-replacement e-value is at most exp(c W (n - 1) / (2 (Nlo - n + 1))) times the without-replacement one."""
    d = np.zeros(cs.size)
    for s in strata:
        if s.get("Nlo") is not None and not math.isfinite(s["N"]):
            d += cs * s["W"] * (s["n"] - 1) / (2.0 * (s["Nlo"] - s["n"] + 1))
    return d


# ----------------------------------------------------------------------------------------------- linear quantities
def lin_A(strata, cs, side):
    A = np.zeros(cs.size)
    for s in strata:
        vals = s["vals"] if side == "low" else [1.0 - x for x in s["vals"]]
        A += stratum_logavg(s["counts"], vals, [0.0] * len(vals), cs * s["W"] / s["n"], s["N"], s["n"], clip=True)
    return A - deflation(strata, cs)


def root_low(A, cs, thr, top):
    """The lower end: the largest theta0 at which the mixed e-value, decreasing in theta0, still reaches e^thr."""
    logpi = -math.log(cs.size)
    f = lambda th: logsumexp(logpi + A - cs * th) - thr
    if f(0.0) < 0:
        return 0.0
    if f(top) >= 0:
        return top
    lo, hi = 0.0, top
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) >= 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    return lo


def linear_interval(strata, alpha, K=K_MIX, span=SPAN):
    cs = c_grid(strata, alpha, cap=False, K=K, span=span)
    thr = math.log(2.0 / alpha)
    top = sum(s["W"] for s in strata)
    lo = root_low(lin_A(strata, cs, "low"), cs, thr, top)
    hi = top - root_low(lin_A(strata, cs, "high"), cs, thr, top)
    return lo, hi, cs


def oracle_interval(strata, alpha, cs):
    """NOT valid: the best single c chosen after seeing the labels. It shows what the mixture costs."""
    fine = np.geomspace(cs[0] / 3, cs[-1] * 3, 120)
    thr = math.log(2.0 / alpha)
    top = sum(s["W"] for s in strata)
    lo = float(np.clip(np.max((lin_A(strata, fine, "low") - thr) / fine), 0.0, top))
    hi = top - float(np.clip(np.max((lin_A(strata, fine, "high") - thr) / fine), 0.0, top))
    return lo, hi


def lin_strata(groups):
    out = []
    for g in groups:
        vals, counts = np.unique(g["x"], return_counts=True)
        if vals.size > 3 or vals.min() < 0 or vals.max() > 1:
            raise ValueError(f"stratum {g.get('h')}: values {vals}")
        out.append({"W": g["W"], "n": g["n"], "N": g["N"], "Nlo": g.get("Nlo"), "vals": [float(x) for x in vals],
                    "counts": [int(c) for c in counts]})
    return out


def normal_linear(groups):
    est = sum(g["W"] * float(g["x"].mean()) for g in groups)
    var = sum(g["W"] ** 2 * (1 - (g["n"] / g["N"] if math.isfinite(g["N"]) else 0.0)) * float(g["x"].var(ddof=1)) / g["n"]
              for g in groups if g["n"] > 1)
    return est, var


def run_linear(name, groups, alpha=0.05, scale=1.0, shift=0.0, extra=True):
    """theta = scale * (sum_h W_h mean_h(x)) + shift, x in [0, 1]."""
    t0 = time.time()
    st = lin_strata(groups)
    est, var = normal_linear(groups)
    h = Z * math.sqrt(var)
    lo, hi, cs = linear_interval(st, alpha)
    T = lambda x: scale * x + shift
    res = {"estimate": T(est), "normal": [T(est - h), T(est + h)], "normal_width": scale * 2 * h,
           "betting": [T(lo), T(hi)], "betting_width": scale * (hi - lo),
           "gaussian_chernoff_width": scale * 2 * math.sqrt(2 * var * math.log(2 / alpha)),
           "c_grid": [float(cs[0]), float(cs[-1]), int(cs.size)],
           "largest_first_lambda": float(max(cs[-1] * s["W"] / s["n"] for s in st))}
    res["betting_over_normal"] = res["betting_width"] / res["normal_width"] if res["normal_width"] > 0 else None
    if extra:
        olo, ohi = oracle_interval(st, alpha, cs)
        res["oracle_single_c_NOT_VALID"] = [T(olo), T(ohi)]
        res["oracle_width"] = scale * (ohi - olo)
        sens = {}
        for K, span in ((6, 10.0), (24, 10.0), (12, 30.0), (12, 3.0)):
            a, b, _ = linear_interval(st, alpha, K=K, span=span)
            sens[f"K{K}_span{span:g}"] = scale * (b - a)
        res["width_by_mixture"] = sens
    res["seconds"] = time.time() - t0
    say(f"{name}: estimate {100 * res['estimate']:.2f}, normal width {100 * res['normal_width']:.2f}, betting width "
        f"{100 * res['betting_width']:.2f} points ({res['seconds']:.0f} s)")
    return res


# ----------------------------------------------------------------------------------------------- ratios
RATIO_AB = {"low": ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),       # categories A (y=1,z=1), B (y=0,z=1), C (z=0): u = y + r(1-z)
            "high": ([0.0, 1.0, 1.0], [0.0, 0.0, -1.0])}     # u' = 1 - u


def ratio_logE(strata, cs, side, r, bound=None, r_null=None):
    a, b = RATIO_AB[side]
    A = np.zeros(cs.size)
    for s in strata:
        A += stratum_logavg(s["counts"], a, b, cs * s["W"] / s["n"], s["N"], s["n"], r=r, clip=False, bound=bound)
    A -= deflation(strata, cs)
    SW = sum(s["W"] for s in strata)
    rn = r if r_null is None else r_null
    nv = rn * SW if side == "low" else (1.0 - rn) * SW
    return logsumexp(-math.log(cs.size) + A - cs * nv)


def ratio_interval(strata, alpha, K=K_MIX, span=SPAN, dmin=2e-4):
    """R = theta_y / theta_z, y <= z in {0, 1}. Each end is certified: every r beyond it is rejected, checked on
    intervals by the per-factor bound of stratum_logavg."""
    if sum(s["counts"][0] + s["counts"][1] for s in strata) == 0:
        return 0.0, 1.0, None
    cs = c_grid(strata, alpha, cap=True, K=K, span=span)
    thr = math.log(2.0 / alpha)
    rej = lambda side, r, bound=None, r_null=None: ratio_logE(strata, cs, side, r, bound, r_null) >= thr
    if not rej("low", 0.0):
        lo_end = 0.0
    else:
        if rej("low", 1.0):
            target = 1.0
        else:
            a, b = 0.0, 1.0
            for _ in range(14):
                m = 0.5 * (a + b)
                a, b = (m, b) if rej("low", m) else (a, m)
            target = a
        r, step = 0.0, max(target / 2, dmin)
        while r < target - 1e-12:
            d = min(step, target - r)
            if rej("low", r, bound=(r + d, d, 1), r_null=r + d):
                r, step = r + d, 2 * d
            elif d <= dmin:
                break
            else:
                step = d / 2
        lo_end = r
    if not rej("high", 1.0):
        hi_end = 1.0
    else:
        if rej("high", 0.0):
            target = 0.0
        else:
            a, b = 0.0, 1.0
            for _ in range(14):
                m = 0.5 * (a + b)
                a, b = (a, m) if rej("high", m) else (m, b)
            target = b
        r, step = 1.0, max((1.0 - target) / 2, dmin)
        while r > target + 1e-12:
            d = min(step, r - target)
            if rej("high", r, bound=(r - d, d, -1), r_null=r - d):
                r, step = r - d, 2 * d
            elif d <= dmin:
                break
            else:
                step = d / 2
        hi_end = r
    return lo_end, hi_end, cs


def ratio_strata(groups):
    out = []
    for g in groups:
        y, z = g["y"], g["z"]
        if (y > z).any():
            raise ValueError("a ratio needs y <= z")
        out.append({"W": g["W"], "n": g["n"], "N": g["N"], "Nlo": g.get("Nlo"),
                    "counts": [int(((y == 1) & (z == 1)).sum()), int(((y == 0) & (z == 1)).sum()), int((z == 0).sum())]})
    return out


def normal_ratio(groups):
    ty = sum(g["W"] * float(g["y"].mean()) for g in groups)
    tz = sum(g["W"] * float(g["z"].mean()) for g in groups)
    if tz <= 0:
        return None, None
    R = ty / tz
    var = sum(g["W"] ** 2 * (1 - (g["n"] / g["N"] if math.isfinite(g["N"]) else 0.0))
              * float((g["y"] - R * g["z"]).var(ddof=1)) / g["n"] for g in groups if g["n"] > 1) / tz ** 2
    return R, var


def run_ratio(name, groups, alpha=0.05, K=K_MIX):
    t0 = time.time()
    R, var = normal_ratio(groups)
    lo, hi, cs = ratio_interval(ratio_strata(groups), alpha, K=K)
    res = {"estimate": R, "betting": [lo, hi], "betting_width": hi - lo,
           "n_y": int(sum(g["y"].sum() for g in groups)), "n_z": int(sum(g["z"].sum() for g in groups)),
           "c_grid": None if cs is None else [float(cs[0]), float(cs[-1]), int(cs.size)]}
    if R is not None:
        h = Z * math.sqrt(var)
        res.update({"normal": [R - h, R + h], "normal_width": 2 * h,
                    "betting_over_normal": (hi - lo) / (2 * h) if h > 0 else None})
    res["seconds"] = time.time() - t0
    say(f"{name}: estimate {R}, betting [{lo:.4f}, {hi:.4f}] ({res['seconds']:.0f} s)")
    return res


# ----------------------------------------------------------------------------------------------- self-tests (exact)
def brute_logavg(counts, a, b, lam, N, r=0.0, clip=True, bound=None):
    """stratum_logavg for one lambda by listing every distinct order of the labels (each stands for the same number
    of orders of the units), for checking the dynamic programme."""
    cats = [j for j, k in enumerate(counts) for _ in range(k)]
    finite = math.isfinite(N)
    vals = [a[j] + b[j] * r for j in range(len(a))]
    if bound is not None:
        r_der, delta, direction = bound
        vd = [a[j] + b[j] * r_der for j in range(len(a))]
    tot = []
    for seq in set(itertools.permutations(cats)):
        S = Sd = dS = 0.0
        lg = 0.0
        for t, j in enumerate(seq):
            lt = lam * ((N - t) / N if finite else 1.0)
            cen = (0.5 + S) / (t + 1)
            if clip:
                cen = min(cen, KAPPA / lt)
            tail = S / (N - t) if finite else 0.0
            lf = math.log1p(lt * (vals[j] - cen)) + lt * (cen + tail)
            if bound is not None:
                cend, dcen = (0.5 + Sd) / (t + 1), dS / (t + 1)
                dtail = dS / (N - t) if finite else 0.0
                slope = lt * (b[j] - dcen) / (1.0 + lt * (vd[j] - cend)) + lt * (dcen + dtail)
                lf -= delta * max(0.0, -direction * slope)
                Sd += vd[j]
                dS += b[j]
            lg += lf
            S += vals[j]
        tot.append(lg)
    return logsumexp(tot) - math.log(len(tot))


def selftest():
    out = {}
    t0 = time.time()
    # 0. the dynamic programme against every order listed, plain and in the certified-bound form
    worst = 0.0
    for n in range(1, 7):
        for smp in compositions(n):
            for lam in (0.3, 1.0, 3.0):
                for N in (n, n + 2, math.inf):
                    dp = float(stratum_logavg(list(smp), [0.0, 0.5, 1.0], [0.0] * 3, np.array([lam]), N, n)[0])
                    worst = max(worst, abs(dp - brute_logavg(list(smp), [0.0, 0.5, 1.0], [0.0] * 3, lam, N)))
                    if lam <= 1.0:
                        for side in ("low", "high"):
                            aa, bb = RATIO_AB[side]
                            bd = (0.55, 0.15, 1) if side == "low" else (0.25, 0.15, -1)
                            dp = float(stratum_logavg(list(smp), aa, bb, np.array([lam]), N, n, r=0.4, clip=False,
                                                      bound=bd)[0])
                            worst = max(worst, abs(dp - brute_logavg(list(smp), aa, bb, lam, N, r=0.4, clip=False,
                                                                     bound=bd)))
    out["dynamic_programme_vs_every_order_max_abs_difference"] = worst
    say(f"selftest 0: DP against every order, largest difference {worst:.2e}")
    # 1. E[e-value at the true mean] over every sample and every order, populations of values 0, 1/2, 1
    lams = (0.2, 0.6, 1.0, 2.0, 5.0)
    keys = ("without_replacement", "with_replacement_deflated", "with_replacement_plain_NEGATIVE_CONTROL")
    worst = {k: 0.0 for k in keys}
    for N in (2, 3, 4, 6):
        for n in range(1, N + 1):
            samples = compositions(n)
            cache = {}
            for lam in lams:
                for smp in samples:
                    for form, Ns in (("wor", N), ("wr", math.inf)):
                        cache[(lam, smp, form)] = float(stratum_logavg(list(smp), [0.0, 0.5, 1.0], [0.0] * 3,
                                                                       np.array([lam]), Ns, n)[0])
            for M in compositions(N):
                mu = (0.5 * M[1] + M[2]) / N
                for lam in lams:
                    D = lam * n * (n - 1) / (2.0 * (N - n + 1))
                    e = dict.fromkeys(keys, 0.0)
                    for smp in samples:
                        lp = sum(log_choose(M[j], smp[j]) for j in range(3)) - log_choose(N, n)
                        if lp == NEG:
                            continue
                        base = lp - lam * n * mu
                        e[keys[0]] += math.exp(base + cache[(lam, smp, "wor")])
                        e[keys[1]] += math.exp(base + cache[(lam, smp, "wr")] - D)
                        e[keys[2]] += math.exp(base + cache[(lam, smp, "wr")])
                    for k in keys:
                        worst[k] = max(worst[k], e[k])
    out["largest_expected_evalue"] = worst
    out["evalue_property_holds"] = worst[keys[0]] <= 1 + 1e-9 and worst[keys[1]] <= 1 + 1e-9
    say(f"selftest 1: {worst}")

    # 2. coverage of the linear interval, every population and sample of two or three small strata
    def lin_cov(Ns, ns, W, alpha):
        H = len(Ns)
        ivs = {}
        for ks in itertools.product(*[range(n + 1) for n in ns]):
            st = [{"W": W[h], "n": ns[h], "N": Ns[h], "Nlo": None, "vals": [0.0, 1.0], "counts": [ns[h] - ks[h], ks[h]]}
                  for h in range(H)]
            lo, hi, cs = linear_interval(st, alpha)
            olo, ohi = oracle_interval(st, alpha, cs)
            est = sum(W[h] * ks[h] / ns[h] for h in range(H))
            var = sum(W[h] ** 2 * (1 - ns[h] / Ns[h]) * ks[h] * (ns[h] - ks[h]) / (ns[h] * (ns[h] - 1)) / ns[h]
                      for h in range(H) if ns[h] > 1)
            ivs[ks] = ((lo, hi), (est - Z * math.sqrt(var), est + Z * math.sqrt(var)), (olo, ohi))
        worst = [1.0, 1.0, 1.0]
        for Ms in itertools.product(*[range(N + 1) for N in Ns]):
            theta = sum(W[h] * Ms[h] / Ns[h] for h in range(H))
            cov = [0.0, 0.0, 0.0]
            for ks, iv in ivs.items():
                lp = sum(log_choose(Ms[h], ks[h]) + log_choose(Ns[h] - Ms[h], ns[h] - ks[h]) - log_choose(Ns[h], ns[h])
                         for h in range(H))
                if lp == NEG:
                    continue
                for j in range(3):
                    if iv[j][0] - 1e-12 <= theta <= iv[j][1] + 1e-12:
                        cov[j] += math.exp(lp)
            worst = [min(x, y) for x, y in zip(worst, cov)]
        return {"betting": worst[0], "normal_theory_nominal": worst[1], "oracle_c_NOT_VALID": worst[2]}

    cells = {}
    for Ns, ns, W in (((6, 4), (3, 2), (0.5, 0.5)), ((6, 4), (3, 2), (0.85, 0.15)), ((5, 5), (4, 1), (0.3, 0.7)),
                      ((5, 4, 3), (2, 2, 1), (0.6, 0.3, 0.1))):
        for alpha in (0.1, 0.3):
            cells[f"N{Ns}_n{ns}_W{W}_alpha{alpha}"] = c = lin_cov(Ns, ns, W, alpha)
            c["holds"] = c["betting"] >= 1 - alpha - 1e-12
    out["linear_min_coverage"] = cells
    say("selftest 2 done")

    # 3. coverage of the ratio interval, and its certified ends against a dense grid
    def ratio_cov(Ns, ns, W, alpha):
        H = len(Ns)
        ivs, viol = {}, 0
        grid = np.linspace(0.0, 1.0, 201)
        for ss in itertools.product(*[compositions(n) for n in ns]):
            st = [{"W": W[h], "n": ns[h], "N": Ns[h], "Nlo": None, "counts": list(ss[h])} for h in range(H)]
            lo, hi, cs = ratio_interval(st, alpha)
            ivs[ss] = (lo, hi)
            if cs is None:
                continue
            thr = math.log(2.0 / alpha)
            first_ok = next((r for r in grid if ratio_logE(st, cs, "low", r) < thr), 1.0)
            last_ok = next((r for r in grid[::-1] if ratio_logE(st, cs, "high", r) < thr), 0.0)
            viol += int(lo > first_ok + 1e-12) + int(hi < last_ok - 1e-12)
        worst = 1.0
        for pops in itertools.product(*[compositions(N) for N in Ns]):
            ty = sum(W[h] * pops[h][0] / Ns[h] for h in range(H))
            tz = sum(W[h] * (pops[h][0] + pops[h][1]) / Ns[h] for h in range(H))
            if tz <= 0:
                continue
            R = ty / tz
            cov = 0.0
            for ss, (lo, hi) in ivs.items():
                lp = sum(sum(log_choose(pops[h][j], ss[h][j]) for j in range(3)) - log_choose(Ns[h], ns[h]) for h in range(H))
                if lp == NEG:
                    continue
                if lo - 1e-12 <= R <= hi + 1e-12:
                    cov += math.exp(lp)
            worst = min(worst, cov)
        return {"betting": worst, "certified_end_beyond_dense_grid": viol, "holds": worst >= 1 - alpha - 1e-12 and viol == 0}

    rc = {}
    for Ns, ns, W in (((4, 4), (2, 2), (0.7, 0.3)), ((5, 3), (3, 2), (0.5, 0.5))):
        for alpha in (0.1, 0.3):
            rc[f"N{Ns}_n{ns}_W{W}_alpha{alpha}"] = ratio_cov(Ns, ns, W, alpha)
    out["ratio_min_coverage"] = rc
    out["seconds"] = time.time() - t0
    say(f"selftest 3 done ({out['seconds']:.0f} s)")
    return out


# ----------------------------------------------------------------------------------------------- data
def strata_groups(rows, info, fx, ratio=False):
    xs = defaultdict(list)
    for r in rows:
        xs[int(r["stratum"])].append(fx(r))
    out = []
    for h in sorted(xs):
        i = info[h]
        g = {"h": h, "W": i["W"], "n": i["n"], "N": i["N"], "Nlo": i.get("Nlo")}
        if len(xs[h]) != i["n"]:
            raise ValueError(f"stratum {h}: {len(xs[h])} rows for n = {i['n']}")
        if ratio:
            arr = np.asarray(xs[h], float)
            g["y"], g["z"] = arr[:, 0], arr[:, 1]
        else:
            g["x"] = np.asarray(xs[h], float)
        out.append(g)
    return out


def gfc_block(rec, areas_path, alpha, quick, K):
    area = {int(r["Strata"]): float(r["strata_ha"]) for r in csv.DictReader(open(areas_path))}
    rows = list(csv.DictReader(open(f"{rec}/exp/out/exp94_gfc_points.csv")))
    n_by = defaultdict(int)
    for r in rows:
        n_by[int(r["stratum"])] += 1
    hs = sorted(h for h in n_by if h in area)
    tot = sum(area[h] for h in hs)
    info = {h: {"n": n_by[h], "N": math.inf, "W": area[h] / tot} for h in hs}
    rows = [r for r in rows if int(r["stratum"]) in info]
    out = {"strata": len(hs), "points": len(rows),
           "strata_without_area": sorted(h for h in n_by if h not in area),
           "strata_without_points": sorted(h for h in area if h not in n_by),
           "worldcover_missing": sum(1 for r in rows if r["worldcover_forest"] == ""),
           "n_min": min(i["n"] for i in info.values()), "n_max": max(i["n"] for i in info.values()),
           "W_max": max(i["W"] for i in info.values()),
           "c_max_for_lambda_1": min(i["n"] / i["W"] for i in info.values()),
           "population": "point frame (N = inf); the 10 m pixel frame (N >= area / 0.01 ha) adds a deflation"}
    ref = lambda r: int(r["ref_forest"])
    gf = lambda r: int(r["gfc_v2"])
    wc = lambda r: int(r["worldcover_forest"]) if r["worldcover_forest"] != "" else -1
    out["oa_gfc_v2"] = run_linear("GFC2020 v2 overall accuracy", strata_groups(rows, info, lambda r: float(gf(r) == ref(r))), alpha)
    out["oa_worldcover"] = run_linear("WorldCover overall accuracy", strata_groups(rows, info, lambda r: float(wc(r) == ref(r))), alpha)
    out["gfc_minus_worldcover"] = run_linear(
        "GFC2020 minus WorldCover", strata_groups(rows, info, lambda r: (float(gf(r) == ref(r)) - float(wc(r) == ref(r)) + 1) / 2),
        alpha, scale=2.0, shift=-1.0)
    info10 = {h: dict(i, Nlo=area[h] / 0.01) for h, i in info.items()}
    out["oa_gfc_v2_10m_pixel_frame"] = run_linear("GFC2020 v2 overall, 10 m pixel frame",
                                                  strata_groups(rows, info10, lambda r: float(gf(r) == ref(r))), alpha,
                                                  extra=False)
    out["forest_share"] = run_linear("forest share (reference)", strata_groups(rows, info, lambda r: float(ref(r) == 1)), alpha)
    if quick:
        return out
    for mname, mf in (("gfc_v2", gf), ("worldcover", wc)):
        for cls, cname in ((1, "forest"), (0, "nonforest")):
            yz = lambda r, mf=mf, cls=cls: (float(mf(r) == cls and ref(r) == cls), float(mf(r) == cls))
            out[f"user_{cname}_{mname}"] = run_ratio(f"user's accuracy {cname} {mname}", strata_groups(rows, info, yz, True), alpha, K)
            yz = lambda r, mf=mf, cls=cls: (float(mf(r) == cls and ref(r) == cls), float(ref(r) == cls))
            out[f"producer_{cname}_{mname}"] = run_ratio(f"producer's accuracy {cname} {mname}",
                                                         strata_groups(rows, info, yz, True), alpha, K)
    return out


def nlcd_block(rec, alpha, quick, K):
    rows = [r for r in csv.DictReader(open(f"{rec}/exp/out/exp94_nlcd_points.csv")) if r["set"] == "B" and r["year"] == "2016"]
    by = defaultdict(list)
    for r in rows:
        by[int(r["stratum"])].append(float(r["weight"]))
    info, table = {}, {}
    for h, w in sorted(by.items()):
        w = np.asarray(w)
        Nf = float(w.sum())
        info[h] = {"n": int(w.size), "N": int(round(Nf))}
        table[h] = {"n": int(w.size), "N": int(round(Nf)), "weight_relative_spread": float(w.max() / w.min() - 1),
                    "N_distance_to_integer": abs(Nf - round(Nf))}
    Ntot = sum(i["N"] for i in info.values())
    for h, i in info.items():
        i["W"] = table[h]["W"] = i["N"] / Ntot
    out = {"strata": table, "points": len(rows), "population": "30 m pixels drawn without replacement, N_h = n_h x weight"}
    m = lambda r: int(r["nlcd"])
    p = lambda r: int(r["ref_nlcd"])
    alt = lambda r: int(r["ref_nlcd_alternate"])
    eff = lambda r: m(r) if m(r) == alt(r) else p(r)          # primary or alternate: the map's class where it matches either
    out["oa_level2_primary"] = run_linear("NLCD 2016 level II, primary", strata_groups(rows, info, lambda r: float(m(r) == p(r))), alpha)
    out["oa_level2_primary_or_alternate"] = run_linear("NLCD 2016 level II, primary or alternate",
                                                       strata_groups(rows, info, lambda r: float(m(r) == eff(r))), alpha)
    out["oa_level1_primary"] = run_linear("NLCD 2016 level I, primary",
                                          strata_groups(rows, info, lambda r: float(m(r) // 10 == p(r) // 10)), alpha)
    out["oa_level1_primary_or_alternate"] = run_linear(
        "NLCD 2016 level I, primary or alternate (level II labels // 10)",
        strata_groups(rows, info, lambda r: float(m(r) // 10 in (p(r) // 10, alt(r) // 10))), alpha)
    info_inf = {h: dict(i, N=math.inf) for h, i in info.items()}
    out["oa_level2_primary_point_frame"] = run_linear("NLCD level II primary, point frame (no finite-population term)",
                                                      strata_groups(rows, info_inf, lambda r: float(m(r) == p(r))), alpha,
                                                      extra=False)
    lc = lambda r: int(r["lcmap"])
    r8 = lambda r: NLCD_TO_LCMAP.get(p(r), 0)
    out["nlcd_minus_lcmap_8_classes"] = run_linear(
        "NLCD minus LCMAP, 8 classes, primary",
        strata_groups(rows, info, lambda r: (float(NLCD_TO_LCMAP.get(m(r), -1) == r8(r)) - float(lc(r) == r8(r)) + 1) / 2),
        alpha, scale=2.0, shift=-1.0)
    if quick:
        return out
    classes = sorted({m(r) for r in rows} | {eff(r) for r in rows})
    L = len(classes)
    out["legend_size"] = L
    per = {}
    for c in classes:
        row = {}
        row["area_share"] = run_linear(f"class {c} share", strata_groups(rows, info, lambda r, c=c: float(eff(r) == c)),
                                       alpha, extra=False)
        row["area_share_joint"] = run_linear(f"class {c} share, joint", strata_groups(rows, info, lambda r, c=c: float(eff(r) == c)),
                                             alpha / (2 * L), extra=False)
        ua = lambda r, c=c: (float(m(r) == c and eff(r) == c), float(m(r) == c))
        pa = lambda r, c=c: (float(m(r) == c and eff(r) == c), float(eff(r) == c))
        ua_p = lambda r, c=c: (float(m(r) == c and p(r) == c), float(m(r) == c))
        row["user"] = run_ratio(f"class {c} user's", strata_groups(rows, info, ua, True), alpha, K)
        row["producer"] = run_ratio(f"class {c} producer's", strata_groups(rows, info, pa, True), alpha, K)
        row["producer_joint"] = run_ratio(f"class {c} producer's, joint", strata_groups(rows, info, pa, True), alpha / (2 * L), K)
        row["user_primary_only"] = run_ratio(f"class {c} user's, primary", strata_groups(rows, info, ua_p, True), alpha, K)
        per[str(c)] = row
    out["per_class_primary_or_alternate"] = per
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) < 2:
        raise SystemExit(__doc__)
    rec, areas = args[0], args[1]
    K = 8 if "--fast" in flags else K_MIX
    out = {"alpha": 0.05, "mixture_size": K_MIX, "mixture_size_ratios": K, "span": SPAN, "kappa": KAPPA,
           "numpy": np.__version__, "python": sys.version.split()[0]}
    t0 = time.time()
    out["selftest"] = selftest()
    if "--selftest-only" not in flags:
        quick = "--quick" in flags
        out["gfc2020"] = gfc_block(rec, areas, 0.05, quick, K)
        out["nlcd2016_setB_2016"] = nlcd_block(rec, 0.05, quick, K)
        if len(args) > 2:
            s = json.load(open(args[2]))
            out["union_bound_from_sizing"] = {
                "gfc_v2_overall_width": s["gfc2020_gfc_v2_overall"]["union_by_weight_width"],
                "worldcover_overall_width": s["gfc2020_worldcover_overall"]["union_by_weight_width"],
                "gfc_minus_worldcover_width": s["gfc2020_gfc_minus_worldcover"]["union_width"],
                "nlcd_level2_primary_width": s["nlcd2016_level2_overall"]["union_by_weight_width"],
                "nlcd_level1_primary_width": s["nlcd2016_level1_overall"]["union_by_weight_width"],
                "nlcd_user_primary_widths": {c: v["union_width"] for c, v in s["nlcd2016_user_accuracy_level2"].items()}}
    out["seconds"] = time.time() - t0
    json.dump(out, sys.stdout, indent=1, default=float)


if __name__ == "__main__":
    main()
