#!/usr/bin/env python
"""Agency workflow, the practitioner's interval check (run on aicr, never on the laptop).

    python practitioner_sbui.py REC AREAS [widths|exact|mc|all] [REPS]

REC is a checkout of the research record (reads exp/out/exp94_nlcd_points.csv and exp/out/exp94_gfc_points.csv);
AREAS is GFC2020's strata_size.csv (Strata, strata_ha). Writes one JSON document to stdout, progress to stderr.
Needs numpy and oe_inferencex (1.17.2) only.

What it measures
- widths: on NLCD 2016 AA (set B, year 2016, 21 strata, Table 3 pixel counts of Wickham et al. 2021) and GFC2020
  (149 strata, areas in ha), for overall accuracy, two-map differences, reference areas, map bias, user's and
  producer's accuracy: the normal-theory interval (Stehman 2014; Cochran 1977 section 6.11), the union bound over
  strata (as sizing.py), and the proposed intervals M1 and M2 below, with widths in points and their ratio to the
  normal-theory width. For overall accuracy also: a stratified Chernoff bound with the nuisance restricted to the same
  box, and M1's sensitivity to gamma, to the stratum order and to dropping the box. NLCD's published Table 4 numbers
  are printed beside ours.
- exact: the coverage of M1 and M2 computed exactly on small populations (every population, every sample), with a
  negative control, and a check of the convex-order lemma behind validity without replacement.
- mc: the coverage of the normal-theory interval and of M1/M2 on plug-in populations built from each agency sample
  (each stratum's points resampled with replacement, the agency's allocation and weights kept).

M1, the stratified betting union-intersection interval, for theta = sum_h W_h mu_h of a variable in [amin, amax]:
  1. Box: each stratum's mean in an exact interval (hypergeometric with counts, Chernoff-KL with areas); gamma is
     split equally over the intervals, so all hold together with probability >= 1 - gamma.
  2. Bets: stratum h bets b_h = lambda_h W_h / n_h per point, lambda_h = sqrt(2 log(1/a) / V_h), V_h the stratified
     variance estimated from the strata processed before h (order: ascending W_h^2 / n_h, a design-only key), capped
     so that every factor stays >= 1 - CAP.
  3. For a null eta (mu_h <= eta_h for all h), E(eta) = prod_h prod_i (1 + b_h (x_hi - eta_h)) has expectation <= 1:
     i.i.d. directly; without replacement because each stratum's product is the exponential of a linear function of
     its category counts and Hoeffding (1963, Theorem 4) bounds a convex function's mean under sampling without
     replacement by its mean with replacement; across strata by independence and the tower property (b_h depends
     only on earlier strata).
  4. theta <= theta0 is rejected when min{log E(eta): eta in the box, sum_h W_h eta_h <= theta0} >= log(1/a). Each
     log-factor is concave in eta_h, so its chord over the box lies below it; the minimum of the chords (a fractional
     knapsack) is a lower bound on the minimum, and rejecting on it keeps the level. The lower end is where that
     piecewise-linear bound crosses log(1/a); the upper end mirrors it.
  a = (alpha - gamma) / 2 per side: coverage >= 1 - alpha for every population and allocation (union bound + Markov).
Ratios R = Y / X (user's and producer's accuracy) invert the test of sum_h W_h E[y - R0 x] <= 0 over a grid of R0,
read in a fixed sequence from each end, so the ends are valid without monotonicity in R0.
M2 (user's accuracy of a class that has a stratum of its own, every unit of it mapped as the class): that stratum's
correct count gets its exact interval at 1 - 2a, the other strata's counts the box; a union bound.
"""
import csv
import itertools
import json
import math
import sys
import time
import traceback

import numpy as np

from oe_inferencex.estimate import hypergeom_interval

Z = 1.959963984540054
ALPHA = 0.05
GAMMA = 0.005                      # the box's share of alpha
CAP = 0.8                          # every betting factor stays >= 1 - CAP
M0 = 10.0                          # pseudo-points of the prior variance in the pooled variance used for the bets
GRID = np.linspace(0.0, 1.0, 2001)
TOL = 1e-12


def say(*parts):
    print(*parts, file=sys.stderr, flush=True)


def side(alpha=ALPHA, gamma=GAMMA):
    return (alpha - gamma) / 2.0


# ----------------------------------------------------------------------------- per-stratum boxes
def _kl(p, q):
    p = np.asarray(p, float)
    q = np.clip(np.asarray(q, float), 1e-300, 1.0 - 1e-15)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0) / q), 0.0)
        y = np.where(p < 1, (1.0 - p) * np.log(np.where(p < 1, 1.0 - p, 1.0) / (1.0 - q)), 0.0)
    return x + y


def kl_box(k, n, err):
    """Two-sided Chernoff-KL interval for a proportion from k of n, total error err (err/2 a side):
    {q : n KL(k/n || q) <= log(2/err)}. Valid for i.i.d. draws and, through the moment generating function, for
    draws without replacement (Hoeffding 1963, Thm 4). The bisection keeps the outer point, so ends round outward."""
    k = np.atleast_1d(np.asarray(k, float))
    n = np.atleast_1d(np.asarray(n, float))
    ok = n > 0
    nn = np.where(ok, n, 1.0)
    p = np.where(ok, k / nn, 0.5)
    c = math.log(2.0 / err) / nn
    a, b = np.zeros(p.shape), p.copy()                 # a outside (or 0), b inside
    for _ in range(80):
        m = 0.5 * (a + b)
        out = _kl(p, m) > c
        a, b = np.where(out, m, a), np.where(out, b, m)
    lo = np.where(p > 0, a, 0.0)
    a2, b2 = p.copy(), np.ones(p.shape)                # a2 inside, b2 outside (or 1)
    for _ in range(80):
        m = 0.5 * (a2 + b2)
        out = _kl(p, m) > c
        a2, b2 = np.where(out, a2, m), np.where(out, m, b2)
    hi = np.where(p < 1, b2, 1.0)
    return np.where(ok, lo, 0.0), np.where(ok, hi, 1.0)


def box(k, n, N, err):
    """Each stratum's interval for its proportion at total error err: exact (hypergeometric) when the counts N are
    known, the Chernoff-KL interval otherwise. A census is its own value; a stratum with no label gives [0, 1]."""
    k = np.atleast_1d(np.asarray(k, float))
    n = np.atleast_1d(np.asarray(n, float))
    if N is None:
        return kl_box(k, n, err)
    N = np.atleast_1d(np.asarray(N, float))
    lo, hi = np.zeros(k.size), np.ones(k.size)
    for i in range(k.size):
        if n[i] <= 0:
            continue
        if n[i] >= N[i]:
            lo[i] = hi[i] = k[i] / n[i]
            continue
        lo[i], hi[i] = hypergeom_interval(int(round(k[i])), int(round(n[i])), int(round(N[i])), conf=1.0 - err)
    return lo, hi


# ----------------------------------------------------------------------------- M1
def m1_lower(vals, cnt, W, lo, hi, amin, amax, a, rev=False):
    """Lower confidence ends (error a, given the box) of sum_h W_h mu_h, one per row g.

    vals (G, H, C): the values a point can take in each stratum; cnt (H, C): how many labelled points took each;
    W (H,) weights summing to 1; lo, hi (G, H): the box; amin, amax (G,): the variable's range."""
    vals = np.asarray(vals, float)
    cnt = np.asarray(cnt, float)
    W = np.asarray(W, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    amin = np.asarray(amin, float)
    amax = np.asarray(amax, float)
    G, H, C = vals.shape
    n = cnt.sum(1)
    nn = np.where(n > 0, n, 1.0)
    sig0 = ((amax - amin) / 2.0) ** 2                                         # (G,) the largest variance there is
    mean = (vals * cnt[None]).sum(2) / nn[None]
    ss = (cnt[None] * (vals - mean[..., None]) ** 2).sum(2)
    s2 = np.where(n[None] > 1, ss / np.maximum(n - 1.0, 1.0)[None], sig0[:, None])
    s2 = (n[None] * s2 + sig0[:, None]) / (n[None] + 1.0)                     # one pseudo-point: never exactly 0
    active = (n[None] > 0) & (hi - lo > 0)                                    # a stratum whose mean is not known
    key = W ** 2 / nn
    order = np.lexsort((np.arange(H), -key if rev else key))
    act = active[:, order]
    Wo, no = W[order], nn[order]
    s2o = s2[:, order]
    con = np.where(act, Wo ** 2 * s2o / no, 0.0)
    P = np.cumsum(con, 1) - con                                               # strata before: their own variances
    nw = np.where(act, no, 0.0)
    nsum = np.cumsum(nw, 1) - nw
    sw = np.where(act, no * s2o, 0.0)
    ssum = np.cumsum(sw, 1) - sw
    pooled = (ssum + M0 * sig0[:, None]) / (nsum + M0)                        # this one and later: the pooled one
    rem = np.where(act, Wo ** 2 / no, 0.0)
    R = np.cumsum(rem[:, ::-1], 1)[:, ::-1]
    V = np.maximum(P + R * pooled, 1e-300)
    lam = np.sqrt(2.0 * math.log(1.0 / a) / V)
    b = np.zeros((G, H))
    b[:, order] = np.where(act, lam * Wo / no, 0.0)
    span = hi - amin[:, None]
    b = np.where(span > 0, np.minimum(b, CAP / np.where(span > 0, span, 1.0)), 0.0)

    def f(eta):
        arg = 1.0 + b[..., None] * (vals - eta[..., None])
        t = np.where(cnt[None] > 0, cnt[None] * np.log(np.maximum(arg, 1e-300)), 0.0)
        return t.sum(2)

    flo, fhi = f(lo), f(hi)
    thr = math.log(1.0 / a) + 1e-9
    base = flo.sum(1)
    start = (W[None] * lo).sum(1)
    end = (W[None] * hi).sum(1)
    capb = W[None] * (hi - lo)
    pos = capb > 0
    drop = np.where(pos, np.maximum(flo - fhi, 0.0), 0.0)
    rate = np.where(pos, drop / np.where(pos, capb, 1.0), -1.0)
    idx = np.argsort(-rate, axis=1, kind="stable")                            # the adversary's best use of the budget
    ds = np.take_along_axis(drop, idx, 1)
    cs = np.take_along_axis(np.where(pos, capb, 0.0), idx, 1)
    cd = np.cumsum(ds, 1)
    cc = np.cumsum(cs, 1)
    after = base[:, None] - cd
    out = np.empty(G)
    for g in range(G):
        if base[g] < thr:
            out[g] = start[g]
            continue
        below = np.flatnonzero(after[g] < thr)
        if below.size == 0:
            out[g] = end[g]
            continue
        j = int(below[0])
        cb = base[g] - (cd[g, j] - ds[g, j])
        t = (cb - thr) / ds[g, j] if ds[g, j] > 0 else 0.0
        out[g] = start[g] + (cc[g, j] - cs[g, j]) + t * cs[g, j]
    return out - TOL


def m1_pair(vals, cnt, W, lo, hi, amin, amax, a, rev=False):
    L = m1_lower(vals, cnt, W, lo, hi, amin, amax, a, rev)
    U = -m1_lower(-np.asarray(vals, float), cnt, W, -np.asarray(hi, float), -np.asarray(lo, float),
                  -np.asarray(amax, float), -np.asarray(amin, float), a, rev)
    return L, U


def m1_linear(vals, cnt, W, lo, hi, amin, amax, alpha=ALPHA, gamma=GAMMA, rev=False):
    a = side(alpha, gamma)
    v = np.asarray(vals, float)[None]
    l2, h2 = np.asarray(lo, float)[None], np.asarray(hi, float)[None]
    L, U = m1_pair(v, cnt, W, l2, h2, np.array([float(amin)]), np.array([float(amax)]), a, rev)
    return float(L[0]), float(U[0])


# ----------------------------------------------------------------------------- normal theory (the agency's)
def strat_normal(n, s1, s2, W, N=None):
    """Stratified mean and its standard error from per-stratum sums of x and x^2 (Cochran 1977, 5A)."""
    n, s1, s2, W = (np.asarray(x, float) for x in (n, s1, s2, W))
    ok = n > 0
    nn = np.where(ok, n, 1.0)
    m = np.where(ok, s1 / nn, 0.0)
    est = float((W * m).sum())
    var_h = np.where(n > 1, (s2 - nn * m ** 2) / np.maximum(n - 1.0, 1.0), 0.0)
    fpc = np.ones(n.size) if N is None else np.clip(1.0 - n / np.asarray(N, float), 0.0, 1.0)
    v = float((W ** 2 * fpc * var_h / nn).sum())
    return est, math.sqrt(max(v, 0.0))


def ratio_normal(kA, kX, n, W, N=None):
    """Combined ratio estimator of Y/X for indicators y <= x and its linearised standard error (Cochran 1977, 6.11;
    Stehman 2014; Wickham et al. 2021, eq. 4 and 5)."""
    kA, kX, n, W = (np.asarray(x, float) for x in (kA, kX, n, W))
    ok = n > 0
    nn = np.where(ok, n, 1.0)
    yb, xb = np.where(ok, kA / nn, 0.0), np.where(ok, kX / nn, 0.0)
    Y, X = float((W * yb).sum()), float((W * xb).sum())
    if X <= 0:
        return None, None
    R = Y / X
    c = np.where(n > 1, nn / np.maximum(n - 1.0, 1.0), 0.0)
    s2y, s2x, sxy = c * yb * (1 - yb), c * xb * (1 - xb), c * (yb - yb * xb)
    fpc = np.ones(n.size) if N is None else np.clip(1.0 - n / np.asarray(N, float), 0.0, 1.0)
    v = float((W ** 2 * fpc * (s2y + R * R * s2x - 2 * R * sxy) / nn).sum()) / X ** 2
    return R, math.sqrt(max(v, 0.0))


def ratio_point(A, X, n, W):
    nn = np.where(np.asarray(n) > 0, n, 1.0)
    x = float((np.asarray(W) * np.asarray(X) / nn).sum())
    return float((np.asarray(W) * np.asarray(A) / nn).sum()) / x if x > 0 else None


# ----------------------------------------------------------------------------- comparator: Chernoff with the box
def chern_lower(k, n, W, lo, hi, a):
    """Lower end of the stratified Chernoff test with the nuisance restricted to the box: the sup over the box (and
    sum W eta <= theta0) of the Bernoulli moment-generating bound on P(theta_hat >= observed). Binary values only;
    every stratum needs a label. Both optimisations err on the conservative side."""
    k, n, W, lo, hi = (np.asarray(x, float) for x in (k, n, W, lo, hi))
    if (n <= 0).any():
        raise ValueError("the Chernoff comparator needs a label in every stratum")
    c = W / n
    t = float((W * k / n).sum())
    la = math.log(a)
    start = float((W * lo).sum())

    def sup_log_mgf(lam, theta0):
        A = np.expm1(np.minimum(lam * c, 700.0))
        if float((W * hi).sum()) <= theta0:
            eta = hi
        else:
            def eta_of(lnu):
                return np.clip(n / (math.exp(lnu) * W) - 1.0 / A, lo, hi)
            u, v = -60.0, 60.0
            for _ in range(90):
                mid = 0.5 * (u + v)
                if float((W * eta_of(mid)).sum()) > theta0:
                    u = mid
                else:
                    v = mid
            eta = eta_of(u)                       # sum W eta >= theta0: at or above the constrained sup
        return float((n * np.log1p(eta * A)).sum())

    def logp(theta0):
        if theta0 >= t:
            return 0.0
        f = lambda ll: sup_log_mgf(math.exp(ll), theta0) - math.exp(ll) * t
        cmax = float(c.max())
        lo_l, hi_l = math.log(1e-4 / cmax), math.log(40.0 / cmax)
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        x1, x2 = hi_l - gr * (hi_l - lo_l), lo_l + gr * (hi_l - lo_l)
        f1, f2 = f(x1), f(x2)
        for _ in range(60):
            if f1 < f2:
                hi_l, x2, f2 = x2, x1, f1
                x1 = hi_l - gr * (hi_l - lo_l)
                f1 = f(x1)
            else:
                lo_l, x1, f1 = x1, x2, f2
                x2 = lo_l + gr * (hi_l - lo_l)
                f2 = f(x2)
        return min(0.0, f1, f2)

    if logp(start) > la:
        return start
    u, v = start, t
    for _ in range(50):
        mid = 0.5 * (u + v)
        if logp(mid) <= la:
            u = mid
        else:
            v = mid
    return u


# ----------------------------------------------------------------------------- reports
def finish(out):
    keys = ("normal", "union", "m1", "m2", "m1_mixed", "chernoff_box", "m1_order_reversed", "m1_no_box")
    nrm = out.get("normal")
    nw = (nrm[1] - nrm[0]) if nrm and nrm[0] is not None else None
    w, r = {}, {}
    for key in keys:
        iv = out.get(key)
        if not iv or iv[0] is None or iv[1] is None:
            continue
        w[key] = 100.0 * (iv[1] - iv[0])
        if nw:
            r[key] = (iv[1] - iv[0]) / nw
    out["width_points"], out["ratio_to_normal"] = w, r


def binary_report(k, n, W, N=None, chern=False, extra=False):
    """A stratified share of a 0/1 variable (overall accuracy, a reference class's area share)."""
    k, n, W = (np.asarray(x, float) for x in (k, n, W))
    H = k.size
    Nn = None if N is None else np.asarray(N, float)
    est, se = strat_normal(n, k, k, W, Nn)
    open_ = (n > 0) if Nn is None else ((n > 0) & (n < Nn))
    Ho = max(int(open_.sum()), 1)
    lo, hi = box(k, n, Nn, GAMMA / Ho)
    vals = np.tile(np.array([0.0, 1.0]), (H, 1))
    cnt = np.stack([n - k, k], 1)
    t0 = time.time()
    L, U = m1_linear(vals, cnt, W, lo, hi, 0.0, 1.0)
    out = {"estimate": est, "se": se, "normal": [est - Z * se, est + Z * se], "m1": [L, U],
           "m1_seconds": time.time() - t0}
    ul, uh = box(k, n, Nn, ALPHA / Ho)
    out["union"] = [float((W * ul).sum()), float((W * uh).sum())]
    if extra:
        out["m1_order_reversed"] = list(m1_linear(vals, cnt, W, lo, hi, 0.0, 1.0, rev=True))
        census = np.zeros(H, bool) if Nn is None else ((n >= Nn) & (n > 0))
        p = np.where(n > 0, k / np.where(n > 0, n, 1.0), 0.0)
        out["m1_no_box"] = list(m1_linear(vals, cnt, W, np.where(census, p, 0.0), np.where(census, p, 1.0), 0.0, 1.0,
                                          gamma=0.0))
        sens = {}
        for gm in (0.001, 0.0025, 0.01, 0.02):
            l2, h2 = box(k, n, Nn, gm / Ho)
            iv = m1_linear(vals, cnt, W, l2, h2, 0.0, 1.0, gamma=gm)
            sens[str(gm)] = {"interval": list(iv), "width_points": 100.0 * (iv[1] - iv[0])}
        out["m1_gamma"] = sens
    if chern:
        try:
            t0 = time.time()
            a = side()
            cl = chern_lower(k, n, W, lo, hi, a)
            cu = 1.0 - chern_lower(n - k, n, W, 1.0 - hi, 1.0 - lo, a)
            out["chernoff_box"] = [cl, cu]
            out["chernoff_seconds"] = time.time() - t0
        except Exception:
            out["chernoff_box_error"] = traceback.format_exc()
    finish(out)
    return out


def diff_report(k10, k01, n, W, N=None):
    """The stratified mean of d = [first] - [second] (two maps' accuracy difference, or a map's area bias with
    [mapped as c, not c] - [not mapped as c, c])."""
    k10, k01, n, W = (np.asarray(x, float) for x in (k10, k01, n, W))
    H = n.size
    Nn = None if N is None else np.asarray(N, float)
    est, se = strat_normal(n, k10 - k01, k10 + k01, W, Nn)
    open_ = (n > 0) if Nn is None else ((n > 0) & (n < Nn))
    Ho = max(int(open_.sum()), 1)
    l10, u10 = box(k10, n, Nn, GAMMA / (2 * Ho))
    l01, u01 = box(k01, n, Nn, GAMMA / (2 * Ho))
    vals = np.tile(np.array([-1.0, 0.0, 1.0]), (H, 1))
    cnt = np.stack([k01, n - k10 - k01, k10], 1)
    t0 = time.time()
    L, U = m1_linear(vals, cnt, W, l10 - u01, u10 - l01, -1.0, 1.0)
    out = {"estimate": est, "se": se, "normal": [est - Z * se, est + Z * se], "m1": [L, U],
           "m1_seconds": time.time() - t0, "discordant": [int(k10.sum()), int(k01.sum())]}
    a10, b10 = box(k10, n, Nn, ALPHA / (2 * Ho))
    a01, b01 = box(k01, n, Nn, ALPHA / (2 * Ho))
    out["union"] = [float((W * (a10 - b01)).sum()), float((W * (b10 - a01)).sum())]
    finish(out)
    return out


def ratio_boxes(kA, kB, n, Nn, kinds, what):
    """Per-stratum bounds on the shares of A (mapped c, reference c) and B (UA: mapped c, reference not c; PA: not
    mapped c, reference c). kinds: 'mixed', 'own' (every unit of the stratum mapped c) or 'other' (no unit mapped c)."""
    H = n.size
    census = np.zeros(H, bool) if Nn is None else ((n >= Nn) & (n > 0))
    needA, needB = np.zeros(H, bool), np.zeros(H, bool)
    for h in range(H):
        if n[h] <= 0 or census[h]:
            continue
        kd = kinds[h]
        if kd == "mixed":
            needA[h] = needB[h] = True
        elif kd == "own":
            needA[h] = True
        elif what == "pa":                      # another class's stratum: A = 0, its omissions B are unknown
            needB[h] = True
    err = GAMMA / max(int(needA.sum() + needB.sum()), 1)
    lA, uA = box(kA, n, Nn, err)
    lB, uB = box(kB, n, Nn, err)
    for h in range(H):
        kd = kinds[h]
        if kd == "own":
            if what == "ua":                    # every unit mapped c: B = 1 - A
                lB[h], uB[h] = 1.0 - uA[h], 1.0 - lA[h]
            else:                               # no omission inside the class's own stratum
                lB[h] = uB[h] = 0.0
        elif kd == "other":
            if what == "ua":
                lA[h] = uA[h] = lB[h] = uB[h] = 0.0
            else:
                lA[h] = uA[h] = 0.0
    return lA, uA, lB, uB


def ratio_m1(kA, kB, n, W, Nn, kinds, what, grid=GRID):
    """M1 for R = sum W pA / sum W (pA + pB): the test of sum_h W_h E[y - R0 (y + b)] <= 0 at every grid value R0,
    read from each end of the grid until the first value not rejected."""
    H = n.size
    lA, uA, lB, uB = ratio_boxes(kA, kB, n, Nn, kinds, what)
    G = grid.size
    Rg = grid[:, None]
    lo = (1.0 - Rg) * lA[None] - Rg * uB[None]
    hi = (1.0 - Rg) * uA[None] - Rg * lB[None]
    vals = np.empty((G, H, 3))
    vals[..., 0] = -Rg
    vals[..., 1] = 0.0
    vals[..., 2] = 1.0 - Rg
    cnt = np.stack([kB, n - kA - kB, kA], 1)
    L, U = m1_pair(vals, cnt, W, lo, hi, -grid, 1.0 - grid, side())
    acc = np.flatnonzero(~(L > TOL))
    if acc.size == 0:
        RL = float(grid[-1])
    elif acc[0] == 0:
        RL = float(grid[0])
    else:
        RL = float(grid[acc[0] - 1])
    acc = np.flatnonzero(~(U < -TOL))
    if acc.size == 0:
        RU = float(grid[0])
    elif acc[-1] == G - 1:
        RU = float(grid[-1])
    else:
        RU = float(grid[acc[-1] + 1])
    return RL, RU


def m2_ua(kA, kB, n, W, Nn, kinds):
    """User's accuracy when one stratum holds only units mapped as the class ('own'): that stratum's correct share
    at its exact interval 1 - 2a, the mixed strata's two counts in the box; 'other' strata contribute nothing."""
    H = n.size
    d = kinds.index("own")
    census = np.zeros(H, bool) if Nn is None else ((n >= Nn) & (n > 0))
    mixed = [h for h in range(H) if kinds[h] == "mixed"]
    open_mixed = [h for h in mixed if n[h] > 0 and not census[h]]
    a = side() if open_mixed else ALPHA / 2.0
    Nd = None if Nn is None else Nn[[d]]
    lAd, uAd = box(kA[[d]], n[[d]], Nd, 2.0 * a)
    lAd, uAd = float(lAd[0]), float(uAd[0])
    if mixed:
        mm = np.array(mixed)
        err = GAMMA / max(2 * len(open_mixed), 1)
        Nm = None if Nn is None else Nn[mm]
        lA, uA = box(kA[mm], n[mm], Nm, err)
        lB, uB = box(kB[mm], n[mm], Nm, err)
        Wm = W[mm]
        A_lo, A_hi, B_lo, B_hi = (float((Wm * x).sum()) for x in (lA, uA, lB, uB))
    else:
        A_lo = A_hi = B_lo = B_hi = 0.0
    Wd = float(W[d])
    return (Wd * lAd + A_lo) / (Wd + A_lo + B_hi), min(1.0, (Wd * uAd + A_hi) / (Wd + A_hi + B_lo))


def ratio_report(kA, kB, n, W, N=None, kinds=None, what="ua", grid=GRID, extras=True, do_m1=True, force_m1=False):
    kA, kB, n, W = (np.asarray(x, float) for x in (kA, kB, n, W))
    H = n.size
    kinds = list(kinds) if kinds is not None else ["mixed"] * H
    Nn = None if N is None else np.asarray(N, float)
    R_hat, se = ratio_normal(kA, kA + kB, n, W, Nn)
    out = {"estimate": R_hat, "se": se, "normal": None if R_hat is None else [R_hat - Z * se, R_hat + Z * se],
           "points": [int(kA.sum()), int(kB.sum())]}
    if R_hat is None and not force_m1:
        return out
    if do_m1:
        t0 = time.time()
        out["m1"] = list(ratio_m1(kA, kB, n, W, Nn, kinds, what, grid))
        out["m1_seconds"] = time.time() - t0
        if extras and any(kd != "mixed" for kd in kinds):
            out["m1_mixed"] = list(ratio_m1(kA, kB, n, W, Nn, ["mixed"] * H, what, grid))
    if what == "ua" and "own" in kinds:
        out["m2"] = list(m2_ua(kA, kB, n, W, Nn, kinds))
    if extras:
        open_ = (n > 0) if Nn is None else ((n > 0) & (n < Nn))
        Ho = max(int(open_.sum()), 1)
        la, ua = box(kA, n, Nn, ALPHA / (2 * Ho))
        lb, ub = box(kB, n, Nn, ALPHA / (2 * Ho))
        A_lo, A_hi, B_lo, B_hi = (float((W * x).sum()) for x in (la, ua, lb, ub))
        out["union"] = [A_lo / (A_lo + B_hi) if A_lo + B_hi > 0 else 0.0,
                        A_hi / (A_hi + B_lo) if A_hi + B_lo > 0 else 1.0]
    finish(out)
    return out


def summarise(q):
    out = {}
    for k, v in q.items():
        if isinstance(v, dict) and "width_points" in v:
            out[k] = {"estimate_pct": None if v.get("estimate") is None else 100 * v["estimate"],
                      "se_pct": None if v.get("se") is None else 100 * v["se"],
                      "width_points": v["width_points"], "ratio_to_normal": v["ratio_to_normal"]}
    for c, row in (q.get("per_class_l2") or {}).items():
        for key, v in row.items():
            if isinstance(v, dict) and "width_points" in v:
                out[f"{c}:{key}"] = {"estimate_pct": None if v.get("estimate") is None else 100 * v["estimate"],
                                     "se_pct": None if v.get("se") is None else 100 * v["se"],
                                     "width_points": v["width_points"], "ratio_to_normal": v["ratio_to_normal"]}
    return out


# ----------------------------------------------------------------------------- NLCD 2016 AA (set B)
# Table 3 of Wickham et al. (2021): pixel counts of the 21 strata (population 8,658,519,089 pixels)
NLCD_N = {1: 144260856, 2: 255518661, 3: 130760739, 4: 59945943, 5: 21576459, 6: 88893116, 7: 825184784,
          8: 979538237, 9: 316640402, 10: 1864314143, 11: 1159802223, 12: 559677646, 13: 1429403249,
          14: 380971449, 15: 117993354, 16: 146679429, 17: 6772002, 18: 73557194, 19: 70038668, 20: 4483066,
          21: 22289822}
# the 15 no-change strata hold one NLCD class in 2011 and 2016 (Table 3); the six change strata are mixed
NLCD_PURE = {1: 11, 2: 21, 3: 22, 4: 23, 5: 24, 6: 31, 7: 41, 8: 42, 9: 43, 10: 52, 11: 71, 12: 81, 13: 82, 14: 90,
             15: 95}
NLCD_CLASSES = [11, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]
NLCD_TO_LCMAP = {21: 1, 22: 1, 23: 1, 24: 1, 81: 2, 82: 2, 52: 3, 71: 3, 41: 4, 42: 4, 43: 4, 11: 5, 90: 6, 95: 6,
                 12: 7, 31: 8}
# Table 4 (NLCD2016 2016, level II, percent and standard error), the text and Table 1 / Table 6 for overall accuracy
PUB = {
    "oa_l2_primary": [72.1, 0.9], "oa_l2_pri_or_alt": [86.4, 0.6], "oa_l1_primary": [79.0, 0.8],
    "oa_l1_pri_or_alt": [90.6, 0.6],
    "note": "Table 4's caption gives OAa = 86.9 (0.6); the text and Table 1 give 86.4 (0.6)",
    "classes": {
        "11": {"ua": [92, 2], "uaa": [99, 1], "pa": [94, 2], "paa": [95, 2], "map_area": 1.7300, "ref_area": 1.6811, "tse": 0.0490, "mb": 0.05},
        "21": {"ua": [72, 3], "uaa": [88, 2], "pa": [42, 3], "paa": [61, 4], "map_area": 2.9802, "ref_area": 5.0945, "tse": 0.3437, "mb": -2.11},
        "22": {"ua": [43, 3], "uaa": [74, 3], "pa": [43, 4], "paa": [70, 6], "map_area": 1.5280, "ref_area": 1.5479, "tse": 0.1530, "mb": -0.02},
        "23": {"ua": [47, 4], "uaa": [77, 3], "pa": [47, 4], "paa": [84, 4], "map_area": 0.7201, "ref_area": 0.7222, "tse": 0.0450, "mb": 0.00},
        "24": {"ua": [63, 3], "uaa": [75, 3], "pa": [49, 4], "paa": [78, 6], "map_area": 0.2642, "ref_area": 0.3392, "tse": 0.0303, "mb": -0.08},
        "31": {"ua": [58, 5], "uaa": [77, 4], "pa": [63, 9], "paa": [77, 9], "map_area": 1.0355, "ref_area": 0.9482, "tse": 0.1459, "mb": 0.09},
        "41": {"ua": [82, 2], "uaa": [90, 2], "pa": [67, 2], "paa": [79, 2], "map_area": 9.7358, "ref_area": 11.9059, "tse": 0.4230, "mb": -2.17},
        "42": {"ua": [85, 2], "uaa": [93, 2], "pa": [75, 2], "paa": [85, 2], "map_area": 11.8023, "ref_area": 13.2366, "tse": 0.4307, "mb": -1.43},
        "43": {"ua": [52, 4], "uaa": [75, 4], "pa": [55, 4], "paa": [78, 4], "map_area": 3.7826, "ref_area": 3.5917, "tse": 0.2810, "mb": 0.19},
        "52": {"ua": [68, 2], "uaa": [87, 2], "pa": [81, 2], "paa": [95, 1], "map_area": 22.7740, "ref_area": 19.0479, "tse": 0.6451, "mb": 3.73},
        "71": {"ua": [65, 3], "uaa": [91, 2], "pa": [66, 2], "paa": [91, 1], "map_area": 14.4242, "ref_area": 14.1393, "tse": 0.6362, "mb": 0.28},
        "81": {"ua": [64, 4], "uaa": [77, 4], "pa": [54, 3], "paa": [76, 3], "map_area": 6.5016, "ref_area": 7.7062, "tse": 0.5071, "mb": -1.20},
        "82": {"ua": [89, 2], "uaa": [93, 1], "pa": [91, 2], "paa": [94, 1], "map_area": 16.7525, "ref_area": 16.3984, "tse": 0.4048, "mb": 0.35},
        "90": {"ua": [45, 4], "uaa": [61, 4], "pa": [86, 3], "paa": [96, 2], "map_area": 4.4696, "ref_area": 2.3380, "tse": 0.1979, "mb": 2.13},
        "95": {"ua": [54, 4], "uaa": [69, 4], "pa": [62, 6], "paa": [80, 6], "map_area": 1.4994, "ref_area": 1.3028, "tse": 0.1311, "mb": 0.20}}}


def _int(s):
    return int(float(s)) if s not in ("", None) else 0


def load_nlcd(rec, year="2016"):
    path = f"{rec}/exp/out/exp94_nlcd_points.csv"
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if r["set"] == "B" and r["year"] == year]
    col = lambda name, cast=_int: np.array([cast(r[name]) for r in rows])
    return {"stratum": col("stratum"), "weight": col("weight", float), "pri": col("ref_nlcd"),
            "alt": col("ref_nlcd_alternate"), "map": col("nlcd"), "lcmap": col("lcmap")}


def nlcd_design(d):
    S = sorted(NLCD_N)
    pos = {s: i for i, s in enumerate(S)}
    H = len(S)
    keep = (d["pri"] > 0) & (d["map"] > 0) & np.isin(d["stratum"], S)
    dd = {k: v[keep] for k, v in d.items()}
    hidx = np.array([pos[int(s)] for s in dd["stratum"]])
    n = np.bincount(hidx, minlength=H).astype(float)
    N = np.array([NLCD_N[s] for s in S], float)
    return S, pos, H, dd, hidx, n, N, N / N.sum(), int((~keep).sum())


def nlcd_pure(dd, hidx, pos):
    viol, pure = {}, {}
    for s, c in NLCD_PURE.items():
        bad = int((dd["map"][hidx == pos[s]] != c).sum())
        if bad:
            viol[str(s)] = bad
        else:
            pure[pos[s]] = c
    return pure, viol


def analyse_nlcd(rec):
    d = load_nlcd(rec)
    S, pos, H, dd, hidx, n, N, W, dropped = nlcd_design(d)
    cnt_of = lambda mask: np.bincount(hidx[mask], minlength=H).astype(float)
    res = {"published": PUB, "design": {"points": int(d["stratum"].size), "dropped_no_label_or_stratum": dropped}}
    strata = {}
    for i, s in enumerate(S):
        ws = dd["weight"][hidx == i]
        strata[str(s)] = {"n": int(n[i]), "N_table3": int(N[i]), "N_from_weights": float(ws.sum()) if ws.size else 0.0,
                          "weight_spread": float(ws.max() - ws.min()) if ws.size else None, "W": float(W[i])}
    res["design"]["strata"] = strata
    res["design"]["max_rel_diff_N"] = max(abs(v["N_from_weights"] - v["N_table3"]) / v["N_table3"]
                                          for v in strata.values())
    pure, viol = nlcd_pure(dd, hidx, pos)
    res["design"]["purity_violations"] = viol
    mp, pri, alt = dd["map"], dd["pri"], dd["alt"]
    agree_p = mp == pri
    agree_a = agree_p | (mp == alt)
    q = {}
    say("nlcd: overall accuracy")
    q["oa_l2_primary"] = binary_report(cnt_of(agree_p), n, W, N, chern=True, extra=True)
    q["oa_l2_pri_or_alt"] = binary_report(cnt_of(agree_a), n, W, N, chern=True)
    l1 = lambda x: x // 10
    ap1 = l1(mp) == l1(pri)
    q["oa_l1_primary"] = binary_report(cnt_of(ap1), n, W, N)
    q["oa_l1_pri_or_alt"] = binary_report(cnt_of(ap1 | (l1(mp) == l1(alt))), n, W, N)
    rstar = np.where(agree_a, mp, pri)          # the reference a point counts as under "primary or alternate"
    pc = {}
    for c in NLCD_CLASSES:
        say(f"nlcd: class {c}")
        kinds = ["own" if pure.get(h) == c else ("other" if h in pure else "mixed") for h in range(H)]
        row = {"own_strata": [S[h] for h in range(H) if kinds[h] == "own"],
               "mixed_strata": [S[h] for h in range(H) if kinds[h] == "mixed"]}
        for tag, ref in (("primary", pri), ("pri_or_alt", rstar)):
            A = cnt_of((mp == c) & (ref == c))
            B = cnt_of((mp == c) & (ref != c))
            O = cnt_of((mp != c) & (ref == c))
            full = tag == "primary"
            row[f"ua_{tag}"] = ratio_report(A, B, n, W, N, kinds, "ua", extras=full)
            row[f"pa_{tag}"] = ratio_report(A, O, n, W, N, kinds, "pa", extras=full)
            row[f"ref_area_{tag}"] = binary_report(cnt_of(ref == c), n, W, N)
            row[f"map_bias_{tag}"] = diff_report(B, O, n, W, N)
        e_map, s_map = strat_normal(n, cnt_of(mp == c), cnt_of(mp == c), W, N)
        row["map_area"] = {"estimate": e_map, "se": s_map}
        num, _ = strat_normal(n, cnt_of((mp == c) & agree_a), cnt_of((mp == c) & agree_a), W, N)
        den, _ = strat_normal(n, cnt_of(pri == c), cnt_of(pri == c), W, N)
        row["paa_variant_primary_denominator"] = num / den if den > 0 else None
        pc[str(c)] = row
    q["per_class_l2"] = pc
    xw = lambda arr: np.array([NLCD_TO_LCMAP.get(int(v), 0) for v in arr])
    p8, a8, m8, lc = xw(pri), xw(alt), xw(mp), dd["lcmap"]
    res["design"]["lcmap_missing"] = int((lc <= 0).sum())
    for tag in ("primary", "pri_or_alt"):
        okA, okB = m8 == p8, lc == p8
        if tag == "pri_or_alt":
            okA, okB = okA | (m8 == a8), okB | (lc == a8)
        say(f"nlcd: NLCD against LCMAP, {tag}")
        q[f"oa_nlcd8_{tag}"] = binary_report(cnt_of(okA), n, W, N)
        q[f"oa_lcmap_{tag}"] = binary_report(cnt_of(okB), n, W, N)
        q[f"nlcd8_minus_lcmap_{tag}"] = diff_report(cnt_of(okA & ~okB), cnt_of(~okA & okB), n, W, N)
    res["quantities"] = q
    res["summary"] = summarise(q)
    return res


# ----------------------------------------------------------------------------- GFC2020
def load_gfc(rec, areas):
    with open(areas) as fh:
        area = {int(r["Strata"]): float(r["strata_ha"]) for r in csv.DictReader(fh)}
    with open(f"{rec}/exp/out/exp94_gfc_points.csv") as fh:
        rows = list(csv.DictReader(fh))
    S = sorted(area)
    pos = {s: i for i, s in enumerate(S)}
    keep = [r for r in rows if int(r["stratum"]) in pos]
    hidx = np.array([pos[int(r["stratum"])] for r in keep])
    ref = np.array([int(r["ref_forest"]) for r in keep])
    g = np.array([int(r["gfc_v2"]) for r in keep])
    wc = np.array([int(r["worldcover_forest"]) if r["worldcover_forest"] != "" else -1 for r in keep])
    A = np.array([area[s] for s in S])
    H = len(S)
    n = np.bincount(hidx, minlength=H).astype(float)
    return {"S": S, "H": H, "hidx": hidx, "ref": ref, "gfc": g, "wc": wc, "W": A / A.sum(), "n": n,
            "points": len(rows), "outside_table": len(rows) - len(keep)}


def analyse_gfc(rec, areas):
    G = load_gfc(rec, areas)
    H, hidx, ref, g, wc, W, n = G["H"], G["hidx"], G["ref"], G["gfc"], G["wc"], G["W"], G["n"]
    cnt_of = lambda mask: np.bincount(hidx[mask], minlength=H).astype(float)
    res = {"design": {"points": G["points"], "outside_table": G["outside_table"], "strata": H,
                      "empty_strata": int((n == 0).sum()), "n_min": int(n[n > 0].min()), "n_max": int(n.max()),
                      "W_max": float(W.max()), "worldcover_missing": int((wc < 0).sum()),
                      "sizes": "areas (ha): each stratum read as a continuum sampled by points; Chernoff-KL boxes; no "
                               "finite-population correction"}}
    q = {}
    ok_g, ok_w = g == ref, wc == ref
    say("gfc: overall accuracy")
    q["oa_gfc_v2"] = binary_report(cnt_of(ok_g), n, W, None, chern=True, extra=True)
    q["oa_worldcover"] = binary_report(cnt_of(ok_w), n, W, None, chern=True)
    q["gfc_minus_worldcover"] = diff_report(cnt_of(ok_g & ~ok_w), cnt_of(~ok_g & ok_w), n, W, None)
    q["forest_share_reference"] = binary_report(cnt_of(ref == 1), n, W, None, chern=True)
    for name, m in (("gfc_v2", g), ("worldcover", wc)):
        for c, cname in ((1, "forest"), (0, "nonforest")):
            say(f"gfc: {name} {cname}")
            Ac = cnt_of((m == c) & (ref == c))
            q[f"ua_{cname}_{name}"] = ratio_report(Ac, cnt_of((m == c) & (ref != c)), n, W, None, None, "ua")
            q[f"pa_{cname}_{name}"] = ratio_report(Ac, cnt_of((m != c) & (ref == c)), n, W, None, None, "pa")
        q[f"forest_map_bias_{name}"] = diff_report(cnt_of((m == 1) & (ref != 1)), cnt_of((m != 1) & (ref == 1)), n, W,
                                                   None)
    res["quantities"] = q
    res["summary"] = summarise(q)
    return res


# ----------------------------------------------------------------------------- exact coverage on small populations
def hyp_pmf(N, K, n):
    tot = math.comb(N, n)
    return np.array([math.comb(K, k) * math.comb(N - K, n - k) / tot for k in range(n + 1)])


def binom_pmf(n, p):
    return np.array([math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)])


def mvh_pmf(counts, n):
    """P(a, b) of a sample of n without replacement from a population with category counts (A, B, rest)."""
    A, B, Rr = (int(x) for x in counts)
    tot = math.comb(A + B + Rr, n)
    return {(a, b): math.comb(A, a) * math.comb(B, b) * math.comb(Rr, n - a - b) / tot
            for a in range(n + 1) for b in range(n + 1 - a)}


def exact_binary(N, n, iid=False, alpha=ALPHA, gamma=GAMMA, pstep=0.05, W=None):
    H = len(n)
    Wv = np.array(W if W is not None else N, float)
    Wv = Wv / Wv.sum()
    Nn = None if iid else np.array(N, float)
    nn = np.array(n, float)
    shape = tuple(x + 1 for x in n)
    methods = ("m1", "union", "normal")
    L = {m: np.zeros(shape) for m in methods}
    U = {m: np.zeros(shape) for m in methods}
    open_ = (nn > 0) if Nn is None else ((nn > 0) & (nn < Nn))
    Ho = max(int(open_.sum()), 1)
    vals = np.tile(np.array([0.0, 1.0]), (H, 1))
    for kc in itertools.product(*[range(x + 1) for x in n]):
        k = np.array(kc, float)
        lo, hi = box(k, nn, Nn, gamma / Ho)
        L["m1"][kc], U["m1"][kc] = m1_linear(vals, np.stack([nn - k, k], 1), Wv, lo, hi, 0.0, 1.0, alpha=alpha,
                                             gamma=gamma)
        ul, uh = box(k, nn, Nn, alpha / Ho)
        L["union"][kc], U["union"][kc] = float((Wv * ul).sum()), float((Wv * uh).sum())
        est, se = strat_normal(nn, k, k, Wv, Nn)
        L["normal"][kc], U["normal"][kc] = est - Z * se, est + Z * se
    worst = {m: (2.0, None) for m in methods}
    width = {m: [] for m in methods}
    if iid:
        pg = np.round(np.arange(0.0, 1.0 + 1e-9, pstep), 10)
        pops = itertools.product(*[range(pg.size)] * H)
    else:
        pops = itertools.product(*[range(x + 1) for x in N])
    for pop in pops:
        if iid:
            ps = [float(pg[i]) for i in pop]
            pm = [binom_pmf(n[h], ps[h]) for h in range(H)]
            theta = float(sum(Wv[h] * ps[h] for h in range(H)))
        else:
            pm = [hyp_pmf(N[h], pop[h], n[h]) for h in range(H)]
            theta = float(sum(Wv[h] * pop[h] / N[h] for h in range(H)))
        P = pm[0]
        for h in range(1, H):
            P = np.multiply.outer(P, pm[h])
        for m in methods:
            cov = float((P * ((L[m] <= theta + 1e-12) & (theta - 1e-12 <= U[m]))).sum())
            if cov < worst[m][0]:
                worst[m] = (cov, [float(x) for x in pop])
            width[m].append(float((P * (U[m] - L[m])).sum()))
    return {"N": list(N), "n": list(n), "iid": iid, "alpha": alpha, "gamma": gamma,
            "min_coverage": {m: worst[m][0] for m in methods}, "worst_population": {m: worst[m][1] for m in methods},
            "mean_expected_width": {m: float(np.mean(width[m])) for m in methods}}


def exact_three(N, n, kind="diff", kinds=None, grid=None):
    """kind 'diff': the mean of d = [A] - [B]; 'ua': the ratio A / (A + B), kinds per stratum ('own': every unit
    mapped as the class). Every population and every sample is enumerated (multivariate hypergeometric)."""
    H = len(n)
    Wv = np.array(N, float) / float(sum(N))
    Nn, nn = np.array(N, float), np.array(n, float)
    kinds = list(kinds) if kinds else ["mixed"] * H
    grid = GRID if grid is None else grid
    outs = [[(a, n[h] - a) for a in range(n[h] + 1)] if kinds[h] == "own" else
            [(a, b) for a in range(n[h] + 1) for b in range(n[h] + 1 - a)] for h in range(H)]
    methods = ["m1", "union", "normal"] + (["m2"] if kind == "ua" and "own" in kinds else [])
    lo = {m: [] for m in methods}
    hi = {m: [] for m in methods}
    full = [0.0, 1.0] if kind == "ua" else [-1.0, 1.0]
    for combo in itertools.product(*outs):
        kA = np.array([c[0] for c in combo], float)
        kB = np.array([c[1] for c in combo], float)
        if kind == "diff":
            r = diff_report(kA, kB, nn, Wv, Nn)
        else:
            r = ratio_report(kA, kB, nn, Wv, Nn, kinds, "ua", grid=grid, force_m1=True)
        for m in methods:
            iv = r.get(m)
            if not iv or iv[0] is None or iv[1] is None:
                iv = full
            lo[m].append(iv[0])
            hi[m].append(iv[1])
    lo = {m: np.array(v) for m, v in lo.items()}
    hi = {m: np.array(v) for m, v in hi.items()}
    popsets = [[(A, N[h] - A) for A in range(N[h] + 1)] if kinds[h] == "own" else
               [(A, B) for A in range(N[h] + 1) for B in range(N[h] + 1 - A)] for h in range(H)]
    worst = {m: (2.0, None) for m in methods}
    widths = {m: [] for m in methods}
    npop = 0
    for pop in itertools.product(*popsets):
        if kind == "diff":
            theta = float(sum(Wv[h] * (pop[h][0] - pop[h][1]) / N[h] for h in range(H)))
        else:
            X = float(sum(Wv[h] * (pop[h][0] + pop[h][1]) / N[h] for h in range(H)))
            if X <= 0:
                continue
            theta = float(sum(Wv[h] * pop[h][0] / N[h] for h in range(H))) / X
        npop += 1
        P = None
        for h in range(H):
            A, B = pop[h]
            pm = mvh_pmf((A, B, N[h] - A - B), n[h])
            v = np.array([pm.get(o, 0.0) for o in outs[h]])
            P = v if P is None else np.outer(P, v).ravel()
        for m in methods:
            cov = float(P[(lo[m] <= theta + 1e-12) & (theta - 1e-12 <= hi[m])].sum())
            if cov < worst[m][0]:
                worst[m] = (cov, [list(x) for x in pop])
            widths[m].append(float((P * (hi[m] - lo[m])).sum()))
    return {"N": list(N), "n": list(n), "kind": kind, "kinds": kinds, "populations": npop,
            "min_coverage": {m: worst[m][0] for m in methods}, "worst_population": {m: worst[m][1] for m in methods},
            "mean_expected_width": {m: float(np.mean(widths[m])) for m in methods}}


def lemma_check(trials=400, seed=0):
    """E[prod_i (1 + b (x_i - eta))] <= 1 under sampling without replacement whenever the population mean is <= eta,
    for three-valued x and a fixed bet b >= 0 (the convex-order step of M1's proof), computed exactly."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        N = int(rng.integers(3, 13))
        n = int(rng.integers(1, min(N, 6) + 1))
        v = np.sort(rng.uniform(-1.0, 1.0, 3))
        cts = [int(x) for x in rng.multinomial(N, rng.dirichlet([1.0, 1.0, 1.0]))]
        mu = float((v * np.array(cts)).sum()) / N
        eta = float(rng.uniform(mu, v[-1])) if mu < v[-1] else mu
        span = eta - v[0]
        b = float(rng.uniform(0.0, CAP / span)) if span > 0 else 0.0
        tot = math.comb(N, n)
        E = 0.0
        for a in range(n + 1):
            for bb in range(n + 1 - a):
                c = n - a - bb
                p = math.comb(cts[0], a) * math.comb(cts[1], bb) * math.comb(cts[2], c) / tot
                if p:
                    E += p * (1 + b * (v[0] - eta)) ** a * (1 + b * (v[1] - eta)) ** bb * (1 + b * (v[2] - eta)) ** c
        worst = max(worst, E)
    return {"trials": trials, "max_expectation": worst, "holds": worst <= 1.0 + 1e-12}


def exact_checks():
    res = {}
    say("exact: binary designs")
    res["binary_2strata_counts"] = exact_binary((12, 20), (6, 5))
    res["binary_3strata_counts"] = exact_binary((8, 10, 15), (4, 4, 6))
    res["binary_2strata_larger"] = exact_binary((30, 40), (12, 10))
    res["binary_census_and_sampled"] = exact_binary((6, 25), (6, 8))
    res["binary_iid_areas"] = exact_binary((7, 3), (10, 6), iid=True, W=(0.7, 0.3))
    res["negative_control_alpha0.9"] = exact_binary((12, 20), (6, 5), alpha=0.9, gamma=0.3)
    say("exact: difference and ratio designs")
    res["difference_2strata"] = exact_three((8, 10), (4, 5), "diff")
    res["ua_ratio_mixed"] = exact_three((8, 8), (4, 4), "ua", grid=np.linspace(0.0, 1.0, 201))
    res["ua_ratio_own_plus_mixed"] = exact_three((10, 8), (5, 4), "ua", kinds=["own", "mixed"],
                                                 grid=np.linspace(0.0, 1.0, 201))
    say("exact: convex-order lemma")
    res["lemma"] = lemma_check()
    return res


# ----------------------------------------------------------------------------- plug-in Monte Carlo
def _record(stats, name, rep, truth, methods):
    for m in methods:
        iv = rep.get(m)
        if not iv or iv[0] is None or iv[1] is None or truth is None:
            continue
        s = stats.setdefault(f"{name}:{m}", {"covered": 0, "draws": 0, "width_sum": 0.0})
        s["draws"] += 1
        s["covered"] += int(iv[0] - 1e-12 <= truth <= iv[1] + 1e-12)
        s["width_sum"] += iv[1] - iv[0]


def mc(rec, areas, reps):
    rng = np.random.default_rng(117)
    stats = {}
    grid = np.linspace(0.0, 1.0, 1001)
    # NLCD 2016 set B: each stratum's points are the stratum; draws with replacement at the agency's n_h
    d = load_nlcd(rec)
    S, pos, H, dd, hidx, n, N, W, _ = nlcd_design(d)
    pure, _ = nlcd_pure(dd, hidx, pos)
    mp, pri = dd["map"], dd["pri"]
    members = [np.flatnonzero(hidx == h) for h in range(H)]
    classes = [11, 31, 43, 52, 90, 95]
    c0 = lambda mask: np.bincount(hidx[mask], minlength=H).astype(float)
    nn = np.where(n > 0, n, 1.0)
    truth = {"oa": float((W * c0(mp == pri) / nn).sum())}
    for c in classes:
        A, B, O = c0((mp == c) & (pri == c)), c0((mp == c) & (pri != c)), c0((mp != c) & (pri == c))
        truth[f"ua{c}"] = ratio_point(A, A + B, n, W)
        truth[f"pa{c}"] = ratio_point(A, A + O, n, W)
    for r in range(reps):
        sel = np.concatenate([rng.choice(mem, size=mem.size, replace=True) for mem in members if mem.size])
        hs, m_, p_ = hidx[sel], mp[sel], pri[sel]
        cf = lambda mask: np.bincount(hs[mask], minlength=H).astype(float)
        _record(stats, "nlcd_oa_primary", binary_report(cf(m_ == p_), n, W, None), truth["oa"], ("normal", "m1"))
        for c in classes:
            kinds = ["own" if pure.get(h) == c else ("other" if h in pure else "mixed") for h in range(H)]
            A, B, O = cf((m_ == c) & (p_ == c)), cf((m_ == c) & (p_ != c)), cf((m_ != c) & (p_ == c))
            _record(stats, f"nlcd_ua{c}", ratio_report(A, B, n, W, None, kinds, "ua", grid=grid, extras=False,
                                                       do_m1=False), truth[f"ua{c}"], ("normal", "m2"))
            _record(stats, f"nlcd_pa{c}", ratio_report(A, O, n, W, None, kinds, "pa", grid=grid, extras=False),
                    truth[f"pa{c}"], ("normal", "m1"))
        if r % 50 == 0:
            say(f"mc nlcd {r}/{reps}")
    # GFC2020
    G = load_gfc(rec, areas)
    H, hidx, ref, g, wc, W, n = G["H"], G["hidx"], G["ref"], G["gfc"], G["wc"], G["W"], G["n"]
    members = [np.flatnonzero(hidx == h) for h in range(H)]
    c0 = lambda mask: np.bincount(hidx[mask], minlength=H).astype(float)
    nn = np.where(n > 0, n, 1.0)
    ok_g, ok_w = g == ref, wc == ref
    A, B, O = c0((g == 1) & (ref == 1)), c0((g == 1) & (ref != 1)), c0((g != 1) & (ref == 1))
    truth = {"oa": float((W * c0(ok_g) / nn).sum()),
             "diff": float((W * (c0(ok_g & ~ok_w) - c0(~ok_g & ok_w)) / nn).sum()),
             "ua": ratio_point(A, A + B, n, W), "pa": ratio_point(A, A + O, n, W)}
    for r in range(reps):
        sel = np.concatenate([rng.choice(mem, size=mem.size, replace=True) for mem in members if mem.size])
        hs, rf, gg, ww = hidx[sel], ref[sel], g[sel], wc[sel]
        cf = lambda mask: np.bincount(hs[mask], minlength=H).astype(float)
        og, ow = gg == rf, ww == rf
        _record(stats, "gfc_oa", binary_report(cf(og), n, W, None), truth["oa"], ("normal", "m1"))
        _record(stats, "gfc_minus_wc", diff_report(cf(og & ~ow), cf(~og & ow), n, W, None), truth["diff"],
                ("normal", "m1"))
        A, B, O = cf((gg == 1) & (rf == 1)), cf((gg == 1) & (rf != 1)), cf((gg != 1) & (rf == 1))
        _record(stats, "gfc_ua_forest", ratio_report(A, B, n, W, None, None, "ua", grid=grid, extras=False),
                truth["ua"], ("normal", "m1"))
        _record(stats, "gfc_pa_forest", ratio_report(A, O, n, W, None, None, "pa", grid=grid, extras=False),
                truth["pa"], ("normal", "m1"))
        if r % 50 == 0:
            say(f"mc gfc {r}/{reps}")
    return {"reps": reps, "truth_is": "each agency sample's own estimates (the plug-in population)",
            "results": {k: {"coverage": v["covered"] / v["draws"], "draws": v["draws"],
                            "mean_width_points": 100.0 * v["width_sum"] / v["draws"]} for k, v in stats.items()}}


# ----------------------------------------------------------------------------- main
def _json(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def guarded(fn, *args):
    t0 = time.time()
    try:
        r = fn(*args)
        if isinstance(r, dict):
            r["seconds"] = time.time() - t0
        return r
    except Exception:
        return {"error": traceback.format_exc(), "seconds": time.time() - t0}


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    rec, areas = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "widths"
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    out = {"script": "practitioner_sbui.py", "mode": mode,
           "constants": {"alpha": ALPHA, "gamma": GAMMA, "per_side": side(), "cap": CAP, "m0": M0,
                         "grid_step": float(GRID[1] - GRID[0])}}
    if mode in ("widths", "all"):
        out["nlcd2016_setB"] = guarded(analyse_nlcd, rec)
        out["gfc2020"] = guarded(analyse_gfc, rec, areas)
    if mode in ("exact", "all"):
        out["exact"] = guarded(exact_checks)
    if mode in ("mc", "all"):
        out["mc"] = guarded(mc, rec, areas, reps)
    json.dump(out, sys.stdout, indent=1, default=_json)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
