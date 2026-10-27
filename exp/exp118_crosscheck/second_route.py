#!/usr/bin/env python3
"""exp118 arm A: grade's intervals by a second route.

Written from the specification in docs/plan/grade_second_route_and_coverage.md alone (sections "The specification"
and "Arm A"), the layout of the data files and the command lines of exp/exp117_agency/e2e_run.sh. It imports neither
the package nor any exp117 script: standard library and numpy only.

Usage: python second_route.py RECORD_ROOT > out.json        (progress goes to stderr)

It rebuilds exp117's runs nlcd2016_l2_alt, nlcd2016_l2_primary, nlcd2011_l2_alt, nlcd2016_8_alt and gfc2020, computes
every overall accuracy, difference, per-class user's and producer's accuracy and reference share (estimate, valid
interval, normal-theory interval, alpha = 0.05) and sets each beside RECORD_ROOT/exp/out/exp117_e2e/<run>.json.

How the two hard parts are computed
- The exact average over orders. Counting every distinct order of a stratum's labels once is the same as drawing the
  labels in a uniformly random permutation (each distinct order arises from the same number of permutations), so the
  average of exp(log-term) is an expectation over draws without replacement from an urn holding the labels. Step t of
  the log-term depends on the past only through S_t, and S_t only through how many labels of each value have been
  drawn, so a forward recursion over those counts carries the expectation exactly: the state is (j, k), the numbers of
  the two less frequent values drawn so far (the most frequent value's count is t - j - k), and each state's mass is
  multiplied by the draw probability (that value's remaining labels over all remaining labels) and by the step's
  factor (1 + l_t (X - centre_t)) exp(l_t (centre_t + S_t / (N - t))). The 12 values of c run together; the mass is
  rescaled every step and the scale kept in logs. A stratum whose labels share one value has one order and is summed
  in closed form. Work is (labels) x (count of the 2nd value + 1) x (count of the 3rd value + 1) x 12. A self-test
  checks the recursion against brute-force enumeration of the distinct orders.
- The hypergeometric interval. P(X >= k | K) and P(X <= k | K) are sums of the probability mass function, each term
  built in logs from factors that are all ratios at most one ((K - i)/(N - i), (N - K - j)/N, log1p(-m/N)) plus
  log C(n, x) from the exact integer, so stratum sizes of billions lose no precision; the smallest and the largest K
  are found by bisection over the integers 0..N (the two tails are monotone in K). A self-test checks the tails and
  the intervals against exact rational arithmetic on small populations.
"""
import csv
import itertools
import json
import math
import os
import sys
import time
from fractions import Fraction

import numpy as np

ALPHA = 0.05
Z = 1.959963984540054
NK = 12
UNK = -(10 ** 9) - 1      # a code that could not be read
NO_ALT = -(10 ** 9) - 2   # no alternate label

# e2e_run.sh: the 8-class crosswalk and the pure-stratum declarations (stratum=class), first map only
T8 = {21: 1, 22: 1, 23: 1, 24: 1, 81: 2, 82: 2, 52: 3, 71: 3, 41: 4, 42: 4, 43: 4, 11: 5, 90: 6, 95: 6, 12: 7, 31: 8}
PURE = "1=11 2=21 3=22 4=23 5=24 6=31 7=41 8=42 9=43 10=52 11=71 12=81 13=82 14=90 15=95"
PURE8 = "1=5 2=1 3=1 4=1 5=1 6=8 7=4 8=4 9=4 10=3 11=3 12=2 13=2 14=6 15=6"

FAMILIES = ("linear", "difference", "box", "pure-stratum rule", "normal theory")
BARS = {"linear": 1e-9, "difference": 1e-9, "box": 1e-9, "pure-stratum rule": 1e-6, "normal theory": 1e-9}

READINGS = [
    "W_h for sizes from weights is N_h / sum of N_h, with N_h the rounded sum of the stratum's weights.",
    "The average over every distinct order is the expectation over a uniformly random permutation of the labels "
    "(each distinct order arises from the same number of permutations); computed exactly by a recursion over counts.",
    "Sums over strata (V, A_k, W_sum) run over sampled strata only; W_u is the share of the strata file's strata "
    "that hold no label.",
    "The lower end is located by bisection to a bracket of 1e-15 and the bracket's lower point (where the condition "
    "holds) is returned.",
    "A difference's estimate and normal theory are those of d = right_a - right_b as a linear quantity "
    "(sum W_h dbar_h); this equals 2 theta_x - 1 when W_u = 0.",
    "A ratio with an empty denominator (no label mapped as, or referenced as, the class) has no estimate and no "
    "normal-theory interval; its box takes 0 for an empty lower denominator and 1 for an empty upper one.",
    "The box is clipped to [0, 1] after the 0/1 rule for empty denominators.",
    "Pure-stratum rule: stratum o's k is its labels that are right (map equals reference or alternate); the mixed "
    "strata are all sampled strata declared pure for no class (including those where the class has no label); "
    "strata declared pure for another class are left out of the sums; the rule needs exactly one declared stratum "
    "for the class, the first map, finite whole-unit sizes and no unsampled stratum. Its estimate and normal theory "
    "are the ratio estimator's.",
    "Hypergeometric tails are compared with a strict '>' at (1 - conf) / 2, conf computed as the specification writes it.",
    "Classes are the union over maps of each map's classes and its reference classes by the rule, plus every class "
    "key of exp117's output.",
    "A point with an unreadable code in the reference or in any map is dropped from the whole run (counted).",
    "nlcd2016_8_alt: an alternate code outside the crosswalk counts as no alternate.",
]

_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:8.1f} s] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- small helpers

def skey(s):
    s = (s or "").strip()
    try:
        f = float(s)
        if math.isfinite(f) and f == int(f):
            return str(int(f))
    except ValueError:
        pass
    return s


def sortkey(k):
    try:
        return (0, int(k), k)
    except ValueError:
        return (1, 0, k)


def code(s):
    s = (s or "").strip()
    if s == "":
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    if not math.isfinite(f):
        return None
    return int(f)


def parse_pure(spec):
    out = {}
    for item in spec.split():
        k, v = item.split("=")
        out[skey(k)] = int(v)
    return out


def fnum(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def ends(v):
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return fnum(v[0]), fnum(v[1])
    return None, None


def ad(a, b):
    """Absolute difference; (value, one side missing)."""
    if a is None and b is None:
        return 0.0, False
    if a is None or b is None:
        return None, True
    return abs(a - b), False


# ---------------------------------------------------------------- the stratum's value: exact average over orders

def stratum_log_average(vals, cnts, N, lam0):
    """log of the average, over every distinct order of the labels, of exp(log-term); one value per lambda0.

    vals: the distinct label values; cnts: how many labels take each; N: the stratum's units (None for a point
    frame); lam0: array of lambda0 = c W_h / n_h, one per c.
    """
    lam0 = np.asarray(lam0, dtype=float)
    nk = lam0.shape[0]
    n = int(sum(cnts))
    finite = N is not None
    if len(vals) == 1:   # one distinct value: a single order
        v = float(vals[0])
        t = np.arange(n, dtype=float)
        if finite:
            l = lam0[:, None] * ((N - t) / N)[None, :]
        else:
            l = np.repeat(lam0[:, None], n, axis=1)
        S = t * v
        centre = np.minimum(((0.5 + S) / (t + 1.0))[None, :], 0.9 / l)
        term = np.log1p(l * (v - centre)) + l * centre
        if finite:
            term = term + l * (S / (N - t))[None, :]
        return term.sum(axis=1)
    if len(vals) > 3:
        raise ValueError(f"more than three distinct values in a stratum: {vals}")
    order = sorted(range(len(vals)), key=lambda q: (-int(cnts[q]), q))
    v = [float(vals[q]) for q in order] + [0.0] * (3 - len(vals))
    c = [int(cnts[q]) for q in order] + [0] * (3 - len(vals))
    va, vb, vc = v
    na, nb, nc = c
    J, KC = nb + 1, nc + 1
    jj = np.arange(J, dtype=float)[:, None]
    kk = np.arange(KC, dtype=float)[None, :]
    SB = jj * vb + kk * vc
    pb_num = nb - jj
    pc_num = nc - kk
    M = np.zeros((nk, J, KC))
    M[:, 0, 0] = 1.0
    logscale = np.zeros(nk)
    for t in range(n):
        i = t - jj - kk                     # labels of the most frequent value drawn so far
        S = i * va + SB
        rem = float(n - t)
        pa = np.clip((na - i) / rem, 0.0, None)
        pb = np.clip(pb_num / rem, 0.0, None)
        pc = np.clip(pc_num / rem, 0.0, None)
        l = lam0 * ((N - t) / N) if finite else lam0
        l3 = l[:, None, None]
        centre = np.minimum(((0.5 + S) / (t + 1.0))[None, :, :], 0.9 / l3)
        if finite:
            g = np.exp(l3 * (centre + (S / (N - t))[None, :, :]))
        else:
            g = np.exp(l3 * centre)
        Mg = M * g
        new = Mg * (pa[None, :, :] * (1.0 + l3 * (va - centre)))
        if J > 1:
            new[:, 1:, :] += (Mg * (pb[None, :, :] * (1.0 + l3 * (vb - centre))))[:, :-1, :]
        if KC > 1:
            new[:, :, 1:] += (Mg * (pc[None, :, :] * (1.0 + l3 * (vc - centre))))[:, :, :-1]
        mx = new.max(axis=(1, 2))
        new /= mx[:, None, None]
        logscale += np.log(mx)
        M = new
    return np.log(M[:, nb, nc]) + logscale


def brute_log_average(labels, N, lam0):
    """The same by enumerating every distinct order (self-test only)."""
    orders = sorted(set(itertools.permutations(labels)))
    out = []
    for lam in lam0:
        terms = []
        for o in orders:
            S = 0.0
            lt = 0.0
            for t, X in enumerate(o):
                l = lam * ((N - t) / N) if N is not None else lam
                centre = min((0.5 + S) / (t + 1), 0.9 / l)
                lt += math.log(1.0 + l * (X - centre)) + l * (centre + (S / (N - t) if N is not None else 0.0))
                S += X
            terms.append(math.exp(lt))
        out.append(math.log(math.fsum(terms) / len(orders)))
    return np.array(out)


def solve_lower(A, c, thr, wsum):
    """Largest theta0 in [0, wsum] with ln(mean_k exp(A_k - c_k theta0)) >= thr."""
    def g(th):
        e = A - c * th
        mx = float(e.max())
        return mx + math.log(float(np.exp(e - mx).sum()) / NK)

    if g(0.0) < thr:
        return 0.0
    if g(wsum) >= thr:
        return wsum
    lo, hi = 0.0, wsum
    for _ in range(300):
        if hi - lo <= 1e-15:
            break
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if g(mid) >= thr:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------- hypergeometric

_LOGC = {}
_TSUF = {}
_HINT = {}


def _logcomb(n):
    r = _LOGC.get(n)
    if r is None:
        r = np.array([math.log(math.comb(n, x)) for x in range(n + 1)])
        _LOGC[n] = r
    return r


def _tsuffix(N, n):
    """T[x] = sum_{m=x}^{n-1} log((N - m) / N), T[n] = 0."""
    key = (N, n)
    r = _TSUF.get(key)
    if r is None:
        r = np.zeros(n + 1)
        if n > 0:
            m = np.arange(n, dtype=float)
            s = np.log1p(-m / float(N))
            r[:n] = np.cumsum(s[::-1])[::-1]
        _TSUF[key] = r
    return r


def hyper_pmf(N, K, n, x_lo, x_hi):
    """P(X = x), x = x_lo..x_hi inside the support, X the marked count among n drawn from N holding K marked."""
    xs = np.arange(x_lo, x_hi + 1)
    a = np.zeros(x_hi + 1)
    if x_hi > 0:
        i = np.arange(x_hi, dtype=float)
        a[1:] = np.cumsum(np.log((K - i) / (N - i)))
    jn = n - x_lo
    Q = np.zeros(jn + 1)
    if jn > 0:
        j = np.arange(jn, dtype=float)
        Q[1:] = np.cumsum(np.log((N - K - j) / float(N)))
    T = _tsuffix(N, n)
    logp = _logcomb(n)[xs] + a[xs] + Q[n - xs] - T[xs]
    return np.exp(logp)


def p_ge(k, n, N, K):
    lo_s, hi_s = max(0, n - (N - K)), min(n, K)
    x_lo = max(k, lo_s)
    if x_lo > hi_s:
        return 0.0
    return math.fsum(hyper_pmf(N, K, n, x_lo, hi_s).tolist())


def p_le(k, n, N, K):
    lo_s, hi_s = max(0, n - (N - K)), min(n, K)
    x_hi = min(k, hi_s)
    if x_hi < lo_s:
        return 0.0
    return math.fsum(hyper_pmf(N, K, n, lo_s, x_hi).tolist())


def hyper_interval(k, n, N, conf):
    """Exact two-sided interval for the marked share: smallest K with P(X >= k | K) > (1 - conf)/2, largest K with
    P(X <= k | K) > (1 - conf)/2, divided by N, widened to contain k/n."""
    key = (k, n, N, conf)
    r = _HINT.get(key)
    if r is not None:
        return r
    a = (1.0 - conf) / 2.0
    lo, hi = -1, N          # P(X >= k | hi) > a holds (K = N gives X = n); lo is a sentinel where it fails
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if p_ge(k, n, N, mid) > a:
            hi = mid
        else:
            lo = mid
    K_lo = hi
    lo, hi = 0, N + 1       # P(X <= k | lo) > a holds (K = 0 gives X = 0); hi is a sentinel where it fails
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if p_le(k, n, N, mid) > a:
            lo = mid
        else:
            hi = mid
    K_hi = lo
    share = k / n
    r = (min(K_lo / N, share), max(K_hi / N, share))
    _HINT[key] = r
    return r


def exact_hyper_interval(k, n, N, conf):
    """The same by exact rational arithmetic and a scan over K (self-test only)."""
    a = (1.0 - conf) / 2.0
    tot = math.comb(N, n)

    def ge(K):
        return Fraction(sum(math.comb(K, x) * math.comb(N - K, n - x) for x in range(k, n + 1)), tot)

    def le(K):
        return Fraction(sum(math.comb(K, x) * math.comb(N - K, n - x) for x in range(0, k + 1)), tot)

    K_lo = min(K for K in range(N + 1) if ge(K) > a)
    K_hi = max(K for K in range(N + 1) if le(K) > a)
    share = k / n
    return min(K_lo / N, share), max(K_hi / N, share)


# ---------------------------------------------------------------- a run: design, points, intervals

class Stratum:
    def __init__(self, key, idx, W, N, Nlo):
        self.key = key
        self.idx = idx
        self.n = int(len(idx))
        self.W = float(W)
        self.N = N
        self.Nlo = Nlo


class Run:
    def __init__(self, name, skeys, ref, alt, maps, design, pure, notes):
        self.name = name
        self.notes = dict(notes)
        ref = np.asarray(ref, dtype=np.int64)
        alt = np.asarray(alt, dtype=np.int64)
        maps = {m: np.asarray(v, dtype=np.int64) for m, v in maps.items()}
        keep = ref != UNK
        for v in maps.values():
            keep &= (v != UNK)
        self.unknown = {"reference": int((ref == UNK).sum())}
        for m, v in maps.items():
            self.unknown[m] = int((v == UNK).sum())
        self.dropped = int((~keep).sum())
        kept = np.flatnonzero(keep)
        skeys = [skeys[q] for q in kept]
        self.ref = ref[keep]
        self.alt = alt[keep]
        self.maps = {m: v[keep] for m, v in maps.items()}
        self.map_names = list(maps)
        self.npoints = len(skeys)
        missing = sorted(set(skeys) - set(design), key=sortkey)
        if missing:
            raise SystemExit(f"{name}: strata of points absent from the design: {missing[:20]}")
        sampled = sorted(set(skeys), key=sortkey)
        pos = {k: q for q, k in enumerate(sampled)}
        sidx = np.array([pos[s] for s in skeys], dtype=np.int64)
        self.strata = [Stratum(k, np.flatnonzero(sidx == q), design[k]["W"], design[k]["N"], design[k]["Nlo"])
                       for q, k in enumerate(sampled)]
        self.unsampled = sorted((k for k in design if k not in pos), key=sortkey)
        self.Wu = math.fsum(design[k]["W"] for k in self.unsampled)
        self.Wsum = math.fsum(st.W for st in self.strata)
        self.V = math.fsum(st.W ** 2 / (4.0 * st.n) for st in self.strata)
        self.pure = dict(pure)
        has = self.alt != NO_ALT
        self.right = {}
        self.refclass = {}
        for m, v in self.maps.items():
            on_alt = has & (v == self.alt)
            self.right[m] = (v == self.ref) | on_alt
            self.refclass[m] = np.where(on_alt, v, self.ref)
        self._c = {}
        self._corr = {}
        self._val = {}
        self.dp_calls = 0

    # -- the betting interval
    def cgrid(self, level):
        r = self._c.get(level)
        if r is None:
            c_lo = math.sqrt(2.0 * math.log(2.0 / level) / self.V)
            r = np.geomspace(c_lo, 10.0 * c_lo, NK)
            self._c[level] = r
        return r

    def correction(self, level):
        r = self._corr.get(level)
        if r is None:
            c = self.cgrid(level)
            r = np.zeros(NK)
            for st in self.strata:
                if st.N is None and st.Nlo is not None:
                    r = r + c * st.W * (st.n - 1) / (2.0 * (st.Nlo - st.n + 1))
            self._corr[level] = r
        return r

    def stratum_value(self, h, level, vals, cnts):
        key = (h, level, vals, cnts)
        r = self._val.get(key)
        if r is None:
            st = self.strata[h]
            lam0 = self.cgrid(level) * st.W / st.n
            r = stratum_log_average(vals, cnts, st.N, lam0)
            self._val[key] = r
            self.dp_calls += 1
        return r

    def lower_end(self, x, level):
        A = np.zeros(NK)
        for h, st in enumerate(self.strata):
            u, cn = np.unique(x[st.idx], return_counts=True)
            A = A + self.stratum_value(h, level, tuple(float(q) for q in u), tuple(int(q) for q in cn))
        A = A - self.correction(level)
        return solve_lower(A, self.cgrid(level), math.log(2.0 / level), self.Wsum)

    def linear(self, x, level):
        lo = self.lower_end(x, level)
        lo_c = self.lower_end(1.0 - x, level)
        hi = min(1.0, self.Wsum - lo_c + self.Wu)
        return lo, hi

    def box(self, y, w, alpha):
        Ly, Uy = self.linear(y, alpha / 2.0)
        Lw, Uw = self.linear(w, alpha / 2.0)
        lo = Ly / (Ly + Uw) if (Ly + Uw) > 0.0 else 0.0
        hi = Uy / (Uy + Lw) if (Uy + Lw) > 0.0 else 1.0
        return min(1.0, max(0.0, lo)), min(1.0, max(0.0, hi))

    # -- normal theory (Stehman 2014)
    def _fpc(self, st):
        return 1.0 if st.N is None else 1.0 - st.n / st.N

    def nt_linear(self, x):
        est = math.fsum(st.W * float(x[st.idx].mean()) for st in self.strata)
        var = 0.0
        for st in self.strata:
            if st.n > 1:
                var += st.W ** 2 * self._fpc(st) * float(x[st.idx].var(ddof=1)) / st.n
        se = math.sqrt(var)
        return est, (est - Z * se, est + Z * se)

    def nt_ratio(self, y, w):
        z = y + w
        Y = math.fsum(st.W * float(y[st.idx].mean()) for st in self.strata)
        Zt = math.fsum(st.W * float(z[st.idx].mean()) for st in self.strata)
        if Zt <= 0.0:
            return None, None
        R = Y / Zt
        u = y - R * z
        var = 0.0
        for st in self.strata:
            if st.n > 1:
                var += st.W ** 2 * self._fpc(st) * float(u[st.idx].var(ddof=1)) / st.n
        se = math.sqrt(var) / Zt
        return R, (R - Z * se, R + Z * se)

    # -- the pure-stratum rule
    def pure_rule(self, m, cls, alpha):
        if m != self.map_names[0]:
            return None, "not the first map"
        declared = sorted((s for s, c in self.pure.items() if c == cls), key=sortkey)
        if len(declared) != 1:
            return None, f"{len(declared)} declared pure strata"
        if self.unsampled:
            return None, "a stratum is unsampled"
        if any(st.N is None for st in self.strata):
            return None, "a stratum's size is not a whole number of units"
        okey = declared[0]
        so = next((st for st in self.strata if st.key == okey), None)
        if so is None:
            return None, "the declared stratum holds no label"
        v = self.maps[m]
        r = self.right[m]
        mapped = v == cls
        mr = mapped & r
        mw = mapped & ~r
        mixed = [st for st in self.strata if st.key not in self.pure]
        Mn = sum(1 for st in mixed if st.n < st.N)
        gamma = alpha / 10.0 if Mn > 0 else 0.0
        k_o = int(r[so.idx].sum())
        p_lo, p_hi = hyper_interval(k_o, so.n, so.N, 1.0 - (alpha - gamma))
        a_lo, a_hi, b_lo, b_hi = [], [], [], []
        mixed_info = []
        for st in mixed:
            ka = int(mr[st.idx].sum())
            kb = int(mw[st.idx].sum())
            if st.n >= st.N:
                al = ah = ka / st.n
                bl = bh = kb / st.n
            else:
                conf = 1.0 - gamma / (2.0 * Mn)
                al, ah = hyper_interval(ka, st.n, st.N, conf)
                bl, bh = hyper_interval(kb, st.n, st.N, conf)
            a_lo.append(st.W * al)
            a_hi.append(st.W * ah)
            b_lo.append(st.W * bl)
            b_hi.append(st.W * bh)
            mixed_info.append({"stratum": st.key, "n": st.n, "N": st.N, "right": ka, "wrong": kb,
                               "right_interval": [al, ah], "wrong_interval": [bl, bh]})
        Wo = so.W
        sal, sah, sbl, sbh = math.fsum(a_lo), math.fsum(a_hi), math.fsum(b_lo), math.fsum(b_hi)
        lower = (Wo * p_lo + sal) / (Wo + sal + sbh)
        upper = min(1.0, (Wo * p_hi + sah) / (Wo + sah + sbl))
        other_pure = [st for st in self.strata if st.key in self.pure and st.key != okey]
        info = {
            "stratum": okey, "n": so.n, "N": so.N, "right": k_o,
            "right_and_mapped_as_class": int(mr[so.idx].sum()),
            "labels_in_stratum_not_mapped_as_class": int((~mapped[so.idx]).sum()),
            "labels_mapped_as_class_in_strata_pure_for_another_class": int(sum(int(mapped[st.idx].sum())
                                                                              for st in other_pure)),
            "M": Mn, "gamma": gamma, "p_interval": [p_lo, p_hi], "mixed": mixed_info,
        }
        return (lower, upper), info

    def classes(self):
        s = set()
        for m in self.map_names:
            s.update(int(q) for q in np.unique(self.maps[m]))
            s.update(int(q) for q in np.unique(self.refclass[m]))
        return s

    def design_summary(self, alpha):
        return {
            "points": self.npoints, "dropped_points": self.dropped, "unknown": self.unknown,
            "strata_sampled": len(self.strata), "strata_unsampled": len(self.unsampled),
            "unsampled": self.unsampled[:50], "W_u": self.Wu, "W_sum": self.Wsum, "V": self.V,
            "c_grid_alpha": [float(q) for q in self.cgrid(alpha)],
            "c_grid_alpha_half": [float(q) for q in self.cgrid(alpha / 2.0)],
            "labels_per_stratum": {st.key: st.n for st in self.strata},
            "W": {st.key: st.W for st in self.strata},
            "N": {st.key: st.N for st in self.strata},
            "N_lo": {st.key: st.Nlo for st in self.strata},
            "notes": self.notes,
        }


# ---------------------------------------------------------------- the runs of e2e_run.sh

def build_nlcd(rows, year, name, legend, use_alt, pure_spec):
    sel = [r for r in rows if r["set"] == "B" and r["year"] == str(year)]
    if not sel:
        raise SystemExit(f"{name}: no set B rows for {year}")
    skeys = [skey(r["stratum"]) for r in sel]
    wts = {}
    for s, r in zip(skeys, sel):
        wts.setdefault(s, []).append(float(r["weight"]))
    sums = {s: math.fsum(w) for s, w in wts.items()}
    sizes = {s: int(round(v)) for s, v in sums.items()}
    total = sum(sizes.values())
    design = {s: {"W": sizes[s] / total, "N": sizes[s], "Nlo": None} for s in sizes}
    ref, alt = [], []
    maps = {"nlcd": []} if legend == "l2" else {"nlcd8": [], "lcmap": []}
    alt_outside = 0
    for r in sel:
        rc, ac, mc = code(r["ref_nlcd"]), code(r["ref_nlcd_alternate"]), code(r["nlcd"])
        if legend == "l2":
            ref.append(UNK if rc is None else rc)
            alt.append(ac if (use_alt and ac is not None) else NO_ALT)
            maps["nlcd"].append(UNK if mc is None else mc)
        else:
            ref.append(T8.get(rc, UNK))
            if use_alt and ac is not None and ac in T8:
                alt.append(T8[ac])
            else:
                if use_alt and ac is not None:
                    alt_outside += 1
                alt.append(NO_ALT)
            maps["nlcd8"].append(T8.get(mc, UNK))
            lc = code(r["lcmap"])
            maps["lcmap"].append(UNK if lc is None else lc)
    notes = {
        "year": year, "legend": legend, "alternate": use_alt, "sizes": "rounded sums of the weights",
        "largest |sum of weights - size|": max(abs(sums[s] - sizes[s]) for s in sizes),
        "alternates outside the crosswalk": alt_outside,
        "pure": pure_spec,
    }
    return Run(name, skeys, ref, alt, maps, design, parse_pure(pure_spec), notes)


def build_gfc(root):
    with open(os.path.join(root, "exp/out/exp94_gfc_points.csv"), newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    areas = {}
    with open(os.path.join(root, "exp/out/exp117_gfc2020_strata_size.csv"), newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                a = float(row[1])
            except ValueError:
                continue   # the header
            k = skey(row[0])
            if k in areas:
                raise SystemExit(f"gfc2020: stratum {k} twice in the strata file")
            areas[k] = a
    total = math.fsum(areas.values())
    unit_area = 1.0
    design = {s: {"W": a / total, "N": None, "Nlo": int(math.floor(a / unit_area))} for s, a in areas.items()}
    skeys = [skey(r["stratum"]) for r in rows]
    ref = [UNK if code(r["ref_forest"]) is None else code(r["ref_forest"]) for r in rows]
    maps = {
        "gfc_v2": [UNK if code(r["gfc_v2"]) is None else code(r["gfc_v2"]) for r in rows],
        "worldcover_forest": [UNK if code(r["worldcover_forest"]) is None else code(r["worldcover_forest"])
                              for r in rows],
    }
    alt = [NO_ALT] * len(rows)
    notes = {"sizes": "areas from the strata file, a point frame with unit area 1 (N_lo = floor(area))",
             "strata_in_file": len(areas), "total_area": total}
    return Run("gfc2020", skeys, ref, alt, maps, design, {}, notes)


# ---------------------------------------------------------------- every quantity of a run

def compute_run(run, alpha, extra_classes):
    res = {}
    pure_diag = {}

    def put(key, fam, est, iv, nt, method, extra=None):
        e = {"estimate": fnum(est),
             "interval": None if iv is None else [fnum(iv[0]), fnum(iv[1])],
             "normal_theory": None if nt is None else [fnum(nt[0]), fnum(nt[1])],
             "method": method}
        if extra:
            e.update(extra)
        res[key] = (fam, e)

    for m in run.map_names:
        x = run.right[m].astype(float)
        est, nt = run.nt_linear(x)
        put(("overall_accuracy", m), "linear", est, run.linear(x, alpha), nt, "betting")
        log(f"{run.name}: overall accuracy of {m} done")
    if len(run.map_names) >= 2:
        a, b = run.map_names[0], run.map_names[1]
        d = run.right[a].astype(float) - run.right[b].astype(float)
        x = (1.0 + d) / 2.0
        est, nt = run.nt_linear(d)
        L, U = run.linear(x, alpha)
        ex, _ = run.nt_linear(x)
        put(("differences", f"{a} minus {b}"), "difference", est, (2.0 * L - 1.0, 2.0 * U - 1.0), nt, "betting",
            {"estimate_as_2theta_minus_1": 2.0 * ex - 1.0, "disagreeing_points": int((d != 0).sum())})
        log(f"{run.name}: difference done")
    classes = sorted(run.classes() | set(extra_classes))
    for m in run.map_names:
        v = run.maps[m]
        r = run.right[m]
        rc = run.refclass[m]
        for cls in classes:
            c = str(cls)
            mapped = v == cls
            refd = rc == cls
            yu = (mapped & r).astype(float)
            wu = (mapped & ~r).astype(float)
            yp = (refd & r).astype(float)
            wp = (refd & ~r).astype(float)
            est, nt = run.nt_ratio(yu, wu)
            bx = run.box(yu, wu, alpha)
            piv, pinfo = run.pure_rule(m, cls, alpha)
            if piv is not None:
                put(("per_class", m, c, "user"), "pure-stratum rule", est, piv, nt, "pure-stratum rule",
                    {"box_interval": [bx[0], bx[1]], "pure_stratum": pinfo["stratum"],
                     "labels_mapped": int(mapped.sum())})
                pure_diag[f"{m}/{c}"] = pinfo
            else:
                put(("per_class", m, c, "user"), "box", est, bx, nt, "box of two betting intervals",
                    {"pure_rule_not_used": pinfo, "labels_mapped": int(mapped.sum())})
            est, nt = run.nt_ratio(yp, wp)
            put(("per_class", m, c, "producer"), "box", est, run.box(yp, wp, alpha), nt,
                "box of two betting intervals", {"labels_reference": int(refd.sum())})
            s = refd.astype(float)
            est, nt = run.nt_linear(s)
            put(("per_class", m, c, "reference_share"), "linear", est, run.linear(s, alpha), nt, "betting")
        log(f"{run.name}: {len(classes)} classes of {m} done (stratum recursions so far {run.dp_calls})")
    return res, pure_diag


# ---------------------------------------------------------------- comparison with exp117's output

class Agg:
    def __init__(self):
        self.best = {}

    def add(self, fam, val, where):
        if val is None:
            return
        b = self.best.get(fam)
        if b is None or val > b["max_abs_diff"]:
            self.best[fam] = {"max_abs_diff": val, "where": where}

    def report(self):
        out = {}
        for fam in FAMILIES + ("estimate",):
            if fam in self.best:
                e = dict(self.best[fam])
                if fam in BARS:
                    e["bar"] = BARS[fam]
                    e["within_bar"] = e["max_abs_diff"] <= BARS[fam]
                out[fam] = e
        intervals = [self.best[f] for f in FAMILIES if f in self.best]
        if intervals:
            out["overall"] = max(intervals, key=lambda e: e["max_abs_diff"])
        every = [self.best[f] for f in FAMILIES + ("estimate",) if f in self.best]
        if every:
            out["overall_including_estimates"] = max(every, key=lambda e: e["max_abs_diff"])
        return out


def lookup(theirs, key):
    d = theirs
    for p in key:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
        if d is None:
            return None
    return d if isinstance(d, dict) else None


def their_keys(theirs):
    keys = set()
    for m in (theirs.get("overall_accuracy") or {}):
        keys.add(("overall_accuracy", m))
    for k in (theirs.get("differences") or {}):
        keys.add(("differences", k))
    for m, cd in (theirs.get("per_class") or {}).items():
        if not isinstance(cd, dict):
            continue
        for c, qd in cd.items():
            if not isinstance(qd, dict):
                continue
            for q in ("user", "producer", "reference_share"):
                if isinstance(qd.get(q), dict):
                    keys.add(("per_class", m, c, q))
    return keys


def their_classes(theirs):
    out = set()
    for cd in (theirs.get("per_class") or {}).values():
        if isinstance(cd, dict):
            for c in cd:
                k = code(str(c))
                if k is not None:
                    out.add(k)
    return out


def compare_run(name, mine, theirs, agg_run, agg_all):
    rows, missing, methods = [], [], []
    for key in sorted(mine):
        fam, me = mine[key]
        where = f"{name}: {'/'.join(key)}"
        th = lookup(theirs, key)
        if th is None:
            missing.append(f"{where}: absent from exp117's output")
            rows.append({"quantity": "/".join(key), "family": fam, "mine": me, "theirs": None, "abs_diff": None})
            continue
        t_est = fnum(th.get("estimate"))
        t_iv = ends(th.get("interval"))
        t_nt = ends(th.get("normal_theory"))
        t_method = th.get("method")
        m_iv = ends(me["interval"])
        m_nt = ends(me["normal_theory"])
        diffs = {}
        for label, a, b, f in (("estimate", me["estimate"], t_est, "estimate"),
                               ("interval_lo", m_iv[0], t_iv[0], fam),
                               ("interval_hi", m_iv[1], t_iv[1], fam),
                               ("normal_theory_lo", m_nt[0], t_nt[0], "normal theory"),
                               ("normal_theory_hi", m_nt[1], t_nt[1], "normal theory")):
            val, miss = ad(a, b)
            diffs[label] = val
            if miss:
                missing.append(f"{where} {label}: mine {a}, theirs {b}")
            agg_run.add(f, val, f"{where} {label}")
            agg_all.add(f, val, f"{where} {label}")
        if t_method is not None and t_method != me["method"]:
            methods.append(f"{where}: mine '{me['method']}', theirs '{t_method}'")
        if key[-1] == "user" and th.get("pure_stratum") is not None and str(th.get("pure_stratum")) != str(
                me.get("pure_stratum")):
            methods.append(f"{where}: pure stratum mine {me.get('pure_stratum')}, theirs {th.get('pure_stratum')}")
        rows.append({"quantity": "/".join(key), "family": fam, "mine": me,
                     "theirs": {"estimate": t_est, "interval": list(t_iv), "normal_theory": list(t_nt),
                                "method": t_method},
                     "abs_diff": diffs})
    only_theirs = sorted("/".join(k) for k in their_keys(theirs) - set(mine))
    return rows, missing, methods, only_theirs


# ---------------------------------------------------------------- self-test of the two hard parts

def selftest():
    rep = {}
    cases = [
        ((0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5), 9),
        ((0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5), None),
        ((0.0, 1.0, 1.0, 0.0, 0.0, 1.0), 6),
        ((1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0), 1000),
        ((0.5, 0.5, 0.5, 0.5), 7),
        ((1.0, 1.0, 1.0), None),
        ((0.0, 1.0, 0.5, 0.0, 1.0, 1.0, 1.0, 0.0), None),
        ((0.5, 0.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5), 11),
    ]
    lam0 = np.array([0.05, 0.4, 1.3, 3.0])
    worst = 0.0
    for labels, N in cases:
        u, cn = np.unique(np.array(labels), return_counts=True)
        dp = stratum_log_average(tuple(float(q) for q in u), tuple(int(q) for q in cn), N, lam0)
        bf = brute_log_average(labels, N, lam0)
        worst = max(worst, float(np.max(np.abs(dp - bf))))
    rep["order_average_vs_enumeration_max_abs_diff"] = worst
    worst_p = 0.0
    bad = []
    for N, n in ((40, 12), (25, 25), (60, 7), (31, 1)):
        tot = math.comb(N, n)
        for K in range(0, N + 1, max(1, N // 13)):
            for k in range(0, n + 1):
                ge = Fraction(sum(math.comb(K, x) * math.comb(N - K, n - x) for x in range(k, n + 1)), tot)
                le = Fraction(sum(math.comb(K, x) * math.comb(N - K, n - x) for x in range(0, k + 1)), tot)
                worst_p = max(worst_p, abs(p_ge(k, n, N, K) - float(ge)), abs(p_le(k, n, N, K) - float(le)))
        for k in range(0, n + 1):
            for conf in (0.95, 0.955, 1.0 - 0.005 / 12.0):
                a = hyper_interval(k, n, N, conf)
                b = exact_hyper_interval(k, n, N, conf)
                if a != b:
                    bad.append({"k": k, "n": n, "N": N, "conf": conf, "mine": list(a), "exact": list(b)})
    rep["hypergeometric_tail_vs_exact_max_abs_diff"] = worst_p
    rep["hypergeometric_intervals_differing_from_exact"] = bad
    # a large population: the two tails at a K near the interval end must straddle the level
    N, n, k = 2_000_000_000, 250, 241
    lo, hi = hyper_interval(k, n, N, 0.955)
    K_lo = int(round(lo * N))
    rep["large_N_check"] = {"N": N, "n": n, "k": k, "interval": [lo, hi],
                            "P(X>=k|K_lo-1)": p_ge(k, n, N, K_lo - 1), "P(X>=k|K_lo)": p_ge(k, n, N, K_lo),
                            "level": (1.0 - 0.955) / 2.0}
    _HINT.clear()
    return rep


# ---------------------------------------------------------------- main

def main():
    if len(sys.argv) != 2:
        print("usage: python second_route.py RECORD_ROOT > out.json", file=sys.stderr)
        sys.exit(2)
    root = sys.argv[1]
    out = {
        "what": "exp118 arm A: every interval of exp117's runs recomputed by a second route written from the "
                "specification alone, set beside exp/out/exp117_e2e/<run>.json",
        "alpha": ALPHA, "z": Z, "numpy": np.__version__, "python": sys.version.split()[0],
        "readings": READINGS,
    }
    log("self-test of the order average and the hypergeometric interval")
    out["selftest"] = st = selftest()
    log(f"self-test: order average vs enumeration {st['order_average_vs_enumeration_max_abs_diff']:.3g}; "
        f"hypergeometric tails vs exact {st['hypergeometric_tail_vs_exact_max_abs_diff']:.3g}; "
        f"intervals differing from exact {len(st['hypergeometric_intervals_differing_from_exact'])}")
    with open(os.path.join(root, "exp/out/exp94_nlcd_points.csv"), newline="", encoding="utf-8-sig") as f:
        nl_rows = list(csv.DictReader(f))
    specs = [
        ("nlcd2016_l2_alt", lambda: build_nlcd(nl_rows, 2016, "nlcd2016_l2_alt", "l2", True, PURE)),
        ("nlcd2016_l2_primary", lambda: build_nlcd(nl_rows, 2016, "nlcd2016_l2_primary", "l2", False, PURE)),
        ("nlcd2011_l2_alt", lambda: build_nlcd(nl_rows, 2011, "nlcd2011_l2_alt", "l2", True, PURE)),
        ("nlcd2016_8_alt", lambda: build_nlcd(nl_rows, 2016, "nlcd2016_8_alt", "8", True, PURE8)),
        ("gfc2020", lambda: build_gfc(root)),
    ]
    agg_all = Agg()
    out["runs"] = {}
    total_missing, total_methods, total_only_theirs = 0, 0, 0
    for name, make in specs:
        t0 = time.time()
        run = make()
        log(f"{name}: {run.npoints} points, {len(run.strata)} sampled strata, {len(run.unsampled)} unsampled "
            f"(W_u {run.Wu:.3g}), maps {run.map_names}, dropped {run.dropped}")
        tpath = os.path.join(root, "exp/out/exp117_e2e", f"{name}.json")
        theirs = {}
        if os.path.exists(tpath):
            with open(tpath) as f:
                theirs = json.load(f)
        else:
            log(f"{name}: {tpath} missing")
        mine, pure_diag = compute_run(run, ALPHA, their_classes(theirs))
        agg_run = Agg()
        rows, missing, methods, only_theirs = compare_run(name, mine, theirs, agg_run, agg_all)
        total_missing += len(missing)
        total_methods += len(methods)
        total_only_theirs += len(only_theirs)
        rep = agg_run.report()
        out["runs"][name] = {
            "design": run.design_summary(ALPHA),
            "their_design": theirs.get("design"),
            "their_unknown": theirs.get("unknown"),
            "max_by_family": rep,
            "one_side_missing": missing,
            "method_disagreements": methods,
            "only_in_exp117": only_theirs,
            "pure_rule_details": pure_diag,
            "quantities": rows,
            "seconds": time.time() - t0,
        }
        summary = ", ".join(f"{f} {rep[f]['max_abs_diff']:.3g}" for f in FAMILIES + ("estimate",) if f in rep)
        log(f"{name}: done in {time.time() - t0:.1f} s; largest |difference|: {summary}; one side missing "
            f"{len(missing)}, method disagreements {len(methods)}, only in exp117 {len(only_theirs)}")
    rep = agg_all.report()
    out["max_by_family"] = rep
    within = all(rep.get(f, {}).get("within_bar", True) for f in FAMILIES)
    out["P1_bars"] = {"bars": BARS, "every_family_within_its_bar": within,
                      "one_side_missing": total_missing, "method_disagreements": total_methods,
                      "only_in_exp117": total_only_theirs}
    out["seconds"] = time.time() - _T0
    for f in FAMILIES + ("estimate", "overall"):
        if f in rep:
            log(f"largest |difference|, {f}: {rep[f]['max_abs_diff']:.3g} at {rep[f]['where']}")
    log(f"every family within its bar: {within}; one side missing {total_missing}; method disagreements "
        f"{total_methods}; only in exp117 {total_only_theirs}")
    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
