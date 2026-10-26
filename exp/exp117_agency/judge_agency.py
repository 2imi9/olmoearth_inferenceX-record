#!/usr/bin/env python
"""Grading a map with an agency's stratified sample: the judge's merged measurement. Run on aicr, never on the laptop.

    python -I judge_agency.py REC AREAS widths   > judge_widths.json   # every quantity, every interval, side by side
    python -I judge_agency.py REC AREAS enum     > judge_enum.json     # exact coverage of what betting_widths skips
    python -I judge_agency.py REC AREAS mc [R]   > judge_mc.json       # plug-in coverage at the agency designs' shapes
    python -I judge_agency.py select SELFTEST JUDGE_ENUM JUDGE_MC JUDGE_WIDTHS BOUNDS_ENUM > judge_select.json

REC is the research record's root (exp/out/exp94_nlcd_points.csv, exp/out/exp94_gfc_points.csv); AREAS is GFC2020's
strata_size.csv. numpy and oe_inferencex only; progress on stderr, JSON on stdout; workers SLURM_CPUS_PER_TASK.
Written blind (the laptop stays idle): nothing here has been run. Read `checks` (widths) and every `holds` (enum)
before any width.

Imported, not copied:
  betting_widths.py (env BETTING_PY, default ../betting/): the CHOSEN interval. Stratum h's e-value
      K_h = prod_t (1 + l_t (X_t - c_t)) exp(l_t (c_t - m_t(mu_h))),  l_t = c W_h (N_h - t) / (N_h n_h),
  c_t a predictable centre clipped at 0.9 / l_t, m_t the without-replacement null mean; averaged over every order of
  the stratum's labels (dynamic programme); log K_h is linear in mu_h with slope -c W_h, so the product over strata
  depends on the means only through theta = sum_h W_h mu_h; mixed over 12 values of c fixed by the design; each end
  where the mixture reaches 2/alpha (Markov). Areas: point frame, deflated for a pixel frame of N_h >= area / unit.
  agency_bounds.py (env BOUNDS_PY, default ../maintainer2/): the FALLBACK, the boxed stratified Chernoff bound (BSC).
Added here:
  - ratios (user's, producer's accuracy) with the centre clipped and no cap on c: the e-value at r' in [r1, r2] is
    bounded below per factor by the least of the factor at r1, r2 and the one r where the clip starts or stops (each
    log-factor is concave in r on either side of it), so a scan certifies every r below the lower end (betting_widths'
    capped form, which needs lambda <= 1, is kept beside as `betting_design1_capped`);
  - unknown values (WorldCover no-data, LCMAP 0) bounded by two observed variables: the lower end from the one that
    counts them against the map, the upper from the one that counts them for it;
  - the pure-stratum rule (M2) for a user's accuracy when the class has one declared pure stratum and counts are known;
  - NLCD's Table 4 as transcribed in practitioner_sbui.py (to check against the PDF) beside the point estimates, for
    the primary-or-alternate rule (reference = the map's class where it equals the alternate label, else the primary);
  - `select`: the rule, fixed before this run, that picks the shipped method per family (linear quantities, ratios):
    betting unless it fails a validity gate or its median width over the family exceeds the fallback's by more than
    10%; the fallback if it passes its own gates; else neither (the union bound stays).
"""
import csv
import importlib.util
import itertools
import json
import math
import multiprocessing
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.05
Z = 1.959963984540054
GAMMA_M2 = 0.005          # the pure-stratum rule's share of alpha for the mixed strata
UNIT_HA = 1.0             # GFC2020's strata_ha counts 100 m cells: a sampling unit of at most 1 ha gives N_h >= strata_ha
WIDTH_RULE = 1.10         # betting is shipped for a family unless its median width exceeds the fallback's by more than this
PROCS = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bw = _load("betting_widths", os.environ.get("BETTING_PY", os.path.join(HERE, "..", "betting", "betting_widths.py")))
_AB = []


def ab():
    """agency_bounds.py, loaded on first use (it imports oe_inferencex)."""
    if not _AB:
        _AB.append(_load("agency_bounds", os.environ.get("BOUNDS_PY", os.path.join(HERE, "..", "maintainer2",
                                                                                  "agency_bounds.py"))))
    return _AB[0]


def hyper_iv(k, n, N, conf):
    from oe_inferencex.estimate import hypergeom_interval
    return hypergeom_interval(int(round(k)), int(n), int(N), conf=conf)


# ------------------------------------------------------------------ published and earlier numbers (to compare with)
# Table 3 of Wickham et al. (2021), pixel counts of the 21 strata, and Table 4 (level II, 2016, percent and standard
# error), as transcribed in practitioner_sbui.py: to be checked against the PDF, which could not be rendered here.
NLCD_N = {1: 144260856, 2: 255518661, 3: 130760739, 4: 59945943, 5: 21576459, 6: 88893116, 7: 825184784,
          8: 979538237, 9: 316640402, 10: 1864314143, 11: 1159802223, 12: 559677646, 13: 1429403249,
          14: 380971449, 15: 117993354, 16: 146679429, 17: 6772002, 18: 73557194, 19: 70038668, 20: 4483066,
          21: 22289822}
# the 15 no-change strata hold one class in both years (the metadata: "15 no-change strata ... and 6 change strata")
NLCD_PURE = {1: 11, 2: 21, 3: 22, 4: 23, 5: 24, 6: 31, 7: 41, 8: 42, 9: 43, 10: 52, 11: 71, 12: 81, 13: 82, 14: 90,
             15: 95}
NLCD_TO_LCMAP = {21: 1, 22: 1, 23: 1, 24: 1, 81: 2, 82: 2, 52: 3, 71: 3, 41: 4, 42: 4, 43: 4, 11: 5, 90: 6, 95: 6,
                 12: 7, 31: 8}
PUB = {
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
    "95": {"ua": [54, 4], "uaa": [69, 4], "pa": [62, 6], "paa": [80, 6], "map_area": 1.4994, "ref_area": 1.3028, "tse": 0.1311, "mb": 0.20}}
PUBLISHED_OA = {   # percent; NLCD from the paper's text and Table 1 via practitioner_sbui.py and the record; GFC from exp94
    ("nlcd2016_B_2016", "oa_level2_primary"): 72.1, ("nlcd2016_B_2016", "oa_level2_primary_or_alternate"): 86.4,
    ("nlcd2016_B_2016", "oa_level1_primary"): 79.0,
    ("nlcd2016_B_2016", "oa_level1_primary_or_alternate_level2_alternate"): 90.6,
    ("nlcd2016_B_2011", "oa_level2_primary_or_alternate"): 86.5, ("gfc2020", "oa_gfc_v2"): 91.0}
SIZING = {  # agency/sizing.json (aicr): the data reading must reproduce these
    "gfc_oa": (0.9150627376612703, (0.9107113928132743, 0.9194140825092664)),
    "gfc_diff": (0.0009360773645878584, (-0.003415911021792898, 0.005288065750968615)),
    "nlcd_l2_primary": (0.7211334217077565, (0.7033254495268878, 0.7389413938886252)),
    "nlcd_ua52_primary_union": (0.5285539240483915, 0.7805659620586974)}
RECORD_ALT = {2016: 0.8638724418296603, 2011: 0.8650733895162801}   # exp94_summary.json, found_after_result


# ------------------------------------------------------------------ the chosen interval: linear quantities
def _st(W, n, N, Nlo, vals, counts):
    return {"W": float(W), "n": int(n), "N": N, "Nlo": Nlo, "vals": [float(v) for v in vals],
            "counts": [int(c) for c in counts]}


def bet_strata(s_lo, s_hi=None, alpha=ALPHA, K=None, span=None):
    """The betting interval for sum_h W_h mu_h: the lower end from the strata s_lo, the upper from s_hi (the same
    strata unless unknown values are bounded). The grid of c comes from the design (W, n, alpha) only."""
    K = bw.K_MIX if K is None else K
    span = bw.SPAN if span is None else span
    s_hi = s_lo if s_hi is None else s_hi
    cs = bw.c_grid(s_lo, alpha, cap=False, K=K, span=span)
    thr = math.log(2.0 / alpha)
    top = sum(s["W"] for s in s_lo)
    lo = bw.root_low(bw.lin_A(s_lo, cs, "low"), cs, thr, top)
    hi = top - bw.root_low(bw.lin_A(s_hi, cs, "high"), cs, thr, top)
    return float(lo), float(hi)


# ------------------------------------------------------------------ the chosen interval: ratios, uncapped
def stratum_logavg_mm(counts, a, b, lam0, N, n, r1, r2, kappa=None):
    """log of the average, over every order of one stratum's labels, of prod_t exp(g_t), g_t the least value over
    r in [r1, r2] of the stratum's log-factor log(1 + l_t (v_t(r) - c_t(r))) + l_t (c_t(r) + S_t(r) / (N - t)), the
    centre c_t(r) = min((1/2 + S_t(r)) / (t + 1), kappa / l_t) (as betting_widths' linear form), for each first-step
    lambda in lam0. Category j's value is a[j] + b[j] r. Each log-factor is concave in r on either side of the one r at
    which the clip starts or stops, so its least value over [r1, r2] is at r1, r2 or that r: the result is at most the
    stratum's e-value (without its -c W mu term) at every r in [r1, r2]. r1 == r2 gives the e-value at that r."""
    kappa = bw.KAPPA if kappa is None else kappa
    counts = [int(x) for x in counts] + [0] * (3 - len(counts))
    a = [float(x) for x in a] + [0.0] * (3 - len(a))
    b = [float(x) for x in b] + [0.0] * (3 - len(b))
    if sum(counts) != n:
        raise ValueError(f"counts {counts} do not add up to n = {n}")
    order = sorted(range(3), key=lambda j: -counts[j])
    k0, k1, k2 = (counts[j] for j in order)
    av, bv = [a[j] for j in order], [b[j] for j in order]
    lam0 = np.asarray(lam0, float).reshape(-1, 1, 1)
    K = lam0.shape[0]
    finite = math.isfinite(N)
    if n == 0:
        return np.zeros(K)

    def g(lt, j, S0, Sb, t):
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            safe = np.where(Sb == 0, 1.0, Sb)
            rb = np.where(Sb == 0, r1, ((t + 1) * kappa / lt - 0.5 - S0) / safe)
            rm = np.clip(rb, r1, r2)
            best = None
            for r in (r1, r2, rm):
                S = S0 + Sb * r
                cen = np.minimum((0.5 + S) / (t + 1), kappa / lt)
                tail = S / (N - t) if finite else 0.0
                f = np.log1p(lt * (av[j] + bv[j] * r - cen)) + lt * (cen + tail)
                best = f if best is None else np.minimum(best, f)
        return best

    if k1 == 0 and k2 == 0:
        t = np.arange(n, dtype=float).reshape(1, -1)
        lt = lam0.reshape(-1, 1) * ((N - t) / N if finite else 1.0)
        return g(lt, 0, t * av[0], t * bv[0], t).sum(axis=1)
    s1 = np.arange(k1 + 1, dtype=float).reshape(1, -1, 1)
    s2 = np.arange(k2 + 1, dtype=float).reshape(1, 1, -1)
    L = np.full((K, k1 + 1, k2 + 1), -np.inf)
    L[:, 0, 0] = 0.0
    for t in range(n):
        s0 = t - s1 - s2
        valid = (s0 >= 0) & (s0 <= k0)
        lt = lam0 * ((N - t) / N) if finite else lam0
        S0 = s0 * av[0] + s1 * av[1] + s2 * av[2]
        Sb = s0 * bv[0] + s1 * bv[1] + s2 * bv[2]
        new = np.full_like(L, -np.inf)
        for j, kj in enumerate((k0, k1, k2)):
            if kj == 0:
                continue
            with np.errstate(invalid="ignore"):
                src = L + np.where(valid, g(lt, j, S0, Sb, t), -np.inf)
            if j == 0:
                new = np.logaddexp(new, np.where(s0 + 1 <= k0, src, -np.inf))
            elif j == 1:
                new[:, 1:, :] = np.logaddexp(new[:, 1:, :], src[:, :-1, :])
            else:
                new[:, :, 1:] = np.logaddexp(new[:, :, 1:], src[:, :, :-1])
        L = new
    return L[:, k1, k2] - (math.lgamma(n + 1) - math.lgamma(k0 + 1) - math.lgamma(k1 + 1) - math.lgamma(k2 + 1))


def ratio_logE_mm(strata, cs, side, r1, r2):
    """log of a lower bound, valid at every r in [r1, r2], of the mixed e-value against R <= r ("low") or R >= r
    ("high"); categories (y=1, z=1), (y=0, z=1), (z=0) with u = y + r (1 - z) ("low") or 1 - u ("high")."""
    a, b = bw.RATIO_AB[side]
    A = np.zeros(cs.size)
    for s in strata:
        A += stratum_logavg_mm(s["counts"], a, b, cs * s["W"] / s["n"], s["N"], s["n"], r1, r2)
    A -= bw.deflation(strata, cs)
    SW = sum(s["W"] for s in strata)
    nv = r2 * SW if side == "low" else (1.0 - r1) * SW         # the largest null value over the interval
    return bw.logsumexp(-math.log(cs.size) + A - cs * nv)


def ratio_interval_mm(strata, alpha=ALPHA, K=None, span=None, dmin=2e-4):
    """R = theta_y / theta_z, y <= z in {0, 1}: every r below the lower end (above the upper) is certified rejected
    on intervals by ratio_logE_mm; the bisection only aims the scan. Returns (low, high)."""
    K = bw.K_MIX if K is None else K
    span = bw.SPAN if span is None else span
    if sum(s["counts"][0] + s["counts"][1] for s in strata) == 0:
        return 0.0, 1.0
    cs = bw.c_grid(strata, alpha, cap=False, K=K, span=span)
    thr = math.log(2.0 / alpha)
    rej = lambda side, r1, r2: ratio_logE_mm(strata, cs, side, r1, r2) >= thr
    if not rej("low", 0.0, 0.0):
        lo_end = 0.0
    else:
        if rej("low", 1.0, 1.0):
            target = 1.0
        else:
            x, y = 0.0, 1.0
            for _ in range(14):
                m = 0.5 * (x + y)
                x, y = (m, y) if rej("low", m, m) else (x, m)
            target = x
        r, step = 0.0, max(target / 2, dmin)
        while r < target - 1e-12:
            d = min(step, target - r)
            if rej("low", r, r + d):
                r, step = r + d, 2 * d
            elif d <= dmin:
                break
            else:
                step = d / 2
        lo_end = r
    if not rej("high", 1.0, 1.0):
        hi_end = 1.0
    else:
        if rej("high", 0.0, 0.0):
            target = 0.0
        else:
            x, y = 0.0, 1.0
            for _ in range(14):
                m = 0.5 * (x + y)
                x, y = (x, m) if rej("high", m, m) else (m, y)
            target = y
        r, step = 1.0, max((1.0 - target) / 2, dmin)
        while r > target + 1e-12:
            d = min(step, r - target)
            if rej("high", r - d, r):
                r, step = r - d, 2 * d
            elif d <= dmin:
                break
            else:
                step = d / 2
        hi_end = r
    return float(lo_end), float(hi_end)


# ------------------------------------------------------------------ the pure-stratum rule (M2)
def m2_ua_arrays(N, n, W, kA, kB, own, mixed, alpha=ALPHA, gamma=GAMMA_M2):
    """User's accuracy of a class whose map area lies in one declared pure stratum `own` (every unit mapped as the
    class) and the strata in `mixed`; every other stratum is pure for another class and adds 0. A union bound: own's
    right share at its exact (hypergeometric) interval at 1 - (alpha - gamma); each mixed stratum's two counts (mapped
    as the class and right, mapped as it and wrong) at 1 - gamma / (2 M), M the mixed strata not labelled in full.
    The ratio rises with own's share and the mixed right counts and falls with the mixed wrong ones."""
    open_mixed = [h for h in mixed if n[h] < N[h]]
    g = gamma if open_mixed else 0.0
    lAd, uAd = hyper_iv(kA[own], n[own], N[own], 1 - (alpha - g))
    A_lo = A_hi = B_lo = B_hi = 0.0
    for h in mixed:
        if n[h] >= N[h]:
            la = ua = kA[h] / n[h]
            lb = ub = kB[h] / n[h]
        else:
            conf = 1 - g / (2 * len(open_mixed))
            la, ua = hyper_iv(kA[h], n[h], N[h], conf)
            lb, ub = hyper_iv(kB[h], n[h], N[h], conf)
        A_lo += W[h] * la
        A_hi += W[h] * ua
        B_lo += W[h] * lb
        B_hi += W[h] * ub
    Wd = float(W[own])
    lo = (Wd * lAd + A_lo) / (Wd + A_lo + B_hi)
    hi = min(1.0, (Wd * uAd + A_hi) / (Wd + A_hi + B_lo))
    return float(lo), float(hi)


def normal_ratio_counts(W, ns, Ns, cts):
    """Stehman's ratio estimate and nominal 95% interval from per-stratum counts (A, B, C): y = A, z = A or B."""
    ty = sum(W[h] * cts[h][0] / ns[h] for h in range(len(W)) if ns[h] > 0)
    tz = sum(W[h] * (cts[h][0] + cts[h][1]) / ns[h] for h in range(len(W)) if ns[h] > 0)
    if tz <= 0:
        return 0.0, 1.0
    R = ty / tz
    var = 0.0
    for h in range(len(W)):
        if ns[h] > 1:
            e = [1 - R] * int(cts[h][0]) + [-R] * int(cts[h][1]) + [0.0] * int(cts[h][2])
            f = ns[h] / Ns[h] if math.isfinite(Ns[h]) else 0.0
            var += W[h] ** 2 * (1 - f) * float(np.var(e, ddof=1)) / ns[h]
    s = math.sqrt(var) / tz
    return R - Z * s, R + Z * s


# ------------------------------------------------------------------ data
def _design(hidx, H, n, W, N, Nlo):
    return {"hidx": hidx, "H": H, "n": n, "W": W, "N": N, "Nlo": Nlo, "masks": [hidx == h for h in range(H)]}


def load_nlcd(rec, year):
    """NLCD 2016 AA, set B, one year: N_h = n_h x weight, drawn without replacement (30 m pixels)."""
    path = os.path.join(rec, "exp", "out", "exp94_nlcd_points.csv")
    rows = [r for r in csv.DictReader(open(path)) if r["set"] == "B" and r["year"] == str(year)]
    iv = lambda r, k: int(float(r[k])) if r[k] not in ("", None) else 0
    st = np.array([int(r["stratum"]) for r in rows])
    w = np.array([float(r["weight"]) for r in rows])
    ids = np.array(sorted(set(st.tolist())))
    hidx = np.searchsorted(ids, st)
    H = int(ids.size)
    n = np.bincount(hidx, minlength=H).astype(int)
    Nf = np.bincount(hidx, weights=w, minlength=H)
    Nint = np.maximum(np.round(Nf), n).astype(np.int64)
    D = _design(hidx, H, n, Nint / Nint.sum(), Nint.astype(float), None)
    D.update({"name": f"nlcd2016_B_{year}", "ids": ids, "Nint": Nint, "Nfpc": Nint.astype(float),
              "ref": np.array([iv(r, "ref_nlcd") for r in rows]), "alt": np.array([iv(r, "ref_nlcd_alternate") for r in rows]),
              "map": np.array([iv(r, "nlcd") for r in rows]), "lcmap": np.array([iv(r, "lcmap") for r in rows])})
    D["rstar"] = np.where(D["map"] == D["alt"], D["map"], D["ref"])
    pure, viol = {}, {}
    for h in range(H):
        s = int(ids[h])
        if s in NLCD_PURE:
            bad = int((D["map"][D["masks"][h]] != NLCD_PURE[s]).sum())
            if bad:
                viol[str(s)] = bad
            else:
                pure[h] = NLCD_PURE[s]
    D["pure"] = pure
    D["checks"] = {
        "points": len(rows), "strata": H, "n_min": int(n.min()), "n_max": int(n.max()), "W_max": float(D["W"].max()),
        "largest_weight_spread_within_a_stratum": max(float(w[hidx == h].max() / w[hidx == h].min() - 1)
                                                     for h in range(H)),
        "largest_distance_of_N_to_an_integer": float(np.max(np.abs(Nf - np.round(Nf)))),
        "largest_relative_difference_to_table3": max((abs(float(Nint[h]) - NLCD_N[int(ids[h])]) / NLCD_N[int(ids[h])]
                                                      for h in range(H) if int(ids[h]) in NLCD_N), default=None),
        "reference_0": int((D["ref"] == 0).sum()), "map_0": int((D["map"] == 0).sum()),
        "alternate_0": int((D["alt"] == 0).sum()), "alternate_equals_primary": int((D["alt"] == D["ref"]).sum()),
        "lcmap_0": int((D["lcmap"] == 0).sum()), "purity_violations": viol,
        "pure_strata_verified": sorted(int(ids[h]) for h in pure)}
    return D


def load_gfc(rec, areas):
    """GFC2020: 149 continental strata with areas in ha; the strata holding points, as sizing.py and the agency."""
    area = {int(r["Strata"]): float(r["strata_ha"]) for r in csv.DictReader(open(areas))}
    rows_all = list(csv.DictReader(open(os.path.join(rec, "exp", "out", "exp94_gfc_points.csv"))))
    rows = [r for r in rows_all if int(r["stratum"]) in area]
    st = np.array([int(r["stratum"]) for r in rows])
    ids = np.array(sorted(set(st.tolist())))
    hidx = np.searchsorted(ids, st)
    H = int(ids.size)
    n = np.bincount(hidx, minlength=H).astype(int)
    A = np.array([area[int(s)] for s in ids])
    D = _design(hidx, H, n, A / A.sum(), np.full(H, math.inf), A / UNIT_HA)
    wc = np.array([int(r["worldcover_forest"]) if r["worldcover_forest"] != "" else -1 for r in rows])
    empty = sorted(set(area) - set(int(s) for s in ids))
    D.update({"name": "gfc2020", "ids": ids, "A": A, "Nint": None, "Nfpc": A / 0.01,
              "ref": np.array([int(r["ref_forest"]) for r in rows]), "gfc": np.array([int(r["gfc_v2"]) for r in rows]),
              "wc": wc, "pure": {}})
    D["checks"] = {"points": len(rows), "strata": H, "rows_outside_area_table": len(rows_all) - len(rows),
                   "strata_without_points": empty,
                   "area_share_without_points": float(sum(area[s] for s in empty) / sum(area.values())),
                   "worldcover_missing": int((wc < 0).sum()), "n_min": int(n.min()), "n_max": int(n.max()),
                   "W_max": float(D["W"].max()), "max_W_over_n": float((D["W"] / n).max()),
                   "c_at_which_lambda_reaches_1": float((n / D["W"]).min())}
    return D


def cnt(D, mask):
    return np.bincount(D["hidx"], weights=np.asarray(mask, dtype=float), minlength=D["H"])


# ------------------------------------------------------------------ quantities on a design
METHODS = ("normal", "union", "betting", "betting_point_frame", "betting_design1_capped", "m2", "bsc")


def finish(out):
    nw = (out["normal"][1] - out["normal"][0]) if isinstance(out.get("normal"), list) else None
    out["widths_points"] = {m: 100 * (out[m][1] - out[m][0]) for m in METHODS if isinstance(out.get(m), list)}
    if nw and nw > 0:
        out["x_normal"] = {m: (out[m][1] - out[m][0]) / nw for m in METHODS if isinstance(out.get(m), list)}
    if isinstance(out.get("betting"), list) and isinstance(out.get("bsc"), list):
        wb = out["bsc"][1] - out["bsc"][0]
        out["betting_over_bsc"] = (out["betting"][1] - out["betting"][0]) / wb if wb > 0 else None
    if isinstance(out.get("m2"), list) and isinstance(out.get("betting"), list):
        wb = out["betting"][1] - out["betting"][0]
        out["m2_over_betting"] = (out["m2"][1] - out["m2"][0]) / wb if wb > 0 else None
    est = out.get("estimate")
    lo_e, hi_e = (est, est) if not isinstance(est, list) else est
    if lo_e is not None:
        out["misses_its_estimate"] = [m for m in METHODS if m != "normal" and isinstance(out.get(m), list)
                                      and not (out[m][0] - 1e-9 <= lo_e and hi_e <= out[m][1] + 1e-9)]
    return out


def lin_strata(D, x, deflate=True):
    out = []
    for h in range(D["H"]):
        vals, counts = np.unique(x[D["masks"][h]], return_counts=True)
        out.append(_st(D["W"][h], D["n"][h], float(D["N"][h]),
                       float(D["Nlo"][h]) if (deflate and D["Nlo"] is not None) else None, vals, counts))
    return out


def normal_linear(D, x):
    est = var = 0.0
    for h in range(D["H"]):
        xs = x[D["masks"][h]]
        est += float(D["W"][h]) * float(xs.mean())
        if xs.size > 1:
            var += float(D["W"][h]) ** 2 * (1 - xs.size / float(D["Nfpc"][h])) * float(xs.var(ddof=1)) / xs.size
    return est, var


def q_linear(D, x_lo, x_hi=None, scale=1.0, shift=0.0, bsc=None, extra=False):
    """theta = scale * sum_h W_h mean_h(x) + shift, x in [0, 1]; with x_hi, unknown values bounded (lower end from
    x_lo, upper from x_hi). bsc: a function giving the fallback's interval in theta's units."""
    t0 = time.time()
    T = lambda v: scale * v + shift
    e_lo, v_lo = normal_linear(D, x_lo)
    e_hi, v_hi = (e_lo, v_lo) if x_hi is None else normal_linear(D, x_hi)
    out = {"family": "linear", "estimate": T(e_lo) if x_hi is None else [T(e_lo), T(e_hi)],
           "normal": [T(e_lo - Z * math.sqrt(v_lo)), T(e_hi + Z * math.sqrt(v_hi))]}
    s_lo = lin_strata(D, x_lo)
    s_hi = s_lo if x_hi is None else lin_strata(D, x_hi)
    lo, hi = bet_strata(s_lo, s_hi)
    out["betting"] = [T(lo), T(hi)]
    out["seconds_betting"] = time.time() - t0
    if extra:
        if D["Nlo"] is not None:
            p_lo = lin_strata(D, x_lo, deflate=False)
            p_hi = p_lo if x_hi is None else lin_strata(D, x_hi, deflate=False)
            a, b = bet_strata(p_lo, p_hi)
            out["betting_point_frame"] = [T(a), T(b)]
        sens = {}
        for K, span in ((6, 10.0), (24, 10.0), (12, 3.0), (12, 30.0)):
            a, b = bet_strata(s_lo, s_hi, K=K, span=span)
            sens[f"K{K}_span{span:g}"] = 100 * scale * (b - a)
        out["betting_width_points_by_mixture"] = sens
        out["gaussian_chernoff_floor_width_points"] = 100 * scale * 2 * math.sqrt(2 * max(v_lo, v_hi) * math.log(2 / ALPHA))
    if bsc is not None:
        t1 = time.time()
        try:
            out["bsc"] = [float(v) for v in bsc()]
        except Exception:
            out["bsc_error"] = traceback.format_exc()
        out["seconds_bsc"] = time.time() - t1
    return finish(out)


def strata_ratio(D, y, z, deflate=True):
    out = []
    for h in range(D["H"]):
        m = D["masks"][h]
        yy, zz = y[m], z[m]
        out.append({"W": float(D["W"][h]), "n": int(D["n"][h]), "N": float(D["N"][h]),
                    "Nlo": float(D["Nlo"][h]) if (deflate and D["Nlo"] is not None) else None,
                    "counts": [int(((yy == 1) & (zz == 1)).sum()), int(((yy == 0) & (zz == 1)).sum()),
                               int((zz == 0).sum())]})
    return out


def normal_ratio(D, y, z):
    ty = tz = 0.0
    for h in range(D["H"]):
        m = D["masks"][h]
        ty += float(D["W"][h]) * float(y[m].mean())
        tz += float(D["W"][h]) * float(z[m].mean())
    if tz <= 0:
        return None, None
    R = ty / tz
    var = 0.0
    for h in range(D["H"]):
        m = D["masks"][h]
        e = y[m] - R * z[m]
        if e.size > 1:
            var += float(D["W"][h]) ** 2 * (1 - e.size / float(D["Nfpc"][h])) * float(e.var(ddof=1)) / e.size
    return R, var / tz ** 2


def q_ratio(D, y_lo, z_lo, y_hi=None, z_hi=None, m2=None, bsc=None, capped=True):
    """R = sum W mean(y) / sum W mean(z), y <= z; with y_hi, z_hi, unknown values bounded."""
    t0 = time.time()
    R_lo, v_lo = normal_ratio(D, y_lo, z_lo)
    R_hi, v_hi = (R_lo, v_lo) if y_hi is None else normal_ratio(D, y_hi, z_hi)
    out = {"family": "ratio", "n_numerator": int(y_lo.sum()), "n_denominator": int(z_lo.sum())}
    if R_lo is None or R_hi is None:
        out["note"] = "no labelled point in the denominator"
        return out
    out["estimate"] = R_lo if y_hi is None else [R_lo, R_hi]
    out["normal"] = [R_lo - Z * math.sqrt(v_lo), R_hi + Z * math.sqrt(v_hi)]
    s_lo = strata_ratio(D, y_lo, z_lo)
    s_hi = None if y_hi is None else strata_ratio(D, y_hi, z_hi)
    lo, hi = ratio_interval_mm(s_lo)
    if s_hi is not None:
        _, hi = ratio_interval_mm(s_hi)
    out["betting"] = [lo, hi]
    out["seconds_betting"] = time.time() - t0
    if capped:
        t1 = time.time()
        try:
            a, b, _ = bw.ratio_interval(s_lo, ALPHA)
            if s_hi is not None:
                _, b, _ = bw.ratio_interval(s_hi, ALPHA)
            out["betting_design1_capped"] = [float(a), float(b)]
        except Exception:
            out["betting_design1_capped_error"] = traceback.format_exc()
        out["seconds_betting_design1_capped"] = time.time() - t1
    if m2 is not None:
        try:
            out["m2"] = list(m2())
        except Exception:
            out["m2_error"] = traceback.format_exc()
    if bsc is not None:
        t1 = time.time()
        try:
            out["bsc"] = [float(v) for v in bsc()]
        except Exception:
            out["bsc_error"] = traceback.format_exc()
        out["seconds_bsc"] = time.time() - t1
    return finish(out)


# the fallback (agency_bounds.py), on the same counts: each returns its interval in the quantity's own units
def bsc_share(D, k_lo, k_hi=None):
    return lambda: ab().share_methods(k_lo, D["n"].astype(float), D["W"], D["Nfpc"], None, D["Nint"],
                                      k_hi=k_hi)["chernoff_boxed"]


def bsc_diff(D, kp_lo, km_lo, kp_hi=None, km_hi=None):
    return lambda: ab().diff_methods(kp_lo, km_lo, D["n"].astype(float), D["W"], D["Nfpc"], None, D["Nint"],
                                     kp_hi=kp_hi, km_hi=km_hi)["chernoff_boxed"]


def bsc_ratio(D, kt, kf):
    return lambda: ab().ratio_methods(kt, kf, D["n"].astype(float), D["W"], D["Nfpc"], None, D["Nint"])["chernoff_boxed"]


def kinds_for(D, c):
    """(the one verified pure stratum of class c or None, the strata pure for no class)"""
    own = [h for h, cl in D["pure"].items() if cl == c]
    mixed = [h for h in range(D["H"]) if h not in D["pure"]]
    return (own[0] if len(own) == 1 else None), mixed


# ------------------------------------------------------------------ the jobs
_JOBS = []
_DATA = {}


def _run_job(i):
    key, f = _JOBS[i]
    t = time.time()
    try:
        out = f()
    except Exception:
        out = {"error": traceback.format_exc()}
    say(f"{'/'.join(key)}: {time.time() - t:.0f} s")
    return key, out


def gfc_ratio(D, mv, cls, what, may_miss):
    ref = D["ref"]
    miss = (mv < 0) if may_miss else np.zeros(mv.size, bool)
    known = mv == cls
    f = lambda m: m.astype(float)
    if what == "user":
        y_lo, z_lo = f(known & (ref == cls)), f(known | (miss & (ref != cls)))
        y_hi, z_hi = f((known | miss) & (ref == cls)), f(known | (miss & (ref == cls)))
        kt, kf = cnt(D, known & (ref == cls)), cnt(D, known & (ref != cls))
    else:
        y_lo, y_hi = f(known & (ref == cls)), f((known | miss) & (ref == cls))
        z_lo = z_hi = f(ref == cls)
        kt, kf = cnt(D, known & (ref == cls)), cnt(D, ~known & ~miss & (ref == cls))
    b = bool(miss.any())
    return q_ratio(D, y_lo, z_lo, y_hi if b else None, z_hi if b else None, bsc=None if b else bsc_ratio(D, kt, kf))


def gfc_jobs(D):
    jobs = []
    for mname in ("gfc_v2", "worldcover"):
        mv = D["gfc"] if mname == "gfc_v2" else D["wc"]
        for cls, cname in ((1, "forest"), (0, "nonforest")):
            for what in ("user", "producer"):
                jobs.append((("gfc2020", f"{what}_{cname}_{mname}"),
                             (lambda mv=mv, cls=cls, what=what, wc=(mname == "worldcover"):
                              gfc_ratio(D, mv, cls, what, wc))))
    ref, g, wc = D["ref"], D["gfc"], D["wc"]
    miss = wc < 0
    okg = g == ref
    okw_lo = (wc == ref) & ~miss
    okw_hi = okw_lo | miss
    b = bool(miss.any())
    f = lambda m: m.astype(float)
    jobs += [
        (("gfc2020", "oa_gfc_v2"), lambda: q_linear(D, f(okg), bsc=bsc_share(D, cnt(D, okg)), extra=True)),
        (("gfc2020", "oa_worldcover"), lambda: q_linear(
            D, f(okw_lo), f(okw_hi) if b else None,
            bsc=bsc_share(D, cnt(D, okw_lo), cnt(D, okw_hi) if b else None))),
        (("gfc2020", "gfc_minus_worldcover"), lambda: q_linear(       # WorldCover's no-data counted right at the lower end
            D, (f(okg) - f(okw_hi) + 1) / 2, (f(okg) - f(okw_lo) + 1) / 2 if b else None, scale=2.0, shift=-1.0,
            bsc=bsc_diff(D, cnt(D, okg & ~okw_hi), cnt(D, ~okg & okw_hi),
                         cnt(D, okg & ~okw_lo) if b else None, cnt(D, ~okg & okw_lo) if b else None), extra=True)),
        (("gfc2020", "forest_share_reference"), lambda: q_linear(D, f(ref == 1), bsc=bsc_share(D, cnt(D, ref == 1)))),
        (("gfc2020", "forest_map_bias_gfc_v2"), lambda: q_linear(
            D, (f(g == 1) - f(ref == 1) + 1) / 2, scale=2.0, shift=-1.0,
            bsc=bsc_diff(D, cnt(D, (g == 1) & (ref != 1)), cnt(D, (g != 1) & (ref == 1)))))]
    return jobs


def nlcd_class(D, c):
    m, r, p = D["map"], D["rstar"], D["ref"]
    mc, rc, pc = m == c, r == c, p == c
    f = lambda x: x.astype(float)
    own, mixed = kinds_for(D, c)
    out = {"n_mapped": int(mc.sum()), "n_reference": int(rc.sum()), "n_reference_primary": int(pc.sum()),
           "own_stratum": None if own is None else int(D["ids"][own])}
    n, W, N = D["n"], D["W"], D["Nint"]
    kA, kB = cnt(D, mc & rc), cnt(D, mc & ~rc)
    out["user"] = q_ratio(D, f(mc & rc), f(mc),
                          m2=(lambda: m2_ua_arrays(N, n, W, kA, kB, own, mixed)) if own is not None else None,
                          bsc=bsc_ratio(D, kA, kB))
    out["producer"] = q_ratio(D, f(mc & rc), f(rc), bsc=bsc_ratio(D, kA, cnt(D, ~mc & rc)))
    kAp, kBp = cnt(D, mc & pc), cnt(D, mc & ~pc)
    up = q_ratio(D, f(mc & pc), f(mc), capped=False,
                 m2=(lambda: m2_ua_arrays(N, n, W, kAp, kBp, own, mixed)) if own is not None else None)
    try:
        up["union"] = [float(v) for v in ab().union_ratio(kAp, kBp, n.astype(float), W, N)]
        finish(up)
    except Exception:
        up["union_error"] = traceback.format_exc()
    out["user_primary"] = up
    out["reference_share"] = q_linear(D, f(rc), bsc=bsc_share(D, cnt(D, rc)))
    out["map_bias"] = q_linear(D, (f(mc) - f(rc) + 1) / 2, scale=2.0, shift=-1.0,
                               bsc=bsc_diff(D, cnt(D, mc & ~rc), cnt(D, ~mc & rc)))
    return out


def nlcd_jobs(D, D11):
    classes = sorted((set(D["map"].tolist()) | set(D["rstar"].tolist())) - {0})
    jobs = [((D["name"], "per_class", str(c)), (lambda c=c: nlcd_class(D, c))) for c in classes]
    m, p, alt = D["map"], D["ref"], D["alt"]
    f = lambda x: x.astype(float)
    ap = m == p
    aa = ap | (m == alt)
    l1p = (m // 10) == (p // 10)
    l1a = l1p | ((m // 10) == (alt // 10))
    for key, x in (("oa_level2_primary", ap), ("oa_level2_primary_or_alternate", aa), ("oa_level1_primary", l1p),
                   ("oa_level1_primary_or_alternate_level2_alternate", l1a)):
        jobs.append(((D["name"], key), (lambda x=x, key=key: q_linear(D, f(x), bsc=bsc_share(D, cnt(D, x)),
                                                                       extra=key.startswith("oa_level2")))))
    cw = lambda arr: np.array([NLCD_TO_LCMAP.get(int(v), 0) for v in arr])
    r8, a8, n8, lc = cw(p), cw(alt), cw(m), D["lcmap"]
    miss = lc == 0
    bnd = bool(miss.any())
    for rule in ("primary", "primary_or_alternate"):
        use = rule != "primary"
        ok_a = ((n8 == r8) | (use & (n8 == a8))) & (r8 > 0)
        ok_b = ((lc == r8) | (use & (lc == a8))) & ~miss & (r8 > 0)
        b_lo, b_hi = ok_b | miss, ok_b                       # LCMAP's no value counted right at the lower end of NLCD - LCMAP
        jobs.append(((D["name"], f"nlcd_minus_lcmap_8class_{rule}"),
                     (lambda ok_a=ok_a, b_lo=b_lo, b_hi=b_hi: q_linear(
                         D, (f(ok_a) - f(b_lo) + 1) / 2, (f(ok_a) - f(b_hi) + 1) / 2 if bnd else None,
                         scale=2.0, shift=-1.0,
                         bsc=bsc_diff(D, cnt(D, ok_a & ~b_lo), cnt(D, ~ok_a & b_lo),
                                      cnt(D, ok_a & ~b_hi) if bnd else None, cnt(D, ~ok_a & b_hi) if bnd else None)))))
    aa11 = (D11["map"] == D11["ref"]) | (D11["map"] == D11["alt"])
    jobs.append(((D11["name"], "oa_level2_primary_or_alternate"),
                 lambda: q_linear(D11, f(aa11), bsc=bsc_share(D11, cnt(D11, aa11)))))
    return jobs


def table4(D):
    """Point estimates by Stehman's estimators beside NLCD's Table 4 (as transcribed), for the reading of the
    alternate label: uaa and paa with the reference taken as the map's class where it equals the alternate."""
    m, p, r = D["map"], D["ref"], D["rstar"]
    agree_a = (m == p) | (m == D["alt"])

    def share(mask):
        return sum(float(D["W"][h]) * float(mask[D["masks"][h]].mean()) for h in range(D["H"]))
    rows = {}
    for c in sorted(int(k) for k in PUB):
        msh, psh, rsh = share(m == c), share(p == c), share(r == c)
        row = {"ua": share((m == c) & (p == c)) / msh if msh > 0 else None,
               "uaa": share((m == c) & (r == c)) / msh if msh > 0 else None,
               "pa": share((m == c) & (p == c)) / psh if psh > 0 else None,
               "paa": share((m == c) & (r == c)) / rsh if rsh > 0 else None,
               "paa_primary_denominator": share((m == c) & agree_a) / psh if psh > 0 else None,
               "map_area_pct": 100 * msh, "ref_area_pct_primary": 100 * psh, "ref_area_pct_rstar": 100 * rsh}
        pub = PUB[str(c)]
        dif = {}
        for k in ("ua", "uaa", "pa", "paa"):
            if row[k] is not None:
                dif[k] = 100 * row[k] - pub[k][0]
        if row["paa_primary_denominator"] is not None:
            dif["paa_primary_denominator"] = 100 * row["paa_primary_denominator"] - pub["paa"][0]
        dif["map_area"] = row["map_area_pct"] - pub["map_area"]
        dif["ref_area_primary"] = row["ref_area_pct_primary"] - pub["ref_area"]
        dif["ref_area_rstar"] = row["ref_area_pct_rstar"] - pub["ref_area"]
        dif["map_bias_primary"] = (row["map_area_pct"] - row["ref_area_pct_primary"]) - pub["mb"]
        dif["map_bias_rstar"] = (row["map_area_pct"] - row["ref_area_pct_rstar"]) - pub["mb"]
        row["minus_published"] = dif
        row["published"] = pub
        rows[str(c)] = row
    keys = ("ua", "uaa", "pa", "paa", "paa_primary_denominator", "map_area", "ref_area_primary", "ref_area_rstar",
            "map_bias_primary", "map_bias_rstar")
    big = {k: max((abs(v["minus_published"][k]) for v in rows.values() if k in v["minus_published"]), default=None)
           for k in keys}
    return {"per_class": rows, "largest_absolute_difference_points": big,
            "reading": ("accuracies are rounded to whole percent in the table: a difference within 0.5 matches; areas "
                        "to 4 decimals of a percent. Whichever of ref_area_primary / ref_area_rstar and of paa / "
                        "paa_primary_denominator matches says how Wickham et al. used the alternate label")}


def sizing_checks(res):
    out = {}

    def close(got, want, tol=1e-8):
        if isinstance(want, (list, tuple)):
            return {"got": list(got), "want": list(want), "ok": all(abs(a - b) <= tol for a, b in zip(got, want))}
        return {"got": got, "want": want, "ok": abs(got - want) <= tol}
    try:
        g = res["gfc2020"]["oa_gfc_v2"]
        out["gfc_oa_estimate"] = close(g["estimate"], SIZING["gfc_oa"][0])
        out["gfc_oa_normal"] = close(g["normal"], SIZING["gfc_oa"][1])
        d = res["gfc2020"]["gfc_minus_worldcover"]
        if not isinstance(d["estimate"], list):
            out["gfc_diff_estimate"] = close(d["estimate"], SIZING["gfc_diff"][0])
            out["gfc_diff_normal"] = close(d["normal"], SIZING["gfc_diff"][1])
        n = res["nlcd2016_B_2016"]["oa_level2_primary"]
        out["nlcd_l2_primary_estimate"] = close(n["estimate"], SIZING["nlcd_l2_primary"][0])
        out["nlcd_l2_primary_normal"] = close(n["normal"], SIZING["nlcd_l2_primary"][1])
        out["nlcd_ua52_primary_union"] = close(res["nlcd2016_B_2016"]["per_class"]["52"]["user_primary"]["union"],
                                               SIZING["nlcd_ua52_primary_union"])
        out["nlcd2016_primary_or_alternate_vs_record"] = close(
            res["nlcd2016_B_2016"]["oa_level2_primary_or_alternate"]["estimate"], RECORD_ALT[2016], 1e-9)
        out["nlcd2011_primary_or_alternate_vs_record"] = close(
            res["nlcd2016_B_2011"]["oa_level2_primary_or_alternate"]["estimate"], RECORD_ALT[2011], 1e-9)
    except Exception:
        out["error"] = traceback.format_exc()
    errs = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                if k.endswith("error"):
                    errs.append(f"{path}/{k}")
                elif k == "misses_its_estimate" and v:
                    errs.append(f"{path}: {v} miss the estimate")
                else:
                    walk(v, f"{path}/{k}")
    out["all_ok"] = all(v.get("ok", True) for v in out.values() if isinstance(v, dict)) and "error" not in out
    walk({k: v for k, v in res.items() if k in ("gfc2020", "nlcd2016_B_2016", "nlcd2016_B_2011")}, "")
    out["errors_and_misses"] = errs                 # reported, not part of all_ok (which is the data reading only)
    out["no_errors_or_misses"] = not errs
    return out


def widths(rec, areas):
    t0 = time.time()
    gfc, nl16, nl11 = load_gfc(rec, areas), load_nlcd(rec, 2016), load_nlcd(rec, 2011)
    ab()                                                     # loaded before the fork, so every worker has it
    jobs = gfc_jobs(gfc)
    jobs = [j for j in jobs if "_" in j[0][1] and j[0][1].split("_")[0] in ("user", "producer")] + \
           nlcd_jobs(nl16, nl11) + [j for j in jobs if j[0][1].split("_")[0] not in ("user", "producer")]
    _JOBS[:] = jobs                                          # the slowest (GFC ratios) first
    res = {}
    say(f"widths: {len(_JOBS)} jobs on {PROCS} workers")
    partial = os.environ.get("JUDGE_PARTIAL", "judge_widths.partial.jsonl")   # each job as it ends: a timeout keeps them
    with multiprocessing.get_context("fork").Pool(PROCS) as pool, open(partial, "a") as part:
        for key, out in pool.imap_unordered(_run_job, range(len(_JOBS)), chunksize=1):
            part.write(json.dumps({"key": list(key), "result": out}, default=_json) + "\n")
            part.flush()
            node = res
            for k in key[:-1]:
                node = node.setdefault(k, {})
            node[key[-1]] = out
    res["data"] = {d["name"]: d["checks"] for d in (gfc, nl16, nl11)}
    res["table4_nlcd2016_level2"] = table4(nl16)
    res["checks"] = sizing_checks(res)
    res["e2e"] = {f"{a}/{b}": {"published_pct": pub, **{k: ((res.get(a) or {}).get(b) or {}).get(k)
                                                     for k in ("estimate", "normal", "betting", "bsc", "widths_points")}}
                  for (a, b), pub in PUBLISHED_OA.items()}
    res["settings"] = {"alpha": ALPHA, "mixture": bw.K_MIX, "span": bw.SPAN, "kappa": bw.KAPPA,
                       "gamma_m2": GAMMA_M2, "gfc_unit_ha": UNIT_HA, "workers": PROCS, "numpy": np.__version__}
    res["seconds"] = time.time() - t0
    return res


# ------------------------------------------------------------------ enum: exact coverage on small populations
def _comps(total, parts):
    return [c for c in itertools.product(range(total + 1), repeat=parts) if sum(c) == total]


def _hyp(N, K, n, k):
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)


def _mvh(pop, out):
    p = 1
    for P, o in zip(pop, out):
        p *= math.comb(int(P), int(o))
    return p / math.comb(int(sum(pop)), int(sum(out)))


def _tally(ivs, outs, pops, prob, theta, claimed, alpha):
    """ivs[o][form] = (lo, hi); prob(pop, o); theta(pop) (None: undefined). Minimum coverage over pops per form."""
    names = list(ivs[outs[0]].keys())
    worst = {f: [2.0, None] for f in names}
    for pop in pops:
        th = theta(pop)
        if th is None:
            continue
        cov = dict.fromkeys(names, 0.0)
        for o in outs:
            p = prob(pop, o)
            if p == 0:
                continue
            for f in names:
                lo, hi = ivs[o][f]
                if lo - 1e-12 <= th <= hi + 1e-12:
                    cov[f] += p
        for f in names:
            if cov[f] < worst[f][0]:
                worst[f] = [cov[f], [list(x) if isinstance(x, tuple) else x for x in pop]]
    res = {f: {"min_coverage": worst[f][0], "at_population": worst[f][1],
               "mean_width": float(np.mean([ivs[o][f][1] - ivs[o][f][0] for o in outs]))} for f in names}
    res["holds"] = all(res[f]["min_coverage"] >= 1 - alpha - 1e-12 for f in claimed)
    nominal = [f for f in names if f.startswith("normal")]
    if nominal:
        res["normal_falls_below_somewhere"] = any(res[f]["min_coverage"] < 1 - alpha for f in nominal)
    return res


def _normal_binary(W, ns, Ns, ks):
    est = sum(W[h] * ks[h] / ns[h] for h in range(len(W)))
    var = 0.0
    for h in range(len(W)):
        if ns[h] > 1:
            f = ns[h] / Ns[h] if math.isfinite(Ns[h]) else 0.0
            var += W[h] ** 2 * (1 - f) * (ks[h] * (ns[h] - ks[h]) / (ns[h] * (ns[h] - 1))) / ns[h]
    return est - Z * math.sqrt(var), est + Z * math.sqrt(var)


def _enum_binary(Ns, ns, W, alpha, forms, claimed):
    H = len(Ns)
    outs = list(itertools.product(*[range(n + 1) for n in ns]))
    ivs = {ks: forms(ks) for ks in outs}
    pops = list(itertools.product(*[range(N + 1) for N in Ns]))
    prob = lambda Ks, ks: float(np.prod([_hyp(Ns[h], Ks[h], ns[h], ks[h]) for h in range(H)]))
    theta = lambda Ks: sum(W[h] * Ks[h] / Ns[h] for h in range(H))
    return {"N": list(Ns), "n": list(ns), "W": list(W),
            **_tally(ivs, outs, pops, prob, theta, claimed, alpha)}


def enum_areas(alpha):
    """Areas only: the point-frame e-value deflated for N_h >= N_lo, under draws WITHOUT replacement from N_h = N_lo."""
    Ns, ns = (5, 7), (3, 4)
    W = [Ns[0] / sum(Ns), Ns[1] / sum(Ns)]

    def forms(ks):
        mk = lambda N, Nlo: [_st(W[h], ns[h], N[h], Nlo[h], [0, 1], [ns[h] - ks[h], ks[h]]) for h in range(2)]
        return {"betting_without_replacement": bet_strata(mk(Ns, (None, None)), alpha=alpha),
                "betting_point_frame_deflated": bet_strata(mk((math.inf, math.inf), (float(Ns[0]), float(Ns[1]))),
                                                           alpha=alpha),
                "betting_point_frame_not_deflated_NOT_CLAIMED": bet_strata(mk((math.inf, math.inf), (None, None)),
                                                                           alpha=alpha),
                "normal_NOMINAL": _normal_binary(W, ns, Ns, ks)}
    return _enum_binary(Ns, ns, W, alpha, forms, ("betting_without_replacement", "betting_point_frame_deflated"))


def enum_census(alpha):
    Ns, ns = (6, 25), (6, 8)
    W = [Ns[0] / sum(Ns), Ns[1] / sum(Ns)]
    forms = lambda ks: {"betting_without_replacement": bet_strata(
        [_st(W[h], ns[h], Ns[h], None, [0, 1], [ns[h] - ks[h], ks[h]]) for h in range(2)], alpha=alpha),
        "normal_NOMINAL": _normal_binary(W, ns, Ns, ks)}
    return _enum_binary(Ns, ns, W, alpha, forms, ("betting_without_replacement",))


def enum_three(alpha):
    """Weights not proportional to the strata's sizes (areas), and a stratum of one label."""
    Ns, ns, W = (4, 6, 3), (2, 3, 1), (0.2, 0.5, 0.3)
    forms = lambda ks: {"betting_without_replacement": bet_strata(
        [_st(W[h], ns[h], Ns[h], None, [0, 1], [ns[h] - ks[h], ks[h]]) for h in range(3)], alpha=alpha),
        "normal_NOMINAL": _normal_binary(W, ns, Ns, ks)}
    return _enum_binary(Ns, ns, W, alpha, forms, ("betting_without_replacement",))


def enum_unknown(alpha):
    """Units the reference could not label: per stratum (right, wrong, unlabelled truly right, unlabelled truly
    wrong); the sample shows (right, wrong, unlabelled). Bounded: the lower end from x = 1 for right only, the upper
    from x = 1 for right or unlabelled. The agency's drop convention beside, not claimed."""
    Ns, ns, W = (5, 5), (3, 3), (0.5, 0.5)
    outs_h = [_comps(n, 3) for n in ns]
    outs = list(itertools.product(*outs_h))

    def iv(o):
        lo_st = [_st(W[h], ns[h], Ns[h], None, [0, 1], [o[h][1] + o[h][2], o[h][0]]) for h in range(2)]
        hi_st = [_st(W[h], ns[h], Ns[h], None, [0, 1], [o[h][1], o[h][0] + o[h][2]]) for h in range(2)]
        res = {"betting_bounded": bet_strata(lo_st, hi_st, alpha=alpha)}
        if all(o[h][0] + o[h][1] > 0 for h in range(2)):
            m = [o[h][0] + o[h][1] for h in range(2)]
            res["normal_drop_NOT_CLAIMED"] = _normal_binary(W, m, (math.inf, math.inf), [o[h][0] for h in range(2)])
        else:
            res["normal_drop_NOT_CLAIMED"] = (0.0, 1.0)
        return res
    ivs = {o: iv(o) for o in outs}
    pops = list(itertools.product(*[_comps(N, 4) for N in Ns]))
    prob = lambda pop, o: float(np.prod([_mvh((pop[h][0], pop[h][1], pop[h][2] + pop[h][3]), o[h]) for h in range(2)]))
    theta = lambda pop: sum(W[h] * (pop[h][0] + pop[h][2]) / Ns[h] for h in range(2))
    return {"N": list(Ns), "n": list(ns), **_tally(ivs, outs, pops, prob, theta, ("betting_bounded",), alpha)}


def enum_difference(alpha):
    """Two maps on the same units: (A only right, B only right, neither or both); d = A - B in {1, -1, 0}."""
    Ns, ns = (5, 6), (3, 3)
    W = [Ns[0] / sum(Ns), Ns[1] / sum(Ns)]
    outs = list(itertools.product(*[_comps(n, 3) for n in ns]))

    def iv(o):
        st = [_st(W[h], ns[h], Ns[h], None, [0.0, 0.5, 1.0], [o[h][1], o[h][2], o[h][0]]) for h in range(2)]
        lo, hi = bet_strata(st, alpha=alpha)
        est = sum(W[h] * (o[h][0] - o[h][1]) / ns[h] for h in range(2))
        var = 0.0
        for h in range(2):
            d = [1.0] * o[h][0] + [-1.0] * o[h][1] + [0.0] * o[h][2]
            var += W[h] ** 2 * (1 - ns[h] / Ns[h]) * float(np.var(d, ddof=1)) / ns[h]
        return {"betting_without_replacement": (2 * lo - 1, 2 * hi - 1),
                "normal_NOMINAL": (est - Z * math.sqrt(var), est + Z * math.sqrt(var))}
    ivs = {o: iv(o) for o in outs}
    pops = list(itertools.product(*[_comps(N, 3) for N in Ns]))
    prob = lambda pop, o: float(np.prod([_mvh(pop[h], o[h]) for h in range(2)]))
    theta = lambda pop: sum(W[h] * (pop[h][0] - pop[h][1]) / Ns[h] for h in range(2))
    return {"N": list(Ns), "n": list(ns), **_tally(ivs, outs, pops, prob, theta, ("betting_without_replacement",),
                                                   alpha)}


def enum_m2(alpha):
    """User's accuracy with one pure stratum (every unit mapped as the class) and one mixed stratum: the pure-stratum
    rule, the uncapped betting ratio and betting_widths' capped one."""
    Ns, ns = (10, 8), (5, 4)
    W = [Ns[0] / sum(Ns), Ns[1] / sum(Ns)]
    gamma = alpha / 10
    outs = [(ko, om) for ko in range(ns[0] + 1) for om in _comps(ns[1], 3)]

    def iv(o):
        ko, om = o
        kA, kB = np.array([ko, om[0]], float), np.array([ns[0] - ko, om[1]], float)
        st = [{"W": W[0], "n": ns[0], "N": Ns[0], "Nlo": None, "counts": [ko, ns[0] - ko, 0]},
              {"W": W[1], "n": ns[1], "N": Ns[1], "Nlo": None, "counts": list(om)}]
        return {"m2_pure_stratum": m2_ua_arrays(np.array(Ns), np.array(ns), np.array(W), kA, kB, 0, [1], alpha, gamma),
                "betting_ratio_uncapped": ratio_interval_mm(st, alpha),
                "betting_ratio_design1_capped": tuple(bw.ratio_interval(st, alpha)[:2]),
                "normal_NOMINAL": normal_ratio_counts(W, ns, Ns, [(ko, ns[0] - ko, 0), om])}
    ivs = {o: iv(o) for o in outs}
    pops = [(K, P) for K in range(Ns[0] + 1) for P in _comps(Ns[1], 3)]
    prob = lambda pop, o: _hyp(Ns[0], pop[0], ns[0], o[0]) * _mvh(pop[1], o[1])
    theta = lambda pop: (W[0] * pop[0] / Ns[0] + W[1] * pop[1][0] / Ns[1]) / (W[0] + W[1] * (pop[1][0] + pop[1][1]) / Ns[1])
    return {"N": list(Ns), "n": list(ns), "gamma": gamma,
            **_tally(ivs, outs, pops, prob, theta,
                     ("m2_pure_stratum", "betting_ratio_uncapped", "betting_ratio_design1_capped"), alpha)}


def enum_ratio_mm(alpha):
    """The uncapped ratio interval: coverage over every population and sample, and its certified ends never beyond
    the first (last) value a dense grid of single-point tests does not reject."""
    out = {}
    for Ns, ns, W in (((4, 4), (2, 2), (0.7, 0.3)), ((5, 3), (3, 2), (0.5, 0.5))):
        outs = list(itertools.product(*[_comps(n, 3) for n in ns]))
        grid = np.linspace(0.0, 1.0, 101)
        viol = 0
        ivs = {}
        for o in outs:
            st = [{"W": W[h], "n": ns[h], "N": Ns[h], "Nlo": None, "counts": list(o[h])} for h in range(2)]
            lo, hi = ratio_interval_mm(st, alpha)
            ivs[o] = {"betting_ratio_uncapped": (lo, hi)}
            if sum(s["counts"][0] + s["counts"][1] for s in st) == 0:
                continue
            cs = bw.c_grid(st, alpha, cap=False)
            thr = math.log(2.0 / alpha)
            first_ok = next((float(r) for r in grid if ratio_logE_mm(st, cs, "low", r, r) < thr), 1.0)
            last_ok = next((float(r) for r in grid[::-1] if ratio_logE_mm(st, cs, "high", r, r) < thr), 0.0)
            viol += int(lo > first_ok + 1e-12) + int(hi < last_ok - 1e-12)
        pops = list(itertools.product(*[_comps(N, 3) for N in Ns]))
        prob = lambda pop, o: float(np.prod([_mvh(pop[h], o[h]) for h in range(2)]))

        def theta(pop):
            tz = sum(W[h] * (pop[h][0] + pop[h][1]) / Ns[h] for h in range(2))
            return None if tz <= 0 else sum(W[h] * pop[h][0] / Ns[h] for h in range(2)) / tz
        r = _tally(ivs, outs, pops, prob, theta, ("betting_ratio_uncapped",), alpha)
        r["certified_end_beyond_dense_grid"] = viol
        r["holds"] = r["holds"] and viol == 0
        out[f"N{Ns}_n{ns}_W{W}"] = r
    out["holds"] = all(v["holds"] for v in out.values() if isinstance(v, dict))
    return out


def brute_mm(counts, a, b, lam, N, r1, r2, kappa=None):
    """stratum_logavg_mm for one lambda by listing every distinct order of the labels."""
    kappa = bw.KAPPA if kappa is None else kappa
    cats = [j for j, k in enumerate(counts) for _ in range(k)]
    finite = math.isfinite(N)
    tot = []
    for seq in set(itertools.permutations(cats)):
        lg = S0 = Sb = 0.0
        for t, j in enumerate(seq):
            lt = lam * ((N - t) / N if finite else 1.0)
            rb = ((t + 1) * kappa / lt - 0.5 - S0) / Sb if Sb != 0 else r1
            vals = []
            for r in (r1, r2, min(max(rb, r1), r2)):
                S = S0 + Sb * r
                cen = min((0.5 + S) / (t + 1), kappa / lt)
                tail = S / (N - t) if finite else 0.0
                vals.append(math.log1p(lt * (a[j] + b[j] * r - cen)) + lt * (cen + tail))
            lg += min(vals)
            S0 += a[j]
            Sb += b[j]
        tot.append(lg)
    return bw.logsumexp(tot) - math.log(len(tot))


def check_mm():
    """The uncapped ratio engine: the dynamic programme against every order listed, and the bound never above the
    plain e-value at a point of its interval (both computed by listing every order)."""
    worst_dp = worst_bound = 0.0
    for n in range(1, 6):
        for smp in bw.compositions(n):
            for lam in (0.3, 1.0, 3.0):
                for N in (n, n + 2, math.inf):
                    for side in ("low", "high"):
                        a, b = bw.RATIO_AB[side]
                        for r1, r2 in ((0.3, 0.5), (0.0, 1.0), (0.62, 0.62), (0.05, 0.2)):
                            dp = float(stratum_logavg_mm(list(smp), a, b, np.array([lam]), N, n, r1, r2)[0])
                            br = brute_mm(list(smp), a, b, lam, N, r1, r2)
                            worst_dp = max(worst_dp, abs(dp - br))
                            for r in np.linspace(r1, r2, 5):
                                worst_bound = max(worst_bound, br - brute_mm(list(smp), a, b, lam, N, float(r), float(r)))
    return {"dynamic_programme_vs_every_order_max_abs_difference": worst_dp,
            "bound_above_the_evalue_inside_its_interval_max": worst_bound,
            "holds": worst_dp < 1e-9 and worst_bound <= 1e-12}


def enum_all():
    t0 = time.time()
    out = {"note": ("alpha 0.1 and 0.3: at 0.05 the threshold 40 is out of reach of these tiny samples and every "
                    "interval is [0, 1]; a form marked NOT_CLAIMED or NOMINAL is shown, not tested")}
    say("enum: the uncapped ratio engine against brute force")
    out["mm_engine"] = check_mm()
    for alpha in (0.1, 0.3):
        blk = out[f"alpha_{alpha:g}"] = {}
        for name, f in (("areas_mode_drawn_without_replacement", enum_areas), ("census_and_sampled", enum_census),
                        ("three_strata_weights_not_proportional_one_label", enum_three),
                        ("unknown_values_bounded", enum_unknown), ("difference", enum_difference),
                        ("pure_stratum_rule_and_ratios", enum_m2), ("ratio_uncapped", enum_ratio_mm)):
            t = time.time()
            try:
                blk[name] = f(alpha)
            except Exception:
                blk[name] = {"holds": False, "error": traceback.format_exc()}
            blk[name]["seconds"] = time.time() - t
            say(f"enum alpha {alpha:g}: {name} holds {blk[name].get('holds')} ({blk[name]['seconds']:.0f} s)")
    out["holds"] = out["mm_engine"]["holds"] and all(v.get("holds") is True for k, blk in out.items()
                                                     if k.startswith("alpha_") for v in blk.values())
    out["seconds"] = time.time() - t0
    return out


# ------------------------------------------------------------------ mc: plug-in coverage at the agency shapes
_MC = {}
_MCJ = []


def mc_linear_iid(D, x):
    H, n, W = D["H"], D["n"], D["W"]
    p = np.array([x[D["masks"][h]].mean() for h in range(H)])
    theta = float((W * p).sum())

    def draw(rng):
        k = rng.binomial(n, p)
        st = [_st(W[h], n[h], math.inf, None, [0, 1], [n[h] - k[h], k[h]]) for h in range(H)]
        lo, hi = bet_strata(st)
        nlo, nhi = _normal_binary(list(W), list(n), [math.inf] * H, list(k))
        return {"betting": (lo <= theta <= hi, hi - lo), "normal": (nlo <= theta <= nhi, nhi - nlo)}
    return {"theta": theta, "draw": draw, "model": "points drawn independently (point frame), each stratum's rate as observed"}


def mc_diff_iid(D, a, b):
    H, n, W = D["H"], D["n"], D["W"]
    pp = np.array([((a == 1) & (b == 0))[D["masks"][h]].mean() for h in range(H)])
    pm = np.array([((a == 0) & (b == 1))[D["masks"][h]].mean() for h in range(H)])
    rest = np.clip(1 - pp - pm, 0.0, 1.0)
    probs = np.stack([pp, pm, rest], 1)
    probs = probs / probs.sum(1, keepdims=True)
    theta = float((W * (pp - pm)).sum())

    def draw(rng):
        k = [rng.multinomial(int(n[h]), probs[h]) for h in range(H)]
        st = [_st(W[h], n[h], math.inf, None, [0.0, 0.5, 1.0], [k[h][1], k[h][2], k[h][0]]) for h in range(H)]
        lo, hi = bet_strata(st)
        lo, hi = 2 * lo - 1, 2 * hi - 1
        est = var = 0.0
        for h in range(H):
            kp, km = int(k[h][0]), int(k[h][1])
            mean = (kp - km) / n[h]
            est += W[h] * mean
            if n[h] > 1:
                var += W[h] ** 2 * ((kp + km - n[h] * mean ** 2) / (n[h] - 1)) / n[h]
        hw = Z * math.sqrt(max(var, 0.0))
        return {"betting": (lo <= theta <= hi, hi - lo), "normal": (est - hw <= theta <= est + hw, 2 * hw)}
    return {"theta": theta, "draw": draw, "model": "points drawn independently, each stratum's (+1, -1, 0) shares as observed"}


def mc_linear_finite(D, x, f):
    H, n, W = D["H"], D["n"], D["W"]
    Np = f * n
    K = np.array([int(round(x[D["masks"][h]].mean() * Np[h])) for h in range(H)])
    theta = float((W * K / Np).sum())

    def draw(rng):
        k = rng.hypergeometric(K, Np - K, n)
        st = [_st(W[h], n[h], float(Np[h]), None, [0, 1], [n[h] - k[h], k[h]]) for h in range(H)]
        lo, hi = bet_strata(st)
        nlo, nhi = _normal_binary(list(W), list(n), [float(v) for v in Np], list(k))
        return {"betting": (lo <= theta <= hi, hi - lo), "normal": (nlo <= theta <= nhi, nhi - nlo)}
    return {"theta": theta, "draw": draw, "model": f"each stratum a population of {f} x n_h units drawn without replacement"}


def mc_ratio_finite(D, c, what, f):
    H, n, W = D["H"], D["n"], D["W"]
    m, r = D["map"], D["rstar"]
    A = (m == c) & (r == c)
    B = (m == c) & (r != c) if what == "user" else (m != c) & (r == c)
    Np = f * n
    pops = []
    for h in range(H):
        mk = D["masks"][h]
        Ah, Bh = int(round(A[mk].mean() * Np[h])), int(round(B[mk].mean() * Np[h]))
        pops.append((Ah, Bh, int(Np[h]) - Ah - Bh))
    theta = (sum(W[h] * pops[h][0] / Np[h] for h in range(H))
             / sum(W[h] * (pops[h][0] + pops[h][1]) / Np[h] for h in range(H)))
    own, mixed = kinds_for(D, c) if what == "user" else (None, None)

    def draw(rng):
        cts = [rng.multivariate_hypergeometric(np.array(pops[h], dtype=np.int64), int(n[h])) for h in range(H)]
        st = [{"W": float(W[h]), "n": int(n[h]), "N": float(Np[h]), "Nlo": None, "counts": [int(v) for v in cts[h]]}
              for h in range(H)]
        lo, hi = ratio_interval_mm(st)
        res = {"betting": (lo <= theta <= hi, hi - lo)}
        nlo, nhi = normal_ratio_counts(list(W), list(n), [float(v) for v in Np], [list(map(int, ct)) for ct in cts])
        res["normal"] = (nlo <= theta <= nhi, nhi - nlo)
        if own is not None:
            kA = np.array([ct[0] for ct in cts], float)
            kB = np.array([ct[1] for ct in cts], float)
            ml, mh = m2_ua_arrays(Np.astype(np.int64), n, W, kA, kB, own, mixed)
            res["m2"] = (ml <= theta <= mh, mh - ml)
        return res
    return {"theta": theta, "draw": draw,
            "model": f"each stratum a population of {f} x n_h units with the observed category shares, drawn without replacement"}


def _mc_job(i):
    name, rep = _MCJ[i]
    c = _MC[name]
    rng = np.random.default_rng([117, c["index"], rep])
    try:
        return name, c["draw"](rng)
    except Exception:
        return name, {"error": traceback.format_exc()}


def mc(rec, areas, R):
    t0 = time.time()
    gfc, nl = load_gfc(rec, areas), load_nlcd(rec, 2016)
    ab()
    okg = (gfc["gfc"] == gfc["ref"]).astype(float)
    okw = (gfc["wc"] == gfc["ref"]).astype(float)
    ap = (nl["map"] == nl["ref"]).astype(float)
    aa = ((nl["map"] == nl["ref"]) | (nl["map"] == nl["alt"])).astype(float)
    cases = {"gfc_oa_point_frame": mc_linear_iid(gfc, okg),
             "gfc_minus_worldcover_point_frame": mc_diff_iid(gfc, okg, okw),
             "nlcd_oa_primary_or_alternate_point_frame": mc_linear_iid(nl, aa),
             "nlcd_oa_primary_population_10n": mc_linear_finite(nl, ap, 10),
             "nlcd_user_52_population_10n": mc_ratio_finite(nl, 52, "user", 10),
             "nlcd_user_23_population_10n": mc_ratio_finite(nl, 23, "user", 10),
             "nlcd_producer_52_population_10n": mc_ratio_finite(nl, 52, "producer", 10)}
    for i, (k, v) in enumerate(cases.items()):
        v["index"] = i
        _MC[k] = v
    _MCJ[:] = [(k, rep) for k in cases for rep in range(R)]
    raw = {k: [] for k in cases}
    say(f"mc: {len(_MCJ)} draws on {PROCS} workers")
    with multiprocessing.get_context("fork").Pool(PROCS) as pool:
        for name, res in pool.imap_unordered(_mc_job, range(len(_MCJ)), chunksize=4):
            raw[name].append(res)
    se = math.sqrt(0.95 * 0.05 / R)
    out = {"draws": R, "alpha": ALPHA, "gate": 0.95 - 3 * se,
           "gate_note": "coverage at least 0.95 - 3 Monte Carlo standard errors at 0.95"}
    for k, rows in raw.items():
        errs = [r["error"] for r in rows if "error" in r]
        good = [r for r in rows if "error" not in r]
        blk = {"theta": cases[k]["theta"], "model": cases[k]["model"], "errors": len(errs),
               "first_error": errs[0] if errs else None}
        for m in ("betting", "m2", "normal"):
            v = [r[m] for r in good if m in r]
            if v:
                blk[m] = {"coverage": float(np.mean([bool(x[0]) for x in v])),
                          "mean_width_points": float(100 * np.mean([x[1] for x in v])), "draws": len(v)}
        blk["holds"] = (not errs) and all(blk[m]["coverage"] >= out["gate"] for m in ("betting", "m2") if m in blk)
        out[k] = blk
    out["holds"] = all(v["holds"] for v in out.values() if isinstance(v, dict) and "holds" in v)
    out["seconds"] = time.time() - t0
    return out


# ------------------------------------------------------------------ select: the rule fixed before the run
BSC_GATES = {"E1_share_counts": "chernoff_boxed", "E1b_share_counts_unequal": "chernoff_boxed",
             "E2_share_binomial_three_strata": "chernoff_boxed",
             "E3_share_areas_mode_drawn_without_replacement": "chernoff_boxed",
             "E4_ratio_counts": "chernoff_boxed", "E5_difference_counts": "chernoff_boxed",
             "E6_unassigned": "bounded_boxed"}


def _nodes(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _nodes(v)


def select(paths):
    names = ("betting_selftest", "judge_enum", "judge_mc", "judge_widths", "bounds_enum")
    J = {}
    for nm, p in zip(names, list(paths) + [None] * (5 - len(paths))):
        try:
            J[nm] = json.load(open(p))
        except Exception:
            J[nm] = None
    gates = {}
    s = (J["betting_selftest"] or {}).get("selftest") or {}
    gates["betting_selftest"] = bool(s) and s.get("evalue_property_holds") is True \
        and s.get("dynamic_programme_vs_every_order_max_abs_difference", 1.0) < 1e-9 \
        and all(v.get("holds") is True for v in (s.get("linear_min_coverage") or {}).values()) \
        and all(v.get("holds") is True for v in (s.get("ratio_min_coverage") or {}).values())
    gates["judge_enum"] = (J["judge_enum"] or {}).get("holds") is True
    gates["judge_mc"] = (J["judge_mc"] or {}).get("holds") is True
    jw = J["judge_widths"] or {}
    gates["data_reading_checks"] = (jw.get("checks") or {}).get("all_ok") is True
    gates["widths_without_errors_or_misses"] = (jw.get("checks") or {}).get("no_errors_or_misses") is True
    be = J["bounds_enum"] or {}
    gates["bsc_enum"] = bool(be) and all(((be.get(k) or {}).get(m) or {}).get("min_coverage", 0.0) >= 0.95 - 1e-9
                                         for k, m in BSC_GATES.items())
    fam = {"linear": [], "ratio": []}
    over_normal = {"linear": [], "ratio": []}
    m2r = []
    for node in _nodes({k: v for k, v in jw.items() if k in ("gfc2020", "nlcd2016_B_2016", "nlcd2016_B_2011")}):
        f = node.get("family")
        if f in fam and node.get("betting_over_bsc") is not None:
            fam[f].append(node["betting_over_bsc"])
        if f in over_normal and (node.get("x_normal") or {}).get("betting") is not None:
            over_normal[f].append(node["x_normal"]["betting"])
        if node.get("m2_over_betting") is not None:
            m2r.append(node["m2_over_betting"])
    med = {k: (float(np.median(v)) if v else None) for k, v in fam.items()}
    betting_valid = gates["betting_selftest"] and gates["judge_enum"] and gates["judge_mc"]
    verdict = {}
    for f in ("linear", "ratio"):
        if betting_valid and (not gates["bsc_enum"] or med[f] is None or med[f] <= WIDTH_RULE):
            verdict[f] = "betting"
        elif gates["bsc_enum"]:
            verdict[f] = "boxed stratified Chernoff (the fallback)"
        else:
            verdict[f] = "neither: keep the union bound and report"
    return {"rule": (f"per family (linear quantities; ratios): the betting interval unless a validity gate fails or its "
                     f"median width over the family exceeds the fallback's by more than {WIDTH_RULE - 1:.0%}; then the "
                     "fallback if its own exact gates hold; else neither. Fixed before the run."),
            "gates": gates, "verdict": verdict, "data_reading_is_a_precondition": gates["data_reading_checks"],
            "median_betting_over_bsc": med, "count_compared": {k: len(v) for k, v in fam.items()},
            "median_betting_over_normal": {k: (float(np.median(v)) if v else None) for k, v in over_normal.items()},
            "largest_betting_over_normal": {k: (float(np.max(v)) if v else None) for k, v in over_normal.items()},
            "median_m2_over_betting_user_accuracy": float(np.median(m2r)) if m2r else None,
            "table4_largest_absolute_difference_points": (jw.get("table4_nlcd2016_level2") or {}).get(
                "largest_absolute_difference_points"),
            "e2e": jw.get("e2e")}


# ------------------------------------------------------------------ main
def _json(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def main():
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "select":
        out = select(a[1:6])
    else:
        if len(a) < 3:
            raise SystemExit(__doc__)
        rec, areas, mode = a[0], a[1], a[2]
        if mode == "widths":
            out = widths(rec, areas)
        elif mode == "enum":
            out = enum_all()
        elif mode == "mc":
            out = mc(rec, areas, int(a[3]) if len(a) > 3 else 200)
        else:
            raise SystemExit(f"unknown mode {mode!r}: widths, enum, mc or select")
    json.dump(out, sys.stdout, indent=1, default=_json)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
