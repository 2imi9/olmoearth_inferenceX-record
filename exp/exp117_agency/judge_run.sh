#!/bin/bash
#SBATCH -J oeix-agency-judge
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 03:30:00
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -u
SRC=/scratch/qi_zim_neu/olmoearth_inferenceX/agency/$SLURM_JOB_ID/src
mkdir -p "$SRC" || exit 1
cat > "$SRC/judge_agency.py" <<'OEIX_FILE_END'
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
OEIX_FILE_END
cat > "$SRC/betting_widths.py" <<'OEIX_FILE_END'
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
OEIX_FILE_END
cat > "$SRC/agency_bounds.py" <<'OEIX_FILE_END'
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
OEIX_FILE_END
cat > "$SRC/strata_size.csv" <<'OEIX_FILE_END'
Strata,strata_ha
1001,18199738
1002,20978746
1003,222478929
1004,1019336791
1006,177268821
1008,27239648
1014,1368465
1015,8616809
1101,46741948
1102,91221495
1103,765162954
1104,206951698
1106,310518525
1114,6217819
1200,4899964
1210,22186007
1221,7224973
1222,27159363
1223,2512686
2001,248837
2002,412896
2003,12131857
2004,48469650
2006,210295297
2008,33488325
2014,42731903
2015,10360753
2025,330065716
2027,511252597
2029,194923270
2030,8795866
2101,2454939
2102,6763167
2103,61462759
2104,19071612
2106,47291256
2114,148930873
2125,154989894
2200,3684946
2210,258777480
2221,10410148
2222,100177481
2223,20090541
2225,51433694
4001,88495150
4002,3881870
4003,9312149
4004,278719706
4006,168962575
4008,54402776
4014,6543421
4015,401298
4025,15995277
4027,14113270
4029,25657290
4101,147213670
4102,71031530
4103,291283066
4104,447407601
4106,393994366
4114,302323312
4125,180161248
4200,41426256
4210,58305508
4221,36700717
4222,45744409
4223,4215300
4224,58379099
4226,12752897
5004,32453
5006,4634460
5008,53256056
5014,505189
5015,103229504
5025,3559792
5027,84437699
5029,12000744
5104,658361
5106,15920035
5114,24483528
5125,227547165
5200,5094508
5210,24379285
5221,13759882
5222,12365144
5223,3431287
5225,3678405
5226,2436873
5227,221966
6004,83234378
6006,178654915
6014,8068
6025,221229995
6027,862483136
6029,81824612
6030,1514071
6040,151154050
6104,437580
6106,19093034
6114,6178122
6125,259879639
6200,1666662
6210,38417027
6221,4052021
6222,67418645
6223,78292697
6224,57716552
6226,2808326
6227,20069962
7001,69667
7002,737654
7003,61845927
7004,351988233
7006,230518975
7008,21013743
7014,40039936
7015,54784644
7101,5244463
7102,987250
7103,1147552
7104,127951
7106,1313840
7114,7137182
7200,418344
7210,12572324
7222,2990168
7223,598730
7225,10339445
8001,216602905
8002,238809747
8003,357151992
8004,83827636
8006,106394804
8008,13361828
8014,78752669
8015,26720535
8029,5966862
8101,65281124
8102,72740572
8103,256411118
8104,30497009
8106,73915576
8114,139262097
8200,3113906
8210,33249600
8221,4898964
8222,19457237
8223,14318643
8226,3555721
OEIX_FILE_END
AREAS="$SRC/strata_size.csv"
[ -f "$AREAS" ] || AREAS=/scratch/qi_zim_neu/olmoearth_inferenceX/agency/1523908/strata_size.csv
[ -f "$AREAS" ] || { echo "no strata_size.csv: copy data/breadth/gfc2020/strata_size.csv into $SRC"; exit 1; }
for f in judge_agency.py betting_widths.py agency_bounds.py; do
  [ -f "$SRC/$f" ] || { echo "missing $SRC/$f"; exit 1; }
done
export UV_CACHE_DIR=/scratch/qi_zim_neu/olmoearth_inferenceX/uv-cache
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BETTING_PY="$SRC/betting_widths.py" BOUNDS_PY="$SRC/agency_bounds.py"
D=/scratch/qi_zim_neu/olmoearth_inferenceX/agency/$SLURM_JOB_ID
export JUDGE_PARTIAL="$D/judge_widths.partial.jsonl"
mkdir -p "$D" && cd "$D" || exit 1
git clone -q --depth 1 https://github.com/2imi9/olmoearth_inferenceX.git pkg || exit 1
git clone -q --depth 1 https://github.com/2imi9/olmoearth_inferenceX-record.git rec || exit 1
cd pkg || exit 1
~/.local/bin/uv sync -q --frozen --python 3.12 || exit 1
PY="$HOME/.local/bin/uv run --frozen --no-sync python -I"

# 1. the chosen interval's exact self-tests (betting_widths.py: DP against every order, the e-value property,
#    linear and ratio coverage by enumeration). Any failure stops the job.
timeout 2400 $PY "$SRC/betting_widths.py" "$D/rec" "$AREAS" --selftest-only > "$D/betting_selftest.json" 2> "$D/betting_selftest.err"
echo "betting selftest exit $?"
$PY - "$D/betting_selftest.json" <<'PY' || { echo "BETTING SELFTEST FAILED: no width is quoted"; exit 2; }
import json, sys
s = json.load(open(sys.argv[1]))["selftest"]
ok = (s["evalue_property_holds"] is True and s["dynamic_programme_vs_every_order_max_abs_difference"] < 1e-9
      and all(c["holds"] for c in s["linear_min_coverage"].values())
      and all(c["holds"] for c in s["ratio_min_coverage"].values()))
print("betting selftest holds:", ok, "| largest E[e-value]:", s["largest_expected_evalue"])
sys.exit(0 if ok else 1)
PY

# 2. the judge's exact enumerations (uncapped ratio engine, areas mode, unknowns, difference, pure-stratum rule)
timeout 1500 $PY "$SRC/judge_agency.py" "$D/rec" "$AREAS" enum > "$D/judge_enum.json" 2> "$D/judge_enum.err"
echo "judge enum exit $?"
# 3. the fallback's exact enumerations (agency_bounds.py)
timeout 2400 $PY "$SRC/agency_bounds.py" "$D/rec" "$AREAS" enum > "$D/bounds_enum.json" 2> "$D/bounds_enum.err"
echo "bounds enum exit $?"
# 4. every quantity, every interval (partial results in judge_widths.partial.jsonl if the limit is reached)
timeout 4200 $PY "$SRC/judge_agency.py" "$D/rec" "$AREAS" widths > "$D/judge_widths.json" 2> "$D/judge_widths.err"
echo "judge widths exit $?"
# 5. plug-in coverage at the agency designs' shapes
timeout 1800 $PY "$SRC/judge_agency.py" "$D/rec" "$AREAS" mc 200 > "$D/judge_mc.json" 2> "$D/judge_mc.err"
echo "judge mc exit $?"
# 6. the verdict, by the rule fixed before the run
$PY "$SRC/judge_agency.py" select "$D/betting_selftest.json" "$D/judge_enum.json" "$D/judge_mc.json" \
    "$D/judge_widths.json" "$D/bounds_enum.json" > "$D/judge_select.json" 2> "$D/judge_select.err"
echo "select exit $?"
cat "$D/judge_select.json"
du -sh "$D"
