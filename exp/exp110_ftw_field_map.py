"""exp110: FTW's 2nd-edition global field map (FTW Global Data 2e), read as the package reads any probability map, at
the FTW benchmark v1 test chips of Belgium, Croatia, Slovenia and Sweden.

Preregistration: docs/plan/ftw_fields.md, frozen on 14 October 2026; the page is the specification. The grade is
refused until the page is frozen; --smoke runs the whole pipeline on synthetic chips anywhere and touches no data.

The reading (the page's Units). Each country's chips side by side in sorted aoi_id order, one row of 256 x 256 chips
on each chip's own mask grid. The map's extent probability p = field + boundary per mask pixel, read as a two-class
probability map by the package at 1.12.1: pixel class p > 0.5, pixel confidence 2|p - 0.5|, window (8 x 8 px) confidence
the mean over its 64 pixels, window class the majority of its pixels with a tie to the more confident pixels. Truth: the
majority of the window's 64 pixels in the 2-class mask; an evenly split window has no majority and is left out. A window
is wrong when the map's class differs from the truth's. Populations: each country's windows with a truth majority, and
the four pooled (the concatenation of the countries' populations).

Order of work (the page). First, for every country and with no label read: the map's window confidence quantiles and
class shares, the no-model controls, PRUE B3's window classes in the stated band order and with the bands reversed, and
D (the windows where the two maps differ). The two B3 stop rules are applied then, before any label is read with B3's
map. Then the label-only and metadata checks, and the arms:
  arm 1  `assess --patch 8 --reference` on each country's mosaic through the command line (its numbers checked
         against the Python arrays); AUROC of minus the window confidence for the window errors (metrics.weighted_auroc,
         unit weights, ties half), capture at 10% in expectation over ties (metrics.capture_at_budget_expected), excess
         AURC; the controls (pixel variance, predicted-class rarity) and the map-only cues; 90% chip-cluster bootstrap
         intervals (2,000 resamples by chip, seed 110) for the AUROC and the lead over the best control
  arm 2  the exact hypergeometric interval at B = 100 and 300: exact coverage (estimate.exact_coverage_srs with
         estimate.hypergeom_interval) and 2,000 draws per cell through sample_for_estimation(design="random") and
         estimate_error_rate, draw r with seed 110 + r; the chip-cluster design (tiles of one chip, 16 windows per chip)
         reported, by 2,000 draws
  arm 3  certify on the B = 300 samples of arm 2 at alpha 0.05 (graded) and 0.10 (reported), delta 0.10, the prefix
         rule and the standard level cut. certify_zone's own steps (zone_order, zone_levels, zone_counts, zone_pvalue,
         apply_zone_rule, and its review-set guard) with the zone order and the guard's percentiles computed once per
         population; certify_zone itself is called on the first VERIFY draws of each population and must agree
  arm 4  2e (map a) against PRUE B3 (map b) on the windows with a truth majority where both maps predict, without the
         windows holding a pixel zero in every band of either S2 window: D, K_A, K_B from the full truth; the exact
         probabilities of plan.which_map_probability at n = 100 and 300 (the verdict estimate prints, rule auto), and
         2,000 draws per cell through sample_disagreement and compare_from_disagreement
The verdicts are the page's bars, applied by `verdicts` and `b3_stop_rules` below.

Outputs: exp/out/exp110_summary.json, aggregates only (per country and pooled) with the inputs' sha256. No per-window or
per-chip array, no chip identifier and no coordinate is written to the repository.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

import oe_inferencex  # noqa: E402
from oe_inferencex import assess as A, estimate as est, metrics, plan  # noqa: E402

PLAN = os.path.join(ROOT, "docs", "plan", "ftw_fields.md")
OUT = os.path.join(EXP_DIR, "out")
SUMMARY = "exp110_summary.json"

# ----------------------------------------------------------------------------- the page's constants
COUNTRIES = ("belgium", "croatia", "slovenia", "sweden")
MAP_YEAR = {"belgium": 2021, "croatia": 2023, "slovenia": 2021, "sweden": 2021}
CHIPS = {"belgium": 198, "croatia": 353, "slovenia": 228, "sweden": 516}          # the page's Data table
SEAM_CHIPS = {"belgium": 0, "croatia": 4, "slovenia": 0, "sweden": 9}
CHIP = 256
PATCH, PATCH_REPORTED = 8, 16                         # 8 px graded, 16 px reported
WINDOWS = {c: n * (CHIP // PATCH) ** 2 for c, n in CHIPS.items()}                # 202,752 ... (1,326,080 in all)
BANDS = ("B04", "B03", "B02", "B08")                  # stored order of each S2 window
ALPHA, ALPHA_REPORTED, DELTA = 0.05, 0.10, 0.10
BUDGETS, CERTIFY_BUDGET = (100, 300), 300
DRAWS, SEED, BOOT, BOOT_LEVEL = 2000, 110, 2000, 0.90
CAPTURE_BUDGET = 0.10
TILE_WINDOWS, PER_TILE = 32, 16                       # --tile 32: one chip per tile; 16 windows per chip
DILATE_NS, DILATE_EW = 1, 2                           # the lenient reading: 7 m, about 1 mask pixel N-S and 2 E-W
# bars
P1_AUROC, P1_CAPTURE, P1_LEAD, P1_COUNTRIES, P1_ERROR_FLOOR = 0.70, 0.25, 0.05, 3, 200
P2_COVERAGE, P2_MC_SE = 0.95, 3.0
P3_VIOLATION, P3_SHARE, P3_COUNTRIES = 0.115, 0.25, 2
P4_WRONG, P4_RIGHT, P4_COUNTRIES, P4_HEADLINE = 0.025, 0.8, 3, 100
B3_AGREE_STOP = 0.70
B3_SWAP = (2, 1, 0, 3)                                # B04, B03, B02, B08 -> B02, B03, B04, B08 (the control's order)
# report-only
VERIFY = 20                                           # draws per population on which certify_zone itself is called
POWER_REPORTED = 0.8
LADDER = (50, 100, 200, 400, 800, 1600, 3200, 6400)
B3_TRAINING_NOTE = ("The page's arm-4 reading says PRUE B3 never trained on these four countries. The B3 run's report "
                    "(14 October 2026, after freezing) found that the v3.1 release notes and prue_efnet3.yaml point the "
                    "other way: B3 most likely trained on these countries' train chips, as the 2e map's model did. The "
                    "checkpoint stores no training list. P4's arithmetic does not depend on it; its reading does, and "
                    "the owner has not settled it.")


def prereg_status(path=PLAN):
    with open(path, encoding="utf-8") as f:
        return "frozen" if "**Status: frozen" in f.read() else "draft"


def log(*a):
    print(*a, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def git_state():
    try:
        commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return commit, dirty


# ----------------------------------------------------------------------------- layout: chips side by side
def row_mosaic(chips):
    """(n, H, W) or (n, C, H, W) chips -> (H, n W) or (C, H, n W): the chips side by side in the given order."""
    a = np.asarray(chips)
    if a.ndim == 3:
        n, h, w = a.shape
        return a.transpose(1, 0, 2).reshape(h, n * w)
    n, c, h, w = a.shape
    return a.transpose(1, 2, 0, 3).reshape(c, h, n * w)


def unmosaic(win, n):
    """A window grid of the row mosaic, (h, n w) -> (n, h, w) per chip."""
    h, nw = win.shape
    return np.ascontiguousarray(win.reshape(h, n, nw // n).transpose(1, 0, 2))


def flat(per_chip):
    """Per-chip window arrays (n, h, w) flattened in the mosaic's raster order (row-major over the window grid of
    the row mosaic), the order the package's flattened window grid has."""
    return row_mosaic(per_chip).ravel()


# ----------------------------------------------------------------------------- readings by the package
def read_extent(p_chips, patch, chunk=None):
    """The package's reading of a two-class probability map: assess_prediction on the row mosaic of the chips'
    extent probabilities. Returns per-chip window arrays (confidence, class, boundary indicator) and the reading's
    label-free summary. `chunk` reads that many chips per mosaic (window values do not depend on the layout; the
    boundary indicator does at the seams, and is read only from a whole-country mosaic)."""
    p = np.asarray(p_chips, dtype=np.float64)
    n = len(p)
    if chunk is None or chunk >= n:
        out = A.assess_prediction(row_mosaic(p), is_logit=False, patch=patch, budgets=(CAPTURE_BUDGET,))
        a = out["arrays"]
        return {"conf": unmosaic(a["confidence"], n), "cls": unmosaic(a["pooled_argmax"], n),
                "boundary": unmosaic(a["boundary"], n), "quantiles": out["confidence_quantiles"],
                "class_share": out["class_share"], "distinct": out["confidence_distinct_pooled"],
                "boundary_window_fraction": out["boundary_window_fraction"]}
    conf, cls = [], []
    for s in range(0, n, chunk):
        out = A.assess_prediction(row_mosaic(p[s:s + chunk]), is_logit=False, patch=patch, budgets=(CAPTURE_BUDGET,))
        m = len(p[s:s + chunk])
        conf.append(unmosaic(out["arrays"]["confidence"], m))
        cls.append(unmosaic(out["arrays"]["pooled_argmax"], m))
    return {"conf": np.concatenate(conf), "cls": np.concatenate(cls)}


def read_three_class(field, boundary, patch, chunk=48):
    """The second measure: (1 - f - b, f, b) as a (3, H, W) probability map, read by assess_prediction (window
    confidence the mean top probability, class the majority of the pixel argmax), a chunk of chips per mosaic."""
    n = len(field)
    conf, cls = [], []
    for s in range(0, n, chunk):
        f = np.asarray(field[s:s + chunk], np.float64)
        b = np.asarray(boundary[s:s + chunk], np.float64)
        sc = np.stack([1.0 - f - b, f, b], axis=1)                     # (m, 3, H, W)
        out = A.assess_prediction(row_mosaic(sc), is_logit=False, patch=patch, budgets=(CAPTURE_BUDGET,))
        conf.append(unmosaic(out["arrays"]["confidence"], len(f)))
        cls.append(unmosaic(out["arrays"]["pooled_argmax"], len(f)))
    return {"conf": np.concatenate(conf), "cls": np.concatenate(cls)}


def truth_windows(mask_chips, patch, n_classes):
    """The majority class of each window of a mask, -1 where the top count is tied: assess's pooling of a reference
    (assess._pooled_argmax with tie=-1), over the mask's own class range."""
    m = row_mosaic(np.asarray(mask_chips, dtype=np.int64))
    k = max(int(m.max()) + 1, n_classes)
    return unmosaic(A._pooled_argmax(m, k, patch, empty=-1, tie=-1), len(mask_chips))


def window_any(pix, patch):
    n, h, w = pix.shape
    return pix.reshape(n, h // patch, patch, w // patch, patch).any(axis=(2, 4))


def dilate_field(mask2, ns=DILATE_NS, ew=DILATE_EW):
    """The lenient truth: the 2-class mask's field pixels dilated within each chip by ns rows (N-S) and ew columns
    (E-W), a box."""
    f = np.asarray(mask2) == 1
    out = f.copy()
    _, H, W = f.shape
    for dy in range(-ns, ns + 1):
        for dx in range(-ew, ew + 1):
            if dy == 0 and dx == 0:
                continue
            out[:, max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] |= \
                f[:, max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
    return out.astype(np.uint8)


# ----------------------------------------------------------------------------- controls and cues (label-free)
def pixel_variance(s2, patch):
    """The pixel-variance control: the mean over the eight bands of the band's variance over the window's pixels,
    each band scaled by its standard deviation over the country's chips, pixels zero in every band left out. A window
    with no pixel left has variance 0. Returns (n, h, w), the windows with no pixel left, the band SDs."""
    s2 = np.asarray(s2)
    n, nb, H, W = s2.shape
    keep = (s2 != 0).any(axis=1)
    g, h = H // patch, W // patch
    k = keep.reshape(n, g, patch, h, patch)
    cnt = k.sum(axis=(2, 4)).astype(np.float64)
    acc = np.zeros((n, g, h))
    sds = []
    for j in range(nb):
        v = s2[:, j].astype(np.float64)
        sd = float(v[keep].std())
        sds.append(sd)
        x = (v / sd).reshape(n, g, patch, h, patch)
        m = (x * k).sum(axis=(2, 4)) / np.maximum(cnt, 1.0)
        dev = (((x - m[:, :, None, :, None]) ** 2) * k).sum(axis=(2, 4))
        acc += np.where(cnt > 0, dev / np.maximum(cnt, 1.0), 0.0)
        del v, x, dev
    return acc / nb, int((cnt == 0).sum()), sds


def rarity(cls):
    """Predicted-class rarity as a suspicion: minus the share of the country's windows the map puts in the window's
    class (rarer = more suspect)."""
    cls = np.asarray(cls)
    out = np.empty(cls.shape)
    for k in np.unique(cls):
        out[cls == k] = -float((cls == k).mean())
    return out


def neighbour_disagreement(cls):
    """The share of the window's neighbours (8-connected) within its chip whose map class differs from its own."""
    cls = np.asarray(cls)
    _, g, h = cls.shape
    diff = np.zeros(cls.shape)
    cnt = np.zeros(cls.shape)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            dst = (slice(None), slice(max(-dy, 0), g + min(-dy, 0)), slice(max(-dx, 0), h + min(-dx, 0)))
            src = (slice(None), slice(max(dy, 0), g + min(dy, 0)), slice(max(dx, 0), h + min(dx, 0)))
            diff[dst] += cls[src] != cls[dst]
            cnt[dst] += 1
    return diff / cnt


# ----------------------------------------------------------------------------- ranking statistics
def rank_stats(u, e, budget=CAPTURE_BUDGET):
    """AUROC (ties half), expected capture at a budget and excess AURC, by the package's metrics."""
    e = np.asarray(e, bool)
    return {"auroc": metrics.weighted_auroc(u, e, np.ones(e.size)),
            "capture_10": metrics.capture_at_budget_expected(u, e, (budget,))[budget],
            "excess_aurc": metrics.excess_aurc(u, e.astype(float))}


def grouped_table(u, e):
    """Groups of equal scores (ascending): values, errors and units per group."""
    vals, inv = np.unique(np.asarray(u, np.float64), return_inverse=True)
    pos = np.bincount(inv.ravel(), weights=np.asarray(e, np.float64).ravel(), minlength=vals.size)
    tot = np.bincount(inv.ravel(), minlength=vals.size).astype(np.float64)
    return vals, pos, tot


def merge_tables(tables):
    vals = np.concatenate([t[0] for t in tables])
    v, inv = np.unique(vals, return_inverse=True)
    pos = np.bincount(inv, weights=np.concatenate([t[1] for t in tables]), minlength=v.size)
    tot = np.bincount(inv, weights=np.concatenate([t[2] for t in tables]), minlength=v.size)
    return v, pos, tot


def grouped_auroc_capture(table, budget=CAPTURE_BUDGET):
    """What metrics.weighted_auroc (unit weights) and metrics.capture_at_budget_expected give, from a group table:
    for pixel-level readings too large to hold the package's arrays at once (checked against them in the run)."""
    _, pos, tot = table
    neg = tot - pos
    P, Nn = pos.sum(), neg.sum()
    auroc = float((pos * (np.cumsum(neg) - neg + 0.5 * neg)).sum() / (P * Nn)) if P > 0 and Nn > 0 else float("nan")
    pd_, td = pos[::-1], tot[::-1]                       # most suspect first
    k = max(1, int(round(budget * td.sum())))
    cum = np.cumsum(td)
    full = cum <= k
    cap = pd_[full].sum()
    part = np.flatnonzero(~full)
    if part.size:
        g = part[0]
        cap += pd_[g] * (k - (cum[g] - td[g])) / td[g]
    return auroc, float(cap / max(P, 1.0)), float(P / tot.sum())


def pair_matrix(u, e, chip, n_chips):
    """M[c, d] = sum over the error windows i of chip c and the correct windows j of chip d of [u_i > u_j] +
    0.5 [u_i == u_j]. With chip multiplicities w, w M w / ((sum w_c E_c)(sum w_c R_c)) is the AUROC of the resampled
    population, exactly as metrics.weighted_auroc gives it with each window weighted by its chip's multiplicity."""
    u = np.asarray(u, np.float64)
    e = np.asarray(e, bool)
    ue, ce = u[e], chip[e]
    ur, cr = u[~e], chip[~e]
    o = np.lexsort((ur, cr))
    ur, cr = ur[o], cr[o]
    lo = np.searchsorted(cr, np.arange(n_chips), "left")
    hi = np.searchsorted(cr, np.arange(n_chips), "right")
    M = np.zeros((n_chips, n_chips))
    for d in range(n_chips):
        S = ur[lo[d]:hi[d]]
        if S.size == 0 or ue.size == 0:
            continue
        lt = np.searchsorted(S, ue, "left")
        le = np.searchsorted(S, ue, "right")
        M[:, d] = np.bincount(ce, weights=lt + 0.5 * (le - lt), minlength=n_chips)
    E = np.bincount(chip[e], minlength=n_chips).astype(np.float64)
    R = np.bincount(chip[~e], minlength=n_chips).astype(np.float64)
    return M, E, R


def matrix_auroc(M, E, R, w):
    return float(w @ M @ w / ((w @ E) * (w @ R)))


def chip_bootstrap(conf, err, ctl, chip, n_chips, boot, seed, level=BOOT_LEVEL):
    """90% chip-cluster bootstrap intervals (resampling chips with replacement) for the AUROC of minus the confidence
    and for its lead over the best control (fixed from the full data)."""
    Mc = pair_matrix(-conf, err, chip, n_chips)
    Mb = pair_matrix(ctl, err, chip, n_chips)
    ones = np.ones(n_chips)
    a0 = metrics.weighted_auroc(-conf, err, np.ones(err.size))
    if abs(matrix_auroc(*Mc, ones) - a0) > 1e-9:
        raise SystemExit("the chip pair matrix does not reproduce metrics.weighted_auroc")
    rng = np.random.default_rng(seed)
    a, lead = np.empty(boot), np.empty(boot)
    for r in range(boot):
        w = np.bincount(rng.integers(0, n_chips, n_chips), minlength=n_chips).astype(np.float64)
        a[r] = matrix_auroc(*Mc, w)
        lead[r] = a[r] - matrix_auroc(*Mb, w)
    q = ((1 - level) / 2, 1 - (1 - level) / 2)
    return {"auroc": [float(np.quantile(a, q[0])), float(np.quantile(a, q[1]))],
            "lead_over_best_control": [float(np.quantile(lead, q[0])), float(np.quantile(lead, q[1]))],
            "resamples": int(boot), "seed": int(seed), "level": level}


# ----------------------------------------------------------------------------- arm 3: certify's own steps
def zone_setup(margin, err):
    """Per population, once: certify's zone order, the full truth along it, and the review-set guard's suspicion
    percentiles exactly as estimate.review_set_check computes them (ties in a fixed random order, generator seed 0)."""
    margin = np.asarray(margin, np.float64)
    N = margin.size
    order, pos = est.zone_order(margin)
    cum = np.concatenate([[0.0], np.cumsum(np.asarray(err, np.float64)[order])])
    o = np.lexsort((np.random.default_rng(0).random(N), -margin))
    ranks = np.empty(N)
    ranks[o] = np.arange(N)
    return {"N": N, "pos": pos, "cum": cum, "pct": (ranks + 0.5) / N}


def certify_steps(Z, idx, wrong, alpha, delta=DELTA):
    """certify_zone(margin, idx, wrong, alpha, delta) with the prefix rule and the standard cut, from Z: the same
    package calls in the same order, its review-set guard included. Returns the certified coverage (None if none),
    the zone's size and its full-truth error rate."""
    idx = np.asarray(idx, int)
    wrong = np.asarray(wrong, np.float64)
    n = idx.size
    if float(np.mean(Z["pct"][idx])) > est.review_set_threshold(n, Z["N"]) + 1e-12:
        return {"coverage": None, "refused": True}
    cov, sizes, _ = est.zone_levels(Z["N"], n, alpha, delta, est.ZONE_GRID, None)
    if not cov:
        return {"coverage": None}
    b, k = est.zone_counts(Z["pos"][idx], wrong, sizes)
    p = np.array([est.zone_pvalue(kk, bb, nn, alpha) for kk, bb, nn in zip(k, b, sizes)])
    _, best = est.apply_zone_rule(p, b, k, alpha, delta, "prefix", n=sizes)
    if best is None:
        return {"coverage": None}
    nz = int(sizes[best])
    return {"coverage": float(cov[best]), "n_zone": nz, "zone_error": float(Z["cum"][nz] / nz)}


# ----------------------------------------------------------------------------- arms 2 and 3
def plan_median_width(N, B, rate):
    """plan's predicted median width of the exact interval at B labels for a map wrong at `rate`: the smallest width,
    among those the interval can take at the counts the draw can give, that plan.error_rate_probability reaches with
    probability at least one half (a bisection; the probability rises with the width)."""
    ks, _ = plan._hyper_pmf(N, int(round(rate * N)), B)
    widths = sorted({hi - lo for lo, hi in (est.hypergeom_interval(int(k), B, N) for k in ks)})
    lo, hi = 0, len(widths) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if plan.error_rate_probability(N, B, widths[mid], rate) >= 0.5:
            hi = mid
        else:
            lo = mid + 1
    return float(widths[lo])


def estimation_and_certify(margin, err, chip, draws, seed, verify, name):
    """Arm 2 (exact coverage, the Monte Carlo route, the chip-cluster design) and arm 3 (certify on arm 2's B = 300
    samples) on one population."""
    t0 = time.time()
    margin = np.asarray(margin, np.float64)
    w_all = np.asarray(err, np.float64)
    N, K = margin.size, int(w_all.sum())
    theta = K / N
    Z = zone_setup(margin, w_all)
    cov = {B: np.zeros(draws, bool) for B in BUDGETS}
    width = {B: np.zeros(draws) for B in BUDGETS}
    t_cov = {B: np.zeros(draws, bool) for B in BUDGETS}
    t_naive = {B: np.zeros(draws, bool) for B in BUDGETS}
    t_width = {B: np.zeros(draws) for B in BUDGETS}
    zones = {a: [] for a in (ALPHA, ALPHA_REPORTED)}
    checked = 0
    for r in range(draws):
        for B in BUDGETS:
            s = est.sample_for_estimation(margin, B, design="random", seed=seed + r)
            idx = s["indices"]
            wr = w_all[idx]
            res = est.estimate_error_rate(s, wr)
            cov[B][r] = res["low"] <= theta <= res["high"]
            width[B][r] = res["high"] - res["low"]
            if B == CERTIFY_BUDGET:
                for a in zones:
                    z = certify_steps(Z, idx, wr, a)
                    if r < verify and a == ALPHA:
                        try:
                            ref = est.certify_zone(margin, idx, wr, a, DELTA)
                            got = (ref["coverage"], ref["n_zone"])
                        except ValueError:
                            got = ("refused", None)
                        mine = ("refused", None) if z.get("refused") else (z["coverage"], z.get("n_zone"))
                        if got != mine:
                            raise SystemExit(f"{name}: certify's steps {mine} differ from certify_zone {got} at draw {r}")
                        checked += 1
                    zones[a].append(z)
            st = est.sample_for_estimation(margin, B, design="tiles", tiles=chip, per_tile=PER_TILE, seed=seed + r)
            rt = est.estimate_error_rate(st, w_all[st["indices"]])
            t_cov[B][r] = rt["low"] <= theta <= rt["high"]
            nv = rt["naive_interval_if_treated_as_random"]
            t_naive[B][r] = nv["low"] <= theta <= nv["high"]
            t_width[B][r] = rt["high"] - rt["low"]
    arm2 = {"N": N, "errors": K, "theta": theta, "cells": {}}
    for B in BUDGETS:
        exact = est.exact_coverage_srs(N, K, B, interval=est.hypergeom_interval)
        mc = float(cov[B].mean())
        se = float(np.sqrt(max(exact * (1 - exact), 1e-12) / draws))
        med_w = float(np.median(width[B]))
        arm2["cells"][str(B)] = {
            "exact_coverage": exact, "mc_coverage": mc, "mc_se": se, "mc_minus_exact_in_se": (mc - exact) / se,
            "median_width": med_w, "width_quantiles": {q: float(np.quantile(width[B], q)) for q in (0.05, 0.5, 0.95)},
            "plan": {"predicted_median_width_at_theta": plan_median_width(N, B, theta),
                     "predicted_median_width_at_default_rate": plan_median_width(N, B, plan.DEFAULT_RATE),
                     "probability_width_at_most_realised_median_at_theta":
                     plan.error_rate_probability(N, B, med_w, theta),
                     "probability_width_at_most_realised_median_at_default_rate":
                     plan.error_rate_probability(N, B, med_w, plan.DEFAULT_RATE),
                     "realised_share_width_at_most_median": float((width[B] <= med_w + 1e-12).mean())},
            "chip_cluster_reported": {"design": f"tiles of one chip (--tile {TILE_WINDOWS}), {PER_TILE} windows per chip",
                                      "coverage": float(t_cov[B].mean()),
                                      "naive_interval_coverage": float(t_naive[B].mean()),
                                      "median_width": float(np.median(t_width[B]))}}
    arm3 = {"budget": CERTIFY_BUDGET, "delta": DELTA, "rule": "prefix", "level_cut": est.ZONE_CUT,
            "certify_zone_checked_on_draws": checked, "by_alpha": {}}
    for a, zs in zones.items():
        c = np.array([z["coverage"] if z["coverage"] is not None else 0.0 for z in zs])
        certified = np.array([z["coverage"] is not None for z in zs])
        zerr = np.array([z["zone_error"] for z in zs if z["coverage"] is not None])
        viol = np.array([z["coverage"] is not None and z["zone_error"] > a for z in zs])
        top25 = max(1, int(round(0.25 * N)))
        e25 = float(Z["cum"][top25] / top25)
        arm3["by_alpha"][f"{a:g}"] = {
            "alpha": a, "min_labels_to_certify": est.min_labels_to_certify(a, DELTA),
            "p_certify_any": float(certified.mean()), "refused_as_review_set": int(sum(bool(z.get("refused")) for z in zs)),
            "violation_share": float(viol.mean()), "violations": int(viol.sum()),
            "median_certified_share": float(np.median(c)),
            "certified_share_quantiles": {q: float(np.quantile(c, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "certified_share_counts": {f"{v:g}": int((c == v).sum()) for v in np.unique(c)},
            "zone_true_error": None if zerr.size == 0 else {"median": float(np.median(zerr)), "max": float(zerr.max()),
                                                            "mean": float(zerr.mean())},
            "share_of_draws_at_or_above_0.25": float((c >= 0.25 - 1e-12).mean()),
            "plan": {"labels_to_test_0.25": plan.zone_floor(0.25, a, DELTA),
                     "true_error_of_the_0.25_zone": e25,
                     "prefix_most_probability_0.25_at_300": plan.zone_level_probability(N, CERTIFY_BUDGET, 0.25, a, e25)
                     if e25 <= a else None}}
    log(f"[exp110] {name}: arms 2-3, {draws} draws, {time.time() - t0:.0f} s")
    return arm2, arm3


# ----------------------------------------------------------------------------- arm 4
def sign_practice(N, T, KA, n):
    """exp107's arm A: n labels from all N windows, McNemar's exact (sign) test on the one-right windows."""
    import exp107_which_map_baselines as e107
    return e107.sign_arm(N, T, KA, min(n, N))


def tool_probability(N, D, KA, KB, n):
    """(P names map a, P names map b) for the tool's draw of n labels among the D differing windows, with the verdict
    estimate prints (rule auto); a budget of at least D labels them all."""
    if D == 0:
        return 0.0, 0.0
    return plan.which_map_probability(N, D, min(n, D), KA, KB)


def oriented(pa, pb, KA, KB):
    if KA > KB:
        return pa, pb
    if KB > KA:
        return pb, pa
    return None, pa + pb                               # no better map: any verdict is a wrong one


def which_map(a, b, t, draws, seed, name):
    t0 = time.time()
    a, b, t = (np.asarray(x).ravel() for x in (a, b, t))
    N = a.size
    diff = a != b
    D = int(diff.sum())
    KA = int((diff & (a == t)).sum())
    KB = int((diff & (b == t)).sum())
    out = {"N": N, "D": D, "K_A": KA, "K_B": KB, "neither": D - KA - KB,
           "better_agreeing": "2e" if KA > KB else "b3" if KB > KA else None,
           "accuracy_2e": float((a == t).mean()), "accuracy_b3": float((b == t).mean()),
           "difference_2e_minus_b3": (KA - KB) / N, "budgets": {}}
    for n in BUDGETS:
        pa, pb = tool_probability(N, D, KA, KB, n)
        right, wrong = oriented(pa, pb, KA, KB)
        cnt = {"a": 0, "b": 0, None: 0}
        for r in range(draws):
            s = est.sample_disagreement(a, b, n, seed=seed + r)
            i = s["indices"]
            cnt[est.compare_from_disagreement(s, a[i], b[i], t[i])["verdict"]] += 1
        mpa, mpb = cnt["a"] / draws, cnt["b"] / draws
        mright, mwrong = oriented(mpa, mpb, KA, KB)
        A_a, A_b = sign_practice(N, D, KA, n)
        Ar, Aw = oriented(A_a, A_b, KA, KB)
        rec = {"labels": min(n, D), "rule": est.which_map_rule(min(n, D), D) if D else None,
               "p_right": right, "p_wrong": wrong, "p_none": None if right is None else max(0.0, 1 - right - wrong),
               "mc": {"draws": draws, "p_right": mright, "p_wrong": mwrong},
               "practice_mcnemar_srs_of_all_windows": {"p_right": Ar, "p_wrong": Aw}}
        if right is not None:
            se = float(np.sqrt(max(right * (1 - right), 1e-6) / draws))
            rec["mc"]["right_minus_exact_in_se"] = (mright - right) / se
        out["budgets"][str(n)] = rec
    if KA != KB:
        better_a = KA > KB

        def first(fn, cap):
            last = None
            for n in LADDER:
                n = min(n, cap)
                if n == last:
                    break
                last = n
                pa, pb = fn(n)
                if (pa if better_a else pb) >= POWER_REPORTED:
                    return n
            return f"not reached by {last}"
        out["labels_for_power_0.8"] = {"tool": first(lambda n: tool_probability(N, D, KA, KB, n), D),
                                       "practice_mcnemar": first(lambda n: sign_practice(N, D, KA, n), N)}
        try:
            pw = plan.plan_which_map(N, D, abs(KA - KB) / N, power=POWER_REPORTED, two_class=True)
            out["plan_which_map_labels_power_0.8"] = pw.get("labels")
        except ValueError as exc:
            out["plan_which_map_labels_power_0.8"] = f"refused: {exc}"
    log(f"[exp110] {name}: arm 4, N {N} D {D} K_A {KA} K_B {KB}, {time.time() - t0:.0f} s")
    return out


# ----------------------------------------------------------------------------- arm 1
def arm1(P, boot, seed, name):
    conf, e = P["conf"], P["err"].astype(bool)
    st = rank_stats(-conf, e)
    out = {"N": int(e.size), "errors": int(e.sum()), "theta": float(e.mean()), **st,
           "oracle_capture_10": metrics.capture_at_budget_expected(e.astype(float), e, (CAPTURE_BUDGET,))[CAPTURE_BUDGET]}
    m = float(np.median(conf[~e])) if (~e).any() else float("nan")
    out["confident_errors"] = {"median_correct_confidence": m, "share_of_errors": float((conf[e] >= m).mean())
                               if e.any() else float("nan")}
    out["controls"] = {k: rank_stats(P[k], e) for k in ("ctl_pixel_variance", "ctl_class_rarity")}
    best = max(out["controls"], key=lambda k: out["controls"][k]["auroc"])
    out["best_control"] = best
    out["lead_over_best_control"] = out["auroc"] - out["controls"][best]["auroc"]
    out["cues_reported"] = {"neighbour_disagreement": rank_stats(P["cue_neighbour"], e),
                            "boundary_first_order": rank_stats(P["cue_boundary_first"], e)}
    out["bootstrap"] = chip_bootstrap(conf, e, P[best], P["chip"], P["n_chips"], boot, seed)
    log(f"[exp110] {name}: arm 1 N {out['N']} errors {out['errors']} theta {out['theta']:.4f} AUROC {out['auroc']:.4f} "
        f"capture {out['capture_10']:.4f} best control {best} lead {out['lead_over_best_control']:.4f}")
    return out


# ----------------------------------------------------------------------------- verdicts
def b3_stop_rules(inventory):
    """The page's two label-free stop rules for B3, before any label is read with B3's map: in every country B3 in
    the stated band order agrees with the 2e map on more windows than B3 with the bands reversed, and on at least 70%
    of the country's windows. Either failing stops arm 4 for the owner."""
    swap = {c: inventory[c]["b3_agreement_stated"] > inventory[c]["b3_agreement_reversed"] for c in inventory}
    agree = {c: inventory[c]["b3_agreement_stated"] >= B3_AGREE_STOP for c in inventory}
    ok = all(swap.values()) and all(agree.values())
    return {"band_swap_control_passes": swap, "agreement_at_least_0.70": agree, "arm4_runs": ok,
            "reason": None if ok else "B3 stop rule: " + "; ".join(
                [f"{c}: reversed bands agree as often or more" for c, v in swap.items() if not v]
                + [f"{c}: stated order agrees on fewer than 70% of windows" for c, v in agree.items() if not v])}


def verdicts(rows, arm4_runs=True):
    """The page's bars on the rows: rows[population] for the four countries and "pooled", each with arm1, arm2,
    arm3 and arm4 blocks as written by `run`."""
    out = {}
    det, met, below = {}, [], []
    for c in COUNTRIES:
        a = rows[c]["arm1"]
        if a["errors"] < P1_ERROR_FLOOR:
            det[c] = "not determined (fewer than 200 window errors)"
            below.append(c)
            continue
        ok = a["auroc"] >= P1_AUROC and a["capture_10"] >= P1_CAPTURE and a["lead_over_best_control"] >= P1_LEAD
        det[c] = "meets" if ok else "does not meet"
        if ok:
            met.append(c)
    p1 = "holds" if len(met) >= P1_COUNTRIES else ("cannot hold (two or more countries below the floor)"
                                                    if len(below) >= 2 else "fails")
    out["P1"] = {"bar": f"AUROC >= {P1_AUROC}, capture at 10% >= {P1_CAPTURE} and AUROC >= best control + {P1_LEAD}, "
                        f"in >= {P1_COUNTRIES} of 4 countries; floor {P1_ERROR_FLOOR} window errors",
                 "by_country": det, "countries_meeting": met, "below_floor": below, "verdict": p1,
                 "holds": p1 == "holds"}
    cells, bad = {}, []
    for pop in COUNTRIES + ("pooled",):
        for B in BUDGETS:
            c = rows[pop]["arm2"]["cells"][str(B)]
            ok = c["exact_coverage"] >= P2_COVERAGE and abs(c["mc_coverage"] - c["exact_coverage"]) <= P2_MC_SE * c["mc_se"]
            cells[f"{pop}:{B}"] = {"exact": c["exact_coverage"], "mc": c["mc_coverage"], "ok": ok}
            if not ok:
                bad.append(f"{pop}:{B}")
    out["P2"] = {"bar": f"exact coverage >= {P2_COVERAGE} at B = 100 and 300 in all 10 cells, Monte Carlo within "
                        f"{P2_MC_SE:g} se", "cells": cells, "failing": bad, "holds": not bad,
                 "verdict": "holds" if not bad else "fails: a defect; stop before anything is recorded"}
    a = f"{ALPHA:g}"
    viol = {pop: rows[pop]["arm3"]["by_alpha"][a]["violation_share"] for pop in COUNTRIES + ("pooled",)}
    p3a = all(v <= P3_VIOLATION for v in viol.values())
    out["P3a"] = {"bar": f"violation share <= {P3_VIOLATION} in each of 5 populations at alpha {ALPHA:g}",
                  "violation_share": viol, "holds": p3a,
                  "verdict": "holds" if p3a else "fails: a defect; stop before anything is recorded"}
    share = {c: rows[c]["arm3"]["by_alpha"][a]["median_certified_share"] for c in COUNTRIES}
    met3 = [c for c in COUNTRIES if share[c] >= P3_SHARE]
    out["P3b"] = {"bar": f"median certified share >= {P3_SHARE} at alpha {ALPHA:g} and {CERTIFY_BUDGET} labels in >= "
                         f"{P3_COUNTRIES} of 4 countries", "median_certified_share": share, "countries_meeting": met3,
                  "holds": len(met3) >= P3_COUNTRIES, "verdict": "holds" if len(met3) >= P3_COUNTRIES else "fails"}
    if not arm4_runs:
        for k in ("P4a", "P4b"):
            out[k] = {"verdict": "not run: arm 4 stopped for the owner by a B3 stop rule", "holds": None}
        return out
    wrong = {f"{pop}:{n}": rows[pop]["arm4"]["budgets"][str(n)]["p_wrong"] for pop in COUNTRIES + ("pooled",)
             for n in BUDGETS}
    p4a = all(v <= P4_WRONG for v in wrong.values())
    out["P4a"] = {"bar": f"P(names the worse-agreeing map) <= {P4_WRONG} at n = 100 and 300, every country and pooled",
                  "p_wrong": wrong, "holds": p4a, "verdict": "holds" if p4a else "fails"}
    right = {c: rows[c]["arm4"]["budgets"][str(P4_HEADLINE)]["p_right"] for c in COUNTRIES}
    met4 = [c for c in COUNTRIES if right[c] is not None and right[c] >= P4_RIGHT]
    out["P4b"] = {"bar": f"P(names the better-agreeing map) >= {P4_RIGHT} at n = {P4_HEADLINE} in >= {P4_COUNTRIES} of 4 "
                         "countries", "p_right": right, "countries_meeting": met4, "holds": len(met4) >= P4_COUNTRIES,
                  "verdict": "holds" if len(met4) >= P4_COUNTRIES else "fails"}
    return out


# ----------------------------------------------------------------------------- data sources
class RealSource:
    """The local FTW data directory (outside the repository). Every file read is checked against the acquisition
    manifest's sha256 before use."""

    def __init__(self, data):
        self.data = data
        self.manifest = {}
        with open(os.path.join(data, "manifest_files.csv"), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.manifest[row["path"]] = row["sha256"]
        self.hashes = {}
        self.s2_hashes = []

    def _check(self, rel):
        h = sha256_file(os.path.join(self.data, rel))
        if self.manifest.get(rel) != h:
            raise SystemExit(f"{rel}: sha256 does not match the acquisition manifest")
        return h

    def _npz(self, c, kind):
        name = {"map": f"map2e_{MAP_YEAR[c]}_on_mask_grid_{c}.npz", "b3": f"b3_on_mask_grid_{c}.npz",
                "b3swap": f"b3swap_on_mask_grid_{c}.npz"}[kind]
        rel = f"v1/{c}/{name}"
        if rel not in self.hashes:
            self.hashes[rel] = self._check(rel)
        return np.load(os.path.join(self.data, rel), allow_pickle=False)

    def chips(self, c):
        with self._npz(c, "map") as z:
            ids, gt, tiles = z["chip_ids"].astype(str), z["grid_transform"].copy(), z["map_tiles"].astype(str)
            if not bool(z["map_valid"].all()):
                raise SystemExit(f"{c}: map_valid is not all True")
        if list(ids) != sorted(ids):
            raise SystemExit(f"{c}: the chips are not in sorted aoi_id order")
        for kind in ("b3", "b3swap"):
            with self._npz(c, kind) as z:
                if not np.array_equal(z["chip_ids"].astype(str), ids):
                    raise SystemExit(f"{c}: {kind} chip order differs from the map's")
        return ids, gt, np.array(["+" in t for t in tiles])

    def map_layers(self, c):
        with self._npz(c, "map") as z:
            return z["map_field"], z["map_boundary"]

    def masks(self, c):
        with self._npz(c, "map") as z:
            return z["mask2"], z["mask3"]

    def b3(self, c):
        with self._npz(c, "b3") as z:
            if list(z["class_order"].astype(str)) != ["background", "field", "boundary"]:
                raise SystemExit(f"{c}: B3 class order is not background, field, boundary")
            return z["b3_field"].astype(np.float64) + z["b3_boundary"], z["s2_allzero_either"]

    def b3swap(self, c):
        with self._npz(c, "b3swap") as z:
            if list(z["band_order_in"].astype(str)) != ["B02", "B03", "B04", "B08"]:
                raise SystemExit(f"{c}: the band-swap run's band order is not B02, B03, B04, B08")
            return z["b3swap_field"].astype(np.float64) + z["b3swap_boundary"]

    def s2(self, c, ids, gt):
        import rasterio
        out = np.empty((len(ids), 8, CHIP, CHIP), np.uint16)
        for i, cid in enumerate(ids):
            for j, win in enumerate(("window_a", "window_b")):
                rel = f"v1/{c}/s2_images/{win}/{cid}.tif"
                self.s2_hashes.append((rel, self._check(rel)))
                with rasterio.open(os.path.join(self.data, rel)) as f:
                    a = f.read()
                    if tuple(f.descriptions) != BANDS or a.shape != (4, CHIP, CHIP) or a.dtype != np.uint16:
                        raise SystemExit(f"{c}: an S2 file is not B04, B03, B02, B08 uint16 256 x 256")
                    if f.crs is None or f.crs.to_epsg() != 4326 or not np.allclose(
                            np.array(tuple(f.transform)[:6]), gt[i], rtol=0, atol=1e-12):
                        raise SystemExit(f"{c}: an S2 file is not on its chip's mask grid")
                out[i, 4 * j:4 * j + 4] = a
        return out

    def ftwq_splits(self):
        rel = "ftwq/index/index.parquet"
        self.hashes[rel] = self._check(rel)
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return None, None
        import pyarrow
        t = pq.read_table(os.path.join(self.data, rel), columns=["country", "aoi_id", "split"]).to_pydict()
        return {(c, a): s for c, a, s in zip(t["country"], t["aoi_id"], t["split"])}, pyarrow.__version__

    def input_record(self):
        s2 = sorted(self.s2_hashes)
        return {"npz_and_index_sha256": dict(sorted(self.hashes.items())),
                "s2_files": {"count": len(s2), "all_match_manifest": True,
                             "sha256_of_their_sha256_in_sorted_path_order":
                             hashlib.sha256("".join(h for _, h in s2).encode()).hexdigest()},
                "manifest_files_csv_sha256": sha256_file(os.path.join(self.data, "manifest_files.csv"))}


class SmokeSource:
    """Synthetic chips with the real data's layout, for the smoke run: fields as blocks, a map that is unsure and
    wrong more often near field edges, B3 a noisier map, its band-swapped run noisier still, S2 noise with a zero strip."""

    def __init__(self, chips=(20, 21, 19, 22), seed=0):
        self.n = dict(zip(COUNTRIES, chips))
        self.rng = np.random.default_rng(seed)
        self.d = {}
        for c in COUNTRIES:
            self.d[c] = self._make(self.n[c])

    def _make(self, n):
        rng = self.rng
        blocks = rng.random((n, 16, 16)) < 0.4
        m2 = np.repeat(np.repeat(blocks, 16, axis=1), 16, axis=2).astype(np.uint8)
        edge = np.zeros_like(m2, bool)
        edge[:, 1:, :] |= m2[:, 1:, :] != m2[:, :-1, :]
        edge[:, :, 1:] |= m2[:, :, 1:] != m2[:, :, :-1]
        m3 = np.where(m2 == 1, np.where(edge, 2, 1), 0).astype(np.uint8)

        def noisy(scale):
            z = rng.normal(0, scale, (n, 32, 32))
            z = np.repeat(np.repeat(z, 8, axis=1), 8, axis=2)
            base = np.where(m2 == 1, 0.8, 0.2) + z + np.where(edge, rng.normal(0, 0.3, m2.shape), 0)
            return np.clip(base, 0.0, 1.0).astype(np.float32)
        p = noisy(0.25)
        field, bnd = (p * 0.9).astype(np.float32), (p * 0.1).astype(np.float32)
        b3 = noisy(0.2)
        swap = noisy(0.6)
        s2 = (rng.integers(100, 3000, (n, 8, CHIP, CHIP)) * (1 + m2[:, None])).astype(np.uint16)
        s2[0, :, :, :12] = 0                                           # a zero strip in every band
        zero = (s2[:, :4] == 0).all(1) | (s2[:, 4:] == 0).all(1)
        ids = np.array([f"smoke_{i:05d}" for i in range(n)])
        seam = np.zeros(n, bool)
        seam[-1] = True
        return {"ids": ids, "gt": np.zeros((n, 6)), "seam": seam, "field": field, "bnd": bnd, "m2": m2, "m3": m3,
                "b3": b3.astype(np.float64), "zero": zero, "swap": swap.astype(np.float64), "s2": s2}

    def chips(self, c):
        return self.d[c]["ids"], self.d[c]["gt"], self.d[c]["seam"]

    def map_layers(self, c):
        return self.d[c]["field"], self.d[c]["bnd"]

    def masks(self, c):
        return self.d[c]["m2"], self.d[c]["m3"]

    def b3(self, c):
        return self.d[c]["b3"], self.d[c]["zero"]

    def b3swap(self, c):
        return self.d[c]["swap"]

    def s2(self, c, ids, gt):
        return self.d[c]["s2"]

    def ftwq_splits(self):
        return {(c, i): "test" for c in COUNTRIES for i in self.d[c]["ids"]}, None

    def input_record(self):
        return {"smoke": "synthetic chips; no file read"}


# ----------------------------------------------------------------------------- the run
def label_free(src, c, smoke):
    """Everything about a country that needs no label: the 2e readings, the controls and cues, B3 in both band
    orders, D. Returns per-chip window arrays and the label-free inventory."""
    t0 = time.time()
    ids, gt, seam = src.chips(c)
    n = len(ids)
    if not smoke and (n != CHIPS[c] or int(seam.sum()) != SEAM_CHIPS[c]):
        raise SystemExit(f"{c}: {n} chips and {int(seam.sum())} seam chips, not the page's {CHIPS[c]} and {SEAM_CHIPS[c]}")
    f, b = src.map_layers(c)
    p = f.astype(np.float64) + b
    if float(p.max()) > 1 + 1e-6 or float(p.min()) < 0:
        raise SystemExit(f"{c}: field + boundary leaves [0, 1]")
    r8 = read_extent(p, PATCH)
    r16 = read_extent(p, PATCH_REPORTED)
    del p
    r3 = read_three_class(f, b, PATCH)
    del f, b
    s2 = src.s2(c, ids, gt)
    zero_a, zero_b = (s2[:, :4] == 0).all(1), (s2[:, 4:] == 0).all(1)
    zero_either = zero_a | zero_b
    var8, var_empty, sds = pixel_variance(s2, PATCH)
    del s2
    b3p, b3zero = src.b3(c)
    if not np.array_equal(np.asarray(b3zero, bool), zero_either):
        raise SystemExit(f"{c}: B3's zero-imagery flag differs from the S2 files'")
    b3 = read_extent(b3p, PATCH, chunk=64)
    del b3p
    sw = read_extent(src.b3swap(c), PATCH, chunk=64)
    zero_win = window_any(zero_either, PATCH)
    W = {"conf8": r8["conf"], "cls8": r8["cls"], "bnd8": r8["boundary"], "conf16": r16["conf"], "cls16": r16["cls"],
         "conf3": r3["conf"], "cls3": r3["cls"], "var8": var8, "rare8": rarity(r8["cls"]),
         "nbr8": neighbour_disagreement(r8["cls"]),
         "bf8": A.boundary_first_score(-r8["conf"], r8["boundary"]),
         "b3cls8": b3["cls"], "swapcls8": sw["cls"], "zero_win8": zero_win, "seam": seam, "ids": ids, "n": n}
    same = b3["cls"] == r8["cls"]
    inv = {"chips": n, "windows": int(r8["cls"].size), "seam_chips": int(seam.sum()),
           "map_confidence_quantiles": r8["quantiles"], "map_class_share": r8["class_share"],
           "map_distinct_window_confidences": r8["distinct"],
           "map_windows_at_confidence_1": int((r8["conf"] == 1.0).sum()),
           "map_16px": {"windows": int(r16["cls"].size), "confidence_quantiles": r16["quantiles"],
                        "class_share": r16["class_share"]},
           "controls": {"pixel_variance_quantiles": {q: float(np.quantile(var8, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                        "pixel_variance_windows_with_no_pixel": var_empty, "band_sd": sds,
                        "pixels_zero_in_one_window_only": int((zero_a ^ zero_b).sum()),
                        "pixels_zero_in_every_band": int((zero_a & zero_b).sum()),
                        "class_rarity_values": sorted({float(v) for v in np.unique(W["rare8"])})},
           "zero_imagery_windows": int(zero_win.sum()),
           "b3_class_share": {int(k): float((b3["cls"] == k).mean()) for k in (0, 1)},
           "D_all_windows": int((~same).sum()),
           "D_without_zero_imagery_windows": int((~same & ~zero_win).sum()),
           "b3_agreement_stated": float(same.mean()),
           "b3_agreement_reversed": float((sw["cls"] == r8["cls"]).mean()),
           "b3_agreement_stated_without_zero_imagery": float(same[~zero_win].mean()),
           "seconds": round(time.time() - t0, 1)}
    log(f"[exp110] {c}: label-free {n} chips, map median confidence {r8['quantiles'][0.5]:.4f}, D {inv['D_all_windows']}, "
        f"B3 agreement stated {inv['b3_agreement_stated']:.4f} reversed {inv['b3_agreement_reversed']:.4f}, "
        f"{inv['seconds']:.0f} s")
    return W, inv


def cli_assess(p_chips, mask2, work):
    """arm 1 through the command line: assess --patch 8 --reference --budgets 0.1 on the country's mosaic."""
    d = tempfile.mkdtemp(prefix="exp110_", dir=work)
    try:
        mp, rp = os.path.join(d, "map.npy"), os.path.join(d, "reference.npy")
        np.save(mp, row_mosaic(np.asarray(p_chips, np.float64)))
        np.save(rp, row_mosaic(np.asarray(mask2, np.int16)))
        cmd = [sys.executable, "-I", "-m", "oe_inferencex.cli", "assess", mp, "--patch", str(PATCH), "--reference", rp,
               "--budgets", str(CAPTURE_BUDGET), "--out", os.path.join(d, "out")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("assess on the command line failed: " + r.stderr[-1500:])
        with open(os.path.join(d, "out", "assessment.json"), encoding="utf-8") as f:
            s = json.load(f)
        return s
    finally:
        shutil.rmtree(d, ignore_errors=True)


def labelled(src, c, W, work, smoke):
    """The labels, after the label-free phase of every country: truth, label-only checks, the command-line check of
    arm 1, and each country's population arrays (in the mosaic's flat order)."""
    t0 = time.time()
    n = W["n"]
    m2, m3 = src.masks(c)
    m2 = np.asarray(m2)
    m3 = np.asarray(m3)
    checks = {"mask2_values": sorted(int(v) for v in np.unique(m2)), "mask3_values": sorted(int(v) for v in np.unique(m3)),
              "mask2_differs_from_mask3_field_or_boundary_pixels": int((m2 != np.isin(m3, (1, 2))).sum())}
    t8 = truth_windows(m2, PATCH, 2)
    t16 = truth_windows(m2, PATCH_REPORTED, 2)
    t3 = truth_windows(m3, PATCH, 3)
    tlen = truth_windows(dilate_field(m2), PATCH, 2)
    checks.update({"windows_truth_tied": int((t8 < 0).sum()), "windows_truth_tied_16px": int((t16 < 0).sum()),
                   "windows_truth3_tied": int((t3 < 0).sum()),
                   "truth_class_share": {k: float((t8[t8 >= 0] == k).mean()) for k in (0, 1)},
                   "truth3_class_share": {k: float((t3[t3 >= 0] == k).mean()) for k in (0, 1, 2)}})
    # the command line's arm 1, against the Python arrays
    f, b = src.map_layers(c)
    p = f.astype(np.float64) + b
    s = cli_assess(p, m2, work)
    del p
    pop = t8 >= 0
    err = (W["cls8"] != t8) & pop
    rc = s["against_reference"]
    u, e = -W["conf8"][pop], err[pop]
    cli = {"n_windows": s["n_windows"], "n_windows_scored": rc["n_windows_scored"],
           "n_windows_reference_tied": rc["n_windows_reference_tied"], "error_rate": rc["error_rate"],
           "aurc_confidence": rc["aurc_confidence"], "excess_aurc_confidence": rc["excess_aurc_confidence"],
           "capture_10_stable_order": rc["error_capture_at_budget"][str(CAPTURE_BUDGET)]["errors_captured_fraction"]}
    agree = (cli["n_windows_scored"] == int(pop.sum()) and cli["n_windows_reference_tied"] == int((~pop).sum())
             and abs(cli["error_rate"] - float(e.mean())) < 1e-12
             and abs(cli["aurc_confidence"] - metrics.aurc_expected(u, e.astype(float))) < 1e-9
             and abs(cli["excess_aurc_confidence"] - metrics.excess_aurc(u, e.astype(float))) < 1e-9)
    if not agree:
        raise SystemExit(f"{c}: the command line's assess disagrees with the Python arrays: {cli}")
    cli["agrees_with_python_arrays"] = True
    # pixel-level 3-class reading (--patch 1): confidence the top probability, class its argmax
    tables = []
    for s0 in range(0, n, 64):
        ff = np.asarray(f[s0:s0 + 64], np.float64)
        bb = np.asarray(b[s0:s0 + 64], np.float64)
        sc = np.stack([1.0 - ff - bb, ff, bb], axis=1)
        tables.append(grouped_table(-sc.max(axis=1), sc.argmax(axis=1) != m3[s0:s0 + 64]))
        del ff, bb, sc
    del f, b
    px = merge_tables(tables)
    # the grouped statistics against the package's metrics, on the windows
    g_auc, g_cap, _ = grouped_auroc_capture(grouped_table(u, e))
    st = rank_stats(u, e)
    if abs(g_auc - st["auroc"]) > 1e-9 or abs(g_cap - st["capture_10"]) > 1e-9:
        raise SystemExit(f"{c}: the grouped AUROC or capture differs from the package's metrics")
    chip = np.broadcast_to(np.arange(n)[:, None, None], t8.shape)
    pop3, pop16, poplen = t3 >= 0, t16 >= 0, tlen >= 0
    keep4 = pop & ~W["zero_win8"]
    P = {"conf": flat(W["conf8"])[flat(pop)], "err": flat(err)[flat(pop)],
         "ctl_pixel_variance": flat(W["var8"])[flat(pop)], "ctl_class_rarity": flat(W["rare8"])[flat(pop)],
         "cue_neighbour": flat(W["nbr8"])[flat(pop)], "cue_boundary_first": flat(W["bf8"])[flat(pop)],
         "chip": flat(chip)[flat(pop)], "n_chips": n,
         "seam": flat(np.broadcast_to(W["seam"][:, None, None], t8.shape))[flat(pop)],
         "a4": {"a": flat(W["cls8"])[flat(keep4)], "b": flat(W["b3cls8"])[flat(keep4)], "t": flat(t8)[flat(keep4)],
                "seam": flat(np.broadcast_to(W["seam"][:, None, None], t8.shape))[flat(keep4)]},
         "r3": {"conf": flat(W["conf3"])[flat(pop3)], "err": flat(W["cls3"] != t3)[flat(pop3)]},
         "r16": {"conf": flat(W["conf16"])[flat(pop16)], "err": flat(W["cls16"] != t16)[flat(pop16)]},
         "len": {"conf": flat(W["conf8"])[flat(poplen)], "err": flat(W["cls8"] != tlen)[flat(poplen)],
                 "main_errors_not_lenient_errors": int((err & ~((W["cls8"] != tlen) & poplen)).sum()),
                 "lenient_errors_not_main_errors": int(((W["cls8"] != tlen) & poplen & ~err).sum())},
         "px": px}
    checks["seconds"] = round(time.time() - t0, 1)
    return P, checks, cli


def pool(Ps):
    """The four countries' populations as one: concatenated, chips renumbered."""
    off, chips = 0, []
    for P in Ps:
        chips.append(P["chip"] + off)
        off += P["n_chips"]
    keys = ("conf", "err", "ctl_pixel_variance", "ctl_class_rarity", "cue_neighbour", "cue_boundary_first", "seam")
    out = {k: np.concatenate([P[k] for P in Ps]) for k in keys}
    out.update({"chip": np.concatenate(chips), "n_chips": off,
                "a4": {k: np.concatenate([P["a4"][k] for P in Ps]) for k in ("a", "b", "t", "seam")},
                "r3": {k: np.concatenate([P["r3"][k] for P in Ps]) for k in ("conf", "err")},
                "r16": {k: np.concatenate([P["r16"][k] for P in Ps]) for k in ("conf", "err")},
                "len": {"conf": np.concatenate([P["len"]["conf"] for P in Ps]),
                        "err": np.concatenate([P["len"]["err"] for P in Ps]),
                        "main_errors_not_lenient_errors": sum(P["len"]["main_errors_not_lenient_errors"] for P in Ps),
                        "lenient_errors_not_main_errors": sum(P["len"]["lenient_errors_not_main_errors"] for P in Ps)},
                "px": None})                    # the pixel-level reading is reported per country only
    return out


def reported(P):
    """The page's reported-not-graded readings of one population."""
    def short(conf, e):
        e = np.asarray(e, bool)
        st = rank_stats(-np.asarray(conf), e)
        return {"N": int(e.size), "errors": int(e.sum()), "theta": float(e.mean()), "auroc": st["auroc"],
                "capture_10": st["capture_10"]}
    keep = ~P["seam"]
    ns = {**short(P["conf"][keep], P["err"][keep])}
    e_ns = P["err"][keep].astype(bool)
    ns["controls_auroc"] = {k: metrics.weighted_auroc(P[k][keep], e_ns, np.ones(e_ns.size))
                            for k in ("ctl_pixel_variance", "ctl_class_rarity")}
    N, K = int(keep.sum()), int(e_ns.sum())
    ns["exact_coverage"] = {str(B): est.exact_coverage_srs(N, K, B, interval=est.hypergeom_interval) for B in BUDGETS}
    a4 = P["a4"]
    k4 = ~a4["seam"]
    a, b, t = a4["a"][k4], a4["b"][k4], a4["t"][k4]
    diff = a != b
    D, KA, KB = int(diff.sum()), int((diff & (a == t)).sum()), int((diff & (b == t)).sum())
    ns["arm4"] = {"N": int(a.size), "D": D, "K_A": KA, "K_B": KB,
                  "budgets": {str(n): dict(zip(("p_right", "p_wrong"), oriented(*tool_probability(a.size, D, KA, KB, n),
                                                                                 KA, KB))) for n in BUDGETS}}
    return {"three_class_8px": short(P["r3"]["conf"], P["r3"]["err"]),
            "three_class_1px": None if P["px"] is None else dict(zip(
                ("auroc", "capture_10", "theta"), grouped_auroc_capture(P["px"])), pixels=int(P["px"][2].sum())),
            "windows_16px": short(P["r16"]["conf"], P["r16"]["err"]),
            "lenient_extent_8px": {**short(P["len"]["conf"], P["len"]["err"]),
                                   "main_errors_not_lenient_errors": P["len"]["main_errors_not_lenient_errors"],
                                   "lenient_errors_not_main_errors": P["len"]["lenient_errors_not_main_errors"]},
            "without_seam_chips": ns}


def run(src, draws, boot, verify, work, smoke):
    t_start = time.time()
    secs = {}
    # phase A: no label read
    W, inv = {}, {}
    for c in COUNTRIES:
        W[c], inv[c] = label_free(src, c, smoke)
    rules = b3_stop_rules(inv)
    secs["label_free"] = round(time.time() - t_start, 1)
    log(f"[exp110] B3 stop rules: arm 4 runs = {rules['arm4_runs']}" + ("" if rules["arm4_runs"] else f" ({rules['reason']})"))
    # metadata (no label): each chip is `test` in the FTW-Q index too
    splits, pa_version = src.ftwq_splits()
    if splits is None:
        meta = {"checked": False, "reason": "pyarrow is not installed in this environment"}
    else:
        got = [splits.get((c, str(i))) for c in COUNTRIES for i in W[c]["ids"]]
        meta = {"checked": True, "chips": len(got), "test_in_ftwq_index": sum(g == "test" for g in got),
                "missing_from_ftwq_index": sum(g is None for g in got),
                "other_split": sum(g not in (None, "test") for g in got), "pyarrow": pa_version}
    # phase B: the labels
    t1 = time.time()
    Ps, rows, checks, cli = {}, {}, {}, {}
    for c in COUNTRIES:
        Ps[c], checks[c], cli[c] = labelled(src, c, W[c], work, smoke)
        W[c] = None
    Ps["pooled"] = pool([Ps[c] for c in COUNTRIES])
    secs["labels_and_cli"] = round(time.time() - t1, 1)
    t3 = time.time()
    for name in COUNTRIES + ("pooled",):
        P = Ps[name]
        t2 = time.time()
        row = {"arm1": arm1(P, boot, SEED, name)}
        row["arm2"], row["arm3"] = estimation_and_certify(P["conf"], P["err"], P["chip"], draws, SEED, verify, name)
        if rules["arm4_runs"]:
            row["arm4"] = which_map(P["a4"]["a"], P["a4"]["b"], P["a4"]["t"], draws, SEED, name)
        row["reported"] = reported(P)
        row["seconds"] = round(time.time() - t2, 1)
        log(f"[exp110] {name}: all arms {row['seconds']:.0f} s")
        rows[name] = row
    v = verdicts(rows, rules["arm4_runs"])
    secs["arms"] = round(time.time() - t3, 1)
    secs["total"] = round(time.time() - t_start, 1)
    return {"label_free": {"countries": inv, "b3_stop_rules": rules},
            "checks": {"labels": checks, "metadata_ftwq": meta, "command_line_arm1": cli},
            "populations": rows, "verdicts": v, "seconds": secs}


def config():
    return {"countries": list(COUNTRIES), "map_year": MAP_YEAR, "chips": CHIPS, "seam_chips": SEAM_CHIPS,
            "patch_graded": PATCH, "patch_reported": PATCH_REPORTED, "alpha": ALPHA, "alpha_reported": ALPHA_REPORTED,
            "delta": DELTA, "budgets": list(BUDGETS), "certify_budget": CERTIFY_BUDGET, "draws": DRAWS,
            "seed": SEED, "draw_seed": "seed 110 + r for draw r (sample_for_estimation, sample_disagreement)",
            "bootstrap": {"resamples": BOOT, "seed": SEED, "level": BOOT_LEVEL, "by": "chip"},
            "capture_budget": CAPTURE_BUDGET, "chip_cluster": {"tile_windows": TILE_WINDOWS, "per_tile": PER_TILE},
            "lenient_dilation_px": {"north_south": DILATE_NS, "east_west": DILATE_EW},
            "bars": {"P1": {"auroc": P1_AUROC, "capture_10": P1_CAPTURE, "lead": P1_LEAD, "countries": P1_COUNTRIES,
                            "error_floor": P1_ERROR_FLOOR},
                     "P2": {"coverage": P2_COVERAGE, "mc_se": P2_MC_SE},
                     "P3": {"violation": P3_VIOLATION, "median_share": P3_SHARE, "countries": P3_COUNTRIES},
                     "P4": {"wrong": P4_WRONG, "right": P4_RIGHT, "countries": P4_COUNTRIES, "headline": P4_HEADLINE},
                     "b3_agreement_stop": B3_AGREE_STOP},
            "reading": {"map": "p = field + boundary per mask pixel, read as a two-class probability map",
                        "truth": "majority of the window's 2-class mask pixels; an even split is left out",
                        "mosaic": "each country's chips side by side in sorted aoi_id order, one row",
                        "pooled": "the four countries' populations concatenated; each window keeps its country's "
                                  "control values (variance scaled by the country's band SDs, rarity within the country)"},
            "certify_route": f"certify_zone's steps with the order and the guard computed once per population; "
                             f"certify_zone itself on the first {VERIFY} draws per population at alpha {ALPHA:g}"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="synthetic chips; no data read")
    ap.add_argument("--data", default=os.environ.get("FTW_DATA"), help="the local FTW data directory")
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--work", default=None, help="directory for the command line's temporary mosaics")
    ap.add_argument("--draws", type=int, default=None, help="smoke only")
    ap.add_argument("--boot", type=int, default=None, help="smoke only")
    a = ap.parse_args(argv)
    if a.smoke:
        draws, boot = a.draws or 30, a.boot or 30
        rec = run(SmokeSource(), draws, boot, min(VERIFY, 3), a.work, smoke=True)
        rec = {"experiment": "exp110 smoke (synthetic chips)", "config": config(), **rec}
        os.makedirs(a.out_dir, exist_ok=True)
        with open(os.path.join(a.out_dir, "exp110_summary_smoke.json"), "w", encoding="utf-8") as f:
            json.dump(jsonable(rec), f, indent=1, allow_nan=False)
        return rec
    status = prereg_status()
    if status != "frozen":
        raise SystemExit(f"{os.path.relpath(PLAN, ROOT)} is {status}: no arm is computed on the FTW data before it is frozen")
    if a.draws is not None or a.boot is not None:
        raise SystemExit("--draws and --boot are for the smoke run; the page fixes 2,000 of each")
    if not a.data or not os.path.isdir(a.data):
        raise SystemExit("--data: the local FTW data directory is needed")
    commit, dirty = git_state()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    src = RealSource(a.data)
    rec = run(src, DRAWS, BOOT, VERIFY, a.work, smoke=False)
    import rasterio
    rec = {"experiment": "exp110: FTW's 2nd-edition field map read as the package reads any probability map, at the FTW "
                         "v1 test chips of Belgium, Croatia, Slovenia and Sweden",
           "prereg_page": os.path.relpath(PLAN, ROOT), "prereg_status": status, "commit": commit, "dirty": dirty,
           "script_sha256": sha256_file(os.path.abspath(__file__)), "started_utc": started,
           "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "environment": {"olmoearth_inferencex": oe_inferencex.__version__, "numpy": np.__version__,
                           "rasterio": rasterio.__version__, "python": platform.python_version(),
                           "machine": platform.machine()},
           "config": config(), "inputs": src.input_record(), **rec,
           "notes": [B3_TRAINING_NOTE,
                     "Not recorded: the page asks an audit that did not write this script to recompute theta, the error "
                     "counts, AUROC, capture, K_A, K_B and one exact coverage cell with plain numpy before anything is "
                     "recorded."],
           "attribution": ["FTW (Fields of The World): the 2nd-edition global field map, the PRUE B3 model and the "
                           "benchmark, CC-BY-4.0", "Belgium: Bron: Dept. LV", "Slovenia: AKTRP (CC-BY-4.0)",
                           "Sweden: Jordbruksverket", "Croatia: APPRRR"]}
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, SUMMARY), "w", encoding="utf-8") as f:
        json.dump(jsonable(rec), f, indent=1, allow_nan=False)
    log(json.dumps(jsonable({k: {kk: vv for kk, vv in v.items() if kk in ("verdict", "holds")}
                             for k, v in rec["verdicts"].items()})))
    return rec


if __name__ == "__main__":
    main()
