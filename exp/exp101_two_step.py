"""exp101: filter, then shake. A two-step error ranking fitted on exp98's graded validation points (F) and tested on
exp99's TimeSync plots (E): set aside the most confident share q of the units by the deployed map's confidence, rank the
rest by how unstable the record's replica of FT-AWF is over 320 perturbed views of the unit's input, and review the top
10%. Preregistered in docs/plan/two_step.md; read that page first, this file is the run.

    python exp/exp101_two_step.py --smoke             # synthetic inputs and a stand-in model, end to end, no torch
    python exp/exp101_two_step.py --inventory --exp98-scores DIR [DIR ...] --labels WINDOWS [--deploy-2023 ROOT ...]
        --exp99-scores DIR [DIR ...] --timesync CSV --request-geometry AI2.geojson --deploy-2017 ROOT [ROOT ...]
                                                      # counts only; allowed before freezing
    python exp/exp101_two_step.py --gpu-smoke --exp98-scores DIR [DIR ...] --labels WINDOWS [--deploy-2023 ROOT ...]
                                                      # V0 on the GPU against the CPU at F's first 32 units; throughput
    python exp/exp101_two_step.py --fit --exp98-scores DIR [DIR ...] --labels WINDOWS [--deploy-2023 ROOT ...]
                                                      # F only: views, instability, the choice of q; before freezing
    python exp/exp101_two_step.py --test (every input above) --boundaries NE.geojson [--fit-record FIT.json]
                                                      # the test on E; refused until the page says frozen

The units (the page's names). F: exp98's 259 graded validation points on the 2023 map, read exactly as exp98 graded
them (find_outputs, open_grid, read_label_windows, locate_all, read_points, covered, readings: the class is the argmax
over the ten channels, the confidence the top-1 probability over the trained channels, as exp100_alignment's exp98_sets
reads them). E: exp99's 271 plots with 2017 input (exp99_transfer.select with its fixed rule, read_map, read_plots_on),
STRICT errors primary and LENIENT reported, design-weighted with Kenya and Tanzania as strata. Before anything is
shaken, the readings must give the record's counts (RECORDED); a reading that does not is refused.

The views. Each unit's input is a block of the deployed run's own rslearn dataset on scratch (twelve monthly item
groups of the twelve bands at 10 m, reordered to OlmoEarth's band order): exp98's 2023 roots for F and exp99's 2017
roots for E; without the 2023 roots, F reads exp89's label windows with oe_inferencex.awf.load_window_full, recorded as
a deviation. The record's replica (exp89 arm A: Ai2's pinned checkpoint, tokens averaged over months and band sets, the
1 x 1 head, bilinear upsampling by 4, exp21's forward pass) is run on every combination of
  orientation (8)  the dihedral group (exp36's transforms, identity first) on the crop's spatial axes, the output mapped
                   back exactly before the unit's pixel is read;
  time (5)         all twelve item groups, or one quarter removed, the nine left passed as a 9-step series with their
                   own month indices (under fast_pass the encoder ignores the mask: the time position comes from the
                   index, the month embedding from the timestamps, and the year is not read);
  context (2)      a 16-px or a 32-px crop;
  phase (4)        the crop's origin moved by 0 to 3 px along both axes (exp21's tiling shift),
so 320 views; V0 (view 0) is the identity, twelve months, 16 px, phase 0, with the unit at (8, 8). Instability s: the
share of the other 319 views whose class (argmax over the ten channels) differs from V0's, ties broken by the mean
total-variation distance between each view's ten-channel softmax and V0's.

The rule R(q): set aside the most confident share q of the units; rank the others by s, most unstable first, ties by
confidence, least confident first; the units set aside follow, least confident first. q is chosen on F from Q_GRID:
the q whose 10% review holds the most errors, ties to the larger q. Capture follows the record's convention
(oe_inferencex.metrics.weighted_capture_at_budget, as exp99_transfer.ranking reads it: the most suspect share b of the
design weight, a tied group straddling the cut counted in proportion).

Predictions (E, STRICT, design-weighted): P1, the 10% review by R(q) holds a share of the errors at least 5 points above
confidence alone's and the 90% within-country paired bootstrap interval of the difference lies above 0; P2, among the
units R(q) does not set aside, the design-weighted AUROC of s exceeds that of confidence, with the 90% interval of the
difference above 0. Floors: 250 plots with all 320 views read, 20 STRICT errors and 20 correct plots among them.
Reported, not graded: LENIENT; F's values for each q; each kind of perturbation alone; the dose (s from k random views);
the grass/shrub plots where STRICT and LENIENT part; the 5% and 20% budgets.

Compute. The views of many units run in one batch per crop size and series length (a 16- or 32-px crop and a 12- or
9-step series change the tensor's shape), up to BATCH_TOKENS encoder tokens per forward. exp89's GPU job 1151050
failed on an index_select device mismatch; in olmoearth_pretrain 0.1.0 the band-index tensors are non-persistent
registered buffers, which model.to(device) moves, so after the move every tensor a module holds as a plain attribute is
moved by hand as well (move_stray_tensors), every parameter, buffer and tensor attribute is checked to be on the device,
and --gpu-smoke checks V0 on the GPU against V0 on the CPU, and the CPU's V0 against exp21.logits_grid itself, before
anything else runs.

Outputs (exp/out by default; the cluster job writes to scratch): exp101_fit.json and exp101_units_fit.npz (--fit),
exp101_summary.json and exp101_units.npz (the test), exp101_inventory.json, exp101_gpusmoke.json, and *_smoke files
(--smoke). Per unit: the set, the label, the errors, the confidence, the 320 views' classes and total-variation
distances to V0, the instability; nothing written holds a coordinate, a transform, a bound, a window's name or a plot's
identifier (e98.dump refuses positional keys; the npz holds per-unit arrays only).

Interpretations of the page, each a choice the page does not settle:
  a. On E the share q set aside is a share of the design weight, as the 10% review is (on F every weight is 1); a group
     of units with tied confidence is set aside whole or not at all.
  b. A crop that would leave its window is clamped at the window's edge by exp21's rule (exp89.crop_at), as exp21 and
     exp89 cropped; such a unit keeps all 320 views, some of them identical, and is counted ("clamped").
  c. A unit in two windows takes the first (dataset roots, then groups, then names, sorted, as the read jobs list
     them) whose twelve item groups are complete and where no crop is clamped, else the first whose twelve groups are
     complete; windows are never mosaicked. Clamped units are counted among errors and correct units separately.
  d. Item group k is month index k, as exp21's replica reads the label windows; the season views keep those indices.
     Timestamps are exp21's: day 15, the month index, 2023 (the year is not read under fast_pass).
  e. The total-variation distance is half the L1 distance between ten-channel softmaxes.
  f. The bootstraps resample units within each country (one stratum on F), the same resamples for every statistic, and
     apply the rule anew in each resample at the fixed q; a resample without an error or without a correct unit is
     skipped and counted.
  g. On E the weights are recomputed over the plots with all 320 views (each country's share of the region over its
     plots with views), and confidence alone's capture is recomputed on the same plots; with all 271 it is exp99's.
  h. The dose draws one subset of views per draw, the same for every unit.
  i. The test recomputes F's views and q; it refuses when q differs from the fit record's.
  j. "Holds the most errors" is the record's tie-aware capture times the errors (a tied group on the cut counted in
     proportion); two q tie when they hold the same within 1e-9. P1's and P2's intervals must lie strictly above 0.
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
for _p in (ROOT, EXP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from oe_inferencex import metrics                   # noqa: E402
import exp89_finetuned_checkpoints as e89           # noqa: E402
import exp98_awf_deployment as e98                  # noqa: E402
import exp99_transfer as e99                        # noqa: E402
import timesync_awf_crosswalk as tsx                # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
PLAN = os.path.join(ROOT, "docs", "plan", "two_step.md")

N_OUT = e98.N_OUT                     # the decoder's ten channels (nine trained, channel 9 untrained)
OLMO_BANDS = e89.OLMO_BANDS           # OlmoEarth's Sentinel-2 band order
N_GROUPS = 12                         # monthly item groups, sentinel2 then sentinel2.1 ... sentinel2.11
PATCH = 4                             # the replica's patch size (exp21)
N_BANDSETS = 3                        # OlmoEarth v1's Sentinel-2 band sets (tokens per patch and timestep)
DAY, YEAR = 15, 2023                  # exp21's legacy timestamps (day, month index, year); the year is not read

# the views: 8 x 5 x 2 x 4 = 320, view 0 is V0
ORIENTATIONS = tuple((k, f) for f in (False, True) for k in range(4))   # exp36's TRANSFORMS: (quarter turns, flip)
QUARTERS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))
TIMES = ((),) + QUARTERS              # the months removed: none, or one quarter
TIME_NAMES = ("all twelve months", "without January to March", "without April to June", "without July to September",
              "without October to December")
CONTEXTS = (16, 32)                   # crop sides in px; multiples of PATCH, so the patch grid maps onto itself
PHASES = (0, 1, 2, 3)                 # the crop's origin moved by this many px along both axes
SHAPE = (len(ORIENTATIONS), len(TIMES), len(CONTEXTS), len(PHASES))
N_VIEWS = int(np.prod(SHAPE))
V0 = 0

# the rule and the predictions (confirmed by the owner on 10 October 2026)
Q_GRID = (0.5, 0.6, 0.7, 0.8)
BUDGET = 0.10
BUDGETS = (0.05, 0.10, 0.20)
P1_MARGIN = 0.05                      # P1: the review's share of errors at least 5 points above confidence alone's
LEVEL = 0.90                          # the bootstrap percentile intervals
BOOT = 2000
SMOKE_BOOT = 200
FLOORS = {"min_with_views": 250, "min_errors": 20, "min_correct": 20}
SMOKE_FLOORS = {"min_with_views": 15, "min_errors": 3, "min_correct": 3}
DOSE_K = (4, 8, 16, 32, 64, 128, 319)
DOSE_DRAWS = 20
SEED = 101
TIE = 1e-9                            # q's tie rule: errors held equal within this
EPS = 1e-12

# what exp98 and exp99 recorded (exp/out/exp98_summary.json graded_validation; exp/out/exp99_summary.json rules and
# rules.strict.ranking.weighted.capture); the run refuses a reading that does not give them
RECORDED = {"F": {"n": 259, "errors": 33},
            "E": {"n": 271, "strict": 91, "lenient": 37, "confidence_capture_0.1_strict": 0.17870709922096475}}
CAPTURE_TOL = 1e-9

# compute
BATCH_TOKENS = 1 << 17                # encoder tokens per forward: (side / 4)^2 x T x 3 per view
GPU_SMOKE_UNITS = 32
GPU_TOL = 1e-3                        # V0's logits on the GPU against the CPU, absolute
REPLICA_TOL = 1e-4                    # this file's sample builder against exp21.logits_grid on the CPU, absolute

CANNOT = ("Whether the gain holds for a model or legend other than FT-AWF's, or for a whole map: the units are points.",
          "Why a view changes the class: a season removed may hide the season that separates grass from shrub, or show "
          "the model a series shorter than it was trained on.",
          "With 271 plots and a 10% review of about 27, a difference of a few errors is within noise; P1's bar is set "
          "so that only a large gain passes.")
READINGS = {"both": "P1 and P2 hold: shaking the less confident part finds errors confidence misses, under transfer; "
                    "the rule becomes a candidate for the package, which needs the model's own inference (the "
                    "package now reads scores only)",
            "P2 only": "P2 holds, P1 does not: the shaking separates errors, but too few to change a 10% review at "
                       "this sample size",
            "neither": "Neither holds: under transfer, as at Ai2's points and on the earlier testbeds, the views add "
                       "nothing confidence does not carry",
            "P1 only": "P1 holds, P2 does not: not one of the page's cases; reported as it stands",
            "not graded": "below a floor or not computable: the predictions are reported without verdicts"}


# ----------------------------------------------------------------------------- helpers
def prereg_status(text=None):
    if text is None:
        with open(PLAN, encoding="utf-8") as f:
            text = f.read()
    return e89.prereg_status(text)


def bkey(b):
    return e89.bkey(b)


def finite(x):
    return e98.finite(x)


def _interval(v, level=LEVEL):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=np.float64)
    if not v.size:
        return {"n_resamples": 0}
    a = (1 - level) / 2
    return {"n_resamples": int(v.size), "lo90": float(np.quantile(v, a)), "hi90": float(np.quantile(v, 1 - a))}


# ----------------------------------------------------------------------------- the views
def view_index(o, t, c, p):
    """The view's index from its orientation, time, context and phase settings (row-major in SHAPE)."""
    return int(np.ravel_multi_index((o, t, c, p), SHAPE))


def view_settings(v):
    """(orientation, time, context, phase) of view v."""
    return tuple(int(i) for i in np.unravel_index(int(v), SHAPE))


# each kind of perturbation alone: its views with the other three kinds at V0's setting
KINDS = {"orientation": [view_index(o, 0, 0, 0) for o in range(1, SHAPE[0])],
         "time": [view_index(0, t, 0, 0) for t in range(1, SHAPE[1])],
         "context": [view_index(0, 0, c, 0) for c in range(1, SHAPE[2])],
         "phase": [view_index(0, 0, 0, p) for p in range(1, SHAPE[3])]}


def season_months(t):
    """The month indices a time setting keeps: all twelve (t = 0), or the year without quarter t - 1."""
    return tuple(m for m in range(N_GROUPS) if m not in TIMES[t])


def season_inputs(series, t):
    """(the series with the removed quarter's item groups left out, the month indices of those left). `series` holds the
    twelve item groups on its second-to-last axis, in month order; the groups left keep their own month indices, so a
    9-step series is January to December without one quarter, never re-indexed 0 to 8."""
    months = season_months(t)
    return np.asarray(series)[..., list(months), :], months


def timestamps(months):
    """exp21's timestamps for a series: one (day 15, month index, 2023) row per step, each the group's own month."""
    return np.array([[DAY, int(m), YEAR] for m in months], dtype=np.int64)


def apply_t(a, k, f, axes):
    """exp36's dihedral transform: k quarter turns, then a flip along the second of `axes` when f."""
    out = np.rot90(a, k, axes=axes)
    return np.flip(out, axis=axes[1]) if f else out


def invert_t(a, k, f, axes):
    """The inverse of apply_t(., k, f, axes)."""
    out = np.flip(a, axis=axes[1]) if f else a
    return np.rot90(out, -k, axes=axes)


def crop_plan(H, W, r, c):
    """Every crop a unit at (r, c) of an H x W window needs, by exp89.crop_at (exp21's rule, clamped at the window's
    edges): ({(context, phase): (r0, c0, rr, cc)}, the block (r_lo, c_lo, r_hi, c_hi) holding them all, whether any crop
    was clamped). (r0, c0) is the crop's corner in the window, (rr, cc) the unit in the crop."""
    plan, clamped = {}, False
    for ci, S in enumerate(CONTEXTS):
        for p in PHASES:
            r0, c0, rr, cc = e89.crop_at(H, W, r, c, S, p)
            clamped |= (rr, cc) != (S // 2 - p, S // 2 - p)
            plan[(ci, p)] = (int(r0), int(c0), int(rr), int(cc))
    r_lo = min(v[0] for v in plan.values())
    c_lo = min(v[1] for v in plan.values())
    r_hi = max(v[0] + CONTEXTS[k[0]] for k, v in plan.items())
    c_hi = max(v[1] + CONTEXTS[k[0]] for k, v in plan.items())
    return plan, (r_lo, c_lo, r_hi, c_hi), bool(clamped)


def make_unit(raw_block, plan, box, clamped):
    """A unit's input in memory: its block (bh, bw, 12 groups, 12 bands, raw values) and each crop relative to it."""
    r_lo, c_lo = box[0], box[1]
    rel = {k: (v[0] - r_lo, v[1] - c_lo, v[2], v[3]) for k, v in plan.items()}
    return {"raw": np.asarray(raw_block, dtype=np.float32), "plan": rel, "clamped": clamped}


def view_input(block, plan, v):
    """(the view's input (S, S, T, 12), its month indices, its transform (k, f), the unit's pixel (rr, cc) in the crop
    as the output is read after the inverse transform) for view v of a unit whose block is `block` (normalized or
    raw). run_views writes the same array straight into its batch."""
    o, t, ci, p = view_settings(v)
    S = CONTEXTS[ci]
    r0, c0, rr, cc = plan[(ci, p)]
    x, months = season_inputs(block[r0:r0 + S, c0:c0 + S], t)
    k, f = ORIENTATIONS[o]
    return np.ascontiguousarray(apply_t(x, k, f, axes=(0, 1))), months, (k, f), (rr, cc)


def run_views(model, units, batch_tokens=BATCH_TOKENS, views=None, log=print):
    """Logits (N, N_VIEWS, 10) at each unit's pixel (NaN for views not asked for), and the run's counts. The views of
    all units go through the model in batches of one crop size and one series length, at most `batch_tokens` encoder
    tokens each; each output is mapped back by the inverse transform before the unit's pixel is read."""
    views = list(range(N_VIEWS)) if views is None else [int(v) for v in views]
    out = np.full((len(units), N_VIEWS, N_OUT), np.nan, dtype=np.float64)
    blocks = [model.normalize(u["raw"]) for u in units]
    t0, n_views, n_batches = time.time(), 0, 0

    def flush(buf):
        nonlocal n_views, n_batches
        if not buf:
            return
        _, t_first, ci_first, _ = view_settings(buf[0][1])
        S, T = CONTEXTS[ci_first], len(season_months(t_first))
        X = np.empty((len(buf), S, S, T, len(OLMO_BANDS)), dtype=np.float32)
        M = np.empty((len(buf), T), dtype=np.int64)
        meta = []
        for j, (i, v) in enumerate(buf):
            o, t, ci, p = view_settings(v)
            r0, c0, rr, cc = units[i]["plan"][(ci, p)]
            k, f = ORIENTATIONS[o]
            months = season_months(t)
            # view_input's array, written straight into the batch: the crop, transformed, its months taken
            np.take(apply_t(blocks[i][r0:r0 + S, c0:c0 + S], k, f, axes=(0, 1)), months, axis=2, out=X[j],
                    mode="clip")
            M[j] = months
            meta.append((i, v, k, f, rr, cc))
        L = np.asarray(model.pixel_logits(X, M), dtype=np.float64)
        for j, (i, v, k, f, rr, cc) in enumerate(meta):
            out[i, v] = invert_t(L[j], k, f, axes=(0, 1))[rr, cc]
        n_views += len(buf)
        n_batches += 1
        if n_batches % 50 == 0:
            log(f"  {n_views} views in {n_batches} batches, {time.time() - t0:.0f} s")
        buf.clear()

    for ci, S in enumerate(CONTEXTS):
        for T in sorted({len(season_months(t)) for t in range(SHAPE[1])}, reverse=True):
            want = [v for v in views if view_settings(v)[2] == ci and len(season_months(view_settings(v)[1])) == T]
            if not want:
                continue
            bs = max(1, int(batch_tokens) // ((S // PATCH) ** 2 * T * N_BANDSETS))
            buf = []
            for i in range(len(units)):
                for v in want:
                    buf.append((i, v))
                    if len(buf) >= bs:
                        flush(buf)
            flush(buf)
    secs = time.time() - t0
    return out, {"views": n_views, "batches": n_batches, "seconds": round(secs, 2),
                 "views_per_second": round(n_views / secs, 1) if secs > 0 else None}


# ----------------------------------------------------------------------------- the instability
def softmax(L, axis=-1):
    L = np.asarray(L, dtype=np.float64)
    z = np.exp(L - L.max(axis=axis, keepdims=True))
    return z / z.sum(axis=axis, keepdims=True)


def view_readings(L):
    """(classes (N, V), distances (N, V)) from logits at the units (N, V, 10): each view's class, the argmax over the
    ten channels, and the total-variation distance between its ten-channel softmax and V0's (0 at V0)."""
    L = np.asarray(L, dtype=np.float64)
    P = softmax(L)
    return L.argmax(-1), 0.5 * np.abs(P - P[:, V0:V0 + 1]).sum(-1)


def instability(classes, tv, views=None):
    """(share, mean distance) per unit: the share of `views` (default every view but V0) whose class differs from V0's,
    and their mean total-variation distance to V0, the share's tie-break."""
    classes, tv = np.asarray(classes), np.asarray(tv, dtype=np.float64)
    views = np.arange(1, classes.shape[1]) if views is None else np.asarray(views, dtype=np.int64)
    return (classes[:, views] != classes[:, [V0]]).mean(1), tv[:, views].mean(1)


def lex_rank(*keys):
    """Dense ranks of the units ordered by `keys`, the first key primary, every key ascending: units equal on every key
    share a rank. Returned as floats, so a higher rank is later in the ascending order."""
    keys = [np.asarray(k, dtype=np.float64).ravel() for k in keys]
    n = keys[0].size
    if n == 0:
        return np.zeros(0)
    order = np.lexsort(keys[::-1])
    K = np.stack([k[order] for k in keys], 1)
    new = np.r_[True, (K[1:] != K[:-1]).any(1)]
    rank = np.empty(n)
    rank[order] = np.cumsum(new) - 1
    return rank


def instability_score(share, mean_tv):
    """s as one suspicion score: the share of views whose class differs from V0's, ties broken by the mean
    total-variation distance; higher is more unstable."""
    return lex_rank(share, mean_tv)


# ----------------------------------------------------------------------------- the rule
def set_aside(u, q, weights):
    """The most confident share q of the units, in weight (u is the confidence's suspicion, minus the log of the top-1
    probability: lower is more confident). Units are taken from the most confident down in groups of equal u, a group
    whole or not at all, while the weight up to the group's end is at most q of the total."""
    u = np.asarray(u, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    aside = np.zeros(u.size, dtype=bool)
    if not u.size or not w.sum() > 0:
        return aside
    order = np.argsort(u, kind="stable")
    s, cw = u[order], np.cumsum(w[order])
    last = np.r_[s[1:] != s[:-1], True]
    end = np.empty(u.size)
    idx = np.flatnonzero(last)
    grp = np.cumsum(np.r_[True, s[1:] != s[:-1]]) - 1
    end[:] = cw[idx][grp]                               # the cumulative weight at the end of each unit's group
    aside[order] = end <= q * cw[-1] + EPS
    return aside


def two_step_score(u, share, mean_tv, q, weights):
    """R(q) as a suspicion score (higher is reviewed first) and the units set aside: the units not set aside first, by
    instability (the share, then the mean distance), most unstable first, ties by confidence, least confident first;
    then the units set aside, least confident first. Units equal on all of it tie."""
    aside = set_aside(u, q, weights)
    keep = ~aside
    score = lex_rank(keep.astype(np.float64), np.where(keep, share, 0.0), np.where(keep, mean_tv, 0.0), u)
    return score, aside


def capture(score, err, weights, budgets=BUDGETS):
    """The record's capture: the share of the weighted errors inside the most suspect share b of the weight, a tied
    group straddling the cut counted in proportion (oe_inferencex.metrics.weighted_capture_at_budget)."""
    return metrics.weighted_capture_at_budget(score, err, weights, budgets)


def review_inclusion(score, weights, budget):
    """Per unit, the part of it inside the review of the most suspect share `budget` of the weight, under the same
    convention as capture: 1 inside, 0 outside, the tied group on the cut in proportion."""
    sc = np.asarray(score, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    out = np.zeros(sc.size)
    if not sc.size or not w.sum() > 0:
        return out
    order = np.argsort(-sc, kind="stable")
    s, ww = sc[order], w[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    grp = np.cumsum(np.r_[True, s[1:] != s[:-1]]) - 1
    w_group = np.add.reduceat(ww, starts)
    w_before = np.r_[0.0, np.cumsum(ww)][starts]
    cut = budget * ww.sum()
    full = (w_before + w_group) <= cut + 1e-12
    frac = np.where(full, 1.0, np.where(w_before < cut, (cut - w_before) / np.maximum(w_group, 1e-300), 0.0))
    first_part = np.flatnonzero(~full & (w_before < cut))
    if first_part.size > 1:                             # only the first group past the full ones straddles the cut
        frac[first_part[1:]] = 0.0
    out[order] = frac[grp]
    return out


def choose_q(u, share, mean_tv, err, weights=None, grid=Q_GRID, budget=BUDGET, tie=TIE):
    """(q, per-q rows): the q of `grid` whose review of the top `budget` of R(q) holds the most errors (capture times
    the weighted errors, so a count on F where every weight is 1), ties within `tie` to the larger q."""
    err = np.asarray(err, dtype=np.float64)
    w = np.ones(err.size) if weights is None else np.asarray(weights, dtype=np.float64)
    rows = {}
    for q in grid:
        score, aside = two_step_score(u, share, mean_tv, q, w)
        cap = capture(score, err, w, (budget,))[budget]
        rows[q] = {"q": q, "capture": cap, "errors_held": cap * float((err * w).sum()),
                   "n_set_aside": int(aside.sum())}
    best = max(r["errors_held"] for r in rows.values())
    q = max(q for q in grid if rows[q]["errors_held"] >= best - tie)
    return q, rows


def statistics(u, s, share, mean_tv, err, weights, q, budgets=BUDGETS):
    """The page's statistics at one q: the review's share of errors by R(q) and by confidence alone at each budget,
    and, among the units R(q) does not set aside, the design-weighted AUROC of s and of confidence."""
    u, s, err, w = (np.asarray(a, dtype=np.float64) for a in (u, s, err, weights))
    score, aside = two_step_score(u, share, mean_tv, q, w)
    capR, capC = capture(score, err, w, budgets), capture(u, err, w, budgets)
    keep = ~aside
    aR = metrics.weighted_auroc(s[keep], err[keep], w[keep])
    aC = metrics.weighted_auroc(u[keep], err[keep], w[keep])
    review = {}
    for b in budgets:
        iR, iC = review_inclusion(score, w, b), review_inclusion(u, w, b)
        review[bkey(b)] = {"two_step": capR[b], "confidence": capC[b], "difference": capR[b] - capC[b],
                           "errors_in_review_two_step": float((iR * err).sum()),
                           "errors_in_review_confidence": float((iC * err).sum()),
                           "units_in_review": float(iR.sum())}
    return {"n": int(err.size), "n_errors": int(err.sum()), "error_rate_weighted": metrics.weighted_mean(err, w),
            "q": q, "set_aside": {"n": int(aside.sum()), "weight_share": float(w[aside].sum() / w.sum()),
                                  "n_errors": int(err[aside].sum())},
            "review": review,
            "not_set_aside": {"n": int(keep.sum()), "n_errors": int(err[keep].sum()), "auroc_instability": aR,
                              "auroc_confidence": aC, "difference": aR - aC}}


def stratified_resampler(strata, n):
    """A function rng -> indices: the units resampled with replacement within each stratum (one stratum for None)."""
    st = np.zeros(n, dtype=np.int64).astype(str) if strata is None else np.asarray(strata).astype(str)
    groups = [np.flatnonzero(st == g) for g in np.unique(st)]

    def draw(rng):
        return np.concatenate([g[rng.integers(0, g.size, g.size)] for g in groups])
    return draw


def paired_bootstrap(u, share, mean_tv, err, weights, strata, q, n_boot=BOOT, seed=SEED, budgets=BUDGETS):
    """The 90% percentile intervals of the paired differences (R(q) minus confidence alone at each budget; the AUROC of
    s minus that of confidence among the units not set aside), from the same within-stratum resamples, the rule applied
    anew in each. A resample without an error or without a correct unit is skipped; one whose units not set aside lack
    either gives no AUROC difference."""
    u, share, mean_tv, err, w = (np.asarray(a, dtype=np.float64) for a in (u, share, mean_tv, err, weights))
    s = instability_score(share, mean_tv)
    draw = stratified_resampler(strata, u.size)
    rng = np.random.default_rng(seed)
    caps = {b: [] for b in budgets}
    aucs, skipped = [], 0
    for _ in range(int(n_boot)):
        idx = draw(rng)
        e = err[idx]
        if not 0 < e.sum() < e.size:
            skipped += 1
            continue
        st = statistics(u[idx], s[idx], share[idx], mean_tv[idx], e, w[idx], q, budgets)
        for b in budgets:
            caps[b].append(st["review"][bkey(b)]["difference"])
        aucs.append(st["not_set_aside"]["difference"])
    return {"n_boot": int(n_boot), "skipped_no_error_or_no_correct": skipped, "seed": seed, "level": LEVEL,
            "resampling": "units with replacement within each stratum (country on E, one stratum on F), the rule "
                          "applied anew in each resample at the fixed q",
            "review_difference": {bkey(b): _interval(caps[b]) for b in budgets},
            "auroc_difference": _interval(aucs)}


def ranking_record(u, share, mean_tv, err, weights, strata, q, n_boot, seed=SEED, budgets=BUDGETS):
    """statistics at q, with the paired bootstrap when n_boot > 0."""
    s = instability_score(share, mean_tv)
    rec = statistics(u, s, share, mean_tv, err, weights, q, budgets)
    if n_boot:
        rec["bootstrap"] = paired_bootstrap(u, share, mean_tv, err, weights, strata, q, n_boot, seed, budgets)
    return rec


def grade(rec, n_with_views, n_errors, n_correct, floors=FLOORS, margin=P1_MARGIN):
    """P1 and P2 on E under STRICT, at their floors."""
    ok = n_with_views >= floors["min_with_views"] and n_errors >= floors["min_errors"] \
        and n_correct >= floors["min_correct"]
    why = (f"fewer than {floors['min_with_views']} plots with all {N_VIEWS} views read, {floors['min_errors']} STRICT "
           f"errors or {floors['min_correct']} correct plots among them")
    r10 = rec["review"][bkey(BUDGET)]
    bt = rec.get("bootstrap") or {}
    iv1 = (bt.get("review_difference") or {}).get(bkey(BUDGET)) or {}
    iv2 = bt.get("auroc_difference") or {}
    rows = {
        "P1": {"rule": f"the {BUDGET:.0%} review by R(q) holds a share of the STRICT errors at least {margin:.2f} "
                       "above confidence alone's, design-weighted, and the 90% within-country bootstrap interval of "
                       "the paired difference lies above 0",
               "value": r10["difference"], "two_step": r10["two_step"], "confidence": r10["confidence"],
               "interval": iv1, "threshold": margin},
        "P2": {"rule": "among the units R(q) does not set aside, the design-weighted AUROC of s for the STRICT errors "
                       "exceeds confidence's, with the 90% bootstrap interval of the difference above 0",
               "value": rec["not_set_aside"]["difference"],
               "auroc_instability": rec["not_set_aside"]["auroc_instability"],
               "auroc_confidence": rec["not_set_aside"]["auroc_confidence"], "interval": iv2, "threshold": 0.0}}
    for p, row in rows.items():
        v, lo = row["value"], row["interval"].get("lo90")
        if not (finite(v) and finite(lo)):
            row.update(graded=False, reason="not computable on these inputs")
        elif not ok:
            row.update(graded=False, reason=why)
        elif p == "P1":
            row.update(graded=True, holds=bool(v >= margin - EPS and lo > 0))
        else:
            row.update(graded=True, holds=bool(v > 0 and lo > 0))
    rows["floors"] = dict(floors, n_with_views=int(n_with_views), n_errors=int(n_errors), n_correct=int(n_correct))
    return rows


def reading(prereg):
    p1, p2 = prereg["P1"], prereg["P2"]
    if not (p1.get("graded") and p2.get("graded")):
        return READINGS["not graded"]
    key = {(True, True): "both", (False, True): "P2 only", (False, False): "neither",
           (True, False): "P1 only"}[(p1["holds"], p2["holds"])]
    return READINGS[key]


# ----------------------------------------------------------------------------- reported, not graded
def kinds_record(classes, tv, u, err, w, strata, q, n_boot):
    """Each kind of perturbation alone: s from its views with the other three kinds at V0's setting."""
    out = {}
    for kind, views in KINDS.items():
        sh, mt = instability(classes, tv, views)
        out[kind] = {"n_views": len(views), "mean_share_errors": float(sh[err > 0].mean()) if (err > 0).any() else None,
                     "mean_share_correct": float(sh[err == 0].mean()) if (err == 0).any() else None,
                     "ranking": ranking_record(u, sh, mt, err, w, strata, q, n_boot)}
    return out


def dose_record(classes, tv, u, err, w, q, ks=DOSE_K, draws=DOSE_DRAWS, seed=SEED):
    """s from k views drawn at random from the 319 (one draw of views for every unit), `draws` draws per k: the
    review's difference at 10% and the AUROC difference among the units not set aside, summarised over the draws."""
    rng = np.random.default_rng(seed)
    others = np.arange(1, N_VIEWS)
    out = {}
    for k in ks:
        d10, dau, cap = [], [], []
        for _ in range(draws):
            views = np.sort(rng.choice(others, size=int(k), replace=False))
            sh, mt = instability(classes, tv, views)
            st = statistics(u, instability_score(sh, mt), sh, mt, err, w, q, (BUDGET,))
            r = st["review"][bkey(BUDGET)]
            d10.append(r["difference"])
            cap.append(r["two_step"])
            dau.append(st["not_set_aside"]["difference"])

        def summ(v):
            v = np.asarray([x for x in v if np.isfinite(x)])
            return {"mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()),
                    "max": float(v.max())} if v.size else None
        out[str(k)] = {"draws": draws, "review_difference_0.1": summ(d10), "two_step_capture_0.1": summ(cap),
                       "auroc_difference": summ(dau)}
    return out


def grass_shrub_record(u, share, mean_tv, strict, lenient, w, q, budget=BUDGET):
    """The plots where STRICT and LENIENT part (their errors differ): how many, their instability against the others',
    and how much of them each review takes (in plots, tie-aware)."""
    part = np.asarray(strict) != np.asarray(lenient)
    score, _ = two_step_score(u, share, mean_tv, q, w)
    iR, iC = review_inclusion(score, w, budget), review_inclusion(u, w, budget)
    return {"n": int(part.sum()), "weight_share": float(w[part].sum() / w.sum()) if w.sum() > 0 else None,
            "mean_share_parting": float(share[part].mean()) if part.any() else None,
            "mean_share_others": float(share[~part].mean()) if (~part).any() else None,
            "in_review_two_step": float(iR[part].sum()), "in_review_confidence": float(iC[part].sum()),
            "budget": budget}


# ----------------------------------------------------------------------------- the units and their readings
def load_fit_units(scores, labels, outcome=True, recorded=None):
    """F: exp98's graded validation points on the 2023 map, read as exp98 graded them. Returns a dict with the points
    (in memory only: CRS, centre of the labelled pixel, label window), and with `outcome` the label, the deployed class,
    the confidence (p1 and its suspicion u = -log p1) and the error. `recorded` ({n, errors}) is checked."""
    entries, _, _ = e98.find_outputs(scores)
    grids = [e98.open_grid(e) for e in entries]
    points, _ = e98.read_label_windows(labels)
    gi, rows, cols, how, off, P, _, _, _ = e98.read_points(grids, e98.locate_all(points, grids))
    cov = e98.covered(P)
    split = np.array([p["split"] for p in points], dtype=object)
    V = np.flatnonzero(cov & (split == "val"))
    S = {"name": "F", "points": [points[k] for k in V], "n": int(V.size),
         "positions": [{"crs": points[k]["crs"], "X": points[k]["X"], "Y": points[k]["Y"]} for k in V],
         "n_grids": len(grids)}
    if not outcome:
        return S
    rd = e98.readings(P[V])
    if V.size and rd["sum_dev"] > e98.SUM_TOLERANCE:
        raise ValueError(f"the probabilities at F's points sum to 1 +- {rd['sum_dev']:.3g}: not the softmax")
    lab = np.array([points[k]["label"] for k in V], dtype=np.int64)
    err = (rd["pred"] != lab).astype(np.float64)
    if recorded is not None and (V.size, int(err.sum())) != (recorded["n"], recorded["errors"]):
        raise ValueError(f"F: {V.size} points and {int(err.sum())} errors, where exp98 recorded {recorded['n']} and "
                         f"{recorded['errors']}: not the graded reading; nothing is shaken")
    S.update(label=lab, pred=np.asarray(rd["pred"], dtype=np.int64), u=np.asarray(rd["u"], dtype=np.float64),
             p1=np.asarray(rd["p1"], dtype=np.float64), wrong=err, wrong_lenient=np.full(V.size, np.nan),
             weight=np.ones(V.size), strata=None, country=np.full(V.size, -1, dtype=np.int64),
             reproduces_record=recorded is not None)
    return S


def load_test_units(scores, timesync, request_geometry, boundaries=None, outcome=True, recorded=None,
                    fixed=e99.FIXED_RULE, geometry_sha256=tsx.AWF_GEOMETRY_SHA256, min_plots=e99.MIN_PLOTS):
    """E: exp99's plots with 2017 input, read as exp99 graded them. Returns a dict with the plots' WGS84 positions (in
    memory only) and with `outcome` the TimeSync label, the deployed class, the confidence, the STRICT and LENIENT
    errors, the country and the region's areas; `recorded` checks the counts and confidence alone's 10% capture."""
    from rasterio.crs import CRS
    P, sel, areas = e99.select(timesync, request_geometry, boundaries, geometry_sha256, min_plots, fixed)
    grids, _, _ = e99.read_map(scores)
    gi, Pr, _, _, _ = e99.read_plots_on(grids, P)
    cov = e98.covered(Pr)
    K = np.flatnonzero(cov)
    wgs = CRS.from_epsg(4326)
    S = {"name": "E", "n": int(K.size), "n_selected": int(len(P["label"])), "n_grids": len(grids),
         "positions": [{"crs": wgs, "X": float(P["lon"][k]), "Y": float(P["lat"][k])} for k in K]}
    ctry_all = np.asarray(P["country"])
    S["by_country"] = {c: int((ctry_all[K] == c).sum()) for c in tsx.COUNTRIES}
    if not outcome:
        return S
    if areas is None:
        raise ValueError("E needs --boundaries: each country's share of the region weights the strata")
    rd = e98.readings(Pr[K])
    if K.size and rd["sum_dev"] > e98.SUM_TOLERANCE:
        raise ValueError(f"the probabilities at E's plots sum to 1 +- {rd['sum_dev']:.3g}: not the softmax")
    lab = np.asarray(P["label"])[K]
    ctry = ctry_all[K]
    pred = np.asarray(rd["pred"], dtype=np.int64)
    strict, lenient = tsx.wrong(lab, pred, "strict"), tsx.wrong(lab, pred, "lenient")
    n_by = {c: int((ctry == c).sum()) for c in tsx.COUNTRIES}
    _, w_plot = tsx.weights_from_areas(areas["by_country_km2"], n_by)
    w = np.array([w_plot[c] for c in ctry], dtype=np.float64)
    u = np.asarray(rd["u"], dtype=np.float64)
    cap = capture(u, strict, w, (BUDGET,))[BUDGET] if K.size else float("nan")
    if recorded is not None:
        got = {"n": int(K.size), "strict": int(strict.sum()), "lenient": int(lenient.sum())}
        want = {k: recorded[k] for k in got}
        if got != want or not abs(cap - recorded["confidence_capture_0.1_strict"]) <= CAPTURE_TOL:
            raise ValueError(f"E: {got} and confidence alone's 10% capture {cap!r}, where exp99 recorded {want} and "
                             f"{recorded['confidence_capture_0.1_strict']!r}: not the graded reading; nothing is "
                             "shaken")
    cidx = {c: i for i, c in enumerate(tsx.COUNTRIES)}
    tidx = {t: i for i, t in enumerate(tsx.TIMESYNC_CLASSES)}
    S.update(label=np.array([tidx[t] for t in lab], dtype=np.int64), pred=pred, u=u,
             p1=np.asarray(rd["p1"], dtype=np.float64), wrong=strict, wrong_lenient=lenient, weight=w,
             strata=ctry, country=np.array([cidx[c] for c in ctry], dtype=np.int64), areas=areas,
             confidence_capture_0_1_all=cap, reproduces_record=recorded is not None)
    return S


# ----------------------------------------------------------------------------- the deployed run's windows
def layer_dir(k):
    return "sentinel2" if k == 0 else f"sentinel2.{k}"


def band_set_dir(wdir, k):
    """The item group's band set directory holding the twelve bands (exp98's and exp99's datasets keep one band set of
    all twelve at 10 m), or None."""
    base = os.path.join(wdir, "layers", layer_dir(k))
    if not os.path.isdir(base):
        return None
    for d in sorted(os.listdir(base)):
        if set(d.split("_")) == set(OLMO_BANDS) and os.path.exists(os.path.join(base, d, "geotiff.tif")):
            return os.path.join(base, d)
    return None


def deployed_windows(roots):
    """Every window of the rslearn dataset roots, in the order the read jobs list them (the roots in order, then the
    groups and the names sorted): its directory, projection, bounds in projection pixels, and whether all twelve item
    groups carry rslearn's completed marker. In memory only."""
    out = []
    for root in roots:
        wroot = os.path.join(root, "windows")
        if not os.path.isdir(wroot):
            raise FileNotFoundError(f"{root} holds no windows/ (the scratch purge?)")
        for group in sorted(os.listdir(wroot)):
            gdir = os.path.join(wroot, group)
            if not os.path.isdir(gdir):
                continue
            for name in sorted(os.listdir(gdir)):
                wdir = os.path.join(gdir, name)
                mpath = os.path.join(wdir, "metadata.json")
                if not os.path.isfile(mpath):
                    continue
                with open(mpath) as f:
                    meta = json.load(f)
                pr = meta.get("projection") or {}
                done = sum(os.path.exists(os.path.join(wdir, "layers", layer_dir(k), "completed"))
                           for k in range(N_GROUPS))
                out.append({"dir": wdir, "crs": str(pr.get("crs")), "res": (float(pr["x_resolution"]),
                                                                           float(pr["y_resolution"])),
                            "b": [int(v) for v in meta["bounds"]], "complete": done == N_GROUPS})
    return out


def locate_units(positions, windows):
    """Per unit, every window holding its pixel, in the windows' order: [(window index, row, column)]. A position is
    taken into each window's CRS (reprojected where they differ) and its projection pixel is floor(X / x_resolution),
    floor(Y / y_resolution), as rslearn's windows and exp98's compare_inputs count pixels."""
    from rasterio.crs import CRS
    from rasterio.warp import transform as warp
    cands = [[] for _ in positions]
    by_src = {}
    for k, p in enumerate(positions):
        by_src.setdefault(CRS.from_user_input(p["crs"]).to_string(), []).append(k)
    by_dst = {}
    for j, w in enumerate(windows):
        by_dst.setdefault(CRS.from_user_input(w["crs"]).to_string(), []).append(j)
    for src, idx in by_src.items():
        X = np.array([positions[k]["X"] for k in idx], dtype=np.float64)
        Y = np.array([positions[k]["Y"] for k in idx], dtype=np.float64)
        for dst, js in by_dst.items():
            xs, ys = (X, Y) if src == dst else (np.asarray(v, dtype=np.float64) for v in warp(src, dst, X, Y))
            for j in js:
                w = windows[j]
                px = np.floor(xs / w["res"][0]).astype(np.int64)
                py = np.floor(ys / w["res"][1]).astype(np.int64)
                b = w["b"]
                inside = (px >= b[0]) & (px < b[2]) & (py >= b[1]) & (py < b[3])
                for i in np.flatnonzero(inside):
                    cands[idx[i]].append((j, int(py[i] - b[1]), int(px[i] - b[0])))
    for c in cands:
        c.sort()
    return cands


def read_window_block(win, box):
    """(bh, bw, 12 groups, 12 bands) raw values of a window's twelve item groups over box (r_lo, c_lo, r_hi, c_hi), in
    OlmoEarth's band order."""
    import rasterio
    from rasterio.windows import Window
    r_lo, c_lo, r_hi, c_hi = box
    H, W = win["b"][3] - win["b"][1], win["b"][2] - win["b"][0]
    out = np.zeros((r_hi - r_lo, c_hi - c_lo, N_GROUPS, len(OLMO_BANDS)), dtype=np.float32)
    for k in range(N_GROUPS):
        d = band_set_dir(win["dir"], k)
        if d is None:
            raise FileNotFoundError(f"item group {k}: no band set of the twelve bands")
        names = os.path.basename(d).split("_")
        with rasterio.open(os.path.join(d, "geotiff.tif")) as src:
            if (src.height, src.width) != (H, W) or src.count != len(names):
                raise ValueError(f"item group {k}: {src.count} bands of {src.height} x {src.width}, where the window "
                                 f"is {H} x {W} with {len(names)} bands")
            a = src.read(window=Window(c_lo, r_lo, c_hi - c_lo, r_hi - r_lo))
        out[:, :, k, :] = a[[names.index(b) for b in OLMO_BANDS]].transpose(1, 2, 0)
    return out


def attach_blocks(S, roots=None, label_fallback=False, log=print):
    """Each unit's input block from the deployed run's windows (the first holding the unit whose twelve item groups are
    complete), or, for F without roots, from exp89's label windows (oe_inferencex.awf.load_window_full). Sets
    S["units"] (None where nothing could be read) and returns the counts."""
    n = len(S["positions"])
    units = [None] * n
    rec = {"n_units": n}
    if roots:
        windows = deployed_windows(roots)
        cands = locate_units(S["positions"], windows)
        rec.update(source="the deployed run's rslearn dataset", n_roots=len(roots), n_windows=len(windows),
                   n_windows_with_12_groups=int(sum(w["complete"] for w in windows)),
                   in_no_window=int(sum(not c for c in cands)),
                   in_two_or_more_windows=int(sum(len(c) > 1 for c in cands)),
                   only_in_windows_without_12_groups=int(sum(bool(c) and not any(windows[j]["complete"]
                                                                                 for j, _, _ in c) for c in cands)))
        failed = {}
        n_switched = 0
        for i, c in enumerate(cands):
            complete = [(j, r, cc) for j, r, cc in c if windows[j]["complete"]]
            if not complete:
                continue
            # the first complete window where no crop is clamped, else the first complete one (choice c)
            plans = []
            for j, r, cc in complete:
                win = windows[j]
                H, W = win["b"][3] - win["b"][1], win["b"][2] - win["b"][0]
                plans.append((j, crop_plan(H, W, r, cc)))
            pick = next(((j, pl) for j, pl in plans if not pl[2]), plans[0])
            n_switched += pick[0] != plans[0][0]          # a later window than the first complete one was taken
            j, (plan, box, clamped) = pick
            win = windows[j]
            try:
                units[i] = make_unit(read_window_block(win, box), plan, box, clamped)
            except Exception as ex:  # noqa: BLE001  (counted; the floor decides)
                failed[type(ex).__name__] = failed.get(type(ex).__name__, 0) + 1
        rec["failed_reads"] = failed
        rec["took_a_later_window_to_avoid_clamping"] = int(n_switched)
    elif label_fallback:
        from oe_inferencex import awf
        rec.update(source="exp89's label windows (oe_inferencex.awf.load_window_full): the deployed 2023 dataset was "
                          "not given, a deviation the page names")
        failed = {}
        for i, p in enumerate(S["points"]):
            try:
                full = awf.load_window_full(p["dir"])
                r, c = p["rc"]
                plan, box, clamped = crop_plan(full.shape[0], full.shape[1], r, c)
                units[i] = make_unit(full[box[0]:box[2], box[1]:box[3]], plan, box, clamped)
            except Exception as ex:  # noqa: BLE001
                failed[type(ex).__name__] = failed.get(type(ex).__name__, 0) + 1
        rec["failed_reads"] = failed
    else:
        raise ValueError(f"set {S['name']}: no dataset roots to read the views' input from")
    S["units"] = units
    rec["read"] = int(sum(u is not None for u in units))
    rec["clamped_at_a_window_edge"] = int(sum(u is not None and u["clamped"] for u in units))
    log(f"{S['name']}: {rec['read']} of {n} units read, {rec['clamped_at_a_window_edge']} clamped at a window edge")
    return rec


# ----------------------------------------------------------------------------- the models
class StandIn:
    """The smoke's model, standing in for the replica: a deterministic function of its input with the replica's
    structure where the views act. A per-pixel linear map of the month-weighted mean of the series (so removing a
    quarter changes it, by its month indices), averaged over 4-px patches (so the phase moves pixels between patches),
    a table indexed by the patch's place in the crop (so a rotated crop is not the same input), a share of the crop's
    mean (so the context matters), a bias that favours class 0, and nearest upsampling by 4. B01 weighs heavily and is
    read about the planted noise's mean, so noise planted in B01 around a unit makes its class unstable, while a
    constant B01 at that mean leaves it alone."""
    name = "stand-in (smoke only)"

    def __init__(self, seed=SEED):
        rng = np.random.default_rng(seed)
        self.A = rng.normal(0.0, 0.5, (len(OLMO_BANDS), N_OUT))
        self.A[OLMO_BANDS.index("B01")] = rng.normal(0.0, 6.0, N_OUT)
        self.ref = np.zeros(len(OLMO_BANDS))
        self.ref[OLMO_BANDS.index("B01")] = SMOKE_NOISE_MAX / 2 / 10000.0     # B01 is read about its noise's mean
        self.bias = np.zeros(N_OUT)
        self.bias[0] = 2.0
        self.pos = rng.normal(0.0, 0.05, (max(CONTEXTS) // PATCH, max(CONTEXTS) // PATCH, N_OUT))
        self.context = 0.3

    def normalize(self, raw):
        return (np.asarray(raw, dtype=np.float64) / 10000.0).astype(np.float32)

    def pixel_logits(self, x, months):
        x = np.asarray(x, dtype=np.float32)
        m = np.asarray(months, dtype=np.float64)
        wt = 1.0 + 0.5 * np.cos(2 * np.pi * m / 12) + 0.3 * np.sin(2 * np.pi * m / 12)
        wt = (wt / wt.sum(1, keepdims=True)).astype(np.float32)
        pix = (np.matmul(x.swapaxes(-1, -2), wt[:, None, None, :, None])[..., 0].astype(np.float64) - self.ref) @ self.A
        B, S = pix.shape[0], pix.shape[1]
        G = S // PATCH
        patch = pix.reshape(B, G, PATCH, G, PATCH, N_OUT).mean((2, 4)) + self.pos[:G, :G] + self.bias \
            + self.context * pix.mean((1, 2))[:, None, None, :]
        return np.repeat(np.repeat(patch, PATCH, 1), PATCH, 2)

    def info(self):
        return {"model": self.name, "device": "cpu"}


def _same_device(t, dev):
    return t.device.type == dev.type and (dev.index is None or t.device.index == dev.index)


def move_stray_tensors(module, device):
    """Move every tensor a module holds as a plain attribute (neither a parameter nor a buffer, alone or inside a list,
    tuple or dict attribute) to `device`, which model.to(device) leaves behind. Returns the dotted names moved."""
    import torch
    dev = torch.device(device)
    moved = []

    def mv(v, name):
        if isinstance(v, torch.Tensor):
            if not _same_device(v, dev):
                moved.append(name)
                return v.to(dev)
            return v
        if isinstance(v, (list, tuple)) and any(isinstance(x, torch.Tensor) for x in v):
            return type(v)(mv(x, f"{name}[{i}]") for i, x in enumerate(v))
        if isinstance(v, dict) and any(isinstance(x, torch.Tensor) for x in v.values()):
            return {k: mv(x, f"{name}[{k}]") for k, x in v.items()}
        return v
    for mname, mod in module.named_modules():
        for attr, val in list(vars(mod).items()):
            if attr in ("_parameters", "_buffers", "_modules") or attr.startswith("__"):
                continue
            new = mv(val, f"{mname}.{attr}" if mname else attr)
            if new is not val:
                object.__setattr__(mod, attr, new)
    return moved


def devices_off(module, device):
    """Names of the parameters, buffers and plain tensor attributes not on `device` (empty when all are)."""
    import torch
    dev = torch.device(device)
    off = [n for n, p in module.named_parameters() if not _same_device(p, dev)]
    off += [n for n, b in module.named_buffers() if not _same_device(b, dev)]
    for mname, mod in module.named_modules():
        for attr, val in vars(mod).items():
            if attr in ("_parameters", "_buffers", "_modules"):
                continue
            if isinstance(val, torch.Tensor) and not _same_device(val, dev):
                off.append(f"{mname}.{attr}" if mname else attr)
    return off


class ReplicaModel:
    """The record's replica of FT-AWF (exp89 arm A: Ai2's pinned checkpoint loaded strictly, exp21's forward pass), with
    a sample builder for any month list: the series normalized as exp21 normalizes it (Normalizer COMPUTED), one
    timestamp row per step with the step's own month index, the mask exp21 builds (all online-encoder, ignored under
    fast_pass), then exactly exp21.logits_grid's encoder call, pooling, head and bilinear upsampling. fp32, TF32 off."""
    name = "exp89 arm A replica (exp21's forward pass)"

    def __init__(self, ckpt=None, device="cpu", log=print):
        A = e89.ARMS["awf"]
        if ckpt:
            got = e89.sha256(ckpt)
            if got != A["model_sha256"]:
                raise RuntimeError(f"--ckpt {ckpt}: sha256 {got}, pinned {A['model_sha256']}")
            path, sha = ckpt, got
        else:
            path, sha = e89.hub_file(A["model"], A["model_file"], A["model_revision"], "model", A["model_sha256"])
        self.rep = e89.Replica("awf", path, device="cpu", checkpoint_sha256=sha)
        self.device, self.moved = "cpu", []
        self._exp21_device("cpu")         # exp21.DEV defaults to cuda on a GPU node; the model starts on the CPU
        if device != "cpu":
            self.to(device)
        log(f"replica loaded on {self.device}")

    @staticmethod
    def _exp21_device(device):
        """exp21.logits_grid builds its sample on exp21.DEV: keep it on the replica's device."""
        import exp21_finetuned_awf as exp21
        exp21.DEV = str(device)

    def to(self, device):
        """Move the encoder, its stray tensor attributes and the head to `device`; refuse if anything stays behind."""
        import torch
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"device {device} asked for, but torch sees no GPU")
        self.rep.model = self.rep.model.to(device)
        self.moved = sorted(set(self.moved) | set(move_stray_tensors(self.rep.model, device)))
        self.rep.w, self.rep.b = self.rep.w.to(device), self.rep.b.to(device)
        self.rep.device = self.device = str(device)
        self._exp21_device(device)
        off = devices_off(self.rep.model, device)
        if off:
            raise RuntimeError(f"after the move, {len(off)} tensors are not on {device}: {off[:5]}")
        return self.moved

    def normalize(self, raw):
        return self.rep.normalize(raw).astype(np.float32)

    def sample(self, x, months):
        import torch
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device)
        B, S, S2, T, _ = xt.shape
        ts = torch.from_numpy(np.stack([timestamps(m) for m in np.asarray(months)])).to(self.device)
        if ts.shape != (B, T, 3):
            raise ValueError(f"timestamps {tuple(ts.shape)} for a series of {T} steps and {B} views")
        mask = torch.ones((B, S, S2, T, self.rep.n_band_sets), device=self.device) * MaskValue.ONLINE_ENCODER.value
        return MaskedOlmoEarthSample(sentinel2_l2a=xt, sentinel2_l2a_mask=mask, timestamps=ts)

    def pixel_logits(self, x, months):
        """(B, S, S, 10) pixel logits of normalized views x (B, S, S, T, 12) with month indices (B, T)."""
        import torch
        with torch.no_grad():
            out = self.rep.model.encoder(self.sample(x, months), fast_pass=True, patch_size=PATCH)
            feat = out["tokens_and_masks"].sentinel2_l2a.mean(dim=[3, 4])          # (B, H', W', 768)
            patch_logits = (feat @ self.rep.w.T + self.rep.b).float()
            pix = torch.nn.functional.interpolate(patch_logits.permute(0, 3, 1, 2), scale_factor=PATCH,
                                                  mode="bilinear", align_corners=False)
            return pix.permute(0, 2, 3, 1).cpu().numpy()

    def exp21_logits(self, raw_crops):
        """exp21.logits_grid itself on raw twelve-month crops (B, S, S, 12, 12): the record's forward pass, for the
        check of this file's sample builder."""
        import exp21_finetuned_awf as exp21
        from oe_inferencex import awf
        old = awf.CROP
        awf.CROP = int(np.asarray(raw_crops).shape[1])   # the mask size stacks_to_sample uses (as exp89 sets it)
        try:
            _, pix = exp21.logits_grid(self.rep.model, self.rep.w, self.rep.b, list(np.asarray(raw_crops)))
        finally:
            awf.CROP = old
        return pix

    def info(self):
        rec = self.rep.info()
        rec.update(model=self.name, device=self.device, moved_tensor_attributes=self.moved)
        try:
            import torch
            if self.device.startswith("cuda"):
                rec["gpu"] = torch.cuda.get_device_name(torch.device(self.device))
            rec["torch"] = torch.__version__
        except Exception:  # noqa: BLE001
            pass
        return rec


def default_device():
    dev = os.environ.get("E101_DEVICE")
    if dev:
        return dev
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ----------------------------------------------------------------------------- shaking a set
def shake(model, S, batch_tokens=BATCH_TOKENS, log=print):
    """Run every view of every readable unit of S; set the logits, the classes, the distances and the instability."""
    ok = np.array([u is not None for u in S["units"]], dtype=bool)
    L = np.full((len(ok), N_VIEWS, N_OUT), np.nan)
    stats = {"views": 0}
    if ok.any():
        L[ok], stats = run_views(model, [u for u in S["units"] if u is not None], batch_tokens, log=log)
    S["views_ok"] = ok & np.isfinite(L).all((1, 2))
    cls = np.full((len(ok), N_VIEWS), -1, dtype=np.int64)
    tv = np.full((len(ok), N_VIEWS), np.nan)
    if S["views_ok"].any():
        cls[S["views_ok"]], tv[S["views_ok"]] = view_readings(L[S["views_ok"]])
    S["classes"], S["tv"] = cls, tv
    sh, mt = np.full(len(ok), np.nan), np.full(len(ok), np.nan)
    if S["views_ok"].any():
        sh[S["views_ok"]], mt[S["views_ok"]] = instability(cls[S["views_ok"]], tv[S["views_ok"]])
    S["share"], S["mean_tv"] = sh, mt
    log(f"{S['name']}: {int(S['views_ok'].sum())} units with all {N_VIEWS} views; {stats.get('views', 0)} views, "
        f"{stats.get('views_per_second')} per second")
    return stats


def set_counts(S, ok):
    out = {"n_with_views": int(ok.sum()), "n_errors": int(np.nansum(S["wrong"][ok])),
           "n_correct": int(ok.sum() - np.nansum(S["wrong"][ok]))}
    out["v0_agreement_with_deployed"] = float((S["classes"][ok, V0] == S["pred"][ok]).mean()) if ok.any() else None
    cl = np.array([u is not None and u["clamped"] for u in S["units"]], dtype=bool)
    out["clamped_errors"] = int((cl & ok & (np.nan_to_num(S["wrong"]) > 0)).sum())
    out["clamped_correct"] = int((cl & ok & (np.nan_to_num(S["wrong"]) == 0)).sum())
    out["mean_share_errors"] = float(S["share"][ok][S["wrong"][ok] > 0].mean()) if (S["wrong"][ok] > 0).any() else None
    out["mean_share_correct"] = float(S["share"][ok][S["wrong"][ok] == 0].mean()) \
        if (S["wrong"][ok] == 0).any() else None
    return out


def fit_record(F, n_boot=0):
    """F's values: each q's 10% review (the q choice), F's statistics at the chosen q, each kind alone and the dose."""
    ok = F["views_ok"]
    u, sh, mt, err = F["u"][ok], F["share"][ok], F["mean_tv"][ok], F["wrong"][ok]
    w = np.ones(int(ok.sum()))
    q, rows = choose_q(u, sh, mt, err, w)
    per_q = {}
    for qq in Q_GRID:
        per_q[str(qq)] = dict(rows[qq], statistics=ranking_record(u, sh, mt, err, w, None, qq, 0))
    rec = {"units": set_counts(F, ok), "q": q,
           "q_rule": f"the q of {list(Q_GRID)} whose {BUDGET:.0%} review by R(q) holds the most of F's errors (the "
                     "record's tie-aware capture times the errors), ties to the larger q",
           "per_q": per_q, "at_q": ranking_record(u, sh, mt, err, w, None, q, n_boot),
           "kinds_reported": kinds_record(F["classes"][ok], F["tv"][ok], u, err, w, None, q, 0),
           "dose_reported": dose_record(F["classes"][ok], F["tv"][ok], u, err, w, q)}
    return rec, q


def test_record(E, q, n_boot=BOOT, floors=FLOORS):
    """E's values at F's q: STRICT (graded), LENIENT, each kind, the dose, the grass/shrub plots and the budgets."""
    ok = E["views_ok"]
    ctry = np.asarray(E["strata"])[ok]
    n_by = {c: int((ctry == c).sum()) for c in tsx.COUNTRIES}
    W, w_plot = tsx.weights_from_areas(E["areas"]["by_country_km2"], n_by)
    w = np.array([w_plot[c] for c in ctry], dtype=np.float64)
    u, sh, mt = E["u"][ok], E["share"][ok], E["mean_tv"][ok]
    strict, lenient = E["wrong"][ok], E["wrong_lenient"][ok]
    E["weight_views"] = np.zeros(len(ok))
    E["weight_views"][ok] = w
    rec = {"units": dict(set_counts(E, ok), by_country=n_by, lenient_errors=int(lenient.sum())),
           "weights": {"by_country": W, "per_plot": w_plot,
                       "rule": "W_c, the country's share of the region's area, over n_c, its plots with all views"},
           "confidence_alone": {"capture_0.1_strict_all_with_input": E.get("confidence_capture_0_1_all"),
                                "capture_0.1_strict_with_views": capture(u, strict, w, (BUDGET,))[BUDGET],
                                "recorded_exp99": RECORDED["E"]["confidence_capture_0.1_strict"]},
           "q": q}
    rec["strict"] = ranking_record(u, sh, mt, strict, w, ctry, q, n_boot)
    rec["lenient_reported"] = ranking_record(u, sh, mt, lenient, w, ctry, q, n_boot)
    rec["kinds_reported"] = kinds_record(E["classes"][ok], E["tv"][ok], u, strict, w, ctry, q, n_boot)
    rec["dose_reported"] = dose_record(E["classes"][ok], E["tv"][ok], u, strict, w, q)
    rec["grass_shrub_reported"] = grass_shrub_record(u, sh, mt, strict, lenient, w, q)
    prereg = grade(rec["strict"], rec["units"]["n_with_views"], rec["units"]["n_errors"], rec["units"]["n_correct"],
                   floors)
    return rec, prereg


def design_record():
    return {"views": {"orientation": [f"{k} quarter turn(s){', flipped' if f else ''}" for k, f in ORIENTATIONS],
                      "time": list(TIME_NAMES), "context_px": list(CONTEXTS), "phase_px": list(PHASES),
                      "n_views": N_VIEWS, "index": "((orientation x 5 + time) x 2 + context) x 4 + phase",
                      "v0": "view 0: identity, all twelve months, 16 px, phase 0, the unit at (8, 8)"},
            "instability": "the share of the other 319 views whose class (argmax over ten channels) differs from V0's; "
                           "ties broken by the mean total-variation distance of the ten-channel softmax to V0's",
            "rule": "R(q): the most confident share q of the units (in design weight) set aside; the others by "
                    "instability, most unstable first, ties by confidence, least confident first; then the units set "
                    "aside, least confident first",
            "q_grid": list(Q_GRID), "budget": BUDGET, "budgets_reported": list(BUDGETS), "p1_margin": P1_MARGIN,
            "level": LEVEL, "kinds": {k: len(v) for k, v in KINDS.items()}, "dose_k": list(DOSE_K),
            "dose_draws": DOSE_DRAWS, "seed": SEED}


# ----------------------------------------------------------------------------- the npz
def units_arrays(sets, q=None):
    """Per-unit arrays of one or more shaken sets, concatenated; no position, no identifier."""
    cols = {k: [] for k in ("set", "label", "pred_deployed", "wrong", "wrong_lenient", "confidence", "suspicion",
                            "weight", "country", "views_read", "clamped", "v0_class", "view_class", "view_tv",
                            "instability_share", "instability_mean_tv", "set_aside_at_q", "two_step_rank")}
    for si, S in enumerate(sets):
        n = len(S["units"])
        ok = S["views_ok"]
        w = S.get("weight_views", np.where(ok, S["weight"], 0.0))
        aside = np.zeros(n, dtype=bool)
        rank = np.full(n, np.nan)
        if q is not None and ok.any():
            sc, a = two_step_score(S["u"][ok], S["share"][ok], S["mean_tv"][ok], q, w[ok])
            aside[ok], rank[ok] = a, sc
        cols["set"].append(np.full(n, 0 if S["name"] == "F" else 1, dtype=np.int8))
        cols["label"].append(S["label"].astype(np.int16))
        cols["pred_deployed"].append(S["pred"].astype(np.int16))
        cols["wrong"].append(S["wrong"])
        cols["wrong_lenient"].append(S["wrong_lenient"])
        cols["confidence"].append(S["p1"])
        cols["suspicion"].append(S["u"])
        cols["weight"].append(w)
        cols["country"].append(S["country"].astype(np.int8))
        cols["views_read"].append(ok)
        cols["clamped"].append(np.array([u is not None and u["clamped"] for u in S["units"]], dtype=bool))
        cols["v0_class"].append(S["classes"][:, V0].astype(np.int16))
        cols["view_class"].append(S["classes"].astype(np.int8))
        cols["view_tv"].append(S["tv"].astype(np.float32))
        cols["instability_share"].append(S["share"])
        cols["instability_mean_tv"].append(S["mean_tv"])
        cols["set_aside_at_q"].append(aside)
        cols["two_step_rank"].append(rank)
    out = {k: np.concatenate(v) for k, v in cols.items()}
    o, t, c, p = np.unravel_index(np.arange(N_VIEWS), SHAPE)
    out.update(set_names=np.array(["F", "E"]), country_names=np.array(list(tsx.COUNTRIES)),
               label_names_F=np.array([e98.CLASS_NAMES.get(i, str(i)) for i in range(N_OUT)]),
               label_names_E=np.array(tsx.TIMESYNC_CLASSES), view_orientation=o.astype(np.int8),
               view_time=t.astype(np.int8), view_context_px=np.array(CONTEXTS)[c].astype(np.int16),
               view_phase_px=np.array(PHASES)[p].astype(np.int8), q=np.array(np.nan if q is None else q))
    return out


def write_npz(arrays, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)


# ----------------------------------------------------------------------------- the modes
def header(mode):
    commit, dirty = e89.git_state()
    return {"experiment": "exp101: filter, then shake. A two-step error ranking (set aside the most confident share q, "
                          "rank the rest by the replica's instability over 320 perturbed views) fitted on exp98's "
                          "graded validation points and tested on exp99's plots",
            "prereg_page": os.path.relpath(PLAN, ROOT), "mode": mode, "utc": e89.utc_now(), "commit": commit,
            "dirty": dirty, "prereg_status": prereg_status(), "design": design_record(), "cannot_show": list(CANNOT)}


def inventory(a, log=print, kw99=None):
    """Counts only: the units of each set and how many of them have a readable block with twelve item groups."""
    t0 = time.time()
    kw99 = kw99 or {}
    rec = header("inventory: counts only; no class, probability or perturbed reading at a unit")
    F = load_fit_units(a.exp98_scores, a.labels, outcome=False)
    rec["F"] = dict(attach_blocks(F, a.deploy_2023, label_fallback=a.f_fallback, log=log), graded_validation_points=F["n"])
    E = load_test_units(a.exp99_scores, a.timesync, a.request_geometry, a.boundaries, outcome=False, **kw99)
    rec["E"] = dict(attach_blocks(E, a.deploy_2017, log=log), selected=E["n_selected"], with_input=E["n"],
                    with_input_by_country=E["by_country"])
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def fit(a, model, log=print, recorded=RECORDED, n_boot=0):
    """F only: load, read, shake, choose q. Returns (record, F)."""
    t0 = time.time()
    rec = header("fit: F only (allowed before freezing)")
    F = load_fit_units(a.exp98_scores, a.labels, recorded=None if recorded is None else recorded["F"])
    rec["inputs_F"] = attach_blocks(F, a.deploy_2023, label_fallback=a.f_fallback, log=log)
    rec["inputs_F"]["reproduces_record"] = F["reproduces_record"]
    rec["compute_F"] = shake(model, F, a.batch_tokens, log)
    rec["model"] = model.info()
    rec["fit"], q = fit_record(F, n_boot)
    log(f"F: q = {q}")
    rec["seconds"] = round(time.time() - t0, 1)
    return rec, F


def run_test(a, model, log=print, recorded=RECORDED, kw99=None, n_boot=BOOT, floors=FLOORS, fit_rec=None):
    """The test: F (q) and E; returns (record, F, E). Refuses when q differs from the fit record's."""
    t0 = time.time()
    kw99 = kw99 or {}
    rec, F = fit(a, model, log, recorded)
    rec.update(header("the test (E), with F recomputed for q"))
    q = rec["fit"]["q"]
    if fit_rec is not None:
        if fit_rec.get("fit", {}).get("q") != q:
            raise ValueError(f"q from F is {q} here and {fit_rec.get('fit', {}).get('q')} in the fit record: the views "
                             "of F did not reproduce; nothing on E is graded")
        rec["fit_record_check"] = {"q_matches": True, "fit_record_commit": fit_rec.get("commit")}
    else:
        rec["fit_record_check"] = {"q_matches": None, "note": "no fit record given; q is F's in this run"}
    E = load_test_units(a.exp99_scores, a.timesync, a.request_geometry, a.boundaries,
                        recorded=None if recorded is None else recorded["E"], **kw99)
    rec["inputs_E"] = attach_blocks(E, a.deploy_2017, log=log)
    rec["inputs_E"]["reproduces_record"] = E["reproduces_record"]
    rec["compute_E"] = shake(model, E, a.batch_tokens, log)
    rec["test"], rec["prereg"] = test_record(E, q, n_boot, floors)
    rec["reading"] = reading(rec["prereg"])
    rec["replica_vs_deployed"] = {
        "F": rec["fit"]["units"]["v0_agreement_with_deployed"],
        "E": rec["test"]["units"]["v0_agreement_with_deployed"],
        "exp98": "exp98 found the replica's class equal to the deployed map's at 91.9% of F; a much lower share means "
                 "the views describe the replica, not the deployed map (reported with the verdicts)"}
    rec["seconds"] = round(time.time() - t0, 1)
    return rec, F, E


def gpu_smoke(a, log=print):
    """V0 on the GPU against V0 on the CPU at F's first units, the CPU's V0 against exp21.logits_grid, and the
    throughput of all 320 views on the GPU."""
    import torch
    t0 = time.time()
    if not str(a.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(f"--gpu-smoke needs a GPU; device {a.device}, cuda available {torch.cuda.is_available()}")
    rec = header("gpu smoke: F's first units (allowed before freezing)")
    F = load_fit_units(a.exp98_scores, a.labels, outcome=True, recorded=RECORDED["F"])
    rec["inputs_F"] = attach_blocks(F, a.deploy_2023, label_fallback=a.f_fallback, log=log)
    idx = [i for i, u in enumerate(F["units"]) if u is not None][:a.units]
    units = [F["units"][i] for i in idx]
    model = ReplicaModel(a.ckpt, device="cpu", log=log)
    # V0 and one view of each other shape (a 9-step series, a 32-px crop, and all four kinds at once)
    checked = [V0, view_index(0, 1, 0, 0), view_index(0, 0, 1, 0), view_index(5, 3, 1, 2)]
    L_cpu, st_cpu = run_views(model, units, a.batch_tokens, views=checked, log=log)
    raw = np.stack([u["raw"][u["plan"][(0, 0)][0]:u["plan"][(0, 0)][0] + CONTEXTS[0],
                             u["plan"][(0, 0)][1]:u["plan"][(0, 0)][1] + CONTEXTS[0]] for u in units])
    pix = model.exp21_logits(raw)
    L21 = np.stack([pix[j][u["plan"][(0, 0)][2], u["plan"][(0, 0)][3]] for j, u in enumerate(units)])
    d21 = float(np.abs(L_cpu[:, V0] - L21).max())
    moved = model.to(a.device)
    L_gpu, st_gpu = run_views(model, units, a.batch_tokens, views=checked, log=log)
    d = float(max(np.abs(L_gpu[:, v] - L_cpu[:, v]).max() for v in checked))
    same = int(sum((L_gpu[:, v].argmax(1) == L_cpu[:, v].argmax(1)).sum() for v in checked))
    log(f"gpu smoke: max |GPU - CPU| logit over {len(checked)} view shapes {d:.2e}; same class {same} of "
        f"{len(units) * len(checked)}; builder vs exp21.logits_grid {d21:.2e}")
    torch.cuda.synchronize()
    _, st_all = run_views(model, units, a.batch_tokens, log=log)
    rate = st_all.get("views_per_second") or 0.0
    n_fit, n_test = RECORDED["F"]["n"] * N_VIEWS, (RECORDED["F"]["n"] + RECORDED["E"]["n"]) * N_VIEWS
    passed = bool(d <= GPU_TOL and same == len(units) * len(checked) and d21 <= REPLICA_TOL)
    rec.update(model=model.info(), units=len(units),
               v0={"views_checked": checked, "max_abs_logit_difference_gpu_cpu": d, "same_class": same,
                   "compared": len(units) * len(checked), "tolerance": GPU_TOL,
                   "cpu_seconds": st_cpu["seconds"], "gpu_seconds": st_gpu["seconds"]},
               replica_check={"max_abs_logit_difference_this_builder_vs_exp21_logits_grid_cpu": d21,
                              "tolerance": REPLICA_TOL},
               v0_agreement_with_deployed=float((L_cpu[:, V0].argmax(1) == F["pred"][idx]).mean()),
               moved_tensor_attributes=moved,
               throughput={"views": st_all["views"], "seconds": st_all["seconds"], "views_per_second": rate,
                           "batch_tokens": a.batch_tokens,
                           "estimated_minutes_fit": round(n_fit / rate / 60, 1) if rate else None,
                           "estimated_minutes_test": round(n_test / rate / 60, 1) if rate else None},
               passed=passed, seconds=round(time.time() - t0, 1))
    return rec


# ----------------------------------------------------------------------------- the smoke
SMOKE_NOISE_HALF = 20            # planted B01 noise: a 40 x 40-px square around the unit, the same in every item group
SMOKE_NOISE_MAX = 20000
SMOKE_WINDOW = 128               # the synthetic 2017 dataset's windows: 128-px cells on the CRS-origin lattice
SMOKE_EPSG = 32737
# The planted sets (in memory, no files): units ordered from the least confident; three confidence bands, each with
# its share of the units, the period of its errors (every k-th unit) and whether its errors are unstable (B01 noise).
# The middle band's errors are unstable, with every SMOKE_UNSTABLE_CORRECT-th correct unit there: confidence ranks them
# poorly and the shaking finds them, so P1 and P2 should hold. The null keeps every block and view and moves the middle
# band's errors to other units at random: the shaking then carries nothing about the errors.
SMOKE_PLANTED_N = {"F": 80, "E": 120}
SMOKE_BANDS = ((0.10, 6, False), (0.40, 3, True), (0.50, 20, False))
SMOKE_UNSTABLE_CORRECT = 10
SMOKE_AREAS = {"kenya": 56.0, "tanzania": 44.0}


def mark_completed(roots):
    """rslearn's completed marker on every item group that holds a band set of the twelve bands."""
    for root in roots:
        for wdir in glob.glob(os.path.join(root, "windows", "*", "*")):
            for k in range(N_GROUPS):
                if band_set_dir(wdir, k):
                    open(os.path.join(wdir, "layers", layer_dir(k), "completed"), "w").close()


def plant_noise(roots, positions, chosen, seed=SEED):
    """Noise in B01 around each chosen unit, in the window attach_blocks reads it from (the same noise in every item
    group): the stand-in model's class there becomes unstable."""
    import rasterio
    from rasterio.windows import Window
    windows = deployed_windows(roots)
    cands = locate_units([positions[i] for i in chosen], windows)
    rng = np.random.default_rng(seed)
    for c in cands:
        pick = next(((j, r, cc) for j, r, cc in c if windows[j]["complete"]), None)
        if pick is None:
            continue
        j, r, cc = pick
        win = windows[j]
        H, W = win["b"][3] - win["b"][1], win["b"][2] - win["b"][0]
        r0, r1 = max(r - SMOKE_NOISE_HALF, 0), min(r + SMOKE_NOISE_HALF, H)
        c0, c1 = max(cc - SMOKE_NOISE_HALF, 0), min(cc + SMOKE_NOISE_HALF, W)
        noise = rng.integers(0, SMOKE_NOISE_MAX, size=(r1 - r0, c1 - c0)).astype(np.uint16)
        for k in range(N_GROUPS):
            d = band_set_dir(win["dir"], k)
            band = os.path.basename(d).split("_").index("B01") + 1
            with rasterio.open(os.path.join(d, "geotiff.tif"), "r+") as dst:
                dst.write(noise, band, window=Window(c0, r0, c1 - c0, r1 - r0))


def make_smoke_2017(root, positions, incomplete=(), duplicate=(), seed=SEED):
    """A synthetic 2017 rslearn dataset: one 128-px window on the CRS-origin lattice for each cell holding a unit, twelve
    item groups of the twelve bands (random in the 10 m bands, zero elsewhere). The cells of `incomplete` units lack
    their last item group; for each `duplicate` unit an incomplete window on its cell is listed first (an earlier
    group), so the reader passes over it."""
    import pyproj
    import rasterio
    from rasterio.transform import from_origin
    bands12 = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{SMOKE_EPSG}", always_xy=True)
    rng = np.random.default_rng(seed)
    cells = {}
    for i, p in enumerate(positions):
        X, Y = to_utm.transform(p["X"], p["Y"])
        px, py = int(np.floor(X / 10.0)), int(np.floor(Y / -10.0))
        key = (px // SMOKE_WINDOW * SMOKE_WINDOW, py // SMOKE_WINDOW * SMOKE_WINDOW)
        cells.setdefault(key, set()).add(i)

    def write(group, name, cx, cy, n_groups):
        wdir = os.path.join(root, "windows", group, name)
        os.makedirs(wdir, exist_ok=True)
        with open(os.path.join(wdir, "metadata.json"), "w") as f:
            json.dump({"projection": {"crs": f"EPSG:{SMOKE_EPSG}", "x_resolution": 10, "y_resolution": -10},
                       "bounds": [cx, cy, cx + SMOKE_WINDOW, cy + SMOKE_WINDOW],
                       "time_range": ["2017-01-01T00:00:00+00:00", "2017-12-31T00:00:00+00:00"]}, f)
        for k in range(n_groups):
            d = os.path.join(wdir, "layers", layer_dir(k), "_".join(bands12))
            os.makedirs(d, exist_ok=True)
            a = np.zeros((12, SMOKE_WINDOW, SMOKE_WINDOW), np.uint16)
            for b in ("B02", "B03", "B04", "B08"):
                a[bands12.index(b)] = rng.integers(500, 3000, size=(SMOKE_WINDOW, SMOKE_WINDOW))
            with rasterio.open(os.path.join(d, "geotiff.tif"), "w", driver="GTiff", height=SMOKE_WINDOW,
                               width=SMOKE_WINDOW, count=12, dtype="uint16", crs=f"EPSG:{SMOKE_EPSG}",
                               transform=from_origin(cx * 10.0, cy * -10.0, 10, 10)) as dst:
                dst.write(a)
            open(os.path.join(wdir, "layers", layer_dir(k), "completed"), "w").close()
    for n, ((cx, cy), members) in enumerate(sorted(cells.items())):
        full = not (members & set(incomplete))
        write("default", f"w{n:03d}", cx, cy, N_GROUPS if full else N_GROUPS - 1)
        if members & set(duplicate):
            write("a_partial", f"w{n:03d}", cx, cy, N_GROUPS - 1)
    return [root]


def planted_set(name, n, null=False, seed=SEED):
    """A synthetic set in memory (SMOKE_BANDS): blocks of random 10 m bands, B01 noise in the unstable units' blocks,
    the confidence falling with the unit's index, errors by band. E is split into two countries weighted by
    SMOKE_AREAS, and every other error of E is correct under LENIENT (the grass/shrub plots). With `null`, the middle
    band's errors are moved to units drawn at random in that band; the blocks are the same."""
    rng = np.random.default_rng(seed + (0 if name == "F" else 1))
    u = np.linspace(1.2, 0.01, n)                             # suspicion u = -log p1: unit 0 the least confident
    wrong, unstable = np.zeros(n), np.zeros(n, dtype=bool)
    lo = 0
    for bi, (share, period, noisy) in enumerate(SMOKE_BANDS):
        hi = n if bi == len(SMOKE_BANDS) - 1 else lo + int(round(share * n))
        idx = np.arange(lo, hi)
        err = idx[(idx - lo) % period == 0]
        wrong[err] = 1.0
        if noisy:
            unstable[err] = True
            ok = idx[wrong[idx] == 0]
            unstable[ok[::SMOKE_UNSTABLE_CORRECT]] = True
            if null:
                wrong[idx] = 0.0
                wrong[rng.choice(idx, err.size, replace=False)] = 1.0
        lo = hi
    plan, box, clamped = crop_plan(1024, 1024, 512, 512)
    units = []
    brng = np.random.default_rng(seed + 7 + (0 if name == "F" else 1))       # the same blocks with and without null
    for i in range(n):
        raw = np.zeros((box[2] - box[0], box[3] - box[1], N_GROUPS, len(OLMO_BANDS)), dtype=np.float32)
        for b in ("B02", "B03", "B04", "B08"):
            raw[..., OLMO_BANDS.index(b)] = brng.integers(500, 3000, size=raw.shape[:3])
        noise = brng.integers(0, SMOKE_NOISE_MAX, size=raw.shape[:2])
        raw[..., OLMO_BANDS.index("B01")] = noise[:, :, None] if unstable[i] else SMOKE_NOISE_MAX / 2
        units.append(make_unit(raw, plan, box, clamped))
    S = {"name": name, "units": units, "u": u, "p1": np.exp(-u), "pred": np.zeros(n, dtype=np.int64),
         "label": wrong.astype(np.int64), "wrong": wrong, "wrong_lenient": np.full(n, np.nan), "weight": np.ones(n),
         "strata": None, "country": np.full(n, -1, dtype=np.int64), "reproduces_record": False, "positions": None,
         "unstable": unstable}
    if name == "E":
        ctry = np.array([list(SMOKE_AREAS)[i % 2] for i in range(n)])
        _, w_plot = tsx.weights_from_areas(SMOKE_AREAS, {c: int((ctry == c).sum()) for c in SMOKE_AREAS})
        len_err = np.flatnonzero(wrong > 0)
        lenient = wrong.copy()
        lenient[len_err[::2]] = 0.0
        S.update(strata=ctry, country=np.array([i % 2 for i in range(n)], dtype=np.int64),
                 weight=np.array([w_plot[c] for c in ctry]), wrong_lenient=lenient,
                 areas={"by_country_km2": dict(SMOKE_AREAS)})
    return S


def planted_smoke(model, n_boot=SMOKE_BOOT, log=print):
    """The planted sets: q fitted on the planted F, then E planted and E null graded at that q (floors lowered)."""
    F = planted_set("F", SMOKE_PLANTED_N["F"])
    E = planted_set("E", SMOKE_PLANTED_N["E"])
    shake(model, F, log=log)
    shake(model, E, log=log)
    fit_rec, q = fit_record(F)
    out = {"design": {"n": dict(SMOKE_PLANTED_N), "bands": [list(b) for b in SMOKE_BANDS],
                      "unstable_correct_period": SMOKE_UNSTABLE_CORRECT},
           "fit": {k: fit_rec[k] for k in ("units", "q", "per_q")}, "q": q}
    for tag, null in (("planted", False), ("null", True)):
        Eq = dict(E)
        if null:
            Z = planted_set("E", SMOKE_PLANTED_N["E"], null=True)
            Eq.update(wrong=Z["wrong"], wrong_lenient=Z["wrong_lenient"], label=Z["label"])
        rec, prereg = test_record(Eq, q, n_boot, SMOKE_FLOORS)
        out[tag] = {"units": rec["units"], "prereg": prereg, "reading": reading(prereg),
                    "auroc_instability_all_units": metrics.weighted_auroc(
                        instability_score(Eq["share"], Eq["mean_tv"]), Eq["wrong"], Eq["weight"])}
    return out, (F, E)


SMOKE_KW99 = dict(fixed=None, geometry_sha256=None, min_plots=e99.SMOKE_MIN_PLOTS)   # exp99's select on its smoke


def make_smoke_inputs(root):
    """The file path's synthetic inputs under root: exp98's and exp99's smoke maps and labels, exp98's smoke deployment
    dataset for F (completed markers added) and a synthetic 2017 dataset for E (one plot's window lacking an item
    group, one plot behind an earlier incomplete window), with B01 noise planted around every error but the least
    confident one and around two correct units of each set. Returns (the run's arguments, the planted counts)."""
    for d in ("e98", "e99", "dep17"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    p98 = e98.make_smoke_inputs(os.path.join(root, "e98"))
    p99 = e99.make_smoke_inputs(os.path.join(root, "e99"))
    dep23 = p98["deployment"]
    mark_completed(dep23)
    F0 = load_fit_units(p98["scores"], p98["labels"])
    E0 = load_test_units(p99["scores"], p99["timesync"], p99["request_geometry"], p99["boundaries"], **SMOKE_KW99)
    rng = np.random.default_rng(SEED)
    planted = {}
    for S in (F0, E0):
        err = np.flatnonzero(S["wrong"] > 0)
        ok = np.flatnonzero(S["wrong"] == 0)
        least = err[np.argmax(S["u"][err])]                     # the least confident error: confidence finds it
        planted[S["name"]] = sorted(set(err.tolist()) - {int(least)} | set(rng.choice(ok, 2, replace=False).tolist()))
    correct_E = [i for i in np.flatnonzero(E0["wrong"] == 0) if i not in planted["E"]]
    incomplete, duplicate = [int(correct_E[0])], [int(planted["E"][0])]
    dep17 = make_smoke_2017(os.path.join(root, "dep17"), E0["positions"], incomplete, duplicate)
    plant_noise(dep23, F0["positions"], planted["F"])
    plant_noise(dep17, E0["positions"], planted["E"])
    a = argparse.Namespace(exp98_scores=p98["scores"], labels=p98["labels"], deploy_2023=dep23,
                           exp99_scores=p99["scores"], timesync=p99["timesync"],
                           request_geometry=p99["request_geometry"], boundaries=p99["boundaries"],
                           deploy_2017=dep17, batch_tokens=BATCH_TOKENS, f_fallback=False)
    counts = {"planted_units": {k: len(v) for k, v in planted.items()},
              "planted_errors": {S["name"]: int(S["wrong"][planted[S["name"]]].sum()) for S in (F0, E0)},
              "incomplete_window_units": len(incomplete), "earlier_incomplete_duplicate_units": len(duplicate)}
    return a, counts


def smoke(out_dir=None, n_boot=SMOKE_BOOT, log=print):
    """Synthetic inputs end to end with the stand-in model. The file path (make_smoke_inputs): the inventory, the fit of
    q, the test with its weights, bootstraps and grades (floors lowered). Then the planted sets (planted_smoke), large
    enough for P1 and P2 to be graded."""
    import tempfile
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        a, counts = make_smoke_inputs(tmp)
        inv = inventory(a, log, SMOKE_KW99)
        model = StandIn()
        fit_rec, _ = fit(a, model, log, recorded=None)
        rec, F, E = run_test(a, model, log, recorded=None, kw99=SMOKE_KW99, n_boot=n_boot, floors=SMOKE_FLOORS,
                             fit_rec=fit_rec)
    rec["mode"] = "smoke: synthetic inputs and a stand-in model; recorded counts not checked; floors lowered to " \
        + json.dumps(SMOKE_FLOORS)
    rec["smoke"] = counts
    rec["inventory"] = {k: inv[k] for k in ("F", "E", "mode")}
    rec["planted_smoke"], _ = planted_smoke(model, n_boot, log)
    rec["seconds"] = round(time.time() - t0, 1)
    fit_rec["mode"] = "smoke: " + fit_rec["mode"]
    out_dir = out_dir or OUT
    os.makedirs(out_dir, exist_ok=True)
    e98.dump(fit_rec, os.path.join(out_dir, "exp101_fit_smoke.json"))
    e98.dump(rec, os.path.join(out_dir, "exp101_summary_smoke.json"))
    write_npz(units_arrays([F, E], rec["fit"]["q"]), os.path.join(out_dir, "exp101_units_smoke.npz"))
    return rec


# ----------------------------------------------------------------------------- main
MODES = ("smoke", "inventory", "gpu_smoke", "fit", "test")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="synthetic inputs and a stand-in model, end to end")
    g.add_argument("--inventory", action="store_true", help="counts only; allowed before the page is frozen")
    g.add_argument("--gpu-smoke", action="store_true", help="V0 on the GPU against the CPU at F's first units, and "
                   "the throughput; allowed before the page is frozen")
    g.add_argument("--fit", action="store_true", help="F only: views, instability, q; allowed before freezing")
    g.add_argument("--test", action="store_true", help="the test on E; refused until the page is frozen")
    ap.add_argument("--exp98-scores", nargs="+", help="exp98's from-olmoearth part directories (the 2023 map)")
    ap.add_argument("--labels", help="the AWF windows root (holding spatial_split/), exp98's --labels")
    ap.add_argument("--deploy-2023", nargs="*", default=None, help="the 2023 run's rslearn dataset roots")
    ap.add_argument("--f-fallback", action="store_true", help="without --deploy-2023, let F read exp89's label windows "
                    "(a deviation the page names); without this flag a missing 2023 dataset is refused")
    ap.add_argument("--exp99-scores", nargs="+", help="exp99's from-olmoearth part directories (the 2017 map)")
    ap.add_argument("--timesync", default=os.path.join(ROOT, "data", "breadth", tsx.TIMESYNC["file"]))
    ap.add_argument("--request-geometry", help="Ai2's AWF prediction_request_geometry.geojson (its sha256 is checked)")
    ap.add_argument("--boundaries", default=None, help="Natural Earth's admin-0 countries (GeoJSON), for E's weights")
    ap.add_argument("--deploy-2017", nargs="+", default=None, help="the 2017 run's rslearn dataset roots")
    ap.add_argument("--fit-record", default=None, help="the fit's exp101_fit.json; the test refuses another q "
                    "(default: <out-dir>/exp101_fit.json when it exists)")
    ap.add_argument("--ckpt", default=None, help="a local copy of the pinned checkpoint (checked against its sha256)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default E101_DEVICE, else cuda when available)")
    ap.add_argument("--batch-tokens", type=int, default=int(os.environ.get("E101_BATCH_TOKENS", BATCH_TOKENS)))
    ap.add_argument("--units", type=int, default=GPU_SMOKE_UNITS, help="--gpu-smoke: how many of F's units")
    ap.add_argument("--out-dir", default=OUT)
    a = ap.parse_args(argv)
    if a.smoke:
        rec = smoke(a.out_dir)

        def short(pr):
            return {p: {k: pr[p].get(k) for k in ("value", "interval", "graded", "holds")} for p in ("P1", "P2")}
        ps = rec["planted_smoke"]
        print(json.dumps({"files": {"q": rec["fit"]["q"], "prereg": short(rec["prereg"])},
                          "planted": {"q": ps["q"], "prereg": short(ps["planted"]["prereg"])},
                          "null": {"q": ps["q"], "prereg": short(ps["null"]["prereg"])}}, indent=1))
        return 0
    if a.test:
        status = prereg_status()
        if status != "frozen":
            print(f"refused: {os.path.relpath(PLAN, ROOT)} is {status}, not frozen; only --smoke, --inventory, "
                  "--gpu-smoke and --fit run before it is frozen")
            return 3
    if not (a.exp98_scores and a.labels):
        ap.error("--exp98-scores and --labels are required outside --smoke")
    if (a.inventory or a.test) and not (a.exp99_scores and a.request_geometry and a.deploy_2017):
        ap.error("--exp99-scores, --request-geometry and --deploy-2017 are required by --inventory and --test")
    if a.test and not a.boundaries:
        ap.error("--test needs --boundaries (each country's share of the region weights the strata)")
    os.makedirs(a.out_dir, exist_ok=True)
    if a.inventory:
        rec = inventory(a)
        e98.dump(rec, os.path.join(a.out_dir, "exp101_inventory.json"))
        print(json.dumps({k: {kk: rec[k][kk] for kk in ("n_units", "read", "clamped_at_a_window_edge")}
                          for k in ("F", "E")}))
        return 0
    a.device = a.device or default_device()
    if a.gpu_smoke:
        rec = gpu_smoke(a)
        e98.dump(rec, os.path.join(a.out_dir, "exp101_gpusmoke.json"))
        print(json.dumps({k: rec[k] for k in ("v0", "replica_check", "throughput", "passed")}, indent=1))
        return 0 if rec["passed"] else 4
    model = ReplicaModel(a.ckpt, device=a.device)
    if a.fit:
        rec, F = fit(a, model)
        e98.dump(rec, os.path.join(a.out_dir, "exp101_fit.json"))
        write_npz(units_arrays([F], rec["fit"]["q"]), os.path.join(a.out_dir, "exp101_units_fit.npz"))
        print(json.dumps({"q": rec["fit"]["q"], "per_q": {k: v["errors_held"] for k, v in rec["fit"]["per_q"].items()},
                          "units": rec["fit"]["units"]}, indent=1))
        return 0
    path = a.fit_record or os.path.join(a.out_dir, "exp101_fit.json")
    fit_rec = None
    if os.path.exists(path):
        with open(path) as f:
            fit_rec = json.load(f)
    rec, F, E = run_test(a, model, fit_rec=fit_rec)
    e98.dump(rec, os.path.join(a.out_dir, "exp101_summary.json"))
    write_npz(units_arrays([F, E], rec["fit"]["q"]), os.path.join(a.out_dir, "exp101_units.npz"))
    print(json.dumps({"q": rec["fit"]["q"], "prereg": rec["prereg"], "reading": rec["reading"]}, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001  (a job log needs the traceback, and the exit code must fail the job)
        traceback.print_exc()
        sys.exit(1)
