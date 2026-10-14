"""exp88: when a modality is missing at inference, does the confidence fall with the accuracy?

Preregistered in docs/plan/missing_modality.md; read that page first, this file is the run. A probe trained on the
S1+S2 embeddings of Ai2's train split is read, with no refit, on the S1+S2, the S1-only and the S2-only embeddings of
the same test windows: full input, optical missing (clouds), radar missing. Three task families carry the same units
under the three inputs: PASTIS (primary), CropHarvest China 6 (replication), CropHarvest Togo 12 (reported, not
graded). OlmoEarth Base; OlmoEarth Large as a replication when its files are cached.

    python exp/exp88_missing_modality.py --smoke             # numpy only: synthetic embeddings, both synthetic cases
    python exp/exp88_missing_modality.py --smoke-torch       # the real probe code on synthetic tensors (needs torch)
    python exp/exp88_missing_modality.py --check-alignment   # identifiers, shapes and split sizes of the real files only
    python exp/exp88_missing_modality.py                     # the full run; refused until the plan page says frozen

What is reused, not rewritten. The probes are the record's: exp54.train_probe (segmentation, fp32, 50 epochs, the
task's probe learning rate from exp54.TASK_LR, 0.01 for PASTIS S1+S2 on Base; Ai2's per-size file for Large through
suite_regression.set_probe_lrs) and exp73._train_cls_probe (classification, Ai2's linear recipe at 0.1, as exp70
fitted every classification task). Window probabilities, decisions, majority labels and validity are
exp75.seg_window_probs (exp54's windows, returned in full); the embedding under each window is
exp74.window_embeddings; the readings, the two no-model controls and their scores are suite_regression.score_units
(exp70.readings_from_probs, exp73.controls_chunked, exp70.score_task, exp70.best_control), so "the margin beats the
best no-model control" is computed exactly as the record computes it. The intervals are oe_inferencex.estimate's.
The loaders are suite_regression.load_task and exp54.load_theirs, the per-unit export format is exp78's in spirit.

Measures, per family and condition, on the same valid units in the same order:
  1. the error rate;
  2. the median margin of the errors and of the correct units, and their shift from full input (also the median of
     the paired per-unit shift, and how many errors are new under the condition);
  3. confident errors: the share of the condition's errors whose margin is at or above the median margin of the
     correct units under full input (the threshold is per family and per probe seed);
  4. the margin's AUROC for errors, its excess AURC, each control's, and its lead over the best control;
  5. on PASTIS, a mixed map: floor(N/2) of the N test tiles drawn with numpy's default_rng(88) take the
     optical-missing units, the rest keep full input. (a) the 5% review set by margin: the share of the cloudy
     part's errors inside it, beside the cloudy part's share of all errors and of the review set's errors; (b) 2,000
     random samples of 300 windows (default_rng(0), exp78's draw seed): the pooled estimate against the cloudy
     part's true rate, and the per-condition (post-stratified) intervals' coverage of each part's true rate.
Predictions, graded on OlmoEarth Base at probe seed 0 (seeds 1-4 and Large are read the same way and reported):
  P1  PASTIS: error rate under optical missing - under full >= 5 points.
  P2  PASTIS: confident-error share under optical missing - under full >= 5 points. If it fails, the confidence
      tracks the lost information; that is reported as the answer.
  P3  under optical missing the margin's lead over the best no-model control is > 0 on PASTIS and on China 6.
  P4  mixed map: |pooled estimate - cloudy part's rate| >= 5 points, and each part's interval covers its true rate
      on >= 93% of the 2,000 draws.
Every threshold is compared with a tolerance of 1e-12, so a float difference that prints as 0.0500 is 0.05.

Interpretations of the plan page, each flagged in the commit that added this file; none changes a threshold:
  a. P3's "best no-model control" is the record's: the better (lower excess AURC) of exp70's two controls, the
     embedding distance (exp74's window embedding, distance from the mean of the probe's own S1+S2 training
     embeddings) and the class rarity, chosen per condition. The lead over the embedding distance alone is
     reported beside it as `reading_embedding_distance_only`, since the page names that control in parentheses.
     A control measured from the mean of the condition's own (S1-only) training embeddings was not computed.
  b. P2 names no task; it is graded on PASTIS, the primary task, like P1. China 6's reading of P1 and P2 is
     reported as replication.
  c. P4's "off by at least 5 points" is graded on the pooled estimator's bias over the 2,000 draws (mean pooled
     estimate minus the cloudy part's true rate), the smaller and so the more conservative of the readings; the
     mean absolute error and the share of single draws off by 5 points are reported beside it.
  d. P4's stratified estimate post-stratifies the one random sample of 300 by condition. Each part's interval is
     the package's interval for a random sample (estimate_error_rate, design "random": the exact hypergeometric
     interval) on that part's labelled windows, with the part's size as the population. A draw that labels no
     window of a part counts as not covering. The finite-population Wilson interval exp78 and exp79 graded is
     reported beside it, and so is the post-stratified whole-map estimate.
  e. The mixed map is drawn over all N test tiles, including tiles with no valid window; the probe reads each tile
     alone (a linear map per token, interpolated within the tile), so a whole tile's units under optical missing
     are exactly the optical-missing condition's units for that tile, and no probe is rerun for the mixed map.
  f. OlmoEarth Large runs when the four files the design reads are cached for every family (its S1+S2 train split
     and the three test splits), at the same five probe seeds as Base, and is reported beside it, never graded.
  g. --smoke replaces the fitted probe by a fixed numpy stand-in (a softmax over class prototypes), because the
     package's environment has no torch; everything after the probabilities is the code the full run uses. The
     fitting and loading code is exercised by --smoke-torch, which the cluster job runs before either real mode:
     exp54's windows reproduced, then the alignment check and the full run (run_all, past the freeze guard only)
     end to end on a synthetic Hub cache read offline, for both encoders.
  h. The full run refuses to start unless the plan page's status line says frozen ("**Status: frozen ...", the
     form the archived agent_trial_v3.md took), so no run on real embeddings can precede the freeze by accident.
     --check-alignment is allowed before it and reads only identifiers (the labels and any other per-unit key a
     file carries, compared in order and reported by digest, never tabulated), shapes, dtypes and split sizes; the
     embeddings are opened memory-mapped where torch allows and no value of theirs enters any computation.
  i. A reported check, not a gate: the full condition's accuracy at each seed beside the record's accuracy of the
     same S1+S2 probe (exp79's per-seed export; exp70 for Base at seed 0).

Outputs. exp/out/exp88_alignment.json (--check-alignment); exp/out/exp88_summary.json and exp/out/exp88_units.npz
(per-unit margin and error for every seed, top probability and decision at seed 0; if the file would exceed about
60 MB, Large's replicate seeds are dropped first, then Base's); _smoke variants from --smoke. --out-dir moves them.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
for _p in (os.path.join(ROOT, "scripts"), ROOT, EXP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from oe_inferencex import estimate as est, metrics      # noqa: E402
import exp70_task_suite as e70                          # noqa: E402
import exp78_error_rate_estimation as e78               # noqa: E402
import suite_regression as sr                           # noqa: E402
import upstream_revision                                # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
PLAN = os.path.join(ROOT, "docs", "plan", "missing_modality.md")
HUB = e70.HUB
BASE, LARGE = "olmoearth_base", "olmoearth_large"
CONDITIONS = ("full", "optical_missing", "radar_missing")
FAMILIES = {
    "pastis": {"full": "pastis_sentinel1_sentinel2", "optical_missing": "pastis_sentinel1",
               "radar_missing": "pastis_sentinel2", "role": "primary"},
    "china6": {"full": "cropharvest_Peoples_Republic_of_China_6_sentinel1_sentinel2",
               "optical_missing": "cropharvest_Peoples_Republic_of_China_6_sentinel1",
               "radar_missing": "cropharvest_Peoples_Republic_of_China_6", "role": "replication"},
    "togo12": {"full": "cropharvest_Togo_12_sentinel2_sentinel1", "optical_missing": "cropharvest_Togo_12_sentinel1",
               "radar_missing": "cropharvest_Togo_12_sentinel2", "role": "reported, not graded"},
}
PRIMARY, REPLICATION = "pastis", "china6"
SEEDS, GRADED_SEED = (0, 1, 2, 3, 4), 0
LARGE_SEEDS = SEEDS                         # the replication encoder is read at the same probe seeds
MIX_SEED, DRAW_SEED = 88, 0
SAMPLE, DRAWS, REVIEW_BUDGET = 300, 2000, 0.05
# the thresholds the owner confirmed on 28 September 2026 (docs/plan/missing_modality.md)
P1_POINTS = P2_POINTS = P4_POINTS = 0.05
P4_COVER = 0.93
EPS = 1e-12
NPZ_LIMIT = 60e6
# what the two synthetic cases of --smoke must grade; tests/test_exp88.py reads these
SMOKE_EXPECTED = {"confident": {"P1": True, "P2": True, "P3": False, "P4": True},
                  "tracks": {"P1": True, "P2": False, "P3": True, "P4": True}}


# ----------------------------------------------------------------------------- small helpers
def _median(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.median(x)) if x.size else float("nan")


def _share(mask):
    mask = np.asarray(mask, bool)
    return float(mask.mean()) if mask.size else float("nan")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(summary, part, ex):
    summary["failures"].append({"part": part, "error": repr(ex), "traceback": traceback.format_exc()})
    print(f"{part} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)


def prereg_status(text=None):
    """'frozen', 'draft' or 'unknown', from the plan page's status line (the first line starting '**Status')."""
    if text is None:
        with open(PLAN, encoding="utf-8") as f:
            text = f.read()
    for line in text.splitlines():
        if line.startswith("**Status"):
            low = line.lower()
            # "frozen" must come first after the label ("**Status: frozen on ...", as the archived agent_trial_v3.md
            # read), so a frozen line that mentions the earlier draft is still frozen and "not frozen" never matches
            if re.match(r"\*\*status:?\*?\*?\s*frozen", low):
                return "frozen"
            if "draft" in low or "not frozen" in low:
                return "draft"
            return "unknown"
    return "unknown"


# ----------------------------------------------------------------------------- measures 1-4
def units_of(p, y):
    """Per-unit margin, top probability, decision and error from a probability matrix (exp70's reading rule)."""
    sig, dec = e70.readings_from_probs(p)
    return {"margin": -sig["margin"], "p1": 1.0 - sig["one_minus_top1"], "dec": dec,
            "err": (dec != np.asarray(y)).astype(np.float64)}


def condition_measures(u, rec, thr, full=None):
    """Measures 1-4 for one condition; `rec` is suite_regression.score_units' record on the same units, `thr` the
    confident-error threshold, `full` the full-input units when this is not the full condition."""
    err = u["err"] > 0.5
    m = u["margin"]
    sig = rec["signals"]
    ctl = {c: sig[c]["excess_aurc"] for c in e70.CONTROLS}
    out = {"n_units": int(err.size), "n_errors": int(err.sum()), "error_rate": float(err.mean()),
           "margin_median_errors": _median(m[err]), "margin_median_correct": _median(m[~err]),
           "confident_errors": {"threshold": thr, "n": int((m[err] >= thr).sum()), "share": _share(m[err] >= thr)},
           "ranking": {"margin_auroc": sig["margin"]["auroc"], "margin_excess_aurc": sig["margin"]["excess_aurc"],
                       "controls_excess_aurc": ctl, "best_control": rec["best_control"],
                       "lead_over_best_control": rec["margin_lead"],
                       "lead_over_embedding_distance": float(ctl["ctl_embedding_distance"] - sig["margin"]["excess_aurc"])},
           "signals": sig}
    if full is not None:
        fe = full["err"] > 0.5
        new = err & ~fe
        out["shift_from_full"] = {
            "error_rate_points": float(err.mean() - fe.mean()),
            "margin_median_errors": out["margin_median_errors"] - _median(full["margin"][fe]),
            "margin_median_correct": out["margin_median_correct"] - _median(full["margin"][~fe]),
            "paired_median_margin_shift": _median(m - full["margin"]),
            "new_errors": int(new.sum()), "errors_corrected": int((fe & ~err).sum()),
            "confident_share_of_new_errors": _share(m[new] >= thr)}
    return out


def evaluate_family(inputs):
    """Measures 1-4 on one family at one probe seed.

    inputs: {"conditions": {condition: {"p": (n, C), "y": (n,), "emb_te": (n, D), "tile": (n,) or None}},
             "emb_tr": (M, D) the probe's training embeddings, "C": classes}. Returns (measures, per-unit arrays)."""
    C, conds = inputs["C"], inputs["conditions"]
    y0 = np.asarray(conds["full"]["y"])
    t0 = conds["full"].get("tile")
    for c in CONDITIONS:
        if not np.array_equal(np.asarray(conds[c]["y"]), y0):
            raise RuntimeError(f"{c}: the units differ from the full input's")
        if (t0 is None) != (conds[c].get("tile") is None) or (t0 is not None and not np.array_equal(conds[c]["tile"], t0)):
            raise RuntimeError(f"{c}: the tiles differ from the full input's")
    units, recs = {}, {}
    for c in CONDITIONS:
        k = conds[c]
        recs[c] = sr.score_units(k["p"], k["emb_te"], inputs["emb_tr"], k["y"], C)
        units[c] = units_of(k["p"], k["y"])
    full = units["full"]
    thr = _median(full["margin"][full["err"] < 0.5])
    meas = {c: condition_measures(units[c], recs[c], thr, None if c == "full" else full) for c in CONDITIONS}
    return ({"n_units": int(y0.size), "n_classes": int(C), "threshold_confident": thr, "conditions": meas},
            {"y": y0, "tile": t0, **units})


# ----------------------------------------------------------------------------- measure 5: the mixed map
def mixed_map(full, cloud, tile, n_tiles, seed=MIX_SEED):
    """floor(n_tiles / 2) tiles drawn at random take the optical-missing units; the rest keep full input."""
    rng = np.random.default_rng(seed)
    cloudy_tiles = np.sort(rng.choice(n_tiles, n_tiles // 2, replace=False))
    cloudy = np.isin(tile, cloudy_tiles)
    return {"cloudy_tiles": cloudy_tiles, "cloudy": cloudy,
            **{k: np.where(cloudy, cloud[k], full[k]) for k in ("margin", "p1", "err")}}


def review_shares(margin, err, cloudy, budget=REVIEW_BUDGET):
    """Measure 5a: the review set is the `budget` share of the map's windows with the lowest margin, ties broken at
    random in expectation (metrics.capture_at_budget_expected)."""
    u = -np.asarray(margin, dtype=np.float64)
    e = np.asarray(err) > 0.5
    cl = np.asarray(cloudy, bool)

    def cap(indicator):
        return float(metrics.capture_at_budget_expected(u, indicator, (budget,))[budget])

    n_err, n_cl = int(e.sum()), int((e & cl).sum())
    c_all, c_cl, c_clear = cap(e), cap(e & cl), cap(e & ~cl)
    in_review = c_all * n_err
    return {"budget": budget, "review_windows": max(1, int(round(budget * e.size))),
            "cloudy_errors_in_review_share": c_cl, "clear_errors_in_review_share": c_clear,
            "all_errors_in_review_share": c_all,
            "cloudy_share_of_all_errors": n_cl / n_err if n_err else float("nan"),
            "cloudy_share_of_review_errors": c_cl * n_cl / in_review if in_review > 0 else float("nan")}


def estimation_study(err, cloudy, budget=SAMPLE, draws=DRAWS, seed=DRAW_SEED):
    """Measure 5b: random samples of `budget` windows from the mixed map; the pooled estimate, and the estimate
    post-stratified by condition with each part's interval, against the known truth, over `draws` draws."""
    err = (np.asarray(err) > 0.5).astype(np.float64)
    cloudy = np.asarray(cloudy, bool)
    N = err.size
    parts = {"cloudy": cloudy, "clear": ~cloudy}
    size = {h: int(m.sum()) for h, m in parts.items()}
    if min(size.values()) == 0:
        raise ValueError(f"a part of the mixed map is empty: {size}")
    truth = {h: float(err[m].mean()) for h, m in parts.items()}
    truth["whole"] = float(err.mean())
    local = np.empty(N, dtype=np.int64)                        # each window's position inside its own part
    for m in parts.values():
        local[m] = np.arange(int(m.sum()))
    strata = cloudy.astype(np.int64)                           # 0 clear, 1 cloudy: the whole-map post-stratified estimate
    sizes = [size["clear"], size["cloudy"]]
    rng = np.random.default_rng(seed)
    pooled = np.empty(draws)
    pooled_cover = {"cloudy": 0, "whole": 0}
    per = {h: {"est": [], "half": [], "n": [], "cover": 0, "cover_wilson": 0, "empty": 0} for h in parts}
    ps_est, ps_cover = [], 0
    for d in range(draws):
        idx = rng.choice(N, budget, replace=False)
        whole = est.estimate_error_rate({"design": "random", "indices": idx, "n_population": N}, err[idx])
        pooled[d] = whole["estimate"]
        pooled_cover["cloudy"] += whole["low"] <= truth["cloudy"] <= whole["high"]
        pooled_cover["whole"] += whole["low"] <= truth["whole"] <= whole["high"]
        for h, m in parts.items():
            sel = idx[m[idx]]
            P = per[h]
            P["n"].append(sel.size)
            if sel.size == 0:
                P["empty"] += 1
                continue
            res = est.estimate_error_rate({"design": "random", "indices": local[sel], "n_population": size[h]}, err[sel])
            P["est"].append(res["estimate"])
            P["half"].append(res["half_width"])
            P["cover"] += res["low"] <= truth[h] <= res["high"]
            wl, wh = est.wilson_interval(int(err[sel].sum()), int(sel.size), size[h])
            P["cover_wilson"] += wl <= truth[h] <= wh
        e_ps, lo_ps, hi_ps = est.stratified_interval_wilson(err, strata, idx, sizes, N)[:3]
        ps_est.append(e_ps)
        ps_cover += lo_ps <= truth["whole"] <= hi_ps
    off = pooled - truth["cloudy"]
    return {"budget": budget, "draws": draws, "seed": seed, "sizes": size, "truth": truth,
            "pooled": {"mean_estimate": float(pooled.mean()), "bias_for_cloudy_part": float(off.mean()),
                       "mean_abs_error_for_cloudy_part": float(np.abs(off).mean()),
                       "share_of_draws_off_by_threshold": float((np.abs(off) >= P4_POINTS - EPS).mean()),
                       "interval_covers_cloudy_part": pooled_cover["cloudy"] / draws,
                       "interval_covers_whole_map": pooled_cover["whole"] / draws,
                       "method": "estimate_error_rate, design random, over the whole mixed map"},
            "stratified": {h: {"coverage": P["cover"] / draws, "coverage_wilson_fpc": P["cover_wilson"] / draws,
                               "mean_estimate": float(np.mean(P["est"])) if P["est"] else float("nan"),
                               "bias": (float(np.mean(P["est"])) - truth[h]) if P["est"] else float("nan"),
                               "mean_half_width": float(np.mean(P["half"])) if P["half"] else float("nan"),
                               "mean_labelled": float(np.mean(P["n"])), "draws_without_a_label": P["empty"]}
                           for h, P in per.items()},
            "post_stratified_whole_map": {"mean_estimate": float(np.mean(ps_est)), "coverage": ps_cover / draws,
                                          "method": "stratified_interval_wilson with the two parts as strata"}}


def evaluate_mixed(units, n_tiles):
    """Measure 5 on the PASTIS units of one probe seed. Returns (measures, the mixed map's arrays)."""
    mm = mixed_map(units["full"], units["optical_missing"], units["tile"], n_tiles)
    e, cl = mm["err"] > 0.5, mm["cloudy"]
    out = {"n_tiles": int(n_tiles), "n_cloudy_tiles": int(mm["cloudy_tiles"].size), "mix_seed": MIX_SEED,
           "n_tiles_with_windows": int(np.unique(units["tile"]).size),
           "n_cloudy_tiles_with_windows": int(np.unique(units["tile"][cl]).size),
           "n_windows": int(e.size), "n_cloudy_windows": int(cl.sum()),
           "error_rate": {"cloudy": float(e[cl].mean()), "clear": float(e[~cl].mean()), "whole": float(e.mean())},
           "review_set": review_shares(mm["margin"], mm["err"], cl),
           "estimation": estimation_study(mm["err"], cl)}
    return out, mm


# ----------------------------------------------------------------------------- the predictions
def read_predictions(seed_res):
    """P1-P4 as the plan page states them, read on one encoder at one probe seed."""
    fams = seed_res.get("families", {})

    def cond(f, c):
        return fams.get(f, {}).get("conditions", {}).get(c)

    out = {}
    readings_12 = {}
    for f in (PRIMARY, REPLICATION):
        full, opt = cond(f, "full"), cond(f, "optical_missing")
        if full and opt:
            rise = opt["error_rate"] - full["error_rate"]
            d = opt["confident_errors"]["share"] - full["confident_errors"]["share"]
            readings_12[f] = {
                "P1": {"holds": bool(rise >= P1_POINTS - EPS), "task": FAMILIES[f]["optical_missing"],
                       "error_rate_full": full["error_rate"], "error_rate_optical_missing": opt["error_rate"],
                       "rise_points": rise, "threshold_points": P1_POINTS},
                "P2": {"holds": bool(np.isfinite(d) and d >= P2_POINTS - EPS),
                       "confident_share_full": full["confident_errors"]["share"],
                       "confident_share_optical_missing": opt["confident_errors"]["share"],
                       "threshold_margin": full["confident_errors"]["threshold"],
                       "rise_points": d, "threshold_points": P2_POINTS}}
    missing = {"holds": None, "reason": f"{PRIMARY} has no result at this seed"}
    out["P1"] = readings_12.get(PRIMARY, {}).get("P1", dict(missing))
    out["P2"] = readings_12.get(PRIMARY, {}).get("P2", dict(missing))
    if out["P2"]["holds"] is not None:
        out["P2"]["reading"] = ("the extra errors under optical missing are confident ones: the review order misses "
                                "them" if out["P2"]["holds"] else
                                "the confidence tracks the lost information, within the measured size; this is "
                                "the answer, not a failure of the experiment")
    out["replication_china6"] = readings_12.get(REPLICATION, {"note": f"{REPLICATION} has no result at this seed"})

    per = {}
    for f in (PRIMARY, REPLICATION):
        c = cond(f, "optical_missing")
        if c:
            r = c["ranking"]
            per[f] = {"lead_over_best_control": r["lead_over_best_control"], "best_control": r["best_control"],
                      "lead_over_embedding_distance": r["lead_over_embedding_distance"],
                      "margin_auroc": r["margin_auroc"], "margin_excess_aurc": r["margin_excess_aurc"]}
    if len(per) == 2:
        out["P3"] = {"holds": bool(all(v["lead_over_best_control"] > 0 for v in per.values())), "per_family": per,
                     "reading_embedding_distance_only": {
                         "holds": bool(all(v["lead_over_embedding_distance"] > 0 for v in per.values())),
                         "note": "reported, not graded: the lead over the embedding-distance control alone"}}
    else:
        out["P3"] = {"holds": None, "reason": f"only {sorted(per)} have a result at this seed", "per_family": per}

    mm = seed_res.get("mixed_map")
    if mm:
        es = mm["estimation"]
        bias = es["pooled"]["bias_for_cloudy_part"]
        cov = {h: es["stratified"][h]["coverage"] for h in ("cloudy", "clear")}
        off = bool(abs(bias) >= P4_POINTS - EPS)
        covered = bool(all(v >= P4_COVER - EPS for v in cov.values()))
        out["P4"] = {"holds": off and covered, "pooled_off_by_threshold": off, "strata_covered": covered,
                     "pooled_bias_points": bias, "truth": es["truth"], "pooled_mean_estimate": es["pooled"]["mean_estimate"],
                     "coverage": cov, "coverage_wilson_fpc": {h: es["stratified"][h]["coverage_wilson_fpc"] for h in cov},
                     "threshold_points": P4_POINTS, "coverage_bar": P4_COVER}
    else:
        out["P4"] = {"holds": None, "reason": "no mixed map at this seed"}
    return out


def grades(summary, model=BASE, seed=GRADED_SEED):
    """The graded reading: one encoder at one probe seed (OlmoEarth Base at seed 0 in the run)."""
    S = summary["results"].get(model, {}).get("seeds", {}).get(str(seed))
    if not S:
        return {"graded_on": f"{model}, probe seed {seed}", "complete": False,
                **{p: {"holds": None, "reason": "no result"} for p in ("P1", "P2", "P3", "P4")}}
    r = S.get("readings") or read_predictions(S)
    return {"graded_on": f"{model}, probe seed {seed}",
            "complete": all(r[p]["holds"] is not None for p in ("P1", "P2", "P3", "P4")), **r}


def replication_table(summary):
    """P1-P4 read at every (encoder, seed) that ran; the graded cell is one of them."""
    return {model: {s: {p: S.get("readings", {}).get(p, {}).get("holds") for p in ("P1", "P2", "P3", "P4")}
                    for s, S in R.get("seeds", {}).items()}
            for model, R in summary["results"].items()}


def planted_checks():
    """The grades on planted numbers at their thresholds, both sides of each."""
    def seed_res(err=(0.20, 0.25), conf=(0.10, 0.15), leads=(0.01, 0.01), bias=-0.10, cov=(0.95, 0.95)):
        def condition(e, c, lead):
            return {"error_rate": e, "confident_errors": {"share": c, "threshold": 0.5},
                    "ranking": {"lead_over_best_control": lead, "best_control": "ctl_class_rarity",
                                "lead_over_embedding_distance": lead, "margin_auroc": 0.7, "margin_excess_aurc": 0.05}}
        fam = {f: {"conditions": {"full": condition(err[0], conf[0], 0.02),
                                  "optical_missing": condition(err[1], conf[1], lead)}}
               for f, lead in zip((PRIMARY, REPLICATION), leads)}
        truth = {"cloudy": 0.30, "clear": 0.20, "whole": 0.25}
        return {"families": fam, "mixed_map": {"estimation": {
            "truth": truth, "pooled": {"bias_for_cloudy_part": bias, "mean_estimate": truth["cloudy"] + bias},
            "stratified": {"cloudy": {"coverage": cov[0], "coverage_wilson_fpc": cov[0]},
                           "clear": {"coverage": cov[1], "coverage_wilson_fpc": cov[1]}}}}}

    got = {
        "P1_at_threshold": read_predictions(seed_res(err=(0.80, 0.85)))["P1"]["holds"],      # 0.85 - 0.80 < 0.05 in floats
        "P1_below": read_predictions(seed_res(err=(0.20, 0.249)))["P1"]["holds"],
        "P2_at_threshold": read_predictions(seed_res(conf=(0.10, 0.15)))["P2"]["holds"],
        "P2_below": read_predictions(seed_res(conf=(0.10, 0.149)))["P2"]["holds"],
        "P2_falls": read_predictions(seed_res(conf=(0.30, 0.10)))["P2"]["holds"],
        "P3_both_ahead": read_predictions(seed_res(leads=(0.01, 0.002)))["P3"]["holds"],
        "P3_replication_behind": read_predictions(seed_res(leads=(0.01, -0.001)))["P3"]["holds"],
        "P3_tie_is_not_a_win": read_predictions(seed_res(leads=(0.0, 0.01)))["P3"]["holds"],
        "P4_holds": read_predictions(seed_res(bias=-0.06, cov=(0.93, 0.95)))["P4"]["holds"],
        "P4_over_estimate_counts": read_predictions(seed_res(bias=0.05, cov=(0.95, 0.95)))["P4"]["holds"],
        "P4_bias_below": read_predictions(seed_res(bias=-0.049, cov=(0.95, 0.95)))["P4"]["holds"],
        "P4_one_part_under_covers": read_predictions(seed_res(bias=-0.10, cov=(0.95, 0.929)))["P4"]["holds"],
        "P1_without_pastis": read_predictions({"families": {}})["P1"]["holds"],
    }
    want = {"P1_at_threshold": True, "P1_below": False, "P2_at_threshold": True, "P2_below": False, "P2_falls": False,
            "P3_both_ahead": True, "P3_replication_behind": False, "P3_tie_is_not_a_win": False, "P4_holds": True,
            "P4_over_estimate_counts": True, "P4_bias_below": False, "P4_one_part_under_covers": False,
            "P1_without_pastis": None}
    bad = {k: (got[k], want[k]) for k in want if got[k] is not want[k]}
    if bad:
        raise AssertionError(f"planted grades wrong: {bad}")
    return got


# ----------------------------------------------------------------------------- per-unit export
def store_units(arrays, model, fam, seed, units):
    base = f"{model}/{fam}"
    if f"{base}/y" not in arrays:
        arrays[f"{base}/y"] = np.asarray(units["y"]).astype(np.int16)
        if units["tile"] is not None:
            arrays[f"{base}/tile"] = np.asarray(units["tile"]).astype(np.int32)
    for c in CONDITIONS:
        u = units[c]
        arrays[f"{base}/seed{seed}/{c}/margin"] = u["margin"].astype(np.float32)
        arrays[f"{base}/seed{seed}/{c}/err"] = u["err"].astype(np.uint8)
        if seed == GRADED_SEED:
            arrays[f"{base}/seed{seed}/{c}/p1"] = u["p1"].astype(np.float32)
            arrays[f"{base}/seed{seed}/{c}/dec"] = np.asarray(u["dec"]).astype(np.int16)


def write_units(arrays, path, limit=NPZ_LIMIT, graded=BASE):
    """Every array while the file stays under `limit` bytes. Over it, the replicate seeds are dropped in two steps,
    first those of the replication encoder (every key not under `graded`), then the graded encoder's; if the file
    is still over the limit with seed 0 alone, nothing is written."""
    if not arrays:
        return {"written": False, "reason": "no arrays"}

    def replicate(k):
        return "/seed" in k and f"/seed{GRADED_SEED}/" not in k

    steps = [("none", arrays),
             ("replication encoder", {k: v for k, v in arrays.items() if k.startswith(f"{graded}/") or not replicate(k)}),
             ("all", {k: v for k, v in arrays.items() if not replicate(k)})]
    for dropped, keep in steps:
        np.savez_compressed(path, **keep)
        size = os.path.getsize(path)
        if size <= limit:
            return {"written": True, "path": os.path.relpath(path, ROOT), "bytes": size, "sha256": _sha256(path),
                    "n_arrays": len(keep), "replicate_seeds_dropped": dropped != "none",
                    "replicate_seeds_dropped_for": dropped}
    os.remove(path)
    return {"written": False, "reason": f"{size} bytes with seed {GRADED_SEED} alone, over the {limit:.0f}-byte limit"}


def new_summary(smoke):
    return {"experiment": "exp88 missing modality: does the confidence fall with the accuracy?", "smoke": smoke,
            "prereg": None, "prereg_page": os.path.relpath(PLAN, ROOT), "prereg_status": prereg_status(),
            "config": {"families": FAMILIES, "conditions": list(CONDITIONS), "seeds": list(SEEDS),
                       "graded": f"{BASE}, probe seed {GRADED_SEED}", "large_seeds": list(LARGE_SEEDS),
                       "mix_seed": MIX_SEED, "draw_seed": DRAW_SEED, "sample": SAMPLE, "draws": DRAWS,
                       "review_budget": REVIEW_BUDGET, "p1_points": P1_POINTS, "p2_points": P2_POINTS,
                       "p4_points": P4_POINTS, "p4_cover": P4_COVER, "tolerance": EPS,
                       "controls": list(e70.CONTROLS)},
            "results": {}, "failures": []}


def dump(summary, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(e78.jsonable(summary), f, indent=1)


# ----------------------------------------------------------------------------- the smoke (numpy only)
PROTO_NORM, SIGMA, BETA = 1.0, 0.35, 4.0        # full-input accuracy about 0.90 on the segmentation family


def stand_in_probe(x, protos, beta=BETA):
    """The smoke's probe: a softmax over the dot products with fixed class prototypes. It stands in for the fitted
    probe only; everything the smoke checks starts from its probabilities."""
    z = beta * (np.asarray(x, dtype=np.float64) @ protos.T)
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def synthetic_family(kind, case, seed, C, n_units=None, n_tiles=None, grid=8, D=16):
    """Synthetic embeddings of one family under the three inputs, on the same units.

    Full input: each unit's embedding is its class prototype plus noise. Radar missing: the prototype at 0.85 of its
    strength. Optical missing, case "confident": 35% of the units carry a wrong class's prototype at 1.5 times the
    strength, so the extra errors are confident ones; case "tracks": every prototype at 0.3 of its strength, so the
    margin falls with the accuracy."""
    rng = np.random.default_rng(seed)
    protos = rng.standard_normal((C, D))
    protos *= PROTO_NORM / np.linalg.norm(protos, axis=1, keepdims=True)
    if kind == "segmentation":
        lab = np.empty((n_tiles, grid, grid), dtype=np.int64)
        for t in range(n_tiles):
            a, b = rng.choice(C, 2, replace=False)
            cut = int(rng.integers(2, grid - 1))
            lab[t, :, :cut], lab[t, :, cut:] = a, b
        ok = np.ones((n_tiles, grid, grid), bool)
        ok[:, 0, :] = False                                     # a row of unlabelled windows per tile
        y = lab.reshape(-1)[ok.reshape(-1)]
        tile = np.repeat(np.arange(n_tiles), grid * grid)[ok.reshape(-1)]
    else:
        y, tile = rng.integers(0, C, n_units), None
    ytr = rng.integers(0, C, 1500)
    emb_tr = protos[ytr] + SIGMA * rng.standard_normal((ytr.size, D))

    def embed(strength, swap=0.0, swap_strength=1.0):
        cls, s = y.copy(), np.full(y.size, float(strength))
        if swap:
            hit = rng.random(y.size) < swap
            cls[hit] = (y[hit] + 1 + rng.integers(0, C - 1, int(hit.sum()))) % C
            s[hit] = swap_strength
        return s[:, None] * protos[cls] + SIGMA * rng.standard_normal((y.size, D))

    views = {"full": embed(1.0),
             "optical_missing": embed(1.0, swap=0.35, swap_strength=1.5) if case == "confident" else embed(0.3),
             "radar_missing": embed(0.85)}
    conds = {c: {"p": stand_in_probe(x, protos), "y": y, "tile": tile, "emb_te": x.astype(np.float32)}
             for c, x in views.items()}
    return {"conditions": conds, "emb_tr": emb_tr.astype(np.float32), "C": C, "n_tiles": n_tiles}


SMOKE_FAMILIES = {"pastis": ("segmentation", 5, {"n_tiles": 48}), "china6": ("classification", 2, {"n_units": 1500}),
                  "togo12": ("classification", 2, {"n_units": 300})}


def smoke(out_dir=None):
    """Both synthetic cases through every measure and every grade; planted grades; the freeze guard's reading."""
    out_dir = out_dir or OUT
    t0 = time.time()
    cases, arrays = {}, {}
    for case in SMOKE_EXPECTED:
        summary = new_summary(smoke=True)
        summary["synthetic_case"] = case
        S = {"families": {}}
        for i, (fam, (kind, C, size)) in enumerate(SMOKE_FAMILIES.items()):
            inputs = synthetic_family(kind, case, seed=10 + i, C=C, **size)
            S["families"][fam], units = evaluate_family(inputs)
            store_units(arrays, case, fam, GRADED_SEED, units)
            if fam == PRIMARY:
                S["mixed_map"], mm = evaluate_mixed(units, inputs["n_tiles"])
                arrays[f"{case}/{fam}/cloudy_tiles"] = mm["cloudy_tiles"].astype(np.int32)
        S["readings"] = read_predictions(S)
        summary["results"] = {"synthetic": {"seeds": {str(GRADED_SEED): S}}}
        summary["prereg"] = grades(summary, model="synthetic")
        summary["replication"] = replication_table(summary)
        cases[case] = summary
    out = {"experiment": "exp88 smoke: synthetic embeddings, two cases", "smoke": True, "cases": cases,
           "planted": planted_checks(),
           "prereg_status_reading": {
               "frozen": prereg_status("**Status: frozen on 1 October 2026, before any run.**"),
               "frozen_naming_the_draft": prereg_status("**Status: frozen on 1 October 2026; the draft of 28 September "
                                                        "is in the history.**"),
               "draft": prereg_status("**Status: draft, not frozen.** Written 28 September.")}}
    assert out["prereg_status_reading"] == {"frozen": "frozen", "frozen_naming_the_draft": "frozen", "draft": "draft"}, \
        out["prereg_status_reading"]
    out["units_npz"] = write_units(arrays, os.path.join(out_dir, "exp88_units_smoke.npz"))
    out["seconds"] = time.time() - t0
    dump(out, os.path.join(out_dir, "exp88_summary_smoke.json"))
    for case, want in SMOKE_EXPECTED.items():
        g = cases[case]["prereg"]
        got = {p: g[p]["holds"] for p in want}
        leads = ", ".join("%s %+.4f" % (f, v["lead_over_best_control"]) for f, v in g["P3"]["per_family"].items())
        print(f"smoke case {case}: " + ", ".join(f"{p} {v}" for p, v in got.items()) +
              f" | P1 rise {g['P1']['rise_points']:+.3f}, P2 rise {g['P2']['rise_points']:+.3f}, P3 leads {leads}, "
              f"P4 bias {g['P4']['pooled_bias_points']:+.3f} coverage {g['P4']['coverage']}", flush=True)
        if got != want:
            raise AssertionError(f"smoke case {case}: graded {got}, the synthetic design implies {want}")
    print(f"smoke OK in {out['seconds']:.1f}s: both cases grade as designed, planted grades hold at their thresholds, "
          "the freeze guard reads the status line", flush=True)
    return out


# ----------------------------------------------------------------------------- the real files
def resolve_cache(arg=None):
    """The directory holding the Hub dataset's cache: HF_HOME itself (the cluster jobs point it there), or its
    paper_embeddings/ (exp54's layout) or hub/ subdirectory."""
    base = arg or os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    folder = "datasets--" + HUB.replace("/", "--")
    for c in (base, os.path.join(base, "paper_embeddings"), os.path.join(base, "hub")):
        if os.path.isdir(os.path.join(c, folder)):
            return c
    return base


def upstream_info(cache):
    pinned = upstream_revision.load()["repos"][HUB]["revision"]
    here = upstream_revision.cached_revision(cache, HUB) if cache else None
    return {"repo": HUB, "revision": here, "recorded_revision": pinned,
            "same_as_the_record": None if here is None else here == pinned}


def cached_path(model, task, split, cache):
    """The cached file's path, or None; never downloads."""
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(HUB, f"{model}/{task}/{split}.pt", cache_dir=cache, repo_type="dataset")
    return p if isinstance(p, str) else None


def design_files(model, cache):
    """The four files the run reads per family: the S1+S2 train split and the three test splits."""
    need = {}
    for fam, spec in FAMILIES.items():
        need[f"{fam}/{spec['full']}/train"] = cached_path(model, spec["full"], "train", cache)
        for c in CONDITIONS:
            need[f"{fam}/{spec[c]}/test"] = cached_path(model, spec[c], "test", cache)
    return need


def _digest(v):
    import torch
    if isinstance(v, torch.Tensor):
        t = v.detach().cpu().contiguous()
        if t.dtype == torch.bfloat16:
            t = t.view(torch.int16)
        return hashlib.sha1(t.numpy().tobytes()).hexdigest()
    return hashlib.sha1(repr(v).encode()).hexdigest()


def describe_file(path):
    """What one file says about its units: keys, the embeddings' shape and dtype (no value enters any computation),
    the labels and any other per-unit key as identifiers, reported by digest and never tabulated. Returns (json-ready
    description, what the comparison needs)."""
    import torch
    import exp54_multiclass_embeddings as e54
    try:
        d = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        mmap = True
    except Exception:                                          # an old-format file cannot be memory-mapped
        d = torch.load(path, map_location="cpu", weights_only=True)
        mmap = False
    emb, lab = d["embeddings"], d["labels"]
    info = {"cached": True, "memory_mapped": mmap, "keys": sorted(d.keys()), "n_units": int(lab.shape[0]),
            "embeddings": {"shape": list(emb.shape), "dtype": str(emb.dtype)},
            "labels": {"shape": list(lab.shape), "dtype": str(lab.dtype), "sha1": _digest(lab)}}
    labels = lab.clone()
    if lab.ndim == 3:
        # how many tiles the labels tell apart: an alignment by labels cannot see a swap between identical label tiles
        info["labels"]["unique_label_tiles"] = len(set(e54.label_keys(labels.numpy())))
    other = {}
    for k in d:
        if k in ("embeddings", "labels"):
            continue
        v = d[k]
        other[k] = v.clone() if isinstance(v, torch.Tensor) else v
        info.setdefault("other_keys", {})[k] = {"type": type(v).__name__,
                                                "shape": list(v.shape) if hasattr(v, "shape") else None,
                                                "length": len(v) if hasattr(v, "__len__") else None, "sha1": _digest(v)}
    held = {"labels": labels, "emb_shape": tuple(emb.shape), "other": other}
    del d, emb, lab
    return info, held


def compare_split(held, split):
    """Do the three inputs' files of one split carry the same units in the same order?"""
    import torch
    have = {c: held[(c, split)] for c in CONDITIONS if (c, split) in held}
    missing = [c for c in CONDITIONS if c not in have]
    if missing:
        return {"aligned": False, "reason": f"not cached: {missing}"}
    ref = have["full"]
    out = {}
    for c in CONDITIONS[1:]:
        h = have[c]
        same_labels = h["labels"].shape == ref["labels"].shape and bool(torch.equal(h["labels"], ref["labels"]))
        other = {}
        for k in sorted(set(ref["other"]) | set(h["other"])):
            a, b = ref["other"].get(k), h["other"].get(k)
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                other[k] = a.shape == b.shape and bool(torch.equal(a, b))
            else:
                other[k] = a is not None and b is not None and _digest(a) == _digest(b)
        chk = {"n_units_equal": h["labels"].shape[0] == ref["labels"].shape[0],
               "labels_equal_in_order": same_labels,
               "embedding_shape_equal": h["emb_shape"] == ref["emb_shape"],
               "other_keys_equal": other}
        chk["aligned"] = bool(chk["n_units_equal"] and same_labels and chk["embedding_shape_equal"] and all(other.values()))
        out[c] = chk
    idkeys = sorted(ref["other"])
    return {"aligned": all(v["aligned"] for v in out.values()), "against_full": out,
            "identifiers": ["labels"] + idkeys,
            "note": None if idkeys else ("the files carry no identifier but the labels: equal labels in the same order "
                                         "rule out any reordering across classes (or across label tiles), not a "
                                         "permutation among units with identical labels")}


def cmd_check_alignment(args):
    """Identifiers, shapes and split sizes of the three inputs of each family; no probe, prediction, error or margin."""
    t0 = time.time()
    cache = resolve_cache(args.cache)
    out_dir = args.out_dir or OUT
    out = {"experiment": "exp88 alignment check", "prereg_status": prereg_status(),
           "reads": "keys, the embeddings' shapes and dtypes, labels and any other per-unit key, split sizes; "
                    "no probe, prediction, error or margin is computed, no embedding value is read",
           "cache": cache, "upstream": upstream_info(cache), "encoders": {}}
    for model in (BASE, LARGE):
        E = {"families": {}}
        for fam, spec in FAMILIES.items():
            F = {"inputs": {}, "splits": {}}
            held = {}
            for c in CONDITIONS:
                F["inputs"][c] = {"task": spec[c]}
                for split in ("train", "test"):
                    path = cached_path(model, spec[c], split, cache)
                    if path is None:
                        F["inputs"][c][split] = {"cached": False}
                        continue
                    try:
                        F["inputs"][c][split], held[(c, split)] = describe_file(path)
                    except Exception as ex:  # noqa: BLE001
                        F["inputs"][c][split] = {"cached": True, "error": repr(ex)}
            for split in ("train", "test"):
                F["splits"][split] = compare_split(held, split)
            F["aligned_all_splits"] = all(F["splits"][s]["aligned"] for s in ("train", "test"))
            F["aligned_for_the_design"] = bool(F["splits"]["test"]["aligned"] and ("full", "train") in held)
            E["families"][fam] = F
            print(f"{model:16s} {fam:7s} test aligned {F['splits']['test']['aligned']}, train aligned "
                  f"{F['splits']['train']['aligned']} | " + ", ".join(
                      f"{c} {F['inputs'][c].get('test', {}).get('embeddings', {}).get('shape')}" for c in CONDITIONS), flush=True)
            del held
        E["cached_for_all_three_inputs"] = all(F["inputs"][c][s].get("cached", False) for F in E["families"].values()
                                               for c in CONDITIONS for s in ("train", "test"))
        E["cached_for_the_design"] = all(v is not None for v in design_files(model, cache).values())
        E["aligned_for_the_design"] = all(F["aligned_for_the_design"] for F in E["families"].values())
        out["encoders"][model] = E
    out["verdict"] = {"base_aligned_for_the_design": out["encoders"][BASE]["aligned_for_the_design"],
                      "large_runs_as_replication": bool(out["encoders"][LARGE]["cached_for_the_design"]
                                                        and out["encoders"][LARGE]["aligned_for_the_design"]),
                      "large_cached_for_all_three_inputs": out["encoders"][LARGE]["cached_for_all_three_inputs"]}
    out["seconds"] = time.time() - t0
    path = os.path.join(out_dir, "exp88_alignment.json")
    dump(out, path)
    print(f"wrote {path}: {out['verdict']}", flush=True)
    return 0 if out["verdict"]["base_aligned_for_the_design"] else 1


def load_family(model, fam, cache):
    """The S1+S2 train and test splits (suite_regression.load_task) and the other two inputs' test splits, refused
    unless their units are the S1+S2 test units in the same order (exp75's check)."""
    import torch
    import exp54_multiclass_embeddings as e54
    spec = FAMILIES[fam]
    L = sr.load_task(model, spec["full"], cache)
    tests = {"full": (L["xte"], L["yte"])}
    for c in CONDITIONS[1:]:
        xte, yte = e54.load_theirs(model, spec[c], "test", cache)
        same = yte.shape == L["yte"].shape and bool(torch.equal(yte, L["yte"]))
        if tuple(xte.shape) != tuple(L["xte"].shape) or not same:
            raise RuntimeError(f"{spec[c]}: the test units do not line up with {spec['full']}'s "
                               f"(embeddings {tuple(xte.shape)} vs {tuple(L['xte'].shape)}, labels equal {same})")
        tests[c] = (xte, yte)
    L["tests"] = tests
    return L


def fit_family(L, seed, views):
    """The S1+S2 probe at one seed, read on the three inputs' test units. `views` caches the seed-independent test
    embeddings per graded unit (and the validity mask, which must not move between seeds)."""
    import torch
    import exp54_multiclass_embeddings as e54
    import exp73_alternatives_suite as e73
    import exp74_suite_encoders as e74
    import exp75_sensor_views as e75
    C = L["C"]
    conds = {}
    if L["family"] == "classification":
        probs = e73._train_cls_probe(L["xtr"], L["ytr"], C, seed)
        for c, (xte, yte) in L["tests"].items():
            if c not in views:
                views[c] = {"emb_te": xte.to(torch.float32).numpy()}
            conds[c] = {"p": probs(xte), "y": yte.numpy(), "tile": None, "emb_te": views[c]["emb_te"]}
        return {"conditions": conds, "emb_tr": L["emb_tr"], "C": C, "n_tiles": None}
    pp = L["patch_px"]
    probe = e54.train_probe(L["xtr"], L["ytr"], pp, C, e54.TASK_LR.get(L["task"], 0.1), seed=seed)
    for c, (xte, yte) in L["tests"].items():
        q = e75.seg_window_probs(probe, xte, yte, pp, C)
        ok = q["ok"]
        if c not in views:
            hw, ww = q["hw"]
            H, W = yte.shape[1], yte.shape[2]
            views[c] = {"ok": ok, "emb_te": e74.window_embeddings(xte.to(torch.float32).numpy(), hw, ww, H, W, e54.WIN)
                        .reshape(-1, xte.shape[-1])[ok]}
        elif not np.array_equal(ok, views[c]["ok"]):
            raise RuntimeError(f"{c}: seed {seed} changed the valid-window set; validity is a label property")
        conds[c] = {"p": q["P"][ok], "y": q["y"][ok], "tile": q["tile"][ok], "emb_te": views[c]["emb_te"]}
    return {"conditions": conds, "emb_tr": L["emb_tr"], "C": C, "n_tiles": int(L["yte"].shape[0])}


def record_accuracy(model, task, seed):
    """The record's test accuracy of the S1+S2 probe at this seed: exp79's per-seed export, else exp70 (Base, seed 0)."""
    p = os.path.join(OUT, "exp79_seeds", f"{model}.json")
    if os.path.exists(p):
        v = json.load(open(p)).get("tasks", {}).get(task)
        if v and seed < len(v.get("seeds", [])):
            return v["seeds"][seed]["test_accuracy"], "exp79_seeds"
    p70 = os.path.join(OUT, "exp70_summary.json")
    if model == BASE and seed == 0 and os.path.exists(p70):
        r = json.load(open(p70))["results"]["tasks"].get(task)
        if r:
            return r["test_accuracy"], "exp70"
    return None, None


def run_model(model, seeds, cache, summary, arrays, out_dir):
    import exp51_their_probe as e51
    import exp54_multiclass_embeddings as e54
    R = summary["results"].setdefault(model, {"families": {}, "seeds": {}})
    R["probe_lrs"] = {FAMILIES[f]["full"]: e54.TASK_LR.get(FAMILIES[f]["full"]) for f in FAMILIES}
    for fam, spec in FAMILIES.items():
        t0 = time.time()
        try:
            L = load_family(model, fam, cache)
        except Exception as ex:  # noqa: BLE001
            fail(summary, f"{model}/{fam} load", ex)
            continue
        seg = L["family"] == "segmentation"
        R["families"][fam] = {"tasks": {c: spec[c] for c in CONDITIONS}, "role": spec["role"], "family": L["family"],
                              "n_classes": L["C"], "train_items": int(L["ytr"].shape[0]), "test_items": int(L["yte"].shape[0]),
                              "embedding_shape": list(L["xte"].shape[1:]), "patch_px": L["patch_px"],
                              "probe_lr": e54.TASK_LR.get(spec["full"], 0.1) if seg else e51.LR,
                              "epochs": e54.EPOCHS if seg else e51.EPOCHS, "load_seconds": round(time.time() - t0, 1)}
        views = {}
        for seed in seeds:
            t1 = time.time()
            try:
                inputs = fit_family(L, seed, views)
                meas, units = evaluate_family(inputs)
                acc, src = record_accuracy(model, spec["full"], seed)
                acc_full = 1.0 - meas["conditions"]["full"]["error_rate"]
                meas["record_check"] = {"full_accuracy": acc_full, "recorded_accuracy": acc, "source": src,
                                        "difference": None if acc is None else acc_full - acc}
                S = R["seeds"].setdefault(str(seed), {"families": {}})
                S["families"][fam] = meas
                store_units(arrays, model, fam, seed, units)
                if fam == PRIMARY:
                    S["mixed_map"], mm = evaluate_mixed(units, inputs["n_tiles"])
                    arrays.setdefault(f"{model}/{fam}/cloudy_tiles", mm["cloudy_tiles"].astype(np.int32))
                meas["seconds"] = round(time.time() - t1, 1)
                cm = meas["conditions"]
                print(f"{model} {fam} seed {seed}: error rate " +
                      ", ".join(f"{c} {cm[c]['error_rate']:.4f}" for c in CONDITIONS) +
                      " | confident errors " + ", ".join(f"{c} {cm[c]['confident_errors']['share']:.3f}" for c in CONDITIONS) +
                      " | lead " + ", ".join(f"{c} {cm[c]['ranking']['lead_over_best_control']:+.4f}" for c in CONDITIONS) +
                      f" | full acc - record {meas['record_check']['difference']} ({meas['seconds']:.0f}s)", flush=True)
            except Exception as ex:  # noqa: BLE001
                fail(summary, f"{model}/{fam}/seed{seed}", ex)
        del L, views
        dump(summary, os.path.join(out_dir, "exp88_summary.partial.json"))
    for S in R["seeds"].values():
        S["readings"] = read_predictions(S)


def run_all(cache, out_dir, seeds=None):
    """The full run on a resolved cache: Base at every seed, Large at its seeds when the design's files are cached.
    Reached only through cmd_run's freeze guard, or from smoke_torch on a synthetic cache."""
    import exp54_multiclass_embeddings as e54
    seeds = seeds or {BASE: SEEDS, LARGE: LARGE_SEEDS}
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    summary = new_summary(smoke=False)
    summary["cache"], summary["upstream"] = cache, upstream_info(cache)
    print(f"upstream {summary['upstream']}", flush=True)
    arrays = {}
    large = design_files(LARGE, cache)
    plan = [(BASE, seeds[BASE])]
    if all(v is not None for v in large.values()):
        plan.append((LARGE, seeds[LARGE]))
        summary["large"] = {"runs": True, "seeds": list(seeds[LARGE])}
    else:
        summary["large"] = {"runs": False, "not_cached": sorted(k for k, v in large.items() if v is None)}
    for model, ss in plan:
        saved = dict(e54.TASK_LR)
        try:
            sr.set_probe_lrs(model, cache)
            run_model(model, ss, cache, summary, arrays, out_dir)
        except Exception as ex:  # noqa: BLE001
            fail(summary, model, ex)
        finally:
            e54.TASK_LR = saved
    summary["prereg"] = grades(summary)
    summary["replication"] = replication_table(summary)
    summary["units_npz"] = write_units(arrays, os.path.join(out_dir, "exp88_units.npz"))
    summary["n_failures"] = len(summary["failures"])
    summary["seconds"] = time.time() - t0
    dump(summary, os.path.join(out_dir, "exp88_summary.json"))
    partial = os.path.join(out_dir, "exp88_summary.partial.json")
    if os.path.exists(partial):
        os.remove(partial)
    g = summary["prereg"]
    print(json.dumps(e78.jsonable({p: g[p].get("holds") for p in ("P1", "P2", "P3", "P4")}), indent=1), flush=True)
    print(f"graded on {g['graded_on']}, complete {g['complete']}; replication {json.dumps(summary['replication'])}; "
          f"{summary['n_failures']} failures; {summary['seconds']:.0f}s", flush=True)
    return summary


def cmd_run(args):
    status = prereg_status()
    if status != "frozen":
        print(f"refused: {os.path.relpath(PLAN, ROOT)} says the preregistration is {status}. No run on real "
              "embeddings comes before freezing; the alignment check is --check-alignment.", flush=True)
        return 2
    summary = run_all(resolve_cache(args.cache), args.out_dir or OUT)
    return 0 if summary["prereg"]["complete"] else 1


# ----------------------------------------------------------------------------- the fitting code on synthetic tensors
FAKE_LARGE_LR = 0.05


def fake_hub_cache(root, seed=0):
    """A Hub cache in the layout hf_hub_download and try_to_load_from_cache read offline, holding synthetic files
    for every file the design names (both encoders, three families, three inputs, both splits) and the per-size
    settings file Large's probe rates are read from. The three inputs of a family share their labels, as on the Hub."""
    import torch
    import exp74_suite_encoders as e74
    rng = np.random.default_rng(seed)
    rev = "0" * 40
    repo = os.path.join(root, "datasets--" + HUB.replace("/", "--"))
    snap = os.path.join(repo, "snapshots", rev)
    os.makedirs(os.path.join(repo, "refs"), exist_ok=True)
    with open(os.path.join(repo, "refs", "main"), "w") as f:
        f.write(rev)

    def put(rel, obj):
        path = os.path.join(snap, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if rel.endswith(".json"):
            with open(path, "w") as f:
                json.dump(obj, f)
        else:
            torch.save(obj, path)

    put(e74.SETTINGS_BY_SIZE[LARGE], {"large_run": {FAMILIES[PRIMARY]["full"]: {"settings": {"probe_lr": FAKE_LARGE_LR}}}})
    h, pp = 8, 4
    for model, D in ((BASE, 16), (LARGE, 24)):
        for fam in FAMILIES:
            for split, n in (("train", 12), ("test", 10)):
                if fam == PRIMARY:
                    lab = np.repeat(np.repeat(rng.integers(0, 4, (n, h, h)), pp, 1), pp, 2)
                    lab[:, :6, :] = -1                               # a row of unlabelled windows, one half-labelled
                    shape = (n, h, h, D)
                else:
                    n = {"train": 200, "test": 150}[split] if fam == REPLICATION else {"train": 90, "test": 60}[split]
                    lab, shape = rng.integers(0, 3, n), (n, D)
                for c in CONDITIONS:
                    emb = torch.tensor(rng.standard_normal(shape), dtype=torch.float32)
                    put(f"{model}/{FAMILIES[fam][c]}/{split}.pt", {"embeddings": emb, "labels": torch.tensor(lab)})
    return root


def smoke_torch(args):
    """The real modes on synthetic tensors, two epochs. First fit_family and evaluate_family through the real probe
    code (exp54, exp73, exp74, exp75), with the segmentation units checked against exp54.window_quantities. Then the
    alignment check and the full run end to end on a synthetic Hub cache read offline (run_all, bypassing only the
    freeze guard, which tests/test_exp88.py checks): both encoders, Large's probe rate read from the cache, every
    grade decided, and a permuted input caught by the alignment comparison and refused by the loader."""
    import tempfile
    import torch
    import exp51_their_probe as e51
    import exp54_multiclass_embeddings as e54
    e51.EPOCHS, e54.EPOCHS = 2, 2
    rng = np.random.default_rng(0)
    D, C, pp, h = 16, 4, 4, 8

    def lab(n):
        return torch.tensor(np.repeat(np.repeat(rng.integers(0, C, (n, h, h)), pp, 1), pp, 2), dtype=torch.int64)

    def mk(*shape):
        return torch.tensor(rng.standard_normal(shape), dtype=torch.float32)

    ytr, yte = lab(12), lab(10)
    yte[:, :6, :] = -1
    xtr = mk(12, h, h, D)
    L = {"task": FAMILIES["pastis"]["full"], "family": "segmentation", "xtr": xtr, "ytr": ytr, "yte": yte, "C": C,
         "patch_px": pp, "emb_tr": xtr.numpy().reshape(-1, D), "tests": {c: (mk(10, h, h, D), yte) for c in CONDITIONS}}
    views = {}
    for seed in (0, 1):
        inputs = fit_family(L, seed, views)
        meas, units = evaluate_family(inputs)
        probe = e54.train_probe(xtr, ytr, pp, C, e54.TASK_LR.get(L["task"], 0.1), seed=seed)
        w = e54.window_quantities(probe, L["tests"]["full"][0], yte, pp, C)
        ok = w["ok"].reshape(-1)
        assert units["full"]["margin"].size == int(ok.sum()) == meas["n_units"]
        assert np.array_equal(units["full"]["dec"], w["dec"].reshape(-1)[ok]), "decisions differ from exp54's windows"
        assert np.allclose(units["full"]["margin"], w["margin"].reshape(-1)[ok], atol=1e-5), "margins differ from exp54's"
        assert np.array_equal(units["full"]["err"], w["err"].reshape(-1)[ok]), "errors differ from exp54's"
        assert np.array_equal(units["tile"], np.repeat(np.arange(10), h * h)[ok])
        assert inputs["conditions"]["optical_missing"]["emb_te"].shape == (int(ok.sum()), D)
        mixed, mm = evaluate_mixed(units, inputs["n_tiles"])
        assert mm["cloudy_tiles"].size == 5 and mixed["n_windows"] == int(ok.sum())
    # classification: three inputs of one set of samples
    ytr_c, yte_c = torch.tensor(rng.integers(0, 2, 300)), torch.tensor(rng.integers(0, 2, 120))
    xtr_c = mk(300, D)
    Lc = {"task": FAMILIES["china6"]["full"], "family": "classification", "xtr": xtr_c, "ytr": ytr_c, "yte": yte_c,
          "C": 2, "patch_px": 1, "emb_tr": xtr_c.numpy(), "tests": {c: (mk(120, D), yte_c) for c in CONDITIONS}}
    inputs = fit_family(Lc, 0, {})
    meas, units = evaluate_family(inputs)
    assert meas["n_units"] == 120 and units["tile"] is None and inputs["conditions"]["full"]["p"].shape == (120, 2)
    S = {"families": {"pastis": evaluate_family(fit_family(L, 0, views))[0], "china6": meas}, "mixed_map": mixed}
    r = read_predictions(S)
    assert all(r[p]["holds"] is not None for p in ("P1", "P2", "P3", "P4")), r

    # the real modes end to end on a synthetic cache, read offline as the cluster job reads the real one
    from huggingface_hub import constants as hf_constants
    os.environ["HF_HUB_OFFLINE"] = "1"
    hf_constants.HF_HUB_OFFLINE = True
    with tempfile.TemporaryDirectory() as tmp:
        cache, out_dir = fake_hub_cache(os.path.join(tmp, "hub")), os.path.join(tmp, "out")
        assert resolve_cache(tmp) == cache and resolve_cache(cache) == cache, "HF_HOME or its hub/ both resolve"
        assert cmd_check_alignment(argparse.Namespace(cache=cache, out_dir=out_dir)) == 0
        al = json.load(open(os.path.join(out_dir, "exp88_alignment.json")))
        assert al["verdict"] == {"base_aligned_for_the_design": True, "large_runs_as_replication": True,
                                 "large_cached_for_all_three_inputs": True}, al["verdict"]
        assert al["encoders"][BASE]["families"][PRIMARY]["inputs"]["optical_missing"]["test"]["embeddings"]["shape"] == [10, 8, 8, 16]
        assert "class_counts" not in json.dumps(al), "the alignment check tabulates no label"
        held = {(c, "test"): describe_file(cached_path(BASE, FAMILIES[PRIMARY][c], "test", cache))[1] for c in CONDITIONS}
        held[("optical_missing", "test")]["labels"] = held[("optical_missing", "test")]["labels"][[1, 0] + list(range(2, 10))]
        bad = compare_split(held, "test")
        assert bad["aligned"] is False and bad["against_full"]["optical_missing"]["labels_equal_in_order"] is False
        assert bad["against_full"]["radar_missing"]["aligned"] is True

        summary = run_all(cache, out_dir, seeds={BASE: (0, 1), LARGE: (0,)})
        g = summary["prereg"]
        assert not summary["failures"], summary["failures"]
        assert g["complete"] and g["graded_on"] == f"{BASE}, probe seed 0", g
        assert summary["large"]["runs"] and set(summary["replication"]) == {BASE, LARGE}
        assert set(summary["replication"][BASE]) == {"0", "1"} and set(summary["replication"][LARGE]) == {"0"}
        assert summary["results"][LARGE]["probe_lrs"][FAMILIES[PRIMARY]["full"]] == FAKE_LARGE_LR, "Ai2's per-size rate"
        assert summary["results"][BASE]["probe_lrs"][FAMILIES[PRIMARY]["full"]] == 0.01, "exp54's table for Base"
        fam0 = summary["results"][BASE]["seeds"]["0"]["families"]
        assert set(fam0) == set(FAMILIES) and fam0[PRIMARY]["n_units"] == 10 * 7 * 8
        assert summary["units_npz"]["written"] and os.path.exists(os.path.join(out_dir, "exp88_summary.json"))
        assert not os.path.exists(os.path.join(out_dir, "exp88_summary.partial.json"))
        with np.load(os.path.join(out_dir, "exp88_units.npz")) as z:
            assert z[f"{BASE}/{PRIMARY}/seed1/optical_missing/err"].size == 560
            assert f"{LARGE}/{PRIMARY}/seed0/full/p1" in z.files
        # a permuted input is refused before any probe reads it
        path = cached_path(BASE, FAMILIES["togo12"]["optical_missing"], "test", cache)
        d = torch.load(path, weights_only=True)
        torch.save({"embeddings": d["embeddings"], "labels": d["labels"].flip(0)}, path)
        try:
            load_family(BASE, "togo12", cache)
        except RuntimeError as ex:
            assert "do not line up" in str(ex)
        else:
            raise AssertionError("a permuted input must be refused")
    print("smoke-torch OK: the S1+S2 probe read on three inputs through exp54/exp73/exp74/exp75's code; windows match "
          "exp54.window_quantities; on a synthetic Hub cache read offline the alignment check passes an aligned cache "
          "and catches a permuted input, the full run grades P1-P4 on both encoders, and the loader refuses a "
          "permuted input", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="numpy only: synthetic embeddings, both synthetic cases")
    ap.add_argument("--smoke-torch", action="store_true", help="the real probe code on synthetic tensors")
    ap.add_argument("--check-alignment", action="store_true", help="identifiers, shapes and split sizes only")
    ap.add_argument("--cache", default=None, help="the Hub cache directory (default: resolved from HF_HOME)")
    ap.add_argument("--out-dir", default=None, help="where the outputs go (default: exp/out)")
    args = ap.parse_args(argv)
    if args.smoke:
        smoke(args.out_dir)
        return 0
    if args.smoke_torch:
        return smoke_torch(args)
    if args.check_alignment:
        return cmd_check_alignment(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
