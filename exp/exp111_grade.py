"""exp111's grader: P1, P2 and P3 per model at the floors, and the report-only readings, from the units files that
exp111_blind_transfer.py --units wrote. docs/plan/blind_transfer.md is the specification and this file its arithmetic;
it is committed with its smoke before the page is frozen, and the blind predict refuses if it changed since the gate.

    python exp/exp111_grade.py --units-dir DIR [--out-dir DIR]   # refused unless the page is frozen, the units record's
                                                                 # draw is the page's draw and the units files are the
                                                                 # ones the record names (sha256)
    python exp/exp111_grade.py --smoke [--out-dir DIR]           # synthetic units with known answers, and the known
                                                                 # answer on exp99's and exp103's committed units

Inputs: exp111_units_v1.npz and exp111_units_v12.npz (exp99's units schema plus stratum, band, cgls_raw and
in_gate17_cell; one row per drawn plot, rows shuffled) and exp111_units_record.json (the draw's sha256, the pool's cell
counts, each country's area outside exp99's region, the drawn plots per cell). Nothing here holds an id or a position.

The quantities (the page's "Predictions"), per model, at the plots with input:
  weights     region: (A'_c / N_c) x (N_cg / m_cg), normalised, recomputed here and checked against the units' weight
              column; standardised to the 271 (P2): label group g gets exp99's share (O 97, W 129, X 45 of 271), split
              within the group as the region weights split it
  suspicion   -log p1 (checked against p1); "confident" means p1 >= CONFIDENT_CUT (exp109's cut at the 271)
  errors      wrong_strict, wrong_lenient (exp99's crosswalk, rechecked here); STRICT-only = wrong under STRICT, right
              under LENIENT
  intervals   2,000 resamples of the plots within each country x label-group cell (seed 111; the same resamples for
              both models), 90% percentile intervals; no correction for multiple predictions
  P1(a)       holds when the region-weighted STRICT AUROC is below 0.70 and its 90% interval lies above 0.50
  P1(b)       holds when the 90% interval of (LENIENT AUROC - STRICT AUROC) lies above 0
  LENIENT     a call, not a bar: "above" when LENIENT's 90% interval lies wholly above 0.70, "below" when wholly
              below, else "not determined"
  P2(a)       holds when, of the confident STRICT errors (standardised weights), the share that is shrubland_savanna on
              TimeSync Open Grassland is at least 0.50
  P2(b)       holds when AUROC(LENIENT errors vs STRICT-correct) - AUROC(STRICT-only errors vs STRICT-correct) has a
              90% interval above 0, AUROC(STRICT-only vs STRICT-correct) is at most 0.65 and AUROC(LENIENT errors vs
              STRICT-correct) is at least 0.65 (standardised weights, point values)
  P3          holds when P(note) >= 0.5: e_S and e_L the region-weighted STRICT and LENIENT error rates of
              shrubland_savanna's confident zone, P(note) = P(K_S >= k* and K_L < k*) with (K_L, K_S - K_L)
              multinomial(20; e_L, e_S - e_L) and k* the fewest errors in 20 that the package's audit calls "above"
              alpha = 0.05 (hypergeom_interval at N = 10^6, interval_verdict): k* = 4
  floors      every prediction: at least 720 plots with input (80% of the draw's 900, exp111_blind_transfer.SIZES: the
              grade refuses a units record of other sizes); P1 and P2: at least 40 errors and 40 correct plots
              under each rule; P2: at least 25 confident STRICT errors and 30 STRICT-only errors; P3: at least 30
              plots in the zone. Below a floor the value is reported without a verdict.
Report-only: unweighted values, capture at 10% and exp99's gap closed, the region-weighted P2 values, the reverse
pair, the fresh plots' own confident share and own top third, P3's components and P(note) at 15 and 30 labels, the
package's literal audit run (seed 111), R1 (the CGLS-disagreement order), R2 (distance bands), R4 (the region's error
rate with the missing-input envelope), R5 (inventory, conditions, the models' agreement), and every point value again
without the plots that lie in a gate17 cell.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import warnings

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
for _p in (ROOT, EXP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import exp111_blind_transfer as b111         # noqa: E402  (CONFIDENT_CUT, the page, the region weights)
import exp99_transfer as e99                 # noqa: E402  (exp99's ranking block: capture and gap closed)
import timesync_awf_crosswalk as tsx         # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
MODELS = ("v1", "v12")
CUT = b111.CONFIDENT_CUT
STRATA = b111.STRATA
COUNTRIES = b111.COUNTRIES
SHRUB, GRASS = tsx.AWF_INDEX["shrubland_savanna"], tsx.AWF_INDEX["grassland_barren"]
OPEN_G, WOODED_G = tsx.TIMESYNC_CLASSES.index("Open Grassland"), tsx.TIMESYNC_CLASSES.index("Wooded Grassland")

# the page's bars and floors (PROPOSED until the owner confirms them; binding at the phase-1 commit; the page is the
# specification). The 720 is 80% of the draw's plots (b111.SIZES, 300/300/300; exp99's 250 of 309): the smoke checks that
# it is at most 80% of them, and check_inputs refuses a units record drawn at other sizes
FLOORS = {"with_input": 720, "errors_per_rule": 40, "correct_per_rule": 40, "confident_strict_errors": 25,
          "strict_only_errors": 30, "zone_plots": 30}
BARS = {"P1a_strict_below": 0.70, "P1a_lower_above": 0.50, "P1b_lower_above": 0.0, "lenient_call": 0.70,
        "P2a_pair_share_min": 0.50, "P2b_gap_lower_above": 0.0, "P2b_strict_only_max": 0.65,
        "P2b_lenient_min": 0.65, "P3_p_note_min": 0.5}
P3 = {"labels": 20, "alpha": 0.05, "conf": 0.95, "N": 10 ** 6, "labels_reported": (15, 30)}
BOOT = {"R": 2000, "seed": 111, "level": 0.90}
# exp99's 271 plots with input, by label group (exp/out/exp99_units.npz; the smoke recounts them there): the
# standardised weights' shares are these plot counts' (0.358, 0.476, 0.166), not exp99's design-weighted group shares
# (0.389, 0.446, 0.166), which exp109's seen values used
STD_COUNTS = {"O": 97, "W": 129, "X": 45}

# The also-accept file of the literal audit run, fixed here before any label is read. Two TimeSync classes share a
# STRICT counterpart (Open Forest and Wooded Grassland: 2; Otherland and Open Grassland: 4), so the reference is coded
# as its STRICT counterpart for seven classes, Open Forest as 10 and Otherland as 11 (EXTENDED_CODE); with this file
# the also-accept reading equals LENIENT exactly (checked in the smoke for every label and map class). The STRICT
# reading is a second call with the reference coded as its STRICT counterpart (STRICT_CODE).
STRICT_CODE = {t: tsx.AWF_INDEX[tsx.STRICT[t]] for t in tsx.TIMESYNC_CLASSES}
EXTENDED_CODE = dict(STRICT_CODE, **{"Open Forest": 10, "Otherland": 11})
ALSO_ACCEPT = {0: [10], 1: [3, 11], 2: [4, 10], 3: [1], 4: [2, 11], 5: [8], 6: [0, 10], 7: [0, 10]}

# R1's crosswalk from the AWF class to the CGLS-LC100 raw codes it accepts (exp109's Arm W, fixed on the page)
_FOREST = tuple(range(111, 117)) + tuple(range(121, 127))
ARM_W = {0: _FOREST, 6: _FOREST, 7: _FOREST, 1: (80, 90, 200), 3: (80, 90, 200),
         2: (20, 30, 100) + tuple(range(121, 127)), 4: (20, 30, 60, 100), 5: (40, 50), 8: (50,), 9: ()}

# the known answer at the 271 (exp109's summary, exp99's weights): the grader's arithmetic on the committed units files
KNOWN_UNITS = {"v1": os.path.join(OUT, "exp99_units.npz"),
               "v12": os.path.join(OUT, "exp103", "transfer2017", "exp99_units.npz")}
KNOWN_SUMMARY = os.path.join(OUT, "exp109_summary.json")
# the numbers as the page prints them (rounded; compared as printed)
KNOWN_PAGE = {"v1": {"strict": "0.638", "lenient": "0.678", "strict_only": "0.601", "lenient_vs_correct": "0.698",
                     "zone_eS": "0.26", "zone_eL": "0.04", "p_note_20": "0.80",
                     "confident_strict_errors": 20, "pair": 11, "confident": 91},
              "v12": {"strict": "0.642", "lenient": "0.730", "strict_only": "0.561", "lenient_vs_correct": "0.741",
                      "zone_eS": "0.33", "zone_eL": "0.01", "p_note_20": "0.93",
                      "confident_strict_errors": 22, "pair": 19, "confident": 91}}


# ----------------------------------------------------------------------------- arithmetic
def wauc(u, e, w):
    """Design-weighted AUROC of the suspicion for the error indicator, ties half (the package's); NaN when degenerate."""
    from oe_inferencex import metrics
    u, e, w = np.asarray(u, np.float64), np.asarray(e, bool), np.asarray(w, np.float64)
    if not (w[e].sum() > 0 and w[~e].sum() > 0):
        return float("nan")
    return float(metrics.weighted_auroc(u, e.astype(np.float64), w))


def wmean(x, w):
    w = np.asarray(w, np.float64)
    return float((np.asarray(x, np.float64) * w).sum() / w.sum()) if w.sum() > 0 else float("nan")


def kstar(n, alpha=P3["alpha"], N=P3["N"], conf=P3["conf"]):
    """The fewest errors among n labels that the audit calls 'above' alpha (the package's interval and verdict)."""
    from oe_inferencex import estimate as est
    for k in range(n + 1):
        if est.interval_verdict(*est.hypergeom_interval(k, n, N, conf), alpha) == "above":
            return k
    return None


def p_note(eS, eL, n, ks):
    """P(K_S >= ks and K_L < ks): (K_L, K_S - K_L, rest) multinomial(n; eL, eS - eL, 1 - eS)."""
    if not (np.isfinite(eS) and np.isfinite(eL)) or ks is None:
        return float("nan")
    pL, pSO = eL, max(eS - eL, 0.0)
    pR = max(1.0 - pL - pSO, 0.0)
    tot = 0.0
    for kL in range(0, ks):
        for kSO in range(max(0, ks - kL), n - kL + 1):
            tot += math.comb(n, kL) * math.comb(n - kL, kSO) * pL ** kL * pSO ** kSO * pR ** (n - kL - kSO)
    return float(tot)


def binom_tail(n, p, k):
    """P(Binomial(n, p) >= k)."""
    return float(sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1)))


def std_weights(stratum, w_region):
    """Standardised to the 271's label mix: group g gets STD_COUNTS[g] / 271, split within the group as the region
    weights split it. stratum: 0, 1, 2 for O, W, X, at the plots with input."""
    tot = sum(STD_COUNTS.values())
    w = np.zeros(len(stratum))
    for g, name in enumerate(STRATA):
        m = stratum == g
        if m.any() and w_region[m].sum() > 0:
            w[m] = STD_COUNTS[name] / tot * w_region[m] / w_region[m].sum()
    return w


def statistics(d, ix=None, full=True):
    """One model's quantities at rows ix of its plots-with-input arrays (all when None)."""
    i = np.arange(d["u"].size) if ix is None else ix
    u, p1, pred, lab = d["u"][i], d["p1"][i], d["pred"][i], d["label"][i]
    es, el, wr, ws = d["es"][i], d["el"][i], d["w_r"][i], d["w_s"][i]
    so = es & ~el
    a, b = el | ~es, ~el                         # LENIENT errors or STRICT-correct; STRICT-only errors or STRICT-correct
    conf = p1 >= d["cut"]
    ce = conf & es
    pair = (pred == SHRUB) & (lab == OPEN_G)
    zone = conf & (pred == SHRUB)
    out = {"auc_strict": wauc(u, es, wr), "auc_lenient": wauc(u, el, wr),
           "auc_lenient_vs_correct_std": wauc(u[a], el[a], ws[a]),
           "auc_strict_only_vs_correct_std": wauc(u[b], es[b], ws[b]),
           "pair_share_std": float(ws[ce & pair].sum() / ws[ce].sum()) if ws[ce].sum() > 0 else float("nan"),
           "zone_eS": wmean(es[zone], wr[zone]) if zone.any() else float("nan"),
           "zone_eL": wmean(el[zone], wr[zone]) if zone.any() else float("nan")}
    out["diff_lenient_minus_strict"] = out["auc_lenient"] - out["auc_strict"]
    out["gap_std"] = out["auc_lenient_vs_correct_std"] - out["auc_strict_only_vs_correct_std"]
    if not full:
        return out
    rev = (pred == GRASS) & (lab == WOODED_G)
    out.update({
        "n": int(u.size), "errors_strict": int(es.sum()), "errors_lenient": int(el.sum()),
        "correct_strict": int((~es).sum()), "correct_lenient": int((~el).sum()), "strict_only": int(so.sum()),
        "confident": int(conf.sum()), "confident_share": float(conf.mean()) if u.size else float("nan"),
        "confident_strict_errors": int(ce.sum()), "confident_strict_errors_pair": int((ce & pair).sum()),
        "zone_plots": int(zone.sum()),
        "auc_strict_unweighted": wauc(u, es, np.ones(u.size)), "auc_lenient_unweighted": wauc(u, el, np.ones(u.size)),
        "reverse_pair_share_std": float(ws[ce & rev].sum() / ws[ce].sum()) if ws[ce].sum() > 0 else float("nan"),
        "strict_only_share_of_confident_strict_errors_std":
            float(ws[ce & so].sum() / ws[ce].sum()) if ws[ce].sum() > 0 else float("nan"),
        "region_weighted": {
            "pair_share": float(wr[ce & pair].sum() / wr[ce].sum()) if wr[ce].sum() > 0 else float("nan"),
            "auc_lenient_vs_correct": wauc(u[a], el[a], wr[a]),
            "auc_strict_only_vs_correct": wauc(u[b], es[b], wr[b])}})
    out["region_weighted"]["gap"] = (out["region_weighted"]["auc_lenient_vs_correct"]
                                     - out["region_weighted"]["auc_strict_only_vs_correct"])
    return out


BOOT_KEYS = ("auc_strict", "auc_lenient", "diff_lenient_minus_strict", "auc_lenient_vs_correct_std",
             "auc_strict_only_vs_correct_std", "gap_std", "pair_share_std", "zone_eS", "zone_eL")


def bootstrap(data, cells, R=BOOT["R"], seed=BOOT["seed"], level=BOOT["level"]):
    """Resample the plots with input within each country x label-group cell, the same resamples for every model;
    the percentile interval of each quantity in BOOT_KEYS, per model."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(cells == c) for c in np.unique(cells)]
    draws = {m: [] for m in data}
    for _ in range(R):
        ix = np.concatenate([rng.choice(g, g.size, replace=True) for g in groups])
        for m, d in data.items():
            s = statistics(d, ix, full=False)
            draws[m].append([s[k] for k in BOOT_KEYS])
    q = ((1 - level) / 2, 1 - (1 - level) / 2)
    out = {}
    for m, v in draws.items():
        v = np.asarray(v, np.float64)
        with warnings.catch_warnings():               # a quantity undefined in every resample (below a floor) is NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            out[m] = {k: [float(np.nanquantile(v[:, j], q[0])), float(np.nanquantile(v[:, j], q[1]))]
                      for j, k in enumerate(BOOT_KEYS)}
        out[m]["resamples"] = int(R)
        out[m]["resamples_undefined"] = {k: int(np.isnan(v[:, j]).sum()) for j, k in enumerate(BOOT_KEYS)}
    return out


def verdict(holds, floors_ok):
    return "no verdict (below a floor)" if not floors_ok else ("holds" if holds else "does not hold")


def grade_model(s, iv):
    """The verdicts of one model from its point statistics s and intervals iv, exactly as the page states them."""
    f_in = s["n"] >= FLOORS["with_input"]
    f_err = min(s["errors_strict"], s["errors_lenient"]) >= FLOORS["errors_per_rule"] and \
        min(s["correct_strict"], s["correct_lenient"]) >= FLOORS["correct_per_rule"]
    f_p2 = s["confident_strict_errors"] >= FLOORS["confident_strict_errors"] and \
        s["strict_only"] >= FLOORS["strict_only_errors"]
    f_p3 = s["zone_plots"] >= FLOORS["zone_plots"]
    floors = {"with_input": f_in, "errors_and_correct_per_rule": f_err, "P2_counts": f_p2, "P3_zone": f_p3}
    aS, loS = s["auc_strict"], iv["auc_strict"][0]
    p1a_upper = aS < BARS["P1a_strict_below"]
    p1a_lower = loS > BARS["P1a_lower_above"]
    p1b = iv["diff_lenient_minus_strict"][0] > BARS["P1b_lower_above"]
    loL, hiL = iv["auc_lenient"]
    call = ("above" if loL > BARS["lenient_call"] else "below" if hiL < BARS["lenient_call"] else "not determined")
    p2a = s["pair_share_std"] >= BARS["P2a_pair_share_min"]
    p2b_gap = iv["gap_std"][0] > BARS["P2b_gap_lower_above"]
    p2b_so = s["auc_strict_only_vs_correct_std"] <= BARS["P2b_strict_only_max"]
    p2b_len = s["auc_lenient_vs_correct_std"] >= BARS["P2b_lenient_min"]
    ks = kstar(P3["labels"])
    pn = p_note(s["zone_eS"], s["zone_eL"], P3["labels"], ks)
    p3 = bool(np.isfinite(pn) and pn >= BARS["P3_p_note_min"])
    ok1 = f_in and f_err
    ok2 = ok1 and f_p2
    ok3 = f_in and f_p3
    return {
        "floors": floors,
        "P1a": {"verdict": verdict(p1a_upper and p1a_lower, ok1), "strict_below_0.70": bool(p1a_upper),
                "interval_above_0.50": bool(p1a_lower), "value": aS, "interval90": iv["auc_strict"]},
        "P1b": {"verdict": verdict(p1b, ok1), "value": s["diff_lenient_minus_strict"],
                "interval90": iv["diff_lenient_minus_strict"]},
        "P1": {"verdict": verdict(p1a_upper and p1a_lower and p1b, ok1)},
        "lenient_call": {"call": call if ok1 else "no call (below a floor)", "value": s["auc_lenient"],
                         "interval90": iv["auc_lenient"]},
        "P2a": {"verdict": verdict(p2a, ok2), "value": s["pair_share_std"], "interval90": iv["pair_share_std"]},
        "P2b": {"verdict": verdict(p2b_gap and p2b_so and p2b_len, ok2), "gap_interval_above_0": bool(p2b_gap),
                "strict_only_at_most_0.65": bool(p2b_so), "lenient_at_least_0.65": bool(p2b_len),
                "gap": s["gap_std"], "gap_interval90": iv["gap_std"],
                "auc_strict_only_vs_correct": s["auc_strict_only_vs_correct_std"],
                "auc_strict_only_vs_correct_interval90": iv["auc_strict_only_vs_correct_std"],
                "auc_lenient_vs_correct": s["auc_lenient_vs_correct_std"],
                "auc_lenient_vs_correct_interval90": iv["auc_lenient_vs_correct_std"]},
        "P2": {"verdict": verdict(p2a and p2b_gap and p2b_so and p2b_len, ok2)},
        "P3": {"verdict": verdict(p3, ok3), "p_note": pn, "k_star": ks, "labels": P3["labels"], "alpha": P3["alpha"],
               "zone_eS": s["zone_eS"], "zone_eL": s["zone_eL"], "zone_eS_interval90": iv["zone_eS"],
               "zone_eL_interval90": iv["zone_eL"], "zone_plots": s["zone_plots"]}}


def p3_report(s):
    out = {}
    for n in (P3["labels"],) + tuple(P3["labels_reported"]):
        ks = kstar(n)
        out[f"labels_{n}"] = {"k_star": ks, "p_note": p_note(s["zone_eS"], s["zone_eL"], n, ks),
                              "p_strict_above": binom_tail(n, s["zone_eS"], ks) if np.isfinite(s["zone_eS"]) else None,
                              "p_lenient_above": binom_tail(n, s["zone_eL"], ks) if np.isfinite(s["zone_eL"]) else None}
    return out


# ----------------------------------------------------------------------------- the inputs
def load_units(units_dir):
    rec = b111.read_json(os.path.join(units_dir, "exp111_units_record.json"))
    units = {m: {k: v for k, v in np.load(os.path.join(units_dir, f"exp111_units_{m}.npz")).items()} for m in MODELS}
    return units, rec


def check_inputs(units_dir, text=None, sizes=None):
    """The page is frozen, the units record's draw is the page's draw, drawn at this commit's sizes (b111.SIZES, which
    the floors are fixed for), and the units files are the ones the record names. PermissionError otherwise (the grade
    refuses)."""
    status = b111.prereg_status(text)
    if status != "frozen":
        raise PermissionError(f"refused: docs/plan/blind_transfer.md is {status}, not frozen")
    rec = b111.read_json(os.path.join(units_dir, "exp111_units_record.json"))
    sz = rec.get("sizes_O_W_X") or []
    if len(sz) != len(STRATA):
        raise PermissionError("refused: the units record holds no draw sizes")
    b111.check_sizes(dict(zip(STRATA, sz)), rec.get("n_plots"), sizes or b111.SIZES)
    want = b111.page_draw_sha(text)
    if want is None or rec.get("draw_sha256") != want:
        raise PermissionError("refused: the units record's draw is not the draw the frozen page holds")
    for m in MODELS:
        name = f"exp111_units_{m}.npz"
        got = sha256_file(os.path.join(units_dir, name))
        if got != (rec.get("sha256") or {}).get(name):
            raise PermissionError(f"refused: {name} is not the file the units record names (sha256)")
    return rec


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepared(units, rec, exclude=None):
    """Per model, the arrays at the plots with input (minus `exclude`), the weights recomputed and checked."""
    a = units[MODELS[0]]
    for k in ("country", "label", "stratum", "covered", "weight", "band", "in_gate17_cell"):
        if any(not np.array_equal(a[k], units[m][k]) for m in MODELS[1:]):
            raise ValueError(f"the models' units files differ in {k!r}: not one shuffled order")
    if [str(x) for x in a["label_names"]] != list(tsx.TIMESYNC_CLASSES) or \
            [str(x) for x in a["stratum_names"]] != list(STRATA) or [str(x) for x in a["country_names"]] != list(COUNTRIES):
        raise ValueError("the units' label, stratum or country names are not exp111's")
    country = np.array([COUNTRIES[i] for i in a["country"]])
    stratum = np.array([STRATA[i] for i in a["stratum"]])
    cov = a["covered"].astype(bool)
    w_full = b111.region_weights(country, stratum, cov, rec["area_outside_region_km2"], rec["pool_counts"])
    if not np.allclose(w_full, a["weight"], rtol=0, atol=1e-12):
        raise ValueError("the units' weight column is not the region weight of the page")
    keep = cov if exclude is None else cov & ~np.asarray(exclude, bool)
    w_r = b111.region_weights(country, stratum, keep, rec["area_outside_region_km2"], rec["pool_counts"])
    K = np.flatnonzero(keep)
    labels = np.array([tsx.TIMESYNC_CLASSES[i] for i in a["label"][K]])
    data = {}
    for m in MODELS:
        d = units[m]
        pred = d["pred"][K].astype(np.int64)
        p1 = d["p1"][K].astype(np.float64)
        u = d["suspicion"][K].astype(np.float64)
        if not np.allclose(u, -np.log(p1), rtol=0, atol=1e-9):
            raise ValueError(f"{m}: the suspicion is not -log p1")
        es, el = d["wrong_strict"][K] > 0, d["wrong_lenient"][K] > 0
        if not (np.array_equal(es, tsx.wrong(labels, pred, "strict") > 0)
                and np.array_equal(el, tsx.wrong(labels, pred, "lenient") > 0)):
            raise ValueError(f"{m}: the units' errors are not the crosswalk's")
        data[m] = {"u": u, "p1": p1, "pred": pred, "label": a["label"][K].astype(np.int64), "es": es, "el": el,
                   "w_r": w_r[K], "w_s": std_weights(a["stratum"][K], w_r[K]), "cut": CUT[m],
                   "condition": d["condition"][K], "cgls": a["cgls_raw"][K].astype(np.int64),
                   "band": a["band"][K].astype(np.int64)}
    cells = a["country"][K].astype(np.int64) * len(STRATA) + a["stratum"][K].astype(np.int64)
    common = {"K": K, "keep": keep, "cells": cells, "country": country, "stratum": stratum, "covered": cov}
    return data, common


# ----------------------------------------------------------------------------- report-only readings
def literal_audit(d, labels_names, seed=BOOT["seed"]):
    """The package's audit run once on the fresh plots (unweighted; a demonstration, not the grade): sample 20 per
    predicted class from the confident zone (p1 >= the cut), read the STRICT-coded reference, then the extended-coded
    reference with ALSO_ACCEPT; the note would fire for a class above alpha under the first and not under the second."""
    from oe_inferencex import estimate as est
    lab = [labels_names[i] for i in d["label"]]
    try:
        smp = est.sample_class_audit(d["p1"], d["pred"], P3["labels"], cut=d["cut"], seed=seed)
    except ValueError as ex:
        return {"not_run": str(ex)}
    idx = np.asarray(smp["indices"])
    ref_s = np.array([STRICT_CODE[lab[i]] for i in idx])
    ref_e = np.array([EXTENDED_CODE[lab[i]] for i in idx])
    r1 = est.class_audit_from_sample(smp, ref_s, alpha=P3["alpha"])
    r2 = est.class_audit_from_sample(smp, ref_e, also_accept=ALSO_ACCEPT, alpha=P3["alpha"])
    per = {}
    for c, row in r1["per_class"].items():
        alt = r2["per_class"][c]["also_accept"]
        per[c] = {"class": tsx.AWF_CLASSES.get(int(c), "channel 9"), "n_zone": row["n_zone"],
                  "n_labelled": row["n_labelled"], "strict_wrong": row["n_wrong"], "strict_verdict": row["verdict"],
                  "also_accept_wrong": alt["n_wrong"], "also_accept_verdict": alt["verdict"],
                  "note_fires": row["verdict"] == "above" and alt["verdict"] != "above",
                  "package_note": r2["per_class"][c].get("note")}
    mc = np.asarray(smp["map_class"])
    same = bool(np.array_equal(mc != ref_s, d["es"][idx]) and np.array_equal(
        np.array([not (mc[k] == ref_e[k] or ref_e[k] in ALSO_ACCEPT.get(int(mc[k]), [])) for k in range(idx.size)]),
        d["el"][idx]))
    return {"seed": seed, "labels_per_class": P3["labels"], "zone_plots": smp["zone"]["n_zone"], "per_class": per,
            "readings_equal_the_crosswalk": same,
            "note": "unweighted on the fresh plots, so Open Grassland is over-represented: a demonstration, not the grade"}


def ranking_report(d, cells):
    out = {}
    for rule, e in (("strict", d["es"]), ("lenient", d["el"])):
        r = e99.ranking(d["u"], e.astype(np.float64), d["w_r"], cells, 0)
        out[rule] = {k: r[k] for k in ("weighted", "unweighted") if k in r}
    return out


def r1_cgls(d):
    """R1: plots whose class does not accept CGLS's raw 2017 class first, ties by suspicion; the AUROC gain over the
    suspicion, region-weighted, under each rule. A plot with no CGLS value is not flagged."""
    flag = np.array([c >= 0 and c not in ARM_W.get(int(p), ()) for c, p in zip(d["cgls"], d["pred"])])
    span = float(np.nanmax(d["u"]) - np.nanmin(d["u"]) + 1.0) if d["u"].size else 1.0
    score = flag * span + d["u"]
    out = {"flagged": int(flag.sum()), "without_cgls": int((d["cgls"] < 0).sum())}
    for rule, e in (("strict", d["es"]), ("lenient", d["el"])):
        a_u, a_o = wauc(d["u"], e, d["w_r"]), wauc(score, e, d["w_r"])
        out[rule] = {"auroc_suspicion": a_u, "auroc_order": a_o, "gain": a_o - a_u}
    return out


def r2_bands(d, cuts):
    out = {"band_cuts_km": cuts}
    for b in range(len(cuts) + 1):
        m = d["band"] == b
        row = {"plots": int(m.sum())}
        for rule, e in (("strict", d["es"]), ("lenient", d["el"])):
            row[rule] = {"error_rate": wmean(e[m], d["w_r"][m]) if m.any() else None,
                         "auroc": wauc(d["u"][m], e[m], d["w_r"][m]) if m.any() else None}
        out[f"band_{b}"] = row
    return out


def r2_seen(known=KNOWN_UNITS):
    """The seen bands from the committed units: exp99's plots inside Ai2's geometry and the rest within 100 km."""
    out = {}
    for m, p in known.items():
        if not os.path.exists(p):
            return {"note": "the committed units files are not here"}
        z = np.load(p)
        K = z["covered"].astype(bool)
        ins = z["inside_awf"].astype(bool)
        row = {}
        for tag, msk in (("inside", K & ins), ("within_100_km", K & ~ins)):
            row[tag] = {"plots": int(msk.sum())}
            for rule in ("strict", "lenient"):
                e = z[f"wrong_{rule}"][msk] > 0
                row[tag][rule] = {"error_rate": wmean(e, z["weight"][msk]),
                                  "auroc": wauc(z["suspicion"][msk], e, z["weight"][msk])}
        out[m] = row
    return out


def r4_region(units, rec, m):
    """R4: the region's error rate under each rule, the countries' label cells as strata (sizes in 10 m pixels: each
    country's area outside exp99's region, split by its pool's cell counts): the package's condition-design interval,
    and the envelope with every plot without input right, then wrong."""
    from oe_inferencex import estimate as est
    a = units[m]
    cov = a["covered"].astype(bool)
    out = {}
    N_c = {c: sum(v for k, v in rec["pool_counts"].items() if k.startswith(c + ":")) for c in COUNTRIES}
    area = rec["area_outside_region_km2"]
    tot_area = sum(area[c] for c in COUNTRIES)
    for rule in ("strict", "lenient"):
        e = a[f"wrong_{rule}"]
        k, n, N, W, drawn = [], [], [], [], []
        for ci, c in enumerate(COUNTRIES):
            for gi, g in enumerate(STRATA):
                msk = (a["country"] == ci) & (a["stratum"] == gi)
                k.append(int(np.nansum(e[msk & cov])))
                n.append(int((msk & cov).sum()))
                drawn.append(int(msk.sum()))
                share = rec["pool_counts"][f"{c}:{g}"] / N_c[c]
                N.append(max(int(round(area[c] * tsx.PIXELS_PER_KM2 * share)), n[-1]))
                W.append(area[c] / tot_area * share)
        theta, lo, hi, L = est._union_interval(k, n, N)
        low = sum(w * kk / dd for w, kk, dd in zip(W, k, drawn) if dd)
        high = sum(w * (kk + dd - nn) / dd for w, kk, nn, dd in zip(W, k, n, drawn) if dd)
        out[rule] = {"estimate": theta, "low": lo, "high": hi, "strata_in_interval": L,
                     "envelope_missing_right": low, "envelope_missing_wrong": high,
                     "method": "the condition design's interval of oe_inferencex.estimate over the six country x label "
                               "cells (exp99's Part A with cells as strata)"}
    return out


def r5_inventory(units, data):
    a = units[MODELS[0]]
    cov = a["covered"].astype(bool)
    by = {}
    for ci, c in enumerate(COUNTRIES):
        for gi, g in enumerate(STRATA):
            msk = (a["country"] == ci) & (a["stratum"] == gi)
            by[f"{c}:{g}"] = {"drawn": int(msk.sum()), "with_input": int((msk & cov).sum())}
    cond = {m: {str(int(v)): int(c) for v, c in zip(*np.unique(data[m]["condition"], return_counts=True))}
            for m in MODELS}
    agree = float((data["v1"]["pred"] == data["v12"]["pred"]).mean()) if data["v1"]["pred"].size else None
    return {"by_label_cell": by, "condition_codes": cond, "models_same_class_share": agree,
            "seen_at_the_271": 0.823}


def own_cut(d):
    """Report-only: the fresh plots' own top third (p1 at or above their own 2/3 quantile) in place of exp109's cut."""
    if not d["p1"].size:
        return None
    c = float(np.quantile(d["p1"], 2 / 3))
    s = statistics(dict(d, cut=c), None, full=True)
    ks = kstar(P3["labels"])
    return {"cut": c, "confident": s["confident"], "pair_share_std": s["pair_share_std"],
            "p_note": p_note(s["zone_eS"], s["zone_eL"], P3["labels"], ks)}


def most_strict_only_class(d):
    conf = d["p1"] >= d["cut"]
    so = d["es"] & ~d["el"]
    w = {int(c): float(d["w_r"][conf & so & (d["pred"] == c)].sum()) for c in np.unique(d["pred"])}
    if not w or max(w.values()) <= 0:
        return None
    c = max(w, key=w.get)
    return {"class": tsx.AWF_CLASSES.get(c, "channel 9"), "weighted_share": w[c] / sum(w.values())}


# ----------------------------------------------------------------------------- the grade
def grade(units, rec, R=BOOT["R"], literal=True, log=print):
    """Every prediction's verdict and the report-only readings, from units already checked (check_inputs)."""
    t0 = time.time()
    data, common = prepared(units, rec)
    iv = bootstrap(data, common["cells"], R)
    models = {}
    for m in MODELS:
        s = statistics(data[m])
        g = grade_model(s, iv[m])
        g["point"] = s
        g["intervals90"] = iv[m]
        g["report_only"] = {"ranking": ranking_report(data[m], common["cells"]), "P3": p3_report(s),
                            "own_top_third": own_cut(data[m]),
                            "class_with_most_confident_strict_only_errors": most_strict_only_class(data[m]),
                            "R1_cgls_order": r1_cgls(data[m]),
                            "R2_distance_bands": r2_bands(data[m], rec.get("band_cuts_km") or []),
                            "R4_region_error_rate": r4_region(units, rec, m)}
        if literal:
            g["report_only"]["literal_audit"] = literal_audit(data[m], list(tsx.TIMESYNC_CLASSES))
        models[m] = g
    # report-only: the same point values without the plots in a gate17 cell
    flag = units[MODELS[0]]["in_gate17_cell"].astype(bool)
    excl = {"plots_in_a_gate17_cell_with_input": int((flag & common["covered"]).sum())}
    if excl["plots_in_a_gate17_cell_with_input"]:
        dx, cx = prepared(units, rec, exclude=flag)
        for m in MODELS:
            s = statistics(dx[m])
            excl[m] = {k: s[k] for k in ("n", "auc_strict", "auc_lenient", "diff_lenient_minus_strict", "pair_share_std",
                                         "auc_lenient_vs_correct_std", "auc_strict_only_vs_correct_std", "zone_eS",
                                         "zone_eL")}
            excl[m]["p_note"] = p_note(s["zone_eS"], s["zone_eL"], P3["labels"], kstar(P3["labels"]))
    both = {}
    for p in ("P1a", "P1b", "P1", "P2a", "P2b", "P2", "P3"):
        vs = [models[m][p]["verdict"] for m in MODELS]
        both[p] = ("holds for both models" if all(v == "holds" for v in vs) else
                   "no verdict for a model (below a floor)" if any(v.startswith("no verdict") for v in vs) else
                   "does not hold for both models")
    # the graded verdicts are v1.2's (amendment of 16 October 2026); both models' are kept beside, report-only
    graded = {p: {m: models[m][p]["verdict"] for m in b111.GRADED_MODELS} for p in both}
    out = {"experiment": "exp111 grade: P1-P3 per model at the floors, the report-only readings",
           "draw_sha256": rec.get("draw_sha256"), "bars": BARS, "floors": FLOORS, "p3": P3, "bootstrap": BOOT,
           "confident_cut": CUT, "standardised_counts_O_W_X": [STD_COUNTS[g] for g in STRATA], "also_accept": {str(k): v for k, v in ALSO_ACCEPT.items()},
           "extended_code": EXTENDED_CODE, "arm_w": {str(k): list(v) for k, v in ARM_W.items()},
           "crosswalk_sha256": tsx.crosswalk_record()["sha256"], "graded_models": list(b111.GRADED_MODELS),
           "verdicts_graded": graded, "verdicts_both_models_report_only": both, "models": models,
           "report_only": {"without_plots_in_a_gate17_cell": excl, "R2_seen": r2_seen(),
                           "R5_inventory": r5_inventory(units, data)},
           "seconds": round(time.time() - t0, 1)}
    log(json.dumps({m: {p: models[m][p]["verdict"] for p in ("P1", "P2", "P3")} for m in MODELS}))
    return out


def write_summary(out, out_dir):
    import exp78_error_rate_estimation as e78
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "exp111_summary.json")
    obj = e78.jsonable(out)
    b111.jobs.check_record_keys(obj, "exp111_summary.json")     # the pack jobs' set (exp111_jobs.RECORD_FORBIDDEN_KEYS)
    b111.e98.check_no_coordinates(obj)
    with open(path + ".partial", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".partial", path)
    return path


# ----------------------------------------------------------------------------- the known answer
def known_answer(known=KNOWN_UNITS, summary=KNOWN_SUMMARY):
    """The grader's arithmetic on the committed units of exp99 (v1) and exp103 (v1.2) at the 271, with exp99's weights
    for every quantity (as exp109 computed them): exp109's AUROCs (to 1e-9 against its summary, and as the page prints
    them), its confident counts (91 plots, 20 and 22 confident STRICT errors, 11 and 19 of them the pair), and the
    shrub zone's rates and P(note) as the page prints them (0.26 / 0.04 and 0.80; 0.33 / 0.01 and 0.93)."""
    if not all(os.path.exists(p) for p in known.values()):
        return {"as_expected": None, "note": "the committed units files are not here"}
    ref = b111.read_json(summary)["models"] if os.path.exists(summary) else None
    out = {}
    ok = True
    for m, p in known.items():
        z = np.load(p)
        K = z["covered"].astype(bool)
        lab = np.array([str(z["label_names"][i]) for i in z["label"][K]])
        pred = z["pred"][K].astype(np.int64)
        d = {"u": z["suspicion"][K].astype(np.float64), "p1": z["p1"][K].astype(np.float64), "pred": pred,
             "label": z["label"][K].astype(np.int64), "es": tsx.wrong(lab, pred, "strict") > 0,
             "el": tsx.wrong(lab, pred, "lenient") > 0, "w_r": z["weight"][K].astype(np.float64),
             "w_s": z["weight"][K].astype(np.float64), "cut": CUT[m]}
        s = statistics(d)
        got = {"strict": s["auc_strict"], "lenient": s["auc_lenient"],
               "strict_only": s["auc_strict_only_vs_correct_std"], "lenient_vs_correct": s["auc_lenient_vs_correct_std"],
               "confident_strict_errors": s["confident_strict_errors"], "pair": s["confident_strict_errors_pair"],
               "confident": s["confident"], "zone_eS": s["zone_eS"], "zone_eL": s["zone_eL"],
               "p_note_20": p_note(s["zone_eS"], s["zone_eL"], 20, kstar(20))}
        page = KNOWN_PAGE[m]
        checks = {k: (f"{got[k]:.{len(v.split('.')[1])}f}" == v if isinstance(v, str) else got[k] == v)
                  for k, v in page.items()}
        if m == "v1":                    # STD_COUNTS: exp99's 271 plots with input, by label group
            grp = np.array([b111.stratum_of(str(z["label_names"][i])) for i in z["label"]])
            got["plots_with_input_by_group"] = {g: int((K & (grp == g)).sum()) for g in STRATA}
            checks["std_counts_are_exp99s"] = got["plots_with_input_by_group"] == STD_COUNTS
        if ref is not None:
            rm = ref["v1" if m == "v1" else "v1.2"]["transfer_2017"]
            a = rm["auroc_weighted"]
            for k, rk in (("strict", "strict"), ("lenient", "lenient"),
                          ("strict_only", "strict_only_errors_vs_strict_correct"),
                          ("lenient_vs_correct", "lenient_errors_vs_strict_correct")):
                checks[f"{k}_exp109_1e-9"] = abs(got[k] - a[rk]) <= 1e-9
            checks["cut_is_exp109s_rounded_down"] = 0 <= rm["confident_third"]["cut_p1"] - CUT[m] < 1e-11
        out[m] = {"got": got, "checks": checks}
        ok &= all(checks.values())
    out["as_expected"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- the synthetic smoke
SMOKE_POOL = {"kenya:O": 380, "kenya:W": 750, "kenya:X": 650, "tanzania:O": 360, "tanzania:W": 730, "tanzania:X": 820}
SMOKE_AREA = {"kenya": 60000.0, "tanzania": 47000.0}
# the base design: per label group, the shares of the plots with input that are STRICT-only errors and LENIENT errors;
# the suspicion's location per error type (standard normal quantiles around it; correct plots at 0)
BASE = {"cover": 0.85, "per_cell": 150, "so": {"O": 0.40, "W": 0.06, "X": 0.06}, "len": {"O": 0.10, "W": 0.10, "X": 0.08},
        "mu": {"pair": -0.10, "rev": 0.25, "soX": 0.25, "len": 1.60, "crop": 1.60, "correct": 0.0, "correct_shrub": 0.5},
        "shrub_shift": 0.0, "error_shift": 0.0}


def _quantiles(n, rng):
    from scipy.stats import norm
    q = norm.ppf((np.arange(n) + 0.5) / n)
    return q[rng.permutation(n)]


def synth_model(design, labels_grp, rng):
    """Classes and suspicion of one model at the plots with input, by design: per group the error types in fixed
    shares, the suspicion a standard-normal quantile around each type's location. Returns pred, label (TimeSync
    names), u, p1."""
    n = labels_grp.size
    pred = np.zeros(n, np.int64)
    lab = np.empty(n, object)
    kind = np.empty(n, object)
    for g in STRATA:
        ix = np.flatnonzero(labels_grp == g)
        m = ix.size
        n_so, n_len = int(round(design["so"][g] * m)), int(round(design["len"][g] * m))
        t = np.array(["so"] * n_so + ["len"] * n_len + ["correct"] * (m - n_so - n_len), object)[rng.permutation(m)]
        for j, k in zip(ix, t):
            if g == "O":
                lab[j] = "Open Grassland"
                pred[j], kind[j] = {"correct": (GRASS, "correct"), "so": (SHRUB, "pair"), "len": (5, "len")}[k]
            elif g == "W":
                lab[j] = "Wooded Grassland"
                pred[j], kind[j] = {"correct": (SHRUB, "correct"), "so": (GRASS, "rev"), "len": (5, "len")}[k]
            else:                                    # the label by position, so that every design has the same
                t3 = ("Cropland", "Otherland", "Open Forest")[j % 3]
                lab[j] = t3
                if t3 == "Cropland":                 # no STRICT-only error exists on Cropland
                    pred[j], kind[j] = (SHRUB, "crop") if k == "len" else (5, "correct")
                elif t3 == "Otherland":
                    pred[j], kind[j] = {"correct": (GRASS, "correct"), "so": (1, "soX"), "len": (5, "len")}[k]
                else:
                    pred[j], kind[j] = {"correct": (SHRUB, "correct"), "so": (0, "soX"), "len": (5, "len")}[k]
    kind[(kind == "correct") & (pred == SHRUB)] = "correct_shrub"
    s = np.zeros(n)
    for k, mu in design["mu"].items():
        ix = np.flatnonzero(kind == k)
        s[ix] = mu + _quantiles(ix.size, rng)
    s = s + design["shrub_shift"] * (pred == SHRUB) + design["error_shift"] * (kind != "correct") + 1e-6
    u = -np.log(CUT["v1"]) * np.exp(0.8 * s)          # u at the cut when s = 0: p1 >= the cut exactly when s <= 0
    return pred, lab, u, kind


def synth(design_v1, design_v12=None, seed=7):
    """A synthetic units pair and record in the units files' schema: 6 cells x per_cell drawn plots, the first `cover`
    of each cell with input, labels and classes by design, region weights by the page's formula."""
    design_v12 = design_v12 or design_v1
    rng = np.random.default_rng(seed)
    n_c = design_v1["per_cell"]
    country, stratum = [], []
    for ci in range(len(COUNTRIES)):
        for gi in range(len(STRATA)):
            country += [ci] * n_c
            stratum += [gi] * n_c
    country, stratum = np.array(country), np.array(stratum)
    n = country.size
    cov = np.zeros(n, bool)
    for c in range(len(COUNTRIES) * len(STRATA)):
        ix = np.flatnonzero(country * len(STRATA) + stratum == c)
        cov[ix[:int(round(design_v1["cover"] * ix.size))]] = True
    grp = np.array([STRATA[g] for g in stratum])
    units = {}
    lab_full = None
    for m, des in (("v1", design_v1), ("v12", design_v12)):
        r = np.random.default_rng(seed + 1)                     # the same labels for both models
        pred_c, lab_c, u_c, _ = synth_model(des, grp[cov], r)
        if lab_full is None:
            lab_full = np.array(["Open Grassland" if g == "O" else "Wooded Grassland" if g == "W" else "Cropland"
                                 for g in grp], object)
            lab_full[cov] = lab_c
        elif not np.array_equal(lab_full[cov], lab_c):
            raise AssertionError("the synthetic labels differ between models")
        pred = np.full(n, -1, np.int64)
        u = np.full(n, np.nan)
        pred[cov], u[cov] = pred_c, -np.log(CUT[m]) / -np.log(CUT["v1"]) * u_c
        p1 = np.exp(-u)
        units[m] = {"pred": pred, "suspicion": u, "p1": p1}
    w = b111.region_weights(np.array([COUNTRIES[i] for i in country]), grp, cov, SMOKE_AREA, SMOKE_POOL)
    common = {"country": country, "country_names": np.array(COUNTRIES),
              "label": np.array([tsx.TIMESYNC_CLASSES.index(t) for t in lab_full]),
              "label_names": np.array(tsx.TIMESYNC_CLASSES), "covered": cov, "inside_awf": np.zeros(n, bool),
              "weight": w, "stratum": stratum, "stratum_names": np.array(STRATA),
              "band": np.arange(n) % 3, "cgls_raw": np.where(np.arange(n) % 5 == 0, -1, 30),
              "in_gate17_cell": np.arange(n) % 97 == 0}
    out = {}
    for m in MODELS:
        d = dict(common)
        d.update(units[m])
        d["condition"] = np.zeros(n, np.int64)
        d["majority_3x3"] = d["pred"].copy()
        labs = np.array([tsx.TIMESYNC_CLASSES[i] for i in d["label"]])
        for rule in ("strict", "lenient"):
            e = np.full(n, np.nan)
            e[cov] = tsx.wrong(labs[cov], d["pred"][cov], rule)
            d[f"wrong_{rule}"] = e
        out[m] = d
    rec = {"draw_sha256": "0" * 64, "pool_counts": SMOKE_POOL, "area_outside_region_km2": SMOKE_AREA,
           "band_cuts_km": [200.0, 300.0]}
    return out, rec


def _variant(**kw):
    d = json.loads(json.dumps(BASE))
    for k, v in kw.items():
        if isinstance(v, dict):
            d[k].update(v)
        else:
            d[k] = v
    return d


# each case: the v1 design (v1.2 keeps BASE unless given), and the verdicts it must give: {model: {arm: expected}}
SMOKE_CASES = {
    "base": (BASE, None, {m: {"P1a": "holds", "P1b": "holds", "P2a": "holds", "P2b": "holds", "P3": "holds",
                              "lenient_call": "above"} for m in MODELS}),
    "floor_with_input": (_variant(cover=0.75), _variant(cover=0.75),
                         {m: {"P1": "no verdict (below a floor)", "P2": "no verdict (below a floor)",
                              "P3": "no verdict (below a floor)"} for m in MODELS}),
    "floor_lenient_errors": (_variant(len={"O": 0.02, "W": 0.02, "X": 0.02}), None,
                             {"v1": {"P1": "no verdict (below a floor)", "P2": "no verdict (below a floor)"},
                              "v12": {"P1": "holds"}}),
    "floor_p2_counts": (_variant(error_shift=6.0), None, {"v1": {"P1": "does not hold",
                                                                 "P2": "no verdict (below a floor)"}}),
    "floor_p3_zone": (_variant(shrub_shift=6.0), None, {"v1": {"P3": "no verdict (below a floor)"},
                                                        "v12": {"P3": "holds"}}),
    "p1a_too_high": (_variant(mu={"pair": 1.4, "rev": 1.4, "soX": 1.4, "len": 2.0, "crop": 2.0}), None,
                     {"v1": {"P1a": "does not hold"}, "v12": {"P1a": "holds"}}),
    "p1a_noise": (_variant(mu={"pair": 0.0, "rev": 0.0, "soX": 0.0, "len": 0.0, "crop": 0.0}), None,
                  {"v1": {"P1a": "does not hold"}, "v12": {"P1a": "holds"}}),
    "p1b_and_gap": (_variant(mu={"pair": 0.6, "rev": 0.6, "soX": 0.6, "len": 0.6, "crop": 0.6}), None,
                    {"v1": {"P1b": "does not hold", "P2b": "does not hold"}, "v12": {"P1b": "holds", "P2b": "holds"}}),
    "p2a_low": (_variant(so={"O": 0.08, "W": 0.30, "X": 0.20}), None,
                {"v1": {"P2a": "does not hold"}, "v12": {"P2a": "holds"}}),
    "p2b_strict_only_high": (_variant(mu={"correct": -0.6, "correct_shrub": -0.6, "pair": 0.4, "rev": 0.4, "soX": 0.4,
                                          "len": 2.0, "crop": 2.0}), None, {"v1": {"P2b": "does not hold"}}),
    "p2b_lenient_low": (_variant(mu={"pair": -0.45, "rev": -0.45, "soX": -0.45, "len": 0.30, "crop": 0.30}), None,
                        {"v1": {"P2b": "does not hold"}}),
    "p3_masked": (_variant(len={"X": 0.60}, mu={"crop": -1.5}), None, {"v1": {"P3": "does not hold"},
                                                                         "v12": {"P3": "holds"}}),
    "lenient_below": (_variant(mu={"len": 0.15, "crop": 0.15}), None, {"v1": {"lenient_call": "below"}}),
    "lenient_not_determined": (_variant(mu={"len": 1.00, "crop": 1.00}), None,
                               {"v1": {"lenient_call": "not determined"}}),
    # the interval cases: the point clears the bar, the 90% interval does not (and the 50% interval does), so a grader
    # that read the point, or a narrower interval, in place of the 90% interval's lower end would give "holds"
    "p1a_interval": (_variant(mu={"pair": 0.12, "rev": 0.12, "soX": 0.12, "len": 0.12, "crop": 0.12,
                                  "correct_shrub": 0.0}), None, {"v1": {"P1a": "does not hold"}}),
    "p1b_interval": (_variant(mu={"pair": 0.4, "rev": 0.4, "soX": 0.4, "len": 0.6, "crop": 0.6}), None,
                     {"v1": {"P1b": "does not hold"}}),
    "p2b_gap_interval": (_variant(mu={"pair": 0.45, "rev": 0.45, "soX": 0.45, "len": 0.7, "crop": 0.7,
                                      "correct_shrub": 0.0}), None, {"v1": {"P2b": "does not hold"}}),
}
# the arms whose sub-conditions must be the ones that failed (and held)
SMOKE_REASONS = {"p1a_too_high": [("P1a", "strict_below_0.70", False)],
                 "p1a_noise": [("P1a", "interval_above_0.50", False)],
                 "p2b_strict_only_high": [("P2b", "strict_only_at_most_0.65", False)],
                 "p2b_lenient_low": [("P2b", "lenient_at_least_0.65", False)],
                 "p1b_and_gap": [("P2b", "gap_interval_above_0", False)],
                 "p1a_interval": [("P1a", "interval_above_0.50", False), ("P1a", "strict_below_0.70", True)],
                 "p2b_gap_interval": [("P2b", "gap_interval_above_0", False), ("P2b", "strict_only_at_most_0.65", True),
                                      ("P2b", "lenient_at_least_0.65", True)]}
# the interval cases' quantity and bar: the point must clear the bar, the grader's 90% interval must equal the plain
# bootstrap's to 1e-12 and reach the bar, and the plain bootstrap's 50% interval must clear it
SMOKE_INTERVALS = {"p1a_interval": ("auc_strict", BARS["P1a_lower_above"]),
                   "p1b_interval": ("diff_lenient_minus_strict", BARS["P1b_lower_above"]),
                   "p2b_gap_interval": ("gap_std", BARS["P2b_gap_lower_above"])}
PLAIN_KEYS = ("auc_strict", "diff_lenient_minus_strict", "gap_std")


def brute_auc(u, e, w):
    """Pairwise weighted AUROC by definition (O(n^2)): the independent known answer."""
    u, e, w = np.asarray(u, np.float64), np.asarray(e, bool), np.asarray(w, np.float64)
    up, un, wp, wn = u[e], u[~e], w[e], w[~e]
    gt = (up[:, None] > un[None, :]) + 0.5 * (up[:, None] == un[None, :])
    return float((wp[:, None] * wn[None, :] * gt).sum() / (wp.sum() * wn.sum()))


def plain_bootstrap(d, cells, R, seed=BOOT["seed"], levels=(0.90, 0.50)):
    """The bootstrap again in plain numpy, independently of bootstrap() and the package's AUROC: the same seed, the
    plots resampled within each country x label-group cell (cells in increasing code, each g[rng.integers(0, n, n)],
    which is what Generator.choice draws), the pairwise AUROC (brute_auc), numpy's percentile. {key: {level: [lo, hi]}}
    for PLAIN_KEYS, at the plots with input of one model (prepared()'s arrays)."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(cells == c) for c in sorted(set(cells.tolist()))]
    vals = {k: [] for k in PLAIN_KEYS}
    for _ in range(R):
        ix = np.concatenate([g[rng.integers(0, g.size, g.size)] for g in groups])
        u, es, el, wr, ws = d["u"][ix], d["es"][ix], d["el"][ix], d["w_r"][ix], d["w_s"][ix]
        a_s, a_l = brute_auc(u, es, wr), brute_auc(u, el, wr)
        a, b = el | ~es, ~el
        vals["auc_strict"].append(a_s)
        vals["diff_lenient_minus_strict"].append(a_l - a_s)
        vals["gap_std"].append(brute_auc(u[a], el[a], ws[a]) - brute_auc(u[b], es[b], ws[b]))
    out = {}
    for k, v in vals.items():
        v = np.asarray(v, np.float64)
        out[k] = {lev: [float(np.quantile(v, (1 - lev) / 2)), float(np.quantile(v, 1 - (1 - lev) / 2))]
                  for lev in levels}
    return out


def p_note_other(eS, eL, n, ks):
    """P(note) another way: sum over K_L < ks of P(K_L) x P(K_SO >= ks - K_L | the rest), with scipy's binomial."""
    from scipy.stats import binom
    pL, pSO = eL, max(eS - eL, 0.0)
    tot = 0.0
    for kL in range(ks):
        q = pSO / (1 - pL) if pL < 1 else 0.0
        tot += binom.pmf(kL, n, pL) * binom.sf(ks - kL - 1, n - kL, q)
    return float(tot)


def smoke(out_dir=None, R=400, log=print):
    t0 = time.time()
    rec = {"mode": "smoke: synthetic units with known answers; the known answer on the committed exp99/exp103 units"}
    cases = {}
    for name, (d1, d12, expect) in SMOKE_CASES.items():
        units, urec = synth(d1, d12 or BASE)
        g = grade(units, urec, R=R, literal=(name == "base"), log=lambda *_: None)
        got, ok = {}, True
        for m, arms in expect.items():
            for arm, want in arms.items():
                v = g["models"][m][arm]["call"] if arm == "lenient_call" else g["models"][m][arm]["verdict"]
                got[f"{m}.{arm}"] = v
                ok &= v == want
        for arm, key, want in SMOKE_REASONS.get(name, []):
            got[f"v1.{arm}.{key}"] = g["models"]["v1"][arm][key]
            ok &= g["models"]["v1"][arm][key] == want
        if name in SMOKE_INTERVALS:
            key, bar = SMOKE_INTERVALS[name]
            dv1 = prepared(units, urec)
            pb = plain_bootstrap(dv1[0]["v1"], dv1[1]["cells"], R)
            point, iv90 = g["models"]["v1"]["point"][key], g["models"]["v1"]["intervals90"][key]
            got["interval"] = {"quantity": key, "bar": bar, "point": point, "grader_90": iv90,
                               "plain_90": pb[key][0.90], "plain_50": pb[key][0.50]}
            got["interval"]["checks"] = {
                "point_clears_the_bar": point > bar,
                "grader_90_equals_plain_90": max(abs(x - y) for x, y in zip(iv90, pb[key][0.90])) <= 1e-12,
                "90_lower_reaches_the_bar": iv90[0] <= bar,
                "50_lower_clears_the_bar": pb[key][0.50][0] > bar}
            ok &= all(got["interval"]["checks"].values())
        cases[name] = {"ok": bool(ok), "got": got,
                       "values_v1": {k: g["models"]["v1"]["point"][k] for k in
                                     ("n", "auc_strict", "auc_lenient", "pair_share_std", "auc_strict_only_vs_correct_std",
                                      "auc_lenient_vs_correct_std", "zone_plots")} | {
                           "p_note": g["models"]["v1"]["P3"]["p_note"],
                           "auc_strict_lo90": g["models"]["v1"]["P1a"]["interval90"][0]}}
        if name == "base":
            base_units, base_rec, base_g = units, urec, g
    rec["cases"] = cases
    # independent known answers on the base case
    data, common = prepared(base_units, base_rec)
    ind = {}
    for m in MODELS:
        d = data[m]
        s = base_g["models"][m]["point"]
        a, b = d["el"] | ~d["es"], ~d["el"]
        ind[m] = {
            "auc_strict": abs(brute_auc(d["u"], d["es"], d["w_r"]) - s["auc_strict"]) < 1e-9,
            "auc_lenient": abs(brute_auc(d["u"], d["el"], d["w_r"]) - s["auc_lenient"]) < 1e-9,
            "auc_lenient_vs_correct_std": abs(brute_auc(d["u"][a], d["el"][a], d["w_s"][a])
                                              - s["auc_lenient_vs_correct_std"]) < 1e-9,
            "auc_strict_only_vs_correct_std": abs(brute_auc(d["u"][b], d["es"][b], d["w_s"][b])
                                                  - s["auc_strict_only_vs_correct_std"]) < 1e-9,
            "std_weights_shares": all(abs(d["w_s"][np.array([STRATA[i] for i in base_units["v1"]["stratum"][common["K"]]])
                                                   == g].sum() - STD_COUNTS[g] / 271) < 1e-12 for g in STRATA),
            "p_note": abs(p_note(s["zone_eS"], s["zone_eL"], 20, 4) - p_note_other(s["zone_eS"], s["zone_eL"], 20, 4)) < 1e-12}
    # the region weight by its formula, computed another way: each cell's plots share A'_c N_cg / N_c equally
    a = base_units["v1"]
    cov = a["covered"].astype(bool)
    w2 = np.zeros(cov.size)
    for ci, c in enumerate(COUNTRIES):
        Nc = sum(v for k, v in SMOKE_POOL.items() if k.startswith(c))
        for gi, g in enumerate(STRATA):
            msk = (a["country"] == ci) & (a["stratum"] == gi) & cov
            w2[msk] = SMOKE_AREA[c] * SMOKE_POOL[f"{c}:{g}"] / Nc / msk.sum()
    ind["region_weights"] = bool(np.allclose(w2 / w2.sum(), a["weight"], rtol=0, atol=1e-15))
    # the bootstrap itself, recomputed in plain numpy on the base case: every PLAIN_KEYS interval to 1e-12, for both
    # models (the same resamples)
    ind["plain_bootstrap_equals_grader"] = {}
    for m in MODELS:
        pb = plain_bootstrap(data[m], common["cells"], R)
        ind["plain_bootstrap_equals_grader"][m] = all(
            max(abs(x - y) for x, y in zip(base_g["models"][m]["intervals90"][k], pb[k][0.90])) <= 1e-12
            for k in PLAIN_KEYS)
    ind["k_star_20"] = kstar(20) == 4
    # the floor of plots with input is at most 80% of the draw (720 of SIZES' 900; the owner may lower it, the page
    # says how; a smaller draw without a lower floor would ask for more than 80%)
    ind["floor_at_most_80_percent_of_sizes"] = FLOORS["with_input"] <= round(0.8 * sum(b111.SIZES.values()))
    ind["p_note_grid"] = all(abs(p_note(eS, eL, n, kstar(n)) - p_note_other(eS, eL, n, kstar(n))) < 1e-12
                             for n in (15, 20, 30) for eS, eL in ((0.3, 0.05), (0.2, 0.0), (0.1, 0.1), (0.45, 0.2)))
    # the also-accept file reads LENIENT exactly, the STRICT coding STRICT, for every label and map class
    acc_s, acc_l = tsx.accepted("strict"), tsx.accepted("lenient")
    ind["also_accept_is_lenient"] = all(
        ((c == EXTENDED_CODE[t]) or (EXTENDED_CODE[t] in ALSO_ACCEPT.get(c, []))) == (c in acc_l[t])
        and (c == STRICT_CODE[t]) == (c in acc_s[t]) for t in tsx.TIMESYNC_CLASSES for c in range(10))
    lit = base_g["models"]["v1"]["report_only"]["literal_audit"]
    ind["literal_readings_equal_the_crosswalk"] = bool(lit.get("readings_equal_the_crosswalk"))
    # determinism: the same units graded twice give the same intervals
    again = grade(base_units, base_rec, R=R, literal=False, log=lambda *_: None)
    ind["deterministic"] = again["models"]["v1"]["intervals90"] == base_g["models"]["v1"]["intervals90"]
    rec["independent"] = ind
    # the refusals: an unfrozen page, a frozen page with another draw, a units file that is not the record's
    with tempfile.TemporaryDirectory() as tmp:
        ud = os.path.join(tmp, "units")
        os.makedirs(ud)
        for m in MODELS:
            np.savez_compressed(os.path.join(ud, f"exp111_units_{m}.npz"), **base_units[m])
        urec = dict(base_rec, sha256={f"exp111_units_{m}.npz": sha256_file(os.path.join(ud, f"exp111_units_{m}.npz"))
                                      for m in MODELS})
        urec["draw_sha256"] = "a" * 64
        urec["sizes_O_W_X"] = [b111.SIZES[k] for k in STRATA]
        urec["n_plots"] = sum(b111.SIZES.values())
        refused = []
        # a draw at other sizes (exp109's Option A, 750 plots) against the floors fixed for SIZES: refused
        with open(os.path.join(ud, "exp111_units_record.json"), "w") as f:
            json.dump(dict(urec, sizes_O_W_X=[300, 300, 150], n_plots=750), f)
        try:
            check_inputs(ud, "**Status: frozen**\n**Draw sha256 (frozen):** `" + "a" * 64 + "`\n")
        except PermissionError:
            refused.append("a draw of other sizes")
        with open(os.path.join(ud, "exp111_units_record.json"), "w") as f:
            json.dump(urec, f)
        for what, text in (("a draft page", "**Status: DRAFT**\n**Draw sha256 (frozen):** `" + "a" * 64 + "`\n"),
                           ("a frozen page with another draw", "**Status: frozen**\n**Draw sha256 (frozen):** `"
                            + "b" * 64 + "`\n")):
            try:
                check_inputs(ud, text)
            except PermissionError:
                refused.append(what)
        accepted = check_inputs(ud, "**Status: frozen**\n**Draw sha256 (frozen):** `" + "a" * 64 + "`\n") is not None
        with open(os.path.join(ud, "exp111_units_v1.npz"), "ab") as f:
            f.write(b"\0")
        try:
            check_inputs(ud, "**Status: frozen**\n**Draw sha256 (frozen):** `" + "a" * 64 + "`\n")
        except PermissionError:
            refused.append("a units file changed after the record")
        # the summary passes the no-position guard (and, under out_dir/rec, the local smoke's pack check)
        path = write_summary(base_g, os.path.join(out_dir, "rec") if out_dir else tmp)
        rec["summary_written"] = os.path.getsize(path) > 0
    rec["refusals"] = refused
    rec["accepts_the_page_draw"] = accepted
    rec["known_answer_271"] = known_answer()
    rec["ok"] = bool(all(c["ok"] for c in cases.values()) and all(all(v.values()) if isinstance(v, dict) else v
                                                                  for v in ind.values())
                     and len(refused) == 4 and accepted and rec["summary_written"]
                     and rec["known_answer_271"]["as_expected"] is not False)
    rec["seconds"] = round(time.time() - t0, 1)
    if out_dir:
        import exp78_error_rate_estimation as e78
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "exp111_grade_smoke.json"), "w") as f:
            json.dump(e78.jsonable(rec), f, indent=1)
    log(json.dumps({"ok": rec["ok"], "cases": {k: v["ok"] for k, v in cases.items()},
                    "known_answer_271": rec["known_answer_271"]["as_expected"], "refusals": refused}))
    return rec


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--units-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resamples", type=int, default=None, help="the smoke's resamples (the grade uses 2,000)")
    a = ap.parse_args(argv)
    if a.smoke:
        rec = smoke(a.out_dir, R=a.resamples or 400)
        return 0 if rec["ok"] else 1
    urec = check_inputs(a.units_dir)
    units, _ = load_units(a.units_dir)
    out = grade(units, urec)
    out["units_record_sha256"] = sha256_file(os.path.join(a.units_dir, "exp111_units_record.json"))
    out["units_sha256"] = urec["sha256"]
    print(write_summary(out, a.out_dir or OUT))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PermissionError as ex:
        print(str(ex))
        sys.exit(3)
