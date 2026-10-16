"""exp111's second route (docs/plan/blind_transfer.md, frozen 6d80ab7): the graded numbers recomputed from the
committed per-plot files with numpy and scipy, written from the independent audit of 16 October 2026 and the frozen
page's definitions. It imports neither the grader (exp/exp111_grade.py), the run's modules (exp/exp111_blind_transfer.py,
exp/exp111_jobs.py) nor the package; it imports the crosswalk (exp/timesync_awf_crosswalk.py, the rules the page names)
only to check the units' error columns.

    python exp/exp111_crosscheck.py [--out exp/out/exp111_crosscheck.json]

Reads exp/out/exp111_units_v1.npz and exp111_units_v12.npz (the run's per-plot files without the cgls_raw column, so R1,
the CGLS-disagreement order, is not recomputed here), exp111_units_record.json (areas and pool counts, for the region
weights), exp111_units_committed.json (the files' sha256) and the gate records under exp/out/exp111 (the passing gate,
the failed one and the v1 route diagnostic). Writes aggregates only.

The page's definitions, as used here:
- region weight of a plot with input in cell (c, g): (A'_c / N_c) x (N_cg / m_cg), m_cg its plots with input;
- standardised weights (P2): label group g gets its share of exp99's 271 (O 97, W 129, X 45); within g, plots split
  as the region weights split them;
- weighted AUROC: P(s_error > s_other) + 0.5 P(tie), each pair weighted by the product of its weights; s = -log p1;
- confident: p1 at or above the model's cut (0.99875695977 v1, 0.99888118198 v1.2);
- the pair: map shrubland_savanna on TimeSync Open Grassland; STRICT-only: wrong under STRICT, right under LENIENT;
- the zone (P3): confident plots of class shrubland_savanna; e_S and e_L its region-weighted STRICT and LENIENT rates;
  k*: the smallest error count out of n labels whose 95% interval in the binomial limit (Clopper-Pearson) has its lower
  end at or above 0.05; P(note) = P(K_S >= k*, K_L <= k* - 1), (K_L, K_S - K_L, rest) multinomial(n; e_L, e_S - e_L,
  1 - e_S), enumerated exactly;
- intervals: 2,000 resamples of the plots with input within each country x label-group cell, cells visited Kenya O, W,
  X then Tanzania O, W, X, each resample's draws in that order from numpy.random.default_rng(111) by
  rng.integers(0, m_cg, m_cg); the same resamples for both models; 90% percentile. This stream reproduces the grade's
  intervals exactly; other streams (other seeds, a multinomial resampler, all of a cell's resamples at once, the cells
  in label-group order) are reported beside to show the Monte Carlo spread of the verdict-bearing ends.
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
from scipy import stats

EXP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP)
OUT = os.path.join(ROOT, "exp", "out")
sys.path.insert(0, EXP)
import timesync_awf_crosswalk as cw  # noqa: E402  (the rules the page names; only to check the error columns)

MODELS = ("v1", "v12")
CUT = {"v1": 0.99875695977, "v12": 0.99888118198}
STD_COUNTS = {"O": 97, "W": 129, "X": 45}                  # exp99's 271 plots with input, by label group
BARS = {"P1a_strict_below": 0.70, "P1a_lower_above": 0.50, "P1b_lower_above": 0.0, "lenient_call": 0.70,
        "P2a_pair_share_min": 0.50, "P2b_gap_lower_above": 0.0, "P2b_strict_only_max": 0.65, "P2b_lenient_min": 0.65,
        "P3_p_note_min": 0.5}
FLOORS = {"with_input": 720, "errors_per_rule": 40, "correct_per_rule": 40, "confident_strict_errors": 25,
          "strict_only_errors": 30, "zone_plots": 30}
R, SEED, LEVEL = 2000, 111, 0.90
ALPHA, CONF = 0.05, 0.95
SHRUB, GRASS = cw.AWF_INDEX["shrubland_savanna"], cw.AWF_INDEX["grassland_barren"]
CELL_ORDER = ("kenya:O", "kenya:W", "kenya:X", "tanzania:O", "tanzania:W", "tanzania:X")
GROUP_ORDER = ("kenya:O", "tanzania:O", "kenya:W", "tanzania:W", "kenya:X", "tanzania:X")
ALT_STREAMS = (("integers", 1, CELL_ORDER), ("integers", 2, CELL_ORDER), ("multinomial", 3, CELL_ORDER),
               ("multinomial", 4, CELL_ORDER), ("cell_block", 111, CELL_ORDER), ("integers", 111, GROUP_ORDER))
GATE_DIRS = {"8dc3a01": os.path.join(OUT, "exp111", "records"), "3dd770d": os.path.join(OUT, "exp111", "gate_3dd770d")}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- statistics
def auc_rows(score, pos, neg, W):
    """Weighted AUROC of `score` for the `pos` plots against the `neg` plots, once per row of W (k x n weights):
    P(s_pos > s_neg) + 0.5 P(s_pos == s_neg), each pair weighted by the product of its weights."""
    W = np.atleast_2d(W)
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    P = np.add.reduceat(W[:, order] * pos[order], starts, axis=1)
    N = np.add.reduceat(W[:, order] * neg[order], starts, axis=1)
    below = np.cumsum(N, axis=1) - N
    num = (P * (below + 0.5 * N)).sum(axis=1)
    den = P.sum(axis=1) * N.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def auc_pairs(score, pos, neg, w):
    """The same statistic by every pair (the check of auc_rows on these data)."""
    sp, wp, sn, wn = score[pos], w[pos], score[neg], w[neg]
    d = sp[:, None] - sn[None, :]
    k = (d > 0) + 0.5 * (d == 0)
    return float((wp[:, None] * wn[None, :] * k).sum() / (wp.sum() * wn.sum()))


def ratio_rows(W, num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W * num).sum(axis=1) / (W * den).sum(axis=1)


def capture(score, err, w, q=0.10):
    """The weighted share of the errors in the most suspect q of the weight (a boundary plot counted in part), with
    the random share, the ceiling min(1, q / error rate) and the share of the gap closed."""
    order = np.argsort(-score, kind="stable")
    ws, es = w[order], err[order]
    W = ws.sum()
    prev = np.cumsum(ws) - ws
    frac = np.clip((q * W - prev) / ws, 0.0, 1.0)
    cap = float((frac * ws * es).sum() / (ws * es).sum())
    rate = float((ws * es).sum() / W)
    ceiling = min(1.0, q / rate)
    return {"capture": cap, "error_rate": rate, "ceiling": ceiling, "gap_closed": (cap - q) / (ceiling - q)}


def pct(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(v) for v in np.percentile(x, [100 * (1 - LEVEL) / 2, 100 * (1 + LEVEL) / 2])]


def multiplicities(cells, n, how, seed, order_idx):
    """R x n resampling counts: each cell's plots with input resampled with replacement, per `how`."""
    rng = np.random.default_rng(seed)
    M = np.zeros((R, n))
    cs = [cells[i] for i in order_idx]
    if how == "cell_block":
        draws = [rng.integers(0, len(c), size=(R, len(c))) for c in cs]
        for r in range(R):
            for c, d in zip(cs, draws):
                M[r] += np.bincount(c[d[r]], minlength=n)
        return M
    for r in range(R):
        for c in cs:
            if how == "integers":
                M[r] += np.bincount(c[rng.integers(0, len(c), len(c))], minlength=n)
            else:
                M[r, c] += rng.multinomial(len(c), np.full(len(c), 1.0 / len(c)))
    return M


def k_star(n):
    """The smallest k whose two-sided 95% Clopper-Pearson interval (the binomial limit) has its lower end >= alpha."""
    for k in range(1, n + 1):
        if stats.beta.ppf((1 - CONF) / 2, k, n - k + 1) >= ALPHA:
            return k
    return None


def p_note(n, eS, eL):
    ks = k_star(n)
    pL, pSo, pOk = eL, max(eS - eL, 0.0), max(1.0 - eS, 0.0)
    tot = 0.0
    for a in range(0, min(ks - 1, n) + 1):                 # a = K_L <= k* - 1
        for b in range(max(0, ks - a), n - a + 1):          # a + b = K_S >= k*
            c = n - a - b
            tot += math.comb(n, a) * math.comb(n - a, b) * pL ** a * pSo ** b * pOk ** c
    return {"k_star": ks, "p_note": tot, "p_strict_above": float(stats.binom.sf(ks - 1, n, eS)),
            "p_lenient_above": float(stats.binom.sf(ks - 1, n, eL))}


# ----------------------------------------------------------------------------- the units
def load_units(rec, committed):
    U, files = {}, {}
    for m in MODELS:
        fn = f"exp111_units_{m}.npz"
        p = os.path.join(OUT, fn)
        h = sha256(p)
        files[fn] = {"sha256": h, "is_the_committed_file": h == committed["files"][fn]["committed_sha256"],
                     "run_sha256": rec["sha256"][fn]}
        z = np.load(p, allow_pickle=False)
        U[m] = {k: z[k] for k in z.files}
    return U, files


def build(U, rec):
    """The shared columns, the cells, the region and standardised weights, and the integrity checks."""
    u = U["v12"]
    cn = [str(x) for x in u["country_names"]]
    gn = [str(x) for x in u["stratum_names"]]
    ln = [str(x) for x in u["label_names"]]
    shared = [k for k in u if k not in ("pred", "suspicion", "p1", "majority_3x3", "wrong_strict", "wrong_lenient")]
    checks = {"rows": int(len(u["country"])), "no_cgls_raw_column": all("cgls_raw" not in U[m] for m in MODELS),
              "shared_columns_equal_across_models": all(
                  np.array_equal(U["v1"][k], U["v12"][k], equal_nan=U["v1"][k].dtype.kind == "f") for k in shared)}
    cell = np.array([f"{cn[c]}:{gn[s]}" for c, s in zip(u["country"], u["stratum"])])
    cov = u["covered"].astype(bool)
    lab = np.array([ln[i] for i in u["label"]])
    lg = np.where(lab == "Open Grassland", "O", np.where(lab == "Wooded Grassland", "W", "X"))
    checks["label_group_is_the_stratum"] = bool(np.array_equal(lg, np.array([gn[s] for s in u["stratum"]])))
    checks["drawn_by_cell"] = {c: int((cell == c).sum()) for c in CELL_ORDER}
    checks["drawn_by_cell_is_the_records"] = checks["drawn_by_cell"] == rec["drawn_by_cell"]
    checks["with_input_by_cell"] = {c: int(((cell == c) & cov).sum()) for c in CELL_ORDER}
    checks["in_gate17_cell"] = int(u["in_gate17_cell"].sum())
    checks["inside_awf"] = int(u["inside_awf"].sum())

    pool, area = rec["pool_counts"], rec["area_outside_region_km2"]
    Nc = {c: sum(v for k, v in pool.items() if k.startswith(c + ":")) for c in cn}
    w_reg = np.zeros(len(cov))
    for c in CELL_ORDER:
        sel = (cell == c) & cov
        country = c.split(":")[0]
        w_reg[sel] = (area[country] / Nc[country]) * (pool[c] / sel.sum())
    checks["region_weights_sum_to_the_area_km2"] = [float(w_reg.sum()), float(sum(area.values()))]
    col = np.where(cov, u["weight"], 0.0)
    checks["weight_column_is_the_region_weight_over_its_sum_max_abs"] = float(np.max(np.abs(w_reg / w_reg.sum() - col)))
    checks["weight_column_zero_without_input"] = bool(np.all(u["weight"][~cov] == 0))

    w_std = np.zeros(len(cov))
    for g, k in STD_COUNTS.items():
        sel = cov & (lg == g)
        w_std[sel] = (k / sum(STD_COUNTS.values())) * w_reg[sel] / w_reg[sel].sum()

    for m in MODELS:
        d = U[m]
        name = [cw.AWF_CLASSES.get(int(p)) for p in d["pred"]]  # channel 9 has no name: wrong under both rules
        for rule in ("strict", "lenient"):
            mine = np.array([0.0 if n_ in cw.RULES[rule][t] else 1.0 for n_, t in zip(name, lab)])
            checks[f"{m}_wrong_{rule}_is_the_crosswalk"] = bool(np.array_equal(mine[cov], d[f"wrong_{rule}"][cov]))
        checks[f"{m}_suspicion_is_minus_log_p1_max_abs"] = float(np.max(np.abs(d["suspicion"][cov] + np.log(d["p1"][cov]))))
        checks[f"{m}_ties_in_suspicion"] = int(cov.sum() - len(np.unique(d["suspicion"][cov])))
        checks[f"{m}_nan_without_input_only"] = bool(np.all(np.isnan(d["p1"][~cov])) and not np.isnan(d["p1"][cov]).any())
    rule_rec = cw.crosswalk_record()
    checks["crosswalk_sha256"] = rule_rec["sha256"]
    cells_idx = [np.flatnonzero(cell[cov] == c) for c in CELL_ORDER]
    return {"cn": cn, "gn": gn, "ln": ln, "cell": cell, "cov": cov, "lab": lab, "lg": lg, "w_reg": w_reg,
            "w_std": w_std, "cells_idx": cells_idx}, checks


# ----------------------------------------------------------------------------- one model
def model(d, m, B, Ms):
    cov, lab = B["cov"], B["lab"]
    s = d["suspicion"][cov]
    ws = d["wrong_strict"][cov] > 0.5
    wl = d["wrong_lenient"][cov] > 0.5
    p1, pred, lb, band = d["p1"][cov], d["pred"][cov], lab[cov], d["band"][cov]
    wr, wsd, one = B["w_reg"][cov], B["w_std"][cov], np.ones(int(cov.sum()))
    so = ws & ~wl
    sc = ~ws
    conf = p1 >= CUT[m]
    cse = conf & ws
    pair = cse & (pred == SHRUB) & (lb == "Open Grassland")
    rev = cse & (pred == GRASS) & (lb == "Wooded Grassland")
    zone = conf & (pred == SHRUB)
    out = {"counts": {
        "n": int(cov.sum()), "errors_strict": int(ws.sum()), "errors_lenient": int(wl.sum()),
        "correct_strict": int((~ws).sum()), "correct_lenient": int((~wl).sum()), "strict_only": int(so.sum()),
        "confident": int(conf.sum()), "confident_strict_errors": int(cse.sum()),
        "confident_strict_errors_pair": int(pair.sum()), "confident_strict_only_errors": int((cse & so).sum()),
        "zone_plots": int(zone.sum()),
        "nearest_p1_above_the_cut": float((p1[conf] - CUT[m]).min()),
        "nearest_p1_below_the_cut": float((CUT[m] - p1[~conf]).min())}}
    c = out["counts"]
    out["floors"] = {
        "with_input": c["n"] >= FLOORS["with_input"],
        "errors_and_correct_per_rule": min(c["errors_strict"], c["errors_lenient"]) >= FLOORS["errors_per_rule"]
        and min(c["correct_strict"], c["correct_lenient"]) >= FLOORS["correct_per_rule"],
        "P2_counts": c["confident_strict_errors"] >= FLOORS["confident_strict_errors"]
        and c["strict_only"] >= FLOORS["strict_only_errors"],
        "P2_counts_if_strict_only_were_counted_among_the_confident":
            c["confident_strict_errors"] >= FLOORS["confident_strict_errors"]
            and c["confident_strict_only_errors"] >= FLOORS["strict_only_errors"],
        "P3_zone": c["zone_plots"] >= FLOORS["zone_plots"]}

    # P1, region weights
    a_s, a_l = auc_rows(s, ws, ~ws, wr)[0], auc_rows(s, wl, ~wl, wr)[0]
    P1 = {"auc_strict": float(a_s), "auc_lenient": float(a_l), "diff_lenient_minus_strict": float(a_l - a_s),
          "auc_strict_by_every_pair": auc_pairs(s, ws, ~ws, wr), "auc_lenient_by_every_pair": auc_pairs(s, wl, ~wl, wr),
          "auc_strict_unweighted": float(auc_rows(s, ws, ~ws, one)[0]),
          "auc_lenient_unweighted": float(auc_rows(s, wl, ~wl, one)[0])}
    P1["diff_unweighted"] = P1["auc_lenient_unweighted"] - P1["auc_strict_unweighted"]
    P1["capture_10"] = {f"{rule}_{wn}": capture(s, e.astype(float), w)
                        for rule, e in (("strict", ws), ("lenient", wl)) for wn, w in (("weighted", wr), ("unweighted", one))}
    # P2, standardised to the 271's label mix; region-weighted and unweighted beside
    def p2(w):
        al, aso = auc_rows(s, wl, sc, w)[0], auc_rows(s, so, sc, w)[0]
        den = w[cse].sum()
        return {"pair_share": float(w[pair].sum() / den), "reverse_pair_share": float(w[rev].sum() / den),
                "strict_only_share_of_confident_strict_errors": float(w[cse & so].sum() / den),
                "auc_lenient_vs_correct": float(al), "auc_strict_only_vs_correct": float(aso), "gap": float(al - aso)}
    P2 = {"standardised": p2(wsd), "region_weighted": p2(wr), "unweighted": p2(one),
          "pair_share_unweighted_count": [int(pair.sum()), int(cse.sum())]}
    # P3
    eS = float((wr * zone * ws).sum() / (wr * zone).sum())
    eL = float((wr * zone * wl).sum() / (wr * zone).sum())
    P3 = {"zone_eS": eS, "zone_eL": eL, "labels": {str(n): p_note(n, eS, eL) for n in (15, 20, 30)},
          "zone_eS_unweighted": float(ws[zone].mean()), "zone_eL_unweighted": float(wl[zone].mean()),
          "zone_by_label": {t: [int((zone & (lb == t)).sum()), int((zone & (lb == t) & ws).sum()),
                                int((zone & (lb == t) & wl).sum())] for t in sorted(set(lb[zone].tolist()))}}

    # intervals: the grade's stream
    M = Ms["primary"]
    bs = auc_rows(s, ws, ~ws, M * wr)
    bl = auc_rows(s, wl, ~wl, M * wr)
    b_al = auc_rows(s, wl, sc, M * wsd)
    b_so = auc_rows(s, so, sc, M * wsd)
    iv = {"auc_strict": pct(bs), "auc_lenient": pct(bl), "diff_lenient_minus_strict": pct(bl - bs),
          "auc_lenient_vs_correct_std": pct(b_al), "auc_strict_only_vs_correct_std": pct(b_so), "gap_std": pct(b_al - b_so),
          "pair_share_std": pct(ratio_rows(M * wsd, pair, cse)),
          "zone_eS": pct(ratio_rows(M * wr, zone & ws, zone)), "zone_eL": pct(ratio_rows(M * wr, zone & wl, zone)),
          "diff_unweighted": pct(auc_rows(s, wl, ~wl, M) - auc_rows(s, ws, ~ws, M))}
    other = {}
    for key, Mo in Ms.items():
        if key == "primary":
            continue
        os_, ol_ = auc_rows(s, ws, ~ws, Mo * wr), auc_rows(s, wl, ~wl, Mo * wr)
        oa, oso = auc_rows(s, wl, sc, Mo * wsd), auc_rows(s, so, sc, Mo * wsd)
        other[key] = {"auc_strict": pct(os_), "auc_lenient": pct(ol_), "diff_lenient_minus_strict": pct(ol_ - os_),
                      "auc_strict_only_vs_correct_std": pct(oso), "gap_std": pct(oa - oso),
                      "pair_share_std": pct(ratio_rows(Mo * wsd, pair, cse))}
    out.update({"P1": P1, "P2": P2, "P3": P3, "intervals90": iv, "intervals90_other_streams": other})
    out["strict_lower_end_over_the_other_streams"] = [min(v["auc_strict"][0] for v in other.values()),
                                                      max(v["auc_strict"][0] for v in other.values())]

    # verdicts, as the page states them
    lo_s, lo_d = iv["auc_strict"][0], iv["diff_lenient_minus_strict"][0]
    lo_l, hi_l = iv["auc_lenient"]
    p1a = a_s < BARS["P1a_strict_below"] and lo_s > BARS["P1a_lower_above"]
    p1b = lo_d > BARS["P1b_lower_above"]
    st = P2["standardised"]
    p2a = st["pair_share"] >= BARS["P2a_pair_share_min"]
    arms = {"gap_interval_above_0": iv["gap_std"][0] > BARS["P2b_gap_lower_above"],
            "strict_only_at_most_0.65": st["auc_strict_only_vs_correct"] <= BARS["P2b_strict_only_max"],
            "lenient_at_least_0.65": st["auc_lenient_vs_correct"] >= BARS["P2b_lenient_min"]}
    p2b = all(arms.values())
    p3 = P3["labels"]["20"]["p_note"] >= BARS["P3_p_note_min"]
    hold = lambda x: "holds" if x else "does not hold"  # noqa: E731
    out["verdicts"] = {"P1a": hold(p1a), "P1b": hold(p1b), "P1": hold(p1a and p1b), "P2a": hold(p2a),
                       "P2b": hold(p2b), "P2b_arms": arms, "P2": hold(p2a and p2b), "P3": hold(p3),
                       "lenient_call": "above" if lo_l > BARS["lenient_call"] else
                       "below" if hi_l < BARS["lenient_call"] else "not determined"}
    # P2b's STRICT-only arm without the standardisation, beside it
    out["strict_only_arm_by_weighting"] = {k: P2[k]["auc_strict_only_vs_correct"] <= BARS["P2b_strict_only_max"]
                                           for k in ("standardised", "region_weighted", "unweighted")}

    # the class holding the most weighted confident STRICT-only errors, by weighting
    def lead(w):
        tot = w[cse & so].sum()
        shares = {cw.AWF_CLASSES.get(int(k), "channel 9"): float(w[cse & so & (pred == k)].sum() / tot)
                  for k in np.unique(pred[cse & so])}
        top = max(shares, key=shares.get)
        return {"class": top, "share": shares[top], "shares": shares}
    out["class_with_most_confident_strict_only_errors"] = {"region_weighted": lead(wr), "standardised": lead(wsd),
                                                           "unweighted": lead(one)}
    # R2: fresh plots by distance band (region weights)
    out["R2_bands"] = {}
    for b in (0, 1, 2):
        sel = band == b
        out["R2_bands"][str(b)] = {
            "plots": int(sel.sum()),
            "strict": {"error_rate": float((wr[sel] * ws[sel]).sum() / wr[sel].sum()),
                       "auroc": float(auc_rows(s[sel], ws[sel], ~ws[sel], wr[sel])[0])},
            "lenient": {"error_rate": float((wr[sel] * wl[sel]).sum() / wr[sel].sum()),
                        "auroc": float(auc_rows(s[sel], wl[sel], ~wl[sel], wr[sel])[0])}}
    out["R5_condition_codes"] = {str(k): int(v) for k, v in zip(*np.unique(d["condition"][cov], return_counts=True))}
    return out


def r4(U, B, rec):
    """The region's error rate with the six cells as strata, and the missing-input envelope (all right, all wrong)."""
    cn = B["cn"]
    area = rec["area_outside_region_km2"]
    pool = rec["pool_counts"]
    Nc = {c: sum(v for k, v in pool.items() if k.startswith(c + ":")) for c in cn}
    W = {c: area[c.split(":")[0]] * pool[c] / Nc[c.split(":")[0]] for c in CELL_ORDER}
    Wt = sum(W.values())
    out = {}
    for m in MODELS:
        out[m] = {}
        for rule in ("strict", "lenient"):
            e, right, wrong = 0.0, 0.0, 0.0
            for c in CELL_ORDER:
                drawn = B["cell"] == c
                have = drawn & B["cov"]
                k = float(np.nansum(U[m][f"wrong_{rule}"][have]))
                miss = int((drawn & ~B["cov"]).sum())
                e += W[c] / Wt * k / have.sum()
                right += W[c] / Wt * k / drawn.sum()
                wrong += W[c] / Wt * (k + miss) / drawn.sum()
            out[m][rule] = {"estimate": e, "envelope_missing_right": right, "envelope_missing_wrong": wrong}
    return out


def gate_records():
    """v1 and v1.2 against their known answers, from the three jobs that ran v1 at gate A's 259 points with the
    formula's table (the failed gate, the passing gate, the diagnostic's formula variant) and the diagnostic's others."""
    out = {}
    for tag, d in GATE_DIRS.items():
        g = read_json(os.path.join(d, "e111_gate.json"))
        A = g["A"]
        row = {"job": g["device"]["job"], "commit": g["commit"], "predict_commit": g["predict_commit"],
               "device": g["device"]["name"], "pass": g["pass"], "A_checks": A["checks"],
               "A": {m: {k: A["models"][m][k] for k in ("covered_both", "same_class", "dp1_above_tol", "spearman_suspicion")}
                     for m in MODELS}}
        if "B" in g:
            B = g["B"]
            row["B_checks"] = B["checks"]
            row["B"] = {m: {**{k: B["models"][m][k] for k in ("covered_both", "same_class", "dp1_above_tol",
                                                               "spearman_suspicion")},
                            "confident_known": B["models"][m]["confident_zone"]["known"],
                            "confident_kept": B["models"][m]["confident_zone"]["kept"]} for m in MODELS}
            row["graded_checks"] = {"A": A.get("graded_checks"), "B": B.get("graded_checks")}
        out[tag] = row
    dg = read_json(os.path.join(GATE_DIRS["8dc3a01"], "e111_v1diag.json"))
    out["v1diag"] = {k: {kk: v[kk] for kk in ("covered_both", "same_class", "dp1_above_tol", "spearman_suspicion",
                                               "would_pass_A1_A2")} for k, v in dg["variants"].items()}
    out["v1diag_failed_variants"] = dg["variants_failed"]
    a = [out["3dd770d"]["A"]["v1"], out["8dc3a01"]["A"]["v1"], out["v1diag"]["formula"]]
    out["v1_formula_route_the_same_in_three_jobs"] = all(
        (x["covered_both"], x["same_class"], x["dp1_above_tol"], round(x["spearman_suspicion"], 12))
        == (a[0]["covered_both"], a[0]["same_class"], a[0]["dp1_above_tol"], round(a[0]["spearman_suspicion"], 12))
        for x in a)
    out["v1_share_changing_class_at_gate_A"] = 1 - a[0]["same_class"] / a[0]["covered_both"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(OUT, "exp111_crosscheck.json"))
    a = ap.parse_args(argv)
    rec = read_json(os.path.join(OUT, "exp111_units_record.json"))
    committed = read_json(os.path.join(OUT, "exp111_units_committed.json"))
    U, files = load_units(rec, committed)
    B, checks = build(U, rec)
    n = int(B["cov"].sum())
    order = {c: i for i, c in enumerate(CELL_ORDER)}
    Ms = {"primary": multiplicities(B["cells_idx"], n, "integers", SEED, range(6))}
    for how, seed, cells in ALT_STREAMS:
        name = f"{how}_seed{seed}" + ("_label_group_order" if cells is GROUP_ORDER else "")
        Ms[name] = multiplicities(B["cells_idx"], n, how, seed, [order[c] for c in cells])
    with open(os.path.abspath(__file__), "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()
    out = {"experiment": "exp111 crosscheck: the graded numbers from the committed per-plot files by a second route",
           "script_sha256": script_sha, "page": "docs/plan/blind_transfer.md (frozen 6d80ab7)",
           "inputs": {"units": files, "units_record_sha256": sha256(os.path.join(OUT, "exp111_units_record.json")),
                      "draw_sha256": rec["draw_sha256"]},
           "bars": BARS, "floors": FLOORS, "confident_cut": CUT, "standardised_counts_O_W_X": list(STD_COUNTS.values()),
           "bootstrap": {"R": R, "seed": SEED, "level": LEVEL, "other_streams": [k for k in Ms if k != "primary"]},
           "checks": checks, "models": {}, "not_recomputed": {
               "R1": "the CGLS-disagreement order needs cgls_raw, which the committed units do not hold",
               "R4_interval": "the package's condition-design interval; only the estimate and the envelope here",
               "literal_audit": "the package's audit functions on one draw (report-only); not rerun here"}}
    for m in MODELS:
        out["models"][m] = model(U[m], m, B, Ms)
    v1, v12 = U["v1"], U["v12"]
    cov = B["cov"]
    out["models_same_class_share"] = float(np.mean(v1["pred"][cov] == v12["pred"][cov]))
    out["R4"] = r4(U, B, rec)
    out["gate"] = gate_records()
    out["verdicts_graded"] = {k: out["models"]["v12"]["verdicts"][k] for k in ("P1a", "P1b", "P1", "P2a", "P2b", "P2", "P3")}
    out["verdicts_v1_report_only"] = {k: out["models"]["v1"]["verdicts"][k] for k in ("P1a", "P1b", "P1", "P2a", "P2b", "P2", "P3")}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps({"graded": out["verdicts_graded"], "v1": out["verdicts_v1_report_only"]}))
    return out


if __name__ == "__main__":
    main()
