"""exp111 (docs/plan/blind_transfer.md), report-only sizing, written before the phase-1 commit. Uses the 271 seen plots
(exp99's v1 and exp103's v1.2 units files) only to carry exp109's per-label behaviour to a fresh label-stratified draw
from the Kenya/Tanzania 2017 pool outside exp99's 309, and Ai2's 259 validation points (exp98's and exp103's units) to
calibrate the pipeline gate. Counts and rates only; no identifier, no position. POOL_2017 is the count of TimeSync 2017
plots by country and label (the public file; the cluster's select step recomputes the pool and refuses unless it gives
exp111_blind_transfer.POOL_CELLS). Writes exp/out/exp111_power.json:
    uv run --frozen python exp/exp111_power.py
"""
import json
import os
import sys
import warnings
from math import comb

import numpy as np

REC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(REC, "exp", "out", "exp111_power.json")
POOL_2017 = {"kenya|Cropland": 217, "kenya|Dense Forest": 76, "kenya|Open Forest": 62, "kenya|Open Grassland": 447,
             "kenya|Open Water": 37, "kenya|Otherland": 283, "kenya|Settlements": 8, "kenya|Vegetated Wetland": 5,
             "kenya|Wooded Grassland": 865, "tanzania|Cropland": 391, "tanzania|Dense Forest": 95,
             "tanzania|Open Forest": 199, "tanzania|Open Grassland": 405, "tanzania|Open Water": 104,
             "tanzania|Otherland": 7, "tanzania|Settlements": 6, "tanzania|Vegetated Wetland": 32,
             "tanzania|Wooded Grassland": 761}
from oe_inferencex.estimate import hypergeom_interval, interval_verdict   # noqa: E402

UNITS = {"v1": "exp/out/exp99_units.npz", "v1.2": "exp/out/exp103/transfer2017/exp99_units.npz"}
VAL = {"v1": "exp/out/exp98_units.npz", "v1.2": "exp/out/exp103/awf2023/exp98_units.npz"}
CUT = {"v1": 0.99875695977, "v1.2": 0.99888118198}       # exp109's confident-third cut at the 271, 11 digits, rounded down
SHRUB, GRASS = 2, 4
GROUPS = ("O", "W", "X")
R = 2000
rng = np.random.default_rng(111)


def wauc(s, pos, w):
    """Weighted AUROC of score s (higher = more suspicious) for pos against ~pos; ties count half."""
    s, pos, w = np.asarray(s, float), np.asarray(pos, bool), np.asarray(w, float)
    if pos.sum() == 0 or (~pos).sum() == 0:
        return np.nan
    uniq, inv = np.unique(s, return_inverse=True)        # ascending score; ties grouped
    wp = np.bincount(inv, weights=w * pos, minlength=uniq.size)
    wn = np.bincount(inv, weights=w * ~pos, minlength=uniq.size)
    cum_n = np.cumsum(wn) - wn
    num = (wp * (cum_n + 0.5 * wn)).sum()
    return float(num / (wp.sum() * wn.sum()))


def load():
    z = {m: np.load(os.path.join(REC, p)) for m, p in UNITS.items()}
    a, b = z["v1"], z["v1.2"]
    assert (a["label"] == b["label"]).all() and (a["country"] == b["country"]).all()
    names = a["label_names"]
    lab = names[a["label"]]
    grp = np.where(lab == "Open Grassland", "O", np.where(lab == "Wooded Grassland", "W", "X"))
    ok = a["covered"] & b["covered"] & (a["pred"] >= 0) & (b["pred"] >= 0)
    return z, lab, grp, ok


def stats(m, d, idx, w_region, w_std):
    """The arms' statistics for one model on rows idx with the two weightings."""
    u = d["suspicion"][idx]
    p1 = d["p1"][idx]
    pred = d["pred"][idx]
    lab = d["lab"][idx]
    ws, wl = d["wrong_strict"][idx] > 0, d["wrong_lenient"][idx] > 0
    so = ws & ~wl
    conf = p1 >= CUT[m]
    pair = (pred == SHRUB) & (lab == "Open Grassland")
    out = {}
    out["auc_strict"] = wauc(u, ws, w_region)
    out["auc_lenient"] = wauc(u, wl, w_region)
    ce = conf & ws
    out["pair_share_std"] = float((w_std * (ce & pair)).sum() / (w_std * ce).sum()) if ce.any() else np.nan
    out["pair_share_region"] = float((w_region * (ce & pair)).sum() / (w_region * ce).sum()) if ce.any() else np.nan
    out["strict_only_share_std"] = float((w_std * (ce & so)).sum() / (w_std * ce).sum()) if ce.any() else np.nan
    keep_so = so | ~ws
    keep_l = wl | ~ws
    out["auc_so_vs_correct_std"] = wauc(u[keep_so], so[keep_so], w_std[keep_so])
    out["auc_len_vs_correct_std"] = wauc(u[keep_l], wl[keep_l], w_std[keep_l])
    out["gap_std"] = out["auc_len_vs_correct_std"] - out["auc_so_vs_correct_std"]
    zone = conf & (pred == SHRUB)
    if zone.any():
        wz = w_region[zone]
        out["zone_eS"] = float((wz * ws[zone]).sum() / wz.sum())
        out["zone_eL"] = float((wz * wl[zone]).sum() / wz.sum())
        out["zone_plots"] = int(zone.sum())
    else:
        out["zone_eS"] = out["zone_eL"] = np.nan
        out["zone_plots"] = 0
    out["n_conf_strict_err"] = int(ce.sum())
    out["n_strict"], out["n_lenient"] = int(ws.sum()), int(wl.sum())
    return out


def p_note(eS, eL, n, kstar):
    """P(the audit's note fires): k_S >= kstar and k_L < kstar, n draws from a zone with LENIENT rate eL and STRICT rate
    eS (eL <= eS), binomial limit (the region's zone is large)."""
    pL, pSO = eL, max(eS - eL, 0.0)
    pR = max(1 - pL - pSO, 0.0)
    tot = 0.0
    for kL in range(0, kstar):
        for kSO in range(0, n - kL + 1):
            if kL + kSO < kstar:
                continue
            kR = n - kL - kSO
            tot += comb(n, kL) * comb(n - kL, kSO) * pL ** kL * pSO ** kSO * pR ** kR
    return tot


def kstar(n, alpha, N=10 ** 6):
    for k in range(n + 1):
        if interval_verdict(*hypergeom_interval(k, n, N, 0.95), alpha) == "above":
            return k
    return None


def main():
    warnings.simplefilter("ignore", RuntimeWarning)
    z, lab, grp, ok = load()
    pool = POOL_2017
    cn = z["v1"]["country_names"]
    ctry = cn[z["v1"]["country"]]
    # the pool by label group: K/T 2017 minus the 309, by country
    poolg = {}
    for c in ("kenya", "tanzania"):
        for g in GROUPS:
            tot = sum(v for k, v in pool.items() if k.startswith(c + "|") and
                      (("O" if k.endswith("Open Grassland") else "W" if k.endswith("Wooded Grassland") else "X") == g))
            poolg[(c, g)] = tot - int(((ctry == c) & (grp == g)).sum())
    poolG = {g: poolg[("kenya", g)] + poolg[("tanzania", g)] for g in GROUPS}
    n_ok = {g: int((ok & (grp == g)).sum()) for g in GROUPS}
    n_all = {g: int((grp == g).sum()) for g in GROUPS}
    input_share = {g: n_ok[g] / n_all[g] for g in GROUPS}
    std_share = {g: n_ok[g] / sum(n_ok.values()) for g in GROUPS}
    data = {}
    for m, d in z.items():
        data[m] = {k: d[k] for k in ("suspicion", "p1", "pred", "wrong_strict", "wrong_lenient", "weight")}
        data[m]["lab"] = lab
    res = {"pool_by_country_group": {f"{c}|{g}": v for (c, g), v in poolg.items()}, "pool_by_group": poolG,
           "seen_271": {"with_input_by_group": n_ok, "all_by_group": n_all,
                        "input_share": {g: round(v, 3) for g, v in input_share.items()},
                        "group_share_with_input": {g: round(v, 3) for g, v in std_share.items()}}}
    idx = np.flatnonzero(ok)
    # seen values: (a) exp99's own weights; (b) label groups reweighted to the pool's mix (the carry-over prediction)
    for m in data:
        w99 = data[m]["weight"][idx]
        g = grp[idx]
        w_pool = np.array([poolG[x] / n_ok[x] for x in g])
        w_std = np.array([std_share[x] / n_ok[x] for x in g])
        res["seen_271"][m] = {"exp99_weights": stats(m, data[m], idx, w99, w99),
                              "pool_mix": stats(m, data[m], idx, w_pool, w_std)}
    # the audit's thresholds
    ks = {f"n{n}_a{a}": kstar(n, a) for n in (15, 20, 30) for a in (0.05, 0.10)}
    res["audit_kstar"] = ks
    for m in data:
        for tag in ("exp99_weights", "pool_mix"):
            s = res["seen_271"][m][tag]
            s["p_note"] = {k: round(p_note(s["zone_eS"], s["zone_eL"], int(k.split("_")[0][1:]), v), 3)
                           for k, v in ks.items()}
    # simulation: fresh draws, groups resampled from the 271 (paired across models)
    designs = {"300/300/150": (300, 300, 150), "200/200/100": (200, 200, 100), "400/400/200": (400, 400, 200),
               "300/300/300": (300, 300, 300)}
    scen = {"exp99_input_loss": input_share, "loss_3pct": {g: 0.97 for g in GROUPS}}
    sims = {}
    for dn, drawn in designs.items():
        for sn, ish in scen.items():
            if sn == "loss_3pct" and dn != "300/300/150":
                continue
            m_g = {g: int(round(dr * ish[g])) for g, dr in zip(GROUPS, drawn)}
            rows = {g: idx[grp[idx] == g] for g in GROUPS}
            acc = {m: [] for m in data}
            for r in range(R):
                pick = np.concatenate([rng.choice(rows[g], m_g[g], replace=True) for g in GROUPS])
                gg = grp[pick]
                w_pool = np.array([poolG[x] / m_g[x] for x in gg])
                w_std = np.array([std_share[x] / m_g[x] for x in gg])
                for m in data:
                    acc[m].append(stats(m, data[m], pick, w_pool, w_std))
            out = {"with_input": m_g}
            for m in data:
                A = {k: np.array([a[k] for a in acc[m]], float) for k in acc[m][0]}
                sd = {k: float(np.nanstd(v)) for k, v in A.items()}
                mean = {k: float(np.nanmean(v)) for k, v in A.items()}
                o = {"mean": {k: round(v, 4) for k, v in mean.items()}, "sd": {k: round(v, 4) for k, v in sd.items()}}
                aS, aL = A["auc_strict"], A["auc_lenient"]
                o["P1_strict_point_below_0.70"] = round(float(np.mean(aS < 0.70)), 3)
                # the lower side (review of 15 Oct): STRICT's 90% interval above 0.50, so that a noise-level ranking
                # does not pass P1(a)
                p1a = (aS < 0.70) & (aS - 1.645 * sd["auc_strict"] > 0.50)
                o["P1a_below_0.70_and_lower90_above_0.50"] = round(float(np.mean(p1a)), 3)
                o["P1a_and_P1b_with_lower_side"] = round(float(np.mean(p1a & ((aL - aS) - 1.645 * np.nanstd(aL - aS) > 0))), 3)
                o["P1_strict_upper90_below_0.70"] = round(float(np.mean(aS + 1.645 * sd["auc_strict"] < 0.70)), 3)
                o["P1_lenient_point_ge_0.70"] = round(float(np.mean(aL >= 0.70)), 3)
                o["P1_lenient_call_above"] = round(float(np.mean(aL - 1.645 * sd["auc_lenient"] > 0.70)), 3)
                o["P1_lenient_call_below"] = round(float(np.mean(aL + 1.645 * sd["auc_lenient"] < 0.70)), 3)
                o["P1_paired_diff_lenient_minus_strict_gt0_lower90"] = round(float(np.mean(
                    (aL - aS) - 1.645 * np.nanstd(aL - aS) > 0)), 3)
                o["P2a_pair_share_std_ge_0.5"] = round(float(np.nanmean(A["pair_share_std"] >= 0.5)), 3)
                o["P2_all_three"] = round(float(np.nanmean((A["pair_share_std"] >= 0.5) & (A["gap_std"] - 1.645 * sd["gap_std"] > 0) & (A["auc_so_vs_correct_std"] <= 0.65))), 3)
                # the lower side of P2(b) (review of 15 Oct): AUROC(LENIENT errors vs STRICT-correct) at least 0.65
                p2b = (A["gap_std"] - 1.645 * sd["gap_std"] > 0) & (A["auc_so_vs_correct_std"] <= 0.65) & \
                    (A["auc_len_vs_correct_std"] >= 0.65)
                o["P2b_lenient_point_ge_0.65"] = round(float(np.nanmean(A["auc_len_vs_correct_std"] >= 0.65)), 3)
                o["P2b_with_lower_side"] = round(float(np.nanmean(p2b)), 3)
                o["P2_with_lower_side"] = round(float(np.nanmean((A["pair_share_std"] >= 0.5) & p2b)), 3)
                o["P1_strict_below_and_diff_gt0"] = round(float(np.mean((aS < 0.70) & ((aL - aS) - 1.645 * np.nanstd(aL - aS) > 0))), 3)
                o["sd_diff_lenient_minus_strict"] = round(float(np.nanstd(aL - aS)), 4)
                o["mean_diff_lenient_minus_strict"] = round(float(np.nanmean(aL - aS)), 4)
                o["P2a_strict_only_share_std_ge_0.5"] = round(float(np.nanmean(A["strict_only_share_std"] >= 0.5)), 3)
                o["P2b_gap_lower90_gt0"] = round(float(np.nanmean(A["gap_std"] - 1.645 * sd["gap_std"] > 0)), 3)
                o["P2b_gap_point_ge_0.05"] = round(float(np.nanmean(A["gap_std"] >= 0.05)), 3)
                o["P2b_so_auc_point_le_0.65"] = round(float(np.nanmean(A["auc_so_vs_correct_std"] <= 0.65)), 3)
                for k, v in ks.items():
                    n = int(k.split("_")[0][1:])
                    pn = np.array([p_note(a["zone_eS"], a["zone_eL"], n, v) for a in acc[m][:400]])
                    o[f"P3_mean_p_note_{k}"] = round(float(np.nanmean(pn)), 3)
                    o[f"P3_p_note_ge_0.8_{k}"] = round(float(np.nanmean(pn >= 0.8)), 3)
                    o[f"P3_p_note_ge_0.5_{k}"] = round(float(np.nanmean(pn >= 0.5)), 3)
                o["P3_zone_plots_mean"] = round(mean["zone_plots"], 1)
                out[m] = o
            # joint: both models
            sims[f"{dn}|{sn}"] = out
    res["simulation"] = {"replicates": R, "seed": 111, "note": "groups resampled with replacement from the 271 seen "
                         "plots with input; region weights pool_g/m_g; P2 weights standardised to the 271's group mix",
                         "designs": sims}
    # gate calibration at Ai2's validation points and the 271
    v = {m: np.load(os.path.join(REC, p)) for m, p in VAL.items()}
    gate = {}
    val = (v["v1"]["split"] == "val") & v["v1"]["covered"] & v["v1.2"]["covered"]
    for m in v:
        d = v[m]
        acc_ = float((d["pred_deployed"][val] == d["label"][val]).mean())
        gate[m] = {"val_with_input": int(val.sum()), "accuracy": round(acc_, 4),
                   "share_p1_below_0.6": round(float((d["p1"][val] < 0.6).mean()), 4),
                   "share_p1_below_0.8": round(float((d["p1"][val] < 0.8).mean()), 4)}
    gate["v1_v12_same_class_val"] = round(float((v["v1"]["pred_deployed"][val] == v["v1.2"]["pred_deployed"][val]).mean()), 4)
    gate["v1_v12_same_class_271"] = round(float((z["v1"]["pred"][idx] == z["v1.2"]["pred"][idx]).mean()), 4)
    for m in z:
        gate[m]["271_share_p1_below_0.6"] = round(float((z[m]["p1"][idx] < 0.6).mean()), 4)
        gate[m]["271_share_p1_below_0.8"] = round(float((z[m]["p1"][idx] < 0.8).mean()), 4)
    res["gate_calibration"] = gate
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
