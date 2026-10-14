"""exp23: does reference instability explain the WorldCover-referenced wins?

Claim under test (README, exp18 reading). Against ESA WorldCover 2021,
tiling instability ranked the water head's disagreements better than the
model's own confidence on 26 of 27 rule-selected scenes (exp13), but lost on
hand-labelled masks (exp18). The repository reads the WorldCover wins as
detection of reference error, which is boundary-structured. That reading has
not been tested directly. This experiment tests it with a measurable
component of reference error: patches where the two published WorldCover
versions (2020 v100 and 2021 v200) disagree about water. Where the
reference product changes between its own versions, its label is unstable;
such patches are a lower bound on reference error (both versions can share
an error, and some version disagreements are genuine year-to-year change).

Pre-specified tests, on the exp13 scenes (rule-selected, at least 8
disagreements, georeferencing re-verified against the cached image):
  T1 enrichment: per scene, share of reference-unstable patches among the
     head's disagreements versus among its agreements; sign test across
     scenes on the difference.
  T2 mechanism: among disagreements, the share of reference-unstable
     patches in the top-k set ranked by tiling instability versus the top-k
     set ranked by confidence (k = number of disagreements); sign test on
     the per-scene difference. If tiling instability wins by finding
     reference error, its flagged disagreements should be unstable more
     often.
  T3 decisive: repeat the exp13 comparison of every signal against
     confidence with reference-unstable patches removed from scoring
     (scenes with at least 8 remaining disagreements). If the reading is
     right, tiling instability's advantage should shrink or reverse; if it
     persists unchanged, the reading is not supported by this component of
     reference error.

Signals are recomputed exactly as in exp13 from the cached features (seed-0
Base head trained on the Katima scene): confidence (negative absolute
logit), cross-model disagreement, aligned tiling instability, the boundary
indicator of exp14, embedding distance, and the no-model NDWI-gradient
control. Georeferencing is recovered as in exp15 and cached this time.

Outputs: exp/out/exp23_reference_instability.csv (per scene), exp23_summary.json,
exp23_reference_instability.png. Caches: exp/out/exp23_geo.npz (crs, transform,
WorldCover 2020 and 2021 pixel maps per scene; ignored by git).
"""
import csv
import json
import os
import sys

import numpy as np
import rasterio
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oe_inferencex.data import fetch_worldcover_window  # noqa: E402
from oe_inferencex.evidence import pool_to_patches, predict_head, predict_logit, train_logistic_head  # noqa: E402
from exp11_hardening import EXISTING, candidate_centers  # noqa: E402
from exp13_stat_corrections import GRID, NON_RULE, SHIFTS, aligned_tile_phase, eaurc, ndwi_gradient, sign_test_p  # noqa: E402
from exp15_boundary_geo import boundary_indicator, scene_georef  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SIZE, PATCH, PAD = 128, 4, 4
GEO_CACHE = os.path.join(OUT, "exp23_geo.npz")
MIN_ERR = 8
SIGS = ["E_case", "tile-phase (aligned)", "boundary", "E_dist", "control"]


def main():
    scenes = dict(np.load(os.path.join(OUT, "exp11_scenes.npz"), allow_pickle=True))
    feats = dict(np.load(os.path.join(OUT, "exp11_feats.npz"), allow_pickle=True))
    z3 = np.load(os.path.join(OUT, "exp03_cache.npz"))
    torch.manual_seed(0)
    hb = train_logistic_head(torch.tensor(feats["tr_base"]), z3["tr_labels"])
    hn = train_logistic_head(torch.tensor(feats["tr_nano"]), z3["tr_labels"])
    a_tr = torch.nn.functional.normalize(torch.tensor(feats["tr_base"]).reshape(-1, feats["tr_base"].shape[-1]), dim=-1)
    coords = dict(EXISTING)
    for name, lon, lat in candidate_centers():
        coords[name] = (lon, lat)
    geo = dict(np.load(GEO_CACHE, allow_pickle=True)) if os.path.exists(GEO_CACHE) else {}

    names = sorted({k.rsplit("_", 1)[0] for k in scenes if k.endswith("_img")})
    rows, per = [], {}
    for name in names:
        if f"{name}_base0" not in feats or name in NON_RULE or name not in coords:
            continue
        lab21_cached = scenes[f"{name}_lab"].astype(bool)
        p_shift = [predict_head(torch.tensor(feats[f"{name}_base{s}"]), *hb) for s in SHIFTS]
        p = p_shift[0]
        logit = predict_logit(torch.tensor(feats[f"{name}_base0"]), *hb)
        p_nano = predict_head(torch.tensor(feats[f"{name}_nano"]), *hn)
        err = ((p > 0.5) != lab21_cached)
        if err.sum() < MIN_ERR:
            continue
        # georeferencing and the two WorldCover versions on the scene grid
        if f"{name}_wc20" not in geo:
            try:
                g = scene_georef(*coords[name], scenes[f"{name}_img"])
            except Exception as exc:
                print(f"{name}: georef fetch failed ({type(exc).__name__}); excluded"); continue
            if g is None:
                print(f"{name}: re-read band does not match cache; excluded")
                geo[f"{name}_valid"] = np.array(0); np.savez(GEO_CACHE, **geo); continue
            crs_wkt, transform = g
            crs = rasterio.crs.CRS.from_wkt(crs_wkt)
            lon, lat = coords[name]
            wc20 = fetch_worldcover_window(lon, lat, crs, rasterio.Affine(*transform), SIZE + PAD, version="1.0.0")
            wc21 = fetch_worldcover_window(lon, lat, crs, rasterio.Affine(*transform), SIZE + PAD, version="2.0.0")
            geo[f"{name}_wc20"], geo[f"{name}_wc21"] = wc20, wc21
            geo[f"{name}_crs"], geo[f"{name}_transform"], geo[f"{name}_valid"] = np.array(crs_wkt), np.array(transform), np.array(1)
            np.savez(GEO_CACHE, **geo)
        if int(geo.get(f"{name}_valid", 0)) == 0:
            continue
        wc20, wc21 = geo[f"{name}_wc20"], geo[f"{name}_wc21"]
        lab20 = pool_to_patches(wc20[:SIZE, :SIZE] == 80, PATCH) > 0.5
        lab21 = pool_to_patches(wc21[:SIZE, :SIZE] == 80, PATCH) > 0.5
        if not np.array_equal(lab21, lab21_cached):
            print(f"{name}: re-fetched 2021 labels differ from the cache ({int((lab21 != lab21_cached).sum())} patches); using the cache")
            lab21 = lab21_cached
        unstable = lab20 != lab21
        f0 = feats[f"{name}_base0"]
        a_ev = torch.nn.functional.normalize(torch.tensor(f0).reshape(-1, f0.shape[-1]), dim=-1)
        knn = torch.topk(1 - (a_ev @ a_tr.T), k=5, largest=False).values.mean(1).reshape(GRID, GRID).numpy()
        sigs = {
            "baseline": -np.abs(logit),
            "E_case": np.abs(p_nano - p),
            "tile-phase (aligned)": aligned_tile_phase(p_shift),
            "boundary": boundary_indicator(p),
            "E_dist": knn,
            "control": ndwi_gradient(scenes[f"{name}_img"]),
        }
        e, u = err.flatten().astype(float), unstable.flatten()
        s = {k: v.flatten() for k, v in sigs.items()}
        n_err = int(e.sum())
        # T1 enrichment
        share_u_err = float(u[e > 0].mean())
        share_u_ok = float(u[e == 0].mean())
        # T2 mechanism: among disagreements, unstable share in each signal's top-k (k = n_err), most suspicious first
        def top_share(sig):
            order = np.argsort(sig, kind="stable")[::-1]
            flagged = order[:n_err]
            fe = flagged[e[flagged] > 0]
            return float(u[fe].mean()) if len(fe) else float("nan"), int(len(fe))
        t2 = {k: top_share(s[k]) for k in ("baseline", "tile-phase (aligned)", "boundary")}
        # T3 decisive: scoring restricted to reference-stable patches
        keep = ~u
        e_st = e[keep]
        n_err_st = int(e_st.sum())
        all_e = {k: eaurc(s[k], e) for k in s}
        st_e = {k: eaurc(s[k][keep], e_st) for k in s} if n_err_st >= MIN_ERR else None
        per[name] = {"all": all_e, "stable": st_e, "n_err": n_err, "n_err_stable": n_err_st}
        row = {"scene": name, "n_patches": int(e.size), "n_errors": n_err, "unstable_fraction": float(u.mean()),
               "unstable_share_among_errors": share_u_err, "unstable_share_among_correct": share_u_ok,
               "t2_unstable_share_topk_confidence": t2["baseline"][0], "t2_unstable_share_topk_tile": t2["tile-phase (aligned)"][0],
               "t2_unstable_share_topk_boundary": t2["boundary"][0], "t2_n_flagged_errors_confidence": t2["baseline"][1],
               "t2_n_flagged_errors_tile": t2["tile-phase (aligned)"][1], "n_errors_stable": n_err_st}
        for k in s:
            row[f"eaurc_all[{k}]"] = all_e[k]
            row[f"eaurc_stable[{k}]"] = st_e[k] if st_e else float("nan")
        rows.append(row)
        print(f"{name:14s} err {n_err:3d} | unstable {u.mean():.3f} | among err {share_u_err:.2f} vs ok {share_u_ok:.2f} | T2 top-k unstable share conf {t2['baseline'][0]:.2f} tile {t2['tile-phase (aligned)'][0]:.2f} | "
              f"stable err {n_err_st:3d} | tile-conf all {all_e['tile-phase (aligned)'] - all_e['baseline']:+.4f} stable {(st_e['tile-phase (aligned)'] - st_e['baseline']) if st_e else float('nan'):+.4f}")

    with open(os.path.join(OUT, "exp23_reference_instability.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    sn = sorted(per)
    summary = {"n_scenes": len(sn), "min_errors": MIN_ERR, "tests": {}}
    d1 = np.array([r["unstable_share_among_errors"] - r["unstable_share_among_correct"] for r in rows])
    w1, l1 = int((d1 > 1e-12).sum()), int((d1 < -1e-12).sum())
    summary["tests"]["T1_enrichment"] = {"median_share_errors": float(np.median([r["unstable_share_among_errors"] for r in rows])),
                                         "median_share_correct": float(np.median([r["unstable_share_among_correct"] for r in rows])),
                                         "pooled_share_errors": float(sum(r["unstable_share_among_errors"] * r["n_errors"] for r in rows) / sum(r["n_errors"] for r in rows)),
                                         "wins_losses_ties": [w1, l1, len(rows) - w1 - l1], "sign_p": sign_test_p(w1, w1 + l1) if w1 + l1 else 1.0}
    d2 = np.array([r["t2_unstable_share_topk_tile"] - r["t2_unstable_share_topk_confidence"] for r in rows if np.isfinite(r["t2_unstable_share_topk_tile"]) and np.isfinite(r["t2_unstable_share_topk_confidence"])])
    w2, l2 = int((d2 > 1e-12).sum()), int((d2 < -1e-12).sum())
    summary["tests"]["T2_mechanism_tile_vs_confidence"] = {"median_diff": float(np.median(d2)), "wins_losses_ties": [w2, l2, len(d2) - w2 - l2],
                                                           "sign_p": sign_test_p(w2, w2 + l2) if w2 + l2 else 1.0, "n_scenes": int(len(d2))}
    t3 = {}
    for k in SIGS:
        d_all = np.array([per[s_]["all"]["baseline"] - per[s_]["all"][k] for s_ in sn])  # >0 = signal better
        st = [s_ for s_ in sn if per[s_]["stable"] is not None]
        d_st = np.array([per[s_]["stable"]["baseline"] - per[s_]["stable"][k] for s_ in st])
        d_all_st = np.array([per[s_]["all"]["baseline"] - per[s_]["all"][k] for s_ in st])
        def wlt(d):
            w_, l_ = int((d > 1e-12).sum()), int((d < -1e-12).sum())
            return [w_, l_, len(d) - w_ - l_], (sign_test_p(w_, w_ + l_) if w_ + l_ else 1.0)
        (wa, pa), (ws, ps), (was, pas) = wlt(d_all), wlt(d_st), wlt(d_all_st)
        t3[k] = {"all_patches_all_scenes": {"wlt": wa, "sign_p": pa, "median_gain": float(np.median(d_all))},
                 "all_patches_same_scenes": {"wlt": was, "sign_p": pas, "median_gain": float(np.median(d_all_st)) if len(d_all_st) else None},
                 "stable_patches": {"wlt": ws, "sign_p": ps, "median_gain": float(np.median(d_st)) if len(d_st) else None, "n_scenes": len(st)}}
    summary["tests"]["T3_signal_vs_confidence"] = t3
    summary["scenes"] = {s_: {"n_err": per[s_]["n_err"], "n_err_stable": per[s_]["n_err_stable"]} for s_ in sn}
    with open(os.path.join(OUT, "exp23_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print(json.dumps(summary["tests"], indent=1, default=float))
    make_figure(rows, t3)


def make_figure(rows, t3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from oe_inferencex import figstyle
    figstyle.setup()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    ax = axes[0]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["unstable_share_among_errors"] for r in rows], 0.4, label="among disagreements", color="#d62728")
    ax.bar(x + 0.2, [r["unstable_share_among_correct"] for r in rows], 0.4, label="among agreements", color="#7f7f7f")
    ax.set_xticks(x); ax.set_xticklabels([r["scene"] for r in rows], rotation=90, fontsize=6)
    ax.set_ylabel("share of patches where WorldCover 2020 and 2021 disagree"); ax.set_title("(a) T1: reference instability by outcome"); ax.legend(fontsize=8)
    ax = axes[1]
    ax.scatter([r["t2_unstable_share_topk_confidence"] for r in rows], [r["t2_unstable_share_topk_tile"] for r in rows], s=18, color="#1f77b4")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
    ax.set_xlabel("unstable share of disagreements flagged by confidence (top-k)"); ax.set_ylabel("unstable share flagged by tiling instability (top-k)")
    ax.set_title("(b) T2: which flagged disagreements are reference-unstable")
    ax = axes[2]
    names = list(t3.keys())
    for i, k in enumerate(names):
        a, s_ = t3[k]["all_patches_same_scenes"], t3[k]["stable_patches"]
        ax.bar(i - 0.2, a["wlt"][0] / max(sum(a["wlt"]), 1), 0.4, color="#ff7f0e", label="all patches" if i == 0 else None)
        ax.bar(i + 0.2, s_["wlt"][0] / max(sum(s_["wlt"]), 1), 0.4, color="#2ca02c", label="reference-stable patches only" if i == 0 else None)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_ylabel("fraction of scenes where the signal beats confidence"); ax.set_title("(c) T3: signal vs confidence, same scenes"); ax.legend(fontsize=8)
    fig.suptitle("exp23: reference instability (WorldCover 2020 vs 2021) and the WorldCover-referenced comparisons", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp23_reference_instability.png"), dpi=150)
    print("figure written")


if __name__ == "__main__":
    main()
