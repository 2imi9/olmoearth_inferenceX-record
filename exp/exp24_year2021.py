"""exp24: does the imagery-to-reference year gap explain the WorldCover wins?

exp23 showed that WorldCover's own version instability does not explain why
tiling instability beats confidence against WorldCover but not against hand
labels. The next candidate is temporal mismatch: the reference is the 2021
map and the imagery was from June to September 2024, so genuine change
between the two years (river channels, floodplain water, clearing) counts
as model error and is boundary-structured. This experiment removes the gap.

Design (pre-specified). The same rule-selected scene centres as exp11/exp13
are re-fetched with Sentinel-2 L2A imagery from June to September 2021
(cloud under 5%, the least cloudy item), WorldCover 2021 v200 is warped onto
each 2021 window's own grid, features are computed exactly as in exp11
(v1-Base at grid shifts 0-3 px and v1-Nano, mean-pooled band-set tokens),
and the water head is retrained within the year on the Katima training
scene's 2021 imagery and 2021 labels (seed 0). Scenes with fewer than 8
disagreements are excluded, as before. Signals as in exp13/exp23:
confidence (negative absolute logit), cross-model disagreement, aligned
tiling instability, the boundary indicator, embedding distance to the 2021
training scene, and the NDWI-gradient control.

Primary test: per signal, wins/losses/ties against confidence in E-AURC
across the 2021 scenes with an exact sign test, side by side with the same
statistic for the 2024 imagery on the same scenes (recomputed from the exp11
cache with the 2024 head). If the year gap explains the 2024 wins, the 2021
tally for tiling instability should fall toward even; if it stays at the 2024
level, temporal mismatch is not the explanation. Secondary: the paired
per-scene change in tiling instability's E-AURC gain (2024 minus 2021), sign
test. Caveat: 2021 L2A products predate processing baseline 04.00 (which
added a 1000 DN offset in 2022), so the 2021 head is trained and evaluated
within that radiometry; cross-year application of a head is not the primary
analysis.

Outputs: exp/out/exp24_year2021.csv, exp24_summary.json, exp24_year2021.png.
Caches (ignored by git): exp/out/exp24_scenes.npz, exp/out/exp24_feats.npz.
"""
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id  # noqa: E402

from oe_inferencex.data import fetch_s2_window, fetch_worldcover_window  # noqa: E402
from oe_inferencex.evidence import pool_to_patches, predict_head, predict_logit, train_logistic_head  # noqa: E402
from exp11_hardening import EXISTING, candidate_centers, embed_gpu  # noqa: E402
from exp13_stat_corrections import GRID, NON_RULE, SHIFTS, aligned_tile_phase, eaurc, ndwi_gradient, sign_test_p  # noqa: E402
from exp15_boundary_geo import boundary_indicator  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SIZE, PATCH, PAD = 128, 4, 4
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN = (24.302, -17.485)  # Zambezi at Katima Mulilo, the exp02/exp03 training scene
YEAR = "2021-06-01/2021-09-30"
MIN_ERR = 8
SCENE_CACHE = os.path.join(OUT, "exp24_scenes.npz")
FEAT_CACHE = os.path.join(OUT, "exp24_feats.npz")
SIGS = ["E_case", "tile-phase (aligned)", "boundary", "E_dist", "control"]


def signals(feats, name, hb, hn, a_tr, img):
    p_shift = [predict_head(torch.tensor(feats[f"{name}_base{s}"]), *hb) for s in SHIFTS]
    p = p_shift[0]
    logit = predict_logit(torch.tensor(feats[f"{name}_base0"]), *hb)
    p_nano = predict_head(torch.tensor(feats[f"{name}_nano"]), *hn)
    f0 = feats[f"{name}_base0"]
    a_ev = torch.nn.functional.normalize(torch.tensor(f0).reshape(-1, f0.shape[-1]), dim=-1)
    knn = torch.topk(1 - (a_ev @ a_tr.T), k=5, largest=False).values.mean(1).reshape(GRID, GRID).numpy()
    return p, {
        "baseline": -np.abs(logit),
        "E_case": np.abs(p_nano - p),
        "tile-phase (aligned)": aligned_tile_phase(p_shift),
        "boundary": boundary_indicator(p),
        "E_dist": knn,
        "control": ndwi_gradient(img),
    }


def tally(per, sn, key):
    d = np.array([per[s]["baseline"] - per[s][key] for s in sn])  # >0 = signal better
    w, l = int((d > 1e-12).sum()), int((d < -1e-12).sum())
    return {"wlt": [w, l, len(d) - w - l], "sign_p": sign_test_p(w, w + l) if w + l else 1.0, "median_gain": float(np.median(d)) if len(d) else None}


def main():
    scenes24 = dict(np.load(os.path.join(OUT, "exp11_scenes.npz"), allow_pickle=True))
    feats24 = dict(np.load(os.path.join(OUT, "exp11_feats.npz"), allow_pickle=True))
    z3 = np.load(os.path.join(OUT, "exp03_cache.npz"))
    coords = dict(EXISTING)
    for name, lon, lat in candidate_centers():
        coords[name] = (lon, lat)
    names = sorted(k.rsplit("_", 1)[0] for k in feats24 if k.endswith("_base0") and k.rsplit("_", 1)[0] not in NON_RULE and k.rsplit("_", 1)[0] in coords)

    # 1. 2021 imagery and labels on their own grids
    scenes = dict(np.load(SCENE_CACHE, allow_pickle=True)) if os.path.exists(SCENE_CACHE) else {}
    for name, (lon, lat) in [("tr", TRAIN)] + [(n_, coords[n_]) for n_ in names]:
        if f"{name}_img" in scenes:
            continue
        try:
            img, date, (crs, transform) = fetch_s2_window(lon, lat, SIZE + PAD, datetime=YEAR)
            wc = fetch_worldcover_window(lon, lat, crs, transform, SIZE + PAD, version="2.0.0")
        except Exception as exc:
            print(f"{name}: 2021 fetch failed ({type(exc).__name__}: {exc})"); continue
        scenes[f"{name}_img"], scenes[f"{name}_date"] = img, np.array(date)
        scenes[f"{name}_lab"] = (pool_to_patches(wc[:SIZE, :SIZE] == 80, PATCH) > 0.5).astype(np.float32)
        np.savez(SCENE_CACHE, **scenes)
        print(f"{name}: 2021 scene cached (date {date})")

    # 2. features as in exp11
    models = {k: load_model_from_id(m).to(DEV).eval() for k, m in (("nano", ModelID.OLMOEARTH_V1_NANO), ("base", ModelID.OLMOEARTH_V1_BASE))}
    feats = dict(np.load(FEAT_CACHE, allow_pickle=True)) if os.path.exists(FEAT_CACHE) else {}
    for name in ["tr"] + names:
        if f"{name}_img" not in scenes or f"{name}_base0" in feats:
            continue
        img, date = scenes[f"{name}_img"], tuple(int(v) for v in scenes[f"{name}_date"])
        if name == "tr":
            feats["tr_base"] = embed_gpu(models["base"], img[:, :SIZE, :SIZE], date)
            feats["tr_nano"] = embed_gpu(models["nano"], img[:, :SIZE, :SIZE], date)
            feats["tr_base0"] = feats["tr_base"]
        else:
            for s in SHIFTS:
                feats[f"{name}_base{s}"] = embed_gpu(models["base"], img[:, s:s + SIZE, s:s + SIZE], date)
            feats[f"{name}_nano"] = embed_gpu(models["nano"], img[:, :SIZE, :SIZE], date)
        np.savez(FEAT_CACHE, **feats)
        print(f"{name}: 2021 features cached")

    # 3. heads: 2021 within-year (primary) and the 2024 head of exp13 (for the 2024 comparison)
    torch.manual_seed(0)
    hb21 = train_logistic_head(torch.tensor(feats["tr_base"]), scenes["tr_lab"])
    hn21 = train_logistic_head(torch.tensor(feats["tr_nano"]), scenes["tr_lab"])
    torch.manual_seed(0)
    hb24 = train_logistic_head(torch.tensor(feats24["tr_base"]), z3["tr_labels"])
    hn24 = train_logistic_head(torch.tensor(feats24["tr_nano"]), z3["tr_labels"])
    a_tr21 = torch.nn.functional.normalize(torch.tensor(feats["tr_base"]).reshape(-1, feats["tr_base"].shape[-1]), dim=-1)
    a_tr24 = torch.nn.functional.normalize(torch.tensor(feats24["tr_base"]).reshape(-1, feats24["tr_base"].shape[-1]), dim=-1)
    tr_acc21 = float(((predict_head(torch.tensor(feats["tr_base"]), *hb21) > 0.5) == scenes["tr_lab"].astype(bool)).mean())

    rows, per21, per24 = [], {}, {}
    for name in names:
        if f"{name}_base0" not in feats:
            continue
        lab21 = scenes[f"{name}_lab"].astype(bool)
        p21, s21 = signals(feats, name, hb21, hn21, a_tr21, scenes[f"{name}_img"])
        err21 = ((p21 > 0.5) != lab21).astype(float).flatten()
        lab24 = scenes24[f"{name}_lab"].astype(bool)
        p24, s24 = signals(feats24, name, hb24, hn24, a_tr24, scenes24[f"{name}_img"])
        err24 = ((p24 > 0.5) != lab24).astype(float).flatten()
        row = {"scene": name, "n_errors_2021": int(err21.sum()), "n_errors_2024": int(err24.sum()),
               "water_fraction_ref_2021grid": float(lab21.mean()), "water_fraction_pred_2021": float((p21 > 0.5).mean()),
               "date_2021": "-".join(str(int(v)) for v in scenes[f"{name}_date"][::-1])}
        if err21.sum() >= MIN_ERR:
            per21[name] = {k: eaurc(v.flatten(), err21) for k, v in s21.items()}
            for k in per21[name]:
                row[f"eaurc_2021[{k}]"] = per21[name][k]
        if err24.sum() >= MIN_ERR:
            per24[name] = {k: eaurc(v.flatten(), err24) for k, v in s24.items()}
            for k in per24[name]:
                row[f"eaurc_2024[{k}]"] = per24[name][k]
        rows.append(row)
        g21 = (per21[name]["baseline"] - per21[name]["tile-phase (aligned)"]) if name in per21 else float("nan")
        g24 = (per24[name]["baseline"] - per24[name]["tile-phase (aligned)"]) if name in per24 else float("nan")
        print(f"{name:14s} err 2021 {int(err21.sum()):3d} / 2024 {int(err24.sum()):3d} | tile gain over confidence: 2021 {g21:+.4f}  2024 {g24:+.4f}")

    with open(os.path.join(OUT, "exp24_year2021.csv"), "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "scene", k))
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    sn21, sn24 = sorted(per21), sorted(per24)
    both = sorted(set(sn21) & set(sn24))
    summary = {"n_scenes_2021": len(sn21), "n_scenes_2024": len(sn24), "n_scenes_both": len(both),
               "train_scene_accuracy_2021": tr_acc21, "signal_vs_confidence": {}}
    for k in SIGS:
        summary["signal_vs_confidence"][k] = {"2021": tally(per21, sn21, k), "2024_same_scenes": tally(per24, both, k), "2021_same_scenes": tally(per21, both, k)}
    d = np.array([(per24[s]["baseline"] - per24[s]["tile-phase (aligned)"]) - (per21[s]["baseline"] - per21[s]["tile-phase (aligned)"]) for s in both])
    w, l = int((d > 1e-12).sum()), int((d < -1e-12).sum())
    summary["tile_gain_2024_minus_2021"] = {"wlt_gain_larger_in_2024": [w, l, len(d) - w - l], "sign_p": sign_test_p(w, w + l) if w + l else 1.0,
                                            "median_diff": float(np.median(d)) if len(d) else None}
    with open(os.path.join(OUT, "exp24_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))
    make_figure(per21, per24, both, summary)


def make_figure(per21, per24, both, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from oe_inferencex import figstyle
    figstyle.setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for i, k in enumerate(SIGS):
        t24, t21 = summary["signal_vs_confidence"][k]["2024_same_scenes"], summary["signal_vs_confidence"][k]["2021_same_scenes"]
        ax.bar(i - 0.2, t24["wlt"][0] / max(sum(t24["wlt"]), 1), 0.4, color="#ff7f0e", label="2024 imagery (as exp13)" if i == 0 else None)
        ax.bar(i + 0.2, t21["wlt"][0] / max(sum(t21["wlt"]), 1), 0.4, color="#1f77b4", label="2021 imagery (WorldCover's year)" if i == 0 else None)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(SIGS))); ax.set_xticklabels(SIGS, rotation=20, fontsize=8)
    ax.set_ylabel("fraction of scenes where the signal beats confidence"); ax.set_title(f"(a) signal vs confidence, {len(both)} scenes scored in both years"); ax.legend(fontsize=8)
    ax = axes[1]
    g24 = [per24[s]["baseline"] - per24[s]["tile-phase (aligned)"] for s in both]
    g21 = [per21[s]["baseline"] - per21[s]["tile-phase (aligned)"] for s in both]
    ax.scatter(g24, g21, s=18, color="#1f77b4")
    lim = max(abs(v) for v in g24 + g21) * 1.05 if both else 1
    ax.plot([-lim, lim], [-lim, lim], "--", color="grey", linewidth=0.8); ax.axhline(0, color="grey", linewidth=0.5); ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("tiling-instability E-AURC gain over confidence, 2024 imagery"); ax.set_ylabel("same gain, 2021 imagery")
    ax.set_title("(b) per-scene gain, 2024 vs 2021 (positive = beats confidence)")
    fig.suptitle("exp24: the WorldCover-referenced comparison with imagery from WorldCover's own year", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp24_year2021.png"), dpi=150)
    print("figure written")


if __name__ == "__main__":
    main()
