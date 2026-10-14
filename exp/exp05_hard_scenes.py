"""Exp 05: conditions under which the confidence baseline degrades.

Heads trained at Katima Mulilo (reusing exp03's cached train window), then
evaluated on two deliberately hard windows:

  HARD - Barotse floodplain interior (wetland-margin water, exp04's weakest
         terrain type). In-domain biome, ambiguous margins.
  OOD  - Zambezi delta mangrove coast, Mozambique (turbid estuary water,
         mangrove canopy). Real domain shift ~1300 km from training.

Per AOI: baseline max-softmax, E_case |Nano-Base|, E_system tile-phase,
E_dist knn-to-train. AURC vs WorldCover water errors. Expectation from the
LLM literature: confidence degrades under distribution shift while the
evidence signals remain informative.
"""
import os

import numpy as np
import torch

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from oe_inferencex.data import fetch_s2_window, fetch_worldcover_window, s2_to_sample, embed
from oe_inferencex.evidence import train_logistic_head, predict_head, risk_coverage, pool_to_patches

AOIS = {
    "hard_barotse": (23.090, -15.400),
    "ood_delta": (36.180, -18.850),
}
SIZE, PATCH, PAD = 128, 4, 4
GRID = SIZE // PATCH
SHIFTS = (0, 1, 2, 3)
TR_CACHE = "exp/out/exp03_cache.npz"  # reuse Katima train window
CACHE = "exp/out/exp05_cache.npz"


def fetch_eval_aoi(name, lon, lat):
    img, date, (crs, transform) = fetch_s2_window(lon, lat, SIZE + PAD)
    wc = fetch_worldcover_window(lon, lat, crs, transform, SIZE + PAD)
    labels = (pool_to_patches(wc[:SIZE, :SIZE] == 80, PATCH) > 0.5).astype(np.float32)
    print(f"{name}: water patches {labels.sum():.0f}/{labels.size} ({100*labels.mean():.1f}%)")
    return img, np.array(date), labels


def main():
    torch.manual_seed(0)
    ztr = np.load(TR_CACHE)
    tr_img, tr_date, tr_labels = ztr["tr_img"], tuple(int(v) for v in ztr["tr_date"]), ztr["tr_labels"]

    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        data = {k: z[k] for k in z.files}
        print("loaded cached eval windows")
    else:
        data = {}
        for name, (lon, lat) in AOIS.items():
            img, date, labels = fetch_eval_aoi(name, lon, lat)
            data[f"{name}_img"], data[f"{name}_date"], data[f"{name}_lab"] = img, date, labels
        np.savez(CACHE, **data)

    models = {
        "nano": load_model_from_id(ModelID.OLMOEARTH_V1_NANO),
        "base": load_model_from_id(ModelID.OLMOEARTH_V1_BASE),
    }
    for m in models.values():
        m.eval()

    # train features + heads (shift 0 only)
    tr_sample = s2_to_sample(tr_img, *tr_date)
    heads, f_tr_base = {}, None
    for mn, model in models.items():
        f_tr = embed(model, tr_sample, PATCH)
        heads[mn] = train_logistic_head(f_tr, tr_labels)
        if mn == "base":
            f_tr_base = f_tr

    all_results = {}
    for name in AOIS:
        img = data[f"{name}_img"]
        date = tuple(int(v) for v in data[f"{name}_date"])
        labels = data[f"{name}_lab"]

        views = {s: img[:, s:s + SIZE, s:s + SIZE] for s in SHIFTS}
        p = {}
        for mn in ("nano", "base"):
            f = embed(models[mn], s2_to_sample(views[0], *date), PATCH)
            p[mn] = predict_head(f, *heads[mn])
            if mn == "base":
                f_ev_base = f
        acc = ((p["base"] > 0.5) == labels.astype(bool)).mean()
        errors = ((p["base"] > 0.5) != labels.astype(bool)).astype(np.float64)
        print(f"\n{name}: Base acc={acc:.3f} err={errors.mean():.3f} ({int(errors.sum())} patches)")

        signals = {
            "baseline max-softmax": 1 - np.maximum(p["base"], 1 - p["base"]),
            "E_case |Nano-Base|": np.abs(p["nano"] - p["base"]),
        }
        shift_p = [p["base"]]
        for s in SHIFTS[1:]:
            f = embed(models["base"], s2_to_sample(views[s], *date), PATCH)
            shift_p.append(predict_head(f, *heads["base"]))
        signals["E_system tile-phase"] = np.stack(shift_p).std(0)

        a = torch.nn.functional.normalize(f_ev_base.reshape(-1, f_ev_base.shape[-1]), dim=-1)
        b = torch.nn.functional.normalize(f_tr_base.reshape(-1, f_tr_base.shape[-1]), dim=-1)
        knn = torch.topk(1 - (a @ b.T), k=5, largest=False).values.mean(1)
        signals["E_dist knn-to-train"] = knn.reshape(GRID, GRID).numpy()

        res = {}
        for sn, sig in signals.items():
            cov, risk, aurc = risk_coverage(sig, errors)
            res[sn] = (cov, risk, aurc)
            print(f"  AURC {sn}: {aurc:.4f}")
        all_results[name] = (res, errors, p["base"], labels, signals)

    from oe_inferencex.figstyle import setup, map_panel, rc_panel
    import matplotlib.pyplot as plt
    setup()
    from olmoearth_pretrain.data.constants import Modality
    bo = Modality.SENTINEL2_L2A.band_order
    aoi_titles = {
        "hard_barotse": "Barotse floodplain interior\n(in-region, ambiguous wetland margins)",
        "ood_delta": "Zambezi delta mangrove coast\n(~1300 km from training region)",
    }
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.5))
    for row, name in enumerate(AOIS):
        res, errors, pb, labels, signals = all_results[name]
        img = data[f"{name}_img"]
        rgb = img[[bo.index(b) for b in ("B04", "B03", "B02")], :SIZE, :SIZE].transpose(1, 2, 0).astype(np.float32)
        k = row * 4
        map_panel(fig, axes[row, 0], np.clip((rgb - 1000) / 2000, 0, 1),
                  f"Sentinel-2 RGB, {name}", None, rgb=True, idx=k)
        map_panel(fig, axes[row, 1], labels, "ESA WorldCover 2021 water\n(reference)",
                  "water patch (fraction > 0.5)", cmap="Blues", idx=k + 1, vmin=0, vmax=1)
        map_panel(fig, axes[row, 2], errors, "Base head vs reference\n(disagreement, counted as error)",
                  "disagreement (binary)", cmap="Reds", idx=k + 2, vmin=0, vmax=1)
        rc_panel(axes[row, 3], res, aoi_titles[name], idx=k + 3)
    fig.suptitle("Water heads trained at Katima Mulilo, evaluated on two difficult scenes. "
                 "Reference labels are weak on these terrains (see docs/results/comparisons.md).", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("exp/out/exp05_hard_scenes.png", bbox_inches="tight")
    print("\nwrote exp/out/exp05_hard_scenes.png")


if __name__ == "__main__":
    main()
