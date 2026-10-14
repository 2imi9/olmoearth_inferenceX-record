"""exp20: first assessment of a real OlmoEarth production output.

Reads windows of the published land cover change rasters (allenai/olmoearth_lcc;
9-band uint8 BigTIFFs served in EPSG:3857 at ~9.55 m; encoder v1.2-Base
fine-tuned) at ten sites in the Zambezi/Chobe/Barotse region and applies the
recipe (docs/method/recipe.md) to what the product exports. Per the dataset
card, band 1 is the change probability, bands 4-5 the source/destination land
cover classes, and bands 6-7 the probabilities of the *change-category* heads
(bands 2-3). No confidence is exported for the land cover classes.

What is measured
  A. Land cover class map (band 4) - no exported confidence, so only the
     boundary cue of the recipe is available. Against ESA WorldCover 2021 as a
     weak reference: full-legend disagreement (context only; legends and dates
     differ) and the water task, where the legends coincide: tie-aware AURC of
     boundary fraction against random and oracle, capture at review budgets,
     boundary share among disagreements versus agreements.
  B. Change prediction (band 1) - the product's own probability. Label-free:
     saturation and tie structure, flagged fraction, confidence at flagged-mask
     boundaries versus interiors, and the source->destination transitions
     predicted where change is flagged (a sanity check, not an accuracy).
  C. Change-category heads (bands 2-3 with 6-7), read only where band 1 >= 128
     as the card advises: distinct values and saturation of the exported
     probabilities.

Nothing is inferred or fine-tuned here; this is the served product. Outputs:
exp/out/exp20_lcc_production.{csv,json}, exp/out/exp20_lcc_kazungula.png, and
the window cache exp/out/exp20_windows.npz (ignored by git).
"""
import csv
import json
import os
import sys
import time

import numpy as np
from rasterio.transform import Affine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oe_inferencex import lcc  # noqa: E402
from oe_inferencex.assess import _boundary, _pool, _pooled_argmax, assess_classmap  # noqa: E402
from oe_inferencex.data import fetch_worldcover_window  # noqa: E402
from oe_inferencex.evidence import aurc_expected  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SIZE = 512          # pixels at ~9.55 m: about 4.9 km
PATCH = 4
BUDGETS = (0.01, 0.05, 0.10)
SITES = {  # lon, lat; all inside published tiles (checked by HEAD request)
    "kazungula": (25.263, -17.788), "kasane_chobe": (25.15, -17.82), "ngoma_chobe": (24.45, -17.92),
    "linyanti": (23.9, -18.3), "savuti": (24.05, -18.57), "mababe": (24.1, -18.9), "kachikau": (24.75, -18.17),
    "barotse_mongu": (23.13, -15.25), "katima": (24.27, -17.50), "zambezi_sesheke": (24.30, -17.47),
}
# LCC land cover legend -> WorldCover 2021 class, for the context comparison only.
LCC_TO_WC = {9: 10, 7: 20, 5: 30, 3: 40, 4: 40, 10: 50, 1: 60, 8: 70, 11: 80, 12: 90, 6: 100}
WC_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
WC_INDEX = {c: i for i, c in enumerate(WC_CLASSES)}
PRE_CATEGORY = {0: "no prediction", 1: "none", 2: "deforestation", 3: "urban_erosion", 4: "wetland_loss", 5: "water_contract",
                6: "removed_crop_structure", 7: "agricultural_activity", 8: "wildfire", 9: "ice_motion", 10: "flooding"}
POST_CATEGORY = {0: "no prediction", 1: "none", 2: "vegetation_growth", 3: "new_building", 4: "new_road", 5: "new_infrastructure",
                 6: "new_crop_field", 7: "new_aquafarm", 8: "site_clearing", 9: "water_expand", 10: "mining", 11: "new_crop_structure",
                 12: "selective_logging", 13: "landslide", 14: "settlement"}


def to_index(lcc_class):
    idx = np.full(lcc_class.shape, -1, dtype=int)
    for k, wc in LCC_TO_WC.items():
        idx[lcc_class == k] = WC_INDEX[wc]
    return idx


def wc_to_index(wc):
    idx = np.full(wc.shape, -1, dtype=int)
    for c, i in WC_INDEX.items():
        idx[wc == c] = i
    return idx


def load_windows():
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(OUT, "exp20_windows.npz")
    windows = dict(np.load(cache, allow_pickle=True)) if os.path.exists(cache) else {}
    for site, (lon, lat) in SITES.items():
        if f"{site}_lcc" in windows:
            continue
        name = lcc.tile_for(lon, lat)[0]
        arr, transform, epsg, _ = lcc.read_window(lon, lat, SIZE, tile=name)
        wc = fetch_worldcover_window(lon, lat, f"EPSG:{epsg}", Affine(*transform), SIZE)
        windows[f"{site}_lcc"], windows[f"{site}_wc"] = arr, wc
        windows[f"{site}_meta"] = np.array(json.dumps({"tile": name, "epsg": epsg, "transform": list(transform)}))
        np.savez_compressed(cache, **windows)
    return windows


def water_task(arr, wc, nodata):
    """Binary water map from band 4 against WorldCover water; boundary is the only
    label-free cue the product allows. Returns the assessment and per-window arrays."""
    hard = (arr[3] == 11).astype(int)
    ref = np.where(wc == 0, -1, (wc == 80).astype(int))
    excluded = nodata | (wc == 0)
    a = assess_classmap(hard, np.zeros(hard.shape), 2, patch=PATCH, nodata_mask=excluded, reference=ref, budgets=BUDGETS,
                        signal="none exported (constant); boundary fraction is the ranker")
    A = a["arrays"]
    valid = A["valid"]
    err = ((A["pooled_argmax"] != _pooled_argmax(ref, 2, PATCH)) & valid)
    e, b = err[valid].astype(float), A["boundary"][valid]
    r = {
        "water_error_rate": float(e.mean()), "water_n_windows": int(valid.sum()), "water_share_windows": float((A["pooled_argmax"][valid] == 1).mean()),
        "water_aurc_boundary": aurc_expected(b, e), "water_aurc_random": float(e.mean()), "water_aurc_oracle": aurc_expected(e, e),
        "water_boundary_share_errors": float((b[e > 0] > 0).mean()) if e.sum() else float("nan"),
        "water_boundary_share_correct": float((b[e == 0] > 0).mean()),
    }
    order = np.argsort(b, kind="stable")[::-1]
    for bud in BUDGETS:
        k = max(1, int(round(bud * len(e))))
        r[f"water_capture@{bud}"] = float(e[order[:k]].sum() / max(e.sum(), 1))
        r[f"water_precision@{bud}"] = float(e[order[:k]].mean())
    return r, A, err


def main():
    windows = load_windows()
    rows, full = [], {}
    for site, (lon, lat) in SITES.items():
        t0 = time.time()
        arr, wc = windows[f"{site}_lcc"], windows[f"{site}_wc"]
        meta = json.loads(str(windows[f"{site}_meta"]))
        nodata = lcc.nodata_mask(arr)
        row = {"site": site, "lon": lon, "lat": lat, "tile": meta["tile"], "size_px": SIZE, "nodata_fraction": float(nodata.mean())}

        # A. land cover map: boundary structure and the weak-reference context
        hard = to_index(arr[3].astype(int)); ref = wc_to_index(wc)
        ok = (hard >= 0) & (ref >= 0) & ~nodata
        pooled_hard = _pooled_argmax(np.where(hard < 0, 0, hard), len(WC_CLASSES), PATCH)
        bnd = _boundary(pooled_hard)
        row.update({
            "lc_full_legend_disagreement": float((hard[ok] != ref[ok]).mean()),
            "lc_boundary_window_fraction": float((bnd > 0).mean()),
            "lc_class_share": json.dumps({lcc.LAND_COVER[int(c)]: round(float((arr[3][~nodata] == c).mean()), 3) for c in np.unique(arr[3][~nodata])}),
        })
        wr, A, err = water_task(arr, wc, nodata)
        row.update(wr)

        # B. change prediction: the product's own probability
        p = arr[0].astype(float) / 255.0
        v = ~nodata
        conf = np.abs(p - 0.5) * 2
        flagged = arr[0] >= 128
        flagged_w = _pool(flagged.astype(float), PATCH) >= 0.5
        bnd_flag = _boundary(flagged_w.astype(int)) > 0
        conf_w = _pool(conf, PATCH)
        vals, counts = np.unique(arr[0][v], return_counts=True)
        row.update({
            "change_flagged_fraction": float(flagged[v].mean()),
            "change_prob_distinct_values": int(len(vals)), "change_prob_share_at_0": float((arr[0][v] == 0).mean()),
            "change_prob_share_at_255": float((arr[0][v] == 255).mean()), "change_prob_share_ambiguous_0.25_0.75": float(((p[v] > 0.25) & (p[v] < 0.75)).mean()),
            "change_conf_boundary_mean": float(conf_w[bnd_flag].mean()) if bnd_flag.any() else float("nan"),
            "change_conf_interior_mean": float(conf_w[~bnd_flag].mean()),
            "change_low_conf_share_boundary": float((conf_w[bnd_flag] < 0.5).mean()) if bnd_flag.any() else float("nan"),
            "change_low_conf_share_interior": float((conf_w[~bnd_flag] < 0.5).mean()),
            "change_flagged_mask_boundary_window_fraction": float(bnd_flag.mean()),
        })
        trans = {}
        if flagged.any():
            src, dst = arr[3][flagged], arr[4][flagged]
            u, c = np.unique(src.astype(int) * 100 + dst.astype(int), return_counts=True)
            for k, n in sorted(zip(u.tolist(), c.tolist()), key=lambda t: -t[1])[:5]:
                trans[f"{lcc.LAND_COVER[k // 100]} -> {lcc.LAND_COVER[k % 100]}"] = round(n / flagged.sum(), 3)
        row["change_top_transitions"] = json.dumps(trans)

        # C. change-category heads where change is flagged
        for band, score_band, legend, key in ((1, 5, PRE_CATEGORY, "pre"), (2, 6, POST_CATEGORY, "post")):
            cat, sc = arr[band][flagged], arr[score_band][flagged]
            if flagged.sum():
                u, c = np.unique(cat, return_counts=True)
                row[f"{key}_category_share_flagged"] = json.dumps({legend[int(k)]: round(n / flagged.sum(), 3) for k, n in zip(u.tolist(), c.tolist())})
                row[f"{key}_score_distinct_flagged"] = int(len(np.unique(sc)))
                row[f"{key}_score_share_at_255_flagged"] = float((sc == 255).mean())
                row[f"{key}_score_median_flagged"] = float(np.median(sc) / 255.0)
            else:
                row[f"{key}_category_share_flagged"] = "{}"; row[f"{key}_score_distinct_flagged"] = 0
                row[f"{key}_score_share_at_255_flagged"] = float("nan"); row[f"{key}_score_median_flagged"] = float("nan")
        row["pre_score_share_at_255_all"] = float((arr[5][v] == 255).mean())
        rows.append(row)
        print(f"{site:16s} LC disagree {row['lc_full_legend_disagreement']:.2f} | water err {row['water_error_rate']:.3f} "
              f"AURC bnd {row['water_aurc_boundary']:.4f} rand {row['water_aurc_random']:.4f} oracle {row['water_aurc_oracle']:.4f} "
              f"cap@5% {row['water_capture@0.05']:.2f} | change flagged {row['change_flagged_fraction']:.3f} ambiguous {row['change_prob_share_ambiguous_0.25_0.75']:.3f} "
              f"| {time.time() - t0:.1f}s")

    with open(os.path.join(OUT, "exp20_lcc_production.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = {
        "n_sites": len(rows),
        "water_aurc_boundary_below_random_sites": int(sum(r["water_aurc_boundary"] < r["water_aurc_random"] for r in rows if r["water_error_rate"] > 0)),
        "sites_with_water_disagreements": int(sum(r["water_error_rate"] > 0 for r in rows)),
        "water_capture@0.05_median_over_sites_with_disagreements": float(np.median([r["water_capture@0.05"] for r in rows if r["water_error_rate"] > 0])),
        "water_boundary_share_errors_median": float(np.nanmedian([r["water_boundary_share_errors"] for r in rows])),
        "water_boundary_share_correct_median": float(np.median([r["water_boundary_share_correct"] for r in rows])),
        "lc_full_legend_disagreement_median": float(np.median([r["lc_full_legend_disagreement"] for r in rows])),
        "change_flagged_fraction_median": float(np.median([r["change_flagged_fraction"] for r in rows])),
        "change_prob_share_ambiguous_median": float(np.median([r["change_prob_share_ambiguous_0.25_0.75"] for r in rows])),
        "change_low_conf_share_boundary_median": float(np.nanmedian([r["change_low_conf_share_boundary"] for r in rows])),
        "change_low_conf_share_interior_median": float(np.nanmedian([r["change_low_conf_share_interior"] for r in rows])),
        "settings": {"size_px": SIZE, "patch": PATCH, "budgets": BUDGETS, "lcc_to_worldcover": LCC_TO_WC,
                     "reference": "ESA WorldCover 2021 v200 (weak; three years older than the product)"},
    }
    def clean(o):  # NaN is not valid JSON; write null
        if isinstance(o, float) and np.isnan(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    with open(os.path.join(OUT, "exp20_lcc_production.json"), "w") as f:
        json.dump(clean({"summary": summary, "sites": rows}), f, indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, indent=1))
    make_figure(windows)


def make_figure(windows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch, Rectangle
    from oe_inferencex import figstyle
    figstyle.setup()
    site = "kazungula"
    arr, wc = windows[f"{site}_lcc"], windows[f"{site}_wc"]
    nodata = lcc.nodata_mask(arr)
    hard = to_index(arr[3].astype(int)); ref = wc_to_index(wc)
    wr, A, err = water_task(arr, wc, nodata)
    wc_names = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 70: "snow", 80: "water", 90: "wetland", 100: "moss"}
    wc_colors = {10: "#006400", 20: "#ffbb22", 30: "#ffff4c", 40: "#f096ff", 50: "#fa0000", 60: "#b4b4b4", 70: "#f0f0f0", 80: "#0064c8", 90: "#0096a0", 100: "#fae6a0"}
    cmap = ListedColormap(["#000000"] + [wc_colors[c] for c in WC_CLASSES])
    norm = BoundaryNorm(np.arange(-1.5, len(WC_CLASSES) + 0.5, 1), cmap.N)
    present = sorted(set(np.unique(hard).tolist()) | set(np.unique(ref).tolist()))
    handles = [Patch(color=wc_colors[WC_CLASSES[i]], label=wc_names[WC_CLASSES[i]]) for i in present if i >= 0]
    handles.append(Patch(color="#000000", label="excluded / no data"))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))
    ax = axes[0, 0]; ax.imshow(hard, cmap=cmap, norm=norm, interpolation="nearest"); ax.set_title("(a) LCC source land cover (band 4), legend mapped to WorldCover")
    ax = axes[0, 1]; ax.imshow(ref, cmap=cmap, norm=norm, interpolation="nearest"); ax.set_title("(b) ESA WorldCover 2021 (weak reference, 3 years older)")
    ax.legend(handles=handles, loc="lower left", fontsize=7, ncol=2, framealpha=0.9)
    ax = axes[0, 2]; im = ax.imshow(np.where(nodata, np.nan, arr[0] / 255.0), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("(c) change probability (band 1), the only exported confidence")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="P(change)")
    ax = axes[1, 0]; im = ax.imshow(A["boundary"], cmap="cividis", vmin=0, vmax=1, interpolation="nearest"); ax.set_title(f"(d) water-map boundary fraction per {PATCH}-px window")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="fraction of 8 neighbours with the other class")
    ax = axes[1, 1]; ax.imshow(np.where(A["valid"], err, np.nan), cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    valid = A["valid"]; b_all = np.where(valid, A["boundary"], -1).flatten()
    k = max(1, int(round(0.05 * valid.sum())))
    for idx in np.argsort(b_all, kind="stable")[::-1][:k]:
        r_, c_ = np.unravel_index(idx, valid.shape)
        ax.add_patch(Rectangle((c_ - 0.5, r_ - 0.5), 1, 1, fill=False, edgecolor="#d62728", linewidth=0.6))
    ax.set_title("(e) water disagreement with WorldCover (black); 5% review set by boundary (red)")
    ax = axes[1, 2]
    e, b = err[valid].astype(float), A["boundary"][valid]
    for name, s, color in (("boundary fraction", b, "#ff7f0e"), ("oracle", e, "#2ca02c")):
        order = np.argsort(s, kind="stable"); kept_err = np.cumsum(e[order]); cov = np.arange(1, len(e) + 1) / len(e)
        ax.plot(cov, kept_err / np.arange(1, len(e) + 1), color=color, label=f"{name} (AURC {aurc_expected(s, e):.3f})")
    ax.axhline(e.mean(), color="grey", linestyle="--", label=f"random ({e.mean():.3f})")
    ax.plot([], [], " ", label="model confidence: not exported for classes")
    ax.set_xlabel("coverage (fraction of windows kept, least boundary first)"); ax.set_ylabel("risk (disagreement rate among kept)")
    ax.set_title("(f) water task: risk-coverage against WorldCover (tie-aware)"); ax.legend(fontsize=8)
    for ax in axes.flat[:5]:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"exp20: production LCC raster at Kazungula ({SIZE} px, about {SIZE * 9.55 / 1000:.1f} km; EPSG:3857; tile {json.loads(str(windows[site + '_meta']))['tile']})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp20_lcc_kazungula.png"), dpi=150)
    print("figure written")


if __name__ == "__main__":
    main()
