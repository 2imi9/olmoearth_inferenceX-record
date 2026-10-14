"""exp25: do the WorldCover-referenced wins live on seasonal water margins?

After exp23 (reference-version instability) and exp24 (the year gap) failed
to explain why tiling instability beats confidence against WorldCover but
not against date-matched hand labels, the remaining measurable candidate is
the mismatch between a single-date image and an annual composite map:
seasonal water margins (floodplains, river banks, pans) are water on some
dates and not on others, an annual map cannot represent them, and they are
boundary-structured. If the wins live there, removing seasonal-water patches
from scoring should remove the advantage.

Seasonality comes from the JRC Global Surface Water product (v1.3 on the
Planetary Computer: seasonality = months with water in 2020, 30 m; occurrence
= percentage of valid observations with water, 1984-2020). Both are warped
(nearest) onto each scene grid and pooled to 4-px patches. Pre-specified
patch classes: seasonal = any pixel with seasonality 1-11 months; otherwise
permanent (seasonality 12 present) or no-water. Secondary definition: any
pixel with occurrence strictly between 0 and 100.

Tests, on the 2024 scenes (exp13 set, grids from exp23) and the 2021 scenes
(exp24 set, grids re-derived from the 2021 lookup):
  T1 enrichment: share of seasonal patches among disagreements versus
     agreements, per scene; sign test.
  T2 exclusion: signals versus confidence (E-AURC, wins/losses/ties, sign
     test) with seasonal patches removed from scoring, on scenes with at
     least 8 remaining disagreements, side by side with all patches on the
     same scenes. The hypothesis predicts the tiling-instability advantage
     collapses.
  T3 complement: the same on seasonal patches only.

Outputs: exp/out/exp25_seasonal_water.csv, exp25_summary.json,
exp25_seasonal_water.png. Cache: exp/out/exp25_jrc.npz (ignored by git).
"""
import csv
import json
import os
import sys

import numpy as np
import rasterio
import rasterio.crs
import rasterio.warp
import torch
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from olmoearth_pretrain.data.constants import Modality  # noqa: E402

from oe_inferencex.data import _catalog  # noqa: E402
from oe_inferencex.evidence import train_logistic_head  # noqa: E402
from exp11_hardening import EXISTING, candidate_centers  # noqa: E402
from exp13_stat_corrections import NON_RULE, eaurc, sign_test_p  # noqa: E402
from exp24_year2021 import SIGS, TRAIN, YEAR, signals  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SIZE, PATCH, PAD = 128, 4, 4
GRID = SIZE // PATCH
MIN_ERR = 8
JRC_CACHE = os.path.join(OUT, "exp25_jrc.npz")


def scene_georef(lon, lat, cached_img, datetime):
    """exp15's deterministic lookup with a date range; returns (crs_wkt, transform) if the
    re-read first band equals the cached image, else None."""
    search = _catalog().search(collections=["sentinel-2-l2a"], intersects={"type": "Point", "coordinates": [lon, lat]},
                               datetime=datetime, query={"eo:cloud_cover": {"lt": 5}})
    item = sorted(search.items(), key=lambda i: i.properties["eo:cloud_cover"])[0]
    band0 = Modality.SENTINEL2_L2A.band_order[0]
    with rasterio.open(item.assets[band0].href) as src:
        crs, transform = src.crs, src.transform
        xs, ys = rasterio.warp.transform("EPSG:4326", crs, [lon], [lat])
        row, col = src.index(xs[0], ys[0])
        n = SIZE + PAD
        win_t = transform * rasterio.Affine.translation(col - n // 2, row - n // 2)
        with WarpedVRT(src, crs=crs, transform=win_t, width=n, height=n, resampling=Resampling.bilinear) as vrt:
            b0 = vrt.read(1)
    if not np.array_equal(b0.astype(np.int32), cached_img[0]):
        return None
    return crs.to_wkt(), np.array(win_t)[:6]


def fetch_jrc(lon, lat, crs_wkt, transform):
    """JRC GSW seasonality (months, 2020) and occurrence (%) on the scene grid, nearest."""
    crs = rasterio.crs.CRS.from_wkt(crs_wkt)
    tr = rasterio.Affine(*transform)
    item = next(_catalog().search(collections=["jrc-gsw"], intersects={"type": "Point", "coordinates": [lon, lat]}, max_items=1).items())
    out = {}
    for asset in ("seasonality", "occurrence"):
        with rasterio.open(item.assets[asset].href) as src:
            with WarpedVRT(src, crs=crs, transform=tr, width=SIZE + PAD, height=SIZE + PAD, resampling=Resampling.nearest) as vrt:
                out[asset] = vrt.read(1)
    return out["seasonality"], out["occurrence"]


def patch_classes(seasonality, occurrence):
    """Per-patch flags on the GRID x GRID lattice."""
    s = seasonality[:SIZE, :SIZE].astype(int)
    o = occurrence[:SIZE, :SIZE].astype(int)
    s = np.where(s > 12, 0, s)            # nodata -> no water
    o = np.where(o > 100, 0, o)
    blocks = s.reshape(GRID, PATCH, GRID, PATCH)
    seasonal = ((blocks >= 1) & (blocks <= 11)).any(axis=(1, 3))
    permanent = (blocks == 12).any(axis=(1, 3)) & ~seasonal
    ob = o.reshape(GRID, PATCH, GRID, PATCH)
    intermittent = ((ob > 0) & (ob < 100)).any(axis=(1, 3))
    return seasonal, permanent, intermittent


def wlt(per, sn, key):
    d = np.array([per[s]["baseline"] - per[s][key] for s in sn])
    w, l = int((d > 1e-12).sum()), int((d < -1e-12).sum())
    return {"wlt": [w, l, len(d) - w - l], "sign_p": sign_test_p(w, w + l) if w + l else 1.0, "median_gain": float(np.median(d)) if len(d) else None, "n": len(d)}


def main():
    coords = dict(EXISTING)
    for name, lon, lat in candidate_centers():
        coords[name] = (lon, lat)
    jrc = dict(np.load(JRC_CACHE, allow_pickle=True)) if os.path.exists(JRC_CACHE) else {}

    # data for both years
    years = {}
    scenes24 = dict(np.load(os.path.join(OUT, "exp11_scenes.npz"), allow_pickle=True))
    feats24 = dict(np.load(os.path.join(OUT, "exp11_feats.npz"), allow_pickle=True))
    z3 = np.load(os.path.join(OUT, "exp03_cache.npz"))
    geo23 = dict(np.load(os.path.join(OUT, "exp23_geo.npz"), allow_pickle=True))
    torch.manual_seed(0)
    hb24 = train_logistic_head(torch.tensor(feats24["tr_base"]), z3["tr_labels"])
    hn24 = train_logistic_head(torch.tensor(feats24["tr_nano"]), z3["tr_labels"])
    years["2024"] = {"scenes": scenes24, "feats": feats24, "hb": hb24, "hn": hn24, "datetime": "2024-06-01/2024-09-30",
                     "geo": {n_: (str(geo23[f"{n_}_crs"]), geo23[f"{n_}_transform"]) for n_ in {k.rsplit("_", 1)[0] for k in geo23 if k.endswith("_crs")}}}
    scenes21 = dict(np.load(os.path.join(OUT, "exp24_scenes.npz"), allow_pickle=True))
    feats21 = dict(np.load(os.path.join(OUT, "exp24_feats.npz"), allow_pickle=True))
    torch.manual_seed(0)
    hb21 = train_logistic_head(torch.tensor(feats21["tr_base"]), scenes21["tr_lab"])
    hn21 = train_logistic_head(torch.tensor(feats21["tr_nano"]), scenes21["tr_lab"])
    years["2021"] = {"scenes": scenes21, "feats": feats21, "hb": hb21, "hn": hn21, "datetime": YEAR, "geo": {}}

    rows, summary = [], {"years": {}}
    for year, Y in years.items():
        feats, scenes = Y["feats"], Y["scenes"]
        a_tr = torch.nn.functional.normalize(torch.tensor(feats["tr_base"]).reshape(-1, feats["tr_base"].shape[-1]), dim=-1)
        names = sorted(k.rsplit("_", 1)[0] for k in feats if k.endswith("_base0") and k.rsplit("_", 1)[0] not in NON_RULE and k.rsplit("_", 1)[0] in coords and k.rsplit("_", 1)[0] != "tr")
        per_all, per_ex, per_only = {}, {}, {}
        for name in names:
            key = f"{year}_{name}"
            if f"{key}_seasonality" not in jrc:
                if name in Y["geo"]:
                    crs_wkt, transform = Y["geo"][name]
                else:
                    try:
                        g = scene_georef(*coords[name], scenes[f"{name}_img"], Y["datetime"])
                    except Exception as exc:
                        print(f"{key}: georef failed ({type(exc).__name__}); excluded"); continue
                    if g is None:
                        print(f"{key}: re-read band does not match cache; excluded"); continue
                    crs_wkt, transform = g
                try:
                    sea, occ = fetch_jrc(*coords[name], crs_wkt, transform)
                except Exception as exc:
                    print(f"{key}: JRC fetch failed ({type(exc).__name__}); excluded"); continue
                jrc[f"{key}_seasonality"], jrc[f"{key}_occurrence"] = sea, occ
                np.savez(JRC_CACHE, **jrc)
            seasonal, permanent, intermittent = patch_classes(jrc[f"{key}_seasonality"], jrc[f"{key}_occurrence"])
            lab = scenes[f"{name}_lab"].astype(bool)
            p, sigs = signals(feats, name, Y["hb"], Y["hn"], a_tr, scenes[f"{name}_img"])
            err = ((p > 0.5) != lab).astype(float).flatten()
            if err.sum() < MIN_ERR:
                continue
            s = {k: v.flatten() for k, v in sigs.items()}
            sea_f, int_f = seasonal.flatten(), intermittent.flatten()
            keep = ~sea_f
            per_all[name] = {k: eaurc(v, err) for k, v in s.items()}
            if err[keep].sum() >= MIN_ERR:
                per_ex[name] = {k: eaurc(v[keep], err[keep]) for k, v in s.items()}
            if err[sea_f].sum() >= MIN_ERR and (~err[sea_f].astype(bool)).sum() >= MIN_ERR:
                per_only[name] = {k: eaurc(v[sea_f], err[sea_f]) for k, v in s.items()}
            row = {"year": year, "scene": name, "n_errors": int(err.sum()), "seasonal_fraction": float(sea_f.mean()), "intermittent_fraction": float(int_f.mean()),
                   "permanent_fraction": float(permanent.mean()), "seasonal_share_among_errors": float(sea_f[err > 0].mean()), "seasonal_share_among_correct": float(sea_f[err == 0].mean()),
                   "intermittent_share_among_errors": float(int_f[err > 0].mean()), "intermittent_share_among_correct": float(int_f[err == 0].mean()),
                   "error_rate_seasonal": float(err[sea_f].mean()) if sea_f.any() else float("nan"), "error_rate_nonseasonal": float(err[keep].mean()),
                   "n_errors_nonseasonal": int(err[keep].sum()), "n_errors_seasonal": int(err[sea_f].sum())}
            for k in s:
                row[f"eaurc_all[{k}]"] = per_all[name][k]
                row[f"eaurc_nonseasonal[{k}]"] = per_ex[name][k] if name in per_ex else float("nan")
                row[f"eaurc_seasonal[{k}]"] = per_only[name][k] if name in per_only else float("nan")
            rows.append(row)
            g_all = per_all[name]["baseline"] - per_all[name]["tile-phase (aligned)"]
            g_ex = (per_ex[name]["baseline"] - per_ex[name]["tile-phase (aligned)"]) if name in per_ex else float("nan")
            print(f"{year} {name:14s} err {int(err.sum()):3d} | seasonal {sea_f.mean():.3f} | among err {row['seasonal_share_among_errors']:.2f} vs ok {row['seasonal_share_among_correct']:.2f} | "
                  f"tile gain all {g_all:+.4f} nonseasonal {g_ex:+.4f} (err {row['n_errors_nonseasonal']})")
        yr = {"n_scenes": len(per_all), "n_scenes_nonseasonal": len(per_ex), "n_scenes_seasonal_only": len(per_only)}
        rr = [r for r in rows if r["year"] == year]
        d1 = np.array([r["seasonal_share_among_errors"] - r["seasonal_share_among_correct"] for r in rr])
        w1, l1 = int((d1 > 1e-12).sum()), int((d1 < -1e-12).sum())
        yr["T1_enrichment"] = {"median_share_errors": float(np.median([r["seasonal_share_among_errors"] for r in rr])),
                               "median_share_correct": float(np.median([r["seasonal_share_among_correct"] for r in rr])),
                               "pooled_share_errors": float(sum(r["seasonal_share_among_errors"] * r["n_errors"] for r in rr) / max(sum(r["n_errors"] for r in rr), 1)),
                               "wins_losses_ties": [w1, l1, len(rr) - w1 - l1], "sign_p": sign_test_p(w1, w1 + l1) if w1 + l1 else 1.0,
                               "median_error_rate_seasonal": float(np.nanmedian([r["error_rate_seasonal"] for r in rr])),
                               "median_error_rate_nonseasonal": float(np.median([r["error_rate_nonseasonal"] for r in rr]))}
        sn_ex, sn_only = sorted(per_ex), sorted(per_only)
        yr["T2_exclusion"] = {k: {"all_patches_same_scenes": wlt(per_all, sn_ex, k), "nonseasonal_patches": wlt(per_ex, sn_ex, k)} for k in SIGS}
        yr["T3_seasonal_only"] = {k: wlt(per_only, sn_only, k) for k in SIGS}
        summary["years"][year] = yr
        print(f"{year}: T1 {yr['T1_enrichment']['wins_losses_ties']} p {yr['T1_enrichment']['sign_p']:.3g} | tile all {yr['T2_exclusion']['tile-phase (aligned)']['all_patches_same_scenes']['wlt']} nonseasonal {yr['T2_exclusion']['tile-phase (aligned)']['nonseasonal_patches']['wlt']} (p {yr['T2_exclusion']['tile-phase (aligned)']['nonseasonal_patches']['sign_p']:.2g}) | seasonal-only {yr['T3_seasonal_only']['tile-phase (aligned)']['wlt']}")

    with open(os.path.join(OUT, "exp25_seasonal_water.csv"), "w", newline="") as fh:
        keys = list(rows[0].keys())
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, "exp25_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    make_figure(rows, summary)


def make_figure(rows, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from oe_inferencex import figstyle
    figstyle.setup()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    rr = [r for r in rows if r["year"] == "2024"]
    x = np.arange(len(rr))
    ax.bar(x - 0.2, [r["seasonal_share_among_errors"] for r in rr], 0.4, color="#d62728", label="among disagreements")
    ax.bar(x + 0.2, [r["seasonal_share_among_correct"] for r in rr], 0.4, color="#7f7f7f", label="among agreements")
    ax.set_xticks(x); ax.set_xticklabels([r["scene"] for r in rr], rotation=90, fontsize=6)
    ax.set_ylabel("share of seasonal-water patches (JRC seasonality 1-11 months)"); ax.set_title("(a) T1, 2024 scenes"); ax.legend(fontsize=8)
    for j, year in enumerate(("2024", "2021")):
        ax = axes[1 + j]
        T2, T3 = summary["years"][year]["T2_exclusion"], summary["years"][year]["T3_seasonal_only"]
        for i, k in enumerate(SIGS):
            a, e, o = T2[k]["all_patches_same_scenes"], T2[k]["nonseasonal_patches"], T3[k]
            ax.bar(i - 0.27, a["wlt"][0] / max(sum(a["wlt"]), 1), 0.27, color="#ff7f0e", label="all patches" if i == 0 else None)
            ax.bar(i, e["wlt"][0] / max(sum(e["wlt"]), 1), 0.27, color="#2ca02c", label="non-seasonal patches only" if i == 0 else None)
            ax.bar(i + 0.27, o["wlt"][0] / max(sum(o["wlt"]), 1), 0.27, color="#9467bd", label="seasonal patches only" if i == 0 else None)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xticks(range(len(SIGS))); ax.set_xticklabels(SIGS, rotation=20, fontsize=8)
        ax.set_ylabel("fraction of scenes where the signal beats confidence")
        ax.set_title(f"({'bc'[j]}) T2/T3, {year} imagery ({summary['years'][year]['n_scenes_nonseasonal']} / {summary['years'][year]['n_scenes_seasonal_only']} scenes)")
        ax.legend(fontsize=7)
    fig.suptitle("exp25: seasonal water (JRC GSW) and the WorldCover-referenced comparisons", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp25_seasonal_water.png"), dpi=150)
    print("figure written")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # the JRC asset leaves GDAL/curl handles that hang the interpreter at exit
