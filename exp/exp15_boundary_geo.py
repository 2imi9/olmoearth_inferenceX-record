"""Exp 15: prediction-boundary proximity combined with the reference-map check.

Signals on the 27 rule-selected exp11 scenes (seed-0 Base head, cached
features; two cache leftovers excluded):
  baseline      - max-softmax confidence
  boundary      - fraction of a patch's 8 neighbors with a different hard
                  label on the model's own prediction map (exp14)
  geo           - E_geo flag: patch lies on an OSM river centerline and the
                  model predicts dry (contradiction with the reference map)
  boundary+geo  - lexicographic: geo-flagged patches first, then by boundary
                  score (a conjunction ordering, not a calibrated fusion)

Georeferencing was not cached by exp11, so each scene's transform is
recovered by re-running the deterministic scene lookup and verifying that
the re-read B02 band equals the cached image; unverified scenes are
excluded. OSM waterway=river ways are fetched per scene with mirror
fallback and rasterized onto the patch grid (exp02 method).

Reported per scene and across scenes: E-AURC of each signal, number of
geo-flagged patches, and the precision of geo flags against the WorldCover
reference (with the caveat that OSM and WorldCover can disagree about
narrow or seasonal rivers, so a flag counted as a false alarm may be a
reference disagreement rather than a model error).
"""
import csv
import json
import os
import urllib.parse
import urllib.request

import numpy as np
import rasterio
import rasterio.warp
import torch
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from olmoearth_pretrain.data.constants import Modality
from oe_inferencex.data import _catalog
from oe_inferencex.evidence import (
    train_logistic_head, predict_head, predict_logit, rasterize_polyline, pool_to_patches,
)
from exp13_stat_corrections import eaurc, sign_test_p, GRID, NON_RULE
from exp11_hardening import candidate_centers, EXISTING, MIRRORS

SIZE, PATCH, PAD = 128, 4, 4
GEO_CACHE = "exp/out/exp15_geo.npz"


def scene_georef(lon, lat, cached_img):
    """Re-run the deterministic lookup; return (crs_wkt, transform) if the
    re-read B02 band matches the cached image, else None."""
    search = _catalog().search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime="2024-06-01/2024-09-30",
        query={"eo:cloud_cover": {"lt": 5}},
    )
    item = sorted(search.items(), key=lambda i: i.properties["eo:cloud_cover"])[0]
    band0 = Modality.SENTINEL2_L2A.band_order[0]
    with rasterio.open(item.assets[band0].href) as src:
        crs, transform = src.crs, src.transform
        xs, ys = rasterio.warp.transform("EPSG:4326", crs, [lon], [lat])
        row, col = src.index(xs[0], ys[0])
        n = SIZE + PAD
        win_t = transform * rasterio.Affine.translation(col - n // 2, row - n // 2)
        with WarpedVRT(src, crs=crs, transform=win_t, width=n, height=n,
                       resampling=Resampling.bilinear) as vrt:
            b0 = vrt.read(1)
    if not np.array_equal(b0.astype(np.int32), cached_img[0]):
        return None
    return crs.to_wkt(), np.array(win_t)[:6]


def osm_centerline(crs_wkt, transform):
    crs = rasterio.crs.CRS.from_wkt(crs_wkt)
    t = rasterio.Affine(*transform)
    corners = [t * p for p in [(0, 0), (SIZE, 0), (SIZE, SIZE), (0, SIZE)]]
    xs, ys = zip(*corners)
    lons, lats = rasterio.warp.transform(crs, "EPSG:4326", xs, ys)
    bbox = f"{min(lats)},{min(lons)},{max(lats)},{max(lons)}"
    query = f'[out:json][timeout:60];way["waterway"="river"]({bbox});out geom;'
    ways = None
    for url in MIRRORS:
        try:
            req = urllib.request.Request(
                url, data=("data=" + urllib.parse.quote(query)).encode(),
                headers={"User-Agent": "oe-inferencex-exp15"})
            ways = json.loads(urllib.request.urlopen(req, timeout=90).read())["elements"]
            break
        except Exception:
            continue
    if ways is None:
        raise RuntimeError("overpass failed")
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    inv = ~t
    for way in ways:
        pts = []
        for nd in way.get("geometry", []):
            x, y = rasterio.warp.transform("EPSG:4326", crs, [nd["lon"]], [nd["lat"]])
            c, r = inv * (x[0], y[0])
            pts.append((r, c))
        if len(pts) > 1:
            mask |= rasterize_polyline(pts, SIZE)
    return pool_to_patches(mask, PATCH) > 0


def boundary_indicator(p):
    hard = (p > 0.5).astype(int)
    pad = np.pad(hard, 1, mode="edge")
    nb = np.zeros_like(p)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                nb += (pad[1 + di:1 + di + GRID, 1 + dj:1 + dj + GRID] != hard)
    return nb / 8.0


def main():
    scenes = dict(np.load("exp/out/exp11_scenes.npz", allow_pickle=True))
    feats = dict(np.load("exp/out/exp11_feats.npz", allow_pickle=True))
    z3 = np.load("exp/out/exp03_cache.npz")
    torch.manual_seed(0)
    hb = train_logistic_head(torch.tensor(feats["tr_base"]), z3["tr_labels"])

    coords = dict(EXISTING)
    for name, lon, lat in candidate_centers():
        coords[name] = (lon, lat)
    geo = dict(np.load(GEO_CACHE, allow_pickle=True)) if os.path.exists(GEO_CACHE) else {}

    names = sorted({k.rsplit("_", 1)[0] for k in scenes if k.endswith("_img")})
    rows, per = [], {}
    for name in names:
        if f"{name}_base0" not in feats or name not in coords or name in NON_RULE:
            continue
        lab = scenes[f"{name}_lab"]
        p = predict_head(torch.tensor(feats[f"{name}_base0"]), *hb)
        logit = predict_logit(torch.tensor(feats[f"{name}_base0"]), *hb)
        err = ((p > 0.5) != lab.astype(bool)).astype(np.float64)
        if err.sum() < 8:
            continue
        if f"{name}_center" not in geo:
            try:
                g = scene_georef(*coords[name], scenes[f"{name}_img"])
            except Exception as exc:
                print(f"{name}: georef fetch failed ({type(exc).__name__})")
                continue
            if g is None:
                print(f"{name}: re-read band does not match cache, excluded")
                geo[f"{name}_center"] = np.zeros((GRID, GRID), dtype=bool)
                geo[f"{name}_valid"] = np.array(0)
                np.savez(GEO_CACHE, **geo)
                continue
            try:
                geo[f"{name}_center"] = osm_centerline(*g)
                geo[f"{name}_valid"] = np.array(1)
            except RuntimeError:
                print(f"{name}: overpass failed, excluded")
                continue
            np.savez(GEO_CACHE, **geo)
        if int(geo[f"{name}_valid"]) == 0:
            continue
        center = geo[f"{name}_center"]
        bnd = boundary_indicator(p)
        geo_flag = (center & (p < 0.5)).astype(np.float64)
        sigs = {
            "baseline": -np.abs(logit),
            "boundary": bnd,
            "geo": geo_flag,
            "boundary+geo": bnd + geo_flag,  # geo-flagged first, then by boundary
        }
        e = err.flatten()
        val = {k: eaurc(v.flatten(), e) for k, v in sigs.items()}
        nflag = int(geo_flag.sum())
        prec = float(err[geo_flag > 0].mean()) if nflag else float("nan")
        per[name] = val
        rows.append({"scene": name, "n_errors": int(err.sum()), "centerline_patches": int(center.sum()),
                     "geo_flags": nflag, "geo_flag_precision_vs_worldcover": f"{prec:.3f}",
                     **{k: f"{v:.5f}" for k, v in val.items()}})
        print(f"{name}: errors {int(err.sum())}, centerline {int(center.sum())}, geo flags {nflag} "
              f"(precision {prec:.2f}), " + ", ".join(f"{k}={v:.4f}" for k, v in val.items()))

    with open("exp/out/exp15_boundary_geo.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    sn = sorted(per); n = len(sn)
    print(f"\nscenes with verified georeferencing: {n}")
    for a, b in (("boundary+geo", "boundary"), ("boundary+geo", "baseline"), ("boundary", "baseline"), ("geo", "baseline")):
        d = np.array([per[s][b] - per[s][a] for s in sn])  # >0 = a better
        wins, losses = int((d > 1e-12).sum()), int((d < -1e-12).sum())
        ties = n - wins - losses
        p_sign = sign_test_p(wins, wins + losses) if wins + losses else 1.0
        print(f"{a:<13} < {b:<9}: W/L/T {wins:>2}/{losses:>2}/{ties:<2} sign p (untied pairs)={p_sign:.3f}  median E-AURC gain {float(np.median(d)):+.4f}")
    flagged = [r for r in rows if r["geo_flags"] > 0]
    if flagged:
        total_flags = sum(r["geo_flags"] for r in flagged)
        hits = sum(r["geo_flags"] * float(r["geo_flag_precision_vs_worldcover"]) for r in flagged)
        base = np.mean([r["n_errors"] / (GRID * GRID) for r in flagged])
        zero = [r["scene"] for r in flagged if float(r["geo_flag_precision_vs_worldcover"]) == 0.0]
        print(f"\nscenes with any geo flag: {len(flagged)}/{n}; pooled precision {hits / total_flags:.3f} "
              f"over {total_flags} flags ({hits / total_flags / base:.1f}x the base error rate {base:.3f}); "
              f"unweighted per-scene mean {np.mean([float(r['geo_flag_precision_vs_worldcover']) for r in flagged]):.2f}")
        print(f"precision-zero scenes: {len(zero)} ({', '.join(zero)})")
    print("wrote exp/out/exp15_boundary_geo.csv")


if __name__ == "__main__":
    main()
