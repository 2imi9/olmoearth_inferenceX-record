#!/usr/bin/env python
"""Scores provider: run Ai2's public fine-tuned AWF model over a small area and write its pre-argmax output.

OlmoEarth Studio's public API returns map tiles and a point lookup (a raw value or a hard class), never per-class
scores and no raster download (exp/out/agent_trial_2026-09-24.md). Running the model directly gives the margin the
review set needs at every pixel. This script writes that output as a georeferenced (C, H, W) float32 raster plus a
JSON manifest, the input `oe-inferencex assess` and `sample` take and the OlmoEarth Agent's review-set tools rank.

Model: allenai/OlmoEarth-v1-FT-AWF-Base (southern Kenya land use and land cover, 10 channels), the one fine-tuned
checkpoint this repository has already run end to end (exp21: a replica without rslearn, 0.881 on the validation
split against Ai2's 0.895). Model loading and the forward pass are exp21's own `load_finetuned` and `logits_grid`,
imported, not rewritten. Mangrove was the alternative and is not used: it was never run here, and its pooling
decoder emits one prediction per patch (its task card), so its map is block-constant.

What is replicated from Ai2's configs (olmoearth_projects/olmoearth_run_data/awf and rslearn), and checked against
the training windows in allenai/olmoearth_projects_awf dataset.tar:
  imagery    dataset.json: Sentinel-2 L2A from Planetary Computer, one mosaic per 30-day period, 12 periods counted
             back from the end of the date window, items sorted by eo:cloud_cover (a later item fills only pixels
             the earlier ones lack, no cloud masking), harmonize=true (the +1000 offset of processing baseline
             04.00 and later removed). Item group 0 is the most recent period: rslearn's PER_PERIOD_MOSAIC order
             when the dataset was built (the training window items.json lists December first), and the order the
             model saw with legacy timestamps (month index = timestep index). All twelve bands are read onto the
             10 m grid with bilinear resampling, as the published dataset.json materializes them (one band set,
             rslearn's default resampling). Checked on training window task_20546e37..._point_32: the same scene
             in all 12 periods and bit-identical 10 m bands; the 20 m and 60 m bands differ by about 1.5% on
             average, because that older dataset stored them at 20 and 40 m and upsampled them nearest at load.
  windowing  model.yaml predict_config: 16-px crops (the training crop size), overlap 4 px, the last crop moved
             inward (rslearn get_window_crop_options), merged by RasterMerger(padding=2): 2 px trimmed from each
             crop's left and top edge except at the area's border, later crops overwriting earlier ones.
  grid       olmoearth_run.yaml: UTM at 10 m, as the training windows are (EPSG:32737 in the AWF region).
The default area is centred on Namanga (-2.55, 36.81), 512 x 512 px: the 5.12 km box holding the densest mix of AWF
annotation points (43 points, 7 of the 9 classes; they include training points), so the map is not one class. The
date window defaults to 2023, the training year, as Ai2's project doc advises.

Output (--out DIR):
  scores.tif     (10, H, W) float32; logits by default, softmax probabilities with --probabilities; NaN where any
                 of the 12 periods has no imagery (the raster's no-data value is NaN); band descriptions are the
                 class names from the task card; channel 9 is the label fill value, never a training target, kept
                 because the served argmax runs over all ten channels.
  manifest.json  model id and the revision actually loaded, area and grid, date window, the imagery ids per period,
                 class names, logits or probabilities, the windowing, software versions, the commands to run next.

The geometry, imagery planning, windowing and writing are torch-free and tested with a stub model
(tests/test_score_area.py); only `build_awf_model` needs the encoder extra. fp32 throughout (TF32 disabled).

    uv run --extra encoder --extra geo python scripts/score_area.py --out DIR [--lat -2.55 --lon 36.81 --size 512]
        [--start 2023-01-01 --end 2023-12-31] [--probabilities]
    oe-inferencex assess DIR/scores.tif --logits --out DIR/assess
Cluster: scripts/score_area.sh.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "exp")
SCRIPTS = os.path.join(ROOT, "scripts")

PROJECT = "awf"
MODEL_REPO = "allenai/OlmoEarth-v1-FT-AWF-Base"          # exp21's REPO; build_awf_model checks they agree
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
# model.yaml predict_config: patch_size 16, overlap_ratio 0.25; RslearnWriter merger RasterMerger(padding: 2)
CROP, OVERLAP, TRIM = 16, 4, 2
# rslearn data_sources/utils.py: the mosaic coverage thresholds
MOSAIC_MIN_ITEM_COVERAGE, MOSAIC_REMAINDER_EPSILON = 0.1, 0.01
HARMONIZE_OFFSET = 1000                                  # BOA_ADD_OFFSET = -1000 from processing baseline 04.00
BASELINE_04_DATE = dt.datetime(2022, 1, 25, tzinfo=dt.timezone.utc)
# What the imagery recipe below implements; the project's dataset.json is read and checked against it.
EXPECTED_QUERY = {"space_mode": "PER_PERIOD_MOSAIC", "period_duration": "30d"}
EXPECTED_SOURCE = {"class_path": "rslearn.data_sources.planetary_computer.Sentinel2", "harmonize": True,
                   "sort_by": "eo:cloud_cover"}
DEFAULTS = {"lat": -2.55, "lon": 36.81, "size": 512, "start": "2023-01-01", "end": "2023-12-31"}


# ----------------------------------------------------------------------------- task card and data recipe
def load_task_card(project=PROJECT):
    """The task card (oe_inferencex.taskcard) as a dict: class legend, nodata, inputs, output channels."""
    from dataclasses import asdict
    from oe_inferencex import taskcard
    return asdict(taskcard.project_card(project))


def load_query_config(card):
    """The Sentinel-2 layer of the project's dataset.json (the card's third source): refuse anything this script does
    not replicate, rather than feed the model imagery built another way."""
    from oe_inferencex import taskcard
    url = next(s for s in card["sources"] if s.endswith("/dataset.json"))
    layer = json.loads(taskcard._get(url))["layers"]["sentinel2"]
    src = layer["data_source"]
    qc = src.get("query_config") or {}
    got = {"class_path": src.get("class_path"), "harmonize": (src.get("init_args") or {}).get("harmonize"),
           "sort_by": (src.get("init_args") or {}).get("sort_by"), **{k: qc.get(k) for k in EXPECTED_QUERY}}
    want = {**EXPECTED_SOURCE, **EXPECTED_QUERY}
    bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
    if bad:
        raise SystemExit(f"{url} no longer matches the imagery recipe this script replicates: {bad} (got, expected)")
    return {"source": url, "max_matches": int(qc["max_matches"]), "period_days": int(qc["period_duration"][:-1]), **got}


def card_facts(card):
    """(class names per output channel, input bands, number of time steps) from the card, with the checks that make
    the card, not this script, the authority on them."""
    task = card["task"]
    n_out = card["outputs"]["decoder_out_channels"][0]
    classes = {int(k): v for k, v in task["classes"].items()}
    nodata = task.get("nodata_value")
    names = []
    for c in range(n_out):
        if c in classes:
            names.append(classes[c])
        elif c == nodata:
            names.append(f"nodata_value_{c}")      # the label fill value: a head channel, never a training target
        else:
            raise SystemExit(f"task card {card['name']}: output channel {c} has no class name and is not the nodata value")
    s2 = card["inputs"]["sentinel2_l2a"]
    return names, list(s2["bands"]), int(s2["n_timesteps"])


# ----------------------------------------------------------------------------- geometry
def utm_epsg(lat, lon):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def area_grid(lat, lon, size, res=10.0):
    """A size x size grid at `res` m in the UTM zone of the centre, its origin on a multiple of `res` (the Sentinel-2
    10 m pixel lattice, as the training windows are)."""
    import rasterio.warp
    from rasterio.transform import Affine
    if size < CROP:
        raise SystemExit(f"--size {size} is smaller than the model's {CROP}-px crop")
    crs = f"EPSG:{utm_epsg(lat, lon)}"
    xs, ys = rasterio.warp.transform("EPSG:4326", crs, [lon], [lat])
    x0 = round((xs[0] - size * res / 2) / res) * res
    y0 = round((ys[0] + size * res / 2) / res) * res
    return {"crs": crs, "transform": Affine(res, 0.0, x0, 0.0, -res, y0), "width": size, "height": size, "res": res}


def grid_box(grid):
    """The grid's footprint as a shapely polygon in the grid's CRS."""
    import shapely.geometry
    from rasterio.transform import array_bounds
    return shapely.geometry.box(*array_bounds(grid["height"], grid["width"], grid["transform"]))


def to_wgs84(shape, crs):
    import rasterio.warp
    import shapely.geometry
    return shapely.geometry.shape(rasterio.warp.transform_geom(crs, "EPSG:4326", shapely.geometry.mapping(shape)))


def from_wgs84(geojson, crs):
    import rasterio.warp
    import shapely.geometry
    return shapely.geometry.shape(rasterio.warp.transform_geom("EPSG:4326", crs, geojson))


# ----------------------------------------------------------------------------- imagery
def parse_day(s):
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)


def periods(start, end, n, days):
    """rslearn's periods: [end - days, end), then back by `days`, while a whole period fits after `start`, at most n.
    Most recent first, which is the item-group order of the AWF training windows."""
    out, pe, d = [], end, dt.timedelta(days=days)
    while pe - d >= start and len(out) < n:
        out.append((pe - d, pe))
        pe -= d
    return out


def harmonize_offset(item):
    """(offset, rule): 1000 for processing baseline 04.00 and later, whose values carry BOA_ADD_OFFSET = -1000."""
    b = item.get("baseline")
    try:
        return (HARMONIZE_OFFSET if float(b) >= 4.0 else 0), f"s2:processing_baseline {b}"
    except (TypeError, ValueError):
        pass
    return (HARMONIZE_OFFSET if item["datetime"] >= BASELINE_04_DATE else 0), "acquisition date (no baseline property)"


def harmonize(arr, offset):
    """rslearn's harmonize callback (data_sources/copernicus.py): subtract the offset, clip at 0, and keep a valid
    pixel that reached 0 at 1 so it is not read as no-data."""
    arr = np.asarray(arr, dtype=np.uint16)
    if not offset:
        return arr
    out = np.clip(arr, offset, None) - np.uint16(offset)
    out[(out == 0) & (arr > 0)] = 1
    return out


def plan_mosaic(aoi, items):
    """rslearn's single-coverage mosaic with max_mosaics=1: in the given order, an item joins if it covers a sizable
    part of what is still uncovered; stop once less than 1% remains. `items` carry `shp` in the AOI's CRS."""
    chosen, remainder = [], None
    for it in items:
        shp = it["shp"]
        target = aoi if remainder is None else remainder
        if not shp.intersects(target):
            continue
        inter = shp.intersection(target).area
        if inter / shp.area < MOSAIC_MIN_ITEM_COVERAGE and inter / target.area < MOSAIC_MIN_ITEM_COVERAGE:
            continue
        chosen.append(it)
        remainder = target.difference(shp)
        if remainder.area / aoi.area < MOSAIC_REMAINDER_EPSILON:
            break
    return chosen


def search_items(aoi_wgs84, start, end):
    """Every Sentinel-2 L2A item touching the area in [start, end], as plain records (hrefs unsigned; signed at read)."""
    import pystac_client
    import shapely.geometry
    client = pystac_client.Client.open(STAC)
    search = client.search(collections=[COLLECTION], intersects=shapely.geometry.mapping(aoi_wgs84),
                           datetime=f"{start.isoformat()}/{end.isoformat()}")
    out = []
    for it in search.items():
        out.append({"id": it.id, "datetime": it.datetime.astimezone(dt.timezone.utc),
                    "cloud": float(it.properties.get("eo:cloud_cover", 100.0)),
                    "baseline": it.properties.get("s2:processing_baseline"), "geometry": it.geometry,
                    "assets": {k: a.href for k, a in it.assets.items()}})
    return out


def read_item(item, bands, grid):
    """(B, H, W) uint16 raw values of one item on the grid, bilinear, one band at a time (oe_inferencex.data's
    pattern, which cannot be imported without torch)."""
    import planetary_computer
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    out = np.zeros((len(bands), grid["height"], grid["width"]), dtype=np.uint16)
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MAX_RETRY": "5", "GDAL_HTTP_RETRY_DELAY": "3"}
    with rasterio.Env(**env):
        for bi, band in enumerate(bands):
            with rasterio.open(planetary_computer.sign(item["assets"][band])) as src:
                with WarpedVRT(src, crs=grid["crs"], transform=grid["transform"], width=grid["width"],
                               height=grid["height"], resampling=Resampling.bilinear) as vrt:
                    out[bi] = vrt.read(1)
    return out


def fetch_stack(grid, start, end, n_periods, period_days, bands, search=None, read=None, log=print):
    """(stack (H, W, T, B) float32 in the model's band order, valid (H, W) bool, per-period imagery records).

    Period t = 0 is the most recent. A pixel is valid when every period has imagery there; rslearn would feed the
    zeros to the model, this script marks the pixel no-data instead. Fewer than n_periods periods with imagery is
    refused, as rslearn's min_matches rejects the window."""
    search = search or search_items
    read = read or read_item
    aoi = grid_box(grid)
    per = periods(start, end, n_periods, period_days)
    if len(per) < n_periods:
        raise SystemExit(f"{start.date()} to {end.date()} holds {len(per)} whole {period_days}-day periods; the model needs {n_periods}")
    items = search(to_wgs84(aoi, grid["crs"]), per[-1][0], per[0][1])
    for it in items:
        it["shp"] = from_wgs84(it["geometry"], grid["crs"])
    H, W = grid["height"], grid["width"]
    stack = np.zeros((H, W, n_periods, len(bands)), dtype=np.float32)
    have = np.zeros((H, W, n_periods), dtype=bool)
    imagery = []
    for t, (ps, pe) in enumerate(per):
        cand = sorted((it for it in items if ps < it["datetime"] < pe), key=lambda it: (it["cloud"], it["id"]))
        chosen = plan_mosaic(aoi, cand)
        if not chosen:
            raise SystemExit(f"period {t} ({ps.date()} to {pe.date()}): no Sentinel-2 item covers the area; "
                             f"the model needs all {n_periods}")
        comp = np.zeros((len(bands), H, W), dtype=np.uint16)
        recs = []
        for it in chosen:
            off, rule = harmonize_offset(it)
            arr = harmonize(read(it, bands, grid), off)
            fill = (comp == 0).all(axis=0) & (arr != 0).any(axis=0)
            comp[:, fill] = arr[:, fill]
            recs.append({"id": it["id"], "datetime": it["datetime"].isoformat(), "eo:cloud_cover": it["cloud"],
                         "s2:processing_baseline": it.get("baseline"), "harmonize_offset": off, "harmonize_rule": rule,
                         "pixels_filled": int(fill.sum())})
        stack[:, :, t, :] = np.moveaxis(comp, 0, -1)
        have[:, :, t] = (comp != 0).any(axis=0)
        imagery.append({"t": t, "period": [ps.isoformat(), pe.isoformat()], "items": recs,
                        "candidates": len(cand), "valid_fraction": float(have[:, :, t].mean())})
        ids = ", ".join(r["id"] for r in recs)
        clouds = ", ".join("%.1f" % r["eo:cloud_cover"] for r in recs)     # no nested quotes: Python 3.11 has no PEP 701
        log(f"  t={t:2d} {ps.date()}..{pe.date()}: {ids} (cloud {clouds}%, valid {have[:, :, t].mean():.3f})")
    return stack, have.all(axis=2), imagery


# ----------------------------------------------------------------------------- windowing
def crop_starts(n, crop=CROP, overlap=OVERLAP):
    """rslearn all_crops_dataset.get_window_crop_options in one dimension: 0, then strides of crop - overlap while a
    whole crop fits before the end, then a last crop flush with the end (it may overlap its neighbour more)."""
    starts = [0] + list(range(crop - overlap, n - crop, crop - overlap))
    if n - crop > 0:
        starts.append(n - crop)
    return starts


def merge_crops(outputs, origins, height, width, trim=TRIM):
    """rslearn RasterMerger on (C, crop, crop) outputs at (x0, y0) origins: sorted by (x0, y0), each crop not on the
    area's left (top) edge loses `trim` px on its left (top), and later crops overwrite earlier ones."""
    C = outputs[0].shape[0]
    merged = np.full((C, height, width), np.nan, dtype=np.float32)
    for k in sorted(range(len(origins)), key=lambda k: origins[k]):
        src, (x0, y0) = outputs[k], origins[k]
        if trim and x0 != 0:
            src, x0 = src[:, :, trim:], x0 + trim
        if trim and y0 != 0:
            src, y0 = src[:, trim:, :], y0 + trim
        merged[:, y0:y0 + src.shape[1], x0:x0 + src.shape[2]] = src
    return merged


def predict_area(stack, model_fn, crop=CROP, overlap=OVERLAP, trim=TRIM, batch=64):
    """Sliding-window logits (C, H, W) over the whole stack. `model_fn` takes (N, crop, crop, T, B) float32 raw
    values and returns (N, C, crop, crop) logits: the fine-tuned model on the cluster, a stub in the tests."""
    H, W = stack.shape[:2]
    origins = [(x, y) for x in crop_starts(W, crop, overlap) for y in crop_starts(H, crop, overlap)]
    outputs = []
    for i in range(0, len(origins), batch):
        chunk = origins[i:i + batch]
        crops = np.stack([stack[y:y + crop, x:x + crop] for x, y in chunk]).astype(np.float32)
        out = np.asarray(model_fn(crops), dtype=np.float32)
        if out.shape[0] != len(chunk) or out.shape[2:] != (crop, crop):
            raise RuntimeError(f"model_fn returned {out.shape} for {len(chunk)} crops of {crop} px")
        outputs.extend(out)
    return merge_crops(outputs, origins, H, W, trim), len(origins)


def softmax(logits, axis=0):
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ----------------------------------------------------------------------------- writing
def write_scores(path, values, valid, grid, band_names, tags):
    """(C, H, W) float32 GeoTIFF, NaN outside `valid` and NaN as the declared no-data value."""
    import rasterio
    arr = np.where(valid[None], values, np.nan).astype(np.float32)
    profile = {"driver": "GTiff", "height": arr.shape[1], "width": arr.shape[2], "count": arr.shape[0],
               "dtype": "float32", "crs": grid["crs"], "transform": grid["transform"], "nodata": float("nan"),
               "compress": "deflate", "predictor": 3, "tiled": True, "blockxsize": 256, "blockysize": 256}
    if arr.shape[1] < 256 or arr.shape[2] < 256:
        profile.update(tiled=False, blockxsize=None, blockysize=None)
        profile = {k: v for k, v in profile.items() if v is not None}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
        for i, name in enumerate(band_names, 1):
            dst.set_band_description(i, name)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def software_versions():
    from importlib import metadata
    v = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("numpy", "rasterio", "shapely", "pystac-client", "planetary-computer", "huggingface-hub",
                "torch", "olmoearth-pretrain", "olmoearth-inferencex"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    try:
        v["repo_commit"] = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                                          check=True).stdout.strip()
        v["repo_dirty"] = bool(subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"],
                                              capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        v["repo_commit"], v["repo_dirty"] = None, None
    return v


def produce(out_dir, grid, stack, valid, imagery, model_fn, model_info, card, recipe, request, probabilities=False,
            batch=64, log=print):
    """Run the model over the stack and write scores.tif and manifest.json. Torch-free given `model_fn`."""
    names, bands, n_t = card_facts(card)
    if stack.shape[2:] != (n_t, len(bands)):
        raise SystemExit(f"the stack is {stack.shape[2:]} (periods, bands); the task card says ({n_t}, {len(bands)})")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    logits, n_crops = predict_area(stack, model_fn, batch=batch)
    seconds = time.time() - t0
    if logits.shape[0] != len(names):
        raise SystemExit(f"the model emits {logits.shape[0]} channels; the task card names {len(names)}")
    if not np.isfinite(logits[:, valid]).all():
        raise SystemExit("the model returned non-finite logits on valid pixels")
    kind = "probabilities" if probabilities else "logits"
    values = softmax(logits).astype(np.float32) if probabilities else logits
    path = os.path.join(out_dir, "scores.tif")
    write_scores(path, values, valid, grid, names,
                 {"values": kind, "model": model_info.get("repo"), "revision": model_info.get("revision"),
                  "manifest": "manifest.json"})
    hard = logits.argmax(0)[valid]
    counts = np.bincount(hard, minlength=len(names))
    flag = "" if probabilities else " --logits"
    t = grid["transform"]
    box = grid_box(grid)
    manifest = {
        "what": "the per-class pre-argmax output of a public fine-tuned OlmoEarth model over one area, written by "
                "scripts/score_area.py; the input of oe-inferencex assess and sample and of the OlmoEarth Agent's "
                "review-set tools",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "scores": {"file": "scores.tif", "sha256": sha256(path), "values": kind, "shape": list(values.shape),
                   "dtype": "float32", "nodata": "NaN", "n_pixels": int(valid.size),
                   "n_nodata_pixels": int((~valid).sum()),
                   "bands": [{"band": i + 1, "channel": i, "name": n} for i, n in enumerate(names)]},
        "classes": {str(i): n for i, n in enumerate(names)},
        "class_note": (f"channel {card['task'].get('nodata_value')} is the label fill value (the task's nodata_value), "
                       "never a training target; it is kept because the served argmax runs over every channel"
                       if card["task"].get("nodata_value") is not None else None),
        "model": model_info,
        "task_card": {k: card[k] for k in ("name", "goal", "encoder", "task", "inputs", "windows", "outputs", "audit",
                                           "sources", "warnings")},
        "area": {"request": request, "crs": grid["crs"], "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
                 "width": grid["width"], "height": grid["height"], "resolution_m": grid["res"],
                 "bounds": list(box.bounds), "bounds_wgs84": list(to_wgs84(box, grid["crs"]).bounds)},
        "date_window": {"start": request["start"], "end": request["end"], "periods": len(imagery),
                        "period_days": recipe.get("period_days"),
                        "order": "t = 0 is the most recent period (the AWF training windows' item-group order)",
                        "timestamps": "legacy: month index = timestep index (exp21's awf.stacks_to_sample)"},
        "imagery_recipe": recipe,
        "imagery": imagery,
        "inference": {"crop_px": CROP, "overlap_px": OVERLAP, "merge_trim_px": TRIM, "n_crops": n_crops,
                      "batch": batch, "seconds": round(seconds, 2),
                      "source": "olmoearth_projects awf model.yaml predict_config (patch_size 16, overlap_ratio 0.25) "
                                "and RslearnWriter RasterMerger(padding=2); rslearn get_window_crop_options"},
        "argmax_pixel_counts": {n: int(c) for n, c in zip(names, counts)},
        "software": software_versions(),
        "next": {"assess": f"oe-inferencex assess scores.tif{flag} --out assess",
                 "sample": f"oe-inferencex sample scores.tif{flag} --budget 300 --out sample/sample.csv"},
        "caveats": [
            "No labels were used. A margin ranks windows for review; how wrong the map is needs a labelled sample "
            "(sample, then estimate or certify).",
            "The model is exp21's replica of Ai2's checkpoint without rslearn: 0.881 on the AWF validation split "
            "against Ai2's reported 0.895.",
            "Each period is the least-cloudy scene over the tile, not cloud-masked, as the training data were built; "
            "a cloudy period enters the model as it is.",
            "The 20 m and 60 m bands are read onto the 10 m grid bilinearly, as the published dataset.json does; the "
            "training windows upsampled them nearest from 20 and 40 m, about 1.5% apart on average on one window.",
        ],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    log(f"wrote {path} ({kind}, {values.shape}, {int((~valid).sum())} no-data pixels) and manifest.json; "
        f"{n_crops} crops in {seconds:.1f}s")
    return manifest


# ----------------------------------------------------------------------------- the model (cluster only)
def _snapshot_revision(repo, filename):
    """The commit of the snapshot a cached download resolves to: the bytes `load_finetuned` read."""
    from huggingface_hub import hf_hub_download
    m = re.search(r"snapshots[/\\]([0-9a-f]{40})[/\\]", hf_hub_download(repo, filename))
    return m.group(1) if m else None


def build_awf_model(bands, n_timesteps):
    """(model_fn, model_info): exp21's fine-tuned AWF replica on the GPU, fp32."""
    import torch
    sys.path.insert(0, EXP)
    sys.path.insert(0, SCRIPTS)
    import exp21_finetuned_awf as exp21
    import upstream_revision
    from huggingface_hub import constants
    from oe_inferencex import awf
    from olmoearth_pretrain.data.constants import Modality

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if exp21.REPO != MODEL_REPO:
        raise SystemExit(f"exp21 loads {exp21.REPO}; this script documents {MODEL_REPO}")
    if list(Modality.SENTINEL2_L2A.band_order) != list(bands):
        raise SystemExit(f"the encoder's band order {Modality.SENTINEL2_L2A.band_order} is not the task card's {bands}")
    if n_timesteps != awf.N_MONTHS:
        raise SystemExit(f"awf.stacks_to_sample stamps {awf.N_MONTHS} months; the task card says {n_timesteps}")
    # stacks_to_sample sizes the encoder mask from this module constant (32, the probe crops); exp21 passed 16-px
    # crops with it anyway, harmless only because fast_pass ignores the mask. Set it to the crop actually passed.
    awf.CROP = CROP
    model, w, b = exp21.load_finetuned()

    def model_fn(crops):
        _, pix = exp21.logits_grid(model, w, b, crops)           # (N, S, S, C) pixel logits
        return np.ascontiguousarray(pix.transpose(0, 3, 1, 2), dtype=np.float32)

    record = upstream_revision.load()["repos"]
    revision = _snapshot_revision(MODEL_REPO, "model.ckpt")
    pinned = record.get(MODEL_REPO, {}).get("revision")
    device = str(exp21.DEV)
    if device == "cuda":
        device += f" ({torch.cuda.get_device_name(0)})"
    info = {"repo": MODEL_REPO, "file": "model.ckpt", "revision": revision,
            "revision_in_record": pinned, "revision_matches_record": revision == pinned if pinned else None,
            "encoder_config": {"repo": "allenai/OlmoEarth-v1-Base",
                               "revision": upstream_revision.cached_revision(constants.HF_HUB_CACHE,
                                                                             "allenai/OlmoEarth-v1-Base", "model")},
            "code": "exp/exp21_finetuned_awf.py load_finetuned (encoder keys loaded strictly into olmoearth_pretrain's "
                    "v1-Base; the 1x1 head) and logits_grid (tokens mean-pooled over timesteps and band sets, head on "
                    "patch features, bilinear x4 upsampling as rslearn's Upsample)",
            "n_classes": int(w.shape[0]), "device": device, "dtype": "float32",
            "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32}}
    if pinned and revision != pinned:
        print(f"WARNING: loaded {MODEL_REPO}@{revision}, the record pins {pinned} (exp21's run)")
    return model_fn, info


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-class scores of Ai2's fine-tuned AWF model over a small area.")
    ap.add_argument("--out", required=True, help="output directory (scores.tif, manifest.json)")
    ap.add_argument("--lat", type=float, default=DEFAULTS["lat"], help="centre latitude (default Namanga, -2.55)")
    ap.add_argument("--lon", type=float, default=DEFAULTS["lon"], help="centre longitude (default 36.81)")
    ap.add_argument("--size", type=int, default=DEFAULTS["size"], help="side in 10 m pixels (default 512)")
    ap.add_argument("--start", default=DEFAULTS["start"], help="date window start (default 2023-01-01)")
    ap.add_argument("--end", default=DEFAULTS["end"], help="date window end (default 2023-12-31)")
    ap.add_argument("--probabilities", action="store_true", help="write softmax probabilities instead of logits")
    ap.add_argument("--batch", type=int, default=64, help="crops per forward pass (default 64)")
    a = ap.parse_args(argv)
    request = {"lat": a.lat, "lon": a.lon, "size_px": a.size, "start": a.start, "end": a.end}
    print(f"request {request}")
    card = load_task_card()
    names, bands, n_t = card_facts(card)
    recipe = load_query_config(card)
    if recipe["max_matches"] != n_t:
        raise SystemExit(f"dataset.json stacks {recipe['max_matches']} periods; the task card says {n_t}")
    res = float(card["windows"].get("window_resolution") or 10.0)
    grid = area_grid(a.lat, a.lon, a.size, res)
    print(f"task card {card['name']}: {len(names)} channels, {n_t} periods of {recipe['period_days']} days; "
          f"grid {grid['crs']} {grid['width']}x{grid['height']} at {res} m")
    t0 = time.time()
    stack, valid, imagery = fetch_stack(grid, parse_day(a.start), parse_day(a.end), n_t, recipe["period_days"], bands)
    print(f"imagery fetched in {time.time() - t0:.0f}s; {int((~valid).sum())} pixels lack a period")
    model_fn, info = build_awf_model(bands, n_t)
    produce(a.out, grid, stack, valid, imagery, model_fn, info, card, recipe, request,
            probabilities=a.probabilities, batch=a.batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
