"""exp98: the record's re-run of Ai2's FT-AWF deployment configuration over the AWF project's request geometry (2023,
10 m, probabilities written), graded at Ai2's own AWF labels. It is not the map Ai2 published.

Preregistered in docs/plan/awf_deployment.md; read that page first, this file is the run. The map is the output of
olmoearth_run (Ai2's FT-AWF checkpoint and AWF configuration, re-run by the record's jobs with the segmentation task
writing its softmax), read by `oe-inferencex from-olmoearth` in parts (exp/jobs/e98_read.sh), one directory per part:

    scores_<EPSG>.tif           10 float32 bands, the softmax over the decoder's 10 channels, NaN where no window predicted
    condition_<EPSG>.tif        int32 on the same grid, the input condition of each pixel, -1 unrecorded
    olmoearth_conditions.json   names the condition codes ("codes": {code: name})
    burned_<EPSG>.tif           optional (exp/exp98_burned.py, Part H): int32, 1 burned in 2023, 0 not, -1 unrecorded
    burned_conditions.json      its codes' names

This file reads those files; it does not import oe_inferencex.olmoearth. Several directories are read as one map:
condition names are merged (a conflict is refused), a point in two grids takes the first that covers its pixel, and a
map pixel covered in two grids counts once.

    python exp/exp98_awf_deployment.py --smoke                     # synthetic inputs end to end; no real file is read
    python exp/exp98_awf_deployment.py --inventory --scores DIR [DIR ...] --labels WINDOWS   # counts only
    python exp/exp98_awf_deployment.py --scores DIR [DIR ...] --labels WINDOWS [--replica NPZ] [--deployment-dataset DS ...]
        [--timesync CSV --boundaries GEOJSON --request-geometry GEOJSON]                      # Part I

The parts (the page's names):
  A  reproduction (graded: P1, P2): the re-run map's accuracy at Ai2's validation points against exp89's replica on
     the same points (paired; P1 an equivalence test on the task-cluster bootstrap interval), with Ai2's 89.5% beside
     it, and how often the two give the same class;
  B  ranking (graded: P3, P4): how the deployed map's own confidence ranks its errors at those points;
  C  input condition (descriptive): errors by the condition layer's code at those points;
  D  inventory (counts): how many validation and training points fall inside the deployed area;
  E  where the points sit in the map (graded: P5; the rest descriptive): their confidence among the map's pixels, the
     map's class and condition shares beside theirs, and a reweighting that is reported, never graded;
  F  what a whole-map answer needs (report-only): `plan` for an error rate within +-5 points and for certify at
     alpha 0.10, and the sequential sample that would be labelled;
  G  inputs (report-only, with --deployment-dataset): the 10 m bands at each label pixel in the deployed dataset
     against the label window's own, and the two datasets' Sentinel-2 configuration;
  H  burned area (report-only, when every part directory holds exp/exp98_burned.py's burned_<EPSG>.tif): the map's
     40 m windows by MODIS burn code (the package's pool_condition), the deployed confidence's quartiles per code and
     each code's share of the least confident 10% of windows; Ai2's validation points by the burn code of their pixel,
     with errors per code. Descriptive, no verdict; absent layers are skipped with a note.
  I  the East Africa TimeSync plots (report-only, with --timesync, --boundaries and --request-geometry): the plots of
     Bullock et al.'s simple random samples of Kenya and Tanzania inside Ai2's request geometry (47, counted before the
     run), their 2017 labels against the deployed 2023 map through the crosswalk of exp/timesync_awf_crosswalk.py
     (STRICT and LENIENT), the disagreement rate with the countries as strata weighted by each one's share of the
     request geometry and the package's exact interval, the source's own ratio estimate beside it, and counts by class.
     2017 labels against a 2023 map: a change of land cover counts as a disagreement. The inventory counts the plots
     on covered pixels only; the run reads their classes. It never stops the graded run.

Readings fixed here (the page states each one):
  - A label is read as oe_inferencex.awf reads it: a window of group spatial_split whose label raster holds exactly one
    pixel other than the fill 9, with a class 0 to 8 (exp89's single_pixel_label on exp89's window list). Where
    oe_inferencex.awf imports (the encoder extra), its list_windows is run on the same directory and must agree.
  - A label's pixel on the score grid. The label raster's own georeference gives the centre of the labelled pixel in
    the window's CRS: (col + 0.5, row + 0.5) through its transform. In the scores' CRS that centre falls in exactly one
    score pixel, floor((X - X0) / a), floor((Y - Y0) / e) for the scores' north-up transform (a, 0, X0, 0, e, Y0): the
    score pixel that contains the label pixel's centre. When the CRSs are equal the centre is used as it is; otherwise
    it is reprojected (rasterio.warp.transform) and the point is counted as reprojected. rslearn writes both the label
    windows and the deployed windows on integer pixel bounds of the same 10 m lattice of a CRS, so with equal CRSs the
    centre sits at the middle of a score pixel, offset 0.5 from its edges; the largest departure from 0.5 is recorded.
  - Covered: all ten bands finite and not all zero (RasterMerger's fill). A point on an uncovered pixel is in the grid
    but not graded.
  - Prediction: the argmax over all ten channels, as rslearn takes it; a point predicted as the untrained channel 9
    counts as an error and is counted (exp89).
  - Confidence: the top-1 softmax probability over the trained channels 0 to 8 (exp89's graded form). A softmax kept
    to a subset of its channels is the full softmax renormalised over the subset, so it is computed from the written
    probabilities exactly. The ranking reads signals.confidence(form="top1") on their logarithms, minus the log of that
    probability, which keeps the order where the probability itself rounds to 1.0 (float32 keeps the small
    probabilities down to about 1e-45). Points where every other trained probability is 0 in float32 tie; counted.
  - Clusters: the annotation task, the window name before "_point_" (exp21's and exp89's 30 validation tasks).

Outputs (exp/out, --out-dir moves them): exp98_summary.json (the run), exp98_units.npz (per point: name, split,
label, the deployed class and confidence, the condition code, the burn code (-2 without the layer), the boundary
cue; no position), exp98_inventory.json (--inventory), exp98_summary_smoke.json (--smoke). Nothing written holds a
coordinate, a transform or a bound.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
for _p in (ROOT, EXP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from oe_inferencex import assess, estimate as est, metrics, plan, sequential as sq, signals, stats   # noqa: E402
import exp89_finetuned_checkpoints as e89                                                          # noqa: E402
import exp98_burned as eb                                                                          # noqa: E402
import timesync_awf_crosswalk as tsx                                                               # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
PLAN = os.path.join(ROOT, "docs", "plan", "awf_deployment.md")
REPLICA = os.path.join(OUT, "exp89_units_awf.npz")

N_OUT = 10
TRAINED = tuple(range(9))
UNTRAINED = (9,)
FILL = 9
GROUP = "spatial_split"
# the AWF legend, olmoearth_projects @ f3c9b0c8 olmoearth_run_data/awf/olmoearth_run.yaml (inference_results_config)
CLASS_NAMES = {0: "woodland_forest", 1: "open_water", 2: "shrubland_savanna", 3: "herbaceous_wetland",
               4: "grassland_barren", 5: "agriculture_settlement", 6: "montane_forest", 7: "lava_forest",
               8: "urban_dense_development"}
AI2_ACCURACY = 0.895               # Ai2's AWF validation accuracy, olmoearth_projects docs/awf.md (344 points)
EXP89_ERROR_RATE = 41 / 344        # exp89 arm A, exp/out/exp89_summary.json
REQUEST_PERIOD = ("2023-01-01", "2023-12-31")   # the request geometry's oe_start_time and oe_end_time (Ai2's file)

SEED = 98
BUDGETS = (0.05, 0.10, 0.20)
BOOT = 2000
SMOKE_BOOT = 200
# The floors and thresholds are set on the points inside the request geometry, counted before the run from exp89's
# replica units and Ai2's geometry file (the page, "What is known before the run"): 250 validation points in 20
# annotation tasks, 24 replica errors. MIN_VAL leaves a margin below 250; MIN_ERRORS is below the 24 expected.
MIN_VAL = 200                      # P1, P2, P5 are graded only with at least this many validation points inside
MIN_ERRORS = 15                    # P3, P4 also need this many errors among them
THRESHOLDS = {"P1": 0.050, "P2": 0.90, "P3": 0.75, "P4": 0.25, "P5": 0.55}
P1_LEVEL = 0.90                    # P1 is an equivalence test: the two-sided 90% task-cluster bootstrap interval of the
                                   # paired accuracy difference lies within +-THRESHOLDS["P1"]
P1_POINT_CHECK = 0.020             # exp89's 2.0-point alignment check on the point difference: reported, not graded
P5_NULL = 0.5                      # P5 also needs the one-sided 95% task-cluster bootstrap lower bound above this
PREDICTIONS = tuple(THRESHOLDS)
EPS = 1e-12

MAP_SAMPLE = 2_000_000             # pixels kept (Bernoulli, seed 98) for the map's confidence distribution
STRIP_ROWS = 256
FLOOR = float(np.finfo(np.float32).smallest_subnormal)
SUM_TOLERANCE = 1e-3
N_STRATA = est.N_STRATA            # the reweighting's confidence strata, quintiles of the map

PLAN_WIDTH = 0.10                  # an error-rate interval of +-5 points
PLAN_ALPHA = 0.10
PLAN_COVERAGES = (0.5, 0.8, 1.0)
PLAN_ZONE_DRAWS = plan.ZONE_DRAWS
SMOKE_ZONE_DRAWS = 40
SMOKE_MAX_LABELS = 2000

S2_LAYER = "sentinel2"
TEN_M = ("B02", "B03", "B04", "B08")
N_GROUPS = 12

# keys that would carry a position; nothing written may hold one (checked before every write)
FORBIDDEN_KEYS = {"transform", "bounds", "bbox", "lon", "lat", "lonlat", "longitude", "latitude", "x", "y", "xy",
                  "coordinates", "geometry", "origin", "centre", "center", "row", "col", "rows", "cols"}


# ----------------------------------------------------------------------------- small helpers
def prereg_status(text=None):
    """'frozen', 'draft' or 'unknown', from this experiment's page (exp89's reader on another page)."""
    if text is None:
        with open(PLAN, encoding="utf-8") as f:
            text = f.read()
    return e89.prereg_status(text)


def bkey(b):
    return e89.bkey(b)


def check_no_coordinates(obj, path=""):
    """Refuse to write an object that carries a key naming a position."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in FORBIDDEN_KEYS:
                raise ValueError(f"refusing to write {path}/{k}: a key that would carry a position")
            check_no_coordinates(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            check_no_coordinates(v, f"{path}[{i}]")


def dump(obj, path):
    check_no_coordinates(obj)
    e89.dump(obj, path)


def finite(x):
    return x is not None and isinstance(x, (int, float, np.floating, np.integer)) and math.isfinite(float(x))


# ----------------------------------------------------------------------------- the deployed map's files
def find_outputs(scores):
    """The grids of one or more from-olmoearth directories (a deployment this size is read in parts, each part to its
    own directory: the reader holds at most 4 GiB of one grid). A list of paths is read in order; the condition names
    are merged across them and a code named differently in two of them is refused. Returns ([{label, scores,
    condition or None}], {code: name} or None, [per-directory metadata])."""
    paths = [scores] if isinstance(scores, str) else list(scores)
    if not paths:
        raise FileNotFoundError("no from-olmoearth directory given")
    entries, names, metas = [], None, []
    for i, path in enumerate(paths):
        e, n, m = _find_outputs_one(path)
        for x in e:
            x["label"] = f"{i}:{x['label']}" if len(paths) > 1 else x["label"]
        entries += e
        metas.append(m)
        if n is not None:
            names = dict(names or {})
            for code, name in n.items():
                if code in names and names[code] != name:
                    raise ValueError(f"condition code {code} is named {names[code]!r} in one from-olmoearth directory "
                                     f"and {name!r} in {path}: the parts were not read with the same inputs")
                names[code] = name
    return entries, names, metas


def _find_outputs_one(scores):
    """[{label, scores, condition or None}] from a from-olmoearth directory (or one scores file), and the condition
    names {code: name} from olmoearth_conditions.json beside it (None without it)."""
    if os.path.isfile(scores):
        d, files = os.path.dirname(os.path.abspath(scores)), [scores]
    else:
        d, files = scores, sorted(glob.glob(os.path.join(glob.escape(scores), "scores_*.tif")))
    if not files:
        raise FileNotFoundError(f"no scores_<EPSG>.tif in {scores}: run `oe-inferencex from-olmoearth` on the run first")
    out = []
    for f in files:
        label = re.sub(r"^scores_|\.tif$", "", os.path.basename(f))
        cond = os.path.join(d, f"condition_{label}.tif")
        burned = os.path.join(d, f"burned_{label}.tif")              # exp/exp98_burned.py, for Part H
        out.append({"label": label, "scores": f, "condition": cond if os.path.exists(cond) else None,
                    "burned": burned if os.path.exists(burned) else None})
    names = None
    cj = os.path.join(d, "olmoearth_conditions.json")
    if os.path.exists(cj):
        with open(cj) as f:
            codes = (json.load(f) or {}).get("codes") or {}
        names = {int(k): str(v) for k, v in codes.items()}
    meta = {}
    sj = os.path.join(d, "olmoearth_output.json")
    if os.path.exists(sj):
        with open(sj) as f:
            s = json.load(f) or {}
        meta = {k: s.get(k) for k in ("kind", "bands", "windows_read", "largest_sum_deviation", "uncovered_pixels",
                                      "source")}
        meta["windows_skipped"] = len(s.get("windows_skipped") or [])
    bj = os.path.join(d, eb.NAMES_JSON)
    meta["burned_conditions"] = None
    if os.path.exists(bj):
        try:              # Part H is report-only: an unreadable names file leaves it out (burned_inputs), no refusal
            with open(bj) as f:
                b = json.load(f) or {}
            meta["burned_conditions"] = {"codes": b.get("codes"), "condition_names": b.get("condition_names"),
                                         "months_missing": (b.get("source") or {}).get("months_missing")}
        except (OSError, ValueError, AttributeError):
            meta["burned_conditions"] = None
    return out, names, meta


def open_grid(entry):
    """The grid's CRS, transform and shape (kept in memory, never written)."""
    import rasterio
    with rasterio.open(entry["scores"]) as src:
        if src.count != N_OUT:
            raise ValueError(f"{entry['scores']} holds {src.count} bands; the AWF decoder writes {N_OUT}")
        t = src.transform
        if t.b != 0 or t.d != 0:
            raise ValueError(f"{entry['scores']} is rotated; rslearn writes north-up rasters")
        g = dict(entry, crs=src.crs, transform=t, shape=(src.height, src.width), dtype=src.dtypes[0])
    if g["condition"]:
        with rasterio.open(g["condition"]) as src:
            if (src.height, src.width) != g["shape"] or src.transform != g["transform"] or src.crs != g["crs"]:
                raise ValueError(f"{g['condition']} is not on the grid of {g['scores']}")
    if g.get("burned"):
        # Part H is report-only: a burned layer off the grid or unreadable is left out with a note, never a refusal
        name = os.path.basename(g["burned"])
        try:
            with rasterio.open(g["burned"]) as src:
                off = (src.height, src.width) != g["shape"] or src.transform != g["transform"] \
                    or src.crs != g["crs"] or src.count != 1
        except Exception as ex:  # noqa: BLE001
            g["burned"] = None
            g["burned_note"] = f"{name} could not be opened ({type(ex).__name__})"
        else:
            if off:
                g["burned"] = None
                g["burned_note"] = f"{name} is not one band on the scores' grid"
    return g


# ----------------------------------------------------------------------------- the label windows
def labels_group_dir(labels):
    """The spatial_split group directory, given it or the windows root above it."""
    if os.path.isdir(os.path.join(labels, GROUP)):
        return os.path.join(labels, GROUP)
    if os.path.basename(os.path.normpath(labels)) == GROUP:
        return labels
    raise FileNotFoundError(f"{labels} is neither the AWF windows root (holding {GROUP}/) nor {GROUP}/ itself")


def read_label_windows(labels):
    """(points, summary). One point per kept window: name, split, task, label, (r, c), the CRS and the centre of the
    labelled pixel (in memory only), the projection pixel of that centre, the time range. exp89's window list and
    single-pixel rule; the label raster's own georeference."""
    import rasterio
    gdir = labels_group_dir(labels)
    root = os.path.dirname(gdir)
    extraction = os.path.dirname(os.path.dirname(root))         # exp89's E89_DATA/awf above dataset/windows
    gone = e89.manifest_missing(extraction)
    if gone:
        raise FileNotFoundError(f"{len(gone)} files of exp89's extraction under {extraction} are gone or changed (the "
                                "30-day scratch purge); extract the pinned tar again (exp/jobs/e89.sh, E89_MODE=inv "
                                "E89_ARM=awf) before reading the labels")
    points, dropped, splits, crss, ranges = [], {}, {}, {}, {}
    for w in e89.list_windows(root, GROUP):
        path = os.path.join(w["dir"], "layers", "label", "category", "geotiff.tif")
        if not os.path.exists(path):
            dropped["no label raster"] = dropped.get("no label raster", 0) + 1
            continue
        with rasterio.open(path) as src:
            lab, crs, tr = src.read(1), src.crs, src.transform
        unit, why = e89.single_pixel_label(lab, FILL, TRAINED)
        if unit is None:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        r, c, cat = unit
        X = tr.c + (c + 0.5) * tr.a + (r + 0.5) * tr.b       # the centre of the labelled pixel, in the window's CRS
        Y = tr.f + (c + 0.5) * tr.d + (r + 0.5) * tr.e
        meta = w["meta"]
        pr = meta.get("projection") or {}
        b = meta.get("bounds")
        xp = yp = None
        if b is not None and "x_resolution" in pr:
            xp, yp = int(b[0]) + c, int(b[1]) + r            # the labelled pixel in projection pixel units
            mx, my = (xp + 0.5) * float(pr["x_resolution"]), (yp + 0.5) * float(pr["y_resolution"])
            if abs(mx - X) > 1e-3 or abs(my - Y) > 1e-3:
                dropped["label raster off its window's metadata"] = dropped.get(
                    "label raster off its window's metadata", 0) + 1
                continue
        tr_meta = meta.get("time_range") or [None, None]
        rng = f"{str(tr_meta[0])[:10]}/{str(tr_meta[1])[:10]}"
        split = w["split"]
        points.append({"name": w["name"], "dir": w["dir"], "split": split, "task": w["name"].split("_point_")[0],
                       "label": int(cat), "rc": (int(r), int(c)), "crs": crs, "X": float(X), "Y": float(Y),
                       "proj_crs": pr.get("crs"), "proj_res": (pr.get("x_resolution"), pr.get("y_resolution")),
                       "pix": (xp, yp), "time_range": rng})
        splits[split] = splits.get(split, 0) + 1
        crss[str(crs)] = crss.get(str(crs), 0) + 1
        ranges[rng] = ranges.get(rng, 0) + 1
    summary = {"group": GROUP, "n_windows_kept": len(points), "by_split": splits, "dropped": dropped,
               "crs": crss, "time_ranges": ranges,
               "time_range_matches_request": {"request": "/".join(REQUEST_PERIOD),
                                              "n_matching": int(ranges.get("/".join(REQUEST_PERIOD), 0))},
               "exp89_manifest": "checked" if gone is not None else "none (not exp89's extraction)",
               "awf_reader_check": awf_reader_check(gdir, points)}
    return points, summary


def awf_reader_check(gdir, points):
    """oe_inferencex.awf.list_windows on the same directory, where it imports (it needs the encoder extra)."""
    try:
        from oe_inferencex import awf
    except ImportError as ex:
        return {"run": False, "reason": f"oe_inferencex.awf does not import here ({type(ex).__name__})"}
    theirs = {(os.path.basename(w), s, r, c, k) for w, s, r, c, k in awf.list_windows(gdir)}
    ours = {(p["name"], p["split"], p["rc"][0], p["rc"][1], p["label"]) for p in points}
    return {"run": True, "n_awf": len(theirs), "n_here": len(ours), "agree": theirs == ours,
            "only_awf": len(theirs - ours), "only_here": len(ours - theirs)}


# ----------------------------------------------------------------------------- a label's pixel on the score grid
def locate_all(points, grids):
    """Per point, every grid whose extent holds it, in the grids' order: [(g, row, col, how, offset)], how "same_crs"
    or "reprojected", offset the largest departure of the label pixel's centre from the middle of the score pixel, in
    pixels. Grids read in parts can overlap (windows on a partition edge, parts that share a window row), so a point
    can sit in two."""
    from rasterio.warp import transform as warp
    cands = [[] for _ in points]
    by_crs = {}
    for k, p in enumerate(points):
        by_crs.setdefault(p["crs"].to_string(), []).append(k)
    for crs_s, idx in by_crs.items():
        idx = np.asarray(idx)
        X = np.array([points[k]["X"] for k in idx])
        Y = np.array([points[k]["Y"] for k in idx])
        for g, G in enumerate(grids):
            same = G["crs"].to_string() == crs_s
            if same:
                xs, ys = X, Y
            else:
                xs, ys = (np.asarray(v, dtype=np.float64) for v in warp(points[idx[0]]["crs"], G["crs"], X, Y))
            t = G["transform"]
            cf, rf = (xs - t.c) / t.a, (ys - t.f) / t.e
            ci, ri = np.floor(cf).astype(np.int64), np.floor(rf).astype(np.int64)
            H, W = G["shape"]
            inside = (ri >= 0) & (ri < H) & (ci >= 0) & (ci < W)
            for j in np.flatnonzero(inside):
                cands[idx[j]].append((g, int(ri[j]), int(ci[j]), "same_crs" if same else "reprojected",
                                      float(max(abs(cf[j] - ci[j] - 0.5), abs(rf[j] - ri[j] - 0.5)))))
    for c in cands:
        c.sort(key=lambda x: (x[3] != "same_crs", x[0]))        # a grid in the label's own CRS first
    return cands


def _first(cands, n):
    gi = np.full(n, -1, np.int64)
    rows, cols = np.zeros(n, np.int64), np.zeros(n, np.int64)
    how = np.array(["outside"] * n, dtype=object)
    off = np.full(n, np.nan)
    for k, c in enumerate(cands):
        if c:
            gi[k], rows[k], cols[k], how[k], off[k] = c[0]
    return gi, rows, cols, how, off


def locate(points, grids):
    """Per point: the first grid that holds it (-1 when outside every grid), its score pixel, how it was found
    ("same_crs", "reprojected" or "outside") and the offset (locate_all). read_points then moves a point to a later grid
    that covers its pixel where the first does not."""
    return _first(locate_all(points, grids), len(points))


def _read_block(src, r, c):
    from rasterio.windows import Window
    return src.read(window=Window(int(c) - 1, int(r) - 1, 3, 3), boundless=True, fill_value=np.nan).astype(np.float64)


def read_points(grids, cands):
    """Per point, the grid that covers its pixel: the first candidate whose ten bands there are covered, else the first
    candidate (in a grid, uncovered). Returns (gi, rows, cols, how, off, probabilities (N, 10) at the label pixel, the
    3x3 blocks (N, 10, 3, 3), condition codes (N,), -2 where the grid has no condition layer, overlap counts). NaN
    outside every grid."""
    import rasterio
    from rasterio.windows import Window
    n = len(cands)
    P = np.full((n, N_OUT), np.nan, np.float64)
    B = np.full((n, N_OUT, 3, 3), np.nan, np.float64)
    C = np.full(n, -2, np.int64)
    chosen = [c[0] if c else None for c in cands]
    srcs = {}
    overlap = {"in_two_or_more_grids": 0, "covered_in_two_or_more": 0, "covered_in_two_differing": 0,
               "covered_only_in_a_later_grid": 0}
    try:
        def src(g):
            if g not in srcs:
                srcs[g] = rasterio.open(grids[g]["scores"])
            return srcs[g]
        for k, cs in enumerate(cands):
            if not cs:
                continue
            blocks = [_read_block(src(g), r, c) for g, r, c, _, _ in cs]
            cov = [bool(covered(b[:, 1, 1][None])[0]) for b in blocks]
            if len(cs) > 1:
                overlap["in_two_or_more_grids"] += 1
                vals = [b[:, 1, 1] for b, ok in zip(blocks, cov) if ok]
                if len(vals) > 1:
                    overlap["covered_in_two_or_more"] += 1
                    if any(not np.allclose(v, vals[0], rtol=0, atol=1e-6) for v in vals[1:]):
                        overlap["covered_in_two_differing"] += 1
                if not cov[0] and any(cov):
                    overlap["covered_only_in_a_later_grid"] += 1
            j = cov.index(True) if any(cov) else 0
            chosen[k] = cs[j]
            B[k] = blocks[j]
            P[k] = blocks[j][:, 1, 1]
    finally:
        for s_ in srcs.values():
            s_.close()
    gi, rows, cols, how, off = _first([[c] if c else [] for c in chosen], n)
    for g, G in enumerate(grids):
        idx = np.flatnonzero(gi == g)
        if idx.size and G["condition"]:
            with rasterio.open(G["condition"]) as s_:
                for k in idx:
                    C[k] = int(s_.read(1, window=Window(int(cols[k]), int(rows[k]), 1, 1))[0, 0])
    return gi, rows, cols, how, off, P, B, C, overlap


def codes_at(grids, gi, rows, cols, key):
    """Per point, the value of a one-band int layer (`key`, e.g. "burned") at its pixel in the grid read_points chose;
    -2 outside every grid or where that grid has no such layer."""
    import rasterio
    from rasterio.windows import Window
    out = np.full(len(gi), -2, np.int64)
    for g, G in enumerate(grids):
        idx = np.flatnonzero(np.asarray(gi) == g)
        if idx.size and G.get(key):
            with rasterio.open(G[key]) as s_:
                for k in idx:
                    out[k] = int(s_.read(1, window=Window(int(cols[k]), int(rows[k]), 1, 1))[0, 0])
    return out


def covered(P):
    """A pixel holds a prediction: every band finite and not all zero (the fill RasterMerger leaves)."""
    P = np.asarray(P, dtype=np.float64)
    return np.isfinite(P).all(-1) & (np.nan_to_num(P) != 0).any(-1)


# ----------------------------------------------------------------------------- readings from probabilities
def log_trained(P):
    """(C_trained, 1, N) log-probabilities of the trained channels, floored at float32's smallest subnormal so that a
    probability float32 rounded to 0 stays finite; a softmax of these is the full softmax renormalised over them."""
    Pt = np.asarray(P, dtype=np.float64)[:, list(TRAINED)]
    return np.log(np.maximum(Pt, FLOOR)).T[:, None, :]


def readings(P):
    """Per point: the prediction over all ten channels, the suspicion (minus the log of the top-1 probability over
    the trained channels, signals.confidence form "top1"; higher = more suspect), that probability, and counts."""
    P = np.asarray(P, dtype=np.float64)
    pred = P.argmax(1)
    u = signals.confidence(log_trained(P), form="top1").ravel()
    p1 = np.exp(-u)
    Pt = P[:, list(TRAINED)]
    top = Pt.argmax(1)
    others = Pt.copy()
    others[np.arange(len(P)), top] = 0.0
    sums = P.sum(1)
    return {"pred": pred, "u": u, "p1": p1, "sum_dev": float(np.abs(sums - 1).max()) if len(P) else 0.0,
            "n_pred_untrained": int(np.isin(pred, UNTRAINED).sum()),
            "n_saturated_float32": int((others == 0).all(1).sum()),
            "n_p1_rounds_to_one": int((p1 >= 1.0).sum())}


def boundary_at(B):
    """exp21's boundary cue at the point: the share of its 8 neighbours whose deployed class differs
    (signals.boundary_indicator on the 3x3 argmax; an uncovered neighbour counts as different)."""
    out = np.full(len(B), np.nan)
    for k, b in enumerate(B):
        ok = np.isfinite(b).all(0) & (np.nan_to_num(b) != 0).any(0)
        hard = np.where(ok, np.nan_to_num(b).argmax(0), -1)
        out[k] = signals.boundary_indicator(hard)[1, 1]
    return out


# ----------------------------------------------------------------------------- the replica (exp89 arm A)
def read_replica(path):
    """{name: (label, pred over all ten channels, suspicion over the trained channels, cluster)} from exp89's units."""
    z = np.load(path, allow_pickle=False)
    L = np.asarray(z["logits"], dtype=np.float64)
    pred = L.argmax(1)
    u = signals.confidence(L[:, list(TRAINED)].T[:, None, :], form="top1").ravel()
    return {str(n): (int(lab), int(p), float(s), str(c))
            for n, lab, p, s, c in zip(z["names"], z["label"], pred, u, z["clusters"])}


# ----------------------------------------------------------------------------- the map, strip by strip
def grid_overlaps(grids):
    """Per grid, the earlier grids it overlaps on the same pixel lattice: [(earlier grid, row shift, col shift)], pixel
    (r, c) of this grid being pixel (r + dr, c + dc) of the earlier one. An overlap in one CRS that cannot be matched
    pixel for pixel (another pixel size, a lattice shifted by a fraction of a pixel) is counted, not deduplicated;
    grids in different CRSs are not compared."""
    def extent(G):
        t, (H, W) = G["transform"], G["shape"]
        return sorted((t.c, t.c + W * t.a)), sorted((t.f, t.f + H * t.e))

    def meet(x, y):
        return x[0][0] < y[0][1] and y[0][0] < x[0][1] and x[1][0] < y[1][1] and y[1][0] < x[1][1]

    same, unmatched = [[] for _ in grids], 0
    for g, G in enumerate(grids):
        tg, (Hg, Wg) = G["transform"], G["shape"]
        for h in range(g):
            E = grids[h]
            if E["crs"].to_string() != G["crs"].to_string() or not meet(extent(G), extent(E)):
                continue
            te = E["transform"]
            dc, dr = (tg.c - te.c) / tg.a, (tg.f - te.f) / tg.e
            if (tg.a, tg.e) != (te.a, te.e) or abs(dc - round(dc)) > 1e-6 or abs(dr - round(dr)) > 1e-6:
                unmatched += 1
                continue
            same[g].append((E, int(round(dr)), int(round(dc))))
    return same, unmatched


def map_pass(grid, q, rng, earlier=(), burn_patch=None):
    """One pass over a grid: covered pixels, the deployed class counts, the condition counts, a Bernoulli sample of
    the suspicion (probability q; q >= 1 keeps every pixel), the largest |sum - 1|. A pixel covered in an earlier grid
    it overlaps (`earlier`: [(grid, row shift, col shift)], grid_overlaps) was counted there and is skipped here, so
    each map pixel counts once. With `burn_patch` and the grid's burned layer, also Part H's windows (burn_windows)
    and the burned codes' pixel counts. Part H is report-only: a failure opening or reading the burned layer, or a
    refusal from burn_windows, stops only Part H's collection for this grid ("burn": {"error": ...}); the pass goes on."""
    import rasterio
    from rasterio.windows import Window
    H, W = grid["shape"]
    n_cov, cls, n_dup = 0, np.zeros(N_OUT, np.int64), 0
    cond_counts, keep, dev, n_sat = {}, [], 0.0, 0
    burn = bool(burn_patch and grid.get("burned"))
    burn_counts, burn_s, burn_c, burn_dropped, burn_error = {}, [], [], 0, None

    def burn_failed(ex):
        nonlocal bsrc, burn_error
        burn_error = f"{os.path.basename(grid['burned'])}: {ex!r}"
        if bsrc is not None:
            try:
                bsrc.close()
            except Exception:  # noqa: BLE001
                pass
        bsrc = None

    with rasterio.open(grid["scores"]) as src:
        csrc = rasterio.open(grid["condition"]) if grid["condition"] else None
        bsrc = None
        if burn:
            try:
                bsrc = rasterio.open(grid["burned"])
            except Exception as ex:  # noqa: BLE001
                burn_failed(ex)
        esrc = [(rasterio.open(Gh["scores"]), Gh["shape"], dr, dc) for Gh, dr, dc in earlier]
        try:
            for r0 in range(0, H, STRIP_ROWS):
                h = min(STRIP_ROWS, H - r0)
                a = src.read(window=Window(0, r0, W, h)).astype(np.float32)
                cov = np.isfinite(a).all(0) & (np.nan_to_num(a) != 0).any(0)
                dup = np.zeros_like(cov)
                for es, (He, We), dr, dc in esrc:
                    ra, rb = max(r0 + dr, 0), min(r0 + h + dr, He)
                    ca, cb = max(dc, 0), min(W + dc, We)
                    if ra >= rb or ca >= cb:
                        continue
                    e = es.read(window=Window(ca, ra, cb - ca, rb - ra)).astype(np.float32)
                    ecov = np.isfinite(e).all(0) & (np.nan_to_num(e) != 0).any(0)
                    sub = cov[ra - r0 - dr:rb - r0 - dr, ca - dc:cb - dc]
                    n_dup += int((sub & ecov).sum())
                    dup[ra - r0 - dr:rb - r0 - dr, ca - dc:cb - dc] |= sub & ecov
                    sub &= ~ecov                                 # a view: clears those pixels in cov
                m = int(cov.sum())
                if not m:
                    if bsrc is not None and dup.any():           # windows wholly in an earlier grid, counted
                        try:
                            burn_dropped += burn_windows(np.full((h, W), np.nan), cov, dup,
                                                         bsrc.read(1, window=Window(0, r0, W, h)), burn_patch)[2]
                        except Exception as ex:  # noqa: BLE001
                            burn_failed(ex)
                    continue
                n_cov += m
                v = a[:, cov]
                cls += np.bincount(v.argmax(0), minlength=N_OUT)
                dev = max(dev, float(np.abs(v.astype(np.float64).sum(0) - 1).max()))
                vt = v[list(TRAINED)]
                srt = np.sort(vt, axis=0)
                n_sat += int((srt[-2] == 0).sum())
                u = signals.confidence(np.log(np.maximum(vt.astype(np.float64), FLOOR))[:, None, :], form="top1")[0]
                keep.append(u if q >= 1 else u[rng.random(m) < q])
                if csrc is not None:
                    c = csrc.read(1, window=Window(0, r0, W, h))[cov]
                    vals, cnt = np.unique(c, return_counts=True)
                    for vv, cc in zip(vals, cnt):
                        cond_counts[int(vv)] = cond_counts.get(int(vv), 0) + int(cc)
                if bsrc is not None:
                    try:
                        b = bsrc.read(1, window=Window(0, r0, W, h))
                        U = np.full((h, W), np.nan)
                        U[cov] = u
                        s_w, c_w, nd = burn_windows(U, cov, dup, b, burn_patch)
                    except Exception as ex:  # noqa: BLE001
                        burn_failed(ex)
                    else:
                        vals, cnt = np.unique(b[cov], return_counts=True)
                        for vv, cc in zip(vals, cnt):
                            burn_counts[int(vv)] = burn_counts.get(int(vv), 0) + int(cc)
                        burn_s.append(s_w)
                        burn_c.append(c_w)
                        burn_dropped += nd
        finally:
            for s_ in [csrc, bsrc] + [x[0] for x in esrc]:
                if s_ is not None:
                    s_.close()
    sample = np.sort(np.concatenate(keep)) if keep else np.zeros(0)
    out = {"n_covered": n_cov, "class_counts": cls, "condition_counts": cond_counts, "sample": sample,
           "largest_sum_deviation": dev, "n_saturated_float32": n_sat, "n_counted_in_an_earlier_grid": n_dup}
    if burn and burn_error is not None:
        out["burn"] = {"error": burn_error}
    elif burn:
        out["burn"] = {"pixel_counts": burn_counts, "suspicion": np.concatenate(burn_s) if burn_s else np.zeros(0),
                       "codes": np.concatenate(burn_c) if burn_c else np.zeros(0, np.int64),
                       "n_windows_dropped_overlap": burn_dropped}
    return out


def percentile_in_map(sample, u):
    """Per point: the share of the map's pixels (its sample) less confident than the point, ties counted half."""
    lo = np.searchsorted(sample, u, side="left")
    hi = np.searchsorted(sample, u, side="right")
    return (sample.size - hi + 0.5 * (hi - lo)) / max(sample.size, 1)


def cluster_bootstrap(fn, clusters, n_boot, seed=SEED):
    """The sorted values of fn(index array) over resamples of the clusters (tasks) with replacement."""
    rng = np.random.default_rng(seed)
    cl = np.asarray(clusters)
    ids = np.unique(cl)
    idx = {c: np.flatnonzero(cl == c) for c in ids}
    vals = []
    for _ in range(n_boot):
        pick = ids[rng.integers(0, len(ids), len(ids))]
        v = fn(np.concatenate([idx[c] for c in pick]))
        if np.isfinite(v):
            vals.append(v)
    return np.sort(np.asarray(vals, dtype=np.float64))


def p1_interval(e_d, e_r, clusters, n_boot, level=P1_LEVEL, seed=SEED):
    """P1's equivalence interval: the two-sided `level` task-cluster bootstrap interval of deployed accuracy minus the
    replica's on the same points (the paired difference, both read on each resample), and its larger end in absolute
    value, which P1 compares with its margin."""
    e_d, e_r = np.asarray(e_d, dtype=np.float64), np.asarray(e_r, dtype=np.float64)
    d = cluster_bootstrap(lambda i: float(e_r[i].mean() - e_d[i].mean()), clusters, n_boot, seed)
    if not d.size:
        return {"n_resamples": 0}
    lo, hi = (float(np.quantile(d, q)) for q in ((1 - level) / 2, (1 + level) / 2))
    return {"level": level, "n_resamples": int(d.size), "lo": lo, "hi": hi, "max_abs_end": max(abs(lo), abs(hi)),
            "n_clusters": int(np.unique(np.asarray(clusters)).size)}


def p5_interval(pct, clusters, n_boot, seed=SEED):
    """P5's uncertainty: the task-cluster bootstrap of the points' mean percentile, its one-sided 95% lower bound (the
    5th percentile of the resamples) and the two-sided 90% interval."""
    pct = np.asarray(pct, dtype=np.float64)
    v = cluster_bootstrap(lambda i: float(pct[i].mean()), clusters, n_boot, seed)
    if not v.size:
        return {"n_resamples": 0}
    return {"n_resamples": int(v.size), "lower_one_sided_95": float(np.quantile(v, 0.05)),
            "hi_one_sided_95": float(np.quantile(v, 0.95)), "n_clusters": int(np.unique(np.asarray(clusters)).size)}


# ----------------------------------------------------------------------------- measures
def ranking_block(u, err, clusters, n_boot, seed=SEED):
    """exp89's signal table (AUROC, AURC, excess AURC, gap closed, capture at 5/10/20% beside random and the
    ceiling) and cluster bootstraps of the AUROC's lead over chance and of the 10% capture's lead over random."""
    err = np.asarray(err, dtype=np.float64)
    n = err.size
    if n == 0 or err.sum() == 0 or err.sum() == n:
        return {"n": int(n), "n_errors": int(err.sum()), "note": "no ranking: no error or no correct point"}
    t = e89.signal_table(u, err)
    auroc = lambda idx: metrics.weighted_auroc(np.asarray(u)[idx], err[idx], np.ones(len(idx)))
    t["auroc_minus_chance_bootstrap"] = stats.paired_cluster_bootstrap(
        auroc, clusters, lambda idx: 0.5, clusters, n_boot=n_boot, seed=seed, min_units=20)
    t["capture_0.1_minus_random_bootstrap"] = e89.capture_bootstrap(u, np.zeros(n), err, clusters, 0.10, n_boot, seed)
    t["n"], t["n_errors"] = int(n), int(err.sum())
    return t


def condition_block(codes, err, names, map_counts=None):
    """Errors at the points by condition code, with the map's share of each code beside the points' (descriptive)."""
    codes = np.asarray(codes)
    err = np.asarray(err, dtype=np.float64)
    out = {}
    total_map = sum(map_counts.values()) if map_counts else 0
    for c in sorted(set(codes.tolist()) | set((map_counts or {}).keys())):
        m = codes == c
        name = "unrecorded" if c == -1 else ("no condition layer" if c == -2 else (names or {}).get(c, str(c)))
        out[name] = {"code": int(c), "n_points": int(m.sum()), "n_errors": int(err[m].sum()),
                     "error_rate": float(err[m].mean()) if m.any() else None,
                     "share_of_points": float(m.mean()) if codes.size else None,
                     "share_of_map": (map_counts.get(c, 0) / total_map) if total_map else None}
    return out


def replica_block(names, label, pred, err, u, clusters, replica, n_boot):
    """Part A against exp89's replica on the same points: accuracies, agreement, the discordant points with the exact
    sign test, a cluster bootstrap of the paired difference, the ranking of each on these points."""
    k = [i for i, nm in enumerate(names) if nm in replica]
    missing = len(names) - len(k)
    if not k:
        return {"n_matched": 0, "n_missing": missing}
    k = np.asarray(k)
    r_lab = np.array([replica[names[i]][0] for i in k])
    r_pred = np.array([replica[names[i]][1] for i in k])
    r_u = np.array([replica[names[i]][2] for i in k])
    r_cl = np.array([replica[names[i]][3] for i in k])
    lab, prd, e_d, cl = np.asarray(label)[k], np.asarray(pred)[k], np.asarray(err, dtype=np.float64)[k], np.asarray(clusters)[k]
    e_r = (r_pred != lab).astype(np.float64)
    d_right_r_wrong = int(((e_d == 0) & (e_r == 1)).sum())
    r_right_d_wrong = int(((e_d == 1) & (e_r == 0)).sum())
    acc_d, acc_r = float(1 - e_d.mean()), float(1 - e_r.mean())
    out = {"n_matched": int(k.size), "n_missing": int(missing),
           "label_mismatches": int((r_lab != lab).sum()), "cluster_mismatches": int((r_cl != cl).sum()),
           "accuracy_deployed": acc_d, "accuracy_replica": acc_r, "difference": acc_d - acc_r,
           "n_errors_deployed": int(e_d.sum()), "n_errors_replica": int(e_r.sum()),
           "agreement": float((prd == r_pred).mean()), "n_same_class": int((prd == r_pred).sum()),
           "discordant": {"deployed_right_replica_wrong": d_right_r_wrong,
                          "replica_right_deployed_wrong": r_right_d_wrong,
                          "both_wrong_different_class": int(((e_d == 1) & (e_r == 1) & (prd != r_pred)).sum()),
                          "sign_test_two_sided_p": stats.sign_test(d_right_r_wrong, r_right_d_wrong)},
           "difference_bootstrap": stats.paired_cluster_bootstrap(
               lambda idx: 1 - e_d[idx].mean(), cl, lambda idx: 1 - e_r[idx].mean(), cl, n_boot=n_boot, seed=SEED,
               min_units=20),
           "suspicion_spearman": stats.spearman(np.asarray(u)[k], r_u),
           "p1_equivalence_interval": p1_interval(e_d, e_r, cl, n_boot),
           "within_2_points_reported": bool(abs(acc_d - acc_r) <= P1_POINT_CHECK + EPS)}
    if 0 < e_d.sum() < e_d.size and 0 < e_r.sum() < e_r.size:
        out["auroc_deployed"] = metrics.weighted_auroc(np.asarray(u)[k], e_d, np.ones(k.size))
        out["auroc_replica"] = metrics.weighted_auroc(r_u, e_r, np.ones(k.size))
        out["auroc_difference_bootstrap"] = stats.paired_cluster_bootstrap(
            lambda idx: metrics.weighted_auroc(np.asarray(u)[k][idx], e_d[idx], np.ones(len(idx))), cl,
            lambda idx: metrics.weighted_auroc(r_u[idx], e_r[idx], np.ones(len(idx))), cl, n_boot=n_boot, seed=SEED,
            min_units=20)
    return out


def reweighted(err, u_pts, sample, n_map):
    """Report-only: the points' error rate post-stratified to the map's confidence quintiles
    (estimate.stratified_mean_and_variance with the map's stratum sizes). It assumes the points are as accurate as the
    map's other pixels of the same confidence, which their design does not give; never graded."""
    if sample.size < N_STRATA or len(err) == 0:
        return {"estimate": None, "reason": "too few map pixels or points"}
    edges = np.quantile(sample, np.linspace(0, 1, N_STRATA + 1)[1:-1])
    s_map = np.searchsorted(edges, sample, side="right")
    s_pts = np.searchsorted(edges, np.asarray(u_pts), side="right")
    share = np.bincount(s_map, minlength=N_STRATA) / sample.size
    n_pts = np.bincount(s_pts, minlength=N_STRATA)
    sizes = np.maximum(np.rint(share * n_map).astype(np.int64), 0)
    out = {"strata": "quintiles of the map's confidence, stratum 0 the most confident",
           "map_share": share.tolist(), "points_per_stratum": n_pts.tolist(),
           "points_error_rate_per_stratum": [float(np.asarray(err)[s_pts == h].mean()) if n_pts[h] else None
                                             for h in range(N_STRATA)],
           "assumption": "the points are as accurate as the map's other pixels of the same confidence; their design "
                         "(expert placement) does not give this, so this is a model-based reading, not an estimate"}
    if (n_pts < est.MIN_PER_STRATUM).any():
        out.update(estimate=None, reason=f"a stratum holds fewer than {est.MIN_PER_STRATUM} points")
        return out
    e, var, starved = est.stratified_mean_and_variance(np.asarray(err, dtype=np.float64), s_pts, np.arange(len(err)),
                                                       sizes.tolist(), int(sizes.sum()))
    out.update(estimate=float(e), accuracy=float(1 - e), variance_if_random_within_strata=float(var),
               starved=int(starved))
    return out


def plan_block(n_windows, rate_points, zone_errors, draws=PLAN_ZONE_DRAWS, max_labels=plan.MAX_LABELS):
    """Part F, report-only: what `plan` says a whole-map answer needs on this map (one window per 10 m pixel)."""
    out = {"n_windows": int(n_windows), "patch_px": 1, "width": PLAN_WIDTH, "alpha": PLAN_ALPHA,
           "note": plan.PLAN_NOTE, "error_rate": {}, "zone": {}}
    rates = {"points": rate_points, "exp89_replica": EXP89_ERROR_RATE, "default_0.5": None}
    for key, r in rates.items():
        if key == "points" and not (finite(r) and 0 < r < 1):
            continue
        res = plan.plan_error_rate(n_windows, PLAN_WIDTH, r, max_labels=max_labels)
        out["error_rate"][key] = {"error_rate": res["error_rate"], "labels": res["labels"],
                                  "probability_at_labels": res["probability_at_labels"],
                                  "refusal": res.get("refusal")}
    for c in PLAN_COVERAGES:
        z = zone_errors.get(c)
        res = plan.plan_zone(n_windows, c, PLAN_ALPHA, zone_error=z if finite(z) else None, draws=draws,
                             max_labels=max_labels, seed=SEED)
        keep = {k: res.get(k) for k in ("coverage", "zone_error", "labels_to_test", "min_labels_inside", "refusal",
                                        "unplanned")}
        for k in ("bonferroni", "prefix_even"):
            if k in res:
                keep[k] = {"labels": res[k]["labels"], "probability_at_labels": res[k]["probability_at_labels"]}
        keep["zone_error_source"] = ("the error rate among the most confident share of the graded validation points "
                                     "(a guess: they are not a sample of the map)") if finite(z) else None
        out["zone"][f"{c:g}"] = keep
    out["certify_min_labels"] = est.min_labels_to_certify(PLAN_ALPHA, est.ZONE_DELTA)
    out["sequential"] = {"min_labels": sq.min_labels_sequential(PLAN_ALPHA, est.ZONE_DELTA),
                         "anchor_at_first_budget_300": sq.anchor_coverage(300, PLAN_ALPHA),
                         "command": "oe-inferencex sample scores_<EPSG>.tif --design sequential --alpha 0.1 "
                                    "--patch 1 --seed 98 --budget <B> --out <labelling sheet outside the repository>",
                         "note": "a random order of the map's pixels, labelled from the top; certify then holds at every "
                                 "look, and estimate reads any labelled prefix as a random sample. The sheet lists "
                                 "window positions and stays with the labellers, never in the repository"}
    return out


# ----------------------------------------------------------------------------- part H: burned area (report-only)
BURN_PATCH = 4                     # assess's default window (40 m); a window takes the majority code of its pixels
BURN_BUDGET = 0.10                 # the least confident 10% of windows, as P4 reads the points
BURN_QUANTILES = (0.25, 0.5, 0.75)
assert STRIP_ROWS % BURN_PATCH == 0, "a window row must not straddle two strips"


def burned_inputs(grids, metas):
    """(names {code: name} or None, note or None): Part H runs only when every grid has its burned layer on its grid
    and every part directory names the codes alike (one exp/exp98_burned.py run)."""
    missing = [G["label"] for G in grids if not G.get("burned")]
    if missing:
        notes = sorted({G["burned_note"] for G in grids if G.get("burned_note")})
        return None, (f"burned_<EPSG>.tif is absent or off the grid in {len(missing)} of {len(grids)} grids"
                      + (f" ({'; '.join(notes)})" if notes else "") + "; run exp/jobs/e98_burned.sh after e98_read.sh")
    codes = [json.dumps((m.get("burned_conditions") or {}).get("codes"), sort_keys=True) for m in metas]
    if len(set(codes)) != 1 or codes[0] == "null":
        return None, (f"{eb.NAMES_JSON} is missing or names the codes differently across the part directories; the "
                      "burned layers were not written by one run")
    return {int(k): str(v) for k, v in json.loads(codes[0]).items()}, None


def burn_windows(u, cov, dup, burned, patch=BURN_PATCH):
    """Part H's windows of one strip that starts on a window row: (suspicion of each kept window, its code, the windows
    dropped because they hold a pixel covered in an earlier grid). `cov` is the strip's coverage with the pixels
    covered in an earlier grid cleared, `dup` those pixels (map_pass). A window is patch x patch pixels; it is valid
    when at least half its pixels are covered in this grid (assess's rule) and kept when none of them was covered in
    an earlier grid, so no map area is in two windows. Its suspicion is the mean over its covered pixels of `u` (minus
    the log of the top-1 trained probability, assess's form='top1' window reading); its code is the package's
    pooling, assess.pool_condition: the code held by most of its covered pixels with a code of 0 or more, -1 on a tie
    or with none."""
    h, w = cov.shape
    if h < patch or w < patch:
        return np.zeros(0), np.zeros(0, np.int64), 0
    hh, ww = h // patch * patch, w // patch * patch

    def blocks(a):
        return a[:hh, :ww].reshape(hh // patch, patch, ww // patch, patch)
    n_cov = blocks(cov).sum(axis=(1, 3))
    s_sum = blocks(np.where(cov, u, 0.0)).sum(axis=(1, 3))
    dupw = blocks(dup).any(axis=(1, 3))
    valid = 2 * blocks(cov | dup).sum(axis=(1, 3)) >= patch * patch     # on this grid's own coverage
    try:
        code = assess.pool_condition(np.asarray(burned), patch, predicted=cov)["grid"]
    except ValueError as ex:          # a strip where no valid window has a code: every window is unrecorded
        if not str(ex).startswith("no window takes a condition"):
            raise
        code = np.full(valid.shape, -1, np.int64)
    keep = valid & ~dupw              # here cov | dup is cov, so pool_condition saw the window whole
    return s_sum[keep] / n_cov[keep], code[keep].astype(np.int64), int((valid & dupw).sum())


def burn_name(code, names):
    return "unrecorded" if code < 0 else (names or {}).get(int(code), str(int(code)))


def burned_map_block(susp, codes, names, budget=BURN_BUDGET):
    """Part H, label-free: per burn code, its windows, their share, the quartiles of the deployed confidence (the
    window's top-1 trained probability, exp(-suspicion), a geometric mean over its pixels) and its share of the least
    confident `budget` of all windows beside its share of all of them. The least confident set is the k = round(budget
    x n) windows of highest suspicion; windows tied at the cut-off are split in proportion, so the shares do not depend
    on raster order."""
    s = np.asarray(susp, dtype=np.float64)
    c = np.asarray(codes, dtype=np.int64)
    n = int(s.size)
    if n == 0:
        return {"n_windows": 0, "note": "no valid window"}
    k = min(n, max(1, int(round(budget * n))))
    cut = float(np.sort(s)[::-1][k - 1])
    above, tied = s > cut, s == cut
    n_above, n_tied = int(above.sum()), int(tied.sum())
    take = k - n_above
    per = {}
    for code in sorted(set(c.tolist())):
        m = c == code
        p = np.exp(-s[m])
        low = (int((above & m).sum()) + take * int((tied & m).sum()) / n_tied) / k
        share = float(m.mean())
        per[burn_name(code, names)] = {
            "code": int(code), "n_windows": int(m.sum()), "share_of_windows": share,
            "confidence_quartiles": {f"{q:g}": float(np.quantile(p, q)) for q in BURN_QUANTILES},
            "share_of_least_confident": float(low), "least_confident_over_all_windows": float(low / share)}
    return {"n_windows": n, "budget": budget, "n_least_confident": k, "n_tied_at_cutoff": n_tied,
            "n_tied_taken": take, "per_code": per}


def burned_points_block(codes, err, tasks, names, map_counts):
    """Part H at the validation points: per burn code of the point's pixel, the points, errors, error rate and
    annotation tasks, with the code's share of the map's covered pixels (condition_block's table). Descriptive: no
    verdict."""
    codes = np.asarray(codes, dtype=np.int64)
    out = condition_block(codes, err, names, map_counts)
    for row in out.values():
        m = codes == row["code"]
        row["n_tasks"] = int(np.unique(np.asarray(tasks)[m]).size) if m.any() else 0
    return out


BURN_LIMITS = ("A MODIS cell is about 463 m on a side, about 2,150 pixels and 134 windows of 40 m: a code says whether "
               "the cell around a window burned, not whether the window did; a scar's edge mixes both codes.",
               "MCD64A1 misses small and short-lived burns (global omission 72.6% against Landsat 8 pairs, MCD64 C6.1 "
               "user guide section 7) and burns under persistent cloud; code 0 is 'no burn detected', not 'unburned'.",
               "A month the stage could not read is a month whose burns read as code 0; its name and months_missing "
               "say which (September 2023 was absent from Planetary Computer on 2026-10-08).",
               "A burn date is a day; the model reads 12 mosaics of 30-day periods. A burned code does not say whether "
               "a scar was fresh in the scenes the model read, or had greened again by the later periods.",
               "Ai2's validation points were placed by experts; few may fall in burned cells, so their error rate per "
               "code is descriptive, with its count, and carries no verdict.")


def burned_block(names, note, parts, codes_pts, V, err, tasks, metas):
    """Part H, report-only: the map's windows and pixels by burn code (label-free) and Ai2's validation points by the
    burn code of their pixel. `parts` are map_pass's "burn" entries, one per grid."""
    out = {"status": "report-only: no threshold and no verdict (docs/plan/awf_deployment.md, Part H)",
           "source": "exp/exp98_burned.py: MODIS MCD64A1 v061 from Planetary Computer, 2023, nearest neighbour onto "
                     "each part's grid", "patch_px": BURN_PATCH, "window_code_rule": assess.RULE_TEXT,
           "cannot_show": list(BURN_LIMITS)}
    if names is None:
        out["note"] = note
        return out
    bc = (metas[0].get("burned_conditions") or {}) if metas else {}
    out.update(names={str(k): v for k, v in sorted(names.items())}, condition_names=bc.get("condition_names"),
               months_missing=bc.get("months_missing"))
    susp = np.concatenate([p["suspicion"] for p in parts]) if parts else np.zeros(0)
    codes = np.concatenate([p["codes"] for p in parts]) if parts else np.zeros(0, np.int64)
    win = burned_map_block(susp, codes, names)
    win["n_windows_dropped_overlapping_an_earlier_grid"] = int(sum(p["n_windows_dropped_overlap"] for p in parts))
    out["windows"] = win
    counts = {}
    for p in parts:
        for c, n in p["pixel_counts"].items():
            counts[c] = counts.get(c, 0) + n
    total = sum(counts.values())
    out["pixels"] = {burn_name(c, names): {"code": int(c), "n": int(n),
                                           "share_of_covered": n / total if total else None}
                     for c, n in sorted(counts.items())}
    V = np.asarray(V)
    if V.size:
        out["validation_points"] = burned_points_block(np.asarray(codes_pts)[V], np.asarray(err)[V],
                                                       np.asarray(tasks)[V], names, counts)
    else:
        out["validation_points"] = {"note": "no validation point on a covered pixel"}
    return out


# ----------------------------------------------------------------------------- part I: TimeSync plots (report-only)
TS_YEAR = 2017                     # the sample's last labelled year (Bullock et al. 2021)
TS_INSIDE = {"kenya": 33, "tanzania": 14}   # plots inside Ai2's request geometry, counted 9 October 2026 (the page)
TS_NOTE = ("2017 labels against a 2023 map: a plot whose land cover changed between the two years counts as a "
           "disagreement. The labels were interpreted from Landsat (30 m) and Google Earth by RCMRD's analysts for "
           "TimeSync's legend, the map is 10 m in AWF's legend, and the two meet through the crosswalk "
           "(exp/timesync_awf_crosswalk.py), STRICT and LENIENT. Within each country the plots are a simple random "
           "sample, so those inside the request geometry are a simple random sample of its part in that country; the "
           "countries are strata weighted by their share of the geometry's area.")


def timesync_points(plots):
    """exp98's point records for TimeSync plots: WGS84 points (in memory only), reprojected onto each score grid by
    locate_all."""
    from rasterio.crs import CRS
    wgs = CRS.from_epsg(4326)
    return [{"crs": wgs, "X": float(x), "Y": float(y)} for x, y in zip(plots["lon"], plots["lat"])]


def timesync_inside(timesync, request_geometry, boundaries=None, geometry_sha256=tsx.AWF_GEOMETRY_SHA256):
    """(the 2017 plots of Kenya and Tanzania inside the request geometry, the geometry's area by country or None). The
    geometry's sha256 is checked (None skips it, for synthetic inputs); nothing positional leaves this function but the
    plots' arrays, which stay in memory."""
    plots = tsx.read_plots(timesync, TS_YEAR)
    geom = tsx.load_request_geometry(request_geometry, geometry_sha256)
    inside = tsx.distance_km(geom, plots["lon"], plots["lat"]) == 0
    areas = tsx.region_areas(geom, boundaries) if boundaries else None
    return tsx.subset(plots, inside), areas


def majority_3x3(B):
    """Per 3 x 3 block (N, 10, 3, 3) around a point: the class most of its covered pixels take (the argmax over all ten
    channels), the lower index on a tie, -1 with no covered pixel. Reported beside the point's own pixel, since a
    TimeSync plot describes a Landsat pixel (30 m) and the map is 10 m (exp94 read Esri's 10 m map so)."""
    out = np.full(len(B), -1, np.int64)
    for k, b in enumerate(B):
        px = np.moveaxis(np.asarray(b, dtype=np.float64), 0, -1).reshape(-1, N_OUT)
        ok = covered(px)
        if ok.any():
            out[k] = int(np.bincount(px[ok].argmax(1), minlength=N_OUT).argmax())
    return out


def timesync_block(grids, plots, areas, outcome):
    """Part I. Counts by country (inside the geometry, in a grid, on a covered pixel) always; with `outcome` (the run
    only) the deployed class and confidence at each covered plot, the disagreement under each crosswalk rule by country
    and by TimeSync class, the stratified rate with the package's exact interval, the ratio estimate beside it, the
    confusion counts and the confidence's AUROC for the disagreements. Report-only."""
    countries = tuple(tsx.COUNTRIES)
    ctry = np.asarray(plots["country"])
    labels = np.asarray(plots["label"])
    cands = locate_all(timesync_points(plots), grids)
    gi, rows, cols, how, off, P, B, C, overlap = read_points(grids, cands)
    cov = covered(P)
    out = {"status": "report-only: no threshold and no verdict (docs/plan/awf_deployment.md, Part I)",
           "year": TS_YEAR, "note": TS_NOTE, "source": {k: tsx.TIMESYNC[k] for k in ("repository", "sha256", "licence",
                                                                                   "citation")},
           "crosswalk": tsx.crosswalk_record(),
           "plots": {c: {"inside_geometry": int((ctry == c).sum()), "in_a_grid": int(((ctry == c) & (gi >= 0)).sum()),
                         "covered": int(((ctry == c) & cov).sum())} for c in countries},
           "expected_inside_geometry": dict(TS_INSIDE),
           "overlapping_grids": overlap,
           "labels_inside": {t: int((labels == t).sum()) for t in tsx.TIMESYNC_CLASSES if (labels == t).any()},
           "label_changes": {"plots_with_two_or_more_labels_2015_2017": int((plots["n_labels_recent"] > 1).sum()),
                             "plots_with_two_or_more_labels_any_year": int((plots["n_labels_all_years"] > 1).sum())}}
    if areas is not None:
        tot = sum(areas["by_country_km2"].values())
        out["areas"] = {"geometry_km2": areas["region_km2"], "by_country_km2": areas["by_country_km2"],
                        "share_by_country": {c: areas["by_country_km2"][c] / tot for c in countries} if tot else None,
                        "share_outside_both_countries": areas["share_outside_countries"],
                        "source": tsx.BOUNDARIES["repository"]}
    if not outcome:
        out["mode"] = "inventory: counts only; no class or probability at a plot is read out"
        return out
    K = np.flatnonzero(cov)
    if not K.size:
        out["note_run"] = "no plot on a covered pixel"
        return out
    rd = readings(P[K])
    pred, u, p1 = rd["pred"], rd["u"], rd["p1"]
    lab, cc = labels[K], ctry[K]
    n_by = {c: int((cc == c).sum()) for c in countries}
    _, w_plot = tsx.weights_from_areas(areas["by_country_km2"], n_by) if areas is not None else (None, None)
    weights = np.array([w_plot[c] for c in cc]) if w_plot else None
    maj = majority_3x3(B[K])
    out["readings"] = {k: v for k, v in rd.items() if k not in ("pred", "u", "p1")}
    rules = {}
    for rule in ("strict", "lenient"):
        w = tsx.wrong(lab, pred, rule)
        k_by = {c: int(w[cc == c].sum()) for c in countries}
        r = {"n": int(K.size), "disagreements": int(w.sum()), "unweighted_rate": float(w.mean()),
             "by_country": {c: {"n": n_by[c], "disagreements": k_by[c]} for c in countries},
             "by_class": {t: {"n": int((lab == t).sum()), "disagreements": int(w[lab == t].sum())}
                          for t in tsx.TIMESYNC_CLASSES if (lab == t).any()},
             "three_by_three_majority_disagreements": int(tsx.wrong(lab, np.where(maj < 0, UNTRAINED[0], maj),
                                                                    rule).sum())}
        if areas is not None:
            r["stratified"] = tsx.stratified_exact(k_by, n_by, areas["by_country_km2"])
            r["ratio_estimate_reported"] = tsx.ratio_estimate(k_by, n_by, areas["country_km2"])
        if 0 < w.sum() < w.size:
            r["auroc_unweighted"] = metrics.weighted_auroc(u, w, np.ones(w.size))
            if weights is not None:
                r["auroc_weighted"] = metrics.weighted_auroc(u, w, weights)
        rules[rule] = r
    out["rules"] = rules
    pairs = {}
    for t, p in zip(lab, pred):
        key = f"{t} -> {CLASS_NAMES.get(int(p), f'channel {int(p)}')}"
        pairs[key] = pairs.get(key, 0) + 1
    out["confusion_counts"] = dict(sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0])))
    out["confidence_quartiles"] = {f"{q:g}": float(np.quantile(p1, q)) for q in (0.25, 0.5, 0.75)}
    return out


# ----------------------------------------------------------------------------- part G: inputs and configuration
def _bandset_names(dirname, layer_cfg):
    """Band names of a band-set directory: its own spelling, or the band set of the layer's config whose rslearn
    directory name (raster_format.py:32-47) it is."""
    if not re.fullmatch(r"[0-9a-f]{64}", dirname):
        return dirname.split("_")
    for bs in (layer_cfg or {}).get("band_sets") or []:
        bands = list(bs.get("bands") or [])
        if any("_" in b for b in bands):
            h = hashlib.sha256(json.dumps(bands).encode()).hexdigest()
        else:
            nm = "_".join(bands)
            h = hashlib.sha256(nm.encode()).hexdigest() if len(nm) > 64 else nm
        if h == dirname:
            return bands
    return None


def _s2_config(ds_dir):
    try:
        with open(os.path.join(ds_dir, "config.json")) as f:
            return ((json.load(f) or {}).get("layers") or {}).get(S2_LAYER)
    except (OSError, ValueError):
        return None


def config_diff(a, b, path=""):
    """The keys where two configs differ, as [(path, a, b)]."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += config_diff(a.get(k), b.get(k), f"{path}.{k}" if path else k)
        return out
    return [] if a == b else [(path, a, b)]


def _layer_dir(k):
    return S2_LAYER if k == 0 else f"{S2_LAYER}.{k}"


def compare_inputs(points, ok, err_by_point, agree_by_point, labels_root, deploy_ds):
    """Part G, report-only. For each graded point whose label window and deployed window share a CRS: the four 10 m
    bands of each of the 12 item groups at the label pixel, in the label window and in the deployed window that holds
    it. Equal everywhere means the deployment read the same scenes there. `deploy_ds` is one rslearn dataset root or a
    list of them (olmoearth_run writes one per partition); their windows are pooled, and a point in two takes the
    first that holds it."""
    import rasterio
    from rasterio.windows import Window
    roots = [deploy_ds] if isinstance(deploy_ds, str) else list(deploy_ds)
    lab_cfg = _s2_config(os.path.dirname(os.path.dirname(labels_group_dir(labels_root))))
    dep_cfgs = [_s2_config(d) for d in roots]
    dep_cfg = dep_cfgs[0]
    out = {"config": {"label_dataset": lab_cfg is not None, "deployment_dataset": dep_cfg is not None,
                      "n_deployment_roots": len(roots),
                      "deployment_roots_share_one_config": all(c == dep_cfg for c in dep_cfgs)}}
    if lab_cfg is not None and dep_cfg is not None:
        out["config"]["differences"] = [{"key": k, "label_dataset": va, "deployment_dataset": vb}
                                        for k, va, vb in config_diff(lab_cfg, dep_cfg)]
    dws = []
    for ri, root in enumerate(roots):
        for mp in sorted(glob.glob(os.path.join(glob.escape(root), "windows", "*", "*", "metadata.json"))):
            with open(mp) as f:
                m = json.load(f)
            pr = m.get("projection") or {}
            dws.append({"dir": os.path.dirname(mp), "crs": pr.get("crs"), "res": (pr.get("x_resolution"),
                        pr.get("y_resolution")), "b": m.get("bounds"), "tr": m.get("time_range"), "cfg": dep_cfgs[ri]})
    ranges = {}
    for w in dws:
        t = w["tr"] or [None, None]
        key = f"{str(t[0])[:10]}/{str(t[1])[:10]}"
        ranges[key] = ranges.get(key, 0) + 1
    out["deployment_windows"] = len(dws)
    out["deployment_time_ranges"] = ranges
    rows = []
    no_window = no_label_layer = no_deployed_layer = incomplete = n_two = 0
    for k in np.flatnonzero(ok):
        p = points[k]
        xp, yp = p["pix"]
        if xp is None:
            no_window += 1
            continue
        hit = [w for w in dws if w["b"] and w["crs"] == p["proj_crs"] and tuple(w["res"]) == tuple(p["proj_res"])
               and w["b"][0] <= xp < w["b"][2] and w["b"][1] <= yp < w["b"][3]]
        if not hit:
            no_window += 1
            continue
        w = hit[0]
        n_two += len(hit) > 1
        lab_v = np.full((N_GROUPS, 4), np.nan)
        dep_v = np.full((N_GROUPS, 4), np.nan)
        r, c = p["rc"]
        for g in range(N_GROUPS):
            lp = os.path.join(p["dir"], "layers", _layer_dir(g), "_".join(TEN_M), "geotiff.tif")
            if os.path.exists(lp):
                with rasterio.open(lp) as src:
                    lab_v[g] = src.read(window=Window(c, r, 1, 1))[:, 0, 0]
            for tif in sorted(glob.glob(os.path.join(glob.escape(w["dir"]), "layers", _layer_dir(g), "*", "geotiff.tif"))):
                names = _bandset_names(os.path.basename(os.path.dirname(tif)), w["cfg"])
                if not names or not all(b in names for b in TEN_M):
                    continue
                with rasterio.open(tif) as src:
                    if src.count != len(names):
                        continue                                  # a multi-timestep raster: not read here
                    v = src.read(window=Window(xp - int(w["b"][0]), yp - int(w["b"][1]), 1, 1))[:, 0, 0]
                dep_v[g] = [v[names.index(b)] for b in TEN_M]
                break
        if np.isnan(lab_v).all():
            no_label_layer += 1
            continue
        if np.isnan(dep_v).all():
            no_deployed_layer += 1                         # a layout this reader does not read (e.g. packed timesteps)
            continue
        if np.isnan(lab_v).any() or np.isnan(dep_v).any():
            incomplete += 1                                # an item group missing on one side: not compared
            continue
        rows.append((k, lab_v, dep_v))
    out["n_points_in_two_deployed_windows"] = n_two
    out["n_points_without_deployed_window"] = no_window
    out["n_points_without_label_imagery"] = no_label_layer
    out["n_points_without_deployed_imagery_read"] = no_deployed_layer
    out["n_points_incomplete"] = incomplete
    if not rows:
        out["n_compared"] = 0
        return out, {}
    same_month = np.array([[np.array_equal(a[g], b[g]) for g in range(N_GROUPS)] for _, a, b in rows])
    same_all = same_month.all(1)
    diffs = np.concatenate([np.abs(a - b)[~same_month[i]] for i, (_, a, b) in enumerate(rows)]) \
        if (~same_month).any() else np.zeros((0, 4))
    # the item groups' order: rslearn's per-period mosaic can order periods either way, so the same scenes in another
    # order would read as different above; reversed and any-order matches say whether that is what happened
    reversed_all = np.array([all(np.array_equal(a[g], b[N_GROUPS - 1 - g]) for g in range(N_GROUPS)) for _, a, b in rows])
    any_order = np.array([sorted(map(tuple, a.tolist())) == sorted(map(tuple, b.tolist())) for _, a, b in rows])
    out.update({"n_compared": len(rows), "bands": list(TEN_M), "n_identical_all_groups": int(same_all.sum()),
                "n_identical_reversed_order": int(reversed_all.sum()),
                "n_identical_any_order": int(any_order.sum()),
                "share_identical_by_group": same_month.mean(0).tolist(),
                "median_abs_difference_where_not_identical": (np.nanmedian(diffs, 0).tolist() if len(diffs) else None)})
    ident = {points[k]["name"]: bool(s) for (k, _, _), s in zip(rows, same_all)}
    split = {}
    for flag in (True, False):
        ks = [k for (k, _, _), s in zip(rows, same_all) if bool(s) is flag and points[k]["split"] == "val"]
        if ks and err_by_point:
            e = np.array([err_by_point[k] for k in ks], dtype=np.float64)
            a = [agree_by_point.get(k) for k in ks]
            a = [x for x in a if x is not None]
            split["identical" if flag else "not_identical"] = {
                "n_val": len(ks), "error_rate": float(e.mean()),
                "agreement_with_replica": float(np.mean(a)) if a else None}
    if err_by_point:                       # outcomes are read only in the run, never in the inventory
        out["validation_points_by_input"] = split
    return out, ident


# ----------------------------------------------------------------------------- the grades
def grade(m, min_val=MIN_VAL, min_errors=MIN_ERRORS):
    """P1 to P5 from the measures. A prediction below its floor is reported with its value and not graded. P1's value
    is the larger end, in absolute value, of the 90% task-cluster bootstrap interval of the paired accuracy difference
    (an equivalence test); P5 holds when the mean percentile reaches its threshold and its one-sided 95% task-cluster
    bootstrap lower bound is above 0.5."""
    n_val, n_err = m.get("n_val"), m.get("n_errors")
    val_ok = n_val is not None and n_val >= min_val
    err_ok = val_ok and n_err is not None and n_err >= min_errors
    floors = {"P1": (val_ok, f"fewer than {min_val} validation points inside the deployed area"),
              "P2": (val_ok, f"fewer than {min_val} validation points inside the deployed area"),
              "P3": (err_ok, f"fewer than {min_val} validation points or {min_errors} errors among them"),
              "P4": (err_ok, f"fewer than {min_val} validation points or {min_errors} errors among them"),
              "P5": (val_ok, f"fewer than {min_val} validation points inside the deployed area")}
    rules = {"P1": f"the {P1_LEVEL:.0%} task-cluster bootstrap interval of deployed accuracy - replica accuracy on the "
                   f"same validation points lies within +-{THRESHOLDS['P1']:.3f} (value: its larger end in absolute "
                   "value)",
             "P2": "share of validation points where the deployed class equals the replica's >= 0.90",
             "P3": "AUROC of the deployed confidence for the deployed map's errors >= 0.75",
             "P4": "share of the errors among the least confident 10% of points >= 0.25",
             "P5": "mean share of the map's pixels less confident than a validation point >= 0.55, and its one-sided "
                   "95% task-cluster bootstrap lower bound > 0.5"}
    values = {"P1": m.get("p1_interval_max_abs"), "P2": m.get("p2_agreement"), "P3": m.get("p3_auroc"),
              "P4": m.get("p4_capture_10"), "P5": m.get("p5_mean_percentile")}
    out = {}
    for p in PREDICTIONS:
        v, thr = values[p], THRESHOLDS[p]
        ok, why = floors[p]
        row = {"rule": rules[p], "threshold": thr, "value": float(v) if finite(v) else None}
        if p == "P1":
            row["point_difference_reported"] = m.get("p1_difference")
        if p == "P5":
            lb = m.get("p5_lower_95")
            row["lower_one_sided_95"] = float(lb) if finite(lb) else None
        if not finite(v) or (p == "P5" and row["lower_one_sided_95"] is None):
            row.update(graded=False, reason="not computable on these inputs")
        elif not ok:
            row.update(graded=False, reason=why)
        elif p == "P1":
            row.update(graded=True, holds=bool(v <= thr + EPS))
        elif p == "P5":
            row.update(graded=True, holds=bool(v >= thr - EPS and row["lower_one_sided_95"] > P5_NULL))
        else:
            row.update(graded=True, holds=bool(v >= thr - EPS))
        out[p] = row
    out["floors"] = {"min_validation_points": min_val, "min_errors": min_errors, "n_validation_points": n_val,
                     "n_errors": n_err}
    return out


# ----------------------------------------------------------------------------- the analysis
def timesync_part(grids, timesync, boundaries, request_geometry, geometry_sha256, outcome, log=print):
    """Part I, wrapped so that it never stops the graded run: absent inputs give a note, a failure its traceback."""
    if not (timesync and request_geometry):
        return {"note": "Part I skipped: --timesync and --request-geometry were not given (exp/jobs/e98.sh fetches the "
                        "TimeSync sample and Natural Earth's boundaries and passes them)"}
    try:
        plots, areas = timesync_inside(timesync, request_geometry, boundaries, geometry_sha256)
        out = timesync_block(grids, plots, areas, outcome)
        if areas is None:
            out["note_weights"] = "no --boundaries: the countries' shares of the geometry, and so the stratified " \
                                  "estimate, are not computed"
        return out
    except Exception as ex:  # noqa: BLE001
        log(f"part I failed (report-only, the run goes on): {ex!r}")
        return {"error": repr(ex), "traceback": traceback.format_exc()}


def analyse(scores, labels, replica_path=None, deployment_dataset=None, inventory=False, n_boot=BOOT,
            min_val=MIN_VAL, min_errors=MIN_ERRORS, zone_draws=PLAN_ZONE_DRAWS, max_labels=plan.MAX_LABELS,
            map_sample=MAP_SAMPLE, log=print, timesync=None, boundaries=None, request_geometry=None,
            geometry_sha256=tsx.AWF_GEOMETRY_SHA256):
    t0 = time.time()
    entries, names, out_meta = find_outputs(scores)
    grids = [open_grid(e) for e in entries]
    burn_names, burn_note = burned_inputs(grids, out_meta)
    points, lab_summary = read_label_windows(labels)
    log(f"{len(points)} labelled windows kept ({lab_summary['by_split']}); {len(grids)} score grid(s)")
    cands = locate_all(points, grids)
    gi, rows, cols, how, off, P, Bk, C, overlap = read_points(grids, cands)
    cov = covered(P)
    split = np.array([p["split"] for p in points], dtype=object)
    mapping = {"rule": "the score pixel that contains the centre of the labelled pixel; reprojected where the CRSs "
                       "differ", "same_crs": int((how == "same_crs").sum()),
               "reprojected": int((how == "reprojected").sum()),
               "largest_offset_from_pixel_middle_px": float(np.nanmax(off)) if np.isfinite(off).any() else None,
               "overlapping_grids": dict(overlap, rule="a point in two grids takes the first whose pixel is covered"),
               "by_split": {}}
    for s in sorted(set(split.tolist())):
        m = split == s
        mapping["by_split"][s] = {"n": int(m.sum()), "outside_every_grid": int((m & (gi < 0)).sum()),
                                  "in_grid_uncovered": int((m & (gi >= 0) & ~cov).sum()),
                                  "covered": int((m & cov).sum())}
    grid_info = [{"label": G["label"], "crs": G["crs"].to_string(), "shape": list(G["shape"]),
                  "scores_file": os.path.basename(G["scores"]),
                  "condition_file": os.path.basename(G["condition"]) if G["condition"] else None,
                  "n_val_covered": int(((gi == g) & cov & (split == "val")).sum()),
                  "n_train_covered": int(((gi == g) & cov & (split == "train")).sum())} for g, G in enumerate(grids)]
    commit, dirty = e89.git_state()
    record = {"experiment": "exp98: the record's re-run of Ai2's FT-AWF deployment configuration (2023, 10 m, "
                            "probabilities written), graded at Ai2's own AWF labels",
              "prereg_page": os.path.relpath(PLAN, ROOT), "utc": e89.utc_now(), "commit": commit, "dirty": dirty,
              "inputs": {"grids": grid_info, "from_olmoearth": out_meta, "condition_names": names,
                         "request_period": "/".join(REQUEST_PERIOD)},
              "labels": lab_summary, "mapping": mapping}
    if inventory:
        record["mode"] = "inventory: counts only; no class, probability or condition at a label pixel is read out"
        if deployment_dataset:
            record["inputs_reported"], _ = compare_inputs(points, cov, {}, {}, labels, deployment_dataset)
        record["timesync_inventory"] = timesync_part(grids, timesync, boundaries, request_geometry, geometry_sha256,
                                                     outcome=False, log=log)
        record["seconds"] = round(time.time() - t0, 1)
        return record, None

    rd = readings(np.where(cov[:, None], P, 0.0)) if len(points) else None
    pred = np.where(cov, rd["pred"], -1)
    labels_arr = np.array([p["label"] for p in points])
    err = np.where(cov, (pred != labels_arr).astype(np.float64), np.nan)
    u, p1 = np.where(cov, rd["u"], np.nan), np.where(cov, rd["p1"], np.nan)
    bnd = boundary_at(Bk)
    names_arr = [p["name"] for p in points]
    tasks = np.array([p["task"] for p in points], dtype=object)
    V = np.flatnonzero(cov & (split == "val"))
    T = np.flatnonzero(cov & (split == "train"))
    dev = float(np.abs(np.nansum(P[cov], 1) - 1).max()) if cov.any() else 0.0
    if dev > SUM_TOLERANCE:
        raise ValueError(f"the probabilities at the label pixels sum to 1 +- {dev:.3g}, not within {SUM_TOLERANCE}: "
                         "they are not the softmax (prob_scales or a rescaled head); nothing is graded")
    rv = readings(P[V]) if V.size else None
    graded = {"n": int(V.size), "n_clusters": int(np.unique(tasks[V]).size) if V.size else 0,
              "readings": {k: v for k, v in (rv or {}).items() if k not in ("pred", "u", "p1")}}
    if V.size:
        graded["accuracy"] = e89.accuracy_block(labels_arr[V], pred[V], err[V], p1[V], tasks[V], N_OUT, CLASS_NAMES)
        acc = graded["accuracy"]["accuracy_window"]
        graded["ai2"] = {"accuracy": AI2_ACCURACY, "difference": acc - AI2_ACCURACY,
                         "within_2_points": bool(abs(acc - AI2_ACCURACY) <= 0.02 + EPS),
                         "matched": bool(V.size == 344),
                         "note": "Ai2's figure is over all 344 validation points, each read at a random position "
                                 "in a 16-px crop; it is matched only when all 344 are graded here"}
        graded["ranking"] = ranking_block(u[V], err[V], tasks[V], n_boot)
        b = bnd[V]
        e = err[V] > 0
        graded["boundary_reported"] = {"share_on_boundary_errors": float((b[e] > 0).mean()) if e.any() else None,
                                       "share_on_boundary_correct": float((b[~e] > 0).mean()) if (~e).any() else None,
                                       "exp21": "0.63 among errors, 0.34 among correct (the replica, 16-px crops)"}
        # the package's own path: assess on the points (patch 1, top-1 over the trained channels) against their labels.
        # Its prediction is the argmax over the trained channels, so a point predicted as channel 9 differs here.
        cond_V = C[V].astype(np.int64)[None, :]
        cnames = None
        if (cond_V >= 0).any():
            cnames = {int(c): (names or {}).get(int(c), str(int(c))) for c in np.unique(cond_V[cond_V >= 0])}
        a = assess.assess_prediction(log_trained(P[V]), is_logit=True, patch=1, budgets=BUDGETS, form="top1",
                                     reference=labels_arr[V][None, :],
                                     condition=cond_V if cnames else None, condition_names=cnames)
        ar = a["against_reference"]
        graded["assess_package_path"] = {
            "n_windows_scored": ar["n_windows_scored"], "error_rate": ar["error_rate"],
            "aurc_confidence": ar["aurc_confidence"],
            "error_capture_at_budget": {bkey(k): v["errors_captured_fraction"] for k, v in ar["error_capture_at_budget"].items()},
            "confusion_pairs": ar.get("confusion_pairs"),
            "same_order_as_ranking": bool(np.array_equal(
                np.argsort(-np.asarray(a["arrays"]["confidence"]).ravel(), kind="stable"), np.argsort(u[V], kind="stable"))),
            "n_p1_rounds_to_one": rv["n_p1_rounds_to_one"],
            "note": "assess predicts the argmax over the trained channels; exp98 (as exp89 and rslearn) over all ten. "
                    "Its confidence is the probability, which ties where it rounds to 1.0; the ranking here reads its "
                    "logarithm, so the two orders can differ only among those points"}
        graded["by_condition"] = condition_block(C[V], err[V], names)
    else:
        graded["note"] = "no validation point falls on a covered pixel of the deployed map"

    # part A against the replica
    replica = None
    if replica_path and os.path.exists(replica_path):
        replica = read_replica(replica_path)
        graded["replica"] = replica_block([names_arr[k] for k in V], labels_arr[V], pred[V], err[V], u[V], tasks[V],
                                          replica, n_boot)
        graded["replica"]["file"] = os.path.relpath(replica_path, ROOT) if replica_path.startswith(ROOT) else \
            os.path.basename(replica_path)
        graded["replica"]["sha256"] = e89.sha256(replica_path)
        exact = (how[V] == "same_crs") & (off[V] <= 0.01)
        if V.size and not exact.all():
            # the page: P1 and P2 are also reported on the same-CRS points that sit at the middle of their pixel;
            # the grade stays on every graded point
            Ve = V[exact]
            graded["replica_same_crs_reported"] = replica_block([names_arr[k] for k in Ve], labels_arr[Ve], pred[Ve],
                                                                err[Ve], u[Ve], tasks[Ve], replica, n_boot)
            graded["replica_same_crs_reported"]["n_left_out"] = int((~exact).sum())
    else:
        graded["replica"] = {"n_matched": 0, "note": "no replica file given"}

    train = {"n": int(T.size), "note": "in-sample: the model was fitted to these points; never graded, never pooled"}
    if T.size:
        train.update(accuracy=float(1 - err[T].mean()), n_errors=int(err[T].sum()),
                     exp89_s2_in_sample_accuracy=0.98046875)

    # part E and the map's own description
    rng = np.random.default_rng(SEED)
    maps, sample_all, n_map, cls_map, cond_map, n_sat_map, dev_map = [], [], 0, np.zeros(N_OUT, np.int64), {}, 0, 0.0
    earlier, unmatched = grid_overlaps(grids)
    q = min(1.0, map_sample / max(sum(G["shape"][0] * G["shape"][1] for G in grids), 1))
    burn_parts = []
    for G, ear in zip(grids, earlier):
        mp = map_pass(G, q, rng, ear, burn_patch=BURN_PATCH if burn_names else None)
        if "burn" in mp:
            burn_parts.append(mp["burn"])
            if "error" in mp["burn"] and burn_names is not None:   # Part H is report-only: left out with a note
                burn_names = None
                burn_note = (f"the burned layer of grid {G['label']} could not be read in the map pass "
                             f"({mp['burn']['error']}); Part H is left out, the run goes on")
        maps.append({"label": G["label"], "n_covered": mp["n_covered"], "sample_probability": q,
                     "n_sampled": int(mp["sample"].size), "largest_sum_deviation": mp["largest_sum_deviation"],
                     "n_saturated_float32": mp["n_saturated_float32"],
                     "n_counted_in_an_earlier_grid": mp["n_counted_in_an_earlier_grid"]})
        sample_all.append(mp["sample"])
        n_map += mp["n_covered"]
        cls_map += mp["class_counts"]
        n_sat_map += mp["n_saturated_float32"]
        dev_map = max(dev_map, mp["largest_sum_deviation"])
        for k, v in mp["condition_counts"].items():
            cond_map[k] = cond_map.get(k, 0) + v
    sample = np.sort(np.concatenate(sample_all)) if sample_all else np.zeros(0)
    rep = {"grids": maps, "n_map_pixels": int(n_map),
           "n_pixels_counted_once_of_overlaps": int(sum(x["n_counted_in_an_earlier_grid"] for x in maps)),
           "n_grid_overlaps_not_deduplicated": int(unmatched),
           "map_class_share": {CLASS_NAMES.get(c, f"channel {c}"): float(cls_map[c] / max(n_map, 1)) for c in range(N_OUT)}}
    if V.size and sample.size:
        pct = percentile_in_map(sample, u[V])
        rep.update({"points_mean_percentile": float(pct.mean()), "points_median_percentile": float(np.median(pct)),
                    "points_mean_percentile_bootstrap": p5_interval(pct, tasks[V], n_boot),
                    "reading": "the share of the map's pixels less confident than a validation point, averaged over "
                               "the points; 0.5 when the points' confidence is distributed as the map's",
                    "map_confidence_quantiles": {str(qq): float(np.exp(-np.quantile(sample, 1 - qq)))
                                                 for qq in assess.QUANTILES},
                    "points_confidence_quantiles": {str(qq): float(np.quantile(p1[V], qq)) for qq in assess.QUANTILES},
                    "points_class_share_deployed": {CLASS_NAMES.get(c, f"channel {c}"): float((pred[V] == c).mean())
                                                    for c in range(N_OUT)},
                    "points_class_share_label": {CLASS_NAMES[c]: float((labels_arr[V] == c).mean()) for c in TRAINED},
                    "reweighted_reported": reweighted(err[V], u[V], sample, n_map)})
        if cond_map:
            graded["by_condition"] = condition_block(C[V], err[V], names, cond_map)
    record.update({"graded_validation": graded, "train_in_sample": train, "where_the_points_sit": rep})

    # part F
    zone_err = {}
    if V.size:
        sel = metrics.selective_accuracy(u[V], 1 - err[V], coverages=tuple(c for c in PLAN_COVERAGES))
        zone_err = {c: float(1 - sel[c]) for c in PLAN_COVERAGES}
    record["plan_reported"] = plan_block(n_map, float(err[V].mean()) if V.size else None, zone_err, zone_draws,
                                         max_labels) if n_map else {"note": "no covered pixel"}

    # part G
    ident = {}
    if deployment_dataset:
        agree = {}
        if replica:
            for k in V:
                r = replica.get(names_arr[k])
                if r is not None:
                    agree[k] = float(r[1] == pred[k])
        record["inputs_reported"], ident = compare_inputs(points, cov, {k: err[k] for k in np.flatnonzero(cov)},
                                                          agree, labels, deployment_dataset)

    # part H (report-only): it never stops the run; a failure is recorded with its traceback and the grades go on
    burn_pts = np.full(len(gi), -2, np.int64)
    try:
        if burn_names is not None:          # Part H skipped: no burn code is read, every point keeps -2
            burn_pts = codes_at(grids, gi, rows, cols, "burned")
        record["burned_area_reported"] = burned_block(burn_names, burn_note, burn_parts, burn_pts, V, err, tasks,
                                                      out_meta)
    except Exception as ex:  # noqa: BLE001
        record["burned_area_reported"] = {"error": repr(ex), "traceback": traceback.format_exc()}
        log(f"part H failed (report-only, the run goes on): {ex!r}")

    # part I (report-only): the TimeSync plots inside the request geometry; never stops the run
    record["timesync_reported"] = timesync_part(grids, timesync, boundaries, request_geometry, geometry_sha256,
                                                outcome=True, log=log)

    # the grades
    rep_blk = graded.get("replica") or {}
    rk = graded.get("ranking") or {}
    matched = bool(rep_blk.get("n_matched"))
    m = {"n_val": int(V.size), "n_errors": int(np.nansum(err[V])) if V.size else 0,
         "p1_interval_max_abs": (rep_blk.get("p1_equivalence_interval") or {}).get("max_abs_end") if matched else None,
         "p1_difference": rep_blk.get("difference") if matched else None,
         "p2_agreement": rep_blk.get("agreement") if matched else None,
         "p3_auroc": rk.get("auroc"), "p4_capture_10": (rk.get("capture") or {}).get(bkey(0.10)),
         "p5_mean_percentile": rep.get("points_mean_percentile"),
         "p5_lower_95": (rep.get("points_mean_percentile_bootstrap") or {}).get("lower_one_sided_95")}
    if matched and rep_blk["n_matched"] < V.size:
        m["p1_interval_max_abs"] = m["p1_difference"] = m["p2_agreement"] = None   # P1, P2 need every point matched
        rep_blk["note"] = "some graded points are not in the replica file, so P1 and P2 are not computed"
    record["prereg"] = grade(m, min_val, min_errors)
    record["seconds"] = round(time.time() - t0, 1)
    units = {"names": np.array(names_arr), "split": split.astype(str), "label": labels_arr,
             "covered": cov, "pred_deployed": pred, "suspicion": u, "p1": p1, "condition": C, "burned": burn_pts,
             "boundary": bnd,
             "input_identical": np.array([ident.get(nm, -1) if nm in ident else -1 for nm in names_arr], dtype=np.int64)}
    return record, units


def write_units(units, path):
    np.savez_compressed(path, **units)


# ----------------------------------------------------------------------------- the synthetic smoke
# A 120 x 160 px map at 10 m, rows 0-19 uncovered. Forty validation points are covered (one of them reprojected from
# another CRS), two lie outside the grid and two on uncovered pixels; ten training points are covered; two windows are
# dropped by the label rule. The forty graded points are ranked by design: errors at the seven least confident and at
# the most confident one, so the AUROC is 7 * 32 / (8 * 32) = 0.875 and the 5%, 10% and 20% captures are 2/8, 4/8 and
# 7/8. One error is predicted as the untrained channel 9 with the label second. The replica agrees on 36 of 40 points:
# it is right where the deployed map is wrong at ranks 0 to 2 and wrong where it is right at rank 20.
# The map is read in two parts, as the real one is (two from-olmoearth directories, two deployment dataset roots):
# part 0 holds columns 0-99, part 1 columns 80-159. Columns 80-89 are covered in both, with equal values, and count
# once; columns 90-99 are NaN in part 0 and covered in part 1, so a point there must be read from part 1. Eleven
# labelled windows sit in both parts (eight graded validation points, two training points, one on the uncovered rows):
# five are covered in both, five only in part 1.
SMOKE_GRID = {"H": 120, "W": 160, "uncovered_rows": 20, "epsg": 32737, "other_epsg": 32637,
              "parts": ((0, 100), (80, 160)), "part0_nan_cols": (90, 100)}
SMOKE_EXPECTED = {"n_val": 40, "n_errors": 8, "accuracy": 32 / 40, "replica_accuracy": 34 / 40, "agreement": 36 / 40,
                  "auroc": 224 / 256, "capture": {"0.05": 2 / 8, "0.1": 4 / 8, "0.2": 7 / 8},
                  "n_pred_untrained": 1, "assess_errors": 7, "train_accuracy": 9 / 10, "reprojected": 1,
                  "mapping_val": {"n": 44, "outside_every_grid": 2, "in_grid_uncovered": 2, "covered": 40},
                  "dropped": 2, "mean_percentile": 367880 / 640000,
                  "by_condition": {0: (30, 3), 1: (8, 3), -1: (2, 2)},
                  "inputs": {"n_compared": 39, "identical": 37, "no_window": 1, "no_label_imagery": 10,
                             "in_two_windows": 10},
                  "overlap": {"in_two_or_more_grids": 11, "covered_in_two_or_more": 5, "covered_in_two_differing": 0,
                              "covered_only_in_a_later_grid": 5, "pixels_counted_once": 10 * 100},
                  "holds": {"P1": False, "P2": True, "P3": True, "P4": True, "P5": True},
                  # Part H: points on row 25 (ranks 4-13, errors 4-6) are unrecorded, rows 40 and 55 (ranks 14-29 and
                  # 35-38, no error) not burned, row 85 (ranks 0-3, 30-34 and 39; errors 0-3 and 39) burned. The
                  # windows of 4 px: window rows 5-6 unrecorded, 7-17 not burned, 18-29 burned, 40 window columns
                  # (23 in part 0, the last one half covered; 17 in part 1, whose first three touch part 0's pixels)
                  "burned": {"points": {-1: (10, 3), 0: (20, 0), 1: (10, 5)},
                             "windows": {-1: 2 * 40, 0: 11 * 40, 1: 12 * 40}, "dropped_overlap": 3 * 25,
                             "pixels": {-1: 8 * 160, 0: 44 * 160, 1: 48 * 160}}}
SMOKE_FLOORS = {"min_val": 20, "min_errors": 5}
SMOKE_BURN_ROWS = {-1: (0, 28), 0: (28, 72), 1: (72, 120)}     # aligned to the 4-px windows
SMOKE_BURN_NAMES = eb.code_names(2023, ["2023-09"])
# Part I: TimeSync plots at pixel centres of the smoke grid, on background pixels (class 2 in columns 0-79, class 4 in
# 80-159, all at 0.6, so the confidence ties), with "Kenya" north of row 65 and "Tanzania" south of it. One Kenya plot
# sits on the uncovered rows, one in columns 90-99 (NaN in part 0, read from part 1); one Kenya plot changed label
# between 2015 and 2017. Two plots lie 5 km outside the request geometry and one, of another country, inside it.
SMOKE_TS_PLOTS = (("kenya", 30, 10, "Wooded Grassland"), ("kenya", 30, 30, "Open Grassland"),
                  ("kenya", 30, 50, "Cropland"), ("kenya", 30, 95, "Open Grassland"), ("kenya", 30, 130, "Otherland"),
                  ("kenya", 10, 40, "Open Grassland"),
                  ("tanzania", 100, 20, "Open Forest"), ("tanzania", 100, 60, "Dense Forest"),
                  ("tanzania", 100, 100, "Wooded Grassland"), ("tanzania", 100, 120, "Open Grassland"),
                  ("tanzania", 100, 150, "Cropland"))
SMOKE_TS_OTHERS = (("kenya", 30, 600, "Cropland"), ("tanzania", 100, 600, "Open Grassland"),
                   ("ethiopia", 30, 70, "Cropland"))
SMOKE_TS_CHANGED = ("kenya", 30, 50, "Open Grassland")         # its 2015 label
SMOKE_TS_BORDER_ROW = 65
SMOKE_TS_EXPECTED = {"inside": {"kenya": 6, "tanzania": 5}, "covered": {"kenya": 5, "tanzania": 5},
                     "strict": {"kenya": 2, "tanzania": 3}, "lenient": {"kenya": 1, "tanzania": 2},
                     "changed_2015_2017": 1}


def make_smoke_timesync(root):
    """Part I's synthetic inputs under root: a TimeSync-style yearly CSV (2015-2017), a request geometry (a longitude
    and latitude box around the smoke grid) and Natural Earth-style boundaries splitting it at SMOKE_TS_BORDER_ROW."""
    import pyproj
    X0, Y0 = 500000.0, 9000000.0
    H, W = SMOKE_GRID["H"], SMOKE_GRID["W"]
    to_ll = pyproj.Transformer.from_crs(f"EPSG:{SMOKE_GRID['epsg']}", "EPSG:4326", always_xy=True)

    def ll(i, j):
        return to_ll.transform(X0 + 10 * j + 5, Y0 - 10 * i - 5)
    lon_c, lat_c = to_ll.transform([X0, X0 + 10 * W, X0, X0 + 10 * W], [Y0, Y0, Y0 - 10 * H, Y0 - 10 * H])
    box = [min(lon_c), min(lat_c), max(lon_c), max(lat_c)]
    _, lat_b = to_ll.transform(X0 + 5 * W, Y0 - 10 * SMOKE_TS_BORDER_ROW)
    paths = {"timesync": os.path.join(root, "timesync.csv"), "boundaries": os.path.join(root, "countries.geojson"),
             "request_geometry": os.path.join(root, "request.geojson")}
    rows = []
    for pid, (ctry, i, j, lab) in enumerate(SMOKE_TS_PLOTS + SMOKE_TS_OTHERS, start=1):
        lon, lat = ll(i, j)
        for y in (2015, 2016, 2017):
            lab_y = SMOKE_TS_CHANGED[3] if (ctry, i, j) == SMOKE_TS_CHANGED[:3] and y == 2015 else lab
            rows.append(f"{pid},{y},{ctry},0,{lat:.10f},{lon:.10f},{lab_y}")
    with open(paths["timesync"], "w") as f:
        f.write("plotid,year,country,country_id,latitude,longitude,landcover\n" + "\n".join(rows) + "\n")

    def poly(x0, y0, x1, y1):
        return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}
    with open(paths["request_geometry"], "w") as f:
        json.dump({"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": poly(*box), "properties": {
            "oe_start_time": "2023-01-01T00:00:00Z", "oe_end_time": "2023-12-31T00:00:00Z"}}]}, f)
    feats = [{"type": "Feature", "properties": {"ADM0_A3": "KEN"}, "geometry": poly(box[0] - 1, lat_b, box[2] + 1, lat_b + 1)},
             {"type": "Feature", "properties": {"ADM0_A3": "TZA"}, "geometry": poly(box[0] - 1, lat_b - 1, box[2] + 1, lat_b)},
             {"type": "Feature", "properties": {"ADM0_A3": "ETH"}, "geometry": poly(box[0] + 5, lat_b + 5, box[2] + 6, lat_b + 6)}]
    with open(paths["boundaries"], "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    return paths


def _smoke_point_vectors():
    """(label, deployed probability vector, replica logits) for the 40 graded points, by confidence rank 0..39."""
    rows = []
    for r in range(40):
        t = 0.35 + 0.0155 * r                     # the top-1 probability over the trained channels
        c = r % 9
        wrong = r < 7 or r == 39
        lab = (c + 1) % 9 if wrong else c
        p = np.full(N_OUT, (1 - t) / 8)
        p[9] = 0.0
        p[c] = t
        if r == 3:                                 # predicted as the untrained channel; its label is the trained top
            lab = c
            p = 0.5 * p
            p[9] = 0.5
        rep_pred = int(np.argmax(p))
        if r in (0, 1, 2):
            rep_pred = lab                         # the replica is right where the deployed map is wrong
        if r == 20:
            rep_pred = (lab + 2) % 9               # and wrong where it is right
        logits = np.zeros(N_OUT)
        logits[9] = -5.0
        logits[rep_pred] = 2.0
        rows.append((lab, p, logits))
    return rows


SMOKE_COND1 = (0, 1, 2, 30, 31, 32, 33, 34)     # ranks in condition 1 (rows 70-119): three errors of eight
SMOKE_UNRECORDED = (3, 39)                        # ranks on an unrecorded pixel: two errors of two


def _smoke_pixels():
    """The 40 graded points' pixels on the grid (row, col), by confidence rank: the ranks of SMOKE_COND1 and
    SMOKE_UNRECORDED on row 85 (condition 1; the last two pixels set to -1), the other thirty on rows 25, 40 and 55
    (condition 0), ten columns apart by 15 px."""
    slots0 = [(25 + 15 * (s // 10), 5 + 15 * (s % 10)) for s in range(30)]
    slots1 = [(85, 5 + 15 * s) for s in range(10)]
    ranks1 = list(SMOKE_COND1) + list(SMOKE_UNRECORDED)
    out, k0 = [], 0
    for r in range(40):
        if r in ranks1:
            out.append(slots1[ranks1.index(r)])
        else:
            out.append(slots0[k0])
            k0 += 1
    return out


def make_smoke_inputs(root):
    """Write the synthetic map, condition layer, label windows, replica and deployment dataset under root."""
    import rasterio
    from rasterio.transform import from_origin
    H, W, U = SMOKE_GRID["H"], SMOKE_GRID["W"], SMOKE_GRID["uncovered_rows"]
    crs = f"EPSG:{SMOKE_GRID['epsg']}"
    X0, Y0 = 500000.0, 9000000.0                   # synthetic; the summary never holds them
    tr = from_origin(X0, Y0, 10, 10)
    col0, row0 = int(X0 / 10), int(Y0 / -10)       # the grid's origin in projection pixel units
    scores = np.zeros((N_OUT, H, W), np.float32)
    bg = np.full(N_OUT, 0.05, np.float32)
    bg[9] = 0.0
    for c_bg, sl in ((2, slice(0, W // 2)), (4, slice(W // 2, W))):
        v = bg.copy()
        v[c_bg] = 0.6
        scores[:, :, sl] = v[:, None, None]
    scores[:, :U, :] = np.nan
    cond = np.zeros((H, W), np.int32)
    cond[70:, :] = 1
    cond[:U, :] = -1
    vecs = _smoke_point_vectors()
    pix = _smoke_pixels()
    for (lab, p, _), (i, j) in zip(vecs, pix):
        scores[:, i, j] = p.astype(np.float32)
    for r in SMOKE_UNRECORDED:
        cond[pix[r]] = -1
    burned = np.full((H, W), -1, np.int32)                  # Part H's layer, by rows (SMOKE_BURN_ROWS)
    for code, (a, b) in SMOKE_BURN_ROWS.items():
        burned[a:b] = code
    out_dirs = []
    for k, (c0, c1) in enumerate(SMOKE_GRID["parts"]):
        out_dir = os.path.join(root, f"oeix_part{k}")
        os.makedirs(out_dir, exist_ok=True)
        part = scores[:, :, c0:c1].copy()
        if k == 0:
            n0, n1 = SMOKE_GRID["part0_nan_cols"]
            part[:, :, n0 - c0:n1 - c0] = np.nan
        prof = dict(driver="GTiff", height=H, width=c1 - c0, crs=crs, transform=from_origin(X0 + 10 * c0, Y0, 10, 10))
        with rasterio.open(os.path.join(out_dir, f"scores_{SMOKE_GRID['epsg']}.tif"), "w", count=N_OUT,
                           dtype="float32", nodata=float("nan"), **prof) as dst:
            dst.write(part)
        with rasterio.open(os.path.join(out_dir, f"condition_{SMOKE_GRID['epsg']}.tif"), "w", count=1, dtype="int32",
                           nodata=-1, **prof) as dst:
            dst.write(cond[None, :, c0:c1])
        with open(os.path.join(out_dir, "olmoearth_conditions.json"), "w") as f:
            json.dump({"codes": {"0": "sentinel2:all:clear", "1": "sentinel2:most:some-cloud"}}, f)
        spath = os.path.join(out_dir, f"scores_{SMOKE_GRID['epsg']}.tif")
        eb.write_burned(os.path.join(out_dir, f"burned_{SMOKE_GRID['epsg']}.tif"), burned[:, c0:c1], spath)
        eb.dump(eb.names_record(SMOKE_BURN_NAMES, {"collection": eb.COLLECTION, "months_missing": ["2023-09"]},
                                [str(SMOKE_GRID["epsg"])]), os.path.join(out_dir, eb.NAMES_JSON))
        out_dirs.append(out_dir)

    # label windows: 63 x 63, the label at (31, 31) unless stated
    wroot = os.path.join(root, "awf", "dataset", "windows", GROUP)
    dep_roots = [os.path.join(root, "deploy", f"part{k}") for k in range(len(SMOKE_GRID["parts"]))]
    s2 = np.random.default_rng(SEED).integers(500, 3000, size=(N_GROUPS, 4, H, W)).astype(np.uint16)

    def window(name, split, i, j, cat, epsg=SMOKE_GRID["epsg"], extra=None, imagery=False, alter=None):
        x0, y0 = col0 + j - 31, row0 + i - 31
        if epsg == SMOKE_GRID["other_epsg"]:
            y0 += 1_000_000                          # 10,000 km of false northing, in 10 m pixels
        wdir = os.path.join(wroot, name)
        os.makedirs(os.path.join(wdir, "layers", "label", "category"), exist_ok=True)
        meta = {"group": GROUP, "name": name, "projection": {"crs": f"EPSG:{epsg}", "x_resolution": 10,
                                                             "y_resolution": -10},
                "bounds": [x0, y0, x0 + 63, y0 + 63],
                "time_range": ["2023-01-01T00:00:00+00:00", "2023-12-31T00:00:00+00:00"],
                "options": {"split": split, "source_task_id": name.split("_point_")[0]}}
        with open(os.path.join(wdir, "metadata.json"), "w") as f:
            json.dump(meta, f)
        lab = np.full((63, 63), FILL, np.int32)
        if cat is not None:
            lab[31, 31] = cat
        for (a, b, v) in (extra or []):
            lab[a, b] = v
        wtr = from_origin(x0 * 10.0, y0 * -10.0, 10, 10)
        with rasterio.open(os.path.join(wdir, "layers", "label", "category", "geotiff.tif"), "w", driver="GTiff",
                           height=63, width=63, count=1, dtype="int32", crs=f"EPSG:{epsg}", transform=wtr) as dst:
            dst.write(lab[None])
        if imagery and epsg == SMOKE_GRID["epsg"]:
            for g in range(N_GROUPS):
                ld = os.path.join(wdir, "layers", _layer_dir(g), "_".join(TEN_M))
                os.makedirs(ld, exist_ok=True)
                blk = np.zeros((4, 63, 63), np.uint16)
                blk[:, 31, 31] = s2[g, :, i, j]
                if alter is not None and g == alter:
                    blk[2, 31, 31] += 7                # B04 differs in one item group
                with rasterio.open(os.path.join(ld, "geotiff.tif"), "w", driver="GTiff", height=63, width=63, count=4,
                                   dtype="uint16", crs=f"EPSG:{epsg}", transform=wtr) as dst:
                    dst.write(blk)

    names = []
    for r, ((lab, _, _), (i, j)) in enumerate(zip(vecs, pix)):
        task = f"task_{r % 6:02d}"
        name = f"{task}_point_{r}"
        names.append((name, lab, task))
        window(name, "val", i, j, lab, epsg=SMOKE_GRID["other_epsg"] if r == 39 else SMOKE_GRID["epsg"],
               imagery=True, alter=5 if r in (10, 11) else None)
    others = [("task_06_point_0", "val", 30, 175, 1), ("task_06_point_1", "val", 40, -20, 2),      # outside
              ("task_07_point_0", "val", 5, 30, 3), ("task_07_point_1", "val", 10, 90, 4)]          # uncovered
    for name, sp, i, j, cat in others:
        window(name, sp, i, j, cat)
        names.append((name, cat, name.split("_point_")[0]))
    for m in range(10):                                                # training points, one wrong
        i, j = 61, 5 + 15 * m
        bgc = 2 if j < W // 2 else 4
        window(f"task_08_point_{m}", "train", i, j, bgc if m else (bgc + 1) % 9)
    window("task_09_point_0", "val", 50, 50, 1, extra=[(10, 10, 2)])   # two labelled pixels: dropped
    window("task_09_point_1", "val", 50, 60, None)                     # no labelled pixel: dropped
    # the label dataset's config and the deployment dataset (one window over the grid, the 12 bands in one band set)
    with open(os.path.join(root, "awf", "dataset", "config.json"), "w") as f:
        json.dump({"layers": {S2_LAYER: {"band_sets": [{"bands": list(TEN_M)}, {"bands": ["B05", "B06"]}],
                                         "data_source": {"init_args": {"harmonize": True}}}}}, f)
    bands12 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    for dep_root, (c0, c1) in zip(dep_roots, SMOKE_GRID["parts"]):        # one window per root, over its part
        os.makedirs(dep_root, exist_ok=True)
        with open(os.path.join(dep_root, "config.json"), "w") as f:
            json.dump({"layers": {S2_LAYER: {"band_sets": [{"bands": bands12}],
                                             "data_source": {"init_args": {"harmonize": True}}}}}, f)
        dw = os.path.join(dep_root, "windows", "default", "w0")
        os.makedirs(dw, exist_ok=True)
        with open(os.path.join(dw, "metadata.json"), "w") as f:
            json.dump({"projection": {"crs": f"EPSG:{SMOKE_GRID['epsg']}", "x_resolution": 10, "y_resolution": -10},
                       "bounds": [col0 + c0, row0, col0 + c1, row0 + H],
                       "time_range": ["2023-01-01T00:00:00+00:00", "2023-12-31T00:00:00+00:00"]}, f)
        for g in range(N_GROUPS):
            ld = os.path.join(dw, "layers", _layer_dir(g), "_".join(bands12))
            os.makedirs(ld, exist_ok=True)
            full = np.zeros((12, H, c1 - c0), np.uint16)
            for bi, b in enumerate(TEN_M):
                full[bands12.index(b)] = s2[g, bi][:, c0:c1]
            with rasterio.open(os.path.join(ld, "geotiff.tif"), "w", driver="GTiff", height=H, width=c1 - c0,
                               count=12, dtype="uint16", crs=f"EPSG:{SMOKE_GRID['epsg']}",
                               transform=from_origin(X0 + 10 * c0, Y0, 10, 10)) as dst:
                dst.write(full)
    # the replica, in exp89's units layout (no position)
    logits = {nm: v[2] for (nm, _, _), v in zip(names[:40], vecs)}
    rn, rl, rlog, rc = [], [], [], []
    for nm, lab, task in names:
        rn.append(nm)
        rl.append(lab)
        L = logits.get(nm)
        if L is None:
            L = np.zeros(N_OUT)
            L[lab] = 2.0
        rlog.append(L)
        rc.append(task)
    rpath = os.path.join(root, "exp89_units_awf.npz")
    np.savez(rpath, names=np.array(rn), label=np.array(rl), logits=np.array(rlog), clusters=np.array(rc))
    return {"scores": out_dirs, "labels": os.path.join(root, "awf", "dataset", "windows"), "replica": rpath,
            "deployment": dep_roots, "timesync": make_smoke_timesync(root)}


def smoke(out_dir=None, n_boot=SMOKE_BOOT, log=print):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_smoke_inputs(tmp)
        ts = dict(paths["timesync"], geometry_sha256=None)
        inv, _ = analyse(paths["scores"], paths["labels"], deployment_dataset=paths["deployment"], inventory=True,
                         log=log, **ts)
        rec, units = analyse(paths["scores"], paths["labels"], replica_path=paths["replica"],
                             deployment_dataset=paths["deployment"], n_boot=n_boot, zone_draws=SMOKE_ZONE_DRAWS,
                             max_labels=SMOKE_MAX_LABELS, log=log, **SMOKE_FLOORS, **ts)
    rec["mode"] = "smoke: synthetic inputs, floors lowered to " + json.dumps(SMOKE_FLOORS)
    rec["inventory"] = {k: inv[k] for k in ("labels", "mapping", "mode", "inputs_reported", "timesync_inventory")}
    out_dir = out_dir or OUT
    os.makedirs(out_dir, exist_ok=True)
    dump(rec, os.path.join(out_dir, "exp98_summary_smoke.json"))
    write_units(units, os.path.join(out_dir, "exp98_units_smoke.npz"))
    return rec


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true", help="synthetic inputs end to end")
    ap.add_argument("--inventory", action="store_true", help="counts only; allowed before the page is frozen")
    ap.add_argument("--scores", nargs="+", help="the from-olmoearth output directories, one per part read "
                    "(scores_<EPSG>.tif and beside it); exp/jobs/e98_read.sh writes them")
    ap.add_argument("--labels", help="the AWF windows root (holding spatial_split/) or spatial_split/ itself")
    ap.add_argument("--replica", default=REPLICA, help="exp89's per-point units for arm A (default %(default)s)")
    ap.add_argument("--deployment-dataset", nargs="+", default=None,
                    help="the rslearn dataset roots olmoearth_run wrote (one per partition), for part G")
    ap.add_argument("--timesync", default=None, help="Part I: the East Africa TimeSync sample's yearly CSV (Bullock et "
                    "al. 2021; exp/timesync_awf_crosswalk.py fetch checks it against its pinned sha256)")
    ap.add_argument("--boundaries", default=None, help="Part I: Natural Earth's admin-0 countries (GeoJSON), for each "
                    "country's share of the request geometry")
    ap.add_argument("--request-geometry", default=None, help="Part I: Ai2's AWF prediction_request_geometry.geojson "
                    "from the pinned olmoearth_projects clone (its sha256 is checked)")
    ap.add_argument("--out-dir", default=OUT)
    a = ap.parse_args(argv)
    if a.smoke:
        rec = smoke(a.out_dir)
        print(json.dumps(rec["prereg"], indent=1))
        return 0
    if not (a.scores and a.labels):
        ap.error("--scores and --labels are required outside --smoke")
    status = prereg_status()
    if not a.inventory and status != "frozen":
        print(f"refused: {os.path.relpath(PLAN, ROOT)} is {status}, not frozen; only --smoke and --inventory run "
              "before it is frozen")
        return 3
    rec, units = analyse(a.scores, a.labels, replica_path=a.replica, deployment_dataset=a.deployment_dataset,
                         inventory=a.inventory, timesync=a.timesync, boundaries=a.boundaries,
                         request_geometry=a.request_geometry)
    rec["prereg_status"] = status
    os.makedirs(a.out_dir, exist_ok=True)
    if a.inventory:
        dump(rec, os.path.join(a.out_dir, "exp98_inventory.json"))
        print(json.dumps(rec["mapping"], indent=1))
        return 0
    dump(rec, os.path.join(a.out_dir, "exp98_summary.json"))
    write_units(units, os.path.join(a.out_dir, "exp98_units.npz"))
    print(json.dumps(rec["prereg"], indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001  (a job log needs the traceback, and the exit code must fail the job)
        traceback.print_exc()
        sys.exit(1)
