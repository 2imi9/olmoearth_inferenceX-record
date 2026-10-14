#!/usr/bin/env python
"""exp92: the package on published map products with an independent probability sample (preregistered 4 October 2026).

Question. Every labelled-route result so far was graded on the record's own probe maps. Here the maps are published
products and the reference is an independent probability sample: do the package's three labelled routes and its
ranking hold there?

Reference. LCMAP Reference Data Product v1.2 (Pengra et al. 2020; USGS ScienceBase 5e42e54be4b0edb47be84535,
LCMAP_CU_20211117_V01_REF.zip, sha256 eb935d96...b4c, public domain): 25,000 plots, a simple random sample of the 30 m
pixels of the conterminous United States, each interpreted for every year 1984-2018 into LCMAP's eight primary
classes (column `LCMAP`). Year 2018. The plots' coordinates are pixel centres of LCMAP's Albers grid.

Maps (2018), read at the plots from Microsoft Planetary Computer (no login):
  A. LCMAP CONUS Collection 1.3, primary land cover `lcpri` (30 m; classes 1-8 as the reference), and its confidence
     `lcpconf` (1-100 a confidence that the label matches the training data; 151-213 provenance codes, kept apart).
  B. Esri 10 m Annual Land Use Land Cover v2 (`io-lulc-annual-v02`), the majority of the 3 x 3 pixels of 10 m around
     the plot centre (ties to the centre pixel), crosswalked to LCMAP's classes: water 1 -> Water, trees 2 -> Tree
     Cover, flooded vegetation 4 -> Wetland, crops 5 -> Cropland, built area 7 -> Developed, bare ground 8 -> Barren,
     snow/ice 9 -> Snow/Ice, rangeland 11 -> Grass/Shrub; clouds 10 and no data 0 leave the plot out of comparisons.
Reading: `lcpri` and Esri at all 25,000 plots; `lcpconf` at a random 5,000 of them (seed 92), drawn before any value is
read. Only the values at the plots are kept, with plot ids and no coordinates.

Population for the simulations: V, the plots where the reference and both maps have a class (and A_V, where the
reference and A have one). The plots stand in for the map: their truth is known, so every rate below is graded exactly.

Q1, which map is more accurate (exp90's design on a product pair). On V the full-truth difference is
Delta = acc_A - acc_B = (K_A - K_B)/N over the D plots where A and B differ. For n in 50, 100 and 200 labels, 2,000
draws each (seed 92), exp90's `cell`: n labels among the D differing plots ("disagree") or n from all of V analysed on
the differing plots they hold ("random"); coverage of Delta, how often the interval names the right map, the wrong
map, and its median width.
Q2, how wrong each map is. For A and B on V, 2,000 simple random samples of 300 plots (seed 92): coverage of the exact
hypergeometric interval for the map's error rate on V, and its median width.
Q3, does the product's own confidence rank its errors. On the `lcpconf` subset, plots with lcpconf 1-100 and a
reference and an A class: the AUROC of (101 - lcpconf) for A's errors with a 95% bootstrap interval (2,000 resamples,
seed 92), the share of A's errors in the 10% least confident plots (tie-aware) against 10% for a random order and the
attainable ceiling, and the excess AURC. Descriptive beside it: the provenance codes' shares and error rates, and the
same ranking statistics for A-B disagreement (1 where the maps differ) as a label-free comparator.

Predictions.
  P1. The disagree design's interval covers Delta on at least 94% of draws at every budget.
  P2. At 100 labels its median width is less than half the random design's.
  P3. The exact interval covers each map's error rate on V on at least 94% of draws.
  P4. LCMAP's confidence ranks its errors: AUROC at least 0.65 with the bootstrap interval's lower end above 0.5, and
      the 10% least confident plots hold at least 20% of the errors (twice a random order).
No prediction about which map is more accurate.

What this cannot show. The reference uses LCMAP's legend and was collected by LCMAP's programme, so B is graded in a
foreign legend through a crosswalk (rangeland against grass/shrub, flooded vegetation against wetland) and at a finer
resolution: Delta measures agreement with this reference, not which map is better in general. Whether any reference
plot was used to train LCMAP Collection 1.3 is not checked here. One year, one country. The plots as a finite
population make Q1 and Q2 exact gradings of the designs on real maps, not new evidence about CONUS beyond the sample.

Amendment, 5 October 2026, before any value was analysed. The first full read stalled on an HTTP request with no
timeout and was stopped at an hour with nothing written. A smoke run then showed that the year's date filter also
returned Esri's 2017 items (their ranges end on 1 January 2018), so some plots could take 2017's class. Items are now
kept only where their range starts in 2018 (both collections), requests time out after 30 s, and each finished tile is
appended to a resumable checkpoint. The design, samples and predictions are unchanged.

Correction, 5 October 2026, after the result. The pre-record audit found that where a plot lies in two overlapping
Esri tiles (1,453 plots, along UTM zone edges), the rule "smallest tile id wins" took an empty window from the tile
outside its zone, so 734 plots with Esri data were recorded as having none and left V. The plot now takes the first
tile, by id, whose 3 x 3 majority is a class; the values come from the same reads (the checkpoint), and the analysis
was rerun unchanged. The record states what the first analysis gave.

Usage.
    uv run --no-sync python exp/exp92_lcmap_products.py extract     # reads the maps at the plots; about ten minutes
    uv run --no-sync python exp/exp92_lcmap_products.py analyze     # writes exp/out/exp92_summary.json
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "exp"))
from oe_inferencex import estimate as est  # noqa: E402
from oe_inferencex import metrics  # noqa: E402

REF = os.path.join(ROOT, "data", "lcmap", "LCMAP_CU_20211117_V01_REF", "LCMAP_CU_20211117_V01_REF.csv")
PRJ = os.path.join(ROOT, "data", "lcmap", "LCMAP_CU_20211117_V01_REF", "LCMAP_CU_20200414_V01_REF.prj")
PLOTS = os.path.join(ROOT, "exp", "out", "exp92_plots.csv")
OUT = os.path.join(ROOT, "exp", "out", "exp92_summary.json")
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
YEAR, N_CONF, SEED, R, NS, N_RATE = 2018, 5000, 92, 2000, (50, 100, 200), 300
CLASSES = {"Developed": 1, "Cropland": 2, "Grass/Shrub": 3, "Tree Cover": 4, "Water": 5, "Wetland": 6, "Snow/Ice": 7, "Barren": 8}
ESRI = {1: 5, 2: 4, 4: 6, 5: 2, 7: 1, 8: 8, 9: 7, 11: 3}          # Esri class -> LCMAP class; 0 and 10 (clouds) -> none
GDAL = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MULTIRANGE="YES",
            VSI_CACHE="TRUE", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2", GDAL_HTTP_TIMEOUT="30",
            GDAL_HTTP_CONNECTTIMEOUT="10")
PARTIAL = os.path.join(ROOT, "exp", "out", "exp92_extract_partial.jsonl")       # one line per finished tile; resumable


def _json(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"} if body else {})
    return json.load(urllib.request.urlopen(req, timeout=60))


def token(collection):
    return _json(f"{STAC}/../sas/v1/token/{collection}".replace("/stac/v1/..", ""))["token"]


def items(collection):
    """Every STAC item of a collection for the year, following the next links."""
    out, body = [], {"collections": [collection], "datetime": f"{YEAR}-01-01/{YEAR}-12-31", "limit": 250}
    page = _json(f"{STAC}/search", body)
    while True:
        out += page["features"]
        nxt = [link for link in page.get("links", []) if link["rel"] == "next"]
        if not nxt:
            # the date filter also returns the previous year's items, whose ranges end on 1 January: keep the year's own
            return [f for f in out if (f["properties"].get("start_datetime") or f["properties"]["datetime"]).startswith(str(YEAR))]
        page = _json(nxt[0]["href"], nxt[0].get("body")) if nxt[0].get("method") == "POST" else _json(nxt[0]["href"])


def reference():
    rows = [r for r in csv.DictReader(open(REF)) if r["image_year"] == str(YEAR)]
    return [(int(r["plotid"]), float(r["x"]), float(r["y"]), CLASSES.get(r["LCMAP"], 0)) for r in rows]


def _inside(lon, lat, ring):
    """Point in polygon, even-odd rule."""
    inside, j = False, len(ring) - 1
    for i in range(len(ring)):
        (xi, yi), (xj, yj) = ring[i][:2], ring[j][:2]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _read(href_fn, plots, crs_src, size):
    """Values at plots from one COG: a size x size window centred on each plot (size 1 or 3)."""
    import rasterio
    from rasterio.warp import transform as wtransform
    from rasterio.windows import Window
    out = {}
    with rasterio.Env(**GDAL), rasterio.open(href_fn()) as ds:
        xs, ys = wtransform(crs_src, ds.crs, [p[1] for p in plots], [p[2] for p in plots])
        for p, x, y in zip(plots, xs, ys):
            r, c = ds.index(x, y)
            h = size // 2
            if not (h <= r < ds.height - h and h <= c < ds.width - h):
                continue
            out[p[0]] = ds.read(1, window=Window(c - h, r - h, size, size))
    return out


def extract():
    import rasterio
    from rasterio.warp import transform as wtransform
    crs_src = rasterio.crs.CRS.from_wkt(open(PRJ).read())
    ref = reference()
    rng = np.random.default_rng(SEED)
    conf_ids = set(int(i) for i in rng.choice([p[0] for p in ref], N_CONF, replace=False))
    lon, lat = wtransform(crs_src, "EPSG:4326", [p[1] for p in ref], [p[2] for p in ref])
    import threading
    lock = threading.Lock()
    tokens = {c: (token(c), time.time()) for c in ("usgs-lcmap-conus-v13", "io-lulc-annual-v02")}   # once, not per thread

    def signer(collection, href):
        def f():
            with lock:                              # refresh at most once per 30 minutes, one thread at a time
                if time.time() - tokens[collection][1] > 1800:
                    tokens[collection] = (token(collection), time.time())
                return href + "?" + tokens[collection][0]
        return f

    jobs = []
    lc = items("usgs-lcmap-conus-v13")
    for it in lc:                                   # LCMAP tiles: membership by the Albers extent of the tile
        tr, (h, w) = it["properties"]["proj:transform"], it["properties"]["proj:shape"]
        x0, y0 = tr[2], tr[5]
        mine = [p for p in ref if x0 <= p[1] < x0 + w * tr[0] and y0 + h * tr[4] < p[2] <= y0]
        if mine:
            jobs.append((it["id"], "lcpri", signer("usgs-lcmap-conus-v13", it["assets"]["lcpri"]["href"]), mine, 1))
            sub = [p for p in mine if p[0] in conf_ids]
            if sub:
                jobs.append((it["id"], "lcpconf", signer("usgs-lcmap-conus-v13", it["assets"]["lcpconf"]["href"]), sub, 1))
    es = items("io-lulc-annual-v02")
    LON, LAT = np.asarray(lon), np.asarray(lat)
    for it in es:                                   # Esri tiles: bounding box first, then the item's footprint
        x0, y0, x1, y1 = it["bbox"]
        near = np.flatnonzero((LON >= x0) & (LON <= x1) & (LAT >= y0) & (LAT <= y1))
        if not near.size:
            continue
        g = it["geometry"]
        polys = [g["coordinates"][0]] if g["type"] == "Polygon" else [pg[0] for pg in g["coordinates"]]
        mine = [ref[i] for i in near if any(_inside(LON[i], LAT[i], pg) for pg in polys)]
        if mine:
            jobs.append((it["id"], "esri", signer("io-lulc-annual-v02", it["assets"]["data"]["href"]), mine, 3))
    print(f"{len(ref)} plots; {len(lc)} LCMAP tiles, {len(es)} Esri tiles; {len(jobs)} reads; conf subset {len(conf_ids)}", flush=True)

    vals = {"lcpri": {}, "lcpconf": {}, "esri": {}}

    def keep(layer, item, pid, v):
        """Esri tiles overlap along UTM zone edges, where one tile's window can be empty: every window is kept, and the
        plot takes the first, by tile id, whose 3 x 3 majority is a class (see esri_majority)."""
        if layer == "esri":
            vals["esri"].setdefault(pid, {})[item] = v
        else:
            vals[layer][pid] = v

    done = set()
    if os.path.exists(PARTIAL):                      # resume: tiles already read are not read again
        for line in open(PARTIAL):
            rec = json.loads(line)
            done.add((rec["item"], rec["layer"]))
            for pid, v in rec["values"].items():
                keep(rec["layer"], rec["item"], int(pid), np.array(v, dtype=np.uint8) if rec["layer"] == "esri" else v)
    todo = [j for j in jobs if (j[0], j[1]) not in done]
    print(f"resuming: {len(done)} tiles done, {len(todo)} to read", flush=True)
    t0 = time.time()

    def run(job):
        item, layer, href, plots, size = job
        t = time.time()
        for attempt in range(3):
            try:
                return item, layer, _read(href, plots, crs_src, size), time.time() - t
            except Exception as e:                  # a transient HTTP failure: retry the tile
                err = e
                time.sleep(3 * (attempt + 1))
        print(f"failed {item} {layer} after 3 tries: {err}", flush=True)
        return item, layer, None, time.time() - t

    from concurrent.futures import as_completed
    with ThreadPoolExecutor(12) as pool, open(PARTIAL, "a") as part:
        futures = [pool.submit(run, j) for j in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            item, layer, got, secs = fut.result()
            if got is None:
                continue
            part.write(json.dumps({"item": item, "layer": layer, "values": {str(pid): (v.tolist() if layer == "esri" else int(v.ravel()[0]))
                                                                             for pid, v in got.items()}}) + "\n")
            part.flush()
            for pid, v in got.items():
                keep(layer, item, pid, v if layer == "esri" else int(v.ravel()[0]))
            if k % 25 == 0 or secs > 60:
                print(f"{k}/{len(todo)} tiles, {time.time() - t0:.0f}s (last {layer} {item}: {len(got)} plots in {secs:.0f}s)", flush=True)

    def esri_majority(wins):
        """The class of the first tile, by id, whose window's majority is a class; 0 where no tile has one."""
        for item in sorted(wins or {}):
            raw, mapped = _majority(wins[item])
            if raw != 0:
                return raw, mapped
        return 0, 0

    def _majority(win):
        flat = win.ravel()
        centre = int(flat[len(flat) // 2])
        counts = np.bincount(flat, minlength=12)
        top = np.flatnonzero(counts == counts.max())
        raw = centre if centre in top else int(top[0])
        return raw, ESRI.get(raw, 0)

    with open(PLOTS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plotid", "ref", "lcmap", "esri_raw", "esri", "in_conf_subset", "lcpconf"])
        for pid, _, _, ref_c in ref:
            raw, mapped = esri_majority(vals["esri"].get(pid))
            w.writerow([pid, ref_c, vals["lcpri"].get(pid, 0), raw, mapped, int(pid in conf_ids),
                        vals["lcpconf"].get(pid, "") if pid in conf_ids else ""])
    print(f"wrote {PLOTS} in {time.time() - t0:.0f}s: lcpri {len(vals['lcpri'])}, esri {len(vals['esri'])}, "
          f"lcpconf {len(vals['lcpconf'])}")


def _auroc(score, err):
    """Mann-Whitney AUROC of score for err (higher score = more suspicious), ties at average ranks."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    s = np.asarray(score)
    for v in np.unique(s):                                  # average ranks over ties
        m = s == v
        ranks[m] = ranks[m].mean()
    pos = err.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ranking(score, err, rng):
    """AUROC with a bootstrap interval, tie-aware capture at 10%, its ceiling, and excess AURC; score: higher = suspect."""
    a = _auroc(score, err)
    boot = []
    n = len(err)
    for _ in range(R):
        i = rng.integers(0, n, n)
        if 0 < err[i].sum() < n:
            boot.append(_auroc(score[i], err[i]))
    cap = float(metrics.capture_at_budget_expected(score, err, (0.10,))[0.10])
    return {"n": int(n), "errors": int(err.sum()), "error_rate": float(err.mean()), "auroc": a,
            "auroc_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "capture_10pct": cap, "ceiling_10pct": float(metrics.attainable_ceiling(0.10, float(err.mean()), n)),
            "excess_aurc": float(metrics.aurc_expected(score, err) - metrics.oracle_aurc(n, int(err.sum())))}


def analyze():
    import exp90_which_map as e90
    e90.R = R
    rows = list(csv.DictReader(open(PLOTS)))
    ref = np.array([int(r["ref"]) for r in rows]); a = np.array([int(r["lcmap"]) for r in rows])
    b = np.array([int(r["esri"]) for r in rows])
    V = (ref > 0) & (a > 0) & (b > 0)
    N = int(V.sum()); ra, rb = (a == ref) & V, (b == ref) & V
    diff = V & (a != b)
    D, KA, KB = int(diff.sum()), int((ra & diff).sum()), int((rb & diff).sum())
    delta = (KA - KB) / N
    rng = np.random.default_rng(SEED)
    q1 = {f"{d}_{n}": e90.cell(rng, delta, N, D, KA, KB, n, d) for n in NS for d in ("disagree", "random")}
    q2 = {}
    for name, right in (("lcmap", ra), ("esri", rb)):
        K = int(N - right.sum())
        ks = rng.hypergeometric(K, N - K, N_RATE, size=R)
        ivs = {k: est.hypergeom_interval(int(k), N_RATE, N) for k in np.unique(ks)}
        lo = np.array([ivs[k][0] for k in ks]); hi = np.array([ivs[k][1] for k in ks]); th = K / N
        q2[name] = {"error_rate": th, "coverage": float(((lo <= th) & (th <= hi)).mean()),
                    "median_width_points": float(np.median(hi - lo) * 100)}
    sub = np.array([r["in_conf_subset"] == "1" for r in rows])
    conf = np.array([int(r["lcpconf"]) if r["lcpconf"] not in ("", None) else 0 for r in rows])
    av = (ref > 0) & (a > 0)
    graded = sub & av & (conf >= 1) & (conf <= 100)
    err = (a != ref)[graded].astype(int)
    q3 = {"lcpconf": ranking((101 - conf[graded]).astype(float), err, rng)}
    both = graded & (b > 0)
    q3["disagreement_comparator"] = ranking((a != b)[both].astype(float), (a != ref)[both].astype(int), rng)
    codes = {}
    for lo_, hi_, name in ((151, 152, "transition 151-152"), (201, 201, "no stable model, NLCD 2001 class (201)"),
                           (202, 202, "insufficient data (202)"), (203, 255, "other provenance codes (203-255)")):
        m = sub & av & (conf >= lo_) & (conf <= hi_)
        codes[name] = {"n": int(m.sum()), "error_rate": float((a != ref)[m].mean()) if m.any() else None}
    q3["provenance_codes"] = codes
    q3["subset"] = {"drawn": int(sub.sum()), "with_reference_and_lcmap": int((sub & av).sum()), "graded_1_100": int(graded.sum())}
    out = {"experiment": "exp92", "preregistered": True, "year": YEAR,
           "plots": {"total": len(rows), "V": N, "reference_class": int((ref > 0).sum()), "lcmap_class": int((a > 0).sum()),
                     "esri_class": int((b > 0).sum())},
           "Q1": {"N": N, "D": D, "K_A": KA, "K_B": KB, "acc_lcmap": float(ra.sum() / N), "acc_esri": float(rb.sum() / N),
                  "delta": delta, "cells": q1},
           "Q2": q2, "Q3": q3}
    # found after the result: how many labels the comparison needs at this small a difference
    rng2 = np.random.default_rng(SEED + 1)
    out["descriptive_added_after_the_result"] = {
        "Q1_larger_budgets": {f"{d}_{n}": e90.cell(rng2, delta, N, D, KA, KB, n, d)
                              for n in (400, 800, 1600, 3200) for d in ("disagree", "random")},
        "esri_unmapped_plots": int((b == 0).sum()), "lcmap_unmapped_plots": int((a == 0).sum()),
        "esri_raw_counts_unmapped": {str(k): int(v) for k, v in zip(*np.unique(np.array([int(r["esri_raw"]) for r in rows])[b == 0], return_counts=True))}}
    out["prereg"] = {
        "P1": {"holds": all(q1[f"disagree_{n}"]["coverage"] >= 0.94 for n in NS)},
        "P2": {"holds": q1["disagree_100"]["median_width_points"] < 0.5 * q1["random_100"]["median_width_points"]},
        "P3": {"holds": all(q2[m]["coverage"] >= 0.94 for m in q2)},
        "P4": {"holds": q3["lcpconf"]["auroc"] >= 0.65 and q3["lcpconf"]["auroc_ci95"][0] > 0.5 and q3["lcpconf"]["capture_10pct"] >= 0.20}}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cmd", choices=("extract", "analyze"))
    a = p.parse_args(argv)
    extract() if a.cmd == "extract" else analyze()


if __name__ == "__main__":
    main()
