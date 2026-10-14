#!/usr/bin/env python
"""exp94: the package on published products across continents, years and themes (preregistered 7 October 2026).

    python exp/exp94_breadth.py extract-nlcd | extract-gfc | extract-eastafrica | extract-europe
    python exp/exp94_breadth.py analyze
    python exp/exp94_breadth.py smoke            # synthetic points, no download, no product value read

**Why.** exp92 and exp93 graded the labelled routes and the product's own confidence on one pair of products (LCMAP and
Esri), one year (2018) and one country (the United States), against LCMAP's own reference sample. This repeats the same
gradings on four more reference samples drawn as probability samples by their producers, each read against published
products, in three more continents, eight more years and one more theme. The reference files (about 29 MB in all) and
the product values at their points are read without an account; no coordinate is written to any output.

**Sources, products and populations.** In each source the reference points with a reference class and a value from
every product of a comparison stand in for the map population, as in exp92: every grading below is exact against
their truth. For the two stratified samples (NLCD, GFC2020) the unweighted points over-represent the rare strata, so
the population is the points, not the map; the design-weighted accuracies are reported beside them as descriptives.

1. NLCD accuracy-assessment points (USGS; Wickham et al.): set A, the 2011 edition's 8,000 stratified points with
   primary reference labels for 2001, 2006 and 2011 and the edition's maps in the file (LC2001v3, LC2006v2, LC2011v1);
   set B, the 2016 edition's 4,629 stratified points with labels for 2011 and 2016 and the maps LC2011, LC2016. Read
   at each point: LCMAP Collection 1.3 primary land cover and its confidence for the same year (Planetary Computer,
   one pixel). Correct means the map's class equals the primary reference label (`Lpri*`); the alternate label is not
   used. The comparison NLCD against LCMAP is in LCMAP's eight classes through USGS's own crosswalk (LCMAP Pathfinder
   NLCD comparison): 21-24 Developed; 81, 82 Cropland; 52, 71 Grass/Shrub; 41-43 Tree Cover; 11 Water; 90, 95 Wetland;
   12 Ice/Snow; 31 Barren. The NLCD maps are also graded in their own sixteen classes.
2. JRC GFC2020 validation set v2 (Colditz et al.; global forest / non-forest for 2020; 21,752 stratified units):
   units inside GAUL with a stratum and a label of forest or non-forest. Products: GFC2020 V2 (its value is in the
   file) and ESA WorldCover 2020 v100 (AWS, one 10 m pixel at the unit's centre; classes 10 tree cover and 95
   mangroves are forest, every other class non-forest). GFC2020 is built partly from WorldCover, so their errors are
   correlated; that is reported, not corrected.
3. East Africa TimeSync simple random sample (Bullock et al. 2021; 2,000 plots in each of Ethiopia, Kenya, Malawi,
   Rwanda, Tanzania, Uganda and Zambia), labels for 2015, 2016 and 2017. Products: Copernicus CGLS-LC100 v3 for the
   same year (Zenodo; discrete class and its Discrete-Classification-proba, 0-100, 255 missing; one 100 m pixel) and
   Esri 10 m Annual LULC v2 for 2017 (Planetary Computer; the 3 x 3 majority, as in exp92). Seven common classes:
   Forest (TimeSync Dense and Open Forest; CGLS 111-126; Esri trees 2), Grass/Shrub (Wooded and Open Grassland; CGLS
   20, 30, 100; Esri rangeland 11), Cropland (Cropland; CGLS 40; Esri 5), Built (Settlements; CGLS 50; Esri 7),
   Water (Open Water; CGLS 80, 200; Esri 1), Wetland (Vegetated Wetland; CGLS 90; Esri flooded vegetation 4), Other
   (Otherland; CGLS 60, 70; Esri bare 8, snow 9). Esri 0 and 10 (clouds) and CGLS 0 and 255 have no class.
4. S2GLC 2017 validation set (Jenerowicz et al.; 52,024 points in 55 Sentinel-2 tiles of 35 European countries;
   two-stage stratified with no weights published). Products: ODSE-LULC v0.2 2017 (EcoDataCube; the filtered dominant
   CORINE class at one 30 m pixel; and, at a random 5,000 of the points drawn with seed 94, the probability COG of
   that class, 0-100) and Esri 2017 (one 10 m pixel; the reference is a Sentinel-2 pixel). Eight common classes:
   Artificial (S2GLC 111; CORINE 111-142; Esri 7), Cropland (211, 221; CORINE 211-223, 241-244; Esri 5), Grass/Shrub
   (231, 322, 323; CORINE 231, 321-324; Esri 11), Tree (311, 312; CORINE 311-313; Esri 2), Bare (331; CORINE 331-334;
   Esri 8), Snow (335; CORINE 335; Esri 9), Wetland (411, 412; CORINE 411-423; Esri 4), Water (511; CORINE 511-523;
   Esri 1). ODSE values 49, 50 and 255 have no class. ODSE publishes no probability layer for CORINE 141 and 523
   (checked by request before any value was read); subset points of those classes are left out of Q3 and Q4 and
   counted.

**Gradings, as in exp92 and exp93.**
- Q1, which of two products agrees better with the reference: exp90's design on each pair (NLCD against LCMAP for
  set A 2001, 2006, 2011 and set B 2011, 2016; GFC2020 against WorldCover; CGLS against Esri 2017; ODSE against Esri
  2017): labels drawn among the points where the pair differs against labels drawn from all points, at 50, 100 and
  200 labels, 2,000 draws each.
- Q2, the error rate: the exact hypergeometric interval at 300 random points (100 for a population under 1,000),
  2,000 draws, for every product and year graded.
- Q3, the product's own confidence as an error ranking: LCMAP `lcpconf` (1-100 only) for its own errors in each NLCD
  set and year; CGLS proba (0-100) for its own errors in each East Africa year; ODSE's probability of its class on
  the 5,000-point subset. AUROC with a 2,000-resample bootstrap interval, tie-aware capture at 10% with its ceiling.
- Q4, the certified zone on that confidence: exp93's design on the package's grid (prefix, Bonferroni, plug-in;
  delta 0.1; alpha half the error rate; 300 labels; 2,000 draws), ties broken by a random permutation (seed 94).

**Predictions** (thresholds as in exp92 and exp93):
- P1: on every pair, the disagree design's 95% interval covers the true difference on at least 94% of draws at 50,
  100 and 200 labels.
- P2: on every pair, its median width at 100 labels is below that of labels drawn from all points.
- P3: every exact error-rate interval covers on at least 94% of draws.
- P4: every product confidence ranks the product's own errors better than chance: the bootstrap interval's lower end
  of the AUROC is above 0.5 in every cell (5 LCMAP cells, 3 CGLS cells, 1 ODSE cell). What makes it fail: a
  confidence that does not belong to the published class (CGLS's discrete map is post-processed after the classifier
  that gave the probability, unconfirmed), or crosswalk errors that the confidence cannot see.
- P5: in every Q4 cell the prefix and Bonferroni zones are wrong more than alpha on at most delta + 3 sqrt(delta (1 -
  delta) / 2000) = 0.120 of draws.
P1, P3 and P5 follow from the procedures' guarantees and test the implementation on real error patterns; P2 and P4
are the empirical predictions. Nothing is predicted about which product is better.

**Descriptive, not predicted.** Accuracies (unweighted and, for NLCD and GFC2020, design-weighted), how often each
design names a product, certified coverage, the share of points with no product value or a confidence code, and
the NLCD maps' accuracy in their own sixteen classes.

**What would invalidate a grading.** A crosswalk changed after the values were read; a point written with its
coordinates; a confidence code read as a confidence; a product year other than the label's.

**After the preregistration (7 October 2026).** Changes to reading only, made before any value was analysed: the GDAL
option GDAL_CACHEMAX was removed (rasterio refused it, so the first run read nothing); CGLS is read from Zenodo by
rows of blocks, one job per country (the same pixels, checked against point reads; Zenodo answered about one request
a second and then HTTP 429); the Europe point lists are built from arrays. Corrections from the pre-record audit,
none of which changes a graded number: the years are seven (2001, 2006, 2011, 2015, 2016, 2017, 2020), not eight
more; the NLCD to LCMAP crosswalk is Table 1-2 of LCMAP's Collection 1.3 Science
Product Guide; the S2GLC validation set (Jenerowicz et al., PANGAEA 934197) is described in Malinowski et al. (2020),
and it leaves out classes under 0.95% of a tile,
so it is not a probability sample of Europe; ODSE's value where it has no class is 0, its sea mask (2,107 of the
2,127 such points are S2GLC water), and none of 49, 50 or 255 occurs (255 is unclassified water in its legend); CGLS
gives no probability (255) at every water and built-up point and at a tenth of the cropland points, so its
confidence cells leave those out; pooled over countries, the East Africa plots are stratified by country and stand
for the plots, not the region; LCMAP Collection 1.3 was trained on the 2011 edition's NLCD 2001 (LC2001v3 here)
through the same crosswalk, so the two products' errors are linked; CGLS's and ODSE's published classes are
post-processed after the classifier that gave the probability. The design-weighted GFC2020 accuracies, listed above
and first missing, were added after the audit, with the other post-audit descriptives under `found_after_result`;
none of them draws a random number, and every graded number is byte-identical.
"""
import argparse
import csv
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "exp"))
from oe_inferencex import estimate as est  # noqa: E402

DATA = os.path.join(ROOT, "data", "breadth")
OUTDIR = os.path.join(ROOT, "exp", "out")
SUMMARY = os.path.join(OUTDIR, "exp94_summary.json")
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SEED, R, NS, N_RATE, N_SUB, B_ZONE = 94, 2000, (50, 100, 200), 300, 5000, 300
GDAL = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MULTIRANGE="YES",
            VSI_CACHE="TRUE", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2", GDAL_HTTP_TIMEOUT="30",
            GDAL_HTTP_CONNECTTIMEOUT="10")

# ----------------------------------------------------------------------------- crosswalks, fixed before any value
LCMAP_NAMES = {1: "Developed", 2: "Cropland", 3: "Grass and shrub", 4: "Tree Cover", 5: "Water", 6: "Wetland", 7: "Ice and snow",
               8: "Barren"}
NLCD_TO_LCMAP = {21: 1, 22: 1, 23: 1, 24: 1, 81: 2, 82: 2, 52: 3, 71: 3, 41: 4, 42: 4, 43: 4, 11: 5, 90: 6, 95: 6, 12: 7,
                 31: 8}
EA_NAMES = {1: "Forest", 2: "Grass and shrub", 3: "Cropland", 4: "Built", 5: "Water", 6: "Wetland", 7: "Other"}
TIMESYNC = {"Dense Forest": 1, "Open Forest": 1, "Wooded Grassland": 2, "Open Grassland": 2, "Cropland": 3,
            "Settlements": 4, "Open Water": 5, "Vegetated Wetland": 6, "Otherland": 7}
CGLS = {**{c: 1 for c in (111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126)}, 20: 2, 30: 2, 100: 2, 40: 3,
        50: 4, 80: 5, 200: 5, 90: 6, 60: 7, 70: 7}
ESRI_EA = {2: 1, 11: 2, 5: 3, 7: 4, 1: 5, 4: 6, 8: 7, 9: 7}
EU_NAMES = {1: "Artificial", 2: "Cropland", 3: "Grass and shrub", 4: "Tree", 5: "Bare", 6: "Snow", 7: "Wetland", 8: "Water"}
S2GLC = {111: 1, 211: 2, 221: 2, 231: 3, 322: 3, 323: 3, 311: 4, 312: 4, 331: 5, 335: 6, 411: 7, 412: 7, 511: 8}
CORINE = {**{c: 1 for c in (111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142)},
          **{c: 2 for c in (211, 212, 213, 221, 222, 223, 241, 242, 243, 244)},
          **{c: 3 for c in (231, 321, 322, 323, 324)}, **{c: 4 for c in (311, 312, 313)},
          **{c: 5 for c in (331, 332, 333, 334)}, 335: 6, **{c: 7 for c in (411, 412, 421, 422, 423)},
          **{c: 8 for c in (511, 512, 521, 522, 523)}}
# ODSE's dominant-class raster holds an index 1-48; the index's CORINE code, from the product's own legend (.qml)
ODSE_INDEX = dict(zip(list(range(1, 44)) + [48],
                      [111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142, 211, 212, 213, 221, 222, 223, 231, 241, 242,
                       243, 244, 311, 312, 313, 321, 322, 323, 324, 331, 332, 333, 334, 335, 411, 412, 421, 422, 423, 511,
                       512, 521, 522, 523]))
ESRI_EU = {7: 1, 5: 2, 11: 3, 2: 4, 8: 5, 9: 6, 4: 7, 1: 8}
GFC_WORLDCOVER_FOREST = (10, 95)
CGLS_RECORDS = {2015: ("3939038", "2015-base"), 2016: ("3518026", "2016-conso"), 2017: ("3518036", "2017-conso")}
ODSE = "https://s3.ecodatacube.eu/arco/lcv_landcover.{}_lucas.corine.eml_{}_30m_0..0cm_20170101_20171231_eumap_epsg3035_v0.2.tif"


# ----------------------------------------------------------------------------- reading
def _json(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json", "User-Agent": "oe-inferencex-exp94"}
                                 if body else {"User-Agent": "oe-inferencex-exp94"})
    return json.load(urllib.request.urlopen(req, timeout=60))


class Signer:
    """Planetary Computer SAS tokens, fetched once per collection and refreshed every 30 minutes under a lock."""
    def __init__(self):
        self.lock, self.tokens = threading.Lock(), {}

    def __call__(self, collection, href):
        def f():
            with self.lock:
                if collection not in self.tokens or time.time() - self.tokens[collection][1] > 1800:
                    t = _json(f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}")["token"]
                    self.tokens[collection] = (t, time.time())
                return href + "?" + self.tokens[collection][0]
        return f


def pc_items(collection, year):
    """The collection's STAC items whose range starts in the year (the date filter also returns the year before)."""
    out, body = [], {"collections": [collection], "datetime": f"{year}-01-01/{year}-12-31", "limit": 250}
    page = _json(f"{STAC}/search", body)
    while True:
        out += page["features"]
        nxt = [link for link in page.get("links", []) if link["rel"] == "next"]
        if not nxt:
            return [f for f in out if (f["properties"].get("start_datetime") or f["properties"]["datetime"]).startswith(str(year))]
        page = _json(nxt[0]["href"], nxt[0].get("body")) if nxt[0].get("method") == "POST" else _json(nxt[0]["href"])


def read_points(href, points, crs_src, size=1, strips=False):
    """Values at points from one raster: a size x size window centred on each (id, x, y) point in crs_src. With strips
    (size 1 only), the points of one row of blocks are read in a single window spanning them: the same pixels, but one
    request per row of blocks instead of one per point (Zenodo answers about one request a second and then HTTP 429)."""
    import rasterio
    from rasterio.warp import transform as wtransform
    from rasterio.windows import Window
    out = {}
    with rasterio.Env(**GDAL), rasterio.open(href() if callable(href) else href) as ds:
        xs, ys = wtransform(crs_src, ds.crs, [p[1] for p in points], [p[2] for p in points])
        rc = [ds.index(x, y) for x, y in zip(xs, ys)]
        h = size // 2
        keep = [(p, int(r), int(c)) for p, (r, c) in zip(points, rc) if h <= r < ds.height - h and h <= c < ds.width - h]
        if not strips:
            for p, r, c in keep:
                out[p[0]] = ds.read(1, window=Window(c - h, r - h, size, size)).ravel().tolist()
            return out
        assert size == 1
        bh = ds.block_shapes[0][0]
        rows = {}
        for p, r, c in keep:
            rows.setdefault(r // bh, []).append((p, r, c))
        for grp in rows.values():
            r0, r1 = min(g[1] for g in grp), max(g[1] for g in grp)
            c0, c1 = min(g[2] for g in grp), max(g[2] for g in grp)
            a = ds.read(1, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1))
            for p, r, c in grp:
                out[p[0]] = [int(a[r - r0, c - c0])]
    return out


def run_jobs(jobs, partial, threads=12):
    """jobs: (key, layer, href, points, crs, size). Resumable: each finished job is a line of `partial`."""
    vals, done = {}, set()
    if os.path.exists(partial):
        for line in open(partial):
            rec = json.loads(line)
            done.add((rec["key"], rec["layer"]))
            for pid, v in rec["values"].items():
                vals.setdefault(rec["layer"], {}).setdefault(pid, {})[rec["key"]] = v
    todo = [j for j in jobs if (j[0], j[1]) not in done]
    print(f"{len(jobs)} reads, {len(done)} already done, {len(todo)} to do", flush=True)
    t0 = time.time()

    def go(job):
        key, layer, href, pts, crs, size = job[:6]
        err = None
        for attempt in range(3):
            try:
                return key, layer, read_points(href, pts, crs, size, strips=len(job) > 6 and job[6])
            except Exception as e:                          # a transient HTTP failure: retry the job
                err = e
                time.sleep(3 * (attempt + 1))
        print(f"failed {key} {layer}: {err}", flush=True)
        return key, layer, None

    with ThreadPoolExecutor(threads) as pool, open(partial, "a") as part:
        futures = [pool.submit(go, j) for j in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            key, layer, got = fut.result()
            if got is None:
                continue
            part.write(json.dumps({"key": key, "layer": layer, "values": {str(p): v for p, v in got.items()}}) + "\n")
            part.flush()
            for pid, v in got.items():
                vals.setdefault(layer, {}).setdefault(str(pid), {})[key] = v
            if k % 50 == 0:
                print(f"{k}/{len(todo)} reads, {time.time() - t0:.0f}s", flush=True)
    return vals


def first_valid(by_key, valid):
    """The value of the first read, by key, that holds a valid value (tiles overlap at their edges)."""
    for key in sorted(by_key or {}):
        v = by_key[key]
        if v and valid(v):
            return v
    return None


def majority(v):
    """exp92's rule: the raw value most frequent in the window, no-data and clouds included, the centre on a tie."""
    flat = np.asarray(v, dtype=np.int64)
    counts = np.bincount(flat, minlength=12)
    top = np.flatnonzero(counts == counts.max())
    centre = int(flat[len(flat) // 2])
    return centre if centre in top else int(top[0])


def write_points(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path}: {len(rows)} rows", flush=True)


# ----------------------------------------------------------------------------- extraction, one source at a time
def extract_nlcd():
    import geopandas as gpd
    signer = Signer()
    sets = {"A": (os.path.join(DATA, "nlcd2011_AA", "NLCD2011AA.shp"), {2001: "LC2001v3", 2006: "LC2006v2", 2011: "LC2011v1"},
                  {2001: "Lpri01", 2006: "Lpri06", 2011: "Lpri11"}),
            "B": (os.path.join(DATA, "nlcd2016_AA", "NLCD2016_Accuracy_ReferenceData_CONUS.shp"), {2011: "LC2011", 2016: "LC2016"},
                  {2011: "Lpri11L2", 2016: "Lpri16L2"})}
    alternate = {"Lpri01": "Lalt01", "Lpri06": "Lalt06", "Lpri11": "Lalt11", "Lpri11L2": "Lalt11L2", "Lpri16L2": "Lalt16L2"}
    from pyproj import CRS, Transformer
    frames, jobs = {}, []
    tiles = {year: pc_items("usgs-lcmap-conus-v13", year) for year in (2001, 2006, 2011, 2016)}
    crs_lcmap = CRS.from_wkt(tiles[2011][0]["properties"]["proj:wkt2"])
    for s, (path, maps, refs) in sets.items():
        g = gpd.read_file(path)
        g["pid"] = [f"{s}{i}" for i in range(len(g))]
        frames[s] = g
        xs, ys = Transformer.from_crs(g.crs, crs_lcmap, always_xy=True).transform(g.geometry.x.values, g.geometry.y.values)
        for year in maps:
            for it in tiles[year]:                  # LCMAP tiles: membership by the Albers extent of the tile, as in exp92
                tr, (h, w) = it["properties"]["proj:transform"], it["properties"]["proj:shape"]
                x0, y0 = tr[2], tr[5]
                inside = np.flatnonzero((xs >= x0) & (xs < x0 + w * tr[0]) & (ys <= y0) & (ys > y0 + h * tr[4]))
                pts = [(g["pid"].iat[i], float(xs[i]), float(ys[i])) for i in inside]
                if pts:
                    for layer in ("lcpri", "lcpconf"):
                        jobs.append((f"{year}:{it['id']}", f"{s}{year}:{layer}",
                                     signer("usgs-lcmap-conus-v13", it["assets"][layer]["href"]), pts, crs_lcmap.to_wkt(), 1))
    vals = run_jobs(jobs, os.path.join(OUTDIR, "exp94_nlcd_partial.jsonl"))
    rows = []
    for s, (path, maps, refs) in sets.items():
        g = frames[s]
        for year in maps:
            pri = vals.get(f"{s}{year}:lcpri", {})
            conf = vals.get(f"{s}{year}:lcpconf", {})
            for i in range(len(g)):
                pid = g["pid"].iat[i]
                p = first_valid(pri.get(pid), lambda v: 1 <= v[0] <= 8)
                c = first_valid(conf.get(pid), lambda v: v[0] > 0)
                rows.append([pid, s, year, int(g["Strata"].iat[i]), float(g["Weight"].iat[i]), int(g[refs[year]].iat[i]),
                             int(g[maps[year]].iat[i]), 0 if p is None else int(p[0]), "" if c is None else int(c[0]),
                             int(g[alternate[refs[year]]].iat[i])])
    write_points(os.path.join(OUTDIR, "exp94_nlcd_points.csv"),
                 ["pid", "set", "year", "stratum", "weight", "ref_nlcd", "nlcd", "lcmap", "lcpconf", "ref_nlcd_alternate"], rows)


def extract_gfc():
    import pandas as pd
    g = pd.read_csv(os.path.join(DATA, "gfc2020", "combined_scenario_all_GFCV2.csv"))
    g = g[(g["gaul"] == 1) & (g["strata"] != 0) & g["forest_class_num"].isin([0, 1])].reset_index(drop=True)
    tiles = {}
    for i in range(len(g)):
        lon, lat = float(g["pixel_center_x"].iat[i]), float(g["pixel_center_y"].iat[i])
        la, lo = int(np.floor(lat / 3) * 3), int(np.floor(lon / 3) * 3)
        name = f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
        tiles.setdefault(name, []).append((str(int(g["sample_id"].iat[i])), lon, lat))
    url = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/ESA_WorldCover_10m_2020_v100_{}_Map.tif"
    jobs = [(t, "worldcover", "/vsicurl/" + url.format(t), pts, "EPSG:4326", 1) for t, pts in sorted(tiles.items())]
    vals = run_jobs(jobs, os.path.join(OUTDIR, "exp94_gfc_partial.jsonl"))["worldcover"]
    rows = []
    for i in range(len(g)):
        sid = str(int(g["sample_id"].iat[i]))
        v = first_valid(vals.get(sid), lambda v: v[0] > 0)
        wc = 0 if v is None else int(v[0])
        rows.append([sid, int(g["strata"].iat[i]), int(g["forest_class_num"].iat[i]), int(g["GFC_v2"].iat[i]), wc,
                     "" if v is None else int(wc in GFC_WORLDCOVER_FOREST)])
    write_points(os.path.join(OUTDIR, "exp94_gfc_points.csv"),
                 ["sample_id", "stratum", "ref_forest", "gfc_v2", "worldcover_raw", "worldcover_forest"], rows)


def extract_eastafrica():
    import pandas as pd
    e = pd.read_csv(os.path.join(DATA, "eastafrica_yearly_point_data.csv"))
    e = e[e["year"].isin([2015, 2016, 2017])]
    e["pid"] = e["country"] + ":" + e["plotid"].astype(str)
    plots = e.drop_duplicates("pid")[["pid", "longitude", "latitude"]]
    pts = [(r.pid, float(r.longitude), float(r.latitude)) for r in plots.itertuples()]
    jobs = []
    # CGLS from Zenodo: one job per year, layer and country, its points read by rows of blocks (strips)
    country = dict(zip(plots["pid"], e.drop_duplicates("pid")["country"]))
    for year, (rec, tag) in CGLS_RECORDS.items():
        for layer in ("map", "proba"):
            name = "Discrete-Classification-" + layer
            href = f"/vsicurl/https://zenodo.org/records/{rec}/files/PROBAV_LC100_global_v3.0.1_{tag}_{name}_EPSG-4326.tif"
            for ctry in sorted(set(country.values())):
                mine = [p for p in pts if country[p[0]] == ctry]
                jobs.append((f"{year}:{ctry}", f"cgls{year}:{layer}", href, mine, "EPSG:4326", 1, True))
    signer = Signer()
    jobs += _esri_jobs(pts, 2017, signer, size=3)
    vals = run_jobs(jobs, os.path.join(OUTDIR, "exp94_eastafrica_partial.jsonl"), threads=2)
    rows = []
    for r in e.itertuples():
        y = int(r.year)
        m = first_valid(vals.get(f"cgls{y}:map", {}).get(r.pid), lambda v: v[0] not in (0, 255))
        pb = first_valid(vals.get(f"cgls{y}:proba", {}).get(r.pid), lambda v: True)
        es = None
        if y == 2017:
            wins = vals.get("esri2017", {}).get(r.pid)
            es = first_valid(wins, lambda v: majority(v) != 0)         # the first tile, by id, with data
        rows.append([r.pid, str(r.country), y, r.landcover, TIMESYNC.get(r.landcover, 0),
                     0 if m is None else int(m[0]), 0 if m is None else CGLS.get(int(m[0]), 0),
                     "" if pb is None else int(pb[0]), "" if es is None else ESRI_EA.get(majority(es), 0)])
    write_points(os.path.join(OUTDIR, "exp94_eastafrica_points.csv"),
                 ["pid", "country", "year", "ref_label", "ref", "cgls_raw", "cgls", "cgls_proba", "esri"], rows)


def _esri_jobs(pts, year, signer, size):
    """Esri items cover UTM tiles: a point goes to every item whose footprint holds it (bbox first, then the polygon)."""
    lon = np.array([p[1] for p in pts]); lat = np.array([p[2] for p in pts])
    jobs = []
    for it in pc_items("io-lulc-annual-v02", year):
        x0, y0, x1, y1 = it["bbox"]
        near = np.flatnonzero((lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1))
        if not near.size:
            continue
        geom = it["geometry"]
        polys = [geom["coordinates"][0]] if geom["type"] == "Polygon" else [pg[0] for pg in geom["coordinates"]]
        mine = [pts[i] for i in near if any(_inside(lon[i], lat[i], pg) for pg in polys)]
        if mine:
            jobs.append((it["id"], f"esri{year}", signer("io-lulc-annual-v02", it["assets"]["data"]["href"]), mine,
                         "EPSG:4326", size))
    return jobs


def _inside(lon, lat, ring):
    inside, j = False, len(ring) - 1
    for i in range(len(ring)):
        (xi, yi), (xj, yj) = ring[i][:2], ring[j][:2]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def extract_europe():
    import geopandas as gpd
    g = gpd.read_file(os.path.join(DATA, "S2GLC_validation_data", "S2GLC_validation_data_LAEA.shp"))
    g["pid"] = [str(i) for i in range(len(g))]
    x, y = g.geometry.x.to_numpy(), g.geometry.y.to_numpy()
    pts3035 = [(pid, float(a), float(b)) for pid, a, b in zip(g["pid"], x, y)]
    ll = g.to_crs(4326)
    lon, lat = ll.geometry.x.to_numpy(), ll.geometry.y.to_numpy()
    pts4326 = [(pid, float(a), float(b)) for pid, a, b in zip(g["pid"], lon, lat)]
    sub = set(np.random.default_rng(SEED).choice(len(g), N_SUB, replace=False).astype(str))
    hcl = "/vsicurl/" + ODSE.format("hcl", "f")
    jobs = [(f"hcl:{k}", "odse_hcl", hcl, pts3035[k:k + 1000], "EPSG:3035", 1) for k in range(0, len(pts3035), 1000)]
    signer = Signer()
    jobs += _esri_jobs(pts4326, 2017, signer, size=1)
    vals = run_jobs(jobs, os.path.join(OUTDIR, "exp94_europe_partial.jsonl"), threads=8)
    dom = {pid: first_valid(v, lambda x: True) for pid, v in vals.get("odse_hcl", {}).items()}
    # second pass on the subset: the probability of each point's own dominant class
    by_class = {}
    for p in pts3035:
        d = dom.get(p[0])
        if p[0] in sub and d is not None and int(d[0]) in ODSE_INDEX:
            by_class.setdefault(ODSE_INDEX[int(d[0])], []).append(p)
    jobs2 = [(f"p{c}:{k}", "odse_prob", "/vsicurl/" + ODSE.format(c, "p"), ps[k:k + 500], "EPSG:3035", 1)
             for c, ps in sorted(by_class.items()) if c not in (141, 523) for k in range(0, len(ps), 500)]
    prob = run_jobs(jobs2, os.path.join(OUTDIR, "exp94_europe_partial2.jsonl"), threads=8).get("odse_prob", {})
    rows = []
    for i in range(len(g)):
        pid = g["pid"].iat[i]
        d = dom.get(pid)
        code = ODSE_INDEX.get(int(d[0]), 0) if d is not None else 0
        es = first_valid(vals.get("esri2017", {}).get(pid), lambda v: ESRI_EU.get(int(v[0]), 0) > 0)
        pr = first_valid(prob.get(pid), lambda v: True)
        rows.append([pid, str(g["TILE"].iat[i]), int(g["S2GLC"].iat[i]), S2GLC.get(int(g["S2GLC"].iat[i]), 0),
                     code, CORINE.get(code, 0), int(pid in sub), "" if pr is None else int(pr[0]),
                     "" if es is None else ESRI_EU.get(int(es[0]), 0)])
    write_points(os.path.join(OUTDIR, "exp94_europe_points.csv"),
                 ["pid", "tile", "ref_s2glc", "ref", "odse_corine", "odse", "in_subset", "odse_prob", "esri"], rows)


# ----------------------------------------------------------------------------- analysis
def _ints(rows, k):
    return np.array([int(r[k]) if r[k] not in ("", None) else 0 for r in rows])


def which_map(ref, a, b, rng):
    import exp90_which_map as e90
    e90.R = R
    V = (ref > 0) & (a > 0) & (b > 0)
    N = int(V.sum()); ra, rb = (a == ref) & V, (b == ref) & V
    diff = V & (a != b)
    D, KA, KB = int(diff.sum()), int((ra & diff).sum()), int((rb & diff).sum())
    cells = {f"{d}_{n}": e90.cell(rng, (KA - KB) / N, N, D, KA, KB, n, d) for n in NS for d in ("disagree", "random")}
    return {"N": N, "D": D, "K_A": KA, "K_B": KB, "acc_a": float(ra.sum() / N), "acc_b": float(rb.sum() / N),
            "delta": (KA - KB) / N, "cells": cells}


def error_rate(wrong, rng):
    N, K = int(wrong.size), int(wrong.sum())
    n = N_RATE if N >= 1000 else 100
    ks = rng.hypergeometric(K, N - K, n, size=R)
    ivs = {k: est.hypergeom_interval(int(k), n, N) for k in np.unique(ks)}
    lo = np.array([ivs[k][0] for k in ks]); hi = np.array([ivs[k][1] for k in ks]); th = K / N
    return {"N": N, "errors": K, "labels": n, "error_rate": th, "coverage": float(((lo <= th) & (th <= hi)).mean()),
            "median_width_points": float(np.median(hi - lo) * 100)}


def confidence_cells(conf, wrong, rng):
    """Q3 and Q4 on one product's confidence (higher = surer) and its own errors."""
    import exp92_lcmap_products as e92
    import exp93_product_zone as e93
    e92.R = R
    q3 = e92.ranking((101 - conf).astype(float), wrong.astype(int), rng)        # as in exp92
    perm = np.random.default_rng(SEED).permutation(conf.size)
    c, w = conf[perm].astype(float), wrong[perm].astype(float)
    order, _ = est.zone_order(c)
    theta = float(w.mean())
    cell = e93.run_cell(w[order], c.size, min(B_ZONE, c.size // 2), theta / 2, {"package": (est.ZONE_GRID, None)}, R)
    rules = cell["grids"]["package"]["rules"]
    q4 = {"alpha": theta / 2, "labels": cell["budget"], "oracle_coverage": cell["grids"]["package"]["oracle_coverage"],
          **{r: {k: rules[r][k] for k in ("violation_rate", "no_zone_rate", "median_coverage")} for r in rules}}
    return q3, q4


def analyze():
    rng = np.random.default_rng(SEED)
    out = {"experiment": "exp94", "preregistered": True, "Q1": {}, "Q2": {}, "Q3": {}, "Q4": {}, "descriptive": {},
           "found_after_result": {}}            # descriptives added after the audit; none of them draws from rng
    # NLCD
    rows = list(csv.DictReader(open(os.path.join(OUTDIR, "exp94_nlcd_points.csv"))))
    for s, year in (("A", 2001), ("A", 2006), ("A", 2011), ("B", 2011), ("B", 2016)):
        rs = [r for r in rows if r["set"] == s and int(r["year"]) == year]
        ref2, nl2, lc = _ints(rs, "ref_nlcd"), _ints(rs, "nlcd"), _ints(rs, "lcmap")
        conf = _ints(rs, "lcpconf"); wts = np.array([float(r["weight"]) for r in rs])
        ref8 = np.array([NLCD_TO_LCMAP.get(int(v), 0) for v in ref2]); nl8 = np.array([NLCD_TO_LCMAP.get(int(v), 0) for v in nl2])
        tag = f"nlcd_{s}{year}"
        out["Q1"][f"{tag}:nlcd_vs_lcmap"] = which_map(ref8, nl8, lc, rng)
        v2 = (ref2 > 0) & (nl2 > 0)
        out["Q2"][f"{tag}:nlcd_level2"] = error_rate((nl2 != ref2)[v2], rng)
        v8 = (ref8 > 0) & (lc > 0)
        out["Q2"][f"{tag}:lcmap"] = error_rate((lc != ref8)[v8], rng)
        g = v8 & (conf >= 1) & (conf <= 100)
        out["Q3"][f"{tag}:lcmap_lcpconf"], out["Q4"][f"{tag}:lcmap_lcpconf"] = confidence_cells(conf[g], (lc != ref8)[g], rng)
        out["descriptive"][tag] = {
            "points": len(rs), "lcmap_missing": int((lc == 0).sum()), "lcpconf_codes": int((v8 & (conf > 100)).sum()),
            "weighted_accuracy_nlcd_level2": float((wts * (nl2 == ref2))[v2].sum() / wts[v2].sum()),
            "weighted_accuracy_lcmap": float((wts * (lc == ref8))[v8].sum() / wts[v8].sum())}
        alt = _ints(rs, "ref_nlcd_alternate")
        v = v8 & (nl8 > 0)
        out["found_after_result"][tag] = {
            "weighted_accuracy_nlcd_8class": float((wts * (nl8 == ref8))[v].sum() / wts[v].sum()),
            # NLCD's published accuracies count a match with the primary OR the alternate label: this reproduces them
            "weighted_accuracy_nlcd_level2_primary_or_alternate": float((wts * ((nl2 == ref2) | (nl2 == alt)))[v2].sum()
                                                                        / wts[v2].sum())}
    # GFC2020
    rows = list(csv.DictReader(open(os.path.join(OUTDIR, "exp94_gfc_points.csv"))))
    ref = _ints(rows, "ref_forest") + 1; gfc = _ints(rows, "gfc_v2") + 1
    wc = np.array([int(r["worldcover_forest"]) + 1 if r["worldcover_forest"] != "" else 0 for r in rows])
    out["Q1"]["gfc2020:gfc_vs_worldcover"] = which_map(ref, gfc, wc, rng)
    out["Q2"]["gfc2020:gfc_v2"] = error_rate(gfc != ref, rng)
    out["Q2"]["gfc2020:worldcover"] = error_rate((wc != ref)[wc > 0], rng)
    strata = _ints(rows, "stratum")
    area = {int(r["Strata"]): float(r["strata_ha"]) for r in csv.DictReader(open(os.path.join(DATA, "gfc2020", "strata_size.csv")))}

    def stratified(correct):
        """Stehman's estimator for a stratified sample: the strata's accuracies weighted by their areas."""
        hs = np.unique(strata)
        return float(sum(area[h] * correct[strata == h].mean() for h in hs) / sum(area[h] for h in hs))
    ea, eb = gfc != ref, wc != ref
    out["descriptive"]["gfc2020"] = {
        "points": len(rows), "worldcover_missing": int((wc == 0).sum()),
        "weighted_accuracy_gfc_v2": stratified(~ea), "weighted_accuracy_worldcover": stratified(~eb),
        "both_wrong": int((ea & eb).sum()), "only_gfc_wrong": int((ea & ~eb).sum()), "only_worldcover_wrong": int((~ea & eb).sum()),
        "error_correlation_phi": float(np.corrcoef(ea, eb)[0, 1])}
    # East Africa
    rows = list(csv.DictReader(open(os.path.join(OUTDIR, "exp94_eastafrica_points.csv"))))
    for year in (2015, 2016, 2017):
        rs = [r for r in rows if int(r["year"]) == year]
        ref, cg = _ints(rs, "ref"), _ints(rs, "cgls")
        pb = np.array([int(r["cgls_proba"]) if r["cgls_proba"] not in ("",) else 255 for r in rs])
        tag = f"eastafrica_{year}"
        v = (ref > 0) & (cg > 0)
        out["Q2"][f"{tag}:cgls"] = error_rate((cg != ref)[v], rng)
        g = v & (pb <= 100)
        out["Q3"][f"{tag}:cgls_proba"], out["Q4"][f"{tag}:cgls_proba"] = confidence_cells(pb[g], (cg != ref)[g], rng)
        out["descriptive"][tag] = {"points": len(rs), "cgls_missing": int((cg == 0).sum()), "proba_missing": int((v & (pb > 100)).sum())}
        out["found_after_result"][tag] = {"proba_missing_by_cgls_class": {
            EA_NAMES[c]: [int((v & (pb > 100) & (cg == c)).sum()), int((v & (cg == c)).sum())] for c in EA_NAMES}}
        if year == 2017:
            es = _ints(rs, "esri")
            out["Q1"][f"{tag}:cgls_vs_esri"] = which_map(ref, cg, es, rng)
            out["Q2"][f"{tag}:esri"] = error_rate((es != ref)[(ref > 0) & (es > 0)], rng)
            out["descriptive"][tag]["esri_missing"] = int((es == 0).sum())
    # Europe
    rows = list(csv.DictReader(open(os.path.join(OUTDIR, "exp94_europe_points.csv"))))
    ref, od, es = _ints(rows, "ref"), _ints(rows, "odse"), _ints(rows, "esri")
    sub = np.array([r["in_subset"] == "1" for r in rows])
    pr = np.array([int(r["odse_prob"]) if r["odse_prob"] != "" else -1 for r in rows])
    out["Q1"]["europe_2017:odse_vs_esri"] = which_map(ref, od, es, rng)
    out["Q2"]["europe_2017:odse"] = error_rate((od != ref)[(ref > 0) & (od > 0)], rng)
    out["Q2"]["europe_2017:esri"] = error_rate((es != ref)[(ref > 0) & (es > 0)], rng)
    g = sub & (ref > 0) & (od > 0) & (pr >= 0) & (pr <= 100)
    out["Q3"]["europe_2017:odse_prob"], out["Q4"]["europe_2017:odse_prob"] = confidence_cells(pr[g], (od != ref)[g], rng)
    out["descriptive"]["europe_2017"] = {"points": len(rows), "odse_missing": int((od == 0).sum()),
                                         "esri_missing": int((es == 0).sum()), "subset_graded": int(g.sum()),
                                         "subset_no_probability_layer": int((sub & np.isin(_ints(rows, "odse_corine"), (141, 523))).sum())}
    out["found_after_result"]["europe_2017"] = {
        "subset_no_odse_class": int((sub & (od == 0)).sum()),
        # ODSE's no-data value at the points is 0 (the preregistration named 49, 50, 255, none of which occurs)
        "odse_no_class_by_reference_class": {EU_NAMES[c]: [int(((od == 0) & (ref == c)).sum()), int((ref == c).sum())]
                                             for c in EU_NAMES}}
    bound = 0.1 + 3 * np.sqrt(0.1 * 0.9 / R)
    out["prereg"] = {
        "P1": {"holds": all(q["cells"][f"disagree_{n}"]["coverage"] >= 0.94 for q in out["Q1"].values() for n in NS)},
        "P2": {"holds": all(q["cells"]["disagree_100"]["median_width_points"] < q["cells"]["random_100"]["median_width_points"]
                            for q in out["Q1"].values())},
        "P3": {"holds": all(q["coverage"] >= 0.94 for q in out["Q2"].values())},
        "P4": {"holds": len(out["Q3"]) == 9 and all(q["auroc_ci95"][0] > 0.5 for q in out["Q3"].values())},
        "P5": {"holds": len(out["Q4"]) == 9 and all(q[r]["violation_rate"] <= bound for q in out["Q4"].values()
                                                     for r in ("prefix", "bonferroni"))}}
    json.dump(out, open(SUMMARY, "w"), indent=1, default=float)
    print(json.dumps(out["prereg"]))


def smoke():
    """Synthetic points through the analysis functions: no download, no product value."""
    rng = np.random.default_rng(0)
    n = 3000
    ref = rng.integers(1, 6, n)
    a = np.where(rng.random(n) < 0.8, ref, rng.integers(1, 6, n))
    b = np.where(rng.random(n) < 0.75, ref, rng.integers(1, 6, n))
    q1 = which_map(ref, a, b, rng)
    q2 = error_rate(a != ref, rng)
    conf = np.where(a == ref, rng.integers(40, 101, n), rng.integers(1, 80, n))
    q3, q4 = confidence_cells(conf, a != ref, rng)
    print(json.dumps({"q1_cov": q1["cells"]["disagree_100"]["coverage"], "q2_cov": q2["coverage"], "auroc": q3["auroc"],
                      "prefix_viol": q4["prefix"]["violation_rate"]}))
    assert majority([5, 5, 2, 2, 2, 5, 0, 0, 5]) == 5 and majority([5, 5, 0, 0, 2, 0, 2, 2, 5]) == 2
    assert majority([0, 0, 0, 0, 0, 5, 5, 5, 5]) == 0
    assert all(NLCD_TO_LCMAP[k] in LCMAP_NAMES for k in NLCD_TO_LCMAP) and len(ODSE_INDEX) == 44
    print("smoke ok")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cmd", choices=("extract-nlcd", "extract-gfc", "extract-eastafrica", "extract-europe", "analyze", "smoke"))
    a = p.parse_args(argv)
    {"extract-nlcd": extract_nlcd, "extract-gfc": extract_gfc, "extract-eastafrica": extract_eastafrica,
     "extract-europe": extract_europe, "analyze": analyze, "smoke": smoke}[a.cmd]()


if __name__ == "__main__":
    main()
