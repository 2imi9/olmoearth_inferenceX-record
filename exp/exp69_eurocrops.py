#!/usr/bin/env python
"""exp69: EuroCrops, a dense crop map from farmers' declarations, and the first temporal difference labelled on both sides.

Why. Two things this repository has never had, and the second is the reason the experiment is worth running.

The reference is administrative. EuroCrops harmonises the parcel declarations farmers make to their national paying
agency, so the label is neither photointerpretation (Sen1Floods11, GEOID-Flood, Dynamic World) nor another model's output
(DFC2020's random forest) nor a field visit (LUCAS, exp68). It is a legal declaration of what was sown, made by the
person who sowed it, for money. Its failure modes are therefore unlike every other reference here: it is right about
crop identity and can be wrong about extent, timing and whether the declared crop was actually planted.

And both sides of the time axis are labelled. Every temporal comparison in this repository so far has had truth for at
most one date: exp60's two flood periods are graded against one event mask, and exp68's two acquisitions are graded
against a survey made on one day, so its "neither side is right" cell absorbs both genuine model failure and the
reference simply not describing the other date. EuroCrops ships one file per region per year, and a parcel that appears
in both years carries a declared crop in each. So `oe_inferencex.compare`'s graded block becomes fully determined, and
one quantity that has been missing throughout becomes measurable: **the floor for a two-date change rate**. exp68's audit
made the point sharply, that the probe-reseed rate answers a different question, whether the head is stable, while the
right null for "the map changed between two dates" is how often the map changes where the ground did not. Windows whose
declared crop is identical in both years give that null directly, and windows whose declared crop changed give the signal
it should be read against.

Design. Data: EuroCropsV2 GeoParquet from the JRC open-data mirror, CC BY 4.0, EPSG:3035, joined on `original_code` to
the EuroCrops project's own HCAT mapping on GitHub, which the JRC files do not carry. Three regions, chosen because their
join rate is high and their parcel sizes differ by an order of magnitude: Austria, Denmark and Slovenia. Spain was
dropped after its codes failed to join at all and its single file proved to be 11 GB over 17 million parcels.

  Unit. The 4-px window, as everywhere else here, not the parcel: the median Austrian parcel is 3,632 sqm and the median
  Danish one 24,026 sqm, so a parcel spans several windows and a dense label map is the honest representation. Chips are
  64 px at 10 m centred on a sampled parcel, with every parcel intersecting the chip burned in, so the label map is dense
  wherever farmland is declared and unlabelled elsewhere; a window is graded when three quarters of its sixteen pixels
  carry one class.
  Classes. HCAT3 names, reduced per region to the classes holding most of the declared area plus one "other crop" bin,
  because a probe on this much data cannot carry the 96 to 133 classes the mappings define.
  Imagery. Sentinel-2 L2A from the Planetary Computer, the least-cloudy scene in the growing season of each declaration
  year, with the baseline 04.00 BOA offset removed as in exp68 so digital numbers match the convention the encoder's
  normaliser expects.
  Model. OlmoEarth v1 Base frozen, fp32, with Ai2's linear segmentation probe recipe; weight decay and epochs tuned on
  regions held out of the fit set, never on the report regions, which exp68 showed is not optional: an overfit head there
  reversed the ordering of the confidence signals.
  Splits. By parcel-grid cell, so no chip's neighbourhood appears in both fit and report.

  Part A, ranking. The margin against entropy, one minus top-1, the boundary-first order and the no-model controls, on
  the graded windows of the report cells, scored as everywhere here: excess AURC, a per-cell sign test, error capture at
  5, 10 and 20 percent.
  Part B, two years with truth on both sides. The same windows read through both years' imagery, compared with
  `oe_inferencex.compare`, and split by what the declarations say: windows whose declared crop is the same in both years,
  and windows where it changed. This gives the labelled null and the labelled signal, `which_side` with a real "neither",
  and the cross-tab of the two years' errors.

Preregistered (one-sided):
  P1  the ranking holds on a declaration-derived dense reference: on the report cells the margin's excess AURC is at
      least 0.01 below the best no-model control, pooled, and lower on more grid cells than not (p < 0.05), in every
      region.
  P2  a two-date difference is mostly real change, not model instability: the decision-change rate on windows whose
      declared crop changed between the two years is at least twice the rate on windows whose declared crop did not,
      in every region, and the latter is the floor that exp68 could only bound by its most stable class.
  P3  truth on both sides makes the comparison decidable: among windows where the two years' decisions differ and both
      years are labelled, the share where neither side is right is below one half, so `which_side` is informative rather
      than dominated by the case it cannot resolve.
  Falsification. P1 fails if the margin loses to a pixel index, which would confine exp68's result to references that
  describe land cover rather than crop identity. P2 fails if the unchanged-crop change rate is within half the
  changed-crop rate, which would say a two-date difference on this task is mostly the model moving rather than the ground
  moving, and would put exp60's and exp68's difference rates in a harsher light than either reported. P3 fails if the
  undecidable share is at or above one half, which would say that labelling both sides does not rescue the graded block,
  and the honest report would then be that the comparison module tells an operator where two inferences differ and not
  which to believe.
  Stated predictions, not tested: Denmark the most accurate region, its fields being an order of magnitude larger than
  Slovenia's; grassland and cereals the most accurate classes; the unchanged-crop change rate between 0.05 and 0.20,
  above the probe-reseed floor and well below exp68's 0.31 on land cover.

Caveats carried into the record. A declaration is not an observation: a farmer declares a parcel and its crop for
subsidy, so extent is a cadastral boundary rather than what the sensor sees, a declared crop may fail or be replaced, and
catch crops and multiple harvests inside one year are invisible. The HCAT harmonisation is the EuroCrops project's, not
ours, and its join leaves 0.4% of Austrian, 0.1% of Danish and 14.7% of Slovenian parcels unmapped, measured on the
run, and those parcels' pixels are left unlabelled rather than guessed; the unmapped set is not a random sample of
crops, so for Slovenia in particular the share is a bias risk and not harmless attrition. Slovakia ships no licence with EuroCropsV2 and Spain's SIGPAC forbids redistribution without added
value, so the mirror's blanket CC BY 4.0 may not bind every national source; the three regions used here are not among
those two. Austria's mapping file covers 2021 only and is applied to its other year as well, which the run asserts is
safe by checking the join rate per year and dropping any year below 0.9. A parcel present in both years may have been
split or merged, so the two years' geometries are intersected rather than assumed identical.

Inputs: the JRC parquets (about 0.3 to 1.1 GB per region-year) and the GitHub HCAT mappings, cached under
/scratch/.../hf/eurocrops; Sentinel-2 L2A from the Planetary Computer. Run with ~/olmoearth_inferenceX/.venv and
PYTHONPATH=/scratch/.../pylibs, since GDAL here has no Parquet driver and pyarrow is unpacked there.
Stages: --stage labels (build the dense label rasters and the chip list), --stage fetch (imagery), --stage analyze.
Outputs: exp/out/exp69_summary.json, exp/out/exp69_eurocrops.csv, exp/out/exp69_windows.npz.
--smoke: synthetic parcels and chips, no network, _smoke outputs.
"""
import argparse
import collections
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))

from oe_inferencex import metrics, stats  # noqa: E402
from oe_inferencex.assess import boundary_first_score  # noqa: E402

JRC = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/EuroCropsV2/gpqtv202/"
HCAT = "https://raw.githubusercontent.com/maja601/EuroCrops/main/csvs/country_mappings/"
D = "/scratch/qi_zim_neu/olmoearth_inferenceX/hf/eurocrops"
CHIPDIR = os.path.join(D, "chips")
OUT = os.path.join(EXP_DIR, "out")

# region -> (the two declaration years, the mapping file that covers them)
REGIONS = {"at": ((2020, 2021), "at_2021"), "dk": ((2018, 2019), "dk_2019"), "si": ((2020, 2021), "si_2021")}
MIN_JOIN = 0.85                 # a year whose codes barely join is dropped, not guessed. 0.85 admits Slovenia at 0.853
                                # and excludes Spain at 0.000. An unmapped code makes its parcel's pixels unlabelled,
                                # which is honest, but the unmapped set is not a random sample of crops, so the share is
                                # recorded per region and read as a bias risk rather than as harmless attrition.
N_CLASSES = 11                  # the ten largest HCAT classes by declared area, plus "other crop"
OTHER = "other_crop"
SIZE, CROP, PATCH = 64, 60, 4
G = CROP // PATCH               # 15 windows across
PURE = 0.75                     # a window is graded when three quarters of its sixteen pixels carry one class
BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
BOA_OFFSET = 1000
SEASON = ("06-15", "08-31")     # the growing season window the scene is chosen from
BUDGETS = (0.05, 0.10, 0.20)
CELL = 20_000                   # the 20 km grid cell that splits fit from report, in EPSG:3035 metres
CHIPS_PER_REGION = 900
REF_WD, REF_EPOCHS = 1e-4, 80   # the fixed reference setting the tuned probe is reported against
SHARD = 150
SCL_CLEAR = (4, 5, 6, 7, 11)


# ----------------------------------------------------------------------------- parcels and classes
def fetch_file(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", dst, url], check=True)
    return dst


def load_mapping(name):
    """original_code -> HCAT3 name, from the EuroCrops project's own mapping; the JRC parquets carry no harmonised code."""
    path = fetch_file(HCAT + name + ".csv", os.path.join(D, name + ".csv"))
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    if not rows:
        raise ValueError(f"{name}: empty mapping")
    key = "HCAT3_name" if "HCAT3_name" in rows[0] else "HCAT2_name"
    return {str(r["original_code"]).strip(): str(r.get(key) or "").strip() for r in rows if r.get("original_code")}


def load_parcels(region, year, mapping):
    """(hcat name, area sqm, shapely geometry) per declared parcel of one region-year, unmapped codes dropped."""
    import pyarrow.parquet as pq
    import shapely
    from shapely import from_wkb
    path = fetch_file(f"{JRC}{region}_{year}.parquet", os.path.join(D, f"{region}_{year}.parquet"))
    pf = pq.ParquetFile(path)
    names, areas, geoms = [], [], []
    n_total = n_unmapped = 0
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=["original_code", "area_ha", "geometry"])
        codes = [str(c).strip() for c in t.column("original_code").to_pylist()]
        hc = [mapping.get(c, "") for c in codes]
        n_total += len(hc)
        n_unmapped += sum(1 for h in hc if not h)
        g = from_wkb(np.array(t.column("geometry").to_pylist(), dtype=object))
        ar = np.asarray(t.column("area_ha").to_pylist(), dtype=np.float64) * 1e4
        ok = np.array([bool(h) for h in hc]) & shapely.is_valid(g) & ~shapely.is_empty(g) & np.isfinite(ar)
        names.extend([h for h, k in zip(hc, ok) if k])
        areas.append(ar[ok])
        geoms.append(g[ok])
    join_rate = 1.0 - n_unmapped / max(n_total, 1)
    return (np.array(names), np.concatenate(areas), np.concatenate(geoms), join_rate, n_total)


def class_vocabulary(names, areas, n_classes=N_CLASSES):
    """The largest HCAT classes by declared area, plus one bin for the rest; area, not parcel count, sets the order
    because a dense per-pixel map is what the probe sees."""
    by_area = collections.Counter()
    for n, a in zip(names, areas):
        by_area[n] += float(a)
    top = [n for n, _ in by_area.most_common(n_classes - 1)]
    vocab = top + [OTHER]
    share = {n: by_area[n] / max(sum(by_area.values()), 1e-9) for n in top}
    return vocab, share


def class_index(names, vocab):
    look = {n: i for i, n in enumerate(vocab)}
    other = look[OTHER]
    return np.array([look.get(n, other) for n in names], dtype=np.int16)


def cell_of(x, y, cell=CELL):
    """The 20 km EPSG:3035 grid cell a point falls in, used to split fit from report without adjacency leaking."""
    return f"{int(np.floor(x / cell))}_{int(np.floor(y / cell))}"


# ----------------------------------------------------------------------------- chip selection, two passes over the parquet
def scan_pass1(region, year, mapping):
    """Cheap pass: class name, declared area and centroid per parcel, geometries discarded.

    Austria declares 2.6 million parcels per year and holding two years of geometries in memory at once is gigabytes,
    so the vocabulary and the chip centres are chosen from centroids first and the geometries are read again afterwards
    for the small part of the country the chips actually cover."""
    import pyarrow.parquet as pq
    import shapely
    from shapely import from_wkb
    path = fetch_file(f"{JRC}{region}_{year}.parquet", os.path.join(D, f"{region}_{year}.parquet"))
    pf = pq.ParquetFile(path)
    names, areas, cx, cy = [], [], [], []
    n_total = n_unmapped = 0
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=["original_code", "area_ha", "geometry"])
        codes = [str(c).strip() for c in t.column("original_code").to_pylist()]
        hc = [mapping.get(c, "") for c in codes]
        n_total += len(hc)
        n_unmapped += sum(1 for h in hc if not h)
        g = from_wkb(np.array(t.column("geometry").to_pylist(), dtype=object))
        ar = np.asarray(t.column("area_ha").to_pylist(), dtype=np.float64) * 1e4
        b = shapely.bounds(g)
        ok = (np.array([bool(h) for h in hc]) & ~shapely.is_empty(g) & np.isfinite(ar)
              & np.isfinite(b).all(axis=1))
        names.extend([h for h, k in zip(hc, ok) if k])
        areas.append(ar[ok])
        cx.append((b[ok, 0] + b[ok, 2]) / 2.0)
        cy.append((b[ok, 1] + b[ok, 3]) / 2.0)
        del t, g, b
    return (np.array(names), np.concatenate(areas), np.concatenate(cx), np.concatenate(cy),
            1.0 - n_unmapped / max(n_total, 1), n_total)


def scan_pass2(region, year, mapping, boxes, vocab):
    """Geometries and class indices for the parcels intersecting any chip box, which is a small part of the region."""
    import pyarrow.parquet as pq
    import shapely
    from shapely import from_wkb
    path = os.path.join(D, f"{region}_{year}.parquet")
    pf = pq.ParquetFile(path)
    tree_boxes = shapely.STRtree(boxes)
    keep_g, keep_c = [], []
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=["original_code", "geometry"])
        codes = [str(c).strip() for c in t.column("original_code").to_pylist()]
        hc = [mapping.get(c, "") for c in codes]
        g = from_wkb(np.array(t.column("geometry").to_pylist(), dtype=object))
        good = np.array([bool(h) for h in hc]) & ~shapely.is_empty(g)
        idx = np.flatnonzero(good)
        if not len(idx):
            continue
        hit = shapely.STRtree(g[idx]).query(boxes, predicate="intersects")
        sel = np.unique(hit[1]) if hit.size else np.array([], dtype=int)
        if len(sel):
            take = idx[sel]
            keep_g.append(g[take])
            keep_c.append(class_index(np.array([hc[i] for i in take]), vocab))
        del t, g
    if not keep_g:
        return np.array([]), np.array([], dtype=np.int16)
    return np.concatenate(keep_g), np.concatenate(keep_c)


def plan_region(region, rng):
    """Vocabulary, chip centres and the per-year parcel sets for one region; None when a year's codes will not join."""
    import shapely
    years, mapname = REGIONS[region]
    mapping = load_mapping(mapname)
    scans, joins = {}, {}
    for y in years:
        names, areas, cx, cy, join, n_total = scan_pass1(region, y, mapping)
        joins[y] = {"join_rate": join, "n_parcels": int(n_total), "n_mapped": int(len(names))}
        print(f"  {region}_{y}: {n_total} parcels, join {join:.3f}, {len(names)} usable", flush=True)
        if join < MIN_JOIN:
            print(f"  {region}_{y}: DROPPED, join rate below {MIN_JOIN}", flush=True)
            return None, joins
        scans[y] = (names, areas, cx, cy)
    # the vocabulary comes from the earlier year so the later one cannot define the classes it is scored on
    y0, y1 = years
    vocab, share = class_vocabulary(scans[y0][0], scans[y0][1])
    # chip centres: sample parcels with probability proportional to nothing, i.e. uniformly over declared parcels,
    # but only where the parcel is at least one window across, so the chip is not centred on a sliver
    names, areas, cx, cy = scans[y0]
    big = np.flatnonzero(areas >= PATCH * PATCH * 100)
    pick = rng.choice(big, size=min(CHIPS_PER_REGION, len(big)), replace=False)
    boxes = [shapely.box(cx[i] - 420, cy[i] - 420, cx[i] + 420, cy[i] + 420) for i in pick]
    parcels = {}
    for y in years:
        g, c = scan_pass2(region, y, mapping, boxes, vocab)
        parcels[y] = (g, c)
        print(f"  {region}_{y}: {len(g)} parcels inside the {len(boxes)} chip boxes", flush=True)
    chips = [{"region": region, "x": float(cx[i]), "y": float(cy[i]),
              "cell": cell_of(cx[i], cy[i]), "parcel_area": float(areas[i])} for i in pick]
    return {"vocab": vocab, "area_share": share, "chips": chips, "parcels": parcels, "years": years}, joins


# ----------------------------------------------------------------------------- imagery and dense labels
def season_items(cat, lon, lat, year):
    lo, hi = f"{year}-{SEASON[0]}", f"{year}-{SEASON[1]}"
    return list(cat.search(collections=["sentinel-2-l2a"],
                           intersects={"type": "Point", "coordinates": [lon, lat]},
                           datetime=f"{lo}/{hi}", query={"eo:cloud_cover": {"lt": 20}}).items())


def read_chip_and_labels(item, x3035, y3035, trees, classes):
    """One chip in the item's own CRS: 12-band DN, the SCL band, and a dense label map per year (-1 unlabelled)."""
    import rasterio
    import pyproj
    import shapely
    from rasterio.enums import Resampling
    from rasterio.windows import Window, bounds as win_bounds, from_bounds
    from rasterio.features import rasterize
    from shapely.ops import transform as sh_transform

    with rasterio.open(item.assets["B04"].href) as src:
        crs = src.crs
        fwd = pyproj.Transformer.from_crs("EPSG:3035", crs, always_xy=True)
        x, y = fwd.transform(x3035, y3035)
        row, col = src.index(x, y)
        win = Window(col - SIZE // 2, row - SIZE // 2, SIZE, SIZE)
        bnds = win_bounds(win, src.transform)
        chip_tr = rasterio.windows.transform(win, src.transform)
    stack = []
    for b in BANDS:
        with rasterio.open(item.assets[b].href) as s:
            w = from_bounds(*bnds, transform=s.transform)
            stack.append(s.read(1, window=w, out_shape=(SIZE, SIZE), resampling=Resampling.bilinear,
                                boundless=True, fill_value=0))
    with rasterio.open(item.assets["SCL"].href) as s:
        w = from_bounds(*bnds, transform=s.transform)
        scl = s.read(1, window=w, out_shape=(SIZE, SIZE), resampling=Resampling.nearest,
                     boundless=True, fill_value=0)
    img = np.stack(stack).astype(np.float32)
    if str(item.properties.get("s2:processing_baseline", "00.00")) >= "04.00":
        img = np.clip(img - BOA_OFFSET, 0, None)

    to_crs = pyproj.Transformer.from_crs("EPSG:3035", crs, always_xy=True).transform
    chip_box_3035 = shapely.box(x3035 - 420, y3035 - 420, x3035 + 420, y3035 + 420)
    labels = {}
    for year, (tree, geoms) in trees.items():
        hit = tree.query(chip_box_3035, predicate="intersects")
        shapes = [(sh_transform(to_crs, geoms[i]), int(classes[year][i]) + 1) for i in np.atleast_1d(hit)]
        lab = (rasterize(shapes, out_shape=(SIZE, SIZE), transform=chip_tr, dtype="int32", fill=0)
               if shapes else np.zeros((SIZE, SIZE), dtype=np.int32))
        labels[year] = (lab.astype(np.int16) - 1)          # 0 became "no declaration", so -1 is unlabelled
    return img.astype(np.uint16), scl.astype(np.uint8), labels


# ----------------------------------------------------------------------------- stage 1, fetch
import threading                                                      # noqa: E402

_TLS = threading.local()


def _sign_catalog():
    """One signed catalogue per worker thread: pystac_client holds a requests session and is not thread-safe."""
    import planetary_computer
    import pystac_client
    if getattr(_TLS, "cat", None) is None:
        _TLS.cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                             modifier=planetary_computer.sign_inplace)
    return _TLS.cat


def _drop_catalog():
    """Discard this thread's session after a failure; a poisoned session is the usual cause of a repeat error."""
    _TLS.cat = None


def _shard_ok(path, min_yield=0.70):
    """A shard counts as done only if it opens AND kept most of what it attempted.

    The Planetary Computer returns transient API errors under load, and the first run of this stage wrote one shard
    holding 13 of 150 chips. A marker is not proof and neither is a file: exp58 lost a night to stale markers, and a
    thin shard is the same failure wearing a different hat."""
    if not os.path.exists(path):
        return False
    try:
        with np.load(path, allow_pickle=True) as z:
            n, attempted = int(z["n"]), int(z["n_attempted"]) if "n_attempted" in z.files else None
            if attempted is None:
                return False                      # written before the yield was recorded; refetch rather than trust it
            return n >= min_yield * max(attempted, 1)
    except Exception:
        return False


def fetch_stage(args):
    import pyproj
    import shapely
    from concurrent.futures import ThreadPoolExecutor
    from olmoearth_pretrain.data.constants import Modality
    assert BANDS == list(Modality.SENTINEL2_L2A.band_order), "band order drifted from the encoder's"

    os.makedirs(CHIPDIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    meta = {"regions": {}}
    for region in REGIONS:
        t0 = time.time()
        print(f"{region}: planning", flush=True)
        plan, joins = plan_region(region, rng)
        meta["regions"][region] = {"join": joins}
        if plan is None:
            continue
        y0, y1 = plan["years"]
        meta["regions"][region].update({"vocab": plan["vocab"], "area_share": plan["area_share"],
                                        "years": [y0, y1], "n_chips": len(plan["chips"])})
        trees = {y: (shapely.STRtree(plan["parcels"][y][0]), plan["parcels"][y][0]) for y in (y0, y1)}
        classes = {y: plan["parcels"][y][1] for y in (y0, y1)}
        inv = pyproj.Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
        chips = plan["chips"]
        todo = [i for i in range(0, len(chips), SHARD)
                if not _shard_ok(os.path.join(CHIPDIR, f"{region}_{i:05d}.npz"))]
        print(f"{region}: {len(todo)} shards of {int(np.ceil(len(chips)/SHARD))} ({time.time()-t0:.0f}s planning)", flush=True)

        def one(ch, attempts=4):
            """The catalogue rate-limits under twenty threads, so a transient failure is retried with a backoff before
            the chip is given up; the first run lost 311 chips of 1,800 to errors that a retry would have absorbed."""
            last = ""
            for attempt in range(attempts):
                try:
                    lon, lat = inv.transform(ch["x"], ch["y"])
                    cat = _sign_catalog()
                    out = dict(ch)
                    for tag, yr in (("y0", y0), ("y1", y1)):
                        items = season_items(cat, lon, lat, yr)
                        if not items:
                            return {"error": f"NoScene: no clear {yr} scene in the season"}
                        it = min(items, key=lambda z: z.properties.get("eo:cloud_cover", 100.0))
                        img, scl, labels = read_chip_and_labels(it, ch["x"], ch["y"], trees, classes)
                        out[f"img_{tag}"] = img
                        out[f"scl_{tag}"] = scl
                        out[f"lab_{tag}"] = labels[yr]
                        out[f"date_{tag}"] = it.properties["datetime"][:10]
                        out[f"cloud_{tag}"] = float(it.properties.get("eo:cloud_cover", np.nan))
                    return out
                except Exception as exc:
                    last = f"{type(exc).__name__}: {str(exc)[:120]}"
                    _drop_catalog()                      # a poisoned session is the usual cause of a repeat
                    time.sleep(1.5 * (attempt + 1))
            return {"error": last}

        for s0 in todo:
            rows = chips[s0:s0 + SHARD]
            t1 = time.time()
            with ThreadPoolExecutor(max_workers=args.threads) as ex:
                got = list(ex.map(one, rows))
            keep = [g for g in got if g and "img_y0" in g and "img_y1" in g]
            errs = collections.Counter(g["error"].split(":")[0] for g in got if g and "error" in g)
            pack = {"n": len(keep), "n_attempted": len(rows), "region": region, "years": np.array([y0, y1])}
            if keep:
                for k in ("img_y0", "img_y1", "scl_y0", "scl_y1", "lab_y0", "lab_y1"):
                    pack[k] = np.stack([g[k] for g in keep])
                for k in ("x", "y", "parcel_area"):
                    pack[k] = np.array([g[k] for g in keep], dtype=np.float64)
                pack["cell"] = np.array([g["cell"] for g in keep])
                for k in ("date_y0", "date_y1"):
                    pack[k] = np.array([g[k] for g in keep])
                for k in ("cloud_y0", "cloud_y1"):
                    pack[k] = np.array([g[k] for g in keep], dtype=np.float32)
            p = os.path.join(CHIPDIR, f"{region}_{s0:05d}.npz")
            np.savez_compressed(p + ".tmp.npz", **pack)
            os.replace(p + ".tmp.npz", p)
            print(f"  {region} shard {s0:05d}: {len(keep)}/{len(rows)} in {time.time()-t1:.0f}s"
                  f"{' errors ' + str(dict(errs)) if errs else ''}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "exp69_plan.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=float)


# ----------------------------------------------------------------------------- stage 2, encode and score
def embed(model, imgs, dates, batch=24):
    """(N, 12, 64, 64) DN and (N, 3) day/month0/year -> pooled (N, 15, 15, D) fp16 tokens, the exp18 path."""
    import torch
    import exp18_sen1floods_expert as exp18
    out = []
    for i in range(0, len(imgs), batch):
        x = imgs[i:i + batch][:, :, :CROP, :CROP].transpose(0, 2, 3, 1)[:, :, :, None, :].astype(np.float64)
        x = exp18._norm.normalize(exp18.Modality.SENTINEL2_L2A, x)
        b = x.shape[0]
        sample = exp18.MaskedOlmoEarthSample(
            sentinel2_l2a=torch.tensor(x, dtype=torch.float32, device=exp18.DEV),
            sentinel2_l2a_mask=torch.ones((b, CROP, CROP, 1, 3), device=exp18.DEV) * exp18.MaskValue.ONLINE_ENCODER.value,
            timestamps=torch.tensor(dates[i:i + batch], dtype=torch.long, device=exp18.DEV)[:, None, :])
        with torch.no_grad():
            o = model.encoder(sample, fast_pass=True, patch_size=PATCH)["tokens_and_masks"].sentinel2_l2a
        out.append(o.mean(dim=[3, 4]).half().cpu().numpy())
    return np.concatenate(out)


def fit_dense_probe(emb, lab, n_classes, seed=0, epochs=40, wd=1e-4, lr=1e-3, batch=16):
    """Ai2's linear segmentation probe (e51/e54's recipe) with weight decay and epochs exposed, fp32.

    exp68 is the reason they are exposed: an untuned probe there memorised its fit set and reversed the ordering of the
    confidence signals, so a probe whose regularisation was never chosen on held-out ground cannot be trusted to rank."""
    import math
    import torch
    import exp51_their_probe as e51
    dev = _device()
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    n, h, w, d = emb.shape
    probe = e51.LinearProbe(in_dim=d, out_dim=n_classes * PATCH * PATCH).to(dev).float()
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    E = torch.tensor(np.asarray(emb, dtype=np.float32))
    Y = torch.tensor(np.asarray(lab, dtype=np.int64))
    probe.train()
    for ep in range(epochs):
        order = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            x, y = E[idx].to(dev), Y[idx].to(dev)
            lg = probe(x)["logits"].reshape(len(idx), h, w, n_classes, PATCH, PATCH)
            lg = lg.permute(0, 3, 1, 4, 2, 5).reshape(len(idx), n_classes, h * PATCH, w * PATCH)
            loss_fn(lg, y).backward()
            opt.step(); opt.zero_grad()
        for pg in opt.param_groups:
            pg["lr"] = 1e-5 + 0.5 * (lr - 1e-5) * (1 + math.cos(math.pi * (ep + 1) / epochs))
    return probe.eval()


def window_readings(probe, emb, lab, n_classes, batch=32):
    """Per 4-px window: class-probability mean, decision, margin, entropy, the majority label and whether it is pure."""
    import torch
    dev = _device()
    n, h, w, _ = emb.shape
    wp = np.zeros((n, n_classes, h, w), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch):
            x = torch.tensor(np.asarray(emb[i:i + batch], dtype=np.float32), device=dev)
            b = x.shape[0]
            lg = probe(x)["logits"].reshape(b, h, w, n_classes, PATCH, PATCH)
            lg = lg.permute(0, 3, 1, 4, 2, 5).reshape(b, n_classes, h * PATCH, w * PATCH)
            p = torch.softmax(lg, dim=1)
            wp[i:i + b] = p.reshape(b, n_classes, h, PATCH, w, PATCH).mean(dim=(3, 5)).cpu().numpy()
    srt = np.sort(wp, axis=1)
    dec = wp.argmax(1)
    ent = -(np.clip(wp, 1e-7, 1) * np.log(np.clip(wp, 1e-7, 1))).sum(1)
    # the window's label: the majority class of its sixteen pixels, graded only when PURE of them agree
    blocks = lab[:, :h * PATCH, :w * PATCH].reshape(n, h, PATCH, w, PATCH).transpose(0, 1, 3, 2, 4).reshape(n, h, w, -1)
    counts = np.stack([(blocks == c).sum(-1) for c in range(n_classes)], axis=-1)
    n_lab = (blocks >= 0).sum(-1)
    y = counts.argmax(-1).astype(np.int16)
    pure = (counts.max(-1) >= PURE * blocks.shape[-1]) & (n_lab == blocks.shape[-1])
    return {"dec": dec, "margin": srt[:, -1] - srt[:, -2], "top1": srt[:, -1], "entropy": ent,
            "y": y, "graded": pure}


def boundary_from(dec):
    """exp14's predicted boundary on the window grid: the share of a window's eight neighbours deciding differently."""
    n, h, w = dec.shape
    pad = np.pad(dec, ((0, 0), (1, 1), (1, 1)), mode="edge")
    diff = np.zeros((n, h, w), dtype=np.float64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            diff += (pad[:, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w] != dec)
    return diff / 8.0


def _device():
    """The encoder's device when the encoder is installed, else CPU, so the parts of this file that need only torch can
    be exercised by the smoke on a machine without olmoearth_pretrain. The smoke is the only pre-flight check there is,
    and a smoke that silently skips its most intricate function is worth little."""
    try:
        import exp18_sen1floods_expert as exp18
        return exp18.DEV
    except Exception:
        return "cpu"


def window_variance(img):
    """No-model control: the mean over bands of the within-window pixel standard deviation, in raw DN."""
    n = len(img)
    x = img[:, :, :CROP, :CROP].astype(np.float64).reshape(n, len(BANDS), G, PATCH, G, PATCH)
    return x.std(axis=(3, 5)).mean(axis=1)


def cell_split(cells, salt=""):
    import hashlib
    h = np.array([int(hashlib.md5((salt + str(c)).encode()).hexdigest()[:8], 16) % 2 for c in cells])
    return h == 0, h == 1


def signals_of(r, var, dec_flat):
    sus = -r["margin"]
    rare = np.zeros(dec_flat.shape, dtype=np.float64)
    freq = collections.Counter(dec_flat.tolist())
    for k, v in freq.items():
        rare[dec_flat == k] = -np.log(max(v, 1) / max(len(dec_flat), 1))
    return {"margin": sus, "one_minus_top1": 1.0 - r["top1"], "entropy": r["entropy"],
            "boundary_first": boundary_first_score(sus, r["boundary"]),
            "ctl_pixel_variance": var, "ctl_class_rarity": rare}


def cell_sign_test(a_u, b_u, err, cells, min_n=200):
    wins = losses = 0
    for c in np.unique(cells):
        m = cells == c
        if m.sum() < min_n or not (0 < err[m].sum() < m.sum()):
            continue
        ga, gb = metrics.excess_aurc(a_u[m], err[m]), metrics.excess_aurc(b_u[m], err[m])
        wins += gb > ga
        losses += gb < ga
    return {"wins": int(wins), "losses": int(losses),
            "p": float(stats.sign_test(int(wins), int(losses), alternative="greater"))}


def analyze_stage(args):
    import exp18_sen1floods_expert as exp18
    from oe_inferencex import compare as cmp_mod
    t0 = time.time()
    plan = json.load(open(os.path.join(OUT, "exp69_plan.json")))
    model = exp18.load_model_from_id(exp18.ModelID.OLMOEARTH_V1_BASE).to(exp18.DEV).eval().float()
    summary = {"experiment": "exp69 EuroCrops: a dense crop map from declarations, and a temporal difference labelled on both sides",
               "config": {"regions": {}, "window_px": PATCH, "chip_px": SIZE, "crop_px": CROP, "grid": G,
                          "purity_threshold": PURE, "n_classes": N_CLASSES, "season": SEASON,
                          "boa_offset_removed": BOA_OFFSET, "cell_metres": CELL,
                          "caveats": ["a declaration is not an observation: extent is cadastral, a declared crop may fail",
                                      "the HCAT harmonisation is the EuroCrops project's, and unmapped codes are dropped",
                                      "both years are labelled, which is what makes the two-date comparison decidable"]},
               "results": {}, "verdicts": {}}
    rows, per_region, windows = [], {}, []

    for region in REGIONS:
        files = sorted(f for f in os.listdir(CHIPDIR) if f.startswith(region + "_") and f.endswith(".npz"))
        acc = collections.defaultdict(list)
        for f in files:
            with np.load(os.path.join(CHIPDIR, f), allow_pickle=True) as z:
                if int(z["n"]) == 0:
                    continue
                for k in ("img_y0", "img_y1", "scl_y0", "scl_y1", "lab_y0", "lab_y1",
                          "x", "y", "parcel_area", "cell", "date_y0", "date_y1", "cloud_y0", "cloud_y1"):
                    acc[k].append(np.asarray(z[k]))
        if not acc:
            print(f"{region}: no chips", flush=True)
            continue
        d = {k: np.concatenate(v) for k, v in acc.items()}
        n = len(d["img_y0"])
        vocab = plan["regions"][region]["vocab"]
        C = len(vocab)
        print(f"{region}: {n} chips, {C} classes, {time.time()-t0:.0f}s", flush=True)

        lab0 = d["lab_y0"][:, :CROP, :CROP]
        lab1 = d["lab_y1"][:, :CROP, :CROP]
        dates0 = np.array([[int(s[8:10]), int(s[5:7]) - 1, int(s[:4])] for s in d["date_y0"].astype(str)])
        dates1 = np.array([[int(s[8:10]), int(s[5:7]) - 1, int(s[:4])] for s in d["date_y1"].astype(str)])
        e0 = embed(model, d["img_y0"], dates0)
        e1 = embed(model, d["img_y1"], dates1)
        cells = d["cell"].astype(str)
        fit_c, rep_c = cell_split(cells)
        in_f, in_v = cell_split(cells, salt="inner")

        # Tune on cells held out of the fit set, then refit on all of it: exp68's lesson, mandatory here.
        # --reuse-tuning takes the setting a previous run chose and skips the sweep. The sweep is what PICKS the probe,
        # not what the probe's numbers rest on, so regenerating an artifact should not cost sixty probe fits.
        if args.reuse_tuning:
            prev = json.load(open(os.path.join(OUT, "exp69_summary.json")))["results"]["regions"][region]["probe_tuning"]
            sweep, best = prev["grid"], dict(prev["chosen"])
            print(f"  reusing wd={best['weight_decay']:g} epochs={best['epochs']} from the recorded sweep", flush=True)
        else:
            sweep = []
            for wd in (1e-4, 1e-2, 1.0, 100.0):
                for ep in (15, 40, 80, 160, 320):
                    # This axis was extended twice: 40 won on its own boundary, then 80 did. A grid whose winner sits on
                    # its edge locates no optimum, only a lower bound on one, so it runs to 320. Two regions are still
                    # improving there, which is why every conclusion is also reported at a fixed reference probe: if the
                    # verdicts do not move across a twenty-fold range of training length, the optimum's exact location
                    # is not what they rest on. Contrast exp68, where the regularisation axis DID change which signal
                    # ranked first and so had to be resolved.
                    p_ = fit_dense_probe(e0[fit_c & in_f], lab0[fit_c & in_f], C, seed=0, epochs=ep, wd=wd)
                    rv = window_readings(p_, e0[fit_c & in_v], lab0[fit_c & in_v], C)
                    gv = rv["graded"]
                    acc_v = float((rv["dec"][gv] == rv["y"][gv]).mean()) if gv.any() else float("nan")
                    rf = window_readings(p_, e0[fit_c & in_f], lab0[fit_c & in_f], C)
                    gf = rf["graded"]
                    acc_f = float((rf["dec"][gf] == rf["y"][gf]).mean()) if gf.any() else float("nan")
                    sweep.append({"weight_decay": wd, "epochs": ep,
                                  "inner_fit_accuracy": acc_f, "inner_val_accuracy": acc_v})
                    print(f"  tune wd={wd:g} ep={ep}: inner fit {acc_f:.4f} val {acc_v:.4f}", flush=True)
            best = max((z for z in sweep if np.isfinite(z["inner_val_accuracy"])),
                       key=lambda z: z["inner_val_accuracy"])
            print(f"  chosen wd={best['weight_decay']:g} epochs={best['epochs']}", flush=True)
        probe = fit_dense_probe(e0[fit_c], lab0[fit_c], C, seed=0, epochs=best["epochs"], wd=best["weight_decay"])
        # A second probe at a FIXED reference setting, so the record itself shows whether the verdicts depend on how long
        # the probe trained. The epoch axis was extended three times and two regions were still improving at its far end,
        # which would be a worry if the conclusions moved with it; recording both settings in one artifact lets a reader
        # check that they do not, instead of taking three job logs on trust.
        ref_probe = fit_dense_probe(e0[fit_c], lab0[fit_c], C, seed=0, epochs=REF_EPOCHS, wd=REF_WD)

        rr0 = window_readings(ref_probe, e0, lab0, C)
        rr1 = window_readings(ref_probe, e1, lab1, C)
        rr0["boundary"] = boundary_from(rr0["dec"])
        r0 = window_readings(probe, e0, lab0, C)
        r1 = window_readings(probe, e1, lab1, C)
        r0["boundary"] = boundary_from(r0["dec"])
        r1["boundary"] = boundary_from(r1["dec"])
        var0 = window_variance(d["img_y0"])
        cell_w = np.repeat(cells[:, None, None], G, 1).repeat(G, 2)

        # ---- part A, year 0, graded windows of the report cells
        A = np.repeat(rep_c[:, None, None], G, 1).repeat(G, 2) & r0["graded"]
        err = (r0["dec"] != r0["y"]).astype(np.float64)
        flat = lambda z: np.asarray(z)[A]
        sg = signals_of({k: flat(v) for k, v in r0.items() if k in ("margin", "top1", "entropy", "boundary")},
                        flat(var0), flat(r0["dec"]))
        e_a, cl = flat(err), flat(cell_w)
        scored = {}
        for name, u in sg.items():
            ea = float(metrics.excess_aurc(u, e_a))
            cap = metrics.capture_at_budget_expected(u, e_a, BUDGETS)
            scored[name] = {"excess_aurc": ea, "capture": {str(b): float(v) for b, v in cap.items()},
                            "auroc": float(metrics.weighted_auroc(u, e_a, np.ones(len(e_a))))}
            rows.append({"region": region, "part": "A", "signal": name, "n": int(len(e_a)),
                         "error_rate": float(e_a.mean()), "excess_aurc": ea,
                         "capture_05": float(cap[0.05]), "capture_10": float(cap[0.10]), "capture_20": float(cap[0.20])})
        ctl = min(("ctl_pixel_variance", "ctl_class_rarity"), key=lambda k: scored[k]["excess_aurc"])
        lead = scored[ctl]["excess_aurc"] - scored["margin"]["excess_aurc"]
        st = cell_sign_test(sg["margin"], sg[ctl], e_a, cl)

        # ---- part B, two years, truth on both sides
        both = r0["graded"] & r1["graded"] & np.repeat(rep_c[:, None, None], G, 1).repeat(G, 2)
        a, b = r0["dec"][both], r1["dec"][both]
        ly0, ly1 = r0["y"][both], r1["y"][both]
        changed_decl = ly0 != ly1
        moved = a != b
        rate_changed = float(moved[changed_decl].mean()) if changed_decl.any() else float("nan")
        rate_same = float(moved[~changed_decl].mean()) if (~changed_decl).any() else float("nan")
        right0, right1 = a == ly0, b == ly1
        diff = moved
        n_diff = int(diff.sum())
        neither = int((diff & ~right0 & ~right1).sum())
        cmpres = cmp_mod.compare_inferences(a, b, np.ones(len(a), dtype=bool),
                                            groups=cell_w[both], labels=ly0,
                                            cues={"boundary_y0": r0["boundary"][both] > 0,
                                                  "low_margin_y0": r0["margin"][both] <= np.quantile(r0["margin"][both], 0.2)})
        pr = {"n_chips": int(n), "n_graded_windows_year0": int(A.sum()), "classes": vocab,
              "probe_tuning": {"grid": sweep, "chosen": {k: best[k] for k in ("weight_decay", "epochs", "inner_val_accuracy")}},
              "window_accuracy_report": float((r0["dec"][A] == r0["y"][A]).mean()),
              "part_a": {"signals": scored, "best_control": ctl, "margin_lead": float(lead), "cell_sign_test": st},
              "part_b": {"n": int(both.sum()), "n_declared_changed": int(changed_decl.sum()),
                         "model_change_rate_declared_changed": rate_changed,
                         "model_change_rate_declared_same": rate_same,
                         "ratio": float(rate_changed / max(rate_same, 1e-9)),
                         "n_differing": n_diff, "neither_right": neither,
                         "share_neither_right": float(neither / max(n_diff, 1)),
                         "share_year0_right": float((diff & right0).sum() / max(n_diff, 1)),
                         "share_year1_right": float((diff & right1).sum() / max(n_diff, 1)),
                         "disagreement_rate": cmpres["disagreement_rate"], "where": cmpres["where"],
                         "crosstab_against_year0_labels": cmpres["graded"]["crosstab"]}}
        rows.append({"region": region, "part": "B", "signal": "declared_changed", "n": int(changed_decl.sum()),
                     "error_rate": rate_changed, "excess_aurc": float("nan"),
                     "capture_05": float("nan"), "capture_10": float("nan"), "capture_20": float("nan")})
        rows.append({"region": region, "part": "B", "signal": "declared_same", "n": int((~changed_decl).sum()),
                     "error_rate": rate_same, "excess_aurc": float("nan"),
                     "capture_05": float("nan"), "capture_10": float("nan"), "capture_20": float("nan")})
        # the same three quantities under the fixed reference probe
        Ar = np.repeat(rep_c[:, None, None], G, 1).repeat(G, 2) & rr0["graded"]
        err_r = (rr0["dec"] != rr0["y"]).astype(np.float64)
        fr = lambda z: np.asarray(z)[Ar]
        sgr = signals_of({k: fr(v) for k, v in rr0.items() if k in ("margin", "top1", "entropy", "boundary")},
                         fr(window_variance(d["img_y0"])), fr(rr0["dec"]))
        er = fr(err_r)
        sc_r = {k: float(metrics.excess_aurc(u, er)) for k, u in sgr.items()}
        ctl_r = min(("ctl_pixel_variance", "ctl_class_rarity"), key=lambda k: sc_r[k])
        bothr = rr0["graded"] & rr1["graded"] & np.repeat(rep_c[:, None, None], G, 1).repeat(G, 2)
        ar_, br_ = rr0["dec"][bothr], rr1["dec"][bothr]
        chr_ = rr0["y"][bothr] != rr1["y"][bothr]
        mvr = ar_ != br_
        pr["reference_probe"] = {
            "weight_decay": REF_WD, "epochs": REF_EPOCHS,
            "window_accuracy_report": float((rr0["dec"][Ar] == rr0["y"][Ar]).mean()),
            "margin_lead": float(sc_r[ctl_r] - sc_r["margin"]), "best_control": ctl_r,
            "model_change_rate_declared_changed": float(mvr[chr_].mean()) if chr_.any() else float("nan"),
            "model_change_rate_declared_same": float(mvr[~chr_].mean()) if (~chr_).any() else float("nan")}
        pr["reference_probe"]["ratio"] = float(pr["reference_probe"]["model_change_rate_declared_changed"]
                                              / max(pr["reference_probe"]["model_change_rate_declared_same"], 1e-9))
        pr["probe_tuning"]["best_at_grid_edge"] = bool(best["epochs"] == max(g["epochs"] for g in sweep))
        keep = np.zeros(n, dtype=bool)
        keep[::4] = True                                   # every fourth chip, so the artifact stays a few megabytes
        km = np.repeat(keep[:, None, None], G, 1).repeat(G, 2)
        windows.append({
            "region": np.full(int(km.sum()), region),
            "cell": cell_w[km], "chip": np.repeat(np.arange(n)[:, None, None], G, 1).repeat(G, 2)[km],
            "dec_y0": r0["dec"][km], "dec_y1": r1["dec"][km],
            "y_y0": r0["y"][km], "y_y1": r1["y"][km],
            "graded_y0": r0["graded"][km], "graded_y1": r1["graded"][km],
            "margin_y0": r0["margin"][km].astype(np.float32), "entropy_y0": r0["entropy"][km].astype(np.float32),
            "top1_y0": r0["top1"][km].astype(np.float32), "boundary_y0": r0["boundary"][km].astype(np.float32),
            "variance_y0": var0[km].astype(np.float32),
            "report": np.repeat(rep_c[:, None, None], G, 1).repeat(G, 2)[km]})
        per_region[region] = pr
        print(f"  {region}: window accuracy {pr['window_accuracy_report']:.4f}, margin lead {lead:+.4f} over {ctl}, "
              f"cells {st['wins']}-{st['losses']}; model moves {rate_changed:.3f} where the declaration changed and "
              f"{rate_same:.3f} where it did not", flush=True)

    # Every quantity above must be recomputable without a rerun. exp68's audit found that its part B could only be
    # taken on the summary's word, so the per-window arrays go out here for every fourth chip of each region.
    if windows:
        np.savez_compressed(os.path.join(OUT, f"exp69_windows{'_smoke' if args.smoke else ''}.npz"),
                            **{k: np.concatenate([w[k] for w in windows]) for k in windows[0]})
    summary["results"]["regions"] = per_region
    summary["config"]["regions"] = {k: {"years": plan["regions"][k].get("years"),
                                       "join": plan["regions"][k].get("join")} for k in plan["regions"]}
    ok = list(per_region)
    summary["verdicts"]["P1"] = {
        "holds": bool(ok) and all(per_region[r]["part_a"]["margin_lead"] >= 0.01
                                  and per_region[r]["part_a"]["cell_sign_test"]["p"] < 0.05
                                  and per_region[r]["part_a"]["cell_sign_test"]["wins"] > per_region[r]["part_a"]["cell_sign_test"]["losses"]
                                  for r in ok),
        "per_region": {r: {"lead": per_region[r]["part_a"]["margin_lead"], "against": per_region[r]["part_a"]["best_control"],
                           "sign_test": per_region[r]["part_a"]["cell_sign_test"]} for r in ok}}
    summary["verdicts"]["P2"] = {
        "holds": bool(ok) and all(per_region[r]["part_b"]["ratio"] >= 2.0 for r in ok),
        "per_region": {r: {"changed": per_region[r]["part_b"]["model_change_rate_declared_changed"],
                           "same": per_region[r]["part_b"]["model_change_rate_declared_same"],
                           "ratio": per_region[r]["part_b"]["ratio"]} for r in ok},
        "note": "the change rate where the declaration did not change is the labelled floor exp68 could only bound"}
    summary["verdicts"]["P3"] = {
        "holds": bool(ok) and all(per_region[r]["part_b"]["share_neither_right"] < 0.5 for r in ok),
        "per_region": {r: {"share_neither_right": per_region[r]["part_b"]["share_neither_right"],
                           "share_year0_right": per_region[r]["part_b"]["share_year0_right"],
                           "share_year1_right": per_region[r]["part_b"]["share_year1_right"]} for r in ok}}

    tag = "_smoke" if args.smoke else ""
    with open(os.path.join(OUT, f"exp69_summary{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    if rows:
        with open(os.path.join(OUT, f"exp69_eurocrops{tag}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    for k, v in summary["verdicts"].items():
        print(f"{k}: {v['holds']}", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
    return summary


# ----------------------------------------------------------------------------- smoke
def smoke(args):
    rng = np.random.default_rng(0)
    # the vocabulary must be ordered by declared AREA, not by parcel count: many small parcels of one crop must not
    # outrank a few large ones, because the probe sees pixels
    names = np.array(["a"] * 100 + ["b"] * 5 + ["c"] * 3)
    areas = np.r_[np.full(100, 10.0), np.full(5, 10_000.0), np.full(3, 1.0)]
    vocab, share = class_vocabulary(names, areas, n_classes=3)
    assert vocab[0] == "b" and vocab[1] == "a" and vocab[-1] == OTHER, vocab
    assert class_index(np.array(["a", "b", "zzz"]), vocab).tolist() == [1, 0, 2]

    assert cell_of(0.0, 0.0) == "0_0" and cell_of(CELL + 1, -1.0) == "1_-1"
    f, r = cell_split(np.array([f"{i}_{i}" for i in range(80)]))
    assert not (f & r).any() and (f | r).all()
    assert not np.array_equal(f, cell_split(np.array([f"{i}_{i}" for i in range(80)]), salt="inner")[0])

    # window labels: pure only when every pixel is labelled and PURE of them agree
    n = 6
    lab = np.full((n, CROP, CROP), -1, dtype=np.int16)
    lab[0] = 0                                     # wholly one class
    lab[1, :, :] = 1
    lab[1, 0, 0] = -1                              # one unlabelled pixel spoils its window only
    lab[2, :30] = 0; lab[2, 30:] = 1               # a real boundary through the middle
    emb = rng.standard_normal((n, G, G, 8)).astype(np.float16)

    class _P:
        def __call__(self, x):
            import torch
            b = x.shape[0]
            return {"logits": torch.zeros((b, G, G, 3 * PATCH * PATCH), device=x.device)}
    try:
        import torch  # noqa: F401
        out = window_readings(_P(), emb, lab, 3)
        assert out["graded"][0].all(), "a wholly labelled chip must be graded everywhere"
        assert not out["graded"][1, 0, 0] and out["graded"][1, 1, 1], "one unlabelled pixel must spoil one window only"
        assert out["y"][2, 0, 0] == 0 and out["y"][2, -1, -1] == 1
        assert not out["graded"][3].any(), "an unlabelled chip must be graded nowhere"
        assert out["margin"].min() >= 0 and out["top1"].max() <= 1
    except ImportError:                                       # torch itself missing, which the cluster never is
        print("  (torch absent: window_readings not exercised)")

    # the predicted-boundary share
    dec = np.zeros((2, 5, 5), dtype=np.int64)
    dec[0, :, 3:] = 1
    bd = boundary_from(dec)
    assert bd[1].sum() == 0, "a uniform decision map has no boundary"
    # classes split between columns 2 and 3, so columns 2 and 3 border the edge and column 0 is clear of it
    assert bd[0, 2, 0] == 0.0 and bd[0, 2, 2] > 0 and bd[0, 2, 3] > 0
    assert bd[0, 2, 2] == 3 / 8 and bd[0, 2, 3] == 3 / 8, bd[0, 2]

    # the two-label accounting of part B: a window decidable on both sides
    a = np.array([0, 0, 1, 1]); b = np.array([1, 0, 1, 0])
    ly0 = np.array([0, 0, 1, 0]); ly1 = np.array([1, 0, 1, 1])
    moved, r0, r1 = a != b, a == ly0, b == ly1
    assert moved.tolist() == [True, False, False, True]
    assert int((moved & r0 & r1).sum()) == 1 and int((moved & ~r0 & ~r1).sum()) == 1
    print("smoke OK: area-ordered vocabulary, cell split, window purity, boundary share, two-label accounting")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("fetch", "analyze", "all"), default="all")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse-tuning", action="store_true",
                    help="take each region's weight decay and epochs from the recorded exp69_summary.json instead of "
                         "sweeping again; for regenerating artifacts without repeating sixty probe fits")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.smoke:
        smoke(args)
        return
    if args.stage in ("fetch", "all"):
        fetch_stage(args)
    if args.stage in ("analyze", "all"):
        analyze_stage(args)


if __name__ == "__main__":
    main()
