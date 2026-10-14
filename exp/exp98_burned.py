"""exp98, the burned-area stage: a condition layer that says where MODIS mapped a burn in 2023, on the grid of each
part of the re-run map (exp/jobs/e98_burned.sh runs it after exp/jobs/e98_read.sh).

Why. East African savanna burns every year, and a fresh burn scar may be mapped confidently wrong. The AWF legend has
one class for grassland and barren, so burnt grass read as bare is no error; burnt shrubland or savanna read as
grassland/barren, or a dark scar read as a dark class (open water, herbaceous wetland, lava forest), would be. With
this layer `oe-inferencex assess --condition` ranks burned and unburned windows apart, and exp98's Part H
(report-only, docs/plan/awf_deployment.md) describes the map's confidence and Ai2's validation points by it.

Source, as verified on 2026-10-08 from the STAC API's metadata and the product's documents (no pixel was read off
the cluster):
  - Microsoft Planetary Computer's STAC API (https://planetarycomputer.microsoft.com/api/stac/v1), collection
    `modis-64A1-061`, "MODIS Burned Area Monthly", MCD64A1 Collection 6.1. One item per MODIS sinusoidal tile
    (2400 x 2400 cells of 463.3 m) and calendar month: start_datetime is the month's first day, end_datetime its last.
    Assets: `Burn_Date` (int16 COG), `QA` (uint8 COG), `Burn_Date_Uncertainty`, `First_Day`, `Last_Day`, `hdf`,
    `metadata`. The hrefs are on Azure blob storage and are signed with planetary_computer.sign_inplace, as
    oe_inferencex/data.py signs its Sentinel-2 and WorldCover reads. This file does not import data.py, which needs
    the encoder extra.
  - Burn_Date (MCD64 Collection 6.1 User Guide, section 3.1.2): the ordinal day of the year on which the cell burned,
    1 to 366; 0 unburned land; -1 unmapped for lack of valid data; -2 water. The file name's day of year is the first
    day of the calendar month the file maps (Table 2); a burn date outside that month is counted, not expected.
  - QA, an 8-bit field (same section): bit 0 land (1) or water (0); bit 1 valid data (1: enough valid data in the
    time series to process the cell; always 0 for water); bit 2 shortened mapping period (burns could not be mapped
    over the whole month); bit 3 relabelled in the contextual phase; bit 4 spare; bits 5-7 a special-condition code for
    a cell classified unburned (0 none, 1 observations too sparse in time, 2 too few training observations or poor
    separability, 3 apparent burn date at the limits of the time series, 4 apparent water contamination, 5 persistent
    hot spot). This stage checks Burn_Date's special values against bits 0 and 1 on every cell it reads and refuses
    a disagreement on more than QA_TOLERANCE of them, so the encoding above is tested on the real files.
  - Accuracy (guide section 7): against Landsat 8 pairs, global omission 72.6% and commission 40.2%, the omission
    mostly small burns; cropland burns are low confidence (section 8.1).
  - Licence: the STAC record says "proprietary" and links LP DAAC's data citation and policies page, which now
    redirects to NASA Earthdata's data use guidance: data from a NASA-led mission are Creative Commons Zero (CC0)
    unless marked otherwise, a citation is requested, and NASA material may not suggest NASA's endorsement. Cite
    Giglio, L., Justice, C., Boschetti, L., Roy, D. (2021), MCD64A1 v061, doi:10.5067/MODIS/MCD64A1.061.
  - September 2023 is not on Planetary Computer. On 2026-10-08 the collection held 268 items for August 2023 and 268
    for October 2023, and none for September 2023 anywhere on Earth, while NASA's CMR listed 268 September 2023
    granules, the tile over this area among them. A missing month is refused unless --allow-missing names it; the job
    names 2023-09. A cell that burned in a month not read reads as not burned, so code 0's name says which months
    were not read, and every record lists them. --item supplies a month from local GeoTIFFs of Burn_Date and QA (for
    instance converted from LP DAAC's granule, which needs an Earthdata login).

The codes, per 500 m MODIS cell over the months read, then per 10 m pixel by nearest neighbour:
     1  burned: a burn date (1 to 366) in any month read
     0  not burned: unburned land (0) in every month read
    -1  unrecorded: no burn date and, in at least one month read, water (-2), unmapped (-1), or no tile over the cell
A code 2, "burned before a Sentinel-2 timestep the model read", is not written. The model reads 12 mosaics of 30-day
periods, each built per pixel from several scenes ordered by cloud cover, so the date of the scene at a pixel is not
in the item groups' time ranges, and items.json (the scenes' footprints) stays on scratch; and every burn before the
last period is followed by later periods, so the code would hold nearly every burned pixel and separate nothing.

Resolution. A MODIS cell is about 463 m on a side: about 2,150 pixels of 10 m and about 134 windows of 40 m (assess's
default 4-pixel window). The layer says whether the 500 m cell around a pixel burned, not whether the pixel did:
along a burn scar's edge, unburned pixels read 1 and burned ones 0. Nearest neighbour gives each 10 m pixel the code
of the MODIS cell that holds its centre; the warp transforms every pixel exactly (tolerance 0). rasterio's default
tolerance allows 0.125 input pixels, up to 58 m here, and on a synthetic 5 km grid it gave 0.1% of the pixels a
neighbouring cell's code (tests/test_exp98_burned.py checks every pixel against its own transform).

Outputs, for each part directory (one from-olmoearth output, exp/jobs/e98_read.sh):
    burned_<EPSG>.tif          int32 on the grid of scores_<EPSG>.tif, the codes above, -1 also the raster's no-data
    burned_conditions.json     the codes' names, the exact --condition-names string, the rule, the source, the months
and, over all parts, a counts-only record (--out-json): per part and grid the covered pixels (the scores' ten bands
finite and not all zero, exp98's rule), the share of them per code, why the unrecorded ones are unrecorded, the share
burned in each month read, and the QA checks. No coordinate, bound, transform, tile or item id is written or printed.

    python exp/exp98_burned.py --scores data/exp98/scores --allow-missing 2023-09     # the job's call
    python exp/exp98_burned.py --scores DIR --item 2023-09 BURN_DATE.tif QA.tif   # September from a local file
"""
import argparse
import calendar
import datetime as dt
import glob
import json
import os
import re
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
OUT_JSON = os.path.join(EXP_DIR, "out", "exp98", "e98_burned.json")

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"      # as oe_inferencex/data.py
COLLECTION = "modis-64A1-061"
BURN_ASSET, QA_ASSET = "Burn_Date", "QA"
CITATION = ("Giglio, L., Justice, C., Boschetti, L., Roy, D. (2021). MODIS/Terra+Aqua Burned Area Monthly L3 Global "
            "500m SIN Grid V061. NASA EOSDIS Land Processes DAAC. doi:10.5067/MODIS/MCD64A1.061")
LICENCE = ("NASA Earthdata data use guidance (the target of the collection's licence link): data from a NASA-led "
           "mission are CC0 unless marked otherwise; a citation is requested; no implied NASA endorsement")
YEAR = 2023                         # exp98's request period, 2023-01-01 to 2023-12-31
ITEM_ID = re.compile(r"^MCD64A1\.A(\d{4})(\d{3})\.h(\d{2})v(\d{2})\.061\.(\d{13})$")

# Burn_Date's values (guide 3.1.2) and the in-memory value for a cell no tile covers, never a Burn_Date value
UNBURNED, UNMAPPED, WATER = 0, -1, -2
FILL = -32768
# the codes written
CODE_BURNED, CODE_NOT, CODE_NONE = 1, 0, -1
# per-cell flags carried to the pixels beside the code (counted in the record)
FLAG_SPECIAL = 1        # unburned land with a QA special-condition code (bits 5-7) in a month read
FLAG_SHORT = 2          # a shortened mapping period (QA bit 2) in a month read
FLAG_WATER = 4          # water (-2) in a month read
FLAG_UNMAPPED = 8       # unmapped (-1) in a month read
FLAG_NOTILE = 16        # no tile over the cell in a month read
FLAG_OFF_MONTH = 32     # a burn date outside its file's calendar month
QA_TOLERANCE = 1e-3     # refused above this share of disagreeing cells
MARGIN_CELLS = 2        # MODIS cells read beyond the part's footprint on each side
STRIP_ROWS = 256
N_BANDS = 10
NAMES_JSON = "burned_conditions.json"

# keys that would carry a position (exp98_awf_deployment.FORBIDDEN_KEYS), an href, or a tile or item id
FORBIDDEN_KEYS = {"transform", "bounds", "bbox", "lon", "lat", "lonlat", "longitude", "latitude", "x", "y", "xy",
                  "coordinates", "geometry", "origin", "centre", "center", "row", "col", "rows", "cols", "href",
                  "id", "item_id", "item_ids", "tile", "tile_id", "tile_ids"}

RULE = ("Per 500 m MODIS cell, over the months of MCD64A1 v061 read for the year: 1 = a burn date (1-366) in any month "
        "read; 0 = unburned land (0) in every month read; -1 = no burn date and, in at least one month read, water "
        "(-2), unmapped for lack of data (-1), or no tile over the cell. Each 10 m pixel takes the code of the MODIS "
        "cell that holds its centre (nearest neighbour, every pixel transformed exactly). A MODIS cell is about 463 m "
        "on a side, about 2,150 pixels of 10 m and 134 windows of 40 m, so the code says whether the cell around a "
        "pixel burned, not whether the pixel did.")
CODE2_NOTE = ("No code 2 (burned before a Sentinel-2 timestep the model read): each of the model's 12 timesteps is a "
              "mosaic of a 30-day period built per pixel from several scenes ordered by cloud cover, so the scene "
              "date at a pixel is not in the item groups' time ranges; and every burn before the last period is "
              "followed by later periods, so the code would hold nearly every burned pixel.")


# ----------------------------------------------------------------------------- small helpers
def check_no_coordinates(obj, path=""):
    """Refuse to write an object that carries a key naming a position, an href or a tile."""
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
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def git_commit():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def month_key(year, month):
    return f"{year:04d}-{month:02d}"


def month_doys(year, month):
    """(first, last) day of the year of a calendar month."""
    first = dt.date(year, month, 1).timetuple().tm_yday
    return first, first + calendar.monthrange(year, month)[1] - 1


def month_abbrev(key):
    return calendar.month_abbr[int(key[5:7])].lower()


def code_names(year, months_missing):
    """{code: name}. Code 0's name says which months were not read, so the caveat travels with every output that
    names the condition."""
    zero = f"mcd64a1:no-burn-{year}"
    if months_missing:
        zero += ":" + "+".join(month_abbrev(m) for m in months_missing) + "-not-read"
    return {CODE_NOT: zero, CODE_BURNED: f"mcd64a1:burned-{year}"}


def condition_names_arg(names):
    """The exact `--condition-names ...` string for oe-inferencex assess and sample."""
    return " ".join(["--condition-names"] + [shlex.quote(f"{c}={n}") for c, n in sorted(names.items())])


# ----------------------------------------------------------------------------- the items
def select_items(features, year=YEAR):
    """{month key: [{"key": tile key, "burn": href, "qa": href, "production": stamp}]} from STAC item dicts of the
    collection, and counts. An item is kept when its id parses as MCD64A1 v061 of `year`, its day of year is the first
    day of its start_datetime's month (guide Table 2) and it carries both assets. Two items of one tile and month keep
    the later production stamp (a reprocessed file; the guide's naming note), and are counted."""
    by, notes = {}, {"items_seen": 0, "items_kept": 0, "duplicates_dropped": 0, "skipped": {}}

    def skip(why):
        notes["skipped"][why] = notes["skipped"].get(why, 0) + 1

    for f in features:
        notes["items_seen"] += 1
        m = ITEM_ID.match(str(f.get("id", "")))
        if not m:
            skip("id not MCD64A1 v061")
            continue
        y, doy, h, v, prod = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4), m.group(5)
        if y != year:
            skip("another year")
            continue
        start = str((f.get("properties") or {}).get("start_datetime") or "")
        if not re.match(rf"^{year:04d}-\d\d-01", start):
            skip("start_datetime not the first day of a month of the year")
            continue
        month = int(start[5:7])
        if month_doys(year, month)[0] != doy:
            skip("day of year in the id is not its month's first day")
            continue
        assets = f.get("assets") or {}
        if BURN_ASSET not in assets or QA_ASSET not in assets:
            skip("Burn_Date or QA asset missing")
            continue
        key = month_key(year, month)
        entry = {"key": f"{h}{v}", "burn": assets[BURN_ASSET]["href"], "qa": assets[QA_ASSET]["href"],
                 "production": prod}
        lst = by.setdefault(key, [])
        same = [e for e in lst if e["key"] == entry["key"]]
        if same:
            notes["duplicates_dropped"] += 1
            if same[0]["production"] >= prod:
                continue
            lst.remove(same[0])
        lst.append(entry)
    notes["items_kept"] = sum(len(v) for v in by.values())
    return by, notes


def search_stac(bbox_wgs84, year=YEAR):
    """The collection's items of `year` over a WGS84 box, as signed dicts. The box stays in memory."""
    import planetary_computer
    import pystac_client
    last = None
    for attempt in range(3):
        try:
            cat = pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)
            search = cat.search(collections=[COLLECTION], bbox=list(bbox_wgs84),
                                datetime=f"{year:04d}-01-01T00:00:00Z/{year:04d}-12-31T23:59:59Z")
            return list(search.items_as_dicts())
        except Exception as ex:  # noqa: BLE001  (network; retried, then refused without the request in the message)
            last = ex
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"the STAC search of {COLLECTION} failed three times ({type(last).__name__})") from None


def items_from_files(specs, year=YEAR):
    """{month key: [{"key", "burn", "qa"}]} from --item MONTH BURN_TIF QA_TIF triples (local GeoTIFFs of one tile each;
    two triples of one month are two tiles)."""
    by = {}
    for k, (month, burn, qa) in enumerate(specs):
        if not re.fullmatch(rf"{year:04d}-(0[1-9]|1[0-2])", month):
            raise ValueError(f"--item month {month!r}: give YYYY-MM within {year}")
        for p in (burn, qa):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"--item {month}: no file {p}")
        by.setdefault(month, []).append({"key": f"file{k}", "burn": burn, "qa": qa, "production": "file"})
    return by


# ----------------------------------------------------------------------------- reading the tiles
def _open_retry(href, what):
    import rasterio
    last = None
    for attempt in range(3):
        try:
            return rasterio.open(href)
        except Exception as ex:  # noqa: BLE001
            last = ex
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"opening {what} failed three times ({type(last).__name__}); the href is not printed") from None


def _read_retry(src, window, what):
    last = None
    for attempt in range(3):
        try:
            return src.read(1, window=window)
        except Exception as ex:  # noqa: BLE001
            last = ex
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"reading {what} failed three times ({type(last).__name__}); the href is not printed") from None


def mosaic_window(lattice, footprint, margin=MARGIN_CELLS):
    """(row0, col0, height, width) on a tile lattice (a, e, c, f: cell sizes and an origin) covering a footprint
    (left, bottom, right, top in the lattice's CRS), with `margin` cells more on every side."""
    a, e, c, f = lattice
    left, bottom, right, top = footprint
    c0 = int(np.floor((left - c) / a)) - margin
    c1 = int(np.ceil((right - c) / a)) + margin
    r0 = int(np.floor((top - f) / e)) - margin                 # e < 0: the top is the smallest row
    r1 = int(np.ceil((bottom - f) / e)) + margin
    return r0, c0, r1 - r0, c1 - c0


def read_month_mosaic(items, lattice, crs, win, what):
    """Burn_Date and QA of one month on the mosaic window: (burn int16, FILL where no tile; qa uint8; the tiles' offsets
    on the lattice, to count distinct tiles). Every tile must lie on the lattice in the same CRS; a tile is read only
    where it meets the window."""
    from rasterio.windows import Window
    r0, c0, h, w = win
    a, e, c, f = lattice
    burn = np.full((h, w), FILL, np.int16)
    qa = np.zeros((h, w), np.uint8)
    offsets = set()
    for k, it in enumerate(items):
        for asset, out in ((BURN_ASSET, burn), (QA_ASSET, qa)):
            src = _open_retry(it["burn" if asset == BURN_ASSET else "qa"], f"{asset} of {what}, tile {k + 1}")
            with src:
                t = src.transform
                if src.crs != crs:
                    raise ValueError(f"{asset} of {what}, tile {k + 1}: its CRS differs from the first tile's")
                if t.b or t.d or not np.isclose(t.a, a, rtol=0, atol=1e-6) or not np.isclose(t.e, e, rtol=0, atol=1e-6):
                    raise ValueError(f"{asset} of {what}, tile {k + 1}: not on the first tile's lattice (cell size)")
                oc, orow = (t.c - c) / a, (t.f - f) / e
                if abs(oc - round(oc)) > 1e-3 or abs(orow - round(orow)) > 1e-3:
                    raise ValueError(f"{asset} of {what}, tile {k + 1}: not on the first tile's lattice (offset)")
                oc, orow = int(round(oc)), int(round(orow))
                if asset == BURN_ASSET and src.dtypes[0] != "int16":
                    raise ValueError(f"Burn_Date of {what}, tile {k + 1}, is {src.dtypes[0]}, not the documented int16")
                # the tile's cells in mosaic coordinates
                ra, rb = max(r0, orow), min(r0 + h, orow + src.height)
                ca, cb = max(c0, oc), min(c0 + w, oc + src.width)
                if ra >= rb or ca >= cb:
                    continue
                v = _read_retry(src, Window(ca - oc, ra - orow, cb - ca, rb - ra), f"{asset} of {what}, tile {k + 1}")
                out[ra - r0:rb - r0, ca - c0:cb - c0] = v
                offsets.add((orow, oc))
    return burn, qa, offsets


# ----------------------------------------------------------------------------- the codes
def combine(burns, qas, doy_ranges):
    """The codes of the cells from the months read: burns and qas are lists of equal-shape arrays (Burn_Date int16
    with FILL where no tile covers a cell, QA uint8), doy_ranges the (first, last) day of each month. Returns
    (code int16, burn-month bits int16 (bit k: a burn in the k-th month given), flags int16, checks). Refused: a
    Burn_Date value outside the documented encoding, and QA disagreeing with Burn_Date's special values on more than
    QA_TOLERANCE of the cells read."""
    shape = burns[0].shape
    burned = np.zeros(shape, bool)
    bits = np.zeros(shape, np.int16)
    flags = np.zeros(shape, np.int16)
    tile_any = np.zeros(shape, bool)
    ck = {"cells_read": 0, "water_qa_disagree": 0, "unmapped_qa_disagree": 0, "mapped_without_valid_bit": 0,
          "burn_dates_outside_month": 0, "cells_per_month": {}}
    for k, (b, q, (d0, d1)) in enumerate(zip(burns, qas, doy_ranges)):
        b = np.asarray(b)
        q = np.asarray(q).astype(np.uint8)
        tile = b != FILL
        bad = tile & ~(((b >= WATER) & (b <= 366)))
        if bad.any():
            raise ValueError(f"Burn_Date holds {int(bad.sum())} cells with values outside -2..366 (e.g. "
                             f"{int(b[bad][0])}): not the documented encoding; nothing is written")
        is_burn = tile & (b >= 1)
        is_water, is_unmapped = tile & (b == WATER), tile & (b == UNMAPPED)
        land_qa, valid_qa = (q & 1) == 1, (q & 2) == 2
        ck["cells_read"] += int(tile.sum())
        ck["water_qa_disagree"] += int((tile & (is_water != ~land_qa)).sum())
        ck["unmapped_qa_disagree"] += int((is_unmapped & valid_qa).sum())
        ck["mapped_without_valid_bit"] += int((tile & (b >= 0) & ~valid_qa).sum())
        off = is_burn & ((b < d0) | (b > d1))
        ck["burn_dates_outside_month"] += int(off.sum())
        burned |= is_burn
        bits |= np.where(is_burn, np.int16(1 << k), np.int16(0))
        flags |= np.where(tile & (b == UNBURNED) & (((q >> 5) & 7) != 0), FLAG_SPECIAL, 0).astype(np.int16)
        flags |= np.where(tile & ((q & 4) == 4), FLAG_SHORT, 0).astype(np.int16)
        flags |= np.where(is_water, FLAG_WATER, 0).astype(np.int16)
        flags |= np.where(is_unmapped, FLAG_UNMAPPED, 0).astype(np.int16)
        flags |= np.where(~tile, FLAG_NOTILE, 0).astype(np.int16)
        flags |= np.where(off, FLAG_OFF_MONTH, 0).astype(np.int16)
        tile_any |= tile
        ck["cells_per_month"][str(k)] = int(tile.sum())
    disagree = ck["water_qa_disagree"] + ck["unmapped_qa_disagree"]
    ck["qa_agrees"] = bool(disagree <= QA_TOLERANCE * max(ck["cells_read"], 1))
    if not ck["qa_agrees"]:
        raise ValueError(f"QA disagrees with Burn_Date's special values on {disagree} of {ck['cells_read']} cells read "
                         f"(water: Burn_Date -2 against QA bit 0 clear; unmapped: Burn_Date -1 with QA bit 1 set), "
                         f"more than {QA_TOLERANCE:g}: the encoding is not the documented one; nothing is written")
    none = (flags & (FLAG_WATER | FLAG_UNMAPPED | FLAG_NOTILE)) != 0
    code = np.where(burned, CODE_BURNED, np.where(none, CODE_NONE, CODE_NOT)).astype(np.int16)
    never = ~tile_any                                           # no tile in any month read: no data at all
    code[never], bits[never], flags[never] = FILL, FILL, FILL
    return code, bits, flags, ck


def to_grid(bands, src_transform, src_crs, dst_transform, dst_crs, shape, threads=4, rows_per_strip=512):
    """(k, h, w) int16 cells onto a (k, H, W) grid, each destination pixel taking the cell that holds its centre.
    Every pixel centre is transformed on its own with pyproj and the cell found by flooring, strip by strip. GDAL's
    warper is not used: with tolerance 0 it still put one pixel of the 43,200 in tests/test_exp98_burned.py in the
    next cell under rasterio 1.4.4 (GDAL 3.10.3) and none under rasterio 1.5.1 (GDAL 3.12.4), so its answer depended
    on the installed version (CI, Python 3.11, after f934f73). FILL in and out: no cell there. `threads` is kept for
    the callers and unused."""
    from pyproj import CRS, Transformer
    src = np.ascontiguousarray(bands, dtype=np.int16)
    k, h, w = src.shape
    H, W = (int(shape[0]), int(shape[1]))
    dst = np.full((k, H, W), FILL, np.int16)
    to_src = Transformer.from_crs(CRS.from_user_input(dst_crs), CRS.from_user_input(src_crs), always_xy=True)
    inv = ~src_transform
    cols = np.arange(W) + 0.5
    for r0 in range(0, H, rows_per_strip):
        rows = np.arange(r0, min(H, r0 + rows_per_strip)) + 0.5
        cc, rr = np.meshgrid(cols, rows)
        xs, ys = dst_transform * (cc, rr)
        sx, sy = to_src.transform(xs, ys)
        fc, fr = inv * (np.asarray(sx), np.asarray(sy))
        ci, ri = np.floor(fc).astype(np.int64), np.floor(fr).astype(np.int64)
        ok = np.isfinite(fc) & np.isfinite(fr) & (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h)
        block = np.full((k,) + ci.shape, FILL, np.int16)
        block[:, ok] = src[:, ri[ok], ci[ok]]
        dst[:, r0:r0 + ci.shape[0]] = block
    return dst


def covered_mask(scores_path):
    """exp98's covered rule on a scores raster, strip by strip: all ten bands finite and not all zero."""
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(scores_path) as src:
        if src.count != N_BANDS:
            raise ValueError(f"{os.path.basename(scores_path)} holds {src.count} bands, not {N_BANDS}")
        H, W = src.height, src.width
        cov = np.zeros((H, W), bool)
        for r0 in range(0, H, STRIP_ROWS):
            h = min(STRIP_ROWS, H - r0)
            a = src.read(window=Window(0, r0, W, h))
            cov[r0:r0 + h] = np.isfinite(a).all(0) & (np.nan_to_num(a) != 0).any(0)
    return cov


# ----------------------------------------------------------------------------- one grid
def burned_for_grid(scores_path, month_items, months, threads=4, log=print):
    """The burned codes on the grid of one scores raster. `month_items` is {month key: [items]} for the months to read
    (those absent from it are not read); `months` the month keys in order. Returns (code int32 (H, W), bits, flags,
    checks, covered mask)."""
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(scores_path) as src:
        crs, transform, H, W = src.crs, src.transform, src.height, src.width
        bounds = src.bounds
    first = next((it for m in months for it in month_items.get(m, [])), None)
    if first is None:
        raise ValueError("no month to read: every month of the year is missing")
    with _open_retry(first["burn"], "the first tile") as t0:
        tcrs, tt = t0.crs, t0.transform
    if tcrs is None or not tcrs.is_projected or tt.b or tt.d:
        raise ValueError("the first Burn_Date tile has no projected CRS or is rotated; the MODIS sinusoidal grid is "
                         "expected (the STAC record's proj:wkt2)")
    lattice = (tt.a, tt.e, tt.c, tt.f)
    footprint = transform_bounds(crs, tcrs, *bounds, densify_pts=21)
    win = mosaic_window(lattice, footprint)
    if win[2] * win[3] > 4000 * 4000:
        raise ValueError(f"the part's footprint spans {win[2]} x {win[3]} MODIS cells; a part is a few tens of km")
    read = [m for m in months if month_items.get(m)]
    burns, qas, ranges, tiles = [], [], [], set()
    for m in read:
        b, q, offs = read_month_mosaic(month_items[m], lattice, tcrs, win, m)
        tiles |= offs
        burns.append(b)
        qas.append(q)
        ranges.append(month_doys(int(m[:4]), int(m[5:7])))
    code5, bits5, flags5, ck = combine(burns, qas, ranges)
    ck["months_read"] = read
    ck["cells_per_month"] = {read[int(k)]: v for k, v in ck["cells_per_month"].items()}
    ck["tiles_read"] = len(tiles)                               # distinct tiles that met the part, a count only
    r0, c0, h, w = win
    a, e, c, f = lattice
    from rasterio.transform import Affine
    mt = Affine(a, 0.0, c + c0 * a, 0.0, e, f + r0 * e)
    g = to_grid(np.stack([code5, bits5, flags5]), mt, tcrs, transform, crs, (H, W), threads)
    nodata = g[0] == FILL
    code = np.where(nodata, CODE_NONE, g[0]).astype(np.int32)
    bits = np.where(nodata, 0, g[1]).astype(np.int16)
    flags = np.where(nodata, FLAG_NOTILE, g[2]).astype(np.int16)
    cov = covered_mask(scores_path)
    log(f"   {H * W} pixels, {int(cov.sum())} covered; {len(read)} months read, {ck['tiles_read']} tile(s); QA agrees: "
        f"{ck['qa_agrees']}")
    return code, bits, flags, ck, cov


def grid_counts(code, bits, flags, cov, read, missing=()):
    """Counts only: shares of the covered pixels per code, why the unrecorded ones are, the share burned in each month
    read (bit k of `bits` is the k-th month of `read`), None for a month not read."""
    n = int(cov.sum())
    c = code[cov]
    fl = flags[cov]
    bt = bits[cov].astype(np.int64)
    by = {str(k): int((c == k).sum()) for k in (CODE_NONE, CODE_NOT, CODE_BURNED)}
    out = {"pixels_covered": n, "pixels_covered_by_code": by,
           "share_of_covered_by_code": {k: (v / n if n else None) for k, v in by.items()}}
    none = c == CODE_NONE
    out["unrecorded_why"] = {"no_tile_in_a_month_read": int((none & ((fl & FLAG_NOTILE) != 0)).sum()),
                             "water_in_a_month_read": int((none & ((fl & FLAG_WATER) != 0)).sum()),
                             "unmapped_in_a_month_read": int((none & ((fl & FLAG_UNMAPPED) != 0)).sum()),
                             "note": "pixels of code -1 with each reason; a pixel can have more than one"}
    notb = c == CODE_NOT
    out["not_burned_with_special_condition_share"] = (float(((fl & FLAG_SPECIAL) != 0)[notb].mean())
                                                      if notb.any() else None)
    out["shortened_mapping_period_share"] = float(((fl & FLAG_SHORT) != 0).mean()) if n else None
    out["burn_date_outside_its_month_pixels"] = int(((fl & FLAG_OFF_MONTH) != 0).sum())
    share = {m: (float(((bt >> k) & 1).mean()) if n else None) for k, m in enumerate(read)}
    share.update({m: None for m in missing})
    out["burned_share_by_month"] = dict(sorted(share.items()))
    nb = sum(((bt >> k) & 1) for k in range(len(read))) if len(read) else np.zeros(0)
    out["burned_in_two_or_more_months_share"] = float((np.asarray(nb) >= 2).mean()) if n else None
    return out


def write_burned(path, code, scores_path):
    """burned_<EPSG>.tif: int32 codes on the scores raster's grid, -1 the no-data."""
    import rasterio
    with rasterio.open(scores_path) as src:
        prof = dict(driver="GTiff", height=src.height, width=src.width, count=1, dtype="int32", crs=src.crs,
                    transform=src.transform, nodata=CODE_NONE, compress="deflate", tiled=True, blockxsize=256,
                    blockysize=256)
    if code.shape[0] < 256 or code.shape[1] < 256:
        prof.update(tiled=False)
        prof.pop("blockxsize")
        prof.pop("blockysize")
    tmp = path + ".partial.tif"
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(np.asarray(code, np.int32)[None])
        dst.update_tags(1, codes="1 burned, 0 not burned, -1 unrecorded (exp/exp98_burned.py)")
    os.replace(tmp, path)
    return path


def names_record(names, source, labels):
    return {"kind": "exp98 burned-area condition from MODIS MCD64A1 v061 (exp/exp98_burned.py)",
            "codes": {str(k): v for k, v in sorted(names.items())},
            "condition_names": condition_names_arg(names),
            "unrecorded": "-1: no burn date and water, unmapped or no tile in a month read; the raster's no-data",
            "rule": RULE, "code_2": CODE2_NOTE, "source": source,
            "grids": {lab: {"burned": f"burned_{lab}.tif", "scores": f"scores_{lab}.tif",
                            "assess": f"oe-inferencex assess scores_{lab}.tif --out <dir> --condition burned_{lab}.tif "
                                      + condition_names_arg(names)} for lab in labels}}


# ----------------------------------------------------------------------------- the parts
def discover_parts(paths):
    """Part directories: each path holding scores_*.tif, or each subdirectory of it that does, sorted."""
    out = []
    for p in paths:
        if glob.glob(os.path.join(glob.escape(p), "scores_*.tif")):
            out.append(p)
            continue
        subs = sorted(d for d in glob.glob(os.path.join(glob.escape(p), "*"))
                      if os.path.isdir(d) and glob.glob(os.path.join(glob.escape(d), "scores_*.tif")))
        if not subs:
            raise FileNotFoundError(f"no scores_<EPSG>.tif in {p} or its subdirectories")
        out += subs
    return out


def grid_footprint_wgs84(scores_path, margin_deg=0.02):
    """The scores grid's box in WGS84, a little larger (in memory only, for the STAC search)."""
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(scores_path) as src:
        w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
    return (w - margin_deg, s - margin_deg, e + margin_deg, n + margin_deg)


def run(score_paths, out_json, allow_missing=(), files=(), stac=True, year=YEAR, redo=False, pilot=False, threads=4,
        log=print, search=search_stac):
    t0 = time.time()
    parts = discover_parts(score_paths)
    months = [month_key(year, m) for m in range(1, 13)]
    allow = set(allow_missing)
    bad = sorted(m for m in allow if m not in months)
    if bad:
        raise ValueError(f"--allow-missing {bad}: months of {year} are YYYY-MM")
    jobs = []
    for d in parts:
        for s in sorted(glob.glob(os.path.join(glob.escape(d), "scores_*.tif"))):
            lab = re.sub(r"^scores_|\.tif$", "", os.path.basename(s))
            jobs.append((d, lab, s, os.path.join(d, f"burned_{lab}.tif")))
    there = [j for j in jobs if os.path.exists(j[3])]
    if there and not redo:
        raise FileExistsError(f"{len(there)} burned_<EPSG>.tif already written; --redo replaces them")
    local = items_from_files(files, year) if files else {}
    log(f"{len(parts)} part directories, {len(jobs)} grids; months from local files: {sorted(local) or 'none'}")
    record = {"stage": "exp98 burned area (exp/exp98_burned.py, exp/jobs/e98_burned.sh)", "pilot": bool(pilot),
              "commit": git_commit(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "rule": RULE, "code_2": CODE2_NOTE, "parts": []}
    results, missing_sets = [], set()
    for d, lab, s, bpath in jobs:
        log(f"== {os.path.basename(d)} / {lab}")
        notes = {}
        found = {}
        if stac:
            found, notes = select_items(search(grid_footprint_wgs84(s), year), year)
        for m, its in local.items():
            found[m] = its                                   # a local month replaces whatever the catalogue holds
        missing = [m for m in months if not found.get(m)]
        unexpected = [m for m in missing if m not in allow]
        if unexpected:
            raise ValueError(f"no MCD64A1 item for {unexpected} over this grid (allowed missing: {sorted(allow)}); "
                             "pass --allow-missing for a month known to be absent, or --item for a local file")
        missing_sets.add(tuple(missing))
        code, bits, flags, ck, cov = burned_for_grid(s, found, months, threads, log)
        read = [m for m in months if found.get(m)]
        entry = {"part": os.path.basename(os.path.normpath(d)), "grid": lab, "pixels": int(code.size),
                 "months_read": read, "months_missing": missing,
                 "months_from_files": sorted(m for m in local if m in read), "stac_items": notes,
                 "qa_check": {k: v for k, v in ck.items() if k not in ("months_read",)}}
        entry.update(grid_counts(code, bits, flags, cov, read, missing))
        results.append((d, lab, s, bpath, code, entry))
        record["parts"].append(entry)
    if len(missing_sets) > 1:
        raise ValueError("the months missing differ between grids, so the codes' names would differ between parts "
                         "and exp98 could not read them as one map; nothing is written")
    missing = list(next(iter(missing_sets))) if missing_sets else []
    names = code_names(year, missing)
    source = {"collection": COLLECTION, "stac_api": STAC, "assets": [BURN_ASSET, QA_ASSET], "product": "MCD64A1 v061",
              "year": year, "months_missing": missing,
              "months_from_files": sorted(local),
              "planetary_computer_gap": ("on 2026-10-08 Planetary Computer held no September 2023 item of this "
                                         "collection anywhere (268 items in August and October 2023); NASA's CMR "
                                         "lists the September granules"),
              "citation": CITATION, "licence": LICENCE,
              "accuracy": ("MCD64 C6.1 user guide section 7: global omission 72.6%, commission 40.2% against Landsat 8 "
                           "pairs, the omission mostly small burns; cropland burns low confidence")}
    by_dir = {}
    for d, lab, s, bpath, code, entry in results:
        write_burned(bpath, code, s)
        by_dir.setdefault(d, []).append(lab)
    for d, labs in by_dir.items():
        dump(names_record(names, source, labs), os.path.join(d, NAMES_JSON))
    tot = {"pixels_covered": sum(e["pixels_covered"] for e in record["parts"])}
    for k in (str(CODE_NONE), str(CODE_NOT), str(CODE_BURNED)):
        n = sum(e["pixels_covered_by_code"][k] for e in record["parts"])
        tot.setdefault("share_of_covered_by_code", {})[k] = n / tot["pixels_covered"] if tot["pixels_covered"] else None
    tot["note"] = "pixel-weighted over the parts; a pixel covered in two overlapping parts counts in each"
    record.update(names={str(k): v for k, v in names.items()}, condition_names=condition_names_arg(names),
                  source=source, totals=tot, seconds=round(time.time() - t0, 1))
    dump(record, out_json)
    log(f"names: {condition_names_arg(names)}")
    log(f"covered pixels by code over all parts: {json.dumps(tot['share_of_covered_by_code'])}")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scores", nargs="+", required=True,
                    help="the from-olmoearth part directories, or the directory holding them (data/exp98/scores)")
    ap.add_argument("--out-json", default=OUT_JSON, help="the counts-only record (default %(default)s)")
    ap.add_argument("--allow-missing", nargs="*", default=[], metavar="YYYY-MM",
                    help="months that may be absent from the catalogue (the job passes 2023-09)")
    ap.add_argument("--item", nargs=3, action="append", default=[], metavar=("YYYY-MM", "BURN_DATE_TIF", "QA_TIF"),
                    help="a month's tile from local GeoTIFFs of Burn_Date and QA (repeat for more tiles or months)")
    ap.add_argument("--no-stac", action="store_true", help="read only --item files; no network")
    ap.add_argument("--year", type=int, default=YEAR)
    ap.add_argument("--redo", action="store_true", help="replace burned rasters written before")
    ap.add_argument("--pilot", action="store_true", help="recorded in the counts (the job passes it with E98_PILOT=1)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    import rasterio
    # remote COGs: no directory listing on open, GDAL's own retries beneath the ones here
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2"):
        run(a.scores, a.out_json, a.allow_missing, [tuple(x) for x in a.item], stac=not a.no_stac, year=a.year,
            redo=a.redo, pilot=a.pilot, threads=a.threads)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001  (the job log needs the traceback, and the exit code must fail the job)
        traceback.print_exc()
        sys.exit(1)
