"""exp104: Ai2's Forest Loss Driver model (allenai/OlmoEarth-v1-FT-ForestLossDriver-Base) graded at 2026 MapBiomas Alerta
alerts in the Amazon. Preregistered in docs/plan/forest_loss_2026.md; read that page first, this file is the run.

    python exp/exp104_forest_loss.py --smoke                    # synthetic frame, polygons, windows and predictions,
                                                                # end to end through the grade; no network, no rslearn
    python exp/exp104_forest_loss.py --frame --private-dir DIR [--snapshot JSON | --fetch] [--record JSON]
    python exp/exp104_forest_loss.py --windows --private-dir DIR [--polygons GEOJSON] [--record JSON]
    python exp/exp104_forest_loss.py --grade --private-dir DIR --probs DIR [--out-dir DIR]   # refused until frozen
  helpers for exp/jobs/e104.sh:
    --check-configs                     exp/jobs/e104_config/ reverses to Ai2's pinned files, byte for byte
    --check-dataset DS [--private-dir DIR]   rslearn 0.1.17 itself (exp102's venv) parses DS: its config, its windows,
                                        each layer's request time range; with DIR, the centres against shapely/pyproj
    --imagery --private-dir DIR         windows with all eight Sentinel-2 item groups completed, by stratum (counts)
    --smoke-dataset DS                  two synthetic windows (no MapBiomas alert) for the GPU smoke

The frame (--frame). MapBiomas Alerta's published layer mapbiomas-alertas:dashboard-alert-shapefile, read as attributes
only (WFS, CQL_FILTER year_detected_at=2026; the snapshot of 11 October 2026 is the one the page fixed). An alert is in
the frame when its biomes list holds Amazônia, its status is published and detected_at runs from 2026-01-01 to
2026-06-30: 4,113 alerts whose sorted codes hash (sha256 of ",".join, first 16 hex digits) to FRAME_SHA_PREFIX; any
other count or hash is refused. The sample (seed 104): agriculture, numpy default_rng(104).choice of 300 of its 3,565
codes sorted ascending; mining (garimpo and mining), all 389; roads, all 7 (report-only). The sample with its codes
goes only to --private-dir (refused inside a git checkout); the record holds counts.

The windows (--windows). The sampled alerts' polygons from the same WFS (EPSG:4326, checked against each alert's own
lat/lng and area_ha), written only to --private-dir; one rslearn window per alert in --private-dir/dataset, whose
config.json is exp/jobs/e104_config/dataset.json: Ai2's FLD dataset.json (olmoearth_projects @ f3c9b0c8) with the
Sentinel-2 source of both layers switched to rslearn's Planetary Computer Sentinel2, harmonized, sorted by
eo:cloud_cover ascending; four scenes that each contain the window (min = max = 4, CONTAINS), the time offsets and
durations Ai2's (byte for byte Ai2's own Planetary Computer version of the file, c698da3c, 24 November 2025). Each
window: 128 x 128 px of EPSG:3857 at Ai2's 9.5546 m (2 pi R / 2^13 / 512, extract_alerts.py:37-40), the polygon's
centroid in pixel (64, 64), time range (t, t) with t the alert's detected_at at 00:00 UTC, as Ai2's extract_alerts sets
oe_start_time = oe_end_time (extract_alerts.py:239-248); rslearn then requests pre [t - 300 d, t - 120 d] and post
[t + 7 d, t + 187 d] (DataSourceConfig.get_request_time_range). Window names carry no code (u0000, ...).

The grade (--grade, refused until the page is frozen). from-olmoearth's per-window read of the output layer
(window_probs.npy, (10, 1, N), and window_probs.csv, the written class beside the argmax), the ten probabilities in
Ai2's order (FLD_CLASSES, checked against the class each feature carries). Prediction: the argmax; confidence: the
top-1 probability; suspicion: minus its log (signals.confidence form "top1"). An error: the prediction is not the
crosswalked MapBiomas class (agriculture -> agriculture; garimpo, mining -> mining; roads -> road). On the graded strata
(agriculture, mining), weights N_h / n_h over the alerts with a prediction:
  P1  the design-weighted AUROC of the suspicion for the errors >= 0.70;
  P2  the share of the gap from random to the ceiling closed by the least confident 10% of the design weight >= 0.25;
  P3  descriptive: the design-weighted error rate with the package's stratified exact interval (each stratum's exact
      hypergeometric interval at 1 - 0.05 / L, weighted by N_h / N), each stratum's exact 95% interval, Ai2's 76.1%
      beside them, the confusion counts;
  floors: 500 alerts with a prediction, 20 errors and 20 correct among them; otherwise no verdict.
Bootstraps: alerts resampled within each stratum (2000, seed 104). Roads are reported, never graded.

Outputs: exp104_summary.json and exp104_units.npz (per alert: stratum, weight, label, prediction, the ten
probabilities, the suspicion; no alert code, no position, no date); records of counts (--frame, --windows, --imagery,
--check-dataset). Every write is checked: no key naming a position (exp98's FORBIDDEN_KEYS) and no value equal to an
alert code of the frame.

Interpretations of the page, each a choice the page does not settle:
  a. "Biome is Amazônia" is "Amazônia is in the alert's biomes list" (10 frame alerts list Amazônia,Cerrado) and
     "published" is status_name "published": the only reading of the snapshot that gives the page's count and hash.
  b. The page's frame (4,113) includes the 151 "others" and the alert without a driver; they have no model class and
     are never sampled. The graded population is agriculture and mining (3,954); roads (7) are report-only.
  c. Alerts without a polygon, a window or imagery are missing at random within their stratum: the weights are
     recomputed over the alerts with a prediction (as exp99 does over plots with input), and the missing are counted.
  d. The centre is the area-weighted centroid of the polygon (all parts, holes subtracted) in EPSG:3857, as Ai2's
     deployment centres its windows (olmoearth_projects deploy/__init__.py:124-131, 228-230 at f3c9b0c8: each alert
     becomes its centroid point, since the window partitioner only uses the centroid; extract_alerts.py's nearest point
     sets only the date and the country). --windows counts, as a diagnostic, the alerts whose centroid falls outside
     the polygon.
  e. The bootstraps resample every graded stratum, the mining census too (as exp99 resamples each country); a
     resample without an error or without a correct alert is skipped and counted.
  f. Ai2's model.yaml applies a random Flip in default_config, which rslearn applies at predict as well
     (rslearn/train/data_module.py:132-135, rslearn/train/dataset.py:698-757; flip.py:39-55): the job fixes the seed
     (--seed_everything 104) and the workers rather than change the configuration.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
for _p in (ROOT, EXP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from oe_inferencex import estimate as est, metrics, signals   # noqa: E402  (numpy only)

OUT = os.path.join(EXP_DIR, "out")
PLAN = os.path.join(ROOT, "docs", "plan", "forest_loss_2026.md")
CONFIG_DIR = os.path.join(EXP_DIR, "jobs", "e104_config")

# ----------------------------------------------------------------------------- the reference: MapBiomas Alerta
WFS_BASE = "https://maps.alerta.mapbiomas.org/geoserver"
WFS_LAYER = "mapbiomas-alertas:dashboard-alert-shapefile"
ATTRS = ("alert_code", "area_ha", "biomes", "detected_at", "image_after_at", "deforestation_driver", "status_name")
FRAME_BIOME = "Amazônia"
FRAME_STATUS = "published"
FRAME_DATES = ("2026-01-01", "2026-06-30")         # detected_at, inclusive
FRAME_SHA_PREFIX = "12bf6fd243b27a9a"              # the page: sha256 of ",".join(sorted int codes), 16 hex digits
FRAME_N = 4113
FRAME_DRIVERS = {"agriculture": 3565, "garimpo": 380, "mining": 9, "roads": 7, "others": 151, "none": 1}
SNAPSHOT = {"read": "2026-10-11", "features": 13829, "file_sha256":
            "087b595bcee861ef59a46ed74ffdf9ded650f44a6fa1e103cd521ec6bd4810ed",
            "note": "the attribute snapshot the page's frame was fixed on; recorded, not required (the frame's hash is "
                    "what is checked)"}

STRATA = {"agriculture": ("agriculture",), "mining": ("garimpo", "mining"), "roads": ("roads",)}
STRATUM_NAMES = tuple(STRATA)
GRADED = ("agriculture", "mining")
REPORT_ONLY = ("roads",)
SAMPLE_SIZE = {"agriculture": 300}                 # the other strata are taken whole
SEED = 104

# ----------------------------------------------------------------------------- the model (Ai2's)
# the class order of Ai2's model.yaml task (classification.py's classes) = exp89's FLD_CLASSES (checked in the tests)
FLD_CLASSES = ("agriculture", "mining", "airstrip", "road", "logging", "burned", "landslide", "hurricane", "river",
               "none")
N_CLASSES = len(FLD_CLASSES)
CROSSWALK = {"agriculture": "agriculture", "garimpo": "mining", "mining": "mining", "roads": "road"}
STRATUM_CLASS = {"agriculture": "agriculture", "mining": "mining", "roads": "road"}
FLD_PRE = ("pre_sentinel2", "pre_sentinel2.1", "pre_sentinel2.2", "pre_sentinel2.3")
FLD_POST = ("post_sentinel2", "post_sentinel2.1", "post_sentinel2.2", "post_sentinel2.3")
FLD_LAYERS = FLD_PRE + FLD_POST
CLASS_PROPERTY, PROB_PROPERTY = "new_label", "probs"   # Ai2's model.yaml task: property_name, prob_property
OUTPUT_LAYER = "output"                                # Ai2's dataset.json declares it (vector); new in a new dataset
MODEL = {"repo": "allenai/OlmoEarth-v1-FT-ForestLossDriver-Base", "revision": "15502f8acb4caed6e3a7d777b0fb569c1f7eb791",
         "file": "model.ckpt", "sha256": "682c20b99f44a81769ad7d3ce1b64d933fae3810bcf5978f9999cfebf1c11855",
         "bytes": 381184175}                            # exp89's pins (ARMS["fld"])
AI2 = {"repo": "allenai/olmoearth_projects", "commit": "f3c9b0c89c7670b3525647dda4c14b76245d682d",
       "dir": "olmoearth_run_data/forest_loss_driver",
       "sha256": {"dataset.json": "fa7f8bf01864ee18daad33cce550c1d257a2d213d72fd414e9a7c9c005e2a96c",
                  "model.yaml": "e0c3829bb74f15c33581395bfacd03ea6df410c48e6829d3a309b98c93634736",
                  "olmoearth_run.yaml": "57ffce34e314eb21744558d3ec5ae9cffd03a648569b48e452da1ec48980c19c"},
       # Ai2's own Planetary Computer version of the FLD dataset.json (24 November 2025), replaced by olmoearth_datasets
       # in 0f7bdccb (9 April 2026); the derived file is byte for byte this one
       "mpc_version": {"commit": "c698da3cab4d353e3d8226615fc074762330fcda",
                       "sha256": "fe9ee5b0a3646e187274a576bd5f82ac9999a770c00ee15815052d7b926b8a50"}}
S2_LAYERS = ("pre_sentinel2", "post_sentinel2")
AI2_S2_SOURCE = {"class_path": "olmoearth_run.runner.tools.rslearn_data_sources.olmoearth_datasets.sentinel2_l2a."
                               "Sentinel2L2A",
                 "init_args": {"cache_dir": "cache/olmoearth_datasets", "harmonize": True,
                               "query": {"sort_by": "CLOUD_COVER", "sort_direction": "ASC"}}}
MPC_S2_SOURCE = {"class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
                 "init_args": {"cache_dir": "cache/planetary_computer", "harmonize": True, "sort_by": "eo:cloud_cover"}}
WANDB_BLOCK = ("  logger:\n    class_path: lightning.pytorch.loggers.WandbLogger\n    init_args:\n"
               "      project: ${WANDB_PROJECT}\n      name: ${WANDB_NAME}\n      entity: ${WANDB_ENTITY}\n")
CSV_BLOCK = ("  logger:\n    class_path: lightning.pytorch.loggers.CSVLogger\n    init_args:\n"
             "      save_dir: ${TRAINER_DATA_PATH}\n      name: log\n")
HEADER_END = "# --- end of exp104 header ---\n"

# ----------------------------------------------------------------------------- the window (Ai2's)
EARTH_R = 6378137.0
PIXEL_SIZE = 2 * math.pi * EARTH_R / 2 ** 13 / 512     # 9.554628535647032, olmoearth_run.yaml and extract_alerts.py
WINDOW_PX = 128
CROP_PX = 64                                           # model.yaml: Pad(size=64, mode="center")
CRS = "EPSG:3857"
GROUP = "e104"
PERIODS = {"pre_sentinel2": (-300, 180), "post_sentinel2": (7, 180)}   # days: time_offset, duration (dataset.json)
N_SCENES = 4
AXIS_TOL_DEG = 0.05            # an alert's lat/lng within this of its polygon's vertex mean
AREA_RATIO_REFUSE = (0.5, 2.0)  # the median of polygon area over area_ha outside this: the wrong geometries

# ----------------------------------------------------------------------------- the grade (the page)
THRESHOLDS = {"P1": 0.70, "P2": 0.25}
FLOORS = {"min_with_prediction": 500, "min_errors": 20, "min_correct": 20}
PREDICTIONS = tuple(THRESHOLDS)
AI2_ACCURACY = 0.761                                   # docs/forest_loss_driver.md (83 of 109 validation windows)
BUDGETS = (0.05, 0.10, 0.20)
BOOT = 2000
SMOKE_BOOT = 200
LEVEL = 0.90
EPS = 1e-12
FLOOR_P = float(np.finfo(np.float64).tiny)
SUM_TOLERANCE = 1e-3

CANNOT = ("Burned, logging, landslide, hurricane, river, airstrip and none: MapBiomas Alerta does not record them, so "
          "the frame tests agriculture against mining (and roads, report-only).",
          "Whether an error is the model's or the reference's: MapBiomas's driver is an analyst's call on "
          "high-resolution imagery and is taken as right.",
          "Alerts outside the Amazon biome, or before 2026.",
          "Alerts without a polygon, a window or four scenes in each period are treated as missing at random within "
          "their stratum; their number is reported.")


# ----------------------------------------------------------------------------- small helpers
def prereg_status(text=None):
    import exp89_finetuned_checkpoints as e89
    if text is None:
        with open(PLAN, encoding="utf-8") as f:
            text = f.read()
    return e89.prereg_status(text)


def bkey(b):
    return repr(float(b))


def finite(x):
    return x is not None and isinstance(x, (int, float, np.floating, np.integer)) and math.isfinite(float(x))


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha16(codes):
    """The page's frame hash: sha256 of ",".join of the sorted integer codes, its first 16 hex digits."""
    return hashlib.sha256(",".join(str(int(c)) for c in sorted(int(c) for c in codes)).encode()).hexdigest()[:16]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inside_a_checkout(path):
    p = os.path.realpath(path)
    while True:
        if os.path.exists(os.path.join(p, ".git")):
            return True
        parent = os.path.dirname(p)
        if parent == p:
            return False
        p = parent


def private_dir(path, create=True):
    """The directory that may hold alert codes and positions: refused inside any git checkout; mode 700."""
    if inside_a_checkout(path):
        raise ValueError(f"refusing a private directory inside a git checkout ({path}): it holds alert codes and "
                         "positions")
    if create:
        os.makedirs(path, exist_ok=True)
        os.chmod(path, 0o700)
    return path


def write_private_json(obj, path):
    tmp = path + ".partial"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def walk_values(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from walk_values(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from walk_values(v)
    else:
        yield obj


def check_no_codes(obj, codes):
    """Refuse an object that carries an alert code as a value or a key (int, or a string of its digits)."""
    if not codes:
        return
    s = {int(c) for c in codes}
    for v in walk_values(obj):
        if isinstance(v, (bool, np.bool_)):
            continue
        if isinstance(v, (int, np.integer)) and int(v) in s:
            raise ValueError("refusing to write an alert code")
        if isinstance(v, (float, np.floating)) and math.isfinite(v) and float(v).is_integer() and int(v) in s:
            raise ValueError("refusing to write an alert code")
        if isinstance(v, str) and v.strip().isdigit() and int(v) in s:
            raise ValueError("refusing to write an alert code")


def dump(obj, path, codes=()):
    """A record: no key naming a position (exp98's guard, inside e98.dump), no alert code; atomic."""
    import exp98_awf_deployment as e98
    check_no_codes(obj, codes)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    e98.dump(obj, path + ".partial")
    os.replace(path + ".partial", path)


def shuffle_units(units, priv=None):
    """The per-alert rows in a random order: the mining stratum is a census and the agriculture sample follows from the
    public rule, so rows kept in alert-code order would match the public alerts. The permutation is drawn from the
    operating system's entropy and kept only in the private directory (never in a tracked file). The row count is the
    stratum column's; the per-row arrays are those of that length, and the name lists (stratum_names, class_names) stay
    as they are. Until this was fixed the row count was read as "the one length every array shares", which the name
    lists broke, so the rows were returned unshuffled without a word (introduced in ce23f6d; found by the audit of
    exp104's grade before recording)."""
    if "stratum" not in units:
        raise ValueError("shuffle_units needs the per-row 'stratum' column to know the row count")
    n = len(units["stratum"])
    names = {"stratum_names", "class_names"}
    rows = [k for k, v in units.items() if k not in names and np.ndim(v) >= 1]
    bad = [k for k in rows if len(units[k]) != n]
    if bad:
        raise ValueError(f"shuffle_units: {bad} are not per-row arrays of length {n}")
    perm = np.random.default_rng(np.random.SeedSequence()).permutation(n)
    if priv:
        np.save(os.path.join(priv, "units_permutation.npy"), perm)
    return {k: (np.asarray(v)[perm] if k in rows else v) for k, v in units.items()}


def save_units(units, path, codes=()):
    import exp98_awf_deployment as e98
    for k, v in units.items():
        if str(k).lower() in e98.FORBIDDEN_KEYS or "code" in str(k).lower():
            raise ValueError(f"refusing to write units key {k!r}")
        a = np.asarray(v)
        if a.dtype.kind in "iu" and codes and np.isin(a, np.asarray(sorted(codes), dtype=np.int64)).any():
            raise ValueError(f"refusing to write units {k!r}: it holds an alert code")
    np.savez_compressed(path, **units)


# ----------------------------------------------------------------------------- WFS (attributes; polygons of the sample)
def _get_json(url, timeout=180, tries=4, log=print):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "olmoearth_inferenceX exp104 (research)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as ex:
            last = ex
            if k < tries - 1:
                log(f"WFS request failed ({type(ex).__name__}); retry {k + 1} of {tries - 1}")
                time.sleep(15 * (k + 1))
    raise RuntimeError(f"the WFS did not answer with JSON after {tries} tries: {last!r}")


def wfs_url(params):
    return WFS_BASE + "/wfs?" + urllib.parse.urlencode(params)


def fetch_snapshot(dest, log=print):
    """The attribute snapshot: every 2026 alert, the page's attributes only (no geometry, no bounding box kept)."""
    d = _get_json(wfs_url({"service": "WFS", "version": "1.0.0", "request": "GetFeature", "typeName": WFS_LAYER,
                           "outputFormat": "application/json", "CQL_FILTER": "year_detected_at=2026",
                           "propertyName": ",".join(ATTRS)}), timeout=600, log=log)
    feats = [{"type": "Feature", "properties": {k: (f.get("properties") or {}).get(k) for k in ATTRS}}
             for f in d.get("features") or []]
    snap = {"type": "FeatureCollection", "features": feats, "timeStamp": d.get("timeStamp") or utc_now(),
            "numberMatched": d.get("numberMatched", d.get("totalFeatures"))}
    write_private_json(snap, dest)
    return dest


def fetch_polygons(codes, batch=50, log=print, get=None):
    """{code: {"geometry", "lat", "lng"}} for the given codes, EPSG:4326, WFS 1.0.0 (x = longitude). `get` replaces
    the HTTP call (url -> FeatureCollection) in the smoke."""
    get = get or (lambda url: _get_json(url, log=log))
    want = [int(c) for c in codes]
    wanted = set(want)
    out = {}
    for i in range(0, len(want), batch):
        part = want[i:i + batch]
        base = {"service": "WFS", "version": "1.0.0", "request": "GetFeature", "typeName": WFS_LAYER,
                "outputFormat": "application/json", "srsName": "EPSG:4326", "propertyName": "alert_code,lat,lng,geom"}
        d = None
        for cql in (f"alert_code IN ({','.join(map(str, part))})",
                    " OR ".join(f"alert_code={c}" for c in part)):
            try:
                d = get(wfs_url(dict(base, CQL_FILTER=cql)))
            except RuntimeError:
                d = None
            if isinstance(d, dict) and isinstance(d.get("features"), list):
                break
        if d is None:
            raise RuntimeError("the WFS answered neither filter form with a FeatureCollection")
        for f in d["features"]:
            p = f.get("properties") or {}
            code = int(p["alert_code"])
            if code not in wanted:
                raise ValueError("the WFS returned an alert that was not asked for")
            if code in out:
                raise ValueError("the WFS returned one alert twice")
            out[code] = {"geometry": f.get("geometry"), "lat": p.get("lat"), "lng": p.get("lng")}
        log(f"polygons: {min(i + batch, len(want))} of {len(want)} asked, {len(out)} received")
    return out


# ----------------------------------------------------------------------------- the frame and the sample
def load_snapshot(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    feats = d.get("features")
    if not isinstance(feats, list) or not feats:
        raise ValueError("the snapshot holds no features")
    props = []
    for f in feats:
        p = f.get("properties") or {}
        miss = [k for k in ATTRS if k not in p]
        if miss:
            raise ValueError(f"a snapshot feature lacks {miss}: not the page's attribute list")
        props.append({k: p[k] for k in ATTRS})
    meta = {"features": len(props), "timeStamp": d.get("timeStamp"), "numberMatched": d.get("numberMatched"),
            "file_sha256": sha256_file(path)}
    return props, meta


def driver_key(d):
    return "none" if d in (None, "") else str(d)


def in_frame(p):
    biomes = [b.strip() for b in str(p.get("biomes") or "").split(",")]
    dt = str(p.get("detected_at") or "")[:10]
    return FRAME_BIOME in biomes and p.get("status_name") == FRAME_STATUS and FRAME_DATES[0] <= dt <= FRAME_DATES[1]


def stratum_of(driver):
    for s, ds in STRATA.items():
        if driver in ds:
            return s
    return None


def build_frame(props, expected_hash=FRAME_SHA_PREFIX, expected_n=FRAME_N, expected_drivers=FRAME_DRIVERS):
    """(the frame's alerts, sorted by code; the counts record). Refuses a count, a driver tally or a hash other than
    the page's (None skips a check, for synthetic inputs)."""
    codes = [int(p["alert_code"]) for p in props]
    if len(set(codes)) != len(codes):
        raise ValueError("the snapshot holds an alert code twice")
    frame = sorted((p for p in props if in_frame(p)), key=lambda p: int(p["alert_code"]))
    drivers = {}
    for p in frame:
        k = driver_key(p["deforestation_driver"])
        drivers[k] = drivers.get(k, 0) + 1
    h = sha16(p["alert_code"] for p in frame)
    rec = {"rule": f"biomes list holds {FRAME_BIOME}, status_name {FRAME_STATUS}, detected_at from {FRAME_DATES[0]} "
                   f"to {FRAME_DATES[1]} inclusive",
           "n_frame": len(frame), "by_driver": dict(sorted(drivers.items())), "sha256_prefix": h,
           "excluded": {"other_biomes": sum(FRAME_BIOME not in str(p.get("biomes") or "").split(",") for p in props),
                        "not_published": sum(p.get("status_name") != FRAME_STATUS for p in props),
                        "detected_outside_dates": sum(not (FRAME_DATES[0] <= str(p.get("detected_at") or "")[:10]
                                                           <= FRAME_DATES[1]) for p in props)},
           "multi_biome_in_frame": sum("," in str(p.get("biomes") or "") for p in frame)}
    if expected_n is not None and len(frame) != expected_n:
        raise ValueError(f"the frame holds {len(frame)} alerts, the page fixed {expected_n}: not the snapshot of "
                         "11 October 2026")
    if expected_drivers is not None and rec["by_driver"] != dict(sorted(expected_drivers.items())):
        raise ValueError(f"the frame's drivers are {rec['by_driver']}, the page fixed {expected_drivers}")
    if expected_hash is not None and h != expected_hash:
        raise ValueError(f"the frame's codes hash to {h}, the page fixed {expected_hash}: refused")
    rec["matches_page"] = bool(h == FRAME_SHA_PREFIX and len(frame) == FRAME_N)
    return frame, rec


def draw_sample(frame, sizes=SAMPLE_SIZE, seed=SEED):
    """The stratified sample: in each stratum the codes sorted ascending; a stratum with a size in `sizes` gets
    numpy default_rng(seed).choice(N_h, n_h, replace=False) (one generator, strata in STRATA's order), sorted; the
    others are taken whole. Each unit's weight N_h / n_h. Returns (units, record)."""
    rng = np.random.default_rng(seed)
    units, rec = [], {"seed": seed, "strata": {}}
    for s in STRATUM_NAMES:
        members = sorted((p for p in frame if stratum_of(driver_key(p["deforestation_driver"])) == s),
                         key=lambda p: int(p["alert_code"]))
        N = len(members)
        if s in sizes:
            n = int(sizes[s])
            if n > N:
                raise ValueError(f"stratum {s} holds {N} alerts, fewer than its sample of {n}")
            pick = np.sort(rng.choice(N, n, replace=False))
            chosen = [members[i] for i in pick]
            how = f"simple random sample, numpy default_rng({seed}).choice({N}, {n}, replace=False) over codes sorted"
        else:
            chosen, n, how = members, N, "all"
        w = N / n if n else 0.0
        rec["strata"][s] = {"N": N, "n": n, "weight": w, "design": how, "graded": s in GRADED,
                            "by_driver": {d: sum(driver_key(p["deforestation_driver"]) == d for p in chosen)
                                          for d in STRATA[s]}}
        for p in chosen:
            units.append({"stratum": s, "alert_code": int(p["alert_code"]),
                          "driver": driver_key(p["deforestation_driver"]), "detected_at": str(p["detected_at"])[:10],
                          "image_after_at": p.get("image_after_at"), "area_ha": p.get("area_ha"), "weight": w})
    for k, u in enumerate(units):
        u["window"] = f"u{k:04d}"
    rec["n_sample"] = len(units)
    rec["not_in_any_stratum"] = sum(stratum_of(driver_key(p["deforestation_driver"])) is None for p in frame)
    return units, rec


def month_counts(units):
    out = {}
    for u in units:
        m = str(u["detected_at"])[:7]
        out[m] = out.get(m, 0) + 1
    return dict(sorted(out.items()))


def run_frame(priv, snapshot=None, fetch=False, record=None, expect=None, sizes=SAMPLE_SIZE, log=print):
    """--frame. `expect` overrides the page's (hash, n, drivers) and `sizes` the sample sizes, for synthetic inputs."""
    private_dir(priv)
    if fetch:
        snapshot = fetch_snapshot(os.path.join(priv, "alerts2026_attrs_fetched.json"), log=log)
    if not snapshot:
        raise ValueError("--frame needs --snapshot or --fetch")
    props, meta = load_snapshot(snapshot)
    eh, en, ed = expect if expect is not None else (FRAME_SHA_PREFIX, FRAME_N, FRAME_DRIVERS)
    frame, frec = build_frame(props, eh, en, ed)
    units, srec = draw_sample(frame, sizes=sizes)
    sample = {"experiment": "exp104", "frame_sha256_prefix": frec["sha256_prefix"], "seed": SEED, "units": units,
              "strata": {s: {"N": v["N"], "n": v["n"]} for s, v in srec["strata"].items()}, "written": utc_now()}
    write_private_json(sample, os.path.join(priv, "e104_sample.json"))
    # every frame code, for the output guards (private)
    write_private_json({"codes": [int(p["alert_code"]) for p in frame]}, os.path.join(priv, "e104_frame_codes.json"))
    rec = {"experiment": "exp104: the frame and the sample (counts only)", "utc": utc_now(),
           "snapshot": {"features": meta["features"], "numberMatched": meta["numberMatched"],
                        "timeStamp": meta["timeStamp"], "file_sha256": meta["file_sha256"],
                        "is_the_pages_snapshot": meta["file_sha256"] == SNAPSHOT["file_sha256"],
                        "fetched": bool(fetch)},
           "frame": frec, "sample": srec,
           "sample_by_month": {s: month_counts([u for u in units if u["stratum"] == s]) for s in STRATUM_NAMES},
           "crosswalk": CROSSWALK, "private_files": ["e104_sample.json", "e104_frame_codes.json"]}
    if record:
        dump(rec, record, codes=[p["alert_code"] for p in frame])
    log(json.dumps({"n_frame": frec["n_frame"], "sha256_prefix": frec["sha256_prefix"],
                    "sample": {s: v["n"] for s, v in srec["strata"].items()}}))
    return rec, units


def read_sample(priv):
    with open(os.path.join(priv, "e104_sample.json"), encoding="utf-8") as f:
        return json.load(f)


def read_frame_codes(priv):
    p = os.path.join(priv, "e104_frame_codes.json")
    if not os.path.isfile(p):
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)["codes"]


# ----------------------------------------------------------------------------- geometry (numpy only)
def mercator(lon, lat):
    """EPSG:3857 (spherical Web Mercator) from longitude and latitude in degrees."""
    lon, lat = np.asarray(lon, np.float64), np.asarray(lat, np.float64)
    return EARTH_R * np.radians(lon), EARTH_R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def parts_of(geom):
    t, c = geom.get("type"), geom.get("coordinates")
    if t == "Polygon":
        return [c]
    if t == "MultiPolygon":
        return list(c)
    raise ValueError(f"an alert's geometry is a {t}, not a polygon")


def _ring_area_centroid(x, y):
    if x[0] != x[-1] or y[0] != y[-1]:
        x, y = np.append(x, x[0]), np.append(y, y[0])
    x0, y0 = x[0], y[0]
    xs, ys = x - x0, y - y0
    cross = xs[:-1] * ys[1:] - xs[1:] * ys[:-1]
    a = cross.sum() / 2
    if a == 0:
        return 0.0, float(x0), float(y0)
    return abs(float(a)), float(((xs[:-1] + xs[1:]) * cross).sum() / (6 * a) + x0), \
        float(((ys[:-1] + ys[1:]) * cross).sum() / (6 * a) + y0)


def _in_ring(px, py, x, y):
    xj, yj = np.roll(x, 1), np.roll(y, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hit = ((y > py) != (yj > py)) & (px < (xj - x) * (py - y) / (yj - y) + x)
    return bool(hit.sum() % 2)


def polygon_facts(geom, swap=False):
    """In EPSG:3857: the area-weighted centroid (holes subtracted), the area in projected m^2, the vertex bounding box,
    whether the centroid lies inside, and the vertex mean in degrees (for the axis check)."""
    tot = sx = sy = 0.0
    xs_all, ys_all, lon_all, lat_all, rings = [], [], [], [], []
    for part in parts_of(geom):
        pr = []
        for i, ring in enumerate(part):
            a = np.asarray(ring, np.float64)[:, :2]
            lon, lat = (a[:, 1], a[:, 0]) if swap else (a[:, 0], a[:, 1])
            x, y = mercator(lon, lat)
            area, cx, cy = _ring_area_centroid(x, y)
            s = 1.0 if i == 0 else -1.0
            tot += s * area
            sx += s * area * cx
            sy += s * area * cy
            xs_all.append(x)
            ys_all.append(y)
            lon_all.append(lon)
            lat_all.append(lat)
            pr.append((x, y))
        rings.append(pr)
    x, y = np.concatenate(xs_all), np.concatenate(ys_all)
    degenerate = not tot > 0
    cx, cy = (float(x.mean()), float(y.mean())) if degenerate else (sx / tot, sy / tot)
    inside = any(_in_ring(cx, cy, *pr[0]) and not any(_in_ring(cx, cy, *h) for h in pr[1:]) for pr in rings)
    return {"cx": cx, "cy": cy, "area_m2": max(tot, 0.0), "degenerate": degenerate, "inside": inside,
            "xmin": float(x.min()), "xmax": float(x.max()), "ymin": float(y.min()), "ymax": float(y.max()),
            "lon_mean": float(np.concatenate(lon_all).mean()), "lat_mean": float(np.concatenate(lat_all).mean())}


def ai2_centre_px(geom, swap=False, res=PIXEL_SIZE):
    """The centre by the letter of Ai2's code (olmoearth_projects deploy/__init__.py:140-145 at f3c9b0c8,
    simplify_features_to_centroids): shapely's centroid of the polygon as published, in WGS84 and unrepaired, here
    projected to EPSG:3857 with pyproj, in pixels (column, row). On a valid polygon it is polygon_facts' centroid up to
    the projection's curvature; on an invalid multi-part polygon it can lie far from it (job 1286487: 24 of the 343
    invalid sampled polygons, up to about 1,870 px, with the cluster's shapely; not reproduced with synthetic polygons
    on shapely 2.1.2, so the mechanism is not established). Reported by the dataset check, not gated."""
    import pyproj
    import shapely.geometry
    shp = shapely.geometry.shape(geom)
    if swap:
        from shapely.ops import transform
        shp = transform(lambda x, y, z=None: (y, x), shp)
    c = shp.centroid
    x, y = pyproj.Transformer.from_crs(4326, 3857, always_xy=True).transform(c.x, c.y)
    return x / res, -y / res


def ring_centre_px(geom, swap=False, res=PIXEL_SIZE):
    """The windows' own centre (polygon_facts: the rings' area-weighted centroid in EPSG:3857, holes subtracted) by a
    second route: each ring projected with pyproj and measured by shapely as a polygon of its own (one ring, so a
    self-intersecting ring's orientation cancels), in pixels (column, row)."""
    import pyproj
    import shapely.geometry
    t = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
    tot = sx = sy = 0.0
    for part in parts_of(geom):
        for i, ring in enumerate(part):
            a = np.asarray(ring, np.float64)[:, :2]
            if len(a) < 3:
                continue                              # no area, as in polygon_facts
            lon, lat = (a[:, 1], a[:, 0]) if swap else (a[:, 0], a[:, 1])
            x, y = t.transform(lon, lat)
            poly = shapely.geometry.Polygon(np.column_stack([x, y]))
            if not poly.area > 0:
                continue                              # a zero-area ring adds nothing to either sum
            sign = 1.0 if i == 0 else -1.0
            tot += sign * poly.area
            sx += sign * poly.area * poly.centroid.x
            sy += sign * poly.area * poly.centroid.y
    return sx / tot / res, -sy / tot / res


def ground_area_m2(f):
    """Projected area times cos^2 of the centroid's latitude: Web Mercator's scale squared, for the area check."""
    lat = math.degrees(2 * math.atan(math.exp(f["cy"] / EARTH_R)) - math.pi / 2)
    return f["area_m2"] * math.cos(math.radians(lat)) ** 2


def window_bounds(cx, cy, px=WINDOW_PX, res=PIXEL_SIZE):
    """Pixel bounds [x0, y0, x1, y1] of the px x px window whose pixel (px/2, px/2) holds (cx, cy): rslearn's pixel
    coordinates are the CRS coordinates over the resolution (x_res = res, y_res = -res)."""
    col, row = math.floor(cx / res), math.floor(cy / -res)
    h = px // 2
    return [col - h, row - h, col + h, row + h]


def event_time(detected_at):
    d = datetime.strptime(str(detected_at)[:10], "%Y-%m-%d")
    return d.replace(tzinfo=timezone.utc)


def request_range(t0, t1, offset_days, duration_days):
    """rslearn's DataSourceConfig.get_request_time_range (rslearn/config/dataset.py:509-530 at v0.1.17): the window's
    range moved by time_offset, then the end set to the start plus duration."""
    s, e = t0 + timedelta(days=offset_days), t1 + timedelta(days=offset_days)
    if duration_days:
        e = s + timedelta(days=duration_days)
    return s, e


def window_metadata(bounds, t):
    """metadata.json as rslearn 0.1.17's FileWindowStorage writes it (rslearn/dataset/storage/file.py:171-187)."""
    return {"projection": {"crs": CRS, "x_resolution": PIXEL_SIZE, "y_resolution": -PIXEL_SIZE},
            "bounds": [int(b) for b in bounds], "time_range": [t.isoformat(), t.isoformat()], "options": {}}


def write_dataset_config(ds, config=None):
    src = config or os.path.join(CONFIG_DIR, "dataset.json")
    os.makedirs(ds, exist_ok=True)
    with open(src, "rb") as f:
        b = f.read()
    with open(os.path.join(ds, "config.json"), "wb") as f:
        f.write(b)
    return hashlib.sha256(b).hexdigest()


def write_window(ds, group, name, bounds, t):
    wd = os.path.join(ds, "windows", group, name)
    os.makedirs(wd, exist_ok=True)
    with open(os.path.join(wd, "metadata.json"), "w") as f:
        json.dump(window_metadata(bounds, t), f)
    return wd


def axis_order(polys):
    """'lonlat' when each alert's own lng/lat sits at its polygon's vertex mean as given, 'latlon' when it does so
    swapped; refused otherwise. Alerts without a parseable lat/lng are left out of the vote."""
    d1, d2 = [], []
    for v in polys.values():
        try:
            lat, lng = float(v["lat"]), float(v["lng"])
        except (TypeError, ValueError, KeyError):
            continue
        f = polygon_facts(v["geometry"])
        d1.append(abs(f["lon_mean"] - lng) + abs(f["lat_mean"] - lat))
        d2.append(abs(f["lat_mean"] - lng) + abs(f["lon_mean"] - lat))
    if not d1:
        raise ValueError("no alert carries a parseable lat/lng: the axis order cannot be checked")
    m1, m2 = float(np.median(d1)), float(np.median(d2))
    if m1 <= AXIS_TOL_DEG and m1 < m2:
        return "lonlat", {"median_deg": m1, "voters": len(d1)}
    if m2 <= AXIS_TOL_DEG and m2 < m1:
        return "latlon", {"median_deg": m2, "voters": len(d1)}
    raise ValueError(f"the polygons do not sit at their alerts' lat/lng either way (medians {m1:.3g}, {m2:.3g} deg)")


def quantiles(v, qs=(0.0, 0.1, 0.5, 0.9, 1.0)):
    v = np.asarray([x for x in v if finite(x)], np.float64)
    return {f"{q:g}": float(np.quantile(v, q)) for q in qs} if v.size else {}


def build_windows(priv, polys, config=None, redo=False, log=print):
    """One window per sampled alert with a polygon, in priv/dataset. Returns the counts record."""
    sample = read_sample(priv)
    units = sample["units"]
    ds = os.path.join(priv, "dataset")
    if os.path.isdir(os.path.join(ds, "windows")):
        if not redo:
            raise ValueError("the private dataset already holds windows/ (--redo replaces it)")
        import shutil
        shutil.rmtree(ds)
    order, ax = axis_order(polys)
    swap = order == "latlon"
    cfg_sha = write_dataset_config(ds, config)
    rec = {"axis_order": order, "axis_check": ax, "config_sha256": cfg_sha,
           "config_is_ai2_mpc_version": cfg_sha == AI2["mpc_version"]["sha256"], "strata": {}}
    ratio, fits_w, fits_c, outside, degenerate = [], 0, 0, 0, 0
    per = {s: {"sampled": 0, "with_polygon": 0, "windows": 0} for s in STRATUM_NAMES}
    n_post_open = 0
    for u in units:
        s = u["stratum"]
        per[s]["sampled"] += 1
        v = polys.get(int(u["alert_code"]))
        if v is None or not v.get("geometry"):
            continue
        per[s]["with_polygon"] += 1
        f = polygon_facts(v["geometry"], swap=swap)
        b = window_bounds(f["cx"], f["cy"])
        t = event_time(u["detected_at"])
        write_window(ds, GROUP, u["window"], b, t)
        per[s]["windows"] += 1
        degenerate += f["degenerate"]
        outside += not f["inside"]
        px0, px1 = f["xmin"] / PIXEL_SIZE, f["xmax"] / PIXEL_SIZE
        py0, py1 = f["ymax"] / -PIXEL_SIZE, f["ymin"] / -PIXEL_SIZE
        fits_w += b[0] <= px0 and px1 <= b[2] and b[1] <= py0 and py1 <= b[3]
        c0 = (b[0] + (WINDOW_PX - CROP_PX) // 2, b[1] + (WINDOW_PX - CROP_PX) // 2)
        fits_c += c0[0] <= px0 and px1 <= c0[0] + CROP_PX and c0[1] <= py0 and py1 <= c0[1] + CROP_PX
        if finite(u.get("area_ha")) and u["area_ha"] > 0:
            ratio.append(ground_area_m2(f) / (float(u["area_ha"]) * 1e4))
        if request_range(t, t, *PERIODS["post_sentinel2"])[1] > datetime.now(timezone.utc):
            n_post_open += 1
    med = float(np.median(ratio)) if ratio else None
    if med is not None and not AREA_RATIO_REFUSE[0] <= med <= AREA_RATIO_REFUSE[1]:
        raise ValueError(f"the polygons' areas are {med:.3g} times area_ha at the median: not the alerts' geometries")
    rec.update(strata=per, n_windows=sum(p["windows"] for p in per.values()),
               centroid_outside_polygon=outside, degenerate_polygons=degenerate,
               polygon_fits_window=fits_w, polygon_fits_centre_crop=fits_c,
               area_ratio_quantiles=quantiles(ratio),
               post_period_still_open=n_post_open,
               window={"crs": CRS, "resolution_m": PIXEL_SIZE, "px": WINDOW_PX, "centre_crop_px": CROP_PX,
                       "time_range": "(detected_at 00:00 UTC, the same)",
                       "request_days": {k: {"time_offset": o, "duration": d, "from": o, "to": o + d}
                                        for k, (o, d) in PERIODS.items()}},
           group=GROUP)
    log(json.dumps({"windows": rec["n_windows"], "by_stratum": {s: p["windows"] for s, p in per.items()},
                    "axis_order": order, "centroid_outside_polygon": outside}))
    return rec


def run_windows(priv, polygons=None, config=None, record=None, redo=False, log=print, get=None):
    private_dir(priv, create=False)
    sample = read_sample(priv)
    codes = [u["alert_code"] for u in sample["units"]]
    path = os.path.join(priv, "e104_polygons.json")
    if polygons:
        with open(polygons, encoding="utf-8") as f:
            polys = {int(k): v for k, v in json.load(f).items()}
    elif os.path.isfile(path) and not redo:
        with open(path, encoding="utf-8") as f:
            polys = {int(k): v for k, v in json.load(f).items()}
        log("polygons read from the private directory (an earlier fetch)")
    else:
        polys = fetch_polygons(codes, log=log, get=get)
        write_private_json({str(k): v for k, v in polys.items()}, path)
    rec = build_windows(priv, polys, config=config, redo=redo, log=log)
    rec = {"experiment": "exp104: the windows (counts only)", "utc": utc_now(), "polygons_received": len(polys),
           "polygons_missing": len(codes) - len(set(codes) & set(polys)), **rec}
    if record:
        dump(rec, record, codes=read_frame_codes(priv) or codes)
    return rec


# ----------------------------------------------------------------------------- the configs (Ai2's, switched source)
def derive_dataset_config(ai2):
    d = json.loads(json.dumps(ai2))
    for name in S2_LAYERS:
        src = d["layers"][name]["data_source"]
        if {k: src[k] for k in ("class_path", "init_args")} != AI2_S2_SOURCE:
            raise ValueError(f"Ai2's {name} source is not the one pinned")
        src.update(json.loads(json.dumps(MPC_S2_SOURCE)))
    return d


def reverse_dataset_config(derived):
    d = json.loads(json.dumps(derived))
    for name in S2_LAYERS:
        src = d["layers"][name]["data_source"]
        if {k: src[k] for k in ("class_path", "init_args")} != MPC_S2_SOURCE:
            raise ValueError(f"the derived {name} source is not Planetary Computer's as documented")
        src.update(json.loads(json.dumps(AI2_S2_SOURCE)))
    return d


def canonical_json(obj):
    """Ai2's dataset.json is json.dumps(indent=2, sort_keys=True) plus a newline (its sha256 reproduces)."""
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode()


def derive_model_yaml(ai2_text, header):
    if ai2_text.count(WANDB_BLOCK) != 1:
        raise ValueError("Ai2's model.yaml does not hold the pinned WandbLogger block once")
    return header + ai2_text.replace(WANDB_BLOCK, CSV_BLOCK)


def reverse_model_yaml(text):
    if HEADER_END not in text:
        raise ValueError("the derived model.yaml has no exp104 header")
    body = text.split(HEADER_END, 1)[1]
    if body.count(CSV_BLOCK) != 1:
        raise ValueError("the derived model.yaml does not hold the CSVLogger block once")
    return body.replace(CSV_BLOCK, WANDB_BLOCK)


def check_configs(config_dir=CONFIG_DIR):
    """The committed configs, reversed, are Ai2's pinned files byte for byte; and the dataset config is Ai2's own
    Planetary Computer version (c698da3c) byte for byte."""
    with open(os.path.join(config_dir, "dataset.json"), "rb") as f:
        db = f.read()
    with open(os.path.join(config_dir, "model.yaml"), encoding="utf-8") as f:
        mt = f.read()
    ds_back = hashlib.sha256(canonical_json(reverse_dataset_config(json.loads(db)))).hexdigest()
    my_back = hashlib.sha256(reverse_model_yaml(mt).encode()).hexdigest()
    rec = {"dataset.json": {"canonical": canonical_json(json.loads(db)) == db,
                            "reversed_is_ai2s": ds_back == AI2["sha256"]["dataset.json"],
                            "is_ai2s_mpc_version": hashlib.sha256(db).hexdigest() == AI2["mpc_version"]["sha256"]},
           "model.yaml": {"reversed_is_ai2s": my_back == AI2["sha256"]["model.yaml"]},
           "ai2": {k: AI2[k] for k in ("repo", "commit", "dir")}}
    rec["ok"] = all(all(v.values()) for k, v in rec.items() if k != "ai2")
    return rec


def dataset_periods(config_path=None):
    with open(config_path or os.path.join(CONFIG_DIR, "dataset.json"), encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for name in S2_LAYERS:
        src = d["layers"][name]["data_source"]
        out[name] = (int(src["time_offset"].rstrip("d")), int(src["duration"].rstrip("d")),
                     src["query_config"]["min_matches"], src["query_config"]["max_matches"],
                     src["query_config"]["space_mode"])
    return out


# ----------------------------------------------------------------------------- rslearn's own reading (exp102's venv)
def check_dataset(ds, priv=None, workers=8):
    """rslearn 0.1.17 parses the dataset and its windows; every window's projection, size and time range, and each
    Sentinel-2 layer's request range by rslearn's own get_request_time_range; with priv, every window's centre pixel
    against a second route to the windows' own definition (ring_centre_px) for every polygon, against rslearn's own
    projection for every valid one, and, for an invalid polygon, the repaired polygon's centroid (make_valid, as rslearn
    repairs it, or buffer(0) where make_valid fails) inside the window's central crop; the letter of Ai2's code
    (ai2_centre_px) is reported, not gated. Counts and offsets only."""
    import importlib.metadata as md
    from upath import UPath
    from rslearn.dataset.dataset import Dataset
    from rslearn.utils.geometry import Projection, STGeometry, WGS84_PROJECTION
    from rasterio.crs import CRS as RCRS
    import shapely.errors
    import shapely.geometry
    d = Dataset(UPath(ds))
    wins = d.load_windows(workers=workers)
    rec = {"rslearn": md.version("rslearn"), "windows": len(wins), "layers": sorted(d.layers), "bad": {}}
    bad = {"projection": 0, "size": 0, "time_range": 0, "pre_range": 0, "post_range": 0, "centre_pixel": 0}
    for name in S2_LAYERS:
        cfg = d.layers[name].data_source
        rec[f"{name}_source"] = {"class_path": cfg.class_path, "space_mode": str(cfg.query_config.space_mode),
                                 "min_matches": cfg.query_config.min_matches,
                                 "max_matches": cfg.query_config.max_matches,
                                 "time_offset_days": cfg.time_offset.total_seconds() / 86400,
                                 "duration_days": cfg.duration.total_seconds() / 86400}
    for w in wins:
        p = w.projection
        bad["projection"] += not (p.crs.to_epsg() == 3857 and abs(p.x_resolution - PIXEL_SIZE) < 1e-9
                                  and abs(p.y_resolution + PIXEL_SIZE) < 1e-9)
        b = w.bounds
        bad["size"] += not (b[2] - b[0] == WINDOW_PX and b[3] - b[1] == WINDOW_PX)
        t0, t1 = w.time_range
        bad["time_range"] += not (t0 == t1 and t0.hour == 0 and t0.utcoffset() == timedelta(0))
        for name, key in (("pre_sentinel2", "pre_range"), ("post_sentinel2", "post_range")):
            got = d.layers[name].data_source.get_request_time_range(w.time_range)
            bad[key] += got != request_range(t0, t1, *PERIODS[name])
    if priv:
        sample = {u["window"]: u for u in read_sample(priv)["units"]}
        with open(os.path.join(priv, "e104_polygons.json"), encoding="utf-8") as f:
            polys = json.load(f)
        order, _ = axis_order({int(k): v for k, v in polys.items()})
        proj = Projection(RCRS.from_epsg(3857), PIXEL_SIZE, -PIXEL_SIZE)
        half = WINDOW_PX // 2

        def at_centre(c, w):
            return (w.bounds[0] + half - 1 <= c[0] <= w.bounds[0] + half + 2 and
                    w.bounds[1] + half - 1 <= c[1] <= w.bounds[1] + half + 2)

        bad["rslearn_centre_valid_polygon"] = 0
        bad["repaired_centre_outside_crop"] = 0
        crop = CROP_PX // 2 - 1
        invalid, bad_geometry, misses, rep_d, ai2_d = 0, 0, [], [], []

        def off(c, w):
            return c[0] - (w.bounds[0] + half), c[1] - (w.bounds[1] + half)

        for w in wins:
            u = sample.get(w.name)
            v = polys.get(str(u["alert_code"])) if u else None
            if v is None:
                bad["centre_pixel"] += 1
                misses.append({"reason": "no sample unit or no polygon"})
                continue
            swap = order == "latlon"
            shp = shapely.geometry.shape(v["geometry"])
            if swap:
                from shapely.ops import transform
                shp = transform(lambda x, y, z=None: (y, x), shp)
            valid = bool(shp.is_valid)
            c = ring_centre_px(v["geometry"], swap=swap)
            if not at_centre(c, w):
                bad["centre_pixel"] += 1
                d = off(c, w)
                misses.append({"reason": "second route to the windows' centroid off the window centre",
                               "valid": valid, "parts": len(parts_of(v["geometry"])),
                               "dx_px": round(d[0], 1), "dy_px": round(d[1], 1)})
            a = ai2_centre_px(v["geometry"], swap=swap)
            if not at_centre(a, w):
                ai2_d.append(round(math.hypot(*off(a, w)), 1))
            try:
                r = STGeometry(WGS84_PROJECTION, shp, None).to_projection(proj).shp.centroid
                r = (r.x, r.y)
            except shapely.errors.GEOSException:
                # make_valid fails on some of MapBiomas's self-touching polygons ("Overlay input is mixed-dimension")
                bad_geometry += 1
                import pyproj
                from shapely.ops import transform as _tf
                t = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
                rp = _tf(lambda x, y, z=None: t.transform(x, y), shp).buffer(0).centroid
                r = (rp.x / PIXEL_SIZE, -rp.y / PIXEL_SIZE)
            if valid:
                bad["rslearn_centre_valid_polygon"] += not at_centre(r, w)
                continue
            invalid += 1
            d = off(r, w)
            if not at_centre(r, w):
                rep_d.append(round(math.hypot(*d), 1))
            if max(abs(d[0]), abs(d[1])) > crop:
                bad["repaired_centre_outside_crop"] += 1
        rec["centre_route"] = ("gated: a second route to the windows' own centroid (pyproj, shapely per ring) for "
                               "every polygon; rslearn's projection for every valid polygon; for an invalid polygon "
                               f"the repaired polygon's centroid within the central {CROP_PX}-px crop")
        rec["invalid_polygons"] = invalid
        rec["repaired_centre_off_window_centre"] = {
            "n": len(rep_d), "distance_px": sorted(rep_d), "could_not_make_valid_buffer0_used": bad_geometry,
            "note": "invalid polygons: make_valid's (or buffer(0)'s) repaired polygon is not the rings' polygon"}
        rec["ai2_code_letter_off_window_centre"] = {
            "n": len(ai2_d), "distance_px": sorted(ai2_d),
            "note": "shapely's centroid of the unrepaired polygon, as Ai2's deploy code takes it; reported, not gated"}
        rec["centre_misses"] = misses
    else:
        del bad["centre_pixel"]
    rec["bad"] = bad
    rec["ok"] = not any(bad.values()) and len(wins) > 0
    return rec


def imagery_counts(priv, ds=None):
    """Windows whose eight Sentinel-2 item groups are completed (layers/<layer>[.k]/completed,
    rslearn/dataset/storage/file.py:29-32, 231-251), by stratum; how many item groups rslearn's prepare matched per
    layer (items.json); and the windows whose output layer is completed, which rslearn predicts exactly where all
    eight are (load_all_layers: rslearn/train/dataset.py:852-886, 905-943). Counts only."""
    ds = ds or os.path.join(priv, "dataset")
    units = read_sample(priv)["units"]
    out = {s: {"windows": 0, "all_eight": 0, "pre_four": 0, "post_four": 0, "none": 0, "with_output": 0,
               "output_without_all_eight": 0, "prepared_groups": {n: {} for n in S2_LAYERS}} for s in STRATUM_NAMES}
    for u in units:
        wd = os.path.join(ds, "windows", GROUP, u["window"])
        if not os.path.isdir(wd):
            continue
        r = out[u["stratum"]]
        r["windows"] += 1
        done = [os.path.exists(os.path.join(wd, "layers", n, "completed")) for n in FLD_LAYERS]
        r["all_eight"] += all(done)
        r["pre_four"] += all(done[:4])
        r["post_four"] += all(done[4:])
        r["none"] += not any(done)
        o = os.path.exists(os.path.join(wd, "layers", OUTPUT_LAYER, "completed"))
        r["with_output"] += o
        r["output_without_all_eight"] += o and not all(done)
        try:
            with open(os.path.join(wd, "items.json"), encoding="utf-8") as f:
                items = json.load(f)
        except (OSError, ValueError):
            items = []
        got = {e.get("layer_name"): len(e.get("serialized_item_groups") or []) for e in items if isinstance(e, dict)}
        for n in S2_LAYERS:
            k = str(got.get(n, "unprepared"))
            r["prepared_groups"][n][k] = r["prepared_groups"][n].get(k, 0) + 1
    tot = {k: sum(out[s][k] for s in STRATUM_NAMES) for k in ("windows", "all_eight", "pre_four", "post_four", "none",
                                                                "with_output", "output_without_all_eight")}
    # the after period of an alert detected after early April 2026 was still open when prepare ran: which scenes it
    # took depends on that date, so the record carries it
    return {"by_stratum": out, "total": tot, "counted_utc": utc_now(),
            "graded_with_all_eight": sum(out[s]["all_eight"] for s in GRADED), "floor": FLOORS["min_with_prediction"]}


SMOKE_SITES = (("s0", -55.40, -11.90, "2025-08-15"), ("s1", -63.10, -9.20, "2025-09-01"))   # synthetic, no alert


def smoke_dataset(ds, config=None):
    """Two synthetic windows for the GPU smoke (no MapBiomas alert; the page allows the pipeline's smoke on synthetic
    inputs): Ai2's window around two fixed points of the Brazilian Amazon, dated 2025."""
    if os.path.isdir(os.path.join(ds, "windows")):
        raise ValueError(f"{ds} already holds windows/")
    write_dataset_config(ds, config)
    for name, lon, lat, date in SMOKE_SITES:
        x, y = mercator(lon, lat)
        write_window(ds, "smoke", name, window_bounds(float(x), float(y)), event_time(date))
    return {"windows": len(SMOKE_SITES), "group": "smoke"}


# ----------------------------------------------------------------------------- the predictions
def read_probs(probs_dir):
    """from-olmoearth's per-window read: (P (N, 10), window ids). Refused unless ten classes summing to 1 and the class
    each feature carries is FLD_CLASSES[argmax] (Ai2's order)."""
    import csv
    P = np.load(os.path.join(probs_dir, "window_probs.npy"))
    if P.ndim != 3 or P.shape[0] != N_CLASSES or P.shape[1] != 1:
        raise ValueError(f"window_probs.npy has shape {P.shape}, not ({N_CLASSES}, 1, N)")
    P = P[:, 0, :].T.astype(np.float64)
    with open(os.path.join(probs_dir, "window_probs.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(P):
        raise ValueError("window_probs.csv and window_probs.npy disagree on the number of predictions")
    names, wrong_order = [], 0
    for i, r in enumerate(rows):
        if int(r["window_col"]) != i:
            raise ValueError("window_probs.csv is not in column order")
        names.append(r["window"])
        wrong_order += r["class"] != FLD_CLASSES[int(np.argmax(P[i]))]
    if wrong_order:
        raise ValueError(f"{wrong_order} prediction(s) carry a class other than FLD_CLASSES[argmax]: the probabilities "
                         "are not in Ai2's class order")
    s = P.sum(1)
    if len(P) and (float(np.abs(s - 1).max()) > SUM_TOLERANCE or P.min() < -1e-6 or P.max() > 1 + 1e-6):
        raise ValueError("the probabilities are not a softmax")
    if len(set(names)) != len(names):
        raise ValueError("a window holds more than one prediction")
    return P, names


def suspicion(P):
    """Minus the log of the top-1 probability: signals.confidence form "top1" on the log probabilities (all ten
    classes trained)."""
    L = np.log(np.maximum(np.asarray(P, np.float64), FLOOR_P)).T[:, :, None]
    return signals.confidence(L, form="top1").ravel()


# ----------------------------------------------------------------------------- the measures
def ranking(u, err, w, strata, n_boot, seed=SEED, budgets=BUDGETS):
    """exp99_transfer.ranking's measures on the graded alerts: AUROC, capture, the ceiling and the gap closed,
    design-weighted and unweighted; bootstraps of the weighted AUROC and of the gap closed at 10%, alerts resampled
    within each stratum, weights kept."""
    u, err, w = (np.asarray(a, dtype=np.float64) for a in (u, err, w))
    n, k = err.size, int(err.sum())
    if k == 0 or k == n:
        return {"n": n, "n_errors": k, "note": "no ranking: no error or no correct alert"}

    def block(ww):
        theta = metrics.weighted_mean(err, ww)
        cap = metrics.weighted_capture_at_budget(u, err, ww, budgets)
        ceil = {b: min(1.0, b / theta) for b in budgets}
        return {"auroc": metrics.weighted_auroc(u, err, ww), "error_rate": theta,
                "capture": {bkey(b): cap[b] for b in budgets}, "random": {bkey(b): b for b in budgets},
                "ceiling": {bkey(b): ceil[b] for b in budgets},
                "gap_closed": {bkey(b): (cap[b] - b) / (ceil[b] - b) if ceil[b] > b + EPS else None for b in budgets},
                "aurc": metrics.weighted_aurc(u, err, ww), "excess_aurc": metrics.weighted_excess_aurc(u, err, ww)}
    out = {"n": n, "n_errors": k, "weighted": block(w), "unweighted": block(np.ones(n))}
    rng = np.random.default_rng(seed)
    st = np.asarray(strata)
    idx = [np.flatnonzero(st == s) for s in sorted(set(st.tolist()))]
    au, gc, skipped = [], [], 0
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(ix, ix.size, replace=True) for ix in idx])
        e_, u_, w_ = err[pick], u[pick], w[pick]
        if not 0 < e_.sum() < e_.size:
            skipped += 1
            continue
        au.append(metrics.weighted_auroc(u_, e_, w_))
        th = metrics.weighted_mean(e_, w_)
        c = metrics.weighted_capture_at_budget(u_, e_, w_, (0.10,))[0.10]
        ce = min(1.0, 0.10 / th)
        if ce > 0.10 + EPS:
            gc.append((c - 0.10) / (ce - 0.10))
    lo_q, hi_q = (1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2
    for name, v in (("auroc_weighted_bootstrap", au), ("gap_closed_0.1_weighted_bootstrap", gc)):
        v = np.sort(np.asarray(v))
        out[name] = {"n_resamples": int(v.size), "skipped": skipped, "resampling": "alerts within each stratum",
                     "seed": seed}
        if v.size:
            out[name].update(lo90=float(np.quantile(v, lo_q)), hi90=float(np.quantile(v, hi_q)))
    return out


def stratified(k_by, n_by, N_by):
    """The package's condition-design interval with the strata as conditions (estimate._union_interval: each
    stratum's exact hypergeometric interval at 1 - 0.05 / L weighted by N_h / N, L the strata not labelled in full;
    at least 95%), and each stratum's own exact 95% interval."""
    names = [s for s in GRADED]
    k = [int(k_by[s]) for s in names]
    n = [int(n_by[s]) for s in names]
    N = [int(N_by[s]) for s in names]
    theta, lo, hi, L = est._union_interval(k, n, N)
    per = {}
    for s, ks, ns, Ns in zip(names, k, n, N):
        a, b = est.hypergeom_interval(ks, ns, Ns) if ns else (0.0, 1.0)
        per[s] = {"N": Ns, "n": ns, "errors": ks, "rate": ks / ns if ns else None, "low": a, "high": b}
    return {"estimate": theta, "low": lo, "high": hi, "strata_in_interval": L, "nominal_coverage": 0.95,
            "weights": {s: Ns / sum(N) for s, Ns in zip(names, N)}, "by_stratum": per,
            "method": "oe_inferencex.estimate._union_interval over the graded strata (exact, union bound)"}


def grade(m, thresholds=THRESHOLDS, floors=FLOORS):
    ok = (m.get("n_with_prediction") or 0) >= floors["min_with_prediction"] and \
        (m.get("n_errors") or 0) >= floors["min_errors"] and (m.get("n_correct") or 0) >= floors["min_correct"]
    why = (f"fewer than {floors['min_with_prediction']} graded alerts with a prediction, {floors['min_errors']} errors "
           f"or {floors['min_correct']} correct")
    rules = {"P1": f"design-weighted AUROC of the suspicion (minus log top-1) for the errors >= {thresholds['P1']}",
             "P2": f"share of the gap from random to the ceiling closed by the least confident 10% of the design "
                   f"weight >= {thresholds['P2']}"}
    values = {"P1": m.get("p1_auroc"), "P2": m.get("p2_gap_closed")}
    out = {}
    for p in PREDICTIONS:
        v = values[p]
        row = {"rule": rules[p], "threshold": thresholds[p], "value": float(v) if finite(v) else None}
        if not finite(v):
            row.update(graded=False, reason="not computable on these inputs")
        elif not ok:
            row.update(graded=False, reason=why)
        else:
            row.update(graded=True, holds=bool(v >= thresholds[p] - EPS))
        out[p] = row
    out["P3"] = {"rule": "descriptive: the design-weighted error rate with the stratified exact interval, each "
                         "stratum's exact interval, Ai2's 76.1% beside them, the confusion counts", "graded": False}
    out["floors"] = dict(floors, n_with_prediction=m.get("n_with_prediction"), n_errors=m.get("n_errors"),
                         n_correct=m.get("n_correct"))
    held = [p for p in PREDICTIONS if out[p].get("graded")]
    if len(held) < len(PREDICTIONS):
        out["reading"] = "below the floors or not computable: the predictions are reported without verdicts"
    else:
        h = [out[p]["holds"] for p in PREDICTIONS]
        out["reading"] = ("P1 and P2 hold: the confidence ranks the errors on a second task" if all(h) else
                          "both fail: the confidence does not rank this model's errors here" if not any(h) else
                          "one holds: the ranking is weaker than a review can use or than the bar asks; both reported")
    return out


def confusion(labels, preds):
    pairs = {}
    for t, p in zip(labels, preds):
        key = f"{FLD_CLASSES[int(t)]} -> {FLD_CLASSES[int(p)]}"
        pairs[key] = pairs.get(key, 0) + 1
    return dict(sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0])))


def analyse(sample, P, names, n_boot=BOOT, floors=FLOORS, thresholds=THRESHOLDS, inside=None, log=print):
    units = sample["units"]
    want = {f"{GROUP}/{u['window']}": k for k, u in enumerate(units)}
    extra = [n for n in names if n not in want]
    if extra:
        raise ValueError(f"{len(extra)} prediction(s) for windows that are not in the sample")
    n = len(units)
    probs = np.full((n, N_CLASSES), np.nan)
    has = np.zeros(n, bool)
    for i, name in enumerate(names):
        probs[want[name]] = P[i]
        has[want[name]] = True
    strat = np.array([u["stratum"] for u in units])
    label = np.array([FLD_CLASSES.index(CROSSWALK[u["driver"]]) for u in units], np.int64)
    pred = np.full(n, -1, np.int64)
    u_ = np.full(n, np.nan)
    p1 = np.full(n, np.nan)
    if has.any():
        pred[has] = probs[has].argmax(1)
        u_[has] = suspicion(probs[has])
        p1[has] = probs[has].max(1)
    err = np.where(has, (pred != label).astype(np.float64), np.nan)
    N_by = {s: int(sample["strata"][s]["N"]) for s in STRATUM_NAMES}
    for s in STRATUM_NAMES:
        if int(sample["strata"][s]["n"]) != int((strat == s).sum()):
            raise ValueError(f"the private sample's {s} stratum lists {(strat == s).sum()} units, its record "
                             f"{sample['strata'][s]['n']}")
    n_by = {s: int((has & (strat == s)).sum()) for s in STRATUM_NAMES}
    weight = np.array([N_by[s] / n_by[s] if has[i] and n_by[s] else 0.0 for i, s in enumerate(strat)])
    G = np.flatnonzero(has & np.isin(strat, GRADED))
    k_by = {s: int(np.nansum(err[(strat == s) & has])) for s in STRATUM_NAMES}
    rec = {"units": {s: {"sampled": int((strat == s).sum()), "with_prediction": n_by[s],
                         "without_prediction": int(((strat == s) & ~has).sum()), "N": N_by[s],
                         "weight_with_prediction": (N_by[s] / n_by[s]) if n_by[s] else None,
                         "graded": s in GRADED} for s in STRATUM_NAMES},
           "weights_rule": "N_h / n_h, n_h the stratum's alerts with a prediction (interpretation c)"}
    if G.size:
        rec["p3_error_rate"] = stratified(k_by, n_by, N_by)
        rec["p3_error_rate"]["unweighted"] = float(err[G].mean())
        rec["p3_error_rate"]["accuracy"] = 1 - rec["p3_error_rate"]["estimate"]
        rec["ai2"] = {"accuracy": AI2_ACCURACY, "error_rate": 1 - AI2_ACCURACY,
                      "on": "Ai2's 109 validation windows (Brazil 75, Colombia 34; ten classes; the windows its checkpoint was chosen on)"}
        rec["confusion"] = {s: confusion(label[(strat == s) & has], pred[(strat == s) & has]) for s in STRATUM_NAMES}
        rec["ranking"] = ranking(u_[G], err[G], weight[G], strat[G], n_boot)
        rec["predicted_classes"] = {s: {FLD_CLASSES[c]: int(((pred == c) & (strat == s)).sum())
                                        for c in range(N_CLASSES) if ((pred == c) & (strat == s)).any()}
                                    for s in STRATUM_NAMES}
        rec["confidence"] = {
            "top1_quantiles": {s: quantiles(p1[(strat == s) & has]) for s in STRATUM_NAMES},
            "weighted_mean_top1": metrics.weighted_mean(p1[G], weight[G]),
            "weighted_accuracy": 1 - metrics.weighted_mean(err[G], weight[G]),
            "reading": "report-only: the mean confidence beside the accuracy on the graded strata"}
    if G.size and inside is not None:
        ins = np.array([inside.get(int(u["alert_code"])) for u in units], dtype=object)
        rec["centroid_inside_polygon"] = {
            "reading": "report-only: the window is centred on the polygon's centroid, as Ai2's deployment centres it; "
                       "where the centroid lies outside the polygon the window's centre shows something else",
            **{name: {"n": int(m.sum()), "errors": int(np.nansum(err[m])),
                      "weighted_error_rate": metrics.weighted_mean(err[m], weight[m]) if m.any() else None}
               for name, m in (("inside", has & np.isin(strat, GRADED) & (ins == True)),     # noqa: E712
                               ("outside", has & np.isin(strat, GRADED) & (ins == False)))}}  # noqa: E712
    R = np.flatnonzero(has & (strat == "roads"))
    rec["roads_reported"] = {"n": int(R.size), "errors": int(np.nansum(err[R])) if R.size else 0,
                             "confusion": confusion(label[R], pred[R]) if R.size else {},
                             "top1": sorted(float(x) for x in p1[R]) if R.size else [],
                             "note": "report-only: fewer than the floors"}
    rk = (rec.get("ranking") or {}).get("weighted") or {}
    m = {"n_with_prediction": int(G.size), "n_errors": int(np.nansum(err[G])) if G.size else 0,
         "p1_auroc": rk.get("auroc"), "p2_gap_closed": (rk.get("gap_closed") or {}).get(bkey(0.10))}
    m["n_correct"] = m["n_with_prediction"] - m["n_errors"]
    rec["prereg"] = grade(m, thresholds, floors)
    sidx = {s: i for i, s in enumerate(STRATUM_NAMES)}
    units_out = {"stratum": np.array([sidx[s] for s in strat], np.int64), "stratum_names": np.array(STRATUM_NAMES),
                 "graded": np.isin(strat, GRADED), "has_prediction": has, "weight": weight, "label": label,
                 "pred": pred, "probs": probs.astype(np.float32), "suspicion": u_, "top1": p1, "error": err,
                 "class_names": np.array(FLD_CLASSES)}
    log(json.dumps({p: rec["prereg"][p].get("value") for p in PREDICTIONS}))
    return rec, units_out


def run_grade(priv, probs_dir, out_dir, status=None, n_boot=BOOT, floors=FLOORS, suffix="", log=print):
    status = prereg_status() if status is None else status
    if status != "frozen":
        raise PermissionError(f"{os.path.relpath(PLAN, ROOT)} is {status}, not frozen: no prediction at a sampled "
                              "alert is read before the page is frozen")
    sample = read_sample(priv)
    P, names = read_probs(probs_dir)
    inside = None
    path = os.path.join(priv, "e104_polygons.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            polys = {int(k): v for k, v in json.load(f).items()}
        swap = axis_order(polys)[0] == "latlon"
        inside = {k: polygon_facts(v["geometry"], swap=swap)["inside"] for k, v in polys.items()}
    rec, units = analyse(sample, P, names, n_boot=n_boot, floors=floors, inside=inside, log=log)
    import exp89_finetuned_checkpoints as e89
    commit, dirty = e89.git_state()
    rec = {"experiment": "exp104: Ai2's Forest Loss Driver model graded at 2026 MapBiomas Alerta alerts (Amazon)",
           "prereg_page": os.path.relpath(PLAN, ROOT), "prereg_status": status, "utc": utc_now(), "commit": commit,
           "dirty": dirty, "model": MODEL, "ai2_config": {k: AI2[k] for k in ("repo", "commit", "dir")},
           "crosswalk": CROSSWALK, "frame_sha256_prefix": sample.get("frame_sha256_prefix"),
           "thresholds": THRESHOLDS, "floors": floors, "bootstrap": {"n": n_boot, "seed": SEED, "level": LEVEL},
           "cannot_show": list(CANNOT), **rec}
    codes = read_frame_codes(priv) or [u["alert_code"] for u in sample["units"]]
    os.makedirs(out_dir, exist_ok=True)
    dump(rec, os.path.join(out_dir, f"exp104_summary{suffix}.json"), codes=codes)
    units = shuffle_units(units, priv)
    save_units(units, os.path.join(out_dir, f"exp104_units{suffix}.npz"), codes=codes)
    return rec, units


# ----------------------------------------------------------------------------- the synthetic smoke
# A synthetic snapshot: 120 agriculture, 25 garimpo, 5 mining, 3 roads, 8 others and 1 without a driver in the frame
# (two of the agriculture alerts list Amazônia,Cerrado), plus alerts outside it: 10 Cerrado, 2 cancelled, 4 detected in
# July. The sample (seed 104): 40 of 120 agriculture (weight 3), all 30 mining, all 3 roads. One agriculture alert has
# no polygon; two agriculture alerts and one mining alert get no imagery (no prediction). The stand-in model: an error
# with probability 0.3 (rng 104), its top-1 probability 0.20 + 0.50 U for an error and 0.35 + 0.60 U for a correct one.
SMOKE_N = {"agriculture": 120, "garimpo": 25, "mining": 5, "roads": 3, "others": 8, None: 1}
SMOKE_SIZES = {"agriculture": 40}
SMOKE_FLOORS = {"min_with_prediction": 50, "min_errors": 5, "min_correct": 5}
SMOKE_CODE0 = 9_100_000
SMOKE_NO_POLYGON = 1          # agriculture units (from the first) without a polygon
SMOKE_NO_IMAGERY = {"agriculture": 2, "mining": 1}
SMOKE_CENTRE = (-60.0, -8.0)  # synthetic: the polygons are laid out on a 0.05-degree lattice from here


def smoke_snapshot(path):
    feats, k = [], 0

    def add(driver, biomes="Amazônia", status="published", date="2026-03-15", area=5.0):
        nonlocal k
        feats.append({"type": "Feature", "properties": {
            "alert_code": SMOKE_CODE0 + 7 * k, "area_ha": area, "biomes": biomes,
            "detected_at": date, "image_after_at": date, "deforestation_driver": driver, "status_name": status}})
        k += 1
    for d, n in SMOKE_N.items():
        for i in range(n):
            add(d, biomes="Amazônia,Cerrado" if d == "agriculture" and i < 2 else "Amazônia",
                date=f"2026-0{1 + i % 6}-{1 + i % 28:02d}", area=2.0 + (i % 9))
    for i in range(10):
        add("agriculture", biomes="Cerrado")
    for i in range(2):
        add("agriculture", status="cancelled")
    for i in range(4):
        add("garimpo", date="2026-07-02")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(feats))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": [feats[i] for i in order], "timeStamp": "synthetic"}, f)
    expect_drivers = {driver_key(d): n for d, n in SMOKE_N.items()}
    frame_codes = [p["properties"]["alert_code"] for p in feats if in_frame(p["properties"])]
    return path, (sha16(frame_codes), sum(SMOKE_N.values()), expect_drivers)


def smoke_polygon(k, area_ha, kind):
    """A synthetic polygon of about area_ha, centred on the k-th lattice point; kind 'square', 'u' (concave, its
    centroid outside), 'holed' or 'multi'."""
    lon0, lat0 = SMOKE_CENTRE[0] + 0.05 * (k % 20), SMOKE_CENTRE[1] - 0.05 * (k // 20)
    side = math.sqrt(area_ha * 1e4) / 111_320.0
    h = side / 2
    sq = [[lon0 - h, lat0 - h], [lon0 + h, lat0 - h], [lon0 + h, lat0 + h], [lon0 - h, lat0 + h], [lon0 - h, lat0 - h]]
    if kind == "u":
        t = side / 4
        ring = [[lon0 - h, lat0 - h], [lon0 + h, lat0 - h], [lon0 + h, lat0 + h], [lon0 + h - t, lat0 + h],
                [lon0 + h - t, lat0 - h + t], [lon0 - h + t, lat0 - h + t], [lon0 - h + t, lat0 + h],
                [lon0 - h, lat0 + h], [lon0 - h, lat0 - h]]
        return {"type": "Polygon", "coordinates": [ring]}
    if kind == "holed":
        q = side / 8
        hole = [[lon0 - q, lat0 - q], [lon0 - q, lat0 + q], [lon0 + q, lat0 + q], [lon0 + q, lat0 - q],
                [lon0 - q, lat0 - q]]
        return {"type": "Polygon", "coordinates": [sq, hole]}
    if kind == "multi":
        off = side
        sq2 = [[x + off * 1.5, y] for x, y in sq]
        return {"type": "MultiPolygon", "coordinates": [[sq], [sq2]]}
    return {"type": "Polygon", "coordinates": [sq]}


SMOKE_KINDS = ("square", "square", "u", "holed", "multi")


def smoke_wfs(units, latlon=False):
    """A stand-in for the WFS: a FeatureCollection per request, the polygons of the codes asked for (one left out)."""
    polys = {}
    for k, u in enumerate(units):
        if u["stratum"] == "agriculture" and k < SMOKE_NO_POLYGON:
            continue
        g = smoke_polygon(k, float(u["area_ha"]), SMOKE_KINDS[k % len(SMOKE_KINDS)])
        f = polygon_facts(g)
        lat = math.degrees(2 * math.atan(math.exp(f["cy"] / EARTH_R)) - math.pi / 2)
        lng = math.degrees(f["cx"] / EARTH_R)
        if latlon:
            g = json.loads(json.dumps(g))
            for part in parts_of(g):
                for ring in part:
                    for pt in ring:
                        pt[0], pt[1] = pt[1], pt[0]
        polys[int(u["alert_code"])] = {"geometry": g, "lat": f"{lat:.6f}", "lng": f"{lng:.6f}"}

    def get(url):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        cql = q["CQL_FILTER"][0]
        asked = [int(s) for s in cql.replace("alert_code IN (", "").replace(")", "").split(",")]
        return {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": polys[c]["geometry"],
             "properties": {"alert_code": c, "lat": polys[c]["lat"], "lng": polys[c]["lng"]}}
            for c in asked if c in polys]}
    return get


def smoke_predictions(priv, seed=SEED):
    """The stand-in for `rslearn model predict`: data.geojson in layers/output/ of each window with imagery (all eight
    Sentinel-2 item groups marked completed), carrying new_label and the ten probabilities as rslearn's
    ClassificationTask writes them (classification.py:186-235). Returns the planted truth per window."""
    units = read_sample(priv)["units"]
    ds = os.path.join(priv, "dataset")
    rng = np.random.default_rng(seed)
    skip = dict(SMOKE_NO_IMAGERY)
    truth = {}
    for u in units:
        wd = os.path.join(ds, "windows", GROUP, u["window"])
        if not os.path.isdir(wd):
            continue
        if skip.get(u["stratum"], 0) > 0:
            skip[u["stratum"]] -= 1
            for n in FLD_PRE:                                  # the pre period found its four scenes, the post did not
                os.makedirs(os.path.join(wd, "layers", n), exist_ok=True)
                open(os.path.join(wd, "layers", n, "completed"), "w").close()
            continue
        for n in FLD_LAYERS:
            os.makedirs(os.path.join(wd, "layers", n), exist_ok=True)
            open(os.path.join(wd, "layers", n, "completed"), "w").close()
        lab = FLD_CLASSES.index(STRATUM_CLASS[u["stratum"]])
        wrong = rng.random() < 0.3
        t = 0.20 + 0.50 * rng.random() if wrong else 0.35 + 0.60 * rng.random()
        pred = int((lab + 1 + rng.integers(0, N_CLASSES - 1)) % N_CLASSES) if wrong else lab
        p = np.full(N_CLASSES, (1 - t) / (N_CLASSES - 1))
        p[pred] = t
        p = (p / p.sum()).astype(np.float32).astype(np.float64)
        ld = os.path.join(wd, "layers", OUTPUT_LAYER)
        os.makedirs(ld, exist_ok=True)
        with open(os.path.join(ld, "data.geojson"), "w") as f:
            json.dump({"type": "FeatureCollection", "properties": {}, "features": [
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]},
                 "properties": {CLASS_PROPERTY: FLD_CLASSES[pred], PROB_PROPERTY: p.tolist()}}]}, f)
        open(os.path.join(ld, "completed"), "w").close()
        truth[u["window"]] = {"stratum": u["stratum"], "label": lab, "pred": pred, "top1": float(p.max())}
    return truth


def smoke(out_dir=None, n_boot=SMOKE_BOOT, log=print, latlon=False):
    """Synthetic snapshot -> frame -> sample -> polygons (a stand-in WFS) -> windows -> configs -> stand-in predictions
    -> from-olmoearth's reader -> the grade (status "frozen" given here: synthetic inputs only)."""
    from oe_inferencex import olmoearth as oe
    out_dir = out_dir or OUT
    with tempfile.TemporaryDirectory() as tmp:
        priv = os.path.join(tmp, "private")
        snap, expect = smoke_snapshot(os.path.join(tmp, "snapshot.json"))
        frec, units = run_frame(priv, snapshot=snap, expect=expect, sizes=SMOKE_SIZES, log=log,
                                record=os.path.join(tmp, "e104_frame_smoke.json"))
        srec = frec["sample"]
        wrec = run_windows(priv, record=os.path.join(tmp, "e104_windows_smoke.json"), log=log,
                           get=smoke_wfs(units, latlon=latlon))
        crec = check_configs()
        truth = smoke_predictions(priv)
        img = imagery_counts(priv)
        s = oe.read_window_probs(os.path.join(priv, "dataset"), os.path.join(priv, "probs"), layer=OUTPUT_LAYER,
                                 prob_property=PROB_PROPERTY, class_property=CLASS_PROPERTY)
        rec, u_out = run_grade(priv, os.path.join(priv, "probs"), out_dir, status="frozen", n_boot=n_boot,
                               floors=SMOKE_FLOORS, suffix="_smoke", log=log)
        with open(os.path.join(tmp, "e104_frame_smoke.json"), encoding="utf-8") as f:
            frame_rec = json.load(f)
        codes = read_frame_codes(priv)
    rec["mode"] = "smoke: synthetic inputs, floors lowered to " + json.dumps(SMOKE_FLOORS)
    rec["smoke"] = {"frame": frame_rec["frame"], "sample": srec, "windows": wrec, "configs": crec, "imagery": img,
                    "reader": {"windows_read": s["windows_read"], "features": s["features"],
                               "class_property": s["class_property"]},
                    "planted": {"n": len(truth), "errors": sum(t["label"] != t["pred"] for t in truth.values())}}
    dump(rec, os.path.join(out_dir, "exp104_summary_smoke.json"), codes=codes)
    return rec, u_out, truth


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="synthetic inputs end to end")
    g.add_argument("--frame", action="store_true", help="the frame and the sample (attributes only)")
    g.add_argument("--windows", action="store_true", help="the sampled alerts' polygons and the rslearn windows")
    g.add_argument("--grade", action="store_true", help="the grade; refused until the page is frozen")
    g.add_argument("--check-configs", action="store_true", help="the committed configs against Ai2's pins")
    g.add_argument("--check-dataset", metavar="DS", help="rslearn 0.1.17 reads DS (run with exp102's venv)")
    g.add_argument("--imagery", action="store_true", help="windows with complete imagery, by stratum")
    g.add_argument("--smoke-dataset", metavar="DS", help="two synthetic windows for the GPU smoke")
    g.add_argument("--check-probs", metavar="DIR", help="the GPU smoke's from-olmoearth read: ten classes in Ai2's "
                                                         "order, a softmax, one prediction per window (counts)")
    ap.add_argument("--private-dir", default=None, help="outside every git checkout: codes, polygons, the dataset")
    ap.add_argument("--snapshot", default=None, help="the attribute snapshot (JSON) for --frame")
    ap.add_argument("--fetch", action="store_true", help="--frame: fetch the attributes from the WFS instead")
    ap.add_argument("--polygons", default=None, help="--windows: a polygons file instead of the WFS")
    ap.add_argument("--redo", action="store_true", help="--windows: refetch and rebuild")
    ap.add_argument("--config", default=None, help="the dataset config (default exp/jobs/e104_config/dataset.json)")
    ap.add_argument("--probs", default=None, help="--grade: from-olmoearth's output directory (window_probs.*)")
    ap.add_argument("--record", default=None, help="where a mode's counts go")
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if a.smoke:
        rec, _, _ = smoke(a.out_dir)
        print(json.dumps(rec["prereg"], indent=1))
        return 0
    if a.check_configs:
        rec = check_configs()
        print(json.dumps(rec))
        if a.record:
            dump(rec, a.record)
        return 0 if rec["ok"] else 1
    if a.smoke_dataset:
        print(json.dumps(smoke_dataset(a.smoke_dataset, a.config)))
        return 0
    if a.check_probs:
        P, names = read_probs(a.check_probs)
        rec = {"predictions": len(names), "classes": N_CLASSES, "order": "Ai2's (checked)",
               "largest_sum_deviation": float(np.abs(P.sum(1) - 1).max()) if len(P) else None}
        rec["ok"] = bool(len(names) == len(SMOKE_SITES) and rec["largest_sum_deviation"] is not None
                         and rec["largest_sum_deviation"] <= 1e-3)
        print(json.dumps(rec))
        if a.record:
            dump(rec, a.record)
        return 0 if rec["ok"] else 1
    if a.check_dataset:
        rec = check_dataset(a.check_dataset, a.private_dir, a.workers)
        print(json.dumps(rec))
        if a.record:
            dump(rec, a.record, codes=read_frame_codes(a.private_dir) if a.private_dir else ())
        return 0 if rec["ok"] else 1
    if not a.private_dir:
        ap.error("--private-dir is required")
    if a.frame:
        run_frame(a.private_dir, snapshot=a.snapshot, fetch=a.fetch, record=a.record)
        return 0
    if a.windows:
        rec = run_windows(a.private_dir, polygons=a.polygons, config=a.config, record=a.record, redo=a.redo)
        return 0 if rec["n_windows"] else 1
    if a.imagery:
        rec = imagery_counts(a.private_dir)
        print(json.dumps(rec["total"]), json.dumps({s: v["all_eight"] for s, v in rec["by_stratum"].items()}))
        if a.record:
            dump(rec, a.record, codes=read_frame_codes(a.private_dir))
        return 0
    if a.grade:
        if not a.probs:
            ap.error("--grade needs --probs")
        status = prereg_status()
        if status != "frozen":
            print(f"refused: {os.path.relpath(PLAN, ROOT)} is {status}, not frozen; no prediction at a sampled "
                  "alert is read before the page is frozen")
            return 3
        rec, _ = run_grade(a.private_dir, a.probs, a.out_dir, status=status)
        print(json.dumps(rec["prereg"], indent=1))
        return 0
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001  (a job log needs the traceback, and the exit code must fail the job)
        traceback.print_exc()
        sys.exit(1)
