"""The East Africa TimeSync sample read against the AWF legend: the crosswalk, fixed on 9 October 2026 before any map
of either experiment was read, and what exp98 (Part I) and exp99 share to read the sample. Preregistered in
docs/plan/awf_transfer.md (exp99) and docs/plan/awf_deployment.md (exp98, Part I).

The sample (Bullock et al. 2021, Land 10, 150; RCMRD and SERVIR; github.com/bullocke/eastafrica, CC0). In each of
seven countries 2,000 locations were drawn as a simple random sample, and local experts interpreted each one in
TimeSync from its Landsat history and Google Earth's high-resolution imagery, one label per year, 1985 to 2017 in the
file. Each plot is a point; its label describes the Landsat-scale plot around it.

The two legends (definitions quoted from Bullock et al. 2021, Table 1; AWF's from Ai2's olmoearth_projects docs/awf.md
at f3c9b0c8, which names the nine classes and defines only woodland forest, ">40% canopy"):

  TimeSync                                                        AWF (FT-AWF's channels 0-8; channel 9 is no-data)
  Dense Forest       tree-covered, canopy over 40%                0 woodland_forest          canopy over 40%
  Open Forest        tree-covered, canopy 15-40%                  1 open_water
  Wooded Grassland   natural, below both forest thresholds, with  2 shrubland_savanna
                     shrubs, very sparse trees or single trees    3 herbaceous_wetland
  Open Grassland     natural, below both forest thresholds,       4 grassland_barren
                     without substantial woody vegetation         5 agriculture_settlement
  Cropland           primarily agriculture, incl. agroforestry    6 montane_forest
                     and fallow                                   7 lava_forest
  Settlements        primarily built: buildings, roads            8 urban_dense_development
  Open Water         lakes, rivers, reservoirs, oceans
  Vegetated Wetland  water part of the year, a mix of vegetation
                     and open water: marshes, swamps, peatlands
  Otherland          barren land, rock outcrops, permanent snow,
                     beaches, salt crusts

STRICT maps each TimeSync class to the one AWF class whose definition is closest, and the map is right at a plot only
when it predicts that class. LENIENT lets a TimeSync class accept every AWF class whose definition overlaps its own, so
that a disagreement under LENIENT is one no reading of the two legends removes. STRICT's class is always in LENIENT's
set, so LENIENT's errors are a subset of STRICT's, and the two error rates bound what the crosswalk can do to the
answer. The reasons, class by class, are in WHY (and in the preregistration pages):

- montane_forest and lava_forest have no TimeSync counterpart: AWF splits forest by setting (mountain slopes, lava
  flows), TimeSync by canopy cover. Neither is any class's STRICT counterpart, so under STRICT a prediction of either is
  always an error; LENIENT accepts both for the two forest classes.
- Open Forest (15-40% canopy) has no AWF counterpart by canopy: woodland_forest needs more than 40%, so STRICT takes the
  AWF class of woody cover below that, shrubland_savanna; LENIENT also accepts the three forest classes.
- Otherland has no AWF class of its own: AWF merges barren land with grassland, so STRICT is grassland_barren.
- urban_dense_development is Settlements' STRICT counterpart (both name land primarily built); LENIENT adds
  agriculture_settlement, AWF's class of farmland with homesteads.
- Channel 9, the untrained no-data channel, is never accepted.

The plots are read at the 10 m map pixel that holds the plot's point. Nothing here, and nothing the experiments write
from it, holds a coordinate: plots are counted, never listed, and the request geometry exists only in memory and, for
exp99, in a file on the cluster's scratch.

    python exp/timesync_awf_crosswalk.py fetch --dest data/breadth     # the cluster jobs: fetch and check both files
    python exp/timesync_awf_crosswalk.py table                         # the crosswalk and its sha256
"""
import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------------- the legends
TIMESYNC_CLASSES = ("Dense Forest", "Open Forest", "Wooded Grassland", "Open Grassland", "Cropland", "Settlements",
                    "Open Water", "Vegetated Wetland", "Otherland")
AWF_CLASSES = {0: "woodland_forest", 1: "open_water", 2: "shrubland_savanna", 3: "herbaceous_wetland",
               4: "grassland_barren", 5: "agriculture_settlement", 6: "montane_forest", 7: "lava_forest",
               8: "urban_dense_development"}
AWF_INDEX = {v: k for k, v in AWF_CLASSES.items()}
UNTRAINED = 9

STRICT = {"Dense Forest": "woodland_forest",
          "Open Forest": "shrubland_savanna",
          "Wooded Grassland": "shrubland_savanna",
          "Open Grassland": "grassland_barren",
          "Cropland": "agriculture_settlement",
          "Settlements": "urban_dense_development",
          "Open Water": "open_water",
          "Vegetated Wetland": "herbaceous_wetland",
          "Otherland": "grassland_barren"}
LENIENT = {"Dense Forest": ("woodland_forest", "montane_forest", "lava_forest"),
           "Open Forest": ("shrubland_savanna", "woodland_forest", "montane_forest", "lava_forest"),
           "Wooded Grassland": ("shrubland_savanna", "grassland_barren"),
           "Open Grassland": ("grassland_barren", "shrubland_savanna"),
           "Cropland": ("agriculture_settlement",),
           "Settlements": ("urban_dense_development", "agriculture_settlement"),
           "Open Water": ("open_water", "herbaceous_wetland"),
           "Vegetated Wetland": ("herbaceous_wetland", "open_water"),
           "Otherland": ("grassland_barren", "open_water")}
RULES = {"strict": {k: (v,) for k, v in STRICT.items()}, "lenient": LENIENT}
WHY = {
    "Dense Forest": "Both legends set this class at a canopy over 40% (TimeSync's Dense Forest, AWF's woodland forest). "
                    "AWF also has montane and lava forest, forests named by setting rather than canopy, which a "
                    "TimeSync forest on a mountain slope or a lava flow can be; LENIENT accepts both.",
    "Open Forest": "A canopy of 15-40% is below AWF's woodland forest (over 40%), and AWF has no class for it; its "
                   "class of woody cover below that is shrubland/savanna. LENIENT adds the three forest classes: two "
                   "interpreters read a canopy near 40% differently, and montane and lava forest carry no published "
                   "canopy threshold.",
    "Wooded Grassland": "Natural land with shrubs or very sparse trees below both forest thresholds is a savanna or "
                        "shrubland. LENIENT adds grassland/barren, since AWF publishes no woody-cover threshold between "
                        "its shrubland/savanna and grassland classes and a sparse-shrub grassland meets both. "
                        "Woodland forest is not accepted even by LENIENT: this class is below 15% canopy and AWF's "
                        "woodland forest above 40%, so the definitions are disjoint.",
    "Open Grassland": "Grass, grass-like plants and forbs without substantial woody vegetation is AWF's "
                      "grassland/barren. LENIENT adds shrubland/savanna for the same unpublished boundary between the "
                      "two AWF classes.",
    "Cropland": "Land used primarily for agriculture is AWF's agriculture/settlement, the only AWF class that names "
                "agriculture; LENIENT accepts nothing else.",
    "Settlements": "Land primarily built (buildings, roads) is AWF's urban/dense development. LENIENT adds "
                   "agriculture/settlement, where AWF puts homesteads among fields.",
    "Open Water": "Open water. LENIENT adds herbaceous wetland: the region's lakes and swamps are seasonal, and a water "
                  "body vegetated for part of the year meets both definitions.",
    "Vegetated Wetland": "A mix of vegetation and open water for part of the year is AWF's herbaceous wetland; by the "
                         "same mix LENIENT adds open water.",
    "Otherland": "Barren land, rock outcrops and salt crusts: AWF merges barren land with grassland. LENIENT adds open "
                 "water for the salt crusts of seasonal lake beds, which are water part of the year.",
}
AWF_ONLY = {"montane_forest": "no TimeSync counterpart (TimeSync splits forest by canopy, AWF by setting): under STRICT "
                              "a prediction of it is always an error; LENIENT accepts it for Dense and Open Forest",
            "lava_forest": "as montane_forest",
            "channel 9": "AWF's untrained no-data channel: never accepted, an error under both rules"}
TIMESYNC_ONLY = {"Open Forest": "no AWF class of 15-40% canopy: STRICT takes shrubland_savanna",
                 "Otherland": "no AWF class of its own: AWF merges barren land with grassland (grassland_barren)"}


def crosswalk_record():
    """The crosswalk as written in every summary, with its sha256, so a later change cannot pass unseen."""
    rec = {"strict": dict(STRICT), "lenient": {k: list(v) for k, v in LENIENT.items()},
           "awf_classes": {str(k): v for k, v in AWF_CLASSES.items()}, "untrained_channel": UNTRAINED}
    rec["sha256"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
    rec["why"] = dict(WHY)
    rec["awf_only"] = dict(AWF_ONLY)
    rec["timesync_only"] = dict(TIMESYNC_ONLY)
    return rec


def accepted(rule):
    """{TimeSync class: frozenset of accepted AWF channel indices} for 'strict' or 'lenient'."""
    return {t: frozenset(AWF_INDEX[a] for a in v) for t, v in RULES[rule].items()}


def wrong(labels, pred, rule):
    """1.0 where the map's class (an index 0-9 over the decoder's channels) is not accepted for the plot's TimeSync
    label under `rule`, 0.0 where it is. Channel 9 is never accepted."""
    acc = accepted(rule)
    unknown = sorted({str(t) for t in labels} - set(acc))
    if unknown:
        raise ValueError(f"TimeSync labels outside the crosswalk: {unknown}")
    return np.array([0.0 if int(p) in acc[str(t)] else 1.0 for t, p in zip(labels, pred)], dtype=np.float64)


def _check_crosswalk():
    assert set(STRICT) == set(TIMESYNC_CLASSES) == set(LENIENT)
    for t in TIMESYNC_CLASSES:
        assert STRICT[t] in LENIENT[t], t                 # LENIENT's errors are a subset of STRICT's
        assert set(LENIENT[t]) <= set(AWF_INDEX), t
        assert len(set(LENIENT[t])) == len(LENIENT[t]), t


_check_crosswalk()

# ----------------------------------------------------------------------------- the pinned sources
TIMESYNC = {"file": "eastafrica_yearly_point_data.csv",
            "url": "https://raw.githubusercontent.com/bullocke/eastafrica/fc2014fcbc55cb75e0fbf801ffd9ca1ba4dc5c02/"
                   "yearly_point_data.csv",
            "repository": "github.com/bullocke/eastafrica @ fc2014fcbc55cb75e0fbf801ffd9ca1ba4dc5c02, yearly_point_data.csv",
            "sha256": "5309a982cdd45a4e7378008b9a6385b6accca7d071bdd652360ef7ef66fc429a",
            "git_blob_sha1": "1713496168b9a4d1c2c1bedbd62dc7f6b72d7b61", "bytes": 26312839, "licence": "CC0-1.0",
            "citation": "Bullock, E.L., Healey, S.P., Yang, Z., et al. (2021) Three decades of land cover change in East "
                        "Africa. Land 10(2), 150. doi:10.3390/land10020150"}
BOUNDARIES = {"file": "ne_10m_admin_0_countries.geojson",
              "url": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
                     "f1890d9f152c896d250a77557a5751a93d494776/geojson/ne_10m_admin_0_countries.geojson",
              "repository": "github.com/nvkelso/natural-earth-vector v5.1.2 (f1890d9f152c896d250a77557a5751a93d494776), "
                            "geojson/ne_10m_admin_0_countries.geojson",
              "sha256": None,                 # not downloaded to the Mac; pinned by its git blob and size instead
              "git_blob_sha1": "5ebc66e25fc1af01edaebe9375c546655e04cf1e", "bytes": 13287234,
              "licence": "public domain (Natural Earth)"}
SOURCES = {"timesync": TIMESYNC, "boundaries": BOUNDARIES}
# Ai2's AWF request geometry, olmoearth_projects @ f3c9b0c8 olmoearth_run_data/awf/prediction_request_geometry.geojson;
# read from the pinned clone on the cluster (exp/jobs/e98_env.sh) and never copied into this repository
AWF_GEOMETRY_SHA256 = "fee3ce01008d26c6f3c239e68be70851035eb157d5da70bd72cb46dbc8623784"
COUNTRIES = {"kenya": "KEN", "tanzania": "TZA"}       # the CSV's country names, Natural Earth's ADM0_A3
SAMPLE_PER_COUNTRY = 2000                            # Bullock et al. 2021: a simple random sample of 2,000 per country
DISTANCE_CRS = "EPSG:32737"   # WGS 84 / UTM 37S: the zone of Ai2's request geometry and of its label windows
AREA_CRS = "EPSG:6933"        # WGS 84 / NSIDC EASE-Grid 2.0 Global, an equal-area projection
DENSIFY_DEG = 0.001           # a GeoJSON edge is straight in longitude and latitude; it is densified before projecting
PIXELS_PER_KM2 = 10_000       # 10 m pixels


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path):
    """git's object id of a file (`git hash-object`): sha1 of 'blob <size>\\0' and the bytes."""
    h = hashlib.sha1()
    h.update(f"blob {os.path.getsize(path)}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path, src):
    """(ok, what was found): the size, the git blob id and, where pinned, the sha256."""
    if not os.path.isfile(path):
        return False, "absent"
    got = {"bytes": os.path.getsize(path), "git_blob_sha1": git_blob_sha1(path), "sha256": sha256_file(path)}
    ok = got["bytes"] == src["bytes"] and got["git_blob_sha1"] == src["git_blob_sha1"] and \
        (src["sha256"] is None or got["sha256"] == src["sha256"])
    return ok, got


def fetch(dest, which=("timesync", "boundaries"), log=print):
    """Each pinned file into `dest` from its public source, unless a verified copy is there: downloaded to a temporary
    file beside it, checked (size, git blob id, sha256 where pinned) and only then renamed into place. Returns
    {name: {path, sha256, git_blob_sha1, bytes, fetched}}."""
    os.makedirs(dest, exist_ok=True)
    out = {}
    for name in which:
        src = SOURCES[name]
        path = os.path.join(dest, src["file"])
        ok, got = verify(path, src)
        fetched = False
        if not ok:
            fd, tmp = tempfile.mkstemp(prefix=src["file"] + ".", suffix=".partial", dir=dest)
            os.close(fd)
            try:
                with urllib.request.urlopen(src["url"], timeout=120) as r, open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                ok, got = verify(tmp, src)
                if not ok:
                    raise ValueError(f"{src['file']} from {src['url']} does not match its pin: {got}")
                os.replace(tmp, path)
                fetched = True
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        out[name] = {"path": path, "fetched": fetched, **got}
        log(f"{src['file']}: {'fetched and ' if fetched else ''}verified, sha256 {got['sha256']}, "
            f"git blob {got['git_blob_sha1']}, {got['bytes']} bytes")
    return out


# ----------------------------------------------------------------------------- reading the sample
def read_plots(path, year, countries=tuple(COUNTRIES), history=True):
    """The plots of `countries` with a label in `year`, as arrays (in memory only; nothing returned is written):
    pid ("country:plotid"), country, label (the year's), lon, lat, and with `history` the number of distinct labels each
    plot carries over every year of the file and over year - 2 to year. Rows are read with the csv module, so the reader
    needs no pandas."""
    import csv
    keep = set(countries)
    rows, hist, recent = {}, {}, {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            c = r["country"]
            if c not in keep:
                continue
            pid = f"{c}:{int(r['plotid'])}"
            y = int(r["year"])
            if history:
                hist.setdefault(pid, set()).add(r["landcover"])
                if year - 2 <= y <= year:
                    recent.setdefault(pid, set()).add(r["landcover"])
            if y == year:
                if pid in rows:
                    raise ValueError(f"two rows for one plot in {year}")
                rows[pid] = (c, r["landcover"], float(r["longitude"]), float(r["latitude"]))
    pids = sorted(rows)
    out = {"pid": np.array(pids, dtype=object),
           "country": np.array([rows[p][0] for p in pids], dtype=object),
           "label": np.array([rows[p][1] for p in pids], dtype=object),
           "lon": np.array([rows[p][2] for p in pids], dtype=np.float64),
           "lat": np.array([rows[p][3] for p in pids], dtype=np.float64)}
    if history:
        out["n_labels_all_years"] = np.array([len(hist[p]) for p in pids], dtype=np.int64)
        out["n_labels_recent"] = np.array([len(recent.get(p, ())) for p in pids], dtype=np.int64)
    bad = sorted({str(t) for t in out["label"]} - set(TIMESYNC_CLASSES))
    if bad:
        raise ValueError(f"labels outside TimeSync's nine classes: {bad}")
    return out


def subset(plots, mask):
    return {k: v[mask] for k, v in plots.items()}


def load_request_geometry(path, sha256=AWF_GEOMETRY_SHA256):
    """Ai2's request geometry as one shapely geometry in WGS84, after its sha256 is checked (sha256=None skips the check,
    for synthetic geometries). Never written anywhere."""
    import shapely
    from shapely.geometry import shape
    if sha256 is not None and sha256_file(path) != sha256:
        raise ValueError(f"{os.path.basename(path)} is not Ai2's AWF request geometry at f3c9b0c8 (sha256 differs)")
    with open(path) as f:
        gj = json.load(f)
    if gj.get("crs") is not None:
        raise ValueError("the request geometry declares a CRS; GeoJSON's default (WGS84) was expected")
    return shapely.union_all([shape(ft["geometry"]) for ft in gj["features"]])


def _transformer(src, dst):
    import pyproj
    return pyproj.Transformer.from_crs(src, dst, always_xy=True)


def _project(geom, src, dst, densify):
    import shapely
    from shapely.ops import transform
    t = _transformer(src, dst)
    return transform(lambda x, y, z=None: t.transform(x, y), shapely.segmentize(geom, densify))


def distance_km(geom, lon, lat):
    """Each plot's distance to the geometry, in km, 0 inside it: measured in EPSG:32737 with the geometry's edges
    densified every 0.001 degree first (a GeoJSON edge is straight in longitude and latitude). Within 130 km of Ai2's
    request geometry this differs from the geodesic distance by at most 0.12 km (checked 9 October 2026)."""
    import shapely
    g = _project(geom, "EPSG:4326", DISTANCE_CRS, DENSIFY_DEG)
    x, y = _transformer("EPSG:4326", DISTANCE_CRS).transform(np.asarray(lon), np.asarray(lat))
    return shapely.distance(g, shapely.points(x, y)) / 1000.0


def area_rule(dist, min_plots, step_km=10, max_km=2000):
    """The smallest buffer D, in steps of `step_km` from 0 (the geometry itself), that holds at least `min_plots`
    plots, and the counts at every step tried: {d_km, n, counts_by_d_km}."""
    counts = {}
    for d in range(0, max_km + step_km, step_km):
        n = int((np.asarray(dist) <= d).sum())
        counts[d] = n
        if n >= min_plots:
            return {"d_km": d, "n": n, "counts_by_d_km": counts}
    raise ValueError(f"no buffer up to {max_km} km holds {min_plots} plots")


def buffer_wgs84(geom, d_km):
    """The d_km buffer of the geometry, built in EPSG:32737 and returned in WGS84 (densified every 500 m)."""
    g = _project(geom, "EPSG:4326", DISTANCE_CRS, DENSIFY_DEG)
    return _project(g.buffer(d_km * 1000.0, quad_segs=64) if d_km > 0 else g, DISTANCE_CRS, "EPSG:4326", 500.0)


def area_km2(geom_wgs84):
    """The area of a WGS84 geometry in km^2, in an equal-area projection (EPSG:6933)."""
    if geom_wgs84.is_empty:
        return 0.0
    return float(_project(geom_wgs84, "EPSG:4326", AREA_CRS, 0.01).area / 1e6)


def country_polygons(path, codes=tuple(COUNTRIES.values())):
    """{ADM0_A3: shapely geometry in WGS84} from Natural Earth's admin-0 countries."""
    import shapely
    from shapely.geometry import shape
    with open(path) as f:
        gj = json.load(f)
    out = {}
    for ft in gj["features"]:
        p = ft.get("properties") or {}
        code = p.get("ADM0_A3") or p.get("adm0_a3")
        if code in codes:
            out[code] = shapely.union_all([out[code], shape(ft["geometry"])]) if code in out else shape(ft["geometry"])
    missing = [c for c in codes if c not in out]
    if missing:
        raise ValueError(f"no feature for {missing} in {os.path.basename(path)}")
    return out


def region_areas(region_wgs84, boundaries):
    """{country name: km^2 of the region inside it}, the region's area, the share of it outside every country named,
    and each country's whole area (for the ratio estimator). Areas in EPSG:6933; no coordinate is returned."""
    polys = country_polygons(boundaries)
    total = area_km2(region_wgs84)
    inside = {name: area_km2(region_wgs84.intersection(polys[code])) for name, code in COUNTRIES.items()}
    return {"region_km2": total, "by_country_km2": inside,
            "share_outside_countries": float(max(0.0, 1 - sum(inside.values()) / total)) if total else None,
            "country_km2": {name: area_km2(polys[code]) for name, code in COUNTRIES.items()}}


def plot_countries_on_boundaries(lon, lat, country, boundaries):
    """How many plots sit inside the Natural Earth polygon of the country their sample was drawn in: the frame of
    Bullock et al. and Natural Earth's boundaries need not agree at a border."""
    import shapely
    polys = country_polygons(boundaries)
    pts = shapely.points(np.asarray(lon), np.asarray(lat))
    inside = np.zeros(len(pts), bool)
    for name, code in COUNTRIES.items():
        m = np.asarray(country) == name
        inside[m] = shapely.contains(polys[code], pts[m])
    return {"n": int(len(pts)), "inside_own_country": int(inside.sum()), "outside_own_country": int((~inside).sum())}


# ----------------------------------------------------------------------------- estimates with countries as strata
def weights_from_areas(areas_km2, n_by_country):
    """Per country: W_c, its share of the region's area, and the design weight of each of its graded plots,
    W_c / n_c (the plots of a country being a simple random sample of its part of the region)."""
    tot = sum(areas_km2.values())
    W = {c: areas_km2[c] / tot for c in areas_km2}
    return W, {c: (W[c] / n_by_country[c] if n_by_country.get(c) else 0.0) for c in areas_km2}


def stratified_exact(k_by, n_by, area_by, conf=0.95):
    """The region's error rate from each country's errors k_c among n_c graded plots, the countries as strata with
    sizes N_c = the region's 10 m pixels in each (area x 10^4): the package's condition-design interval
    (oe_inferencex.estimate._union_interval, which estimate_error_rate uses for strata): each stratum's exact
    hypergeometric interval at 1 - 0.05/L weighted by its share, L the strata not labelled in full; at least 95% by
    construction, conditional on the n_c. With each country's own exact 95% interval beside it."""
    from oe_inferencex import estimate as est
    if abs(conf - 0.95) > 1e-12:
        raise ValueError("the condition design's interval is at 95%")
    names = sorted(area_by)
    N = [max(int(round(area_by[c] * PIXELS_PER_KM2)), int(n_by.get(c, 0))) for c in names]
    k = [int(k_by.get(c, 0)) for c in names]
    n = [int(n_by.get(c, 0)) for c in names]
    theta, lo, hi, L = est._union_interval(k, n, N)
    per = {}
    for c, kc, nc, Nc in zip(names, k, n, N):
        a, b = est.hypergeom_interval(kc, nc, Nc) if nc else (0.0, 1.0)
        per[c] = {"n": nc, "errors": kc, "rate": kc / nc if nc else None, "low": a, "high": b, "pixels": Nc}
    return {"estimate": theta, "low": lo, "high": hi, "strata_in_interval": L, "nominal_coverage": conf,
            "weights": {c: Nc / sum(N) for c, Nc in zip(names, N)},
            "method": "the condition design's interval of oe_inferencex.estimate: each country's exact hypergeometric "
                      "interval at 1 - 0.05/L, weighted by its share of the area (union bound)",
            "by_country": per}


def ratio_estimate(k_by, n_by, country_km2, m=SAMPLE_PER_COUNTRY):
    """Reported beside: the estimator Bullock et al. used for domains that cross national borders (Stehman 2014's
    ratio estimator for a stratified sample, the countries as strata of m plots each): R = sum_c A_c k_c / m over
    sum_c A_c n_c / m, A_c the country's area, with its linearised standard error. It needs no area of the region,
    only of the countries; it is approximate (normal theory), where stratified_exact is exact given the n_c."""
    num = den = 0.0
    for c in country_km2:
        num += country_km2[c] * k_by.get(c, 0) / m
        den += country_km2[c] * n_by.get(c, 0) / m
    if den <= 0:
        return {"estimate": None}
    R = num / den
    var = 0.0
    for c in country_km2:
        kc, nc = k_by.get(c, 0), n_by.get(c, 0)
        dbar = (kc - R * nc) / m
        ss = kc * (1 - R) ** 2 + (nc - kc) * R ** 2              # sum of d_i^2 over the domain's plots, 0 elsewhere
        s2 = (ss - m * dbar ** 2) / (m - 1)
        var += country_km2[c] ** 2 * s2 / m
    se = math.sqrt(max(var, 0.0)) / den
    return {"estimate": R, "standard_error": se, "low_normal": max(0.0, R - 1.959963984540054 * se),
            "high_normal": min(1.0, R + 1.959963984540054 * se),
            "region_km2_estimated_from_plots": {c: country_km2[c] * n_by.get(c, 0) / m for c in country_km2},
            "method": "Stehman (2014) ratio estimator over the countries' samples of 2,000 (as Bullock et al. 2021 for "
                      "ecoregions); normal-theory interval, reported, not exact"}


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="fetch and check the TimeSync sample and Natural Earth's countries")
    f.add_argument("--dest", default=os.path.join(ROOT, "data", "breadth"))
    f.add_argument("--only", choices=tuple(SOURCES), default=None)
    sub.add_parser("table", help="print the crosswalk and its sha256")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        fetch(a.dest, (a.only,) if a.only else tuple(SOURCES))
    else:
        rec = crosswalk_record()
        for t in TIMESYNC_CLASSES:
            print(f"{t:18s} STRICT {STRICT[t]:24s} LENIENT {', '.join(LENIENT[t])}")
        print("sha256", rec["sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
