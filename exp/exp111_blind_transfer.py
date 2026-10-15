"""exp111: the blind transfer test. Ai2's FT-AWF checkpoint (v1) and the record's v1.2 fine-tune (exp102), run through
rslearn 0.1.17 directly (no olmoearth_run, no olmoearth-runner) on small windows that sit on Ai2's deployment lattice,
at East Africa TimeSync 2017 plots of Kenya and Tanzania that no model has been run at. Preregistered in
docs/plan/blind_transfer.md (DRAFT until the owner freezes it); read that page first, this file is the run's
bookkeeping. The grading of P1-P3 is exp/exp111_grade.py (committed with this file, before freezing); this one ends
at the units file.

    python exp/exp111_blind_transfer.py --smoke [--out-dir DIR]          # synthetic plots, windows, outputs, gate, units
    python exp/exp111_blind_transfer.py --select --private-dir PRIV --timesync CSV --request-geometry AI2.geojson
        --boundaries NE.geojson [--record JSON] [--redo]   # exp99's 309, the pool, the draw (SIZES: no option)
    python exp/exp111_blind_transfer.py --windows --private-dir PRIV --labels EXP89_WINDOWS [--deploy-inv JSON
        --deploy-run DIR] [--record JSON] [--redo]
    python exp/exp111_blind_transfer.py --read ROOT MODEL --private-dir PRIV [--record JSON] [--redo]
    python exp/exp111_blind_transfer.py --gate A|B --private-dir PRIV --gate-file JSON [--identity-dir DIR]
        [--device-file JSON]
    python exp/exp111_blind_transfer.py --read-aprime --private-dir PRIV --aprime-plan JSON [--record JSON]
    python exp/exp111_blind_transfer.py --check-draw --private-dir PRIV  # draw.json against the frozen page's sha256
    python exp/exp111_blind_transfer.py --units --private-dir PRIV --gate-file JSON --out-dir DIR   # refused until frozen

Reruns are safe: --select keeps an existing draw.json and gate17.json when this run's draw is the same (sha256 of the
sorted ids, sizes, seed, order) and refuses a different one unless --redo; --windows keeps a root whose config and
windows equal this run's (the prepared imagery stays) and refuses a different one unless --redo, which removes it.
--redo is refused once the page is frozen (it would replace the frozen draw, or remove blind17 with phase 2's layers).

The selection (--select; allowed before freezing, run on the cluster, never on the Mac):
  1. exp99's area rule on the pinned TimeSync file and Ai2's pinned request geometry (exp99_transfer.select, which
     refuses anything but 309 plots: 219 Kenya, 90 Tanzania, 47 inside). These are the gate-B plots.
  2. The pool: every Kenya and Tanzania plot with a 2017 label outside the 309, refused unless 3,691 in the six cells
     of POOL_CELLS (country x label group; O = Open Grassland, W = Wooded Grassland, X = the seven other classes).
  3. The draw (DRAW_RULE, at SIZES: a constant of this file, part of the commit, never an option; the grade's 720 floor
     is 80% of it, and check_draw, the units and the grade refuse a draw of other sizes): each label group's size
     split between the countries in proportion to the pool by largest
     remainder; one numpy Generator, default_rng(SEED), visits the cells in CELL_ORDER and permutes each cell's sorted
     plot ids (rng.permutation of the cell's size); the cell's first m ids are drawn. A cell's permutation depends only
     on the seed and the cells before it, so a smaller size is a prefix of a larger one (300/300/150 inside
     300/300/300). Recorded: the counts, the allocation, the sha256 of the sorted drawn ids and the numpy and Python
     versions (default_rng's stream is numpy's); the ids, labels and positions go only to PRIV (mode 700, refused
     inside a git checkout). The draw runs once, on the cluster; the page freezes its sha256, and every phase-2 step
     refuses a draw.json whose sorted ids do not give it (--check-draw).
  4. Each drawn plot's distance to Ai2's geometry (exp99's rule, EPSG:32737, 0.001-degree densified edges) and the
     tertile cuts of the drawn plots' distances, rounded to 10 km (R2's bands); each country's area outside exp99's
     region (Natural Earth, EPSG:6933), for the region weights.

The windows (--windows; allowed before freezing). One rslearn window per unit, group "e111", in three roots that are
never mixed: gate23 (Ai2's 259 validation points exp98 graded, from exp89's AWF label windows, 2023), gate17 (exp99's
309 plots, 2017) and blind17 (the draw, 2017). Each window:
  - projection: the plot's UTM zone at 10 m by rslearn's rule (pyproj's first UTM CRS whose area holds the point, as
    rslearn.utils.get_utm_ups_crs and Ai2's GridPartitioner(use_utm) take it); a gate-A point keeps its label window's
    CRS and pixel;
  - bounds: WINDOW_PX x WINDOW_PX (124 = 16 + 9 x 12) inside the 1,024-px cell of the CRS-origin lattice that holds
    the unit's pixel, starting on that cell's 12-px crop lattice (crops of 16 px at overlap 4), with the pixel 56 to
    67 px from the start on each axis, or the window flush with the cell's edge when the pixel is nearer than that.
    rslearn 0.1.17 cuts the window into the cell's own crops (get_window_crop_options), and RasterMerger keeps for the
    pixel the crop the full cell keeps (exp111_jobs.py check-dataset checks both with rslearn's code);
  - time range: the year's request range, [YYYY-01-01T00:00Z, YYYY-12-31T00:00Z], as exp98's and exp99's deployment
    windows hold it (e111.sh compares with deploy-time-range); rslearn cuts twelve 30-day periods back from the end.
  - names a0000 (gate23, exp98's unit order), g0000 (gate17, the sorted plot ids, which is exp99's row order) and
    b0000 (blind17, draw order). The units maps (names, projections, pixels, ids) are private.
  Caps: each root holds exactly its unit count, the total at most MAX_TOTAL_WINDOWS; the blind and gate plots are
  disjoint by id. Label-free and counted: the drawn plots at which exp99 (v1) or exp103 (v1.2) wrote an output, never
  read at the plot or joined to its label. With exp99's 2017 deployment on scratch (--deploy-inv, --deploy-run), the
  flag is the union of two rules: a deployment window (1,024 px, in its own CRS: exp99 wrote 10 of its 313 windows in
  EPSG:32736) holds the plot, or the plot's (EPSG, 1,024-px cell) is a gate17 unit's cell; without it, only the second
  rule, which is then a lower bound (the record says which). Each blind unit carries the flag (in_gate17_cell) into
  the units file, so the grade reports them separately.

The read (--read; gate roots before freezing, blind17 after). A view of the root (symlinks: each window's metadata.json,
its layers/output_<model>/ and the SCL sidecar's layers/sentinel2_scl[.N]/), then oe_inferencex.olmoearth's
read_output and condition_from_inputs (what `oe-inferencex from-olmoearth --conditions --inputs sentinel2_scl` runs),
one window per part, as exp99 read its windows. Each unit is then read by exp98's and exp99's code (find_outputs,
open_grid, locate_all, read_points, covered, readings, majority_3x3): the score pixel holding the label pixel's centre
(gate A) or the plot's point (TimeSync). A second route reads the unit's own GeoTIFF at the unit's own pixel; the two
must agree. Per unit, to PRIV only: the ten probabilities, class, p1, suspicion, covered, condition, 3 x 3 majority.

The gate (--gate A, then --gate B; one rule set, GATE_BARS, written on the page as A1-A4 and B1-B2, PROPOSED and set
before any gate number exists). Each model against its committed known answer at the units with input in both runs:
  A (2023, Ai2's 259 validation points; exp98's v1, exp103's v1.2): A1/A3 the same class at 98% or more, A2/A4 at
  most 2% of the units with |p1 - known p1| > 0.05 (v1 / v1.2); floor 240 units with input in both.
  B (2017, exp99's 309 plots; exp99's v1, exp103's v1.2; only after A passed; no label read): B1 (v1) and B2 (v1.2)
  each hold four checks: the same class at 98% or more; |dp1| as A2; the Spearman correlation of the suspicion -log p1
  with the known suspicion at least 0.95; and of the known answer's confident plots (p1 at or above the page's cut,
  CONFIDENT_CUT: the 91 of the 271), at least 95% at or above the cut here. Floor 250 plots with input in both.
  Report-only beside: the accuracy against Ai2's labels (gate A), agreement at the units whose known p1 is 0.6 or more
  and the near-ties below it, the confusion of the differing units, the plots that gain or lose input, A' (v1 under
  0.1.17 on exp98's own 1,024-px windows) and the item-identity split. Writes e111_gate.json with the GPU's name and
  the commit the gate's predictions ran at (predict_commit, from the gate job's device record, --device-file) beside
  the commit gatecheck ran at (commit); the gate does not pass without both. The blind predict refuses unless it says
  pass, on the same device, and every phase-2 mode refuses if a guarded file changed since either commit.

The units (--units; refused until the page is frozen, the gate passed and draw.json is the page's draw): per model,
exp99's units schema (country, country_names, label, label_names, covered, inside_awf, pred, suspicion, p1, condition,
majority_3x3, weight, wrong_strict, wrong_lenient) plus stratum, stratum_names, band, cgls_raw, in_gate17_cell, both
models' rows in one shuffled order (the permutation from the operating system's entropy, kept only in PRIV: exp104's
audit lesson; a public seed would let anyone undo it), so exp99's and exp109's readers take them as they are. The
weight is the region weight (A'_c / N_c) x (N_cg / m_cg), normalised to sum 1.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import traceback

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
for _p in (ROOT, EXP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import exp89_finetuned_checkpoints as e89   # noqa: E402
import exp98_awf_deployment as e98          # noqa: E402
import exp99_transfer as e99                # noqa: E402
import exp111_jobs as jobs                  # noqa: E402  (standard library and numpy at import)
import timesync_awf_crosswalk as tsx        # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
PLAN = os.path.join(ROOT, "docs", "plan", "blind_transfer.md")
CONFIG_DIR = os.path.join(EXP_DIR, "jobs", "e111_config")
SCL_CONFIG = os.path.join(EXP_DIR, "jobs", "e98_config", "scl_dataset.json")

GROUP = "e111"
YEAR = 2017
SEED = 111
COUNTRIES = tuple(tsx.COUNTRIES)                     # ("kenya", "tanzania")
STRATA = ("O", "W", "X")
STRATUM_NAMES = {"O": "Open Grassland", "W": "Wooded Grassland", "X": "other (seven classes)"}
STRATUM_OF = {"Open Grassland": "O", "Wooded Grassland": "W"}
CELL_ORDER = tuple((c, s) for c in COUNTRIES for s in STRATA)
# the draw's sizes: PROPOSED on the page (300/300/300; exp109's Option A 300/300/150), binding at the phase-1 commit.
# A constant, not an option: the grade's floor of 720 plots with input is 80% of their sum (the smoke checks the pair),
# and check_draw, the units and the grade refuse a draw of other sizes
SIZES = {"O": 300, "W": 300, "X": 300}
# the pool, counted on the Mac by exp109's overnight rebuild (overnight/exp109/pool.json) and copied to the page; the
# selection refuses any other count
POOL_CELLS = {"kenya:O": 379, "kenya:W": 755, "kenya:X": 647, "tanzania:O": 361, "tanzania:W": 733, "tanzania:X": 816}
POOL_N = 3691
DRAW_RULE = ("per label group, the size split between Kenya and Tanzania in proportion to the pool by largest "
             "remainder (ties to Kenya); one numpy Generator default_rng(seed) visits the cells kenya:O, kenya:W, "
             "kenya:X, tanzania:O, tanzania:W, tanzania:X and draws rng.permutation(N_cell) over each cell's plot ids "
             "sorted as strings ('country:plotid'); the cell's first m ids in that order are drawn")
BAND_STEP_KM = 10

CELL_PX = jobs.CELL_PX
CROP_PX = jobs.CROP_PX
STRIDE_PX = jobs.STRIDE_PX
WINDOW_PX = 124
LEAD_PX = 56                                          # the unit pixel sits 56 to 67 px into the window on each axis
RES = jobs.RES
assert (WINDOW_PX - CROP_PX) % STRIDE_PX == 0 and (CELL_PX - CROP_PX) % STRIDE_PX == 0
assert (CELL_PX - WINDOW_PX) % STRIDE_PX == 0
TIME_RANGES = {y: tuple(t.isoformat() for t in v) for y, v in jobs.TIME_RANGES.items()}
ROOTS = {"gate23": {"year": 2023, "prefix": "a", "n": 259},
         "gate17": {"year": 2017, "prefix": "g", "n": e99.FIXED_RULE["n_selected"]},
         "blind17": {"year": 2017, "prefix": "b", "n": None}}          # n: the draw's size
MAX_TOTAL_WINDOWS = 1500
DEPLOY17_WINDOWS = 313                                # exp99's 2017 deployment (its inventory): a read of fewer is partial
MODELS = {"v1": "output_v1", "v12": "output_v12"}
SCL = jobs.SCL

KNOWN = {("gate23", "v1"): os.path.join(OUT, "exp98_units.npz"),
         ("gate23", "v12"): os.path.join(OUT, "exp103", "awf2023", "exp98_units.npz"),
         ("gate17", "v1"): os.path.join(OUT, "exp99_units.npz"),
         ("gate17", "v12"): os.path.join(OUT, "exp103", "transfer2017", "exp99_units.npz")}
# The gate's one rule set (the page's "The gates", A1-A4 and B1-B2; PROPOSED before any gate number exists). Every
# share is over the units with input in both runs (the known answer's and this one's); input gained or lost is counted
# beside and bounded by the floors. 98%: an equivalent pipeline on the same scenes should agree at nearly every unit;
# exp98's deployment against exp89's replica (another imagery route) agreed at 91.9%, v1 against v1.2 at 88.4% (2023)
# and 82.3% (2017). The p1 checks, because P1-P3 grade p1's ranking and a confident cut, not only the class: v1 against
# v1.2 has |dp1| > 0.05 at 49 of 259 and 87 of 271, a Spearman correlation of p1 of 0.780 and 0.747, and keeps 65 of
# v1's 91 confident plots at or above v1's own cut (the gate's rule; 64 at v1.2's cut).
GATE_BARS = {"same_class_min_share": 0.98, "dp1_tol": 0.05, "dp1_max_share": 0.02,
             "spearman_suspicion_min": 0.95, "confident_zone_min_share": 0.95,
             "floor_with_input_both": {"gate23": 240, "gate17": 250}}
GATE_REPORT_ONLY = {"accuracy_points": 0.03, "near_tie_p1": 0.6}
# exp109's confident cut at the 271 (the 91st-largest p1, 11 digits, rounded down), the grade's "confident" and gate B's
# confident-zone check
CONFIDENT_CUT = {"v1": 0.99875695977, "v12": 0.99888118198}
# the page's frozen draw: a line "**Draw sha256 (frozen):** `<64 hex>`" (the sha256 of the sorted drawn ids)
DRAW_SHA_RE = re.compile(r"^\*\*Draw sha256[^*\n]*\*\*\s*`([0-9a-f]{64})`", re.M | re.I)
EXP98_ACCURACY = {"v1": 226, "v12": 227}             # of the 259 (exp98, exp103 awf2023), for the record's beside


# ----------------------------------------------------------------------------- small helpers
def prereg_status(text=None):
    if text is None:
        if not os.path.exists(PLAN):
            return "absent"
        with open(PLAN, encoding="utf-8") as f:
            text = f.read()
    return e89.prereg_status(text)


def page_draw_sha(text=None):
    """The draw sha256 the page froze (DRAW_SHA_RE), or None."""
    if text is None:
        if not os.path.exists(PLAN):
            return None
        with open(PLAN, encoding="utf-8") as f:
            text = f.read()
    m = DRAW_SHA_RE.search(text)
    return m.group(1).lower() if m else None


def check_sizes(sizes, n=None, want=SIZES):
    """A draw at the sizes of this commit (SIZES), whose plots number their sum: PermissionError otherwise (a draw of
    other sizes would be graded against a floor fixed for SIZES)."""
    got = {k: int(v) for k, v in dict(sizes).items()}
    if got != {k: int(v) for k, v in want.items()}:
        raise PermissionError(f"refused: the draw's sizes {got} are not this commit's SIZES {dict(want)} (the floors are "
                              "fixed for them; a change of sizes is a commit to SIZES and FLOORS, and a new draw)")
    if n is not None and int(n) != sum(got.values()):
        raise PermissionError(f"refused: the draw holds {n} plots, not the {sum(got.values())} its sizes give")


def check_draw(priv, text=None, sizes=SIZES):
    """draw.json is the page's draw: its sorted ids give its own recorded sha256 and the one the frozen page holds,
    and it was drawn at this commit's sizes. Returns the sha256; PermissionError otherwise."""
    d = read_json(os.path.join(priv, "draw.json"))
    check_sizes(d.get("sizes") or {}, len(d["units"]), sizes)
    got = sha_ids([u["pid"] for u in d["units"]])
    if got != d.get("sha256_sorted_ids"):
        raise PermissionError("refused: draw.json's ids do not give the sha256 it records")
    want = page_draw_sha(text)
    if want is None:
        raise PermissionError("refused: docs/plan/blind_transfer.md holds no frozen draw sha256")
    if got != want:
        raise PermissionError(f"refused: draw.json's sha256 {got[:12]}... is not the page's {want[:12]}...")
    return got


def private_dir(path, create=True):
    """The directory that may hold plot ids, labels, positions and the datasets: refused inside any git checkout;
    mode 700."""
    if e99.inside_a_checkout(path):
        raise ValueError(f"refusing a private directory inside a git checkout ({path})")
    if create:
        os.makedirs(path, exist_ok=True)
        os.chmod(path, 0o700)
    return path


def write_private(obj, path):
    jobs.write_json(obj, path, private=True)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def dump(obj, path):
    """A record: no key naming an id or a position (exp111_jobs.check_record_keys, the pack jobs' set, which holds
    exp98's), atomic."""
    jobs.check_record_keys(obj, os.path.basename(path))
    e98.check_no_coordinates(obj)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    e89.dump(obj, path + ".partial")
    os.replace(path + ".partial", path)


def sha_ids(ids):
    return hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()


def stratum_of(label):
    return STRATUM_OF.get(str(label), "X")


# ----------------------------------------------------------------------------- the draw
def allocate(sizes, cell_counts):
    """{(country, stratum): m}: each stratum's size split between the countries in proportion to their pool counts,
    by largest remainder; a tie goes to the country first in COUNTRIES."""
    out = {}
    for s in STRATA:
        N = {c: cell_counts[(c, s)] for c in COUNTRIES}
        tot = sum(N.values())
        if sizes[s] > tot:
            raise ValueError(f"stratum {s}: {sizes[s]} requested, {tot} in the pool")
        q = {c: sizes[s] * N[c] / tot if tot else 0.0 for c in COUNTRIES}
        m = {c: int(math.floor(q[c])) for c in COUNTRIES}
        left = sizes[s] - sum(m.values())
        for c in sorted(COUNTRIES, key=lambda c: (-(q[c] - m[c]), COUNTRIES.index(c)))[:left]:
            m[c] += 1
        for c in COUNTRIES:
            if m[c] > N[c]:
                raise ValueError(f"cell {c}:{s}: {m[c]} allocated, {N[c]} in the pool")
            out[(c, s)] = m[c]
    return out


def draw(cells, sizes, seed=SEED):
    """cells: {(country, stratum): [plot ids]}. Returns (the drawn ids in draw order with their cell, the
    allocation). DRAW_RULE."""
    alloc = allocate(sizes, {k: len(v) for k, v in cells.items()})
    rng = np.random.default_rng(seed)
    drawn = []
    for key in CELL_ORDER:
        ids = sorted(cells[key])
        perm = rng.permutation(len(ids))
        drawn += [(ids[i], key) for i in perm[:alloc[key]]]
    return drawn, alloc


def band_cuts(dist_km, step=BAND_STEP_KM):
    """R2's far bands: the tertile cuts of the drawn plots' distances, rounded to `step` km."""
    q = np.quantile(np.asarray(dist_km, dtype=np.float64), [1 / 3, 2 / 3])
    return [float(step * round(v / step)) for v in q]


def same_draw(old, new):
    """Two draw.json files hold the same draw: seed, sizes, pool counts, the sha256 and the drawn ids in draw order."""
    return (old.get("seed") == new["seed"] and old.get("sizes") == new["sizes"]
            and old.get("pool_counts") == new["pool_counts"] and old.get("sha256_sorted_ids") == new["sha256_sorted_ids"]
            and [u.get("pid") for u in old.get("units", [])] == [u["pid"] for u in new["units"]])


def same_gate17(old, new):
    return [u.get("pid") for u in old.get("units", [])] == [u["pid"] for u in new["units"]]


def keep_or_write(obj, path, same, redo, what):
    """'kept' when `path` holds the same content (same(old, obj)), 'written' when absent or (with redo) replaced;
    ValueError when it differs and redo is off."""
    if os.path.exists(path):
        if same(read_json(path), obj):
            return "kept"
        if not redo:
            raise ValueError(f"{what} exists and differs from this run's; E111_REDO=select replaces it (only before "
                             "the page is frozen)")
    write_private(obj, path)
    return "written"


def run_select(timesync, request_geometry, boundaries, priv, sizes=SIZES, seed=SEED, geometry_sha256=tsx.AWF_GEOMETRY_SHA256,
               fixed=e99.FIXED_RULE, min_plots=e99.MIN_PLOTS, pool_cells=POOL_CELLS, log=print, redo=False):
    """exp99's 309 (the gate-B plots), the pool, the draw, the distances and areas. Private files: gate17.json and
    draw.json in `priv`, kept when an earlier run left the same draw (a rerun after a later failure), refused when it
    left a different one unless redo (refused once the page is frozen); returns the record (counts, allocation,
    sha256, cuts, areas, versions)."""
    refuse_redo_if_frozen(redo, "--select")
    priv = private_dir(priv)
    P, sel_rec, areas = e99.select(timesync, request_geometry, boundaries, geometry_sha256, min_plots, fixed)
    if areas is None:
        raise ValueError("--select needs --boundaries: the region weights need each country's area outside exp99's region")
    allp = tsx.read_plots(timesync, YEAR, history=False)
    excluded = set(P["pid"].tolist())
    keep = np.array([p not in excluded for p in allp["pid"]])
    pool = tsx.subset(allp, keep)
    if int((~keep).sum()) != len(excluded):
        raise ValueError("exp99's plots are not all in the 2017 file read for the pool")
    cells = {k: [] for k in CELL_ORDER}
    info = {}
    for i, pid in enumerate(pool["pid"]):
        key = (str(pool["country"][i]), stratum_of(pool["label"][i]))
        cells[key].append(str(pid))
        info[str(pid)] = i
    counts = {f"{c}:{s}": len(cells[(c, s)]) for c, s in CELL_ORDER}
    if pool_cells is not None and (counts != pool_cells or sum(counts.values()) != POOL_N):
        raise ValueError(f"the pool's cells are {counts}, the page fixed {pool_cells}: not the sample or the rule the "
                         "page was written on")
    drawn, alloc = draw(cells, sizes, seed)
    ids = [d[0] for d in drawn]
    if set(ids) & excluded:
        raise ValueError("a drawn plot is one of exp99's")
    idx = np.array([info[p] for p in ids], dtype=np.int64)
    geom = tsx.load_request_geometry(request_geometry, geometry_sha256)
    dist = tsx.distance_km(geom, pool["lon"][idx], pool["lat"][idx]) if len(idx) else np.zeros(0)
    if len(idx) and float(np.min(dist)) <= sel_rec["d_km"]:
        raise ValueError("a drawn plot lies within exp99's buffer")
    cuts = band_cuts(dist) if len(idx) else []
    band = np.searchsorted(np.asarray(cuts), dist, side="right") if len(idx) else np.zeros(0, int)
    outside = {c: areas["country_km2"][c] - areas["by_country_km2"][c] for c in COUNTRIES}
    gate17 = {"year": YEAR, "order": "sorted plot id (exp99's row order)",
              "units": [{"pid": str(P["pid"][k]), "country": str(P["country"][k]), "inside": bool(P["inside"][k]),
                         "lon": float(P["lon"][k]), "lat": float(P["lat"][k])} for k in range(len(P["pid"]))]}
    blind = {"year": YEAR, "seed": seed, "sizes": sizes, "rule": DRAW_RULE, "sha256_sorted_ids": sha_ids(ids),
             "pool_counts": counts, "area_outside_region_km2": outside, "band_cuts_km": cuts,
             "units": [{"pid": p, "country": key[0], "stratum": key[1], "label": str(pool["label"][info[p]]),
                        "lon": float(pool["lon"][info[p]]), "lat": float(pool["lat"][info[p]]),
                        "dist_km": float(dist[k]), "band": int(band[k])} for k, (p, key) in enumerate(drawn)]}
    files = {"gate17.json": keep_or_write(gate17, os.path.join(priv, "gate17.json"), same_gate17, redo, "gate17.json"),
             "draw.json": keep_or_write(blind, os.path.join(priv, "draw.json"), same_draw, redo, "draw.json")}
    rec = {"experiment": "exp111 selection: exp99's 309 (gate B), the pool and the blind draw", "utc": e89.utc_now(),
           "private_files": files,
           "versions": {"numpy": np.__version__, "python": platform.python_version(),
                        "rng": "numpy.random.default_rng(seed), PCG64; permutation per cell in CELL_ORDER"},
           "exp99_rule": {k: sel_rec[k] for k in ("d_km", "n_selected", "by_country", "inside_request_geometry")},
           "pool": {"n": int(sum(counts.values())), "cells": counts, "matches_page": pool_cells is not None},
           "draw": {"seed": seed, "sizes_O_W_X": [int(sizes[k]) for k in STRATA], "allocation": {f"{c}:{s}": m for (c, s), m in alloc.items()},
                    "n": len(ids), "sha256_sorted_ids": blind["sha256_sorted_ids"], "rule": DRAW_RULE},
           "distance": {"band_cuts_km": cuts, "min_km_rounded": round(float(np.min(dist)), -1) if len(idx) else None,
                        "rule": "exp99's: EPSG:32737, the geometry's edges densified every 0.001 degree"},
           "areas": {"outside_exp99_region_km2": outside, "country_km2": areas["country_km2"],
                     "exp99_region_by_country_km2": areas["by_country_km2"], "crs": tsx.AREA_CRS},
           "timesync": {k: tsx.TIMESYNC[k] for k in ("repository", "sha256")},
           "request_geometry_sha256": geometry_sha256}
    log(json.dumps({"pool": rec["pool"]["n"], "drawn": len(ids), "sha256": rec["draw"]["sha256_sorted_ids"][:16],
                    "files": files}))
    return rec


# ----------------------------------------------------------------------------- window geometry
def utm_epsg(lon, lat):
    """rslearn.utils.get_utm_ups_crs's rule (rslearn 0.1.17, utils/get_utm_ups_crs.py:25-52): the first WGS 84 UTM CRS
    pyproj lists for a point-sized area of interest. exp111_jobs.py check-dataset compares with rslearn's own call."""
    import pyproj
    if not -80 <= lat <= 84:
        raise ValueError("outside the UTM band")
    infos = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84", area_of_interest=pyproj.aoi.AreaOfInterest(west_lon_degree=lon, south_lat_degree=lat,
                                                                       east_lon_degree=lon, north_lat_degree=lat))
    if not infos:
        raise ValueError("no UTM zone")
    return int(infos[0].code)


def project_pixel(epsg, lon, lat):
    """(px, py, X, Y): the point in the CRS and the 10 m pixel holding it in rslearn's pixel coordinates (x = X / 10,
    y = Y / -10)."""
    import pyproj
    X, Y = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
    return int(math.floor(X / RES)), int(math.floor(Y / -RES)), float(X), float(Y)


def snap(p, size=WINDOW_PX, cell=CELL_PX, stride=STRIDE_PX, lead=LEAD_PX):
    """(start, end, cell origin) on one axis: the window of `size` px inside the `cell`-px lattice cell holding pixel
    p, starting on the cell's crop lattice (a multiple of `stride` from the cell's origin), with p `lead` to
    lead + stride - 1 px from the start, or flush with the cell's edge when p is nearer to it."""
    c0 = cell * (p // cell)
    j = (p - c0 - lead) // stride
    j = min(max(j, 0), (cell - size) // stride)
    s = c0 + stride * j
    return s, s + size, c0


def window_bounds(px, py):
    x0, x1, X0 = snap(px)
    y0, y1, Y0 = snap(py)
    return [x0, y0, x1, y1], [X0, Y0]


def window_metadata(epsg, bounds, year):
    """metadata.json as rslearn 0.1.17's FileWindowStorage writes it (rslearn/dataset/storage/file.py:171-187)."""
    t0, t1 = TIME_RANGES[year]
    return {"projection": {"crs": f"EPSG:{epsg}", "x_resolution": RES, "y_resolution": -RES},
            "bounds": [int(b) for b in bounds], "time_range": [t0, t1], "options": {}}


def make_unit(name, epsg, px, py, **extra):
    b, cell = window_bounds(px, py)
    return dict(name=name, epsg=int(epsg), px=int(px), py=int(py), bounds=b, cell=cell, **extra)


def timesync_units(rows, prefix):
    out = []
    for k, r in enumerate(rows):
        epsg = utm_epsg(r["lon"], r["lat"])
        px, py, _, _ = project_pixel(epsg, r["lon"], r["lat"])
        extra = {k2: r[k2] for k2 in ("pid", "country", "stratum", "lon", "lat", "inside") if k2 in r}
        out.append(make_unit(f"{prefix}{k:04d}", epsg, px, py, **extra))
    return out


def gate23_units(labels, exp98_units=KNOWN[("gate23", "v1")]):
    """Ai2's validation points exp98 graded (split val and covered: 259), in exp98's unit order, each at its label
    window's own CRS and labelled pixel (exp98's read_label_windows; exp89's extraction on scratch)."""
    u = np.load(exp98_units)
    names = [str(n) for n, s, c in zip(u["names"], u["split"], u["covered"]) if s == "val" and c]
    points, _ = e98.read_label_windows(labels)
    by = {p["name"]: p for p in points}
    missing = [n for n in names if n not in by]
    if missing:
        raise ValueError(f"{len(missing)} of exp98's graded validation points have no label window")
    out = []
    for k, n in enumerate(names):
        p = by[n]
        if None in tuple(p["proj_res"]) or tuple(float(v) for v in p["proj_res"]) != (RES, -RES) or None in p["pix"]:
            raise ValueError("a label window is not at 10 m, or its metadata has no bounds")
        crs = str(p["proj_crs"])
        if not crs.startswith("EPSG:"):
            raise ValueError("a label window's CRS has no EPSG code")
        out.append(make_unit(f"a{k:04d}", int(crs.split(":")[1]), p["pix"][0], p["pix"][1], key=n, X=p["X"], Y=p["Y"]))
    return out


def unit_facts(units):
    """Counts only: CRSs, distinct cells, the unit pixel's place in its window, windows flush with a cell edge."""
    epsg, cells, lead, flush = {}, set(), [], 0
    for u in units:
        epsg[str(u["epsg"])] = epsg.get(str(u["epsg"]), 0) + 1
        cells.add((u["epsg"], *u["cell"]))
        b = u["bounds"]
        lead += [u["px"] - b[0], u["py"] - b[1]]
        flush += b[0] == u["cell"][0] or b[1] == u["cell"][1] or b[2] == u["cell"][0] + CELL_PX \
            or b[3] == u["cell"][1] + CELL_PX
    return {"epsg": epsg, "distinct_cells": len(cells), "lead_px_min": int(min(lead)) if lead else None,
            "lead_px_max": int(max(lead)) if lead else None, "flush_with_a_cell_edge": int(flush)}


def check_units(units, n):
    """The caps: exactly n units, unique names, every window WINDOW_PX square, inside its cell, on the crop lattice."""
    bad = {"size": 0, "outside_cell": 0, "off_lattice": 0, "pixel_outside": 0}
    for u in units:
        b, (X0, Y0) = u["bounds"], u["cell"]
        bad["size"] += (b[2] - b[0], b[3] - b[1]) != (WINDOW_PX, WINDOW_PX)
        bad["outside_cell"] += not (X0 <= b[0] and b[2] <= X0 + CELL_PX and Y0 <= b[1] and b[3] <= Y0 + CELL_PX)
        bad["off_lattice"] += bool((b[0] - X0) % STRIDE_PX or (b[1] - Y0) % STRIDE_PX)
        bad["pixel_outside"] += not (b[0] <= u["px"] < b[2] and b[1] <= u["py"] < b[3])
    if len(units) != n or len({u["name"] for u in units}) != n or any(bad.values()):
        raise ValueError(f"the units fail the caps: {len(units)} of {n}, {bad}")
    return bad


def _same_metadata(md, want):
    """A window's metadata.json against the one write_root writes: CRS, resolution, bounds and the time range as
    instants (rslearn may re-serialise it)."""
    from datetime import datetime
    try:
        p, q = md["projection"], want["projection"]
        same = (p["crs"] == q["crs"] and float(p["x_resolution"]) == float(q["x_resolution"])
                and float(p["y_resolution"]) == float(q["y_resolution"])
                and [int(b) for b in md["bounds"]] == want["bounds"])
        return same and [datetime.fromisoformat(t) for t in md["time_range"]] == \
            [datetime.fromisoformat(t) for t in want["time_range"]]
    except (KeyError, TypeError, ValueError):
        return False


def root_matches(ds, units, year, config=None):
    """An existing root is this run's: config.json byte for byte, exactly the units' windows in group GROUP, each
    window's metadata as write_root writes it. Prepared imagery and outputs are not looked at."""
    src = config or os.path.join(CONFIG_DIR, "dataset.json")
    cfg = os.path.join(ds, "config.json")
    if not os.path.isfile(cfg):
        return False
    with open(cfg, "rb") as f1, open(src, "rb") as f2:
        if f1.read() != f2.read():
            return False
    have = jobs.list_windows(ds) if os.path.isdir(os.path.join(ds, "windows")) else []
    if {(g, n) for g, n, _ in have} != {(GROUP, u["name"]) for u in units} or len(have) != len(units):
        return False
    for u in units:
        try:
            md = read_json(os.path.join(ds, "windows", GROUP, u["name"], "metadata.json"))
        except (OSError, ValueError):
            return False
        if not _same_metadata(md, window_metadata(u["epsg"], u["bounds"], year)):
            return False
    return True


def write_root(ds, units, year, config=None, redo=False):
    """config.json (the bytes of e111_config/dataset.json) and one window per unit. An existing root that is this
    run's (root_matches) is kept with its imagery ('kept'); a different one is refused unless redo, which removes it."""
    src = config or os.path.join(CONFIG_DIR, "dataset.json")
    if os.path.lexists(ds):
        if root_matches(ds, units, year, config):
            return jobs.sha256_file(src), "kept"
        if not redo:
            raise ValueError(f"{os.path.basename(ds)} exists and is not this run's windows; E111_REDO=select removes "
                             "it (its prepared imagery is fetched again)")
        shutil.rmtree(ds)
    os.makedirs(os.path.join(ds, "windows", GROUP))
    shutil.copyfile(src, os.path.join(ds, "config.json"))
    for u in units:
        wd = os.path.join(ds, "windows", GROUP, u["name"])
        os.makedirs(wd)
        with open(os.path.join(wd, "metadata.json"), "w") as f:
            json.dump(window_metadata(u["epsg"], u["bounds"], year), f)
    return jobs.sha256_file(src), "written"


def in_cells_of(units, others):
    """Per unit, whether its (EPSG, 1,024-px cell) is the cell of one of `others` (label-free)."""
    cells = {(int(o["epsg"]), int(o["cell"][0]), int(o["cell"][1])) for o in others}
    return [(int(u["epsg"]), int(u["cell"][0]), int(u["cell"][1])) in cells for u in units]


def refuse_redo_if_frozen(redo, what):
    """E111_REDO=select after freezing would replace the frozen draw, or remove blind17 with phase 2's output layers."""
    if redo and prereg_status() == "frozen":
        raise PermissionError(f"refused: {what} --redo (E111_REDO=select) after docs/plan/blind_transfer.md is frozen: "
                              "it would replace the frozen draw or remove blind17 with phase 2's outputs")


def in_deployment_windows(units, dep):
    """Per unit, whether a deployment window holds the unit's point (jobs.deployment_windows of exp99's 2017 run: the
    1,024-px windows exp99 and exp103 predicted), in that window's own CRS: the point is projected to each CRS the
    deployment used (exp99 wrote windows in EPSG:32736 as well as 32737). Label-free; positions stay in memory."""
    import pyproj
    out = np.zeros(len(units), bool)
    if not units:
        return out.tolist()
    lon = np.array([float(u["lon"]) for u in units])
    lat = np.array([float(u["lat"]) for u in units])
    by_epsg = {}
    for d in dep:
        if d[5] is None:
            raise ValueError("a deployment window's CRS has no EPSG code")
        by_epsg.setdefault(int(d[5]), []).append(d[6])
    for epsg, bl in sorted(by_epsg.items()):
        X, Y = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
        px = np.floor(np.asarray(X) / RES).astype(np.int64)[:, None]
        py = np.floor(np.asarray(Y) / -RES).astype(np.int64)[:, None]
        B = np.asarray(bl, np.int64)
        out |= ((B[None, :, 0] <= px) & (px < B[None, :, 2]) & (B[None, :, 1] <= py) & (py < B[None, :, 3])).any(1)
    return out.tolist()


def run_windows(priv, labels, redo=False, config=None, known23=KNOWN[("gate23", "v1")], n_expected=None, log=print,
                deploy17=None, deploy17_windows=DEPLOY17_WINDOWS):
    """The three roots (module docstring). deploy17: (inventory, run directory) of exp99's 2017 deployment, when it is
    still on scratch, for the in_gate17_cell flag's first rule. A deployment that cannot be read whole (the purge
    removes files one by one, so a run directory can outlive some metadata.json or lose whole windows; a read of
    other than deploy17_windows windows is partial) falls back to the own-cell rule, a lower bound, and the record
    says so and why; it never fails the step."""
    refuse_redo_if_frozen(redo, "--windows")
    priv = private_dir(priv)
    g17 = read_json(os.path.join(priv, "gate17.json"))
    drw = read_json(os.path.join(priv, "draw.json"))
    units = {"gate23": gate23_units(labels, known23) if labels else [],
             "gate17": timesync_units(g17["units"], ROOTS["gate17"]["prefix"]),
             "blind17": timesync_units(drw["units"], ROOTS["blind17"]["prefix"])}
    n_expected = n_expected or {"gate23": ROOTS["gate23"]["n"], "gate17": ROOTS["gate17"]["n"],
                                "blind17": int(sum(drw["sizes"].values()))}
    if {u["pid"] for u in units["gate17"]} & {u["pid"] for u in units["blind17"]}:
        raise ValueError("a blind plot is one of the gate plots")
    by_cell = in_cells_of(units["blind17"], units["gate17"])
    by_dep, n_dep_windows, dep_state = None, None, "not on scratch (absent when the step ran)"
    if deploy17:
        dep, why = jobs.deployment_windows_or_reason(*deploy17, expected=deploy17_windows)
        if dep is None:
            dep_state = why
        else:
            dep_state = "read whole"
            n_dep_windows = len(dep)
            by_dep = in_deployment_windows(units["blind17"], dep)
    flags = [bool(a or b) for a, b in zip(by_cell, by_dep)] if by_dep is not None else list(by_cell)
    for u, f in zip(units["blind17"], flags):
        u["in_gate17_cell"] = bool(f)
    total = sum(len(v) for v in units.values())
    if total > MAX_TOTAL_WINDOWS:
        raise ValueError(f"{total} windows, above the cap of {MAX_TOTAL_WINDOWS}")
    rec = {"experiment": "exp111 windows", "utc": e89.utc_now(), "window_px": WINDOW_PX, "cell_px": CELL_PX,
           "crop_px": CROP_PX, "stride_px": STRIDE_PX, "group": GROUP, "roots": {}, "total": total,
           "blind_plots_in_a_gate17_cell": {
               "n": int(sum(flags)),
               "n_by_own_cell_rule": int(sum(by_cell)),
               "n_by_deployment_windows": int(sum(by_dep)) if by_dep is not None else None,
               "deployment_windows_read": n_dep_windows,
               "deployment_2017": dep_state,
               "method": ("exp99's 2017 deployment windows (each in its own CRS) or a gate17 unit's (EPSG, 1,024-px "
                          "cell): the union" if by_dep is not None else
                          "a gate17 unit's (EPSG, 1,024-px cell) only, exp99's deployment being gone or unreadable: a "
                          "lower bound (exp99 held 313 windows in 264 cells, 10 in EPSG:32736, and 51 plots in two or "
                          "more)"),
               "rule": "an output of exp99 (v1) or exp103 (v1.2) exists on scratch at the drawn plot, never read at it "
                       "or joined to its label; reported separately by the grade (in_gate17_cell)"}}
    for name, us in units.items():
        if not us:
            rec["roots"][name] = {"windows": 0, "note": "no units given (--labels absent)"}
            continue
        check_units(us, n_expected[name])
        year = ROOTS[name]["year"]
        sha, how = write_root(os.path.join(priv, "roots", name), us, year, config, redo)
        write_private({"root": name, "year": year, "group": GROUP, "units": us}, os.path.join(priv, f"units_{name}.json"))
        rec["roots"][name] = {"windows": len(us), "year": year, "time_range": list(TIME_RANGES[year]),
                              "config_sha256": sha, "root": how, **unit_facts(us)}
        if name == "blind17":
            cnt, inc = {}, {}
            for u in us:
                k = f"{u['country']}:{u['stratum']}"
                cnt[k] = cnt.get(k, 0) + 1
                inc[k] = inc.get(k, 0) + int(u["in_gate17_cell"])
            rec["roots"][name]["by_cell"] = cnt
            rec["blind_plots_in_a_gate17_cell"]["by_cell"] = inc
    log(json.dumps({k: [v.get("windows"), v.get("root")] for k, v in rec["roots"].items()}))
    return rec


# ----------------------------------------------------------------------------- the read
def merged_config(ds_config, scl_config=SCL_CONFIG):
    """The view's config.json: the root's layers and the SCL sidecar's, so that the reader knows SCL's band name."""
    a, b = read_json(ds_config), read_json(scl_config)
    layers = dict(a["layers"])
    layers.update(b["layers"])
    return {"layers": layers}


def build_view(ds, scl, layer, view, names=None):
    """VIEW as an rslearn dataset the reader reads: per window with layers/<layer>/ completed (or the names given),
    symlinks to its metadata.json and layers/<layer>/ and to the sidecar's layers/sentinel2_scl[.N]/. The sidecar's
    metadata.json must equal the window's byte for byte."""
    if os.path.lexists(view):
        shutil.rmtree(view)
    os.makedirs(os.path.join(view, "windows", GROUP))
    with open(os.path.join(view, "config.json"), "w") as f:
        json.dump(merged_config(os.path.join(ds, "config.json")), f, indent=1)
    c = {"windows": 0, "with_scl": 0, "missing_sidecar": 0, "metadata_differs": 0}
    for g, n, p in jobs.list_windows(ds):
        if (names is not None and n not in names) or (names is None and not jobs.layer_done(p, layer)):
            continue
        vw = os.path.join(view, "windows", g, n)
        os.makedirs(os.path.join(vw, "layers"))
        os.symlink(os.path.join(p, "metadata.json"), os.path.join(vw, "metadata.json"))
        os.symlink(os.path.join(p, "layers", layer), os.path.join(vw, "layers", layer))
        c["windows"] += 1
        sw = os.path.join(scl, "windows", g, n)
        if not os.path.isdir(sw):
            c["missing_sidecar"] += 1
            continue
        with open(os.path.join(sw, "metadata.json"), "rb") as f1, open(os.path.join(p, "metadata.json"), "rb") as f2:
            c["metadata_differs"] += f1.read() != f2.read()
        scls = [d for d in sorted(os.listdir(os.path.join(sw, "layers"))) if d == SCL or d.startswith(SCL + ".")] \
            if os.path.isdir(os.path.join(sw, "layers")) else []
        for d in scls:
            os.symlink(os.path.join(sw, "layers", d), os.path.join(vw, "layers", d))
        c["with_scl"] += bool(scls)
    if c["missing_sidecar"] or c["metadata_differs"] or c["with_scl"] != c["windows"]:
        raise ValueError(f"the SCL sidecar does not match the predicted windows: {json.dumps(c)}")
    return c


def read_parts(view, layer, parts, conditions=True, log=print):
    """oe_inferencex.olmoearth's read_output and condition_from_inputs (from-olmoearth --conditions --inputs
    sentinel2_scl), one window per part directory, as exp99 read its windows."""
    from oe_inferencex import olmoearth as oe
    os.makedirs(parts, exist_ok=True)
    done, dev, t0 = [], 0.0, time.time()
    wins = [n for g, n, p in jobs.list_windows(view)]
    for i, n in enumerate(wins):
        out = os.path.join(parts, n)
        if os.path.lexists(out):
            shutil.rmtree(out)
        s = oe.read_output(view, out, layer=layer, group=GROUP, windows=[f"{GROUP}/{n}"])
        if conditions:
            oe.condition_from_inputs(view, out, inputs=[SCL], layer=layer, group=GROUP, windows=[f"{GROUP}/{n}"])
        if s["bands"] != 10 or s["windows_read"] != 1 or len(s["grids"]) != 1:
            raise ValueError(f"part {i}: {s['bands']} bands, {s['windows_read']} windows read, {len(s['grids'])} grids")
        dev = max(dev, float(s["largest_sum_deviation"] or 0.0))
        done.append(out)
        if (i + 1) % 200 == 0:
            log(f"{i + 1} of {len(wins)} windows read ({time.time() - t0:.0f} s)")
    return done, dev


def esc(s):
    """A window id as a literal fnmatch pattern (e98_read.sh, exp103_jobs.esc)."""
    return "".join(f"[{ch}]" if ch in "[]*?" else ch for ch in s)


def unit_points(units):
    """exp98's point records: a gate-A point at its label pixel's centre in its own CRS, a TimeSync plot at its
    point in WGS84 (e98.timesync_points), reprojected onto each grid by e98.locate_all."""
    from rasterio.crs import CRS
    pts = []
    for u in units:
        if "X" in u:
            pts.append({"crs": CRS.from_epsg(int(u["epsg"])), "X": float(u["X"]), "Y": float(u["Y"])})
        else:
            pts += e98.timesync_points({"lon": [u["lon"]], "lat": [u["lat"]]})
    return pts


def direct_route(ds, layer, units):
    """The second route: each unit's own window's GeoTIFF at the unit's own pixel (row py - y0, column px - x0)."""
    import rasterio
    from rasterio.windows import Window
    V = np.full((len(units), 10), np.nan)
    for k, u in enumerate(units):
        wd = os.path.join(ds, "windows", GROUP, u["name"])
        tif = os.path.join(wd, "layers", layer, jobs.P, "geotiff.tif")
        if not jobs.layer_done(wd, layer) or not os.path.exists(tif):
            continue
        b = u["bounds"]
        with rasterio.open(tif) as src:
            V[k] = src.read(window=Window(u["px"] - b[0], u["py"] - b[1], 1, 1))[:, 0, 0]
    return V


def read_units(parts, units):
    """exp99's reading of the units on the parts: per unit the grid that covers its pixel, the ten probabilities, the
    3 x 3 block and the condition code; then covered, the class, p1 and the suspicion, the 3 x 3 majority."""
    grids, names, meta = e99.read_map(parts)
    cands = e98.locate_all(unit_points(units), grids)
    # Each unit has a window of its own; another unit's window can overlap it (two plots within about 1.2 km), and
    # exp99's "first covered grid" would then read the unit there. A unit is read in its own window only (the part
    # named after it); the overlaps are counted. Units read without names (A', exp98's windows) keep exp99's rule.
    in_other = 0
    if all(u.get("name") for u in units):
        part_of = [os.path.basename(os.path.dirname(G["scores"])) for G in grids]
        for k, u in enumerate(units):
            in_other += any(part_of[c[0]] != u["name"] for c in cands[k])
            cands[k] = [c for c in cands[k] if part_of[c[0]] == u["name"]]
    gi, rows, cols, how, off, P, B, C, overlap = e98.read_points(grids, cands)
    overlap = dict(overlap, units_also_inside_another_units_window=int(in_other))
    cov = e98.covered(P)
    n = len(units)
    pred, u_, p1 = np.full(n, -1, np.int64), np.full(n, np.nan), np.full(n, np.nan)
    if cov.any():
        rd = e98.readings(P[cov])
        if rd["sum_dev"] > e98.SUM_TOLERANCE:
            raise ValueError("the probabilities at the units do not sum to 1 within 1e-3")
        pred[cov], u_[cov], p1[cov] = rd["pred"], rd["u"], rd["p1"]
    maj = np.full(n, -1, np.int64)
    maj[cov] = e98.majority_3x3(B[cov])
    own = np.array([g >= 0 and os.path.basename(os.path.dirname(grids[g]["scores"])) == us["name"]
                    for g, us in zip(gi, units)])
    return {"P": P, "covered": cov, "pred": pred, "suspicion": u_, "p1": p1, "condition": C, "majority_3x3": maj,
            "own_window": own, "reprojected": np.asarray(how) == "reprojected", "offset": off}, names, overlap


def run_read(priv, root, model, redo=False, keep_parts=False, log=print):
    """Read one root's output layer of one model into PRIV/read/<root>_<model>.npz (unit order). Refused for the blind
    root before the page is frozen."""
    if root == "blind17":
        if prereg_status() != "frozen":
            raise PermissionError("refused: no blind output is read before docs/plan/blind_transfer.md is frozen")
        check_draw(priv)
    layer = MODELS[model]
    ds, scl = os.path.join(priv, "roots", root), os.path.join(priv, "scl", root)
    m = read_json(os.path.join(priv, f"units_{root}.json"))
    units = m["units"]
    rd = os.path.join(priv, "read")
    dest = os.path.join(rd, f"{root}_{model}.npz")
    if os.path.exists(dest) and not redo:
        raise ValueError(f"{os.path.basename(dest)} exists; --redo replaces it")
    view, parts = os.path.join(rd, f"view_{root}_{model}"), os.path.join(rd, f"parts_{root}_{model}")
    if os.path.lexists(parts):
        shutil.rmtree(parts)
    vc = build_view(ds, scl, layer, view)
    part_dirs, dev = read_parts(view, layer, parts, log=log)
    r, cond_names, overlap = read_units(part_dirs, units)
    V = direct_route(ds, layer, units)
    both = r["covered"] & np.isfinite(V).all(1)
    diff = float(np.nanmax(np.abs(r["P"][both] - V[both]))) if both.any() else None
    agree = bool(both.sum() == r["covered"].sum() == int(np.isfinite(V).all(1).sum()) and (diff is None or diff <= 1e-6)
                 and bool(r["own_window"][r["covered"]].all()))
    np.savez_compressed(dest, **{k: v for k, v in r.items()}, direct=V)
    os.chmod(dest, 0o600)
    if not keep_parts:              # about 0.7 MB per window; a re-read rebuilds them from the dataset
        shutil.rmtree(parts)
        shutil.rmtree(view)
    rec = {"root": root, "model": model, "layer": layer, "units": len(units), "view": vc, "parts": len(part_dirs),
           "largest_sum_deviation": dev, "covered": int(r["covered"].sum()),
           "reprojected": int(r["reprojected"].sum()), "read_from_own_window": int(r["own_window"].sum()),
           "direct_route": {"units_with_output": int(np.isfinite(V).all(1).sum()), "largest_difference": diff,
                            "agrees": agree},
           "condition_names": {str(k): v for k, v in (cond_names or {}).items()}, "overlap": overlap,
           "ok": agree}
    log(json.dumps({k: rec[k] for k in ("root", "model", "units", "covered", "ok")}))
    return rec


def run_read_aprime(priv, plan_path, log=print):
    """Diagnostic A': v1 under rslearn 0.1.17 on exp98's own deployment windows (layer output_e111_v1), read at the
    gate-A points as exp98 read them; PRIV/read/aprime_v1.npz."""
    plan = read_json(plan_path)
    units = read_json(os.path.join(priv, "units_gate23.json"))["units"]
    rd = os.path.join(priv, "read")
    parts_all = []
    for p in plan:
        view = os.path.join(rd, f"view_aprime_r{p['root_index']}")
        parts = os.path.join(rd, f"parts_aprime_r{p['root_index']}")
        for d in (view, parts):
            if os.path.lexists(d):
                shutil.rmtree(d)
        os.makedirs(os.path.join(view, "windows"))
        n_written = 0
        for wid in p["windows"]:
            g, n = wid.split("/", 1)
            src = os.path.join(p["root"], "windows", g, n)
            if not jobs.layer_done(src, "output_e111_v1"):
                continue
            vw = os.path.join(view, "windows", g, n)
            os.makedirs(os.path.join(vw, "layers"))
            os.symlink(os.path.join(src, "metadata.json"), os.path.join(vw, "metadata.json"))
            os.symlink(os.path.join(src, "layers", "output_e111_v1"), os.path.join(vw, "layers", "output_e111_v1"))
            n_written += 1
        from oe_inferencex import olmoearth as oe
        os.makedirs(parts)
        for k, wid in enumerate(sorted(w for w in p["windows"] if os.path.isdir(os.path.join(view, "windows", w)))):
            out = os.path.join(parts, f"w{k:04d}")
            oe.read_output(view, out, layer="output_e111_v1", windows=[esc(wid)])
            parts_all.append(out)
    r, _, _ = read_units(parts_all, [dict(u, name="") for u in units])
    dest = os.path.join(rd, "aprime_v1.npz")
    np.savez_compressed(dest, **r)
    os.chmod(dest, 0o600)
    rec = {"units": len(units), "covered": int(r["covered"].sum()), "parts": len(parts_all)}
    log(json.dumps(rec))
    return rec


# ----------------------------------------------------------------------------- the gate
def _load_read(priv, root, model):
    path = os.path.join(priv, "read", f"{root}_{model}.npz")
    return dict(np.load(path)) if os.path.exists(path) else None


def compare(new, known_pred, known_p1, known_cov, known_u=None, label=None, cut=None, bars=GATE_BARS):
    """One model's run against its known answer, at the units with input in both runs ("both"): the same class, the
    units with |p1 - known p1| above the tolerance, the Spearman correlation of the suspicion (-log p1) with the known
    suspicion and, given the page's confident cut, how many of the known answer's confident units stay at or above it.
    Report-only beside: input gained and lost, agreement at the units whose known p1 is at least 0.6 and the near-ties
    below it, the confusion of the differing units and, given labels, both accuracies."""
    n = len(known_pred)
    cov = np.asarray(new["covered"], bool)
    kc = np.asarray(known_cov, bool)
    both = cov & kc
    nb = int(both.sum())
    kp1 = np.asarray(known_p1, np.float64)
    p1 = np.asarray(new["p1"], np.float64)
    same = both & (np.asarray(new["pred"]) == np.asarray(known_pred))
    dp1 = np.where(both, np.abs(p1 - kp1), np.nan)
    big = both & (dp1 > bars["dp1_tol"])
    tie = GATE_REPORT_ONLY["near_tie_p1"]
    firm = both & (kp1 >= tie)
    out = {"units": int(n), "known_covered": int(kc.sum()), "covered_here": int(cov.sum()), "covered_both": nb,
           "gained_input": int((cov & ~kc).sum()), "lost_input": int((~cov & kc).sum()),
           "same_class": int(same.sum()), "same_class_share": float(same.sum() / nb) if nb else None,
           "dp1_above_tol": int(big.sum()), "dp1_allowed": int(math.floor(bars["dp1_max_share"] * nb)),
           "dp1_mean": float(np.nanmean(dp1)) if nb else None, "dp1_max": float(np.nanmax(dp1)) if nb else None,
           "report_only": {"near_tie_p1": tie, "units_known_p1_at_least": int(firm.sum()),
                           "same_class_share_there": float((same & firm).sum() / firm.sum()) if firm.any() else None,
                           "near_ties_below": int((both & ~firm).sum()),
                           "near_ties_differing": int((both & ~firm & ~same).sum())},
           "confusion_differing": {}}
    for a, b in zip(np.asarray(known_pred)[both & ~same], np.asarray(new["pred"])[both & ~same]):
        k = f"{e98.CLASS_NAMES.get(int(a), int(a))} -> {e98.CLASS_NAMES.get(int(b), int(b))}"
        out["confusion_differing"][k] = out["confusion_differing"].get(k, 0) + 1
    ku = -np.log(kp1) if known_u is None else np.asarray(known_u, np.float64)
    if nb > 2:
        from scipy.stats import spearmanr
        r = spearmanr(np.asarray(new["suspicion"], np.float64)[both], ku[both]).statistic
        out["spearman_suspicion"] = float(r) if np.isfinite(r) else None
    else:
        out["spearman_suspicion"] = None
    if cut is not None:
        zk = both & (kp1 >= cut)
        zn = both & (p1 >= cut)
        out["confident_zone"] = {"cut": float(cut), "known": int(zk.sum()), "kept": int((zk & zn).sum()),
                                 "kept_share": float((zk & zn).sum() / zk.sum()) if zk.any() else None,
                                 "entered_report_only": int((zn & ~zk).sum())}
    if label is not None:
        lab = np.asarray(label)
        out["accuracy_here"] = float(((np.asarray(new["pred"]) == lab) & both).sum() / max(1, nb))
        out["accuracy_known"] = float(((np.asarray(known_pred) == lab) & both).sum() / max(1, nb))
    return out


def model_checks(c, bars=GATE_BARS, zone=False):
    """The rule set's checks of one model's compare(): class, dp1 and, at gate B, the suspicion's rank and the
    confident zone. A check whose quantity cannot be computed fails."""
    out = {"class": c["same_class_share"] is not None and c["same_class_share"] >= bars["same_class_min_share"],
           "dp1": c["covered_both"] > 0 and c["dp1_above_tol"] <= c["dp1_allowed"]}
    if zone:
        out["spearman"] = c["spearman_suspicion"] is not None and c["spearman_suspicion"] >= bars["spearman_suspicion_min"]
        z = c.get("confident_zone") or {}
        out["confident_zone"] = z.get("kept_share") is not None and z["kept_share"] >= bars["confident_zone_min_share"]
    return out


def gate_a(priv, bars=GATE_BARS, known=KNOWN, aprime=True, identity_flags=None):
    """Gate A (2023): A1/A2 v1 against exp98 (class, dp1), A3/A4 v1.2 against exp103's 2023 run; floor."""
    u = {m: np.load(known[("gate23", m)]) for m in MODELS}
    sel = (u["v1"]["split"] == "val") & u["v1"]["covered"]
    if not np.array_equal(u["v1"]["names"], u["v12"]["names"]):
        raise ValueError("exp98's and exp103's 2023 units are not in one order")
    units = read_json(os.path.join(priv, "units_gate23.json"))["units"]
    names = [str(n) for n in u["v1"]["names"][sel]]
    if [x["key"] for x in units] != names:
        raise ValueError("the gate-A units are not exp98's 259 in exp98's order")
    out = {"units": len(units), "bars": bars, "models": {}}
    for m in MODELS:
        new = _load_read(priv, "gate23", m)
        if new is None:
            raise FileNotFoundError(f"no read of gate23 for {m}")
        k = u[m]
        ku = k["suspicion"][sel] if "suspicion" in k.files else None
        out["models"][m] = compare(new, k["pred_deployed"][sel], k["p1"][sel], k["covered"][sel], ku, k["label"][sel],
                                   None, bars)
        if identity_flags:
            fl = read_json(identity_flags)
            idf = np.array([fl.get(x["name"]) == "identical" for x in units])
            for tag, msk in (("identical_items", idf), ("differing_items", ~idf)):
                sub = {kk: (vv[msk] if np.ndim(vv) and len(vv) == len(units) else vv) for kk, vv in new.items()}
                out["models"][m][tag] = compare(sub, k["pred_deployed"][sel][msk], k["p1"][sel][msk],
                                                k["covered"][sel][msk], None, None, None, bars)
    c1, c12 = model_checks(out["models"]["v1"], bars), model_checks(out["models"]["v12"], bars)
    floor = bars["floor_with_input_both"]["gate23"]
    out["checks"] = {"A1": c1["class"], "A2": c1["dp1"], "A3": c12["class"], "A4": c12["dp1"],
                     "A_floor": min(out["models"][m]["covered_both"] for m in MODELS) >= floor}
    out["pass"] = bool(all(out["checks"].values()))
    out["report_only"] = {"accuracy_within_points": {
        m: abs(out["models"][m]["accuracy_here"] - out["models"][m]["accuracy_known"])
        <= GATE_REPORT_ONLY["accuracy_points"] + 1e-12 for m in MODELS},
        "accuracy_note": "report-only: v1.2's classes given as v1's pass it (226 against 227 of the 259)"}
    ap = _load_read(priv, "aprime", "v1") if aprime else None
    if ap is not None:
        k = u["v1"]
        out["diagnostic_aprime"] = compare(ap, k["pred_deployed"][sel], k["p1"][sel], k["covered"][sel], None, None,
                                           None, bars)
    if "accuracy_known" in out["models"]["v1"]:
        from oe_inferencex import metrics
        lab = u["v1"]["label"][sel]
        new = _load_read(priv, "gate23", "v1")
        c = np.asarray(new["covered"], bool)
        err = (new["pred"][c] != lab[c]).astype(float)
        if 0 < err.sum() < err.size:
            out["report_only"]["auroc_new_p1_for_its_errors"] = float(
                metrics.weighted_auroc(new["suspicion"][c], err, np.ones(err.size)))
    return out


def gate_b(priv, bars=GATE_BARS, known=KNOWN, cut=CONFIDENT_CUT):
    """Gate B (2017, no label read): B1 (v1 against exp99) and B2 (v1.2 against exp103's 2017 run), each class, dp1,
    the suspicion's Spearman correlation and the confident zone; floor."""
    u = {m: np.load(known[("gate17", m)]) for m in MODELS}
    units = read_json(os.path.join(priv, "units_gate17.json"))["units"]
    names = [str(x) for x in u["v1"]["country_names"]]
    ctry = np.array([names[i] for i in u["v1"]["country"]])
    ours = np.array([x["country"] for x in units])
    inside = np.array([bool(x["inside"]) for x in units])
    if len(units) != len(ctry) or not (ours == ctry).all() or not (inside == u["v1"]["inside_awf"]).all():
        raise ValueError("the gate-B units do not map onto exp99's rows (country or inside flag differs)")
    out = {"units": len(units), "bars": bars, "mapping": "sorted plot ids; country and inside flags checked",
           "models": {}}
    for m in MODELS:
        new = _load_read(priv, "gate17", m)
        if new is None:
            raise FileNotFoundError(f"no read of gate17 for {m}")
        k = u[m]
        ku = k["suspicion"] if "suspicion" in k.files else None
        out["models"][m] = compare(new, k["pred"], k["p1"], k["covered"], ku, None, cut[m], bars)
    out["checks"] = {}
    for tag, m in (("B1", "v1"), ("B2", "v12")):
        for name, ok in model_checks(out["models"][m], bars, zone=True).items():
            out["checks"][f"{tag}_{name}"] = ok
    floor = bars["floor_with_input_both"]["gate17"]
    out["checks"]["B_floor"] = min(out["models"][m]["covered_both"] for m in MODELS) >= floor
    out["B1"] = all(v for k2, v in out["checks"].items() if k2.startswith("B1_"))
    out["B2"] = all(v for k2, v in out["checks"].items() if k2.startswith("B2_"))
    out["report_only"] = {m: {k2: out["models"][m][k2] for k2 in ("covered_here", "known_covered", "gained_input",
                                                                   "lost_input")} for m in MODELS}
    out["pass"] = bool(all(out["checks"].values()))
    return out


def run_gate(stage, priv, gate_file, bars=GATE_BARS, known=KNOWN, identity_dir=None, commit=None, device_file=None,
             cut=CONFIDENT_CUT):
    g = read_json(gate_file) if os.path.exists(gate_file) else {"experiment": "exp111 known-answer gate"}
    g["commit"] = commit or e89.git_state()[0]
    g["utc"] = e89.utc_now()
    g["bars"] = bars
    g["report_only_settings"] = GATE_REPORT_ONLY
    g["confident_cut"] = CONFIDENT_CUT
    if stage == "A":
        flags = os.path.join(identity_dir, "identity_gate23.json") if identity_dir else None
        g["A"] = gate_a(priv, bars, known, identity_flags=flags if flags and os.path.exists(flags) else None)
        g.pop("B", None)
        # the GPU the gate's predictions ran on (gpu_prep's record): the blind predict refuses another one; and the
        # commit the gate job ran (the record's commit): gatecheck and phase 2 refuse a guarded change since it
        g["device"] = read_json(device_file) if device_file and os.path.exists(device_file) else None
        g["predict_commit"] = (g["device"] or {}).get("commit")
    else:
        if not (g.get("A") or {}).get("pass"):
            raise PermissionError("refused: gate B reads exp99's plots only after gate A passed")
        g["B"] = gate_b(priv, bars, known, cut)
    g["pass"] = bool((g.get("A") or {}).get("pass") and (g.get("B") or {}).get("pass")
                     and (g.get("device") or {}).get("name")
                     and re.fullmatch(r"[0-9a-f]{40}", str(g.get("predict_commit") or ""))
                     and re.fullmatch(r"[0-9a-f]{40}", str(g.get("commit") or "")))
    dump(g, gate_file)
    return g


# ----------------------------------------------------------------------------- the units file
def cgls_by_pid(path, year=YEAR):
    import csv
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["year"]) == year and r.get("cgls_raw") not in (None, ""):
                out[r["pid"]] = int(float(r["cgls_raw"]))
    return out


def region_weights(country, stratum, covered, outside_km2, pool_counts):
    """(A'_c / N_c) x (N_cg / m_cg) per covered plot, 0 otherwise, normalised to sum 1: the region weight of the page
    (each country's pool a simple random sample of its area outside exp99's region; within it, each label cell drawn
    at its own rate; plots without input missing at random within their cell)."""
    w = np.zeros(len(country))
    N_c = {c: sum(v for k, v in pool_counts.items() if k.startswith(c + ":")) for c in COUNTRIES}
    for c in COUNTRIES:
        for s in STRATA:
            msk = (country == c) & (stratum == s) & covered
            if msk.any():
                w[msk] = (outside_km2[c] / N_c[c]) * (pool_counts[f"{c}:{s}"] / msk.sum())
    return w / w.sum() if w.sum() > 0 else w


def build_units(draw_rec, reads, cgls=None, in_cell=None):
    """Per model, exp99's units schema plus exp111's columns, in draw order (shuffled by the caller). in_cell: per drawn
    plot, in draw order, whether it lies in a gate17 cell (the blind units map's in_gate17_cell)."""
    us = draw_rec["units"]
    in_cell = np.zeros(len(us), bool) if in_cell is None else np.asarray(in_cell, bool)
    if in_cell.size != len(us):
        raise ValueError("in_gate17_cell does not hold one flag per drawn plot")
    country = np.array([u["country"] for u in us])
    stratum = np.array([u["stratum"] for u in us])
    label = np.array([u["label"] for u in us])
    tidx = {t: i for i, t in enumerate(tsx.TIMESYNC_CLASSES)}
    covs = [np.asarray(r["covered"], bool) for r in reads.values()]
    if any(not np.array_equal(covs[0], c) for c in covs[1:]):
        raise ValueError("the models' outputs cover different plots, though they read the same windows")
    cov = covs[0]
    w = region_weights(country, stratum, cov, draw_rec["area_outside_region_km2"], draw_rec["pool_counts"])
    common = {"country": np.array([COUNTRIES.index(c) for c in country], np.int64),
              "country_names": np.array(COUNTRIES),
              "label": np.array([tidx[t] for t in label], np.int64), "label_names": np.array(tsx.TIMESYNC_CLASSES),
              "covered": cov, "inside_awf": np.zeros(len(us), bool), "weight": w,
              "stratum": np.array([STRATA.index(s) for s in stratum], np.int64), "stratum_names": np.array(STRATA),
              "band": np.array([u["band"] for u in us], np.int64),
              "cgls_raw": np.array([(cgls or {}).get(u["pid"], -1) for u in us], np.int64),
              "in_gate17_cell": in_cell}
    out = {}
    for m, r in reads.items():
        d = dict(common)
        d.update(pred=r["pred"], suspicion=r["suspicion"], p1=r["p1"], condition=r["condition"],
                 majority_3x3=r["majority_3x3"])
        for rule in ("strict", "lenient"):
            e = np.full(len(us), np.nan)
            e[cov] = tsx.wrong(label[cov], r["pred"][cov], rule)
            d[f"wrong_{rule}"] = e
        out[m] = d
    return out


def shuffle_together(units_by_model, priv=None):
    n = len(next(iter(units_by_model.values()))["country"])
    perm = np.random.default_rng(np.random.SeedSequence()).permutation(n)
    if priv:
        np.save(os.path.join(priv, "units_permutation.npy"), perm)
    names = {"country_names", "label_names", "stratum_names"}
    out = {}
    for m, d in units_by_model.items():
        bad = [k for k, v in d.items() if k not in names and len(v) != n]
        if bad:
            raise ValueError(f"not per-row: {bad}")
        out[m] = {k: (v if k in names else np.asarray(v)[perm]) for k, v in d.items()}
    return out


def save_units(d, path):
    for k in d:
        if str(k).lower() in e98.FORBIDDEN_KEYS or "pid" in str(k).lower() or "id" == str(k).lower():
            raise ValueError(f"refusing to write units key {k!r}")
    np.savez_compressed(path, **d)


def run_units(priv, gate_file, out_dir, cgls_csv=None, smoke=False, log=print):
    if not smoke:
        status = prereg_status()
        if status != "frozen":
            raise PermissionError(f"refused: docs/plan/blind_transfer.md is {status}, not frozen; no blind output is "
                                  "joined with a label before it is")
        if not read_json(gate_file).get("pass"):
            raise PermissionError("refused: the known-answer gate did not pass")
        draw_sha = check_draw(priv)
    drw = read_json(os.path.join(priv, "draw.json"))
    if smoke:
        draw_sha = sha_ids([u["pid"] for u in drw["units"]])
    bu = read_json(os.path.join(priv, "units_blind17.json"))["units"]
    if [u["pid"] for u in bu] != [u["pid"] for u in drw["units"]]:
        raise ValueError("the blind units map is not the draw in draw order")
    reads = {m: _load_read(priv, "blind17", m) for m in MODELS}
    if any(v is None for v in reads.values()):
        raise FileNotFoundError("a blind read is missing")
    units = shuffle_together(build_units(drw, reads, cgls_by_pid(cgls_csv),
                                         [bool(u.get("in_gate17_cell")) for u in bu]), priv)
    os.makedirs(out_dir, exist_ok=True)
    alloc = {}
    for u in drw["units"]:
        alloc[f"{u['country']}:{u['stratum']}"] = alloc.get(f"{u['country']}:{u['stratum']}", 0) + 1
    # what the grader needs besides the units files, aggregates only: the draw it was frozen with, the region weights'
    # inputs (to recompute the weight column), the drawn plots per cell
    rec = {"experiment": "exp111 units (exp99's schema per model; rows shuffled, permutation private)",
           "utc": e89.utc_now(), "n_plots": len(drw["units"]), "draw_sha256": draw_sha,
           "sizes_O_W_X": [int(drw["sizes"][k]) for k in STRATA], "drawn_by_cell": alloc, "pool_counts": drw["pool_counts"],
           "area_outside_region_km2": drw["area_outside_region_km2"], "band_cuts_km": drw["band_cuts_km"],
           "in_gate17_cell": int(sum(bool(u.get("in_gate17_cell")) for u in bu)), "models": {}}
    for m, d in units.items():
        save_units(d, os.path.join(out_dir, f"exp111_units_{m}.npz"))
        c = d["covered"]
        rec["models"][m] = {"covered": int(c.sum()), "errors_strict": int(np.nansum(d["wrong_strict"])),
                            "errors_lenient": int(np.nansum(d["wrong_lenient"]))}
    rec["sha256"] = {f"exp111_units_{m}.npz": jobs.sha256_file(os.path.join(out_dir, f"exp111_units_{m}.npz"))
                     for m in units}
    dump(rec, os.path.join(out_dir, "exp111_units_record.json"))
    log(json.dumps(rec["models"]))
    return rec


# ----------------------------------------------------------------------------- the synthetic smoke
# Two synthetic "countries", kenya north and tanzania south of a latitude line, a 0.06-degree request rectangle and
# plots at known distances: 24 within 20 km (the gate-B analogue, D = 20 km at a floor of 22) and a pool beyond it,
# labelled so that every cell holds plots. Windows for every gate and drawn plot, then synthetic outputs: a softmax per
# pixel whose top class encodes the window index, SCL item groups, and the reader, the gate and the units on them.
SMOKE_RECT = (36.90, -1.33, 36.96, -1.27)      # in UTM 37S, near a zone and a cell boundary
SMOKE_BORDER_LAT = -1.30
SMOKE_SIZES = {"O": 4, "W": 4, "X": 3}
SMOKE_LABELS = ("Open Grassland", "Wooded Grassland", "Cropland", "Otherland", "Open Forest")


def make_smoke_sample(root):
    x0, y0, x1, y1 = SMOKE_RECT
    rows, pid = [], 0
    rng = np.random.default_rng(5)

    def add(country, lon, lat, label):
        nonlocal pid
        pid += 1
        for y in (2016, 2017):
            rows.append(f"{pid},{y},{country},0,{lat:.10f},{lon:.10f},{label}")
    for k in range(24):                              # near: inside or within about 15 km
        north = k % 3 != 0
        lat = (SMOKE_BORDER_LAT + 0.01 + 0.002 * (k % 5)) if north else (SMOKE_BORDER_LAT - 0.01 - 0.002 * (k % 5))
        lon = x0 + 0.004 * (k % 12) if k < 12 else x1 + 0.03 + 0.005 * (k % 7)
        add("kenya" if north else "tanzania", lon, lat, SMOKE_LABELS[k % len(SMOKE_LABELS)])
    for k in range(60):                              # the pool: 0.5 to 1.5 degrees away
        north = k % 2 == 0
        lat = SMOKE_BORDER_LAT + (0.3 + 0.01 * k) * (1 if north else -1)
        lon = x0 - 0.5 - rng.uniform(0, 1.0)
        add("kenya" if north else "tanzania", lon, lat, SMOKE_LABELS[k % len(SMOKE_LABELS)])
    paths = {"timesync": os.path.join(root, "timesync.csv"), "request_geometry": os.path.join(root, "req.geojson"),
             "boundaries": os.path.join(root, "countries.geojson")}
    with open(paths["timesync"], "w") as f:
        f.write("plotid,year,country,country_id,latitude,longitude,landcover\n" + "\n".join(rows) + "\n")

    def poly(a, b, c, d):
        return {"type": "Polygon", "coordinates": [[[a, b], [c, b], [c, d], [a, d], [a, b]]]}
    with open(paths["request_geometry"], "w") as f:
        json.dump({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                                                             "geometry": poly(x0, y0, x1, y1)}]}, f)
    with open(paths["boundaries"], "w") as f:
        json.dump({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"ADM0_A3": "KEN"}, "geometry": poly(34.0, SMOKE_BORDER_LAT, 38.0, 1.0)},
            {"type": "Feature", "properties": {"ADM0_A3": "TZA"}, "geometry": poly(34.0, -4.0, 38.0, SMOKE_BORDER_LAT)}]},
            f)
    return paths


SMOKE_POINTS = (("interior, zone 37S", 36.9312, -1.2987), ("near a cell's west edge, zone 37S", 36.0004, -2.1002),
                ("north of the equator, zone 36N", 35.9899, 0.4413))


def smoke_geometry():
    """The window rule on three synthetic points, checked against the rule's own claims and exp111_jobs.owner_1d."""
    out = []
    for what, lon, lat in SMOKE_POINTS:
        epsg = utm_epsg(lon, lat)
        px, py, _, _ = project_pixel(epsg, lon, lat)
        u = make_unit("s", epsg, px, py)
        b, (X0, Y0) = u["bounds"], u["cell"]
        crop_same = (jobs.owner_1d(px, b[0], b[2]) == jobs.owner_1d(px, X0, X0 + CELL_PX)
                     and jobs.owner_1d(py, b[1], b[3]) == jobs.owner_1d(py, Y0, Y0 + CELL_PX))
        out.append({"what": what, "epsg": epsg, "size": [b[2] - b[0], b[3] - b[1]], "lead": [px - b[0], py - b[1]],
                    "start_on_lattice": [(b[0] - X0) % STRIDE_PX == 0, (b[1] - Y0) % STRIDE_PX == 0],
                    "inside_cell": X0 <= b[0] and b[2] <= X0 + CELL_PX and Y0 <= b[1] and b[3] <= Y0 + CELL_PX,
                    "plot_crop_is_the_cells": crop_same, "flush": [b[0] == X0, b[1] == Y0, b[2] == X0 + CELL_PX,
                                                                  b[3] == Y0 + CELL_PX]})
    # every pixel offset in a cell: size, lattice, containment and the crop the cell keeps
    bad = 0
    for p in range(0, CELL_PX):
        s, e, c0 = snap(p)
        bad += not (e - s == WINDOW_PX and c0 <= s and e <= c0 + CELL_PX and (s - c0) % STRIDE_PX == 0 and s <= p < e
                    and jobs.owner_1d(p, s, e) == jobs.owner_1d(p, c0, c0 + CELL_PX))
    return out, bad


def smoke_vector(k, cls):
    p = np.full(10, 0.02)
    p[9] = 0.0
    p[cls] = 0.0
    p[cls] = 1.0 - p.sum()
    p[(cls + 1) % 9] += 0.001 * (k % 7)
    p[cls] -= 0.001 * (k % 7)
    return p


def write_smoke_outputs(priv, root, model, cls_of, scl_ok=True):
    """Synthetic predictions for every window of a root: 12 completed sentinel2 groups (an empty band set), an output
    layer whose every pixel holds smoke_vector(k, background) except the unit pixel (smoke_vector(k, cls_of(k))), and
    the SCL sidecar's 12 groups (code 4, vegetation; 9, cloud, on the unit pixel in group 0)."""
    import rasterio
    from rasterio.transform import from_origin
    m = read_json(os.path.join(priv, f"units_{root}.json"))
    ds, scl = os.path.join(priv, "roots", root), os.path.join(priv, "scl", root)
    os.makedirs(scl, exist_ok=True)
    shutil.copyfile(SCL_CONFIG, os.path.join(scl, "config.json"))
    for k, u in enumerate(m["units"]):
        b = u["bounds"]
        H, W = b[3] - b[1], b[2] - b[0]
        tr = from_origin(b[0] * RES, -b[1] * RES, RES, RES)
        wd = os.path.join(ds, "windows", GROUP, u["name"])
        for gidx in range(jobs.N_GROUPS):
            gd = os.path.join(wd, "layers", jobs.INPUT if gidx == 0 else f"{jobs.INPUT}.{gidx}")
            os.makedirs(gd, exist_ok=True)
            open(os.path.join(gd, "completed"), "w").close()
        bg = smoke_vector(k, 4)
        arr = np.repeat(np.repeat(bg[:, None, None], H, 1), W, 2).astype(np.float32)
        arr[:, u["py"] - b[1], u["px"] - b[0]] = smoke_vector(k, cls_of(k)).astype(np.float32)
        od = os.path.join(wd, "layers", MODELS[model], jobs.P)
        os.makedirs(od, exist_ok=True)
        with rasterio.open(os.path.join(od, "geotiff.tif"), "w", driver="GTiff", height=H, width=W, count=10,
                           dtype="float32", crs=f"EPSG:{u['epsg']}", transform=tr) as dst:
            dst.write(arr)
        open(os.path.join(wd, "layers", MODELS[model], "completed"), "w").close()
        sw = os.path.join(scl, "windows", GROUP, u["name"])
        os.makedirs(sw, exist_ok=True)
        shutil.copyfile(os.path.join(wd, "metadata.json"), os.path.join(sw, "metadata.json"))
        for gidx in range(jobs.N_GROUPS):
            gd = os.path.join(sw, "layers", SCL if gidx == 0 else f"{SCL}.{gidx}", "SCL")
            os.makedirs(gd, exist_ok=True)
            a = np.full((1, H, W), 4, np.uint8)
            if gidx == 0:
                a[0, u["py"] - b[1], u["px"] - b[0]] = 9
            with rasterio.open(os.path.join(gd, "geotiff.tif"), "w", driver="GTiff", height=H, width=W, count=1,
                               dtype="uint8", crs=f"EPSG:{u['epsg']}", transform=tr) as dst:
                dst.write(a)
            open(os.path.join(os.path.dirname(gd), "completed"), "w").close()


import contextlib   # noqa: E402  (the smoke's stand-in page status)


@contextlib.contextmanager
def _pretend_status(status):
    """The smoke only: prereg_status() answers `status` inside the block."""
    global prereg_status
    keep = prereg_status
    prereg_status = lambda text=None: status            # noqa: E731
    try:
        yield
    finally:
        prereg_status = keep


def glob_metadata(run):
    """Every metadata.json under a deployment run directory (the smoke's partial purge)."""
    out = []
    for dp, _, fs in os.walk(run):
        out += [os.path.join(dp, f) for f in fs if f == "metadata.json"]
    return out


def _smoke_deployment(tmp, blind):
    """A synthetic 2017 deployment in exp99's layout (inventory + run directory): one 1,024-px window in a CRS other
    than blind unit 0's own zone that holds its point, one in unit 1's own zone that holds its pixel, one far away."""
    import pyproj
    run = os.path.join(tmp, "deploy17_run")
    wins = []
    u0, u1 = blind[0], blind[1]
    other = 32736 if int(u0["epsg"]) != 32736 else 32737
    X, Y = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{other}", always_xy=True).transform(u0["lon"], u0["lat"])
    px, py = int(math.floor(X / RES)), int(math.floor(Y / -RES))
    wins.append((other, [CELL_PX * (px // CELL_PX), CELL_PX * (py // CELL_PX)]))
    wins.append((int(u1["epsg"]), list(u1["cell"])))
    wins.append((int(u1["epsg"]), [u1["cell"][0] + 50 * CELL_PX, u1["cell"][1]]))
    for k, (epsg, (x0, y0)) in enumerate(wins):
        wd = os.path.join(run, "rootdir", "windows", f"group_{k}", f"w{k}")
        os.makedirs(wd)
        with open(os.path.join(wd, "metadata.json"), "w") as f:
            json.dump({"projection": {"crs": f"EPSG:{epsg}", "x_resolution": RES, "y_resolution": -RES},
                       "bounds": [x0, y0, x0 + CELL_PX, y0 + CELL_PX], "time_range": list(TIME_RANGES[2017]),
                       "options": {}}, f)
    inv = os.path.join(tmp, "deploy17_inventory.json")
    with open(inv, "w") as f:
        json.dump({"roots": [{"root_id": "root_0", "root": "rootdir"}]}, f)
    return inv, run, other


def smoke(out_dir=None, log=print):
    import tempfile
    out_dir = out_dir or os.path.join(OUT, "exp111_smoke")
    os.makedirs(out_dir, exist_ok=True)
    # the records this smoke writes, named as the cluster's $REC names them; the local smoke runs the pack jobs' check
    # (exp111_jobs.py pack-check) over this directory
    recdir = os.path.join(out_dir, "rec")
    os.makedirs(recdir, exist_ok=True)
    rec = {"mode": "smoke: synthetic plots, geometry, outputs; no TimeSync file, no imagery, no model"}
    geo, bad_offsets = smoke_geometry()
    rec["window_rule"] = {"points": geo, "cell_offsets_failing_the_rule": bad_offsets}
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_smoke_sample(tmp)
        priv = os.path.join(tmp, "priv")
        P, sel, _ = e99.select(paths["timesync"], paths["request_geometry"], paths["boundaries"], None, 22, None)
        n_gate = int(sel["n_selected"])
        # the draw: allocation, determinism, nesting
        s_rec = run_select(paths["timesync"], paths["request_geometry"], paths["boundaries"], priv, SMOKE_SIZES, SEED,
                           None, None, 22, None, log=lambda *_: None)
        d1 = read_json(os.path.join(priv, "draw.json"))
        run_select(paths["timesync"], paths["request_geometry"], paths["boundaries"], priv + "_b",
                            {"O": 4, "W": 4, "X": 2}, SEED, None, None, 22, None, log=lambda *_: None)
        d2 = read_json(os.path.join(priv + "_b", "draw.json"))
        by_cell = {}
        for d in (d1, d2):
            for u in d["units"]:
                by_cell.setdefault(id(d), {}).setdefault(f"{u['country']}:{u['stratum']}", []).append(u["pid"])
        nested = all(by_cell[id(d2)].get(c, []) == by_cell[id(d1)][c][:len(by_cell[id(d2)].get(c, []))]
                     for c in by_cell[id(d1)])
        again = run_select(paths["timesync"], paths["request_geometry"], paths["boundaries"], priv + "_c", SMOKE_SIZES,
                           SEED, None, None, 22, None, log=lambda *_: None)
        # a rerun over the same private directory keeps the same draw; a different draw is refused unless redo
        rerun = run_select(paths["timesync"], paths["request_geometry"], paths["boundaries"], priv, SMOKE_SIZES, SEED,
                           None, None, 22, None, log=lambda *_: None)
        try:
            run_select(paths["timesync"], paths["request_geometry"], paths["boundaries"], priv, {"O": 4, "W": 4, "X": 2},
                       SEED, None, None, 22, None, log=lambda *_: None)
            other_refused = False
        except ValueError:
            other_refused = True
        redone = run_select(paths["timesync"], paths["request_geometry"], paths["boundaries"], priv + "_c",
                            {"O": 4, "W": 4, "X": 2}, SEED, None, None, 22, None, log=lambda *_: None, redo=True)
        dump(s_rec, os.path.join(recdir, "e111_select.json"))       # the record passes the no-position guard
        # E111_REDO=select after freezing is refused, before anything is replaced
        redo_frozen = {}
        with _pretend_status("frozen"):
            try:
                run_select(paths["timesync"], paths["request_geometry"], paths["boundaries"], priv + "_c",
                           {"O": 4, "W": 4, "X": 2}, SEED, None, None, 22, None, log=lambda *_: None, redo=True)
                redo_frozen["select"] = False
            except PermissionError:
                redo_frozen["select"] = True
        rec["select"] = {"gate_plots": n_gate, "pool": s_rec["pool"], "draw": s_rec["draw"],
                         "same_sha_twice": again["draw"]["sha256_sorted_ids"] == s_rec["draw"]["sha256_sorted_ids"],
                         "smaller_size_is_a_prefix": nested, "band_cuts_km": s_rec["distance"]["band_cuts_km"],
                         "areas_outside_km2": s_rec["areas"]["outside_exp99_region_km2"],
                         "versions": s_rec["versions"],
                         "rerun_keeps_the_same_draw": rerun["private_files"] == {"gate17.json": "kept", "draw.json": "kept"},
                         "rerun_refuses_another_draw": other_refused,
                         "redo_replaces_it": redone["private_files"]["draw.json"] == "written"}
        # gate-A analogue: three synthetic label points, written straight into a units list
        g23 = [make_unit(f"a{k:04d}", 32737, *project_pixel(32737, lon, lat)[:2], key=f"syn_point_{k}",
                         X=project_pixel(32737, lon, lat)[2], Y=project_pixel(32737, lon, lat)[3])
               for k, (lon, lat) in enumerate(((36.93, -1.31), (36.95, -1.28), (36.91, -1.32)))]
        w = {"gate23": g23, "gate17": timesync_units(read_json(os.path.join(priv, "gate17.json"))["units"], "g"),
             "blind17": timesync_units(d1["units"], "b")}
        for u, f in zip(w["blind17"], in_cells_of(w["blind17"], w["gate17"])):
            u["in_gate17_cell"] = bool(f)
        hows = {}
        for name, us in w.items():
            check_units(us, len(us))
            hows[name] = write_root(os.path.join(priv, "roots", name), us, ROOTS[name]["year"])[1]
            write_private({"root": name, "year": ROOTS[name]["year"], "group": GROUP, "units": us},
                          os.path.join(priv, f"units_{name}.json"))
        rec["windows"] = {name: {"windows": len(us), **unit_facts(us)} for name, us in w.items()}
        # a rerun keeps a root that is this run's; a changed root is refused unless redo (which rebuilds it)
        kept = write_root(os.path.join(priv, "roots", "gate17"), w["gate17"], 2017)[1]
        wd0 = os.path.join(priv, "roots", "gate17", "windows", GROUP, w["gate17"][0]["name"], "metadata.json")
        md0 = read_json(wd0)
        md0["bounds"] = [b + STRIDE_PX for b in md0["bounds"]]
        with open(wd0, "w") as f:
            json.dump(md0, f)
        try:
            write_root(os.path.join(priv, "roots", "gate17"), w["gate17"], 2017)
            changed_refused = False
        except ValueError:
            changed_refused = True
        rebuilt = write_root(os.path.join(priv, "roots", "gate17"), w["gate17"], 2017, redo=True)[1]
        # run_windows itself over the same private directory (gate23 has no label windows here): the roots are kept,
        # and the in_gate17_cell flag takes exp99's deployment windows when they are given (a synthetic deployment:
        # one window in another CRS holding blind unit 0, one holding unit 1), the own-cell rule otherwise
        n_exp = {"gate23": 0, "gate17": len(w["gate17"]), "blind17": len(w["blind17"])}
        wr_cell = run_windows(priv, None, n_expected=n_exp, log=lambda *_: None)
        inv17, run17, other_crs = _smoke_deployment(tmp, w["blind17"])
        n_dep = len(jobs.deployment_windows(inv17, run17))
        wr_dep = run_windows(priv, None, n_expected=n_exp, log=lambda *_: None, deploy17=(inv17, run17),
                             deploy17_windows=n_dep)
        dump(wr_dep, os.path.join(recdir, "e111_windows.json"))
        flags_dep = [u["in_gate17_cell"] for u in read_json(os.path.join(priv, "units_blind17.json"))["units"]]
        # the same deployment partly purged (one window's metadata.json gone, its directory kept): the step falls back
        # to the own-cell rule and says why, instead of failing select
        part = os.path.join(tmp, "deploy17_partial")
        shutil.copytree(run17, part)
        gone = sorted(glob_metadata(part))[0]
        os.remove(gone)
        wr_part = run_windows(priv, None, n_expected=n_exp, log=lambda *_: None, deploy17=(inv17, part),
                              deploy17_windows=n_dep)
        dump(wr_part, os.path.join(tmp, "windows_partial.json"))
        flags_part = [u["in_gate17_cell"] for u in read_json(os.path.join(priv, "units_blind17.json"))["units"]]
        # a whole window directory gone (list_windows does not see it): fewer windows than the inventory's, partial
        part2 = os.path.join(tmp, "deploy17_dir_gone")
        shutil.copytree(run17, part2)
        shutil.rmtree(os.path.dirname(sorted(glob_metadata(part2))[0]))
        wr_part2 = run_windows(priv, None, n_expected=n_exp, log=lambda *_: None, deploy17=(inv17, part2),
                               deploy17_windows=n_dep)
        flags_part2 = [u["in_gate17_cell"] for u in read_json(os.path.join(priv, "units_blind17.json"))["units"]]
        run_windows(priv, None, n_expected=n_exp, log=lambda *_: None, deploy17=(inv17, run17),
                    deploy17_windows=n_dep)   # back to the whole
        own_cell = in_cells_of(w["blind17"], w["gate17"])
        # the deployment rule recomputed one plot and one window at a time (project_pixel), independently of
        # in_deployment_windows' vectorised version
        dep_w = jobs.deployment_windows(inv17, run17)
        indep = [any(dw[6][0] <= project_pixel(dw[5], u["lon"], u["lat"])[0] < dw[6][2]
                     and dw[6][1] <= project_pixel(dw[5], u["lon"], u["lat"])[1] < dw[6][3] for dw in dep_w)
                 for u in w["blind17"]]
        with _pretend_status("frozen"):
            try:
                run_windows(priv, None, redo=True, n_expected=n_exp, log=lambda *_: None)
                redo_frozen["windows"] = False
            except PermissionError:
                redo_frozen["windows"] = True
        rec["windows_record"] = {
            "roots_kept": all(v.get("root") == "kept" for k, v in wr_dep["roots"].items() if k != "gate23"),
            "own_cell_method": wr_cell["blind_plots_in_a_gate17_cell"]["method"].startswith("a gate17 unit's"),
            "deployment_method": wr_dep["blind_plots_in_a_gate17_cell"]["method"].startswith("exp99's 2017"),
            "by_deployment_windows_as_recomputed": wr_dep["blind_plots_in_a_gate17_cell"]["n_by_deployment_windows"]
            == sum(indep),
            "other_crs_window_flags_unit0": bool(flags_dep[0] and indep[0]),
            "own_crs_window_flags_unit1": bool(flags_dep[1] and indep[1]),
            "union_with_own_cell": flags_dep == [bool(a or b) for a, b in zip(own_cell, indep)],
            "deployment_crs_other_than_unit0": other_crs != int(w["blind17"][0]["epsg"]),
            "partial_purge_falls_back": bool(
                wr_part["blind_plots_in_a_gate17_cell"]["method"].startswith("a gate17 unit's")
                and wr_part["blind_plots_in_a_gate17_cell"]["deployment_2017"].startswith("unreadable (FileNotFoundError")
                and wr_part["blind_plots_in_a_gate17_cell"]["n_by_deployment_windows"] is None
                and flags_part == own_cell),
            "window_directory_gone_falls_back": bool(
                wr_part2["blind_plots_in_a_gate17_cell"]["method"].startswith("a gate17 unit's")
                and wr_part2["blind_plots_in_a_gate17_cell"]["deployment_2017"].startswith(f"partial ({n_dep - 1} of")
                and flags_part2 == own_cell),
            "redo_refused_when_frozen": redo_frozen}
        # the gate17-cell count on a constructed case: one blind unit placed in a gate unit's cell
        probe = [dict(w["blind17"][0]), dict(w["blind17"][1], epsg=w["gate17"][0]["epsg"], cell=w["gate17"][0]["cell"])]
        rec["windows_rerun"] = {"first": hows, "rerun_kept": kept == "kept", "changed_root_refused": changed_refused,
                                "redo_rebuilt": rebuilt == "written",
                                "in_gate17_cell_smoke_draw": int(sum(u["in_gate17_cell"] for u in w["blind17"])),
                                "in_gate17_cell_probe": in_cells_of(probe, w["gate17"])}
        # synthetic outputs: v1's class at unit k is k % 9, v1.2's the same except every fifth unit
        reads = {}
        for name in w:
            for m in MODELS:
                write_smoke_outputs(priv, name, m, (lambda k: k % 9) if m == "v1" else (lambda k: (k + (k % 5 == 0)) % 9))
                r = run_read(priv, name, m, log=lambda *_: None) if name != "blind17" else None
                if name == "blind17":
                    r = _smoke_read_blind(priv, name, m)
                reads[f"{name}_{m}"] = r
        rec["read"] = {k: {kk: v[kk] for kk in ("units", "covered", "read_from_own_window", "direct_route", "ok")}
                       for k, v in reads.items()}
        for k, v in reads.items():
            dump(v, os.path.join(recdir, f"e111_read_{k}.json"))
        # a copy of the roots with their synthetic outputs, for the rslearn-side smoke (exp111_jobs.py in the venv)
        keep = os.path.join(out_dir, "smoke_private")
        if os.path.lexists(keep):
            shutil.rmtree(keep)
        shutil.copytree(priv, keep, symlinks=True, ignore=shutil.ignore_patterns("view_*", "parts_*"))
        got = _load_read(priv, "gate17", "v1")
        rec["read"]["classes_as_written"] = bool((got["pred"] == np.arange(len(got["pred"])) % 9).all())
        rec["read"]["condition_codes"] = sorted({int(c) for c in got["condition"]})
        # the gate on synthetic known answers: known = what was written, with two gate-A units changed
        kn = _smoke_known(priv, tmp, w)
        sbars = dict(GATE_BARS, floor_with_input_both={"gate23": len(w["gate23"]), "gate17": len(w["gate17"])})
        ga = gate_a(priv, bars=sbars, known=kn, aprime=False)
        gfile = os.path.join(tmp, "gate.json")
        g = {"A": ga}
        jobs.write_json(e98_jsonable(g), gfile)
        scut = {m: 0.8 for m in MODELS}            # the synthetic p1 sit near 0.82: every unit is "confident"
        gb = gate_b(priv, bars=sbars, known=kn, cut=scut)
        rec["gate"] = {"A_checks": ga["checks"], "A_v1_same": ga["models"]["v1"]["same_class"],
                       "A_v12_same": ga["models"]["v12"]["same_class"], "B_checks": gb["checks"],
                       "B_v1_same_share": gb["models"]["v1"]["same_class_share"]}
        # the designed answers: one of three v1 classes differs (A1 fails; A2, A3, A4 and the floor hold), B agrees
        # everywhere (every B check holds)
        rec["gate"]["as_designed"] = bool(ga["models"]["v1"]["same_class"] == 2 and not ga["checks"]["A1"]
                                          and ga["checks"]["A2"] and ga["checks"]["A3"] and ga["checks"]["A4"]
                                          and ga["checks"]["A_floor"] and gb["pass"] and gb["B1"] and gb["B2"])
        # run_gate on a passing synthetic known answer (no unit changed): it passes with the GPU's name and both commits
        # (gatecheck's and the gate job's, from its device record), and does not pass without the gate job's commit
        kn_pass = _smoke_known(priv, tmp, w, differ=False)
        dev_ok = os.path.join(tmp, "device_gate.json")
        jobs.write_json({"name": "NVIDIA B200", "commit": "b" * 40, "job": None}, dev_ok)
        dev_nocommit = os.path.join(tmp, "device_gate_nocommit.json")
        jobs.write_json({"name": "NVIDIA B200", "commit": None, "job": None}, dev_nocommit)
        gate_rec = os.path.join(recdir, "e111_gate.json")
        if os.path.exists(gate_rec):
            os.remove(gate_rec)
        run_gate("A", priv, gate_rec, bars=sbars, known=kn_pass, commit="a" * 40, device_file=dev_ok, cut=scut)
        g_ok = run_gate("B", priv, gate_rec, bars=sbars, known=kn_pass, commit="a" * 40, cut=scut)
        g_nc_file = os.path.join(tmp, "gate_nocommit.json")
        run_gate("A", priv, g_nc_file, bars=sbars, known=kn_pass, commit="a" * 40, device_file=dev_nocommit, cut=scut)
        g_nc = run_gate("B", priv, g_nc_file, bars=sbars, known=kn_pass, commit="a" * 40, cut=scut)
        rec["gate_commits"] = {"passes_with_both": bool(g_ok["pass"] and g_ok["predict_commit"] == "b" * 40
                                                        and g_ok["commit"] == "a" * 40),
                               "fails_without_predict_commit": bool(g_nc["A"]["pass"] and g_nc["B"]["pass"]
                                                                    and not g_nc["pass"])}
        # the refusals: gate B after a failed A, the units before freezing
        refused = []
        try:
            run_gate("B", priv, gfile, known=kn, commit="smoke")
        except PermissionError:
            refused.append("gate B after a failed gate A")
        try:
            run_units(priv, gfile, os.path.join(tmp, "units_refused"), log=lambda *_: None)
        except PermissionError:
            refused.append("units before freezing")
        sha1 = sha_ids([u["pid"] for u in d1["units"]])
        for what, text in (("a page without a draw sha256", "**Status: frozen**\n"),
                           ("a page with another draw", "**Status: frozen**\n**Draw sha256 (frozen):** `" + "0" * 64 + "`\n")):
            try:
                check_draw(priv, text, SMOKE_SIZES)
            except PermissionError:
                refused.append(f"check-draw against {what}")
        rec["check_draw_accepts_the_page_draw"] = check_draw(
            priv, "**Status: frozen**\n**Draw sha256 (frozen):** `" + sha1 + "`\n", SMOKE_SIZES) == sha1
        # the sizes are the commit's: the page's own draw at other sizes is refused by check_draw (as the phase-2
        # guard and the units call it, with SIZES), and check_sizes refuses exp109's Option A against 300/300/300
        sizes_refused = []
        try:
            check_draw(priv, "**Status: frozen**\n**Draw sha256 (frozen):** `" + sha1 + "`\n")
        except PermissionError:
            sizes_refused.append("the smoke's draw against SIZES")
        for other in ({"O": 300, "W": 300, "X": 150}, {"O": 400, "W": 400, "X": 200}):
            try:
                check_sizes(other, sum(other.values()))
            except PermissionError:
                sizes_refused.append(f"{other['O']}/{other['W']}/{other['X']}")
        try:
            check_sizes(SIZES, sum(SIZES.values()) - 1)
        except PermissionError:
            sizes_refused.append("a plot count that is not the sizes' sum")
        check_sizes(SIZES, sum(SIZES.values()))
        rec["sizes_refused"] = sizes_refused
        rec["refusals"] = refused
        rec["gate_on_committed_files"] = smoke_gate_on_committed(tmp) if all(os.path.exists(v) for v in KNOWN.values()) \
            else {"as_expected": None, "note": "the committed units files are not here"}
        # the units: exp99's keys present, both models in one shuffled order
        res = run_units(priv, gfile, os.path.join(tmp, "units"), smoke=True, log=lambda *_: None)
        shutil.copyfile(os.path.join(tmp, "units", "exp111_units_record.json"),
                        os.path.join(recdir, "exp111_units_record.json"))
        u1 = np.load(os.path.join(tmp, "units", "exp111_units_v1.npz"))
        u12 = np.load(os.path.join(tmp, "units", "exp111_units_v12.npz"))
        e99_keys = set(np.load(KNOWN[("gate17", "v1")]).files) if os.path.exists(KNOWN[("gate17", "v1")]) else set()
        rec["units"] = {"record": res["models"], "has_exp99_keys": e99_keys <= set(u1.files) if e99_keys else None,
                        "same_row_order": bool((u1["label"] == u12["label"]).all() and (u1["band"] == u12["band"]).all()),
                        "weights_sum": float(u1["weight"].sum()), "keys": sorted(u1.files),
                        "in_gate17_cell_column": bool("in_gate17_cell" in u1.files
                                                      and int(u1["in_gate17_cell"].sum()) == res["in_gate17_cell"])}
        # the grader reads these units files as written (schema, weights, errors, suspicion); 11 plots are below
        # every floor, so every verdict must be "no verdict"
        import exp111_grade as gr
        ug, rg = gr.load_units(os.path.join(tmp, "units"))
        gg = gr.grade(ug, rg, R=50, log=lambda *_: None)
        rec["units"]["grader_reads_them"] = all(gg["models"][m][p]["verdict"].startswith("no verdict")
                                                for m in gr.MODELS for p in ("P1", "P2", "P3"))
    rec["ok"] = bool(not bad_offsets and all(all(p["start_on_lattice"]) and p["inside_cell"] and p["plot_crop_is_the_cells"]
                                             and p["size"] == [WINDOW_PX, WINDOW_PX] for p in geo)
                     and rec["select"]["same_sha_twice"] and rec["select"]["smaller_size_is_a_prefix"]
                     and all(v["ok"] for k, v in rec["read"].items() if isinstance(v, dict))
                     and rec["read"]["classes_as_written"] and rec["units"]["same_row_order"]
                     and rec["gate"]["as_designed"] and len(rec["refusals"]) == 4
                     and rec["check_draw_accepts_the_page_draw"]
                     and len(rec["sizes_refused"]) == 4
                     and rec["select"]["rerun_keeps_the_same_draw"] and rec["select"]["rerun_refuses_another_draw"]
                     and rec["select"]["redo_replaces_it"] and rec["windows_rerun"]["rerun_kept"]
                     and rec["windows_rerun"]["changed_root_refused"] and rec["windows_rerun"]["redo_rebuilt"]
                     and rec["windows_rerun"]["in_gate17_cell_probe"] == [False, True]
                     and rec["units"]["in_gate17_cell_column"] and rec["units"]["grader_reads_them"]
                     and rec["gate_on_committed_files"]["as_expected"] is not False
                     and rec["units"]["has_exp99_keys"] is not False and abs(rec["units"]["weights_sum"] - 1) < 1e-9
                     and all(v is True for k, v in rec["windows_record"].items() if k != "redo_refused_when_frozen")
                     and rec["windows_record"]["redo_refused_when_frozen"] == {"select": True, "windows": True}
                     and rec["gate_commits"]["passes_with_both"] and rec["gate_commits"]["fails_without_predict_commit"])
    jobs.write_json(e98_jsonable(rec), os.path.join(out_dir, "exp111_smoke.json"))
    log(json.dumps({"ok": rec["ok"], "gate_A": rec["gate"]["A_checks"], "gate_B": rec["gate"]["B_checks"]}))
    return rec


def smoke_gate_on_committed(tmp, known=KNOWN):
    """The gate's code on the committed files themselves (reads only committed units files). 'New' reads equal to
    exp98's, exp103's and exp99's committed values must pass every check. v1.2's values given as v1's must fail A1, A2,
    and B1's four checks (the models agree at about 88% and 82%, |dp1| > 0.05 at 49 and 87 units, Spearman about 0.78
    and 0.75, 65 of v1's 91 confident plots at or above v1's cut). p1 lowered by 0.06 at six units below p1 0.9, classes kept, must fail only
    A2 and B1's dp1; five of v1's 91 confident plots moved just below the cut must fail only B1's confident zone."""
    priv = os.path.join(tmp, "committed_priv")
    os.makedirs(os.path.join(priv, "read"), exist_ok=True)
    u = {k: np.load(v) for k, v in known.items()}
    sel = (u[("gate23", "v1")]["split"] == "val") & u[("gate23", "v1")]["covered"]
    names = [str(n) for n in u[("gate23", "v1")]["names"][sel]]
    write_private({"root": "gate23", "units": [{"name": f"a{k:04d}", "key": n} for k, n in enumerate(names)]},
                  os.path.join(priv, "units_gate23.json"))
    c17 = u[("gate17", "v1")]
    cn = [str(x) for x in c17["country_names"]]
    write_private({"root": "gate17", "units": [{"name": f"g{k:04d}", "country": cn[int(c)], "inside": bool(i)}
                                               for k, (c, i) in enumerate(zip(c17["country"], c17["inside_awf"]))]},
                  os.path.join(priv, "units_gate17.json"))

    def put(root, model, src, msk=None, edit=None):
        d = src
        pred = d["pred_deployed"] if "pred_deployed" in d.files else d["pred"]
        msk = np.ones(len(pred), bool) if msk is None else msk
        p1 = d["p1"][msk].astype(np.float64).copy()
        if edit is not None:
            p1 = edit(p1, d["covered"][msk].astype(bool))
        np.savez(os.path.join(priv, "read", f"{root}_{model}.npz"), pred=pred[msk], p1=p1,
                 suspicion=-np.log(p1), covered=d["covered"][msk].astype(bool))

    def lower6(p1, cov):                      # six covered units with 0.5 < p1 < 0.9: p1 - 0.06, class unchanged
        ix = np.flatnonzero(cov & (p1 > 0.5) & (p1 < 0.9))[:6]
        p1[ix] -= 0.06
        return p1

    def below_cut(p1, cov):                   # five of the confident plots: just below the cut
        ix = np.flatnonzero(cov & (p1 >= CONFIDENT_CUT["v1"]))[:5]
        p1[ix] = CONFIDENT_CUT["v1"] - 1e-7
        return p1
    out = {}
    for tag in ("same", "v12_as_v1", "dp1_six", "zone_five"):
        swap = tag == "v12_as_v1"
        e23 = lower6 if tag == "dp1_six" else None
        e17 = lower6 if tag == "dp1_six" else below_cut if tag == "zone_five" else None
        put("gate23", "v1", u[("gate23", "v12" if swap else "v1")], sel, e23)
        put("gate23", "v12", u[("gate23", "v12")], sel)
        put("gate17", "v1", u[("gate17", "v12" if swap else "v1")], None, e17)
        put("gate17", "v12", u[("gate17", "v12")])
        ga, gb = gate_a(priv, known=known, aprime=False), gate_b(priv, known=known)
        out[tag] = {"A_v1_share": round(ga["models"]["v1"]["same_class_share"], 4), "A_pass": ga["pass"],
                    "A_checks": ga["checks"], "B_v1_share": round(gb["models"]["v1"]["same_class_share"], 4),
                    "B_v1_spearman": round(gb["models"]["v1"]["spearman_suspicion"], 4),
                    "B_v1_zone": gb["models"]["v1"]["confident_zone"],
                    "B_v1_dp1_above": gb["models"]["v1"]["dp1_above_tol"], "B_pass": gb["pass"],
                    "B_checks": gb["checks"]}
    fails = {t: sorted(k for k, v in {**out[t]["A_checks"], **out[t]["B_checks"]}.items() if not v) for t in out}
    out["failing_checks"] = fails
    out["as_expected"] = bool(
        out["same"]["A_pass"] and out["same"]["B_pass"] and out["same"]["A_v1_share"] == 1.0
        and fails["v12_as_v1"] == ["A1", "A2", "B1_class", "B1_confident_zone", "B1_dp1", "B1_spearman"]
        and fails["dp1_six"] == ["A2", "B1_dp1"] and fails["zone_five"] == ["B1_confident_zone"])
    return out


def e98_jsonable(obj):
    import exp78_error_rate_estimation as e78
    return e78.jsonable(obj)


def _smoke_read_blind(priv, root, model):
    """The blind read in the smoke, which run_read refuses before freezing (the guard is the point)."""
    try:
        run_read(priv, root, model)
        raise AssertionError("run_read read the blind root before freezing")
    except PermissionError:
        pass
    global prereg_status, check_draw
    keep, keep_draw = prereg_status, check_draw
    sha = sha_ids([u["pid"] for u in read_json(os.path.join(priv, "draw.json"))["units"]])
    prereg_status = lambda text=None: "frozen"            # noqa: E731  (smoke only)
    check_draw = lambda p, text=None: keep_draw(p, "**Draw sha256 (frozen):** `" + sha + "`", SMOKE_SIZES)  # noqa: E731
    try:
        return run_read(priv, root, model, log=lambda *_: None)
    finally:
        prereg_status, check_draw = keep, keep_draw


def _smoke_known(priv, tmp, w, differ=True):
    """Synthetic 'committed' units in exp98's and exp99's layouts for the smoke's gate units (differ: one of v1's
    gate-A known answers differs, so A1 fails)."""
    kn = {}
    n23, n17 = len(w["gate23"]), len(w["gate17"])
    tag = "" if differ else "_same"
    for m in MODELS:
        r = _load_read(priv, "gate23", m)
        pred = r["pred"].copy()
        if m == "v1" and differ:
            pred[0] = (pred[0] + 1) % 9                      # one known answer differs
        path = os.path.join(tmp, f"known23_{m}{tag}.npz")
        np.savez(path, names=np.array([u["key"] for u in w["gate23"]]), split=np.array(["val"] * n23),
                 label=r["pred"].copy(), covered=np.ones(n23, bool), pred_deployed=pred, p1=r["p1"])
        kn[("gate23", m)] = path
        r = _load_read(priv, "gate17", m)
        names = list(COUNTRIES)
        path = os.path.join(tmp, f"known17_{m}{tag}.npz")
        np.savez(path, country=np.array([names.index(u["country"]) for u in w["gate17"]]),
                 country_names=np.array(names), inside_awf=np.array([u["inside"] for u in w["gate17"]]),
                 pred=r["pred"], p1=r["p1"], covered=np.ones(n17, bool))
        kn[("gate17", m)] = path
    return kn


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--select", action="store_true")
    g.add_argument("--windows", action="store_true")
    g.add_argument("--read", nargs=2, metavar=("ROOT", "MODEL"))
    g.add_argument("--read-aprime", action="store_true")
    g.add_argument("--gate", choices=("A", "B"))
    g.add_argument("--check-draw", action="store_true")
    g.add_argument("--units", action="store_true")
    ap.add_argument("--private-dir", default=None)
    ap.add_argument("--timesync", default=os.path.join(ROOT, "data", "breadth", tsx.TIMESYNC["file"]))
    ap.add_argument("--request-geometry", default=None)
    ap.add_argument("--boundaries", default=None)
    ap.add_argument("--labels", default=None, help="exp89's AWF label windows (gate A's points)")
    ap.add_argument("--deploy-inv", default=None, help="exp99's 2017 deployment inventory (in_gate17_cell's first rule)")
    ap.add_argument("--deploy-run", default=None, help="exp99's 2017 deployment run directory")
    ap.add_argument("--gate-file", default=None)
    ap.add_argument("--identity-dir", default=None)
    ap.add_argument("--aprime-plan", default=None)
    ap.add_argument("--device-file", default=None, help="gpu_prep's record of the gate job's GPU (gate A)")
    ap.add_argument("--cgls", default=os.path.join(OUT, "exp94_eastafrica_points.csv"))
    ap.add_argument("--record", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        rec = smoke(a.out_dir)
        return 0 if rec["ok"] else 1
    if not a.private_dir:
        ap.error("--private-dir is required")
    if a.select:
        if not (a.request_geometry and a.boundaries):
            ap.error("--select needs --request-geometry and --boundaries")
        rec = run_select(a.timesync, a.request_geometry, a.boundaries, a.private_dir, SIZES, redo=a.redo)
    elif a.windows:
        if bool(a.deploy_inv) != bool(a.deploy_run):
            ap.error("--deploy-inv and --deploy-run go together")
        rec = run_windows(a.private_dir, a.labels, redo=a.redo,
                          deploy17=(a.deploy_inv, a.deploy_run) if a.deploy_inv else None)
    elif a.read:
        root, model = a.read
        if root not in ROOTS or model not in MODELS:
            ap.error(f"--read takes a root of {tuple(ROOTS)} and a model of {tuple(MODELS)}")
        rec = run_read(a.private_dir, root, model, redo=a.redo)
    elif a.read_aprime:
        rec = run_read_aprime(a.private_dir, a.aprime_plan)
    elif a.gate:
        rec = run_gate(a.gate, a.private_dir, a.gate_file, identity_dir=a.identity_dir, device_file=a.device_file)
        side = rec.get(a.gate) or {}
        print(json.dumps({"stage": a.gate, "checks": side.get("checks"), "pass": side.get("pass")}))
        return 0 if side.get("pass") else 4
    elif a.check_draw:
        sha = check_draw(a.private_dir)
        print(json.dumps({"draw": "the page's", "sha256": sha[:16]}))
        return 0
    elif a.units:
        rec = run_units(a.private_dir, a.gate_file, a.out_dir, a.cgls)
    if a.record:
        dump(rec, a.record)
    return 0 if rec.get("ok", True) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PermissionError as ex:
        print(str(ex))
        sys.exit(3)
    except Exception:  # noqa: BLE001  (a job log needs the traceback, and the exit code must fail the job)
        traceback.print_exc()
        sys.exit(1)
