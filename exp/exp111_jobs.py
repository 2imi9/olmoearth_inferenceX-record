"""exp111's rslearn-side helpers (exp/jobs/e111.sh; docs/plan/blind_transfer.md is the specification). They run in the
rslearn 0.1.17 venv (exp102's, $SCRATCH/e102/venv) or with any Python 3.11+: the standard library and numpy at import,
rslearn, rasterio and torch only inside the subcommands that need them (pack-check needs the standard library only, so
the pack jobs can run it with whatever Python is left). Nothing here grades and nothing here reads a TimeSync label: the
units maps hold window names, projections and pixels (private), the draw's country and label group for the per-stratum
input counts, never a label.

    no-runner          refuse if olmoearth_run or olmoearth-runner is importable or installed in this Python
    versions           the venv's package versions against the pins (rslearn 0.1.17, olmoearth-pretrain-minimal 0.1.2,
                       torch 2.14.1, lightning 2.6.6, jsonargparse 4.52.0: the versions exp103 recorded in exp102's venv);
                       --core-only checks the first two (the Mac's local smoke, whose torch is another build)
    device             the GPU this job runs on (torch.cuda.get_device_name, capability, torch's arch list) and the
                       commit the job runs (--commit), recorded; --expect-gate FILE refuses unless it is the device
                       e111_gate.json recorded
    check-configs      e111_config/dataset.json against exp98's (one key added), the SCL and model configs by sha256
    check-dataset      rslearn 0.1.17 parses a root: config (mode, periods, matches, time order), every window's
                       projection, bounds and time range against the units map; rslearn's own UTM rule, crop lattice
                       (get_window_crop_options) and RasterMerger trimming give each unit pixel the deployment's crop
    precheck           after prepare and the SCL sidecar: the windows `model predict` keeps (check_window) equal the
                       windows with twelve item groups; partial windows, and with --scl the windows with twelve input
                       groups whose SCL sidecar is not complete, are refused, or, within the failure allowance
                       (--max-drop-share, e111.sh's MAX_FAILED_SHARE), their input layers and sidecar move to a
                       quarantine directory of this run's own (--run-tag) so that they count as no input
    prepare-summary    rslearn's "Layer L: prepared=..., failed=..." (prepare) or "materialized=..., failed=..."
                       (materialize) line of a log, as counts; refused above a failed share; --plus RECORD chains an
                       earlier summary (its failures added, its denominator kept: the root's windows)
    check-groups       the smoke root after prepare: twelve groups, newest first, plausible harmonized reflectance
    inventory          windows with 12 / some / no item groups, rejected, failed at materialize, quarantined (only
                       while they have no input), bytes; for the blind root by country and label group (counts)
    pack-check         every JSON file under the given directories against RECORD_FORBIDDEN_KEYS: the pack jobs' check
                       and every record writer's (write_json, exp111_blind_transfer.dump, the grade's summary)
    scl-sidecar        the SCL sidecar root: the sentinel2 item groups of each window with all twelve groups completed,
                       copied in order as sentinel2_scl; a sidecar window whose root window has no complete input is
                       removed
    scl-check          the sidecar against the root's windows with input (after precheck's quarantine): the same
                       windows, twelve SCL groups each, SCL codes and coverage agreement on sampled windows
    check-output       a predicted layer: 10 float32 bands, the window's size, every window with input written, none
                       without, sampled windows a softmax (e103's check); the record names the Slurm job
    gate-records       gatecheck's check of the gate job's records: the device record (the gate's GPU and
                       predict_commit) and the four output checks, all passing and all written by one Slurm job
    load-check         build the LightningModule from model.yaml (CPU, offline) and load a checkpoint's state dict:
                       0 missing keys, no shape mismatch, a strict load through the module's own load hooks, a plain
                       weights_only load, and any key those hooks drop only the legacy temporal table, within
                       LEGACY_MAX (0.01) of the on-the-fly encoding; whether it equals it (LEGACY_EQUAL) is recorded
    aprime-config      for diagnostic A': model.yaml restricted (predict_config groups + names) to exp98's deployment
                       windows that hold the gate-A points; nothing else differs (checked after rslearn's substitution)
    item-identity      per gate unit, our sentinel2 item groups against those of the deployment window holding it
                       (skipped, with a note, when the deployment cannot be read whole: a partial purge)
    deploy-time-range  the distinct time ranges of a deployment run's windows, against the literal one (windows whose
                       metadata.json the purge removed are counted as unreadable, the rest checked)
    remove-layer       remove an output_e111_* layer from every window of a root (no other name)
    watchdog           a background guard for prepare: stops the job above a size cap or if the window count changes
    parse-model        rslearn 0.1.17 parses model.yaml (predict, --print_config: no class is instantiated, nothing
                       downloaded) and the parsed values are checked
    crop-lattice       rslearn's crop options for a window size against the 1,024-px deployment cell (no files)

Printed and recorded: counts, versions, sha256s. Window names (a0000, g0000, b0000), bounds, item and tile names stay
in the private directory; what the job shows of rslearn's logs is masked by e111.sh.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import signal
import sys
import time
from datetime import datetime, timedelta, timezone

try:                      # pack-check runs with the standard library alone (the pack jobs' fallback Python)
    import numpy as np
except ImportError:       # pragma: no cover
    np = None

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(EXP_DIR, "jobs", "e111_config")
E98_CONFIG = os.path.join(EXP_DIR, "jobs", "e98_config")
MODEL_YAML = os.path.join(EXP_DIR, "jobs", "e102_config", "model.yaml")
SHA = {"e98_dataset.json": "ad076afb56cf3ba2018a3148400d0684ddd449fce54cd208a12e2693249dd328",
       "scl_dataset.json": "e3d8de8b0d9dad5facef07098389075854b20da42381a9dd3926f09704387a01",
       "model.yaml": "b8af519ae70cc6ce9b6654cc254808ea6812dd12ad2b0360cd2aadc4099f5e9e"}
ADDED_KEY = ("layers", "sentinel2", "data_source", "query_config", "per_period_mosaic_reverse_time_order")
PINS = {"rslearn": "0.1.17", "olmoearth-pretrain-minimal": "0.1.2"}
# exp103's predict records (exp/out/exp103/jobs/records/e103_predict17_env.json): the rest of exp102's venv
PINS_GPU = {"torch": "2.14.1", "lightning": "2.6.6", "jsonargparse": "4.52.0"}
RUNNER_MODULES = ("olmoearth_run", "olmoearth_runner")
RUNNER_DISTS = ("olmoearth-run", "olmoearth-runner", "olmoearth_run", "olmoearth_runner")

INPUT = "sentinel2"
SCL = "sentinel2_scl"
N_GROUPS = 12
P = "_".join(f"p{i}" for i in range(10))           # rslearn's band-set directory of p0..p9
CELL_PX = 1024                                     # Ai2's GridPartitioner grid_size (olmoearth_run.yaml at f3c9b0c8)
CROP_PX = 16                                       # model.yaml predict_config patch_size
OVERLAP_PX = 4                                     # round(16 * 0.25)
TRIM_PX = 2                                        # RasterMerger(padding=2): overlap_pixels 4, half trimmed
STRIDE_PX = CROP_PX - OVERLAP_PX
RES = 10.0
PERIOD = timedelta(days=30)
TIME_RANGES = {2017: (datetime(2017, 1, 1, tzinfo=timezone.utc), datetime(2017, 12, 31, tzinfo=timezone.utc)),
               2023: (datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 12, 31, tzinfo=timezone.utc))}
LAYER_RE = re.compile(r"output_(v1|v12|e111_v1)")
REMOVABLE_PREFIX = "output_e111_"
MAX_FAILED_SHARE_CAP = 0.005                       # the page's cap on the owner's failure allowance (e111.sh)
# A record (anything under $OUT that the pack jobs copy) never holds a key naming an id or a position: exp98's
# FORBIDDEN_KEYS (exp98_awf_deployment.py) plus the plot id, the pixel and the cell. One set, one function, used by the
# pack jobs (pack-check), by every record writer here (write_json without private) and by exp111_blind_transfer.dump
# and the grade's summary; the local smoke runs pack-check over every record it writes.
RECORD_FORBIDDEN_KEYS = frozenset({"transform", "bounds", "bbox", "lon", "lat", "lonlat", "longitude", "latitude", "x",
                                   "y", "xy", "coordinates", "geometry", "origin", "centre", "center", "row", "col",
                                   "rows", "cols", "pid", "px", "py", "cell"})


# ----------------------------------------------------------------------------- small helpers
def forbidden_key_paths(obj, path=""):
    """The paths of every key in a JSON-like object that RECORD_FORBIDDEN_KEYS names (case-insensitive)."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in RECORD_FORBIDDEN_KEYS:
                out.append(f"{path}/{k}")
            out += forbidden_key_paths(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += forbidden_key_paths(v, f"{path}[{i}]")
    return out


def check_record_keys(obj, what="a record"):
    """Refuse a record that holds a key naming an id or a position (ValueError)."""
    bad = forbidden_key_paths(obj)
    if bad:
        raise ValueError(f"refusing to write {what}: keys that would carry an id or a position: {bad[:5]}")


def pack_check(dirs):
    """Every *.json file under `dirs` (recursively): [(file, key path)] of the forbidden keys, and the files read."""
    bad, files = [], 0
    for d in dirs:
        for dp, _, fs in os.walk(d):
            for fn in sorted(fs):
                if not fn.endswith(".json"):
                    continue
                p = os.path.join(dp, fn)
                with open(p) as f:
                    obj = json.load(f)
                files += 1
                bad += [(os.path.relpath(p, d), k) for k in forbidden_key_paths(obj)]
    return bad, files


def write_json(obj, path, private=False):
    """Atomic JSON. A record (private False) passes check_record_keys first; a private file (mode 600) is not checked."""
    if not private:
        check_record_keys(obj, os.path.basename(path))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".partial"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600 if private else 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_units(path):
    """The private units map the windows step wrote: {root, year, group, units: [{name, epsg, px, py, bounds, cell,
    lon?, lat?, country?, stratum?}]} -> (map, {name: unit})."""
    with open(path) as f:
        m = json.load(f)
    by = {u["name"]: u for u in m["units"]}
    if len(by) != len(m["units"]):
        raise SystemExit("the units map repeats a window name")
    return m, by


def list_windows(root):
    """[(group, name, path)] sorted, hidden entries left out."""
    wd = os.path.join(root, "windows")
    out = []
    for g in sorted(os.listdir(wd)):
        gd = os.path.join(wd, g)
        if g.startswith(".") or not os.path.isdir(gd):
            continue
        for n in sorted(os.listdir(gd)):
            p = os.path.join(gd, n)
            if not n.startswith(".") and os.path.isdir(p):
                out.append((g, n, p))
    return out


def groups_completed(path, layer=INPUT):
    return sum(os.path.exists(os.path.join(path, "layers", layer if k == 0 else f"{layer}.{k}", "completed"))
               for k in range(N_GROUPS))


def has_input(path):
    """rslearn 0.1.17's test for an input read with load_all_item_groups: item group 0 completed
    (rslearn/train/dataset.py:852-886)."""
    return os.path.exists(os.path.join(path, "layers", INPUT, "completed"))


def layer_done(path, layer):
    return os.path.exists(os.path.join(path, "layers", layer, "completed"))


def periods(t0, t1, duration=PERIOD, max_matches=N_GROUPS):
    """rslearn 0.1.17's period cut (rslearn/data_sources/utils.py:440-462): backwards from the end of the window's time
    range, whole periods only, at most max_matches."""
    out, end = [], t1
    while end - duration >= t0 and len(out) < max_matches:
        out.append((end - duration, end))
        end = end - duration
    return out


def owner_1d(p, b0, b1, crop=CROP_PX, overlap=OVERLAP_PX, trim=TRIM_PX):
    """Which crop's output RasterMerger keeps at pixel p of a window spanning [b0, b1), on one axis: the crop starts of
    get_window_crop_options (rslearn/train/all_crops_dataset.py:26-80), each crop trimmed by `trim` on the side that is
    not the window's start, outputs pasted in increasing order so that a later crop overwrites
    (rslearn/train/prediction_writer.py:128-185). Returns the crop's start, or None."""
    starts = [b0] + list(range(b0 + crop - overlap, b1 - crop, crop - overlap))
    if b1 - crop > b0:
        starts.append(b1 - crop)
    own = None
    for s in sorted(starts):
        lo = s if s == b0 else s + trim
        if lo <= p < s + crop:
            own = s
    return own


# ----------------------------------------------------------------------------- subcommands
def cmd_no_runner(a):
    import importlib.metadata as md
    import importlib.util
    found = [m for m in RUNNER_MODULES if importlib.util.find_spec(m) is not None]
    dists = []
    for d in RUNNER_DISTS:
        try:
            md.version(d)
            dists.append(d)
        except md.PackageNotFoundError:
            pass
    if found or dists:
        print(f"refused: this Python can import {found} / has {dists} installed; exp111 runs rslearn directly")
        return 1
    print("no olmoearth_run or olmoearth-runner in this Python")
    return 0


def cmd_versions(a):
    import importlib.metadata as md

    def ver(p):
        try:
            return md.version(p)
        except md.PackageNotFoundError:
            return None
    v = {p: ver(p) for p in ("rslearn", "olmoearth-pretrain-minimal", "torch", "lightning", "jsonargparse", "rasterio",
                             "numpy", "huggingface-hub", "planetary-computer", "pystac-client")}
    pins = dict(PINS) if a.core_only else {**PINS, **PINS_GPU}
    bad = {p: (v[p], want) for p, want in pins.items() if (v[p] or "").split("+")[0] != want}
    rec = {"python": sys.version.split()[0], "versions": v, "pins": pins, "ok": not bad,
           "core_only": bool(a.core_only), "job": os.environ.get("SLURM_JOB_ID")}
    if a.record:
        write_json(rec, a.record)
    print(json.dumps(rec))
    if bad:
        print(f"versions differ from the pins: {bad}")
        return 1
    return 0


def same_device(recorded, name):
    """The gate's device and this one are the same GPU model (torch.cuda.get_device_name)."""
    return bool(recorded) and bool(name) and str(recorded).strip() == str(name).strip()


def cmd_device(a):
    """The GPU this job runs on and the commit it runs (--commit, E111_SHA), recorded; with --expect-gate, refused
    unless the gate's record names the same GPU. The gate job's record is the gate's predict_commit (e111_gate.json):
    gatecheck and every phase-2 mode refuse when a guarded file changed since it. --name stands in for torch's answer
    (the local smoke has no GPU)."""
    if a.commit is not None and not re.fullmatch(r"[0-9a-f]{40}", a.commit):
        print("refused: --commit takes the full 40-character commit")
        return 2
    rec = {"job": os.environ.get("SLURM_JOB_ID"), "commit": a.commit}
    if a.name:
        rec.update(name=a.name, source="--name (smoke)")
    else:
        import torch
        if not torch.cuda.is_available():
            print("refused: torch sees no GPU")
            return 2
        c = torch.cuda.get_device_capability(0)
        arch = f"sm_{c[0]}{c[1]}"
        rec.update(name=torch.cuda.get_device_name(0), capability=list(c), arch=arch, torch=torch.__version__,
                   cuda=torch.version.cuda, arch_in_build=arch in torch.cuda.get_arch_list(), source="torch.cuda")
        if not rec["arch_in_build"]:
            print(f"refused: torch has no kernels for {arch}")
            return 2
    if a.record:
        write_json(rec, a.record)
    print(json.dumps(rec))
    if a.expect_gate:
        with open(a.expect_gate) as f:
            g = json.load(f)
        want = (g.get("device") or {}).get("name")
        if not same_device(want, rec["name"]):
            print(f"refused: the gate ran on {want!r}, this job is on {rec['name']!r}; predict on the gate's device")
            return 2
    return 0


def _get(d, path):
    for k in path:
        d = d[k]
    return d


def cmd_check_configs(a):
    import copy
    ours_path = os.path.join(a.config_dir, "dataset.json")
    with open(ours_path) as f:
        ours = json.load(f)
    with open(os.path.join(E98_CONFIG, "dataset.json")) as f:
        e98 = json.load(f)
    back = copy.deepcopy(ours)
    added = _get(back, ADDED_KEY[:-1]).pop(ADDED_KEY[-1], "absent")
    rec = {"dataset.json": {"sha256": sha256_file(ours_path), "added_key": ".".join(ADDED_KEY), "added_value": added,
                            "equals_e98_without_it": back == e98},
           "e98_dataset.json": sha256_file(os.path.join(E98_CONFIG, "dataset.json")),
           "scl_dataset.json": sha256_file(os.path.join(E98_CONFIG, "scl_dataset.json")),
           "model.yaml": sha256_file(a.model)}
    rec["ok"] = bool(added is True and back == e98 and rec["e98_dataset.json"] == SHA["e98_dataset.json"]
                     and rec["scl_dataset.json"] == SHA["scl_dataset.json"] and rec["model.yaml"] == SHA["model.yaml"])
    if a.reverse:
        write_json(back, a.reverse)
        rec["reverse_written"] = True
    if a.record:
        write_json(rec, a.record)
    print(json.dumps(rec))
    return 0 if rec["ok"] else 1


def cmd_check_dataset(a):
    """rslearn 0.1.17 itself reads the root (Dataset(): pydantic, extra="forbid"; load_windows: every metadata.json)."""
    import importlib.metadata as md
    from upath import UPath
    from rslearn.dataset.dataset import Dataset
    from rslearn.train.all_crops_dataset import get_window_crop_options
    from rslearn.utils.get_utm_ups_crs import get_utm_ups_crs
    m, by = load_units(a.units)
    year = int(m["year"])
    t0, t1 = TIME_RANGES[year]
    ds = Dataset(UPath(a.root))
    layer = ds.layers[INPUT]
    src = layer.data_source
    qc = src.query_config
    rec = {"root": m["root"], "rslearn": md.version("rslearn"), "layers": sorted(ds.layers),
           "sentinel2": {"class_path": src.class_path, "space_mode": str(qc.space_mode),
                         "period_duration_days": qc.period_duration.total_seconds() / 86400 if qc.period_duration else None,
                         "min_matches": qc.min_matches, "max_matches": qc.max_matches,
                         "reverse_time_order": qc.per_period_mosaic_reverse_time_order,
                         "harmonize": (src.init_args or {}).get("harmonize"),
                         "sort_by": (src.init_args or {}).get("sort_by")}}
    want = {"class_path": "rslearn.data_sources.planetary_computer.Sentinel2", "space_mode": "PER_PERIOD_MOSAIC",
            "period_duration_days": 30.0, "min_matches": N_GROUPS, "max_matches": N_GROUPS, "reverse_time_order": True,
            "harmonize": True, "sort_by": "eo:cloud_cover"}
    rec["sentinel2_as_pinned"] = {k: rec["sentinel2"][k] == v for k, v in want.items()}
    wins = ds.load_windows(workers=a.workers)
    # counters only; named so that no key is a forbidden record key (RECORD_FORBIDDEN_KEYS: "bounds" was one, and
    # pack refused the record)
    bad = {"not_in_units_map": 0, "projection": 0, "window_bounds_differ": 0, "time_range": 0, "request_range": 0,
           "rslearn_utm": 0, "crop_off_lattice": 0, "plot_crop_differs_from_cell": 0, "outside_cell": 0,
           "plot_outside_window": 0}
    sizes, epsg, margins, n_utm = {}, {}, [], 0
    for w in wins:
        u = by.get(w.name)
        if u is None:
            bad["not_in_units_map"] += 1
            continue
        p = w.projection
        bad["projection"] += not (p.crs.to_epsg() == int(u["epsg"]) and abs(p.x_resolution - RES) < 1e-9
                                  and abs(p.y_resolution + RES) < 1e-9)
        b = tuple(int(v) for v in w.bounds)
        bad["window_bounds_differ"] += b != tuple(u["bounds"])
        sizes[f"{b[2] - b[0]}x{b[3] - b[1]}"] = sizes.get(f"{b[2] - b[0]}x{b[3] - b[1]}", 0) + 1
        epsg[str(p.crs.to_epsg())] = epsg.get(str(p.crs.to_epsg()), 0) + 1
        bad["time_range"] += tuple(w.time_range) != (t0, t1)
        bad["request_range"] += tuple(src.get_request_time_range(w.time_range)) != (t0, t1)
        if u.get("lon") is not None:
            n_utm += 1
            bad["rslearn_utm"] += get_utm_ups_crs(float(u["lon"]), float(u["lat"])).to_epsg() != int(u["epsg"])
        X0, Y0 = u["cell"]
        bad["outside_cell"] += not (X0 <= b[0] and b[2] <= X0 + CELL_PX and Y0 <= b[1] and b[3] <= Y0 + CELL_PX)
        crops = get_window_crop_options((CROP_PX, CROP_PX), (OVERLAP_PX, OVERLAP_PX), b)
        bad["crop_off_lattice"] += any((c[0] - X0) % STRIDE_PX or (c[1] - Y0) % STRIDE_PX for c in crops)
        px, py = int(u["px"]), int(u["py"])
        if not (b[0] <= px < b[2] and b[1] <= py < b[3]):
            bad["plot_outside_window"] += 1
            continue
        mine = (owner_1d(px, b[0], b[2]), owner_1d(py, b[1], b[3]))
        cell = (owner_1d(px, X0, X0 + CELL_PX), owner_1d(py, Y0, Y0 + CELL_PX))
        bad["plot_crop_differs_from_cell"] += mine != cell
        # the distance from the plot pixel to the nearest window edge the deployment's cell does not share
        d = [px - b[0] if b[0] != X0 else 10 ** 6, b[2] - 1 - px if b[2] != X0 + CELL_PX else 10 ** 6,
             py - b[1] if b[1] != Y0 else 10 ** 6, b[3] - 1 - py if b[3] != Y0 + CELL_PX else 10 ** 6]
        margins.append(min(d))
    pr = periods(t0, t1)
    rec.update(windows=len(wins), expected=len(m["units"]), sizes=sizes, epsg=epsg, bad=bad,
               units_with_lonlat_checked_by_rslearn_utm=n_utm,
               min_margin_to_unshared_edge_px=int(min(margins)) if margins else None,
               periods={"n": len(pr), "newest": [pr[0][0].date().isoformat(), pr[0][1].date().isoformat()] if pr else None,
                        "oldest": [pr[-1][0].date().isoformat(), pr[-1][1].date().isoformat()] if pr else None})
    rec["ok"] = bool(len(wins) == len(m["units"]) and not any(bad.values()) and all(rec["sentinel2_as_pinned"].values())
                     and len(pr) == N_GROUPS and (not margins or min(margins) >= TRIM_PX + 1))
    if a.record:
        write_json(rec, a.record)
    print(json.dumps(rec))
    return 0 if rec["ok"] else 1


def model_inputs(config_path):
    """model.yaml's non-target data inputs as rslearn DataInputs (exp103's precheck)."""
    import yaml
    from rslearn.train.dataset import DataInput
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    out = {}
    for k, v in cfg["data"]["init_args"]["inputs"].items():
        if v.get("is_target"):
            continue
        out[k] = DataInput(data_type=v["data_type"], layers=list(v["layers"]), bands=v.get("bands"),
                           required=v.get("required", True), passthrough=v.get("passthrough", False),
                           load_all_layers=v.get("load_all_layers", False),
                           load_all_item_groups=v.get("load_all_item_groups", False))
    return out


def full_input(path):
    """All twelve sentinel2 item groups completed, group 0 among them: the windows the sidecar, the SCL check and the
    prediction take."""
    return groups_completed(path) == N_GROUPS and has_input(path)


def quarantine_dir(base, root, tag):
    """A directory of this run's own under <base>/<root>/: <tag>, or <tag>.2, <tag>.3, ... when an earlier run of the
    same tag left one (a window is never moved into an existing destination, so nothing nests)."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(tag)):
        raise SystemExit("--run-tag takes letters, digits, '_', '.' and '-' only")
    d, k = os.path.join(base, root, tag), 1
    while os.path.lexists(d):
        k += 1
        d = os.path.join(base, root, f"{tag}.{k}")
    return d


def quarantined_names(base, root):
    """{window name: [run tags]} of every run's quarantine of `root` (<base>/<root>/<tag>/<window>/)."""
    out = {}
    rd = os.path.join(base, root) if base else None
    if not rd or not os.path.isdir(rd):
        return out
    for tag in sorted(os.listdir(rd)):
        td = os.path.join(rd, tag)
        if os.path.isdir(td):
            for n in sorted(os.listdir(td)):
                if os.path.isdir(os.path.join(td, n)):
                    out.setdefault(n, []).append(tag)
    return out


def cmd_precheck(a):
    """Partial windows (1 to 11 item groups, or groups without group 0) would be predicted with fewer timesteps, and a
    window with twelve input groups whose SCL sidecar is not complete (--scl; the SCL materialize failed there) would be
    read without its conditions. Both are refused, unless their share of the root's windows is within --max-drop-share
    (e111.sh's MAX_FAILED_SHARE, the owner's failure allowance, fixed at the phase-1 commit): then each one's input
    layers, and its sidecar window, move to <quarantine>/<root>/<run tag>/<window>/ (a directory of this run's own) and
    it counts as no input (the inventory counts it by cell). Above the allowance nothing is moved. A later
    `--from prepare` fetches a quarantined window again (its completion markers moved with it)."""
    import importlib.metadata as md
    from upath import UPath
    from rslearn.dataset.dataset import Dataset
    from rslearn.train.dataset import check_window
    m, by = load_units(a.units)
    fs = {"windows": 0, "with_12_groups": 0, "partial": 0, "none": 0, "scl_incomplete": 0}
    drop = []
    for g, n, p in list_windows(a.root):
        k = groups_completed(p)
        fs["windows"] += 1
        fs["with_12_groups"] += k == N_GROUPS
        fs["none"] += k == 0
        if 0 < k < N_GROUPS or (k == N_GROUPS) != has_input(p):
            fs["partial"] += 1
            drop.append((g, n, p))
        elif a.scl and k == N_GROUPS and groups_completed(os.path.join(a.scl, "windows", g, n), SCL) != N_GROUPS:
            fs["scl_incomplete"] += 1
            drop.append((g, n, p))
    quarantined = 0
    share = len(drop) / max(1, fs["windows"])
    fs["dropped_share"] = share
    fs["max_drop_share"] = a.max_drop_share
    if drop and a.max_drop_share > 0 and share <= a.max_drop_share:
        qdir = quarantine_dir(a.quarantine, m["root"], a.run_tag)
        for g, n, p in drop:
            dest = os.path.join(qdir, n)
            os.makedirs(os.path.join(dest, "layers"))
            for d in glob.glob(os.path.join(glob.escape(p), "layers", INPUT + "*")):
                shutil.move(d, os.path.join(dest, "layers", os.path.basename(d)))
            sdir = os.path.join(a.scl, "windows", g, n) if a.scl else None
            if sdir and os.path.isdir(sdir):
                shutil.move(sdir, os.path.join(dest, "scl_sidecar"))
            quarantined += 1
        fs["quarantined"] = quarantined
        fs["quarantine_run"] = os.path.basename(qdir)
    ds = Dataset(UPath(a.root))
    wins = ds.load_windows(workers=a.workers)
    inputs = model_inputs(a.config)
    kept = {w.name for w in wins if check_window(inputs, w)[0] is not None}
    full = {n for g, n, p in list_windows(a.root) if groups_completed(p) == N_GROUPS}
    rec = {"root": m["root"], "rslearn": md.version("rslearn"), "filesystem": fs, "scl_checked": bool(a.scl),
           "windows_rslearn_predicts": len(kept), "same_as_twelve_groups": kept == full,
           "windows_match_units_map": {w.name for w in wins} == set(by)}
    if a.scl:
        rec["scl_complete_where_input"] = all(
            groups_completed(os.path.join(a.scl, "windows", g, n), SCL) == N_GROUPS
            for g, n, p in list_windows(a.root) if full_input(p))
    rec["ok"] = bool(rec["same_as_twelve_groups"] and rec["windows_match_units_map"]
                     and (not drop or quarantined == len(drop)) and rec.get("scl_complete_where_input", True))
    if a.record:
        write_json(rec, a.record)
    print(json.dumps(rec))
    if not rec["ok"]:
        print("precheck failed: windows with 1 to 11 item groups, or with an incomplete SCL sidecar, would be predicted "
              f"with fewer timesteps or read without conditions (their share {share:.4f} is above the failure allowance "
              f"{a.max_drop_share}: MAX_FAILED_SHARE in e111.sh, fixed at the phase-1 commit), or rslearn keeps other "
              "windows")
        return 1
    return 0


SUMMARY_RE = re.compile(r"Layer (\w+): prepared=(\d+), skipped=(\d+), rejected=(\d+), failed=(\d+)")
MAT_RE = re.compile(r"Layer (\w+): materialized=(\d+), failed=(\d+)")
TOTAL_RE = re.compile(r"Total windows requested: (\d+)")


def cmd_prepare_summary(a):
    """rslearn 0.1.17's summary lines (rslearn/main.py:629-650 for prepare, 1040-1056 for materialize) for --layer:
    the last one in the log, and the share of the windows that failed against --max-failed-share. --plus RECORD chains
    an earlier summary record of the same root: its failures (failed_total) are added and its denominator kept. Pass 3
    chains prepare -> materialize -> the SCL sidecar's materialize, so that the failure allowance bounds the sum of the
    three, over the root's windows (the sidecar holds only the windows with input, so its own count of requested
    windows is not the denominator)."""
    with open(a.log, errors="replace") as f:
        text = f.read()
    rows = [dict(zip(("layer", "prepared", "skipped", "rejected", "failed"), (m.group(1), *map(int, m.groups()[1:]))))
            for m in SUMMARY_RE.finditer(text)]
    rows += [dict(zip(("layer", "materialized", "failed"), (m.group(1), *map(int, m.groups()[1:]))))
             for m in MAT_RE.finditer(text)]
    tot = [int(x) for x in TOTAL_RE.findall(text)]
    mine = [r for r in rows if r["layer"] == a.layer]
    plus, denom = 0, (tot[-1] if tot else None)
    if a.plus:
        with open(a.plus) as f:
            earlier = json.load(f)
        plus = earlier.get("failed_total")
        if plus is None:          # a record written before failed_total existed
            plus = int(((earlier.get("summary") or {}).get("failed")) or 0) + \
                int(earlier.get("failed_added_from_earlier_record") or 0)
        plus = int(plus)
        denom = earlier.get("denominator") or earlier.get("total_windows_requested")
    rec = {"layer": a.layer, "summary": mine[-1] if mine else None, "total_windows_requested": tot[-1] if tot else None,
           "failed_added_from_earlier_record": plus, "denominator": denom, "max_failed_share": a.max_failed_share}
    rec["failed_total"] = (mine[-1]["failed"] + plus) if mine else None
    fail_share = (rec["failed_total"] / max(1, int(denom))) if (mine and denom) else None
    rec["failed_share"] = fail_share
    ok = bool(mine) and (fail_share is not None) and (fail_share <= a.max_failed_share)
    rec["ok"] = bool(ok)
    print(json.dumps(rec))
    if a.record:
        write_json(rec, a.record)
    return 0 if ok else 1


def _group_starts(entry):
    """The start of each item group's time range, from items.json: group_time_ranges when rslearn wrote them, else the
    first item's own time range."""
    gtr = entry.get("group_time_ranges")
    if gtr:
        return [datetime.fromisoformat(t[0]) if t else None for t in gtr]
    out = []
    for grp in entry.get("serialized_item_groups") or []:
        tr = ((grp[0] if grp else {}).get("geometry") or {}).get("time_range") if grp else None
        out.append(datetime.fromisoformat(tr[0]) if tr else None)
    return out


def cmd_check_groups(a):
    """After prepare and materialize on the smoke root: every window has twelve item groups, newest first (the
    deployment's order), and its first group's reflectance is uint16 in twelve bands with a plausible harmonized
    median over covered pixels (B02 between 50 and 10,000)."""
    import rasterio
    res = {"windows": 0, "twelve_groups": 0, "newest_first": 0, "reflectance_plausible": 0, "medians_b02": []}
    for g, n, p in list_windows(a.root):
        res["windows"] += 1
        try:
            with open(os.path.join(p, "items.json")) as f:
                ent = [e for e in json.load(f) if e.get("layer_name") == INPUT]
        except (OSError, ValueError):
            ent = []
        starts = _group_starts(ent[0]) if ent else []
        res["twelve_groups"] += len(starts) == N_GROUPS and groups_completed(p) == N_GROUPS
        res["newest_first"] += len(starts) == N_GROUPS and None not in starts and \
            all(starts[i] > starts[i + 1] for i in range(len(starts) - 1))
        tifs = glob.glob(os.path.join(glob.escape(p), "layers", INPUT, "*", "geotiff.tif"))
        if len(tifs) == 1:
            with rasterio.open(tifs[0]) as src:
                arr = src.read()
                ok = src.count == 12 and src.dtypes[0] == "uint16"
            cov = (arr != 0).any(axis=0)
            med = float(np.median(arr[1][cov])) if cov.any() else None     # band order B01, B02, ... (dataset.json)
            res["medians_b02"].append(med)
            res["reflectance_plausible"] += bool(ok and med is not None and 50 <= med <= 10000)
    w = res["windows"]
    res["ok"] = bool(w and res["twelve_groups"] == w and res["newest_first"] == w and res["reflectance_plausible"] == w)
    if a.record:
        write_json(res, a.record)
    print(json.dumps(res))
    return 0 if res["ok"] else 1


def cmd_inventory(a):
    """Counts per root and, for the blind root, per country x label group. 'quarantined': windows a precheck (of this
    or an earlier prepare run) moved aside within the failure allowance and that have no input now (they also count
    under 'none'); a window quarantined earlier that a later `--from prepare` fetched again counts under
    'quarantined_earlier_now_with_input' instead. 'materialize_failed': prepared with twelve groups, none completed,
    not quarantined (a materialize failure within the allowance)."""
    m, by = load_units(a.units)
    qnames = quarantined_names(a.quarantine, m["root"]) if a.quarantine else {}
    strata = {}
    tot = {"windows": 0, "with_12_groups": 0, "partial": 0, "none": 0, "prepared_rejected": 0, "unprepared": 0,
           "materialize_failed": 0, "quarantined": 0, "quarantined_earlier_now_with_input": 0}
    for g, n, p in list_windows(a.root):
        u = by.get(n, {})
        key = f"{u.get('country', 'all')}:{u.get('stratum', 'all')}"
        s = strata.setdefault(key, {k: 0 for k in tot})
        k = groups_completed(p)
        try:
            with open(os.path.join(p, "items.json")) as f:
                items = json.load(f)
            got = {e.get("layer_name"): len(e.get("serialized_item_groups") or []) for e in items}
            prepared = got.get(INPUT)
        except (OSError, ValueError):
            prepared = None
        q = n in qnames
        for d in (s, tot):
            d["windows"] += 1
            d["with_12_groups"] += k == N_GROUPS
            d["partial"] += 0 < k < N_GROUPS
            d["none"] += k == 0
            d["prepared_rejected"] += prepared is not None and prepared < N_GROUPS
            d["unprepared"] += prepared is None
            d["materialize_failed"] += k == 0 and prepared == N_GROUPS and not q
            d["quarantined"] += q and k == 0
            d["quarantined_earlier_now_with_input"] += q and k > 0
    nbytes = 0
    for dp, _, fs in os.walk(a.root):
        for fn in fs:
            try:
                nbytes += os.lstat(os.path.join(dp, fn)).st_size
            except OSError:
                pass
    rec = {"root": m["root"], "total": tot, "by_stratum": strata if len(strata) > 1 else None, "bytes": nbytes,
           "quarantine_runs": len({t for ts in qnames.values() for t in ts}),
           "counted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if a.record:
        write_json(rec, a.record)
    print(json.dumps({"root": m["root"], "total": tot, "gb": round(nbytes / 1e9, 2)}))
    return 0


def cmd_scl_sidecar(a):
    """exp99's sidecar (e99_prepare.sh), unchanged in substance: the root's sentinel2 item groups, copied in order into
    items.json as sentinel2_scl, beside a copy of metadata.json; `rslearn dataset materialize` then reads the SCL band
    of the same scenes, nearest neighbour, unharmonized (e98_config/CHANGES.md). Only windows with all twelve sentinel2
    groups completed get a sidecar window: a window that failed prepare or materialize (within the failure allowance)
    has no input to condition, and a sidecar for it would hold SCL groups its root window lacks. A sidecar window
    whose root window has no complete input now (an earlier run's) is removed."""
    with open(a.scl_config) as f:
        cfg = json.load(f)
    os.makedirs(a.scl, exist_ok=True)
    with open(os.path.join(a.scl, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    made = kept = skipped = 0
    want_names = set()
    for g, n, p in list_windows(a.root):
        items = os.path.join(p, "items.json")
        if not (full_input(p) and os.path.exists(items)):
            skipped += 1
            continue
        with open(items) as f:
            s2 = [ld for ld in json.load(f) if ld["layer_name"] == INPUT]
        if not s2 or len(s2[0].get("serialized_item_groups") or []) != N_GROUPS:
            skipped += 1
            continue
        want_names.add((g, n))
        want = [{"layer_name": SCL, "serialized_item_groups": s2[0]["serialized_item_groups"], "materialized": False}]
        sdir = os.path.join(a.scl, "windows", g, n)
        sitems = os.path.join(sdir, "items.json")
        if os.path.exists(sitems):
            with open(sitems) as f:
                old = json.load(f)
            if [o["serialized_item_groups"] for o in old] == [w["serialized_item_groups"] for w in want]:
                kept += 1
                continue
        if os.path.lexists(sdir):
            shutil.rmtree(sdir)
        os.makedirs(sdir)
        shutil.copy2(os.path.join(p, "metadata.json"), os.path.join(sdir, "metadata.json"))
        with open(sitems, "w") as f:
            json.dump(want, f)
        made += 1
    stale = 0
    if os.path.isdir(os.path.join(a.scl, "windows")):
        for g, n, p in list_windows(a.scl):
            if (g, n) not in want_names:
                shutil.rmtree(p)
                stale += 1
    print(json.dumps({"sidecar_windows_written": made, "unchanged": kept, "root_windows_without_full_input": skipped,
                      "stale_sidecar_windows_removed": stale}))
    return 0


def _groups(wdir, name):
    ldir = os.path.join(wdir, "layers")
    out = {}
    if not os.path.isdir(ldir):
        return out
    for d in os.listdir(ldir):
        mm = re.fullmatch(re.escape(name) + r"(?:\.(\d+))?", d)
        if mm and os.path.exists(os.path.join(ldir, d, "completed")):
            out[int(mm.group(1) or 0)] = os.path.join(ldir, d)
    return out


def cmd_scl_check(a):
    """After precheck's quarantine: the sidecar holds exactly the root's windows with input, each with the root
    window's twelve groups as completed SCL groups; on eight sampled windows, every group's SCL raster sits on its
    reflectance raster's grid, its codes are 0 to 11 and its coverage agrees with the reflectance's (98% or more of
    the pixels). A difference is counted and refused, never a traceback."""
    import rasterio
    check = {"windows": 0, "root_windows_with_input": 0, "missing_sidecar": 0, "sidecar_without_input": 0,
             "mismatched_group_counts": 0, "sampled": [], "codes": {}}
    codes = np.zeros(256, np.int64)
    root_in = {(g, n): p for g, n, p in list_windows(a.root) if full_input(p)}
    side = {(g, n): p for g, n, p in list_windows(a.scl)} if os.path.isdir(os.path.join(a.scl, "windows")) else {}
    check["root_windows_with_input"] = len(root_in)
    check["windows"] = len(side)
    check["missing_sidecar"] = len(set(root_in) - set(side))
    check["sidecar_without_input"] = len(set(side) - set(root_in))
    wins = []
    for key in sorted(set(root_in) & set(side)):
        if sorted(_groups(root_in[key], INPUT)) != sorted(_groups(side[key], SCL)):
            check["mismatched_group_counts"] += 1
        else:
            wins.append(key)
    picks = sorted(set(np.linspace(0, len(wins) - 1, min(8, len(wins))).round().astype(int).tolist())) if wins else []
    for i in picks:
        g_run, g_scl = _groups(root_in[wins[i]], INPUT), _groups(side[wins[i]], SCL)
        for gi in sorted(g_scl):
            with rasterio.open(os.path.join(g_scl[gi], "SCL", "geotiff.tif")) as s:
                arr = s.read(1)
            refl = glob.glob(os.path.join(glob.escape(g_run[gi]), "*", "geotiff.tif"))
            if len(refl) != 1:
                check["sampled"].append({"window_index": int(i), "group": gi, "reflectance_rasters": len(refl),
                                         "coverage_agreement": None})
                continue
            with rasterio.open(refl[0]) as s:
                b = s.read()
            if b.shape[1:] != arr.shape:
                check["sampled"].append({"window_index": int(i), "group": gi, "off_grid": True,
                                         "coverage_agreement": None})
                continue
            codes += np.bincount(arr.ravel(), minlength=256)
            check["sampled"].append({"window_index": int(i), "group": gi,
                                     "coverage_agreement": float(((arr != 0) == (b != 0).any(axis=0)).mean())})
    check["codes"] = {str(k): int(v) for k, v in enumerate(codes) if v}
    agrees = [s["coverage_agreement"] for s in check["sampled"]]
    check["ok"] = bool(check["windows"] > 0 and check["missing_sidecar"] == 0 and check["sidecar_without_input"] == 0
                       and check["mismatched_group_counts"] == 0
                       and set(int(k) for k in check["codes"]) <= set(range(12)) and agrees
                       and None not in agrees and min(agrees) >= 0.98)
    if a.record:
        write_json(check, a.record)
    print(json.dumps({k: v for k, v in check.items() if k != "sampled"}))
    return 0 if check["ok"] else 1


def cmd_check_output(a):
    """e98_predict.sh's and e103's check of an output layer (counts only)."""
    import rasterio
    if not LAYER_RE.fullmatch(a.layer):
        raise SystemExit(f"refused: {a.layer} is not an exp111 output layer")
    names = None
    if a.names:
        with open(a.names) as f:
            names = set(json.load(f))
    res = {"layer": a.layer, "job": os.environ.get("SLURM_JOB_ID"), "windows": 0, "with_12_groups": 0,
           "with_output": 0, "missing_output": 0, "output_without_input": 0, "bad_bandset": 0, "bad_header": 0,
           "sampled": []}
    outs = []
    for g, n, p in list_windows(a.root):
        if names is not None and n not in names:
            continue
        full, hout = groups_completed(p) == N_GROUPS, layer_done(p, a.layer)
        res["windows"] += 1
        res["with_12_groups"] += full
        res["with_output"] += hout
        res["missing_output"] += full and not hout
        res["output_without_input"] += hout and not has_input(p)
        if not hout:
            continue
        ldir = os.path.join(p, "layers", a.layer)
        sets = sorted(d for d in os.listdir(ldir) if os.path.isfile(os.path.join(ldir, d, "geotiff.tif")))
        if sets != [P]:
            res["bad_bandset"] += 1
            continue
        tif = os.path.join(ldir, P, "geotiff.tif")
        with open(os.path.join(p, "metadata.json")) as f:
            b = json.load(f)["bounds"]
        with rasterio.open(tif) as src:
            ok = src.count == 10 and all(d == "float32" for d in src.dtypes) and \
                (src.width, src.height) == (b[2] - b[0], b[3] - b[1])
        res["bad_header"] += not ok
        outs.append(tif)
    picks = sorted(set(np.linspace(0, len(outs) - 1, min(a.sample, len(outs))).round().astype(int).tolist())) \
        if outs else []
    ok = True
    for i in picks:
        with rasterio.open(outs[i]) as src:
            arr = src.read()
        s = arr.sum(axis=0, dtype=np.float64)
        cov = s > 0
        e = {"window_index": int(i), "finite": bool(np.isfinite(arr).all()), "min": float(arr.min()),
             "max": float(arr.max()), "uncovered_share": float(1 - cov.mean()),
             "max_abs_sum_minus_1": float(np.abs(s[cov] - 1).max()) if cov.any() else None,
             "argmax_is_9_share": float((arr.argmax(axis=0)[cov] == 9).mean()) if cov.any() else None}
        e["ok"] = bool(e["finite"] and e["min"] >= -1e-6 and e["max"] <= 1 + 1e-6 and e["max_abs_sum_minus_1"] is not None
                       and e["max_abs_sum_minus_1"] <= 1e-3)
        ok &= e["ok"]
        res["sampled"].append(e)
    res["ok"] = bool(ok and picks and res["missing_output"] == 0 and res["output_without_input"] == 0
                     and res["bad_bandset"] == 0 and res["bad_header"] == 0
                     and (names is not None or res["with_output"] == res["with_12_groups"]))
    if a.record:
        write_json(res, a.record)
    print(json.dumps({k: v for k, v in res.items() if k != "sampled"}))
    return 0 if res["ok"] else 1


GATE_OUTPUTS = tuple((r, m) for r in ("gate23", "gate17") for m in ("v1", "v12"))


def gate_records_problems(rec_dir):
    """What gatecheck refuses in the gate job's records (empty when they are one gate's): the device record (the gate's
    GPU and predict_commit, written by the gate job only after its four output checks passed) and the four output
    checks, each passing, all written by the same Slurm job. A gate job removes all five when it starts, so a gate that
    died part way leaves no device record, and an older job's output checks cannot stand beside a newer device record."""
    probs = []
    dev_path = os.path.join(rec_dir, "e111_device_gate.json")
    if not os.path.isfile(dev_path):
        return ["no record of the gate's GPU (e111_device_gate.json): the gate job did not finish its four predictions"]
    with open(dev_path) as f:
        dev = json.load(f)
    if not re.fullmatch(r"[0-9a-f]{40}", str(dev.get("commit") or "")):
        probs.append("the device record holds no 40-character commit (predict_commit)")
    if not dev.get("name"):
        probs.append("the device record names no GPU")
    job = dev.get("job")
    if not job:
        probs.append("the device record names no Slurm job")
    for r, m in GATE_OUTPUTS:
        p = os.path.join(rec_dir, f"e111_{r}_{m}_output.json")
        if not os.path.isfile(p):
            probs.append(f"no output check for {r} {m}")
            continue
        with open(p) as f:
            o = json.load(f)
        if o.get("ok") is not True:
            probs.append(f"the output check for {r} {m} did not pass")
        if job and o.get("job") != job:
            probs.append(f"the output check for {r} {m} is another job's than the device record's")
    return probs


def cmd_gate_records(a):
    probs = gate_records_problems(a.rec)
    if probs:
        print("refused: " + "; ".join(probs) + " (resubmit with --from gate)")
        return 2
    with open(os.path.join(a.rec, "e111_device_gate.json")) as f:
        dev = json.load(f)
    print(json.dumps({"gate_job_records": "one job's, all passing", "predict_commit": dev["commit"][:12]}))
    return 0


def build_cli(config, extra=()):
    """rslearn's own model handler, without running (rslearn/lightning_cli.py:535-567): the classes are instantiated
    from the config (the encoder's base weights come from HF_HOME; HF_HUB_OFFLINE=1 keeps it offline)."""
    from lightning.pytorch import LightningDataModule, LightningModule
    from rslearn.arg_parser import RslearnArgumentParser
    from rslearn.lightning_cli import RslearnLightningCLI
    from rslearn.utils.jsonargparse import init_jsonargparse
    init_jsonargparse()
    return RslearnLightningCLI(model_class=LightningModule, datamodule_class=LightningDataModule,
                               args=["--config", config, *extra], subclass_mode_model=True, subclass_mode_data=True,
                               save_config_kwargs={"overwrite": True}, parser_class=RslearnArgumentParser, run=False)


# The one key a module's own load hook may drop: olmoearth_pretrain's legacy temporal position table. Checkpoints saved
# before the encoder computed it on the fly (get_1d_sincos_pos_encoding) still carry it, and the encoder's
# load_state_dict pre-hook (_drop_pos_embed_hook) deletes it, so Lightning's strict restore passes. v1's fine-tuned copy
# differs from the on-the-fly encoding by up to 0.0037 (env job 1361497; the v1-Base copy by 6e-8), so under rslearn
# 0.1.17 v1 runs with the formula's table, not its own: the owner accepted that on 15 October 2026 and the gates judge
# it. LEGACY_MAX is only a guard against a table that is not this one; LEGACY_EQUAL says whether it is the formula's.
LEGACY_DROPPABLE = re.compile(r"(^|\.)composite_encodings\.pos_embed$")
LEGACY_EQUAL = 1e-5
LEGACY_MAX = 0.01
# exp89's rule for loading a checkpoint: weights_only=True; other pickled globals become inert stubs; a global from
# these modules is refused (exp89's list with the call-building modules added: operator, functools, types, _codecs)
REFUSED_PREFIXES = ("os.", "posix.", "nt.", "subprocess.", "sys.", "builtins.", "__builtin__.", "shutil.", "socket.",
                    "importlib.", "runpy.", "pickle.", "marshal.", "code.", "ctypes.", "pty.", "multiprocessing.",
                    "webbrowser.", "operator.", "_operator.", "functools.", "types.", "_codecs.")


def _stub_class(fullname):
    module, _, name = fullname.rpartition(".")

    def init(self, *args, **kwargs):
        self.__dict__["_stub_args"] = (args, kwargs)

    def setstate(self, state):
        self.__dict__["_stub_state"] = state
    return type(name, (), {"__module__": module, "__qualname__": name, "__init__": init, "__setstate__": setstate,
                           "__doc__": f"inert stand-in for {fullname}, never imported"})


def load_checkpoint_safely(path):
    """(the checkpoint's state_dict as stored, how), exp89's rule: torch.load with weights_only=True first; if the
    checkpoint pickles other globals (Lightning's, jsonargparse's), each becomes an inert stub class under its own name
    and the load is retried with weights_only=True, so nothing from those packages is imported or run. A global from a
    module REFUSED_PREFIXES lists is refused. The state_dict is returned unfiltered (non-tensor entries and its
    _metadata kept), as Lightning hands it to load_state_dict."""
    import torch
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        how = "weights_only"
    except Exception as first:                                     # an unsupported global
        unsafe = list(torch.serialization.get_unsafe_globals_in_checkpoint(path))
        bad = [g for g in unsafe if g.startswith(REFUSED_PREFIXES)]
        if bad or not unsafe:
            raise RuntimeError(f"refused: the checkpoint pickles {bad or 'no listed global'} ({type(first).__name__})") \
                from None
        with torch.serialization.safe_globals([(_stub_class(g), g) for g in unsafe]):
            ck = torch.load(path, map_location="cpu", weights_only=True)
        how = f"weights_only with {len(unsafe)} inert stub classes: {sorted(unsafe)}"
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    if not isinstance(sd, dict):
        raise RuntimeError("refused: the checkpoint holds no state_dict")
    return sd, how


def legacy_table_diffs(module, sd, keys, rows=False):
    """{key: max |checkpoint value - get_1d_sincos_pos_encoding(arange(T), D)|}, or with rows the T per-row maxima, for
    each key that is the legacy table:
    named as LEGACY_DROPPABLE says, its owner in `module` a CompositeEncodings (the class whose hook drops it), a real
    floating (T, D) tensor (leading unit dimensions squeezed) with T > 0 and D the owner's embedding_dim_per_embedding_type,
    the width the encoder computes the table at. None for any other key."""
    import torch
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.encodings import get_1d_sincos_pos_encoding
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import CompositeEncodings
    out = {}
    for k in keys:
        out[k] = None
        v = sd[k]
        if not (LEGACY_DROPPABLE.search(k) and isinstance(v, torch.Tensor) and torch.is_floating_point(v)):
            continue
        try:
            owner = module.get_submodule(k.rsplit(".", 1)[0])
        except AttributeError:
            continue
        if not isinstance(owner, CompositeEncodings):
            continue
        t = v.detach().to(torch.float32)
        while t.ndim > 2 and t.shape[0] == 1:
            t = t[0]
        d = int(getattr(owner, "embedding_dim_per_embedding_type", -1))
        if t.ndim != 2 or t.shape[0] == 0 or t.shape[1] != d or d % 2:
            continue
        ref = get_1d_sincos_pos_encoding(torch.arange(t.shape[0]), d).to(torch.float32)
        diff = (t - ref).abs()
        out[k] = [round(float(x), 6) for x in diff.max(dim=1).values] if rows else float(diff.max())
    return out


def load_check_result(module, sd):
    """The load check's verdict: a strict load_state_dict (which runs the module's own load hooks, as Lightning's
    restore does) must pass with no missing key, no shape mismatch and no non-tensor entry, and every key the module
    does not hold (so one its hooks dropped or ignored) must be the legacy table within LEGACY_MAX of the on-the-fly
    encoding; whether it equals it (LEGACY_EQUAL) is recorded."""
    import torch
    missing, unexpected, shapes = compare_state_dicts(module, sd)
    non_tensor = sorted(k for k, v in sd.items() if not isinstance(v, torch.Tensor))
    loaded, error = False, None
    try:
        module.load_state_dict(sd, strict=True)              # torch copies sd (and its _metadata) before the hooks run
        loaded = True
    except (RuntimeError, TypeError, ValueError) as ex:
        error = f"{type(ex).__name__}: {str(ex)[:300]}"
    accepted = unexpected if loaded else []                 # keys the module lacks that the strict load accepted
    diffs = legacy_table_diffs(module, sd, accepted)
    legacy_ok = all(d is not None and d <= LEGACY_MAX for d in diffs.values())
    return {"missing": len(missing), "unexpected": len(unexpected), "shape_mismatch": len(shapes),
            "non_tensor": len(non_tensor), "first_missing": missing[:5], "first_unexpected": unexpected[:5],
            "first_shape_mismatch": shapes[:5], "first_non_tensor": non_tensor[:5],
            "strict_load": loaded, "load_error": error, "accepted_unexpected": accepted,
            "legacy_table_max_abs_diff": diffs, "legacy_table_row_max_abs_diff": legacy_table_diffs(module, sd, accepted,
                                                                                                   rows=True),
            "legacy_table_is_the_formula": {k: d is not None and d <= LEGACY_EQUAL for k, d in diffs.items()},
            "legacy_equal": LEGACY_EQUAL, "legacy_max": LEGACY_MAX,
            "ok": bool(loaded and not missing and not shapes and not non_tensor and legacy_ok)}


def compare_state_dicts(module, sd):
    """(missing, unexpected, shape mismatches) of loading `sd` into `module`, the way Lightning restores a checkpoint
    with strict loading (missing or unexpected keys fail it)."""
    own = module.state_dict()
    missing = sorted(set(own) - set(sd))
    unexpected = sorted(set(sd) - set(own))
    shapes = sorted(k for k in set(own) & set(sd)
                    if tuple(own[k].shape) != tuple(getattr(sd[k], "shape", (None,))))
    return missing, unexpected, shapes


def cmd_load_check(a):
    os.environ["MODEL_ID"] = a.model_id
    cli = build_cli(a.config, ("--trainer.accelerator", "cpu", "--trainer.devices", "1",
                               "--trainer.strategy", "auto", "--trainer.logger", "false"))
    sd, how = load_checkpoint_safely(a.ckpt)
    n_module = len(cli.model.state_dict())
    rec = {"model_id": a.model_id, "checkpoint_sha256": sha256_file(a.ckpt) if a.hash else None, "load": how,
           "keys_in_checkpoint": len(sd), "keys_in_module": n_module, **load_check_result(cli.model, sd)}
    # rslearn's predict loads --ckpt_path with weights_only=True (LightningCLI._parse_ckpt_path) and restores through
    # Lightning, whose torch.load leaves weights_only at torch's default (True since torch 2.6): a checkpoint that needs
    # stubs here would fail there, on the GPU, so it fails here
    rec["plain_weights_only"] = how == "weights_only"
    rec["ok"] = bool(rec["ok"] and rec["plain_weights_only"])
    if a.record:
        write_json(rec, a.record)
    print(json.dumps(rec))
    return 0 if rec["ok"] else 1


def deployment_windows(inv_path, run_dir):
    """[(root_index, group, name, path, epsg, bounds)] of a deployment run (exp98's or exp99's inventory)."""
    with open(inv_path) as f:
        inv = json.load(f)
    out = []
    for ri, r in enumerate(inv["roots"]):
        root = os.path.join(run_dir, r["root"])
        for g, n, p in list_windows(root):
            with open(os.path.join(p, "metadata.json")) as f:
                md_ = json.load(f)
            crs = md_["projection"]["crs"]
            mm = re.fullmatch(r"EPSG:(\d+)", crs)
            out.append((ri, root, g, n, p, int(mm.group(1)) if mm else None, [int(v) for v in md_["bounds"]]))
    return out


def deployment_windows_or_reason(inv_path, run_dir, expected=None):
    """(deployment_windows(...), None), or (None, reason) when the deployment cannot be read whole. The scratch purge
    removes files one by one (about 7 to 9 November for exp98's and exp99's), so a run directory can outlive some of
    its windows' metadata.json, or lose whole window directories, which list_windows does not see: with expected, a
    read of another count is not whole either. The reason names the exception's type only: its message could name a
    window."""
    try:
        dep = deployment_windows(inv_path, run_dir)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as ex:
        return None, f"unreadable ({type(ex).__name__}; a partial purge?)"
    if expected is not None and len(dep) != expected:
        return None, f"partial ({len(dep)} of {expected} windows read; a partial purge?)"
    return dep, None


def holding(units, dep):
    """{unit name: [deployment windows whose bounds hold the unit's pixel in the same CRS]}."""
    out = {}
    for u in units:
        out[u["name"]] = [d for d in dep if d[5] == int(u["epsg"]) and d[6][0] <= u["px"] < d[6][2]
                          and d[6][1] <= u["py"] < d[6][3]]
    return out


def cmd_aprime_config(a):
    import copy
    import yaml
    m, by = load_units(a.units)
    dep = deployment_windows(a.inv, a.run)
    hold = holding(m["units"], dep)
    chosen = {}
    for name, ds in hold.items():
        for d in ds:
            chosen.setdefault((d[0], d[1]), {}).setdefault(d[2], set()).add(d[3])
    if not chosen:
        raise SystemExit("no deployment window holds a gate-A point")
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    plan = []
    for (ri, root), groups in sorted(chosen.items()):
        der = copy.deepcopy(cfg)
        pc = der["data"]["init_args"].setdefault("predict_config", {})
        pc["groups"] = sorted(groups)
        pc["names"] = sorted(set().union(*groups.values()))
        back = copy.deepcopy(der)
        del back["data"]["init_args"]["predict_config"]["groups"], back["data"]["init_args"]["predict_config"]["names"]
        if back != cfg:
            raise SystemExit("the derived config differs from model.yaml beyond predict_config groups and names")
        path = os.path.join(a.out_dir, f"model_aprime_r{ri}.yaml")
        os.makedirs(a.out_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write("# exp111 A': exp/jobs/e102_config/model.yaml with data.init_args.predict_config.groups and .names "
                    "set to exp98's deployment windows that hold a gate-A point (exp/exp111_jobs.py aprime-config)\n")
            yaml.safe_dump(der, f, sort_keys=False)
        names = sorted(f"{g}/{n}" for g, ns in groups.items() for n in ns)
        plan.append({"root_index": ri, "root": root, "config": path, "windows": names})
    write_json(plan, os.path.join(a.out_dir, "aprime_plan.json"), private=True)
    rec = {"units": len(m["units"]), "units_in_no_deployment_window": sum(not v for v in hold.values()),
           "deployment_windows_to_predict": sum(len(p["windows"]) for p in plan), "roots": len(plan)}
    if a.record:
        write_json(rec, a.record)
    print(json.dumps(rec))
    return 0


def _item_names(wdir, layer=INPUT):
    try:
        with open(os.path.join(wdir, "items.json")) as f:
            items = json.load(f)
    except (OSError, ValueError):
        return None
    for e in items:
        if e.get("layer_name") == layer:
            return [[it.get("name") for it in grp] for grp in (e.get("serialized_item_groups") or [])]
    return None


def cmd_item_identity(a):
    """Report-only (the item-identity split beside gate A and B). A deployment that cannot be read whole is a note:
    the record says so, no flags file is written (gatecheck removes an earlier one first), and the job goes on."""
    m, by = load_units(a.units)
    dep, why = deployment_windows_or_reason(a.inv, a.run)
    if dep is None:
        rec = {"units": len(m["units"]), "skipped": f"the deployment is {why}"}
        if a.record:
            write_json(rec, a.record)
        print(json.dumps(rec))
        print("note: item identity skipped (report-only); the gate does not depend on it")
        return 0
    hold = holding(m["units"], dep)
    flags, rec = {}, {"units": len(m["units"]), "identical": 0, "differ": 0, "no_deployment_window": 0,
                      "no_items_here": 0, "no_items_there": 0, "same_scenes_other_order": 0}
    for u in m["units"]:
        ours = _item_names(os.path.join(a.root, "windows", m["group"], u["name"]))
        theirs = [_item_names(d[4]) for d in hold[u["name"]]]
        if not hold[u["name"]]:
            rec["no_deployment_window"] += 1
            flags[u["name"]] = "no_deployment_window"
        elif ours is None:
            rec["no_items_here"] += 1
            flags[u["name"]] = "no_items_here"
        elif all(t is None for t in theirs):
            rec["no_items_there"] += 1
            flags[u["name"]] = "no_items_there"
        elif any(t == ours for t in theirs):
            rec["identical"] += 1
            flags[u["name"]] = "identical"
        else:
            rec["differ"] += 1
            flags[u["name"]] = "differ"
            if any(t is not None and sorted(map(sorted, t)) == sorted(map(sorted, ours)) for t in theirs):
                rec["same_scenes_other_order"] += 1
    write_json(flags, a.flags_out, private=True)
    if a.record:
        write_json(rec, a.record)
    print(json.dumps(rec))
    return 0


def cmd_deploy_time_range(a):
    """The distinct time ranges of a deployment run's windows against the literal one. A window whose metadata.json
    is gone or unreadable (a partial purge) is counted as unreadable, a root whose windows directory is gone as missing,
    and the rest are checked; with none readable the check is recorded as not made (checked false) and the literal is used unchecked, as when the dataset is gone."""
    seen, unreadable, roots_missing = {}, 0, 0
    with open(a.inv) as f:
        inv = json.load(f)
    for r in inv["roots"]:
        rd = os.path.join(a.run, r["root"])
        if not os.path.isdir(os.path.join(rd, "windows")):
            roots_missing += 1
            continue
        for g, n, p in list_windows(rd):
            try:
                with open(os.path.join(p, "metadata.json")) as f:
                    tr = json.load(f).get("time_range")
                key = tuple(datetime.fromisoformat(t).astimezone(timezone.utc).isoformat() for t in tr) if tr else None
            except (OSError, ValueError, TypeError):
                unreadable += 1
                continue
            seen[key] = seen.get(key, 0) + 1
    want = tuple(t.isoformat() for t in TIME_RANGES[a.year])
    rec = {"year": a.year, "distinct": len(seen), "windows": sum(seen.values()), "unreadable": unreadable,
           "roots_missing": roots_missing, "roots": len(inv["roots"]), "checked": bool(seen),
           "values": [{"time_range": list(k) if k else None, "windows": v} for k, v in seen.items()],
           "equals_literal": list(seen) == [want]}
    rec["ok"] = bool(rec["equals_literal"] or not seen)
    if a.record:
        write_json(rec, a.record)
    print(json.dumps(rec))
    return 0 if rec["ok"] else 1


def cmd_remove_layer(a):
    if not a.layer.startswith(REMOVABLE_PREFIX):
        raise SystemExit(f"refused: only {REMOVABLE_PREFIX}* layers may be removed")
    roots = [a.root] if a.root else []
    if a.plan:
        with open(a.plan) as f:
            roots += sorted({p["root"] for p in json.load(f)})
    n = 0
    for root in roots:
        for g, name, p in list_windows(root):
            d = os.path.join(p, "layers", a.layer)
            if os.path.lexists(d):
                shutil.rmtree(d)
                n += 1
    print(f"removed layers/{a.layer}/ from {n} window(s)")
    return 0


def dir_bytes(path):
    n = 0
    for dp, _, fs in os.walk(path):
        for fn in fs:
            try:
                n += os.lstat(os.path.join(dp, fn)).st_blocks * 512
            except OSError:
                pass
    return n


def count_windows(roots):
    return {r: len(list_windows(r)) if os.path.isdir(os.path.join(r, "windows")) else 0 for r in roots}


def cmd_watchdog(a):
    """Every --interval seconds: the bytes under --dir (allocated blocks) and the window count of each root. Above
    --max-gb, or when a root's window count changes, it writes the reason to --reason, cancels the Slurm job --scancel
    (rslearn's workers run under `timeout`, in their own process groups) and sends SIGTERM to the process group
    --pgid (the job script's), then exits. It exits by itself when --parent is gone."""
    start = count_windows(a.roots)
    while True:
        if a.parent:
            try:
                os.kill(a.parent, 0)          # signal 0: is the job's shell still there
            except OSError:
                return 0
        gb = dir_bytes(a.dir) / 1e9
        now = count_windows(a.roots)
        why = None
        if gb > a.max_gb:
            why = f"watchdog: {gb:.1f} GB under the private directory, above the {a.max_gb} GB cap"
        elif now != start:
            why = f"watchdog: the window count changed ({sum(start.values())} -> {sum(now.values())})"
        if why:
            with open(a.reason, "w") as f:
                f.write(why + "\n")
            print(why, flush=True)
            if a.scancel and a.scancel != "none":
                import subprocess
                subprocess.run(["scancel", a.scancel], check=False)      # the whole job, rslearn's process groups too
            if a.pgid:
                os.killpg(a.pgid, signal.SIGTERM)
            return 3
        if a.once:
            print(json.dumps({"gb": round(gb, 3), "windows": sum(now.values())}))
            return 0
        time.sleep(a.interval)


def cmd_parse_model(a):
    """rslearn 0.1.17's `model predict` parser on model.yaml (--print_config: parsed and printed, nothing
    instantiated), for each model id; the values exp111 relies on, checked."""
    import contextlib
    import io
    import yaml
    out = {}
    ok = True
    for mid in a.model_ids:
        os.environ["MODEL_ID"] = mid
        buf = io.StringIO()
        code = None
        with contextlib.redirect_stdout(buf):
            try:
                from lightning.pytorch import LightningDataModule, LightningModule
                from rslearn.arg_parser import RslearnArgumentParser
                from rslearn.lightning_cli import RslearnLightningCLI
                from rslearn.utils.jsonargparse import init_jsonargparse
                init_jsonargparse()
                RslearnLightningCLI(model_class=LightningModule, datamodule_class=LightningDataModule,
                                    args=["predict", "--config", a.config, "--print_config"], subclass_mode_model=True,
                                    subclass_mode_data=True, save_config_kwargs={"overwrite": True},
                                    parser_class=RslearnArgumentParser)
            except SystemExit as ex:
                code = ex.code
        cfg = yaml.safe_load(buf.getvalue())

        def ia(d):             # jsonargparse prints every class as {class_path, init_args}
            return d["init_args"] if isinstance(d, dict) and "init_args" in d else d
        mdl = ia(ia(cfg["model"])["model"])
        enc = ia(mdl["encoder"][0])
        data = ia(cfg["data"])
        task = ia(ia(ia(data["task"])["tasks"]["segment"]))
        pc = ia(data["predict_config"])
        writer = ia([c for c in cfg["trainer"]["callbacks"] if c["class_path"].endswith("RslearnWriter")][0])
        bs = writer["layer_config"]["band_sets"][0]
        s2 = ia(data["inputs"]["sentinel2_l2a"])
        got = {"exit": code, "model_id": enc.get("model_id"), "encoder_patch_size": enc.get("patch_size"),
               "output_probs": task.get("output_probs"), "num_classes": task.get("num_classes"),
               "predict_patch": pc.get("patch_size"), "predict_overlap_ratio": pc.get("overlap_ratio"),
               "load_all_patches": pc.get("load_all_patches"), "skip_targets": pc.get("skip_targets"),
               "predict_transforms": [t["class_path"].rsplit(".", 1)[-1] for t in pc.get("transforms") or []],
               "merger_padding": ia(writer["merger"]).get("padding"),
               "writer_layer": writer.get("output_layer"), "writer_bands": bs.get("bands"),
               "writer_dtype": str(bs.get("dtype")), "writer_path_is_dataset": writer.get("path") == data.get("path"),
               "input_bands": s2["bands"], "load_all_item_groups": s2.get("load_all_item_groups"),
               "nodata_value": task.get("nodata_value"), "prob_scales": task.get("prob_scales")}
        want = {"exit": 0, "model_id": mid, "output_probs": True, "num_classes": 10, "predict_patch": 16,
                "predict_overlap_ratio": 0.25, "load_all_patches": True, "skip_targets": True,
                "predict_transforms": ["OlmoEarthNormalize"], "merger_padding": 2, "writer_path_is_dataset": True,
                "writer_layer": os.environ.get("PREDICTION_OUTPUT_LAYER"), "nodata_value": 9, "prob_scales": None,
                "input_bands": ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"],
                "writer_bands": [f"p{i}" for i in range(10)], "load_all_item_groups": True}
        got["as_expected"] = {k: got[k] == v for k, v in want.items()}
        got["dtype_float32"] = "FLOAT32" in got["writer_dtype"].upper()
        ok &= all(got["as_expected"].values()) and got["dtype_float32"]
        out[mid] = got
    rec = {"config_sha256": sha256_file(a.config), "parsed": out, "ok": bool(ok)}
    if a.record:
        write_json(rec, a.record)
    print(json.dumps(rec))
    return 0 if ok else 1


def cmd_crop_lattice(a):
    """rslearn's crop options for a window of --size px against the cell's (no files, no data)."""
    from rslearn.train.all_crops_dataset import get_window_crop_options
    cell = get_window_crop_options((CROP_PX, CROP_PX), (OVERLAP_PX, OVERLAP_PX), (0, 0, CELL_PX, CELL_PX))
    cell_x = sorted({c[0] for c in cell})
    res = {}
    for size in a.sizes:
        offs = range(0, CELL_PX - size + 1, STRIDE_PX)
        off_lat = 0
        differs = 0
        for x0 in offs:
            w = get_window_crop_options((CROP_PX, CROP_PX), (OVERLAP_PX, OVERLAP_PX), (x0, 0, x0 + size, size))
            xs = sorted({c[0] for c in w})
            off_lat += any(x not in cell_x for x in xs)
            for p in range(x0 + TRIM_PX, x0 + size - TRIM_PX):
                differs += owner_1d(p, x0, x0 + size) != owner_1d(p, 0, CELL_PX)
        res[str(size)] = {"window_starts_tried": len(offs), "windows_with_an_off_lattice_crop": off_lat,
                          "interior_pixels_whose_crop_differs_from_the_cell": differs}
    rec = {"cell_crops_per_axis": len(cell_x), "cell_crop_starts_on_12": all(x % STRIDE_PX == 0 for x in cell_x),
           "by_window_size": res}
    print(json.dumps(rec))
    return 0


def cmd_pack_check(a):
    """The pack jobs' check, the same function the record writers use: refused (exit 1) when any JSON file under the
    directories holds a key that would carry an id or a position. --expect-files: refused unless at least that many
    JSON files were read (the smoke's guard against checking an empty directory)."""
    bad, files = pack_check(a.dir)
    if bad:
        print(f"refused to pack: keys that would carry an id or a position ({len(bad)}): {bad[:5]}")
        return 1
    if a.expect_files is not None and files < a.expect_files:
        print(f"refused: {files} JSON files read, fewer than the {a.expect_files} expected")
        return 1
    print(f"pack contents checked: {files} JSON files, no id or position keys")
    return 0


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p_(name, func, *args):
        p = sub.add_parser(name)
        for spec in args:
            flag, kw = spec
            p.add_argument(flag, **kw)
        p.set_defaults(func=func)
        return p

    rec = ("--record", {"default": None})
    units = ("--units", {"required": True, "help": "the private units map of the root"})
    root = ("--root", {"required": True})
    workers = ("--workers", {"type": int, "default": 8})
    p_("no-runner", cmd_no_runner)
    p_("versions", cmd_versions, ("--core-only", {"action": "store_true"}), rec)
    p_("device", cmd_device, ("--expect-gate", {"default": None}), ("--name", {"default": None}),
       ("--commit", {"default": None}), rec)
    p_("check-configs", cmd_check_configs, ("--config-dir", {"default": CONFIG_DIR}), ("--model", {"default": MODEL_YAML}),
       ("--reverse", {"default": None}), rec)
    p_("check-dataset", cmd_check_dataset, root, units, workers, rec)
    p_("precheck", cmd_precheck, root, units, ("--config", {"default": MODEL_YAML}), workers,
       ("--max-drop-share", {"type": float, "default": 0.0}), ("--quarantine", {"default": None}),
       ("--scl", {"default": None, "help": "the root's SCL sidecar: windows with input need all twelve SCL groups"}),
       ("--run-tag", {"default": os.environ.get("SLURM_JOB_ID") or "local",
                      "help": "this run's quarantine subdirectory (e111.sh: the Slurm job)"}), rec)
    p_("prepare-summary", cmd_prepare_summary, ("--log", {"required": True}), ("--layer", {"default": INPUT}),
       ("--max-failed-share", {"type": float, "default": 1.0}), ("--plus", {"default": None}), rec)
    p_("check-groups", cmd_check_groups, root, rec)
    p_("inventory", cmd_inventory, root, units, ("--quarantine", {"default": None}), rec)
    p_("scl-sidecar", cmd_scl_sidecar, root, ("--scl", {"required": True}),
       ("--scl-config", {"default": os.path.join(E98_CONFIG, "scl_dataset.json")}))
    p_("scl-check", cmd_scl_check, root, ("--scl", {"required": True}), rec)
    p_("check-output", cmd_check_output, root, ("--layer", {"required": True}), ("--names", {"default": None}),
       ("--sample", {"type": int, "default": 8}), rec)
    p_("gate-records", cmd_gate_records, ("--rec", {"required": True, "help": "the records directory ($REC)"}))
    p_("load-check", cmd_load_check, ("--config", {"default": MODEL_YAML}), ("--ckpt", {"required": True}),
       ("--model-id", {"required": True}), ("--hash", {"action": "store_true"}), rec)
    p_("aprime-config", cmd_aprime_config, units, ("--inv", {"required": True}), ("--run", {"required": True}),
       ("--config", {"default": MODEL_YAML}), ("--out-dir", {"required": True}), rec)
    p_("item-identity", cmd_item_identity, root, units, ("--inv", {"required": True}), ("--run", {"required": True}),
       ("--flags-out", {"required": True}), rec)
    p_("deploy-time-range", cmd_deploy_time_range, ("--inv", {"required": True}), ("--run", {"required": True}),
       ("--year", {"type": int, "required": True}), rec)
    p_("remove-layer", cmd_remove_layer, ("--root", {"default": None}), ("--plan", {"default": None}),
       ("--layer", {"required": True}))
    p_("watchdog", cmd_watchdog, ("--dir", {"required": True}), ("--roots", {"nargs": "*", "default": []}),
       ("--max-gb", {"type": float, "default": 25.0}), ("--interval", {"type": float, "default": 60.0}),
       ("--pgid", {"type": int, "default": 0}), ("--scancel", {"default": None}), ("--parent", {"type": int, "default": 0}),
       ("--reason", {"required": True}), ("--once", {"action": "store_true"}))
    p_("parse-model", cmd_parse_model, ("--config", {"default": MODEL_YAML}),
       ("--model-ids", {"nargs": "+", "default": ["OLMOEARTH_V1_BASE", "OLMOEARTH_V1_2_BASE"]}), rec)
    p_("crop-lattice", cmd_crop_lattice, ("--sizes", {"type": int, "nargs": "+", "default": [124, 128]}))
    p_("pack-check", cmd_pack_check, ("--dir", {"action": "append", "required": True}),
       ("--expect-files", {"type": int, "default": None}))
    a = ap.parse_args(argv)
    if a.cmd == "precheck" and a.max_drop_share > 0 and not a.quarantine:
        ap.error("--max-drop-share above 0 needs --quarantine")
    if a.cmd == "precheck" and not 0 <= a.max_drop_share <= MAX_FAILED_SHARE_CAP:
        ap.error(f"--max-drop-share is at most {MAX_FAILED_SHARE_CAP} (the page's cap on the failure allowance)")
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
