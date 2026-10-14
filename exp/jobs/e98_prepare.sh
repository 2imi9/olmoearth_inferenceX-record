#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH -t 12:00:00
#SBATCH -J e98prep
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
# exp98 stage 2 of 4 (exp/jobs/E98_README.md): olmoearth_run's partition and build_dataset stages for Ai2's AWF
# project, over its own request geometry and dates, with the e98 configs; then the SCL sidecar. CPU only.
#   ssh aicr "sbatch --parsable --dependency=afterok:$ENV_JOB" < exp/jobs/e98_prepare.sh
# E98_SHA=<full sha> checks out that commit instead of origin/main. E98_SKIP_BUILD=1 skips build_dataset (a built run
# whose inventory or SCL step failed); the config snapshot, inventory (with its completeness check) and SCL steps always
# run, so a partial build is still refused. E98_ALLOW_INCOMPLETE=1 lets a partial area go on, recorded as such.
# E98_PILOT=1 runs the same chain over a small sub-area of Ai2's request geometry (README "Pilot first"): the snapshot's
# prediction_request_geometry.geojson becomes Ai2's geometry clipped to a square of 0.05 degrees around its
# representative_point() (about 1 to 4 cells of 1024 px), and every path is a pilot one (awf_pilot_config, awf_pilot_run,
# awf_pilot_scl, logs/pilot_*, *_pilot.json, exp/out/exp98/pilot/), so nothing mixes with a full run.
#
# What build_dataset runs is olmoearth_run's (olmoearth_projects/olmoearth_run/olmoearth_run.py:128-158 at f3c9b0c8:
# one_stage always partitions, then BUILD_DATASET calls runner.build_dataset over every partition). Ai2's AWF doc
# says inference covers every 1024 x 1024 cell of 10 m pixels that intersects the request geometry
# (docs/awf.md at f3c9b0c8), and olmoearth_run.yaml partitions the request on a 1-degree grid
# (partition_request_geometry) and windows it on a 1024-pixel grid in UTM (prepare_window_geometries).
#
# Data volume, estimated (not measured): the request geometry covers about 19,800 km2, about 224 cells of
# 10.24 x 10.24 km in one UTM zone, so about 220 to 280 windows (more if windows on a partition edge are duplicated).
# Per window the sentinel2 layer holds 12 item groups x 12 bands x 1024 x 1024 uint16 = 288 MiB raw, LZW-compressed
# GeoTIFFs (rslearn/utils/raster_format.py:505 @ v0.0.27), so about 40 to 65 GB on scratch. Reading it pulls a
# similar amount from Planetary Computer (COG windows of every scene in each group, often 2 scenes) plus one product
# XML per band read for harmonization (rslearn/data_sources/planetary_computer.py:450-467). The SCL sidecar adds
# 12 x 1024 x 1024 uint8 per window, about 1 GB raw, far less compressed. Expected 1 to 6 hours; the 12-hour limit
# leaves room for Planetary Computer throttling (no subscription key is set).
set -euo pipefail
# The soft limit on open files is 1,024 on the cpu nodes (hard 131,072); olmoearth_run's worker pools exhaust it
# (job 1243542 failed in 26 s on "Too many open files" while starting its dataset-build pool).
ulimit -n 65536
REPO=/home/qi_zim_neu/olmoearth_inferenceX
cd "$REPO" || exit 1
git fetch -q origin main
if [ -n "${E98_SHA:-}" ]; then
  # A pinned stage checks out its own commit, whatever main has become since the chain was submitted (the chain spans
  # many hours and main moves). The checkout is left detached; a later `git reset --hard origin/main` still works on it.
  git cat-file -e "$E98_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E98_SHA" \
    || { echo "E98_SHA=$E98_SHA is not a commit on origin"; exit 2; }
  git checkout -q --force --detach "$E98_SHA"
  [ "$(git rev-parse HEAD)" = "$E98_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E98_SHA=$E98_SHA (give the full sha)"; exit 2; }
else
  git reset -q --hard origin/main
fi
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
DEPLOY=$SCRATCH/deploy
export E98_PILOT=${E98_PILOT:-0}
case "$E98_PILOT" in 0|1) ;; *) echo "E98_PILOT must be 0 or 1, got '$E98_PILOT'"; exit 2;; esac
if [ "$E98_PILOT" = 1 ]; then TAG=awf_pilot SFX=_pilot LP=pilot_ OUT=$REPO/exp/out/exp98/pilot
else TAG=awf SFX= LP= OUT=$REPO/exp/out/exp98; fi
RUN=$DEPLOY/${TAG}_run         # olmoearth_run's --scratch_path
CFG=$DEPLOY/${TAG}_config      # the config directory every stage reads (--config_path), a snapshot taken here
SCL=$DEPLOY/${TAG}_scl         # the SCL sidecar datasets
INV=$DEPLOY/prepare_inventory$SFX.json
CFGSHA=$DEPLOY/${TAG}_config.sha256
PY=$DEPLOY/venv/bin/python
CLONE=$DEPLOY/olmoearth_projects
export HF_HOME=$SCRATCH/hf
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
NCPU=${SLURM_CPUS_PER_TASK:-8}
# model.yaml's ${VAR}s; rslearn substitutes an unset one with "" (rslearn/template_params.py:19-24 @ v0.0.27).
# olmoearth_run is expected to set DATASET_PATH itself; the rest are set here so that none is empty.
export NUM_WORKERS=$NCPU
# olmoearth_run's dataset build starts os.cpu_count() * DATASET_BUILD_WORKERS_PER_CPU processes
# (olmoearth_run/runner/steps/dataset_build_step_definition.py:40, default 4 in olmoearth_run/config.py:36), and
# os.cpu_count() is the node's 128, not the job's CPUs: 512 processes. One per node CPU is enough for a fetch-bound build.
export DATASET_BUILD_WORKERS_PER_CPU=1
export PREDICTION_OUTPUT_LAYER=output
export TRAINER_DATA_PATH=$DEPLOY/${LP}trainer_data
export EXTRA_FILES_PATH=$DEPLOY/${LP}extra_files
export WANDB_MODE=disabled WANDB_PROJECT=e98 WANDB_NAME=e98_awf WANDB_ENTITY=e98-local
export E98_RUN=$RUN E98_CFG=$CFG E98_SCL=$SCL E98_DEPLOY=$DEPLOY E98_REPO=$REPO E98_NCPU=$NCPU
export E98_INV=$INV E98_SCLCHK=$DEPLOY/scl_check$SFX.json E98_CFGSHA=$CFGSHA

# Upstream logs may print geometry or window bounds. They stay in $DEPLOY/logs; what this job echoes from them is
# masked: decimals with 2+ places and integers of 4+ digits become <n>.
redact() { sed -E 's/-?[0-9]+\.[0-9]{2,}/<n>/g; s/-?[0-9]{4,}/<n>/g'; }

echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp98 prepare, commit $(git rev-parse HEAD), $NCPU CPUs, pilot $E98_PILOT =="
[ -s "$DEPLOY/versions.json" ] && [ -x "$PY" ] || { echo "no deployment environment: run e98_env.sh first"; exit 1; }
[ "$(git -C "$CLONE" rev-parse HEAD)" = "f3c9b0c89c7670b3525647dda4c14b76245d682d" ] || { echo "clone moved"; exit 1; }
# The venv must carry e98_env.sh's rslearn get_item_by_name fix (its .pth); without it the first direct materialize
# fails after about 34 minutes (job 1243637). A venv built before the fix fails here in seconds instead.
"$PY" -c 'from rslearn.data_sources.direct_materialize_data_source import DirectMaterializeDataSource as D; assert "get_item_by_name" not in vars(D), "rerun e98_env.sh (rslearn fix)"' \
  || { echo "the venv lacks the rslearn get_item_by_name fix: rerun e98_env.sh"; exit 1; }
CKPT=$(cat "$DEPLOY/checkpoint_path.txt")
[ -s "$CKPT" ] || { echo "checkpoint missing at the path e98_env.sh recorded; rerun e98_env.sh"; exit 1; }
read -r ST_BUILD _ _ _ < "$DEPLOY/stage_spelling.txt"
d=$CLONE/olmoearth_projects
while :; do
  if [ -e "$d/.env" ]; then echo "a .env exists in $d, which main.py's load_dotenv would read; move it away"; exit 2; fi
  [ "$d" = / ] && break
  d=$(dirname "$d")
done
mkdir -p "$DEPLOY/logs" "$OUT" "$TRAINER_DATA_PATH" "$EXTRA_FILES_PATH"

echo "== the config snapshot: e98's model.yaml and dataset.json, Ai2's olmoearth_run.yaml and request geometry =="
NEW=$(mktemp -d "$DEPLOY/${TAG}_config.new.XXXXXX")
cp exp/jobs/e98_config/model.yaml exp/jobs/e98_config/dataset.json "$NEW/"
cp "$CLONE/olmoearth_run_data/awf/olmoearth_run.yaml" "$CLONE/olmoearth_run_data/awf/prediction_request_geometry.geojson" "$NEW/"
E98_NEW=$NEW "$PY" - <<'EOF'
# Refuses unless the snapshot is Ai2's AWF config with exactly the documented changes (e98_config/CHANGES.md).
import hashlib, json, os, yaml
from rslearn.config import DatasetConfig
new, awf = os.environ["E98_NEW"], os.path.join(os.environ["E98_DEPLOY"], "olmoearth_projects", "olmoearth_run_data", "awf")
pins = {"dataset.json": "7da20e92b51741f3a2c61137e305bd5a8b59938a107213372099886897751b4b",
        "model.yaml": "08e0432044c5c67c683372c85c632eb31085b6ce8b4dae49b740475e5b37a7ea",
        "olmoearth_run.yaml": "1e1c8f709aadd86f84795723d35e07fc0aa79cef6e076c9a75d1ebec36d705ba",
        "prediction_request_geometry.geojson": "fee3ce01008d26c6f3c239e68be70851035eb157d5da70bd72cb46dbc8623784"}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
for f, h in pins.items():
    assert sha(os.path.join(awf, f)) == h, f"Ai2's {f} is not the pinned one"
for f in ("olmoearth_run.yaml", "prediction_request_geometry.geojson"):
    assert sha(os.path.join(new, f)) == pins[f], f
def diff(x, y, path=""):
    if isinstance(x, dict) and isinstance(y, dict):
        out = []
        for k in sorted(set(x) | set(y), key=str):
            if k not in x: out.append(("added", f"{path}/{k}", y[k]))
            elif k not in y: out.append(("removed", f"{path}/{k}", x[k]))
            else: out += diff(x[k], y[k], f"{path}/{k}")
        return out
    if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        return [d for i, (u, v) in enumerate(zip(x, y)) for d in diff(u, v, f"{path}[{i}]")]
    return [] if x == y else [("changed", path, (x, y))]
P = [f"p{i}" for i in range(10)]
got = diff(yaml.safe_load(open(os.path.join(awf, "model.yaml"))), yaml.safe_load(open(os.path.join(new, "model.yaml"))))
want = [("added", "/data/init_args/task/init_args/tasks/segment/init_args/output_probs", True),
        ("added", "/trainer/callbacks[3]/init_args/layer_config",
         {"type": "RASTER", "band_sets": [{"bands": P, "dtype": "FLOAT32"}]})]
assert got == want, f"model.yaml differs from Ai2's by {got}"
got = diff(json.load(open(os.path.join(awf, "dataset.json"))), json.load(open(os.path.join(new, "dataset.json"))))
assert got == [("changed", "/layers/output/band_sets[0]/bands", (["output"], P))], f"dataset.json differs by {got}"
DatasetConfig.model_validate(json.load(open(os.path.join(new, "dataset.json"))))
print("snapshot = Ai2's AWF config + output_probs + the p0..p9 float32 output layer; nothing else differs")
if os.environ["E98_PILOT"] == "1":
    # Ai2's geometry (its sha256 checked above) clipped to a 0.05-degree square around representative_point(), a point
    # inside it, so the square's part inside the geometry is never empty. Same FeatureCollection, same properties
    # (oe_start_time, oe_end_time); one Polygon per feature as in Ai2's file (the largest part if a clip splits one).
    # Only its sha256 and an area share are printed, never a coordinate.
    import shapely
    from shapely.geometry import mapping, shape
    gp = os.path.join(new, "prediction_request_geometry.geojson")
    gj = json.load(open(gp))
    req = shapely.union_all([shape(f["geometry"]) for f in gj["features"]])
    c = req.representative_point()
    square = shapely.box(c.x - 0.025, c.y - 0.025, c.x + 0.025, c.y + 0.025)
    feats = []
    for f in gj["features"]:
        g = shape(f["geometry"]).intersection(square)
        if g.is_empty:
            continue
        if g.geom_type == "MultiPolygon" and f["geometry"]["type"] == "Polygon":
            g = max(g.geoms, key=lambda p: p.area)
        assert g.geom_type == f["geometry"]["type"], (g.geom_type, f["geometry"]["type"])
        feats.append({**f, "geometry": mapping(g)})
    assert feats, "the pilot square meets no feature of Ai2's geometry"
    pilot = {**gj, "features": feats}
    with open(gp, "w") as fo:
        json.dump(pilot, fo)
    share = shapely.union_all([shape(f["geometry"]) for f in feats]).area / req.area
    print(f"pilot geometry: {len(feats)} of {len(gj['features'])} feature(s), {share:.2e} of Ai2's area (in degrees^2), "
          f"sha256 {sha(gp)}")
EOF
if [ -d "$CFG" ]; then
  if diff -rq "$CFG" "$NEW" >/dev/null; then rm -rf "$NEW"
  elif [ -d "$RUN" ]; then
    echo "$CFG differs from this commit's configs and $RUN exists: a run must not mix configs. Move both away to start over."
    diff -r "$CFG" "$NEW" --exclude=prediction_request_geometry.geojson || true
    rm -rf "$NEW"; exit 2
  else rm -rf "$CFG"; mv "$NEW" "$CFG"; fi
else mv "$NEW" "$CFG"; fi
( cd "$CFG" && sha256sum model.yaml dataset.json olmoearth_run.yaml prediction_request_geometry.geojson ) > "$CFGSHA"

if [ "${E98_SKIP_BUILD:-0}" != 1 ]; then
  echo "== $(date -Is) build_dataset (olmoearth_run one_stage --stage $ST_BUILD) =="
  LOG=$DEPLOY/logs/${LP}build_dataset-${SLURM_JOB_ID:-none}.log
  mkdir -p "$RUN"
  cd "$DEPLOY" || exit 1
  if ! "$PY" -m olmoearth_projects.main olmoearth_run one_stage --config_path "$CFG" --scratch_path "$RUN" \
       --checkpoint_path "$CKPT" --stage "$ST_BUILD" > "$LOG" 2>&1; then
    echo "build_dataset failed; the last lines of $LOG, masked:"; tail -n 60 "$LOG" | redact; exit 1
  fi
  cd "$REPO" || exit 1
  echo "$(date -Is) build_dataset done; $(grep -c -i -E 'error|exception' "$LOG" || true) lines of $LOG mention an error"
fi

echo "== the built dataset, inventoried and checked for completeness (counts and shares only; no bounds, no names) =="
# Root directories and window groups are named by olmoearth_run's 1-degree partitioner, whose naming is not public and
# may carry a cell's corner. Here they are root_<i> and group_<j>; the real names stay in this scratch inventory's
# "root" and "group_names" fields, which the tracked copy (exp/out/exp98/) leaves out.
# Completeness: every window has all 12 monthly sentinel2 item groups (Ai2's AWF config asks for 12 periods), every
# window has sentinel2 items, and the windows cover the request geometry (the share of its area outside every window,
# measured in the pixel grid of the most common window projection, at most 1e-4). Otherwise prepare stops with
# "complete": false in the inventory, which predict and collect refuse; E98_ALLOW_INCOMPLETE=1 lets a partial area go on,
# recorded as "allow_incomplete": true and carried into predict_check.json and e98_collect.json.
"$PY" - <<'EOF'
import collections, glob, json, os, re
import shapely
from rslearn.config import DatasetConfig
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
run, deploy = os.environ["E98_RUN"], os.environ["E98_DEPLOY"]
allow = os.environ.get("E98_ALLOW_INCOMPLETE", "0") == "1"
roots = sorted(os.path.dirname(c) for c in glob.glob(os.path.join(run, "**", "config.json"), recursive=True)
               if os.path.isdir(os.path.join(os.path.dirname(c), "windows")))
assert roots, "no rslearn dataset (config.json beside windows/) under the run directory"
# compared as parsed rslearn layer configs (LayerConfig.__eq__ compares model_dump, rslearn/config/dataset.py:529-537),
# so defaults written out by olmoearth_run do not count as a difference
ours = DatasetConfig.model_validate(json.load(open(os.path.join(os.environ["E98_CFG"], "dataset.json")))).layers
inv = {"run": run, "pilot": os.environ["E98_PILOT"] == "1", "roots": []}
boxes = []                                  # (projection as sorted JSON, pixel bounds) of every window, for coverage only
for i, root in enumerate(roots):
    layers = DatasetConfig.model_validate(json.load(open(os.path.join(root, "config.json")))).layers
    r = {"root_id": f"root_{i}", "root": os.path.relpath(root, run), "layers": sorted(layers),
         "config_layers_equal_e98": {k: layers.get(k) == v for k, v in ours.items()},
         "windows": 0, "with_sentinel2": 0, "groups_completed": collections.Counter(),
         "items_per_group": collections.Counter(), "sizes": collections.Counter(), "crs": collections.Counter()}
    groups = collections.Counter()
    for wdir in sorted(glob.glob(os.path.join(root, "windows", "*", "*"))):
        if not os.path.isfile(os.path.join(wdir, "metadata.json")):
            continue
        meta = json.load(open(os.path.join(wdir, "metadata.json")))
        b = meta["bounds"]
        boxes.append((json.dumps(meta["projection"], sort_keys=True), tuple(b)))
        r["windows"] += 1
        groups[meta["group"]] += 1
        r["sizes"][f"{b[2] - b[0]}x{b[3] - b[1]}"] += 1
        r["crs"][str(meta["projection"]["crs"])] += 1
        ldir = os.path.join(wdir, "layers")
        done = [d for d in (os.listdir(ldir) if os.path.isdir(ldir) else [])
                if re.fullmatch(r"sentinel2(\.\d+)?", d) and os.path.exists(os.path.join(ldir, d, "completed"))]
        r["groups_completed"][len(done)] += 1
        items = os.path.join(wdir, "items.json")
        if os.path.exists(items):
            s2 = [ld for ld in json.load(open(items)) if ld["layer_name"] == "sentinel2"]
            if s2 and s2[0]["serialized_item_groups"]:
                r["with_sentinel2"] += 1
            for ld in s2:
                for g in ld["serialized_item_groups"]:
                    r["items_per_group"][len(g)] += 1
    r["group_names"] = sorted(groups)                           # scratch only
    r["windows_per_group"] = {f"group_{j}": groups[g] for j, g in enumerate(r["group_names"])}
    r["bytes"] = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(root) for f in fs)
    for k in ("groups_completed", "items_per_group", "sizes", "crs"):
        r[k] = {str(a): n for a, n in sorted(r[k].items(), key=lambda t: str(t[0]))}
    inv["roots"].append(r)
    print(f"{r['root_id']}: {r['windows']} windows in {len(groups)} group(s), {r['with_sentinel2']} with "
          f"sentinel2 item groups; completed groups per window {r['groups_completed']}; sizes {r['sizes']}; "
          f"{len(r['crs'])} CRS; {r['bytes'] / 2**30:.1f} GiB; config layers equal to e98's: {r['config_layers_equal_e98']}")

# coverage of the request geometry by the windows, in the pixel grid of the most common window projection
gj = json.load(open(os.path.join(os.environ["E98_CFG"], "prediction_request_geometry.geojson")))
assert gj.get("crs") is None, "the request geometry declares a CRS; GeoJSON's default (WGS84) was expected"
req = shapely.union_all([shapely.geometry.shape(f["geometry"]) for f in gj["features"]])
target_key = collections.Counter(p for p, _ in boxes).most_common(1)[0][0]
target = Projection.deserialize(json.loads(target_key))
req_px = STGeometry(WGS84_PROJECTION, shapely.segmentize(req, 0.01), None).to_projection(target).shp
cover = []
for p, b in boxes:
    box = shapely.box(*b)
    if p != target_key:
        box = STGeometry(Projection.deserialize(json.loads(p)), shapely.segmentize(box, 64), None).to_projection(target).shp
    cover.append(box)
uncovered = float(req_px.difference(shapely.union_all(cover)).area / req_px.area)
# informational: cells of the target windows' own grid (size and offset from those windows) that overlap the request
tb = [b for p, b in boxes if p == target_key]
w, h = collections.Counter((b[2] - b[0], b[3] - b[1]) for b in tb).most_common(1)[0][0]
ox, oy = tb[0][0] % w, tb[0][1] % h
x0, y0, x1, y1 = req_px.bounds
expected = sum(1 for cx in range(int((x0 - ox) // w), int((x1 - ox) // w) + 1)
               for cy in range(int((y0 - oy) // h), int((y1 - oy) // h) + 1)
               if req_px.intersection(shapely.box(ox + cx * w, oy + cy * h, ox + (cx + 1) * w, oy + (cy + 1) * h)).area > 0)

n = sum(r["windows"] for r in inv["roots"])
n12 = sum(r["groups_completed"].get("12", 0) for r in inv["roots"])
ns2 = sum(r["with_sentinel2"] for r in inv["roots"])
inv.update(windows=n, windows_with_12_groups=n12, windows_with_sentinel2=ns2,
           distinct_window_cells=len(set(boxes)), projections=len({p for p, _ in boxes}),
           expected_cells_on_main_grid=expected, request_share_outside_windows=uncovered)
inv["complete"] = bool(n > 0 and n12 == n and ns2 == n and uncovered <= 1e-4)
inv["allow_incomplete"] = allow
print(f"{n} windows ({len(set(boxes))} distinct cells; {expected} cells of the main window grid overlap the request "
      f"geometry), {n12} with all 12 sentinel2 item groups, {ns2} with sentinel2 items; share of the request geometry "
      f"outside every window {uncovered:.2e}; complete: {inv['complete']}")
json.dump(inv, open(os.environ["E98_INV"], "w"), indent=1)
assert n12 > 0, "no window has all 12 sentinel2 item groups completed"
bad = [r["root_id"] for r in inv["roots"] if not r["config_layers_equal_e98"].get("sentinel2")]
assert not bad, f"config.json's sentinel2 layer is not the e98 (= Ai2's) one in {bad}"
if not all(r["config_layers_equal_e98"].get("output") for r in inv["roots"]):
    print("NOTE: config.json's output layer is not e98's p0..p9; the writer then relies on model.yaml's layer_config")
if not inv["complete"]:
    assert allow, ("the built dataset does not cover the AWF project's request geometry completely (above); rerun "
                   "build_dataset (without E98_SKIP_BUILD), or set E98_ALLOW_INCOMPLETE=1 to go on with a partial area")
    print("WARNING: going on with a partial area (E98_ALLOW_INCOMPLETE=1); every later record says so")
EOF

echo "== $(date -Is) the SCL sidecar: the run's sentinel2 item groups, re-read for their SCL band =="
"$PY" - <<'EOF'
import glob, json, os, shutil
run, scl = os.environ["E98_RUN"], os.environ["E98_SCL"]
scl_cfg = json.load(open(os.path.join(os.environ["E98_REPO"], "exp", "jobs", "e98_config", "scl_dataset.json")))
inv = json.load(open(os.environ["E98_INV"]))
made = 0
for r in inv["roots"]:
    root, side = os.path.join(run, r["root"]), os.path.join(scl, r["root"])
    os.makedirs(side, exist_ok=True)
    with open(os.path.join(side, "config.json"), "w") as f:
        json.dump(scl_cfg, f, indent=1)
    for wdir in sorted(glob.glob(os.path.join(root, "windows", "*", "*"))):
        items = os.path.join(wdir, "items.json")
        if not os.path.exists(items):
            continue
        s2 = [ld for ld in json.load(open(items)) if ld["layer_name"] == "sentinel2"]
        if not s2:
            continue
        want = [{"layer_name": "sentinel2_scl", "serialized_item_groups": s2[0]["serialized_item_groups"],
                 "materialized": False}]
        sdir = os.path.join(side, os.path.relpath(wdir, root))
        sitems = os.path.join(sdir, "items.json")
        if os.path.exists(sitems):
            old = json.load(open(sitems))
            if [o["serialized_item_groups"] for o in old] == [w["serialized_item_groups"] for w in want]:
                continue                       # already built from these items; materialize skips completed groups
            shutil.rmtree(sdir)                # the run's items changed: rebuild this window
        os.makedirs(sdir)
        shutil.copy2(os.path.join(wdir, "metadata.json"), os.path.join(sdir, "metadata.json"))
        with open(sitems, "w") as f:
            json.dump(want, f)
        made += 1
print(f"sidecar windows written this time: {made}")
EOF
k=0
for side in $("$PY" -c 'import json,os; inv=json.load(open(os.environ["E98_INV"])); print(" ".join(os.path.join(os.environ["E98_SCL"], r["root"]) for r in inv["roots"]))'); do
  k=$((k + 1))
  LOG=$DEPLOY/logs/${LP}scl_materialize-$k-${SLURM_JOB_ID:-none}.log
  if ! "$DEPLOY/venv/bin/rslearn" dataset materialize --root "$side" --workers "$NCPU" \
       --retry-max-attempts 5 --retry-backoff-seconds 60 > "$LOG" 2>&1; then
    echo "SCL materialize failed for root_$((k - 1)); the last lines of $LOG, masked:"; tail -n 40 "$LOG" | redact; exit 1
  fi
done

echo "== the sidecar checked against the run: group counts, SCL codes, coverage agreement on sampled windows =="
"$PY" - <<'EOF'
import glob, json, os, re
import numpy as np, rasterio
run, scl, deploy = os.environ["E98_RUN"], os.environ["E98_SCL"], os.environ["E98_DEPLOY"]
inv = json.load(open(os.environ["E98_INV"]))
def groups(wdir, name):
    ldir = os.path.join(wdir, "layers")
    if not os.path.isdir(ldir):
        return {}
    out = {}
    for d in os.listdir(ldir):
        m = re.fullmatch(re.escape(name) + r"(?:\.(\d+))?", d)
        if m and os.path.exists(os.path.join(ldir, d, "completed")):
            out[int(m.group(1) or 0)] = os.path.join(ldir, d)
    return out
check = {"windows": 0, "mismatched_group_counts": 0, "sampled": [], "codes": {}}
codes = np.zeros(256, np.int64)
for r in inv["roots"]:
    root, side = os.path.join(run, r["root"]), os.path.join(scl, r["root"])
    wins = sorted(glob.glob(os.path.join(side, "windows", "*", "*")))
    for w in wins:
        rel = os.path.relpath(w, side)
        g_run, g_scl = groups(os.path.join(root, rel), "sentinel2"), groups(w, "sentinel2_scl")
        check["windows"] += 1
        if sorted(g_run) != sorted(g_scl):
            check["mismatched_group_counts"] += 1
    picks = sorted(set(np.linspace(0, len(wins) - 1, min(8, len(wins))).round().astype(int).tolist())) if wins else []
    for i in picks:
        w = wins[i]
        rel = os.path.relpath(w, side)
        g_run, g_scl = groups(os.path.join(root, rel), "sentinel2"), groups(w, "sentinel2_scl")
        for gi in sorted(g_scl):
            with rasterio.open(os.path.join(g_scl[gi], "SCL", "geotiff.tif")) as s:
                a = s.read(1)
                shape_scl = a.shape
            refl = glob.glob(os.path.join(g_run[gi], "*", "geotiff.tif"))
            assert len(refl) == 1, refl
            with rasterio.open(refl[0]) as s:
                b = s.read()
            assert b.shape[1:] == shape_scl, (b.shape, shape_scl)
            codes += np.bincount(a.ravel(), minlength=256)
            cov_s, cov_r = a != 0, (b != 0).any(axis=0)
            agree = float((cov_s == cov_r).mean())
            b02 = b[1].astype(np.float64)               # band order B01, B02, ... as in dataset.json
            cloud, clear = np.isin(a, (8, 9)), np.isin(a, (4, 5))
            check["sampled"].append({"window_index": int(i), "group": gi, "coverage_agreement": agree,
                                     "scl_covered": float(cov_s.mean()),
                                     "b02_cloud_over_clear": (float(b02[cloud].mean() / b02[clear].mean())
                                                              if cloud.sum() > 100 and clear.sum() > 100 else None)})
check["codes"] = {str(k): int(v) for k, v in enumerate(codes) if v}
json.dump(check, open(os.environ["E98_SCLCHK"], "w"), indent=1)
agrees = [s["coverage_agreement"] for s in check["sampled"]]
print(f"{check['windows']} sidecar windows; {check['mismatched_group_counts']} with another group count than the run; "
      f"{len(agrees)} sampled groups, coverage agreement min {min(agrees) if agrees else None}; SCL codes {check['codes']}")
assert check["windows"] > 0 and check["mismatched_group_counts"] == 0
assert set(int(k) for k in check["codes"]) <= set(range(12)), "values outside SCL's 0..11"
assert agrees and min(agrees) >= 0.98, "SCL coverage does not match the reflectance coverage of the same group"
EOF
"$PY" - <<'EOF'
import json, os
inv = json.load(open(os.environ["E98_INV"]))
inv["scl_check"] = json.load(open(os.environ["E98_SCLCHK"]))
inv["config_sha256"] = dict(reversed(l.split()) for l in open(os.environ["E98_CFGSHA"]))
inv["job"] = os.environ.get("SLURM_JOB_ID")
json.dump(inv, open(os.environ["E98_INV"], "w"), indent=1)
EOF
# the tracked copy: without the real root and group names (scratch keeps them; see the inventory step)
"$PY" - "$INV" "$OUT/e98_prepare_inventory.json" <<'EOF'
import json, sys
inv = json.load(open(sys.argv[1]))
inv.pop("run", None)
for r in inv["roots"]:
    r.pop("root", None)
    r.pop("group_names", None)
json.dump(inv, open(sys.argv[2], "w"), indent=1)
EOF
echo "== $(date -Is) done: $INV, $OUT/e98_prepare_inventory.json; $(du -sh "$RUN" | cut -f1) in $RUN =="
