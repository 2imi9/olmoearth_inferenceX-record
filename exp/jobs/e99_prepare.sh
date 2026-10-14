#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 48
#SBATCH --mem=192G
#SBATCH -t 12:00:00
#SBATCH -J e99prep
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
# exp99 stage 2 (exp/jobs/E99_README.md): olmoearth_run's partition and build_dataset stages for Ai2's AWF project with
# exp98's configs (e98_config: output_probs, the p0..p9 float32 layer) over exp99's request geometry, one small square
# around each TimeSync plot the area rule selected, dated 2017 (e99_select.sh wrote it on scratch); then the SCL
# sidecar, as exp98's prepare does. CPU only. At most 48 CPUs: the cpu partition caps each user's CPUs
# (QOSMaxCpuPerUserLimit), and a 64-CPU job reaches the cap and blocks every other cpu job of ours.
#   ssh aicr "E99_SHA=$SHA sbatch --parsable --dependency=afterok:$SELECT_JOB" < exp/jobs/e99_prepare.sh
# E99_SHA=<full sha> checks out that commit in exp99's clone instead of origin/main. E99_PILOT=1 runs the same chain
# over the pilot's 4 plots, on pilot paths (awf2017_pilot_*, *_e99_pilot.json, exp/out/exp99/pilot/), so nothing mixes
# with a full run. E99_SKIP_BUILD=1 skips build_dataset (a built run whose inventory or SCL step failed); the snapshot,
# inventory (with its completeness check) and SCL steps always run. E99_ALLOW_INCOMPLETE=1 lets a run go on whose
# windows miss item groups or plots (2017 had Sentinel-2A alone until mid-year, so a 30-day period with no scene is
# possible); every later record then says so, and the plots without input are counted by the analysis.
#
# Volume, estimated from exp98's (not measured): 309 plots in about 263 distinct 1024-px cells (if the window grid is
# anchored at the CRS origin; 10 of the plots lie in UTM zone 36, the rest in 37), so about 265 to 300 windows, about
# 45 to 75 GB of Sentinel-2 on scratch and 1 to 7 hours; the pilot 4 windows. The 12-hour limit leaves room for
# Planetary Computer throttling.
set -euo pipefail
# The soft limit on open files is 1,024 on the cpu nodes (hard 131,072); olmoearth_run's worker pools exhaust it
# (job 1243542 failed in 26 s on "Too many open files" while starting its dataset-build pool).
ulimit -n 65536
REPO=/home/qi_zim_neu/olmoearth_inferenceX
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
CODE=$SCRATCH/e99_code
if [ ! -d "$CODE/.git" ]; then
  git clone -q "$(git -C "$REPO" remote get-url origin)" "$CODE"
fi
cd "$CODE" || exit 1
git fetch -q origin main
if [ -n "${E99_SHA:-}" ]; then
  # A pinned stage checks out its own commit, whatever main has become since the chain was submitted.
  git cat-file -e "$E99_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E99_SHA" \
    || { echo "E99_SHA=$E99_SHA is not a commit on origin"; exit 2; }
  git checkout -q --force --detach "$E99_SHA"
  [ "$(git rev-parse HEAD)" = "$E99_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E99_SHA=$E99_SHA (give the full sha)"; exit 2; }
else
  git checkout -q --force --detach origin/main
fi
DEPLOY=$SCRATCH/deploy
SEL=$DEPLOY/e99_select
export E99_PILOT=${E99_PILOT:-0}
case "$E99_PILOT" in 0|1) ;; *) echo "E99_PILOT must be 0 or 1, got '$E99_PILOT'"; exit 2;; esac
if [ "$E99_PILOT" = 1 ]; then TAG=awf2017_pilot SFX=_e99_pilot LP=e99_pilot_ OUT=$REPO/exp/out/exp99/pilot
  GEOM=$SEL/prediction_request_geometry_pilot.geojson
else TAG=awf2017 SFX=_e99 LP=e99_ OUT=$REPO/exp/out/exp99; GEOM=$SEL/prediction_request_geometry.geojson; fi
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
export NUM_WORKERS=$NCPU
# olmoearth_run's dataset build starts os.cpu_count() * DATASET_BUILD_WORKERS_PER_CPU processes, and os.cpu_count() is
# the node's 128, not the job's CPUs (exp98's prepare): one per node CPU is enough for a fetch-bound build.
export DATASET_BUILD_WORKERS_PER_CPU=1
export PREDICTION_OUTPUT_LAYER=output
export TRAINER_DATA_PATH=$DEPLOY/${LP}trainer_data
export EXTRA_FILES_PATH=$DEPLOY/${LP}extra_files
export WANDB_MODE=disabled WANDB_PROJECT=e99 WANDB_NAME=e99_awf2017 WANDB_ENTITY=e99-local
export E99_TAG=$TAG E99_RUN=$RUN E99_CFG=$CFG E99_SCL=$SCL E99_DEPLOY=$DEPLOY E99_CODE=$CODE E99_NCPU=$NCPU E99_GEOM=$GEOM
export E99_INV=$INV E99_SCLCHK=$DEPLOY/scl_check$SFX.json E99_CFGSHA=$CFGSHA E99_SEL=$SEL

# Upstream logs may print geometry or window bounds. They stay in $DEPLOY/logs; what this job echoes from them is
# masked: decimals with 2+ places and integers of 4+ digits become <n>.
redact() { sed -E 's/-?[0-9]+\.[0-9]{2,}/<n>/g; s/-?[0-9]{4,}/<n>/g'; }

echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp99 prepare, commit $(git rev-parse HEAD), $NCPU CPUs, pilot $E99_PILOT =="
[ -s "$DEPLOY/versions.json" ] && [ -x "$PY" ] || { echo "no deployment environment: run exp/jobs/e98_env.sh first"; exit 1; }
[ "$(git -C "$CLONE" rev-parse HEAD)" = "f3c9b0c89c7670b3525647dda4c14b76245d682d" ] || { echo "clone moved"; exit 1; }
# exp98's environment, reused: the venv must carry e98_env.sh's rslearn get_item_by_name fix (its .pth)
"$PY" -c 'from rslearn.data_sources.direct_materialize_data_source import DirectMaterializeDataSource as D; assert "get_item_by_name" not in vars(D), "rerun e98_env.sh (rslearn fix)"' \
  || { echo "the venv lacks the rslearn get_item_by_name fix: rerun e98_env.sh"; exit 1; }
CKPT=$(cat "$DEPLOY/checkpoint_path.txt")
[ -s "$CKPT" ] || { echo "checkpoint missing at the path e98_env.sh recorded; rerun e98_env.sh"; exit 1; }
read -r ST_BUILD _ _ _ < "$DEPLOY/stage_spelling.txt"
test -s "$GEOM" && test -s "$SEL/request.sha256" || { echo "no request geometry at $GEOM: run e99_select.sh first"; exit 1; }
( cd "$SEL" && sha256sum -c --quiet request.sha256 ) || { echo "$SEL changed since e99_select.sh wrote it"; exit 1; }
d=$CLONE/olmoearth_projects
while :; do
  if [ -e "$d/.env" ]; then echo "a .env exists in $d, which main.py's load_dotenv would read; move it away"; exit 2; fi
  [ "$d" = / ] && break
  d=$(dirname "$d")
done
mkdir -p "$DEPLOY/logs" "$OUT" "$TRAINER_DATA_PATH" "$EXTRA_FILES_PATH"

echo "== the config snapshot: exp98's model.yaml and dataset.json, Ai2's olmoearth_run.yaml, exp99's request geometry =="
NEW=$(mktemp -d "$DEPLOY/${TAG}_config.new.XXXXXX")
cp exp/jobs/e98_config/model.yaml exp/jobs/e98_config/dataset.json "$NEW/"
cp "$CLONE/olmoearth_run_data/awf/olmoearth_run.yaml" "$NEW/"
cp "$GEOM" "$NEW/prediction_request_geometry.geojson"
E99_NEW=$NEW "$PY" - <<'EOF'
# Refuses unless the snapshot is Ai2's AWF config with exactly exp98's documented changes (e98_config/CHANGES.md), and
# the request geometry has the structure of Ai2's file with 2017 in Ai2's property names. Only counts are printed.
import hashlib, json, os, yaml
from rslearn.config import DatasetConfig
new, awf = os.environ["E99_NEW"], os.path.join(os.environ["E99_DEPLOY"], "olmoearth_projects", "olmoearth_run_data", "awf")
pins = {"dataset.json": "7da20e92b51741f3a2c61137e305bd5a8b59938a107213372099886897751b4b",
        "model.yaml": "08e0432044c5c67c683372c85c632eb31085b6ce8b4dae49b740475e5b37a7ea",
        "olmoearth_run.yaml": "1e1c8f709aadd86f84795723d35e07fc0aa79cef6e076c9a75d1ebec36d705ba",
        "prediction_request_geometry.geojson": "fee3ce01008d26c6f3c239e68be70851035eb157d5da70bd72cb46dbc8623784"}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
for f, h in pins.items():
    assert sha(os.path.join(awf, f)) == h, f"Ai2's {f} is not the pinned one"
assert sha(os.path.join(new, "olmoearth_run.yaml")) == pins["olmoearth_run.yaml"]
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
POLY = {"Polygon": "polygonal", "MultiPolygon": "polygonal"}    # Ai2's one Polygon; exp99's one MultiPolygon of squares
def structure(gj):
    return (sorted(gj), sorted({tuple(sorted(f)) for f in gj["features"]}),
            sorted({(POLY.get(f["geometry"]["type"], f["geometry"]["type"]), tuple(sorted(f["geometry"])))
                    for f in gj["features"]}),
            sorted({tuple(sorted(f["properties"])) for f in gj["features"]}))
ai2 = json.load(open(os.path.join(awf, "prediction_request_geometry.geojson")))
req = json.load(open(os.path.join(new, "prediction_request_geometry.geojson")))
assert structure(req) == structure(ai2), "the request geometry's structure is not Ai2's"
periods = {(f["properties"]["oe_start_time"], f["properties"]["oe_end_time"]) for f in req["features"]}
assert periods == {("2017-01-01T00:00:00Z", "2017-12-31T00:00:00Z")}, periods
# one feature, as Ai2's file: per-square features multiplied the windows (job 1247349, see exp99_transfer.request_squares)
assert len(req["features"]) == 1 and req["features"][0]["geometry"]["type"] == "MultiPolygon", "one MultiPolygon feature"
nsq = len(req["features"][0]["geometry"]["coordinates"])
open(os.path.join(os.environ["E99_DEPLOY"], f"{os.environ['E99_TAG']}_n_squares.txt"), "w").write(f"{nsq}\n")
print(f"snapshot = Ai2's AWF config + output_probs + the p0..p9 float32 output layer; request geometry: one feature, "
      f"{nsq} square(s) dated 2017 in Ai2's structure, sha256 "
      f"{sha(os.path.join(new, 'prediction_request_geometry.geojson'))}")
EOF
if [ "${E99_FRESH:-0}" = 1 ] && [ -d "$RUN" ]; then
  # E99_FRESH=1: start over, removing an earlier run, its config snapshot and its SCL sidecar (scratch only)
  echo "$(date -Is) E99_FRESH=1: removing the earlier run ($(du -sh "$RUN" | cut -f1)), its config and SCL sidecar"
  rm -rf "$RUN" "$CFG" "$SCL"
fi
if [ -d "$CFG" ]; then
  if diff -rq "$CFG" "$NEW" >/dev/null; then rm -rf "$NEW"
  elif [ -d "$RUN" ]; then
    echo "$CFG differs from this commit's configs and $RUN exists: a run must not mix configs. Move both away to start over."
    diff -r "$CFG" "$NEW" --exclude=prediction_request_geometry.geojson || true
    rm -rf "$NEW"; exit 2
  else rm -rf "$CFG"; mv "$NEW" "$CFG"; fi
else mv "$NEW" "$CFG"; fi
( cd "$CFG" && sha256sum model.yaml dataset.json olmoearth_run.yaml prediction_request_geometry.geojson ) > "$CFGSHA"

if [ "${E99_SKIP_BUILD:-0}" != 1 ]; then
  echo "== $(date -Is) build_dataset (olmoearth_run one_stage --stage $ST_BUILD) =="
  LOG=$DEPLOY/logs/${LP}build_dataset-${SLURM_JOB_ID:-none}.log
  if [ -d "$RUN" ] && [ -n "$(ls -A "$RUN" 2>/dev/null)" ]; then
    # olmoearth_run would reuse partitions and windows left by another request geometry
    [ "${E99_FRESH:-0}" = 1 ] || { echo "$RUN holds an earlier run; E99_FRESH=1 removes it first"; exit 2; }
    echo "$(date -Is) removing the earlier run in $RUN ($(du -sh "$RUN" | cut -f1)), E99_FRESH=1"; rm -rf "$RUN"
  fi
  mkdir -p "$RUN"
  cd "$DEPLOY" || exit 1
  NSQ=$(cat "$DEPLOY/${TAG}_n_squares.txt")
  "$PY" -m olmoearth_projects.main olmoearth_run one_stage --config_path "$CFG" --scratch_path "$RUN" \
       --checkpoint_path "$CKPT" --stage "$ST_BUILD" > "$LOG" 2>&1 &
  BPID=$!
  # Watchdog: a plot-window run needs about one window per square (two where a square straddles a cell edge, plus
  # partition-edge duplicates). More than 3 per square means the windowing went wrong (job 1247349 made 30 per square
  # and ran 7 h): stop at once rather than fetch the imagery for them.
  while kill -0 "$BPID" 2>/dev/null; do
    sleep 60
    NW=$(ls -d "$RUN"/dataset_*/windows/*/* 2>/dev/null | wc -l)
    if [ "$NW" -gt $((3 * NSQ)) ]; then
      echo "$(date -Is) $NW windows for $NSQ squares (more than 3 per square): stopping build_dataset"
      kill "$BPID"; sleep 5; kill -9 "$BPID" 2>/dev/null || true; exit 3
    fi
  done
  if ! wait "$BPID"; then
    echo "build_dataset failed; the last lines of $LOG, masked:"; tail -n 60 "$LOG" | redact; exit 1
  fi
  cd "$CODE" || exit 1
  echo "$(date -Is) build_dataset done; $(grep -c -i -E 'error|exception' "$LOG" || true) lines of $LOG mention an error"
fi

echo "== the built dataset, inventoried and checked (counts and shares only; no bounds, no names) =="
# As exp98's prepare, plus three things a plot-window run needs: every plot's point inside a window (the squares' area
# is too small for the area share alone to show a missing window), how many windows each plot falls in, and whether
# every window sits on the lattice of its own size from the CRS origin. On that lattice a pixel's window, and so its
# crop positions and its scenes, depend on the pixel alone, not on which other plots were requested, which the
# preregistration's design argument assumes; it is recorded, not enforced.
"$PY" - <<'EOF'
import collections, glob, json, os, re
import shapely
from rslearn.config import DatasetConfig
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
run, deploy = os.environ["E99_RUN"], os.environ["E99_DEPLOY"]
allow = os.environ.get("E99_ALLOW_INCOMPLETE", "0") == "1"
roots = sorted(os.path.dirname(c) for c in glob.glob(os.path.join(run, "**", "config.json"), recursive=True)
               if os.path.isdir(os.path.join(os.path.dirname(c), "windows")))
assert roots, "no rslearn dataset (config.json beside windows/) under the run directory"
ours = DatasetConfig.model_validate(json.load(open(os.path.join(os.environ["E99_CFG"], "dataset.json")))).layers
inv = {"run": run, "pilot": os.environ["E99_PILOT"] == "1", "experiment": "exp99", "roots": []}
boxes, full = [], []                          # (projection key, pixel bounds); and whether the window has 12 groups
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
        r["windows"] += 1
        groups[meta["group"]] += 1
        r["sizes"][f"{b[2] - b[0]}x{b[3] - b[1]}"] += 1
        r["crs"][str(meta["projection"]["crs"])] += 1
        ldir = os.path.join(wdir, "layers")
        done = [d for d in (os.listdir(ldir) if os.path.isdir(ldir) else [])
                if re.fullmatch(r"sentinel2(\.\d+)?", d) and os.path.exists(os.path.join(ldir, d, "completed"))]
        r["groups_completed"][len(done)] += 1
        boxes.append((json.dumps(meta["projection"], sort_keys=True), tuple(b)))
        full.append(len(done) == 12)
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
    print(f"{r['root_id']}: {r['windows']} windows in {len(groups)} group(s), {r['with_sentinel2']} with sentinel2 "
          f"item groups; completed groups per window {r['groups_completed']}; sizes {r['sizes']}; {len(r['crs'])} CRS; "
          f"{r['bytes'] / 2**30:.1f} GiB; config layers equal to e98's: {r['config_layers_equal_e98']}")

# every plot (each square's centre) against the windows, in each window's own projection
gj = json.load(open(os.path.join(os.environ["E99_CFG"], "prediction_request_geometry.geojson")))
centres = [g.centroid for f in gj["features"] for g in getattr(shapely.geometry.shape(f["geometry"]), "geoms",
                                                                  [shapely.geometry.shape(f["geometry"])])]
by_proj = collections.defaultdict(list)
for k, (p, b) in enumerate(boxes):
    by_proj[p].append(k)
hits = [[] for _ in centres]
for p, ks in by_proj.items():
    proj = Projection.deserialize(json.loads(p))
    for j, c in enumerate(centres):
        q = STGeometry(WGS84_PROJECTION, c, None).to_projection(proj).shp
        for k in ks:
            b = boxes[k][1]
            if b[0] <= q.x < b[2] and b[1] <= q.y < b[3]:
                hits[j].append(k)
n_plots = len(centres)
in_none = sum(1 for h in hits if not h)
in_full = sum(1 for h in hits if any(full[k] for k in h))
lattice = sum(1 for p, b in boxes if b[0] % (b[2] - b[0]) == 0 and b[1] % (b[3] - b[1]) == 0)
# coverage of the squares' area, in the pixel grid of the most common window projection, as exp98's prepare measures
target_key = collections.Counter(p for p, _ in boxes).most_common(1)[0][0]
target = Projection.deserialize(json.loads(target_key))
req = shapely.union_all([shapely.geometry.shape(f["geometry"]) for f in gj["features"]])
req_px = STGeometry(WGS84_PROJECTION, shapely.segmentize(req, 0.00001), None).to_projection(target).shp
cover = []
for p, b in boxes:
    box = shapely.box(*b)
    if p != target_key:
        box = STGeometry(Projection.deserialize(json.loads(p)), shapely.segmentize(box, 64), None).to_projection(target).shp
    cover.append(box)
uncovered = float(req_px.difference(shapely.union_all(cover)).area / req_px.area)
n = sum(r["windows"] for r in inv["roots"])
n12 = sum(full)
ns2 = sum(r["with_sentinel2"] for r in inv["roots"])
inv.update(windows=n, windows_with_12_groups=n12, windows_with_sentinel2=ns2, distinct_window_cells=len(set(boxes)),
           projections=len({p for p, _ in boxes}), request_share_outside_windows=uncovered,
           windows_on_crs_origin_lattice=lattice,
           plots={"n": n_plots, "in_no_window": in_none, "in_one_window": sum(1 for h in hits if len(h) == 1),
                  "in_two_or_more_windows": sum(1 for h in hits if len(h) > 1),
                  "in_a_window_with_12_groups": in_full})
inv["complete"] = bool(n > 0 and n12 == n and ns2 == n and in_none == 0 and uncovered <= 1e-4)
inv["allow_incomplete"] = allow
print(f"{n} windows ({len(set(boxes))} distinct cells, {lattice} on the CRS-origin lattice of their size), {n12} with "
      f"all 12 sentinel2 item groups, {ns2} with sentinel2 items; plots: {n_plots}, in no window {in_none}, in a window "
      f"with 12 groups {in_full}, in two or more windows {inv['plots']['in_two_or_more_windows']}; share of the "
      f"squares' area outside every window {uncovered:.2e}; complete: {inv['complete']}")
json.dump(inv, open(os.environ["E99_INV"], "w"), indent=1)
assert n12 > 0, "no window has all 12 sentinel2 item groups completed"
bad = [r["root_id"] for r in inv["roots"] if not r["config_layers_equal_e98"].get("sentinel2")]
assert not bad, f"config.json's sentinel2 layer is not the e98 (= Ai2's) one in {bad}"
if not all(r["config_layers_equal_e98"].get("output") for r in inv["roots"]):
    print("NOTE: config.json's output layer is not e98's p0..p9; the writer then relies on model.yaml's layer_config")
if not inv["complete"]:
    assert allow, ("the built dataset is incomplete (above): a window without 12 item groups or a plot in no window; "
                   "rerun build_dataset (without E99_SKIP_BUILD), or set E99_ALLOW_INCOMPLETE=1 to go on (the analysis "
                   "counts the plots without input and the preregistration's floor decides)")
    print("WARNING: going on with an incomplete run (E99_ALLOW_INCOMPLETE=1); every later record says so")
EOF

echo "== $(date -Is) the SCL sidecar: the run's sentinel2 item groups, re-read for their SCL band =="
"$PY" - <<'EOF'
import glob, json, os, shutil
run, scl = os.environ["E99_RUN"], os.environ["E99_SCL"]
scl_cfg = json.load(open(os.path.join(os.environ["E99_CODE"], "exp", "jobs", "e98_config", "scl_dataset.json")))
inv = json.load(open(os.environ["E99_INV"]))
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
                continue
            shutil.rmtree(sdir)
        os.makedirs(sdir)
        shutil.copy2(os.path.join(wdir, "metadata.json"), os.path.join(sdir, "metadata.json"))
        with open(sitems, "w") as f:
            json.dump(want, f)
        made += 1
print(f"sidecar windows written this time: {made}")
EOF
k=0
for side in $("$PY" -c 'import json,os; inv=json.load(open(os.environ["E99_INV"])); print(" ".join(os.path.join(os.environ["E99_SCL"], r["root"]) for r in inv["roots"]))'); do
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
run, scl = os.environ["E99_RUN"], os.environ["E99_SCL"]
inv = json.load(open(os.environ["E99_INV"]))
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
            assert len(refl) == 1, f"{len(refl)} reflectance rasters in an item group, not 1"
            with rasterio.open(refl[0]) as s:
                b = s.read()
            assert b.shape[1:] == shape_scl, (b.shape, shape_scl)
            codes += np.bincount(a.ravel(), minlength=256)
            cov_s, cov_r = a != 0, (b != 0).any(axis=0)
            agree = float((cov_s == cov_r).mean())
            b02 = b[1].astype(np.float64)
            cloud, clear = np.isin(a, (8, 9)), np.isin(a, (4, 5))
            check["sampled"].append({"window_index": int(i), "group": gi, "coverage_agreement": agree,
                                     "scl_covered": float(cov_s.mean()),
                                     "b02_cloud_over_clear": (float(b02[cloud].mean() / b02[clear].mean())
                                                              if cloud.sum() > 100 and clear.sum() > 100 else None)})
check["codes"] = {str(k): int(v) for k, v in enumerate(codes) if v}
json.dump(check, open(os.environ["E99_SCLCHK"], "w"), indent=1)
agrees = [s["coverage_agreement"] for s in check["sampled"]]
print(f"{check['windows']} sidecar windows; {check['mismatched_group_counts']} with another group count than the run; "
      f"{len(agrees)} sampled groups, coverage agreement min {min(agrees) if agrees else None}; SCL codes {check['codes']}")
assert check["windows"] > 0 and check["mismatched_group_counts"] == 0
assert set(int(k) for k in check["codes"]) <= set(range(12)), "values outside SCL's 0..11"
assert agrees and min(agrees) >= 0.98, "SCL coverage does not match the reflectance coverage of the same group"
EOF
"$PY" - <<'EOF'
import json, os
inv = json.load(open(os.environ["E99_INV"]))
inv["scl_check"] = json.load(open(os.environ["E99_SCLCHK"]))
inv["config_sha256"] = dict(reversed(l.split()) for l in open(os.environ["E99_CFGSHA"]))
inv["job"] = os.environ.get("SLURM_JOB_ID")
json.dump(inv, open(os.environ["E99_INV"], "w"), indent=1)
EOF
# the tracked copy: without the real root and group names (scratch keeps them)
"$PY" - "$INV" "$OUT/e99_prepare_inventory.json" <<'EOF'
import json, sys
inv = json.load(open(sys.argv[1]))
inv.pop("run", None)
for r in inv["roots"]:
    r.pop("root", None)
    r.pop("group_names", None)
json.dump(inv, open(sys.argv[2], "w"), indent=1)
EOF
echo "== $(date -Is) done: $INV, $OUT/e99_prepare_inventory.json; $(du -sh "$RUN" | cut -f1) in $RUN =="
