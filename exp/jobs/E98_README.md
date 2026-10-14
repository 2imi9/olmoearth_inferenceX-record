# exp98 jobs: Ai2's FT-AWF model through olmoearth_run, with probabilities and SCL

What the four jobs do: run Ai2's fine-tuned AWF land-cover model (`allenai/OlmoEarth-v1-FT-AWF-Base`) through
`olmoearth_run` over the AWF project's own request geometry and dates (2023-01-01 to 2023-12-31), as a deployment
would. Two things differ: each window's output layer holds the 10-channel softmax as float32 bands `p0` to `p9`
instead of the argmax id, and a sidecar dataset reads the Sentinel-2 scene classification (SCL) of the same scenes,
for the package's condition layer. The result is a compact rslearn dataset in the home directory that
`oe-inferencex from-olmoearth` reads.

No coordinate goes into a tracked file or a job log. The request geometry is taken from the pinned `olmoearth_projects`
clone and checked by its sha256; it is never copied into this repository or the home data directory. Dataset roots and
window groups, which olmoearth_run's partitioner names in a way that is not public (perhaps after a 1-degree cell's
corner), appear in tracked files and logs only as `root_<i>` and `group_<j>`; their real names stay in
`deploy/prepare_inventory.json` on scratch. Upstream logs go to `deploy/logs/` on scratch and may hold window bounds;
what the jobs echo from them is masked. The home data directory holds what the reader needs: each window's
`metadata.json` (its pixel bounds) and layers, not `items.json` (the scenes' WGS84 footprints), which stays on scratch.

## Files

| File | Partition | What it does |
|---|---|---|
| `e98_config/model.yaml` | | Ai2's AWF `model.yaml` plus `output_probs: true` and the writer's `layer_config` (p0..p9, float32) |
| `e98_config/dataset.json` | | Ai2's AWF `dataset.json` with the output layer as p0..p9 float32 |
| `e98_config/scl_dataset.json` | | the SCL sidecar's dataset config |
| `e98_config/CHANGES.md` | | every change, why, and the source lines it rests on |
| `e98_env.sh` | cpu, 8 CPUs, 32 GB, 2 h | Python 3.11 venv under `deploy/`, `olmoearth_projects` at `f3c9b0c8`, `olmoearth-runner` 0.1.14, the checkpoint, smokes, `versions.json` |
| `e98_prepare.sh` | cpu, 64 CPUs, 256 GB, 12 h | config snapshot and its check, `build_dataset`, inventory, SCL sidecar and its check |
| `e98_predict.sh` | rtx-batch, 1 GPU, 16 CPUs, 128 GB, 24 h | `run_inference`, then a check of every output layer; `E98_STAGE=post` runs `postprocess` and `combine` on cpu |
| `e98_collect.sh` | cpu, 8 CPUs, 32 GB, 3 h | output and SCL layers to `~/olmoearth_inferenceX/data/exp98/awf_run/`, with a sha256 manifest |
| `e98_read.sh` | cpu, 8 CPUs, 64 GB, 4 h | `oe-inferencex from-olmoearth` on every root, in parts of at most 3 GiB of grid whose windows do not overlap, each part to `data/exp98/scores/r<root>_p<part>/`; needs the reader merged into main |
| `e98_burned.sh` | cpu, 8 CPUs, 64 GB, 3 h | MODIS burned area (MCD64A1 v061, 2023) on each part's grid, `burned_<EPSG>.tif` beside its scores, for Part H and `assess --condition` |
| `e98.sh` | cpu, 8 CPUs, 32 GB, 2 h | exp98 itself (inventory, or the run once the page is frozen), reading every part directory as one map; for Part I it fetches the East Africa TimeSync sample and Natural Earth's countries into `data/breadth/` (checked against their pins in `exp/timesync_awf_crosswalk.py`) and reads Ai2's request geometry from the pinned clone; without them Part I is skipped with a note |

Scratch layout, all under `/scratch/qi_zim_neu/olmoearth_inferenceX/deploy/`: `python/`, `venv/`,
`olmoearth_projects/`, `awf_config/` (the config directory every stage reads, snapshotted by prepare), `awf_run/`
(olmoearth_run's `--scratch_path`), `awf_scl/` (the sidecar), `logs/`, `introspect/`, and the JSON records.

## Pilot first

Before the long fetch, the same four stages run end to end on a small sub-area with `E98_PILOT=1`. prepare checks Ai2's
geometry's sha256 as always, then replaces the snapshot's `prediction_request_geometry.geojson` with Ai2's geometry
clipped to a square of 0.05 degrees around its `representative_point()` (a point inside it): the same FeatureCollection,
feature properties and dates, about 30 km², 1 to 4 windows. Only the clipped file's sha256 is printed and recorded
(`awf_pilot_config.sha256`). Every path is separate from a full run's: `deploy/awf_pilot_config/`, `awf_pilot_run/`,
`awf_pilot_scl/`, `pilot_trainer_data/`, `logs/pilot_*`, `prepare_inventory_pilot.json`, `scl_check_pilot.json`,
`predict_check_pilot.json`; `data/exp98/pilot/{awf_run,scores,assess}/`; records in `exp/out/exp98/pilot/`. The
inventory says `"pilot": true`, and each later stage refuses records whose `pilot` differs from its own `E98_PILOT`.
`E98_PILOT` must be 0 or 1; any other value is refused, so a mistyped pilot never starts the full fetch.

The env built before the rslearn `get_item_by_name` fix (in `e98_env.sh`) lacks it, so the pilot starts with an env
rerun (it reuses the venv and adds the fix); prepare also refuses a venv without the fix in seconds. Commit and push
the `e98_env.sh` change first. The pilot prepare keeps the header's 64 CPUs and 256 GB (olmoearth_run starts one
build worker per node CPU, 128, whatever the job asks for), so a pass also tests the full run's resources; only the
time limit is shorter.

```bash
SHA=$(git rev-parse HEAD)
ENV=$(ssh aicr "E98_SHA=$SHA sbatch --parsable" < exp/jobs/e98_env.sh)
PREP=$(ssh aicr "E98_PILOT=1 E98_SHA=$SHA sbatch --parsable -t 02:00:00 --dependency=afterok:$ENV" < exp/jobs/e98_prepare.sh)
PRED=$(ssh aicr "E98_PILOT=1 E98_SHA=$SHA sbatch --parsable -p rtx-batch --gpus=1 -t 02:00:00 --dependency=afterok:$PREP" < exp/jobs/e98_predict.sh)
COLL=$(ssh aicr "E98_PILOT=1 E98_SHA=$SHA sbatch --parsable -t 01:00:00 --dependency=afterok:$PRED" < exp/jobs/e98_collect.sh)
READ=$(ssh aicr "E98_PILOT=1 E98_SHA=$SHA sbatch --parsable -t 01:00:00 --dependency=afterok:$COLL" < exp/jobs/e98_read.sh)
BURN=$(ssh aicr "E98_PILOT=1 E98_SHA=$SHA sbatch --parsable -t 01:00:00 --dependency=afterok:$READ" < exp/jobs/e98_burned.sh)
```

What to check (`slurm/e98prep-<job>.out` and so on):

- **prepare**: `pilot geometry: 1 of 1 feature(s)`, a share of Ai2's area near 1.5e-03, and its sha256; then 1 to 4
  windows, all with 12 item groups and sentinel2 items, share outside every window at most 1e-4, `complete: True`; the
  SCL line with 0 mismatched group counts, coverage agreement at least 0.98, codes within 0 to 11; `done`. The
  build_dataset time per window, times about 224 windows, is the first measured estimate of the full prepare.
- **predict**: `"ok": true`, `"pilot": true`, `missing_output` 0; each sampled window 10 float32 bands summing to 1
  within 1e-3, `argmax_is_9_share` near 0. Its run_inference time per window, likewise, estimates the full predict.
- **collect**: `manifest_verified` true in `exp/out/exp98/pilot/e98_collect.json`, `reader_reads_whole_root` true.
- **read**: one line per part and grid with `bands` 10, `max_abs_sum_minus_1` at most 1e-3, `covered_share` (near 1
  inside the windows' grid), `condition_codes` (pixel counts per code; -1 where no window reads), `assess_exit` 0 and
  `assess_windows_ranked`; then `windows read <n>; ok True`. The same is in `exp/out/exp98/pilot/e98_pilot_checks.json`.
- **burned**: the names line (`--condition-names 0=mcd64a1:no-burn-2023:sep-not-read 1=mcd64a1:burned-2023`), months
  missing `['2023-09']` only, one line per part with the share of covered pixels per code and `QA agrees True`, then
  `assess_exit` 0 with windows ranked and windows per condition (`exp/out/exp98/pilot/e98_burned_assess.json`).

If all four hold, run the full chain below (without `E98_PILOT`); it shares nothing with the pilot but the env, so the
pilot's `deploy/awf_pilot_*` and `data/exp98/pilot/` can stay or be removed.

## Submit

Commit and push first: every job sets the home checkout to `origin/main`, or to `E98_SHA` when it is given, and
prepare reads `exp/jobs/e98_config/` from it. With `E98_SHA` (a full sha on origin) each stage checks out that commit,
detached, whatever main has become since, so pushes while the chain waits change nothing it runs. From the Mac (the
scripts travel on stdin; the login node runs `sbatch` only):

```bash
SHA=$(git rev-parse HEAD)          # optional pin: every stage checks out this commit
ENV=$(ssh aicr "E98_SHA=$SHA sbatch --parsable" < exp/jobs/e98_env.sh)
```

Read the env log before going on (it is short, and it is where the unknowns below first show): the two smoke lines,
the stage values, and `deploy/introspect/olmoearth_run_grep.txt` (which env vars and paths olmoearth_run uses).
Then the chain, each stage only if the previous one exited 0:

```bash
PREP=$(ssh aicr "E98_SHA=$SHA sbatch --parsable --dependency=afterok:$ENV" < exp/jobs/e98_prepare.sh)
PRED=$(ssh aicr "E98_SHA=$SHA sbatch --parsable -p rtx-batch --gpus=1 -t 24:00:00 --dependency=afterok:$PREP" < exp/jobs/e98_predict.sh)
COLL=$(ssh aicr "E98_SHA=$SHA sbatch --parsable --dependency=afterok:$PRED" < exp/jobs/e98_collect.sh)
# after the from-olmoearth branch is merged into main (e98_read.sh refuses otherwise):
READ=$(ssh aicr "sbatch --parsable --dependency=afterok:$COLL" < exp/jobs/e98_read.sh)
BURN=$(ssh aicr "E98_SHA=$SHA sbatch --parsable --dependency=afterok:$READ" < exp/jobs/e98_burned.sh)
INV=$(ssh aicr "E98_MODE=inv E98_DEPLOY=auto sbatch --parsable --dependency=afterok:$BURN" < exp/jobs/e98.sh)
# optional, olmoearth_run's own mosaics (exp98 does not read them), after collect and never beside it:
POST=$(ssh aicr "E98_SHA=$SHA E98_STAGE=post sbatch --parsable -p cpu -t 04:00:00 --dependency=afterok:$COLL" < exp/jobs/e98_predict.sh)
```

Every stage starts by resetting the one home checkout, so no two jobs that do so may run at once (one can fail on
`.git/index.lock`): no other job of this repository (`e98.sh`, any `eNN.sh`) may overlap the chain; chain it on `$COLL`
(or `$POST`). Post also must not run while collect hashes and copies the windows (what `postprocess` does to them is
not public), so it is chained on collect and refuses until `e98_collect.json` says `manifest_verified`.

The alternative GPU is one B200: `-p b200-batch` in place of `-p rtx-batch`. Both allow 24 hours. The env job is
idempotent (it reuses the venv unless `E98_REBUILD=1`). A prepare that failed after `build_dataset` can be resubmitted
with `E98_SKIP_BUILD=1`; its completeness check still runs, so a `build_dataset` that timed out or partly failed is
refused there and needs a resubmission without `E98_SKIP_BUILD` (whether olmoearth_run resumes is unknown, below).
`E98_ALLOW_INCOMPLETE=1` at prepare lets a partial area go on; every later record then says `area_complete: false`.
Collect refuses to overwrite an earlier collection unless `E98_RECOLLECT=1`.

## Burned area (after read)

`e98_burned.sh` runs `exp/exp98_burned.py` on every part directory `e98_read.sh` wrote: for each grid it searches
Planetary Computer's STAC API for MODIS MCD64A1 v061 (`modis-64A1-061`, monthly burned area, 500 m) in 2023, mosaics
the tiles over the part, and writes `burned_<EPSG>.tif` beside `scores_<EPSG>.tif` (int32, nearest neighbour onto the
same grid: 1 burned in a month read, 0 not, -1 unrecorded) with `burned_conditions.json` (the exact
`--condition-names`). Counts only go to `exp/out/exp98[/pilot]/e98_burned.json`; then `assess --condition` runs on the
smallest grid as a check that the layer plugs in (`e98_burned_assess.json`). Planetary Computer holds no September 2023
item of the collection (checked 8 October 2026; NASA's CMR lists the granules), so the job allows that month missing
(`E98_BURN_ALLOW_MISSING`, default `2023-09`) and refuses any other; code 0's name records it. `E98_REBURN=1` replaces
an earlier layer; a reread (`E98_REREAD=1`) removes it with the scores, so rerun this stage after one. It resets the
home checkout like every stage, so chain it between read and `e98.sh`, never beside another job:

```bash
BURN=$(ssh aicr "E98_SHA=$SHA sbatch --parsable --dependency=afterok:$READ" < exp/jobs/e98_burned.sh)
```

Expected minutes: twelve months of one or two tiles, read only where they meet a part, and an exact warp of each part.
Part H (`docs/plan/awf_deployment.md`) reads the layer in the run; without it Part H is skipped with a note.

## Durations and volume (estimates, not measured)

| Stage | Expected | Limit | Why |
|---|---|---|---|
| env | 15 to 45 min | 2 h | the install (torch and the CUDA libraries are about 3 GB from the uv cache or the index) and the 1.04 GB checkpoint (already in `HF_HOME` if exp89's copy survived the purge) |
| prepare | 1 to 6 h | 12 h | about 40 to 65 GB of Sentinel-2 windows read from Planetary Computer, no subscription key |
| predict | 1 to 6 h | 24 h | about 1.6 M crops of 16 px at batch size 4; data loading dominates |
| post | 0.5 to 2 h | 4 h | mosaics of 10 float32 bands |
| collect | 10 to 40 min | 3 h | about 10 GB copied and hashed, plus `du` of the home directory |

The request geometry covers about 19,800 km² (its polygon area, computed in one UTM zone), about 224 cells of
1024 x 1024 pixels at 10 m, so about 220 to 280 windows. On scratch: about 40 to 65 GB of input imagery (12 item groups
x 12 bands x 1024² x uint16 per window, LZW), about 9 GB of probabilities (10 x 1024² x float32 per window), well
under 1 GB of SCL. In the home directory (100 GiB quota): about 10 GB, the probabilities and SCL only; collect refuses
if home use plus the copy would pass 95 GiB. The 12-band imagery stays on scratch and is purged after 30 days unless
`E98_KEEP_IMAGERY=1` (which the quota check will usually refuse). Anything that reads the deployment's own input bands
(the exp98 analysis job's `E98_DEPLOY`, which is `deploy/awf_run/<root>`, the real name of `root_<i>` in
`deploy/prepare_inventory.json`) has to run within those 30 days.

## Outputs

- `exp/out/exp98/e98_versions.json`: Python, package versions, git shas, the uv overrides, the checkpoint's revision,
  bytes and sha256, the e98 configs' sha256.
- `exp/out/exp98/e98_prepare_inventory.json`: per dataset root (`root_<i>`), windows, windows per group (`group_<j>`),
  completed item groups per window, items per group, window sizes, CRS count, bytes, whether `config.json`'s layers
  equal e98's; over the run, `windows`, `windows_with_12_groups`, `windows_with_sentinel2`, `distinct_window_cells`,
  `expected_cells_on_main_grid` (informational), `request_share_outside_windows`, `complete`, `allow_incomplete`; the
  SCL check; the config sha256. No real root or group names.
- `exp/out/exp98/e98_predict_check.json`: `area_complete`, windows, windows without input, with input and with output,
  and per sampled window the band count, dtype, range, largest deviation of the band sum from 1, share of pixels whose
  argmax is the untrained class 9.
- `exp/out/exp98/e98_collect.json`: `area_complete`, per root the windows copied and the grid the reader would hold
  (`reader_grids`, `reader_reads_whole_root`), files, bytes, the manifest's sha256, `manifest_verified`.
- `data/exp98/awf_run/` in the home checkout (gitignored): `root_<i>/config.json`, `root_<i>/windows/<group>/<name>/`
  with `metadata.json`, `layers/output/p0_p1_p2_p3_p4_p5_p6_p7_p8_p9/geotiff.tif` and
  `layers/sentinel2_scl[.N]/SCL/geotiff.tif` (each with its `completed` marker), `records/`, `MANIFEST.sha256`.

Next: `e98_read.sh` reads every root in parts with `oe-inferencex from-olmoearth --window ...`, each part to
`data/exp98/scores/r<root>_p<part>/`, with `parts.json` (which windows each part holds) beside them, and counts only in
`exp/out/exp98/e98_read.json`. `e98.sh` then reads all the part directories as one map (unknown 3). A dry run
of the four jobs' Python on a synthetic three-window tree, read back by the reader on the from-olmoearth branch
(22504f8), gave 10 probability bands and an SCL condition layer of 12 timesteps.

## How to tell each stage worked

- **env**: the log ends with `done`; it printed `task: output_probs gives (10, 16, 16)` and `writer: 10 float32 bands at
  layers/output/p0_p1_p2_p3_p4_p5_p6_p7_p8_p9/geotiff.tif, equal to the input`, `versions as pinned; torch 2.7.1+cu128
  CUDA 12.8`, the checkpoint's sha256 as pinned, and the stage values. `deploy/pip_check.txt` lists the nvidia-*
  conflicts the overrides create on purpose; anything else in it deserves a look.
- **prepare**: the log ends with `done`. In the inventory: `complete` true (enforced unless `E98_ALLOW_INCOMPLETE=1`):
  every window at `groups_completed` "12", `windows_with_sentinel2` equal to `windows`, and
  `request_share_outside_windows` at most 1e-4 (the share of the request geometry's area no window covers, measured
  in the pixel grid of the most common window projection); `distinct_window_cells` near
  `expected_cells_on_main_grid`; `config_layers_equal_e98` true for `sentinel2` (enforced) and for `output` (otherwise a NOTE: the writer
  then relies on `layer_config` alone). In `scl_check`: `mismatched_group_counts` 0, `coverage_agreement` at least 0.98 on
  every sampled group (enforced), codes within 0 to 11 (enforced), and `b02_cloud_over_clear` above 1 where it is
  computed (clouds are brighter in B02 than vegetation; a value near 1 everywhere would suggest SCL from other scenes).
- **predict**: `predict_check.json` has `ok: true`, `area_complete: true`, `missing_output` 0,
  `windows_without_input` 0, `bad_bandset` 0; every sampled window has 10
  float32 bands within [0, 1] summing to 1 within 1e-3. `argmax_is_9_share` should be near 0: class 9 is never a label.
- **collect**: `e98_collect.json` has `manifest_verified: true`. Recheck at any time with
  `cd ~/olmoearth_inferenceX/data/exp98/awf_run && sha256sum -c --quiet MANIFEST.sha256`.

## Where this departs from a deployment, and why

1. **olmoearth-runner 0.1.14 (rslearn 0.0.27), not the lock's 0.1.12 (rslearn 0.0.23).** Ai2's README installs with
   `uv sync` from `uv.lock`, which resolves `olmoearth-runner` 0.1.12, and 0.1.12 pins `rslearn==0.0.23`, which has no
   `output_probs`. 0.1.14 pins `rslearn==0.0.27`, which has it (`rslearn/train/tasks/segmentation.py:44`). Between the two,
   the OlmoEarth wrapper changed only a warning (`rslearn/models/olmoearth_pretrain/model.py`), and the PER_PERIOD_MOSAIC
   order keeps its old default (`per_period_mosaic_reverse_time_order` True, `rslearn/config/dataset.py:351`).
2. **torch 2.7.1+cu128, not PyPI's torch 2.7.1.** The runner pins the CUDA 12.6 libraries (`nvidia-*-cu12` 12.6.x in its
   requires_dist), and the CUDA 12.6 build of torch 2.7 carries no kernels for the Blackwell GPUs of `rtx-batch` and
   `b200-batch`. The same torch release is installed from the CUDA 12.8 index, with the nvidia-* versions this
   repository's `uv.lock` resolves for it. predict checks the GPU's architecture is in the build before it runs.
3. **The SCL comes from a sidecar dataset, not from a band set on the model's `sentinel2` layer** (`e98_config/CHANGES.md`):
   rslearn 0.0.27's Planetary Computer `Sentinel2` source has no SCL asset (`planetary_computer.py:312-326`), and a band
   set on that layer would be resampled bilinearly with it (`rslearn/config/dataset.py:490-493`,
   `rslearn/dataset/materialize.py:481`). The sidecar materializes the run's own item groups, group for group, with
   nearest resampling and no harmonization, so each SCL raster describes the scene the model read.
4. **Environment variables model.yaml leaves to the caller** are set so none is substituted empty (rslearn replaces an
   unset `${VAR}` with "", `rslearn/template_params.py:19-24`): `NUM_WORKERS` (the job's CPUs),
   `PREDICTION_OUTPUT_LAYER=output`, `TRAINER_DATA_PATH`, `EXTRA_FILES_PATH`, and W&B disabled (`WANDB_MODE=disabled`).
   `DATASET_PATH` is left to olmoearth_run.

## Unknowns (not verifiable from public source)

1. **olmoearth_run's internals.** Its source is not on GitHub, only the wheel on PyPI, which was not downloaded to the Mac.
   Unknown: how `build_dataset` runs prepare, ingest and materialize (workers, retries, whether a resubmission resumes);
   where it puts the rslearn dataset(s) under `--scratch_path` (one, or one per 1-degree partition; the jobs discover
   every `config.json` beside a `windows/`); the window group names; whether it writes `config.json` from our
   `dataset.json` (prepare reports it); which env vars it sets; whether `run_inference` uses model.yaml's writer or its
   own (both the writer's `layer_config` and the dataset's output layer say p0..p9, so either way); whether
   `postprocess` and `combine` (CombineGeotiff, nodata 9) accept a 10-band float output. `e98_env.sh` greps the installed
   package into `deploy/introspect/` so these can be read on the cluster before prepare.
2. **Duplicate windows at partition edges.** If olmoearth_run makes a window per partition where a 1024-pixel cell
   crosses a 1-degree line, two windows overlap. `from-olmoearth` refuses overlapping windows whose probabilities differ
   (`oe_inferencex/olmoearth.py` on the from-olmoearth branch). `e98_read.sh` never puts two overlapping windows in
   one part; exp98 counts a map pixel covered in two parts once, gives a label point in two parts to the one that
   covers it, and counts the points covered in two parts whose probabilities differ.
3. **The reader's memory limit.** `from-olmoearth` holds one CRS's grid in memory up to 4 GiB (`MAX_GRID_BYTES`,
   `oe_inferencex/olmoearth.py:55` at 58d0a34) at 44 bytes per pixel for the scores (10 float32 bands and an int32
   owner, :371). This area at 10 m is about 15,000 x 13,000 pixels, about 8 GiB: the reader would refuse the run as one
   grid. `e98_read.sh` uses the reader's own remedy, `--window` parts each to its own `--out` (at most 3 GiB of grid
   each, `E98_PART_GIB`), and exp98 reads every part directory as one map (`--scores DIR [DIR ...]`; the smoke reads
   two overlapping parts). The reader is on the from-olmoearth branch: `e98_read.sh` refuses until it is on main.
4. **one_stage's `--stage` spelling.** jsonargparse may take the enum's names (`BUILD_DATASET`) or its values
   (`build_dataset`); env reads which from `--help` and the other jobs use that.
5. **The writer's `layer_config` in YAML.** Parsed by jsonargparse 4.35.0 into rslearn's pydantic `LayerConfig`; env's
   smoke parses and runs it before any real job. Ai2's own configs pass it the same way (enum names).
6. **Identity with Ai2's deployed map.** rslearn 0.0.27 against 0.0.23 and torch's CUDA 12.8 build against 12.6 may shift
   probabilities in the last digits and so flip an argmax at near-ties; the scenes are chosen at run time from today's
   Planetary Computer catalogue, which may differ from the one Ai2's run saw. Nothing public gives Ai2's per-pixel output
   to compare with.
7. **The SCL asset on Planetary Computer.** The sidecar reads the STAC asset key `SCL` of `sentinel-2-l2a`; rslearn
   v0.1.17 lists it (`planetary_computer.py:323`), 0.0.27 does not use it. If an item lacks it, materialize fails and
   prepare stops.
8. **The OlmoEarth-v1-Base download.** Building the encoder fetches `allenai/OlmoEarth-v1-Base` at its latest Hub
   revision (`olmoearth_pretrain.model_loader.load_model_from_id`, unpinned in 0.0.2); the checkpoint's weights then
   replace it, if Lightning loads the checkpoint strictly (its default). env records the checkpoint's decoder shapes.
9. **Cluster limits.** The cpu partition's maximum wall time and node memory are not documented in the repository;
   prepare asks for 64 CPUs, 256 GB and 12 hours. Home use is measured with `du`, not a quota tool.
10. **Volumes and durations** above are estimates from the polygon's area and the file formats, not measurements.
