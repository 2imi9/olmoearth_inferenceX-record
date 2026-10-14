# exp99 jobs: Ai2's FT-AWF deployment configuration on 2017 imagery, over the windows holding TimeSync plots

What the jobs do: run Ai2's fine-tuned AWF land-cover model through `olmoearth_run` with exactly exp98's configuration
(`exp/jobs/e98_config/`: `output_probs`, the ten float32 bands p0 to p9, the SCL sidecar), in exp98's deployment
environment, but over a request geometry of one small square around each East Africa TimeSync plot the area rule of
`docs/plan/awf_transfer.md` selects (309 plots within 100 km of Ai2's AWF request geometry), dated 2017. olmoearth_run
then predicts the 1,024 x 1,024-pixel windows that hold a plot. The result is read by `oe-inferencex from-olmoearth`, one
window per part, and graded by `exp/exp99_transfer.py` once the page is frozen.

No coordinate goes into a tracked file or a job log. The request geometry holds the plots' positions: `e99_select.sh`
writes it on scratch only (`deploy/e99_select/`), records only its sha256, and `exp/exp99_transfer.py` refuses to write
it inside any git checkout. Dataset roots and window groups appear in tracked files only as `root_<i>` and `group_<j>`,
as in exp98. Upstream logs stay in `deploy/logs/`; what the jobs echo from them is masked.

## Why exp99 runs from its own clone

exp98's jobs reset the one home checkout at their start, and its README forbids two such jobs at once: they can collide
on `.git/index.lock`, and a reset under a running job changes the files it reads. exp99's chain runs while exp98's is
still going, so every exp99 stage checks out its code in its own clone, `/scratch/qi_zim_neu/olmoearth_inferenceX/e99_code`
(cloned from the home checkout's `origin` on first use, then `git fetch` and a detached checkout of `origin/main` or
`E99_SHA`, exactly as exp98's stages pin `E98_SHA`). It never touches the home checkout's working tree. Records and data
still land in the home checkout, in paths no tracked file holds: `exp/out/exp99/`, `exp/out/exp99_*`, `data/exp99/`,
`data/breadth/`. Two exp99 stages must not run at once (they share the clone): chain them, as below.

## Files

| File | Partition | What it does |
|---|---|---|
| `e99_select.sh` | cpu, 4 CPUs, 16 GB, 1 h | fetches the TimeSync sample (GitHub, pinned sha256) and Natural Earth's countries (pinned git blob) into `data/breadth/`; the area rule (refuses counts other than D = 100 km, 309 plots, 219 and 90, 47 inside); the region's area by country; the request geometries (full and pilot) on scratch; `exp/out/exp99/e99_select.json` |
| `e99_prepare.sh` | cpu, **48 CPUs**, 192 GB, 12 h | config snapshot (exp98's model.yaml and dataset.json, Ai2's olmoearth_run.yaml, the request geometry; each checked), `build_dataset`, inventory (12 periods per window, every plot in a window, windows on the CRS-origin lattice), SCL sidecar and its check |
| `e99_predict.sh` | GPU, given at submission (rtx-batch or b200-batch 24 h; b200-devel 4 h), 16 CPUs, 128 GB | `run_inference`, then a check of every output layer |
| `e99_collect.sh` | cpu, 8 CPUs, 32 GB, 3 h | output and SCL layers to `~/olmoearth_inferenceX/data/exp99/awf2017_run/`, with a sha256 manifest |
| `e99_read.sh` | cpu, 8 CPUs, 32 GB, 3 h | `oe-inferencex from-olmoearth --conditions --inputs sentinel2_scl`, one window per part, to `data/exp99/scores/r<root>_w<k>/` |
| `e99.sh` | cpu, 8 CPUs, 32 GB, 3 h | exp99 itself: `E99_MODE=inv` (counts) before the page is frozen, `E99_MODE=run` after |

The cpu partition caps each user's CPUs (`QOSMaxCpuPerUserLimit`); a 64-CPU job reaches the cap and blocks every other
cpu job of ours, so exp99's prepare asks for 48, leaving room for exp98's 8-CPU stages beside it. While exp98's 64-CPU
prepare runs, every exp99 cpu job waits in the queue; the dependencies keep the order.

Scratch layout, under `/scratch/qi_zim_neu/olmoearth_inferenceX/`: `e99_code/` (the clone and its `.venv`),
`deploy/e99_select/` (the request geometries and their sha256), `deploy/awf2017_config/`, `deploy/awf2017_run/`,
`deploy/awf2017_scl/`, `deploy/prepare_inventory_e99.json`, `deploy/scl_check_e99.json`, `deploy/predict_check_e99.json`,
`deploy/logs/e99_*`; the pilot's with `awf2017_pilot_` and `_e99_pilot`. Shared with exp98, read only: `deploy/venv/`,
`deploy/olmoearth_projects/` (Ai2's request geometry and configs, checked by sha256), `deploy/checkpoint_path.txt`,
`deploy/stage_spelling.txt`, `deploy/versions.json`.

## Submit

Commit and push first (every stage checks out `origin/main`, or `E99_SHA`). From the Mac; the login node runs `sbatch`
only. The pilot (4 plots) first, then the full run:

```bash
SHA=$(git rev-parse HEAD)
SEL=$(ssh aicr "E99_SHA=$SHA sbatch --parsable" < exp/jobs/e99_select.sh)
# the pilot: 4 plots, every stage on pilot paths
PP=$(ssh aicr "E99_PILOT=1 E99_SHA=$SHA sbatch --parsable -t 02:00:00 --dependency=afterok:$SEL" < exp/jobs/e99_prepare.sh)
PD=$(ssh aicr "E99_PILOT=1 E99_SHA=$SHA sbatch --parsable -p b200-devel --gpus=1 -t 02:00:00 --dependency=afterok:$PP" < exp/jobs/e99_predict.sh)
PC=$(ssh aicr "E99_PILOT=1 E99_SHA=$SHA sbatch --parsable -t 01:00:00 --dependency=afterok:$PD" < exp/jobs/e99_collect.sh)
PR=$(ssh aicr "E99_PILOT=1 E99_SHA=$SHA sbatch --parsable -t 01:00:00 --dependency=afterok:$PC" < exp/jobs/e99_read.sh)
PI=$(ssh aicr "E99_PILOT=1 E99_MODE=inv E99_SHA=$SHA sbatch --parsable --dependency=afterok:$PR" < exp/jobs/e99.sh)
# the full run, after the pilot's logs are read (or chained on $PI)
PREP=$(ssh aicr "E99_SHA=$SHA sbatch --parsable --dependency=afterok:$PI" < exp/jobs/e99_prepare.sh)
PRED=$(ssh aicr "E99_SHA=$SHA sbatch --parsable -p rtx-batch --gpus=1 -t 24:00:00 --dependency=afterok:$PREP" < exp/jobs/e99_predict.sh)
COLL=$(ssh aicr "E99_SHA=$SHA sbatch --parsable --dependency=afterok:$PRED" < exp/jobs/e99_collect.sh)
READ=$(ssh aicr "E99_SHA=$SHA sbatch --parsable --dependency=afterok:$COLL" < exp/jobs/e99_read.sh)
INV=$(ssh aicr "E99_MODE=inv E99_SHA=$SHA sbatch --parsable --dependency=afterok:$READ" < exp/jobs/e99.sh)
# once the owner has confirmed the thresholds and the page says frozen (commit, push):
RUN=$(ssh aicr 'E99_MODE=run sbatch --parsable' < exp/jobs/e99.sh)
```

`-p b200-batch` replaces `-p rtx-batch` for one B200. A prepare that failed after `build_dataset` can be resubmitted
with `E99_SKIP_BUILD=1`. If some window lacks one of the 12 periods (Sentinel-2B joined only in mid-2017), prepare
refuses with `complete: false`; resubmit with `E99_SKIP_BUILD=1 E99_ALLOW_INCOMPLETE=1` and the analysis counts the
plots without input, against the page's floor of 250. `E99_RESELECT=1`, `E99_RECOLLECT=1` and `E99_REREAD=1` replace an
earlier select, collection or read. `E99_MODE=run` reads exp98's 2023 parts for Part F only once exp98's page is frozen
(`E99_EXP98=none` skips them).

## What to check

- **select:** two `verified` lines (the sample's sha256 `5309a982...`, Natural Earth's git blob `5ebc66e2...`); then
  `{"d_km": 100, "n_selected": 309, "by_country": {"kenya": 219, "tanzania": 90}, "inside_request_geometry":
  {"kenya": 33, "tanzania": 14}}`, the weights (about 0.60 and 0.40), `share_outside_both_countries` near 0, and the two
  geometries' feature counts (309 and 4) and sha256.
- **prepare:** the snapshot line (309 or 4 squares dated 2017 in Ai2's structure); then windows, all with 12 item groups
  and sentinel2 items, plots in no window 0, windows on the CRS-origin lattice (expected: all), `complete: True`; the
  SCL line as in exp98 (0 mismatched group counts, coverage agreement at least 0.98, codes within 0 to 11); `done`.
  The pilot's build_dataset time per window times about 280 estimates the full prepare.
- **predict:** `"ok": true`, `missing_output` 0; each sampled window 10 float32 bands summing to 1 within 1e-3,
  `argmax_is_9_share` near 0.
- **collect:** `manifest_verified` true in `exp/out/exp99[/pilot]/e99_collect.json`.
- **read:** one part per window, `parts_read` equal to the windows; the pilot's checks as exp98's (`ok True`).
- **inventory:** `exp99_inventory.json`'s `plots`: selected 309 (219, 90), in a grid and covered by country (the pilot:
  4 covered). Fewer than 250 covered voids the grades; the owner may change the floor before freezing, never after.

## Durations and volume (estimates, not measured)

| Stage | Expected | Limit | Why |
|---|---|---|---|
| select | 5 to 15 min | 1 h | two downloads (26 MB and 13 MB) and the clone's first `uv sync` |
| prepare | 1 to 7 h | 12 h | about 265 to 300 windows of 12 periods x 12 bands from Planetary Computer, 45 to 75 GB on scratch |
| predict | 1 to 7 h | 24 h | 7,225 crops of 16 px per window at batch size 4 |
| collect | 10 to 40 min | 3 h | about 10 to 12 GB to the home directory (quota checked) |
| read | 15 to 40 min | 3 h | one `from-olmoearth` per window |
| e99 (run) | 20 to 60 min | 3 h | one pass over the windows' pixels; `plan` at about 1.08 x 10^9 pixels for three rates |

## Unknowns (checked on the cluster, not here)

1. **2017 imagery.** Whether every window finds a scene in each of the 12 periods of 30 days (Sentinel-2A alone until
   mid-2017), and how Planetary Computer's 2017 L2A and rslearn's harmonization treat older processing baselines.
2. **The window lattice.** Ai2's doc says inference covers every 1,024 x 1,024 grid cell that intersects the geometry;
   whether olmoearth_run anchors that grid at the CRS origin (so a plot's window does not depend on the other plots) is
   in its unpublished source. Prepare records `windows_on_crs_origin_lattice`.
3. **One feature per plot.** Ai2's file holds one Polygon feature; this one holds 309. Two plots in one cell may give
   one window or two identical ones; the reader reads one window per part, and the analysis counts a plot in two grids.
4. **The clone.** `e99_code` is cloned from the home checkout's `origin` URL on first use; if that URL needs credentials
   a compute node lacks, the first stage fails at once with git's message.
5. **plan at 10^9 pixels.** numpy's multivariate hypergeometric sampler refuses populations of 10^9 or more, so the zone
   plan uses 999,999,999 windows (Part E of the page).
