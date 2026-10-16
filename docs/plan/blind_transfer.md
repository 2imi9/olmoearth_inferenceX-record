# A blind transfer test: do exp99's ranking miss and exp109's diagnosis hold at TimeSync plots no model has seen? (exp111 preregistration)

**Status: decided, not frozen.** Written 14 October 2026; revised 15 October 2026 after three independent reviews
(15 findings, then 11, then 5, all addressed below). On 15 October 2026 the owner confirmed, as proposed: the bars of
P1 to P3, the floors, the gate (A1 to A4, B1, B2 and their floors), the draw sizes (300/300/300, seed 111), the failure
allowance (0) and R3's exclusion. **They are binding at the phase-1 commit**, because the code phase 1 runs holds every
one of them: `exp111_grade.py`'s `FLOORS` and `BARS`, `exp111_blind_transfer.py`'s `GATE_BARS` and `SIZES`, and
`e111.sh`'s `MAX_FAILED_SHARE`. None is a submission option. The 720 floor is 80% of `SIZES`' 900 plots, and the
phase-2 guard, the units and the grade refuse a draw of other sizes. Every phase-2 step refuses if a guarded file
changed since the gate. A later change costs a gate rerun at least, and a new draw and prepare for the sizes
("Decisions for the owner" lists each cost). After freezing, none of them can change. At the phase-1 commit no cluster
job, download or prediction has been made for it. The sizing numbers come from `exp/exp111_power.py`
(`exp/out/exp111_power.json`), which reads only committed units files of exp98, exp99 and exp103. Scripts:
`exp/exp111_blind_transfer.py` (selection, windows, read, gate, units), `exp/exp111_grade.py` (the grade of P1 to P3,
written and smoked before any fresh plot was read) and `exp/exp111_jobs.py`; jobs `exp/jobs/e111.sh` and
`exp/jobs/e111_submit.sh` with `exp/jobs/e111_config/`, and the local smoke `exp/jobs/e111_local_smoke.sh`.

**Draw sha256 (frozen):** `(written here at freezing, from phase 1's e111_select.json)`

- **Allowed before freezing:**
  - the draw (below), which reads only TimeSync 2017 labels, countries and the distance to Ai2's request geometry;
  - the windows, their imagery and the label-free checks (windows, item groups, coverage, condition codes, the count
    of drawn plots in a gate cell);
  - the smokes on synthetic windows and synthetic units, the grader's included;
  - the gates A and B at points and plots the record has already graded (below). Their pass or fail goes on this
    page before freezing.
- **Not allowed before freezing:**
  - any prediction at a fresh plot, by either model (the predict job refuses until the page says frozen);
  - any join of a fresh plot's model output with its TimeSync label, its CGLS class or its distance band;
  - running the grader on real data (it refuses unless the page is frozen and holds the draw's sha256).

## What is and is not blind

**Blind (never read at, or joined to, any label):**

- either model's output at any TimeSync plot outside exp99's 309;
- any join of such an output with a TimeSync label, a CGLS class or a distance band;
- so P1, P2 and P3 are all graded on outputs nobody has read.

**One qualification, counted before freezing.** exp99 (v1) and exp103 (v1.2) predicted 313 windows of 1,024 px in 264
cells around exp99's 309 plots, 10 of them in EPSG:32736, with 51 plots in two or more
(`exp/out/exp99/e99_prepare_inventory.json`). A drawn plot just beyond 100 km can lie in one of those windows, so an
output at it exists on scratch, though it was never read at the plot or joined to its label. The windows step flags
those plots, label-free, and records the count and the method (`e111_windows.json`):

- **While exp99's 2017 deployment is on scratch and readable whole** (all 313 of its windows; until about 8 November): a
  plot is flagged when one of exp99's own windows holds it, in that window's CRS, or when its (EPSG, 1,024-px cell) is a
  gate17 unit's cell.
- **After the purge, or once it has started:** only the second rule. That count is a lower bound, because it misses
  exp99's other-zone and neighbouring windows. The purge removes files one by one, so a run directory can outlive some
  of its windows' `metadata.json` or lose whole windows; such a deployment (a metadata file unreadable, or fewer than
  313 windows read) is treated as gone, and the record says why. The select step does not fail for it.

Each flagged plot carries the flag into the units file, and the grade reports every point value again without them.

**Not blind:**

- **The hypotheses.** Every prediction below comes from exp109's report-only analysis of the 271 plots exp99 and exp103
  graded, so the bars were proposed after seeing those numbers. What is new is the set of plots.
- **The labels.** The TimeSync 2017 labels of every Kenya and Tanzania plot were read before:
  - exp94 graded CGLS-LC100 against them;
  - exp109 sized the pool by label;
  - the draw reads them by design, since it is stratified by label.
- **CGLS against TimeSync at the pool.** It was read at class level at every Kenya and Tanzania plot (exp94; exp109's
  overnight analysis). exp109's woody-cover line between Open and Wooded Grassland was fitted on 2,228 plots outside
  the 309, so the pool's labels joined with CGLS cover fractions are seen too. The CGLS order below (R1) is blind only
  in the models' calls.
- **The gates.** Gate A runs at Ai2's 259 validation points (exp98, exp103) and gate B at exp99's 309 plots (exp99,
  exp103). Both are seen. Gate B compares classes, p1, the suspicion's rank and the confident zone with exp99's and
  exp103's committed values; it reads no label.

## The question

exp99 (v1 FT-AWF) and exp103 (the record's v1.2 fine-tune) ranked their errors at 0.638 and 0.642 (design-weighted
AUROC, STRICT crosswalk). That was at 271 East Africa TimeSync 2017 plots within 100 km of the AWF area, below the
preregistered 0.70. exp109 then looked at those same plots (report-only, not preregistered) and found:

- The confident STRICT errors lie mostly on one contested pair. The map says shrubland_savanna where TimeSync says Open
  Grassland, at 11 of v1's 20 and 19 of v1.2's 22 confident STRICT errors. Neither legend publishes the boundary
  between those two classes.
- The confidence ranks the STRICT-only errors near chance (0.601 and 0.561) and the LENIENT errors better (0.698 and
  0.741).

All of that was on plots that had already been examined. exp111 asks the same questions blind, at plots no model's
output has been read at, with the models run through rslearn directly:

1. **P1.** Does the ranking miss the 0.70 bar again under STRICT, while still ranking better than chance, and does the
   legend account for part of the miss (LENIENT above STRICT)?
2. **P2.** Does exp109's diagnosis replicate? That is: is the contested pair most of the confident STRICT errors, and
   does the confidence rank STRICT-only errors below every product the record graded but LENIENT errors better?
3. **P3.** Does the package's per-class audit, at 20 labels per class, catch it in a new place? It should flag the
   contested class under STRICT and clear it under the crosswalk's LENIENT pairs, fixed before any label is read.

Report-only: the CGLS-disagreement order, a distance-band reading, the region's error rate.

**Scope (the owner's rule of 13 October 2026).** This tests the tool and a diagnosis. It proposes no model fix. A
confirmed legend pair goes to Ai2 as a finding.

## What exp109 predicts, plainly

Carried to the pool's label mix, which keeps the 271's behaviour within each label group and reweights the groups to
the pool (`power.json`, `seen_271.*.pool_mix`):

| | v1 | v1.2 | seen at the 271 (exp99's weights) |
|---|---|---|---|
| AUROC, STRICT | 0.644 | 0.664 | 0.638 / 0.642 |
| AUROC, LENIENT | 0.704 | 0.744 | 0.678 / 0.730 |
| LENIENT minus STRICT | +0.060 | +0.080 | +0.040 / +0.088 |
| Pair's share of confident STRICT errors (at the 271's label mix) | 0.55 | 0.86 | 0.56 / 0.84 |
| AUROC, STRICT-only errors against STRICT-correct | 0.609 | 0.556 | 0.601 / 0.561 |
| AUROC, LENIENT errors against STRICT-correct | 0.716 | 0.750 | 0.698 / 0.741 |
| shrubland_savanna's confident zone: STRICT / LENIENT error rate | 0.22 / 0.10 | 0.27 / 0.05 | 0.26 / 0.04 and 0.33 / 0.01 |
| P(the audit's note fires), 20 labels, α = 0.05 | 0.56 | 0.81 | 0.80 / 0.93 |

So exp109 predicts:

- **STRICT below the bar.** Both models stay below 0.70, well above chance.
- **LENIENT near the bar.** v1 sits on 0.70 and v1.2 sits above it.
- **The legend costs ranking.** LENIENT is above STRICT by 0.06 to 0.08.
- **The pair dominates** v1.2's confident STRICT errors, and holds about half of v1's.
- **The audit names the pair for v1.2.** For v1 it is a coin flip.

The reason for v1 is that the pool holds more non-grassland land (40% against 17% at the 271). There, v1's confident
shrub calls on Cropland are genuine errors, which raise LENIENT's rate in the shrub zone to about 10%.

**Assumption, untested.** The pool lies more than 100 km from Ai2's geometry, and genuine errors are likely to be more
frequent there. The 271's "other" group is mostly Cropland (33 of 45 with input); the pool's other group is 39%
Cropland.

## Data

**The sample** (Bullock, Healey, Yang et al. 2021, *Land* 10, 150). The public file is pinned as exp99 pinned it
(`github.com/bullocke/eastafrica` at `fc2014fc...`, `yearly_point_data.csv`, sha256 `5309a982...`, CC0). In each country
it is a simple random sample of 2,000 plots, with one TimeSync label per year.

**The pool.** These are the Kenya and Tanzania plots with a 2017 label that are not among exp99's 309. exp99's area rule
reproduces those 309 from the pinned file and Ai2's pinned request geometry (`fee3ce01...`): D = 100 km, 219 in Kenya and
90 in Tanzania, 47 inside. exp109's overnight rebuild matched all 309 on the Mac. So the pool is every plot more than
100 km from the geometry, 3,691 plots (`overnight/exp109/pool.json`):

| Label group | Kenya | Tanzania | Pool |
|---|---|---|---|
| Open Grassland (O) | 379 | 361 | 740 |
| Wooded Grassland (W) | 755 | 733 | 1,488 |
| Other (X): Cropland 564, Otherland 282, Open Forest 259, Dense Forest 167, Open Water 140, Vegetated Wetland 37, Settlements 14 | 647 | 816 | 1,463 |
| All | 1,781 | 1,910 | 3,691 |

Within a country, the pool is a simple random sample of that country's part of the region: the country minus exp99's
100 km buffer. The argument is exp99's: a region fixed by distance alone, conditional on the count.

**The draw (fixed here, before any prediction; sizes confirmed by the owner, binding at the phase-1 commit).**

- The strata are country x label group (6 cells).
- 300 O, 300 W and 300 X are drawn. Within each group they are split between the countries in proportion to the pool,
  by largest remainder: O 154 and 146, W 152 and 148, X 133 and 167. That is 900 plots.
- **The sizes are code, not an option.** They are `SIZES` in `exp111_blind_transfer.py`, part of the phase-1 commit,
  and the 720 floor is 80% of them. No submission option changes them. The phase-2 guard (`--check-draw`), the units
  and the grade each refuse a draw of other sizes, so a mismatch stops before any blind prediction.
- One numpy Generator, `numpy.random.default_rng(111)`, visits the cells in a fixed order (Kenya O, W, X, then
  Tanzania O, W, X) and permutes each cell's plot ids sorted as strings; the cell's first m ids are drawn. A smaller
  size is therefore a prefix of a larger one.
- **One route.** The draw runs once, on the cluster, in phase 1 (exp99's select pattern: fetch or reuse the pinned
  files, check their sha256, apply the rule). `e111_select.json` records the numpy and Python versions, because
  `default_rng`'s stream is numpy's. Anyone with the public file and the geometry can rerun it. There is no second
  route on the Mac: the Mac does not read the TimeSync labels for this test.
- **Privacy.** The sha256 of the sorted drawn ids, the counts and the distance-band cuts (R2) go on this page at
  freezing, the sha256 on the "Draw sha256 (frozen)" line at the top. The ids live only in a private file (mode 700 on
  the cluster's scratch), never in git and never printed. Every phase-2 step refuses a `draw.json` whose sorted ids do
  not give that sha256. Every record is written through one key check, and the pack jobs run the same check before
  anything leaves the cluster: a key that names an id or a position (exp98's list, plus `pid`, `px`, `py`, `cell`) is
  refused. The local smoke runs that check over every record it writes, under the names the cluster gives them.
- **Reruns.** A resubmitted select keeps an existing draw that equals its own (sha256, sizes, seed, order) and refuses
  a different one unless the owner sets `E111_REDO=select`. Once the page is frozen, both the job and the script
  refuse `E111_REDO=select`: it would replace the frozen draw, or remove blind17 with phase 2's outputs.
- **Why 300/300/300 and not exp109's 300/300/150.** The X group carries the largest design weights, so its size limits
  the region-level arms. Going from 150 to 300 lifts v1's P1 from 0.79 to 0.93 and P3 from 0.58 to 0.66 (sizing,
  below). It costs 150 more small windows, about 1 GB of scratch. 400/400/200 gains less for the region arms and
  would use 54% of the Open pool. 300/300/300 leaves 440 O, 1,188 W and 1,163 X plots unseen for a later blind test.

**Design weights.**

- **Region (P1, P3, report-only rates).** A plot with input in cell (c, g) has weight (A'_c / N_c) x (N_cg / m_cg).
  - A'_c is country c's Natural Earth area minus its part of exp99's region (60,574 km² Kenya, 47,105 km² Tanzania).
    Natural Earth v5.1.2 is pinned as in exp99, and the areas are computed on the cluster as exp99 computed them.
  - N_c is the country's pool count (1,781 and 1,910).
  - N_cg is the cell's pool count, and m_cg is its plots with input.
  - Plots without input are taken as missing at random within their cell. exp99 showed this can fail, so the envelope
    (all missing plots right, then all wrong) is reported.
  - The grader recomputes the weight from the units record (areas, pool counts) and refuses a units file whose weight
    column differs.
- **Standardised to the 271 (P2).** Each label group gets its share of the plot counts of exp99's 271 plots with
  input (O 97, W 129, X 45 of 271: 0.358, 0.476, 0.166). Within a group, plots are split as the region weights split
  them.
  - **Why.** P2 replicates exp109's statistic on new plots. Without standardising, a shift in the label mix (Open
    Grassland is 20% of the pool and 36% of the 309) would move the pair's share whatever the mechanism.
  - **Close to exp109's, not identical.** exp109's seen values (0.56 and 0.84 and the rest) used exp99's design
    weights, whose group shares are 0.389, 0.446 and 0.166. The predictions above and the grader use the same plot
    counts, so they agree with each other; against exp109's numbers the comparison is near, not exact.
  - The region-weighted values are reported beside.

## The maps: rslearn directly, pinned, gated

The owner promised Ai2 that predictions would run through rslearn directly. Nothing here imports `olmoearth_run` or
`olmoearth-runner`, and every rslearn job refuses if either can be imported in its venv. The pipeline is exp104's
route, which already ran on the cluster:

- exp111's script writes the windows;
- `rslearn dataset prepare`, `ingest` (a no-op) and `materialize` fetch Sentinel-2 L2A from Planetary Computer;
- `rslearn model predict` writes the ten-channel softmax as a float32 layer;
- `oe-inferencex from-olmoearth --conditions --inputs sentinel2_scl` reads each window as one part, as exp99 and
  exp103 did.

**Pins (the configuration whose known answers the record holds, not the newest).**

- **Library.** rslearn 0.1.17 in its own venv, with olmoearth-pretrain-minimal 0.1.2, torch 2.14.1, lightning 2.6.6
  and jsonargparse 4.52.0, the versions exp103 recorded in exp102's venv (`e103_predict17_env.json`). This is the
  version exp102 trained in, exp103 predicted v1.2 with over exp98's and exp99's datasets, and exp104 used for a full
  prepare-to-predict run. Every job checks all five; a rebuilt venv pins all five.
- **Record environment.** The selection, read, gate, units and grade run in the record's environment with
  `uv run --frozen` (`uv.lock` as committed; never re-resolved).
- **Dataset config.** Ai2's AWF `dataset.json` from `allenai/olmoearth_projects` at `f3c9b0c8`, with exp98's output
  layer (p0 to p9, float32) and exp98's SCL sidecar config. Its time order is pinned explicitly to the deployment's: the
  12 periods in reverse time order, which was rslearn 0.0.27's default (`per_period_mosaic_reverse_time_order`, noted in
  `exp/jobs/e98_env.sh`). This holds whatever 0.1.17's default is.
- **Model config.** `exp/jobs/e102_config/model.yaml`, which is Ai2's recipe in rslearn 0.1.17's form, with
  `output_probs` and the float32 writer layer. It keeps 16-px crops at overlap 0.25, so the stride is 12 px.
  `MODEL_ID` selects the encoder.
- **v1.** Ai2's checkpoint `allenai/OlmoEarth-v1-FT-AWF-Base` at revision `a347b154...`, sha256 `1036b76b...` (exp98's
  pins), with `OLMOEARTH_V1_BASE`.
  - The env job checks that this checkpoint passes a strict load under rslearn 0.1.17 (as amended below), on CPU, before
    any imagery is fetched. The checkpoint is read with `torch.load(weights_only=True)` (exp89's rule, with the list
    widened: a global from a listed module such as os, subprocess, builtins, operator or functools is refused, and any
    other becomes an inert stub), and the check fails unless the plain `weights_only` load works, because rslearn's
    predict loads `--ckpt_path` that way (LightningCLI) and restores through Lightning, which does too.
  - If it does not, v1 predicts with rslearn 0.0.29 installed directly (the first release with exp98's
    `get_item_by_name` fix), still not through the runner. The gates are rerun on that route before freezing. The
    owner decides.
  - **Amended 15 October 2026, twice, after two phase-1 submissions and before any imagery was fetched or any prediction
    made.** The first submission's env job (1355980) refused v1: its checkpoint holds one key the module does not,
    `model.encoder.0.model.composite_encodings.pos_embed`, OlmoEarth's legacy temporal position table. The encoder now
    computes that table on the fly (`get_1d_sincos_pos_encoding`) and its own `load_state_dict` pre-hook drops the
    stored copy, so Lightning's strict restore passes; the record's check compared key sets without running that hook.
    The check was changed to run the strict load through the module's own hooks and to accept a key the module lacks
    only when it is that table, held by a CompositeEncodings module (the class whose hook drops it), a real floating
    tensor at the width the encoder computes it at, with values equal to the on-the-fly encoding within 1e-5. The second
    submission's env job (1361497) then found the strict load passing and the plain `weights_only` load working, but
    v1's stored table differing from the on-the-fly encoding by up to 0.0037 (the same table in the locally cached
    OlmoEarth v1-Base weights, the encoder v1 was fine-tuned from, differs by at most 6e-8); a likely cause, not
    checked, is that the table was trained along with the encoder during Ai2's fine-tune: Ai2's recipe
    (exp/jobs/e102_config/model.yaml) freezes the encoder until epoch 20 with FreezeUnfreeze, and Lightning's unfreezing
    sets every encoder parameter trainable, the table included. Storage in bfloat16 would give a difference of the same
    size; the per-row record separates the two, because row 0 (sin 0, cos 0) is exact in every float format and would
    stay at 0 under a precision cause only. So under rslearn 0.1.17 v1 runs with the formula's table, not its own, and
    is not bit for bit the model exp98 and exp99 ran. The owner accepted that on 15 October 2026, and the gates judge
    it: they compare v1's predictions under this route with exp98's and exp99's known answers at the gate's bars
    (below). The check now accepts the table when it is within 0.01 of the formula (a guard against a table that is not
    this one, set after seeing 0.0037, so not blind) and records the difference, per row too, and whether the table is
    the formula's; any other extra, missing, reshaped or non-tensor key still fails it. exp89's inventory of the same
    checkpoint (sha256 `1036b76b...`) lists the table at 12 x 192 and loaded it with plain `weights_only`. No bar,
    floor, size or allowance changed. If gate A or B fails for v1, the imagery prepared for the gates is kept and the
    next route is the owner's call: v1 with its own table (new code that restores the stored table at predict time, so a
    guarded-file commit and a gate rerun) or rslearn 0.0.29 (not built; a second environment of about 4 GB).
  - **Amended 16 October 2026, overnight, after gate A failed for v1 and before any blind prediction.** The gate at
    3dd770d (job 1365118) ran v1 with the formula's table: 245 of 259 points the same class as exp98's (A1 needs 98%),
    39 with |Δp1| above 0.05 (A2 allows 5), Spearman 0.916. v1.2 reproduced exp103 at all 259 (|Δp1| at most 0.044,
    Spearman 0.9996), but exp103 ran on this same stack and v1.2's encoder (rope_3d_mixed) never uses the additive
    temporal table, so v1.2 shows the pipeline and the GPU are repeatable, not that the library change is harmless for
    v1. A′, v1 on exp98's own deployment windows, gave the same 245, so the imagery is not the cause. exp98 predicted on
    b200-devel too (job 1245574, a B200), so the GPU model is the same; torch (2.7.1+cu128 then, 2.14.1 now), rslearn
    (0.0.27, 0.1.17) and olmoearth_pretrain (0.0.2, minimal 0.1.2) are not. Candidate causes, none yet isolated: (1) the
    trained table (the load check's per-row record shows row 0 0.0031 from the formula, which no float format's rounding
    explains, so the table was trained in Ai2's fine-tune); (2) the encodings' arithmetic: olmoearth_pretrain 0.1.0
    added them through a float32 buffer, so under bfloat16 autocast the tokens left it in float32, while 0.1.2 builds
    the buffer in the tokens' dtype (a review's proxy on the v1-Base encoder put this about 30 times above the table's
    effect); (3) torch and rslearn's OlmoEarth wrapper. The committed route stays the owner's choice of 15 October
    (`V1_TABLE=formula` in `exp/jobs/e111.sh`). `exp/exp111_v1table.py` builds three candidate routes (`table`: the
    stored table; `table_fp32buf`: that, with the encodings added in float32; `table_fp32`: that, without autocast), and
    the report-only mode `v1diag` runs them and the formula at gate A's 259 points, compared with exp98's known answer
    as A1 and A2 compare it. This was built and run overnight because the owner asked for the work to continue; choosing
    a route after seeing gate A's known answers is a second look at them, so whichever route the owner chooses, both
    gate runs and every variant tried are reported beside the result. Running exp98's own stack directly would need its
    environment rebuilt: the scratch purge has emptied its CUDA libraries (`deploy/venv/.../nvidia/cublas/lib`), a
    download that is the owner's call. No bar, floor, size or allowance changed.
- **v1.2.** exp102's checkpoint (epoch 60, sha256 `5ce88827...`, public at `2imi9/OlmoEarth-v1_2-FT-AWF-Base` at
  `7a1db147...`) with `OLMOEARTH_V1_2_BASE`, as exp103 ran it.
- **One GPU model.** The gate and the blind predict both run on `b200-devel`, on B200 GPUs, as exp102 and exp103 did
  (they ran on `b200-batch`; the device check compares the GPU, not the partition). The gate records
  `torch.cuda.get_device_name` in `e111_gate.json`, and predict refuses another device. With bf16 autocast, near-ties
  can differ between GPU architectures, so a gate passed on one would not certify a predict on another.
- **The gate's code.** The gate job records the commit it ran (`predict_commit` in `e111_gate.json`), beside the
  commit gatecheck ran (`commit`). gatecheck refuses if a guarded file changed between the two, so a resubmission
  `--from gatecheck` at a later commit cannot pass predictions made with other code. Every phase-2 step refuses if a
  guarded file changed since either commit.
- **One gate job's records.** A gate job first removes every record of an earlier gate: its device record, its four
  output checks and `e111_gate.json`. It writes its own device record, which holds `predict_commit` and its Slurm job,
  only after all four of its output checks pass, and each output check names the job too. gatecheck refuses unless the
  five records are present, passing and one job's. So a gate rerun that dies part way leaves nothing gatecheck accepts,
  and an earlier gate's predictions cannot pass under a later gate's commit. The guarded files are the exp111 code and
  configs, every record module the exp111 modules import (the smoke derives that list from the import graph), and the
  record environment's pins.
- **Patrick's point.** The configs in `olmoearth_projects` may be outdated. The pinned config is kept because the
  gates' known answers belong to it. Ai2's main-branch AWF configs are byte-identical to the pinned `f3c9b0c8`
  versions (last change 10 December 2025), so "outdated" points at the runner and rslearn pins. No newer config is run.

**The windows.**

- Each window is 124 x 124 px at 10 m (16 + 9 x 12, a whole number of crop strides), one per plot.
- It uses the plot's UTM zone, as Ai2's projection rule gives it (rslearn's own `get_utm_ups_crs`, compared at every
  window).
- Its origin sits on the deployment's crop lattice. That lattice is the 1,024-px cell anchored at the CRS origin; all
  313 of exp99's windows lay on it (`e99_prepare_inventory.json`), offset by a multiple of 12 px. So the crops that
  cover the plot's pixel are the crops the deployment ran. (With 128 px, rslearn 0.1.17's last crop of every window
  falls off that lattice; 124 px is exact at every one of the cell's 1,024 offsets.)
- A plot more than 56 px from its cell's edge sits 56 to 67 px inside its window. A plot nearer the edge gets a
  window flush with that edge.
- The time range is the year's request range, 1 January to 31 December 2017, as exp99's deployment windows hold it.
  rslearn cuts 12 periods of 30 days backwards from 31 December, so the newest period is 1 to 31 December and the
  earliest starts 5 January.
- Compared with exp99's 1,024-px windows this is about 68 times less imagery per plot. That is why the test costs
  gigabytes, not exp109's estimated 225 GB.
- Three dataset roots are never mixed: gate23 (Ai2's 259 points, 2023), gate17 (exp99's 309, 2017) and blind17 (the
  900, 2017). 1,468 windows in all, capped at 1,500.

**Reading (exp99's rules, unchanged).**

- Each plot takes the score pixel that holds its point, read only from its own window (windows of nearby plots can
  overlap; the overlaps are counted).
- The class is the argmax over all ten channels. Channel 9 is an error under both rules.
- The confidence p1 is the top-1 over channels 0 to 8, and the suspicion is −log p1.
- Probabilities that do not sum to 1 within 1e-3 are refused.
- The errors are STRICT and LENIENT from `exp/timesync_awf_crosswalk.py` (rules sha256 `4c8b4528...`, fixed on
  9 October 2026).

### The gates (a known answer before any fresh plot; one rule set, confirmed, binding at the phase-1 commit)

The same pipeline, with the same pins and on the same GPU model, runs at Ai2's 259 validation points with 2023 input
(gate A) and at exp99's 309 plots with 2017 input (gate B). Each model is compared with its committed known answer at
the units with input in both runs. Input gained or lost is counted beside and bounded by the floors.

**Gate A (2023): Ai2's 259 validation points.** Known answers: exp98's deployed v1 (`exp/out/exp98_units.npz`,
`pred_deployed`, `p1`) and exp103's v1.2 (`exp/out/exp103/awf2023/exp98_units.npz`).

| Check | Rule |
|---|---|
| A1 | v1: the same class as exp98's at 98% or more of the points with input in both (at most 5 of 259 differ) |
| A2 | v1: at most 2% of those points (5 of 259) with \|p1 − known p1\| > 0.05 |
| A3 | v1.2: the same class as exp103's at 98% or more |
| A4 | v1.2: at most 2% with \|p1 − known p1\| > 0.05 |
| Floor | at least 240 of the 259 with input in both, for each model |

**Gate B (2017): exp99's 309 plots, only after gate A passed.** Known answers: exp99's v1 (`exp/out/exp99_units.npz`)
and exp103's v1.2 (`exp/out/exp103/transfer2017/exp99_units.npz`), the 271 with input. Rows map to plots by the public
sample's sorted ids, with country and inside flags checked.

| Check | Rule, for v1 (B1) against exp99 and for v1.2 (B2) against exp103 |
|---|---|
| Class | the same class at 98% or more of the plots with input in both |
| p1 | at most 2% (5 of 271) with \|p1 − known p1\| > 0.05 |
| Rank | Spearman correlation of the suspicion −log p1 with the known suspicion at least 0.95 |
| Confident zone | of the known answer's confident plots (p1 at or above the cut below: 91 of the 271), at least 95% (87 of 91) at or above the cut here |
| Floor | at least 250 of the 271 with input in both, for each model |

**PASS = A1 to A4, gate A's floor, B1 and B2 (all four checks each) and gate B's floor.** `exp111_blind_transfer.py`'s
`GATE_BARS` holds exactly these numbers. The gate's GPU and both of its commits must be recorded, or the gate does not
pass.

**Amended 16 October 2026 (the owner's decision, after gate A at 3dd770d and the v1 route diagnostic, before any blind
prediction): v1.2 is the graded model.** v1 does not reproduce exp98 under rslearn 0.1.17 by any route tried: the
report-only diagnostic (job 1369140, at gate A's 259 points) gave 245 of 259 the same class with the formula's table,
245 with v1's stored table, 245 with that and the encodings added in float32, and 244 with no autocast, each with 39 or
40 points above the |Δp1| tolerance and the same confusions (grassland_barren to shrubland_savanna 6, woodland_forest to
shrubland_savanna 4), so the stored table, the encodings' precision and autocast are ruled out and the difference lies
elsewhere between the two stacks (rslearn 0.0.27 against 0.1.17, olmoearth_pretrain 0.0.2 against minimal 0.1.2, torch
2.7.1 against 2.14.1; not isolated). So the gates pass on A3, A4, gate A's floor, B2 (all four checks) and gate B's
floor, counted on v1.2 (`GRADED_MODELS` and `GRADED_CHECKS` in `exp111_blind_transfer.py`); A1, A2 and B1 are computed
and reported. P1 to P3 are graded for v1.2 (`verdicts_graded` in the grade); v1 still predicts the blind plots and its
verdicts are reported beside, report-only, as the verdicts of a model that does not reproduce its known answer on this
route. The bars, floors, sizes and allowance are unchanged. The decision was made after seeing gate A's known answers,
not any blind plot; both gate runs and the diagnostic are reported beside the result. That a fine-tuned OlmoEarth
checkpoint's predictions change with the library version (5.4% of the 259 points change class) is itself a finding,
recorded with the result.

**Report-only beside the gates, never a verdict:**

- the accuracy against Ai2's labels at the 259, beside exp98's 226 and exp103's 227 (within 3 points is not a check:
  v1.2's classes given as v1's would pass it);
- agreement at the units whose known p1 is 0.6 or more, and the near-ties below it that differ;
- the confusion of the differing units, the units gaining or losing input, the plots entering the confident zone;
- diagnostic A′ (v1 under 0.1.17 on exp98's own 1,024-px windows that hold the 259 points) and the item-identity
  split (agreement for windows whose 12 item groups equal the deployment window's, and for the rest).

**Why these levels.**

- **98% the same class.** An equivalent pipeline on the same scenes should agree at nearly every unit; only a near-tie
  can flip. These bars were set before v1's table was known (Pins, amendment); for v1 they now also bound the effect of
  running the formula's table instead of its own, with v1.2, which has no legacy table, as the control. exp98's own
  deployment against exp89's replica, a different imagery route, agreed at 91.9%, and two different models (v1 and v1.2)
  agree at 88.4% of the 259 and 82.3% of the 271. So 98% sits well above an imagery change and far above a model change.
  3.5% (v1) and 2.7% (v1.2) of the 259 have p1 below 0.6, and 7.4% and 6.6% of the 271; agreement above 0.6 is reported
  beside so that a failure on near-ties is visible as such.
- **The p1, rank and zone checks.** P1 to P3 grade p1's ranking and a confident cut, not only the class, so the gate
  checks them in 2017, the test year, for both models. v1 against v1.2 shows how far apart two models are: |Δp1| above
  0.05 at 49 of 259 and 87 of 271 units; Spearman correlations of p1 of 0.780 and 0.747; 65 of v1's 91 confident
  plots at or above the cut. The smoke runs the gate's own code on the committed files: they pass every check as
  themselves, v1.2's values given as v1's fail A1, A2 and all four of B1's checks, p1 lowered by 0.06 at six units
  fails only A2 and B1's p1 check, and five confident plots moved just below the cut fail only B1's zone check.

**If a gate fails.**

- Nothing is predicted at a fresh plot.
- The failure is diagnosed (versions, config, time order, item sets, v1's temporal table, A′) and reported to Ai2 as a
  finding. v1.2 has no legacy table, so v1.2 passing while v1 fails points at v1's table or its library route and is not
  by itself evidence for Patrick's point; A′ and the item-identity split (report-only) now measure rslearn 0.1.17
  together with the table for v1, not the pipeline alone.
- If gate B fails while gate A passes, the 2017 imagery is suspected, for example Planetary Computer items reprocessed
  since 9 October, rather than the pipeline.
- A pixel-level diagnosis needs exp98's and exp99's deployment datasets, which the scratch purge removes around 7 to 9
  November (the table under "Timing and purge-bound inputs").

## Predictions (all bars confirmed by the owner; binding at the phase-1 commit)

`exp/exp111_grade.py` computes every quantity and verdict below exactly as written here; it is committed and smoked
before freezing, and the blind predict refuses if it changed since the gate.

**Floors (per model).** Below a floor the value is reported without a verdict.

- Every prediction: at least 720 of the 900 plots have 2017 input (80%, as exp99's 250 of 309; the floor is 80% of
  `SIZES`). Windows that still fail after prepare's third pass, or whose SCL fails, within the failure allowance
  (below), count as plots without input in their cell.
- P1 and P2: at least 40 errors and 40 correct plots under each rule.
- P2: at least 25 confident STRICT errors and 30 STRICT-only errors.
- P3: at least 30 plots in shrubland_savanna's confident zone.
- Expected values (sizing, exp99's input loss): 769 with input; v1 about 270 STRICT and 120 LENIENT errors; v1.2 about
  280 and 140; 61 and 63 confident STRICT errors; 137 and 147 plots in the shrub zone.

**"Confident"** means p1 at or above exp109's cut, the 91st-largest p1 at the 271, written to 11 digits and rounded
down: 0.99875695977 (v1) and 0.99888118198 (v1.2). The fresh plots' own share above the cut and their own top third
are reported beside.

**The intervals** come from a bootstrap that resamples the plots with input within each country x label-group cell
(2,000 resamples, seed 111, the same resamples for both models, 90% percentile). There is no correction for multiple
predictions. The two models are not independent: they share Ai2's labels and recipe and the same plots.

### P1: the ranking, under STRICT and under LENIENT (the frozen exp99 bar, 0.70)

The design-weighted (region) AUROC of the suspicion for the errors at the plots with input, per model.

- **P1(a), STRICT: below the bar, above chance.** Holds when the STRICT AUROC is below **0.70** and its 90% interval
  lies above **0.50**. exp109's prediction and a replication of exp99's and exp103's miss; the lower side keeps a
  noise-level confidence from passing it.
- **P1(b), the legend's share.** Holds when LENIENT's AUROC minus STRICT's, on the same plots, has a 90% interval above
  **0**.
- **P1 holds for a model** when (a) and (b) both hold; the prediction is for both models (amended 16 October
  2026: graded for v1.2; v1's verdict is reported beside).
- **LENIENT against 0.70, as a call, not a bar.**
  - "Above" when the 90% interval lies wholly above 0.70.
  - "Below" when it lies wholly below.
  - "Not determined" otherwise.
  - exp109 predicts "near the bar": not determined for v1, above or not determined for v1.2.
- The unweighted values and the capture at 10% (exp99's gap closed) are reported beside.

### P2: exp109's diagnosis, replicated blind (standardised to the 271's label mix)

- **P2(a), the pair.** Holds when, of the confident STRICT errors, the share that is shrubland_savanna on TimeSync Open
  Grassland is at least **0.50** ("most"), for both models. Seen: 0.56 (v1) and 0.84 (v1.2).
- **P2(b), what the confidence sees.** Holds for a model when all three of these hold:
  - AUROC(LENIENT errors against STRICT-correct) minus AUROC(STRICT-only errors against STRICT-correct) has a 90%
    interval above **0**;
  - AUROC(STRICT-only errors against STRICT-correct) is at most **0.65**: below every product confidence the record
    graded on a probability sample (0.720 to 0.787);
  - AUROC(LENIENT errors against STRICT-correct) is at least **0.65**: the confidence still ranks the genuine errors,
    so a degraded or noise-level confidence cannot pass the arm above.
- **P2 holds for a model** when (a) and (b) both hold; the prediction is for both models (amended 16 October
  2026: graded for v1.2; v1's verdict is reported beside).
- The share of confident STRICT errors that are STRICT-only, the reverse pair (grassland_barren on Wooded Grassland) and
  the region-weighted values are reported.

### P3: the per-class audit flags the contested class under STRICT and clears it under LENIENT

This is the package's audit (`sample --design audit --per-class 20 --zone-cut <the cut>`, then
`estimate --alpha 0.05 --also-accept pairs.json`), read through its operating characteristic at the fresh plots.

- **The zone and its rates.** The zone is shrubland_savanna's confident zone: fresh plots with input, p1 at the cut or
  above, class shrubland_savanna. e_S and e_L are its region-weighted STRICT and LENIENT error rates.
- **When the note fires.** With 20 labels and the package's 95% interval (a zone of the region's size, N = 10^6, the
  binomial limit), the class is "above" α = 0.05 at 4 or more errors (k* = 4, computed from `hypergeom_interval` and
  `interval_verdict`). The audit's note fires when the STRICT reading is above and the also-accept reading is not.
  - P(note) = P(K_S ≥ 4 and K_L ≤ 3), with (K_L, K_S − K_L) multinomial(20; e_L, e_S − e_L).
- **Bar:** P(note) at least **0.5** for each model, meaning the audit names the pair more often than not
  (amended 16 October 2026: graded for v1.2; v1's is reported beside).
  - The component probabilities P(K_S ≥ 4) and P(K_L ≥ 4), the 90% intervals of e_S and e_L, and P(note) at 15 and 30
    labels are reported.
- **"Clears" means not flagged.** Twenty labels cannot certify a class below 5%: 0 errors in 20 still leaves an upper
  bound of 16.8%.
- **The also-accept file (fixed now in `exp111_grade.py`, `ALSO_ACCEPT` and `EXTENDED_CODE`, before any label).**
  - It must be written in an extended code, because a STRICT counterpart is shared by two TimeSync classes. Open
    Forest and Wooded Grassland both map to 2; Otherland and Open Grassland both map to 4.
  - So the reference is coded as the STRICT counterpart for seven classes, Open Forest as 10 and Otherland as 11, and
    the file is `{0: [10], 1: [3, 11], 2: [4, 10], 3: [1], 4: [2, 11], 5: [8], 6: [0, 10], 7: [0, 10]}`.
  - With it, the also-accept reading equals LENIENT exactly; the smoke checks this for every TimeSync class and every
    map class. The STRICT reading is a second call on the same draw with the reference coded as its STRICT
    counterpart.
  - The grade uses the crosswalk's own errors (`wrong_strict`, `wrong_lenient`), so the coding matters only for the
    literal run below.
- **Literal run (report-only).** The package's audit functions are run once on the fresh plots (seed 111): one sample,
  read with the STRICT-coded reference and then with the extended-coded reference and the file, to show the note's text
  and the pairs it names. The draw is unweighted, so Open Grassland is over-represented; it is a demonstration, not the
  grade.
- **If the contested class is not shrubland_savanna,** P3 is still graded on shrubland_savanna. The class holding the
  most weighted confident STRICT-only errors is reported.

### Sizing (the 271 resampled within label groups, exp99's input loss, 2,000 draws; `power.json`)

The probability that each prediction holds when exp109's per-group behaviour carries over, recomputed on 15 October with
the lower sides:

| Prediction | v1: 300/300/300 | v1: 300/300/150 | v1.2: 300/300/300 | v1.2: 300/300/150 |
|---|---|---|---|---|
| P1(a) STRICT below 0.70, interval above 0.50 | 0.996 | 0.986 | 0.959 | 0.917 |
| P1(b) LENIENT − STRICT, interval above 0 | 0.93 | 0.80 | 1.00 | 0.99 |
| P1(a) and (b) | 0.925 | 0.792 | 0.959 | 0.911 |
| LENIENT point ≥ 0.70 / call "above" | 0.57 / 0.07 | 0.55 / 0.05 | 0.99 / 0.72 | 0.95 / 0.51 |
| P2(a) pair share ≥ 0.50 | 0.79 | 0.77 | 1.00 | 1.00 |
| P2(b): gap above 0 / STRICT-only ≤ 0.65 / LENIENT ≥ 0.65 | 0.95 / 0.945 / 0.99 | 0.92 / 0.945 / 0.985 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| P2(b) whole | 0.91 | 0.88 | 1.00 | 1.00 |
| P2 whole | 0.745 | 0.72 | 1.00 | 1.00 |
| P3 P(note) ≥ 0.5 at 20 labels (15 / 30 labels) | 0.66 (0.10 / 0.90) | 0.58 (0.06 / 0.82) | 1.00 (0.83 / 1.00) | 0.995 (0.80 / 1.00) |

Read with care:

- **Optimistic.** The simulation ignores the country weights (a design effect) and the farther transfer, and it
  reads the intervals from the replicates' spread (a normal approximation), not from a bootstrap per replicate.
- **The X group.** It is resampled from the 271's mostly-Cropland other group.
- **The lower sides cost little.** STRICT sits near 0.64 with a spread of about 0.02, so its interval clears 0.50 in
  every draw; LENIENT errors against STRICT-correct sit near 0.72 (v1) and 0.75 (v1.2).
- **15 labels are too few for v1.** 30 would lift v1's P3 to 0.90; the owner's range was 15 to 20.
- **What the design decides.** P1(a), P2(b) and v1.2's arms are near-certain if exp109 is right, so a miss there is
  informative. v1's P2(a) and P3 are the uncertain arms.

## Report-only (no bar, no verdict)

- **R1, the CGLS-disagreement order.**
  - **The order.** Plots where the model's class does not accept CGLS-LC100's raw 2017 class come first, ties by
    suspicion.
  - **The crosswalk.** exp109's Arm W, every accepted code, fixed now (`exp111_grade.ARM_W`):
    - woodland, montane and lava forest accept 111 to 116 and 121 to 126;
    - open_water and herbaceous_wetland each accept 80, 90 and 200;
    - shrubland_savanna accepts 20, 30, 100 and 121 to 126;
    - grassland_barren accepts 20, 30, 60 and 100;
    - agriculture_settlement accepts 40 and 50;
    - urban_dense_development accepts 50;
    - channel 9 accepts nothing;
    - a plot with no CGLS value is not flagged.
  - **What is reported.** The region-weighted AUROC gain over the suspicion, under STRICT and LENIENT, per model.
  - **Data.** CGLS's raw class comes from `exp/out/exp94_eastafrica_points.csv` (2017 rows), joined by public plot id
    in the private draw file, after freezing.
  - **Seen at the 271, in-sample and chosen among 4 variants:** +0.060 and +0.055 under STRICT, +0.163 and +0.141
    under LENIENT. Here it is blind only in the models' calls.
- **R2, distance to the AWF area.** Error rates and AUROC under each rule, by band:
  - inside Ai2's geometry (exp99's 27 plots with input; seen);
  - near, within 100 km (exp99's other 244; seen);
  - far, the fresh plots in three bands at the tertiles of their distance. The cuts are computed at the draw by
    exp99's distance rule (EPSG:32737, edges densified every 0.001 degree), rounded to 10 km and recorded here before
    any prediction.
  - This is the 3-band version of exp99's inside/outside split, after MIND the Gap (arXiv 2609.25454). It does not
    separate distance from landscape.
- **R4, the region's error rate.** This is exp99's Part A for the new region: under each rule, the six country x label
  cells as strata, the package's condition-design interval, and the missing-input envelope.
- **R5, inventory.** Plots without input by cell (windows quarantined within the failure allowance among them),
  condition codes at the plots, and both models' agreement at the fresh plots (82.3% at the 271).
- **The flagged plots** (where an output of exp99 or exp103 exists; "One qualification" above). Every point value
  of P1 to P3 again without them, with their count and the flag's method.

**Not in this test: R3, a flip count across extra years.** The owner's candidate cue (a class that flips 2016 → 2017
→ 2018 and back, aimed at LENIENT errors) is not part of exp111: no roots, time ranges, predictions or reader exist for
it. It is an optional extension the owner may add before the phase-1 commit, at about double the imagery and prepare
time, with its own 2016 and 2018 roots, its own flip-order reader and its own smoke, and its own bars. Added after
that commit, it is new guarded code, so the gate is rerun. 2016 has Sentinel-2A only, so more plots may lack it.
Without that addition nothing here reads another year.

## Analysis steps (the order the scripts run them)

1. **Select (phase 1, cluster).** Read the pinned TimeSync file. Apply exp99's area rule and refuse unless it gives 309
   (219, 90, 47 inside). Build the pool and refuse unless it gives the counts above. Make the draw; record its sha256,
   the allocation, the band cuts, the areas and the numpy and Python versions.
2. **Windows (phase 1).** Write the three roots; flag the drawn plots at which exp99 or exp103 wrote an output (by
   exp99's own windows while they are on scratch and all 313 readable, else by the gate17 cells, a lower bound).
3. **Prepare (phase 1).** In this order, for each root:
   - prepare and materialize, in three passes;
   - the SCL sidecar of the windows with all twelve groups, and its materialize, whose failures are added to pass 3's
     against the failure allowance;
   - the precheck, which quarantines partial windows and windows whose SCL failed, within the allowance;
   - the SCL check over the windows with input;
   - the input count by cell (the inventory).

   Run the gates.
4. **Freeze.** The owner reads the gates and the inventory, writes the draw's sha256 on the "Draw sha256 (frozen)"
   line and freezes the page. The bars, floors, sizes and allowance were decided at the phase-1 commit; changing one
   here costs what "Decisions for the owner" says.
5. **Predict, read, units (phase 2).** Each step refuses unless the page is frozen, the gate passed, both of the
   gate's commits (the gate job's and gatecheck's) are ancestors with the guarded files unchanged since, and
   `draw.json` gives the page's sha256. Predict also refuses another GPU than the gate's. The read refuses
   probabilities that do not sum to 1.
6. **The units file** (`exp111_units_v1.npz`, `exp111_units_v12.npz`) holds per plot: country, label group, label,
   covered, the model's class and p1, suspicion, condition, 3 x 3 majority, the region weight, both rules' errors,
   distance band, CGLS raw class and the gate17-cell flag. It holds no identifier and no position. Its rows are shuffled
   with a permutation from the operating system's entropy, kept only in the private directory, so that no row lines up
   with the public file's order and nobody can undo the shuffle from a seed (exp104's audit lesson). The units record
   holds the draw's sha256, the files' sha256, the areas, the pool counts and the drawn plots per cell.
7. **Grade.** `exp111_grade.py --smoke`, then the grade: it refuses unless the page is frozen, the units record's draw
   is the page's and the units files are the ones the record names; it recomputes the weights and the errors, grades
   P1, P2 and P3 at the floors, then R1, R2, R4, R5 and the gate17-cell exclusion, and writes
   `exp111_summary.json` (aggregates only).

An audit that did not write the scripts recomputes the grades from the units file with plain numpy before anything is
recorded. Its script becomes the claims' second route.

## What follows, whatever the outcome

- **P1(a) and (b) hold.** exp99's and exp103's miss replicates on new plots beyond 100 km, the ranking stays above
  chance, and part of the miss is the legend boundary. The record's statement that the review order is weaker under
  transfer with STRICT then rests on a blind test.
- **P1(a) fails above (STRICT at or above 0.70).** The miss was local to the 309 near the AWF area. exp109's diagnosis
  describes those plots, not the transfer.
- **P1(a) fails below (STRICT's interval reaches 0.50).** The confidence does not rank errors at all in the new place;
  that is a stronger finding than exp99's, and P2's reading is moot.
- **LENIENT "above".** The genuine errors rank at the bar in a new place, so the ranking stands under transfer once the
  legend is set aside.
  - **"Below".** Transfer costs ranking beyond the legend, and the review order is not recommended for a transferred
    map without labels.
  - **"Not determined".** As exp99.
- **P2 holds.** The diagnosis replicates blind. Under STRICT, the confident errors are mostly the contested pair, and
  the confidence cannot see them while it still sees the genuine errors. This is the finding for Ai2: a boundary
  neither legend publishes.
  - **P2 fails.** The pattern was particular to the 271.
- **P3 holds.** In a new place, about 20 labels per predicted class find the class and name the pair (the package's
  note), with no model change.
  - **P3 fails for a model.** 20 labels are too few there, or genuine errors in the zone mask the pair. Both are
    reported with e_S and e_L.
- **A gate fails.** There is no fresh prediction. The failure itself, rslearn-direct against Ai2's deployed
  configuration, goes to Ai2.

## What would change the conclusion

- **The area rule, the pool or the draw differ from this page.** The scripts refuse.
- **Fewer than 720 plots with input.** There are no verdicts. Plots without input are reported by cell. The floor, and
  the sizes it is 80% of, are binding at the phase-1 commit. If the label-free inventory shows the shortfall before
  freezing, the owner may still lower the floor, at the cost of a commit to `exp111_grade.py` and a gate rerun
  (below), and this page then says the floor was changed after the inventory was seen. The smoke refuses a floor above
  80% of `SIZES`' plots, so smaller sizes need a lower floor in the same commit. After freezing it cannot change.
- **v1 needs rslearn 0.0.29.** The gates are rerun on that route before freezing, and the page names it.
- **Windows near a cell edge, or drawn plots where an output of exp99 or exp103 exists.** They are counted; the
  gates' agreement near edges and the grades without the flagged plots are reported separately. The record says
  whether the flag used exp99's own windows or, after their purge, only the gate17 cells (a lower bound).

## What it cannot show

- **Whose boundary is right.** Labels are taken as right. The STRICT-only bucket is defined by the crosswalk, so it
  cannot separate a legend convention, a systematic TimeSync interpretation practice, or a gap in what AWF's labels
  support. Only AWF-protocol labels could (exp109's Part B, not run).
- **Ai2's 1,024-px deployment, literally.** It is the same checkpoint and configuration at the plot's pixel, with v1
  running the formula's temporal table instead of its own (Pins, amendment), to the extent the gates show.
- **Two independent confirmations.** The two models share Ai2's labels and recipe and are graded on the same plots.
- **Why a difference from exp99 arises.** The pool is farther (beyond 100 km) and its land differs, so place, distance
  and land cover are not separated; R2 only describes them.
- **The region beyond the plots with input.** Region weights assume missing at random within a cell. The envelope is
  reported.
- **A literal audit of the region's map.** P3 is the audit's operating characteristic computed from plot estimates of
  the zone's rates. It is not 20 labels drawn from the map's own pixels.
- **Other years, places or legends.** It covers 2017, Kenya and Tanzania, one legend pair, and two fine-tunes of one
  task.
- **CGLS as an independent witness.** CGLS-LC100 shares WorldCover's lineage, a pretraining target of both encoders,
  so R1 measures agreement with a related map.

## Cost (from the job scripts' requests; estimates where marked, not measured)

**Jobs and their requests** (`exp/jobs/e111_submit.sh`; account `p2026_0089_neu`):

| Phase | Job | Partition | CPUs | Memory | Limit | Core-hours at the limit | Expected |
|---|---|---|---|---|---|---|---|
| 1 | env | cpu | 4 | 32 GB | 1 h | 4 | 10 to 40 min |
| 1 | select | cpu | 4 | 16 GB | 30 min | 2 | 5 to 15 min |
| 1 | prepare | cpu | 32 | 64 GB | 8 h | 256 | 1.5 to 5 h (48 to 160 core-hours) |
| 1 | gate | b200-devel, 1 GPU | 8 | 64 GB | 2 h 30 | 20 | 30 to 90 min (A′ up to 1 h of it) |
| 1 | gatecheck | cpu | 8 | 32 GB | 1 h | 8 | 10 to 30 min |
| 1 | pack1 | cpu | 1 | 4 GB | 15 min | 0.25 | 1 min |
| 2 | predict | b200-devel, 1 GPU | 8 | 64 GB | 1 h | 8 | about 15 min |
| 2 | read | cpu | 8 | 32 GB | 1 h | 8 | 10 to 30 min |
| 2 | units | cpu | 4 | 16 GB | 30 min | 2 | minutes |
| 2 | grade | cpu | 4 | 16 GB | 30 min | 2 | minutes |
| 2 | pack2 | cpu | 1 | 4 GB | 15 min | 0.25 | 1 min |

- **CPU.** At most about 290 core-hours in phase 1 and 20 in phase 2; expected about 60 to 170 and 5. The prepare job
  takes 32 of the 48-core per-user cap, so other cpu jobs are not blocked.
- **GPU.** At most 3.5 GPU-hours (gate 2.5, predict 1), both on b200-devel (QOS interactive, at most 4 such jobs at
  once); expected about 1 to 1.5.
- **Windows.** 1,468: 259 (gate23) + 309 (gate17) + 900 (blind17); the cap is 1,500.
- **Wall time.** Phase 1 about 3 to 8 h after it starts, phase 2 about 1 to 2 h, plus queue time.

**Disk (bytes stored):**

| Where | What | Size |
|---|---|---|
| scratch, `e111_private` | the three roots' imagery (12 groups x 12 bands x 124² x 2 B), the SCL sidecars, both models' outputs, the reads | about 9 GB at most (about 5.8 MB per window, uncompressed); the watchdog stops prepare above 25 GB |
| scratch, `deploy` | A′'s transient layer in exp98's dataset, removed by the gate's exit trap, gatecheck and pack1 | up to about 11 GB, for at most a few hours |
| scratch, `e111_code` | the clone and its `.venv` (`uv run --frozen`) | about 0.2 GB of git objects and about 0.5 GB installed |
| scratch, `e111_out` | records, masked logs, trainer directories, units, the summary, two tarballs | under 0.1 GB |
| scratch, `e111_private/t` | the jobs' temporary files: rslearn's window lists and its forkserver sockets (amended 16 October 2026: under `e111_private/tmp/<tag>-<job>` the socket path reached the 107-character limit and prepare 1364257 failed before any imagery; now `t/<job>.<n>`) | KB to MB |
| scratch, only if purged | a rebuilt rslearn venv (`e111/venv`); the checkpoints and base weights (`hf`); the public inputs (`e111_data`) | up to about 8 GB; about 3 GB; 39 MB |
| /home | Slurm logs only, nothing collected; the env job refuses above 90 GiB or 450,000 files, or when the size is unknown; the end of prepare records the usage as a note only (it writes nothing to /home, and a refusal there would cancel the chain after hours of work) | KB to a few MB, about a dozen files |
| Mac | the two tarballs and the units files | a few MB |

**Network (bytes transferred, apart from bytes stored):**

- **Planetary Computer** (Sentinel-2 L2A and SCL, no key): about 50 to 150 GB read for about 7 GB stored (estimate:
  each 124-px read pulls whole internal COG tiles per band and item). This is the transfer to watch after the
  9 October incident (530 GB fetched by a partitioner); the window count is fixed by construction and capped.
- **PyPI**, the record environment's first `uv run --frozen` in the fresh clone (geopandas, rasterio, pyproj, shapely,
  scipy, matplotlib, numpy, olmoearth-inferencex): about 0.2 GB.
- **GitHub**, the clone of the record repo: about 0.2 GB.

**Downloads, confirmed by the owner on 15 October 2026** (each one only where marked):

| Download | Size | When |
|---|---|---|
| Sentinel-2 L2A and SCL, Planetary Computer | about 7 GB stored, 50 to 150 GB transferred (estimate) | always (phase 1, prepare) |
| the record repo clone (GitHub) and the record environment (PyPI) | about 0.2 GB and 0.2 GB | always (phase 1, first job) |
| v1's checkpoint, `allenai/OlmoEarth-v1-FT-AWF-Base` at `a347b154...` (`E111_HF_OFFLINE=0`) | 1.04 GB | likely for any run after about 20 October (purged by 30 October at the latest, possibly already); else only if purged from `$SCRATCH/hf` |
| v1.2's checkpoint, public copy `2imi9/OlmoEarth-v1_2-FT-AWF-Base` at `7a1db147...` | 1.16 GB | only if purged from `e102/runs` and not cached |
| the encoder base weights, `allenai/OlmoEarth-v1-Base` and `allenai/OlmoEarth-v1_2-Base` | about 0.4 GB each (estimate) | likely for any run after about 20 October, as v1's checkpoint; else only if purged from `$SCRATCH/hf` |
| a rebuilt rslearn venv (`E111_BUILD_VENV=1`): rslearn 0.1.17, torch 2.14.1 with its CUDA libraries | about 4 GB (estimate), about 8 GB installed | only if exp102's venv is purged |
| the TimeSync file and Natural Earth (`E111_FETCH=1`) | 26 MB and 13 MB | only if `~/olmoearth_inferenceX/data/breadth` lacks a verified copy |
| Ai2's request geometry (`E111_FETCH=1`, sha256 `fee3ce01...`) | KB | only if purged from `$SCRATCH/deploy` |
| Ai2's AWF label dataset for gate A's positions (`exp/jobs/e89.sh`, `E89_MODE=inv E89_ARM=awf`, a separate job) | 1.87 GB | only if exp89's label windows are purged |

**Duo pushes:** phase 1 = 2 (one ssh submits the whole chain; one scp fetches the tarball), phase 2 = 2, so 4 in all.
Each resubmission (`--from`) adds 1, and so does the separate exp89 job if gate A's positions must be re-extracted. The
pack jobs ask for Slurm mail at their end (`--mail-type=END,FAIL`, listed on aicr's Slurm page; no address is written in
the script, so it goes to the account's default). If no mail comes, the result arrives by the one scheduled fetch,
planned for after the expected finish; nothing polls.

## Timing and purge-bound inputs

aicr's scratch purges files older than 30 days; exp89's extraction showed the age is the file's date, not its last
read. Each input below is checked by the env job, which refuses with the replacement named.

| Input | Where | Written | Purged about | Needed for | Replacement |
|---|---|---|---|---|---|
| Ai2's request geometry | `$SCRATCH/deploy` | 8 Oct | 7 Nov | the draw (select) | `E111_FETCH=1`: KB from GitHub, sha256 checked |
| exp98's deployment dataset (2023) | `$SCRATCH/deploy` | 8 to 9 Oct | 7 to 8 Nov | A′, item identity, the 2023 time-range check (diagnostics) | none at a reasonable cost (about 87 GB); the diagnostics are skipped |
| exp99's deployment dataset (2017) | `$SCRATCH/deploy` | 9 to 10 Oct | 8 to 9 Nov | item identity, the 2017 time-range check (diagnostics) | none at a reasonable cost (about 90 GB); skipped |
| exp89's AWF label windows | `$SCRATCH/data/exp89` | 10 Oct (re-extracted) | 9 Nov | gate A's point positions | `e89.sh` inventory job: 1.87 GB tar, one more submission and push |
| exp102's rslearn venv | `$SCRATCH/e102/venv` | 10 to 11 Oct | 9 to 10 Nov | every rslearn step | `E111_BUILD_VENV=1`: about 4 GB from PyPI, pinned |
| the v1.2 checkpoint | `$SCRATCH/e102/runs` | 11 Oct (job 1277383) | 10 Nov | v1.2's predictions | the public copy, 1.16 GB |
| v1's checkpoint and both base weights | `$SCRATCH/hf` | not recorded: first cached in September, read again 8 to 11 Oct | by 30 October at the latest, possibly already (the purge goes by the file's date, a September one) | every prediction | `E111_HF_OFFLINE=0`: 1.04 GB and about 0.8 GB, pinned; likely needed for any run after about 20 October |
| the e111 roots themselves | `$SCRATCH/e111_private` | at prepare | 30 days after prepare | phase 2 | prepare again (the Planetary Computer transfer again) |
| the TimeSync file, Natural Earth | `~/olmoearth_inferenceX/data/breadth` | 9 Oct | not purged (/home) | the draw | `E111_FETCH=1` |

- **When to submit phase 1:** 15 to 18 October, or 21 October to 5 November. Maintenance is the third Tuesday,
  20 October 05:00 to 17:00 ET (and 17 November); a job whose limit crosses it stays pending, so a chain submitted on
  19 October could wait a day.
- **Before about 5 November** keeps every diagnostic input above, so that a gate failure can be diagnosed against the
  old windows, and keeps the v1.2 checkpoint and the venv without a download. It does not keep v1's checkpoint and the
  base weights: after about 20 October, plan on the 1.04 GB and 0.8 GB refetch (the env job refuses with it named).
- **Phase 2** must run within 30 days of prepare, and before about 9 November or with the venv and checkpoint
  downloads above.

**Risks.**

- **Planetary Computer.** It may throttle, and its 2017 items may have been reprocessed (gate B would show it).
  Prepare runs three passes:
  - pass 1 stops the job if more than 2% of a root's windows fail (a systematic failure);
  - pass 2 retries;
  - after pass 3, failures beyond the failure allowance stop the job; `--from prepare` resumes it.
  - With the allowance at its confirmed 0, a window that always fails (a corrupt or missing asset) blocks the test until
    the allowance is changed, which is a commit to `e111.sh` and a gate rerun.
- **What the allowance covers, in order.** Above 0, prepare_root runs these steps in this order:
  - The SCL sidecar takes only the windows with all twelve groups, so a failed window gets no SCL to compare.
  - The SCL materialize ignores a window's error, and its failures are added to pass 3's. Prepare's, materialize's
    and the SCL's failures together must stay within the allowance, over the root's windows.
  - The precheck then moves the input layers and sidecar of each partial window, and of each window whose SCL failed,
    into a quarantine directory of that job's own. Those windows count as no input, by cell.
  - Only then does the SCL check compare the sidecar with the windows that have input.

  rslearn 0.1.17 marks a window's twelve groups completed together, after writing them, so a failed window usually has
  none, and pass 3's count holds it. A window left partial is not repaired by later passes, because rslearn skips a
  window whose first group is completed; the precheck counts it from the files. A later `--from prepare` fetches a
  quarantined window again. The inventory then
  counts it as fetched, not quarantined, and a second quarantine goes to a new directory. The local smoke runs
  `e111.sh`'s own prepare_root in this order, with rslearn stood in for: a partial window, a failed window, an SCL
  failure, the allowance's sum, the allowance at 0, and both kinds of rerun.
- **A phase 1 near the purge** (after about 7 November). The purge removes files one by one. A partly purged 2017
  deployment makes the gate17-cell flag fall back to the own-cell rule (a lower bound, recorded with the reason). The
  time-range check counts the windows it cannot read and checks the rest, and item identity becomes a note. None of
  this fails a job. Before about 7 November all of it runs whole.
- **rslearn 0.1.17's defaults** may differ from 0.0.27's. The time order is pinned and the gates check the rest.
- **The v1 checkpoint under 0.1.17.** It loads (the env job's strict load, on CPU, before any imagery), with the
  formula's temporal table instead of its own (Pins, amendment); the gates judge the difference, and the next routes are
  named there.
- **Retries.** Every job fails fast on the first systematic error, with no silent retry loop and no polling from the
  Mac. A resubmission from a later step (`--from prepare|gate|gatecheck`) costs one Duo push and repeats nothing that
  succeeded: select keeps an equal draw and roots, prepare resumes, and only the gate and gatecheck replace their own
  outputs. `--from gatecheck` is refused when a guarded file changed since the gate job's commit, or when the gate's
  records are not one job's (a gate that died part way); then `--from gate`.

## Not part of exp111

- **exp109's Stage 0** (the full softmax at the 271 and at exp98's points, before the purge) is a separate decision.
- **Part B**, AWF-protocol labels at the contested plots, is the only route to whose boundary it is. It needs Ai2's
  annotators.
- **R3, the flip count across extra years** (above): not in this test; the owner left it out on 15 October 2026.
- **No model fix and no tool change before the evidence.** Per the owner's tool-scope rule, nothing found here enters
  the package unless a later preregistered test supports it.

## Decisions for the owner

The owner confirmed each default on 15 October 2026. Each is binding at the phase-1 commit, the commit phase 1 runs
and the gate passes with. A change after that commit and before freezing costs what the last column says; after
freezing, nothing here can change (a change would be a new preregistration).

| Decision | Confirmed | Where it is fixed | A later change costs |
|---|---|---|---|
| The prediction bars: P1 (a) STRICT below 0.70 with its 90% interval above 0.50, (b) LENIENT − STRICT above 0, LENIENT as a call; P2(a) at least 0.50; P2(b) the gap above 0, STRICT-only at most 0.65, LENIENT errors at least 0.65; P3 P(note) at least 0.5 at 20 labels, α = 0.05 | as written above | `exp111_grade.py` `BARS`, `P3` | a commit to a guarded file, so the gate is rerun: `e111_submit.sh phase1 --from gate` (gate up to 2.5 GPU-hours on b200-devel, gatecheck up to 8 core-hours, pack1; 1 Duo push to submit and 1 to fetch). The imagery is kept. |
| The floors: 720 plots with input; 40 errors and 40 correct per rule; 25 confident STRICT errors and 30 STRICT-only; 30 plots in the zone | as written above | `exp111_grade.py` `FLOORS` | the same gate rerun |
| The gate: A1 to A4, B1 and B2 (98% the same class, at most 2% with \|Δp1\| above 0.05, Spearman at least 0.95, 95% of the confident zone kept), floors 240 and 250 | as tabled | `exp111_blind_transfer.py` `GATE_BARS` | the same gate rerun |
| The draw sizes | 300/300/300, seed 111 (the alternative: exp109's Option A, 300/300/150) | `exp111_blind_transfer.py` `SIZES`, with the 720 floor (80% of them) in `exp111_grade.py` `FLOORS`; not a submission option (`E111_SIZES` is refused) | a commit to both guarded files, so the gate is rerun, plus `E111_REDO=select` (before freezing only): a new draw and a new blind17 root, its imagery fetched again (prepare: about 1 to 3 h at 32 CPUs, 30 to 90 GB transferred); 2 Duo pushes. A draw of other sizes than the commit's is refused by the phase-2 guard, the units and the grade, before any blind prediction |
| The failure allowance after prepare's third pass | `MAX_FAILED_SHARE` = 0; at most 0.005. It bounds the sum of prepare's, materialize's and the SCL materialize's failures, and the share of windows the precheck quarantines (partial, or SCL failed) | `e111.sh` | a commit to a guarded file: `--from prepare` (which resumes and quarantines within the new allowance) and the gate rerun; 2 Duo pushes |
| R3, the flip count across extra years | left out | no code exists | its own extension: 2016 and 2018 roots (about double the imagery and prepare time), its own reader, smoke and bars, all committed before the phase-1 commit; added later, it is new guarded code and so a gate rerun, plus its prepare |

Also confirmed by the owner on 15 October 2026, not fixed in code:

1. **The pipeline:** rslearn 0.1.17 for both models, on b200-devel for gate and predict, with the 0.0.29 fallback for v1
   if its checkpoint does not load. On 15 October 2026, after the second env job, the owner also accepted that v1 runs
   with the formula's temporal table instead of its own, the gates judging the difference (Pins, amendment). After gate
   A failed for v1 on that route (16 October), the route stays the formula until the owner chooses one of the candidates
   the report-only diagnostic compares (Pins, amendment of 16 October).
2. **Cluster runs and downloads:** the plan above, phase 1 to be submitted by 18 October; each submission and each
   fetch is still confirmed at the time, with the costs above stated again together with /home and /scratch usage;
   the downloads in the table above, each only where marked (v1's checkpoint and the base weights are likely after
   about 20 October).
3. **R1's CGLS crosswalk and R2's bands:** as written, report-only (`ARM_W` is in `exp111_grade.py`, so it is fixed
   at the phase-1 commit like the bars).
