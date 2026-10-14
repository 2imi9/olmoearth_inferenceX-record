# A re-run of Ai2's FT-AWF deployment configuration, graded at Ai2's own AWF labels (exp98 preregistration)

**Status: frozen on 9 October 2026, before any prediction was read at a label point.** Written 8 October 2026; floors, P1 and P5 revised the same day after a review, before any run. The owner confirmed the thresholds and the floors on 8 October 2026. Parts H (burned area) and I (East Africa TimeSync) were added before freezing and are report-only. At freezing, the full-area map was not yet written (inference queued); the pilot read 4 windows and counted pixels only, reading no label.

**Result, 9 October 2026 (added at recording; the frozen text is unchanged).** Run once at the label points at 7417425, recorded after an independent audit ([comparisons](../results/comparisons.md#a-re-run-of-ai2s-ft-awf-deployment-graded-at-ai2s-own-points-exp98)). V holds 259 validation points in 20 tasks, with 33 errors; the floors are met. P1 does not hold: the 90% interval of the difference, −5.1 to +0.8 points, reaches 0.15 points past the ±5-point margin (point difference −2.3 points, sign test p = 0.26). So reproduction within 5 points was not shown, and no difference was shown either. P2 (91.9%), P3 (AUROC 0.849), P4 (42.4%) and P5 (mean percentile 64.7%, lower bound 61.4%) hold. Changes after freezing are listed under Deviations.

**What is graded.** The map here is the record's own re-run of Ai2's AWF deployment configuration (Ai2's checkpoint,
`olmoearth_run.yaml`, request geometry and 2023 period, at 10 m), with two changes so that it can be graded: the model
writes its softmax (`output_probs`, ten float32 bands p0 to p9) instead of the argmax id, and a sidecar reads the
Sentinel-2 scene classification (SCL) for the condition layer (`exp/jobs/e98_config/CHANGES.md`). Its scenes are fetched
from Planetary Computer in October 2026. It is **not** the map Ai2 published (its viewer output for the Amboseli
region), and nothing here measures that map. "The deployed map" below is short for this re-run.

- **Allowed before freezing:**
  - the smoke on synthetic inputs (`python exp/exp98_awf_deployment.py --smoke`, `tests/test_exp98.py`);
  - the inventory (`--inventory`). It counts the labelled points inside the deployed area by split and says how
    each was mapped to a pixel. It also reads the label windows' time ranges. Given the deployed rslearn dataset, it
    also checks whether the deployment read the same 10 m band values at the label pixels, and how the two
    datasets' Sentinel-2 settings differ. It reads whether a label pixel holds a prediction. It never reads which
    class the map gives there, or how confident it is;
  - the burned-area layer (`exp/exp98_burned.py`, job `exp/jobs/e98_burned.sh`) and its `assess --condition` check
    on one part. They read MODIS and the map, never a label pixel. Part H itself, which reads the burn code and the
    outcome at the points, runs only in the run;
  - Part I's counts in the inventory: how many East Africa TimeSync plots lie inside the request geometry, in a grid
    and on a covered pixel, by country. The map's class at a plot is read only in the run.
- **Not allowed before freezing:** any class, probability or condition code read at a label pixel, and anything
  computed from them. The script refuses the run until this page says frozen.
- **Open before freezing:**
  - the floors (200 validation points, 15 errors), set on the in-area count below; the inventory confirms the count
    on the deployed windows;
  - the thresholds of P1 to P5;
  - the deployment's own settings, written here when the cluster jobs that run olmoearth_run
    (`exp/jobs/e98_env.sh`, `e98_prepare.sh`, `e98_predict.sh`, `e98_collect.sh`) are pinned and have run:
    - the revisions of olmoearth_projects, rslearn and olmoearth-runner;
    - `output_probs: true`, with `prob_scales` unset;
    - the writer's layer of ten float32 bands, p0 to p9;
    - where the SCL comes from (a band set or a sidecar layer), so that the condition layer records cloud as well as
      coverage.

## The question

The owner asked for Ai2's deployed FT-AWF map to be graded "using their previous labels": the expert points of
Ai2's AWF dataset. Ai2's published map holds argmax ids, not probabilities, so the record re-runs Ai2's deployment
configuration with probabilities written, and grades that re-run.

Two numbers exist. Both are evaluations, not a deployment.

- **Ai2's 89.5%** on its 344 validation points. Each point is read from its own 63-px label window, cropped to
  16 px around it.
- **The record's replica, 88.1%** on the same points (exp89 arm A, 41 errors). It reads each point from its label
  window too, in a 16-px crop centred on it.

A deployment, and so this re-run, is a different pipeline from the evaluation:

- olmoearth_run cuts the AWF project's request geometry into 1,024-px UTM windows.
- It fetches 12 monthly Sentinel-2 mosaics for each window.
- It runs the model on 16-px patches that overlap by 4 px, and stitches them into one map.

The same model can therefore give a different class at the same point. The crop around the point differs, the point
sits elsewhere in the crop, and the scenes were fetched again.

Eight questions:

1. **Part A.** Does the re-run of the deployment reproduce the evaluation at Ai2's validation points?
2. **Part B.** Does the deployed map's own confidence rank its errors there?
3. **Part C.** At those points, do the errors differ by the input condition the condition layer records?
   Descriptive.
4. **Part D.** How many of Ai2's points fall inside the deployed area? Counts only.
5. **Part E.** Do the points sit where the map is typical, or in its confident part?
6. **Part F.** What would an answer for the whole map cost? Report-only.
7. **Part H.** Where MODIS mapped a burn in 2023, is the map less confident, and are Ai2's points wrong more often?
   Report-only, no verdict. (Part G, the inputs at the label pixels, is a check, not a question.)
8. **Part I.** At the plots of an independent random sample inside the request geometry (the East Africa TimeSync
   sample, 47 plots), how often does the deployed 2023 map disagree with their 2017 labels? Report-only, no verdict.
   Added 9 October 2026, before freezing and before any map was read.

**Why this matters to the OlmoEarth team.** Ai2 publishes 89.5% beside the model, and its deployment pipeline is what
makes the maps users see. exp98 answers three things, about that pipeline as the record re-runs it:

- whether a map made by the deployment configuration keeps that accuracy at the points it was measured on;
- whether the map's confidence can direct a review;
- where Ai2's validation points sit in the map, which bounds what 89.5% says about the map as a whole.

## What is known before the run

Each fact, with its source. Revisions: olmoearth_projects @ f3c9b0c8, rslearn @ 5780896a, the AWF dataset
`allenai/olmoearth_projects_awf` @ da8eb6aa.

**The labels.**

- **1,459 label windows: 1,115 train and 344 val.** All are in EPSG:32737, all 63 x 63 px, and all carry the time
  range 2023-01-01 to 2023-12-31 (364 days). Each holds 12 Sentinel-2 item groups. Each records the options `split`
  and `source_task_id`.
  - Source: `exp/out/exp89_inventory_awf.json`, exp89's inventory of the pinned tar.
  - Ai2's doc counts 1,469 labelled points. The ten not in the tar are not explained.
- **The validation labels by class** (same source): shrubland/savanna 116, grassland/barren 72,
  agriculture/settlement 56, woodland forest 45, urban/dense development 27, open water 11, montane forest 10,
  herbaceous wetland 6, lava forest 1.
- **How the labels were made** (olmoearth_projects `docs/awf.md`):
  - AWF's experts annotated the points, using Planet imagery as the main reference.
  - Each point carries the time range 2023-01 to 2023-12.
  - The split hashes a 128 x 128-px grid into train (75%) and val (25%). The 128-px grid is also in
    `olmoearth_run.yaml`'s SpatialDataSplitter.
- **The label windows' preparation** (`olmoearth_run.yaml`): a window buffer of 31 px, so 63 px at 10 m, with label
  nodata 9.

**How Ai2 measured 89.5%.**

- The model (`model.yaml`) is OlmoEarth v1 Base at patch 4, with a decoder that upsamples x4 and then applies a 1x1
  convolution from 768 to 10 channels. The SegmentationTask has 10 classes and nodata 9, so channel 9 is never
  trained.
- Validation applies the default transforms: Pad(31, center), then Crop(16).
  - Center Pad crops the 63-px window to pixels 16 to 46, so the label sits at index 15
    (`rslearn/train/transforms/pad.py`).
  - Crop with no offset draws its start at random from 0 to 14 (`rslearn/train/transforms/crop.py`). The labelled
    pixel is therefore always inside the crop, at a random position from 1 to 15 in each axis.
  - So all 344 points count in Ai2's figure, each read at a random position in its crop.
  - exp89's replica centres each point instead, at row and column 8.
- The checkpoint was chosen on this validation accuracy: ModelCheckpoint keeps the top one by `val_segment/accuracy`.
  So 89.5% is a selected figure, and somewhat optimistic.

**How the map is deployed** (`olmoearth_run.yaml`, `dataset.json`, `model.yaml`,
`prediction_request_geometry.geojson`):

- **The request geometry** is one polygon of about 19,900 km² (about 150 x 133 km), in one UTM zone.
  - Its time properties run from 2023-01-01 to 2023-12-31.
  - The size and zone were computed locally from Ai2's file; no coordinate is recorded.
  - Ai2's doc calls its published inference output "the Amboseli national park region".
- **Partitioning:** a 1° grid, then 1,024-px windows at 10 m in UTM.
- **Inputs:** Sentinel-2 from Planetary Computer, with `harmonize: true`, `sort_by: eo:cloud_cover`, 12 periods of
  30 days (min and max matches 12) and `PER_PERIOD_MOSAIC`.
  - The deployment dataset declares one band set of all 12 bands at 10 m.
  - The training dataset stores its bands at their native resolutions: B02, B03, B04, B08 at 10 m;
    B05, B06, B07, B8A, B11, B12 at 20 m; B01, B09 at 60 m (exp89's inventory, `band_sets`).
  - So the 20 m and 60 m bands reach the model by different paths in deployment and in evaluation. How rslearn
    resamples in each case is not checked here.
- **Prediction:** all patches, 16 px, overlap ratio 0.25. The RasterMerger's `padding: 2` means a 4-px overlap
  (`rslearn/train/prediction_writer.py`): 2 px are trimmed from each side of a crop that is not on the window's edge,
  and later crops overwrite earlier ones.
  - So a pixel of the map is read at a position in its patch set by the grid. That is like Ai2's random crop and
    unlike the replica's centred one.
- **The output layer:** by default `dataset.json` declares one float32 band. A SegmentationTask writes argmax ids
  unless `output_probs: true` (`rslearn/train/tasks/segmentation.py`:54-55, 107-110, 198-232). The deployment jobs
  set it, with a 10-band layer.

**What the record measured on the evaluated model.**

- **exp89 arm A** (`exp/out/exp89_summary.json`):
  - accuracy 88.1%, 41 errors in 344 points; the errors fall in 18 of the 30 annotation tasks;
  - the top-1 confidence over the trained channels ranks the errors with AUROC 0.867;
  - a 10% review holds 41.5% of the errors, against 9.9% for random and a ceiling of 82.9%;
  - no point is predicted as channel 9;
  - in-sample accuracy on 512 training windows is 98.0% (`exp89_s2_awf.json`).
- **exp21** (`exp/out/exp21_finetuned_awf.csv`, computed 8 October 2026):
  - the replica's class changes between 16-px and 32-px crops at 22 of 344 points (6.4%), with 41 and 42 errors;
  - it changes under 0 to 3 px shifts at 18 points (5.2%);
  - reading the containing patch's logits instead of the interpolated pixel gives 89.8% against 88.1%;
  - 63% of errors and 34% of correct points sit on the boundary of the model's own predicted classes.

**The points inside the request geometry** (computed 8 October 2026, counts only; no position is written). The
window centres in `exp/out/exp89_units_awf.npz` against Ai2's `prediction_request_geometry.geojson` at f3c9b0c8, one
axis-aligned rectangle in one UTM zone:

- **250 of the 344 validation points** fall inside it, in **20 of the 30 annotation tasks**; 11 of those tasks hold a
  replica error. Task sizes run from 4 to 40 points.
- With a 0.1° margin (Ai2's doc: inference runs on every 1,024-px cell that intersects the geometry, so the windows
  reach past it by up to 10.24 km), **260 points** in 21 tasks.
- **The replica on the 250 in-area points:** 226 right, **24 errors** (90.4% accuracy); AUROC 0.877; the least
  confident 10% holds 41.7% of the errors (10 of 24). With the margin: 28 errors in 260 points.
- So V, the graded set, is expected to hold about 250 to 260 points in about 20 tasks, and about 24 errors if the
  re-run keeps the replica's accuracy. Every rationale below is computed on this set, not on all 344 points. The
  inventory counts the points on the deployed windows' covered pixels, which can differ by the margin and by
  uncovered pixels.

**The time.**

- The label windows and the request share the period 2023-01-01 to 2023-12-31, so both fetch the same 12 periods.
  The year matches.
- Not known: the date of the Planet imagery each expert annotated from. The windows do not hold it. Ai2's
  `annotation_task_features.geojson` may, and it is not read here.
- A point whose ground changed within 2023, or whose annotation was made from imagery of another year, cannot be told
  from a model error.

**The TimeSync plots inside the request geometry (Part I)** (counted 9 October 2026 from the sample's 2017 rows;
`docs/plan/awf_transfer.md` describes the sample and the crosswalk):

- **47 plots: 33 in Kenya, 14 in Tanzania.** Within each country the sample is a simple random sample of 2,000 locations
  (Bullock et al. 2021), so the plots of a country inside the geometry are a simple random sample of its part of the
  geometry, given their number.
- Their 2017 labels: Open Grassland 26, Wooded Grassland 11, Cropland 6, Dense Forest 2, Otherland 1, Open Forest 1. One
  plot carries two labels over 1985 to 2017, none within 2015 to 2017.

## The labels' design, and what follows

- **The points are not a probability sample of the deployed map.**
  - AWF's experts placed them in annotation tasks, presumably where a class could be identified on Planet imagery.
  - Their selection probabilities are unknown, so no point can be weighted to stand for a share of the map.
  - The validation points cluster in 30 annotation tasks.
- **The split is spatial but coarse.** Validation cells are 1.28 km cells hashed apart from training cells, and they
  border training cells.
- **What the labels support:**
  - **(a)** the deployed map's accuracy at Ai2's validation points, comparable with Ai2's 89.5% and with the
    replica's 88.1% on the same points. This says whether the deployment pipeline reproduces the evaluation, and only
    at those points;
  - **(b)** how well the deployed map's own confidence ranks its errors at those points: the AUROC, and the share of
    errors in the least confident 10% of graded points against random;
  - **(c)** the error rate at those points by the condition layer's code (coverage of the 12 timesteps, and SCL cloud
    where recorded). This is descriptive: few points per condition, and no design;
  - **(d)** how many validation and training points fall inside the deployed area. Counts only, and no position is
    written.
- **What the labels do NOT support:**
  - a whole-map error rate with a guarantee;
  - a certified zone;
  - per-class accuracy of the map, whether user's accuracy, producer's accuracy or area. Any per-class number here is
    a recall at the points;
  - anything about pixels unlike the points: transitions, mixed pixels, or places the experts did not label.
- **The training points are in-sample.** The model was fitted to them, so they are reported apart, never pooled and
  never graded.
- **Labels are taken as right,** as in exp89. An "error" is a disagreement with Ai2's label.

## How the run reads the files

`oe-inferencex from-olmoearth` writes three files per run:

- `scores_<EPSG>.tif`: ten float32 bands and NaN no-data;
- `condition_<EPSG>.tif`: int32, with -1 unrecorded;
- `olmoearth_conditions.json`: its `codes` name the condition codes.

exp98 reads these files. It does not import `oe_inferencex.olmoearth`, which another branch is writing.

- **The labels.**
  - A label is read with exp89's window list and single-pixel rule: exactly one pixel other than the fill 9, with a
    class from 0 to 8, in group `spatial_split`.
  - Where `oe_inferencex.awf` imports (the encoder extra), its `list_windows` runs on the same directory and must
    agree. The result is recorded.
  - If exp89's extraction manifest shows files lost to the scratch purge, the run refuses.
- **A label's pixel on the score grid (the CRS and pixel-centre convention).**
  - The label raster's own georeference gives the centre of the labelled pixel, (col + 0.5, row + 0.5), in the
    window's CRS. This is checked against the window's `metadata.json` (bounds in projection pixels times the
    resolution) to 1 mm.
  - That centre falls in exactly one score pixel: floor((X − X0)/a), floor((Y − Y0)/e) for the scores' north-up
    transform (a, 0, X0, 0, e, Y0). The point takes that score pixel.
  - When the CRSs differ, the centre is reprojected with `rasterio.warp.transform` and the point is counted as
    reprojected. exp89 found every label window in EPSG:32737.
  - rslearn writes label windows and deployed windows on integer pixel bounds of the same 10 m lattice of a CRS. So
    with equal CRSs the labelled pixel is a score pixel, and its centre is the middle of that pixel. The largest
    departure from the middle is recorded; it is expected to be 0.
- **Covered:** all ten bands finite and not all zero, zero being RasterMerger's fill. A point on an uncovered pixel is
  in the grid but not graded.
- **A map read in parts.** The reader holds at most 4 GiB of one grid, and this map is about 2 x 10⁸ pixels at 44
  bytes each, so `exp/jobs/e98_read.sh` reads every dataset root (olmoearth_run writes one per partition) in parts, each
  to its own directory, and no part holds two overlapping windows. exp98 reads all the part directories as one map:
  - the condition names are merged, and a code named differently in two parts is refused;
  - a point in two grids takes the first whose pixel is covered; the points in two grids, those covered in both, and
    those whose probabilities differ there are counted;
  - a map pixel covered in two grids on the same pixel lattice counts once (Parts C, E and F); an overlap that cannot
    be matched pixel for pixel is counted and reported.
- **Prediction:** the argmax over all ten channels, as rslearn takes it. A point predicted as channel 9 counts as an
  error, and is counted (exp89).
- **Confidence:** the top-1 softmax probability over the trained channels 0 to 8 (exp89's graded form).
  - A softmax restricted to a subset of its channels equals the full softmax renormalised over that subset, so the
    confidence is computed exactly from the written probabilities: max p_i / Σ_{i≤8} p_i.
  - The ranking reads `signals.confidence(form="top1")` on their logarithms, which is minus the log of that
    probability. This keeps the order where the probability rounds to 1.0. Float32 keeps the small probabilities down
    to about 1e-45.
  - Points where every other trained probability is 0 in float32 tie. They are counted.
- **Probabilities that do not sum to 1** within 1e-3 at the label pixels are refused (prob_scales, a rescaled head).
- **Clusters:** the annotation task, which is the window name before `_point_` (exp21's and exp89's 30 tasks).
- **The graded set V:** the validation points on covered pixels.

## Parts and measures

- **Part A, reproduction.** On V, exp89's replica (`exp/out/exp89_units_awf.npz`) is matched by window name. Its
  labels and clusters must equal the windows'; a mismatch is counted.
  - the deployed accuracy, the replica's accuracy on the same points, and their difference;
  - the share of points where the two give the same class;
  - the discordant points: deployed right and replica wrong, against the reverse. The exact sign test
    (`stats.sign_test`) is reported, with a cluster bootstrap of the paired difference
    (`stats.paired_cluster_bootstrap`, 2,000 resamples of the tasks, seed 98);
  - **P1's interval:** the two-sided 90% task-cluster bootstrap interval of the paired difference (2,000 resamples of
    the tasks, seed 98, both accuracies read on each resample);
  - the 2.0-point check that exp89's alignment used, on the point difference: reported, not graded;
  - Ai2's 89.5% beside it. It is matched only when all 344 points are graded, which this area does not allow (250 of
    344 inside it);
  - if Ai2 shares the argmax raster of its published run, the share of label pixels where it equals the re-run's
    argmax is reported (report-only). It is not public as a raster, and nothing here depends on it;
  - per-class recall and the confusion matrix at the points (`exp89.accuracy_block`); selective accuracy and ECE;
  - the boundary cue at the point (`signals.boundary_indicator` on the 3 x 3 argmax), errors against correct, beside
    exp21's 0.63 and 0.34. Reported.
- **Part B, ranking.** exp89's signal table on V (`metrics`):
  - AUROC, AURC, excess AURC and the share of the gap closed;
  - capture at 5%, 10% and 20%, against random and against the ceiling;
  - cluster bootstraps of the AUROC's lead over 0.5 and of the 10% capture's lead over random.
  - The replica's AUROC on the same points, with a paired bootstrap of the difference, and the Spearman correlation
    of the two confidences, are reported.
  - The package's own path, `assess.assess_prediction` at patch 1 with `--reference` the labels, is run on the points
    and reported. It predicts over the trained channels only, so it differs exactly where a point is predicted as
    channel 9.
- **Part C, condition (descriptive).** For each code at V's pixels: the points, the errors and the error rate, with
  the code's share of the map's covered pixels beside the points' share. -1 is "unrecorded".
- **Part D, inventory (counts).** By split: outside every grid, in a grid but uncovered, covered; same CRS against
  reprojected; points in two grids; the label windows' time ranges against the request's. The expected count is
  known (250 to 260 validation points); the inventory checks it on the deployed windows.
- **Part E, where the points sit.**
  - One pass over the map, strip by strip, counts its covered pixels, its deployed classes and its condition codes.
  - It keeps a Bernoulli sample of the confidence: 2,000,000 pixels in expectation, seed 98; every pixel on a smaller
    map.
  - **The points' percentile:** for each point in V, the share of the map's pixels less confident than it, ties
    counted half, averaged over V. It is 0.5 when the points' confidence is distributed as the map's. A task-cluster
    bootstrap of that mean (2,000 resamples, seed 98) gives its one-sided 95% lower bound, the 5th percentile.
  - Reported beside it: the map's and the points' confidence quantiles; the map's class shares against the points'
    deployed and labelled shares; the condition shares.
  - **Reported, never graded:** the points' error rate post-stratified to the map's confidence quintiles
    (`estimate.stratified_mean_and_variance`). It assumes that the points are as accurate as the map's other pixels of
    the same confidence, which their design does not give. It is withheld when a quintile holds fewer than 2 points.
- **Part F, what a whole-map answer needs (report-only).** `plan` on the map's covered pixels (patch 1):
  - the random labels for a 95% error-rate interval no wider than 10 points (±5). This is planned at the points'
    error rate, at exp89's 11.9% and at the default 50%;
  - certify at α 0.10 (δ 0.10) for zones of 50%, 80% and 100% of the map, at the error rate of the most confident
    share of V. That rate is a guess, since the points are not a sample;
  - the labels a zone needs inside it: 22 for one look, 34 for the sequential rule;
  - the sequential sample that would be labelled: `oe-inferencex sample scores_<EPSG>.tif --design sequential
    --alpha 0.1 --patch 1 --seed 98 --budget B`. It is a random order of the map's pixels, labelled from the top by
    AWF's experts on the same legend. `estimate` reads any labelled prefix, and `certify` holds at every look. The
    sheet lists positions, so it stays with the labellers and never enters the repository.
  - For orientation, at about 2 x 10⁸ pixels (computed 8 October 2026 with 1.8.0):
    - 211 random labels give ±5 points at 11.9% error, and 402 at 50%;
    - a map wrong about 12% of the time cannot be certified whole at α 0.10.
- **Part G, inputs (report-only, with `--deployment-dataset`).**
  - For each covered point whose label window and deployed window share a CRS: the four 10 m bands (B02, B03, B04,
    B08) of each of the 12 item groups at the label pixel, in the label window and in the deployed window.
    - These four bands are stored at 10 m in both datasets, so no resampling stands between them.
    - Equal values mean the deployment read the same scene there.
    - Item group k is compared with item group k. Matches in reversed order and in any order are counted beside it.
      rslearn's per-period mosaic can order its periods either way, and the same scenes in another order would show
      as different inputs. They would also mean that the model reads its months in an order it was not trained on.
    - These pixels are read from the deployment's scratch directory, which is purged after 30 days. Part G has to run
      within that time.
  - In the run only, the validation points' error rate and agreement with the replica, split by identical inputs or
    not.
  - The two datasets' Sentinel-2 settings, as a diff of their `config.json`.

- **Part H, burned area (report-only: no threshold, no verdict).**
  - **Why.** East African savanna burns every year, much of it in the dry seasons. A fresh burn scar is dark and
    bare for weeks, and the hypothesis is that the map reads it confidently wrong. In the AWF legend grassland and
    barren are one class, so burnt grass read as bare ground is no error; the errors to look for are burnt shrubland
    or savanna read as grassland/barren, and a dark scar read as one of the dark classes (open water, herbaceous
    wetland, lava forest). Where a model reads an input unlike its training it can be confidently wrong (exp88), and
    those errors come late in a review by confidence. A burn layer lets `assess` rank burned and unburned windows
    apart, and Part H says what the map and the points show by burn.
  - **The layer.** `exp/exp98_burned.py` (job `exp/jobs/e98_burned.sh`, after `e98_read.sh`) writes
    `burned_<EPSG>.tif` beside each part's scores, on the same grid, with `burned_conditions.json` naming the codes.
    Per 500 m MODIS cell, over the 2023 months read: **1** a burn date in any month; **0** unburned land in every
    month; **-1** unrecorded (no burn date, and water, unmapped or no tile in some month). Each 10 m pixel takes the
    code of the cell that holds its centre (nearest neighbour, every pixel transformed exactly). A code 2, "burned
    before a timestep the model read", is not written: each timestep is a 30-day mosaic built per pixel from several
    scenes by cloud cover, so the scene date at a pixel is not in the item groups' time ranges, and every burn
    before the last period is followed by later periods, so the code would hold nearly every burned pixel.
  - **The map (label-free).** Windows of 4 px (40 m, `assess`'s default), each with the code most of its covered
    pixels hold (`assess.pool_condition`; a tie or no code is unrecorded). Per code: the windows and their share;
    the quartiles of the deployed confidence (the top-1 trained probability, as a window's geometric mean, which is
    how `assess` reads a window with form `top1`); the code's share of the least confident 10% of all windows,
    beside its share of all windows (windows tied at the cut-off are split in proportion, so raster order does not
    decide). The codes' shares of the covered pixels. A window holding a pixel already covered in an earlier part is
    dropped and counted, so no area counts twice.
  - **Ai2's validation points (V).** Per code of the point's own pixel: the points, the errors, the error rate, the
    annotation tasks, and the code's share of the map's pixels beside its share of the points. Descriptive, with
    counts; no interval and no verdict.
  - **What it can show.** Whether burned windows are over-represented among the least confident (a review by
    confidence would reach them) or sit as confident as the rest (which a confidently mis-mapped scar would also
    give; labels are needed to tell), and, at the points, the error rate by burn with its count.
  - **What it cannot show.**
    - **Resolution.** A MODIS cell is about 463 m on a side: about 2,150 pixels of 10 m and 134 windows of 40 m. The
      code says whether the cell around a pixel burned, not whether the pixel did; a scar's edge mixes both codes.
    - **Detection.** MCD64A1 misses small burns: against Landsat 8 pairs its global omission error is 72.6% and its
      commission 40.2%, the omission mostly small burns (MCD64 Collection 6.1 user guide, section 7). Burns under
      persistent cloud go unmapped or late. Code 0 means no burn detected, not unburned.
    - **A missing month.** Planetary Computer holds no September 2023 item of the collection (checked 8 October 2026:
      268 items each for August and October 2023, none for September anywhere), though NASA's CMR lists the September
      granules. September lies in this region's long dry season (June to October), so burns that month read as code 0. The job
      allows that month missing and refuses any other; code 0's name says so
      (`mcd64a1:no-burn-2023:sep-not-read`), and every record lists the months not read. A local GeoTIFF of the
      month (`--item`, for instance converted from LP DAAC's granule, which needs an Earthdata login) closes the gap.
    - **Timing.** A burn date is a day; the model reads 12 mosaics of 30-day periods. A burned code does not say
      whether the scar was fresh in the scenes the model read, or green again by the later periods.
    - **The points.** They were placed by experts, presumably where a class was clear; few may fall in burned cells.
      A point labelled from Planet imagery of an unknown date can disagree with a 2023 scar for reasons that are not
      the model's.
  - **Source and licence.** NASA's MCD64A1 Collection 6.1 (Terra and Aqua MODIS, monthly, 500 m), from Microsoft
    Planetary Computer's STAC API, collection `modis-64A1-061`, assets `Burn_Date` (int16: day of the year of the
    burn, 1 to 366; 0 unburned land; -1 unmapped for lack of data; -2 water) and `QA` (uint8 bit field: bit 0 land,
    bit 1 valid data, bit 2 shortened mapping period, bits 5 to 7 why an unburned cell was classified so), signed
    with `planetary_computer` as `oe_inferencex/data.py` signs its reads. The stage checks Burn_Date's -1 and -2
    against QA's bits 0 and 1 on every cell read and refuses a disagreement on more than 0.1% of them. The
    collection's licence field says "proprietary" and links LP DAAC's data policies, which now lead to NASA
    Earthdata's data use guidance: data from a NASA-led mission are CC0 unless marked otherwise, a citation is
    requested, and no NASA endorsement may be implied. Cite Giglio, Justice, Boschetti and Roy (2021),
    doi:10.5067/MODIS/MCD64A1.061.

- **Part I, the East Africa TimeSync plots (report-only: no threshold, no verdict).**
  - **Why.** Ai2's validation points cannot give an area-wide rate (above). The East Africa TimeSync sample (Bullock et
    al. 2021, RCMRD and SERVIR; `github.com/bullocke/eastafrica` at fc2014fc, CC0) is the one probability sample with
    plots inside the geometry: a simple random sample of 2,000 locations per country, each labelled every year by local
    analysts from Landsat and Google Earth. Its labels end in 2017.
  - **The plots.** The 2017 rows of Kenya and Tanzania whose point lies inside Ai2's request geometry (47, above),
    each read at the score pixel that holds its point (reprojected from WGS84, as `locate_all` does), as exp98 reads a
    label point: the argmax over all ten channels, the top-1 trained confidence.
  - **The crosswalk.** TimeSync's nine classes against AWF's through `exp/timesync_awf_crosswalk.py`, fixed before any
    map was read and shared with exp99 (its table and reasons are in `docs/plan/awf_transfer.md`): STRICT maps each
    TimeSync class to one AWF class; LENIENT lets a class accept every AWF class whose definition overlaps its own, so
    its disagreements are a subset of STRICT's. montane_forest and lava_forest are accepted only by LENIENT (for the two
    forest classes); channel 9 never.
  - **Measures, under each rule.** Disagreements by country; the disagreement rate with the countries as strata, each
    weighted by its share of the request geometry's area (Natural Earth's admin-0 countries, v5.1.2, in EPSG:6933), and
    the package's exact interval (each country's exact hypergeometric interval at 1 - 0.05/2, weighted: at least 95%
    given the counts); Bullock et al.'s ratio estimator beside it; disagreements by TimeSync class and at the 3 x 3
    majority; the confusion counts; the confidence's AUROC for the disagreements, unweighted and weighted.
  - **What it cannot show.** These are 2017 labels against a 2023 map: a plot whose land cover changed between the two
    years counts as a disagreement (none of the 47 changed label from 2015 to 2017, which says nothing of 2018 to
    2023). A TimeSync label describes a Landsat-scale plot; the map is 10 m. Labels and map meet through a crosswalk,
    whose two rules bound its effect. With 47 plots the interval is wide: 7 of 33 and 3 of 14 disagreements would
    give about 6% to 45% at weights of 0.7 and 0.3 (the union over two strata costs width). Part I
    grades nothing and changes no grade; exp99 (`docs/plan/awf_transfer.md`) asks the same of a 2017 map over 309 plots.
  - **Inputs.** `exp/jobs/e98.sh` fetches the sample and the boundaries from their public sources, checks them against
    their pins (the sample's sha256, computed from the Mac's copy), and reads Ai2's geometry from the pinned
    `olmoearth_projects` clone; without them Part I is skipped with a note. It never stops the graded run: a failure is
    recorded with its traceback.

## Predictions

Thresholds are proposed, and the owner confirms them before freezing. P1 is two-sided; the others are one-sided.

**Floors.** Set on the in-area set above (about 250 validation points in 20 tasks, about 24 errors expected).

- P1, P2 and P5 are graded only if V holds at least **200** validation points: a margin below the 250 to 260
  available, so a few points on uncovered pixels do not cost the verdict.
- P3 and P4 also need at least **15** errors in V. About 24 are expected if the re-run keeps the replica's accuracy;
  fewer than 15 means an accuracy above about 94%, far on the favourable side of P1. exp89's 40-error floor is not
  used: on this set it would be reached only by a map that makes 16% errors or more, so whether P3 and P4 were graded
  would depend on P1 having failed.
- Below a floor, the value is reported without a verdict.

**The predictions.**

- **P1, the deployment reproduces the evaluation's accuracy (an equivalence test).** On V, the two-sided 90%
  task-cluster bootstrap interval of deployed accuracy minus the replica's on the same points lies within ±5.0 points.
  The graded value is the interval's larger end in absolute value.
  - Why an interval and not the point difference. On the replica about 6% of points change class with the crop
    (exp21), so the paired difference spreads by about 0.25/√n, 1.6 points at 250. exp89's 2.0-point check on the
    point difference would then fail by chance about one time in five (21%) with no real effect, and about one in
    three (32%) at the 10% discordance P2 accepts, more with the clustering in 20 tasks. A grade that close to a coin
    flip says little.
  - Why 5.0 points. A simulation on the 250 in-area points and their 20 tasks (no real effect, discordant points
    split evenly, 300 draws, 400 resamples each) gave a mean 90% half-width of 2.5 points at 6.4% discordance and 3.1
    at 10%. With a ±5.0 margin P1 holds about 85% and 64% of the time; with ±4.0, about 62% and 38%; with ±2.0, almost
    never. So P1 holding says the re-run's accuracy at these points is within 5 points of the evaluation's, at 90%
    confidence; P1 not holding says that was not shown, which is weaker than a shown difference. The sign test and the
    interval say which.
  - The point difference against exp89's 2.0 points is reported beside it. exp21's replica sat 1.4 points from Ai2's
    figure; Ai2's own table moves 1.0 point between 16-px and 32-px windows.
- **P2, the deployment gives the evaluation's classes.** On V, the deployed class equals the replica's at ≥ 90% of
  the points.
  - The replica's own class changes at 6.4% of points between 16-px and 32-px crops, and at 5.2% under 0 to 3 px
    shifts (exp21).
  - The deployment moves the crop by up to about 6 px and fetches the scenes again. 90% allows about 1.5 times the
    crop-size change. At 250 points and 93.6% agreement the binomial standard error is about 1.5 points.
- **P3, the deployed confidence ranks its errors.** On V, the AUROC of the deployed confidence for the deployed map's
  errors is ≥ 0.75.
  - The replica's is 0.877 on the 250 in-area points.
  - Its Hanley–McNeil standard error at 24 errors and 226 correct points is about 0.046, so 0.75 is about 2.7
    standard errors below. With so few errors the AUROC is uncertain; its task-cluster bootstrap is reported.
  - The question is whether the deployment keeps most of the evaluation's ranking, not whether it matches it.
- **P4, a small review finds many errors.** On V, the least confident 10% of points hold ≥ 25% of the deployed map's
  errors.
  - The replica's share on the in-area points is 41.7% (10 of 24), against about 10% for random.
  - 25% is 2.5 times random, and about 1.7 binomial standard errors below the replica (0.10 at 24 errors). At about
    24 errors one error moves the share by about 4 points, so P4 is a coarse check.
- **P5, the points sit in the confident part of the map.** The mean share of the map's pixels that are less
  confident than a validation point is ≥ 0.55, **and** its one-sided 95% task-cluster bootstrap lower bound is above
  0.5. It would be 0.5 if the points' confidence were distributed as the map's.
  - The rationale is a prior about how experts place points: where a class is clear on the imagery. It is not a
    measurement in the record, and is said so here.
  - 0.55 is the smallest departure worth reporting. The lower bound guards against chance: over about 250 points in
    20 tasks the mean's iid standard error is about 0.018 and clustering makes it larger, so 0.55 alone is only
    about 1.4 to 2.8 standard errors above 0.5.

No correction for multiple predictions. Each prediction is graded on its own, as in exp89.

## Analysis steps (the order the script runs them)

1. List the label windows and apply the label rule. Cross-check with `oe_inferencex.awf` where it imports, and check
   exp89's extraction manifest.
2. Open each part directory's `scores_<EPSG>.tif` with its condition layer; ten bands, north-up, the same grid.
   Merge the condition names across parts.
3. Map each labelled pixel to its score pixel in every grid that holds it (above), and take the first grid that
   covers it. Read the ten bands in a 3 x 3 block and the condition code.
4. Count by split and mapping (Part D). The inventory stops here, plus Part G's counts.
5. Refuse unless the page is frozen. Refuse probabilities that do not sum to 1.
6. On V: the prediction, the confidence, the errors, accuracy and recall, then Part A against the replica, Part B,
   the package's assess path, Part C.
7. The map pass, each pixel counted once across overlapping parts, then Part E.
8. Part F (`plan`).
9. Part G, if the deployed dataset is given.
10. Part H, if every part directory holds its burned layer (read in the map pass of step 7); otherwise a note. It is
    report-only and never stops the run: a burned layer that cannot be opened or read, at any step, leaves Part H
    out with a note, and any other failure is recorded in the summary with its traceback.
11. Part I, given the TimeSync sample, the boundaries and Ai2's geometry; otherwise a note. Report-only, and like Part H
    it never stops the run. The summary holds it as `timesync_reported` (the inventory's counts as
    `timesync_inventory`); no plot's identifier or position is written.
12. Grade P1 to P5 at the floors. Write `exp/out/exp98_summary.json` and `exp98_units.npz`. The units file holds no
    position: name, split, label, deployed class, confidence, condition code, burn code, boundary cue and input
    match.

The cluster job is `exp/jobs/e98.sh`. It runs on the CPU partition, chained with `--dependency=afterok` on
`exp/jobs/e98_burned.sh` (Part H's layer), which is chained on `exp/jobs/e98_read.sh`, which writes the
from-olmoearth part directories (after `e98_collect.sh`, and once the from-olmoearth reader is on main).

## What follows, whatever the outcome

- **P1 and P2 hold.** At Ai2's validation points the re-run gives the evaluation's classes and an accuracy within 5
  points of it. The record may say so: "at Ai2's N validation points inside the AWF request geometry, the record's
  re-run of Ai2's deployment configuration (2023, 10 m, scenes fetched in October 2026) is X% accurate (90% interval
  of the difference [a, b]), against Y% for the evaluated replica on the same points". This is a statement about
  those points and this re-run, not about the map, and not about the map Ai2 published.
- **P1 holds and P2 fails.** The accuracy is within 5 points, but the classes differ at more than 10% of the points. The
  map's class at a pixel depends on the pipeline (crop position, scenes) more than one accuracy figure shows. Part G
  says whether the inputs differed.
- **P1 does not hold.** Reproduction within 5 points is not shown. If the interval lies wholly outside 0 (the sign
  test agreeing), the re-run differs from the evaluation; if it straddles 0, the points are too few to tell. Part G
  separates input from context:
  - if the points with identical 10 m inputs keep the replica's accuracy and the others lose it, the scenes are the
    cause: mosaics fetched again, or the 20 m and 60 m bands resampled differently;
  - otherwise the crop position and context are the cause.

  It is reported to Ai2 as such.
- **P3 and P4 hold.** The deployed map's confidence ranks its errors at these points as the evaluation's did. The
  package's review order applies to the deployed map, at these points.
- **P3 or P4 fails.** A confidence that ranked errors in evaluation does not do so in deployment. That is a deployment
  effect worth reporting, and the review order is not recommended on this map without a probability sample.
- **P5 holds.** The validation points sit in the confident part of the re-run map. If points and map pixels of
  equal confidence were equally accurate, which this design cannot check, the map's accuracy would be below the
  points' accuracy, since confidence ranks errors (exp89). The reweighted figure is that conditional reading, reported
  as a model-based reading, not an estimate.
- **P5 fails.** The points are no more confident than the map's pixels. One reason to doubt that 89.5% transfers to
  the map goes away; no guarantee follows.
- **In every case,** Part F says what labels a whole-map answer needs. A sequential sample can be drawn from the
  deployed map for AWF's experts to label later.

## What would change the conclusion

- **The replica's labels or clusters disagree with the windows'** (`label_mismatches` > 0): Part A is void until the
  cause is found.
- **A graded point is reprojected, or sits off the middle of its pixel by more than 0.01 px:** P1 and P2 are also
  reported on the same-CRS, zero-offset points. The grade stays on V.
- **No point falls inside the deployed area, or too few for a floor:** the parts are reported without verdicts. If
  the inventory shows that, the owner may change the floors before freezing, never after.
- **Points covered in two parts with different probabilities** (`covered_in_two_differing` > 0): the grade stays on
  the first covering part, and the count is reported beside P1 and P2.
- **The condition layer is absent:** Part C is empty. Without an SCL band it records coverage only.
- **Part G shows different inputs:** the grades do not change, and Part G explains them.

## What it cannot say

- **Anything about the map Ai2 published.** The graded map is the record's re-run; Ai2's published map was made
  with other package versions and scenes fetched at another time (`exp/jobs/E98_README.md`, unknown 6).
- **The deployed map's error rate, with or without a guarantee.** It cannot certify a zone, and it cannot give
  per-class user's or producer's accuracy, or class areas.
- **Accuracy outside the deployed area, in another year, or at pixels unlike the points.**
- **Whether the labels are right.** They are taken as right. Points annotated on Planet imagery of an unknown date
  can disagree with 2023 Sentinel-2 for reasons that are not the model's.
- **Independence.** The points cluster in about 20 annotation tasks; intervals resample tasks. With about 24 errors
  the ranking measures are uncertain, and their intervals are reported.
- **Train points.** They are in-sample. And since the checkpoint was chosen on validation accuracy, the validation
  figure is optimistic too, as a selected statistic.
- **Conditions.** The layer records gaps and SCL cloud only, not haze, smoke or season. With few points per code it is
  descriptive.
- **Burned area.** Part H's layer is MODIS at 500 m, misses small burns and, from Planetary Computer, September 2023;
  it says where a cell burned in 2023, not whether a pixel was a fresh scar in the scenes the model read. Part H
  describes; it grades nothing.
- **The TimeSync plots.** Part I compares 2017 labels with a 2023 map through a crosswalk at 47 plots: it describes;
  it grades nothing.

## Deviations

None yet. Any change after freezing is listed here with its date, its reason, and whether any number had been read.

*Added at recording, 9 October 2026. The line above is the frozen text.*

- **9 October 2026, after freezing and before any label point was read: the burned layer's code.**
  `exp/exp98_burned.py` (Part H's layer only) now maps each 10 m pixel to the MODIS cell that holds its centre by
  transforming every centre with pyproj and flooring (7417425). It replaces GDAL's tolerance-0 nearest-neighbour warp,
  whose answer depended on the installed GDAL version: one test pixel in 43,200 moved a cell under GDAL 3.10.3, and CI
  failed on e928049. The rule on this page is unchanged ("each pixel takes the code of the cell that holds its centre,
  every pixel transformed exactly"); the code now follows it on every version. The burned rasters built at f934f73
  were rebuilt with it (06:18 to 06:27 UTC) before the inventory (06:30 UTC) and the graded run (06:38 UTC), both at
  7417425. P1 to P5 do not read the burned layer.
- **P5's map sample.** It holds 911,933 pixels, not 2,000,000 in expectation as Part E states: the sampling
  probability was set on the eleven grids' full area (515,899,392 pixels, with overlaps and uncovered pixels) rather
  than on the covered pixels counted once. The effect on P5's mean is negligible (Monte Carlo error about 0.0005 at
  most). Found by the pre-record audit; the grade is unchanged.
- **Interpretation (says less than this page).** The "P1 does not hold" paragraph's attribution through Part G is not
  applied: no difference was shown, Part G's split is 258 points against 1, and Part G compares only the 10 m bands.
- **Run history (no deviation).** The first inventory job (1247543) refused before reading any label, because the
  scratch purge had removed files of exp89's extraction of the pinned AWF tar. The tar was extracted again (job
  1247661) and checked against exp89's manifest. The analysis ran once at the label points. At freezing the inference
  had started and not finished, rather than being queued: the predict job (1245574) wrote its check at 05:29 UTC, eight
  minutes after the freeze (05:21, 7b12dc4's commit time; the predict time is from the session's job log, not in the
  committed records); no label was read before the inventory at 06:30 UTC.
