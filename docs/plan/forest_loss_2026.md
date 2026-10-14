# A second task: Ai2's Forest Loss Driver model graded at 2026 MapBiomas Alerta alerts (exp104 preregistration)

**Status: frozen on 11 October 2026, before the model predicted any sampled alert.** Written 11 October 2026. The
owner confirmed the design, the thresholds and the floors below on 11 October 2026, as proposed. Before freezing: the
windows (696, checked by recheck job 1286733 at 7fd4995), the imagery (prepare job 1286734: all 696 windows hold the
four scenes before and the four after; 689 in the graded strata, against the floor of 500; Sentinel-2 fetched from
Planetary Computer on 11 October 2026, counted at 22:04 UTC) and the GPU smoke on two synthetic windows (job 1286735:
ten classes in Ai2's order, probabilities summing to one within 8e-8).

**Result, 11 October 2026 (added at recording; the frozen text is unchanged).** Predicted (1289592) and read (1289593)
at 4c2f6c2; graded at deace9d (1290492) after the audit found the units file's rows unshuffled (the first grade, 1289594,
had the same numbers); recorded after an independent audit
([comparisons](../results/comparisons.md#a-second-task-ai2s-forest-loss-driver-model-at-2026-mapbiomas-alerta-alerts-exp104)).
689 alerts graded, 71 disagreements. P1 holds on its point value (design-weighted AUROC 0.770, 90% interval 0.682 to
0.855, which contains 0.70); P2 holds (0.457 of the gap, 0.339 to 0.644). P3: 10.9% disagreement (exact 8.0% to 14.4%),
not comparable with Ai2's 76.1%. Post hoc: the 18 confusions between agriculture and mining are confident and rank at
0.400; the 53 others at 0.945.

- **Allowed before freezing:** the sampling frame and the sample (attributes only: code, driver, biome, dates, area,
  status), the windows and their imagery, the pipeline's smoke on synthetic or unsampled inputs.
- **Not allowed before freezing:** any prediction of the model at a sampled alert.

## The question

Every graded result of the verification layer on a deployed OlmoEarth model is for one task, land-cover segmentation
in southern Kenya (exp98, exp99, exp101, exp103). Does the model's own confidence rank its errors on a different task:
Ai2's Forest Loss Driver model, which classifies the driver of a forest loss event (one class per event, ten classes),
graded at independent labels drawn after the model was trained?

## The model

`allenai/OlmoEarth-v1-FT-ForestLossDriver-Base` (OlmoEarth v1 Base fine-tuned by Ai2 on Amazon Conservation's labels in
Peru, Brazil and Colombia; Ai2 reports 76.1% on 109 validation windows; its training config is dated 2025-11-04), with
Ai2's own configuration (`allenai/olmoearth_projects`, `olmoearth_run_data/forest_loss_driver/`): four Sentinel-2 L2A
scenes before the event and four after, 128-px windows in EPSG:3857, the classification task writing the ten class
probabilities (`prob_property`). It is run with `rslearn model predict` (rslearn 0.1.17), the imagery from Microsoft
Planetary Computer, as the record's other deployments read Sentinel-2. The prediction is the argmax of the ten
probabilities; the confidence is the top-1 probability.

## The reference: MapBiomas Alerta

MapBiomas Alerta publishes every validated deforestation alert in Brazil, each refined on high-resolution imagery by an
analyst who records its probable driver (CC BY). The record reads the published layer
`mapbiomas-alertas:dashboard-alert-shapefile` from MapBiomas's public GeoServer (no account).

- **The frame:** the published alerts whose biome is Amazônia (the model's training biome), detected from 1 January to
  30 June 2026 (after the model was trained, and with at least three months of Sentinel-2 after each event by the
  snapshot), as read on 11 October 2026: 4,113 alerts (3,565 agriculture, 380 garimpo, 9 mining, 7 roads, 151 others,
  1 without a driver). The frame's sorted alert codes hash to 12bf6fd243b27a9a (sha256, first 16 hex digits); the list
  stays on scratch and is never committed (an alert code locates an alert on MapBiomas's platform).
- **The crosswalk to the model's classes:** agriculture -> agriculture; garimpo and mining -> mining; roads -> road.
  Others, urban expansion, reservoirs, energy, natural causes and alerts without a driver have no model class and are
  outside the frame.
- **The sample (stratified, seed 104):** agriculture, a simple random sample of 300 of 3,565; mining (garimpo and
  mining), all 389; roads, all 7, reported, never graded (fewer than the floors). Each alert's weight is its stratum's
  size over its sample size (11.88 for agriculture, 1 for mining), recomputed over the alerts with a prediction (below).
- **The window:** centred on the alert polygon's centroid (as Ai2's deployment centres its windows), the event date
  the alert's `detected_at`, the before and after periods as Ai2's configuration sets them relative to that date (four
  scenes from 300 to 120 days before, four from 7 to 187 days after, the least cloudy first). For alerts detected
  after early April 2026 the after period was still open when the imagery was fetched, so the date it was fetched is
  recorded. At the build (11 October 2026), 221 of the 696 sampled alerts' centroids lie outside their own polygon
  (irregular or multi-part alerts), so the window's centre shows something other than the loss there; Ai2's deployment
  centres its windows the same way.
- **Readings fixed with the script (before freezing):** "biome Amazônia" means Amazônia is in the alert's biomes list
  (10 frame alerts list Amazônia and Cerrado), and "published" is the status; the frame's 4,113 include the 151 others
  and the alert without a driver, which are never sampled (the graded population is agriculture and mining, 3,954).
  Alerts without a polygon, a window or complete imagery are treated as missing at random within their stratum: each
  graded alert's weight is its stratum's size over the stratum's alerts with a prediction, and the missing are
  counted. The bootstraps resample every graded stratum, the mining census too (as exp99 resamples each country). Ai2's
  random Flip, which rslearn also applies at prediction, is left in place with the seed fixed (104). The per-alert
  rows of the units file are written in a random order whose permutation stays on scratch, so that they cannot be
  matched to MapBiomas's public alerts by their order. Each window's centre (the rings' area-weighted centroid, holes
  subtracted) is checked by a second route to that definition (pyproj, and shapely per ring) for every alert, by
  rslearn's projection for every valid polygon, and, for an invalid polygon, by the repaired polygon's centroid
  (make_valid, as rslearn repairs it, or buffer(0)) lying in the model's central 64-px crop. 343 of the 696 polygons are
  invalid (MapBiomas's multi-part alerts). The letter of Ai2's code, shapely's centroid of the unrepaired polygon,
  agrees on valid polygons but lies 2 to about 1,870 px from the area centroid on 24 invalid multi-part ones (job
  1286487); the windows keep the area centroid, Ai2's stated intent ("the
  centroid"), and those 24 are counted as a deviation from the letter of its code. OlmoEarth Studio's own window
  partitioner (`FixedWindowPartitioner`, olmoearth-runner 0.1.14, read on PyPI) takes a polygon through rslearn's
  `to_projection` and centres the 128-px window on the projected polygon's centroid: the rslearn route. rslearn
  0.1.17's `to_projection` repairs with make_valid, which fails on 2 of the sampled polygons, the kind of failure Ai2's
  deploy code avoids by sending centroid points (its docstring cites Studio job failures on complex polygons); the
  rslearn version Studio runs was not checked. The windows agree with that route within 2 px on every valid polygon
  and on 338 of the 343 invalid ones (job 1286487).

## Predictions (on the graded strata, agriculture and mining, design-weighted; confirmed by the owner)

- **P1 (ranking):** the design-weighted AUROC of the model's confidence for its errors is at least 0.70 (exp99's bar).
- **P2 (review):** the least confident 10% of the design weight close at least 0.25 of the gap from random to the
  ceiling (exp99's bar).
- **P3 (descriptive):** the design-weighted error rate with the package's stratified exact interval, each stratum's
  rate with its exact interval, and Ai2's 76.1% beside them; the confusion counts; the errors and the design-weighted
  error rate at the alerts whose centroid lies inside their polygon and at those whose centroid lies outside.

**Floors:** at least 500 alerts with imagery and a prediction, and at least 20 errors and 20 correct among them;
otherwise the predictions are reported without verdicts.

## What would change the conclusion

- **P1 and P2 hold:** the confidence ranks the errors on a second task, a per-event classifier, at labels drawn after
  training: the layer's ranking is not specific to the Kenyan land-cover model.
- **P1 fails, P2 holds, or the reverse:** as exp99, the ranking is weaker than a review can use or than the bar asks;
  both reported.
- **Both fail:** the confidence does not rank this model's errors here; the record says so beside exp98 and exp99.

## What it cannot show

- Burned, logging, landslide, hurricane, river, airstrip and none: MapBiomas Alerta does not record them, so the frame
  tests agriculture against mining (and roads, report-only).
- Whether an error is the model's or the reference's: MapBiomas's driver is an analyst's call on high-resolution imagery
  and is taken as right.
- Alerts outside the Amazon biome, or before 2026.

## Outputs

`exp/out/exp104_summary.json` and `exp/out/exp104_units.npz` (per alert: stratum, weight, label, prediction, the ten
probabilities; no alert code, no position). Jobs `exp/jobs/e104.sh` on the aicr cluster, pinned to a full commit.
