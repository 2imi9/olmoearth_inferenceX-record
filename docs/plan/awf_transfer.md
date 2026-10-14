# A transfer test: Ai2's FT-AWF deployment configuration on 2017 imagery, graded at an independent random sample around the AWF area (exp99 preregistration)

**Status: frozen on 9 October 2026, before any class or probability was read at a plot.** Written 9 October 2026, before any 2017 map existed. The area rule, the crosswalk and the design are fixed here. The owner confirmed the thresholds and floors below on 9 October 2026, as proposed. At freezing, the pilot's inventory had counted plots in windows only, and the full run's first prepare (job 1247349) had been cancelled for making 9,298 windows (see The request geometry, below); its rerun on one MultiPolygon request was in progress.

**Result, 10 October 2026 (added at recording; the frozen text is unchanged).** Run once at the plots, graded at
52489b1, and recorded after an independent audit ([comparisons](../results/comparisons.md#a-transfer-test-ai2s-ft-awf-deployment-configuration-on-2017-imagery-graded-at-an-independent-random-sample-exp99)).
271 of the 309 plots have input (42 of 313 windows got no 2017 imagery), with 91 STRICT errors; the floors are met.
P1 does not hold: the design-weighted AUROC is 0.638 (90% bootstrap 0.579 to 0.696); LENIENT's 0.678 also misses 0.70,
but its interval (0.599 to 0.756) contains it. P2 holds on the point value: 0.434 of the gap closed (interval 0.10 to
0.65, which includes 0.25). P3: 35.5% under STRICT (26.0% to 45.8%) and 13.4% under LENIENT (7.5% to 21.9%), call
"not determined". The 38 plots without input are not missing at random, so these figures describe the plots with 2017
input, not the 100 km region. Changes after freezing are listed under Deviations.

- **Allowed before freezing:**
  - the smoke on synthetic inputs (`python exp/exp99_transfer.py --smoke`, `tests/test_exp99.py`);
  - the area rule (`--select`, job `exp/jobs/e99_select.sh`): counts by country and by distance, the region's area by
    country, and the request geometry of squares around the plots, written to the cluster's scratch only;
  - the deployment jobs (`exp/jobs/e99_prepare.sh`, `e99_predict.sh`, `e99_collect.sh`, `e99_read.sh`), their pilot
    on 4 plots, and their label-free checks (windows, bands, sums, coverage, condition codes);
  - the inventory (`--inventory`, `exp/jobs/e99.sh` with `E99_MODE=inv`): how many selected plots fall in a window and
    on a covered pixel, by country.
- **Not allowed before freezing:** any class, probability or condition code read at a plot's pixel, and anything
  computed from them. The script refuses the run until this page says frozen.

## The question

exp98 grades Ai2's deployment configuration at Ai2's own validation points: expert-placed points with unknown
selection probabilities, so no area-wide error rate and no certified zone can follow from them. Independent reference
data inside the AWF area is scarce: the one probability sample with plots there, the East Africa TimeSync sample, has
47 plots inside Ai2's request geometry (exp98's Part I reads them against the 2023 map) and labels that end in 2017.

exp99 asks, with the model and configuration held fixed and the year and area moved to where an independent random
sample exists:

1. **Part A.** What is the error rate of FT-AWF's 2017 map over the region around the AWF area, at a probability
   sample, with an exact interval, and how does it compare with Ai2's 89.5%? (P3, descriptive.)
2. **Part B.** Does the map's own confidence rank its errors at that sample? (P1, P2.)
3. **Parts C to F**, report-only: errors by class, by input condition and at the 3 x 3 majority; where the plots sit
   among the windows' pixels; what `plan` says certify would need; the 47 plots inside Ai2's geometry on the 2017 map
   and on exp98's 2023 map.

The owner approved this design on 9 October 2026.

## What is known before the run

**The sample** (Bullock, Healey, Yang et al. 2021, *Land* 10, 150; RCMRD and SERVIR):

- In each of seven countries, a simple random sample of 2,000 locations. Local experts interpreted each in TimeSync from
  its Landsat history and Google Earth's high-resolution imagery, one land-cover label per year.
- File: `github.com/bullocke/eastafrica` at `fc2014fcbc55cb75e0fbf801ffd9ca1ba4dc5c02`, `yearly_point_data.csv`,
  26,312,839 bytes, sha256 `5309a982cdd45a4e7378008b9a6385b6accca7d071bdd652360ef7ef66fc429a`, CC0. The Mac's copy
  (`data/breadth/eastafrica_yearly_point_data.csv`, used by exp94) has GitHub's git blob id
  (`1713496168b9a4d1c2c1bedbd62dc7f6b72d7b61`). The cluster jobs fetch the file from GitHub and check it against that
  sha256; nothing is copied from the Mac.
- 33 years per plot (1985 to 2017 in the file); the 2017 rows are complete: 2,000 plots in every country, 0 missing.

**The area rule**, applied on 9 October 2026 to the 2017 rows of Kenya and Tanzania against Ai2's request geometry
(`olmoearth_projects` at f3c9b0c8, `olmoearth_run_data/awf/prediction_request_geometry.geojson`, sha256 `fee3ce01...`;
its position is never written):

- A plot's distance to the geometry is measured in EPSG:32737 (the geometry's UTM zone), with the geometry's edges
  densified every 0.001 degree first; 0 inside. D is the smallest multiple of 10 km whose buffer holds at least 300
  plots.
- Counts by D: 0 km 47, 10 km 59, 20 km 76, 30 km 97, 40 km 124, 50 km 153, 60 km 186, 70 km 218, 80 km 247,
  90 km 282, **100 km 309**.
- **D = 100 km: 309 plots, 219 in Kenya and 90 in Tanzania; 47 of them inside the geometry (33 and 14).**
- The geodesic distance (WGS84) gives the same counts at 80, 90, 100 and 110 km (389 against 390 at 120 km); the two
  distances differ by at most 0.12 km within 130 km. The nearest plots to the 100 km line lie 0.22 km inside and
  0.11 km outside it.
- `exp/exp99_transfer.py` and `e99_select.sh` refuse any other count.

**The region:** the 100 km buffer of the geometry, 107,679 km² (EPSG:6933). Each country's part is a stratum, weighted by
its share of the region's area, computed on the cluster from Natural Earth's admin-0 countries (v5.1.2, public domain;
`e99_select.json`). From the plots alone (the plots per country times the country's area over 2,000), the parts are about
63,500 km² in Kenya and 42,600 km² in Tanzania, 106,100 km² together, so the weights are about 0.60 and 0.40, and the
region lies within the two countries.

**The labels of the 309 in 2017** (the reference, read before any map): Wooded Grassland 138, Open Grassland 112,
Cropland 44, Otherland 8, Dense Forest 4, Open Forest 2, Open Water 1. 250 of the 309 are the two grassland classes, so
the crosswalk's rule for them decides much of the error rate. 14 plots carry two or more labels over 1985 to 2017, 2 of
them within 2015 to 2017. Of the 47 inside the geometry, 1 and 0.

**The windows:** 10 plots lie in UTM zone 36, 299 in zone 37. If olmoearth_run's 1,024-px window grid is anchored at the
CRS origin, the 309 plots fall in 263 distinct cells, so about 265 to 300 windows (a square on a cell edge adds one):
the size of exp98's run.

**What the record measured on this sample:** exp94 graded Copernicus CGLS-LC100 against the 2017 labels of all seven
countries in a seven-class legend that merges grass and shrub: 39.2% error, and its own probability ranked its errors
with AUROC 0.720 (10% capture 0.164 against a ceiling of 0.239). exp89's replica of FT-AWF at Ai2's 344 validation
points: AUROC 0.867, 10% capture 0.415 against a ceiling of 0.829.

## The design, and its limits

- **Within a country, the plots inside the region are a simple random sample of the country's part of the region,
  given their number.** The 2,000 plots of a country are a simple random sample of its area. A region chosen by
  geography (a distance to Ai2's geometry), not by label or map, is a fixed subset of that area; conditional on how
  many plots fall in it, every set of that many locations inside it is equally likely. So the plots of Kenya inside the
  region are a simple random sample of Kenya's part of it, and those of Tanzania of Tanzania's.
- **The countries are strata.** The region's error rate is sum_c W_c k_c / n_c, W_c the country's share of the
  region's area, k_c the errors among its n_c graded plots. The package's condition-design interval
  (`oe_inferencex.estimate`, the interval `estimate_error_rate` gives for strata) weights each stratum's exact
  hypergeometric interval at 1 - 0.05/2: at least 95% by construction, conditional on the n_c. The population is the
  region's 10 m pixels; a plot is read at the pixel that holds its point, which for a uniform point is a uniform pixel.
  Bullock et al.'s own estimator for a domain across borders, Stehman's (2014) ratio estimator over the countries'
  samples of 2,000, is reported beside it (normal theory, needs only the countries' areas).
- **The map at a plot is the region's map at that pixel,** if the window grid depends on the pixel alone: Ai2's doc says
  inference covers "all 1024×1024 grid cells intersecting the geometry", and on a grid anchored at the CRS origin a
  pixel's window, crop positions and scenes are the same whichever plots were requested. `e99_prepare.sh` records how
  many windows sit on that lattice; if some do not, the map at a plot also depends on which other plots were
  requested, and the run says so.

Limits, stated before the run:

- **Plots without input.** A window whose 2017 imagery lacks a 30-day period (Sentinel-2B joined Sentinel-2A only in
  mid-2017, so early-2017 periods are thinner) gives its plots no prediction. They are treated as missing at random
  within their country, which cannot be checked; their labels are reported. The floor below caps them at 59 of 309.
- **Frames.** Bullock et al.'s country frames and Natural Earth's boundaries need not agree at a border; the plots
  outside their own country's Natural Earth polygon are counted.
- **Scale.** A TimeSync label describes a Landsat-scale plot (30 m) interpreted with high-resolution imagery; the map is
  10 m. The 3 x 3 majority around the plot is reported beside the plot's own pixel.
- **Labels are taken as right,** as in exp89 and exp98.
- **The windows are a cluster sample chosen through the plots.** Only the plots are a probability sample; the windows'
  other pixels are described (Part D), never graded.
- **"The same landscape type"** is approximated by distance alone: the region also holds Kilimanjaro's and other
  forests, farmland and towns that the AWF labels cover sparsely.

## The map

The same configuration as exp98 (`exp/jobs/E98_README.md`): Ai2's FT-AWF checkpoint, `olmoearth_run.yaml`, exp98's
`model.yaml` and `dataset.json` (`output_probs: true`, ten float32 bands p0 to p9) and the SCL sidecar, in exp98's
deployment environment (olmoearth-runner 0.1.14, rslearn 0.0.27 with the `get_item_by_name` fix), reused, not rebuilt.
Two things differ:

- **The request geometry:** one square of about 11 m (0.0001 degree) around each of the 309 plots, in the structure of
  Ai2's file (a FeatureCollection holding one feature: here one MultiPolygon of the 309 squares, where Ai2's is one
  Polygon) with Ai2's property names, `oe_start_time` 2017-01-01 and `oe_end_time` 2017-12-31. olmoearth_run then
  predicts the 1,024-px windows that hold a plot. One feature per square, as first written, made olmoearth_run window
  each 1-degree cell once per plot in it (its GridPartitioner turns each feature into the whole cell, unclipped): the
  first prepare (job 1247349) made 9,298 windows for 309 plots and was cancelled before freezing; the prepare stops
  itself above 3 windows per square. The file holds positions: it is written on the cluster's scratch only, and only
  its sha256 is recorded.
- **The year:** 12 periods of 30 days from 1 January 2017, from Planetary Computer's Sentinel-2 L2A. The model was
  fine-tuned on 2023 imagery.

A pilot first runs the chain on 4 of the plots (seed 99). The jobs are in `exp/jobs/E99_README.md`.

## How the run reads the map

- Each plot's point is transformed into each score grid's CRS and takes the score pixel that holds it (exp98's
  `locate_all`, every plot "reprojected"). A plot in two grids takes the first whose pixel is covered.
- **Covered:** all ten bands finite and not all zero. A plot on an uncovered pixel, or in no window, has no input.
- **Prediction:** the argmax over all ten channels; a plot predicted as the untrained channel 9 is an error under both
  rules. **Confidence:** the top-1 softmax over the trained channels 0 to 8, read as exp98 reads it.
- Probabilities that do not sum to 1 within 1e-3 at the plots are refused.

## The crosswalk (fixed 9 October 2026, before any map; shared with exp98's Part I)

`exp/timesync_awf_crosswalk.py`, sha256 of its STRICT and LENIENT rules
`4c8b452896a880189fc51922037acbdb4abc21e05c496e0acc67000a860d1846` (`tests/test_exp99.py` fails on any change).
Definitions: TimeSync's from Bullock et al. 2021, Table 1; AWF's from Ai2's `docs/awf.md`, which names the nine classes
and defines only woodland forest (">40% canopy").

| TimeSync class (definition) | STRICT: the map is right only at | LENIENT also accepts | Why |
|---|---|---|---|
| Dense Forest (canopy over 40%) | woodland_forest | montane_forest, lava_forest | both legends put this class above 40%; AWF splits forest by setting, which a TimeSync forest on a slope or a lava flow can be |
| Open Forest (canopy 15-40%) | shrubland_savanna | woodland_forest, montane_forest, lava_forest | AWF has no class of 15-40% canopy; below its woodland forest's 40% its woody class is shrubland/savanna; a canopy near 40% reads differently to two interpreters |
| Wooded Grassland (below both thresholds, shrubs or sparse trees) | shrubland_savanna | grassland_barren | a savanna or shrubland by definition; AWF publishes no woody threshold between its shrubland and grassland classes. Not woodland_forest: below 15% against above 40%, disjoint |
| Open Grassland (no substantial woody vegetation) | grassland_barren | shrubland_savanna | the same unpublished boundary |
| Cropland (primarily agriculture) | agriculture_settlement | (none) | the only AWF class that names agriculture |
| Settlements (primarily built) | urban_dense_development | agriculture_settlement | both name built land; AWF puts homesteads among fields in agriculture/settlement |
| Open Water | open_water | herbaceous_wetland | the region's lakes and swamps are seasonal |
| Vegetated Wetland (vegetation and open water) | herbaceous_wetland | open_water | the same mix |
| Otherland (barren, rock, snow, beaches, salt crusts) | grassland_barren | open_water | AWF merges barren land with grassland; salt crusts of seasonal lake beds are water part of the year |

- STRICT's class is in LENIENT's set for every class, so LENIENT's errors are a subset of STRICT's; the two rates bound
  what the crosswalk can do.
- **AWF classes TimeSync lacks:** montane_forest and lava_forest are no class's STRICT counterpart, so under STRICT a
  prediction of either is always an error; LENIENT accepts both for the two forest classes. Channel 9 is never accepted.
- **TimeSync classes AWF lacks:** Open Forest (no AWF class of 15-40% canopy) and Otherland (no AWF class of its own).
- The 2017 labels of the 309 include no Settlements and no Vegetated Wetland; their rules are fixed all the same.

## Parts and measures

- **Part A, the error rate (P3, descriptive).** Under each rule: k_c and n_c by country; the stratified estimate with
  the condition-design interval (95%); each country's exact interval; the ratio estimate beside it; the unweighted rate.
  Beside Ai2's 89.5% (an error rate of 10.5%).
- **Part B, the ranking (P1, P2 on STRICT; LENIENT reported).** At the plots with input, design-weighted (weight
  W_c / n_c, the region) and unweighted (the plots): the AUROC of the confidence for the errors; capture at 5%, 10% and
  20% of the weight, beside random (the budget) and the ceiling (min(1, budget / error rate)); the share of the gap from
  random to the ceiling closed, (capture - budget) / (ceiling - budget); the lift (capture / budget); AURC and excess
  AURC. Bootstraps resampling plots within each country (2,000, seed 99) of the weighted AUROC and of the gap closed at
  10%.
- **Part C (descriptive).** Errors by TimeSync class, by condition code (the from-olmoearth condition layer: coverage of
  the 12 periods and SCL cloud), at the 3 x 3 majority; the confusion counts; the plots without input by country and
  label.
- **Part D, where the plots sit (report-only).** One pass over the windows' covered pixels (a pixel in two windows
  counted once, a Bernoulli sample of 2,000,000): the share of the windows' pixels less confident than a plot, averaged
  over the plots. About 0.5 if the plots sit in their windows as random pixels do, the converse of exp98's P5.
- **Part E, what certify would need (report-only).** `plan` on the region (its 10 m pixels, about 1.08 x 10^9): the
  random labels for an error-rate interval no wider than 10 points at each rule's rate and at 50%; for zones of 50%, 80%
  and 100% at alpha 0.10, the labels certify needs, at the error rate of the most confident share of the plots (a guess:
  the zone is defined on the region's own confidence ranking). numpy's hypergeometric sampler refuses populations of 10^9
  or more, so the zone is planned on 999,999,999 windows, which moves no budget. **Certify itself is not applied:** a
  zone is the most confident share of every window of the map, and a run over the windows that hold a plot does not
  give the region's ranking.
- **Part F, the 47 plots inside Ai2's geometry (report-only).** Their errors on the 2017 map under each rule; with
  exp98's 2023 map (`--exp98-scores`, read only once exp98's page is frozen), the paired table (both right, right only
  in 2017, right only in 2023, both wrong) with the exact sign test. Same plots, same labels, two years' maps: the
  difference mixes the imagery's year, the satellites available (Sentinel-2A alone until mid-2017) and processing, and
  land change after 2017, which counts against the 2023 map since the labels are from 2017. It is descriptive and does
  not apportion a gap; with 47 plots the paired counts are small. exp98's Part I reads the same plots against the 2023
  map in exp98's run.

## Predictions (thresholds and floors confirmed by the owner on 9 October 2026)

**Floors.** P1 and P2 are graded only if at least **250** of the 309 plots have input, and there are at least **20**
errors and **20** correct plots under STRICT. Below a floor the value is reported without a verdict.

- 250 leaves room for 59 plots without input (19%): a window short of 12 periods in early 2017 is possible, and missing
  plots weaken the design argument, so more than a fifth missing voids the grades.
- At 20 errors and 230 correct plots the AUROC's Hanley-McNeil standard error is about 0.06; fewer errors make P1 and P2
  coin flips. CGLS's 39% on this sample makes far more errors likely; fewer than 20 would mean an error rate under 8%,
  far better than Ai2's figure.

**P1, the confidence ranks the errors at an independent random sample.** The design-weighted AUROC of the confidence for
the STRICT errors at the plots is at least **0.70**.

- Every product confidence the record graded on a probability sample sits between 0.720 and 0.787 (exp94: CGLS on this
  very sample 0.720, LCMAP on NLCD's 0.72 to 0.74, ODSE on S2GLC 0.787). exp89's replica reached 0.867 at Ai2's own
  points and exp98's P3 asks 0.75 there. Here the model is read off its training year and area, and the errors include
  the crosswalk's boundary cases, so a lower bar is set: 0.70 is below every recorded value on a probability sample.
- At about 300 plots and 100 errors the AUROC's standard error is about 0.03, so 0.70 is about 1.6 standard errors below
  0.75.
- STRICT is graded because it is the one-to-one reading accuracy assessment uses; LENIENT is reported beside it.

**P2, a small review finds errors beyond chance.** Under STRICT, the least confident 10% of the plots (in design weight)
close at least **0.25** of the gap from random to the ceiling: (capture - 0.10) / (min(1, 0.10 / error rate) - 0.10).

- The owner's suggestion was "at least twice random's share of errors". At the error rates likely here it grades the
  error rate more than the ranking: capture at 10% cannot exceed 0.10 / error rate (the package's own
  `metrics.weighted_capture_at_budget` says so), so at CGLS's 42% on this sample twice random means closing 72% of the
  gap, which CGLS (AUROC 0.720) did not (0.46), and at 15% it means only 18%. The gap closed does not move with the
  error rate.
- Recorded values: CGLS on this sample 0.46, LCMAP on NLCD's 0.42 to 0.44, ODSE 0.28, exp89's replica 0.43 at Ai2's
  points. 0.25 is below all of them. "Twice random" (the lift at 10%) is reported beside it.

**P3, descriptive: the error rate beside Ai2's 89.5%.** No threshold. Called in advance:

- **a transfer gap** when even LENIENT's 95% interval lies wholly above 15.5% (Ai2's 10.5% error plus 5 points, exp98's
  equivalence margin);
- **no gap shown** when even STRICT's 95% interval lies wholly below 15.5%;
- otherwise **not determined**.

How precise this is, computed before the run with the plots spread as the rule gives them (219 and 90, weights about
0.60 and 0.40): the interval is about 31% to 50% at a 40% rate, 13% to 29% at 20%, 5% to 17% at 10%, wider than one simple
random sample's (35% to 46% at 40%) because each country's interval is taken at 97.5%. So a gap is called when LENIENT's
rate is above about 23%, and none is shown only when STRICT's is below about 8.5%.

A gap says that the map, read at a random sample of the region against an independent reference, is less accurate than
Ai2's validation figure by more than the margin. It does not say why: the year, the place, the legend and the
crosswalk, and Ai2's points sitting in the confident part of the map (exp98's P5) all move it. Part F compares the
2017 and 2023 maps at the same 47 plots and labels; that difference mixes the imagery's year, the satellites and
processing, and land change after 2017, so it does not isolate the year. exp98's Part I gives the 2023 map at the same
plots.

No correction for multiple predictions; each is graded on its own, as in exp98.

## Analysis steps (the order the script runs them)

1. Read the 2017 plots of Kenya and Tanzania, apply the area rule and refuse any count other than those above; with
   the boundaries, the region's area by country.
2. Open every from-olmoearth part directory as one map; map each plot to its pixel; count by country (the inventory
   stops here).
3. Refuse unless the page is frozen. Refuse probabilities that do not sum to 1.
4. The predictions and confidences; the errors under each rule; the design weights.
5. Part A, Part B, Part C; the transfer-gap call.
6. Part D (one pass over the windows), Part E (`plan`), Part F.
7. Grade P1 and P2 at the floors. Write `exp/out/exp99_summary.json` and `exp99_units.npz` (per plot: country, label,
   covered, class, confidence, condition code, 3 x 3 majority, weight, errors; no identifier and no position).

The cluster job is `exp/jobs/e99.sh`, chained on `exp/jobs/e99_read.sh`.

## What follows, whatever the outcome

- **P1 and P2 hold.** The map's confidence ranks its errors at an independent random sample of the region, in a year
  and places the model was not fine-tuned on: the package's review order applies beyond Ai2's own points.
- **P1 or P2 fails.** STRICT stays the graded rule, but a failure on it has two competing readings: transfer, or the
  crosswalk's grass/shrub boundary (250 of the 309 plots are the two grassland classes, and that boundary has no
  published AWF threshold, so STRICT's errors there partly measure the legend boundary, not the map's mistakes). The
  LENIENT values of P1 and P2 are set beside the verdict as the check. If both rules fail, transfer is the likelier
  reading: the ranking seen at Ai2's points (exp89, exp98) does not carry to a random sample under transfer, and the
  review order is not recommended on a transferred map without labels. If only STRICT fails, the cause is not
  determined.
- **P3.** The error rate, with its exact interval under each rule, is the record's first design-based error rate for an
  OlmoEarth deployment; the call says whether it departs from Ai2's figure by more than 5 points. With a gap, Part F
  describes the 2017 and 2023 maps side by side at the 47 plots; it does not say how much of the gap the year explains.
- **In every case,** Part E says how many labels a certified zone of the region's map would need.

## What would change the conclusion

- **The area rule's counts differ from those above** (another file or geometry): the run refuses.
- **Fewer than 250 plots with input:** no verdicts; the plots without input are reported by country and label. If the
  inventory shows that before freezing, the owner may change the floor, never after.
- **Windows off the CRS-origin lattice** (`e99_prepare_inventory.json`): the design argument's last step does not hold
  for those windows; the grades stand on the plots, with the count reported.
- **Plots outside their own country's Natural Earth polygon:** counted; the weights stand.

## What it cannot say

- **Anything about Ai2's 2023 map or exp98's re-run,** except at the 47 plots (Part F).
- **A certified zone** (Part E says why) or per-class user's or producer's accuracy (the classes are too few here).
- **Why a gap arises:** the year, the place, the legend and the crosswalk are not separated. Part F's comparison at
  47 plots mixes the year with satellites, processing and land change after 2017.
- **Whether the labels are right:** they are taken as right; a 30 m interpretation against a 10 m map adds disagreement
  that is not the model's.

## Deviations

None yet. Any change after freezing is listed here with its date, its reason, and whether any number had been read.

*Added at recording, 10 October 2026. The line above is the frozen text.*

- **10 October 2026, after the numbers were read: the plots without input are not missing at random (an assumption of
  this page; the grades are kept).** The limits above treat them as missing at random within their country, "which
  cannot be checked". The per-plot file partly checks it, and it fails: 20 of the 47 plots inside Ai2's geometry lack
  input, against 18 of the 262 outside it (Fisher's exact test p = 4 × 10⁻⁹), and Cropland (11 of 44) and Open
  Grassland (15 of 112) lack it more often than Wooded Grassland (9 of 138). The record states the graded set as the
  271 plots with 2017 input and says that the design-weighted rates and AUROC describe that set, not the 100 km region.
  It gives the envelope with the 38 filled as all correct or all wrong: STRICT 31.2% to 43.4% (interval envelope 22.6%
  to 53.1%), LENIENT 11.8% to 24.1% (6.5% to 33.2%). P1 to P3 are graded as frozen. P3's call would become "transfer
  gap" if most of the 38 were errors, and no filling makes it "no gap shown".
- **After freezing, before any class or probability was read: an incomplete area allowed.** 42 of the 313 windows
  received no 2017 Sentinel-2 item group at all (0 of 12, not fewer than 12; the records do not establish why). The
  rerun prepare's build refused with `complete: false`, and the prepare was resubmitted (job 1252876) with
  `E99_SKIP_BUILD=1 E99_ALLOW_INCOMPLETE=1`, a route `exp/jobs/E99_README.md` and `e99_prepare.sh` had provided since
  7b12dc4. This page names only the floor of 250 plots with input, and that floor decided (271).
- **Interpretation (says less than this page).** "If both rules fail, transfer is the likelier reading" is kept as the
  rule's label, not a finding: LENIENT's AUROC, 0.678, has a 90% interval of 0.599 to 0.756, which contains 0.70, and
  the run does not separate the year, the place, the legend boundary and the placement of the points (Part D's mean
  percentile 0.48 against exp98's P5, 0.647). "The record's first design-based error rate for an OlmoEarth deployment"
  holds for the plots with 2017 input only. "The region ... is a fixed subset" (the design, first bullet): D was set
  from the sample's own counts by a rule that reads no label and no map; the stopping rule's effect on the
  within-country argument is not corrected.
- **Part F.** It is reported on the 27 of the 47 inside plots that have 2017 input (the other 20 lack it), and its
  paired cells are crosschecked only at their margins: the 2023 classes at the plots are not saved.
- **Part D's sample.** It holds 1,725,553 pixels, not 2,000,000 in expectation: as in exp98, the sampling probability was
  set on the grids' full area rather than on the covered pixels counted once. Report-only.
- **The per-plot file.** Besides the fields step 7 lists it carries `inside_awf`, and its rows follow the public
  sample's sorted plot ids, so with the public file and the area rule a row can be tied to a public plot. It holds no
  position and no identifier.
- **The weights (no deviation).** Natural Earth's areas give 0.563 for Kenya and 0.437 for Tanzania (60,574 and
  47,105 km²), as the rule fixes; the 0.60 and 0.40 above were an estimate from the plots.
- **Before freezing (no deviation).** exp98's Part I had read the 2023 map at the 47 inside plots (c5092b4, 07:30 UTC
  on 9 October) before this page was frozen (52489b1, 13:46:59 UTC); the status line holds for the 2017 map. No
  threshold, floor, crosswalk or analysis rule changed after the draft at 7b12dc4, except the request geometry
  (7af2bd5, before freezing). The pilot (jobs 1247344 to 1247346) ran on the first request geometry, one feature per
  square (sha256 3cf53d16).
- **Run history (no deviation).** Prepare 1247349 (one feature per plot, 9,298 windows) was cancelled before
  freezing. After it: select at 13:44:53 UTC on 9 October (request sha256 a9dd3841), prepare 1252876, predict 1252877,
  collect 1252878 (7,330 files, 14.3 GB to the cluster's /home, manifest sha256 8f77213e). The read job 1252879 failed
  on the /home quota and produced no numbers; the collected run was moved to the cluster's scratch with a symlink, its
  manifest verified again on 10 October (7,330 files, 8f77213e), and read again (job 1267771). Inventory 1267772 and
  the graded run 1267773 followed on 10 October at 52489b1 with a clean tree. Scratch is purged after 30 days. The job
  ids from 1252879 on and the second verification come from the session's job log, not the committed records.
