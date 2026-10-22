# A map made from OlmoEarth v1.3 embeddings: does the tool rank, estimate, certify and compare it at a probability sample? (exp112 preregistration)

**Status: frozen (22 October 2026), the owner having confirmed the predictions as proposed.** Written and frozen
before any embedding at a plot of S was read (the draft is 2ab2759) and before any statistic of the map was computed.

## The question

Ai2's OlmoEarth v1.3 (22 October 2026) makes precomputed embeddings the main way to use OlmoEarth: a user fits a small
classifier on labelled points and applies it to an area. Its report evaluates the embeddings by readout accuracy on
class-balanced draws and gives no confidence layer and no accuracy of a map made from them. The package's
`from-embeddings` command (unreleased, 7f910f2) writes such a map's class probabilities, and its Usage page says this
route is not graded. Here it is graded, at an independent probability sample: does the classifier's confidence rank the
map's errors, and do the labelled routes keep their guarantees?

## Data

- **Reference.** LCMAP Reference Data Product v1.2 (exp92's file, sha256 eb935d96...b4c, public domain): 25,000 plots, a
  simple random sample of the 30 m pixels of the conterminous United States, column `LCMAP` for 2018, LCMAP's eight
  primary classes (exp92's `CLASSES`). The file is local and was read for exp92 to exp94; no new reference data.
- **Embeddings.** OlmoEarth v1.3, year 2018, all 128 dimensions, from Ai2's store on Source Cooperative
  (`s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr`, anonymous). Dequantized by the store's own
  `geoemb:quantization` (signed power 2, scale 127.5, -128 no data), the formula of the dataset's README.
- **Coverage, checked before freezing** (22 October 2026, from `provenance/windows/2018.parquet`, read with the
  owner's agreement): all 25,000 plots lie in a 2018 window marked `written`.
- **Second maps for arm 4** (exp92's committed `exp92_plots.csv`, 2018): LCMAP CONUS Collection 1.3 (`lcmap`) and Esri
  10 m LULC crosswalked to LCMAP's classes (`esri`).

## Units

- **The plots drawn.** S: 12,000 of the plots with a 2018 class, drawn with `numpy.random.default_rng(112)` from the
  sorted plot ids, before any embedding is read. 12,000 and not 25,000 is the owner's choice of 22 October (option A:
  128 dimensions within about 70 GB of reads).
- **A plot's embedding.** The 10 m pixel of the 2018 array that holds the plot's centre (the plot is a 30 m pixel of
  LCMAP's Albers grid; its centre goes to the zone's UTM by pyproj). A plot whose pixel has no embedding is left out
  and counted.
- **The fit set.** A permutation of S by `default_rng(113)`; F is its first 1,000 plots that have an embedding. Report
  only: its first 300 and first 3,000 (nested in the same permutation).
- **The map.** `sklearn.linear_model.LogisticRegression(max_iter=2000)`, all else at the library's defaults, fitted on
  F's 128-dimensional vectors and LCMAP classes, predicts every other plot of S that has an embedding. These plots, T,
  are the population: the map's units, each with its true class. T is a simple random sample of CONUS's pixels as S
  is. Error: predicted class differs from the reference.
- **Confidence.** The top predicted probability; a plot's suspicion is minus it.

## Arms

1. **Ranking.** On T: AUROC of the suspicion for the errors (ties half), expected capture at 10% (the package's
   metrics), and two no-model controls: predicted-class rarity (exp110's `rarity`, minus the share of T in the plot's
   predicted class) and the label-free distance of the plot's vector from F's mean vector (exp70's control). The best
   control is the one with the higher AUROC.
2. **The interval.** On T: the exact coverage of the package's interval for a simple random sample at B = 100 and 300
   (`estimate.exact_coverage_srs`), and a Monte Carlo route, 2,000 draws per budget (seed 112) through
   `sample_for_estimation(design="random")` and `estimate_error_rate`.
3. **The certified zone.** On arm 2's B = 300 draws: certify at α = 0.05 and 0.10, δ = 0.10, the prefix rule
   (exp110's `certify_steps`, checked against `certify_zone` on the first 20 draws).
4. **Which map.** On the plots of T where the second map has a class: the v1.3 map (A) against LCMAP Collection 1.3
   (B), then against Esri (B'), with N, D (plots where they differ), K_A, K_B (where only one is right) and the
   tool's probabilities of naming each map at n = 100 and 300 labels drawn among the differing plots (exp110's
   `tool_probability`, `plan.which_map_probability`).

## Predictions (proposed; confirmed by the owner before freezing)

**P1, the ranking.** (a) AUROC ≥ 0.70 and capture at 10% ≥ 0.25 (2.5 times random); and (b) AUROC at least 0.05 above
the best control's. Floor: fewer than 200 errors in T leaves P1 not determined.

**P2, the interval (a check).** Exact coverage ≥ 0.95 at B = 100 and 300, and the Monte Carlo route within 3 of its
standard errors of it. A failure is a defect and stops the run before anything is recorded.

**P3, the certified zone.** (a), a check: at α = 0.05 the share of draws that certify a zone wrong more often than α
is at most 0.115 (δ plus about 2.2 Monte Carlo standard errors at 2,000 draws). (b): at α = 0.05 and 300 labels, the
median certified share is at least 0.25 of T.

**P4, which map (a check).** At n = 100 and 300 the tool names the map that agrees worse with the reference with
probability at most 0.025, against LCMAP Collection 1.3 and against Esri. Which map is better is not predicted.

The bars are exp110's, unchanged, so that the two products and this route are read on one scale.

**Reported, not graded:** the map's error rate on T at each fit size (300, 1,000, 3,000) and the ranking statistics at
300 and 3,000; the same at the first 64 dimensions (the nested prefix, from the same reads); a 90% bootstrap interval
for the AUROC and for the lead (2,000 resamples of T, seed 112); each class's user's and producer's accuracy on T; the
share of errors that are confident (confidence at or above the median of the correct plots, as exp106); the
probabilities that arm 4 names the better map; the distribution of the 2018 vectors' norms (one box read on 22 October
gave a median of 2.91 in 2018 and 3.34 in 2025, so a classifier fitted on one year may not carry to another); the share
of plots in 2018 windows computed without the 48-pixel margin (`context_pixels` 0).

## What would change the conclusion

- **P1 holds.** A map made the way v1.3 invites, a classifier on embeddings and labelled points, ranks its own errors
  at an independent probability sample, and the Usage page can say so for this case.
- **P1(a) fails.** The classifier's confidence does not find its errors here; Usage says so and the route stays
  ungraded beyond this case.
- **P1(b) fails, (a) holds.** A no-model control does as well; the confidence adds little over class rarity or distance.
- **P2, P3(a) or P4 fails.** A defect: stop, find it, record the defect, not the numbers.
- **P3(b) fails.** 300 random labels do not certify a quarter of this map at 5%; the map's confident part is not that
  clean, which is a property of the map, not a failure of the tool.

## What it cannot show

- The fit points are a random sample of CONUS. A user's few points usually sit in one area and the map covers that
  area, so this map is better spread than a user's; the regional case is not graded here.
- One year, one country, one classifier, LCMAP's legend. The reference labels are taken as right (Pengra et al. 2020
  measured 88% agreement between two analysts on a subsample).
- The plots stand in for the map: every rate is exact on T, not on CONUS's whole map.
- Whether any reference plot trained LCMAP Collection 1.3 is not checked (as exp92), so arm 4's comparison with it is
  of agreement with this reference, not of maps in general.

## What would invalidate the run

- Any change, after the first embedding is read, to S, the seeds, the fit sizes, the classifier's settings, the
  population rules or the bars.
- Reading embeddings at plots outside S.
- Grading before the page is frozen: the analysis refuses to run on a draft page.

## Independent check before anything is recorded

An independent audit recomputes P1 to P4 from the committed per-plot table with its own code (numpy and scipy only), as
exp110's did, before the result enters Summary or Findings.

## Outputs

- `exp/out/exp112_summary.json`: the verdicts, the arms' numbers, the reported quantities.
- `exp/out/exp112_plots.csv`: per plot of S, the plot id, whether it is in F (and at which fit size), the reference
  class, whether it has an embedding, the predicted class, the top probability. No coordinates and no embeddings.
- The embeddings stay off the repository (the OlmoEarth Artifact License; their size), on the cluster's scratch and
  the owner's machine.

## Cost

About 12,000 x 6 MB, roughly 70 GB of reads from Source Cooperative (two 64-band chunks of 256 x 256 pixels per plot;
the store moves no less). One CPU job, 16 CPUs, 32 GB, 16 concurrent reads, under 4 hours, resumable from a checkpoint
of finished plots; it stores 128 bytes per plot, about 1.5 MB. The analysis is a second short CPU job.

## Decisions taken at freezing

- 22 October 2026: the owner confirmed P1 to P4, the bars (exp110's), the sizes (12,000 plots, 1,000 to fit, 300 and 3,000 reported) and the reads (option A, 128 dimensions) as proposed. Before freezing only four plots of S were read, in a smoke run that printed their norms; nothing of the map was computed.

## Result (recorded 22 October 2026; the text above is the frozen page, unchanged)

Run once: extract job 1464443 (12,000 plots read in 775 s, none without an embedding), analysis job 1464959 at the
frozen f22f434 (olmoearth-inferencex 1.12.1 from PyPI, scikit-learn, zarr and obstore added by `uv run --with`).
Recorded after the independent audit below.

- **P1 holds.** On T (11,000 plots) the map errs on 0.1173 of them (1,290). Its top probability ranks the errors at
  AUROC 0.876 (90% bootstrap 0.869 to 0.884), and the least confident 10% hold 0.416 of them (ceiling 0.853). The lead
  over the better control is 0.366 (0.351 to 0.380); both controls are near chance (predicted-class rarity 0.488,
  distance from F's mean vector 0.511), so P1(b) is a weak bar here. 2.7% of the errors are as confident as the median
  correct plot.
- **P2 holds.** Exact coverage 0.960 at B = 100 and 0.955 at B = 300; the Monte Carlo route is within 0.27 standard
  errors of each.
- **P3(a) holds.** A certified zone wrong more often than α on 0.0075 of the draws at α = 0.05 (0.011 at 0.10).
- **P3(b) does not hold.** The median certified share at α = 0.05 is 0.075; see the note on its reading below.
- **P4 holds.** The tool names the map that agrees worse with the reference with probability below 1e-8 at both
  budgets against both second maps.
- **Reported.** Fit 300: θ 0.133, AUROC 0.858; fit 3,000: θ 0.107, AUROC 0.882; the first 64 dimensions at fit 1,000:
  θ 0.118, AUROC 0.877. Against LCMAP Collection 1.3 the map agrees with the reference on 0.883 of the plots against
  0.823 (N 10,989, D 1,949, K_A 1,189, K_B 536), against Esri 0.883 and 0.797 (N 10,998, D 2,027, K_A 1,401, K_B 457);
  100 labels name the v1.3 map with probability 0.948 and 0.999. Per class, among the predicted classes the producer's accuracy is lowest for
  Barren (0.442) and Wetland (0.594); Snow/Ice, absent from F, is never predicted (2 plots in T). The 2018 vectors'
  norms run 2.60 to 3.80 (1st to 99th percentile), median 3.08. All 12,000 plots lie in 2018 windows computed with the
  48-pixel margin (`context_pixels` 48 in `provenance/windows/2018.parquet`; the share without it is 0; checked by hand on
  22 October, not in the run's outputs).

## Notes at recording (22 October 2026, after the run; the frozen text is unchanged)

- **The reading of a P3(b) failure in "What would change the conclusion" is wrong for this run, and the verdict stands
  as written.** At 300 labels and α = 0.05 the standard level cut first tests the 15% zone, which expects 45 labels:
  exactly the 45 that certify at α = 0.05, δ = 0.10 with no error (`min_labels_to_certify`). So any map certifies a
  zone with probability at most 0.525 (the hypergeometric chance of 45 or more labels of 300 in 1,650 of 11,000
  plots), and a 2,000-draw median reaches 0.25 only when that probability is above one half: with no error in the
  15% zone it is 0.525, with one 0.510, with two 0.494. The bar is a knife-edge: it passes only a map whose most
  confident 15% holds at most one error (a perfect map passes with probability 0.99). This map's holds two (0.0012),
  so it certifies with exact probability 0.494 (guard aside), and 1,000 of the run's 2,000 draws certified nothing:
  its median, 0.075, is the midpoint between none and the first level (0.15). Beyond that edge the confident part is
  clean: the most confident 50% err at 0.0093, 70% at 0.028, 80% at 0.049; when a draw certifies, its median share is
  0.70, and at α = 0.10 the median is 0.85. Usage already says a level holds fewer than b_min labels about half the
  time at the budget where it enters (exp95); the bar, copied from exp110, was set at that budget. exp110's P3(b) sat
  on the same edge (Belgium's and Slovenia's probabilities, 0.525, are the limit; Croatia's 0.517 and Sweden's 0.484
  sit below it), so its "holds in 3 of 4" turned on whether each country's most confident 15% was almost free of
  errors; the note is posted there.
- **What was read before freezing.** S was drawn and four plots of S (all in T) were read at 16:18 in a smoke run that
  printed only their norms; the analysis code was run end to end on synthetic vectors (none equal to a real one) to
  test it. No statistic of the real map was computed before f22f434. The draft (2ab2759) and the freeze were committed
  eight seconds apart, after the owner's confirmation in conversation. So the Status line's "frozen before any
  embedding at a plot of S was read" does not hold: four plots of S were read 34 minutes before the draft and the
  freeze were committed, as the Decisions line says.
- **The audit.** The page said the audit would recompute P1 to P4 from the per-plot table with numpy and scipy only.
  The distance control needs the embeddings, which the table does not hold, so the second route
  (`exp/exp112_crosscheck.py`, written by an agent that saw neither the run's script nor its outputs) rebuilt S and the
  fit sets from the seeds, read the private vectors and refitted the map with scikit-learn, a check that also refits the map. It
  imports neither the run's script nor the package. Every deterministic number matches (P4's tail probabilities to within the package's 1e-12 truncation); its own Monte
  Carlo draws agree
  within their errors; P1 to P4 come out the same. A separate review read the run against this page.
- **Implementation details.** Arm 2's draw r uses seed 112 + r, the same at B = 100 and 300; the run's certify steps
  were checked against `certify_zone` on the first 20 draws at α = 0.05. The package's which-map probabilities drop
  up to 1e-12 of each tail, so their smallest values (the run's 0.0) are below 1e-11; the second route's exact values
  are 1.4e-9 at most.
- **Provenance.** The reference zip's sha256 is eb935d96...b4c (checked by the extract job), its CSV's 0959e322...;
  the vectors file's e74adb89... The per-plot table carries the OlmoEarth Artifact License's attribution and use
  restrictions (`exp/out/exp112_NOTICE.md`).
- **What it shows and what it does not.** A default logistic regression on 1,000 random plots' v1.3 vectors ranks its
  own errors well at an independent probability sample, across fit sizes and widths. The fit points are a random sample
  of the same population, the favourable case; a user's regional fit is not graded. The v1.3 map's better agreement
  than LCMAP Collection 1.3 and Esri is agreement with the reference that trained it (its legend and interpreters'
  conventions), not evidence that it is the more accurate map.
