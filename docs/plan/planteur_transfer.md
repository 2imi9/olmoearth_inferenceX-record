# Train on five territories, map the sixth: does the tool still find the errors, and which embedding is better there? (exp113 preregistration)

**Status: frozen (22 October 2026), the owner having confirmed the predictions as proposed.** Written (draft 513b94d)
and frozen before any PLANTEUR embedding or label was read and before any map was fitted.

## The question

exp112 graded a map made from OlmoEarth v1.3 embeddings in the favourable case: its fit points were a random sample of
the same population it was graded on. The common case is regional: a classifier fitted where labels exist and applied
somewhere else. Ai2's PLANTEUR benchmark (22 October 2026) measures that case for crop types in France's overseas
territories, and every model loses about half its accuracy on a territory it was not trained on. What PLANTEUR does not
say is whether, when accuracy falls that far, the map still knows where it is wrong. exp113 asks that, and whether
about 100 labels tell which of three embeddings gives the better map on the held-out territory.

## Data

- **PLANTEUR** (`allenai/planteur-benchmark` on Hugging Face, ODC-BY; code MIT): 2,750 patches of 128 x 128 pixels at
  10 m in Guadeloupe (735), Martinique (550), Guyane (506), Réunion (813), Mayotte (140) and Saint-Martin (6). Labels
  `ANNOTATIONS/TARGET_<id>.npy`, eight classes (0 Background, 1 Meadow, 2 Fruits, vegetables, flowers, 3 Orchard,
  4 Sugarcane, 5 Banana, 6 Pineapple, 7 Tropical tuber) and 255 void, from France's parcel registry.
- **Embeddings, 2019, on the label grid** (pixel [r, c] of an embedding is label pixel [r, c]):
  OlmoEarth v1.3 (`EMB_OLMOEARTH_V1_3`, 128 dimensions, int8 round-tripped as the paper's probes saw them; OlmoEarth
  Artifact License), TESSERA v2 (`EMB_TESSERA_V2`, 128; CC0) and AlphaEarth Foundations (`EMB_AEF`, 64; CC-BY 4.0).
  None of the three has missing patches in this release (`EMB_MISSING.json` lists TESSERA v1 only, which is not used).
- **Splits.** `splits/loio.json`: for each of the five territories, its patches are the test set and the others' train
  patches the fit set (Saint-Martin is never held out). The default spatial-block split (`metadata.geojson`, fold 1
  train, fold 5 test) is the in-distribution contrast.

## Units

- **The fit set of a configuration.** 100,000 pixels drawn uniformly at random, without replacement, from the non-void
  pixels of the configuration's train patches (`numpy.random.default_rng(113)` per configuration, the same pixels for
  the three embeddings). Validation patches are not used.
- **The map.** Per configuration and embedding, a scikit-learn pipeline of `StandardScaler()` and
  `LogisticRegression(max_iter=2000)`, all else at the library's defaults, fitted on the fit set's vectors and classes,
  predicts every pixel of the configuration's test patches: class probabilities, eight per pixel.
- **Windows.** 4 x 4 pixels (40 m), the package's default, inside each patch (32 x 32 windows per patch). A window's
  predicted class is the majority of its pixels' predicted classes, its confidence the mean of their top probability,
  as `assess` reads a probability map. Its true class is the majority of its non-void label pixels; a window more than
  half void, or whose label majority is tied, is left out and counted. Error: predicted window class differs from the
  true one.
- **The population** of a configuration and embedding is the windows of its test patches.

## Arms

1. **Ranking.** Per population: AUROC of minus the window confidence for the errors (ties half), expected capture at
   10%, and three no-model controls: predicted-class rarity (minus the population's share of the window's predicted
   class), the window's distance from the fit set's mean vector (its pixels' mean vector), and the map's boundary
   indicator (the share of a window's eight neighbours with another predicted class). The best control has the higher
   AUROC.
2. **The interval.** Exact coverage of the package's interval for a simple random sample of B = 100 and 400 windows
   (`estimate.exact_coverage_srs`).
3. **The certified zone.** 2,000 simple random samples of 400 windows per population (seed 113 + r), certify at
   α = 0.05 and 0.10, δ = 0.10, the prefix rule: the share of draws certifying a zone wrong more often than α, the share
   certifying any zone, and the certified share. 400 labels, not 300: at α = 0.05 a budget of 300 is where the first
   level enters and even a perfect map certifies only about half the time (exp112).
4. **Which embedding.** Per held-out territory, the three pairs of maps (v1.3 against TESSERA v2, v1.3 against
   AlphaEarth, TESSERA v2 against AlphaEarth) on the windows where both have a class: N, D, K_A, K_B, the two maps'
   agreement with the labels, and the tool's probability of naming each map at n = 100 and 400 labels drawn among the
   D differing windows (`plan.which_map_probability`).

## Predictions (proposed; confirmed by the owner before freezing)

**P1, the ranking under transfer (graded on OlmoEarth v1.3).** In at least 4 of the 5 held-out territories:
(a) AUROC ≥ 0.70 and capture at 10% ≥ 0.25; and (b) AUROC at least 0.05 above the best control's. Floor: a territory
with fewer than 200 window errors is not determined; if two or more are, P1 cannot hold.

**P2, transfer costs ranking.** For OlmoEarth v1.3, the mean AUROC over the five held-out territories is at least
0.05 below the AUROC on the default split's test patches. (exp99 and exp111 saw the ranking weaken when the place
changed; this tests that on a benchmark built for it.)

**P3, the interval (a check).** Exact coverage ≥ 0.95 at B = 100 and 400 for every population (18 maps on 6
configurations). A failure is a defect and stops the run before anything is recorded.

**P4, the certified zone (a check).** At α = 0.05 the share of draws that certify a zone wrong more often than α is at
most 0.115 for every population of the five held-out territories (δ plus about 2.2 Monte Carlo standard errors).

**P5, which embedding (a check).** At n = 100 and 400 the tool names the map that agrees worse with the labels with
probability at most 0.025, in every pair and territory where the two maps' agreement differs.

**Reported, not graded:** P1's statistics for TESSERA v2 and AlphaEarth, and for every map on the default split; the
window error rate of each map; each territory's 90% bootstrap interval (2,000 resamples of patches, seed 113) for v1.3's
AUROC and lead; the share of errors that are confident (as exp106); the certified share and the chance of certifying
any zone; arm 4's probabilities of naming the better map, and whether the better map on each territory is the one the
v1.3 report ranks first overall; the error rate inside and outside a one-pixel class-edge band (the package's
`boundary`, as condition strata), since crop parcels make edges common; the mean IoU of each map over the eight
classes, to compare with the v1.3 report's linear probes.

## What would change the conclusion

- **P1 holds.** A map fitted on other territories still ranks its own errors where it is applied, at least on this
  benchmark; Usage can say so for the regional case.
- **P1(a) fails.** Under transfer the confidence does not find the errors; a regional map needs random labels, and the
  ranking is not to be trusted there.
- **P2 holds.** Transfer costs ranking as well as accuracy, as exp99 suggested; **P2 fails.** The ranking survives
  transfer even when accuracy falls, which would be the more useful finding for users.
- **P3, P4 or P5 fails.** A defect: stop, find it, record the defect, not the numbers.

## What it cannot show

- One benchmark, one year (2019), crop types in tropical islands, one classifier and fit size. The fit pixels are a
  random sample of the train patches, not a user's few points.
- Background is land outside the parcel registry plus parcels with no source class: cropped land not in the registry
  is labelled Background, so some "errors" are the registry's gaps, not the map's.
- The windows of a test patch are not independent; the bootstrap resamples patches, but the arms' draws treat the
  windows as the population they are.
- The three embeddings were made by different teams with different inputs; arm 4 says which agrees better with these
  labels on this territory, not which embedding is better in general.

## What would invalidate the run

- Any change, after the first label or embedding is read, to the configurations, seeds, fit size, classifier, window
  rules, bars or the choice of v1.3 as the graded embedding.
- Grading before the page is frozen: the analysis refuses to run on a draft page.

## Independent check before anything is recorded

An independent second route, written without the run's script or outputs, recomputes the graded numbers from the
labels and embeddings, as exp110's and exp112's did.

## Outputs

- `exp/out/exp113_summary.json`: aggregates per configuration, embedding and territory; no patch-level or window-level
  list, no coordinate.
- The data and the per-window arrays stay on the cluster's scratch.

## Cost

Reads: the three embeddings (about 23 + 23 + 11.5 GB), the labels (45 MB), `metadata.geojson` (15 MB) and the split
files, to the cluster's scratch, about 58 GB; no imagery. One CPU job, 32 CPUs, 64 GB, a few hours at most: 18 fits of
100,000 pixels and predictions over up to 13 million pixels per configuration, streamed patch by patch.

## Decisions taken at freezing

- 22 October 2026: the owner confirmed P1 to P5 as proposed, including P2's direction (transfer costs ranking), read
  as a question whose failure would be the more useful finding; the three embeddings and the fit size of 100,000
  pixels were agreed earlier the same day. Only the dataset's README, `EMB_MISSING.json` and the two split files
  were read before freezing (formats and patch counts); no label, embedding or `metadata.geojson` was.

## Result (recorded 23 October 2026; the text above is the frozen page, unchanged)

Run once: fetch job 1467573 (54 GB), analysis job 1468432 at 3f63ffc (olmoearth-inferencex 1.12.1 from PyPI,
scikit-learn 1.9.1 added by `uv run --with`), 864 s on 16 CPUs; the run history is in the notes below. Each of the 18
fits drew exactly 100,000 pixels and converged (187 to 363 iterations). The truth rule left out 825 to 11,981 windows
per configuration. Recorded after the independent audit below.

| Held-out territory | Window error rate | v1.3 AUROC (90% bootstrap) | Capture at 10% | Best control (AUROC) | Lead (90% bootstrap) |
|---|---|---|---|---|---|
| Guadeloupe | 0.108 | 0.874 (0.870 to 0.879) | 0.420 | boundary (0.738) | 0.137 (0.132 to 0.142) |
| Martinique | 0.112 | 0.857 (0.848 to 0.866) | 0.403 | boundary (0.633) | 0.224 (0.215 to 0.233) |
| Guyane | 0.172 | 0.816 (0.795 to 0.837) | 0.323 | boundary (0.513) | 0.303 (0.282 to 0.323) |
| Réunion | 0.251 | 0.876 (0.868 to 0.884) | 0.320 | distance (0.787) | 0.089 (0.079 to 0.099) |
| Mayotte | 0.005 | 0.742 (0.710 to 0.773) | 0.286 | rarity (0.504) | 0.238 (0.206 to 0.270) |
| Default split (not in P1) | 0.088 | 0.880 | 0.471 | boundary (0.701) | 0.179 |

- **P1 holds, in 5 of 5.** Every territory has at least 748 window errors, so none falls under the floor.
- **P2 does not hold.** The mean held-out AUROC is 0.833 against 0.880 on the default split, a drop of 0.047 against
  the bar of 0.05. The drop is not uniform: 0.006 (Guadeloupe), 0.023 (Martinique), 0.064 (Guyane), 0.004 (Réunion) and
  0.138 (Mayotte). For TESSERA v2 the drop is 0.054 and for AlphaEarth 0.079 (reported).
- **P3 holds.** The lowest exact coverage of the 36 cells is 0.953 (Réunion, AlphaEarth, B = 100).
- **P4 holds.** The highest violation share at α = 0.05 among the 15 held-out populations is 0.0505 (Martinique,
  TESSERA v2), against the bar of 0.115.
- **P5 holds.** The highest probability of naming the map that agrees worse is 0.0108 (Guyane, v1.3 against
  AlphaEarth, n = 100).
- **Reported.** TESSERA v2 meets P1's two parts in 4 of 5 (not Mayotte: AUROC 0.696, capture 0.227). AlphaEarth also
  meets them in 4 of 5 (not Réunion: capture 0.196, lead 0.044). Window error rates, v1.3, TESSERA v2 and AlphaEarth:
  Guadeloupe 0.108, 0.117, 0.127; Martinique 0.112, 0.116, 0.124; Guyane 0.172, 0.171, 0.172; Réunion 0.251, 0.252,
  0.358; Mayotte 0.0052, 0.0055, 0.0058; default 0.088, 0.088, 0.099. Share of v1.3's errors as confident as the median
  correct window: 0.031, 0.052, 0.093, 0.057 and 0.148 in the five territories. The certified zone, 400 labels, v1.3,
  α = 0.05:
  - chance of certifying any zone: 0.85, 0.81, 0.26, 0.37 and 0.99;
  - median certified share: 0.70, 0.65, 0, 0 and 1.0;
  - at α = 0.10 the median certified share is 0.90, 0.90, 0.65, 0.60 and 1.0;
  - on the default split, 0.92 and 0.80 at α = 0.05.

  Which embedding: the map that agrees best with the labels is v1.3 in Guadeloupe, Martinique, Réunion and Mayotte,
  and TESSERA v2 in Guyane. Window error rate inside and outside the one-pixel class-edge band, v1.3:
  - Guadeloupe 0.434 and 0.074;
  - Martinique 0.437 and 0.092;
  - Guyane 0.543 and 0.170;
  - Réunion 0.781 and 0.218;
  - Mayotte 0.179 (28 band windows) and 0.005;
  - default 0.388 and 0.065.

  Pixel mIoU over the eight classes, v1.3: 0.308, 0.319, 0.120, 0.127 and 0.125 held out, 0.356 on the default split.
  Means over the five held-out territories: v1.3 0.200, TESSERA v2 0.185, AlphaEarth 0.168.

## Notes at recording (23 October 2026, after the run; the frozen text is unchanged)

- **The run history.** The run script was committed after freezing. It was smoke-tested on 10 patches (32963c3; code
  checks only) and changed after a pre-run review (9838301).
  - The first run, job 1467772 at 9838301, computed every map. It failed at the verdict step, on a which-map result
    stored among the pairs, a bug from the pre-run fixes. Its log had printed each map's statistics.
  - The fix, 3f63ffc, moved that result to its own key and nothing else. The smoke now runs the verdict step.
  - The rerun at 32 CPUs, job 1468220, was queued until 03:29, so it was cancelled. It was resubmitted as job 1468432
    with 16 CPUs and 32 GB, against the 32 CPUs and 64 GB in Cost. It ran the whole analysis again.

  None of these commits changed what "What would invalidate the run" lists. The data took 54 GB of scratch, not 58.
- **The interval P3 graded.** Arm 2 names "the package's interval for a simple random sample"
  (`estimate.exact_coverage_srs`). Both routes graded the exact hypergeometric interval, which `estimate` prints for a
  random design, and which covers at least 95% for every (N, K, B). That function's default argument is the
  finite-population Wilson interval. Under that reading P3 would fail: the second route puts its lowest coverage at
  0.895 (Mayotte, TESSERA v2, B = 100), where an error rate near 0.5% meets 100 labels. The verdict stands on the
  interval the package prints.
- **In three of five territories the map is close to an all-Background map.**
  - Guyane's registry is 14% Meadow, which v1.3 maps at IoU 0.007.
  - Réunion's is 16% Sugarcane, mapped at IoU 0.013.
  - Mayotte's labels are 99.4% Background.
  - A map that called every pixel Background would score mIoU 0.093 to 0.124. The three territories score 0.120 to
    0.127.

  There the AUROC mostly measures lower confidence on registry crop land the map calls Background. In Guadeloupe and
  Martinique the held-out map does classify crops: Sugarcane IoU 0.71 and 0.72, Banana 0.49 and 0.50.
  - Mayotte's error rate of 0.5% is its registry crop share, not the map's skill. A 10% review list there would hold
    about 1.5% errors.
  - Its which-map cells are a census or nearly so: the two maps differ on 40 to 120 windows.
- **What the leads measure.** P1(b) is informative only in three territories, where a control ranks errors on its own:
  - Guadeloupe and Martinique, the boundary indicator at 0.738 and 0.633;
  - Réunion, the distance from the fit mean at 0.787.

  In Guyane and Mayotte every control is between 0.38 and 0.51, so there the lead mostly measures the AUROC's distance
  from 0.5, as in exp112.
- **P2's failure is narrow, not a ranking unaffected by transfer.** The drop misses the bar by 0.003, and the page set
  no interval for it. It averages small drops (Guadeloupe, Réunion) with large ones (Guyane 0.064, Mayotte 0.138).
  The default split's test patches pool all territories, so its AUROC is not any one territory's in-distribution value.
  The other two embeddings lose more (0.054 and 0.079).
- **Which embedding.** Agreement with the labels differs by 0.12 points or less:
  - in Guyane and Mayotte;
  - in Réunion, between v1.3 and TESSERA v2.

  It differs by more in Guadeloupe and Martinique, where v1.3 leads by 0.4 to 1.9 points, and in Réunion, where
  AlphaEarth trails by 10.6 to 10.7 points. Where the gap is 0.4 points or more, 400 labels name the better map with
  probability 0.65 to 1.0. Where it is 0.12 or less, the probability is 0.04 to 0.16, except in Mayotte, whose cells
  are a census. The v1.3 report's PLANTEUR table ranks v1.3 first, then TESSERA v2, then AlphaEarth, on both its splits
  (Table 2: leave-one-island-out mIoU 24.9, 19.1 and 15.8). This run's mean held-out mIoU has the same order. That is
  consistent with the report, not a test of it. The report's numbers come from a linear probe with a tuned learning
  rate, on embeddings computed without the released ones' Searchlight inference. This run uses windows, a default
  logistic regression and 100,000 fit pixels. Its mIoU counts Background, which is 75% to 99% of each territory's
  labelled pixels. So the two sets of numbers are not comparable.
- **The labels.** Errors are counted against France's parcel registry. Cropped land outside it is Background, so part
  of what the map gets "wrong" may be the registry's gaps (as "What it cannot show" says).
- **The audit.** The second route (`exp/exp113_crosscheck.py` at 12b4a4a, job 1469340) was written by an agent that saw
  neither the run's script nor its outputs, and imports neither. It uses numpy, scipy and scikit-learn.
  - It draws the same fit pixels: the selection's sha256 matches in all six configurations.
  - It fits the stored float32 vectors where the run converts them to float64. That flips a few windows: error counts
    differ by up to 1%, the v1.3 AUROC by at most 0.003, and the other maps' by at most 0.005.
  - Its own Monte Carlo draws agree within their errors: 3 of 72 certify z-scores exceed 2 and none exceeds 3.
  - Every verdict is the same, and the best-agreeing map and best control are the same everywhere.
  - P3's minimum sits in a different cell on that route (0.956, Réunion, v1.3, B = 400). The coverage of a fixed B jumps
    with K, and the two routes' K for Réunion and AlphaEarth fall on either side of a jump.

  A separate review read the run against this page and found no blocking defect. The run's check of its certify steps
  against `certify_zone` covered α = 0.05 only.
- **What it shows and what it does not.** On this benchmark the ranking held in every held-out territory. In three of
  them the map transfers little beyond Background, so the confidence finds the coarse error, crop land called
  Background. Where the map does transfer, in Guadeloupe and Martinique, it ranks its errors at 0.874 and 0.857, above
  a boundary control that is informative there. Labelled routes kept their guarantees at all 18 maps. One benchmark,
  one year, crop types on tropical islands, one classifier and fit size.
