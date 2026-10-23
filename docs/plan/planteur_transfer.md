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
