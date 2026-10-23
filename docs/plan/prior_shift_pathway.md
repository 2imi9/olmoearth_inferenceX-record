# Prior shift or feature shift? Splitting exp113's transfer failure with Bayes' rule (exp114 preregistration)

**Status: draft (23 October 2026).** Written before any statistic of this page was computed. exp113's own outputs
(window error rates, per-class IoU, predicted and true class shares) were read before writing it.

## The question

exp113 fitted a logistic regression on embeddings from some territories and applied it to a held-out one. In three
of the five held-out territories the map was close to all Background. Its confidence still ranked its errors, but
it missed most crop land (Guyane: Meadow is 14% of the labels, mapped at IoU 0.007).

Bayes' rule splits any classifier's posterior into two parts:

    p(y | x)  ∝  p(x | y) · π(y)

So a classifier can fail on new ground in two ways:
- **prior (label) shift:** the class shares π change, while each class looks the same, p(x | y);
- **feature shift:** p(x | y) changes, so a crop looks different in the new territory.

The two need different remedies. Under prior shift the map can be repaired without refitting. The correct target
posterior is

    p_t(y | x)  ∝  p_s(y | x) · π_t(y) / π_s(y)

and π_t can be estimated without labels. EM finds it (Saerens, Latinne and Decaestecker 2002, Neural Computation
14:21-41). Its log-likelihood is concave in the priors, so the maximum is unique. BBSE finds it by inverting the
source confusion matrix (Lipton, Wang and Smola 2018, ICML). Under feature shift no prior correction helps: the map
needs labels from the new ground.

There is one identity that tells the two apart. If the shift is prior-only and the source posteriors are calibrated,
the mean of the corrected posteriors over the target equals π_t. So once the true π_t is known, the gap between the
mean corrected posterior and π_t measures how much of the failure is not a prior shift.

exp114 asks:
1. which pathway produced exp113's held-out failures;
2. whether EM, without labels, estimates the target's class shares better than the map's own shares;
3. whether about 400 random labels, with exact intervals, can tell when the label-free estimate is wrong.

Point 3 is a check any user of any classifier map could run.

## Data and maps

- **The maps.** exp113's 18 maps (six configurations: five leave-one-island-out folds and the default split; three
  2019 embeddings: OlmoEarth v1.3, TESSERA v2, AlphaEarth). Each is refitted exactly as exp113 fitted it: the same
  100,000 fit pixels (seed 113, `exp/exp113_planteur_transfer.py`'s `fit_maps`), StandardScaler and
  LogisticRegression(max_iter=2000), float64, scikit-learn 1.9.1.
- **The reproduction check.** The run stops unless two things match exp113. The fit selection's sha256 must equal
  exp113's in every configuration. The refitted maps' window classes must equal exp113's stored window classes on at
  least 99.9% of windows for every map.
- **Units.** The classifier's own unit, the non-void pixel.
  - **Target:** every non-void pixel of the configuration's test patches.
  - **Source for BBSE:** every non-void pixel of the configuration's validation patches. For the five folds these are
    `splits/loio.json`'s `val`. For the default split they are `Fold` 4, the dataset's validation fold.
  - **π_s:** the class shares of the 100,000 fit pixels.
  - **π_t:** the class shares of the target's labels.
- **Main crop**, for the three territories exp113 found closest to all Background, fixed from exp113's recorded truth
  shares:
  - Guyane: Meadow (14.3% of the pixels);
  - Réunion: Sugarcane (16.0%);
  - Mayotte: the seven crop classes together (0.6%).

## Arms

1. **Oracle correction (the pathway split).** Each map's posteriors are corrected with the true π_t. Reported before
   and after, per map:
   - pixel accuracy, 8-class pixel mIoU, and the window error rate (exp113's window rules);
   - each class's predicted share and producer's accuracy;
   - the main crop's recovered share: its predicted (argmax) share after correction over its true share.

   The **pathway gap** is the total-variation distance ½ Σ_c |s_c − π_t(c)|, where s_c is the mean corrected
   posterior over the target. The same gap with no correction, s_c = the mean uncorrected posterior against π_t, is
   reported too.
2. **Label-free priors.**
   - **EM:** iterate π ← mean of p_s · π/π_s, normalised, from π_s until the largest change is below 1e-10, or 10,000
     iterations (the count is reported).
   - **BBSE:** solve C w = μ by least squares, then clip w at 0. Here C[i, j] is the share of source pixels predicted
     i whose label is j, and μ[i] is the share of target pixels predicted i. The estimate is π_t(j) ∝ w_j · π_v(j),
     with π_v the validation set's class shares.

   For each, report the total-variation distance to π_t, beside the map's own predicted (argmax) shares and π_s. The
   EM-corrected map's measures are reported as in arm 1.
3. **Random labels.** 1,000 simple random samples of n = 100 and 400 target pixels per map (draw r with seed 114 + r).
   - The exact 95% interval for each class's share is `estimate.hypergeom_interval` at level 1 − 0.05/8 (Bonferroni
     over the eight classes).
   - **The pathway test:** reject "the label-free priors are the target's" when EM's estimate of any class lies
     outside its interval. Report the rejection probability per map.
   - **The test's size:** the exact probability that the same test rejects the true π_t. By Bonferroni it is at most
     0.05 on every map; it is computed exactly from the class counts.
   - Reported only: the map corrected with the label-estimated priors, (k_c + 1)/(n + 8), mean over the draws.

## Predictions (written by me; see the decision at freezing)

**P1, the pathway (graded on OlmoEarth v1.3).** In at least 2 of the 3 territories closest to all Background
(Guyane, Réunion, Mayotte):
- the oracle pathway gap is at least 0.05;
- the gap is at least 3 times the default split's.

That is, those failures are mostly feature shift, not prior shift.

**P2, label-free priors.** EM's estimate is closer to π_t, in total variation, than the map's own predicted shares,
in at least 10 of the 15 held-out maps.

**P3, the test's power at 400 labels.** On every held-out v1.3 map where EM's estimate is at least 0.05 from π_t in
total variation, 400 random labels reject it with probability at least 0.8.

**P4, the test's size (a check).** The exact probability of rejecting the true π_t at n = 100 and 400 is at most 0.05
on all 18 maps. A failure is a defect: stop, find it, record the defect.

**Reported, not graded:**
- every quantity of every arm, for all 18 maps;
- BBSE beside EM, and the oracle- and EM-corrected maps against the uncorrected one;
- whether oracle correction raises pixel accuracy, which it should in expectation if the shift is prior-only and the
  posteriors are calibrated.

## What would change the conclusion

- **P1 holds.** exp113's collapses are mostly feature shift. Repairing the map needs labels from the new ground, and
  the per-class labelled routes are the check that matters.
- **P1 fails.** A prior correction explains much of them, so a map fitted elsewhere can be repaired by re-estimating
  the class shares. The tool could then offer that correction, with the source priors as an input.
- **P2 fails.** Label-free prior estimates are not to be trusted under this shift, whatever their uniqueness.
- **P3 holds.** A few hundred random labels catch a wrong label-free estimate, and the check is cheap enough to
  recommend.

## What it cannot show

- One benchmark, one year, crop types on tropical islands, one classifier and one fit size.
- Miscalibration and feature shift both open the pathway gap. The default split's gap is the in-distribution
  reference, but the two are not separated further.
- The main crop of each territory was chosen from exp113's outputs, which were read before this page.

## What would invalidate the run

- Any change, after the first statistic of this page is computed, to the maps, units, seeds, draws, bars or
  estimators.
- Grading before the page is frozen: the analysis refuses to run on a draft page.

## Independent check before anything is recorded

An independent second route recomputes the graded numbers. It is written without the run's script or outputs.

## Outputs

`exp/out/exp114_summary.json`: aggregates per map, with no pixel-level or patch-level list and no coordinate. The data
stay on the cluster's scratch.

## Cost

No new download: exp113's data (54 GB) and fit selections on the cluster's scratch. One CPU job, 16 CPUs, 32 GB, up to
3 hours.
