# Prior shift or feature shift? Splitting exp113's transfer failure with Bayes' rule (exp114 preregistration)

**Status: frozen (23 October 2026), by me, on the owner's overnight approval; the owner has not reviewed the
predictions.** Written and frozen before any statistic of this page was computed. exp113's own outputs (window error
rates, per-class IoU, predicted and true class shares) were read before writing it.

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
   - Reported only: the map corrected with the label-estimated priors, (k_c + 1)/(n + 8), mean over the first 100
     draws.

## Predictions (written by me; see the decision at freezing)

**P1, the pathway (graded on OlmoEarth v1.3).** In at least 2 of the 3 territories closest to all Background
(Guyane, Réunion, Mayotte):
- the oracle pathway gap is at least 0.05;
- the gap is at least 3 times the default split's.

That is, those failures are mostly feature shift, not prior shift.

**P2, label-free priors.** EM's estimate is closer to π_t, in total variation, than the map's own predicted shares,
in at least 10 of the 15 held-out maps.

**P3, the test's power at 400 labels.** On every held-out v1.3 map where EM's estimate is at least 0.05 from π_t in
total variation, 400 random labels reject it with probability at least 0.8. If no v1.3 map is that far off, P3 is
not determined.

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

## Decisions taken at freezing

- 23 October 2026, about 02:00. The owner asked for maths-trial experiments to run overnight, in rounds, with results
  by the morning ("I approve what you can do now"), and went to sleep. The arms, bars and predictions above are mine,
  frozen without the owner's review, and the owner reviews them in the morning.
- Read before freezing: exp113's recorded outputs, and the dataset README. The README notes a class-balanced label
  set for training (`ANNOTATIONS_BALANCED`), which exp113 did not use and which is not downloaded. Nothing of this
  page was computed.
- The first commit of this page (15a84ed) carried the draft status line while its message and the plan index said
  frozen; this commit makes the page say what was decided. No statistic was computed in between.

## Result (recorded 23 October 2026; the text above is the frozen page, unchanged)

Run once: job 1471685 at 9bf5f54, on 16 CPUs (olmoearth-inferencex 1.12.1, scikit-learn 1.9.1). It took 1,971 s by
the job log; the summary records no run time, and its per-map timings sum to 1,344 s.

Before any statistic of the page was computed, all 18 refitted maps matched exp113's stored windows exactly:
- class agreement 1.000000 and confidence difference 0 on every window;
- the patch, window and truth arrays identical;
- the target class counts, fit selections and split files equal.

After each configuration, the uncorrected mIoU also equalled exp113's. EM converged on every map, in at most 296
iterations. Recorded after the independent audit and a review below.

| v1.3 map | Pathway gap, uncorrected → oracle | Main crop: mapped area over true area, map → oracle | Main crop found (producer's accuracy), map → oracle | EM's distance to the true shares | 400 labels reject EM (draws; exact) |
|---|---|---|---|---|---|
| Guadeloupe | 0.041 → 0.016 | | | 0.112 | 1.000; 1.000 |
| Martinique | 0.058 → 0.053 | | | 0.125 | 1.000; 1.000 |
| Guyane | 0.130 → **0.089** | Meadow 0.008 → 0.093 | 0.007 → 0.071 | 0.486 | 1.000; 1.000 |
| Réunion | 0.213 → **0.198** | Sugarcane 0.014 → 0.102 | 0.013 → 0.097 | 0.354 | 1.000; 1.000 |
| Mayotte | 0.040 → 0.005 | crops 0.014 → 0.000 | 0.001 → 0.000 | 0.498 | 1.000; 1.000 |
| default split | 0.007 → 0.009 | | | 0.021 | 0.896; 0.875 |

- **P1 holds** (2 of 3). The oracle gap is 0.089 in Guyane and 0.198 in Réunion. Both are above 0.05 and more than 3
  times the default split's 0.009.
  - Mayotte's 0.005 does not meet the bar, but the gap could not have met it there. Its crops are 0.61% of the pixels,
    so even a map that found none of them would open a gap of at most about 0.006.
  - After the correction, Mayotte's mean posterior holds 0.28% crops against the true 0.61%, about the shortfall seen
    in Guyane's Meadow. The corrected map finds no crop.
  - Martinique, not among the three, also meets the criterion (0.053).
- **P2 does not hold.** EM's estimate is closer to the true shares than the map's own predicted shares in 3 of the 15
  held-out maps (Martinique TESSERA v2 and AlphaEarth, Guyane TESSERA v2), against a bar of 10. Its distance to the
  truth runs from 0.006 to 0.681.
- **P3 holds.** On all five held-out v1.3 maps EM is at least 0.05 from the truth. 400 random pixel labels reject it
  with probability 1.000 in the draws and 1.000 exactly.
- **P4 holds.** The exact probability of rejecting the true shares is at most 0.0232 over the 36 cells. These hold 12
  distinct values, since the size depends only on each configuration's class counts. The union bound is at most
  0.0246.
- **Reported.**
  - Oracle correction raises pixel accuracy in 13 of the 15 held-out maps. The exceptions are AlphaEarth in Guyane
    (0.826 → 0.824) and in Réunion (0.638 → 0.612).
  - BBSE is closer than the map's own shares in 8 of 15. It is far off on all three default-split maps, where there is
    no shift: 0.300, 0.191 and 0.905, against the maps' 0.044, 0.057 and 0.054. Its error does not follow the
    condition number of its validation confusion matrix.
  - Across all 15 held-out maps, 400 labels reject EM with probability 0.910 to 1.000 in the draws (0.914 to 1.000
    exactly). On the default split they reject it with exact probability 0.875 (v1.3), 0.880 (TESSERA v2) and 0.9998
    (AlphaEarth), although EM is only 0.014 to 0.041 off there. With 100 labels the draws give 0.988 to 1.000 on the
    held-out v1.3 maps and 0.419 on the default split (exact 0.408).
  - Correcting with the priors estimated from 400 random labels, averaged over the first 100 draws (no second route
    computed this):
    - Guyane's Meadow is mapped on 0.095 of its true area and found at 0.072;
    - Réunion's Sugarcane on 0.094, and found at 0.089.

## Notes at recording (23 October 2026, after the run; the frozen text is unchanged)

- **The run history.** The script was written after freezing (afb8005) and exercised on synthetic PLANTEUR-shaped data.
  - The smoke job (1471535) ran on 6 test patches per configuration with 5,000-pixel fits, which are not the page's
    maps. Its log printed per-map numbers for those tiny fits. The smoke now prints only sizes and timings.
  - The pre-run review found one blocking defect: P4 was graded on the union bound, not the exact joint probability
    the page names. It also found that the reproduction check stopped only after statistics had been computed.
  - Both were fixed in 9bf5f54 before the run, together with faster EM and report draws (same estimators), partial
    output and a 12-hour limit. The page's Cost said up to 3 hours.
- **What the gap shows, and what it does not.** Under prior shift with calibrated posteriors, correcting with the true
  shares makes the mean posterior equal those shares. In Guyane and Réunion it does not.
  - The oracle cuts Guyane's gap by a third (0.130 → 0.089) and Réunion's by little (0.213 → 0.198). Part of
    Guyane's failure is prior shift; most of both is not.
  - After the correction the mean posterior puts 6.3% on Guyane's Meadow, against 14.3% in the labels. The corrected
    map draws 1.3% Meadow.
  - The page's own caveat stands: out of distribution, feature shift and miscalibration both open the gap, and this
    run does not separate them. "Mostly feature shift" means the remaining gap is not explained by the class shares.
    It does not show that only labels from the new ground could repair the map; that was not tested.
- **Why the label-free estimate fails.** EM maximises a likelihood that is concave in the priors. Its estimate is right
  only if the shift is a prior shift. Under feature shift it amplifies the map's biases:
  - Guyane, v1.3: 31.4% Pineapple and 17.5% Tropical tuber, where the truth is 0.1% and 0.2%;
  - Réunion, v1.3: 35.7% Pineapple;
  - Mayotte, v1.3: 49.9% Orchard.

  It also drives some classes to zero: both routes' estimates hold exact zeros and values near 1e-308. Concavity alone
  does not make the maximum unique, and this run does not show it is.
- **What the test says, and what it does not.** 400 random labels test a label-free estimate of the class shares with
  an exact size. The run drew pixels; whether the window labels a user draws for the error rate or the certified
  zone serve as well is not tested.
  - They rejected EM with probability at least 0.91 on every held-out map. They also rejected it on the default split (0.875 exactly for v1.3), where
    EM was close and its correction raised accuracy slightly (0.9053 → 0.9064). There EM puts about zero on classes
    such as Orchard and Tropical tuber, and any sample that contains them rejects it.
  - So a rejection says the label-free shares are not to be trusted. It does not say which pathway caused the failure.
  - The pathway needs the shares themselves, which the same labels estimate with exact intervals. A gap with its
    uncertainty, from labels alone, is not graded here.
- **The audit.** The second route (`exp/exp114_crosscheck.py` at 0b8cb00, job 1471687, 1,714 s) was written by an agent
  that opened neither the run's script nor its outputs.
  - **What it saw.** One commit message, afb8005's, in `git log`. That message described P4 as a union bound, and the
    second route graded P4 on the union bound; its exact joint value, also reported, holds too.
  - **What it recomputed.** It refits the maps from exp113's saved fit pixels and recomputes with its own code. Every
    number it computed matches within 1.1e-9: class shares, accuracies, window error rates, gaps, EM's estimates and
    iteration counts, BBSE, and the exact sizes and powers.
  - **What it did not compute.** The label-corrected maps, and the check against exp113's stored windows. It compares
    with exp113's recorded numbers instead.
  - **The draws.** The run's draws agree with the exact rejection probabilities: 2 of 36 z-scores exceed 2, the largest
    2.34. In the second route's own draws, one cell (Guadeloupe, TESSERA v2, n = 400) had one non-rejection in 1,000,
    where the exact chance of any is 0.011. Elsewhere |z| ≤ 1.83.
  - **The verdicts.** All four are the same.
- **The review of this write-up.** A review before it was committed corrected several readings in the first draft:
  - Mayotte, first called prior shift;
  - BBSE's errors, first blamed on ill-conditioning;
  - which route's draws were compared;
  - a "practical rule" that the default split contradicts;
  - mapped area quoted as if it were the share found.
- **What it cannot show.** One benchmark and one classifier, and the gap does not separate miscalibration from feature
  shift out of distribution. The default split's 0.009 bounds miscalibration only in-distribution.
