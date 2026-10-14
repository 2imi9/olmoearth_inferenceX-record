# When a modality is missing: does the confidence fall with the accuracy? (exp88 preregistration)

**Status: frozen on 28 September 2026, before any result on real embeddings.** Drafted and confirmed by the owner
the same day (055f14b, b30b256); the readings below were fixed in e5bdc04. Nothing on this page changes after this
line.

- **Before freezing,** only two runs touched the real files, both allowed by the draft:
  - the smoke tests on synthetic embeddings;
  - the alignment check (job 1107181, commit dac3f08), which read identifiers and shapes and computed no probe,
    prediction or error.
- **The alignment check passed** for all three families on OlmoEarth Base and Large. The three inputs carry the same
  units, with labels equal in order, on the train and test splits. The files are the record's revision of Ai2's
  embeddings (6ea2c79). On PASTIS, 1,966 of the 1,984 test tiles have a label sequence no other tile shares, so a swap
  the check cannot see is confined to the other 18. Large runs as the replication.

## The question

A model trained on radar and optical input (Sentinel-1 and Sentinel-2) is sometimes run with one of them missing,
for example when clouds cover the optical image. Its errors then rise. The question is whether its confidence falls
with them:

- **If it does,** the review order still points to the windows to check. Only the error estimate needs a sample from
  the same conditions.
- **If it does not,** the extra errors are confident ones. The review order misses them, and only a random sample
  drawn under the same conditions finds them. An error rate measured on clear days says nothing about a cloudy map.

The package has not tested this. Every result in the record ranks errors on a model evaluated with the inputs it
was trained on. The nearest evidence points both ways:

- **exp84:** the margin beats the best no-model control in every sensor group (S1, S2, Landsat, S1+S2). But each probe
  there was trained on its own sensor.
- **Nandi Sentinel-1:** the margin loses where the input carries little information (probe accuracy 0.40 or less).
- **Lehmann et al. 2026** (arXiv 2608.16614): geospatial foundation models grow more overconfident under corruption
  and shift, and confidence-based abstention fails there.

## Design

- **Data.** Ai2's published embeddings (allenai/olmoearth-paper-embeddings, exp54's cache). Three task families carry
  the same samples under three inputs: S1 only, S2 only, and S1+S2.
  - **PASTIS**, 19 classes, 458,638 windows. This is the primary task.
  - **CropHarvest China 6**, 4,397 samples. This is the replication.
  - **CropHarvest Togo 12**, 306 samples. It is reported and not graded, being too small and badly fitted
    (exp84).

  A smoke job first checks that the three inputs of each family line up sample by sample. exp63 checked PASTIS's S2
  against S1+S2.
- **Encoder.** OlmoEarth Base, as in the record. OlmoEarth Large is a replication if the smoke job finds its
  embeddings cached for all three inputs.
- **The probe** is trained on the S1+S2 embeddings of the train split with exp54's recipe (fp32, the task's learning
  rate, 50 epochs, seed 0), and nowhere else. The seeds 1 to 4 replicate it.
- **Three conditions on the test split,** with the same probe and the same windows:
  - **full:** S1+S2;
  - **optical missing (cloud):** S1-only embeddings;
  - **radar missing:** S2-only embeddings.

  This is how the encoder meets a missing modality: it encodes what it is given.
- **A mixed map.** On PASTIS, half of the test tiles are drawn at random (seed 88) to take the optical-missing
  embeddings and the rest keep full input. This is the cloudy part of one map.

## Measures (per condition)

1. **Error rate.**
2. **The margin of the errors and of the correct windows:** medians, and the shift from full input.
3. **Confident errors:** the share of a condition's errors whose margin is at or above the median margin of the
   correct windows under full input. These errors sit in the half of the map a reviewer would trust.
4. **Ranking within the condition:** the margin's AUROC for errors, and its excess over the best no-model control
   (embedding distance, exp74's definition), as in the record.
5. **On the mixed map:**
   - the share of the cloudy part's errors that fall in a 5% review set ranked by margin, against the cloudy part's
     share of all errors;
   - a random sample of 300 windows, with the error rate estimated pooled and stratified by condition: the pooled
     estimate's error for the cloudy part, and each stratum's interval coverage over 2,000 draws.

## Predictions (one-sided; thresholds confirmed by the owner on 28 September 2026)

- **P1, errors rise without optical input.** On PASTIS, the error rate under optical-missing exceeds full input's by
  at least 5 points.
- **P2, the confidence does not fall with them (the question).** The share of confident errors (measure 3) is at
  least 5 points higher under optical-missing than under full input.
  - If P2 fails (the share rises by less than 5 points), the confidence tracks the lost information, and the review
    order still points where to look. That outcome is reported as the answer, not as a failure of the experiment.
- **P3, the margin still ranks errors inside the cloudy part.** Under optical-missing, the margin beats the best
  no-model control on PASTIS and on CropHarvest China 6.
- **P4, a pooled estimate misstates the cloudy part, a stratified one does not.** On the mixed map:
  - the pooled estimate, read as the cloudy part's error rate, is off by at least 5 points;
  - the condition-stratified intervals cover each part's true rate in at least 93% of 2,000 draws (the package's
    coverage bar, exp79).

## Readings fixed before freezing

The implementation (`exp/exp88_missing_modality.py`, 0a072fa) and its review raised points the predictions above
leave open. They are fixed here, before any result; no threshold changes.

- **Grading.** Every prediction is graded on OlmoEarth Base at probe seed 0. Seeds 1 to 4 and Large are reported.
- **P2's task** is PASTIS, as P1's. China 6's P1 and P2 are reported as replication.
- **P3's control** is the record's best no-model control: the better, by excess AURC, of the embedding distance and
  the class rarity, chosen per condition (exp70, exp74, exp84). The embedding distance is measured from the mean of
  the probe's own S1+S2 training embeddings. The lead over the embedding distance alone is reported, not graded.
- **P4's "off by at least 5 points"** is the pooled estimator's bias for the cloudy part over the 2,000 draws (its
  mean estimate minus the part's true rate), not the mean absolute error of single draws. Both are reported.
- **P4's stratified estimate** post-stratifies the one random sample of 300 by condition. Each part's interval is the
  package's interval for a random sample (`estimate_error_rate`, the exact hypergeometric interval), with the part's
  size as the population. A draw that labels no window of a part counts as not covering.
- **Consequences, stated so the result is not over-read:**
  - The exact interval covers at least 95% by construction, so P4's coverage half checks the implementation more
    than it tests a hypothesis. The finite-population Wilson interval that exp78 and exp79 graded is reported beside
    it.
  - On a map half cloudy, the pooled estimate misses the cloudy part's rate by half the gap between the parts. P4's
    bias half therefore needs a rise of about 10 points in P1's measure.
- **The alignment check** compares the three inputs' labels in order. A swap between two windows with identical
  labels cannot be seen, which weakens the check on China 6 and Togo, with few classes. The count of tiles with a
  unique label sequence is reported.

## What follows, whatever the outcome

- **If P2 holds,** the package says in its review output that the ranking is not valid across input conditions. It
  offers a design stratified by condition (cloud cover or modalities present) whenever the map records it. The
  certification is issued per condition.
- **If P2 fails,** the record says the margin tracks a missing modality, within the measured size. The per-condition
  estimate is still recommended (P4).

## Limits stated in advance

- **The embeddings of the missing-modality conditions are Ai2's single-sensor embeddings,** which are the encoder's
  reading of that input alone. Real clouds are partial and patchy. Whole-tile removal is the extreme case.
- **Three task families, crops only.** The question for water or land cover stays open.
- **Probes on frozen embeddings,** as everywhere in the record. A fine-tuned model may behave differently.

## Result (appended 28 September 2026, after the run; nothing above changed)

Job 1107807 at 3485aac. **P1, P2, P3 and P4 hold** on OlmoEarth Base at seed 0, at every other seed and on Large.
On PASTIS without the optical input the error rate rises from 19.5% to 73.6% and the share of errors that look
confident from 6.0% to 59.8%. P2 does not replicate on China 6, where the optical input matters little. The
numbers, the mechanism and the limits are in
[comparisons.md](../results/comparisons.md#when-a-modality-is-missing-does-the-confidence-fall-with-the-accuracy-exp88).
The follow-up this page fixed for P2 holding, a review output and a design stratified by input condition, is not
built yet.

Built (appended 29 September 2026; nothing above changed): the follow-up is in the package, not yet released (released in 1.4.0): `scope` in the outputs of `assess`, `estimate` and `certify`, and a condition layer (`--condition`) that ranks each condition on its own, splits the labels equally across the conditions, gives each its exact interval, and certifies each condition that holds enough labels with delta split over them.

## Addendum, 30 September 2026: a probe trained on the input it reads (nothing above changed)

Not preregistered. A red team found that the record already held, for PASTIS, a probe trained on the S1 embeddings
alone (exp70's, exported per unit by exp78) on the same 458,638 windows. Read beside this page's probe (OlmoEarth
Base, seed 0), it errs on 28.4% with a margin AUROC of 0.79, and 3.9% of its errors reach this page's threshold,
which is set by the S1+S2 probe. This page's probe, trained on S1+S2 and read on S1, gives 73.6%, 0.59 and 59.8%. So P1 and P2 hold for a probe run on an input
combination it was not trained on. They are not a property of the missing input as such. With matched probes on the
half-cloudy map, the parts still err on 28.3% and 19.7%, so the per-condition estimate this page recommends stays
justified. The ranking argument for it holds only where part of the map was read by a model on inputs it was not
trained on. On China 6, a probe trained on each input finds the optical input the more useful one (35.6% wrong on
radar alone, 26.6% on optical alone). The numbers, Large's and the limits are in
[comparisons.md](../results/comparisons.md#when-a-modality-is-missing-does-the-confidence-fall-with-the-accuracy-exp88)
and `exp/out/exp88_matched_head.json`.
