# Filter, then shake: a two-step error ranking fitted on exp98 and tested on exp99 (exp101 preregistration)

**Status: frozen on 10 October 2026, after the inventory and the GPU smoke and before any perturbed reading at a unit
of F or E outside the smoke's first 32 units of F.** Written 10 October 2026. The owner confirmed the thresholds, the
floors and the q grid below on 10 October 2026, as proposed (P1's bar of 5 points chosen over 3 points and over no
minimum), and asked for the views to be scaled up on GPUs. Before freezing: the inventory (job 1274633, at 2a07349)
read all 259 units of F and all 271 of E with twelve item groups, 18 and 22 of them clamped at a window edge; its count
of units that took a later window was wrong (an identity test on a fresh tuple counted every unclamped pick), fixed
before freezing, and the choice itself was right. The GPU smoke (job 1274634, NVIDIA B200) passed: at F's first 32
units, four view shapes gave the same class on the GPU and the CPU (128 of 128; logits within 2.1e-5), the script's
sample builder matched exp21's forward pass exactly, the replica's reference view agreed with the deployed map at 30 of
the 32, and the GPU ran 196 views per second (the fit about 7 minutes, the test about 14).

**Result, 11 October 2026 (added at recording; the frozen text is unchanged).** Fitted (job 1275097, an RTX PRO 6000)
and tested once (job 1275098, a B200) at a1b878e, recorded after an independent audit ([comparisons](../results/comparisons.md#filter-then-shake-a-two-step-error-ranking-under-transfer-exp101)).
q = 0.6 was chosen on F and reproduced by the test. P1 does not hold: R(0.6)'s 10% review holds 13.5% of E's STRICT
errors against confidence alone's 17.9% (difference -4.4 points, interval -8.1 to +4.3). P2 does not hold: AUROC 0.541
against 0.528 among the 112 plots not set aside (+0.013, interval -0.090 to +0.131). The floors are met. The replica's
reference view gives the deployed map's class at 91.5% of F (237 of 259; exp98's label-window replica, 238) and 88.9%
of E. On F, the fit set, the review holds 51.2% against 42.4%, an optimistic figure since q was chosen there.

- **Allowed before freezing:** the smoke on synthetic inputs (`python exp/exp101_two_step.py --smoke`,
  `tests/test_exp101.py`); an inventory that counts, at exp98's validation points and exp99's plots, the units whose
  crop and twelve item groups can be read (no class, probability or perturbed reading); everything on the fit set F.
- **Not allowed before freezing:** any perturbed reading at exp99's plots, and anything computed from one. The script
  refuses the test until this page says frozen.

## The question

The owner's idea (10 October 2026): as a strainer lifts the soup off a hotpot, first set aside the part of the map the
model is sure of; what is left is mostly boundaries and mixed ground (63% of exp98's errors sit on a predicted-class
boundary, exp21). Then shake what is left, looking at it from several perspectives, and see what holds. Under transfer
the model's confidence ranks its errors less well (exp99: AUROC 0.638 under STRICT, against 0.849 at Ai2's points in
exp98), and its least confident tenth closes 0.434 of the gap from random to the ceiling. Does a second step, ranking
the less confident part by how unstable the model's class is when its input is perturbed, catch more of the errors in a
10% review than confidence alone?

## What is known before the run

- Perturbation and disagreement signals used alone have behaved as a smoothed confidence in this record: flip-and-rotate
  consistency (exp36) has a Spearman correlation of 0.67 to 0.94 with confidence and lost to it on Sen1Floods11's hand
  labels; tiling instability tied confidence on the FT-AWF replica's 344 points (AURC 0.0235 against 0.0262, exp21) and
  correlates with it at 0.95; a shift changes the replica's class at 5% of errors and 1% of correct points (exp21).
- No perturbation signal has been tested conditionally, inside the less confident part, on FT-AWF in the record. On 10
  October, while this page was drafted, the two-step rule was tried on exp21's table (the replica's 344 points, of which
  exp98's 259 are a subset, so the fit set): with the most confident 50% set aside, tiling instability ranks the rest at
  AUROC 0.775 against confidence's 0.739; at 70%, 0.718 against 0.635; at 80%, 0.585 against 0.613; with 26 to 39 errors
  left, noise-level. Crop-size disagreement and class flips ranked near chance there.
- exp99's labels and the deployed map's confidence at its plots are recorded; the perturbed readings at the plots are
  not. Under STRICT the design-weighted capture of the least confident 10% is 0.179, random 0.100 and the ceiling 0.281,
  so the most a ranking can add at that budget is about 10 points.

## Units and data

- **F, the fit set:** exp98's 259 graded validation points on the 2023 re-run map (33 errors), Ai2's AWF labels.
- **E, the test set:** exp99's 271 plots with 2017 input (91 errors under STRICT, the primary rule; 37 under LENIENT,
  reported), design-weighted with Kenya and Tanzania as strata as in exp99.
- **Errors and confidence:** the deployed map's, exactly as exp98 and exp99 graded them (the class over all ten
  channels; the top-1 probability over the trained channels).
- **Inputs for the views:** a crop around each unit from the deployed run's own rslearn dataset on scratch (all twelve
  monthly item groups, the twelve bands at 10 m): exp98's 2023 root for F and exp99's 2017 root for E. If the 2023 root
  has been purged, F reads exp89's label windows instead (as exp89 did), and that is recorded as a deviation.

## The views and the instability

The record's replica of FT-AWF (exp89 arm A: Ai2's checkpoint, the encoder's tokens averaged over months and band
sets, the 1 × 1 head, bilinear upsampling by 4) is run on crops around each unit. The owner asked (10 October 2026)
for the shaking to be scaled up on GPUs, so a unit is looked at from every combination of four kinds of perturbation,
batched as one tensor per unit:

- **Orientation (8):** the eight elements of the dihedral group (identity, three rotations, four reflections) applied
  to the crop; the output is mapped back exactly before reading the unit's pixel (on crops whose sides are multiples of
  4 px, the patch grid maps onto itself).
- **Time (5):** all twelve monthly item groups, or one quarter of the year removed (January to March, April to June,
  July to September, October to December), the nine remaining groups passed with their own month indices (the model
  ignores masks in its fast pass, so removing the months is the only way to hide them).
- **Context (2):** a 16-px or a 32-px crop.
- **Phase (4):** the unit's position within its 4-px patch moved by 0, 1, 2 or 3 px along both axes, by moving the
  crop's origin (exp21's tiling shift).

That is 8 × 5 × 2 × 4 = 320 views. **V0, the reference,** is the identity, all twelve months, the 16-px crop, phase 0,
with the unit at (8, 8). **Instability** s of a unit: the share of the other 319 views whose class at the unit (argmax
over the ten channels) differs from V0's; ties are broken by the mean total-variation distance between each view's
probabilities and V0's.

**Compute.** One GPU on the aicr cluster (`b200-devel` or `rtx-devel`), after a GPU smoke of at most 15 minutes. exp89
ran this replica on the CPU because the encoder's band-index tensors stayed on the CPU when the model moved to the GPU
(job 1151050); the script moves them, and the GPU smoke checks that V0 on the GPU matches V0 on the CPU at the fit
set's units to within float tolerance before anything else runs. If the GPU path cannot be made to work, the same views
run on the CPU partition, recorded as a deviation.

## The two-step rule

R(q): set aside the most confident share q of the units (by the deployed map's top-1 probability); rank the others by s,
most unstable first, ties by confidence, least confident first; the units set aside follow, least confident first. The
10% review takes the top of R(q). q is chosen on F from {0.5, 0.6, 0.7, 0.8}: the one whose 10% review holds the most
errors, ties to the larger q (closer to confidence alone); F's four values are reported. Nothing else is fitted.

## Choices this page fixes (written with the script, before freezing)

- **a. The share set aside** is a share of the design weight on E, as the 10% review is (on F every weight is 1); units
  with tied confidence are set aside together or not at all.
- **b. A crop that would leave its window** is clamped at the window's edge by exp21's rule (exp89's `crop_at`), as
  exp21 and exp89 cropped; such a unit keeps all 320 views, some of them identical, and is counted.
- **c. A unit in two windows** takes the first (roots, then groups, then names, sorted) whose twelve item groups are
  complete and where no crop is clamped, else the first whose twelve groups are complete; windows are never mosaicked.
  Clamped units are counted among errors and correct units separately.
- **d. Item group k is month index k**, as exp21's replica reads the label windows; the season views keep those indices;
  timestamps are exp21's (day 15, the month index, 2023; the year is not read in the fast pass).
- **e. The total-variation distance** is half the L1 distance between the ten-channel softmaxes.
- **f. The bootstraps** resample units within each country (one stratum on F), use the same resamples for every
  statistic, and apply the rule anew in each resample at the fixed q; a resample without an error or without a correct
  unit is skipped and counted.
- **g. E's weights** are recomputed over the plots with all 320 views (each country's share of the region over its
  plots with views), and confidence alone's capture is recomputed on the same plots; with all 271 it is exp99's 0.179.
- **h. The dose** draws one subset of views per draw, the same for every unit.
- **i. The test** recomputes F's views and q, and refuses when q differs from the fit's record.
- **j. "Holds the most errors"** is the record's tie-aware capture times the errors (a tied group on the cut counted in
  proportion); two q tie when they hold the same within 1e-9. P1's and P2's intervals must lie strictly above 0.

**Order of the jobs.** The inventory (CPU; it also builds the environment), then the GPU smoke, both allowed before
freezing; the page is frozen after them, once the inventory's clamped counts are known; then the fit and the test.

## Predictions (on E, STRICT, design-weighted; confirmed by the owner)

- **P1 (the review):** the 10% review by R(q) holds a share of STRICT errors at least 5 points above confidence alone's
  (0.179), and the 90% within-country bootstrap interval of the paired difference lies above 0.
- **P2 (the shaking itself):** among the units R(q) does not set aside, s ranks the STRICT errors better than confidence
  does: design-weighted AUROC of s above that of confidence on the same units, with the 90% bootstrap interval of the
  difference above 0.
- **Reported, not graded:** LENIENT's values; F's values for each q; each kind of perturbation alone (its views with
  the other three kinds at V0's setting); the dose of shaking, P1's and P2's statistics with s computed from k views
  drawn at random from the 319 (k = 4, 8, 16, 32, 64, 128 and 319; 20 draws each, fixed seed), to see whether more
  views rank better; the grass/shrub plots, where STRICT and LENIENT part; the 5% and 20% budgets.

**Floors:** at least 250 plots of E with all 320 views read, and at least 20 STRICT errors and 20 correct plots among
them; otherwise the predictions are reported without verdicts.

## What would change the conclusion

- **P1 and P2 hold:** shaking the less confident part finds errors confidence misses, under transfer; the rule becomes a
  candidate for the package, which needs the model's own inference (the package now reads scores only).
- **P2 holds, P1 does not:** the shaking separates errors, but too few to change a 10% review at this sample size.
- **Neither holds:** under transfer, as at Ai2's points and on the earlier testbeds, the views add nothing confidence
  does not carry.
- **The replica's V0 class differs from the deployed map's at many units** (exp98 found the same class at 91.9% of F):
  the views describe the replica, not the deployed map; reported with the verdicts.

## What it cannot show

- Whether the gain holds for a model or legend other than FT-AWF's, or for a whole map: the units are points.
- Why a view changes the class: a season removed may hide the season that separates grass from shrub, or show the
  model a series shorter than it was trained on.
- With 271 plots and a 10% review of about 27, a difference of a few errors is within noise; P1's bar is set so that
  only a large gain passes.

## Outputs

`exp/out/exp101_summary.json` and `exp/out/exp101_units.npz` (per unit: set, label, errors, confidence, the 320 views'
classes and their distances to V0, the instability; no position, no identifier). Job `exp/jobs/e101.sh` on the aicr
cluster, one GPU, pinned to a full commit.

## Deviations

- **The GPU fix.** The page says the script moves the encoder's band-index tensors to the GPU; in the installed
  olmoearth_pretrain 0.1.0 they are registered buffers, which move with the model, so the mover moved nothing
  (`moved_tensor_attributes` is empty). The cause of exp89's job 1151050 is not known; the GPU smoke's equality check
  is what showed the GPU path right.
- **The inventory's count of windows changed to avoid clamping** was wrong in job 1274633 (an identity test on a fresh
  tuple counted every unclamped pick); fixed before freezing, as the status line says. The fit and the test count
  correctly.
- **Two GPUs.** The page names one GPU; the fit ran on an NVIDIA RTX PRO 6000 and the test on a B200, which the GPU
  smoke checked. The test recomputed F's views on the B200: the classes are identical at all 259 x 320 views and the
  distances to V0 within 1e-5, and q is the same.
- **The summary's `reading` field** uses the page's sentence for neither prediction holding ("as at Ai2's points and
  on the earlier testbeds, the views add nothing confidence does not carry"); at Ai2's points (F) the views did add,
  so the record reads the result as the comparisons page states it, not as that sentence.

