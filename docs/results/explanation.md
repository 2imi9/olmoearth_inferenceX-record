# Why a window is suspect: cue enrichment on identical windows (exp37)

The explanation layer (`oe_inferencex.explain`) attaches to each window of
a review set the label-free cues that fire, each with its measured share
among error windows against correct windows and the experiment that
measured it. exp37 measured every cue the package derives on identical
windows, one error definition per testbed: Sen1Floods11 Bolivia hand labels
(81,984 valid 4-px windows in 440 tiles, 7,248 errors of the exp18 head,
tile-clustered bootstrap) and the 27 WorldCover rule scenes (27,648
windows, 1,844 errors, scene-clustered bootstrap). Descriptive, no
preregistered test. Source `exp/out/exp37_summary.json`; the per-window
tables `exp/out/exp37_patches_{bolivia,scenes}.npz` reproduce the shares
(`tests/test_recorded.py`).

![One flood tile with its review set and true errors, one flagged window traced through the four cue layers, and the measured enrichment per cue](../figures/explanation_layer.png)

*Sen1Floods11 Bolivia tile 218: orange squares are the 5% review set, red the hand-label errors; the marked window carries the boundary, low-confidence and tiling cues; the bars are the enrichments below.*

Cues, per window: **boundary** = indicator > 0; **low confidence** = the
20% least confident windows of the unit, ties included; **unstable** =
tile-phase in the unit's top 20%; **NDWI-ambiguous** = |patch-mean
NDWI| < 0.1; **flip/rotation disagreement** = exp36's std over the eight
dihedral transforms in the unit's top 20%.

| Cue | Bolivia hand labels: share among errors / correct, enrichment [95% CI] | Error rate among cue windows (base 0.088) | WorldCover scenes: errors / correct, enrichment [CI] |
|---|---|---|---|
| on a prediction boundary | 0.750 / 0.214, 3.5x [3.2, 3.8] | 0.25 | 0.854 / 0.201, 4.3x [3.5, 5.4] |
| among the least confident 20% | 0.589 / 0.163, 3.6x [3.4, 3.9] | 0.26 | 0.568 / 0.174, 3.3x [2.7, 4.0] |
| unstable under a sub-patch shift | 0.583 / 0.164, 3.6x [3.3, 3.8] | 0.26 | 0.744 / 0.161, 4.6x [3.9, 5.4] |
| spectrally ambiguous (\|NDWI\| < 0.1) | 0.483 / 0.067, 7.2x [6.2, 8.5] | 0.41 | 0.202 / 0.053, 3.8x [2.3, 6.9] |
| disagrees under flips and rotations | 0.579 / 0.164, 3.5x [3.3, 3.8] | 0.26 | 0.652 / 0.168, 3.9x [3.3, 4.6] |

On the scenes every cue is enriched on 8 of 8 rivers, NDWI ambiguity on 7.

**Coverage.** 95% of the Bolivia error windows carry at least one cue
(93.5% on the scenes) and 80% carry two or more; an error window carries
3.0 cues on average, a correct window 0.8, and 65% of the correct windows
carry none. The 5% of errors without any cue are the ones the explanation
cannot help with.

**Inside confidence's review set.** At the 5% budget on Bolivia (4,043
windows, error rate 0.379), 73% of the flagged windows sit on a boundary,
88% are unstable, 95% disagree under flips and rotations, and 27% are
NDWI-ambiguous. Two cues separate the set further: the error rate is 0.46
with the boundary cue against 0.16 without, and 0.51 with the NDWI cue
against 0.33 without. The tiling and dihedral cues do not (0.38 against
0.37, and 0.38 against 0.41), because almost every low-confidence window
carries them. At 20% the pattern holds (boundary 0.38 against 0.09, NDWI
0.47 against 0.20, unstable 0.29 against 0.19). The low-confidence cue is
the ranker itself and fires on every window of its own review set.

**Co-occurrence among errors (Bolivia).** Boundary with low confidence
50%, with instability 52%, with dihedral disagreement 49%; NDWI ambiguity
with each of the others 24 to 35%. Spectral ambiguity is the cue least tied
to the rest, which is why it adds precision inside the set.

**Reading.** Four of the five cues are one phenomenon seen four ways: a
low-margin window on a class boundary is also the window that flips under
a shift of the tiling or a rotation. The explanation of a typical review
window is therefore "on a boundary, among the least confident, unstable",
three facets of one fact. The fifth cue, spectral ambiguity, is the one
that tells a reviewer something the ranker did not: it covers a quarter of
the review set, carries the highest error rate of any cue, and is
task-specific (water against land), so it is the cue to derive from the
input for this task and to look for the analogue of in others. Putting it
ahead in the review order was then tested (exp38, [comparisons.md](comparisons.md#spectral-ambiguity-first-then-boundary-then-confidence-exp38)):
with a budget per tile it captures more errors than the boundary-first
order at every budget; with one budget over the whole area it does so only
at 20%, and on the WorldCover reference it loses. A cue that explains
errors is not automatically a better order to review them in.
