# Recipe: auditing a prediction map without labels

What the first twenty-five experiments support doing, and not doing, when judging where
a prediction map is wrong in a region with no labels. Each item cites the
experiment behind it; the evidence is in
[../results/comparisons.md](../results/comparisons.md), and for the
experiments archived on 10 October 2026 in its
[v1.9.0 copy](https://github.com/2imi9/olmoearth_inferenceX/blob/v1.9.0/docs/results/comparisons.md). Index at
[../TECHNIQUES.md](../TECHNIQUES.md).

**What this page does not yet carry.** Forty-odd experiments after exp25 have
produced conclusions a practitioner would want and this page has not been
rewritten around them. The largest omissions: the comparison half of the
package, which measures where two inferences of one place differ and, with
labels, which side to believe (exp57 to exp63, exp65); the progression across
four kinds of reference, photointerpretation, another model's output, ground
observation and farmers' declarations, and what each one does and does not
license (exp66 to exp69); the warning that a small readout which has memorised
its practice data reverses which confidence signal ranks best, so a ranking
comparison is unreadable without its readout's generalisation gap beside it
(exp68); and the floor a two-date difference must be read against, which is how
often the map moves where the ground did not, between 6% and 37% on crops
(exp69), not the rate at which reseeding the head moves it. Until this page is
rewritten, [../TECHNIQUES.md](../TECHNIQUES.md) is the complete index and
[../results/comparisons.md](../results/comparisons.md) the evidence.

## Do

**1. Rank errors by the model's own confidence.** On every expert-labelled
testbed — AWF point labels (exp04, exp16), Sen1Floods11 dense masks with a
geographic hold-out (exp18), the fine-tuned AWF model itself (exp21) — the
logit margin ranked errors at least as well as every constructed signal.
For a two-class map, score the absolute logit, so saturated probabilities do
not tie. For a map of more than two classes, one minus the top probability
ranked errors slightly better than the probability margin on 14 of 16
multi-class tasks, and the logit margin worse than the probability margin on
all 16 (exp76).
The package's default reads the top probability from class probabilities and
the logit margin from logits (`--logits`); in Python, `form="top1"` reads the
top probability from logits.

Do not compare confidence across inputs a model was not trained on. Where part
of the map was read from such an input combination, such as radar alone under
cloud for a model trained on radar plus optical, the model can be confidently
wrong there. On PASTIS such a probe of OlmoEarth Base was 73.6% wrong, and
59.8% of its errors looked confident against 6.0% with both inputs; a probe
trained on radar alone was 28.4% wrong and ranked its errors with an AUROC of
0.79, against 0.83 with both inputs, and 3.9% of its errors reached the same
threshold, which is set by the probe trained on both. OlmoEarth Large's share rose only from 5.7% to 12.8-13.8% (exp88).
Even with a model trained on each input, the error rate can differ between
input conditions. If the map records each pixel's input condition, pass it as
`--condition`, and each condition is ranked, sampled and certified on its own.

**2. Use prediction-boundary proximity as a triage cue, not a ranker.**
About 75% of error patches lie on boundaries of the model's own prediction
map against about 20% of correct patches, on both references (exp14, exp16,
exp18). It says *where* errors live; confidence still orders them better.

**3. Validate any proposed signal on expert labels before believing it.**
Signals that beat confidence on 26 of 27 scenes against ESA WorldCover were
equal or worse on hand-labelled masks (exp13 vs exp18). Three explanations
for the gap have been tested and rejected (exp23, exp24, exp25), so this is
not a solved discrepancy you can reason around — measure it.

**4. If a reference product is all you have, run a no-model control.** A
pixel statistic (for water, the NDWI gradient) scored on the same errors
exposes reference-omission scenes, where it outranks every model signal
(exp06 and exp09, archived;
[v1.9.0](https://github.com/2imi9/olmoearth_inferenceX/blob/v1.9.0/docs/results/comparisons.md#no-model-image-statistic-controls-exp06)).

**5. Evaluate the way exp11-exp13 do.** Pre-register scene selection before
fetching scenes; hold out geographically; score with tie-aware AURC or
excess AURC across scenes; test with exact sign tests on untied pairs and a
block bootstrap; report wins, losses and ties, not means dominated by
high-error scenes. Details in [protocol.md](protocol.md).

**6. Report operating points alongside AURC.** Anyone acting on the ranking
needs the error fraction captured at a fixed review budget, not an area
under a curve. `assess_prediction` and `assess_classmap` in
`oe_inferencex/assess.py` report review sets and capture at 1, 5 and 10%
budgets; exp20 reports them on a served product.

**7. State accuracy with a coverage and a calibration check.** The
fine-tuned AWF model is 0.881 accurate on held-out expert points but
overconfident (ECE 0.080; 0.93 accurate where it says 0.99). Abstaining on
the 20% least confident points raises accuracy to 0.945 (exp21). A bare
accuracy without its coverage and reliability curve overstates the model.

## Do not

**8. Do not perturb by masking input content.** Occlusion instability was
the worst signal on every scene (exp08). The rule that generalizes:
perturbations that *preserve* scene content while changing the tokenization
expose model pathology; perturbations that *remove* content measure context
reliance instead.

**9. Do not use distance to the training scene as an error ranker.** It
fails in-domain (exp03, exp13) and on expert labels (exp18); its apparent
win under geographic shift was a scale artifact on a scene whose errors were
spectrally trivial (exp06, archived,
[v1.9.0](https://github.com/2imi9/olmoearth_inferenceX/blob/v1.9.0/docs/results/comparisons.md#no-model-image-statistic-controls-exp06);
exp13).

**10. Do not expect hidden-state signals to transfer from language models.**
Logit-lens settling, representation drift and attention entropy were all
worse than confidence (exp17).

**11. Do not build ensembles from one model family.** A stronger same-family
partner makes disagreement *worse* (exp10, archived;
[v1.9.0](https://github.com/2imi9/olmoearth_inferenceX/blob/v1.9.0/docs/results/comparisons.md#rater-strength-vs-diversity-exp10)),
equal-weight aggregation hurts (exp03, exp04, exp07), and Dawid-Skene
reliability estimation is invalid because the family errs together (exp07).
Disagreement from a different *input view* (the v1 band-set tokens) is
informative against WorldCover (exp17) but still loses on hand labels
(exp18).

**12. Do not fuse channels.** Prepending a reference-map check to boundary
proximity did not help (exp15), and no combination has been found that beats
confidence — so there is nothing to fuse yet.

**13. Do not read a reference-map check as model error without width
filtering.** OSM river centerlines mostly flag disagreements between
reference maps on narrow channels (exp15).

## For OlmoEarth deployments specifically

- **The served LCC rasters export no class confidence.** Bands 6-7 belong to
  the change-category heads and are not a class confidence (exp20). Until a
  class confidence is exported, boundary fraction of the class map is the
  only label-free cue available on the product; it captured a median 0.88 of
  water disagreements with WorldCover at a 5% review budget. Exporting the
  class head's top-1 minus top-2 logit alongside bands 4-5 would let the
  recipe's primary signal run on the product.
- **Read the product through its dataset card.** Threshold band 1 at 128
  before reading bands 2-9, and take the CRS from the file — the served
  rasters are EPSG:3857, not the UTM grid in the file name.
- **v1.2 differs from v1** in two ways that matter here: it uses rotary
  position encoding, which does *not* reduce sub-patch tiling instability
  (larger in v1.2, exp19); and it tokenizes Sentinel-2 as one band-set token
  per patch, so v1's band-set signal has no v1.2 counterpart.
- **Two of Ai2's fine-tuned models have been run end to end** from the
  published checkpoints (exp21, exp89; Forest Loss Driver report-only):
  confidence still ranks FT-AWF's errors best on its validation points and
  tiling instability ties it. Through FT-AWF's deployment configuration on
  2017 imagery, graded at the plots of a random sample that had imagery, its
  confidence ranked the errors with an AUROC of 0.64 under a one-to-one
  crosswalk, below the preregistered 0.70, and the record's own v1.2
  fine-tune did the same (exp99, exp103); the run cannot say whether the
  year, the place or the legend caused it. A preregistered blind test at
  890 fresh plots more than 100 km from the area did not repeat the miss
  for v1.2 (AUROC 0.729, interval 0.701 to 0.758), though most confident
  errors under the one-to-one crosswalk there were still the grass/shrub
  legend pair when standardised to exp99's label mix (0.611; 0.334 weighted
  by region) (exp111). Ai2's v1
  checkpoint did not reproduce its earlier predictions under rslearn 0.1.17
  (exp111), so pin the library version when rerunning a fine-tuned model.
  Forest Loss Driver, graded at
  2026 MapBiomas Alerta alerts in the Amazon in two of its ten classes,
  ranked its disagreements at 0.77, the preregistered bar on the point value,
  except the confident confusions between agriculture and mining (exp104).
  Mangrove, LFMC and EcosystemTypeMapping have not been run; the water
  results are for linear probes on frozen encoders.

## Which confidence on a multi-class map

On a binary map every form of the model's confidence is one ranking. On a
multi-class map rank by one minus the top probability, `form="top1"` in
`assess_prediction` and `signals.confidence`, which is computed tie-free from the
logits; the top-1 minus top-2 logit margin, the default kept from 1.0.0, was the
weakest form on Ai2's suite (exp76).

## Compute budget per reading

Every audit here scales with inference, not with labels. The margin, the
entropy, the boundary indicator and the spectral cues cost nothing beyond the
one forward pass that made the map. Crop dependence and tiling instability cost
one pass per offset (four here), the shift-averaged decision the same, and a
five-seed ensemble five; the record rejected the ensemble readings and keeps
the crop readings as map properties, not rankers. On an encoder whose forward
pass is dearer, a test-time-training design costs about four attention passes
per layer, the single-pass readings are the only affordable default at scale;
see [the ViT3 readiness page](../plan/vit3_readiness.md).

## What would change this recipe

A signal that beats confidence on expert-labelled dense maps, on more than
one task family, at a fixed review budget. None has been found. One change
has been made: a map read from different inputs in different places is ranked
and estimated per input condition (exp88). On PASTIS a probe trained on radar
plus optical and run without the optical input made confident errors, and a
whole-map rate misstated the cloudy part. A probe trained on radar alone made
few confident errors, but the two parts still erred at different rates, which
is why the estimate stays per condition. On CropHarvest China 6 the probe
trained on both inputs lost little without the optical input, and its
confidence tracked the lost information. Open items are in
[../plan/roadmap.md](../plan/roadmap.md).
