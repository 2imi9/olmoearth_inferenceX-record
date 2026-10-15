# Findings

This page summarises what the experiments established about assessing Earth-observation classification maps, without
labels and with a labelled sample. Each result is a claim in the [claim ledger](method/claims.md), checked against its
artifact under `exp/out/`, with its evidence in the [results record](results/comparisons.md).

## In short

- The model's own confidence ranks a map's errors far better than a random order. On Ai2's published embedding suite
  (24 tasks, linear probes on OlmoEarth Base) it gets a median 0.68 of the way from a random order to a perfect one,
  and a review of the least confident 10% finds about twice the errors a random 10% finds. It beats a random order in
  all 3,560 cells of sixteen encoders, their tasks and ten probe seeds.
- The suite's two controls, the distance in embedding space and how rare the probe's predicted class is, are near
  chance, so beating them there shows little. Informative controls were beaten on external references: ground
  survey (LUCAS), Dynamic World's expert tiles and farmers' crop declarations (EuroCrops).
- Exceptions include a water index (NDWI) that ranks flood errors as well as or better than confidence on
  Sen1Floods11 (one event under a Sentinel-2 probe, the whole test split under a Sentinel-1 probe) and on 2 of 45
  GEOID-Flood areas, both in one emergency activation, a sensor control on 1 GEOID-Flood area, and other encoders
  losing to a control on small or poorly fitted crop tasks (CropHarvest Togo, Nandi Sentinel-1).
- A model run on an input combination it was not trained on, such as radar alone under cloud for a model trained on
  radar plus optical, can be sure and wrong, and its errors then come late in the review order.
- Errors concentrate on prediction boundaries. Whether reviewing boundary windows first beats confidence alone depends
  on the event, the unit and the metric.
- Two inferences of one area differ mostly on prediction boundaries, the sensor moves shared errors more than the
  backbone, and which side is right needs labels, though few: drawn where two maps differ, 100 labels named the more
  accurate map on 60% of draws over 2,514 pairs of the record's probe maps, against 13% for 100 labels from the whole
  map, and 43% against 11% on the pairs where the 100 labels are a sample (exp90, not preregistered).
- A few hundred random labels give an error rate with an interval that holds its coverage, per-class accuracies and a
  certified zone, provided the labels are right. On two published products (LCMAP and Esri land cover) against
  LCMAP's random reference sample, the error-rate interval held and labels drawn where the maps differ found the
  better-agreeing map on 95% of draws with about 800 labels, where a whole-map sample needed more than 3,200 (exp92;
  found after the result); on LCMAP's own confidence the certified zone kept its guarantee and, at 300 labels and α
  half the error rate, returned no zone on most draws (exp93). On five more reference samples and seven products, in
  the United States, East Africa, Europe and worldwide from 2001 to 2020, the same gradings held, and three products'
  own confidences ranked their errors (exp94).
- On a second task, Ai2's Forest Loss Driver model graded at 2026 MapBiomas Alerta alerts in the Amazon (689 agriculture
  and mining alerts, labels drawn after training), the confidence ranked the disagreements with the analysts' drivers at
  a design-weighted AUROC of 0.770, the preregistered bar on its point value (interval 0.682 to 0.855), though not the
  confident confusions between agriculture and mining (exp104). <!-- claim:exp104-forest-loss-confidence-ranks-the-disagreements -->
- On another organisation's deployed product outside land cover, FTW's 2nd-edition global field map at 1,295 FTW
  benchmark test chips in Belgium, Croatia, Slovenia and Sweden, the map's own probability ranked its window errors at
  AUROC 0.837 to 0.936 and met the preregistered ranking bar in 3 of 4 countries; in Belgium the rarity of the predicted
  class came within 0.018 of it (exp110). <!-- claim:exp110-ftw-map-confidence-ranks-its-window-errors -->
  The exact interval, the certified zone and the which-map design kept their guarantees there. <!-- claim:exp110-ftw-labelled-routes-keep-their-guarantees -->
  300 random labels certified a zone on about half the draws, and 100 labels drawn where it and FTW's PRUE B3 model
  differ named the better-agreeing map in 3 of 4 countries (B3 in Belgium and Sweden; both maps' models most likely
  trained on these countries' train chips); the labels are parcel declarations and LPIS blocks taken as right, and the
  benchmark's 2-class truth turned out to be field only. <!-- claim:exp110-ftw-which-map-at-100-labels -->
- On three published products, disagreement with a second product (Esri) flagged their confident errors 4.1 and 5.0
  times as often as their confident correct rows on two of them and 1.45 times on the third, so the preregistered
  twofold on all three did not hold; a related product (NLCD for LCMAP) meets the ratio too, and most flagged confident
  rows are correct (exp106). <!-- claim:exp106-second-opinion-flags-confident-errors-on-two-of-three -->
- Stratifying the labels by the model's confidence, the package's default, gave a narrower valid interval for the
  overall error rate than stratifying by the map's classes, the usual practice, on 14 of 18 probe-map tasks at 300
  labels (the preregistered bar, met exactly; median width 0.85), and the gain comes from the strata (exp105). <!-- claim:exp105-confidence-design-narrower-than-map-class-strata -->
- A map made by Ai2's own deployment configuration for its fine-tuned AWF model, re-run with probabilities kept, gives
  the evaluation's class at 92% of Ai2's validation points that fall on it, and its confidence ranks its errors there
  (AUROC 0.85). Its accuracy there was not shown to be within 5 points of the evaluation replica's, and no difference
  was shown. The points, placed by experts, sit in the confident part of the map and give no whole-map accuracy
  (exp98).
- The same configuration on 2017 imagery, graded at an independent random sample around the AWF area, is wrong at 35%
  of the plots with imagery under a one-to-one crosswalk and 13% under a lenient one. Its confidence ranks those errors
  with AUROC 0.64 (90% interval 0.58 to 0.696), below the preregistered 0.70, though the least confident 10% still hold
  errors beyond chance (0.43 of the gap closed against a bar of 0.25, interval 0.10 to 0.65). The plots without imagery
  are not missing at random, so these figures describe the plots with imagery, not the whole region (exp99). A
  fine-tune of OlmoEarth v1.2 for the same task gives the same verdicts at both samples (exp103).
- No signal from inside the encoder beats confidence, and a consensus of encoders does not estimate accuracy.
- For a language-model agent at 7B the package is decisive when its tools are given directly; the agent as shipped
  found them on 17 of 40 runs. At 27B a model with a numpy sandbox nearly matches the package's ranking, and every arm
  without a sandbox declined to pick between two maps, including one with no package and no data, so the benchmark
  does not show that the package causes the decline.

![The assessment pipeline on a real scene: Sentinel-2 bands, the frozen encoder and head, prediction, confidence and boundary layers, the review set at a 5% budget, the cues per window](figures/pipeline.png)

*One scene (okavango_80) through the pipeline. The layers are the actual rasters; the orange squares are the 5% review
set in the boundary-first order.*

## Ranking errors without labels

The package scores each window, the block of pixels reviewed as a unit, by the model's confidence averaged over the
window's pixels: the top class probability for class probabilities, the gap between the two highest logits with
`--logits`, and for a two-class probability map the distance from 0.5. The experiments graded the margin between the
two highest class probabilities of the window's mean probabilities; exp76 compares the forms. A ranking is graded by
excess AURC, the area under the risk-coverage curve minus that of a perfect ranking, against a random order and
against controls that do not read the model's confidence, such as class rarity (how rare the predicted class is),
embedding distance or the normalised difference water index (NDWI).

On the 24 tasks of Ai2's published embedding suite for which a margin is defined, read through linear probes on
OlmoEarth Base, the margin takes a median 0.68 of the gap between a random and a perfect ranking, and labels reach a
fifth to a third of the remainder (exp70, exp65). <!-- claim:margin-takes-two-thirds-of-the-ranking-headroom -->
That median averages over every review budget. A review of the 10% least confident units finds a median 0.214 of the
errors, against 0.10 for a random 10% and 0.410 for a perfect order (exp70). <!-- claim:suite-review-at-ten-percent -->
Over sixteen encoders, their tasks and ten probe seeds, the margin ranks errors better than a random order in all
3,560 cells; the lowest AUROC is 0.503, close to chance (exp79). <!-- claim:exp79-margin-beats-random-everywhere -->
The margin also beats the better of the suite's two controls, how rare the probe's predicted class is and the
distance to the training mean, on all 24 tasks (sign test p = 6e-08) and on all 14 distinct sources (p = 6.1e-05)
(exp70). <!-- claim:suite-margin-wins-every-task -->
Those controls are near chance on this suite: the better one closes a median 0.094 of the gap and is no better than a
random order on 9 of the 24 tasks (exp70). <!-- claim:suite-controls-are-near-chance -->
The preregistration was not blind on six of the seven segmentation tasks, whose margin scores were already on record
from exp54 and exp51; on the other 18 tasks the margin beats both controls on 18 of 18 (p = 3.8e-06). One of those
18, AWF Sentinel-2, had been graded before on the AWF project's own split, where confidence ranked best (exp04,
exp16); on the other 17 the margin wins 17 of 17 (p = 7.6e-06).
<!-- claim:suite-six-segmentation-outcomes-were-on-record -->
Under ten probe seeds OlmoEarth Base's lead over the better control stays positive on all 24 tasks (exp79).
<!-- claim:exp79-base-margin-wins-under-every-seed -->
On all sixteen encoders the margin beats the better control on at least 75% of each encoder's tasks under every seed,
the lowest share being 21 of 24 for Clay Large (exp79). <!-- claim:exp79-headline-holds-under-every-seed-on-every-encoder -->
Over the other fifteen encoders and ten seeds, every loss is on the three 306-sample CropHarvest Togo tasks or on Nandi
Sentinel-1, where the probe is at most 0.40 accurate (exp84). <!-- claim:exp84-losses-are-small-or-badly-fitted-tasks -->
It also beats a five-seed probe ensemble on 22 of 24 tasks and nearest-neighbour and Mahalanobis distances on 24 of
24 (exp73). <!-- claim:suite-margin-beats-the-strong-alternatives -->
On the seven segmentation tasks the ensemble's primary reading, vote disagreement, is near random (AUROC 0.505 to
0.537); against the spread of the five seeds' margins the margin wins 7 of 7 by a median 0.018 (exp73).
<!-- claim:exp73-segmentation-ensemble-vote-is-near-random -->
One minus the top probability ranks slightly better than the margin on 14 of 16 multi-class tasks; on the 8 binary
tasks the two give one ranking (exp76). <!-- claim:top1-beats-the-margin-on-multiclass -->

The controls carry information on the external references, and the margin beats them there too. On 4,778 LUCAS
polygons, whose reference is a surveyor's observation on the ground, the margin beats the best control computed from
the imagery, the pixel variance inside a window, on 94 regions against 32. Its lead over a signal read from the map
alone, how much a window's class disagrees with its neighbours, is smaller: 0.029 of excess AURC (cluster bootstrap
0.011 to 0.049) (exp68). <!-- claim:lucas-ranking-survives-ground-observation -->
Dynamic World's own margin ranks its errors better than a class-rarity control on 315 expert-annotated tiles against
91 (exp67). <!-- claim:dw-margin-ranks-a-production-model -->
That control finds 0.217 of Dynamic World's errors in a 10% review, about twice a random 10%, and on EuroCrops the
class-rarity control ranks errors with an AUROC of 0.70 in two of the three countries (exp67, exp69).
<!-- claim:external-references-carry-informative-controls -->
On DFC2020 land cover the margin beats the NDWI control on all three sensor arms (exp66).
<!-- claim:dfc2020-margin-beats-pixel-control -->
On 106,274 windows of farmers' crop declarations it beats the best no-model control in all three regions and on all
but four of 147 grid cells (exp69). <!-- claim:eurocrops-ranking-holds-on-declarations -->

On Sen1Floods11 Bolivia, one flood event, NDWI ranks the model's errors as well as its confidence under OlmoEarth v1
and better under v1.2 (exp45, exp47). <!-- claim:bolivia-ndwi-exception -->
Under the Sentinel-1 probe NDWI instead beats confidence on the multi-region test split and loses on Bolivia (exp51).
<!-- claim:s1-probe-ndwi-flip -->
The fine-tuned Sentinel-2 model's confidence beats NDWI on Bolivia under both backbones, so that exception belonged to
the frozen encoder (exp52). <!-- claim:finetune-dissolves-bolivia-exception -->
On 45 GEOID-Flood areas of interest NDWI wins on 2 for permanent water and a sensor control on 1 for post-event water;
both permanent-water cases lie in one Copernicus emergency activation, an exception rate of one activation in nine
(exp55). <!-- claim:geoid-exception-rate -->
The ordering among confidence readings degrades as the probe overfits, by four ten-thousandths of excess AURC across
the suite, and only the severe memorisation of a LUCAS probe reversed it (exp68, exp70).
<!-- claim:suite-probe-gap-degrades-the-ordering -->
On the median GEOID-Flood area the 5% of windows that confidence flags hold 88% of the permanent-water errors and 64%
of the post-event water errors, 17.5 and 12.8 times a random 5%, and 16.9 and 12.1 times when areas are clustered by
activation (exp55). <!-- claim:geoid-capture-effect-size -->

## Review order and explanations

On Bolivia hand labels 75% of the error windows lie on a prediction boundary, against 21% of the correct windows
(exp36, exp37). <!-- claim:errors-sit-on-boundaries -->
There, reviewing boundary windows first, in confidence order, and then the interior catches more errors than
confidence alone at review budgets (shares of windows reviewed) of 5% and 10%, as preregistered, and not at 20%
(exp36). <!-- claim:boundary-first-review-order -->
Inside confidence's review set on Bolivia the error rate is 0.38 at a 5% budget, 0.33 at 10% and 0.26 at 20%, against
8.8% overall (exp37). <!-- claim:review-set-error-rate -->
Across GEOID-Flood events neither order wins consistently. At a 5% budget the boundary-first order captures more
errors than confidence on 27% to 40% of events by target and fewer on 20% to 40%, the rest tying; by per-event excess
AURC it is ahead on 28 to 30 of about 45 events, by tiny margins; pooled, the two tie (exp55).
<!-- claim:geoid-boundary-first-minority -->

Each flagged window lists its label-free cues. A cue's enrichment, its share among error windows over its share among
correct ones, is 3.5 on Bolivia for a prediction boundary (0.750 against 0.214), 3.6 for the least confident 20%
(0.589 against 0.163) and for tiling instability (0.583 against 0.164), and 7.2 for spectral ambiguity, NDWI near zero
(0.483 against 0.067), with error rates of 0.25, 0.26, 0.26 and 0.41 among windows carrying each (exp37).
<!-- claim:cue-enrichment-table -->
On the suite's seven segmentation tasks the boundary cue's enrichment runs from 1.24 on m-cashew-plant, where 76% of
windows sit on a boundary, to 8.13 on MADOS, where 10% do; its rank correlation is 1.00 with the ceiling that boundary
prevalence sets and -0.26 with class count, and the preregistered bar of 2 fails on m-cashew-plant (exp82).
<!-- claim:boundary-cue-enrichment-is-fragmentation-not-class-count -->
Coarsening OlmoEarth Base's map to 2 × 2 and 4 × 4 blocks lowers the cue's enrichment on all seven tasks at both
steps (exp82). <!-- claim:boundary-cue-weakens-with-the-window-grain -->

The fine-tuned AWF model is 0.93 accurate where it claims 0.99, so a stated accuracy needs a coverage; its 80% most
confident windows are 0.945 accurate against 0.881 overall (exp21). <!-- claim:accuracy-needs-coverage -->
The served land-cover-change rasters export no class confidence, and boundary fraction alone captures a median 0.88
of their disagreements with WorldCover water at a 5% budget, on the sites that have any (exp20).
<!-- claim:served-product-boundary-triage -->

## The demo's tile

`oe-inferencex demo` assesses one Dynamic World tile with expert annotation, bundled with the package. The tile was
chosen by a rule fixed before any tile was examined: the lower median by error capture at a 5% budget among the 18 of
Dynamic World's 409 expert-annotated test tiles that are annotated on at least 90% of windows, have at least three
classes on 5% of the windows each and have an error rate between 5% and 35%. It is a typical tile, not the most
favourable one. <!-- claim:demo-sample-is-the-median-tile -->
On that tile 19% of the scored windows are wrong. Of the 5% of windows flagged without labels, 67% are wrong; they
hold 17% of the map's errors, where no selection of 5% could hold more than 26%. The 10% flagged hold 32% and the 20%
flagged 55%, and 97% of the 5% flagged sit on a boundary of the model's own map. <!-- claim:demo-sample-hit-rate -->

## Comparing two inferences

The comparison module measures how two inferences of one area differ. On all 16 preregistered pairs, across crop
offsets, backbones, sensors, encoders and fine-tuning, disagreement windows lie on a prediction boundary 3.3 to 7.1
times as often as agreeing ones on Sen1Floods11 and 21 times on the median GEOID-Flood event (exp57).
<!-- claim:atlas-disagreement-is-boundary-located -->
Reading Sentinel-1 instead of Sentinel-2 moves a flood model's error set far more than swapping the backbone (phi, the
correlation of two error masks, 0.38 to 0.39 against 0.70), so the sensor is the largest lever and the backbone the
smallest (exp46). <!-- claim:modality-dominates-shared-errors -->
Fine-tuning corrects 35 of the frozen probe's 63 errors on the same AWF windows, so the fixed window grid is not their cause
(exp21). <!-- claim:fine-tuning-corrects-half -->

Labels favour one side only where the difference changed the model; the fine-tuned model is right on 71% to 75% of
its disagreements with the frozen head, while under crop offsets and backbones either side is right on 39% to 61%
(exp57). <!-- claim:atlas-which-side -->
Without labels, the side with the larger margin is right on 51% to 70% of disagreement windows, above a coin flip on
all 15 pairs and short of the preregistered 55% on four pairs, so the prediction fails (exp58).
<!-- claim:tool-vs-diff-resolution -->
A side rule fitted to labels on frozen pairs and forced onto the frozen-against-fine-tuned pair is right on 43.8% and
49.2% of differing windows, below the raw margin's 61.1% and 55.4% and the pair's own fit's 74.9% and 72.8%, so the
module refuses another model family unless forced (exp65). <!-- claim:calibrate-family-lock -->

With labels drawn among the windows where two maps differ, the package says which is more accurate (`sample --other`,
then `estimate`). On every pair of encoders run on the same units of the suite's 24 tasks (2,514 pairs with the full
truth; not preregistered; linear probes, not map products), 100 such labels named the right map on 60.3% of draws,
against 12.7% for 100 windows of the whole map analysed on the differing windows they held, and 42.8% against 11.3% on
the 1,709 pairs that differ on more than 100 windows. The lowest coverage over the pairs was 95.5%, and the wrong map
was named on at most 4.0% of a pair's draws (exp90). <!-- claim:exp90-which-map-few-labels -->
These are the interval's verdicts; from 1.10.0 the exact sign test decides while the labels are at most a fifth of the
differing windows, where it named the better map more often on five of nine product pairs (exp107).

Two Sentinel-2 acquisitions a median 118 days apart change decision on 31.2% of 7,109 LUCAS polygons against a 0.30%
reseed floor, a rate that follows phenology, 44.5% on cropland against 20.8% on woodland (exp68).
<!-- claim:lucas-two-dates-are-phenology -->
Because EuroCrops labels both years, the floor for a two-date change rate is measurable there; where the declared crop
did not change, 0.060 to 0.366 of windows change decision, against 0.589 to 0.843 where it did, a floor one to two
orders of magnitude above the reseed rate (exp69). <!-- claim:eurocrops-the-labelled-floor-for-a-two-date-difference -->
Where two years' inferences differ, the model is right about both on 0.051 to 0.367 of windows, the difference being
the model following a real crop rotation (exp69). <!-- claim:eurocrops-a-difference-can-be-the-model-tracking-the-ground -->

## Estimating accuracy from a labelled sample

From 300 random windows the design-based 95% interval for the error rate covers the truth on 0.933 to 0.961 of 2,000
draws on all seven segmentation tasks (exp78). <!-- claim:design-based-interval-is-honest -->
The same budget spent as 18 tiles of 16 windows gives an ordinary independent-sample interval that covers on 0.506 to
0.777 of draws on six of the seven; a cluster correction restores 0.913 to 0.932 on the four tasks with full tiles and
fails on MADOS, whose tiles hold 1 to 400 windows (0.598) (exp78). <!-- claim:tile-sampling-breaks-the-naive-interval -->
The tiles design as the package ships it, which takes tiles until the budget is met and uses a t quantile, covers on
0.945 to 0.954 of draws on five tasks and falls short on Sen1Floods11 (0.843) and MADOS (0.685), the two tasks where a
tenth of the tiles hold most of the errors (exp78). <!-- claim:shipped-tile-design-coverage -->
Choosing the labelled windows by confidence saves up to 2.51 times the labels at equal half-width, above the 1.8 the
preregistered honesty check allowed (exp78). <!-- claim:confidence-saves-a-quarter-to-a-half-of-the-labels -->

The same labels certify a trust zone, the largest most-confident part of the map with error rate at most α, and the
certificate may fail with probability at most δ. At δ = 0.1 the zone certified by the exact-test rules exceeded α on
at most 0.080 of 2,000 draws in all 112 cells of the suite, and the plug-in rule's on up to 0.557 (exp80). <!-- claim:trust-zone-guarantee-holds -->
With 300 labels and α at half the error rate, the prefix rule certifies a zone on most draws on 14 of 21 tasks,
covering a median 0.50 of the map (exp80). <!-- claim:trust-zone-coverage-at-300-labels -->
On a published product's own confidence (LCMAP Collection 1.3's `lcpconf`, graded against LCMAP's random reference
plots; preregistered) the guaranteed rules held δ in all 24 cells, the prefix zone exceeding α on at most 6.1% of
draws; at 300 labels and α 8.86%, half the error rate, the plug-in's exceeded it on 46.6%, and the prefix rule
returned no zone on 87% of draws, its first zone, a tenth of the plots, being 6% wrong with about 30 labels (exp93). <!-- claim:exp93-zone-guarantee-holds-on-a-product -->
Adding labels can make certify certify less: a level is tested once its zone expects the b_min labels a clean zone
needs, then holds fewer than b_min about half the time, and the prefix rule must pass it first. On 34 committed maps
the mean certified coverage fell by more than 0.05 of the map when labels were added on 14 (exp95, preregistered); a
cut at three times b_min halved those falls but certified less at about 100 labels on 24 of the 34.
<!-- claim:exp95-standard-cut-falls-when-labels-are-added -->
A budget-aware cut designed after that, the ramp, met its
preregistered bar on 332 held-out maps at its seed (it halved the fall on 77% of the maps where it exceeded 0.05), but
across 9 seeds the bar held on 5, so certify keeps the standard cut and offers the ramp as an option (exp96).
<!-- claim:exp96-ramp-passed-at-its-seed-and-not-across-seeds -->
The guarantee holds for one budget fixed before the
labels: certifying again after adding labels, at every budget of a doubling ladder, gave a zone wrong more than alpha
on 21% to 30% of draws on six maps where one budget's certificate is wrong at most 12.25% of the time (exp96's audit).
<!-- claim:certify-guarantee-is-per-budget-not-per-look -->
A certificate that holds at every look exists and was graded: zones tested by e-processes in a fixed sequence from an
anchor fixed before the labels kept the guarantee on 366 maps, the largest violation 0.1175 at δ 0.1 over 3,122 rates
(exp97, preregistered). <!-- claim:exp97-sequential-certificate-holds-at-every-look -->
certify as shipped, rerun after every 5% more labels (a median 77 looks), certified a zone wrong at some look more than
17.5% of the time on 132 of the same maps. <!-- claim:exp97-rerunning-certify-breaks-its-guarantee -->
The price is labels and an anchor: to reach half the largest mean certified share the one-look rule reaches, a median
1.71 times as many labels, and no zone smaller than the anchor is ever certified.
<!-- claim:exp97-sequential-certificate-costs-labels -->
The package's intervals for user's accuracy (how often a class call is right), producer's accuracy (how much of a class is found)
and class share leave 14 of 628 cells below 0.93 coverage and none below 0.879; the Wald form common in the literature
leaves 107 of 522, the worst at 0.019 (exp81). <!-- claim:per-class-intervals-wald-fails-wilson-nearly-holds -->
Choosing fine-tuning tiles by the audit's suspicion is worse than random on the multi-region test split at every
budget from 100 to 1,000 labels, and the preregistered predictions of a saving fail (exp56).
<!-- claim:audit-does-not-save-labels -->

On published products against an independent probability sample (LCMAP Collection 1.3 and Esri 10 m land cover for
2018, read at LCMAP's 25,000 simple-random reference plots; preregistered), 300 random plots gave an error-rate interval
that covered on 95.0% and 95.65% of draws; labels drawn among the 21.6% of plots where the maps differ covered their
2.7-point accuracy difference on at least 98.5% of draws at 50 to 200 labels and never named the wrong map (they named
LCMAP on 6% to 29% of draws), at about half the width of labels drawn from all plots (exp92). <!-- claim:exp92-products-labelled-routes-hold -->
LCMAP's own confidence ranks its errors there with an AUROC of 0.753, and its 10% least confident plots hold 28.7% of
the errors (exp92). <!-- claim:exp92-lcmap-confidence-ranks-its-errors -->

On five more reference samples from four sources (NLCD's 2011 and 2016 accuracy-assessment points, JRC's GFC2020
validation set, the East Africa TimeSync sample, the S2GLC validation set in Europe) and seven products (NLCD, LCMAP,
GFC2020, WorldCover, CGLS-LC100, Esri 10 m, ODSE-LULC) from 2001 to 2020, with the points standing in for the map
(preregistered), labels drawn where two products differ covered their difference on 98.05% to 99.25% of draws in all
eight pairs, at a third to two thirds of the width of labels drawn from all points, and named the worse-agreeing
product on at most 0.35% of draws; where the products differ by 4 to 6 points, 200 such labels named the
better-agreeing one on 83% to 95% of draws, against 13% to 25% for labels from all points. The exact error-rate
interval covered on 95.05% to 96.25% of draws in all 18 cells (exp94). <!-- claim:exp94-labelled-routes-hold-on-four-references -->
LCMAP's, CGLS-LC100's and ODSE-LULC's own confidences rank their errors there, AUROC 0.720 to 0.787 in nine cells, and
on them the certified zone's guaranteed rules were wrong more than α on at most 2.8% of draws, the plug-in's on 39% to
53% (exp94). <!-- claim:exp94-product-confidences-rank-their-errors --> <!-- claim:exp94-zone-guarantee-holds-on-three-products -->

## A deployed OlmoEarth map at Ai2's own points

The record re-ran Ai2's FT-AWF deployment configuration with olmoearth_run over the AWF project's request geometry
(2023, 10 m, probabilities written). This is the record's re-run, not the map Ai2 published. It was graded at the
259 of Ai2's 344 validation points that fall on the map (exp98, preregistered). The re-run is 87.3% accurate there,
against 89.6% for the evaluated replica on the same points. Reproduction within 5 points was not shown: the 90%
interval of the difference, −5.1 to +0.8 points, reaches 0.15 points past the margin. No difference was shown either:
the interval contains 0, and the sign test gives p = 0.26. <!-- claim:exp98-equivalence-within-5-points-not-shown -->
The re-run gives the replica's class at 92% of the points. Its own confidence ranks its errors there with AUROC 0.849,
and the least confident 10% of the points hold 42% of its errors. <!-- claim:exp98-deployment-gives-the-evaluations-classes-and-ranking -->
Ai2's points sit in the confident part of the map: on average a point is more confident than 65% of the map's pixels.
The points were placed by experts, so they give no whole-map accuracy. If points and pixels of equal confidence were
equally accurate, which the design cannot check, the map would be less accurate than the points, since at the points
the error rate rises as the confidence falls.
<!-- claim:exp98-validation-points-sit-in-the-confident-part -->
At the 47 plots of an independent random sample inside the area (East Africa TimeSync, 2017 labels), the 2023 map
disagrees at 25 under a strict crosswalk and at 3 under a lenient one. So the crosswalk between the two legends, not
the map, decides most of that figure, and neither is the 2023 map's error rate (report-only).

## The same deployment on 2017 imagery, at an independent random sample

The record ran the same configuration on 2017 imagery over the windows holding the 309 East Africa TimeSync plots
within 100 km of the AWF area (a simple random sample per country; exp99, preregistered). 42 of the 313 windows got no
2017 imagery, so 271 plots were graded. The 38 without imagery are not missing at random: 20 of the 47 plots inside
Ai2's geometry lack it, against 18 of the 262 outside. So every figure below describes the plots with 2017 imagery, not
the whole region. The map's confidence ranks its errors with a design-weighted AUROC of 0.638 under a strict crosswalk
(90% interval 0.579 to 0.696), below the preregistered 0.70, and 0.678 under a lenient one, whose interval contains
0.70. The frozen rule calls this "transfer is the likelier reading". The run does not separate the year, the place, the
legend boundary and the move from Ai2's expert-placed points to a random sample. <!-- claim:exp99-confidence-misses-the-ranking-bar-under-both-rules -->
The least confident 10% of the plots close 0.434 of the gap from a random review to the best one, so the second
prediction holds on its point value. Its interval, 0.10 to 0.65, includes the bar of 0.25. <!-- claim:exp99-least-confident-tenth-closes-the-gap-at-the-point-value -->
The error rate is 35.5% under the strict crosswalk (exact interval 26.0% to 45.8%) and 13.4% under the lenient one
(7.5% to 21.9%), so whether it is more than 5 points worse than Ai2's 10.5% is not determined. Most of the difference
between the two rules is the grass/shrub boundary, which AWF does not define. Filling the 38 plots without imagery as
all right or all wrong moves the strict rate between 31.2% and 43.4%, and the lenient one between 11.8% and 24.1%
(report-only). <!-- claim:exp99-error-rate-of-plots-with-2017-input -->

The record's own fine-tune of OlmoEarth v1.2 Base for the same task, with Ai2's recipe and labels (exp102), gives the
same verdicts under the same frozen rules (exp103, preregistered). At Ai2's points its confidence ranks the errors with
AUROC 0.814, and the least confident 10% hold 37.5% of them. <!-- claim:exp103-v12-ranking-holds-at-ai2s-points -->
At the 2017 plots the strict AUROC is 0.642 (90% interval 0.582 to 0.703), below 0.70. The lenient one, 0.730, meets
the bar on its point value, but its interval contains 0.70, as v1's did, so this is not a difference between the
models. The two models share the task, the labels and the region, so this is a replication across model versions, not
an independent test of the shift. <!-- claim:exp103-v12-transfer-ranking-misses-under-strict -->

## The package as an agent tool

On forty task cards with three preregistered predictions (exp64), Qwen3.8-27B-NVFP4 with the package as tools
reproduces its review set, grounds 99.4% of its stated numbers in tool outputs and declines to pick a side on every
comparison card; the OlmoEarth Agent as shipped found the tools unaided and did the same.
<!-- claim:agent-benchmark-tool-arm-reproduces-the-package -->
A model given only numpy and the arrays captures 0.904 of the package's errors, so the ranking prediction fails.
<!-- claim:agent-benchmark-sandbox-rediscovers-the-ranking -->
It grounds 94.2% of its numbers against 99.4%, short of the preregistered gap, so the grounding prediction fails.
<!-- claim:agent-benchmark-grounding-not-decisive -->
The third prediction holds as graded, the tool arm declining the side question on 10 of 10 comparison cards against
the sandbox's 1 of 10. It does not show that the package causes the decline: arm D, with only the card's text and no
package or data, also declined on all 30 of its comparison runs, and the rubric scores any decline as full marks.
<!-- claim:agent-benchmark-decline-holds -->
With Qwen2.5-7B-Instruct all three predictions hold: given the package's tools directly, the tool arm grounds 100.0%
of its numbers against 0.1% and captures 0.997 of the package's errors against 0.144.
<!-- claim:agent-benchmark-package-is-a-floor-at-7b -->
At that size the agent as shipped calls the review-set tool on 17 of 40 runs, rejecting the prediction that it
reproduces the package. <!-- claim:agent-benchmark-7b-agent-does-not-find-the-tool -->

Since 1.4.0 the package serves its commands to an agent as a local MCP server. In a pilot on 2 October 2026 (five
questions on the outputs of Ai2's FT-AWF and Forest Loss Driver models; not preregistered, no committed artifact), the
tools' numbers were right and a strong model's replies kept them with their limits, while a small model's replies
failed on all five, mostly by dropping limits; after the texts put each tool's main limit inside its conclusion, two of
its five passed. `estimate` and `certify` refuse a review set, so advice to label one cannot become a wrong number,
though a reply can still mislead.

## What was tried and rejected

The [technique ledger](TECHNIQUES.md) lists each rejected technique with its reason. No signal from the encoder's
internals, its pretraining objective, a posterior over the probe head, feature-space typicality, a second model of the
same family, or flip-and-rotate consistency ranks errors better than confidence on expert labels.
<!-- claim:no-encoder-internal-signal-beats-confidence -->
As a side result, OlmoEarth v1's latent masked-image-modelling pretraining target has effective rank 2 on real
scenes, and a whitened target is 57% to 70% predictable from context (exp32 to exp34).
<!-- claim:target-effective-rank-2 -->

## Limits

The hand-labelled flood testbeds are one event and one split of the same dataset, and the suite is read through
linear probes on Ai2's embeddings. A confidence ranking reviews the model's confident errors last. They are common
when a model is run on an input combination it was not trained on. On PASTIS a probe trained on radar plus optical
and read on radar alone, as under cloud, was 73.6% wrong, and 59.8% of OlmoEarth Base's errors looked as confident as
a typical correct window, against 6.0% with both inputs; OlmoEarth Large's share rose only from 5.7% to 12.8-13.8%.
<!-- claim:missing-optical-errors-are-confident -->
A probe trained on radar alone was 28.4% wrong and ranked its errors with an AUROC of 0.79, against 0.83 with both
inputs; 3.9% of its errors reached the same threshold, which is set by the probe trained on both. So the risk is a model read on inputs it was not trained on, not the missing input as such (exp88).
<!-- claim:exp88-matched-head-ranks-normally -->
A whole-map error rate can still misstate a part read from other inputs. On a map with half its tiles read by the
first probe without the optical input, random samples estimate 46.9% on average, while the cloudy half's rate is
74.1%. Even with the radar-only probe on the cloudy half, the two halves err on 28.3% and 19.7%. Where a map records
each pixel's input condition, the package ranks, samples and certifies each condition on its own (`--condition`, since
1.4.0; exp88). <!-- claim:pooled-error-rate-misstates-the-cloudy-part -->
Every interval and certified zone describes agreement with the reviewer's labels, which the package treats as right.
If the reviewer makes mistakes, the true rate can fall outside them; no experiment here measured how often human
reviewers err (exp91 measured it for a vision-language model). Labelling blind, with the map's class hidden, keeps the reviewer from anchoring on the map's class;
how much that changes the labels was not measured. Since 1.5.0 a reviewer can write `?` for a window that cannot be
judged; `estimate` bounds it both ways and `certify` counts it as wrong. `estimate --reviewer-false-alarm E0
--reviewer-miss E1` widens the interval to the sharp bounds for a reviewer whose error rates the user states (proved in
`lean/` of the [v1.9.0 tree](https://github.com/2imi9/olmoearth_inferenceX/tree/v1.9.0/lean)); the rates are the
user's, not measured, and `certify` takes none.
Claude (claude-opus-5-5), shown Sentinel-2 views of Sen1Floods11 Bolivia blind, was not a substitute for the reviewer:
it judged 104 of 150 random windows and agreed with the hand labels on 85.6% of those (the map, on the same windows,
95.2%, found after the result); with its 46 ? counted both ways its labels gave an interval of 4.2 to 46.9% where the
hand labels gave 5.2 to 15.2%, and on 30 wrong calibration windows it missed 14 of the 19 it judged, which widens the
interval to 0 to 100% (exp91, preregistered). <!-- claim:exp91-vlm-reviewer-shares-the-maps-errors -->
On Sen1Floods11, where eight encoders share 82% to 87% of their errors, a Dawid-Skene consensus, which estimates each
encoder's accuracy from agreement alone, returns 0.975 to 0.983 for maps 0.883 to 0.914 accurate, and its rank
correlation with the true accuracies misses the preregistered 0.8 on MADOS (0.71) (exp83).
<!-- claim:consensus-order-recovered-only-where-true-gaps-are-large -->
Open items are in the [roadmap](plan/roadmap.md).

## How a claim is recorded

A candidate signal is scored beside the model's confidence and a no-model control on identical windows, with the
primary test written down before the run; expert labels grade a signal and never train it. Each claim is an entry in
`docs/claims.yaml` whose check must hold on its artifact. Claims from exp79 on also name a second test that reaches
the number by another route, and some earlier claims do too; most claims before exp79, the suite's headline among
them, have only their check. The full rules are in the [protocol](method/protocol.md).
