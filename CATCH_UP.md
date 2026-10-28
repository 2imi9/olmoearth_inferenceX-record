# Where this stands: a five-minute catch-up for the OlmoEarth team

Ziming Qi, 17 October 2026; updated 27 October 2026.

## What it is

A checker for Earth-observation maps. It reads what a model writes out (per-class scores, or a product's class map
with its confidence) and tells you:

- where the map is likely wrong;
- how wrong it is, from a small set of labels;
- which part you can trust;
- which of two maps is better.

It never runs the model, so it works on OlmoEarth's outputs, other models' maps and published products.
`pip install "olmoearth-inferencex[geo,mcp]"` (1.20.0). Code and pictures: https://github.com/2imi9/olmoearth_inferenceX

Words used below: a **window** is a small block of pixels (4 by 4 by default); the **margin** is the top class score
minus the second; the **review set** is the least confident windows, to check first (it is not a sample, so it says
nothing about the error rate); **capture** is the share of a map's errors in the review set; **AUROC** is how well the
confidence ranks errors above correct windows (0.5 is chance, 1 is perfect); a **valid interval** holds at any sample
size, where a normal-theory one holds only approximately in large samples.

## With an agent

`oe-inferencex mcp` is a local MCP server with nine tools: where to look first, which windows to label, the error
rate, the part to trust, two maps compared, a label budget, a typed answer (yes, no or undetermined; which map; a
share) and a certificate check, plus a guide. Each tool but the guide replies with the conclusion, its limits and the next step, so an agent can quote the
limit with the number.

An earlier benchmark (exp64, 16 September, before the MCP server) tested two of these steps, the review set and the
two-map comparison. Each **card** is a map with hidden labels (40 cards, all binary water tasks). Each **arm** is what
the model is given: A the package's two commands as tools, B numpy and the same arrays, C a template with no model, D
the card's text only. Arms A to D were preregistered. A fifth arm, not preregistered, was my OlmoEarth Agent
(github.com/2imi9/OlmoEarth-Agent, a research demo on your Studio API), with a review-set skill I wrote for it.

- **Qwen3.8 27B (NVFP4).** Two of the three preregistered predictions failed. With numpy alone, the model grounded
  94% of its numbers in its own output (99.4% with the tools) and matched most of the ranking (0.90 of the package's
  capture; it was handed the margin by name). But it gave no gradeable answer on 27 of 120 runs, against none with the
  tools. The third prediction, declining to pick a map without labels, held, but the text-only arm declined as often,
  so it does not show what the package adds. My agent, free to choose among its 41 tools, picked the review-set tool
  on 40 of 40 runs and the compare tool on 10 of 10 comparisons, and reached 0.96 of the package's capture.
- **Qwen2.5 7B.** With the tools, the model grounded 100% of its numbers and reached 0.997 of the package's capture;
  with numpy alone, 0.1% and 0.14. The reasons it gave for each window held on only 55% of them (99.5% at 27B). My
  agent called the tool on 17 of 40 runs, and on 30 with the skill pinned.

With these two models on binary water cards: the 27B model got most of the ranking with numpy alone, and the tools
removed its unanswered runs; the 7B model needed the tools and still misstated its reasons on 45% of windows; and my
agent at 7B called the tool on 17 of 40 runs unaided and on 30 with the skill pinned (0.76 of the package's
capture), so pinning helped but did not make it reliable.

## New since 23 October (1.14.0 to 1.20.0)

- **Grading a map with an agency's own sample.** `grade` reads a published stratified reference sample. On NLCD 2016
  and GFC2020 it reproduces the published overall accuracies, and gives each accuracy a valid interval beside the
  agency's normal-theory one (exp117).
- **Normal-theory per-class intervals fall short in simulation.** In 5,000 simulated samples at NLCD 2016's design
  (populations built from its sample copied ten times), the usual normal-theory intervals covered below the
  preregistered bar (94.1%) on 33 of 91 quantities, all per class, one as low as 71%. On overall accuracy they covered
  94.8% to 95.1%. The valid intervals held on all 91, at 1.6 times the width overall and 1.7 to 2.4 times per class
  (medians), up to 8.4 times (exp118).
- **NLCD's Table 4.** `grade` reproduces every per-class accuracy by the primary label (15 user's, 15 producer's). The
  table's alternate-rule user's accuracies (a point counts right if the map matches its primary or alternate label)
  match, 15 of 15, only when the 2011 primary label is also accepted, among the rules tried; its alternate-rule
  producer's accuracies match no rule tried (the best, 7 of 15) (exp117).
- **Per-profile verdicts** (`sample --design profile`). A map split before labelling into up to 12 profiles
  (confidence band, class, grid block, input condition) gets good, bad or undecided for each; all the verdicts are
  right together except on at most 10% of samples by default. Checked by exact enumeration in the package's tests,
  not yet on a real map.
- **Two maps, proved.** With labels where two maps differ, the sign test names the less accurate map at most 2.5% of
  the time (at 95%), while the labels are at most a fifth of the windows where they differ.
- Exact per-class intervals, a design stratified by map class, and a re-checkable certificate of the trusted zone.

## What I did with your suggestions

| You suggested | What I did | What it showed |
|---|---|---|
| Copernicus EMS flood maps | 55 flood areas of interest in GEOID-Flood from 11 CEMS activations; 45 scored per task, from 9 activations for permanent water and 10 for water after the flood (exp55) | On the median area, the review set (5% of windows) holds 88% of the permanent-water errors and 64% of the errors after the flood (84% and 61% averaged by activation). The preregistered control won in one activation of nine (EMSR273) and one of ten; a pre-event water index (NDWI) also beat the radar model on 4 of the 45 areas after the flood |
| DFC2020 | Sentinel-1, Sentinel-2 and both (exp66) | Results stay internal until the IEEE GRSS IADF committee and TUM approve publishing them, as the contest terms require; if you want to take them further, that approval comes first. Its 10 m reference is a random forest, not hand labels |
| Dynamic World | 409 expert-annotated tiles (exp67) | Its own probabilities rank its errors better than class rarity, on 315 tiles against 91 |
| LUCAS | 4,778 field-surveyed polygons (exp68) | The ranking holds against what surveyors saw on the ground: it beats the best control computed from the imagery (pixel variance) on 94 regions against 32. A signal read from the map alone (neighbourhood disagreement) comes closer: it wins 39 of 126 |
| EuroCrops | Farmers' declarations in three regions (exp69) | It beats the best no-model control in all three, and in all but 4 of 147 grid cells |
| Run rslearn directly, not the runner | Every run since then uses rslearn directly | Filed [#72](https://github.com/allenai/olmoearth_projects/issues/72): the AWF config writes only the argmax class map. And [#73](https://github.com/allenai/olmoearth_projects/issues/73): olmoearth-runner 0.1.14, the version I chose, pins rslearn 0.0.27, whose Planetary Computer `get_item_by_name` raises NotImplementedError (fixed in 0.0.29; your lock's older runner predates it), and its dataset build sizes its worker pool by the whole node, not the job |
| HOTOSM open call | Not pursued | It funds sub-metre models, not tools |

Separately, on Sen1Floods11: on the Bolivia event, a no-model water index ranked my frozen S2 head's errors better
than its confidence did, and fine-tuning with your recipe removed that gap (exp52). With radar alone, the index still
wins on the multi-region test split, with your S1 probe and after fine-tuning (exp51, exp52).

## On your own models

- **FT-AWF v1, my re-run of your deployment config** (exp98), at the 259 of your 344 validation points inside the
  re-run map: its confidence ranks its errors at AUROC 0.85, and the least confident 10% of points hold 42% of them
  (14 of 33). The re-run is 87.3% accurate there, against 89.6% for a replica of your evaluation, a gap the data
  cannot tell from none (of the 20 points where exactly one is right, 13 favour the replica, p = 0.26). Your points
  agree best with the map about two pixels east, where 89.2% agree, a position found on the same points (exp100).
- **FT-AWF on 2017 imagery in Kenya and Tanzania.** Near the AWF area the confidence missed the preregistered bar
  (AUROC 0.64 against 0.70; exp99). A preregistered blind test on 890 fresh TimeSync plots (a random sample of
  Landsat-interpreted plots) gave 0.73 (90% interval
  0.70 to 0.76), graded on my v1.2 fine-tune because v1 did not reproduce its earlier map on the current stack; v1 gave
  the same 0.73, report-only (exp111). About half of the confident errors (13 of 25) are one contested legend pair:
  the map's shrubland_savanna where TimeSync says Open Grassland. The labels cannot say whose boundary is right.
- **Reproducibility.** On the same imagery, the v1 AWF checkpoint gives a different map under rslearn 0.1.17 than in
  my earlier olmoearth_run run (rslearn 0.0.27): 14 of the 259 points change class (26 of 263 at the 2017 plots),
  mostly among grassland, shrubland and woodland, and accuracy is unchanged (227 against 226). My earlier run itself
  departed from your lock (olmoearth-runner 0.1.14, rslearn 0.0.27 with one patch, torch 2.7.1). Ruled out: v1's
  stored temporal encoding table (current olmoearth_pretrain replaces it with its formula at load; restoring it
  changed nothing), the encodings' precision and autocast. The two runs also differ in olmoearth_pretrain (0.0.2
  against olmoearth-pretrain-minimal 0.1.2), torch (2.7.1 against 2.14.1) and the route (the runner against rslearn
  directly); I did not isolate the cause. My v1.2 fine-tune, whose encoder has no such table, repeats its earlier map
  on the new route at 259 of 259 points (exp111).
- **A v1.2 AWF fine-tune** with your recipe, only the encoder swapped:
  https://huggingface.co/2imi9/OlmoEarth-v1_2-FT-AWF-Base
- **OlmoEarth v1.3 embeddings** (exp112): logistic regression on the 2018 embeddings of 1,000 random LCMAP reference
  plots, checked at 11,000 others. The map is wrong on 11.7% of them, and its own confidence ranks those errors at
  AUROC 0.876.
- **A regional fit on PLANTEUR** (exp113): the same kind of classifier on 2019 embeddings, fitted on the other
  territories and applied to one held out, five times. Its confidence ranks its errors at AUROC 0.74 to 0.88 in every
  held-out territory. In three of the five the map is close to all Background, so there it mostly finds crop land the
  map missed. The best-agreeing embedding is v1.3 in four territories and TESSERA v2 in Guyane, mostly by small gaps.
- **Why it collapsed** (exp114): corrected with the true class shares, the Guyane and Réunion maps still miss them, so
  most of the failure is not a change of class shares and reweighting does not repair it. EM's label-free estimate of
  the shares (Saerens et al.'s prior correction) was no closer than the map's own on 12 of 15 held-out maps (BBSE,
  Lipton et al.'s, was closer on 8 of 15, but far off where there was no shift); 400 random labels rejected EM's
  estimate on every held-out v1.3 map.
- **Per class, with exact intervals** (exp115): on 39 labelled maps the guarantees held. 400 random labels flagged
  Guyane's Meadow and Réunion's Sugarcane, each drawn on 0.1% of the map against a true 14% to 16%, with probability
  1.000. Three of the five predictions failed, two of them by one class each. Most classes under 5% of a map need more
  than 400 labels to be flagged reliably.
- **Forest Loss Driver** at 2026 MapBiomas alerts (exp104): its confidence ranks its disagreements with the analysts
  at AUROC 0.77 (90% interval 0.68 to 0.86), on the 2 of its 10 classes the alerts cover. Post hoc, on 18 alerts:
  where it and MapBiomas disagree between agriculture and mining, it is as confident as when it agrees (AUROC 0.40);
  the run cannot tell the model's errors from mixed-driver alerts.
- **Another organisation's product**, the Fields of the World field map (exp110): its confidence ranks its errors at
  AUROC 0.84 to 0.94 in all four countries and met every preregistered bar in three (in Belgium a map-only control
  came within 0.018). With 100 labels where two field maps differ, the sign test named the one that agrees better with
  the parcel records with probability 0.86 to 0.999 in three of the four countries.

## For the model side

- **Run a head only on the inputs it was trained on.** On PASTIS with OlmoEarth Base, a probe trained on S1+S2 and
  read on radar alone, as under cloud, was wrong on 73.6% of windows, and 59.8% of its errors looked confident (margin
  AUROC 0.59). A probe trained on radar alone was wrong on 28.4%, with 3.9% confident (AUROC 0.79). Large behaves the
  same way. Caveats: "confident" is read at the full-input probe's threshold for both; the two probes were trained at
  different learning rates; radar alone here means a whole tile without optical input, not patchy cloud; and on
  CropHarvest China 6 the confident share fell when the optical input was removed. This comes from a follow-up that
  was not preregistered (exp88's addendum); whether training with modality dropout gives the same is not tested.
- **Say what `accuracy` means in rslearn.** SegmentationTask and ClassificationTask log torchmetrics'
  `MulticlassAccuracy`, which averages per class by default (macro): the mean of the class recalls, not the share of
  pixels right. On an imbalanced map the two differ. The task reference calls it "an accuracy metric"; three examples
  set `average: micro` and one says why, so a config that does not set it logs the macro mean (rslearn v0.1.19:
  `train/tasks/segmentation.py:274-279`, `classification.py:271-273`).
- **A small rslearn bug.** In ClassificationTask, RegressionTask and DetectionTask, the `continue` in the `filters`
  loop sits in the inner loop, so no feature is ever filtered out (rslearn v0.1.19: `train/tasks/classification.py:152-154`,
  `regression.py:135-137`, `detection.py:143-145`).

## Papers worth a look

- Lehmann, Gawlikowski, Ekim, Corley and Zhu, "Beyond Accuracy: Assessing Calibration of Geospatial Foundation Models
  and Their Sensitivity to Distribution Shifts", arXiv:2608.16614 (2026). Across 16 frozen encoders, the geospatial
  ones drift further into overconfidence under shift than ImageNet ones, and confidence-based abstention cannot defer
  around confidently wrong predictions. Related to the head result above, which is an input mismatch rather than a
  shift.
- Gonzalez-Calabuig et al., "SHRUG-FM: Reliability-Aware Foundation Models for Earth Observation", arXiv:2511.10370.
  It abstains using input-space and embedding-space out-of-distribution signals together with predictive
  uncertainty, and beats entropy alone on burn scars, floods and landslides. Its gate is fitted on labels, per image;
  ported to windows here (exp49), its signals did not beat the model's own confidence.
- Neverova, Wolf, Taylor and Nebout, "ModDrop: adaptive multi-modal gesture recognition", arXiv:1501.00102. Training
  with whole modalities dropped, so one model copes when an input is missing: the untested alternative to a head per
  input set.
- Garg, Balakrishnan, Lipton, Neyshabur and Sedghi, "Leveraging Unlabeled Data to Predict Out-of-Distribution
  Performance", ICLR 2022 (arXiv:2201.04234). It predicts accuracy on a shifted region without labels. Not tested on
  OlmoEarth maps; in the record, labels remain what decides.
- Sethi et al., "Fabrication After Tool Failure: Tool-Augmented Agents Assert Values Their Tools Did Not Return",
  arXiv:2609.14758 (2026). For an agent: when a tool returns "ok" with an unusable value, up to 45% of answers are
  dishonest (a value the tool did not return, or a made-up reason to decline); requiring a status line before the
  answer cut dishonesty from 14.1% to 0.87%.

## Where you could take it, when there is time

- **Your annotators.** Measure how often a reviewer is wrong. The tool can widen its intervals for it, but no human
  reviewer's rate has been measured (a vision-language model's was: it missed 74% of the map's errors, exp91).
- **Your platform.** Write per-class probabilities for more models (in rslearn, `output_probs: true` to a float32
  output layer; no project config I read in October set it) and run the tool on them.
- **Your release checks.** Confirm that a released checkpoint gives the same predictions after a dependency upgrade
  (rslearn, olmoearth_pretrain, torch).

## Links

- Every result, with its preregistration where there is one, and its audit:
  https://github.com/2imi9/olmoearth_inferenceX-record
- Technical report (PDF): https://github.com/2imi9/olmoearth_inferenceX-record/blob/main/report/main.pdf
- The blind transfer test: https://github.com/2imi9/olmoearth_inferenceX-record/blob/main/docs/plan/blind_transfer.md
