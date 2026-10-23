# Where this stands: a five-minute catch-up for the OlmoEarth team

Ziming Qi, 17 October 2026; updated 23 October 2026.

## What it is

A checker for Earth-observation maps. It reads the per-class scores a model writes out and tells you:

- where the map is likely wrong;
- how wrong it is, from a small set of labels;
- which part you can trust;
- which of two maps is better.

It never runs the model, so it works on OlmoEarth's outputs, other models' maps and published products.
`pip install olmoearth-inferencex` (1.13.0). Code and pictures: https://github.com/2imi9/olmoearth_inferenceX

## What I did with your suggestions

| You suggested | What I did | What it showed |
|---|---|---|
| Copernicus EMS flood maps | 55 flood areas through GEOID-Flood (exp55 to exp61) | The model's own margin puts most errors in the top 5% of windows. The one exception came from a frozen head and went away after fine-tuning with your recipe |
| DFC2020 | Sentinel-1, Sentinel-2 and both (exp66) | The margin beats a water-index baseline with every sensor |
| Dynamic World | 406 expert-annotated tiles (exp67) | Its own confidence ranks its errors better than a class-rarity baseline on 315 of them |
| LUCAS | 4,778 field-surveyed polygons (exp68) | The ranking holds against what surveyors saw on the ground: 94 regions to 32 |
| EuroCrops | Farmers' declarations in three countries (exp69) | It beats the best no-model baseline in all three, and in all but 4 of 147 grid cells |
| Run rslearn directly, not the runner | Every run since then uses rslearn directly | I filed the config and runner problems as [#72](https://github.com/allenai/olmoearth_projects/issues/72) and [#73](https://github.com/allenai/olmoearth_projects/issues/73) |
| HOTOSM open call | Not pursued | It funds sub-metre models, not tools |

## On your own models

- **FT-AWF v1 at your 259 validation points** (exp98): its confidence ranks its errors at AUROC 0.85, and the least
  confident 10% of points hold 42% of them.
- **Moving FT-AWF to 2017 in Kenya and Tanzania.** The first test, near the AWF area, fell short of the 0.70 bar
  (exp99). A preregistered blind test on 890 fresh plots reached 0.73 (exp111). Most confident errors there are a
  legend clash, shrubland against TimeSync's "Open Grassland", not a model failure.
- **Reproducibility.** On the same imagery, the v1 AWF checkpoint gives a slightly different map on rslearn 0.1.17
  than on 0.0.27: 14 of the 259 points change class, mostly among grassland, shrubland and woodland, and accuracy
  stays the same (227 against 226). The stored temporal table, the encodings' precision and autocast are ruled out;
  I did not pin down which of the three library changes (rslearn, olmoearth_pretrain, torch) causes it (exp111).
- **A v1.2 AWF fine-tune** with your recipe, only the encoder swapped:
  https://huggingface.co/2imi9/OlmoEarth-v1_2-FT-AWF-Base
- **OlmoEarth v1.3 embeddings** (exp112): logistic regression on the 2018 embeddings of 1,000 random LCMAP reference
  plots, checked at 11,000 others. The map is wrong on 11.7% of them, and its own confidence ranks those errors at
  AUROC 0.876. The fit points were spread over the whole country. The 2018 vectors run smaller than the dataset README
  says (median norm 3.08 against 3.4).
- **A regional fit on PLANTEUR** (exp113): the same kind of classifier on 2019 embeddings, fitted on the other
  territories and applied to one held out, five times. Its confidence ranks its errors at AUROC 0.74 to 0.88 in every
  held-out territory. In three of the five the map is close to all Background, so there it mostly finds crop land the
  map missed. The best-agreeing embedding is v1.3 in four territories and TESSERA v2 in Guyane, mostly by small gaps.
- **Why it collapsed** (exp114): corrected with the true class shares, the Guyane and Réunion maps still miss them, so
  most of the failure is not a change of class shares and reweighting does not repair it. A label-free estimate of the
  shares was no closer than the map's own on 12 of 15 held-out maps; 400 random labels rejected it on every held-out
  v1.3 map.
- **Per class, with exact intervals** (exp115): on 39 labelled maps the guarantees held. 400 random labels flagged
  Guyane's Meadow and Réunion's Sugarcane with probability 1.000. Three of the five predictions failed, two of them by
  one class each. Classes under 1% of a map need more labels.
- **Forest Loss Driver** at 2026 MapBiomas alerts (exp104): its confidence ranks its disagreements with the analysts
  at AUROC 0.77.
- **Another organisation's product**, the Fields of the World field map (exp110): the ranking held in 3 of 4
  countries, and 100 labels picked which of two field maps agrees better with the parcel records in 3 of 4.

## Where you could take it, when there is time

- **Your annotators.** Measure how often a reviewer is wrong. The tool can widen its intervals for it, but nobody has
  measured the rate.
- **Your platform.** Export per-class probabilities for more models and run the tool on them.
- **Your release checks.** Confirm that a released checkpoint still gives its launch predictions on a new rslearn.

## Links

- Every result, with its preregistration and audit: https://github.com/2imi9/olmoearth_inferenceX-record
- Technical report (PDF): https://github.com/2imi9/olmoearth_inferenceX-record/blob/main/report/main.pdf
- The blind transfer test: https://github.com/2imi9/olmoearth_inferenceX-record/blob/main/docs/plan/blind_transfer.md
