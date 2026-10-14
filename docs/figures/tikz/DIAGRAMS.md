# Diagrams with real rasters

Three TikZ diagrams of the audit, drawn on the repository's own data (scene okavango_80 from the exp11 archive with
its exp37 per-window table; Sen1Floods11 Bolivia tile 218 with its hand labels). Layered slanted planes with shadows
for rasters, geometric shapes for the model parts, few words.

| File | Shows |
|---|---|
| `architecture.tex` | the package's structure (README, docs index, report), laid out like the OlmoEarth architecture figure: real Sentinel-2 imagery of scene okavango_80 (1.28 km at 10 m/px) through an encoder (narrowing) and decoder (widening) to the two outputs the package reads, the class map and the confidence; then pooling to windows, the lane without reference labels (review set, disagreement map) and the lane with a labelled sample (the differing windows sampled for which map is better, by an exact test, and the size of the difference; a random sample for the error rate and the certified zone; each class's confident calls sampled for the classes not to trust); dashed groups, chevrons for operations, squares for windows, pink boxes for results. Panel margins are checked on the export |
| `audit_pipeline.tex` | Sentinel-2 bands in, frozen encoder and head, prediction / confidence / boundary layers, the review set at a 5% budget (boundary first) on the scene, the reasons card, the reviewer |
| `explanation_layer.tex` | one flood tile with the review set and the true errors, one flagged window traced through the four cue layers, the measured enrichment per cue as bars |
| `compare_inferences.tex` | the two-period, two-sensor square on one GEOID-Flood chip (event EMSR273-1, Gruemirë on Lake Shkodër, exp62 chip 7: the lake, permanent water, and its flooded shore): the four pieces of evidence with their dates (the Sentinel-2 composite and the Sentinel-1 pass before the event from GEOID, the Sentinel-1 pass after it, and WorldFloods v2's Sentinel-2 scene of 2018-03-28), the four decisions on identical windows, the four differences that each isolate one axis drawn on the scene (same period across sensors, same sensor across dates), and the label bridge boxed apart; imagery from `rasters/cmp_square7.npz` (cut once on the cluster), decisions from `exp/out/exp60_masks.npz` and `exp/out/exp62_masks.npz`, numbers from `make_compare_rasters.py` (exp60, exp62) |
| `protocol.tex` | a candidate signal next to the model's confidence and the no-model control, scored on both references on identical windows; the verdict; labels grade, never train |

Build:

```bash
uv run python docs/figures/tikz/make_rasters.py      # writes rasters/*.png and the review-set / error window lists
uv run python docs/figures/tikz/make_compare_rasters.py   # the comparison figure's rasters and number macros (exp57 / exp58 artifacts)
cd docs/figures/tikz && for f in architecture audit_pipeline explanation_layer protocol compare_inferences; do pdflatex -interaction=nonstopmode $f.tex; done
osascript -l JavaScript pdf2png.js audit_pipeline.pdf ../pipeline.png 3600   # and the other two; sips would blur (72 dpi base)
```

`style.tex` holds the shared macros: `\raster` (one slanted plane), `\stack` (three offset planes), `\cells` and
`\flatcells` (outline r/c windows of a grid on a plane), the `card`, `part` and `flow` styles. The window lists are
TeX macros written by the raster script (`rasters/scene_review.tex` defines `\sceneReview`, and so on).
