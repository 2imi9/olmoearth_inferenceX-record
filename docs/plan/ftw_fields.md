# Another organisation's deployed field map: does the tool rank, estimate, certify and compare on FTW's global field map? (exp110 preregistration)

**Status: frozen on 14 October 2026, before any metric was computed.** Written 14 October 2026. The owner confirmed
the predictions and their bars as proposed on 14 October 2026 (P1: AUROC at least 0.70, capture at 10% at least 0.25
and a lead of at least 0.05 over the best no-model control, in at least 3 of 4 countries; P3(b): a median certified
share of at least a quarter at α = 0.05 and 300 labels, in at least 2 of 4; P4(b): the better-agreeing map named with
probability at least 0.8 at n = 100, in at least 3 of 4; α = 0.05 and δ = 0.10; the checks P2, P3(a) and P4(a) as
stated). The window (8 px graded, 16 px reported), the population rules and the arms are frozen as proposed. The bars
were confirmed before the second map (PRUE B3) was run and before any label-free inventory of either map (the map's
confidence quantiles, the windows where the two maps differ); none of that inventory was seen before freezing.

**Result, 14 October 2026 (added at recording; the frozen text is unchanged).** Run once at cf59be4
(`exp/exp110_ftw_field_map.py`, `exp/out/exp110_summary.json`) and recorded after an independent audit that recomputed
368 numbers with its own numpy and scipy code and found no mismatch; its script is this page's second route,
`exp/exp110_crosscheck.py` (`exp/out/exp110_crosscheck.json`)
([comparisons](../results/comparisons.md#another-organisations-deployed-field-map-ftws-2nd-edition-map-at-the-ftw-benchmarks-test-chips-exp110)).
The run script was untracked when it ran; the sha256 the summary records matches the committed file. Graded as this
page defines the truth (the 2-class mask), the three predictions and the three checks hold:
- **P1 holds** (3 of 4): Croatia, Slovenia and Sweden meet all three bars (AUROC 0.870, 0.936, 0.863; lead over the
  best no-model control 0.127, 0.098, 0.307). Belgium meets the AUROC (0.837) and capture (0.308) bars but leads the
  predicted-class rarity by 0.018 (90% chip bootstrap 0.004 to 0.034), short of 0.05. No country is below the floor.
- **P3(b) holds** (3 of 4, fragile): median certified shares 0.40, 0.45, 0.90 and 0.00 (Sweden). Each country
  certifies any zone with an exact probability near one half (0.525, 0.517, 0.525 and 0.484), so each median sits on
  that edge; an independent set of 2,000 draws gave Croatia 0.55.
  *Note, 22 October 2026 (exp112):* at 300 labels and α = 0.05 any map, a perfect one included, certifies a zone
  with probability at most about 0.525 (the 15% level enters there, expecting exactly the 45 labels it needs), so this
  bar turned on whether each country's most confident 15% was almost free of errors; the four probabilities sit at or
  below that limit (Belgium and Slovenia at it). See
  [exp112](../results/comparisons.md#a-map-made-from-olmoearth-v13-embeddings-at-lcmaps-probability-sample-exp112).
- **P4(b) holds** (3 of 4): 100 labels name the better-agreeing map with probability 0.858 (Belgium, B3), 0.999
  (Croatia, the 2e map) and 0.993 (Sweden, B3); Slovenia (0.141) and pooled (0.063) do not reach 0.8.
- **The checks hold:** P2's exact coverage is at least 0.957 in all 10 cells, the Monte Carlo within 1.69 standard
  errors; P3(a)'s largest share of draws certifying a zone wrong more often than α is 0.040 (bar 0.115); P4(a)'s
  largest probability of naming the worse map is 0.0036 (bar 0.025).

**Amendment, 14 October 2026 (after the run).** Two findings made after freezing, ruled on by the owner on 14 October
2026: the first when the labels were read with the maps, the second in the B3 run's report, before any label was read.
Neither changes a bar or a verdict.
- **The truth is field only.** The label-only check this page lists found that the 2-class mask is not field plus
  boundary: in all four countries it equals the 3-class mask's field class pixel for pixel, and every boundary pixel
  is coded background (449,464, 968,577, 334,035 and 951,825 pixels). The page assumed otherwise (its base rates and
  "field extent" are field plus boundary), so its main measure compares the map's field + boundary with a field-only
  truth. The owner ruled that the graded result follows this page literally: truth = the 2-class mask. The field +
  boundary reading (truth = the majority of the 3-class mask in {1, 2}) is reported beside it, dated, as a difference
  found after the labels were read with the maps; it never replaces a verdict. Under it the pooled window error rate
  is 0.1025 against 0.1110, lower in Belgium, Croatia and Slovenia and higher in Sweden (0.1450 against 0.1415), so the
  boundary pixels coded background account for only part of the window errors; the six verdicts are the same.
- **B3's training countries.** The Gate section and P4's reading say PRUE B3 never trained on Belgium, Croatia,
  Slovenia or Sweden. That matches FTW's older v1 CC-BY configuration, not the checkpoint used: FTW's v3.1 release
  notes say the CC-BY PRUE models used the `prue_efnet*.yaml` configurations without the non-commercial countries, and
  `prue_efnet3.yaml` trains on all four, none of them non-commercial. The checkpoint stores no country list, so B3
  most likely trained on these countries' train chips, as the 2e map's model did. The test chips are the held-out
  test split for both maps. The page's expectation of a gap in the 2e map's favour is withdrawn; B3 is the
  better-agreeing map in Belgium, Sweden and pooled under both truth readings. Arm 4 says which map agrees better on
  these chips, not which generalises better.

Changed at freezing, with the bars' numbers unchanged: the steps the draft allowed before freezing (the label-free
inventory, the B3 run and its band-swap control, the label-only and metadata checks) were not taken before it. They
come after freezing and before any arm's metric; the label-free stop rules for B3 under What would invalidate the run
now apply before any label is read with B3's map rather than before freezing.

- **Not blind for:** the data were downloaded and aligned on 14 October 2026, and no metric was computed (the
  acquisition manifest, kept outside the repository). That manifest and the gate (summarised under The gate below)
  computed only shapes, counts, value ranges, nodata and zero shares, alignment checks (grid against grid, imagery
  against imagery) and sha256. The gate compared ftw-q with the map on three Rwanda train chips. It never opened a
  label, and Rwanda is not an evaluation country. The following is already known:
  - **Label base rates, from the manifest's 3-class pixel counts.** Field plus boundary is 27.7% of the mask pixels in
    Belgium, 32.3% in Croatia, 11.6% in Slovenia and 39.2% in Sweden. Boundary alone is 3.5%, 4.2%, 2.2% and 2.8%. No
    unknown class (3) occurs.
  - **About the map, label-free.** Its values run 0 to 1, and field plus boundary never exceeds 1. It covers every chip
    (`map_valid` is all True). Thirteen chips lie across a tile seam. The 2e README calls its images "selected
    examples, not an accuracy assessment", and its vector score is "not calibrated against ground truth".
  - **Published figures that touch these chips.** The ftw-q model card reports object F1 0.526 on the FTW-Q test split
    (all 7,817 test chips, these 1,295 among them). The FTW papers publish per-country results for other models. None of
    these is a window error rate of the 2e map or a ranking by its confidence, and no bar below was set from them.
- **Not done before freezing** (allowed by the draft; they come after freezing and before any arm's metric):
  - Label-free inventory: the map's window confidence quantiles and class shares per country, the no-model controls'
    values, the PRUE B3 run, and D (the windows where the two maps differ) per country.
  - The B3 band-swap control against the 2e map. This compares map with map, not with labels.
  - Label-only checks: whether the 2-class mask equals the union of the 3-class mask's field and boundary classes; the
    count of windows whose truth has no majority; the truth class shares per window.
  - Metadata checks: each of the 1,295 chips is `test` in the FTW-Q index as well.
  - The smoke run on synthetic arrays.
- **Not computed before freezing:** anything that uses a label together with the 2e map, the B3 map or a control. That
  covers errors, agreement with labels, AUROC, capture, coverage, certified zones, K_A and K_B, and any check of the B3
  run against published per-country numbers.

## The question

Every result in the record so far is on OlmoEarth models and probes, or on land-cover products, graded against
land-cover references. FTW's 2nd-edition global field map (FTW Global Data 2e, CC-BY-4.0) differs on three counts:
another organisation made it, it is deployed, and its task is not land cover (field extent and field boundaries,
2017 to 2025). It ships per-pixel probabilities (uint8 / 255) for field and boundary. FTW benchmark v1 holds dense
reference masks for its test chips.

The question: read as the package reads any probability map, does the product's own probability:

1. rank its window errors well enough for review;
2. give honest error-rate intervals from a random sample of windows;
3. certify a confident zone that keeps its guarantee; and
4. let a few labels say which of two field maps is more accurate?

## The gate (phase 0, 14 October 2026), stated as a condition

**Condition G: the 2e map was made by ftw-q.** ftw-q is the CC-BY quarterly model on Hugging Face (sha256
2a24bb92…eaf4). It was trained on the FTW-Q train split, which holds train chips of all four evaluation countries:
Belgium 1,554, Croatia 2,778, Slovenia 1,733 and Sweden 3,802. Their test chips are a separate split.

**What the gate showed.** The gate report is kept outside the repository; this is its summary. Its verdict: the 2e
map's model (`unet_balanced_fp32.onnx`, sha256 e12bcdb9…282e, the hash every 2e tile carries) is the public ftw-q. On
the region of Rwanda that can be rebuilt exactly (three train chips on one 2021 tile, 196,608 pixels per band), ftw-q
reproduces the map's field and boundary bands to within one quantisation step
(max |diff| 1/255; 99.994% of pixels identical). It does so with the public pipeline's settings, with no spatial
offset, and a band-swap control fails, as it should. The two ONNX files differ in bytes but compute the same function.

**What it did not show:**
- **Other tiles.** Identity on other tiles is inferred from the model hash every tile carries, not tested.
- **The production code.** The code that produced the map (prov_code 5d01d4b1b222) is not public.
- **Nodata fill.** The map's nodata fill (`window31`) was not replicated.
- **The test chips themselves.** None of the 7 Rwanda test chips can be rebuilt exactly.
- **Band order.** The tiles' `source_bands` tag contradicts the README. The band-swap control shows the input order was
  B04, B03, B02, B08.

**If G holds** (the working assumption):
- The test chips are held out for the map in the sense of "not these chips". They are not held out from these
  countries, years or landscapes.
- Whether FTW's splits are spatially blocked was not checked, so a test chip can neighbour a train chip.
- The map is therefore in distribution by country. PRUE B3 is not: its CC-BY config never trained on Belgium,
  Croatia, Slovenia or Sweden.

**If G fails** (a tile with another hash, or evidence that production used other weights), the test chips'
held-out status for the map is unknown. For each arm:
- **Arm 1 (ranking).** The result still describes the product on these chips. It cannot be read as the product on
  unseen land, because a model that trained on a chip makes fewer errors there, and the errors left are mostly
  label noise and definition, which it can hold confidently.
- **Arms 2 and 3 (estimation, certify).** These stay valid, since they are design-based and hold for any map. The rate
  and the zone describe agreement on chips the model may have trained on, so they can understate the error on new land.
- **Arm 4 (two maps).** The comparison is confounded: a verdict for the 2e map could be exposure to training data. It
  is reported, and it is not read as which map generalises better.

## Data

| Country | Labels (source) | Label year, map edition | Test chips | Windows (8 px) | 2e tiles | Labels' licence (checked 14 Oct 2026) |
|---|---|---|---|---|---|---|
| Belgium (Flanders) | Landbouwgebruikspercelen LV 2021, Dept. LV, via EuroCrops | 2021, 2021 | 198 | 202,752 | 1 | Flemish model licence for free reuse; attribution "Bron: Dept. LV" |
| Croatia | ARKOD parcels (APPRRR), photo-interpreted | 2023, 2023 | 353 | 361,472 | 3 | "Open Data", linking to a .docx that was not opened; the weakest documented |
| Slovenia | AKTRP declarations, via EuroCrops | 2021, 2021 | 228 | 233,472 | 3 | CC-BY-4.0 |
| Sweden | Jordbruksverket LPIS blocks (one block can hold several fields) | 2021, 2021 | 516 | 528,384 | 6 | FTW: "No restrictions on public access"; provider record "CCBY", and elsewhere "conditions unknown" |
| **Total** | | | **1,295** | **1,326,080** | 13 | none is non-commercial |

**The data on disk** (a local data directory kept outside the repository; its files and their sha256 are listed in
the acquisition manifest, also kept outside the repository; no tile or chip identifier is named on this page):
- Test chips are those with `split == "test"` in each `chips_<country>.parquet`. Each has a 2-class mask, a 3-class
  mask, window_a and window_b. All four share one grid: EPSG:4326, 256 x 256, stored south-up.
- The chips are not 1,536 m squares. They measure about 1,530 m N-S, and E-W 960 to 975 m (Belgium), 1,053 to
  1,088 m (Croatia and Slovenia) and 773 to 875 m (Sweden). A mask pixel is about 6.0 m N-S by 3.0 to 4.3 m E-W.
- The map is the label year's 2e edition, read by HTTP range requests and area-averaged (GDAL `average`, exact
  transformer) from its 2.5 m UTM grid onto each chip's own mask grid, `v1/<country>/map2e_<year>_on_mask_grid_<country>.npz` in that directory.
  The masks are in the same npz, row 0 = south for all arrays.
- The 13 seam chips are filled from two tiles and flagged `map_contained_single_tile = False`: Croatia 4, Sweden 9.
- The masks declare nodata = 0, which is also the background class. The declared nodata is never applied.
- Second map (arm 4): the PRUE U-Net EfficientNet-B3 CC-BY checkpoint (release v3.1, 53,242,827 B, sha256
  cd51f020…c5a6d4, matching the release digest; CC-BY-4.0). It has not been loaded or run.
- The map and both models are CC-BY-4.0. Latvia, Lithuania, Portugal and South Africa (non-commercial labels) are not
  used.

## Units

**Window.** 8 x 8 mask pixels on each chip's own mask grid, so 32 x 32 = 1,024 windows per chip. 8 divides 256, so a
window never crosses a chip. A window is about 48 m N-S by 24 to 35 m E-W, 0.12 to 0.17 ha. That is close to the
area of the record's usual window, 4 px at 10 m (0.16 ha), and spans about 5 x 2.4 to 3.5 of the map's 10 m input
pixels. 16 x 16 windows (331,520 in all) are reported, not graded.

**Truth (main measure: field extent).** A window's class is the majority of its 64 pixels in the 2-class mask (field
against background). An evenly split window (32 against 32) has no majority. As in `assess --reference`, it is left out
of every population, and the count is reported.

**The map's reading (main).** The map's extent probability is p = field + boundary = 1 − background, per mask pixel.
It is read as a two-class probability map, `(H, W)`, as the package reads one:
- pixel class p > 0.5;
- pixel confidence 2|p − 0.5| (the margin between the two class probabilities);
- the window's confidence is the mean over its 64 pixels;
- the window's class is the majority of its pixels, and a tie goes to the class whose pixels are more confident
  (`assess._pooled_argmax`).

So the tool reads the map's per-pixel field probability through its distance from the cut. The product ships no
logits, so the package's warning about ties where probabilities saturate applies: a window wholly at 0/255 or 255/255
ties at confidence 1. AUROC counts ties half, and capture is taken in expectation over ties.

**Error.** A window is wrong when the map's window class differs from the truth's.

**Second measure (3-class, reported).**
- **Map:** the probabilities (1 − f − b, f, b) as a `(3, H, W)` map, with confidence the window mean of the top
  probability and class the majority of the pixel argmax.
- **Truth:** the majority of the 3-class mask, a tied top count left out.
- **Why it comes second:** boundary pixels are 2.2% to 4.2% of the mask pixels, so a boundary-majority window will be
  rare. At this window size the 3-class reading is mostly the extent reading. It is also reported at single mask pixels
  (`--patch 1`).

**Populations.** Each country's windows with a truth majority, plus the four countries pooled as one population. The
main populations keep the seam chips, because the product has two predictions there and the fill rule is a user's
choice. Results without them are reported.

**How the package is run.** For each country the script writes one mosaic array of its chips, side by side in sorted
aoi_id order, each chip at a multiple of 256 px. Empty slots are NaN in the map and a code outside the classes in the
truth. The script runs the package at 1.12.1 on these mosaics, through the command line where a route has one and its
Python functions for the repeated draws. The package's boundary cue at chip seams in a mosaic is meaningless and is not
graded.

## Arms

**Arm 1, review ranking (`assess --patch 8`).** Computed per country and pooled.
- **Measures:**
  - AUROC of the map's suspicion (minus the window confidence) for the window errors, ties counted half
    (`metrics.weighted_auroc`, every window weighted 1);
  - capture at 10%: the share of errors among the 10% least confident windows, in expectation over ties
    (`metrics.capture_at_budget_expected`);
  - excess AURC, reported.
- **Random:** AUROC 0.5 and capture 0.10.
- **No-model controls (label-free; the record's two):**
  - *Pixel variance:* the mean over the eight bands (B04, B03, B02, B08 of window_a and window_b) of the band's
    variance over the window's pixels. Each band is scaled by its standard deviation over the country's chips, and
    pixels that are zero in every band are left out. Higher means more suspect.
  - *Predicted-class rarity:* the share of the country's windows the map puts in the window's class. Rarer means more
    suspect.
  - The **best no-model control** is the one with the higher AUROC in each country. That choice uses the labels, which
    is the strict comparison for the confidence.
- **Map-only cues (reported, not controls):** the neighbourhood disagreement (the share of the window's neighbours
  within its chip whose map class differs from its own) and the boundary-first order (`--order boundary_first`).

**Arm 2, error-rate estimation (`sample --design random`, `estimate`).** θ is each population's full-truth window
error rate.
- **Design:** a simple random sample of B windows, B = 100 and 300, read with the package's exact hypergeometric
  interval (`estimate.hypergeom_interval`).
- **Coverage, exactly:** the hypergeometric probability that the interval contains θ (`estimate.exact_coverage_srs`
  with that interval; no Monte Carlo).
- **Second route:** 2,000 draws per cell (seed 110) through `sample_for_estimation(design="random")` and
  `estimate_error_rate`, the functions behind the command line, against the exact coverage.
- **Reported:** the interval's median width, and `plan`'s predicted width beside the realised one.
- **Reported, not graded: the chip-cluster design** a reviewer labelling by chip would use. This is `--design tiles`
  with one chip per tile (`--tile 32`) and 16 windows per chip. Its coverage is computed by 2,000 draws, because exp78
  found this interval under-covers where a few tiles hold most errors.

**Arm 3, certify (`certify`).** The B = 300 random samples of arm 2, 2,000 draws per population (seed 110).
- **Settings:** α = 0.05 (graded) and 0.10 (reported), δ = 0.10, the prefix rule and the standard level cut, all
  package defaults except α.
- **Violation:** a draw whose certified zone has a full-truth window error rate above α.
- **Reported:**
  - the probability of certifying any zone;
  - the certified share's distribution;
  - the zone's true error rate;
  - `min_labels_to_certify` (45 at α = 0.05).

**Arm 4, two maps (`sample --other`, `estimate`; phase 2).**
- **Map b, how it is run:**
  - The PRUE B3 model runs on the same chips from the benchmark's own imagery: window_b then window_a, stacked into 8
    bands in the stored order B04, B03, B02, B08, divided by 3000. This is the stacking and normalisation of the
    baselines' dataset code; the training augmentation swaps the two windows at random.
  - It outputs a 3-class softmax on the chip grid, and its extent probability is field + boundary, read as the 2e map's
    is.
  - The class order is read from ftw-baselines' code before the run.
  - The checkpoint is opened only with `torch.load(..., weights_only=True)`, and its state dict is loaded into
    `segmentation_models_pytorch.Unet("efficientnet-b3", in_channels=8, classes=3)`. If `weights_only=True` refuses the
    checkpoint, arm 4 stops and the owner decides. It is never unpickled otherwise, and no downloaded code is run.
- **Population:**
  - The windows with a truth majority where both maps predict.
  - Windows holding any pixel that is zero in every band of either S2 window are left out, a label-free rule (B3 reads
    zeros there). This affects 13 chips, up to 13.9% of a Sweden chip.
- **The full-truth answer:**
  - D is the number of windows where the two maps' classes differ.
  - K_A is the number of those where only the 2e map is right, and K_B where only B3 is. With two classes, K_A + K_B = D.
  - The better-agreeing map is the one with the larger K.
- **The tool's verdict:** n labels drawn among the D differing windows, with the verdict `estimate` prints (the exact
  sign test while n ≤ D/5, the interval past it).
- **Probabilities:** the probability that it names the better map, the worse map or neither, at n = 100 and 300. These
  are exact from (N, D, K_A, K_B) by `plan.which_map_probability`.
- **Second route:** 2,000 draws per cell through `sample_disagreement` and `compare_from_disagreement`.

## Predictions (confirmed by the owner on 14 October 2026, as proposed)

**P1, the ranking (arm 1).** In at least 3 of the 4 countries:
- (a) the map's confidence ranks its window errors at AUROC ≥ 0.70, with capture at 10% ≥ 0.25, which is 2.5 times
  random; and
- (b) its AUROC is at least 0.05 above the best no-model control's.

Floor: a country with fewer than 200 window errors is not determined for P1. If two or more countries fall below the
floor, P1 cannot hold.

**P2, the interval (arm 2; a check, not a prediction).** The exact coverage is at least 0.95 at B = 100 and 300 in
each of the four countries and pooled (10 cells), and the Monte Carlo route agrees with it within 3 Monte Carlo
standard errors.
- The interval covers at least 95% by construction for a random sample whose labels are taken as right. So P2 checks
  that the package's pooling and reading of a third party's map deliver that coverage.
- A failure is a defect. It stops the run before anything is recorded.

**P3, the certified zone (arm 3).**
- **(a), a check:** at α = 0.05, the share of draws that certify a zone wrong more often than α is at most 0.115
  in each of the 5 populations. That is δ plus about 2.2 Monte Carlo standard errors at 2,000 draws (one SE is
  0.0067).
- **(b), what certify buys:** at α = 0.05 and 300 labels, the median certified share over the draws is at least 0.25
  of the windows in at least 2 of the 4 countries.

**P4, which map (arm 4).**
- **(a), a check:** at n = 100 and 300, the tool names the worse-agreeing map with probability at most 0.025 in every
  country and pooled.
- **(b):** at n = 100, it names the better-agreeing map with probability at least 0.8 in at least 3 of the 4 countries.

As in exp107, (b) is arithmetic on (N, D, K_A, K_B) once those counts are known. What it measures is mostly the size of
the gap between the maps. B3 never saw these countries and the 2e model did, so a gap in the 2e map's favour is
expected. (b) is kept because it is the route a user would run.

**Reported, not graded:**
- the full-truth θ per country, the 3-class reading, the 16-px windows, and the populations without the 13 seam chips;
- the share of errors that are confident: confidence at or above the median confidence of the correct windows, as
  exp106 defined it;
- 90% chip-cluster bootstrap intervals (2,000 resamples by chip, seed 110) for the AUROC and for the lead over the best
  control;
- the oracle capture at 10%;
- a lenient extent reading, in which the truth's field pixels are dilated by 7 m, about 1 mask pixel N-S and 2 E-W.
  This bounds how much of the error is FTW-Q's buffered boundary definition (under What it cannot show);
- practice's McNemar exact test on a simple random sample of n of all windows (exp107's arm A) beside arm 4, and the
  labels each design needs to name the better map with probability 0.8;
- `plan`'s advice for each arm's budget beside what the budget delivered.

## What would change the conclusion

- **P1 holds.** On another organisation's deployed product, a field-extent task and its benchmark's reference, the
  product's shipped probability ranks its window errors as the record's maps do, ahead of what the imagery alone gives.
  `assess` can be pointed at the FTW map as shipped.
- **P1(a) fails.** The product's errors are not where it is unsure. Read with the share of confident errors and by
  country: Sweden's blocks and undeclared fields can make confident disagreements that are reference, not model.
- **P1(b) fails, (a) holds.** For field extent, the errors sit in mixed windows that any heterogeneity measure finds.
  The confidence adds little over the imagery's variance here.
- **P2 or P3(a) fails.** A defect in the pipeline: stop, find it, and record the defect, not the numbers.
- **P3(b) holds.** 300 random labels certify a quarter or more of this product's windows at 5% in most countries.
  **P3(b) fails.** Either the product's confident part is wrong more often than 5% (read with the zone's true error
  rate), or 300 labels are too few. `plan` says which.
- **P4(b) fails.** The two maps are too close on these chips for 100 labels. The record gives (D, K_A, K_B) and the
  labels needed.
- **G fails.** The consequences are as stated under The gate. Arms 2 and 3 stand as checks, and arms 1 and 4 are read
  only for these chips.

## What it cannot show

- **The labels are taken as right.**
  - The masks are administrative declarations and LPIS blocks. A field that was not declared is background, so a map
    that finds it is counted wrong.
  - Sweden's blocks hold several fields and can include non-crop features. Croatia's parcels are photo-interpreted.
    Belgium is Flanders only.
  - No reviewer labels anything. Label cost and reviewer error are not measured.
- **The map's definitions differ from the masks'.**
  - The map was trained on FTW-Q labels at 2.5 m, where the boundary class is the polygon outlines buffered by 7 m
    (FTW-Q dataset card). If the buffer runs both ways from the line, the extent the map learned reaches about 7 m
    beyond each v1 polygon, and part of the edge errors is definition. The lenient reading bounds that part; it does
    not remove it.
  - B3 was trained on v1's own definitions, which favours it at edges in arm 4.
- **The package reads area-averaged probabilities on the mask grid, not the product's 2.5 m pixels.** A window's
  confidence is a mean over averaged pixels. A user running `assess` on the native tile would pool the product's own
  pixels, and that reading is not graded.
- **θ is the error rate over the test chips' windows, not the country's map.** The test chips are not a probability
  sample of a country, and the pooled rate weights the countries by window count, Sweden most.
- **Four European countries, one product, one year each.** Each country's train chips trained the map's model (under
  G). This is one non-land-cover task, not a general test.
- **Arm 4 compares two maps, not two models.** They differ in model, input imagery (the benchmark's two S2 windows
  against quarterly mosaics), training data (B3 never saw these countries) and label definition.
- **Croatia's imagery dates** match seasons to grids by list position, an assumption. This matters only for B3.
- **The seam chips** carry two tiles' predictions.
- **One window size is graded.** The 16-px result is reported.
- **Windows are not independent.** Windows within a chip, and chips within a tile, share errors. The intervals are
  design-based over windows, which the random draw justifies. The AUROC's bootstrap is by chip.

## What would invalidate the run

- **θ, K_A or K_B computed from a sample** rather than from the full truth.
- **The map read as field alone** rather than field + boundary, or B3's class order or preprocessing wrong.
  - A label-free control is run before any label is read with B3's map: B3 with its input bands reversed (B02, B03,
    B04, B08) must agree with the 2e map on fewer windows than with the stated order.
  - If B3 in the stated order agrees with the 2e map on fewer than 70% of a country's windows, arm 4 stops for the
    owner before any label is read with B3's map.
- **Row order.** The map, masks and B3 output must share one row order (row 0 = south). B3's output is written in its
  S2 files' order, which is the masks'.
- **A window crossing a chip**, or the masks' declared nodata applied (it would delete the background class).
- **Population rules changed after the labels are read.** This covers the seam chips, the zero-imagery windows and the
  tied windows. Every rule above is fixed label-free.

## Independent check before anything is recorded

An audit that did not write the script recomputes from the npz, with plain numpy and without the package's pooling:
- each country's θ, error count, AUROC and capture;
- K_A and K_B;
- one exact coverage cell.

It also checks the summary's input hashes against the acquisition manifest's file list (sha256 per file), kept
outside the repository. Each claim registered in
`docs/claims.yaml` names this second route.

## Outputs

- **Summary:** `exp/out/exp110_summary.json`. It holds aggregates only, per country and pooled, plus the inputs'
  sha256.
- **Script:** `exp/exp110_ftw_field_map.py`, numpy plus olmoearth-inferencex 1.12.1 (tag v1.12.1). Its crosscheck is
  `exp/exp110_crosscheck.py`.
- **What stays local.** No per-window file is committed, a departure from the record's habit of committing units
  files. The licences ask for attribution, Croatia's terms are weakly documented, and per-window truth is derived label
  data. The audit and any rerun read the local data whose hashes the summary lists. B3's per-chip outputs stay in
  the local data directory, outside git, with their sha256 in the summary.
- **Attribution in the record:**
  - FTW for the map and both models (CC-BY-4.0);
  - "Bron: Dept. LV" for Belgium;
  - the providers for Slovenia (AKTRP, CC-BY-4.0), Sweden (Jordbruksverket) and Croatia (APPRRR).

## Cost

- **Arms 1 to 3:** local, on the Mac. The I/O is done: 1.85 GB on disk, the four npz files 263 MB. The largest country
  (Sweden, 528,384 windows) fits in memory. The ranking and the exact coverage take seconds. Certify's 2,000 draws per
  population are minutes, not measured.
- **Arm 4:** B3 on 1,295 chips of 256 x 256 x 8 is a few minutes of CPU, not measured.
  - It needs torch and segmentation-models-pytorch in a local venv outside the repository. That is a PyPI download of
    a few hundred MB, to be approved; free disk is about 15 GB.
  - The alternative is one cluster CPU job (srun or sbatch, `-A p2026_0089_neu`, nothing on the login node), which first
    copies the 2,590 S2 files (1.36 GB) and the checkpoint. The owner chooses.
- **Labels:** none are bought; the truth is the benchmark's masks.
- **Licences:** aggregates only are reported. Croatia is in the frozen populations, under its "Open Data" terms as
  linked (the .docx was not opened); the four-country bars were confirmed with it in.

## Decisions taken at freezing

1. **The bars:** confirmed by the owner on 14 October 2026 as proposed: P1 (0.70, 0.25, +0.05; 3 of 4), P3(b) (a
   quarter at 5%; 2 of 4) and P4(b) (0.8 at n = 100; 3 of 4). Also α = 0.05 and δ = 0.10.
2. **The window:** 8 px graded and 16 px reported, as proposed.
3. **The populations:** seam chips in (main), results without them reported; all four countries, Croatia in.
4. **B3:** the bars were confirmed without seeing the label-free inventory (D, the band-swap control). Where B3 runs
   (on the Mac after the venv download, or as one cluster CPU job) changes no rule on this page; the owner chooses
   before arm 4 runs.
