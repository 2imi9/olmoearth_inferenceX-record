# Review of the changes since 1.2.0 (69bebdb): compare, explain, assess, reliability, calibrate, signals, evidence, taskcard, CLI assess/compare

I changed no repository files. Every reproduction below was run with `uv run --no-sync` on HEAD 4a80efe. The scripts are in the scratchpad (`p_*.py`, `h.py`). `S` stands for the scratchpad directory.

Findings are ranked from most to least severe.

---

## 1. DEFECT: the new grid check lets shifted rasters through, because its tolerance is relative (`cli._same_grid`)

`_same_grid` compares the transforms with `np.allclose(...)` at its default `rtol=1e-5`. The tolerance therefore grows with the size of the origin coordinate:
- UTM northing near 5e6: 50 m allowed
- Web Mercator easting near 1.2e7: 120 m allowed

A map shifted by one whole 4-px window (40 m) is compared as though co-registered, with no warning. `--labels` and `--groups` go through the same check.

```
# p_grid.py: one uint8 array written three times, EPSG:32633 at 10 m (b's origin 40 m further south), and EPSG:3857 (b 100 m east)
oe-inferencex compare ga.tif gb.tif   --out ...  ->  0 of 256 windows differ (0.00%)          # 4 px N-S shift, accepted
oe-inferencex compare ga.tif gb_x.tif --out ...  ->  gb_x.tif is not on the first map's grid   # 4 px E-W shift at x=5e5, caught
oe-inferencex compare wa.tif wb.tif   --out ...  ->  0 of 256 windows differ (0.00%)          # 10 px shift in Web Mercator, accepted
np.allclose((10,0,500000,0,-10,5000000),(10,0,500000,0,-10,4999960))  ->  True
```

The docstring says this check exists to catch shifted maps. It catches a shift in x at UTM eastings but not the same shift in y. The fix is an absolute tolerance of a small fraction of the pixel size, e.g. `atol=1e-3*abs(pixel)` with `rtol=0`.

## 2. DEFECT: `assess --reference` crashes when the reference grades no window

When no window has both a prediction and a majority reference label, `_assess` takes an early return. That return has no `error_capture_at_budget`, but `cmd_assess` prints that key unconditionally. This happens with sparse point labels (fewer than 50% of a window labelled), with an all-no-data reference, and with a reference that is evenly split in every window. The JSON has already been written when the command dies with exit 1 and a traceback.

```
# one labelled pixel per 4x4 window
oe-inferencex assess p16.npy --out oa3 --reference ref_sparse.npy
  File ".../cli.py", line 164, in cmd_assess
    ... s["against_reference"]["error_capture_at_budget"].items() ...
KeyError: 'error_capture_at_budget'
exit=1   (assessment.json, explanation.json, review_set_*.csv already on disk)
```

`p_assess.py` gives the same KeyError for an all-tied reference and for an all-`-1` reference.

## 3. DEFECT: a tied window is broken differently for a class map and a score map, so identical decisions "differ" (CLI compare)

`decisions()` returns no confidence for a hard class map, so `_pooled_argmax` sends a tie to the lowest class index. This is the class-0 bias that audit finding 10 removed elsewhere. A probability or score map sends the same tie to the more confident voters. Comparing one map exported both ways therefore reports differences that do not exist. Usage.md says without qualification that "a tied window goes to the more confident voters".

```
# one window: 8 px class 1, 8 px class 0; the probability version has the same decisions (p>0.5)
pixel decisions identical: True
oe-inferencex compare tie_cls.npy tie_prob.npy  ->  1 of 1 windows differ (100.00%);  CSV: a=0, b=1

# a smooth 512x512 map, once as uint8 classes and once as its probabilities
oe-inferencex compare field_cls.npy field_prob.npy   ->  292 of 16384 windows differ (1.78%)
oe-inferencex compare field_cls.npy field_cls.npy    ->  0 of 16384 windows differ (0.00%)
```

1.78% is of the same order as the 2–4% reseed floor the record uses for drift. The same mechanism also lets confidence-only drift count as a decision difference. With 1e-3 noise that changes no pixel's decision (`field_prob.npy` against `field_prob_b.npy`), 2 of 16384 windows differ. Two runs with identical decisions can therefore fail the "must give 0" expectation.

## 4. DEFECT (CLI/API inconsistency): the API grades unlabelled (negative) labels as a class

The CLI drops label codes below 0 (`lv &= lab_i >= 0`), and `assess` and `explain.confusion_pairs` treat a negative reference as "no label". `compare_inferences` in Python compares the maps against `-1` as if it were a class, so every unlabelled window counts as an error for both maps. The same arrays give different graded numbers:

```
# p_api_lab2.py: a, b 2x4; labels [[0,1,-1,-1],[-1,-1,-1,-1]]
CLI crosstab : {'n': 2, 'errors_a': 1, 'errors_b': 0, 'both': 0}  which_side: {'n_disagree': 1, 'a_right': 0, 'b_right': 1, 'neither': 0}
API crosstab : {'n': 8, 'errors_a': 7, 'errors_b': 6, 'both': 6}  which_side: {'n_disagree': 2, 'a_right': 0, 'b_right': 1, 'neither': 1}
```

The API needs either the CLI's `labels >= 0` restriction or a refusal of negative labels.

## 5. DEFECT: `fit_side` compares its held-out share with baselines computed on different rows

Since this change `held_out.share_right` covers only the rows whose fold could be fitted. `baseline.share_right`, `always_a`, `always_b` and `neither_right` still cover every disagreement row. When a fold is unscored, the report sets numbers from two populations side by side:

```
# p_side.py: two tiles, tile 0 where b is always right, tile 1 a coin; folds=2
held_out: {'share_right': 1.0} | scored rows: 200 of 400
baseline: {'reading': 'm', 'share_right': 0.475} always_a: 0.225 always_b: 0.775
baseline on the SAME scored rows (tile 0): 0.47 | always_b on them: 1.0
```

Read as written, the report says the fitted rule beats "always b" by 22 points. On the rows it was scored on, the two tie. The per-group block already restricts both sides to `scored`; the pooled block does not.

## 6. DEFECT (API): with `form="top1"`, the quantiles and the confidence array are on different scales

`assess_prediction(..., is_logit=True, form="top1")` now converts `confidence_quantiles` to probabilities and adds `confidence_scale`. `arrays["confidence"]` stays as the window mean of log p, and it feeds the explanation rows and any threshold a caller applies. The code comment itself says users set thresholds from these quantiles:

```
confidence_quantiles: {0.05: 0.457, 0.25: 0.483, 0.5: 0.504, 0.75: 0.523, 0.95: 0.56}
arrays[confidence] range: -0.795 -0.528
share of windows below the reported 25% quantile: 1.0
```

The CLI does not use `top1`, so only the Python API is affected.

## 7. DOC: the boundary prevalence note quotes a 2.1x floor that exp82's own records break below the 0.9 cut-off

The note tells every map below 90% prevalence that "the error rate inside the boundary set stayed at least 2.1 times the rate outside on every task (exp82)". That floor is OlmoEarth Base at its native grain. The recorded artifacts show lower risk ratios well below 90%:
- `exp82_cues/olmoearth_base_x2.json`, cashew: 1.50 at prevalence 0.852
- `olmoearth_base_x4.json`, cashew: 1.30 at 0.896; PASTIS: 1.78–1.88 at 0.65–0.70
- Copernicus-FM, cashew: 1.81 at 0.674
- CROMA Base, cashew: 1.96 at 0.714

A map at 0.893 gets the floor and no saturation clause, although the record measured 1.3 at 0.896:

```
0.893 | 89% of this map's windows sit on a prediction boundary; ... enrichment ran from 8.1x on a map with 10% boundary windows
        to 1.2x at 76%, while the error rate inside the boundary set stayed at least 2.1 times the rate outside on every task (exp82)
```

Above 0.9 the same note says both "at least 2.1 times ... on every task" and "fell to 1.3 times ... at 90% ... below the rate outside at 97%". An agent reading it gets two contradictory statements, both attributed to "the suite".

## 8. DOC: the verified range quoted for `low_confidence` was measured with a different cut from the cue's definition

The cue is defined as "the least confident 20% of the scene's windows", and `derive_cues` cuts within the assessed map. `EXP82_LOW_CONFIDENCE` is exp82's cue cut over the whole task's pooled windows, per `exp82_cue_verification.py` ("the least confident 20% of the task's valid windows"). exp82 also measured the per-tile cut, `low_confidence_per_tile`, and it is lower on every task:

```
quote: "...; on the suite's seven segmentation tasks 2.5x to 5.1x (exp82)"
mados          pooled-cut 5.09  per-tile-cut 1.37
sen1floods11   pooled-cut 4.51  per-tile-cut 2.46
pastis_s1      pooled-cut 3.70  per-tile-cut 2.84
m_sa_crop_type pooled-cut 4.30  per-tile-cut 3.03   (per-tile range over the 7 tasks: 1.37x to 3.35x)
```

A user who assesses one tile or scene gets a range the record did not measure for that setting.

## 9. HARDENING: half-given dates bypass the refusal, and the graded text states facts nobody gave

- With only one of `date_a`/`date_b`, the status is `"unstated"` and the reading says "the dates ... were not given". Grading goes ahead even when the labels' date shows they differ from map a:
  ```
  compare --date-a 2020-06-15 --labels lab.npy --labels-date 2020-09-15
    dates: unstated | graded_against: "the dates were not given; this grades both maps ... as if all three describe the same moment"
  ```
  The CLI note also says "(--date-a, --date-b) were not given" when `--date-a` was given. `agent_integration.md` says the command refuses unless `--labels-date` is given. That is true only when both map dates are given.
- `same_time` with no `labels_date`: `graded_against` reads "both maps and the labels describe the same time". The labels' date was never supplied.
- `--labels-date` without `--labels` is accepted silently.

## 10. HARDENING: a missing date (NaT) crashes with an unrelated error

```
dates_reading(np.datetime64("NaT"), "2020-01-01")   -> AttributeError 'NoneType' object has no attribute 'isoformat'
dates_reading(pd.NaT, "2020-06-16")                 -> TypeError Cannot compare NaT with datetime.date object
compare_inferences(..., labels=lab, dates=(np.datetime64('NaT'), np.datetime64('2020-06-15')))  -> AttributeError
```

NaT is how a date column with gaps arrives from pandas or numpy.

Also low: a `datetime64[W]` value is read as numpy's Thursday-start week (`np.datetime64('2020-01-06','W')` becomes 2020-01-02 to 2020-01-08), not the ISO week.

## 11. HARDENING: label codes still size the pooling (`--labels`; also class maps)

The groups path was remapped to 0..K-1 because "pooling allocates one count per id up to the largest". The labels path, `_pooled_argmax(..., max(n_lab, 2), ...)`, and the class-map path were not. One stray large code, such as an untagged 65535 fill or a stray 60000, makes compare allocate one window-grid array per code:

```
512x512 maps, labels in {10,20,30}                         : 0.46 s real,  202 MB max RSS
same, with 8 label rows coded 60000                        : 26.0 s real, 3.28 GB max RSS
```

That is 16,384 windows. A full Sentinel-2 tile at `--patch 4` has about 460 times as many and would run out of memory.

## 12. HARDENING: `assess --reference` has no grid or shape check, while `compare --labels` does

```
reference in EPSG:32618, 400 km away: assess -> runs, "against the reference: error capture 1% -> 0.00 ..."
same file as compare --labels         -> "sref_far.tif is not on the first map's grid ..."
reference of another shape: assess    -> ValueError: operands could not be broadcast together with shapes (16,16) (8,8)
```

## 13. HARDENING: `dawid_skene` input and edge handling

```
iters=0                          -> UnboundLocalError: cannot access local variable 'conf'
bool votes (votes > 0), C=2      -> IndexError: boolean index did not match indexed array along axis 2
float votes / a NaN vote         -> IndexError: only integers ... are valid indices   (the range check passes NaN)
n_classes=4, class 3 never voted -> 80 iterations, 80 "divide by zero encountered in log" RuntimeWarnings (result is finite)
tol=0, iters=300                 -> {'converged': False, 'last_change': 0.0}   (use <=)
single rater                     -> 1000 iterations, converged False, reliability 0.83 (unidentifiable; worth refusing or saying)
```

Perfect agreement (reliability 0.9998, converged in 1 iteration), and a rater that never votes one class (272 iterations, converged), both behave.

## 14. DOC: the Usage example grades the cross-date pair that the next paragraph says is refused

`docs/Usage.md` grades GEOID EMSR275-1 (pre-event S2 head against post-event S1 head) against the post-event labels with no `dates`. It runs with `status` "unstated" and prints `share_b_right` 0.79. The "Dates" paragraph directly below uses this example to explain that cross-date grading "is refused unless the labels' date is given". Passing any two different dates (illustrative ones tried) triggers the refusal. The example should pass `dates=` and `labels_date=`.

## 15. HARDENING (low): NDWI is 0 whenever green + NIR <= 0, not only "where both bands are 0"

```
green 0.004, nir -0.006  -> NDWI 0.0   (inside the ndwi_ambiguous band |NDWI| < 0.1; the pixel is dark water, green > NIR)
green 0.03,  nir -0.005  -> NDWI 1.4   (outside [-1, 1])
```

Negative near-infrared reflectance over water is common in L2A after the BOA offset (DN 1040/950 with the -1000 offset gives these values).

## 16. HARDENING (low): the new refusals reach the user as tracebacks, and compare and assess disagree on an empty map

`assess` on a map with no valid window, or with `--patch` larger than the map, raises `ValueError` with a full traceback instead of a `SystemExit` message. `compare` on a map that is all no-data exits 0 with "0 of 0 windows differ (undefined)".

## 17. HARDENING (low): taskcard reports a real project as "not found" under a GitHub rate limit

When the contents API returns 403, `_variants` falls back to `model.yaml`. kenya_lulc_croptype has no `model.yaml`, so the command ends with `LookupError: neither model.yaml nor olmoearth_run.yaml could be read for kenya_lulc_croptype; is the name right?`. Simulated by making `_get` raise HTTP 403 for the API URL only.

## 18. Minor (DOC or HARDENING, one line each)

- `determinism_check` docstring: `passes` is "None without a floor". It is also None when no window was compared.
- CLI note "the labels carry 4 classes and the two maps predict at most 2" for a label raster holding the single class 3. It counts max id + 1, not classes.
- `fit_side` with groups and no disagreement window raises "cross-fitting by group needs at least two groups, got 0". Without groups the same input returns NaN.
- `confusion_pairs` truncates float references (`[0.6, 1.4]` becomes `[0, 1]`) and reads `valid=0.4` as True. The package convention elsewhere is `> 0.5`.

---

## Checked, nothing found

- **`dates_reading`, `_period`:**
  - Periods that touch (Jan/Feb: `different_time`, 1 day), that nest (`overlapping_time`), that share a start with different ends, and equal-ended intervals are all read correctly.
  - A reversed interval and trailing characters are refused.
  - Offsets are read in UTC. Month and year `datetime64` values cover their whole period.
  - `'2020-06'` and `'2020'` strings are refused with a clear message. That is not symmetric with `datetime64('2020-06')`, but it is not silent.
- **`_graded_against`:** correct text when `labels_date` equals map a's or map b's single date or period, including an annual period that overlaps a single date.
- **Refusal:** `different_time` and `overlapping_time` with labels and no `labels_date` refuse in both the CLI and the API.
- **`_boundary_valid` against `boundary_indicator`:** identical away from the invalid cell on 200 random maps (sizes 1–8, one invalid cell). Edge replication matches `np.pad(mode="edge")`.
- **`confusion_pairs`:**
  - Ordering and tie order, `top=0`, negative references, no errors: all handled.
  - The docstring's 18% (ForestNet) to 61% (BreizhCrops) matches `exp82_summary.json` (0.185, 0.608).
- **Boundary note numbers:** 8.1x at 10% (MADOS 0.102), 1.2x at 76% (cashew 0.765), 2.1 (cashew RR 2.14 at native grain), 1.3 at 90% (0.896), and below 1 at 97% (AnySat 0.966, RR 0.81) all match the exp82 artifacts. The problem is where the 2.1 is applied (finding 7), not its value.
- **CLI compare edge cases:**
  - One window, no disagreement, a single-class map with labels of a class never predicted, and a group with no valid window all give correct numbers and nulls.
  - `over_groups` counts the empty group as undefined.
  - The `-1` no-group id is removed from both `per_group` blocks.
  - The graded block covers only majority-label windows, and the label-free numbers are unchanged by `--labels`.
- **Cue quotes:** `low_confidence` at a quantile other than 0.2 is refused a measured number, as intended.
- **taskcard, live on 4 projects (mangrove, ecosystem_type_mapping, awf, lfmc):**
  - `window_buffer`, `nodata_value` and the splitter's `grid_size` (10.0 against the partitioner's 0.1) match the upstream YAML.
  - The legend is read from `allowed_values`, and the pooling decoder marks mangrove as not dense.
- **`evidence.pool_to_patches`:** crops a ragged edge.
- **`calibrate._folds`:** refuses a single group.
- **The in-scope test files** (compare, explain, assess, reliability, cli, calibrate, signals, audit_fixes, exp82, exp83): 148 passed, 2 skipped.

Outside the diff and not reported as a finding: `fit_ranker`'s `held_out_lead_over_best_single` has the same population mismatch as finding 5. `best_single_excess_aurc` covers all rows and the held-out AURC covers the scored rows.
