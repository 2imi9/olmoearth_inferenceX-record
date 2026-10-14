# Review: estimate / per-class / certify / CLI changes since 1.2.0 (69bebdb)

Scope read: `git diff 69bebdb` of `oe_inferencex/estimate.py`, `cli.py`, `metrics.py`, `stats.py`, plus the unchanged
`sample_for_estimation` / `estimate_error_rate` / `stratified_interval_wilson` / `review_set_check` that the new code
calls. Every repro below was run with `uv run --no-sync`. Scripts and synthetic maps are in
`/private/tmp/claude-501/-Users-lucas-Desktop-Github-olmoearth-inferenceX/08a8fdd9-9831-4e63-bef7-8b20afa649c5/scratchpad/rv/`
(`bin.npy` is a 160x160 binary probability map, `cls3.npy` a 3x160x160 score map; both have 1600 windows at patch 4).
No repository file was changed. `tests/test_estimate.py` passes (43 passed).

Ranked most severe first.

---

## 1. DEFECT: `certify --rule bonferroni` crashes on every successful certification

`certify_zone` sets `out["note"]` only for `plugin` and `prefix` (estimate.py:1073-1077), and for bonferroni only
when windows tie across the threshold. `cmd_certify` always prints `{res['note']}` (cli.py:572). The JSON and the mask
are written first, then the command exits 1 with a traceback. No test runs the CLI with bonferroni.

```
oe-inferencex sample bin.npy --budget 300 --design random --out rand.csv   # then wrong=0 in every row
oe-inferencex certify rand.csv --alpha 0.05 --rule bonferroni
  File ".../oe_inferencex/cli.py", line 572, in cmd_certify
    f"{res['note']}\nwrote {out} and the window mask {mask_path}")
KeyError: 'note'
exit=1
```

## 2. DEFECT: a floating-point sum just above 1 refuses valid samples (`--per-class` when every label is right; `estimate` when every label is wrong)

The stratified estimate is `sum_h (N_h/N) * p_h`. When every p_h is 1, that sum can round to 1.0000000000000002.
The estimate is then passed as `k = est * n` to `wilson_interval`, which rejects any k greater than n. This happens in
`stratified_interval_wilson` (confidence design, all labels wrong) and in `_wilson_eff` (overall accuracy under both
designs, all labels right). Under the random design the weights are `N_map/N`, so the input doesn't need to be unusual.
The refusal is deterministic for a given map.

```python
# random design, every labelled window right, random maps (N 400-5000, 2-7 classes)
E.estimate_per_class(smp, mc[smp["indices"]], mc)
-> 12 of 400 random maps refused; e.g. (3105, 7, 'wilson_interval needs 0 <= k <= n, got k=200.00000000000006, n=200.0')
# any 5-way strata partition: 307 of 20000 have sum(N_h/N) > 1
```
CLI, on a 16x79-window 3-class map (`ov.npy`) with the default confidence design, every window right:
```
oe-inferencex estimate ov.csv              -> error rate 0.0%, 95% interval 0.0% to 1.3% ...   (fine)
oe-inferencex estimate ov.csv --per-class  -> estimate --per-class: wilson_interval needs 0 <= k <= n, got k=300.00000000000006, n=300.0   exit=1
# same map, every window wrong (wrong=1):
oe-inferencex estimate ovw.csv             -> estimate: wilson_interval needs 0 <= k <= n, got k=300.00000000000006, n=300.0   exit=1
```
A clean map at 300 labels often shows no error, so the `--per-class` case is the likely one. Suggested fix: clip the
estimate to [0, 1] before building the interval in `_wilson_eff`/`stratified_interval_wilson`.

## 3. DEFECT: producer's accuracy is None ("n/a") under the random design for a class the map never predicts; the confidence design reports 0%

In the random branch the producer's accuracy is computed only `if Nhat_j > 0 and n_i[c] > 0` (estimate.py:796). For
a reference class the map never predicts, `N_map[c] = 0` and so `n_i[c] = 0`. But Olofsson's eq. 7 gives exactly
0 there, the worst possible producer's accuracy. The confidence design correctly returns 0 with an interval.

```python
# map predicts 0/1 only; 10% of the ground is class 2
random     | class 2: n_labelled_reference 31 | producer's accuracy None | user's None
confidence | class 2: n_labelled_reference 35 | producer's accuracy {'estimate': 0.0, 'low': 0.0, 'high': 0.0989} | user's None
```
The CLI prints `producer's n/a` for it (see finding 8B, class 11), which hides a class that is entirely missed.

## 4. DEFECT: `certify` refuses a genuine random sample when about half the map or more ties at the least-confident margin

`review_set_check` compares the sample's median mid-rank percentile with a threshold derived for continuous
percentiles, 0.5 + 2.4/sqrt(n). A tied block of share s at the suspect end has mid-rank percentile 1 - s/2. For s
between about 0.5 and 0.72, a random sample's median lands inside that block, which sits above 0.64. `certify` runs
this check even though the CLI already knows from the sidecar that the design is random.

```python
# N=4000; a share of windows tied at the lowest margin; 50 random samples of 300 each
30% tied: median 0.474 (thr 0.639); random samples refused 0/50
50% tied: median 0.750 (thr 0.639); random samples refused 27/50
60% tied: median 0.700 (thr 0.639); random samples refused 50/50
70% tied: median 0.650 (thr 0.639); random samples refused 50/50
```
CLI (`half.npy` is bin.npy with 60% of pixels exactly 0.5):
```
oe-inferencex sample half.npy --budget 300 --design random --out half.csv   # wrong=0 everywhere
oe-inferencex certify half.csv --alpha 0.05
certify: these 300 windows sit at a median suspicion percentile of 0.70, above 0.64: an enriched set, not a random
sample, and a zone certified on it would be wrong. Draw the sample at random.     exit=1
```
`estimate_from_indices` goes through the same check.

## 5. HARDENING (serious): `certify` and `estimate --per-class` never check that the recomputed map is the one the sample was drawn on

`_map_windows` checks only the window-grid shape. Several things go unchecked:
- `certify` never compares its population with `side["n_population"]`.
- The sidecar does not record `--nodata`.
- The CSV's `confidence` column is not compared with the recomputed margin.
- For the confidence design, `estimate_per_class` takes its population from the sample, so its N check can never fire.

(a) A different map on the same grid is certified from this map's labels:
```
oe-inferencex certify rand.csv --alpha 0.05 --scores cls3.npy      # rand.csv was drawn on bin.npy
the 100% most confident windows (1600 of 1600, confidence margin >= 0.4505) are wrong at most 5% ...   exit=0
CSV confidence of the first 3 sampled windows: [0.5063, 0.4133, 0.5415]
cls3.npy confidence at the same windows:       [0.5151, 0.5163, 0.5334]
```
(b) A sample drawn with `--nodata 0`, then certified without it (no-data stored as 0 in 10% of the map):
```
sample n_population 1440 | certify n_population 1600 | coverage None     (no message; `grep -c nodata nd.json` -> 0)
```
With 40% no-data the same mismatch surfaced only as a misleading "enriched set, not a random sample" refusal.
`estimate --per-class` refuses the random-design case, but its message tells the user to "pass the map's class per
window ... negative where the map has no data", not to pass `--nodata`.

(c) Confidence design `--per-class` against another map (sample drawn on cls3.npy, `--scores bin.npy`): exit 0, a
2-class table for a 3-class sample.

## 6. DEFECT (regression, stats.py): `paired_cluster_bootstrap` splits every cluster in two when one side's ids are float

The new branch `if ca.dtype.kind != cb.dtype.kind: astype(str)` turns int 1 into '1' and float 1.0 into '1.0'. At
69bebdb the concatenation promoted both to float and they matched. The two subsets are then resampled independently,
which the docstring says the function exists to avoid.
```python
ca = np.repeat(np.arange(10),10); cb = ca.astype(float)
int vs int   : difference_lo -0.2483, difference_hi -0.1098
int vs float : difference_lo -0.2656, difference_hi -0.0923
ids after the str cast: ['0' '0.0' '1' '1.0'] ... count 20
```

## 7. DEFECT (edge): the random-design overall interval under-covers near a census and at some realistic cells, although the exact form is already in the module

`estimate_error_rate` (random design) uses the finite-population Wilson interval. `hypergeom_interval` was adopted
for the user's accuracy because that Wilson form fell below 0.93, but the overall rate is the same sampling situation.
Exact coverage from `exact_coverage_srs`:
```
N=1000 B=N-1: min over K 0.794 (K=206); cells below 0.9: 214 of 999
N=1000 B=N-5: min over K 0.832 (K=36);  cells below 0.9: 42 of 999
N=300  B=N-1: min 0.797;  N=5000 B=N-1: min 0.794
realistic B=300: (400,12,300) wilson+fpc 0.918 | hypergeom 0.987 ; (1000,10,300) 0.926 | 0.990 ; (22598,68,300) 0.938 | 0.987
```
No warning is given for the overall rate near a census; the near-census warning exists only per class.

Relatedly, on a full census the confidence design reports a non-zero-width interval where the random design returns
the point. This errs wide, but it contradicts "a census has no sampling error":
```
random     census: estimate 0.1125  interval [0.1125, 0.1125]
confidence census: estimate 0.1125  interval [0.0851, 0.1472]   (true 0.1125)
```

## 8. HARDENING: the CLI's per-class inputs are validated by count only, or not at all

A. `per_class_note` compares the number of wrong windows, not which windows (cli.py:507). It is also written only to
the JSON and never printed. Here `wrong` marks rows 0-9 and `reference_class` marks rows 10-19:
```
error rate 3.3% ... ; 'per_class_note' in JSON: False ; 1 - overall_accuracy = 0.0338
```
B. A `reference_class` outside the map's classes is accepted. The CLI never passes `n_classes`, although a
(C,H,W) map fixes it. A typo "11" on a 3-class map printed classes 3-10 as phantom rows, and class 11 showed
`producer's n/a` (finding 3). Exit 0.

C. `reference_class` = "inf" or `index` = "inf" raises `OverflowError: cannot convert float infinity to integer`
as a traceback. `int(float(s))` raises OverflowError, which the `except ValueError` doesn't catch.

## 9. DOC: the CLI labels every per-class warning "[warning: few labels]"

cli.py:526 appends `[warning: few labels]` whenever a row has any warning, including the near-census and thin-strata
notes. With 1500 of 1600 windows labelled:
```
class 0: user's accuracy 94% (94-95), producer's 96% (95-96), ...  [warning: few labels]
JSON: 0 504 498 | nearly every window of this class is labelled; ...
```

## 10. HARDENING (API): `estimate_per_class` and `certify_zone` accept bad indices and class maps

```
random     duplicate index accepted by estimate_per_class   (estimate_error_rate refuses the same sample)
confidence duplicate index accepted by estimate_per_class
random     negative index -1 accepted by estimate_per_class  (wraps to the last window)
confidence negative index: KeyError -1
confidence, map_class -1 at 5 unlabelled population windows -> ValueError 'list' argument must have no negative elements
confidence, map_class of 2500 windows for a sample drawn on 2000 -> accepted
non-integer map_class -> IndexError arrays used as indices must be of integer (or boolean) type
certify_zone(..., indices with -1) -> accepted, coverage 1.0 ; estimate_from_indices likewise
certify_zone(..., rule="bonferoni") with a 40-label budget -> returns "cannot certify any zone" instead of refusing the typo
```
The `estimate_per_class` docstring says the count of non-negative `map_class` entries must equal the sample's
population. That is not enforced for the confidence design.

## 11. HARDENING (API): `sample_for_estimation` samples no-data windows when given assess's confidence, as its docstring says to

The docstring says to pass `margin` as assess's `arrays["confidence"]` flattened, which is NaN at no-data.
Non-finite margins are not excluded here, although `review_set_check` and `zone_order` exclude them.
```
windows 256 valid 160 NaN confidence 96
random     | n_population 256 | sampled no-data windows 38 of 100
confidence | n_population 256 | sampled no-data windows 17 of 100 | strata sizes [160, 0, 0, 0, 96]   (no "note")
certify_zone on the random sample: 38 labelled window(s) are outside the valid map
```
A NaN in `p1` silently floors that stratum at Q_FLOOR, because Python's `max(0.02, nan)` returns 0.02:
```
allocation with finite p1: [75, 72, 65, 53, 35] | with one NaN p1 in stratum 0: [27, 88, 80, 65, 40]
p1 longer than the map: accepted ; shorter: IndexError
```
The CLI is safe from this: it passes `valid` and a `p1` pooled over valid pixels.

## 12. HARDENING: `certify --alpha 1e-17` gives a ZeroDivisionError traceback

`min_labels_to_certify` computes `log(1.0 - alpha)`, which is 0.0 for alpha below about 1.1e-16. `math.log1p(-alpha)`
is the stable form (at alpha 1e-12 the current value is 2,302,636,031,263 against the exact 2,302,585,092,993).

## 13. HARDENING: a stale zone mask survives a later run that certifies nothing

After `certify rand.csv --alpha 0.05` writes `rand_zone.npy`, running `certify rand.csv --alpha 0.005` rewrites
`rand_zone.json` with `coverage: None` and no `zone_mask` key. The old `rand_zone.npy` from the first run is left beside it.

## 14. HARDENING: `expected_calibration_error` returns 0.0 when no unit has finite data

`expected_calibration_error([nan, nan], [1, 0])` gives `(0.0, [])`. That reads as "perfectly calibrated" where the
value is undefined, which is the distinction `_pct` in cli.py exists to keep.

## 15. DOC: the Usage.md module table (edited since 69bebdb) contradicts the code

- It says "the stratified Wald interval", but the user path uses `stratified_interval_wilson`, the Wilson interval
  on the effective sample size. `stratified_interval`'s own docstring says so.
- It says per-class quantities have "Wilson intervals on the effective sample size", but under a random design the
  user's accuracy uses the exact hypergeometric interval, as the prose above the table correctly says.

---

## Checked, no defect found

- `_hyper_tail` against scipy over 20,000 random (N, K, n, k), including k outside the support: max absolute error
  7.9e-13. `hypergeom_cdf`: 6.7e-13.
- `hypergeom_interval` and `zone_upper_bound` against brute-force inversion. They differ only at knife-edge cells
  where a tail equals a/delta exactly (e.g. (16, 3, 2): P = 14/560 = 0.025), where float rounding decides. Immaterial.
- Timing at N = 4M: 0.7 ms and 0.1 ms.
- `min_labels_to_certify`: 45 at 5% and 255 at 0.9% (delta 0.1), as documented. The prefix and bonferroni
  acceptance logic, `zone_counts`, and zone sizes behave correctly. `upper_bound` is consistent with acceptance:
  UB < K0/N whenever p <= delta.
- `zone_pvalue`'s `floor(alpha*n)` can land one below the exact K0 (e.g. 0.29*100 = 28.999...). This only makes it
  conservative.
- The new `wilson_interval` finite-population form covers (100, 1, 30) at 1.0, as its docstring says. The census
  point and k = 0 reaching 0 were also checked.
- `rank_sum_test` exact branch equals a brute-force permutation on 200 random tied cases (max difference 0). It
  matches scipy exact (0.0373 / 0.0746) and asymptotic at n = 201. n = 200 runs in 0.23 s.
- Also fine: `sign_test` with numpy ints; `paired_comparison` NaN exclusion and `perm_p`; `block_bootstrap_indices`
  and `cluster_bootstrap_difference` guards; `spearman` NaN.
- Olofsson eq. 7 (producer's accuracy) and the share and overall variances in the random branch match the paper term
  by term.
- CSV intake: blank `wrong`, 0.5, TRUE, reordered or extra rows, and a BOM are all refused or handled. The tile
  design runs end to end. `--budget` 0 or above N is refused.
- Seen but not reported:
  - The zero-error fallback makes a class's share interval about 7x wider than with one error (widths 0.106 vs
    0.015). It errs wide: coverage was 0.996-1.0 in 2000 draws.
  - The model-assisted intervals collapse to [0, 0] when no labelled window is wrong. This is already on record in
    the plan's amendment (the 27%-coverage EuroSAT arm), and the tool doesn't use these estimators.
