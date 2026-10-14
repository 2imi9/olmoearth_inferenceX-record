# Release review for 1.3.0 (report only)

Repository: /Users/lucas/Desktop/Github/olmoearth_inferenceX, HEAD b2f687f (plus uncommitted exp/out files, not packaged).
Reference: v1.2.0 = 69bebdb (the tag exists on origin only; not fetched locally, so every diff below is `69bebdb..HEAD`).
Reviewer lens: what a user upgrading from 1.2.0, or installing fresh, meets. Prior line reviews not redone.

**Working tree moved during the review.** At the start `git status` showed only exp/out changes. While this review
ran, uncommitted edits appeared in `oe_inferencex/cli.py` (a `map_class` column in the `sample` CSV), `CHANGELOG.md`,
`docs/Usage.md`, `docs/claims.yaml`, `tests/test_cli.py`, `tests/test_known_answer.py` and `tests/test_upstream_revisions.py`. Everything below reviews **HEAD b2f687f**
unless it says "working tree"; the runtime checks ran from the working tree, and only `cmd_sample`'s CSV differs
from HEAD there. Whatever ships should be committed and re-checked against section 1's list.

Method: the 1.2.0 tree was extracted with `git archive 69bebdb` into the scratchpad, and the same inputs were run
through both versions: every public signature (`sigs.py`), the output keys and values of every public function
that returns a dict (`keys.py`), and each CLI command's JSON keys, CSV headers and output files (`cli/run_cli.sh`,
`cli/jdiff.py`). Scripts live in the scratchpad next to this report. Scratch outputs of this review: `v120/`, `cli/`, `rr_*` (`rr_head_src` is a `git archive HEAD` export, `rr_dist` its wheel and sdist, `rr_venv_low` the lowest-deps venv). One housekeeping note: a `head_src` path in the shared scratchpad was `rm -rf`-ed before this review recreated it (and then renamed it to `rr_head_src`), and two 1.2.0-named build artifacts were briefly written into the existing `dist/` before being moved to `rr_dist/`. If another task had its own `head_src`, it is gone.

## 1. Public API compatibility with 1.2.0

**Nothing was removed or renamed.** Every 1.2.0 name in `oe_inferencex.__all__` and every public module-level function
is still there. Signatures changed only by gaining optional trailing parameters: `compare_inferences(..., dates=None,
labels_date=None)`, `exact_coverage_srs(N, K, B, interval=None)`, `review_set_threshold(n, N=None)`,
`explain.Cue(..., measured_quantile=None, verified=None)`, `taskcard.project_card(project, variant='', model_file=...,
run_file=...)`, `taskcard.main(argv=None)`, and `dawid_skene(votes, n_classes, iters=1000, tol=1e-6, return_info=False)`,
which moved to `reliability` and is still importable from `evidence`. Across the tested paths no JSON output key was
removed; every change adds keys or changes a value. New exports: `hypergeom_interval` and `rank_sum_test`.
The CLI has one new command, `certify`, and new options: `compare --date-a/--date-b/--labels-date` and
`estimate --per-class/--scores/--nodata`. No 1.2.0 option was removed, renamed or had its default changed
(from `--help` under both versions, `help_both.txt`).

Changes a script or an upgrading user can see, classified as (a) intended and in CHANGELOG "Unreleased",
(b) intended but not stated there, (c) looks unintended:

| # | Change (1.2.0 -> HEAD) | Evidence | Class |
|---|---|---|---|
| 1.1 | `estimate_error_rate` / `estimate_from_indices` / `oe-inferencex estimate` under `design="random"`: the interval is now the exact hypergeometric one (e.g. 10 wrong of 60 labelled from 240 windows: [0.1041, 0.2693] -> [0.0958, 0.2667]), and **`method` changes from `"Wilson with finite-population correction"` to `"exact hypergeometric interval (simple random sample of a finite map)"`** | `estimate.py:490-497`; keys.py run | (a) for the interval (CHANGELOG line "Under a random sample the overall error rate ... now use the exact hypergeometric interval"). The `method` string change is **not stated**. A script that branches on `method` (as the repo's own test does, `r["method"].startswith("stratified")`) would break. SHOULD-FIX: one CHANGELOG clause naming the new string |
| 1.2 | `wilson_interval(k, n, N)` with N given: new numbers (Korn-Graubard effective size); (3, 30, 100): [0.0473, 0.2435] -> [0.0406, 0.2258]. `exact_coverage_srs` defaults to it and moves with it | `estimate.py:48-78` | (a) |
| 1.3 | `review_set_check` decides on the **mean** percentile (new key `mean_suspicion_percentile`; `median_suspicion_percentile` kept, no longer decisive). Ties are ranked in a fixed random order instead of mid-ranks. `review_set_threshold(300)` 0.640 -> 0.567. `estimate_from_indices` and `certify_zone` inherit this | `estimate.py:550-596` | (a) (the "A random sample was refused as a review set ..." item) |
| 1.4 | `assess_prediction(..., form="top1")`: `arrays["confidence"]` and `confidence_quantiles` go from log-probability (median -0.69) to probability (0.50), plus a new `confidence_scale` key | `assess.py:210-224`; keys.py | (a) |
| 1.5 | `assess` against a reference: an evenly split reference window is left unscored (new `n_windows_reference_tied`), new keys `population`, `error_capture_at_budget.<b>.n_reviewed`, `confusion_pairs`. When no window can be graded, `against_reference` becomes `{n_windows_scored: 0, n_windows_reference_tied, error_rate: NaN, note}` with none of the AURC keys | `assess.py:314-340` (HEAD) | (a) (tie change and confusion pairs stated; the reduced block replaces a 1.2.0 crash) |
| 1.6 | Prediction ties in `_pooled_argmax` go to the more confident class, not the lowest index, so `pooled_argmax`, the boundary share and error capture move (synthetic run: boundary windows 89.4% -> 90.7%) | `assess.py:56` `_pooled_argmax` | (a) ("Tied windows no longer go to class 0") |
| 1.7 | `assess_prediction` raises `ValueError` for a window larger than the map and for a map with no valid window. Top-level `pixels_outside_window_grid` key and a ragged-edge warning are new | `assess.py:242-280` `_assess` | (a) |
| 1.8 | `compare_inferences` output always carries `dates` (status `"unstated"` when not given) and, when graded, `graded.graded_against`. A negative label is now "unlabelled" (1.2.0 graded -1 as a class). With `dates` of different times and labels, it raises unless `labels_date` is given | `compare.py` `compare_inferences`, `dates_reading` | (a) |
| 1.9 | `oe-inferencex compare`: with a hard class map on either side, a tied window is left out (synthetic 32x32 run: 1024 -> 877 compared windows, 128 -> 76 differ). `--labels` no longer restricts the label-free numbers. Windows in no group are `-1`, not group 0. `--threshold` now applies to integer-stored continuous maps. New refusals: grids that differ, `--labels-date` without `--labels`, no comparable window | `cli.py` `cmd_compare` | (a) |
| 1.10 | **`disagreement.tif` / `.npy` dtype: uint8 / bool -> float32 with NaN** where nothing was compared. Under 1.2.0, `x[np.load("diff/disagreement.npy")]` worked. Under HEAD it raises `IndexError: arrays used as indices must be of integer (or boolean) type` (verified on both versions' outputs) | `cli.py:379` (HEAD) | (a) in substance ("`disagreement.tif` is NaN where nothing was compared"), but the dtype change and the `.npy` case are not stated. SHOULD-FIX: say the file is now float32 (1 differ, 0 agree, NaN not compared), so a boolean-mask reader must use `== 1` |
| 1.11 | `comparison.json`: new `dates.*`, `inputs.date_a/date_b/labels_date`, `graded.graded_against`, and `notes` (now present on more runs) | jdiff.py | (a) |
| 1.12 | `sample` sidecar JSON gains `nodata`. A 1.2.0 sidecar has none; `estimate --per-class` and `certify` then read the raster's own no-data. A 1.2.0 sample drawn with `--nodata` is caught by the population check with a message naming `--nodata`, not a wrong number. Upgrade path verified: 1.2.0 random, confidence and tile samples, filled and read by HEAD's `estimate`, `estimate --per-class` and `certify`, all run or refuse as documented | `cli.py:459`, `_map_windows` | additive, not stated; NOTE |
| 1.13 | `fit_ranker`: `held_out_lead_over_best_single` is now computed on the scored rows (new `best_single_excess_aurc_on_scored_rows`). `fit_side`: every comparator on the scored rows (new `comparators_cover`, `n_unscored_rows`, `n_scored_rows`, `unscored_note`). A single group now raises | `calibrate.py` | (a) |
| 1.14 | `stats.paired_comparison`: new key `n_undefined`. NaN gains are excluded. **`perm_p` is now `(hits + 1)/(n_perm + 1)` for every input**, not only where it was 0 (seeded example 0.0660 -> 0.0679) | `stats.py:105-124` | partly (a): the CHANGELOG says the test "dropped exactly tied patterns and could report 0", but not that every permutation p-value now carries the +1 correction. SHOULD-FIX (one clause) |
| 1.15 | `stats.block_bootstrap_indices` raises when `block` does not divide `grid` (1.2.0 returned indices that never drew the edge) | `stats.py` | (a) only as "two smaller refusals"; NOTE |
| 1.16 | `metrics.expected_calibration_error` returns `(nan, [])` when no unit is finite and **raises for confidence outside [0, 1]** | `metrics.py:146-160` | NaN case (a); the [0, 1] refusal is not stated. NOTE |
| 1.17 | `explain.library_table()` rows gain `measured_quantile` and `verified`. `Cue.quote()` appends the exp82 range for `boundary` and `low_confidence`, and fills `{quantile}` by default. `explain_review_set` gains `boundary_prevalence_note`. `reference_unstable` shares 0.10/0.007 -> 0.108/0.0079 | `explain.py` | (a) |
| 1.18 | `signals.ndwi` is scale-free, with negative bands clipped. `sample_for_estimation` drops non-finite margins from the population (same seed gives the same indices when margins are finite: verified for all four designs). `stats.sign_test`, `spearman`, `wins_losses_ties`, `reliability.kmeans` / `input_extremity`, `determinism_check`, `evidence.pool_to_patches`, `dawid_skene` (cap 50 -> 1000) | diff | (a) |
| 1.19 | `python -m oe_inferencex.taskcard`: default `--out` changed `exp/out/taskcards.json` -> `taskcards.json`, markdown goes beside `--out` (1.2.0 always wrote `docs/method/taskcards.md`), and it refuses no arguments or `--all` with projects | `taskcard.py:365-370` | (a) for the markdown and the refusal; the default `--out` path change is not stated. NOTE |
| 1.20 | **Working tree only, not in HEAD:** the `sample` CSV gains a `map_class` column before `wrong` (column order changes for anyone reading by position). The working-tree CHANGELOG now has a line for it ("The column is added before `wrong`; a CSV without it (written by 1.2.0) still reads"). If it ships, keep that line | `git diff -- oe_inferencex/cli.py` | (a) in the working tree; would be (b) if the code were committed without the CHANGELOG line |

No change looked unintended (class c). No change is both breaking and undisclosed, so section 1 has **no BLOCKER**.
Items 1.1, 1.10 and 1.14 are disclosed in substance but leave out the part a parsing script trips on, so they are
SHOULD-FIX (one clause each in the CHANGELOG).

## 2. CHANGELOG accuracy (HEAD's "Unreleased", behaviour claims only)

Each behaviour claim was checked against the code, and wherever it could be run, by running it. Numbers quoted
from experiments are out of scope.

**Claims that hold (verified by running unless marked "code"):**
- certify: exact hypergeometric tests; prefix and Bonferroni rules, plug-in as `rule="plugin"` in the API (code).
  The refusal when the budget cannot certify works: `certify --alpha 0.01` on 200 labels says 230 are needed.
  `--alpha 1e-17` gives a finite count and a refusal, not a crash. `certify --rule bonferroni` on a certifying case
  (alpha 0.2 and 0.3) exits 0 with the zone written. `certify` on a confidence-design CSV is refused. `certify
  --scores <another map>` is refused ("not the map the sample was drawn on").
- per-class: exact hypergeometric user's accuracy under a random draw, Wilson on the effective size elsewhere,
  `interval="wald"` kept, warnings `near census`, `thin strata`, `few errors` (1 to 4 errors, `RARE_ERRORS = 5`)
  and `never predicted` (code). Tile samples are refused (run). A never-predicted class gets producer's accuracy
  exactly 0 (code). The overall accuracy beside the table is `estimate`'s interval, and the post-stratified point
  is kept under `overall_accuracy_post_stratified`, random design only (code).
- `hypergeom_interval` matches a scipy brute-force tail inversion on 300 random (k, n, N) cells, widened to include
  k/n: 0 mismatches. Its exact coverage was at least 0.95 on the four cells tried. It takes about 1 ms at N = 4,000,000.
- explain: the library carries the exp82 range (`verified`, `verified_range`), and `boundary_prevalence_note` is
  reported. `_boundary_valid` now equals `boundary_indicator` in value away from invalid windows (run on a
  20 x 20 map). `library_table()` leaves no raw `{quantile}` placeholder. Quoting at another cut says the
  enrichment was measured only at 20%.
- compare dates: `2018-03-11x`, `...Zjunk` and `2018-03-11 junk` are refused. `np.datetime64("2018-03")` becomes
  `2018-03-01/2018-03-31`. `2018-03-11T23:30:00-05:00` is read as 2018-03-12 (UTC). One date alone gives
  `partly_stated` and grading is refused. NaT is unstated. Different dates with labels and no `labels_date` are
  refused, and `graded_against` names the matching map. Undated runs add the "dates were not given" note.
- compare CLI: with a class map on either side tied windows are left out and counted in `notes`. A stray code of
  60000 now costs 37 MB of peak memory. `--labels` leaves the label-free block unchanged (code), and grading
  covers only majority-label windows, with a note giving the counts.
- `assess --reference` with an all-no-data or checkerboard (all-tied) reference exits 0 and prints "nothing was
  graded", where the CHANGELOG says 1.2.0 crashed.
- The three CI test layers exist (`tests/test_cli_matrix.py`, `test_properties.py`, `test_consistency.py`), and
  `check.yml` runs `pytest -q` on Python 3.11 and 3.13. `scripts/engine_determinism.py` and the three `docs/plan/`
  pages named in the entry exist.

**Statements the code does not quite do:**

- **2.1 NOTE: "A stratified or tile sample is refused" is true of the command line only.** The sentence sits under
  "`estimate.certify_zone` and `oe-inferencex certify`". `certify_zone(margin, indices, wrong, alpha)` takes indices,
  not the sample dict, so it cannot see the design. It refuses only a set the review-set check flags. On the shipped
  Dynamic World tile, confidence-design and proportional samples of 300 were accepted 20 times out of 20: the
  allocation there is near-proportional (`[59, 59, 61, 61, 60]`), so no wrong number. On a skewed map (allocation
  `[138, 45, 39, 39, 39]`) all 20 were refused. The docstring says the indices "must be a simple random sample",
  which is correct. Fix: scope the CHANGELOG sentence to the command line, or say the API refuses an enriched set.
- **2.2 NOTE: "Above `explain.BOUNDARY_SATURATED` (0.9) the prevalence note now says the cue is not a reason on this
  map."** The code fires at `p >= 0.9` (`explain.py:215`) and says "the boundary cue is at most a weak reason"
  (`explain.py:220`). The meaning is close, but the CHANGELOG quotes a stronger statement than the tool makes.

**User-visible changes the CHANGELOG leaves out** (details and evidence in section 1):
- 2.3 SHOULD-FIX: the random-design `method` string (1.1).
- 2.4 SHOULD-FIX: the compare disagreement raster's dtype, bool/uint8 -> float32 (1.10).
- 2.5 SHOULD-FIX: `perm_p` gains the +1 correction on every call (1.14).
- 2.6 NOTE: `expected_calibration_error` raises for confidences outside [0, 1] (1.16). The taskcard CLI's default
  `--out` changed (1.19). The `sample` sidecar gains `nodata` (1.12). The new options `estimate --scores/--nodata`
  and `certify --scores/--nodata/--out` are not named. The CHANGELOG names `certify` and `--per-class`, and
  Usage covers `--alpha` and `--delta`.
- 2.7 NOTE (text an agent reads, not in the CHANGELOG): the compare note "N windows are split evenly between two classes
  on a map with no confidence to break the tie" also counts windows tied on the side that has a confidence. With a
  score map against a class map, both sides' ties are left out (`weighted=False`), so the note can blame the wrong map.
  The number is right; the wording is loose.
- 2.8 NOTE, no action implied: `certify` prints "the exact upper bound on the zone's error rate at that level is X%".
  That bound is a per-zone bound at `delta`, computed after the zone was selected, and it is not what the guarantee
  covers (the guarantee is `alpha`). A 1,000-draw simulation (`ub_sim.py`, N = 4,000, 300 labels, alpha = delta = 0.1)
  found true error above alpha on 0.000 to 0.043 of draws, and above the printed bound on 0.039 to 0.114 of draws
  (Bonferroni, smooth error profile: 0.114, Monte Carlo SE about 0.01). So no violation is shown, but the bound is
  not covered by the stated guarantee. If it should read as a guarantee, it needs its own grading.

- **2.9 SHOULD-FIX: `graded.graded_against` compares date strings for equality, so a labels date inside a map's
  period is called "the date of neither map".** The CHANGELOG says `graded_against` "says which map the labels match
  in time", and Usage says a period "covers a composite or an annual map". With `dates=("2020-01-01",
  "2020-06-01/2020-06-30")` and `labels_date="2020-06-15"`, `graded_against` reads: "the labels describe 2020-06-15,
  the date of neither map (2020-01-01, 2020-06-01/2020-06-30); each map is counted wrong wherever the ground
  changed between its date and the labels'". An annual map dated `2020-01-01/2020-12-31` labelled mid-year gets the
  same sentence. When both maps are 2020-06-01 and the labels are `2020-06-01/2020-06-30`, it says "where the ground
  changed between them both maps are counted wrong". The numbers are unaffected. The misleading part is the text an
  agent reads on a documented path. Location: `compare.py:414` (`matched = [s for s in ("a", "b") if reading[s] == lab]`).
  Reproduce: `_graded_against(dates_reading("2020-01-01", "2020-06-01/2020-06-30", "2020-06-15"))`. A fix would test
  whether the labels' period lies inside the map's.

## 3. CLI help text against behaviour

`--help` was run for the top level and every command (`help_both.txt`, both versions). Every option in docs/Usage.md,
README.md, docs/index.md and docs/method/agent_integration.md exists in the parser. The flags Usage names that are
not CLI flags (`--model`, `--signal`, `--extra`) belong to scripts and uv, and are presented that way.

- **3.1 SHOULD-FIX: `estimate --scores` and `estimate --nodata` are silently ignored without `--per-class`.**
  `oe-inferencex estimate up_rand.csv --scores /nonexistent.npy --nodata 5` exits 0 and prints the rate. With
  `--per-class` the same `--scores` is refused as not found. The help for `--scores` ("the raster `sample` was run on,
  if it has moved") does not say it applies only with `--per-class`, and `--nodata` has no help at all
  (`cli.py:736-737` HEAD).
- **3.2 SHOULD-FIX: `estimate --nodata` and `certify --nodata` have no help text.** They are new in 1.3.0, and their
  behaviour is not guessable. The default is the value in the sample's sidecar, not the raster's own. A value that
  differs from the sidecar is refused. A 1.2.0 sidecar has no `nodata`, so any value is accepted
  (`cli.py:535-538` `_map_windows`, options at `cli.py:737` and `:748`, HEAD). Suggested help: "the no-data value `sample` was run with (default: the one in
  the sidecar); only needed for a sample written by 1.2.0".
- **3.3 NOTE: `certify --rule` and `--scores` appear nowhere in Usage.md or README.** The help text describes them
  correctly. `--delta` default 0.1, `--alpha` required, and `--out` placing the `.npy` mask beside the JSON all match
  the code. `certify` exits 0 when no zone is certified, printing the reason and deleting a stale mask (verified).
- **3.4 NOTE: the `--date-a/--date-b` help does not say that grading with only one of them is refused**
  ("give both --date-a and --date-b, or neither, before grading"). The refusal message itself is clear.
  `--labels-date` "required with --labels when the two maps describe different times" also covers overlapping
  periods, which is consistent.
- **3.5 NOTE (unchanged since 1.2.0):** `sample --logits/--patch/--nodata/--seed` and `compare --nodata` have no help.
  `assess --budgets` and `--order` do not show their defaults (0.01 0.05 0.10; confidence). The top-level
  description and `cli.py`'s module docstring (`cli.py:1-16`) mention only assess, compare and demo.
  `oe-inferencex --version` does not exist.
- **3.6 NOTE: docs/Usage.md:79 still says "Four commands cover the label-free halves and the one question that
  needs labels".** HEAD has five besides `demo`: assess, compare, sample, estimate and certify.

## 4. Docstrings against behaviour (estimate.py, compare.py, explain.py, assess.py, and __init__)

The old random-design text is gone from the docstrings: no docstring still describes Wilson-with-FPC as the
random-design interval (the module docstring, `estimate.py:16-18`, says exact hypergeometric), and none presents the
post-stratified overall accuracy as the reported interval. The median rule survives in one docstring:

- **4.1 SHOULD-FIX: `review_set_check` docstring, `estimate.py:567-568`:** "Could these windows be a random sample
  of the map? The median suspicion percentile of the sample: 0.5 for a random draw, near 1 for the tool's own
  review set". The function decides on the mean (`looks_like_a_review_set = mean > thr`, `estimate.py:594`), and
  the threshold is for the mean (`review_set_threshold`, `estimate.py:550-563`). The module docstring, the constant's
  comment, Usage and the CHANGELOG all say "mean". This one line was missed, and it is what `help(ox.review_set_check)` shows.
- **4.2 SHOULD-FIX: `dates_reading` docstring, `compare.py:299-301`, lists the statuses as "unstated" (a date is
  missing), "same_time", "different_time" and "overlapping_time".** The code also returns **"partly_stated"** when
  exactly one map date is given (`compare.py` `dates_reading`; `dates_reading("2018-03-11", None)["status"] ==
  "partly_stated"`). Only both missing gives "unstated". A caller branching on the documented statuses misses the
  one that refuses grading. The CHANGELOG ("one map date alone is now 'partly stated'") and the CLI both handle it.
- **4.3 SHOULD-FIX: `assess_prediction` docstring, `assess.py:155-156`:** "the logit margin was the weakest form on
  all 16 (exp76)". This release corrected the same sentence in `signals.confidence` (`signals.py:35-37`) to "the weakest
  of the three forms on 14 of 16 by AUROC and 15 of 16 by excess AURC (exp76), tying for best on awf_sentinel2".
  The copy in `assess.py` was not corrected, so the package now states two versions of one recorded result.
- **4.4 NOTE: incomplete rather than wrong.** `oe_inferencex/__init__.py:26` lists the CLI as "`oe-inferencex assess`,
  `compare`, `sample` and `estimate`" (no `certify`, no `demo`), and its `estimate` line does not mention per-class or
  the trusted zone. The `compare.py` module docstring (`compare.py:1-33`) lists its functions without `dates_reading`
  or `determinism_check`, and its Conventions paragraph does not mention dates or that a negative label means
  unlabelled; `compare_inferences`' own docstring does. The `explain_review_set` docstring does not mention the new
  `boundary_prevalence_note` key. The `estimate.py` module docstring ("Three things a user needs") does not mention
  per-class accuracy, `certify_zone` or the model-assisted estimators.
- Checked and consistent: `wilson_interval` (Korn-Graubard form described), `hypergeom_interval`,
  `exact_coverage_srs`, `review_set_threshold` (mean; 0.567 at 300), `estimate_from_indices` (mean),
  `estimate_per_class` (hypergeometric user's accuracy under random, Wilson elsewhere, Wald kept, tiles refused),
  `certify_zone`, `zone_*`, `min_labels_to_certify`, `compare_inferences` (dates, `graded_against`, negative labels),
  `determinism_check`, `confusion_pairs`, and the `assess.py` module docstring.

## 5. Package metadata and a fresh install

**Fresh install at the lower bounds, run.** A wheel and sdist were built from a `git archive HEAD` export
(`rr_head_src`, `rr_dist`). The wheel was installed into a new Python 3.11.12 venv with **numpy 1.26.4, pyyaml
6.0.1, huggingface_hub 0.20.0** and nothing else. Results:
- HEAD's test suite under those versions: **845 passed, 5 skipped** (skips: rasterio, scipy, olmoearth_pretrain).
- The documented CLI paths ran with the same install: `demo`, `demo --made-up`, `assess --reference`, `sample`
  (random, confidence, with `--nodata -9999` on a 3-class map with a no-data strip), `estimate --per-class`
  (random and confidence), `certify` (prefix, bonferroni, and a wrong `--nodata`, which was refused), and
  `compare --labels --date-a --date-b --labels-date`. All exited as documented.
- A grep of the package for numpy-2-only APIs (`np.bool`, `np.concat`, `np.astype`, `np.unique_*`, `sort(stable=)`,
  `asarray(copy=)`, `.mT`, the `np.linalg` array-API names and others) found none. The new code uses
  `np.random.default_rng(0).random`, `np.lexsort` and `np.unique(..., return_inverse=True)`, all present in 1.26.
  No numpy lower-bound problem.
- pyyaml 6.0 (the exact lower bound) installs on 3.11.
- `twine check` passes on both artifacts. The wheel holds the package, `sample/dynamic_world_tile.npz` and
  `NOTICE.txt`, and LICENSE. The sdist carries no tests (`MANIFEST.in: prune tests`), as intended.
- All README links return 200 (raw.githubusercontent images, readthedocs pages, GitHub LICENSE / CITATION.cff /
  CHANGELOG). PyPI holds 1.1.0 to 1.2.0, so 1.3.0 is free.

**Every place the version string lives (all still 1.2.0 at HEAD; the release commit must change each):**
1. `pyproject.toml:3`, `version = "1.2.0"`. This is what `uv build` stamps and `publish.yml` uploads. Nothing
   checks it against the tag: a v1.3.0 GitHub release published from an unbumped tree would try to upload 1.2.0 and
   fail on PyPI's duplicate check. That fails safe, but loudly.
2. `oe_inferencex/__init__.py:62`, the source-checkout fallback `__version__ = "1.2.0"`.
3. `CITATION.cff:8`, `version: 1.2.0`, and `CITATION.cff:9`, `date-released: 2026-09-22`.
4. `CHANGELOG.md:3`, `## Unreleased`, which becomes `## 1.3.0 (date)`.
5. `uv.lock:881-882`, the editable project entry `version = "1.2.0"`. CI runs `uv run --frozen`, which does not
   notice a stale lock, so run `uv lock` after the bump or the committed lock lags the release.
6. The git tag `v1.3.0`. The local clone does not have `v1.2.0` (it exists on origin at 69bebdb); `git fetch --tags`
   before tagging.
No other file (README, docs, mkdocs.yml) carries the version.

**Metadata notes (none blocks):**
- 5.1 NOTE: `requires-python = ">=3.11,<3.14"`. The cap dates from the first scaffold (ccfa85d). The core
  package is numpy-only, and the cap seems to come from the torch extra. A user on Python 3.14 gets no installable
  version. Installing on 3.14 was not tested here (uv offers only 3.14.0a6 in this environment).
- 5.2 NOTE: README images come from `raw.githubusercontent.com/.../main/...` and the docs links from
  `readthedocs.io/en/latest/`. The PyPI page of 1.3.0 therefore shows whatever `main` and `latest` hold later,
  including a demo figure whose caption numbers are fixed in the README text. This predates 1.3.0.
- 5.3 NOTE: the `summary` / description still reads "Label-free audit ... which windows to review first, why, and
  how two inferences differ". It is accurate, but it does not mention the labelled half (estimate, per-class,
  certify) that 1.2.0 and 1.3.0 added.
- 5.4 NOTE (tests, not the package): `tests/test_cli.py:12` skips the whole module without rasterio, but its reason
  says "the .npy path is tested below regardless". Under the lowest-deps install, none of test_cli.py ran. CI
  installs the geo extra, so the tests run there. My smoke runs covered the `.npy` CLI paths.
- 5.5 NOTE (pre-existing): `assess`/`sample` on a (C, H, W) logit map print "multi-class logit margin: ... pass
  form='top1'", but the CLI has no `--form` option, so a command-line user cannot act on the advice.
- Classifiers, `license = "Apache-2.0"` with `license-files`, `License-Expression` in METADATA 2.4, and the
  `Changelog` URL are fine.

## 6. A regression found while checking the upgrade path

- **6.1 BLOCKER (narrow; one-line fix): a full census is refused as "an enriched set" by `estimate_from_indices`
  and `certify`, in about 2 to 5% of cases. 1.2.0 accepted it.** `review_set_threshold(n, N)` applies a
  finite-population factor `(N - n)/(N - 1)` (`estimate.py:562-563`), which is 0 at a census, so the threshold is
  exactly 0.5. The census's mean percentile is 0.5 up to float rounding, and the test is a strict `mean > thr`
  (`estimate.py:594`). Whenever the sum rounds up, a census is refused as a review set.
  - Python: `rng = np.random.default_rng(0); m = rng.normal(size=100); ox.estimate_from_indices(np.arange(100),
    (rng.random(100) < 0.1).astype(int), m)`. Under 1.2.0 this returns the census point `0.11, [0.11, 0.11]`. Under
    HEAD it raises `ValueError: these 100 windows sit at a mean suspicion percentile of 0.50, above 0.50, the most a
    random sample of that size reaches ...`. `certify_zone(m, np.arange(100), np.zeros(100), 0.1)` raises the same way.
  - Command line: a `(40, 20)` float32 logit map from `np.random.default_rng(20001).normal(0, 2, (40, 20))`, then
    `oe-inferencex sample m.npy --logits --budget 50 --design random --seed 1 --out c.csv`, `wrong` = 0 on every row,
    then `oe-inferencex certify c.csv --alpha 0.1` gives: "certify: these 50 windows sit at a mean suspicion
    percentile of 0.50, above 0.50: an enriched set, not a random sample, and a zone certified on it would be wrong.
    Draw the sample at random."
  - How often: a census in arange order was refused for 51 of the 2,999 population sizes 2 to 3,000. In random order
    (as `sample` writes the indices) it was refused 88 times in 1,962 (N = 20 to 1,000, two orders each). A random
    draw of N - 1 windows was never refused (0 of 1,050), so only the census is degenerate.
  - Why it matters here: this release adds census handling on purpose ("A census under the confidence design is
    reported as the point", the widening "near a census" in `hypergeom_interval`). A census is by definition not an
    enriched sample, and the error message states the opposite. It is not a wrong number, but valid, documented input
    raises where 1.2.0 answered, and the refusal's reason is false.
  - Fix, for the maintainer: skip the check when `inside.sum() >= pop.size`, or compare with a tolerance
    (`mean > thr + 1e-12`). Either way, add a census test for `estimate_from_indices` and `certify_zone`. This review
    edited nothing.

## Findings by severity

| Severity | # | Finding | Where | Reproduction or quote |
|---|---|---|---|---|
| BLOCKER | 6.1 | A full census is refused as "an enriched set" by `estimate_from_indices` and `certify` (2 to 5% of cases, float rounding against a threshold of exactly 0.5). A regression from 1.2.0; one-line fix | `estimate.py:562-563, 594` | `estimate_from_indices(np.arange(100), wrong, rng.normal(size=100))` with `rng = default_rng(0)`: 1.2.0 gives 0.11; HEAD raises |
| SHOULD-FIX | 1.1 / 2.3 | Random-design `method` string changed and the CHANGELOG does not say so | `estimate.py:497` | `"Wilson with finite-population correction"` -> `"exact hypergeometric interval (simple random sample of a finite map)"` |
| SHOULD-FIX | 1.10 / 2.4 | `disagreement.tif/.npy` changed from bool/uint8 to float32 with NaN; a boolean-mask reader breaks | `cli.py:379` | `x[np.load("diff/disagreement.npy")]`: IndexError under HEAD |
| SHOULD-FIX | 1.14 / 2.5 | `paired_comparison.perm_p` is now `(hits+1)/(n_perm+1)` on every call; the CHANGELOG mentions only the zero case | `stats.py:120-123` | seeded example 0.0660 -> 0.0679 |
| SHOULD-FIX | 2.9 | `graded_against` says "the date of neither map" when the labels' date lies inside a map's period (composite, annual map) | `compare.py:414` | `_graded_against(dates_reading("2020-01-01", "2020-06-01/2020-06-30", "2020-06-15"))` |
| SHOULD-FIX | 3.1 | `estimate --scores/--nodata` silently ignored without `--per-class`; the help does not say so | `cli.py:736-737` | `estimate s.csv --scores /nonexistent.npy` exits 0 |
| SHOULD-FIX | 3.2 | `estimate --nodata` and `certify --nodata` (new) have no help; default is the sidecar's value, a different value is refused | `cli.py:535-538, 737, 748` | `certify --help` shows `--nodata NODATA` with no text |
| SHOULD-FIX | 4.1 | `review_set_check` docstring says it reads the median; it decides on the mean | `estimate.py:567-568` | "The median suspicion percentile of the sample: 0.5 for a random draw" |
| SHOULD-FIX | 4.2 | `dates_reading` docstring omits the `partly_stated` status | `compare.py:299-301` | `dates_reading("2018-03-11", None)["status"] == "partly_stated"` |
| SHOULD-FIX | 4.3 | `assess_prediction` docstring keeps "weakest form on all 16 (exp76)", which this release corrected in `signals.confidence` | `assess.py:155-156` vs `signals.py:35-37` | quoted in 4.3 |
| NOTE | 1.12 | `sample` sidecar gains `nodata` (additive; 1.2.0 sidecars handled, with a clear refusal when their `--nodata` was set) | `cli.py:459` | verified with a 1.2.0 sample drawn with `--nodata -9999` |
| NOTE | 1.15 | `block_bootstrap_indices` refuses a block that does not divide the grid ("two smaller refusals") | `stats.py:152` | |
| NOTE | 1.16 | `expected_calibration_error` raises for confidence outside [0, 1], not stated | `metrics.py:159-160` | |
| NOTE | 1.19 | taskcard CLI default `--out` changed from `exp/out/taskcards.json` to `taskcards.json` | `taskcard.py:370` | |
| NOTE | 1.20 | Working tree only: `map_class` column added to the `sample` CSV. The working-tree CHANGELOG has its line; keep it if it ships | `git diff -- oe_inferencex/cli.py` | |
| NOTE | 2.1 | "A stratified or tile sample is refused" holds for the CLI; `certify_zone` refuses only an enriched set | CHANGELOG, `estimate.py` `certify_zone` | near-proportional confidence sample accepted 20/20 on the shipped tile |
| NOTE | 2.2 | CHANGELOG says the saturated note calls the boundary cue "not a reason"; the code says "at most a weak reason", at `>=` 0.9 | `explain.py:215-220` | |
| NOTE | 2.6 | New options `estimate --scores/--nodata`, `certify --scores/--nodata/--out` not named in the CHANGELOG | | |
| NOTE | 2.7 | compare's tie note says "on a map with no confidence" and also counts ties on the side that has one | `cli.py` `cmd_compare` | |
| NOTE | 2.8 | certify's printed "exact upper bound ... at that level" is a per-zone bound after selection, outside the guarantee (simulated miss 0.039-0.114 at delta 0.1; not shown to exceed) | `estimate.py` `certify_zone` | `ub_sim.py` |
| NOTE | 3.3-3.6 | `certify --rule/--scores` missing from Usage; the one-date grading refusal is not in `--date-a/-b` help; pre-existing help gaps; Usage.md:79 "Four commands" | | |
| NOTE | 4.4 | `__init__` docstring lists the CLI without `certify`; the compare/explain/estimate module docstrings do not cover the new features | `__init__.py:26` | |
| NOTE | 5.1-5.5 | `requires-python <3.14` shuts out 3.14; README images/docs point at `main`/`latest`; package description predates the labelled half; `test_cli.py` skips whole-module without rasterio despite its reason; the CLI advises `form='top1'` it cannot take | | |

Also required for the release commit (not findings): bump the version in the six places in section 5
(pyproject, `__init__` fallback, CITATION.cff version and date, CHANGELOG heading, `uv lock`, tag). Commit or drop
the working-tree changes, then re-run section 1's key diff (`keys.py`, `cli/run_cli.sh`, `cli/jdiff.py`) on the final
tree.

No wrong number was found on a documented path. The compatibility check found no removed or renamed public name,
option or JSON key. Lower-bound installs work (845 tests pass on numpy 1.26.4 / Python 3.11).
