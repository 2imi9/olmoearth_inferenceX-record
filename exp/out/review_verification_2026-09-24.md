# Verification of the 33 review fixes (working tree on 4a80efe), 2026-09-24

Method: every reproduction re-run with `uv run --no-sync` against the working tree, and where a before/after
comparison matters, against a copy of HEAD's package (`git archive HEAD oe_inferencex` into
scratchpad/vf/head, imported via PYTHONPATH from the scratchpad). Scripts and maps are in scratchpad/vf/.
No repository file was edited. (Report written incrementally; sections below fill in as checks finish.)

## Progress log (appended as each check finishes)

- pytest: `uv run --no-sync pytest -q` -> 875 passed, 2 skipped, 44 warnings in 78 s.
- C1 FIXED: ga/gb (40 m N-S shift at UTM northing 5e6) and wa/wb (100 m at Web Mercator 1.2e7) now refused "not on the first map's grid"; ga vs ga passes.
- C2 FIXED: all-tied, all -1 and one-pixel-per-window references print "no window has both a prediction and a majority reference label, so nothing was graded", exit 0 (HEAD: KeyError).
- C3 PARTLY FIXED: tie_cls vs tie_prob and field_cls vs field_prob now 0 differ (HEAD 1/1 and 305/16384). Confidence-only drift between two probability maps still flips a tie (field_prob vs field_prob_b: 1 of 16384 both HEAD and now; the weighted path was left alone deliberately). Usage.md still says without qualification "a tied window goes to the more confident voters".
- C4 FIXED: p_api_lab2 API and CLI both {'n': 2, 'errors_a': 1, 'errors_b': 0, 'both': 0}.
- C5 FIXED: p_side baseline 0.47, always_a 0.0, always_b 1.0 on the 200 scored rows; "comparators_cover": "the scored rows".
- C6 FIXED: form="top1" array range 0.341-0.436, share below the reported q25 = 0.25 (HEAD 1.0); margin form unchanged.
- C7 FIXED: the new note's numbers all match exp82 artifacts (see record section).
- C8 FIXED: EXP82_LOW_CONFIDENCE equals exp82 `low_confidence_per_tile` on all 7 tasks to 4 d.p.; quoted range 1.4x-3.4x.
- C9 FIXED: one map date + labels refused in CLI and API; --labels-date without --labels refused; same_time without labels_date says the labels' date was not given; the "(--date-a, --date-b) were not given" note no longer appears when --date-a is given.
- C10 FIXED: NaT (numpy generic, [D], pandas) reads as unstated/partly_stated; datetime64[W] refused with a named reason.
- C11 FIXED: stray 60000 label code, 512x512: HEAD 17.4 s / 3.8 GB RSS, now 0.15 s / 60 MB.
- C12 FIXED: assess --reference in another CRS, shifted 40 m, or of another shape all refused (HEAD: ran, or broadcast traceback).
- C13 FIXED: iters=0, NaN, 1-D, single rater refused; bool and float votes work; tol=0 converges (255 it.); no log-0 warnings; ordinary 3-rater result bit-identical to HEAD.
- C14 FIXED (by explanation, not by passing dates): example numbers reproduce (0.0943, 3310 of 35084, share_b_right 0.790, 0.768, status "unstated"); passing dates refuses until labels_date is given, as the new prose says.
- C15 FIXED: (0.004,-0.006) -> 1.0, (0.03,-0.005) -> 1.0, (-0.01,0.05) -> -1.0; DN and ordinary reflectance unchanged.
- C16 FIXED: assess --patch > map refused with a message; compare on an all-no-data map refused.
- C17 FIXED: 403 on the listing -> "could not be read (HTTP 403, GitHub's API rate limit) ... Retry later"; real 404 still "has no olmoearth_run_data/nope".
- C18 FIXED: determinism_check passes None with no window; fit_side with groups and no disagreement returns NaN; confusion_pairs refuses 0.6/1.4 and reads valid=0.4 as False; label note names class [3].
- E1 FIXED: certify --rule bonferroni prints the Bonferroni note, exit 0 (HEAD KeyError 'note').
- E2 FIXED: r1/r2 and CLI ov.npy: --per-class all-right and estimate all-wrong now run (HEAD refused k=300.00000000000006).
- E3 FIXED: never-predicted class producer's accuracy {0, 0, 0} under both designs (HEAD: None random, [0, 0.099] confidence). Exact given the class exists; sound.
- E12 FIXED: alpha 1e-17 -> "needs 230258509299404544" labels, exit 0; min_labels_to_certify(1e-12) = 2302585092993 (exact 2302585092995 by the linear approximation).
- E13 FIXED: a later run that certifies nothing removes rand_zone.npy.
- E4 FIXED: genuine random samples of 300 from N=4000 with 50/60/70% tied at the suspect end refused 0/50 each (HEAD 25/50, 50/50, 50/50); `certify` on rv's half.npy random sample now certifies (HEAD refused as "enriched set"). Cost (see MUST STATE): at 60% tied, the tool's own 10% review set (160 windows) now reads median 0.675 < thr 0.690 and passes as random; HEAD flagged it (0.692). At 20% and 40% tied the review sets are still flagged at 5%, 10%, 30%.
- E5 FIXED: (a) certify rand.csv --scores cls3.npy refused ("confidence at the sampled windows is not the one the CSV records"); (b) the sidecar now records nodata, certify without --nodata recomputes 1440 windows as drawn, a different --nodata refused; a HEAD-era sidecar without the key is refused on the population count (1600 vs 1440) instead of silently certifying; (c) confidence-design --per-class against bin.npy refused. Round trip: confidence at 9/6/5/4 significant digits and index "73.0" accepted; at 3 digits refused (see MUST STATE).
- E7 FIXED: random design now uses the exact hypergeometric interval (widened to hold k/n). Exact coverage: N=1000 n=999 min 0.979, n=995 min 0.975, N=300 n=299 min 0.977 (K on a step-7 grid; Wilson+fpc was 0.794/0.832/0.797); (400,12,300) 0.987, (1000,10,300) 0.990, (22598,68,300) 0.987. Confidence-design census now returns the point (0.075, 0.075, 0.075). Confidence and proportional designs bit-identical to HEAD on 300 random non-census samples. Decision sound.
- E8 FIXED: (A) row-level mismatch reported and printed ("on 20 row(s), first row 2 ..."); (B) reference_class 11 on a 3-class map refused; (C) "inf" in reference_class or index refused with a message.
- E9 FIXED: the tag names the reason ("[warning: near census]" on a 1500-of-1600 sample, "[warning: few errors]" elsewhere); JSON carries warning_codes.
- E6 FIXED: int vs float cluster ids now give the int-vs-int interval (-0.29, -0.10); HEAD (-0.314, -0.060). int vs str still matched.
- E10 FIXED: duplicate and negative indices refused by estimate_per_class under both designs; map_class -1 inside the confidence population, map_class of 2500 for a 2000-window sample, and non-integer map_class refused with named messages; certify_zone and estimate_from_indices refuse -1 and N+5 ("outside the valid map"); rule="bonferoni" refused before the budget check.
- E11 FIXED: 96 NaN-margin windows of 256: random and confidence designs now draw from 160 (HEAD 256, with 45 and 41 no-data windows sampled); NaN p1 at a valid window, p1 longer or shorter all refused. CLI round trip with --nodata -9999 (confidence design): sample -> estimate -> estimate --per-class without re-passing --nodata works (n_population 1400).
- E14 FIXED: expected_calibration_error([nan, nan], [1, 0]) -> (nan, []); ordinary input unchanged.
- E15 FIXED in Usage.md (table now names the exact hypergeometric interval for the random draw and Wilson on the effective sample size for the confidence design). Left stale: the estimate.py module docstring, lines 16-17, still says "Wilson with a finite-population correction for a random sample". No experiment calls estimate_error_rate (exp78 uses wilson_interval directly), so no recorded number moves.

## After this report (written by the author of the fixes, 2026-09-24)

The verifier stalled twice before its record check; everything above is its own. Acted on: the review-set check now
decides on the MEAN suspicion percentile with ties ranked in a random order (at 60% tied the review set is refused
again at 5%, 10% and 30%, and 100 random samples pass; tests/test_estimate.py); the CSV confidence tolerance follows
the digits written (three digits accepted, another map still refused; tests/test_cli.py); the Usage sentence on tied
windows and the module docstring of estimate.py are corrected. The record sentences of the exp81 section were
checked against the artifacts by the ledger's recomputing checks (tests/test_claims.py).
