# Audit: the exact random-design user's-accuracy interval, the partial rerun, and the exp81 draft (2026-09-23)

I used my own code throughout (`scratchpad/aei/t1*.py`, `t2.py`, `t3*.py`, plus inline scripts). The package was called only to compare against. My references were: an exact rational scan over every K (`math.comb`, a = 1/40), a scipy scan over every K, and my own bisection on scipy tails with exact-integer checks at the boundaries. My grader re-implements P1, the Wald block, P2 and P3 from the per-cell numbers. Tests: `test_estimate`, `test_per_class` and `test_compare` give 74 passed; `test_cli -k "date or compare"` gives 13 passed. All five claim `check` expressions evaluate True. The problems below are ones the checks do not test.

## BLOCKING

1. **"The package warns for the near-census and thinly sampled cases" is false for the thinly sampled cells.** The P1 preregistration says "the package warns for that case before the result is recorded". I ran `estimate_per_class` on all 2,000 draws. The thin-strata warning fired on 0 of 2,000 graded draws for Brick Kiln confidence B=300 class 0 PA and class 1 UA, and on 0 of 1,980 for Nandi Landsat class 1 UA. No warning of any kind fired on MADOS class 7 PA (0 of 2,000).
   - Why it never fires: the warning needs a stratum sampled below half the overall rate. On Brick Kiln the stratum rates are [0.58, 0.235, 0.226, 0.23, 0.23] against an overall rate of 0.30, so the threshold is 0.15. On Nandi Landsat the lowest rates, 0.182, equal the threshold of 0.182 but do not fall below it. No stratum qualifies.
   - What actually happens: 11 of Brick Kiln's 15 class errors sit in the most heavily sampled stratum. The other four sit in strata sampled at 0.77× the overall rate.
   - EuroSAT's five cells get only the small-count warning, and on a minority of graded draws: 11/179, 163/300, 5/140, 165/300 and 2/167 in a 300-draw check.
   - What to do: either change the sentence (the warning covers near-census only), or fix the warning and regrade. As the draft stands, 4 of the 14 failing Base cells never warn, and the 5 EuroSAT cells warn on only 1% to 55% of graded draws, against the preregistration's rule.

## MUST STATE

2. **The Base artifact's `units` field is wrong for 17 of 24 tasks.** It now reads `exp/out/exp79_units/olmoearth_base`, the last invocation's directory. I matched each cell's per-class `n_map_true` and `n_wrong` against both exports. The two exports differ in `dec` on 10 of 17 classification tasks and in `margin` and `p1` on all 17. On the 9 graded tasks where the counts can tell the exports apart, every cell matches exp78's export and not exp79's, the rerun random cells included. On the rest, y and dec are identical, so the random cells are the same whichever export was read. So the numbers are right and the metadata is wrong. `config.seconds` (557) is also the last invocation's time only. The draft's own sentence ("17 classification tasks ... from exp78's export") is correct.
3. **The overall-accuracy sentence misattributes a cell.** The draft says "lowest 0.911 on Brick Kiln and 0.913 on MADOS at 300 random labels". Brick Kiln's 0.9115 is a **confidence-design, Wald-form** cell. The 8 of 128 count mixes both forms. The form the tool prints (Wilson) is below 0.93 on 3 of 64 cells: MADOS random 0.913, PASTIS S2 random 0.9245, Togo S2+S1 confidence 0.927. The comparison with 0.933 (exp78's SRS coverage) holds for MADOS only.
4. **Copernicus-FM has 15 failing cells, not "15 cells below 0.93".** One of the 15 fails only the bias clause: BreizhCrops random B=1000 class 0 PA, coverage 0.946, bias 1.023. Both the "On other encoders" paragraph and the claim `per-class-intervals-hold-on-other-encoders` say "11, 15, 8 and 12 cells below 0.93". Say "fail P1 (14 of Copernicus-FM's below 0.93, one on bias)". Clay's 0.858 cell fails both clauses (bias 1.055). Base's 14 and its Wald 107 are all coverage failures.
5. **"The error-adjusted share fails no cell of the 128 classes" reads as P1, and there it is false.** Share cells fail P1 on every other encoder: Clay 2, Copernicus-FM 4, AnySat 1, CROMA 3. The claim's wording ("exclusion rates") is right; the comparisons.md sentence should say P3.
6. **Prediction (d) failed on 4 random-design share cells on three encoders, not two.** They are Base EuroSAT classes 3 and 6 (0.927, 0.9075), AnySat EuroSAT class 0 (0.921) and CROMA EuroSAT class 0 (0.929). The draft names only Base's two.
7. **The EuroSAT cells are listed but their mechanism is not named, although the text says every cell has one.** I simulated 2,000 draws.
   - Class 6 share: 2 omitted and 4 committed windows. Coverage is 1.00 on every draw except when omission and commission errors are drawn unevenly: 0.42 with one omission and no commission (235 draws), 0.00 with two omissions and none committed (45 draws). Total 0.9075.
   - Class 3 share: coverage 0.09 and 0.00 when commissions are drawn without omissions (150 draws).
   - The mechanism: the share's variance is built from a handful of rare off-diagonal windows, so the estimate jumps by a whole window-weight that the interval cannot reach. This is the same family as MADOS. "EuroSAT's small classes" is also loose: every EuroSAT class holds 97 to 102 of 1,000 windows.
8. **The ten widened cells did not "now reach the truth".** The width ratio is measured against the score form. The ten cells above 1.10 (Base 8, AnySat 1, CROMA 1; seven of them one-error EuroSAT classes) already covered 1.000 under the score form. What they miss is the variance-only form. The widening bought no coverage on them.
9. **P2 compares two interval forms, and every change in its count is the form change.** The confidence-design cells are byte-identical and the random draws use the same seeds.
   - Base went from 5/21 (score form) to 2/21 through three flips, all downward: ForestNet 1.008→0.961, Nandi Landsat 1.029→0.976, Nandi S1 1.001→0.939.
   - The other four encoders show 3→3, 3→3, 3→3 and 3→2 on the tallies, but hide 2, 2, 4 and 1 offsetting flips. The Togo tasks flip up because the exact interval is narrower near census; the others flip down.
   - Per-task factor, exact over score form: median 0.941–0.953 on every encoder (range 0.889–1.42).
   - The two forms have different realised coverage. The graded random UA covers 0.969–0.972 on average against the confidence design's 0.944–0.950. So the ratio is biased about 5% in the tool's favour. Brick Kiln is 0.726 under the score form and 0.6845 now; MADOS is 0.700 and 0.692.
   - The verdict is robust. Under the like-for-like score form, the median-ranked task sits at 0.88–0.96 on all five encoders, still far below 1. P2 as graded is still a fair test of its verdict but not of its counts or extremes. State that the ratio now sets an exact, conservative interval against a nominal one. The claim note says "about 5% wider" but not that this biases the ratio.
10. **The brief's list has one wrong number: CROMA's P2 is 2 of 12, not 3 of 12.** The draft and the claim already say 2 of 12. Nandi S1 went 1.006→0.950.
11. **Findings.md:350 says "none below 0.88", but the lowest cell is 0.879** (Togo S2 confidence class 0 UA). Say "none below 0.879" or "about 0.88".
12. **"Producer's accuracy ... 0.953 after" does not reproduce from the current artifact.** The random-design graded PA median is 0.9517 over the 54 classification cells and 0.9503 over all 106 cells.
13. **"The six EuroSAT cells now cover 0.95 to 1.00" is true but loose.** The six cells failing under the variance-only form were classes 0, 4, 6, 7, 8 and 9 (0.63–0.90). They now cover 1.000 on five and 0.9915 on class 6. "0.99 to 1.00" is exact.
14. **Prediction (a) holds on graded cells only.** Ungraded random-UA cells with eligible share below 0.5 go to 0.875 (Base BreizhCrops class 0, 8 draws) and 0.000 (AnySat So2Sat class 8, 1 draw: n_c=31 of N_c=60, exact coverage 0.964 there). Say "graded".
15. **`hypergeom_interval` API (minor).**
    - Non-integer counts are silently truncated: `(5.5, 10, 100)` and `(np.float64(5.9), ...)` both return k=5's interval, and `True` is read as 1.
    - `conf` is not validated: `conf=-0.5` returns an inverted interval (0.35, 0.26), and `conf=1.5` acts as conf=1.
    - `estimate_per_class` always passes ints and 0.95, so no recorded number is affected.
16. **compare dates (commit 20196d1): real but minor defects.**
    - `_period` truncates strings to 10 characters, so `'2020-01-011'` is read as 2020-01-01 and `'2020-01-111'` as 2020-01-11, silently, and `'2020-01-01garbage'` is accepted.
    - A coarse-unit `datetime64` becomes its first day, not the period it denotes: `datetime64('2020-06')` against `'2020-06-15'` gives "different_time, 14 days", and `datetime64('2020')` against `'2020-12-31'` gives 365 days.
    - Time zones: a tz-aware datetime or offset string keeps its local calendar date. The same instant at 23:00-05:00 and 04:00Z reads as "different_time, 1 day", and with labels the comparison is refused unless `labels_date` is given. The docstring says the time part is ignored; the effect of the offset should be said too.

## FINE

17. **`hypergeom_interval` equals my references.** 420 (k, n, N) cases, N from 2 to 1e6, including k=0, k=n, n=1 and n=N−1: 133 exact rational scans, 182 scipy scans and 105 bisections all agree.
    - Two apparent mismatches, (0,1,1e6) and (1,1,1e6), are exact ties (tail exactly 1/40). There the package matches exact arithmetic and scipy does not; the maximum discrepancy is 1e-6, one K.
    - A further 60 exact rational scans at N 120–400 give 0 mismatches.
    - Exact-integer checks at N 1e5–1e6 find 6 "wrong" boundaries, all exact ties at (1,1,3e5) and (0,1,3e5) that the package includes. They make the interval wider, never narrower. The cause is that `a` = (1−0.95)/2 = 0.025000000000000022 in floating point.
18. **The bisection's monotonicity holds.** The tail inversion's premise is a theorem: X|K+1 stochastically dominates X|K by at most one. Numerically, `_hyper_tail` has non-monotone steps of at most 2.2e-11 in K, but the predicate "tail > a" was never non-monotone over 60 full K scans.
    - `_hyper_tail` is within 6.6e-11 relative of scipy where the tail lies in (1e-4, 0.5).
    - The bisection brackets are right: K_lo is in [k, N−n+k], where the upper tail equals 1 at the top. For a < 0.5 the set of K kept is never empty.
19. **Coverage is at least 95% by construction.** The standard argument holds: excluded from below exactly when P(X ≥ k | K) ≤ a. Exact coverage on 11,086 (N, K, n) cells has minimum 0.95005 at (306, 134, 153), none below 0.95, median 0.9915.
    - On the 10,898 cells with N ≤ 1000, the minimum is 0.95005.
    - On the 188 cells with N from 1e4 to 1e6, the minimum is 0.9508.
    - On the prior audit's grid (168 cells), the minimum is 0.9514.
    - Coverage conditional on n_c ≥ 30 inherits the bound, because the random design is an SRS, so each class subsample is an SRS given n_c.
20. **Edge cases and `lru_cache`.**
    - n=N gives the point k/n, n=0 gives (0, 1), and N=1 is fine.
    - numpy ints, Python ints and integer-valued floats hash equal and share one cache entry with an identical tuple of Python floats. Keys that compare equal have equal `int()`, so the cache cannot return a stale or wrong-typed result. Exceptions are not cached.
21. **The partial rerun is clean.** A key-by-key diff of all five artifacts against the score-form snapshots finds changes only in the following:
    - random-design `user_accuracy.coverage` (242 / 128 / 106 / 130 / 110 entries) and `median_half_width` (278 / 136 / 122 / 138 / 124);
    - random-cell `seconds`, the `carried_over_designs` rows (24 / 17 / 14 / 17 / 14), `config.seconds`, `prereg`, and Base's `units` (item 2).
    - Every confidence-design value, every random PA and share value, `share_excludes_map`, and every overall coverage are unchanged. So are the UA estimate, bias and eligible share, the cell order and the task sets.
    - The random Wald cells' UA equals the Wilson cells' UA on all entries.
22. **I regenerated 38 random-UA cells independently** with my own sampler replica and my own scipy-based exact interval (Base EuroSAT, Togo S2, China S1 B=1000, MADOS, PASTIS S1 B=1000, Clay Togo S2+S1, AnySat EuroSAT). The maximum difference against the artifact is 0.0 in coverage and 0.0 in median half-width.
23. **The Monte Carlo agrees with the enumeration.** I enumerated the exact coverage of 244 graded random-UA cells with N_c ≤ 20,000, conditional on n_c ≥ 30, across all five encoders. The minimum exact coverage is 0.9541. The Monte Carlo is below 0.95 on 2 cells, and |z| is at most 3.3 over 244 cells. That cell is CROMA China S1 B=300 class 0: 0.952 against 0.9654. The draws are shared across encoders, so the cells are not independent, and this spread is what noise looks like.
24. **The numbers to be recorded reproduce with my grader, except item 10 (CROMA P2).**
    - **Base P1:** 628 cells, 14 failing, lowest 0.879. 13 classification failures and 1 segmentation (MADOS confidence B=1000 class 7 PA, 0.8995, truth 0.99983).
    - **Base random UA:** no graded random-UA cell below 0.93. The lowest is 0.947 on segmentation (PASTIS S1 B=1000 class 0) and 0.950 on classification (China S1 B=1000 class 1).
    - **Base Wald block:** 522 cells, 107 failing, lowest 0.019 (MADOS random B=300 class 7 PA), 44 of them on segmentation.
    - **Base P2 and P3:** P2 is 2/21 (EuroSAT 1.0132, So2Sat 1.1291; lowest Brick Kiln 0.6845, then MADOS 0.6915). P3 is 42 cells, 0 failing (≥3 SE: minimum 0.832; <1 SE: maximum 0.1335). Ungraded, 12 of 116 fail, all with 17.6 expected labels or fewer, as the draft says.
    - **The four other encoders:**

      | Encoder | Lowest random-UA coverage | P1 failing | Wald failing / cells | P3 cells |
      |---|---|---|---|---|
      | Clay Large | 0.952 | 11 | 39/268 | 34 |
      | Copernicus-FM | 0.949 | 15 | 62/229 | 31 |
      | AnySat | 0.943 | 8 | 30/269 | 33 |
      | CROMA Base | 0.952 | 12 | 53/228 | 30 |

      Lowest shipped-form failure 0.858; Wald worst 0.152 / 0.316 / 0.333 / 0.293. P3 fails no cell; the CROMA So2Sat class 7 cell at 0.755 is ungraded.
    - **Half-width ratio, new over old:** median 1.057 / 1.048 / 1.049 / 1.052 / 1.050. Above 1.10 on 8 of Base's 106 cells and 2 of 202 on the other four. On the Togo cells the maximum is 1.010 (CROMA; the others are 0.91–1.00).
    - **Overall accuracy:** 8 of 128 Base cells below 0.93, lowest 0.9115 (Brick Kiln confidence Wald) and 0.913 (MADOS random Wilson). The overall coverage is unchanged by the rerun.
25. **MADOS class 7 PA, confidence design, B=1000.**
    - Mechanism: exactly one of the 5,850 reference class 7 windows is missed (map calls it class 5). It sits in the least-confident stratum, which is sampled at 0.095, about twice the 0.044 overall rate. It is drawn on 201 of 2,000 draws. When it is drawn, its HT weight of about 10.5 pulls the estimate to about 0.9982, and the Wilson-eff interval's upper bound (about 0.99967) stops short of 0.99983, so coverage is 0/201. Otherwise it is 1799/1799. Coverage = 1 − P(drawn) = 0.8995 exactly.
    - No package warning fires on this cell (0 of 2,000 draws).
26. **The draft otherwise matches the artifacts.**
    - The failure breakdown 5 / 3 / 5 / 1 is correct.
    - The P3 detail is correct: two So2Sat cells at 0.775 and 0.23; nine PASTIS and MADOS cells at 0.208–0.332; the cashew cell at 6,016 SE.
    - The P2 history is correct: 3/14 under the variance-only form (HEAD) and 5/14 under the score form.
    - "Every other cell identical to the digit" holds.
    - AnySat EuroSAT class 9 now covers 1.00.
    - The date handling of leap days, invalid dates ('2020-02-30', '2021-02-29'), inclusive touching intervals (overlapping), adjacent intervals (1 day apart), and a date against the same one-day interval (same_time) is correct.
