# What a map user is owed per class (exp81 preregistration)

Written before the run, on 23 September 2026. Runs as `exp/exp81_per_class.py`; the numbers land in the ledger
like every other experiment.

## The gap

`estimate` answers "how wrong is this map" with one number and an interval. The map-accuracy literature has
settled, over a decade, on what a producer owes instead: per class, the **user's accuracy** (of the windows the
map calls *c*, the share that are *c*), the **producer's accuracy** (of the windows that are *c*, the share the
map found) and the **error-adjusted area** of *c*, each with a standard error, all from one probability sample
(Olofsson et al. 2014; Stehman and Foody 2019; Stehman and Wagner 2023; the CEOS LPV land-cover protocol, 2025).
A flood user acts on "water is missed 19% of the time and the flooded area is 8.1 ± 0.9 km², not 9.4", not on
"the map is 8% wrong". Nothing in the package computes a per-class quantity, and nothing in the record grades
one. This run adds both, from the sample `sample_for_estimation` already draws.

Stehman and Wagner (2023) add a warning this run tests: a sample allocated to estimate one target well, here the
overall error rate under the confidence design, serves the per-class targets worse.

## The estimands

For a task with N labelled windows, map class `m_i` and reference class `r_i`, both in `{0..C−1}`:

- user's accuracy `UA_c = #{m = c, r = c} / #{m = c}`
- producer's accuracy `PA_c = #{m = c, r = c} / #{r = c}`
- error-adjusted share `A_c = #{r = c} / N`, against the map's own share `#{m = c} / N`
- overall accuracy `#{m = r} / N`, already exp78's estimand in its complement.

Truth is read from the whole population; labels are revealed only for sampled windows.

## Estimators, one implementation (`oe_inferencex.estimate.estimate_per_class`)

Under a **random** sample of B windows: `UA_c` is the Wilson interval with finite-population correction on the
labelled windows the map calls *c*, because those are a simple random sample of that class's windows; `A_c` is
the post-stratified estimator by map class (Olofsson et al. 2014, eq. 4 and 5); `PA_c` follows their eq. 7. Under
the **confidence** design (margin quintiles, Neyman for the overall rate) a class cuts across strata, so every
quantity is a ratio of Horvitz–Thompson totals with the linearised variance (Cochran 1977, §6.11), and `A_c` a
Horvitz–Thompson total. Wald intervals at 1.96 for the ratio forms, as the field uses; an interval on fewer than
30 labelled windows is reported with a warning, and the Monte Carlo grades coverage only where that condition
holds, since that is when the package reports it without one. Tile samples are refused by the function: the
per-class cluster form is a later run. All of it is checked by enumeration and against an independent transcription
of Olofsson's equations in `tests/test_per_class.py` before this page was written.

## Design

Tasks: the 17 classification tasks now, from exp78's export, which carries the reference class per unit; the 7
segmentation tasks from exp79's seed-0 export as soon as it lands (exp78's segmentation export wrote no reference
class; the disclosed facts for those tasks are appended below before their run). Designs: random and confidence.
Budgets B ∈ {300, 1000}, headline 300; tasks with N ≤ B skip that budget. R = 2,000 draws, seed 0 to 1,999.

## Disclosed population facts (classification), what they fix in advance

Computed from labels alone before this page was written. `disc/SE` is the population gap between the map's
class share and the true share, in units of the expected standard error of a share estimate at B = 300 under a
random draw with the finite-population correction; it decides P3's cells before any estimator runs.

| task | N | C | smallest map share | classes expected under 30 labels at B = 300 | classes with disc ≥ 3 SE | classes with disc < 1 SE |
|---|---|---|---|---|---|---|
| EuroSAT | 1000 | 10 | 0.097 | 3 | 0 | 10 |
| Brick Kiln | 999 | 2 | 0.330 | 0 | 0 | 2 |
| ForestNet | 993 | 12 | 0.015 | 8 | 0 | 4 |
| So2Sat | 986 | 17 | 0.024 | 17 | 1 | 7 |
| AWF Landsat / S1 / S2 | 200 | 9 | 0.005–0.010 | (N < 300: no headline cell) | — | — |
| Nandi Landsat / S1 / S2 | 825 | 6 | 0.105–0.124 | 0 | 1 / 2 / 0 | 0 / 1 / 2 |
| CropHarvest Togo, three arms | 306 | 2 | 0.39–0.47 | 0 | 2 each (a 98% census makes any gap many SEs) | 0 |
| CropHarvest China, three arms | 4397 | 2 | 0.16–0.24 | 0 | 2 each | 0 |
| BreizhCrops | 122614 | 9 | 0.000 | 5 | 1 | 5 |

So at 300 labels the 17-class So2Sat cannot report a single class without a warning, ForestNet reports four of
twelve and BreizhCrops four of nine; that is the CEOS point that the budget scales with the number of classes,
and it is recorded as such rather than hidden by averaging.

## Predictions

**P1, every interval the package reports without a warning is honest.** On every (task, design, budget, class,
quantity) cell where the class holds at least 30 labelled windows on at least half the draws, the Monte Carlo
coverage over those draws is at least **0.93** and the mean estimate is within **2%** of the truth (bias ratio in
[0.98, 1.02]). *Why 0.93:* exp78's bar, four standard deviations below nominal at R = 2,000. *Why the bias clause:*
the ratio estimators are consistent, not unbiased; at 30 labels a ratio's bias is of order 1/n, so 2% is the
tolerance the estimator earns, and a larger bias is a defect. *What makes it fail:* the linearised variance
under-covering for a class concentrated in one or two strata, where a stratum's within-class count is small even
when the class total passes 30; Wald on a producer's accuracy near 1. **A P1 failure is graded as a defect of the
named cell, and the package warns for that case before the result is recorded.**

**P2, the design that serves the overall rate serves the classes worse.** At B = 300, the median over classes of
the ratio of the confidence design's user's-accuracy half-width to the random design's exceeds **1** on at least
**half** of the tasks with both designs. *Why:* Neyman for the overall rate sends labels to the low-confidence
strata; a class whose windows are confident is then under-labelled and its user's accuracy widens. This is
Stehman and Wagner's warning, and the bar is the weakest one that would still contradict it if it failed. *What
makes it fail:* classes spread evenly across the margin strata, in which case the confidence design costs the
classes nothing and the tool can keep it as the default for both purposes.

**P3, the adjusted share moves exactly where the population says it should.** At B = 300 under the random design,
per class: where the population gap between the map's share and the true share is at least **3** expected
standard errors, the 95% interval of the error-adjusted share excludes the map's share on at least **80%** of
draws; where the gap is below **1** standard error, on at most **20%**. *Why:* a gap of 3 SE is detected with
power about 0.85 by a two-sided 95% interval, and a gap under 1 SE with at most about 0.17; the bars sit inside
those values. *What makes it fail:* a post-stratified variance that is too small (the FPC applied twice, or a map
class with no labels dropping its weight silently), which shows as exclusion far above 20% on the null cells.

**Descriptive.** Per task: the number of classes reported without a warning at 300 and 1,000 labels; per-class
half-widths; the per-class bias ratio; the overall-accuracy coverage beside exp78's.

## Amendment, 23 September 2026, after the first run on the classification tasks and before the rerun

The first run graded the Wald intervals this page promised. **P1 failed on 77 of 324 cells**, with coverage as
low as 0.30, and the failures have three causes, two of them this page's and one the field's:

1. **Wald collapses to a point.** Where no sampled window of a class is wrong, the linearised or post-stratified
   variance is zero and the interval is [1, 1]; a class that is 98% right is then "covered" on no draw. 34 of the
   77 cells sit at a truth beyond 0.95 or below 0.05. The package already handles the same collapse for the
   overall stratified rate with Wilson on the effective sample size (Korn and Graubard 1998); the per-class
   estimators now use the same form by default, with the class's labelled count as the fallback size when the
   estimated variance is zero. The Wald form stays available and **both are rerun and recorded**: the Wald numbers
   are the measurement of the field's convention at these budgets, the Wilson numbers grade what ships.
2. **Wilson's discreteness on EuroSAT.** Six random-design user's-accuracy cells at 0.63–0.69: each class holds
   100 windows with one error, and 30 labels either see it (30% of draws, interval up to 0.988, truth 0.990 not
   covered) or not. The exact coverage Wilson can achieve at (N = 100, K = 1, n = 30) is 0.70 by enumeration, and
   the Monte Carlo sits there. exp79's P5 rule applies: such a cell passes when its coverage is within 0.02 of the
   exact achievable value; the cells are listed in the record as discreteness, not as a defect.
3. **A grading artefact of this page.** Bias was measured over the draws where the class held at least 30
   labels; for a class expected to hold exactly 30, that conditions on being over-sampled and manufactures an
   upward bias of 3–11% with coverage above nominal (12 cells, all reference shares of classes near the 30-label
   line). Bias is now measured over every draw, unconditionally; coverage stays conditional, since that is when
   the interval is reported. The conditional ratio is kept in the artifact beside it.

P2 and P3 are graded on the amended interval form without change of bar. The first run's P2 (4 of 14 tasks
wider) and P3 (3 null cells at exclusion 0.25–0.29) are kept in the record as the first reading.

## Second amendment, 23 September 2026, from the independent audit of the rerun, before the third run

The audit (its own transcription of the estimators, its own Monte Carlo with its own seeds) confirmed every
estimator and every failing cell, and found two things that would have made the record wrong:

1. **The design-comparison grader read the Wald cells.** `grade_p2` kept the last cell per design, which was the
   Wald one; its "4 of 14" was the first run's number to every digit. On the shipped form it is 3 of 14. Fixed by
   filtering on the interval form; the verdict (fails) does not change, the numbers do.
2. **The producer's accuracy under a random draw lacked the finite-population correction** that the user's
   accuracy and the shares carry, so its interval was over-wide (median coverage 0.983 across the 54 graded cells;
   eight times too wide on a near-census draw) and "P1 holds on the PA cells" would have been a statement about
   slack, not honesty. This page said "eq. 7", and Olofsson's eq. 7 has no correction because their sampling
   fraction is negligible; at 300 of 1,000 windows it is not. Each map-class term now carries `(1 − n_i / N_i)`,
   as its siblings do, and the random-design cells are rerun.

Three grading rules are tightened, each stated with its reason: the share's coverage is graded on every draw,
since conditioning on the reference-class count conditions on the estimate itself (the amendment above removed
that conditioning from bias and missed coverage); a discreteness cell passes when it is within the larger of
0.02 and three Monte Carlo standard errors of the exact achievable coverage at its eligible count (the two
EuroSAT cells still failing sat at 1.75 and 2.8 standard errors, and ten independent blocks put 3 of 50 outside
the fixed 0.02); and P3's "expected standard error" is the post-stratified estimator's own at the expected counts,
not the simple-random one this page used, which is 1.4 to 1.9 times larger and put three cells at 1.0–1.4 true
standard errors where a 95% interval excludes 17–29% of the time, exactly as observed. The simple-random ratio is
kept beside it in the artifact.

**Third run, one selection rule added after it.** Grading the share on every draw for every class swept in
classes the map hardly ever predicts (BreizhCrops never predicts three of nine), whose share estimate rests on a
handful of windows and whose bias ratio near zero is meaningless (0.72 to 1.54 with coverage 0.97–0.99). The
share cells graded are those the package reports without a warning, an expected labelled count of at least 30 in
the reference class; their coverage is over every draw. That leaves 328 cells: 13 below the bar, six
discreteness cells passing by the exact rule.

Two mechanisms behind the nine confidence-design cells at 0.87–0.93, both anticipated on this page and neither a
formula error, are now warnings in the package: a class nearly all of whose windows are labelled (the estimate
takes 22 distinct values on Togo and a normal interval covers 0.884 even with the exact variance), and a class
whose windows sit in confidence strata the overall-rate allocation samples thinly, so its rare errors are not
drawn on 35–70% of draws (Brick Kiln, Nandi Landsat: coverage 0.80–0.90 on those draws, 0.98 on the rest).

## Independent check before anything is recorded

`tests/test_per_class.py`: Horvitz–Thompson totals and their variance unbiased over every stratified sample of a
12-window population; the random-design estimators equal to an independent transcription of Olofsson's equations
4–7 and to the package's Wilson interval; both designs' variance estimators within 15% of the Monte Carlo variance
on a 4,000-window synthetic map with margin-dependent classes, estimates unbiased to three standard errors. After
the run, an adversarial read of the implementation on the real files by a separate agent.

## What would invalidate the run

- Truth read from the sample.
- Coverage reported over all draws, including those where the package would have warned, so that a small class
  hides a defect or manufactures one.
- Treating a tile sample as random: refused by the function.

## Cost

No cluster for the classification tasks; the segmentation tasks wait for exp79's export. Minutes per task on
numpy. Ships as `oe-inferencex estimate --per-class` after P1 holds.

## Fourth amendment, 23 September 2026: the finite-population form, written before the rerun

The generality run on AnySat's export left one shipped-form cell far below the bar: EuroSAT class 9 (100 map
windows, one wrong) under the random design at 300 labels covered 0.621, missing the exact-coverage tolerance by
0.004. Brute-forcing that cell by hand found the cause in the package, not in the draw. `wilson_interval`'s
finite-population correction multiplied only the `p(1 − p)/n` term inside the root and left Wilson's centre and
its `z²/(4n²)` term at `n`; with one error seen in 30 of 100 the interval was [0.0121, 0.161] against a truth of
0.010, so a class holding one error in a hundred windows was excluded whenever its error was sampled. Its exact
hypergeometric coverage is 0.70 at (100, 1, 30) and 0.50 at (300, 1, 150). The first run's six EuroSAT cells at
0.63–0.69, which the record read as "Wilson's discreteness, not a defect", were this form. The interval is now
the score-test inversion at the effective size `n (N − 1)/(N − n)` (Korn and Graubard 1998, the form the
package's stratified interval already used), which covers 1.00 at (100, 1, 30), reaches 0 and 1 at the edges, and
differs from the old form by at most 1.7e-4 at 300 of 22,598 (`tests/test_estimate.py`). (A first draft of this
paragraph claimed the score form's exact coverage never dips below 0.92 on a grid with N ≤ 1000 and K ≥ 2; the
independent audit found 0.830 at (1000, 3, 60), and the sixth amendment below records what followed.)

**Expectation for the rerun, stated before it.** The six EuroSAT discreteness cells cover at or above 0.93 and
the "discreteness" exemption is not needed on any cell; every other shipped-form cell moves by less than 0.01,
because the correction only matters where a class's labelled count is a large fraction of its map count; the
Wald counts do not move (Wald has no such term). Base's classification tasks are rerun with `--merge` after the
segmentation run finishes; the segmentation cells, whose map classes hold thousands of windows against tens of
labels, are left as run (the two forms agree there to 1e-4). The four other encoders are rerun in full.

## Fifth amendment, 23 September 2026: P3 grades the classes the package reports, written from the segmentation run

The segmentation run (the seven tasks from exp79's Base export, merged into the summary) graded P3 on every class
in the exclusion table, and one cell of m-cashew-plant read 6,016 standard errors with an exclusion rate of 0.0:
class 0, which the map predicts on 0.6% of windows and the reference never holds, so its expected reference count
at 300 labels is 1.8, its expected post-stratified variance is zero, and its interval (a Wilson fallback on two
labels) cannot exclude anything. The package warns for such a class rather than reporting it. P3 is therefore
restricted, as the share's P1 cells were by the second amendment, to classes with an expected reference count of
at least 30 labels; the classification cells recorded on 23 September are regraded under the same rule and the
counts restated where they move. Ten further segmentation cells fail P3's null clause (discrepancies of 0.5 to
1.0 standard errors excluded on 0.21 to 0.33 of draws against a bar of 0.20); whether they survive the
restriction is graded, not assumed.

## Sixth amendment, 23 September 2026: an exact interval for the random-design user's accuracy, and no exemption

Written after the four-encoder rerun under the score form and before any run under the form below. An
independent audit of the fourth amendment (`exp/out/audit_wilson_fpc_2026-09-23.md`) found three things, each
verified here by enumeration.

- **The exact-coverage exemption is a tautology.** `exact_ua_coverage` computes the exact coverage of the same
  interval the Monte Carlo grades, so a cell whose shortfall is the interval's own always passes it; it exempted
  the variance-only form's defect (0.70 at (100, 1, 30)) and then the score form's near-census shortfall (the
  three Togo cells at 0.925–0.929). It is removed. The exact coverage stays in the failing-cell report as a
  diagnostic, so a reader can see whether the Monte Carlo agrees with the enumeration, but it passes nothing.
- **The fourth amendment's expectation failed.** "Every other shipped-form cell moves by less than 0.01" did not
  hold: on Clay Large 36 per-class cells moved by more than 0.01 (the largest 0.073), among them the six Togo
  random-design user's-accuracy cells, from 1.000 to 0.927–0.976, because Togo labels 300 of 306 windows and the
  correction matters most there. The three exemptions the four-encoder rerun reported were cells the score form
  created, not pre-existing discreteness.
- **No normal-theory form meets a per-cell bar on a small finite class.** On a grid of (N_c, K_c, n_c) with
  N_c ≤ 1000 the score form falls below 0.93 on 40 of 164 cells, the worst at 0.80; the equal-tailed exact
  hypergeometric interval (the tail inversion that the trusted zone's tests already use) never falls below 0.951,
  and on the near-census Togo classes it is no wider than the score form.

**The change.** Under a random sample, the labelled windows the map calls class c are a simple random sample of
that class's N_c windows, so the user's accuracy has an exact interval: every count K of correct windows whose
two tails at the observed count both exceed 2.5%, divided by N_c (`estimate.hypergeom_interval`). The random-design
user's accuracy uses it; its coverage is at least 95% by construction on every class. Nothing else changes: the
producer's accuracy, the shares and every confidence-design interval keep Wilson on the effective sample size,
because none of them is a single hypergeometric count. The Wald option does not apply to this interval.

**Predictions for the rerun, stated before it.** (a) No random-design user's-accuracy cell of any of the five
encoders covers below 0.93, on the classification tasks or on Base's segmentation tasks. (b) The median
half-width of those cells grows by at most 10% against the score form, and on the near-census Togo classes by
at most 2%. (c) Every other cell is identical to the score-form run to the digit, since its code path is
untouched; this is checked by comparing the artifacts, not assumed. (d) The P1 failures that remain are
confidence-design and producer's-accuracy cells, each with the mechanism the record already names (near-census
labelling under the confidence design, thinly sampled strata) or a new one stated.

**What is rerun.** The random-design cells of every task under all five encoders (`--designs random --merge`,
which replaces only those cells in each task's row); the confidence-design cells are carried over unchanged.

**Result, written after the rerun and the second audit** (`exp/out/audit_exact_interval_2026-09-23.md`). (a) held:
no graded random-design user's-accuracy cell covers below 0.943 on any of the five encoders (0.947 on Base's
segmentation tasks, 0.950 on its classification tasks). (b) held as a median, half-widths 5% wider on every
encoder, and on the Togo classes (at most 1% wider), but not cell by cell: ten cells widened by 10 to 15%, seven
of them one-error EuroSAT classes the score form already covered. (c) held: the audit diffed every value and
found only random-design user's-accuracy entries changed. (d) failed on four random-design share cells on
EuroSAT that it did not name. The rerun covered the four other encoders' classification tasks only, as their
first runs did. The audit also found the mechanism this page and the record gave for the Brick Kiln and Nandi
Landsat cells wrong: none of their strata is sampled at under half the overall rate, so the thin-strata warning
never fired on them, and their shortfall is the rare-error mechanism the EuroSAT and MADOS cells share (a class
whose accuracy rests on a handful of errors that a draw misses or catches at a large weight). The package does
not warn for that case yet; the warning's text no longer cites these cells.

## Seventh amendment, 23 September 2026: a warning for the rare-error case, written before it is measured

The second audit named the mechanism behind nine of the fourteen P1 shortfalls: a class whose accuracy rests on a
handful of errors, so a draw that misses them, or catches one at a large weight, moves the estimate further than a
normal-theory interval reaches. The package now warns (`warning_codes` "few errors") when the producer's
accuracy, the share, or under the confidence design the user's accuracy rests on 1 to 4 sampled errors of the
kind it counts (`estimate.RARE_ERRORS`). Zero sampled errors is not warned: the interval then falls back to the
labelled count and errs wide (coverage 0.996 to 1.0 in the review's check).

**Predictions, stated before the run** (`exp/exp81_rare_error_warning.py`, 2,000 draws per cell, Base's export).
(a) On the nine rare-error cells (EuroSAT's five, MADOS class 7, Brick Kiln's two, Nandi Landsat's one) the warning
fires on at least **80%** of the draws whose interval misses the truth. (b) On those cells, coverage among the
unwarned draws is at least **0.93**. (c) Across every graded (draw, class, quantity) of the seventeen
classification tasks at 300 labels under both designs, the warning fires on at most **20%**, so it marks a case
rather than every class of a clean map. *What makes each fail:* (a) a shortfall that happens on draws with five or
more errors, where the mechanism named is not the one at work; (b) the same; (c) a threshold so loose it warns
everywhere, which would make the warning noise.
