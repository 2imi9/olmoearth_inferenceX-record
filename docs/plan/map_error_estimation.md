# How wrong is this map? (exp78 preregistration)

Written before the run. Runs as `exp/exp78_error_rate_estimation.py`; the numbers land in the ledger like every
other experiment.

## The gap

The tool ranks the windows of a map by the model's own confidence, so a reviewer knows where to look first. The
question a map user actually asks first is a different one, and [Usage](https://olmoearth-inferencex.readthedocs.io/en/latest/Usage/) says plainly that the package
refuses it: *"What the package never says, for any map: how wrong the map is."* That is **estimation**, not
ranking, and no experiment here has attempted it.

It is worth attempting because the ingredients exist. A reviewer who labels a budget of windows has a sample; the
package already carries design-weighted estimators and cluster bootstraps (`oe_inferencex/metrics.py:143-247`,
`oe_inferencex/stats.py:22-158`) that exp68 used over a stratified population. The open questions are whether the
resulting interval is honest, whether the confidence ranking makes the sample cheaper, and by how much.

## The estimand

For task *t*, let `U_t` be the windows carrying a reference label (the `ok` rule of
`exp/exp54_multiclass_embeddings.py:138`: at least half the window's pixels are labelled). With `d_i` the map's
decision and `y_i` the reference's majority label, `e_i = 1{d_i != y_i}`, and

> **θ_t = (1/N_t) Σ e_i** — the finite-population window error rate of *this* map against *this* reference.

Three things this is not, each a way to misread the result. It is not a generalisation error: the randomness is
the reviewer's draw, not the data-generating process, so the intervals are design-based and the finite-population
correction applies. It is not the error rate of the whole map: windows with no reference sit outside `U_t`, so what
is estimated is *the error rate over the part a reviewer could adjudicate*. And it is not pixel accuracy.

Because every unit on these testbeds is labelled, θ_t is known exactly and equals `error_rate` in
`exp/out/exp70_tasks.csv`. Labels are withheld and revealed only for sampled units. **That is the only way an
interval can be graded**, and it is why this runs on the suite rather than on a real unlabelled map.

## Arms

Two axes, kept separate: the confidence can decide **which units get labelled** or **how the unlabelled ones are
used**, and the record should say which pays.

| id | design, all at the same labelled budget B | uses confidence |
|---|---|---|
| D1 | simple random sample without replacement | no — the baseline |
| D2p | stratified by margin quintile, proportional allocation | yes, no labels spent |
| D2c | same strata, Neyman allocation from the model's own confidence | yes, no labels spent |
| D2★ | same strata, Neyman from the true stratum rates — not deployable, reported as the ceiling | — |
| D4 | two-stage: T tiles, m windows each, B = T·m | no |
| D5 | tiles stratified by mean margin, then D4 within | yes |

| id | estimator | interval |
|---|---|---|
| E1 | design-based mean: Wilson with FPC for D1, stratified Wald-t for D2*, ultimate-cluster for D4/D5 | analytic |
| E2 | **naive Wilson on the B labels as if they were an SRS** — deliberately mis-specified for D4/D5 | analytic |
| E3ps | post-stratification: an SRS draw re-weighted to the known quintile sizes | analytic |
| E3d | difference estimator with the label-free `g_i = 1 - p1_i` as predictor, cross-fitted by tile | residual |

Grading is a Monte Carlo of R = 2,000 draws per (task, design, estimator, B), recording point estimate, interval,
coverage and width. Headline B = 300, with B ∈ {100, 300, 1000} reported.

## Disclosed pilot, and what it moved

**Three of the seven tasks have already been seen.** Before writing this, I ran the coverage study offline on
committed artifacts: MADOS and PASTIS Sentinel-2 from `exp/out/exp63_masks.npz`, and Sen1Floods11's two splits
from `exp/out/exp65_readings.npz`. Results at B = 300, m = 16:

| arm | error rate | SRS coverage (E1) | naive coverage after tile sampling (E2) | design effect |
|---|---|---|---|---|
| MADOS | 0.074 | 0.933 | 0.535 | 9.77 |
| PASTIS S2 | 0.190 | 0.958 | 0.760 | 2.70 |
| Sen1Floods11 Bolivia | 0.089 | 0.957 | 0.711 | 2.78 |
| Sen1Floods11 test | 0.047 | 0.955 | 0.718 | 3.84 |

This changed P2's threshold. The draft asked for naive coverage **under 0.75**; PASTIS Sentinel-2 sits at 0.760,
and PASTIS is three of the seven tasks, so the prediction would have failed on arithmetic while the effect it
describes was plainly present on every arm. The threshold is **0.85**, chosen with these four arms in view.

**Consequence, binding on the write-up:** MADOS, Sen1Floods11 and the PASTIS variants are pilot, not confirmation.
P2's genuinely out-of-sample tasks are **m-cashew-plant and m-SA-crop-type**, and the two counts are reported
separately. P1, P3 and P4 were not piloted and are tested on all seven.

## Predictions

**P1, the estimator is honest under the design it assumes.** Under D1 with E1, the nominal-95% interval covers θ_t
on at least **0.93** of 2,000 draws at B = 300, on all 7 tasks. *Why 0.93:* binomial noise at R = 2,000 has
sd 0.005, so 0.93 is four sd below nominal and a failure is a failure. Wilson rather than Wald because at MADOS's
error rate a B = 300 sample holds about 22 errors, where Wald is known to under-cover. *What makes it fail:* an
FPC against the wrong N, a variance formula that treats a stratified sample as simple. Under D1 the units are
exchangeable by construction, so **a P1 failure is a bug, and catching it is P1's job.** No other prediction is
graded until P1 holds. *Known risk, stated in advance:* MADOS piloted at 0.933, three sd below nominal, so P1 may
fail there on Wilson's discreteness rather than on a defect; if it does, that is what will be recorded.

**P2, the interval a reviewer would actually compute is badly wrong.** At B = 300 spent as T = 19 tiles of m = 16
windows (D4), the naive interval (E2) covers on **under 0.85** of draws on at least **5 of 7** tasks, and the
median design effect over the seven is **at least 2.5**. *Why:* the four piloted arms ran 0.535 to 0.760 with
design effects 2.70 to 9.77; 0.85 clears all four while remaining a substantive claim, since an interval covering
85% when it claims 95% is materially wrong. *What makes it fail:* errors that are spatially diffuse on the four
unpiloted tasks. Two of the piloted arms sit near a design effect of 2.7, and the two highest-error tasks were
never measured.

**P3, confidence-guided labelling helps most where the map is already good.** At B = 300 the best deployable
confidence-guided arm (best of D2p, D2c, E3ps) has median interval-width ratio **at most 0.92** against D1/E1 over
the seven tasks with coverage still at least 0.93; **and** the ratio is **at most 0.88** on both tasks whose error
rate is below 0.10 while exceeding **0.94** on at least one task whose error rate exceeds 0.30. *Why, and this is
a mechanism rather than a curve fit:* the gain from stratification is governed by the spread of √(p_h(1−p_h))
across strata, and that function is nearly flat for p between 0.25 and 0.7, so a map that is 35% wrong has little
to gain however well its errors are ranked. A model-free upper bound on the width ratio follows from the capture
at a 20% budget already in `exp/out/exp70_summary.json`: 0.775 on MADOS, 0.835 on Sen1Floods11, 0.897 on PASTIS
S2, 0.971 on m-cashew-plant. Median 0.897, so 0.92 leaves room to fall short of the ceiling without collapsing.
**This is the opposite of an operator's intuition**, which is why it is worth stating in advance. *What makes it
fail, in two different ways:* the ratio exceeds 0.92 everywhere, meaning the ranking does not separate error rates
enough to matter for estimation and the tool's ranking is for finding errors only; or the error-rate clause
reverses, meaning something other than the √(pq) mechanism is driving it. P3 was not piloted.

**P4, the honesty check: it does not change the order of magnitude of the budget.** No arm achieves a labelled-
budget saving of more than **1.8×** against D1/E1 at equal half-width on any task, and at B = 300 no arm produces a
half-width below **±4 points** on any task whose error rate exceeds 0.20. *Why:* the budget ratio is one over the
width ratio squared, and the model-free ceilings above give at most 1.66. An arm beating 1.8 has broken an
assumption, and P1 is where that shows. The second clause is arithmetic a reader can redo: at p = 0.34, B = 300,
the SRS half-width is ±5.4 points. *This is the prediction that answers the question honestly.* The tool's
ranking captures 17× a random review at a 5% budget; on **estimating** the same ranking is worth a quarter of the
labels, not an order of magnitude.

## How it runs, and why the cluster does almost nothing

Two stages, deliberately split so that the statistical work happens where it can be audited and repeated.

1. **`--stage export`, on the cluster.** Fit exp70's probe on all 24 tasks and write per-unit `margin`, `p1`,
   `dec`, `y`, `err` and packed validity to `exp/out/exp78_units/<task>.npz` with a `manifest.json`, about 72 MB.
   This is the only part that needs the 792 GB embedding cache. It also ends a standing cost: exp70 committed only
   summaries, so every reanalysis of the suite has needed a fresh embedding load.
2. **`--stage estimate`, locally, numpy only, no torch.** The whole Monte Carlo, re-runnable and auditable.

**Gate on the baseline before anything is graded.** Arm A must reproduce exp70's recorded accuracy on all seven
segmentation tasks (MADOS 0.9264 at 22,598 windows, Sen1Floods11 0.9155 at 592,385, and the rest as tabulated in
[the exp77 amendment](scene_contamination.md)). If it does not, the export is not the suite's per-unit data and
every future reanalysis built on it is about a different map.

## What would invalidate the run

- **θ recomputed from the sample** rather than read from the full population: coverage would then be meaningless
  and would look excellent. θ must equal `exp70_tasks.csv`'s `error_rate`.
- **A stratum with fewer than two sampled units**, which breaks the stratified variance estimator. Floor of 2 per
  stratum, and the rate at which it binds is reported.
- **Fitting the difference estimator's predictor on the sample without cross-fitting**, which introduces an
  O(1/B) bias that is not negligible at B = 100. Cross-fit by tile; report the uncross-fitted version beside it.
- **Tasks too small to carry the budget.** `awf_*` have 200 units. Eleven tasks can carry B = 300; the other
  thirteen are excluded with the reason recorded, as exp70 excluded `m_bigearthnet`. The export still covers 24.
- **The chip is not the reviewer's unit.** A 64×64-pixel tile is 640 m across; a reviewer opens a scene. The
  tile-level design effect is therefore a *lower bound* on the deployment one. This is a caveat, not a fixable
  defect, and it is carried into the record.
- **Counting seven tasks as seven datasets.** Three PASTIS variants are one source. Every count is reported by
  task and by the five distinct sources.

## Cost

One sbatch on `cpu` for the export: the probe fits are one pass where exp77 did five, so roughly 20 minutes for
the segmentation tasks plus 10 for the classification ones, and a few minutes of I/O. Request four hours. The
estimation stage is minutes on a laptop. No new dependencies; fp32 throughout.

## Addendum, 30 September 2026: two corrections to this page (nothing above changed)

- **P1 was piloted.** The pilot table above reports P1's own statistic, the simple random sample's coverage (E1),
  on four arms, MADOS at 0.933 among them. "P1, P3 and P4 were not piloted" is wrong for P1. The record's exp78
  section now says that MADOS's place on the bar was seen in the pilot, not predicted.
- **D4 drew 18 tiles, not 19.** The run took `max(1, B // m)` tiles, 300 // 16 = 18, each with up to 16 windows.
  P2 above says T = 19.

The package's own tiles design, which takes tiles until the budget is met and uses a t quantile, was graded
separately on 30 September 2026 (`exp/out/exp78_shipped_tiles.json`); the result is in the exp78 section of
[comparisons.md](../results/comparisons.md#how-wrong-is-this-map-what-a-reviewers-labels-buy-exp78).
