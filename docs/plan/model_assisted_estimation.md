# Does the map's own confidence sharpen the error rate once labels exist? (exp85 preregistration)

Written before the run, on 23 September 2026. Runs as `exp/exp85_model_assisted.py`; the numbers land in the
ledger like every other experiment.

## The gap

exp78's plan listed an arm it never ran: a difference estimator that uses the map's own confidence on every
unlabelled window as a predictor of error, `g_i = 1 − p1_i`, and labels only to correct it. The record's text
never claimed it ran; the estimator the tool ships uses the confidence to decide *which* windows to label (the
stratified design) and nothing else. The field's rule (`docs/plan/methodology_redesign.md`, section 1) is to use
the predictions on the unlabelled units through a debiased difference estimator whose coefficient is tuned so
that the interval is never wider than the classical one: prediction-powered inference with a tuned λ (Angelopoulos,
Duchi and Zrnic 2023, PPI++), its stratified form (Fisch et al. 2024), and the observation that this is the survey-
sampling difference estimator (Mozer et al. 2026). The record already bounds what it can buy: exp78's P4 found no
design saving more than 1.8× labels against a random sample at equal precision, and the model-free ceiling from
the margin's capture is 1.66×. This run measures the arm, records its honesty, and says whether it ships.

## The estimand

θ_t, the finite-population window error rate of exp78, on OlmoEarth Base's seven segmentation tasks from exp78's
export (the tasks exp78 graded, so every number here sits beside one already recorded), and, descriptively, on the
seventeen classification tasks. Labels are revealed only for sampled windows; `g_i` is known for every window.

## Arms, all at B = 300, R = 2,000 draws, seed 0

| id | design | estimator | interval |
|---|---|---|---|
| D1/E1 | random | mean of the labels | Wilson with FPC (exp78's record) |
| D1/E1w | random | mean of the labels | Wald with FPC, the reference PPI++'s guarantee is stated against |
| D1/E3d | random | difference estimator, λ = 1: `ḡ_N + (ē_n − ḡ_n)` | Wald, `(1 − f) s²(e − g)/n` |
| D1/E3d++ | random | PPI++: `λ̂ = s(e, g)/s²(g)` on the sample, `λ̂ ḡ_N + (ē_n − λ̂ ḡ_n)` | Wald, `(1 − f) s²(e − λ̂ g)/n` |
| D2c/E1 | confidence (exp78's) | stratified mean | Wald (exp78's record) |
| D2c/E3s | confidence | stratified PPI: per stratum `λ̂_h`, `Σ W_h [λ̂_h ḡ_{N_h} + (ē_h − λ̂_h ḡ_h)]` | Wald, `Σ W_h² (1 − f_h) s²_h(e − λ̂_h g)/n_h` |

`ḡ_N` and `ḡ_{N_h}` are population constants (the confidence is known on every window), so they add no variance.
A stratum with fewer than three labels takes λ̂_h = 0 (its classical estimate). Bias is recorded beside coverage
for every arm, as the audit rule requires.

## Predictions

**P1, the arms are honest.** D1/E3d++ and D2c/E3s cover θ_t on at least **0.93** of draws on all seven tasks, with
a bias ratio in **[0.98, 1.02]**. *Why:* the difference estimator is exactly unbiased for any fixed λ under a
random draw (checked by enumeration in `tests/test_model_assisted.py`); tuning λ on the sample adds a bias of
order 1/n, which 2% covers at n = 300; 0.93 is exp78's bar. *What makes it fail:* the tuned coefficient's
small-sample bias on a task with few sampled errors (MADOS, about 22), which would show as coverage below the bar
there first.

**P2, the tuned arm is never worse than the classical one.** The median half-width of D1/E3d++ is at most
**1.00** times that of D1/E1w on every task. *Why:* PPI++'s point, that λ̂ = 0 recovers the classical estimator, so
the tuned one cannot lose more than the tuning noise; against the Wald reference that noise is the only thing
that could push the ratio above 1. *What makes it fail:* a ratio above 1.00 anywhere, which is a defect in the
tuning, not a finding.

**P3, the gain is small because the design already took it.** The median over the seven tasks of the half-width
ratio D2c/E3s against D2c/E1 is at most **0.95**; and the ratio D1/E3d++ against D1/E1w is at most **0.90** on the
two tasks whose error rate is below 0.10 (MADOS, Sen1Floods11) — the place exp78 found the confidence worth most.
*Why:* within a confidence stratum the residual correlation of `g` with error is what remains after the strata
have conditioned on the margin quintile, so the stratified arm gains little; the unstratified arm gains the whole
correlation, which exp78 measured as a 0.63–0.80 width ratio for the design on the clean tasks. *What makes it
fail:* a median above 0.98 for D2c/E3s, in which case the arm is recorded as "no gain at this ranking quality" and
does not ship as a flag.

**P4, why the coefficient is tuned.** Plain D1/E3d (λ = 1) has a larger median half-width than D1/E1w on at least
**2 of 7** tasks. *Why:* `g = 1 − p1` overstates accuracy by a median 0.061 on the suite (exp80's calibration gap),
and a predictor that is right in direction but wrong in scale adds variance at λ = 1 that tuning removes.

**P5, exp78's honesty bound stands.** No arm saves more than **1.8×** labels against D1/E1 at equal half-width on
any task. *Why:* the capture-based ceiling is 1.66×; a saving above 1.8× means a leak (a predictor fitted on the
sample), and P1 is where it shows.

**Descriptive.** The seventeen classification tasks under the same arms; the tuned λ̂ per task and stratum; the
correlation of `g` with error within and across strata.

## Independent check before anything is recorded

`tests/test_model_assisted.py`: over every random sample of a 12-window population the difference estimator is
exactly unbiased for λ ∈ {0, 0.5, 1} and its variance estimator unbiased for the exact variance; the stratified
form the same over every stratified sample; the tuned coefficient converges to the population covariance ratio.
Then an adversarial read by a separate agent on the real per-unit files.

## Amendment, 23 September 2026, from the independent audit, written after the run and before the record

The audit recomputed every estimator by hand (agreement to 1e-16), reproduced the Monte Carlo with its own
seeds, and found the arithmetic right and two of this page's premises wrong.

- **P5 is withdrawn as written.** This page said exp78 "found no design saving more than 1.8×"; exp78's P4
  *failed* at 2.51× on MADOS and the record says so. The grader's 8.3× is EuroSAT's stratified arm with a
  collapsed 27%-coverage interval, not a saving, and there is no leak (bias 0.9996, the predictor label-free).
  The recordable statement is the model-assisted increment over its classical twin: 1.035–1.054× labels on every
  segmentation task.
- **P4's reasoning was a category error.** A level offset in the predictor is absorbed exactly by the sample
  correction and adds no variance; λ = 1 beats the classical estimator whenever the population coefficient exceeds
  0.5, and it is 0.93–1.04 on the seven tasks, so tuning buys 0.2–0.7% of width. P4 fails (λ = 1 is never wider).
- **P1 holds at the bar and is seed-dependent.** MADOS's stratified model-assisted arm recorded 0.9305, one
  draw above 0.93; over ten seeds it is 0.932 ± 0.002 and below the bar on three of ten. The model-assisted
  stratified arm covers half a point less than the classical stratified arm on all seven tasks. The mechanism:
  the per-stratum coefficient tuned on a small stratum whose predictor barely varies explodes (66 on a 26-label
  stratum; |λ_h| > 3 on 20–78% of draws on some strata), and the Wald variance treats it as fixed. The tuning
  floor of 3 labels was far too low; the package's default is now **30**, at which the stratified arm's width
  gain vanishes. The run's artifact stays as run (floor 3).
- **The classification cells are a harness inheritance, not a finding about the arm.** The Monte Carlo allocates
  as exp78 did (no floor on the assumed stratum rate), so on EuroSAT four strata get 2–6 labels and the interval is
  degenerate on 74% of draws for the classical stratified arm too (0.27 both); the shipped package allocates with
  `Q_FLOOR` and uses Wilson on the effective sample size, which cover 0.93–0.95 there. The classification rows
  are descriptive and say this.
- **P3 fails robustly** (stratified median 0.978 with a paired standard error of 0.0003; Sen1Floods11's random-
  sample ratio 0.901 against 0.90 by 1.5 standard errors), and the Cost section's rule decides: **the arm does not
  ship.** The tests exercised neither the tuned stratified form nor small strata; a test at the floor is added.

## What would invalidate the run

- `g` computed from anything a label touched; it is `1 − p1` from the export and nothing else.
- θ recomputed from the sample.
- A saving above 1.8× reported as a gain rather than as a leak.

## Cost

No cluster; exp78's export is committed; minutes. Ships as `estimate_error_rate(..., predictor=g)` only if P1
holds and P3's stratified ratio is at or below 0.95; otherwise the record says what the arm is worth and the
tool stays as it is.
