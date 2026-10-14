# Which part of the map can be trusted, with a guarantee? (exp80 preregistration)

Written before the run, on 23 September 2026, from the population facts disclosed below and from theory. Runs as
`exp/exp80_trust_zone.py`; the numbers land in the ledger like every other experiment.

## The gap

The record says an accuracy needs a coverage (exp21: the fine-tuned model is 0.93 accurate where it claims 0.99;
keep the 80% most confident windows and it is 0.945). That sentence is descriptive: it was read off the labels of
the whole test set. A map user cannot do that. What they can do is label a budget of windows, and what they want
to hear is not a risk-coverage curve but a **zone**: *the windows above this confidence are wrong at most α of the
time, and that statement fails with probability at most δ over my labelling.* The package's `estimate` gives an
error rate for the whole map with an interval; it does not give a zone, and nothing in the record grades one.

The field has a rule for this. Bates, Angelopoulos, Lei, Malik and Jordan (2021, *Distribution-free,
risk-controlling prediction sets*, [arXiv:2101.02703](https://arxiv.org/abs/2101.02703)) choose a threshold from
held-out labels so that a monotone risk is controlled at α with probability 1−δ, using a pointwise upper confidence
bound and the monotonicity; Angelopoulos, Bates, Candès, Jordan and Lei (2021, *Learn then Test*,
[arXiv:2110.01052](https://arxiv.org/abs/2110.01052)) reframe the choice as multiple hypothesis testing so that no
monotonicity is needed. Neither has been applied to a map audit, and neither says what a budget of 300 labels
buys on a real map. This is that measurement, on the 24 tasks whose every unit is labelled, so that the guarantee
can be graded rather than believed.

**Not conformal risk control.** Angelopoulos et al. 2022 ([arXiv:2208.02814](https://arxiv.org/abs/2208.02814))
controls the *expected* loss over all units, which for a zone is the share of all windows that are both inside the
zone and wrong. The quantity a map user asks about is the error rate *inside* the zone, in high probability.
That is the Learn-then-Test setting, and it is the only one tested here.

## The estimand

For task *t* with the adjudicable windows `U_t` of exp78 (N windows, decision `d_i`, reference `y_i`,
`e_i = 1{d_i != y_i}`, margin `m_i`), order the windows by margin, descending, ties broken by index. The **zone at
coverage c** is the first `n_c = round(c·N)` windows of that order, `S_c`. Its **risk** is

> **R_t(c) = (1/n_c) Σ_{i ∈ S_c} e_i** — the finite-population error rate inside the zone.

The grid of candidate coverages is `c ∈ {0.05, 0.10, …, 1.00}`. A procedure sees a simple random sample of B
labelled windows drawn from `U_t` and returns a coverage `ĉ` (or "no zone"). It is **valid** at (α, δ) when

> **P( R_t(ĉ) > α ) ≤ δ** over the reviewer's draw, with "no zone" counting as a success,

and it is **useful** to the extent `ĉ` approaches the oracle `c*(α) = max{c : R_t(c) ≤ α}`. As in exp78, the
randomness is the draw and nothing else; θ_t = R_t(1) is exp70's recorded error rate, read from the population
and never from the sample.

## The two things labels can buy, and the arithmetic that limits them

Under a simple random sample, the sampled windows that fall inside `S_c` are a simple random sample of `S_c`
(hypergeometric), so with `b_c` of them holding `k_c` errors the exact one-sided p-value against
`H_c : R_t(c) > α` is `P(X ≤ k_c)` for `X ~ Hypergeom(n_c, ⌊α n_c⌋ + 1, b_c)`, the null count that makes the
p-value largest. With no errors among `b` labels this p-value is at most `(1−α)^b` (the binomial bound), so a zone
can only ever be certified when it holds at least

> **b_min(α, δ) = ⌈ ln δ / ln(1−α) ⌉ labels** — 45 at α = 0.05, 114 at 0.02, 255 at 0.009, all at δ = 0.1.

This is the honest refusal the tool must be able to make: at B = 300 and δ = 0.1, no zone smaller than
`b_min / B` of the map can be certified whatever its true quality, and an α below about 0.008 cannot be certified
at all. The grid is therefore cut at `c_min(α, δ, B) = b_min / B`, rounded up to the grid, before any label is
seen; that cut is a function of (α, δ, B) alone and is reported with every result.

## Arms

All three see the same draw, the same grid above `c_min`, and return the largest coverage they accept.

| id | rule | what it assumes | what it is for |
|---|---|---|---|
| A0 | **plug-in**: largest c with `k_c / b_c ≤ α` | nothing, and it has no guarantee | what a reviewer would do without the theory |
| A1 | **monotone prefix** (Bates et al. 2021): largest c such that every grid coverage `c' ≤ c` has exact upper bound `UCB_δ(k_{c'}, b_{c'}, n_{c'}) ≤ α`, where `UCB_δ` is the largest `K/n` not rejected at level δ | `R_t(c)` nondecreasing in c | the powerful rule; valid when the risk is monotone, because a violation requires the pointwise bound to fail at the single fixed coverage just above `c*` |
| A2 | **Bonferroni Learn-then-Test** (Angelopoulos et al. 2021): test every grid coverage at level `δ / J` with the exact p-value, accept the largest rejected | nothing about monotonicity | the assumption-free rule; the price of not assuming is measured against A1 |

Budgets B ∈ {100, 300, 1000}, headline 300; δ = 0.1 throughout; α ∈ {θ_t / 2, 0.05}: the relative level asks for
a zone twice as good as the map, the absolute level is the number a user would type. R = 2,000 draws per
(task, α, B), seed 0. Every arm's p-values and bounds are hypergeometric and exact; no normal approximation.

## Disclosed population facts, and what they fix in advance

These are properties of the labelled populations, not of any estimator, computed from `exp/out/exp78_units/`
before this page was written. They fix expectations; they are not confirmation of anything below.

| fact | value |
|---|---|
| tasks on which `R_t(c)` is nondecreasing across the whole grid | 11 of 24 |
| largest decrease between adjacent grid coverages, where any | 0.034 (CropHarvest Togo S2); 0.023 (Togo S1); 0.021 (Togo S2+S1); 0.017 (Nandi S1); 0.010 (AWF S1); the other eight ≤ 0.006 |
| oracle coverage at α = θ_t / 2 | from 0.001 (Nandi S1) and 0.150 (ForestNet) to 0.973 (EuroSAT); 17 tasks above 0.5 |
| oracle coverage at α = 0.05 | 1.0 on EuroSAT and Brick Kiln; 0.93 MADOS; 0.90 Sen1Floods11; 0.56 both PASTIS S2 arms; 0.001 to 0.35 on the rest |
| calibration gap, mean top-1 probability minus accuracy | median +0.061, max +0.184 (So2Sat); the map claims more than it delivers on 19 of 24 tasks |

So A1's assumption is exactly true on 11 tasks and broken by at most a few hundredths on the rest, which is the
regime in which its guarantee is most worth testing; and at α = θ_t / 2 the tasks with the lowest error rates
(EuroSAT 0.018, Brick Kiln 0.036) have `b_min` above 255 or 114 labels, so at B = 300 they will certify nothing on
most draws or only the near-whole map. That is not a defect; it is the arithmetic the tool will have to state.

## Predictions

**P1, validity, from the theorem.** For A2, on every (task, α, B) cell, the Monte Carlo frequency of
`R_t(ĉ) > α` is at most **δ + 3·√(δ(1−δ)/R) = 0.120**. For A1 the same bound holds on **every cell of the 11
monotone tasks**. *Why:* both are theorems under their assumptions; R = 2,000 draws give the frequency a
standard error of 0.0067. *What makes it fail:* a wrong p-value (the null count off by one, a normal approximation
where the exact sum was promised, a sample inside the zone that is not hypergeometric because ties were broken
after seeing the draw). **A P1 failure on A2 or on a monotone task is a bug, and no other prediction is graded
until it is fixed.** On the 13 non-monotone tasks A1 is *not* covered by its theorem; its violation frequency is
recorded per cell and reported beside the largest decrease in `R_t(c)` on that task, and the record will say
whether a decrease of the sizes above is enough to break the guarantee at B = 300.

**P2, the run agrees with the arithmetic.** Replace the random counts by their expectations, `b̄_c = B n_c / N`
and `k̄_c = b̄_c R_t(c)`, and run each rule once; call the result `c̄` ("the arithmetic"). At B = 300 and α = θ_t / 2,
for A1 and for A2 separately, on **every task** the modal outcome across draws (zone or no zone) equals the
arithmetic's, and where both give a zone the median `ĉ` is within **two grid steps (0.10)** of `c̄`. *Why two
steps:* the count in the zone at the boundary has a standard deviation of about √(b α (1−α)), one or two errors at
these budgets, and the risk curve rises by about one grid step per error there; a larger disagreement means the
implementation is not the procedure this page describes. *What makes it fail:* an off-by-one in the grid cut, a
p-value computed on the wrong `n_c`, or a curve so flat near α that the stopping coverage is genuinely bimodal;
the last is reported per task if it happens and is not a defect.

**P3, the guarantee is worth having.** At B = 300 and α = θ_t / 2, A0 violates `R_t(ĉ) > α` on **more than δ =
0.10** of draws on at least **18 of 24** tasks. *Why:* the plug-in stops where the sample rate first crosses α,
which is where the true rate is near α and the sample rate exceeds it with probability near one half; the
exceptions allowed are tasks whose curve crosses α at a coverage with almost no labels, where the plug-in's
behaviour is a coin on a few units. *What makes it fail:* a plug-in that violates rarely, which would mean the
risk curves are so steep at α that the sample rarely straddles it; then the guarantee costs coverage for little,
and the record says so.

**Descriptive, not predicted.** Per (task, α, B) and arm: median and 10th-percentile certified coverage, the
share of draws returning "no zone", the oracle `c*(α)`, the ratio of A1's to A2's median coverage (the price of
not assuming monotonicity), and `c_min(α, δ, B)`. The calibration gap above is recorded per task as the answer to
"what does the map alone say about its error rate": nothing a user should act on.

## Independent check before anything is recorded

1. **An enumeration test** (`tests/test_estimate_exact.py` pattern): on a population small enough to enumerate
   every sample (N = 20, B = 6, 38,760 draws), the exact violation probability of A2, and of A1 on a monotone
   population, is at most δ for every α on a grid; the hypergeometric p-value equals an explicit rational sum.
2. **Two implementations of the p-value** (the package's log-gamma sum and a `fractions.Fraction` sum in the
   test) agree to 1e-12 on every (n, K, b, k) up to n = 60.
3. **An adversarial read of the implementation by a separate agent**, on the real per-unit files, before the
   summary is written into the record: ties, `c_min` rounding, the direction of the prefix, and whether θ is
   ever recomputed from the sample.

## What would invalidate the run

- θ_t or `R_t(c)` recomputed from the sample rather than the population: coverage would be meaningless.
- A tie order chosen after the draw: the zone must be a fixed set for the hypergeometric argument to hold.
- Grading A1 as valid on a non-monotone task and calling it a theorem; the theorem does not cover it.
- Calling a certified zone "trustworthy" without its α and δ: the output is a three-number statement.

## Amendment, 23 September 2026, from the independent audit, written after the run and before the record

The audit (an agent with its own hypergeometric sums, its own zone logic and its own Monte Carlo, on the real
files) found nothing that changes a number and four things this page got wrong, corrected here without moving
any prediction's grade:

1. **The monotone tasks are 10 of 24, not 11.** Miscounted from the same table; the ten are So2Sat, Nandi
   Landsat, CropHarvest China (S2), BreizhCrops, MADOS, the three PASTIS arms, m-cashew-plant and m-SA-crop-type.
   The code graded P1 per cell on whether the risk is nondecreasing across the *tested* levels, which is what the
   theorem needs and is stricter than this page's per-task rule; P1's verdict is the same under either.
2. **`b_min` is a floor only in the large-population limit.** On a zone nearly all of whose windows are labelled
   the exact hypergeometric p-value is smaller than the binomial bound, so the cut can discard a certifiable
   level on a small map; it never admits an uncertifiable one. The docstring and the sentence above now say so.
3. **P2 failed on the two near-census tasks for a reason this page did not foresee:** with 300 of 306 windows
   labelled, the first tested zone is fully drawn on 52% of draws (p = 0) and short by one on the rest, so the
   arithmetic's rounded expected count lands on the wrong side of δ. P2 holds on all nineteen other tasks. The
   page should have excluded cells with B/N above 0.9 or defined the arithmetic on modal counts; it did neither,
   and P2 is recorded as failed as written.
4. **P3's bar was written for 24 tasks when only 21 have a 300-label cell** (the three AWF tasks have 200 units).
   The plug-in violates on 16 of 21; a proportional bar would pass and the literal one fails. P3 is recorded as
   mis-stated and failed as written, with the decomposition: the five below δ are the three near-census Togo cells,
   Nandi S1 (no zone exists) and EuroSAT (α = 0.009, the steep-curve case named above).

Also stated: the α = θ/2 and α = 0.05 cells at one budget share the same draws (the generator is reset per cell),
so they are not independent replications; zone sizes use round-half-to-even; the summary's `oracle_coverage` is
on the tested grid, the table above is continuous. The tie plateaus at margin exactly 1.0 (377 of EuroSAT's 1,000
windows) mean that "margin ≥ threshold" is not the zone there; the tool now reports the tie counts and returns
the certified set itself.

## Cost

No cluster. `exp/out/exp78_units/` for OlmoEarth Base is committed; the estimation is numpy only, minutes per
task. When the exp79 seed-0 exports for the other fifteen encoders arrive, the same stage runs on each and the
record states whether the certifiable coverage is a property of the task or of the encoder, as exp79 P4 does for
the design effect. The procedure ships as `oe_inferencex.estimate.certify_zone` and `oe-inferencex certify`
only after P1 holds; the experiment imports the package's implementation so that there is one.

## Addendum, 30 September 2026: the prefix rule needs no monotone risk (nothing above changed)

Rule A1 above is described as valid when the risk is monotone, and A2 as the assumption-free rule. That understated
A1. The zone order and the grid are fixed before any label is read, each level has an exact p-value, and the rule
stops at the first level it cannot reject. That is fixed-sequence testing, so a false certification needs the first
true null to be rejected, which happens with probability at most δ whatever the shape of the risk curve. Learn then
Test (Angelopoulos et al. 2021) gives the general argument. A test in `tests/test_trust_zone.py` now enumerates every
error pattern of maps of 10, 11 and 12 windows against every draw, and both rules hold δ on all of them. So exp80's
result on the tasks whose risk is not monotone is what the rule guarantees, not luck. Neither rule is the more
powerful in general. Bonferroni can certify more where the most confident windows hold many errors, as a toy map in
`tests/test_trust_zone.py` shows. In exp80 it certified a zone on more draws than the prefix rule on 25 of the 112
cells, and on fewer on 72. Brick Kiln is not such a case: there Bonferroni certified nothing on 74.8% of draws, and
the 0.75 the results record gives is its median over the rest, while the prefix rule certified 0.50 on every draw.
The package's printed note was corrected to say that Bonferroni can certify more, in changes not yet released (released in 1.4.0).
