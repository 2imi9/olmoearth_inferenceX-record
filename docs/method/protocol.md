# Evaluation protocol

![The test every claim passed: a candidate scored next to the model's confidence and a no-model control, on both references on identical windows; labels grade, never train](../figures/protocol.png)

How results in this repository are produced, weighted, and named. Every
other document assumes this page and does not restate it. Index at
[TECHNIQUES.md](../TECHNIQUES.md).

## What is being measured

The apparatus answers one question in two forms: **given two things that
disagree, which should you believe, and is the difference real?**

- **Error ranking.** Each signal assigns every map window a suspicion score;
  the score is judged by how well it ranks the windows the model gets wrong.
- **Cross-inference comparison.** Two inferences of the same scene, shifted
  crops, backbones, sensors, a frozen and a fine-tuned model, input years,
  are compared on the windows both predicted: how much they differ, where
  the difference sits, whether two differences are the same set, all
  measured without labels; then, where labels exist, which side is right
  and what one side corrects and breaks (`oe_inferencex.compare`, exp57).

The machinery is the same for both, and is deliberately signal-agnostic:
`aurc_expected(uncertainty, errors)` takes any score vector and any error
vector. Signals are label-free; labels only grade the signals, never train
them. The machinery lives in the package: `oe_inferencex.metrics` (this
section's metrics), `oe_inferencex.signals` (confidence, the boundary
indicator, aligned tile-phase, the pixel controls) and `oe_inferencex.stats`
(the tests below); `tests/` reproduces the recorded numbers from the
committed artifacts.

A signal is credible only if it beats two references at once:

- **the model's own confidence**, scored as the negative absolute logit
  (top-1 minus top-2 for multiclass) rather than `1 - max probability`, so
  saturated probabilities do not tie;
- **a no-model pixel control**, computed from pixel values alone (for
  water, the NDWI gradient magnitude).

Beating one but not the other is not support.

## How results are scored

The statistics settled in exp13 and used by every experiment after it:

| Element | Choice | Why |
|---|---|---|
| Metric | Tie-aware AURC, and excess AURC (E-AURC = AURC minus the oracle's) across scenes | The boundary score has nine levels and float32 sigmoid saturation ties confidence at zero on many patches, so a stable-sort AURC would depend on raster order. E-AURC makes absolute levels comparable across scenes; it changes no signal-minus-baseline difference. |
| Perturbation alignment | Shifted prediction maps upsampled to pixels, placed at their true offset on a common canvas, then pooled back | exp05/exp09/exp11 compared unaligned patch grids, so shifted patches covered different ground. |
| Per-scene uncertainty | 4x4-patch block bootstrap, B=1000 | Percentile intervals are biased for a rank statistic on high-error scenes, so they are indicative only. |
| Cross-scene tests | Exact sign test on untied pairs, plus a sign-flip permutation test on mean E-AURC differences | The sign test is scale-free and is reported as primary; mean-based tests are dominated by high-error scenes. |
| Reporting | Wins / losses / ties per scene, not means | Same reason. |

### The closed forms these statistics divide by

Eleven quantities appear as bare formulas throughout this record: six behind the ranking statistics and, since
the estimator shipped, five behind the error-rate estimate. Each is derived here once, so that a reader does not
have to take any of them on trust, and each is checked — in `tests/test_formulas.py` and
`tests/test_estimate_exact.py` — against a brute-force computation or a full enumeration written separately from
the package's own. The derivations are short; the reason they are written
down at all is that on 20 September 2026 a recorded claim was false and its own check passed, because the check
read its number back from the artifact that produced it. A formula inside a check is an assumption of the test,
never its subject.

**Notation.** `n` units, `k` of them errors, error rate `e = k/n`. A ranking rejects the most suspect unit first.
At coverage `c` (the fraction kept, least suspect first), the selective risk is the error rate among those kept.
AURC is the mean of that risk over all `n` coverage levels.

**1. The perfect ranking's AURC.** The oracle rejects every error before any correct unit. Keeping `i` units
therefore keeps `max(0, i - (n - k))` errors, so

```
AURC(oracle) = mean over i = 1..n of  max(0, i - (n - k)) / i
```

which is `metrics.oracle_aurc(n, k)` exactly. Its continuous limit follows by writing `i = cn` and integrating:
the risk is zero while `c <= 1 - e` and `(c - 1 + e)/c` above it, so

```
∫ from 1-e to 1 of (c - 1 + e)/c dc = [c - (1-e) ln c] from 1-e to 1 = e + (1-e) ln(1-e)
```

**These two are not interchangeable.** They agree only as `n` grows, with a gap of order `1/n`. Mixing them —
an exact oracle in a numerator and the limit in the denominator of one fraction — moved a published figure in
this record from 0.553 to 0.551 before it was caught. Use `metrics.oracle_aurc`; the limit is for reading, not
for computing.

**2. A random ranking's AURC is the error rate.** A ranking uncorrelated with the errors keeps, in expectation, a
fraction `e` of errors at every coverage, so the selective risk is `e` at every `c` and its mean is `e`. The
exactly-tied case is the same statement without sampling noise: when every unit has the same score, the tie-aware
AURC is `e` identically.

**3. The ranking headroom.** Combining the two: a random ranking scores `e`, a perfect one scores
`AURC(oracle)`, so the gap any ranking can close is `e - AURC(oracle)`, and the share a given reading closes is

```
headroom = 1 - E-AURC(reading) / (e - AURC(oracle))
```

One at the oracle, zero at chance. This is the statistic behind "the margin takes a median 0.68". It is a
rescaling of AURC, not a new measurement, and it is the normalisation that makes tasks with different error rates
comparable — which is also why both its terms must use the same oracle.

**4. The attainable ceiling at a review budget.** A review of `k_rev` units contains at most `k_rev` errors, and
at most all `E` of them, so the captured share is at most `min(k_rev, E)/E = min(1, k_rev/E)`; with `k_rev = bn`
and `E = en` that is `min(1, b/e)`. The oracle attains it. Two conventions differ on small maps: the nominal
budget `b`, and the realised review set `k_rev = max(1, round(bn))` that the code actually scores.
`metrics.attainable_ceiling` takes either, and passing `n` selects the realised one.

The consequence is exp68's audit correction: **a capture compared between two populations with different error
rates is partly a comparison of ceilings.** Report the share of the ceiling, or a ceiling-free statistic such as
AUROC or an odds ratio.

**5. The enrichment ceiling.** A cue's enrichment is its share among error units over its share among correct
units. Since the latter is at most 1, enrichment is bounded by `1 / (share among correct)`. A cue that fires on
few correct units can therefore post a large enrichment while separating nothing, which is why the odds ratio is
preferred wherever the two subsets being compared differ in how often the cue fires.

**6. AUGRC is an affine function of the failure AUROC, so it cannot reorder readings.** The generalised risk
counts an error only while it is still kept and divides by the whole population rather than by the kept set. Order
the units least-suspect-first; an error at position `r` from that end is kept at `n - r + 1` of the `n` coverage
levels, so

```
AUGRC = (1/n^2) * sum over errors of (n - r_j + 1)
```

Write `c_j` for the number of correct units less suspect than error `j`. Then `r_j - 1 = c_j + (errors below j)`,
and summing over the `k` errors gives `sum r_j = AUROC_f * k(n-k) + k(k-1)/2 + k`, because `sum c_j` counts exactly
the (error, correct) pairs the failure AUROC is the rate of. Substituting and writing `e = k/n`:

```
AUGRC = (1 - AUROC_f) * e * (1 - e)  +  e^2 / 2  +  e / (2n)
```

Checked to machine precision (1e-16) over 96 cases in `tests/test_formulas.py`; `metrics.augrc` computes it from
the definition and `metrics.augrc_from_auroc` from this identity.

Two consequences the record uses. First, for a fixed error set and fixed `n` this is **exactly affine and strictly
decreasing** in the failure AUROC, with slope `-e(1-e)`, which is negative for any map that is neither perfect nor
wholly wrong. AUGRC therefore cannot order two readings differently from the failure AUROC, which is why exp76
could answer the AUGRC challenge from exp70's recorded AUROC without recomputing anything. Second, the last two
terms do not depend on the reading at all, so they cancel in any comparison of readings on one task — but they do
**not** cancel across tasks, so an AUGRC compared between tasks is partly a comparison of error rates.

The `e/(2n)` term is the discrete correction the continuous derivation misses. On the suite's smaller tasks it is
not negligible, so the continuous form is for reading and the identity above is for computing.

### The estimator's closed forms

`oe_inferencex.estimate` answers how wrong a map is from a labelled sample. Everything it computes is
finite-population sampling theory, and unlike the ranking statistics above each claim can be checked **exactly**
on a small population by enumerating every possible sample; `tests/test_estimate_exact.py` does so, with
rational arithmetic where the claim is an equality. Notation: `N` windows, `θ` the true window error rate;
strata `h = 1..H` of sizes `N_h`, weights `W_h = N_h/N`, rates `θ_h`, and `S_h² = N_h θ_h(1−θ_h)/(N_h−1)` the
stratum variance on the `N_h − 1` convention; a sample takes `n_h` from stratum `h` without replacement,
`f_h = n_h/N_h`, and observes `p_h`.

**7. The stratified estimate is unbiased and its variance is exact.** `θ̂ = Σ_h W_h p_h`. Within a stratum the
draw is simple random without replacement, so `E[p_h] = θ_h` and `θ̂` is unbiased; the strata are drawn
independently, so

```
Var(θ̂) = Σ_h W_h² (1 − f_h) S_h² / n_h
```

which is Cochran's result. The estimator uses `p_h(1−p_h)/(n_h−1)` in place of `S_h²/n_h`; that is `s_h²/n_h`
with `s_h²` the sample variance on `n_h − 1`, and `E[s_h²] = S_h²` under SRS without replacement, so the variance
the interval is built from is unbiased for the true one. Checked by enumerating all 180 stratified samples of a
12-unit population: the mean estimate equals `θ` and the mean estimated variance equals the enumerated variance
of the estimate, both to 1e-12.

**8. Neyman's allocation minimises that variance.** Minimise `Σ W_h² S_h²/n_h` subject to `Σ n_h = B`
(dropping the `1 − f_h` terms, which do not depend on the allocation's shape once `B` is fixed). The Lagrange
condition `−W_h² S_h²/n_h² = λ` gives `n_h ∝ W_h S_h ∝ N_h S_h`, so

```
n_h = B · N_h S_h / Σ_j N_j S_j
```

The design does not know `S_h`, so it uses the model's own confidence: `S_h ≈ √(q_h(1−q_h))` with `q_h` the
stratum's mean `1 − p₁`. That is why no label is spent estimating stratum rates, and why the gain is largest
where confidence separates the stratum rates most — the clean maps of exp78. The continuous optimum is exact;
the package's integer allocation floors it and hands the remainder to the strata with the most units left,
which costs 1.3 to 2.3% of variance on tiny cases and 0.1 to 0.5% on exp78's real strata at a budget of 300
(measured, not assumed; the rule is kept as exp78 ran it so its recorded numbers remain the package's output).

**9. Wilson's coverage is a hypergeometric sum.** For a simple random sample of `B` from `N` with `K` errors,
the error count `k` is hypergeometric, so the exact coverage of the interval is
`Σ_k P(k; N, K, B) · 1[θ ∈ CI(k)]`. This is what `exact_coverage_srs` computes and what a Monte Carlo coverage
is judged against (exp79, P5), so that Wilson's discreteness — MADOS at 0.933 in exp78 — is read as the
interval's property and not as a defect. Checked against a full enumeration of subsets on three populations.
The same sum judges any interval, and it passes nothing by itself: a Monte Carlo of a deterministic interval
converges to that interval's exact coverage whatever its defect, so agreement tests the sampler, not the interval
(exp81's audit). Since 23 September the tool's random-design interval is the tail inversion of this
hypergeometric count (`hypergeom_interval`), whose exact coverage is at least 95% for every `(N, K, B)`; the
Wilson form stays in the experiments' harness, which is what exp78 and exp79 graded.

**10. Labels taken tile by tile.** `T` tiles of `m` windows, `t` tiles drawn. The cluster-sample mean has
variance `(1 − t/T) S_b²/t` with `S_b²` the variance of tile means, and a simple random sample of the same
`n = tm` windows has `(1 − n/N) S²/n`. Their ratio is **exactly** `m S_b²/S²`, and writing the total sum of
squares as between plus within, `(N−1)S² = (T−1) m S_b² + (N−T) S_w²`, it equals `1 + (m−1)ρ` with `ρ` the
ANOVA intra-cluster correlation up to a term of order `1/T`. Checked by enumerating all 56 draws of 3 tiles from
8 (the ratio to 1e-10) and at `T = 40` (the design effect within 3%). This is why the ordinary formula on
tile-sampled labels claims 95% and delivers 51 to 78%: it uses `S²/n` where the truth is `m S_b²/S²` times
that, and on the six exp78 tasks where it failed that factor ran 2.7 to 9.8 (1.17 on m-cashew-plant, where it held).

All of that assumes tiles of equal size. When tile `i` holds `n_i` windows the quantity of interest is still the
window rate `θ = Σ n_i Ȳ_i / Σ n_i`, and the unweighted mean of tile means estimates `mean_i Ȳ_i`, the rate of the
average *tile*, which is a different number whenever size and error rate are related. The estimator is the ratio
`θ̂ = Σ n_i ȳ_i / Σ n_i` over sampled tiles, with ultimate-cluster variance
`Σ n_i²(ȳ_i − θ̂)² / (t(t−1) n̄²)`; on equal tiles it is exactly the mean of tile means, which the tests assert.
MADOS is the case where the difference matters: tiles of 1 to 400 valid windows, an average tile wrong 13.3% of
the time against 7.4% of windows, and an unweighted estimator 1.78 times the truth. The ratio estimator is
unbiased there (1.03 over 1,000 draws, Monte Carlo SE 0.03) but its interval from 18 tiles covers 0.60, because a
variance estimated from so few and so unequal clusters is itself unreliable: with tiles like these, the design is
wrong and no estimator applied afterwards rescues it.

**11. What labelling the review set gives.** The review set at budget `b` is the `k = bN` most suspect
windows; `capture(b)` is the share of all `E = θN` errors it holds. The rate a reviewer computes on it is
therefore

```
rate on the review set = capture(b) · E / k = capture(b) · θ / b
```

so the inflation over the truth is `capture(b)/b`, the enrichment the record already measures. With exp70's
recorded `capture(0.05) = 0.291` and `θ = 0.0736` on MADOS this gives 0.428, which is what labelling the 5%
review set of exp78's export and dividing returns. The guard in `estimate_from_indices` refuses that sample by
its mean suspicion percentile, but the size of the mistake it prevents is not a measurement: it is
`capture/b`, and it is 5.8× on MADOS because the ranking is good.

### How a difference is measured

The comparison half, settled in exp57 and probed in exp58; the arithmetic is
`oe_inferencex.compare`, one implementation for every experiment:

| Element | Choice | Why |
|---|---|---|
| Unit | Two hard decisions on identical windows, with the validity mask of the windows both predicted; probabilities are thresholded and class scores argmaxed before the call | A difference is a property of a window, not of a threshold; floating-point noise between two probability maps is not a difference. |
| Label-free readings, in this order | The disagreement rate pooled and per tile or event; the enrichment of each label-free cue among the disagreement windows against the agreement windows (boundary, low confidence, tiling instability, spectral ambiguity, the explanation layer's cues); the pairwise phi of disagreement sets across head draws and across pairs | These are the statements a user can make about two maps with no reference. They come first so that the graded reading never leaks into them. |
| Graded readings, second | Which side matches the label where the two disagree; the cross-tab of the two error maps (corrected, broken, both, their phi) | The label bridge from exp52, kept separate: it says what a difference *means*, the readings above say what it *is*. |
| Groups | One vote per tile or event with at least three disagreement windows (twenty for an event), a one-sided exact sign test that the preregistered direction holds on more groups than not, and the share of groups where it flips | The per-event machinery of exp55 generalised; windows of one tile are not independent draws (the limit below). |
| Preregistration | Each comparison states, before the run, the enrichment or overlap it predicts and the falsification | As for signals: a difference that is merely reported is not a finding. |
| What a difference does not say | Which side is right. The more confident side wins only slightly more often than a coin flip on the disagreement windows, and confidence does not order them (exp58); the package therefore reports both sides and resolves nothing without labels | The disagreement windows are the boundary windows where confidence has run out; deciding between two inferences there is a labelled question. |

### Known limits of these tests

- **The 27 scenes are not 27 independent draws.** They sample fixed fractions
  along eight named rivers, so scenes on one river share its reference
  errors, season and channel morphology. Re-running the headline sign test
  with one vote per river (majority of that river's scenes) gives tiling
  instability **8/8 rivers, p=0.008**, against 26/27 scenes, p=4e-07. The
  result survives clustering; its significance is three orders of magnitude
  weaker than the scene-level figure suggests, and the scene-level p-value
  should not be quoted on its own. Reproduce with
  `uv run --extra geo python exp/summary_transfer.py`.
- **A null is not a demonstration of equality.** Where a signal is reported
  as not beating confidence at p > 0.05 (aligned tile-phase on Sen1Floods11
  Bolivia, 163/187, p=0.22) the claim is that no advantage was shown, not
  that the two are equivalent. No equivalence test has been run.
- **Cross-testbed comparisons change several things at once.** Moving from
  the WorldCover scenes to Sen1Floods11 changes the reference, the task
  (permanent against flood water), the geography, the unit of analysis
  (scene against tile), the input processing (L2A against L1C through the
  L2A path) and the head's training set. That the advantage does not
  transfer is established; *which* of those differences causes it is not,
  and that is the open question exp23-exp25 attack.

Splits are geographic hold-outs. Scene selection is pre-registered before
any scene is fetched (the rule is in
[results/comparisons.md](../results/comparisons.md)); its coordinates have
been cached in `exp/out/rule_candidates.json` since exp25, after an Overpass
mirror failure silently dropped rivers from one run's scene set.

## Evidence tiers

Later tiers override earlier ones where they disagree.

1. **Authoritative — expert labels.** exp18 (Sen1Floods11 hand labels,
   geographic hold-out) for the water task; exp21 (the fine-tuned AWF model
   end to end) and exp16/exp04 (AWF expert points) for the classification
   task. These supersede WorldCover-referenced results wherever they
   conflict.
2. **Authoritative — WorldCover reference.** exp13 on the 27 rule-selected
   scenes (`exp/out/exp13_corrected_stats.csv`), with exp14 supplying the
   mechanism. Claims from this tier are claims about disagreement with a
   weak reference, not about model error.
3. **Directional only.** Single-scene results (exp01-exp08) and exp09's
   seven hand-chosen scenes. Their tile-phase numbers were computed before
   the alignment fix and are not comparable; they are kept in
   [../../exp/NOTES.md](../../exp/NOTES.md), not in the results docs.

## Status terms

| Term | Meaning |
|---|---|
| **supported** | At least one experiment consistent with the claim, under stated conditions |
| **mixed** | Results differ across conditions |
| **partial** | Some evidence; a key condition untested |
| **rejected** | Tested and contradicted |
| **untested** | No experiment yet |
| **blocked** | Requires something unavailable |
| **out of scope (v1)** | Deliberately excluded |

## Related work and positioning

The individual signal families are not new, and the ledger should not be
read as claiming they are:

- Confidence-based map assessment appears in the CEOS WGCV land cover
  validation protocols, as a complement to reference-data assessment.
- Test-time-augmentation uncertainty has been applied to EO segmentation
  (e.g. landslide mapping), following Wang et al. 2019 in medical imaging.
- The Area of Applicability / Dissimilarity Index (Meyer & Pebesma 2021) is
  adopted in spatial statistics via the CAST and waywiser packages, for
  tabular predictor spaces.
- SHRUG-FM (CVPR 2026 EarthVision) performs embedding-space OOD detection
  for EO foundation models.
- Ensemble disagreement is standard uncertainty practice in mainstream ML.
- Boundary-concentrated error is well known in segmentation and land-cover
  validation (trimap and Boundary-IoU evaluation; mixed-pixel effects,
  Foody 2002; Radoux and Bogaert 2017), and is not claimed as new.

**Upstream evaluation this is measured against:**
[rslearn segmentation tasks](https://github.com/allenai/rslearn/blob/master/rslearn/train/tasks/segmentation.py),
the [AWF task config](https://github.com/allenai/olmoearth_projects/blob/main/olmoearth_run_data/awf/model.yaml)
whose classes and split are reused here, and
[olmoearth_pretrain/evals](https://github.com/allenai/olmoearth_pretrain/tree/main/olmoearth_pretrain/evals).

**What is known and what is new.** These verdicts come from a literature search of 27 September 2026
(`exp/out/literature_positioning_2026-09-27.md`), which is not exhaustive, with one verdict changed on 30 September:
the memo lists the head-to-head of the margin against no-model controls on the suite as appearing new, and the red
team of 30 September found those controls near chance there, so it is not claimed below. They replace a paragraph
that said no EO work evaluates selective prediction with risk-coverage curves or uses disagreement as an audit
signal; the related-work page itself lists such work.

- **Review order, ranking errors by confidence.** Known. The model's own confidence is the standard baseline for
  finding misclassified examples (Hendrycks & Gimpel 2017). In land cover, per-pixel confidence maps go back to
  McIver & Friedl 2001, and maps of predicted per-pixel accuracy to Khatami et al. 2017. EO segmentation uncertainty
  has been scored by how well it finds errors (Rey et al. 2025), quality control of land-cover maps has been
  prioritised by uncertainty (Choi et al. 2026), and selective prediction with risk-coverage curves exists for EO
  foundation models (SHRUG-FM). Partly new: the label-free ranking within one map framed as a review order, reported
  as the share of the random-to-perfect gap it closes. Beating the suite's no-model controls is not a contribution,
  because those controls are near chance there.
- **Error estimation.** Known. The estimator is standard (Olofsson et al. 2014); Stehman 2014 covers strata that are
  not map classes, Stehman 1997 shows that clustered samples understate the standard error, and Kumar & Raj 2018
  stratify by a classifier's own confidence to estimate its accuracy. Model-assisted and prediction-powered estimators
  do the same job (Stehman 2009; Angelopoulos et al. 2023; Shirota 2026). Partly new: coverage measured on many
  encoder-task cells for foundation-model maps, and the size of the tile-sampling design effect for such maps.
- **Certified zones.** Known. Keeping the most confident part and bounding its error with probability 1 − δ is
  Geifman & El-Yaniv 2017. The prefix rule is fixed-sequence testing, justified in general by Learn then Test
  (Angelopoulos et al. 2021). Partly new: the calibration sample is a probability design over a map's windows, and
  the output is a certified region of a map.
- **Comparing two maps.** Known. Disagreement maps between land-cover products, used to point reviewers at places
  to check (Fritz & See 2008), and paired accuracy tests on a shared sample (Foody 2004). Partly new: each model's
  confidence attached to where two foundation-model maps differ, in one tool. Little more can be defended.

**The contribution, stated as measured scope.** The statistical parts all exist. This repository joins them in one
tool that reads a model's scores after the fact, and measures the result at scale. The margin's ranking was graded
over 16 encoders, each encoder's 20 to 24 tasks, and 10 probe seeds: it beats a random order in every one of the
3,560 cells, and on OlmoEarth Base it gets a median 0.68 of the way from a random to a perfect order. The interval's
coverage was graded on 111 encoder-task cells. Labels collected by tile break the naive interval, with a median design effect near 3. And
exp88 gives a deployment warning: a model run on an input combination it was not trained on can be confidently wrong.
The perturbation-based instability signal of exp13 and exp14 is not part of the contribution. It beat confidence
against WorldCover and failed against expert labels (exp18, exp21), and the technique ledger rates it mixed.
<!-- claim:exp79-margin-beats-random-everywhere --> <!-- claim:margin-takes-two-thirds-of-the-ranking-headroom --> <!-- claim:exp79-estimation-findings-hold-on-every-encoder --> <!-- claim:tile-sampling-breaks-the-naive-interval --> <!-- claim:exp88-matched-head-ranks-normally -->
