# Can raters from different families estimate a map's accuracy without labels? (exp83 preregistration)

Written before the run, on 23 September 2026. Runs as `exp/exp83_consensus.py`; the numbers land in the ledger
like every other experiment.

## The gap

The record rejected label-free reliability estimation within one model family: Dawid–Skene over OlmoEarth
Nano/Tiny/Base on AWF overestimated every model (+0.106, +0.114, +0.059) and inverted their order, because a
family errs together and agreement on errors is read as competence (exp07; `docs/results/signals.md`). The same
page names the designed fix, an out-of-family rater, and calls it untested. The field's version is agreement-on-
the-line and disagreement-as-error (Baek et al. 2022; Jiang et al. 2022), whose stated condition is diversity
across the raters. The suite's multi-encoder masks (exp63: six encoders on MADOS and PASTIS S2; exp57: eight on
Sen1Floods11) are exactly that panel, with the labels held out to grade. Two questions a map user asks that the
record cannot answer today: *how accurate is each of these maps without labels*, and *which encoder to deploy on
a region with none*.

## The estimands

Per task and encoder *k*: the true accuracy `a_k` over the windows every panel member predicted (labels grade
only); the **Dawid–Skene estimate** `â_k` (`oe_inferencex.evidence.dawid_skene`, the reliability of rater *k*, EM
over the panel's hard votes, labels untouched); the **hidden share** `h_k = (â_k − a_k) / (1 − a_k)`, the share of
*k*'s error mass that agreement hides (0 = the estimate is exact, 1 = every error is read as correct), which is
what exp07's gaps become once divided by each model's error rate (0.43, 0.58, 0.32); the **majority-shared error
share** `m_k`, the share of *k*'s errors on which a strict majority of the *other* raters make the same wrong
call, a population fact computed from labels; the mean pairwise disagreement `d̄_k` of *k* with the others (the
GDE reading of error); and the encoder ranking by `â_k` against the ranking by `a_k`.

Panels: MADOS and PASTIS S2, six encoders (OlmoEarth Base draw 0, Galileo Base, CROMA Base, TerraMind Base, Clay
Large, AnySat) over the windows with a reference label; Sen1Floods11, eight encoders (those and Satlas Base and
Panopticon) over the windows all eight predicted. The within-family comparator: Dawid–Skene over OlmoEarth Base's
three probe draws on MADOS and PASTIS S2.

## Disclosed population facts (labels used only here, before the run)

| task | encoders and true accuracy | majority-shared error share `m_k` | OlmoEarth draws' pairwise disagreement |
|---|---|---|---|
| MADOS (22,598 windows, 15 classes) | OlmoEarth 0.926, Galileo 0.929, CROMA 0.916, TerraMind 0.939, Clay 0.856, AnySat 0.823 | 0.48, 0.42, 0.45, 0.57, 0.16, 0.20 | 0.003–0.006 |
| PASTIS S2 (458,638, 19 classes) | 0.810, 0.766, 0.782, 0.772, 0.673, 0.779 | 0.66, 0.62, 0.64, 0.64, 0.40, 0.62 | 0.002 |
| Sen1Floods11 (563,969, binary) | OlmoEarth 0.913, Galileo 0.914, CROMA 0.913, TerraMind 0.911, Clay 0.911, Satlas 0.883, AnySat 0.910, Panopticon 0.909 | 0.86, 0.86, 0.85, 0.87, 0.86, 0.59, 0.83, 0.82 | — |

So the panel is diverse on MADOS, half-shared on PASTIS, and nearly unanimous in its errors on Sen1Floods11,
where eight frozen encoders fed the same Sentinel-2 chips miss the same water. The weak encoders (Clay, AnySat,
Satlas) err on their own. Those facts fix the predictions.

## Predictions

**P1, the hidden share is the majority-shared share.** For every encoder on every task, `h_k` lies within
**0.10** of `m_k`. *Why:* Dawid–Skene infers the truth from the panel's plurality; an error the majority shares is
invisible to it and counted as correct, an error the encoder makes alone is seen; the residual is the
soft-vote weighting and the confusion-matrix smoothing, which a tenth covers. *What makes it fail:* EM settling
on a labelling that is not the plurality (a strong rater dominating), which would show as `h_k` far from `m_k` in
either direction; or the estimate being driven by the prior rather than the votes on the 19-class task.

**P2, the order survives where the panel is diverse and not where it is unanimous.** The Spearman correlation
between the rankings by `â_k` and by `a_k` is at least **0.8** on MADOS and on PASTIS S2, and the estimated best
encoder is the true best on both; on Sen1Floods11 it is **below 0.8**. *Why:* on the two multi-class tasks the
weak encoders' idiosyncratic errors are seen and the strong encoders' shared errors are hidden roughly alike, so
the order is kept while every level is inflated; on Sen1Floods11 the seven strong encoders sit within 0.006 of
one another in accuracy and share 82–87% of their errors, so the estimate cannot resolve them. *What makes it
fail:* an inversion among the strong encoders on MADOS or PASTIS (TerraMind's 0.57 shared share is the largest,
so it is the one a label-free user would be most likely to over-rank).

**P3, disagreement reads error, with the same blind spot.** The Spearman correlation between `d̄_k` and the true
error rate `1 − a_k` is at least **0.8** on MADOS and PASTIS S2; and `d̄_k` understates the error rate on every
encoder and task (the GDE estimate is biased low under shared errors), by a factor that is largest on
Sen1Floods11. *What makes it fail:* an encoder whose disagreement with the panel exceeds its own error rate,
which means the panel, not the encoder, is wrong there.

**Descriptive.** The within-family comparator on MADOS and PASTIS S2 (three OlmoEarth draws that disagree on
0.2–0.6% of windows: the hidden share it gives is the reference point exp07 measured at 0.32–0.58 on AWF); the
agreement-with-majority estimator beside Dawid–Skene; per-class confusion matrices Dawid–Skene infers for the
best encoder against the true ones.

## Independent check before anything is recorded

A test in `tests/test_exp83.py` runs a Dawid–Skene EM written independently (no call into the package) on a
synthetic panel of conditionally independent raters with known confusions and shows both implementations
recover the true accuracies within 0.02; the majority-shared share is checked against a brute-force count. Then
an adversarial read by a separate agent on the real masks.

## Amendment, 23 September 2026, from the independent audit, written after the run and before the record

The audit reproduced every number with its own loader and its own EM, and found one thing that blocked the
record: **the package's Dawid–Skene stopped at 50 iterations and the 15- and 19-class panels need 224 and
261**, so the first run's estimates were snapshots (OlmoEarth's hidden share on MADOS 0.45 at 50 iterations,
0.28 at convergence). `reliability.dawid_skene` now runs to its stopping rule (largest posterior change below
1e-6, cap 1,000) and reports whether it converged; the run below is the converged one. No verdict changed.

**All three predictions fail, and the audit named the mechanism this page missed.** Dawid–Skene's reliability is
the mean posterior mass on the rater's own vote, so a rater is *credited* for errors the panel shares (read as
right) and *debited* for being right where a majority of the others agree on a wrong label (read as wrong). This
page had only the credit. On PASTIS the credit equals the majority-shared share to within 0.06 and the debit
(0.14–0.34) accounts for the whole gap; on MADOS the true best encoder, TerraMind, is right while a strict
majority of the others share a wrong label on 0.36 of its error count, and its hidden share is 0.12 against a
shared share of 0.57. The inferred truth is the plurality (98%, 89% and 99% of windows on the three tasks), it
never beats the best single rater, and the estimator is near-certain of it.

Stated from the audit: the MADOS best-encoder inversion is a coin (chip bootstrap over the 841 labelled tiles:
OlmoEarth's estimate exceeds TerraMind's by 0.001 with a standard error of 0.010, P = 0.57, while the true gap
between the top two is 2 standard errors); the PASTIS identification is real (19 of 20 resamples); on
Sen1Floods11 the true best is itself barely resolved (0.0015, SE 0.0008). P3 failed on two of three clauses:
disagreement ranks error on MADOS (0.83) and not on PASTIS (−0.03), where the most accurate encoder disagrees
most with the strong raters because it is right alone; and disagreement *overstates* error on MADOS (five of six
encoders) because the two weak encoders' idiosyncratic errors inflate everyone's disagreement. The
within-family comparator is the unanimous limit (hidden 0.94–0.995), not a replication of exp07's three
different models (0.32–0.58), and is recorded as that. The first version of `tests/test_exp83.py`'s correlated-
raters test built its shared errors unanimous, so it exercised only the credit and would have passed whatever
tonight found; a test with a strong rater right alone, which the debit fails, is added.

## What would invalidate the run

- Labels entering the votes, the panel, or the EM in any way; they grade `a_k`, `m_k` and the rankings only.
- Grading on windows some panel members did not predict; the population is the intersection.
- Treating an inflated but well-ordered estimate as an accuracy: the record must present `â_k` with `h_k`.

## What changes in the tool if the predictions hold

`evidence.dawid_skene` gains a documented use: with raters from different families the *order* of maps is
recoverable without labels and the *level* is not, by a hidden share the tool can state from the panel's
shared-error structure; with one family it recovers neither. The `compare` module's "why errors are shared"
gains the number that says how much of a map's error a panel could never see.

## Cost

No cluster; both mask files are committed. Dawid–Skene over 564,000 windows and eight raters is seconds.
