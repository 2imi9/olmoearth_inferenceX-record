# Methodology redesign against the field's rules (23 September 2026)

How the field examines the inference of foundation vision models, what rules it has settled on, where this
record follows them, and what changes follow. Written from a five-angle literature sweep run on the night of
22–23 September (59 papers, 2021–2026, abstracts and method sections; the full list with what each examines is
kept beside this page as `exp/out/methodology_sweep_2026-09-23.json`) and a synthesis against the record. The
sweep's angles: evaluation protocol for failure detection and probing; label-free accuracy estimation; label-
efficient statistically valid evaluation; examination of Earth-observation foundation models; systematic error
discovery and auditing frameworks.

The standing filters apply to everything below: a proposal must tell a map user something they cannot know today,
stay inside the audit (labels grade or calibrate, nothing trains), and have headroom. Two premises the sweep
confirmed rather than shook: plain confidence is the field's default winner for ranking a model's own errors
(Jaeger 2023; Claros Olivares & Brockmeier 2025; Steinmetz 2026), and no single score dominates across tasks. The
record's 24-task, 16-encoder result is the Earth-observation instance of that finding, not an exception to it.

## 1. The field's rules, and where the record stands

Forty-two rules were extracted; four the record follows in full, seventeen do not apply to a label-free audit of a
frozen probe, and twenty-one are followed partly or not at all. The twenty-one, grouped by what they would change:

**Rules about how a ranking claim is stated.**

| rule | set by | status | what is missing |
|---|---|---|---|
| Report the confound between failure-detection score and classifier accuracy explicitly | Jaeger 2023; Steinmetz 2026 | partial | excess AURC is the statistic, but the correlation of the margin's AUROC with probe accuracy over the (task, encoder) cells is never reported |
| Control multiplicity when many rankers are raced; treat the winning family as a property of the representation | Claros Olivares & Brockmeier 2025 | partial | one-sided sign tests per alternative, no family-wise control; Holm over the nineteen tests changes no verdict (smallest p 1.1e-4 against 0.0026) and should be on the page |
| Report geospatial results within capability groups (sensor, resolution, temporality) with seed repetition | Simumba et al. 2025 (GEO-Bench-2); Ramos-Pollán et al. 2024 | partial | seeds: exp79 tonight; groups: only classification against segmentation, no sensor or split-type table |
| Splits must be region-disjoint and results reported per class, because foundation models collapse to frequent classes under regional shift | Shang et al. 2026 | partial | split type is not on the task cards; exp70 holds train–test gaps (So2Sat 0.26, ForestNet 0.23) that mark the shifted ones |
| Validate an uncertainty method by downstream use, ablate aggregation separately, grade calibration beside ranking | Kahl et al. 2024 (ValUES) | partial | downstream uses graded (review order at budgets, estimation); aggregation ablated (exp76); calibration of the margin never graded on the suite |
| Robustness is the drop from each model's own clean baseline under physically grounded product degradations | Faruk et al. 2026 (RSPDBench); Lehmann et al. 2026 | not done | every grade is on clean inputs; perturbations appear only as candidate signals |
| Grade misclassification and out-of-support inputs jointly | Li et al. 2026 | not done | no out-of-support stratum exists in Ai2's splits |

**Rules about what a map user is owed.**

| rule | set by | status | what is missing |
|---|---|---|---|
| A map user is owed user's accuracy, producer's accuracy and error-adjusted area per class with standard errors | Stehman & Wagner 2023; CEOS LPV protocol 2025; Olofsson 2014 | not done | `estimate` returns one error rate with an interval; no per-class quantity anywhere |
| Plan the sample size backwards from a target precision | Adebayo et al. 2026 | not done | the tool grades a given budget; exp78's half-widths at three budgets are the ingredients of a planner |
| State the reference error rate and bound the estimate for it; bias grows for rare classes | Foody 2024; CEOS 2025 | partial | references treated as fallible in exp23 and exp68; `estimate` treats the reviewer's labels as exact |
| Say which outputs are an accuracy assessment in the protocol's sense; a confidence layer is not one | CEOS LPV 2025 | partial | Usage says `estimate` needs labels; it does not say in the protocol's words that the rankers are review-ordering aids |
| Beyond an estimate with an interval, give a threshold with a guaranteed risk bound | Angelopoulos et al. 2022, 2021; Bates et al. 2021 | **done tonight** | exp80, below |
| Use predictions on the unlabelled units through a debiased difference estimator with a tuned coefficient, so the interval is never wider than the classical one | Angelopoulos, Duchi & Zrnic 2023 (PPI++); Fisch et al. 2024 | not done | exp78's plan lists the arm E3d; it was never run, and no document claims it was |
| A label-free accuracy estimate is graded by its error against the truth across many (model, test set) pairs and shipped with its failure mode (prevalence shift) | Garg et al. 2022 (ATC); Guillory et al. 2021; Lu et al. 2023; Fluehmann et al. 2025 | not done | the tool refuses the question; the per-unit exports show why: the map's mean confidence overstates its accuracy by a median 0.061 and up to 0.184 on the suite |
| Estimate the confusion matrix, not one accuracy, when users need per-class numbers | Fluehmann et al. 2025 | not done | same as the first row of this table |
| With a family of models, unlabelled agreement lies on the accuracy line; random head initialisation is the diversity that works | Baek et al. 2022; Saxena et al. 2024 | not done | the ten-seed family exists after tonight, but only seed-0 decisions are exported |
| Detect deployment shift label-free from the frozen feature space, graded against an oracle-labelled shift | Ekim et al. 2024 (TARDIS) | not done | no labelled shift stratum exists to grade it against; parked |

**Rules about a defensible "why".**

| rule | set by | status | what is missing |
|---|---|---|---|
| A coherent slice is not a verified hypothesis; the named group must show elevated error on held-out data | Johnson et al. 2023 | partial | cue enrichments were measured on one testbed with clustered intervals and are quoted on every map |
| Enumerate attribute conjunctions with a per-cell statistic and a noise model for the attribute | Chen, Zhao & Xu 2025 (HiBug2) | partial | cues are reported one at a time |
| In segmentation the systematic error is a (predicted, reference) confusion pair on a coherent sub-group | Singh et al. 2024 | not done | no confusion-pair report in `explain` or `compare` |
| Report false-confidence concentration as a property | Cenacchi et al. 2026 | not done | its ingredients each lost as rankers (exp73); the spatial analogue is exp78's design effect |

## 2. What changes, in order

Each item passes the three filters or says which it fails. Cost is stated in what it needs, not in hours.

1. **A trusted zone with a guarantee (exp80, run and graded tonight; `docs/plan/trust_zone.md`).** The record
   said "an accuracy needs a coverage" and could not give one. Now, from 300 random labels, the tool certifies the
   largest most-confident share of a map that is wrong at most α of the time, with the statement failing on at
   most δ of draws. Two rules: a monotone-prefix rule (Bates et al. 2021) and an assumption-free Bonferroni rule
   (Angelopoulos et al. 2021), both with exact hypergeometric tests, beside the plug-in a reviewer would use
   unaided. Ships as `estimate.certify_zone` and `oe-inferencex certify` once the audit clears it.
2. **Per-class accuracy assessment from the same sample (exp81, preregistration next).** User's accuracy,
   producer's accuracy and error-adjusted area per map class with design-based standard errors, from the sample
   `sample` already draws; a class-aware design beside the confidence design; and the budget a target precision
   needs. This is the field's definition of an accuracy assessment; the tool has been answering a narrower
   question. Needs per-unit reference labels, which exp79's exports carry (exp78's segmentation exports do not).
3. **Where the lead holds (exp82).** The headline re-tabulated by sensor, task family, class count, unit count and
   split type; the accuracy confound as a correlation with an interval; Holm across the raced alternatives; the
   spread of the margin's lead across tile blocks. Local, from committed artifacts and exp79's seeds. It qualifies
   the claim rather than changing it, and a user can locate their own case in it.
4. **A verification rule for `explain`.** Every cue's enrichment recomputed per task on held-out units with a
   clustered interval; the library value quoted only where it survives; confusion pairs for multi-class maps.
   Needs the exports' decisions and grids; local.
5. **The model-assisted arm (PPI++ / stratified PPI) in exp78's harness.** Bounded by exp78's own ceiling (no
   saving above 1.8× against a random sample), so its value is precision and a record that says what was run. Local.
6. **A label-free error-rate estimate with its failure mode stated (ATC, DoC).** The sweep's rule is to grade it
   by error against the truth across many cells and to ship it as a pilot, never as the assessment. The honest
   fact tonight is the calibration gap above; whether a fitted threshold repairs it needs the probe's validation
   split outputs, which the exports lack. One cluster job per encoder when the queue allows.

Not taken, with the one decisive reason each: degradation stress axes (need re-encoding under product defects);
joint out-of-support grading (no such stratum in the suite); TARDIS-style shift flags (nothing labelled to grade
against); agreement-on-the-line (needs all ten seeds' decisions exported; queued behind the above); embedding-
geometry diagnostics (no route to a statement a user acts on, and exp73/exp77 say feature statistics carry no
error information here); sequential and active designs (exp78's confidence design already sits within 0.0007
half-width of the oracle allocation, so there is no allocation headroom to claim); label-free score aggregation
(another ranker; exp65's fusion gained 0.0006); training-data influence (a developer's question).

## 3. What the night's runs said

Each ran, was audited by a separate agent on the real files, and entered the record with its failures stated;
the sections are in `docs/results/comparisons.md`.

- **exp80, the trusted zone.** The guarantee held on all 112 cells (largest violation 0.080 for the prefix rule,
  0.0045 for Bonferroni, bound 0.120), including the fourteen tasks where the prefix rule's assumption is broken by
  up to 0.034. At 300 labels and α = half the map's error rate: a zone on most draws on 14 of 21 tasks, median
  coverage 0.50 against an oracle 0.60; the plug-in violated its own α on up to 56% of draws. Two secondary
  predictions failed as written (a near-census arithmetic; a bar written for 24 tasks where 21 have a cell).
- **exp81, per class.** The field's Wald interval collapses to a point whenever a class shows no sampled error
  (107 of 522 cells below 0.93 on the 24 tasks, the worst at 0.02); the shipped forms leave 14 of 628 short, the
  lowest 0.879, each with a named mechanism, and the random-design user's accuracy is now an exact hypergeometric
  interval that covers at least 95% by construction. Stehman and Wagner's warning did not hold: the confidence
  design narrows the per-class intervals on 19 of 21 tasks. The audits found the producer's accuracy lacking its
  finite-population correction, the design comparison reading the wrong cells, and, a night later, a defect in
  the finite-population Wilson form that a self-referential exemption had hidden; all fixed before recording.
- **exp82, the "why" verified.** The boundary cue is real on all seven segmentation tasks (risk ratio 2.1 to
  10.5) but its quoted enrichment runs from 1.24 to 8.13 and is set by the map's fragmentation, not its class
  count as the page predicted; the Bolivia value lies outside every task's interval; the conjunction is purer than
  either cue everywhere; boundary-first loses to confidence on the suite's flood split. The library now carries
  the range and reports the map's boundary prevalence.
- **exp83, consensus across families.** All three predictions failed: Dawid–Skene infers the panel's plurality,
  credits an encoder for the errors the panel shares and debits it for being right alone, so the level is
  inflated everywhere and the order is recovered only where the true gaps are large (PASTIS, not MADOS or the
  flood split). The audit found the estimator stopping at 50 iterations where 224 and 261 were needed; fixed
  before recording. The front page's sentence that an error rate needs a reference stands, with a second reason.
- **exp85, the model-assisted arm.** The map's confidence as an error predictor, corrected by the labels, is
  exact and honest and narrows the interval 7–16% under a random sample, with a coefficient that lands at 1; once
  the confidence design has stratified it adds 2% of width for half a point of coverage, so by the page's own rule
  it does not ship. Two of my predictions failed (the calibration-gap argument was a category error; the 1.8×
  bound was an exp78 prediction that had already failed), and the audit found the tuning floor letting the
  per-stratum coefficient explode on small strata; fixed before recording.
- **The generality reruns (exp80, exp81, exp82 on Clay Large, Copernicus-FM, AnySat and CROMA Base's seed-0
  exports).** The guarantee held on every cell of every encoder; the field's Wald interval failed and the shipped
  form nearly held on each; the boundary cue's ordering of the seven tasks by enrichment was the same under all
  five encoders. One exception, and it is the mechanism in its extreme form: AnySat's probe grid on the two m-*
  tasks is four times coarser, 97% of its cashew windows sit on a boundary, and the cue enriches nothing there.
  The boundary cue is a property of the window grid as much as of the map, and `explain` now says so when a map's
  prevalence is that high; whether Base's own export saturates at that grain is the addendum to
  `docs/plan/cue_verification.md`.
- **exp84** is preregistered on the fifteen other encoders' seeds and runs as their exports land; exp79's
  OlmoEarth Base export passed the record's gate on all 24 tasks on the B200 and the margin's lead survived all
  ten seeds on every task, with the GPU's contribution measured beside the seed's (`exp/out/exp79_engine/`).

## Addendum, 30 September 2026 (nothing above changed)

Item 1 calls the prefix rule a monotone-prefix rule and the Bonferroni rule assumption-free. The prefix rule is
fixed-sequence testing: the order and the grid are fixed before any label is read, so it holds δ on any map, whatever
the shape of its error rate. Both rules are valid, and neither is the more powerful in general: over exp80's 112
cells Bonferroni certified a zone on more draws on 25 and the prefix rule on 72. See the addendum of the same date in
[trust_zone.md](trust_zone.md).
