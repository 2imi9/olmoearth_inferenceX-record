# Where the lead holds: the headline by group, with its confound, its multiplicity and its spread (exp84 preregistration)

Written on 23 September 2026, before the fifteen other encoders' exp79 exports exist; runs as
`exp/exp84_where_the_lead_holds.py` on each export as it lands. The numbers land in the ledger like every other
experiment.

## The gap

The record's headline, *the margin beats the best no-model control on 24 of 24 tasks for OlmoEarth Base and on
at least 75% of the tasks each of sixteen encoders carries*, is reported per task and per source. The field's
rules for stating such a result (`docs/plan/methodology_redesign.md`, section 1) ask for more: report it within
capability groups so a user can find their own case (GEO-Bench-2); report the confound between a failure-
detection score and the classifier's accuracy (Jaeger et al. 2023; Steinmetz et al. 2026); control multiplicity
when many comparisons are raced (Claros Olivares and Brockmeier 2025); and report the spread of the result across
spatial blocks, not only its pooled value (Ramos-Pollán et al. 2024). None of the four is on the page today.

## Disclosure, before the predictions

While preparing this page I computed the following from committed summaries (seed 0, 356 encoder–task cells)
and from OlmoEarth Base's ten probe seeds in the RTX engine copy of exp79 (`exp/out/exp79_engine/`):

| quantity | seed 0, 356 cells | OlmoEarth Base, ten seeds |
|---|---|---|
| Spearman(accuracy, margin AUROC) | 0.61 | 0.74 to 0.80 |
| Spearman(accuracy, lead over the best control) | −0.46 | −0.40 to −0.53 |
| median lead by sensor group (S2, Landsat, S1+S2, S1) | 0.109, 0.118, 0.132, 0.136 | range across groups 0.037 to 0.050 |
| the ten losses | all on CropHarvest Togo S1 (306 units) or Nandi S1 (accuracy ≤ 0.32) | Base's smallest lead +0.0001 (seed 3, Togo S1) |
| Holm over the 19 sign tests (16 encoders, 3 alternatives) | every test survives; largest adjusted p 2.2e-4 | — |

These are therefore **not predictions** and are recorded below as disclosed facts. The predictions are on what
has not been seen: the fifteen other encoders' ten seeds, and the block spread, which no one has computed.

**Addendum, later the same night, before any of the fifteen was graded.** OlmoEarth Base's graded (B200) export
landed after this page was written: its ten seeds give Spearman(accuracy, AUROC) 0.74–0.80, Spearman(accuracy,
lead) −0.39 to −0.51, no losses, and a sensor-group range of the median lead up to **0.070**, above the 0.037–0.050
the RTX engine copy showed and above P3's bar of 0.06. The audit traced the whole excess to one cell: at seed 4
the S1 group's median is AWF Sentinel-1's lead, a 200-unit task whose lead differs by 0.021 between the two
engines (0.177 on the B200, 0.157 on the RTX); the other nine seeds give 0.037–0.042 on both engines. P3's bar
stays where it was written; a five-task group's median is one task's number, and the record will say so if the
bar fails for that reason. Also disclosed: the block spread on Base's segmentation tasks (descriptive, now computed) has the margin's
lead over the class-rarity control positive in 98–100% of the 50 tile groups on six tasks and in 79% of the 34
groups with any error on MADOS, whose 10th-percentile lead is −0.094.

## The estimands

Per (encoder, task, seed) from exp79's exports: the probe's test accuracy, the margin's AUROC for its errors, and
its lead over the best no-model control (the smaller of the class-rarity and embedding-distance controls'
excess AURC minus the margin's). Groups: sensor (S1, S2, Landsat, S1+S2, from the task name; So2Sat is S1+S2 and
ForestNet Landsat), unit count (≤ 306, ≤ 1,000, larger) and class count (2, 6–10, ≥ 12). The confound is the
Spearman correlation over an encoder's tasks at one seed. The block spread, on OlmoEarth Base's seven
segmentation tasks from exp78's export: the tiles in export order cut into 50 consecutive groups of equal window
count, and in each group the margin's excess AURC against the class-rarity control's (the one control computable
per group from the export; the embedding-distance control needs the bank). Groups are chip-order groups, not
geographic blocks, and the record says so.

## Predictions (on the fifteen encoders' seeds)

**P1, the confound is a property of the protocol, not of one encoder.** For every one of the fifteen encoders,
under every one of its ten seeds, Spearman(accuracy, margin AUROC) over its tasks is at least **0.4** and
Spearman(accuracy, lead) is at most **0**. *Why:* a probe that is right more often has fewer errors to rank and
ranks them more cleanly (AUROC rises with accuracy), while the excess AURC's lead shrinks on clean tasks because
there is less to gain; Base shows 0.74–0.80 and −0.40 to −0.53, the 356 seed-0 cells 0.61 and −0.46; 0.4 leaves
room for encoders carrying 20 or 21 tasks. *What makes it fail:* an encoder whose ranking quality does not follow
its accuracy, which would be worth naming.

**P2, the thin cells are small or bad cells, not a sensor.** Every (encoder, task, seed) on which the margin
loses to the best control is a task with at most **306** units or a probe accuracy at most **0.40**. *Why:* the
ten seed-0 losses sit on exactly those two tasks, and a loss elsewhere would mean a new mechanism.

**P3, no sensor is a hole.** For every encoder and seed, the range across sensor groups of the median lead is at
most **0.06**. *Why:* Base's is 0.037–0.050; encoders carrying fewer tasks per group are noisier, so 0.06.

**P4, the bar survives Holm.** Recomputed from the seeds: for every encoder, the per-seed sign test of "margin
beats the best control" survives a Holm correction over the fifteen encoders at every seed. *Why:* the seed-0
p-values are at most 1.1e-4 against a Holm bar of 0.0033.

**Descriptive.** The block spread on Base: per segmentation task, the share of the 50 tile groups in which the
margin's excess AURC is below the class-rarity control's, and the 10th percentile of the lead across groups. The
group tables (sensor, unit count, class count) with medians and counts, per encoder and pooled.

## Independent check before anything is recorded

A test recomputes one encoder's confound and group medians from its export by a second route (pandas-free,
explicit loops) and the Holm adjustment with an independent implementation; then an adversarial read.

## Cost

None beyond exp79's exports. Runs per encoder as each export lands; the record is written when all fifteen have.

## Result, 24 September 2026

Graded on the fifteen encoders' 150 encoder-seeds and recomputed by an independent route
([comparisons](../results/comparisons.md#where-the-lead-holds-by-group-with-its-confound-and-its-multiplicity-exp84)).
P2 and P4 hold. P1 fails as written: the confound is positive on all 150 encoder-seeds but below 0.4 on 14 (Galileo
Nano under every seed). P3 fails as written: the sensor-group range exceeds 0.06 on 48, though every group's median
lead is at least +0.062 and Sentinel-2 is the lowest group on 125. Two statements in the disclosure above were
wrong, and neither moves a verdict: the record's ten seed-0 losses are on Togo Sentinel-1 (4), Togo Sentinel-2+1
(2), Togo Sentinel-2 (1) and Nandi Sentinel-1 (3), not on Togo Sentinel-1 or Nandi Sentinel-1 alone; and Base's
"+0.0001 (seed 3)" is the RTX engine copy's value, the graded B200 export's being +0.0017 (seed 5).
