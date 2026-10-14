# Is the record bigger than its own seed noise? (exp79 preregistration)

Written before the run, on 22 September 2026. Runs as `exp/exp79_seed_floors.py`; the numbers land in the ledger
like every other experiment.

## The gap

The suite's headline sentences rest on one probe seed. exp70 scored OlmoEarth Base on 24 tasks at seed 0; exp74
scored fifteen other encoders the same way, seed 0; and the verdict that carries into every conversation with Ai2
— *the margin beats the best no-model control on at least 75% of tasks, on 16 of 16 encoders* — has never been
reseeded. A reseed floor exists on seven segmentation tasks only (exp77). The smallest leads on the record are

| task | margin lead over the best control | units |
|---|---|---|
| CropHarvest Togo Sentinel-1 | +0.0157 | 306 |
| MADOS | +0.0165 | 22,598 |
| Nandi Sentinel-1 | +0.0168 | 825 |
| EuroSAT | +0.0196 | 1,000 |

Nobody knows whether those are wins or noise. Also unknown: whether the design effect that breaks a reviewer's
naive interval (exp78, P2) is a property of the task or of OlmoEarth, since exp78 exported one encoder.

Two questions, one set of embedding loads. Every load serves ten probe seeds, and the seed-0 fit also writes the
per-unit export exp78 needs for the other fifteen encoders.

## What is measured

For every encoder of Ai2's published embedding suite (16) and every task it carries (24, or 20–21 for the
outside families, as exp74 records), the suite's own probe is fitted at seeds 0 to 9, and each seed is scored
exactly as exp70 scores it: excess AURC and AUROC of the margin, entropy and one-minus-top-1, beside the two
no-model controls. The embedding-distance control is seed-independent and computed once per (encoder, task);
class rarity depends on the decision and is recomputed per seed, as exp70 does. At seed 0, the per-unit
quantities (`margin`, `p1`, `dec`, `err`, packed validity, tile grid) are written in exp78's format.

**Gate G, before any prediction is graded.** At seed 0, every (encoder, task) reproduces the test accuracy the
record holds — `exp70_summary.json` for OlmoEarth Base, `exp74_summary.json` for the rest — to the four decimals
the record states. An encoder that fails the gate is reported and its seeds are not comparable to the record;
the seeds are still recorded, because they still answer the noise question for that encoder against itself.

## Predictions

Thresholds are derived from the record, and the reasoning is written down so a wrong threshold is attributable.

**P1, the headline survives reseeding.** Under each of the 10 seeds, exp74's bar — the margin beats the best
control on at least 75% of the tasks the encoder carries, one-sided sign test p < 0.05 — holds on **all 16
encoders**. *Why all 16 and not a hedge:* at seed 0 the closest encoders to the bar are galileo_tiny and
satlas_base at 19 of 21, four wins above it; every other encoder has five or more. A reseed would have to flip
four wins to losses on one encoder to fail this, and the four smallest leads on the record are all on OlmoEarth
Base, which sits at 24 of 24. *What makes it fail:* small-n classification tasks where a probe at a different
seed lands on a different decision set, moving the class-rarity control as well as the margin.

**P2, which wins are noise.** On OlmoEarth Base, the margin beats the best control under **all 10 seeds** on at
least **20 of 24** tasks, and every task that flips on any seed is named in the record with its lead's seed
spread. *Why 20:* the four smallest leads above are the plausible flips; the exp77 reseed spread of accuracy on
the segmentation tasks was 2e-06 to 4.9e-04, so a lead of 0.016 on a task of 22,598 windows is not expected to
flip, but 0.016 on 306 units may. *What makes it fail:* more than four tasks flipping, which would mean the
"24 of 24" sentence about OlmoEarth Base is a seed-0 fact.

**P3, the encoder ordering is stable in its sets, not its order.** The per-encoder median headroom
(`headroom_by_encoder.json`'s statistic, recomputed per seed) has Spearman rank correlation **at least 0.9**
with the seed-0 ordering under every seed; the seed-0 top three (olmoearth_large 0.693, galileo_base 0.680,
olmoearth_base 0.680) contain the best encoder under every seed; and the seed-0 bottom two (anysat 0.566,
satlas_base 0.551) contain the worst under every seed. *Why sets:* the gap between first and second is 0.013,
which per-task seed noise of order 0.01 in headroom units can plausibly swap once it is a median over 24 tasks;
predicting the exact order would be predicting a coin. *What makes it fail:* an encoder from the middle of the
table reaching the top or bottom under some seed, which would mean the record's "the ceiling belongs to the
task, not the model" reads too much into seed-0 medians.

**P4, the naive interval's failure belongs to the task.** With the seed-0 export, exp78's estimation study is
rerun on every encoder's segmentation tasks at B = 300: on at least **13 of 16** encoders, the naive
tile-sampled interval covers below 0.85 on all but at most two of the segmentation tasks that encoder carries,
and the median design effect over those tasks is at least 2.0. *Why 13:* OlmoEarth Base gave 6 of 7 with
m-cashew-plant the exception at a design effect of 1.17; an encoder whose errors on some task are dense enough
to be spatially diffuse could add a second exception, so two are allowed per encoder, and three encoders are
left for surprise. *Why 2.0 rather than exp78's 2.5:* Base's median was 2.94 on seven tasks; encoders that
carry fewer segmentation tasks have a noisier median. *What makes it fail:* the design effect collapsing on
encoders other than OlmoEarth, which would make exp78's warning a fact about one model.

**P5, the estimator is honest for every encoder.** Under a simple random sample of 300 windows, the
design-based interval's Monte Carlo coverage is at least **0.93** on every (encoder, segmentation task) cell
with 2,000 draws — or, where a cell sits below 0.93, it is within **0.01** of the exact coverage of the same
Wilson interval at that population size, error count and budget, computed by enumeration over the
hypergeometric draw. *Why the second clause:* exp78 put MADOS exactly at 0.933 and named Wilson's discreteness
as the reason in advance; across 112 cells that reason will recur, and the honest test of an estimator is
against what its interval can achieve at that (N, K, B), not against a round number. *What makes it fail:* a
cell whose Monte Carlo coverage sits more than 0.01 below the exact value, which is a bug in the estimator or
the sampler and nothing else.

## What would invalidate the run

- **Gate G failing on an encoder and the seeds being graded anyway as comparable to the record.** They are
  graded against that encoder's own seed 0 only, and the encoder is named.
- **A seed that changes the valid-window set.** Validity is a property of the labels, not the probe;
  `same_valid_windows` is asserted per (encoder, task) across seeds.
- **Counting encoders that carry fewer tasks as if they carried 24.** Every count is per encoder over the tasks
  it carries, as exp74 counts, and the number carried is beside it.
- **Reading a P3 set as an order.** The record may say "the best is one of three"; it may not say which.
- **The per-unit export for fifteen encoders is about 700 MB and is not committed.** It lives in the cluster
  home directory, which survives the scratch purge, with a manifest of per-file SHA-256s committed beside the
  summaries; the estimation stage is rerun locally from a fetched copy and the summary it writes is what the
  record cites.

## Cost

Sixteen GPU jobs, one per encoder, on `b200-batch`, after one setup job that updates the checkout so the
sixteen never touch git concurrently. Per encoder: 24 embedding loads, 240 probe fits, 24 control computations.
exp77 fitted five probes per segmentation task on a CPU node in 1 h 40 m for seven tasks; on a GPU the fit is
the small part and the load is the large one, so about one hour per encoder is the estimate. The estimation
stage (P4, P5) runs locally on numpy from the fetched export. No new dependencies; fp32 throughout.

## First reading, 23 September 2026

OlmoEarth Base's export landed first and was recorded on its own
([comparisons](../results/comparisons.md#is-the-record-bigger-than-its-own-seed-noise-first-reading-olmoearth-base-and-the-engine-exp79)):
gate G holds on the B200, P2 holds at 24 of 24 with the smallest lead named, and the RTX run that preceded it, my
choice of partition, became the engine-determinism measurement. Two things learned for the fifteen others: the
development partition's four-hour limit cuts the slowest encoders' last task (AnySat lost m-SA-crop-type and
finishes it in a follow-up job merged by `scripts/merge_exp79_partial.py`), and the gate for those fifteen is an
RTX gate, since exp74 was run on an RTX (job 881793). P1, P3, P4 and P5 are graded when all sixteen are in.

## Final reading, 24 September 2026

All sixteen exports graded ([comparisons](../results/comparisons.md#is-the-record-bigger-than-its-own-seed-noise-final-reading-all-sixteen-encoders-exp79)),
after an independent recomputation agreed with every verdict (`exp/out/audit_exp79_exp84_2026-09-24.md`). P1 to P5
hold. Gate G fails on five encoders (CopernicusFM, CROMA Base and Large, TerraMind Base and Large), on segmentation
tasks only, by up to 5.4e-4, on the record's own GPU model; they are graded against their own seed 0 as item 1 of
the amendment requires. Two counts on this page were wrong and are corrected here rather than above: P5 has 111
cells, not 112, because Satlas Base does not carry PASTIS Sentinel-1+2; and P4's seven segmentation tasks are five
datasets, PASTIS appearing under three sensors.

## Amendment, 22 September 2026, written while array 1022737 was queued and before any number was read

An adversarial read of this page against `exp/exp79_seed_floors.py`, done by two independent readers in
parallel while the sixteen jobs waited for GPUs, found that the export stage is sound for every encoder (the
any-grid path of exp74, traced on six non-Base grids) and that the **grading** stage had four defects of one
class: it counted over *what ran* where this page says *what the record holds*. Each is corrected in the code
before the first export has finished, and each correction has a test that isolates it.

1. **The export's task set was never compared with the record's.** `cmd_export` files any load failure —
   an out-of-memory on a large encoder's largest task included — under `absent`, and nothing then noticed that
   an encoder recorded at 21 tasks came back with 20. A lost task raises P1's share (19 of 20 is 0.95 where 19
   of 21 is 0.905), shrinks P2's "of 24", moves P3's median and lowers P4's exception count. **Gate G now
   requires the export to carry exactly the tasks the record holds for that encoder** (exp70's 24 for Base,
   exp74's per-encoder set for the rest), names every missing task with the reason the export logged, and
   fails the encoder. P4 counts a task the record carries but the export lost as an exception, since it cannot
   be shown to under-cover.
2. **P2 accepted "all seeds present" as "all 10 seeds" and "20 of whatever ran" as "20 of 24".** It now
   requires every task to carry exactly ten seeds and the task count to equal the record's, and reports both.
3. **P1, P3 and P5 could hold on a partial run.** Three encoders' files under `exp79_seeds/` would have graded
   "all 16 encoders" as true. Every "all" is now against the sixteen by name, and the grade names which are
   missing.
4. **P3's named sets came from this run's recomputed seed 0, not from the record this page cites.** The record's
   third-to-fourth gap is 0.005 — olmoearth_base 0.6800 against terramind_large 0.6749 — which is below the
   seed noise this page itself invokes, so a rerun could have quietly tested a different top three. The sets
   are now read from `headroom_by_encoder.json`; the Spearman correlation is against the record's seed-0 values;
   and **Gate G additionally requires this run's recomputed seed-0 median headroom to agree with the record's to
   1e-3** per encoder, so that a seed 0 that is a different quantity is a gate failure rather than a silent
   change of reference.

One threshold on this page was set without its Monte Carlo noise and is corrected here, with the arithmetic.
**P5's "within 0.01 of the exact coverage"** is 1.75 standard errors at a coverage of 0.93 over R = 2,000 draws
(one SE is √(0.93·0.07/2000) = 0.0057), so over 112 cells it would have failed about four honest cells by
chance and P5 could not have held on an estimator that was correct. The tolerance is now **the larger of 0.01
and three standard errors at that cell's exact coverage** — 0.017 at 0.93 — which puts the expected number of
false failures over 112 cells below 0.2. It remains one-sided, as this page's own failure clause ("sits more
than 0.01 below") always was. This is a loosening of a number and a tightening of the test: it now tests the
estimator rather than the random number generator.

Two wordings are corrected without changing what is computed: Gate G's "to the four decimals the record states"
means |difference| < 1e-4, since the record stores full-precision floats; and an encoder that fails Gate G is
marked inside every P1–P3 block as not comparable to the record, rather than only at the top level.

Also on the record before the numbers: the classification probes use Ai2's single learning rate (0.1) for every
encoder, as exp70 and exp74 did; the per-task rates set_probe_lrs records in each export apply to the
segmentation tasks alone. And the export stage keeps a float32 copy of the train embeddings for all ten seeds
and makes one of the test embeddings per seed, so the largest-D encoders may exceed the 200 GB requested on
m-SA-crop-type; if one does, item 1 above is what makes that a named gate failure instead of a smaller
denominator, and that (encoder, task) is rerun alone with more memory.

**P5's second clause, restated on 23 September before the remaining encoders land.** exp81's audit showed that
a rule passing a cell because its Monte Carlo coverage matches the exact coverage of the same interval cannot
detect a defect in the interval itself: the Monte Carlo of a deterministic interval always converges to that
interval's exact coverage, so the clause tests the sampler and the harness, not the interval. It hid a
finite-population defect in `wilson_interval` for a night on exp81's per-class cells. P5 keeps the clause for
what it does test, but a cell that holds only through it is reported beside the verdict as the interval's own
shortfall, with its (N, K, B) and exact value, and is never counted as the estimator being honest there. At the
time of writing no cell has used it: 35 cells on five encoders, the lowest at 0.933. At this study's sampling
fractions (300 of at least 12,800 windows) the finite-population forms of 22 and 23 September give identical
exact coverage on every recorded cell.
